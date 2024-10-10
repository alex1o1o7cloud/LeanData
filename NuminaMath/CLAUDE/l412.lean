import Mathlib

namespace first_month_sale_is_6400_l412_41275

/-- Represents the sales data for a grocer over six months -/
structure GrocerSales where
  month2 : ℕ
  month3 : ℕ
  month4 : ℕ
  month5 : ℕ
  month6 : ℕ
  average : ℕ

/-- Calculates the sale in the first month given the sales data -/
def firstMonthSale (sales : GrocerSales) : ℕ :=
  6 * sales.average - (sales.month2 + sales.month3 + sales.month4 + sales.month5 + sales.month6)

/-- Theorem stating that the first month's sale is 6400 given the specific sales data -/
theorem first_month_sale_is_6400 (sales : GrocerSales) 
  (h1 : sales.month2 = 7000)
  (h2 : sales.month3 = 6800)
  (h3 : sales.month4 = 7200)
  (h4 : sales.month5 = 6500)
  (h5 : sales.month6 = 5100)
  (h6 : sales.average = 6500) :
  firstMonthSale sales = 6400 := by
  sorry


end first_month_sale_is_6400_l412_41275


namespace inscribed_angles_sum_l412_41260

/-- Given a circle divided into 18 equal arcs, if central angle x spans 3 arcs
    and central angle y spans 6 arcs, then the sum of the corresponding
    inscribed angles x and y is 90°. -/
theorem inscribed_angles_sum (x y : ℝ) : 
  (18 : ℝ) * x = 360 →  -- The circle is divided into 18 equal arcs
  3 * x = y →           -- Central angle y is twice central angle x
  2 * x = 60 →          -- Central angle x spans 3 arcs (3 * 20° = 60°)
  x / 2 + y / 2 = 90    -- Sum of inscribed angles x and y is 90°
  := by sorry

end inscribed_angles_sum_l412_41260


namespace number_of_players_is_five_l412_41235

/-- Represents the number of chips each player receives -/
def chips_per_player (m : ℕ) (n : ℕ) : ℕ := n * m

/-- Represents the number of chips taken by the i-th player -/
def chips_taken (i : ℕ) (m : ℕ) (remaining : ℕ) : ℕ :=
  i * m + remaining / 6

/-- The main theorem stating that the number of players is 5 -/
theorem number_of_players_is_five (m : ℕ) (total_chips : ℕ) :
  ∃ (n : ℕ),
    n = 5 ∧
    (∀ i : ℕ, i ≤ n →
      chips_taken i m (total_chips - (chips_per_player m i)) =
      chips_per_player m n) :=
sorry

end number_of_players_is_five_l412_41235


namespace billys_old_score_l412_41224

/-- Billy's video game score problem -/
theorem billys_old_score (points_per_round : ℕ) (rounds_to_beat : ℕ) (old_score : ℕ) : 
  points_per_round = 2 → rounds_to_beat = 363 → old_score = points_per_round * rounds_to_beat → old_score = 726 := by
  sorry

#check billys_old_score

end billys_old_score_l412_41224


namespace athlete_speed_l412_41248

/-- Given an athlete who runs 200 meters in 40 seconds, prove that their speed is 5 meters per second. -/
theorem athlete_speed (distance : Real) (time : Real) (speed : Real) 
  (h1 : distance = 200) 
  (h2 : time = 40) 
  (h3 : speed = distance / time) : speed = 5 := by
  sorry

end athlete_speed_l412_41248


namespace arthur_total_distance_l412_41295

/-- Represents the distance walked in a single direction --/
structure DirectionalDistance :=
  (blocks : ℕ)

/-- Calculates the total number of blocks walked --/
def total_blocks (east west north south : DirectionalDistance) : ℕ :=
  east.blocks + west.blocks + north.blocks + south.blocks

/-- Converts blocks to miles --/
def blocks_to_miles (blocks : ℕ) : ℚ :=
  (blocks : ℚ) * (1 / 4 : ℚ)

/-- Theorem: Arthur's total walking distance is 5.75 miles --/
theorem arthur_total_distance :
  let east := DirectionalDistance.mk 8
  let north := DirectionalDistance.mk 10
  let south := DirectionalDistance.mk 5
  let west := DirectionalDistance.mk 0
  blocks_to_miles (total_blocks east west north south) = 5.75 := by
  sorry

end arthur_total_distance_l412_41295


namespace circle_radius_in_ellipse_l412_41205

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 + 6 * y^2 = 8

-- Define the condition of two circles being externally tangent
def externally_tangent_circles (r : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4 * r^2

-- Define the condition of a circle being internally tangent to the ellipse
def internally_tangent_to_ellipse (r : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse_equation x y ∧ (x - r)^2 + y^2 = r^2

-- State the theorem
theorem circle_radius_in_ellipse (r : ℝ) :
  externally_tangent_circles r →
  internally_tangent_to_ellipse r →
  r = Real.sqrt 10 / 3 :=
by
  sorry


end circle_radius_in_ellipse_l412_41205


namespace arithmetic_calculation_l412_41286

theorem arithmetic_calculation : 8 / 4 - 3 - 10 + 3 * 7 = 10 := by
  sorry

end arithmetic_calculation_l412_41286


namespace smallest_n_satisfying_conditions_l412_41268

theorem smallest_n_satisfying_conditions : ∃ N : ℕ, 
  (∀ m : ℕ, m < N → ¬(3 ∣ m ∧ 11 ∣ m ∧ m % 12 = 6)) ∧ 
  (3 ∣ N ∧ 11 ∣ N ∧ N % 12 = 6) ∧
  N = 66 := by sorry

end smallest_n_satisfying_conditions_l412_41268


namespace remaining_meat_l412_41280

/-- Given an initial amount of meat and the amounts used for meatballs and spring rolls,
    prove that the remaining amount of meat is 12 kilograms. -/
theorem remaining_meat (initial_meat : ℝ) (meatball_fraction : ℝ) (spring_roll_meat : ℝ)
    (h1 : initial_meat = 20)
    (h2 : meatball_fraction = 1 / 4)
    (h3 : spring_roll_meat = 3) :
    initial_meat - (initial_meat * meatball_fraction) - spring_roll_meat = 12 :=
by sorry

end remaining_meat_l412_41280


namespace number_of_friends_l412_41226

theorem number_of_friends : ℕ :=
  let melanie_cards : ℕ := sorry
  let benny_cards : ℕ := sorry
  let sally_cards : ℕ := sorry
  let jessica_cards : ℕ := sorry
  have total_cards : melanie_cards + benny_cards + sally_cards + jessica_cards = 12 := by sorry
  4

#check number_of_friends

end number_of_friends_l412_41226


namespace favorite_numbers_sum_of_squares_l412_41213

/-- The sum of the squares of Misty's, Glory's, and Dawn's favorite numbers -/
def sumOfSquares (gloryFavorite : ℕ) : ℕ :=
  let mistyFavorite := gloryFavorite / 3
  let dawnFavorite := gloryFavorite * 2
  mistyFavorite ^ 2 + gloryFavorite ^ 2 + dawnFavorite ^ 2

/-- Theorem stating that the sum of squares of the favorite numbers is 1,035,000 -/
theorem favorite_numbers_sum_of_squares :
  sumOfSquares 450 = 1035000 := by
  sorry

#eval sumOfSquares 450

end favorite_numbers_sum_of_squares_l412_41213


namespace pen_cost_calculation_l412_41299

def notebook_cost (pen_cost : ℝ) : ℝ := 3 * pen_cost

theorem pen_cost_calculation (total_cost : ℝ) (num_notebooks : ℕ) 
  (h1 : total_cost = 18)
  (h2 : num_notebooks = 4) :
  ∃ (pen_cost : ℝ), 
    pen_cost = 1.5 ∧ 
    total_cost = num_notebooks * (notebook_cost pen_cost) :=
by
  sorry

end pen_cost_calculation_l412_41299


namespace intersection_complement_subset_condition_l412_41261

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1: When a = -2, A ∩ (ℝ \ B) = {x | -1 ≤ x ≤ 1}
theorem intersection_complement (a : ℝ) (h : a = -2) :
  A a ∩ (Set.univ \ B) = {x : ℝ | -1 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: A ⊆ B if and only if a ≥ 2
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ a ≥ 2 := by sorry

end intersection_complement_subset_condition_l412_41261


namespace rat_value_formula_l412_41279

/-- The number value of a letter in a shifted alphabet with offset N -/
def letterValue (position : ℕ) (N : ℕ) : ℕ := position + N

/-- The sum of letter values for the word "rat" in a shifted alphabet with offset N -/
def ratSum (N : ℕ) : ℕ := letterValue 18 N + letterValue 1 N + letterValue 20 N

/-- The length of the word "rat" -/
def ratLength : ℕ := 3

/-- The number value of the word "rat" in a shifted alphabet with offset N -/
def ratValue (N : ℕ) : ℕ := ratSum N * ratLength

theorem rat_value_formula (N : ℕ) : ratValue N = 117 + 9 * N := by
  sorry

end rat_value_formula_l412_41279


namespace kids_in_restaurant_group_l412_41231

/-- Represents the number of kids in a restaurant group given certain conditions. -/
def number_of_kids (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) : ℕ :=
  total_people - (total_cost / adult_meal_cost)

/-- Theorem stating that given the problem conditions, the number of kids is 9. -/
theorem kids_in_restaurant_group :
  let total_people : ℕ := 13
  let adult_meal_cost : ℕ := 7
  let total_cost : ℕ := 28
  number_of_kids total_people adult_meal_cost total_cost = 9 := by
sorry

#eval number_of_kids 13 7 28

end kids_in_restaurant_group_l412_41231


namespace franks_age_l412_41207

theorem franks_age (frank : ℕ) (gabriel : ℕ) : 
  gabriel = frank - 3 → 
  frank + gabriel = 17 → 
  frank = 10 := by sorry

end franks_age_l412_41207


namespace average_yield_is_15_l412_41233

def rice_field_yields : List ℝ := [12, 13, 15, 17, 18]

theorem average_yield_is_15 :
  (rice_field_yields.sum / rice_field_yields.length : ℝ) = 15 := by
  sorry

end average_yield_is_15_l412_41233


namespace product_scaling_l412_41208

theorem product_scaling (a b c : ℕ) (ha : a = 268) (hb : b = 74) (hc : c = 19832) 
  (h : a * b = c) : (2.68 : ℝ) * 0.74 = 1.9832 := by
  sorry

end product_scaling_l412_41208


namespace point_on_line_implies_tan_2theta_l412_41212

theorem point_on_line_implies_tan_2theta (θ : ℝ) : 
  2 * Real.sin θ + Real.cos θ = 0 → Real.tan (2 * θ) = -4/3 := by
  sorry

end point_on_line_implies_tan_2theta_l412_41212


namespace fraction_sum_squared_l412_41245

theorem fraction_sum_squared (a b c : ℝ) (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 := by
  sorry

end fraction_sum_squared_l412_41245


namespace inequality_proof_l412_41290

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a^3 + b^3 + a*b*c) + 1 / (b^3 + c^3 + a*b*c) + 1 / (c^3 + a^3 + a*b*c) ≤ 1 / (a*b*c) :=
by sorry

end inequality_proof_l412_41290


namespace committee_probability_l412_41240

def total_members : ℕ := 30
def boys : ℕ := 12
def girls : ℕ := 18
def committee_size : ℕ := 5

theorem committee_probability :
  (Nat.choose total_members committee_size - 
   (Nat.choose boys committee_size + Nat.choose girls committee_size)) / 
   Nat.choose total_members committee_size = 133146 / 142506 := by
sorry

end committee_probability_l412_41240


namespace obrien_current_hats_l412_41259

/-- The number of hats Policeman O'Brien has after theft -/
def obrien_hats_after_theft (simpson_hats : ℕ) (stolen_hats : ℕ) : ℕ :=
  2 * simpson_hats + 5 - stolen_hats

/-- Theorem stating the number of hats Policeman O'Brien has after theft -/
theorem obrien_current_hats (simpson_hats stolen_hats : ℕ) 
  (h1 : simpson_hats = 15) :
  obrien_hats_after_theft simpson_hats stolen_hats = 35 - stolen_hats := by
  sorry

#check obrien_current_hats

end obrien_current_hats_l412_41259


namespace orange_ribbons_count_l412_41283

/-- The number of ribbons in a container with yellow, purple, orange, and black ribbons. -/
def total_ribbons : ℚ :=
  let black_ribbons : ℚ := 45
  let black_fraction : ℚ := 1 - (1/4 + 3/8 + 1/8)
  black_ribbons / black_fraction

/-- The number of orange ribbons in the container. -/
def orange_ribbons : ℚ := (1/8) * total_ribbons

/-- Theorem stating that the number of orange ribbons is 22.5. -/
theorem orange_ribbons_count : orange_ribbons = 22.5 := by
  sorry

end orange_ribbons_count_l412_41283


namespace subset_iff_m_range_l412_41220

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 1}

theorem subset_iff_m_range (m : ℝ) : B m ⊆ A ↔ m ≥ -1/2 := by
  sorry

end subset_iff_m_range_l412_41220


namespace inequality_proof_l412_41281

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (b + c) + 1 / (a + c) + 1 / (a + b) ≥ 9 / (2 * (a + b + c)) := by
  sorry

end inequality_proof_l412_41281


namespace eventual_stability_l412_41278

/-- Represents a line of 2018 natural numbers -/
def Line := Fin 2018 → ℕ

/-- Applies the frequency counting operation to a line -/
def frequency_count (l : Line) : Line := sorry

/-- Predicate to check if two lines are identical -/
def identical (l1 l2 : Line) : Prop := ∀ i, l1 i = l2 i

/-- Theorem stating that repeated frequency counting eventually leads to identical lines -/
theorem eventual_stability (initial : Line) : 
  ∃ n : ℕ, ∀ m ≥ n, identical (frequency_count^[m] initial) (frequency_count^[m+1] initial) := by
  sorry

end eventual_stability_l412_41278


namespace quarter_circles_sum_exceeds_circumference_l412_41243

/-- Theorem: As the number of divisions approaches infinity, the sum of the lengths of quarter-circles
    constructed on equal parts of a circle's circumference exceeds the original circumference. -/
theorem quarter_circles_sum_exceeds_circumference (r : ℝ) (hr : r > 0) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → π * π * r > 2 * π * r := by
  sorry

#check quarter_circles_sum_exceeds_circumference

end quarter_circles_sum_exceeds_circumference_l412_41243


namespace f_max_range_l412_41201

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then a * Real.log x - x^2 - 2
  else if x < 0 then x + 1/x + a
  else 0  -- This value doesn't matter as x ≠ 0 in the problem

theorem f_max_range (a : ℝ) :
  (∀ x : ℝ, f a x ≤ f a (-1)) →
  0 ≤ a ∧ a ≤ 2 * Real.exp 3 :=
sorry

end f_max_range_l412_41201


namespace max_pies_36_l412_41254

/-- Calculates the maximum number of pies that can be made given a certain number of apples,
    where every two pies require 12 apples and every third pie needs an extra apple. -/
def max_pies (total_apples : ℕ) : ℕ :=
  let basic_pies := 2 * (total_apples / 12)
  let extra_apples := basic_pies / 3
  let adjusted_apples := total_apples - extra_apples
  let full_sets := adjusted_apples / 12
  let remaining_apples := adjusted_apples % 12
  2 * full_sets + if remaining_apples ≥ 6 then 1 else 0

/-- Theorem stating that given 36 apples, the maximum number of pies that can be made is 9. -/
theorem max_pies_36 : max_pies 36 = 9 := by
  sorry

end max_pies_36_l412_41254


namespace sufficient_not_necessary_condition_l412_41265

theorem sufficient_not_necessary_condition (x : ℝ) :
  (∀ x, x > 0 → x * (x + 1) > 0) ∧
  (∃ x, x * (x + 1) > 0 ∧ ¬(x > 0)) :=
by sorry

end sufficient_not_necessary_condition_l412_41265


namespace problem_solution_l412_41229

theorem problem_solution (x y z : ℝ) 
  (h1 : x + y = 5) 
  (h2 : z^2 = x*y + y - 9) : 
  x + 2*y + 3*z = 8 := by
sorry

end problem_solution_l412_41229


namespace peach_count_l412_41236

/-- Given a basket of peaches with red and green peaches, calculate the total number of peaches -/
def total_peaches (red_peaches green_peaches : ℕ) : ℕ :=
  red_peaches + green_peaches

/-- Theorem: Given 1 basket with 4 red peaches and 6 green peaches, the total number of peaches is 10 -/
theorem peach_count : total_peaches 4 6 = 10 := by
  sorry

end peach_count_l412_41236


namespace prob_both_odd_is_one_sixth_l412_41252

/-- The set of numbers to draw from -/
def S : Finset ℕ := {1, 2, 3, 4}

/-- A function to determine if a number is odd -/
def isOdd (n : ℕ) : Bool := n % 2 = 1

/-- The set of all possible pairs of numbers drawn without replacement -/
def allPairs : Finset (ℕ × ℕ) := S.product S |>.filter (fun (a, b) => a ≠ b)

/-- The set of pairs where both numbers are odd -/
def oddPairs : Finset (ℕ × ℕ) := allPairs.filter (fun (a, b) => isOdd a ∧ isOdd b)

/-- The probability of drawing two odd numbers without replacement -/
def probBothOdd : ℚ := (oddPairs.card : ℚ) / (allPairs.card : ℚ)

theorem prob_both_odd_is_one_sixth : probBothOdd = 1 / 6 := by
  sorry

end prob_both_odd_is_one_sixth_l412_41252


namespace inverse_proportion_quadrants_l412_41219

/-- An inverse proportion function passing through (3, -5) is in the second and fourth quadrants -/
theorem inverse_proportion_quadrants :
  ∀ k : ℝ,
  (∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = k / x) ∧ f 3 = -5) →
  (∀ x y : ℝ, x ≠ 0 ∧ y = k / x → (x > 0 ∧ y < 0) ∨ (x < 0 ∧ y > 0)) :=
by sorry


end inverse_proportion_quadrants_l412_41219


namespace median_is_2040201_l412_41203

/-- The list of numbers containing integers from 1 to 2020, their squares, and their cubes -/
def numberList : List ℕ := 
  (List.range 2020).map (λ x => x + 1) ++
  (List.range 2020).map (λ x => (x + 1)^2) ++
  (List.range 2020).map (λ x => (x + 1)^3)

/-- The length of the number list -/
def listLength : ℕ := 6060

/-- The position of the lower median element -/
def lowerMedianPos : ℕ := listLength / 2

/-- The position of the upper median element -/
def upperMedianPos : ℕ := lowerMedianPos + 1

/-- The lower median element -/
def lowerMedian : ℕ := 2020^2

/-- The upper median element -/
def upperMedian : ℕ := 1^3

/-- The median of the number list -/
def median : ℕ := (lowerMedian + upperMedian) / 2

/-- Theorem stating that the median of the number list is 2040201 -/
theorem median_is_2040201 : median = 2040201 := by
  sorry

end median_is_2040201_l412_41203


namespace bike_savings_time_l412_41296

/-- The cost of the mountain bike in dollars -/
def bike_cost : ℕ := 600

/-- The total birthday money Chandler received in dollars -/
def birthday_money : ℕ := 60 + 40 + 20

/-- The amount Chandler earns per week from his paper route in dollars -/
def weekly_earnings : ℕ := 20

/-- The number of weeks it takes to save enough money for the bike -/
def weeks_to_save : ℕ := 24

/-- Theorem stating that it takes 24 weeks to save enough money for the bike -/
theorem bike_savings_time :
  birthday_money + weekly_earnings * weeks_to_save = bike_cost := by
  sorry

end bike_savings_time_l412_41296


namespace tinas_earnings_l412_41218

/-- Calculates the total earnings for a worker given their hourly rate, hours worked per day, 
    number of days worked, and regular hours per day before overtime. -/
def calculate_earnings (hourly_rate : ℚ) (hours_per_day : ℕ) (days_worked : ℕ) (regular_hours : ℕ) : ℚ :=
  let regular_pay := hourly_rate * regular_hours * days_worked
  let overtime_hours := if hours_per_day > regular_hours then hours_per_day - regular_hours else 0
  let overtime_rate := hourly_rate * (1 + 1/2)
  let overtime_pay := overtime_rate * overtime_hours * days_worked
  regular_pay + overtime_pay

/-- Theorem stating that Tina's earnings for 5 days of work at 10 hours per day 
    with an $18.00 hourly rate is $990.00. -/
theorem tinas_earnings : 
  calculate_earnings 18 10 5 8 = 990 := by
  sorry

end tinas_earnings_l412_41218


namespace problem_solution_l412_41234

theorem problem_solution : 
  ∀ a b : ℝ, 
  (∃ k : ℝ, k^2 = a + b - 5 ∧ (k = 3 ∨ k = -3)) →
  (a - b + 4)^(1/3) = 2 →
  a = 9 ∧ b = 5 ∧ Real.sqrt (4 * (a - b)) = 4 :=
by
  sorry

end problem_solution_l412_41234


namespace alien_eggs_count_l412_41272

-- Define a function to convert a number from base 7 to base 10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

-- Theorem statement
theorem alien_eggs_count :
  base7ToBase10 [1, 2, 3] = 162 := by
  sorry

end alien_eggs_count_l412_41272


namespace digit_sum_inequalities_l412_41277

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

theorem digit_sum_inequalities :
  (∀ k : ℕ, sumOfDigits k ≤ 8 * sumOfDigits (8 * k)) ∧
  (∀ N : ℕ, sumOfDigits N ≤ 5 * sumOfDigits (5^5 * N)) := by sorry

end digit_sum_inequalities_l412_41277


namespace at_least_one_shot_hit_l412_41239

theorem at_least_one_shot_hit (p q : Prop) : 
  (p ∨ q) ↔ ¬(¬p ∧ ¬q) := by sorry

end at_least_one_shot_hit_l412_41239


namespace lcm_hcf_problem_l412_41202

/-- Given two positive integers with LCM 2310, HCF 30, and one of them being 210, prove the other is 330 -/
theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  B = 210 →
  A = 330 := by sorry

end lcm_hcf_problem_l412_41202


namespace sqrt_seven_decimal_part_l412_41274

theorem sqrt_seven_decimal_part (a : ℝ) : 
  (2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3) → 
  (a = Real.sqrt 7 - 2) → 
  ((Real.sqrt 7 + 2) * a = 3) := by
sorry

end sqrt_seven_decimal_part_l412_41274


namespace combine_like_terms_l412_41271

theorem combine_like_terms (a : ℝ) : 3 * a + 2 * a = 5 * a := by
  sorry

end combine_like_terms_l412_41271


namespace basketball_points_per_basket_l412_41273

theorem basketball_points_per_basket 
  (matthew_points : ℕ) 
  (shawn_points : ℕ) 
  (total_baskets : ℕ) 
  (h1 : matthew_points = 9) 
  (h2 : shawn_points = 6) 
  (h3 : total_baskets = 5) : 
  (matthew_points + shawn_points) / total_baskets = 3 := by
sorry

end basketball_points_per_basket_l412_41273


namespace complex_fraction_simplification_l412_41215

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (1 - 3*i) / (2 + 5*i) = (-13 : ℝ) / 29 - (11 : ℝ) / 29 * i := by
  sorry

end complex_fraction_simplification_l412_41215


namespace sum_of_gcd_and_lcm_equals_90_l412_41282

def numbers : List Nat := [18, 36, 72]

theorem sum_of_gcd_and_lcm_equals_90 : 
  (numbers.foldl Nat.gcd numbers.head!) + (numbers.foldl Nat.lcm numbers.head!) = 90 := by
  sorry

end sum_of_gcd_and_lcm_equals_90_l412_41282


namespace players_quit_l412_41223

def video_game_problem (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  initial_players - (total_lives / lives_per_player)

theorem players_quit (initial_players : ℕ) (lives_per_player : ℕ) (total_lives : ℕ)
  (h1 : initial_players = 8)
  (h2 : lives_per_player = 5)
  (h3 : total_lives = 15) :
  video_game_problem initial_players lives_per_player total_lives = 5 := by
  sorry

#eval video_game_problem 8 5 15

end players_quit_l412_41223


namespace perpendicular_line_equation_l412_41292

/-- A line passing through (1,3) perpendicular to 2x-6y-8=0 has equation y+3x-6=0 -/
theorem perpendicular_line_equation :
  let l₁ : Set (ℝ × ℝ) := {p | 2 * p.1 - 6 * p.2 - 8 = 0}
  let l₂ : Set (ℝ × ℝ) := {p | p.2 + 3 * p.1 - 6 = 0}
  let point : ℝ × ℝ := (1, 3)
  (point ∈ l₂) ∧
  (∀ (p₁ p₂ : ℝ × ℝ), p₁ ∈ l₁ → p₂ ∈ l₁ → p₁ ≠ p₂ →
    ∀ (q₁ q₂ : ℝ × ℝ), q₁ ∈ l₂ → q₂ ∈ l₂ → q₁ ≠ q₂ →
      ((p₂.1 - p₁.1) * (q₂.1 - q₁.1) + (p₂.2 - p₁.2) * (q₂.2 - q₁.2) = 0)) :=
by
  sorry

end perpendicular_line_equation_l412_41292


namespace clapping_theorem_l412_41244

/-- Represents the clapping pattern of a person -/
structure ClappingPattern where
  interval : ℕ
  start_time : ℕ

/-- Checks if a clapping pattern results in a clap at the given time -/
def claps_at (pattern : ClappingPattern) (time : ℕ) : Prop :=
  ∃ k : ℕ, time = pattern.start_time + k * pattern.interval

theorem clapping_theorem (jirka_start petr_start : ℕ) :
  jirka_start ≤ 15 ∧ petr_start ≤ 15 ∧
  claps_at { interval := 7, start_time := jirka_start } 90 ∧
  claps_at { interval := 13, start_time := petr_start } 90 →
  (jirka_start = 6 ∨ jirka_start = 13) ∧ petr_start = 12 := by
  sorry

#check clapping_theorem

end clapping_theorem_l412_41244


namespace recipe_total_cups_l412_41256

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of flour -/
def totalCups (ratio : RecipeRatio) (flourAmount : ℕ) : ℕ :=
  let unitAmount := flourAmount / ratio.flour
  unitAmount * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem: Given the specified ratio and flour amount, the total cups of ingredients is 30 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := { butter := 2, flour := 5, sugar := 3 }
  let flourAmount : ℕ := 15
  totalCups ratio flourAmount = 30 := by
  sorry


end recipe_total_cups_l412_41256


namespace x_plus_y_value_l412_41228

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 3000)
  (eq2 : x + 3000 * Real.sin y = 2999)
  (y_range : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2999 := by
sorry

end x_plus_y_value_l412_41228


namespace driver_net_pay_rate_l412_41200

/-- Calculate the net rate of pay for a driver --/
theorem driver_net_pay_rate (travel_time : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (payment_rate : ℝ) (gasoline_cost : ℝ) :
  travel_time = 3 ∧ 
  speed = 45 ∧ 
  fuel_efficiency = 36 ∧ 
  payment_rate = 0.60 ∧ 
  gasoline_cost = 2.50 → 
  (payment_rate * speed * travel_time - 
   (speed * travel_time / fuel_efficiency) * gasoline_cost) / travel_time = 23.875 := by
  sorry

end driver_net_pay_rate_l412_41200


namespace quadratic_root_square_l412_41227

theorem quadratic_root_square (p : ℝ) : 
  (∃ a b : ℝ, a ≠ b ∧ 
   a^2 - p*a + p = 0 ∧ 
   b^2 - p*b + p = 0 ∧ 
   (a = b^2 ∨ b = a^2)) ↔ 
  (p = 2 + Real.sqrt 5 ∨ p = 2 - Real.sqrt 5) :=
sorry

end quadratic_root_square_l412_41227


namespace binomial_distribution_properties_l412_41287

/-- Represents the probability of success in a single trial -/
def p : ℝ := 0.6

/-- Represents the number of trials -/
def n : ℕ := 5

/-- Expected value of a binomial distribution -/
def expected_value : ℝ := n * p

/-- Variance of a binomial distribution -/
def variance : ℝ := n * p * (1 - p)

theorem binomial_distribution_properties :
  expected_value = 3 ∧ variance = 1.2 := by
  sorry

end binomial_distribution_properties_l412_41287


namespace polynomial_factorization_l412_41288

theorem polynomial_factorization (x : ℝ) :
  5 * (x + 3) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 =
  (5 * x^2 + 81 * x + 315) * (x + 3) * (x + 213) := by
  sorry

end polynomial_factorization_l412_41288


namespace part_one_part_two_l412_41269

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 2 * x^2 - 3 * x * y + y^2 + 2 * x + 2 * y
def B (x y : ℝ) : ℝ := 4 * x^2 - 6 * x * y + 2 * y^2 - 3 * x - y

-- Part 1
theorem part_one : B 2 (-1/5) - 2 * A 2 (-1/5) = -13 := by sorry

-- Part 2
theorem part_two (a : ℝ) : 
  (∃ x y : ℝ, (|x - 2*a| + (y - 3)^2 = 0) ∧ (B x y - 2 * A x y = a)) → a = -1 := by sorry

end part_one_part_two_l412_41269


namespace unique_p_q_solution_l412_41216

theorem unique_p_q_solution :
  ∀ p q : ℝ,
    p ≠ q →
    p > 1 →
    q > 1 →
    1 / p + 1 / q = 1 →
    p * q = 9 →
    ((p = (9 + 3 * Real.sqrt 5) / 2 ∧ q = (9 - 3 * Real.sqrt 5) / 2) ∨
     (p = (9 - 3 * Real.sqrt 5) / 2 ∧ q = (9 + 3 * Real.sqrt 5) / 2)) :=
by sorry

end unique_p_q_solution_l412_41216


namespace halloween_candy_distribution_l412_41262

theorem halloween_candy_distribution (initial_candy : ℕ) (eaten_candy : ℕ) (num_piles : ℕ) : 
  initial_candy = 32 → 
  eaten_candy = 12 → 
  num_piles = 4 → 
  (initial_candy - eaten_candy) / num_piles = 5 := by
  sorry

end halloween_candy_distribution_l412_41262


namespace corveus_sleep_hours_l412_41217

/-- Represents the number of days in a week -/
def days_in_week : ℕ := 7

/-- Represents the doctor's recommended hours of sleep per day -/
def recommended_sleep_per_day : ℕ := 6

/-- Represents the sleep deficit in hours per week -/
def sleep_deficit_per_week : ℕ := 14

/-- Calculates Corveus's actual sleep hours per day -/
def actual_sleep_per_day : ℚ :=
  (recommended_sleep_per_day * days_in_week - sleep_deficit_per_week) / days_in_week

/-- Proves that Corveus sleeps 4 hours per day given the conditions -/
theorem corveus_sleep_hours :
  actual_sleep_per_day = 4 := by sorry

end corveus_sleep_hours_l412_41217


namespace modular_congruence_l412_41206

theorem modular_congruence (a b n : ℤ) : 
  a % 48 = 25 →
  b % 48 = 80 →
  150 ≤ n →
  n ≤ 191 →
  (a - b) % 48 = n % 48 ↔ n = 185 := by
sorry

end modular_congruence_l412_41206


namespace china_forex_reserves_scientific_notation_l412_41276

-- Define the original amount in billions of US dollars
def original_amount : ℚ := 10663

-- Define the number of significant figures to retain
def significant_figures : ℕ := 3

-- Define the function to convert to scientific notation with given significant figures
def to_scientific_notation (x : ℚ) (sig_figs : ℕ) : ℚ × ℤ := sorry

-- Theorem statement
theorem china_forex_reserves_scientific_notation :
  let (mantissa, exponent) := to_scientific_notation (original_amount * 1000000000) significant_figures
  mantissa = 1.07 ∧ exponent = 12 := by sorry

end china_forex_reserves_scientific_notation_l412_41276


namespace unique_solution_prime_power_equation_l412_41222

theorem unique_solution_prime_power_equation :
  ∀ (p q : ℕ) (n m : ℕ),
    Prime p → Prime q → n ≥ 2 → m ≥ 2 →
    (p^n = q^m + 1 ∨ p^n = q^m - 1) →
    (p = 2 ∧ n = 3 ∧ q = 3 ∧ m = 2) :=
by sorry

end unique_solution_prime_power_equation_l412_41222


namespace black_ball_probability_l412_41242

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  orange : ℕ
  black : ℕ
  white : ℕ

/-- Calculates the probability of picking a ball of a specific color -/
def probability (counts : BallCounts) (color : ℕ) : ℚ :=
  color / (counts.orange + counts.black + counts.white)

/-- The main theorem to be proved -/
theorem black_ball_probability (counts : BallCounts) 
  (h1 : counts.orange = 8)
  (h2 : counts.black = 7)
  (h3 : counts.white = 6) :
  probability counts counts.black = 1 / 3 := by
  sorry

end black_ball_probability_l412_41242


namespace lateral_edge_length_l412_41284

/-- A rectangular prism with 8 vertices and a given sum of lateral edge lengths -/
structure RectangularPrism :=
  (vertices : Nat)
  (lateral_edges_sum : ℝ)
  (is_valid : vertices = 8)

/-- The number of lateral edges in a rectangular prism -/
def lateral_edges_count : Nat := 4

/-- Theorem: In a valid rectangular prism, if the sum of lateral edges is 56,
    then each lateral edge has length 14 -/
theorem lateral_edge_length (prism : RectangularPrism)
    (h_sum : prism.lateral_edges_sum = 56) :
    prism.lateral_edges_sum / lateral_edges_count = 14 := by
  sorry

#check lateral_edge_length

end lateral_edge_length_l412_41284


namespace vector_perpendicular_condition_l412_41291

-- Define the vectors m and n
def m : Fin 2 → ℝ := ![1, 3]
def n (t : ℝ) : Fin 2 → ℝ := ![2, t]

-- Define the condition for perpendicularity
def perpendicular (t : ℝ) : Prop :=
  (m 0 + n t 0) * (m 0 - n t 0) + (m 1 + n t 1) * (m 1 - n t 1) = 0

-- State the theorem
theorem vector_perpendicular_condition (t : ℝ) :
  perpendicular t → t = Real.sqrt 6 ∨ t = -Real.sqrt 6 := by
  sorry

end vector_perpendicular_condition_l412_41291


namespace elvis_editing_time_l412_41210

theorem elvis_editing_time (num_songs : ℕ) (studio_hours : ℕ) (record_time : ℕ) (write_time : ℕ)
  (h1 : num_songs = 10)
  (h2 : studio_hours = 5)
  (h3 : record_time = 12)
  (h4 : write_time = 15) :
  (studio_hours * 60) - (num_songs * write_time + num_songs * record_time) = 30 := by
  sorry

end elvis_editing_time_l412_41210


namespace local_value_of_four_l412_41221

/-- The local value of a digit in a number. -/
def local_value (digit : ℕ) (place : ℕ) : ℕ := digit * (10 ^ place)

/-- The sum of local values of all digits in 2345. -/
def total_sum : ℕ := 2345

/-- The local values of digits 2, 3, and 5 in 2345. -/
def known_values : ℕ := local_value 2 3 + local_value 3 2 + local_value 5 0

/-- The local value of the remaining digit (4) in 2345. -/
def remaining_value : ℕ := total_sum - known_values

theorem local_value_of_four :
  remaining_value = local_value 4 1 :=
sorry

end local_value_of_four_l412_41221


namespace fisherman_tuna_count_l412_41246

/-- The number of Red snappers the fisherman gets every day -/
def red_snappers : ℕ := 8

/-- The cost of a Red snapper in dollars -/
def red_snapper_cost : ℕ := 3

/-- The cost of a Tuna in dollars -/
def tuna_cost : ℕ := 2

/-- The total earnings of the fisherman in dollars per day -/
def total_earnings : ℕ := 52

/-- The number of Tunas the fisherman gets every day -/
def tuna_count : ℕ := (total_earnings - red_snappers * red_snapper_cost) / tuna_cost

theorem fisherman_tuna_count : tuna_count = 14 := by
  sorry

end fisherman_tuna_count_l412_41246


namespace sqrt_six_over_sqrt_two_equals_sqrt_three_l412_41211

theorem sqrt_six_over_sqrt_two_equals_sqrt_three :
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end sqrt_six_over_sqrt_two_equals_sqrt_three_l412_41211


namespace max_value_expression_l412_41255

theorem max_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, 3 * (a - x) * (2 * x + Real.sqrt (x^2 + 4 * b^2)) ≤ 3 * a^2 + 12 * b^2) ∧
  (∃ x : ℝ, 3 * (a - x) * (2 * x + Real.sqrt (x^2 + 4 * b^2)) = 3 * a^2 + 12 * b^2) :=
by sorry

end max_value_expression_l412_41255


namespace tetrahedron_has_four_faces_l412_41230

/-- A tetrahedron is a type of pyramid with a triangular base -/
structure Tetrahedron where
  is_pyramid : Bool
  has_triangular_base : Bool

/-- The number of faces in a tetrahedron -/
def num_faces (t : Tetrahedron) : Nat :=
  4

theorem tetrahedron_has_four_faces (t : Tetrahedron) :
  t.is_pyramid = true → t.has_triangular_base = true → num_faces t = 4 := by
  sorry

end tetrahedron_has_four_faces_l412_41230


namespace vector_same_direction_l412_41266

open Real

/-- Given two vectors a and b in ℝ², prove that if they have the same direction,
    a = (1, -√3), and |b| = 1, then b = (1/2, -√3/2) -/
theorem vector_same_direction (a b : ℝ × ℝ) :
  (∃ k : ℝ, b = k • a) →  -- same direction
  a = (1, -Real.sqrt 3) →
  Real.sqrt ((b.1)^2 + (b.2)^2) = 1 →
  b = (1/2, -(Real.sqrt 3)/2) := by
sorry

end vector_same_direction_l412_41266


namespace bales_stacked_l412_41257

theorem bales_stacked (initial_bales current_bales : ℕ) 
  (h1 : initial_bales = 54)
  (h2 : current_bales = 82) :
  current_bales - initial_bales = 28 := by
  sorry

end bales_stacked_l412_41257


namespace family_meeting_impossible_l412_41258

theorem family_meeting_impossible (n : ℕ) (h : n = 9) :
  ¬ ∃ (handshakes : ℕ), 2 * handshakes = n * 3 :=
by
  sorry

end family_meeting_impossible_l412_41258


namespace system_of_equations_sum_l412_41225

theorem system_of_equations_sum (a b c x y z : ℝ) 
  (eq1 : 14 * x + b * y + c * z = 0)
  (eq2 : a * x + 24 * y + c * z = 0)
  (eq3 : a * x + b * y + 43 * z = 0)
  (ha : a ≠ 14)
  (hb : b ≠ 24)
  (hc : c ≠ 43)
  (hx : x ≠ 0) :
  a / (a - 14) + b / (b - 24) + c / (c - 43) = 1 := by
sorry

end system_of_equations_sum_l412_41225


namespace soccer_league_games_l412_41297

/-- The number of teams in the soccer league -/
def num_teams : ℕ := 10

/-- The number of games each team plays with every other team -/
def games_per_pair : ℕ := 2

/-- The total number of games played in the season -/
def total_games : ℕ := num_teams * (num_teams - 1) * games_per_pair / 2

theorem soccer_league_games :
  total_games = 90 :=
sorry

end soccer_league_games_l412_41297


namespace complement_union_theorem_l412_41241

def U : Set ℕ := {0, 1, 3, 5, 6, 8}
def A : Set ℕ := {1, 5, 8}
def B : Set ℕ := {2}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 3, 6} := by sorry

end complement_union_theorem_l412_41241


namespace gcd_sum_product_is_one_l412_41267

/-- The sum of 1234 and 4321 -/
def sum_numbers : ℕ := 1234 + 4321

/-- The product of 1, 2, 3, and 4 -/
def product_digits : ℕ := 1 * 2 * 3 * 4

/-- Theorem stating that the greatest common divisor of the sum of 1234 and 4321,
    and the product of 1, 2, 3, and 4 is 1 -/
theorem gcd_sum_product_is_one : Nat.gcd sum_numbers product_digits = 1 := by
  sorry

end gcd_sum_product_is_one_l412_41267


namespace hyperbola_equation_l412_41298

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1, eccentricity e = 5/4, 
    and right focus F₂(5,0), prove that the equation of C is x²/16 - y²/9 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1) →  -- Equation of hyperbola C
  (a/b)^2 + 1 = (5/4)^2 →               -- Eccentricity e = 5/4
  5^2 = a^2 + b^2 →                     -- Right focus F₂(5,0)
  a^2 = 16 ∧ b^2 = 9 :=
by sorry

end hyperbola_equation_l412_41298


namespace tan_beta_minus_2alpha_l412_41289

theorem tan_beta_minus_2alpha (α β : Real) 
  (h1 : Real.tan α = 1 / 2) 
  (h2 : Real.tan (α - β) = -1 / 3) : 
  Real.tan (β - 2 * α) = -1 / 7 := by
  sorry

end tan_beta_minus_2alpha_l412_41289


namespace expression_evaluation_l412_41264

theorem expression_evaluation :
  let x : ℚ := -1
  let expr := (x - 3) / (2 * x - 4) / ((5 / (x - 2)) - x - 2)
  expr = -1/4 := by
sorry

end expression_evaluation_l412_41264


namespace reciprocal_of_proper_fraction_greater_l412_41247

theorem reciprocal_of_proper_fraction_greater {a b : ℚ} (h1 : 0 < a) (h2 : a < b) :
  b / a > a / b :=
sorry

end reciprocal_of_proper_fraction_greater_l412_41247


namespace existence_of_n_l412_41238

theorem existence_of_n (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_cd : c * d = 1) : 
  ∃ n : ℤ, a * b ≤ (n : ℝ)^2 ∧ (n : ℝ)^2 ≤ (a + c) * (b + d) :=
sorry

end existence_of_n_l412_41238


namespace equation_solution_l412_41204

theorem equation_solution (x : Real) :
  8.414 * Real.cos x + Real.sqrt (3/2 - Real.cos x ^ 2) - Real.cos x * Real.sqrt (3/2 - Real.cos x ^ 2) = 1 ↔
  (∃ k : ℤ, x = 2 * Real.pi * ↑k) ∨
  (∃ k : ℤ, x = Real.pi / 4 + 2 * Real.pi * ↑k) ∨
  (∃ k : ℤ, x = 3 * Real.pi / 4 + 2 * Real.pi * ↑k) := by
  sorry

end equation_solution_l412_41204


namespace intersection_seq_100th_term_l412_41250

def geometric_seq (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

def arithmetic_seq (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def intersection_seq (n : ℕ) : ℝ := 2^(4 * n - 3)

theorem intersection_seq_100th_term :
  intersection_seq 100 = 2^397 :=
by sorry

end intersection_seq_100th_term_l412_41250


namespace odd_function_property_l412_41253

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_symmetry : ∀ x, f x = f (2 - x))
  (h_value : f (-1) = 1) :
  f 2021 = -1 := by
  sorry

end odd_function_property_l412_41253


namespace quadratic_non_real_roots_l412_41251

/-- A quadratic equation of the form x^2 - 8x + c has non-real roots if and only if c > 16 -/
theorem quadratic_non_real_roots (c : ℝ) : 
  (∀ x : ℂ, x^2 - 8*x + c = 0 → x.im ≠ 0) ↔ c > 16 := by
  sorry

end quadratic_non_real_roots_l412_41251


namespace largest_k_for_real_root_l412_41285

/-- The quadratic function f(x) parameterized by k -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x + (k-1)^2

/-- The discriminant of f(x) as a function of k -/
def discriminant (k : ℝ) : ℝ := (-k)^2 - 4*(k-1)^2

/-- Theorem: The largest possible real value of k such that f has at least one real root is 2 -/
theorem largest_k_for_real_root :
  ∀ k : ℝ, (∃ x : ℝ, f k x = 0) → k ≤ 2 ∧ 
  ∃ x : ℝ, f 2 x = 0 :=
by sorry

end largest_k_for_real_root_l412_41285


namespace jane_pens_after_month_l412_41209

def alex_pens (week : ℕ) : ℕ := 4 * 2^week

def jane_pens : ℕ := alex_pens 3 - 16

theorem jane_pens_after_month : jane_pens = 16 := by
  sorry

end jane_pens_after_month_l412_41209


namespace largest_number_with_6_and_3_l412_41270

def largest_two_digit_number (d1 d2 : Nat) : Nat :=
  max (10 * d1 + d2) (10 * d2 + d1)

theorem largest_number_with_6_and_3 :
  largest_two_digit_number 6 3 = 63 := by
sorry

end largest_number_with_6_and_3_l412_41270


namespace distance_to_market_is_40_l412_41263

/-- The distance between Andy's house and the market -/
def distance_to_market (distance_to_school : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance - 2 * distance_to_school

/-- Theorem: Given the conditions, the distance to the market is 40 meters -/
theorem distance_to_market_is_40 :
  let distance_to_school : ℕ := 50
  let total_distance : ℕ := 140
  distance_to_market distance_to_school total_distance = 40 := by
  sorry

end distance_to_market_is_40_l412_41263


namespace vector_collinearity_l412_41214

/-- Given vectors a, b, and c in ℝ², prove that if (a + 2b) is collinear with (3a - c),
    then the y-component of b equals -79/14. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (h : a = (2, -3) ∧ b.1 = 4 ∧ c = (-1, 1)) :
  (∃ (k : ℝ), k • (a + 2 • b) = 3 • a - c) → b.2 = -79/14 := by
sorry

end vector_collinearity_l412_41214


namespace ratio_equality_l412_41232

theorem ratio_equality (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) :
  (a / 3) / (b / 2) = 1 := by
sorry

end ratio_equality_l412_41232


namespace subtraction_to_addition_division_to_multiplication_problem_1_problem_2_l412_41249

-- Problem 1
theorem subtraction_to_addition (a b : ℤ) : a - b = a + (-b) := by sorry

-- Problem 2
theorem division_to_multiplication (a : ℚ) (b : ℚ) (h : b ≠ 0) :
  a / b = a * (1 / b) := by sorry

-- Specific instances
theorem problem_1 : -8 - 5 = -8 + (-5) := by sorry

theorem problem_2 : (1 : ℚ) / 2 / (-2) = (1 : ℚ) / 2 * (-1 / 2) := by sorry

end subtraction_to_addition_division_to_multiplication_problem_1_problem_2_l412_41249


namespace max_monotone_interval_l412_41237

theorem max_monotone_interval (f : ℝ → ℝ) (h : f = λ x => Real.sin (Real.pi * x - Real.pi / 6)) :
  (∃ m : ℝ, m = 2/3 ∧ 
   (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ m → f x₁ < f x₂) ∧
   (∀ m' : ℝ, m' > m → ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ m' ∧ f x₁ ≥ f x₂)) :=
sorry

end max_monotone_interval_l412_41237


namespace gcd_2_exp_1020_minus_1_2_exp_1031_minus_1_l412_41294

theorem gcd_2_exp_1020_minus_1_2_exp_1031_minus_1 :
  Nat.gcd (2^1020 - 1) (2^1031 - 1) = 1 := by
  sorry

end gcd_2_exp_1020_minus_1_2_exp_1031_minus_1_l412_41294


namespace parallel_vectors_sum_l412_41293

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (2, 1) and b = (x, -2) for some x ∈ ℝ, then a + b = (-2, -1) -/
theorem parallel_vectors_sum (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![x, -2]
  (∃ (k : ℝ), a = k • b) →
  a + b = ![(-2), (-1)] := by
sorry

end parallel_vectors_sum_l412_41293
