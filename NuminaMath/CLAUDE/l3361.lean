import Mathlib

namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3361_336105

theorem sum_remainder_mod_seven (n : ℤ) : 
  ((7 - n) + (n + 3) + n^2) % 7 = (3 + n^2) % 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l3361_336105


namespace NUMINAMATH_CALUDE_subset_conditions_l3361_336124

/-- Given sets A and B, prove the conditions for m when A is a proper subset of B -/
theorem subset_conditions (m : ℝ) : 
  let A : Set ℝ := {3, m^2}
  let B : Set ℝ := {1, 3, 2*m-1}
  (A ⊂ B) → (m^2 ≠ 1 ∧ m^2 ≠ 2*m-1 ∧ m^2 ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_subset_conditions_l3361_336124


namespace NUMINAMATH_CALUDE_biker_bob_route_l3361_336130

theorem biker_bob_route (distance_AB : Real) (net_west : Real) (initial_north : Real) :
  distance_AB = 20.615528128088304 →
  net_west = 5 →
  initial_north = 15 →
  ∃ x : Real, 
    x ≥ 0 ∧ 
    (x + initial_north)^2 + net_west^2 = distance_AB^2 ∧ 
    abs (x - 5.021531) < 0.000001 := by
  sorry

#check biker_bob_route

end NUMINAMATH_CALUDE_biker_bob_route_l3361_336130


namespace NUMINAMATH_CALUDE_score_difference_l3361_336110

theorem score_difference (score : ℕ) (h : score = 15) : 3 * score - 2 * score = 15 := by
  sorry

end NUMINAMATH_CALUDE_score_difference_l3361_336110


namespace NUMINAMATH_CALUDE_robins_bracelet_cost_l3361_336116

/-- Represents a friend's name -/
inductive Friend
| jessica
| tori
| lily
| patrice

/-- Returns the number of letters in a friend's name -/
def nameLength (f : Friend) : Nat :=
  match f with
  | .jessica => 7
  | .tori => 4
  | .lily => 4
  | .patrice => 7

/-- The cost of a single bracelet in dollars -/
def braceletCost : Nat := 2

/-- The list of Robin's friends -/
def friendsList : List Friend := [Friend.jessica, Friend.tori, Friend.lily, Friend.patrice]

/-- Theorem: The total cost for Robin's bracelets is $44 -/
theorem robins_bracelet_cost : 
  (friendsList.map nameLength).sum * braceletCost = 44 := by
  sorry


end NUMINAMATH_CALUDE_robins_bracelet_cost_l3361_336116


namespace NUMINAMATH_CALUDE_bike_rides_total_l3361_336148

/-- The number of times Billy rode his bike -/
def billy_rides : ℕ := 17

/-- The number of times John rode his bike -/
def john_rides : ℕ := 2 * billy_rides

/-- The number of times their mother rode her bike -/
def mother_rides : ℕ := john_rides + 10

/-- The total number of times they rode their bikes -/
def total_rides : ℕ := billy_rides + john_rides + mother_rides

theorem bike_rides_total : total_rides = 95 := by
  sorry

end NUMINAMATH_CALUDE_bike_rides_total_l3361_336148


namespace NUMINAMATH_CALUDE_mixture_weight_approx_140_l3361_336192

/-- Represents the weight ratio of almonds to walnuts in the mixture -/
def almond_to_walnut_ratio : ℚ := 5

/-- Represents the weight of almonds in the mixture in pounds -/
def almond_weight : ℚ := 116.67

/-- Calculates the total weight of the mixture -/
def total_mixture_weight : ℚ :=
  almond_weight + (almond_weight / almond_to_walnut_ratio)

/-- Theorem stating that the total weight of the mixture is approximately 140 pounds -/
theorem mixture_weight_approx_140 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |total_mixture_weight - 140| < ε :=
sorry

end NUMINAMATH_CALUDE_mixture_weight_approx_140_l3361_336192


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3361_336136

/-- The length of the major axis of an ellipse formed by intersecting a right circular cylinder --/
def majorAxisLength (cylinderRadius : ℝ) (majorAxisLongerRatio : ℝ) : ℝ :=
  2 * cylinderRadius * (1 + majorAxisLongerRatio)

/-- Theorem stating the length of the major axis of the ellipse --/
theorem ellipse_major_axis_length :
  majorAxisLength 2 0.3 = 5.2 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3361_336136


namespace NUMINAMATH_CALUDE_cracker_difference_l3361_336140

theorem cracker_difference (marcus_crackers : ℕ) (nicholas_crackers : ℕ) : 
  marcus_crackers = 27 →
  nicholas_crackers = 15 →
  ∃ (mona_crackers : ℕ), 
    marcus_crackers = 3 * mona_crackers ∧
    nicholas_crackers = mona_crackers + 6 :=
by
  sorry

end NUMINAMATH_CALUDE_cracker_difference_l3361_336140


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l3361_336146

def total_people : ℕ := 15
def num_men : ℕ := 10
def num_women : ℕ := 5
def selected : ℕ := 5

theorem probability_at_least_one_woman :
  let p := 1 - (num_men.choose selected / total_people.choose selected)
  p = 917 / 1001 := by sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l3361_336146


namespace NUMINAMATH_CALUDE_total_work_hours_l3361_336129

theorem total_work_hours (hours_per_day : ℕ) (days_worked : ℕ) : 
  hours_per_day = 3 → days_worked = 5 → hours_per_day * days_worked = 15 :=
by sorry

end NUMINAMATH_CALUDE_total_work_hours_l3361_336129


namespace NUMINAMATH_CALUDE_hill_climbing_speed_l3361_336178

/-- Proves that the average speed while climbing is 2.625 km/h given the conditions of the journey -/
theorem hill_climbing_speed 
  (uphill_time : ℝ) 
  (downhill_time : ℝ) 
  (total_average_speed : ℝ) 
  (h1 : uphill_time = 4)
  (h2 : downhill_time = 2)
  (h3 : total_average_speed = 3.5) : 
  (total_average_speed * (uphill_time + downhill_time)) / (2 * uphill_time) = 2.625 := by
  sorry

#eval (3.5 * (4 + 2)) / (2 * 4)  -- This should evaluate to 2.625

end NUMINAMATH_CALUDE_hill_climbing_speed_l3361_336178


namespace NUMINAMATH_CALUDE_hyperbola_ratio_range_l3361_336177

/-- A hyperbola with foci F₁ and F₂, and a point G satisfying specific conditions -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  G : ℝ × ℝ
  hC : G.1^2 / a^2 - G.2^2 / b^2 = 1
  hG : Real.sqrt ((G.1 - F₁.1)^2 + (G.2 - F₁.2)^2) = 7 * Real.sqrt ((G.1 - F₂.1)^2 + (G.2 - F₂.2)^2)

/-- The range of b/a for a hyperbola satisfying the given conditions -/
theorem hyperbola_ratio_range (h : Hyperbola) : 0 < h.b / h.a ∧ h.b / h.a ≤ Real.sqrt 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_ratio_range_l3361_336177


namespace NUMINAMATH_CALUDE_jack_remaining_notebooks_l3361_336100

-- Define the initial number of notebooks for Gerald
def gerald_notebooks : ℕ := 8

-- Define Jack's initial number of notebooks relative to Gerald's
def jack_initial_notebooks : ℕ := gerald_notebooks + 13

-- Define the number of notebooks Jack gives to Paula
def notebooks_to_paula : ℕ := 5

-- Define the number of notebooks Jack gives to Mike
def notebooks_to_mike : ℕ := 6

-- Theorem: Jack has 10 notebooks left
theorem jack_remaining_notebooks :
  jack_initial_notebooks - (notebooks_to_paula + notebooks_to_mike) = 10 := by
  sorry

end NUMINAMATH_CALUDE_jack_remaining_notebooks_l3361_336100


namespace NUMINAMATH_CALUDE_factorization_equality_l3361_336144

theorem factorization_equality (y a : ℝ) : 3*y*a^2 - 6*y*a + 3*y = 3*y*(a-1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3361_336144


namespace NUMINAMATH_CALUDE_hypotenuse_length_of_special_triangle_l3361_336188

theorem hypotenuse_length_of_special_triangle : 
  ∀ (a b c : ℝ), 
  (a^2 - 17*a + 60 = 0) → 
  (b^2 - 17*b + 60 = 0) → 
  (a ≠ b) →
  (c^2 = a^2 + b^2) →
  c = 13 := by
sorry

end NUMINAMATH_CALUDE_hypotenuse_length_of_special_triangle_l3361_336188


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l3361_336126

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - k*x + 1 = (x - a)^2) → (k = 2 ∨ k = -2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l3361_336126


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_q_sufficient_not_necessary_l3361_336183

-- Define the propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0

-- Theorem for part (1)
theorem range_of_x_when_a_is_one (x : ℝ) (h : p 1 x ∧ q x) : 2 < x ∧ x < 3 := by
  sorry

-- Theorem for part (2)
theorem range_of_a_when_q_sufficient_not_necessary (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x, q x → p a x)
  (h3 : ∃ x, p a x ∧ ¬q x) : 
  1 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_one_range_of_a_when_q_sufficient_not_necessary_l3361_336183


namespace NUMINAMATH_CALUDE_paulas_remaining_money_l3361_336158

/-- Calculates the remaining money after shopping given the initial amount, 
    number of shirts, cost per shirt, and cost of pants. -/
def remaining_money (initial : ℕ) (num_shirts : ℕ) (shirt_cost : ℕ) (pants_cost : ℕ) : ℕ :=
  initial - (num_shirts * shirt_cost + pants_cost)

/-- Theorem stating that Paula's remaining money after shopping is $74 -/
theorem paulas_remaining_money :
  remaining_money 109 2 11 13 = 74 := by
  sorry

end NUMINAMATH_CALUDE_paulas_remaining_money_l3361_336158


namespace NUMINAMATH_CALUDE_cone_syrup_amount_l3361_336155

/-- The amount of chocolate syrup used on each shake in ounces -/
def syrup_per_shake : ℕ := 4

/-- The number of shakes sold -/
def num_shakes : ℕ := 2

/-- The number of cones sold -/
def num_cones : ℕ := 1

/-- The total amount of chocolate syrup used in ounces -/
def total_syrup : ℕ := 14

/-- The amount of chocolate syrup used on each cone in ounces -/
def syrup_per_cone : ℕ := total_syrup - (syrup_per_shake * num_shakes)

theorem cone_syrup_amount : syrup_per_cone = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_syrup_amount_l3361_336155


namespace NUMINAMATH_CALUDE_sally_net_earnings_two_months_l3361_336151

-- Define the given values
def last_month_work_income : ℝ := 1000
def last_month_work_expenses : ℝ := 200
def last_month_side_hustle : ℝ := 150
def work_income_increase : ℝ := 0.1
def work_expenses_increase : ℝ := 0.15
def side_hustle_increase : ℝ := 0.2
def tax_rate : ℝ := 0.25

-- Define the calculation functions
def calculate_net_work_income (income : ℝ) (expenses : ℝ) : ℝ :=
  income - expenses - (tax_rate * income)

def calculate_total_net_earnings (work_income : ℝ) (side_hustle : ℝ) : ℝ :=
  calculate_net_work_income work_income last_month_work_expenses + side_hustle

-- Theorem statement
theorem sally_net_earnings_two_months :
  let last_month := calculate_total_net_earnings last_month_work_income last_month_side_hustle
  let this_month := calculate_total_net_earnings 
    (last_month_work_income * (1 + work_income_increase))
    (last_month_side_hustle * (1 + side_hustle_increase))
  last_month + this_month = 1475 := by sorry

end NUMINAMATH_CALUDE_sally_net_earnings_two_months_l3361_336151


namespace NUMINAMATH_CALUDE_max_gold_tokens_l3361_336111

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  gold : ℕ

/-- Represents an exchange booth --/
structure Booth where
  red_in : ℕ
  blue_in : ℕ
  red_out : ℕ
  blue_out : ℕ
  gold_out : ℕ

/-- Checks if an exchange is possible at a given booth --/
def canExchange (tokens : TokenCount) (booth : Booth) : Bool :=
  tokens.red ≥ booth.red_in ∧ tokens.blue ≥ booth.blue_in

/-- Performs an exchange at a given booth --/
def exchange (tokens : TokenCount) (booth : Booth) : TokenCount :=
  { red := tokens.red - booth.red_in + booth.red_out,
    blue := tokens.blue - booth.blue_in + booth.blue_out,
    gold := tokens.gold + booth.gold_out }

/-- The main theorem to prove --/
theorem max_gold_tokens : ∃ (final : TokenCount),
  let initial := TokenCount.mk 100 100 0
  let booth1 := Booth.mk 3 0 0 2 1
  let booth2 := Booth.mk 0 4 2 0 1
  (¬ canExchange final booth1 ∧ ¬ canExchange final booth2) ∧
  final.gold = 133 ∧
  (∀ (other : TokenCount),
    (¬ canExchange other booth1 ∧ ¬ canExchange other booth2) →
    other.gold ≤ final.gold) :=
sorry

end NUMINAMATH_CALUDE_max_gold_tokens_l3361_336111


namespace NUMINAMATH_CALUDE_randys_trees_l3361_336120

/-- Proves that Randy has 5 less coconut trees compared to half the number of mango trees -/
theorem randys_trees (mango_trees : ℕ) (total_trees : ℕ) (coconut_trees : ℕ) : 
  mango_trees = 60 →
  total_trees = 85 →
  coconut_trees = total_trees - mango_trees →
  coconut_trees < mango_trees / 2 →
  mango_trees / 2 - coconut_trees = 5 := by
sorry

end NUMINAMATH_CALUDE_randys_trees_l3361_336120


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l3361_336174

theorem square_sum_given_sum_and_product (a b : ℝ) : a + b = 8 → a * b = -2 → a^2 + b^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l3361_336174


namespace NUMINAMATH_CALUDE_four_digit_divisibility_l3361_336134

def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem four_digit_divisibility (p q : ℕ) : 
  is_two_digit_prime p ∧ 
  is_two_digit_prime q ∧ 
  p ≠ q ∧
  (100 * p + q) % ((p + q) / 2) = 0 ∧ 
  (100 * q + p) % ((p + q) / 2) = 0 →
  ({p, q} : Set ℕ) = {13, 53} ∨ 
  ({p, q} : Set ℕ) = {19, 47} ∨ 
  ({p, q} : Set ℕ) = {23, 43} ∨ 
  ({p, q} : Set ℕ) = {29, 37} :=
by sorry

end NUMINAMATH_CALUDE_four_digit_divisibility_l3361_336134


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3361_336149

/-- The eccentricity range of a hyperbola -/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let e := Real.sqrt (1 + b^2 / a^2)
  let c := Real.sqrt (a^2 + b^2)
  let d := a * b / c
  d ≥ Real.sqrt 2 / 3 * c →
  Real.sqrt 6 / 2 ≤ e ∧ e ≤ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l3361_336149


namespace NUMINAMATH_CALUDE_dans_car_mpg_l3361_336128

/-- Calculates the miles per gallon of Dan's car given the cost of gas and distance traveled on a certain amount of money. -/
theorem dans_car_mpg (gas_cost : ℝ) (miles : ℝ) (spent : ℝ) : 
  gas_cost = 4 → miles = 432 → spent = 54 → (miles / (spent / gas_cost)) = 32 :=
by sorry

end NUMINAMATH_CALUDE_dans_car_mpg_l3361_336128


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3361_336153

theorem quadratic_factorization (x : ℝ) : x^2 + 2*x - 3 = (x + 3) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3361_336153


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l3361_336142

-- Define the triangle AEF
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the area function
def area : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

-- Define the parallel relation
def parallel : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop := sorry

-- Define the cyclic property
def is_cyclic : Quadrilateral → Prop := sorry

-- Define the distance function
def distance : (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry

-- Define the theorem
theorem area_ratio_theorem (AEF : Triangle) (ABCD : Quadrilateral) :
  distance AEF.B AEF.C = 20 →
  distance AEF.A AEF.B = 21 →
  distance AEF.A AEF.C = 21 →
  parallel ABCD.B ABCD.D AEF.B AEF.C →
  is_cyclic ABCD →
  distance ABCD.B ABCD.C = 3 →
  distance ABCD.C ABCD.D = 4 →
  (area ABCD.A ABCD.B ABCD.C + area ABCD.A ABCD.C ABCD.D) / area AEF.A AEF.B AEF.C = 49 / 400 :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l3361_336142


namespace NUMINAMATH_CALUDE_amy_balloon_count_l3361_336141

/-- The number of balloons James has -/
def james_balloons : ℕ := 1222

/-- The difference between James' and Amy's balloon counts -/
def difference : ℕ := 208

/-- Amy's balloon count -/
def amy_balloons : ℕ := james_balloons - difference

theorem amy_balloon_count : amy_balloons = 1014 := by
  sorry

end NUMINAMATH_CALUDE_amy_balloon_count_l3361_336141


namespace NUMINAMATH_CALUDE_correct_train_sequence_l3361_336106

-- Define the actions
inductive TrainAction
  | BuyTicket
  | WaitForTrain
  | CheckTicket
  | BoardTrain
  | RepairTrain

-- Define a sequence of actions
def ActionSequence := List TrainAction

-- Define the possible sequences
def sequenceA : ActionSequence := [TrainAction.BuyTicket, TrainAction.WaitForTrain, TrainAction.CheckTicket, TrainAction.BoardTrain]
def sequenceB : ActionSequence := [TrainAction.WaitForTrain, TrainAction.BuyTicket, TrainAction.BoardTrain, TrainAction.CheckTicket]
def sequenceC : ActionSequence := [TrainAction.BuyTicket, TrainAction.WaitForTrain, TrainAction.BoardTrain, TrainAction.CheckTicket]
def sequenceD : ActionSequence := [TrainAction.RepairTrain, TrainAction.BuyTicket, TrainAction.CheckTicket, TrainAction.BoardTrain]

-- Define the correct sequence
def correctSequence : ActionSequence := sequenceA

-- Theorem stating that sequenceA is the correct sequence
theorem correct_train_sequence : correctSequence = sequenceA := by
  sorry


end NUMINAMATH_CALUDE_correct_train_sequence_l3361_336106


namespace NUMINAMATH_CALUDE_gcd_90_250_l3361_336115

theorem gcd_90_250 : Nat.gcd 90 250 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_250_l3361_336115


namespace NUMINAMATH_CALUDE_cube_root_monotone_l3361_336122

theorem cube_root_monotone (a b : ℝ) (h : a ≤ b) : a ^ (1/3) ≤ b ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_monotone_l3361_336122


namespace NUMINAMATH_CALUDE_train_length_calculation_l3361_336185

/-- Prove that given two trains of equal length running on parallel lines in the same direction,
    with the faster train moving at 46 km/hr and the slower train at 36 km/hr,
    if the faster train completely passes the slower train in 18 seconds,
    then the length of each train is 25 meters. -/
theorem train_length_calculation (faster_speed slower_speed : ℝ) (passing_time : ℝ) (train_length : ℝ) :
  faster_speed = 46 →
  slower_speed = 36 →
  passing_time = 18 →
  (faster_speed - slower_speed) * (5 / 18) * passing_time = 2 * train_length →
  train_length = 25 := by
  sorry


end NUMINAMATH_CALUDE_train_length_calculation_l3361_336185


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3361_336172

-- Define the quadratic expression
def quadratic (k : ℝ) : ℝ := 8 * k^2 - 16 * k + 28

-- Define the completed square form
def completed_square (k a b c : ℝ) : ℝ := a * (k + b)^2 + c

-- Theorem statement
theorem quadratic_rewrite :
  ∃ (a b c : ℝ), 
    (∀ k, quadratic k = completed_square k a b c) ∧ 
    (c / b = -20) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3361_336172


namespace NUMINAMATH_CALUDE_cakes_per_person_l3361_336186

theorem cakes_per_person (total_cakes : ℕ) (num_friends : ℕ) (h1 : total_cakes = 8) (h2 : num_friends = 4) :
  total_cakes / num_friends = 2 := by
  sorry

end NUMINAMATH_CALUDE_cakes_per_person_l3361_336186


namespace NUMINAMATH_CALUDE_at_least_one_zero_one_is_zero_l3361_336161

theorem at_least_one_zero (a b : ℝ) : (a ≠ 0 ∧ b ≠ 0) → False := by
  sorry

theorem one_is_zero (a b : ℝ) : a = 0 ∨ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_zero_one_is_zero_l3361_336161


namespace NUMINAMATH_CALUDE_solve_for_a_l3361_336164

theorem solve_for_a : ∀ a : ℝ, (∃ x : ℝ, x = 1 ∧ 2 * x - a = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l3361_336164


namespace NUMINAMATH_CALUDE_expected_balls_original_value_l3361_336113

/-- The number of balls arranged in a circle -/
def num_balls : ℕ := 7

/-- The number of interchanges performed -/
def num_interchanges : ℕ := 4

/-- The probability of a specific ball being chosen for an interchange -/
def prob_chosen : ℚ := 2 / 7

/-- The probability of a ball being in its original position after the interchanges -/
def prob_original_position : ℚ :=
  (1 - prob_chosen) ^ num_interchanges +
  (num_interchanges.choose 2) * prob_chosen ^ 2 * (1 - prob_chosen) ^ 2 +
  prob_chosen ^ num_interchanges

/-- The expected number of balls in their original positions -/
def expected_balls_original : ℚ := num_balls * prob_original_position

theorem expected_balls_original_value :
  expected_balls_original = 3.61 := by sorry

end NUMINAMATH_CALUDE_expected_balls_original_value_l3361_336113


namespace NUMINAMATH_CALUDE_probability_of_selection_X_l3361_336154

theorem probability_of_selection_X 
  (prob_Y : ℝ) 
  (prob_X_and_Y : ℝ) 
  (h1 : prob_Y = 2/5) 
  (h2 : prob_X_and_Y = 0.13333333333333333) :
  ∃ (prob_X : ℝ), prob_X = 1/3 ∧ prob_X_and_Y = prob_X * prob_Y :=
by
  sorry

end NUMINAMATH_CALUDE_probability_of_selection_X_l3361_336154


namespace NUMINAMATH_CALUDE_car_speed_problem_l3361_336109

theorem car_speed_problem (v : ℝ) : 
  (∀ (t : ℝ), t = 3 → (70 - v) * t = 60) → v = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3361_336109


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l3361_336125

-- Define the function
def f (x : ℝ) : ℝ := 2 * |x - 3| - 8

-- Theorem statement
theorem minimum_point_of_translated_graph :
  ∀ x : ℝ, f x ≥ f 3 ∧ f 3 = -8 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l3361_336125


namespace NUMINAMATH_CALUDE_smallest_norm_l3361_336191

open Real
open InnerProductSpace

/-- Given a vector v such that ‖v + (4, 2)‖ = 10, the smallest possible value of ‖v‖ is 10 - 2√5 -/
theorem smallest_norm (v : ℝ × ℝ) (h : ‖v + (4, 2)‖ = 10) :
  ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ ∀ u : ℝ × ℝ, ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ :=
by sorry

end NUMINAMATH_CALUDE_smallest_norm_l3361_336191


namespace NUMINAMATH_CALUDE_v2_value_for_f_at_2_l3361_336102

def f (x : ℝ) : ℝ := 2 * x^5 - 3 * x + 2 * x^2 - x + 5

def qin_jiushao_v2 (a b c d e : ℝ) (x : ℝ) : ℝ :=
  (a * x + b) * x + c

theorem v2_value_for_f_at_2 :
  let a := 2
  let b := 3
  let c := 0
  qin_jiushao_v2 a b c 5 (-4) 2 = 14 := by sorry

end NUMINAMATH_CALUDE_v2_value_for_f_at_2_l3361_336102


namespace NUMINAMATH_CALUDE_initial_squares_step_increase_squares_after_five_steps_l3361_336197

/-- The number of squares after n steps in the square subdivision process -/
def num_squares (n : ℕ) : ℕ := 5 + 4 * n

/-- The square subdivision process starts with 5 squares -/
theorem initial_squares : num_squares 0 = 5 := by sorry

/-- Each step adds 4 new squares -/
theorem step_increase (n : ℕ) : num_squares (n + 1) = num_squares n + 4 := by sorry

/-- The number of squares after 5 steps is 25 -/
theorem squares_after_five_steps : num_squares 5 = 25 := by sorry

end NUMINAMATH_CALUDE_initial_squares_step_increase_squares_after_five_steps_l3361_336197


namespace NUMINAMATH_CALUDE_homogeneous_polynomial_theorem_l3361_336199

variable {n : ℕ}

-- Define a homogeneous polynomial of degree n
def IsHomogeneousPolynomial (f : ℝ → ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ (x y t : ℝ), f (t * x) (t * y) = t^n * f x y

theorem homogeneous_polynomial_theorem (f : ℝ → ℝ → ℝ) (n : ℕ) 
  (h1 : IsHomogeneousPolynomial f n)
  (h2 : f 1 0 = 1)
  (h3 : ∀ (a b c : ℝ), f (a + b) c + f (b + c) a + f (c + a) b = 0) :
  ∀ (x y : ℝ), f x y = (x - 2*y) * (x + y)^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_homogeneous_polynomial_theorem_l3361_336199


namespace NUMINAMATH_CALUDE_min_value_expression_l3361_336139

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  ∃ (min : ℝ), min = 162 ∧ ∀ x y z, x > 0 → y > 0 → z > 0 → x * y * z = 27 →
    x^2 + 6*x*y + 9*y^2 + 3*z^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3361_336139


namespace NUMINAMATH_CALUDE_age_problem_l3361_336156

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 27 →
  b = 10 :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l3361_336156


namespace NUMINAMATH_CALUDE_uncle_dave_ice_cream_l3361_336195

/-- The number of ice cream sandwiches Uncle Dave bought -/
def total_ice_cream_sandwiches : ℕ := sorry

/-- The number of Uncle Dave's nieces -/
def number_of_nieces : ℕ := 11

/-- The number of ice cream sandwiches each niece would get -/
def sandwiches_per_niece : ℕ := 13

/-- Theorem stating that the total number of ice cream sandwiches is 143 -/
theorem uncle_dave_ice_cream : total_ice_cream_sandwiches = number_of_nieces * sandwiches_per_niece := by
  sorry

end NUMINAMATH_CALUDE_uncle_dave_ice_cream_l3361_336195


namespace NUMINAMATH_CALUDE_largest_sphere_on_torus_l3361_336170

/-- Represents a torus generated by revolving a circle about the z-axis -/
structure Torus where
  inner_radius : ℝ
  outer_radius : ℝ
  circle_center : ℝ × ℝ × ℝ
  circle_radius : ℝ

/-- Represents a sphere centered on the z-axis -/
structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Checks if a sphere touches the horizontal plane -/
def touches_plane (s : Sphere) : Prop :=
  s.center.2.2 = s.radius

/-- Checks if a sphere touches the top of a torus -/
def touches_torus (t : Torus) (s : Sphere) : Prop :=
  (t.circle_center.1 - s.center.1) ^ 2 + (t.circle_center.2.2 - s.center.2.2) ^ 2 = (s.radius + t.circle_radius) ^ 2

theorem largest_sphere_on_torus (t : Torus) (s : Sphere) :
  t.inner_radius = 3 ∧
  t.outer_radius = 5 ∧
  t.circle_center = (4, 0, 1) ∧
  t.circle_radius = 1 ∧
  s.center.1 = 0 ∧
  s.center.2.1 = 0 ∧
  touches_plane s ∧
  touches_torus t s →
  s.radius = 4 :=
sorry

end NUMINAMATH_CALUDE_largest_sphere_on_torus_l3361_336170


namespace NUMINAMATH_CALUDE_sum_of_three_smallest_solutions_l3361_336118

def is_solution (x : ℝ) : Prop :=
  x > 0 ∧ x - ⌊x⌋ = 1 / (⌊x⌋^2)

def smallest_solutions : Set ℝ :=
  {x | is_solution x ∧ ∀ y, is_solution y → x ≤ y}

theorem sum_of_three_smallest_solutions :
  ∃ (a b c : ℝ), a ∈ smallest_solutions ∧ b ∈ smallest_solutions ∧ c ∈ smallest_solutions ∧
  (∀ x ∈ smallest_solutions, x = a ∨ x = b ∨ x = c) ∧
  a + b + c = 9 + 17/36 :=
sorry

end NUMINAMATH_CALUDE_sum_of_three_smallest_solutions_l3361_336118


namespace NUMINAMATH_CALUDE_max_remainder_and_dividend_l3361_336131

theorem max_remainder_and_dividend (star : ℕ) (triangle : ℕ) :
  star / 7 = 102 ∧ star % 7 = triangle →
  triangle ≤ 6 ∧
  (triangle = 6 → star = 720) :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_and_dividend_l3361_336131


namespace NUMINAMATH_CALUDE_ratio_to_twelve_l3361_336119

theorem ratio_to_twelve : ∃ x : ℝ, (5 : ℝ) / 1 = x / 12 ∧ x = 60 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_twelve_l3361_336119


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3361_336112

theorem inequality_solution_set : 
  {x : ℝ | -x^2 - x + 6 > 0} = Set.Ioo (-3 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3361_336112


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3361_336163

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x < 0)) ↔ (∃ x : ℝ, x^2 - x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3361_336163


namespace NUMINAMATH_CALUDE_correct_num_technicians_l3361_336173

/-- The number of technicians in a workshop. -/
def num_technicians : ℕ := 7

/-- The total number of workers in the workshop. -/
def total_workers : ℕ := 42

/-- The average salary of all workers in the workshop. -/
def avg_salary_all : ℕ := 8000

/-- The average salary of technicians in the workshop. -/
def avg_salary_technicians : ℕ := 18000

/-- The average salary of non-technician workers in the workshop. -/
def avg_salary_others : ℕ := 6000

/-- Theorem stating that the number of technicians is correct given the workshop conditions. -/
theorem correct_num_technicians :
  num_technicians = 7 ∧
  num_technicians + (total_workers - num_technicians) = total_workers ∧
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_others) / total_workers = avg_salary_all :=
by sorry

end NUMINAMATH_CALUDE_correct_num_technicians_l3361_336173


namespace NUMINAMATH_CALUDE_square_plus_product_equals_twelve_plus_two_sqrt_six_l3361_336127

theorem square_plus_product_equals_twelve_plus_two_sqrt_six :
  ∀ a b : ℝ,
  a = Real.sqrt 6 + 1 →
  b = Real.sqrt 6 - 1 →
  a^2 + a*b = 12 + 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_product_equals_twelve_plus_two_sqrt_six_l3361_336127


namespace NUMINAMATH_CALUDE_divisibility_condition_l3361_336194

theorem divisibility_condition (a b : ℤ) : 
  (∃ d : ℕ, d ≥ 2 ∧ ∀ n : ℕ, n > 0 → (d : ℤ) ∣ (a^n + b^n + 1)) ↔ 
  ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 3 = 1 ∧ b % 3 = 1)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3361_336194


namespace NUMINAMATH_CALUDE_tg_arccos_leq_sin_arctg_l3361_336168

theorem tg_arccos_leq_sin_arctg (x : ℝ) : 
  x ∈ Set.Icc (-1 : ℝ) 1 →
  (Real.tan (Real.arccos x) ≤ Real.sin (Real.arctan x) ↔ 
   x ∈ Set.Icc (-(Real.sqrt (Real.sqrt (1/2)))) 0 ∪ Set.Icc (Real.sqrt (Real.sqrt (1/2))) 1) :=
by sorry

end NUMINAMATH_CALUDE_tg_arccos_leq_sin_arctg_l3361_336168


namespace NUMINAMATH_CALUDE_not_all_structure_diagrams_are_tree_shaped_l3361_336171

/-- Represents a structure diagram -/
structure StructureDiagram where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Property: Elements show conceptual subordination and logical sequence -/
def shows_conceptual_subordination (sd : StructureDiagram) : Prop :=
  sorry

/-- Property: Can reflect relationships and overall characteristics -/
def reflects_relationships (sd : StructureDiagram) : Prop :=
  sorry

/-- Property: Can reflect system details thoroughly -/
def reflects_details (sd : StructureDiagram) : Prop :=
  sorry

/-- Property: Is tree-shaped -/
def is_tree_shaped (sd : StructureDiagram) : Prop :=
  sorry

/-- Theorem: Not all structure diagrams are tree-shaped -/
theorem not_all_structure_diagrams_are_tree_shaped :
  ¬ (∀ sd : StructureDiagram, is_tree_shaped sd) :=
sorry

end NUMINAMATH_CALUDE_not_all_structure_diagrams_are_tree_shaped_l3361_336171


namespace NUMINAMATH_CALUDE_place_mat_length_l3361_336160

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) : 
  r = 6 ∧ n = 8 ∧ w = 1 ∧ 
  (∀ i : Fin n, ∃ p₁ p₂ : ℝ × ℝ, 
    (p₁.1 - r)^2 + p₁.2^2 = r^2 ∧
    (p₂.1 - r)^2 + p₂.2^2 = r^2 ∧
    (p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2 = w^2) ∧
  (∀ i : Fin n, ∃ q₁ q₂ : ℝ × ℝ,
    (q₁.1 - r)^2 + q₁.2^2 < r^2 ∧
    (q₂.1 - r)^2 + q₂.2^2 < r^2 ∧
    (q₂.1 - q₁.1)^2 + (q₂.2 - q₁.2)^2 = y^2 ∧
    (∃ j : Fin n, j ≠ i ∧ (q₂.1 = q₁.1 ∨ q₂.2 = q₁.2))) →
  y = 3 * Real.sqrt (2 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_place_mat_length_l3361_336160


namespace NUMINAMATH_CALUDE_min_points_tenth_game_l3361_336196

def points_four_games : List ℕ := [18, 22, 15, 19]

def average_greater_than_19 (total_points : ℕ) : Prop :=
  (total_points : ℚ) / 10 > 19

theorem min_points_tenth_game 
  (h1 : (points_four_games.sum : ℚ) / 4 > (List.sum (List.take 6 points_four_games) : ℚ) / 6)
  (h2 : ∃ (p : ℕ), average_greater_than_19 (points_four_games.sum + List.sum (List.take 6 points_four_games) + p)) :
  ∃ (p : ℕ), p ≥ 9 ∧ average_greater_than_19 (points_four_games.sum + List.sum (List.take 6 points_four_games) + p) ∧
  ∀ (q : ℕ), q < 9 → ¬average_greater_than_19 (points_four_games.sum + List.sum (List.take 6 points_four_games) + q) :=
sorry

end NUMINAMATH_CALUDE_min_points_tenth_game_l3361_336196


namespace NUMINAMATH_CALUDE_system_solution_no_solution_l3361_336121

-- Problem 1
theorem system_solution (x y : ℝ) :
  x - y = 8 ∧ 3*x + y = 12 → x = 5 ∧ y = -3 := by sorry

-- Problem 2
theorem no_solution (x : ℝ) :
  x ≠ 1 → 3 / (x - 1) - (x + 2) / (x * (x - 1)) ≠ 0 := by sorry

end NUMINAMATH_CALUDE_system_solution_no_solution_l3361_336121


namespace NUMINAMATH_CALUDE_gcd_102_238_l3361_336108

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end NUMINAMATH_CALUDE_gcd_102_238_l3361_336108


namespace NUMINAMATH_CALUDE_square_binomial_divided_by_negative_square_l3361_336187

theorem square_binomial_divided_by_negative_square (m : ℝ) (hm : m ≠ 0) :
  (2 * m^2 - m)^2 / (-m^2) = -4 * m^2 + 4 * m - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_divided_by_negative_square_l3361_336187


namespace NUMINAMATH_CALUDE_triangle_inequality_violation_l3361_336162

theorem triangle_inequality_violation (a b c : ℝ) : 
  a = 1 ∧ b = 2 ∧ c = 7 → ¬(a + b > c ∧ b + c > a ∧ c + a > b) := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_violation_l3361_336162


namespace NUMINAMATH_CALUDE_symmetric_point_x_axis_l3361_336138

/-- Given a point P(2, 5), its symmetric point with respect to the x-axis has coordinates (2, -5) -/
theorem symmetric_point_x_axis : 
  let P : ℝ × ℝ := (2, 5)
  let symmetric_point := (P.1, -P.2)
  symmetric_point = (2, -5) := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_x_axis_l3361_336138


namespace NUMINAMATH_CALUDE_A_B_red_mutually_exclusive_not_contradictory_l3361_336193

-- Define the set of cards
inductive Card : Type
| Black : Card
| Red : Card
| White : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards to people
def Distribution := Person → Card

-- Define the event "A gets the red card"
def A_gets_red (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "B gets the red card"
def B_gets_red (d : Distribution) : Prop := d Person.B = Card.Red

-- Theorem stating that "A gets the red card" and "B gets the red card" are mutually exclusive but not contradictory
theorem A_B_red_mutually_exclusive_not_contradictory :
  (∀ d : Distribution, ¬(A_gets_red d ∧ B_gets_red d)) ∧
  (∃ d1 d2 : Distribution, A_gets_red d1 ∧ B_gets_red d2) :=
sorry

end NUMINAMATH_CALUDE_A_B_red_mutually_exclusive_not_contradictory_l3361_336193


namespace NUMINAMATH_CALUDE_polynomial_root_property_l3361_336166

/-- A polynomial of degree 4 with real coefficients -/
def PolynomialDegree4 (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

/-- The derivative of a polynomial of degree 4 -/
def DerivativePolynomialDegree4 (a b c : ℝ) (x : ℝ) : ℝ := 4*x^3 + 3*a*x^2 + 2*b*x + c

theorem polynomial_root_property (a b c d : ℝ) :
  let f := PolynomialDegree4 a b c d
  let f' := DerivativePolynomialDegree4 a b c
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) →
  (∃ w x y z : ℝ, w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z ∧
    f w = 0 ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) ∨
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f x = 0 ∧ f y = 0 ∧ f z = 0 ∧
    (f' x = 0 ∨ f' y = 0 ∨ f' z = 0)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_property_l3361_336166


namespace NUMINAMATH_CALUDE_chessboard_coverage_l3361_336150

/-- An L-shaped piece covers exactly 3 squares -/
def L_shape_coverage : ℕ := 3

/-- A unit square piece covers exactly 1 square -/
def unit_square_coverage : ℕ := 1

/-- Predicate to determine if an n×n chessboard can be covered -/
def can_cover (n : ℕ) : Prop :=
  ∃ k : ℕ, n^2 = k * L_shape_coverage ∨ n^2 = k * L_shape_coverage + unit_square_coverage

theorem chessboard_coverage (n : ℕ) :
  ¬(can_cover n) ↔ n % 3 = 2 :=
sorry

end NUMINAMATH_CALUDE_chessboard_coverage_l3361_336150


namespace NUMINAMATH_CALUDE_square_perimeter_l3361_336157

theorem square_perimeter (side_length : ℝ) (h : side_length = 13) : 
  4 * side_length = 52 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3361_336157


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l3361_336182

theorem nested_fraction_evaluation : 
  1 + 1 / (1 + 1 / (2 + 1)) = 7 / 4 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l3361_336182


namespace NUMINAMATH_CALUDE_tan_630_undefined_l3361_336184

theorem tan_630_undefined :
  ¬∃ (x : ℝ), Real.tan (630 * π / 180) = x :=
by
  sorry


end NUMINAMATH_CALUDE_tan_630_undefined_l3361_336184


namespace NUMINAMATH_CALUDE_kids_wearing_shoes_l3361_336107

theorem kids_wearing_shoes (total : ℕ) (socks : ℕ) (both : ℕ) (barefoot : ℕ) 
  (h_total : total = 22)
  (h_socks : socks = 12)
  (h_both : both = 6)
  (h_barefoot : barefoot = 8) :
  total - barefoot - (socks - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_kids_wearing_shoes_l3361_336107


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l3361_336133

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l3361_336133


namespace NUMINAMATH_CALUDE_square_ratio_problem_l3361_336145

theorem square_ratio_problem (A₁ A₂ : ℝ) (s₁ s₂ : ℝ) :
  A₁ / A₂ = 300 / 147 →
  A₁ = s₁^2 →
  A₂ = s₂^2 →
  4 * s₁ = 60 →
  s₁ / s₂ = 10 / 7 ∧ s₂ = 21 / 2 :=
by sorry

end NUMINAMATH_CALUDE_square_ratio_problem_l3361_336145


namespace NUMINAMATH_CALUDE_fraction_equality_implies_sum_l3361_336165

theorem fraction_equality_implies_sum (α β : ℝ) : 
  (∀ x : ℝ, (x - α) / (x + β) = (x^2 - 64*x + 992) / (x^2 + 56*x - 3168)) →
  α + β = 82 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_sum_l3361_336165


namespace NUMINAMATH_CALUDE_probability_square_or_triangle_l3361_336117

theorem probability_square_or_triangle :
  let total_figures : ℕ := 10
  let num_triangles : ℕ := 4
  let num_squares : ℕ := 3
  let num_circles : ℕ := 3
  let favorable_outcomes : ℕ := num_triangles + num_squares
  (favorable_outcomes : ℚ) / total_figures = 7 / 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_square_or_triangle_l3361_336117


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l3361_336101

/-- A positive arithmetic geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ (a (n + 1) - a n) = (a (n + 2) - a (n + 1))
    ∧ (a (n + 1))^2 = (a n) * (a (n + 2))

theorem arithmetic_geometric_sequence_property
  (a : ℕ → ℝ) (h : ArithmeticGeometricSequence a)
  (h_eq : a 1 * a 5 + 2 * a 3 * a 6 + a 1 * a 11 = 16) :
  a 3 + a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l3361_336101


namespace NUMINAMATH_CALUDE_max_dot_product_regular_octagon_l3361_336143

/-- Regular octagon with side length 1 -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : ∀ i j : Fin 8, 
    (i.val + 1) % 8 = j.val → 
    Real.sqrt ((vertices i).1 - (vertices j).1)^2 + ((vertices i).2 - (vertices j).2)^2 = 1

/-- Vector between two points -/
def vector (A : RegularOctagon) (i j : Fin 8) : ℝ × ℝ :=
  ((A.vertices j).1 - (A.vertices i).1, (A.vertices j).2 - (A.vertices i).2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem max_dot_product_regular_octagon (A : RegularOctagon) :
  ∃ (i j : Fin 8), ∀ (k l : Fin 8),
    dot_product (vector A k l) (vector A 0 1) ≤ dot_product (vector A i j) (vector A 0 1) ∧
    dot_product (vector A i j) (vector A 0 1) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_dot_product_regular_octagon_l3361_336143


namespace NUMINAMATH_CALUDE_units_digit_of_4589_pow_1276_l3361_336147

theorem units_digit_of_4589_pow_1276 : ∃ n : ℕ, 4589^1276 ≡ 1 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_4589_pow_1276_l3361_336147


namespace NUMINAMATH_CALUDE_ping_pong_paddles_sold_l3361_336114

/-- Given the total sales and average price per pair of ping pong paddles,
    prove the number of pairs sold. -/
theorem ping_pong_paddles_sold
  (total_sales : ℝ)
  (avg_price : ℝ)
  (h1 : total_sales = 735)
  (h2 : avg_price = 9.8) :
  total_sales / avg_price = 75 := by
  sorry

end NUMINAMATH_CALUDE_ping_pong_paddles_sold_l3361_336114


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3361_336180

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  (1 / (a + b) + 1 / (b + c) + 1 / (c + a)) ≥ 1.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3361_336180


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l3361_336103

def B : Set ℕ := {n | ∃ k : ℕ, n = k + (k + 1) + (k + 2) + (k + 3)}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ b ∈ B, d ∣ b) ∧ (∀ m : ℕ, (∀ b ∈ B, m ∣ b) → m ∣ d) ∧ d = 2 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l3361_336103


namespace NUMINAMATH_CALUDE_rationalize_denominator_l3361_336135

theorem rationalize_denominator (x : ℝ) (hx : x > 0) :
  (1 : ℝ) / (x^(1/3) + (27 : ℝ)^(1/3)) = (4 : ℝ)^(1/3) / (2 + 3 * (4 : ℝ)^(1/3)) :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l3361_336135


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3361_336152

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + 2*x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + 2*x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3361_336152


namespace NUMINAMATH_CALUDE_divisibility_criterion_l3361_336190

theorem divisibility_criterion (n : ℕ+) : 
  (n + 2 : ℕ) ∣ (n^3 + 3*n + 29 : ℕ) ↔ n = 1 ∨ n = 3 ∨ n = 13 :=
sorry

end NUMINAMATH_CALUDE_divisibility_criterion_l3361_336190


namespace NUMINAMATH_CALUDE_matthias_balls_without_holes_l3361_336176

/-- The number of balls without holes in Matthias' collection -/
def balls_without_holes (total_soccer : ℕ) (total_basketball : ℕ) (soccer_with_holes : ℕ) (basketball_with_holes : ℕ) : ℕ :=
  (total_soccer - soccer_with_holes) + (total_basketball - basketball_with_holes)

/-- Theorem stating the total number of balls without holes in Matthias' collection -/
theorem matthias_balls_without_holes :
  balls_without_holes 180 75 125 49 = 81 := by
  sorry

end NUMINAMATH_CALUDE_matthias_balls_without_holes_l3361_336176


namespace NUMINAMATH_CALUDE_worker_completion_time_l3361_336169

/-- Given two workers who can complete a job together in a certain time,
    and one worker's individual completion time, find the other worker's time. -/
theorem worker_completion_time
  (total_time : ℝ)
  (together_time : ℝ)
  (b_time : ℝ)
  (h1 : together_time > 0)
  (h2 : b_time > 0)
  (h3 : total_time > 0)
  (h4 : 1 / together_time = 1 / total_time + 1 / b_time) :
  total_time = 15 :=
sorry

end NUMINAMATH_CALUDE_worker_completion_time_l3361_336169


namespace NUMINAMATH_CALUDE_smallest_product_l3361_336123

def S : Finset Int := {-10, -3, 0, 4, 6}

theorem smallest_product (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int) (hx : x ∈ S) (hy : y ∈ S), x * y ≤ a * b ∧ x * y = -60 :=
by sorry

end NUMINAMATH_CALUDE_smallest_product_l3361_336123


namespace NUMINAMATH_CALUDE_cosine_product_theorem_l3361_336137

theorem cosine_product_theorem :
  (1 + Real.cos (π / 10)) * (1 + Real.cos (3 * π / 10)) *
  (1 + Real.cos (7 * π / 10)) * (1 + Real.cos (9 * π / 10)) =
  (3 - Real.sqrt 5) / 32 := by
sorry

end NUMINAMATH_CALUDE_cosine_product_theorem_l3361_336137


namespace NUMINAMATH_CALUDE_yumi_counting_l3361_336189

def reduce_number (start : ℕ) (amount : ℕ) (times : ℕ) : ℕ :=
  start - amount * times

theorem yumi_counting :
  reduce_number 320 10 4 = 280 := by
  sorry

end NUMINAMATH_CALUDE_yumi_counting_l3361_336189


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3361_336132

theorem linear_equation_solution :
  ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 ∧ x = -30 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3361_336132


namespace NUMINAMATH_CALUDE_steel_to_tin_ratio_l3361_336181

-- Define the masses of the bars
def mass_copper : ℝ := 90
def mass_steel : ℝ := mass_copper + 20

-- Define the total mass of all bars
def total_mass : ℝ := 5100

-- Define the number of bars of each type
def num_bars : ℕ := 20

-- Theorem statement
theorem steel_to_tin_ratio : 
  ∃ (mass_tin : ℝ), 
    mass_tin > 0 ∧ 
    num_bars * (mass_steel + mass_copper + mass_tin) = total_mass ∧ 
    mass_steel / mass_tin = 2 := by
  sorry

end NUMINAMATH_CALUDE_steel_to_tin_ratio_l3361_336181


namespace NUMINAMATH_CALUDE_total_distance_travelled_l3361_336159

theorem total_distance_travelled (distance_by_land distance_by_sea : ℕ) 
  (h1 : distance_by_land = 451)
  (h2 : distance_by_sea = 150) : 
  distance_by_land + distance_by_sea = 601 := by
sorry

end NUMINAMATH_CALUDE_total_distance_travelled_l3361_336159


namespace NUMINAMATH_CALUDE_roof_collapse_time_l3361_336104

/-- The number of days it takes for Bill's roof to collapse under the weight of leaves -/
def days_to_collapse (roof_capacity : ℕ) (leaves_per_day : ℕ) (leaves_per_pound : ℕ) : ℕ :=
  (roof_capacity * leaves_per_pound) / leaves_per_day

/-- Theorem stating that it takes 5000 days for Bill's roof to collapse -/
theorem roof_collapse_time :
  days_to_collapse 500 100 1000 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_roof_collapse_time_l3361_336104


namespace NUMINAMATH_CALUDE_rsa_factorization_l3361_336179

theorem rsa_factorization :
  ∃ (p q : ℕ), 
    p.Prime ∧ 
    q.Prime ∧ 
    p * q = 400000001 ∧ 
    p = 19801 ∧ 
    q = 20201 := by
  sorry

end NUMINAMATH_CALUDE_rsa_factorization_l3361_336179


namespace NUMINAMATH_CALUDE_percentage_added_to_a_l3361_336175

-- Define the ratio of a to b
def ratio_a_b : ℚ := 4 / 5

-- Define the percentage decrease for m
def decrease_percent : ℚ := 80

-- Define the ratio of m to x
def ratio_m_x : ℚ := 1 / 7

-- Define the function to calculate x given a and P
def x_from_a (a : ℚ) (P : ℚ) : ℚ := a * (1 + P / 100)

-- Define the function to calculate m given b
def m_from_b (b : ℚ) : ℚ := b * (1 - decrease_percent / 100)

-- Theorem statement
theorem percentage_added_to_a (a b : ℚ) (P : ℚ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = ratio_a_b) 
  (h4 : m_from_b b / x_from_a a P = ratio_m_x) : P = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_added_to_a_l3361_336175


namespace NUMINAMATH_CALUDE_largest_solution_of_equation_l3361_336198

theorem largest_solution_of_equation (x : ℝ) :
  (((15 * x^2 - 40 * x + 16) / (4 * x - 3)) + 3 * x = 7 * x + 2) →
  x ≤ -14 + Real.sqrt 218 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_of_equation_l3361_336198


namespace NUMINAMATH_CALUDE_mike_oil_changes_l3361_336167

/-- Represents the time in minutes for various car maintenance tasks and Mike's work schedule --/
structure CarMaintenance where
  wash_time : Nat  -- Time to wash a car in minutes
  oil_change_time : Nat  -- Time to change oil in minutes
  tire_change_time : Nat  -- Time to change a set of tires in minutes
  total_work_time : Nat  -- Total work time in minutes
  cars_washed : Nat  -- Number of cars washed
  tire_sets_changed : Nat  -- Number of tire sets changed

/-- Calculates the number of cars Mike changed oil on given the car maintenance data --/
def calculate_oil_changes (data : CarMaintenance) : Nat :=
  let total_wash_time := data.wash_time * data.cars_washed
  let total_tire_change_time := data.tire_change_time * data.tire_sets_changed
  let remaining_time := data.total_work_time - total_wash_time - total_tire_change_time
  remaining_time / data.oil_change_time

/-- Theorem stating that given the problem conditions, Mike changed oil on 6 cars --/
theorem mike_oil_changes (data : CarMaintenance) 
  (h1 : data.wash_time = 10)
  (h2 : data.oil_change_time = 15)
  (h3 : data.tire_change_time = 30)
  (h4 : data.total_work_time = 4 * 60)
  (h5 : data.cars_washed = 9)
  (h6 : data.tire_sets_changed = 2) :
  calculate_oil_changes data = 6 := by
  sorry

end NUMINAMATH_CALUDE_mike_oil_changes_l3361_336167
