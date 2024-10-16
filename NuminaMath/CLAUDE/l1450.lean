import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1450_145044

theorem quadratic_equation_coefficients :
  let original_eq : ℝ → Prop := λ x => 3 * x^2 - 1 = 6 * x
  let general_form : ℝ → ℝ → ℝ → ℝ → Prop := λ a b c x => a * x^2 + b * x + c = 0
  ∃ (a b c : ℝ), (∀ x, original_eq x ↔ general_form a b c x) ∧ a = 3 ∧ b = -6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l1450_145044


namespace NUMINAMATH_CALUDE_play_attendance_l1450_145076

theorem play_attendance (total_people : ℕ) (adult_price child_price : ℚ) (total_receipts : ℚ) 
  (h1 : total_people = 610)
  (h2 : adult_price = 2)
  (h3 : child_price = 1)
  (h4 : total_receipts = 960) :
  ∃ (adults children : ℕ), 
    adults + children = total_people ∧ 
    adult_price * adults + child_price * children = total_receipts ∧
    children = 260 := by
  sorry

end NUMINAMATH_CALUDE_play_attendance_l1450_145076


namespace NUMINAMATH_CALUDE_johns_paintball_expense_l1450_145024

/-- The amount John spends on paintballs per month -/
def monthly_paintball_expense (plays_per_month : ℕ) (boxes_per_play : ℕ) (cost_per_box : ℕ) : ℕ :=
  plays_per_month * boxes_per_play * cost_per_box

/-- Theorem stating John's monthly paintball expense -/
theorem johns_paintball_expense :
  monthly_paintball_expense 3 3 25 = 225 := by
  sorry

end NUMINAMATH_CALUDE_johns_paintball_expense_l1450_145024


namespace NUMINAMATH_CALUDE_scheme1_higher_sale_price_l1450_145088

def original_price : ℝ := 15000

def scheme1_price (p : ℝ) : ℝ :=
  p * (1 - 0.25) * (1 - 0.15) * (1 - 0.05) * (1 + 0.30)

def scheme2_price (p : ℝ) : ℝ :=
  p * (1 - 0.40) * (1 + 0.30)

theorem scheme1_higher_sale_price :
  scheme1_price original_price > scheme2_price original_price :=
by sorry

end NUMINAMATH_CALUDE_scheme1_higher_sale_price_l1450_145088


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_range_l1450_145013

/-- An ellipse with equation x²/4 + y²/t = 1 -/
structure Ellipse (t : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2/4 + y^2/t = 1

/-- The distance from a point on the ellipse to one of its foci -/
noncomputable def distance_to_focus (t : ℝ) (e : Ellipse t) : ℝ :=
  sorry  -- Definition omitted as it's not directly given in the problem

/-- The theorem stating the range of t for which the distance to a focus is always greater than 1 -/
theorem ellipse_focus_distance_range :
  ∀ t : ℝ, (∀ e : Ellipse t, distance_to_focus t e > 1) →
    t ∈ Set.union (Set.Ioo 3 4) (Set.Ioo 4 (25/4)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_range_l1450_145013


namespace NUMINAMATH_CALUDE_currency_denominations_l1450_145046

/-- The number of different denominations that can be formed with a given number of coins/bills of three types -/
def total_denominations (fifty_cent : ℕ) (five_yuan : ℕ) (hundred_yuan : ℕ) : ℕ :=
  let single_denom := fifty_cent + five_yuan + hundred_yuan
  let double_denom := fifty_cent * five_yuan + five_yuan * hundred_yuan + hundred_yuan * fifty_cent
  let triple_denom := fifty_cent * five_yuan * hundred_yuan
  single_denom + double_denom + triple_denom

/-- Theorem stating that the total number of denominations with 3 fifty-cent coins, 
    6 five-yuan bills, and 4 one-hundred-yuan bills is 139 -/
theorem currency_denominations : 
  total_denominations 3 6 4 = 139 := by
  sorry

end NUMINAMATH_CALUDE_currency_denominations_l1450_145046


namespace NUMINAMATH_CALUDE_investment_problem_l1450_145098

/-- The solution to Susie Q's investment problem -/
theorem investment_problem (total_investment : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) 
  (years : ℕ) (final_amount : ℝ) (investment1 : ℝ) :
  total_investment = 1500 →
  interest_rate1 = 0.04 →
  interest_rate2 = 0.06 →
  years = 2 →
  final_amount = 1700.02 →
  investment1 * (1 + interest_rate1) ^ years + 
    (total_investment - investment1) * (1 + interest_rate2) ^ years = final_amount →
  investment1 = 348.095 := by
    sorry

end NUMINAMATH_CALUDE_investment_problem_l1450_145098


namespace NUMINAMATH_CALUDE_fourth_year_students_without_glasses_l1450_145079

theorem fourth_year_students_without_glasses 
  (total_students : ℕ) 
  (fourth_year_students : ℕ) 
  (students_with_glasses : ℕ) 
  (students_without_glasses : ℕ) :
  total_students = 8 * fourth_year_students - 32 →
  students_with_glasses = students_without_glasses + 10 →
  total_students = 1152 →
  fourth_year_students = students_with_glasses + students_without_glasses →
  students_without_glasses = 69 :=
by sorry

end NUMINAMATH_CALUDE_fourth_year_students_without_glasses_l1450_145079


namespace NUMINAMATH_CALUDE_handball_final_score_l1450_145057

/-- Represents the score of a handball match -/
structure Score where
  home : ℕ
  visitors : ℕ

/-- Calculates the final score given the initial score and goals scored in the second half -/
def finalScore (initial : Score) (visitorGoals : ℕ) : Score :=
  { home := initial.home + 2 * visitorGoals,
    visitors := initial.visitors + visitorGoals }

/-- Theorem stating the final score of the handball match -/
theorem handball_final_score :
  ∀ (initial : Score) (visitorGoals : ℕ),
    initial.home = 9 →
    initial.visitors = 14 →
    let final := finalScore initial visitorGoals
    (final.home = final.visitors + 1) →
    final.home = 21 ∧ final.visitors = 20 := by
  sorry

#check handball_final_score

end NUMINAMATH_CALUDE_handball_final_score_l1450_145057


namespace NUMINAMATH_CALUDE_mona_repeat_players_group2_l1450_145041

/-- Represents the game scenario for Mona --/
structure GameScenario where
  group_size : ℕ
  num_groups : ℕ
  unique_players : ℕ
  repeat_players_group1 : ℕ

/-- Calculates the number of repeat players in the second group --/
def repeat_players_group2 (scenario : GameScenario) : ℕ :=
  (scenario.group_size - 1) * scenario.num_groups - scenario.unique_players - scenario.repeat_players_group1

/-- Theorem stating that the number of repeat players in the second group is 1 --/
theorem mona_repeat_players_group2 (scenario : GameScenario) 
  (h1 : scenario.group_size = 5)
  (h2 : scenario.num_groups = 9)
  (h3 : scenario.unique_players = 33)
  (h4 : scenario.repeat_players_group1 = 2) :
  repeat_players_group2 scenario = 1 := by
  sorry

#eval repeat_players_group2 ⟨5, 9, 33, 2⟩

end NUMINAMATH_CALUDE_mona_repeat_players_group2_l1450_145041


namespace NUMINAMATH_CALUDE_flooring_cost_is_14375_l1450_145015

/-- Represents the dimensions and cost of a rectangular room -/
structure RectRoom where
  length : Float
  width : Float
  cost_per_sqm : Float

/-- Represents the dimensions and cost of an L-shaped room -/
structure LShapeRoom where
  rect1_length : Float
  rect1_width : Float
  rect2_length : Float
  rect2_width : Float
  cost_per_sqm : Float

/-- Represents the dimensions and cost of a triangular room -/
structure TriRoom where
  base : Float
  height : Float
  cost_per_sqm : Float

/-- Calculates the total cost of flooring for all rooms -/
def total_flooring_cost (room1 : RectRoom) (room2 : LShapeRoom) (room3 : TriRoom) : Float :=
  (room1.length * room1.width * room1.cost_per_sqm) +
  ((room2.rect1_length * room2.rect1_width + room2.rect2_length * room2.rect2_width) * room2.cost_per_sqm) +
  (0.5 * room3.base * room3.height * room3.cost_per_sqm)

/-- Theorem stating that the total flooring cost for the given rooms is $14,375 -/
theorem flooring_cost_is_14375 
  (room1 : RectRoom)
  (room2 : LShapeRoom)
  (room3 : TriRoom)
  (h1 : room1 = { length := 5.5, width := 3.75, cost_per_sqm := 400 })
  (h2 : room2 = { rect1_length := 4, rect1_width := 2.5, rect2_length := 2, rect2_width := 1.5, cost_per_sqm := 350 })
  (h3 : room3 = { base := 3.5, height := 2, cost_per_sqm := 450 }) :
  total_flooring_cost room1 room2 room3 = 14375 := by
  sorry

end NUMINAMATH_CALUDE_flooring_cost_is_14375_l1450_145015


namespace NUMINAMATH_CALUDE_ned_weekly_earnings_l1450_145005

/-- Calculates weekly earnings from selling left-handed mice --/
def weekly_earnings (normal_price : ℝ) (price_increase_percent : ℝ) 
                    (daily_sales : ℕ) (open_days : ℕ) : ℝ :=
  let left_handed_price := normal_price * (1 + price_increase_percent)
  let daily_earnings := left_handed_price * daily_sales
  daily_earnings * open_days

/-- Theorem stating Ned's weekly earnings --/
theorem ned_weekly_earnings :
  weekly_earnings 120 0.3 25 4 = 15600 := by
  sorry

#eval weekly_earnings 120 0.3 25 4

end NUMINAMATH_CALUDE_ned_weekly_earnings_l1450_145005


namespace NUMINAMATH_CALUDE_river_width_l1450_145083

/-- The width of a river given its depth, flow rate, and volume of water per minute. -/
theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) :
  depth = 3 →
  flow_rate_kmph = 2 →
  volume_per_minute = 3200 →
  ∃ (width : ℝ), abs (width - 32) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_river_width_l1450_145083


namespace NUMINAMATH_CALUDE_integral_equals_zero_l1450_145099

theorem integral_equals_zero : 
  ∫ x in (0 : ℝ)..1, (4 * Real.sqrt (1 - x) - Real.sqrt (3 * x + 1)) / 
    ((Real.sqrt (3 * x + 1) + 4 * Real.sqrt (1 - x)) * (3 * x + 1)^2) = 0 := by sorry

end NUMINAMATH_CALUDE_integral_equals_zero_l1450_145099


namespace NUMINAMATH_CALUDE_optimal_fare_and_passenger_change_l1450_145016

/-- Demand function -/
def demand (p : ℝ) : ℝ := 4200 - 100 * p

/-- Train fare -/
def train_fare : ℝ := 4

/-- Train capacity -/
def train_capacity : ℝ := 800

/-- Bus company cost function -/
def bus_cost (y : ℝ) : ℝ := 10 * y + 225

/-- Optimal bus fare -/
def optimal_bus_fare : ℝ := 22

/-- Change in total passengers if train service closes -/
def passenger_change : ℝ := -400

/-- Theorem stating the optimal bus fare and passenger change -/
theorem optimal_fare_and_passenger_change :
  (∃ (p : ℝ), p = optimal_bus_fare ∧
    ∀ (p' : ℝ), p' > train_fare →
      p * (demand p - train_capacity) - bus_cost (demand p - train_capacity) ≥
      p' * (demand p' - train_capacity) - bus_cost (demand p' - train_capacity)) ∧
  (demand (26) - (demand optimal_bus_fare - train_capacity + train_capacity) = passenger_change) :=
sorry

end NUMINAMATH_CALUDE_optimal_fare_and_passenger_change_l1450_145016


namespace NUMINAMATH_CALUDE_finite_good_not_divisible_by_k_l1450_145031

/-- The number of divisors of an integer n -/
def τ (n : ℕ) : ℕ := sorry

/-- An integer n is "good" if for all m < n, we have τ(m) < τ(n) -/
def is_good (n : ℕ) : Prop :=
  ∀ m < n, τ m < τ n

/-- The set of good integers not divisible by k is finite -/
theorem finite_good_not_divisible_by_k (k : ℕ) (h : k ≥ 1) :
  {n : ℕ | is_good n ∧ ¬k ∣ n}.Finite :=
sorry

end NUMINAMATH_CALUDE_finite_good_not_divisible_by_k_l1450_145031


namespace NUMINAMATH_CALUDE_telescope_visual_range_l1450_145056

/-- 
Given a telescope that increases the visual range by 50% to reach 150 kilometers,
prove that the initial visual range without the telescope was 100 kilometers.
-/
theorem telescope_visual_range 
  (increased_range : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : increased_range = 150) 
  (h2 : increase_percentage = 0.5) : 
  increased_range / (1 + increase_percentage) = 100 := by
  sorry

end NUMINAMATH_CALUDE_telescope_visual_range_l1450_145056


namespace NUMINAMATH_CALUDE_peanut_difference_l1450_145036

theorem peanut_difference (jose_peanuts kenya_peanuts : ℕ) 
  (h1 : jose_peanuts = 85)
  (h2 : kenya_peanuts = 133)
  (h3 : kenya_peanuts > jose_peanuts) :
  kenya_peanuts - jose_peanuts = 48 := by
  sorry

end NUMINAMATH_CALUDE_peanut_difference_l1450_145036


namespace NUMINAMATH_CALUDE_lilibeth_baskets_l1450_145023

/-- The number of strawberries each basket holds -/
def strawberries_per_basket : ℕ := 50

/-- The total number of strawberries picked by Lilibeth and her friends -/
def total_strawberries : ℕ := 1200

/-- The number of people picking strawberries (Lilibeth and her three friends) -/
def number_of_pickers : ℕ := 4

/-- The number of baskets Lilibeth filled -/
def baskets_filled : ℕ := 6

theorem lilibeth_baskets :
  strawberries_per_basket * number_of_pickers * baskets_filled = total_strawberries :=
by sorry

end NUMINAMATH_CALUDE_lilibeth_baskets_l1450_145023


namespace NUMINAMATH_CALUDE_prime_divisor_count_l1450_145070

theorem prime_divisor_count (p : ℕ) (hp : Prime p) 
  (h : ∃ k : ℤ, (28^p - 1 : ℤ) = k * (2*p^2 + 2*p + 1)) : 
  Prime (2*p^2 + 2*p + 1) := by
sorry

end NUMINAMATH_CALUDE_prime_divisor_count_l1450_145070


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1450_145093

theorem x_minus_y_value (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 17) 
  (eq2 : x + 3 * y = 5) : 
  x - y = 73 / 13 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1450_145093


namespace NUMINAMATH_CALUDE_find_x_l1450_145007

theorem find_x : ∃ X : ℤ, X - (5 - (6 + 2 * (7 - 8 - 5))) = 89 ∧ X = 100 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1450_145007


namespace NUMINAMATH_CALUDE_line_through_points_l1450_145068

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem line_through_points (a b : V) (h : a ≠ b) :
  ∃ t : ℝ, (1/2 : ℝ) • a + (1/2 : ℝ) • b = a + t • (b - a) :=
sorry

end NUMINAMATH_CALUDE_line_through_points_l1450_145068


namespace NUMINAMATH_CALUDE_largest_stamps_per_page_l1450_145011

theorem largest_stamps_per_page (a b c : ℕ) 
  (ha : a = 924) (hb : b = 1386) (hc : c = 1848) : 
  Nat.gcd a (Nat.gcd b c) = 462 := by
  sorry

end NUMINAMATH_CALUDE_largest_stamps_per_page_l1450_145011


namespace NUMINAMATH_CALUDE_shirt_cost_calculation_l1450_145096

/-- The amount Sandy spent on clothes -/
def total_spent : ℚ := 33.56

/-- The amount Sandy spent on shorts -/
def shorts_cost : ℚ := 13.99

/-- The amount Sandy spent on a jacket -/
def jacket_cost : ℚ := 7.43

/-- The amount Sandy spent on a shirt -/
def shirt_cost : ℚ := total_spent - shorts_cost - jacket_cost

theorem shirt_cost_calculation :
  shirt_cost = 12.14 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_calculation_l1450_145096


namespace NUMINAMATH_CALUDE_friend_payment_amount_l1450_145089

/-- The cost per item for each food item --/
def hamburger_cost : ℚ := 3
def fries_cost : ℚ := 6/5  -- 1.20 as a rational number
def soda_cost : ℚ := 1/2
def spaghetti_cost : ℚ := 27/10
def milkshake_cost : ℚ := 5/2
def nuggets_cost : ℚ := 7/2

/-- The number of each item ordered --/
def hamburger_count : ℕ := 5
def fries_count : ℕ := 4
def soda_count : ℕ := 5
def spaghetti_count : ℕ := 1
def milkshake_count : ℕ := 3
def nuggets_count : ℕ := 2

/-- The discount percentage as a rational number --/
def discount_percent : ℚ := 1/10

/-- The percentage of the bill paid by the birthday friend --/
def birthday_friend_percent : ℚ := 3/10

/-- The number of friends splitting the remaining bill --/
def remaining_friends : ℕ := 4

/-- The theorem stating that each remaining friend will pay $6.22 --/
theorem friend_payment_amount : 
  let total_bill := hamburger_cost * hamburger_count + 
                    fries_cost * fries_count +
                    soda_cost * soda_count +
                    spaghetti_cost * spaghetti_count +
                    milkshake_cost * milkshake_count +
                    nuggets_cost * nuggets_count
  let discounted_bill := total_bill * (1 - discount_percent)
  let remaining_bill := discounted_bill * (1 - birthday_friend_percent)
  remaining_bill / remaining_friends = 311/50  -- 6.22 as a rational number
  := by sorry

end NUMINAMATH_CALUDE_friend_payment_amount_l1450_145089


namespace NUMINAMATH_CALUDE_problem_solution_l1450_145077

theorem problem_solution : 
  ((-1: ℤ) ^ 10 * 2 + (-2) ^ 3 / 4 = 0) ∧ 
  ((-24: ℚ) * (5/6 - 4/3 + 3/8) = 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1450_145077


namespace NUMINAMATH_CALUDE_debby_candy_eaten_l1450_145047

/-- Given that Debby initially had 12 pieces of candy and ended up with 3 pieces,
    prove that she ate 9 pieces. -/
theorem debby_candy_eaten (initial : ℕ) (final : ℕ) (eaten : ℕ) 
    (h1 : initial = 12) 
    (h2 : final = 3) 
    (h3 : initial = final + eaten) : eaten = 9 := by
  sorry

end NUMINAMATH_CALUDE_debby_candy_eaten_l1450_145047


namespace NUMINAMATH_CALUDE_cos_one_third_solutions_l1450_145027

theorem cos_one_third_solutions (α : Real) (h1 : α ∈ Set.Icc 0 (2 * Real.pi)) (h2 : Real.cos α = 1/3) :
  α = Real.arccos (1/3) ∨ α = 2 * Real.pi - Real.arccos (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cos_one_third_solutions_l1450_145027


namespace NUMINAMATH_CALUDE_ferry_tourist_count_l1450_145010

/-- The number of trips the ferry makes -/
def num_trips : ℕ := 7

/-- The number of tourists on the first trip -/
def initial_tourists : ℕ := 100

/-- The decrease in tourists per trip -/
def tourist_decrease : ℕ := 2

/-- The total number of tourists transported -/
def total_tourists : ℕ := 
  (num_trips * (2 * initial_tourists - (num_trips - 1) * tourist_decrease)) / 2

theorem ferry_tourist_count : total_tourists = 658 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourist_count_l1450_145010


namespace NUMINAMATH_CALUDE_fair_expenses_correct_l1450_145094

/-- Calculates the total amount spent at a fair given the following conditions:
  - Entrance fee for persons under 18: $5
  - Entrance fee for persons 18 and older: 20% more than $5
  - Cost per ride: $0.50
  - One adult (Joe) and two children (6-year-old twin brothers)
  - Each person took 3 rides
-/
def fairExpenses (childEntranceFee adultEntranceFeeIncrease ridePrice : ℚ) 
                 (numChildren numAdults numRidesPerPerson : ℕ) : ℚ :=
  let childrenEntranceFees := childEntranceFee * numChildren
  let adultEntranceFee := childEntranceFee * (1 + adultEntranceFeeIncrease)
  let adultEntranceFees := adultEntranceFee * numAdults
  let totalEntranceFees := childrenEntranceFees + adultEntranceFees
  let totalRideCost := ridePrice * numRidesPerPerson * (numChildren + numAdults)
  totalEntranceFees + totalRideCost

/-- Theorem stating that the total amount spent at the fair under the given conditions is $20.50 -/
theorem fair_expenses_correct : 
  fairExpenses 5 0.2 0.5 2 1 3 = 41/2 := by
  sorry

end NUMINAMATH_CALUDE_fair_expenses_correct_l1450_145094


namespace NUMINAMATH_CALUDE_cos_theta_value_l1450_145073

theorem cos_theta_value (x y : ℝ) (θ : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (hθ : θ ∈ Set.Ioo (π/4) (π/2))
  (h1 : y / Real.sin θ = x / Real.cos θ)
  (h2 : 10 / (x^2 + y^2) = 3 / (x * y)) :
  Real.cos θ = Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_theta_value_l1450_145073


namespace NUMINAMATH_CALUDE_conference_handshakes_l1450_145053

/-- Represents a conference with a fixed number of attendees and handshakes per person -/
structure Conference where
  attendees : ℕ
  handshakes_per_person : ℕ

/-- Calculates the minimum number of unique handshakes in a conference -/
def min_handshakes (conf : Conference) : ℕ :=
  conf.attendees * conf.handshakes_per_person / 2

/-- Theorem stating that in a conference of 30 people where each person shakes hands
    with exactly 3 others, the minimum number of unique handshakes is 45 -/
theorem conference_handshakes :
  let conf : Conference := { attendees := 30, handshakes_per_person := 3 }
  min_handshakes conf = 45 := by
  sorry


end NUMINAMATH_CALUDE_conference_handshakes_l1450_145053


namespace NUMINAMATH_CALUDE_lesser_fraction_l1450_145038

theorem lesser_fraction (x y : ℚ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : x + y = 13/14) 
  (h4 : x * y = 1/8) : 
  min x y = 163/625 := by
sorry

end NUMINAMATH_CALUDE_lesser_fraction_l1450_145038


namespace NUMINAMATH_CALUDE_rectangle_area_l1450_145090

theorem rectangle_area (l w : ℝ) (h1 : l = 15) (h2 : (2 * l + 2 * w) / w = 5) :
  l * w = 150 :=
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1450_145090


namespace NUMINAMATH_CALUDE_angle_calculations_l1450_145069

/-- Given points P and Q on the terminal sides of angles α and β, 
    prove the values of sin(α - β) and cos(α + β) -/
theorem angle_calculations (P Q : ℝ × ℝ) (α β : ℝ) : 
  P = (-3, 4) → Q = (-1, -2) → 
  (P.1 = (Real.cos α) * Real.sqrt (P.1^2 + P.2^2)) →
  (P.2 = (Real.sin α) * Real.sqrt (P.1^2 + P.2^2)) →
  (Q.1 = (Real.cos β) * Real.sqrt (Q.1^2 + Q.2^2)) →
  (Q.2 = (Real.sin β) * Real.sqrt (Q.1^2 + Q.2^2)) →
  Real.sin (α - β) = -2 * Real.sqrt 5 / 5 ∧ 
  Real.cos (α + β) = 11 * Real.sqrt 5 / 25 := by
  sorry


end NUMINAMATH_CALUDE_angle_calculations_l1450_145069


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l1450_145002

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

theorem product_trailing_zeros :
  trailing_zeros 100 = 24 :=
sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l1450_145002


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1450_145008

theorem election_winner_percentage (total_votes winner_majority : ℕ) 
  (h_total : total_votes = 500) 
  (h_majority : winner_majority = 200) : 
  (((total_votes + winner_majority) / 2) / total_votes : ℚ) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1450_145008


namespace NUMINAMATH_CALUDE_some_number_value_l1450_145054

theorem some_number_value (x : ℝ) (some_number : ℝ) 
  (h1 : x = 5)
  (h2 : (x / some_number) + 3 = 4) : 
  some_number = 5 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l1450_145054


namespace NUMINAMATH_CALUDE_train_length_l1450_145052

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 24 → speed * time * (1000 / 3600) = 420 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1450_145052


namespace NUMINAMATH_CALUDE_abacus_problem_l1450_145021

def is_valid_abacus_division (upper lower : ℕ) : Prop :=
  upper ≥ 100 ∧ upper < 1000 ∧ lower ≥ 100 ∧ lower < 1000 ∧
  upper + lower = 1110 ∧
  (∃ k : ℕ, upper = k * lower) ∧
  (∃ a b c : ℕ, upper = 100 * a + 10 * b + c ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c)

theorem abacus_problem : ∃ upper lower : ℕ, is_valid_abacus_division upper lower ∧ upper = 925 := by
  sorry

end NUMINAMATH_CALUDE_abacus_problem_l1450_145021


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l1450_145050

/-- 
Given a line passing through points (1, 3) and (3, 7),
prove that the sum of its slope (m) and y-intercept (b) is equal to 3.
-/
theorem line_slope_intercept_sum (m b : ℝ) : 
  (3 = m * 1 + b) → (7 = m * 3 + b) → m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l1450_145050


namespace NUMINAMATH_CALUDE_circles_intersect_implies_equilateral_l1450_145091

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- Predicate that checks if any two circles intersect -/
def circlesIntersect (t : Triangle) : Prop :=
  t.c/2 ≤ t.a/4 + t.b/4 ∧ t.a/2 ≤ t.b/4 + t.c/4 ∧ t.b/2 ≤ t.c/4 + t.a/4

/-- Predicate that checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- Theorem stating that if circles drawn around midpoints of a triangle's sides
    with radii 1/4 of the side lengths intersect, then the triangle is equilateral -/
theorem circles_intersect_implies_equilateral (t : Triangle) :
  circlesIntersect t → isEquilateral t :=
by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_implies_equilateral_l1450_145091


namespace NUMINAMATH_CALUDE_f_of_x_plus_one_l1450_145097

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem f_of_x_plus_one (x : ℝ) : f (x + 1) = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_of_x_plus_one_l1450_145097


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1450_145032

theorem polynomial_expansion (x : ℝ) : 
  (x - 3) * (x + 5) * (x^2 + 9) = x^4 + 2*x^3 - 6*x^2 + 18*x - 135 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1450_145032


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1450_145074

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ = 16 and a₉ = 80,
    prove that a₆ = 48. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a)
    (h_3 : a 3 = 16)
    (h_9 : a 9 = 80) : 
  a 6 = 48 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1450_145074


namespace NUMINAMATH_CALUDE_expression_equals_zero_l1450_145001

theorem expression_equals_zero :
  (-1 : ℝ) ^ 2022 + |-2| - (1/2 : ℝ) ^ 0 - 2 * Real.tan (π/4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l1450_145001


namespace NUMINAMATH_CALUDE_sum_of_squares_l1450_145064

theorem sum_of_squares (a b : ℝ) (h1 : a - b = 6) (h2 : a * b = 7) :
  a^2 + b^2 = 50 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1450_145064


namespace NUMINAMATH_CALUDE_matrix_power_four_l1450_145080

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l1450_145080


namespace NUMINAMATH_CALUDE_line_passes_through_points_l1450_145078

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point lies on a given line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The two given points -/
def p1 : Point := ⟨-1, 1⟩
def p2 : Point := ⟨3, 9⟩

/-- The line we want to prove passes through the given points -/
def line : Line := ⟨2, -1, 3⟩

/-- Theorem stating that the given line passes through both points -/
theorem line_passes_through_points : 
  p1.liesOn line ∧ p2.liesOn line := by sorry

end NUMINAMATH_CALUDE_line_passes_through_points_l1450_145078


namespace NUMINAMATH_CALUDE_opposite_sign_pair_l1450_145081

theorem opposite_sign_pair : ∃! (x : ℝ), (x > 0 ∧ x * x = 7) ∧ 
  (∀ a b : ℝ, (a = 131 ∧ b = 1 - 31) ∨ 
              (a = x ∧ b = -x) ∨ 
              (a = 1/3 ∧ b = Real.sqrt (1/9)) ∨ 
              (a = 5^2 ∧ b = (-5)^2) →
   (a + b = 0 ∧ a * b < 0) ↔ (a = x ∧ b = -x)) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sign_pair_l1450_145081


namespace NUMINAMATH_CALUDE_absolute_value_equation_l1450_145018

theorem absolute_value_equation (a b c : ℝ) :
  (∀ x y z : ℝ, |a*x + b*y + c*z| + |b*x + c*y + a*z| + |c*x + a*y + b*z| = |x| + |y| + |z|) →
  ((a = 0 ∧ b = 0 ∧ (c = 1 ∨ c = -1)) ∨
   (a = 0 ∧ c = 0 ∧ (b = 1 ∨ b = -1)) ∨
   (b = 0 ∧ c = 0 ∧ (a = 1 ∨ a = -1))) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l1450_145018


namespace NUMINAMATH_CALUDE_minimal_value_of_f_l1450_145062

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem minimal_value_of_f :
  ∃ (x_min : ℝ), f x_min = Real.exp (-1) ∧ ∀ (x : ℝ), f x ≥ Real.exp (-1) :=
sorry

end NUMINAMATH_CALUDE_minimal_value_of_f_l1450_145062


namespace NUMINAMATH_CALUDE_solution_and_parabola_equivalence_l1450_145009

-- Define the set of solutions for x - 3 > 0
def solution_set : Set ℝ := {x | x - 3 > 0}

-- Define the set of points on the parabola y = x^2 - 1
def parabola_points : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 - 1}

theorem solution_and_parabola_equivalence :
  (solution_set = {x : ℝ | x > 3}) ∧
  (parabola_points = {p : ℝ × ℝ | p.2 = p.1^2 - 1}) := by
  sorry

end NUMINAMATH_CALUDE_solution_and_parabola_equivalence_l1450_145009


namespace NUMINAMATH_CALUDE_absolute_value_zero_l1450_145055

theorem absolute_value_zero (y : ℚ) : |2 * y - 3| = 0 ↔ y = 3/2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_zero_l1450_145055


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l1450_145035

theorem base_10_to_base_7 : 
  ∃ (a b c d : ℕ), 
    875 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 3 ∧ c = 6 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l1450_145035


namespace NUMINAMATH_CALUDE_min_value_theorem_l1450_145092

theorem min_value_theorem (x y : ℝ) (h : x^2 + y^2 + x*y = 315) :
  x^2 + y^2 - x*y ≥ 105 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1450_145092


namespace NUMINAMATH_CALUDE_city_households_l1450_145029

/-- The number of deer that entered the city -/
def num_deer : ℕ := 100

/-- The number of households in the city -/
def num_households : ℕ := 75

theorem city_households : 
  (num_households < num_deer) ∧ 
  (4 * num_households = 3 * num_deer) := by
  sorry

end NUMINAMATH_CALUDE_city_households_l1450_145029


namespace NUMINAMATH_CALUDE_deposit_percentage_l1450_145033

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) : 
  deposit = 130 → remaining = 1170 → (deposit / (deposit + remaining)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_deposit_percentage_l1450_145033


namespace NUMINAMATH_CALUDE_probability_A_makes_basket_on_kth_shot_l1450_145066

/-- The probability that player A takes k shots to make the basket -/
def P (k : ℕ) : ℝ :=
  (0.24 ^ (k - 1)) * 0.4

/-- Theorem stating the probability formula for player A making a basket on the k-th shot -/
theorem probability_A_makes_basket_on_kth_shot (k : ℕ) :
  P k = (0.24 ^ (k - 1)) * 0.4 :=
by
  sorry

#check probability_A_makes_basket_on_kth_shot

end NUMINAMATH_CALUDE_probability_A_makes_basket_on_kth_shot_l1450_145066


namespace NUMINAMATH_CALUDE_power_function_monotonic_increase_l1450_145072

-- Define the power function that passes through (2, 4)
def f (x : ℝ) := x^2

-- Theorem statement
theorem power_function_monotonic_increase :
  (∀ x : ℝ, f x = x^2) →
  f 2 = 4 →
  ∀ x y : ℝ, 0 < x → x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_power_function_monotonic_increase_l1450_145072


namespace NUMINAMATH_CALUDE_equation_sum_zero_l1450_145087

theorem equation_sum_zero (a b c : ℝ) 
  (h1 : a + b / c = 1) 
  (h2 : b + c / a = 1) 
  (h3 : c + a / b = 1) : 
  a * b + b * c + c * a = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_sum_zero_l1450_145087


namespace NUMINAMATH_CALUDE_planes_parallel_criterion_l1450_145075

/-- Two lines are parallel -/
def parallel_lines (m n : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two planes are parallel -/
def parallel_planes (α β : Plane) : Prop := sorry

/-- A line is contained in a plane -/
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem planes_parallel_criterion 
  (m n : Line) (α β : Plane) 
  (h1 : ¬ (line_in_plane m α ∧ line_in_plane m β))
  (h2 : ¬ (line_in_plane n α ∧ line_in_plane n β))
  (h3 : parallel_lines m n)
  (h4 : line_perp_plane m α)
  (h5 : line_perp_plane n β) : 
  parallel_planes α β := by sorry

end NUMINAMATH_CALUDE_planes_parallel_criterion_l1450_145075


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1450_145039

theorem cubic_roots_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 ∧ 
  q^3 - 8*q^2 + 10*q - 3 = 0 ∧ 
  r^3 - 8*r^2 + 10*r - 3 = 0 → 
  (p / (q*r + 2)) + (q / (p*r + 2)) + (r / (p*q + 2)) = 367/183 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1450_145039


namespace NUMINAMATH_CALUDE_fruit_bag_capacity_l1450_145019

theorem fruit_bag_capacity
  (apple_weight : ℕ)
  (orange_weight : ℕ)
  (total_apple_weight : ℕ)
  (num_bags : ℕ)
  (h1 : apple_weight = 4)
  (h2 : orange_weight = 3)
  (h3 : total_apple_weight = 84)
  (h4 : num_bags = 3) :
  let num_apples : ℕ := total_apple_weight / apple_weight
  let num_oranges : ℕ := num_apples
  let total_orange_weight : ℕ := num_oranges * orange_weight
  let total_fruit_weight : ℕ := total_apple_weight + total_orange_weight
  let bag_capacity : ℕ := total_fruit_weight / num_bags
  bag_capacity = 49 := by sorry

end NUMINAMATH_CALUDE_fruit_bag_capacity_l1450_145019


namespace NUMINAMATH_CALUDE_equation_one_solutions_l1450_145059

theorem equation_one_solutions :
  let f : ℝ → ℝ := λ x => x^2 - 2*x - 1
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l1450_145059


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1450_145048

theorem max_value_of_expression (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 2) : 
  (Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2-a)^2 * (2-b)^2 * (2-c)^2)) ≤ 16 ∧ 
  ∃ (a' b' c' : ℝ), 0 ≤ a' ∧ a' ≤ 2 ∧ 
                    0 ≤ b' ∧ b' ≤ 2 ∧ 
                    0 ≤ c' ∧ c' ≤ 2 ∧ 
                    Real.sqrt (a'^2 * b'^2 * c'^2) + Real.sqrt ((2-a')^2 * (2-b')^2 * (2-c')^2) = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1450_145048


namespace NUMINAMATH_CALUDE_absolute_value_sum_zero_implies_product_l1450_145028

theorem absolute_value_sum_zero_implies_product (x y : ℝ) :
  |x - 1| + |y + 3| = 0 → (x + 1) * (y - 3) = -12 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_zero_implies_product_l1450_145028


namespace NUMINAMATH_CALUDE_other_integer_is_30_l1450_145034

theorem other_integer_is_30 (a b : ℤ) (h1 : 3 * a + 2 * b = 135) (h2 : a = 25 ∨ b = 25) : 
  (a ≠ 25 → b = 30) ∧ (b ≠ 25 → a = 30) := by
  sorry

end NUMINAMATH_CALUDE_other_integer_is_30_l1450_145034


namespace NUMINAMATH_CALUDE_ellipse_foci_y_axis_l1450_145004

theorem ellipse_foci_y_axis (k : ℝ) :
  (∀ x y : ℝ, x^2 / (9 - k) + y^2 / (k - 4) = 1) →
  (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ b^2 = a^2 + c^2 ∧
    ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / (9 - k) + y^2 / (k - 4) = 1) →
  13/2 < k ∧ k < 9 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_y_axis_l1450_145004


namespace NUMINAMATH_CALUDE_tire_price_proof_l1450_145037

theorem tire_price_proof : 
  let fourth_tire_discount : ℝ := 0.75
  let total_cost : ℝ := 270
  let regular_price : ℝ := 72
  (3 * regular_price + fourth_tire_discount * regular_price = total_cost) →
  regular_price = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_tire_price_proof_l1450_145037


namespace NUMINAMATH_CALUDE_sum_of_x_coordinates_is_seven_l1450_145022

-- Define the system of equations
def equation1 (x : ℝ) : ℝ := |x^2 - 8*x + 15|
def equation2 (x : ℝ) : ℝ := 8 - x

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | equation1 x = equation2 x}

-- State the theorem
theorem sum_of_x_coordinates_is_seven :
  ∃ (x₁ x₂ : ℝ), x₁ ∈ solution_set ∧ x₂ ∈ solution_set ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_x_coordinates_is_seven_l1450_145022


namespace NUMINAMATH_CALUDE_perimeter_of_special_region_l1450_145067

/-- The perimeter of a region bounded by four arcs, each being three-quarters of a circle
    constructed on the sides of a unit square, is equal to 3π. -/
theorem perimeter_of_special_region : Real := by
  -- Define the side length of the square
  let square_side : Real := 1

  -- Define the radius of each circle (half the side length)
  let circle_radius : Real := square_side / 2

  -- Define the length of a full circle with this radius
  let full_circle_length : Real := 2 * Real.pi * circle_radius

  -- Define the length of three-quarters of this circle
  let arc_length : Real := (3 / 4) * full_circle_length

  -- Define the perimeter as four times the arc length
  let perimeter : Real := 4 * arc_length

  -- Prove that this perimeter equals 3π
  sorry

end NUMINAMATH_CALUDE_perimeter_of_special_region_l1450_145067


namespace NUMINAMATH_CALUDE_car_selection_proof_l1450_145006

theorem car_selection_proof (num_cars : ℕ) (num_clients : ℕ) (selections_per_client : ℕ)
  (h1 : num_cars = 10)
  (h2 : num_clients = 15)
  (h3 : selections_per_client = 2)
  (h4 : ∀ i, 1 ≤ i ∧ i ≤ num_cars → ∃ (x : ℕ), x > 0) :
  ∀ i, 1 ≤ i ∧ i ≤ num_cars → ∃ (x : ℕ), x = 3 :=
by sorry

end NUMINAMATH_CALUDE_car_selection_proof_l1450_145006


namespace NUMINAMATH_CALUDE_cone_division_ratio_l1450_145095

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the division of a cone into two parts -/
structure ConeDivision where
  cone : Cone
  ratio : ℝ

/-- Calculates the surface area ratio of the smaller cone to the whole cone -/
def surfaceAreaRatio (d : ConeDivision) : ℝ := 
  d.ratio ^ 2

/-- Calculates the volume ratio of the smaller cone to the whole cone -/
def volumeRatio (d : ConeDivision) : ℝ := 
  d.ratio ^ 3

theorem cone_division_ratio (d : ConeDivision) 
  (h1 : d.cone.height = 4)
  (h2 : d.cone.baseRadius = 3)
  (h3 : surfaceAreaRatio d = volumeRatio d) :
  d.ratio = 125 / 387 := by
  sorry

#eval (125 : Nat) + 387

end NUMINAMATH_CALUDE_cone_division_ratio_l1450_145095


namespace NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_41_l1450_145084

theorem right_triangle_with_hypotenuse_41 :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →
  c = 41 →
  a < b →
  a = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_41_l1450_145084


namespace NUMINAMATH_CALUDE_brownie_triangles_l1450_145058

theorem brownie_triangles (pan_length : ℝ) (pan_width : ℝ) 
                          (triangle_base : ℝ) (triangle_height : ℝ) :
  pan_length = 15 →
  pan_width = 24 →
  triangle_base = 3 →
  triangle_height = 4 →
  (pan_length * pan_width) / ((1/2) * triangle_base * triangle_height) = 60 := by
  sorry

end NUMINAMATH_CALUDE_brownie_triangles_l1450_145058


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l1450_145049

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

/-- The 11th term of a geometric sequence is 648, given that its 5th term is 8 and its 8th term is 72. -/
theorem geometric_sequence_11th_term (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h5 : a 5 = 8) (h8 : a 8 = 72) : a 11 = 648 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l1450_145049


namespace NUMINAMATH_CALUDE_expression_equality_l1450_145043

theorem expression_equality : (34 + 7)^2 - (7^2 + 34^2 + 7 * 34) = 238 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1450_145043


namespace NUMINAMATH_CALUDE_photo_size_is_1_5_l1450_145042

/-- The size of a photo in kilobytes -/
def photo_size : ℝ := sorry

/-- The total storage space of the drive in kilobytes -/
def total_storage : ℝ := 2000 * photo_size

/-- The space used by 400 photos in kilobytes -/
def space_400_photos : ℝ := 400 * photo_size

/-- The space used by 12 200-kilobyte videos in kilobytes -/
def space_12_videos : ℝ := 12 * 200

/-- Theorem stating that the size of each photo is 1.5 kilobytes -/
theorem photo_size_is_1_5 : photo_size = 1.5 := by
  have h1 : total_storage = space_400_photos + space_12_videos := sorry
  sorry

end NUMINAMATH_CALUDE_photo_size_is_1_5_l1450_145042


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l1450_145045

/-- Returns the number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + num_digits (n / 10)

/-- Returns the leftmost digit of a positive integer -/
def leftmost_digit (n : ℕ) : ℕ :=
  n / (10 ^ (num_digits n - 1))

/-- Returns the number after removing the leftmost digit -/
def remove_leftmost_digit (n : ℕ) : ℕ :=
  n % (10 ^ (num_digits n - 1))

/-- Checks if a number satisfies the condition -/
def satisfies_condition (n : ℕ) : Prop :=
  remove_leftmost_digit n = n / 19

theorem smallest_satisfying_number :
  ∀ n : ℕ, n > 0 → n < 1350 → ¬(satisfies_condition n) ∧ satisfies_condition 1350 :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l1450_145045


namespace NUMINAMATH_CALUDE_lifespan_survey_is_sample_l1450_145082

/-- Represents a collection of data from a survey --/
structure SurveyData where
  size : Nat
  provinces : Nat
  dataType : Type

/-- Defines what constitutes a sample in statistical terms --/
def IsSample (data : SurveyData) : Prop :=
  data.size < population_size ∧ data.size > 0
  where population_size : Nat := 1000000  -- Arbitrary large number for illustration

/-- The theorem to be proved --/
theorem lifespan_survey_is_sample :
  let survey : SurveyData := {
    size := 2500,
    provinces := 11,
    dataType := Nat  -- Assuming lifespan is measured in years
  }
  IsSample survey := by sorry


end NUMINAMATH_CALUDE_lifespan_survey_is_sample_l1450_145082


namespace NUMINAMATH_CALUDE_expense_recording_l1450_145071

-- Define a type for financial transactions
structure FinancialTransaction where
  amount : ℤ
  isIncome : Bool

-- Define a function to record a transaction
def recordTransaction (t : FinancialTransaction) : ℤ :=
  if t.isIncome then t.amount else -t.amount

-- Theorem statement
theorem expense_recording :
  ∀ (income expense : FinancialTransaction),
    income.isIncome = true →
    income.amount = 500 →
    recordTransaction income = 500 →
    expense.isIncome = false →
    expense.amount = 400 →
    recordTransaction expense = -400 := by
  sorry

end NUMINAMATH_CALUDE_expense_recording_l1450_145071


namespace NUMINAMATH_CALUDE_no_self_inverse_plus_one_function_l1450_145000

theorem no_self_inverse_plus_one_function : ¬∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_self_inverse_plus_one_function_l1450_145000


namespace NUMINAMATH_CALUDE_division_problem_l1450_145030

theorem division_problem (n t : ℝ) (hn : n > 0) (ht : t > 0) 
  (h : n / t = (n + 2) / (t + 7)) : 
  ∃ z, n / t = (n + 3) / (t + z) ∧ z = 21 / 2 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1450_145030


namespace NUMINAMATH_CALUDE_solve_diamond_equation_l1450_145040

-- Define the binary operation ◊
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- State the properties of the operation
axiom diamond_prop1 (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  diamond a (diamond b c) = diamond a b * c

axiom diamond_prop2 (a : ℝ) (ha : a ≠ 0) :
  diamond a a = 1

-- State the theorem to be proved
theorem solve_diamond_equation :
  ∃ x : ℝ, x ≠ 0 ∧ diamond 2016 (diamond 6 x) = 100 ∧ x = 25 / 84 := by
  sorry

end NUMINAMATH_CALUDE_solve_diamond_equation_l1450_145040


namespace NUMINAMATH_CALUDE_weight_problem_l1450_145085

theorem weight_problem (a b c d e f g h : ℝ) 
  (h1 : (a + b + c + f) / 4 = 80)
  (h2 : (a + b + c + d + e + f) / 6 = 82)
  (h3 : g = d + 5)
  (h4 : h = e - 4)
  (h5 : (c + d + e + f + g + h) / 6 = 83) :
  a + b = 167 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l1450_145085


namespace NUMINAMATH_CALUDE_log3_derivative_l1450_145012

theorem log3_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 3) x = 1 / (x * Real.log 3) := by
sorry

end NUMINAMATH_CALUDE_log3_derivative_l1450_145012


namespace NUMINAMATH_CALUDE_louisa_average_speed_l1450_145026

/-- Proves that given the travel conditions, Louisa's average speed was 40 miles per hour -/
theorem louisa_average_speed :
  ∀ (v : ℝ),
  v > 0 →
  (280 / v) = (160 / v) + 3 →
  v = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_louisa_average_speed_l1450_145026


namespace NUMINAMATH_CALUDE_solve_for_y_l1450_145061

theorem solve_for_y (x y : ℤ) (h1 : x^2 + x + 6 = y - 6) (h2 : x = -5) : y = 32 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1450_145061


namespace NUMINAMATH_CALUDE_nested_expression_value_l1450_145017

def nested_expression : ℕ :=
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))))))

theorem nested_expression_value : nested_expression = 87380 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l1450_145017


namespace NUMINAMATH_CALUDE_total_cement_is_15_1_l1450_145065

/-- The amount of cement used for Lexi's street in tons -/
def lexis_street_cement : ℝ := 10

/-- The amount of cement used for Tess's street in tons -/
def tesss_street_cement : ℝ := 5.1

/-- The total amount of cement used by Roadster's Paving Company in tons -/
def total_cement : ℝ := lexis_street_cement + tesss_street_cement

/-- Theorem stating that the total cement used is 15.1 tons -/
theorem total_cement_is_15_1 : total_cement = 15.1 := by
  sorry

end NUMINAMATH_CALUDE_total_cement_is_15_1_l1450_145065


namespace NUMINAMATH_CALUDE_x_value_l1450_145051

variables (x y z k l m : ℝ)

theorem x_value (h1 : x * y = k * (x + y))
                (h2 : x * z = l * (x + z))
                (h3 : y * z = m * (y + z))
                (hk : k ≠ 0) (hl : l ≠ 0) (hm : m ≠ 0)
                (hkl : k * l + k * m - l * m ≠ 0) :
  x = (2 * k * l * m) / (k * l + k * m - l * m) :=
by sorry

end NUMINAMATH_CALUDE_x_value_l1450_145051


namespace NUMINAMATH_CALUDE_trumpington_band_max_size_l1450_145020

theorem trumpington_band_max_size :
  ∃ (n : ℕ), 
    (20 * n ≡ 4 [MOD 26]) ∧ 
    (20 * n < 1000) ∧
    (∀ (m : ℕ), (20 * m ≡ 4 [MOD 26]) ∧ (20 * m < 1000) → 20 * m ≤ 20 * n) ∧
    (20 * n = 940) :=
by sorry

end NUMINAMATH_CALUDE_trumpington_band_max_size_l1450_145020


namespace NUMINAMATH_CALUDE_sum_58_29_rounded_to_nearest_ten_l1450_145060

/-- Rounds a number to the nearest multiple of 10 -/
def roundToNearestTen (x : ℤ) : ℤ :=
  10 * ((x + 5) / 10)

/-- The sum of 58 and 29 rounded to the nearest ten is 90 -/
theorem sum_58_29_rounded_to_nearest_ten :
  roundToNearestTen (58 + 29) = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_58_29_rounded_to_nearest_ten_l1450_145060


namespace NUMINAMATH_CALUDE_sum_distances_geq_6r_sum_squared_distances_geq_12r_squared_l1450_145063

-- Define a triangle in a plane
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define a point in the plane
def Point : Type := ℝ × ℝ

-- Define the distance function
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the radius of the inscribed circle
def inRadius (t : Triangle) : ℝ := sorry

-- Define Ra, Rb, Rc
def Ra (t : Triangle) (M : Point) : ℝ := distance M t.A
def Rb (t : Triangle) (M : Point) : ℝ := distance M t.B
def Rc (t : Triangle) (M : Point) : ℝ := distance M t.C

-- Theorem 1
theorem sum_distances_geq_6r (t : Triangle) (M : Point) :
  Ra t M + Rb t M + Rc t M ≥ 6 * inRadius t := sorry

-- Theorem 2
theorem sum_squared_distances_geq_12r_squared (t : Triangle) (M : Point) :
  Ra t M ^ 2 + Rb t M ^ 2 + Rc t M ^ 2 ≥ 12 * (inRadius t) ^ 2 := sorry

end NUMINAMATH_CALUDE_sum_distances_geq_6r_sum_squared_distances_geq_12r_squared_l1450_145063


namespace NUMINAMATH_CALUDE_tan_sum_pi_12_pi_4_l1450_145003

theorem tan_sum_pi_12_pi_4 : 
  Real.tan (π / 12) + Real.tan (π / 4) = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_sum_pi_12_pi_4_l1450_145003


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l1450_145086

theorem shaded_area_between_circles (r₁ r₂ chord_length : ℝ) 
  (h₁ : r₁ = 40)
  (h₂ : r₂ = 60)
  (h₃ : chord_length = 100)
  (h₄ : r₁ < r₂)
  (h₅ : chord_length^2 = 4 * (r₂^2 - r₁^2)) : -- Condition for tangency
  (π * r₂^2 - π * r₁^2) = 2000 * π :=
sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l1450_145086


namespace NUMINAMATH_CALUDE_baseball_average_runs_l1450_145014

theorem baseball_average_runs (games : ℕ) (runs_once : ℕ) (runs_twice : ℕ) (runs_thrice : ℕ)
  (h_games : games = 6)
  (h_once : runs_once = 1)
  (h_twice : runs_twice = 4)
  (h_thrice : runs_thrice = 5)
  (h_pattern : 1 * runs_once + 2 * runs_twice + 3 * runs_thrice = games * 4) :
  (1 * runs_once + 2 * runs_twice + 3 * runs_thrice) / games = 4 := by
sorry

end NUMINAMATH_CALUDE_baseball_average_runs_l1450_145014


namespace NUMINAMATH_CALUDE_reading_ratio_is_two_l1450_145025

/-- The minimum number of pages assigned for reading -/
def min_assigned : ℕ := 25

/-- The number of extra pages Harrison read -/
def harrison_extra : ℕ := 10

/-- The number of extra pages Pam read compared to Harrison -/
def pam_extra : ℕ := 15

/-- The number of pages Sam read -/
def sam_pages : ℕ := 100

/-- Calculate the number of pages Harrison read -/
def harrison_pages : ℕ := min_assigned + harrison_extra

/-- Calculate the number of pages Pam read -/
def pam_pages : ℕ := harrison_pages + pam_extra

/-- The ratio of pages Sam read to pages Pam read -/
def reading_ratio : ℚ := sam_pages / pam_pages

theorem reading_ratio_is_two : reading_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_reading_ratio_is_two_l1450_145025
