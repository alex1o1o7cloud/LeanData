import Mathlib

namespace stating_mooncake_packing_solution_l2770_277023

/-- Represents the number of mooncakes in a large bag -/
def large_bag : ℕ := 9

/-- Represents the number of mooncakes in a small package -/
def small_package : ℕ := 4

/-- Represents the total number of mooncakes -/
def total_mooncakes : ℕ := 35

/-- 
Theorem stating that there exist non-negative integers x and y 
such that 9x + 4y = 35, and x + y is minimized
-/
theorem mooncake_packing_solution :
  ∃ x y : ℕ, large_bag * x + small_package * y = total_mooncakes ∧
  ∀ a b : ℕ, large_bag * a + small_package * b = total_mooncakes → x + y ≤ a + b :=
sorry

end stating_mooncake_packing_solution_l2770_277023


namespace intersection_when_a_is_3_range_of_a_when_subset_l2770_277000

def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ a}
def B : Set ℝ := {x | x ≥ 2}

theorem intersection_when_a_is_3 : A 3 ∩ B = {x | 2 ≤ x ∧ x ≤ 3} := by sorry

theorem range_of_a_when_subset : 
  (∀ a : ℝ, A a ⊆ (Set.univ \ B)) → 
  {a : ℝ | ∃ x, x ∈ A a} = Set.Iio 2 := by sorry

end intersection_when_a_is_3_range_of_a_when_subset_l2770_277000


namespace angle_z_is_90_l2770_277099

-- Define a triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)

-- Define the property that the sum of angles in a triangle is 180°
axiom triangle_angle_sum (t : Triangle) : t.X + t.Y + t.Z = 180

-- Theorem: If the sum of angles X and Y is 90°, then angle Z is 90°
theorem angle_z_is_90 (t : Triangle) (h : t.X + t.Y = 90) : t.Z = 90 := by
  sorry

end angle_z_is_90_l2770_277099


namespace unattainable_y_value_l2770_277036

theorem unattainable_y_value (x : ℝ) (h : x ≠ -4/3) :
  ¬∃ y : ℝ, y = -1/3 ∧ y = (2 - x) / (3*x + 4) := by
  sorry

end unattainable_y_value_l2770_277036


namespace trigonometric_expressions_equal_half_l2770_277014

theorem trigonometric_expressions_equal_half :
  let expr1 := Real.sin (15 * π / 180) * Real.cos (15 * π / 180)
  let expr2 := Real.cos (π / 8)^2 - Real.sin (π / 8)^2
  let expr3 := Real.tan (22.5 * π / 180) / (1 - Real.tan (22.5 * π / 180)^2)
  (expr1 ≠ 1/2 ∧ expr2 ≠ 1/2 ∧ expr3 = 1/2) :=
by sorry

end trigonometric_expressions_equal_half_l2770_277014


namespace no_real_solutions_l2770_277008

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 4) :=
by sorry

end no_real_solutions_l2770_277008


namespace probability_three_integer_points_l2770_277035

/-- Square with diagonal endpoints (1/4, 3/4) and (-1/4, -3/4) -/
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    p.1 = t/4 - (1-t)/4 ∧ p.2 = 3*t/4 - 3*(1-t)/4}

/-- Random point v = (x, y) where 0 ≤ x ≤ 100 and 0 ≤ y ≤ 100 -/
def V : Set (ℝ × ℝ) :=
  {v : ℝ × ℝ | 0 ≤ v.1 ∧ v.1 ≤ 100 ∧ 0 ≤ v.2 ∧ v.2 ≤ 100}

/-- Translated copy of S centered at v -/
def T (v : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (q : ℝ × ℝ), q ∈ S ∧ p.1 = q.1 + v.1 ∧ p.2 = q.2 + v.2}

/-- Set of integer points -/
def IntegerPoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (m n : ℤ), p.1 = m ∧ p.2 = n}

/-- Probability measure on V -/
noncomputable def P : (Set (ℝ × ℝ)) → ℝ := sorry

theorem probability_three_integer_points :
  P {v ∈ V | (T v ∩ IntegerPoints).ncard = 3} = 3/100 := sorry

end probability_three_integer_points_l2770_277035


namespace fibonacci_congruence_existence_and_uniqueness_l2770_277068

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_congruence_existence_and_uniqueness :
  ∃! (a b m : ℕ), 0 < a ∧ a < m ∧ 0 < b ∧ b < m ∧
    (∀ n : ℕ, n > 0 → (fibonacci n - a * n * (b ^ n)) % m = 0) ∧
    a = 2 ∧ b = 3 ∧ m = 5 := by
  sorry

end fibonacci_congruence_existence_and_uniqueness_l2770_277068


namespace geometric_sequence_property_l2770_277056

/-- Given a geometric sequence {a_n}, prove that if a_2 * a_6 = 36, then a_4 = ±6 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_prod : a 2 * a 6 = 36) : a 4 = 6 ∨ a 4 = -6 := by
  sorry

end geometric_sequence_property_l2770_277056


namespace pineapple_cost_proof_l2770_277054

/-- Given the cost of pineapples and shipping, prove the total cost per pineapple -/
theorem pineapple_cost_proof (pineapple_cost : ℚ) (num_pineapples : ℕ) (shipping_cost : ℚ) 
  (h1 : pineapple_cost = 5/4)  -- $1.25 represented as a rational number
  (h2 : num_pineapples = 12)
  (h3 : shipping_cost = 21) :
  (pineapple_cost * num_pineapples + shipping_cost) / num_pineapples = 3 := by
  sorry

end pineapple_cost_proof_l2770_277054


namespace entrance_charge_is_twelve_l2770_277087

/-- The entrance charge for the strawberry fields -/
def entrance_charge (standard_price : ℕ) (paid_amount : ℕ) (picked_amount : ℕ) : ℕ :=
  standard_price * picked_amount - paid_amount

/-- Proof that the entrance charge is $12 -/
theorem entrance_charge_is_twelve :
  entrance_charge 20 128 7 = 12 := by
  sorry

end entrance_charge_is_twelve_l2770_277087


namespace nathan_air_hockey_games_l2770_277050

/-- The number of times Nathan played basketball -/
def basketball_games : ℕ := 4

/-- The cost of each game in tokens -/
def tokens_per_game : ℕ := 3

/-- The total number of tokens Nathan used -/
def total_tokens : ℕ := 18

/-- The number of times Nathan played air hockey -/
def air_hockey_games : ℕ := 2

theorem nathan_air_hockey_games :
  air_hockey_games = (total_tokens - basketball_games * tokens_per_game) / tokens_per_game :=
by sorry

end nathan_air_hockey_games_l2770_277050


namespace students_playing_both_sports_l2770_277097

theorem students_playing_both_sports (total : ℕ) (hockey : ℕ) (basketball : ℕ) (neither : ℕ) :
  total = 25 →
  hockey = 15 →
  basketball = 16 →
  neither = 4 →
  hockey + basketball - (total - neither) = 10 :=
by sorry

end students_playing_both_sports_l2770_277097


namespace lucky_sum_equality_l2770_277053

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to select k distinct numbers from 1 to n with sum s -/
def sumCombinations (n k s : ℕ) : ℕ := sorry

/-- The probability of selecting k balls from n balls with sum s -/
def probability (n k s : ℕ) : ℚ :=
  (sumCombinations n k s : ℚ) / (choose n k : ℚ)

theorem lucky_sum_equality (N : ℕ) :
  probability N 10 63 = probability N 8 44 ↔ N = 18 := by
  sorry

end lucky_sum_equality_l2770_277053


namespace cubic_root_sum_l2770_277009

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 10*p - 3 = 0 →
  q^3 - 8*q^2 + 10*q - 3 = 0 →
  r^3 - 8*r^2 + 10*r - 3 = 0 →
  p / (q*r + 2) + q / (p*r + 2) + r / (p*q + 2) = 8/5 := by
sorry

end cubic_root_sum_l2770_277009


namespace union_of_A_and_B_l2770_277038

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x ≤ 3}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end union_of_A_and_B_l2770_277038


namespace petrol_price_equation_l2770_277066

/-- The original price of petrol per gallon -/
def P : ℝ := 2.11

/-- The reduction rate in price -/
def reduction_rate : ℝ := 0.1

/-- The additional gallons that can be bought after price reduction -/
def additional_gallons : ℝ := 5

/-- The fixed amount of money spent -/
def fixed_amount : ℝ := 200

theorem petrol_price_equation :
  fixed_amount / ((1 - reduction_rate) * P) - fixed_amount / P = additional_gallons := by
sorry

end petrol_price_equation_l2770_277066


namespace treasure_chest_coins_l2770_277048

theorem treasure_chest_coins : ∃ n : ℕ, 
  (n > 0) ∧
  (n % 8 = 2) ∧ 
  (n % 7 = 6) ∧ 
  (∀ m : ℕ, m > 0 ∧ m < n → (m % 8 ≠ 2 ∨ m % 7 ≠ 6)) →
  (n % 9 = 7) := by
sorry

end treasure_chest_coins_l2770_277048


namespace second_meal_cost_l2770_277065

/-- The cost of a meal consisting of burgers, shakes, and cola. -/
structure MealCost where
  burger : ℝ
  shake : ℝ
  cola : ℝ

/-- The theorem stating the cost of the second meal given the costs of two other meals. -/
theorem second_meal_cost 
  (meal1 : MealCost) 
  (meal2 : MealCost) 
  (h1 : 3 * meal1.burger + 7 * meal1.shake + meal1.cola = 120)
  (h2 : meal2.burger + meal2.shake + meal2.cola = 39)
  (h3 : meal1 = meal2) :
  4 * meal1.burger + 10 * meal1.shake + meal1.cola = 160.5 := by
  sorry

end second_meal_cost_l2770_277065


namespace eliminate_y_implies_opposite_coefficients_l2770_277072

/-- Given a system of linear equations in two variables x and y,
    prove that if the sum of the equations directly eliminates y,
    then the coefficients of y in the two equations are opposite numbers. -/
theorem eliminate_y_implies_opposite_coefficients 
  (a b c d : ℝ) (k₁ k₂ : ℝ) : 
  (∀ x y : ℝ, a * x + b * y = k₁ ∧ c * x + d * y = k₂) →
  (∀ x : ℝ, (a + c) * x = k₁ + k₂) →
  b + d = 0 :=
sorry

end eliminate_y_implies_opposite_coefficients_l2770_277072


namespace hotel_room_pricing_and_schemes_l2770_277015

theorem hotel_room_pricing_and_schemes :
  ∀ (price_A price_B : ℕ) (schemes : List (ℕ × ℕ)),
  (∃ n : ℕ, 6000 = n * price_A ∧ 4400 = n * price_B) →
  price_A = price_B + 80 →
  (∀ (a b : ℕ), (a, b) ∈ schemes → a + b = 30) →
  (∀ (a b : ℕ), (a, b) ∈ schemes → 2 * a ≥ b) →
  (∀ (a b : ℕ), (a, b) ∈ schemes → a * price_A + b * price_B ≤ 7600) →
  price_A = 300 ∧ price_B = 220 ∧ schemes = [(10, 20), (11, 19), (12, 18)] := by
  sorry

end hotel_room_pricing_and_schemes_l2770_277015


namespace gcd_280_2155_l2770_277049

theorem gcd_280_2155 : Nat.gcd 280 2155 = 35 := by
  sorry

end gcd_280_2155_l2770_277049


namespace impossibleToGet2015Stacks_l2770_277026

/-- Represents a collection of token stacks -/
structure TokenStacks where
  stacks : List Nat
  inv : stacks.sum = 2014

/-- Represents the allowed operations on token stacks -/
inductive Operation
  | Split : Nat → Nat → Operation  -- Split a stack into two
  | Merge : Nat → Nat → Operation  -- Merge two stacks

/-- Applies an operation to the token stacks -/
def applyOperation (ts : TokenStacks) (op : Operation) : TokenStacks :=
  match op with
  | Operation.Split i j => { stacks := i :: j :: ts.stacks.tail, inv := sorry }
  | Operation.Merge i j => { stacks := (i + j) :: ts.stacks.tail.tail, inv := sorry }

/-- The main theorem to prove -/
theorem impossibleToGet2015Stacks (ts : TokenStacks) :
  ¬∃ (ops : List Operation), (ops.foldl applyOperation ts).stacks = List.replicate 2015 1 :=
sorry

end impossibleToGet2015Stacks_l2770_277026


namespace thirteenth_most_likely_friday_l2770_277096

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents the Gregorian calendar -/
structure GregorianCalendar where
  /-- The current year in the 400-year cycle -/
  year : Nat
  /-- Whether the current year is a leap year -/
  is_leap_year : Bool
  /-- The day of the week for the 1st of January of the current year -/
  first_day : DayOfWeek

/-- Counts the occurrences of the 13th falling on each day of the week in a 400-year cycle -/
def count_13ths (calendar : GregorianCalendar) : DayOfWeek → Nat
  | _ => sorry

/-- Theorem: The 13th day of the month falls on Friday more often than on any other day
    in a complete 400-year cycle of the Gregorian calendar -/
theorem thirteenth_most_likely_friday (calendar : GregorianCalendar) :
  ∀ d : DayOfWeek, d ≠ DayOfWeek.Friday → count_13ths calendar DayOfWeek.Friday > count_13ths calendar d := by
  sorry

#check thirteenth_most_likely_friday

end thirteenth_most_likely_friday_l2770_277096


namespace leaky_cistern_fill_time_l2770_277064

/-- Calculates the additional time needed to fill a leaky cistern -/
theorem leaky_cistern_fill_time 
  (fill_time : ℝ) 
  (empty_time : ℝ) 
  (h1 : fill_time = 4) 
  (h2 : empty_time = 20 / 3) : 
  (1 / ((1 / fill_time) - (1 / empty_time))) - fill_time = 6 := by
  sorry

end leaky_cistern_fill_time_l2770_277064


namespace cylinder_height_relationship_l2770_277007

theorem cylinder_height_relationship (r1 h1 r2 h2 : ℝ) :
  r1 > 0 ∧ h1 > 0 ∧ r2 > 0 ∧ h2 > 0 →
  r2 = 1.2 * r1 →
  π * r1^2 * h1 = π * r2^2 * h2 →
  h1 = 1.44 * h2 :=
by sorry

end cylinder_height_relationship_l2770_277007


namespace election_result_l2770_277078

theorem election_result (total_votes : ℕ) (invalid_percentage : ℚ) (second_candidate_votes : ℕ) : 
  total_votes = 7000 →
  invalid_percentage = 1/5 →
  second_candidate_votes = 2520 →
  (((1 - invalid_percentage) * total_votes - second_candidate_votes) / ((1 - invalid_percentage) * total_votes) : ℚ) = 11/20 := by
sorry

end election_result_l2770_277078


namespace restaurant_donates_24_l2770_277046

/-- The restaurant's donation policy -/
def donation_rate : ℚ := 2 / 10

/-- The average customer donation -/
def avg_customer_donation : ℚ := 3

/-- The number of customers -/
def num_customers : ℕ := 40

/-- The restaurant's donation function -/
def restaurant_donation (customer_total : ℚ) : ℚ :=
  (customer_total / 10) * 2

/-- Theorem: The restaurant donates $24 given the conditions -/
theorem restaurant_donates_24 :
  restaurant_donation (avg_customer_donation * num_customers) = 24 := by
  sorry

end restaurant_donates_24_l2770_277046


namespace simplify_expression_l2770_277018

theorem simplify_expression : 18 * (8 / 15) * (1 / 12) = 4 / 5 := by
  sorry

end simplify_expression_l2770_277018


namespace square_sum_xy_l2770_277082

theorem square_sum_xy (x y : ℕ+) 
  (h1 : x * y + x + y = 119)
  (h2 : x^2 * y + x * y^2 = 1680) :
  x^2 + y^2 = 1057 := by
  sorry

end square_sum_xy_l2770_277082


namespace factorization_of_x_squared_minus_four_l2770_277002

theorem factorization_of_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

end factorization_of_x_squared_minus_four_l2770_277002


namespace unique_number_l2770_277022

def is_valid_number (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧
  ∃ (d : ℕ), d < 10 ∧
    (n - d * 10000 + n) = 54321 ∨
    (n - d * 1000 + n) = 54321 ∨
    (n - d * 100 + n) = 54321 ∨
    (n - d * 10 + n) = 54321 ∨
    (n - d + n) = 54321

theorem unique_number : ∀ n : ℕ, is_valid_number n ↔ n = 49383 := by sorry

end unique_number_l2770_277022


namespace three_zero_points_implies_k_leq_neg_two_l2770_277085

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then k * x + 2 else Real.log x

theorem three_zero_points_implies_k_leq_neg_two (k : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    |f k x₁| + k = 0 ∧ |f k x₂| + k = 0 ∧ |f k x₃| + k = 0) →
  k ≤ -2 :=
by sorry

end three_zero_points_implies_k_leq_neg_two_l2770_277085


namespace smallest_integer_y_l2770_277010

theorem smallest_integer_y : ∃ y : ℤ, (y : ℚ) / 4 + 3 / 7 > 2 / 3 ∧ ∀ z : ℤ, z < y → (z : ℚ) / 4 + 3 / 7 ≤ 2 / 3 := by
  sorry

end smallest_integer_y_l2770_277010


namespace candy_bar_calories_l2770_277016

theorem candy_bar_calories (total_calories : ℕ) (total_bars : ℕ) (h1 : total_calories = 2016) (h2 : total_bars = 42) :
  (total_calories / total_bars) / 12 = 4 :=
by sorry

end candy_bar_calories_l2770_277016


namespace cos_neg_three_pi_half_l2770_277034

theorem cos_neg_three_pi_half : Real.cos (-3 * π / 2) = 0 := by
  sorry

end cos_neg_three_pi_half_l2770_277034


namespace min_correct_problems_is_16_l2770_277032

/-- AMC 10 scoring system and John's strategy -/
structure AMC10 where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Nat
  unanswered_points : Nat
  min_total_score : Nat

/-- Calculate the minimum number of correctly solved problems -/
def min_correct_problems (test : AMC10) : Nat :=
  let unanswered := test.total_problems - test.attempted_problems
  let unanswered_score := unanswered * test.unanswered_points
  let required_score := test.min_total_score - unanswered_score
  (required_score + test.correct_points - 1) / test.correct_points

/-- Theorem: The minimum number of correctly solved problems is 16 -/
theorem min_correct_problems_is_16 (test : AMC10) 
  (h1 : test.total_problems = 25)
  (h2 : test.attempted_problems = 20)
  (h3 : test.correct_points = 7)
  (h4 : test.unanswered_points = 2)
  (h5 : test.min_total_score = 120) :
  min_correct_problems test = 16 := by
  sorry

end min_correct_problems_is_16_l2770_277032


namespace county_population_distribution_l2770_277031

theorem county_population_distribution (less_than_10k : ℝ) (between_10k_and_100k : ℝ) :
  less_than_10k = 25 →
  between_10k_and_100k = 59 →
  less_than_10k + between_10k_and_100k = 84 :=
by sorry

end county_population_distribution_l2770_277031


namespace sum_of_powers_mod_five_l2770_277030

theorem sum_of_powers_mod_five (n : ℕ) (hn : n > 0) : 
  (1^n + 2^n + 3^n + 4^n + 5^n) % 5 = 0 :=
sorry

end sum_of_powers_mod_five_l2770_277030


namespace watch_cost_price_l2770_277019

theorem watch_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (additional_amount : ℝ) :
  loss_percentage = 10 →
  gain_percentage = 4 →
  additional_amount = 168 →
  ∃ (cost_price : ℝ),
    cost_price * (1 - loss_percentage / 100) + additional_amount = cost_price * (1 + gain_percentage / 100) ∧
    cost_price = 1200 :=
by sorry

end watch_cost_price_l2770_277019


namespace solution_existence_l2770_277011

theorem solution_existence (k : ℕ+) :
  (∃ x y : ℕ+, x * (x + k) = y * (y + 1)) ↔ (k = 1 ∨ k ≥ 4) := by
  sorry

end solution_existence_l2770_277011


namespace intersecting_circles_distance_l2770_277062

theorem intersecting_circles_distance (R r d : ℝ) : 
  R > 0 → r > 0 → R > r → 
  (∃ (x y : ℝ × ℝ), (x.1 - y.1)^2 + (x.2 - y.2)^2 = d^2 ∧ 
    ∃ (p : ℝ × ℝ), (p.1 - x.1)^2 + (p.2 - x.2)^2 = R^2 ∧ 
                   (p.1 - y.1)^2 + (p.2 - y.2)^2 = r^2) →
  R - r < d ∧ d < R + r :=
by sorry

end intersecting_circles_distance_l2770_277062


namespace frequency_of_boys_born_l2770_277044

theorem frequency_of_boys_born (total : ℕ) (boys : ℕ) (h1 : total = 1000) (h2 : boys = 515) :
  (boys : ℚ) / total = 0.515 := by
sorry

end frequency_of_boys_born_l2770_277044


namespace cubic_root_sum_of_eighth_powers_l2770_277040

theorem cubic_root_sum_of_eighth_powers (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → (b^3 - b + 1 = 0) → (c^3 - c + 1 = 0) → 
  a^8 + b^8 + c^8 = 10 := by
  sorry

end cubic_root_sum_of_eighth_powers_l2770_277040


namespace class_size_proof_l2770_277081

theorem class_size_proof (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 60) 
  (h3 : remaining_average = 90) (h4 : excluded_count = 5) : 
  ∃ (n : ℕ), n = 15 ∧ 
  (n : ℝ) * total_average = 
    ((n : ℝ) - excluded_count) * remaining_average + (excluded_count : ℝ) * excluded_average :=
by
  sorry

#check class_size_proof

end class_size_proof_l2770_277081


namespace new_person_weight_l2770_277067

/-- Given a group of 8 people where one person weighing 45 kg is replaced by a new person,
    and the average weight increases by 6 kg, the weight of the new person is 93 kg. -/
theorem new_person_weight (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 45 →
  avg_increase = 6 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 93 :=
by sorry

end new_person_weight_l2770_277067


namespace max_constant_inequality_l2770_277003

theorem max_constant_inequality (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∀ k : ℝ, (∀ a b c d : ℝ, 0 ≤ a ∧ a ≤ 1 → 0 ≤ b ∧ b ≤ 1 → 0 ≤ c ∧ c ≤ 1 → 0 ≤ d ∧ d ≤ 1 →
    a^2*b + b^2*c + c^2*d + d^2*a + 4 ≥ k*(a^3 + b^3 + c^3 + d^3)) → k ≤ 2 :=
by sorry

end max_constant_inequality_l2770_277003


namespace rainville_rainfall_2006_l2770_277006

/-- The total rainfall in Rainville in 2006 given the average monthly rainfall in 2005 and the increase in 2006 -/
theorem rainville_rainfall_2006 (rainfall_2005 rainfall_increase : ℝ) : 
  rainfall_2005 = 50.0 →
  rainfall_increase = 3 →
  (rainfall_2005 + rainfall_increase) * 12 = 636 := by
  sorry

end rainville_rainfall_2006_l2770_277006


namespace extracted_25_30_is_120_l2770_277070

/-- Represents the number of questionnaires collected for each age group -/
structure QuestionnaireCount where
  group_8_12 : ℕ
  group_13_18 : ℕ
  group_19_24 : ℕ
  group_25_30 : ℕ

/-- Represents the sample extracted from the collected questionnaires -/
structure SampleCount where
  total : ℕ
  group_13_18 : ℕ

/-- Calculates the number of questionnaires extracted from the 25-30 age group -/
def extracted_25_30 (collected : QuestionnaireCount) (sample : SampleCount) : ℕ :=
  (collected.group_25_30 * sample.group_13_18) / collected.group_13_18

theorem extracted_25_30_is_120 (collected : QuestionnaireCount) (sample : SampleCount) :
  collected.group_8_12 = 120 →
  collected.group_13_18 = 180 →
  collected.group_19_24 = 240 →
  sample.total = 300 →
  sample.group_13_18 = 60 →
  extracted_25_30 collected sample = 120 := by
  sorry

#check extracted_25_30_is_120

end extracted_25_30_is_120_l2770_277070


namespace set_operations_and_subset_l2770_277033

-- Define the sets A, B, and C
def A : Set ℝ := {x | -3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < 2*a + 1}

-- State the theorem
theorem set_operations_and_subset :
  (A ∩ B = {x | 2 < x ∧ x < 6}) ∧
  (A ∪ (Set.univ \ B) = {x | x < 6 ∨ 9 ≤ x}) ∧
  (∀ a : ℝ, C a ⊆ A → a ≤ 5/2) :=
sorry

end set_operations_and_subset_l2770_277033


namespace negation_of_proposition_negation_of_specific_proposition_l2770_277017

theorem negation_of_proposition (P : ℝ → Prop) : 
  (¬∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬P x) :=
by sorry

theorem negation_of_specific_proposition : 
  (¬∀ x : ℝ, x^2 + x - 1 < 0) ↔ (∃ x : ℝ, x^2 + x - 1 ≥ 0) :=
by sorry

end negation_of_proposition_negation_of_specific_proposition_l2770_277017


namespace min_value_implies_t_l2770_277001

-- Define the function f
def f (x t : ℝ) : ℝ := |x - t| + |5 - x|

-- State the theorem
theorem min_value_implies_t (t : ℝ) : 
  (∃ (m : ℝ), m = 3 ∧ ∀ (x : ℝ), f x t ≥ m) → t = 2 ∨ t = 8 := by
  sorry

end min_value_implies_t_l2770_277001


namespace sum_of_roots_l2770_277029

-- Define the cubic equation
def cubic_equation (p q d x : ℝ) : Prop := 2 * x^3 - p * x^2 + q * x - d = 0

-- Define the theorem
theorem sum_of_roots (p q d x₁ x₂ x₃ : ℝ) 
  (h_distinct : x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃)
  (h_roots : cubic_equation p q d x₁ ∧ cubic_equation p q d x₂ ∧ cubic_equation p q d x₃)
  (h_positive : p > 0 ∧ q > 0 ∧ d > 0)
  (h_relation : q = 2 * d) :
  x₁ + x₂ + x₃ = p / 2 := by
sorry

end sum_of_roots_l2770_277029


namespace tangent_line_property_l2770_277089

/-- Given a function f: ℝ → ℝ, if the tangent line to the graph of f at the point (2, f(2))
    has the equation 2x - y - 3 = 0, then f(2) + f'(2) = 3. -/
theorem tangent_line_property (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  (∀ x y, y = f 2 → 2 * x - y - 3 = 0 ↔ y = 2 * x - 3) →
  f 2 + deriv f 2 = 3 := by
sorry

end tangent_line_property_l2770_277089


namespace nh4_2so4_weight_l2770_277028

/-- Atomic weight of Nitrogen in g/mol -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Atomic weight of Sulfur in g/mol -/
def S_weight : ℝ := 32.07

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Number of Nitrogen atoms in (NH4)2SO4 -/
def N_count : ℕ := 2

/-- Number of Hydrogen atoms in (NH4)2SO4 -/
def H_count : ℕ := 8

/-- Number of Sulfur atoms in (NH4)2SO4 -/
def S_count : ℕ := 1

/-- Number of Oxygen atoms in (NH4)2SO4 -/
def O_count : ℕ := 4

/-- Number of moles of (NH4)2SO4 -/
def moles : ℝ := 7

/-- Molecular weight of (NH4)2SO4 in g/mol -/
def molecular_weight : ℝ := N_weight * N_count + H_weight * H_count + S_weight * S_count + O_weight * O_count

theorem nh4_2so4_weight : moles * molecular_weight = 924.19 := by
  sorry

end nh4_2so4_weight_l2770_277028


namespace min_product_of_three_numbers_l2770_277043

theorem min_product_of_three_numbers (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1)
  (x_leq_2y : x ≤ 2*y)
  (y_leq_2z : y ≤ 2*z) :
  x * y * z ≥ 6 / 343 := by
sorry

end min_product_of_three_numbers_l2770_277043


namespace sqrt_equation_solution_l2770_277084

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (x - 5) + 1 = 10 → x = 86 := by
  sorry

end sqrt_equation_solution_l2770_277084


namespace expression_simplification_l2770_277021

theorem expression_simplification (x y k : ℝ) 
  (hk : k ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = k * y) :
  (x - k / x) * (y + 1 / (k * y)) = (x^2 * k - k^3) / x^2 := by
  sorry

end expression_simplification_l2770_277021


namespace smallest_cube_root_with_small_fraction_l2770_277069

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (r : ℝ) : 
  (0 < n) →
  (0 < r) →
  (r < 1 / 500) →
  (m : ℝ)^(1/3) = n + r →
  (∀ k < n, ¬∃ s, (0 < s) ∧ (s < 1 / 500) ∧ (∃ l : ℕ, (l : ℝ)^(1/3) = k + s)) →
  n = 13 := by
  sorry

#check smallest_cube_root_with_small_fraction

end smallest_cube_root_with_small_fraction_l2770_277069


namespace cos_theta_minus_phi_l2770_277020

theorem cos_theta_minus_phi (θ φ : ℝ) :
  Complex.exp (θ * Complex.I) = (4 / 5 : ℂ) + (3 / 5 : ℂ) * Complex.I →
  Complex.exp (φ * Complex.I) = (5 / 13 : ℂ) - (12 / 13 : ℂ) * Complex.I →
  Real.cos (θ - φ) = -16 / 65 := by
sorry

end cos_theta_minus_phi_l2770_277020


namespace total_faces_painted_l2770_277042

/-- The number of cuboids painted by Ezekiel -/
def num_cuboids : ℕ := 5

/-- The number of faces on each cuboid -/
def faces_per_cuboid : ℕ := 6

/-- Theorem stating the total number of faces painted by Ezekiel -/
theorem total_faces_painted :
  num_cuboids * faces_per_cuboid = 30 := by
  sorry

end total_faces_painted_l2770_277042


namespace square_difference_of_solutions_l2770_277073

theorem square_difference_of_solutions (α β : ℝ) : 
  (α^2 = 2*α + 1) → (β^2 = 2*β + 1) → (α ≠ β) → (α - β)^2 = 8 := by sorry

end square_difference_of_solutions_l2770_277073


namespace probability_matching_shoes_l2770_277075

theorem probability_matching_shoes (n : ℕ) (h : n = 9) :
  let total_shoes := 2 * n
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := n
  (matching_pairs : ℚ) / total_combinations = 1 / 17 := by
  sorry

end probability_matching_shoes_l2770_277075


namespace leftmost_row_tiles_l2770_277039

/-- Represents the number of tiles in each row of the floor -/
def tileSequence (firstRow : ℕ) : ℕ → ℕ
  | 0 => firstRow
  | n + 1 => tileSequence firstRow n - 2

/-- The sum of tiles in all rows -/
def totalTiles (firstRow : ℕ) : ℕ :=
  (List.range 9).map (tileSequence firstRow) |>.sum

theorem leftmost_row_tiles :
  ∃ (firstRow : ℕ), totalTiles firstRow = 405 ∧ firstRow = 53 := by
  sorry

end leftmost_row_tiles_l2770_277039


namespace unique_solution_floor_equation_l2770_277041

theorem unique_solution_floor_equation :
  ∃! c : ℝ, c + ⌊c⌋ = 25.6 :=
by
  sorry

end unique_solution_floor_equation_l2770_277041


namespace min_value_quadratic_l2770_277061

theorem min_value_quadratic (x : ℝ) : x^2 - 3*x + 2023 ≥ 2020 + 3/4 := by
  sorry

end min_value_quadratic_l2770_277061


namespace problem_solution_l2770_277095

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 2|

-- State the theorem
theorem problem_solution :
  ∀ m : ℝ,
  (∀ x : ℝ, f m (x + 2) ≥ 0 ↔ x ∈ Set.Icc (-1) 1) →
  (m = 1 ∧
   {x : ℝ | |x + 1| + |x - 2| > 4 * m} = {x : ℝ | x < -3/2 ∨ x > 5/2}) :=
by sorry

end problem_solution_l2770_277095


namespace sum_of_coefficients_l2770_277079

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, x^10 + x^4 + 1 = a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 := by
sorry

end sum_of_coefficients_l2770_277079


namespace sum_with_reverse_has_even_digit_l2770_277086

def is_17_digit (n : ℕ) : Prop := 10^16 ≤ n ∧ n < 10^17

def reverse_number (n : ℕ) : ℕ :=
  let digits := List.reverse (Nat.digits 10 n)
  List.foldl (λ acc d => acc * 10 + d) 0 digits

def has_even_digit (n : ℕ) : Prop :=
  ∃ d, d ∈ Nat.digits 10 n ∧ Even d

theorem sum_with_reverse_has_even_digit (n : ℕ) (h : is_17_digit n) :
  has_even_digit (n + reverse_number n) := by
  sorry

end sum_with_reverse_has_even_digit_l2770_277086


namespace trig_product_equals_one_sixteenth_l2770_277013

theorem trig_product_equals_one_sixteenth :
  (1 - Real.sin (π / 12)) * (1 - Real.sin (5 * π / 12)) *
  (1 - Real.sin (7 * π / 12)) * (1 - Real.sin (11 * π / 12)) = 1 / 16 := by
  sorry

end trig_product_equals_one_sixteenth_l2770_277013


namespace parabola_point_ordinate_l2770_277098

/-- Represents a parabola y = ax^2 with a > 0 -/
structure Parabola where
  a : ℝ
  a_pos : a > 0

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = p.a * x^2

theorem parabola_point_ordinate (p : Parabola) (M : PointOnParabola p) 
    (focus_directrix_dist : (1 : ℝ) / (2 * p.a) = 1)
    (M_to_focus_dist : Real.sqrt ((M.x - 0)^2 + (M.y - 1 / (4 * p.a))^2) = 5) :
    M.y = 9/2 := by
  sorry

end parabola_point_ordinate_l2770_277098


namespace caterpillar_climb_days_l2770_277024

/-- The number of days it takes for a caterpillar to climb a pole -/
def climbingDays (poleHeight : ℕ) (dayClimb : ℕ) (nightSlide : ℕ) : ℕ :=
  let netClimbPerDay := dayClimb - nightSlide
  let daysToAlmostTop := (poleHeight - dayClimb) / netClimbPerDay
  daysToAlmostTop + 1

/-- Theorem stating that it takes 16 days for the caterpillar to reach the top -/
theorem caterpillar_climb_days :
  climbingDays 20 5 4 = 16 := by
  sorry

end caterpillar_climb_days_l2770_277024


namespace integer_between_sqrt_twelve_l2770_277012

theorem integer_between_sqrt_twelve : ∃ (m : ℤ), m < 2 * Real.sqrt 3 ∧ 2 * Real.sqrt 3 < m + 1 → m = 3 := by
  sorry

end integer_between_sqrt_twelve_l2770_277012


namespace calculation_difference_l2770_277060

theorem calculation_difference : ∀ x : ℝ, (x - 3) + 49 = 66 → (3 * x + 49) - 66 = 43 := by
  sorry

end calculation_difference_l2770_277060


namespace sum_of_square_roots_equals_one_l2770_277083

theorem sum_of_square_roots_equals_one (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 15) :
  Real.sqrt b / (a + b) + Real.sqrt c / (b + c) + Real.sqrt a / (c + a) = 1 := by
  sorry

end sum_of_square_roots_equals_one_l2770_277083


namespace domain_transformation_l2770_277047

-- Define a real-valued function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x²)
def domain_f_squared : Set ℝ := Set.Ioc (-3) 1

-- Define the domain of f(x-1)
def domain_f_shifted : Set ℝ := Set.Ico 1 10

-- Theorem statement
theorem domain_transformation (h : ∀ x, x ∈ domain_f_squared ↔ f (x^2) ∈ Set.range f) :
  ∀ x, x ∈ domain_f_shifted ↔ f (x - 1) ∈ Set.range f :=
sorry

end domain_transformation_l2770_277047


namespace bumper_car_line_problem_l2770_277063

theorem bumper_car_line_problem (initial_people : ℕ) (joined : ℕ) (final_people : ℕ) :
  initial_people = 12 →
  joined = 15 →
  final_people = 17 →
  ∃ (left : ℕ), initial_people - left + joined = final_people ∧ left = 10 :=
by sorry

end bumper_car_line_problem_l2770_277063


namespace sqrt_minus_three_minus_m_real_l2770_277092

theorem sqrt_minus_three_minus_m_real (m : ℝ) :
  (∃ (x : ℝ), x ^ 2 = -3 - m) ↔ m ≤ -3 :=
by sorry

end sqrt_minus_three_minus_m_real_l2770_277092


namespace arithmetic_sequence_problem_l2770_277091

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_3 : a 3 = 9)
  (h_9 : a 9 = 3) :
  (∀ n : ℕ, a n = 12 - n) ∧ 
  (∀ n : ℕ, n ≥ 13 → a n < 0) ∧
  (∀ n : ℕ, n < 13 → a n ≥ 0) :=
by sorry

end arithmetic_sequence_problem_l2770_277091


namespace problem_statement_l2770_277094

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : a * b = 1) : 
  (a + b ≥ 2) ∧ (a^3 + b^3 ≥ 2) := by
  sorry

end problem_statement_l2770_277094


namespace scientific_notation_18_million_l2770_277076

theorem scientific_notation_18_million :
  (18000000 : ℝ) = 1.8 * (10 : ℝ) ^ 7 :=
sorry

end scientific_notation_18_million_l2770_277076


namespace equality_of_sides_from_equal_angles_l2770_277052

-- Define a structure for 3D points
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a function to calculate the angle between three points
def angle (p1 p2 p3 : Point3D) : ℝ := sorry

-- Define a function to calculate the distance between two points
def distance (p1 p2 : Point3D) : ℝ := sorry

-- Define a predicate to check if four points are non-coplanar
def nonCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

theorem equality_of_sides_from_equal_angles 
  (A B C D : Point3D) 
  (h1 : nonCoplanar A B C D)
  (h2 : angle A B C = angle A D C)
  (h3 : angle B A D = angle B C D) :
  distance A B = distance C D ∧ distance B C = distance A D := by
  sorry

end equality_of_sides_from_equal_angles_l2770_277052


namespace complex_equation_proof_l2770_277051

theorem complex_equation_proof (z : ℂ) (h : z = -1/2 + (Real.sqrt 3 / 2) * Complex.I) : 
  z^2 + z + 1 = 0 := by sorry

end complex_equation_proof_l2770_277051


namespace shelby_rain_time_l2770_277071

/-- Represents the speed of Shelby's scooter in miles per hour -/
structure ScooterSpeed where
  normal : ℝ  -- Speed when not raining
  rain : ℝ    -- Speed when raining

/-- Represents Shelby's journey -/
structure Journey where
  total_distance : ℝ  -- Total distance covered in miles
  total_time : ℝ      -- Total time taken in minutes
  rain_time : ℝ       -- Time driven in rain in minutes

/-- Checks if the given journey satisfies the conditions of Shelby's ride -/
def is_valid_journey (speed : ScooterSpeed) (j : Journey) : Prop :=
  speed.normal = 40 ∧
  speed.rain = 25 ∧
  j.total_distance = 20 ∧
  j.total_time = 40 ∧
  j.total_distance = (speed.normal / 60) * (j.total_time - j.rain_time) + (speed.rain / 60) * j.rain_time

theorem shelby_rain_time (speed : ScooterSpeed) (j : Journey) 
  (h : is_valid_journey speed j) : j.rain_time = 27 := by
  sorry

end shelby_rain_time_l2770_277071


namespace sum_of_factors_36_l2770_277077

theorem sum_of_factors_36 : (List.sum (List.filter (λ x => 36 % x = 0) (List.range 37))) = 91 := by
  sorry

end sum_of_factors_36_l2770_277077


namespace combinatorics_problem_l2770_277027

theorem combinatorics_problem :
  (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial (15 - 6))) = 5005 ∧
  Nat.factorial 6 = 720 := by
sorry

end combinatorics_problem_l2770_277027


namespace expression_equals_m_times_ten_to_1006_l2770_277057

theorem expression_equals_m_times_ten_to_1006 : 
  (3^1005 + 7^1006)^2 - (3^1005 - 7^1006)^2 = 114337548 * 10^1006 := by
  sorry

end expression_equals_m_times_ten_to_1006_l2770_277057


namespace optimal_square_perimeter_l2770_277045

/-- Given a wire of length 1 cut into two pieces to form a square and a circle,
    the perimeter of the square that minimizes the sum of their areas is π / (π + 4) -/
theorem optimal_square_perimeter :
  ∃ (x : ℝ), x > 0 ∧ x < 1 ∧
  (∀ (y : ℝ), y > 0 → y < 1 →
    x^2 / 16 + (1 - x)^2 / (4 * π) ≤ y^2 / 16 + (1 - y)^2 / (4 * π)) ∧
  x = π / (π + 4) := by
  sorry

end optimal_square_perimeter_l2770_277045


namespace intersection_count_l2770_277005

-- Define the two curves
def curve1 (x y : ℝ) : Prop := 3 * x^2 + 2 * y^2 = 6
def curve2 (x y : ℝ) : Prop := x^2 - 2 * y^2 = 1

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  curve1 x y ∧ curve2 x y

-- Define a function to count distinct intersection points
def count_distinct_intersections : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem intersection_count :
  count_distinct_intersections = 4 :=
sorry

end intersection_count_l2770_277005


namespace tyler_saltwater_animals_l2770_277059

/-- The number of aquariums Tyler has -/
def num_aquariums : ℕ := 8

/-- The number of animals in each aquarium -/
def animals_per_aquarium : ℕ := 64

/-- The total number of saltwater animals Tyler has -/
def total_animals : ℕ := num_aquariums * animals_per_aquarium

theorem tyler_saltwater_animals :
  total_animals = 512 :=
sorry

end tyler_saltwater_animals_l2770_277059


namespace min_disks_needed_l2770_277080

/-- Represents the capacity of each disk in MB -/
def diskCapacity : ℚ := 1.44

/-- Represents the file sizes in MB -/
def fileSizes : List ℚ := [0.9, 0.6, 0.45, 0.3]

/-- Represents the quantity of each file size -/
def fileQuantities : List ℕ := [5, 10, 10, 5]

/-- Calculates the total storage required for all files in MB -/
def totalStorage : ℚ :=
  List.sum (List.zipWith (· * ·) (List.map (λ x => (x : ℚ)) fileQuantities) fileSizes)

/-- Theorem: The minimum number of disks needed is 15 -/
theorem min_disks_needed : 
  ∃ (n : ℕ), n = 15 ∧ 
  n * diskCapacity ≥ totalStorage ∧
  ∀ m : ℕ, m * diskCapacity ≥ totalStorage → m ≥ n :=
by sorry

end min_disks_needed_l2770_277080


namespace geometric_sequence_sum_l2770_277093

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, ∃ r : ℝ, r ≠ 0 ∧ a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end geometric_sequence_sum_l2770_277093


namespace original_machines_work_hours_l2770_277004

/-- The number of original machines in the factory -/
def original_machines : ℕ := 3

/-- The number of hours the new machine works per day -/
def new_machine_hours : ℕ := 12

/-- The production rate of each machine in kg per hour -/
def production_rate : ℕ := 2

/-- The selling price of the material in dollars per kg -/
def selling_price : ℕ := 50

/-- The total earnings of the factory in one day in dollars -/
def total_earnings : ℕ := 8100

/-- Theorem stating that the original machines work 23 hours a day -/
theorem original_machines_work_hours : 
  ∃ h : ℕ, 
    (original_machines * production_rate * h + new_machine_hours * production_rate) * selling_price = total_earnings ∧ 
    h = 23 := by
  sorry

end original_machines_work_hours_l2770_277004


namespace triangle_problem_l2770_277090

theorem triangle_problem (A B C : Real) (a b c : Real) :
  let m : Real × Real := (Real.sqrt 3, 1 - Real.cos A)
  let n : Real × Real := (Real.sin A, -1)
  (m.1 * n.1 + m.2 * n.2 = 0) →  -- m ⊥ n
  (a = 2) →
  (Real.cos B = Real.sqrt 3 / 3) →
  (A = 2 * Real.pi / 3 ∧ b = 4 * Real.sqrt 2 / 3) := by
  sorry


end triangle_problem_l2770_277090


namespace man_son_age_ratio_l2770_277074

/-- Represents the age ratio of a man to his son after two years -/
def age_ratio (son_age : ℕ) (age_difference : ℕ) : ℚ :=
  (son_age + age_difference + 2) / (son_age + 2)

theorem man_son_age_ratio :
  let son_age : ℕ := 20
  let age_difference : ℕ := 22
  age_ratio son_age age_difference = 2 := by
  sorry

end man_son_age_ratio_l2770_277074


namespace arithmetic_sequence_sum_15_l2770_277025

/-- An arithmetic sequence with common difference d ≠ 0 -/
def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_sum_15 
  (a : ℕ → ℚ) (d : ℚ) 
  (h_arith : arithmetic_sequence a d)
  (h_eq : a 5^2 + a 7^2 + 16*d = a 9^2 + a 11^2) :
  sum_arithmetic a 15 = 15 := by
sorry

end arithmetic_sequence_sum_15_l2770_277025


namespace abs_three_plus_one_l2770_277058

theorem abs_three_plus_one (a : ℝ) : 
  (|a| = 3) → (a + 1 = 4 ∨ a + 1 = -2) := by sorry

end abs_three_plus_one_l2770_277058


namespace number_equality_l2770_277055

theorem number_equality (x : ℝ) : (0.4 * x = 0.3 * 50) → x = 37.5 := by
  sorry

end number_equality_l2770_277055


namespace parabola_y_intercepts_l2770_277088

/-- The number of y-intercepts of the parabola x = 3y^2 - 5y - 2 -/
def num_y_intercepts : ℕ := 2

/-- The equation of the parabola -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 5 * y - 2

theorem parabola_y_intercepts :
  (∃ (s : Finset ℝ), s.card = num_y_intercepts ∧
    ∀ y ∈ s, parabola_equation y = 0) :=
by sorry

end parabola_y_intercepts_l2770_277088


namespace encryption_proof_l2770_277037

def encrypt (x : ℕ) : ℕ :=
  if x % 2 = 1 ∧ 1 ≤ x ∧ x ≤ 26 then
    (x + 1) / 2
  else if x % 2 = 0 ∧ 1 ≤ x ∧ x ≤ 26 then
    x / 2 + 13
  else
    0

def letter_to_num (c : Char) : ℕ :=
  match c with
  | 'a' => 1 | 'b' => 2 | 'c' => 3 | 'd' => 4 | 'e' => 5
  | 'f' => 6 | 'g' => 7 | 'h' => 8 | 'i' => 9 | 'j' => 10
  | 'k' => 11 | 'l' => 12 | 'm' => 13 | 'n' => 14 | 'o' => 15
  | 'p' => 16 | 'q' => 17 | 'r' => 18 | 's' => 19 | 't' => 20
  | 'u' => 21 | 'v' => 22 | 'w' => 23 | 'x' => 24 | 'y' => 25
  | 'z' => 26
  | _ => 0

def num_to_letter (n : ℕ) : Char :=
  match n with
  | 1 => 'a' | 2 => 'b' | 3 => 'c' | 4 => 'd' | 5 => 'e'
  | 6 => 'f' | 7 => 'g' | 8 => 'h' | 9 => 'i' | 10 => 'j'
  | 11 => 'k' | 12 => 'l' | 13 => 'm' | 14 => 'n' | 15 => 'o'
  | 16 => 'p' | 17 => 'q' | 18 => 'r' | 19 => 's' | 20 => 't'
  | 21 => 'u' | 22 => 'v' | 23 => 'w' | 24 => 'x' | 25 => 'y'
  | 26 => 'z'
  | _ => ' '

theorem encryption_proof :
  (encrypt (letter_to_num 'l'), 
   encrypt (letter_to_num 'o'), 
   encrypt (letter_to_num 'v'), 
   encrypt (letter_to_num 'e')) = 
  (letter_to_num 's', 
   letter_to_num 'h', 
   letter_to_num 'x', 
   letter_to_num 'c') := by
  sorry

end encryption_proof_l2770_277037
