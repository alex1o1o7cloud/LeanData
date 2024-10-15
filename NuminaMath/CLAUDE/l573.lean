import Mathlib

namespace NUMINAMATH_CALUDE_prime_cube_difference_to_sum_of_squares_l573_57303

theorem prime_cube_difference_to_sum_of_squares (p a b : ℕ) : 
  Prime p → (∃ a b : ℕ, p = a^3 - b^3) → (∃ c d : ℕ, p = c^2 + 3*d^2) := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_difference_to_sum_of_squares_l573_57303


namespace NUMINAMATH_CALUDE_festival_allowance_days_l573_57349

/-- Calculates the maximum number of full days for festival allowance --/
def maxAllowanceDays (staffCount : Nat) (dailyRate : Nat) (totalAmount : Nat) (pettyCashAmount : Nat) : Nat :=
  let totalAvailable := totalAmount + pettyCashAmount
  (totalAvailable - pettyCashAmount) / (staffCount * dailyRate)

theorem festival_allowance_days :
  maxAllowanceDays 20 100 65000 1000 = 32 := by
  sorry

end NUMINAMATH_CALUDE_festival_allowance_days_l573_57349


namespace NUMINAMATH_CALUDE_hundredth_card_is_ninth_l573_57381

/-- Represents the cyclic order of cards in a standard deck --/
def cardCycle : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

/-- The number of cards in a cycle --/
def cycleLength : ℕ := 13

/-- The position we want to find --/
def targetPosition : ℕ := 100

/-- Function to get the equivalent position in the cycle --/
def cyclicPosition (n : ℕ) : ℕ :=
  (n - 1) % cycleLength + 1

theorem hundredth_card_is_ninth (h : targetPosition = 100) :
  cyclicPosition targetPosition = 9 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_card_is_ninth_l573_57381


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l573_57399

/-- Given a line l: 3x + 4y - 12 = 0, prove that 3x + 4y - 9 = 0 is the equation of the line
    that passes through the point (-1, 3) and has the same slope as line l. -/
theorem parallel_line_through_point (x y : ℝ) : 
  let l : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 12 = 0
  let m : ℝ := -3 / 4  -- slope of line l
  let new_line : ℝ → ℝ → Prop := λ x y => 3 * x + 4 * y - 9 = 0
  (∀ x y, l x y → (y - 0) = m * (x - 0)) →  -- l has slope m
  new_line (-1) 3 →  -- new line passes through (-1, 3)
  (∀ x y, new_line x y → (y - 3) = m * (x - (-1))) →  -- new line has slope m
  ∀ x y, new_line x y ↔ 3 * x + 4 * y - 9 = 0 := by
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l573_57399


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l573_57358

def regular_decagon : Nat := 10

/-- The number of distinct interior points where two or more diagonals intersect in a regular decagon -/
def intersection_points (n : Nat) : Nat :=
  Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points regular_decagon = 210 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l573_57358


namespace NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l573_57332

/-- Given a triangle ABC with centroid G, if GA² + GB² + GC² = 72, then AB² + AC² + BC² = 216 -/
theorem triangle_centroid_distance_sum (A B C G : ℝ × ℝ) : 
  (G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) →
  ((G.1 - A.1)^2 + (G.2 - A.2)^2 + 
   (G.1 - B.1)^2 + (G.2 - B.2)^2 + 
   (G.1 - C.1)^2 + (G.2 - C.2)^2 = 72) →
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 + 
   (A.1 - C.1)^2 + (A.2 - C.2)^2 + 
   (B.1 - C.1)^2 + (B.2 - C.2)^2 = 216) := by
sorry

end NUMINAMATH_CALUDE_triangle_centroid_distance_sum_l573_57332


namespace NUMINAMATH_CALUDE_only_parallel_converse_true_l573_57302

-- Define the basic concepts
def Line : Type := sorry
def Angle : Type := sorry
def Triangle : Type := sorry

-- Define properties and relations
def parallel (l1 l2 : Line) : Prop := sorry
def alternateInteriorAngles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry
def isosceles (t : Triangle) : Prop := sorry
def acute (t : Triangle) : Prop := sorry
def rightAngle (a : Angle) : Prop := sorry
def correspondingAngles (a1 a2 : Angle) (t1 t2 : Triangle) : Prop := sorry
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem stating that only the converse of proposition B is true
theorem only_parallel_converse_true :
  (∀ t : Triangle, acute t → isosceles t) = False ∧
  (∀ l1 l2 : Line, ∀ a1 a2 : Angle, alternateInteriorAngles a1 a2 l1 l2 → parallel l1 l2) = True ∧
  (∀ t1 t2 : Triangle, ∀ a1 a2 : Angle, correspondingAngles a1 a2 t1 t2 → congruent t1 t2) = False ∧
  (∀ a1 a2 : Angle, a1 = a2 → rightAngle a1 ∧ rightAngle a2) = False :=
sorry

end NUMINAMATH_CALUDE_only_parallel_converse_true_l573_57302


namespace NUMINAMATH_CALUDE_race_length_l573_57339

theorem race_length (L : ℝ) 
  (h1 : L - 70 = L * ((L - 100) / L))  -- A beats B by 70 m
  (h2 : L - 163 = (L - 100) * ((L - 163) / (L - 100)))  -- B beats C by 100 m
  (h3 : L - 163 = L * ((L - 163) / L))  -- A beats C by 163 m
  : L = 1000 := by
  sorry

end NUMINAMATH_CALUDE_race_length_l573_57339


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l573_57300

/-- A function satisfying the given functional equations -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (Real.sqrt 3 / 3 * x) = Real.sqrt 3 * f x - 2 * Real.sqrt 3 / 3 * x) ∧
  (∀ x y, y ≠ 0 → f x * f y = f (x * y) + f (x / y))

/-- The theorem stating that x + 1/x is the only function satisfying the given equations -/
theorem unique_satisfying_function :
  ∀ f : ℝ → ℝ, SatisfyingFunction f ↔ ∀ x, f x = x + 1/x :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l573_57300


namespace NUMINAMATH_CALUDE_table_sticks_prove_table_sticks_l573_57359

/-- The number of sticks of wood a chair makes -/
def chair_sticks : ℕ := 6

/-- The number of sticks of wood a stool makes -/
def stool_sticks : ℕ := 2

/-- The number of sticks of wood Mary needs to burn per hour -/
def sticks_per_hour : ℕ := 5

/-- The number of chairs Mary chopped -/
def chairs_chopped : ℕ := 18

/-- The number of tables Mary chopped -/
def tables_chopped : ℕ := 6

/-- The number of stools Mary chopped -/
def stools_chopped : ℕ := 4

/-- The number of hours Mary can keep warm -/
def hours_warm : ℕ := 34

/-- The theorem stating that a table makes 9 sticks of wood -/
theorem table_sticks : ℕ :=
  let total_sticks := hours_warm * sticks_per_hour
  let chair_total := chairs_chopped * chair_sticks
  let stool_total := stools_chopped * stool_sticks
  let table_total := total_sticks - chair_total - stool_total
  table_total / tables_chopped

/-- Proof of the theorem -/
theorem prove_table_sticks : table_sticks = 9 := by
  sorry


end NUMINAMATH_CALUDE_table_sticks_prove_table_sticks_l573_57359


namespace NUMINAMATH_CALUDE_power_product_squared_l573_57372

theorem power_product_squared (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_squared_l573_57372


namespace NUMINAMATH_CALUDE_bug_meeting_time_l573_57369

theorem bug_meeting_time (r₁ r₂ v₁ v₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 3) 
  (h₃ : v₁ = 4 * Real.pi) (h₄ : v₂ = 3 * Real.pi) : ∃ t : ℝ, t = 2.5 ∧ 
  (∃ n₁ n₂ : ℕ, t * v₁ = 2 * Real.pi * r₁ * n₁ ∧ 
   t * v₂ = 2 * Real.pi * r₂ * (n₂ + 1/4)) := by
  sorry

end NUMINAMATH_CALUDE_bug_meeting_time_l573_57369


namespace NUMINAMATH_CALUDE_range_of_a_l573_57370

def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a}

theorem range_of_a (a : ℝ) :
  (a > 0) →
  ((1 ∈ A a) ∨ (2 ∈ A a)) ∧
  ¬((1 ∈ A a) ∧ (2 ∈ A a)) →
  1 < a ∧ a ≤ 2 :=
by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l573_57370


namespace NUMINAMATH_CALUDE_binomial_coefficient_seven_three_l573_57337

theorem binomial_coefficient_seven_three : Nat.choose 7 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_seven_three_l573_57337


namespace NUMINAMATH_CALUDE_hyperbola_equation_l573_57365

/-- Given a hyperbola and a parabola with specific properties, prove the equation of the hyperbola -/
theorem hyperbola_equation (a b : ℝ) (P : ℝ × ℝ) :
  a > 0 → b > 0 →
  (∃ F : ℝ × ℝ, F = (2, 0) ∧ 
    (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ (x - F.1)^2/a^2 - (y - F.2)^2/b^2 = 1) ∧
    (P.2^2 = 8*P.1) ∧
    ((P.1 - F.1)^2 + (P.2 - F.2)^2 = 25)) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 ↔ x^2 - y^2/3 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l573_57365


namespace NUMINAMATH_CALUDE_octal_246_equals_166_l573_57395

/-- Converts a base-8 digit to its base-10 equivalent -/
def octal_to_decimal (digit : ℕ) : ℕ := digit

/-- Represents a base-8 number as a list of digits -/
def octal_number : List ℕ := [2, 4, 6]

/-- Converts a base-8 number to its base-10 equivalent -/
def octal_to_decimal_conversion (num : List ℕ) : ℕ :=
  List.foldl (fun acc (digit : ℕ) => acc * 8 + octal_to_decimal digit) 0 num.reverse

theorem octal_246_equals_166 :
  octal_to_decimal_conversion octal_number = 166 := by
  sorry

end NUMINAMATH_CALUDE_octal_246_equals_166_l573_57395


namespace NUMINAMATH_CALUDE_irwin_family_hike_distance_l573_57380

/-- The total distance hiked by Irwin's family during their camping trip. -/
def total_distance_hiked (car_to_stream stream_to_meadow meadow_to_campsite : ℝ) : ℝ :=
  car_to_stream + stream_to_meadow + meadow_to_campsite

/-- Theorem stating that the total distance hiked by Irwin's family is 0.7 miles. -/
theorem irwin_family_hike_distance :
  total_distance_hiked 0.2 0.4 0.1 = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_irwin_family_hike_distance_l573_57380


namespace NUMINAMATH_CALUDE_discount_order_matters_l573_57356

def original_price : ℚ := 50
def fixed_discount : ℚ := 10
def percentage_discount : ℚ := 0.25

def price_fixed_then_percentage : ℚ := (original_price - fixed_discount) * (1 - percentage_discount)
def price_percentage_then_fixed : ℚ := (original_price * (1 - percentage_discount)) - fixed_discount

theorem discount_order_matters :
  price_percentage_then_fixed < price_fixed_then_percentage ∧
  (price_fixed_then_percentage - price_percentage_then_fixed) * 100 = 250 := by
  sorry

end NUMINAMATH_CALUDE_discount_order_matters_l573_57356


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l573_57308

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (p^3 - 25*p^2 + 90*p - 73 = 0) →
  (q^3 - 25*q^2 + 90*q - 73 = 0) →
  (r^3 - 25*r^2 + 90*r - 73 = 0) →
  (∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 25*s^2 + 90*s - 73) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 256 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l573_57308


namespace NUMINAMATH_CALUDE_two_digit_powers_of_three_l573_57398

theorem two_digit_powers_of_three :
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99)) ∧
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99) ∧ Finset.card s = 2) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_powers_of_three_l573_57398


namespace NUMINAMATH_CALUDE_pentagon_count_l573_57316

/-- The number of points on the circumference of the circle -/
def n : ℕ := 15

/-- The number of vertices in each pentagon -/
def k : ℕ := 5

/-- The number of different convex pentagons that can be formed -/
def num_pentagons : ℕ := n.choose k

theorem pentagon_count : num_pentagons = 3003 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_count_l573_57316


namespace NUMINAMATH_CALUDE_moon_mission_cost_share_l573_57351

/-- Calculates the individual share of a total cost divided equally among a population -/
def individual_share (total_cost : ℕ) (population : ℕ) : ℚ :=
  (total_cost : ℚ) / (population : ℚ)

/-- Proves that the individual share of 40 billion dollars among 200 million people is 200 dollars -/
theorem moon_mission_cost_share :
  individual_share (40 * 10^9) (200 * 10^6) = 200 := by
  sorry

end NUMINAMATH_CALUDE_moon_mission_cost_share_l573_57351


namespace NUMINAMATH_CALUDE_soldiers_on_great_wall_count_l573_57331

/-- The number of soldiers in beacon towers along the Great Wall --/
def soldiers_on_great_wall (wall_length : ℕ) (tower_interval : ℕ) (soldiers_per_tower : ℕ) : ℕ :=
  (wall_length / tower_interval) * soldiers_per_tower

/-- Theorem stating the number of soldiers on the Great Wall --/
theorem soldiers_on_great_wall_count :
  soldiers_on_great_wall 7300 5 2 = 2920 := by
  sorry

end NUMINAMATH_CALUDE_soldiers_on_great_wall_count_l573_57331


namespace NUMINAMATH_CALUDE_lemon_ratio_l573_57354

-- Define the number of lemons for each person
def levi_lemons : ℕ := 5
def jayden_lemons : ℕ := levi_lemons + 6
def ian_lemons : ℕ := 66  -- This is derived from the total, not given directly
def eli_lemons : ℕ := ian_lemons / 2
def total_lemons : ℕ := 115

-- Theorem statement
theorem lemon_ratio : 
  levi_lemons = 5 ∧
  jayden_lemons = levi_lemons + 6 ∧
  eli_lemons = ian_lemons / 2 ∧
  levi_lemons + jayden_lemons + eli_lemons + ian_lemons = total_lemons ∧
  total_lemons = 115 →
  jayden_lemons * 3 = eli_lemons := by
  sorry

end NUMINAMATH_CALUDE_lemon_ratio_l573_57354


namespace NUMINAMATH_CALUDE_tavern_keeper_pays_for_beer_l573_57387

/-- Represents the currency of a country -/
structure Currency where
  name : String
  value : ℚ

/-- Represents a country with its currency and exchange rate -/
structure Country where
  name : String
  currency : Currency
  exchangeRate : ℚ

/-- Represents a transaction in a country -/
structure Transaction where
  country : Country
  amountPaid : ℚ
  itemCost : ℚ
  changeReceived : ℚ

/-- The beer lover's transactions -/
def beerLoverTransactions (anchuria gvaiasuela : Country) : List Transaction := sorry

/-- The tavern keeper's profit or loss -/
def tavernKeeperProfit (transactions : List Transaction) : ℚ := sorry

/-- Theorem stating that the tavern keeper pays for the beer -/
theorem tavern_keeper_pays_for_beer (anchuria gvaiasuela : Country) 
  (h1 : anchuria.currency.value = gvaiasuela.currency.value)
  (h2 : anchuria.exchangeRate = 90 / 100)
  (h3 : gvaiasuela.exchangeRate = 90 / 100)
  (h4 : ∀ t ∈ beerLoverTransactions anchuria gvaiasuela, t.itemCost = 10 / 100) :
  tavernKeeperProfit (beerLoverTransactions anchuria gvaiasuela) < 0 := by
  sorry

#check tavern_keeper_pays_for_beer

end NUMINAMATH_CALUDE_tavern_keeper_pays_for_beer_l573_57387


namespace NUMINAMATH_CALUDE_blue_balls_unchanged_l573_57341

/-- Represents the number of balls of each color in the box -/
structure BallCount where
  red : Nat
  blue : Nat
  yellow : Nat

/-- The operation of adding yellow balls to the box -/
def addYellowBalls (initial : BallCount) (added : Nat) : BallCount :=
  { red := initial.red,
    blue := initial.blue,
    yellow := initial.yellow + added }

theorem blue_balls_unchanged (initial : BallCount) (added : Nat) :
  (addYellowBalls initial added).blue = initial.blue :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_unchanged_l573_57341


namespace NUMINAMATH_CALUDE_vector_collinearity_l573_57391

/-- Given vectors a, b, and c in R², prove that if a = (-2, 0), b = (2, 1), c = (x, 1),
    and 3a + b is collinear with c, then x = -4. -/
theorem vector_collinearity (a b c : ℝ × ℝ) (x : ℝ) :
  a = (-2, 0) →
  b = (2, 1) →
  c = (x, 1) →
  ∃ (k : ℝ), k ≠ 0 ∧ (3 • a + b) = k • c →
  x = -4 := by
sorry

end NUMINAMATH_CALUDE_vector_collinearity_l573_57391


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l573_57368

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

theorem fourth_term_of_sequence (y : ℝ) :
  let a₁ := 8
  let a₂ := 32 * y^2
  let a₃ := 128 * y^4
  let r := a₂ / a₁
  geometric_sequence a₁ r 4 = 512 * y^6 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l573_57368


namespace NUMINAMATH_CALUDE_sqrt_sum_fractions_l573_57352

theorem sqrt_sum_fractions : Real.sqrt (1/9 + 1/16) = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fractions_l573_57352


namespace NUMINAMATH_CALUDE_cherie_boxes_count_l573_57301

/-- The number of boxes Koby bought -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of sparklers in each of Cherie's boxes -/
def cherie_sparklers_per_box : ℕ := 8

/-- The number of whistlers in each of Cherie's boxes -/
def cherie_whistlers_per_box : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 33

/-- The number of boxes Cherie bought -/
def cherie_boxes : ℕ := 1

theorem cherie_boxes_count : 
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box) + 
  cherie_boxes * (cherie_sparklers_per_box + cherie_whistlers_per_box) = 
  total_fireworks :=
by sorry

end NUMINAMATH_CALUDE_cherie_boxes_count_l573_57301


namespace NUMINAMATH_CALUDE_carly_job_applications_l573_57325

/-- The number of job applications Carly sent to her home state -/
def home_state_apps : ℕ := 200

/-- The number of job applications Carly sent to the neighboring state -/
def neighboring_state_apps : ℕ := 2 * home_state_apps

/-- The number of job applications Carly sent to each of the other 3 states -/
def other_state_apps : ℕ := neighboring_state_apps - 50

/-- The number of other states Carly sent applications to -/
def num_other_states : ℕ := 3

/-- The total number of job applications Carly sent -/
def total_applications : ℕ := home_state_apps + neighboring_state_apps + (num_other_states * other_state_apps)

theorem carly_job_applications : total_applications = 1650 := by
  sorry

end NUMINAMATH_CALUDE_carly_job_applications_l573_57325


namespace NUMINAMATH_CALUDE_probability_sum_less_than_4_l573_57364

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a point satisfies a condition within a given square --/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The condition x + y < 4 --/
def sumLessThan4 (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 4

theorem probability_sum_less_than_4 :
  let s : Square := { bottomLeft := (0, 0), topRight := (3, 3) }
  probability s sumLessThan4 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_less_than_4_l573_57364


namespace NUMINAMATH_CALUDE_adjacent_pairs_after_10_minutes_l573_57317

/-- Represents the number of adjacent pairs of the same letter after n minutes -/
def a (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n+1 => 2^n - 1 - a n

/-- The transformation rule applied for n minutes -/
def transform (n : ℕ) : String :=
  match n with
  | 0 => "A"
  | n+1 => String.replace (transform n) "A" "AB" |>.replace "B" "BA"

theorem adjacent_pairs_after_10_minutes :
  (transform 10).length = 1024 ∧ a 10 = 341 := by
  sorry

#eval a 10  -- Should output 341

end NUMINAMATH_CALUDE_adjacent_pairs_after_10_minutes_l573_57317


namespace NUMINAMATH_CALUDE_james_age_l573_57367

/-- Represents the ages of Dan, James, and Lisa --/
structure Ages where
  dan : ℕ
  james : ℕ
  lisa : ℕ

/-- The conditions of the problem --/
def age_conditions (ages : Ages) : Prop :=
  ∃ (k : ℕ),
    ages.dan = 6 * k ∧
    ages.james = 5 * k ∧
    ages.lisa = 4 * k ∧
    ages.dan + 4 = 28 ∧
    ages.james + ages.lisa = 3 * (ages.james - ages.lisa)

/-- The theorem to prove --/
theorem james_age (ages : Ages) :
  age_conditions ages → ages.james = 20 := by
  sorry

end NUMINAMATH_CALUDE_james_age_l573_57367


namespace NUMINAMATH_CALUDE_power_zero_eq_one_l573_57360

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_zero_eq_one_l573_57360


namespace NUMINAMATH_CALUDE_visit_either_not_both_l573_57373

def probability_chile : ℝ := 0.5
def probability_madagascar : ℝ := 0.5

theorem visit_either_not_both :
  probability_chile + probability_madagascar - 2 * (probability_chile * probability_madagascar) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_visit_either_not_both_l573_57373


namespace NUMINAMATH_CALUDE_circle_properties_l573_57321

-- Define the circle C in polar coordinates
def C (ρ θ : ℝ) : Prop := ρ^2 = 4*ρ*(Real.cos θ + Real.sin θ) - 6

-- Define the circle C in rectangular coordinates
def C_rect (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 2

-- Theorem statement
theorem circle_properties :
  -- 1. Equivalence of polar and rectangular equations
  (∀ x y : ℝ, C_rect x y ↔ ∃ ρ θ : ℝ, C ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧
  -- 2. Maximum value of x + y is 6
  (∀ x y : ℝ, C_rect x y → x + y ≤ 6) ∧
  -- 3. (3, 3) is on C and achieves the maximum
  C_rect 3 3 ∧ 3 + 3 = 6 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l573_57321


namespace NUMINAMATH_CALUDE_prob_exactly_one_red_l573_57338

structure Box where
  red : ℕ
  black : ℕ

def Box.total (b : Box) : ℕ := b.red + b.black

def prob_red (b : Box) : ℚ := b.red / b.total

def prob_black (b : Box) : ℚ := b.black / b.total

def box_A : Box := ⟨1, 2⟩
def box_B : Box := ⟨2, 2⟩

theorem prob_exactly_one_red : 
  prob_red box_A * prob_black box_B + prob_black box_A * prob_red box_B = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_exactly_one_red_l573_57338


namespace NUMINAMATH_CALUDE_yulgi_pocket_money_l573_57333

/-- Proves that Yulgi's pocket money is 3600 won given the problem conditions -/
theorem yulgi_pocket_money :
  ∀ (y g : ℕ),
  y + g = 6000 →
  (y + g) - (y - g) = 4800 →
  y > g →
  y = 3600 := by
sorry

end NUMINAMATH_CALUDE_yulgi_pocket_money_l573_57333


namespace NUMINAMATH_CALUDE_sin_pi_plus_A_implies_cos_three_pi_half_minus_A_l573_57385

theorem sin_pi_plus_A_implies_cos_three_pi_half_minus_A 
  (A : ℝ) (h : Real.sin (π + A) = 1/2) : 
  Real.cos ((3/2) * π - A) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_plus_A_implies_cos_three_pi_half_minus_A_l573_57385


namespace NUMINAMATH_CALUDE_chinese_dream_essay_contest_l573_57362

theorem chinese_dream_essay_contest (total : ℕ) (seventh : ℕ) (eighth : ℕ) :
  total = 118 →
  seventh = eighth / 2 - 2 →
  total = seventh + eighth →
  seventh = 38 := by
sorry

end NUMINAMATH_CALUDE_chinese_dream_essay_contest_l573_57362


namespace NUMINAMATH_CALUDE_orthocenter_of_triangle_l573_57375

/-- The orthocenter of a triangle ABC --/
def orthocenter (A B C : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The orthocenter of triangle ABC with given coordinates --/
theorem orthocenter_of_triangle :
  let A : ℝ × ℝ × ℝ := (2, 3, 4)
  let B : ℝ × ℝ × ℝ := (4, 1, 1)
  let C : ℝ × ℝ × ℝ := (1, 5, 6)
  orthocenter A B C = (-79/3, 91/3, 41/3) := by sorry

end NUMINAMATH_CALUDE_orthocenter_of_triangle_l573_57375


namespace NUMINAMATH_CALUDE_probability_of_specific_hand_l573_57394

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of cards drawn -/
def NumDraws : ℕ := 5

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Probability of the specific outcome -/
def SpecificOutcomeProbability : ℚ := 3 / 832

theorem probability_of_specific_hand :
  let prob_ace : ℚ := NumAces / StandardDeck
  let prob_non_ace_suit : ℚ := (StandardDeck - NumAces) / StandardDeck
  let prob_specific_suit : ℚ := (StandardDeck / NumSuits) / StandardDeck
  NumSuits * prob_ace * prob_non_ace_suit * prob_specific_suit * prob_specific_suit = SpecificOutcomeProbability :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_hand_l573_57394


namespace NUMINAMATH_CALUDE_base7_addition_l573_57382

/-- Converts a base 7 number represented as a list of digits to base 10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- Converts a base 10 number to base 7 represented as a list of digits --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem base7_addition :
  toBase7 (toBase10 [3, 5, 4, 1] + toBase10 [4, 1, 6, 3, 2]) = [5, 5, 4, 5, 2] := by
  sorry

end NUMINAMATH_CALUDE_base7_addition_l573_57382


namespace NUMINAMATH_CALUDE_committee_selection_l573_57318

theorem committee_selection (n m : ℕ) (hn : n = 20) (hm : m = 3) :
  Nat.choose n m = 1140 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l573_57318


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l573_57330

theorem no_solution_to_inequalities :
  ¬∃ (x y z t : ℝ),
    (abs x < abs (y - z + t)) ∧
    (abs y < abs (x - z + t)) ∧
    (abs z < abs (x - y + t)) ∧
    (abs t < abs (x - y + z)) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l573_57330


namespace NUMINAMATH_CALUDE_journey_speed_proof_l573_57344

/-- Proves that given a journey in three equal parts with speeds 5 km/hr, v km/hr, and 15 km/hr,
    where the total time is 11 minutes and the total distance is 1.5 km, the value of v is 10 km/hr. -/
theorem journey_speed_proof (v : ℝ) : 
  let total_distance : ℝ := 1.5 -- km
  let part_distance : ℝ := total_distance / 3
  let total_time : ℝ := 11 / 60 -- hours
  let time1 : ℝ := part_distance / 5
  let time2 : ℝ := part_distance / v
  let time3 : ℝ := part_distance / 15
  time1 + time2 + time3 = total_time → v = 10 := by sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l573_57344


namespace NUMINAMATH_CALUDE_speed_in_still_water_l573_57328

def upstream_speed : ℝ := 25
def downstream_speed : ℝ := 31

theorem speed_in_still_water :
  (upstream_speed + downstream_speed) / 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_speed_in_still_water_l573_57328


namespace NUMINAMATH_CALUDE_fish_tank_problem_l573_57315

/-- Given 3 fish tanks with a total of 100 fish, where two tanks have twice as many fish as the first tank, prove that the first tank contains 20 fish. -/
theorem fish_tank_problem (first_tank : ℕ) : 
  first_tank + 2 * first_tank + 2 * first_tank = 100 → first_tank = 20 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l573_57315


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l573_57326

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 - b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a + b = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l573_57326


namespace NUMINAMATH_CALUDE_factor_in_range_l573_57323

theorem factor_in_range : ∃ m : ℕ, 
  (201212200619 : ℕ) % m = 0 ∧ 
  (6 * 10^9 : ℕ) < m ∧ 
  m < (13 * 10^9 : ℕ) / 2 ∧
  m = 6490716149 := by
sorry

end NUMINAMATH_CALUDE_factor_in_range_l573_57323


namespace NUMINAMATH_CALUDE_probability_of_two_primes_l573_57329

/-- A function that determines if a number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of integers from 1 to 30 inclusive -/
def integerSet : Finset ℕ := sorry

/-- The set of prime numbers from 1 to 30 inclusive -/
def primeSet : Finset ℕ := sorry

/-- The number of ways to choose 2 items from a set of size n -/
def choose (n : ℕ) (k : ℕ) : ℕ := sorry

theorem probability_of_two_primes :
  (choose (Finset.card primeSet) 2 : ℚ) / (choose (Finset.card integerSet) 2) = 10 / 87 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_two_primes_l573_57329


namespace NUMINAMATH_CALUDE_complex_equation_solution_l573_57311

theorem complex_equation_solution (a b : ℝ) :
  (Complex.I : ℂ) * 2 + 1 = (Complex.I + 1) * (Complex.I * b + a) →
  a = 3/2 ∧ b = 1/2 :=
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l573_57311


namespace NUMINAMATH_CALUDE_unique_a_l573_57343

/-- The equation is quadratic in x -/
def is_quadratic (a : ℝ) : Prop :=
  |a - 1| = 2

/-- The coefficient of the quadratic term is non-zero -/
def coeff_nonzero (a : ℝ) : Prop :=
  a - 3 ≠ 0

/-- The value of a that satisfies the conditions -/
theorem unique_a : ∃! a : ℝ, is_quadratic a ∧ coeff_nonzero a :=
  sorry

end NUMINAMATH_CALUDE_unique_a_l573_57343


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l573_57390

/-- An isosceles triangle with side lengths 2 and 4 has a perimeter of 10 -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 2 ∧ b = 4 ∧ c = 4 ∧  -- Two sides are equal (isosceles) and one side is 2
  a + b > c ∧ b + c > a ∧ a + c > b →  -- Triangle inequality
  a + b + c = 10 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l573_57390


namespace NUMINAMATH_CALUDE_chessboard_tiling_exists_l573_57374

/-- Represents a chessboard of size 2^n x 2^n -/
structure Chessboard (n : ℕ) where
  size : Fin (2^n) × Fin (2^n)

/-- Represents an L-shaped triomino -/
inductive Triomino
  | L : Triomino

/-- Represents a tiling of a chessboard -/
def Tiling (n : ℕ) := Chessboard n → Option Triomino

/-- States that a tiling is valid for a chessboard with one square removed -/
def is_valid_tiling (n : ℕ) (t : Tiling n) (removed : Fin (2^n) × Fin (2^n)) : Prop :=
  ∀ (pos : Fin (2^n) × Fin (2^n)), pos ≠ removed → t ⟨pos⟩ = some Triomino.L

/-- Theorem: For any 2^n x 2^n chessboard with one square removed, 
    there exists a valid tiling using L-shaped triominoes -/
theorem chessboard_tiling_exists (n : ℕ) (removed : Fin (2^n) × Fin (2^n)) :
  ∃ (t : Tiling n), is_valid_tiling n t removed := by
  sorry

end NUMINAMATH_CALUDE_chessboard_tiling_exists_l573_57374


namespace NUMINAMATH_CALUDE_decimal_to_binary_23_l573_57376

theorem decimal_to_binary_23 : 
  (23 : ℕ) = (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0) := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_23_l573_57376


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l573_57304

/-- A regular polygon with side length 6 units and exterior angle 60 degrees has a perimeter of 36 units. -/
theorem regular_polygon_perimeter (s : ℝ) (θ : ℝ) (h1 : s = 6) (h2 : θ = 60) :
  let n : ℝ := 360 / θ
  s * n = 36 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l573_57304


namespace NUMINAMATH_CALUDE_baker_cakes_theorem_l573_57345

/-- Represents the number of cakes Baker made -/
def cakes_made : ℕ := sorry

/-- Represents the number of pastries Baker made -/
def pastries_made : ℕ := 153

/-- Represents the number of pastries Baker sold -/
def pastries_sold : ℕ := 8

/-- Represents the number of cakes Baker sold -/
def cakes_sold : ℕ := 97

/-- Represents the difference between cakes sold and pastries sold -/
def difference_sold : ℕ := 89

theorem baker_cakes_theorem : 
  pastries_made = 153 ∧ 
  pastries_sold = 8 ∧ 
  cakes_sold = 97 ∧ 
  difference_sold = 89 ∧ 
  cakes_sold - pastries_sold = difference_sold → 
  cakes_made = 97 :=
by sorry

end NUMINAMATH_CALUDE_baker_cakes_theorem_l573_57345


namespace NUMINAMATH_CALUDE_quadratic_roots_max_value_l573_57377

theorem quadratic_roots_max_value (t q : ℝ) (a₁ a₂ : ℝ) : 
  (∀ (n : ℕ), 1 ≤ n → n ≤ 2010 → a₁^n + a₂^n = a₁ + a₂) →
  a₁^2 - t*a₁ + q = 0 →
  a₂^2 - t*a₂ + q = 0 →
  (∀ (x : ℝ), x^2 - t*x + q ≠ 0 ∨ x = a₁ ∨ x = a₂) →
  (1 / a₁^2011 + 1 / a₂^2011) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_max_value_l573_57377


namespace NUMINAMATH_CALUDE_cupcakes_per_package_l573_57392

theorem cupcakes_per_package (packages : ℕ) (eaten : ℕ) (left : ℕ) :
  packages = 3 →
  eaten = 5 →
  left = 7 →
  ∃ cupcakes_per_package : ℕ,
    cupcakes_per_package * packages = eaten + left ∧
    cupcakes_per_package = 4 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_per_package_l573_57392


namespace NUMINAMATH_CALUDE_ellipse_m_value_l573_57307

/-- An ellipse with equation x²/(10-m) + y²/(m-2) = 1, major axis on x-axis, and focal distance 4 -/
structure Ellipse (m : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 / (10 - m) + y^2 / (m - 2) = 1)
  (major_axis_x : ℝ → ℝ)
  (focal_distance : ℝ)
  (h_focal_distance : focal_distance = 4)

/-- The value of m for the given ellipse is 4 -/
theorem ellipse_m_value (m : ℝ) (e : Ellipse m) : m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_m_value_l573_57307


namespace NUMINAMATH_CALUDE_second_digit_of_three_digit_number_l573_57350

/-- Given a three-digit number xyz, if 100x + 10y + z - (x + y + z) = 261, then y = 7 -/
theorem second_digit_of_three_digit_number (x y z : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ z ≥ 0 ∧ z ≤ 9 →
  100 * x + 10 * y + z - (x + y + z) = 261 →
  y = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_digit_of_three_digit_number_l573_57350


namespace NUMINAMATH_CALUDE_minimum_value_of_f_l573_57322

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  ∀ x : ℝ, x > 2 → f x ≥ f 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_l573_57322


namespace NUMINAMATH_CALUDE_nested_expression_value_l573_57355

theorem nested_expression_value : (3*(3*(2*(2*(2*(3+2)+1)+1)+2)+1)+1) = 436 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l573_57355


namespace NUMINAMATH_CALUDE_sample_size_theorem_l573_57309

/-- Represents the types of products produced by the factory -/
inductive ProductType
  | A
  | B
  | C

/-- Represents the quantity ratio of products -/
def quantity_ratio : ProductType → ℕ
  | ProductType.A => 2
  | ProductType.B => 3
  | ProductType.C => 5

/-- Calculates the total ratio sum -/
def total_ratio : ℕ := quantity_ratio ProductType.A + quantity_ratio ProductType.B + quantity_ratio ProductType.C

/-- Represents the number of Type B products in the sample -/
def type_b_sample : ℕ := 24

/-- Theorem: If 24 units of Type B are drawn in a stratified random sample 
    from a production with ratio 2:3:5, then the total sample size is 80 -/
theorem sample_size_theorem : 
  (type_b_sample * total_ratio) / quantity_ratio ProductType.B = 80 := by
  sorry

end NUMINAMATH_CALUDE_sample_size_theorem_l573_57309


namespace NUMINAMATH_CALUDE_union_equals_A_l573_57305

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}

theorem union_equals_A (a : ℝ) : (A ∪ B a = A) → (a = 2 ∨ a = 3) := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l573_57305


namespace NUMINAMATH_CALUDE_co2_formation_l573_57386

-- Define the chemical reaction
structure Reaction where
  hcl : ℕ
  nahco3 : ℕ
  co2 : ℕ

-- Define the stoichiometric ratio
def stoichiometric_ratio (r : Reaction) : Prop :=
  r.hcl = r.nahco3 ∧ r.co2 = r.hcl

-- Define the theorem
theorem co2_formation (r : Reaction) (h1 : stoichiometric_ratio r) (h2 : r.hcl = 3) (h3 : r.nahco3 = 3) :
  r.co2 = min r.hcl r.nahco3 := by
  sorry

#check co2_formation

end NUMINAMATH_CALUDE_co2_formation_l573_57386


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l573_57366

-- Problem 1
theorem factorization_problem_1 (a b : ℝ) :
  a^2 * (a - b) + 4 * b^2 * (b - a) = (a - b) * (a + 2*b) * (a - 2*b) := by sorry

-- Problem 2
theorem factorization_problem_2 (m : ℝ) :
  m^4 - 1 = (m^2 + 1) * (m + 1) * (m - 1) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l573_57366


namespace NUMINAMATH_CALUDE_a_equals_two_l573_57348

theorem a_equals_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
  (h3 : ∀ x : ℝ, a^(2*x - 4) ≤ 2^(x^2 - 2*x)) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_a_equals_two_l573_57348


namespace NUMINAMATH_CALUDE_initial_books_count_l573_57320

theorem initial_books_count (action_figures : ℕ) (added_books : ℕ) (difference : ℕ) : 
  action_figures = 7 →
  added_books = 4 →
  difference = 1 →
  ∃ (initial_books : ℕ), 
    initial_books + added_books + difference = action_figures ∧
    initial_books = 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_books_count_l573_57320


namespace NUMINAMATH_CALUDE_final_weight_gain_l573_57378

def weight_change (initial_weight : ℕ) (final_weight : ℕ) : ℕ :=
  let first_loss := 12
  let second_gain := 2 * first_loss
  let third_loss := 3 * first_loss
  let weight_after_third_loss := initial_weight - first_loss + second_gain - third_loss
  final_weight - weight_after_third_loss

theorem final_weight_gain (initial_weight final_weight : ℕ) 
  (h1 : initial_weight = 99)
  (h2 : final_weight = 81) :
  weight_change initial_weight final_weight = 6 := by
  sorry

#eval weight_change 99 81

end NUMINAMATH_CALUDE_final_weight_gain_l573_57378


namespace NUMINAMATH_CALUDE_trigonometric_fraction_equals_two_l573_57379

theorem trigonometric_fraction_equals_two :
  (3 - Real.sin (70 * π / 180)) / (2 - Real.cos (10 * π / 180) ^ 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_fraction_equals_two_l573_57379


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l573_57340

theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 2 →                            -- First term is 2
  a 5 = 8 →                            -- Fifth term is 8
  a 2 * a 3 * a 4 = 64 := by            -- Product of middle terms is 64
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l573_57340


namespace NUMINAMATH_CALUDE_quadratic_root_implies_b_value_l573_57314

theorem quadratic_root_implies_b_value (b : ℝ) : 
  (Complex.I ^ 2 = -1) →
  ((3 : ℂ) + Complex.I) ^ 2 - 6 * ((3 : ℂ) + Complex.I) + b = 0 →
  b = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_b_value_l573_57314


namespace NUMINAMATH_CALUDE_cos_225_degrees_l573_57324

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l573_57324


namespace NUMINAMATH_CALUDE_matthew_crackers_l573_57306

theorem matthew_crackers (initial_crackers remaining_crackers crackers_per_friend : ℕ) 
  (h1 : initial_crackers = 23)
  (h2 : remaining_crackers = 11)
  (h3 : crackers_per_friend = 6) :
  (initial_crackers - remaining_crackers) / crackers_per_friend = 2 :=
by sorry

end NUMINAMATH_CALUDE_matthew_crackers_l573_57306


namespace NUMINAMATH_CALUDE_equation_solution_a_l573_57334

theorem equation_solution_a : 
  ∀ (a b c : ℤ), 
  (∀ x : ℝ, (x - a) * (x - 5) + 4 = (x + b) * (x + c)) → 
  (a = 0 ∨ a = 1) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_a_l573_57334


namespace NUMINAMATH_CALUDE_cylinder_radii_ratio_l573_57397

/-- Given two cylinders of the same height, this theorem proves that if their volumes are 40 cc
    and 360 cc respectively, then the ratio of their radii is 1:3. -/
theorem cylinder_radii_ratio (h : ℝ) (r₁ r₂ : ℝ) 
  (h_pos : h > 0) (r₁_pos : r₁ > 0) (r₂_pos : r₂ > 0) :
  π * r₁^2 * h = 40 → π * r₂^2 * h = 360 → r₁ / r₂ = 1 / 3 := by
  sorry

#check cylinder_radii_ratio

end NUMINAMATH_CALUDE_cylinder_radii_ratio_l573_57397


namespace NUMINAMATH_CALUDE_blue_marble_ratio_l573_57363

/-- Proves that the ratio of blue marbles to total marbles is 1:2 -/
theorem blue_marble_ratio (total : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 164 → 
  red = total / 4 →
  green = 27 →
  yellow = 14 →
  blue = total - (red + green + yellow) →
  (blue : ℚ) / total = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_blue_marble_ratio_l573_57363


namespace NUMINAMATH_CALUDE_function_simplification_l573_57388

theorem function_simplification (θ : Real) 
  (h1 : θ ∈ Set.Icc π (2 * π)) 
  (h2 : Real.tan θ = 2) : 
  ((1 + Real.sin θ + Real.cos θ) * (Real.sin (θ / 2) - Real.cos (θ / 2))) / 
   Real.sqrt (2 + 2 * Real.cos θ) = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_simplification_l573_57388


namespace NUMINAMATH_CALUDE_old_clock_slower_by_12_minutes_l573_57336

/-- Represents the time interval between consecutive coincidences of hour and minute hands -/
def coincidence_interval : ℕ := 66

/-- Represents the number of coincidences in a 24-hour period -/
def coincidences_per_day : ℕ := 22

/-- Represents the number of minutes in a standard 24-hour day -/
def standard_day_minutes : ℕ := 24 * 60

/-- Represents the number of minutes in the old clock's 24-hour period -/
def old_clock_day_minutes : ℕ := coincidence_interval * coincidences_per_day

theorem old_clock_slower_by_12_minutes :
  old_clock_day_minutes - standard_day_minutes = 12 := by sorry

end NUMINAMATH_CALUDE_old_clock_slower_by_12_minutes_l573_57336


namespace NUMINAMATH_CALUDE_prop1_prop2_prop3_prop4_l573_57361

-- Define the function f
variable (f : ℝ → ℝ)

-- Proposition 1
theorem prop1 (h : ∀ x, f (1 + 2*x) = f (1 - 2*x)) :
  ∀ x, f (2 - x) = f x :=
sorry

-- Proposition 2
theorem prop2 :
  ∀ x, f (x - 2) = f (2 - x) :=
sorry

-- Proposition 3
theorem prop3 (h1 : ∀ x, f x = f (-x)) (h2 : ∀ x, f (2 + x) = -f x) :
  ∀ x, f (4 - x) = f x :=
sorry

-- Proposition 4
theorem prop4 (h1 : ∀ x, f x = -f (-x)) (h2 : ∀ x, f x = f (-x - 2)) :
  ∀ x, f (2 - x) = f x :=
sorry

end NUMINAMATH_CALUDE_prop1_prop2_prop3_prop4_l573_57361


namespace NUMINAMATH_CALUDE_line_translation_l573_57384

/-- A line in the xy-plane. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Vertical translation of a line. -/
def vertical_translate (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - d }

theorem line_translation (l : Line) :
  l.slope = 3 ∧ l.intercept = 2 →
  vertical_translate l 3 = { slope := 3, intercept := -1 } := by
  sorry

end NUMINAMATH_CALUDE_line_translation_l573_57384


namespace NUMINAMATH_CALUDE_dividing_line_halves_area_l573_57319

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The T-shaped region -/
def TRegion : Set Point := {p | 
  (0 ≤ p.x ∧ p.x ≤ 4 ∧ 0 ≤ p.y ∧ p.y ≤ 4) ∨
  (4 < p.x ∧ p.x ≤ 7 ∧ 0 ≤ p.y ∧ p.y ≤ 2)
}

/-- The line y = (1/2)x -/
def DividingLine (p : Point) : Prop :=
  p.y = (1/2) * p.x

/-- The area of a region -/
noncomputable def area (s : Set Point) : ℝ := sorry

/-- The part of the region above the line -/
def UpperRegion : Set Point :=
  {p ∈ TRegion | p.y > (1/2) * p.x}

/-- The part of the region below the line -/
def LowerRegion : Set Point :=
  {p ∈ TRegion | p.y < (1/2) * p.x}

/-- The theorem stating that the line y = (1/2)x divides the T-shaped region in half -/
theorem dividing_line_halves_area : 
  area UpperRegion = area LowerRegion := by sorry

end NUMINAMATH_CALUDE_dividing_line_halves_area_l573_57319


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_5_l573_57393

theorem smallest_positive_integer_ending_in_3_divisible_by_5 : 
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 10 = 3 → m % 5 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_5_l573_57393


namespace NUMINAMATH_CALUDE_max_x_minus_y_is_sqrt5_l573_57389

theorem max_x_minus_y_is_sqrt5 (x y : ℝ) (h : x^2 + 2*x*y + y^2 + 4*x^2*y^2 = 4) :
  ∃ (max : ℝ), (∀ (a b : ℝ), a^2 + 2*a*b + b^2 + 4*a^2*b^2 = 4 → a - b ≤ max) ∧ max = Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_is_sqrt5_l573_57389


namespace NUMINAMATH_CALUDE_count_four_digit_numbers_l573_57357

/-- The number of ways to select 3 different digits from 0 to 9 -/
def select_three_digits : ℕ := Nat.choose 10 3

/-- The number of four-digit numbers formed by selecting three different digits from 0 to 9,
    where one digit may appear twice -/
def four_digit_numbers : ℕ := 3888

/-- Theorem stating that the number of four-digit numbers formed by selecting
    three different digits from 0 to 9 (where one digit may appear twice) is 3888 -/
theorem count_four_digit_numbers :
  four_digit_numbers = 3888 :=
by sorry

end NUMINAMATH_CALUDE_count_four_digit_numbers_l573_57357


namespace NUMINAMATH_CALUDE_equation_solution_l573_57383

theorem equation_solution :
  ∃ (square : ℚ),
    (((13/5 : ℚ) - ((17/2 : ℚ) - square) / (7/2 : ℚ)) / ((2 : ℚ) / 15)) = 2 ∧
    square = (1/3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l573_57383


namespace NUMINAMATH_CALUDE_angle_system_solution_l573_57310

theorem angle_system_solution (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (eq1 : 2 * Real.sin (2 * β) = 3 * Real.sin (2 * α))
  (eq2 : Real.tan β = 3 * Real.tan α) :
  α = Real.arctan (Real.sqrt 7 / 7) ∧ β = Real.arctan (3 * Real.sqrt 7 / 7) := by
sorry

end NUMINAMATH_CALUDE_angle_system_solution_l573_57310


namespace NUMINAMATH_CALUDE_modulus_of_one_over_one_minus_i_l573_57313

theorem modulus_of_one_over_one_minus_i :
  let z : ℂ := 1 / (1 - I)
  ‖z‖ = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_one_over_one_minus_i_l573_57313


namespace NUMINAMATH_CALUDE_transformed_area_theorem_l573_57371

-- Define the transformation matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 8, -2]

-- Define the original region's area
def original_area : ℝ := 15

-- Theorem statement
theorem transformed_area_theorem :
  let transformed_area := original_area * abs (Matrix.det A)
  transformed_area = 570 := by sorry

end NUMINAMATH_CALUDE_transformed_area_theorem_l573_57371


namespace NUMINAMATH_CALUDE_nine_chapters_problem_l573_57396

/-- Represents the problem from "The Nine Chapters on the Mathematical Art" -/
theorem nine_chapters_problem (x y : ℤ) : 
  (∀ (z : ℤ), z * x = y → (8 * x - 3 = y ↔ z = 8) ∧ (7 * x + 4 = y ↔ z = 7)) →
  (8 * x - 3 = y ∧ 7 * x + 4 = y) :=
by sorry

end NUMINAMATH_CALUDE_nine_chapters_problem_l573_57396


namespace NUMINAMATH_CALUDE_salary_change_l573_57327

theorem salary_change (x : ℝ) :
  (1 - x / 100) * (1 + x / 100) = 0.75 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_l573_57327


namespace NUMINAMATH_CALUDE_smallest_base_for_256_is_correct_l573_57353

/-- The smallest base in which 256 (decimal) has exactly 4 digits -/
def smallest_base_for_256 : ℕ := 5

/-- Predicate to check if a number has exactly 4 digits in a given base -/
def has_exactly_four_digits (n : ℕ) (base : ℕ) : Prop :=
  base ^ 3 ≤ n ∧ n < base ^ 4

theorem smallest_base_for_256_is_correct :
  (has_exactly_four_digits 256 smallest_base_for_256) ∧
  (∀ b : ℕ, 0 < b → b < smallest_base_for_256 → ¬(has_exactly_four_digits 256 b)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_256_is_correct_l573_57353


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l573_57346

theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 25 * x^2 + n * x + 4 = 0) ↔ n = 20 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l573_57346


namespace NUMINAMATH_CALUDE_orange_juice_serving_size_l573_57347

/-- Calculates the size of each serving of orange juice in ounces. -/
def serving_size (concentrate_cans : ℕ) (water_ratio : ℕ) (concentrate_oz : ℕ) (total_servings : ℕ) : ℚ :=
  let total_cans := concentrate_cans * (water_ratio + 1)
  let total_oz := total_cans * concentrate_oz
  (total_oz : ℚ) / total_servings

/-- Proves that the size of each serving is 6 ounces under the given conditions. -/
theorem orange_juice_serving_size :
  serving_size 34 3 12 272 = 6 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_serving_size_l573_57347


namespace NUMINAMATH_CALUDE_stone_breadth_proof_l573_57312

/-- Given a hall and stones with specific dimensions, prove the breadth of each stone. -/
theorem stone_breadth_proof (hall_length : ℝ) (hall_width : ℝ) (stone_length : ℝ) (stone_count : ℕ) 
  (h1 : hall_length = 36)
  (h2 : hall_width = 15)
  (h3 : stone_length = 0.8)
  (h4 : stone_count = 1350) :
  ∃ (stone_width : ℝ), 
    stone_width = 0.5 ∧ 
    (hall_length * hall_width * 100) = (stone_count : ℝ) * stone_length * stone_width * 100 := by
  sorry


end NUMINAMATH_CALUDE_stone_breadth_proof_l573_57312


namespace NUMINAMATH_CALUDE_max_area_fence_enclosure_l573_57335

/-- Represents a rectangular fence enclosure --/
structure FenceEnclosure where
  length : ℝ
  width : ℝ
  perimeter_eq : length + width = 200
  length_constraint : length ≥ 90
  width_constraint : width ≥ 50
  ratio_constraint : length ≤ 2 * width

/-- The area of a fence enclosure --/
def area (f : FenceEnclosure) : ℝ := f.length * f.width

/-- Theorem stating the maximum area of the fence enclosure --/
theorem max_area_fence_enclosure :
  ∃ (f : FenceEnclosure), ∀ (g : FenceEnclosure), area f ≥ area g ∧ area f = 10000 :=
sorry

end NUMINAMATH_CALUDE_max_area_fence_enclosure_l573_57335


namespace NUMINAMATH_CALUDE_triangle_problem_l573_57342

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- positive side lengths
  0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 →  -- acute angles
  A + B + C = π →  -- angles sum to π
  b = Real.sqrt 2 * a * Real.sin B →  -- given condition
  (A = π/4 ∧ 
   (b = Real.sqrt 6 ∧ c = Real.sqrt 3 + 1 → a = 2)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l573_57342
