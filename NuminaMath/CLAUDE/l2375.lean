import Mathlib

namespace NUMINAMATH_CALUDE_certain_chinese_book_l2375_237526

def total_books : ℕ := 12
def chinese_books : ℕ := 10
def math_books : ℕ := 2
def drawn_books : ℕ := 3

theorem certain_chinese_book :
  ∀ (drawn : Finset ℕ),
    drawn.card = drawn_books →
    drawn ⊆ Finset.range total_books →
    ∃ (book : ℕ), book ∈ drawn ∧ book < chinese_books :=
sorry

end NUMINAMATH_CALUDE_certain_chinese_book_l2375_237526


namespace NUMINAMATH_CALUDE_product_digit_sum_l2375_237573

def repeat_digits (d : ℕ) (n : ℕ) : ℕ :=
  d * (10^(3*n) - 1) / 999

def number1 : ℕ := repeat_digits 400 333
def number2 : ℕ := repeat_digits 606 333

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10
def units_digit (n : ℕ) : ℕ := n % 10

theorem product_digit_sum :
  tens_digit (number1 * number2) + units_digit (number1 * number2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l2375_237573


namespace NUMINAMATH_CALUDE_distribution_schemes_7_5_2_l2375_237591

/-- The number of ways to distribute n identical items among k recipients,
    where two recipients must receive at least m items each. -/
def distribution_schemes (n k m : ℕ) : ℕ :=
  sorry

/-- The specific case for 7 items, 5 recipients, and 2 items minimum for two recipients -/
theorem distribution_schemes_7_5_2 :
  distribution_schemes 7 5 2 = 35 :=
sorry

end NUMINAMATH_CALUDE_distribution_schemes_7_5_2_l2375_237591


namespace NUMINAMATH_CALUDE_largest_two_digit_multiple_of_seven_l2375_237569

def digits : Set Nat := {3, 5, 6, 7}

def is_two_digit (n : Nat) : Prop :=
  10 ≤ n ∧ n < 100

def formed_from_digits (n : Nat) : Prop :=
  ∃ (d1 d2 : Nat), d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ n = 10 * d1 + d2

theorem largest_two_digit_multiple_of_seven :
  ∀ n : Nat, is_two_digit n → formed_from_digits n → n % 7 = 0 →
  n ≤ 63 :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_multiple_of_seven_l2375_237569


namespace NUMINAMATH_CALUDE_largest_expression_l2375_237563

theorem largest_expression : 
  (100 - 0 > 0 / 100) ∧ (100 - 0 > 0 * 100) := by sorry

end NUMINAMATH_CALUDE_largest_expression_l2375_237563


namespace NUMINAMATH_CALUDE_park_area_l2375_237532

/-- Represents a rectangular park with given properties -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  ratio : length / breadth = 1 / 3
  perimeter : length * 2 + breadth * 2 = 1600

/-- The area of the rectangular park is 120000 square meters -/
theorem park_area (park : RectangularPark) : park.length * park.breadth = 120000 := by
  sorry

end NUMINAMATH_CALUDE_park_area_l2375_237532


namespace NUMINAMATH_CALUDE_sin_75_cos_75_l2375_237558

theorem sin_75_cos_75 : Real.sin (75 * π / 180) * Real.cos (75 * π / 180) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_75_cos_75_l2375_237558


namespace NUMINAMATH_CALUDE_fraction_cube_equality_l2375_237522

theorem fraction_cube_equality : (64000 ^ 3 : ℚ) / (16000 ^ 3) = 64 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cube_equality_l2375_237522


namespace NUMINAMATH_CALUDE_spam_email_ratio_l2375_237582

theorem spam_email_ratio (total : ℕ) (important : ℕ) (promotional_fraction : ℚ) 
  (h1 : total = 400)
  (h2 : important = 180)
  (h3 : promotional_fraction = 2/5) :
  (total - important - (total - important) * promotional_fraction : ℚ) / total = 33/100 := by
  sorry

end NUMINAMATH_CALUDE_spam_email_ratio_l2375_237582


namespace NUMINAMATH_CALUDE_student_tickets_sold_l2375_237529

theorem student_tickets_sold (total_tickets : ℕ) (total_money : ℕ) 
  (student_price : ℕ) (nonstudent_price : ℕ) 
  (h1 : total_tickets = 821)
  (h2 : total_money = 1933)
  (h3 : student_price = 2)
  (h4 : nonstudent_price = 3) :
  ∃ (student_tickets : ℕ), 
    student_tickets + (total_tickets - student_tickets) = total_tickets ∧
    student_price * student_tickets + nonstudent_price * (total_tickets - student_tickets) = total_money ∧
    student_tickets = 530 :=
by
  sorry

#check student_tickets_sold

end NUMINAMATH_CALUDE_student_tickets_sold_l2375_237529


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2375_237510

-- Define the set D
def D : Set ℝ := {x | x < -4 ∨ x > 0}

-- Define proposition p
def p (a : ℝ) : Prop := a ∈ D

-- Define proposition q
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 - a*x₀ - a ≤ -3

-- State the theorem
theorem necessary_not_sufficient_condition :
  (∀ a : ℝ, q a → p a) ∧ (∃ a : ℝ, p a ∧ ¬q a) →
  D = {x : ℝ | x < -4 ∨ x > 0} :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l2375_237510


namespace NUMINAMATH_CALUDE_race_length_l2375_237505

/-- Represents the race scenario -/
structure Race where
  length : ℝ
  samTime : ℝ
  johnTime : ℝ
  headStart : ℝ

/-- The race satisfies the given conditions -/
def validRace (r : Race) : Prop :=
  r.samTime = 17 ∧
  r.johnTime = r.samTime + 5 ∧
  r.headStart = 15 ∧
  r.length / r.samTime = (r.length - r.headStart) / r.johnTime

/-- The theorem to be proved -/
theorem race_length (r : Race) (h : validRace r) : r.length = 66 := by
  sorry

end NUMINAMATH_CALUDE_race_length_l2375_237505


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2375_237552

theorem arithmetic_mean_problem (x : ℝ) : (x + 1 = (5 + 7) / 2) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2375_237552


namespace NUMINAMATH_CALUDE_toy_bear_production_efficiency_l2375_237578

theorem toy_bear_production_efficiency (B H : ℝ) (H' : ℝ) : 
  B > 0 → H > 0 →
  (1.8 * B = 2 * (B / H) * H') →
  (H - H') / H * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_toy_bear_production_efficiency_l2375_237578


namespace NUMINAMATH_CALUDE_calculate_expression_l2375_237523

theorem calculate_expression : (-5) / ((1 / 4) - (1 / 3)) * 12 = 720 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2375_237523


namespace NUMINAMATH_CALUDE_ones_divisible_by_l_l2375_237515

theorem ones_divisible_by_l (l : ℕ) (h1 : ¬ 2 ∣ l) (h2 : ¬ 5 ∣ l) :
  ∃ n : ℕ, l ∣ n ∧ ∀ d : ℕ, d ∈ (n.digits 10) → d = 1 :=
sorry

end NUMINAMATH_CALUDE_ones_divisible_by_l_l2375_237515


namespace NUMINAMATH_CALUDE_symmetric_axis_of_shifted_quadratic_unique_symmetric_axis_l2375_237594

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * (x + 3)^2 - 2

-- Define the symmetric axis
def symmetric_axis : ℝ := -3

-- Theorem statement
theorem symmetric_axis_of_shifted_quadratic :
  ∀ x : ℝ, f (symmetric_axis + x) = f (symmetric_axis - x) := by
  sorry

-- The symmetric axis is unique
theorem unique_symmetric_axis :
  ∀ h : ℝ, h ≠ symmetric_axis →
  ∃ x : ℝ, f (h + x) ≠ f (h - x) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_axis_of_shifted_quadratic_unique_symmetric_axis_l2375_237594


namespace NUMINAMATH_CALUDE_volunteer_schedule_lcm_l2375_237586

theorem volunteer_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 10))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_lcm_l2375_237586


namespace NUMINAMATH_CALUDE_probability_of_heart_is_one_third_l2375_237572

/-- A deck of cards with only spades, hearts, and clubs -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (is_valid : cards = suits * cards_per_suit)

/-- The probability of drawing a specific suit from the deck -/
def probability_of_suit (d : Deck) : Rat :=
  d.cards_per_suit / d.cards

theorem probability_of_heart_is_one_third :
  ∀ (d : Deck), d.cards = 39 → d.suits = 3 → d.cards_per_suit = 13 →
  probability_of_suit d = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_heart_is_one_third_l2375_237572


namespace NUMINAMATH_CALUDE_pebble_count_l2375_237561

theorem pebble_count (white_pebbles : ℕ) (red_pebbles : ℕ) : 
  white_pebbles = 20 → 
  red_pebbles = white_pebbles / 2 → 
  white_pebbles + red_pebbles = 30 := by
sorry

end NUMINAMATH_CALUDE_pebble_count_l2375_237561


namespace NUMINAMATH_CALUDE_arithmetic_equalities_l2375_237516

theorem arithmetic_equalities : 
  (-20 + (-14) - (-18) - 13 = -29) ∧ 
  ((-2) * 3 + (-5) - 4 / (-1/2) = -3) ∧ 
  ((-3/8 - 1/6 + 3/4) * (-24) = -5) ∧ 
  (-81 / (9/4) * |(-4/9)| - (-3)^3 / 27 = -15) := by sorry

end NUMINAMATH_CALUDE_arithmetic_equalities_l2375_237516


namespace NUMINAMATH_CALUDE_athlete_heartbeats_l2375_237524

/-- The number of heartbeats during a race --/
def heartbeats_during_race (heart_rate : ℕ) (race_distance : ℕ) (pace : ℕ) : ℕ :=
  heart_rate * race_distance * pace

/-- Proof that the athlete's heart beats 19200 times during the race --/
theorem athlete_heartbeats :
  heartbeats_during_race 160 20 6 = 19200 := by
  sorry

#eval heartbeats_during_race 160 20 6

end NUMINAMATH_CALUDE_athlete_heartbeats_l2375_237524


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l2375_237581

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (5 * z) / (3 * x + y) + (5 * x) / (y + 3 * z) + (2 * y) / (x + z) ≥ 2 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (5 * c) / (3 * a + b) + (5 * a) / (b + 3 * c) + (2 * b) / (a + c) = 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l2375_237581


namespace NUMINAMATH_CALUDE_remainder_sum_mod_13_l2375_237519

theorem remainder_sum_mod_13 (a b c d : ℤ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9) :
  (a + b + c + d) % 13 = 11 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_13_l2375_237519


namespace NUMINAMATH_CALUDE_probability_nine_red_in_eleven_draws_l2375_237584

/-- The probability of drawing exactly 9 red balls in 11 draws, with the 11th draw being red,
    from a bag containing 6 white balls and 3 red balls (with replacement) -/
theorem probability_nine_red_in_eleven_draws :
  let total_balls : ℕ := 9
  let red_balls : ℕ := 3
  let white_balls : ℕ := 6
  let total_draws : ℕ := 11
  let red_draws : ℕ := 9
  let p_red : ℚ := red_balls / total_balls
  let p_white : ℚ := white_balls / total_balls
  Nat.choose (total_draws - 1) (red_draws - 1) * p_red ^ red_draws * p_white ^ (total_draws - red_draws) =
    Nat.choose 10 8 * (1 / 3) ^ 9 * (2 / 3) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_probability_nine_red_in_eleven_draws_l2375_237584


namespace NUMINAMATH_CALUDE_vehicle_inspection_is_systematic_l2375_237570

/-- Represents a vehicle's license plate -/
structure LicensePlate where
  number : Nat

/-- Represents a sampling method -/
inductive SamplingMethod
  | Systematic
  | Other

/-- The criterion for selecting a vehicle based on its license plate -/
def selectionCriterion (plate : LicensePlate) : Bool :=
  plate.number % 10 = 5

/-- The sampling method used in the vehicle inspection process -/
def vehicleInspectionSampling : SamplingMethod :=
  SamplingMethod.Systematic

/-- Theorem stating that the vehicle inspection sampling method is systematic sampling -/
theorem vehicle_inspection_is_systematic :
  vehicleInspectionSampling = SamplingMethod.Systematic :=
sorry

end NUMINAMATH_CALUDE_vehicle_inspection_is_systematic_l2375_237570


namespace NUMINAMATH_CALUDE_total_shot_cost_l2375_237520

-- Define the given conditions
def pregnant_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def shots_per_puppy : ℕ := 2
def cost_per_shot : ℕ := 5

-- Define the theorem
theorem total_shot_cost : 
  pregnant_dogs * puppies_per_dog * shots_per_puppy * cost_per_shot = 120 := by
  sorry

end NUMINAMATH_CALUDE_total_shot_cost_l2375_237520


namespace NUMINAMATH_CALUDE_joe_is_94_point_5_inches_tall_l2375_237576

-- Define the heights of Sara, Joe, and Alex
variable (S J A : ℝ)

-- Define the conditions from the problem
def combined_height : ℝ → ℝ → ℝ → Prop :=
  λ s j a => s + j + a = 180

def joe_height : ℝ → ℝ → Prop :=
  λ s j => j = 2 * s + 6

def alex_height : ℝ → ℝ → Prop :=
  λ s a => a = s - 3

-- Theorem statement
theorem joe_is_94_point_5_inches_tall
  (h1 : combined_height S J A)
  (h2 : joe_height S J)
  (h3 : alex_height S A) :
  J = 94.5 :=
sorry

end NUMINAMATH_CALUDE_joe_is_94_point_5_inches_tall_l2375_237576


namespace NUMINAMATH_CALUDE_notebook_and_pen_prices_l2375_237574

def notebook_price : ℝ := 12
def pen_price : ℝ := 6

theorem notebook_and_pen_prices :
  (2 * notebook_price + pen_price = 30) ∧
  (notebook_price = 2 * pen_price) :=
by sorry

end NUMINAMATH_CALUDE_notebook_and_pen_prices_l2375_237574


namespace NUMINAMATH_CALUDE_f_symmetry_l2375_237547

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + Real.pi^2 * x^2) - Real.pi * x) + Real.pi

theorem f_symmetry (m : ℝ) : f m = 3 → f (-m) = 2 * Real.pi - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetry_l2375_237547


namespace NUMINAMATH_CALUDE_adam_shelf_capacity_l2375_237507

/-- The number of action figures that can fit on each shelf. -/
def figures_per_shelf : ℕ := 9

/-- The number of shelves in Adam's room. -/
def number_of_shelves : ℕ := 3

/-- The total number of action figures that can fit on all shelves. -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

theorem adam_shelf_capacity :
  total_figures = 27 :=
by sorry

end NUMINAMATH_CALUDE_adam_shelf_capacity_l2375_237507


namespace NUMINAMATH_CALUDE_fruit_seller_apples_l2375_237542

theorem fruit_seller_apples (initial_stock : ℕ) (remaining_stock : ℕ) 
  (sell_percentage : ℚ) (h1 : sell_percentage = 40 / 100) 
  (h2 : remaining_stock = 420) 
  (h3 : remaining_stock = initial_stock - (sell_percentage * initial_stock).floor) : 
  initial_stock = 700 := by
sorry

end NUMINAMATH_CALUDE_fruit_seller_apples_l2375_237542


namespace NUMINAMATH_CALUDE_co_captains_probability_l2375_237585

def team_sizes : List Nat := [6, 8, 9, 10]
def num_teams : Nat := 4
def co_captains_per_team : Nat := 3

def probability_co_captains (n : Nat) : Rat :=
  6 / (n * (n - 1) * (n - 2))

theorem co_captains_probability : 
  (1 / num_teams) * (team_sizes.map probability_co_captains).sum = 37 / 1680 := by
  sorry

end NUMINAMATH_CALUDE_co_captains_probability_l2375_237585


namespace NUMINAMATH_CALUDE_target_breaking_permutations_l2375_237501

theorem target_breaking_permutations :
  let total_targets : ℕ := 10
  let column_a_targets : ℕ := 4
  let column_b_targets : ℕ := 4
  let column_c_targets : ℕ := 2
  (column_a_targets + column_b_targets + column_c_targets = total_targets) →
  (Nat.factorial total_targets) / 
  (Nat.factorial column_a_targets * Nat.factorial column_b_targets * Nat.factorial column_c_targets) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_target_breaking_permutations_l2375_237501


namespace NUMINAMATH_CALUDE_fruit_cost_price_l2375_237589

/-- Calculates the total cost price of fruits sold given their selling prices, loss ratios, and quantities. -/
def total_cost_price (apple_sp orange_sp banana_sp : ℚ) 
                     (apple_loss orange_loss banana_loss : ℚ) 
                     (apple_qty orange_qty banana_qty : ℕ) : ℚ :=
  let apple_cp := apple_sp / (1 - apple_loss)
  let orange_cp := orange_sp / (1 - orange_loss)
  let banana_cp := banana_sp / (1 - banana_loss)
  apple_cp * apple_qty + orange_cp * orange_qty + banana_cp * banana_qty

/-- The total cost price of fruits sold is 947.45 given the specified conditions. -/
theorem fruit_cost_price : 
  total_cost_price 18 24 12 (1/6) (1/8) (1/4) 10 15 20 = 947.45 := by
  sorry

#eval total_cost_price 18 24 12 (1/6) (1/8) (1/4) 10 15 20

end NUMINAMATH_CALUDE_fruit_cost_price_l2375_237589


namespace NUMINAMATH_CALUDE_range_of_m_l2375_237577

-- Define propositions P and Q
def P (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*m*x + m ≠ 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 ≥ 0

-- Define the condition that either P or Q is true, and both P and Q are false
def condition (m : ℝ) : Prop := 
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m)

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, condition m ↔ ((-2 ≤ m ∧ m ≤ 0) ∨ (1 ≤ m ∧ m ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2375_237577


namespace NUMINAMATH_CALUDE_sequence_2002nd_term_l2375_237527

theorem sequence_2002nd_term : 
  let sequence : ℕ → ℕ := λ n => n^2 - 1
  sequence 2002 = 4008003 := by sorry

end NUMINAMATH_CALUDE_sequence_2002nd_term_l2375_237527


namespace NUMINAMATH_CALUDE_factorization_equality_l2375_237583

theorem factorization_equality (x : ℝ) : 4 * x^3 - x = x * (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2375_237583


namespace NUMINAMATH_CALUDE_luke_rounds_played_l2375_237543

/-- The number of points Luke scored in total -/
def total_points : ℕ := 154

/-- The number of points Luke gained in each round -/
def points_per_round : ℕ := 11

/-- The number of rounds Luke played -/
def rounds_played : ℕ := total_points / points_per_round

theorem luke_rounds_played :
  rounds_played = 14 :=
by sorry

end NUMINAMATH_CALUDE_luke_rounds_played_l2375_237543


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l2375_237518

theorem largest_divisor_of_five_consecutive_integers (n : ℤ) :
  ∃ (k : ℤ), k * 60 = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ∧
  ∀ (m : ℤ), m > 60 → ¬∃ (j : ℤ), j * m = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_integers_l2375_237518


namespace NUMINAMATH_CALUDE_quadratic_coefficient_difference_l2375_237537

theorem quadratic_coefficient_difference (a b : ℝ) : 
  (∀ x, ax^2 - b*x + 1 = 0 ↔ x = -1/2 ∨ x = 2) → a - b = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_difference_l2375_237537


namespace NUMINAMATH_CALUDE_linear_function_properties_l2375_237571

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 1

-- Define the properties to be proven
theorem linear_function_properties :
  (∀ x y : ℝ, f x - f y = -2 * (x - y)) ∧  -- Slope is -2
  (f 0 = 1) ∧                              -- y-intercept is (0, 1)
  (∃ x y z : ℝ, f x > 0 ∧ x > 0 ∧          -- Passes through first quadrant
               f y < 0 ∧ y > 0 ∧           -- Passes through second quadrant
               f z < 0 ∧ z < 0) ∧          -- Passes through fourth quadrant
  (∀ x y : ℝ, x < y → f x > f y)           -- Slope is negative
  := by sorry

end NUMINAMATH_CALUDE_linear_function_properties_l2375_237571


namespace NUMINAMATH_CALUDE_cone_distance_theorem_l2375_237557

/-- Represents a right circular cone -/
structure RightCircularCone where
  slantHeight : ℝ
  topRadius : ℝ

/-- The shortest distance between two points on a cone's surface -/
def shortestDistance (cone : RightCircularCone) (pointA pointB : ℝ × ℝ) : ℝ := sorry

theorem cone_distance_theorem (cone : RightCircularCone) 
  (h1 : cone.slantHeight = 21)
  (h2 : cone.topRadius = 14) :
  let midpoint : ℝ × ℝ := (cone.slantHeight / 2, 0)
  let oppositePoint : ℝ × ℝ := (cone.slantHeight / 2, cone.topRadius)
  Int.floor (shortestDistance cone midpoint oppositePoint) = 18 := by sorry

end NUMINAMATH_CALUDE_cone_distance_theorem_l2375_237557


namespace NUMINAMATH_CALUDE_inequalities_and_minimum_value_l2375_237592

theorem inequalities_and_minimum_value :
  (∀ a b, a > b ∧ b > 0 → (1 / a : ℝ) < (1 / b)) ∧
  (∀ a b, a > b ∧ b > 0 → a - 1 / a > b - 1 / b) ∧
  (∀ a b, a > b ∧ b > 0 → (2 * a + b) / (a + 2 * b) < a / b) ∧
  (∀ a b, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 → 2 / a + 1 / b ≥ 9 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ 2 / a + 1 / b = 9) :=
by sorry


end NUMINAMATH_CALUDE_inequalities_and_minimum_value_l2375_237592


namespace NUMINAMATH_CALUDE_fraction_equalities_l2375_237508

theorem fraction_equalities : 
  (126 : ℚ) / 84 = 21 / 18 ∧ (268 : ℚ) / 335 = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_equalities_l2375_237508


namespace NUMINAMATH_CALUDE_complex_exp_seven_pi_over_two_eq_i_l2375_237503

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State the theorem
theorem complex_exp_seven_pi_over_two_eq_i :
  cexp (Complex.I * (7 * Real.pi / 2)) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exp_seven_pi_over_two_eq_i_l2375_237503


namespace NUMINAMATH_CALUDE_root_transformation_l2375_237540

/-- Given that s₁, s₂, and s₃ are the roots of x³ - 4x² + 9 = 0,
    prove that 3s₁, 3s₂, and 3s₃ are the roots of x³ - 12x² + 243 = 0 -/
theorem root_transformation (s₁ s₂ s₃ : ℂ) : 
  (s₁^3 - 4*s₁^2 + 9 = 0) ∧ 
  (s₂^3 - 4*s₂^2 + 9 = 0) ∧ 
  (s₃^3 - 4*s₃^2 + 9 = 0) → 
  ((3*s₁)^3 - 12*(3*s₁)^2 + 243 = 0) ∧ 
  ((3*s₂)^3 - 12*(3*s₂)^2 + 243 = 0) ∧ 
  ((3*s₃)^3 - 12*(3*s₃)^2 + 243 = 0) := by
sorry

end NUMINAMATH_CALUDE_root_transformation_l2375_237540


namespace NUMINAMATH_CALUDE_perpendicular_lines_not_both_perpendicular_to_plane_l2375_237551

-- Define the plane α
variable (α : Set (ℝ × ℝ × ℝ))

-- Define lines a and b
variable (a b : Set (ℝ × ℝ × ℝ))

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define what it means for a line to be perpendicular to a plane
def perpendicular_to_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- The theorem
theorem perpendicular_lines_not_both_perpendicular_to_plane :
  a ≠ b →
  perpendicular a b →
  ¬(perpendicular_to_plane a α ∧ perpendicular_to_plane b α) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_not_both_perpendicular_to_plane_l2375_237551


namespace NUMINAMATH_CALUDE_masha_number_is_1001_l2375_237567

/-- Represents the possible operations Vasya could have performed -/
inductive Operation
  | Sum
  | Product

/-- Checks if a number is a valid choice for Sasha or Masha -/
def is_valid_choice (n : ℕ) : Prop :=
  n > 0 ∧ n ≤ 2002

/-- Checks if Sasha can determine Masha's number -/
def sasha_can_determine (a b : ℕ) (op : Operation) : Prop :=
  match op with
  | Operation.Sum => ¬∃ c, is_valid_choice c ∧ c ≠ b ∧ a + c = 2002
  | Operation.Product => ¬∃ c, is_valid_choice c ∧ c ≠ b ∧ a * c = 2002

/-- Checks if Masha can determine Sasha's number -/
def masha_can_determine (a b : ℕ) (op : Operation) : Prop :=
  match op with
  | Operation.Sum => ¬∃ c, is_valid_choice c ∧ c ≠ a ∧ c + b = 2002
  | Operation.Product => ¬∃ c, is_valid_choice c ∧ c ≠ a ∧ c * b = 2002

theorem masha_number_is_1001 (a b : ℕ) (op : Operation) :
  is_valid_choice a →
  is_valid_choice b →
  (op = Operation.Sum → a + b = 2002) →
  (op = Operation.Product → a * b = 2002) →
  ¬(sasha_can_determine a b op) →
  ¬(masha_can_determine a b op) →
  b = 1001 := by
  sorry


end NUMINAMATH_CALUDE_masha_number_is_1001_l2375_237567


namespace NUMINAMATH_CALUDE_no_preimage_range_l2375_237509

def f (x : ℝ) : ℝ := -x^2 + 2*x

theorem no_preimage_range (k : ℝ) : 
  (∀ x : ℝ, f x ≠ k) ↔ k > 1 := by sorry

end NUMINAMATH_CALUDE_no_preimage_range_l2375_237509


namespace NUMINAMATH_CALUDE_average_of_three_liquids_l2375_237534

/-- Given the average of water and milk is 94 liters and there are 100 liters of coffee,
    prove that the average of water, milk, and coffee is 96 liters. -/
theorem average_of_three_liquids (water_milk_avg : ℝ) (coffee : ℝ) :
  water_milk_avg = 94 →
  coffee = 100 →
  (2 * water_milk_avg + coffee) / 3 = 96 := by
sorry

end NUMINAMATH_CALUDE_average_of_three_liquids_l2375_237534


namespace NUMINAMATH_CALUDE_integer_division_l2375_237559

theorem integer_division (x : ℤ) :
  (∃ k : ℤ, (5 * x + 2) = 17 * k) ↔ (∃ m : ℤ, x = 17 * m + 3) :=
by sorry

end NUMINAMATH_CALUDE_integer_division_l2375_237559


namespace NUMINAMATH_CALUDE_hannah_age_l2375_237598

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Represents Hannah and her brothers -/
structure Family where
  hannah : Person
  brothers : List Person

/-- The given conditions of the problem -/
def problem_conditions (f : Family) : Prop :=
  f.brothers.length = 3 ∧
  ∀ b ∈ f.brothers, b.age = 8 ∧
  f.hannah.age = 2 * (f.brothers.map Person.age).sum

/-- The theorem to prove -/
theorem hannah_age (f : Family) :
  problem_conditions f → f.hannah.age = 48 := by
  sorry

end NUMINAMATH_CALUDE_hannah_age_l2375_237598


namespace NUMINAMATH_CALUDE_actual_average_speed_l2375_237528

/-- Given that increasing the speed by 12 miles per hour reduces the time by 1/4,
    prove that the actual average speed is 36 miles per hour. -/
theorem actual_average_speed : 
  ∃ v : ℝ, v > 0 ∧ v / (v + 12) = 3/4 ∧ v = 36 := by sorry

end NUMINAMATH_CALUDE_actual_average_speed_l2375_237528


namespace NUMINAMATH_CALUDE_room_height_calculation_l2375_237525

theorem room_height_calculation (length width diagonal : ℝ) (h_length : length = 12)
    (h_width : width = 8) (h_diagonal : diagonal = 17) :
  ∃ height : ℝ, height = 9 ∧ diagonal^2 = length^2 + width^2 + height^2 :=
by
  sorry

end NUMINAMATH_CALUDE_room_height_calculation_l2375_237525


namespace NUMINAMATH_CALUDE_log_inequality_l2375_237597

theorem log_inequality (n : ℕ+) (k : ℕ) (h : k = (Nat.factors n).card) :
  Real.log n ≥ k * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2375_237597


namespace NUMINAMATH_CALUDE_carpet_price_falls_below_8_at_945_l2375_237553

def initial_price : ℝ := 10.00
def reduction_rate : ℝ := 0.9
def target_price : ℝ := 8.00

def price_after_n_reductions (n : ℕ) : ℝ :=
  initial_price * (reduction_rate ^ n)

theorem carpet_price_falls_below_8_at_945 :
  price_after_n_reductions 3 < target_price ∧
  price_after_n_reductions 2 ≥ target_price :=
by sorry

end NUMINAMATH_CALUDE_carpet_price_falls_below_8_at_945_l2375_237553


namespace NUMINAMATH_CALUDE_cubic_function_extrema_difference_l2375_237500

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 - 3*x + b

/-- The derivative of f with respect to x -/
def f' (a x : ℝ) : ℝ := 3*x^2 + 2*a*x - 3

theorem cubic_function_extrema_difference (a b : ℝ) :
  f' a (-1) = 0 →
  ∃ (x_max x_min : ℝ), 
    (∀ x, f a b x ≤ f a b x_max) ∧ 
    (∀ x, f a b x_min ≤ f a b x) ∧ 
    f a b x_max - f a b x_min = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_extrema_difference_l2375_237500


namespace NUMINAMATH_CALUDE_reeya_fourth_subject_score_l2375_237596

theorem reeya_fourth_subject_score 
  (score1 score2 score3 : ℕ) 
  (average : ℚ) 
  (h1 : score1 = 55)
  (h2 : score2 = 67)
  (h3 : score3 = 76)
  (h4 : average = 67)
  (h5 : ∀ s : ℕ, s ≤ 100) -- Assuming all scores are out of 100
  : ∃ score4 : ℕ, 
    (score1 + score2 + score3 + score4 : ℚ) / 4 = average ∧ 
    score4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_reeya_fourth_subject_score_l2375_237596


namespace NUMINAMATH_CALUDE_no_three_numbers_exist_l2375_237580

theorem no_three_numbers_exist : ¬∃ (a b c : ℕ), 
  a > 1 ∧ b > 1 ∧ c > 1 ∧ 
  (a * a - 1) % b = 0 ∧ (a * a - 1) % c = 0 ∧
  (b * b - 1) % a = 0 ∧ (b * b - 1) % c = 0 ∧
  (c * c - 1) % a = 0 ∧ (c * c - 1) % b = 0 :=
by sorry


end NUMINAMATH_CALUDE_no_three_numbers_exist_l2375_237580


namespace NUMINAMATH_CALUDE_first_number_problem_l2375_237549

theorem first_number_problem (x y : ℤ) (h1 : y = 43) (h2 : x + 2 * y = 124) : x = 38 := by
  sorry

end NUMINAMATH_CALUDE_first_number_problem_l2375_237549


namespace NUMINAMATH_CALUDE_toys_sold_is_eighteen_l2375_237560

/-- The number of toys sold by a man, given the selling price, gain, and cost price per toy. -/
def number_of_toys_sold (selling_price gain cost_per_toy : ℕ) : ℕ :=
  (selling_price - gain) / cost_per_toy

/-- Theorem stating that the number of toys sold is 18 under the given conditions. -/
theorem toys_sold_is_eighteen :
  let selling_price : ℕ := 25200
  let cost_per_toy : ℕ := 1200
  let gain : ℕ := 3 * cost_per_toy
  number_of_toys_sold selling_price gain cost_per_toy = 18 := by
sorry

#eval number_of_toys_sold 25200 (3 * 1200) 1200

end NUMINAMATH_CALUDE_toys_sold_is_eighteen_l2375_237560


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l2375_237566

theorem quadratic_rewrite (j : ℝ) : 
  ∃ (c p q : ℝ), 9 * j^2 - 12 * j + 27 = c * (j + p)^2 + q ∧ q / p = -69 / 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l2375_237566


namespace NUMINAMATH_CALUDE_derivative_at_alpha_l2375_237536

open Real

theorem derivative_at_alpha (α : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ sin α - cos x
  deriv f α = sin α := by sorry

end NUMINAMATH_CALUDE_derivative_at_alpha_l2375_237536


namespace NUMINAMATH_CALUDE_min_sum_grid_l2375_237531

theorem min_sum_grid (a b c d : ℕ+) (h : a * b + c * d + a * c + b * d = 2015) :
  a + b + c + d ≥ 88 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_grid_l2375_237531


namespace NUMINAMATH_CALUDE_absent_laborers_count_l2375_237538

/-- Represents the number of laborers originally employed -/
def total_laborers : ℕ := 20

/-- Represents the original number of days planned to complete the work -/
def original_days : ℕ := 15

/-- Represents the actual number of days taken to complete the work -/
def actual_days : ℕ := 20

/-- Represents the total amount of work in laborer-days -/
def total_work : ℕ := total_laborers * original_days

/-- Calculates the number of absent laborers -/
def absent_laborers : ℕ := total_laborers - (total_work / actual_days)

theorem absent_laborers_count : absent_laborers = 5 := by
  sorry

end NUMINAMATH_CALUDE_absent_laborers_count_l2375_237538


namespace NUMINAMATH_CALUDE_prob_sum_le_10_is_25_72_l2375_237564

/-- The number of possible outcomes when rolling three fair six-sided dice -/
def total_outcomes : ℕ := 6^3

/-- The number of favorable outcomes (sum ≤ 10) when rolling three fair six-sided dice -/
def favorable_outcomes : ℕ := 75

/-- The probability of rolling three fair six-sided dice and obtaining a sum less than or equal to 10 -/
def prob_sum_le_10 : ℚ := favorable_outcomes / total_outcomes

theorem prob_sum_le_10_is_25_72 : prob_sum_le_10 = 25 / 72 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_le_10_is_25_72_l2375_237564


namespace NUMINAMATH_CALUDE_integer_difference_l2375_237595

theorem integer_difference (S L : ℤ) : 
  S = 10 → 
  S + L = 30 → 
  5 * S > 2 * L → 
  5 * S - 2 * L = 10 := by
sorry

end NUMINAMATH_CALUDE_integer_difference_l2375_237595


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_4_range_of_a_l2375_237588

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |x - 1|

-- Theorem for part (I)
theorem solution_set_f_greater_than_4 :
  {x : ℝ | f x > 4} = {x : ℝ | x < -2 ∨ x > 0} := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc (-3/2) 1, a + 1 > f x) → a > 3/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_4_range_of_a_l2375_237588


namespace NUMINAMATH_CALUDE_train_distance_l2375_237546

/-- Given a train traveling at a certain speed for a certain time, 
    calculate the distance covered. -/
theorem train_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
    (h1 : speed = 150) 
    (h2 : time = 8) 
    (h3 : distance = speed * time) : 
  distance = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_distance_l2375_237546


namespace NUMINAMATH_CALUDE_triangle_area_passes_through_1_2_passes_through_neg1_6_x_intercept_correct_y_intercept_correct_l2375_237575

/-- A linear function passing through (1, 2) and (-1, 6) -/
def linear_function (x : ℝ) : ℝ := -2 * x + 4

/-- The x-intercept of the linear function -/
def x_intercept : ℝ := 2

/-- The y-intercept of the linear function -/
def y_intercept : ℝ := 4

/-- Theorem: The area of the triangle formed by the x-intercept, y-intercept, and origin is 4 -/
theorem triangle_area : (1/2 : ℝ) * x_intercept * y_intercept = 4 := by
  sorry

/-- The linear function passes through (1, 2) -/
theorem passes_through_1_2 : linear_function 1 = 2 := by
  sorry

/-- The linear function passes through (-1, 6) -/
theorem passes_through_neg1_6 : linear_function (-1) = 6 := by
  sorry

/-- The x-intercept is correct -/
theorem x_intercept_correct : linear_function x_intercept = 0 := by
  sorry

/-- The y-intercept is correct -/
theorem y_intercept_correct : linear_function 0 = y_intercept := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_passes_through_1_2_passes_through_neg1_6_x_intercept_correct_y_intercept_correct_l2375_237575


namespace NUMINAMATH_CALUDE_intersection_complement_theorem_l2375_237514

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 4}
def B : Set Nat := {4, 5}

theorem intersection_complement_theorem :
  A ∩ (U \ B) = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_theorem_l2375_237514


namespace NUMINAMATH_CALUDE_constant_dot_product_l2375_237587

-- Define an equilateral triangle ABC with side length 2
def Triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2

-- Define a point P on side BC
def PointOnBC (B C P : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • B + t • C

-- Vector dot product
def dot_product (v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1) + (v.2 * w.2)

-- Vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 + w.1, v.2 + w.2)

-- Vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

theorem constant_dot_product
  (A B C P : ℝ × ℝ)
  (h1 : Triangle A B C)
  (h2 : PointOnBC B C P) :
  dot_product (vec_sub P A) (vec_add (vec_sub B A) (vec_sub C A)) = 6 :=
sorry

end NUMINAMATH_CALUDE_constant_dot_product_l2375_237587


namespace NUMINAMATH_CALUDE_different_color_sock_pairs_l2375_237556

theorem different_color_sock_pairs (white : ℕ) (brown : ℕ) (blue : ℕ) : 
  white = 5 → brown = 4 → blue = 3 → 
  (white * brown + brown * blue + white * blue = 47) :=
by
  sorry

end NUMINAMATH_CALUDE_different_color_sock_pairs_l2375_237556


namespace NUMINAMATH_CALUDE_no_rain_probability_l2375_237565

theorem no_rain_probability (p : ℚ) (h : p = 2/3) : (1 - p)^4 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l2375_237565


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l2375_237579

theorem right_triangle_side_length (Q R S : ℝ) (cosR : ℝ) (RS : ℝ) :
  cosR = 3 / 5 →
  RS = 10 →
  (Q - R) * (S - R) = 0 →  -- This represents the right angle at R
  (Q - S) * (Q - S) = 8 * 8 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l2375_237579


namespace NUMINAMATH_CALUDE_exponential_range_condition_l2375_237562

theorem exponential_range_condition (a : ℝ) :
  (∀ x > 0, a^x > 1) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_exponential_range_condition_l2375_237562


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2375_237512

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) (pos_e : e > 0) (pos_f : f > 0)
  (sum_eq_8 : a + b + c + d + e + f = 8) :
  (1 / a + 9 / b + 4 / c + 25 / d + 16 / e + 49 / f) ≥ 1352 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2375_237512


namespace NUMINAMATH_CALUDE_min_bodyguards_tournament_l2375_237554

/-- A tournament where each bodyguard is defeated by at least three others -/
def BodyguardTournament (n : ℕ) := 
  ∃ (defeats : Fin n → Fin n → Prop),
    (∀ i j k : Fin n, i ≠ j → ∃ l : Fin n, defeats l i ∧ defeats l j) ∧
    (∀ i : Fin n, ∃ j k l : Fin n, j ≠ i ∧ k ≠ i ∧ l ≠ i ∧ defeats j i ∧ defeats k i ∧ defeats l i)

/-- The minimum number of bodyguards in a tournament satisfying the conditions is 7 -/
theorem min_bodyguards_tournament : 
  (∃ n : ℕ, BodyguardTournament n) ∧ 
  (∀ m : ℕ, m < 7 → ¬BodyguardTournament m) ∧
  BodyguardTournament 7 :=
sorry

end NUMINAMATH_CALUDE_min_bodyguards_tournament_l2375_237554


namespace NUMINAMATH_CALUDE_average_score_remaining_students_l2375_237504

theorem average_score_remaining_students
  (k : ℕ) 
  (h1 : k > 12)
  (h2 : (k : ℝ) * 8 = k * 8)  -- Total score of all students
  (h3 : (12 : ℝ) * 14 = 168)  -- Total score of 12 students
  : (k * 8 - 168) / (k - 12) = (8 * k - 168) / (k - 12) := by
  sorry

#check average_score_remaining_students

end NUMINAMATH_CALUDE_average_score_remaining_students_l2375_237504


namespace NUMINAMATH_CALUDE_sum_of_cubes_and_cube_of_sum_l2375_237517

theorem sum_of_cubes_and_cube_of_sum : (5 + 7)^3 + (5^3 + 7^3) = 2196 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_and_cube_of_sum_l2375_237517


namespace NUMINAMATH_CALUDE_fair_attendance_l2375_237513

/-- Represents the number of children attending the fair -/
def num_children : ℕ := sorry

/-- Represents the number of adults attending the fair -/
def num_adults : ℕ := sorry

/-- The admission fee for children in cents -/
def child_fee : ℕ := 150

/-- The admission fee for adults in cents -/
def adult_fee : ℕ := 400

/-- The total number of people attending the fair -/
def total_people : ℕ := 2200

/-- The total amount collected in cents -/
def total_amount : ℕ := 505000

theorem fair_attendance : 
  num_children + num_adults = total_people ∧
  num_children * child_fee + num_adults * adult_fee = total_amount →
  num_children = 1500 :=
sorry

end NUMINAMATH_CALUDE_fair_attendance_l2375_237513


namespace NUMINAMATH_CALUDE_second_group_average_l2375_237506

theorem second_group_average (n₁ : ℕ) (n₂ : ℕ) (avg₁ : ℝ) (avg_total : ℝ) :
  n₁ = 30 →
  n₂ = 20 →
  avg₁ = 20 →
  avg_total = 24 →
  ∃ avg₂ : ℝ,
    (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) = avg_total ∧
    avg₂ = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_group_average_l2375_237506


namespace NUMINAMATH_CALUDE_prob_at_least_one_woman_l2375_237599

/-- The probability of selecting at least one woman when choosing 4 people at random from a group of 10 men and 5 women is 29/36. -/
theorem prob_at_least_one_woman (total : ℕ) (men : ℕ) (women : ℕ) (selection : ℕ) : 
  total = 15 → men = 10 → women = 5 → selection = 4 → 
  (1 - (men.choose selection / total.choose selection : ℚ)) = 29/36 := by
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_woman_l2375_237599


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l2375_237530

def club_members : ℕ := 30
def boys : ℕ := 18
def girls : ℕ := 12

theorem president_vice_president_selection :
  (boys * girls) + (girls * boys) = 432 :=
by sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l2375_237530


namespace NUMINAMATH_CALUDE_parabolas_intersect_on_circle_l2375_237511

/-- Two parabolas intersect on a circle -/
theorem parabolas_intersect_on_circle :
  ∃ (center : ℝ × ℝ) (r : ℝ),
    (∀ (x y : ℝ),
      (y = (x - 2)^2 ∧ x + 6 = (y + 1)^2) →
      ((x - center.1)^2 + (y - center.2)^2 = r^2)) ∧
    r^2 = 33/2 := by
  sorry

end NUMINAMATH_CALUDE_parabolas_intersect_on_circle_l2375_237511


namespace NUMINAMATH_CALUDE_june_sweets_l2375_237535

theorem june_sweets (total : ℕ) (june may april : ℚ) : 
  total = 90 ∧ 
  may = (3 / 4) * june ∧ 
  april = (2 / 3) * may ∧ 
  total = april + may + june → 
  june = 40 := by
sorry

end NUMINAMATH_CALUDE_june_sweets_l2375_237535


namespace NUMINAMATH_CALUDE_money_division_l2375_237544

theorem money_division (p q r : ℕ) (total : ℕ) : 
  p * 4 = q * 5 →
  q * 10 = r * 9 →
  r = 400 →
  total = p + q + r →
  total = 1210 := by
sorry

end NUMINAMATH_CALUDE_money_division_l2375_237544


namespace NUMINAMATH_CALUDE_pearl_cutting_theorem_l2375_237555

/-- Represents a string of pearls -/
structure PearlString where
  color : Bool  -- true for black, false for white
  length : Nat
  length_pos : length > 0

/-- The state of the pearl-cutting process -/
structure PearlState where
  strings : List PearlString
  step : Nat

/-- The rules for cutting pearls -/
def cut_pearls (k : Nat) (state : PearlState) : PearlState :=
  sorry

/-- Predicate to check if a white pearl is isolated -/
def has_isolated_white_pearl (state : PearlState) : Prop :=
  sorry

/-- Predicate to check if there's a string of at least two black pearls -/
def has_two_or_more_black_pearls (state : PearlState) : Prop :=
  sorry

/-- The main theorem -/
theorem pearl_cutting_theorem (k b w : Nat) (h1 : k > 0) (h2 : b > w) (h3 : w > 1) :
  ∀ (final_state : PearlState),
    (∃ (initial_state : PearlState),
      initial_state.strings = [PearlString.mk true b sorry, PearlString.mk false w sorry] ∧
      final_state = cut_pearls k initial_state) →
    has_isolated_white_pearl final_state →
    has_two_or_more_black_pearls final_state :=
  sorry

end NUMINAMATH_CALUDE_pearl_cutting_theorem_l2375_237555


namespace NUMINAMATH_CALUDE_larger_number_is_eight_l2375_237548

theorem larger_number_is_eight (x y : ℕ) (h1 : x = 2 * y) (h2 : x * y = 40) (h3 : x + y = 14) : x = 8 := by
  sorry

#check larger_number_is_eight

end NUMINAMATH_CALUDE_larger_number_is_eight_l2375_237548


namespace NUMINAMATH_CALUDE_determinant_example_l2375_237568

/-- Definition of a second-order determinant -/
def second_order_determinant (a b c d : ℝ) : ℝ := a * d - b * c

/-- Theorem: The determinant of the matrix [[2, 1], [-3, 4]] is 11 -/
theorem determinant_example : second_order_determinant 2 (-3) 1 4 = 11 := by
  sorry

end NUMINAMATH_CALUDE_determinant_example_l2375_237568


namespace NUMINAMATH_CALUDE_gcd_3Sn_nplus1_le_1_l2375_237541

def square_sum (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem gcd_3Sn_nplus1_le_1 (n : ℕ+) :
  Nat.gcd (3 * square_sum n) (n + 1) ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_gcd_3Sn_nplus1_le_1_l2375_237541


namespace NUMINAMATH_CALUDE_range_of_a_l2375_237521

theorem range_of_a (a : ℝ) (h_a_pos : a > 0) : 
  (∀ m : ℝ, (3 * a < m ∧ m < 4 * a) → (1 < m ∧ m < 3/2)) →
  (1/3 ≤ a ∧ a ≤ 3/8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2375_237521


namespace NUMINAMATH_CALUDE_prob_A_level_l2375_237533

/-- The probability of producing a B-level product -/
def prob_B : ℝ := 0.03

/-- The probability of producing a C-level product -/
def prob_C : ℝ := 0.01

/-- Theorem: The probability of selecting an A-level product is 0.96 -/
theorem prob_A_level (h1 : prob_B = 0.03) (h2 : prob_C = 0.01) :
  1 - (prob_B + prob_C) = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_level_l2375_237533


namespace NUMINAMATH_CALUDE_sqrt_point_five_equals_sqrt_two_over_two_l2375_237545

theorem sqrt_point_five_equals_sqrt_two_over_two :
  Real.sqrt 0.5 = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_point_five_equals_sqrt_two_over_two_l2375_237545


namespace NUMINAMATH_CALUDE_A_is_uncountable_l2375_237593

-- Define the set A as the closed interval [0, 1]
def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Theorem stating that A is uncountable
theorem A_is_uncountable : ¬ (Countable A) := by
  sorry

end NUMINAMATH_CALUDE_A_is_uncountable_l2375_237593


namespace NUMINAMATH_CALUDE_number_ratio_l2375_237539

theorem number_ratio (x y z : ℝ) (k : ℝ) : 
  y = 2 * x →
  z = k * y →
  (x + y + z) / 3 = 165 →
  y = 90 →
  z / y = 4 := by
sorry

end NUMINAMATH_CALUDE_number_ratio_l2375_237539


namespace NUMINAMATH_CALUDE_f_properties_l2375_237502

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := |x| + m / x - 1

theorem f_properties (m : ℝ) :
  -- 1. Monotonicity when m = 2
  (m = 2 → ∀ x y, x < y ∧ y < 0 → f m x > f m y) ∧
  -- 2. Condition for f(2^x) > 0
  (∀ x, f m (2^x) > 0 ↔ m > 1/4) ∧
  -- 3. Number of zeros
  ((∃! x, f m x = 0) ↔ (m > 1/4 ∨ m < -1/4)) ∧
  ((∃ x y, x ≠ y ∧ f m x = 0 ∧ f m y = 0 ∧ ∀ z, f m z = 0 → z = x ∨ z = y) ↔ (m = 1/4 ∨ m = 0 ∨ m = -1/4)) ∧
  ((∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f m x = 0 ∧ f m y = 0 ∧ f m z = 0 ∧ ∀ w, f m w = 0 → w = x ∨ w = y ∨ w = z) ↔ (0 < m ∧ m < 1/4) ∨ (-1/4 < m ∧ m < 0)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2375_237502


namespace NUMINAMATH_CALUDE_twentieth_base5_is_40_l2375_237550

/-- Converts a decimal number to its base 5 representation -/
def toBase5 (n : ℕ) : ℕ :=
  if n < 5 then n
  else 10 * toBase5 (n / 5) + (n % 5)

/-- The 20th number in base 5 sequence -/
def twentieth_base5 : ℕ := toBase5 20

theorem twentieth_base5_is_40 : twentieth_base5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_base5_is_40_l2375_237550


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l2375_237590

/-- Given a rectangular prism with side areas 15, 10, and 6 (in square inches),
    where the dimension associated with the smallest area is the hypotenuse of a right triangle
    formed by the other two dimensions, prove that the volume of the prism is 30 cubic inches. -/
theorem rectangular_prism_volume (a b c : ℝ) 
  (h1 : a * b = 15)
  (h2 : b * c = 10)
  (h3 : a * c = 6)
  (h4 : c^2 = a^2 + b^2) : 
  a * b * c = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l2375_237590
