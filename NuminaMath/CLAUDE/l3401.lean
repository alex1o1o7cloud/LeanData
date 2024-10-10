import Mathlib

namespace judy_pencil_cost_l3401_340141

/-- Calculates the cost of pencils for a given number of days based on weekly usage and pack price -/
def pencil_cost (weekly_usage : ℕ) (days_per_week : ℕ) (pencils_per_pack : ℕ) (pack_price : ℕ) (total_days : ℕ) : ℕ :=
  let daily_usage := weekly_usage / days_per_week
  let total_pencils := daily_usage * total_days
  let packs_needed := (total_pencils + pencils_per_pack - 1) / pencils_per_pack
  packs_needed * pack_price

theorem judy_pencil_cost : pencil_cost 10 5 30 4 45 = 12 := by
  sorry

#eval pencil_cost 10 5 30 4 45

end judy_pencil_cost_l3401_340141


namespace class_size_from_marking_error_l3401_340186

/-- The number of pupils in a class where a marking error occurred. -/
def num_pupils : ℕ := by sorry

/-- The difference between the incorrectly entered mark and the correct mark. -/
def mark_difference : ℚ := 73 - 65

/-- The increase in class average due to the marking error. -/
def average_increase : ℚ := 1/2

theorem class_size_from_marking_error :
  num_pupils = 16 := by sorry

end class_size_from_marking_error_l3401_340186


namespace product_divisible_by_five_l3401_340168

theorem product_divisible_by_five :
  ∃ k : ℤ, 1495 * 1781 * 1815 * 1999 = 5 * k := by
  sorry

end product_divisible_by_five_l3401_340168


namespace expression_equals_one_l3401_340101

theorem expression_equals_one (a b c k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0)
  (sum_zero : a + b + c = 0) (a_squared : a^2 = k * b^2) :
  (a^2 * b^2) / ((a^2 - b*c) * (b^2 - a*c)) +
  (a^2 * c^2) / ((a^2 - b*c) * (c^2 - a*b)) +
  (b^2 * c^2) / ((b^2 - a*c) * (c^2 - a*b)) = 1 :=
by sorry

end expression_equals_one_l3401_340101


namespace correct_observation_value_l3401_340154

theorem correct_observation_value
  (n : ℕ)
  (initial_mean : ℝ)
  (wrong_value : ℝ)
  (corrected_mean : ℝ)
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : wrong_value = 21)
  (h4 : corrected_mean = 36.54)
  : ∃ (correct_value : ℝ),
    n * corrected_mean = n * initial_mean - wrong_value + correct_value ∧
    correct_value = 48 :=
by sorry

end correct_observation_value_l3401_340154


namespace apple_price_theorem_l3401_340165

/-- The relationship between the selling price and quantity of apples -/
def apple_price_relation (x y : ℝ) : Prop :=
  y = 8 * x

/-- The price increase per kg of apples -/
def price_increase_per_kg : ℝ := 8

theorem apple_price_theorem (x y : ℝ) :
  (∀ (x₁ x₂ : ℝ), x₂ - x₁ = 1 → apple_price_relation x₂ y₂ → apple_price_relation x₁ y₁ → y₂ - y₁ = price_increase_per_kg) →
  apple_price_relation x y :=
sorry

end apple_price_theorem_l3401_340165


namespace trigonometric_equalities_l3401_340172

theorem trigonometric_equalities (α : ℝ) (h : 2 * Real.sin α + Real.cos α = 0) :
  (2 * Real.cos α - Real.sin α) / (Real.sin α + Real.cos α) = 5 ∧
  Real.sin α / (Real.sin α ^ 3 - Real.cos α ^ 3) = 5/3 := by
  sorry

end trigonometric_equalities_l3401_340172


namespace episodes_per_season_l3401_340158

theorem episodes_per_season 
  (days : ℕ) 
  (episodes_per_day : ℕ) 
  (seasons : ℕ) 
  (h1 : days = 10)
  (h2 : episodes_per_day = 6)
  (h3 : seasons = 4)
  (h4 : (days * episodes_per_day) % seasons = 0) : 
  (days * episodes_per_day) / seasons = 15 := by
sorry

end episodes_per_season_l3401_340158


namespace carol_nickels_l3401_340167

/-- Represents the contents of Carol's piggy bank -/
structure PiggyBank where
  quarters : ℕ
  nickels : ℕ
  total_cents : ℕ
  nickel_quarter_diff : nickels = quarters + 7
  total_value : total_cents = 5 * nickels + 25 * quarters

/-- Theorem stating that Carol has 21 nickels in her piggy bank -/
theorem carol_nickels (bank : PiggyBank) (h : bank.total_cents = 455) : bank.nickels = 21 := by
  sorry

end carol_nickels_l3401_340167


namespace dessert_combinations_eq_twelve_l3401_340124

/-- The number of dessert options available -/
def num_desserts : ℕ := 4

/-- The number of courses in the meal -/
def num_courses : ℕ := 2

/-- Function to calculate the number of ways to order the dessert -/
def dessert_combinations : ℕ := num_desserts * (num_desserts - 1)

/-- Theorem stating that the number of ways to order the dessert is 12 -/
theorem dessert_combinations_eq_twelve : dessert_combinations = 12 := by
  sorry

end dessert_combinations_eq_twelve_l3401_340124


namespace initial_peaches_proof_l3401_340166

/-- The number of peaches Mike picked from the orchard -/
def peaches_picked : ℕ := 52

/-- The total number of peaches Mike has now -/
def total_peaches : ℕ := 86

/-- The initial number of peaches at Mike's roadside fruit dish -/
def initial_peaches : ℕ := total_peaches - peaches_picked

theorem initial_peaches_proof : initial_peaches = 34 := by
  sorry

end initial_peaches_proof_l3401_340166


namespace container_volume_ratio_l3401_340116

theorem container_volume_ratio : 
  ∀ (v1 v2 v3 : ℝ), 
    v1 > 0 → v2 > 0 → v3 > 0 →
    (2/3 : ℝ) * v1 = (1/2 : ℝ) * v2 →
    (1/2 : ℝ) * v2 = (3/5 : ℝ) * v3 →
    v1 / v3 = 6/5 := by
  sorry

end container_volume_ratio_l3401_340116


namespace percentage_relation_l3401_340177

theorem percentage_relation (A B C x y : ℝ) : 
  A > 0 ∧ B > 0 ∧ C > 0 →
  A = B * (1 + x / 100) →
  A = C * (1 - y / 100) →
  A = 120 →
  B = 100 →
  C = 150 →
  x = 20 ∧ y = 20 := by
sorry

end percentage_relation_l3401_340177


namespace second_group_size_l3401_340147

theorem second_group_size :
  ∀ (n : ℕ), 
  -- First group has 20 students with average height 20 cm
  (20 : ℝ) * 20 = 400 ∧
  -- Second group has n students with average height 20 cm
  (n : ℝ) * 20 = 20 * n ∧
  -- Combined group has 31 students with average height 20 cm
  (31 : ℝ) * 20 = 620 ∧
  -- Total height of combined groups equals sum of individual group heights
  400 + 20 * n = 620
  →
  n = 11 := by
sorry

end second_group_size_l3401_340147


namespace function_non_negative_range_l3401_340109

theorem function_non_negative_range (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x = x^2 - 4*x + a) →
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) →
  a ∈ Set.Ici 3 := by
sorry

end function_non_negative_range_l3401_340109


namespace equation_solution_l3401_340136

theorem equation_solution (t : ℝ) :
  (8.410 * Real.sqrt 3 * Real.sin t - Real.sqrt (2 * Real.sin t ^ 2 - Real.sin (2 * t) + 3 * Real.cos t ^ 2) = 0) ↔
  (∃ k : ℤ, t = π / 4 + 2 * k * π ∨ t = -Real.arctan 3 + π * (2 * k + 1)) :=
by sorry

end equation_solution_l3401_340136


namespace difference_of_squares_l3401_340106

theorem difference_of_squares (a b : ℝ) : (a + b) * (b - a) = b^2 - a^2 := by
  sorry

end difference_of_squares_l3401_340106


namespace constant_value_proof_l3401_340131

theorem constant_value_proof (x y : ℝ) (a : ℝ) 
  (h1 : (a * x + 8 * y) / (x - 2 * y) = 29)
  (h2 : x / (2 * y) = 3 / 2) : 
  a = 7 := by sorry

end constant_value_proof_l3401_340131


namespace profit_distribution_l3401_340199

theorem profit_distribution (total_profit : ℝ) (num_employees : ℕ) (employee_share : ℝ) :
  total_profit = 50 →
  num_employees = 9 →
  employee_share = 5 →
  (total_profit - num_employees * employee_share) / total_profit * 100 = 10 := by
  sorry

end profit_distribution_l3401_340199


namespace arithmetic_sequence_common_difference_l3401_340110

/-- Proves that in an arithmetic sequence with a₁ = 2 and a₃ = 8, the common difference is 3 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (h1 : a 1 = 2)  -- First term is 2
  (h3 : a 3 = 8)  -- Third term is 8
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- Definition of arithmetic sequence
  : a 2 - a 1 = 3 :=
by sorry

end arithmetic_sequence_common_difference_l3401_340110


namespace supplementary_angle_theorem_l3401_340138

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define addition for Angle
def Angle.add (a b : Angle) : Angle :=
  let totalMinutes := a.minutes + b.minutes
  let extraDegrees := totalMinutes / 60
  { degrees := a.degrees + b.degrees + extraDegrees,
    minutes := totalMinutes % 60 }

-- Define subtraction for Angle
def Angle.sub (a b : Angle) : Angle :=
  let totalMinutes := (a.degrees * 60 + a.minutes) - (b.degrees * 60 + b.minutes)
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60 }

-- Define the given complementary angle
def complementaryAngle : Angle := { degrees := 54, minutes := 38 }

-- Define 90 degrees
def rightAngle : Angle := { degrees := 90, minutes := 0 }

-- Define 180 degrees
def straightAngle : Angle := { degrees := 180, minutes := 0 }

-- Theorem statement
theorem supplementary_angle_theorem :
  let angle := Angle.sub rightAngle complementaryAngle
  Angle.sub straightAngle angle = { degrees := 144, minutes := 38 } := by sorry

end supplementary_angle_theorem_l3401_340138


namespace probability_sum_to_15_l3401_340191

/-- Represents a standard playing card --/
inductive Card
| Number (n : Nat)
| Face
| Ace

/-- A standard 52-card deck --/
def Deck : Finset Card := sorry

/-- The set of number cards (2 through 10) in a standard deck --/
def NumberCards : Finset Card := sorry

/-- The number of ways to choose two cards that sum to 15 from number cards --/
def SumTo15Ways : Nat := sorry

/-- The total number of ways to choose two cards from a 52-card deck --/
def TotalWays : Nat := sorry

/-- The probability of selecting two number cards that sum to 15 --/
theorem probability_sum_to_15 :
  (SumTo15Ways : ℚ) / TotalWays = 16 / 884 := by sorry

end probability_sum_to_15_l3401_340191


namespace assignment_schemes_l3401_340195

def total_students : ℕ := 6
def selected_students : ℕ := 4
def restricted_students : ℕ := 2
def restricted_tasks : ℕ := 1

theorem assignment_schemes :
  (total_students.factorial / (total_students - selected_students).factorial) -
  (restricted_students * (total_students - 1).factorial / (total_students - selected_students).factorial) = 240 :=
sorry

end assignment_schemes_l3401_340195


namespace hoseok_payment_l3401_340140

/-- The price of item (a) bought by Hoseok at the mart -/
def item_price : ℕ := 7450

/-- The number of 1000 won bills used -/
def bills_1000 : ℕ := 7

/-- The number of 100 won coins used -/
def coins_100 : ℕ := 4

/-- The number of 10 won coins used -/
def coins_10 : ℕ := 5

/-- The denomination of the bills used -/
def bill_value : ℕ := 1000

/-- The denomination of the first type of coins used -/
def coin_value_100 : ℕ := 100

/-- The denomination of the second type of coins used -/
def coin_value_10 : ℕ := 10

theorem hoseok_payment :
  item_price = bills_1000 * bill_value + coins_100 * coin_value_100 + coins_10 * coin_value_10 :=
by sorry

end hoseok_payment_l3401_340140


namespace infinitely_many_pairs_satisfying_conditions_l3401_340128

theorem infinitely_many_pairs_satisfying_conditions :
  ∃ (a : ℕ → ℕ), (∀ n : ℕ,
    (Nat.gcd (a n) (a (n + 1)) = 1) ∧
    ((a n) ∣ ((a (n + 1))^2 - 5)) ∧
    ((a (n + 1)) ∣ ((a n)^2 - 5))) :=
sorry

end infinitely_many_pairs_satisfying_conditions_l3401_340128


namespace ball_count_proof_l3401_340103

/-- Proves that given 9 yellow balls in a box and a 30% probability of drawing a yellow ball,
    the total number of balls in the box is 30. -/
theorem ball_count_proof (yellow_balls : ℕ) (probability : ℚ) (total_balls : ℕ) : 
  yellow_balls = 9 → probability = 3/10 → (yellow_balls : ℚ) / total_balls = probability → total_balls = 30 := by
  sorry

end ball_count_proof_l3401_340103


namespace correct_day_is_thursday_l3401_340102

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define the statements made by each person
def statement_A (today : DayOfWeek) : Prop := today = DayOfWeek.Friday
def statement_B (today : DayOfWeek) : Prop := today = DayOfWeek.Wednesday
def statement_C (today : DayOfWeek) : Prop := ¬(statement_A today ∨ statement_B today)
def statement_D (today : DayOfWeek) : Prop := today ≠ DayOfWeek.Thursday

-- Define the condition that only one statement is correct
def only_one_correct (today : DayOfWeek) : Prop :=
  (statement_A today ∧ ¬statement_B today ∧ ¬statement_C today ∧ ¬statement_D today) ∨
  (¬statement_A today ∧ statement_B today ∧ ¬statement_C today ∧ ¬statement_D today) ∨
  (¬statement_A today ∧ ¬statement_B today ∧ statement_C today ∧ ¬statement_D today) ∨
  (¬statement_A today ∧ ¬statement_B today ∧ ¬statement_C today ∧ statement_D today)

-- Theorem stating that Thursday is the only day satisfying all conditions
theorem correct_day_is_thursday :
  ∃! today : DayOfWeek, only_one_correct today ∧ today = DayOfWeek.Thursday :=
sorry

end correct_day_is_thursday_l3401_340102


namespace coin_difference_l3401_340184

def coin_values : List Nat := [5, 15, 20]

def target_amount : Nat := 50

def min_coins (values : List Nat) (target : Nat) : Nat :=
  sorry

def max_coins (values : List Nat) (target : Nat) : Nat :=
  sorry

theorem coin_difference :
  max_coins coin_values target_amount - min_coins coin_values target_amount = 6 :=
by sorry

end coin_difference_l3401_340184


namespace equal_probabilities_l3401_340161

/-- Represents a box containing colored balls -/
structure Box where
  red_balls : ℕ
  green_balls : ℕ

/-- Represents the state of both boxes -/
structure BoxState where
  red_box : Box
  green_box : Box

/-- Initial state of the boxes -/
def initial_state : BoxState :=
  { red_box := { red_balls := 100, green_balls := 0 },
    green_box := { red_balls := 0, green_balls := 100 } }

/-- State after first transfer (8 red balls from red to green box) -/
def first_transfer (state : BoxState) : BoxState :=
  { red_box := { red_balls := state.red_box.red_balls - 8, green_balls := state.red_box.green_balls },
    green_box := { red_balls := state.green_box.red_balls + 8, green_balls := state.green_box.green_balls } }

/-- Probability of drawing a specific color from a box -/
def prob_draw (box : Box) (color : String) : ℚ :=
  match color with
  | "red" => box.red_balls / (box.red_balls + box.green_balls)
  | "green" => box.green_balls / (box.red_balls + box.green_balls)
  | _ => 0

/-- Theorem stating the equality of probabilities after transfers and mixing -/
theorem equal_probabilities (final_state : BoxState) 
    (h1 : final_state.red_box.green_balls + final_state.green_box.green_balls = 100) 
    (h2 : final_state.red_box.red_balls + final_state.green_box.red_balls = 100) :
    prob_draw final_state.red_box "green" = prob_draw final_state.green_box "red" :=
  sorry


end equal_probabilities_l3401_340161


namespace absolute_difference_in_terms_of_sum_and_product_l3401_340143

theorem absolute_difference_in_terms_of_sum_and_product (x₁ x₂ a b : ℝ) 
  (h_sum : x₁ + x₂ = a) (h_product : x₁ * x₂ = b) : 
  |x₁ - x₂| = Real.sqrt (a^2 - 4*b) := by
  sorry

end absolute_difference_in_terms_of_sum_and_product_l3401_340143


namespace sqrt_sum_squares_geq_sqrt2_sum_l3401_340120

theorem sqrt_sum_squares_geq_sqrt2_sum (a b : ℝ) :
  Real.sqrt (a^2 + b^2) ≥ (Real.sqrt 2 / 2) * (a + b) := by
  sorry

end sqrt_sum_squares_geq_sqrt2_sum_l3401_340120


namespace multiply_powers_l3401_340181

theorem multiply_powers (a : ℝ) : 6 * a^2 * (1/2 * a^3) = 3 * a^5 := by
  sorry

end multiply_powers_l3401_340181


namespace arccos_equation_solution_l3401_340148

theorem arccos_equation_solution :
  ∃! x : ℝ, Real.arccos (2 * x) - Real.arccos x = π / 3 ∧ x = -1/2 := by
  sorry

end arccos_equation_solution_l3401_340148


namespace parabola_equation_l3401_340163

/-- A parabola with axis of symmetry x = -2 has the standard form equation y² = 8x -/
theorem parabola_equation (p : ℝ) (h : p > 0) :
  (∀ x y : ℝ, y^2 = 2*p*x) → -- Standard form of parabola
  (-p/2 = -2) →             -- Axis of symmetry
  (∀ x y : ℝ, y^2 = 8*x) :=  -- Resulting equation
by sorry

end parabola_equation_l3401_340163


namespace herd_division_l3401_340118

theorem herd_division (herd : ℚ) : 
  (1/3 : ℚ) + (1/6 : ℚ) + (1/9 : ℚ) + (8 : ℚ) / herd = 1 → 
  herd = 144/7 := by
  sorry

end herd_division_l3401_340118


namespace triangles_from_points_on_circle_l3401_340125

def points_on_circle : ℕ := 10
def vertices_per_triangle : ℕ := 3

theorem triangles_from_points_on_circle :
  Nat.choose points_on_circle vertices_per_triangle = 120 := by
  sorry

end triangles_from_points_on_circle_l3401_340125


namespace constant_function_no_monotonicity_l3401_340198

open Function Set

theorem constant_function_no_monotonicity 
  {f : ℝ → ℝ} {I : Set ℝ} (hI : Interval I) :
  (∀ x ∈ I, HasDerivAt f (0 : ℝ) x) → 
  ∃ c, ∀ x ∈ I, f x = c :=
sorry

end constant_function_no_monotonicity_l3401_340198


namespace field_trip_attendance_l3401_340135

/-- The number of people who went on the field trip -/
def total_people (num_vans : ℕ) (num_buses : ℕ) (people_per_van : ℕ) (people_per_bus : ℕ) : ℕ :=
  num_vans * people_per_van + num_buses * people_per_bus

/-- Theorem stating that the total number of people on the field trip is 342 -/
theorem field_trip_attendance : total_people 9 10 8 27 = 342 := by
  sorry

end field_trip_attendance_l3401_340135


namespace equal_profit_percentage_l3401_340159

def shopkeeper_profit (total_quantity : ℝ) (portion1_percentage : ℝ) (portion2_percentage : ℝ) (profit_percentage : ℝ) : Prop :=
  portion1_percentage + portion2_percentage = 100 ∧
  portion1_percentage > 0 ∧
  portion2_percentage > 0 ∧
  profit_percentage ≥ 0

theorem equal_profit_percentage 
  (total_quantity : ℝ) 
  (portion1_percentage : ℝ) 
  (portion2_percentage : ℝ) 
  (total_profit_percentage : ℝ) 
  (h : shopkeeper_profit total_quantity portion1_percentage portion2_percentage total_profit_percentage) :
  ∃ (individual_profit_percentage : ℝ),
    individual_profit_percentage = total_profit_percentage ∧
    individual_profit_percentage * portion1_percentage / 100 + 
    individual_profit_percentage * portion2_percentage / 100 = 
    total_profit_percentage := by
  sorry

end equal_profit_percentage_l3401_340159


namespace inequality_holds_iff_l3401_340176

theorem inequality_holds_iff (m : ℝ) :
  (∀ x : ℝ, (x^2 + m*x - 1) / (2*x^2 - 2*x + 3) < 1) ↔ m > -6 ∧ m < 2 := by
  sorry

end inequality_holds_iff_l3401_340176


namespace twentieth_fisherman_catch_l3401_340196

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_nineteen : ℕ) 
  (h1 : total_fishermen = 20)
  (h2 : total_fish = 10000)
  (h3 : fish_per_nineteen = 400) :
  total_fish - (total_fishermen - 1) * fish_per_nineteen = 2400 := by
  sorry

end twentieth_fisherman_catch_l3401_340196


namespace min_yacht_capacity_l3401_340129

/-- Represents the number of sheikhs --/
def num_sheikhs : ℕ := 10

/-- Represents the number of wives per sheikh --/
def wives_per_sheikh : ℕ := 100

/-- Represents the total number of wives --/
def total_wives : ℕ := num_sheikhs * wives_per_sheikh

/-- Represents the law: a woman must not be with a man other than her husband unless her husband is present --/
def law_compliant (n : ℕ) : Prop :=
  ∀ (women_on_bank : ℕ) (men_on_bank : ℕ),
    women_on_bank ≤ total_wives ∧ men_on_bank ≤ num_sheikhs →
    (women_on_bank ≤ n ∨ men_on_bank = num_sheikhs ∨ women_on_bank = 0)

/-- Theorem stating that the smallest yacht capacity that allows all sheikhs and wives to cross the river while complying with the law is 10 --/
theorem min_yacht_capacity :
  ∃ (n : ℕ), n = 10 ∧ law_compliant n ∧ ∀ (m : ℕ), m < n → ¬law_compliant m :=
sorry

end min_yacht_capacity_l3401_340129


namespace population_growth_duration_l3401_340179

/-- Proves that given specific population growth rates and a total net increase,
    the duration of the period is 24 hours. -/
theorem population_growth_duration :
  let birth_rate : ℕ := 3  -- people per second
  let death_rate : ℕ := 1  -- people per second
  let net_increase_rate : ℕ := birth_rate - death_rate
  let total_net_increase : ℕ := 172800
  let duration_seconds : ℕ := total_net_increase / net_increase_rate
  let seconds_per_hour : ℕ := 3600
  duration_seconds / seconds_per_hour = 24 := by
  sorry

end population_growth_duration_l3401_340179


namespace line_l_equation_and_symmetric_points_l3401_340155

/-- Parabola defined by y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Line l intersecting the parabola -/
def Line_l : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (m b : ℝ), p.2 = m * p.1 + b}

/-- Point P that bisects segment AB -/
def P : ℝ × ℝ := (2, 2)

/-- A and B are points where line l intersects the parabola -/
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

theorem line_l_equation_and_symmetric_points :
  (∀ p ∈ Line_l, 2 * p.1 - p.2 - 2 = 0) ∧
  (∃ (C D : ℝ × ℝ), C ∈ Parabola ∧ D ∈ Parabola ∧
    (∀ p ∈ Line_l, (C.1 + D.1) * p.2 = (C.2 + D.2) * p.1 + C.1 * D.2 - C.2 * D.1) ∧
    (∀ p ∈ {p : ℝ × ℝ | p.1 + 2 * p.2 - 19 = 0}, p = C ∨ p = D)) :=
sorry

end line_l_equation_and_symmetric_points_l3401_340155


namespace dividend_divisor_product_l3401_340153

theorem dividend_divisor_product (d : ℤ) (D : ℤ) : 
  D = d + 78 → D = 6 * d + 3 → D * d = 1395 := by
sorry

end dividend_divisor_product_l3401_340153


namespace angle_in_first_or_third_quadrant_l3401_340107

/-- Represents the four quadrants in a coordinate system -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines the quadrant of an angle given in degrees -/
def angle_quadrant (α : ℝ) : Quadrant :=
  sorry

/-- Theorem: For any integer k, the angle α = k·180° + 45° lies in either the first or third quadrant -/
theorem angle_in_first_or_third_quadrant (k : ℤ) :
  let α := k * 180 + 45
  (angle_quadrant α = Quadrant.first) ∨ (angle_quadrant α = Quadrant.third) :=
sorry

end angle_in_first_or_third_quadrant_l3401_340107


namespace total_prime_factors_is_27_l3401_340113

/-- The total number of prime factors in the expression (4)^11 * (7)^3 * (11)^2 -/
def totalPrimeFactors : ℕ :=
  let four_factorization := 2 * 2
  let four_exponent := 11
  let seven_exponent := 3
  let eleven_exponent := 2
  (four_factorization * four_exponent) + seven_exponent + eleven_exponent

/-- Theorem stating that the total number of prime factors in the given expression is 27 -/
theorem total_prime_factors_is_27 : totalPrimeFactors = 27 := by
  sorry

end total_prime_factors_is_27_l3401_340113


namespace price_reduction_percentage_l3401_340134

theorem price_reduction_percentage (original_price reduction_amount : ℝ) : 
  original_price = 500 → 
  reduction_amount = 250 → 
  (reduction_amount / original_price) * 100 = 50 :=
by sorry

end price_reduction_percentage_l3401_340134


namespace pebble_ratio_l3401_340170

def total_pebbles : ℕ := 30
def white_pebbles : ℕ := 20

def red_pebbles : ℕ := total_pebbles - white_pebbles

theorem pebble_ratio : 
  (red_pebbles : ℚ) / white_pebbles = 1 / 2 := by sorry

end pebble_ratio_l3401_340170


namespace events_mutually_exclusive_but_not_contradictory_l3401_340192

-- Define the bag contents
def total_balls : ℕ := 4
def red_balls : ℕ := 2
def black_balls : ℕ := 2

-- Define the events
def exactly_one_black (drawn : ℕ) : Prop := drawn = 1
def exactly_two_black (drawn : ℕ) : Prop := drawn = 2

-- Define mutual exclusivity
def mutually_exclusive (event1 event2 : ℕ → Prop) : Prop :=
  ∀ n, ¬(event1 n ∧ event2 n)

-- Define non-contradictory
def non_contradictory (event1 event2 : ℕ → Prop) : Prop :=
  ∃ n, event1 n ∨ event2 n

-- Theorem statement
theorem events_mutually_exclusive_but_not_contradictory :
  mutually_exclusive exactly_one_black exactly_two_black ∧
  non_contradictory exactly_one_black exactly_two_black :=
sorry

end events_mutually_exclusive_but_not_contradictory_l3401_340192


namespace james_carrot_sticks_l3401_340183

theorem james_carrot_sticks (before after total : ℕ) : 
  before = 22 → after = 15 → total = before + after → total = 37 := by sorry

end james_carrot_sticks_l3401_340183


namespace charlottes_distance_l3401_340182

/-- The distance between Charlotte's home and school -/
def distance : ℝ := 60

/-- The time taken for Charlotte's one-way journey in hours -/
def journey_time : ℝ := 6

/-- Charlotte's average speed in miles per hour -/
def average_speed : ℝ := 10

/-- Theorem stating that the distance is equal to the product of average speed and journey time -/
theorem charlottes_distance : distance = average_speed * journey_time := by
  sorry

end charlottes_distance_l3401_340182


namespace tooth_fairy_calculation_l3401_340151

theorem tooth_fairy_calculation (total_amount : ℕ) (total_teeth : ℕ) (lost_teeth : ℕ) (first_tooth_amount : ℕ) :
  total_teeth = 20 →
  total_amount = 54 →
  lost_teeth = 2 →
  first_tooth_amount = 20 →
  (total_amount - first_tooth_amount) / (total_teeth - lost_teeth - 1) = 2 :=
by sorry

end tooth_fairy_calculation_l3401_340151


namespace unique_symmetric_matrix_condition_l3401_340100

/-- A symmetric 2x2 matrix with real entries -/
structure SymmetricMatrix2x2 where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The trace of a symmetric 2x2 matrix -/
def trace (M : SymmetricMatrix2x2) : ℝ := M.x + M.z

/-- The determinant of a symmetric 2x2 matrix -/
def det (M : SymmetricMatrix2x2) : ℝ := M.x * M.z - M.y * M.y

/-- The main theorem -/
theorem unique_symmetric_matrix_condition (a b : ℝ) :
  (∃! M : SymmetricMatrix2x2, trace M = a ∧ det M = b) ↔ ∃ t : ℝ, a = 2 * t ∧ b = t ^ 2 := by
  sorry

end unique_symmetric_matrix_condition_l3401_340100


namespace committee_formation_ways_l3401_340117

theorem committee_formation_ways (n m : ℕ) (hn : n = 10) (hm : m = 4) : 
  Nat.choose n m = 210 := by
  sorry

end committee_formation_ways_l3401_340117


namespace specific_ellipse_equation_l3401_340119

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  /-- The focal length of the ellipse -/
  focal_length : ℝ
  /-- The x-coordinate of one directrix -/
  directrix_x : ℝ

/-- The equation of an ellipse -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 8 + y^2 / 4 = 1

/-- Theorem stating the equation of the specific ellipse -/
theorem specific_ellipse_equation (e : Ellipse) 
  (h1 : e.focal_length = 4)
  (h2 : e.directrix_x = -4) :
  ∀ x y : ℝ, ellipse_equation e x y := by sorry

end specific_ellipse_equation_l3401_340119


namespace two_color_distance_l3401_340187

/-- A type representing colors --/
inductive Color
| Red
| Blue

/-- A two-coloring of the plane --/
def Coloring := ℝ × ℝ → Color

/-- Predicate to check if both colors are used in a coloring --/
def BothColorsUsed (c : Coloring) : Prop :=
  (∃ p : ℝ × ℝ, c p = Color.Red) ∧ (∃ p : ℝ × ℝ, c p = Color.Blue)

/-- The main theorem --/
theorem two_color_distance (c : Coloring) (h : BothColorsUsed c) (a : ℝ) (ha : a > 0) :
  ∃ p q : ℝ × ℝ, c p ≠ c q ∧ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = a :=
sorry

end two_color_distance_l3401_340187


namespace find_s_value_l3401_340133

/-- Given a relationship between R, S, and T, prove the value of S for specific R and T -/
theorem find_s_value (c : ℝ) (R S T : ℝ → ℝ) :
  (∀ x, R x = c * (S x / T x)) →  -- Relationship between R, S, and T
  R 1 = 2 →                       -- Initial condition for R
  S 1 = 1/2 →                     -- Initial condition for S
  T 1 = 4/3 →                     -- Initial condition for T
  R 2 = Real.sqrt 75 →            -- New condition for R
  T 2 = Real.sqrt 32 →            -- New condition for T
  S 2 = 45/4 :=                   -- Conclusion: value of S
by sorry

end find_s_value_l3401_340133


namespace cubic_root_theorem_l3401_340144

-- Define the cubic root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the polynomial
def f (p q : ℚ) (x : ℝ) : ℝ := x^3 + p*x^2 + q*x + 45

-- State the theorem
theorem cubic_root_theorem (p q : ℚ) :
  f p q (2 - 3 * cubeRoot 5) = 0 → p = -6 := by
  sorry

end cubic_root_theorem_l3401_340144


namespace circle_center_and_radius_l3401_340137

/-- Given a circle with equation x^2 + y^2 - 2x + 4y + 3 = 0, 
    its center is at (1, -2) and its radius is √2 -/
theorem circle_center_and_radius 
  (x y : ℝ) 
  (h : x^2 + y^2 - 2*x + 4*y + 3 = 0) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
    center = (1, -2) ∧ 
    radius = Real.sqrt 2 ∧
    (x - center.1)^2 + (y - center.2)^2 = radius^2 := by
  sorry

end circle_center_and_radius_l3401_340137


namespace right_triangle_perimeter_l3401_340108

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 90 →
  a = 18 →
  a^2 + b^2 = c^2 →
  a + b + c = 28 + 2 * Real.sqrt 106 := by
sorry

end right_triangle_perimeter_l3401_340108


namespace boys_to_girls_ratio_l3401_340121

theorem boys_to_girls_ratio (T : ℚ) (G : ℚ) (h : T > 0) (h1 : G > 0) (h2 : (1/2) * G = (1/6) * T) :
  (T - G) / G = 2 := by
  sorry

end boys_to_girls_ratio_l3401_340121


namespace machine_production_rate_l3401_340164

/-- The number of shirts a machine can make in one minute, given the total number of shirts and total time -/
def shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) : ℚ :=
  total_shirts / total_minutes

/-- Theorem stating that the machine makes 7 shirts per minute -/
theorem machine_production_rate :
  shirts_per_minute 196 28 = 7 := by
  sorry

end machine_production_rate_l3401_340164


namespace reflection_line_sum_l3401_340188

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line is (10,6),
    then m + b = 107/6 -/
theorem reflection_line_sum (m b : ℚ) : 
  (∃ (x y : ℚ), 
    -- Midpoint of original and reflected point lies on the line
    (x + 2)/2 = 6 ∧ (y + 3)/2 = 9/2 ∧ y = m*x + b ∧
    -- Perpendicular slope condition
    (x - 2)*(10 - 2) + (y - 3)*(6 - 3) = 0 ∧
    -- Distance equality condition
    (x - 2)^2 + (y - 3)^2 = (10 - x)^2 + (6 - y)^2) →
  m + b = 107/6 := by
sorry

end reflection_line_sum_l3401_340188


namespace max_angle_is_90_deg_l3401_340169

/-- A regular quadrilateral prism with height half the side length of its base -/
structure RegularQuadPrism where
  base_side : ℝ
  height : ℝ
  height_eq_half_base : height = base_side / 2

/-- A point on the edge AB of the prism -/
def PointOnAB (prism : RegularQuadPrism) := {x : ℝ // 0 ≤ x ∧ x ≤ prism.base_side}

/-- The angle A₁MC₁ where M is a point on AB -/
def angleA1MC1 (prism : RegularQuadPrism) (m : PointOnAB prism) : ℝ := sorry

/-- The maximum value of angle A₁MC₁ is 90° -/
theorem max_angle_is_90_deg (prism : RegularQuadPrism) :
  ∃ (m : PointOnAB prism), angleA1MC1 prism m = π / 2 ∧
  ∀ (m' : PointOnAB prism), angleA1MC1 prism m' ≤ π / 2 :=
sorry

end max_angle_is_90_deg_l3401_340169


namespace middle_group_frequency_l3401_340190

/-- Represents a frequency distribution histogram with 5 rectangles. -/
structure Histogram where
  rectangles : Fin 5 → ℝ
  total_sample : ℝ
  middle_equals_sum : rectangles 2 = (rectangles 0) + (rectangles 1) + (rectangles 3) + (rectangles 4)
  sample_size : total_sample = 100

/-- The frequency of the middle group in the histogram is 50. -/
theorem middle_group_frequency (h : Histogram) : h.rectangles 2 = 50 := by
  sorry

end middle_group_frequency_l3401_340190


namespace domain_of_g_l3401_340105

-- Define the function f with domain [0,4]
def f : Set ℝ := Set.Icc 0 4

-- Define the function g
def g (f : Set ℝ) : Set ℝ := {x | x ∈ f ∧ x^2 ∈ f}

-- Theorem statement
theorem domain_of_g (f : Set ℝ) (hf : f = Set.Icc 0 4) : 
  g f = Set.Icc 0 2 := by sorry

end domain_of_g_l3401_340105


namespace intersection_of_A_and_B_l3401_340104

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l3401_340104


namespace biography_increase_l3401_340130

theorem biography_increase (T : ℝ) (h1 : T > 0) : 
  let initial_bio := 0.20 * T
  let final_bio := 0.32 * T
  (final_bio - initial_bio) / initial_bio = 0.60
  := by sorry

end biography_increase_l3401_340130


namespace product_of_nonreal_roots_l3401_340115

theorem product_of_nonreal_roots : ∃ (r₁ r₂ : ℂ),
  (r₁ ∈ {z : ℂ | z^4 - 4*z^3 + 6*z^2 - 4*z = 2005 ∧ z.im ≠ 0}) ∧
  (r₂ ∈ {z : ℂ | z^4 - 4*z^3 + 6*z^2 - 4*z = 2005 ∧ z.im ≠ 0}) ∧
  r₁ ≠ r₂ ∧
  r₁ * r₂ = 1 + Real.sqrt 2006 :=
by sorry

end product_of_nonreal_roots_l3401_340115


namespace family_pizza_order_correct_l3401_340146

def family_pizza_order (adults : Nat) (children : Nat) (adult_slices : Nat) (child_slices : Nat) (slices_per_pizza : Nat) : Nat :=
  let total_slices := adults * adult_slices + children * child_slices
  (total_slices + slices_per_pizza - 1) / slices_per_pizza

theorem family_pizza_order_correct :
  family_pizza_order 2 12 5 2 6 = 6 := by
  sorry

end family_pizza_order_correct_l3401_340146


namespace trajectory_forms_two_rays_l3401_340111

/-- The trajectory of a point P(x, y) with a constant difference of 2 in its distances to points M(1, 0) and N(3, 0) forms two rays. -/
theorem trajectory_forms_two_rays :
  ∀ (x y : ℝ),
  |((x - 1)^2 + y^2).sqrt - ((x - 3)^2 + y^2).sqrt| = 2 →
  ∃ (a b : ℝ), y = a * x + b ∨ y = -a * x + b :=
by sorry

end trajectory_forms_two_rays_l3401_340111


namespace number_problem_l3401_340132

theorem number_problem : ∃ x : ℝ, 4 * x + 7 * x = 55 ∧ x = 5 := by
  sorry

end number_problem_l3401_340132


namespace bob_anne_distance_difference_l3401_340194

/-- Represents the dimensions of a rectangular block in Geometrytown --/
structure BlockDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle --/
def rectanglePerimeter (d : BlockDimensions) : ℕ :=
  2 * (d.length + d.width)

/-- Represents the street width in Geometrytown --/
def streetWidth : ℕ := 30

/-- Calculates Bob's running distance around the block --/
def bobDistance (d : BlockDimensions) : ℕ :=
  rectanglePerimeter { length := d.length + 2 * streetWidth, width := d.width + 2 * streetWidth }

/-- Calculates Anne's running distance around the block --/
def anneDistance (d : BlockDimensions) : ℕ :=
  rectanglePerimeter d

/-- The main theorem stating the difference between Bob's and Anne's running distances --/
theorem bob_anne_distance_difference (d : BlockDimensions) 
    (h1 : d.length = 300) 
    (h2 : d.width = 500) : 
    bobDistance d - anneDistance d = 240 := by
  sorry

end bob_anne_distance_difference_l3401_340194


namespace no_integer_solutions_for_perfect_square_l3401_340126

theorem no_integer_solutions_for_perfect_square : 
  ¬ ∃ (x : ℤ), ∃ (y : ℤ), x^4 + 4*x^3 + 10*x^2 + 4*x + 29 = y^2 := by
  sorry

end no_integer_solutions_for_perfect_square_l3401_340126


namespace set_operations_correctness_l3401_340173

variable {α : Type*}
variable (A B C : Set α)

theorem set_operations_correctness :
  (A ∪ B = B ∪ A) ∧
  (A ∪ (B ∪ C) = (A ∪ B) ∪ C) ∧
  (A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)) := by
  sorry

end set_operations_correctness_l3401_340173


namespace triangle_area_l3401_340114

theorem triangle_area (b c : ℝ) (angle_C : ℝ) (h1 : b = 1) (h2 : c = Real.sqrt 3) (h3 : angle_C = 2 * Real.pi / 3) :
  (1 / 2) * b * c * Real.sin (Real.pi / 6) = Real.sqrt 3 / 4 := by
sorry

end triangle_area_l3401_340114


namespace power_equality_l3401_340180

theorem power_equality (x : ℝ) (h : (2 : ℝ) ^ (3 * x) = 7) : (8 : ℝ) ^ (x + 1) = 56 := by
  sorry

end power_equality_l3401_340180


namespace square_sum_geq_product_sum_l3401_340149

theorem square_sum_geq_product_sum (a b c : ℝ) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + c*a ∧
  (a^2 + b^2 + c^2 = a*b + b*c + c*a ↔ a = b ∧ b = c) :=
by sorry

end square_sum_geq_product_sum_l3401_340149


namespace luke_stickers_l3401_340185

theorem luke_stickers (initial bought birthday given_away used : ℕ) :
  initial = 20 →
  bought = 12 →
  birthday = 20 →
  given_away = 5 →
  used = 8 →
  initial + bought + birthday - given_away - used = 39 := by
  sorry

end luke_stickers_l3401_340185


namespace cyclic_number_property_l3401_340139

def digit_set (n : ℕ) : Set ℕ :=
  {d | ∃ k, n = d + 10 * k ∨ k = d + 10 * n}

def has_same_digits (a b : ℕ) : Prop :=
  digit_set a = digit_set b

theorem cyclic_number_property (n : ℕ) (h : n = 142857) :
  ∀ k : ℕ, k ≥ 1 → k ≤ 6 → has_same_digits n (n * k) :=
by sorry

end cyclic_number_property_l3401_340139


namespace chili_paste_can_size_l3401_340112

/-- Proves that the size of smaller chili paste cans is 15 ounces -/
theorem chili_paste_can_size 
  (larger_can_size : ℕ) 
  (larger_can_count : ℕ) 
  (extra_smaller_cans : ℕ) 
  (smaller_can_size : ℕ) : 
  larger_can_size = 25 → 
  larger_can_count = 45 → 
  extra_smaller_cans = 30 → 
  (larger_can_count + extra_smaller_cans) * smaller_can_size = larger_can_count * larger_can_size → 
  smaller_can_size = 15 := by
sorry

end chili_paste_can_size_l3401_340112


namespace project_bolts_boxes_l3401_340122

/-- The number of bolts in each box of bolts -/
def bolts_per_box : ℕ := 11

/-- The number of boxes of nuts purchased -/
def boxes_of_nuts : ℕ := 3

/-- The number of nuts in each box of nuts -/
def nuts_per_box : ℕ := 15

/-- The number of bolts left over -/
def bolts_leftover : ℕ := 3

/-- The number of nuts left over -/
def nuts_leftover : ℕ := 6

/-- The total number of bolts and nuts used for the project -/
def total_used : ℕ := 113

/-- The minimum number of boxes of bolts purchased -/
def min_boxes_of_bolts : ℕ := 7

theorem project_bolts_boxes :
  ∃ (boxes_of_bolts : ℕ),
    boxes_of_bolts * bolts_per_box ≥
      total_used - (boxes_of_nuts * nuts_per_box - nuts_leftover) + bolts_leftover ∧
    boxes_of_bolts = min_boxes_of_bolts :=
by sorry

end project_bolts_boxes_l3401_340122


namespace B_power_150_is_identity_l3401_340197

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end B_power_150_is_identity_l3401_340197


namespace timothy_total_cost_l3401_340171

/-- Calculates the total cost of Timothy's purchases --/
def total_cost (land_acres : ℕ) (land_price_per_acre : ℕ) (house_price : ℕ) 
  (num_cows : ℕ) (cow_price : ℕ) (num_chickens : ℕ) (chicken_price : ℕ)
  (solar_install_hours : ℕ) (solar_install_rate : ℕ) (solar_equipment_price : ℕ) : ℕ :=
  land_acres * land_price_per_acre +
  house_price +
  num_cows * cow_price +
  num_chickens * chicken_price +
  solar_install_hours * solar_install_rate + solar_equipment_price

/-- Theorem stating that the total cost of Timothy's purchases is $147,700 --/
theorem timothy_total_cost :
  total_cost 30 20 120000 20 1000 100 5 6 100 6000 = 147700 := by
  sorry

end timothy_total_cost_l3401_340171


namespace complex_number_in_third_quadrant_l3401_340174

theorem complex_number_in_third_quadrant : 
  let z : ℂ := (3 - 2*I) / I
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end complex_number_in_third_quadrant_l3401_340174


namespace cone_height_l3401_340178

/-- The height of a cone given its slant height and lateral area -/
theorem cone_height (l : ℝ) (area : ℝ) (h : l = 13 ∧ area = 65 * Real.pi) :
  ∃ (r : ℝ), r > 0 ∧ area = Real.pi * r * l ∧ Real.sqrt (l^2 - r^2) = 12 := by
  sorry

end cone_height_l3401_340178


namespace speed_AB_is_60_l3401_340189

-- Define the distances and speeds
def distance_BC : ℝ := 1  -- We can use any positive real number as the base distance
def distance_AB : ℝ := 2 * distance_BC
def speed_BC : ℝ := 20
def average_speed : ℝ := 36

-- Define the speed from A to B as a variable we want to solve for
def speed_AB : ℝ := sorry

-- Theorem statement
theorem speed_AB_is_60 : speed_AB = 60 := by
  sorry

end speed_AB_is_60_l3401_340189


namespace cosine_value_in_special_triangle_l3401_340156

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the theorem
theorem cosine_value_in_special_triangle (t : Triangle) 
  (h1 : t.c = 2 * t.a)  -- Given condition: c = 2a
  (h2 : Real.sin t.B ^ 2 = Real.sin t.A * Real.sin t.C)  -- Given condition: sin²B = sin A * sin C
  : Real.cos t.B = 3/4 := by
  sorry

end cosine_value_in_special_triangle_l3401_340156


namespace t_shirts_per_package_l3401_340142

theorem t_shirts_per_package (packages : ℕ) (total_shirts : ℕ) 
  (h1 : packages = 71) (h2 : total_shirts = 426) : 
  total_shirts / packages = 6 := by
sorry

end t_shirts_per_package_l3401_340142


namespace increasing_seq_with_properties_is_geometric_l3401_340162

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define the properties
def Property1 (a : Sequence) : Prop :=
  ∀ i j, i > j → ∃ m, a i ^ 2 / a j = a m

def Property2 (a : Sequence) : Prop :=
  ∀ n, n ≥ 3 → ∃ k l, k > l ∧ a n = a k ^ 2 / a l

-- Define increasing sequence
def IncreasingSeq (a : Sequence) : Prop :=
  ∀ n m, n < m → a n < a m

-- Define geometric sequence
def GeometricSeq (a : Sequence) : Prop :=
  ∃ r, ∀ n, a (n + 1) = r * a n

-- State the theorem
theorem increasing_seq_with_properties_is_geometric (a : Sequence) 
  (h_inc : IncreasingSeq a) 
  (h1 : Property1 a) 
  (h2 : Property2 a) : 
  GeometricSeq a := by
  sorry

end increasing_seq_with_properties_is_geometric_l3401_340162


namespace stock_price_drop_l3401_340193

theorem stock_price_drop (initial_price : ℝ) (x : ℝ) 
  (h1 : initial_price > 0)
  (h2 : x > 0 ∧ x < 100)
  (h3 : (1 + 0.3) * (1 - x / 100) * (1 + 0.2) * initial_price = 1.17 * initial_price) :
  x = 25 := by
sorry

end stock_price_drop_l3401_340193


namespace special_triangle_ratio_l3401_340127

/-- A scalene triangle with two medians equal to two altitudes -/
structure SpecialTriangle where
  -- The triangle is scalene
  is_scalene : Bool
  -- Two medians are equal to two altitudes
  two_medians_equal_altitudes : Bool

/-- The ratio of the third median to the third altitude -/
def third_median_altitude_ratio (t : SpecialTriangle) : ℚ :=
  7 / 2

/-- Theorem stating the ratio of the third median to the third altitude -/
theorem special_triangle_ratio (t : SpecialTriangle) 
  (h1 : t.is_scalene = true) 
  (h2 : t.two_medians_equal_altitudes = true) : 
  third_median_altitude_ratio t = 7 / 2 := by
  sorry

end special_triangle_ratio_l3401_340127


namespace game_result_l3401_340175

def game_operation (n : ℕ) : ℕ :=
  if n % 2 = 1 then n + 3 else n / 2

def reaches_one_in (n : ℕ) (steps : ℕ) : Prop :=
  ∃ (seq : Fin steps.succ → ℕ), 
    seq 0 = n ∧ 
    seq steps = 1 ∧ 
    ∀ i : Fin steps, seq (i.succ) = game_operation (seq i)

theorem game_result :
  {n : ℕ | reaches_one_in n 5} = {1, 8, 16, 10, 13} := by sorry

end game_result_l3401_340175


namespace total_lateness_l3401_340150

/-- Given a student who is 20 minutes late and four other students who are each 10 minutes later than the first student, 
    the total time of lateness for all five students is 140 minutes. -/
theorem total_lateness (charlize_lateness : ℕ) (classmates_count : ℕ) (additional_lateness : ℕ) : 
  charlize_lateness = 20 →
  classmates_count = 4 →
  additional_lateness = 10 →
  charlize_lateness + classmates_count * (charlize_lateness + additional_lateness) = 140 :=
by sorry

end total_lateness_l3401_340150


namespace smallest_p_value_l3401_340152

theorem smallest_p_value (p q : ℕ+) (h1 : (5 : ℚ) / 8 < p / q) (h2 : p / q < (7 : ℚ) / 8) (h3 : p + q = 2005) :
  p.val ≥ 772 ∧ (∀ (p' : ℕ+), p'.val ≥ 772 → (5 : ℚ) / 8 < p' / (2005 - p') → p' / (2005 - p') < (7 : ℚ) / 8 → p'.val ≤ p.val) :=
sorry

end smallest_p_value_l3401_340152


namespace alex_initial_silk_amount_l3401_340145

/-- The amount of silk Alex had in storage initially -/
def initial_silk_amount (num_friends : ℕ) (silk_per_friend : ℕ) (num_dresses : ℕ) (silk_per_dress : ℕ) : ℕ :=
  num_friends * silk_per_friend + num_dresses * silk_per_dress

/-- Theorem stating that Alex had 600 meters of silk initially -/
theorem alex_initial_silk_amount :
  initial_silk_amount 5 20 100 5 = 600 := by sorry

end alex_initial_silk_amount_l3401_340145


namespace cubic_sum_l3401_340160

theorem cubic_sum (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 12) : x^3 + y^3 = 224 := by
  sorry

end cubic_sum_l3401_340160


namespace cow_fraction_sold_l3401_340123

/-- Represents the number of animals on a petting farm. -/
structure PettingFarm where
  cows : ℕ
  dogs : ℕ

/-- Represents the state of the petting farm before and after selling animals. -/
structure FarmState where
  initial : PettingFarm
  final : PettingFarm

theorem cow_fraction_sold (farm : FarmState) : 
  farm.initial.cows = 184 →
  farm.initial.cows = 2 * farm.initial.dogs →
  farm.final.dogs = farm.initial.dogs / 4 →
  farm.final.cows + farm.final.dogs = 161 →
  (farm.initial.cows - farm.final.cows) / farm.initial.cows = 1/4 := by
  sorry

end cow_fraction_sold_l3401_340123


namespace both_correct_undetermined_l3401_340157

/-- Represents a class of students and their test performance -/
structure ClassTestResults where
  total_students : ℕ
  correct_q1 : ℕ
  correct_q2 : ℕ
  absent : ℕ

/-- Predicate to check if the number of students who answered both questions correctly is determinable -/
def both_correct_determinable (c : ClassTestResults) : Prop :=
  ∃ (n : ℕ), n ≤ c.correct_q1 ∧ n ≤ c.correct_q2 ∧ n = c.correct_q1 + c.correct_q2 - (c.total_students - c.absent)

/-- Theorem stating that the number of students who answered both questions correctly is undetermined -/
theorem both_correct_undetermined (c : ClassTestResults)
  (h1 : c.total_students = 25)
  (h2 : c.correct_q1 = 22)
  (h3 : c.absent = 3)
  (h4 : c.correct_q2 ≤ c.total_students - c.absent)
  (h5 : c.correct_q2 > 0) :
  ¬ both_correct_determinable c := by
  sorry


end both_correct_undetermined_l3401_340157
