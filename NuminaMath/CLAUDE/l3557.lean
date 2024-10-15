import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l3557_355721

theorem sqrt_sum_equality : ∃ (a b c : ℕ+), 
  (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) + Real.sqrt 11 * Real.sqrt 3 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    (Real.sqrt 3 + (1 / Real.sqrt 3) + Real.sqrt 11 + (1 / Real.sqrt 11) + Real.sqrt 11 * Real.sqrt 3 = (a' * Real.sqrt 3 + b' * Real.sqrt 11) / c') → 
    c ≤ c') ∧
  a = 84 ∧ b = 44 ∧ c = 33 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l3557_355721


namespace NUMINAMATH_CALUDE_kathryn_gave_skittles_l3557_355759

def cheryl_start : ℕ := 8
def cheryl_end : ℕ := 97

theorem kathryn_gave_skittles : cheryl_end - cheryl_start = 89 := by
  sorry

end NUMINAMATH_CALUDE_kathryn_gave_skittles_l3557_355759


namespace NUMINAMATH_CALUDE_mary_chopped_chairs_l3557_355733

/-- Represents the number of sticks of wood produced by different furniture types -/
structure FurnitureWood where
  chair : ℕ
  table : ℕ
  stool : ℕ

/-- Represents the furniture Mary chopped up -/
structure ChoppedFurniture where
  chairs : ℕ
  tables : ℕ
  stools : ℕ

/-- Calculates the total number of sticks from chopped furniture -/
def totalSticks (fw : FurnitureWood) (cf : ChoppedFurniture) : ℕ :=
  fw.chair * cf.chairs + fw.table * cf.tables + fw.stool * cf.stools

theorem mary_chopped_chairs :
  ∀ (fw : FurnitureWood) (cf : ChoppedFurniture) (burn_rate hours_warm : ℕ),
    fw.chair = 6 →
    fw.table = 9 →
    fw.stool = 2 →
    burn_rate = 5 →
    hours_warm = 34 →
    cf.tables = 6 →
    cf.stools = 4 →
    totalSticks fw cf = burn_rate * hours_warm →
    cf.chairs = 18 := by
  sorry

end NUMINAMATH_CALUDE_mary_chopped_chairs_l3557_355733


namespace NUMINAMATH_CALUDE_temperature_difference_l3557_355794

/-- Given the highest and lowest temperatures in a city on a certain day, 
    calculate the temperature difference. -/
theorem temperature_difference 
  (highest_temp lowest_temp : ℤ) 
  (h_highest : highest_temp = 11)
  (h_lowest : lowest_temp = -1) :
  highest_temp - lowest_temp = 12 := by
  sorry

end NUMINAMATH_CALUDE_temperature_difference_l3557_355794


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l3557_355782

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 16 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l3557_355782


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3557_355713

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The ninth term of an arithmetic sequence is 32, given that its third term is 20 and its sixth term is 26. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℕ)
  (h_arith : ArithmeticSequence a)
  (h_third : a 3 = 20)
  (h_sixth : a 6 = 26) :
  a 9 = 32 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3557_355713


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l3557_355768

theorem inequality_and_equality_condition (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 ≥ 6 * Real.sqrt 3 ∧ 
  (a^2 + b^2 + c^2 + (1/a + 1/b + 1/c)^2 = 6 * Real.sqrt 3 ↔ 
   a = b ∧ b = c ∧ c = Real.rpow 3 (1/4)) :=
sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l3557_355768


namespace NUMINAMATH_CALUDE_club_equation_solution_l3557_355718

/-- Define the ♣ operation -/
def club (A B : ℝ) : ℝ := 3 * A + 2 * B + 7

/-- Theorem stating that 17 is the unique solution to A ♣ 6 = 70 -/
theorem club_equation_solution :
  ∃! A : ℝ, club A 6 = 70 ∧ A = 17 := by
  sorry

end NUMINAMATH_CALUDE_club_equation_solution_l3557_355718


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3557_355786

theorem cost_price_calculation (cost_price : ℝ) : 
  cost_price * 1.20 * 0.91 = cost_price + 16 → cost_price = 200 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l3557_355786


namespace NUMINAMATH_CALUDE_problem_solution_l3557_355707

theorem problem_solution :
  (∀ x : ℝ, x^2 - x + 1 ≥ 0) ∧
  (∀ p q : Prop, (¬(p ∨ q) → (¬p ∧ ¬q))) ∧
  ((∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
   (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ x ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3557_355707


namespace NUMINAMATH_CALUDE_parabola_translation_correct_l3557_355764

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 2x² -/
def original_parabola : Parabola := { a := 2, b := 0, c := 0 }

/-- Translates a parabola horizontally by h units and vertically by k units -/
def translate (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + k }

/-- The resulting parabola after translation -/
def translated_parabola : Parabola :=
  translate (translate original_parabola 3 0) 0 4

theorem parabola_translation_correct :
  translated_parabola.a = 2 ∧
  translated_parabola.b = -12 ∧
  translated_parabola.c = 22 :=
sorry

end NUMINAMATH_CALUDE_parabola_translation_correct_l3557_355764


namespace NUMINAMATH_CALUDE_linear_function_properties_l3557_355703

/-- A linear function passing through two points and intersecting a horizontal line -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

theorem linear_function_properties 
  (k b : ℝ) 
  (h_k : k ≠ 0)
  (h_point_A : LinearFunction k b 0 = 1)
  (h_point_B : LinearFunction k b 1 = 2)
  (h_intersect : ∃ x, LinearFunction k b x = 4) :
  (k = 1 ∧ b = 1) ∧ 
  (∃ x, x = 3 ∧ LinearFunction k b x = 4) ∧
  (∀ x, x < 3 → (2/3 * x + 2 > LinearFunction k b x ∧ 2/3 * x + 2 < 4)) := by
  sorry

#check linear_function_properties

end NUMINAMATH_CALUDE_linear_function_properties_l3557_355703


namespace NUMINAMATH_CALUDE_function_properties_l3557_355709

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- Define the specific function h
def h (m : ℝ) (x : ℝ) : ℝ := f 2 9 x - m + 1

-- Theorem statement
theorem function_properties :
  (∃ (a b : ℝ), f a b (-1) = 0 ∧ 
   (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b x ≥ f a b (-1)) ∧
   (∀ x : ℝ, f a b x = x^3 + 6*x^2 + 9*x + 4)) ∧
  (∀ m : ℝ, (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    h m x₁ = 0 ∧ h m x₂ = 0 ∧ h m x₃ = 0) ↔ 1 < m ∧ m < 5) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3557_355709


namespace NUMINAMATH_CALUDE_portrait_price_ratio_l3557_355750

def price_8inch : ℝ := 5
def daily_8inch_sales : ℕ := 3
def daily_16inch_sales : ℕ := 5
def earnings_3days : ℝ := 195

def price_ratio : ℝ := 2

theorem portrait_price_ratio :
  let daily_earnings := daily_8inch_sales * price_8inch + daily_16inch_sales * (price_ratio * price_8inch)
  earnings_3days = 3 * daily_earnings :=
by sorry

end NUMINAMATH_CALUDE_portrait_price_ratio_l3557_355750


namespace NUMINAMATH_CALUDE_sheila_work_hours_l3557_355779

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  monday_wednesday_friday_hours : ℕ
  tuesday_thursday_hours : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Theorem stating the number of hours Sheila works on Monday, Wednesday, and Friday --/
theorem sheila_work_hours (schedule : WorkSchedule) : 
  schedule.monday_wednesday_friday_hours = 24 :=
by
  have h1 : schedule.tuesday_thursday_hours = 6 * 2 := by sorry
  have h2 : schedule.weekly_earnings = 468 := by sorry
  have h3 : schedule.hourly_rate = 13 := by sorry
  sorry

end NUMINAMATH_CALUDE_sheila_work_hours_l3557_355779


namespace NUMINAMATH_CALUDE_insect_crawl_properties_l3557_355723

def crawl_distances : List ℤ := [5, -3, 10, -8, -6, 12, -10]

theorem insect_crawl_properties :
  let cumulative_distances := crawl_distances.scanl (· + ·) 0
  (crawl_distances.sum = 0) ∧
  (cumulative_distances.map (Int.natAbs)).maximum? = some 14 ∧
  ((crawl_distances.map Int.natAbs).sum = 54) := by
  sorry

end NUMINAMATH_CALUDE_insect_crawl_properties_l3557_355723


namespace NUMINAMATH_CALUDE_postcard_collection_average_l3557_355761

/-- 
Given an arithmetic sequence with:
- First term: 10
- Common difference: 12
- Number of terms: 7
Prove that the average of all terms is 46.
-/
theorem postcard_collection_average : 
  let first_term := 10
  let common_diff := 12
  let num_days := 7
  let last_term := first_term + (num_days - 1) * common_diff
  (first_term + last_term) / 2 = 46 := by
sorry

end NUMINAMATH_CALUDE_postcard_collection_average_l3557_355761


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l3557_355757

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l3557_355757


namespace NUMINAMATH_CALUDE_finite_game_has_winning_strategy_l3557_355752

/-- Represents a two-player game with finite choices and finite length -/
structure FiniteGame where
  /-- The maximum number of moves before the game ends -/
  max_moves : ℕ
  /-- The number of possible choices for each move -/
  num_choices : ℕ
  /-- Predicate to check if the game has ended -/
  is_game_over : (List ℕ) → Bool
  /-- Predicate to determine the winner (true for player A, false for player B) -/
  winner : (List ℕ) → Bool

/-- Definition of a winning strategy for a player -/
def has_winning_strategy (game : FiniteGame) (player : Bool) : Prop :=
  ∃ (strategy : List ℕ → ℕ),
    ∀ (game_state : List ℕ),
      (game_state.length < game.max_moves) →
      (game.is_game_over game_state = false) →
      (game_state.length % 2 = if player then 0 else 1) →
      (strategy game_state ≤ game.num_choices) ∧
      (∃ (final_state : List ℕ),
        final_state.length ≤ game.max_moves ∧
        game.is_game_over final_state = true ∧
        game.winner final_state = player)

/-- Theorem: In a finite two-player game, one player must have a winning strategy -/
theorem finite_game_has_winning_strategy (game : FiniteGame) :
  has_winning_strategy game true ∨ has_winning_strategy game false :=
sorry

end NUMINAMATH_CALUDE_finite_game_has_winning_strategy_l3557_355752


namespace NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3557_355736

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (9 * a^3 - 27 * a + 54 = 0) →
  (9 * b^3 - 27 * b + 54 = 0) →
  (9 * c^3 - 27 * c + 54 = 0) →
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 18 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_cubes_l3557_355736


namespace NUMINAMATH_CALUDE_min_entries_to_four_coins_l3557_355726

/-- Represents the state of coins and last entry -/
structure CoinState :=
  (coins : ℕ)
  (lastEntry : ℕ)

/-- Defines the coin machine rules -/
def coinMachine (entry : ℕ) : ℕ :=
  match entry with
  | 7 => 3
  | 8 => 11
  | 9 => 4
  | _ => 0

/-- Checks if an entry is valid -/
def isValidEntry (state : CoinState) (entry : ℕ) : Bool :=
  state.coins ≥ entry ∧ entry ≠ state.lastEntry ∧ (entry = 7 ∨ entry = 8 ∨ entry = 9)

/-- Makes an entry and returns the new state -/
def makeEntry (state : CoinState) (entry : ℕ) : CoinState :=
  { coins := state.coins - entry + coinMachine entry,
    lastEntry := entry }

/-- Defines the minimum number of entries to reach the target -/
def minEntries (start : ℕ) (target : ℕ) : ℕ := sorry

/-- Theorem stating the minimum number of entries to reach 4 coins from 15 coins is 4 -/
theorem min_entries_to_four_coins :
  minEntries 15 4 = 4 := by sorry

end NUMINAMATH_CALUDE_min_entries_to_four_coins_l3557_355726


namespace NUMINAMATH_CALUDE_square_plus_one_ge_two_abs_l3557_355742

theorem square_plus_one_ge_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_ge_two_abs_l3557_355742


namespace NUMINAMATH_CALUDE_x_equals_y_l3557_355787

theorem x_equals_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) : y = x := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_l3557_355787


namespace NUMINAMATH_CALUDE_race_theorem_l3557_355771

/-- Represents a runner in a race -/
structure Runner where
  speed : ℝ

/-- Represents a race between two runners -/
structure Race where
  distance : ℝ
  runner_a : Runner
  runner_b : Runner

/-- The result of the first race -/
def first_race_result (r : Race) (d : ℝ) : Prop :=
  r.distance / r.runner_b.speed = (r.distance - d) / r.runner_a.speed

/-- The theorem to be proved -/
theorem race_theorem (h d : ℝ) (r : Race) 
  (h_pos : h > 0)
  (d_pos : d > 0)
  (first_race : first_race_result r d)
  (h_eq : r.distance = h) :
  let second_race_time := (h + d/2) / r.runner_a.speed
  let second_race_b_distance := second_race_time * r.runner_b.speed
  h - second_race_b_distance = d * (d + h) / (2 * h) := by
  sorry

end NUMINAMATH_CALUDE_race_theorem_l3557_355771


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l3557_355728

-- Define the variables x, y, and z as positive real numbers
variable (x y z : ℝ)

-- Define the conditions
def positive_conditions : Prop := x > 0 ∧ y > 0 ∧ z > 0
def equation_condition : Prop := x - 2*y + 3*z = 0

-- State the theorem
theorem minimum_value_theorem 
  (h1 : positive_conditions x y z) 
  (h2 : equation_condition x y z) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), ∀ (a b c : ℝ), 
    positive_conditions a b c → 
    equation_condition a b c → 
    f x y z ≤ f a b c :=
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l3557_355728


namespace NUMINAMATH_CALUDE_rabbits_to_add_correct_rabbits_to_add_l3557_355743

theorem rabbits_to_add (initial_rabbits : ℕ) (park_rabbits : ℕ) : ℕ :=
  let final_rabbits := park_rabbits / 3
  final_rabbits - initial_rabbits

theorem correct_rabbits_to_add :
  rabbits_to_add 13 60 = 7 := by sorry

end NUMINAMATH_CALUDE_rabbits_to_add_correct_rabbits_to_add_l3557_355743


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3557_355704

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x < 2}

theorem sufficient_not_necessary : 
  (∀ a : ℝ, a ∈ M → a ∈ N) ∧ 
  (∃ a : ℝ, a ∈ N ∧ a ∉ M) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3557_355704


namespace NUMINAMATH_CALUDE_clock_hands_opposite_l3557_355777

/-- Represents the number of minutes past 10:00 --/
def x : ℝ := 13

/-- The rate at which the minute hand moves (degrees per minute) --/
def minute_hand_rate : ℝ := 6

/-- The rate at which the hour hand moves (degrees per minute) --/
def hour_hand_rate : ℝ := 0.5

/-- The angle between the minute and hour hands when they are opposite --/
def opposite_angle : ℝ := 180

theorem clock_hands_opposite : 
  0 < x ∧ x < 60 ∧
  minute_hand_rate * (6 + x) + hour_hand_rate * (120 - x + 3) = opposite_angle :=
by sorry

end NUMINAMATH_CALUDE_clock_hands_opposite_l3557_355777


namespace NUMINAMATH_CALUDE_cosine_symmetry_center_l3557_355773

/-- The symmetry center of the cosine function with a phase shift --/
theorem cosine_symmetry_center (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos (2 * x - π / 4)
  ∃ c : ℝ × ℝ, c = (π / 8, 0) ∧ 
    (∀ x : ℝ, f (c.1 + x) = f (c.1 - x)) :=
by sorry

end NUMINAMATH_CALUDE_cosine_symmetry_center_l3557_355773


namespace NUMINAMATH_CALUDE_b_current_age_b_current_age_proof_l3557_355778

theorem b_current_age : ℕ → ℕ → Prop :=
  fun a b =>
    (a = b + 15) →  -- A is 15 years older than B
    (a - 5 = 2 * (b - 5)) →  -- Five years ago, A's age was twice B's age
    (b = 20)  -- B's current age is 20

-- The proof is omitted
theorem b_current_age_proof : ∃ (a b : ℕ), b_current_age a b :=
  sorry

end NUMINAMATH_CALUDE_b_current_age_b_current_age_proof_l3557_355778


namespace NUMINAMATH_CALUDE_jason_pears_count_l3557_355702

/-- The number of pears Jason picked -/
def jason_pears : ℕ := 105 - (47 + 12)

/-- The total number of pears picked -/
def total_pears : ℕ := 105

/-- The number of pears Keith picked -/
def keith_pears : ℕ := 47

/-- The number of pears Mike picked -/
def mike_pears : ℕ := 12

theorem jason_pears_count : jason_pears = 46 := by sorry

end NUMINAMATH_CALUDE_jason_pears_count_l3557_355702


namespace NUMINAMATH_CALUDE_rectangle_division_possible_l3557_355719

theorem rectangle_division_possible : ∃ (w1 h1 w2 h2 w3 h3 : ℕ+), 
  (w1 * h1 : ℕ) + (w2 * h2 : ℕ) + (w3 * h3 : ℕ) = 100 * 70 ∧
  (w1 : ℕ) ≤ 100 ∧ (h1 : ℕ) ≤ 70 ∧
  (w2 : ℕ) ≤ 100 ∧ (h2 : ℕ) ≤ 70 ∧
  (w3 : ℕ) ≤ 100 ∧ (h3 : ℕ) ≤ 70 ∧
  2 * (w1 * h1 : ℕ) = (w2 * h2 : ℕ) ∧
  2 * (w2 * h2 : ℕ) = (w3 * h3 : ℕ) := by
  sorry

#check rectangle_division_possible

end NUMINAMATH_CALUDE_rectangle_division_possible_l3557_355719


namespace NUMINAMATH_CALUDE_least_common_denominator_l3557_355744

theorem least_common_denominator : 
  let denominators : List Nat := [3, 4, 5, 6, 8, 9, 10]
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 4) 5) 6) 8) 9) 10 = 360 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l3557_355744


namespace NUMINAMATH_CALUDE_geometry_propositions_l3557_355775

structure Geometry3D where
  Line : Type
  Plane : Type
  parallel : Line → Plane → Prop
  perpendicular : Line → Plane → Prop
  plane_parallel : Plane → Plane → Prop
  plane_perpendicular : Plane → Plane → Prop

variable (G : Geometry3D)

theorem geometry_propositions 
  (l : G.Line) (α β : G.Plane) (h_diff : α ≠ β) :
  (∃ l α β, G.parallel l α ∧ G.parallel l β ∧ ¬ G.plane_parallel α β) ∧
  (∀ l α β, G.perpendicular l α ∧ G.perpendicular l β → G.plane_parallel α β) ∧
  (∃ l α β, G.perpendicular l α ∧ G.parallel l β ∧ ¬ G.plane_parallel α β) ∧
  (∃ l α β, G.plane_perpendicular α β ∧ G.parallel l α ∧ ¬ G.perpendicular l β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l3557_355775


namespace NUMINAMATH_CALUDE_f_difference_l3557_355730

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as sigma(n) / n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(540) - f(180) = 7/90 -/
theorem f_difference : f 540 - f 180 = 7 / 90 := by sorry

end NUMINAMATH_CALUDE_f_difference_l3557_355730


namespace NUMINAMATH_CALUDE_probability_rain_july_approx_l3557_355774

/-- The probability of rain on at most 1 day in July, given the daily rain probability and number of days. -/
def probability_rain_at_most_one_day (daily_prob : ℝ) (num_days : ℕ) : ℝ :=
  (1 - daily_prob) ^ num_days + num_days * daily_prob * (1 - daily_prob) ^ (num_days - 1)

/-- Theorem stating that the probability of rain on at most 1 day in July is approximately 0.271. -/
theorem probability_rain_july_approx : 
  ∃ ε > 0, ε < 0.001 ∧ 
  |probability_rain_at_most_one_day (1/20) 31 - 0.271| < ε :=
sorry

end NUMINAMATH_CALUDE_probability_rain_july_approx_l3557_355774


namespace NUMINAMATH_CALUDE_initial_necklaces_count_l3557_355714

theorem initial_necklaces_count (initial_earrings : ℕ) 
  (total_jewelry : ℕ) : 
  initial_earrings = 15 →
  total_jewelry = 57 →
  ∃ (initial_necklaces : ℕ),
    initial_necklaces = 15 ∧
    2 * initial_necklaces + initial_earrings + 
    (2/3 : ℚ) * initial_earrings + 
    (1/5 : ℚ) * ((2/3 : ℚ) * initial_earrings) = total_jewelry :=
by sorry

end NUMINAMATH_CALUDE_initial_necklaces_count_l3557_355714


namespace NUMINAMATH_CALUDE_martha_crayon_count_l3557_355780

def final_crayon_count (initial : ℕ) (first_purchase : ℕ) (contest_win : ℕ) (second_purchase : ℕ) : ℕ :=
  (initial / 2) + first_purchase + contest_win + second_purchase

theorem martha_crayon_count :
  final_crayon_count 18 20 15 25 = 69 := by
  sorry

end NUMINAMATH_CALUDE_martha_crayon_count_l3557_355780


namespace NUMINAMATH_CALUDE_total_results_l3557_355735

theorem total_results (average : ℝ) (first_five_avg : ℝ) (last_seven_avg : ℝ) (fifth_result : ℝ)
  (h1 : average = 42)
  (h2 : first_five_avg = 49)
  (h3 : last_seven_avg = 52)
  (h4 : fifth_result = 147) :
  ∃ n : ℕ, n = 11 ∧ n * average = 5 * first_five_avg + 7 * last_seven_avg - fifth_result := by
  sorry

end NUMINAMATH_CALUDE_total_results_l3557_355735


namespace NUMINAMATH_CALUDE_odd_sum_probability_l3557_355716

/-- The first 15 prime numbers -/
def first_15_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

/-- The number of ways to select 3 primes from the first 15 primes -/
def total_selections : Nat := Nat.choose 15 3

/-- The number of ways to select 3 primes from the first 15 primes such that their sum is odd -/
def odd_sum_selections : Nat := Nat.choose 14 2

theorem odd_sum_probability :
  (odd_sum_selections : ℚ) / total_selections = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l3557_355716


namespace NUMINAMATH_CALUDE_output_for_twelve_l3557_355763

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 20 then step1 - 2 else step1 / 2

theorem output_for_twelve : function_machine 12 = 34 := by sorry

end NUMINAMATH_CALUDE_output_for_twelve_l3557_355763


namespace NUMINAMATH_CALUDE_stratified_sampling_type_D_l3557_355746

/-- Calculates the number of units to be selected from a specific product type in stratified sampling -/
def stratifiedSampleSize (totalProduction : ℕ) (typeProduction : ℕ) (totalSample : ℕ) : ℕ :=
  (typeProduction * totalSample) / totalProduction

/-- The problem statement -/
theorem stratified_sampling_type_D :
  let totalProduction : ℕ := 100 + 200 + 300 + 400
  let typeDProduction : ℕ := 400
  let totalSample : ℕ := 50
  stratifiedSampleSize totalProduction typeDProduction totalSample = 20 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_type_D_l3557_355746


namespace NUMINAMATH_CALUDE_small_cuboid_length_l3557_355741

/-- Represents the dimensions of a cuboid -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.length * c.width * c.height

/-- Theorem: Given a large cuboid of 16m x 10m x 12m and small cuboids of Lm x 4m x 3m,
    if 32 small cuboids can be formed from the large cuboid, then L = 5m -/
theorem small_cuboid_length
  (large : Cuboid)
  (small : Cuboid)
  (h1 : large.length = 16)
  (h2 : large.width = 10)
  (h3 : large.height = 12)
  (h4 : small.width = 4)
  (h5 : small.height = 3)
  (h6 : volume large = 32 * volume small) :
  small.length = 5 := by
  sorry

end NUMINAMATH_CALUDE_small_cuboid_length_l3557_355741


namespace NUMINAMATH_CALUDE_factorization_equality_l3557_355793

theorem factorization_equality (a b : ℝ) : a^3 + 2*a^2*b + a*b^2 = a*(a+b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3557_355793


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3557_355724

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 3 + a 4 + a 5 = 12) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3557_355724


namespace NUMINAMATH_CALUDE_payment_ways_formula_l3557_355792

/-- The number of ways to pay n euros using 1-euro and 2-euro coins -/
def paymentWays (n : ℕ) : ℕ := n / 2 + 1

/-- Theorem: The number of ways to pay n euros using 1-euro and 2-euro coins
    is equal to ⌊n/2⌋ + 1 -/
theorem payment_ways_formula (n : ℕ) :
  paymentWays n = n / 2 + 1 := by
  sorry

#check payment_ways_formula

end NUMINAMATH_CALUDE_payment_ways_formula_l3557_355792


namespace NUMINAMATH_CALUDE_door_pole_equation_l3557_355795

/-- 
Given a rectangular door and a pole:
- The door's diagonal length is x
- The pole's length is x
- When placed horizontally, the pole extends 4 feet beyond the door's width
- When placed vertically, the pole extends 2 feet beyond the door's height

This theorem proves that the equation (x-2)^2 + (x-4)^2 = x^2 holds true for this configuration.
-/
theorem door_pole_equation (x : ℝ) : (x - 2)^2 + (x - 4)^2 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_door_pole_equation_l3557_355795


namespace NUMINAMATH_CALUDE_irrigation_canal_construction_l3557_355762

/-- Irrigation Canal Construction Problem -/
theorem irrigation_canal_construction
  (total_length : ℝ)
  (team_b_extra : ℝ)
  (time_ratio : ℝ)
  (cost_a : ℝ)
  (cost_b : ℝ)
  (total_time : ℝ)
  (h_total_length : total_length = 1650)
  (h_team_b_extra : team_b_extra = 30)
  (h_time_ratio : time_ratio = 3/2)
  (h_cost_a : cost_a = 90000)
  (h_cost_b : cost_b = 120000)
  (h_total_time : total_time = 14) :
  ∃ (rate_a rate_b total_cost : ℝ),
    rate_a = 60 ∧
    rate_b = 90 ∧
    total_cost = 2340000 ∧
    rate_b = rate_a + team_b_extra ∧
    (total_length / rate_b) * time_ratio = (total_length / rate_a) ∧
    ∃ (solo_days : ℝ),
      solo_days * rate_a + (total_time - solo_days) * (rate_a + rate_b) = total_length ∧
      total_cost = solo_days * cost_a + total_time * cost_a + (total_time - solo_days) * cost_b :=
by sorry

end NUMINAMATH_CALUDE_irrigation_canal_construction_l3557_355762


namespace NUMINAMATH_CALUDE_min_packs_for_120_cans_l3557_355772

/-- Represents a combination of soda packs -/
structure SodaPacks where
  pack8 : ℕ
  pack15 : ℕ
  pack32 : ℕ

/-- Calculates the total number of cans for a given combination of packs -/
def totalCans (packs : SodaPacks) : ℕ :=
  8 * packs.pack8 + 15 * packs.pack15 + 32 * packs.pack32

/-- Calculates the total number of packs for a given combination -/
def totalPacks (packs : SodaPacks) : ℕ :=
  packs.pack8 + packs.pack15 + packs.pack32

/-- Theorem: The minimum number of packs to buy exactly 120 cans is 6 -/
theorem min_packs_for_120_cans : 
  ∃ (min_packs : SodaPacks), 
    totalCans min_packs = 120 ∧ 
    totalPacks min_packs = 6 ∧
    ∀ (other_packs : SodaPacks), 
      totalCans other_packs = 120 → 
      totalPacks other_packs ≥ totalPacks min_packs :=
by
  sorry

end NUMINAMATH_CALUDE_min_packs_for_120_cans_l3557_355772


namespace NUMINAMATH_CALUDE_existence_of_sum_greater_than_one_l3557_355706

theorem existence_of_sum_greater_than_one : 
  ¬(∀ (x y : ℝ), x + y ≤ 1) := by sorry

end NUMINAMATH_CALUDE_existence_of_sum_greater_than_one_l3557_355706


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3557_355720

/-- An isosceles triangle with congruent sides of length 8 and perimeter 25 has a base of length 9. -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruent_side := 8
    let perimeter := 25
    (2 * congruent_side + base = perimeter) →
    base = 9

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3557_355720


namespace NUMINAMATH_CALUDE_octagon_interior_angle_l3557_355701

/-- The measure of each interior angle in a regular octagon -/
def interior_angle_octagon : ℝ := 135

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem octagon_interior_angle :
  interior_angle_octagon = (sum_interior_angles octagon_sides) / octagon_sides :=
by sorry

end NUMINAMATH_CALUDE_octagon_interior_angle_l3557_355701


namespace NUMINAMATH_CALUDE_no_bribed_judges_probability_l3557_355785

def total_judges : ℕ := 14
def valid_scores : ℕ := 7
def bribed_judges : ℕ := 2

def probability_no_bribed_judges : ℚ := 3/13

theorem no_bribed_judges_probability :
  (Nat.choose (total_judges - bribed_judges) valid_scores * Nat.choose bribed_judges 0) /
  Nat.choose total_judges valid_scores = probability_no_bribed_judges := by
  sorry

end NUMINAMATH_CALUDE_no_bribed_judges_probability_l3557_355785


namespace NUMINAMATH_CALUDE_fraction_equality_l3557_355731

theorem fraction_equality (a : ℕ+) : 
  (a : ℚ) / ((a : ℚ) + 35) = 865 / 1000 → a = 225 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3557_355731


namespace NUMINAMATH_CALUDE_square_plot_area_l3557_355776

/-- Proves that a square plot with given fence costs has an area of 144 square feet -/
theorem square_plot_area (cost_per_foot : ℝ) (total_cost : ℝ) : 
  cost_per_foot = 58 → total_cost = 2784 → 
  (total_cost / (4 * cost_per_foot))^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_area_l3557_355776


namespace NUMINAMATH_CALUDE_sin_210_degrees_l3557_355729

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l3557_355729


namespace NUMINAMATH_CALUDE_infinitely_many_planes_through_collinear_points_lines_in_different_planes_may_be_skew_unique_plane_through_parallel_lines_l3557_355711

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (on_line : Point → Line → Prop)
variable (on_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- Statement B
theorem infinitely_many_planes_through_collinear_points 
  (A B C : Point) (m : Line) 
  (h1 : on_line A m) (h2 : on_line B m) (h3 : on_line C m) :
  ∃ (P : Set Plane), Infinite P ∧ ∀ p ∈ P, on_plane m p :=
sorry

-- Statement C
theorem lines_in_different_planes_may_be_skew 
  (m n : Line) (α β : Plane) 
  (h1 : on_plane m α) (h2 : on_plane n β) :
  ∃ (skew : Line → Line → Prop), skew m n :=
sorry

-- Statement D
theorem unique_plane_through_parallel_lines 
  (m n : Line) (h : parallel m n) :
  ∃! p : Plane, on_plane m p ∧ on_plane n p :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_planes_through_collinear_points_lines_in_different_planes_may_be_skew_unique_plane_through_parallel_lines_l3557_355711


namespace NUMINAMATH_CALUDE_susan_stationery_purchase_l3557_355797

theorem susan_stationery_purchase (pencil_cost : ℚ) (pen_cost : ℚ) (total_spent : ℚ) (pencils_bought : ℕ) :
  pencil_cost = 25 / 100 →
  pen_cost = 80 / 100 →
  total_spent = 20 →
  pencils_bought = 16 →
  ∃ (pens_bought : ℕ),
    (pencils_bought : ℚ) * pencil_cost + (pens_bought : ℚ) * pen_cost = total_spent ∧
    pencils_bought + pens_bought = 36 :=
by sorry

end NUMINAMATH_CALUDE_susan_stationery_purchase_l3557_355797


namespace NUMINAMATH_CALUDE_bowling_ball_weights_l3557_355748

/-- The weight of a single canoe in pounds -/
def canoe_weight : ℕ := 36

/-- The number of bowling balls that weigh the same as the canoes -/
def num_bowling_balls : ℕ := 9

/-- The number of canoes that weigh the same as the bowling balls -/
def num_canoes : ℕ := 4

/-- The weight of a single bowling ball in pounds -/
def bowling_ball_weight : ℕ := canoe_weight * num_canoes / num_bowling_balls

/-- The total weight of five bowling balls in pounds -/
def five_bowling_balls_weight : ℕ := bowling_ball_weight * 5

theorem bowling_ball_weights :
  bowling_ball_weight = 16 ∧ five_bowling_balls_weight = 80 := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weights_l3557_355748


namespace NUMINAMATH_CALUDE_sum_of_integers_from_1_to_3_l3557_355751

theorem sum_of_integers_from_1_to_3 : 
  (Finset.range 3).sum (fun i => i + 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_from_1_to_3_l3557_355751


namespace NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l3557_355788

theorem tan_fifteen_ratio_equals_sqrt_three : 
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_ratio_equals_sqrt_three_l3557_355788


namespace NUMINAMATH_CALUDE_derivative_at_zero_l3557_355796

theorem derivative_at_zero (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2*x*(deriv f (-1))) :
  deriv f 0 = 4 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_zero_l3557_355796


namespace NUMINAMATH_CALUDE_f_derivative_f_extrema_log_inequality_l3557_355725

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x

-- State the theorems
theorem f_derivative (a : ℝ) (x : ℝ) (h : x ≠ 0) :
  deriv (f a) x = (a * x - 1) / (a * x^2) :=
sorry

theorem f_extrema (e : ℝ) (h_e : e > 0) :
  let f_1 := f 1
  ∃ (max_val min_val : ℝ),
    (∀ x ∈ Set.Icc (1/e) e, f_1 x ≤ max_val) ∧
    (∃ x ∈ Set.Icc (1/e) e, f_1 x = max_val) ∧
    (∀ x ∈ Set.Icc (1/e) e, f_1 x ≥ min_val) ∧
    (∃ x ∈ Set.Icc (1/e) e, f_1 x = min_val) ∧
    max_val = e - 2 ∧ min_val = 0 :=
sorry

theorem log_inequality (n : ℕ) (h : n > 1) :
  Real.log (n / (n - 1)) > 1 / n :=
sorry

end NUMINAMATH_CALUDE_f_derivative_f_extrema_log_inequality_l3557_355725


namespace NUMINAMATH_CALUDE_combined_rent_C_and_D_l3557_355734

-- Define the parameters for C and D
def oxen_C : ℕ := 15
def months_C : ℕ := 3
def rent_Z : ℕ := 100

def oxen_D : ℕ := 20
def months_D : ℕ := 6
def rent_W : ℕ := 120

-- Define the function to calculate rent
def calculate_rent (months : ℕ) (monthly_rent : ℕ) : ℕ :=
  months * monthly_rent

-- Theorem statement
theorem combined_rent_C_and_D :
  calculate_rent months_C rent_Z + calculate_rent months_D rent_W = 1020 := by
  sorry


end NUMINAMATH_CALUDE_combined_rent_C_and_D_l3557_355734


namespace NUMINAMATH_CALUDE_has_unique_prime_divisor_l3557_355738

theorem has_unique_prime_divisor (n m : ℕ) (h1 : n > m) (h2 : m > 0) :
  ∃ p : ℕ, Prime p ∧ (p ∣ (2^n - 1)) ∧ ¬(p ∣ (2^m - 1)) := by
  sorry

end NUMINAMATH_CALUDE_has_unique_prime_divisor_l3557_355738


namespace NUMINAMATH_CALUDE_total_food_count_l3557_355737

/-- The total number of hotdogs and hamburgers brought by neighbors -/
theorem total_food_count : ℕ := by
  -- Define the number of hotdogs brought by each neighbor
  let first_neighbor_hotdogs : ℕ := 75
  let second_neighbor_hotdogs : ℕ := first_neighbor_hotdogs - 25
  let third_neighbor_hotdogs : ℕ := 35
  let fourth_neighbor_hotdogs : ℕ := 2 * third_neighbor_hotdogs

  -- Define the number of hamburgers brought
  let one_neighbor_hamburgers : ℕ := 60
  let another_neighbor_hamburgers : ℕ := 3 * one_neighbor_hamburgers

  -- Calculate total hotdogs and hamburgers
  let total_hotdogs : ℕ := first_neighbor_hotdogs + second_neighbor_hotdogs + 
                           third_neighbor_hotdogs + fourth_neighbor_hotdogs
  let total_hamburgers : ℕ := one_neighbor_hamburgers + another_neighbor_hamburgers
  let total_food : ℕ := total_hotdogs + total_hamburgers

  -- Prove that the total is 470
  have : total_food = 470 := by sorry

  exact 470

end NUMINAMATH_CALUDE_total_food_count_l3557_355737


namespace NUMINAMATH_CALUDE_four_students_three_events_outcomes_l3557_355799

/-- The number of possible outcomes for champions in a competition --/
def championOutcomes (students : ℕ) (events : ℕ) : ℕ :=
  students ^ events

/-- Theorem: Given 4 students and 3 events, the number of possible outcomes for champions is 64 --/
theorem four_students_three_events_outcomes :
  championOutcomes 4 3 = 64 := by
  sorry

end NUMINAMATH_CALUDE_four_students_three_events_outcomes_l3557_355799


namespace NUMINAMATH_CALUDE_basketball_score_ratio_l3557_355740

/-- Given the scores of three basketball players, prove that the ratio of Tim's points to Ken's points is 1:2 -/
theorem basketball_score_ratio 
  (joe tim ken : ℕ)  -- Scores of Joe, Tim, and Ken
  (h1 : tim = joe + 20)  -- Tim scored 20 points more than Joe
  (h2 : joe + tim + ken = 100)  -- Total points scored is 100
  (h3 : tim = 30)  -- Tim scored 30 points
  : tim * 2 = ken :=
by sorry

end NUMINAMATH_CALUDE_basketball_score_ratio_l3557_355740


namespace NUMINAMATH_CALUDE_job_crop_production_l3557_355700

/-- Represents the land allocation of Job's farm --/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  future_expansion : ℕ
  cattle : ℕ

/-- Calculates the land used for crop production --/
def crop_production (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.future_expansion + farm.cattle)

/-- Theorem stating that Job's land used for crop production is 70 hectares --/
theorem job_crop_production :
  let job_farm : FarmLand := {
    total := 150,
    house_and_machinery := 25,
    future_expansion := 15,
    cattle := 40
  }
  crop_production job_farm = 70 := by sorry

end NUMINAMATH_CALUDE_job_crop_production_l3557_355700


namespace NUMINAMATH_CALUDE_complex_real_condition_l3557_355767

theorem complex_real_condition (a : ℝ) : 
  let z : ℂ := (a - 3 : ℂ) + (a^2 - 2*a - 3 : ℂ) * Complex.I
  (z.im = 0) → (a = 3 ∨ a = -1) := by
sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3557_355767


namespace NUMINAMATH_CALUDE_local_max_range_l3557_355755

def f' (x a : ℝ) : ℝ := a * (x + 1) * (x - a)

theorem local_max_range (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, (deriv f) x = f' x a)
  (h2 : IsLocalMax f a) :
  -1 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_local_max_range_l3557_355755


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3557_355727

theorem smallest_number_satisfying_conditions : 
  ∃ N : ℕ, 
    N > 0 ∧ 
    N % 4 = 0 ∧ 
    (N + 9) % 2 = 1 ∧ 
    (∀ M : ℕ, M > 0 → M % 4 = 0 → (M + 9) % 2 = 1 → M ≥ N) ∧
    N = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3557_355727


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3557_355783

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3557_355783


namespace NUMINAMATH_CALUDE_faye_apps_left_l3557_355708

/-- The number of apps left after deletion -/
def apps_left (initial : ℕ) (deleted : ℕ) : ℕ :=
  initial - deleted

/-- Theorem stating that Faye has 4 apps left -/
theorem faye_apps_left : apps_left 12 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_faye_apps_left_l3557_355708


namespace NUMINAMATH_CALUDE_square_area_proof_l3557_355766

theorem square_area_proof (x : ℚ) : 
  (5 * x - 20 = 25 - 2 * x) → 
  ((5 * x - 20)^2 : ℚ) = 7225 / 49 := by
sorry

end NUMINAMATH_CALUDE_square_area_proof_l3557_355766


namespace NUMINAMATH_CALUDE_special_collection_loans_l3557_355717

theorem special_collection_loans (initial_books : ℕ) (final_books : ℕ) (return_rate : ℚ) :
  initial_books = 75 →
  final_books = 66 →
  return_rate = 70 / 100 →
  ∃ (loaned_books : ℕ), loaned_books = 30 ∧ 
    final_books = initial_books - (1 - return_rate) * loaned_books := by
  sorry

end NUMINAMATH_CALUDE_special_collection_loans_l3557_355717


namespace NUMINAMATH_CALUDE_triangle_area_13_14_15_l3557_355784

/-- The area of a triangle with sides 13, 14, and 15 is 84 -/
theorem triangle_area_13_14_15 : ∃ (area : ℝ), area = 84 ∧ 
  (∀ (s : ℝ), s = (13 + 14 + 15) / 2 → 
    area = Real.sqrt (s * (s - 13) * (s - 14) * (s - 15))) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_13_14_15_l3557_355784


namespace NUMINAMATH_CALUDE_sum_of_max_min_values_l3557_355739

theorem sum_of_max_min_values (f : ℝ → ℝ) (h : f = fun x ↦ 9 * (Real.cos x)^4 + 12 * (Real.sin x)^2 - 4) :
  (⨆ x, f x) + (⨅ x, f x) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_values_l3557_355739


namespace NUMINAMATH_CALUDE_playground_insects_l3557_355749

/-- Calculates the number of remaining insects in the playground --/
def remaining_insects (initial_bees initial_beetles initial_ants initial_termites
                       initial_praying_mantises initial_ladybugs initial_butterflies
                       initial_dragonflies : ℕ)
                      (bees_left beetles_taken ants_left termites_moved
                       ladybugs_left butterflies_left dragonflies_left : ℕ) : ℕ :=
  (initial_bees - bees_left) +
  (initial_beetles - beetles_taken) +
  (initial_ants - ants_left) +
  (initial_termites - termites_moved) +
  initial_praying_mantises +
  (initial_ladybugs - ladybugs_left) +
  (initial_butterflies - butterflies_left) +
  (initial_dragonflies - dragonflies_left)

/-- Theorem stating that the number of remaining insects is 54 --/
theorem playground_insects :
  remaining_insects 15 7 12 10 2 10 11 8 6 2 4 3 2 3 1 = 54 := by
  sorry

end NUMINAMATH_CALUDE_playground_insects_l3557_355749


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l3557_355753

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, x ≤ 3 ↔ (x : ℝ)^4 / (x : ℝ)^2 < 15 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l3557_355753


namespace NUMINAMATH_CALUDE_puzzle_pieces_l3557_355712

theorem puzzle_pieces (total_pieces : ℕ) (edge_difference : ℕ) (non_red_decrease : ℕ) : 
  total_pieces = 91 → 
  edge_difference = 24 → 
  non_red_decrease = 2 → 
  ∃ (red_pieces : ℕ) (non_red_pieces : ℕ), 
    red_pieces + non_red_pieces = total_pieces ∧ 
    non_red_pieces * non_red_decrease = edge_difference ∧
    red_pieces = 79 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_pieces_l3557_355712


namespace NUMINAMATH_CALUDE_parabola_translation_l3557_355781

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = x^2 --/
def original_parabola : Parabola := { a := 1, b := 0, c := 0 }

/-- Translates a parabola vertically --/
def translate_vertical (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + d }

/-- Translates a parabola horizontally --/
def translate_horizontal (p : Parabola) (d : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * d + p.b, c := p.a * d^2 - p.b * d + p.c }

/-- The resulting parabola after translations --/
def result_parabola : Parabola :=
  translate_horizontal (translate_vertical original_parabola 3) 5

theorem parabola_translation :
  result_parabola.a = 1 ∧
  result_parabola.b = -10 ∧
  result_parabola.c = 28 := by
  sorry

#check parabola_translation

end NUMINAMATH_CALUDE_parabola_translation_l3557_355781


namespace NUMINAMATH_CALUDE_cubic_root_form_l3557_355769

theorem cubic_root_form : ∃ (x : ℝ), 
  8 * x^3 - 3 * x^2 - 3 * x - 1 = 0 ∧ 
  x = (Real.rpow 81 (1/3 : ℝ) + Real.rpow 9 (1/3 : ℝ) + 1) / 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_form_l3557_355769


namespace NUMINAMATH_CALUDE_age_difference_l3557_355798

theorem age_difference (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : 3 * b = 2 * a) (h4 : a + b = 60) : a - b = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3557_355798


namespace NUMINAMATH_CALUDE_different_size_circles_not_one_tangent_l3557_355732

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_positive : radius > 0

-- Define the number of common tangents between two circles
def num_common_tangents (c1 c2 : Circle) : ℕ := sorry

-- Theorem statement
theorem different_size_circles_not_one_tangent (c1 c2 : Circle) :
  c1.radius ≠ c2.radius →
  num_common_tangents c1 c2 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_different_size_circles_not_one_tangent_l3557_355732


namespace NUMINAMATH_CALUDE_smallest_number_with_properties_l3557_355710

theorem smallest_number_with_properties : 
  ∃ (n : ℕ), n = 153846 ∧ 
  (∀ m : ℕ, m < n → 
    (m % 10 = 6 ∧ 
     ∃ k : ℕ, 6 * 10^k + (m - 6) / 10 = 4 * m) → False) ∧
  n % 10 = 6 ∧
  ∃ k : ℕ, 6 * 10^k + (n - 6) / 10 = 4 * n :=
sorry

end NUMINAMATH_CALUDE_smallest_number_with_properties_l3557_355710


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3557_355770

theorem cubic_equation_solution (y : ℝ) (h : y ≠ 0) :
  (3 * y)^5 = (9 * y)^4 ↔ y = 27 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3557_355770


namespace NUMINAMATH_CALUDE_five_pq_odd_l3557_355715

theorem five_pq_odd (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) :
  Odd (5 * p * q) := by
  sorry

end NUMINAMATH_CALUDE_five_pq_odd_l3557_355715


namespace NUMINAMATH_CALUDE_alien_invasion_characteristics_l3557_355765

-- Define the characteristics of an alien species invasion
structure AlienInvasion where
  j_shaped_growth : Bool
  unrestricted_growth : Bool
  threatens_biodiversity : Bool
  eliminated_if_unadapted : Bool

-- Define the correct characteristics of an alien invasion
def correct_invasion : AlienInvasion :=
  { j_shaped_growth := true,
    unrestricted_growth := false,
    threatens_biodiversity := true,
    eliminated_if_unadapted := true }

-- Theorem: The correct characteristics of an alien invasion are as defined
theorem alien_invasion_characteristics :
  ∃ (invasion : AlienInvasion),
    invasion.j_shaped_growth ∧
    ¬invasion.unrestricted_growth ∧
    invasion.threatens_biodiversity ∧
    invasion.eliminated_if_unadapted :=
by
  sorry


end NUMINAMATH_CALUDE_alien_invasion_characteristics_l3557_355765


namespace NUMINAMATH_CALUDE_annie_laps_bonnie_l3557_355756

/-- The length of the circular track in meters -/
def track_length : ℝ := 500

/-- Annie's speed relative to Bonnie's -/
def annie_speed_ratio : ℝ := 1.5

/-- The number of laps Annie has run when she first laps Bonnie -/
def annie_laps : ℝ := 3

theorem annie_laps_bonnie :
  track_length > 0 →
  annie_speed_ratio = 1.5 →
  (annie_laps * track_length) / annie_speed_ratio = (annie_laps - 1) * track_length :=
by sorry

end NUMINAMATH_CALUDE_annie_laps_bonnie_l3557_355756


namespace NUMINAMATH_CALUDE_inscribed_sphere_ratio_l3557_355705

/-- A regular tetrahedron with height H and an inscribed sphere of radius R -/
structure RegularTetrahedron where
  H : ℝ
  R : ℝ
  H_pos : H > 0
  R_pos : R > 0

/-- The ratio of the inscribed sphere radius to the tetrahedron height is 1:4 -/
theorem inscribed_sphere_ratio (t : RegularTetrahedron) : t.R / t.H = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_ratio_l3557_355705


namespace NUMINAMATH_CALUDE_hot_dog_sales_l3557_355745

theorem hot_dog_sales (total : ℕ) (first_innings : ℕ) (left_unsold : ℕ) :
  total = 91 →
  first_innings = 19 →
  left_unsold = 45 →
  total - (first_innings + left_unsold) = 27 := by
  sorry

end NUMINAMATH_CALUDE_hot_dog_sales_l3557_355745


namespace NUMINAMATH_CALUDE_five_roots_sum_l3557_355722

noncomputable def f (x : ℝ) : ℝ :=
  if x = 2 then 1 else Real.log (abs (x - 2))

theorem five_roots_sum (b c : ℝ) 
  (h : ∃ x₁ x₂ x₃ x₄ x₅ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧ 
                           x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
                           x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧ x₄ ≠ x₅ ∧
                           (f x₁)^2 + b * (f x₁) + c = 0 ∧
                           (f x₂)^2 + b * (f x₂) + c = 0 ∧
                           (f x₃)^2 + b * (f x₃) + c = 0 ∧
                           (f x₄)^2 + b * (f x₄) + c = 0 ∧
                           (f x₅)^2 + b * (f x₅) + c = 0) :
  ∃ x₁ x₂ x₃ x₄ x₅ : ℝ, f (x₁ + x₂ + x₃ + x₄ + x₅) = 3 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_five_roots_sum_l3557_355722


namespace NUMINAMATH_CALUDE_seventh_power_of_complex_l3557_355790

theorem seventh_power_of_complex (z : ℂ) : 
  z = (Real.sqrt 3 + Complex.I) / 2 → z^7 = -Real.sqrt 3 / 2 - Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_power_of_complex_l3557_355790


namespace NUMINAMATH_CALUDE_triple_hash_72_l3557_355754

-- Define the # operation
def hash (N : ℝ) : ℝ := 0.5 * N - 1

-- Theorem statement
theorem triple_hash_72 : hash (hash (hash 72)) = 7.25 := by
  sorry

end NUMINAMATH_CALUDE_triple_hash_72_l3557_355754


namespace NUMINAMATH_CALUDE_operation_result_l3557_355791

-- Define a type for the allowed operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

theorem operation_result 
  (diamond circ : Operation) 
  (h : (apply_op diamond 20 4) / (apply_op circ 12 4) = 2) :
  (apply_op diamond 9 3) / (apply_op circ 15 5) = 27 / 20 :=
by sorry

end NUMINAMATH_CALUDE_operation_result_l3557_355791


namespace NUMINAMATH_CALUDE_sum_property_unique_l3557_355758

/-- The property that the sum of the first n natural numbers can be written as n followed by three digits in base 10 -/
def sum_property (n : ℕ) : Prop :=
  ∃ k : ℕ, k < 1000 ∧ (n * (n + 1)) / 2 = 1000 * n + k

/-- Theorem stating that 1999 is the only natural number satisfying the sum property -/
theorem sum_property_unique : ∀ n : ℕ, sum_property n ↔ n = 1999 :=
sorry

end NUMINAMATH_CALUDE_sum_property_unique_l3557_355758


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3557_355747

theorem sqrt_equation_solution (z : ℝ) :
  (Real.sqrt 1.5 / Real.sqrt 0.81 + Real.sqrt z / Real.sqrt 0.49 = 3.0751133491652576) →
  z = 1.44 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3557_355747


namespace NUMINAMATH_CALUDE_min_sum_given_product_l3557_355789

theorem min_sum_given_product (a b : ℝ) : 
  a > 0 → b > 0 → a + b = a * b → (∀ x y : ℝ, x > 0 → y > 0 → x + y = x * y → a + b ≤ x + y) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l3557_355789


namespace NUMINAMATH_CALUDE_shaded_area_proof_l3557_355760

theorem shaded_area_proof (square_side : ℝ) (triangle_side : ℝ) : 
  square_side = 40 →
  triangle_side = 25 →
  square_side^2 - 2 * (1/2 * triangle_side^2) = 975 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_proof_l3557_355760
