import Mathlib

namespace fraction_of_three_fourths_half_5060_l2777_277752

theorem fraction_of_three_fourths_half_5060 : 
  let total := (3/4 : ℚ) * (1/2 : ℚ) * 5060
  759.0000000000001 / total = 0.4 := by sorry

end fraction_of_three_fourths_half_5060_l2777_277752


namespace fractional_equation_solution_l2777_277717

theorem fractional_equation_solution :
  ∃ (x : ℝ), (3 / (x - 3) - 1 = 1 / (3 - x)) ∧ (x = 7) :=
by
  sorry

end fractional_equation_solution_l2777_277717


namespace marking_exists_l2777_277710

/-- Represents a 50x50 board with some cells occupied -/
def Board := Fin 50 → Fin 50 → Bool

/-- Represents a marking of free cells on the board -/
def Marking := Fin 50 → Fin 50 → Bool

/-- Check if a marking is valid (at most 99 cells marked) -/
def valid_marking (b : Board) (m : Marking) : Prop :=
  (Finset.sum Finset.univ (fun i => Finset.sum Finset.univ (fun j => if m i j then 1 else 0))) ≤ 99

/-- Check if the total number of marked and originally occupied cells in a row is even -/
def row_even (b : Board) (m : Marking) (i : Fin 50) : Prop :=
  Even (Finset.sum Finset.univ (fun j => if b i j || m i j then 1 else 0))

/-- Check if the total number of marked and originally occupied cells in a column is even -/
def col_even (b : Board) (m : Marking) (j : Fin 50) : Prop :=
  Even (Finset.sum Finset.univ (fun i => if b i j || m i j then 1 else 0))

/-- Main theorem: For any board configuration, there exists a valid marking that makes all rows and columns even -/
theorem marking_exists (b : Board) : ∃ m : Marking, 
  valid_marking b m ∧ 
  (∀ i : Fin 50, row_even b m i) ∧ 
  (∀ j : Fin 50, col_even b m j) := by
  sorry

end marking_exists_l2777_277710


namespace solution_set_part1_range_of_a_part2_l2777_277736

-- Part 1
def f (x : ℝ) : ℝ := |x - 1| - 2

theorem solution_set_part1 :
  {x : ℝ | f x + |2*x - 3| > 0} = {x : ℝ | x > 2 ∨ x < 2/3} := by sorry

-- Part 2
def g (a x : ℝ) : ℝ := |x - a| - 2

theorem range_of_a_part2 (a : ℝ) :
  (∃ x, g a x > |x - 3|) → a < 1 ∨ a > 5 := by sorry

end solution_set_part1_range_of_a_part2_l2777_277736


namespace chord_length_sum_l2777_277737

/-- Representation of a circle with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Check if a circle is internally tangent to another -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c2.radius - c1.radius)^2

/-- Check if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Main theorem -/
theorem chord_length_sum (C1 C2 C3 : Circle) (m n p : ℕ) : 
  C1.radius = 4 →
  C2.radius = 10 →
  are_externally_tangent C1 C2 →
  is_internally_tangent C1 C3 →
  is_internally_tangent C2 C3 →
  are_collinear C1.center C2.center C3.center →
  (m.gcd p = 1) →
  (∀ q : ℕ, Prime q → n % (q^2) ≠ 0) →
  (∃ (chord_length : ℝ), chord_length = m * Real.sqrt n / p ∧ 
    chord_length^2 = 4 * (C3.radius^2 - ((C3.radius - C1.radius) * (C3.radius - C2.radius) / (C1.radius + C2.radius))^2)) →
  m + n + p = 405 := by
  sorry

end chord_length_sum_l2777_277737


namespace managers_wage_l2777_277766

/-- Proves that the manager's hourly wage is $7.50 given the wage relationships between manager, chef, and dishwasher -/
theorem managers_wage (manager chef dishwasher : ℝ) 
  (h1 : chef = dishwasher * 1.2)
  (h2 : dishwasher = manager / 2)
  (h3 : chef = manager - 3) :
  manager = 7.5 := by
sorry

end managers_wage_l2777_277766


namespace soccer_league_games_l2777_277716

/-- The number of games played in a soccer league -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a soccer league with 14 teams, where each team plays every other team once,
    the total number of games played is 91. -/
theorem soccer_league_games :
  num_games 14 = 91 := by
  sorry

end soccer_league_games_l2777_277716


namespace infinitely_many_winning_starts_l2777_277706

theorem infinitely_many_winning_starts : 
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ 
    ¬∃ k : ℕ, n = k^2 ∧
    ¬∃ k : ℕ, n + (n + 1) = k^2 ∧
    ∃ k : ℕ, (n + (n + 1)) + (n + 2) = k^2 :=
by sorry

end infinitely_many_winning_starts_l2777_277706


namespace ellipse_k_range_l2777_277763

/-- An ellipse with equation x^2 + ky^2 = 2 and foci on the y-axis -/
structure Ellipse where
  k : ℝ
  eq : ∀ x y : ℝ, x^2 + k * y^2 = 2
  foci_on_y : True  -- This is a placeholder for the foci condition

/-- The range of k for a valid ellipse with foci on the y-axis -/
def valid_k_range (e : Ellipse) : Prop :=
  0 < e.k ∧ e.k < 1

/-- Theorem stating that for any ellipse with the given properties, k must be in (0, 1) -/
theorem ellipse_k_range (e : Ellipse) : valid_k_range e := by
  sorry

end ellipse_k_range_l2777_277763


namespace manu_win_probability_l2777_277700

def coin_flip_game (num_players : ℕ) (manu_position : ℕ) (manu_heads_needed : ℕ) : ℚ :=
  sorry

theorem manu_win_probability :
  coin_flip_game 4 4 2 = 1 / 30 := by sorry

end manu_win_probability_l2777_277700


namespace total_net_increase_l2777_277757

/-- Represents a time period with birth and death rates -/
structure TimePeriod where
  birthRate : Nat
  deathRate : Nat

/-- Calculates the net population increase for a given time period -/
def netIncrease (tp : TimePeriod) : Nat :=
  (tp.birthRate - tp.deathRate) * 10800

/-- The four time periods in a day -/
def dayPeriods : List TimePeriod := [
  { birthRate := 4, deathRate := 3 },
  { birthRate := 8, deathRate := 3 },
  { birthRate := 10, deathRate := 4 },
  { birthRate := 6, deathRate := 2 }
]

/-- Theorem: The total net population increase in one day is 172,800 -/
theorem total_net_increase : 
  (dayPeriods.map netIncrease).sum = 172800 := by
  sorry

end total_net_increase_l2777_277757


namespace total_count_equals_115248_l2777_277721

/-- The number of digits that can be used (excluding 3, 6, and 9) -/
def available_digits : ℕ := 7

/-- The number of non-zero digits that can be used as the first digit -/
def first_digit_choices : ℕ := 6

/-- Calculates the number of n-digit numbers without 3, 6, or 9 -/
def count_numbers (n : ℕ) : ℕ :=
  first_digit_choices * available_digits^(n - 1)

/-- The total number of 5 and 6-digit numbers without 3, 6, or 9 -/
def total_count : ℕ := count_numbers 5 + count_numbers 6

theorem total_count_equals_115248 : total_count = 115248 := by
  sorry

end total_count_equals_115248_l2777_277721


namespace minimum_cans_for_target_gallons_l2777_277772

/-- The number of ounces in one gallon -/
def ounces_per_gallon : ℕ := 128

/-- The number of ounces each can holds -/
def ounces_per_can : ℕ := 16

/-- The number of gallons we want to have at least -/
def target_gallons : ℚ := 3/2

theorem minimum_cans_for_target_gallons :
  let total_ounces := (target_gallons * ounces_per_gallon).ceil
  let num_cans := (total_ounces + ounces_per_can - 1) / ounces_per_can
  num_cans = 12 := by sorry

end minimum_cans_for_target_gallons_l2777_277772


namespace no_intersection_condition_l2777_277727

theorem no_intersection_condition (k : ℝ) : 
  -1 ≤ k ∧ k ≤ 1 → 
  (∀ x : ℝ, x = k * π / 2 → ¬∃ y : ℝ, y = Real.tan (2 * x + π / 4)) ↔ 
  (k = 1 / 4 ∨ k = -3 / 4) := by
sorry

end no_intersection_condition_l2777_277727


namespace rectangle_area_l2777_277713

/-- A rectangle with length thrice its breadth and perimeter 48 meters has an area of 108 square meters. -/
theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 48) : l * b = 108 := by
  sorry

end rectangle_area_l2777_277713


namespace max_knights_and_courtiers_l2777_277770

/-- Represents the number of people at the king's table -/
def kings_table : ℕ := 7

/-- Represents the minimum number of courtiers -/
def min_courtiers : ℕ := 12

/-- Represents the maximum number of courtiers -/
def max_courtiers : ℕ := 18

/-- Represents the minimum number of knights -/
def min_knights : ℕ := 10

/-- Represents the maximum number of knights -/
def max_knights : ℕ := 20

/-- Represents the rule that the lunch of a knight plus the lunch of a courtier equals the lunch of the king -/
def lunch_rule (courtiers knights : ℕ) : Prop :=
  (1 : ℚ) / courtiers + (1 : ℚ) / knights = (1 : ℚ) / kings_table

/-- The main theorem stating the maximum number of knights and courtiers -/
theorem max_knights_and_courtiers :
  ∃ (k c : ℕ), 
    min_courtiers ≤ c ∧ c ≤ max_courtiers ∧
    min_knights ≤ k ∧ k ≤ max_knights ∧
    lunch_rule c k ∧
    (∀ (k' c' : ℕ), 
      min_courtiers ≤ c' ∧ c' ≤ max_courtiers ∧
      min_knights ≤ k' ∧ k' ≤ max_knights ∧
      lunch_rule c' k' →
      k' ≤ k) ∧
    k = 14 ∧ c = 14 :=
  sorry

end max_knights_and_courtiers_l2777_277770


namespace triangle_larger_segment_is_82_5_l2777_277765

/-- A triangle with sides a, b, c, where c is the longest side --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  h_c_longest : c ≥ a ∧ c ≥ b

/-- The angle opposite to the longest side of the triangle --/
def Triangle.angle_opposite_longest (t : Triangle) : ℝ := sorry

/-- The altitude to the longest side of the triangle --/
def Triangle.altitude_to_longest (t : Triangle) : ℝ := sorry

/-- The larger segment cut off by the altitude on the longest side --/
def Triangle.larger_segment (t : Triangle) : ℝ := sorry

theorem triangle_larger_segment_is_82_5 (t : Triangle) 
  (h_sides : t.a = 40 ∧ t.b = 90 ∧ t.c = 100) 
  (h_angle : t.angle_opposite_longest = Real.pi / 3) : 
  t.larger_segment = 82.5 := by sorry

end triangle_larger_segment_is_82_5_l2777_277765


namespace range_of_a_for_always_positive_quadratic_l2777_277711

theorem range_of_a_for_always_positive_quadratic :
  ∀ (a : ℝ), (∀ (x : ℝ), a * x^2 - 3 * a * x + 9 > 0) ↔ (0 ≤ a ∧ a < 4) := by
  sorry

end range_of_a_for_always_positive_quadratic_l2777_277711


namespace min_value_of_h_neg_infinity_to_zero_l2777_277745

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- The function h(x) defined in terms of f(x) and g(x) -/
def h (f g : ℝ → ℝ) (a b : ℝ) (x : ℝ) : ℝ := a * f x ^ 3 - b * g x - 2

theorem min_value_of_h_neg_infinity_to_zero 
  (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : IsOdd f) (hg : IsOdd g)
  (hmax : ∃ x > 0, ∀ y > 0, h f g a b y ≤ h f g a b x ∧ h f g a b x = 5) :
  ∃ x < 0, ∀ y < 0, h f g a b y ≥ h f g a b x ∧ h f g a b x = -9 :=
sorry

end min_value_of_h_neg_infinity_to_zero_l2777_277745


namespace triangle_right_angle_l2777_277733

theorem triangle_right_angle (a b : ℝ) (A B : Real) (h : a + b = a / Real.tan A + b / Real.tan B) :
  A + B = π / 2 := by
  sorry

end triangle_right_angle_l2777_277733


namespace reflection_distance_l2777_277758

/-- Given a point A with coordinates (1, -3), prove that the distance between A
    and its reflection A' over the y-axis is 2. -/
theorem reflection_distance : 
  let A : ℝ × ℝ := (1, -3)
  let A' : ℝ × ℝ := (-1, -3)  -- Reflection of A over y-axis
  ‖A - A'‖ = 2 := by
  sorry

end reflection_distance_l2777_277758


namespace author_writing_speed_l2777_277715

/-- Calculates the average words written per hour given total words, total hours, and break hours. -/
def average_words_per_hour (total_words : ℕ) (total_hours : ℕ) (break_hours : ℕ) : ℚ :=
  total_words / (total_hours - break_hours)

/-- Theorem stating that given the specific conditions, the average words per hour is 550. -/
theorem author_writing_speed :
  average_words_per_hour 55000 120 20 = 550 := by
  sorry

end author_writing_speed_l2777_277715


namespace final_retail_price_l2777_277781

/-- Calculates the final retail price of a machine given wholesale price, markup, discount, and desired profit percentage. -/
theorem final_retail_price
  (wholesale_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (desired_profit_percentage : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : markup_percentage = 1)
  (h3 : discount_percentage = 0.2)
  (h4 : desired_profit_percentage = 0.6)
  : wholesale_price * (1 + markup_percentage) * (1 - discount_percentage) = 144 :=
by sorry

end final_retail_price_l2777_277781


namespace negation_of_proposition_cubic_inequality_negation_l2777_277759

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem cubic_inequality_negation : 
  (¬∀ x : ℝ, x^3 + 2 < 0) ↔ (∃ x : ℝ, x^3 + 2 ≥ 0) :=
by sorry

end negation_of_proposition_cubic_inequality_negation_l2777_277759


namespace fred_seashells_l2777_277720

def seashells_problem (initial_seashells given_away_seashells : ℕ) : Prop :=
  initial_seashells - given_away_seashells = 22

theorem fred_seashells : seashells_problem 47 25 := by
  sorry

end fred_seashells_l2777_277720


namespace problem_solution_l2777_277719

theorem problem_solution (m n : ℝ) : 
  (∃ k : ℝ, k^2 = 3*m + 1 ∧ (k = 2 ∨ k = -2)) →
  (∃ l : ℝ, l^3 = 5*n - 2 ∧ l = 2) →
  m = 1 ∧ n = 2 ∧ (∃ r : ℝ, r^2 = 4*m + 5/2*n ∧ (r = 3 ∨ r = -3)) :=
by sorry

end problem_solution_l2777_277719


namespace marble_distribution_l2777_277783

theorem marble_distribution (n : ℕ) (hn : n = 480) :
  (Finset.filter (fun m => m > 1 ∧ m < n) (Finset.range (n + 1))).card = 22 :=
by sorry

end marble_distribution_l2777_277783


namespace mika_bought_26_stickers_l2777_277726

/-- Represents the number of stickers Mika has at different stages -/
structure StickerCount where
  initial : Nat
  birthday : Nat
  given_away : Nat
  used : Nat
  remaining : Nat

/-- Calculates the number of stickers Mika bought from the store -/
def stickers_bought (s : StickerCount) : Nat :=
  s.remaining + s.given_away + s.used - s.initial - s.birthday

/-- Theorem stating that Mika bought 26 stickers from the store -/
theorem mika_bought_26_stickers (s : StickerCount) 
  (h1 : s.initial = 20)
  (h2 : s.birthday = 20)
  (h3 : s.given_away = 6)
  (h4 : s.used = 58)
  (h5 : s.remaining = 2) :
  stickers_bought s = 26 := by
  sorry

#eval stickers_bought { initial := 20, birthday := 20, given_away := 6, used := 58, remaining := 2 }

end mika_bought_26_stickers_l2777_277726


namespace island_not_maya_l2777_277748

-- Define the possible states for an inhabitant
inductive InhabitantState
  | Knight
  | Knave

-- Define the island name
structure IslandName where
  name : String

-- Define the statements made by the inhabitants
def statement_A (state_A state_B : InhabitantState) (island : IslandName) : Prop :=
  (state_A = InhabitantState.Knave ∨ state_B = InhabitantState.Knave) ∧ island.name = "Maya"

def statement_B (state_A state_B : InhabitantState) (island : IslandName) : Prop :=
  statement_A state_A state_B island

-- Define the truthfulness of statements based on the inhabitant's state
def is_truthful (state : InhabitantState) (statement : Prop) : Prop :=
  (state = InhabitantState.Knight ∧ statement) ∨ (state = InhabitantState.Knave ∧ ¬statement)

-- Theorem statement
theorem island_not_maya (state_A state_B : InhabitantState) (island : IslandName) :
  (is_truthful state_A (statement_A state_A state_B island) ∧
   is_truthful state_B (statement_B state_A state_B island)) →
  island.name ≠ "Maya" :=
by sorry

end island_not_maya_l2777_277748


namespace scientific_notation_of_8200000_l2777_277734

theorem scientific_notation_of_8200000 :
  (8200000 : ℝ) = 8.2 * (10 ^ 6) := by sorry

end scientific_notation_of_8200000_l2777_277734


namespace seats_filled_percentage_l2777_277779

/-- Given a hall with total seats and vacant seats, calculate the percentage of filled seats -/
def percentage_filled (total_seats vacant_seats : ℕ) : ℚ :=
  (total_seats - vacant_seats : ℚ) / total_seats * 100

/-- Theorem: In a hall with 600 seats where 300 are vacant, 50% of the seats are filled -/
theorem seats_filled_percentage :
  percentage_filled 600 300 = 50 := by
  sorry

#eval percentage_filled 600 300

end seats_filled_percentage_l2777_277779


namespace exists_divisor_in_range_l2777_277754

theorem exists_divisor_in_range : ∃ n : ℕ, 
  100 ≤ n ∧ n ≤ 1997 ∧ (n ∣ 2 * n + 2) ∧ n = 946 := by
  sorry

end exists_divisor_in_range_l2777_277754


namespace computer_table_markup_l2777_277714

/-- The percentage markup on a product's cost price, given the selling price and cost price. -/
def percentageMarkup (sellingPrice costPrice : ℚ) : ℚ :=
  (sellingPrice - costPrice) / costPrice * 100

/-- Proof that the percentage markup on a computer table is 30% -/
theorem computer_table_markup :
  percentageMarkup 8450 6500 = 30 := by
  sorry

end computer_table_markup_l2777_277714


namespace percentage_men_not_speaking_french_or_spanish_l2777_277773

theorem percentage_men_not_speaking_french_or_spanish :
  let total_men_percentage : ℚ := 100
  let french_speaking_men_percentage : ℚ := 55
  let spanish_speaking_men_percentage : ℚ := 35
  let other_languages_men_percentage : ℚ := 10
  (total_men_percentage = french_speaking_men_percentage + spanish_speaking_men_percentage + other_languages_men_percentage) →
  (other_languages_men_percentage = 10) :=
by sorry

end percentage_men_not_speaking_french_or_spanish_l2777_277773


namespace product_range_l2777_277723

theorem product_range (a b : ℝ) (g : ℝ → ℝ) (h₁ : a > 0) (h₂ : b > 0) 
  (h₃ : g = fun x => 2^x) (h₄ : g a * g b = 2) : 
  0 < a * b ∧ a * b ≤ 1/4 := by
sorry

end product_range_l2777_277723


namespace cookie_radius_l2777_277728

theorem cookie_radius (x y : ℝ) : 
  (x^2 + y^2 - 6.5 = x + 3*y) → 
  ∃ (h k r : ℝ), r = 3 ∧ (x - h)^2 + (y - k)^2 = r^2 := by
  sorry

end cookie_radius_l2777_277728


namespace max_prime_value_l2777_277784

theorem max_prime_value (a b : ℕ) (p : ℕ) (h_prime : Nat.Prime p) 
  (h_eq : p = (b / 4) * Real.sqrt ((2 * a - b) / (2 * a + b))) : 
  p ≤ 5 ∧ ∃ (a' b' : ℕ), (5 : ℕ) = (b' / 4) * Real.sqrt ((2 * a' - b') / (2 * a' + b')) := by
  sorry

end max_prime_value_l2777_277784


namespace geometric_place_of_tangent_points_l2777_277707

/-- Given a circle with center O(0,0) and radius r in a right-angled coordinate system,
    the geometric place of points S(x,y) whose adjoint lines are tangents to the circle
    is defined by the equation 1/x^2 + 1/y^2 = 1/r^2 -/
theorem geometric_place_of_tangent_points (r : ℝ) (h : r > 0) :
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 →
    (∃ x₁ y₁ : ℝ, x₁^2 + y₁^2 = r^2 ∧ x₁ * x + y₁ * y = r^2) ↔
    1 / x^2 + 1 / y^2 = 1 / r^2 :=
by sorry

end geometric_place_of_tangent_points_l2777_277707


namespace books_per_student_l2777_277704

theorem books_per_student (total_books : ℕ) (students_day1 students_day2 students_day3 students_day4 : ℕ) : 
  total_books = 120 →
  students_day1 = 4 →
  students_day2 = 5 →
  students_day3 = 6 →
  students_day4 = 9 →
  total_books / (students_day1 + students_day2 + students_day3 + students_day4) = 5 := by
sorry

end books_per_student_l2777_277704


namespace triangle_side_product_l2777_277791

noncomputable def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_side_product (a b c : ℝ) :
  Triangle a b c →
  (a + b)^2 - c^2 = 4 →
  Real.cos (60 * π / 180) = 1/2 →
  a * b = 4/3 := by
  sorry

end triangle_side_product_l2777_277791


namespace intersection_point_l2777_277746

-- Define the two linear functions
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := -2*x + 6

-- State the theorem
theorem intersection_point :
  ∃! p : ℝ × ℝ, f p.1 = p.2 ∧ g p.1 = p.2 ∧ p = (1, 4) := by
  sorry

end intersection_point_l2777_277746


namespace triangle_inequality_l2777_277743

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (-a^2 + b^2 + c^2) * (a^2 - b^2 + c^2) * (a^2 + b^2 - c^2) ≤ a^2 * b^2 * c^2 := by
  sorry

end triangle_inequality_l2777_277743


namespace republican_votes_for_candidate_a_l2777_277795

theorem republican_votes_for_candidate_a (total_voters : ℝ) 
  (h1 : total_voters > 0) 
  (democrat_percent : ℝ) 
  (h2 : democrat_percent = 0.60)
  (republican_percent : ℝ) 
  (h3 : republican_percent = 1 - democrat_percent)
  (democrat_votes_for_a_percent : ℝ) 
  (h4 : democrat_votes_for_a_percent = 0.65)
  (total_votes_for_a_percent : ℝ) 
  (h5 : total_votes_for_a_percent = 0.47) : 
  (total_votes_for_a_percent * total_voters - democrat_votes_for_a_percent * democrat_percent * total_voters) / 
  (republican_percent * total_voters) = 0.20 := by
sorry

end republican_votes_for_candidate_a_l2777_277795


namespace regression_line_not_necessarily_through_points_l2777_277782

/-- Sample data point -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Linear regression model -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Predicts y value for a given x using the linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.a + model.b * x

/-- Checks if a point lies on the regression line -/
def pointOnLine (model : LinearRegression) (point : DataPoint) : Prop :=
  predict model point.x = point.y

theorem regression_line_not_necessarily_through_points 
  (model : LinearRegression) (data : List DataPoint) : 
  ¬ (∀ point ∈ data, pointOnLine model point) := by
  sorry

#check regression_line_not_necessarily_through_points

end regression_line_not_necessarily_through_points_l2777_277782


namespace min_distance_to_plane_l2777_277762

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  d : ℝ

/-- The distance between a point and a plane -/
def distPointToPlane (p : Point3D) (plane : Plane) : ℝ :=
  sorry

/-- The distance between two points -/
def distBetweenPoints (p1 p2 : Point3D) : ℝ :=
  sorry

theorem min_distance_to_plane (α β γ : Plane) (A P : Point3D) :
  -- Planes are mutually perpendicular
  (α.normal.x * β.normal.x + α.normal.y * β.normal.y + α.normal.z * β.normal.z = 0) →
  (β.normal.x * γ.normal.x + β.normal.y * γ.normal.y + β.normal.z * γ.normal.z = 0) →
  (γ.normal.x * α.normal.x + γ.normal.y * α.normal.y + γ.normal.z * α.normal.z = 0) →
  -- A is on plane α
  (distPointToPlane A α = 0) →
  -- Distance from A to plane β is 3
  (distPointToPlane A β = 3) →
  -- Distance from A to plane γ is 3
  (distPointToPlane A γ = 3) →
  -- P is on plane α
  (distPointToPlane P α = 0) →
  -- Distance from P to plane β is twice the distance from P to point A
  (distPointToPlane P β = 2 * distBetweenPoints P A) →
  -- The minimum distance from points on the trajectory of P to plane γ is 3 - √3
  (∃ (P' : Point3D), distPointToPlane P' α = 0 ∧
    distPointToPlane P' β = 2 * distBetweenPoints P' A ∧
    distPointToPlane P' γ = 3 - Real.sqrt 3 ∧
    ∀ (P'' : Point3D), distPointToPlane P'' α = 0 →
      distPointToPlane P'' β = 2 * distBetweenPoints P'' A →
      distPointToPlane P'' γ ≥ 3 - Real.sqrt 3) :=
by
  sorry

end min_distance_to_plane_l2777_277762


namespace min_quotient_is_20_5_l2777_277749

/-- Represents a three-digit number with digits a, b, and c -/
structure ThreeDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a > 0
  b_nonzero : b > 0
  c_nonzero : c > 0
  all_different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  b_relation : b = a + 1
  c_relation : c = b + 1

/-- The quotient of the number divided by the sum of its digits -/
def quotient (n : ThreeDigitNumber) : ℚ :=
  (100 * n.a + 10 * n.b + n.c) / (n.a + n.b + n.c)

/-- The theorem stating that the minimum quotient is 20.5 -/
theorem min_quotient_is_20_5 :
  ∀ n : ThreeDigitNumber, quotient n ≥ 20.5 ∧ ∃ n : ThreeDigitNumber, quotient n = 20.5 :=
sorry

end min_quotient_is_20_5_l2777_277749


namespace probability_six_odd_in_eight_rolls_l2777_277780

theorem probability_six_odd_in_eight_rolls (n : ℕ) (p : ℚ) : 
  n = 8 →                   -- number of rolls
  p = 1/2 →                 -- probability of rolling an odd number
  (n.choose 6 : ℚ) * p^6 * (1 - p)^(n - 6) = 7/64 := by
sorry

end probability_six_odd_in_eight_rolls_l2777_277780


namespace expression_simplification_l2777_277735

theorem expression_simplification (x : ℝ) (h : x + 2 = Real.sqrt 2) :
  ((x^2 + 1) / x + 2) / ((x - 3) * (x + 1) / (x^2 - 3*x)) = Real.sqrt 2 - 1 := by
  sorry

end expression_simplification_l2777_277735


namespace smallest_b_value_l2777_277725

theorem smallest_b_value (a b : ℕ+) (h1 : a - b = 8) 
  (h2 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  ∀ c : ℕ+, c < b → ¬(∃ d : ℕ+, d - c = 8 ∧ 
    Nat.gcd ((d^3 + c^3) / (d + c)) (d * c) = 16) :=
by sorry

end smallest_b_value_l2777_277725


namespace new_students_average_age_l2777_277708

/-- Given a class where:
    - The original number of students is 8
    - The original average age is 40 years
    - 8 new students join
    - The new average age of the entire class is 36 years
    This theorem proves that the average age of the new students is 32 years. -/
theorem new_students_average_age
  (original_count : Nat)
  (original_avg : ℝ)
  (new_count : Nat)
  (new_total_avg : ℝ)
  (h1 : original_count = 8)
  (h2 : original_avg = 40)
  (h3 : new_count = 8)
  (h4 : new_total_avg = 36) :
  (((original_count + new_count) * new_total_avg) - (original_count * original_avg)) / new_count = 32 := by
  sorry


end new_students_average_age_l2777_277708


namespace line_not_in_fourth_quadrant_l2777_277739

/-- Represents a line in a 2D Cartesian coordinate system -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Determines if a point (x, y) is in the fourth quadrant -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Determines if a line passes through the fourth quadrant -/
def passes_through_fourth_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, y = l.slope * x + l.y_intercept ∧ is_in_fourth_quadrant x y

/-- The main theorem: the line y = 2x + 1 does not pass through the fourth quadrant -/
theorem line_not_in_fourth_quadrant :
  ¬ passes_through_fourth_quadrant (Line.mk 2 1) := by
  sorry

end line_not_in_fourth_quadrant_l2777_277739


namespace parabola_y_intercept_l2777_277742

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (b c : ℝ) : 
  ((-1 : ℝ)^2 + b*(-1) + c = -11) → 
  ((3 : ℝ)^2 + b*3 + c = 17) → 
  c = -7 := by
sorry

end parabola_y_intercept_l2777_277742


namespace infinite_geometric_series_ratio_l2777_277797

/-- Proves that for an infinite geometric series with first term 400 and sum 2500, the common ratio is 21/25 -/
theorem infinite_geometric_series_ratio : ∃ (r : ℝ), 
  let a : ℝ := 400
  let S : ℝ := 2500
  r > 0 ∧ r < 1 ∧ S = a / (1 - r) ∧ r = 21 / 25 := by
  sorry

end infinite_geometric_series_ratio_l2777_277797


namespace simplify_expression_l2777_277712

theorem simplify_expression : 
  (81 ^ (1/4) - Real.sqrt (17/2)) ^ 2 = 17.5 - 3 * Real.sqrt 34 := by
  sorry

end simplify_expression_l2777_277712


namespace intersection_empty_implies_k_leq_neg_one_l2777_277777

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

-- Theorem statement
theorem intersection_empty_implies_k_leq_neg_one (k : ℝ) : 
  M ∩ N k = ∅ → k ≤ -1 := by
  sorry

end intersection_empty_implies_k_leq_neg_one_l2777_277777


namespace six_playing_cards_distribution_l2777_277718

/-- Given a deck of cards with playing cards and instruction cards,
    distributed as evenly as possible among a group of people,
    calculate the number of people who end up with exactly 6 playing cards. -/
def people_with_six_playing_cards (total_cards : ℕ) (playing_cards : ℕ) (instruction_cards : ℕ) (num_people : ℕ) : ℕ :=
  let cards_per_person := total_cards / num_people
  let extra_cards := total_cards % num_people
  let playing_cards_distribution := playing_cards / num_people
  let extra_playing_cards := playing_cards % num_people
  min extra_playing_cards (num_people - instruction_cards)

theorem six_playing_cards_distribution :
  people_with_six_playing_cards 60 52 8 9 = 7 := by
  sorry

end six_playing_cards_distribution_l2777_277718


namespace z_equals_2_minus_12i_z_is_pure_imaginary_l2777_277798

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + 5*m + 6) (m^2 - 2*m - 15)

/-- Theorem for the first condition -/
theorem z_equals_2_minus_12i (m : ℝ) : z m = Complex.mk 2 (-12) ↔ m = -1 := by sorry

/-- Theorem for the second condition -/
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = -2 := by sorry

end z_equals_2_minus_12i_z_is_pure_imaginary_l2777_277798


namespace turtle_difference_l2777_277789

/-- Given the following conditions about turtle ownership:
  1. Trey has 9 times as many turtles as Kris
  2. Kris has 1/3 as many turtles as Kristen
  3. Layla has twice as many turtles as Trey
  4. Tim has half as many turtles as Kristen
  5. Kristen has 18 turtles

  Prove that Trey has 45 more turtles than Tim. -/
theorem turtle_difference (kristen tim trey kris layla : ℕ) : 
  kristen = 18 →
  kris = kristen / 3 →
  trey = 9 * kris →
  layla = 2 * trey →
  tim = kristen / 2 →
  trey - tim = 45 := by
sorry

end turtle_difference_l2777_277789


namespace turtle_ratio_l2777_277747

/-- Prove that given the conditions, the ratio of turtles Kris has to Kristen has is 1:4 -/
theorem turtle_ratio : 
  ∀ (kris trey kristen : ℕ),
  trey = 5 * kris →
  kris + trey + kristen = 30 →
  kristen = 12 →
  kris.gcd kristen = 3 →
  (kris / 3 : ℚ) / (kristen / 3 : ℚ) = 1 / 4 := by
sorry


end turtle_ratio_l2777_277747


namespace symmetry_point_l2777_277771

/-- Given two points A and B in a 2D plane, they are symmetric with respect to the origin
    if the sum of their coordinates is (0, 0) -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 + B.1 = 0 ∧ A.2 + B.2 = 0

theorem symmetry_point :
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (2, -3)
  symmetric_wrt_origin A B → B = (2, -3) := by
sorry

end symmetry_point_l2777_277771


namespace pen_price_l2777_277764

theorem pen_price (total_pens : ℕ) (total_cost : ℚ) (regular_price : ℚ) : 
  total_pens = 20 ∧ total_cost = 30 ∧ 
  (regular_price * (total_pens / 2) + (regular_price / 2) * (total_pens / 2) = total_cost) →
  regular_price = 2 := by
sorry

end pen_price_l2777_277764


namespace male_students_count_l2777_277732

def scienceGroup (x : ℕ) : Prop :=
  ∃ (total : ℕ), total = x + 2 ∧ 
  (Nat.choose x 2) * (Nat.choose 2 1) = 20

theorem male_students_count :
  ∀ x : ℕ, scienceGroup x → x = 5 := by sorry

end male_students_count_l2777_277732


namespace geometric_sequence_ratio_two_l2777_277738

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_ratio_two (a : ℕ → ℝ) 
  (h : geometric_sequence a) 
  (h_ratio : ∀ n : ℕ, a (n + 1) = 2 * a n) :
  (2 * a 2 + a 3) / (2 * a 4 + a 5) = 1 / 6 := by
  sorry

end geometric_sequence_ratio_two_l2777_277738


namespace fourth_term_of_solution_sequence_l2777_277750

def is_solution (x : ℤ) : Prop := x^2 - 2*x - 3 < 0

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fourth_term_of_solution_sequence :
  ∃ a : ℕ → ℤ,
    (∀ n : ℕ, is_solution (a n)) ∧
    arithmetic_sequence a ∧
    (a 4 = 3 ∨ a 4 = -1) := by sorry

end fourth_term_of_solution_sequence_l2777_277750


namespace meenas_bottle_caps_l2777_277741

theorem meenas_bottle_caps (initial : ℕ) : 
  (initial : ℚ) * (1 + 0.4) * (1 - 0.2) = initial + 21 → initial = 175 :=
by
  sorry

end meenas_bottle_caps_l2777_277741


namespace book_length_l2777_277774

theorem book_length (area : ℝ) (width : ℝ) (h1 : area = 50) (h2 : width = 10) :
  area / width = 5 := by
  sorry

end book_length_l2777_277774


namespace afternoon_sales_problem_l2777_277792

/-- Calculates the number of cookies sold in the afternoon given the initial count,
    morning sales, lunch sales, and remaining cookies. -/
def afternoon_sales (initial : ℕ) (morning_dozens : ℕ) (lunch : ℕ) (remaining : ℕ) : ℕ :=
  initial - (morning_dozens * 12 + lunch) - remaining

theorem afternoon_sales_problem :
  afternoon_sales 120 3 57 11 = 16 := by
  sorry

end afternoon_sales_problem_l2777_277792


namespace son_work_time_l2777_277731

/-- Given a man can do a piece of work in 5 days, and together with his son they can do it in 3 days,
    prove that the son can do the work alone in 7.5 days. -/
theorem son_work_time (man_time : ℝ) (combined_time : ℝ) (son_time : ℝ) 
    (h1 : man_time = 5)
    (h2 : combined_time = 3) :
    son_time = 7.5 := by
  sorry

end son_work_time_l2777_277731


namespace trapezoid_bc_length_l2777_277786

-- Define the trapezoid and its properties
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  ab_length : ℝ
  cd_length : ℝ
  ad_cd_angle : ℝ

-- Define the theorem
theorem trapezoid_bc_length (t : Trapezoid) 
  (h1 : t.area = 200)
  (h2 : t.altitude = 10)
  (h3 : t.ab_length = 15)
  (h4 : t.cd_length = 25)
  (h5 : t.ad_cd_angle = π/4)
  : ∃ (bc_length : ℝ), bc_length = (200 - (25 * Real.sqrt 5 + 25 * Real.sqrt 21)) / 10 := by
  sorry


end trapezoid_bc_length_l2777_277786


namespace exterior_angle_measure_l2777_277740

/-- Given a regular polygon with sum of interior angles 1260°, 
    prove that each exterior angle measures 40° -/
theorem exterior_angle_measure (n : ℕ) : 
  (n - 2) * 180 = 1260 → 360 / n = 40 := by
  sorry

end exterior_angle_measure_l2777_277740


namespace odd_increasing_function_property_l2777_277701

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f is monotonically increasing if x < y implies f(x) < f(y) -/
def IsMonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem odd_increasing_function_property (f : ℝ → ℝ) 
    (h_odd : IsOdd f) (h_mono : IsMonoIncreasing f) :
    (∀ a b : ℝ, f a + f (b - 1) = 0) → 
    (∀ a b : ℝ, a + b = 1) :=
  sorry

end odd_increasing_function_property_l2777_277701


namespace cubic_root_sum_l2777_277787

theorem cubic_root_sum (a b : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + a * (Complex.I * Real.sqrt 2 + 2) + b = 0 → 
  a + b = 14 := by
sorry

end cubic_root_sum_l2777_277787


namespace eliminate_quadratic_term_l2777_277744

/-- The polynomial we're working with -/
def polynomial (x n : ℝ) : ℝ := 4*x^2 + 2*(7 + 3*x - 3*x^2) - n*x^2

/-- The coefficient of x^2 in the expanded polynomial -/
def quadratic_coefficient (n : ℝ) : ℝ := 4 - 6 - n

theorem eliminate_quadratic_term :
  ∃ (n : ℝ), ∀ (x : ℝ), polynomial x n = 6*x + 14 ∧ n = -2 :=
sorry

end eliminate_quadratic_term_l2777_277744


namespace student_ratio_proof_l2777_277794

/-- Proves that the ratio of elementary school students to other students is 8/9 -/
theorem student_ratio_proof 
  (m n : ℕ) -- number of elementary and other students
  (a b : ℝ) -- average heights of elementary and other students
  (α β : ℝ) -- given constants
  (h1 : a = α * b) -- condition 1
  (h2 : α = 3/4) -- given value of α
  (h3 : a = β * ((a * m + b * n) / (m + n))) -- condition 2
  (h4 : β = 19/20) -- given value of β
  : m / n = 8/9 := by
  sorry


end student_ratio_proof_l2777_277794


namespace square_reciprocal_sum_implies_fourth_power_reciprocal_sum_l2777_277751

theorem square_reciprocal_sum_implies_fourth_power_reciprocal_sum
  (x : ℝ) (h : x^2 + 1/x^2 = 6) : x^4 + 1/x^4 = 34 := by
  sorry

end square_reciprocal_sum_implies_fourth_power_reciprocal_sum_l2777_277751


namespace total_stripes_is_22_l2777_277760

/-- The number of stripes on each of Olga's tennis shoes -/
def olga_stripes : ℕ := 3

/-- The number of stripes on each of Rick's tennis shoes -/
def rick_stripes : ℕ := olga_stripes - 1

/-- The number of stripes on each of Hortense's tennis shoes -/
def hortense_stripes : ℕ := olga_stripes * 2

/-- The number of shoes each person has -/
def shoes_per_person : ℕ := 2

/-- The total number of stripes on all their shoes combined -/
def total_stripes : ℕ := shoes_per_person * (olga_stripes + rick_stripes + hortense_stripes)

theorem total_stripes_is_22 : total_stripes = 22 := by
  sorry

end total_stripes_is_22_l2777_277760


namespace nina_travel_distance_l2777_277703

theorem nina_travel_distance (x : ℕ) : 
  (12 * x + 12 * (2 * x) = 14400) → x = 400 := by
  sorry

end nina_travel_distance_l2777_277703


namespace gcd_problem_l2777_277755

theorem gcd_problem (n : ℕ) : 
  80 ≤ n ∧ n ≤ 100 → Nat.gcd 36 n = 12 → n = 84 ∨ n = 96 := by
  sorry

end gcd_problem_l2777_277755


namespace mary_overtime_rate_increase_l2777_277778

/-- Represents Mary's work schedule and pay structure -/
structure MaryWorkSchedule where
  maxHours : ℕ
  regularHours : ℕ
  regularRate : ℚ
  maxEarnings : ℚ

/-- Calculates the percentage increase in overtime rate compared to regular rate -/
def overtimeRateIncrease (schedule : MaryWorkSchedule) : ℚ :=
  let regularEarnings := schedule.regularHours * schedule.regularRate
  let overtimeEarnings := schedule.maxEarnings - regularEarnings
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let overtimeRate := overtimeEarnings / overtimeHours
  ((overtimeRate - schedule.regularRate) / schedule.regularRate) * 100

/-- Theorem stating that Mary's overtime rate increase is 25% -/
theorem mary_overtime_rate_increase :
  let schedule := MaryWorkSchedule.mk 70 20 8 660
  overtimeRateIncrease schedule = 25 := by
  sorry

end mary_overtime_rate_increase_l2777_277778


namespace arithmetic_sequence_problem_l2777_277722

def is_arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

def nth_term (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_problem (x y : ℝ) (m : ℕ) 
  (h1 : is_arithmetic_sequence (Real.log (x^2 * y^5)) (Real.log (x^4 * y^9)) (Real.log (x^7 * y^12)))
  (h2 : nth_term (Real.log (x^2 * y^5)) 
               ((Real.log (x^4 * y^9)) - (Real.log (x^2 * y^5))) 
               10 = Real.log (y^m)) :
  m = 55 := by
  sorry

end arithmetic_sequence_problem_l2777_277722


namespace passing_train_speed_is_50_l2777_277702

/-- The speed of the passing train in km/h -/
def passing_train_speed : ℝ := 50

/-- The speed of the passenger's train in km/h -/
def passenger_train_speed : ℝ := 40

/-- The time taken for the passing train to pass completely in seconds -/
def passing_time : ℝ := 3

/-- The length of the passing train in meters -/
def passing_train_length : ℝ := 75

/-- Theorem stating that the speed of the passing train is 50 km/h -/
theorem passing_train_speed_is_50 :
  passing_train_speed = 50 :=
sorry

end passing_train_speed_is_50_l2777_277702


namespace equality_of_ratios_implies_k_eighteen_l2777_277709

theorem equality_of_ratios_implies_k_eighteen 
  (x y z k : ℝ) 
  (h : (7 : ℝ) / (x + y) = k / (x + z) ∧ k / (x + z) = (11 : ℝ) / (z - y)) : 
  k = 18 := by
sorry

end equality_of_ratios_implies_k_eighteen_l2777_277709


namespace sum_remainder_seven_l2777_277724

theorem sum_remainder_seven (n : ℤ) : (7 - n + (n + 3)) % 7 = 3 := by sorry

end sum_remainder_seven_l2777_277724


namespace sequence_2021st_term_l2777_277756

/-- The sequence function that gives the n-th term of the sequence -/
def sequenceFunction (n : ℕ) : ℕ := sorry

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The property that the n-th positive integer appears n times in the sequence -/
axiom sequence_property (n : ℕ) : 
  ∀ k, triangularNumber (n - 1) < k ∧ k ≤ triangularNumber n → sequenceFunction k = n

/-- The theorem stating that the 2021st term of the sequence is 64 -/
theorem sequence_2021st_term : sequenceFunction 2021 = 64 := by sorry

end sequence_2021st_term_l2777_277756


namespace mango_tree_columns_count_l2777_277790

/-- The number of columns of mango trees in a garden with given dimensions -/
def mango_tree_columns (garden_length : ℕ) (tree_distance : ℕ) (boundary_distance : ℕ) : ℕ :=
  let available_length := garden_length - 2 * boundary_distance
  let spaces := available_length / tree_distance
  spaces + 1

/-- Theorem stating that the number of mango tree columns is 12 given the specified conditions -/
theorem mango_tree_columns_count :
  mango_tree_columns 32 2 5 = 12 := by
  sorry

end mango_tree_columns_count_l2777_277790


namespace prime_arithmetic_progression_l2777_277799

theorem prime_arithmetic_progression (p₁ p₂ p₃ : ℕ) (d : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ 
  p₁ > 3 ∧ p₂ > 3 ∧ p₃ > 3 ∧
  p₂ = p₁ + d ∧ p₃ = p₂ + d → 
  d % 6 = 0 := by
  sorry

#check prime_arithmetic_progression

end prime_arithmetic_progression_l2777_277799


namespace particular_number_exists_l2777_277768

theorem particular_number_exists : ∃! x : ℝ, 2 * ((x / 23) - 67) = 102 := by
  sorry

end particular_number_exists_l2777_277768


namespace complement_intersection_theorem_l2777_277761

open Set

def U : Set Nat := {0, 1, 2, 3, 4}
def M : Set Nat := {0, 1, 2}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end complement_intersection_theorem_l2777_277761


namespace soccer_ball_hexagons_l2777_277776

/-- Represents a soccer ball with black pentagons and white hexagons -/
structure SoccerBall where
  black_pentagons : ℕ
  white_hexagons : ℕ
  pentagon_sides : ℕ
  hexagon_sides : ℕ
  pentagon_hexagon_connections : ℕ
  hexagon_pentagon_connections : ℕ
  hexagon_hexagon_connections : ℕ

/-- Theorem stating the number of white hexagons on a soccer ball with specific conditions -/
theorem soccer_ball_hexagons (ball : SoccerBall) :
  ball.black_pentagons = 12 ∧
  ball.pentagon_sides = 5 ∧
  ball.hexagon_sides = 6 ∧
  ball.pentagon_hexagon_connections = 5 ∧
  ball.hexagon_pentagon_connections = 3 ∧
  ball.hexagon_hexagon_connections = 3 →
  ball.white_hexagons = 20 := by
  sorry

end soccer_ball_hexagons_l2777_277776


namespace four_noncoplanar_points_determine_four_planes_l2777_277705

-- Define a Point type
def Point : Type := ℝ × ℝ × ℝ

-- Define a Plane type
structure Plane where
  normal : ℝ × ℝ × ℝ
  d : ℝ

-- Define a function to check if points are coplanar
def are_coplanar (p1 p2 p3 p4 : Point) : Prop := sorry

-- Define a function to create a plane from three points
def plane_from_points (p1 p2 p3 : Point) : Plane := sorry

-- Define a function to count the number of unique planes
def count_unique_planes (planes : List Plane) : Nat := sorry

-- Theorem statement
theorem four_noncoplanar_points_determine_four_planes 
  (p1 p2 p3 p4 : Point) 
  (h : ¬ are_coplanar p1 p2 p3 p4) : 
  count_unique_planes [
    plane_from_points p1 p2 p3,
    plane_from_points p1 p2 p4,
    plane_from_points p1 p3 p4,
    plane_from_points p2 p3 p4
  ] = 4 := by sorry

end four_noncoplanar_points_determine_four_planes_l2777_277705


namespace inscribed_circle_theorem_l2777_277767

theorem inscribed_circle_theorem (r : ℝ) (a b c : ℝ) :
  r > 0 ∧ a > 0 ∧ b > 0 ∧ c > 0 →
  r = 4 →
  a + b = 14 →
  (∃ (s : ℝ), s = (a + b + c) / 2 ∧ s * r = Real.sqrt (s * (s - a) * (s - b) * (s - c))) →
  (c = 13 ∧ b = 15) ∨ (c = 15 ∧ b = 13) :=
by sorry

end inscribed_circle_theorem_l2777_277767


namespace vessel_base_length_l2777_277785

/-- Given a cube immersed in a rectangular vessel, calculate the length of the vessel's base. -/
theorem vessel_base_length (cube_edge : ℝ) (vessel_width : ℝ) (water_rise : ℝ) : 
  cube_edge = 15 →
  vessel_width = 14 →
  water_rise = 12.053571428571429 →
  (cube_edge ^ 3) / (vessel_width * water_rise) = 20 := by
  sorry

end vessel_base_length_l2777_277785


namespace odd_sum_floor_condition_l2777_277753

theorem odd_sum_floor_condition (p a b : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) 
  (ha : 0 < a ∧ a < p) (hb : 0 < b ∧ b < p) :
  (a + b = p) ↔ 
  (∀ n : ℕ, 0 < n → n < p → 
    ∃ k : ℕ, k % 2 = 1 ∧ 
      (⌊(2 * a * n : ℚ) / p⌋ + ⌊(2 * b * n : ℚ) / p⌋ : ℤ) = k) :=
by sorry

end odd_sum_floor_condition_l2777_277753


namespace max_value_theorem_l2777_277769

theorem max_value_theorem (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  ∀ x y z, 0 ≤ x ∧ x ≤ 2 → 0 ≤ y ∧ y ≤ 2 → 0 ≤ z ∧ z ≤ 2 →
    Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤
    Real.sqrt (x^2 * y^2 * z^2) + Real.sqrt ((2 - x) * (2 - y) * (2 - z)) →
    Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((2 - a) * (2 - b) * (2 - c)) ≤ 8 :=
by
  sorry

end max_value_theorem_l2777_277769


namespace log_equation_relationships_l2777_277796

/-- Given real numbers a and b satisfying log_(1/2)(a) = log_(1/3)(b), 
    exactly 2 out of 5 given relationships cannot hold true. -/
theorem log_equation_relationships (a b : ℝ) 
  (h : Real.log a / Real.log (1/2) = Real.log b / Real.log (1/3)) : 
  ∃! (s : Finset (Fin 5)), s.card = 2 ∧ 
  (∀ i ∈ s, match i with
    | 0 => ¬(a > b ∧ b > 1)
    | 1 => ¬(0 < b ∧ b < a ∧ a < 1)
    | 2 => ¬(b > a ∧ a > 1)
    | 3 => ¬(0 < a ∧ a < b ∧ b < 1)
    | 4 => ¬(a = b)
  ) ∧
  (∀ i ∉ s, match i with
    | 0 => (a > b ∧ b > 1)
    | 1 => (0 < b ∧ b < a ∧ a < 1)
    | 2 => (b > a ∧ a > 1)
    | 3 => (0 < a ∧ a < b ∧ b < 1)
    | 4 => (a = b)
  ) := by
  sorry

end log_equation_relationships_l2777_277796


namespace loan_payment_difference_l2777_277793

/-- Calculates the monthly payment for a loan -/
def monthly_payment (loan_amount : ℚ) (months : ℕ) : ℚ :=
  loan_amount / months

/-- Represents the loan details -/
structure LoanDetails where
  amount : ℚ
  short_term_months : ℕ
  long_term_months : ℕ

theorem loan_payment_difference (loan : LoanDetails) 
  (h1 : loan.amount = 6000)
  (h2 : loan.short_term_months = 24)
  (h3 : loan.long_term_months = 60) :
  monthly_payment loan.amount loan.short_term_months - 
  monthly_payment loan.amount loan.long_term_months = 150 := by
  sorry


end loan_payment_difference_l2777_277793


namespace cost_reduction_percentage_l2777_277788

/-- Proves the percentage reduction in cost price given specific conditions --/
theorem cost_reduction_percentage
  (original_cost : ℝ)
  (original_profit_rate : ℝ)
  (price_reduction : ℝ)
  (new_profit_rate : ℝ)
  (h1 : original_cost = 40)
  (h2 : original_profit_rate = 0.25)
  (h3 : price_reduction = 8.40)
  (h4 : new_profit_rate = 0.30)
  : ∃ (reduction_rate : ℝ),
    reduction_rate = 0.20 ∧
    (1 + new_profit_rate) * (original_cost * (1 - reduction_rate)) =
    (1 + original_profit_rate) * original_cost - price_reduction :=
by sorry

end cost_reduction_percentage_l2777_277788


namespace dairy_water_mixture_l2777_277775

theorem dairy_water_mixture (original_price selling_price : ℝ) 
  (h1 : selling_price = original_price * 1.25) : 
  (selling_price - original_price) / selling_price = 0.2 := by
  sorry

end dairy_water_mixture_l2777_277775


namespace equation_solutions_l2777_277730

theorem equation_solutions :
  (∀ x : ℝ, 16 * x^2 = 49 ↔ x = 7/4 ∨ x = -7/4) ∧
  (∀ x : ℝ, (x - 2)^2 = 64 ↔ x = 10 ∨ x = -6) :=
by sorry

end equation_solutions_l2777_277730


namespace function_identity_l2777_277729

theorem function_identity (f : ℚ → ℚ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) : 
  ∀ x : ℚ, f x = x + 1 := by
sorry

end function_identity_l2777_277729
