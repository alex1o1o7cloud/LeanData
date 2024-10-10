import Mathlib

namespace ada_original_seat_l3427_342722

-- Define the seats
inductive Seat : Type
| one : Seat
| two : Seat
| three : Seat
| four : Seat
| five : Seat
| six : Seat

-- Define the friends
inductive Friend : Type
| ada : Friend
| bea : Friend
| ceci : Friend
| dee : Friend
| edie : Friend
| fred : Friend

-- Define the seating arrangement as a function from Friend to Seat
def Seating := Friend → Seat

-- Define the movement function
def move (s : Seating) : Seating :=
  fun f => match f with
    | Friend.bea => match s Friend.bea with
      | Seat.one => Seat.two
      | Seat.two => Seat.three
      | Seat.three => Seat.four
      | Seat.four => Seat.five
      | Seat.five => Seat.six
      | Seat.six => Seat.six
    | Friend.ceci => match s Friend.ceci with
      | Seat.one => Seat.one
      | Seat.two => Seat.one
      | Seat.three => Seat.one
      | Seat.four => Seat.two
      | Seat.five => Seat.three
      | Seat.six => Seat.four
    | Friend.dee => s Friend.edie
    | Friend.edie => s Friend.dee
    | Friend.fred => s Friend.fred
    | Friend.ada => s Friend.ada

-- Theorem stating Ada's original seat
theorem ada_original_seat (initial : Seating) :
  (move initial) Friend.ada = Seat.one →
  initial Friend.ada = Seat.two :=
by
  sorry


end ada_original_seat_l3427_342722


namespace line_slope_thirty_degrees_l3427_342724

theorem line_slope_thirty_degrees (m : ℝ) : 
  (∃ (x y : ℝ), x + m * y - 3 = 0) →
  (Real.tan (30 * π / 180) = -1 / m) →
  m = -Real.sqrt 3 := by
sorry

end line_slope_thirty_degrees_l3427_342724


namespace color_drawing_cost_is_240_l3427_342713

/-- The cost of a color drawing given the cost of a black and white drawing and the additional percentage for color. -/
def color_drawing_cost (bw_cost : ℝ) (color_percentage : ℝ) : ℝ :=
  bw_cost * (1 + color_percentage)

/-- Theorem stating that the cost of a color drawing is $240 given the specified conditions. -/
theorem color_drawing_cost_is_240 :
  color_drawing_cost 160 0.5 = 240 := by
  sorry

end color_drawing_cost_is_240_l3427_342713


namespace max_dinner_income_is_136_80_l3427_342792

/-- Represents the chef's restaurant scenario -/
structure RestaurantScenario where
  -- Lunch meals
  pasta_lunch : ℕ
  chicken_lunch : ℕ
  fish_lunch : ℕ
  -- Prices
  pasta_price : ℚ
  chicken_price : ℚ
  fish_price : ℚ
  -- Sold during lunch
  pasta_sold_lunch : ℕ
  chicken_sold_lunch : ℕ
  fish_sold_lunch : ℕ
  -- Dinner meals
  pasta_dinner : ℕ
  chicken_dinner : ℕ
  fish_dinner : ℕ
  -- Discount rate
  discount_rate : ℚ

/-- Calculates the maximum total income during dinner -/
def max_dinner_income (s : RestaurantScenario) : ℚ :=
  let pasta_unsold := s.pasta_lunch - s.pasta_sold_lunch
  let chicken_unsold := s.chicken_lunch - s.chicken_sold_lunch
  let fish_unsold := s.fish_lunch - s.fish_sold_lunch
  let discounted_pasta_price := s.pasta_price * (1 - s.discount_rate)
  let discounted_chicken_price := s.chicken_price * (1 - s.discount_rate)
  let discounted_fish_price := s.fish_price * (1 - s.discount_rate)
  (s.pasta_dinner * s.pasta_price + pasta_unsold * discounted_pasta_price) +
  (s.chicken_dinner * s.chicken_price + chicken_unsold * discounted_chicken_price) +
  (s.fish_dinner * s.fish_price + fish_unsold * discounted_fish_price)

/-- The chef's restaurant scenario -/
def chef_scenario : RestaurantScenario := {
  pasta_lunch := 8
  chicken_lunch := 5
  fish_lunch := 4
  pasta_price := 12
  chicken_price := 15
  fish_price := 18
  pasta_sold_lunch := 6
  chicken_sold_lunch := 3
  fish_sold_lunch := 3
  pasta_dinner := 2
  chicken_dinner := 2
  fish_dinner := 1
  discount_rate := 1/10
}

/-- Theorem stating the maximum total income during dinner -/
theorem max_dinner_income_is_136_80 :
  max_dinner_income chef_scenario = 136.8 := by sorry


end max_dinner_income_is_136_80_l3427_342792


namespace proportion_solution_l3427_342725

theorem proportion_solution (x : ℝ) :
  (0.75 : ℝ) / x = (4.5 : ℝ) / (7/3 : ℝ) →
  x = (0.75 : ℝ) * (7/3 : ℝ) / (4.5 : ℝ) :=
by sorry

end proportion_solution_l3427_342725


namespace brownie_pieces_l3427_342765

/-- Proves that a 24-inch by 30-inch pan can be divided into exactly 60 pieces of 3-inch by 4-inch brownies. -/
theorem brownie_pieces (pan_length : ℕ) (pan_width : ℕ) (piece_length : ℕ) (piece_width : ℕ) :
  pan_length = 24 →
  pan_width = 30 →
  piece_length = 3 →
  piece_width = 4 →
  (pan_length * pan_width) / (piece_length * piece_width) = 60 :=
by sorry

end brownie_pieces_l3427_342765


namespace platform_length_platform_length_proof_l3427_342760

/-- Calculates the length of a platform given train parameters --/
theorem platform_length (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let total_distance := train_speed_mps * crossing_time
  total_distance - train_length

/-- Proves that the platform length is 340 m given the specific parameters --/
theorem platform_length_proof :
  platform_length 160 72 25 = 340 := by
  sorry

end platform_length_platform_length_proof_l3427_342760


namespace grade_difference_l3427_342723

theorem grade_difference (x y : ℕ) (h : 3 * y = 4 * x) :
  y - x = 3 ∧ y - x = 4 :=
sorry

end grade_difference_l3427_342723


namespace chessboard_not_fully_covered_l3427_342728

/-- Represents the dimensions of a square chessboard -/
def BoardSize : ℕ := 10

/-- Represents the number of squares covered by one L-shaped tromino piece -/
def SquaresPerPiece : ℕ := 3

/-- Represents the number of L-shaped tromino pieces available -/
def NumberOfPieces : ℕ := 25

/-- Theorem stating that the chessboard cannot be fully covered by the given pieces -/
theorem chessboard_not_fully_covered :
  NumberOfPieces * SquaresPerPiece < BoardSize * BoardSize := by
  sorry

end chessboard_not_fully_covered_l3427_342728


namespace problem_statement_l3427_342702

-- Define the function f
noncomputable def f (a m x : ℝ) : ℝ := m * x^a + (Real.log (1 + x))^a - a * Real.log (1 - x) - 2

-- State the theorem
theorem problem_statement (a : ℝ) (h1 : a^(1/2) ≤ 3) (h2 : Real.log 3 / Real.log a ≤ 1/2) :
  ((0 < a ∧ a < 1) ∨ a = 9) ∧
  (a > 1 → ∃ m : ℝ, f a m (1/2) = a → f a m (-1/2) = -13) :=
sorry

end problem_statement_l3427_342702


namespace coefficient_of_x_in_second_equation_l3427_342751

theorem coefficient_of_x_in_second_equation 
  (x y z : ℝ) 
  (eq1 : 6*x - 5*y + 3*z = 22)
  (eq2 : x + 8*y - 11*z = 7/4)
  (eq3 : 5*x - 6*y + 2*z = 12)
  (sum_xyz : x + y + z = 10) :
  1 = 1 := by sorry

end coefficient_of_x_in_second_equation_l3427_342751


namespace sector_perimeter_l3427_342763

/-- Given a circular sector with central angle 4 radians and area 2 cm², 
    its perimeter is 6 cm -/
theorem sector_perimeter (θ : Real) (A : Real) (r : Real) : 
  θ = 4 → A = 2 → (1/2) * θ * r^2 = A → r + r + θ * r = 6 := by
  sorry

end sector_perimeter_l3427_342763


namespace triangle_3_5_7_l3427_342787

/-- A set of three line segments can form a triangle if and only if the sum of the lengths of any two sides is greater than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

/-- Prove that the set of line segments (3cm, 5cm, 7cm) can form a triangle. -/
theorem triangle_3_5_7 : can_form_triangle 3 5 7 := by
  sorry

end triangle_3_5_7_l3427_342787


namespace marilyn_initial_caps_l3427_342789

/-- The number of bottle caps Marilyn has initially -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Nancy gives to Marilyn -/
def nancy_caps : ℕ := 36

/-- The total number of bottle caps Marilyn has after receiving Nancy's caps -/
def total_caps : ℕ := 87

/-- Theorem stating that Marilyn's initial number of bottle caps is 51 -/
theorem marilyn_initial_caps : 
  initial_caps + nancy_caps = total_caps → initial_caps = 51 := by sorry

end marilyn_initial_caps_l3427_342789


namespace town_population_l3427_342791

theorem town_population (increase_rate : ℝ) (future_population : ℕ) :
  increase_rate = 0.1 →
  future_population = 242 →
  ∃ present_population : ℕ,
    present_population * (1 + increase_rate) = future_population ∧
    present_population = 220 := by
  sorry

end town_population_l3427_342791


namespace ryans_leaf_collection_l3427_342748

/-- Given Ryan's leaf collection scenario, prove the number of remaining leaves. -/
theorem ryans_leaf_collection :
  let initial_leaves : ℕ := 89
  let first_loss : ℕ := 24
  let second_loss : ℕ := 43
  initial_leaves - first_loss - second_loss = 22 :=
by
  sorry

end ryans_leaf_collection_l3427_342748


namespace rectangle_intersection_sum_l3427_342709

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  equation : ℝ → ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if four points form a rectangle -/
def form_rectangle (p1 p2 p3 p4 : ℝ × ℝ) : Prop := sorry

/-- Predicate to check if a point lies on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

/-- Predicate to check if a point lies on a line -/
def on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

theorem rectangle_intersection_sum (m n k : ℝ) :
  let c : Circle := ⟨(-3/2, -1/2), fun x y k => x^2 + y^2 + 3*x + y + k = 0⟩
  let l₁ : Line := ⟨3, m⟩
  let l₂ : Line := ⟨3, n⟩
  (∃ p1 p2 p3 p4 : ℝ × ℝ,
    on_circle p1 c ∧ on_circle p2 c ∧ on_circle p3 c ∧ on_circle p4 c ∧
    ((on_line p1 l₁ ∧ on_line p2 l₁) ∨ (on_line p1 l₁ ∧ on_line p3 l₁) ∨
     (on_line p1 l₁ ∧ on_line p4 l₁) ∨ (on_line p2 l₁ ∧ on_line p3 l₁) ∨
     (on_line p2 l₁ ∧ on_line p4 l₁) ∨ (on_line p3 l₁ ∧ on_line p4 l₁)) ∧
    ((on_line p1 l₂ ∧ on_line p2 l₂) ∨ (on_line p1 l₂ ∧ on_line p3 l₂) ∨
     (on_line p1 l₂ ∧ on_line p4 l₂) ∨ (on_line p2 l₂ ∧ on_line p3 l₂) ∨
     (on_line p2 l₂ ∧ on_line p4 l₂) ∨ (on_line p3 l₂ ∧ on_line p4 l₂)) ∧
    form_rectangle p1 p2 p3 p4) →
  m + n = 8 := by sorry

end rectangle_intersection_sum_l3427_342709


namespace circle_max_area_center_l3427_342755

/-- Given a circle represented by the equation x^2 + y^2 + kx + 2y + k^2 = 0 in the Cartesian 
coordinate system, this theorem states that when the circle has maximum area, its center 
coordinates are (-k/2, -1). -/
theorem circle_max_area_center (k : ℝ) :
  let circle_equation := fun (x y : ℝ) => x^2 + y^2 + k*x + 2*y + k^2 = 0
  let center := (-k/2, -1)
  let is_max_area := ∀ k' : ℝ, 
    (∃ x y, circle_equation x y) → 
    (∃ x' y', x'^2 + y'^2 + k'*x' + 2*y' + k'^2 = 0 ∧ 
              (x' - (-k'/2))^2 + (y' - (-1))^2 ≤ (x - (-k/2))^2 + (y - (-1))^2)
  is_max_area → 
  ∃ x y, circle_equation x y ∧ 
         (x - center.1)^2 + (y - center.2)^2 = 
         (x - (-k/2))^2 + (y - (-1))^2 := by
  sorry

end circle_max_area_center_l3427_342755


namespace possible_values_of_a_l3427_342705

theorem possible_values_of_a (a b c d : ℕ+) 
  (h1 : a > b ∧ b > c ∧ c > d)
  (h2 : a + b + c + d = 2014)
  (h3 : a^2 - b^2 + c^2 - d^2 = 2014) :
  ∃! s : Finset ℕ+, s.card = 502 ∧ ∀ x : ℕ+, x ∈ s ↔ 
    ∃ b' c' d' : ℕ+, 
      x > b' ∧ b' > c' ∧ c' > d' ∧
      x + b' + c' + d' = 2014 ∧
      x^2 - b'^2 + c'^2 - d'^2 = 2014 :=
by
  sorry

end possible_values_of_a_l3427_342705


namespace factor_x_10_minus_1024_l3427_342781

theorem factor_x_10_minus_1024 (x : ℝ) :
  x^10 - 1024 = (x^5 + 32) * (x^(5/2) + Real.sqrt 32) * (x^(5/2) - Real.sqrt 32) := by
  sorry

end factor_x_10_minus_1024_l3427_342781


namespace tub_volume_ratio_l3427_342799

theorem tub_volume_ratio :
  ∀ (V₁ V₂ : ℝ), V₁ > 0 → V₂ > 0 →
  (3/4 : ℝ) * V₁ = (2/3 : ℝ) * V₂ →
  V₁ / V₂ = 8/9 := by
sorry

end tub_volume_ratio_l3427_342799


namespace imaginary_unit_cube_l3427_342750

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end imaginary_unit_cube_l3427_342750


namespace expression_evaluation_l3427_342730

theorem expression_evaluation : 
  2 * 3 + 4 - 5 / 6 = 37 / 3 := by
  sorry

end expression_evaluation_l3427_342730


namespace power_comparison_l3427_342771

theorem power_comparison : 1.6^0.3 > 0.9^3.1 := by
  sorry

end power_comparison_l3427_342771


namespace arithmetic_sequence_tenth_term_l3427_342740

theorem arithmetic_sequence_tenth_term
  (a₁ a₁₇ : ℚ)
  (h₁ : a₁ = 2 / 3)
  (h₂ : a₁₇ = 3 / 2)
  (h_arith : ∀ n : ℕ, n > 0 → ∃ d : ℚ, a₁₇ = a₁ + (17 - 1) * d ∧ ∀ k : ℕ, k > 0 → a₁ + (k - 1) * d = a₁ + (k - 1) * ((a₁₇ - a₁) / 16)) :
  a₁ + 9 * ((a₁₇ - a₁) / 16) = 109 / 96 :=
by sorry

end arithmetic_sequence_tenth_term_l3427_342740


namespace min_value_theorem_l3427_342720

theorem min_value_theorem (a b k m n : ℝ) : 
  a > 0 → 
  a ≠ 1 → 
  (∀ x, a^(x-1) + 1 = b → x = k) → 
  m > 0 → 
  n > 0 → 
  m + n = b - k → 
  ∀ m' n', m' > 0 → n' > 0 → m' + n' = b - k → 
    9/m + 1/n ≤ 9/m' + 1/n' :=
by sorry

end min_value_theorem_l3427_342720


namespace cube_volume_from_diagonal_l3427_342757

theorem cube_volume_from_diagonal (d : ℝ) (h : d = 8 * Real.sqrt 3) :
  let s := d / Real.sqrt 3
  s ^ 3 = 512 := by sorry

end cube_volume_from_diagonal_l3427_342757


namespace average_time_to_find_waldo_l3427_342754

theorem average_time_to_find_waldo (num_books : ℕ) (puzzles_per_book : ℕ) (total_time : ℕ) :
  num_books = 15 →
  puzzles_per_book = 30 →
  total_time = 1350 →
  (total_time : ℚ) / (num_books * puzzles_per_book : ℚ) = 3 := by
  sorry

end average_time_to_find_waldo_l3427_342754


namespace relay_race_distance_l3427_342714

/-- A runner in the relay race -/
structure Runner where
  name : String
  speed : Real
  time : Real

/-- The relay race -/
def RelayRace (runners : List Runner) (totalTime : Real) : Prop :=
  (List.sum (List.map (fun r => r.speed * r.time) runners) = 17) ∧
  (List.sum (List.map (fun r => r.time) runners) = totalTime)

theorem relay_race_distance :
  let sadie : Runner := ⟨"Sadie", 3, 2⟩
  let ariana : Runner := ⟨"Ariana", 6, 0.5⟩
  let sarah : Runner := ⟨"Sarah", 4, 2⟩
  let runners : List Runner := [sadie, ariana, sarah]
  let totalTime : Real := 4.5
  RelayRace runners totalTime := by sorry

end relay_race_distance_l3427_342714


namespace cabin_rental_security_deposit_l3427_342703

/-- Calculate the security deposit for a cabin rental --/
theorem cabin_rental_security_deposit :
  let rental_period : ℕ := 14 -- 2 weeks
  let daily_rate : ℚ := 125
  let pet_fee : ℚ := 100
  let service_fee_rate : ℚ := 1/5 -- 20%
  let security_deposit_rate : ℚ := 1/2 -- 50%

  let rental_cost := rental_period * daily_rate
  let subtotal := rental_cost + pet_fee
  let service_fee := subtotal * service_fee_rate
  let total_cost := subtotal + service_fee
  let security_deposit := total_cost * security_deposit_rate

  security_deposit = 1110
  := by sorry

end cabin_rental_security_deposit_l3427_342703


namespace max_value_of_f_l3427_342706

noncomputable def f (x : ℝ) : ℝ := min (3 * x + 1) (min (-1/3 * x + 2) (x + 4))

theorem max_value_of_f :
  ∃ (M : ℝ), M = 5/2 ∧ ∀ (x : ℝ), f x ≤ M ∧ ∃ (x₀ : ℝ), f x₀ = M :=
sorry

end max_value_of_f_l3427_342706


namespace extended_twelve_basketball_conference_games_l3427_342790

/-- Calculates the number of games in a basketball conference with specific rules --/
def conference_games (teams_per_division : ℕ) (divisions : ℕ) (intra_division_games : ℕ) : ℕ :=
  let total_teams := teams_per_division * divisions
  let games_per_team := (teams_per_division - 1) * intra_division_games + teams_per_division * (divisions - 1)
  total_teams * games_per_team / 2

/-- Theorem stating the number of games in the Extended Twelve Basketball Conference --/
theorem extended_twelve_basketball_conference_games :
  conference_games 8 2 3 = 232 := by
  sorry

end extended_twelve_basketball_conference_games_l3427_342790


namespace expected_rolls_in_year_l3427_342721

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieOutcome
  | Composite
  | Prime
  | RollAgain

/-- The probability distribution of the die outcomes -/
def dieProb : DieOutcome → ℚ
  | DieOutcome.Composite => 3/8
  | DieOutcome.Prime => 1/2
  | DieOutcome.RollAgain => 1/8

/-- The expected number of rolls on a single day -/
def expectedRollsPerDay : ℚ := 1

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- The expected number of rolls in a non-leap year -/
def expectedRollsInYear : ℚ := expectedRollsPerDay * daysInYear

theorem expected_rolls_in_year :
  expectedRollsInYear = 365 := by sorry

end expected_rolls_in_year_l3427_342721


namespace cost_exceeds_fifty_l3427_342716

/-- Calculates the total cost of items after discount and tax --/
def total_cost (pizza_price : ℝ) (juice_price : ℝ) (chips_price : ℝ) (chocolate_price : ℝ)
  (pizza_count : ℕ) (juice_count : ℕ) (chips_count : ℕ) (chocolate_count : ℕ)
  (pizza_discount : ℝ) (sales_tax : ℝ) : ℝ :=
  let pizza_cost := pizza_price * pizza_count
  let juice_cost := juice_price * juice_count
  let chips_cost := chips_price * chips_count
  let chocolate_cost := chocolate_price * chocolate_count
  let discounted_pizza_cost := pizza_cost * (1 - pizza_discount)
  let subtotal := discounted_pizza_cost + juice_cost + chips_cost + chocolate_cost
  subtotal * (1 + sales_tax)

/-- Theorem: The total cost exceeds $50 --/
theorem cost_exceeds_fifty :
  total_cost 15 4 3.5 1.25 3 4 2 5 0.1 0.05 > 50 := by
  sorry

end cost_exceeds_fifty_l3427_342716


namespace deceased_member_income_l3427_342795

theorem deceased_member_income
  (initial_members : ℕ)
  (final_members : ℕ)
  (initial_average : ℚ)
  (final_average : ℚ)
  (h1 : initial_members = 4)
  (h2 : final_members = 3)
  (h3 : initial_average = 782)
  (h4 : final_average = 650)
  : (initial_members : ℚ) * initial_average - (final_members : ℚ) * final_average = 1178 := by
  sorry

end deceased_member_income_l3427_342795


namespace at_most_two_sides_equal_to_longest_diagonal_l3427_342796

-- Define a convex polygon
def ConvexPolygon : Type := sorry

-- Define the concept of a diagonal in a polygon
def diagonal (p : ConvexPolygon) : Type := sorry

-- Define the length of a side or diagonal
def length {T : Type} (x : T) : ℝ := sorry

-- Define the longest diagonal of a polygon
def longest_diagonal (p : ConvexPolygon) : diagonal p := sorry

-- Define a function that counts the number of sides equal to the longest diagonal
def count_sides_equal_to_longest_diagonal (p : ConvexPolygon) : ℕ := sorry

-- Theorem statement
theorem at_most_two_sides_equal_to_longest_diagonal (p : ConvexPolygon) :
  count_sides_equal_to_longest_diagonal p ≤ 2 := by sorry

end at_most_two_sides_equal_to_longest_diagonal_l3427_342796


namespace repetend_of_5_17_l3427_342778

/-- The repetend of a fraction is the repeating sequence of digits in its decimal representation. -/
def is_repetend (n : ℕ) (d : ℕ) (r : ℕ) : Prop :=
  ∃ (k : ℕ), 10^6 * (10 * n - d * r) = d * (10^k - 1)

/-- The 6-digit repetend in the decimal representation of 5/17 is 294117. -/
theorem repetend_of_5_17 : is_repetend 5 17 294117 := by
  sorry

end repetend_of_5_17_l3427_342778


namespace equation_solution_l3427_342764

theorem equation_solution (x : ℝ) : 
  (8 / (Real.sqrt (x - 5) - 10) + 2 / (Real.sqrt (x - 5) - 5) + 
   9 / (Real.sqrt (x - 5) + 5) + 16 / (Real.sqrt (x - 5) + 10) = 0) ↔ 
  (x = 145 / 9 ∨ x = 1200 / 121) :=
by sorry

end equation_solution_l3427_342764


namespace oil_depth_calculation_l3427_342798

/-- Represents a right cylindrical tank -/
structure Tank where
  height : ℝ
  base_diameter : ℝ

/-- Calculates the volume of oil in the tank when lying on its side -/
def oil_volume_side (tank : Tank) (depth : ℝ) : ℝ :=
  sorry

/-- Calculates the depth of oil when the tank is standing upright -/
def oil_depth_upright (tank : Tank) (volume : ℝ) : ℝ :=
  sorry

/-- Theorem: For the given tank dimensions and side oil depth, 
    the upright oil depth is approximately 2.2 feet -/
theorem oil_depth_calculation (tank : Tank) (side_depth : ℝ) :
  tank.height = 20 →
  tank.base_diameter = 6 →
  side_depth = 4 →
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
    |oil_depth_upright tank (oil_volume_side tank side_depth) - 2.2| < ε :=
sorry

end oil_depth_calculation_l3427_342798


namespace triangle_properties_l3427_342780

/-- Properties of a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about triangle properties -/
theorem triangle_properties (t : Triangle) :
  (t.c = 2 ∧ t.C = π / 3 ∧ (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3) →
  (Real.cos (t.A + t.B) = -1 / 2 ∧ t.a = 2 ∧ t.b = 2) ∧
  (t.B > π / 2 ∧ Real.cos t.A = 3 / 5 ∧ Real.sin t.B = 12 / 13) →
  Real.sin t.C = 16 / 65 := by
  sorry


end triangle_properties_l3427_342780


namespace special_op_is_addition_l3427_342711

/-- An operation on real numbers satisfying (a * b) * c = a + b + c for all a, b, c -/
def special_op (a b : ℝ) : ℝ := sorry

/-- The property that (a * b) * c = a + b + c for all a, b, c -/
axiom special_op_property (a b c : ℝ) : special_op (special_op a b) c = a + b + c

/-- Theorem: The special operation is equivalent to addition -/
theorem special_op_is_addition (a b : ℝ) : special_op a b = a + b := by
  sorry

end special_op_is_addition_l3427_342711


namespace cube_opposite_face_l3427_342794

structure Cube where
  faces : Finset Char
  adjacent : Char → Char → Prop

def opposite (c : Cube) (f1 f2 : Char) : Prop :=
  f1 ∈ c.faces ∧ f2 ∈ c.faces ∧ ¬c.adjacent f1 f2 ∧
  ∀ f3 ∈ c.faces, f3 ≠ f1 ∧ f3 ≠ f2 → (c.adjacent f1 f3 ↔ ¬c.adjacent f2 f3)

theorem cube_opposite_face (c : Cube) :
  c.faces = {'x', 'A', 'B', 'C', 'D', 'E', 'F'} →
  c.adjacent 'x' 'A' →
  c.adjacent 'x' 'D' →
  c.adjacent 'x' 'F' →
  c.adjacent 'E' 'D' →
  ¬c.adjacent 'x' 'E' →
  opposite c 'x' 'B' := by
  sorry

end cube_opposite_face_l3427_342794


namespace g_equals_4_at_2_l3427_342707

/-- The function g(x) = 5x - 6 -/
def g (x : ℝ) : ℝ := 5 * x - 6

/-- Theorem: For the function g(x) = 5x - 6, the value of a that satisfies g(a) = 4 is a = 2 -/
theorem g_equals_4_at_2 : g 2 = 4 := by
  sorry

end g_equals_4_at_2_l3427_342707


namespace sum_of_squares_of_roots_l3427_342742

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (3 * a^3 + 2 * a^2 - 3 * a - 8 = 0) →
  (3 * b^3 + 2 * b^2 - 3 * b - 8 = 0) →
  (3 * c^3 + 2 * c^2 - 3 * c - 8 = 0) →
  a^2 + b^2 + c^2 = 22 / 9 := by
sorry

end sum_of_squares_of_roots_l3427_342742


namespace cylinder_volume_equals_cube_surface_l3427_342729

theorem cylinder_volume_equals_cube_surface (side : ℝ) (h r V : ℝ) : 
  side = 3 → 
  6 * side^2 = 2 * π * r^2 + 2 * π * r * h → 
  h = r → 
  V = π * r^2 * h → 
  V = (81 * Real.sqrt 3 / 2) * Real.sqrt 5 / Real.sqrt π :=
by sorry

end cylinder_volume_equals_cube_surface_l3427_342729


namespace first_term_of_geometric_series_l3427_342744

/-- The first term of an infinite geometric series with common ratio 1/4 and sum 80 is 60 -/
theorem first_term_of_geometric_series : 
  ∀ (a : ℝ), 
  (a * (1 - (1/4)⁻¹) = 80) → 
  a = 60 := by
sorry

end first_term_of_geometric_series_l3427_342744


namespace smallest_n_divisible_by_2010_l3427_342784

theorem smallest_n_divisible_by_2010 (a : ℕ → ℤ) 
  (h1 : ∃ k, a 1 = 2 * k + 1)
  (h2 : ∀ n : ℕ, n > 0 → n * (a (n + 1) - a n + 3) = a (n + 1) + a n + 3)
  (h3 : ∃ k, a 2009 = 2010 * k) :
  ∃ n : ℕ, n ≥ 2 ∧ (∃ k, a n = 2010 * k) ∧ (∀ m, 2 ≤ m ∧ m < n → ¬∃ k, a m = 2010 * k) ∧ n = 671 :=
sorry

end smallest_n_divisible_by_2010_l3427_342784


namespace unique_triple_l3427_342737

theorem unique_triple : 
  ∀ a b c : ℝ,
  (a > 0 ∧ b > 0 ∧ c > 0) →
  (a + b + c = 3) →
  (a^2 - a ≥ 1 - b*c) →
  (b^2 - b ≥ 1 - a*c) →
  (c^2 - c ≥ 1 - a*b) →
  (a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end unique_triple_l3427_342737


namespace mod_equivalence_unique_solution_l3427_342768

theorem mod_equivalence_unique_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -4982 [ZMOD 9] ∧ n = 4 := by
  sorry

end mod_equivalence_unique_solution_l3427_342768


namespace system_solution_l3427_342739

theorem system_solution :
  ∃! (x y : ℝ),
    Real.sqrt (2016.5 + x) + Real.sqrt (2016.5 + y) = 114 ∧
    Real.sqrt (2016.5 - x) + Real.sqrt (2016.5 - y) = 56 ∧
    x = 1232.5 ∧ y = 1232.5 := by
  sorry

end system_solution_l3427_342739


namespace stratified_sampling_result_l3427_342749

/-- Calculates the number of high school students selected in a stratified sampling -/
def stratified_sampling (total_students : ℕ) (selected_students : ℕ) (high_school_students : ℕ) : ℕ :=
  (high_school_students * selected_students) / total_students

/-- Theorem: In a stratified sampling of 15 students from 165 students, 
    where 66 are high school students, 6 high school students will be selected -/
theorem stratified_sampling_result : 
  stratified_sampling 165 15 66 = 6 := by
  sorry

end stratified_sampling_result_l3427_342749


namespace parallel_line_through_point_l3427_342769

/-- Given a line L1 with equation 6x - 3y = 9 and a point P (1, -2),
    prove that the line L2 passing through P and parallel to L1
    has the equation y = 2x - 4 in slope-intercept form. -/
theorem parallel_line_through_point (x y : ℝ) :
  (6 * x - 3 * y = 9) →  -- Equation of L1
  (∃ m b : ℝ, ∀ x y : ℝ, y = m * x + b ↔ 6 * x - 3 * y = 9) →  -- L1 in slope-intercept form
  (∃ m' b' : ℝ, 
    (m' = 2 ∧ b' = -4) ∧  -- Equation of L2: y = 2x - 4
    (m' * 1 + b' = -2) ∧  -- L2 passes through (1, -2)
    (∀ x y : ℝ, y = m' * x + b' ↔ y = 2 * x - 4)) :=
by sorry


end parallel_line_through_point_l3427_342769


namespace set_equality_implies_difference_l3427_342759

theorem set_equality_implies_difference (a b : ℝ) :
  ({a, 1} : Set ℝ) = {0, a + b} → b - a = 1 := by
  sorry

end set_equality_implies_difference_l3427_342759


namespace rationalize_denominator_l3427_342752

theorem rationalize_denominator : (14 : ℝ) / Real.sqrt 14 = Real.sqrt 14 := by
  sorry

end rationalize_denominator_l3427_342752


namespace total_pencils_l3427_342736

/-- Given an initial number of pencils and a number of pencils added,
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) :
  initial + added = initial + added :=
by sorry

end total_pencils_l3427_342736


namespace min_value_of_b_l3427_342718

def S (n : ℕ) : ℕ := 2^n - 1

def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => S (n + 1) - S n

def b (n : ℕ) : ℝ := (a n)^2 - 7*(a n) + 6

theorem min_value_of_b :
  ∃ (m : ℝ), ∀ (n : ℕ), b n ≥ m ∧ ∃ (k : ℕ), b k = m ∧ m = -6 := by
  sorry

end min_value_of_b_l3427_342718


namespace perpendicular_line_plane_implies_perpendicular_lines_l3427_342734

structure Plane where
  -- Define plane structure

structure Line where
  -- Define line structure

-- Define perpendicularity between a line and a plane
def perpendicular_line_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Define a line being contained in a plane
def line_in_plane (l : Line) (p : Plane) : Prop :=
  sorry

-- Define perpendicularity between two lines
def perpendicular_lines (l1 l2 : Line) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_line_plane_implies_perpendicular_lines
  (m n : Line) (α : Plane) 
  (h1 : m ≠ n) 
  (h2 : perpendicular_line_plane m α) 
  (h3 : line_in_plane n α) : 
  perpendicular_lines m n :=
sorry

end perpendicular_line_plane_implies_perpendicular_lines_l3427_342734


namespace trig_equation_solution_l3427_342733

theorem trig_equation_solution (z : ℝ) :
  (1 - Real.sin z ^ 6 - Real.cos z ^ 6) / (1 - Real.sin z ^ 4 - Real.cos z ^ 4) = 2 * (Real.cos (3 * z)) ^ 2 →
  ∃ k : ℤ, z = π / 18 * (6 * ↑k + 1) ∨ z = π / 18 * (6 * ↑k - 1) :=
by sorry

end trig_equation_solution_l3427_342733


namespace rebus_solution_l3427_342773

/-- Represents a two-digit number (or single-digit for YA) -/
def TwoDigitNum := {n : ℕ // n < 100}

/-- The rebus equation -/
def rebusEquation (ya oh my : TwoDigitNum) : Prop :=
  ya.val + 8 * oh.val = my.val

/-- All digits in the equation are different -/
def differentDigits (ya oh my : TwoDigitNum) : Prop :=
  ya.val ≠ oh.val ∧ ya.val ≠ my.val ∧ oh.val ≠ my.val

theorem rebus_solution :
  ∃! (ya oh my : TwoDigitNum),
    rebusEquation ya oh my ∧
    differentDigits ya oh my ∧
    ya.val = 0 ∧
    oh.val = 12 ∧
    my.val = 96 :=
sorry

end rebus_solution_l3427_342773


namespace megacorp_oil_refining_earnings_l3427_342701

/-- MegaCorp's financial data and fine calculation --/
theorem megacorp_oil_refining_earnings 
  (daily_mining_earnings : ℝ)
  (monthly_expenses : ℝ)
  (fine_amount : ℝ)
  (fine_rate : ℝ)
  (days_per_month : ℕ)
  (months_per_year : ℕ)
  (h1 : daily_mining_earnings = 3000000)
  (h2 : monthly_expenses = 30000000)
  (h3 : fine_amount = 25600000)
  (h4 : fine_rate = 0.01)
  (h5 : days_per_month = 30)
  (h6 : months_per_year = 12) :
  ∃ daily_oil_earnings : ℝ,
    daily_oil_earnings = 5111111.11 ∧
    fine_amount = fine_rate * months_per_year * 
      (days_per_month * (daily_mining_earnings + daily_oil_earnings) - monthly_expenses) :=
by sorry

end megacorp_oil_refining_earnings_l3427_342701


namespace rational_sum_theorem_l3427_342738

theorem rational_sum_theorem (x y : ℚ) 
  (hx : |x| = 5) 
  (hy : |y| = 2) 
  (hxy : |x - y| = x - y) : 
  x + y = 7 ∨ x + y = 3 := by
sorry

end rational_sum_theorem_l3427_342738


namespace smallest_square_side_length_l3427_342712

theorem smallest_square_side_length : ∃ (n : ℕ), 
  (∀ (a b c d : ℕ), a * b * c * d = n * n) ∧ 
  (∃ (x y z w : ℕ), x * 7 = n ∧ y * 8 = n ∧ z * 9 = n ∧ w * 10 = n) ∧
  (∀ (m : ℕ), 
    (∃ (a b c d : ℕ), a * b * c * d = m * m) ∧ 
    (∃ (x y z w : ℕ), x * 7 = m ∧ y * 8 = m ∧ z * 9 = m ∧ w * 10 = m) →
    m ≥ n) ∧
  n = 1008 := by
  sorry

end smallest_square_side_length_l3427_342712


namespace no_parallel_solution_perpendicular_solutions_l3427_342782

-- Define the lines
def line1 (m : ℝ) (x y : ℝ) : Prop := (2*m^2 + m - 3)*x + (m^2 - m)*y = 2*m
def line2 (x y : ℝ) : Prop := x - y = 1

def line3 (a : ℝ) (x y : ℝ) : Prop := a*x + (1 - a)*y = 3
def line4 (a : ℝ) (x y : ℝ) : Prop := (a - 1)*x + (2*a + 3)*y = 2

-- Define parallel and perpendicular conditions
def parallel (m : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ 2*m^2 + m - 3 = k ∧ m^2 - m = -k

def perpendicular (a : ℝ) : Prop := a*(a - 1) + (1 - a)*(2*a + 3) = 0

-- State the theorems
theorem no_parallel_solution : ¬∃ m : ℝ, parallel m := sorry

theorem perpendicular_solutions : ∀ a : ℝ, perpendicular a ↔ (a = 1 ∨ a = -3) := sorry

end no_parallel_solution_perpendicular_solutions_l3427_342782


namespace eliana_steps_l3427_342715

def steps_day1 (x : ℕ) := 200 + x
def steps_day2 (x : ℕ) := 2 * steps_day1 x
def steps_day3 (x : ℕ) := steps_day2 x + 100

theorem eliana_steps (x : ℕ) :
  steps_day1 x + steps_day2 x + steps_day3 x = 1600 → x = 100 := by
  sorry

end eliana_steps_l3427_342715


namespace paper_boat_time_l3427_342710

/-- The time it takes for a paper boat to travel along an embankment -/
theorem paper_boat_time (embankment_length : ℝ) (boat_length : ℝ) 
  (downstream_time : ℝ) (upstream_time : ℝ) 
  (h1 : embankment_length = 50) 
  (h2 : boat_length = 10)
  (h3 : downstream_time = 5)
  (h4 : upstream_time = 4) : 
  ∃ (paper_boat_time : ℝ), paper_boat_time = 40 := by
  sorry

end paper_boat_time_l3427_342710


namespace units_digit_of_M_M15_l3427_342726

-- Define the Modified Lucas sequence
def M : ℕ → ℕ
  | 0 => 3
  | 1 => 2
  | (n + 2) => M (n + 1) + M n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_M_M15 : unitsDigit (M (M 15)) = 7 := by
  sorry

end units_digit_of_M_M15_l3427_342726


namespace fuel_spending_reduction_l3427_342753

theorem fuel_spending_reduction 
  (old_efficiency : ℝ) 
  (old_fuel_cost : ℝ) 
  (efficiency_improvement : ℝ) 
  (fuel_cost_increase : ℝ) : 
  let new_efficiency : ℝ := old_efficiency * (1 + efficiency_improvement)
  let new_fuel_cost : ℝ := old_fuel_cost * (1 + fuel_cost_increase)
  let old_trip_cost : ℝ := old_fuel_cost
  let new_trip_cost : ℝ := (1 / (1 + efficiency_improvement)) * new_fuel_cost
  let cost_reduction : ℝ := (old_trip_cost - new_trip_cost) / old_trip_cost
  efficiency_improvement = 0.75 ∧ 
  fuel_cost_increase = 0.30 → 
  cost_reduction = 25 / 28 := by
sorry

end fuel_spending_reduction_l3427_342753


namespace problem_statement_l3427_342774

theorem problem_statement (ℓ : ℝ) (h : (1 + ℓ)^2 / (1 + ℓ^2) = 13/37) :
  (1 + ℓ)^3 / (1 + ℓ^3) = 156/1369 := by
  sorry

end problem_statement_l3427_342774


namespace misread_weight_calculation_l3427_342775

/-- Proves that the misread weight in a class of 20 boys is 56 kg given the initial and correct average weights --/
theorem misread_weight_calculation (n : ℕ) (initial_avg : ℝ) (correct_avg : ℝ) (correct_weight : ℝ) :
  n = 20 →
  initial_avg = 58.4 →
  correct_avg = 58.65 →
  correct_weight = 61 →
  ∃ (misread_weight : ℝ),
    misread_weight = 56 ∧
    n * initial_avg + (correct_weight - misread_weight) = n * correct_avg :=
by
  sorry

end misread_weight_calculation_l3427_342775


namespace triangle_area_is_12_l3427_342767

/-- The area of the triangular region bounded by the coordinate axes and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the bounding line -/
def boundingLine (x y : ℝ) : Prop := 3 * x + 2 * y = 12

theorem triangle_area_is_12 : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≥ 0 ∧ y₁ ≥ 0 ∧ 
    x₂ ≥ 0 ∧ y₂ ≥ 0 ∧ 
    boundingLine x₁ y₁ ∧ 
    boundingLine x₂ y₂ ∧ 
    x₁ ≠ x₂ ∧ 
    triangleArea = (1/2) * x₁ * y₂ := by
  sorry

end triangle_area_is_12_l3427_342767


namespace white_dandelions_on_saturday_l3427_342770

/-- Represents the state of dandelions in a meadow on a given day -/
structure DandelionState :=
  (yellow : ℕ)
  (white : ℕ)

/-- The dandelion blooming cycle -/
def dandelionCycle : ℕ := 5

/-- The number of days a dandelion remains yellow -/
def yellowDays : ℕ := 3

/-- The state of dandelions on Monday -/
def mondayState : DandelionState :=
  { yellow := 20, white := 14 }

/-- The state of dandelions on Wednesday -/
def wednesdayState : DandelionState :=
  { yellow := 15, white := 11 }

/-- The number of days between Monday and Saturday -/
def daysToSaturday : ℕ := 5

theorem white_dandelions_on_saturday :
  (wednesdayState.yellow + wednesdayState.white) - mondayState.yellow =
  (mondayState.yellow + mondayState.white + daysToSaturday - dandelionCycle) :=
by sorry

end white_dandelions_on_saturday_l3427_342770


namespace hyperbola_equation_l3427_342700

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a line with slope m and y-intercept c -/
structure Line where
  m : ℝ
  c : ℝ

/-- States that a line passes through a focus of the hyperbola -/
def passes_through_focus (l : Line) (h : Hyperbola) : Prop :=
  ∃ x y, y = l.m * x + l.c ∧ x^2 + y^2 = h.a^2 + h.b^2

/-- States that a line is parallel to an asymptote of the hyperbola -/
def parallel_to_asymptote (l : Line) (h : Hyperbola) : Prop :=
  l.m = h.b / h.a ∨ l.m = -h.b / h.a

theorem hyperbola_equation (h : Hyperbola) (l : Line) 
  (h_focus : passes_through_focus l h)
  (h_parallel : parallel_to_asymptote l h)
  (h_line : l.m = 2 ∧ l.c = 10) :
  h.a^2 = 5 ∧ h.b^2 = 20 := by sorry

end hyperbola_equation_l3427_342700


namespace bruce_calculators_l3427_342793

-- Define the given conditions
def total_money : ℕ := 200
def crayon_cost : ℕ := 5
def book_cost : ℕ := 5
def calculator_cost : ℕ := 5
def bag_cost : ℕ := 10
def crayon_packs : ℕ := 5
def books : ℕ := 10
def bags : ℕ := 11

-- Define the theorem
theorem bruce_calculators :
  let crayon_total := crayon_cost * crayon_packs
  let book_total := book_cost * books
  let remaining_after_books := total_money - (crayon_total + book_total)
  let bag_total := bag_cost * bags
  let remaining_for_calculators := remaining_after_books - bag_total
  remaining_for_calculators / calculator_cost = 3 := by
  sorry

end bruce_calculators_l3427_342793


namespace fundraising_ratio_approx_one_third_l3427_342777

/-- The ratio of Miss Rollin's class contribution to the total school fundraising --/
def fundraising_ratio : ℚ :=
  let johnson_amount := 2300
  let sutton_amount := johnson_amount / 2
  let rollin_amount := sutton_amount * 8
  let total_after_fees := 27048
  let total_before_fees := total_after_fees / 0.98
  rollin_amount / total_before_fees

theorem fundraising_ratio_approx_one_third :
  abs (fundraising_ratio - 1/3) < 0.001 := by
  sorry

end fundraising_ratio_approx_one_third_l3427_342777


namespace onion_to_carrot_ratio_l3427_342758

/-- Represents the number of vegetables Maria wants to cut -/
structure Vegetables where
  potatoes : ℕ
  carrots : ℕ
  onions : ℕ
  green_beans : ℕ

/-- The conditions of Maria's vegetable cutting plan -/
def cutting_plan (v : Vegetables) : Prop :=
  v.carrots = 6 * v.potatoes ∧
  v.onions = v.carrots ∧
  v.green_beans = v.onions / 3 ∧
  v.potatoes = 2 ∧
  v.green_beans = 8

theorem onion_to_carrot_ratio (v : Vegetables) 
  (h : cutting_plan v) : v.onions = v.carrots := by
  sorry

#check onion_to_carrot_ratio

end onion_to_carrot_ratio_l3427_342758


namespace max_a6_value_l3427_342745

theorem max_a6_value (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 1) ≤ (a (n + 2) + a n) / 2)
  (h2 : a 1 = 1)
  (h3 : a 404 = 2016) :
  ∃ M, a 6 ≤ M ∧ M = 26 :=
by sorry

end max_a6_value_l3427_342745


namespace picture_book_shelves_l3427_342747

theorem picture_book_shelves (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ)
  (h1 : books_per_shelf = 8)
  (h2 : mystery_shelves = 5)
  (h3 : total_books = 72) :
  (total_books - mystery_shelves * books_per_shelf) / books_per_shelf = 4 := by
  sorry

end picture_book_shelves_l3427_342747


namespace chord_length_is_four_l3427_342779

/-- Given a line and a circle in 2D space, prove that the length of the chord
    intercepted by the line and the circle is equal to 4. -/
theorem chord_length_is_four (x y : ℝ) : 
  (x + 2 * y - 2 = 0) →  -- Line equation
  ((x - 2)^2 + y^2 = 4) →  -- Circle equation
  ∃ (a b c d : ℝ), 
    (a + 2 * b - 2 = 0) ∧  -- Point (a, b) on the line
    ((a - 2)^2 + b^2 = 4) ∧  -- Point (a, b) on the circle
    (c + 2 * d - 2 = 0) ∧  -- Point (c, d) on the line
    ((c - 2)^2 + d^2 = 4) ∧  -- Point (c, d) on the circle
    (a ≠ c ∨ b ≠ d) ∧  -- (a, b) and (c, d) are distinct points
    ((a - c)^2 + (b - d)^2 = 4^2)  -- Distance between points is 4
  := by sorry

end chord_length_is_four_l3427_342779


namespace square_division_exists_l3427_342772

-- Define a rectangle
structure Rectangle where
  width : ℚ
  height : ℚ

-- Define a function to check if all numbers in a list are distinct
def allDistinct (list : List ℚ) : Prop :=
  ∀ i j, i ≠ j → list.get! i ≠ list.get! j

-- State the theorem
theorem square_division_exists : ∃ (rectangles : List Rectangle),
  -- There are 5 rectangles
  rectangles.length = 5 ∧
  -- The sum of areas equals 1
  (rectangles.map (λ r => r.width * r.height)).sum = 1 ∧
  -- All widths and heights are distinct
  allDistinct (rectangles.map (λ r => r.width) ++ rectangles.map (λ r => r.height)) :=
sorry

end square_division_exists_l3427_342772


namespace parallelogram_perimeter_l3427_342762

/-- Represents a parallelogram EFGH with given side lengths and diagonal -/
structure Parallelogram where
  EF : ℝ
  FG : ℝ
  EH : ℝ

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ := 2 * (p.EF + p.FG)

/-- Theorem: The perimeter of parallelogram EFGH is 140 units -/
theorem parallelogram_perimeter (p : Parallelogram) 
  (h1 : p.EF = 40)
  (h2 : p.FG = 30)
  (h3 : p.EH = 50) : 
  perimeter p = 140 := by
  sorry

end parallelogram_perimeter_l3427_342762


namespace unique_integers_sum_l3427_342756

theorem unique_integers_sum (x : ℝ) : x = Real.sqrt ((Real.sqrt 77) / 2 + 5 / 2) →
  ∃! (a b c : ℕ+), 
    x^100 = 4*x^98 + 18*x^96 + 19*x^94 - x^50 + (a : ℝ)*x^46 + (b : ℝ)*x^44 + (c : ℝ)*x^40 ∧
    (a : ℕ) + (b : ℕ) + (c : ℕ) = 534 := by
  sorry

end unique_integers_sum_l3427_342756


namespace pyramid_lego_count_l3427_342761

/-- Calculates the number of legos for a square level -/
def square_level (side : ℕ) : ℕ := side * side

/-- Calculates the number of legos for a rectangular level -/
def rectangular_level (length width : ℕ) : ℕ := length * width

/-- Calculates the number of legos for a triangular level -/
def triangular_level (side : ℕ) : ℕ := side * (side + 1) / 2 - 3

/-- Calculates the total number of legos for the pyramid -/
def total_legos : ℕ :=
  square_level 10 + rectangular_level 8 6 + triangular_level 4 + 1

theorem pyramid_lego_count : total_legos = 156 := by
  sorry

end pyramid_lego_count_l3427_342761


namespace factorization_sum_l3427_342719

theorem factorization_sum (a b : ℤ) : 
  (∀ x : ℝ, 25 * x^2 - 160 * x - 336 = (5 * x + a) * (5 * x + b)) → 
  a + 2 * b = 20 := by
  sorry

end factorization_sum_l3427_342719


namespace consecutive_integers_squares_minus_product_l3427_342731

theorem consecutive_integers_squares_minus_product (n : ℕ) :
  n = 9 → (n^2 + (n+1)^2) - (n * (n+1)) = 91 := by
  sorry

end consecutive_integers_squares_minus_product_l3427_342731


namespace log_division_simplification_l3427_342741

theorem log_division_simplification :
  Real.log 27 / Real.log (1 / 27) = -1 := by
  sorry

end log_division_simplification_l3427_342741


namespace simplify_power_of_power_l3427_342786

theorem simplify_power_of_power (x : ℝ) : (2 * x^3)^3 = 8 * x^9 := by
  sorry

end simplify_power_of_power_l3427_342786


namespace T_properties_l3427_342785

-- Define the operation T
def T (m n x y : ℚ) : ℚ := (m*x + n*y) * (x + 2*y)

-- State the theorem
theorem T_properties (m n : ℚ) (hm : m ≠ 0) (hn : n ≠ 0) :
  T m n 1 (-1) = 0 ∧ T m n 0 2 = 8 →
  (m = 1 ∧ n = 1) ∧
  (∀ x y : ℚ, x^2 ≠ y^2 → T m n x y = T m n y x → m = 2*n) :=
by sorry

end T_properties_l3427_342785


namespace correct_addition_l3427_342766

theorem correct_addition (x : ℤ) (h : x + 42 = 50) : x + 24 = 32 := by
  sorry

end correct_addition_l3427_342766


namespace y_derivative_l3427_342783

noncomputable def y (x : ℝ) : ℝ := Real.sqrt x + (1/3) * Real.arctan (Real.sqrt x) + (8/3) * Real.arctan (Real.sqrt x / 2)

theorem y_derivative (x : ℝ) (h : x > 0) : 
  deriv y x = (3 * x^2 + 16 * x + 32) / (6 * Real.sqrt x * (x + 1) * (x + 4)) :=
sorry

end y_derivative_l3427_342783


namespace minimum_nickels_needed_l3427_342727

def jacket_cost : ℚ := 45
def ten_dollar_bills : ℕ := 4
def quarters : ℕ := 10
def nickel_value : ℚ := 0.05

theorem minimum_nickels_needed : 
  ∀ n : ℕ, (ten_dollar_bills * 10 + quarters * 0.25 + n * nickel_value ≥ jacket_cost) → n ≥ 50 := by
  sorry

end minimum_nickels_needed_l3427_342727


namespace ellipse_vertex_distance_l3427_342743

/-- The distance between vertices of an ellipse with equation x²/121 + y²/49 = 1 is 22 -/
theorem ellipse_vertex_distance :
  let a : ℝ := Real.sqrt 121
  let b : ℝ := Real.sqrt 49
  let ellipse_equation := fun (x y : ℝ) => x^2 / 121 + y^2 / 49 = 1
  2 * a = 22 :=
by sorry

end ellipse_vertex_distance_l3427_342743


namespace expression_value_l3427_342732

theorem expression_value (x y : ℝ) (h : x - 2*y = -1) : 6 + 2*x - 4*y = 4 := by
  sorry

end expression_value_l3427_342732


namespace zoe_drank_bottles_l3427_342735

def initial_bottles : ℕ := 42
def bought_bottles : ℕ := 30
def final_bottles : ℕ := 47

theorem zoe_drank_bottles :
  ∃ (drank_bottles : ℕ), initial_bottles - drank_bottles + bought_bottles = final_bottles ∧ drank_bottles = 25 := by
  sorry

end zoe_drank_bottles_l3427_342735


namespace gain_percent_calculation_l3427_342708

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h1 : cost_price = 100)
  (h2 : selling_price = 115) :
  (selling_price - cost_price) / cost_price * 100 = 15 := by
sorry

end gain_percent_calculation_l3427_342708


namespace arithmetic_sequence_first_term_l3427_342717

/-- Sum of first n terms of an arithmetic sequence -/
def S (a : ℚ) (n : ℕ) : ℚ := n * (2 * a + (n - 1) * 5) / 2

/-- The theorem statement -/
theorem arithmetic_sequence_first_term (a : ℚ) :
  (∃ c : ℚ, ∀ n : ℕ, n > 0 → S a (2 * n) / S a n = c) →
  a = 5 / 2 := by
  sorry

end arithmetic_sequence_first_term_l3427_342717


namespace arithmetic_sequence_sum_24_l3427_342704

/-- An arithmetic sequence where a_n = 2n - 3 -/
def a (n : ℕ) : ℤ := 2 * n - 3

/-- Sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℤ := (n : ℤ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem arithmetic_sequence_sum_24 :
  ∃ m : ℕ, m > 0 ∧ S m = 24 ∧ ∀ k : ℕ, k > 0 ∧ S k = 24 → k = m :=
by sorry

end arithmetic_sequence_sum_24_l3427_342704


namespace arithmetic_mean_ge_geometric_mean_l3427_342788

theorem arithmetic_mean_ge_geometric_mean (a b : ℝ) : (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end arithmetic_mean_ge_geometric_mean_l3427_342788


namespace geometry_propositions_l3427_342797

theorem geometry_propositions (p₁ p₂ p₃ p₄ : Prop) 
  (h₁ : p₁) (h₂ : ¬p₂) (h₃ : ¬p₃) (h₄ : p₄) : 
  (p₁ ∧ p₄) ∧ (¬p₂ ∨ p₃) ∧ (¬p₃ ∨ ¬p₄) ∧ ¬(p₁ ∧ p₂) := by
  sorry

end geometry_propositions_l3427_342797


namespace opposite_signs_and_greater_magnitude_l3427_342776

theorem opposite_signs_and_greater_magnitude (a b : ℝ) : 
  a * b < 0 → a + b < 0 → 
  ((a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0)) ∧ 
  (max (abs a) (abs b) > min (abs a) (abs b)) :=
by sorry

end opposite_signs_and_greater_magnitude_l3427_342776


namespace chicken_wings_distribution_l3427_342746

theorem chicken_wings_distribution (num_friends : ℕ) (initial_wings : ℕ) (additional_wings : ℕ) :
  num_friends = 4 →
  initial_wings = 9 →
  additional_wings = 7 →
  (initial_wings + additional_wings) / num_friends = 4 :=
by
  sorry

end chicken_wings_distribution_l3427_342746
