import Mathlib

namespace base_5_representation_of_89_l1302_130227

def to_base_5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: to_base_5 (n / 5)

theorem base_5_representation_of_89 :
  to_base_5 89 = [4, 2, 3] :=
by sorry

end base_5_representation_of_89_l1302_130227


namespace sum_f_negative_l1302_130253

/-- The function f(x) = -x^3 - x -/
def f (x : ℝ) : ℝ := -x^3 - x

/-- Theorem: For a, b, c ∈ ℝ satisfying a + b > 0, b + c > 0, and c + a > 0,
    it follows that f(a) + f(b) + f(c) < 0 -/
theorem sum_f_negative (a b c : ℝ) (hab : a + b > 0) (hbc : b + c > 0) (hca : c + a > 0) :
  f a + f b + f c < 0 := by
  sorry

end sum_f_negative_l1302_130253


namespace complex_equation_solution_l1302_130284

theorem complex_equation_solution : ∃ (x y : ℝ), (2*x - 1 : ℂ) + I = y - (2 - y)*I ∧ x = 2 ∧ y = 3 := by
  sorry

end complex_equation_solution_l1302_130284


namespace derivative_x_squared_cos_l1302_130271

/-- The derivative of x^2 * cos(x) is 2x * cos(x) - x^2 * sin(x) -/
theorem derivative_x_squared_cos (x : ℝ) :
  deriv (λ x => x^2 * Real.cos x) x = 2 * x * Real.cos x - x^2 * Real.sin x := by
  sorry

end derivative_x_squared_cos_l1302_130271


namespace fifi_hangers_l1302_130233

theorem fifi_hangers (total green blue yellow pink : ℕ) : 
  total = 16 →
  green = 4 →
  blue = green - 1 →
  yellow = blue - 1 →
  total = green + blue + yellow + pink →
  pink = 7 := by
sorry

end fifi_hangers_l1302_130233


namespace wang_li_final_score_l1302_130277

/-- Calculates the weighted average score given individual scores and weights -/
def weightedAverage (writtenScore demonstrationScore interviewScore : ℚ) 
  (writtenWeight demonstrationWeight interviewWeight : ℚ) : ℚ :=
  (writtenScore * writtenWeight + demonstrationScore * demonstrationWeight + interviewScore * interviewWeight) /
  (writtenWeight + demonstrationWeight + interviewWeight)

/-- Theorem stating that Wang Li's final score is 94 given the specified scores and weights -/
theorem wang_li_final_score :
  weightedAverage 96 90 95 5 3 2 = 94 := by
  sorry


end wang_li_final_score_l1302_130277


namespace max_prime_difference_l1302_130210

def is_prime (n : ℕ) : Prop := sorry

def are_distinct {α : Type*} (l : List α) : Prop := sorry

theorem max_prime_difference (a b c : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : is_prime a ∧ is_prime b ∧ is_prime c ∧ 
        is_prime (a+b-c) ∧ is_prime (a+c-b) ∧ is_prime (b+c-a) ∧ is_prime (a+b+c))
  (h3 : are_distinct [a, b, c, a+b-c, a+c-b, b+c-a, a+b+c])
  (h4 : (a + b = 800) ∨ (a + c = 800) ∨ (b + c = 800)) :
  ∃ d : ℕ, d ≤ 1594 ∧ 
  d = max (a+b+c) (max a (max b (max c (max (a+b-c) (max (a+c-b) (b+c-a)))))) -
      min (a+b+c) (min a (min b (min c (min (a+b-c) (min (a+c-b) (b+c-a)))))) ∧
  ∀ d' : ℕ, d' ≤ d := by sorry

end max_prime_difference_l1302_130210


namespace smallest_n_exceeding_100000_l1302_130231

def sequence_term (n : ℕ) : ℕ := 9 + 10 * (n - 1)

def sequence_sum (n : ℕ) : ℕ := 
  Finset.sum (Finset.range n) (λ i => sequence_term (i + 1))

theorem smallest_n_exceeding_100000 : 
  (∀ k < 142, sequence_sum k ≤ 100000) ∧ 
  sequence_sum 142 > 100000 := by
  sorry

end smallest_n_exceeding_100000_l1302_130231


namespace total_tickets_l1302_130252

/-- The number of tickets Dave used to buy toys -/
def tickets_for_toys : ℕ := 12

/-- The number of tickets Dave used to buy clothes -/
def tickets_for_clothes : ℕ := 7

/-- The difference between tickets used for toys and clothes -/
def tickets_difference : ℕ := 5

/-- Theorem: Given the conditions, Dave won 19 tickets in total -/
theorem total_tickets : 
  tickets_for_toys + tickets_for_clothes = 19 ∧
  tickets_for_toys = tickets_for_clothes + tickets_difference :=
sorry

end total_tickets_l1302_130252


namespace parabola_with_vertex_two_three_l1302_130251

/-- A parabola with vertex (h, k) has the general form y = a(x - h)² + k, where a ≠ 0 -/
def is_parabola (f : ℝ → ℝ) (h k a : ℝ) : Prop :=
  ∀ x, f x = a * (x - h)^2 + k ∧ a ≠ 0

/-- The vertex of a parabola f is the point (h, k) -/
def has_vertex (f : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∃ a : ℝ, is_parabola f h k a

theorem parabola_with_vertex_two_three (f : ℝ → ℝ) :
  has_vertex f 2 3 → (∀ x, f x = -(x - 2)^2 + 3) :=
by
  sorry


end parabola_with_vertex_two_three_l1302_130251


namespace problem_statement_l1302_130228

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

theorem problem_statement :
  (3 ∈ A) ∧ (∀ k : ℤ, 4*k - 2 ∉ A) := by sorry

end problem_statement_l1302_130228


namespace monotone_increasing_condition_l1302_130258

/-- The function f(x) = kx - ln x is monotonically increasing on (1, +∞) if and only if k ≥ 1 -/
theorem monotone_increasing_condition (k : ℝ) :
  (∀ x > 1, Monotone (fun x => k * x - Real.log x)) ↔ k ≥ 1 := by
sorry

end monotone_increasing_condition_l1302_130258


namespace moon_carbon_percentage_l1302_130234

/-- Represents the composition and weight of a celestial body -/
structure CelestialBody where
  weight : ℝ
  iron_percent : ℝ
  carbon_percent : ℝ
  other_percent : ℝ
  other_weight : ℝ

/-- The moon's composition and weight -/
def moon : CelestialBody := {
  weight := 250,
  iron_percent := 50,
  carbon_percent := 20,  -- This is what we want to prove
  other_percent := 30,
  other_weight := 75
}

/-- Mars' composition and weight -/
def mars : CelestialBody := {
  weight := 500,
  iron_percent := 50,
  carbon_percent := 20,
  other_percent := 30,
  other_weight := 150
}

/-- Theorem stating that the moon's carbon percentage is 20% -/
theorem moon_carbon_percentage :
  moon.carbon_percent = 20 ∧
  moon.iron_percent = 50 ∧
  moon.other_percent = 100 - moon.iron_percent - moon.carbon_percent ∧
  moon.weight = 250 ∧
  mars.weight = 2 * moon.weight ∧
  mars.iron_percent = moon.iron_percent ∧
  mars.carbon_percent = moon.carbon_percent ∧
  mars.other_percent = moon.other_percent ∧
  mars.other_weight = 150 ∧
  moon.other_weight = mars.other_weight / 2 := by
  sorry


end moon_carbon_percentage_l1302_130234


namespace interior_angles_increase_l1302_130223

/-- The sum of interior angles of a convex polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

theorem interior_angles_increase (n : ℕ) :
  sum_interior_angles n = 2340 → sum_interior_angles (n + 4) = 3060 := by
  sorry

end interior_angles_increase_l1302_130223


namespace rearrangement_writing_time_l1302_130249

/-- The number of distinct letters in the name --/
def name_length : ℕ := 7

/-- The number of rearrangements that can be written per minute --/
def rearrangements_per_minute : ℕ := 15

/-- The total number of minutes required to write all rearrangements --/
def total_minutes : ℕ := 336

/-- Theorem stating that the total time to write all rearrangements of a 7-letter name
    at a rate of 15 rearrangements per minute is 336 minutes --/
theorem rearrangement_writing_time :
  (Nat.factorial name_length) / rearrangements_per_minute = total_minutes := by
  sorry

end rearrangement_writing_time_l1302_130249


namespace right_triangle_hypotenuse_l1302_130226

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 2 → b = 3 → c^2 = a^2 + b^2 → c = Real.sqrt 13 := by
  sorry

end right_triangle_hypotenuse_l1302_130226


namespace child_ticket_cost_l1302_130201

theorem child_ticket_cost (adult_price : ℕ) (total_attendees : ℕ) (total_revenue : ℕ) (child_attendees : ℕ) : 
  adult_price = 60 →
  total_attendees = 280 →
  total_revenue = 14000 →
  child_attendees = 80 →
  ∃ (child_price : ℕ), child_price = 25 ∧
    total_revenue = adult_price * (total_attendees - child_attendees) + child_price * child_attendees :=
by
  sorry

end child_ticket_cost_l1302_130201


namespace remove_matches_no_rectangle_l1302_130266

-- Define the structure of the grid
def Grid := List (List Bool)

-- Define a function to check if a grid contains a rectangle
def containsRectangle (grid : Grid) : Bool := sorry

-- Define the initial 4x4 grid
def initialGrid : Grid := sorry

-- Define a function to remove matches from the grid
def removeMatches (grid : Grid) (numToRemove : Nat) : Grid := sorry

-- Theorem statement
theorem remove_matches_no_rectangle :
  ∃ (finalGrid : Grid),
    (removeMatches initialGrid 11 = finalGrid) ∧
    (containsRectangle finalGrid = false) := by
  sorry

end remove_matches_no_rectangle_l1302_130266


namespace inverse_difference_inverse_l1302_130255

theorem inverse_difference_inverse (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - z⁻¹)⁻¹ = x * z / (z - x) :=
by sorry

end inverse_difference_inverse_l1302_130255


namespace holly_weekly_pill_count_l1302_130259

/-- Calculates the total number of pills Holly takes in a week -/
def weekly_pill_count (insulin_per_day : ℕ) (blood_pressure_per_day : ℕ) : ℕ :=
  let anticonvulsants_per_day := 2 * blood_pressure_per_day
  let daily_total := insulin_per_day + blood_pressure_per_day + anticonvulsants_per_day
  7 * daily_total

/-- Proves that Holly takes 77 pills in a week given her daily requirements -/
theorem holly_weekly_pill_count : 
  weekly_pill_count 2 3 = 77 := by
  sorry

end holly_weekly_pill_count_l1302_130259


namespace kingfisher_pelican_fish_difference_l1302_130230

theorem kingfisher_pelican_fish_difference (pelican_fish : ℕ) (kingfisher_fish : ℕ) (fisherman_fish : ℕ) : 
  pelican_fish = 13 →
  kingfisher_fish > pelican_fish →
  fisherman_fish = 3 * (pelican_fish + kingfisher_fish) →
  fisherman_fish = pelican_fish + 86 →
  kingfisher_fish - pelican_fish = 7 := by
sorry

end kingfisher_pelican_fish_difference_l1302_130230


namespace carter_to_dog_height_ratio_l1302_130224

-- Define the heights in inches
def dog_height : ℕ := 24
def betty_height_feet : ℕ := 3
def height_difference : ℕ := 12

-- Theorem to prove
theorem carter_to_dog_height_ratio :
  let betty_height_inches : ℕ := betty_height_feet * 12
  let carter_height : ℕ := betty_height_inches + height_difference
  carter_height / dog_height = 2 := by
sorry

end carter_to_dog_height_ratio_l1302_130224


namespace teacher_health_survey_l1302_130265

theorem teacher_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 150)
  (h_high_bp : high_bp = 90)
  (h_heart_trouble : heart_trouble = 60)
  (h_both : both = 30) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 20 := by
sorry

end teacher_health_survey_l1302_130265


namespace product_inequality_l1302_130273

theorem product_inequality (a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h₁ : a₁ > 1) (h₂ : a₂ > 1) (h₃ : a₃ > 1) (h₄ : a₄ > 1) (h₅ : a₅ > 1) : 
  16 * (a₁ * a₂ * a₃ * a₄ * a₅ + 1) ≥ (1 + a₁) * (1 + a₂) * (1 + a₃) * (1 + a₄) * (1 + a₅) := by
  sorry

end product_inequality_l1302_130273


namespace odd_sum_probability_l1302_130260

/-- Represents a tile with a number from 1 to 12 -/
def Tile := Fin 12

/-- Represents a player's selection of 4 tiles -/
def PlayerSelection := Finset Tile

/-- The set of all possible tile selections -/
def AllSelections : Finset (PlayerSelection × PlayerSelection × PlayerSelection) :=
  sorry

/-- Checks if a player's selection sum is odd -/
def isOddSum (selection : PlayerSelection) : Bool :=
  sorry

/-- The set of selections where all players have odd sums -/
def OddSumSelections : Finset (PlayerSelection × PlayerSelection × PlayerSelection) :=
  sorry

/-- The probability of all players obtaining an odd sum -/
theorem odd_sum_probability :
  (Finset.card OddSumSelections : ℚ) / (Finset.card AllSelections : ℚ) = 16 / 385 :=
sorry

end odd_sum_probability_l1302_130260


namespace intersection_of_A_and_B_l1302_130246

def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {3} := by sorry

end intersection_of_A_and_B_l1302_130246


namespace four_integers_average_l1302_130268

theorem four_integers_average (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (a + b + c + d : ℚ) / 4 = 5 →
  ∀ w x y z : ℕ+, w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  (w + x + y + z : ℚ) / 4 = 5 →
  (max a (max b (max c d)) - min a (min b (min c d)) : ℤ) ≥ 
  (max w (max x (max y z)) - min w (min x (min y z)) : ℤ) →
  (a + b + c + d - max a (max b (max c d)) - min a (min b (min c d)) : ℚ) / 2 = 5/2 :=
by sorry

end four_integers_average_l1302_130268


namespace value_of_x_l1302_130247

theorem value_of_x (x y z : ℚ) : 
  x = (1/2) * y → 
  y = (1/5) * z → 
  z = 60 → 
  x = 6 := by
sorry

end value_of_x_l1302_130247


namespace chessboard_coloring_l1302_130237

/-- A move on the chessboard changes the color of all squares in a 2x2 area. -/
def ChessboardMove (n : ℕ) := Fin n → Fin n → Bool

/-- The initial chessboard coloring. -/
def InitialChessboard (n : ℕ) : Fin n → Fin n → Bool :=
  λ i j => (i.val + j.val) % 2 = 0

/-- A sequence of moves. -/
def MoveSequence (n : ℕ) := List (ChessboardMove n)

/-- Apply a move to the chessboard. -/
def ApplyMove (board : Fin n → Fin n → Bool) (move : ChessboardMove n) : Fin n → Fin n → Bool :=
  λ i j => board i j ≠ move i j

/-- Apply a sequence of moves to the chessboard. -/
def ApplyMoveSequence (n : ℕ) (board : Fin n → Fin n → Bool) (moves : MoveSequence n) : Fin n → Fin n → Bool :=
  moves.foldl ApplyMove board

/-- Check if all squares on the board have the same color. -/
def AllSameColor (board : Fin n → Fin n → Bool) : Prop :=
  ∀ i j k l, board i j = board k l

/-- Main theorem: There exists a finite sequence of moves that turns all squares
    the same color if and only if n is divisible by 4. -/
theorem chessboard_coloring (n : ℕ) (h : n ≥ 3) :
  (∃ (moves : MoveSequence n), AllSameColor (ApplyMoveSequence n (InitialChessboard n) moves)) ↔
  4 ∣ n := by
  sorry

end chessboard_coloring_l1302_130237


namespace orthocenter_on_altitude_ratio_HD_HA_is_zero_l1302_130281

/-- A triangle with sides 11, 12, and 13 -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = 11)
  (hb : b = 12)
  (hc : c = 13)

/-- The orthocenter of the triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The altitude from vertex A to side BC -/
def altitude_AD (t : Triangle) : ℝ := sorry

/-- The point D where the altitude AD intersects BC -/
def point_D (t : Triangle) : ℝ × ℝ := sorry

/-- The point A of the triangle -/
def point_A (t : Triangle) : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem orthocenter_on_altitude (t : Triangle) :
  let H := orthocenter t
  let D := point_D t
  distance H D = 0 := by sorry

theorem ratio_HD_HA_is_zero (t : Triangle) :
  let H := orthocenter t
  let D := point_D t
  let A := point_A t
  distance H D / distance H A = 0 := by sorry

end orthocenter_on_altitude_ratio_HD_HA_is_zero_l1302_130281


namespace negation_of_conditional_l1302_130278

theorem negation_of_conditional (a b : ℝ) :
  ¬(a > b → 2*a > 2*b) ↔ (a ≤ b → 2*a ≤ 2*b) := by sorry

end negation_of_conditional_l1302_130278


namespace problem_1_l1302_130270

theorem problem_1 : Real.sqrt 9 * 3⁻¹ + 2^3 / |(-2)| = 5 := by sorry

end problem_1_l1302_130270


namespace inequality_proof_l1302_130235

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_equality : c^2 + a*b = a^2 + b^2) : 
  c^2 + a*b ≤ a*c + b*c := by
  sorry

end inequality_proof_l1302_130235


namespace count_seven_up_to_2017_l1302_130269

/-- Count of digit 7 in a natural number -/
def count_seven (n : ℕ) : ℕ := sorry

/-- Sum of count_seven for all numbers from 1 to n -/
def sum_count_seven (n : ℕ) : ℕ := sorry

theorem count_seven_up_to_2017 : sum_count_seven 2017 = 602 := by sorry

end count_seven_up_to_2017_l1302_130269


namespace book_distribution_l1302_130216

theorem book_distribution (x : ℕ) : 
  (∃ n : ℕ, x = 5 * n + 6) ∧ 
  (1 ≤ x - 7 * ((x - 6) / 5 - 1) ∧ x - 7 * ((x - 6) / 5 - 1) < 7) ↔ 
  (1 ≤ x - 7 * ((x - 6) / 5 - 1) ∧ x - 7 * ((x - 6) / 5 - 1) < 7) :=
sorry

end book_distribution_l1302_130216


namespace count_negative_numbers_l1302_130238

def number_list : List ℝ := [3, 0, -5, 0.48, -(-7), -|(-8)|, -((-4)^2)]

theorem count_negative_numbers : 
  (number_list.filter (λ x => x < 0)).length = 3 := by sorry

end count_negative_numbers_l1302_130238


namespace tan_alpha_values_l1302_130274

theorem tan_alpha_values (α : Real) 
  (h : 2 * Real.sin α ^ 2 + Real.sin α * Real.cos α - 3 * Real.cos α ^ 2 = 7/5) : 
  Real.tan α = 2 ∨ Real.tan α = -11/3 := by
sorry

end tan_alpha_values_l1302_130274


namespace blueberry_baskets_l1302_130206

theorem blueberry_baskets (initial_berries : ℕ) (total_berries : ℕ) : 
  initial_berries = 20 →
  total_berries = 200 →
  (total_berries / initial_berries) - 1 = 9 :=
by
  sorry

end blueberry_baskets_l1302_130206


namespace boat_round_trip_time_specific_boat_round_trip_time_l1302_130208

/-- Calculate the total time for a round trip by boat -/
theorem boat_round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let downstream_time := distance / downstream_speed
  let upstream_time := distance / upstream_speed
  let total_time := downstream_time + upstream_time
  total_time

/-- The total time taken for the specific round trip is approximately 947.6923 hours -/
theorem specific_boat_round_trip_time : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
  |boat_round_trip_time 22 4 10080 - 947.6923| < ε :=
sorry

end boat_round_trip_time_specific_boat_round_trip_time_l1302_130208


namespace midpoint_square_area_l1302_130200

theorem midpoint_square_area (A : ℝ) (h : A = 144) : 
  let s := Real.sqrt A
  let midpoint_side := Real.sqrt ((s/2)^2 + (s/2)^2)
  let midpoint_area := midpoint_side^2
  midpoint_area = 72 := by sorry

end midpoint_square_area_l1302_130200


namespace not_all_odd_l1302_130221

theorem not_all_odd (a b c d : ℕ) (h1 : a = b * c + d) (h2 : d < b) : 
  ¬(Odd a ∧ Odd b ∧ Odd c ∧ Odd d) := by
  sorry

end not_all_odd_l1302_130221


namespace geometric_series_ratio_l1302_130292

/-- 
Given two infinite geometric series:
- First series with first term a₁ = 8 and second term b₁ = 2
- Second series with first term a₂ = 8 and second term b₂ = 2 + m
If the sum of the second series is three times the sum of the first series,
then m = 4.
-/
theorem geometric_series_ratio (m : ℝ) : 
  let a₁ : ℝ := 8
  let b₁ : ℝ := 2
  let a₂ : ℝ := 8
  let b₂ : ℝ := 2 + m
  let r₁ : ℝ := b₁ / a₁
  let r₂ : ℝ := b₂ / a₂
  let s₁ : ℝ := a₁ / (1 - r₁)
  let s₂ : ℝ := a₂ / (1 - r₂)
  s₂ = 3 * s₁ → m = 4 := by
  sorry


end geometric_series_ratio_l1302_130292


namespace point_order_on_increasing_line_a_less_than_b_l1302_130289

/-- A line in 2D space defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a given line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

theorem point_order_on_increasing_line
  (l : Line)
  (p1 p2 : Point)
  (h_slope : l.slope > 0)
  (h_x : p1.x < p2.x)
  (h_on1 : p1.liesOn l)
  (h_on2 : p2.liesOn l) :
  p1.y < p2.y :=
sorry

theorem a_less_than_b :
  let l : Line := { slope := 2/3, intercept := -3 }
  let p1 : Point := { x := -1, y := a }
  let p2 : Point := { x := 1/2, y := b }
  p1.liesOn l → p2.liesOn l → a < b :=
sorry

end point_order_on_increasing_line_a_less_than_b_l1302_130289


namespace simple_interest_problem_l1302_130248

/-- Given a principal amount P and an interest rate r, 
    if P(1 + 2r) = 710 and P(1 + 7r) = 1020, then P = 586 -/
theorem simple_interest_problem (P r : ℝ) 
  (h1 : P * (1 + 2 * r) = 710)
  (h2 : P * (1 + 7 * r) = 1020) : 
  P = 586 := by sorry

end simple_interest_problem_l1302_130248


namespace multiplicative_inverse_203_mod_301_l1302_130243

theorem multiplicative_inverse_203_mod_301 : ∃ x : ℕ, x < 301 ∧ (203 * x) % 301 = 1 :=
by
  use 29
  sorry

end multiplicative_inverse_203_mod_301_l1302_130243


namespace optimal_probability_l1302_130239

-- Define the probability of success for a single shot
variable (p : ℝ)

-- Define the number of successful shots as a random variable
def X : ℕ → ℝ
  | n => p^n * (1 - p)

-- Define the probability of making between 35 and 69 shots
def P_35_to_69 (p : ℝ) : ℝ :=
  p^35 - p^70

-- State the theorem
theorem optimal_probability :
  ∃ (p : ℝ), p > 0 ∧ p < 1 ∧
  (∀ q : ℝ, q > 0 → q < 1 → P_35_to_69 q ≤ P_35_to_69 p) ∧
  p = (1/2)^(1/35) :=
sorry

end optimal_probability_l1302_130239


namespace cricket_runs_l1302_130245

theorem cricket_runs (a b c : ℕ) : 
  a + b + c = 95 →
  3 * a = b →
  5 * b = c →
  c = 75 := by
sorry

end cricket_runs_l1302_130245


namespace jeans_cost_per_pair_l1302_130297

def leonard_cost : ℕ := 250
def michael_backpack_cost : ℕ := 100
def total_spent : ℕ := 450
def jeans_pairs : ℕ := 2

theorem jeans_cost_per_pair : 
  (total_spent - leonard_cost - michael_backpack_cost) / jeans_pairs = 50 :=
by sorry

end jeans_cost_per_pair_l1302_130297


namespace order_parts_count_l1302_130254

-- Define the master's productivity per hour
def master_productivity : ℕ → Prop :=
  λ y => y > 5

-- Define the apprentice's productivity relative to the master
def apprentice_productivity (y : ℕ) : ℕ := y - 2

-- Define the total number of parts in the order
def total_parts (y : ℕ) : ℕ := 2 * y * (y - 2) / (y - 4)

-- Theorem statement
theorem order_parts_count :
  ∀ y : ℕ,
    master_productivity y →
    (∃ t : ℕ, t * y = total_parts y) →
    2 * (apprentice_productivity y) * (t - 1) = total_parts y →
    total_parts y = 24 :=
by
  sorry


end order_parts_count_l1302_130254


namespace divisor_sum_theorem_l1302_130212

def sum_of_geometric_series (a r : ℕ) (n : ℕ) : ℕ := (a * (r^(n+1) - 1)) / (r - 1)

theorem divisor_sum_theorem (k m : ℕ) :
  (sum_of_geometric_series 1 2 k) * (sum_of_geometric_series 1 5 m) = 930 →
  k + m = 6 := by
sorry

end divisor_sum_theorem_l1302_130212


namespace min_reciprocal_sum_l1302_130207

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

end min_reciprocal_sum_l1302_130207


namespace composite_number_quotient_l1302_130295

def composite_numbers : List ℕ := [4, 6, 8, 9, 10, 14, 15, 16, 18, 20, 21, 22, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35, 36, 38, 39]

def product_6th_to_15th : ℕ := (List.take 10 (List.drop 5 composite_numbers)).prod

def product_16th_to_25th : ℕ := (List.take 10 (List.drop 15 composite_numbers)).prod

theorem composite_number_quotient :
  (product_6th_to_15th : ℚ) / product_16th_to_25th =
  (14 * 15 * 16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 : ℚ) /
  (27 * 28 * 30 * 32 * 33 * 34 * 35 * 36 * 38 * 39 : ℚ) :=
by sorry

end composite_number_quotient_l1302_130295


namespace no_real_roots_quadratic_l1302_130220

theorem no_real_roots_quadratic (b : ℝ) : ∀ x : ℝ, x^2 - b*x + 1 ≠ 0 := by
  sorry

end no_real_roots_quadratic_l1302_130220


namespace total_pears_is_105_l1302_130244

/-- The number of pears picked by Jason -/
def jason_pears : ℕ := 46

/-- The number of pears picked by Keith -/
def keith_pears : ℕ := 47

/-- The number of pears picked by Mike -/
def mike_pears : ℕ := 12

/-- The total number of pears picked -/
def total_pears : ℕ := jason_pears + keith_pears + mike_pears

/-- Theorem stating that the total number of pears picked is 105 -/
theorem total_pears_is_105 : total_pears = 105 := by
  sorry

end total_pears_is_105_l1302_130244


namespace angle_values_l1302_130215

/-- Given an angle α with terminal side passing through point P(-3, m) and cosα = -3/5,
    prove the values of m, sinα, and tanα. -/
theorem angle_values (α : Real) (m : Real) 
    (h1 : ∃ (x y : Real), x = -3 ∧ y = m ∧ Real.cos α * Real.sqrt (x^2 + y^2) = x)
    (h2 : Real.cos α = -3/5) :
    (m = 4 ∨ m = -4) ∧ 
    ((Real.sin α = 4/5 ∧ Real.tan α = -4/3) ∨ 
     (Real.sin α = -4/5 ∧ Real.tan α = 4/3)) := by
  sorry

end angle_values_l1302_130215


namespace complement_of_M_in_N_l1302_130204

def M : Set ℕ := {2, 3, 4}
def N : Set ℕ := {0, 2, 3, 4, 5}

theorem complement_of_M_in_N :
  N \ M = {0, 5} := by sorry

end complement_of_M_in_N_l1302_130204


namespace min_value_of_expression_l1302_130211

theorem min_value_of_expression :
  ∃ (x : ℝ), (8 - x) * (6 - x) * (8 + x) * (6 + x) = -196 ∧
  ∀ (y : ℝ), (8 - y) * (6 - y) * (8 + y) * (6 + y) ≥ -196 := by
  sorry

end min_value_of_expression_l1302_130211


namespace hex_to_decimal_conversion_l1302_130280

/-- Given a hexadecimal number m02₍₆₎ that is equivalent to 146 in decimal, 
    prove that m = 4. -/
theorem hex_to_decimal_conversion (m : ℕ) : 
  (2 + m * 6^2 = 146) → m = 4 := by
  sorry

end hex_to_decimal_conversion_l1302_130280


namespace min_value_expression_min_value_achievable_l1302_130275

theorem min_value_expression (x : ℝ) (h : x > -1) :
  2 * x + 1 / (x + 1) ≥ 2 * Real.sqrt 2 - 2 :=
by sorry

theorem min_value_achievable :
  ∃ x > -1, 2 * x + 1 / (x + 1) = 2 * Real.sqrt 2 - 2 :=
by sorry

end min_value_expression_min_value_achievable_l1302_130275


namespace inequality_proof_l1302_130209

theorem inequality_proof (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h1 : x₁ ≥ x₂) (h2 : x₂ ≥ x₃) (h3 : x₃ ≥ x₄) (h4 : x₄ ≥ x₅) (h5 : x₅ ≥ 0) :
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 ≥ 25/2 * (x₄^2 + x₅^2) ∧
  ((x₁ + x₂ + x₃ + x₄ + x₅)^2 = 25/2 * (x₄^2 + x₅^2) ↔ x₁ = x₂ ∧ x₂ = x₃ ∧ x₃ = x₄ ∧ x₄ = x₅) :=
by sorry

end inequality_proof_l1302_130209


namespace triangle_area_quadrilateral_area_n_gon_area_l1302_130218

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

def point (m i : ℕ) : ℝ × ℝ :=
  (fibonacci (m + 2 * i - 1), fibonacci (m + 2 * i))

def polygon_area (n m : ℕ) : ℝ :=
  let vertices := List.range n |>.map (point m)
  -- Area calculation using Shoelace formula
  sorry

theorem triangle_area (m : ℕ) :
  polygon_area 3 m = 0.5 := by sorry

theorem quadrilateral_area (m : ℕ) :
  polygon_area 4 m = 2.5 := by sorry

theorem n_gon_area (n m : ℕ) (h : n ≥ 3) :
  polygon_area n m = (fibonacci (2 * n - 2) - n + 1) / 2 := by sorry

end triangle_area_quadrilateral_area_n_gon_area_l1302_130218


namespace min_value_of_function_l1302_130285

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  x - 4 + 9 / (x + 1) ≥ 1 ∧
  (x - 4 + 9 / (x + 1) = 1 ↔ x = 2) :=
by sorry

end min_value_of_function_l1302_130285


namespace square_garden_multiple_l1302_130287

/-- Given a square garden with perimeter 40 feet and area equal to a multiple of the perimeter plus 20, prove that the multiple is 2. -/
theorem square_garden_multiple (side : ℝ) (multiple : ℝ) : 
  side > 0 →
  4 * side = 40 →
  side^2 = multiple * 40 + 20 →
  multiple = 2 := by sorry

end square_garden_multiple_l1302_130287


namespace greatest_integer_in_ratio_l1302_130229

theorem greatest_integer_in_ratio (a b c : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 84 →
  2 * b = 5 * a →
  7 * a = 2 * c →
  max a (max b c) = 42 := by
sorry

end greatest_integer_in_ratio_l1302_130229


namespace valid_coloring_iff_odd_l1302_130293

/-- A valid coloring of an n-gon satisfies the given conditions --/
def ValidColoring (n : ℕ) (P : Set (Fin n)) (coloring : Fin n → Fin n → Fin n) : Prop :=
  -- P represents the vertices of the n-gon
  (∀ i j : Fin n, coloring i j < n) ∧ 
  -- For any three distinct colors, there exists a triangle with those colors
  (∀ c₁ c₂ c₃ : Fin n, c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ → 
    ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
      coloring i j = c₁ ∧ coloring j k = c₂ ∧ coloring i k = c₃)

/-- A valid coloring of an n-gon exists if and only if n is odd --/
theorem valid_coloring_iff_odd (n : ℕ) :
  (∃ P : Set (Fin n), ∃ coloring : Fin n → Fin n → Fin n, ValidColoring n P coloring) ↔ Odd n :=
sorry

end valid_coloring_iff_odd_l1302_130293


namespace smallest_solution_of_equation_l1302_130214

theorem smallest_solution_of_equation :
  let f (x : ℝ) := (3 * x) / (x - 3) + (3 * x^2 - 27) / x
  ∃ (smallest : ℝ), smallest = (2 - Real.sqrt 31) / 3 ∧
    f smallest = 16 ∧
    ∀ (y : ℝ), y ≠ 3 ∧ y ≠ 0 ∧ f y = 16 → y ≥ smallest :=
by sorry

end smallest_solution_of_equation_l1302_130214


namespace f_two_expression_l1302_130294

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  (f 1 = 2) ∧ 
  (∀ x y : ℝ, f (x * y + f x + 1) = x * f y + f x)

/-- The main theorem stating that f(2) can be expressed as c + 2 -/
theorem f_two_expression 
  (f : ℝ → ℝ) 
  (h : FunctionalEquation f) :
  ∃ c : ℝ, f 2 = c + 2 :=
sorry

end f_two_expression_l1302_130294


namespace chicken_food_consumption_l1302_130279

/-- Given Dany's farm animals and their food consumption, calculate the amount of food each chicken eats per day. -/
theorem chicken_food_consumption 
  (num_cows : ℕ) 
  (num_sheep : ℕ) 
  (num_chickens : ℕ) 
  (cow_sheep_consumption : ℕ) 
  (total_consumption : ℕ) 
  (h1 : num_cows = 4) 
  (h2 : num_sheep = 3) 
  (h3 : num_chickens = 7) 
  (h4 : cow_sheep_consumption = 2) 
  (h5 : total_consumption = 35) : 
  ((total_consumption - (num_cows + num_sheep) * cow_sheep_consumption) / num_chickens : ℚ) = 3 := by
  sorry

end chicken_food_consumption_l1302_130279


namespace inverse_matrices_sum_l1302_130272

/-- Two 3x3 matrices that are inverses of each other -/
def A (x y z w : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x, 2, y],
    ![3, 3, 4],
    ![z, 6, w]]

def B (j k l m : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![-6, j, -12],
    ![k, -14, l],
    ![3, m, 5]]

/-- The theorem stating that the sum of all variables in the inverse matrices equals 52 -/
theorem inverse_matrices_sum (x y z w j k l m : ℝ) :
  (A x y z w) * (B j k l m) = 1 →
  x + y + z + w + j + k + l + m = 52 := by
  sorry

end inverse_matrices_sum_l1302_130272


namespace jasons_betta_fish_count_jasons_betta_fish_count_is_five_l1302_130222

/-- The number of betta fish Jason has, given:
  1. The moray eel eats 20 guppies per day.
  2. Each betta fish eats 7 guppies per day.
  3. The total number of guppies needed per day is 55. -/
theorem jasons_betta_fish_count : ℕ :=
  let moray_eel_guppies : ℕ := 20
  let betta_fish_guppies_per_day : ℕ := 7
  let total_guppies_per_day : ℕ := 55
  let betta_fish_count := (total_guppies_per_day - moray_eel_guppies) / betta_fish_guppies_per_day
  5

/-- Proof that Jason has 5 betta fish -/
theorem jasons_betta_fish_count_is_five : jasons_betta_fish_count = 5 := by
  sorry

end jasons_betta_fish_count_jasons_betta_fish_count_is_five_l1302_130222


namespace no_valid_coloring_l1302_130219

/-- Represents a coloring of a 4x4 grid -/
def Coloring := Fin 4 → Fin 4 → Fin 8

/-- Checks if two cells are adjacent in a 4x4 grid -/
def adjacent (r1 c1 r2 c2 : Fin 4) : Prop :=
  (r1 = r2 ∧ (c1 = c2 + 1 ∨ c2 = c1 + 1)) ∨
  (c1 = c2 ∧ (r1 = r2 + 1 ∨ r2 = r1 + 1))

/-- Checks if a coloring satisfies the condition that every pair of colors
    appears on adjacent cells -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ color1 color2 : Fin 8, color1 < color2 →
    ∃ r1 c1 r2 c2 : Fin 4, 
      adjacent r1 c1 r2 c2 ∧
      c r1 c1 = color1 ∧ c r2 c2 = color2

/-- The main theorem stating that no valid coloring exists -/
theorem no_valid_coloring : ¬∃ c : Coloring, valid_coloring c := by
  sorry

end no_valid_coloring_l1302_130219


namespace problem_solution_l1302_130267

theorem problem_solution (x y : ℝ) (h1 : x > y) 
  (h2 : x^2*y^2 + x^2 + y^2 + 2*x*y = 40) (h3 : x*y + x + y = 8) : 
  x = 3 + Real.sqrt 7 := by
sorry

end problem_solution_l1302_130267


namespace triangle_angle_C_l1302_130202

theorem triangle_angle_C (a b : ℝ) (A : ℝ) :
  a = 1 →
  b = Real.sqrt 2 →
  2 * Real.sin A * (Real.cos (π / 4))^2 + Real.cos A * Real.sin (π / 2) - Real.sin A = 3 / 2 →
  ∃ (C : ℝ), (C = 7 * π / 12 ∨ C = π / 12) ∧ 
  (∃ (B : ℝ), A + B + C = π ∧ Real.sin A / a = Real.sin B / b) :=
by sorry

end triangle_angle_C_l1302_130202


namespace male_attendees_fraction_l1302_130299

theorem male_attendees_fraction (M F : ℝ) : 
  M + F = 1 → 
  (7/8 : ℝ) * M + (4/5 : ℝ) * F = 0.845 → 
  M = 0.6 := by
sorry

end male_attendees_fraction_l1302_130299


namespace inverse_propositions_l1302_130286

-- Definitions for geometric concepts
def Line : Type := sorry
def Angle : Type := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def corresponding_angles_equal (l1 l2 : Line) : Prop := sorry

-- Definition for last digit
def last_digit (n : ℕ) : ℕ := n % 10

theorem inverse_propositions :
  -- 1. If two lines are parallel, then the corresponding angles are equal
  (∀ (l1 l2 : Line), parallel l1 l2 → corresponding_angles_equal l1 l2) ∧
  -- 2. There exist a and b such that a² = b² but a ≠ b
  (∃ (a b : ℝ), a^2 = b^2 ∧ a ≠ b) ∧
  -- 3. There exists a number divisible by 5 whose last digit is not 0
  (∃ (n : ℕ), n % 5 = 0 ∧ last_digit n ≠ 0) := by
  sorry

end inverse_propositions_l1302_130286


namespace absolute_value_equation_solution_l1302_130298

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 4| + 3 * x = 12 :=
by
  sorry

end absolute_value_equation_solution_l1302_130298


namespace solution_in_interval_l1302_130263

theorem solution_in_interval : ∃ x₀ : ℝ, (Real.exp x₀ + x₀ = 2) ∧ (0 < x₀ ∧ x₀ < 1) := by sorry

end solution_in_interval_l1302_130263


namespace inequality_implies_a_geq_4_l1302_130296

theorem inequality_implies_a_geq_4 (a : ℝ) (h_a_pos : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1 / x + a / y) ≥ 9) →
  a ≥ 4 := by
sorry

end inequality_implies_a_geq_4_l1302_130296


namespace parallelogram_area_l1302_130250

/-- The area of a parallelogram with one angle measuring 100 degrees and two consecutive sides of lengths 10 inches and 18 inches is equal to 180 sin(10°) square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 18) (h3 : θ = 100 * π / 180) :
  a * b * Real.sin ((π / 2) - (θ / 2)) = 180 * Real.sin (10 * π / 180) := by
  sorry

end parallelogram_area_l1302_130250


namespace system_of_equations_solution_l1302_130225

theorem system_of_equations_solution :
  ∃ (x y z : ℝ),
    (4*x - 3*y + z = -9) ∧
    (2*x + 5*y - 3*z = 8) ∧
    (x + y + 2*z = 5) ∧
    (x = 1 ∧ y = -1 ∧ z = 3) := by
  sorry

end system_of_equations_solution_l1302_130225


namespace composite_sum_of_powers_l1302_130261

-- Define the problem statement
theorem composite_sum_of_powers (a b c d : ℕ+) (h : a * b = c * d) :
  ∃ (m n : ℕ), m > 1 ∧ n > 1 ∧ a^2016 + b^2016 + c^2016 + d^2016 = m * n :=
by sorry


end composite_sum_of_powers_l1302_130261


namespace eraser_ratio_l1302_130240

theorem eraser_ratio (andrea_erasers : ℕ) (anya_extra_erasers : ℕ) :
  andrea_erasers = 4 →
  anya_extra_erasers = 12 →
  (andrea_erasers + anya_extra_erasers) / andrea_erasers = 4 :=
by
  sorry

end eraser_ratio_l1302_130240


namespace multiple_condition_l1302_130213

theorem multiple_condition (n : ℕ) : 
  n = 1475 → 0 < n → n < 2006 → ∃ k : ℕ, 2006 * n = k * (2006 + n) :=
sorry

end multiple_condition_l1302_130213


namespace trent_kept_tadpoles_l1302_130282

/-- The number of tadpoles Trent initially caught -/
def initial_tadpoles : ℕ := 180

/-- The percentage of tadpoles Trent let go -/
def percent_released : ℚ := 75 / 100

/-- The number of tadpoles Trent kept -/
def tadpoles_kept : ℕ := 45

/-- Theorem stating that the number of tadpoles Trent kept is equal to 45 -/
theorem trent_kept_tadpoles : 
  (initial_tadpoles : ℚ) * (1 - percent_released) = tadpoles_kept := by sorry

end trent_kept_tadpoles_l1302_130282


namespace combined_mpg_l1302_130276

/-- The combined miles per gallon of two cars given their individual mpg and relative distances driven -/
theorem combined_mpg (ray_mpg tom_mpg : ℝ) (h1 : ray_mpg = 48) (h2 : tom_mpg = 24) : 
  let s : ℝ := 1  -- Tom's distance (arbitrary non-zero value)
  let ray_distance := 2 * s
  let tom_distance := s
  let total_distance := ray_distance + tom_distance
  let total_fuel := ray_distance / ray_mpg + tom_distance / tom_mpg
  total_distance / total_fuel = 36 := by
sorry


end combined_mpg_l1302_130276


namespace representation_of_2020_as_sum_of_five_cubes_l1302_130290

theorem representation_of_2020_as_sum_of_five_cubes :
  ∃ (n : ℤ), 2020 = (n + 2)^3 + n^3 + (-n - 1)^3 + (-n - 1)^3 + (-2)^3 :=
by
  use 337
  sorry

end representation_of_2020_as_sum_of_five_cubes_l1302_130290


namespace no_positive_integer_solutions_l1302_130291

theorem no_positive_integer_solutions : 
  ¬ ∃ (a b c : ℕ+), (a * b + b * c = 66) ∧ (a * c + b * c = 35) := by
  sorry

end no_positive_integer_solutions_l1302_130291


namespace complex_number_theorem_l1302_130241

theorem complex_number_theorem (z : ℂ) (b : ℝ) :
  z = (Complex.I ^ 3) / (1 - Complex.I) →
  (∃ (y : ℝ), z + b = Complex.I * y) →
  b = -1/2 := by
sorry

end complex_number_theorem_l1302_130241


namespace fractional_inequality_solution_l1302_130257

def solution_set (x : ℝ) : Prop :=
  x ∈ (Set.Iic 2 \ {2}) ∪ Set.Ici 3

theorem fractional_inequality_solution :
  {x : ℝ | (x - 3) / (x - 2) ≥ 0} = {x : ℝ | solution_set x} :=
by sorry

end fractional_inequality_solution_l1302_130257


namespace equation_solutions_l1302_130262

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = (3 + Real.sqrt 15) / 3 ∧ x2 = (3 - Real.sqrt 15) / 3 ∧ 
    3 * x1^2 - 6 * x1 - 2 = 0 ∧ 3 * x2^2 - 6 * x2 - 2 = 0) ∧
  (∃ y1 y2 : ℝ, y1 = 3 ∧ y2 = 5 ∧ 
    (y1 - 3)^2 = 2 * y1 - 6 ∧ (y2 - 3)^2 = 2 * y2 - 6) :=
by sorry

end equation_solutions_l1302_130262


namespace quadrant_I_solution_condition_l1302_130264

theorem quadrant_I_solution_condition (c : ℝ) :
  (∃ x y : ℝ, 2 * x - y = 5 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ -2 < c ∧ c < 8/5 := by
  sorry

end quadrant_I_solution_condition_l1302_130264


namespace consecutive_cube_product_divisible_by_504_l1302_130232

theorem consecutive_cube_product_divisible_by_504 (a : ℤ) : 
  504 ∣ ((a^3 - 1) * a^3 * (a^3 + 1)) :=
by sorry

end consecutive_cube_product_divisible_by_504_l1302_130232


namespace max_sum_of_three_integers_with_product_24_l1302_130205

theorem max_sum_of_three_integers_with_product_24 :
  (∃ (a b c : ℕ+), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 ∧
    ∀ (x y z : ℕ+), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x * y * z = 24 →
      x + y + z ≤ a + b + c) ∧
  (∀ (a b c : ℕ+), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 →
    a + b + c ≤ 15) :=
by sorry

end max_sum_of_three_integers_with_product_24_l1302_130205


namespace arcsin_equation_solution_l1302_130203

theorem arcsin_equation_solution :
  ∀ x : ℝ, Real.arcsin x + Real.arcsin (3 * x) = π / 2 →
  x = 1 / Real.sqrt 10 ∨ x = -(1 / Real.sqrt 10) := by
sorry

end arcsin_equation_solution_l1302_130203


namespace library_books_remaining_l1302_130242

/-- Calculates the number of remaining books after two days of borrowing. -/
def remaining_books (initial : ℕ) (day1_borrowed : ℕ) (day2_borrowed : ℕ) : ℕ :=
  initial - (day1_borrowed + day2_borrowed)

/-- Theorem stating the number of remaining books in the library scenario. -/
theorem library_books_remaining :
  remaining_books 100 10 20 = 70 := by
  sorry

end library_books_remaining_l1302_130242


namespace intersection_x_coordinate_l1302_130236

-- Define the two lines
def line1 (x y : ℝ) : Prop := y = 3 * x - 20
def line2 (x y : ℝ) : Prop := 3 * x + y = 100

-- Theorem statement
theorem intersection_x_coordinate : 
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 20 := by
  sorry

end intersection_x_coordinate_l1302_130236


namespace distance_ratio_is_two_thirds_l1302_130288

/-- Represents the scenario where a person is between two points -/
structure WalkRideScenario where
  /-- Distance from the person to the apartment -/
  dist_to_apartment : ℝ
  /-- Distance from the person to the library -/
  dist_to_library : ℝ
  /-- Walking speed -/
  walking_speed : ℝ
  /-- Assumption that distances and speed are positive -/
  dist_apartment_pos : 0 < dist_to_apartment
  dist_library_pos : 0 < dist_to_library
  speed_pos : 0 < walking_speed
  /-- Assumption that the person is between the apartment and library -/
  between_points : dist_to_apartment + dist_to_library > 0

/-- The theorem stating that under the given conditions, the ratio of distances is 2/3 -/
theorem distance_ratio_is_two_thirds (scenario : WalkRideScenario) :
  scenario.dist_to_library / scenario.walking_speed =
  scenario.dist_to_apartment / scenario.walking_speed +
  (scenario.dist_to_apartment + scenario.dist_to_library) / (5 * scenario.walking_speed) →
  scenario.dist_to_apartment / scenario.dist_to_library = 2 / 3 := by
  sorry

end distance_ratio_is_two_thirds_l1302_130288


namespace prime_pair_sum_both_prime_prime_pair_product_l1302_130283

/-- Two prime numbers that sum to 101 -/
def prime_pair : (ℕ × ℕ) := sorry

/-- The sum of the prime pair is 101 -/
theorem prime_pair_sum : prime_pair.1 + prime_pair.2 = 101 := sorry

/-- Both numbers in the pair are prime -/
theorem both_prime : 
  Nat.Prime prime_pair.1 ∧ Nat.Prime prime_pair.2 := sorry

/-- The product of the prime pair is 194 -/
theorem prime_pair_product : 
  prime_pair.1 * prime_pair.2 = 194 := sorry

end prime_pair_sum_both_prime_prime_pair_product_l1302_130283


namespace solve_inequality_range_of_m_l1302_130256

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |2*x - 1|

-- Theorem 1: Solving the inequality f(x) > 3-4x
theorem solve_inequality : 
  ∀ x : ℝ, f x > 3 - 4*x ↔ x > 3/5 := by sorry

-- Theorem 2: Finding the range of m
theorem range_of_m : 
  (∀ x : ℝ, f x + |1 - x| ≥ 6*m^2 - 5*m) ↔ m ∈ Set.Icc (-1/6 : ℝ) 1 := by sorry

end solve_inequality_range_of_m_l1302_130256


namespace planes_parallel_if_perpendicular_to_same_line_l1302_130217

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the theorem
theorem planes_parallel_if_perpendicular_to_same_line
  (m n : Line) (α β : Plane)
  (h_not_coincident_lines : m ≠ n)
  (h_not_coincident_planes : α ≠ β)
  (h_m_perp_α : perpendicular m α)
  (h_m_perp_β : perpendicular m β) :
  parallel α β :=
sorry

end planes_parallel_if_perpendicular_to_same_line_l1302_130217
