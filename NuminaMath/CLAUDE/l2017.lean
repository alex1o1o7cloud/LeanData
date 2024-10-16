import Mathlib

namespace NUMINAMATH_CALUDE_triangle_rectangle_area_l2017_201782

theorem triangle_rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : 
  square_area = 1600 ∧ rectangle_breadth = 10 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := (2/5) * circle_radius
  let triangle_height := 3 * circle_radius
  let triangle_area := (1/2) * rectangle_length * triangle_height
  let rectangle_area := rectangle_length * rectangle_breadth
  triangle_area + rectangle_area = 1120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_rectangle_area_l2017_201782


namespace NUMINAMATH_CALUDE_black_cards_taken_out_l2017_201764

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (black_cards : Nat)
  (remaining_black : Nat)

/-- Definition of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    black_cards := 26,
    remaining_black := 22 }

/-- Theorem: The number of black cards taken out is 4 -/
theorem black_cards_taken_out (d : Deck) (h1 : d = standard_deck) :
  d.black_cards - d.remaining_black = 4 := by
  sorry

end NUMINAMATH_CALUDE_black_cards_taken_out_l2017_201764


namespace NUMINAMATH_CALUDE_function_difference_theorem_l2017_201775

theorem function_difference_theorem (p q c : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  let f : ℝ → ℝ := λ x => p * x^6 + q * x^4 + 3 * x - Real.sqrt 2
  let d := f c - f (-c)
  d = 6 * c := by sorry

end NUMINAMATH_CALUDE_function_difference_theorem_l2017_201775


namespace NUMINAMATH_CALUDE_tate_total_years_l2017_201796

/-- Calculates the total years spent by Tate in education and experiences --/
def totalYears (typicalHighSchoolYears : ℕ) : ℕ :=
  let highSchoolYears := typicalHighSchoolYears - 1
  let travelYears := 2
  let bachelorsYears := 2 * highSchoolYears
  let workExperienceYears := 1
  let phdYears := 3 * (highSchoolYears + bachelorsYears)
  highSchoolYears + travelYears + bachelorsYears + workExperienceYears + phdYears

/-- Theorem stating that Tate's total years spent is 39 --/
theorem tate_total_years : totalYears 4 = 39 := by
  sorry

end NUMINAMATH_CALUDE_tate_total_years_l2017_201796


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2017_201753

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2017_201753


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2017_201702

theorem inscribed_circle_radius (a b c : ℝ) (r : ℝ) : 
  a = 5 → b = 12 → c = 13 →
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area = 1.5 * (a + b + c) - 12 →
  r = 33 / 15 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2017_201702


namespace NUMINAMATH_CALUDE_solution_comparison_l2017_201770

theorem solution_comparison (a a' b b' c : ℝ) (ha : a ≠ 0) (ha' : a' ≠ 0) :
  ((c - b) / a > (c - b') / a') ↔ ((c - b') / a' < (c - b) / a) := by sorry

end NUMINAMATH_CALUDE_solution_comparison_l2017_201770


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l2017_201791

theorem cubic_polynomial_root (Q : ℝ → ℝ) : 
  (∀ x, Q x = x^3 - 6*x^2 + 12*x - 11) →
  (∃ a b c : ℤ, ∀ x, Q x = x^3 + a*x^2 + b*x + c) →
  Q (Real.rpow 3 (1/3) + 2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l2017_201791


namespace NUMINAMATH_CALUDE_even_function_sum_l2017_201739

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x + 1

-- Define the property of being an even function
def is_even_function (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc a (a + 1) → g x = g (-x)

-- Theorem statement
theorem even_function_sum (a b : ℝ) :
  is_even_function (f a b) a → a + a^b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_sum_l2017_201739


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2017_201794

/-- Given a hyperbola with focal length 2c = 26 and a²/c = 25/13, 
    its standard equation is x²/25 - y²/144 = 1 or y²/25 - x²/144 = 1 -/
theorem hyperbola_equation (c : ℝ) (a : ℝ) (h1 : 2 * c = 26) (h2 : a^2 / c = 25 / 13) :
  (∃ x y : ℝ, x^2 / 25 - y^2 / 144 = 1) ∨ (∃ x y : ℝ, y^2 / 25 - x^2 / 144 = 1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2017_201794


namespace NUMINAMATH_CALUDE_roof_ratio_l2017_201725

theorem roof_ratio (length width : ℝ) : 
  length * width = 576 →
  length - width = 36 →
  length / width = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_roof_ratio_l2017_201725


namespace NUMINAMATH_CALUDE_two_valid_colorings_l2017_201728

/-- Represents the three possible colors for a hexagon. -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents a position in the hexagonal grid. -/
structure Position :=
  (row : ℕ) (col : ℕ)

/-- Represents the hexagonal grid. -/
def HexGrid := Position → Color

/-- Checks if two positions are adjacent in the hexagonal grid. -/
def are_adjacent (p1 p2 : Position) : Bool :=
  sorry

/-- Checks if a coloring of the hexagonal grid is valid. -/
def is_valid_coloring (grid : HexGrid) : Prop :=
  (grid ⟨1, 1⟩ = Color.Red) ∧
  (∀ p1 p2, are_adjacent p1 p2 → grid p1 ≠ grid p2)

/-- The number of valid colorings for the hexagonal grid. -/
def num_valid_colorings : ℕ :=
  sorry

/-- Theorem stating that there are exactly 2 valid colorings of the hexagonal grid. -/
theorem two_valid_colorings : num_valid_colorings = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_valid_colorings_l2017_201728


namespace NUMINAMATH_CALUDE_cube_root_equivalence_l2017_201700

theorem cube_root_equivalence (x : ℝ) (hx : x > 0) : 
  (x^2 * x^(1/4))^(1/3) = x^(3/4) := by sorry

end NUMINAMATH_CALUDE_cube_root_equivalence_l2017_201700


namespace NUMINAMATH_CALUDE_polynomial_transformation_l2017_201704

theorem polynomial_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^4 + x^3 - 5*x^2 + x + 1 = 0 ↔ x^2 * (y^2 + y - 7) = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l2017_201704


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2017_201781

/-- The quadratic function f(x) = 3x^2 + 6x + 9 has its minimum value at x = -1 -/
theorem quadratic_minimum (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 + 6 * x + 9
  ∀ y : ℝ, f (-1) ≤ f y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2017_201781


namespace NUMINAMATH_CALUDE_f_a_equals_two_l2017_201726

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_a_equals_two (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ a → f x = f (-x)) →
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ a → f x = x^2 + 1) →
  f a = 2 := by sorry

end NUMINAMATH_CALUDE_f_a_equals_two_l2017_201726


namespace NUMINAMATH_CALUDE_divisibility_by_1961_l2017_201760

theorem divisibility_by_1961 (n : ℕ) : ∃ k : ℤ, 5^(2*n) * 3^(4*n) - 2^(6*n) = k * 1961 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1961_l2017_201760


namespace NUMINAMATH_CALUDE_max_value_sum_sqrt_l2017_201701

theorem max_value_sum_sqrt (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 8) : 
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ 3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sum_sqrt_l2017_201701


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_neg_four_l2017_201797

theorem at_least_one_not_greater_than_neg_four
  (a b c : ℝ)
  (ha : a < 0)
  (hb : b < 0)
  (hc : c < 0) :
  (a + 4 / b ≤ -4) ∨ (b + 4 / c ≤ -4) ∨ (c + 4 / a ≤ -4) := by
sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_neg_four_l2017_201797


namespace NUMINAMATH_CALUDE_multiple_with_few_digits_l2017_201756

open Nat

theorem multiple_with_few_digits (k : ℕ) (h : k > 1) :
  ∃ p : ℕ, p.gcd k = k ∧ p < k^4 ∧ (∃ (d₁ d₂ d₃ d₄ : ℕ) (h : d₁ < 10 ∧ d₂ < 10 ∧ d₃ < 10 ∧ d₄ < 10),
    ∀ d : ℕ, d ∈ p.digits 10 → d = d₁ ∨ d = d₂ ∨ d = d₃ ∨ d = d₄) :=
by sorry

end NUMINAMATH_CALUDE_multiple_with_few_digits_l2017_201756


namespace NUMINAMATH_CALUDE_fishmonger_sales_l2017_201799

/-- The total amount of fish sold by a fishmonger in two weeks, given the first week's sales and a multiplier for the second week. -/
def total_fish_sales (first_week : ℕ) (multiplier : ℕ) : ℕ :=
  first_week + multiplier * first_week

/-- Theorem stating that if a fishmonger sold 50 kg of salmon in the first week and three times that amount in the second week, the total amount of fish sold in two weeks is 200 kg. -/
theorem fishmonger_sales : total_fish_sales 50 3 = 200 := by
  sorry

#eval total_fish_sales 50 3

end NUMINAMATH_CALUDE_fishmonger_sales_l2017_201799


namespace NUMINAMATH_CALUDE_jessicas_purchases_total_cost_l2017_201720

/-- The total cost of Jessica's purchases is $21.95, given that she spent $10.22 on a cat toy and $11.73 on a cage. -/
theorem jessicas_purchases_total_cost : 
  let cat_toy_cost : ℚ := 10.22
  let cage_cost : ℚ := 11.73
  cat_toy_cost + cage_cost = 21.95 := by sorry

end NUMINAMATH_CALUDE_jessicas_purchases_total_cost_l2017_201720


namespace NUMINAMATH_CALUDE_apples_picked_total_l2017_201750

/-- The number of apples Benny picked -/
def benny_apples : ℕ := 2

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The total number of apples picked -/
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_picked_total : total_apples = 11 := by
  sorry

end NUMINAMATH_CALUDE_apples_picked_total_l2017_201750


namespace NUMINAMATH_CALUDE_matthew_baking_time_l2017_201722

/-- Represents the time in hours for Matthew's baking process -/
structure BakingTime where
  assembly : ℝ
  normalBaking : ℝ
  decoration : ℝ
  bakingMultiplier : ℝ

/-- Calculates the total time for Matthew's baking process on the day the oven failed -/
def totalBakingTime (bt : BakingTime) : ℝ :=
  bt.assembly + (bt.normalBaking * bt.bakingMultiplier) + bt.decoration

/-- Theorem stating that Matthew's total baking time on the day the oven failed is 5 hours -/
theorem matthew_baking_time :
  ∀ bt : BakingTime,
  bt.assembly = 1 →
  bt.normalBaking = 1.5 →
  bt.decoration = 1 →
  bt.bakingMultiplier = 2 →
  totalBakingTime bt = 5 := by
sorry

end NUMINAMATH_CALUDE_matthew_baking_time_l2017_201722


namespace NUMINAMATH_CALUDE_triangle_existence_l2017_201752

/-- Represents a line in 2D space -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Theorem stating the existence of a triangle given specific conditions -/
theorem triangle_existence 
  (base_length : ℝ) 
  (base_direction : ℝ × ℝ) 
  (angle_difference : ℝ) 
  (third_vertex_line : Line) : 
  ∃ (t : Triangle), 
    -- The base of the triangle has the given length
    (Real.sqrt ((t.B.1 - t.A.1)^2 + (t.B.2 - t.A.2)^2) = base_length) ∧
    -- The direction of the base matches the given direction
    ((t.B.1 - t.A.1, t.B.2 - t.A.2) = base_direction) ∧
    -- The difference between the base angles is as specified
    (∃ (α β : ℝ), α > β ∧ α - β = angle_difference) ∧
    -- The third vertex lies on the given line
    (∃ (k : ℝ), t.C = (third_vertex_line.point.1 + k * third_vertex_line.direction.1,
                       third_vertex_line.point.2 + k * third_vertex_line.direction.2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_existence_l2017_201752


namespace NUMINAMATH_CALUDE_stream_speed_relationship_l2017_201793

-- Define the boat speeds and distances
def low_speed : ℝ := 20
def high_speed : ℝ := 40
def downstream_distance : ℝ := 26
def upstream_distance : ℝ := 14

-- Define the stream speeds as variables
variable (x y : ℝ)

-- Define the theorem
theorem stream_speed_relationship :
  (downstream_distance / (low_speed + x) = upstream_distance / (high_speed - y)) →
  380 = 7 * x + 13 * y :=
by
  sorry


end NUMINAMATH_CALUDE_stream_speed_relationship_l2017_201793


namespace NUMINAMATH_CALUDE_initial_game_cost_l2017_201709

theorem initial_game_cost (triple_value : ℝ → ℝ) (sold_percentage : ℝ) (sold_amount : ℝ) :
  triple_value = (λ x => 3 * x) →
  sold_percentage = 0.4 →
  sold_amount = 240 →
  ∃ (initial_cost : ℝ), sold_percentage * triple_value initial_cost = sold_amount ∧ initial_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_initial_game_cost_l2017_201709


namespace NUMINAMATH_CALUDE_piggy_bank_pennies_l2017_201749

theorem piggy_bank_pennies (num_compartments : ℕ) (initial_pennies : ℕ) (added_pennies : ℕ) : 
  num_compartments = 12 → 
  initial_pennies = 2 → 
  added_pennies = 6 → 
  (num_compartments * (initial_pennies + added_pennies)) = 96 := by
sorry

end NUMINAMATH_CALUDE_piggy_bank_pennies_l2017_201749


namespace NUMINAMATH_CALUDE_log_216_equals_3_log_36_l2017_201729

theorem log_216_equals_3_log_36 : Real.log 216 = 3 * Real.log 36 := by
  sorry

end NUMINAMATH_CALUDE_log_216_equals_3_log_36_l2017_201729


namespace NUMINAMATH_CALUDE_tiffany_cans_l2017_201747

theorem tiffany_cans (x : ℕ) : x + 4 = 8 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_tiffany_cans_l2017_201747


namespace NUMINAMATH_CALUDE_smallest_w_proof_l2017_201741

def smallest_w : ℕ := 79092

theorem smallest_w_proof :
  ∀ w : ℕ,
  w > 0 →
  (∃ k : ℕ, 1452 * w = 2^4 * 3^3 * 13^3 * k) →
  w ≥ smallest_w :=
by
  sorry

#check smallest_w_proof

end NUMINAMATH_CALUDE_smallest_w_proof_l2017_201741


namespace NUMINAMATH_CALUDE_rotation_result_l2017_201759

/-- Represents a 3D vector -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotates a vector 180° about the y-axis -/
def rotateY180 (v : Vector3D) : Vector3D :=
  { x := -v.x, y := v.y, z := -v.z }

/-- The given vector -/
def givenVector : Vector3D :=
  { x := 2, y := -1, z := 1 }

/-- The expected result after rotation -/
def expectedResult : Vector3D :=
  { x := -2, y := -1, z := -1 }

theorem rotation_result :
  rotateY180 givenVector = expectedResult := by sorry

end NUMINAMATH_CALUDE_rotation_result_l2017_201759


namespace NUMINAMATH_CALUDE_people_who_got_off_l2017_201719

theorem people_who_got_off (initial_people : ℕ) (people_left : ℕ) : 
  initial_people = 48 → people_left = 31 → initial_people - people_left = 17 := by
  sorry

end NUMINAMATH_CALUDE_people_who_got_off_l2017_201719


namespace NUMINAMATH_CALUDE_game_ends_after_54_rounds_l2017_201737

/-- Represents a player in the token game -/
structure Player where
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  playerA : Player
  playerB : Player
  playerC : Player
  rounds : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- Main theorem: The game ends after exactly 54 rounds -/
theorem game_ends_after_54_rounds :
  let initialState : GameState := {
    playerA := { tokens := 20 },
    playerB := { tokens := 19 },
    playerC := { tokens := 18 },
    rounds := 0
  }
  ∃ (finalState : GameState),
    (finalState.rounds = 54) ∧
    (gameEnded finalState) ∧
    (∀ (intermediateState : GameState),
      intermediateState.rounds < 54 →
      ¬(gameEnded intermediateState)) :=
  sorry

end NUMINAMATH_CALUDE_game_ends_after_54_rounds_l2017_201737


namespace NUMINAMATH_CALUDE_unique_solution_cubic_system_l2017_201707

theorem unique_solution_cubic_system (x y z : ℝ) :
  x + y + z = 3 →
  x^2 + y^2 + z^2 = 3 →
  x^3 + y^3 + z^3 = 3 →
  x = 1 ∧ y = 1 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_system_l2017_201707


namespace NUMINAMATH_CALUDE_paul_remaining_money_l2017_201746

/-- The amount of money Paul had for shopping -/
def initial_money : ℕ := 15

/-- The cost of bread -/
def bread_cost : ℕ := 2

/-- The cost of butter -/
def butter_cost : ℕ := 3

/-- The cost of juice (twice the price of bread) -/
def juice_cost : ℕ := 2 * bread_cost

/-- The total cost of groceries -/
def total_cost : ℕ := bread_cost + butter_cost + juice_cost

/-- The remaining money after shopping -/
def remaining_money : ℕ := initial_money - total_cost

theorem paul_remaining_money :
  remaining_money = 6 :=
sorry

end NUMINAMATH_CALUDE_paul_remaining_money_l2017_201746


namespace NUMINAMATH_CALUDE_angle_c_is_right_angle_l2017_201795

theorem angle_c_is_right_angle (A B C : ℝ) (a b c : ℝ) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (a > 0) → (b > 0) → (c > 0) →
  (A + B + C = π) →
  (a / Real.sin B + b / Real.sin A = 2 * c) →
  C = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_c_is_right_angle_l2017_201795


namespace NUMINAMATH_CALUDE_k_gt_one_sufficient_k_gt_one_not_necessary_k_gt_one_sufficient_not_necessary_l2017_201705

/-- The equation of a possible hyperbola -/
def hyperbola_equation (k x y : ℝ) : Prop :=
  x^2 / (k - 1) - y^2 / (k + 1) = 1

/-- Condition for the equation to represent a hyperbola -/
def is_hyperbola (k : ℝ) : Prop :=
  (k - 1) * (k + 1) > 0

/-- k > 1 is sufficient for the equation to represent a hyperbola -/
theorem k_gt_one_sufficient (k : ℝ) (h : k > 1) : is_hyperbola k := by sorry

/-- k > 1 is not necessary for the equation to represent a hyperbola -/
theorem k_gt_one_not_necessary : ∃ k : ℝ, is_hyperbola k ∧ ¬(k > 1) := by sorry

/-- k > 1 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem k_gt_one_sufficient_not_necessary :
  (∀ k : ℝ, k > 1 → is_hyperbola k) ∧ (∃ k : ℝ, is_hyperbola k ∧ ¬(k > 1)) := by sorry

end NUMINAMATH_CALUDE_k_gt_one_sufficient_k_gt_one_not_necessary_k_gt_one_sufficient_not_necessary_l2017_201705


namespace NUMINAMATH_CALUDE_product_mod_seven_l2017_201706

theorem product_mod_seven : (2015 * 2016 * 2017 * 2018) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l2017_201706


namespace NUMINAMATH_CALUDE_roger_apps_deletion_l2017_201792

/-- The number of apps Roger must delete for optimal phone function -/
def apps_to_delete (max_apps : ℕ) (recommended_apps : ℕ) : ℕ :=
  2 * recommended_apps - max_apps

/-- Theorem stating the number of apps Roger must delete -/
theorem roger_apps_deletion :
  apps_to_delete 50 35 = 20 := by
  sorry

end NUMINAMATH_CALUDE_roger_apps_deletion_l2017_201792


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l2017_201787

/-- The line y = kx + 1 (k ∈ ℝ) always has a common point with the curve x²/5 + y²/m = 1
    if and only if m ≥ 1 and m ≠ 5, where m is a non-negative real number. -/
theorem line_ellipse_intersection (m : ℝ) (h_m_nonneg : m ≥ 0) :
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) ↔ m ≥ 1 ∧ m ≠ 5 :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l2017_201787


namespace NUMINAMATH_CALUDE_pear_juice_percentage_is_correct_l2017_201779

/-- The amount of pear juice produced by a single pear -/
def pear_juice_per_fruit : ℚ := 10 / 5

/-- The amount of orange juice produced by a single orange -/
def orange_juice_per_fruit : ℚ := 12 / 3

/-- The number of each fruit used in the blend -/
def fruits_in_blend : ℕ := 10

/-- The total amount of pear juice in the blend -/
def pear_juice_in_blend : ℚ := fruits_in_blend * pear_juice_per_fruit

/-- The total amount of orange juice in the blend -/
def orange_juice_in_blend : ℚ := fruits_in_blend * orange_juice_per_fruit

/-- The total amount of juice in the blend -/
def total_juice_in_blend : ℚ := pear_juice_in_blend + orange_juice_in_blend

/-- The percentage of pear juice in the blend -/
def pear_juice_percentage : ℚ := pear_juice_in_blend / total_juice_in_blend * 100

theorem pear_juice_percentage_is_correct : 
  pear_juice_percentage = 100/3 := by sorry

end NUMINAMATH_CALUDE_pear_juice_percentage_is_correct_l2017_201779


namespace NUMINAMATH_CALUDE_dave_trays_first_table_l2017_201723

/-- The number of trays Dave can carry per trip -/
def trays_per_trip : ℕ := 9

/-- The number of trips Dave made -/
def total_trips : ℕ := 8

/-- The number of trays Dave picked up from the second table -/
def trays_from_second_table : ℕ := 55

/-- The number of trays Dave picked up from the first table -/
def trays_from_first_table : ℕ := trays_per_trip * total_trips - trays_from_second_table

theorem dave_trays_first_table : trays_from_first_table = 17 := by
  sorry

end NUMINAMATH_CALUDE_dave_trays_first_table_l2017_201723


namespace NUMINAMATH_CALUDE_no_small_order_of_two_l2017_201711

theorem no_small_order_of_two (p : ℕ) (h1 : Prime p) (h2 : ∃ k : ℕ, p = 4 * k + 1) (h3 : Prime (2 * p + 1)) :
  ¬ ∃ k : ℕ, k < 2 * p ∧ (2 : ZMod (2 * p + 1))^k = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_small_order_of_two_l2017_201711


namespace NUMINAMATH_CALUDE_special_ellipse_minor_axis_length_special_ellipse_minor_axis_length_is_four_l2017_201769

/-- An ellipse passing through five given points with specific properties -/
structure SpecialEllipse where
  -- The five points the ellipse passes through
  p₁ : ℝ × ℝ := (-1, -1)
  p₂ : ℝ × ℝ := (0, 0)
  p₃ : ℝ × ℝ := (0, 4)
  p₄ : ℝ × ℝ := (4, 0)
  p₅ : ℝ × ℝ := (4, 4)
  -- The center of the ellipse
  center : ℝ × ℝ := (2, 2)
  -- The ellipse has axes parallel to the coordinate axes
  axes_parallel : Bool

/-- The length of the minor axis of the special ellipse is 4 -/
theorem special_ellipse_minor_axis_length (e : SpecialEllipse) : ℝ :=
  4

/-- The main theorem: The length of the minor axis of the special ellipse is 4 -/
theorem special_ellipse_minor_axis_length_is_four (e : SpecialEllipse) :
  special_ellipse_minor_axis_length e = 4 := by
  sorry

end NUMINAMATH_CALUDE_special_ellipse_minor_axis_length_special_ellipse_minor_axis_length_is_four_l2017_201769


namespace NUMINAMATH_CALUDE_angle_terminal_side_theorem_l2017_201732

theorem angle_terminal_side_theorem (θ : Real) :
  let P : ℝ × ℝ := (-4, 3)
  let r : ℝ := Real.sqrt (P.1^2 + P.2^2)
  (∃ t : ℝ, t > 0 ∧ t * (Real.cos θ) = P.1 ∧ t * (Real.sin θ) = P.2) →
  3 * Real.sin θ + Real.cos θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_theorem_l2017_201732


namespace NUMINAMATH_CALUDE_area_triangle_OCD_l2017_201727

/-- Given a trapezoid ABCD and a parallelogram ABGH inscribed within it,
    this theorem calculates the area of triangle OCD. -/
theorem area_triangle_OCD (S_ABCD S_ABGH : ℝ) (h1 : S_ABCD = 320) (h2 : S_ABGH = 80) :
  ∃ (S_OCD : ℝ), S_OCD = 45 :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_OCD_l2017_201727


namespace NUMINAMATH_CALUDE_acorn_price_multiple_l2017_201763

theorem acorn_price_multiple :
  let alice_acorns : ℕ := 3600
  let alice_price_per_acorn : ℕ := 15
  let bob_total_payment : ℕ := 6000
  let alice_total_payment := alice_acorns * alice_price_per_acorn
  (alice_total_payment : ℚ) / bob_total_payment = 9 := by
  sorry

end NUMINAMATH_CALUDE_acorn_price_multiple_l2017_201763


namespace NUMINAMATH_CALUDE_prob_jill_draws_spade_l2017_201717

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Probability of drawing a spade from a standard deck -/
def ProbSpade : ℚ := NumSpades / StandardDeck

/-- Probability of not drawing a spade from a standard deck -/
def ProbNotSpade : ℚ := 1 - ProbSpade

/-- Probability that Jill draws a spade in a single round -/
def ProbJillSpadeInRound : ℚ := ProbNotSpade * ProbSpade

/-- Probability that neither Jack nor Jill draws a spade in a round -/
def ProbNoSpadeInRound : ℚ := ProbNotSpade * ProbNotSpade

theorem prob_jill_draws_spade :
  (ProbJillSpadeInRound / (1 - ProbNoSpadeInRound)) = 3 / 7 :=
sorry

end NUMINAMATH_CALUDE_prob_jill_draws_spade_l2017_201717


namespace NUMINAMATH_CALUDE_tournament_committee_count_l2017_201736

/-- The number of teams in the frisbee league -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 8

/-- The number of members selected from the host team for the committee -/
def host_committee_size : ℕ := 4

/-- The number of members selected from each non-host team for the committee -/
def non_host_committee_size : ℕ := 3

/-- The total size of the tournament committee -/
def total_committee_size : ℕ := 13

/-- The number of possible tournament committees -/
def num_possible_committees : ℕ := 3443073600

theorem tournament_committee_count :
  (num_teams * (Nat.choose team_size host_committee_size) * 
   (Nat.choose team_size non_host_committee_size) ^ (num_teams - 1)) = num_possible_committees :=
by sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l2017_201736


namespace NUMINAMATH_CALUDE_calculation_proof_l2017_201743

theorem calculation_proof : (0.0077 * 4.5) / (0.05 * 0.1 * 0.007) = 989.2857142857143 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2017_201743


namespace NUMINAMATH_CALUDE_cement_mixture_water_fraction_l2017_201703

/-- The fraction of water in a cement mixture -/
def water_fraction (total_weight sand_fraction gravel_weight : ℚ) : ℚ :=
  1 - sand_fraction - (gravel_weight / total_weight)

/-- Proof that the fraction of water in the cement mixture is 2/5 -/
theorem cement_mixture_water_fraction :
  let total_weight : ℚ := 40
  let sand_fraction : ℚ := 1/4
  let gravel_weight : ℚ := 14
  water_fraction total_weight sand_fraction gravel_weight = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_cement_mixture_water_fraction_l2017_201703


namespace NUMINAMATH_CALUDE_solution_difference_l2017_201716

-- Define the function f
def f (c₁ c₂ c₃ : ℕ) (x : ℝ) : ℝ :=
  (x^2 - 6*x + c₁) * (x^2 - 6*x + c₂) * (x^2 - 6*x + c₃)

-- Define the set M
def M (c₁ c₂ c₃ : ℕ) : Set ℕ :=
  {x : ℕ | f c₁ c₂ c₃ x = 0}

-- State the theorem
theorem solution_difference (c₁ c₂ c₃ : ℕ) :
  (c₁ ≥ c₂) → (c₂ ≥ c₃) →
  (∃ x₁ x₂ x₃ x₄ x₅ : ℕ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₁ ≠ x₅ ∧
                         x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₂ ≠ x₅ ∧
                         x₃ ≠ x₄ ∧ x₃ ≠ x₅ ∧
                         x₄ ≠ x₅ ∧
                         M c₁ c₂ c₃ = {x₁, x₂, x₃, x₄, x₅}) →
  c₁ - c₃ = 4 :=
by sorry

end NUMINAMATH_CALUDE_solution_difference_l2017_201716


namespace NUMINAMATH_CALUDE_combined_tennis_preference_l2017_201710

/-- Calculates the combined percentage of students preferring tennis across three schools -/
theorem combined_tennis_preference (north_students : ℕ) (north_tennis_pct : ℚ)
  (south_students : ℕ) (south_tennis_pct : ℚ)
  (valley_students : ℕ) (valley_tennis_pct : ℚ)
  (h1 : north_students = 1800)
  (h2 : north_tennis_pct = 25 / 100)
  (h3 : south_students = 3000)
  (h4 : south_tennis_pct = 50 / 100)
  (h5 : valley_students = 800)
  (h6 : valley_tennis_pct = 30 / 100) :
  (north_students * north_tennis_pct +
   south_students * south_tennis_pct +
   valley_students * valley_tennis_pct) /
  (north_students + south_students + valley_students) =
  39 / 100 := by
  sorry

end NUMINAMATH_CALUDE_combined_tennis_preference_l2017_201710


namespace NUMINAMATH_CALUDE_greatest_constant_inequality_l2017_201778

theorem greatest_constant_inequality (α : ℝ) (hα : α > 0) :
  ∃ C : ℝ, (C = 8) ∧ 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y + y * z + z * x = α →
    (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ C * (x / z + z / x + 2)) ∧
  (∀ C' : ℝ, C' > C →
    ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * y + y * z + z * x = α ∧
      (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) < C' * (x / z + z / x + 2)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_constant_inequality_l2017_201778


namespace NUMINAMATH_CALUDE_exist_positive_reals_satisfying_inequalities_l2017_201745

theorem exist_positive_reals_satisfying_inequalities :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 + c^2 > 2 ∧
    a^3 + b^3 + c^3 < 2 ∧
    a^4 + b^4 + c^4 > 2 := by
  sorry

end NUMINAMATH_CALUDE_exist_positive_reals_satisfying_inequalities_l2017_201745


namespace NUMINAMATH_CALUDE_circle_not_intersecting_diagonal_probability_l2017_201733

/-- The probability that a circle of radius 1 randomly placed inside a 15 × 36 rectangle
    does not intersect the diagonal of the rectangle -/
theorem circle_not_intersecting_diagonal_probability : ℝ := by
  -- Define the rectangle dimensions
  let rectangle_width : ℝ := 15
  let rectangle_height : ℝ := 36
  
  -- Define the circle radius
  let circle_radius : ℝ := 1
  
  -- Define the valid region for circle center
  let valid_region_width : ℝ := rectangle_width - 2 * circle_radius
  let valid_region_height : ℝ := rectangle_height - 2 * circle_radius
  
  -- Calculate the area of the valid region
  let valid_region_area : ℝ := valid_region_width * valid_region_height
  
  -- Define the safe area where the circle doesn't intersect the diagonal
  let safe_area : ℝ := 375
  
  -- Calculate the probability
  let probability : ℝ := safe_area / valid_region_area
  
  -- Prove that the probability equals 375/442
  sorry

#eval (375 : ℚ) / 442

end NUMINAMATH_CALUDE_circle_not_intersecting_diagonal_probability_l2017_201733


namespace NUMINAMATH_CALUDE_range_of_f_l2017_201744

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 2*x + 3

-- State the theorem
theorem range_of_f :
  {y | ∃ x ≥ 0, f x = y} = {y | y ≥ 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2017_201744


namespace NUMINAMATH_CALUDE_tony_packs_count_l2017_201765

/-- The number of pens in each pack -/
def pens_per_pack : ℕ := 3

/-- The number of packs Kendra has -/
def kendra_packs : ℕ := 4

/-- The number of pens Kendra and Tony each keep for themselves -/
def pens_kept : ℕ := 2

/-- The number of friends who receive pens -/
def friends : ℕ := 14

/-- The number of packs Tony has -/
def tony_packs : ℕ := 2

theorem tony_packs_count :
  tony_packs * pens_per_pack + kendra_packs * pens_per_pack = 
  friends + 2 * pens_kept :=
by sorry

end NUMINAMATH_CALUDE_tony_packs_count_l2017_201765


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_fibonacci_factorial_series_l2017_201734

def last_two_digits (n : ℕ) : ℕ := n % 100

def fibonacci_factorial_series : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

def factorial_ends_in_zeros (n : ℕ) : Prop := n > 10 → last_two_digits (n.factorial) = 0

theorem sum_of_last_two_digits_fibonacci_factorial_series :
  factorial_ends_in_zeros 11 →
  (fibonacci_factorial_series.map (λ n => last_two_digits n.factorial)).sum = 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_fibonacci_factorial_series_l2017_201734


namespace NUMINAMATH_CALUDE_power_equation_solution_l2017_201771

theorem power_equation_solution : 
  ∃! x : ℤ, (10 : ℝ) ^ x * (10 : ℝ) ^ 652 = 1000 ∧ x = -649 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2017_201771


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_negative_five_l2017_201767

theorem no_solution_implies_m_equals_negative_five (m : ℝ) : 
  (∀ x : ℝ, x ≠ -1 → (3*x - 2)/(x + 1) ≠ 2 + m/(x + 1)) → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_negative_five_l2017_201767


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l2017_201754

/-- The count of three-digit numbers without exactly two identical adjacent digits -/
def validThreeDigitNumbers : ℕ :=
  let totalThreeDigitNumbers := 900
  let excludedNumbers := 162
  totalThreeDigitNumbers - excludedNumbers

/-- Theorem stating that the count of valid three-digit numbers is 738 -/
theorem valid_three_digit_numbers_count :
  validThreeDigitNumbers = 738 := by
  sorry

end NUMINAMATH_CALUDE_valid_three_digit_numbers_count_l2017_201754


namespace NUMINAMATH_CALUDE_adjusted_retail_price_l2017_201762

/-- The adjusted retail price of a shirt given its cost price and price adjustments -/
theorem adjusted_retail_price 
  (a : ℝ) -- Cost price per shirt in yuan
  (m : ℝ) -- Initial markup percentage
  (n : ℝ) -- Price adjustment percentage
  : ℝ := by
  -- The adjusted retail price is a(1+m%)n% yuan
  sorry

#check adjusted_retail_price

end NUMINAMATH_CALUDE_adjusted_retail_price_l2017_201762


namespace NUMINAMATH_CALUDE_cookies_per_bag_l2017_201715

theorem cookies_per_bag 
  (chocolate_chip : ℕ) 
  (oatmeal : ℕ) 
  (baggies : ℕ) 
  (h1 : chocolate_chip = 2) 
  (h2 : oatmeal = 16) 
  (h3 : baggies = 6) 
  : (chocolate_chip + oatmeal) / baggies = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l2017_201715


namespace NUMINAMATH_CALUDE_lower_limit_of_set_D_l2017_201714

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def SetD : Set ℕ := {n : ℕ | isPrime n ∧ n ≤ 25}

theorem lower_limit_of_set_D (rangeD : ℕ) (h_range : rangeD = 12) :
  ∃ (lower : ℕ), lower = 13 ∧ 
    (∀ n ∈ SetD, n ≥ lower) ∧
    (∃ m ∈ SetD, m = lower) ∧
    (∃ max ∈ SetD, max - lower = rangeD) :=
sorry

end NUMINAMATH_CALUDE_lower_limit_of_set_D_l2017_201714


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2017_201786

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x < -1)) ↔ (∃ x : ℝ, x ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2017_201786


namespace NUMINAMATH_CALUDE_max_value_expression_l2017_201761

theorem max_value_expression (x y : ℝ) : 2 * y^2 - y^4 - x^2 - 3 * x ≤ 13/4 := by sorry

end NUMINAMATH_CALUDE_max_value_expression_l2017_201761


namespace NUMINAMATH_CALUDE_no_geometric_progression_l2017_201798

/-- The sequence a_n defined as 3^n - 2^n -/
def a (n : ℕ) : ℤ := 3^n - 2^n

/-- Theorem stating that no three consecutive terms of the sequence form a geometric progression -/
theorem no_geometric_progression (m n : ℕ) (h : m < n) :
  a m * a (2*n - m) < a n ^ 2 ∧ a n ^ 2 < a m * a (2*n - m + 1) :=
by sorry

end NUMINAMATH_CALUDE_no_geometric_progression_l2017_201798


namespace NUMINAMATH_CALUDE_flour_per_new_crust_l2017_201772

/-- Amount of flour per pie crust in cups -/
def flour_per_crust : ℚ := 1 / 8

/-- Number of pie crusts made daily -/
def daily_crusts : ℕ := 40

/-- Total flour used daily in cups -/
def total_flour : ℚ := daily_crusts * flour_per_crust

/-- Number of new pie crusts -/
def new_crusts : ℕ := 50

/-- Number of cakes -/
def cakes : ℕ := 10

/-- Flour used for cakes in cups -/
def cake_flour : ℚ := 1

/-- Theorem stating the amount of flour per new pie crust -/
theorem flour_per_new_crust : 
  (total_flour - cake_flour) / new_crusts = 2 / 25 := by sorry

end NUMINAMATH_CALUDE_flour_per_new_crust_l2017_201772


namespace NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l2017_201785

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a 1 - a 0

theorem middle_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_first : a 0 = 12) 
  (h_last : a 6 = 54) :
  a 3 = 33 := by
sorry

end NUMINAMATH_CALUDE_middle_term_of_arithmetic_sequence_l2017_201785


namespace NUMINAMATH_CALUDE_floor_paving_cost_l2017_201712

-- Define the room dimensions and cost per square meter
def room_length : ℝ := 5.5
def room_width : ℝ := 3.75
def cost_per_sq_meter : ℝ := 700

-- Define the function to calculate the total cost
def total_cost (length width cost_per_unit : ℝ) : ℝ :=
  length * width * cost_per_unit

-- Theorem statement
theorem floor_paving_cost :
  total_cost room_length room_width cost_per_sq_meter = 14437.50 := by
  sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l2017_201712


namespace NUMINAMATH_CALUDE_soccer_substitutions_mod_1000_l2017_201784

/-- Number of ways to make n substitutions -/
def num_substitutions (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => 11 * (13 - k) * num_substitutions k

/-- Total number of ways to make up to 4 substitutions -/
def total_substitutions : ℕ :=
  (List.range 5).map num_substitutions |> List.sum

theorem soccer_substitutions_mod_1000 :
  total_substitutions % 1000 = 25 := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_mod_1000_l2017_201784


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2017_201757

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a) 
  (h_a3 : a 3 = 5) 
  (h_a5 : a 5 = 3) : 
  a 8 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2017_201757


namespace NUMINAMATH_CALUDE_robin_gum_packages_l2017_201776

theorem robin_gum_packages :
  ∀ (packages : ℕ),
  (7 * packages + 6 = 41) →
  packages = 5 := by
sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l2017_201776


namespace NUMINAMATH_CALUDE_francies_allowance_l2017_201780

/-- Francie's allowance problem -/
theorem francies_allowance (x : ℚ) : 
  (∀ (total_saved half_spent remaining : ℚ),
    total_saved = 8 * x + 6 * 6 →
    half_spent = total_saved / 2 →
    remaining = half_spent - 35 →
    remaining = 3) →
  x = 5 := by sorry

end NUMINAMATH_CALUDE_francies_allowance_l2017_201780


namespace NUMINAMATH_CALUDE_cube_greater_than_one_iff_l2017_201748

theorem cube_greater_than_one_iff (x : ℝ) : x > 1 ↔ x^3 > 1 := by sorry

end NUMINAMATH_CALUDE_cube_greater_than_one_iff_l2017_201748


namespace NUMINAMATH_CALUDE_max_n_for_coprime_with_prime_l2017_201755

/-- A function that checks if a list of integers is pairwise coprime -/
def IsPairwiseCoprime (list : List Int) : Prop :=
  ∀ i j, i ≠ j → i ∈ list → j ∈ list → Int.gcd i j = 1

/-- A function that checks if a number is prime -/
def IsPrime (n : Int) : Prop :=
  n > 1 ∧ ∀ m, 1 < m → m < n → ¬(n % m = 0)

/-- The main theorem -/
theorem max_n_for_coprime_with_prime : 
  (∀ (list : List Int), list.length = 5 → 
    (∀ x ∈ list, x ≥ 1 ∧ x ≤ 48) → 
    IsPairwiseCoprime list → 
    (∃ x ∈ list, IsPrime x)) ∧ 
  (∃ (list : List Int), list.length = 5 ∧ 
    (∀ x ∈ list, x ≥ 1 ∧ x ≤ 49) ∧ 
    IsPairwiseCoprime list ∧ 
    (∀ x ∈ list, ¬IsPrime x)) := by
  sorry

end NUMINAMATH_CALUDE_max_n_for_coprime_with_prime_l2017_201755


namespace NUMINAMATH_CALUDE_trigonometric_inequality_equivalence_l2017_201766

theorem trigonometric_inequality_equivalence 
  (θ₁ θ₂ θ₃ θ₄ : ℝ) : 
  (∃ x : ℝ, (Real.cos θ₁)^2 * (Real.cos θ₂)^2 - (Real.sin θ₁ * Real.sin θ₂ - x)^2 ≥ 0 ∧ 
            (Real.cos θ₃)^2 * (Real.cos θ₄)^2 - (Real.sin θ₃ * Real.sin θ₄ - x)^2 ≥ 0) 
  ↔ 
  (Real.sin θ₁)^2 + (Real.sin θ₂)^2 + (Real.sin θ₃)^2 + (Real.sin θ₄)^2 ≤ 
    2 * (1 + Real.sin θ₁ * Real.sin θ₂ * Real.sin θ₃ * Real.sin θ₄ + 
         Real.cos θ₁ * Real.cos θ₂ * Real.cos θ₃ * Real.cos θ₄) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_inequality_equivalence_l2017_201766


namespace NUMINAMATH_CALUDE_smallest_b_in_geometric_sequence_l2017_201783

theorem smallest_b_in_geometric_sequence (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →  -- all terms are positive
  (∃ r : ℝ, 0 < r ∧ a * r = b ∧ b * r = c) →  -- geometric sequence condition
  a * b * c = 125 →  -- product condition
  ∀ x : ℝ, (0 < x ∧ ∃ y z : ℝ, 0 < y ∧ 0 < z ∧ 
    (∃ r : ℝ, 0 < r ∧ y * r = x ∧ x * r = z) ∧ 
    y * x * z = 125) → 
  5 ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_in_geometric_sequence_l2017_201783


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l2017_201758

theorem root_difference_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0) →
  (∃ r s : ℝ, r^2 - k*r + 10 = 0 ∧ s^2 - k*s + 10 = 0) →
  (∀ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0 →
              (r+3)^2 - k*(r+3) + 10 = 0 ∧ (s+3)^2 - k*(s+3) + 10 = 0) →
  k = 3 :=
by sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l2017_201758


namespace NUMINAMATH_CALUDE_square_diagonals_are_equal_l2017_201735

-- Define the basic shapes
class Square
class Parallelogram

-- Define the property of having equal diagonals
def has_equal_diagonals (α : Type*) := Prop

-- State the given conditions
axiom square_equal_diagonals : has_equal_diagonals Square
axiom parallelogram_equal_diagonals : has_equal_diagonals Parallelogram
axiom square_is_parallelogram : Square → Parallelogram

-- State the theorem to be proved
theorem square_diagonals_are_equal : has_equal_diagonals Square := by
  sorry

end NUMINAMATH_CALUDE_square_diagonals_are_equal_l2017_201735


namespace NUMINAMATH_CALUDE_anna_rearrangement_time_l2017_201751

def name : String := "Anna"
def letters : ℕ := 4
def repetitions : List ℕ := [2, 2]  -- 'A' repeated twice, 'N' repeated twice
def rearrangements_per_minute : ℕ := 8

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def total_rearrangements : ℕ :=
  factorial letters / (factorial repetitions[0]! * factorial repetitions[1]!)

def time_in_minutes : ℚ :=
  total_rearrangements / rearrangements_per_minute

theorem anna_rearrangement_time :
  time_in_minutes / 60 = 0.0125 := by sorry

end NUMINAMATH_CALUDE_anna_rearrangement_time_l2017_201751


namespace NUMINAMATH_CALUDE_average_length_of_strings_l2017_201721

def string1_length : ℝ := 2
def string2_length : ℝ := 6
def num_strings : ℕ := 2

theorem average_length_of_strings :
  (string1_length + string2_length) / num_strings = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_length_of_strings_l2017_201721


namespace NUMINAMATH_CALUDE_computers_per_month_l2017_201774

theorem computers_per_month 
  (production_rate : ℝ) 
  (interval : ℝ) 
  (days_per_month : ℕ) 
  (h1 : production_rate = 4.5)
  (h2 : interval = 0.5)
  (h3 : days_per_month = 28) :
  ⌊(production_rate / interval) * (days_per_month * 24)⌋ = 6048 := by
  sorry

end NUMINAMATH_CALUDE_computers_per_month_l2017_201774


namespace NUMINAMATH_CALUDE_pet_store_house_cats_l2017_201731

theorem pet_store_house_cats 
  (initial_siamese : ℕ)
  (sold : ℕ)
  (remaining : ℕ)
  (h1 : initial_siamese = 19)
  (h2 : sold = 56)
  (h3 : remaining = 8) :
  ∃ initial_house : ℕ, 
    initial_house = 45 ∧ 
    initial_siamese + initial_house = sold + remaining :=
by sorry

end NUMINAMATH_CALUDE_pet_store_house_cats_l2017_201731


namespace NUMINAMATH_CALUDE_parabola_vertex_l2017_201788

/-- The parabola defined by y = -(x+2)^2 + 6 has its vertex at (-2, 6) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -(x + 2)^2 + 6 → (∀ t : ℝ, y ≤ -(t + 2)^2 + 6) → (x = -2 ∧ y = 6) :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2017_201788


namespace NUMINAMATH_CALUDE_original_average_proof_l2017_201708

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℚ) : 
  n = 10 → new_avg = 160 → new_avg = 2 * original_avg → original_avg = 80 := by
  sorry

end NUMINAMATH_CALUDE_original_average_proof_l2017_201708


namespace NUMINAMATH_CALUDE_existence_of_nth_root_l2017_201790

theorem existence_of_nth_root (n b : ℕ) (h_n : n > 1) (h_b : b > 1)
  (h : ∀ k : ℕ, k > 1 → ∃ a_k : ℤ, (k : ℤ) ∣ (b - a_k ^ n)) :
  ∃ A : ℤ, (A : ℤ) ^ n = b :=
sorry

end NUMINAMATH_CALUDE_existence_of_nth_root_l2017_201790


namespace NUMINAMATH_CALUDE_contrapositive_ab_zero_extreme_values_count_l2017_201740

-- Proposition ②
theorem contrapositive_ab_zero (a b : ℝ) :
  (a * b = 0 → a = 0 ∨ b = 0) ↔ (a ≠ 0 ∧ b ≠ 0 → a * b ≠ 0) :=
sorry

-- Proposition ④
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2

theorem extreme_values_count :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
    (∀ (x : ℝ), f x ≤ f x₁ ∨ f x ≥ f x₁) ∧
    (∀ (x : ℝ), f x ≤ f x₂ ∨ f x ≥ f x₂) ∧
    (∀ (x : ℝ), x ≠ x₁ ∧ x ≠ x₂ →
      ¬(∀ (y : ℝ), f y ≤ f x ∨ f y ≥ f x)) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_ab_zero_extreme_values_count_l2017_201740


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2017_201724

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_inequality
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : arithmetic_sequence a d)
  (h2 : d ≠ 0)
  (h3 : ∀ n : ℕ, a n > 0) :
  a 1 * a 8 < a 4 * a 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l2017_201724


namespace NUMINAMATH_CALUDE_smallest_polygon_with_lighting_property_l2017_201718

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A point in the plane -/
def Point := ℝ × ℝ

/-- Predicate to check if a point is inside a polygon -/
def isInside (p : Point) (poly : Polygon n) : Prop := sorry

/-- Predicate to check if a point on a side of the polygon is lightened by a bulb -/
def isLightened (p : Point) (side : Fin n) (poly : Polygon n) (bulb : Point) : Prop := sorry

/-- Predicate to check if a polygon satisfies the lighting property -/
def hasLightingProperty (poly : Polygon n) : Prop :=
  ∃ bulb : Point, isInside bulb poly ∧
    ∀ side : Fin n, ∃ p : Point, ¬isLightened p side poly bulb

/-- Predicate to check if two bulbs light up the whole perimeter -/
def lightsWholePerimeter (poly : Polygon n) (bulb1 bulb2 : Point) : Prop :=
  ∀ side : Fin n, ∀ p : Point, isLightened p side poly bulb1 ∨ isLightened p side poly bulb2

theorem smallest_polygon_with_lighting_property :
  (∀ n < 6, ¬∃ poly : Polygon n, hasLightingProperty poly) ∧
  (∃ poly : Polygon 6, hasLightingProperty poly) ∧
  (∀ poly : Polygon 6, ∃ bulb1 bulb2 : Point, lightsWholePerimeter poly bulb1 bulb2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_polygon_with_lighting_property_l2017_201718


namespace NUMINAMATH_CALUDE_fraction_units_and_exceed_l2017_201742

def fraction_units (numerator denominator : ℕ) : ℕ := numerator

def units_to_exceed (start target : ℕ) : ℕ :=
  if start ≥ target then 1 else target - start + 1

theorem fraction_units_and_exceed :
  (fraction_units 5 8 = 5) ∧
  (units_to_exceed 5 8 = 4) := by sorry

end NUMINAMATH_CALUDE_fraction_units_and_exceed_l2017_201742


namespace NUMINAMATH_CALUDE_max_n_is_largest_l2017_201768

/-- Represents the sum of digits of a natural number -/
def S (a : ℕ) : ℕ := sorry

/-- Checks if all digits of a natural number are distinct -/
def has_distinct_digits (n : ℕ) : Prop := sorry

/-- The maximum natural number satisfying the given conditions -/
def max_n : ℕ := 3210

theorem max_n_is_largest :
  ∀ n : ℕ, 
  has_distinct_digits n → 
  S (3 * n) = 3 * S n → 
  n ≤ max_n :=
sorry

end NUMINAMATH_CALUDE_max_n_is_largest_l2017_201768


namespace NUMINAMATH_CALUDE_price_difference_chips_pretzels_l2017_201713

/-- The price difference between chips and pretzels -/
theorem price_difference_chips_pretzels :
  ∀ (pretzel_price chip_price : ℕ),
    pretzel_price = 4 →
    2 * chip_price + 2 * pretzel_price = 22 →
    chip_price > pretzel_price →
    chip_price - pretzel_price = 3 := by
  sorry

end NUMINAMATH_CALUDE_price_difference_chips_pretzels_l2017_201713


namespace NUMINAMATH_CALUDE_percentage_failed_both_subjects_l2017_201738

theorem percentage_failed_both_subjects 
  (failed_hindi : Real) 
  (failed_english : Real) 
  (passed_both : Real) 
  (h1 : failed_hindi = 32) 
  (h2 : failed_english = 56) 
  (h3 : passed_both = 24) : 
  Real := by
  sorry

end NUMINAMATH_CALUDE_percentage_failed_both_subjects_l2017_201738


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2017_201773

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l2017_201773


namespace NUMINAMATH_CALUDE_product_property_l2017_201730

theorem product_property : ∃ (n : ℕ), 10 ≤ n ∧ n ≤ 99 ∧ (∃ (k : ℤ), 4.02 * (n : ℝ) = (k : ℝ)) ∧ 10 * (4.02 * (n : ℝ)) = 2010 := by
  sorry

end NUMINAMATH_CALUDE_product_property_l2017_201730


namespace NUMINAMATH_CALUDE_circle_radius_is_five_l2017_201777

/-- Triangle ABC with vertices A(2,0), B(8,0), and C(5,5) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {⟨2, 0⟩, ⟨8, 0⟩, ⟨5, 5⟩}

/-- The circle circumscribing triangle ABC -/
def circumcircle : Set (ℝ × ℝ) :=
  sorry

/-- A square with side length 5 -/
def square_PQRS : Set (ℝ × ℝ) :=
  sorry

/-- The radius of the circumcircle -/
def radius (circle : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Two vertices of the square lie on the sides of the triangle -/
axiom square_vertices_on_triangle :
  ∃ (P Q : ℝ × ℝ), P ∈ square_PQRS ∧ Q ∈ square_PQRS ∧
    (P.1 - 2) / 3 = P.2 / 5 ∧ (Q.1 - 5) / 3 = -Q.2 / 5

/-- The other two vertices of the square lie on the circumcircle -/
axiom square_vertices_on_circle :
  ∃ (R S : ℝ × ℝ), R ∈ square_PQRS ∧ S ∈ square_PQRS ∧
    R ∈ circumcircle ∧ S ∈ circumcircle

/-- The side length of the square is 5 -/
axiom square_side_length :
  ∀ (X Y : ℝ × ℝ), X ∈ square_PQRS → Y ∈ square_PQRS →
    X ≠ Y → (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 25

theorem circle_radius_is_five :
  radius circumcircle = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_five_l2017_201777


namespace NUMINAMATH_CALUDE_rope_cutting_l2017_201789

theorem rope_cutting (total_length : ℝ) (ratio_short : ℝ) (ratio_long : ℝ) :
  total_length = 35 →
  ratio_short = 3 →
  ratio_long = 4 →
  (ratio_long / (ratio_short + ratio_long)) * total_length = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_l2017_201789
