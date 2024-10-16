import Mathlib

namespace NUMINAMATH_CALUDE_standard_deck_two_card_selections_l3917_391740

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (h_total : total_cards = suits * cards_per_suit)

/-- The number of ways to select two different cards from a deck, where order matters -/
def two_card_selections (d : Deck) : Nat :=
  d.total_cards * (d.total_cards - 1)

/-- Theorem: The number of ways to select two different cards from a standard deck of 52 cards, where order matters, is 2652 -/
theorem standard_deck_two_card_selections :
  ∃ (d : Deck), d.total_cards = 52 ∧ d.suits = 4 ∧ d.cards_per_suit = 13 ∧ two_card_selections d = 2652 := by
  sorry

end NUMINAMATH_CALUDE_standard_deck_two_card_selections_l3917_391740


namespace NUMINAMATH_CALUDE_game_price_is_correct_l3917_391700

/-- The price of each game Zachary sold -/
def game_price : ℝ := 5

/-- The number of games Zachary sold -/
def zachary_games : ℕ := 40

/-- The amount of money Zachary received -/
def zachary_amount : ℝ := game_price * zachary_games

/-- The amount of money Jason received -/
def jason_amount : ℝ := zachary_amount * 1.3

/-- The amount of money Ryan received -/
def ryan_amount : ℝ := jason_amount + 50

/-- The total amount received by all three friends -/
def total_amount : ℝ := 770

theorem game_price_is_correct : 
  zachary_amount + jason_amount + ryan_amount = total_amount := by sorry

end NUMINAMATH_CALUDE_game_price_is_correct_l3917_391700


namespace NUMINAMATH_CALUDE_triangle_ratio_proof_l3917_391764

theorem triangle_ratio_proof (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b^2 = a * c →
  a^2 + b * c = c^2 + a * c →
  c / (b * Real.sin B) = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_proof_l3917_391764


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3917_391785

/-- For a quadratic equation x^2 + 2(k-1)x + k^2 - 1 = 0, 
    the equation has real roots if and only if k ≤ 1 -/
theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*(k-1)*x + k^2 - 1 = 0) ↔ k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3917_391785


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3917_391748

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes
  (l m : Line) (α β : Plane)
  (h1 : perp_line_plane l α)
  (h2 : subset_line_plane m β)
  (h3 : parallel_planes α β) :
  perp_lines l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l3917_391748


namespace NUMINAMATH_CALUDE_similar_right_triangles_l3917_391732

theorem similar_right_triangles (y : ℝ) : 
  -- First triangle with legs 15 and 12
  let a₁ : ℝ := 15
  let b₁ : ℝ := 12
  -- Second triangle with legs y and 9
  let a₂ : ℝ := y
  let b₂ : ℝ := 9
  -- Triangles are similar (corresponding sides are proportional)
  a₁ / a₂ = b₁ / b₂ →
  -- The value of y is 11.25
  y = 11.25 := by
sorry

end NUMINAMATH_CALUDE_similar_right_triangles_l3917_391732


namespace NUMINAMATH_CALUDE_tom_coins_value_l3917_391797

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The number of quarters Tom found -/
def num_quarters : ℕ := 10

/-- The number of dimes Tom found -/
def num_dimes : ℕ := 3

/-- The number of nickels Tom found -/
def num_nickels : ℕ := 4

/-- The number of pennies Tom found -/
def num_pennies : ℕ := 200

/-- The total value of the coins Tom found in dollars -/
def total_value : ℚ := num_quarters * quarter_value + num_dimes * dime_value + 
                       num_nickels * nickel_value + num_pennies * penny_value

theorem tom_coins_value : total_value = 5 := by
  sorry

end NUMINAMATH_CALUDE_tom_coins_value_l3917_391797


namespace NUMINAMATH_CALUDE_planes_perp_to_line_are_parallel_line_in_plane_perp_to_other_plane_implies_planes_perp_l3917_391751

-- Define basic geometric objects
variable (P Q R : Plane) (L M : Line)

-- Define geometric relationships
def perpendicular (L : Line) (P : Plane) : Prop := sorry
def parallel (P Q : Plane) : Prop := sorry
def contains (P : Plane) (L : Line) : Prop := sorry

-- Theorem 1: Two planes perpendicular to the same line are parallel to each other
theorem planes_perp_to_line_are_parallel 
  (h1 : perpendicular L P) (h2 : perpendicular L Q) : parallel P Q := by sorry

-- Theorem 2: If a line within a plane is perpendicular to another plane, 
-- then these two planes are perpendicular to each other
theorem line_in_plane_perp_to_other_plane_implies_planes_perp 
  (h1 : contains P L) (h2 : perpendicular L Q) : perpendicular P Q := by sorry

end NUMINAMATH_CALUDE_planes_perp_to_line_are_parallel_line_in_plane_perp_to_other_plane_implies_planes_perp_l3917_391751


namespace NUMINAMATH_CALUDE_wage_difference_l3917_391796

/-- Represents the hourly wages at Joe's Steakhouse -/
structure JoesSteakhouseWages where
  manager : ℝ
  dishwasher : ℝ
  chef : ℝ
  manager_wage : manager = 8.50
  dishwasher_wage : dishwasher = manager / 2
  chef_wage : chef = dishwasher * 1.20

/-- The difference between a manager's hourly wage and a chef's hourly wage is $3.40 -/
theorem wage_difference (w : JoesSteakhouseWages) : w.manager - w.chef = 3.40 := by
  sorry

end NUMINAMATH_CALUDE_wage_difference_l3917_391796


namespace NUMINAMATH_CALUDE_concert_problem_l3917_391728

/-- Represents the number of songs sung by each friend -/
structure SongCount where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ
  laura : ℕ

/-- Conditions for the concert problem -/
def ConcertConditions (sc : SongCount) : Prop :=
  sc.hanna = 9 ∧
  sc.mary = 3 ∧
  sc.alina > sc.mary ∧ sc.alina < sc.hanna ∧
  sc.tina > sc.mary ∧ sc.tina < sc.hanna ∧
  sc.laura > sc.mary ∧ sc.laura < sc.hanna

/-- The total number of songs performed -/
def TotalSongs (sc : SongCount) : ℕ :=
  (sc.mary + sc.alina + sc.tina + sc.hanna + sc.laura) / 4

/-- Theorem stating that under the given conditions, the total number of songs is 9 -/
theorem concert_problem (sc : SongCount) :
  ConcertConditions sc → TotalSongs sc = 9 := by
  sorry


end NUMINAMATH_CALUDE_concert_problem_l3917_391728


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3917_391752

theorem fraction_equation_solution : 
  ∀ (A B : ℚ), 
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ 5 → 
    (B * x - 13) / (x^2 - 7*x + 10) = A / (x - 2) + 5 / (x - 5)) → 
  A + B = 31/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3917_391752


namespace NUMINAMATH_CALUDE_inequality_proof_l3917_391789

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3917_391789


namespace NUMINAMATH_CALUDE_cost_price_calculation_l3917_391783

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 600)
  (h2 : profit_percentage = 25) : 
  ∃ (cost_price : ℝ), 
    selling_price = cost_price * (1 + profit_percentage / 100) ∧ 
    cost_price = 480 :=
by
  sorry

#check cost_price_calculation

end NUMINAMATH_CALUDE_cost_price_calculation_l3917_391783


namespace NUMINAMATH_CALUDE_unique_composite_with_square_predecessor_divisors_l3917_391760

/-- A natural number is composite if it has a proper divisor greater than 1 -/
def IsComposite (n : ℕ) : Prop := ∃ d : ℕ, 1 < d ∧ d < n ∧ n % d = 0

/-- A natural number is a perfect square if it's equal to some integer squared -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

/-- Property: for every natural divisor d of n, d-1 is a perfect square -/
def HasSquarePredecessorDivisors (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 0 → n % d = 0 → IsPerfectSquare (d - 1)

theorem unique_composite_with_square_predecessor_divisors :
  ∃! n : ℕ, IsComposite n ∧ HasSquarePredecessorDivisors n ∧ n = 10 :=
sorry

end NUMINAMATH_CALUDE_unique_composite_with_square_predecessor_divisors_l3917_391760


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l3917_391763

/-- Represents a convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ

/-- Calculates the maximum number of intersections between two convex polygons -/
def max_intersections (P₁ P₂ : ConvexPolygon) (k : ℕ) : ℕ :=
  k * P₂.sides

/-- Theorem stating the maximum number of intersections between two convex polygons -/
theorem max_intersections_theorem 
  (P₁ P₂ : ConvexPolygon) 
  (k : ℕ) 
  (h₁ : P₁.sides ≤ P₂.sides) 
  (h₂ : k ≤ P₁.sides) : 
  max_intersections P₁ P₂ k = k * P₂.sides :=
by
  sorry

#check max_intersections_theorem

end NUMINAMATH_CALUDE_max_intersections_theorem_l3917_391763


namespace NUMINAMATH_CALUDE_intersection_points_concyclic_and_share_radical_axis_l3917_391726

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
def Point := ℝ × ℝ

/-- Given two circles and two lines, returns the new circle formed by the intersection of chords -/
def newCircle (C₁ C₂ : Circle) (L₁ L₂ : Line) : Circle :=
  sorry

/-- Checks if three circles share a common radical axis -/
def shareCommonRadicalAxis (C₁ C₂ C₃ : Circle) : Prop :=
  sorry

/-- Main theorem: The four intersection points of chords lie on a new circle that shares
    a common radical axis with the original two circles -/
theorem intersection_points_concyclic_and_share_radical_axis
  (C₁ C₂ : Circle) (L₁ L₂ : Line) :
  let C := newCircle C₁ C₂ L₁ L₂
  shareCommonRadicalAxis C₁ C₂ C :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_points_concyclic_and_share_radical_axis_l3917_391726


namespace NUMINAMATH_CALUDE_distance_between_4th_and_26th_red_lights_l3917_391799

/-- The distance in feet between two red lights in a repeating pattern -/
def distance_between_red_lights (n m : ℕ) : ℚ :=
  let inches_between_lights : ℕ := 4
  let pattern_length : ℕ := 5
  let inches_per_foot : ℕ := 12
  let position (k : ℕ) : ℕ := 1 + (k - 1) / 2 * pattern_length + 2 * ((k - 1) % 2)
  let gaps : ℕ := position m - position n
  (gaps * inches_between_lights : ℚ) / inches_per_foot

/-- The theorem stating the distance between the 4th and 26th red lights -/
theorem distance_between_4th_and_26th_red_lights :
  distance_between_red_lights 4 26 = 18.33 :=
sorry

end NUMINAMATH_CALUDE_distance_between_4th_and_26th_red_lights_l3917_391799


namespace NUMINAMATH_CALUDE_line_parametric_to_standard_l3917_391736

/-- Given a line with parametric equations x = -2 - 2t and y = 3 + √2 t,
    prove that its standard form is x + √2 y + 2 - 3√2 = 0 -/
theorem line_parametric_to_standard :
  ∀ (t x y : ℝ),
  (x = -2 - 2*t ∧ y = 3 + Real.sqrt 2 * t) →
  x + Real.sqrt 2 * y + 2 - 3 * Real.sqrt 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_parametric_to_standard_l3917_391736


namespace NUMINAMATH_CALUDE_monomial_replacement_l3917_391758

theorem monomial_replacement (x : ℝ) : 
  let expression := (x^4 - 3)^2 + (x^3 + 3*x)^2
  ∃ (a b c d : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    expression = a * x^n₁ + b * x^n₂ + c * x^n₃ + d * x^n₄ ∧
    n₁ > n₂ ∧ n₂ > n₃ ∧ n₃ > n₄ ∧
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_monomial_replacement_l3917_391758


namespace NUMINAMATH_CALUDE_mary_younger_than_albert_l3917_391753

/-- Proves that Mary is 10 years younger than Albert given the conditions -/
theorem mary_younger_than_albert (albert_age mary_age betty_age : ℕ) : 
  albert_age = 2 * mary_age →
  albert_age = 4 * betty_age →
  betty_age = 5 →
  albert_age - mary_age = 10 := by
sorry

end NUMINAMATH_CALUDE_mary_younger_than_albert_l3917_391753


namespace NUMINAMATH_CALUDE_scout_sale_profit_l3917_391762

/-- Represents the scout troop's candy bar sale scenario -/
structure CandyBarSale where
  total_bars : ℕ
  purchase_price : ℚ
  sold_bars : ℕ
  selling_price : ℚ

/-- Calculates the profit for the candy bar sale -/
def calculate_profit (sale : CandyBarSale) : ℚ :=
  sale.selling_price * sale.sold_bars - sale.purchase_price * sale.total_bars

/-- The specific candy bar sale scenario from the problem -/
def scout_sale : CandyBarSale :=
  { total_bars := 2000
  , purchase_price := 3 / 4
  , sold_bars := 1950
  , selling_price := 2 / 3 }

/-- Theorem stating that the profit for the scout troop's candy bar sale is -200 -/
theorem scout_sale_profit :
  calculate_profit scout_sale = -200 := by
  sorry


end NUMINAMATH_CALUDE_scout_sale_profit_l3917_391762


namespace NUMINAMATH_CALUDE_train_length_l3917_391738

/-- The length of a train given its speed, the speed of a man moving in the opposite direction, and the time it takes for the train to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (crossing_time : ℝ) : 
  train_speed = 25 →
  man_speed = 2 →
  crossing_time = 44 →
  (train_speed + man_speed) * crossing_time * (1000 / 3600) = 330 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l3917_391738


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3917_391723

/-- A rectangular solid with prime edge lengths and volume 231 has surface area 262 -/
theorem rectangular_solid_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 231 →
  2 * (a * b + b * c + a * c) = 262 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l3917_391723


namespace NUMINAMATH_CALUDE_karen_pickup_cases_l3917_391787

/-- The number of boxes Karen sold -/
def boxes_sold : ℕ := 36

/-- The number of boxes in each case -/
def boxes_per_case : ℕ := 12

/-- The number of cases Karen needs to pick up -/
def cases_to_pickup : ℕ := boxes_sold / boxes_per_case

theorem karen_pickup_cases : cases_to_pickup = 3 := by
  sorry

end NUMINAMATH_CALUDE_karen_pickup_cases_l3917_391787


namespace NUMINAMATH_CALUDE_quadratic_to_linear_inequality_l3917_391798

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := x^2 + a*x + b

-- Define the linear function
def g (a b : ℝ) (x : ℝ) := a*x + b

-- State the theorem
theorem quadratic_to_linear_inequality 
  (a b : ℝ) 
  (h : ∀ x : ℝ, f a b x > 0 ↔ x < -3 ∨ x > 1) :
  ∀ x : ℝ, g a b x < 0 ↔ x < 3/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_to_linear_inequality_l3917_391798


namespace NUMINAMATH_CALUDE_raviraj_cycled_20km_l3917_391770

/-- Represents a point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents Raviraj's cycling journey -/
def raviraj_journey (final_distance : ℝ) : Prop :=
  ∃ (home : Point) (last_turn : Point) (final : Point),
    -- Initial movements
    last_turn.x = home.x - 10 ∧
    last_turn.y = home.y ∧
    -- Final position
    final.x = last_turn.x - 20 ∧
    final.y = last_turn.y ∧
    -- Distance to home is 30 km
    (final.x - home.x)^2 + (final.y - home.y)^2 = final_distance^2

/-- The theorem stating that Raviraj cycled 20 km after the third turn -/
theorem raviraj_cycled_20km : raviraj_journey 30 → 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_raviraj_cycled_20km_l3917_391770


namespace NUMINAMATH_CALUDE_point_on_line_ratio_l3917_391722

/-- Given five points O, A, B, C, D on a straight line with specified distances,
    and a point P between B and C satisfying a ratio condition,
    prove that OP has the given value. -/
theorem point_on_line_ratio (a b c d k : ℝ) :
  let OA := a
  let OB := k * b
  let OC := c
  let OD := k * d
  ∀ P : ℝ, OB ≤ P ∧ P ≤ OC →
  (a - P) / (P - k * d) = k * (k * b - P) / (P - c) →
  P = (a * c + k * b * d) / (a + c - k * b + k * d - 1 + k) :=
by sorry

end NUMINAMATH_CALUDE_point_on_line_ratio_l3917_391722


namespace NUMINAMATH_CALUDE_sandwich_shop_period_length_l3917_391744

/-- Represents the Eat "N Go Mobile Sausage Sandwich Shop scenario -/
structure SandwichShop where
  jalapeno_strips_per_sandwich : ℕ
  minutes_per_sandwich : ℕ
  total_jalapeno_strips : ℕ

/-- Calculates the period length in minutes for a given SandwichShop scenario -/
def period_length (shop : SandwichShop) : ℕ :=
  (shop.total_jalapeno_strips / shop.jalapeno_strips_per_sandwich) * shop.minutes_per_sandwich

/-- Theorem stating that under the given conditions, the period length is 60 minutes -/
theorem sandwich_shop_period_length :
  ∀ (shop : SandwichShop),
    shop.jalapeno_strips_per_sandwich = 4 →
    shop.minutes_per_sandwich = 5 →
    shop.total_jalapeno_strips = 48 →
    period_length shop = 60 := by
  sorry


end NUMINAMATH_CALUDE_sandwich_shop_period_length_l3917_391744


namespace NUMINAMATH_CALUDE_cricketer_average_difference_is_13_l3917_391768

def cricketer_average_difference (runs_A runs_B : ℕ) (innings_A innings_B : ℕ) 
  (increase_A increase_B : ℚ) : ℚ :=
  let avg_A : ℚ := (runs_A : ℚ) / (innings_A : ℚ)
  let avg_B : ℚ := (runs_B : ℚ) / (innings_B : ℚ)
  (avg_B + increase_B) - (avg_A + increase_A)

theorem cricketer_average_difference_is_13 :
  cricketer_average_difference 125 145 20 18 5 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_difference_is_13_l3917_391768


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l3917_391781

-- Define the set M
def M : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

-- Define the set N
def N : Set ℝ := {x | x > 1}

-- Theorem statement
theorem intersection_equals_interval : {x : ℝ | 1 < x ∧ x ≤ 2} = M ∩ N := by sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l3917_391781


namespace NUMINAMATH_CALUDE_spherical_rotation_l3917_391746

/-- Given a point with rectangular coordinates (-3, 2, 5) and corresponding 
    spherical coordinates (r, θ, φ), the point with spherical coordinates 
    (r, θ, φ-π/2) has rectangular coordinates (-3, 2, -5). -/
theorem spherical_rotation (r θ φ : Real) : 
  r * Real.sin φ * Real.cos θ = -3 ∧ 
  r * Real.sin φ * Real.sin θ = 2 ∧ 
  r * Real.cos φ = 5 → 
  r * Real.sin (φ - π/2) * Real.cos θ = -3 ∧
  r * Real.sin (φ - π/2) * Real.sin θ = 2 ∧
  r * Real.cos (φ - π/2) = -5 := by
sorry

end NUMINAMATH_CALUDE_spherical_rotation_l3917_391746


namespace NUMINAMATH_CALUDE_power_sum_equals_w_minus_one_l3917_391737

theorem power_sum_equals_w_minus_one (w : ℂ) (hw : w^2 - w + 1 = 0) : 
  w^98 + w^99 + w^100 + w^101 + w^102 = w - 1 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_w_minus_one_l3917_391737


namespace NUMINAMATH_CALUDE_line_arrangement_count_l3917_391776

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of people in the line. -/
def totalPeople : ℕ := 7

/-- The number of people in the family that must stay together. -/
def familySize : ℕ := 3

/-- The number of individual entities to arrange (family counts as one entity). -/
def entities : ℕ := totalPeople - familySize + 1

/-- The number of ways to arrange the line of people with the family staying together. -/
def arrangements : ℕ := factorial entities * factorial familySize

theorem line_arrangement_count : arrangements = 720 := by sorry

end NUMINAMATH_CALUDE_line_arrangement_count_l3917_391776


namespace NUMINAMATH_CALUDE_lost_revenue_example_l3917_391793

/-- Represents a movie theater with its capacity, ticket price, and sold tickets. -/
structure MovieTheater where
  capacity : Nat
  ticketPrice : Nat
  soldTickets : Nat

/-- Calculates the lost revenue for a movie theater. -/
def lostRevenue (theater : MovieTheater) : Nat :=
  theater.capacity * theater.ticketPrice - theater.soldTickets * theater.ticketPrice

/-- Theorem: The lost revenue for the given theater is $208.00. -/
theorem lost_revenue_example :
  let theater : MovieTheater := {
    capacity := 50,
    ticketPrice := 8,
    soldTickets := 24
  }
  lostRevenue theater = 208 := by
  sorry


end NUMINAMATH_CALUDE_lost_revenue_example_l3917_391793


namespace NUMINAMATH_CALUDE_professors_seating_arrangements_l3917_391749

/-- Represents the seating arrangement problem with professors and students. -/
structure SeatingArrangement where
  totalChairs : Nat
  numStudents : Nat
  numProfessors : Nat
  professorsBetweenStudents : Bool

/-- Calculates the number of ways professors can choose their chairs. -/
def waysToChooseChairs (arrangement : SeatingArrangement) : Nat :=
  sorry

/-- Theorem stating that the number of ways to choose chairs is 24 for the given problem. -/
theorem professors_seating_arrangements
  (arrangement : SeatingArrangement)
  (h1 : arrangement.totalChairs = 11)
  (h2 : arrangement.numStudents = 7)
  (h3 : arrangement.numProfessors = 4)
  (h4 : arrangement.professorsBetweenStudents = true) :
  waysToChooseChairs arrangement = 24 := by
  sorry

end NUMINAMATH_CALUDE_professors_seating_arrangements_l3917_391749


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3917_391741

theorem fraction_subtraction (x : ℝ) : x * 8000 - (1 / 20) * (1 / 100) * 8000 = 796 → x = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3917_391741


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l3917_391727

theorem fraction_equality_solution : ∃! x : ℚ, (1 + x) / (5 + x) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l3917_391727


namespace NUMINAMATH_CALUDE_units_digit_27_45_l3917_391739

theorem units_digit_27_45 : (27 ^ 45) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_27_45_l3917_391739


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l3917_391712

noncomputable def f' (f'1 : ℝ) : ℝ → ℝ := fun x ↦ 2 * f'1 / x - 1

theorem f_derivative_at_one :
  ∃ f'1 : ℝ, (f' f'1) 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l3917_391712


namespace NUMINAMATH_CALUDE_intersection_range_l3917_391730

-- Define the curve
def curve (x y : ℝ) : Prop :=
  Real.sqrt (1 - (y - 1)^2) = abs x - 1

-- Define the line
def line (k x y : ℝ) : Prop :=
  k * x - y = 2

-- Define the intersection condition
def intersect_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    curve x₁ y₁ ∧ curve x₂ y₂ ∧
    line k x₁ y₁ ∧ line k x₂ y₂

-- Define the range of k
def k_range (k : ℝ) : Prop :=
  (k ≥ -2 ∧ k < -4/3) ∨ (k > 4/3 ∧ k ≤ 2)

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, intersect_at_two_points k ↔ k_range k :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l3917_391730


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l3917_391786

theorem concentric_circles_radii_difference 
  (r : ℝ) 
  (h : r > 0) 
  (R : ℝ) 
  (area_ratio : π * R^2 = 4 * (π * r^2)) : 
  R - r = r :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l3917_391786


namespace NUMINAMATH_CALUDE_stratified_sample_male_count_l3917_391747

/-- Represents a stratified sample from a population of students -/
structure StratifiedSample where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  sample_female : ℕ
  sample_male : ℕ

/-- Theorem stating that in a given stratified sample, the number of male students in the sample is 18 -/
theorem stratified_sample_male_count 
  (sample : StratifiedSample) 
  (h1 : sample.total_students = 680)
  (h2 : sample.male_students = 360)
  (h3 : sample.female_students = 320)
  (h4 : sample.sample_female = 16)
  (h5 : sample.female_students * sample.sample_male = sample.male_students * sample.sample_female) :
  sample.sample_male = 18 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_male_count_l3917_391747


namespace NUMINAMATH_CALUDE_pony_lesson_cost_l3917_391779

/-- The cost per lesson for Andrea's pony, given the following conditions:
  * Monthly pasture rent is $500
  * Daily food cost is $10
  * There are two lessons per week
  * Total annual expenditure on the pony is $15890
-/
theorem pony_lesson_cost : 
  let monthly_pasture_rent : ℕ := 500
  let daily_food_cost : ℕ := 10
  let lessons_per_week : ℕ := 2
  let total_annual_cost : ℕ := 15890
  let annual_pasture_cost : ℕ := monthly_pasture_rent * 12
  let annual_food_cost : ℕ := daily_food_cost * 365
  let annual_lessons : ℕ := lessons_per_week * 52
  let lesson_cost : ℕ := (total_annual_cost - (annual_pasture_cost + annual_food_cost)) / annual_lessons
  lesson_cost = 60 :=
by sorry

end NUMINAMATH_CALUDE_pony_lesson_cost_l3917_391779


namespace NUMINAMATH_CALUDE_multiply_special_polynomials_l3917_391791

theorem multiply_special_polynomials (x : ℝ) : 
  (x^4 + 16*x^2 + 256) * (x^2 - 16) = x^6 - 4096 := by
sorry

end NUMINAMATH_CALUDE_multiply_special_polynomials_l3917_391791


namespace NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l3917_391701

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem largest_prime_divisor_of_factorial_sum :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (factorial 12 + factorial 13) ∧
    ∀ (q : ℕ), Nat.Prime q → q ∣ (factorial 12 + factorial 13) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_of_factorial_sum_l3917_391701


namespace NUMINAMATH_CALUDE_nuts_cost_to_age_ratio_l3917_391771

/-- The ratio of the cost of a pack of nuts to Betty's age -/
theorem nuts_cost_to_age_ratio : 
  ∀ (doug_age betty_age : ℕ) (num_packs total_cost : ℕ),
  doug_age = 40 →
  doug_age + betty_age = 90 →
  num_packs = 20 →
  total_cost = 2000 →
  (total_cost / num_packs : ℚ) / betty_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_nuts_cost_to_age_ratio_l3917_391771


namespace NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l3917_391707

theorem three_digit_multiples_of_seven : 
  (Finset.filter (fun k => 100 ≤ 7 * k ∧ 7 * k ≤ 999) (Finset.range 1000)).card = 128 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l3917_391707


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l3917_391742

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sum of the first three terms of the binomial expansion -/
def first_three_sum (n : ℕ) : ℕ := binomial n 0 + binomial n 1 + binomial n 2

/-- The constant term in the expansion -/
def constant_term (n : ℕ) : ℤ := binomial n 4 * (-2)^4

/-- The coefficient with the largest absolute value in the expansion -/
def largest_coeff (n : ℕ) : ℤ := binomial n 8 * 2^8

theorem binomial_expansion_properties :
  ∃ n : ℕ, 
    first_three_sum n = 79 ∧ 
    constant_term n = 7920 ∧ 
    largest_coeff n = 126720 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l3917_391742


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3917_391790

/-- Given a parabola x^2 = (1/4)y, the distance between its focus and directrix is 1/8 -/
theorem parabola_focus_directrix_distance (x y : ℝ) :
  x^2 = (1/4) * y → (distance_focus_directrix : ℝ) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3917_391790


namespace NUMINAMATH_CALUDE_midpoint_sum_l3917_391721

/-- Given that C = (4, 3) is the midpoint of line segment AB, where A = (2, 7) and B = (x, y), prove that x + y = 5. -/
theorem midpoint_sum (x y : ℝ) : 
  (4 : ℝ) = (2 + x) / 2 → 
  (3 : ℝ) = (7 + y) / 2 → 
  x + y = 5 := by
sorry

end NUMINAMATH_CALUDE_midpoint_sum_l3917_391721


namespace NUMINAMATH_CALUDE_simplify_polynomial_l3917_391769

theorem simplify_polynomial (x : ℝ) : 
  3*x + 5 - 4*x^2 + 2*x - 7 + x^2 - 3*x + 8 = -3*x^2 + 2*x + 6 := by
sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l3917_391769


namespace NUMINAMATH_CALUDE_quarters_remaining_l3917_391772

/-- Calculates the number of quarters remaining after paying for a dress -/
theorem quarters_remaining (initial_quarters : ℕ) (dress_cost : ℚ) (quarter_value : ℚ) : 
  initial_quarters = 160 → 
  dress_cost = 35 → 
  quarter_value = 1/4 → 
  initial_quarters - (dress_cost / quarter_value).floor = 20 := by
sorry

end NUMINAMATH_CALUDE_quarters_remaining_l3917_391772


namespace NUMINAMATH_CALUDE_inequality_proof_l3917_391735

theorem inequality_proof (a b x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3917_391735


namespace NUMINAMATH_CALUDE_orange_mango_difference_l3917_391773

/-- Represents the total produce in kilograms for each fruit type -/
structure FruitProduce where
  mangoes : ℕ
  apples : ℕ
  oranges : ℕ

/-- Calculates the total revenue given the price per kg and total produce -/
def totalRevenue (price : ℕ) (produce : FruitProduce) : ℕ :=
  price * (produce.mangoes + produce.apples + produce.oranges)

/-- Theorem stating the difference between orange and mango produce -/
theorem orange_mango_difference (produce : FruitProduce) : 
  produce.mangoes = 400 →
  produce.apples = 2 * produce.mangoes →
  produce.oranges > produce.mangoes →
  totalRevenue 50 produce = 90000 →
  produce.oranges - produce.mangoes = 200 := by
sorry

end NUMINAMATH_CALUDE_orange_mango_difference_l3917_391773


namespace NUMINAMATH_CALUDE_no_real_solutions_for_equation_l3917_391761

theorem no_real_solutions_for_equation : ¬∃ x : ℝ, x + Real.sqrt (x - 2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_equation_l3917_391761


namespace NUMINAMATH_CALUDE_circle_center_transformation_l3917_391710

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Translates a point up by a given amount -/
def translate_up (p : ℝ × ℝ) (amount : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + amount)

/-- The final position of the center of circle S after transformations -/
def final_position (initial : ℝ × ℝ) : ℝ × ℝ :=
  translate_up (reflect_x (reflect_y initial)) 5

theorem circle_center_transformation :
  final_position (3, -4) = (-3, 9) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_transformation_l3917_391710


namespace NUMINAMATH_CALUDE_exists_satisfying_quadratic_l3917_391795

/-- A quadratic function satisfying the given conditions -/
def satisfying_quadratic (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧
  (∀ x, |x| ≤ 1 → |f x| ≤ 1) ∧
  (|f 2| ≥ 7)

/-- There exists a quadratic function satisfying the given conditions -/
theorem exists_satisfying_quadratic : ∃ f : ℝ → ℝ, satisfying_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_exists_satisfying_quadratic_l3917_391795


namespace NUMINAMATH_CALUDE_vegetables_sold_mass_l3917_391711

/-- Proves that given 15 kg of carrots, 13 kg of zucchini, and 8 kg of broccoli,
    if a merchant sells half of the total vegetables, the mass of vegetables sold is 18 kg. -/
theorem vegetables_sold_mass 
  (carrots : ℝ) 
  (zucchini : ℝ) 
  (broccoli : ℝ) 
  (h1 : carrots = 15)
  (h2 : zucchini = 13)
  (h3 : broccoli = 8) :
  (carrots + zucchini + broccoli) / 2 = 18 := by
  sorry

#check vegetables_sold_mass

end NUMINAMATH_CALUDE_vegetables_sold_mass_l3917_391711


namespace NUMINAMATH_CALUDE_clock_notes_in_week_total_notes_in_week_l3917_391731

/-- Represents the ringing pattern for a single hour -/
structure HourPattern where
  quarter_past : Nat
  half_past : Nat
  quarter_to : Nat
  on_hour : Nat → Nat

/-- Represents the ringing pattern for a 12-hour period (day or night) -/
structure PeriodPattern where
  pattern : HourPattern
  on_hour_even : Nat → Nat
  on_hour_odd : Nat → Nat

def day_pattern : PeriodPattern :=
  { pattern := 
    { quarter_past := 2
      half_past := 4
      quarter_to := 6
      on_hour := λ h => 8
    }
    on_hour_even := λ h => h
    on_hour_odd := λ h => h / 2
  }

def night_pattern : PeriodPattern :=
  { pattern := 
    { quarter_past := 3
      half_past := 5
      quarter_to := 7
      on_hour := λ h => 9
    }
    on_hour_even := λ h => h / 2
    on_hour_odd := λ h => h
  }

def count_notes_for_period (pattern : PeriodPattern) : Nat :=
  12 * (pattern.pattern.quarter_past + pattern.pattern.half_past + pattern.pattern.quarter_to) +
  (pattern.pattern.on_hour 6 + pattern.on_hour_even 6 +
   pattern.pattern.on_hour 8 + pattern.on_hour_even 8 +
   pattern.pattern.on_hour 10 + pattern.on_hour_even 10 +
   pattern.pattern.on_hour 12 + pattern.on_hour_even 12 +
   pattern.pattern.on_hour 2 + pattern.on_hour_even 2 +
   pattern.pattern.on_hour 4 + pattern.on_hour_even 4 +
   pattern.pattern.on_hour 7 + pattern.on_hour_odd 7 +
   pattern.pattern.on_hour 9 + pattern.on_hour_odd 9 +
   pattern.pattern.on_hour 11 + pattern.on_hour_odd 11 +
   pattern.pattern.on_hour 1 + pattern.on_hour_odd 1 +
   pattern.pattern.on_hour 3 + pattern.on_hour_odd 3 +
   pattern.pattern.on_hour 5 + pattern.on_hour_odd 5)

theorem clock_notes_in_week :
  count_notes_for_period day_pattern + count_notes_for_period night_pattern = 471 ∧
  471 * 7 = 3297 := by sorry

theorem total_notes_in_week : (count_notes_for_period day_pattern + count_notes_for_period night_pattern) * 7 = 3297 := by sorry

end NUMINAMATH_CALUDE_clock_notes_in_week_total_notes_in_week_l3917_391731


namespace NUMINAMATH_CALUDE_symmetric_points_m_value_l3917_391724

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to the y-axis
def symmetricAboutYAxis (p1 p2 : Point2D) : Prop :=
  p1.x = -p2.x ∧ p1.y = p2.y

-- Theorem statement
theorem symmetric_points_m_value :
  let A : Point2D := ⟨-3, 4⟩
  let B : Point2D := ⟨3, m⟩
  symmetricAboutYAxis A B → m = 4 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_m_value_l3917_391724


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_20pi_l3917_391729

/-- Represents a triangular pyramid with vertex P and base ABC. -/
structure TriangularPyramid where
  PA : ℝ
  AB : ℝ
  angleCBA : ℝ
  perpendicular : Bool

/-- Calculates the surface area of the circumscribed sphere of a triangular pyramid. -/
def circumscribedSphereSurfaceArea (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The surface area of the circumscribed sphere of the given triangular pyramid is 20π. -/
theorem circumscribed_sphere_surface_area_is_20pi :
  let pyramid := TriangularPyramid.mk 2 2 (π/6) true
  circumscribedSphereSurfaceArea pyramid = 20 * π :=
by sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_is_20pi_l3917_391729


namespace NUMINAMATH_CALUDE_tan_sum_reciprocal_l3917_391713

theorem tan_sum_reciprocal (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_reciprocal_l3917_391713


namespace NUMINAMATH_CALUDE_smallest_binary_multiple_of_225_l3917_391734

def is_binary_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_225 :
  (∀ m : ℕ, m < 111111100 → ¬(225 ∣ m ∧ is_binary_number m)) ∧
  (225 ∣ 111111100 ∧ is_binary_number 111111100) :=
sorry

end NUMINAMATH_CALUDE_smallest_binary_multiple_of_225_l3917_391734


namespace NUMINAMATH_CALUDE_shadow_boundary_is_constant_l3917_391775

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The xy-plane -/
def xyPlane : Set Point3D := {p : Point3D | p.z = 0}

/-- Light source position -/
def lightSource : Point3D := ⟨0, -4, 3⟩

/-- The sphere in the problem -/
def problemSphere : Sphere := ⟨⟨0, 0, 2⟩, 2⟩

/-- A point on the boundary of the shadow -/
structure ShadowBoundaryPoint where
  x : ℝ
  y : ℝ

/-- The boundary function of the shadow -/
def shadowBoundary (p : ShadowBoundaryPoint) : Prop :=
  p.y = -19/4

theorem shadow_boundary_is_constant (s : Sphere) (l : Point3D) :
  s = problemSphere →
  l = lightSource →
  ∀ p : ShadowBoundaryPoint, shadowBoundary p := by
  sorry

#check shadow_boundary_is_constant

end NUMINAMATH_CALUDE_shadow_boundary_is_constant_l3917_391775


namespace NUMINAMATH_CALUDE_positive_real_solution_l3917_391716

theorem positive_real_solution (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a^b = b^a) (h4 : b = 4*a) : a = Real.rpow 4 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_solution_l3917_391716


namespace NUMINAMATH_CALUDE_point_C_coordinates_l3917_391719

-- Define the points and vectors
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)

-- Define vector AB
def vecAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector AC in terms of AB
def vecAC : ℝ × ℝ := (2 * vecAB.1, 2 * vecAB.2)

-- Define point C
def C : ℝ × ℝ := (A.1 + vecAC.1, A.2 + vecAC.2)

-- Theorem statement
theorem point_C_coordinates : C = (-3, 9) := by
  sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l3917_391719


namespace NUMINAMATH_CALUDE_circle_in_circle_theorem_l3917_391766

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

-- Define a point
def Point := ℝ × ℝ

-- Define what it means for a point to be inside a circle
def is_inside (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 < c.radius^2

-- Define what it means for a point to be on a circle
def is_on (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define what it means for one circle to be contained in another
def is_contained (c1 c2 : Circle) : Prop :=
  ∀ (p : Point), is_on p c1 → is_inside p c2

-- State the theorem
theorem circle_in_circle_theorem (ω : Circle) (A B : Point) 
  (h1 : is_inside A ω) (h2 : is_inside B ω) : 
  ∃ (ω' : Circle), is_on A ω' ∧ is_on B ω' ∧ is_contained ω' ω := by
  sorry

end NUMINAMATH_CALUDE_circle_in_circle_theorem_l3917_391766


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l3917_391725

/-- A figure composed of a square and two equilateral triangles -/
structure SquareTriangleFigure where
  square_side : ℝ
  triangle_side : ℝ

/-- The perimeter of the SquareTriangleFigure -/
def perimeter (fig : SquareTriangleFigure) : ℝ :=
  2 * fig.square_side + 2 * fig.triangle_side

/-- Theorem stating that the perimeter of the specific figure is 10 units -/
theorem specific_figure_perimeter :
  let fig : SquareTriangleFigure := ⟨3, 2⟩
  perimeter fig = 10 := by sorry

end NUMINAMATH_CALUDE_specific_figure_perimeter_l3917_391725


namespace NUMINAMATH_CALUDE_glass_bowl_selling_price_l3917_391743

theorem glass_bowl_selling_price
  (total_bowls : ℕ)
  (cost_per_bowl : ℚ)
  (sold_bowls : ℕ)
  (percentage_gain : ℚ)
  (h1 : total_bowls = 115)
  (h2 : cost_per_bowl = 18)
  (h3 : sold_bowls = 104)
  (h4 : percentage_gain = 0.004830917874396135)
  : ∃ (selling_price : ℚ), selling_price = 20 ∧ 
    selling_price * sold_bowls = cost_per_bowl * total_bowls * (1 + percentage_gain) :=
by sorry

end NUMINAMATH_CALUDE_glass_bowl_selling_price_l3917_391743


namespace NUMINAMATH_CALUDE_scientific_notation_of_small_number_l3917_391703

theorem scientific_notation_of_small_number :
  ∃ (a : ℝ) (n : ℤ), 0.0000003 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3 ∧ n = -7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_small_number_l3917_391703


namespace NUMINAMATH_CALUDE_inverse_proportion_relationship_l3917_391705

theorem inverse_proportion_relationship (k x₁ x₂ y₁ y₂ : ℝ) :
  k ≠ 0 →
  x₁ < 0 →
  0 < x₂ →
  y₁ = k / x₁ →
  y₂ = k / x₂ →
  k < 0 →
  y₂ < 0 ∧ 0 < y₁ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_relationship_l3917_391705


namespace NUMINAMATH_CALUDE_portfolio_calculations_l3917_391714

/-- Represents a stock with its yield and quote -/
structure Stock where
  yield : ℝ
  quote : ℝ

/-- Calculates the weighted average yield of a portfolio -/
def weightedAverageYield (stocks : List Stock) (proportions : List ℝ) : ℝ :=
  sorry

/-- Calculates the overall quote of a portfolio -/
def overallQuote (stocks : List Stock) (proportions : List ℝ) (totalInvestment : ℝ) : ℝ :=
  sorry

/-- Theorem stating that weighted average yield and overall quote can be calculated -/
theorem portfolio_calculations 
  (stocks : List Stock) 
  (proportions : List ℝ) 
  (totalInvestment : ℝ) 
  (h1 : stocks.length = 3)
  (h2 : proportions.length = 3)
  (h3 : proportions.sum = 1)
  (h4 : totalInvestment > 0) :
  ∃ (avgYield overallQ : ℝ), 
    avgYield = weightedAverageYield stocks proportions ∧ 
    overallQ = overallQuote stocks proportions totalInvestment :=
  sorry

end NUMINAMATH_CALUDE_portfolio_calculations_l3917_391714


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3917_391754

theorem quadratic_coefficient (b : ℝ) (n : ℝ) : 
  (∀ x, x^2 + b*x + 56 = (x + n)^2 + 12) → 
  b > 0 → 
  b = 4 * Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3917_391754


namespace NUMINAMATH_CALUDE_max_candies_eaten_l3917_391708

theorem max_candies_eaten (n : Nat) (h : n = 46) : 
  (n * (n - 1)) / 2 = 1035 := by
  sorry

#check max_candies_eaten

end NUMINAMATH_CALUDE_max_candies_eaten_l3917_391708


namespace NUMINAMATH_CALUDE_baker_cakes_theorem_l3917_391720

/-- The number of cakes sold by the baker -/
def cakes_sold : ℕ := 145

/-- The number of cakes left after selling -/
def cakes_left : ℕ := 72

/-- The total number of cakes made by the baker -/
def total_cakes : ℕ := cakes_sold + cakes_left

theorem baker_cakes_theorem : total_cakes = 217 := by
  sorry

end NUMINAMATH_CALUDE_baker_cakes_theorem_l3917_391720


namespace NUMINAMATH_CALUDE_find_M_and_N_l3917_391782

theorem find_M_and_N :
  ∀ M N : ℕ,
  0 < M ∧ M < 10 ∧ 0 < N ∧ N < 10 →
  8 * 10^7 + M * 10^6 + 420852 * 9 = N * 10^7 + 9889788 * 11 →
  M = 5 ∧ N = 6 := by
sorry

end NUMINAMATH_CALUDE_find_M_and_N_l3917_391782


namespace NUMINAMATH_CALUDE_budgets_equal_in_1996_l3917_391759

/-- Represents the year when the budgets of projects Q and V are equal -/
def year_budgets_equal (initial_q initial_v increase_q decrease_v : ℕ) : ℕ :=
  let n := (initial_v - initial_q) / (increase_q + decrease_v)
  1990 + n

/-- Theorem stating that the budgets of projects Q and V are equal in 1996 -/
theorem budgets_equal_in_1996 :
  year_budgets_equal 540000 780000 30000 10000 = 1996 := by
  sorry

#eval year_budgets_equal 540000 780000 30000 10000

end NUMINAMATH_CALUDE_budgets_equal_in_1996_l3917_391759


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l3917_391794

theorem popped_kernel_probability (total_kernels : ℝ) (h_total_positive : 0 < total_kernels) : 
  let white_kernels := (3 / 5) * total_kernels
  let yellow_kernels := (2 / 5) * total_kernels
  let popped_white := (2 / 5) * white_kernels
  let popped_yellow := (4 / 5) * yellow_kernels
  let total_popped := popped_white + popped_yellow
  (popped_white / total_popped) = (3 / 7) :=
by sorry

end NUMINAMATH_CALUDE_popped_kernel_probability_l3917_391794


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l3917_391788

theorem wire_cutting_problem (piece_length : ℕ) : 
  piece_length > 0 ∧
  9 * piece_length ≤ 1000 ∧
  9 * piece_length ≤ 1100 ∧
  10 * piece_length > 1100 →
  piece_length = 111 :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l3917_391788


namespace NUMINAMATH_CALUDE_horse_food_per_day_l3917_391706

/-- Given the ratio of sheep to horses, the number of sheep, and the total amount of horse food,
    calculate the amount of food per horse. -/
theorem horse_food_per_day (sheep_ratio : ℕ) (horse_ratio : ℕ) (num_sheep : ℕ) (total_food : ℕ) :
  sheep_ratio = 5 →
  horse_ratio = 7 →
  num_sheep = 40 →
  total_food = 12880 →
  (total_food / (horse_ratio * num_sheep / sheep_ratio) : ℚ) = 230 := by
  sorry

end NUMINAMATH_CALUDE_horse_food_per_day_l3917_391706


namespace NUMINAMATH_CALUDE_sequence_properties_l3917_391702

def is_arithmetic_progression (s : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, s (n + 1) - s n = d

theorem sequence_properties
  (a b c : ℕ+ → ℝ)
  (h1 : ∀ n : ℕ+, b n = a n - 2 * a (n + 1))
  (h2 : ∀ n : ℕ+, c n = a (n + 1) + 2 * a (n + 2) - 2) :
  (is_arithmetic_progression a → is_arithmetic_progression b) ∧
  (is_arithmetic_progression b ∧ is_arithmetic_progression c →
    ∃ d : ℝ, ∀ n : ℕ+, n ≥ 2 → a (n + 1) - a n = d) ∧
  (is_arithmetic_progression b ∧ b 1 + a 3 = 0 → is_arithmetic_progression a) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l3917_391702


namespace NUMINAMATH_CALUDE_article_price_reduction_l3917_391767

theorem article_price_reduction (reduced_price : ℝ) (reduction_percentage : ℝ) (original_price : ℝ) : 
  reduced_price = 608 ∧ 
  reduction_percentage = 24 ∧ 
  reduced_price = original_price * (1 - reduction_percentage / 100) → 
  original_price = 800 := by
  sorry

end NUMINAMATH_CALUDE_article_price_reduction_l3917_391767


namespace NUMINAMATH_CALUDE_linear_function_composition_l3917_391717

/-- Given two functions f and g, where f is a linear function with real coefficients a and b,
    and g is defined as g(x) = 3x - 4, prove that a + b = 11/3 if g(f(x)) = 4x + 3 for all x. -/
theorem linear_function_composition (a b : ℝ) :
  (∀ x, (3 * ((a * x + b) : ℝ) - 4) = 4 * x + 3) →
  a + b = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_composition_l3917_391717


namespace NUMINAMATH_CALUDE_simplify_sqrt_240_l3917_391745

theorem simplify_sqrt_240 : Real.sqrt 240 = 4 * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_240_l3917_391745


namespace NUMINAMATH_CALUDE_inequality_equivalence_inequality_positive_reals_l3917_391780

-- Problem 1
theorem inequality_equivalence (x : ℝ) : (x + 2) / (2 - 3 * x) > 1 ↔ 0 < x ∧ x < 2 / 3 := by sorry

-- Problem 2
theorem inequality_positive_reals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 ≥ a*b + b*c + a*c := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_inequality_positive_reals_l3917_391780


namespace NUMINAMATH_CALUDE_actors_in_one_hour_show_l3917_391757

/-- Calculates the number of actors in a show given the show duration, performance time per set, and number of actors per set. -/
def actors_in_show (show_duration : ℕ) (performance_time : ℕ) (actors_per_set : ℕ) : ℕ :=
  (show_duration / performance_time) * actors_per_set

/-- Proves that given the specified conditions, the number of actors in a 1-hour show is 20. -/
theorem actors_in_one_hour_show :
  actors_in_show 60 15 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_actors_in_one_hour_show_l3917_391757


namespace NUMINAMATH_CALUDE_x_over_y_value_l3917_391792

theorem x_over_y_value (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x + y)^2019 + x^2019 + 30*x + 5*y = 0) : 
  x / y = -1 / 6 := by
sorry

end NUMINAMATH_CALUDE_x_over_y_value_l3917_391792


namespace NUMINAMATH_CALUDE_vertical_line_not_conic_section_l3917_391755

/-- The equation |y-3| = √((x+4)² + (y-3)²) describes a vertical line x = -4 -/
theorem vertical_line_not_conic_section :
  ∀ x y : ℝ, |y - 3| = Real.sqrt ((x + 4)^2 + (y - 3)^2) ↔ x = -4 :=
by sorry

end NUMINAMATH_CALUDE_vertical_line_not_conic_section_l3917_391755


namespace NUMINAMATH_CALUDE_infinite_divisibility_equivalence_l3917_391774

theorem infinite_divisibility_equivalence :
  ∀ (a b c : ℕ+),
  (∃ (S : Set ℕ+), Set.Infinite S ∧ ∀ (n : ℕ+), n ∈ S → (a + n) ∣ (b + c * n!)) ↔
  (∃ (k : ℕ) (t : ℤ), a = 2 * k + 1 ∧ b = t.natAbs ∧ c = (t.natAbs * (2 * k).factorial)) :=
by sorry

end NUMINAMATH_CALUDE_infinite_divisibility_equivalence_l3917_391774


namespace NUMINAMATH_CALUDE_distance_between_cities_l3917_391778

/-- The distance between two cities given specific travel conditions -/
theorem distance_between_cities (cara_speed dan_min_speed : ℝ) 
  (dan_delay : ℝ) (h1 : cara_speed = 30) (h2 : dan_min_speed = 36) 
  (h3 : dan_delay = 1) : 
  ∃ D : ℝ, D = 180 ∧ D / dan_min_speed = D / cara_speed - dan_delay := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l3917_391778


namespace NUMINAMATH_CALUDE_m_range_l3917_391784

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/2 + y^2/m = 1 ∧ m > 2

def q (m : ℝ) : Prop := ∃ (x y : ℝ), (m+4)*x^2 - (m+2)*y^2 = (m+4)*(m+2)

-- Define the range of m
def range_m (m : ℝ) : Prop := m < -4 ∨ (-2 < m ∧ m ≤ 2)

-- State the theorem
theorem m_range : 
  (∀ m : ℝ, ¬(p m ∧ q m)) → 
  (∀ m : ℝ, p m → q m) → 
  (∀ m : ℝ, range_m m ↔ (¬(p m) ∧ q m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3917_391784


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3917_391715

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (2, x + 2)
  are_parallel a b → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3917_391715


namespace NUMINAMATH_CALUDE_evaluate_expression_l3917_391718

theorem evaluate_expression :
  -(16 / 4 * 11 - 70 + 5 * 11) = -29 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3917_391718


namespace NUMINAMATH_CALUDE_roots_properties_l3917_391756

theorem roots_properties (a b : ℝ) 
  (h1 : a^2 - 6*a + 4 = 0) 
  (h2 : b^2 - 6*b + 4 = 0) 
  (h3 : a > b) : 
  (a > 0 ∧ b > 0) ∧ 
  (((Real.sqrt a - Real.sqrt b) / (Real.sqrt a + Real.sqrt b)) = Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_roots_properties_l3917_391756


namespace NUMINAMATH_CALUDE_lee_cookies_with_five_cups_l3917_391750

/-- Given that Lee can make 24 cookies with 3 cups of flour,
    this function calculates how many cookies he can make with any number of cups. -/
def cookies_per_cups (cups : ℚ) : ℚ :=
  (24 / 3) * cups

/-- Theorem stating that Lee can make 40 cookies with 5 cups of flour. -/
theorem lee_cookies_with_five_cups :
  cookies_per_cups 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_lee_cookies_with_five_cups_l3917_391750


namespace NUMINAMATH_CALUDE_highway_length_l3917_391765

/-- The length of a highway where two cars meet --/
theorem highway_length (v1 v2 t : ℝ) (h1 : v1 = 25) (h2 : v2 = 45) (h3 : t = 2.5) :
  (v1 + v2) * t = 175 := by
  sorry

end NUMINAMATH_CALUDE_highway_length_l3917_391765


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3917_391709

theorem sum_of_roots_quadratic (x : ℝ) :
  x^2 - 6*x + 8 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = 6 ∧ x = r₁ ∨ x = r₂ :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3917_391709


namespace NUMINAMATH_CALUDE_min_value_a_l3917_391733

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 3 - x / Real.exp x

theorem min_value_a :
  (∃ (a : ℝ), ∀ (x : ℝ), x ≥ -2 → f x ≤ a) ∧ 
  (∀ (b : ℝ), (∃ (x : ℝ), x ≥ -2 ∧ f x > b) → b < 1 - 1 / Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l3917_391733


namespace NUMINAMATH_CALUDE_parallelogram_product_l3917_391777

-- Define the parallelogram EFGH
def EFGH (EF FG GH HE : ℝ) : Prop :=
  EF = GH ∧ FG = HE

-- Theorem statement
theorem parallelogram_product (x y : ℝ) :
  EFGH 47 (6 * y^2) (3 * x + 7) 27 →
  x * y = 20 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_product_l3917_391777


namespace NUMINAMATH_CALUDE_ski_price_calculation_l3917_391704

theorem ski_price_calculation (initial_price : ℝ) 
  (morning_discount : ℝ) (noon_increase : ℝ) (afternoon_discount : ℝ) : 
  initial_price = 200 →
  morning_discount = 0.4 →
  noon_increase = 0.25 →
  afternoon_discount = 0.2 →
  (initial_price * (1 - morning_discount) * (1 + noon_increase) * (1 - afternoon_discount)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_ski_price_calculation_l3917_391704
