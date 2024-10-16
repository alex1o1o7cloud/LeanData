import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_product_l1918_191866

theorem units_digit_product (n : ℕ) : (2^2023 * 5^2024 * 11^2025) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_product_l1918_191866


namespace NUMINAMATH_CALUDE_power_of_product_with_negative_l1918_191861

theorem power_of_product_with_negative (a b : ℝ) : (-a * b^2)^3 = -a^3 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_with_negative_l1918_191861


namespace NUMINAMATH_CALUDE_choose_four_from_fifteen_l1918_191849

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : n = 15 ∧ k = 4 → Nat.choose n k = 1365 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_fifteen_l1918_191849


namespace NUMINAMATH_CALUDE_tangent_slope_at_negative_two_l1918_191805

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- Define the point of interest
def point : ℝ × ℝ := (-2, -8)

-- State the theorem
theorem tangent_slope_at_negative_two :
  (deriv f) point.1 = 12 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_negative_two_l1918_191805


namespace NUMINAMATH_CALUDE_min_points_to_win_correct_l1918_191844

/-- Represents a chess tournament with 6 players where each player plays 2 games against every other player. -/
structure ChessTournament where
  num_players : ℕ
  games_per_pair : ℕ
  win_points : ℚ
  draw_points : ℚ
  loss_points : ℚ

/-- The minimum number of points needed to guarantee a player has more points than any other player -/
def min_points_to_win (t : ChessTournament) : ℚ := 9.5

/-- Theorem stating that 9.5 points is the minimum required to guarantee winning the tournament -/
theorem min_points_to_win_correct (t : ChessTournament) 
  (h1 : t.num_players = 6)
  (h2 : t.games_per_pair = 2)
  (h3 : t.win_points = 1)
  (h4 : t.draw_points = 0.5)
  (h5 : t.loss_points = 0) :
  ∀ (p : ℚ), p < min_points_to_win t → 
  ∃ (other_player_points : ℚ), other_player_points ≥ p ∧ other_player_points ≤ (t.num_players - 1) * t.games_per_pair * t.win_points :=
sorry

end NUMINAMATH_CALUDE_min_points_to_win_correct_l1918_191844


namespace NUMINAMATH_CALUDE_expression_simplification_l1918_191852

theorem expression_simplification (α : Real) (h : π < α ∧ α < (3*π)/2) :
  Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) + Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) = -2 / Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1918_191852


namespace NUMINAMATH_CALUDE_toms_age_ratio_l1918_191894

/-- Tom's age problem -/
theorem toms_age_ratio (T N : ℚ) : T > 0 → N > 0 →
  (∃ (a b c d : ℚ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ T = a + b + c + d) →
  (T - N = 3 * (T - 4 * N)) →
  T / N = 11 / 2 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l1918_191894


namespace NUMINAMATH_CALUDE_running_match_participants_l1918_191892

theorem running_match_participants : 
  ∀ (n : ℕ), 
  (∃ (participant : ℕ), 
    participant ≤ n ∧ 
    participant > 0 ∧
    n - 1 = 25) →
  n = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_running_match_participants_l1918_191892


namespace NUMINAMATH_CALUDE_coffee_shop_sales_teas_sold_l1918_191845

/-- The number of teas sold at a coffee shop -/
def num_teas : ℕ := 6

/-- The number of lattes sold at a coffee shop -/
def num_lattes : ℕ := 32

/-- Theorem stating the relationship between lattes and teas sold -/
theorem coffee_shop_sales : num_lattes = 4 * num_teas + 8 := by
  sorry

/-- Theorem proving the number of teas sold -/
theorem teas_sold : num_teas = 6 := by
  sorry

end NUMINAMATH_CALUDE_coffee_shop_sales_teas_sold_l1918_191845


namespace NUMINAMATH_CALUDE_students_excelling_both_tests_l1918_191864

theorem students_excelling_both_tests 
  (total : ℕ) 
  (physical : ℕ) 
  (intellectual : ℕ) 
  (neither : ℕ) 
  (h1 : total = 50) 
  (h2 : physical = 40) 
  (h3 : intellectual = 31) 
  (h4 : neither = 4) :
  physical + intellectual - (total - neither) = 25 :=
by sorry

end NUMINAMATH_CALUDE_students_excelling_both_tests_l1918_191864


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1918_191835

/-- The weight of a single kayak in pounds -/
def kayak_weight : ℚ := 32

/-- The number of kayaks -/
def num_kayaks : ℕ := 4

/-- The number of bowling balls -/
def num_bowling_balls : ℕ := 9

/-- The weight of a single bowling ball in pounds -/
def bowling_ball_weight : ℚ := 128 / 9

theorem bowling_ball_weight_proof :
  num_bowling_balls * bowling_ball_weight = num_kayaks * kayak_weight :=
by sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1918_191835


namespace NUMINAMATH_CALUDE_calzone_time_is_124_l1918_191818

/-- The total time spent on making calzones -/
def total_calzone_time (onion_time garlic_pepper_time knead_time rest_time assemble_time : ℕ) : ℕ :=
  onion_time + garlic_pepper_time + knead_time + rest_time + assemble_time

/-- Theorem stating the total time spent on making calzones is 124 minutes -/
theorem calzone_time_is_124 : 
  ∀ (onion_time garlic_pepper_time knead_time rest_time assemble_time : ℕ),
    onion_time = 20 →
    garlic_pepper_time = onion_time / 4 →
    knead_time = 30 →
    rest_time = 2 * knead_time →
    assemble_time = (knead_time + rest_time) / 10 →
    total_calzone_time onion_time garlic_pepper_time knead_time rest_time assemble_time = 124 :=
by
  sorry


end NUMINAMATH_CALUDE_calzone_time_is_124_l1918_191818


namespace NUMINAMATH_CALUDE_unique_five_digit_multiple_of_6_l1918_191847

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

theorem unique_five_digit_multiple_of_6 :
  ∃! d : ℕ, d < 10 ∧ is_divisible_by_6 (47360 + d) ∧ sum_of_digits (47360 + d) % 3 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_five_digit_multiple_of_6_l1918_191847


namespace NUMINAMATH_CALUDE_rectangle_area_l1918_191882

def square_side : ℝ := 15
def rectangle_length : ℝ := 18

theorem rectangle_area (rectangle_width : ℝ) :
  (4 * square_side = 2 * (rectangle_length + rectangle_width)) →
  (rectangle_length * rectangle_width = 216) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1918_191882


namespace NUMINAMATH_CALUDE_nested_expression_value_l1918_191863

theorem nested_expression_value : (3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2) = 1457 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l1918_191863


namespace NUMINAMATH_CALUDE_old_belt_time_correct_l1918_191899

/-- The time it takes for the old conveyor belt to move one day's coal output -/
def old_belt_time : ℝ := 21

/-- The time it takes for the new conveyor belt to move one day's coal output -/
def new_belt_time : ℝ := 15

/-- The time it takes for both belts together to move one day's coal output -/
def combined_time : ℝ := 8.75

/-- Theorem stating that the old conveyor belt time is correct given the conditions -/
theorem old_belt_time_correct :
  1 / old_belt_time + 1 / new_belt_time = 1 / combined_time :=
by sorry

end NUMINAMATH_CALUDE_old_belt_time_correct_l1918_191899


namespace NUMINAMATH_CALUDE_sarah_picked_45_apples_l1918_191827

/-- The number of apples Sarah's brother picked -/
def brother_apples : ℕ := 9

/-- The factor by which Sarah picked more apples than her brother -/
def sarah_factor : ℕ := 5

/-- The number of apples Sarah picked -/
def sarah_apples : ℕ := sarah_factor * brother_apples

theorem sarah_picked_45_apples : sarah_apples = 45 := by
  sorry

end NUMINAMATH_CALUDE_sarah_picked_45_apples_l1918_191827


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l1918_191879

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b*x + 4032 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b'*x + 4032 = (x + r) * (x + s)) ∧
  b = 128 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l1918_191879


namespace NUMINAMATH_CALUDE_f_monotonic_and_odd_l1918_191802

def f (x : ℝ) : ℝ := x^3

theorem f_monotonic_and_odd : 
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x) := by sorry

end NUMINAMATH_CALUDE_f_monotonic_and_odd_l1918_191802


namespace NUMINAMATH_CALUDE_distance_to_line_l1918_191836

/-- Given two perpendicular lines and a point, prove the distance to a third line -/
theorem distance_to_line (m : ℝ) : 
  (∀ x y, 2*x + y - 2 = 0 → x + m*y - 1 = 0 → (2 : ℝ) * (-1/m) = -1) →
  let P := (m, m)
  (abs (P.1 + P.2 + 3) / Real.sqrt 2 : ℝ) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_l1918_191836


namespace NUMINAMATH_CALUDE_min_difference_is_one_l1918_191816

/-- Triangle with integer side lengths and specific conditions -/
structure Triangle where
  DE : ℕ
  EF : ℕ
  FD : ℕ
  perimeter_eq : DE + EF + FD = 3010
  side_order : DE < EF ∧ EF ≤ FD

/-- The smallest possible difference between EF and DE is 1 -/
theorem min_difference_is_one (t : Triangle) : 
  (∀ t' : Triangle, t'.EF - t'.DE ≥ 1) ∧ (∃ t' : Triangle, t'.EF - t'.DE = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_difference_is_one_l1918_191816


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l1918_191885

-- Define the number of available toppings
def n : ℕ := 9

-- Define the number of toppings to choose
def k : ℕ := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove
theorem pizza_toppings_combinations :
  combination n k = 84 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l1918_191885


namespace NUMINAMATH_CALUDE_rectangle_fitting_theorem_l1918_191889

/-- A rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Predicate to check if one rectangle fits inside another -/
def fits_inside (r1 r2 : Rectangle) : Prop :=
  (r1.width ≤ r2.width ∧ r1.height ≤ r2.height) ∨ 
  (r1.width ≤ r2.height ∧ r1.height ≤ r2.width)

/-- The main theorem -/
theorem rectangle_fitting_theorem (n : ℕ) (h : n ≥ 2018) 
  (S : Finset Rectangle) 
  (hS : S.card = n + 1) 
  (hSides : ∀ r ∈ S, r.width ∈ Finset.range (n + 1) ∧ r.height ∈ Finset.range (n + 1)) :
  ∃ (A B C : Rectangle), A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ 
    fits_inside A B ∧ fits_inside B C :=
by sorry

end NUMINAMATH_CALUDE_rectangle_fitting_theorem_l1918_191889


namespace NUMINAMATH_CALUDE_sequence_length_l1918_191814

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem sequence_length : 
  ∃ n : ℕ, n = 757 ∧ arithmetic_sequence 2 4 n = 3026 := by sorry

end NUMINAMATH_CALUDE_sequence_length_l1918_191814


namespace NUMINAMATH_CALUDE_passengers_taken_at_second_station_is_12_l1918_191876

/-- Represents the number of passengers on a train at different stages --/
structure TrainPassengers where
  initial : Nat
  after_first_drop : Nat
  after_first_pickup : Nat
  after_second_drop : Nat
  final : Nat

/-- Calculates the number of passengers taken at the second station --/
def passengers_taken_at_second_station (train : TrainPassengers) : Nat :=
  train.final - train.after_second_drop

/-- Theorem stating the number of passengers taken at the second station --/
theorem passengers_taken_at_second_station_is_12 :
  ∃ (train : TrainPassengers),
    train.initial = 270 ∧
    train.after_first_drop = train.initial - train.initial / 3 ∧
    train.after_first_pickup = train.after_first_drop + 280 ∧
    train.after_second_drop = train.after_first_pickup / 2 ∧
    train.final = 242 ∧
    passengers_taken_at_second_station train = 12 := by
  sorry

#check passengers_taken_at_second_station_is_12

end NUMINAMATH_CALUDE_passengers_taken_at_second_station_is_12_l1918_191876


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1918_191801

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ + 10)^2 = (4*x₁ + 6)*(x₁ + 8) ∧ 
  (x₂ + 10)^2 = (4*x₂ + 6)*(x₂ + 8) ∧ 
  (abs (x₁ - 2.131) < 0.001) ∧ 
  (abs (x₂ + 8.131) < 0.001) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1918_191801


namespace NUMINAMATH_CALUDE_grocery_shopping_remainder_l1918_191804

/-- Calculates the remaining amount after grocery shopping --/
def remaining_amount (initial_amount bread_cost candy_cost cereal_cost milk_cost : ℚ) : ℚ :=
  let initial_purchases := bread_cost + 2 * candy_cost + cereal_cost
  let after_initial := initial_amount - initial_purchases
  let fruit_cost := 0.2 * after_initial
  let after_fruit := after_initial - fruit_cost
  let after_milk := after_fruit - 2 * milk_cost
  let turkey_cost := 0.25 * after_milk
  after_milk - turkey_cost

/-- Theorem stating the remaining amount after grocery shopping --/
theorem grocery_shopping_remainder :
  remaining_amount 100 4 3 6 4.5 = 43.65 := by
  sorry

end NUMINAMATH_CALUDE_grocery_shopping_remainder_l1918_191804


namespace NUMINAMATH_CALUDE_remaining_laps_after_sunday_morning_l1918_191800

def total_required_laps : ℕ := 198

def friday_morning_laps : ℕ := 23
def friday_afternoon_laps : ℕ := 12
def friday_evening_laps : ℕ := 28

def saturday_morning_laps : ℕ := 35
def saturday_afternoon_laps : ℕ := 27

def sunday_morning_laps : ℕ := 15

def friday_total : ℕ := friday_morning_laps + friday_afternoon_laps + friday_evening_laps
def saturday_total : ℕ := saturday_morning_laps + saturday_afternoon_laps
def laps_before_sunday_break : ℕ := friday_total + saturday_total + sunday_morning_laps

theorem remaining_laps_after_sunday_morning :
  total_required_laps - laps_before_sunday_break = 58 := by
  sorry

end NUMINAMATH_CALUDE_remaining_laps_after_sunday_morning_l1918_191800


namespace NUMINAMATH_CALUDE_no_zeros_implies_a_less_than_negative_one_l1918_191871

theorem no_zeros_implies_a_less_than_negative_one (a : ℝ) :
  (∀ x : ℝ, 4^x - 2^(x+1) - a ≠ 0) → a < -1 :=
by sorry

end NUMINAMATH_CALUDE_no_zeros_implies_a_less_than_negative_one_l1918_191871


namespace NUMINAMATH_CALUDE_southton_time_capsule_depth_l1918_191862

theorem southton_time_capsule_depth :
  let southton_depth : ℝ := 9
  let northton_depth : ℝ := 48
  northton_depth = 4 * southton_depth + 12 →
  southton_depth = 9 := by
sorry

end NUMINAMATH_CALUDE_southton_time_capsule_depth_l1918_191862


namespace NUMINAMATH_CALUDE_common_roots_product_l1918_191865

/-- Given two cubic equations with two common roots, prove their product is 4∛5 -/
theorem common_roots_product (C D : ℝ) : 
  ∃ (u v w t : ℝ), 
    (u^3 + C*u - 20 = 0) ∧ 
    (v^3 + C*v - 20 = 0) ∧ 
    (w^3 + C*w - 20 = 0) ∧
    (u^3 + D*u^2 - 40 = 0) ∧ 
    (v^3 + D*v^2 - 40 = 0) ∧ 
    (t^3 + D*t^2 - 40 = 0) ∧
    (u ≠ v) ∧ (u ≠ w) ∧ (v ≠ w) ∧
    (u ≠ t) ∧ (v ≠ t) →
    u * v = 4 * Real.rpow 5 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_common_roots_product_l1918_191865


namespace NUMINAMATH_CALUDE_parallel_perpendicular_lines_l1918_191803

/-- Given a point P and a line l, prove the equations of parallel and perpendicular lines through P --/
theorem parallel_perpendicular_lines 
  (P : ℝ × ℝ) 
  (l : ℝ → ℝ → Prop) 
  (hl : l = fun x y => 3 * x - y - 7 = 0) 
  (hP : P = (2, 1)) :
  let parallel_line := fun x y => 3 * x - y - 5 = 0
  let perpendicular_line := fun x y => x - 3 * y + 1 = 0
  (∀ x y, parallel_line x y ↔ (3 * x - y = 3 * P.1 - P.2)) ∧ 
  (parallel_line P.1 P.2) ∧
  (∀ x y, perpendicular_line x y ↔ (x - 3 * y = P.1 - 3 * P.2)) ∧ 
  (perpendicular_line P.1 P.2) ∧
  (∀ x₁ y₁ x₂ y₂, l x₁ y₁ → l x₂ y₂ → x₁ ≠ x₂ → (y₁ - y₂) / (x₁ - x₂) = 3) ∧
  (∀ x₁ y₁ x₂ y₂, perpendicular_line x₁ y₁ → perpendicular_line x₂ y₂ → x₁ ≠ x₂ → 
    (y₁ - y₂) / (x₁ - x₂) = -1/3) := by
  sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_lines_l1918_191803


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_meaningful_range_l1918_191817

-- Define the property of being meaningful for a square root
def is_meaningful (x : ℝ) : Prop := x ≥ 0

-- State the theorem
theorem sqrt_x_minus_3_meaningful_range (x : ℝ) :
  is_meaningful (x - 3) → x ≥ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_meaningful_range_l1918_191817


namespace NUMINAMATH_CALUDE_triangle_area_in_circle_l1918_191820

theorem triangle_area_in_circle (r : ℝ) (h : r = 3) : 
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Sides are positive
    a = b ∧ c = a * Real.sqrt 2 ∧  -- Sides are in ratio 1:1:√2
    c = 2 * r ∧  -- Diameter of circle
    (1/2) * a * b = 9 := by  -- Area of triangle
  sorry

end NUMINAMATH_CALUDE_triangle_area_in_circle_l1918_191820


namespace NUMINAMATH_CALUDE_linear_function_proof_l1918_191833

theorem linear_function_proof (k b : ℝ) :
  (1 * k + b = -2) →
  (-1 * k + b = -4) →
  (3 * k + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_proof_l1918_191833


namespace NUMINAMATH_CALUDE_relationship_increases_with_ratio_difference_l1918_191888

-- Define the structure for a 2x2 contingency table
structure ContingencyTable :=
  (a b c d : ℕ)

-- Define the ratios
def ratio1 (t : ContingencyTable) : ℚ := t.a / (t.a + t.b)
def ratio2 (t : ContingencyTable) : ℚ := t.c / (t.c + t.d)

-- Define the difference between ratios
def ratioDifference (t : ContingencyTable) : ℚ := |ratio1 t - ratio2 t|

-- Define a measure of relationship possibility (e.g., chi-square value)
noncomputable def relationshipPossibility (t : ContingencyTable) : ℝ := sorry

-- State the theorem
theorem relationship_increases_with_ratio_difference (t : ContingencyTable) :
  ∀ (t1 t2 : ContingencyTable),
    ratioDifference t1 < ratioDifference t2 →
    relationshipPossibility t1 < relationshipPossibility t2 :=
sorry

end NUMINAMATH_CALUDE_relationship_increases_with_ratio_difference_l1918_191888


namespace NUMINAMATH_CALUDE_lucy_fish_count_l1918_191840

/-- The number of fish Lucy needs to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy wants to have -/
def total_fish : ℕ := 280

/-- The number of fish Lucy currently has -/
def current_fish : ℕ := total_fish - fish_to_buy

theorem lucy_fish_count : current_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l1918_191840


namespace NUMINAMATH_CALUDE_max_distance_line_l1918_191834

/-- The point through which the line passes -/
def point : ℝ × ℝ := (1, 2)

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := x + 2*y - 5 = 0

/-- Theorem stating that the given line equation represents the line with maximum distance from the origin passing through the specified point -/
theorem max_distance_line :
  line_equation point.1 point.2 ∧
  ∀ (a b c : ℝ), (a*point.1 + b*point.2 + c = 0) →
    (a^2 + b^2 ≤ 1^2 + 2^2) :=
sorry

end NUMINAMATH_CALUDE_max_distance_line_l1918_191834


namespace NUMINAMATH_CALUDE_johns_trip_duration_l1918_191813

/-- The duration of John's trip given his travel conditions -/
def trip_duration (first_country_duration : ℕ) (num_countries : ℕ) : ℕ :=
  first_country_duration + 2 * first_country_duration * (num_countries - 1)

/-- Theorem stating that John's trip duration is 10 weeks -/
theorem johns_trip_duration :
  trip_duration 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_johns_trip_duration_l1918_191813


namespace NUMINAMATH_CALUDE_grid_figure_boundary_theorem_l1918_191896

/-- A grid figure is a shape cut from grid paper along grid lines without holes. -/
structure GridFigure where
  -- Add necessary fields here
  no_holes : Bool

/-- Represents a set of straight cuts along grid lines. -/
structure GridCuts where
  total_length : ℕ
  divides_into_cells : Bool

/-- Checks if a grid figure has a straight boundary segment of at least given length. -/
def has_straight_boundary_segment (figure : GridFigure) (length : ℕ) : Prop :=
  sorry

theorem grid_figure_boundary_theorem (figure : GridFigure) (cuts : GridCuts) :
  figure.no_holes ∧ 
  cuts.total_length = 2017 ∧
  cuts.divides_into_cells →
  has_straight_boundary_segment figure 2 :=
by sorry

end NUMINAMATH_CALUDE_grid_figure_boundary_theorem_l1918_191896


namespace NUMINAMATH_CALUDE_product_zero_in_special_set_l1918_191809

theorem product_zero_in_special_set (n : ℕ) (h : n = 1997) (S : Finset ℝ) 
  (hS : S.card = n) 
  (hSum : ∀ x ∈ S, (S.sum id - x) ∈ S) : 
  S.prod id = 0 := by
sorry

end NUMINAMATH_CALUDE_product_zero_in_special_set_l1918_191809


namespace NUMINAMATH_CALUDE_photo_difference_l1918_191821

theorem photo_difference (initial_photos : ℕ) (final_photos : ℕ) : 
  initial_photos = 400 →
  final_photos = 920 →
  (final_photos - initial_photos) - (initial_photos / 2) = 120 :=
by
  sorry

#check photo_difference

end NUMINAMATH_CALUDE_photo_difference_l1918_191821


namespace NUMINAMATH_CALUDE_complete_square_sum_l1918_191838

theorem complete_square_sum (x : ℝ) : 
  (x^2 - 10*x + 15 = 0) → 
  ∃ (d e : ℤ), ((x + d : ℝ)^2 = e) ∧ (d + e = 5) :=
by sorry

end NUMINAMATH_CALUDE_complete_square_sum_l1918_191838


namespace NUMINAMATH_CALUDE_sodium_bisulfite_moles_l1918_191857

-- Define the molecules and their molar quantities
structure Reaction :=
  (NaHSO3 : ℝ)  -- moles of Sodium bisulfite
  (HCl : ℝ)     -- moles of Hydrochloric acid
  (H2O : ℝ)     -- moles of Water

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.NaHSO3 = r.HCl ∧ r.NaHSO3 = r.H2O

-- Theorem statement
theorem sodium_bisulfite_moles :
  ∀ r : Reaction,
  r.HCl = 1 →        -- 1 mole of Hydrochloric acid is used
  r.H2O = 1 →        -- The reaction forms 1 mole of Water
  balanced_equation r →  -- The reaction equation is balanced
  r.NaHSO3 = 1 :=    -- The number of moles of Sodium bisulfite is 1
by
  sorry

end NUMINAMATH_CALUDE_sodium_bisulfite_moles_l1918_191857


namespace NUMINAMATH_CALUDE_max_hands_for_54_coincidences_l1918_191812

/-- Represents a clock with minute hands moving in opposite directions -/
structure Clock where
  coincidences : ℕ  -- Number of coincidences in an hour
  handsForward : ℕ  -- Number of hands moving forward
  handsBackward : ℕ -- Number of hands moving backward

/-- The total number of hands on the clock -/
def Clock.totalHands (c : Clock) : ℕ := c.handsForward + c.handsBackward

/-- Predicate to check if the clock configuration is valid -/
def Clock.isValid (c : Clock) : Prop :=
  c.handsForward * c.handsBackward * 2 = c.coincidences

/-- Theorem stating the maximum number of hands for a clock with 54 coincidences -/
theorem max_hands_for_54_coincidences :
  ∃ (c : Clock), c.coincidences = 54 ∧ c.isValid ∧
  ∀ (d : Clock), d.coincidences = 54 → d.isValid → d.totalHands ≤ c.totalHands :=
by
  sorry

end NUMINAMATH_CALUDE_max_hands_for_54_coincidences_l1918_191812


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1918_191843

theorem quadratic_inequality_solution_set 
  (a b c α β : ℝ) 
  (h1 : α > 0) 
  (h2 : β > α) 
  (h3 : ∀ x, ax^2 + b*x + c > 0 ↔ α < x ∧ x < β) :
  ∀ x, c*x^2 + b*x + a > 0 ↔ 1/β < x ∧ x < 1/α :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1918_191843


namespace NUMINAMATH_CALUDE_square_difference_equality_l1918_191830

theorem square_difference_equality : 1010^2 - 990^2 - 1005^2 + 995^2 = 20000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l1918_191830


namespace NUMINAMATH_CALUDE_invertible_function_problem_l1918_191897

theorem invertible_function_problem (g : ℝ → ℝ) (c : ℝ) 
  (h_invertible : Function.Bijective g)
  (h_gc : g c = 3)
  (h_g3 : g 3 = 5) :
  c - 3 = -3 := by
  sorry

end NUMINAMATH_CALUDE_invertible_function_problem_l1918_191897


namespace NUMINAMATH_CALUDE_shaded_area_is_16_l1918_191811

/-- Represents the shaded area of a 6x6 grid with triangles and trapezoids -/
def shadedArea (gridSize : Nat) (triangleCount : Nat) (trapezoidCount : Nat) 
  (triangleSquares : Nat) (trapezoidSquares : Nat) : Nat :=
  triangleCount * triangleSquares + trapezoidCount * trapezoidSquares

/-- Theorem stating that the shaded area of the described grid is 16 square units -/
theorem shaded_area_is_16 : 
  shadedArea 6 2 4 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_16_l1918_191811


namespace NUMINAMATH_CALUDE_remainder_of_sum_of_powers_l1918_191859

theorem remainder_of_sum_of_powers (n : ℕ) : (20^16 + 201^6) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_of_powers_l1918_191859


namespace NUMINAMATH_CALUDE_ezekiel_hike_third_day_l1918_191873

/-- Represents a three-day hike --/
structure ThreeDayHike where
  total_distance : ℕ
  day1_distance : ℕ
  day2_distance : ℕ

/-- Calculates the distance covered on the third day of a three-day hike --/
def third_day_distance (hike : ThreeDayHike) : ℕ :=
  hike.total_distance - (hike.day1_distance + hike.day2_distance)

/-- Theorem stating that for the given hike parameters, the third day distance is 22 km --/
theorem ezekiel_hike_third_day :
  let hike : ThreeDayHike := {
    total_distance := 50,
    day1_distance := 10,
    day2_distance := 18
  }
  third_day_distance hike = 22 := by
  sorry


end NUMINAMATH_CALUDE_ezekiel_hike_third_day_l1918_191873


namespace NUMINAMATH_CALUDE_remainder_theorem_l1918_191807

theorem remainder_theorem (z : ℕ) (hz : z > 0) (hz_div : 4 ∣ z) : (z * (2 + 4 + z) + 3) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1918_191807


namespace NUMINAMATH_CALUDE_complex_power_modulus_l1918_191825

theorem complex_power_modulus : 
  Complex.abs ((1 / 2 : ℂ) + (Complex.I * Real.sqrt 3 / 2)) ^ 12 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l1918_191825


namespace NUMINAMATH_CALUDE_kaleb_books_l1918_191855

theorem kaleb_books (initial_books sold_books new_books : ℕ) :
  initial_books = 34 →
  sold_books = 17 →
  new_books = 7 →
  initial_books - sold_books + new_books = 24 :=
by sorry

end NUMINAMATH_CALUDE_kaleb_books_l1918_191855


namespace NUMINAMATH_CALUDE_reflect_point_coordinates_l1918_191823

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

theorem reflect_point_coordinates :
  let P : Point := { x := 4, y := -1 }
  reflectAcrossYAxis P = { x := -4, y := -1 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_point_coordinates_l1918_191823


namespace NUMINAMATH_CALUDE_min_value_of_f_range_of_m_l1918_191893

noncomputable section

open Real MeasureTheory

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := log x + a / x - 1
def g (x : ℝ) : ℝ := x + 1 / x

-- Part 1
theorem min_value_of_f :
  (∀ x > 0, f 2 x ≥ log 2) ∧ (∃ x > 0, f 2 x = log 2) := by sorry

-- Part 2
theorem range_of_m :
  let f' := f (-1)
  {m : ℝ | ∃ x ∈ Set.Icc 1 (Real.exp 1), g x < m * (f' x + 1)} =
  Set.Ioi ((Real.exp 2 + 1) / (Real.exp 1 - 1)) ∪ Set.Iio (-2) := by sorry

end

end NUMINAMATH_CALUDE_min_value_of_f_range_of_m_l1918_191893


namespace NUMINAMATH_CALUDE_max_ab_value_l1918_191831

theorem max_ab_value (a b : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → |a * x + b| ≤ 1) → 
  a * b ≤ (1 : ℝ) / 4 := by sorry

end NUMINAMATH_CALUDE_max_ab_value_l1918_191831


namespace NUMINAMATH_CALUDE_discount_percentage_is_five_percent_l1918_191854

def cameras_cost : ℝ := 2 * 110
def frames_cost : ℝ := 3 * 120
def total_cost : ℝ := cameras_cost + frames_cost
def discounted_price : ℝ := 551

theorem discount_percentage_is_five_percent :
  (total_cost - discounted_price) / total_cost * 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_is_five_percent_l1918_191854


namespace NUMINAMATH_CALUDE_yard_length_26_trees_l1918_191806

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 26 equally spaced trees and 20 meters between trees is 500 meters -/
theorem yard_length_26_trees : 
  yard_length 26 20 = 500 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_26_trees_l1918_191806


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1918_191819

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + x < 0 ↔ -1/2 < x ∧ x < 1/2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1918_191819


namespace NUMINAMATH_CALUDE_cuboid_4x3x3_two_sided_cubes_l1918_191850

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  width : ℕ
  length : ℕ
  height : ℕ

/-- Represents a cube with given side length -/
structure Cube where
  side : ℕ

/-- Function to calculate the number of cubes with paint on exactly two sides -/
def cubesWithTwoPaintedSides (c : Cuboid) (numCubes : ℕ) : ℕ :=
  sorry

/-- Theorem stating that a 4x3x3 cuboid cut into 36 equal-sized cubes has 16 cubes with paint on exactly two sides -/
theorem cuboid_4x3x3_two_sided_cubes :
  let c : Cuboid := { width := 4, length := 3, height := 3 }
  cubesWithTwoPaintedSides c 36 = 16 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_4x3x3_two_sided_cubes_l1918_191850


namespace NUMINAMATH_CALUDE_min_value_theorem_l1918_191839

theorem min_value_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * (a + b + c) + b * c = 4 + 2 * Real.sqrt 3) :
  (∀ x y z, x > 0 → y > 0 → z > 0 → x * (x + y + z) + y * z = 4 + 2 * Real.sqrt 3 →
    2 * x + y + z ≥ 2 * Real.sqrt 3 + 2) ∧
  (∃ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x * (x + y + z) + y * z = 4 + 2 * Real.sqrt 3 ∧
    2 * x + y + z = 2 * Real.sqrt 3 + 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1918_191839


namespace NUMINAMATH_CALUDE_tangent_line_quadratic_l1918_191853

/-- Given a quadratic function f(x) = x² + ax + b, if the tangent line
    to f at x = 0 is x - y + 1 = 0, then a = 1 and b = 1 -/
theorem tangent_line_quadratic (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  let f' : ℝ → ℝ := λ x ↦ 2*x + a
  (∀ x, f' x = (deriv f) x) →
  (f' 0 = 1) →
  (f 0 = 1) →
  a = 1 ∧ b = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_quadratic_l1918_191853


namespace NUMINAMATH_CALUDE_final_replacement_weight_is_140_l1918_191869

/-- The weight of the final replacement person in a series of replacements --/
def final_replacement_weight (initial_people : ℕ) (initial_weight : ℝ) 
  (first_increase : ℝ) (second_decrease : ℝ) (third_increase : ℝ) : ℝ :=
  let first_replacement := initial_weight + initial_people * first_increase
  let second_replacement := first_replacement - initial_people * second_decrease
  second_replacement + initial_people * third_increase - 
    (second_replacement - initial_people * second_decrease)

/-- Theorem stating the weight of the final replacement person --/
theorem final_replacement_weight_is_140 :
  final_replacement_weight 10 70 4 2 5 = 140 := by
  sorry


end NUMINAMATH_CALUDE_final_replacement_weight_is_140_l1918_191869


namespace NUMINAMATH_CALUDE_shaded_area_is_36_l1918_191875

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square -/
structure Square :=
  (bottomLeft : Point)
  (sideLength : ℝ)

/-- Represents a right triangle -/
structure RightTriangle :=
  (bottomLeft : Point)
  (base : ℝ)
  (height : ℝ)

/-- Calculates the area of the shaded region -/
def shadedArea (square : Square) (triangle : RightTriangle) : ℝ :=
  sorry

/-- Theorem stating the area of the shaded region is 36 square units -/
theorem shaded_area_is_36 (square : Square) (triangle : RightTriangle) :
  square.bottomLeft = Point.mk 0 0 →
  square.sideLength = 12 →
  triangle.bottomLeft = Point.mk 12 0 →
  triangle.base = 12 →
  triangle.height = 12 →
  shadedArea square triangle = 36 :=
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_36_l1918_191875


namespace NUMINAMATH_CALUDE_quadratic_trinomial_from_complete_square_l1918_191874

/-- 
Given a quadratic trinomial p(x) = Ax² + Bx + C, if its complete square form 
is x⁴ - 6x³ + 7x² + ax + b, then p(x) = x² - 3x - 1 or p(x) = -x² + 3x + 1.
-/
theorem quadratic_trinomial_from_complete_square (A B C a b : ℝ) :
  (∀ x, A * x^2 + B * x + C = x^4 - 6*x^3 + 7*x^2 + a*x + b) →
  ((A = 1 ∧ B = -3 ∧ C = -1) ∨ (A = -1 ∧ B = 3 ∧ C = 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_from_complete_square_l1918_191874


namespace NUMINAMATH_CALUDE_emma_wrapping_time_l1918_191837

/-- Represents the time (in hours) it takes for Emma to wrap presents individually -/
def emma_time : ℝ := 6

/-- Represents the time (in hours) it takes for Troy to wrap presents individually -/
def troy_time : ℝ := 8

/-- Represents the time (in hours) Emma and Troy work together -/
def together_time : ℝ := 2

/-- Represents the additional time (in hours) Emma works alone after Troy leaves -/
def emma_extra_time : ℝ := 2.5

theorem emma_wrapping_time :
  emma_time = 6 ∧
  (together_time * (1 / emma_time + 1 / troy_time) + emma_extra_time / emma_time = 1) :=
sorry

end NUMINAMATH_CALUDE_emma_wrapping_time_l1918_191837


namespace NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l1918_191829

/-- Given a quadratic inequality x^2 - ax + b < 0 with solution set {x | 1 < x < 2},
    prove that the sum of coefficients a and b is equal to 5. -/
theorem quadratic_inequality_coefficient_sum (a b : ℝ) : 
  (∀ x, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_coefficient_sum_l1918_191829


namespace NUMINAMATH_CALUDE_min_value_expression_l1918_191895

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3) :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1918_191895


namespace NUMINAMATH_CALUDE_interest_rate_satisfies_conditions_interest_rate_unique_solution_l1918_191870

/-- The principal amount -/
def P : ℝ := 6800.000000000145

/-- The time period in years -/
def t : ℝ := 2

/-- The difference between compound interest and simple interest -/
def diff : ℝ := 17

/-- The interest rate as a percentage -/
def r : ℝ := 5

/-- Theorem stating that the given interest rate satisfies the conditions -/
theorem interest_rate_satisfies_conditions :
  P * (1 + r / 100) ^ t - P - (P * r * t / 100) = diff := by
  sorry

/-- Theorem stating that the given interest rate is the unique solution -/
theorem interest_rate_unique_solution :
  ∀ x : ℝ, P * (1 + x / 100) ^ t - P - (P * x * t / 100) = diff → x = r := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_satisfies_conditions_interest_rate_unique_solution_l1918_191870


namespace NUMINAMATH_CALUDE_total_profit_is_60000_l1918_191881

/-- Calculates the total profit of a partnership given the investments and one partner's share of the profit -/
def calculate_total_profit (a_investment b_investment c_investment : ℕ) (c_profit : ℕ) : ℕ :=
  let total_parts := a_investment / 9000 + b_investment / 9000 + c_investment / 9000
  let c_parts := c_investment / 9000
  let profit_per_part := c_profit / c_parts
  total_parts * profit_per_part

/-- Proves that the total profit is $60,000 given the specific investments and c's profit share -/
theorem total_profit_is_60000 :
  calculate_total_profit 45000 63000 72000 24000 = 60000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_60000_l1918_191881


namespace NUMINAMATH_CALUDE_line_intersection_triangle_l1918_191846

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a Line type
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

-- Define a function to check if three points are collinear
def collinear (A B C : Point) : Prop :=
  (B.y - A.y) * (C.x - A.x) = (C.y - A.y) * (B.x - A.x)

-- Define a function to check if a point lies on a line
def pointOnLine (P : Point) (L : Line) : Prop :=
  L.a * P.x + L.b * P.y + L.c = 0

-- Define a function to check if a line intersects a segment
def lineIntersectsSegment (L : Line) (A B : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    pointOnLine (Point.mk (A.x + t * (B.x - A.x)) (A.y + t * (B.y - A.y))) L

-- Main theorem
theorem line_intersection_triangle (A B C : Point) (L : Line) :
  ¬collinear A B C →
  ¬pointOnLine A L →
  ¬pointOnLine B L →
  ¬pointOnLine C L →
  (¬lineIntersectsSegment L B C ∧ ¬lineIntersectsSegment L C A ∧ ¬lineIntersectsSegment L A B) ∨
  (lineIntersectsSegment L B C ∧ lineIntersectsSegment L C A ∧ ¬lineIntersectsSegment L A B) ∨
  (lineIntersectsSegment L B C ∧ ¬lineIntersectsSegment L C A ∧ lineIntersectsSegment L A B) ∨
  (¬lineIntersectsSegment L B C ∧ lineIntersectsSegment L C A ∧ lineIntersectsSegment L A B) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_triangle_l1918_191846


namespace NUMINAMATH_CALUDE_geometric_sequence_value_l1918_191824

/-- A geometric sequence with sum of first n terms S_n = a · 2^n + a - 2 -/
def GeometricSequence (a : ℝ) : ℕ → ℝ := fun n ↦ 
  if n = 0 then 0 else (a * 2^n + a - 2) - (a * 2^(n-1) + a - 2)

/-- The sum of the first n terms of the geometric sequence -/
def SumFirstNTerms (a : ℝ) : ℕ → ℝ := fun n ↦ a * 2^n + a - 2

theorem geometric_sequence_value (a : ℝ) : 
  (∀ n : ℕ, n ≥ 1 → GeometricSequence a (n+1) / GeometricSequence a n = GeometricSequence a 2 / GeometricSequence a 1) →
  a = 1 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_value_l1918_191824


namespace NUMINAMATH_CALUDE_chord_count_l1918_191887

theorem chord_count (n : ℕ) (h : n = 10) : Nat.choose n 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_chord_count_l1918_191887


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1918_191822

theorem quadratic_real_roots (a b c : ℝ) :
  (∃ x : ℝ, (a^2 + b^2 + c^2) * x^2 + 2*(a + b + c) * x + 3 = 0) ↔
  (a = b ∧ b = c ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1918_191822


namespace NUMINAMATH_CALUDE_no_valid_numbers_l1918_191842

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ (a x : ℕ), n = 1000 * a + x ∧ 100 ≤ x ∧ x < 1000 ∧ 8 * x = n

theorem no_valid_numbers : ¬∃ (n : ℕ), is_valid_number n := by
  sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l1918_191842


namespace NUMINAMATH_CALUDE_same_monotonicity_intervals_l1918_191872

def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x + 2

def f' (x : ℝ) : ℝ := 3*x^2 - 12*x + 9

theorem same_monotonicity_intervals :
  (∀ x ∈ Set.Icc 1 2, (∀ y ∈ Set.Icc 1 2, x ≤ y → f x ≥ f y ∧ f' x ≥ f' y)) ∧
  (∀ x ∈ Set.Ioi 3, (∀ y ∈ Set.Ioi 3, x ≤ y → f x ≤ f y ∧ f' x ≤ f' y)) ∧
  (∀ a b : ℝ, a < b ∧ 
    ((a < 1 ∧ b > 1) ∨ (a < 2 ∧ b > 2) ∨ (a < 3 ∧ b > 3)) →
    ¬(∀ x ∈ Set.Icc a b, (∀ y ∈ Set.Icc a b, x ≤ y → 
      (f x ≤ f y ∧ f' x ≤ f' y) ∨ (f x ≥ f y ∧ f' x ≥ f' y)))) :=
by sorry

#check same_monotonicity_intervals

end NUMINAMATH_CALUDE_same_monotonicity_intervals_l1918_191872


namespace NUMINAMATH_CALUDE_abs_a_plus_b_equals_three_sqrt_three_l1918_191858

/-- The function f as defined in the problem -/
def f (x y : ℝ) : ℝ := 3 * x^2 + 3 * x * y + 1

/-- The theorem statement -/
theorem abs_a_plus_b_equals_three_sqrt_three
  (a b : ℝ)
  (h1 : f a b + 1 = 42)
  (h2 : f b a = 42) :
  |a + b| = 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_a_plus_b_equals_three_sqrt_three_l1918_191858


namespace NUMINAMATH_CALUDE_visitors_in_scientific_notation_l1918_191883

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem visitors_in_scientific_notation :
  toScientificNotation 20300 = ScientificNotation.mk 2.03 4 sorry := by sorry

end NUMINAMATH_CALUDE_visitors_in_scientific_notation_l1918_191883


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l1918_191848

theorem log_equality_implies_golden_ratio (a b : ℝ) :
  a > 0 ∧ b > 0 →
  Real.log a / Real.log 8 = Real.log b / Real.log 18 ∧
  Real.log a / Real.log 8 = Real.log (a + b) / Real.log 32 →
  b / a = (1 + Real.sqrt 5) / 2 := by
sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l1918_191848


namespace NUMINAMATH_CALUDE_largest_n_is_max_l1918_191868

/-- The largest positive integer n such that there exist n real numbers
    satisfying the given inequality. -/
def largest_n : ℕ := 31

/-- The condition that must be satisfied by the n real numbers. -/
def satisfies_condition (x : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i < j → j ≤ n →
    (1 + x i * x j)^2 ≤ 0.99 * (1 + (x i)^2) * (1 + (x j)^2)

/-- The main theorem stating that largest_n is indeed the largest such n. -/
theorem largest_n_is_max :
  (∃ x : ℕ → ℝ, satisfies_condition x largest_n) ∧
  (∀ m : ℕ, m > largest_n → ¬∃ x : ℕ → ℝ, satisfies_condition x m) :=
sorry

end NUMINAMATH_CALUDE_largest_n_is_max_l1918_191868


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1918_191826

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1918_191826


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1918_191856

-- Define the relationship between x and y
def inverse_variation (x y : ℝ) (k : ℝ) : Prop := x * y^3 = k

-- Theorem statement
theorem inverse_variation_problem (x₁ x₂ y₁ y₂ k : ℝ) 
  (h1 : inverse_variation x₁ y₁ k)
  (h2 : x₁ = 8)
  (h3 : y₁ = 1)
  (h4 : y₂ = 2)
  (h5 : inverse_variation x₂ y₂ k) :
  x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1918_191856


namespace NUMINAMATH_CALUDE_one_seventh_minus_one_eleventh_equals_100_l1918_191898

theorem one_seventh_minus_one_eleventh_equals_100 :
  let N : ℚ := 1925
  (N / 7) - (N / 11) = 100 := by
  sorry

end NUMINAMATH_CALUDE_one_seventh_minus_one_eleventh_equals_100_l1918_191898


namespace NUMINAMATH_CALUDE_f_decreasing_after_2_l1918_191878

def f (x : ℝ) : ℝ := -(x - 2)^2 + 3

theorem f_decreasing_after_2 :
  ∀ x₁ x₂ : ℝ, 2 < x₁ → x₁ < x₂ → f x₂ < f x₁ := by
  sorry

end NUMINAMATH_CALUDE_f_decreasing_after_2_l1918_191878


namespace NUMINAMATH_CALUDE_prime_roots_sum_reciprocals_l1918_191841

theorem prime_roots_sum_reciprocals (p q m : ℕ) : 
  Prime p → Prime q → 
  (p : ℝ)^2 - 99*p + m = 0 → 
  (q : ℝ)^2 - 99*q + m = 0 → 
  (p : ℝ)/q + (q : ℝ)/p = 9413/194 := by
  sorry

end NUMINAMATH_CALUDE_prime_roots_sum_reciprocals_l1918_191841


namespace NUMINAMATH_CALUDE_range_of_a_l1918_191832

def A (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 = 0}
def B : Set ℝ := {1, 2}

theorem range_of_a (a : ℝ) : 
  (A a ∪ B = B) ↔ a ∈ Set.Icc (-2 : ℝ) 2 ∧ a ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1918_191832


namespace NUMINAMATH_CALUDE_boat_rental_problem_l1918_191808

theorem boat_rental_problem :
  ∀ (big_boats small_boats : ℕ),
    big_boats + small_boats = 12 →
    6 * big_boats + 4 * small_boats = 58 →
    big_boats = 5 ∧ small_boats = 7 := by
  sorry

end NUMINAMATH_CALUDE_boat_rental_problem_l1918_191808


namespace NUMINAMATH_CALUDE_max_value_of_f_l1918_191810

def f (x : ℝ) : ℝ := -x^2 + 2*x + 8

theorem max_value_of_f :
  ∃ (M : ℝ), M = 9 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1918_191810


namespace NUMINAMATH_CALUDE_problem_235_l1918_191815

theorem problem_235 (x y : ℝ) : 
  y + Real.sqrt (x^2 + y^2) = 16 ∧ x - y = 2 → x = 8 ∧ y = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_235_l1918_191815


namespace NUMINAMATH_CALUDE_tire_price_proof_l1918_191884

/-- The regular price of one tire -/
def regular_price : ℝ := 79

/-- The sale price of the fourth tire -/
def fourth_tire_price : ℝ := 3

/-- The total cost of four tires -/
def total_cost : ℝ := 240

theorem tire_price_proof :
  3 * regular_price + fourth_tire_price = total_cost :=
by sorry

end NUMINAMATH_CALUDE_tire_price_proof_l1918_191884


namespace NUMINAMATH_CALUDE_unique_birth_year_exists_l1918_191877

def sumOfDigits (year : Nat) : Nat :=
  (year / 1000) + ((year / 100) % 10) + ((year / 10) % 10) + (year % 10)

theorem unique_birth_year_exists : 
  ∃! year : Nat, 1900 ≤ year ∧ year < 2003 ∧ 2003 - year = sumOfDigits year := by
  sorry

end NUMINAMATH_CALUDE_unique_birth_year_exists_l1918_191877


namespace NUMINAMATH_CALUDE_perfect_squares_among_options_l1918_191860

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def option_a : ℕ := 3^3 * 4^4 * 5^5
def option_b : ℕ := 3^4 * 4^5 * 5^6
def option_c : ℕ := 3^6 * 4^4 * 5^6
def option_d : ℕ := 3^5 * 4^6 * 5^5
def option_e : ℕ := 3^6 * 4^6 * 5^4

theorem perfect_squares_among_options :
  (¬ is_perfect_square option_a) ∧
  (is_perfect_square option_b) ∧
  (is_perfect_square option_c) ∧
  (¬ is_perfect_square option_d) ∧
  (is_perfect_square option_e) := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_among_options_l1918_191860


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l1918_191880

-- Equation 1: x^2 - 4x + 3 = 0
theorem solve_equation_1 : 
  ∃ x₁ x₂ : ℝ, x₁^2 - 4*x₁ + 3 = 0 ∧ x₂^2 - 4*x₂ + 3 = 0 ∧ x₁ = 1 ∧ x₂ = 3 := by
  sorry

-- Equation 2: (x + 1)(x - 2) = 4
theorem solve_equation_2 : 
  ∃ x₁ x₂ : ℝ, (x₁ + 1)*(x₁ - 2) = 4 ∧ (x₂ + 1)*(x₂ - 2) = 4 ∧ x₁ = -2 ∧ x₂ = 3 := by
  sorry

-- Equation 3: 3x(x - 1) = 2 - 2x
theorem solve_equation_3 : 
  ∃ x₁ x₂ : ℝ, 3*x₁*(x₁ - 1) = 2 - 2*x₁ ∧ 3*x₂*(x₂ - 1) = 2 - 2*x₂ ∧ x₁ = 1 ∧ x₂ = -2/3 := by
  sorry

-- Equation 4: 2x^2 - 4x - 1 = 0
theorem solve_equation_4 : 
  ∃ x₁ x₂ : ℝ, 2*x₁^2 - 4*x₁ - 1 = 0 ∧ 2*x₂^2 - 4*x₂ - 1 = 0 ∧ 
  x₁ = (2 + Real.sqrt 6) / 2 ∧ x₂ = (2 - Real.sqrt 6) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l1918_191880


namespace NUMINAMATH_CALUDE_equation_solution_l1918_191886

theorem equation_solution : 
  ∀ x : ℝ, (Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 4) ↔ 
  (x = 15 + 4 * Real.sqrt 11 ∨ x = 15 - 4 * Real.sqrt 11) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1918_191886


namespace NUMINAMATH_CALUDE_budget_research_development_l1918_191828

theorem budget_research_development (transportation utilities equipment supplies salaries research_development : ℝ) : 
  transportation = 20 →
  utilities = 5 →
  equipment = 4 →
  supplies = 2 →
  salaries = 216 / 360 * 100 →
  transportation + utilities + equipment + supplies + salaries + research_development = 100 →
  research_development = 9 := by
sorry

end NUMINAMATH_CALUDE_budget_research_development_l1918_191828


namespace NUMINAMATH_CALUDE_nice_set_property_l1918_191851

def nice (P : Set (ℤ × ℤ)) : Prop :=
  (∀ a b, (a, b) ∈ P → (b, a) ∈ P) ∧
  (∀ a b c d, (a, b) ∈ P → (c, d) ∈ P → (a + c, b - d) ∈ P)

theorem nice_set_property (p q : ℤ) (h1 : Nat.gcd p.natAbs q.natAbs = 1) 
  (h2 : p % 2 ≠ q % 2) :
  ∀ (P : Set (ℤ × ℤ)), nice P → (p, q) ∈ P → P = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_nice_set_property_l1918_191851


namespace NUMINAMATH_CALUDE_shaded_area_13x5_grid_l1918_191890

/-- Represents a rectangular grid with a shaded region --/
structure ShadedGrid where
  width : ℕ
  height : ℕ
  shaded_area : ℝ

/-- Calculates the area of the shaded region in the grid --/
def calculate_shaded_area (grid : ShadedGrid) : ℝ :=
  let total_area := grid.width * grid.height
  let triangle_area := (grid.width * grid.height) / 2
  total_area - triangle_area

/-- Theorem stating that the shaded area of a 13x5 grid with an excluded triangle is 32.5 --/
theorem shaded_area_13x5_grid :
  ∃ (grid : ShadedGrid),
    grid.width = 13 ∧
    grid.height = 5 ∧
    calculate_shaded_area grid = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_13x5_grid_l1918_191890


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l1918_191867

/-- Given a group of cows and chickens, if the total number of legs is 18 more than
    twice the total number of heads, then the number of cows is 9. -/
theorem cow_chicken_problem (cows chickens : ℕ) : 
  4 * cows + 2 * chickens = 2 * (cows + chickens) + 18 → cows = 9 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l1918_191867


namespace NUMINAMATH_CALUDE_scientific_notation_of_million_l1918_191891

/-- Prove that 1.6369 million is equal to 1.6369 × 10^6 -/
theorem scientific_notation_of_million (x : ℝ) : 
  x * 1000000 = x * (10 ^ 6) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_million_l1918_191891
