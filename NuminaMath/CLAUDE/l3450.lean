import Mathlib

namespace NUMINAMATH_CALUDE_baseball_team_groups_l3450_345057

/-- The number of groups formed from new and returning players -/
def number_of_groups (new_players returning_players players_per_group : ℕ) : ℕ :=
  (new_players + returning_players) / players_per_group

/-- Theorem stating that the number of groups is 9 given the specific conditions -/
theorem baseball_team_groups : number_of_groups 48 6 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_baseball_team_groups_l3450_345057


namespace NUMINAMATH_CALUDE_special_sequence_remainder_l3450_345067

def sequence_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n < 69 → 3 * a n = a (n - 1) + a (n + 1)

theorem special_sequence_remainder :
  ∀ a : ℕ → ℤ,
  sequence_condition a →
  a 0 = 0 →
  a 1 = 1 →
  a 2 = 3 →
  a 3 = 8 →
  a 4 = 21 →
  ∃ k : ℤ, a 69 = 6 * k + 4 :=
by sorry

end NUMINAMATH_CALUDE_special_sequence_remainder_l3450_345067


namespace NUMINAMATH_CALUDE_fourth_term_is_eight_l3450_345028

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum function
  first_term : a 1 = -1
  sum_property : ∀ n, S n = n * (a 1 + a n) / 2
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The theorem stating that a_4 = 8 given the conditions -/
theorem fourth_term_is_eight (seq : ArithmeticSequence) (sum_4 : seq.S 4 = 14) :
  seq.a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_eight_l3450_345028


namespace NUMINAMATH_CALUDE_additional_flies_needed_l3450_345068

/-- Represents the number of flies eaten by the frog each day of the week -/
def flies_eaten_per_day : List Nat := [3, 2, 4, 5, 1, 2, 3]

/-- Calculates the total number of flies eaten in a week -/
def total_flies_needed : Nat := flies_eaten_per_day.sum

/-- Number of flies Betty caught in the morning -/
def morning_catch : Nat := 5

/-- Number of flies Betty caught in the afternoon -/
def afternoon_catch : Nat := 6

/-- Number of flies that escaped -/
def escaped_flies : Nat := 1

/-- Calculates the total number of flies Betty successfully caught -/
def total_flies_caught : Nat := morning_catch + afternoon_catch - escaped_flies

/-- Theorem stating the number of additional flies Betty needs -/
theorem additional_flies_needed : 
  total_flies_needed - total_flies_caught = 10 := by sorry

end NUMINAMATH_CALUDE_additional_flies_needed_l3450_345068


namespace NUMINAMATH_CALUDE_custom_op_solution_l3450_345021

/-- Custom operation defined for integers -/
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

/-- Theorem stating that given the custom operation, if 11b = 110, then b = 12 -/
theorem custom_op_solution :
  ∀ b : ℤ, customOp 11 b = 110 → b = 12 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_solution_l3450_345021


namespace NUMINAMATH_CALUDE_unique_real_solution_l3450_345076

theorem unique_real_solution (a : ℝ) :
  (∃! x : ℝ, x^3 - a*x^2 - 2*a*x + a^2 - 1 = 0) ↔ a < 3/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_real_solution_l3450_345076


namespace NUMINAMATH_CALUDE_cost_price_per_meter_is_58_l3450_345080

/-- Calculates the cost price per meter of cloth given the total length,
    total selling price, and profit per meter. -/
def costPricePerMeter (totalLength : ℕ) (totalSellingPrice : ℕ) (profitPerMeter : ℕ) : ℕ :=
  (totalSellingPrice - totalLength * profitPerMeter) / totalLength

/-- Proves that the cost price per meter of cloth is 58 rupees given the
    specified conditions. -/
theorem cost_price_per_meter_is_58 :
  costPricePerMeter 78 6788 29 = 58 := by sorry

end NUMINAMATH_CALUDE_cost_price_per_meter_is_58_l3450_345080


namespace NUMINAMATH_CALUDE_rhombus_area_l3450_345069

/-- The area of a rhombus with side length 4 cm and an acute angle of 45° is 8 cm². -/
theorem rhombus_area (side_length : ℝ) (acute_angle : ℝ) : 
  side_length = 4 → acute_angle = π / 4 → 
  (side_length * side_length * Real.sin acute_angle) = 8 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3450_345069


namespace NUMINAMATH_CALUDE_radius_difference_is_zero_l3450_345051

/-- A circle with center C tangent to positive x and y-axes and externally tangent to another circle -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  tangent_to_axes : center.1 = radius ∧ center.2 = radius
  externally_tangent : (radius - 2)^2 + radius^2 = (radius + 2)^2

/-- The radius difference between the largest and smallest possible radii is 0 -/
theorem radius_difference_is_zero : 
  ∀ (c₁ c₂ : TangentCircle), c₁.radius - c₂.radius = 0 := by
  sorry

end NUMINAMATH_CALUDE_radius_difference_is_zero_l3450_345051


namespace NUMINAMATH_CALUDE_winning_candidate_percentage_l3450_345003

/-- Given an election with two candidates, prove that the winning candidate
    received 60% of the votes under the given conditions. -/
theorem winning_candidate_percentage
  (total_votes : ℕ)
  (winning_margin : ℕ)
  (h_total : total_votes = 1400)
  (h_margin : winning_margin = 280) :
  (winning_votes : ℕ) →
  (losing_votes : ℕ) →
  (winning_votes + losing_votes = total_votes) →
  (winning_votes = losing_votes + winning_margin) →
  (winning_votes : ℚ) / total_votes = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_winning_candidate_percentage_l3450_345003


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_implies_a_equals_one_l3450_345072

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem fourth_term_coefficient_implies_a_equals_one (x a : ℝ) :
  (binomial 9 3 : ℝ) * a^3 = 84 → a = 1 := by sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_implies_a_equals_one_l3450_345072


namespace NUMINAMATH_CALUDE_factory_weekly_production_l3450_345009

/-- Represents a toy factory with its production characteristics -/
structure ToyFactory where
  daysPerWeek : ℕ
  dailyProduction : ℕ
  constDailyProduction : Bool

/-- Calculates the weekly production of toys for a given factory -/
def weeklyProduction (factory : ToyFactory) : ℕ :=
  factory.daysPerWeek * factory.dailyProduction

/-- Theorem: The weekly production of the given factory is 6500 toys -/
theorem factory_weekly_production :
  ∀ (factory : ToyFactory),
    factory.daysPerWeek = 5 →
    factory.dailyProduction = 1300 →
    factory.constDailyProduction = true →
    weeklyProduction factory = 6500 := by
  sorry

end NUMINAMATH_CALUDE_factory_weekly_production_l3450_345009


namespace NUMINAMATH_CALUDE_min_value_expression_l3450_345016

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5/3) :
  4 / (a + 2*b) + 9 / (2*a + b) ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3450_345016


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l3450_345022

/-- Given a hyperbola with equation x^2 - y^2 = 3, 
    if k₁ and k₂ are the slopes of its two asymptotes, 
    then k₁k₂ = -1 -/
theorem hyperbola_asymptote_slopes (k₁ k₂ : ℝ) : 
  (∀ x y : ℝ, x^2 - y^2 = 3 → 
    (∃ a b : ℝ, (y = k₁ * x + a ∨ y = k₂ * x + b) ∧ 
      (∀ ε > 0, ∃ x₀ > 0, ∀ x > x₀, 
        |y - k₁ * x| < ε ∨ |y - k₂ * x| < ε))) →
  k₁ * k₂ = -1 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l3450_345022


namespace NUMINAMATH_CALUDE_circle_equation_from_chord_l3450_345007

/-- Given a circle with center at the origin and a chord of length 8 cut by the line 3x + 4y + 15 = 0,
    prove that the equation of the circle is x^2 + y^2 = 25 -/
theorem circle_equation_from_chord (x y : ℝ) :
  let center := (0 : ℝ × ℝ)
  let chord_line := {(x, y) | 3 * x + 4 * y + 15 = 0}
  let chord_length := 8
  ∃ (r : ℝ), r > 0 ∧
    (∀ (p : ℝ × ℝ), p ∈ chord_line → dist center p ≤ r) ∧
    (∃ (p q : ℝ × ℝ), p ∈ chord_line ∧ q ∈ chord_line ∧ p ≠ q ∧ dist p q = chord_length) →
  x^2 + y^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_from_chord_l3450_345007


namespace NUMINAMATH_CALUDE_factorial_ratio_l3450_345049

theorem factorial_ratio : Nat.factorial 10 / (Nat.factorial 4 * Nat.factorial 6) = 210 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l3450_345049


namespace NUMINAMATH_CALUDE_sin_has_property_T_l3450_345050

-- Define property T
def has_property_T (f : ℝ → ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (deriv f x₁) * (deriv f x₂) = -1

-- State the theorem
theorem sin_has_property_T :
  has_property_T Real.sin :=
sorry

end NUMINAMATH_CALUDE_sin_has_property_T_l3450_345050


namespace NUMINAMATH_CALUDE_max_area_inscribed_equilateral_triangle_l3450_345025

/-- The maximum area of an equilateral triangle inscribed in a 13x14 rectangle --/
theorem max_area_inscribed_equilateral_triangle :
  ∃ (A : ℝ),
    A = (183 : ℝ) * Real.sqrt 3 ∧
    ∀ (s : ℝ),
      0 ≤ s →
      s ≤ 13 →
      s * Real.sqrt 3 / 2 ≤ 14 →
      s^2 * Real.sqrt 3 / 4 ≤ A :=
by sorry

#eval (183 : Nat) + 3 + 0

end NUMINAMATH_CALUDE_max_area_inscribed_equilateral_triangle_l3450_345025


namespace NUMINAMATH_CALUDE_union_equals_reals_l3450_345073

-- Define the sets E and F
def E : Set ℝ := {x | x^2 - 5*x - 6 > 0}
def F (a : ℝ) : Set ℝ := {x | x - 5 < a}

-- State the theorem
theorem union_equals_reals (a : ℝ) (h : (11 : ℝ) ∈ F a) : E ∪ F a = Set.univ := by
  sorry

end NUMINAMATH_CALUDE_union_equals_reals_l3450_345073


namespace NUMINAMATH_CALUDE_cafeteria_combos_l3450_345064

/-- Represents the number of options for each part of the lunch combo -/
structure LunchOptions where
  mainDishes : Nat
  sides : Nat
  drinks : Nat
  desserts : Nat

/-- Calculates the total number of distinct lunch combos -/
def totalCombos (options : LunchOptions) : Nat :=
  options.mainDishes * options.sides * options.drinks * options.desserts

/-- The specific lunch options available in the cafeteria -/
def cafeteriaOptions : LunchOptions :=
  { mainDishes := 3
  , sides := 2
  , drinks := 2
  , desserts := 2 }

theorem cafeteria_combos :
  totalCombos cafeteriaOptions = 24 := by
  sorry

#eval totalCombos cafeteriaOptions

end NUMINAMATH_CALUDE_cafeteria_combos_l3450_345064


namespace NUMINAMATH_CALUDE_sum_of_squared_sums_of_roots_l3450_345006

theorem sum_of_squared_sums_of_roots (p q r : ℝ) : 
  (p^3 - 15*p^2 + 25*p - 10 = 0) → 
  (q^3 - 15*q^2 + 25*q - 10 = 0) → 
  (r^3 - 15*r^2 + 25*r - 10 = 0) → 
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 400 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squared_sums_of_roots_l3450_345006


namespace NUMINAMATH_CALUDE_subgroup_equality_l3450_345029

variable {G : Type*} [Group G]

theorem subgroup_equality (S : Set G) (x s : G) (hs : s ∈ Subgroup.closure S) :
  Subgroup.closure (S ∪ {x}) = Subgroup.closure (S ∪ {x * s}) ∧
  Subgroup.closure (S ∪ {x}) = Subgroup.closure (S ∪ {s * x}) := by
  sorry

end NUMINAMATH_CALUDE_subgroup_equality_l3450_345029


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l3450_345008

theorem geometric_progression_ratio (a b c d x y z r : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 → x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x ≠ y → y ≠ z → x ≠ z →
  a * x * (y - z) ≠ 0 →
  b * y * (z - x) ≠ 0 →
  c * z * (x - y) ≠ 0 →
  d * x * (y - z) ≠ 0 →
  a * x * (y - z) ≠ b * y * (z - x) →
  b * y * (z - x) ≠ c * z * (x - y) →
  c * z * (x - y) ≠ d * x * (y - z) →
  (∃ k : ℝ, k ≠ 0 ∧ 
    b * y * (z - x) = k * (a * x * (y - z)) ∧
    c * z * (x - y) = k * (b * y * (z - x)) ∧
    d * x * (y - z) = k * (c * z * (x - y))) →
  r^3 + r^2 + r + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l3450_345008


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3450_345089

/-- For a parabola with equation x^2 = (1/2)y, the distance from its focus to its directrix is 1/4 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), x^2 = (1/2) * y → 
  ∃ (focus_x focus_y directrix_y : ℝ),
    (focus_x = 0 ∧ 
     focus_y = 1/8 ∧ 
     directrix_y = -1/8 ∧
     focus_y - directrix_y = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l3450_345089


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3450_345005

/-- Given two 2D vectors a and b, where a is parallel to b, prove that m = -1 --/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (1, 2) →
  b = (2, 3 - m) →
  (∃ (k : ℝ), a = k • b) →
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3450_345005


namespace NUMINAMATH_CALUDE_equation_equivalence_l3450_345042

theorem equation_equivalence (x y : ℝ) 
  (hx : x ≠ 0 ∧ x ≠ 5) (hy : y ≠ 0 ∧ y ≠ 7) : 
  (3 / x + 2 / y = 1 / 3) ↔ (x = 9 * y / (y - 6)) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3450_345042


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3450_345014

theorem complex_number_quadrant : ∃ (z : ℂ), 
  (z + Complex.I) * (1 - 2 * Complex.I) = 2 ∧ 
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3450_345014


namespace NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l3450_345047

theorem cos_36_minus_cos_72_eq_half :
  Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_36_minus_cos_72_eq_half_l3450_345047


namespace NUMINAMATH_CALUDE_rectangle_ordering_l3450_345058

-- Define a rectangle in a Cartesian plane
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

-- Define the "preferable" relation
def preferable (a b : Rectangle) : Prop :=
  (a.x_max ≤ b.x_min) ∨ (a.y_max ≤ b.y_min)

-- Main theorem
theorem rectangle_ordering {n : ℕ} (rectangles : Fin n → Rectangle) 
  (h_nonoverlap : ∀ i j, i ≠ j → 
    (rectangles i).x_max ≤ (rectangles j).x_min ∨
    (rectangles j).x_max ≤ (rectangles i).x_min ∨
    (rectangles i).y_max ≤ (rectangles j).y_min ∨
    (rectangles j).y_max ≤ (rectangles i).y_min) :
  ∃ (σ : Equiv.Perm (Fin n)), ∀ i j, i < j → 
    preferable (rectangles (σ i)) (rectangles (σ j)) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ordering_l3450_345058


namespace NUMINAMATH_CALUDE_exists_n_divides_1991_l3450_345004

theorem exists_n_divides_1991 : ∃ n : ℕ, n > 2 ∧ (2 * 10^(n+1) - 9) % 1991 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_n_divides_1991_l3450_345004


namespace NUMINAMATH_CALUDE_tree_cutting_percentage_l3450_345055

theorem tree_cutting_percentage (initial_trees : ℕ) (final_trees : ℕ) (replant_rate : ℕ) : 
  initial_trees = 400 → 
  final_trees = 720 → 
  replant_rate = 5 → 
  (100 * (final_trees - initial_trees)) / (initial_trees * (replant_rate - 1)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tree_cutting_percentage_l3450_345055


namespace NUMINAMATH_CALUDE_fraction_of_ripe_oranges_eaten_l3450_345046

def total_oranges : ℕ := 96
def ripe_oranges : ℕ := total_oranges / 2
def unripe_oranges : ℕ := total_oranges - ripe_oranges
def eaten_unripe : ℕ := unripe_oranges / 8
def uneaten_oranges : ℕ := 78

theorem fraction_of_ripe_oranges_eaten :
  (total_oranges - uneaten_oranges - eaten_unripe) / ripe_oranges = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_fraction_of_ripe_oranges_eaten_l3450_345046


namespace NUMINAMATH_CALUDE_order_of_surds_l3450_345077

theorem order_of_surds : 
  let a : ℝ := Real.sqrt 5 - Real.sqrt 3
  let b : ℝ := Real.sqrt 3 - 1
  let c : ℝ := Real.sqrt 7 - Real.sqrt 5
  b > a ∧ a > c := by sorry

end NUMINAMATH_CALUDE_order_of_surds_l3450_345077


namespace NUMINAMATH_CALUDE_binomial_12_9_l3450_345031

theorem binomial_12_9 : Nat.choose 12 9 = 220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_12_9_l3450_345031


namespace NUMINAMATH_CALUDE_total_cakes_served_l3450_345044

/-- The number of cakes served on Sunday -/
def sunday_cakes : ℕ := 3

/-- The number of cakes served during lunch on Monday -/
def monday_lunch_cakes : ℕ := 5

/-- The number of cakes served during dinner on Monday -/
def monday_dinner_cakes : ℕ := 6

/-- The number of cakes thrown away on Tuesday -/
def tuesday_thrown_cakes : ℕ := 4

/-- The total number of cakes served on Monday -/
def monday_total_cakes : ℕ := monday_lunch_cakes + monday_dinner_cakes

/-- The number of cakes initially prepared for Tuesday (before throwing away) -/
def tuesday_initial_cakes : ℕ := 2 * monday_total_cakes

/-- The total number of cakes served on Tuesday after throwing away some -/
def tuesday_final_cakes : ℕ := tuesday_initial_cakes - tuesday_thrown_cakes

/-- Theorem stating that the total number of cakes served over three days is 32 -/
theorem total_cakes_served : sunday_cakes + monday_total_cakes + tuesday_final_cakes = 32 := by
  sorry

end NUMINAMATH_CALUDE_total_cakes_served_l3450_345044


namespace NUMINAMATH_CALUDE_complex_magnitude_from_equation_l3450_345017

theorem complex_magnitude_from_equation (z : ℂ) : 
  Complex.I * (1 - z) = 1 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_from_equation_l3450_345017


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l3450_345019

theorem distance_to_y_axis (x : ℝ) :
  let P : ℝ × ℝ := (x, -5)
  (|P.2| = (1/2 : ℝ) * |P.1|) → |P.1| = 10 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l3450_345019


namespace NUMINAMATH_CALUDE_parametric_represents_curve_l3450_345098

-- Define the curve
def curve (x : ℝ) : ℝ := x^2

-- Define the parametric equations
def parametric_x (t : ℝ) : ℝ := t
def parametric_y (t : ℝ) : ℝ := t^2

-- Theorem statement
theorem parametric_represents_curve :
  ∀ (t : ℝ), curve (parametric_x t) = parametric_y t :=
sorry

end NUMINAMATH_CALUDE_parametric_represents_curve_l3450_345098


namespace NUMINAMATH_CALUDE_pedestrian_cyclist_speed_problem_l3450_345087

/-- The problem setup and solution for the pedestrian and cyclist speed problem -/
theorem pedestrian_cyclist_speed_problem :
  let distance : ℝ := 40 -- km
  let pedestrian_start_time : ℝ := 0 -- 4:00 AM
  let first_cyclist_start_time : ℝ := 3 + 1/3 -- 7:20 AM
  let second_cyclist_start_time : ℝ := 4.5 -- 8:30 AM
  let meetup_distance : ℝ := distance / 2
  let second_meetup_time_diff : ℝ := 1 -- hour

  ∃ (pedestrian_speed cyclist_speed : ℝ),
    pedestrian_speed > 0 ∧
    cyclist_speed > 0 ∧
    pedestrian_speed < cyclist_speed ∧
    -- First cyclist catches up with pedestrian at midpoint
    meetup_distance = pedestrian_speed * (first_cyclist_start_time + 
      (meetup_distance - pedestrian_speed * first_cyclist_start_time) / (cyclist_speed - pedestrian_speed)) ∧
    -- Second cyclist meets pedestrian one hour after first meetup
    distance = pedestrian_speed * (second_cyclist_start_time + 
      (meetup_distance - pedestrian_speed * first_cyclist_start_time) / (cyclist_speed - pedestrian_speed) + 
      second_meetup_time_diff) + 
      cyclist_speed * ((distance - meetup_distance) / cyclist_speed - 
      ((meetup_distance - pedestrian_speed * first_cyclist_start_time) / (cyclist_speed - pedestrian_speed) + 
      second_meetup_time_diff)) ∧
    pedestrian_speed = 5 ∧
    cyclist_speed = 30
  := by sorry

end NUMINAMATH_CALUDE_pedestrian_cyclist_speed_problem_l3450_345087


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l3450_345053

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_special_case
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_roots : a 2 * a 6 = 81 ∧ a 2 + a 6 = 34) :
  a 4 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l3450_345053


namespace NUMINAMATH_CALUDE_characterize_valid_k_l3450_345099

/-- A coloring of the complete graph on n vertices using k colors -/
def GraphColoring (n : ℕ) (k : ℕ) := Fin n → Fin n → Fin k

/-- Property: for any k vertices, all edges between them have different colors -/
def HasUniqueColors (n : ℕ) (k : ℕ) (coloring : GraphColoring n k) : Prop :=
  ∀ (vertices : Finset (Fin n)), vertices.card = k →
    (∀ (i j : Fin n), i ∈ vertices → j ∈ vertices → i ≠ j →
      ∀ (x y : Fin n), x ∈ vertices → y ∈ vertices → x ≠ y → (x, y) ≠ (i, j) →
        coloring i j ≠ coloring x y)

/-- The set of valid k values for a 10-vertex graph -/
def ValidK : Set ℕ := {k | k ≥ 5 ∧ k ≤ 10}

/-- Main theorem: characterization of valid k for a 10-vertex graph -/
theorem characterize_valid_k :
  ∀ k, k ∈ ValidK ↔ ∃ (coloring : GraphColoring 10 k), HasUniqueColors 10 k coloring :=
sorry

end NUMINAMATH_CALUDE_characterize_valid_k_l3450_345099


namespace NUMINAMATH_CALUDE_unique_pricing_l3450_345082

/-- Represents the price of a sewage treatment equipment model in thousand dollars. -/
structure ModelPrice where
  price : ℝ

/-- Represents the pricing of two sewage treatment equipment models A and B. -/
structure EquipmentPricing where
  modelA : ModelPrice
  modelB : ModelPrice

/-- Checks if the given equipment pricing satisfies the problem conditions. -/
def satisfiesConditions (pricing : EquipmentPricing) : Prop :=
  pricing.modelA.price = pricing.modelB.price + 5 ∧
  2 * pricing.modelA.price + 3 * pricing.modelB.price = 45

/-- Theorem stating that the only pricing satisfying the conditions is A at 12 and B at 7. -/
theorem unique_pricing :
  ∀ (pricing : EquipmentPricing),
    satisfiesConditions pricing →
    pricing.modelA.price = 12 ∧ pricing.modelB.price = 7 := by
  sorry

end NUMINAMATH_CALUDE_unique_pricing_l3450_345082


namespace NUMINAMATH_CALUDE_central_cell_only_solution_l3450_345039

/-- Represents a 5x5 grid with boolean values (true for "+", false for "-") -/
def Grid := Fin 5 → Fin 5 → Bool

/-- Represents a subgrid position and size -/
structure Subgrid where
  row : Fin 5
  col : Fin 5
  size : Nat
  size_valid : 2 ≤ size ∧ size ≤ 5

/-- Flips the signs in a subgrid -/
def flip_subgrid (g : Grid) (sg : Subgrid) : Grid :=
  λ i j => if i < sg.row + sg.size ∧ j < sg.col + sg.size
           then !g i j
           else g i j

/-- Checks if all cells in the grid are positive -/
def all_positive (g : Grid) : Prop :=
  ∀ i j, g i j = true

/-- Initial grid with only the specified cell negative -/
def initial_grid (row col : Fin 5) : Grid :=
  λ i j => ¬(i = row ∧ j = col)

/-- Theorem stating that only the central cell as initial negative allows for all positive end state -/
theorem central_cell_only_solution :
  ∀ (row col : Fin 5),
    (∃ (moves : List Subgrid), all_positive (moves.foldl flip_subgrid (initial_grid row col))) ↔
    (row = 2 ∧ col = 2) :=
  sorry

end NUMINAMATH_CALUDE_central_cell_only_solution_l3450_345039


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3450_345034

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 - I) / (1 + I^2023)
  (z.re > 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3450_345034


namespace NUMINAMATH_CALUDE_election_winner_votes_l3450_345091

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (3 * total_votes) / 4 - total_votes / 4 = 500) : 
  (3 * total_votes) / 4 = 750 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_votes_l3450_345091


namespace NUMINAMATH_CALUDE_smallest_n_value_l3450_345070

def count_factors_of_five (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem smallest_n_value (a b c m n : ℕ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 2010 →
  c = 710 →
  a.factorial * b.factorial * c.factorial = m * (10 ^ n) →
  ¬(10 ∣ m) →
  (∀ k, k < n → ∃ p, p > 0 ∧ ¬(10 ∣ p) ∧ 
    a.factorial * b.factorial * c.factorial ≠ p * (10 ^ k)) →
  n = 500 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_value_l3450_345070


namespace NUMINAMATH_CALUDE_unique_integer_perfect_square_l3450_345075

theorem unique_integer_perfect_square : 
  ∃! x : ℤ, ∃ y : ℤ, x^4 + 8*x^3 + 18*x^2 + 8*x + 36 = y^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_perfect_square_l3450_345075


namespace NUMINAMATH_CALUDE_max_value_expression_l3450_345061

def A : Set Int := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

theorem max_value_expression (v w x y z : Int) 
  (hv : v ∈ A) (hw : w ∈ A) (hx : x ∈ A) (hy : y ∈ A) (hz : z ∈ A)
  (h_vw : v * w = x) (h_w : w ≠ 0) :
  (∀ v' w' x' y' z' : Int, 
    v' ∈ A → w' ∈ A → x' ∈ A → y' ∈ A → z' ∈ A → 
    v' * w' = x' → w' ≠ 0 →
    v * x - y * z ≥ v' * x' - y' * z') →
  v * x - y * z = 150 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3450_345061


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3450_345083

theorem purely_imaginary_complex_number (m : ℝ) : 
  (m^2 - m : ℂ) + m*I = (0 : ℝ) + (m^2 : ℂ)*I → m = 1 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3450_345083


namespace NUMINAMATH_CALUDE_shaded_area_percentage_l3450_345000

/-- Given two congruent squares with side length 12 that overlap to form a 12 by 20 rectangle,
    prove that 20% of the rectangle's area is shaded. -/
theorem shaded_area_percentage (side_length : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) :
  side_length = 12 →
  rectangle_width = 12 →
  rectangle_length = 20 →
  (side_length * side_length - rectangle_width * rectangle_length) / (rectangle_width * rectangle_length) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_percentage_l3450_345000


namespace NUMINAMATH_CALUDE_power_calculation_l3450_345066

theorem power_calculation : ((13^13 / 13^12)^3 * 3^3) / 3^6 = 27 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l3450_345066


namespace NUMINAMATH_CALUDE_sqrt45_same_type_as_sqrt5_l3450_345081

-- Define the property of being "of the same type as √5"
def same_type_as_sqrt5 (x : ℝ) : Prop :=
  ∃ (k : ℝ), x = k * Real.sqrt 5

-- State the theorem
theorem sqrt45_same_type_as_sqrt5 :
  same_type_as_sqrt5 (Real.sqrt 45) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt45_same_type_as_sqrt5_l3450_345081


namespace NUMINAMATH_CALUDE_binary_sum_equals_318_l3450_345040

/-- Convert a binary number represented as a string to its decimal equivalent -/
def binary_to_decimal (s : String) : ℕ :=
  s.foldl (fun acc c => 2 * acc + c.toString.toNat!) 0

/-- The sum of 11111111₂ and 111111₂ in base 10 -/
theorem binary_sum_equals_318 :
  binary_to_decimal "11111111" + binary_to_decimal "111111" = 318 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_318_l3450_345040


namespace NUMINAMATH_CALUDE_percentage_difference_l3450_345037

theorem percentage_difference (p q : ℝ) (h : q = 0.8 * p) : p = 1.25 * q := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3450_345037


namespace NUMINAMATH_CALUDE_log_product_theorem_l3450_345043

theorem log_product_theorem (c d : ℕ+) : 
  (d.val - c.val = 435) → 
  (Real.log d.val / Real.log c.val = 2) → 
  (c.val + d.val = 930) := by
sorry

end NUMINAMATH_CALUDE_log_product_theorem_l3450_345043


namespace NUMINAMATH_CALUDE_female_officers_count_l3450_345045

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 500 →
  female_on_duty_ratio = 1/4 →
  female_ratio = 1/2 →
  (female_on_duty_ratio * (total_on_duty * female_ratio)) / female_on_duty_ratio = 1000 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l3450_345045


namespace NUMINAMATH_CALUDE_right_handed_players_count_l3450_345030

theorem right_handed_players_count (total_players throwers : ℕ) 
  (h1 : total_players = 120)
  (h2 : throwers = 45)
  (h3 : throwers ≤ total_players)
  (h4 : 5 * (total_players - throwers) % 5 = 0) -- Ensures divisibility by 5
  : (throwers + (3 * (total_players - throwers) / 5) : ℕ) = 90 := by
  sorry

end NUMINAMATH_CALUDE_right_handed_players_count_l3450_345030


namespace NUMINAMATH_CALUDE_number_problem_l3450_345010

theorem number_problem (x : ℝ) : 0.3 * x = 0.6 * 150 + 120 → x = 700 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3450_345010


namespace NUMINAMATH_CALUDE_bertha_age_difference_l3450_345084

structure Grandparents where
  arthur : ℕ
  bertha : ℕ
  christoph : ℕ
  dolores : ℕ

def is_valid_grandparents (g : Grandparents) : Prop :=
  (max g.arthur (max g.bertha (max g.christoph g.dolores))) - 
  (min g.arthur (min g.bertha (min g.christoph g.dolores))) = 4 ∧
  g.arthur = g.bertha + 2 ∧
  g.christoph = g.dolores + 2 ∧
  g.bertha < g.dolores

theorem bertha_age_difference (g : Grandparents) (h : is_valid_grandparents g) :
  g.bertha + 2 = (g.arthur + g.bertha + g.christoph + g.dolores) / 4 := by
  sorry

#check bertha_age_difference

end NUMINAMATH_CALUDE_bertha_age_difference_l3450_345084


namespace NUMINAMATH_CALUDE_pattern_1010_is_BCDA_l3450_345093

/-- Represents the four vertices of a square -/
inductive Vertex
| A
| B
| C
| D

/-- Represents a square configuration -/
def Square := List Vertex

/-- The initial square configuration -/
def initial_square : Square := [Vertex.A, Vertex.B, Vertex.C, Vertex.D]

/-- Performs a 90-degree counterclockwise rotation on a square -/
def rotate (s : Square) : Square := 
  match s with
  | [a, b, c, d] => [b, c, d, a]
  | _ => s

/-- Reflects a square over its horizontal line of symmetry -/
def reflect (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [d, c, b, a]
  | _ => s

/-- Applies the alternating pattern of rotation and reflection n times -/
def apply_pattern (s : Square) (n : Nat) : Square :=
  match n with
  | 0 => s
  | n + 1 => if n % 2 == 0 then rotate (apply_pattern s n) else reflect (apply_pattern s n)

theorem pattern_1010_is_BCDA : 
  apply_pattern initial_square 1010 = [Vertex.B, Vertex.C, Vertex.D, Vertex.A] := by
  sorry

end NUMINAMATH_CALUDE_pattern_1010_is_BCDA_l3450_345093


namespace NUMINAMATH_CALUDE_equal_solution_is_two_l3450_345063

/-- Given a system of equations for nonnegative real numbers, prove that the only solution where all numbers are equal is 2. -/
theorem equal_solution_is_two (n : ℕ) (x : ℕ → ℝ) : 
  n > 2 →
  (∀ k, k ∈ Finset.range n → x k ≥ 0) →
  (∀ k, k ∈ Finset.range n → x k + x ((k + 1) % n) = (x ((k + 2) % n))^2) →
  (∀ i j, i ∈ Finset.range n → j ∈ Finset.range n → x i = x j) →
  (∀ k, k ∈ Finset.range n → x k = 2) := by
sorry

end NUMINAMATH_CALUDE_equal_solution_is_two_l3450_345063


namespace NUMINAMATH_CALUDE_power_equality_l3450_345059

theorem power_equality (x : ℝ) : (1/4 : ℝ) * (2^32) = 4^x → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3450_345059


namespace NUMINAMATH_CALUDE_exists_multiple_factorization_l3450_345024

/-- The set Vn for a given n > 2 -/
def Vn (n : ℕ) : Set ℕ :=
  {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

/-- A number is indecomposable in Vn if it cannot be expressed as a product of two elements in Vn -/
def Indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ Vn n ∧ ¬∃ p q : ℕ, p ∈ Vn n ∧ q ∈ Vn n ∧ p * q = m

/-- The main theorem statement -/
theorem exists_multiple_factorization (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ Vn n ∧
    ∃ (f₁ f₂ : List ℕ),
      f₁ ≠ f₂ ∧
      (∀ x ∈ f₁, Indecomposable n x) ∧
      (∀ x ∈ f₂, Indecomposable n x) ∧
      r = f₁.prod ∧
      r = f₂.prod :=
sorry

end NUMINAMATH_CALUDE_exists_multiple_factorization_l3450_345024


namespace NUMINAMATH_CALUDE_cosine_sum_equality_l3450_345011

theorem cosine_sum_equality : 
  Real.cos (15 * π / 180) * Real.cos (30 * π / 180) + 
  Real.cos (105 * π / 180) * Real.sin (30 * π / 180) = 
  Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cosine_sum_equality_l3450_345011


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3450_345012

def repeating_decimal_3 : ℚ := 1/3
def repeating_decimal_6 : ℚ := 2/3

theorem sum_of_repeating_decimals : 
  repeating_decimal_3 + repeating_decimal_6 = 1 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3450_345012


namespace NUMINAMATH_CALUDE_boats_geometric_sum_l3450_345020

def geometric_sum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem boats_geometric_sum :
  geometric_sum 5 3 5 = 605 := by
  sorry

end NUMINAMATH_CALUDE_boats_geometric_sum_l3450_345020


namespace NUMINAMATH_CALUDE_price_markup_markdown_l3450_345002

theorem price_markup_markdown (x : ℝ) (h : x > 0) : x * (1 + 0.1) * (1 - 0.1) < x := by
  sorry

end NUMINAMATH_CALUDE_price_markup_markdown_l3450_345002


namespace NUMINAMATH_CALUDE_ladder_problem_l3450_345052

theorem ladder_problem (c a b : ℝ) : 
  c = 25 → a = 15 → c^2 = a^2 + b^2 → b = 20 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l3450_345052


namespace NUMINAMATH_CALUDE_probability_both_slate_is_correct_l3450_345094

def slate_rocks : ℕ := 14
def pumice_rocks : ℕ := 20
def granite_rocks : ℕ := 10

def total_rocks : ℕ := slate_rocks + pumice_rocks + granite_rocks

def probability_both_slate : ℚ := (slate_rocks : ℚ) / total_rocks * ((slate_rocks - 1) : ℚ) / (total_rocks - 1)

theorem probability_both_slate_is_correct : probability_both_slate = 13 / 1892 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_slate_is_correct_l3450_345094


namespace NUMINAMATH_CALUDE_triangle_max_area_l3450_345036

/-- Given a triangle ABC with sides a, b, c and angles A, B, C opposite to them respectively,
    prove that if b(2-cos A) = a(cos B+1) and a + c = 4, then the maximum area of the triangle is √3. -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (b * (2 - Real.cos A) = a * (Real.cos B + 1)) →
  (a + c = 4) →
  (∃ (S : ℝ), S = (1/2) * a * c * Real.sin B ∧ 
   ∀ (S' : ℝ), S' = (1/2) * a * c * Real.sin B → S' ≤ S) →
  S = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3450_345036


namespace NUMINAMATH_CALUDE_arc_length_for_120_degrees_l3450_345033

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the central angle in degrees
def central_angle : ℝ := 120

-- Define pi as a real number (since Lean doesn't have a built-in pi constant)
noncomputable def π : ℝ := Real.pi

-- State the theorem
theorem arc_length_for_120_degrees (r : ℝ) (θ : ℝ) :
  r = radius → θ = central_angle →
  (θ / 360) * (2 * π * r) = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_arc_length_for_120_degrees_l3450_345033


namespace NUMINAMATH_CALUDE_system_no_solution_implies_m_equals_two_l3450_345015

/-- Represents a 2x3 augmented matrix -/
structure AugmentedMatrix (α : Type*) :=
  (a11 a12 a13 a21 a22 a23 : α)

/-- Checks if the given augmented matrix represents a system with no real solution -/
def has_no_real_solution (A : AugmentedMatrix ℝ) : Prop :=
  ∀ x y : ℝ, A.a11 * x + A.a12 * y ≠ A.a13 ∨ A.a21 * x + A.a22 * y ≠ A.a23

theorem system_no_solution_implies_m_equals_two :
  ∀ m : ℝ, 
    let A : AugmentedMatrix ℝ := ⟨m, 4, m + 2, 1, m, m⟩
    has_no_real_solution A → m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_system_no_solution_implies_m_equals_two_l3450_345015


namespace NUMINAMATH_CALUDE_sqrt_x_minus_9_meaningful_l3450_345048

theorem sqrt_x_minus_9_meaningful (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 9) ↔ x ≥ 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_9_meaningful_l3450_345048


namespace NUMINAMATH_CALUDE_phone_bill_ratio_l3450_345001

theorem phone_bill_ratio (jan_total feb_total internet_charge : ℚ)
  (h1 : jan_total = 46)
  (h2 : feb_total = 76)
  (h3 : internet_charge = 16) :
  (feb_total - internet_charge) / (jan_total - internet_charge) = 2 := by
  sorry

end NUMINAMATH_CALUDE_phone_bill_ratio_l3450_345001


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3450_345092

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The common difference of an arithmetic sequence -/
def CommonDifference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_a7 : a 7 = 25)
  (h_a4 : a 4 = 13) :
  ∃ d : ℝ, CommonDifference a d ∧ d = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3450_345092


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l3450_345032

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 3 * x + y + a = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem line_passes_through_circle_center (a : ℝ) :
  line_equation (circle_center.1) (circle_center.2) a →
  circle_equation (circle_center.1) (circle_center.2) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l3450_345032


namespace NUMINAMATH_CALUDE_company_fund_problem_l3450_345054

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (initial_fund = 60 * n - 10) ∧ 
  (initial_fund = 50 * n + 150) → 
  initial_fund = 950 :=
sorry

end NUMINAMATH_CALUDE_company_fund_problem_l3450_345054


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3450_345056

/-- Given a geometric sequence {a_n} with common ratio q = 2,
    if the arithmetic mean of a_2 and 2a_3 is 5, then a_1 = 1 -/
theorem geometric_sequence_first_term
  (a : ℕ → ℝ)  -- a is the sequence
  (h_geom : ∀ n, a (n + 1) = 2 * a n)  -- geometric sequence with ratio 2
  (h_mean : (a 2 + 2 * a 3) / 2 = 5)  -- arithmetic mean condition
  : a 1 = 1 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3450_345056


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3450_345074

theorem inequality_solution_set 
  (a b : ℝ) (ha : a < 0) : 
  {x : ℝ | a * x + b < 0} = {x : ℝ | x > -b / a} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3450_345074


namespace NUMINAMATH_CALUDE_theater_lost_revenue_l3450_345090

/-- Calculates the lost revenue for a movie theater given its capacity, ticket price, and actual tickets sold. -/
theorem theater_lost_revenue (capacity : ℕ) (ticket_price : ℚ) (tickets_sold : ℕ) :
  capacity = 50 →
  ticket_price = 8 →
  tickets_sold = 24 →
  (capacity : ℚ) * ticket_price - (tickets_sold : ℚ) * ticket_price = 208 := by
  sorry

end NUMINAMATH_CALUDE_theater_lost_revenue_l3450_345090


namespace NUMINAMATH_CALUDE_inequality_proof_l3450_345095

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + b^2 + c^2 = 1) :
  (a * b / c) + (b * c / a) + (c * a / b) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3450_345095


namespace NUMINAMATH_CALUDE_existence_implies_a_bound_l3450_345088

/-- Given a > 0, prove that if there exists x₀ ∈ (0, 1/2] such that f(x₀) > g(x₀), then a > -3 + √17 -/
theorem existence_implies_a_bound (a : ℝ) (h₁ : a > 0) : 
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioc 0 (1/2) ∧ 
    a^2 * x₀^3 - 3*a * x₀^2 + 2 > -3*a * x₀ + 3) → 
  a > -3 + Real.sqrt 17 := by
sorry

/-- Definition of f(x) -/
def f (a x : ℝ) : ℝ := a^2 * x^3 - 3*a * x^2 + 2

/-- Definition of g(x) -/
def g (a x : ℝ) : ℝ := -3*a * x + 3

end NUMINAMATH_CALUDE_existence_implies_a_bound_l3450_345088


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3450_345071

def A : Set ℝ := {x | |x - 1| < 2}

def B : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = 2^x}

theorem intersection_of_A_and_B : A ∩ B = Set.Ico 1 3 := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3450_345071


namespace NUMINAMATH_CALUDE_power_inequality_l3450_345026

theorem power_inequality (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2) : a^b < b^a := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l3450_345026


namespace NUMINAMATH_CALUDE_tree_growth_condition_l3450_345085

/-- Represents the annual growth of a tree over 6 years -/
structure TreeGrowth where
  initial_height : ℝ
  annual_increase : ℝ

/-- Calculates the height of the tree after a given number of years -/
def height_after_years (t : TreeGrowth) (years : ℕ) : ℝ :=
  t.initial_height + t.annual_increase * years

/-- Theorem stating the condition for the tree's growth -/
theorem tree_growth_condition (t : TreeGrowth) : 
  t.initial_height = 4 ∧ 
  height_after_years t 6 = height_after_years t 4 + (1/7) * height_after_years t 4 →
  t.annual_increase = 2/5 :=
sorry

end NUMINAMATH_CALUDE_tree_growth_condition_l3450_345085


namespace NUMINAMATH_CALUDE_cycles_alignment_min_cycles_alignment_l3450_345018

/-- The length of the letter cycle -/
def letter_cycle_length : ℕ := 6

/-- The length of the digit cycle -/
def digit_cycle_length : ℕ := 4

/-- The theorem stating when both cycles will simultaneously return to their original state -/
theorem cycles_alignment (m : ℕ) (h1 : m > 0) (h2 : m % letter_cycle_length = 0) (h3 : m % digit_cycle_length = 0) :
  m ≥ 12 :=
sorry

/-- The theorem stating that 12 is the least number satisfying the conditions -/
theorem min_cycles_alignment :
  12 % letter_cycle_length = 0 ∧ 12 % digit_cycle_length = 0 ∧
  ∀ (k : ℕ), k > 0 → k % letter_cycle_length = 0 → k % digit_cycle_length = 0 → k ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_cycles_alignment_min_cycles_alignment_l3450_345018


namespace NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l3450_345035

theorem diagonal_not_parallel_to_sides (n : ℕ) (h : n > 0) :
  n * (2 * n - 3) > 2 * n * (n - 2) :=
sorry

end NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l3450_345035


namespace NUMINAMATH_CALUDE_count_good_pairs_l3450_345062

def is_good_pair (a p : ℕ) : Prop :=
  a > p ∧ (a^3 + p^3) % (a^2 - p^2) = 0

def is_prime_less_than_20 (p : ℕ) : Prop :=
  Nat.Prime p ∧ p < 20

theorem count_good_pairs :
  ∃ (S : Finset (ℕ × ℕ)), 
    S.card = 24 ∧
    (∀ (a p : ℕ), (a, p) ∈ S ↔ is_good_pair a p ∧ is_prime_less_than_20 p) :=
sorry

end NUMINAMATH_CALUDE_count_good_pairs_l3450_345062


namespace NUMINAMATH_CALUDE_second_level_treasures_is_two_l3450_345013

/-- Represents the number of points scored per treasure -/
def points_per_treasure : ℕ := 4

/-- Represents the number of treasures found on the first level -/
def first_level_treasures : ℕ := 6

/-- Represents the total score -/
def total_score : ℕ := 32

/-- Calculates the number of treasures found on the second level -/
def second_level_treasures : ℕ :=
  (total_score - (first_level_treasures * points_per_treasure)) / points_per_treasure

/-- Theorem stating that the number of treasures found on the second level is 2 -/
theorem second_level_treasures_is_two : second_level_treasures = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_level_treasures_is_two_l3450_345013


namespace NUMINAMATH_CALUDE_odd_function_extension_l3450_345065

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_extension
  (f : ℝ → ℝ)
  (odd : is_odd f)
  (pos_def : ∀ x > 0, f x = x - 1) :
  ∀ x < 0, f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_extension_l3450_345065


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3450_345023

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 9 > 0} = {x : ℝ | x < -3 ∨ x > 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3450_345023


namespace NUMINAMATH_CALUDE_coupon_redemption_schedule_l3450_345079

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday    => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday   => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday  => DayOfWeek.Friday
  | DayOfWeek.Friday    => DayOfWeek.Saturday
  | DayOfWeek.Saturday  => DayOfWeek.Sunday
  | DayOfWeek.Sunday    => DayOfWeek.Monday

def advance_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advance_days (next_day d) n

def is_saturday (d : DayOfWeek) : Prop :=
  d = DayOfWeek.Saturday

theorem coupon_redemption_schedule :
  let start_day := DayOfWeek.Monday
  let days_between_redemptions := 15
  let num_coupons := 7
  ∀ i, i < num_coupons →
    ¬(is_saturday (advance_days start_day (i * days_between_redemptions))) :=
by sorry

end NUMINAMATH_CALUDE_coupon_redemption_schedule_l3450_345079


namespace NUMINAMATH_CALUDE_certain_number_equation_l3450_345097

theorem certain_number_equation (x : ℝ) : 28 = (4/5) * x + 8 ↔ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l3450_345097


namespace NUMINAMATH_CALUDE_equation_solutions_l3450_345060

theorem equation_solutions :
  (∃ x : ℝ, 2 * (x + 3) = 5 * x ∧ x = 2) ∧
  (∃ x : ℝ, (x - 3) / 0.5 - (x + 4) / 0.2 = 1.6 ∧ x = -9.2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3450_345060


namespace NUMINAMATH_CALUDE_largest_common_value_less_than_1000_l3450_345038

def arithmetic_progression_1 (n : ℕ) : ℕ := 4 + 5 * n
def arithmetic_progression_2 (m : ℕ) : ℕ := 7 + 8 * m

def is_common_value (a : ℕ) : Prop :=
  ∃ n m : ℕ, arithmetic_progression_1 n = a ∧ arithmetic_progression_2 m = a

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a < 1000 ∧ is_common_value a ∧
  ∀ b : ℕ, b < 1000 → is_common_value b → b ≤ a :=
by
  use 999
  sorry

#eval arithmetic_progression_1 199  -- Should evaluate to 999
#eval arithmetic_progression_2 124  -- Should evaluate to 999

end NUMINAMATH_CALUDE_largest_common_value_less_than_1000_l3450_345038


namespace NUMINAMATH_CALUDE_total_prairie_area_l3450_345078

def prairie_size (dust_covered : ℕ) (untouched : ℕ) : ℕ :=
  dust_covered + untouched

theorem total_prairie_area : prairie_size 64535 522 = 65057 := by
  sorry

end NUMINAMATH_CALUDE_total_prairie_area_l3450_345078


namespace NUMINAMATH_CALUDE_min_value_of_f_f_attains_min_l3450_345096

/-- The minimum value of a function f given certain conditions -/
theorem min_value_of_f (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : c * y + b * z = a)
  (h2 : a * z + c * x = b)
  (h3 : b * x + a * y = c) :
  (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z)) ≥ (1/2 : ℝ) := by
  sorry

/-- The function f attains its minimum value -/
theorem f_attains_min (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x y z : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧
    c * y + b * z = a ∧
    a * z + c * x = b ∧
    b * x + a * y = c ∧
    x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_f_attains_min_l3450_345096


namespace NUMINAMATH_CALUDE_no_consecutive_perfect_squares_l3450_345041

theorem no_consecutive_perfect_squares (a b : ℤ) : a^2 - b^2 = 1 → (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_consecutive_perfect_squares_l3450_345041


namespace NUMINAMATH_CALUDE_expression_evaluation_l3450_345027

theorem expression_evaluation : 
  |(-1/2 : ℝ)| + ((-27 : ℝ) ^ (1/3 : ℝ)) - (1/4 : ℝ).sqrt + (12 : ℝ).sqrt * (3 : ℝ).sqrt = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3450_345027


namespace NUMINAMATH_CALUDE_f_sum_property_l3450_345086

def f (x : ℝ) : ℝ := 5*x^6 - 3*x^5 + 4*x^4 + x^3 - 2*x^2 - 2*x + 8

theorem f_sum_property : f 5 = 20 → f 5 + f (-5) = 68343 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_property_l3450_345086
