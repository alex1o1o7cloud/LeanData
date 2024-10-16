import Mathlib

namespace NUMINAMATH_CALUDE_cheryl_material_usage_l998_99836

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 5 / 9) 
  (h2 : material2 = 1 / 3) 
  (h3 : leftover = 8 / 24) : 
  material1 + material2 - leftover = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l998_99836


namespace NUMINAMATH_CALUDE_midpoint_locus_l998_99862

/-- The locus of midpoints between a fixed point and points on a circle -/
theorem midpoint_locus (A B P : ℝ × ℝ) (m n x y : ℝ) :
  A = (4, -2) →
  B = (m, n) →
  m^2 + n^2 = 4 →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  (x - 2)^2 + (y + 1)^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_locus_l998_99862


namespace NUMINAMATH_CALUDE_odd_prime_square_root_l998_99856

theorem odd_prime_square_root (p k : ℕ) : 
  Prime p → 
  Odd p → 
  k > 0 → 
  ∃ n : ℕ, n > 0 ∧ n * n = k * k - p * k → 
  k = (p + 1)^2 / 4 := by
sorry

end NUMINAMATH_CALUDE_odd_prime_square_root_l998_99856


namespace NUMINAMATH_CALUDE_negative_x_sqrt_squared_diff_l998_99899

theorem negative_x_sqrt_squared_diff (x : ℝ) (h : x < 0) : x - Real.sqrt ((x - 1)^2) = 2*x - 1 := by
  sorry

end NUMINAMATH_CALUDE_negative_x_sqrt_squared_diff_l998_99899


namespace NUMINAMATH_CALUDE_linear_functions_inequality_l998_99829

theorem linear_functions_inequality (k : ℝ) :
  (∀ x > -1, k * x - 2 < 2 * x + 3) →
  -3 ≤ k ∧ k ≤ 2 ∧ k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_functions_inequality_l998_99829


namespace NUMINAMATH_CALUDE_sin_negative_690_degrees_l998_99845

theorem sin_negative_690_degrees : Real.sin ((-690 : ℝ) * π / 180) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_690_degrees_l998_99845


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l998_99838

theorem quadratic_function_proof (f g : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) →  -- f is quadratic
  f 0 = 12 →                                      -- f(0) = 12
  (∀ x, g x = 2^x * f x) →                        -- g(x) = 2^x * f(x)
  (∀ x, g (x + 1) - g x ≥ 2^(x + 1) * x^2) →      -- g(x+1) - g(x) ≥ 2^(x+1) * x^2
  (∀ x, f x = 2 * x^2 - 8 * x + 12) ∧             -- f(x) = 2x^2 - 8x + 12
  (∀ x, g x = (2 * x^2 - 8 * x + 12) * 2^x) :=    -- g(x) = (2x^2 - 8x + 12) * 2^x
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l998_99838


namespace NUMINAMATH_CALUDE_triangle_area_approximation_l998_99893

theorem triangle_area_approximation (α β : Real) (k l m : Real) :
  α = π / 6 →
  β = π / 4 →
  k = 3 →
  l = 2 →
  m = 4 →
  let γ : Real := π - α - β
  let S := ((k * Real.sin α + l * Real.sin β + m * Real.sin γ) ^ 2) / (2 * Real.sin α * Real.sin β * Real.sin γ)
  |S - 67| < 0.5 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_approximation_l998_99893


namespace NUMINAMATH_CALUDE_centric_sequence_bound_and_extremal_points_l998_99890

/-- The set of points (x, y) in R^2 such that x^2 + y^2 ≤ 1 -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}

/-- A sequence of points in R^2 -/
def Sequence := ℕ → ℝ × ℝ

/-- The circumcenter of a triangle formed by three points -/
noncomputable def circumcenter (p1 p2 p3 : ℝ × ℝ) : ℝ × ℝ := sorry

/-- A centric sequence satisfies the given properties -/
def IsCentric (A : Sequence) : Prop :=
  A 0 = (0, 0) ∧ A 1 = (1, 0) ∧
  ∀ n : ℕ, circumcenter (A n) (A (n+1)) (A (n+2)) ∈ C

theorem centric_sequence_bound_and_extremal_points :
  ∀ A : Sequence, IsCentric A →
    (A 2012).1^2 + (A 2012).2^2 ≤ 4048144 ∧
    (∀ x y : ℝ, x^2 + y^2 = 4048144 →
      (∃ A : Sequence, IsCentric A ∧ A 2012 = (x, y)) →
      ((x = -1006 ∧ y = 1006 * Real.sqrt 3) ∨
       (x = -1006 ∧ y = -1006 * Real.sqrt 3))) :=
by sorry

end NUMINAMATH_CALUDE_centric_sequence_bound_and_extremal_points_l998_99890


namespace NUMINAMATH_CALUDE_emily_shopping_expense_l998_99844

def total_spent (art_supplies_cost skirt_cost number_of_skirts : ℕ) : ℕ :=
  art_supplies_cost + skirt_cost * number_of_skirts

theorem emily_shopping_expense :
  let art_supplies_cost : ℕ := 20
  let skirt_cost : ℕ := 15
  let number_of_skirts : ℕ := 2
  total_spent art_supplies_cost skirt_cost number_of_skirts = 50 := by
  sorry

end NUMINAMATH_CALUDE_emily_shopping_expense_l998_99844


namespace NUMINAMATH_CALUDE_balloon_multiple_l998_99855

def nancy_balloons : ℝ := 7.0
def mary_balloons : ℝ := 1.75

theorem balloon_multiple : nancy_balloons / mary_balloons = 4 := by
  sorry

end NUMINAMATH_CALUDE_balloon_multiple_l998_99855


namespace NUMINAMATH_CALUDE_intersection_points_properties_l998_99818

open Real

theorem intersection_points_properties (k : ℝ) (h_k : k > 0) :
  let f := fun x => Real.exp x
  let g := fun x => Real.exp (-x)
  let n := f k
  let m := g k
  n < 2 * m →
  (n + m < 3 * Real.sqrt 2 / 2) ∧
  (n - m < Real.sqrt 2 / 2) ∧
  (n^(m + 1) < (m + 1)^n) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_properties_l998_99818


namespace NUMINAMATH_CALUDE_belle_rawhide_bones_l998_99835

/-- The number of dog biscuits Belle eats every evening -/
def dog_biscuits : ℕ := 4

/-- The cost of one dog biscuit in dollars -/
def dog_biscuit_cost : ℚ := 1/4

/-- The cost of one rawhide bone in dollars -/
def rawhide_bone_cost : ℚ := 1

/-- The total cost of Belle's treats for a week in dollars -/
def weekly_treat_cost : ℚ := 21

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of rawhide bones Belle eats every evening -/
def rawhide_bones : ℕ := 2

theorem belle_rawhide_bones :
  (dog_biscuits : ℚ) * dog_biscuit_cost * (days_in_week : ℚ) +
  (rawhide_bones : ℚ) * rawhide_bone_cost * (days_in_week : ℚ) =
  weekly_treat_cost :=
sorry

end NUMINAMATH_CALUDE_belle_rawhide_bones_l998_99835


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l998_99806

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 3)^11 = a + a₁*(x - 2) + a₂*(x - 2)^2 + a₃*(x - 2)^3 + 
    a₄*(x - 2)^4 + a₅*(x - 2)^5 + a₆*(x - 2)^6 + a₇*(x - 2)^7 + a₈*(x - 2)^8 + 
    a₉*(x - 2)^9 + a₁₀*(x - 2)^10 + a₁₁*(x - 2)^11 + a₁₂*(x - 2)^12 + a₁₃*(x - 2)^13) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l998_99806


namespace NUMINAMATH_CALUDE_division_problem_l998_99804

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 122 → quotient = 6 → remainder = 2 → 
  dividend = divisor * quotient + remainder →
  divisor = 20 := by sorry

end NUMINAMATH_CALUDE_division_problem_l998_99804


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l998_99820

/-- The number of ways to choose a pair of socks with different colors -/
def different_color_sock_pairs (white brown blue : ℕ) : ℕ :=
  white * brown + white * blue + brown * blue

/-- Theorem: Given 5 white socks, 3 brown socks, and 2 blue socks,
    there are 31 ways to choose a pair of socks with different colors -/
theorem sock_pair_combinations :
  different_color_sock_pairs 5 3 2 = 31 := by
  sorry

#eval different_color_sock_pairs 5 3 2

end NUMINAMATH_CALUDE_sock_pair_combinations_l998_99820


namespace NUMINAMATH_CALUDE_tom_age_l998_99848

theorem tom_age (carla_age : ℕ) (tom_age : ℕ) (dave_age : ℕ) : 
  (tom_age = 2 * carla_age - 1) →
  (dave_age = carla_age + 3) →
  (carla_age + tom_age + dave_age = 30) →
  tom_age = 13 := by
sorry

end NUMINAMATH_CALUDE_tom_age_l998_99848


namespace NUMINAMATH_CALUDE_square_sum_formula_l998_99834

theorem square_sum_formula (x y z a b c d : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (h4 : (x + y + z)^2 = d) : 
  x^2 + y^2 + z^2 = d - 2*(a + b + c) := by
sorry

end NUMINAMATH_CALUDE_square_sum_formula_l998_99834


namespace NUMINAMATH_CALUDE_polynomial_factorization_l998_99871

theorem polynomial_factorization (k : ℤ) :
  ∃ (p q : Polynomial ℤ),
    Polynomial.degree p = 4 ∧
    Polynomial.degree q = 4 ∧
    (X : Polynomial ℤ)^8 + (4 * k^4 - 8 * k^2 + 2) * X^4 + 1 = p * q :=
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l998_99871


namespace NUMINAMATH_CALUDE_log_one_half_of_one_eighth_l998_99880

theorem log_one_half_of_one_eighth (a : ℝ) : a = Real.log 0.125 / Real.log (1/2) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_one_half_of_one_eighth_l998_99880


namespace NUMINAMATH_CALUDE_combined_age_of_siblings_l998_99859

-- Define the ages of the siblings
def aaron_age : ℕ := 15
def sister_age : ℕ := 3 * aaron_age
def henry_age : ℕ := 4 * sister_age
def alice_age : ℕ := aaron_age - 2

-- Theorem to prove
theorem combined_age_of_siblings : aaron_age + sister_age + henry_age + alice_age = 253 := by
  sorry

end NUMINAMATH_CALUDE_combined_age_of_siblings_l998_99859


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_relation_l998_99876

-- Define the necessary structures
structure Line3D where
  -- Add necessary fields for a 3D line

structure Plane3D where
  -- Add necessary fields for a 3D plane

-- Define the relationships
def parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def within (l : Line3D) (p : Plane3D) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

def skew_lines (l1 l2 : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem line_parallel_to_plane_relation (m n : Line3D) (α : Plane3D) 
    (h1 : parallel m α) (h2 : within n α) :
  parallel_lines m n ∨ skew_lines m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_relation_l998_99876


namespace NUMINAMATH_CALUDE_library_schedule_l998_99815

theorem library_schedule (sam fran mike julio : ℕ) 
  (h_sam : sam = 5)
  (h_fran : fran = 8)
  (h_mike : mike = 10)
  (h_julio : julio = 12) :
  Nat.lcm (Nat.lcm (Nat.lcm sam fran) mike) julio = 120 := by
  sorry

end NUMINAMATH_CALUDE_library_schedule_l998_99815


namespace NUMINAMATH_CALUDE_game_ends_in_41_rounds_l998_99812

/-- Represents a player in the token game -/
structure Player where
  name : String
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : List Player
  rounds : ℕ

/-- Simulates one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (i.e., a player has run out of tokens) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Simulates the entire game until it ends -/
def playGame (initialState : GameState) : GameState :=
  sorry

/-- Theorem stating that the game ends after exactly 41 rounds -/
theorem game_ends_in_41_rounds :
  let initialState : GameState :=
    { players := [
        { name := "D", tokens := 16 },
        { name := "E", tokens := 15 },
        { name := "F", tokens := 13 }
      ],
      rounds := 0
    }
  let finalState := playGame initialState
  finalState.rounds = 41 ∧ isGameOver finalState := by
  sorry

end NUMINAMATH_CALUDE_game_ends_in_41_rounds_l998_99812


namespace NUMINAMATH_CALUDE_functional_equation_solution_l998_99878

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y + f (x + y) = x * y) →
  (∀ x : ℝ, f x = x - 1) ∨ (∀ x : ℝ, f x = -x - 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l998_99878


namespace NUMINAMATH_CALUDE_factorization_equality_l998_99827

theorem factorization_equality (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l998_99827


namespace NUMINAMATH_CALUDE_total_seashells_l998_99883

theorem total_seashells (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) 
  (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l998_99883


namespace NUMINAMATH_CALUDE_ellipse_chord_theorem_l998_99842

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Defines if a point lies on an ellipse -/
def Point.onEllipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Defines if a point is the midpoint of two other points -/
def isMidpoint (m p1 p2 : Point) : Prop :=
  m.x = (p1.x + p2.x) / 2 ∧ m.y = (p1.y + p2.y) / 2

/-- Defines if three points are collinear -/
def areCollinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem ellipse_chord_theorem (e : Ellipse) (p1 p2 m : Point) :
  e.a = 6 ∧ e.b = 3 →
  p1.onEllipse e ∧ p2.onEllipse e →
  isMidpoint m p1 p2 →
  m = Point.mk 4 2 →
  areCollinear p1 p2 (Point.mk 0 4) :=
sorry

end NUMINAMATH_CALUDE_ellipse_chord_theorem_l998_99842


namespace NUMINAMATH_CALUDE_alpha_beta_relation_l998_99870

open Real

theorem alpha_beta_relation (α β : ℝ) :
  π / 2 < α ∧ α < π ∧
  π / 2 < β ∧ β < π ∧
  (1 - cos (2 * α)) * (1 + sin β) = sin (2 * α) * cos β →
  2 * α + β = 5 * π / 2 := by
sorry

end NUMINAMATH_CALUDE_alpha_beta_relation_l998_99870


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l998_99875

def normal_distribution (μ σ : ℝ) : Type := sorry

theorem two_std_dev_below_mean 
  (μ σ : ℝ) 
  (dist : normal_distribution μ σ) 
  (h_μ : μ = 14.5) 
  (h_σ : σ = 1.5) : 
  μ - 2 * σ = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l998_99875


namespace NUMINAMATH_CALUDE_marble_selection_theorem_l998_99868

/-- The number of marbles John has in total -/
def total_marbles : ℕ := 15

/-- The number of colors with exactly two marbles each -/
def special_colors : ℕ := 3

/-- The number of marbles for each special color -/
def marbles_per_special_color : ℕ := 2

/-- The number of marbles to be chosen -/
def marbles_to_choose : ℕ := 5

/-- The number of special colored marbles to be chosen -/
def special_marbles_to_choose : ℕ := 2

/-- The number of ways to choose the marbles under the given conditions -/
def ways_to_choose : ℕ := 1008

theorem marble_selection_theorem :
  (Nat.choose special_colors special_marbles_to_choose) *
  (Nat.choose marbles_per_special_color 1) ^ special_marbles_to_choose *
  (Nat.choose (total_marbles - special_colors * marbles_per_special_color) (marbles_to_choose - special_marbles_to_choose)) =
  ways_to_choose := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_theorem_l998_99868


namespace NUMINAMATH_CALUDE_teahouse_on_tuesday_or_thursday_not_all_plays_on_tuesday_heavenly_sound_not_on_wednesday_thunderstorm_not_only_on_tuesday_l998_99813

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday

-- Define the plays
inductive Play
| Thunderstorm
| Teahouse
| HeavenlySound
| ShatteredHoofbeats

def Schedule := Day → Play

def valid_schedule (s : Schedule) : Prop :=
  (s Day.Monday ≠ Play.Thunderstorm) ∧
  (s Day.Thursday ≠ Play.Thunderstorm) ∧
  (s Day.Monday ≠ Play.Teahouse) ∧
  (s Day.Wednesday ≠ Play.Teahouse) ∧
  (s Day.Wednesday ≠ Play.HeavenlySound) ∧
  (s Day.Thursday ≠ Play.HeavenlySound) ∧
  (s Day.Monday ≠ Play.ShatteredHoofbeats) ∧
  (s Day.Thursday ≠ Play.ShatteredHoofbeats) ∧
  (∀ d1 d2, d1 ≠ d2 → s d1 ≠ s d2)

theorem teahouse_on_tuesday_or_thursday :
  ∃ (s : Schedule), valid_schedule s ∧
    (s Day.Tuesday = Play.Teahouse ∨ s Day.Thursday = Play.Teahouse) :=
by sorry

theorem not_all_plays_on_tuesday :
  ¬∃ (s : Schedule), valid_schedule s ∧
    (s Day.Tuesday = Play.Thunderstorm ∧
     s Day.Tuesday = Play.Teahouse ∧
     s Day.Tuesday = Play.HeavenlySound ∧
     s Day.Tuesday = Play.ShatteredHoofbeats) :=
by sorry

theorem heavenly_sound_not_on_wednesday :
  ∀ (s : Schedule), valid_schedule s →
    s Day.Wednesday ≠ Play.HeavenlySound :=
by sorry

theorem thunderstorm_not_only_on_tuesday :
  ∃ (s1 s2 : Schedule), valid_schedule s1 ∧ valid_schedule s2 ∧
    s1 Day.Tuesday = Play.Thunderstorm ∧
    s2 Day.Wednesday = Play.Thunderstorm :=
by sorry

end NUMINAMATH_CALUDE_teahouse_on_tuesday_or_thursday_not_all_plays_on_tuesday_heavenly_sound_not_on_wednesday_thunderstorm_not_only_on_tuesday_l998_99813


namespace NUMINAMATH_CALUDE_triangle_area_l998_99817

-- Define the three lines
def line1 (x : ℝ) : ℝ := 2 * x + 1
def line2 (x : ℝ) : ℝ := -x + 4
def line3 (x : ℝ) : ℝ := -1

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (1, 3)
def vertex2 : ℝ × ℝ := (-1, -1)
def vertex3 : ℝ × ℝ := (5, -1)

-- Theorem statement
theorem triangle_area : 
  let vertices := [vertex1, vertex2, vertex3]
  let xs := vertices.map Prod.fst
  let ys := vertices.map Prod.snd
  abs ((xs[0] * (ys[1] - ys[2]) + xs[1] * (ys[2] - ys[0]) + xs[2] * (ys[0] - ys[1])) / 2) = 12 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_l998_99817


namespace NUMINAMATH_CALUDE_card_movement_strategy_exists_no_guaranteed_ace_strategy_l998_99860

/-- Represents a deck of cards arranged in a circle with one free spot -/
structure CircularDeck :=
  (cards : Fin 52 → Option (Fin 52))
  (free_spot : Fin 53)
  (initial_positions : Fin 52 → Fin 53)

/-- Represents a strategy for naming cards -/
def Strategy := ℕ → Fin 52

/-- Checks if a card is next to the free spot -/
def is_next_to_free_spot (deck : CircularDeck) (card : Fin 52) : Prop :=
  sorry

/-- Moves a card to the free spot if it's adjacent -/
def move_card (deck : CircularDeck) (card : Fin 52) : CircularDeck :=
  sorry

/-- Applies a strategy to a deck for a given number of steps -/
def apply_strategy (deck : CircularDeck) (strategy : Strategy) (steps : ℕ) : CircularDeck :=
  sorry

/-- Checks if all cards are not in their initial positions -/
def all_cards_moved (deck : CircularDeck) : Prop :=
  sorry

/-- Checks if the ace of spades is not next to the free spot -/
def ace_not_next_to_free (deck : CircularDeck) : Prop :=
  sorry

theorem card_movement_strategy_exists :
  ∃ (strategy : Strategy), ∀ (initial_deck : CircularDeck),
  ∃ (steps : ℕ), all_cards_moved (apply_strategy initial_deck strategy steps) :=
sorry

theorem no_guaranteed_ace_strategy :
  ¬∃ (strategy : Strategy), ∀ (initial_deck : CircularDeck),
  ∃ (steps : ℕ), ace_not_next_to_free (apply_strategy initial_deck strategy steps) :=
sorry

end NUMINAMATH_CALUDE_card_movement_strategy_exists_no_guaranteed_ace_strategy_l998_99860


namespace NUMINAMATH_CALUDE_last_digits_divisible_by_4_l998_99831

-- Define a function to check if a number is divisible by 4
def divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

-- Define a function to get the last digit of a number
def last_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem last_digits_divisible_by_4 :
  ∃! (s : Finset ℕ), (∀ n ∈ s, ∃ m : ℕ, divisible_by_4 m ∧ last_digit m = n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_last_digits_divisible_by_4_l998_99831


namespace NUMINAMATH_CALUDE_set_difference_M_N_l998_99896

def M : Set ℕ := {1, 3, 5, 7, 9}
def N : Set ℕ := {2, 3, 5}

def setDifference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem set_difference_M_N : setDifference M N = {1, 7, 9} := by
  sorry

end NUMINAMATH_CALUDE_set_difference_M_N_l998_99896


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l998_99846

theorem algebraic_expression_equality : 
  let a : ℝ := Real.sqrt 3 + 2
  (a - Real.sqrt 2) * (a + Real.sqrt 2) - a * (a - 3) = 3 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l998_99846


namespace NUMINAMATH_CALUDE_problem_1_l998_99801

theorem problem_1 : Real.sqrt 12 + 3 - 2^2 + |1 - Real.sqrt 3| = 3 * Real.sqrt 3 - 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l998_99801


namespace NUMINAMATH_CALUDE_cereal_eating_time_l998_99849

/-- The time taken for two people to eat a certain amount of cereal together -/
def time_to_eat_together (fat_rate : ℚ) (thin_rate : ℚ) (amount : ℚ) : ℚ :=
  amount / (fat_rate + thin_rate)

/-- Theorem: Given the eating rates and amount of cereal, prove that it takes 45 minutes to finish -/
theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 45  -- Mr. Thin's eating rate in pounds per minute
  let amount : ℚ := 4  -- Amount of cereal in pounds
  time_to_eat_together fat_rate thin_rate amount = 45 := by
  sorry

#eval time_to_eat_together (1/15 : ℚ) (1/45 : ℚ) 4

end NUMINAMATH_CALUDE_cereal_eating_time_l998_99849


namespace NUMINAMATH_CALUDE_parabola_properties_l998_99853

-- Define the parabola C
def C (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem parabola_properties :
  ∃ (p : ℝ), 
    (∃ (x y : ℝ), C p x y ∧ focus_line x y) →
    (p = 8 ∧ 
     ∀ (x : ℝ), (x = -4) ↔ (∃ (y : ℝ), C p x y ∧ ∀ (x' y' : ℝ), C p x' y' → (x - x')^2 + (y - y')^2 ≥ (x + 4)^2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_properties_l998_99853


namespace NUMINAMATH_CALUDE_subtract_negatives_l998_99894

theorem subtract_negatives : (-7) - (-5) = -2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negatives_l998_99894


namespace NUMINAMATH_CALUDE_even_function_property_l998_99828

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : is_even (λ x ↦ f (x + 2))) 
  (h3 : f 1 = π / 3) : 
  f 3 + f (-3) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l998_99828


namespace NUMINAMATH_CALUDE_vegetable_field_division_l998_99808

theorem vegetable_field_division (total_area : ℚ) (num_parts : ℕ) 
  (h1 : total_area = 5)
  (h2 : num_parts = 8) :
  (1 : ℚ) / num_parts = 1 / 8 ∧ total_area / num_parts = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_field_division_l998_99808


namespace NUMINAMATH_CALUDE_problem_statement_l998_99803

def A (n r : ℕ) : ℕ := n.factorial / (n - r).factorial

def C (n r : ℕ) : ℕ := n.factorial / (r.factorial * (n - r).factorial)

theorem problem_statement : A 6 2 + C 6 4 = 45 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l998_99803


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l998_99811

theorem unique_solution_logarithmic_equation :
  ∃! (x : ℝ), x > 0 ∧ (Real.log x / Real.log 4) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 4 ∧ x^2 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_equation_l998_99811


namespace NUMINAMATH_CALUDE_cos_beta_value_l998_99840

theorem cos_beta_value (α β : Real) (P : ℝ × ℝ) :
  P = (3, 4) →
  P.1 = 3 * Real.cos α ∧ P.2 = 3 * Real.sin α →
  Real.cos (α + β) = 1/3 →
  β ∈ Set.Ioo 0 Real.pi →
  Real.cos β = (3 + 8 * Real.sqrt 2) / 15 := by
  sorry

end NUMINAMATH_CALUDE_cos_beta_value_l998_99840


namespace NUMINAMATH_CALUDE_right_triangle_area_l998_99891

theorem right_triangle_area (leg1 leg2 : ℝ) (h1 : leg1 = 45) (h2 : leg2 = 48) :
  (1/2 : ℝ) * leg1 * leg2 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l998_99891


namespace NUMINAMATH_CALUDE_problem_solution_l998_99889

/-- The function f(x) = x^2 - 1 --/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The function g(x) = a|x-1| --/
def g (a x : ℝ) : ℝ := a * |x - 1|

/-- The function h(x) = |f(x)| + g(x) --/
def h (a x : ℝ) : ℝ := |f x| + g a x

theorem problem_solution (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) →
  (a ≤ -2 ∧
   (∀ x ∈ Set.Icc 0 1,
      (a ≥ -3 → h a x ≤ a + 3) ∧
      (a < -3 → h a x ≤ 0))) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l998_99889


namespace NUMINAMATH_CALUDE_remainder_5462_div_9_l998_99805

theorem remainder_5462_div_9 : 5462 % 9 = 8 := by sorry

end NUMINAMATH_CALUDE_remainder_5462_div_9_l998_99805


namespace NUMINAMATH_CALUDE_sum_of_fractions_inequality_l998_99830

theorem sum_of_fractions_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ (3 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_inequality_l998_99830


namespace NUMINAMATH_CALUDE_toy_distribution_l998_99841

/-- Given a number of pens and toys distributed among students, 
    where each student receives the same number of pens and toys, 
    prove that the number of toys is a multiple of the number of students. -/
theorem toy_distribution (num_pens : ℕ) (num_toys : ℕ) (num_students : ℕ) 
  (h1 : num_pens = 451)
  (h2 : num_students = 41)
  (h3 : num_pens % num_students = 0)
  (h4 : num_toys % num_students = 0) :
  ∃ k : ℕ, num_toys = num_students * k :=
sorry

end NUMINAMATH_CALUDE_toy_distribution_l998_99841


namespace NUMINAMATH_CALUDE_abc_fraction_equality_l998_99874

theorem abc_fraction_equality (a b c : ℝ) 
  (h1 : a * b + b * c + a * c = 1)
  (h2 : a ≠ 1 ∧ a ≠ -1)
  (h3 : b ≠ 1 ∧ b ≠ -1)
  (h4 : c ≠ 1 ∧ c ≠ -1) :
  a / (1 - a^2) + b / (1 - b^2) + c / (1 - c^2) = 
  4 * a * b * c / ((1 - a^2) * (1 - b^2) * (1 - c^2)) := by
sorry

end NUMINAMATH_CALUDE_abc_fraction_equality_l998_99874


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l998_99858

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  a_1_eq : a 1 = 4
  a_7_sq_eq : (a 7) ^ 2 = (a 1) * (a 10)
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Sum of the first n terms of the sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = -1/3 * n + 13/3) ∧
  (∃ n : ℕ, S_n seq n = 26 ∧ (n = 12 ∨ n = 13) ∧ ∀ m : ℕ, S_n seq m ≤ 26) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l998_99858


namespace NUMINAMATH_CALUDE_cuboid_area_volume_l998_99885

/-- Cuboid properties -/
def Cuboid (a b c : ℝ) : Prop :=
  c * Real.sqrt (a^2 + b^2) = 60 ∧
  a * Real.sqrt (b^2 + c^2) = 4 * Real.sqrt 153 ∧
  b * Real.sqrt (a^2 + c^2) = 12 * Real.sqrt 10

/-- Theorem: Surface area and volume of the cuboid -/
theorem cuboid_area_volume (a b c : ℝ) (h : Cuboid a b c) :
  2 * (a * b + b * c + a * c) = 192 ∧ a * b * c = 144 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_area_volume_l998_99885


namespace NUMINAMATH_CALUDE_least_marbles_ten_marbles_john_marbles_l998_99877

theorem least_marbles (m : ℕ) : m > 0 ∧ m % 7 = 3 ∧ m % 4 = 2 → m ≥ 10 := by
  sorry

theorem ten_marbles : 10 % 7 = 3 ∧ 10 % 4 = 2 := by
  sorry

theorem john_marbles : ∃ m : ℕ, m > 0 ∧ m % 7 = 3 ∧ m % 4 = 2 ∧ ∀ n : ℕ, (n > 0 ∧ n % 7 = 3 ∧ n % 4 = 2) → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_least_marbles_ten_marbles_john_marbles_l998_99877


namespace NUMINAMATH_CALUDE_carolyn_unicorns_l998_99825

/-- Calculates the number of unicorns Carolyn wants to embroider --/
def number_of_unicorns (stitches_per_minute : ℕ) (flower_stitches : ℕ) (unicorn_stitches : ℕ) 
  (godzilla_stitches : ℕ) (total_minutes : ℕ) (number_of_flowers : ℕ) : ℕ :=
  let total_stitches := stitches_per_minute * total_minutes
  let flower_total_stitches := flower_stitches * number_of_flowers
  let remaining_stitches := total_stitches - flower_total_stitches - godzilla_stitches
  remaining_stitches / unicorn_stitches

/-- Theorem stating that Carolyn wants to embroider 3 unicorns --/
theorem carolyn_unicorns : 
  number_of_unicorns 4 60 180 800 1085 50 = 3 := by
  sorry

end NUMINAMATH_CALUDE_carolyn_unicorns_l998_99825


namespace NUMINAMATH_CALUDE_tangent_length_correct_l998_99857

/-- Two circles S₁ and S₂ touching at point A with radii R and r respectively (R > r).
    B is a point on S₁ such that AB = a. -/
structure TangentCircles where
  R : ℝ
  r : ℝ
  a : ℝ
  h₁ : R > r
  h₂ : R > 0
  h₃ : r > 0
  h₄ : a > 0

/-- The length of the tangent from B to S₂ -/
noncomputable def tangentLength (c : TangentCircles) (external : Bool) : ℝ :=
  if external then
    c.a * Real.sqrt ((c.R + c.r) / c.R)
  else
    c.a * Real.sqrt ((c.R - c.r) / c.R)

theorem tangent_length_correct (c : TangentCircles) :
  (∀ external, tangentLength c external = 
    if external then c.a * Real.sqrt ((c.R + c.r) / c.R)
    else c.a * Real.sqrt ((c.R - c.r) / c.R)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_length_correct_l998_99857


namespace NUMINAMATH_CALUDE_rectangle_ratio_l998_99866

theorem rectangle_ratio (width : ℕ) (area : ℕ) (length : ℕ) : 
  width = 7 → 
  area = 196 → 
  length * width = area → 
  ∃ k : ℕ, length = k * width → 
  (length : ℚ) / width = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l998_99866


namespace NUMINAMATH_CALUDE_complex_equation_solution_l998_99863

theorem complex_equation_solution (m n : ℝ) (i : ℂ) 
  (h1 : i * i = -1)
  (h2 : m / (1 + i) = 1 - n * i) : 
  m - n = 1 := by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l998_99863


namespace NUMINAMATH_CALUDE_dog_tricks_conversion_l998_99895

def base5_to_base10 (a b c d : ℕ) : ℕ :=
  d * 5^0 + c * 5^1 + b * 5^2 + a * 5^3

theorem dog_tricks_conversion :
  base5_to_base10 1 2 3 4 = 194 := by
  sorry

end NUMINAMATH_CALUDE_dog_tricks_conversion_l998_99895


namespace NUMINAMATH_CALUDE_abs_3x_plus_5_not_positive_l998_99809

theorem abs_3x_plus_5_not_positive (x : ℚ) : ¬(|3*x + 5| > 0) ↔ x = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_abs_3x_plus_5_not_positive_l998_99809


namespace NUMINAMATH_CALUDE_amp_five_two_squared_l998_99864

/-- The & operation defined for real numbers -/
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

/-- Theorem stating that (5 & 2)^2 = 441 -/
theorem amp_five_two_squared : (amp 5 2)^2 = 441 := by
  sorry

end NUMINAMATH_CALUDE_amp_five_two_squared_l998_99864


namespace NUMINAMATH_CALUDE_shoe_cost_l998_99802

/-- Given a suit purchase, a discount, and a total paid amount, prove the cost of shoes. -/
theorem shoe_cost (suit_price discount total_paid : ℤ) (h1 : suit_price = 430) (h2 : discount = 100) (h3 : total_paid = 520) :
  suit_price + (total_paid + discount - suit_price) = total_paid + discount := by
  sorry

#eval 520 + 100 - 430  -- Expected output: 190

end NUMINAMATH_CALUDE_shoe_cost_l998_99802


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l998_99810

theorem arithmetic_expression_evaluation : 3 * 4 + (2 * 5)^2 - 6 * 2 = 100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l998_99810


namespace NUMINAMATH_CALUDE_fifty_three_days_from_friday_l998_99872

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek
| Sunday : DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek

def days_in_week : Nat := 7

def friday_to_int : Nat := 5

def add_days (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match (friday_to_int + n) % days_in_week with
  | 0 => DayOfWeek.Sunday
  | 1 => DayOfWeek.Monday
  | 2 => DayOfWeek.Tuesday
  | 3 => DayOfWeek.Wednesday
  | 4 => DayOfWeek.Thursday
  | 5 => DayOfWeek.Friday
  | _ => DayOfWeek.Saturday

theorem fifty_three_days_from_friday :
  add_days DayOfWeek.Friday 53 = DayOfWeek.Tuesday := by
  sorry

end NUMINAMATH_CALUDE_fifty_three_days_from_friday_l998_99872


namespace NUMINAMATH_CALUDE_root_equation_problem_l998_99882

theorem root_equation_problem (c d n r s : ℝ) : 
  (c^2 - n*c + 3 = 0) → 
  (d^2 - n*d + 3 = 0) → 
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) → 
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) → 
  s = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l998_99882


namespace NUMINAMATH_CALUDE_total_basketballs_donated_prove_total_basketballs_l998_99826

/-- Calculates the total number of basketballs donated to a school --/
theorem total_basketballs_donated (total_donations : ℕ) (basketball_hoops : ℕ) (pool_floats : ℕ) 
  (footballs : ℕ) (tennis_balls : ℕ) : ℕ :=
  let basketballs_with_hoops := basketball_hoops / 2
  let undamaged_pool_floats := pool_floats * 3 / 4
  let accounted_donations := basketball_hoops + undamaged_pool_floats + footballs + tennis_balls
  let separate_basketballs := total_donations - accounted_donations
  basketballs_with_hoops + separate_basketballs

/-- Proves that the total number of basketballs donated is 90 --/
theorem prove_total_basketballs :
  total_basketballs_donated 300 60 120 50 40 = 90 := by
  sorry

end NUMINAMATH_CALUDE_total_basketballs_donated_prove_total_basketballs_l998_99826


namespace NUMINAMATH_CALUDE_train_speed_is_6_l998_99837

/-- The speed of a train in km/hr, given its length and time to cross a pole -/
def train_speed (length : Float) (time : Float) : Float :=
  (length / time) * 3.6

/-- Theorem: The speed of the train is 6 km/hr -/
theorem train_speed_is_6 :
  let length : Float := 3.3333333333333335
  let time : Float := 2
  train_speed length time = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_is_6_l998_99837


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l998_99814

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- State the theorem
theorem f_odd_and_increasing : 
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l998_99814


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l998_99819

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 + 9 * x + 6 < 0) ↔ (-2 < x ∧ x < -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l998_99819


namespace NUMINAMATH_CALUDE_slipper_price_calculation_l998_99881

/-- Given a pair of slippers with original price P, prove that with a 10% discount,
    $5.50 embroidery cost per shoe, $10.00 shipping, and $66.00 total cost,
    the original price P must be $50.00. -/
theorem slipper_price_calculation (P : ℝ) : 
  (0.90 * P + 2 * 5.50 + 10.00 = 66.00) → P = 50.00 := by
  sorry

end NUMINAMATH_CALUDE_slipper_price_calculation_l998_99881


namespace NUMINAMATH_CALUDE_geometric_sequence_product_constant_geometric_sequence_product_l998_99888

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The product of terms equidistant from the beginning and end is constant -/
theorem geometric_sequence_product_constant {a : ℕ → ℝ} (h : geometric_sequence a) :
  ∀ n k : ℕ, a n * a (k + 1 - n) = a 1 * a k :=
sorry

/-- Main theorem: If a₃a₄ = 2 in a geometric sequence, then a₁a₂a₃a₄a₅a₆ = 8 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) 
  (h2 : a 3 * a 4 = 2) : a 1 * a 2 * a 3 * a 4 * a 5 * a 6 = 8 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_constant_geometric_sequence_product_l998_99888


namespace NUMINAMATH_CALUDE_solution_range_l998_99822

theorem solution_range (a : ℝ) : 
  (∃ x : ℝ, x - 1 ≥ a^2 ∧ x - 4 < 2*a) → 
  a ∈ Set.Ioo (-1 : ℝ) 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_range_l998_99822


namespace NUMINAMATH_CALUDE_min_area_rectangle_l998_99824

theorem min_area_rectangle (l w : ℕ) : 
  l > 0 ∧ w > 0 ∧ 2 * (l + w) = 84 → l * w ≥ 41 := by
  sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l998_99824


namespace NUMINAMATH_CALUDE_isabella_trip_l998_99843

def exchange_rate : ℚ := 8 / 5

def spent_aud : ℕ := 80

def remaining_aud (e : ℕ) : ℕ := e + 20

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.repr.toList.map (fun c => c.toNat - '0'.toNat)
  digits.sum

theorem isabella_trip (e : ℕ) : 
  (exchange_rate * e : ℚ) - spent_aud = remaining_aud e →
  e = 167 ∧ sum_of_digits e = 14 := by
  sorry

end NUMINAMATH_CALUDE_isabella_trip_l998_99843


namespace NUMINAMATH_CALUDE_milk_calculation_l998_99865

/-- The initial amount of milk in quarts -/
def initial_milk : ℝ := 1000

/-- The percentage of butterfat in the initial milk -/
def initial_butterfat_percent : ℝ := 4

/-- The percentage of butterfat in the final milk -/
def final_butterfat_percent : ℝ := 3

/-- The amount of cream separated in quarts -/
def separated_cream : ℝ := 50

/-- The percentage of butterfat in the separated cream -/
def cream_butterfat_percent : ℝ := 23

theorem milk_calculation :
  initial_milk = 1000 ∧
  initial_butterfat_percent / 100 * initial_milk =
    final_butterfat_percent / 100 * (initial_milk - separated_cream) +
    cream_butterfat_percent / 100 * separated_cream :=
by sorry

end NUMINAMATH_CALUDE_milk_calculation_l998_99865


namespace NUMINAMATH_CALUDE_sole_mart_meals_l998_99832

theorem sole_mart_meals (initial_meals : ℕ) (given_away : ℕ) (left : ℕ) 
  (h1 : initial_meals = 113)
  (h2 : given_away = 85)
  (h3 : left = 78) :
  initial_meals + (given_away + left) - initial_meals = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_sole_mart_meals_l998_99832


namespace NUMINAMATH_CALUDE_pizza_order_proof_l998_99833

/-- The number of slices in each pizza -/
def slices_per_pizza : ℕ := 12

/-- The number of slices Dean ate -/
def dean_slices : ℕ := slices_per_pizza / 2

/-- The number of slices Frank ate -/
def frank_slices : ℕ := 3

/-- The number of slices Sammy ate -/
def sammy_slices : ℕ := slices_per_pizza / 3

/-- The number of slices left over -/
def leftover_slices : ℕ := 11

/-- The total number of pizzas Dean ordered -/
def total_pizzas : ℕ := 2

theorem pizza_order_proof :
  (dean_slices + frank_slices + sammy_slices + leftover_slices) / slices_per_pizza = total_pizzas :=
by sorry

end NUMINAMATH_CALUDE_pizza_order_proof_l998_99833


namespace NUMINAMATH_CALUDE_number_of_paths_equals_combinations_l998_99879

-- Define the grid size
def gridSize : Nat := 6

-- Define the total number of moves
def totalMoves : Nat := gridSize * 2

-- Define the number of rightward (or downward) moves
def directionMoves : Nat := gridSize

-- Theorem statement
theorem number_of_paths_equals_combinations :
  (Nat.choose totalMoves directionMoves) = 924 := by
  sorry

end NUMINAMATH_CALUDE_number_of_paths_equals_combinations_l998_99879


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l998_99850

/-- Given a cube with surface area 24 sq cm, prove its volume is 8 cubic cm -/
theorem cube_volume_from_surface_area :
  ∀ (side_length : ℝ),
  (6 * side_length^2 = 24) →
  side_length^3 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l998_99850


namespace NUMINAMATH_CALUDE_range_of_a_l998_99807

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → a ≥ x^2 - 2*x - 1) → 
  a ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l998_99807


namespace NUMINAMATH_CALUDE_exists_n_sigma_gt_3n_forall_k_exists_n_sigma_gt_kn_l998_99821

-- Define the sum of divisors function
def sigma (n : ℕ) : ℕ := sorry

-- Theorem 1: There exists a positive integer n such that σ(n) > 3n
theorem exists_n_sigma_gt_3n : ∃ n : ℕ, n > 0 ∧ sigma n > 3 * n := by sorry

-- Theorem 2: For any real number k > 1, there exists a positive integer n such that σ(n) > kn
theorem forall_k_exists_n_sigma_gt_kn : ∀ k : ℝ, k > 1 → ∃ n : ℕ, n > 0 ∧ (sigma n : ℝ) > k * n := by sorry

end NUMINAMATH_CALUDE_exists_n_sigma_gt_3n_forall_k_exists_n_sigma_gt_kn_l998_99821


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l998_99898

theorem point_in_second_quadrant (x : ℝ) : 
  (x - 2 < 0 ∧ 2*x - 1 > 0) → (1/2 < x ∧ x < 2) :=
by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l998_99898


namespace NUMINAMATH_CALUDE_existence_of_a_for_minimum_value_l998_99851

theorem existence_of_a_for_minimum_value (e : Real) (h_e : e > 0) : ∃ a : Real,
  (∀ x : Real, 0 < x ∧ x ≤ e → ax - Real.log x ≥ 3) ∧
  (∃ x : Real, 0 < x ∧ x ≤ e ∧ ax - Real.log x = 3) ∧
  a = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_for_minimum_value_l998_99851


namespace NUMINAMATH_CALUDE_smallest_valid_purchase_l998_99892

def is_valid_purchase (n : ℕ) : Prop :=
  n % 12 = 0 ∧ n % 10 = 0 ∧ n % 9 = 0 ∧ n % 8 = 0 ∧
  n % 18 = 0 ∧ n % 24 = 0 ∧ n % 20 = 0 ∧ n % 30 = 0

theorem smallest_valid_purchase :
  ∃ (n : ℕ), is_valid_purchase n ∧ ∀ (m : ℕ), is_valid_purchase m → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_purchase_l998_99892


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l998_99823

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, n > 0 → a (n + 1) / a n = r

theorem geometric_sequence_ninth_term (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, n > 0 → a (n + 1) / a n = 2^n) →
  a 1 = 1 →
  a 9 = 2^36 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l998_99823


namespace NUMINAMATH_CALUDE_rectangular_plot_fence_l998_99884

theorem rectangular_plot_fence (short_side : ℝ) : 
  short_side > 0 →
  2 * short_side + 2 * (3 * short_side) = 640 →
  short_side = 80 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_fence_l998_99884


namespace NUMINAMATH_CALUDE_sum_inequality_l998_99800

theorem sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) + 
    (1 / 2) * ((a * b) / c + (b * c) / a + (c * a) / b) := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l998_99800


namespace NUMINAMATH_CALUDE_augmented_matrix_of_system_l998_99839

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x + 5 * y + 6 = 0
def equation2 (x y : ℝ) : Prop := 4 * x - 3 * y - 7 = 0

-- Define the augmented matrix
def augmented_matrix : Matrix (Fin 2) (Fin 3) ℝ :=
  !![3, 5, -6;
     4, -3, 7]

-- Theorem statement
theorem augmented_matrix_of_system :
  ∀ (x y : ℝ), equation1 x y ∧ equation2 x y →
  augmented_matrix = !![3, 5, -6; 4, -3, 7] := by
  sorry

end NUMINAMATH_CALUDE_augmented_matrix_of_system_l998_99839


namespace NUMINAMATH_CALUDE_amanda_kimberly_distance_l998_99869

/-- The distance between Amanda's house and Kimberly's house -/
def distance : ℝ := 6

/-- The time Amanda spent walking -/
def walking_time : ℝ := 3

/-- Amanda's walking speed -/
def walking_speed : ℝ := 2

/-- Theorem: The distance between Amanda's house and Kimberly's house is 6 miles -/
theorem amanda_kimberly_distance : distance = walking_time * walking_speed := by
  sorry

end NUMINAMATH_CALUDE_amanda_kimberly_distance_l998_99869


namespace NUMINAMATH_CALUDE_inscribed_circle_distance_relation_l998_99873

/-- A right-angled triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Distance from the center of the inscribed circle to the right-angle vertex -/
  l : ℝ
  /-- Distance from the center of the inscribed circle to one of the other vertices -/
  m : ℝ
  /-- Distance from the center of the inscribed circle to the remaining vertex -/
  n : ℝ
  /-- l, m, and n are positive -/
  l_pos : l > 0
  m_pos : m > 0
  n_pos : n > 0

/-- The theorem relating the distances from the center of the inscribed circle to the vertices -/
theorem inscribed_circle_distance_relation (t : RightTriangleWithInscribedCircle) :
  1 / t.l^2 = 1 / t.m^2 + 1 / t.n^2 + Real.sqrt 2 / (t.m * t.n) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_distance_relation_l998_99873


namespace NUMINAMATH_CALUDE_m_minus_n_values_l998_99847

theorem m_minus_n_values (m n : ℤ) 
  (hm : |m| = 14)
  (hn : |n| = 23)
  (hmn_pos : m + n > 0) :
  m - n = -9 ∨ m - n = -37 := by
sorry

end NUMINAMATH_CALUDE_m_minus_n_values_l998_99847


namespace NUMINAMATH_CALUDE_tenth_term_is_399_l998_99861

def a (n : ℕ) : ℕ := (2*n - 1) * (2*n + 1)

theorem tenth_term_is_399 : a 10 = 399 := by sorry

end NUMINAMATH_CALUDE_tenth_term_is_399_l998_99861


namespace NUMINAMATH_CALUDE_divisor_problem_l998_99867

theorem divisor_problem (f y d : ℕ) : 
  (∃ k : ℕ, f = k * d + 3) →
  (∃ l : ℕ, y = l * d + 4) →
  (∃ m : ℕ, f + y = m * d + 2) →
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l998_99867


namespace NUMINAMATH_CALUDE_alkaline_probability_is_two_fifths_l998_99854

/-- Represents the total number of solutions -/
def total_solutions : ℕ := 5

/-- Represents the number of alkaline solutions -/
def alkaline_solutions : ℕ := 2

/-- Represents the probability of selecting an alkaline solution -/
def alkaline_probability : ℚ := alkaline_solutions / total_solutions

/-- Theorem stating that the probability of selecting an alkaline solution is 2/5 -/
theorem alkaline_probability_is_two_fifths : 
  alkaline_probability = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_alkaline_probability_is_two_fifths_l998_99854


namespace NUMINAMATH_CALUDE_sum_of_products_l998_99897

theorem sum_of_products (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 12)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 21) :
  x*y + y*z + x*z = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l998_99897


namespace NUMINAMATH_CALUDE_chocolate_pieces_per_box_l998_99852

theorem chocolate_pieces_per_box 
  (total_boxes : ℕ) 
  (given_away : ℕ) 
  (remaining_pieces : ℕ) 
  (h1 : total_boxes = 12) 
  (h2 : given_away = 7) 
  (h3 : remaining_pieces = 30) : 
  remaining_pieces / (total_boxes - given_away) = 6 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_pieces_per_box_l998_99852


namespace NUMINAMATH_CALUDE_second_calculator_price_l998_99887

def total_calculators : ℕ := 85
def total_sales : ℚ := 3875
def first_calculator_count : ℕ := 35
def first_calculator_price : ℚ := 67

theorem second_calculator_price :
  let second_calculator_count := total_calculators - first_calculator_count
  let first_calculator_total := first_calculator_count * first_calculator_price
  let second_calculator_total := total_sales - first_calculator_total
  second_calculator_total / second_calculator_count = 30.6 := by
sorry

end NUMINAMATH_CALUDE_second_calculator_price_l998_99887


namespace NUMINAMATH_CALUDE_ellipse_point_distance_l998_99886

theorem ellipse_point_distance (P : ℝ × ℝ) :
  (P.1^2 / 6 + P.2^2 / 2 = 1) →
  (Real.sqrt ((P.1 + 2)^2 + P.2^2) + Real.sqrt ((P.1 - 2)^2 + P.2^2) +
   Real.sqrt (P.1^2 + (P.2 + 1)^2) + Real.sqrt (P.1^2 + (P.2 - 1)^2) = 4 * Real.sqrt 6) →
  (abs P.2 = Real.sqrt (6 / 13)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_point_distance_l998_99886


namespace NUMINAMATH_CALUDE_tension_force_in_rod_system_l998_99816

/-- The tension force in a weightless rod system with a suspended weight. -/
theorem tension_force_in_rod_system (m g : ℝ) (T₀ T₁ T₂ : ℝ) : 
  m = 2 →
  g = 10 →
  T₂ = 1/4 * m * g →
  T₁ = 3/4 * m * g →
  T₀ * (1/4) + T₂ = T₁ * (1/2) →
  T₀ = 10 := by sorry

end NUMINAMATH_CALUDE_tension_force_in_rod_system_l998_99816
