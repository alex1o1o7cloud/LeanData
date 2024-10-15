import Mathlib

namespace NUMINAMATH_CALUDE_circles_tangent_m_value_l2445_244527

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1
def C₂ (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 6*x - 8*y + m = 0

-- Define the tangency condition
def are_tangent (C₁ C₂ : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y ∧
  ∀ (x' y' : ℝ), C₁ x' y' ∧ C₂ x' y' → (x' = x ∧ y' = y)

-- Theorem statement
theorem circles_tangent_m_value :
  are_tangent C₁ (C₂ · · 9) → ∀ m : ℝ, are_tangent C₁ (C₂ · · m) → m = 9 :=
by sorry

end NUMINAMATH_CALUDE_circles_tangent_m_value_l2445_244527


namespace NUMINAMATH_CALUDE_inequality_holds_l2445_244531

theorem inequality_holds (p q : ℝ) (h_p : 0 < p) (h_p_upper : p < 2) (h_q : 0 < q) :
  (4 * (p * q^2 + 2 * p^2 * q + 2 * q^2 + 2 * p * q)) / (p + q) > 3 * p^2 * q :=
by sorry

end NUMINAMATH_CALUDE_inequality_holds_l2445_244531


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l2445_244513

/-- Represents a digit (1-9) -/
def Digit := { n : ℕ // 1 ≤ n ∧ n ≤ 9 }

/-- Converts a two-digit number to its decimal representation -/
def twoDigitToNum (a b : Digit) : ℕ := 10 * a.val + b.val

/-- Converts a three-digit number with all digits the same to its decimal representation -/
def threeDigitSameToNum (c : Digit) : ℕ := 100 * c.val + 10 * c.val + c.val

theorem cryptarithm_solution :
  ∃! (a b c : Digit),
    a.val ≠ b.val ∧ b.val ≠ c.val ∧ a.val ≠ c.val ∧
    twoDigitToNum a b + a.val * threeDigitSameToNum c = 247 ∧
    a.val = 2 ∧ b.val = 5 ∧ c.val = 1 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l2445_244513


namespace NUMINAMATH_CALUDE_new_barbell_cost_l2445_244558

def old_barbell_cost : ℝ := 250
def price_increase_percentage : ℝ := 30

theorem new_barbell_cost : 
  old_barbell_cost * (1 + price_increase_percentage / 100) = 325 := by
  sorry

end NUMINAMATH_CALUDE_new_barbell_cost_l2445_244558


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2445_244529

theorem solution_set_quadratic_inequality :
  Set.Icc (-(1/2) : ℝ) 1 = {x : ℝ | 2 * x^2 - x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2445_244529


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l2445_244516

-- Define the sets M and N
def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

-- Define the interval (1, 2]
def interval_one_two : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- Theorem statement
theorem intersection_equals_interval : M ∩ N = interval_one_two := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l2445_244516


namespace NUMINAMATH_CALUDE_opposite_numbers_product_l2445_244506

theorem opposite_numbers_product (x y : ℝ) : 
  (|x - 3| + |y + 1| = 0) → xy = -3 := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_product_l2445_244506


namespace NUMINAMATH_CALUDE_pet_food_cost_differences_l2445_244539

/-- Calculates the total cost including tax -/
def totalCostWithTax (quantity : Float) (price : Float) (taxRate : Float) : Float :=
  quantity * price * (1 + taxRate)

/-- Theorem: Sum of differences between pet food costs -/
theorem pet_food_cost_differences (dogQuantity catQuantity birdQuantity fishQuantity : Float)
  (dogPrice catPrice birdPrice fishPrice : Float) (taxRate : Float)
  (h1 : dogQuantity = 600.5)
  (h2 : catQuantity = 327.25)
  (h3 : birdQuantity = 415.75)
  (h4 : fishQuantity = 248.5)
  (h5 : dogPrice = 24.99)
  (h6 : catPrice = 19.49)
  (h7 : birdPrice = 15.99)
  (h8 : fishPrice = 13.89)
  (h9 : taxRate = 0.065) :
  let dogCost := totalCostWithTax dogQuantity dogPrice taxRate
  let catCost := totalCostWithTax catQuantity catPrice taxRate
  let birdCost := totalCostWithTax birdQuantity birdPrice taxRate
  let fishCost := totalCostWithTax fishQuantity fishPrice taxRate
  (dogCost - catCost) + (catCost - birdCost) + (birdCost - fishCost) = 12301.9002 :=
by sorry

end NUMINAMATH_CALUDE_pet_food_cost_differences_l2445_244539


namespace NUMINAMATH_CALUDE_y_coordinate_of_point_p_l2445_244574

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def foci_distance : ℝ := 6

-- Define the sum of distances from P to foci
def sum_distances_to_foci : ℝ := 10

-- Define the radius of the inscribed circle
def inscribed_circle_radius : ℝ := 1

-- Main theorem
theorem y_coordinate_of_point_p :
  ∀ x y : ℝ,
  is_on_ellipse x y →
  x ≥ 0 →
  y > 0 →
  y = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_y_coordinate_of_point_p_l2445_244574


namespace NUMINAMATH_CALUDE_player_A_wins_l2445_244523

/-- Represents a card with a digit from 0 to 6 -/
inductive Card : Type
| zero | one | two | three | four | five | six

/-- Represents a player in the game -/
inductive Player : Type
| A | B

/-- Represents the state of the game -/
structure GameState :=
(remaining_cards : List Card)
(player_A_cards : List Card)
(player_B_cards : List Card)
(current_player : Player)

/-- Checks if a list of cards can form a number divisible by 17 -/
def can_form_divisible_by_17 (cards : List Card) : Bool :=
  sorry

/-- Determines the winner of the game given optimal play -/
def optimal_play_winner (initial_state : GameState) : Player :=
  sorry

/-- The main theorem stating that Player A wins with optimal play -/
theorem player_A_wins :
  ∀ (initial_state : GameState),
    initial_state.remaining_cards = [Card.zero, Card.one, Card.two, Card.three, Card.four, Card.five, Card.six] →
    initial_state.player_A_cards = [] →
    initial_state.player_B_cards = [] →
    initial_state.current_player = Player.A →
    optimal_play_winner initial_state = Player.A :=
  sorry

end NUMINAMATH_CALUDE_player_A_wins_l2445_244523


namespace NUMINAMATH_CALUDE_lamppost_combinations_lamppost_problem_l2445_244583

theorem lamppost_combinations : Nat → Nat → Nat
| n, k => Nat.choose n k

theorem lamppost_problem :
  let total_posts : Nat := 11
  let posts_to_turn_off : Nat := 3
  let available_positions : Nat := total_posts - 4  -- Subtracting 2 for each end and 2 for adjacent positions
  lamppost_combinations available_positions posts_to_turn_off = 35 := by
  sorry

end NUMINAMATH_CALUDE_lamppost_combinations_lamppost_problem_l2445_244583


namespace NUMINAMATH_CALUDE_greatest_number_with_special_remainder_l2445_244538

theorem greatest_number_with_special_remainder : ∃ n : ℕ, 
  (n % 91 = (n / 91) ^ 2) ∧ 
  (∀ m : ℕ, m > n → m % 91 ≠ (m / 91) ^ 2) ∧
  n = 900 := by
sorry

end NUMINAMATH_CALUDE_greatest_number_with_special_remainder_l2445_244538


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2445_244561

def A : Set ℝ := {x | x^2 ≠ 1}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem possible_values_of_a (a : ℝ) (h : A ∪ B a = A) : a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2445_244561


namespace NUMINAMATH_CALUDE_dollar_cube_difference_l2445_244597

-- Define the $ operation for real numbers
def dollar (a b : ℝ) : ℝ := (a - b)^3

-- Theorem statement
theorem dollar_cube_difference (x y : ℝ) :
  dollar ((x - y)^3) ((y - x)^3) = -8 * (y - x)^9 := by
  sorry

end NUMINAMATH_CALUDE_dollar_cube_difference_l2445_244597


namespace NUMINAMATH_CALUDE_production_difference_formula_l2445_244534

/-- The number of widgets David produces per hour on Monday -/
def w (t : ℝ) : ℝ := 2 * t

/-- The number of hours David works on Monday -/
def monday_hours (t : ℝ) : ℝ := t

/-- The number of hours David works on Tuesday -/
def tuesday_hours (t : ℝ) : ℝ := t - 1

/-- The number of widgets David produces per hour on Tuesday -/
def tuesday_rate (t : ℝ) : ℝ := w t + 5

/-- The difference in widget production between Monday and Tuesday -/
def production_difference (t : ℝ) : ℝ :=
  w t * monday_hours t - tuesday_rate t * tuesday_hours t

theorem production_difference_formula (t : ℝ) :
  production_difference t = -3 * t + 5 := by
  sorry

end NUMINAMATH_CALUDE_production_difference_formula_l2445_244534


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_zero_l2445_244546

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | |x - 1| < 2}

theorem M_intersect_N_equals_zero : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_zero_l2445_244546


namespace NUMINAMATH_CALUDE_julios_fishing_time_l2445_244590

/-- Julio's fishing problem -/
theorem julios_fishing_time (catch_rate : ℕ) (fish_lost : ℕ) (final_fish : ℕ) (h : ℕ) : 
  catch_rate = 7 → fish_lost = 15 → final_fish = 48 → 
  catch_rate * h - fish_lost = final_fish → h = 9 := by
sorry

end NUMINAMATH_CALUDE_julios_fishing_time_l2445_244590


namespace NUMINAMATH_CALUDE_common_tangent_lines_C₁_C₂_l2445_244579

/-- Circle C₁ with equation x² + y² - 2x = 0 -/
def C₁ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 = 0}

/-- Circle C₂ with equation x² + (y - √3)² = 4 -/
def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + (p.2 - Real.sqrt 3)^2 = 4}

/-- The number of common tangent lines between two circles -/
def commonTangentLines (c1 c2 : Set (ℝ × ℝ)) : ℕ :=
  sorry

/-- Theorem stating that the number of common tangent lines between C₁ and C₂ is 2 -/
theorem common_tangent_lines_C₁_C₂ :
  commonTangentLines C₁ C₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_common_tangent_lines_C₁_C₂_l2445_244579


namespace NUMINAMATH_CALUDE_simplify_expression_l2445_244584

theorem simplify_expression : (2^5 + 7^3) * (2^3 - (-2)^2)^8 = 24576000 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2445_244584


namespace NUMINAMATH_CALUDE_ap_terms_count_l2445_244591

theorem ap_terms_count (n : ℕ) (a d : ℚ) : 
  Odd n → 
  (n + 1) / 2 * (2 * a + ((n + 1) / 2 - 1) * d) = 30 →
  (n - 1) / 2 * (2 * (a + d) + ((n - 1) / 2 - 1) * d) = 36 →
  n / 2 * (2 * a + (n - 1) * d) = 66 →
  a + (n - 1) * d - a = 12 →
  n = 9 := by
sorry

end NUMINAMATH_CALUDE_ap_terms_count_l2445_244591


namespace NUMINAMATH_CALUDE_rectangular_garden_width_l2445_244504

theorem rectangular_garden_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 768 →
  width = 16 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_width_l2445_244504


namespace NUMINAMATH_CALUDE_gold_bar_distribution_l2445_244555

theorem gold_bar_distribution (initial_bars : ℕ) (lost_bars : ℕ) (friends : ℕ) 
  (h1 : initial_bars = 100)
  (h2 : lost_bars = 20)
  (h3 : friends = 4)
  (h4 : friends > 0) :
  (initial_bars - lost_bars) / friends = 20 := by
  sorry

end NUMINAMATH_CALUDE_gold_bar_distribution_l2445_244555


namespace NUMINAMATH_CALUDE_angle_A_value_perimeter_range_l2445_244520

-- Define the triangle
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : a * Real.cos C + Real.sqrt 3 * Real.sin C - b - c = 0
axiom positive_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom angle_sum : A + B + C = Real.pi
axiom law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Part 1: Prove that A = π/3
theorem angle_A_value : A = Real.pi / 3 := by sorry

-- Part 2: Prove the perimeter range
theorem perimeter_range (h_acute : A < Real.pi/2 ∧ B < Real.pi/2 ∧ C < Real.pi/2) (h_c : c = 3) :
  (3 * Real.sqrt 3 + 9) / 2 < a + b + c ∧ a + b + c < 9 + 3 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_A_value_perimeter_range_l2445_244520


namespace NUMINAMATH_CALUDE_divisor_sum_condition_l2445_244501

theorem divisor_sum_condition (n : ℕ+) :
  (∃ (a b c : ℕ+), a + b + c = n ∧ a ∣ b ∧ b ∣ c ∧ a < b ∧ b < c) ↔ 
  n ∉ ({1, 2, 3, 4, 5, 6, 8, 12, 24} : Set ℕ+) :=
by sorry

end NUMINAMATH_CALUDE_divisor_sum_condition_l2445_244501


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2445_244551

/-- Given a triangle with sides in the ratio 1/2 : 1/3 : 1/4 and longest side 48 cm, its perimeter is 104 cm -/
theorem triangle_perimeter (a b c : ℝ) (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : a / b = 3 / 2 ∧ a / c = 2 ∧ b / c = 4 / 3) (h3 : a = 48) : 
  a + b + c = 104 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2445_244551


namespace NUMINAMATH_CALUDE_joe_cars_count_l2445_244526

/-- Proves that Joe will have 62 cars after getting 12 more cars, given he initially had 50 cars. -/
theorem joe_cars_count (initial_cars : ℕ) (additional_cars : ℕ) : 
  initial_cars = 50 → additional_cars = 12 → initial_cars + additional_cars = 62 := by
  sorry

end NUMINAMATH_CALUDE_joe_cars_count_l2445_244526


namespace NUMINAMATH_CALUDE_f_10_eq_756_l2445_244550

/-- The polynomial function f(x) = x^3 - 2x^2 - 5x + 6 -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 6

/-- Theorem: f(10) = 756 -/
theorem f_10_eq_756 : f 10 = 756 := by
  sorry

end NUMINAMATH_CALUDE_f_10_eq_756_l2445_244550


namespace NUMINAMATH_CALUDE_g_composition_result_l2445_244512

-- Define the complex function g
noncomputable def g (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^3 else -z^3

-- State the theorem
theorem g_composition_result :
  g (g (g (g (1 + Complex.I)))) = -8192 - 45056 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_g_composition_result_l2445_244512


namespace NUMINAMATH_CALUDE_evaluate_expression_l2445_244568

theorem evaluate_expression : 10010 - 12 * 3 * 2 = 9938 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2445_244568


namespace NUMINAMATH_CALUDE_max_extra_time_matches_2016_teams_l2445_244592

/-- Represents a hockey tournament -/
structure HockeyTournament where
  num_teams : Nat
  regular_win_points : Nat
  regular_loss_points : Nat
  extra_time_win_points : Nat
  extra_time_loss_points : Nat

/-- The maximum number of matches that could have ended in extra time -/
def max_extra_time_matches (tournament : HockeyTournament) : Nat :=
  sorry

/-- Theorem stating the maximum number of extra time matches for the given tournament -/
theorem max_extra_time_matches_2016_teams 
  (tournament : HockeyTournament)
  (h1 : tournament.num_teams = 2016)
  (h2 : tournament.regular_win_points = 3)
  (h3 : tournament.regular_loss_points = 0)
  (h4 : tournament.extra_time_win_points = 2)
  (h5 : tournament.extra_time_loss_points = 1) :
  max_extra_time_matches tournament = 1512 :=
sorry

end NUMINAMATH_CALUDE_max_extra_time_matches_2016_teams_l2445_244592


namespace NUMINAMATH_CALUDE_computer_accessories_cost_l2445_244536

/-- Proves that the amount spent on computer accessories is $12 -/
theorem computer_accessories_cost (initial_amount : ℕ) (snack_cost : ℕ) (remaining_amount : ℕ) : 
  initial_amount = 48 →
  snack_cost = 8 →
  remaining_amount = initial_amount / 2 + 4 →
  initial_amount - (remaining_amount + snack_cost) = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_computer_accessories_cost_l2445_244536


namespace NUMINAMATH_CALUDE_cubic_function_property_l2445_244560

/-- Given a cubic function f(x) = ax³ + bx - 4 where a and b are constants,
    if f(-2) = 2, then f(2) = -10 -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a * x^3 + b * x - 4)
    (h2 : f (-2) = 2) : 
  f 2 = -10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2445_244560


namespace NUMINAMATH_CALUDE_consecutive_cubes_l2445_244540

theorem consecutive_cubes (a b c d : ℤ) (y z w x v : ℤ) : 
  (d = c + 1 ∧ c = b + 1 ∧ b = a + 1) → 
  (v = x + 1 ∧ x = w + 1 ∧ w = z + 1 ∧ z = y + 1) →
  ((a^3 + b^3 + c^3 = d^3) ↔ (a = 3 ∧ b = 4 ∧ c = 5 ∧ d = 6)) ∧
  (y^3 + z^3 + w^3 + x^3 ≠ v^3) := by
sorry

end NUMINAMATH_CALUDE_consecutive_cubes_l2445_244540


namespace NUMINAMATH_CALUDE_isosceles_triangle_angles_l2445_244563

/-- An isosceles triangle with one angle of 40 degrees has two equal angles of 70 degrees each. -/
theorem isosceles_triangle_angles (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = b →            -- Two angles are equal (isosceles property)
  c = 40 →           -- The third angle is 40°
  a = 70 :=          -- Each of the two equal angles is 70°
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_angles_l2445_244563


namespace NUMINAMATH_CALUDE_initial_storks_count_storks_on_fence_l2445_244522

/-- Given a fence with birds and storks, prove the initial number of storks. -/
theorem initial_storks_count (initial_birds : ℕ) (additional_birds : ℕ) (stork_bird_difference : ℕ) : ℕ :=
  let final_birds := initial_birds + additional_birds
  let storks := final_birds + stork_bird_difference
  storks

/-- Prove that the number of storks initially on the fence is 6. -/
theorem storks_on_fence :
  initial_storks_count 2 3 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_initial_storks_count_storks_on_fence_l2445_244522


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2445_244575

theorem complex_fraction_simplification :
  let z₁ : ℂ := Complex.mk 5 7
  let z₂ : ℂ := Complex.mk 2 3
  z₁ / z₂ = Complex.mk (31 / 13) (-1 / 13) := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2445_244575


namespace NUMINAMATH_CALUDE_five_coins_all_heads_or_tails_prob_l2445_244587

/-- The probability of getting all heads or all tails when flipping n fair coins -/
def all_heads_or_tails_prob (n : ℕ) : ℚ :=
  2 / 2^n

/-- Theorem: The probability of getting all heads or all tails when flipping 5 fair coins is 1/16 -/
theorem five_coins_all_heads_or_tails_prob :
  all_heads_or_tails_prob 5 = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_five_coins_all_heads_or_tails_prob_l2445_244587


namespace NUMINAMATH_CALUDE_f_properties_l2445_244576

def f (x : ℝ) := x^3 - 3*x

theorem f_properties :
  (∀ y, (∃ x, x = 0 ∧ y = f x) → y = 0) ∧
  (∀ x, x < -1 → (∀ h > 0, f (x + h) > f x)) ∧
  (∀ x, x > 1 → (∀ h > 0, f (x + h) > f x)) ∧
  (∀ x, -1 < x ∧ x < 1 → (∀ h > 0, f (x + h) < f x)) ∧
  (f (-1) = 2) ∧
  (f 1 = -2) ∧
  (∀ x, f x ≤ 2) ∧
  (∀ x, f x ≥ -2) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2445_244576


namespace NUMINAMATH_CALUDE_range_of_T_l2445_244502

theorem range_of_T (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x + y + z = 30) (h5 : 3 * x + y - z = 50) :
  let T := 5 * x + 4 * y + 2 * z
  ∃ (T_min T_max : ℝ), T_min = 120 ∧ T_max = 130 ∧ T_min ≤ T ∧ T ≤ T_max :=
by sorry

end NUMINAMATH_CALUDE_range_of_T_l2445_244502


namespace NUMINAMATH_CALUDE_min_distance_complex_circles_l2445_244544

theorem min_distance_complex_circles (z w : ℂ) 
  (hz : Complex.abs (z + 2 + 4*I) = 2)
  (hw : Complex.abs (w - 5 - 6*I) = 4) :
  ∃ (m : ℝ), m = Real.sqrt 149 - 6 ∧ ∀ (z' w' : ℂ), 
    Complex.abs (z' + 2 + 4*I) = 2 → 
    Complex.abs (w' - 5 - 6*I) = 4 → 
    Complex.abs (z' - w') ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_distance_complex_circles_l2445_244544


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l2445_244542

theorem quadratic_roots_real_and_equal : ∃ x : ℝ, 
  x^2 - 4*x*Real.sqrt 5 + 20 = 0 ∧ 
  (∀ y : ℝ, y^2 - 4*y*Real.sqrt 5 + 20 = 0 → y = x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l2445_244542


namespace NUMINAMATH_CALUDE_three_numbers_sum_l2445_244549

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ordered numbers
  b = 10 ∧  -- Median is 10
  (a + b + c) / 3 = a + 15 ∧  -- Mean is 15 more than least
  (a + b + c) / 3 = c - 20  -- Mean is 20 less than greatest
  → a + b + c = 45 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l2445_244549


namespace NUMINAMATH_CALUDE_reading_program_classes_l2445_244503

/-- The number of classes in a school with a specific reading program. -/
def number_of_classes (s : ℕ) : ℕ :=
  if s = 0 then 0 else 1

theorem reading_program_classes (s : ℕ) (h : s > 0) :
  let books_per_student_per_year := 4 * 12
  let total_books_read := 48
  number_of_classes s = 1 ∧ s * books_per_student_per_year = total_books_read :=
by sorry

end NUMINAMATH_CALUDE_reading_program_classes_l2445_244503


namespace NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2445_244553

def is_monic_quartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + (p 0)

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) :
  is_monic_quartic p →
  p 1 = 3 →
  p 2 = 7 →
  p 3 = 13 →
  p 4 = 21 →
  p 6 = 163 := by
sorry

end NUMINAMATH_CALUDE_monic_quartic_polynomial_value_l2445_244553


namespace NUMINAMATH_CALUDE_max_roses_for_1000_budget_l2445_244552

/-- Represents the price of roses for different quantities -/
structure RosePrices where
  individual : ℚ
  dozen : ℚ
  two_dozen : ℚ
  five_dozen : ℚ
  hundred : ℚ

/-- Calculates the maximum number of roses that can be purchased with a given budget -/
def maxRoses (prices : RosePrices) (budget : ℚ) : ℕ :=
  sorry

/-- The theorem stating that given the specific rose prices and a $1000 budget, 
    the maximum number of roses that can be purchased is 548 -/
theorem max_roses_for_1000_budget :
  let prices : RosePrices := {
    individual := 5.3,
    dozen := 36,
    two_dozen := 50,
    five_dozen := 110,
    hundred := 180
  }
  maxRoses prices 1000 = 548 := by
  sorry

end NUMINAMATH_CALUDE_max_roses_for_1000_budget_l2445_244552


namespace NUMINAMATH_CALUDE_ellipse_k_values_l2445_244533

def ellipse_equation (x y k : ℝ) : Prop := x^2/5 + y^2/k = 1

def eccentricity (e : ℝ) : Prop := e = Real.sqrt 10 / 5

theorem ellipse_k_values (k : ℝ) :
  (∃ x y, ellipse_equation x y k) ∧ eccentricity (Real.sqrt 10 / 5) →
  k = 3 ∨ k = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_k_values_l2445_244533


namespace NUMINAMATH_CALUDE_marcus_rachel_percentage_l2445_244509

def marcus_score : ℕ := 5 * 3 + 10 * 2 + 8 * 1 + 2 * 4
def brian_score : ℕ := 6 * 3 + 8 * 2 + 9 * 1 + 1 * 4
def rachel_score : ℕ := 4 * 3 + 12 * 2 + 7 * 1 + 0 * 4
def team_total_score : ℕ := 150

theorem marcus_rachel_percentage :
  (marcus_score + rachel_score : ℚ) / team_total_score * 100 = 62.67 := by
  sorry

end NUMINAMATH_CALUDE_marcus_rachel_percentage_l2445_244509


namespace NUMINAMATH_CALUDE_john_savings_l2445_244564

/-- Calculates the yearly savings when splitting an apartment --/
def yearly_savings (old_rent : ℕ) (price_increase_percent : ℕ) (num_people : ℕ) : ℕ :=
  let new_rent := old_rent + old_rent * price_increase_percent / 100
  let individual_share := new_rent / num_people
  let monthly_savings := old_rent - individual_share
  monthly_savings * 12

/-- Theorem: John saves $7680 per year by splitting the new apartment --/
theorem john_savings : yearly_savings 1200 40 3 = 7680 := by
  sorry

end NUMINAMATH_CALUDE_john_savings_l2445_244564


namespace NUMINAMATH_CALUDE_closest_to_N_div_M_l2445_244580

/-- Mersenne prime M -/
def M : ℕ := 2^127 - 1

/-- Mersenne prime N -/
def N : ℕ := 2^607 - 1

/-- Approximation of log_2 -/
def log2_approx : ℝ := 0.3010

/-- Theorem stating that 10^144 is closest to N/M among given options -/
theorem closest_to_N_div_M :
  let options : List ℝ := [10^140, 10^142, 10^144, 10^146]
  ∀ x ∈ options, |((N : ℝ) / M) - 10^144| ≤ |((N : ℝ) / M) - x| :=
sorry

end NUMINAMATH_CALUDE_closest_to_N_div_M_l2445_244580


namespace NUMINAMATH_CALUDE_toy_car_factory_ratio_l2445_244586

/-- The ratio of cars made today to cars made yesterday -/
def car_ratio (cars_yesterday cars_today : ℕ) : ℚ :=
  cars_today / cars_yesterday

theorem toy_car_factory_ratio : 
  let cars_yesterday : ℕ := 60
  let total_cars : ℕ := 180
  let cars_today : ℕ := total_cars - cars_yesterday
  car_ratio cars_yesterday cars_today = 2 := by
sorry

end NUMINAMATH_CALUDE_toy_car_factory_ratio_l2445_244586


namespace NUMINAMATH_CALUDE_point_b_coordinates_l2445_244569

/-- Given points A and C, and the condition that vector AB is -2 times vector BC,
    prove that the coordinates of point B are (-2, -1). -/
theorem point_b_coordinates (A B C : ℝ × ℝ) : 
  A = (2, 3) → 
  C = (0, 1) → 
  B - A = -2 * (C - B) →
  B = (-2, -1) := by
sorry

end NUMINAMATH_CALUDE_point_b_coordinates_l2445_244569


namespace NUMINAMATH_CALUDE_triangle_properties_l2445_244517

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  A + B + C = π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  a / Real.sin A = 2 * Real.sqrt 3 ∧
  a * Real.sin C + Real.sqrt 3 * c * Real.cos A = 0 →
  a = 3 ∧
  (b + c = Real.sqrt 11 →
    1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2445_244517


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2445_244588

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- Theorem: The common difference of an arithmetic sequence is 1, 
    given S_3 = 6 and a_1 = 1 -/
theorem arithmetic_sequence_common_difference :
  ∀ d : ℚ, S 3 1 d = 6 → d = 1 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2445_244588


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_terms_l2445_244514

def arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i j, i < j → j ≤ n → a j - a i = (j - i : ℝ) * (a 2 - a 1)

theorem arithmetic_sequence_n_terms
  (a : ℕ → ℝ) (n : ℕ)
  (h1 : arithmetic_sequence a n)
  (h2 : a 1 + a 2 + a 3 = 20)
  (h3 : a (n-2) + a (n-1) + a n = 130)
  (h4 : (Finset.range n).sum a = 200) :
  n = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_terms_l2445_244514


namespace NUMINAMATH_CALUDE_power_function_through_point_l2445_244594

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

-- Define the condition that f passes through (9, 3)
def passesThroughPoint (f : ℝ → ℝ) : Prop :=
  f 9 = 3

-- Theorem statement
theorem power_function_through_point (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) (h2 : passesThroughPoint f) : f 25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2445_244594


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l2445_244548

-- Define the polynomial and divisor
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 9*x - 6
def g (x : ℝ) : ℝ := x^2 - x + 4

-- Define the quotient and remainder
def q (x : ℝ) : ℝ := x - 3
def r (x : ℝ) : ℝ := 2*x + 6

-- Theorem statement
theorem polynomial_division_theorem :
  ∀ x : ℝ, f x = g x * q x + r x :=
by sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l2445_244548


namespace NUMINAMATH_CALUDE_worker_count_l2445_244557

theorem worker_count : ∃ (x : ℕ), 
  x > 0 ∧ 
  (7200 / x + 400) * (x - 3) = 7200 ∧ 
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_worker_count_l2445_244557


namespace NUMINAMATH_CALUDE_exists_k_not_equal_f_diff_l2445_244532

/-- f(n) is the largest integer k such that 2^k divides n -/
def f (n : ℕ) : ℕ := Nat.log2 (n.gcd (2^n))

/-- Theorem statement -/
theorem exists_k_not_equal_f_diff (n : ℕ) (h : n ≥ 2) (a : Fin n → ℕ)
  (h_sorted : ∀ i j, i < j → a i < a j) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧
    ∀ i j : Fin n, j ≤ i → f (a i - a j) ≠ k :=
  sorry

end NUMINAMATH_CALUDE_exists_k_not_equal_f_diff_l2445_244532


namespace NUMINAMATH_CALUDE_horse_grazing_width_l2445_244559

/-- Represents a rectangular field with a horse tethered to one corner. -/
structure GrazingField where
  length : ℝ
  width : ℝ
  rope_length : ℝ
  grazing_area : ℝ

/-- Theorem stating the width of the field that the horse can graze. -/
theorem horse_grazing_width (field : GrazingField)
  (h_length : field.length = 45)
  (h_rope : field.rope_length = 22)
  (h_area : field.grazing_area = 380.132711084365)
  : field.width = 22 := by
  sorry

end NUMINAMATH_CALUDE_horse_grazing_width_l2445_244559


namespace NUMINAMATH_CALUDE_six_containing_triangles_l2445_244572

/-- Represents a quadrilateral composed of small equilateral triangles -/
structure TriangleQuadrilateral where
  /-- The total number of small equilateral triangles in the quadrilateral -/
  total_triangles : ℕ
  /-- The number of small triangles per side of the largest equilateral triangle -/
  max_side_length : ℕ
  /-- Assertion that the total number of triangles is 18 -/
  h_total : total_triangles = 18

/-- Counts the number of equilateral triangles containing a marked triangle -/
def count_containing_triangles (q : TriangleQuadrilateral) : ℕ :=
  sorry

/-- Theorem stating that there are exactly 6 equilateral triangles containing the marked triangle -/
theorem six_containing_triangles (q : TriangleQuadrilateral) :
  count_containing_triangles q = 6 :=
sorry

end NUMINAMATH_CALUDE_six_containing_triangles_l2445_244572


namespace NUMINAMATH_CALUDE_cannot_afford_both_phones_l2445_244541

/-- Represents the financial situation of Alexander and Natalia --/
structure FinancialSituation where
  alexander_salary : ℕ
  natalia_salary : ℕ
  utilities_expenses : ℕ
  loan_expenses : ℕ
  cultural_expenses : ℕ
  vacation_savings : ℕ
  dining_expenses : ℕ
  phone_a_cost : ℕ
  phone_b_cost : ℕ

/-- Theorem stating that Alexander and Natalia cannot afford both phones --/
theorem cannot_afford_both_phones (fs : FinancialSituation) 
  (h1 : fs.alexander_salary = 125000)
  (h2 : fs.natalia_salary = 61000)
  (h3 : fs.utilities_expenses = 17000)
  (h4 : fs.loan_expenses = 15000)
  (h5 : fs.cultural_expenses = 7000)
  (h6 : fs.vacation_savings = 20000)
  (h7 : fs.dining_expenses = 60000)
  (h8 : fs.phone_a_cost = 57000)
  (h9 : fs.phone_b_cost = 37000) :
  fs.alexander_salary + fs.natalia_salary - 
  (fs.utilities_expenses + fs.loan_expenses + fs.cultural_expenses + 
   fs.vacation_savings + fs.dining_expenses) < 
  fs.phone_a_cost + fs.phone_b_cost :=
by sorry

end NUMINAMATH_CALUDE_cannot_afford_both_phones_l2445_244541


namespace NUMINAMATH_CALUDE_no_integer_solution_cube_equation_l2445_244507

theorem no_integer_solution_cube_equation :
  ¬ ∃ (x y z : ℤ), x^3 + y^3 = z^3 + 4 := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_cube_equation_l2445_244507


namespace NUMINAMATH_CALUDE_sum_base6_1452_2354_l2445_244547

/-- Converts a base 6 number represented as a list of digits to its decimal equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- Converts a decimal number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 6) ((m % 6) :: acc)
    go n []

/-- The main theorem: sum of 1452₆ and 2354₆ in base 6 is 4250₆ -/
theorem sum_base6_1452_2354 :
  decimalToBase6 (base6ToDecimal [1, 4, 5, 2] + base6ToDecimal [2, 3, 5, 4]) = [4, 2, 5, 0] := by
  sorry

end NUMINAMATH_CALUDE_sum_base6_1452_2354_l2445_244547


namespace NUMINAMATH_CALUDE_taco_truck_revenue_is_66_l2445_244535

/-- Calculates the total revenue of a taco truck during lunch rush -/
def taco_truck_revenue (soft_taco_price hard_taco_price : ℕ)
  (family_soft_tacos family_hard_tacos : ℕ)
  (additional_customers : ℕ) : ℕ :=
  let total_soft_tacos := family_soft_tacos + 2 * additional_customers
  let soft_taco_revenue := soft_taco_price * total_soft_tacos
  let hard_taco_revenue := hard_taco_price * family_hard_tacos
  soft_taco_revenue + hard_taco_revenue

/-- The total revenue of the taco truck during lunch rush is $66 -/
theorem taco_truck_revenue_is_66 :
  taco_truck_revenue 2 5 3 4 10 = 66 := by
  sorry

end NUMINAMATH_CALUDE_taco_truck_revenue_is_66_l2445_244535


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2445_244545

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 7, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l2445_244545


namespace NUMINAMATH_CALUDE_car_speed_l2445_244578

/-- Given a car that travels 495 km in 5 hours, its speed is 99 km/h. -/
theorem car_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 495 ∧ time = 5 ∧ speed = distance / time → speed = 99 :=
by sorry

end NUMINAMATH_CALUDE_car_speed_l2445_244578


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2445_244521

theorem arithmetic_mean_problem (a b c d : ℝ) : 
  (a + b + c + d + 106) / 5 = 92 →
  (a + b + c + d) / 4 = 88.5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2445_244521


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2445_244505

/-- Proves that the speed of a boat in still water is 24 km/hr, given the speed of the stream and downstream travel information. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 168)
  (h3 : downstream_time = 6)
  : ∃ (boat_speed : ℝ), boat_speed = 24 ∧ downstream_distance = (boat_speed + stream_speed) * downstream_time :=
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2445_244505


namespace NUMINAMATH_CALUDE_investment_growth_l2445_244543

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof that $30,697 grows to at least $50,000 in 10 years at 5% interest -/
theorem investment_growth :
  let initial_deposit : ℝ := 30697
  let interest_rate : ℝ := 0.05
  let years : ℕ := 10
  let target_amount : ℝ := 50000
  compound_interest initial_deposit interest_rate years ≥ target_amount :=
by
  sorry

#check investment_growth

end NUMINAMATH_CALUDE_investment_growth_l2445_244543


namespace NUMINAMATH_CALUDE_outfit_choices_l2445_244589

/-- The number of shirts, pants, and hats available -/
def num_items : ℕ := 8

/-- The number of colors available for each type of clothing -/
def num_colors : ℕ := 8

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_items * num_items * num_items

/-- The number of outfit combinations where shirt and pants are the same color -/
def matching_combinations : ℕ := num_colors * num_items

/-- The number of valid outfit choices -/
def valid_outfits : ℕ := total_combinations - matching_combinations

theorem outfit_choices :
  valid_outfits = 448 :=
sorry

end NUMINAMATH_CALUDE_outfit_choices_l2445_244589


namespace NUMINAMATH_CALUDE_unfair_coin_expected_value_l2445_244524

def coin_flip_expected_value (p_heads : ℚ) (p_tails : ℚ) (gain_heads : ℚ) (loss_tails : ℚ) : ℚ :=
  p_heads * gain_heads + p_tails * (-loss_tails)

theorem unfair_coin_expected_value :
  let p_heads : ℚ := 3/5
  let p_tails : ℚ := 2/5
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := 6
  coin_flip_expected_value p_heads p_tails gain_heads loss_tails = 3/5 :=
by
  sorry

end NUMINAMATH_CALUDE_unfair_coin_expected_value_l2445_244524


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l2445_244567

theorem pure_imaginary_condition (a b : ℝ) : 
  (∀ x y : ℝ, x + y * Complex.I = Complex.I * y → x = 0) ∧
  (∃ x y : ℝ, x = 0 ∧ x + y * Complex.I ≠ Complex.I * y) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l2445_244567


namespace NUMINAMATH_CALUDE_expression_value_l2445_244595

theorem expression_value (a b c d x : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |x| = 3) : 
  3 * (a + b) - (-c * d) ^ 2021 + x = 4 ∨ 3 * (a + b) - (-c * d) ^ 2021 + x = -2 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l2445_244595


namespace NUMINAMATH_CALUDE_garden_area_proof_l2445_244565

theorem garden_area_proof (x : ℝ) : 
  (x + 2) * (x + 3) = 182 → x^2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_proof_l2445_244565


namespace NUMINAMATH_CALUDE_trevor_eggs_left_l2445_244562

/-- Given the number of eggs laid by each chicken and the number of eggs dropped,
    prove that the number of eggs Trevor has left is equal to the total number
    of eggs collected minus the number of eggs dropped. -/
theorem trevor_eggs_left (gertrude blanche nancy martha dropped : ℕ) :
  gertrude + blanche + nancy + martha - dropped =
  (gertrude + blanche + nancy + martha) - dropped :=
by sorry

end NUMINAMATH_CALUDE_trevor_eggs_left_l2445_244562


namespace NUMINAMATH_CALUDE_cistern_filling_time_l2445_244511

/-- Given a cistern with two taps, one that can fill it in 5 hours and another that can empty it in 6 hours,
    calculate the time it takes to fill the cistern when both taps are opened simultaneously. -/
theorem cistern_filling_time (fill_time empty_time : ℝ) (h_fill : fill_time = 5) (h_empty : empty_time = 6) :
  (fill_time * empty_time) / (empty_time - fill_time) = 30 := by
  sorry

#check cistern_filling_time

end NUMINAMATH_CALUDE_cistern_filling_time_l2445_244511


namespace NUMINAMATH_CALUDE_flower_problem_l2445_244573

theorem flower_problem (total : ℕ) (roses_fraction : ℚ) (carnations : ℕ) (tulips : ℕ) :
  total = 40 →
  roses_fraction = 2 / 5 →
  carnations = 14 →
  tulips = total - (roses_fraction * total + carnations) →
  tulips = 10 := by
sorry

end NUMINAMATH_CALUDE_flower_problem_l2445_244573


namespace NUMINAMATH_CALUDE_mary_balloons_l2445_244537

-- Define the number of Nancy's balloons
def nancy_balloons : ℕ := 7

-- Define the ratio of Mary's balloons to Nancy's
def mary_ratio : ℕ := 4

-- Theorem to prove
theorem mary_balloons : nancy_balloons * mary_ratio = 28 := by
  sorry

end NUMINAMATH_CALUDE_mary_balloons_l2445_244537


namespace NUMINAMATH_CALUDE_compare_negative_two_and_three_l2445_244577

theorem compare_negative_two_and_three : -2 > -3 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_two_and_three_l2445_244577


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_k_range_below_x_axis_k_values_for_unit_area_l2445_244508

-- Define the line equation
def line_equation (k x : ℝ) : ℝ := k * x + k - 1

-- Part 1: Prove that the line passes through (-1, -1) for all k
theorem line_passes_through_fixed_point (k : ℝ) :
  line_equation k (-1) = -1 := by sorry

-- Part 2: Prove the range of k when the line is below x-axis for -4 < x < 4
theorem k_range_below_x_axis (k : ℝ) :
  (∀ x, -4 < x ∧ x < 4 → line_equation k x < 0) ↔ -1/3 ≤ k ∧ k ≤ 1/5 := by sorry

-- Part 3: Prove the values of k when the triangle area is 1
theorem k_values_for_unit_area (k : ℝ) :
  (∃ x y, x > 0 ∧ y > 0 ∧ line_equation k x = 0 ∧ line_equation k 0 = y ∧ x * y / 2 = 1) ↔
  (k = 2 + Real.sqrt 3 ∨ k = 2 - Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_k_range_below_x_axis_k_values_for_unit_area_l2445_244508


namespace NUMINAMATH_CALUDE_inverse_not_in_M_log_in_M_iff_exp_plus_square_in_M_l2445_244571

-- Define the set M
def M : Set (ℝ → ℝ) :=
  {f | ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1}

-- Theorem 1
theorem inverse_not_in_M :
  (fun x => 1 / x) ∉ M := sorry

-- Theorem 2
theorem log_in_M_iff (a : ℝ) :
  (fun x => Real.log (a / (x^2 + 1))) ∈ M ↔ 
  3 - Real.sqrt 5 ≤ a ∧ a ≤ 3 + Real.sqrt 5 := sorry

-- Theorem 3
theorem exp_plus_square_in_M :
  (fun x => 2^x + x^2) ∈ M := sorry

end NUMINAMATH_CALUDE_inverse_not_in_M_log_in_M_iff_exp_plus_square_in_M_l2445_244571


namespace NUMINAMATH_CALUDE_equation_solutions_l2445_244528

theorem equation_solutions :
  (∀ x : ℝ, (5 - 2*x)^2 - 16 = 0 ↔ (x = 1/2 ∨ x = 9/2)) ∧
  (∀ x : ℝ, 2*(x - 3) = x^2 - 9 ↔ (x = 3 ∨ x = -1)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2445_244528


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l2445_244599

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (l m : Line) (α : Plane) :
  parallel l m → perpendicular l α → perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l2445_244599


namespace NUMINAMATH_CALUDE_rope_length_proof_l2445_244554

theorem rope_length_proof (L : ℝ) 
  (h1 : L - 42 > 0)  -- Ensures the first rope has positive remaining length
  (h2 : L - 12 > 0)  -- Ensures the second rope has positive remaining length
  (h3 : L - 12 = 4 * (L - 42)) : 2 * L = 104 := by
  sorry

end NUMINAMATH_CALUDE_rope_length_proof_l2445_244554


namespace NUMINAMATH_CALUDE_frequency_not_exceeding_15_minutes_l2445_244530

def duration_intervals : List (Real × Real) := [(0, 5), (5, 10), (10, 15), (15, 20)]
def frequencies : List Nat := [20, 16, 9, 5]

def total_calls : Nat := frequencies.sum

def calls_not_exceeding_15 : Nat := (frequencies.take 3).sum

theorem frequency_not_exceeding_15_minutes : 
  (calls_not_exceeding_15 : Real) / total_calls = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_frequency_not_exceeding_15_minutes_l2445_244530


namespace NUMINAMATH_CALUDE_closest_estimate_l2445_244515

-- Define the constants from the problem
def cars_observed : ℕ := 8
def observation_time : ℕ := 20
def delay_time : ℕ := 15
def total_time : ℕ := 210  -- 3 minutes and 30 seconds in seconds

-- Define the function to calculate the estimated number of cars
def estimate_cars : ℚ :=
  let rate : ℚ := cars_observed / observation_time
  let missed_cars : ℚ := rate * delay_time
  let observed_cars : ℚ := rate * (total_time - delay_time)
  missed_cars + observed_cars

-- Define the given options
def options : List ℕ := [120, 150, 210, 240, 280]

-- Theorem statement
theorem closest_estimate :
  ∃ (n : ℕ), n ∈ options ∧ 
  ∀ (m : ℕ), m ∈ options → |n - estimate_cars| ≤ |m - estimate_cars| ∧
  n = 120 := by
  sorry

end NUMINAMATH_CALUDE_closest_estimate_l2445_244515


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2445_244510

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 3*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 3 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2445_244510


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l2445_244570

theorem cubic_polynomial_roots (x₁ x₂ x₃ s t u : ℝ) 
  (h₁ : x₁ + x₂ + x₃ = s) 
  (h₂ : x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = t) 
  (h₃ : x₁ * x₂ * x₃ = u) : 
  (X : ℝ) → (X - x₁) * (X - x₂) * (X - x₃) = X^3 - s*X^2 + t*X - u := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l2445_244570


namespace NUMINAMATH_CALUDE_multiply_2a_3a_l2445_244556

theorem multiply_2a_3a (a : ℝ) : 2 * a * (3 * a) = 6 * a^2 := by sorry

end NUMINAMATH_CALUDE_multiply_2a_3a_l2445_244556


namespace NUMINAMATH_CALUDE_at_least_one_not_greater_than_one_l2445_244596

theorem at_least_one_not_greater_than_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / b ≤ 1) ∨ (b / c ≤ 1) ∨ (c / a ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_greater_than_one_l2445_244596


namespace NUMINAMATH_CALUDE_coloring_books_per_shelf_l2445_244518

theorem coloring_books_per_shelf
  (initial_stock : ℕ)
  (books_sold : ℕ)
  (num_shelves : ℕ)
  (h1 : initial_stock = 86)
  (h2 : books_sold = 37)
  (h3 : num_shelves = 7)
  : (initial_stock - books_sold) / num_shelves = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_coloring_books_per_shelf_l2445_244518


namespace NUMINAMATH_CALUDE_point_M_coordinates_l2445_244593

-- Define the curve
def curve (x : ℝ) : ℝ := 2 * x^2 + 1

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 4 * x

-- Theorem statement
theorem point_M_coordinates :
  ∃ (x y : ℝ), 
    curve y = curve x ∧ 
    curve_derivative x = -4 ∧ 
    x = -1 ∧ 
    y = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l2445_244593


namespace NUMINAMATH_CALUDE_det_A_eq_11_l2445_244519

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -2; 3, 1]

theorem det_A_eq_11 : A.det = 11 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_11_l2445_244519


namespace NUMINAMATH_CALUDE_female_listeners_l2445_244500

/-- Given a radio station survey with total listeners and male listeners,
    prove the number of female listeners. -/
theorem female_listeners (total_listeners male_listeners : ℕ) :
  total_listeners = 130 →
  male_listeners = 62 →
  total_listeners - male_listeners = 68 := by
  sorry

end NUMINAMATH_CALUDE_female_listeners_l2445_244500


namespace NUMINAMATH_CALUDE_johannes_earnings_today_l2445_244525

/-- Represents the earnings and sales of a vegetable shop owner over three days -/
structure VegetableShopEarnings where
  cabbage_price : ℝ
  wednesday_earnings : ℝ
  friday_earnings : ℝ
  total_cabbage_sold : ℝ

/-- Calculates the earnings for today given the total earnings and previous days' earnings -/
def earnings_today (shop : VegetableShopEarnings) : ℝ :=
  shop.cabbage_price * shop.total_cabbage_sold - (shop.wednesday_earnings + shop.friday_earnings)

/-- Theorem stating that given the specific conditions, Johannes earned $42 today -/
theorem johannes_earnings_today :
  let shop : VegetableShopEarnings := {
    cabbage_price := 2,
    wednesday_earnings := 30,
    friday_earnings := 24,
    total_cabbage_sold := 48
  }
  earnings_today shop = 42 := by sorry

end NUMINAMATH_CALUDE_johannes_earnings_today_l2445_244525


namespace NUMINAMATH_CALUDE_second_smallest_is_four_probability_l2445_244582

def set_size : ℕ := 15
def selection_size : ℕ := 8
def target_number : ℕ := 4

def favorable_outcomes : ℕ := 924
def total_outcomes : ℕ := 6435

theorem second_smallest_is_four_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 4 / 27 := by
  sorry

end NUMINAMATH_CALUDE_second_smallest_is_four_probability_l2445_244582


namespace NUMINAMATH_CALUDE_interest_calculation_l2445_244581

/-- Simple interest calculation function -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ := 1) : ℝ :=
  principal * rate * time

theorem interest_calculation 
  (r : ℝ) -- Interest rate as a decimal
  (h1 : simpleInterest 5000 r = 250) -- Condition for the initial investment
  : simpleInterest 20000 r = 1000 := by
  sorry

#check interest_calculation

end NUMINAMATH_CALUDE_interest_calculation_l2445_244581


namespace NUMINAMATH_CALUDE_bales_in_shed_l2445_244585

theorem bales_in_shed (initial_barn : ℕ) (added : ℕ) (final_barn : ℕ) : 
  initial_barn = 47 → added = 35 → final_barn = 82 → 
  final_barn = initial_barn + added → initial_barn + added = 82 → 0 = final_barn - (initial_barn + added) :=
by
  sorry

end NUMINAMATH_CALUDE_bales_in_shed_l2445_244585


namespace NUMINAMATH_CALUDE_max_value_of_m_l2445_244598

theorem max_value_of_m :
  (∀ x : ℝ, x < m → x^2 - 2*x - 8 > 0) →
  (∃ x : ℝ, x^2 - 2*x - 8 > 0 ∧ x ≥ m) →
  (∀ ε > 0, ∃ x : ℝ, x < -2 + ε ∧ x ≥ m) →
  m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_m_l2445_244598


namespace NUMINAMATH_CALUDE_f_properties_imply_m_range_l2445_244566

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the set of valid m values
def valid_m : Set ℝ := {m | m ≤ -2 ∨ m ≥ 2 ∨ m = 0}

theorem f_properties_imply_m_range :
  (∀ x, x ∈ [-1, 1] → f (-x) = -f x) →  -- f is odd
  f 1 = 1 →  -- f(1) = 1
  (∀ a b, a ∈ [-1, 1] → b ∈ [-1, 1] → a + b ≠ 0 → (f a + f b) / (a + b) > 0) →  -- given inequality
  (∀ m, (∀ x a, x ∈ [-1, 1] → a ∈ [-1, 1] → f x ≤ m^2 - 2*a*m + 1) ↔ m ∈ valid_m) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_imply_m_range_l2445_244566
