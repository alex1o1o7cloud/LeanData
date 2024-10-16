import Mathlib

namespace NUMINAMATH_CALUDE_rectangle_roots_l2243_224314

def polynomial (b : ℝ) (z : ℂ) : ℂ :=
  z^4 - 10*z^3 + 16*b*z^2 - 2*(3*b^2 - 5*b + 4)*z + 6

def forms_rectangle (b : ℝ) : Prop :=
  ∃ (z₁ z₂ z₃ z₄ : ℂ),
    polynomial b z₁ = 0 ∧
    polynomial b z₂ = 0 ∧
    polynomial b z₃ = 0 ∧
    polynomial b z₄ = 0 ∧
    (z₁.re = z₂.re ∧ z₁.im = -z₂.im) ∧
    (z₃.re = z₄.re ∧ z₃.im = -z₄.im) ∧
    (z₁.re - z₃.re = z₂.im - z₄.im) ∧
    (z₁.im - z₃.im = z₄.re - z₂.re)

theorem rectangle_roots :
  ∀ b : ℝ, forms_rectangle b ↔ (b = 5/3 ∨ b = 2) :=
sorry

end NUMINAMATH_CALUDE_rectangle_roots_l2243_224314


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_l2243_224305

theorem quadratic_real_solutions : ∃ (x : ℝ), x^2 + 3*x - 2 = 0 ∧
  (∀ (x : ℝ), 2*x^2 - x + 1 ≠ 0) ∧
  (∀ (x : ℝ), x^2 - 2*x + 2 ≠ 0) ∧
  (∀ (x : ℝ), x^2 + 2 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_l2243_224305


namespace NUMINAMATH_CALUDE_distance_travelled_l2243_224323

theorem distance_travelled (initial_speed : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) :
  initial_speed = 12 →
  faster_speed = 18 →
  additional_distance = 30 →
  ∃ (actual_distance : ℝ),
    actual_distance / initial_speed = (actual_distance + additional_distance) / faster_speed ∧
    actual_distance = 60 := by
  sorry

end NUMINAMATH_CALUDE_distance_travelled_l2243_224323


namespace NUMINAMATH_CALUDE_gcd_of_sums_of_squares_l2243_224381

theorem gcd_of_sums_of_squares : 
  Nat.gcd (118^2 + 227^2 + 341^2) (119^2 + 226^2 + 340^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_sums_of_squares_l2243_224381


namespace NUMINAMATH_CALUDE_g_range_l2243_224332

noncomputable def g (x : ℝ) : ℝ := 
  (Real.cos x ^ 3 + 5 * Real.cos x ^ 2 + 2 * Real.cos x + 3 * Real.sin x ^ 2 - 9) / (Real.cos x - 1)

theorem g_range (x : ℝ) (h : Real.cos x ≠ 1) : 
  6 ≤ g x ∧ g x < 12 := by sorry

end NUMINAMATH_CALUDE_g_range_l2243_224332


namespace NUMINAMATH_CALUDE_simplify_expression_l2243_224372

theorem simplify_expression (x : ℝ) : (2*x)^3 + (3*x)*(x^2) = 11*x^3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2243_224372


namespace NUMINAMATH_CALUDE_sock_pair_count_l2243_224350

/-- The number of ways to choose a pair of socks of different colors -/
def different_color_pairs (white brown blue : ℕ) : ℕ :=
  white * brown + white * blue + brown * blue

/-- Theorem: The number of ways to choose a pair of socks of different colors
    from a drawer containing 5 white, 4 brown, and 3 blue distinguishable socks
    is equal to 47. -/
theorem sock_pair_count :
  different_color_pairs 5 4 3 = 47 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_count_l2243_224350


namespace NUMINAMATH_CALUDE_ivan_dice_count_l2243_224328

theorem ivan_dice_count (x : ℕ) : 
  x + 2*x = 60 → x = 20 := by
  sorry

end NUMINAMATH_CALUDE_ivan_dice_count_l2243_224328


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_specific_root_implies_m_and_other_root_l2243_224329

/-- Given a quadratic equation x^2 + 2x - (m-2) = 0 with real roots -/
def quadratic_equation (x m : ℝ) : Prop := x^2 + 2*x - (m-2) = 0

/-- The discriminant of the quadratic equation is non-negative -/
def has_real_roots (m : ℝ) : Prop := 4*m - 4 ≥ 0

theorem quadratic_real_roots_condition (m : ℝ) :
  has_real_roots m ↔ m ≥ 1 := by sorry

theorem specific_root_implies_m_and_other_root :
  ∀ m : ℝ, quadratic_equation 1 m → m = 3 ∧ quadratic_equation (-3) m := by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_specific_root_implies_m_and_other_root_l2243_224329


namespace NUMINAMATH_CALUDE_find_divisor_l2243_224343

theorem find_divisor : ∃ (d : ℕ), d > 1 ∧ 
  (1054 + 4 = 1058) ∧ 
  (1058 % d = 0) ∧
  (∀ k : ℕ, k < 4 → (1054 + k) % d ≠ 0) →
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2243_224343


namespace NUMINAMATH_CALUDE_marked_price_calculation_l2243_224326

theorem marked_price_calculation (total_cost : ℚ) (discount_rate : ℚ) : 
  total_cost = 50 → discount_rate = 1/10 → 
  ∃ (marked_price : ℚ), marked_price = 250/9 ∧ 
  2 * (marked_price * (1 - discount_rate)) = total_cost :=
by sorry

end NUMINAMATH_CALUDE_marked_price_calculation_l2243_224326


namespace NUMINAMATH_CALUDE_power_nine_mod_hundred_l2243_224308

theorem power_nine_mod_hundred : 9^2050 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_nine_mod_hundred_l2243_224308


namespace NUMINAMATH_CALUDE_modulus_of_2_plus_i_times_i_l2243_224391

theorem modulus_of_2_plus_i_times_i : Complex.abs ((2 + Complex.I) * Complex.I) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_2_plus_i_times_i_l2243_224391


namespace NUMINAMATH_CALUDE_parallel_line_slope_l2243_224335

/-- The slope of any line parallel to 3x + 6y = -21 is -1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : 3 * a + 6 * b = -21) :
  ∃ (m : ℝ), m = -1/2 ∧ ∀ (x y : ℝ), 3 * x + 6 * y = -21 → y = m * x + c :=
sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l2243_224335


namespace NUMINAMATH_CALUDE_sphere_plane_intersection_area_l2243_224307

theorem sphere_plane_intersection_area (r : ℝ) (h : r = 1) :
  ∃ (d : ℝ), 0 < d ∧ d < r ∧
  (2 * π * r * (r - d) = π * r^2) ∧
  (2 * π * r * d = 3 * π * r^2) ∧
  π * (r^2 - d^2) = (3 * π) / 4 := by
sorry

end NUMINAMATH_CALUDE_sphere_plane_intersection_area_l2243_224307


namespace NUMINAMATH_CALUDE_largest_divisible_n_l2243_224397

theorem largest_divisible_n : 
  ∀ n : ℕ, n > 5376 → ¬(((n : ℤ)^3 + 200) % (n - 8) = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisible_n_l2243_224397


namespace NUMINAMATH_CALUDE_impossible_to_break_record_duke_game_impossible_l2243_224358

/-- Represents the constraints and conditions for Duke's basketball game --/
structure GameConstraints where
  old_record : ℕ
  points_to_tie : ℕ
  points_to_break : ℕ
  free_throws : ℕ
  regular_baskets : ℕ
  normal_three_pointers : ℕ
  max_attempts : ℕ

/-- Calculates the total points scored based on the number of each type of shot --/
def total_points (ft reg tp : ℕ) : ℕ :=
  ft + 2 * reg + 3 * tp

/-- Theorem stating that it's impossible to break the record under the given constraints --/
theorem impossible_to_break_record (gc : GameConstraints) : 
  ¬∃ (tp : ℕ), 
    total_points gc.free_throws gc.regular_baskets tp = gc.old_record + gc.points_to_tie + gc.points_to_break ∧
    gc.free_throws + gc.regular_baskets + tp ≤ gc.max_attempts :=
by
  sorry

/-- The specific game constraints for Duke's final game --/
def duke_game : GameConstraints :=
  { old_record := 257
  , points_to_tie := 17
  , points_to_break := 5
  , free_throws := 5
  , regular_baskets := 4
  , normal_three_pointers := 2
  , max_attempts := 10
  }

/-- Theorem applying the impossibility proof to Duke's specific game --/
theorem duke_game_impossible : 
  ¬∃ (tp : ℕ), 
    total_points duke_game.free_throws duke_game.regular_baskets tp = 
      duke_game.old_record + duke_game.points_to_tie + duke_game.points_to_break ∧
    duke_game.free_throws + duke_game.regular_baskets + tp ≤ duke_game.max_attempts :=
by
  apply impossible_to_break_record duke_game

end NUMINAMATH_CALUDE_impossible_to_break_record_duke_game_impossible_l2243_224358


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_squared_l2243_224394

theorem sum_and_reciprocal_squared (x : ℝ) (h : x + 1/x = 6) : x^2 + 1/x^2 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_squared_l2243_224394


namespace NUMINAMATH_CALUDE_quadratic_value_at_zero_l2243_224366

-- Define the quadratic function
def f (h : ℝ) (x : ℝ) : ℝ := -(x + h)^2

-- Define the axis of symmetry
def axis_of_symmetry (h : ℝ) : ℝ := -3

-- Theorem statement
theorem quadratic_value_at_zero (h : ℝ) : 
  axis_of_symmetry h = -3 → f h 0 = -9 := by
  sorry

#check quadratic_value_at_zero

end NUMINAMATH_CALUDE_quadratic_value_at_zero_l2243_224366


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2243_224341

/-- Sum of first n terms of an arithmetic sequence -/
def T (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The problem statement -/
theorem arithmetic_sequence_first_term
  (h : ∃ (k : ℚ), ∀ (n : ℕ), n > 0 → T a₁ 5 (2*n) / T a₁ 5 n = k) :
  a₁ = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2243_224341


namespace NUMINAMATH_CALUDE_division_problem_l2243_224322

theorem division_problem : (5 + 1/2) / (2/11) = 121/4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2243_224322


namespace NUMINAMATH_CALUDE_sally_bought_three_frames_l2243_224310

/-- The number of photograph frames Sally bought -/
def frames_bought (frame_cost change_received total_paid : ℕ) : ℕ :=
  (total_paid - change_received) / frame_cost

/-- Theorem stating that Sally bought 3 photograph frames -/
theorem sally_bought_three_frames :
  frames_bought 3 11 20 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sally_bought_three_frames_l2243_224310


namespace NUMINAMATH_CALUDE_log_sum_evaluation_l2243_224364

theorem log_sum_evaluation : 
  Real.log 16 / Real.log 2 + 3 * (Real.log 8 / Real.log 2) + 2 * (Real.log 4 / Real.log 2) - Real.log 64 / Real.log 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_evaluation_l2243_224364


namespace NUMINAMATH_CALUDE_square_points_sum_l2243_224325

/-- Square with side length 1000 -/
structure Square :=
  (side : ℝ)
  (is_1000 : side = 1000)

/-- Points on the side of the square -/
structure PointOnSide (S : Square) :=
  (pos : ℝ)
  (on_side : 0 ≤ pos ∧ pos ≤ S.side)

/-- Condition that E is between A and F -/
def between (A E F : ℝ) : Prop := A ≤ E ∧ E ≤ F

/-- Angle in degrees -/
def angle (θ : ℝ) := 0 ≤ θ ∧ θ < 360

/-- Distance between two points on a line -/
def distance (x y : ℝ) := |x - y|

/-- Representation of BF as p + q√r -/
structure IrrationalForm :=
  (p q r : ℕ)
  (r_not_square : ∀ (n : ℕ), n > 1 → r % (n^2) ≠ 0)

theorem square_points_sum (S : Square) 
  (E F : PointOnSide S)
  (AE_less_BF : E.pos < S.side - F.pos)
  (E_between_A_F : between 0 E.pos F.pos)
  (angle_EOF : angle 30)
  (EF_length : distance E.pos F.pos = 500)
  (BF_form : IrrationalForm)
  (BF_value : S.side - F.pos = BF_form.p + BF_form.q * Real.sqrt BF_form.r) :
  BF_form.p + BF_form.q + BF_form.r = 253 := by
  sorry

end NUMINAMATH_CALUDE_square_points_sum_l2243_224325


namespace NUMINAMATH_CALUDE_postcard_height_l2243_224387

theorem postcard_height (perimeter width : ℝ) (h_perimeter : perimeter = 20) (h_width : width = 6) :
  let height := (perimeter - 2 * width) / 2
  height = 4 := by sorry

end NUMINAMATH_CALUDE_postcard_height_l2243_224387


namespace NUMINAMATH_CALUDE_teresas_pencil_sharing_l2243_224321

/-- Proves that each sibling receives 13 pencils given the conditions of Teresa's pencil sharing problem -/
theorem teresas_pencil_sharing :
  -- Define the given conditions
  let total_pencils : ℕ := 14 + 35
  let pencils_to_keep : ℕ := 10
  let num_siblings : ℕ := 3
  let pencils_to_share : ℕ := total_pencils - pencils_to_keep
  -- Define the theorem
  pencils_to_share / num_siblings = 13 := by
  sorry

end NUMINAMATH_CALUDE_teresas_pencil_sharing_l2243_224321


namespace NUMINAMATH_CALUDE_investment_ratio_is_one_to_one_l2243_224378

-- Define the interest rates
def interest_rate_1 : ℝ := 0.05
def interest_rate_2 : ℝ := 0.06

-- Define the total interest earned
def total_interest : ℝ := 520

-- Define the investment amounts
def investment_1 : ℝ := 2000
def investment_2 : ℝ := 2000

-- Theorem statement
theorem investment_ratio_is_one_to_one :
  (investment_1 * interest_rate_1 + investment_2 * interest_rate_2 = total_interest) →
  (investment_1 / investment_2 = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_investment_ratio_is_one_to_one_l2243_224378


namespace NUMINAMATH_CALUDE_factorization_equality_l2243_224300

theorem factorization_equality (x : ℝ) : 
  75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2243_224300


namespace NUMINAMATH_CALUDE_cube_volume_problem_l2243_224337

theorem cube_volume_problem (V₁ : ℝ) (V₂ : ℝ) (A₁ : ℝ) (A₂ : ℝ) (s₁ : ℝ) (s₂ : ℝ) :
  V₁ = 8 →
  s₁ ^ 3 = V₁ →
  A₁ = 6 * s₁ ^ 2 →
  A₂ = 3 * A₁ →
  A₂ = 6 * s₂ ^ 2 →
  V₂ = s₂ ^ 3 →
  V₂ = 24 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l2243_224337


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_l2243_224318

def expression (x : ℝ) : ℝ :=
  4 * (x^2 - 2*x^3 + 2*x) + 2 * (x + 3*x^3 - 2*x^2 + 4*x^5 - x^3) - 6 * (2 + x - 5*x^3 - 2*x^2)

theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expression))) 0 / 6 = 26 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_l2243_224318


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_8154_l2243_224306

/-- Calculates the cost of white washing a room with given dimensions and openings -/
def whitewashingCost (length width height : ℝ) (doorLength doorWidth : ℝ)
  (windowLength windowWidth : ℝ) (windowCount : ℕ) (costPerSquareFoot : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let doorArea := doorLength * doorWidth
  let windowArea := windowLength * windowWidth * windowCount
  let netArea := wallArea - doorArea - windowArea
  netArea * costPerSquareFoot

/-- Theorem stating the cost of white washing the room with given specifications -/
theorem whitewashing_cost_is_8154 :
  whitewashingCost 25 15 12 6 3 4 3 3 9 = 8154 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_is_8154_l2243_224306


namespace NUMINAMATH_CALUDE_cistern_length_l2243_224395

theorem cistern_length (width : ℝ) (depth : ℝ) (wet_area : ℝ) (length : ℝ) : 
  width = 4 →
  depth = 1.25 →
  wet_area = 49 →
  wet_area = (length * width) + (2 * length * depth) + (2 * width * depth) →
  length = 6 := by
sorry

end NUMINAMATH_CALUDE_cistern_length_l2243_224395


namespace NUMINAMATH_CALUDE_power_simplification_l2243_224382

theorem power_simplification : 2^6 * 8^3 * 2^12 * 8^6 = 2^45 := by sorry

end NUMINAMATH_CALUDE_power_simplification_l2243_224382


namespace NUMINAMATH_CALUDE_coin_piles_theorem_l2243_224351

/-- Represents the number of coins in each pile -/
structure CoinPiles :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Performs the coin transfers as described in the problem -/
def transfer (piles : CoinPiles) : CoinPiles :=
  let step1 := CoinPiles.mk (piles.first - piles.second) (piles.second + piles.second) piles.third
  let step2 := CoinPiles.mk step1.first (step1.second - step1.third) (step1.third + step1.third)
  CoinPiles.mk (step2.first + step2.third) step2.second (step2.third - step2.first)

/-- The main theorem stating the original number of coins in each pile -/
theorem coin_piles_theorem (piles : CoinPiles) :
  transfer piles = CoinPiles.mk 16 16 16 →
  piles = CoinPiles.mk 22 14 12 :=
by sorry

end NUMINAMATH_CALUDE_coin_piles_theorem_l2243_224351


namespace NUMINAMATH_CALUDE_travel_group_combinations_l2243_224345

def total_friends : ℕ := 12
def friends_to_choose : ℕ := 5
def previously_traveled_friends : ℕ := 6

theorem travel_group_combinations : 
  (total_friends.choose friends_to_choose) - 
  ((total_friends - previously_traveled_friends).choose friends_to_choose) = 786 := by
  sorry

end NUMINAMATH_CALUDE_travel_group_combinations_l2243_224345


namespace NUMINAMATH_CALUDE_chess_team_arrangement_count_l2243_224373

def chess_team_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  if num_boys + num_girls ≠ 7 then 0
  else if num_boys ≠ 3 then 0
  else if num_girls ≠ 4 then 0
  else Nat.factorial num_boys * Nat.factorial num_girls

theorem chess_team_arrangement_count :
  chess_team_arrangements 3 4 = 144 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_arrangement_count_l2243_224373


namespace NUMINAMATH_CALUDE_rainy_days_probability_exists_l2243_224327

theorem rainy_days_probability_exists :
  ∃ (n : ℕ), n > 0 ∧ 
    (Nat.choose n 3 : ℝ) * (1/2)^3 * (1/2)^(n-3) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_rainy_days_probability_exists_l2243_224327


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2243_224380

theorem inequality_solution_set (x : ℝ) : 
  (5 / (x + 2) ≥ 1 ∧ x + 2 ≠ 0) ↔ -2 < x ∧ x ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2243_224380


namespace NUMINAMATH_CALUDE_not_prime_4k4_plus_1_and_k4_plus_4_l2243_224338

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_4k4_plus_1_and_k4_plus_4 (k : ℕ) : 
  ¬(is_prime (4 * k^4 + 1)) ∧ ¬(is_prime (k^4 + 4)) := by
  sorry


end NUMINAMATH_CALUDE_not_prime_4k4_plus_1_and_k4_plus_4_l2243_224338


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2243_224384

theorem sum_of_four_numbers : 2367 + 3672 + 6723 + 7236 = 19998 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2243_224384


namespace NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2243_224376

theorem sum_of_squares_lower_bound (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_lower_bound_l2243_224376


namespace NUMINAMATH_CALUDE_min_value_m_plus_2n_l2243_224361

theorem min_value_m_plus_2n (m n : ℝ) (h : m - n^2 = 0) : 
  ∀ x y : ℝ, x - y^2 = 0 → m + 2*n ≤ x + 2*y :=
by sorry

end NUMINAMATH_CALUDE_min_value_m_plus_2n_l2243_224361


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2243_224315

theorem fraction_sum_equality : 
  (1 : ℚ) / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 5 / 6 = -5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2243_224315


namespace NUMINAMATH_CALUDE_line_y_axis_intersection_l2243_224324

/-- A line passing through two points intersects the y-axis at a specific point -/
theorem line_y_axis_intersection (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 3 →
  y₁ = 20 →
  x₂ = -7 →
  y₂ = 2 →
  ∃ y : ℝ, y = 14.6 ∧ (y - y₁) / (0 - x₁) = (y₂ - y₁) / (x₂ - x₁) :=
by sorry

end NUMINAMATH_CALUDE_line_y_axis_intersection_l2243_224324


namespace NUMINAMATH_CALUDE_inscribed_circle_radii_theorem_l2243_224365

/-- A regular pyramid with base ABCD and apex S -/
structure RegularPyramid where
  /-- The length of the base diagonal AC -/
  base_diagonal : ℝ
  /-- The cosine of the angle SBD -/
  cos_angle_sbd : ℝ
  /-- Assumption that the pyramid is regular -/
  regular : True
  /-- Assumption that the base diagonal AC = 1 -/
  base_diagonal_eq_one : base_diagonal = 1
  /-- Assumption that cos(∠SBD) = 2/3 -/
  cos_angle_sbd_eq_two_thirds : cos_angle_sbd = 2/3

/-- The set of possible radii for circles inscribed in planar sections of the pyramid -/
def inscribed_circle_radii (p : RegularPyramid) : Set ℝ :=
  {r : ℝ | (0 < r ∧ r ≤ 1/6) ∨ r = 1/3}

/-- Theorem stating the possible radii of inscribed circles in the regular pyramid -/
theorem inscribed_circle_radii_theorem (p : RegularPyramid) :
  ∀ r : ℝ, r ∈ inscribed_circle_radii p ↔ (0 < r ∧ r ≤ 1/6) ∨ r = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radii_theorem_l2243_224365


namespace NUMINAMATH_CALUDE_fraction_simplification_l2243_224320

theorem fraction_simplification : (4 * 5) / 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2243_224320


namespace NUMINAMATH_CALUDE_hair_cut_first_day_l2243_224398

/-- Given that Elizabeth had her hair cut on two consecutive days, with a total of 0.88 inches
    cut off and 0.5 inches cut off on the second day, this theorem proves that 0.38 inches
    were cut off on the first day. -/
theorem hair_cut_first_day (total : ℝ) (second_day : ℝ) (h1 : total = 0.88) (h2 : second_day = 0.5) :
  total - second_day = 0.38 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_first_day_l2243_224398


namespace NUMINAMATH_CALUDE_trapezoid_has_two_heights_l2243_224330

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides. -/
structure Trapezoid where
  vertices : Fin 4 → ℝ × ℝ
  parallel_sides : ∃ (i j : Fin 4), i ≠ j ∧ (vertices i).1 = (vertices j).1

/-- The number of heights in a trapezoid -/
def num_heights (t : Trapezoid) : ℕ := 2

/-- Theorem: A trapezoid has exactly 2 heights -/
theorem trapezoid_has_two_heights (t : Trapezoid) : num_heights t = 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_has_two_heights_l2243_224330


namespace NUMINAMATH_CALUDE_ellipse_condition_l2243_224316

/-- The equation of the graph --/
def equation (x y k : ℝ) : Prop :=
  x^2 + 4*y^2 - 10*x + 56*y = k

/-- Definition of a non-degenerate ellipse --/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  ∃ a b c d e : ℝ, a > 0 ∧ b > 0 ∧ 
    ∀ x y : ℝ, equation x y k ↔ (x - c)^2 / a + (y - d)^2 / b = e

/-- The main theorem --/
theorem ellipse_condition (k : ℝ) :
  is_non_degenerate_ellipse k ↔ k > -221 :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2243_224316


namespace NUMINAMATH_CALUDE_total_spent_is_89_10_l2243_224362

/-- The total amount spent by Edward and his friend after the discount -/
def total_spent_after_discount (
  trick_deck_price : ℝ) 
  (edward_decks : ℕ) 
  (edward_hat_price : ℝ)
  (friend_decks : ℕ)
  (friend_wand_price : ℝ)
  (discount_rate : ℝ) : ℝ :=
  let total_before_discount := 
    trick_deck_price * (edward_decks + friend_decks) + edward_hat_price + friend_wand_price
  total_before_discount * (1 - discount_rate)

/-- Theorem stating that the total amount spent after the discount is $89.10 -/
theorem total_spent_is_89_10 :
  total_spent_after_discount 9 4 12 4 15 0.1 = 89.10 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_89_10_l2243_224362


namespace NUMINAMATH_CALUDE_x1_range_l2243_224399

/-- The function f as defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (x + 1) - Real.exp x + x^2 + 2 * m * (x - 1)

/-- The theorem stating the range of x1 -/
theorem x1_range (m : ℝ) (hm : m > 0) :
  {x1 : ℝ | ∀ x2, x1 + x2 = 1 → f m x1 ≥ f m x2} = Set.Ici (1/2) :=
sorry

end NUMINAMATH_CALUDE_x1_range_l2243_224399


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2243_224393

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2243_224393


namespace NUMINAMATH_CALUDE_system_solution_l2243_224313

theorem system_solution : 
  ∀ (x y : ℝ), x > 0 ∧ y > 0 → 
  (x - 3 * Real.sqrt (x * y) - 2 * Real.sqrt (x / y) + 6 = 0 ∧ 
   x^2 * y^2 + x^4 = 82) → 
  ((x = 3 ∧ y = 1/3) ∨ (x = Real.rpow 33 (1/4) ∧ y = 4 / Real.rpow 33 (1/4))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2243_224313


namespace NUMINAMATH_CALUDE_meeting_speed_l2243_224301

/-- Given two people 55 miles apart, where one walks at 6 mph and the other walks 25 miles before they meet, prove that the speed of the second person is 5 mph. -/
theorem meeting_speed (total_distance : ℝ) (fred_speed : ℝ) (sam_distance : ℝ) :
  total_distance = 55 →
  fred_speed = 6 →
  sam_distance = 25 →
  (total_distance - sam_distance) / fred_speed = sam_distance / ((total_distance - sam_distance) / fred_speed) :=
by sorry

end NUMINAMATH_CALUDE_meeting_speed_l2243_224301


namespace NUMINAMATH_CALUDE_negation_false_l2243_224302

theorem negation_false : ¬∃ (x y : ℝ), x > 2 ∧ y > 3 ∧ x + y ≤ 5 := by sorry

end NUMINAMATH_CALUDE_negation_false_l2243_224302


namespace NUMINAMATH_CALUDE_solve_inequality_system_simplify_expression_l2243_224370

-- Part 1: System of inequalities
theorem solve_inequality_system (x : ℝ) :
  (x + 2) / 5 < 1 ∧ 3 * x - 1 ≥ 2 * x ↔ 1 ≤ x ∧ x < 3 := by sorry

-- Part 2: Algebraic expression
theorem simplify_expression (m : ℝ) (hm : m ≠ 0) :
  (m - 1 / m) * (m^2 - m) / (m^2 - 2*m + 1) = m + 1 := by sorry

end NUMINAMATH_CALUDE_solve_inequality_system_simplify_expression_l2243_224370


namespace NUMINAMATH_CALUDE_jerry_reading_pages_l2243_224309

theorem jerry_reading_pages : ∀ (total_pages pages_read_saturday pages_remaining : ℕ),
  total_pages = 93 →
  pages_read_saturday = 30 →
  pages_remaining = 43 →
  total_pages - pages_read_saturday - pages_remaining = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_jerry_reading_pages_l2243_224309


namespace NUMINAMATH_CALUDE_no_real_m_for_reciprocal_sum_l2243_224374

theorem no_real_m_for_reciprocal_sum (m : ℝ) : ¬ (∃ x₁ x₂ : ℝ,
  (m * x₁^2 - 2*x₁ + m*(m^2 + 1) = 0) ∧
  (m * x₂^2 - 2*x₂ + m*(m^2 + 1) = 0) ∧
  (x₁ ≠ x₂) ∧
  (1/x₁ + 1/x₂ = m)) := by
  sorry

#check no_real_m_for_reciprocal_sum

end NUMINAMATH_CALUDE_no_real_m_for_reciprocal_sum_l2243_224374


namespace NUMINAMATH_CALUDE_paint_can_display_space_l2243_224344

/-- Calculates the total number of cans in a triangular arrangement -/
def totalCans (n : ℕ) : ℕ := n * (n + 1) * 3 / 2

/-- Calculates the total space required for the cans -/
def totalSpace (n : ℕ) (spacePerCan : ℕ) : ℕ := 
  (n * (n + 1) * 3 / 2) * spacePerCan

theorem paint_can_display_space : 
  ∃ n : ℕ, totalCans n = 242 ∧ totalSpace n 50 = 3900 := by
  sorry

end NUMINAMATH_CALUDE_paint_can_display_space_l2243_224344


namespace NUMINAMATH_CALUDE_largest_less_than_point_seven_l2243_224386

theorem largest_less_than_point_seven : 
  let S : Set ℚ := {8/10, 1/2, 9/10, 1/3}
  ∀ x ∈ S, x < 7/10 → x ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_largest_less_than_point_seven_l2243_224386


namespace NUMINAMATH_CALUDE_circle_equation_through_points_l2243_224336

theorem circle_equation_through_points :
  let equation (x y : ℝ) := x^2 + y^2 - 4*x - 6*y
  ∀ (x y : ℝ), equation x y = 0 →
    (x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_through_points_l2243_224336


namespace NUMINAMATH_CALUDE_teacher_assignment_theorem_l2243_224334

def number_of_teachers : ℕ := 4
def number_of_classes : ℕ := 3

-- Define a function that calculates the number of ways to assign teachers to classes
def ways_to_assign_teachers (teachers : ℕ) (classes : ℕ) : ℕ :=
  sorry -- The actual calculation goes here

theorem teacher_assignment_theorem :
  ways_to_assign_teachers number_of_teachers number_of_classes = 36 :=
by sorry

end NUMINAMATH_CALUDE_teacher_assignment_theorem_l2243_224334


namespace NUMINAMATH_CALUDE_geometric_progression_constant_l2243_224360

theorem geometric_progression_constant (x : ℝ) : 
  (70 + x)^2 = (30 + x) * (150 + x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_constant_l2243_224360


namespace NUMINAMATH_CALUDE_exponential_characterization_l2243_224331

/-- A continuous function satisfying f(x+y) = f(x)f(y) is of the form aˣ for some a > 0 -/
theorem exponential_characterization (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_nonzero : ∃ x₀, f x₀ ≠ 0) 
  (h_mult : ∀ x y, f (x + y) = f x * f y) : 
  ∃ a > 0, ∀ x, f x = Real.exp (x * Real.log a) := by
sorry

end NUMINAMATH_CALUDE_exponential_characterization_l2243_224331


namespace NUMINAMATH_CALUDE_puppy_weight_l2243_224304

/-- Given the weights of a puppy, a smaller cat, and a larger cat, prove that the puppy weighs 5 pounds. -/
theorem puppy_weight (p s l : ℝ) 
  (total_weight : p + s + l = 30)
  (puppy_larger_cat : p + l = 3 * s)
  (puppy_smaller_cat : p + s = l - 5) :
  p = 5 := by
  sorry

end NUMINAMATH_CALUDE_puppy_weight_l2243_224304


namespace NUMINAMATH_CALUDE_socorro_training_time_l2243_224396

/-- Calculates the total training time in hours given daily training times and number of days -/
def total_training_time (mult_time : ℕ) (div_time : ℕ) (days : ℕ) (mins_per_hour : ℕ) : ℚ :=
  (mult_time + div_time) * days / mins_per_hour

/-- Proves that Socorro's total training time is 5 hours -/
theorem socorro_training_time :
  total_training_time 10 20 10 60 = 5 := by
  sorry

end NUMINAMATH_CALUDE_socorro_training_time_l2243_224396


namespace NUMINAMATH_CALUDE_equation_equivalence_l2243_224392

theorem equation_equivalence (x : ℝ) : 
  (4 * x^2 + 1 = (2*x + 1)^2) ∨ (4 * x^2 + 1 = (2*x - 1)^2) ↔ (4*x = 0 ∨ -4*x = 0) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2243_224392


namespace NUMINAMATH_CALUDE_wild_sorghum_and_corn_different_species_reproductive_isolation_wild_sorghum_corn_l2243_224377

/-- Represents a plant species -/
structure Species where
  name : String
  chromosomes : Nat

/-- Defines reproductive isolation between two species -/
def reproductiveIsolation (s1 s2 : Species) : Prop :=
  s1.chromosomes ≠ s2.chromosomes

/-- Defines whether two species are the same -/
def sameSpecies (s1 s2 : Species) : Prop :=
  s1.chromosomes = s2.chromosomes ∧ ¬reproductiveIsolation s1 s2

/-- Wild sorghum species -/
def wildSorghum : Species :=
  { name := "Wild Sorghum", chromosomes := 22 }

/-- Corn species -/
def corn : Species :=
  { name := "Corn", chromosomes := 20 }

/-- Theorem stating that wild sorghum and corn are not the same species -/
theorem wild_sorghum_and_corn_different_species :
  ¬sameSpecies wildSorghum corn :=
by
  sorry

/-- Theorem stating that there is reproductive isolation between wild sorghum and corn -/
theorem reproductive_isolation_wild_sorghum_corn :
  reproductiveIsolation wildSorghum corn :=
by
  sorry

end NUMINAMATH_CALUDE_wild_sorghum_and_corn_different_species_reproductive_isolation_wild_sorghum_corn_l2243_224377


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2243_224375

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_first : a 1 = 3) 
  (h_third : a 3 = 5) : 
  a 7 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l2243_224375


namespace NUMINAMATH_CALUDE_quadratic_root_implies_a_leq_4_l2243_224333

/-- The quadratic function f(x) = x^2 + 4x + a has a real root implies a ≤ 4 -/
theorem quadratic_root_implies_a_leq_4 (a : ℝ) :
  (∃ x : ℝ, x^2 + 4*x + a = 0) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_a_leq_4_l2243_224333


namespace NUMINAMATH_CALUDE_mothers_current_age_l2243_224339

-- Define Samson's current age
def samson_age : ℕ := 6

-- Define the time difference
def years_ago : ℕ := 4

-- Define the relationship between Samson's and his mother's age 4 years ago
def mother_age_ratio : ℕ := 4

-- Theorem to prove
theorem mothers_current_age :
  -- Given conditions
  (samson_age : ℕ) →
  (years_ago : ℕ) →
  (mother_age_ratio : ℕ) →
  -- Conclusion
  (samson_age - years_ago) * mother_age_ratio + years_ago = 16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_mothers_current_age_l2243_224339


namespace NUMINAMATH_CALUDE_cubic_equation_natural_roots_l2243_224342

theorem cubic_equation_natural_roots :
  ∃! P : ℝ, ∀ x : ℕ,
    (5 * x^3 - 5 * (P + 1) * x^2 + (71 * P - 1) * x + 1 = 66 * P) →
    (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
      (5 * a^3 - 5 * (P + 1) * a^2 + (71 * P - 1) * a + 1 = 66 * P) ∧
      (5 * b^3 - 5 * (P + 1) * b^2 + (71 * P - 1) * b + 1 = 66 * P) ∧
      (5 * c^3 - 5 * (P + 1) * c^2 + (71 * P - 1) * c + 1 = 66 * P)) →
    P = 76 :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_natural_roots_l2243_224342


namespace NUMINAMATH_CALUDE_x_range_theorem_l2243_224383

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 3 * x + sin x

-- State the theorem
theorem x_range_theorem (h1 : ∀ x ∈ Set.Ioo (-1) 1, deriv f x = 3 + cos x)
                        (h2 : f 0 = 0)
                        (h3 : ∀ x, f (1 - x) + f (1 - x^2) < 0) :
  ∃ x ∈ Set.Ioo 1 (Real.sqrt 2), True :=
sorry

end NUMINAMATH_CALUDE_x_range_theorem_l2243_224383


namespace NUMINAMATH_CALUDE_log_sum_fifty_twenty_l2243_224303

theorem log_sum_fifty_twenty : Real.log 50 / Real.log 10 + Real.log 20 / Real.log 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_fifty_twenty_l2243_224303


namespace NUMINAMATH_CALUDE_afternoon_campers_l2243_224317

theorem afternoon_campers (evening_campers : ℕ) (afternoon_evening_difference : ℕ) 
  (h1 : evening_campers = 10)
  (h2 : afternoon_evening_difference = 24) :
  evening_campers + afternoon_evening_difference = 34 := by
  sorry

end NUMINAMATH_CALUDE_afternoon_campers_l2243_224317


namespace NUMINAMATH_CALUDE_range_of_a_l2243_224390

-- Define the conditions
def condition_p (x a : ℝ) : Prop := -4 < x - a ∧ x - a < 4

def condition_q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Define the theorem
theorem range_of_a :
  (∀ x a : ℝ, condition_q x → condition_p x a) →
  ∀ a : ℝ, -1 ≤ a ∧ a ≤ 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2243_224390


namespace NUMINAMATH_CALUDE_johnny_work_hours_l2243_224353

/-- Given Johnny's hourly wage and total earnings, prove the number of hours he worked. -/
theorem johnny_work_hours (hourly_wage : ℚ) (total_earnings : ℚ) (h1 : hourly_wage = 13/4) (h2 : total_earnings = 26) :
  total_earnings / hourly_wage = 8 := by
  sorry

end NUMINAMATH_CALUDE_johnny_work_hours_l2243_224353


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l2243_224354

/-- Given a quadratic equation mx^2 - (m+2)x + m/4 = 0 with two distinct real roots,
    if the sum of the reciprocals of the roots is 4m, then m = 2 -/
theorem quadratic_roots_reciprocal_sum (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, m * x^2 - (m + 2) * x + m / 4 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ ≠ x₂ →
  1 / x₁ + 1 / x₂ = 4 * m →
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l2243_224354


namespace NUMINAMATH_CALUDE_total_points_theorem_l2243_224367

/-- The total number of participating teams -/
def num_teams : ℕ := 70

/-- The total number of points earned on question 33 -/
def points_q33 : ℕ := 3

/-- The total number of points earned on question 34 -/
def points_q34 : ℕ := 6

/-- The total number of points earned on question 35 -/
def points_q35 : ℕ := 4

/-- The total number of points A earned over all participating teams on questions 33, 34, and 35 -/
def A : ℕ := points_q33 + points_q34 + points_q35

theorem total_points_theorem : A = 13 := by sorry

end NUMINAMATH_CALUDE_total_points_theorem_l2243_224367


namespace NUMINAMATH_CALUDE_specific_pairings_probability_eva_tom_june_leo_probability_l2243_224348

/-- The probability of two specific pairings in a class -/
theorem specific_pairings_probability (n : ℕ) (h : n ≥ 28) :
  (1 : ℚ) / (n - 1) * (1 : ℚ) / (n - 2) = 1 / ((n - 1) * (n - 2)) :=
sorry

/-- The probability of Eva being paired with Tom and June being paired with Leo -/
theorem eva_tom_june_leo_probability :
  (1 : ℚ) / 27 * (1 : ℚ) / 26 = 1 / 702 :=
sorry

end NUMINAMATH_CALUDE_specific_pairings_probability_eva_tom_june_leo_probability_l2243_224348


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2243_224389

theorem fraction_subtraction :
  (4 : ℚ) / 5 - (1 : ℚ) / 5 = (3 : ℚ) / 5 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2243_224389


namespace NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l2243_224347

/-- Given a group of children with known happiness status and gender distribution,
    calculate the number of children who are neither happy nor sad. -/
theorem children_neither_happy_nor_sad
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (boys : ℕ)
  (girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : boys = 18)
  (h5 : girls = 42)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : total_children = boys + girls)
  : total_children - (happy_children + sad_children) = 20 := by
  sorry

end NUMINAMATH_CALUDE_children_neither_happy_nor_sad_l2243_224347


namespace NUMINAMATH_CALUDE_least_sum_of_primes_l2243_224371

theorem least_sum_of_primes (p q : ℕ) : 
  Prime p → Prime q → 
  (∀ n : ℕ, n > 0 → (n^(3*p*q) - n) % (3*p*q) = 0) → 
  (∀ p' q' : ℕ, Prime p' → Prime q' → 
    (∀ n : ℕ, n > 0 → (n^(3*p'*q') - n) % (3*p'*q') = 0) → 
    p' + q' ≥ p + q) →
  p + q = 28 := by
sorry

end NUMINAMATH_CALUDE_least_sum_of_primes_l2243_224371


namespace NUMINAMATH_CALUDE_range_of_a_l2243_224349

def A : Set ℝ := {x | 1 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x | -a < x ∧ x ≤ a + 3}

theorem range_of_a (a : ℝ) : B a ⊆ (A ∩ B a) → a ≤ -1 ∧ a ∈ Set.Iic (-1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2243_224349


namespace NUMINAMATH_CALUDE_surface_area_difference_l2243_224340

/-- Calculates the difference between the sum of surface areas of smaller cubes
    and the surface area of a larger cube containing them. -/
theorem surface_area_difference (larger_volume : ℝ) (num_smaller_cubes : ℕ) (smaller_volume : ℝ) :
  larger_volume = 64 →
  num_smaller_cubes = 64 →
  smaller_volume = 1 →
  (num_smaller_cubes : ℝ) * (6 * smaller_volume ^ (2/3)) - 6 * larger_volume ^ (2/3) = 288 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_difference_l2243_224340


namespace NUMINAMATH_CALUDE_vector_problem_l2243_224357

/-- Given two collinear vectors a and b in ℝ², with b = (1, -2) and a ⋅ b = -10,
    prove that a = (-2, 4) and |a + c| = 5 where c = (6, -7) -/
theorem vector_problem (a b c : ℝ × ℝ) : 
  (∃ (k : ℝ), a = k • b) →  -- a and b are collinear
  b = (1, -2) → 
  a.1 * b.1 + a.2 * b.2 = -10 →  -- dot product
  c = (6, -7) → 
  a = (-2, 4) ∧ 
  Real.sqrt ((a.1 + c.1)^2 + (a.2 + c.2)^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l2243_224357


namespace NUMINAMATH_CALUDE_field_trip_cost_l2243_224363

def total_cost (students : ℕ) (teachers : ℕ) (bus_capacity : ℕ) (rental_cost : ℕ) (toll_cost : ℕ) : ℕ :=
  let total_people := students + teachers
  let buses_needed := (total_people + bus_capacity - 1) / bus_capacity
  buses_needed * (rental_cost + toll_cost)

theorem field_trip_cost :
  total_cost 252 8 41 300000 7500 = 2152500 :=
by sorry

end NUMINAMATH_CALUDE_field_trip_cost_l2243_224363


namespace NUMINAMATH_CALUDE_solution_set_g_range_of_a_l2243_224311

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := |x| + 2 * |x + 2 - a|

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := g a (x - 2)

-- Part 1: Solution set for g(x) ≤ 4 when a = 3
theorem solution_set_g (x : ℝ) :
  g 3 x ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 := by sorry

-- Part 2: Range of a such that f(x) ≥ 1 for all x ∈ ℝ
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_g_range_of_a_l2243_224311


namespace NUMINAMATH_CALUDE_program_output_l2243_224385

theorem program_output : ∀ (a b : ℕ), a = 1 → b = 2 → a + b = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_program_output_l2243_224385


namespace NUMINAMATH_CALUDE_number_of_topics_six_students_three_groups_ninety_arrangements_l2243_224369

theorem number_of_topics (num_students : Nat) (num_groups : Nat) (num_arrangements : Nat) : Nat :=
  let students_per_group := num_students / num_groups
  let ways_to_divide := num_arrangements / (num_groups^students_per_group)
  ways_to_divide

theorem six_students_three_groups_ninety_arrangements :
  number_of_topics 6 3 90 = 1 := by sorry

end NUMINAMATH_CALUDE_number_of_topics_six_students_three_groups_ninety_arrangements_l2243_224369


namespace NUMINAMATH_CALUDE_non_monotonic_function_parameter_range_l2243_224368

/-- The function f(x) = (1/3)x^3 - x^2 + ax - 5 is not monotonic in the interval [-1, 2] -/
theorem non_monotonic_function_parameter_range (a : ℝ) : 
  (∃ x y, x ∈ Set.Icc (-1 : ℝ) 2 ∧ y ∈ Set.Icc (-1 : ℝ) 2 ∧ x < y ∧ 
    ((1/3)*x^3 - x^2 + a*x) > ((1/3)*y^3 - y^2 + a*y)) ↔ 
  a ∈ Set.Ioo (-3 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_non_monotonic_function_parameter_range_l2243_224368


namespace NUMINAMATH_CALUDE_unique_integer_pair_for_equal_W_values_l2243_224388

/-- The polynomial W(x) = x^4 - 3x^3 + 5x^2 - 9x -/
def W (x : ℤ) : ℤ := x^4 - 3*x^3 + 5*x^2 - 9*x

/-- Theorem: The only pair of different integers (a, b) satisfying W(a) = W(b) is (1, 2) -/
theorem unique_integer_pair_for_equal_W_values :
  ∀ a b : ℤ, a ≠ b ∧ W a = W b ↔ (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_pair_for_equal_W_values_l2243_224388


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2243_224319

theorem complex_equation_solution (m : ℝ) :
  (Complex.I : ℂ)^2 = -1 →
  (↑m + 2 * Complex.I) * (2 - Complex.I) = 4 + 3 * Complex.I →
  m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2243_224319


namespace NUMINAMATH_CALUDE_cheese_slices_left_l2243_224379

/-- Represents the number of slices in each pizza -/
def slices_per_pizza : ℕ := 16

/-- Represents the total number of people -/
def total_people : ℕ := 4

/-- Represents the number of pepperoni slices left -/
def pepperoni_left : ℕ := 1

/-- Represents the number of people who eat both types of pizza -/
def people_eating_both : ℕ := 3

/-- Calculates the total number of slices eaten by the person who only eats pepperoni -/
def pepperoni_only_eater_slices : ℕ := slices_per_pizza - (pepperoni_left + 1)

/-- Calculates the number of pepperoni slices eaten by people who eat both types -/
def pepperoni_eaten_by_both : ℕ := slices_per_pizza - pepperoni_only_eater_slices - pepperoni_left

/-- Theorem stating that the number of cheese slices left is 7 -/
theorem cheese_slices_left : 
  slices_per_pizza - (pepperoni_eaten_by_both / people_eating_both * people_eating_both) = 7 := by
  sorry

end NUMINAMATH_CALUDE_cheese_slices_left_l2243_224379


namespace NUMINAMATH_CALUDE_circle_symmetry_l2243_224356

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 1 = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  2*x - y + 3 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x+3)^2 + (y-2)^2 = 2

-- Theorem statement
theorem circle_symmetry :
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
  original_circle x₁ y₁ →
  symmetric_circle x₂ y₂ →
  ∃ (x_m y_m : ℝ),
    symmetry_line x_m y_m ∧
    x_m = (x₁ + x₂) / 2 ∧
    y_m = (y₁ + y₂) / 2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2243_224356


namespace NUMINAMATH_CALUDE_weight_on_switch_l2243_224355

theorem weight_on_switch (total_weight : ℕ) (additional_weight : ℕ) 
  (h1 : total_weight = 712)
  (h2 : additional_weight = 478) :
  total_weight - additional_weight = 234 := by
  sorry

end NUMINAMATH_CALUDE_weight_on_switch_l2243_224355


namespace NUMINAMATH_CALUDE_intersection_point_expression_value_l2243_224359

/-- Given a point P(a,b) at the intersection of y=x-2 and y=1/x,
    prove that (a-a²/(a+b)) ÷ (a²b²/(a²-b²)) equals 2 -/
theorem intersection_point_expression_value (a b : ℝ) 
  (h1 : b = a - 2)
  (h2 : b = 1 / a)
  (h3 : a ≠ 0)
  (h4 : b ≠ 0)
  (h5 : a ≠ b)
  (h6 : a + b ≠ 0) :
  (a - a^2 / (a + b)) / (a^2 * b^2 / (a^2 - b^2)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_expression_value_l2243_224359


namespace NUMINAMATH_CALUDE_allison_marbles_count_l2243_224312

theorem allison_marbles_count (albert angela allison : ℕ) 
  (h1 : albert = 3 * angela)
  (h2 : angela = allison + 8)
  (h3 : albert + allison = 136) :
  allison = 28 := by
sorry

end NUMINAMATH_CALUDE_allison_marbles_count_l2243_224312


namespace NUMINAMATH_CALUDE_find_b_value_l2243_224352

theorem find_b_value (a b : ℝ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 2) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_b_value_l2243_224352


namespace NUMINAMATH_CALUDE_kitae_pencils_l2243_224346

def total_pens : ℕ := 12
def pencil_cost : ℕ := 1000
def pen_cost : ℕ := 1300
def total_spent : ℕ := 15000

theorem kitae_pencils (pencils : ℕ) (pens : ℕ) 
  (h1 : pencils + pens = total_pens)
  (h2 : pencil_cost * pencils + pen_cost * pens = total_spent) :
  pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_kitae_pencils_l2243_224346
