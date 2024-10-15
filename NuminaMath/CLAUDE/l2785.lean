import Mathlib

namespace NUMINAMATH_CALUDE_files_remaining_l2785_278599

theorem files_remaining (initial_files initial_apps final_apps files_deleted : ℕ) :
  initial_files = 24 →
  initial_apps = 13 →
  final_apps = 17 →
  files_deleted = 3 →
  initial_files - (final_apps - initial_apps) - files_deleted = 17 := by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l2785_278599


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2785_278541

theorem multiplication_puzzle :
  (142857 * 5 = 714285) ∧ (142857 * 3 = 428571) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2785_278541


namespace NUMINAMATH_CALUDE_same_remainder_difference_divisible_l2785_278518

theorem same_remainder_difference_divisible (a m b : ℤ) : 
  (∃ r : ℤ, a % b = r ∧ m % b = r) → b ∣ (a - m) := by sorry

end NUMINAMATH_CALUDE_same_remainder_difference_divisible_l2785_278518


namespace NUMINAMATH_CALUDE_geometric_series_problem_l2785_278531

theorem geometric_series_problem (n : ℝ) : 
  let a₁ : ℝ := 18
  let r₁ : ℝ := 6 / 18
  let S₁ : ℝ := a₁ / (1 - r₁)
  let r₂ : ℝ := (6 + n) / 18
  let S₂ : ℝ := a₁ / (1 - r₂)
  S₂ = 5 * S₁ → n = 9.6 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_problem_l2785_278531


namespace NUMINAMATH_CALUDE_division_count_is_eight_l2785_278583

/-- Represents an L-shaped piece consisting of three cells -/
structure LPiece where
  -- Define properties of an L-shaped piece if needed

/-- Represents a 3 × 6 rectangle -/
structure Rectangle where
  width : Nat
  height : Nat

/-- Represents a division of the rectangle into L-shaped pieces -/
structure Division where
  pieces : List LPiece

/-- Function to count valid divisions of the rectangle into L-shaped pieces -/
def countValidDivisions (rect : Rectangle) : Nat :=
  sorry

/-- Theorem stating that the number of ways to divide a 3 × 6 rectangle 
    into L-shaped pieces of three cells is 8 -/
theorem division_count_is_eight :
  let rect : Rectangle := { width := 6, height := 3 }
  countValidDivisions rect = 8 := by
  sorry

end NUMINAMATH_CALUDE_division_count_is_eight_l2785_278583


namespace NUMINAMATH_CALUDE_f_equal_range_l2785_278540

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/2) * x + (3/2) else Real.log x

theorem f_equal_range (m n : ℝ) (h1 : m < n) (h2 : f m = f n) :
  n - m ∈ Set.Icc (5 - 2 * Real.log 2) (Real.exp 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_f_equal_range_l2785_278540


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l2785_278580

/-- A geometric sequence {a_n} satisfying given conditions -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), ∀ (k : ℕ), a (k + 1) = q * a k

theorem geometric_sequence_solution (a : ℕ → ℚ) :
  geometric_sequence a →
  a 3 + a 6 = 36 →
  a 4 + a 7 = 18 →
  (∃ n : ℕ, a n = 1/2) →
  ∃ n : ℕ, a n = 1/2 ∧ n = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l2785_278580


namespace NUMINAMATH_CALUDE_donovan_lap_time_donovan_lap_time_is_45_l2785_278511

/-- Given two runners on a circular track, this theorem proves the lap time of the slower runner. -/
theorem donovan_lap_time (michael_lap_time : ℕ) (laps_to_pass : ℕ) : ℕ :=
  let michael_total_time := michael_lap_time * laps_to_pass
  let donovan_laps := laps_to_pass - 1
  michael_total_time / donovan_laps

/-- Proves that Donovan's lap time is 45 seconds given the problem conditions. -/
theorem donovan_lap_time_is_45 :
  donovan_lap_time 40 9 = 45 := by
  sorry

end NUMINAMATH_CALUDE_donovan_lap_time_donovan_lap_time_is_45_l2785_278511


namespace NUMINAMATH_CALUDE_odd_numbers_sum_product_l2785_278592

theorem odd_numbers_sum_product (n : ℕ) (odds : Finset ℕ) (a b c : ℕ) : 
  n = 1997 →
  odds.card = n →
  (∀ x ∈ odds, Odd x) →
  (odds.sum id = odds.prod id) →
  a ∈ odds ∧ b ∈ odds ∧ c ∈ odds →
  a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 →
  a.Prime ∧ b.Prime ∧ c.Prime →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a = 5 ∧ b = 7 ∧ c = 59) ∨ (a = 5 ∧ b = 59 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 5 ∧ c = 59) ∨ (a = 7 ∧ b = 59 ∧ c = 5) ∨ 
  (a = 59 ∧ b = 5 ∧ c = 7) ∨ (a = 59 ∧ b = 7 ∧ c = 5) :=
by sorry


end NUMINAMATH_CALUDE_odd_numbers_sum_product_l2785_278592


namespace NUMINAMATH_CALUDE_gcd_90_210_l2785_278512

theorem gcd_90_210 : Nat.gcd 90 210 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_90_210_l2785_278512


namespace NUMINAMATH_CALUDE_probability_one_black_ball_l2785_278567

def total_balls : ℕ := 4
def black_balls : ℕ := 2
def white_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem probability_one_black_ball :
  (Nat.choose black_balls 1 * Nat.choose white_balls 1) / Nat.choose total_balls drawn_balls = 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_black_ball_l2785_278567


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2785_278571

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- Given that point A(1, a) and point B(b, -2) are symmetric with respect to the origin O, prove that a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin 1 a b (-2)) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2785_278571


namespace NUMINAMATH_CALUDE_carpet_for_room_l2785_278558

/-- Calculates the minimum number of whole square yards of carpet needed for a rectangular room with overlap -/
def carpet_needed (length width overlap : ℕ) : ℕ :=
  let adjusted_length := length + 2 * overlap
  let adjusted_width := width + 2 * overlap
  let area := adjusted_length * adjusted_width
  (area + 8) / 9  -- Adding 8 before division by 9 to round up

theorem carpet_for_room : carpet_needed 15 9 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_carpet_for_room_l2785_278558


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_35_l2785_278537

theorem smallest_divisible_by_18_and_35 : 
  ∀ n : ℕ, n > 0 → (18 ∣ n) → (35 ∣ n) → n ≥ 630 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_35_l2785_278537


namespace NUMINAMATH_CALUDE_cookie_problem_l2785_278544

theorem cookie_problem :
  ∃! N : ℕ, 0 < N ∧ N < 150 ∧ N % 13 = 5 ∧ N % 8 = 3 :=
by sorry

end NUMINAMATH_CALUDE_cookie_problem_l2785_278544


namespace NUMINAMATH_CALUDE_nonagon_diagonals_count_l2785_278503

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := (nonagon_sides * (nonagon_sides - 3)) / 2

theorem nonagon_diagonals_count : nonagon_diagonals = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_count_l2785_278503


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l2785_278527

theorem smallest_four_digit_multiple_of_112 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 112 ∣ n → 1008 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_112_l2785_278527


namespace NUMINAMATH_CALUDE_abc_value_l2785_278552

theorem abc_value (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (sum_squares : a^2 + b^2 + c^2 = 2)
  (sum_cubes : a^3 + b^3 + c^3 = 3) :
  a * b * c = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_abc_value_l2785_278552


namespace NUMINAMATH_CALUDE_larger_number_problem_l2785_278536

theorem larger_number_problem (x y : ℝ) : 
  y = x + 10 →  -- One number exceeds another by 10
  x = y / 2 →   -- The smaller number is half the larger number
  x + y = 34 →  -- Their sum is 34
  y = 20        -- The larger number is 20
:= by sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2785_278536


namespace NUMINAMATH_CALUDE_octagon_perimeter_l2785_278551

/-- The perimeter of a regular octagon with side length 2 units is 16 units. -/
theorem octagon_perimeter : ℕ → ℕ → ℕ
  | 8, 2 => 16
  | _, _ => 0

#check octagon_perimeter

end NUMINAMATH_CALUDE_octagon_perimeter_l2785_278551


namespace NUMINAMATH_CALUDE_substitution_result_l2785_278547

theorem substitution_result (x y : ℝ) :
  y = 2 * x + 1 ∧ 5 * x - 2 * y = 7 →
  5 * x - 4 * x - 2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_substitution_result_l2785_278547


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l2785_278563

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let bounceSequence := (List.range numBounces).map (fun n => initialHeight * bounceRatio^n)
  let ascents := bounceSequence.sum
  let descents := initialHeight + (List.take (numBounces - 1) bounceSequence).sum
  ascents + descents

/-- The problem statement -/
theorem ball_bounce_distance :
  ∃ (d : ℝ), abs (totalDistance 20 (2/3) 4 - d) < 1 ∧ Int.floor d = 68 := by
  sorry


end NUMINAMATH_CALUDE_ball_bounce_distance_l2785_278563


namespace NUMINAMATH_CALUDE_train_speed_l2785_278517

/-- Calculates the speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length time_to_pass : ℝ) :
  train_length = 300 →
  bridge_length = 115 →
  time_to_pass = 42.68571428571429 →
  ∃ (speed : ℝ), abs (speed - 35.01) < 0.01 ∧ 
    speed = (train_length + bridge_length) / time_to_pass * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2785_278517


namespace NUMINAMATH_CALUDE_subset_sum_theorem_l2785_278582

theorem subset_sum_theorem (a₁ a₂ a₃ a₄ : ℝ) 
  (h : (a₁ + a₂) + (a₁ + a₃) + (a₁ + a₄) + (a₂ + a₃) + (a₂ + a₄) + (a₃ + a₄) + 
       (a₁ + a₂ + a₃) + (a₁ + a₂ + a₄) + (a₁ + a₃ + a₄) + (a₂ + a₃ + a₄) = 28) :
  a₁ + a₂ + a₃ + a₄ = 4 := by
sorry

end NUMINAMATH_CALUDE_subset_sum_theorem_l2785_278582


namespace NUMINAMATH_CALUDE_total_cows_l2785_278581

theorem total_cows (cows_per_herd : ℕ) (num_herds : ℕ) (h1 : cows_per_herd = 40) (h2 : num_herds = 8) :
  cows_per_herd * num_herds = 320 := by
  sorry

end NUMINAMATH_CALUDE_total_cows_l2785_278581


namespace NUMINAMATH_CALUDE_chess_match_probability_l2785_278532

theorem chess_match_probability (p_win p_draw : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_draw = 0.2) : 
  p_win + p_draw = 0.6 := by
sorry

end NUMINAMATH_CALUDE_chess_match_probability_l2785_278532


namespace NUMINAMATH_CALUDE_min_value_of_sequence_sequence_satisfies_conditions_l2785_278502

def sequence_a (n : ℕ) : ℝ :=
  if n = 0 then 0
  else if n = 1 then 98
  else 102 + (n - 2) * (2 * n + 2)

theorem min_value_of_sequence (n : ℕ) (h : n > 0) :
  sequence_a n / n ≥ 26 ∧ ∃ m : ℕ, m > 0 ∧ sequence_a m / m = 26 :=
by
  sorry

theorem sequence_satisfies_conditions :
  sequence_a 2 = 102 ∧
  ∀ n : ℕ, n > 0 → sequence_a (n + 1) - sequence_a n = 4 * n :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sequence_sequence_satisfies_conditions_l2785_278502


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2785_278561

theorem quadratic_inequality_solution :
  {x : ℝ | x^2 - 40*x + 350 ≤ 0} = Set.Icc 10 30 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2785_278561


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2785_278573

/-- Quadratic trinomial with integer coefficients -/
structure QuadraticTrinomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate a quadratic trinomial at a given x -/
def evaluate (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  (q.a : ℝ) * x^2 + (q.b : ℝ) * x + (q.c : ℝ)

/-- A quadratic trinomial is positive for all real x -/
def IsAlwaysPositive (q : QuadraticTrinomial) : Prop :=
  ∀ x : ℝ, evaluate q x > 0

theorem quadratic_inequality {f g : QuadraticTrinomial} 
  (hf : IsAlwaysPositive f) 
  (hg : IsAlwaysPositive g)
  (h : ∀ x : ℝ, evaluate f x / evaluate g x ≥ Real.sqrt 2) :
  ∀ x : ℝ, evaluate f x / evaluate g x > Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2785_278573


namespace NUMINAMATH_CALUDE_problem_solution_l2785_278504

theorem problem_solution (x y : ℝ) (hx_pos : x > 0) :
  (2/3 : ℝ) * x = (144/216 : ℝ) * (1/x) ∧ y * (1/x) = Real.sqrt x → x = 1 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2785_278504


namespace NUMINAMATH_CALUDE_three_card_selection_l2785_278591

/-- The number of cards in the special deck -/
def deck_size : ℕ := 60

/-- The number of cards to be picked -/
def cards_to_pick : ℕ := 3

/-- The number of ways to choose and order 3 different cards from a 60-card deck -/
def ways_to_pick : ℕ := 205320

/-- Theorem stating that the number of ways to choose and order 3 different cards
    from a 60-card deck is equal to 205320 -/
theorem three_card_selection :
  (deck_size * (deck_size - 1) * (deck_size - 2)) = ways_to_pick :=
by sorry

end NUMINAMATH_CALUDE_three_card_selection_l2785_278591


namespace NUMINAMATH_CALUDE_min_production_time_proof_l2785_278548

/-- Represents the production capacity of a factory --/
structure FactoryCapacity where
  typeI : ℝ  -- Units of Type I produced per day
  typeII : ℝ -- Units of Type II produced per day

/-- Represents the total order quantity --/
structure OrderQuantity where
  typeI : ℝ
  typeII : ℝ

/-- Calculates the minimum production time given factory capacities and order quantity --/
def minProductionTime (factoryA factoryB : FactoryCapacity) (order : OrderQuantity) : ℝ :=
  sorry

/-- Theorem stating the minimum production time for the given problem --/
theorem min_production_time_proof 
  (factoryA : FactoryCapacity)
  (factoryB : FactoryCapacity)
  (order : OrderQuantity)
  (h1 : factoryA.typeI = 30 ∧ factoryA.typeII = 20)
  (h2 : factoryB.typeI = 50 ∧ factoryB.typeII = 40)
  (h3 : order.typeI = 1500 ∧ order.typeII = 800) :
  minProductionTime factoryA factoryB order = 31.25 :=
sorry

end NUMINAMATH_CALUDE_min_production_time_proof_l2785_278548


namespace NUMINAMATH_CALUDE_picnic_theorem_l2785_278560

-- Define the propositions
variable (P : Prop) -- "The picnic on Sunday will be held"
variable (Q : Prop) -- "The weather is fair on Sunday"

-- State the given condition
axiom given_statement : (¬P → ¬Q)

-- State the theorem to be proved
theorem picnic_theorem : Q → P := by sorry

end NUMINAMATH_CALUDE_picnic_theorem_l2785_278560


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l2785_278579

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 5683 : ℤ) ≡ 420 [ZMOD 17] ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 5683 : ℤ) ≡ 420 [ZMOD 17] → x ≤ y :=
by
  use 7
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l2785_278579


namespace NUMINAMATH_CALUDE_min_value_ab_l2785_278577

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b + 3 = a * b) :
  a * b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ + 3 = a₀ * b₀ ∧ a₀ * b₀ = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_ab_l2785_278577


namespace NUMINAMATH_CALUDE_constant_c_value_l2785_278555

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end NUMINAMATH_CALUDE_constant_c_value_l2785_278555


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l2785_278578

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 4 ∧ b = 5 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c = 3 ∨ c = Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l2785_278578


namespace NUMINAMATH_CALUDE_slower_walking_speed_l2785_278533

/-- Proves that the slower walking speed is 10 km/hr given the conditions of the problem -/
theorem slower_walking_speed 
  (actual_distance : ℝ) 
  (faster_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 13.333333333333332)
  (h2 : faster_speed = 25)
  (h3 : additional_distance = 20)
  : ∃ (v : ℝ), 
    actual_distance / v = (actual_distance + additional_distance) / faster_speed ∧ 
    v = 10 := by
  sorry

end NUMINAMATH_CALUDE_slower_walking_speed_l2785_278533


namespace NUMINAMATH_CALUDE_fractional_linear_transformation_cross_ratio_l2785_278530

theorem fractional_linear_transformation_cross_ratio 
  (a b c d : ℝ) (h : a * d - b * c ≠ 0)
  (x₁ x₂ x₃ x₄ : ℝ) :
  let y : ℝ → ℝ := λ x => (a * x + b) / (c * x + d)
  let y₁ := y x₁
  let y₂ := y x₂
  let y₃ := y x₃
  let y₄ := y x₄
  (y₃ - y₁) / (y₃ - y₂) / ((y₄ - y₁) / (y₄ - y₂)) = 
  (x₃ - x₁) / (x₃ - x₂) / ((x₄ - x₁) / (x₄ - x₂)) :=
by sorry

end NUMINAMATH_CALUDE_fractional_linear_transformation_cross_ratio_l2785_278530


namespace NUMINAMATH_CALUDE_tea_shop_problem_l2785_278507

/-- Tea shop problem -/
theorem tea_shop_problem 
  (cost_A : ℝ) 
  (cost_B : ℝ) 
  (num_B_more : ℕ) 
  (cost_ratio : ℝ) 
  (total_boxes : ℕ) 
  (sell_price_A : ℝ) 
  (sell_price_B : ℝ) 
  (discount : ℝ) 
  (profit : ℝ)
  (h1 : cost_A = 4000)
  (h2 : cost_B = 8400)
  (h3 : num_B_more = 10)
  (h4 : cost_ratio = 1.4)
  (h5 : total_boxes = 100)
  (h6 : sell_price_A = 300)
  (h7 : sell_price_B = 400)
  (h8 : discount = 0.3)
  (h9 : profit = 5800) :
  ∃ (cost_per_A cost_per_B : ℝ) (num_A num_B : ℕ),
    cost_per_A = 200 ∧ 
    cost_per_B = 280 ∧ 
    num_A = 40 ∧ 
    num_B = 60 ∧
    cost_B / cost_per_B - cost_A / cost_per_A = num_B_more ∧
    cost_per_B = cost_ratio * cost_per_A ∧
    num_A + num_B = total_boxes ∧
    (sell_price_A - cost_per_A) * (num_A / 2) + 
    (sell_price_A * (1 - discount) - cost_per_A) * (num_A / 2) +
    (sell_price_B - cost_per_B) * (num_B / 2) + 
    (sell_price_B * (1 - discount) - cost_per_B) * (num_B / 2) = profit :=
by
  sorry

end NUMINAMATH_CALUDE_tea_shop_problem_l2785_278507


namespace NUMINAMATH_CALUDE_rhombus_area_theorem_l2785_278556

/-- A rhombus with perpendicular bisecting diagonals -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  perpendicular_bisectors : Bool

/-- Calculate the area of a rhombus given its properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem: The area of a rhombus with side length √117 and diagonals differing by 8 units is 101 -/
theorem rhombus_area_theorem (r : Rhombus) 
    (h1 : r.side_length = Real.sqrt 117)
    (h2 : r.diagonal_difference = 8)
    (h3 : r.perpendicular_bisectors = true) : 
  rhombus_area r = 101 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_theorem_l2785_278556


namespace NUMINAMATH_CALUDE_set_equality_l2785_278596

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by sorry

end NUMINAMATH_CALUDE_set_equality_l2785_278596


namespace NUMINAMATH_CALUDE_debby_zoo_pictures_l2785_278587

theorem debby_zoo_pictures : ∀ (zoo_pics museum_pics deleted_pics remaining_pics : ℕ),
  museum_pics = 12 →
  deleted_pics = 14 →
  remaining_pics = 22 →
  zoo_pics + museum_pics - deleted_pics = remaining_pics →
  zoo_pics = 24 := by
sorry

end NUMINAMATH_CALUDE_debby_zoo_pictures_l2785_278587


namespace NUMINAMATH_CALUDE_bacteria_growth_l2785_278585

/-- The number of cells after a given number of days, where the initial population
    doubles every two days. -/
def cell_population (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * 2^(days / 2)

/-- Theorem stating that given an initial population of 4 cells that double
    every two days, the number of cells after 10 days is 64. -/
theorem bacteria_growth : cell_population 4 10 = 64 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l2785_278585


namespace NUMINAMATH_CALUDE_sin_period_2x_minus_pi_div_6_l2785_278562

/-- The minimum positive period of y = sin(2x - π/6) is π -/
theorem sin_period_2x_minus_pi_div_6 (x : ℝ) :
  let f := fun x => Real.sin (2 * x - π / 6)
  ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
  (∀ q, q > 0 → (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = π :=
sorry

end NUMINAMATH_CALUDE_sin_period_2x_minus_pi_div_6_l2785_278562


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2785_278576

/-- The sum of the infinite series ∑_{k=1}^∞ (k^2 / 3^k) is equal to 7/8 -/
theorem infinite_series_sum : 
  ∑' k : ℕ+, (k : ℝ)^2 / 3^(k : ℝ) = 7/8 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2785_278576


namespace NUMINAMATH_CALUDE_man_mass_on_boat_l2785_278508

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth boat_sink_height : ℝ) : ℝ :=
  boat_length * boat_breadth * boat_sink_height * 1000

/-- Theorem stating that a man who causes a 7m x 3m boat to sink by 1cm has a mass of 210 kg. -/
theorem man_mass_on_boat : 
  mass_of_man 7 3 0.01 = 210 := by sorry

end NUMINAMATH_CALUDE_man_mass_on_boat_l2785_278508


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2785_278516

/-- Given a right triangle with one leg of length 15 and the angle opposite that leg measuring 30°, 
    the hypotenuse has length 30. -/
theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (h1 : leg = 15) (h2 : angle = 30) :
  let hypotenuse := 2 * leg
  hypotenuse = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2785_278516


namespace NUMINAMATH_CALUDE_total_swim_time_l2785_278543

def freestyle : ℕ := 48

def backstroke (f : ℕ) : ℕ := f + 4

def butterfly (b : ℕ) : ℕ := b + 3

def breaststroke (t : ℕ) : ℕ := t + 2

theorem total_swim_time :
  freestyle + backstroke freestyle + butterfly (backstroke freestyle) + breaststroke (butterfly (backstroke freestyle)) = 212 := by
  sorry

end NUMINAMATH_CALUDE_total_swim_time_l2785_278543


namespace NUMINAMATH_CALUDE_point_position_on_line_l2785_278535

/-- Given points on a line, prove the position of a point P satisfying a ratio condition -/
theorem point_position_on_line (a b c d e : ℝ) :
  ∀ (O A B C D E P : ℝ),
    O < A ∧ A < B ∧ B < C ∧ C < D ∧  -- Points are ordered on the line
    A - O = 2 * a ∧                  -- OA = 2a
    B - O = b ∧                      -- OB = b
    C - O = 3 * c ∧                  -- OC = 3c
    D - O = d ∧                      -- OD = d
    E - O = e ∧                      -- OE = e
    B ≤ P ∧ P ≤ C ∧                  -- P is between B and C
    (A - P) * (P - E) = (B - P) * (P - C) →  -- AP:PE = BP:PC
  P - O = (b * e - 6 * a * c) / (2 * a + 3 * c - b - e) :=
by sorry

end NUMINAMATH_CALUDE_point_position_on_line_l2785_278535


namespace NUMINAMATH_CALUDE_sandy_token_ratio_l2785_278586

theorem sandy_token_ratio : 
  ∀ (total_tokens : ℕ) (num_siblings : ℕ) (extra_tokens : ℕ),
    total_tokens = 1000000 →
    num_siblings = 4 →
    extra_tokens = 375000 →
    ∃ (tokens_per_sibling : ℕ),
      tokens_per_sibling * num_siblings + (tokens_per_sibling + extra_tokens) = total_tokens ∧
      (tokens_per_sibling + extra_tokens) * 2 = total_tokens :=
by sorry

end NUMINAMATH_CALUDE_sandy_token_ratio_l2785_278586


namespace NUMINAMATH_CALUDE_tan_sum_product_equals_sqrt_three_l2785_278506

theorem tan_sum_product_equals_sqrt_three : 
  Real.tan (17 * π / 180) + Real.tan (43 * π / 180) + 
  Real.sqrt 3 * Real.tan (17 * π / 180) * Real.tan (43 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_equals_sqrt_three_l2785_278506


namespace NUMINAMATH_CALUDE_semicircle_to_cone_volume_l2785_278584

/-- The volume of a cone formed by rolling up a semicircle -/
theorem semicircle_to_cone_volume (R : ℝ) (R_pos : R > 0) :
  let r := R / 2
  let h := R * (Real.sqrt 3) / 2
  (1 / 3) * Real.pi * r^2 * h = (Real.pi * R^3 * Real.sqrt 3) / 24 :=
by sorry

end NUMINAMATH_CALUDE_semicircle_to_cone_volume_l2785_278584


namespace NUMINAMATH_CALUDE_term1_and_term2_are_like_terms_l2785_278570

-- Define a structure for terms
structure Term where
  coefficient : ℚ
  x_power : ℕ
  y_power : ℕ

-- Define what it means for two terms to be like terms
def are_like_terms (t1 t2 : Term) : Prop :=
  t1.x_power = t2.x_power ∧ t1.y_power = t2.y_power

-- Define the two terms we're comparing
def term1 : Term := { coefficient := 4, x_power := 2, y_power := 1 }
def term2 : Term := { coefficient := -1, x_power := 2, y_power := 1 }

-- Theorem stating that term1 and term2 are like terms
theorem term1_and_term2_are_like_terms : are_like_terms term1 term2 := by
  sorry


end NUMINAMATH_CALUDE_term1_and_term2_are_like_terms_l2785_278570


namespace NUMINAMATH_CALUDE_y_coordinate_order_l2785_278542

/-- A quadratic function passing through three specific points -/
def quadratic_function (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- The y-coordinate of point A -/
def y₁ (c : ℝ) : ℝ := quadratic_function c (-1)

/-- The y-coordinate of point B -/
def y₂ (c : ℝ) : ℝ := quadratic_function c 2

/-- The y-coordinate of point C -/
def y₃ (c : ℝ) : ℝ := quadratic_function c 5

/-- Theorem stating the order of y-coordinates -/
theorem y_coordinate_order (c : ℝ) : y₁ c > y₃ c ∧ y₃ c > y₂ c := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_order_l2785_278542


namespace NUMINAMATH_CALUDE_right_triangle_sin_z_l2785_278514

theorem right_triangle_sin_z (X Y Z : ℝ) : 
  X + Y + Z = π →
  X = π / 2 →
  Real.cos Y = 3 / 5 →
  Real.sin Z = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_z_l2785_278514


namespace NUMINAMATH_CALUDE_quadratic_point_on_graph_l2785_278566

theorem quadratic_point_on_graph (a m : ℝ) (ha : a > 0) (hm : m ≠ 0) :
  (3 = -a * m^2 + 2 * a * m + 3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_on_graph_l2785_278566


namespace NUMINAMATH_CALUDE_games_within_division_is_16_l2785_278568

/-- Represents a baseball league with two divisions -/
structure BaseballLeague where
  /-- Number of games played against each team in the same division -/
  n : ℕ
  /-- Number of games played against each team in the other division -/
  m : ℕ
  /-- n is greater than 3m -/
  n_gt_3m : n > 3 * m
  /-- m is greater than 6 -/
  m_gt_6 : m > 6
  /-- Total number of games each team plays is 96 -/
  total_games : 4 * n + 5 * m = 96

/-- The number of games a team plays within its own division -/
def games_within_division (league : BaseballLeague) : ℕ := 4 * league.n

/-- Theorem stating that the number of games played within a team's division is 16 -/
theorem games_within_division_is_16 (league : BaseballLeague) :
  games_within_division league = 16 := by
  sorry

end NUMINAMATH_CALUDE_games_within_division_is_16_l2785_278568


namespace NUMINAMATH_CALUDE_inequality_proof_l2785_278597

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (a^2 * b + a + b^2) * (a * b^2 + a^2 + b) > 9 * a^2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2785_278597


namespace NUMINAMATH_CALUDE_football_cost_l2785_278528

theorem football_cost (total_cost marbles_cost baseball_cost : ℚ)
  (h1 : total_cost = 20.52)
  (h2 : marbles_cost = 9.05)
  (h3 : baseball_cost = 6.52) :
  total_cost - marbles_cost - baseball_cost = 4.95 := by
  sorry

end NUMINAMATH_CALUDE_football_cost_l2785_278528


namespace NUMINAMATH_CALUDE_distance_focus_to_line_l2785_278534

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point A (directrix intersection with x-axis)
def A : ℝ × ℝ := (-1, 0)

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * (x + 1)

-- State the theorem
theorem distance_focus_to_line :
  let d := Real.sqrt 3
  ∃ (x y : ℝ), parabola x y ∧ 
               line_l (A.1) (A.2) ∧
               (F.1 - x)^2 + (F.2 - y)^2 = d^2 :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_line_l2785_278534


namespace NUMINAMATH_CALUDE_age_proof_l2785_278500

/-- The age of a person whose current age is three times what it was six years ago. -/
def age : ℕ := 9

theorem age_proof : age = 9 := by
  have h : age = 3 * (age - 6) := by sorry
  sorry

end NUMINAMATH_CALUDE_age_proof_l2785_278500


namespace NUMINAMATH_CALUDE_complement_union_M_N_l2785_278572

universe u

def U : Finset ℕ := {1,2,3,4,5}
def M : Finset ℕ := {1,2}
def N : Finset ℕ := {3,4}

theorem complement_union_M_N : (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_union_M_N_l2785_278572


namespace NUMINAMATH_CALUDE_vector_at_negative_one_l2785_278525

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point_at_zero : ℝ × ℝ × ℝ
  point_at_one : ℝ × ℝ × ℝ

/-- The vector on the line at a given parameter value -/
def vector_at_t (line : ParameterizedLine) (t : ℝ) : ℝ × ℝ × ℝ :=
  let (x₀, y₀, z₀) := line.point_at_zero
  let (x₁, y₁, z₁) := line.point_at_one
  (x₀ + t * (x₁ - x₀), y₀ + t * (y₁ - y₀), z₀ + t * (z₁ - z₀))

theorem vector_at_negative_one (line : ParameterizedLine) 
  (h₀ : line.point_at_zero = (2, 6, 16))
  (h₁ : line.point_at_one = (1, 1, 8)) :
  vector_at_t line (-1) = (3, 11, 24) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_negative_one_l2785_278525


namespace NUMINAMATH_CALUDE_gcd_lcm_product_18_42_l2785_278598

theorem gcd_lcm_product_18_42 : Nat.gcd 18 42 * Nat.lcm 18 42 = 756 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_18_42_l2785_278598


namespace NUMINAMATH_CALUDE_pete_age_triple_son_l2785_278595

/-- 
Given:
- Pete's current age is 35
- Pete's son's current age is 9

Prove that in 4 years, Pete will be exactly three times older than his son.
-/
theorem pete_age_triple_son (pete_age : ℕ) (son_age : ℕ) : 
  pete_age = 35 → son_age = 9 → 
  ∃ (years : ℕ), years = 4 ∧ pete_age + years = 3 * (son_age + years) :=
by sorry

end NUMINAMATH_CALUDE_pete_age_triple_son_l2785_278595


namespace NUMINAMATH_CALUDE_unique_triplet_divisibility_l2785_278549

theorem unique_triplet_divisibility :
  ∃! (a b c : ℕ), 
    (∀ n : ℕ, (∀ p < 2015, Nat.Prime p → ¬(p ∣ n)) → 
      (n + c ∣ a^n + b^n + n)) ∧
    a = 1 ∧ b = 1 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_divisibility_l2785_278549


namespace NUMINAMATH_CALUDE_function_shift_l2785_278553

/-- Given a function f(x) = (x(x+3))/2, prove that f(x-1) = (x^2 + x - 2)/2 -/
theorem function_shift (x : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x * (x + 3)) / 2
  f (x - 1) = (x^2 + x - 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_function_shift_l2785_278553


namespace NUMINAMATH_CALUDE_sixteen_not_valid_l2785_278538

/-- Represents a set of lines in a plane -/
structure LineSet where
  numLines : ℕ
  intersectionCount : ℕ

/-- Checks if a LineSet is valid according to the problem conditions -/
def isValidLineSet (ls : LineSet) : Prop :=
  ls.intersectionCount = 10 ∧
  ∃ (n k : ℕ), n > 1 ∧ k > 0 ∧ ls.numLines = n * k ∧ (n - 1) * k = ls.intersectionCount

/-- Theorem stating that 16 cannot be a valid number of lines in the set -/
theorem sixteen_not_valid : ¬ (∃ (ls : LineSet), ls.numLines = 16 ∧ isValidLineSet ls) := by
  sorry


end NUMINAMATH_CALUDE_sixteen_not_valid_l2785_278538


namespace NUMINAMATH_CALUDE_cubic_fraction_factorization_l2785_278521

theorem cubic_fraction_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_fraction_factorization_l2785_278521


namespace NUMINAMATH_CALUDE_pythagorean_theorem_for_triples_scaled_right_triangle_is_pythagorean_l2785_278519

/-- Pythagorean numbers are positive integers that can be the lengths of the sides of a right triangle. -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_theorem_for_triples :
  ∀ (a b c : ℕ), isPythagoreanTriple a b c → a * a + b * b = c * c :=
sorry

theorem scaled_right_triangle_is_pythagorean :
  ∀ (a b c : ℕ), isPythagoreanTriple a b c → isPythagoreanTriple (2*a) (2*b) (2*c) :=
sorry

end NUMINAMATH_CALUDE_pythagorean_theorem_for_triples_scaled_right_triangle_is_pythagorean_l2785_278519


namespace NUMINAMATH_CALUDE_finger_multiplication_rule_l2785_278559

theorem finger_multiplication_rule (n : ℕ) (h : 1 ≤ n ∧ n ≤ 9) : 9 * n = 10 * (n - 1) + (10 - n) := by
  sorry

end NUMINAMATH_CALUDE_finger_multiplication_rule_l2785_278559


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2785_278589

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (1 + x) + (1 + x)^2 + (1 + x)^3 + (1 + x)^4 + (1 + x)^5 + (1 + x)^6 + (1 + x)^7 + (1 + x)^8
           = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 502 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2785_278589


namespace NUMINAMATH_CALUDE_modulus_of_z_l2785_278575

theorem modulus_of_z (i z : ℂ) (hi : i * i = -1) (hz : i * z = 3 + 4 * i) : Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2785_278575


namespace NUMINAMATH_CALUDE_girls_in_first_year_l2785_278520

theorem girls_in_first_year 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (boys_in_sample : ℕ) 
  (h1 : total_students = 2400) 
  (h2 : sample_size = 80) 
  (h3 : boys_in_sample = 42) : 
  ℕ := by
  sorry

#check girls_in_first_year

end NUMINAMATH_CALUDE_girls_in_first_year_l2785_278520


namespace NUMINAMATH_CALUDE_time_difference_per_question_l2785_278554

def english_questions : ℕ := 30
def math_questions : ℕ := 15
def english_time_hours : ℚ := 1
def math_time_hours : ℚ := (3/2)

def english_time_minutes : ℚ := english_time_hours * 60
def math_time_minutes : ℚ := math_time_hours * 60

def english_time_per_question : ℚ := english_time_minutes / english_questions
def math_time_per_question : ℚ := math_time_minutes / math_questions

theorem time_difference_per_question :
  math_time_per_question - english_time_per_question = 4 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_per_question_l2785_278554


namespace NUMINAMATH_CALUDE_three_lines_theorem_l2785_278522

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne_points : point1 ≠ point2

/-- Three lines in 3D space -/
structure ThreeLines where
  line1 : Line3D
  line2 : Line3D
  line3 : Line3D

/-- Predicate to check if three lines are coplanar -/
def are_coplanar (lines : ThreeLines) : Prop :=
  sorry

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate to check if three lines intersect at a single point -/
def intersect_at_point (lines : ThreeLines) : Prop :=
  sorry

/-- Predicate to check if three lines are parallel -/
def are_parallel (lines : ThreeLines) : Prop :=
  sorry

/-- Theorem stating that three non-coplanar lines with no two being skew
    either intersect at a single point or are parallel -/
theorem three_lines_theorem (lines : ThreeLines) 
  (h1 : ¬ are_coplanar lines)
  (h2 : ¬ are_skew lines.line1 lines.line2)
  (h3 : ¬ are_skew lines.line1 lines.line3)
  (h4 : ¬ are_skew lines.line2 lines.line3) :
  intersect_at_point lines ∨ are_parallel lines :=
sorry

end NUMINAMATH_CALUDE_three_lines_theorem_l2785_278522


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2785_278557

theorem rationalize_denominator : (14 : ℝ) / Real.sqrt 14 = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2785_278557


namespace NUMINAMATH_CALUDE_algebraic_simplification_l2785_278501

theorem algebraic_simplification (a b : ℝ) : -a^2 * (-2*a*b) + 3*a * (a^2*b - 1) = 5*a^3*b - 3*a := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l2785_278501


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l2785_278565

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 1986

/-- The number of books in Oak Grove's school libraries -/
def school_libraries_books : ℕ := 5106

/-- The total number of books in Oak Grove libraries -/
def total_books : ℕ := public_library_books + school_libraries_books

theorem oak_grove_library_books : total_books = 7092 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l2785_278565


namespace NUMINAMATH_CALUDE_third_butcher_delivery_l2785_278509

theorem third_butcher_delivery (package_weight : ℕ) (first_butcher : ℕ) (second_butcher : ℕ) (total_weight : ℕ) : 
  package_weight = 4 →
  first_butcher = 10 →
  second_butcher = 7 →
  total_weight = 100 →
  ∃ third_butcher : ℕ, 
    third_butcher * package_weight + first_butcher * package_weight + second_butcher * package_weight = total_weight ∧
    third_butcher = 8 :=
by sorry

end NUMINAMATH_CALUDE_third_butcher_delivery_l2785_278509


namespace NUMINAMATH_CALUDE_total_leaves_calculation_l2785_278524

/-- Calculates the total number of leaves falling from cherry and maple trees -/
def total_leaves (initial_cherry : ℕ) (initial_maple : ℕ) 
                 (cherry_ratio : ℕ) (maple_ratio : ℕ) 
                 (cherry_leaves : ℕ) (maple_leaves : ℕ) : ℕ :=
  (initial_cherry * cherry_ratio * cherry_leaves) + 
  (initial_maple * maple_ratio * maple_leaves)

/-- Theorem stating that the total number of leaves is 3650 -/
theorem total_leaves_calculation : 
  total_leaves 7 5 2 3 100 150 = 3650 := by
  sorry

#eval total_leaves 7 5 2 3 100 150

end NUMINAMATH_CALUDE_total_leaves_calculation_l2785_278524


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l2785_278513

theorem sum_of_cubes_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_sum_squares : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l2785_278513


namespace NUMINAMATH_CALUDE_eliana_steps_theorem_l2785_278569

/-- The number of steps Eliana walked on the first day -/
def first_day_steps : ℕ := 200 + 300

/-- The number of steps Eliana walked on the second day -/
def second_day_steps : ℕ := 2 * first_day_steps

/-- The additional steps Eliana walked on the third day -/
def third_day_additional_steps : ℕ := 100

/-- The total number of steps Eliana walked during the three days -/
def total_steps : ℕ := first_day_steps + second_day_steps + third_day_additional_steps

theorem eliana_steps_theorem : total_steps = 1600 := by
  sorry

end NUMINAMATH_CALUDE_eliana_steps_theorem_l2785_278569


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l2785_278546

-- Problem 1
theorem problem_one : (Real.sqrt 3)^2 + |1 - Real.sqrt 3| + ((-27 : ℝ)^(1/3)) = Real.sqrt 3 - 1 := by
  sorry

-- Problem 2
theorem problem_two : (Real.sqrt 12 - Real.sqrt (1/3)) * Real.sqrt 6 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l2785_278546


namespace NUMINAMATH_CALUDE_mans_speed_against_current_l2785_278526

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
theorem mans_speed_against_current 
  (speed_with_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_with_current = 12)
  (h2 : current_speed = 2) :
  speed_with_current - 2 * current_speed = 8 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_against_current_l2785_278526


namespace NUMINAMATH_CALUDE_shifted_parabola_vertex_l2785_278574

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 4

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := original_parabola (x - 1) - 2

-- Theorem statement
theorem shifted_parabola_vertex :
  ∃ (vertex_x vertex_y : ℝ),
    vertex_x = 0 ∧
    vertex_y = 2 ∧
    ∀ (x : ℝ), shifted_parabola x ≤ shifted_parabola vertex_x :=
by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_vertex_l2785_278574


namespace NUMINAMATH_CALUDE_inequality_proof_l2785_278590

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) : 
  (a * (3 * a - 1)) / (1 + a^2) + 
  (b * (3 * b - 1)) / (1 + b^2) + 
  (c * (3 * c - 1)) / (1 + c^2) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2785_278590


namespace NUMINAMATH_CALUDE_age_ratio_theorem_l2785_278594

/-- Represents the ages of John and Mary -/
structure Ages where
  john : ℕ
  mary : ℕ

/-- The conditions of the problem -/
def problem_conditions (a : Ages) : Prop :=
  (a.john - 5 = 2 * (a.mary - 5)) ∧ 
  (a.john - 12 = 3 * (a.mary - 12))

/-- The ratio condition we're looking for -/
def ratio_condition (a : Ages) (years : ℕ) : Prop :=
  3 * (a.mary + years) = 2 * (a.john + years)

/-- The main theorem -/
theorem age_ratio_theorem (a : Ages) :
  problem_conditions a → ∃ years : ℕ, years = 9 ∧ ratio_condition a years := by
  sorry


end NUMINAMATH_CALUDE_age_ratio_theorem_l2785_278594


namespace NUMINAMATH_CALUDE_tom_solo_time_is_four_l2785_278593

/-- The time it takes Avery to build the wall alone, in hours -/
def avery_time : ℝ := 2

/-- The time Avery and Tom work together, in hours -/
def together_time : ℝ := 1

/-- The time it takes Tom to finish the wall after Avery leaves, in hours -/
def tom_finish_time : ℝ := 1

/-- The time it takes Tom to build the wall alone, in hours -/
def tom_solo_time : ℝ := 4

/-- Theorem stating that Tom's solo time is 4 hours -/
theorem tom_solo_time_is_four :
  (1 / avery_time + 1 / tom_solo_time) * together_time + 
  (1 / tom_solo_time) * tom_finish_time = 1 →
  tom_solo_time = 4 := by
sorry

end NUMINAMATH_CALUDE_tom_solo_time_is_four_l2785_278593


namespace NUMINAMATH_CALUDE_square_root_of_nine_l2785_278539

theorem square_root_of_nine :
  ∃ x : ℝ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_square_root_of_nine_l2785_278539


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l2785_278515

theorem complex_expression_equals_negative_two :
  let A := (Real.sqrt 6 + Real.sqrt 2) * (Real.sqrt 3 - 2) * Real.sqrt (Real.sqrt 3 + 2)
  A = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_two_l2785_278515


namespace NUMINAMATH_CALUDE_unique_solution_l2785_278510

/-- Two functions f and g from ℝ to ℝ satisfying the given functional equation -/
def SatisfyEquation (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - g y) = (g x)^2 - y

/-- The theorem stating that the only functions satisfying the equation are the identity function -/
theorem unique_solution {f g : ℝ → ℝ} (h : SatisfyEquation f g) :
    (∀ x : ℝ, f x = x) ∧ (∀ x : ℝ, g x = x) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2785_278510


namespace NUMINAMATH_CALUDE_bat_wings_area_is_four_l2785_278529

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the "bat wings" shape -/
structure BatWings where
  rect : Rectangle
  quarterCircleRadius : ℝ

/-- Calculate the area of the "bat wings" -/
noncomputable def batWingsArea (bw : BatWings) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem bat_wings_area_is_four :
  ∀ (bw : BatWings),
    bw.rect.width = 4 ∧
    bw.rect.height = 5 ∧
    bw.quarterCircleRadius = 2 →
    batWingsArea bw = 4 := by
  sorry

end NUMINAMATH_CALUDE_bat_wings_area_is_four_l2785_278529


namespace NUMINAMATH_CALUDE_seating_theorem_l2785_278523

/-- The number of seats in the row -/
def n : ℕ := 7

/-- The number of people to be seated -/
def k : ℕ := 2

/-- The number of different seating arrangements for two people in n seats
    with at least one empty seat between them -/
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.factorial / ((n - k).factorial * k.factorial)) - ((n - 1).factorial / ((n - k - 1).factorial * k.factorial))

theorem seating_theorem : seating_arrangements n k = 30 := by
  sorry

end NUMINAMATH_CALUDE_seating_theorem_l2785_278523


namespace NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l2785_278505

theorem xy_zero_necessary_not_sufficient (x y : ℝ) :
  (x^2 + y^2 = 0 → x * y = 0) ∧
  ∃ x y : ℝ, x * y = 0 ∧ x^2 + y^2 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_xy_zero_necessary_not_sufficient_l2785_278505


namespace NUMINAMATH_CALUDE_puzzle_completion_l2785_278545

theorem puzzle_completion (P : ℝ) : 
  (P ≥ 0) →
  (P ≤ 1) →
  ((1 - P) * 0.8 * 0.7 * 1000 = 504) →
  (P = 0.1) := by
sorry

end NUMINAMATH_CALUDE_puzzle_completion_l2785_278545


namespace NUMINAMATH_CALUDE_sector_central_angle_l2785_278588

theorem sector_central_angle (arc_length : Real) (area : Real) :
  arc_length = π → area = 2 * π → ∃ (r : Real) (α : Real),
    r > 0 ∧ α > 0 ∧ area = 1/2 * r * arc_length ∧ arc_length = r * α ∧ α = π/4 :=
by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2785_278588


namespace NUMINAMATH_CALUDE_elijah_card_decks_l2785_278550

theorem elijah_card_decks (total_cards : ℕ) (cards_per_deck : ℕ) (h1 : total_cards = 312) (h2 : cards_per_deck = 52) :
  total_cards / cards_per_deck = 6 :=
by sorry

end NUMINAMATH_CALUDE_elijah_card_decks_l2785_278550


namespace NUMINAMATH_CALUDE_sum_and_product_of_averages_l2785_278564

def avg1 : ℚ := (0 + 100) / 2

def avg2 : ℚ := (0 + 50) / 2

def avg3 : ℚ := 560 / 8

theorem sum_and_product_of_averages :
  avg1 + avg2 + avg3 = 145 ∧ avg1 * avg2 * avg3 = 87500 := by sorry

end NUMINAMATH_CALUDE_sum_and_product_of_averages_l2785_278564
