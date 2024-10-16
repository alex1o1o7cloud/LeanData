import Mathlib

namespace NUMINAMATH_CALUDE_sequence_eleventh_term_l2257_225753

/-- Given a sequence a₁, a₂, ..., where a₁ = 3 and aₙ₊₁ - aₙ = n for n ≥ 1,
    prove that a₁₁ = 58. -/
theorem sequence_eleventh_term (a : ℕ → ℕ) 
  (h1 : a 1 = 3)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = n) : 
  a 11 = 58 := by
  sorry

end NUMINAMATH_CALUDE_sequence_eleventh_term_l2257_225753


namespace NUMINAMATH_CALUDE_dinner_lunch_difference_l2257_225714

-- Define the number of cakes served during lunch
def lunch_cakes : ℕ := 6

-- Define the number of cakes served during dinner
def dinner_cakes : ℕ := 9

-- Theorem stating the difference between dinner and lunch cakes
theorem dinner_lunch_difference : dinner_cakes - lunch_cakes = 3 := by
  sorry

end NUMINAMATH_CALUDE_dinner_lunch_difference_l2257_225714


namespace NUMINAMATH_CALUDE_fraction_ordering_l2257_225777

theorem fraction_ordering : 
  (5 : ℚ) / 19 < 7 / 21 ∧ 7 / 21 < 9 / 23 := by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l2257_225777


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2257_225704

def A : Set ℝ := {x | x ≥ -1}
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}

theorem intersection_A_complement_B : A ∩ (Set.univ \ B) = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2257_225704


namespace NUMINAMATH_CALUDE_river_speed_proof_l2257_225707

theorem river_speed_proof (rowing_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) 
  (h1 : rowing_speed = 6)
  (h2 : total_time = 1)
  (h3 : total_distance = 5.76) :
  ∃ (river_speed : ℝ),
    river_speed = 1.2 ∧
    (total_distance / 2) / (rowing_speed - river_speed) +
    (total_distance / 2) / (rowing_speed + river_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_river_speed_proof_l2257_225707


namespace NUMINAMATH_CALUDE_max_intersection_points_l2257_225738

/-- The maximum number of intersection points given 8 planes in 3D space -/
theorem max_intersection_points (n : ℕ) (h : n = 8) : 
  (Nat.choose n 3 : ℕ) = 56 := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_l2257_225738


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2257_225728

-- Define the hyperbola and its properties
def Hyperbola (m : ℝ) : Prop :=
  m > 0 ∧ ∃ x y : ℝ, x^2 / m - y^2 = 1

-- Define the asymptotic line
def AsymptoticLine (x y : ℝ) : Prop :=
  x + 3 * y = 0

-- Theorem statement
theorem hyperbola_asymptote (m : ℝ) :
  Hyperbola m → (∃ x y : ℝ, AsymptoticLine x y) → m = 9 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2257_225728


namespace NUMINAMATH_CALUDE_ratio_problem_l2257_225784

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 2) : 
  (a + b) / (b + c) = 4/9 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2257_225784


namespace NUMINAMATH_CALUDE_range_of_a_l2257_225703

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Condition: For all x ∈ ℝ, f'(x) < x
axiom f'_less_than_x : ∀ x, f' x < x

-- Condition: f(1-a) - f(a) ≤ 1/2 - a
axiom inequality_condition : ∀ a, f (1 - a) - f a ≤ 1/2 - a

-- Theorem: The range of values for a is a ≤ 1/2
theorem range_of_a : ∀ a, (∀ x, f (1 - x) - f x ≤ 1/2 - x) → a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2257_225703


namespace NUMINAMATH_CALUDE_direction_vector_form_l2257_225788

/-- Given a line passing through two points, prove that its direction vector
    has a specific form. -/
theorem direction_vector_form (p1 p2 : ℝ × ℝ) (c : ℝ) : 
  p1 = (-6, 1) →
  p2 = (-1, 5) →
  (p2.1 - p1.1, p2.2 - p1.2) = (5, c) →
  c = 4 := by
  sorry

end NUMINAMATH_CALUDE_direction_vector_form_l2257_225788


namespace NUMINAMATH_CALUDE_right_angled_parallelopiped_l2257_225768

structure Parallelopiped where
  AB : ℝ
  AA' : ℝ

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

def is_right_angled (M N P : Point) : Prop :=
  (M.x - N.x) * (P.x - N.x) + (M.y - N.y) * (P.y - N.y) + (M.z - N.z) * (P.z - N.z) = 0

theorem right_angled_parallelopiped (p : Parallelopiped) (N : Point) :
  p.AB = 12 * Real.sqrt 3 →
  p.AA' = 18 →
  N.x = 9 * Real.sqrt 3 ∧ N.y = 0 ∧ N.z = 0 →
  ∃ P : Point, P.x = 0 ∧ P.y = 0 ∧ P.z = 27 / 2 ∧
    ∀ M : Point, M.x = 12 * Real.sqrt 3 → M.z = 18 →
      is_right_angled M N P := by
  sorry

#check right_angled_parallelopiped

end NUMINAMATH_CALUDE_right_angled_parallelopiped_l2257_225768


namespace NUMINAMATH_CALUDE_mathematics_players_count_l2257_225762

def total_players : ℕ := 15
def physics_players : ℕ := 9
def both_subjects : ℕ := 3

theorem mathematics_players_count :
  ∃ (math_players : ℕ),
    math_players = total_players - physics_players + both_subjects ∧
    math_players = 9 :=
by sorry

end NUMINAMATH_CALUDE_mathematics_players_count_l2257_225762


namespace NUMINAMATH_CALUDE_processing_box_function_l2257_225773

-- Define the types of boxes in a flowchart
inductive FlowchartBox
  | Processing
  | Decision
  | Terminal
  | InputOutput

-- Define the functions of boxes in a flowchart
def boxFunction : FlowchartBox → String
  | FlowchartBox.Processing => "assignment and calculation"
  | FlowchartBox.Decision => "determine execution direction"
  | FlowchartBox.Terminal => "start and end of algorithm"
  | FlowchartBox.InputOutput => "handle data input and output"

-- Theorem statement
theorem processing_box_function :
  boxFunction FlowchartBox.Processing = "assignment and calculation" := by
  sorry

end NUMINAMATH_CALUDE_processing_box_function_l2257_225773


namespace NUMINAMATH_CALUDE_existence_of_unequal_positive_numbers_l2257_225719

theorem existence_of_unequal_positive_numbers : ∃ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≠ b ∧ a + b = a * b := by
  sorry

end NUMINAMATH_CALUDE_existence_of_unequal_positive_numbers_l2257_225719


namespace NUMINAMATH_CALUDE_math_book_cost_l2257_225741

theorem math_book_cost (total_books : ℕ) (history_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 80 →
  history_book_cost = 5 →
  total_price = 373 →
  math_books = 27 →
  ∃ (math_book_cost : ℕ),
    math_book_cost * math_books + history_book_cost * (total_books - math_books) = total_price ∧
    math_book_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_math_book_cost_l2257_225741


namespace NUMINAMATH_CALUDE_solve_for_y_l2257_225799

theorem solve_for_y (x y : ℚ) (h1 : x = 102) (h2 : x^3*y - 3*x^2*y + 3*x*y = 106200) : 
  y = 10/97 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l2257_225799


namespace NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l2257_225737

theorem unique_n_satisfying_conditions :
  ∃! n : ℤ,
    0 ≤ n ∧ n ≤ 8 ∧
    ∃ x : ℤ,
      x > 0 ∧
      (-4567 + x ≥ 0) ∧
      (∀ y : ℤ, y > 0 ∧ -4567 + y ≥ 0 → x ≤ y) ∧
      n ≡ -4567 + x [ZMOD 9] ∧
    n = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_conditions_l2257_225737


namespace NUMINAMATH_CALUDE_min_value_theorem_l2257_225754

theorem min_value_theorem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  let A := Real.sqrt (x + 3) + Real.sqrt (y + 7) + Real.sqrt (z + 12)
  let B := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)
  A^2 - B^2 ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2257_225754


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2257_225718

/-- For a quadratic equation x^2 + 4x + k = 0 to have real roots, k must be less than or equal to 4 -/
theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, x^2 + 4*x + k = 0) ↔ k ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l2257_225718


namespace NUMINAMATH_CALUDE_f_composition_neg_two_l2257_225787

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

-- State the theorem
theorem f_composition_neg_two : f (f (-2)) = 1/2 := by sorry

end NUMINAMATH_CALUDE_f_composition_neg_two_l2257_225787


namespace NUMINAMATH_CALUDE_red_balls_count_l2257_225729

theorem red_balls_count (total white green yellow purple : ℕ) (prob : ℚ) : 
  total = 60 ∧ 
  white = 22 ∧ 
  green = 10 ∧ 
  yellow = 7 ∧ 
  purple = 6 ∧ 
  prob = 65 / 100 ∧ 
  (white + green + yellow : ℚ) / total = prob →
  total - (white + green + yellow + purple) = 0 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l2257_225729


namespace NUMINAMATH_CALUDE_satellite_forecast_probability_l2257_225701

theorem satellite_forecast_probability (p_a p_b : ℝ) (h_a : p_a = 0.8) (h_b : p_b = 0.75) :
  1 - (1 - p_a) * (1 - p_b) = 0.95 := by
  sorry

end NUMINAMATH_CALUDE_satellite_forecast_probability_l2257_225701


namespace NUMINAMATH_CALUDE_base7_multiplication_addition_l2257_225723

/-- Converts a base 7 number represented as a list of digits to a natural number -/
def base7ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 7 * acc + d) 0

/-- Converts a natural number to its base 7 representation as a list of digits -/
def natToBase7 (n : Nat) : List Nat :=
  if n < 7 then [n]
  else (n % 7) :: natToBase7 (n / 7)

theorem base7_multiplication_addition :
  (base7ToNat [5, 2]) * (base7ToNat [3]) + (base7ToNat [4, 4, 1]) =
  base7ToNat [3, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_base7_multiplication_addition_l2257_225723


namespace NUMINAMATH_CALUDE_select_computers_l2257_225791

theorem select_computers (type_a : ℕ) (type_b : ℕ) : 
  type_a = 4 → type_b = 5 → 
  (Nat.choose type_a 2 * Nat.choose type_b 1) + (Nat.choose type_a 1 * Nat.choose type_b 2) = 70 := by
  sorry

end NUMINAMATH_CALUDE_select_computers_l2257_225791


namespace NUMINAMATH_CALUDE_units_digit_of_2_pow_20_minus_1_l2257_225749

theorem units_digit_of_2_pow_20_minus_1 : 
  (2^20 - 1) % 10 = 5 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_2_pow_20_minus_1_l2257_225749


namespace NUMINAMATH_CALUDE_school_election_votes_l2257_225759

theorem school_election_votes (total_votes : ℕ) 
  (h1 : 45 = (3 : ℕ) * total_votes / 8)
  (h2 : (1 : ℕ) * total_votes / 4 + (3 : ℕ) * total_votes / 8 ≤ total_votes) : 
  total_votes = 120 := by
sorry

end NUMINAMATH_CALUDE_school_election_votes_l2257_225759


namespace NUMINAMATH_CALUDE_original_solution_concentration_l2257_225758

/-- Proves that given the conditions, the original solution's concentration is 50% -/
theorem original_solution_concentration
  (replaced_portion : ℝ)
  (h_replaced : replaced_portion = 0.8181818181818182)
  (x : ℝ)
  (h_result : x / 100 * (1 - replaced_portion) + 30 / 100 * replaced_portion = 40 / 100) :
  x = 50 :=
sorry

end NUMINAMATH_CALUDE_original_solution_concentration_l2257_225758


namespace NUMINAMATH_CALUDE_hat_price_reduction_l2257_225778

theorem hat_price_reduction (original_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  original_price = 12 ∧ first_reduction = 0.2 ∧ second_reduction = 0.25 →
  original_price * (1 - first_reduction) * (1 - second_reduction) = 7.2 := by
sorry

end NUMINAMATH_CALUDE_hat_price_reduction_l2257_225778


namespace NUMINAMATH_CALUDE_lake_shore_distance_l2257_225793

/-- Given two points A and B on the shore of a lake, and a point C chosen such that
    CA = 50 meters, CB = 30 meters, and ∠ACB = 120°, prove that the distance AB is 70 meters. -/
theorem lake_shore_distance (A B C : ℝ × ℝ) : 
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let CB := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let cos_ACB := ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2)) / (CA * CB)
  CA = 50 ∧ CB = 30 ∧ cos_ACB = -1/2 → AB = 70 := by
  sorry


end NUMINAMATH_CALUDE_lake_shore_distance_l2257_225793


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2257_225725

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 2) :
  let set := [1 - 2 / n, 1 - 2 / n] ++ List.replicate (n - 2) 1
  (set.sum / n : ℚ) = 1 - 4 / n^2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2257_225725


namespace NUMINAMATH_CALUDE_lukes_coin_piles_l2257_225782

theorem lukes_coin_piles (num_quarter_piles : ℕ) : 
  (∃ (num_dime_piles : ℕ), 
    num_quarter_piles = num_dime_piles ∧ 
    3 * num_quarter_piles + 3 * num_dime_piles = 30) → 
  num_quarter_piles = 5 := by
sorry

end NUMINAMATH_CALUDE_lukes_coin_piles_l2257_225782


namespace NUMINAMATH_CALUDE_purchase_cost_l2257_225769

/-- The cost of a single can of soda in dollars -/
def soda_cost : ℝ := 1

/-- The number of soda cans purchased -/
def num_sodas : ℕ := 3

/-- The number of soups purchased -/
def num_soups : ℕ := 2

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 1

/-- The cost of a single soup in dollars -/
def soup_cost : ℝ := num_sodas * soda_cost

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℝ := 3 * soup_cost

/-- The total cost of the purchase in dollars -/
def total_cost : ℝ := num_sodas * soda_cost + num_soups * soup_cost + num_sandwiches * sandwich_cost

theorem purchase_cost : total_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l2257_225769


namespace NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l2257_225750

theorem smallest_divisor_with_remainder (n : ℕ) : 
  (∃ (k : ℕ), n = 10 * k) ∧ 
  (19^19 + 19) % n = 18 ∧ 
  (∀ m : ℕ, m < n → m % 10 = 0 → (19^19 + 19) % m ≠ 18) → 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_with_remainder_l2257_225750


namespace NUMINAMATH_CALUDE_right_triangle_area_l2257_225779

/-- The area of a right triangle with hypotenuse 10√2 cm and one angle 45° is 50 cm² -/
theorem right_triangle_area (h : ℝ) (α : ℝ) (A : ℝ) : 
  h = 10 * Real.sqrt 2 →  -- hypotenuse is 10√2 cm
  α = 45 * π / 180 →      -- one angle is 45°
  A = h^2 / 4 →           -- area formula for 45-45-90 triangle
  A = 50 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_area_l2257_225779


namespace NUMINAMATH_CALUDE_g_composition_two_roots_l2257_225709

def g (d : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + d

theorem g_composition_two_roots (d : ℝ) : 
  (∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ ∀ x, g d (g d x) = 0 ↔ x = r₁ ∨ x = r₂) ↔ d = 8 := by
  sorry

end NUMINAMATH_CALUDE_g_composition_two_roots_l2257_225709


namespace NUMINAMATH_CALUDE_part1_part2_l2257_225760

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

-- Part 1
theorem part1 (a b : ℝ) (h1 : a ≠ 0) :
  (∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  a = -1 ∧ b = 4 := by sorry

-- Part 2
theorem part2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : f a b 1 = 2) :
  (∀ a' b', a' > 0 → b' > 0 → f a' b' 1 = 2 → 1/a + 4/b ≤ 1/a' + 4/b') →
  1/a + 4/b = 9 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l2257_225760


namespace NUMINAMATH_CALUDE_C_nec_not_suff_A_l2257_225734

-- Define the propositions
variable (A B C : Prop)

-- Define the relationships between A, B, and C
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_nec_and_suff_C : (B ↔ C)

-- State the theorem to be proved
theorem C_nec_not_suff_A : (C → A) ∧ ¬(A → C) := by sorry

end NUMINAMATH_CALUDE_C_nec_not_suff_A_l2257_225734


namespace NUMINAMATH_CALUDE_xyz_value_l2257_225748

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : 
  x * y * z = 5 := by sorry

end NUMINAMATH_CALUDE_xyz_value_l2257_225748


namespace NUMINAMATH_CALUDE_juan_saw_eight_pickup_trucks_l2257_225740

/-- The number of pickup trucks Juan saw -/
def num_pickup_trucks : ℕ := sorry

/-- The total number of tires on all vehicles Juan saw -/
def total_tires : ℕ := 101

/-- The number of cars Juan saw -/
def num_cars : ℕ := 15

/-- The number of bicycles Juan saw -/
def num_bicycles : ℕ := 3

/-- The number of tricycles Juan saw -/
def num_tricycles : ℕ := 1

/-- The number of tires on a car -/
def tires_per_car : ℕ := 4

/-- The number of tires on a bicycle -/
def tires_per_bicycle : ℕ := 2

/-- The number of tires on a tricycle -/
def tires_per_tricycle : ℕ := 3

/-- The number of tires on a pickup truck -/
def tires_per_pickup : ℕ := 4

theorem juan_saw_eight_pickup_trucks : num_pickup_trucks = 8 := by
  sorry

end NUMINAMATH_CALUDE_juan_saw_eight_pickup_trucks_l2257_225740


namespace NUMINAMATH_CALUDE_distance_between_cities_l2257_225733

/-- The distance between two cities given the speeds and times of two cars traveling between them -/
theorem distance_between_cities (meeting_time : ℝ) (car_b_speed : ℝ) (car_a_remaining_time : ℝ) :
  let car_a_speed := car_b_speed * meeting_time / car_a_remaining_time
  let total_distance := (car_a_speed + car_b_speed) * meeting_time
  meeting_time = 6 ∧ car_b_speed = 69 ∧ car_a_remaining_time = 4 →
  total_distance = (69 * 6 / 4 + 69) * 6 := by
sorry

end NUMINAMATH_CALUDE_distance_between_cities_l2257_225733


namespace NUMINAMATH_CALUDE_sequence_property_l2257_225781

theorem sequence_property (u : ℕ → ℤ) : 
  (∀ n m : ℕ, u (n * m) = u n + u m) → 
  (∀ n : ℕ, u n = 0) := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l2257_225781


namespace NUMINAMATH_CALUDE_consumption_increase_after_tax_reduction_l2257_225772

/-- 
Given a commodity with tax and consumption, prove that if the tax is reduced by 20% 
and the revenue decreases by 8%, then the consumption must have increased by 15%.
-/
theorem consumption_increase_after_tax_reduction (T C : ℝ) 
  (h1 : T > 0) (h2 : C > 0) : 
  (0.80 * T) * (C * (1 + 15/100)) = 0.92 * (T * C) := by
  sorry

end NUMINAMATH_CALUDE_consumption_increase_after_tax_reduction_l2257_225772


namespace NUMINAMATH_CALUDE_unique_solution_l2257_225755

theorem unique_solution : ∃! (x : ℕ), 
  x > 0 ∧ 
  let n := x^2 + 4*x + 23
  let d := 3*x + 7
  n = d*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2257_225755


namespace NUMINAMATH_CALUDE_hockey_league_season_games_l2257_225747

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 16 teams, where each team faces all other teams 10 times,
    the total number of games played in the season is 1200. -/
theorem hockey_league_season_games :
  hockey_league_games 16 10 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_hockey_league_season_games_l2257_225747


namespace NUMINAMATH_CALUDE_ludwigs_weekly_earnings_l2257_225726

/-- Calculates the weekly earnings of a worker with a specific work schedule and daily salary. -/
def weeklyEarnings (fullDays halfDays : ℕ) (dailySalary : ℚ) : ℚ :=
  (fullDays : ℚ) * dailySalary + (halfDays : ℚ) * (dailySalary / 2)

/-- Theorem stating that a worker with 4 full days, 3 half days, and a daily salary of $10 earns $55 per week. -/
theorem ludwigs_weekly_earnings :
  weeklyEarnings 4 3 10 = 55 := by
  sorry

#eval weeklyEarnings 4 3 10

end NUMINAMATH_CALUDE_ludwigs_weekly_earnings_l2257_225726


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2257_225730

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Product of the first n terms of a sequence -/
def ProductOfTerms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (fun acc i => acc * a (i + 1)) 1

theorem geometric_sequence_property (a : ℕ → ℝ) (m : ℕ) 
  (h_geo : GeometricSequence a)
  (h_prop : a (m - 1) * a (m + 1) - 2 * a m = 0)
  (h_product : ProductOfTerms a (2 * m - 1) = 128) :
  m = 4 := by
sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l2257_225730


namespace NUMINAMATH_CALUDE_max_value_f_in_interval_l2257_225720

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2

-- State the theorem
theorem max_value_f_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc (-1) 1 ∧ f c = 2 ∧ ∀ x ∈ Set.Icc (-1) 1, f x ≤ f c :=
sorry

end NUMINAMATH_CALUDE_max_value_f_in_interval_l2257_225720


namespace NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l2257_225792

theorem no_simultaneous_integer_fractions :
  ¬ ∃ (n : ℤ), (∃ (k m : ℤ), (n - 6 : ℚ) / 15 = k ∧ (n - 5 : ℚ) / 24 = m) :=
by sorry

end NUMINAMATH_CALUDE_no_simultaneous_integer_fractions_l2257_225792


namespace NUMINAMATH_CALUDE_diseased_corn_plants_l2257_225780

theorem diseased_corn_plants (grid_size : Nat) (h : grid_size = 2015) :
  let center := grid_size / 2 + 1
  let days_to_corner := center - 1
  days_to_corner * 2 = 2014 :=
sorry

end NUMINAMATH_CALUDE_diseased_corn_plants_l2257_225780


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l2257_225702

theorem no_solution_absolute_value_equation :
  ¬∃ x : ℝ, |(-2 * x + 1)| + 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l2257_225702


namespace NUMINAMATH_CALUDE_book_sale_revenue_book_sale_revenue_proof_l2257_225756

theorem book_sale_revenue : ℕ → ℕ → ℕ → Prop :=
  fun total_books sold_price remaining_books =>
    (2 * total_books = 3 * remaining_books) →
    (total_books - remaining_books) * sold_price = 288

-- Proof
theorem book_sale_revenue_proof :
  book_sale_revenue 108 4 36 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_book_sale_revenue_proof_l2257_225756


namespace NUMINAMATH_CALUDE_units_digit_sum_powers_l2257_225736

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising a number to a power, considering only the units digit -/
def powerUnitsDigit (base : ℕ) (exp : ℕ) : ℕ :=
  (unitsDigit base ^ exp) % 10

theorem units_digit_sum_powers : unitsDigit (powerUnitsDigit 53 107 + powerUnitsDigit 97 59) = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_powers_l2257_225736


namespace NUMINAMATH_CALUDE_original_team_size_l2257_225727

/-- The number of players originally in the bowling team. -/
def original_players : ℕ := sorry

/-- The average weight of the original team in kg. -/
def original_avg : ℝ := 76

/-- The weight of the first new player in kg. -/
def new_player1 : ℝ := 110

/-- The weight of the second new player in kg. -/
def new_player2 : ℝ := 60

/-- The new average weight of the team after the two new players join, in kg. -/
def new_avg : ℝ := 78

/-- Theorem stating that the original number of players in the team is 7. -/
theorem original_team_size :
  (original_avg * original_players + new_player1 + new_player2) / (original_players + 2) = new_avg →
  original_players = 7 := by sorry

end NUMINAMATH_CALUDE_original_team_size_l2257_225727


namespace NUMINAMATH_CALUDE_vector_problems_l2257_225776

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, -3)

theorem vector_problems :
  -- Part I: Dot product
  (a.1 * b.1 + a.2 * b.2 = -2) ∧
  -- Part II: Parallel vector with given magnitude
  (∃ (c : ℝ × ℝ), (c.2 / c.1 = a.2 / a.1) ∧ 
                  (c.1^2 + c.2^2 = 20) ∧ 
                  ((c = (-2, -4)) ∨ (c = (2, 4)))) ∧
  -- Part III: Perpendicular vectors condition
  (∃ (k : ℝ), ((b.1 + k * a.1)^2 + (b.2 + k * a.2)^2 = 
               (b.1 - k * a.1)^2 + (b.2 - k * a.2)^2) ∧
              (k^2 = 5)) :=
by sorry


end NUMINAMATH_CALUDE_vector_problems_l2257_225776


namespace NUMINAMATH_CALUDE_xy_plus_y_squared_l2257_225761

theorem xy_plus_y_squared (x y : ℝ) (h : x * (x + y) = x^2 + y + 12) :
  x * y + y^2 = y^2 + y + 12 := by
  sorry

end NUMINAMATH_CALUDE_xy_plus_y_squared_l2257_225761


namespace NUMINAMATH_CALUDE_marked_up_percentage_l2257_225713

theorem marked_up_percentage 
  (cost_price selling_price : ℝ)
  (discount_percentage : ℝ)
  (h1 : cost_price = 540)
  (h2 : selling_price = 456)
  (h3 : discount_percentage = 26.570048309178745) :
  (((selling_price / (1 - discount_percentage / 100) - cost_price) / cost_price) * 100 = 15) :=
by sorry

end NUMINAMATH_CALUDE_marked_up_percentage_l2257_225713


namespace NUMINAMATH_CALUDE_point_outside_region_l2257_225706

def planar_region (x y : ℝ) : Prop := 2 * x + 3 * y < 6

theorem point_outside_region :
  ¬(planar_region 0 2) ∧
  (planar_region 0 0) ∧
  (planar_region 1 1) ∧
  (planar_region 2 0) :=
sorry

end NUMINAMATH_CALUDE_point_outside_region_l2257_225706


namespace NUMINAMATH_CALUDE_zero_point_location_l2257_225771

-- Define the function f
variable {f : ℝ → ℝ}

-- Define the property of having exactly one zero point in an interval
def has_unique_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a < x ∧ x < b ∧ f x = 0

-- State the theorem
theorem zero_point_location (h1 : has_unique_zero f 0 16)
                            (h2 : has_unique_zero f 0 8)
                            (h3 : has_unique_zero f 0 4)
                            (h4 : has_unique_zero f 0 2) :
  ¬∃ x, 2 < x ∧ x < 16 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_zero_point_location_l2257_225771


namespace NUMINAMATH_CALUDE_boat_distance_downstream_l2257_225765

/-- Calculates the distance traveled downstream given boat speed, stream speed, and time -/
def distanceDownstream (boatSpeed streamSpeed time : ℝ) : ℝ :=
  (boatSpeed + streamSpeed) * time

/-- Proves that the distance traveled downstream is 54 km under the given conditions -/
theorem boat_distance_downstream :
  let boatSpeed : ℝ := 10
  let streamSpeed : ℝ := 8
  let time : ℝ := 3
  distanceDownstream boatSpeed streamSpeed time = 54 := by
sorry

#eval distanceDownstream 10 8 3

end NUMINAMATH_CALUDE_boat_distance_downstream_l2257_225765


namespace NUMINAMATH_CALUDE_max_value_problem_l2257_225721

theorem max_value_problem (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos₁ : 0 < a₁) (h_pos₂ : 0 < a₂) (h_pos₃ : 0 < a₃) (h_pos₄ : 0 < a₄)
  (h₁ : a₁ ≥ a₂ * a₃^2) (h₂ : a₂ ≥ a₃ * a₄^2) 
  (h₃ : a₃ ≥ a₄ * a₁^2) (h₄ : a₄ ≥ a₁ * a₂^2) : 
  a₁ * a₂ * a₃ * a₄ * (a₁ - a₂ * a₃^2) * (a₂ - a₃ * a₄^2) * 
  (a₃ - a₄ * a₁^2) * (a₄ - a₁ * a₂^2) ≤ 1 / 256 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l2257_225721


namespace NUMINAMATH_CALUDE_inequality_empty_solution_set_l2257_225744

theorem inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2*a ≥ 0) → a ≥ (Real.sqrt 3 + 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_empty_solution_set_l2257_225744


namespace NUMINAMATH_CALUDE_least_positive_linear_combination_l2257_225766

theorem least_positive_linear_combination : 
  ∃ (n : ℕ), n > 0 ∧ (∀ (m : ℤ), (∃ (x y : ℤ), m = 24*x + 16*y) → m = 0 ∨ m.natAbs ≥ n) ∧ 
  (∃ (x y : ℤ), n = 24*x + 16*y) :=
by sorry

end NUMINAMATH_CALUDE_least_positive_linear_combination_l2257_225766


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2257_225767

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let series_sum := a * (1 - r^n) / (1 - r)
  let a := (1 : ℚ) / 4
  let r := -(1 : ℚ) / 4
  let n := 5
  series_sum = 205 / 1024 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2257_225767


namespace NUMINAMATH_CALUDE_common_root_cubics_theorem_l2257_225712

/-- Two cubic equations with two common roots -/
structure CommonRootCubics where
  A : ℝ
  B : ℝ
  C : ℝ
  root1 : ℝ
  root2 : ℝ
  eq1_holds : ∀ x : ℝ, x^3 + A*x^2 + 20*x + C = 0 ↔ x = root1 ∨ x = root2 ∨ x = -A - root1 - root2
  eq2_holds : ∀ x : ℝ, x^3 + B*x^2 + 100 = 0 ↔ x = root1 ∨ x = root2 ∨ x = -B - root1 - root2

theorem common_root_cubics_theorem (cubics : CommonRootCubics) :
  cubics.C = 100 ∧ cubics.root1 * cubics.root2 = 5 * Real.rpow 5 (1/3) := by sorry

end NUMINAMATH_CALUDE_common_root_cubics_theorem_l2257_225712


namespace NUMINAMATH_CALUDE_box_dimensions_l2257_225722

theorem box_dimensions (a b c : ℝ) 
  (h1 : a + c = 17) 
  (h2 : a + b = 13) 
  (h3 : b + c = 20) 
  (h4 : a < b) 
  (h5 : b < c) : 
  a = 5 ∧ b = 8 ∧ c = 12 := by
sorry

end NUMINAMATH_CALUDE_box_dimensions_l2257_225722


namespace NUMINAMATH_CALUDE_vowels_on_board_l2257_225795

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The number of times each vowel is written -/
def times_written : ℕ := 2

/-- The total number of vowels written on the board -/
def total_vowels : ℕ := num_vowels * times_written

theorem vowels_on_board : total_vowels = 10 := by
  sorry

end NUMINAMATH_CALUDE_vowels_on_board_l2257_225795


namespace NUMINAMATH_CALUDE_water_requirement_proof_l2257_225786

/-- The number of households in the village -/
def num_households : ℕ := 10

/-- The total amount of water available in litres -/
def total_water : ℕ := 2000

/-- The number of months the water lasts -/
def num_months : ℕ := 10

/-- The number of litres of water required per household per month -/
def water_per_household_per_month : ℚ :=
  total_water / (num_households * num_months)

theorem water_requirement_proof :
  water_per_household_per_month = 20 := by
  sorry

end NUMINAMATH_CALUDE_water_requirement_proof_l2257_225786


namespace NUMINAMATH_CALUDE_cos_C_value_angle_C_measure_l2257_225710

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Part 1
theorem cos_C_value (abc : Triangle) 
  (h1 : Real.sin abc.A = 5/13) 
  (h2 : Real.cos abc.B = 3/5) : 
  Real.cos abc.C = -16/65 := by sorry

-- Part 2
theorem angle_C_measure (abc : Triangle) 
  (h : ∃ p : ℝ, (Real.tan abc.A)^2 + p * (Real.tan abc.A + 1) + 1 = 0 ∧ 
                (Real.tan abc.B)^2 + p * (Real.tan abc.B + 1) + 1 = 0) : 
  abc.C = 3 * Real.pi / 4 := by sorry

end NUMINAMATH_CALUDE_cos_C_value_angle_C_measure_l2257_225710


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2257_225711

theorem imaginary_part_of_complex_expression :
  Complex.im (((1 : ℂ) + Complex.I) / ((1 : ℂ) - Complex.I) + (1 - Complex.I)^2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2257_225711


namespace NUMINAMATH_CALUDE_money_distribution_l2257_225746

theorem money_distribution (a b c : ℕ) 
  (total : a + b + c = 350)
  (ac_sum : a + c = 200)
  (bc_sum : b + c = 350) :
  c = 200 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2257_225746


namespace NUMINAMATH_CALUDE_brother_payment_l2257_225763

/-- Margaux's daily earnings from her money lending company -/
structure DailyEarnings where
  friend : ℝ
  brother : ℝ
  cousin : ℝ

/-- The total earnings after a given number of days -/
def total_earnings (e : DailyEarnings) (days : ℝ) : ℝ :=
  (e.friend + e.brother + e.cousin) * days

/-- Theorem stating that Margaux's brother pays $8 per day -/
theorem brother_payment (e : DailyEarnings) :
  e.friend = 5 ∧ e.cousin = 4 ∧ total_earnings e 7 = 119 → e.brother = 8 := by
  sorry

end NUMINAMATH_CALUDE_brother_payment_l2257_225763


namespace NUMINAMATH_CALUDE_smallest_three_digit_with_digit_product_24_l2257_225716

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem smallest_three_digit_with_digit_product_24 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 24 → 146 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_with_digit_product_24_l2257_225716


namespace NUMINAMATH_CALUDE_cube_minus_self_divisible_by_six_l2257_225796

theorem cube_minus_self_divisible_by_six (n : ℕ) : 6 ∣ (n^3 - n) := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_self_divisible_by_six_l2257_225796


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l2257_225745

theorem quadratic_inequality_properties (a b c : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + c > 0 ↔ -3 < x ∧ x < 2) → 
  (a < 0 ∧ a - b + c > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l2257_225745


namespace NUMINAMATH_CALUDE_museum_group_time_l2257_225708

/-- Proves that the time taken for each group to go through the museum is 24 minutes -/
theorem museum_group_time (total_students : ℕ) (num_groups : ℕ) (time_per_student : ℕ) : 
  total_students = 18 → num_groups = 3 → time_per_student = 4 → 
  (total_students / num_groups) * time_per_student = 24 := by
  sorry

end NUMINAMATH_CALUDE_museum_group_time_l2257_225708


namespace NUMINAMATH_CALUDE_remaining_clothing_l2257_225731

theorem remaining_clothing (initial : ℕ) (donated_first : ℕ) (thrown_away : ℕ) : 
  initial = 100 →
  donated_first = 5 →
  thrown_away = 15 →
  initial - (donated_first + 3 * donated_first + thrown_away) = 65 := by
  sorry

end NUMINAMATH_CALUDE_remaining_clothing_l2257_225731


namespace NUMINAMATH_CALUDE_line_condition_perpendicular_condition_equal_intercepts_condition_l2257_225770

/-- The equation of a line with parameter m -/
def line_equation (m x y : ℝ) : Prop :=
  (m^2 - 2*m - 3)*x + (2*m^2 + m - 1)*y + 5 - 2*m = 0

/-- The condition for the equation to represent a line -/
theorem line_condition (m : ℝ) : 
  (∃ x y, line_equation m x y) ↔ m ≠ -1 :=
sorry

/-- The condition for the line to be perpendicular to the x-axis -/
theorem perpendicular_condition (m : ℝ) :
  (m^2 - 2*m - 3 = 0 ∧ 2*m^2 + m - 1 ≠ 0) ↔ m = 1/2 :=
sorry

/-- The condition for the line to have equal intercepts on both axes -/
theorem equal_intercepts_condition (m : ℝ) :
  (∃ a ≠ 0, line_equation m a 0 ∧ line_equation m 0 (-a)) ↔ m = -2 :=
sorry

end NUMINAMATH_CALUDE_line_condition_perpendicular_condition_equal_intercepts_condition_l2257_225770


namespace NUMINAMATH_CALUDE_movie_book_difference_l2257_225715

/-- The number of movies in the 'crazy silly school' series -/
def num_movies : ℕ := 47

/-- The number of books in the 'crazy silly school' series -/
def num_books : ℕ := 23

/-- Theorem: The difference between the number of movies and books in the 'crazy silly school' series is 24 -/
theorem movie_book_difference : num_movies - num_books = 24 := by
  sorry

end NUMINAMATH_CALUDE_movie_book_difference_l2257_225715


namespace NUMINAMATH_CALUDE_min_side_length_triangle_l2257_225797

theorem min_side_length_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  a + b = 2 →
  C = 2 * π / 3 →
  c = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos C)) →
  c ≥ Real.sqrt 3 ∧ (c = Real.sqrt 3 ↔ a = b) := by
sorry

end NUMINAMATH_CALUDE_min_side_length_triangle_l2257_225797


namespace NUMINAMATH_CALUDE_pizza_toppings_l2257_225785

theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (bacon_slices : ℕ) 
  (h1 : total_slices = 15)
  (h2 : cheese_slices = 8)
  (h3 : bacon_slices = 13)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range cheese_slices ∨ slice ∈ Finset.range bacon_slices)) :
  ∃ both_toppings : ℕ, both_toppings = 6 ∧ 
    cheese_slices + bacon_slices - both_toppings = total_slices :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_l2257_225785


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l2257_225735

theorem factor_implies_d_value (d : ℚ) : 
  (∀ x : ℚ, (x - 5) ∣ (d*x^4 + 19*x^3 - 10*d*x^2 + 45*x - 90)) → 
  d = -502/75 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l2257_225735


namespace NUMINAMATH_CALUDE_negative_solution_existence_l2257_225752

/-- The inequality x^2 < 4 - |x - a| has at least one negative solution if and only if a ∈ [-17/4, 4). -/
theorem negative_solution_existence (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ x^2 < 4 - |x - a|) ↔ -17/4 ≤ a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_negative_solution_existence_l2257_225752


namespace NUMINAMATH_CALUDE_inscribed_circle_probability_l2257_225775

/-- The probability of a point randomly chosen within a right-angled triangle
    with legs 8 and 15 lying inside its inscribed circle is 3π/20. -/
theorem inscribed_circle_probability : 
  let a : ℝ := 8
  let b : ℝ := 15
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let r : ℝ := (a * b) / (a + b + c)
  let triangle_area : ℝ := (1/2) * a * b
  let circle_area : ℝ := Real.pi * r^2
  (circle_area / triangle_area) = (3 * Real.pi) / 20 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_probability_l2257_225775


namespace NUMINAMATH_CALUDE_largest_n_unique_k_l2257_225764

theorem largest_n_unique_k : ∃ (n : ℕ), n > 0 ∧ n = 112 ∧
  (∃! (k : ℤ), (8 : ℚ)/15 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 7/13) ∧
  (∀ (m : ℕ), m > n → ¬∃! (k : ℤ), (8 : ℚ)/15 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 7/13) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_unique_k_l2257_225764


namespace NUMINAMATH_CALUDE_inequality_proof_l2257_225717

theorem inequality_proof (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_3 : a + b + c + d = 3) :
  1/a^3 + 1/b^3 + 1/c^3 + 1/d^3 ≤ 1/(a*b*c*d)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2257_225717


namespace NUMINAMATH_CALUDE_angle_B_measure_l2257_225742

-- Define the hexagon PROBLEMS
structure Hexagon where
  P : ℝ
  R : ℝ
  O : ℝ
  B : ℝ
  L : ℝ
  S : ℝ

-- Define the conditions
def is_valid_hexagon (h : Hexagon) : Prop :=
  h.P = h.R ∧ h.P = h.B ∧ 
  h.O + h.S = 180 ∧ 
  h.L = 90 ∧
  h.P + h.R + h.O + h.B + h.L + h.S = 720

-- State the theorem
theorem angle_B_measure (h : Hexagon) (hvalid : is_valid_hexagon h) : h.B = 150 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_measure_l2257_225742


namespace NUMINAMATH_CALUDE_same_color_probability_l2257_225794

/-- The probability that two remaining chairs are of the same color -/
theorem same_color_probability 
  (black_chairs : ℕ) 
  (brown_chairs : ℕ) 
  (h1 : black_chairs = 15) 
  (h2 : brown_chairs = 18) :
  (black_chairs * (black_chairs - 1) + brown_chairs * (brown_chairs - 1)) / 
  ((black_chairs + brown_chairs) * (black_chairs + brown_chairs - 1)) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_same_color_probability_l2257_225794


namespace NUMINAMATH_CALUDE_range_of_p_l2257_225751

/-- The function p(x) = x^4 + 6x^2 + 9 -/
def p (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

/-- The domain of the function -/
def domain : Set ℝ := { x | x ≥ 0 }

/-- The range of the function -/
def range : Set ℝ := { y | ∃ x ∈ domain, p x = y }

theorem range_of_p : range = { y | y ≥ 9 } := by sorry

end NUMINAMATH_CALUDE_range_of_p_l2257_225751


namespace NUMINAMATH_CALUDE_equation_solution_l2257_225790

theorem equation_solution : ∃ x : ℝ, (6000 - (105 / x) = 5995) ∧ x = 21 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2257_225790


namespace NUMINAMATH_CALUDE_expression_equals_one_l2257_225743

theorem expression_equals_one 
  (m n k : ℝ) 
  (h : m = 1 / (n * k)) : 
  1 / (1 + m + m * n) + 1 / (1 + n + n * k) + 1 / (1 + k + k * m) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2257_225743


namespace NUMINAMATH_CALUDE_f_composition_eq_one_fourth_l2257_225789

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_eq_one_fourth :
  f (f (1/9)) = 1/4 := by sorry

end NUMINAMATH_CALUDE_f_composition_eq_one_fourth_l2257_225789


namespace NUMINAMATH_CALUDE_common_chord_length_l2257_225757

/-- The length of the common chord of two circles -/
theorem common_chord_length (c1 c2 : ℝ × ℝ → Prop) : 
  (∀ x y, c1 (x, y) ↔ x^2 + y^2 = 4) →
  (∀ x y, c2 (x, y) ↔ x^2 + y^2 - 2*y - 6 = 0) →
  ∃ l : ℝ, l = 2 * Real.sqrt 3 ∧ 
    ∀ x y, (c1 (x, y) ∧ c2 (x, y)) → 
      (x^2 + y^2 = 4 ∧ y = -1) ∨ 
      (x^2 + y^2 - 2*y - 6 = 0 ∧ y = -1) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_length_l2257_225757


namespace NUMINAMATH_CALUDE_fraction_equals_seven_l2257_225798

theorem fraction_equals_seven : (2^2016 + 3 * 2^2014) / (2^2016 - 3 * 2^2014) = 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_seven_l2257_225798


namespace NUMINAMATH_CALUDE_max_large_planes_is_seven_l2257_225739

/-- Calculates the maximum number of planes that can fit in a hangar -/
def max_planes (hangar_length : ℕ) (plane_length : ℕ) (safety_gap : ℕ) : ℕ :=
  (hangar_length) / (plane_length + safety_gap)

/-- Theorem: The maximum number of large planes in the hangar is 7 -/
theorem max_large_planes_is_seven :
  max_planes 900 110 10 = 7 := by
  sorry

#eval max_planes 900 110 10

end NUMINAMATH_CALUDE_max_large_planes_is_seven_l2257_225739


namespace NUMINAMATH_CALUDE_power_product_rule_l2257_225774

theorem power_product_rule (a : ℝ) : (a * a^3)^2 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_power_product_rule_l2257_225774


namespace NUMINAMATH_CALUDE_shortest_distance_parabola_line_l2257_225700

/-- The shortest distance between a point on the parabola y = x^2 - 4x + 11 
    and a point on the line y = 2x - 6 is 8/√5. -/
theorem shortest_distance_parabola_line : 
  let parabola := {P : ℝ × ℝ | P.2 = P.1^2 - 4*P.1 + 11}
  let line := {Q : ℝ × ℝ | Q.2 = 2*Q.1 - 6}
  ∃ (P : ℝ × ℝ) (Q : ℝ × ℝ), P ∈ parabola ∧ Q ∈ line ∧
    ∀ (P' : ℝ × ℝ) (Q' : ℝ × ℝ), P' ∈ parabola → Q' ∈ line →
      Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ 8 / Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_parabola_line_l2257_225700


namespace NUMINAMATH_CALUDE_sine_inequality_holds_only_at_zero_l2257_225705

theorem sine_inequality_holds_only_at_zero (y : Real) :
  (y ∈ Set.Icc 0 (Real.pi / 2)) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (x + y) ≤ Real.sin x + Real.sin y) ↔
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_sine_inequality_holds_only_at_zero_l2257_225705


namespace NUMINAMATH_CALUDE_bottle_caps_given_l2257_225783

theorem bottle_caps_given (initial_caps : Real) (remaining_caps : Real) 
  (h1 : initial_caps = 7.0)
  (h2 : remaining_caps = 5.0) :
  initial_caps - remaining_caps = 2.0 := by
sorry

end NUMINAMATH_CALUDE_bottle_caps_given_l2257_225783


namespace NUMINAMATH_CALUDE_parabola_properties_l2257_225724

-- Define the parabola and its properties
def parabola (a b c m n t x₀ : ℝ) : Prop :=
  a > 0 ∧
  m = a + b + c ∧
  n = 16*a + 4*b + c ∧
  t = -b / (2*a) ∧
  3*a + b = 0 ∧
  m < c ∧ c < n ∧
  x₀ ≠ 1 ∧
  m = a * x₀^2 + b * x₀ + c

-- State the theorem
theorem parabola_properties (a b c m n t x₀ : ℝ) 
  (h : parabola a b c m n t x₀) : 
  m < n ∧ 1/2 < t ∧ t < 2 ∧ 0 < x₀ ∧ x₀ < 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2257_225724


namespace NUMINAMATH_CALUDE_combined_probability_l2257_225732

/-- The probability that Xavier solves Problem A -/
def p_xa : ℚ := 1/5

/-- The probability that Yvonne solves Problem A -/
def p_ya : ℚ := 1/2

/-- The probability that Zelda solves Problem A -/
def p_za : ℚ := 5/8

/-- The probability that Xavier solves Problem B -/
def p_xb : ℚ := 2/9

/-- The probability that Yvonne solves Problem B -/
def p_yb : ℚ := 3/5

/-- The probability that Zelda solves Problem B -/
def p_zb : ℚ := 1/4

/-- The probability that Xavier solves Problem C -/
def p_xc : ℚ := 1/4

/-- The probability that Yvonne solves Problem C -/
def p_yc : ℚ := 3/8

/-- The probability that Zelda solves Problem C -/
def p_zc : ℚ := 9/16

/-- The theorem stating the probability of the combined event -/
theorem combined_probability : 
  p_xa * p_ya * p_yb * (1 - p_yc) * (1 - p_xc) * (1 - p_zc) = 63/2048 := by
  sorry

end NUMINAMATH_CALUDE_combined_probability_l2257_225732
