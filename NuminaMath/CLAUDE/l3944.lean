import Mathlib

namespace quadratic_equations_distinct_roots_l3944_394408

theorem quadratic_equations_distinct_roots (n : ℕ) (a b : Fin n → ℝ) 
  (h_n : n ≥ 2)
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j) :
  ¬ (∀ i : Fin n, ∃ j : Fin n, (a i)^2 - a j * (a i) + b j = 0 ∨ (b i)^2 - a j * (b i) + b j = 0) :=
by sorry

end quadratic_equations_distinct_roots_l3944_394408


namespace license_plate_update_l3944_394407

/-- The number of choices for the first section in the original format -/
def original_first : Nat := 5

/-- The number of choices for the second section in the original format -/
def original_second : Nat := 3

/-- The number of choices for the third section in both formats -/
def third : Nat := 5

/-- The number of additional choices for the first section in the updated format -/
def additional_first : Nat := 1

/-- The number of additional choices for the second section in the updated format -/
def additional_second : Nat := 1

/-- The largest possible number of additional license plates after updating the format -/
def additional_plates : Nat := 45

theorem license_plate_update :
  (original_first + additional_first) * (original_second + additional_second) * third -
  original_first * original_second * third = additional_plates := by
  sorry

end license_plate_update_l3944_394407


namespace geometric_sequence_m_value_l3944_394487

/-- A geometric sequence with common ratio not equal to 1 -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  r ≠ 1 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_m_value
  (a : ℕ → ℝ) (r : ℝ) (m : ℕ)
  (h_geom : geometric_sequence a r)
  (h_eq1 : a 5 * a 6 + a 4 * a 7 = 18)
  (h_eq2 : a 1 * a m = 9) :
  m = 10 := by
  sorry

end geometric_sequence_m_value_l3944_394487


namespace graph_shift_down_2_l3944_394442

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define a vertical shift transformation
def vertical_shift (f : ℝ → ℝ) (c : ℝ) : ℝ → ℝ := λ x => f x - c

-- Theorem stating that y = f(x) - 2 is equivalent to shifting y = f(x) down by 2 units
theorem graph_shift_down_2 :
  ∀ x : ℝ, vertical_shift f 2 x = f x - 2 :=
by
  sorry

#check graph_shift_down_2

end graph_shift_down_2_l3944_394442


namespace remainders_of_p_squared_mod_120_l3944_394470

theorem remainders_of_p_squared_mod_120 (p : Nat) (h_prime : Nat.Prime p) (h_greater_than_5 : p > 5) :
  ∃ (r₁ r₂ : Nat), r₁ ≠ r₂ ∧ 
  (∀ (r : Nat), r < 120 → (p^2 % 120 = r ↔ r = r₁ ∨ r = r₂)) := by
  sorry

end remainders_of_p_squared_mod_120_l3944_394470


namespace ellipse_equation_l3944_394428

/-- Given an ellipse C with semi-major axis a, semi-minor axis b, and eccentricity e,
    where a line passing through the right focus intersects C at points A and B,
    forming a triangle AF₁B with perimeter p, prove that the equation of C is x²/3 + y²/2 = 1 --/
theorem ellipse_equation (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0)
  (e : ℝ) (h_e : e = Real.sqrt 3 / 3)
  (p : ℝ) (h_p : p = 4 * Real.sqrt 3) :
  a^2 = 3 ∧ b^2 = 2 := by sorry

end ellipse_equation_l3944_394428


namespace sixth_card_is_twelve_l3944_394477

/-- A function that checks if a list of 6 integers can be divided into 3 pairs with equal sums -/
def can_be_paired (numbers : List ℕ) : Prop :=
  numbers.length = 6 ∧
  ∃ (a b c d e f : ℕ),
    numbers = [a, b, c, d, e, f] ∧
    a + b = c + d ∧ c + d = e + f

theorem sixth_card_is_twelve :
  ∀ (x : ℕ),
    x ≥ 1 ∧ x ≤ 20 →
    can_be_paired [2, 4, 9, 17, 19, x] →
    x = 12 := by
  sorry

end sixth_card_is_twelve_l3944_394477


namespace rhombus_area_theorem_l3944_394417

/-- Represents a rhombus with side length and diagonal -/
structure Rhombus where
  side_length : ℝ
  diagonal1 : ℝ

/-- Calculates the area of a rhombus given its side length and one diagonal -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

theorem rhombus_area_theorem (r : Rhombus) :
  r.side_length = 2 * Real.sqrt 5 →
  r.diagonal1 = 4 →
  rhombus_area r = 16 :=
by sorry

end rhombus_area_theorem_l3944_394417


namespace monster_feast_l3944_394429

theorem monster_feast (sequence : Fin 3 → ℕ) 
  (double_next : ∀ i : Fin 2, sequence (Fin.succ i) = 2 * sequence i)
  (total_consumed : sequence 0 + sequence 1 + sequence 2 = 847) :
  sequence 0 = 121 := by
sorry

end monster_feast_l3944_394429


namespace half_dollar_difference_l3944_394423

/-- Represents the number of coins of each type -/
structure CoinCount where
  nickels : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- The problem constraints -/
def valid_coin_count (c : CoinCount) : Prop :=
  c.nickels + c.quarters + c.half_dollars = 60 ∧
  5 * c.nickels + 25 * c.quarters + 50 * c.half_dollars = 1000

/-- The set of all valid coin counts -/
def valid_coin_counts : Set CoinCount :=
  {c | valid_coin_count c}

/-- The maximum number of half-dollars in any valid coin count -/
noncomputable def max_half_dollars : ℕ :=
  ⨆ (c : CoinCount) (h : c ∈ valid_coin_counts), c.half_dollars

/-- The minimum number of half-dollars in any valid coin count -/
noncomputable def min_half_dollars : ℕ :=
  ⨅ (c : CoinCount) (h : c ∈ valid_coin_counts), c.half_dollars

/-- The main theorem -/
theorem half_dollar_difference :
  max_half_dollars - min_half_dollars = 15 := by
  sorry

end half_dollar_difference_l3944_394423


namespace inequality_equivalence_l3944_394409

theorem inequality_equivalence (x : ℝ) :
  (2 * x + 3) / (x + 3) > (4 * x + 5) / (3 * x + 8) ↔ 
  x < -3 ∨ x > -3/2 :=
by sorry

end inequality_equivalence_l3944_394409


namespace f_even_eq_ten_times_f_odd_l3944_394411

/-- The function f(k) counts the number of k-digit integers (including those with leading zeros)
    whose digits can be permuted to form a number divisible by 11. -/
def f (k : ℕ) : ℕ := sorry

/-- For any positive integer m, f(2m) = 10 * f(2m-1) -/
theorem f_even_eq_ten_times_f_odd (m : ℕ+) : f (2 * m) = 10 * f (2 * m - 1) := by sorry

end f_even_eq_ten_times_f_odd_l3944_394411


namespace amount_distributed_l3944_394445

/-- Proves that the amount distributed is 12000 given the conditions of the problem -/
theorem amount_distributed (A : ℕ) : 
  (A / 20 = A / 25 + 120) → A = 12000 := by
  sorry

end amount_distributed_l3944_394445


namespace electric_guitar_price_l3944_394475

theorem electric_guitar_price (total_guitars : ℕ) (total_revenue : ℕ) 
  (acoustic_price : ℕ) (electric_count : ℕ) : 
  total_guitars = 9 → 
  total_revenue = 3611 → 
  acoustic_price = 339 → 
  electric_count = 4 → 
  (total_revenue - (total_guitars - electric_count) * acoustic_price) / electric_count = 479 :=
by sorry

end electric_guitar_price_l3944_394475


namespace fibonacci_mod_127_l3944_394435

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_mod_127 :
  (∀ m : ℕ, m < 256 → (fib m % 127 ≠ 0 ∨ fib (m + 1) % 127 ≠ 1)) ∧
  fib 256 % 127 = 0 ∧ fib 257 % 127 = 1 :=
sorry

end fibonacci_mod_127_l3944_394435


namespace matrix_equation_solution_l3944_394474

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; 3, 5*x]
  ∀ x : ℝ, (A.det = 16) ↔ (x = Real.sqrt (22/15) ∨ x = -Real.sqrt (22/15)) :=
by sorry

end matrix_equation_solution_l3944_394474


namespace x_sum_greater_than_two_over_a_l3944_394451

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 
  (deriv (f a) x) / Real.exp (a * x)

theorem x_sum_greater_than_two_over_a 
  (a : ℝ) (x₁ x₂ : ℝ) (ha : a > 0) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hx_dist : x₁ ≠ x₂) (hg_eq_f : g a x₁ = f a x₂) : 
  x₁ + x₂ > 2 / a := by
  sorry

end x_sum_greater_than_two_over_a_l3944_394451


namespace fraction_sum_equality_l3944_394482

theorem fraction_sum_equality (x y z : ℝ) 
  (h : x / (30 - x) + y / (75 - y) + z / (50 - z) = 9) :
  6 / (30 - x) + 15 / (75 - y) + 10 / (50 - z) = 2.4 := by
  sorry

end fraction_sum_equality_l3944_394482


namespace initial_speed_is_80_l3944_394447

/-- Represents the speed and duration of a segment of the trip -/
structure TripSegment where
  speed : ℝ
  duration : ℝ

/-- Calculates the distance traveled given speed and time -/
def distance (segment : TripSegment) : ℝ :=
  segment.speed * segment.duration

/-- Represents Jeff's road trip -/
def JeffsTrip (initial_speed : ℝ) : List TripSegment :=
  [{ speed := initial_speed, duration := 6 },
   { speed := 60, duration := 4 },
   { speed := 40, duration := 2 }]

/-- Calculates the total distance of the trip -/
def totalDistance (trip : List TripSegment) : ℝ :=
  trip.map distance |>.sum

theorem initial_speed_is_80 :
  ∃ (v : ℝ), totalDistance (JeffsTrip v) = 800 ∧ v = 80 := by
  sorry

end initial_speed_is_80_l3944_394447


namespace tangent_line_at_2_min_value_in_interval_l3944_394418

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Theorem for the tangent line equation
theorem tangent_line_at_2 : 
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ y - f 2 = f' 2 * (x - 2) :=
sorry

-- Theorem for the minimum value in the interval [-3, 3]
theorem min_value_in_interval : 
  ∃ x₀ ∈ Set.Icc (-3 : ℝ) 3, ∀ x ∈ Set.Icc (-3 : ℝ) 3, f x₀ ≤ f x ∧ f x₀ = -17 :=
sorry

end tangent_line_at_2_min_value_in_interval_l3944_394418


namespace tan_sum_over_cos_simplification_l3944_394448

theorem tan_sum_over_cos_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (40 * π / 180) + Real.tan (50 * π / 180)) / 
  Real.cos (10 * π / 180) = 8 * Real.sqrt 3 / 3 := by
  sorry

end tan_sum_over_cos_simplification_l3944_394448


namespace polynomial_value_at_zero_l3944_394434

def polynomial_condition (p : ℝ → ℝ) : Prop :=
  ∃ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ),
    ∀ x, p x = a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀

theorem polynomial_value_at_zero (p : ℝ → ℝ) :
  polynomial_condition p →
  (∀ n : Nat, n ≤ 7 → p (3^n) = 1 / 3^n) →
  p 0 = 3280 / 2187 :=
by sorry

end polynomial_value_at_zero_l3944_394434


namespace valid_arrangement_exists_l3944_394453

def is_valid_arrangement (n : ℕ) : Prop :=
  (∀ d : ℕ, d ∈ Finset.range 10 → (n.digits 10).count d = 1) ∧
  (∀ k : ℕ, 2 ≤ k → k ≤ 18 → n % k = 0)

theorem valid_arrangement_exists : ∃ n : ℕ, is_valid_arrangement n :=
sorry

end valid_arrangement_exists_l3944_394453


namespace sin_negative_1740_degrees_l3944_394437

theorem sin_negative_1740_degrees : Real.sin ((-1740 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by sorry

end sin_negative_1740_degrees_l3944_394437


namespace special_triangle_all_angles_60_l3944_394484

/-- A triangle with angles in arithmetic progression and sides in geometric progression -/
structure SpecialTriangle where
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Sides of the triangle opposite to angles A, B, C respectively
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles form an arithmetic progression
  angle_progression : ∃ (d : ℝ), B - A = C - B
  -- One angle is 60°
  one_angle_60 : A = 60 ∨ B = 60 ∨ C = 60
  -- Sum of angles is 180°
  angle_sum : A + B + C = 180
  -- Sides form a geometric progression
  side_progression : b^2 = a * c
  -- Triangle inequality
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  -- All sides are positive
  positive_sides : a > 0 ∧ b > 0 ∧ c > 0

/-- Theorem: In a SpecialTriangle, all angles are 60° -/
theorem special_triangle_all_angles_60 (t : SpecialTriangle) : t.A = 60 ∧ t.B = 60 ∧ t.C = 60 := by
  sorry

end special_triangle_all_angles_60_l3944_394484


namespace parabola_midpoint_locus_ratio_l3944_394454

/-- A parabola with vertex and focus -/
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ

/-- The locus of midpoints of chords of a parabola -/
def midpointLocus (p : Parabola) (angle : ℝ) : Parabola :=
  sorry

/-- The ratio of distances between foci and vertices of two related parabolas -/
def focusVertexRatio (p1 p2 : Parabola) : ℝ :=
  sorry

theorem parabola_midpoint_locus_ratio (p : Parabola) :
  let q := midpointLocus p (π / 2)
  focusVertexRatio p q = 7 / 8 := by
  sorry

end parabola_midpoint_locus_ratio_l3944_394454


namespace cards_per_pack_l3944_394415

theorem cards_per_pack (num_packs : ℕ) (num_pages : ℕ) (cards_per_page : ℕ) : 
  num_packs = 60 → num_pages = 42 → cards_per_page = 10 →
  (num_pages * cards_per_page) / num_packs = 7 := by
  sorry

end cards_per_pack_l3944_394415


namespace quadratic_distinct_roots_l3944_394427

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ m < -6 ∨ m > 6 := by
sorry

end quadratic_distinct_roots_l3944_394427


namespace levis_brother_additional_scores_l3944_394400

/-- Proves that Levi's brother scored 3 more times given the initial conditions and Levi's goal -/
theorem levis_brother_additional_scores :
  ∀ (levi_initial : ℕ) (brother_initial : ℕ) (levi_additional : ℕ) (goal_difference : ℕ),
    levi_initial = 8 →
    brother_initial = 12 →
    levi_additional = 12 →
    goal_difference = 5 →
    ∃ (brother_additional : ℕ),
      levi_initial + levi_additional = brother_initial + brother_additional + goal_difference ∧
      brother_additional = 3 :=
by sorry

end levis_brother_additional_scores_l3944_394400


namespace train_passing_time_l3944_394450

/-- The length of the high-speed train in meters -/
def high_speed_train_length : ℝ := 400

/-- The length of the regular train in meters -/
def regular_train_length : ℝ := 600

/-- The time in seconds it takes for a passenger on the high-speed train to see the regular train pass -/
def high_speed_observation_time : ℝ := 3

/-- The time in seconds it takes for a passenger on the regular train to see the high-speed train pass -/
def regular_observation_time : ℝ := 2

theorem train_passing_time :
  (regular_train_length / high_speed_observation_time) * regular_observation_time = high_speed_train_length :=
by sorry

end train_passing_time_l3944_394450


namespace green_blue_tile_difference_l3944_394471

/-- Proves that the difference between green and blue tiles after adding two borders is 29 -/
theorem green_blue_tile_difference : 
  let initial_blue : ℕ := 13
  let initial_green : ℕ := 6
  let tiles_per_border : ℕ := 18
  let borders_added : ℕ := 2
  let final_green : ℕ := initial_green + borders_added * tiles_per_border
  let final_blue : ℕ := initial_blue
  final_green - final_blue = 29 := by
sorry


end green_blue_tile_difference_l3944_394471


namespace adams_game_rounds_l3944_394476

/-- Given Adam's total score and points per round, prove the number of rounds played --/
theorem adams_game_rounds (total_points : ℕ) (points_per_round : ℕ) 
  (h1 : total_points = 283) 
  (h2 : points_per_round = 71) : 
  total_points / points_per_round = 4 := by
  sorry

end adams_game_rounds_l3944_394476


namespace intersection_implies_sum_l3944_394472

-- Define the sets M, N, and K
def M : Set ℝ := {x : ℝ | x^2 - 4*x < 0}
def N (m : ℝ) : Set ℝ := {x : ℝ | m < x ∧ x < 5}
def K (n : ℝ) : Set ℝ := {x : ℝ | 3 < x ∧ x < n}

-- State the theorem
theorem intersection_implies_sum (m n : ℝ) : 
  M ∩ N m = K n → m + n = 7 := by
  sorry

end intersection_implies_sum_l3944_394472


namespace one_and_two_thirds_of_x_is_45_l3944_394493

theorem one_and_two_thirds_of_x_is_45 : ∃ x : ℚ, (5/3) * x = 45 ∧ x = 27 := by
  sorry

end one_and_two_thirds_of_x_is_45_l3944_394493


namespace function_lower_bound_l3944_394439

theorem function_lower_bound (c : ℝ) : ∀ x : ℝ, x^2 - 2*x + c ≥ c - 1 := by
  sorry

end function_lower_bound_l3944_394439


namespace frog_jump_probability_l3944_394479

/-- Represents a jump in 3D space -/
structure Jump where
  direction : Real × Real × Real
  length : Real

/-- Calculates the final position after a series of jumps -/
def finalPosition (jumps : List Jump) : Real × Real × Real :=
  sorry

/-- Calculates the distance between two points in 3D space -/
def distance (p1 p2 : Real × Real × Real) : Real :=
  sorry

/-- Calculates the probability of an event given a sample space -/
def probability (event : α → Prop) (sampleSpace : Set α) : Real :=
  sorry

theorem frog_jump_probability :
  let jumps := [
    { direction := sorry, length := 1 },
    { direction := sorry, length := 2 },
    { direction := sorry, length := 3 }
  ]
  let start := (0, 0, 0)
  let final := finalPosition jumps
  probability (λ jumps => distance start final ≤ 2) (sorry : Set (List Jump)) = 1/5 :=
sorry

end frog_jump_probability_l3944_394479


namespace special_function_properties_l3944_394464

/-- An increasing function f defined on (-1, +∞) with the property f(xy) = f(x) + f(y) -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x > -1 ∧ y > -1 → f (x * y) = f x + f y) ∧
  (∀ x y, x > -1 ∧ y > -1 ∧ x < y → f x < f y)

theorem special_function_properties
    (f : ℝ → ℝ)
    (hf : SpecialFunction f)
    (h3 : f 3 = 1) :
  (f 9 = 2) ∧
  (∀ a, a > -1 → (f a > f (a - 1) + 2 ↔ 0 < a ∧ a < 9/8)) :=
by sorry

end special_function_properties_l3944_394464


namespace acute_triangle_with_largest_five_times_smallest_l3944_394432

theorem acute_triangle_with_largest_five_times_smallest (α β γ : ℕ) : 
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- All angles are positive
  α + β + γ = 180 →  -- Sum of angles in a triangle
  α ≤ 89 ∧ β ≤ 89 ∧ γ ≤ 89 →  -- Acute triangle condition
  α ≥ β ∧ β ≥ γ →  -- Ordering of angles
  α = 5 * γ →  -- Largest angle is five times the smallest
  (α = 85 ∧ β = 78 ∧ γ = 17) := by
  sorry

#check acute_triangle_with_largest_five_times_smallest

end acute_triangle_with_largest_five_times_smallest_l3944_394432


namespace sum_of_roots_quadratic_l3944_394496

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 2*x₁ - 8 = 0) → (x₂^2 - 2*x₂ - 8 = 0) → x₁ + x₂ = 2 := by
  sorry

end sum_of_roots_quadratic_l3944_394496


namespace square_area_after_cuts_l3944_394438

theorem square_area_after_cuts (x : ℝ) : 
  x > 0 → x - 3 > 0 → x - 5 > 0 → 
  x^2 - (x - 3) * (x - 5) = 81 → 
  x^2 = 144 := by
sorry

end square_area_after_cuts_l3944_394438


namespace area_to_paint_l3944_394446

-- Define the wall dimensions
def wall_height : ℝ := 10
def wall_width : ℝ := 15

-- Define the unpainted area dimensions
def unpainted_height : ℝ := 3
def unpainted_width : ℝ := 5

-- Theorem to prove
theorem area_to_paint : 
  wall_height * wall_width - unpainted_height * unpainted_width = 135 := by
  sorry

end area_to_paint_l3944_394446


namespace quadratic_two_zeros_l3944_394458

-- Define a quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Theorem statement
theorem quadratic_two_zeros (a b c : ℝ) (h : a * c < 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  quadratic a b c x₁ = 0 ∧ 
  quadratic a b c x₂ = 0 ∧
  ∀ x : ℝ, quadratic a b c x = 0 → x = x₁ ∨ x = x₂ :=
sorry

end quadratic_two_zeros_l3944_394458


namespace instantaneous_velocity_at_4_hours_l3944_394452

-- Define the motion equation
def s (t : ℝ) : ℝ := t^3 + t^2

-- Define the instantaneous velocity function
def v (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_4_hours (h : 4 > 0) :
  v 4 = 56 := by sorry

end instantaneous_velocity_at_4_hours_l3944_394452


namespace max_area_rectangle_l3944_394410

theorem max_area_rectangle (perimeter : ℕ) (h : perimeter = 148) :
  ∃ (length width : ℕ),
    2 * (length + width) = perimeter ∧
    ∀ (l w : ℕ), 2 * (l + w) = perimeter → l * w ≤ length * width ∧
    length * width = 1369 := by
  sorry

end max_area_rectangle_l3944_394410


namespace cube_of_negative_product_l3944_394440

theorem cube_of_negative_product (a b : ℝ) : (-2 * a * b) ^ 3 = -8 * a ^ 3 * b ^ 3 := by
  sorry

end cube_of_negative_product_l3944_394440


namespace distance_in_one_hour_l3944_394463

/-- The number of seconds in one hour -/
def seconds_per_hour : ℕ := 3600

/-- The speed of the object in feet per second -/
def speed : ℕ := 3

/-- The distance traveled by an object moving at a constant speed for a given time -/
def distance_traveled (speed : ℕ) (time : ℕ) : ℕ := speed * time

/-- Theorem: An object traveling at 3 feet per second will cover 10800 feet in one hour -/
theorem distance_in_one_hour :
  distance_traveled speed seconds_per_hour = 10800 := by
  sorry

end distance_in_one_hour_l3944_394463


namespace sin_cos_pi_12_l3944_394431

theorem sin_cos_pi_12 : Real.sin (π / 12) * Real.cos (π / 12) = 1 / 4 := by
  sorry

end sin_cos_pi_12_l3944_394431


namespace greatest_integer_radius_for_circle_l3944_394492

theorem greatest_integer_radius_for_circle (A : ℝ) (h : A < 50 * Real.pi) :
  ∃ (r : ℕ), r * r * Real.pi ≤ A ∧ ∀ (s : ℕ), s * s * Real.pi ≤ A → s ≤ r ∧ r = 7 :=
by sorry

end greatest_integer_radius_for_circle_l3944_394492


namespace angle_measure_l3944_394425

/-- The measure of an angle in degrees, given that its supplement is four times its complement. -/
theorem angle_measure : ∃ x : ℝ, 
  (0 < x) ∧ (x < 180) ∧ 
  (180 - x = 4 * (90 - x)) ∧ 
  (x = 60) := by
  sorry

end angle_measure_l3944_394425


namespace gcf_of_75_and_90_l3944_394489

theorem gcf_of_75_and_90 : Nat.gcd 75 90 = 15 := by
  sorry

end gcf_of_75_and_90_l3944_394489


namespace partition_spread_bound_l3944_394483

/-- The number of partitions of a natural number -/
def P (n : ℕ) : ℕ := sorry

/-- The spread of a partition -/
def spread (partition : List ℕ) : ℕ := sorry

/-- The sum of spreads of all partitions of a natural number -/
def Q (n : ℕ) : ℕ := sorry

/-- Theorem: Q(n) ≤ √(2n) · P(n) for all natural numbers n -/
theorem partition_spread_bound (n : ℕ) : Q n ≤ Real.sqrt (2 * n) * P n := by sorry

end partition_spread_bound_l3944_394483


namespace bucket_capacity_l3944_394403

theorem bucket_capacity : ∀ (x : ℚ), 
  (13 * x = 91 * 6) → x = 42 := by
  sorry

end bucket_capacity_l3944_394403


namespace expression_evaluation_l3944_394416

theorem expression_evaluation (a : ℝ) (h : a^2 + 2*a - 1 = 0) :
  (((a^2 - 1) / (a^2 - 2*a + 1) - 1 / (1 - a)) / (1 / (a^2 - a))) = 1 := by
  sorry

end expression_evaluation_l3944_394416


namespace bread_in_pond_l3944_394488

theorem bread_in_pond (total_bread : ℕ) : 
  (total_bread / 2 : ℕ) + 13 + 7 + 30 = total_bread → total_bread = 100 := by
  sorry

end bread_in_pond_l3944_394488


namespace overlapping_circles_common_chord_l3944_394468

theorem overlapping_circles_common_chord 
  (r : ℝ) 
  (h1 : r = 12) 
  (h2 : r > 0) : 
  let d := r -- distance between centers
  let x := Real.sqrt (r^2 - (r/2)^2) -- half-length of common chord
  2 * x = 12 * Real.sqrt 3 := by sorry

end overlapping_circles_common_chord_l3944_394468


namespace circle_equation_from_line_intersection_l3944_394441

/-- Given a line in polar coordinates that intersects the polar axis, 
    this theorem proves the equation of a circle centered at the intersection point. -/
theorem circle_equation_from_line_intersection (ρ θ : ℝ) :
  (ρ * Real.cos (θ + π/4) = Real.sqrt 2) →
  ∃ C : ℝ × ℝ,
    (C.1 = 2 ∧ C.2 = 0) ∧
    (∀ (ρ' θ' : ℝ), (ρ' * Real.cos θ' - C.1)^2 + (ρ' * Real.sin θ' - C.2)^2 = 1 ↔
                     ρ'^2 - 4*ρ'*Real.cos θ' + 3 = 0) := by
  sorry


end circle_equation_from_line_intersection_l3944_394441


namespace budget_allocation_l3944_394461

def total_budget : ℝ := 40000000

def policing_percentage : ℝ := 0.35
def education_percentage : ℝ := 0.25
def healthcare_percentage : ℝ := 0.15

def remaining_budget : ℝ := total_budget * (1 - (policing_percentage + education_percentage + healthcare_percentage))

theorem budget_allocation :
  remaining_budget = 10000000 := by sorry

end budget_allocation_l3944_394461


namespace equation_system_proof_l3944_394491

theorem equation_system_proof (x y m : ℝ) 
  (eq1 : x + m = 4) 
  (eq2 : y - 3 = m) : 
  x + y = 7 := by
  sorry

end equation_system_proof_l3944_394491


namespace absolute_value_equals_cosine_roots_l3944_394499

theorem absolute_value_equals_cosine_roots :
  ∃! (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℝ, |x| = Real.cos x ↔ (x = a ∨ x = b ∨ x = c)) :=
sorry

end absolute_value_equals_cosine_roots_l3944_394499


namespace rectangle_longer_side_l3944_394465

/-- Given a square with two pairs of identical isosceles triangles cut off, leaving a rectangle,
    if the total area cut off is 250 m² and one side of the rectangle is 1.5 times the length of the other,
    then the length of the longer side of the rectangle is 7.5√5 meters. -/
theorem rectangle_longer_side (x y : ℝ) : 
  x^2 + y^2 = 250 →  -- Total area cut off
  x = y →            -- Isosceles triangles condition
  1.5 * y = max x (1.5 * y) →  -- One side is 1.5 times the other
  max x (1.5 * y) = 7.5 * Real.sqrt 5 :=
by sorry

end rectangle_longer_side_l3944_394465


namespace product_13_factor_l3944_394401

theorem product_13_factor (w : ℕ+) (h1 : w ≥ 468) 
  (h2 : ∃ (k : ℕ), 2^4 * 3^3 * k = 1452 * w) : 
  (∃ (m : ℕ), 13^1 * m = 1452 * w) ∧ 
  (∀ (n : ℕ), n > 1 → ¬(∃ (m : ℕ), 13^n * m = 1452 * w)) :=
sorry

end product_13_factor_l3944_394401


namespace school_population_problem_l3944_394443

theorem school_population_problem :
  ∀ (initial_girls initial_boys : ℕ),
    initial_boys = initial_girls + 51 →
    (100 * initial_girls) / (initial_girls + initial_boys) = 
      (100 * (initial_girls - 41)) / ((initial_girls - 41) + (initial_boys - 19)) + 4 →
    initial_girls = 187 ∧ initial_boys = 238 :=
by
  sorry

#check school_population_problem

end school_population_problem_l3944_394443


namespace alien_energy_conversion_l3944_394456

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

theorem alien_energy_conversion :
  base7ToBase10 [1, 2, 3] = 162 := by
  sorry

end alien_energy_conversion_l3944_394456


namespace kelly_initial_games_l3944_394495

/-- The number of games Kelly gave away -/
def games_given_away : ℕ := 91

/-- The number of games Kelly has left -/
def games_left : ℕ := 92

/-- The initial number of games Kelly had -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_initial_games : initial_games = 183 := by
  sorry

end kelly_initial_games_l3944_394495


namespace cost_of_graveling_specific_lawn_l3944_394404

/-- Calculates the cost of graveling two intersecting roads on a rectangular lawn. -/
def cost_of_graveling (lawn_length lawn_width road_width gravel_cost : ℝ) : ℝ :=
  let road_length_area := lawn_length * road_width
  let road_width_area := (lawn_width - road_width) * road_width
  let total_area := road_length_area + road_width_area
  total_area * gravel_cost

/-- The cost of graveling two intersecting roads on a 70m × 60m lawn with 10m wide roads at Rs. 3 per sq m is Rs. 3600. -/
theorem cost_of_graveling_specific_lawn :
  cost_of_graveling 70 60 10 3 = 3600 := by
  sorry

end cost_of_graveling_specific_lawn_l3944_394404


namespace hyperbola_standard_equation_l3944_394421

/-- Given a hyperbola with asymptotes y = ± 1/3 x and one focus at (0, 2√5),
    prove that its standard equation is y²/2 - x²/18 = 1 -/
theorem hyperbola_standard_equation 
  (asymptote : ℝ → ℝ)
  (focus : ℝ × ℝ)
  (h1 : ∀ x, asymptote x = 1/3 * x ∨ asymptote x = -1/3 * x)
  (h2 : focus = (0, 2 * Real.sqrt 5)) :
  ∃ f : ℝ × ℝ → ℝ, ∀ x y, f (x, y) = 0 ↔ y^2/2 - x^2/18 = 1 :=
sorry

end hyperbola_standard_equation_l3944_394421


namespace min_expression_l3944_394462

theorem min_expression (k : ℝ) (x y z t : ℝ) 
  (h1 : k ≥ 0) 
  (h2 : x > 0) (h3 : y > 0) (h4 : z > 0) (h5 : t > 0) 
  (h6 : x + y + z + t = k) : 
  x / (1 + y^2) + y / (1 + x^2) + z / (1 + t^2) + t / (1 + z^2) ≥ 4 * k / (4 + k^2) := by
  sorry

end min_expression_l3944_394462


namespace room_tile_coverage_l3944_394424

-- Define the room dimensions
def room_length : ℕ := 12
def room_width : ℕ := 20

-- Define the number of tiles
def num_tiles : ℕ := 40

-- Define the size of each tile
def tile_size : ℕ := 1

-- Theorem to prove
theorem room_tile_coverage : 
  (num_tiles : ℚ) / (room_length * room_width) = 1 / 6 := by
  sorry

end room_tile_coverage_l3944_394424


namespace geometric_sequence_formula_l3944_394444

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) = q * a n

theorem geometric_sequence_formula
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n : ℕ, n ≥ 1 → a n > 0)
  (h_first : a 1 = 1)
  (h_sum : a 1 + a 2 + a 3 = 7) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^(n - 1) :=
sorry

end geometric_sequence_formula_l3944_394444


namespace roger_apps_deletion_l3944_394436

/-- The number of apps Roger must delete for optimal phone function -/
def apps_to_delete (max_apps : ℕ) (recommended_apps : ℕ) : ℕ :=
  2 * recommended_apps - max_apps

/-- Theorem stating the number of apps Roger must delete -/
theorem roger_apps_deletion :
  apps_to_delete 50 35 = 20 := by
  sorry

end roger_apps_deletion_l3944_394436


namespace eight_additional_people_needed_l3944_394433

/-- The number of additional people needed to mow a lawn and trim its edges -/
def additional_people_needed (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ) : ℕ :=
  let total_person_hours := initial_people * initial_time
  let people_mowing := total_person_hours / new_time
  let people_trimming := people_mowing / 3
  let total_people_needed := people_mowing + people_trimming
  total_people_needed - initial_people

/-- Theorem stating that 8 additional people are needed under the given conditions -/
theorem eight_additional_people_needed :
  additional_people_needed 8 3 2 = 8 := by
  sorry

end eight_additional_people_needed_l3944_394433


namespace Q_sufficient_not_necessary_for_P_l3944_394413

-- Define the property P(x) as x^2 - 1 > 0
def P (x : ℝ) : Prop := x^2 - 1 > 0

-- Define the condition Q(x) as x < -1
def Q (x : ℝ) : Prop := x < -1

-- Theorem stating that Q is sufficient but not necessary for P
theorem Q_sufficient_not_necessary_for_P :
  (∀ x : ℝ, Q x → P x) ∧ ¬(∀ x : ℝ, P x → Q x) :=
sorry

end Q_sufficient_not_necessary_for_P_l3944_394413


namespace no_gcd_inverting_function_l3944_394422

theorem no_gcd_inverting_function :
  ¬ (∃ f : ℕ+ → ℕ+, ∀ a b : ℕ+, Nat.gcd a.val b.val = 1 ↔ Nat.gcd (f a).val (f b).val > 1) :=
sorry

end no_gcd_inverting_function_l3944_394422


namespace shirt_double_discount_l3944_394412

theorem shirt_double_discount (original_price : ℝ) (discount_rate : ℝ) : 
  original_price = 32 → 
  discount_rate = 0.25 → 
  (1 - discount_rate) * (1 - discount_rate) * original_price = 18 := by
sorry

end shirt_double_discount_l3944_394412


namespace train_speed_l3944_394469

theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 500) (h2 : crossing_time = 10) :
  train_length / crossing_time = 50 := by
  sorry

end train_speed_l3944_394469


namespace committee_selection_l3944_394405

theorem committee_selection (boys girls : ℕ) (h1 : boys = 21) (h2 : girls = 14) :
  (Nat.choose (boys + girls) 4) - (Nat.choose boys 4 + Nat.choose girls 4) = 45374 :=
sorry

end committee_selection_l3944_394405


namespace trapezoid_height_l3944_394426

/-- A trapezoid with given side lengths has a height of 12 cm -/
theorem trapezoid_height (a b c d : ℝ) (ha : a = 25) (hb : b = 4) (hc : c = 20) (hd : d = 13) :
  ∃ h : ℝ, h = 12 ∧ h^2 = c^2 - ((a - b) / 2)^2 ∧ h^2 = d^2 - ((a - b) / 2)^2 := by
  sorry


end trapezoid_height_l3944_394426


namespace triangle_angle_relation_l3944_394480

/-- In a right-angled triangle ABC, given the measures of its angles, prove the relationship between x and y. -/
theorem triangle_angle_relation (x y : ℝ) : 
  x > 0 → y > 0 → x + 3 * y = 90 → x + y = 90 - 2 * y := by
  sorry

end triangle_angle_relation_l3944_394480


namespace average_value_function_m_range_l3944_394467

/-- A function is an average value function on [a, b] if there exists x₀ ∈ (a, b) such that f(x₀) = (f(b) - f(a)) / (b - a) -/
def IsAverageValueFunction (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ : ℝ, a < x₀ ∧ x₀ < b ∧ f x₀ = (f b - f a) / (b - a)

/-- The quadratic function f(x) = x² - mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x - 1

theorem average_value_function_m_range :
  ∀ m : ℝ, IsAverageValueFunction (f m) (-1) 1 ↔ 0 < m ∧ m < 2 := by
  sorry

end average_value_function_m_range_l3944_394467


namespace task_completion_time_l3944_394457

/-- Proves that the total time to complete a task is 8 days given the specified conditions -/
theorem task_completion_time 
  (john_rate : ℚ) 
  (jane_rate : ℚ) 
  (jane_leave_before_end : ℕ) :
  john_rate = 1/16 →
  jane_rate = 1/12 →
  jane_leave_before_end = 5 →
  ∃ (total_days : ℕ), total_days = 8 ∧ 
    (john_rate + jane_rate) * (total_days - jane_leave_before_end : ℚ) + 
    john_rate * (jane_leave_before_end : ℚ) = 1 :=
by sorry

end task_completion_time_l3944_394457


namespace meaningful_fraction_l3944_394498

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (2*x + 1)/(x - 2)) ↔ x ≠ 2 :=
by sorry

end meaningful_fraction_l3944_394498


namespace circle_in_diamond_l3944_394460

-- Define the sets M and N
def M (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = a^2}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 + p.2| + |p.1 - p.2| ≤ 2}

-- State the theorem
theorem circle_in_diamond (a : ℝ) (h : a > 0) :
  M a ⊆ N ↔ a ≤ 1 := by sorry

end circle_in_diamond_l3944_394460


namespace sin_alpha_value_l3944_394490

theorem sin_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) : 
  Real.sin α = Real.sqrt 5 / 3 := by
sorry

end sin_alpha_value_l3944_394490


namespace f_symmetry_solutions_l3944_394402

def f_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = x^3 + 6

theorem f_symmetry_solutions (f : ℝ → ℝ) (hf : f_condition f) :
  {x : ℝ | x ≠ 0 ∧ f x = f (-x)} = {(1/2)^(1/6), -(1/2)^(1/6)} := by
  sorry

end f_symmetry_solutions_l3944_394402


namespace books_read_total_l3944_394481

/-- The number of books read by Megan, Kelcie, and Greg -/
def total_books (megan_books kelcie_books greg_books : ℕ) : ℕ :=
  megan_books + kelcie_books + greg_books

/-- Theorem stating the total number of books read by Megan, Kelcie, and Greg -/
theorem books_read_total :
  ∃ (megan_books kelcie_books greg_books : ℕ),
    megan_books = 32 ∧
    kelcie_books = megan_books / 4 ∧
    greg_books = 2 * kelcie_books + 9 ∧
    total_books megan_books kelcie_books greg_books = 65 :=
by
  sorry


end books_read_total_l3944_394481


namespace parallel_vectors_x_value_l3944_394459

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem statement
theorem parallel_vectors_x_value :
  parallel a (b x) → x = 6 := by sorry

end parallel_vectors_x_value_l3944_394459


namespace vector_scalar_add_l3944_394485

theorem vector_scalar_add : 
  3 • !![5, -3] + !![(-4), 9] = !![11, 0] := by sorry

end vector_scalar_add_l3944_394485


namespace housing_price_growth_equation_l3944_394494

/-- Represents the annual growth rate of housing prices -/
def average_annual_growth_rate : ℝ := sorry

/-- The initial housing price in 2018 (yuan per square meter) -/
def initial_price : ℝ := 5000

/-- The final housing price in 2020 (yuan per square meter) -/
def final_price : ℝ := 6500

/-- The number of years of growth -/
def years_of_growth : ℕ := 2

/-- Theorem stating that the given equation correctly represents the housing price growth -/
theorem housing_price_growth_equation :
  initial_price * (1 + average_annual_growth_rate) ^ years_of_growth = final_price :=
sorry

end housing_price_growth_equation_l3944_394494


namespace university_theater_ticket_sales_l3944_394455

theorem university_theater_ticket_sales 
  (total_tickets : ℕ) 
  (adult_price senior_price : ℕ) 
  (total_receipts : ℕ) 
  (h1 : total_tickets = 510)
  (h2 : adult_price = 21)
  (h3 : senior_price = 15)
  (h4 : total_receipts = 8748) :
  ∃ (adult_tickets senior_tickets : ℕ),
    adult_tickets + senior_tickets = total_tickets ∧
    adult_price * adult_tickets + senior_price * senior_tickets = total_receipts ∧
    senior_tickets = 327 :=
by sorry

end university_theater_ticket_sales_l3944_394455


namespace sin_negative_three_pi_halves_l3944_394497

theorem sin_negative_three_pi_halves : Real.sin (-3 * π / 2) = 1 := by
  sorry

end sin_negative_three_pi_halves_l3944_394497


namespace tina_postcard_price_l3944_394419

/-- Proves that the price per postcard is $5, given the conditions of Tina's postcard sales. -/
theorem tina_postcard_price :
  let postcards_per_day : ℕ := 30
  let days_sold : ℕ := 6
  let total_earned : ℕ := 900
  let total_postcards : ℕ := postcards_per_day * days_sold
  let price_per_postcard : ℚ := total_earned / total_postcards
  price_per_postcard = 5 := by sorry

end tina_postcard_price_l3944_394419


namespace subtraction_problem_l3944_394449

theorem subtraction_problem :
  572 - 275 = 297 := by
  sorry

end subtraction_problem_l3944_394449


namespace complex_equality_implies_a_equals_three_l3944_394430

theorem complex_equality_implies_a_equals_three (a : ℝ) : 
  let z : ℂ := (a + Complex.I) / (2 - Complex.I)
  (z.re = z.im) → a = 3 := by
  sorry

end complex_equality_implies_a_equals_three_l3944_394430


namespace set_A_characterization_union_A_B_characterization_l3944_394406

def A : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 2*x) / Real.log 10}
def B : Set ℝ := {y | ∃ x, y = Real.sqrt x + 1}

theorem set_A_characterization : A = {x | x < 0 ∨ x > 2} := by sorry

theorem union_A_B_characterization : A ∪ B = {x | x < 0 ∨ x ≥ 1} := by sorry

end set_A_characterization_union_A_B_characterization_l3944_394406


namespace union_of_A_and_B_l3944_394466

def A : Set ℝ := {x | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 ≤ x ∧ x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -3 ≤ x ∧ x ≤ 4} := by sorry

end union_of_A_and_B_l3944_394466


namespace apples_ratio_l3944_394478

def apples_problem (tuesday wednesday thursday : ℕ) : Prop :=
  tuesday = 4 ∧
  thursday = tuesday / 2 ∧
  tuesday + wednesday + thursday = 14

theorem apples_ratio : 
  ∀ tuesday wednesday thursday : ℕ,
  apples_problem tuesday wednesday thursday →
  wednesday = 2 * tuesday :=
by sorry

end apples_ratio_l3944_394478


namespace three_solutions_condition_l3944_394414

theorem three_solutions_condition (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    a * x₁ = |Real.log x₁| ∧ a * x₂ = |Real.log x₂| ∧ a * x₃ = |Real.log x₃|) ∧
  (∀ x : ℝ, a * x ≥ 0) ↔ 
  (-1 / Real.exp 1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1 / Real.exp 1) := by
  sorry

end three_solutions_condition_l3944_394414


namespace parallelogram_reflection_l3944_394420

-- Define the reflection across x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Define the reflection across x = 3
def reflect_x3 (p : ℝ × ℝ) : ℝ × ℝ := (6 - p.1, p.2)

-- Define the composition of both reflections
def double_reflect (p : ℝ × ℝ) : ℝ × ℝ := reflect_x3 (reflect_x p)

theorem parallelogram_reflection :
  double_reflect (4, 1) = (2, -1) := by
  sorry

end parallelogram_reflection_l3944_394420


namespace gopal_krishan_ratio_l3944_394473

/-- The ratio of money between Gopal and Krishan given the conditions -/
theorem gopal_krishan_ratio :
  ∀ (ram gopal krishan : ℕ),
  ram = 735 →
  krishan = 4335 →
  7 * gopal = 17 * ram →
  (gopal : ℚ) / krishan = 1785 / 4335 :=
by sorry

end gopal_krishan_ratio_l3944_394473


namespace trig_identity_l3944_394486

theorem trig_identity : 
  Real.sqrt (1 + Real.sin 6) + Real.sqrt (1 - Real.sin 6) = -2 * Real.cos 3 := by
  sorry

end trig_identity_l3944_394486
