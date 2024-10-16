import Mathlib

namespace NUMINAMATH_CALUDE_exists_integers_for_n_squared_and_cubed_l3725_372511

theorem exists_integers_for_n_squared_and_cubed (n : ℕ) : 
  (∃ a b : ℤ, n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 := by
sorry

end NUMINAMATH_CALUDE_exists_integers_for_n_squared_and_cubed_l3725_372511


namespace NUMINAMATH_CALUDE_hexagon_extension_l3725_372563

/-- Regular hexagon ABCDEF with side length 3 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Point Y is on the extension of AB such that AY = 2AB -/
def extend_side (h : RegularHexagon) (Y : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Y = h.A + t • (h.B - h.A) ∧ dist h.A Y = 2 * h.side_length

/-- The length of FY -/
def FY_length (h : RegularHexagon) (Y : ℝ × ℝ) : ℝ :=
  dist h.F Y

theorem hexagon_extension (h : RegularHexagon) (Y : ℝ × ℝ) 
  (ext : extend_side h Y) : FY_length h Y = 15 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_extension_l3725_372563


namespace NUMINAMATH_CALUDE_circle_whisper_game_l3725_372548

theorem circle_whisper_game (a b c d e f : ℕ) : 
  a + b + c + d + e + f = 18 →
  a + b = 16 →
  b + c = 12 →
  e + f = 8 →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_circle_whisper_game_l3725_372548


namespace NUMINAMATH_CALUDE_cricket_matches_count_l3725_372523

theorem cricket_matches_count (total_average : ℝ) (first_four_average : ℝ) (last_three_average : ℝ) :
  total_average = 56 →
  first_four_average = 46 →
  last_three_average = 69.33333333333333 →
  ∃ (n : ℕ), n = 7 ∧ n * total_average = 4 * first_four_average + 3 * last_three_average :=
by sorry

end NUMINAMATH_CALUDE_cricket_matches_count_l3725_372523


namespace NUMINAMATH_CALUDE_inequality_range_l3725_372522

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (3 : ℝ)^(x^2 - 2*a*x) > (1/3 : ℝ)^(x + 1)) → 
  -1/2 < a ∧ a < 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l3725_372522


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l3725_372591

theorem min_value_squared_sum (x y : ℝ) : (x + y)^2 + (x - 1/y)^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l3725_372591


namespace NUMINAMATH_CALUDE_initial_oranges_count_l3725_372521

/-- Proves that the initial number of oranges in a bin was 50, given the described changes and final count. -/
theorem initial_oranges_count (initial thrown_away added final : ℕ) 
  (h1 : thrown_away = 40)
  (h2 : added = 24)
  (h3 : final = 34)
  (h4 : initial - thrown_away + added = final) : initial = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l3725_372521


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3725_372565

theorem complex_number_quadrant (z : ℂ) (h : z * (2 - Complex.I) = 2 + Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3725_372565


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3725_372547

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} := by sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l3725_372547


namespace NUMINAMATH_CALUDE_arithmetic_proof_l3725_372525

theorem arithmetic_proof : 4 * (8 - 3) - 2 * 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_proof_l3725_372525


namespace NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l3725_372537

/-- A linear function defined by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  yIntercept : ℝ

/-- The four quadrants of the Cartesian plane -/
inductive Quadrant
  | first
  | second
  | third
  | fourth

/-- Determines if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  sorry

/-- The specific linear function y = 2x + 1 -/
def f : LinearFunction :=
  { slope := 2, yIntercept := 1 }

theorem linear_function_not_in_fourth_quadrant :
  ¬ passesThrough f Quadrant.fourth :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_fourth_quadrant_l3725_372537


namespace NUMINAMATH_CALUDE_fraction_sum_l3725_372582

theorem fraction_sum : 2/5 + 4/50 + 3/500 + 8/5000 = 0.4876 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3725_372582


namespace NUMINAMATH_CALUDE_unique_c_value_l3725_372507

/-- A polynomial has exactly one real root if and only if its discriminant is zero -/
def has_one_real_root (b c : ℝ) : Prop :=
  b ^ 2 = 4 * c

/-- The product of all possible values of c satisfying the conditions -/
def product_of_c_values (b c : ℝ) : ℝ :=
  -- This is a placeholder; the actual computation would be more complex
  1

theorem unique_c_value (b c : ℝ) 
  (h1 : has_one_real_root b c)
  (h2 : b = c^2 + 1) :
  product_of_c_values b c = 1 := by
  sorry

#check unique_c_value

end NUMINAMATH_CALUDE_unique_c_value_l3725_372507


namespace NUMINAMATH_CALUDE_counterexample_exists_l3725_372598

theorem counterexample_exists : ∃ (a b : ℝ), a^2 > b^2 ∧ a ≤ b := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l3725_372598


namespace NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l3725_372546

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_plane_implies_parallel 
  (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_to_plane_implies_parallel_l3725_372546


namespace NUMINAMATH_CALUDE_road_travel_cost_l3725_372579

/-- The cost of traveling two intersecting roads on a rectangular lawn -/
theorem road_travel_cost (lawn_length lawn_width road_width travel_cost_per_sqm : ℕ) : 
  lawn_length = 90 ∧ 
  lawn_width = 60 ∧ 
  road_width = 10 ∧ 
  travel_cost_per_sqm = 3 →
  (road_width * lawn_width + road_width * lawn_length - road_width * road_width) * travel_cost_per_sqm = 4200 :=
by sorry

end NUMINAMATH_CALUDE_road_travel_cost_l3725_372579


namespace NUMINAMATH_CALUDE_largest_prime_factor_largest_prime_factor_of_expression_l3725_372555

theorem largest_prime_factor (n : ℕ) : ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Nat.Prime q → q ∣ n → q ≤ p :=
  sorry

theorem largest_prime_factor_of_expression : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (18^3 + 12^4 - 6^5) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (18^3 + 12^4 - 6^5) → q ≤ p ∧ p = 23 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_largest_prime_factor_of_expression_l3725_372555


namespace NUMINAMATH_CALUDE_one_in_M_l3725_372506

def M : Set ℕ := {1, 2, 3}

theorem one_in_M : 1 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_one_in_M_l3725_372506


namespace NUMINAMATH_CALUDE_percent_relation_l3725_372572

theorem percent_relation (x y z : ℝ) (p : ℝ) 
  (h1 : y = 0.75 * x) 
  (h2 : z = 2 * x) 
  (h3 : p / 100 * z = 1.2 * y) : 
  p = 45 := by sorry

end NUMINAMATH_CALUDE_percent_relation_l3725_372572


namespace NUMINAMATH_CALUDE_triangle_angle_45_degrees_l3725_372587

theorem triangle_angle_45_degrees (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- angles are positive
  A + B + C = 180 → -- sum of angles in a triangle is 180°
  B + C = 3 * A → -- given condition
  A = 45 ∨ B = 45 ∨ C = 45 := by sorry

end NUMINAMATH_CALUDE_triangle_angle_45_degrees_l3725_372587


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3725_372584

-- Define the solution set
def solution_set : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Define the inequality
def inequality (x : ℝ) : Prop := (2 - x) / (x + 1) ≥ 0

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3725_372584


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l3725_372503

theorem largest_n_divisibility : 
  ∀ n : ℕ, n > 1098 → ¬(n + 11 ∣ n^3 + 101) ∧ (1098 + 11 ∣ 1098^3 + 101) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l3725_372503


namespace NUMINAMATH_CALUDE_binomial_18_6_l3725_372592

theorem binomial_18_6 : Nat.choose 18 6 = 4767 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_6_l3725_372592


namespace NUMINAMATH_CALUDE_dog_grooming_time_l3725_372570

theorem dog_grooming_time :
  let short_hair_time : ℕ := 10 -- Time to dry a short-haired dog in minutes
  let full_hair_time : ℕ := 2 * short_hair_time -- Time to dry a full-haired dog
  let short_hair_count : ℕ := 6 -- Number of short-haired dogs
  let full_hair_count : ℕ := 9 -- Number of full-haired dogs
  let total_time : ℕ := short_hair_time * short_hair_count + full_hair_time * full_hair_count
  total_time / 60 = 4 -- Total time in hours
  := by sorry

end NUMINAMATH_CALUDE_dog_grooming_time_l3725_372570


namespace NUMINAMATH_CALUDE_girls_ran_nine_miles_l3725_372590

/-- The number of laps run by boys -/
def boys_laps : ℕ := 34

/-- The additional laps run by girls compared to boys -/
def additional_girls_laps : ℕ := 20

/-- The fraction of a mile that one lap represents -/
def lap_mile_fraction : ℚ := 1 / 6

/-- The total number of laps run by girls -/
def girls_laps : ℕ := boys_laps + additional_girls_laps

/-- The number of miles run by girls -/
def girls_miles : ℚ := girls_laps * lap_mile_fraction

theorem girls_ran_nine_miles : girls_miles = 9 := by
  sorry

end NUMINAMATH_CALUDE_girls_ran_nine_miles_l3725_372590


namespace NUMINAMATH_CALUDE_triangle_angle_C_l3725_372515

noncomputable def f (x φ : Real) : Real :=
  2 * Real.sin x * (Real.cos (φ / 2))^2 + Real.cos x * Real.sin φ - Real.sin x

theorem triangle_angle_C (φ A B C : Real) (a b c : Real) :
  0 < φ ∧ φ < Real.pi ∧
  (∀ x, f x φ ≥ f Real.pi φ) ∧
  Real.cos (2 * C) - Real.cos (2 * A) = 2 * Real.sin (Real.pi / 3 + C) * Real.sin (Real.pi / 3 - C) ∧
  a = 1 ∧
  b = Real.sqrt 2 ∧
  f A φ = Real.sqrt 3 / 2 ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧
  a / Real.sin A = c / Real.sin C
  →
  C = 7 * Real.pi / 12 ∨ C = Real.pi / 12 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l3725_372515


namespace NUMINAMATH_CALUDE_least_n_with_zero_in_factorization_l3725_372543

/-- A function that checks if a positive integer contains the digit 0 -/
def containsZero (n : ℕ+) : Prop :=
  ∃ (k : ℕ), n.val = 10 * k ∨ n.val % 10 = 0

/-- A function that checks if all factorizations of 10^n contain a zero -/
def allFactorizationsContainZero (n : ℕ) : Prop :=
  ∀ (a b : ℕ+), a * b = 10^n → (containsZero a ∨ containsZero b)

/-- The main theorem stating that 8 is the least positive integer satisfying the condition -/
theorem least_n_with_zero_in_factorization :
  (allFactorizationsContainZero 8) ∧
  (∀ m : ℕ, m < 8 → ¬(allFactorizationsContainZero m)) :=
sorry

end NUMINAMATH_CALUDE_least_n_with_zero_in_factorization_l3725_372543


namespace NUMINAMATH_CALUDE_sum_of_factors_72_l3725_372508

/-- Sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

/-- The theorem stating that the sum of positive factors of 72 is 195 -/
theorem sum_of_factors_72 : sum_of_factors 72 = 195 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_72_l3725_372508


namespace NUMINAMATH_CALUDE_even_function_implies_a_value_l3725_372568

def f (x a : ℝ) : ℝ := (x + 1) * (2 * x + 3 * a)

theorem even_function_implies_a_value :
  (∀ x, f x a = f (-x) a) → a = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_value_l3725_372568


namespace NUMINAMATH_CALUDE_solve_water_problem_l3725_372504

def water_problem (initial_water evaporated_water rain_duration rain_rate final_water : ℝ) : Prop :=
  let water_after_evaporation := initial_water - evaporated_water
  let rainwater_added := (rain_duration / 10) * rain_rate
  let water_after_rain := water_after_evaporation + rainwater_added
  let water_drained := water_after_rain - final_water
  water_drained = 3500

theorem solve_water_problem :
  water_problem 6000 2000 30 350 1550 := by
  sorry

end NUMINAMATH_CALUDE_solve_water_problem_l3725_372504


namespace NUMINAMATH_CALUDE_second_derivative_zero_not_implies_extreme_point_l3725_372532

open Real

-- Define the function f(x) = x^3
def f (x : ℝ) := x^3

-- Define what it means for a point to be an extreme point
def is_extreme_point (f : ℝ → ℝ) (x₀ : ℝ) :=
  ∀ x, |x - x₀| < 1 → f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem second_derivative_zero_not_implies_extreme_point :
  ∃ x₀ : ℝ, (deriv (deriv f)) x₀ = 0 ∧ ¬(is_extreme_point f x₀) := by
  sorry


end NUMINAMATH_CALUDE_second_derivative_zero_not_implies_extreme_point_l3725_372532


namespace NUMINAMATH_CALUDE_function_properties_l3725_372559

-- Define the function f
def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

-- State the theorem
theorem function_properties (a b c : ℝ) :
  (∀ x < 0, ∀ y < x, f a b c x < f a b c y) →  -- f is decreasing on (-∞, 0)
  (∀ x ∈ Set.Ioo 0 1, ∀ y ∈ Set.Ioo 0 1, x < y → f a b c x < f a b c y) →  -- f is increasing on (0, 1)
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f a b c x = 0 ∧ f a b c y = 0 ∧ f a b c z = 0) →  -- f has three real roots
  f a b c 1 = 0 →  -- 1 is a root of f
  b = 0 ∧ f a b c 2 > -5/2 ∧ 3/2 < a ∧ a < 2 * Real.sqrt 2 - 1 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3725_372559


namespace NUMINAMATH_CALUDE_return_speed_l3725_372597

/-- Given two towns and a person's travel speeds, calculate the return speed -/
theorem return_speed (d : ℝ) (v_xy v_total : ℝ) (h1 : v_xy = 54) (h2 : v_total = 43.2) :
  let v_yx := 2 * v_total * v_xy / (2 * v_xy - v_total)
  v_yx = 36 := by sorry

end NUMINAMATH_CALUDE_return_speed_l3725_372597


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3725_372526

/-- Given a geometric sequence {a_n} where a_2010 = 8a_2007, prove that the common ratio q is 2 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (h : a 2010 = 8 * a 2007) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3725_372526


namespace NUMINAMATH_CALUDE_fresh_fruit_amount_l3725_372566

-- Define the total amount of fruit sold
def total_fruit : ℕ := 9792

-- Define the amount of frozen fruit sold
def frozen_fruit : ℕ := 3513

-- Define the amount of fresh fruit sold
def fresh_fruit : ℕ := total_fruit - frozen_fruit

-- Theorem to prove
theorem fresh_fruit_amount : fresh_fruit = 6279 := by
  sorry

end NUMINAMATH_CALUDE_fresh_fruit_amount_l3725_372566


namespace NUMINAMATH_CALUDE_count_D_eq_3_is_18_l3725_372533

/-- D(n) is the number of pairs of different adjacent digits in the binary representation of n -/
def D (n : ℕ) : ℕ := sorry

/-- The count of positive integers n ≤ 200 for which D(n) = 3 -/
def count_D_eq_3 : ℕ := sorry

theorem count_D_eq_3_is_18 : count_D_eq_3 = 18 := by sorry

end NUMINAMATH_CALUDE_count_D_eq_3_is_18_l3725_372533


namespace NUMINAMATH_CALUDE_download_calculation_l3725_372550

/-- Calculates the number of songs that can be downloaded given internet speed, song size, and time. -/
def songs_downloaded (internet_speed : ℕ) (song_size : ℕ) (time_minutes : ℕ) : ℕ :=
  (internet_speed * 60 * time_minutes) / song_size

/-- Theorem stating that with given conditions, 7200 songs can be downloaded. -/
theorem download_calculation :
  let internet_speed : ℕ := 20  -- MBps
  let song_size : ℕ := 5        -- MB
  let time_minutes : ℕ := 30    -- half an hour
  songs_downloaded internet_speed song_size time_minutes = 7200 := by
sorry

end NUMINAMATH_CALUDE_download_calculation_l3725_372550


namespace NUMINAMATH_CALUDE_arithmetic_puzzle_2016_l3725_372509

/-- Represents a basic arithmetic operation --/
inductive Operation
  | Add
  | Subtract
  | Multiply
  | Divide

/-- Represents an arithmetic expression --/
inductive Expr
  | Num (n : ℕ)
  | Op (op : Operation) (e1 e2 : Expr)
  | Paren (e : Expr)

/-- Evaluates an arithmetic expression --/
def eval : Expr → ℚ
  | Expr.Num n => n
  | Expr.Op Operation.Add e1 e2 => eval e1 + eval e2
  | Expr.Op Operation.Subtract e1 e2 => eval e1 - eval e2
  | Expr.Op Operation.Multiply e1 e2 => eval e1 * eval e2
  | Expr.Op Operation.Divide e1 e2 => eval e1 / eval e2
  | Expr.Paren e => eval e

/-- Checks if an expression uses digits 1 through 9 in sequence --/
def usesDigitsInSequence : Expr → Bool
  | _ => sorry  -- Implementation omitted for brevity

theorem arithmetic_puzzle_2016 :
  ∃ (e : Expr), usesDigitsInSequence e ∧ eval e = 2016 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_puzzle_2016_l3725_372509


namespace NUMINAMATH_CALUDE_x_difference_l3725_372586

theorem x_difference (x₁ x₂ : ℝ) : 
  ((x₁ + 3)^2 / (3*x₁ + 65) = 2) →
  ((x₂ + 3)^2 / (3*x₂ + 65) = 2) →
  x₁ ≠ x₂ →
  |x₁ - x₂| = 22 := by
sorry

end NUMINAMATH_CALUDE_x_difference_l3725_372586


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l3725_372567

/-- Represents a rectangle with side lengths x and y -/
structure Rectangle where
  x : ℝ
  y : ℝ

/-- Represents the configuration of rectangles and squares -/
structure CrossConfiguration where
  inner_square_side : ℝ
  outer_square_side : ℝ
  rectangle : Rectangle

/-- The cross configuration satisfies the given conditions -/
def valid_configuration (c : CrossConfiguration) : Prop :=
  c.outer_square_side = 3 * c.inner_square_side ∧
  c.rectangle.y = c.inner_square_side ∧
  c.rectangle.x + c.inner_square_side = c.outer_square_side

/-- The theorem stating the ratio of rectangle sides -/
theorem rectangle_side_ratio (c : CrossConfiguration) 
  (h : valid_configuration c) : 
  c.rectangle.x / c.rectangle.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l3725_372567


namespace NUMINAMATH_CALUDE_circle_radius_reduction_l3725_372594

theorem circle_radius_reduction (r : ℝ) (h : r > 0) :
  let new_area_ratio := 1 - 0.18999999999999993
  let new_radius_ratio := 1 - 0.1
  (new_radius_ratio * r) ^ 2 = new_area_ratio * r ^ 2 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_reduction_l3725_372594


namespace NUMINAMATH_CALUDE_quadratic_roots_differ_by_two_l3725_372512

/-- For a quadratic equation ax^2 + bx + c = 0 where a ≠ 0, 
    if the roots of the equation differ by 2, then c = (b^2 / (4a)) - a -/
theorem quadratic_roots_differ_by_two (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x y : ℝ, x - y = 2 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  c = (b^2 / (4 * a)) - a := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_differ_by_two_l3725_372512


namespace NUMINAMATH_CALUDE_concept_laws_theorem_l3725_372540

/-- Probability of M laws being included in the Concept -/
def prob_M_laws_included (K N M : ℕ) (p : ℝ) : ℝ :=
  Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- Expected number of laws included in the Concept -/
def expected_laws_included (K N : ℕ) (p : ℝ) : ℝ :=
  K * (1 - (1 - p)^N)

/-- Theorem stating the probability of M laws being included and the expected number of laws -/
theorem concept_laws_theorem (K N M : ℕ) (p : ℝ) 
    (hK : K > 0) (hN : N > 0) (hM : M ≤ K) (hp : 0 ≤ p ∧ p ≤ 1) :
  prob_M_laws_included K N M p = Nat.choose K M * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M) ∧
  expected_laws_included K N p = K * (1 - (1 - p)^N) := by
  sorry

end NUMINAMATH_CALUDE_concept_laws_theorem_l3725_372540


namespace NUMINAMATH_CALUDE_floor_sufficiency_not_necessity_l3725_372561

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Theorem statement
theorem floor_sufficiency_not_necessity :
  (∀ x y : ℝ, floor x = floor y → |x - y| < 1) ∧
  (∃ x y : ℝ, |x - y| < 1 ∧ floor x ≠ floor y) :=
by sorry

end NUMINAMATH_CALUDE_floor_sufficiency_not_necessity_l3725_372561


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3725_372535

theorem simplify_square_roots : Real.sqrt 49 - Real.sqrt 256 = -9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3725_372535


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3725_372501

theorem sum_of_x_and_y (x y : ℤ) : x - y = 200 → y = 235 → x + y = 670 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3725_372501


namespace NUMINAMATH_CALUDE_investment_growth_l3725_372575

theorem investment_growth (initial_investment : ℝ) (first_year_loss_rate : ℝ) (second_year_gain_rate : ℝ) :
  initial_investment = 150 →
  first_year_loss_rate = 0.1 →
  second_year_gain_rate = 0.25 →
  let first_year_amount := initial_investment * (1 - first_year_loss_rate)
  let final_amount := first_year_amount * (1 + second_year_gain_rate)
  let overall_gain_rate := (final_amount - initial_investment) / initial_investment
  overall_gain_rate = 0.125 := by
  sorry

end NUMINAMATH_CALUDE_investment_growth_l3725_372575


namespace NUMINAMATH_CALUDE_no_additional_savings_when_purchasing_together_l3725_372576

/-- Represents the store's window offer -/
structure WindowOffer where
  price : ℕ  -- Price per window
  buy : ℕ    -- Number of windows to buy
  free : ℕ   -- Number of free windows

/-- Calculates the cost for a given number of windows under the offer -/
def calculateCost (offer : WindowOffer) (windowsNeeded : ℕ) : ℕ :=
  let fullSets := windowsNeeded / (offer.buy + offer.free)
  let remainingWindows := windowsNeeded % (offer.buy + offer.free)
  fullSets * (offer.price * offer.buy) + min remainingWindows offer.buy * offer.price

/-- Theorem stating that there's no additional savings when purchasing together -/
theorem no_additional_savings_when_purchasing_together 
  (offer : WindowOffer)
  (daveWindows : ℕ)
  (dougWindows : ℕ) :
  offer.price = 150 ∧ 
  offer.buy = 6 ∧ 
  offer.free = 2 ∧
  daveWindows = 9 ∧
  dougWindows = 10 →
  (calculateCost offer daveWindows + calculateCost offer dougWindows) - 
  calculateCost offer (daveWindows + dougWindows) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_additional_savings_when_purchasing_together_l3725_372576


namespace NUMINAMATH_CALUDE_student_pet_difference_l3725_372516

/-- Represents a fourth-grade classroom -/
structure Classroom where
  students : ℕ
  rabbits : ℕ
  birds : ℕ

/-- The number of fourth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- A fourth-grade classroom at Green Park Elementary -/
def green_park_classroom : Classroom := {
  students := 22,
  rabbits := 3,
  birds := 2
}

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * green_park_classroom.students

/-- The total number of pets (rabbits and birds) in all classrooms -/
def total_pets : ℕ := num_classrooms * (green_park_classroom.rabbits + green_park_classroom.birds)

/-- Theorem: The difference between the total number of students and the total number of pets is 85 -/
theorem student_pet_difference : total_students - total_pets = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_pet_difference_l3725_372516


namespace NUMINAMATH_CALUDE_smallest_lcm_five_digit_gcd_five_l3725_372569

/-- Given positive 5-digit integers m and n with gcd(m,n) = 5, 
    the smallest possible value for lcm[m,n] is 20030010 -/
theorem smallest_lcm_five_digit_gcd_five (m n : ℕ) 
  (h1 : 10000 ≤ m ∧ m < 100000) 
  (h2 : 10000 ≤ n ∧ n < 100000) 
  (h3 : Nat.gcd m n = 5) : 
  Nat.lcm m n ≥ 20030010 ∧ ∃ (a b : ℕ), 
    10000 ≤ a ∧ a < 100000 ∧ 
    10000 ≤ b ∧ b < 100000 ∧ 
    Nat.gcd a b = 5 ∧ 
    Nat.lcm a b = 20030010 :=
by sorry

end NUMINAMATH_CALUDE_smallest_lcm_five_digit_gcd_five_l3725_372569


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l3725_372534

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_terms (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => sum_of_terms b n + b (n + 1)

theorem sequence_sum_theorem (a b c : ℕ → ℝ) (d : ℝ) :
  d > 0 ∧
  arithmetic_sequence a d ∧
  a 2 + a 5 = 12 ∧
  a 2 * a 5 = 27 ∧
  b 1 = 3 ∧
  (∀ n : ℕ, b (n + 1) = 2 * sum_of_terms b n + 3) ∧
  (∀ n : ℕ, c n = a n / b n) →
  ∀ n : ℕ, sum_of_terms c n = 1 - (n + 1 : ℝ) / 3^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l3725_372534


namespace NUMINAMATH_CALUDE_complex_coordinates_l3725_372574

theorem complex_coordinates (z : ℂ) : z = (2 + Complex.I) / Complex.I → 
  Complex.re z = 1 ∧ Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_coordinates_l3725_372574


namespace NUMINAMATH_CALUDE_square_of_trinomial_13_5_3_l3725_372599

theorem square_of_trinomial_13_5_3 : (13 + 5 + 3)^2 = 441 := by
  sorry

end NUMINAMATH_CALUDE_square_of_trinomial_13_5_3_l3725_372599


namespace NUMINAMATH_CALUDE_inequality_proof_l3725_372539

theorem inequality_proof (x y z t : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ t ≥ 0) 
  (h_sum : x + y + z + t = 4) : 
  Real.sqrt (x^2 + t^2) + Real.sqrt (z^2 + 1) + Real.sqrt (z^2 + t^2) + 
  Real.sqrt (y^2 + x^2) + Real.sqrt (y^2 + 64) ≥ 13 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3725_372539


namespace NUMINAMATH_CALUDE_price_change_l3725_372589

theorem price_change (x : ℝ) (h : x > 0) : x * (1 - 0.2) * (1 + 0.2) < x := by
  sorry

end NUMINAMATH_CALUDE_price_change_l3725_372589


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3725_372514

theorem sin_alpha_value (α : ℝ) (h : 3 * Real.sin (2 * α) = Real.cos α) : 
  Real.sin α = 1/6 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3725_372514


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3725_372583

theorem adult_ticket_cost 
  (total_seats : ℕ) 
  (child_ticket_cost : ℚ) 
  (num_children : ℕ) 
  (total_revenue : ℚ) 
  (h1 : total_seats = 250) 
  (h2 : child_ticket_cost = 4) 
  (h3 : num_children = 188) 
  (h4 : total_revenue = 1124) :
  let num_adults : ℕ := total_seats - num_children
  let adult_ticket_cost : ℚ := (total_revenue - (↑num_children * child_ticket_cost)) / ↑num_adults
  adult_ticket_cost = 6 := by
sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3725_372583


namespace NUMINAMATH_CALUDE_town_population_proof_l3725_372542

/-- The annual decrease rate of the town's population -/
def annual_decrease_rate : ℝ := 0.2

/-- The population after 2 years -/
def population_after_2_years : ℝ := 19200

/-- The initial population of the town -/
def initial_population : ℝ := 30000

theorem town_population_proof :
  let remaining_rate := 1 - annual_decrease_rate
  (remaining_rate ^ 2) * initial_population = population_after_2_years :=
by sorry

end NUMINAMATH_CALUDE_town_population_proof_l3725_372542


namespace NUMINAMATH_CALUDE_distance_between_points_l3725_372552

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (-2, -3)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3725_372552


namespace NUMINAMATH_CALUDE_cost_price_75_equals_selling_price_40_implies_87_5_percent_gain_l3725_372527

/-- Calculates the gain percent given the ratio of cost price to selling price -/
def gainPercent (costPriceRatio sellingPriceRatio : ℕ) : ℚ :=
  ((sellingPriceRatio : ℚ) / (costPriceRatio : ℚ) - 1) * 100

/-- Theorem stating that if the cost price of 75 articles equals the selling price of 40 articles, 
    then the gain percent is 87.5% -/
theorem cost_price_75_equals_selling_price_40_implies_87_5_percent_gain :
  gainPercent 75 40 = 87.5 := by sorry

end NUMINAMATH_CALUDE_cost_price_75_equals_selling_price_40_implies_87_5_percent_gain_l3725_372527


namespace NUMINAMATH_CALUDE_second_quadrant_angle_ratio_l3725_372554

theorem second_quadrant_angle_ratio (x : Real) : 
  (π/2 < x) ∧ (x < π) →  -- x is in the second quadrant
  (Real.tan x)^2 + 3*(Real.tan x) - 4 = 0 → 
  (Real.sin x + Real.cos x) / (2*(Real.sin x) - Real.cos x) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_second_quadrant_angle_ratio_l3725_372554


namespace NUMINAMATH_CALUDE_book_selling_price_l3725_372553

theorem book_selling_price (cost_price : ℝ) (profit_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 250 →
  profit_percentage = 20 →
  selling_price = cost_price * (1 + profit_percentage / 100) →
  selling_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_selling_price_l3725_372553


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_24_l3725_372502

theorem product_of_four_consecutive_integers_divisible_by_24 (n : ℤ) :
  ∃ k : ℤ, n * (n + 1) * (n + 2) * (n + 3) = 24 * k :=
by sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_divisible_by_24_l3725_372502


namespace NUMINAMATH_CALUDE_people_per_cubic_yard_l3725_372513

theorem people_per_cubic_yard (people_per_yard : ℕ) : 
  (9000 * people_per_yard - 6400 * people_per_yard = 208000) → 
  people_per_yard = 80 := by
sorry

end NUMINAMATH_CALUDE_people_per_cubic_yard_l3725_372513


namespace NUMINAMATH_CALUDE_production_average_l3725_372558

theorem production_average (n : ℕ) : 
  (∀ (past_total : ℕ), past_total = n * 50 →
   ∀ (new_total : ℕ), new_total = past_total + 90 →
   (new_total : ℚ) / (n + 1 : ℚ) = 52) →
  n = 19 := by
sorry

end NUMINAMATH_CALUDE_production_average_l3725_372558


namespace NUMINAMATH_CALUDE_tiger_catch_distance_l3725_372556

/-- Calculates the distance a tiger travels from a zoo given specific conditions --/
def tiger_distance (initial_speed : ℝ) (initial_time : ℝ) (slow_speed : ℝ) (slow_time : ℝ) (chase_speed : ℝ) (chase_time : ℝ) : ℝ :=
  initial_speed * initial_time + slow_speed * slow_time + chase_speed * chase_time

/-- Proves that the tiger is caught 140 miles away from the zoo --/
theorem tiger_catch_distance :
  let initial_speed : ℝ := 25
  let initial_time : ℝ := 7
  let slow_speed : ℝ := 10
  let slow_time : ℝ := 4
  let chase_speed : ℝ := 50
  let chase_time : ℝ := 0.5
  tiger_distance initial_speed initial_time slow_speed slow_time chase_speed chase_time = 140 := by
  sorry

#eval tiger_distance 25 7 10 4 50 0.5

end NUMINAMATH_CALUDE_tiger_catch_distance_l3725_372556


namespace NUMINAMATH_CALUDE_B_power_101_l3725_372530

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![0, 0, 1],
    ![0, 0, 0],
    ![-1, 0, 0]]

theorem B_power_101 : B^101 = B := by sorry

end NUMINAMATH_CALUDE_B_power_101_l3725_372530


namespace NUMINAMATH_CALUDE_circus_acrobats_l3725_372536

/-- Represents the number of acrobats in the circus show -/
def acrobats : ℕ := 11

/-- Represents the number of elephants in the circus show -/
def elephants : ℕ := 4

/-- Represents the number of clowns in the circus show -/
def clowns : ℕ := 10

/-- The total number of legs in the circus show -/
def total_legs : ℕ := 58

/-- The total number of heads in the circus show -/
def total_heads : ℕ := 25

/-- Theorem stating that the number of acrobats is 11 given the conditions of the circus show -/
theorem circus_acrobats :
  (2 * acrobats + 4 * elephants + 2 * clowns = total_legs) ∧
  (acrobats + elephants + clowns = total_heads) ∧
  (acrobats = 11) := by
  sorry

end NUMINAMATH_CALUDE_circus_acrobats_l3725_372536


namespace NUMINAMATH_CALUDE_sum_edges_vertices_faces_l3725_372519

/-- A rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The number of edges in a rectangular prism -/
def num_edges (p : RectangularPrism) : ℕ := 12

/-- The number of vertices in a rectangular prism -/
def num_vertices (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (p : RectangularPrism) : ℕ := 6

/-- The sum of edges, vertices, and faces in a rectangular prism is 26 -/
theorem sum_edges_vertices_faces (p : RectangularPrism) :
  num_edges p + num_vertices p + num_faces p = 26 := by
  sorry

#check sum_edges_vertices_faces

end NUMINAMATH_CALUDE_sum_edges_vertices_faces_l3725_372519


namespace NUMINAMATH_CALUDE_complex_multiplication_l3725_372520

theorem complex_multiplication (i : ℂ) : i * i = -1 → (2 + 3*i) * (3 - 2*i) = 12 + 5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l3725_372520


namespace NUMINAMATH_CALUDE_normal_dist_probability_l3725_372557

-- Define the normal distribution
def normal_dist (μ σ : ℝ) : Type := Unit

-- Define the probability function
noncomputable def P (X : normal_dist 4 1) (a b : ℝ) : ℝ := sorry

-- State the theorem
theorem normal_dist_probability 
  (X : normal_dist 4 1) 
  (h1 : P X (4 - 2) (4 + 2) = 0.9544) 
  (h2 : P X (4 - 1) (4 + 1) = 0.6826) : 
  P X 5 6 = 0.1359 := by sorry

end NUMINAMATH_CALUDE_normal_dist_probability_l3725_372557


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l3725_372531

theorem geometric_sequence_ninth_term :
  let a₁ : ℚ := 5
  let r : ℚ := 3/4
  let n : ℕ := 9
  let aₙ : ℕ → ℚ := λ k => a₁ * r^(k - 1)
  aₙ n = 32805/65536 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l3725_372531


namespace NUMINAMATH_CALUDE_tangent_slope_implies_b_over_a_equals_two_l3725_372595

/-- A quadratic function f(x) = ax² + b with a tangent line of slope 2 at (1,3) -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b

/-- The derivative of f -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * a * x

theorem tangent_slope_implies_b_over_a_equals_two (a b : ℝ) :
  f a b 1 = 3 ∧ f_derivative a 1 = 2 → b / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_b_over_a_equals_two_l3725_372595


namespace NUMINAMATH_CALUDE_saturday_to_weekday_ratio_total_weekly_time_correct_total_weekly_time_is_four_hours_l3725_372528

/-- Represents the number of minutes Elle practices piano on different days of the week. -/
structure PracticeTimes where
  weekday : Nat  -- Practice time on each weekday (Monday to Friday)
  saturday : Nat -- Practice time on Saturday
  total_weekly : Nat -- Total practice time in the week

/-- Represents the practice schedule of Elle -/
def elles_practice : PracticeTimes where
  weekday := 30
  saturday := 90
  total_weekly := 240

/-- The ratio of Saturday practice time to weekday practice time is 3:1 -/
theorem saturday_to_weekday_ratio :
  elles_practice.saturday / elles_practice.weekday = 3 := by
  sorry

/-- The total weekly practice time is correct -/
theorem total_weekly_time_correct :
  elles_practice.total_weekly = elles_practice.weekday * 5 + elles_practice.saturday := by
  sorry

/-- The total weekly practice time is 4 hours -/
theorem total_weekly_time_is_four_hours :
  elles_practice.total_weekly = 4 * 60 := by
  sorry

end NUMINAMATH_CALUDE_saturday_to_weekday_ratio_total_weekly_time_correct_total_weekly_time_is_four_hours_l3725_372528


namespace NUMINAMATH_CALUDE_intersection_determines_m_and_n_l3725_372529

open Set

def A : Set ℝ := {x | |x + 2| < 3}
def B (m : ℝ) : Set ℝ := {x | (x - m) * (x - 2) < 0}

theorem intersection_determines_m_and_n (m n : ℝ) :
  A ∩ B m = Ioo (-1) n → m = -1 ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_determines_m_and_n_l3725_372529


namespace NUMINAMATH_CALUDE_sausage_cost_per_pound_l3725_372571

theorem sausage_cost_per_pound : 
  let packages : ℕ := 3
  let pounds_per_package : ℕ := 2
  let total_cost : ℕ := 24
  let total_pounds := packages * pounds_per_package
  let cost_per_pound := total_cost / total_pounds
  cost_per_pound = 4 := by sorry

end NUMINAMATH_CALUDE_sausage_cost_per_pound_l3725_372571


namespace NUMINAMATH_CALUDE_stratified_sample_sum_l3725_372596

/-- Represents the number of items in each food category -/
structure FoodCategories where
  grains : ℕ
  vegetableOils : ℕ
  animalDerived : ℕ
  fruitsAndVegetables : ℕ

/-- Calculates the total number of items across all categories -/
def totalItems (fc : FoodCategories) : ℕ :=
  fc.grains + fc.vegetableOils + fc.animalDerived + fc.fruitsAndVegetables

/-- Calculates the number of items to be sampled from a category in stratified sampling -/
def stratifiedSampleSize (categorySize sampleSize totalSize : ℕ) : ℕ :=
  (categorySize * sampleSize) / totalSize

/-- Theorem: In a stratified sample of 20 items from the given food categories,
    the sum of items from vegetable oils and fruits and vegetables is 6 -/
theorem stratified_sample_sum (fc : FoodCategories) 
    (h1 : fc.grains = 40)
    (h2 : fc.vegetableOils = 10)
    (h3 : fc.animalDerived = 30)
    (h4 : fc.fruitsAndVegetables = 20)
    (h5 : totalItems fc = 100)
    (sampleSize : ℕ)
    (h6 : sampleSize = 20) :
    stratifiedSampleSize fc.vegetableOils sampleSize (totalItems fc) +
    stratifiedSampleSize fc.fruitsAndVegetables sampleSize (totalItems fc) = 6 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sample_sum_l3725_372596


namespace NUMINAMATH_CALUDE_intersection_A_B_l3725_372551

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}

theorem intersection_A_B : A ∩ B = {3, 5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3725_372551


namespace NUMINAMATH_CALUDE_mike_shortfall_l3725_372505

def max_marks : ℕ := 800
def pass_percentage : ℚ := 30 / 100
def mike_score : ℕ := 212

theorem mike_shortfall :
  (↑max_marks * pass_percentage).floor - mike_score = 28 :=
sorry

end NUMINAMATH_CALUDE_mike_shortfall_l3725_372505


namespace NUMINAMATH_CALUDE_max_marks_proof_l3725_372577

/-- Given a passing threshold, actual score, and shortfall, calculates the maximum possible marks -/
def calculate_max_marks (passing_threshold : ℚ) (actual_score : ℕ) (shortfall : ℕ) : ℚ :=
  (actual_score + shortfall : ℚ) / passing_threshold

/-- Proves that the maximum marks is 617.5 given the problem conditions -/
theorem max_marks_proof (passing_threshold : ℚ) (actual_score : ℕ) (shortfall : ℕ) 
    (h1 : passing_threshold = 0.4)
    (h2 : actual_score = 212)
    (h3 : shortfall = 35) :
  calculate_max_marks passing_threshold actual_score shortfall = 617.5 := by
  sorry

#eval calculate_max_marks 0.4 212 35

end NUMINAMATH_CALUDE_max_marks_proof_l3725_372577


namespace NUMINAMATH_CALUDE_parabola_intersection_points_l3725_372573

/-- The x-coordinates of the intersection points of two parabolas -/
theorem parabola_intersection_points (x : ℝ) :
  (3 * x^2 - 4 * x + 7 = 6 * x^2 + x + 3) ↔ 
  (x = (5 + Real.sqrt 73) / -6 ∨ x = (5 - Real.sqrt 73) / -6) := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_points_l3725_372573


namespace NUMINAMATH_CALUDE_base_conversion_sum_fraction_l3725_372510

/-- Given that 546 in base 7 is equal to xy9 in base 10, where x and y are single digits,
    prove that (x + y + 9) / 21 = 6 / 7 -/
theorem base_conversion_sum_fraction :
  ∃ (x y : ℕ), x < 10 ∧ y < 10 ∧ 
  (5 * 7^2 + 4 * 7 + 6 : ℕ) = x * 100 + y * 10 + 9 →
  (x + y + 9 : ℚ) / 21 = 6 / 7 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_sum_fraction_l3725_372510


namespace NUMINAMATH_CALUDE_inequality_proof_l3725_372588

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * d) - Real.sqrt (b * c)| ≤ 1) : 
  (a * e + b / e) * (c * e + d / e) ≥ (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3725_372588


namespace NUMINAMATH_CALUDE_teacher_age_l3725_372564

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (total_avg_age : ℝ) :
  num_students = 10 →
  student_avg_age = 15 →
  total_avg_age = 16 →
  (num_students : ℝ) * student_avg_age + (num_students + 1) * total_avg_age - num_students * total_avg_age = 26 :=
by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l3725_372564


namespace NUMINAMATH_CALUDE_flowchart_result_for_6_l3725_372562

-- Define the function that represents the flowchart logic
def flowchart_program (n : ℕ) : ℕ :=
  -- The actual implementation is not provided, so we'll use a placeholder
  sorry

-- Theorem statement
theorem flowchart_result_for_6 : flowchart_program 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_flowchart_result_for_6_l3725_372562


namespace NUMINAMATH_CALUDE_f_three_point_five_l3725_372500

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the property that f(x+2) is an odd function
axiom f_odd (x : ℝ) : f (-(x + 2)) = -f (x + 2)

-- Define the property that f(x) = 2x for x ∈ (0,2)
axiom f_linear (x : ℝ) : x > 0 → x < 2 → f x = 2 * x

-- Theorem to prove
theorem f_three_point_five : f 3.5 = -1 := by sorry

end NUMINAMATH_CALUDE_f_three_point_five_l3725_372500


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l3725_372545

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l3725_372545


namespace NUMINAMATH_CALUDE_monogram_count_l3725_372581

def alphabet_size : ℕ := 26

theorem monogram_count : (alphabet_size.choose 2) = 325 := by sorry

end NUMINAMATH_CALUDE_monogram_count_l3725_372581


namespace NUMINAMATH_CALUDE_vasya_reads_entire_book_l3725_372585

theorem vasya_reads_entire_book :
  let first_day : ℚ := 1/2
  let second_day : ℚ := (1/3) * (1 - first_day)
  let first_two_days : ℚ := first_day + second_day
  let third_day : ℚ := (1/2) * first_two_days
  first_day + second_day + third_day = 1 := by sorry

end NUMINAMATH_CALUDE_vasya_reads_entire_book_l3725_372585


namespace NUMINAMATH_CALUDE_intersection_of_tangents_l3725_372541

/-- A curve defined by y = x + 1/x for x > 0 -/
def C : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1 / p.1 ∧ p.1 > 0}

/-- A line passing through (0,1) with slope k -/
def line (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * p.1 + 1}

/-- The intersection points of the line with the curve C -/
def intersection_points (k : ℝ) : Set (ℝ × ℝ) := C ∩ line k

/-- The tangent line to C at a point (x, y) -/
def tangent_line (x : ℝ) : Set (ℝ × ℝ) := 
  {p | p.2 = (1 - 1/x^2) * p.1 + 2/x}

theorem intersection_of_tangents (k : ℝ) :
  ∀ M N : ℝ × ℝ, M ∈ intersection_points k → N ∈ intersection_points k → M ≠ N →
  ∃ P : ℝ × ℝ, P ∈ tangent_line M.1 ∧ P ∈ tangent_line N.1 ∧ 
  P.1 = 2 ∧ 2 < P.2 ∧ P.2 < 2.5 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_tangents_l3725_372541


namespace NUMINAMATH_CALUDE_jerry_games_won_l3725_372517

theorem jerry_games_won (ken dave jerry : ℕ) 
  (h1 : ken = dave + 5)
  (h2 : dave = jerry + 3)
  (h3 : ken + dave + jerry = 32) : 
  jerry = 7 := by
  sorry

end NUMINAMATH_CALUDE_jerry_games_won_l3725_372517


namespace NUMINAMATH_CALUDE_john_calculation_l3725_372593

theorem john_calculation (n : ℕ) (h : n = 40) : n^2 - (n - 1)^2 = 2*n - 1 := by
  sorry

#check john_calculation

end NUMINAMATH_CALUDE_john_calculation_l3725_372593


namespace NUMINAMATH_CALUDE_percentage_relation_l3725_372518

theorem percentage_relation (x y z : ℝ) :
  y = 0.3 * z →
  x = 0.36 * z →
  x = y * 1.2 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3725_372518


namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l3725_372544

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 4 * y + 6 = 0
def l₂ (a x y : ℝ) : Prop := ((3/4) * a + 1) * x + a * y - 3/2 = 0

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y ↔ l₂ a x y) ↔ a = 4 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, l₁ a x y → l₂ a x y → x * x + y * y = 0) ↔ (a = 0 ∨ a = -20/3) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l3725_372544


namespace NUMINAMATH_CALUDE_circus_performance_time_l3725_372580

/-- Represents the time each entertainer stands on their back legs -/
structure CircusTime where
  pulsar : ℝ
  polly : ℝ
  petra : ℝ
  penny : ℝ
  parker : ℝ

/-- Calculates the total time all entertainers stand on their back legs -/
def totalTime (ct : CircusTime) : ℝ :=
  ct.pulsar + ct.polly + ct.petra + ct.penny + ct.parker

/-- Theorem stating the conditions and the result to be proved -/
theorem circus_performance_time :
  ∀ (ct : CircusTime),
    ct.pulsar = 10 →
    ct.polly = 3 * ct.pulsar →
    ct.petra = ct.polly / 6 →
    ct.penny = 2 * (ct.pulsar + ct.polly + ct.petra) →
    ct.parker = (ct.pulsar + ct.polly + ct.petra + ct.penny) / 4 →
    totalTime ct = 168.75 := by
  sorry


end NUMINAMATH_CALUDE_circus_performance_time_l3725_372580


namespace NUMINAMATH_CALUDE_second_prize_proportion_l3725_372578

theorem second_prize_proportion (total winners : ℕ) 
  (first second third : ℕ) 
  (h1 : first + second + third = winners)
  (h2 : (first + second : ℚ) / winners = 3 / 4)
  (h3 : (second + third : ℚ) / winners = 2 / 3) :
  (second : ℚ) / winners = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_second_prize_proportion_l3725_372578


namespace NUMINAMATH_CALUDE_circle_radius_increase_l3725_372549

/-- Given a circle with radius r, prove that when the radius is increased by 5 and the area is quadrupled, the original radius was 5 and the new perimeter is 20π. -/
theorem circle_radius_increase (r : ℝ) : 
  (π * (r + 5)^2 = 4 * π * r^2) → 
  (r = 5 ∧ 2 * π * (r + 5) = 20 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_increase_l3725_372549


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3725_372560

theorem right_triangle_inequality (a b c : ℝ) (n : ℕ) 
  (h_right_triangle : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_n_ge_3 : n ≥ 3) : 
  a^n + b^n < c^n := by
sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3725_372560


namespace NUMINAMATH_CALUDE_f_4_1981_l3725_372524

def f : ℕ → ℕ → ℕ 
  | 0, y => y + 1
  | x + 1, 0 => f x 1
  | x + 1, y + 1 => f x (f (x + 1) y)

theorem f_4_1981 : f 4 1981 = 2^(2^(2^(1981 + 1) + 1)) - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_4_1981_l3725_372524


namespace NUMINAMATH_CALUDE_factorization_identity_l3725_372538

theorem factorization_identity (m : ℝ) : m^2 + 3*m = m*(m + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_identity_l3725_372538
