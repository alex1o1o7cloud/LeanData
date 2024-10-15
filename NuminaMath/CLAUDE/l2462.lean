import Mathlib

namespace NUMINAMATH_CALUDE_stock_price_change_l2462_246240

theorem stock_price_change (total_stocks : ℕ) (higher_price_stocks : ℕ) 
  (h1 : total_stocks = 1980)
  (h2 : higher_price_stocks = (total_stocks - higher_price_stocks) * 6 / 5) :
  higher_price_stocks = 1080 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l2462_246240


namespace NUMINAMATH_CALUDE_factorization_equality_l2462_246282

theorem factorization_equality (a b : ℝ) : a^2 - 2*a*b = a*(a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2462_246282


namespace NUMINAMATH_CALUDE_x_equals_seven_l2462_246297

theorem x_equals_seven (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 7 * x^2 + 14 * x * y = x^3 + 3 * x^2 * y) : x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_seven_l2462_246297


namespace NUMINAMATH_CALUDE_right_angled_triangle_345_l2462_246281

theorem right_angled_triangle_345 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a / b = 3 / 4) (h5 : b / c = 4 / 5) : a^2 + b^2 = c^2 := by
sorry


end NUMINAMATH_CALUDE_right_angled_triangle_345_l2462_246281


namespace NUMINAMATH_CALUDE_total_work_hours_l2462_246228

theorem total_work_hours (hours_per_day : ℕ) (days_worked : ℕ) (total_hours : ℕ) : 
  hours_per_day = 3 → days_worked = 6 → total_hours = hours_per_day * days_worked → total_hours = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_work_hours_l2462_246228


namespace NUMINAMATH_CALUDE_triangle_problem_l2462_246260

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > b →
  a = 5 →
  c = 6 →
  Real.sin B = 3/5 →
  b = Real.sqrt 13 ∧
  Real.sin A = (3 * Real.sqrt 13) / 13 ∧
  Real.sin (2 * A + π/4) = (7 * Real.sqrt 2) / 26 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2462_246260


namespace NUMINAMATH_CALUDE_profit_percentage_doubling_l2462_246224

theorem profit_percentage_doubling (cost_price : ℝ) (original_selling_price : ℝ) :
  original_selling_price = cost_price * 1.3 →
  let double_price := original_selling_price * 2
  let new_profit_percentage := (double_price - cost_price) / cost_price * 100
  new_profit_percentage = 160 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_doubling_l2462_246224


namespace NUMINAMATH_CALUDE_right_triangle_semicircle_segments_l2462_246223

theorem right_triangle_semicircle_segments 
  (a b : ℝ) 
  (ha : a = 75) 
  (hb : b = 100) : 
  ∃ (x y : ℝ), 
    x = 48 ∧ 
    y = 36 ∧ 
    x * (a^2 + b^2) = a * b^2 ∧ 
    y * (a^2 + b^2) = b * a^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_semicircle_segments_l2462_246223


namespace NUMINAMATH_CALUDE_min_simultaneous_return_time_l2462_246218

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]

def is_simultaneous_return (t : Nat) (horse_times : List Nat) : Bool :=
  (horse_times.filter (fun time => t % time = 0)).length ≥ 4

theorem min_simultaneous_return_time :
  let horse_times := first_seven_primes
  (∃ (t : Nat), t > 0 ∧ is_simultaneous_return t horse_times) ∧
  (∀ (t : Nat), 0 < t ∧ t < 210 → ¬is_simultaneous_return t horse_times) ∧
  is_simultaneous_return 210 horse_times :=
by sorry

end NUMINAMATH_CALUDE_min_simultaneous_return_time_l2462_246218


namespace NUMINAMATH_CALUDE_points_lost_in_last_round_l2462_246213

-- Define the variables
def first_round_points : ℕ := 17
def second_round_points : ℕ := 6
def final_points : ℕ := 7

-- Define the theorem
theorem points_lost_in_last_round :
  (first_round_points + second_round_points) - final_points = 16 := by
  sorry

end NUMINAMATH_CALUDE_points_lost_in_last_round_l2462_246213


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l2462_246269

theorem arithmetic_geometric_mean_ratio (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a + b) / 2 = 3 * Real.sqrt (a * b)) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |a / b - 34| < ε :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_ratio_l2462_246269


namespace NUMINAMATH_CALUDE_add_squared_terms_l2462_246225

theorem add_squared_terms (a : ℝ) : a^2 + 3*a^2 = 4*a^2 := by
  sorry

end NUMINAMATH_CALUDE_add_squared_terms_l2462_246225


namespace NUMINAMATH_CALUDE_largest_b_for_no_real_roots_l2462_246263

theorem largest_b_for_no_real_roots : ∃ (b : ℤ),
  (∀ (x : ℝ), x^3 + b*x^2 + 15*x + 22 ≠ 0) ∧
  (∀ (b' : ℤ), b' > b → ∃ (x : ℝ), x^3 + b'*x^2 + 15*x + 22 = 0) ∧
  b = 5 := by
  sorry


end NUMINAMATH_CALUDE_largest_b_for_no_real_roots_l2462_246263


namespace NUMINAMATH_CALUDE_valuable_files_count_l2462_246262

def initial_download : ℕ := 800
def first_deletion_rate : ℚ := 70 / 100
def second_download : ℕ := 400
def second_deletion_rate : ℚ := 3 / 5

theorem valuable_files_count : 
  (initial_download - (initial_download * first_deletion_rate).floor) + 
  (second_download - (second_download * second_deletion_rate).floor) = 400 :=
by sorry

end NUMINAMATH_CALUDE_valuable_files_count_l2462_246262


namespace NUMINAMATH_CALUDE_function_properties_l2462_246284

/-- Given a function f(x) = a*sin(2x) + cos(2x) where f(π/3) = (√3 - 1)/2,
    prove properties about the value of a, the maximum value of f(x),
    and the intervals where f(x) is monotonically decreasing. -/
theorem function_properties (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * Real.sin (2 * x) + Real.cos (2 * x)) :
  f (π / 3) = (Real.sqrt 3 - 1) / 2 →
  (a = 1 ∧ 
   (∃ M, M = Real.sqrt 2 ∧ ∀ x, f x ≤ M) ∧
   ∀ k : ℤ, ∀ x ∈ Set.Icc (k * π + π / 4) (k * π + 3 * π / 4), 
     ∀ y ∈ Set.Icc (k * π + π / 4) (k * π + 3 * π / 4), 
       x ≤ y → f y ≤ f x) :=
by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2462_246284


namespace NUMINAMATH_CALUDE_puzzle_pieces_missing_l2462_246295

theorem puzzle_pieces_missing (total : ℕ) (border : ℕ) (trevor : ℕ) (joe : ℕ) 
  (h1 : total = 500)
  (h2 : border = 75)
  (h3 : trevor = 105)
  (h4 : joe = 3 * trevor) :
  total - (border + trevor + joe) = 5 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_pieces_missing_l2462_246295


namespace NUMINAMATH_CALUDE_wuyang_cup_result_l2462_246202

-- Define the teams
inductive Team : Type
| A : Team
| B : Team
| C : Team
| D : Team

-- Define the positions
inductive Position : Type
| Champion : Position
| RunnerUp : Position
| Third : Position
| Last : Position

-- Define the result type
def Result := Team → Position

-- Define the predictor type
inductive Predictor : Type
| Jia : Predictor
| Yi : Predictor
| Bing : Predictor

-- Define the prediction type
def Prediction := Predictor → Team → Position

-- Define the correctness of a prediction
def is_correct (pred : Prediction) (result : Result) (p : Predictor) (t : Team) : Prop :=
  pred p t = result t

-- Define the condition that each predictor is half right and half wrong
def half_correct (pred : Prediction) (result : Result) (p : Predictor) : Prop :=
  (∃ t1 t2 : Team, t1 ≠ t2 ∧ is_correct pred result p t1 ∧ is_correct pred result p t2) ∧
  (∃ t3 t4 : Team, t3 ≠ t4 ∧ ¬is_correct pred result p t3 ∧ ¬is_correct pred result p t4)

-- Define the predictions
def predictions (pred : Prediction) : Prop :=
  pred Predictor.Jia Team.C = Position.RunnerUp ∧
  pred Predictor.Jia Team.D = Position.Third ∧
  pred Predictor.Yi Team.D = Position.Last ∧
  pred Predictor.Yi Team.A = Position.RunnerUp ∧
  pred Predictor.Bing Team.C = Position.Champion ∧
  pred Predictor.Bing Team.B = Position.RunnerUp

-- State the theorem
theorem wuyang_cup_result :
  ∀ (pred : Prediction) (result : Result),
    predictions pred →
    (∀ p : Predictor, half_correct pred result p) →
    result Team.C = Position.Champion ∧
    result Team.A = Position.RunnerUp ∧
    result Team.D = Position.Third ∧
    result Team.B = Position.Last :=
sorry

end NUMINAMATH_CALUDE_wuyang_cup_result_l2462_246202


namespace NUMINAMATH_CALUDE_arcade_candy_cost_l2462_246254

theorem arcade_candy_cost (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candies : ℕ) :
  whack_a_mole_tickets = 26 →
  skee_ball_tickets = 19 →
  candies = 5 →
  (whack_a_mole_tickets + skee_ball_tickets) / candies = 9 :=
by sorry

end NUMINAMATH_CALUDE_arcade_candy_cost_l2462_246254


namespace NUMINAMATH_CALUDE_march_temperature_data_inconsistent_l2462_246299

/-- Represents the statistical data for March temperatures --/
structure MarchTemperatureData where
  mean : ℝ
  median : ℝ
  variance : ℝ
  mean_eq_zero : mean = 0
  median_eq_four : median = 4
  variance_eq : variance = 15.917

/-- Theorem stating that the given data is inconsistent --/
theorem march_temperature_data_inconsistent (data : MarchTemperatureData) :
  (data.mean - data.median)^2 > data.variance := by
  sorry

#check march_temperature_data_inconsistent

end NUMINAMATH_CALUDE_march_temperature_data_inconsistent_l2462_246299


namespace NUMINAMATH_CALUDE_arc_length_quarter_circle_l2462_246206

/-- Given a circle D with circumference 72 feet and an arc EF subtended by a central angle of 90°,
    prove that the length of arc EF is 18 feet. -/
theorem arc_length_quarter_circle (D : Real) (EF : Real) :
  D = 72 → -- Circumference of circle D is 72 feet
  EF = D / 4 → -- Arc EF is subtended by a 90° angle (1/4 of the circle)
  EF = 18 := by sorry

end NUMINAMATH_CALUDE_arc_length_quarter_circle_l2462_246206


namespace NUMINAMATH_CALUDE_right_triangle_sets_l2462_246270

theorem right_triangle_sets : ∃! (a b c : ℕ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 6 ∧ b = 8 ∧ c = 10) ∨
   (a = 5 ∧ b = 2 ∧ c = 5) ∨
   (a = 5 ∧ b = 12 ∧ c = 13)) ∧
  ¬(a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l2462_246270


namespace NUMINAMATH_CALUDE_unique_star_solution_l2462_246249

/-- Definition of the ⋆ operation -/
def star (x y : ℝ) : ℝ := 5*x - 4*y + 2*x*y

/-- Theorem stating that there exists exactly one real number y such that 4 ⋆ y = 20 -/
theorem unique_star_solution : ∃! y : ℝ, star 4 y = 20 := by
  sorry

end NUMINAMATH_CALUDE_unique_star_solution_l2462_246249


namespace NUMINAMATH_CALUDE_solution_set_f_gt_3_range_of_a_l2462_246266

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 1|

-- Define the function g (although not used in the proof)
def g (x : ℝ) : ℝ := |2*x - 1| + 3

-- Theorem 1: The solution set of f(x) > 3 is (0, +∞)
theorem solution_set_f_gt_3 : 
  {x : ℝ | f x > 3} = {x : ℝ | x > 0} := by sorry

-- Theorem 2: If f(x) + 1 < 4^a - 5×2^a has a solution, then a < 0 or a > 2
theorem range_of_a (a : ℝ) : 
  (∃ x, f x + 1 < 4^a - 5*2^a) → (a < 0 ∨ a > 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_gt_3_range_of_a_l2462_246266


namespace NUMINAMATH_CALUDE_max_value_ratio_l2462_246247

theorem max_value_ratio (a b c d : ℝ) 
  (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0)
  (h5 : a^2 + b^2 + c^2 + d^2 = ((a + b + c + d)^2) / 3) :
  (a + c) / (b + d) ≤ (7 + 2 * Real.sqrt 6) / 5 ∧ 
  ∃ a b c d, a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d > 0 ∧
    a^2 + b^2 + c^2 + d^2 = ((a + b + c + d)^2) / 3 ∧
    (a + c) / (b + d) = (7 + 2 * Real.sqrt 6) / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_ratio_l2462_246247


namespace NUMINAMATH_CALUDE_cos_90_degrees_l2462_246280

theorem cos_90_degrees : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_l2462_246280


namespace NUMINAMATH_CALUDE_smallest_x_for_cube_equation_l2462_246230

theorem smallest_x_for_cube_equation (N : ℕ+) (h : 1260 * x = N^3) : 
  ∃ (x : ℕ), x = 7350 ∧ ∀ (y : ℕ), 1260 * y = N^3 → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_cube_equation_l2462_246230


namespace NUMINAMATH_CALUDE_inequality_solution_l2462_246279

theorem inequality_solution (a x : ℝ) : 
  (x - a) * (x - a^2) < 0 ↔ 
  ((a < 0 ∨ a > 1) ∧ a < x ∧ x < a^2) ∨ 
  (0 < a ∧ a < 1 ∧ a^2 < x ∧ x < a) ∨ 
  (a = 0 ∨ a = 1 ∧ False) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2462_246279


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_plus_double_factorial_l2462_246272

/-- Double factorial of a natural number -/
def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

/-- Sum of factorials from 1 to n -/
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => Nat.factorial (i + 1))

/-- Theorem: The units digit of the sum of factorials from 1 to 12 plus 12!! is 3 -/
theorem units_digit_sum_factorials_plus_double_factorial :
  (sum_factorials 12 + double_factorial 12) % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_plus_double_factorial_l2462_246272


namespace NUMINAMATH_CALUDE_sum_of_products_l2462_246267

theorem sum_of_products (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) 
  (h : ∀ x : ℝ, x^6 + 2*x^5 + x^4 + x^3 + x^2 + 2*x + 1 = 
    (x^2 + a₁*x + d₁) * (x^2 + a₂*x + d₂) * (x^2 + a₃*x + d₃)) : 
  a₁*d₁ + a₂*d₂ + a₃*d₃ = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2462_246267


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_l2462_246210

theorem half_abs_diff_squares : (1 / 2 : ℝ) * |23^2 - 19^2| = 84 := by
  sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_l2462_246210


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2462_246235

-- Define the quadratic function
def f (x : ℝ) := 3 * x^2 - 7 * x - 6

-- Define the solution set
def solution_set : Set ℝ := {x | -2/3 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2462_246235


namespace NUMINAMATH_CALUDE_difference_23rd_21st_triangular_l2462_246209

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_23rd_21st_triangular : 
  triangular_number 23 - triangular_number 21 = 45 := by
  sorry

end NUMINAMATH_CALUDE_difference_23rd_21st_triangular_l2462_246209


namespace NUMINAMATH_CALUDE_tangent_line_parallel_to_y_equals_4x_l2462_246211

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_line_parallel_to_y_equals_4x :
  ∃! P₀ : ℝ × ℝ, 
    P₀.1 = 1 ∧ 
    P₀.2 = 0 ∧ 
    (∀ x : ℝ, f x = x^3 + x - 2) ∧
    (deriv f P₀.1 = 4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_to_y_equals_4x_l2462_246211


namespace NUMINAMATH_CALUDE_max_sides_convex_polygon_l2462_246237

/-- The maximum number of sides in a convex polygon with interior angles in arithmetic sequence -/
theorem max_sides_convex_polygon (n : ℕ) : n ≤ 8 :=
  let interior_angle (k : ℕ) := 100 + 10 * (k - 1)
  have h1 : ∀ k, k ≤ n → interior_angle k < 180 := by sorry
  have h2 : ∀ k, 1 ≤ k → k ≤ n → 0 < interior_angle k := by sorry
  have h3 : (n - 2) * 180 = (interior_angle 1 + interior_angle n) * n / 2 := by sorry
sorry

#check max_sides_convex_polygon

end NUMINAMATH_CALUDE_max_sides_convex_polygon_l2462_246237


namespace NUMINAMATH_CALUDE_chord_length_l2462_246294

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def C1 : Circle := { center := (0, 0), radius := 6 }
def C2 : Circle := { center := (18, 0), radius := 12 }
def C3 : Circle := { center := (38, 0), radius := 38 }
def C4 : Circle := { center := (58, 0), radius := 20 }

-- Define the properties of the circles
def externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c1.radius + c2.radius)^2

def internally_tangent (c1 c2 : Circle) : Prop :=
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 = (c2.radius - c1.radius)^2

def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Theorem statement
theorem chord_length :
  externally_tangent C1 C2 ∧
  internally_tangent C1 C3 ∧
  internally_tangent C2 C3 ∧
  externally_tangent C3 C4 ∧
  collinear C1.center C2.center C3.center →
  ∃ (chord_length : ℝ), chord_length = 10 * Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_chord_length_l2462_246294


namespace NUMINAMATH_CALUDE_closest_point_proof_l2462_246200

-- Define the cheese location
def cheese : ℝ × ℝ := (10, 10)

-- Define the mouse's path
def mouse_path (x : ℝ) : ℝ := -4 * x + 16

-- Define the point of closest approach
def closest_point : ℝ × ℝ := (2, 8)

-- Theorem statement
theorem closest_point_proof :
  -- The closest point is on the mouse's path
  mouse_path closest_point.1 = closest_point.2 ∧
  -- The closest point is indeed the closest to the cheese
  ∀ x : ℝ, x ≠ closest_point.1 →
    (x - cheese.1)^2 + (mouse_path x - cheese.2)^2 >
    (closest_point.1 - cheese.1)^2 + (closest_point.2 - cheese.2)^2 ∧
  -- The sum of coordinates of the closest point is 10
  closest_point.1 + closest_point.2 = 10 :=
sorry

end NUMINAMATH_CALUDE_closest_point_proof_l2462_246200


namespace NUMINAMATH_CALUDE_job_completion_time_l2462_246214

theorem job_completion_time (x : ℝ) (h1 : x > 0) (h2 : 9/x + 4/10 = 1) : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_time_l2462_246214


namespace NUMINAMATH_CALUDE_odd_prime_square_difference_l2462_246220

theorem odd_prime_square_difference (d : ℕ) : 
  Nat.Prime d → 
  d % 2 = 1 → 
  ∃ m : ℕ, 89 - (d + 3)^2 = m^2 → 
  d = 5 := by sorry

end NUMINAMATH_CALUDE_odd_prime_square_difference_l2462_246220


namespace NUMINAMATH_CALUDE_subtraction_of_decimals_l2462_246293

theorem subtraction_of_decimals : 3.75 - 1.46 = 2.29 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_decimals_l2462_246293


namespace NUMINAMATH_CALUDE_probability_is_three_fifths_l2462_246236

/-- The set of letters in the word "STATISTICS" -/
def statistics_letters : Finset Char := {'S', 'T', 'A', 'I', 'C'}

/-- The set of letters in the word "TEST" -/
def test_letters : Finset Char := {'T', 'E', 'S'}

/-- The number of occurrences of each letter in "STATISTICS" -/
def letter_count (c : Char) : ℕ :=
  if c = 'S' then 3
  else if c = 'T' then 3
  else if c = 'A' then 1
  else if c = 'I' then 2
  else if c = 'C' then 1
  else 0

/-- The total number of tiles -/
def total_tiles : ℕ := statistics_letters.sum letter_count

/-- The number of tiles with letters from "TEST" -/
def test_tiles : ℕ := (statistics_letters ∩ test_letters).sum letter_count

/-- The probability of selecting a tile with a letter from "TEST" -/
def probability : ℚ := test_tiles / total_tiles

theorem probability_is_three_fifths : probability = 3 / 5 := by
  sorry


end NUMINAMATH_CALUDE_probability_is_three_fifths_l2462_246236


namespace NUMINAMATH_CALUDE_fraction_irreducible_l2462_246287

theorem fraction_irreducible (n : ℤ) : 
  Nat.gcd (Int.natAbs (2*n^2 + 9*n - 17)) (Int.natAbs (n + 6)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l2462_246287


namespace NUMINAMATH_CALUDE_tetrahedron_sphere_area_l2462_246238

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  K : Point3D
  L : Point3D
  M : Point3D
  N : Point3D

/-- Represents a sphere -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Calculates the Euclidean distance between two points -/
def distance (p q : Point3D) : ℝ := sorry

/-- Calculates the angle between three points -/
def angle (p q r : Point3D) : ℝ := sorry

/-- Calculates the spherical distance between two points on a sphere -/
def sphericalDistance (s : Sphere) (p q : Point3D) : ℝ := sorry

/-- Checks if a point is on a sphere -/
def isOnSphere (s : Sphere) (p : Point3D) : Prop := sorry

/-- The set of points on the sphere satisfying the distance condition -/
def distanceSet (s : Sphere) (t : Tetrahedron) : Set Point3D :=
  {p : Point3D | isOnSphere s p ∧ 
    sphericalDistance s p t.K + sphericalDistance s p t.L + 
    sphericalDistance s p t.M + sphericalDistance s p t.N ≤ 6 * Real.pi}

/-- Calculates the area of a set on a sphere -/
def sphericalArea (s : Sphere) (set : Set Point3D) : ℝ := sorry

theorem tetrahedron_sphere_area 
  (t : Tetrahedron) 
  (s : Sphere) 
  (h1 : distance t.K t.L = 5)
  (h2 : distance t.N t.M = 6)
  (h3 : angle t.L t.M t.N = 35 * Real.pi / 180)
  (h4 : angle t.K t.N t.M = 35 * Real.pi / 180)
  (h5 : angle t.L t.N t.M = 55 * Real.pi / 180)
  (h6 : angle t.K t.M t.N = 55 * Real.pi / 180)
  (h7 : isOnSphere s t.K ∧ isOnSphere s t.L ∧ isOnSphere s t.M ∧ isOnSphere s t.N) :
  sphericalArea s (distanceSet s t) = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_sphere_area_l2462_246238


namespace NUMINAMATH_CALUDE_triangle_inequality_l2462_246231

theorem triangle_inequality (m : ℝ) : m > 0 → (3 + 4 > m ∧ 3 + m > 4 ∧ 4 + m > 3) → m = 5 := by
  sorry

#check triangle_inequality

end NUMINAMATH_CALUDE_triangle_inequality_l2462_246231


namespace NUMINAMATH_CALUDE_jason_cantaloupes_l2462_246276

theorem jason_cantaloupes (total keith fred : ℕ) (h1 : total = 65) (h2 : keith = 29) (h3 : fred = 16) :
  total - keith - fred = 20 := by
  sorry

end NUMINAMATH_CALUDE_jason_cantaloupes_l2462_246276


namespace NUMINAMATH_CALUDE_intersection_implies_a_leq_neg_one_l2462_246268

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x^2 - 2*x - 3)}
def B (a : ℝ) : Set ℝ := {x | ∃ y, y = Real.sqrt (a - x)}

-- State the theorem
theorem intersection_implies_a_leq_neg_one (a : ℝ) : A ∩ B a = B a → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_leq_neg_one_l2462_246268


namespace NUMINAMATH_CALUDE_x_in_terms_of_z_l2462_246265

theorem x_in_terms_of_z (x y z : ℝ) 
  (eq1 : 0.35 * (400 + y) = 0.20 * x)
  (eq2 : x = 2 * z^2)
  (eq3 : y = 3 * z - 5) :
  x = 2 * z^2 := by
sorry

end NUMINAMATH_CALUDE_x_in_terms_of_z_l2462_246265


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2462_246283

theorem simplify_polynomial (y : ℝ) :
  (2*y - 1) * (4*y^10 + 2*y^9 + 4*y^8 + 2*y^7) = 8*y^11 + 6*y^9 - 2*y^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2462_246283


namespace NUMINAMATH_CALUDE_total_pupils_after_addition_l2462_246205

/-- Given a school with an initial number of girls and boys, and additional girls joining,
    calculate the total number of pupils after the new girls joined. -/
theorem total_pupils_after_addition (initial_girls initial_boys additional_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  additional_girls = 418 →
  initial_girls + initial_boys + additional_girls = 1346 := by
  sorry

end NUMINAMATH_CALUDE_total_pupils_after_addition_l2462_246205


namespace NUMINAMATH_CALUDE_smallest_non_square_product_of_four_primes_l2462_246271

/-- A function that checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that checks if a number is prime --/
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

/-- A function that checks if a number is the product of four primes --/
def is_product_of_four_primes (n : ℕ) : Prop :=
  ∃ p q r s : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧ n = p * q * r * s

theorem smallest_non_square_product_of_four_primes :
  (∀ m : ℕ, m < 24 → ¬(is_product_of_four_primes m ∧ ¬is_perfect_square m)) ∧
  (is_product_of_four_primes 24 ∧ ¬is_perfect_square 24) :=
sorry

end NUMINAMATH_CALUDE_smallest_non_square_product_of_four_primes_l2462_246271


namespace NUMINAMATH_CALUDE_logarithm_sum_l2462_246296

theorem logarithm_sum (a : ℝ) (h : 1 + a^3 = 9) : 
  Real.log a / Real.log (1/4) + Real.log 8 / Real.log a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_l2462_246296


namespace NUMINAMATH_CALUDE_sum_cube_inequality_l2462_246208

theorem sum_cube_inequality (x1 x2 x3 x4 : ℝ) 
  (h_pos1 : x1 > 0) (h_pos2 : x2 > 0) (h_pos3 : x3 > 0) (h_pos4 : x4 > 0)
  (h_cond1 : x1^3 + x3^3 + 3*x1*x3 = 1)
  (h_cond2 : x2 + x4 = 1) : 
  (x1 + 1/x1)^3 + (x2 + 1/x2)^3 + (x3 + 1/x3)^3 + (x4 + 1/x4)^3 ≥ 125/4 := by
sorry

end NUMINAMATH_CALUDE_sum_cube_inequality_l2462_246208


namespace NUMINAMATH_CALUDE_hardcover_non_fiction_count_l2462_246286

/-- Represents the number of books in Thabo's collection -/
def total_books : ℕ := 500

/-- Represents the fraction of fiction books in the collection -/
def fiction_fraction : ℚ := 2/5

/-- Represents the fraction of non-fiction books in the collection -/
def non_fiction_fraction : ℚ := 3/5

/-- Represents the difference between paperback and hardcover non-fiction books -/
def non_fiction_difference : ℕ := 50

/-- Represents the ratio of paperback to hardcover fiction books -/
def fiction_ratio : ℕ := 2

theorem hardcover_non_fiction_count :
  ∃ (hnf : ℕ), 
    (hnf : ℚ) + (hnf + non_fiction_difference : ℚ) = total_books * non_fiction_fraction ∧
    hnf = 125 := by
  sorry

end NUMINAMATH_CALUDE_hardcover_non_fiction_count_l2462_246286


namespace NUMINAMATH_CALUDE_no_natural_product_l2462_246203

theorem no_natural_product (n : ℕ) : ¬∃ (a b : ℕ), 3 * n + 1 = a * b := by
  sorry

end NUMINAMATH_CALUDE_no_natural_product_l2462_246203


namespace NUMINAMATH_CALUDE_diana_bike_time_l2462_246246

/-- Proves that Diana will take 6 hours to get home given the specified conditions -/
theorem diana_bike_time : 
  let total_distance : ℝ := 10
  let initial_speed : ℝ := 3
  let initial_time : ℝ := 2
  let tired_speed : ℝ := 1
  let initial_distance := initial_speed * initial_time
  let remaining_distance := total_distance - initial_distance
  let tired_time := remaining_distance / tired_speed
  initial_time + tired_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_diana_bike_time_l2462_246246


namespace NUMINAMATH_CALUDE_zero_exponent_rule_l2462_246215

theorem zero_exponent_rule (a b : ℤ) (hb : b ≠ 0) : (a / b : ℚ) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_rule_l2462_246215


namespace NUMINAMATH_CALUDE_function_overlap_with_inverse_l2462_246227

theorem function_overlap_with_inverse (a b c d : ℝ) (h1 : a ≠ 0 ∨ c ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ (a * x + b) / (c * x + d)
  (∀ x, f (f x) = x) →
  ((a + d = 0 ∧ ∃ k, f = λ x ↦ (k * x + b) / (c * x - k)) ∨ f = id) :=
by sorry

end NUMINAMATH_CALUDE_function_overlap_with_inverse_l2462_246227


namespace NUMINAMATH_CALUDE_complex_fraction_equals_two_l2462_246217

theorem complex_fraction_equals_two (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^2 + a*b + b^2 = 0) : 
  (a^12 + b^12) / (a + b)^12 = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_two_l2462_246217


namespace NUMINAMATH_CALUDE_k_values_l2462_246288

def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (1 - p.2) / (1 + p.1) = 3}

def B (k : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + 3}

theorem k_values (k : ℝ) : A ∩ B k = ∅ → k = 2 ∨ k = -3 := by
  sorry

end NUMINAMATH_CALUDE_k_values_l2462_246288


namespace NUMINAMATH_CALUDE_min_value_at_neg_pi_half_l2462_246250

/-- The function f(x) = x + 2cos(x) has its minimum value on the interval [-π/2, 0] at x = -π/2 -/
theorem min_value_at_neg_pi_half :
  let f : ℝ → ℝ := λ x ↦ x + 2 * Real.cos x
  let a : ℝ := -π/2
  let b : ℝ := 0
  ∀ x ∈ Set.Icc a b, f a ≤ f x := by
  sorry

end NUMINAMATH_CALUDE_min_value_at_neg_pi_half_l2462_246250


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2462_246233

theorem right_triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b < c) 
    (h4 : a^2 + b^2 = c^2) : 
  (1/a + 1/b + 1/c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) ∧ 
  ∃ (a' b' c' : ℝ), 0 < a' ∧ a' ≤ b' ∧ b' < c' ∧ a'^2 + b'^2 = c'^2 ∧
    (1/a' + 1/b' + 1/c') = (5 + 3 * Real.sqrt 2) / (a' + b' + c') := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2462_246233


namespace NUMINAMATH_CALUDE_total_cost_is_44_l2462_246289

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda in dollars -/
def soda_cost : ℕ := 3

/-- The cost of a single cookie in dollars -/
def cookie_cost : ℕ := 1

/-- The number of sandwiches to purchase -/
def num_sandwiches : ℕ := 4

/-- The number of sodas to purchase -/
def num_sodas : ℕ := 6

/-- The number of cookies to purchase -/
def num_cookies : ℕ := 10

/-- Theorem stating that the total cost of the purchase is $44 -/
theorem total_cost_is_44 :
  num_sandwiches * sandwich_cost + num_sodas * soda_cost + num_cookies * cookie_cost = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_44_l2462_246289


namespace NUMINAMATH_CALUDE_sequence_properties_l2462_246248

/-- Definition of sequence a_n -/
def a (n : ℕ) : ℝ := sorry

/-- Definition of S_n as the sum of first n terms of a_n -/
def S (n : ℕ) : ℝ := sorry

/-- Definition of sequence b_n -/
def b (n : ℕ) : ℝ := sorry

/-- Definition of sequence c_n -/
def c (n : ℕ) : ℝ := a n * b n

/-- Definition of T_n as the sum of first n terms of c_n -/
def T (n : ℕ) : ℝ := sorry

/-- Main theorem -/
theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = (S n + 2) / 2) ∧ 
  (b 1 = 1) ∧
  (∀ n : ℕ, n ≥ 1 → b n - b (n + 1) + 2 = 0) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2^n) ∧
  (∀ n : ℕ, n ≥ 1 → b n = 2*n - 1) ∧
  (∀ n : ℕ, n ≥ 1 → T n = (2*n - 3) * 2^(n+1) + 6) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l2462_246248


namespace NUMINAMATH_CALUDE_tenth_line_correct_l2462_246201

def ninthLine : String := "311311222113111231131112322211231231131112"

def countConsecutive (s : String) : String :=
  sorry

theorem tenth_line_correct : 
  countConsecutive ninthLine = "13211321322111312211" := by
  sorry

end NUMINAMATH_CALUDE_tenth_line_correct_l2462_246201


namespace NUMINAMATH_CALUDE_total_distance_is_62_l2462_246292

/-- Calculates the total distance walked over three days given specific conditions --/
def total_distance_walked (day1_distance : ℕ) (day1_speed : ℕ) : ℕ :=
  let day1_hours := day1_distance / day1_speed
  let day2_hours := day1_hours - 1
  let day2_speed := day1_speed + 1
  let day2_distance := day2_hours * day2_speed
  let day3_hours := day1_hours
  let day3_speed := day2_speed
  let day3_distance := day3_hours * day3_speed
  day1_distance + day2_distance + day3_distance

/-- Theorem stating that the total distance walked is 62 miles --/
theorem total_distance_is_62 : total_distance_walked 18 3 = 62 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_62_l2462_246292


namespace NUMINAMATH_CALUDE_fifteenth_triangular_sum_fifteenth_sixteenth_triangular_l2462_246273

/-- Triangular number sequence -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 15th triangular number is 120 -/
theorem fifteenth_triangular : triangular_number 15 = 120 := by sorry

/-- The sum of the 15th and 16th triangular numbers is 256 -/
theorem sum_fifteenth_sixteenth_triangular : 
  triangular_number 15 + triangular_number 16 = 256 := by sorry

end NUMINAMATH_CALUDE_fifteenth_triangular_sum_fifteenth_sixteenth_triangular_l2462_246273


namespace NUMINAMATH_CALUDE_allen_book_pages_l2462_246239

/-- Calculates the total number of pages in a book based on daily reading rate and days to finish -/
def total_pages (pages_per_day : ℕ) (days_to_finish : ℕ) : ℕ :=
  pages_per_day * days_to_finish

/-- Proves that Allen's book has 120 pages given his reading rate and time to finish -/
theorem allen_book_pages :
  let pages_per_day : ℕ := 10
  let days_to_finish : ℕ := 12
  total_pages pages_per_day days_to_finish = 120 := by
  sorry

end NUMINAMATH_CALUDE_allen_book_pages_l2462_246239


namespace NUMINAMATH_CALUDE_correct_representation_l2462_246264

/-- Represents "a number that is 3 more than twice x" -/
def number_3_more_than_twice_x (x : ℝ) : ℝ := 2 * x + 3

/-- The algebraic expression 2x + 3 correctly represents "a number that is 3 more than twice x" -/
theorem correct_representation (x : ℝ) :
  number_3_more_than_twice_x x = 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_representation_l2462_246264


namespace NUMINAMATH_CALUDE_cart_distance_theorem_l2462_246253

/-- Represents a cart with two wheels -/
structure Cart where
  front_wheel_circumference : ℝ
  back_wheel_circumference : ℝ

/-- Calculates the distance traveled by the cart -/
def distance_traveled (c : Cart) (back_wheel_revolutions : ℝ) : ℝ :=
  c.back_wheel_circumference * back_wheel_revolutions

theorem cart_distance_theorem (c : Cart) (back_wheel_revolutions : ℝ) :
  c.front_wheel_circumference = 30 →
  c.back_wheel_circumference = 33 →
  c.front_wheel_circumference * (back_wheel_revolutions + 5) = c.back_wheel_circumference * back_wheel_revolutions →
  distance_traveled c back_wheel_revolutions = 1650 := by
  sorry

#check cart_distance_theorem

end NUMINAMATH_CALUDE_cart_distance_theorem_l2462_246253


namespace NUMINAMATH_CALUDE_six_digit_number_divisible_by_7_8_9_l2462_246285

theorem six_digit_number_divisible_by_7_8_9 : ∃ (n₁ n₂ : ℕ),
  n₁ ≠ n₂ ∧
  523000 ≤ n₁ ∧ n₁ < 524000 ∧
  523000 ≤ n₂ ∧ n₂ < 524000 ∧
  n₁ % 7 = 0 ∧ n₁ % 8 = 0 ∧ n₁ % 9 = 0 ∧
  n₂ % 7 = 0 ∧ n₂ % 8 = 0 ∧ n₂ % 9 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_divisible_by_7_8_9_l2462_246285


namespace NUMINAMATH_CALUDE_christopher_age_l2462_246226

theorem christopher_age (c g : ℕ) : 
  c = 2 * g ∧ 
  c - 9 = 5 * (g - 9) → 
  c = 24 := by
sorry

end NUMINAMATH_CALUDE_christopher_age_l2462_246226


namespace NUMINAMATH_CALUDE_log_equation_solution_l2462_246251

theorem log_equation_solution :
  ∃! x : ℝ, (Real.log (Real.sqrt (7 * x + 3)) + Real.log (Real.sqrt (4 * x + 5)) = 1 / 2 + Real.log 3) ∧
             (7 * x + 3 > 0) ∧ (4 * x + 5 > 0) := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l2462_246251


namespace NUMINAMATH_CALUDE_remainder_divisibility_l2462_246277

theorem remainder_divisibility (x : ℤ) : 
  (∃ k : ℤ, x = 63 * k + 27) → (∃ m : ℤ, x = 8 * m + 3) :=
by sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l2462_246277


namespace NUMINAMATH_CALUDE_max_value_of_z_l2462_246259

-- Define the objective function
def z (x y : ℝ) : ℝ := 4 * x + 3 * y

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  x - y - 2 ≥ 0 ∧ 2 * x + y - 2 ≤ 0 ∧ y + 4 ≥ 0

-- Theorem statement
theorem max_value_of_z :
  ∃ (max : ℝ), max = 8 ∧
  (∀ x y : ℝ, feasible_region x y → z x y ≤ max) ∧
  (∃ x y : ℝ, feasible_region x y ∧ z x y = max) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2462_246259


namespace NUMINAMATH_CALUDE_sqrt_simplification_l2462_246252

theorem sqrt_simplification : (Real.sqrt 2 * Real.sqrt 20) / Real.sqrt 5 = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l2462_246252


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_eleven_l2462_246229

theorem sum_of_digits_of_seven_to_eleven (n : ℕ) : 
  (3 + 4)^11 % 100 = 43 → 
  (((3 + 4)^11 / 10) % 10 + (3 + 4)^11 % 10) = 7 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_eleven_l2462_246229


namespace NUMINAMATH_CALUDE_power_equation_solution_l2462_246257

theorem power_equation_solution :
  ∃ x : ℕ, (1000 : ℝ)^7 / (10 : ℝ)^x = 10000 ∧ x = 17 := by sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2462_246257


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2462_246245

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) ↔ (∀ x : ℝ, x^2 + 2*x + 5 ≠ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l2462_246245


namespace NUMINAMATH_CALUDE_monotonic_function_characterization_l2462_246261

-- Define the types of our functions
def MonotonicFunction (f : ℝ → ℝ) : Prop := 
  ∀ x y, x ≤ y → f x ≤ f y

def StrictlyMonotonicFunction (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem monotonic_function_characterization 
  (u : ℝ → ℝ) 
  (h_u_monotonic : MonotonicFunction u) 
  (h_exists_f : ∃ f : ℝ → ℝ, 
    StrictlyMonotonicFunction f ∧ 
    (∀ x y : ℝ, f (x + y) = f x * u y + f y)) : 
  ∃ k : ℝ, ∀ x : ℝ, u x = Real.exp (k * x) := by
sorry

end NUMINAMATH_CALUDE_monotonic_function_characterization_l2462_246261


namespace NUMINAMATH_CALUDE_unique_quadratic_root_condition_l2462_246244

theorem unique_quadratic_root_condition (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 1/b) * x + c = 0) ↔ 
  c = 3/2 := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_condition_l2462_246244


namespace NUMINAMATH_CALUDE_fabric_cutting_l2462_246291

theorem fabric_cutting (fabric_length fabric_width dress_length dress_width : ℕ) 
  (h1 : fabric_length = 140)
  (h2 : fabric_width = 75)
  (h3 : dress_length = 45)
  (h4 : dress_width = 26)
  : ∃ (n : ℕ), n ≥ 8 ∧ n * dress_length * dress_width ≤ fabric_length * fabric_width := by
  sorry

end NUMINAMATH_CALUDE_fabric_cutting_l2462_246291


namespace NUMINAMATH_CALUDE_intersection_A_B_when_a_is_one_range_of_a_when_B_subset_complementA_l2462_246221

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | (a * x - 1) * (x + 2) ≥ 0}

-- Define the complement of A
def complementA : Set ℝ := {x | x ≤ -1 ∨ x ≥ 2}

-- Theorem for part (1)
theorem intersection_A_B_when_a_is_one :
  A ∩ B 1 = {x | 1 ≤ x ∧ x < 2} := by sorry

-- Theorem for part (2)
theorem range_of_a_when_B_subset_complementA :
  ∀ a > 0, (B a ⊆ complementA) ↔ (0 < a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_when_a_is_one_range_of_a_when_B_subset_complementA_l2462_246221


namespace NUMINAMATH_CALUDE_set_intersection_and_union_l2462_246256

def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - (a+2)*x + 2*a^2 - a + 1 = 0}

theorem set_intersection_and_union (a : ℝ) :
  (A ∩ B a = {2} → a = 1/2) ∧
  (A ∪ B a = A → a ≤ 0 ∨ a = 1 ∨ a > 8/7) := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_and_union_l2462_246256


namespace NUMINAMATH_CALUDE_square_areas_and_perimeters_l2462_246298

theorem square_areas_and_perimeters (x : ℝ) : 
  (x^2 + 8*x + 16 = (x + 4)^2) ∧ 
  (4*x^2 - 12*x + 9 = (2*x - 3)^2) ∧ 
  (4*(x + 4) + 4*(2*x - 3) = 32) → 
  x = 7/3 := by
sorry

end NUMINAMATH_CALUDE_square_areas_and_perimeters_l2462_246298


namespace NUMINAMATH_CALUDE_decimal_to_base5_l2462_246207

theorem decimal_to_base5 :
  ∃ (a b c : ℕ), a < 5 ∧ b < 5 ∧ c < 5 ∧ 88 = c * 5^2 + b * 5^1 + a * 5^0 ∧ 
  (a = 3 ∧ b = 2 ∧ c = 3) := by
sorry

end NUMINAMATH_CALUDE_decimal_to_base5_l2462_246207


namespace NUMINAMATH_CALUDE_rectangle_arrangement_probability_l2462_246204

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a square -/
structure Square where
  side_length : ℕ

/-- Represents a line segment connecting midpoints of opposite sides of a square -/
structure MidpointLine where
  square : Square

/-- Represents an arrangement of rectangles in a square -/
structure Arrangement where
  square : Square
  rectangles : List Rectangle

/-- Checks if an arrangement is valid (no overlapping rectangles) -/
def is_valid_arrangement (arr : Arrangement) : Prop := sorry

/-- Checks if an arrangement crosses the midpoint line -/
def crosses_midpoint_line (arr : Arrangement) (line : MidpointLine) : Prop := sorry

/-- Counts the number of valid arrangements -/
def count_valid_arrangements (square : Square) (rect_type : Rectangle) (num_rect : ℕ) : ℕ := sorry

/-- Counts the number of valid arrangements that don't cross the midpoint line -/
def count_non_crossing_arrangements (square : Square) (rect_type : Rectangle) (num_rect : ℕ) (line : MidpointLine) : ℕ := sorry

theorem rectangle_arrangement_probability :
  let square := Square.mk 4
  let rect_type := Rectangle.mk 1 2
  let num_rect := 8
  let line := MidpointLine.mk square
  let total_arrangements := count_valid_arrangements square rect_type num_rect
  let non_crossing_arrangements := count_non_crossing_arrangements square rect_type num_rect line
  (non_crossing_arrangements : ℚ) / total_arrangements = 25 / 36 := by sorry

end NUMINAMATH_CALUDE_rectangle_arrangement_probability_l2462_246204


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l2462_246274

theorem nested_sqrt_value : 
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l2462_246274


namespace NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l2462_246255

theorem sqrt_eight_minus_sqrt_two_equals_sqrt_two : 
  Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_minus_sqrt_two_equals_sqrt_two_l2462_246255


namespace NUMINAMATH_CALUDE_solution_in_interval_l2462_246212

theorem solution_in_interval : ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x + x - 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_interval_l2462_246212


namespace NUMINAMATH_CALUDE_square_equation_solution_l2462_246216

theorem square_equation_solution : ∃ x : ℝ, (12 - x)^2 = (x + 3)^2 ∧ x = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2462_246216


namespace NUMINAMATH_CALUDE_tenth_power_sum_l2462_246242

theorem tenth_power_sum (a b : ℝ) (h1 : a + b = 1) (h2 : a^2 + b^2 = 3) : a^10 + b^10 = 93 := by
  sorry

end NUMINAMATH_CALUDE_tenth_power_sum_l2462_246242


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2462_246243

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 864 → volume = 1728 → 
  (∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧ 
    volume = side_length^3) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2462_246243


namespace NUMINAMATH_CALUDE_triangle_properties_l2462_246232

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The angle C in radians -/
def angle_C (t : Triangle) : ℝ :=
  sorry

/-- The angle B in radians -/
def angle_B (t : Triangle) : ℝ :=
  sorry

theorem triangle_properties (t : Triangle) 
  (ha : t.a = 3) 
  (hb : t.b = 5) 
  (hc : t.c = 7) : 
  (angle_C t = 2 * Real.pi / 3) ∧ 
  (Real.sin (angle_B t + Real.pi / 3) = 4 * Real.sqrt 3 / 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2462_246232


namespace NUMINAMATH_CALUDE_class_gender_ratio_l2462_246241

theorem class_gender_ratio :
  ∀ (girls boys : ℕ),
  girls = boys + 6 →
  girls + boys = 36 →
  (girls : ℚ) / (boys : ℚ) = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l2462_246241


namespace NUMINAMATH_CALUDE_parallel_line_plane_l2462_246290

-- Define the types for lines and planes
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (line_parallel : Line → Plane → Prop)

-- Theorem statement
theorem parallel_line_plane 
  (α β : Plane) (n : Line) 
  (h1 : parallel α β) 
  (h2 : subset n α) : 
  line_parallel n β :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_l2462_246290


namespace NUMINAMATH_CALUDE_number_difference_l2462_246278

theorem number_difference (n : ℕ) (h : n = 15) : n * 13 - n = 180 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2462_246278


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2462_246219

def P (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def Q (a : ℝ) : Set ℝ := {a^2+1, 2*a-1, a-3}

theorem intersection_implies_a_value :
  ∀ a : ℝ, P a ∩ Q a = {-3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2462_246219


namespace NUMINAMATH_CALUDE_median_exists_for_seven_prices_l2462_246275

theorem median_exists_for_seven_prices (prices : List ℝ) (h : prices.length = 7) :
  ∃ (median : ℝ), median ∈ prices ∧ 
    (prices.filter (λ x => x ≤ median)).length ≥ 4 ∧
    (prices.filter (λ x => x ≥ median)).length ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_median_exists_for_seven_prices_l2462_246275


namespace NUMINAMATH_CALUDE_problem_solution_l2462_246258

theorem problem_solution (p q r : ℝ) 
  (h1 : p / q = 5 / 4)
  (h2 : p = r^2)
  (h3 : Real.sin r = 3 / 5) : 
  2 * p + q = 44.8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2462_246258


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2462_246234

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (ha : a = 5) 
  (hb : b = 7) 
  (hc : c = 10) : 
  a + b > c ∧ b + c > a ∧ c + a > b := by sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2462_246234


namespace NUMINAMATH_CALUDE_oranges_in_bin_l2462_246222

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (added : ℕ) :
  initial ≥ thrown_away →
  initial - thrown_away + added = initial + added - thrown_away :=
by sorry

end NUMINAMATH_CALUDE_oranges_in_bin_l2462_246222
