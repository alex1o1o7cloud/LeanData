import Mathlib

namespace NUMINAMATH_CALUDE_complete_factorization_w4_minus_81_l3753_375309

theorem complete_factorization_w4_minus_81 (w : ℝ) : 
  w^4 - 81 = (w - 3) * (w + 3) * (w^2 + 9) := by
  sorry

end NUMINAMATH_CALUDE_complete_factorization_w4_minus_81_l3753_375309


namespace NUMINAMATH_CALUDE_five_solutions_l3753_375307

/-- The system of equations has exactly 5 real solutions -/
theorem five_solutions (x y z w θ : ℝ) : 
  x = 2*z + 2*w + z*w*x →
  y = 2*w + 2*x + w*x*y →
  z = 2*x + 2*y + x*y*z →
  w = 2*y + 2*z + y*z*w →
  w = Real.sin θ ^ 2 →
  ∃! (s : Finset (ℝ × ℝ × ℝ × ℝ)), s.card = 5 ∧ ∀ (a b c d : ℝ), (a, b, c, d) ∈ s ↔ 
    (a = 2*c + 2*d + c*d*a ∧
     b = 2*d + 2*a + d*a*b ∧
     c = 2*a + 2*b + a*b*c ∧
     d = 2*b + 2*c + b*c*d ∧
     d = Real.sin θ ^ 2) :=
by sorry


end NUMINAMATH_CALUDE_five_solutions_l3753_375307


namespace NUMINAMATH_CALUDE_max_value_of_f_l3753_375349

-- Define the function
def f (x : ℝ) : ℝ := x * abs x - 2 * x + 1

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 16 ∧ ∀ x : ℝ, |x + 1| ≤ 6 → f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3753_375349


namespace NUMINAMATH_CALUDE_order_of_numbers_l3753_375365

theorem order_of_numbers : Real.log 0.76 < 0.76 ∧ 0.76 < 60.7 := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3753_375365


namespace NUMINAMATH_CALUDE_committee_probability_l3753_375323

def total_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def committee_size : ℕ := 5

def probability_at_least_one_of_each : ℚ := 1705 / 1771

theorem committee_probability :
  let total_committees := Nat.choose total_members committee_size
  let all_one_gender := Nat.choose boys committee_size + Nat.choose girls committee_size
  (1 : ℚ) - (all_one_gender : ℚ) / (total_committees : ℚ) = probability_at_least_one_of_each :=
sorry

end NUMINAMATH_CALUDE_committee_probability_l3753_375323


namespace NUMINAMATH_CALUDE_arcade_candy_cost_l3753_375398

theorem arcade_candy_cost (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (candies : ℕ) :
  whack_a_mole_tickets = 8 →
  skee_ball_tickets = 7 →
  candies = 3 →
  (whack_a_mole_tickets + skee_ball_tickets) / candies = 5 :=
by sorry

end NUMINAMATH_CALUDE_arcade_candy_cost_l3753_375398


namespace NUMINAMATH_CALUDE_alex_corn_purchase_l3753_375313

/-- The price of corn per pound -/
def corn_price : ℝ := 1.20

/-- The price of beans per pound -/
def bean_price : ℝ := 0.60

/-- The total number of pounds of corn and beans bought -/
def total_pounds : ℝ := 30

/-- The total cost of the purchase -/
def total_cost : ℝ := 27.00

/-- The amount of corn bought in pounds -/
def corn_amount : ℝ := 15.0

theorem alex_corn_purchase :
  ∃ (bean_amount : ℝ),
    corn_amount + bean_amount = total_pounds ∧
    corn_price * corn_amount + bean_price * bean_amount = total_cost :=
by
  sorry

end NUMINAMATH_CALUDE_alex_corn_purchase_l3753_375313


namespace NUMINAMATH_CALUDE_circle_tangent_sum_radii_l3753_375359

theorem circle_tangent_sum_radii : 
  ∀ r : ℝ, 
  (r > 0) →
  ((r - 4)^2 + r^2 = (r + 2)^2) →
  (∃ r₁ r₂ : ℝ, (r = r₁ ∨ r = r₂) ∧ r₁ + r₂ = 12) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_sum_radii_l3753_375359


namespace NUMINAMATH_CALUDE_train_cars_problem_l3753_375387

theorem train_cars_problem (passenger_cars cargo_cars : ℕ) : 
  cargo_cars = passenger_cars / 2 + 3 →
  passenger_cars + cargo_cars + 2 = 71 →
  passenger_cars = 44 := by
  sorry

end NUMINAMATH_CALUDE_train_cars_problem_l3753_375387


namespace NUMINAMATH_CALUDE_horner_method_polynomial_evaluation_l3753_375340

def horner_method (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

def polynomial (x : ℝ) : ℝ :=
  3 * x^4 - x^2 + 2 * x + 1

theorem horner_method_polynomial_evaluation :
  let coeffs := [3, 0, -1, 2, 1]
  let x := 2
  let v₃ := (horner_method (coeffs.take 4) x)
  v₃ = 22 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_polynomial_evaluation_l3753_375340


namespace NUMINAMATH_CALUDE_scientific_notation_of_56_5_million_l3753_375390

theorem scientific_notation_of_56_5_million :
  56500000 = 5.65 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_56_5_million_l3753_375390


namespace NUMINAMATH_CALUDE_l_shape_perimeter_l3753_375335

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents the L-shape configuration -/
structure LShape where
  vertical : Rectangle
  horizontal : Rectangle
  overlap : ℝ

/-- Calculates the perimeter of the L-shape -/
def LShape.perimeter (l : LShape) : ℝ :=
  l.vertical.perimeter + l.horizontal.perimeter - 2 * l.overlap

theorem l_shape_perimeter :
  let l : LShape := {
    vertical := { width := 3, height := 6 },
    horizontal := { width := 4, height := 2 },
    overlap := 1
  }
  l.perimeter = 28 := by sorry

end NUMINAMATH_CALUDE_l_shape_perimeter_l3753_375335


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3753_375343

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 2) / a (n + 1) = a (n + 1) / a n
  sum_1_2 : a 1 + a 2 = 30
  sum_3_4 : a 3 + a 4 = 60

/-- The theorem stating that a_7 + a_8 = 240 for the given geometric sequence -/
theorem geometric_sequence_sum (seq : GeometricSequence) : seq.a 7 + seq.a 8 = 240 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3753_375343


namespace NUMINAMATH_CALUDE_min_ttetrominoes_on_chessboard_l3753_375379

/-- Represents a chessboard as an 8x8 grid -/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Represents a T-tetromino -/
structure TTetromino where
  center : Fin 8 × Fin 8
  orientation : Fin 4

/-- Checks if a T-tetromino can be placed on the board -/
def canPlaceTTetromino (board : Chessboard) (t : TTetromino) : Bool :=
  sorry

/-- Places a T-tetromino on the board -/
def placeTTetromino (board : Chessboard) (t : TTetromino) : Chessboard :=
  sorry

/-- Checks if any T-tetromino can be placed on the board -/
def canPlaceAnyTTetromino (board : Chessboard) : Bool :=
  sorry

/-- The main theorem stating that 7 is the minimum number of T-tetrominoes -/
theorem min_ttetrominoes_on_chessboard :
  ∀ (n : Nat),
    (∃ (board : Chessboard) (tetrominoes : List TTetromino),
      tetrominoes.length = n ∧
      (∀ t ∈ tetrominoes, canPlaceTTetromino board t) ∧
      ¬canPlaceAnyTTetromino (tetrominoes.foldl placeTTetromino board)) →
    n ≥ 7 :=
  sorry

end NUMINAMATH_CALUDE_min_ttetrominoes_on_chessboard_l3753_375379


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l3753_375337

theorem fraction_product_simplification : 
  (2 : ℚ) / 3 * 3 / 4 * 4 / 5 * 5 / 6 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l3753_375337


namespace NUMINAMATH_CALUDE_sqrt_144_div_6_l3753_375327

theorem sqrt_144_div_6 : Real.sqrt 144 / 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_144_div_6_l3753_375327


namespace NUMINAMATH_CALUDE_excircle_radius_eq_semiperimeter_implies_right_angle_l3753_375356

/-- A triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- The excircle of a triangle -/
structure Excircle (T : Triangle) where
  center : Point
  radius : ℝ

/-- The semiperimeter of a triangle -/
def semiperimeter (T : Triangle) : ℝ := sorry

/-- A triangle is right-angled -/
def is_right_angled (T : Triangle) : Prop := sorry

/-- Main theorem: If the radius of the excircle equals the semiperimeter, 
    then the triangle is right-angled -/
theorem excircle_radius_eq_semiperimeter_implies_right_angle 
  (T : Triangle) (E : Excircle T) : 
  E.radius = semiperimeter T → is_right_angled T := by sorry

end NUMINAMATH_CALUDE_excircle_radius_eq_semiperimeter_implies_right_angle_l3753_375356


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3753_375362

theorem min_value_quadratic (a : ℝ) : a^2 - 4*a + 9 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3753_375362


namespace NUMINAMATH_CALUDE_jelly_bean_division_l3753_375301

theorem jelly_bean_division (initial_amount : ℕ) (eaten_amount : ℕ) (num_piles : ℕ) 
  (h1 : initial_amount = 36)
  (h2 : eaten_amount = 6)
  (h3 : num_piles = 3)
  (h4 : initial_amount > eaten_amount) :
  (initial_amount - eaten_amount) / num_piles = 10 := by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_division_l3753_375301


namespace NUMINAMATH_CALUDE_black_cells_remain_even_one_black_cell_impossible_l3753_375369

/-- Represents a chessboard -/
structure Chessboard :=
  (black_cells : ℕ)

/-- Represents a repainting operation on a 2x2 square -/
def repaint (board : Chessboard) : Chessboard :=
  { black_cells := board.black_cells + (4 - 2 * (board.black_cells % 4)) }

/-- Initial chessboard state -/
def initial_board : Chessboard :=
  { black_cells := 32 }

/-- Theorem stating that the number of black cells remains even after any number of repainting operations -/
theorem black_cells_remain_even (n : ℕ) :
  ∀ (board : Chessboard),
  (board.black_cells % 2 = 0) →
  ((repaint^[n] board).black_cells % 2 = 0) :=
sorry

/-- Main theorem: It's impossible to have exactly one black cell after repainting operations -/
theorem one_black_cell_impossible :
  ¬ ∃ (n : ℕ), (repaint^[n] initial_board).black_cells = 1 :=
sorry

end NUMINAMATH_CALUDE_black_cells_remain_even_one_black_cell_impossible_l3753_375369


namespace NUMINAMATH_CALUDE_cookies_to_mike_is_23_l3753_375308

/-- The number of cookies Uncle Jude gave to Mike -/
def cookies_to_mike (total cookies_to_tim cookies_in_fridge : ℕ) : ℕ :=
  total - (cookies_to_tim + 2 * cookies_to_tim + cookies_in_fridge)

/-- Theorem: Uncle Jude gave 23 cookies to Mike -/
theorem cookies_to_mike_is_23 :
  cookies_to_mike 256 15 188 = 23 := by
  sorry

end NUMINAMATH_CALUDE_cookies_to_mike_is_23_l3753_375308


namespace NUMINAMATH_CALUDE_greatest_satisfying_n_l3753_375364

-- Define the sum of the first n positive integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the factorial of n
def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define the primality check
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the condition for n
def satisfies_condition (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧
  is_prime (n + 2) ∧
  ¬(factorial n % sum_first_n n = 0)

-- Theorem statement
theorem greatest_satisfying_n :
  satisfies_condition 995 ∧
  ∀ m, satisfies_condition m → m ≤ 995 :=
sorry

end NUMINAMATH_CALUDE_greatest_satisfying_n_l3753_375364


namespace NUMINAMATH_CALUDE_expand_expression_l3753_375346

theorem expand_expression (x y : ℝ) : 5 * (4 * x^2 + 3 * x * y - 4) = 20 * x^2 + 15 * x * y - 20 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l3753_375346


namespace NUMINAMATH_CALUDE_initial_number_count_l3753_375361

theorem initial_number_count (n : ℕ) (S : ℝ) : 
  S / n = 62 →
  (S - 45 - 55) / (n - 2) = 62.5 →
  n = 50 := by
sorry

end NUMINAMATH_CALUDE_initial_number_count_l3753_375361


namespace NUMINAMATH_CALUDE_u_1990_equals_one_l3753_375329

def u : ℕ → ℕ
  | 0 => 0
  | n + 1 => if n % 2 = 0 then 1 - u (n / 2) else u (n / 2)

theorem u_1990_equals_one : u 1990 = 1 := by
  sorry

end NUMINAMATH_CALUDE_u_1990_equals_one_l3753_375329


namespace NUMINAMATH_CALUDE_function_always_negative_implies_a_range_l3753_375375

theorem function_always_negative_implies_a_range 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h : ∀ x ∈ Set.Ioo 0 1, f x < 0) 
  (h_def : ∀ x, f x = x * |x - a| - 2) : 
  -1 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_function_always_negative_implies_a_range_l3753_375375


namespace NUMINAMATH_CALUDE_dans_balloons_l3753_375324

theorem dans_balloons (fred_balloons sam_balloons total_balloons : ℕ) 
  (h1 : fred_balloons = 10)
  (h2 : sam_balloons = 46)
  (h3 : total_balloons = 72) :
  total_balloons - (fred_balloons + sam_balloons) = 16 := by
  sorry

end NUMINAMATH_CALUDE_dans_balloons_l3753_375324


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3753_375315

theorem complex_equation_solution (z : ℂ) : Complex.I * (z - 1) = 1 + Complex.I * Complex.I → z = 2 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3753_375315


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l3753_375393

/-- The quadratic function f(x) = 2(x-1)^2 - 3 -/
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 - 3

/-- The x-coordinate of the vertex of f -/
def vertex_x : ℝ := 1

/-- The y-coordinate of the vertex of f -/
def vertex_y : ℝ := -3

/-- Theorem: The vertex of the quadratic function f(x) = 2(x-1)^2 - 3 is (1, -3) -/
theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f vertex_x) ∧ f vertex_x = vertex_y :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l3753_375393


namespace NUMINAMATH_CALUDE_unique_real_solution_and_two_imaginary_l3753_375370

-- Define the system of equations
def equation1 (x y : ℂ) : Prop := y = (x - 1)^2
def equation2 (x y : ℂ) : Prop := x * y + y = 2

-- Define a solution pair
def is_solution (x y : ℂ) : Prop := equation1 x y ∧ equation2 x y

-- Define the set of all solution pairs
def solution_set : Set (ℂ × ℂ) := {p | is_solution p.1 p.2}

-- State the theorem
theorem unique_real_solution_and_two_imaginary :
  ∃! (x y : ℝ), is_solution x y ∧
  ∃ (a b c d : ℝ), a ≠ 0 ∧ c ≠ 0 ∧
    is_solution (x + a * I) (y + b * I) ∧
    is_solution (x + c * I) (y + d * I) ∧
    (x + a * I ≠ x + c * I) ∧
    (∀ (u v : ℂ), is_solution u v → (u = x ∧ v = y) ∨ 
                                    (u = x + a * I ∧ v = y + b * I) ∨ 
                                    (u = x + c * I ∧ v = y + d * I)) :=
by sorry

end NUMINAMATH_CALUDE_unique_real_solution_and_two_imaginary_l3753_375370


namespace NUMINAMATH_CALUDE_S_bounds_l3753_375368

theorem S_bounds (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  let S := Real.sqrt (a * b / ((b + c) * (c + a))) +
           Real.sqrt (b * c / ((a + c) * (b + a))) +
           Real.sqrt (c * a / ((b + c) * (b + a)))
  1 ≤ S ∧ S ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_S_bounds_l3753_375368


namespace NUMINAMATH_CALUDE_speed_gain_per_week_baseball_training_speed_gain_l3753_375342

/-- Calculates the speed gained per week given initial speed, training details, and final speed increase. -/
theorem speed_gain_per_week 
  (initial_speed : ℝ) 
  (training_sessions : ℕ) 
  (weeks_per_session : ℕ) 
  (speed_increase_percent : ℝ) : ℝ :=
  let final_speed := initial_speed * (1 + speed_increase_percent / 100)
  let total_speed_gain := final_speed - initial_speed
  let total_weeks := training_sessions * weeks_per_session
  total_speed_gain / total_weeks

/-- Proves that the speed gained per week is 1 mph under the given conditions. -/
theorem baseball_training_speed_gain :
  speed_gain_per_week 80 4 4 20 = 1 := by
  sorry

end NUMINAMATH_CALUDE_speed_gain_per_week_baseball_training_speed_gain_l3753_375342


namespace NUMINAMATH_CALUDE_papa_carlo_solution_l3753_375354

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : minutes < 60

/-- Represents a clock with its displayed time and offset -/
structure Clock where
  displayed_time : Time
  offset : Int

def Papa_Carlo_problem (clocks : Vector Clock 4) : Prop :=
  ∃ (correct_time : Time),
    (clocks.get 0).offset = -2 ∧
    (clocks.get 1).offset = -3 ∧
    (clocks.get 2).offset = 4 ∧
    (clocks.get 3).offset = 5 ∧
    (clocks.get 0).displayed_time = Time.mk 14 54 (by norm_num) ∧
    (clocks.get 1).displayed_time = Time.mk 14 57 (by norm_num) ∧
    (clocks.get 2).displayed_time = Time.mk 15 2 (by norm_num) ∧
    (clocks.get 3).displayed_time = Time.mk 15 3 (by norm_num) ∧
    correct_time = Time.mk 14 59 (by norm_num)

theorem papa_carlo_solution (clocks : Vector Clock 4) 
  (h : Papa_Carlo_problem clocks) : 
  ∃ (correct_time : Time), correct_time = Time.mk 14 59 (by norm_num) :=
by sorry

end NUMINAMATH_CALUDE_papa_carlo_solution_l3753_375354


namespace NUMINAMATH_CALUDE_smallest_result_is_16_l3753_375317

def S : Finset Nat := {2, 3, 5, 7, 11, 13}

theorem smallest_result_is_16 :
  ∃ (a b c : Nat), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a + b) * c = 16 ∧
  ∀ (x y z : Nat), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  (x + y) * z ≥ 16 := by
sorry

end NUMINAMATH_CALUDE_smallest_result_is_16_l3753_375317


namespace NUMINAMATH_CALUDE_smallest_possible_a_l3753_375302

theorem smallest_possible_a (a b c : ℚ) :
  a > 0 ∧
  (∃ n : ℚ, a + b + c = n) ∧
  (∀ x y : ℚ, y = a * x^2 + b * x + c ↔ y + 2/3 = a * (x - 1/3)^2) →
  ∀ a' : ℚ, (a' > 0 ∧
    (∃ b' c' : ℚ, (∃ n : ℚ, a' + b' + c' = n) ∧
    (∀ x y : ℚ, y = a' * x^2 + b' * x + c' ↔ y + 2/3 = a' * (x - 1/3)^2))) →
  a ≤ a' ∧ a = 3/8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l3753_375302


namespace NUMINAMATH_CALUDE_remainder_96_104_div_9_l3753_375373

theorem remainder_96_104_div_9 : (96 * 104) % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_96_104_div_9_l3753_375373


namespace NUMINAMATH_CALUDE_problem_solution_l3753_375382

def A (a : ℝ) : ℝ := a + 2
def B (a : ℝ) : ℝ := 2 * a^2 - 3 * a + 10
def C (a : ℝ) : ℝ := a^2 + 5 * a - 3

theorem problem_solution :
  (∀ a : ℝ, A a < B a) ∧
  (∀ a : ℝ, (a < -5 ∨ a > 1) → C a > A a) ∧
  (∀ a : ℝ, (a = -5 ∨ a = 1) → C a = A a) ∧
  (∀ a : ℝ, (-5 < a ∧ a < 1) → C a < A a) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3753_375382


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l3753_375353

theorem triangle_third_side_length 
  (a b : ℝ) 
  (cos_C : ℝ) 
  (h1 : a = 5) 
  (h2 : b = 3) 
  (h3 : cos_C = -3/5) : 
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2*a*b*cos_C ∧ c = 2 * Real.sqrt 13 :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l3753_375353


namespace NUMINAMATH_CALUDE_value_of_X_l3753_375388

theorem value_of_X : ∃ X : ℚ, (1/3 : ℚ) * (1/4 : ℚ) * X = (1/4 : ℚ) * (1/6 : ℚ) * 120 ∧ X = 60 := by
  sorry

end NUMINAMATH_CALUDE_value_of_X_l3753_375388


namespace NUMINAMATH_CALUDE_sum_of_inverse_points_l3753_375385

/-- Given an invertible function f, if f(a) = 3 and f(b) = 7, then a + b = 0 -/
theorem sum_of_inverse_points (f : ℝ → ℝ) (a b : ℝ) 
  (h_inv : Function.Injective f) 
  (h_a : f a = 3) 
  (h_b : f b = 7) : 
  a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_inverse_points_l3753_375385


namespace NUMINAMATH_CALUDE_quadratic_value_l3753_375383

/-- A quadratic function with specific properties -/
def f (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_value (a b c : ℝ) :
  (∀ x, f a b c x ≤ 8) ∧  -- maximum value is 8
  (f a b c (-2) = 8) ∧    -- maximum occurs at x = -2
  (f a b c 1 = 4) →       -- passes through (1, 4)
  f a b c (-3) = 68/9 :=  -- value at x = -3 is 68/9
by sorry

end NUMINAMATH_CALUDE_quadratic_value_l3753_375383


namespace NUMINAMATH_CALUDE_triangle_sinC_l3753_375341

theorem triangle_sinC (A B C : ℝ) (hABC : A + B + C = π) 
  (hAC : 3 = 2 * Real.sqrt 3 * (Real.sin A / Real.sin B))
  (hA : A = 2 * B) : 
  Real.sin C = Real.sqrt 6 / 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_sinC_l3753_375341


namespace NUMINAMATH_CALUDE_polynomial_roots_l3753_375399

theorem polynomial_roots : ∃ (x : ℝ), x^5 - 3*x^4 + 3*x^2 - x - 6 = 0 ↔ x = -1 ∨ x = 1 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3753_375399


namespace NUMINAMATH_CALUDE_second_point_x_coordinate_l3753_375378

/-- Given two points on a line, prove the x-coordinate of the second point -/
theorem second_point_x_coordinate 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) -- First point (m, n) satisfies the line equation
  (h2 : m + 2 = 2 * (n + 1) + 5) -- Second point (m+2, n+1) satisfies the line equation
  : m + 2 = 2 * n + 7 := by
  sorry

end NUMINAMATH_CALUDE_second_point_x_coordinate_l3753_375378


namespace NUMINAMATH_CALUDE_angle_triple_complement_l3753_375305

theorem angle_triple_complement (x : ℝ) : x = 3 * (90 - x) → x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_l3753_375305


namespace NUMINAMATH_CALUDE_inequality_proof_l3753_375303

theorem inequality_proof (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  (1.7 : ℝ)^(0.3 : ℝ) > (0.9 : ℝ)^(3.1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3753_375303


namespace NUMINAMATH_CALUDE_wendy_recycling_points_l3753_375325

def points_per_bag : ℕ := 5
def total_bags : ℕ := 11
def unrecycled_bags : ℕ := 2

theorem wendy_recycling_points : 
  (total_bags - unrecycled_bags) * points_per_bag = 45 := by
  sorry

end NUMINAMATH_CALUDE_wendy_recycling_points_l3753_375325


namespace NUMINAMATH_CALUDE_max_value_of_a_l3753_375318

theorem max_value_of_a (a : ℝ) : 
  (∀ x y : ℝ, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 4 → 2 * x^2 - 2 * a * x * y + y^2 ≥ 0) →
  a ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3753_375318


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3753_375355

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + x + 1 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3753_375355


namespace NUMINAMATH_CALUDE_planar_edge_pairs_4_2_3_l3753_375322

/-- A rectangular prism with edge dimensions a, b, and c. -/
structure RectangularPrism where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The number of unordered pairs of edges that determine a plane in a rectangular prism. -/
def planarEdgePairs (prism : RectangularPrism) : ℕ :=
  sorry

/-- Theorem: The number of unordered pairs of edges that determine a plane
    in a rectangular prism with edge dimensions 4, 2, and 3 is equal to 42. -/
theorem planar_edge_pairs_4_2_3 :
  planarEdgePairs { a := 4, b := 2, c := 3 } = 42 := by
  sorry

end NUMINAMATH_CALUDE_planar_edge_pairs_4_2_3_l3753_375322


namespace NUMINAMATH_CALUDE_total_arrangements_eq_192_l3753_375363

/-- Represents the number of classes to be scheduled -/
def num_classes : ℕ := 6

/-- Represents the number of time slots in a day -/
def num_slots : ℕ := 6

/-- Represents the number of morning slots (first 4 periods) -/
def morning_slots : ℕ := 4

/-- Represents the number of afternoon slots (last 2 periods) -/
def afternoon_slots : ℕ := 2

/-- The number of ways to arrange the Chinese class in the morning -/
def chinese_arrangements : ℕ := morning_slots

/-- The number of ways to arrange the Biology class in the afternoon -/
def biology_arrangements : ℕ := afternoon_slots

/-- The number of remaining classes after scheduling Chinese and Biology -/
def remaining_classes : ℕ := num_classes - 2

/-- The number of remaining slots after scheduling Chinese and Biology -/
def remaining_slots : ℕ := num_slots - 2

/-- Calculates the total number of possible arrangements -/
def total_arrangements : ℕ :=
  chinese_arrangements * biology_arrangements * (remaining_classes.factorial)

/-- Theorem stating that the total number of arrangements is 192 -/
theorem total_arrangements_eq_192 : total_arrangements = 192 := by
  sorry

end NUMINAMATH_CALUDE_total_arrangements_eq_192_l3753_375363


namespace NUMINAMATH_CALUDE_james_matches_count_l3753_375312

/-- The number of boxes in a dozen -/
def boxesPerDozen : ℕ := 12

/-- The number of dozens of boxes James has -/
def dozensOfBoxes : ℕ := 5

/-- The number of matches in each box -/
def matchesPerBox : ℕ := 20

/-- Theorem: Given the conditions, James has 1200 matches -/
theorem james_matches_count :
  dozensOfBoxes * boxesPerDozen * matchesPerBox = 1200 := by
  sorry

end NUMINAMATH_CALUDE_james_matches_count_l3753_375312


namespace NUMINAMATH_CALUDE_alpha_in_third_quadrant_l3753_375395

theorem alpha_in_third_quadrant (α : Real) 
  (h1 : Real.tan (α - 3 * Real.pi) > 0) 
  (h2 : Real.sin (-α + Real.pi) < 0) : 
  Real.pi < α ∧ α < 3 * Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_in_third_quadrant_l3753_375395


namespace NUMINAMATH_CALUDE_largest_solution_quadratic_l3753_375338

theorem largest_solution_quadratic (x : ℝ) : 
  (9 * x^2 - 51 * x + 70 = 0) → x ≤ 70/9 :=
by sorry

end NUMINAMATH_CALUDE_largest_solution_quadratic_l3753_375338


namespace NUMINAMATH_CALUDE_snail_speed_ratio_l3753_375397

-- Define the speeds and times
def speed_snail1 : ℝ := 2
def time_snail1 : ℝ := 20
def time_snail3 : ℝ := 2

-- Define the relationship between snail speeds
def speed_snail3 (speed_snail2 : ℝ) : ℝ := 5 * speed_snail2

-- Define the race distance
def race_distance : ℝ := speed_snail1 * time_snail1

-- Theorem statement
theorem snail_speed_ratio :
  ∃ (speed_snail2 : ℝ),
    speed_snail3 speed_snail2 * time_snail3 = race_distance ∧
    speed_snail2 / speed_snail1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_snail_speed_ratio_l3753_375397


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_l3753_375306

theorem consecutive_page_numbers (n : ℕ) : 
  n * (n + 1) * (n + 2) = 35280 → n + (n + 1) + (n + 2) = 96 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_l3753_375306


namespace NUMINAMATH_CALUDE_power_729_minus_reciprocal_l3753_375311

theorem power_729_minus_reciprocal (x : ℂ) (h : x - 1/x = Complex.I * 2) :
  x^729 - 1/(x^729) = Complex.I * 2 := by
  sorry

end NUMINAMATH_CALUDE_power_729_minus_reciprocal_l3753_375311


namespace NUMINAMATH_CALUDE_circle_a_range_l3753_375336

-- Define the equation of the circle
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1 = 0

-- Define the condition for the equation to represent a circle
def is_circle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y a

-- Theorem stating the range of a for which the equation represents a circle
theorem circle_a_range :
  ∀ a : ℝ, is_circle a ↔ -2 < a ∧ a < 2/3 :=
sorry

end NUMINAMATH_CALUDE_circle_a_range_l3753_375336


namespace NUMINAMATH_CALUDE_eggs_needed_for_scaled_cake_l3753_375316

/-- Represents the recipe for sponge cake -/
structure Recipe where
  eggs : ℝ
  flour : ℝ
  sugar : ℝ

/-- Calculates the total mass of the cake from a recipe -/
def totalMass (r : Recipe) : ℝ := r.eggs + r.flour + r.sugar

/-- The original recipe -/
def originalRecipe : Recipe := { eggs := 300, flour := 120, sugar := 100 }

/-- Theorem: The amount of eggs needed for 2600g of sponge cake is 1500g -/
theorem eggs_needed_for_scaled_cake (desiredMass : ℝ) 
  (h : desiredMass = 2600) : 
  (originalRecipe.eggs / totalMass originalRecipe) * desiredMass = 1500 := by
  sorry

end NUMINAMATH_CALUDE_eggs_needed_for_scaled_cake_l3753_375316


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l3753_375319

/- Given a point B with three angles around it -/
def point_B (angle_ABC angle_ABD angle_CBD : ℝ) : Prop :=
  /- ∠CBD is a right angle -/
  angle_CBD = 90 ∧
  /- The sum of angles around point B is 200° -/
  angle_ABC + angle_ABD + angle_CBD = 200 ∧
  /- The measure of ∠ABD is 70° -/
  angle_ABD = 70

/- Theorem statement -/
theorem angle_ABC_measure :
  ∀ (angle_ABC angle_ABD angle_CBD : ℝ),
  point_B angle_ABC angle_ABD angle_CBD →
  angle_ABC = 40 := by
sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l3753_375319


namespace NUMINAMATH_CALUDE_sandwich_count_l3753_375328

theorem sandwich_count (sandwich_price : ℚ) (soda_price : ℚ) (soda_count : ℕ) (total_cost : ℚ) :
  sandwich_price = 149/100 →
  soda_price = 87/100 →
  soda_count = 4 →
  total_cost = 646/100 →
  ∃ (sandwich_count : ℕ), sandwich_count = 2 ∧ 
    sandwich_count * sandwich_price + soda_count * soda_price = total_cost :=
by sorry

end NUMINAMATH_CALUDE_sandwich_count_l3753_375328


namespace NUMINAMATH_CALUDE_pauls_paint_cans_l3753_375320

theorem pauls_paint_cans 
  (initial_rooms : ℕ) 
  (lost_cans : ℕ) 
  (remaining_rooms : ℕ) 
  (h1 : initial_rooms = 50)
  (h2 : lost_cans = 5)
  (h3 : remaining_rooms = 38) :
  (initial_rooms : ℚ) * lost_cans / (initial_rooms - remaining_rooms) = 21 :=
by sorry

end NUMINAMATH_CALUDE_pauls_paint_cans_l3753_375320


namespace NUMINAMATH_CALUDE_mk_97_check_one_l3753_375391

theorem mk_97_check_one (x : ℝ) : x = 1 ↔ x ≠ 0 ∧ 4 * (x^2 - x) = 0 := by sorry

end NUMINAMATH_CALUDE_mk_97_check_one_l3753_375391


namespace NUMINAMATH_CALUDE_jerry_zinc_consumption_l3753_375374

/-- Calculates the total milligrams of zinc consumed from antacids -/
def total_zinc_mg (large_antacid_count : ℕ) (large_antacid_weight : ℝ) (large_antacid_zinc_percent : ℝ)
                  (small_antacid_count : ℕ) (small_antacid_weight : ℝ) (small_antacid_zinc_percent : ℝ) : ℝ :=
  ((large_antacid_count : ℝ) * large_antacid_weight * large_antacid_zinc_percent +
   (small_antacid_count : ℝ) * small_antacid_weight * small_antacid_zinc_percent) * 1000

/-- Theorem stating the total zinc consumed by Jerry -/
theorem jerry_zinc_consumption :
  total_zinc_mg 2 2 0.05 3 1 0.15 = 650 := by
  sorry

end NUMINAMATH_CALUDE_jerry_zinc_consumption_l3753_375374


namespace NUMINAMATH_CALUDE_walkway_area_is_416_l3753_375396

/-- Represents a garden with flower beds and walkways -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed_width : ℝ
  bed_height : ℝ
  walkway_width : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkway_area (g : Garden) : ℝ :=
  let total_width := g.columns * g.bed_width + (g.columns + 1) * g.walkway_width
  let total_height := g.rows * g.bed_height + (g.rows + 1) * g.walkway_width
  let total_area := total_width * total_height
  let bed_area := g.rows * g.columns * g.bed_width * g.bed_height
  total_area - bed_area

/-- Theorem stating that the walkway area for the given garden is 416 square feet -/
theorem walkway_area_is_416 (g : Garden) 
  (h1 : g.rows = 4)
  (h2 : g.columns = 3)
  (h3 : g.bed_width = 8)
  (h4 : g.bed_height = 3)
  (h5 : g.walkway_width = 2) : 
  walkway_area g = 416 := by
  sorry

end NUMINAMATH_CALUDE_walkway_area_is_416_l3753_375396


namespace NUMINAMATH_CALUDE_initial_balloons_eq_sum_l3753_375377

/-- The number of balloons Tom initially had -/
def initial_balloons : ℕ := 30

/-- The number of balloons Tom gave to Fred -/
def balloons_given : ℕ := 16

/-- The number of balloons Tom has left -/
def balloons_left : ℕ := 14

/-- Theorem stating that the initial number of balloons is equal to
    the sum of balloons given away and balloons left -/
theorem initial_balloons_eq_sum :
  initial_balloons = balloons_given + balloons_left := by
  sorry

end NUMINAMATH_CALUDE_initial_balloons_eq_sum_l3753_375377


namespace NUMINAMATH_CALUDE_solve_equation_l3753_375357

theorem solve_equation : (45 : ℚ) / (8 - 3/4) = 180/29 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3753_375357


namespace NUMINAMATH_CALUDE_first_three_digits_of_quotient_l3753_375300

/-- The dividend a as a real number -/
def a : ℝ := 0.1234567891011

/-- The divisor b as a real number -/
def b : ℝ := 0.51504948

/-- Theorem stating that the first three digits of a/b are 0.239 -/
theorem first_three_digits_of_quotient (ha : a > 0) (hb : b > 0) :
  0.239 * b ≤ a ∧ a < 0.24 * b :=
sorry

end NUMINAMATH_CALUDE_first_three_digits_of_quotient_l3753_375300


namespace NUMINAMATH_CALUDE_problem1_problem2_problem3_problem4_l3753_375333

-- 1. Prove that 9999×2222+3333×3334 = 33330000
theorem problem1 : 9999 * 2222 + 3333 * 3334 = 33330000 := by sorry

-- 2. Prove that 96%×25+0.75+0.25 = 25
theorem problem2 : (96 / 100) * 25 + 0.75 + 0.25 = 25 := by sorry

-- 3. Prove that 5/8 + 7/10 + 3/8 + 3/10 = 2
theorem problem3 : 5/8 + 7/10 + 3/8 + 3/10 = 2 := by sorry

-- 4. Prove that 3.7 × 6/5 - 2.2 ÷ 5/6 = 1.8
theorem problem4 : 3.7 * (6/5) - 2.2 / (5/6) = 1.8 := by sorry

end NUMINAMATH_CALUDE_problem1_problem2_problem3_problem4_l3753_375333


namespace NUMINAMATH_CALUDE_m_range_if_f_increasing_l3753_375334

/-- Piecewise function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^2 + 2*m*x - 2 else 1 + Real.log x

/-- Theorem stating that if f is increasing, then m is in [1, 2] -/
theorem m_range_if_f_increasing (m : ℝ) :
  (∀ x y, x < y → f m x < f m y) → m ∈ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_m_range_if_f_increasing_l3753_375334


namespace NUMINAMATH_CALUDE_kamals_english_marks_l3753_375381

/-- Proves that given Kamal's marks in four subjects and his average across five subjects, his marks in the fifth subject (English) are 66. -/
theorem kamals_english_marks 
  (math_marks : ℕ) 
  (physics_marks : ℕ) 
  (chemistry_marks : ℕ) 
  (biology_marks : ℕ) 
  (average_marks : ℕ) 
  (h1 : math_marks = 65)
  (h2 : physics_marks = 77)
  (h3 : chemistry_marks = 62)
  (h4 : biology_marks = 75)
  (h5 : average_marks = 69)
  (h6 : average_marks * 5 = math_marks + physics_marks + chemistry_marks + biology_marks + english_marks) :
  english_marks = 66 := by
  sorry

#check kamals_english_marks

end NUMINAMATH_CALUDE_kamals_english_marks_l3753_375381


namespace NUMINAMATH_CALUDE_passing_percentage_is_36_percent_l3753_375372

/-- The passing percentage for an engineering exam --/
def passing_percentage (failed_marks : ℕ) (scored_marks : ℕ) (max_marks : ℕ) : ℚ :=
  ((scored_marks + failed_marks : ℚ) / max_marks) * 100

/-- Theorem: The passing percentage is 36% --/
theorem passing_percentage_is_36_percent :
  passing_percentage 14 130 400 = 36 := by
  sorry

end NUMINAMATH_CALUDE_passing_percentage_is_36_percent_l3753_375372


namespace NUMINAMATH_CALUDE_chord_length_l3753_375367

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C₃ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 14/5 = 0

-- Define the common chord of C₁ and C₂
def common_chord (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- Theorem statement
theorem chord_length :
  ∃ (chord_length : ℝ),
    chord_length = 4 ∧
    ∀ (x y : ℝ),
      common_chord x y →
      C₃ x y →
      (∃ (x' y' : ℝ),
        common_chord x' y' ∧
        C₃ x' y' ∧
        (x - x')^2 + (y - y')^2 = chord_length^2) :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l3753_375367


namespace NUMINAMATH_CALUDE_legs_on_ground_for_ten_horses_l3753_375351

/-- Represents the number of legs walking on the ground given the conditions of the problem --/
def legs_on_ground (num_horses : ℕ) : ℕ :=
  let num_men := num_horses
  let num_walking_men := num_men / 2
  let men_legs := num_walking_men * 2
  let horse_legs := num_horses * 4
  men_legs + horse_legs

/-- Theorem stating that with 10 horses, there are 50 legs walking on the ground --/
theorem legs_on_ground_for_ten_horses :
  legs_on_ground 10 = 50 := by
  sorry

end NUMINAMATH_CALUDE_legs_on_ground_for_ten_horses_l3753_375351


namespace NUMINAMATH_CALUDE_sandy_payment_l3753_375350

/-- Represents the cost and quantity of a coffee shop item -/
structure Item where
  price : ℚ
  quantity : ℕ

/-- Calculates the total cost of an order -/
def orderTotal (items : List Item) : ℚ :=
  items.foldl (fun acc item => acc + item.price * item.quantity) 0

/-- Proves that Sandy paid $20 given the order details and change received -/
theorem sandy_payment (cappuccino iced_tea cafe_latte espresso : Item)
    (change : ℚ) :
    cappuccino.price = 2 →
    iced_tea.price = 3 →
    cafe_latte.price = 3/2 →
    espresso.price = 1 →
    cappuccino.quantity = 3 →
    iced_tea.quantity = 2 →
    cafe_latte.quantity = 2 →
    espresso.quantity = 2 →
    change = 3 →
    orderTotal [cappuccino, iced_tea, cafe_latte, espresso] + change = 20 := by
  sorry


end NUMINAMATH_CALUDE_sandy_payment_l3753_375350


namespace NUMINAMATH_CALUDE_total_cost_calculation_l3753_375366

def cost_per_pound : ℝ := 0.45

def sugar_weight : ℝ := 40
def flour_weight : ℝ := 16

def total_cost : ℝ := cost_per_pound * (sugar_weight + flour_weight)

theorem total_cost_calculation : total_cost = 25.20 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_calculation_l3753_375366


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l3753_375386

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_implies_parallel
  (a b c : Line) (α β γ : Plane)
  (h1 : perpendicular a α)
  (h2 : perpendicular b β)
  (h3 : parallel_lines a b) :
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_parallel_l3753_375386


namespace NUMINAMATH_CALUDE_limit_exponential_arctangent_sine_l3753_375358

/-- The limit of (e^(4x) - e^(-2x)) / (2 arctan(x) - sin(x)) as x approaches 0 is 6 -/
theorem limit_exponential_arctangent_sine :
  let f : ℝ → ℝ := λ x => (Real.exp (4 * x) - Real.exp (-2 * x)) / (2 * Real.arctan x - Real.sin x)
  ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |f x - 6| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_exponential_arctangent_sine_l3753_375358


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l3753_375347

theorem cricket_team_average_age (team_size : ℕ) (captain_age : ℕ) (wicket_keeper_age_diff : ℕ) :
  team_size = 11 →
  captain_age = 25 →
  wicket_keeper_age_diff = 5 →
  let total_age := team_size * (captain_age + 7)
  let remaining_players := team_size - 2
  let remaining_age := total_age - (captain_age + (captain_age + wicket_keeper_age_diff))
  (remaining_age / remaining_players) + 1 = total_age / team_size →
  total_age / team_size = 32 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l3753_375347


namespace NUMINAMATH_CALUDE_angle_problem_l3753_375330

theorem angle_problem (α β : Real) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h_distance : Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = Real.sqrt 10 / 5)
  (h_tan : Real.tan (α/2) = 1/2) :
  (Real.cos (α - β) = 4/5) ∧
  (Real.cos α = 3/5) ∧
  (Real.cos β = 24/25) := by
sorry

end NUMINAMATH_CALUDE_angle_problem_l3753_375330


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l3753_375345

theorem quadratic_function_max_value (a b c : ℝ) : 
  (∃ a' ∈ Set.Icc 1 2, ∀ x ∈ Set.Icc 1 2, a' * x^2 + b * x + c ≤ 1) →
  (∀ m : ℝ, 7 * b + 5 * c ≤ m → m ≥ -6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l3753_375345


namespace NUMINAMATH_CALUDE_product_equals_four_l3753_375394

theorem product_equals_four : 16 * 0.5 * 4 * 0.125 = 4 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_four_l3753_375394


namespace NUMINAMATH_CALUDE_inequality_preservation_l3753_375314

theorem inequality_preservation (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3753_375314


namespace NUMINAMATH_CALUDE_sector_to_cone_l3753_375310

/-- Proves that a 270° sector of a circle with radius 12 forms a cone with base radius 9 and slant height 12 -/
theorem sector_to_cone (sector_angle : Real) (circle_radius : Real) 
  (h1 : sector_angle = 270)
  (h2 : circle_radius = 12) : 
  let base_radius := (sector_angle / 360) * (2 * Real.pi * circle_radius) / (2 * Real.pi)
  let slant_height := circle_radius
  (base_radius = 9 ∧ slant_height = 12) := by
  sorry

end NUMINAMATH_CALUDE_sector_to_cone_l3753_375310


namespace NUMINAMATH_CALUDE_angle_abc_measure_l3753_375326

theorem angle_abc_measure (angle_cbd angle_abd angle_abc : ℝ) 
  (h1 : angle_cbd = 90)
  (h2 : angle_abd = 60)
  (h3 : angle_abc + angle_abd + angle_cbd = 190) :
  angle_abc = 40 := by
  sorry

end NUMINAMATH_CALUDE_angle_abc_measure_l3753_375326


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3753_375376

/-- Given two vectors a and b in ℝ², prove that if a = (1, 2) and b = (x, 4) are perpendicular, then x = -8 -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 4]
  (∀ i : Fin 2, a i * b i = 0) → x = -8 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3753_375376


namespace NUMINAMATH_CALUDE_benny_comic_books_l3753_375339

theorem benny_comic_books (initial : ℕ) : 
  (initial / 2 + 6 = 17) → initial = 22 := by
  sorry

end NUMINAMATH_CALUDE_benny_comic_books_l3753_375339


namespace NUMINAMATH_CALUDE_det_matrix_l3753_375332

def matrix (y : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![y + 2, 2*y, 2*y;
     2*y, y + 2, 2*y;
     2*y, 2*y, y + 2]

theorem det_matrix (y : ℝ) :
  Matrix.det (matrix y) = 5*y^3 - 10*y^2 + 12*y + 8 := by
  sorry

end NUMINAMATH_CALUDE_det_matrix_l3753_375332


namespace NUMINAMATH_CALUDE_mike_notebooks_count_l3753_375348

theorem mike_notebooks_count :
  ∀ (total_spent blue_cost : ℕ) (red_count green_count : ℕ) (red_cost green_cost : ℕ),
    total_spent = 37 →
    red_count = 3 →
    green_count = 2 →
    red_cost = 4 →
    green_cost = 2 →
    blue_cost = 3 →
    total_spent = red_count * red_cost + green_count * green_cost + 
      ((total_spent - (red_count * red_cost + green_count * green_cost)) / blue_cost) * blue_cost →
    red_count + green_count + (total_spent - (red_count * red_cost + green_count * green_cost)) / blue_cost = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_mike_notebooks_count_l3753_375348


namespace NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l3753_375389

theorem largest_multiple_of_8_under_100 :
  ∃ n : ℕ, n * 8 = 96 ∧ n * 8 < 100 ∧ ∀ m : ℕ, m * 8 < 100 → m * 8 ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_8_under_100_l3753_375389


namespace NUMINAMATH_CALUDE_water_remaining_l3753_375384

theorem water_remaining (initial : ℚ) (used : ℚ) (remaining : ℚ) : 
  initial = 3 ∧ used = 11/4 → remaining = initial - used → remaining = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_l3753_375384


namespace NUMINAMATH_CALUDE_sphere_division_l3753_375321

theorem sphere_division (R : ℝ) : 
  (∃ (n : ℕ), n = 216 ∧ (4 / 3 * Real.pi * R^3 = n * (4 / 3 * Real.pi * 1^3))) ↔ R = 6 :=
sorry

end NUMINAMATH_CALUDE_sphere_division_l3753_375321


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3753_375380

/-- An isosceles trapezoid with the given properties has an area of 54000/3 square centimeters -/
theorem isosceles_trapezoid_area : 
  ∀ (leg diagonal longer_base : ℝ),
  leg = 40 →
  diagonal = 50 →
  longer_base = 60 →
  ∃ (area : ℝ),
  area = 54000 / 3 ∧
  area = (longer_base + (longer_base - 2 * (Real.sqrt (leg^2 - ((100/3)^2))))) * (100/3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_area_l3753_375380


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_l3753_375344

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem right_triangle_perimeter : ∃ (a b c : ℕ),
  a = 11 ∧
  is_pythagorean_triple a b c ∧
  a + b + c = 132 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_l3753_375344


namespace NUMINAMATH_CALUDE_cube_frame_wire_ratio_l3753_375352

/-- The ratio of wire lengths used by two people constructing cube frames -/
theorem cube_frame_wire_ratio : 
  ∀ (wire_a wire_b : ℕ) (pieces_a : ℕ) (volume : ℕ),
  wire_a = 8 →
  pieces_a = 12 →
  wire_b = 2 →
  volume = wire_a^3 →
  (wire_a * pieces_a) / (wire_b * 12 * volume) = 1 / 128 :=
by sorry

end NUMINAMATH_CALUDE_cube_frame_wire_ratio_l3753_375352


namespace NUMINAMATH_CALUDE_ladder_construction_possible_l3753_375331

/-- Represents the ladder construction problem --/
def ladder_problem (total_wood rung_length rung_spacing side_support_length climbing_height : ℝ) : Prop :=
  let num_rungs : ℝ := climbing_height / rung_spacing + 1
  let wood_for_rungs : ℝ := num_rungs * rung_length
  let wood_for_supports : ℝ := 2 * side_support_length
  let total_wood_needed : ℝ := wood_for_rungs + wood_for_supports
  let leftover_wood : ℝ := total_wood - total_wood_needed
  total_wood_needed ≤ total_wood ∧ leftover_wood = 36.5

/-- Theorem stating that the ladder can be built with the given conditions --/
theorem ladder_construction_possible : 
  ladder_problem 300 1.5 0.5 56 50 := by
  sorry

#check ladder_construction_possible

end NUMINAMATH_CALUDE_ladder_construction_possible_l3753_375331


namespace NUMINAMATH_CALUDE_color_tv_cost_price_l3753_375371

/-- The cost price of a color TV satisfying the given conditions -/
def cost_price : ℝ := 3000

/-- The selling price before discount -/
def selling_price (cost : ℝ) : ℝ := cost * 1.4

/-- The discounted price -/
def discounted_price (price : ℝ) : ℝ := price * 0.8

/-- The profit is the difference between the discounted price and the cost price -/
def profit (cost : ℝ) : ℝ := discounted_price (selling_price cost) - cost

theorem color_tv_cost_price : 
  profit cost_price = 360 :=
sorry

end NUMINAMATH_CALUDE_color_tv_cost_price_l3753_375371


namespace NUMINAMATH_CALUDE_brass_weight_l3753_375304

theorem brass_weight (copper_ratio : ℚ) (zinc_ratio : ℚ) (zinc_weight : ℚ) : 
  copper_ratio = 3 → 
  zinc_ratio = 7 → 
  zinc_weight = 70 → 
  (copper_ratio + zinc_ratio) * (zinc_weight / zinc_ratio) = 100 :=
by sorry

end NUMINAMATH_CALUDE_brass_weight_l3753_375304


namespace NUMINAMATH_CALUDE_pumpkin_patch_problem_l3753_375392

def pumpkin_pie_filling_cans (total_pumpkins : ℕ) (price_per_pumpkin : ℕ) (total_earnings : ℕ) (pumpkins_per_can : ℕ) : ℕ :=
  let pumpkins_sold := total_earnings / price_per_pumpkin
  let remaining_pumpkins := total_pumpkins - pumpkins_sold
  remaining_pumpkins / pumpkins_per_can

theorem pumpkin_patch_problem :
  pumpkin_pie_filling_cans 83 3 96 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_pumpkin_patch_problem_l3753_375392


namespace NUMINAMATH_CALUDE_shirts_per_minute_l3753_375360

/-- An industrial machine that makes shirts -/
structure ShirtMachine where
  /-- The number of shirts made in 6 minutes -/
  shirts_in_6_min : ℕ
  /-- The number of minutes (6) -/
  minutes : ℕ
  /-- Assumption that the machine made 36 shirts in 6 minutes -/
  h_shirts : shirts_in_6_min = 36
  /-- Assumption that the time period is 6 minutes -/
  h_minutes : minutes = 6

/-- Theorem stating that the machine makes 6 shirts per minute -/
theorem shirts_per_minute (machine : ShirtMachine) : 
  machine.shirts_in_6_min / machine.minutes = 6 := by
  sorry

#check shirts_per_minute

end NUMINAMATH_CALUDE_shirts_per_minute_l3753_375360
