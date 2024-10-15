import Mathlib

namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3899_389945

/-- An isosceles triangle with side lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 3 ∧ b = 7 ∧ c = 7 →  -- Two sides are 7, one side is 3
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 17 := by  -- Perimeter is 17
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3899_389945


namespace NUMINAMATH_CALUDE_inequality_proof_l3899_389967

theorem inequality_proof (x y z : ℝ) :
  x^2 + y^2 + z^2 - x*y - y*z - z*x ≥ max (3*(x-y)^2/4) (max (3*(y-z)^2/4) (3*(z-x)^2/4)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3899_389967


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3899_389933

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 6) :
  ∃ (M : ℝ), M = 3 * Real.sqrt 8 ∧ 
  Real.sqrt (3*x + 2) + Real.sqrt (3*y + 2) + Real.sqrt (3*z + 2) ≤ M ∧
  ∃ (x' y' z' : ℝ), x' ≥ 0 ∧ y' ≥ 0 ∧ z' ≥ 0 ∧ x' + y' + z' = 6 ∧
    Real.sqrt (3*x' + 2) + Real.sqrt (3*y' + 2) + Real.sqrt (3*z' + 2) = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3899_389933


namespace NUMINAMATH_CALUDE_function_is_identity_or_reflection_l3899_389971

-- Define the function f
def f (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x + b

-- State the theorem
theorem function_is_identity_or_reflection (a b : ℝ) :
  (∀ x : ℝ, f a b (f a b x) = x) →
  ((a = 1 ∧ b = 0) ∨ ∃ c : ℝ, a = -1 ∧ ∀ x : ℝ, f a b x = -x + c) :=
by sorry

end NUMINAMATH_CALUDE_function_is_identity_or_reflection_l3899_389971


namespace NUMINAMATH_CALUDE_square_plus_cube_equals_one_l3899_389976

theorem square_plus_cube_equals_one : 3^2 + (-2)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_cube_equals_one_l3899_389976


namespace NUMINAMATH_CALUDE_f_6n_l3899_389923

def f : ℕ → ℤ
  | 0 => 0
  | n + 1 => 
    if n % 6 = 0 ∨ n % 6 = 1 then f n + 3
    else if n % 6 = 2 ∨ n % 6 = 5 then f n + 1
    else f n + 2

theorem f_6n (n : ℕ) : f (6 * n) = 12 * n := by
  sorry

end NUMINAMATH_CALUDE_f_6n_l3899_389923


namespace NUMINAMATH_CALUDE_subset_sum_implies_total_sum_l3899_389980

theorem subset_sum_implies_total_sum (a₁ a₂ a₃ : ℝ) :
  (a₁ + a₂ + a₁ + a₃ + a₂ + a₃ + (a₁ + a₂) + (a₁ + a₃) + (a₂ + a₃) = 12) →
  (a₁ + a₂ + a₃ = 4) := by
  sorry

end NUMINAMATH_CALUDE_subset_sum_implies_total_sum_l3899_389980


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l3899_389960

/-- An isosceles triangle with perimeter 16 and one side 4 has a base of 4 -/
theorem isosceles_triangle_base (a b c : ℝ) : 
  a + b + c = 16 →  -- perimeter is 16
  a = b →           -- isosceles triangle condition
  a = 4 →           -- one side is 4
  c = 4 :=          -- prove that the base is 4
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l3899_389960


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l3899_389911

theorem no_solution_implies_a_leq_two (a : ℝ) : 
  (∀ x : ℝ, ¬(x > 1 ∧ x < a - 1)) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l3899_389911


namespace NUMINAMATH_CALUDE_count_valid_insertions_l3899_389995

/-- The number of different three-digit numbers that can be inserted into 689???20312 to make it approximately 69 billion when rounded -/
def valid_insertions : ℕ :=
  let ten_million_digits := {5, 6, 7, 8, 9}
  let other_digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  (Finset.card ten_million_digits) * (Finset.card other_digits) * (Finset.card other_digits)

theorem count_valid_insertions : valid_insertions = 500 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_insertions_l3899_389995


namespace NUMINAMATH_CALUDE_infinitely_many_triangular_squares_l3899_389938

/-- Definition of triangular numbers -/
def T (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate for a number being square -/
def is_square (k : ℕ) : Prop := ∃ m : ℕ, k = m * m

/-- The recurrence relation for generating triangular square numbers -/
axiom recurrence_relation (n : ℕ) : T (4 * n * (n + 1)) = 4 * T n * (2 * n + 1)^2

/-- Theorem: There are infinitely many numbers that are both triangular and square -/
theorem infinitely_many_triangular_squares :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ is_square (T n) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_triangular_squares_l3899_389938


namespace NUMINAMATH_CALUDE_mans_swimming_speed_l3899_389914

/-- 
Given a man who swims against a current, this theorem proves his swimming speed in still water.
-/
theorem mans_swimming_speed 
  (distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (h1 : distance = 40) 
  (h2 : time = 5) 
  (h3 : current_speed = 12) : 
  ∃ (speed : ℝ), speed = 20 ∧ distance = time * (speed - current_speed) :=
by sorry

end NUMINAMATH_CALUDE_mans_swimming_speed_l3899_389914


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l3899_389966

theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 30 * π → π * r^2 = 225 * π :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l3899_389966


namespace NUMINAMATH_CALUDE_solution_of_system_l3899_389998

def augmented_matrix : Matrix (Fin 2) (Fin 3) ℝ := !![1, -1, 1; 1, 1, 3]

theorem solution_of_system (x y : ℝ) : 
  x = 2 ∧ y = 1 ↔ 
  (augmented_matrix 0 0 * x + augmented_matrix 0 1 * y = augmented_matrix 0 2) ∧
  (augmented_matrix 1 0 * x + augmented_matrix 1 1 * y = augmented_matrix 1 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_system_l3899_389998


namespace NUMINAMATH_CALUDE_volleyball_game_employees_l3899_389910

theorem volleyball_game_employees (managers : ℕ) (teams : ℕ) (people_per_team : ℕ) :
  managers = 3 →
  teams = 3 →
  people_per_team = 2 →
  teams * people_per_team - managers = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_volleyball_game_employees_l3899_389910


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3899_389946

theorem fraction_multiplication : (2 : ℚ) / 3 * (3 : ℚ) / 8 = (1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3899_389946


namespace NUMINAMATH_CALUDE_ellipse_iff_range_l3899_389990

/-- The equation of an ellipse with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 / (2 - m) + y^2 / (m + 1) = 1

/-- The range of m for which the equation represents an ellipse -/
def ellipse_range (m : ℝ) : Prop :=
  (m > -1 ∧ m < 1/2) ∨ (m > 1/2 ∧ m < 2)

/-- Theorem stating that the equation represents an ellipse if and only if m is in the specified range -/
theorem ellipse_iff_range (m : ℝ) : is_ellipse m ↔ ellipse_range m := by
  sorry

end NUMINAMATH_CALUDE_ellipse_iff_range_l3899_389990


namespace NUMINAMATH_CALUDE_least_common_period_is_36_l3899_389978

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

/-- A function is periodic with period p -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least common positive period for all functions satisfying the condition -/
def LeastCommonPeriod : ℝ := 36

/-- Main theorem: The least common positive period for all functions satisfying the condition is 36 -/
theorem least_common_period_is_36 :
  ∀ f : ℝ → ℝ, SatisfiesCondition f →
  (∀ p : ℝ, p > 0 → IsPeriodic f p → p ≥ LeastCommonPeriod) ∧
  (∃ f : ℝ → ℝ, SatisfiesCondition f ∧ IsPeriodic f LeastCommonPeriod) :=
sorry

end NUMINAMATH_CALUDE_least_common_period_is_36_l3899_389978


namespace NUMINAMATH_CALUDE_min_lines_to_cover_plane_l3899_389934

-- Define the circle on a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define a line on a plane
structure Line :=
  (a b c : ℝ)

-- Define a reflection of a point with respect to a line
def reflect (p : ℝ × ℝ) (l : Line) : ℝ × ℝ := sorry

-- Define a function to check if a point is covered by a circle
def is_covered (p : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Define a function to perform a finite sequence of reflections
def reflect_sequence (c : Circle) (lines : List Line) : Circle := sorry

-- Theorem statement
theorem min_lines_to_cover_plane (c : Circle) :
  ∃ (lines : List Line),
    (lines.length = 3) ∧
    (∀ (p : ℝ × ℝ), ∃ (seq : List Line),
      (∀ (l : Line), l ∈ seq → l ∈ lines) ∧
      is_covered p (reflect_sequence c seq)) ∧
    (∀ (lines' : List Line),
      lines'.length < 3 →
      ∃ (p : ℝ × ℝ), ∀ (seq : List Line),
        (∀ (l : Line), l ∈ seq → l ∈ lines') →
        ¬is_covered p (reflect_sequence c seq)) :=
sorry

end NUMINAMATH_CALUDE_min_lines_to_cover_plane_l3899_389934


namespace NUMINAMATH_CALUDE_average_pushups_l3899_389965

theorem average_pushups (david zachary emily : ℕ) : 
  david = 510 ∧ 
  david = zachary + 210 ∧ 
  david = emily + 132 → 
  (david + zachary + emily) / 3 = 396 := by
    sorry

end NUMINAMATH_CALUDE_average_pushups_l3899_389965


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3899_389925

theorem quadratic_inequality_solution (x : ℝ) :
  (3 * x^2 - x - 4 ≥ 0) ↔ (x ≤ -1 ∨ x ≥ 4/3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3899_389925


namespace NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3899_389930

theorem muffin_banana_price_ratio :
  ∀ (muffin_price banana_price : ℚ),
  (5 * muffin_price + 4 * banana_price = 20) →
  (3 * muffin_price + 18 * banana_price = 60) →
  muffin_price / banana_price = 13 / 4 := by
sorry

end NUMINAMATH_CALUDE_muffin_banana_price_ratio_l3899_389930


namespace NUMINAMATH_CALUDE_mike_camera_purchase_l3899_389962

/-- Given:
  - The new camera model costs 30% more than the current model
  - The old camera costs $4000
  - Mike gets $200 off a $400 lens

  Prove that Mike paid $5400 for the camera and lens. -/
theorem mike_camera_purchase (old_camera_cost : ℝ) (lens_cost : ℝ) (lens_discount : ℝ) :
  old_camera_cost = 4000 →
  lens_cost = 400 →
  lens_discount = 200 →
  let new_camera_cost := old_camera_cost * 1.3
  let discounted_lens_cost := lens_cost - lens_discount
  new_camera_cost + discounted_lens_cost = 5400 := by
  sorry

end NUMINAMATH_CALUDE_mike_camera_purchase_l3899_389962


namespace NUMINAMATH_CALUDE_audrey_lost_six_pieces_l3899_389968

/-- Represents the number of pieces in a chess game -/
structure ChessGame where
  total_pieces : ℕ
  audrey_pieces : ℕ
  thomas_pieces : ℕ

/-- The initial state of a chess game -/
def initial_chess_game : ChessGame :=
  { total_pieces := 32
  , audrey_pieces := 16
  , thomas_pieces := 16 }

/-- The final state of the chess game after pieces are lost -/
def final_chess_game : ChessGame :=
  { total_pieces := 21
  , audrey_pieces := 21 - (initial_chess_game.thomas_pieces - 5)
  , thomas_pieces := initial_chess_game.thomas_pieces - 5 }

/-- Theorem stating that Audrey lost 6 pieces -/
theorem audrey_lost_six_pieces :
  initial_chess_game.audrey_pieces - final_chess_game.audrey_pieces = 6 := by
  sorry


end NUMINAMATH_CALUDE_audrey_lost_six_pieces_l3899_389968


namespace NUMINAMATH_CALUDE_price_large_bottle_correct_l3899_389913

/-- The price of a large bottle, given the following conditions:
  * 1365 large bottles were purchased at this price
  * 720 small bottles were purchased at $1.42 each
  * The average price of all bottles was approximately $1.73
-/
def price_large_bottle : ℝ := 1.89

theorem price_large_bottle_correct : 
  let num_large : ℕ := 1365
  let num_small : ℕ := 720
  let price_small : ℝ := 1.42
  let avg_price : ℝ := 1.73
  let total_bottles : ℕ := num_large + num_small
  let total_cost : ℝ := (num_large : ℝ) * price_large_bottle + (num_small : ℝ) * price_small
  abs (total_cost / (total_bottles : ℝ) - avg_price) < 0.01 ∧ 
  abs (price_large_bottle - 1.89) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_price_large_bottle_correct_l3899_389913


namespace NUMINAMATH_CALUDE_parallel_condition_l3899_389937

-- Define the structure for a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define when two lines are parallel
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∨ l1.b ≠ 0

-- Define the two lines from the problem
def line1 (a : ℝ) : Line := ⟨2, a, -1⟩
def line2 (b : ℝ) : Line := ⟨b, 2, 1⟩

theorem parallel_condition (a b : ℝ) :
  (parallel (line1 a) (line2 b) → a * b = 4) ∧
  ∃ a b, a * b = 4 ∧ ¬parallel (line1 a) (line2 b) := by sorry

end NUMINAMATH_CALUDE_parallel_condition_l3899_389937


namespace NUMINAMATH_CALUDE_kantana_chocolates_l3899_389916

/-- Represents the number of chocolates Kantana buys for herself each Saturday -/
def chocolates_for_self : ℕ := sorry

/-- Represents the number of Saturdays in the month -/
def saturdays_in_month : ℕ := 4

/-- Represents the number of chocolates bought for Charlie's birthday -/
def chocolates_for_charlie : ℕ := 10

/-- Represents the total number of chocolates bought in the month -/
def total_chocolates : ℕ := 22

/-- Theorem stating that Kantana buys 2 chocolates for herself each Saturday -/
theorem kantana_chocolates : 
  chocolates_for_self = 2 ∧ 
  (chocolates_for_self + 1) * saturdays_in_month + chocolates_for_charlie = total_chocolates :=
by sorry

end NUMINAMATH_CALUDE_kantana_chocolates_l3899_389916


namespace NUMINAMATH_CALUDE_sector_area_l3899_389970

theorem sector_area (θ r : ℝ) (h1 : θ = 3) (h2 : r = 4) : 
  (1/2) * θ * r^2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_sector_area_l3899_389970


namespace NUMINAMATH_CALUDE_max_sum_of_square_roots_max_sum_of_square_roots_achievable_l3899_389993

theorem max_sum_of_square_roots (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 8) :
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ Real.sqrt 78 :=
by sorry

theorem max_sum_of_square_roots_achievable :
  ∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 8 ∧
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) = Real.sqrt 78 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_square_roots_max_sum_of_square_roots_achievable_l3899_389993


namespace NUMINAMATH_CALUDE_max_k_value_l3899_389940

theorem max_k_value (a b k : ℝ) : 
  a > 0 → b > 0 → a + b = 1 → a^2 + b^2 ≥ k → k ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_max_k_value_l3899_389940


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3899_389983

/-- A rectangular prism. -/
structure RectangularPrism :=
  (vertices : Nat)
  (edges : Nat)

/-- The number of diagonals in a rectangular prism. -/
def num_diagonals (rp : RectangularPrism) : Nat :=
  sorry

/-- Theorem: A rectangular prism with 12 vertices and 18 edges has 24 diagonals. -/
theorem rectangular_prism_diagonals :
  ∀ (rp : RectangularPrism), rp.vertices = 12 → rp.edges = 18 → num_diagonals rp = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l3899_389983


namespace NUMINAMATH_CALUDE_perimeter_ratio_square_to_rectangle_l3899_389915

/-- The ratio of the perimeter of a square with side length 700 to the perimeter of a rectangle with length 400 and width 300 is 2:1 -/
theorem perimeter_ratio_square_to_rectangle : 
  let square_side : ℕ := 700
  let rect_length : ℕ := 400
  let rect_width : ℕ := 300
  let square_perimeter : ℕ := 4 * square_side
  let rect_perimeter : ℕ := 2 * (rect_length + rect_width)
  (square_perimeter : ℚ) / rect_perimeter = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ratio_square_to_rectangle_l3899_389915


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3899_389964

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_mersenne_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ n = 2^p - 1 ∧ is_prime n

theorem largest_mersenne_prime_under_500 :
  (∀ m : ℕ, is_mersenne_prime m → m < 500 → m ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3899_389964


namespace NUMINAMATH_CALUDE_cost_change_l3899_389987

theorem cost_change (t : ℝ) (b₁ b₂ : ℝ) (h : t * b₂^4 = 16 * t * b₁^4) :
  b₂ = 2 * b₁ := by
  sorry

end NUMINAMATH_CALUDE_cost_change_l3899_389987


namespace NUMINAMATH_CALUDE_set_operation_equality_l3899_389979

theorem set_operation_equality (M N P : Set ℕ) : 
  M = {1, 2, 3} → N = {2, 3, 4} → P = {3, 5} → 
  (M ∩ N) ∪ P = {2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_equality_l3899_389979


namespace NUMINAMATH_CALUDE_expand_product_l3899_389948

theorem expand_product (y : ℝ) : 5 * (y - 6) * (y + 9) = 5 * y^2 + 15 * y - 270 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l3899_389948


namespace NUMINAMATH_CALUDE_count_ways_2016_l3899_389984

/-- The number of ways to write 2016 as the sum of twos and threes, ignoring order -/
def ways_to_write_2016 : ℕ :=
  (Finset.range 337).card

/-- The theorem stating that there are 337 ways to write 2016 as the sum of twos and threes -/
theorem count_ways_2016 : ways_to_write_2016 = 337 := by
  sorry

end NUMINAMATH_CALUDE_count_ways_2016_l3899_389984


namespace NUMINAMATH_CALUDE_sequence_sixth_term_l3899_389953

theorem sequence_sixth_term :
  let seq : ℕ → ℚ := fun n => 1 / (n * (n + 1))
  seq 6 = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sixth_term_l3899_389953


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l3899_389994

/-- The function f(x) = x^2 / 2^x is monotonically increasing on the interval (0, 2/ln(2)) -/
theorem monotonic_increasing_interval (f : ℝ → ℝ) : 
  (∀ x, f x = x^2 / 2^x) →
  (∃ a b, a = 0 ∧ b = 2 / Real.log 2 ∧
    ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l3899_389994


namespace NUMINAMATH_CALUDE_modulus_of_z_l3899_389972

open Complex

theorem modulus_of_z (z : ℂ) (h : z * (1 + I) = I) : abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3899_389972


namespace NUMINAMATH_CALUDE_water_depth_in_cistern_l3899_389939

/-- Calculates the depth of water in a rectangular cistern given its dimensions and wet surface area. -/
theorem water_depth_in_cistern
  (length : ℝ)
  (width : ℝ)
  (total_wet_surface_area : ℝ)
  (h1 : length = 8)
  (h2 : width = 6)
  (h3 : total_wet_surface_area = 83) :
  ∃ (depth : ℝ), depth = 1.25 ∧ 
    total_wet_surface_area = length * width + 2 * (length + width) * depth :=
by sorry


end NUMINAMATH_CALUDE_water_depth_in_cistern_l3899_389939


namespace NUMINAMATH_CALUDE_ball_bounce_distance_l3899_389906

/-- The total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  let descendDistances := Finset.sum (Finset.range bounces) (fun i => initialHeight * reboundFactor^i)
  let ascendDistances := Finset.sum (Finset.range (bounces - 1)) (fun i => initialHeight * reboundFactor^(i+1))
  descendDistances + ascendDistances

/-- Theorem stating the total distance traveled by the ball -/
theorem ball_bounce_distance :
  totalDistance 120 (1/3) 5 = 278.52 := by
  sorry

end NUMINAMATH_CALUDE_ball_bounce_distance_l3899_389906


namespace NUMINAMATH_CALUDE_singh_gain_l3899_389921

/-- Represents the game outcome for three players -/
structure GameOutcome where
  ashtikar : ℚ
  singh : ℚ
  bhatia : ℚ

/-- The initial amount each player starts with -/
def initial_amount : ℚ := 70

/-- The theorem stating Singh's gain in the game -/
theorem singh_gain (outcome : GameOutcome) : 
  outcome.ashtikar + outcome.singh + outcome.bhatia = 3 * initial_amount ∧
  outcome.ashtikar = (1/2) * outcome.singh ∧
  outcome.bhatia = (1/4) * outcome.singh →
  outcome.singh - initial_amount = 50 := by
sorry

end NUMINAMATH_CALUDE_singh_gain_l3899_389921


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_l3899_389901

/-- Represents an unfair eight-sided die -/
structure UnfairDie where
  /-- The probability of rolling an 8 -/
  prob_eight : ℚ
  /-- The probability of rolling any number from 1 to 7 -/
  prob_others : ℚ
  /-- The sum of all probabilities is 1 -/
  prob_sum : prob_eight + 7 * prob_others = 1
  /-- The probability of rolling an 8 is 3/8 -/
  eight_is_three_eighths : prob_eight = 3 / 8

/-- Calculates the expected value of a roll of the unfair die -/
def expected_value (d : UnfairDie) : ℚ :=
  (d.prob_others * (1 + 2 + 3 + 4 + 5 + 6 + 7)) + (d.prob_eight * 8)

/-- Theorem stating that the expected value of the unfair die is 77/14 -/
theorem unfair_die_expected_value (d : UnfairDie) : expected_value d = 77 / 14 := by
  sorry

end NUMINAMATH_CALUDE_unfair_die_expected_value_l3899_389901


namespace NUMINAMATH_CALUDE_ivy_cupcakes_l3899_389944

theorem ivy_cupcakes (morning_cupcakes : ℕ) (afternoon_difference : ℕ) : 
  morning_cupcakes = 20 →
  afternoon_difference = 15 →
  morning_cupcakes + (morning_cupcakes + afternoon_difference) = 55 :=
by
  sorry

end NUMINAMATH_CALUDE_ivy_cupcakes_l3899_389944


namespace NUMINAMATH_CALUDE_class_factory_arrangements_l3899_389954

/-- The number of classes -/
def num_classes : ℕ := 5

/-- The number of factories -/
def num_factories : ℕ := 4

/-- The number of ways to arrange classes into factories -/
def arrangements : ℕ := 240

/-- Theorem stating the number of arrangements -/
theorem class_factory_arrangements :
  (∀ (arrangement : Fin num_classes → Fin num_factories),
    (∀ f : Fin num_factories, ∃ c : Fin num_classes, arrangement c = f) →
    (∀ c : Fin num_classes, arrangement c < num_factories)) →
  arrangements = 240 :=
sorry

end NUMINAMATH_CALUDE_class_factory_arrangements_l3899_389954


namespace NUMINAMATH_CALUDE_gabby_needs_ten_more_dollars_l3899_389947

def makeup_set_cost : ℕ := 65
def gabby_initial_savings : ℕ := 35
def mom_additional_money : ℕ := 20

theorem gabby_needs_ten_more_dollars : 
  makeup_set_cost - (gabby_initial_savings + mom_additional_money) = 10 := by
sorry

end NUMINAMATH_CALUDE_gabby_needs_ten_more_dollars_l3899_389947


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3899_389989

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 6) (h2 : x^2 - y^2 = 12) : 
  |x - y| = 2 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3899_389989


namespace NUMINAMATH_CALUDE_equation_solutions_l3899_389985

theorem equation_solutions (x y z v : ℤ) : 
  (x^2 + y^2 + z^2 = 2*x*y*z ↔ x = 0 ∧ y = 0 ∧ z = 0) ∧
  (x^2 + y^2 + z^2 + v^2 = 2*x*y*z*v ↔ x = 0 ∧ y = 0 ∧ z = 0 ∧ v = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3899_389985


namespace NUMINAMATH_CALUDE_m_range_l3899_389999

theorem m_range (x : ℝ) :
  (∀ x, 1/3 < x ∧ x < 1/2 → m - 1 < x ∧ x < m + 1) ∧
  (∃ x, m - 1 < x ∧ x < m + 1 ∧ (x ≤ 1/3 ∨ 1/2 ≤ x)) →
  -1/2 ≤ m ∧ m ≤ 4/3 ∧ m ≠ -1/2 ∧ m ≠ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3899_389999


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3899_389902

theorem sqrt_equation_solution (x : ℝ) : 
  (Real.sqrt (4 * x + 6) / Real.sqrt (8 * x + 12) = Real.sqrt 2 / 2) → x ≥ -3/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3899_389902


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9999_l3899_389950

theorem largest_prime_factor_of_9999 : ∃ (p : ℕ), p.Prime ∧ p ∣ 9999 ∧ ∀ (q : ℕ), q.Prime → q ∣ 9999 → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9999_l3899_389950


namespace NUMINAMATH_CALUDE_sequence_ratio_l3899_389975

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that the ratio of the sum of certain terms to another term equals 5/2. -/
theorem sequence_ratio (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  (1 : ℝ) < a₁ ∧ a₁ < a₂ ∧ a₂ < 4 ∧  -- arithmetic sequence condition
  (∃ r : ℝ, r > 0 ∧ b₁ = r ∧ b₂ = r^2 ∧ b₃ = r^3 ∧ 4 = r^4) →  -- geometric sequence condition
  (a₁ + a₂) / b₂ = 5/2 := by
sorry

end NUMINAMATH_CALUDE_sequence_ratio_l3899_389975


namespace NUMINAMATH_CALUDE_heart_equal_set_is_four_lines_l3899_389919

-- Define the ♥ operation
def heart (a b : ℝ) : ℝ := a^3 * b - a^2 * b^2 + a * b^3

-- Define the set of points satisfying x ♥ y = y ♥ x
def heart_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | heart p.1 p.2 = heart p.2 p.1}

-- Theorem statement
theorem heart_equal_set_is_four_lines :
  heart_equal_set = {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2 ∨ p.1 = -p.2} :=
by sorry

end NUMINAMATH_CALUDE_heart_equal_set_is_four_lines_l3899_389919


namespace NUMINAMATH_CALUDE_order_of_magnitudes_l3899_389961

theorem order_of_magnitudes (x a : ℝ) (hx : x < 0) (ha : a = 2 * x) :
  x^2 < a * x ∧ a * x < a^2 := by
  sorry

end NUMINAMATH_CALUDE_order_of_magnitudes_l3899_389961


namespace NUMINAMATH_CALUDE_tan_function_property_l3899_389900

/-- Given a function f(x) = a tan(bx) where a and b are positive constants,
    if f has roots at ±π/4 and passes through (π/8, 1), then a · b = 2 -/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, a * Real.tan (b * x) = 0 ↔ x = π/4 ∨ x = -π/4) →
  a * Real.tan (b * π/8) = 1 →
  a * b = 2 := by sorry

end NUMINAMATH_CALUDE_tan_function_property_l3899_389900


namespace NUMINAMATH_CALUDE_f_equation_l3899_389922

-- Define the function f
def f : ℝ → ℝ := fun x => sorry

-- State the theorem
theorem f_equation : ∀ x : ℝ, f (x + 1) = x^2 - 5*x + 4 → f x = x^2 - 7*x + 10 := by
  sorry

end NUMINAMATH_CALUDE_f_equation_l3899_389922


namespace NUMINAMATH_CALUDE_power_of_two_equality_l3899_389997

theorem power_of_two_equality (x : ℕ) : (1 / 8 : ℚ) * (2 : ℚ)^36 = (2 : ℚ)^x → x = 33 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l3899_389997


namespace NUMINAMATH_CALUDE_tan_240_plus_sin_neg_420_l3899_389986

theorem tan_240_plus_sin_neg_420 :
  Real.tan (240 * π / 180) + Real.sin ((-420) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_240_plus_sin_neg_420_l3899_389986


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l3899_389951

theorem stratified_sampling_theorem (total_population : ℕ) (female_population : ℕ) (sample_size : ℕ) (female_sample : ℕ) :
  total_population = 2400 →
  female_population = 1000 →
  female_sample = 40 →
  (female_sample : ℚ) / sample_size = (female_population : ℚ) / total_population →
  sample_size = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l3899_389951


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l3899_389963

/-- A line in the xy-plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Returns true if a point (x, y) is on the given line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.yIntercept

theorem parallel_line_y_intercept (b : Line) :
  parallel b { slope := -3, yIntercept := 6 } →
  pointOnLine b 4 (-1) →
  b.yIntercept = 11 := by sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l3899_389963


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3899_389996

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- Checks if a point lies on the hyperbola -/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- Checks if a line is an asymptote of the hyperbola -/
def Hyperbola.is_asymptote (h : Hyperbola) (m : ℝ) : Prop :=
  m = h.b / h.a ∨ m = -h.b / h.a

/-- The main theorem -/
theorem hyperbola_equation (h : Hyperbola) :
  h.contains 3 (Real.sqrt 2) ∧
  h.is_asymptote (1/3) ∧
  h.is_asymptote (-1/3) →
  h.a^2 = 153 ∧ h.b^2 = 17 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3899_389996


namespace NUMINAMATH_CALUDE_initial_concentration_proof_l3899_389928

/-- Proves that the initial concentration of a solution is 45% given the specified conditions -/
theorem initial_concentration_proof (initial_concentration : ℝ) : 
  (0.5 * initial_concentration + 0.5 * 0.25 = 0.35) → initial_concentration = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_initial_concentration_proof_l3899_389928


namespace NUMINAMATH_CALUDE_alcohol_percentage_solution_y_l3899_389973

theorem alcohol_percentage_solution_y :
  let alcohol_x : ℝ := 0.1  -- 10% alcohol in solution x
  let volume_x : ℝ := 300   -- 300 mL of solution x
  let volume_y : ℝ := 900   -- 900 mL of solution y
  let total_volume : ℝ := volume_x + volume_y
  let final_alcohol_percentage : ℝ := 0.25  -- 25% alcohol in final solution
  let alcohol_y : ℝ := (final_alcohol_percentage * total_volume - alcohol_x * volume_x) / volume_y
  alcohol_y = 0.3  -- 30% alcohol in solution y
  := by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_solution_y_l3899_389973


namespace NUMINAMATH_CALUDE_expressions_are_integers_l3899_389903

-- Define the expressions as functions
def expr1 (m n : ℕ) : ℚ := (m + n).factorial / (m.factorial * n.factorial)

def expr2 (m n : ℕ) : ℚ := ((2*m).factorial * (2*n).factorial) / 
  (m.factorial * n.factorial * (m + n).factorial)

def expr3 (m n : ℕ) : ℚ := ((5*m).factorial * (5*n).factorial) / 
  (m.factorial * n.factorial * (3*m + n).factorial * (3*n + m).factorial)

def expr4 (m n : ℕ) : ℚ := ((3*m + 3*n).factorial * (3*n).factorial * (2*m).factorial * (2*n).factorial) / 
  ((2*m + 3*n).factorial * (m + 2*n).factorial * m.factorial * (n.factorial^2) * (m + n).factorial)

-- Theorem statement
theorem expressions_are_integers (m n : ℕ) : 
  (∃ k : ℤ, expr1 m n = k) ∧ 
  (∃ k : ℤ, expr2 m n = k) ∧ 
  (∃ k : ℤ, expr3 m n = k) ∧ 
  (∃ k : ℤ, expr4 m n = k) := by
  sorry

end NUMINAMATH_CALUDE_expressions_are_integers_l3899_389903


namespace NUMINAMATH_CALUDE_students_getting_B_l3899_389909

theorem students_getting_B (grade_A : ℚ) (grade_C : ℚ) (grade_D : ℚ) (grade_F : ℚ) (passing_grade : ℚ) :
  grade_A = 1/4 →
  grade_C = 1/8 →
  grade_D = 1/12 →
  grade_F = 1/24 →
  passing_grade = 0.875 →
  grade_A + grade_C + grade_D + grade_F + 3/8 = passing_grade :=
by sorry

end NUMINAMATH_CALUDE_students_getting_B_l3899_389909


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l3899_389908

/-- A regular polygon with interior angles of 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l3899_389908


namespace NUMINAMATH_CALUDE_square_root_division_l3899_389929

theorem square_root_division : Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_division_l3899_389929


namespace NUMINAMATH_CALUDE_power_of_fraction_three_fourths_cubed_l3899_389982

theorem power_of_fraction_three_fourths_cubed :
  (3 / 4 : ℚ) ^ 3 = 27 / 64 := by
  sorry

end NUMINAMATH_CALUDE_power_of_fraction_three_fourths_cubed_l3899_389982


namespace NUMINAMATH_CALUDE_one_root_l3899_389935

/-- A quadratic function f(x) = x^2 + bx + c with discriminant 2020 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The discriminant of f(x) = 0 is 2020 -/
axiom discriminant_is_2020 (b c : ℝ) : b^2 - 4*c = 2020

/-- The equation f(x - 2020) + f(x) = 0 has exactly one root -/
theorem one_root (b c : ℝ) : ∃! x, f b c (x - 2020) + f b c x = 0 :=
sorry

end NUMINAMATH_CALUDE_one_root_l3899_389935


namespace NUMINAMATH_CALUDE_may_greatest_drop_l3899_389905

/-- Represents the months in the first half of 2022 -/
inductive Month
  | january
  | february
  | march
  | april
  | may
  | june

/-- Represents the price change for each month -/
def priceChange (m : Month) : ℝ :=
  match m with
  | .january => -1.0
  | .february => 1.5
  | .march => -3.0
  | .april => 2.0
  | .may => -4.0
  | .june => -1.5

/-- The economic event occurred in May -/
def economicEventMonth : Month := .may

/-- Defines the greatest monthly drop in price -/
def hasGreatestDrop (m : Month) : Prop :=
  ∀ m', priceChange m ≤ priceChange m'

/-- Theorem stating that May has the greatest monthly drop in price -/
theorem may_greatest_drop :
  hasGreatestDrop .may :=
sorry

end NUMINAMATH_CALUDE_may_greatest_drop_l3899_389905


namespace NUMINAMATH_CALUDE_school_sections_l3899_389991

theorem school_sections (boys girls : ℕ) (h1 : boys = 408) (h2 : girls = 288) :
  let section_size := Nat.gcd boys girls
  let boy_sections := boys / section_size
  let girl_sections := girls / section_size
  boy_sections + girl_sections = 29 := by
sorry

end NUMINAMATH_CALUDE_school_sections_l3899_389991


namespace NUMINAMATH_CALUDE_paint_joined_cubes_paint_divided_cube_cube_division_l3899_389920

-- Constants
def paint_coverage : ℝ := 100 -- 1 mL covers 100 cm²

-- Theorem 1
theorem paint_joined_cubes (small_edge large_edge : ℝ) (h1 : small_edge = 10) (h2 : large_edge = 20) :
  (6 * small_edge^2 + 6 * large_edge^2 - 2 * small_edge^2) / paint_coverage = 28 :=
sorry

-- Theorem 2
theorem paint_divided_cube (original_paint : ℝ) (h : original_paint = 54) :
  2 * (original_paint / 6) = 18 :=
sorry

-- Theorem 3
theorem cube_division (original_paint additional_paint : ℝ) (n : ℕ)
  (h1 : original_paint = 54) (h2 : additional_paint = 216) :
  6 * (original_paint / 6) * n = original_paint + additional_paint →
  n = 5 :=
sorry

end NUMINAMATH_CALUDE_paint_joined_cubes_paint_divided_cube_cube_division_l3899_389920


namespace NUMINAMATH_CALUDE_sector_arc_length_l3899_389958

/-- The length of an arc of a sector with given central angle and radius -/
def arcLength (centralAngle : Real) (radius : Real) : Real :=
  radius * centralAngle

theorem sector_arc_length :
  let centralAngle : Real := π / 5
  let radius : Real := 20
  arcLength centralAngle radius = 4 * π := by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3899_389958


namespace NUMINAMATH_CALUDE_min_max_inequality_l3899_389981

theorem min_max_inequality (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ (p q x m n y : ℕ),
    p = min a b ∧
    q = min c d ∧
    x = max p q ∧
    m = max a b ∧
    n = max c d ∧
    y = min m n ∧
    ((x > y) ∨ (x < y)) :=
by sorry

end NUMINAMATH_CALUDE_min_max_inequality_l3899_389981


namespace NUMINAMATH_CALUDE_revenue_in_scientific_notation_l3899_389941

/-- Represents 1 billion in scientific notation -/
def billion : ℝ := 10^9

/-- The tourism revenue in billions -/
def revenue : ℝ := 2.93

theorem revenue_in_scientific_notation : 
  revenue * billion = 2.93 * (10 : ℝ)^9 := by sorry

end NUMINAMATH_CALUDE_revenue_in_scientific_notation_l3899_389941


namespace NUMINAMATH_CALUDE_company_workers_l3899_389952

theorem company_workers (total : ℕ) (men : ℕ) : 
  (total / 3 : ℚ) = (total / 3 : ℕ) →
  (2 * total / 10 : ℚ) * (total / 3 : ℚ) = ((2 * total / 10 : ℕ) * (total / 3 : ℕ) : ℚ) →
  (4 * total / 10 : ℚ) * (2 * total / 3 : ℚ) = ((4 * total / 10 : ℕ) * (2 * total / 3 : ℕ) : ℚ) →
  men = 112 →
  (4 * total / 10 : ℚ) * (2 * total / 3 : ℚ) + (total / 3 : ℚ) - (2 * total / 10 : ℚ) * (total / 3 : ℚ) = men →
  total - men = 98 :=
by sorry

end NUMINAMATH_CALUDE_company_workers_l3899_389952


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3899_389943

theorem fraction_to_decimal : (17 : ℚ) / 50 = 0.34 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3899_389943


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_largest_coefficient_l3899_389918

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b * c / (b + c - a)) + (a * c / (a + c - b)) + (a * b / (a + b - c)) ≥ (a + b + c) :=
sorry

theorem largest_coefficient (k : ℝ) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b > c → b + c > a → c + a > b →
    (b * c / (b + c - a)) + (a * c / (a + c - b)) + (a * b / (a + b - c)) ≥ k * (a + b + c)) →
  k ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_largest_coefficient_l3899_389918


namespace NUMINAMATH_CALUDE_telescope_payment_difference_l3899_389942

theorem telescope_payment_difference (joan_payment karl_payment : ℕ) : 
  joan_payment = 158 →
  joan_payment + karl_payment = 400 →
  2 * joan_payment - karl_payment = 74 := by
sorry

end NUMINAMATH_CALUDE_telescope_payment_difference_l3899_389942


namespace NUMINAMATH_CALUDE_modulus_of_z_l3899_389926

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1) : Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l3899_389926


namespace NUMINAMATH_CALUDE_total_players_l3899_389988

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ)
  (h1 : kabadi = 10)
  (h2 : kho_kho_only = 30)
  (h3 : both = 5) :
  kabadi - both + kho_kho_only = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l3899_389988


namespace NUMINAMATH_CALUDE_sin_graph_shift_l3899_389957

open Real

theorem sin_graph_shift (f g : ℝ → ℝ) (ω : ℝ) (h : ω = 2) :
  (∀ x, f x = sin (ω * x + π / 6)) →
  (∀ x, g x = sin (ω * x)) →
  ∃ shift, ∀ x, f x = g (x - shift) ∧ shift = π / 12 :=
sorry

end NUMINAMATH_CALUDE_sin_graph_shift_l3899_389957


namespace NUMINAMATH_CALUDE_cereal_original_price_l3899_389904

def initial_money : ℝ := 60
def celery_price : ℝ := 5
def bread_price : ℝ := 8
def milk_original_price : ℝ := 10
def milk_discount : ℝ := 0.1
def potato_price : ℝ := 1
def potato_quantity : ℕ := 6
def money_left : ℝ := 26
def cereal_discount : ℝ := 0.5

theorem cereal_original_price :
  let milk_price := milk_original_price * (1 - milk_discount)
  let potato_total := potato_price * potato_quantity
  let spent_on_known_items := celery_price + bread_price + milk_price + potato_total
  let total_spent := initial_money - money_left
  let cereal_discounted_price := total_spent - spent_on_known_items
  cereal_discounted_price / (1 - cereal_discount) = 12 := by sorry

end NUMINAMATH_CALUDE_cereal_original_price_l3899_389904


namespace NUMINAMATH_CALUDE_arccos_neg_half_equals_two_pi_thirds_l3899_389932

theorem arccos_neg_half_equals_two_pi_thirds :
  Real.arccos (-1/2) = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_neg_half_equals_two_pi_thirds_l3899_389932


namespace NUMINAMATH_CALUDE_rex_saved_100_nickels_l3899_389956

/-- Represents the number of coins of each type saved by the children -/
structure Savings where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Converts a number of coins to their value in cents -/
def coinsToCents (s : Savings) : ℕ :=
  s.pennies + 5 * s.nickels + 10 * s.dimes

/-- The main theorem: Given the conditions, Rex saved 100 nickels -/
theorem rex_saved_100_nickels (s : Savings) 
    (h1 : s.pennies = 200)
    (h2 : s.dimes = 330)
    (h3 : coinsToCents s = 4000) : 
  s.nickels = 100 := by
  sorry

end NUMINAMATH_CALUDE_rex_saved_100_nickels_l3899_389956


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3899_389969

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * l + 2 * w = 2 * Real.pi * r) : 
  (l * w) / (Real.pi * r ^ 2) = 18 / Real.pi ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3899_389969


namespace NUMINAMATH_CALUDE_express_1997_using_fours_l3899_389931

theorem express_1997_using_fours : 
  4 * 444 + 44 * 4 + 44 + 4 / 4 = 1997 :=
by sorry

end NUMINAMATH_CALUDE_express_1997_using_fours_l3899_389931


namespace NUMINAMATH_CALUDE_work_completion_time_l3899_389974

def work_rate (days : ℕ) : ℚ := 1 / days

def johnson_rate : ℚ := work_rate 10
def vincent_rate : ℚ := work_rate 40
def alice_rate : ℚ := work_rate 20
def bob_rate : ℚ := work_rate 30

def day1_rate : ℚ := johnson_rate + vincent_rate
def day2_rate : ℚ := alice_rate + bob_rate

def two_day_cycle_rate : ℚ := day1_rate + day2_rate

theorem work_completion_time : ∃ n : ℕ, n * two_day_cycle_rate ≥ 1 ∧ n * 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3899_389974


namespace NUMINAMATH_CALUDE_range_of_y_given_inequality_l3899_389927

/-- Custom multiplication operation on real numbers -/
def custom_mult (x y : ℝ) : ℝ := x * (1 - y)

/-- The theorem stating the range of y given the conditions -/
theorem range_of_y_given_inequality :
  (∀ x : ℝ, custom_mult (x - y) (x + y) < 1) →
  ∃ a b : ℝ, a = -1/2 ∧ b = 3/2 ∧ ∀ y : ℝ, a < y ∧ y < b :=
by sorry

end NUMINAMATH_CALUDE_range_of_y_given_inequality_l3899_389927


namespace NUMINAMATH_CALUDE_polynomial_divisible_by_nine_l3899_389955

theorem polynomial_divisible_by_nine (n : ℤ) : ∃ k : ℤ, n^6 - 3*n^5 + 4*n^4 - 3*n^3 + 4*n^2 - 3*n = 9*k := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisible_by_nine_l3899_389955


namespace NUMINAMATH_CALUDE_statement_equivalence_l3899_389936

theorem statement_equivalence (x y : ℝ) :
  ((x > 1 ∧ y < -3) → x - y > 4) ↔ (x - y ≤ 4 → x ≤ 1 ∨ y ≥ -3) :=
by sorry

end NUMINAMATH_CALUDE_statement_equivalence_l3899_389936


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3899_389924

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  length : ℕ
  mean : ℚ
  first : ℚ
  last : ℚ

/-- The new mean after removing the first and last numbers -/
def new_mean (seq : ArithmeticSequence) : ℚ :=
  ((seq.length : ℚ) * seq.mean - seq.first - seq.last) / ((seq.length : ℚ) - 2)

/-- Theorem stating the property of the specific arithmetic sequence -/
theorem arithmetic_sequence_property :
  let seq := ArithmeticSequence.mk 60 42 30 70
  new_mean seq = 41.7241 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3899_389924


namespace NUMINAMATH_CALUDE_peach_difference_l3899_389992

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 12 →
  jake_peaches + 1 = jill_peaches →
  steven_peaches = jake_peaches + 16 →
  steven_peaches - jill_peaches = 15 := by
  sorry

end NUMINAMATH_CALUDE_peach_difference_l3899_389992


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l3899_389949

/-- A line passing through (1,2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  k : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (1,2) -/
  point_condition : k + b = 2
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : k ≠ -1 → b = k * b

/-- The equation of the line is either 2x - y = 0 or x + y - 3 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.k = 2 ∧ l.b = 0) ∨ (l.k = 1 ∧ l.b = 1) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l3899_389949


namespace NUMINAMATH_CALUDE_sequence_square_l3899_389912

theorem sequence_square (a : ℕ → ℕ) :
  a 1 = 1 ∧
  (∀ n : ℕ, n ≥ 2 → a n = a (n - 1) + 2 * n - 1) →
  ∀ n : ℕ, n > 0 → a n = n^2 := by
sorry

end NUMINAMATH_CALUDE_sequence_square_l3899_389912


namespace NUMINAMATH_CALUDE_alex_remaining_money_l3899_389959

def weekly_income : ℝ := 500
def tax_rate : ℝ := 0.10
def tithe_rate : ℝ := 0.10
def water_bill : ℝ := 55

theorem alex_remaining_money :
  weekly_income - (weekly_income * tax_rate + weekly_income * tithe_rate + water_bill) = 345 :=
by sorry

end NUMINAMATH_CALUDE_alex_remaining_money_l3899_389959


namespace NUMINAMATH_CALUDE_interest_rate_difference_l3899_389977

theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_diff : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) :
  principal = 6000 →
  time = 3 →
  interest_diff = 360 →
  (principal * rate2 * time) / 100 = (principal * rate1 * time) / 100 + interest_diff →
  rate2 - rate1 = 2 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l3899_389977


namespace NUMINAMATH_CALUDE_equal_side_sums_exist_l3899_389907

def triangle_numbers : List ℕ := List.range 9 |>.map (· + 2016)

structure TriangleArrangement where
  positions : Fin 9 → ℕ
  is_valid : ∀ n, positions n ∈ triangle_numbers

def side_sum (arr : TriangleArrangement) (side : Fin 3) : ℕ :=
  match side with
  | 0 => arr.positions 0 + arr.positions 1 + arr.positions 2
  | 1 => arr.positions 2 + arr.positions 3 + arr.positions 4
  | 2 => arr.positions 4 + arr.positions 5 + arr.positions 0

theorem equal_side_sums_exist : 
  ∃ (arr : TriangleArrangement), ∀ (i j : Fin 3), side_sum arr i = side_sum arr j :=
sorry

end NUMINAMATH_CALUDE_equal_side_sums_exist_l3899_389907


namespace NUMINAMATH_CALUDE_min_cost_for_zoo_visit_l3899_389917

/-- Represents the ticket pricing structure for the zoo --/
structure TicketPrices where
  adult : ℕ
  child : ℕ
  group : ℕ
  group_min : ℕ

/-- Calculates the total cost for a group given the pricing and number of adults and children --/
def calculate_cost (prices : TicketPrices) (adults children : ℕ) : ℕ :=
  min (prices.adult * adults + prices.child * children)
      (min (prices.group * (adults + children))
           (prices.group * prices.group_min + prices.child * (adults + children - prices.group_min)))

/-- Theorem stating the minimum cost for the given group --/
theorem min_cost_for_zoo_visit (prices : TicketPrices) 
    (h1 : prices.adult = 150)
    (h2 : prices.child = 60)
    (h3 : prices.group = 100)
    (h4 : prices.group_min = 5) :
  calculate_cost prices 4 7 = 860 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_zoo_visit_l3899_389917
