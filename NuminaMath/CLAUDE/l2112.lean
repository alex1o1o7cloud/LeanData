import Mathlib

namespace NUMINAMATH_CALUDE_decimal_sum_equals_fraction_l2112_211233

theorem decimal_sum_equals_fraction : 
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_equals_fraction_l2112_211233


namespace NUMINAMATH_CALUDE_paths_on_4x10_grid_with_forbidden_segments_l2112_211225

/-- Represents a grid with forbidden segments -/
structure Grid where
  height : ℕ
  width : ℕ
  forbidden_segments : ℕ

/-- Calculates the number of paths on a grid with forbidden segments -/
def count_paths (g : Grid) : ℕ :=
  let total_paths := Nat.choose (g.height + g.width) g.height
  let forbidden_paths := g.forbidden_segments * (Nat.choose (g.height + g.width / 2 - 2) (g.height - 2) * Nat.choose (g.width / 2 + 2) 2)
  total_paths - forbidden_paths

/-- Theorem stating the number of paths on a 4x10 grid with two forbidden segments -/
theorem paths_on_4x10_grid_with_forbidden_segments :
  count_paths { height := 4, width := 10, forbidden_segments := 2 } = 161 := by
  sorry

end NUMINAMATH_CALUDE_paths_on_4x10_grid_with_forbidden_segments_l2112_211225


namespace NUMINAMATH_CALUDE_min_total_cost_l2112_211241

-- Define the probabilities and costs
def prob_event : ℝ := 0.3
def loss : ℝ := 400 -- in ten thousand yuan
def cost_A : ℝ := 45 -- in ten thousand yuan
def cost_B : ℝ := 30 -- in ten thousand yuan
def prob_no_event_A : ℝ := 0.9
def prob_no_event_B : ℝ := 0.85

-- Define the total cost function for each scenario
def total_cost_none : ℝ := prob_event * loss
def total_cost_A : ℝ := cost_A + (1 - prob_no_event_A) * loss
def total_cost_B : ℝ := cost_B + (1 - prob_no_event_B) * loss
def total_cost_both : ℝ := cost_A + cost_B + (1 - prob_no_event_A * prob_no_event_B) * loss

-- Theorem: Implementing measure A results in the minimum total cost
theorem min_total_cost :
  total_cost_A = 85 ∧ 
  total_cost_A ≤ total_cost_none ∧ 
  total_cost_A ≤ total_cost_B ∧ 
  total_cost_A ≤ total_cost_both :=
sorry

end NUMINAMATH_CALUDE_min_total_cost_l2112_211241


namespace NUMINAMATH_CALUDE_sqrt_15_times_sqrt_3_minus_4_between_2_and_3_l2112_211293

theorem sqrt_15_times_sqrt_3_minus_4_between_2_and_3 :
  2 < Real.sqrt 15 * Real.sqrt 3 - 4 ∧ Real.sqrt 15 * Real.sqrt 3 - 4 < 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_times_sqrt_3_minus_4_between_2_and_3_l2112_211293


namespace NUMINAMATH_CALUDE_library_repacking_l2112_211243

/-- Given a number of boxes and books per box, calculate the number of books left over when repacking into new boxes with a different number of books per box. -/
def books_left_over (initial_boxes : ℕ) (initial_books_per_box : ℕ) (new_books_per_box : ℕ) : ℕ :=
  let total_books := initial_boxes * initial_books_per_box
  total_books % new_books_per_box

/-- Prove that given 1575 boxes with 45 books each, when repacking into boxes of 50 books each, the number of books left over is 25. -/
theorem library_repacking : books_left_over 1575 45 50 = 25 := by
  sorry

end NUMINAMATH_CALUDE_library_repacking_l2112_211243


namespace NUMINAMATH_CALUDE_unique_cds_l2112_211234

theorem unique_cds (shared : ℕ) (alice_total : ℕ) (bob_unique : ℕ) 
  (h1 : shared = 12)
  (h2 : alice_total = 23)
  (h3 : bob_unique = 8) :
  alice_total - shared + bob_unique = 19 :=
by sorry

end NUMINAMATH_CALUDE_unique_cds_l2112_211234


namespace NUMINAMATH_CALUDE_equal_ratios_sum_l2112_211269

theorem equal_ratios_sum (K L M : ℚ) : 
  (4 : ℚ) / 7 = K / 63 ∧ (4 : ℚ) / 7 = 84 / L ∧ (4 : ℚ) / 7 = M / 98 → 
  K + L + M = 239 := by
  sorry

end NUMINAMATH_CALUDE_equal_ratios_sum_l2112_211269


namespace NUMINAMATH_CALUDE_square_nine_on_top_l2112_211284

-- Define the grid of squares
def Grid := Fin 4 → Fin 4 → Fin 16

-- Define the initial configuration of the grid
def initial_grid : Grid :=
  fun i j => i * 4 + j + 1

-- Define the folding operations
def fold_top_over_bottom (g : Grid) : Grid :=
  fun i j => g (3 - i) j

def fold_bottom_over_top (g : Grid) : Grid :=
  fun i j => g i j

def fold_right_over_left (g : Grid) : Grid :=
  fun i j => g i (3 - j)

def fold_left_over_right (g : Grid) : Grid :=
  fun i j => g i j

-- Define the complete folding sequence
def fold_sequence (g : Grid) : Grid :=
  fold_left_over_right ∘ fold_right_over_left ∘ fold_bottom_over_top ∘ fold_top_over_bottom $ g

-- Theorem stating that after the folding sequence, square 9 is on top
theorem square_nine_on_top :
  (fold_sequence initial_grid) 0 0 = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_nine_on_top_l2112_211284


namespace NUMINAMATH_CALUDE_subtraction_correction_l2112_211244

theorem subtraction_correction (x : ℤ) : x - 63 = 24 → x - 36 = 51 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_correction_l2112_211244


namespace NUMINAMATH_CALUDE_f_zero_at_three_l2112_211256

/-- The polynomial function f(x) = 3x^4 - 2x^3 + x^2 - 4x + r -/
def f (x r : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + x^2 - 4 * x + r

/-- Theorem stating that f(3) = 0 if and only if r = -186 -/
theorem f_zero_at_three (r : ℝ) : f 3 r = 0 ↔ r = -186 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_at_three_l2112_211256


namespace NUMINAMATH_CALUDE_vector_magnitude_l2112_211231

/-- Given two vectors a and b in ℝ², where a is parallel to (a - b),
    prove that the magnitude of (a + b) is 3√5/2. -/
theorem vector_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![x, 1]
  (∃ (k : ℝ), a = k • (a - b)) →
  ‖a + b‖ = 3 * Real.sqrt 5 / 2 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l2112_211231


namespace NUMINAMATH_CALUDE_square_root_of_four_l2112_211216

theorem square_root_of_four : 
  {x : ℝ | x^2 = 4} = {2, -2} := by sorry

end NUMINAMATH_CALUDE_square_root_of_four_l2112_211216


namespace NUMINAMATH_CALUDE_pet_store_birds_l2112_211298

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := 4

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 8

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 2

/-- The total number of birds in the pet store -/
def total_birds : ℕ := num_cages * (parrots_per_cage + parakeets_per_cage)

theorem pet_store_birds :
  total_birds = 40 :=
sorry

end NUMINAMATH_CALUDE_pet_store_birds_l2112_211298


namespace NUMINAMATH_CALUDE_inequality_proof_l2112_211221

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2112_211221


namespace NUMINAMATH_CALUDE_valid_triples_l2112_211273

-- Define the type for our triples
def Triple := (Nat × Nat × Nat)

-- Define the conditions
def satisfiesConditions (t : Triple) : Prop :=
  let (a, b, c) := t
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧  -- positive integers
  (a ≤ b) ∧ (b ≤ c) ∧  -- ordered
  (Nat.gcd a (Nat.gcd b c) = 1) ∧  -- gcd(a,b,c) = 1
  ((a + b + c) ∣ (a^12 + b^12 + c^12)) ∧
  ((a + b + c) ∣ (a^23 + b^23 + c^23)) ∧
  ((a + b + c) ∣ (a^11004 + b^11004 + c^11004))

-- The theorem
theorem valid_triples :
  {t : Triple | satisfiesConditions t} = {(1,1,1), (1,1,4)} := by
  sorry

end NUMINAMATH_CALUDE_valid_triples_l2112_211273


namespace NUMINAMATH_CALUDE_black_raisins_amount_l2112_211230

/-- The amount of yellow raisins added (in cups) -/
def yellow_raisins : ℝ := 0.3

/-- The total amount of raisins added (in cups) -/
def total_raisins : ℝ := 0.7

/-- The amount of black raisins added (in cups) -/
def black_raisins : ℝ := total_raisins - yellow_raisins

theorem black_raisins_amount : black_raisins = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_black_raisins_amount_l2112_211230


namespace NUMINAMATH_CALUDE_polyhedron_sum_l2112_211252

structure Polyhedron where
  faces : ℕ
  triangles : ℕ
  pentagons : ℕ
  T : ℕ
  P : ℕ
  V : ℕ
  faces_sum : faces = triangles + pentagons
  faces_20 : faces = 20
  triangles_twice_pentagons : triangles = 2 * pentagons
  euler : V - ((3 * triangles + 5 * pentagons) / 2) + faces = 2

def vertex_sum (poly : Polyhedron) : ℕ := 100 * poly.P + 10 * poly.T + poly.V

theorem polyhedron_sum (poly : Polyhedron) (h1 : poly.T = 2) (h2 : poly.P = 2) : 
  vertex_sum poly = 238 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_sum_l2112_211252


namespace NUMINAMATH_CALUDE_laptop_final_price_l2112_211294

/-- The final price of a laptop after successive discounts --/
theorem laptop_final_price (original_price : ℝ) (discount1 discount2 discount3 : ℝ) :
  original_price = 1200 →
  discount1 = 0.1 →
  discount2 = 0.2 →
  discount3 = 0.05 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) = 820.80 := by
  sorry

#check laptop_final_price

end NUMINAMATH_CALUDE_laptop_final_price_l2112_211294


namespace NUMINAMATH_CALUDE_sixth_doll_size_l2112_211278

def doll_size (n : ℕ) : ℚ :=
  243 * (2/3)^(n-1)

theorem sixth_doll_size : doll_size 6 = 32 := by
  sorry

end NUMINAMATH_CALUDE_sixth_doll_size_l2112_211278


namespace NUMINAMATH_CALUDE_last_number_not_one_l2112_211248

def board_sum : ℕ := (2012 * 2013) / 2

theorem last_number_not_one :
  ∀ (operations : ℕ) (final_number : ℕ),
    (operations < 2011 → final_number ≠ 1) ∧
    (operations = 2011 → final_number % 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_last_number_not_one_l2112_211248


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2112_211295

theorem polynomial_factorization (x : ℝ) :
  (x^3 - x + 3)^2 = x^6 - 2*x^4 + 6*x^3 + x^2 - 6*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2112_211295


namespace NUMINAMATH_CALUDE_solve_for_y_l2112_211260

theorem solve_for_y (x y : ℚ) (h1 : x - y = 20) (h2 : 3 * (x + y) = 15) : y = -15/2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2112_211260


namespace NUMINAMATH_CALUDE_function_characterization_l2112_211258

theorem function_characterization 
  (f : ℕ → ℕ) 
  (h1 : ∀ x y : ℕ, (x + y) ∣ (f x + f y))
  (h2 : ∀ x : ℕ, x ≥ 1395 → x^3 ≥ 2 * f x) :
  ∃ k : ℕ, k ≤ 1395^2 / 2 ∧ ∀ n : ℕ, f n = k * n :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l2112_211258


namespace NUMINAMATH_CALUDE_second_triangle_invalid_l2112_211280

-- Define the sides of the triangle
def a : ℝ := 15
def b : ℝ := 15
def c : ℝ := 30

-- Define the condition for a valid triangle (triangle inequality)
def is_valid_triangle (x y z : ℝ) : Prop :=
  x + y > z ∧ y + z > x ∧ z + x > y

-- Theorem statement
theorem second_triangle_invalid :
  ¬(is_valid_triangle a b c) :=
sorry

end NUMINAMATH_CALUDE_second_triangle_invalid_l2112_211280


namespace NUMINAMATH_CALUDE_tangent_sum_l2112_211235

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition that f is tangent to y = -x + 8 at (5, f(5))
def is_tangent_at_5 (f : ℝ → ℝ) : Prop :=
  f 5 = -5 + 8 ∧ deriv f 5 = -1

-- State the theorem
theorem tangent_sum (f : ℝ → ℝ) (h : is_tangent_at_5 f) :
  f 5 + deriv f 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_l2112_211235


namespace NUMINAMATH_CALUDE_intersection_locus_is_circle_l2112_211292

/-- The locus of intersection points of two parametric lines forms a circle -/
theorem intersection_locus_is_circle :
  ∀ (x y u : ℝ),
  (u * x - 3 * y - 2 * u = 0) →
  (2 * x - 3 * u * y + u = 0) →
  ∃ (center_x center_y radius : ℝ),
  (x - center_x)^2 + (y - center_y)^2 = radius^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_locus_is_circle_l2112_211292


namespace NUMINAMATH_CALUDE_only_math_is_75_l2112_211245

/-- Represents the number of students in different subject combinations -/
structure StudentCounts where
  total : ℕ
  math : ℕ
  foreignLanguage : ℕ
  science : ℕ
  allThree : ℕ

/-- The actual student counts from the problem -/
def actualCounts : StudentCounts :=
  { total := 120
  , math := 85
  , foreignLanguage := 65
  , science := 75
  , allThree := 20 }

/-- Calculate the number of students taking only math -/
def onlyMathCount (counts : StudentCounts) : ℕ :=
  counts.math - (counts.total - (counts.math + counts.foreignLanguage + counts.science - counts.allThree))

/-- Theorem stating that the number of students taking only math is 75 -/
theorem only_math_is_75 : onlyMathCount actualCounts = 75 := by
  sorry

end NUMINAMATH_CALUDE_only_math_is_75_l2112_211245


namespace NUMINAMATH_CALUDE_min_perimeter_non_congruent_isosceles_triangles_l2112_211272

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- The area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

theorem min_perimeter_non_congruent_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    t1.base * 9 = t2.base * 10 ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      s1.base * 9 = s2.base * 10 →
      perimeter t1 ≤ perimeter s1 ∧
      perimeter t1 = 728 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_non_congruent_isosceles_triangles_l2112_211272


namespace NUMINAMATH_CALUDE_expression_evaluation_l2112_211286

theorem expression_evaluation (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (a*b + b*c + c*d + d*a + a*c + b*d)⁻¹ *
  ((a*b)⁻¹ + (b*c)⁻¹ + (c*d)⁻¹ + (d*a)⁻¹ + (a*c)⁻¹ + (b*d)⁻¹) = (a*b*c*d)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2112_211286


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2112_211274

theorem complex_magnitude_problem (z : ℂ) (h : z = 3 + I) :
  Complex.abs (z^2 - 3*z) = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2112_211274


namespace NUMINAMATH_CALUDE_line_in_first_third_quadrants_positive_slope_l2112_211249

/-- A line passing through the first and third quadrants -/
structure LineInFirstThirdQuadrants where
  k : ℝ
  k_neq_zero : k ≠ 0
  passes_through_first_third : ∀ x y : ℝ, y = k * x → 
    ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0))

/-- Theorem: If a line y = kx passes through the first and third quadrants, then k > 0 -/
theorem line_in_first_third_quadrants_positive_slope 
  (line : LineInFirstThirdQuadrants) : line.k > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_in_first_third_quadrants_positive_slope_l2112_211249


namespace NUMINAMATH_CALUDE_cookie_brownie_difference_l2112_211220

/-- Represents the daily cookie and brownie activity -/
structure DailyActivity where
  eaten_cookies : ℕ
  eaten_brownies : ℕ
  baked_cookies : ℕ
  baked_brownies : ℕ

/-- Calculates the final number of cookies and brownies after a week -/
def final_counts (initial_cookies : ℕ) (initial_brownies : ℕ) (activities : List DailyActivity) : ℕ × ℕ :=
  activities.foldl
    (fun (acc : ℕ × ℕ) (day : DailyActivity) =>
      (acc.1 - day.eaten_cookies + day.baked_cookies,
       acc.2 - day.eaten_brownies + day.baked_brownies))
    (initial_cookies, initial_brownies)

/-- The theorem to be proved -/
theorem cookie_brownie_difference :
  let initial_cookies := 60
  let initial_brownies := 10
  let activities : List DailyActivity := [
    ⟨2, 1, 10, 0⟩,
    ⟨4, 2, 0, 4⟩,
    ⟨3, 1, 5, 2⟩,
    ⟨5, 1, 0, 0⟩,
    ⟨4, 3, 8, 0⟩,
    ⟨3, 2, 0, 1⟩,
    ⟨2, 1, 0, 5⟩
  ]
  let (final_cookies, final_brownies) := final_counts initial_cookies initial_brownies activities
  final_cookies - final_brownies = 49 := by
  sorry

end NUMINAMATH_CALUDE_cookie_brownie_difference_l2112_211220


namespace NUMINAMATH_CALUDE_sequence_ratio_l2112_211209

-- Define the arithmetic sequence
def arithmetic_sequence (a b : ℝ) : Prop :=
  ∃ r : ℝ, b - a = r ∧ -4 - b = r ∧ a - (-1) = r

-- Define the geometric sequence
def geometric_sequence (c d e : ℝ) : Prop :=
  ∃ q : ℝ, c = -1 * q ∧ d = c * q ∧ e = d * q ∧ -4 = e * q

-- State the theorem
theorem sequence_ratio (a b c d e : ℝ) 
  (h1 : arithmetic_sequence a b)
  (h2 : geometric_sequence c d e) :
  (b - a) / d = 1/2 := by sorry

end NUMINAMATH_CALUDE_sequence_ratio_l2112_211209


namespace NUMINAMATH_CALUDE_length_of_DE_l2112_211213

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the angles
variable (angle_BAE angle_CBE angle_DCE : ℝ)

-- Define the side lengths
variable (AE AB BC CD : ℝ)

-- Define t
variable (t : ℝ)

-- State the theorem
theorem length_of_DE (h1 : angle_BAE = 90) (h2 : angle_CBE = 90) (h3 : angle_DCE = 90)
                     (h4 : AE = Real.sqrt 5) (h5 : AB = Real.sqrt 4) (h6 : BC = Real.sqrt 3)
                     (h7 : CD = Real.sqrt t) (h8 : t = 4) :
  Real.sqrt ((CD^2) + (Real.sqrt ((BC^2) + (Real.sqrt (AB^2 + AE^2))^2))^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_length_of_DE_l2112_211213


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_37_seconds_l2112_211259

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (train_length : ℝ) 
  (initial_distance : ℝ) : ℝ :=
  let jogger_speed_ms := jogger_speed * 1000 / 3600
  let train_speed_ms := train_speed * 1000 / 3600
  let relative_speed := train_speed_ms - jogger_speed_ms
  let total_distance := initial_distance + train_length
  total_distance / relative_speed

/-- The train passes the jogger in 37 seconds under the given conditions -/
theorem train_passes_jogger_in_37_seconds : 
  train_passing_jogger_time 9 45 120 250 = 37 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_train_passes_jogger_in_37_seconds_l2112_211259


namespace NUMINAMATH_CALUDE_first_player_wins_l2112_211229

/-- Represents a position on the rectangular table -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents the state of the game -/
structure GameState :=
  (table : Set Position)
  (occupied : Set Position)
  (currentPlayer : Bool)

/-- Checks if a position is valid on the table -/
def isValidPosition (state : GameState) (pos : Position) : Prop :=
  pos ∈ state.table ∧ pos ∉ state.occupied

/-- Represents a move in the game -/
def makeMove (state : GameState) (pos : Position) : GameState :=
  { state with
    occupied := state.occupied ∪ {pos},
    currentPlayer := ¬state.currentPlayer
  }

/-- Checks if the game is over (no more valid moves) -/
def isGameOver (state : GameState) : Prop :=
  ∀ pos, pos ∈ state.table → pos ∈ state.occupied

/-- Theorem: The first player has a winning strategy -/
theorem first_player_wins :
  ∀ (initialState : GameState),
  initialState.currentPlayer = true →
  ∃ (strategy : GameState → Position),
  (∀ state, isValidPosition state (strategy state)) →
  (∀ state, ¬isGameOver state → 
    ∃ (opponentMove : Position),
    isValidPosition state opponentMove →
    isGameOver (makeMove (makeMove state (strategy state)) opponentMove)) :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l2112_211229


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2112_211281

theorem fraction_evaluation : (36 - 12) / (12 - 4) = 3 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2112_211281


namespace NUMINAMATH_CALUDE_complex_modulus_product_l2112_211217

theorem complex_modulus_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l2112_211217


namespace NUMINAMATH_CALUDE_meal_combinations_l2112_211267

/-- The number of dishes available in the restaurant -/
def num_dishes : ℕ := 15

/-- The number of ways one person can choose their meal -/
def individual_choices (n : ℕ) : ℕ := n + n * n

/-- The total number of meal combinations for two people -/
def total_combinations (n : ℕ) : ℕ := (individual_choices n) * (individual_choices n)

/-- Theorem stating the total number of meal combinations -/
theorem meal_combinations : total_combinations num_dishes = 57600 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l2112_211267


namespace NUMINAMATH_CALUDE_smallest_n_for_interval_condition_l2112_211266

theorem smallest_n_for_interval_condition : ∃ (n : ℕ), n > 0 ∧
  (∀ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 →
    ∃ (k : ℕ), (m : ℚ) / 1993 < (k : ℚ) / n ∧ (k : ℚ) / n < ((m : ℚ) + 1) / 1994) ∧
  (∀ (n' : ℕ), 0 < n' ∧ n' < n →
    ∃ (m : ℕ), 1 ≤ m ∧ m ≤ 1992 ∧
      ∀ (k : ℕ), ((m : ℚ) / 1993 ≥ (k : ℚ) / n' ∨ (k : ℚ) / n' ≥ ((m : ℚ) + 1) / 1994)) ∧
  n = 3987 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_interval_condition_l2112_211266


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l2112_211282

/-- Given two parallel 2D vectors a and b, prove that 2a - b equals (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →
  (2 • a - b) = ![4, -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l2112_211282


namespace NUMINAMATH_CALUDE_triangle_sum_equals_22_l2112_211237

/-- The triangle operation defined as 2a - b + c -/
def triangle_op (a b c : ℤ) : ℤ := 2*a - b + c

/-- The vertices of the first triangle -/
def triangle1 : List ℤ := [3, 7, 5]

/-- The vertices of the second triangle -/
def triangle2 : List ℤ := [6, 2, 8]

theorem triangle_sum_equals_22 : 
  triangle_op triangle1[0] triangle1[1] triangle1[2] + 
  triangle_op triangle2[0] triangle2[1] triangle2[2] = 22 := by
sorry

end NUMINAMATH_CALUDE_triangle_sum_equals_22_l2112_211237


namespace NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_l2112_211291

/-- Represents a hyperbola in the xy-plane -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  p : ℝ
  vertex : ℝ × ℝ
  focus : ℝ × ℝ
  equation : ℝ → ℝ → Prop

/-- Given hyperbola and conditions on a parabola, proves the equation of the parabola -/
theorem parabola_equation_from_hyperbola (h : Hyperbola) (p : Parabola) :
  h.equation = (fun x y => 16 * x^2 - 9 * y^2 = 144) →
  p.vertex = (0, 0) →
  (p.focus = (3, 0) ∨ p.focus = (-3, 0)) →
  (p.equation = (fun x y => y^2 = 24 * x) ∨ p.equation = (fun x y => y^2 = -24 * x)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_from_hyperbola_l2112_211291


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2112_211290

theorem fraction_to_decimal : (31 : ℚ) / (2 * 5^6) = 0.000992 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2112_211290


namespace NUMINAMATH_CALUDE_root_sum_theorem_l2112_211200

theorem root_sum_theorem (a b c : ℝ) : 
  a^3 - 24*a^2 + 50*a - 42 = 0 →
  b^3 - 24*b^2 + 50*b - 42 = 0 →
  c^3 - 24*c^2 + 50*c - 42 = 0 →
  (a / (1/a + b*c)) + (b / (1/b + c*a)) + (c / (1/c + a*b)) = 476/43 := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l2112_211200


namespace NUMINAMATH_CALUDE_not_strictly_monotone_sequence_l2112_211204

/-- d(k) denotes the number of natural divisors of a natural number k -/
def d (k : ℕ) : ℕ := (Finset.filter (· ∣ k) (Finset.range (k + 1))).card

/-- The sequence {d(n^2+1)}_{n=n_0}^∞ is not strictly monotone -/
theorem not_strictly_monotone_sequence (n_0 : ℕ) :
  ∃ m n : ℕ, m > n ∧ n ≥ n_0 ∧ d (m^2 + 1) ≤ d (n^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_not_strictly_monotone_sequence_l2112_211204


namespace NUMINAMATH_CALUDE_seats_per_bus_is_60_field_trip_problem_l2112_211227

/-- Represents the field trip scenario -/
structure FieldTrip where
  total_students : ℕ
  num_buses : ℕ
  all_accommodated : Bool

/-- Calculates the number of seats per bus -/
def seats_per_bus (trip : FieldTrip) : ℕ :=
  trip.total_students / trip.num_buses

/-- Theorem stating that the number of seats per bus is 60 -/
theorem seats_per_bus_is_60 (trip : FieldTrip) 
  (h1 : trip.total_students = 180)
  (h2 : trip.num_buses = 3)
  (h3 : trip.all_accommodated = true) : 
  seats_per_bus trip = 60 := by
  sorry

/-- Main theorem proving the field trip problem -/
theorem field_trip_problem : 
  ∃ (trip : FieldTrip), seats_per_bus trip = 60 ∧ 
    trip.total_students = 180 ∧ 
    trip.num_buses = 3 ∧ 
    trip.all_accommodated = true := by
  sorry

end NUMINAMATH_CALUDE_seats_per_bus_is_60_field_trip_problem_l2112_211227


namespace NUMINAMATH_CALUDE_odd_digits_base4_157_l2112_211255

/-- Converts a natural number to its base-4 representation -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of odd digits in a list of natural numbers -/
def countOddDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The number of odd digits in the base-4 representation of 157₁₀ is 2 -/
theorem odd_digits_base4_157 : countOddDigits (toBase4 157) = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_digits_base4_157_l2112_211255


namespace NUMINAMATH_CALUDE_solution_set_equivalent_to_inequality_l2112_211268

def solution_set : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

def inequality (x : ℝ) : Prop := -x^2 + 3*x - 2 ≥ 0

theorem solution_set_equivalent_to_inequality :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
by sorry

end NUMINAMATH_CALUDE_solution_set_equivalent_to_inequality_l2112_211268


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2112_211275

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 → x ≠ 3 → x ≠ 4 →
  (x^3 - 4*x^2 + 5*x - 7) / ((x - 1)*(x - 2)*(x - 3)*(x - 4)) =
  5/6 / (x - 1) + (-5/2) / (x - 2) + 1/2 / (x - 3) + 13/6 / (x - 4) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2112_211275


namespace NUMINAMATH_CALUDE_expression_equality_l2112_211202

theorem expression_equality (θ : Real) 
  (h1 : π / 4 < θ) (h2 : θ < π / 2) : 
  2 * Real.cos θ + Real.sqrt (1 - 2 * Real.sin (π - θ) * Real.cos θ) = Real.sin θ + Real.cos θ := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2112_211202


namespace NUMINAMATH_CALUDE_at_least_one_basketball_l2112_211212

/-- Represents the total number of balls -/
def totalBalls : ℕ := 8

/-- Represents the number of basketballs -/
def numBasketballs : ℕ := 6

/-- Represents the number of volleyballs -/
def numVolleyballs : ℕ := 2

/-- Represents the number of balls to be chosen -/
def chosenBalls : ℕ := 3

/-- Theorem stating that at least one basketball is always chosen -/
theorem at_least_one_basketball : 
  ∀ (selection : Finset (Fin totalBalls)), 
  selection.card = chosenBalls → 
  ∃ (i : Fin totalBalls), i ∈ selection ∧ i.val < numBasketballs :=
sorry

end NUMINAMATH_CALUDE_at_least_one_basketball_l2112_211212


namespace NUMINAMATH_CALUDE_lowest_sale_price_percentage_l2112_211285

theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (additional_discount : ℝ) : 
  list_price = 80 ∧ 
  max_regular_discount = 0.5 ∧ 
  additional_discount = 0.2 → 
  (list_price * (1 - max_regular_discount) - list_price * additional_discount) / list_price = 0.3 := by
sorry

end NUMINAMATH_CALUDE_lowest_sale_price_percentage_l2112_211285


namespace NUMINAMATH_CALUDE_chord_length_is_four_l2112_211240

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := y = x + 2

def C2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

-- Define the chord length
def chord_length (C1 C2 : ℝ → ℝ → Prop) : ℝ :=
  4 -- The actual calculation is omitted, we just state the result

-- Theorem statement
theorem chord_length_is_four :
  chord_length C1 C2 = 4 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_four_l2112_211240


namespace NUMINAMATH_CALUDE_min_value_abc_l2112_211250

theorem min_value_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + 2*a*b + 2*a*c + 4*b*c = 16) :
  ∃ m : ℝ, ∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
  x^2 + 2*x*y + 2*x*z + 4*y*z = 16 → m ≤ x + y + z :=
sorry

end NUMINAMATH_CALUDE_min_value_abc_l2112_211250


namespace NUMINAMATH_CALUDE_local_min_implies_a_eq_one_l2112_211271

/-- The function f(x) = x³ - 2ax² + a²x + 1 has a local minimum at x = 1 -/
def has_local_min_at_one (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ), f x ≥ f 1

/-- The function f(x) = x³ - 2ax² + a²x + 1 -/
def f (a x : ℝ) : ℝ := x^3 - 2*a*x^2 + a^2*x + 1

theorem local_min_implies_a_eq_one (a : ℝ) :
  has_local_min_at_one (f a) a → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_local_min_implies_a_eq_one_l2112_211271


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2112_211205

theorem quadratic_equation_roots (m n : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + n = 0 ∧ x₂^2 + m*x₂ + n = 0) →
  (n = 3 - m ∧ (∀ x : ℝ, x^2 + m*x + n = 0 → x < 0) → 2 ≤ m ∧ m < 3) ∧
  (∃ t : ℝ, ∀ m n : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + n = 0 ∧ x₂^2 + m*x₂ + n = 0) →
    t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2 ∧
    t = 9/8 ∧
    ∀ t' : ℝ, (∀ m n : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + n = 0 ∧ x₂^2 + m*x₂ + n = 0) →
      t' ≤ (m-1)^2 + (n-1)^2 + (m-n)^2) → t' ≤ t) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2112_211205


namespace NUMINAMATH_CALUDE_road_length_l2112_211247

/-- Given 10 trees planted on one side of a road at intervals of 10 meters,
    with trees at both ends, prove that the length of the road is 90 meters. -/
theorem road_length (num_trees : ℕ) (interval : ℕ) : 
  num_trees = 10 → interval = 10 → (num_trees - 1) * interval = 90 := by
  sorry

end NUMINAMATH_CALUDE_road_length_l2112_211247


namespace NUMINAMATH_CALUDE_parabola_translation_l2112_211283

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let original := Parabola.mk 6 0 0
  let translated := translate original 2 3
  y = 6 * x^2 → y = translated.a * (x - 2)^2 + translated.b * (x - 2) + translated.c :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2112_211283


namespace NUMINAMATH_CALUDE_conic_is_hyperbola_l2112_211257

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop := x^2 + 2*x - 8*y^2 = 0

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b h k : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  ∀ x y, f x y ↔ (x - h)^2 / a^2 - (y - k)^2 / b^2 = 1

/-- Theorem stating that the given equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation := by
  sorry

end NUMINAMATH_CALUDE_conic_is_hyperbola_l2112_211257


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2112_211207

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 4*I
  let z₂ : ℂ := -1 + 2*I
  z₁ / z₂ = -5/3 + 10/3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2112_211207


namespace NUMINAMATH_CALUDE_problem_solution_l2112_211276

theorem problem_solution : 
  (-24 / (1/2 - 1/6 + 1/3) = -36) ∧ 
  (-1^3 - |(-9)| + 3 + 6 * (-1/3)^2 = -19/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2112_211276


namespace NUMINAMATH_CALUDE_specific_prism_properties_l2112_211288

/-- A right prism with a triangular base -/
structure TriangularPrism where
  base_side_a : ℝ
  base_side_b : ℝ
  base_side_c : ℝ
  section_cut_a : ℝ
  section_cut_b : ℝ
  section_cut_c : ℝ

/-- Calculate the volume of the bounded figure -/
def bounded_volume (prism : TriangularPrism) : ℝ :=
  sorry

/-- Calculate the total surface area of the bounded figure -/
def bounded_surface_area (prism : TriangularPrism) : ℝ :=
  sorry

/-- Theorem stating the volume and surface area of the specific prism -/
theorem specific_prism_properties :
  let prism : TriangularPrism := {
    base_side_a := 6,
    base_side_b := 8,
    base_side_c := 10,
    section_cut_a := 12,
    section_cut_b := 12,
    section_cut_c := 18
  }
  bounded_volume prism = 336 ∧ bounded_surface_area prism = 396 :=
by sorry

end NUMINAMATH_CALUDE_specific_prism_properties_l2112_211288


namespace NUMINAMATH_CALUDE_ratio_of_fractions_l2112_211214

theorem ratio_of_fractions (x y : ℝ) (h1 : 5 * x = 6 * y) (h2 : x * y ≠ 0) :
  (1 / 3 * x) / (1 / 5 * y) = 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_fractions_l2112_211214


namespace NUMINAMATH_CALUDE_fibonacci_primitive_roots_l2112_211264

theorem fibonacci_primitive_roots (p : Nat) (g : Nat) (k : Nat) 
    (h1 : Nat.Prime p)
    (h2 : IsPrimitiveRoot g p)
    (h3 : g^2 % p = (g + 1) % p)
    (h4 : p = 4*k + 3) :
  IsPrimitiveRoot (g - 1) p ∧ 
  (g - 1)^(2*k + 3) % p = (g - 2) % p ∧
  IsPrimitiveRoot (g - 2) p :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_primitive_roots_l2112_211264


namespace NUMINAMATH_CALUDE_david_trip_expenses_l2112_211224

theorem david_trip_expenses (initial_amount remaining_amount : ℕ) 
  (h1 : initial_amount = 1800)
  (h2 : remaining_amount = 500)
  (h3 : initial_amount > remaining_amount) :
  initial_amount - remaining_amount - remaining_amount = 800 := by
  sorry

end NUMINAMATH_CALUDE_david_trip_expenses_l2112_211224


namespace NUMINAMATH_CALUDE_annes_age_l2112_211279

theorem annes_age (maude emile anne : ℕ) 
  (h1 : anne = 2 * emile)
  (h2 : emile = 6 * maude)
  (h3 : maude = 8) :
  anne = 96 := by
  sorry

end NUMINAMATH_CALUDE_annes_age_l2112_211279


namespace NUMINAMATH_CALUDE_inverse_function_domain_l2112_211262

-- Define the function f(x) = -x(x+2)
def f (x : ℝ) : ℝ := -x * (x + 2)

-- State the theorem
theorem inverse_function_domain :
  {y : ℝ | ∃ x ≥ 0, f x = y} = Set.Iic 0 := by sorry

end NUMINAMATH_CALUDE_inverse_function_domain_l2112_211262


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2112_211265

-- Problem 1
theorem problem_1 : (π - 1)^0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + |(-3)| = 4 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a ≠ 1) : (1 - 1/a) / ((a^2 - 2*a + 1) / a) = 1 / (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2112_211265


namespace NUMINAMATH_CALUDE_cones_problem_l2112_211201

-- Define the radii of the three cones
def r1 (r : ℝ) : ℝ := 2 * r
def r2 (r : ℝ) : ℝ := 3 * r
def r3 (r : ℝ) : ℝ := 10 * r

-- Define the radius of the smaller base of the truncated cone
def R : ℝ := 15

-- Define the distances between the centers of the bases of cones
def d12 (r : ℝ) : ℝ := 5 * r
def d13 (r : ℝ) : ℝ := 12 * r
def d23 (r : ℝ) : ℝ := 13 * r

-- Define the distances from the center of the truncated cone to the centers of the other cones
def dC1 (r : ℝ) : ℝ := r1 r + R
def dC2 (r : ℝ) : ℝ := r2 r + R
def dC3 (r : ℝ) : ℝ := r3 r + R

-- Theorem statement
theorem cones_problem (r : ℝ) (h_pos : r > 0) :
  225 * (r1 r + R)^2 = (30 * r - 10 * R)^2 + (30 * r - 3 * R)^2 → r = 29 := by
  sorry

end NUMINAMATH_CALUDE_cones_problem_l2112_211201


namespace NUMINAMATH_CALUDE_balloon_distribution_difference_l2112_211226

/-- Represents the number of balloons of each color brought by a person -/
structure Balloons :=
  (red : ℕ)
  (blue : ℕ)
  (green : ℕ)

/-- Calculates the total number of balloons -/
def totalBalloons (b : Balloons) : ℕ := b.red + b.blue + b.green

theorem balloon_distribution_difference :
  let allan_brought := Balloons.mk 150 75 30
  let jake_brought := Balloons.mk 100 50 45
  let allan_forgot := 25
  let allan_distributed := totalBalloons { red := allan_brought.red,
                                           blue := allan_brought.blue - allan_forgot,
                                           green := allan_brought.green }
  let jake_distributed := totalBalloons jake_brought
  allan_distributed - jake_distributed = 35 := by sorry

end NUMINAMATH_CALUDE_balloon_distribution_difference_l2112_211226


namespace NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l2112_211222

theorem smallest_absolute_value_of_z (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z + 6*I) = 22) :
  ∃ (w : ℂ), Complex.abs (z - 15) + Complex.abs (z + 6*I) = 22 ∧ 
             Complex.abs w ≤ Complex.abs z ∧
             Complex.abs w = 45/11 :=
by sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_of_z_l2112_211222


namespace NUMINAMATH_CALUDE_associates_hired_l2112_211208

theorem associates_hired (initial_partners initial_associates : ℕ) 
  (new_associates : ℕ) (hired_associates : ℕ) : 
  initial_partners = 18 →
  initial_partners * 63 = 2 * initial_associates →
  (initial_partners) * 34 = (initial_associates + hired_associates) →
  hired_associates = 45 := by sorry

end NUMINAMATH_CALUDE_associates_hired_l2112_211208


namespace NUMINAMATH_CALUDE_pencil_distribution_problem_l2112_211219

/-- The number of ways to distribute pencils among friends -/
def distribute_pencils (total_pencils : ℕ) (num_friends : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of distributions for the given problem -/
theorem pencil_distribution_problem :
  distribute_pencils 10 4 = 58 :=
sorry

end NUMINAMATH_CALUDE_pencil_distribution_problem_l2112_211219


namespace NUMINAMATH_CALUDE_quadratic_problem_l2112_211218

/-- A quadratic function f(x) = ax^2 + bx + c -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_problem (a b c : ℝ) (f : ℝ → ℝ) (h_f : f = QuadraticFunction a b c) :
  (∀ x, f x ≤ 4) ∧ -- The maximum value of f(x) is 4
  (f 2 = 4) ∧ -- The maximum occurs at x = 2
  (f 0 = -20) ∧ -- The graph passes through (0, -20)
  (∃ m, f 5 = m) -- The graph passes through (5, m)
  → f 5 = -50 := by sorry

end NUMINAMATH_CALUDE_quadratic_problem_l2112_211218


namespace NUMINAMATH_CALUDE_number_divided_by_004_l2112_211236

theorem number_divided_by_004 :
  ∃ x : ℝ, x / 0.04 = 100.9 ∧ x = 4.036 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_004_l2112_211236


namespace NUMINAMATH_CALUDE_dave_apps_left_l2112_211206

/-- The number of files Dave has left on his phone -/
def files_left : ℕ := 5

/-- The difference between the number of apps and files Dave has left -/
def app_file_difference : ℕ := 7

/-- The number of apps Dave has left on his phone -/
def apps_left : ℕ := files_left + app_file_difference

theorem dave_apps_left : apps_left = 12 := by
  sorry

end NUMINAMATH_CALUDE_dave_apps_left_l2112_211206


namespace NUMINAMATH_CALUDE_wheel_radius_increase_l2112_211263

/-- Calculates the increase in wheel radius given the original and new odometer readings,
    and the original wheel radius. --/
theorem wheel_radius_increase (original_reading : ℝ) (new_reading : ℝ) (original_radius : ℝ) 
  (h1 : original_reading = 390)
  (h2 : new_reading = 380)
  (h3 : original_radius = 12)
  (h4 : original_reading > new_reading) :
  ∃ (increase : ℝ), 
    0.265 < increase ∧ increase < 0.275 ∧ 
    (2 * Real.pi * (original_radius + increase) * new_reading = 
     2 * Real.pi * original_radius * original_reading) :=
by sorry

end NUMINAMATH_CALUDE_wheel_radius_increase_l2112_211263


namespace NUMINAMATH_CALUDE_system_solution_l2112_211251

theorem system_solution (x y m : ℝ) 
  (eq1 : 2*x + y = 1) 
  (eq2 : x + 2*y = 2) 
  (eq3 : x + y = 2*m - 1) : 
  m = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_l2112_211251


namespace NUMINAMATH_CALUDE_class_average_theorem_l2112_211253

theorem class_average_theorem (group1_percent : Real) (group1_avg : Real)
                              (group2_percent : Real) (group2_avg : Real)
                              (group3_percent : Real) (group3_avg : Real) :
  group1_percent = 0.45 →
  group1_avg = 0.95 →
  group2_percent = 0.50 →
  group2_avg = 0.78 →
  group3_percent = 1 - group1_percent - group2_percent →
  group3_avg = 0.60 →
  round ((group1_percent * group1_avg + group2_percent * group2_avg + group3_percent * group3_avg) * 100) = 85 :=
by
  sorry

#check class_average_theorem

end NUMINAMATH_CALUDE_class_average_theorem_l2112_211253


namespace NUMINAMATH_CALUDE_dice_probability_l2112_211299

/-- The number of dice --/
def n : ℕ := 8

/-- The number of sides on each die --/
def sides : ℕ := 8

/-- The number of favorable outcomes (dice showing a number less than 5) --/
def k : ℕ := 4

/-- The probability of a single die showing a number less than 5 --/
def p : ℚ := 1/2

/-- The probability of exactly k out of n dice showing a number less than 5 --/
def probability : ℚ := (n.choose k) * p^k * (1-p)^(n-k)

theorem dice_probability : probability = 35/128 := by sorry

end NUMINAMATH_CALUDE_dice_probability_l2112_211299


namespace NUMINAMATH_CALUDE_owls_on_fence_l2112_211228

/-- The number of owls that joined the fence -/
def owls_joined (initial : ℕ) (final : ℕ) : ℕ := final - initial

theorem owls_on_fence (initial : ℕ) (final : ℕ) 
  (h_initial : initial = 3) 
  (h_final : final = 5) : 
  owls_joined initial final = 2 := by
  sorry

end NUMINAMATH_CALUDE_owls_on_fence_l2112_211228


namespace NUMINAMATH_CALUDE_crayons_per_friend_l2112_211277

theorem crayons_per_friend (total_crayons : ℕ) (num_friends : ℕ) (crayons_per_friend : ℕ) : 
  total_crayons = 210 → num_friends = 30 → crayons_per_friend = total_crayons / num_friends →
  crayons_per_friend = 7 := by
sorry

end NUMINAMATH_CALUDE_crayons_per_friend_l2112_211277


namespace NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l2112_211297

theorem lattice_points_on_hyperbola : 
  ∃! (points : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ points ↔ x^2 - y^2 = 65) ∧ 
    points.card = 8 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l2112_211297


namespace NUMINAMATH_CALUDE_factorial_ratio_simplification_l2112_211203

theorem factorial_ratio_simplification : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_simplification_l2112_211203


namespace NUMINAMATH_CALUDE_total_apples_in_basket_l2112_211287

theorem total_apples_in_basket (red_apples green_apples : ℕ) 
  (h1 : red_apples = 7) 
  (h2 : green_apples = 2) : 
  red_apples + green_apples = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_in_basket_l2112_211287


namespace NUMINAMATH_CALUDE_tim_books_l2112_211254

theorem tim_books (sam_books : ℕ) (total_books : ℕ) (h1 : sam_books = 52) (h2 : total_books = 96) :
  total_books - sam_books = 44 := by
  sorry

end NUMINAMATH_CALUDE_tim_books_l2112_211254


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2112_211232

/-- The eccentricity of an ellipse with given conditions is between 0 and √2/2 -/
theorem ellipse_eccentricity_range (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}
  let B := (0, b)
  let e := Real.sqrt (a^2 - b^2) / a
  (∀ p ∈ C, Real.sqrt ((p.1 - B.1)^2 + (p.2 - B.2)^2) ≤ 2*b) →
  0 < e ∧ e ≤ Real.sqrt 2 / 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_range_l2112_211232


namespace NUMINAMATH_CALUDE_cos_n_eq_sin_312_l2112_211210

theorem cos_n_eq_sin_312 :
  ∃ (n : ℤ), -90 ≤ n ∧ n ≤ 90 ∧ (Real.cos (n * π / 180) = Real.sin (312 * π / 180)) ∧ n = 42 := by
  sorry

end NUMINAMATH_CALUDE_cos_n_eq_sin_312_l2112_211210


namespace NUMINAMATH_CALUDE_pqr_product_l2112_211270

theorem pqr_product (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (h1 : p ∣ (q * r - 1)) (h2 : q ∣ (r * p - 1)) (h3 : r ∣ (p * q - 1)) :
  p * q * r = 30 := by
  sorry

end NUMINAMATH_CALUDE_pqr_product_l2112_211270


namespace NUMINAMATH_CALUDE_investment_problem_l2112_211296

/-- Represents the investment scenario described in the problem -/
structure Investment where
  total : ℝ
  interest : ℝ
  known_rate : ℝ
  unknown_amount : ℝ

/-- The theorem statement representing the problem -/
theorem investment_problem (inv : Investment) 
  (h1 : inv.total = 15000)
  (h2 : inv.interest = 1023)
  (h3 : inv.known_rate = 0.075)
  (h4 : inv.unknown_amount = 8200)
  (h5 : inv.unknown_amount + (inv.total - inv.unknown_amount) * inv.known_rate = inv.interest) :
  inv.unknown_amount = 8200 := by
  sorry

#check investment_problem

end NUMINAMATH_CALUDE_investment_problem_l2112_211296


namespace NUMINAMATH_CALUDE_ceiling_sqrt_244_l2112_211215

theorem ceiling_sqrt_244 : ⌈Real.sqrt 244⌉ = 16 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_244_l2112_211215


namespace NUMINAMATH_CALUDE_remainder_17_pow_2046_mod_23_l2112_211242

theorem remainder_17_pow_2046_mod_23 : 17^2046 % 23 = 22 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_pow_2046_mod_23_l2112_211242


namespace NUMINAMATH_CALUDE_child_share_proof_l2112_211239

theorem child_share_proof (total_money : ℕ) (ratio : List ℕ) : 
  total_money = 4500 →
  ratio = [2, 4, 5, 4] →
  (ratio[0]! + ratio[1]!) * total_money / ratio.sum = 1800 := by
  sorry

end NUMINAMATH_CALUDE_child_share_proof_l2112_211239


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l2112_211246

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 20 →
  a * b + c + d = 105 →
  a * d + b * c = 225 →
  c * d = 144 →
  a^2 + b^2 + c^2 + d^2 ≤ 150 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l2112_211246


namespace NUMINAMATH_CALUDE_dryer_weight_l2112_211261

def bridge_weight_limit : ℕ := 20000
def empty_truck_weight : ℕ := 12000
def soda_crates : ℕ := 20
def soda_crate_weight : ℕ := 50
def dryer_count : ℕ := 3
def loaded_truck_weight : ℕ := 24000

theorem dryer_weight (h1 : bridge_weight_limit = 20000)
                     (h2 : empty_truck_weight = 12000)
                     (h3 : soda_crates = 20)
                     (h4 : soda_crate_weight = 50)
                     (h5 : dryer_count = 3)
                     (h6 : loaded_truck_weight = 24000) :
  let soda_weight := soda_crates * soda_crate_weight
  let produce_weight := 2 * soda_weight
  let truck_soda_produce_weight := empty_truck_weight + soda_weight + produce_weight
  let total_dryer_weight := loaded_truck_weight - truck_soda_produce_weight
  total_dryer_weight / dryer_count = 3000 := by
sorry

end NUMINAMATH_CALUDE_dryer_weight_l2112_211261


namespace NUMINAMATH_CALUDE_product_def_l2112_211223

theorem product_def (a b c d e f : ℝ) : 
  a * b * c = 130 →
  b * c * d = 65 →
  c * d * e = 500 →
  (a * f) / (c * d) = 1 →
  d * e * f = 250 := by
sorry

end NUMINAMATH_CALUDE_product_def_l2112_211223


namespace NUMINAMATH_CALUDE_johns_pre_raise_earnings_l2112_211238

/-- The amount John makes per week after the raise, in dollars. -/
def post_raise_earnings : ℝ := 60

/-- The percentage increase of John's earnings. -/
def percentage_increase : ℝ := 50

/-- John's weekly earnings before the raise, in dollars. -/
def pre_raise_earnings : ℝ := 40

/-- Theorem stating that John's pre-raise earnings were $40, given the conditions. -/
theorem johns_pre_raise_earnings : 
  pre_raise_earnings * (1 + percentage_increase / 100) = post_raise_earnings := by
  sorry

end NUMINAMATH_CALUDE_johns_pre_raise_earnings_l2112_211238


namespace NUMINAMATH_CALUDE_isle_of_misfortune_l2112_211211

/-- Represents a person who is either a knight (truth-teller) or a liar -/
inductive Person
| Knight
| Liar

/-- The total number of people in the group -/
def total_people : Nat := 101

/-- A function that returns true if removing a person results in a majority of liars -/
def majority_liars_if_removed (knights : Nat) (liars : Nat) (person : Person) : Prop :=
  match person with
  | Person.Knight => liars ≥ knights - 1
  | Person.Liar => liars - 1 ≥ knights

theorem isle_of_misfortune :
  ∀ (knights liars : Nat),
    knights + liars = total_people →
    (∀ (p : Person), majority_liars_if_removed knights liars p) →
    knights = 50 ∧ liars = 51 := by
  sorry

end NUMINAMATH_CALUDE_isle_of_misfortune_l2112_211211


namespace NUMINAMATH_CALUDE_B_and_C_complementary_l2112_211289

-- Define the sample space (outcomes of rolling a fair die)
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define event B (up-facing side's number is no more than 3)
def B : Finset Nat := {1, 2, 3}

-- Define event C (up-facing side's number is at least 4)
def C : Finset Nat := {4, 5, 6}

-- Theorem stating that B and C are complementary
theorem B_and_C_complementary : B ∪ C = Ω ∧ B ∩ C = ∅ := by
  sorry


end NUMINAMATH_CALUDE_B_and_C_complementary_l2112_211289
