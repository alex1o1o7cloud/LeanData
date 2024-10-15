import Mathlib

namespace NUMINAMATH_CALUDE_baseball_gear_sale_l3813_381319

theorem baseball_gear_sale (bat_price glove_original_price glove_discount cleats_price total_amount : ℝ)
  (h1 : bat_price = 10)
  (h2 : glove_original_price = 30)
  (h3 : glove_discount = 0.2)
  (h4 : cleats_price = 10)
  (h5 : total_amount = 79) :
  let glove_sale_price := glove_original_price * (1 - glove_discount)
  let other_gear_total := bat_price + glove_sale_price + 2 * cleats_price
  total_amount - other_gear_total = 25 := by
sorry

end NUMINAMATH_CALUDE_baseball_gear_sale_l3813_381319


namespace NUMINAMATH_CALUDE_first_player_wins_l3813_381300

/-- Represents a chessboard with knights on opposite corners -/
structure Chessboard :=
  (squares : Finset (ℕ × ℕ))
  (knight1 : ℕ × ℕ)
  (knight2 : ℕ × ℕ)

/-- Represents a move in the game -/
def Move := ℕ × ℕ

/-- Checks if a knight can reach another position on the board -/
def can_reach (board : Chessboard) (start finish : ℕ × ℕ) : Prop :=
  sorry

/-- Represents the game state -/
structure GameState :=
  (board : Chessboard)
  (current_player : ℕ)

/-- Applies a move to the game state -/
def apply_move (state : GameState) (move : Move) : GameState :=
  sorry

/-- Checks if a player has a winning strategy -/
def has_winning_strategy (player : ℕ) (state : GameState) : Prop :=
  sorry

/-- The main theorem stating that the first player has a winning strategy -/
theorem first_player_wins (initial_board : Chessboard) :
  has_winning_strategy 1 { board := initial_board, current_player := 1 } :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l3813_381300


namespace NUMINAMATH_CALUDE_simplify_fraction_l3813_381384

theorem simplify_fraction : 18 * (8 / 12) * (1 / 27) = 4 / 9 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3813_381384


namespace NUMINAMATH_CALUDE_subtracted_value_l3813_381352

theorem subtracted_value (n : ℝ) (v : ℝ) (h1 : n = 1) (h2 : 3 * n - v = 2 * n) : v = 1 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l3813_381352


namespace NUMINAMATH_CALUDE_product_minus_sum_probability_l3813_381327

def valid_pair (a b : ℕ) : Prop :=
  a ≤ 10 ∧ b ≤ 10 ∧ a * b - (a + b) > 4

def total_pairs : ℕ := 100

def valid_pairs : ℕ := 44

theorem product_minus_sum_probability :
  (valid_pairs : ℚ) / total_pairs = 11 / 25 := by sorry

end NUMINAMATH_CALUDE_product_minus_sum_probability_l3813_381327


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_system_l3813_381374

theorem unique_solution_logarithmic_system :
  ∃! (x y : ℝ), 
    Real.log (x^2 + y^2) / Real.log 10 = 1 + Real.log 8 / Real.log 10 ∧
    Real.log (x + y) / Real.log 10 - Real.log (x - y) / Real.log 10 = Real.log 3 / Real.log 10 ∧
    x + y > 0 ∧
    x - y > 0 ∧
    x = 8 ∧
    y = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_system_l3813_381374


namespace NUMINAMATH_CALUDE_vanya_climb_ratio_l3813_381347

-- Define the floors for Anya and Vanya
def anya_floor : ℕ := 2
def vanya_floor : ℕ := 6
def start_floor : ℕ := 1

-- Define the climbs for Anya and Vanya
def anya_climb : ℕ := anya_floor - start_floor
def vanya_climb : ℕ := vanya_floor - start_floor

-- Theorem statement
theorem vanya_climb_ratio :
  (vanya_climb : ℚ) / (anya_climb : ℚ) = 5 := by sorry

end NUMINAMATH_CALUDE_vanya_climb_ratio_l3813_381347


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3813_381322

theorem fraction_to_decimal (h : 243 = 3^5) : 7 / 243 = 0.00224 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3813_381322


namespace NUMINAMATH_CALUDE_stating_regular_ngon_diagonal_difference_l3813_381335

/-- 
Given a regular n-gon with n > 5, this function calculates the length of its longest diagonal.
-/
noncomputable def longest_diagonal (n : ℕ) (side_length : ℝ) : ℝ := sorry

/-- 
Given a regular n-gon with n > 5, this function calculates the length of its shortest diagonal.
-/
noncomputable def shortest_diagonal (n : ℕ) (side_length : ℝ) : ℝ := sorry

/-- 
Theorem stating that for a regular n-gon with n > 5, the difference between 
the longest diagonal and the shortest diagonal is equal to the side length 
if and only if n = 9.
-/
theorem regular_ngon_diagonal_difference (n : ℕ) (side_length : ℝ) : 
  n > 5 → 
  (longest_diagonal n side_length - shortest_diagonal n side_length = side_length ↔ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_stating_regular_ngon_diagonal_difference_l3813_381335


namespace NUMINAMATH_CALUDE_good_numbers_up_to_17_and_18_not_good_l3813_381333

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- A number m is "good" if there exists a positive integer n such that m = n / d(n) -/
def is_good (m : ℕ+) : Prop :=
  ∃ n : ℕ+, (n : ℚ) / d n = m

theorem good_numbers_up_to_17_and_18_not_good :
  (∀ m : ℕ+, m ≤ 17 → is_good m) ∧ ¬ is_good 18 := by sorry

end NUMINAMATH_CALUDE_good_numbers_up_to_17_and_18_not_good_l3813_381333


namespace NUMINAMATH_CALUDE_circle_point_x_coordinate_l3813_381399

theorem circle_point_x_coordinate 
  (x : ℝ) 
  (h1 : (x - 6)^2 + 10^2 = 12^2) : 
  x = 6 + 2 * Real.sqrt 11 ∨ x = 6 - 2 * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_circle_point_x_coordinate_l3813_381399


namespace NUMINAMATH_CALUDE_colored_square_theorem_l3813_381355

/-- Represents a coloring of a square grid -/
def Coloring (n : ℕ) := Fin (n^2 + 1) → Fin (n^2 + 1)

/-- Counts the number of distinct colors in a given row or column -/
def distinctColors (n : ℕ) (c : Coloring n) (isRow : Bool) (index : Fin (n^2 + 1)) : ℕ :=
  sorry

theorem colored_square_theorem (n : ℕ) (c : Coloring n) :
  ∃ (isRow : Bool) (index : Fin (n^2 + 1)), distinctColors n c isRow index ≥ n + 1 :=
sorry

end NUMINAMATH_CALUDE_colored_square_theorem_l3813_381355


namespace NUMINAMATH_CALUDE_equidistant_centers_l3813_381343

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the structure for a triangle
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

-- Define the structure for a circle
structure Circle where
  center : Point2D
  radius : ℝ

def is_right_triangle (t : Triangle) : Prop := sorry

def altitude_to_hypotenuse (t : Triangle) : Point2D := sorry

def inscribed_circle (t : Triangle) : Circle := sorry

def touch_point_on_hypotenuse (c : Circle) (t : Triangle) : Point2D := sorry

def distance (p1 p2 : Point2D) : ℝ := sorry

theorem equidistant_centers (ABC : Triangle) (H₃ : Point2D) :
  is_right_triangle ABC →
  H₃ = altitude_to_hypotenuse ABC →
  let O := (inscribed_circle ABC).center
  let O₁ := (inscribed_circle ⟨ABC.A, ABC.C, H₃⟩).center
  let O₂ := (inscribed_circle ⟨ABC.B, ABC.C, H₃⟩).center
  let T := touch_point_on_hypotenuse (inscribed_circle ABC) ABC
  distance O T = distance O₁ T ∧ distance O T = distance O₂ T :=
by sorry

end NUMINAMATH_CALUDE_equidistant_centers_l3813_381343


namespace NUMINAMATH_CALUDE_division_chain_l3813_381361

theorem division_chain : (88 / 4) / 2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_division_chain_l3813_381361


namespace NUMINAMATH_CALUDE_final_fruit_juice_percentage_l3813_381370

/-- Given an initial mixture of punch and some added pure fruit juice,
    calculate the final percentage of fruit juice in the punch. -/
theorem final_fruit_juice_percentage
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_juice : ℝ)
  (h1 : initial_volume = 2)
  (h2 : initial_percentage = 0.1)
  (h3 : added_juice = 0.4)
  : (initial_volume * initial_percentage + added_juice) / (initial_volume + added_juice) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_final_fruit_juice_percentage_l3813_381370


namespace NUMINAMATH_CALUDE_polynomial_roots_l3813_381314

theorem polynomial_roots : 
  let p : ℝ → ℝ := λ x => x^3 + 2*x^2 - 5*x - 6
  ∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 2 ∨ x = -3 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l3813_381314


namespace NUMINAMATH_CALUDE_cafeteria_cottage_pies_l3813_381357

theorem cafeteria_cottage_pies :
  ∀ (lasagna_count : ℕ) (lasagna_mince : ℕ) (cottage_pie_mince : ℕ) (total_mince : ℕ),
    lasagna_count = 100 →
    lasagna_mince = 2 →
    cottage_pie_mince = 3 →
    total_mince = 500 →
    ∃ (cottage_pie_count : ℕ),
      cottage_pie_count * cottage_pie_mince + lasagna_count * lasagna_mince = total_mince ∧
      cottage_pie_count = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_cottage_pies_l3813_381357


namespace NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l3813_381391

/-- A sequence of binary digits (0 or 1) -/
def BinarySequence := List Nat

/-- Count the number of (1,0) pairs with even number of digits between them -/
def countEvenPairs (seq : BinarySequence) : Nat :=
  sorry

/-- Count the number of (1,0) pairs with odd number of digits between them -/
def countOddPairs (seq : BinarySequence) : Nat :=
  sorry

/-- The main theorem: For any binary sequence, the number of (1,0) pairs
    with even number of digits between is greater than or equal to
    the number of (1,0) pairs with odd number of digits between -/
theorem even_pairs_ge_odd_pairs (seq : BinarySequence) :
  countEvenPairs seq ≥ countOddPairs seq :=
sorry

end NUMINAMATH_CALUDE_even_pairs_ge_odd_pairs_l3813_381391


namespace NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l3813_381371

theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) 
  (h_sphere : r_sphere = 7)
  (h_cylinder : r_cylinder = 4) :
  let h_cylinder := Real.sqrt (r_sphere^2 - r_cylinder^2)
  let v_sphere := (4/3) * π * r_sphere^3
  let v_cylinder := π * r_cylinder^2 * h_cylinder
  v_sphere - v_cylinder = ((1372/3) - 16 * Real.sqrt 132) * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_cylinder_volume_difference_l3813_381371


namespace NUMINAMATH_CALUDE_largest_common_divisor_l3813_381311

theorem largest_common_divisor :
  ∃ (n : ℕ), n = 30 ∧
  n ∣ 420 ∧
  n < 60 ∧
  n ∣ 90 ∧
  ∀ (m : ℕ), m ∣ 420 → m < 60 → m ∣ 90 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l3813_381311


namespace NUMINAMATH_CALUDE_max_profit_theorem_l3813_381377

/-- Represents the daily sales volume as a function of unit price -/
def sales_volume (x : ℝ) : ℝ := -100 * x + 5000

/-- Represents the daily profit as a function of unit price -/
def daily_profit (x : ℝ) : ℝ := (sales_volume x) * (x - 6)

/-- The theorem stating the maximum profit and the price at which it occurs -/
theorem max_profit_theorem :
  let x_min : ℝ := 6
  let x_max : ℝ := 32
  ∀ x ∈ Set.Icc x_min x_max,
    daily_profit x ≤ daily_profit 28 ∧
    daily_profit 28 = 48400 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_theorem_l3813_381377


namespace NUMINAMATH_CALUDE_distance_to_school_l3813_381326

theorem distance_to_school (walking_speed run_speed : ℝ) 
  (run_distance total_time : ℝ) : 
  walking_speed = 70 →
  run_speed = 210 →
  run_distance = 600 →
  total_time ≤ 20 →
  ∃ (walk_distance : ℝ),
    walk_distance ≥ 0 ∧
    run_distance / run_speed + walk_distance / walking_speed ≤ total_time ∧
    walk_distance + run_distance ≤ 1800 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_school_l3813_381326


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3813_381397

/-- Given a circle D with equation x^2 - 8x + y^2 + 14y = -28,
    prove that the sum of its center coordinates and radius is -3 + √37 -/
theorem circle_center_radius_sum :
  let D : Set (ℝ × ℝ) := {p | (p.1^2 - 8*p.1 + p.2^2 + 14*p.2 = -28)}
  ∃ (c d s : ℝ),
    (∀ (x y : ℝ), (x, y) ∈ D ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c + d + s = -3 + Real.sqrt 37 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3813_381397


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3813_381383

theorem sum_of_cubes_of_roots (P : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) :
  P = (fun x ↦ x^3 - 3*x - 1) →
  P x₁ = 0 →
  P x₂ = 0 →
  P x₃ = 0 →
  x₁^3 + x₂^3 + x₃^3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_roots_l3813_381383


namespace NUMINAMATH_CALUDE_class_mean_score_l3813_381325

theorem class_mean_score 
  (n : ℕ) 
  (h1 : n > 15) 
  (overall_mean : ℝ) 
  (h2 : overall_mean = 10) 
  (group_mean : ℝ) 
  (h3 : group_mean = 16) : 
  let remaining_mean := (n * overall_mean - 15 * group_mean) / (n - 15)
  remaining_mean = (10 * n - 240) / (n - 15) := by
sorry

end NUMINAMATH_CALUDE_class_mean_score_l3813_381325


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_problem_l3813_381337

theorem smallest_integer_gcd_lcm_problem (x : ℕ) (a b : ℕ) : 
  x > 0 →
  a > 0 →
  b > 0 →
  a = 72 →
  Nat.gcd a b = x + 6 →
  Nat.lcm a b = x * (x + 6) →
  ∃ m : ℕ, (∀ n : ℕ, n > 0 ∧ 
    Nat.gcd 72 n = x + 6 ∧ 
    Nat.lcm 72 n = x * (x + 6) → 
    m ≤ n) ∧ 
    m = 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_problem_l3813_381337


namespace NUMINAMATH_CALUDE_compute_expression_l3813_381309

theorem compute_expression : 7^2 + 4*5 - 2^3 = 61 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3813_381309


namespace NUMINAMATH_CALUDE_first_candidate_percentage_l3813_381315

theorem first_candidate_percentage (P : ℝ) (total_marks : ℝ) : 
  P = 199.99999999999997 →
  0.45 * total_marks = P + 25 →
  (P - 50) / total_marks * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_first_candidate_percentage_l3813_381315


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3813_381341

theorem pure_imaginary_complex_number (x : ℝ) : 
  let z : ℂ := (x^2 - 1) + (x - 1) * I
  (∃ (y : ℝ), z = y * I) → x = -1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l3813_381341


namespace NUMINAMATH_CALUDE_problem_solution_l3813_381368

def problem (A B X : ℕ) : Prop :=
  A > 0 ∧ B > 0 ∧
  Nat.gcd A B = 20 ∧
  A = 300 ∧
  Nat.lcm A B = 20 * X * 15

theorem problem_solution :
  ∀ A B X, problem A B X → X = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3813_381368


namespace NUMINAMATH_CALUDE_f_extrema_l3813_381354

noncomputable def f (x : ℝ) : ℝ := -x^3 + 3*x + 1

theorem f_extrema : 
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ f x) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ f x) := by
  sorry

end NUMINAMATH_CALUDE_f_extrema_l3813_381354


namespace NUMINAMATH_CALUDE_inequality_proof_l3813_381367

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) :
  1/a^2 + 1/b^2 + 1/c^2 ≥ 2*(a^3 + b^3 + c^3)/(a*b*c) + 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3813_381367


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3813_381344

theorem polynomial_factorization (y : ℝ) : 
  y^8 - 4*y^6 + 6*y^4 - 4*y^2 + 1 = (y-1)^4 * (y+1)^4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3813_381344


namespace NUMINAMATH_CALUDE_point_c_values_l3813_381305

/-- Represents a point on a number line --/
structure Point where
  value : ℝ

/-- The distance between two points on a number line --/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_c_values (b c : Point) : 
  b.value = 3 → distance b c = 2 → (c.value = 1 ∨ c.value = 5) := by
  sorry

end NUMINAMATH_CALUDE_point_c_values_l3813_381305


namespace NUMINAMATH_CALUDE_sarah_game_multiple_l3813_381366

/-- The game's formula to predict marriage age -/
def marriage_age_formula (name_length : ℕ) (current_age : ℕ) (multiple : ℕ) : ℕ :=
  name_length + multiple * current_age

/-- Proof that the multiple in Sarah's game is 2 -/
theorem sarah_game_multiple : ∃ (multiple : ℕ), 
  marriage_age_formula 5 9 multiple = 23 ∧ multiple = 2 :=
by sorry

end NUMINAMATH_CALUDE_sarah_game_multiple_l3813_381366


namespace NUMINAMATH_CALUDE_marks_books_count_l3813_381308

/-- Given that Mark started with $85, each book costs $5, and he is left with $35, 
    prove that the number of books he bought is 10. -/
theorem marks_books_count (initial_amount : ℕ) (book_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  sorry

end NUMINAMATH_CALUDE_marks_books_count_l3813_381308


namespace NUMINAMATH_CALUDE_parallel_vectors_m_equals_one_l3813_381345

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_equals_one :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (m, m - 4)
  are_parallel a b → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_equals_one_l3813_381345


namespace NUMINAMATH_CALUDE_fraction_inequality_l3813_381372

def numerator (x : ℝ) : ℝ := 7 * x - 3

def denominator (x : ℝ) : ℝ := x^2 - x - 12

def valid_x (x : ℝ) : Prop := denominator x ≠ 0

def inequality_holds (x : ℝ) : Prop := numerator x ≥ denominator x

def solution_set : Set ℝ := {x | x ∈ Set.Icc (-1) 3 ∪ Set.Ioo 3 4 ∪ Set.Ico 4 9}

theorem fraction_inequality :
  {x : ℝ | inequality_holds x ∧ valid_x x} = solution_set := by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3813_381372


namespace NUMINAMATH_CALUDE_parabola_translation_l3813_381317

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 - 1

-- Define the translation
def left_translation : ℝ := 2
def up_translation : ℝ := 1

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x + left_translation)^2

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, y = original_parabola (x + left_translation) + up_translation 
  ↔ y = translated_parabola x := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3813_381317


namespace NUMINAMATH_CALUDE_percentage_of_absent_students_l3813_381332

theorem percentage_of_absent_students (total : ℕ) (present : ℕ) : 
  total = 50 → present = 44 → (total - present) * 100 / total = 12 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_absent_students_l3813_381332


namespace NUMINAMATH_CALUDE_abc_product_l3813_381349

theorem abc_product (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 29) (h5 : (1 : ℚ) / a + 1 / b + 1 / c + 399 / (a * b * c) = 1) :
  a * b * c = 992 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l3813_381349


namespace NUMINAMATH_CALUDE_discount_rate_sum_l3813_381318

-- Define the normal prices and quantities
def biography_price : ℝ := 20
def mystery_price : ℝ := 12
def biography_quantity : ℕ := 5
def mystery_quantity : ℕ := 3

-- Define the total savings and mystery discount rate
def total_savings : ℝ := 19
def mystery_discount_rate : ℝ := 0.375

-- Define the function to calculate the total discount rate
def total_discount_rate (biography_discount_rate : ℝ) : ℝ :=
  biography_discount_rate + mystery_discount_rate

-- Theorem statement
theorem discount_rate_sum :
  ∃ (biography_discount_rate : ℝ),
    biography_discount_rate > 0 ∧
    biography_discount_rate < 1 ∧
    (biography_price * biography_quantity * (1 - biography_discount_rate) +
     mystery_price * mystery_quantity * (1 - mystery_discount_rate) =
     biography_price * biography_quantity + mystery_price * mystery_quantity - total_savings) ∧
    total_discount_rate biography_discount_rate = 0.43 :=
by sorry

end NUMINAMATH_CALUDE_discount_rate_sum_l3813_381318


namespace NUMINAMATH_CALUDE_min_operations_to_2187_l3813_381320

/-- Represents the possible operations on the calculator --/
inductive Operation
  | AddOne
  | TimesThree

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.TimesThree => n * 3

/-- Checks if a sequence of operations transforms 1 into the target --/
def isValidSequence (ops : List Operation) (target : ℕ) : Prop :=
  ops.foldl applyOperation 1 = target

/-- The main theorem to prove --/
theorem min_operations_to_2187 :
  ∃ (ops : List Operation), isValidSequence ops 2187 ∧ 
    ops.length = 7 ∧ 
    (∀ (other_ops : List Operation), isValidSequence other_ops 2187 → other_ops.length ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_min_operations_to_2187_l3813_381320


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l3813_381350

theorem arithmetic_mean_problem (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 27)
  (h3 : r - p = 34) : 
  (p + q) / 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l3813_381350


namespace NUMINAMATH_CALUDE_shortest_side_l3813_381342

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem shortest_side (t : Triangle) (h : t.a^2 + t.b^2 > 5 * t.c^2) : 
  t.c < t.a ∧ t.c < t.b := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_l3813_381342


namespace NUMINAMATH_CALUDE_winner_depends_on_n_l3813_381310

/-- Represents a player in the game -/
inductive Player
| Bela
| Jenn

/-- Represents the game state -/
structure GameState where
  n : ℕ
  choices : List ℝ

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : ℝ) : Prop :=
  0 ≤ move ∧ move ≤ state.n ∧ ∀ c ∈ state.choices, |move - c| > 1.5

/-- Determines if the game is over -/
def is_game_over (state : GameState) : Prop :=
  ∀ move, ¬(is_valid_move state move)

/-- Determines the winner of the game -/
def winner (state : GameState) : Player :=
  if state.choices.length % 2 = 0 then Player.Jenn else Player.Bela

/-- The main theorem stating that the winner depends on the specific value of n -/
theorem winner_depends_on_n :
  ∃ n m : ℕ,
    n > 5 ∧ m > 5 ∧
    (∃ state1 : GameState, state1.n = n ∧ is_game_over state1 ∧ winner state1 = Player.Bela) ∧
    (∃ state2 : GameState, state2.n = m ∧ is_game_over state2 ∧ winner state2 = Player.Jenn) :=
  sorry


end NUMINAMATH_CALUDE_winner_depends_on_n_l3813_381310


namespace NUMINAMATH_CALUDE_right_angles_in_five_days_l3813_381390

/-- The number of times clock hands form a right angle in a 12-hour period -/
def right_angles_per_12hours : ℕ := 22

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days we're considering -/
def days : ℕ := 5

/-- Theorem: The number of times clock hands form a right angle in 5 days is 220 -/
theorem right_angles_in_five_days :
  (right_angles_per_12hours * 2 * days) = 220 := by sorry

end NUMINAMATH_CALUDE_right_angles_in_five_days_l3813_381390


namespace NUMINAMATH_CALUDE_twenty_factorial_digits_sum_l3813_381340

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem twenty_factorial_digits_sum (B A : ℕ) : 
  B < 10 → A < 10 → 
  ∃ k : ℕ, factorial 20 = k * 10000 + B * 100 + A * 10 → 
  B + A = 10 := by
  sorry

end NUMINAMATH_CALUDE_twenty_factorial_digits_sum_l3813_381340


namespace NUMINAMATH_CALUDE_slope_angle_of_line_l3813_381365

theorem slope_angle_of_line (x y : ℝ) :
  let line_eq := x * Real.tan (π / 6) + y - 7 = 0
  let slope := -Real.tan (π / 6)
  let slope_angle := Real.arctan (-slope)
  slope_angle = 5 * π / 6 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l3813_381365


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_negative_l3813_381346

/-- If the equation 2x^2 + (m+1)x + m = 0 has one positive root and one negative root, then m < 0 -/
theorem quadratic_roots_imply_m_negative (m : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ 2 * x^2 + (m + 1) * x + m = 0 ∧ 2 * y^2 + (m + 1) * y + m = 0) →
  m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_negative_l3813_381346


namespace NUMINAMATH_CALUDE_quadratic_trinomial_square_l3813_381321

theorem quadratic_trinomial_square (a b c : ℝ) :
  (∀ x : ℤ, ∃ y : ℤ, a * x^2 + b * x + c = y^2) →
  (∃ m n k : ℤ, 2 * a = m ∧ 2 * b = n ∧ c = k^2) ∧
  (∃ p q r : ℤ, a = p ∧ b = q ∧ c = r^2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_square_l3813_381321


namespace NUMINAMATH_CALUDE_assignments_count_l3813_381303

/-- The number of interest groups available --/
def num_groups : ℕ := 3

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of ways to assign students to interest groups --/
def num_assignments : ℕ := num_groups ^ num_students

/-- Theorem stating that the number of assignments is 81 --/
theorem assignments_count : num_assignments = 81 := by
  sorry

end NUMINAMATH_CALUDE_assignments_count_l3813_381303


namespace NUMINAMATH_CALUDE_first_number_solution_l3813_381312

theorem first_number_solution (y : ℝ) (h : y = -4.5) :
  ∃ x : ℝ, x * y = 2 * x - 36 → x = 36 / 6.5 := by
  sorry

end NUMINAMATH_CALUDE_first_number_solution_l3813_381312


namespace NUMINAMATH_CALUDE_chickpea_flour_amount_l3813_381360

def rye_flour : ℕ := 5
def whole_wheat_bread_flour : ℕ := 10
def whole_wheat_pastry_flour : ℕ := 2
def total_flour : ℕ := 20

theorem chickpea_flour_amount :
  total_flour - (rye_flour + whole_wheat_bread_flour + whole_wheat_pastry_flour) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chickpea_flour_amount_l3813_381360


namespace NUMINAMATH_CALUDE_fraction_ordering_l3813_381356

def t₁ : ℚ := (100^100 + 1) / (100^90 + 1)
def t₂ : ℚ := (100^99 + 1) / (100^89 + 1)
def t₃ : ℚ := (100^101 + 1) / (100^91 + 1)
def t₄ : ℚ := (101^101 + 1) / (101^91 + 1)
def t₅ : ℚ := (101^100 + 1) / (101^90 + 1)
def t₆ : ℚ := (99^99 + 1) / (99^89 + 1)
def t₇ : ℚ := (99^100 + 1) / (99^90 + 1)

theorem fraction_ordering :
  t₆ < t₇ ∧ t₇ < t₂ ∧ t₂ < t₁ ∧ t₁ < t₃ ∧ t₃ < t₅ ∧ t₅ < t₄ :=
by sorry

end NUMINAMATH_CALUDE_fraction_ordering_l3813_381356


namespace NUMINAMATH_CALUDE_minimum_k_value_l3813_381323

theorem minimum_k_value (m n : ℕ+) :
  (1 : ℝ) / (m + n : ℝ)^2 ≤ (1/8) * ((1 : ℝ) / m^2 + 1 / n^2) ∧
  ∀ k : ℝ, (∀ a b : ℕ+, (1 : ℝ) / (a + b : ℝ)^2 ≤ k * ((1 : ℝ) / a^2 + 1 / b^2)) →
    k ≥ 1/8 :=
by sorry

end NUMINAMATH_CALUDE_minimum_k_value_l3813_381323


namespace NUMINAMATH_CALUDE_circle_sequence_circumference_sum_l3813_381362

theorem circle_sequence_circumference_sum (r₁ r₂ r₃ r₄ : ℝ) : 
  r₁ = 1 →                           -- radius of first circle is 1
  r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ < r₄ →       -- radii are increasing
  r₂ / r₁ = r₃ / r₂ ∧ r₃ / r₂ = r₄ / r₃ →  -- circles form a geometric progression
  r₄^2 * Real.pi = 64 * Real.pi →    -- area of fourth circle is 64π
  2 * Real.pi * r₂ + 2 * Real.pi * r₃ = 12 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_circle_sequence_circumference_sum_l3813_381362


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l3813_381376

/-- Given two circles C₁ and C₂, prove that their common chord lies on the line 3x - 4y + 6 = 0 -/
theorem common_chord_of_circles (x y : ℝ) :
  (x^2 + y^2 + 2*x - 6*y + 1 = 0) ∧ (x^2 + y^2 - 4*x + 2*y - 11 = 0) →
  (3*x - 4*y + 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l3813_381376


namespace NUMINAMATH_CALUDE_field_path_area_and_cost_l3813_381351

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

/-- Theorem: For a 60m x 55m field with a 2.5m wide path, the path area is 600 sq m
    and the construction cost at Rs. 2 per sq m is Rs. 1200 -/
theorem field_path_area_and_cost :
  let field_length : ℝ := 60
  let field_width : ℝ := 55
  let path_width : ℝ := 2.5
  let cost_per_unit : ℝ := 2
  (path_area field_length field_width path_width = 600) ∧
  (construction_cost (path_area field_length field_width path_width) cost_per_unit = 1200) :=
by sorry

end NUMINAMATH_CALUDE_field_path_area_and_cost_l3813_381351


namespace NUMINAMATH_CALUDE_second_part_speed_l3813_381392

/-- Proves that given a total distance of 20 miles, where the first 10 miles are traveled at 12 miles per hour,
    and the average speed for the entire trip is 10.909090909090908 miles per hour,
    the speed for the second part of the trip is 10 miles per hour. -/
theorem second_part_speed
  (total_distance : ℝ)
  (first_part_distance : ℝ)
  (first_part_speed : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 20)
  (h2 : first_part_distance = 10)
  (h3 : first_part_speed = 12)
  (h4 : average_speed = 10.909090909090908)
  : ∃ (second_part_speed : ℝ),
    second_part_speed = 10 ∧
    average_speed = (first_part_distance / first_part_speed + (total_distance - first_part_distance) / second_part_speed) / (total_distance / average_speed) :=
by
  sorry

end NUMINAMATH_CALUDE_second_part_speed_l3813_381392


namespace NUMINAMATH_CALUDE_bike_distance_proof_l3813_381396

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a bike traveling at 8 m/s for 6 seconds covers 48 meters -/
theorem bike_distance_proof :
  let speed : ℝ := 8
  let time : ℝ := 6
  distance speed time = 48 := by
  sorry

end NUMINAMATH_CALUDE_bike_distance_proof_l3813_381396


namespace NUMINAMATH_CALUDE_expression_simplification_l3813_381386

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) :
  (a^(7/3) - 2*a^(5/3)*b^(2/3) + a*b^(4/3)) / (a^(5/3) - a^(4/3)*b^(1/3) - a*b^(2/3) + a^(2/3)*b) / a^(1/3) = a^(1/3) + b^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3813_381386


namespace NUMINAMATH_CALUDE_mice_elimination_time_l3813_381381

/-- Represents the rate at which cats hunt mice -/
def hunting_rate : ℝ := 0.1

/-- Represents the total amount of work to eliminate all mice -/
def total_work : ℝ := 1

/-- Represents the number of days taken by initial cats -/
def initial_days : ℕ := 5

/-- Represents the initial number of cats -/
def initial_cats : ℕ := 2

/-- Represents the final number of cats -/
def final_cats : ℕ := 5

theorem mice_elimination_time :
  let initial_work := hunting_rate * initial_cats * initial_days
  let remaining_work := total_work - initial_work
  let final_rate := hunting_rate * final_cats
  initial_days + (remaining_work / final_rate) = 7 := by sorry

end NUMINAMATH_CALUDE_mice_elimination_time_l3813_381381


namespace NUMINAMATH_CALUDE_election_result_l3813_381375

/-- Represents the result of an election with three candidates. -/
structure ElectionResult where
  totalVotes : ℕ
  votesA : ℕ
  votesB : ℕ
  votesC : ℕ

/-- Theorem stating the correct election results given the conditions. -/
theorem election_result : ∃ (result : ElectionResult),
  result.totalVotes = 10000 ∧
  result.votesA = 3400 ∧
  result.votesB = 4800 ∧
  result.votesC = 2900 ∧
  result.votesA = (34 * result.totalVotes) / 100 ∧
  result.votesB = (48 * result.totalVotes) / 100 ∧
  result.votesB = result.votesA + 1400 ∧
  result.votesA = result.votesC + 500 ∧
  result.totalVotes = result.votesA + result.votesB + result.votesC :=
by
  sorry

#check election_result

end NUMINAMATH_CALUDE_election_result_l3813_381375


namespace NUMINAMATH_CALUDE_game_cost_l3813_381301

def initial_money : ℕ := 12
def toy_cost : ℕ := 2
def num_toys : ℕ := 2

theorem game_cost : 
  initial_money - (toy_cost * num_toys) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_game_cost_l3813_381301


namespace NUMINAMATH_CALUDE_scaled_badge_height_l3813_381302

/-- Calculates the height of a scaled rectangle while maintaining proportionality -/
def scaledHeight (originalWidth originalHeight scaledWidth : ℚ) : ℚ :=
  (originalHeight * scaledWidth) / originalWidth

/-- Theorem stating that scaling a 4x3 rectangle to width 12 results in height 9 -/
theorem scaled_badge_height :
  let originalWidth : ℚ := 4
  let originalHeight : ℚ := 3
  let scaledWidth : ℚ := 12
  scaledHeight originalWidth originalHeight scaledWidth = 9 := by
  sorry

end NUMINAMATH_CALUDE_scaled_badge_height_l3813_381302


namespace NUMINAMATH_CALUDE_tenth_pebble_count_l3813_381339

def pebble_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | (n + 4) => pebble_sequence (n + 3) + 3 * (n + 4) - 2

theorem tenth_pebble_count : pebble_sequence 9 = 145 := by
  sorry

end NUMINAMATH_CALUDE_tenth_pebble_count_l3813_381339


namespace NUMINAMATH_CALUDE_sin_135_degrees_l3813_381316

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_135_degrees_l3813_381316


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3813_381328

theorem gcd_factorial_eight_and_factorial_six_squared : 
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_and_factorial_six_squared_l3813_381328


namespace NUMINAMATH_CALUDE_perpendicular_a_parallel_distance_l3813_381380

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := 2 * a * x + y - 1 = 0
def l₂ (a x y : ℝ) : Prop := a * x + (a - 1) * y + 1 = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := 
  ∃ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ ∧ (x₁ - x₂) * (y₁ - y₂) = -1

-- Define parallelism of lines
def parallel (a : ℝ) : Prop := 
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ → 2 * a * (a - 1) = a

-- Theorem for perpendicular case
theorem perpendicular_a : ∀ a : ℝ, perpendicular a → a = -1 ∨ a = 1/2 :=
sorry

-- Theorem for parallel case
theorem parallel_distance : ∀ a : ℝ, parallel a → a ≠ 1 → 
  ∃ d : ℝ, d = (3 * Real.sqrt 10) / 10 ∧ 
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ → 
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = d^2)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_a_parallel_distance_l3813_381380


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3813_381306

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2 - x| ≥ 1} = {x : ℝ | x ≤ 1 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l3813_381306


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3813_381331

theorem sum_of_fractions_equals_one (x y z : ℝ) (h : x * y * z = 1) :
  (1 / (1 + x + x * y)) + (1 / (1 + y + y * z)) + (1 / (1 + z + z * x)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l3813_381331


namespace NUMINAMATH_CALUDE_victoria_wheat_flour_packets_l3813_381385

/-- Calculates the number of wheat flour packets bought given the initial amount,
    costs of items, and remaining balance. -/
def wheat_flour_packets (initial_amount : ℕ) (rice_cost : ℕ) (rice_packets : ℕ) 
                        (soda_cost : ℕ) (wheat_flour_cost : ℕ) (remaining_balance : ℕ) : ℕ :=
  let total_spent := initial_amount - remaining_balance
  let rice_soda_cost := rice_cost * rice_packets + soda_cost
  let wheat_flour_total := total_spent - rice_soda_cost
  wheat_flour_total / wheat_flour_cost

/-- Theorem stating that Victoria bought 3 packets of wheat flour -/
theorem victoria_wheat_flour_packets : 
  wheat_flour_packets 500 20 2 150 25 235 = 3 := by
  sorry

end NUMINAMATH_CALUDE_victoria_wheat_flour_packets_l3813_381385


namespace NUMINAMATH_CALUDE_simplify_expression_l3813_381398

theorem simplify_expression : 18 * (7 / 12) * (1 / 6) + 1 / 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3813_381398


namespace NUMINAMATH_CALUDE_find_tuesday_date_l3813_381378

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in a month -/
structure Date where
  day : ℕ
  month : ℕ
  dayOfWeek : DayOfWeek

/-- Given conditions of the problem -/
def problemConditions (tuesdayDate : Date) (thirdFridayDate : Date) : Prop :=
  tuesdayDate.dayOfWeek = DayOfWeek.Tuesday ∧
  thirdFridayDate.dayOfWeek = DayOfWeek.Friday ∧
  thirdFridayDate.day = 15 ∧
  thirdFridayDate.day + 3 = 18

/-- The theorem to prove -/
theorem find_tuesday_date (tuesdayDate : Date) (thirdFridayDate : Date) :
  problemConditions tuesdayDate thirdFridayDate →
  tuesdayDate.day = 29 ∧ tuesdayDate.month + 1 = thirdFridayDate.month :=
by sorry

end NUMINAMATH_CALUDE_find_tuesday_date_l3813_381378


namespace NUMINAMATH_CALUDE_sequence_2011th_term_l3813_381330

theorem sequence_2011th_term (a : ℕ → ℝ) 
  (h1 : a 1 = 0)
  (h2 : ∀ n : ℕ, a n + a (n + 1) = 2) : 
  a 2011 = 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_2011th_term_l3813_381330


namespace NUMINAMATH_CALUDE_bakery_rolls_combinations_l3813_381387

theorem bakery_rolls_combinations :
  let total_rolls : ℕ := 9
  let kinds_of_rolls : ℕ := 4
  let min_per_kind : ℕ := 1
  let remaining_rolls : ℕ := total_rolls - kinds_of_rolls * min_per_kind
  Nat.choose (kinds_of_rolls + remaining_rolls - 1) remaining_rolls = 56 := by
  sorry

end NUMINAMATH_CALUDE_bakery_rolls_combinations_l3813_381387


namespace NUMINAMATH_CALUDE_sum_172_83_base4_l3813_381388

/-- Converts a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

/-- Checks if a list of natural numbers represents a valid base 4 number -/
def isValidBase4 (l : List ℕ) : Prop :=
  ∀ d ∈ l, d < 4

theorem sum_172_83_base4 :
  toBase4 (172 + 83) = [3, 3, 3, 3, 3] ∧ isValidBase4 [3, 3, 3, 3, 3] := by
  sorry

end NUMINAMATH_CALUDE_sum_172_83_base4_l3813_381388


namespace NUMINAMATH_CALUDE_exists_n_order_of_two_congruent_l3813_381324

/-- The order of 2 in n! -/
def v (n : ℕ) : ℕ := sorry

/-- For any positive integers a and m, there exists n > 1 such that v(n) ≡ a (mod m) -/
theorem exists_n_order_of_two_congruent (a m : ℕ+) : ∃ n : ℕ, n > 1 ∧ v n % m = a % m := by
  sorry

end NUMINAMATH_CALUDE_exists_n_order_of_two_congruent_l3813_381324


namespace NUMINAMATH_CALUDE_basketball_not_tabletennis_l3813_381334

theorem basketball_not_tabletennis (total : ℕ) (basketball : ℕ) (tabletennis : ℕ) (neither : ℕ)
  (h1 : total = 40)
  (h2 : basketball = 24)
  (h3 : tabletennis = 16)
  (h4 : neither = 6) :
  basketball - (basketball + tabletennis - (total - neither)) = 18 :=
by sorry

end NUMINAMATH_CALUDE_basketball_not_tabletennis_l3813_381334


namespace NUMINAMATH_CALUDE_extra_money_spent_theorem_l3813_381358

/-- Represents the price of radishes and pork ribs last month and this month --/
structure PriceData where
  radish_last : ℝ
  pork_last : ℝ
  radish_this : ℝ
  pork_this : ℝ

/-- Calculates the extra money spent given the price data and quantities --/
def extra_money_spent (p : PriceData) (radish_qty : ℝ) (pork_qty : ℝ) : ℝ :=
  radish_qty * (p.radish_this - p.radish_last) + pork_qty * (p.pork_this - p.pork_last)

/-- Theorem stating the extra money spent on radishes and pork ribs --/
theorem extra_money_spent_theorem (a : ℝ) :
  let p : PriceData := {
    radish_last := a,
    pork_last := 7 * a + 2,
    radish_this := 1.25 * a,
    pork_this := 1.2 * (7 * a + 2)
  }
  extra_money_spent p 3 2 = 3.55 * a + 0.8 := by
  sorry

#check extra_money_spent_theorem

end NUMINAMATH_CALUDE_extra_money_spent_theorem_l3813_381358


namespace NUMINAMATH_CALUDE_evaluate_expression_l3813_381382

/-- Given x = -1 and y = 2, prove that -2x²y-3(2xy-x²y)+4xy evaluates to 6 -/
theorem evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = 2) :
  -2 * x^2 * y - 3 * (2 * x * y - x^2 * y) + 4 * x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3813_381382


namespace NUMINAMATH_CALUDE_imaginary_part_of_f_i_over_i_l3813_381313

-- Define the complex function f(x) = x^3 - 1
def f (x : ℂ) : ℂ := x^3 - 1

-- State the theorem
theorem imaginary_part_of_f_i_over_i :
  Complex.im (f Complex.I / Complex.I) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_f_i_over_i_l3813_381313


namespace NUMINAMATH_CALUDE_min_days_for_progress_ratio_l3813_381373

theorem min_days_for_progress_ratio : ∃ n : ℕ, n = 23 ∧ 
  (∀ x : ℕ, (1.2 : ℝ)^x / (0.8 : ℝ)^x ≥ 10000 → x ≥ n) ∧
  (1.2 : ℝ)^n / (0.8 : ℝ)^n ≥ 10000 :=
by sorry

end NUMINAMATH_CALUDE_min_days_for_progress_ratio_l3813_381373


namespace NUMINAMATH_CALUDE_isosceles_triangle_coordinates_l3813_381304

-- Define the points
def A : ℝ × ℝ := (13, 7)
def B : ℝ × ℝ := (5, -1)
def D : ℝ × ℝ := (2, 2)

-- Define the theorem
theorem isosceles_triangle_coordinates :
  ∃ (C : ℝ × ℝ),
    -- AB = AC (isosceles triangle)
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
    -- AD ⟂ BC (altitude condition)
    (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0 ∧
    -- D is on BC
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ D.1 = t * B.1 + (1 - t) * C.1 ∧ D.2 = t * B.2 + (1 - t) * C.2 ∧
    -- C has coordinates (-1, 5)
    C = (-1, 5) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_coordinates_l3813_381304


namespace NUMINAMATH_CALUDE_supermarket_spending_l3813_381363

theorem supermarket_spending (total : ℚ) : 
  (1/4 : ℚ) * total + (1/3 : ℚ) * total + (1/6 : ℚ) * total + 6 = total →
  total = 24 := by
  sorry

end NUMINAMATH_CALUDE_supermarket_spending_l3813_381363


namespace NUMINAMATH_CALUDE_percent_sum_of_x_l3813_381307

theorem percent_sum_of_x (x y z v w : ℝ) : 
  (0.45 * z = 0.39 * y) →
  (y = 0.75 * x) →
  (v = 0.80 * z) →
  (w = 0.60 * y) →
  (v + w = 0.97 * x) :=
by sorry

end NUMINAMATH_CALUDE_percent_sum_of_x_l3813_381307


namespace NUMINAMATH_CALUDE_other_communities_count_l3813_381379

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ)
  (h_total : total = 850)
  (h_muslim : muslim_percent = 44 / 100)
  (h_hindu : hindu_percent = 32 / 100)
  (h_sikh : sikh_percent = 10 / 100) :
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 119 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l3813_381379


namespace NUMINAMATH_CALUDE_multiply_fractions_l3813_381369

theorem multiply_fractions (a b : ℝ) : (1 / 3) * a^2 * (-6 * a * b) = -2 * a^3 * b := by
  sorry

end NUMINAMATH_CALUDE_multiply_fractions_l3813_381369


namespace NUMINAMATH_CALUDE_complex_multiplication_sum_l3813_381353

theorem complex_multiplication_sum (a b : ℝ) (i : ℂ) : 
  i ^ 2 = -1 → 
  a + b * i = (1 + i) * (2 - i) → 
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_sum_l3813_381353


namespace NUMINAMATH_CALUDE_y_value_at_27_l3813_381395

-- Define the relation between y and x
def y (k : ℝ) (x : ℝ) : ℝ := k * x^(1/3)

-- State the theorem
theorem y_value_at_27 (k : ℝ) :
  y k 8 = 4 → y k 27 = 6 := by
  sorry

end NUMINAMATH_CALUDE_y_value_at_27_l3813_381395


namespace NUMINAMATH_CALUDE_initial_average_marks_l3813_381329

theorem initial_average_marks (n : ℕ) (wrong_mark correct_mark : ℝ) (correct_avg : ℝ) :
  n = 10 →
  wrong_mark = 50 →
  correct_mark = 10 →
  correct_avg = 96 →
  (n * correct_avg * n - (wrong_mark - correct_mark)) / n = 92 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_average_marks_l3813_381329


namespace NUMINAMATH_CALUDE_intersection_point_l3813_381364

def A : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 - 1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1 + 3}

theorem intersection_point (m : ℝ × ℝ) (hA : m ∈ A) (hB : m ∈ B) : m = (4, 7) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l3813_381364


namespace NUMINAMATH_CALUDE_calculation_proof_l3813_381394

theorem calculation_proof : (2.5 * (30.1 + 0.5)) / 1.5 = 51 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3813_381394


namespace NUMINAMATH_CALUDE_hyperbola_probability_l3813_381393

-- Define the set of possible values for m and n
def S : Set ℕ := {1, 2, 3}

-- Define the condition for (m, n) to be on the hyperbola
def on_hyperbola (m n : ℕ) : Prop := n = 6 / m

-- Define the probability space
def total_outcomes : ℕ := 6

-- Define the favorable outcomes
def favorable_outcomes : ℕ := 2

-- State the theorem
theorem hyperbola_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_probability_l3813_381393


namespace NUMINAMATH_CALUDE_roots_sum_fraction_eq_neg_two_l3813_381348

theorem roots_sum_fraction_eq_neg_two (z₁ z₂ : ℂ) 
  (h₁ : z₁^2 + z₁ + 1 = 0) 
  (h₂ : z₂^2 + z₂ + 1 = 0) 
  (h₃ : z₁ ≠ z₂) : 
  z₂ / (z₁ + 1) + z₁ / (z₂ + 1) = -2 := by sorry

end NUMINAMATH_CALUDE_roots_sum_fraction_eq_neg_two_l3813_381348


namespace NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l3813_381338

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 47)
  (h3 : max_ac_no_stripes = 45) :
  ∃ (min_cars_with_stripes : ℕ), 
    min_cars_with_stripes = 8 ∧ 
    ∀ (cars_with_stripes : ℕ), 
      cars_with_stripes ≥ min_cars_with_stripes →
      cars_with_stripes + max_ac_no_stripes ≥ total_cars - cars_without_ac :=
by
  sorry

end NUMINAMATH_CALUDE_min_cars_with_racing_stripes_l3813_381338


namespace NUMINAMATH_CALUDE_triangle_area_from_squares_l3813_381359

/-- Given four squares with areas 256, 64, 225, and 49, prove that the area of the triangle formed by three of these squares is 60 -/
theorem triangle_area_from_squares (s₁ s₂ s₃ s₄ : ℝ) 
  (h₁ : s₁ = 256) (h₂ : s₂ = 64) (h₃ : s₃ = 225) (h₄ : s₄ = 49) : 
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ a^2 = s₃ ∧ b^2 = s₂ ∧ c^2 = s₁ ∧ (1/2 * a * b = 60) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_squares_l3813_381359


namespace NUMINAMATH_CALUDE_lottery_probability_l3813_381389

def eligible_numbers : Finset ℕ := {1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 80, 100}

def valid_combinations : ℕ := 10

theorem lottery_probability :
  let total_combinations := Nat.choose (Finset.card eligible_numbers - 1) 5
  (valid_combinations : ℚ) / total_combinations = 10 / 3003 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l3813_381389


namespace NUMINAMATH_CALUDE_unique_value_at_two_l3813_381336

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x * f y - f (x * y) = x^2 + y^2

/-- The theorem stating that f(2) = 5 for any function satisfying the functional equation -/
theorem unique_value_at_two (f : ℝ → ℝ) (h : FunctionalEquation f) : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_value_at_two_l3813_381336
