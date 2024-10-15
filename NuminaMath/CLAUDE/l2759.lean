import Mathlib

namespace NUMINAMATH_CALUDE_ab_plus_cd_value_l2759_275916

theorem ab_plus_cd_value (a b c d : ℝ) 
  (eq1 : a + b + c = 3)
  (eq2 : a + b + d = -5)
  (eq3 : a + c + d = 10)
  (eq4 : b + c + d = -1) :
  a * b + c * d = -274/9 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_value_l2759_275916


namespace NUMINAMATH_CALUDE_waldo_total_time_l2759_275988

/-- The number of "Where's Waldo?" books -/
def num_books : ℕ := 15

/-- The number of puzzles per book -/
def puzzles_per_book : ℕ := 30

/-- The average time (in minutes) to find Waldo in a puzzle -/
def time_per_puzzle : ℕ := 3

/-- The total time (in minutes) to find Waldo in all puzzles across all books -/
def total_time : ℕ := num_books * puzzles_per_book * time_per_puzzle

theorem waldo_total_time : total_time = 1350 := by
  sorry

end NUMINAMATH_CALUDE_waldo_total_time_l2759_275988


namespace NUMINAMATH_CALUDE_original_number_proof_l2759_275972

theorem original_number_proof (x : ℚ) : 
  2 + (1 / x) = 10 / 3 → x = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2759_275972


namespace NUMINAMATH_CALUDE_ellipse_condition_l2759_275989

/-- The equation m(x^2 + y^2 + 2y + 1) = (x - 2y + 3)^2 represents an ellipse if and only if m ∈ (5, +∞) -/
theorem ellipse_condition (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) ↔ m > 5 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2759_275989


namespace NUMINAMATH_CALUDE_gold_distribution_l2759_275977

theorem gold_distribution (n : ℕ) (a₁ : ℚ) (d : ℚ) : 
  n = 10 → 
  (4 * a₁ + 6 * d = 3) → 
  (3 * a₁ + 24 * d = 4) → 
  d = 7/78 :=
by sorry

end NUMINAMATH_CALUDE_gold_distribution_l2759_275977


namespace NUMINAMATH_CALUDE_first_digit_powers_of_3_and_7_l2759_275955

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def first_digit (n : ℕ) : ℕ :=
  if n < 10 then n else first_digit (n / 10)

theorem first_digit_powers_of_3_and_7 :
  ∃ (m n : ℕ), is_three_digit (3^m) ∧ is_three_digit (7^n) ∧ 
  first_digit (3^m) = first_digit (7^n) ∧
  first_digit (3^m) = 3 ∧
  ∀ (k : ℕ), k ≠ 3 → 
    ¬(∃ (p q : ℕ), is_three_digit (3^p) ∧ is_three_digit (7^q) ∧ 
    first_digit (3^p) = first_digit (7^q) ∧ first_digit (3^p) = k) :=
by sorry

end NUMINAMATH_CALUDE_first_digit_powers_of_3_and_7_l2759_275955


namespace NUMINAMATH_CALUDE_books_remaining_after_sale_l2759_275981

-- Define the initial number of books
def initial_books : Nat := 136

-- Define the number of books sold
def books_sold : Nat := 109

-- Theorem to prove
theorem books_remaining_after_sale : 
  initial_books - books_sold = 27 := by sorry

end NUMINAMATH_CALUDE_books_remaining_after_sale_l2759_275981


namespace NUMINAMATH_CALUDE_circle_and_tangent_line_l2759_275954

/-- Circle C with equation x^2+y^2-8x+6y+21=0 -/
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1^2 + p.2^2 - 8*p.1 + 6*p.2 + 21) = 0}

/-- Point A with coordinates (-6, 7) -/
def point_A : ℝ × ℝ := (-6, 7)

/-- A line is tangent to a circle if it intersects the circle at exactly one point -/
def is_tangent_line (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∃! p, p ∈ l ∩ c

/-- The set of all lines passing through point A -/
def lines_through_A : Set (Set (ℝ × ℝ)) :=
  {l | point_A ∈ l ∧ ∃ k, l = {p | p.2 - 7 = k * (p.1 + 6)}}

theorem circle_and_tangent_line :
  ∃ l ∈ lines_through_A,
    is_tangent_line l circle_C ∧
    (∃ c r, c = (4, -3) ∧ r = 2 ∧
      ∀ p ∈ circle_C, (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    (l = {p | 3*p.1 + 4*p.2 - 10 = 0} ∨ l = {p | 4*p.1 + 3*p.2 + 3 = 0}) :=
  sorry

end NUMINAMATH_CALUDE_circle_and_tangent_line_l2759_275954


namespace NUMINAMATH_CALUDE_absolute_value_and_quadratic_equation_l2759_275957

theorem absolute_value_and_quadratic_equation :
  ∀ (b c : ℝ),
  (∀ x : ℝ, |x - 4| = 3 ↔ x^2 + b*x + c = 0) →
  b = -8 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_and_quadratic_equation_l2759_275957


namespace NUMINAMATH_CALUDE_x_lt_5_necessary_not_sufficient_l2759_275931

theorem x_lt_5_necessary_not_sufficient :
  (∀ x : ℝ, -2 < x ∧ x < 4 → x < 5) ∧
  (∃ x : ℝ, x < 5 ∧ ¬(-2 < x ∧ x < 4)) :=
by sorry

end NUMINAMATH_CALUDE_x_lt_5_necessary_not_sufficient_l2759_275931


namespace NUMINAMATH_CALUDE_point_on_circle_after_rotation_l2759_275973

def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

def arc_length (θ : ℝ) : ℝ := θ

theorem point_on_circle_after_rotation 
  (start_x start_y : ℝ) 
  (θ : ℝ) 
  (h_start : unit_circle start_x start_y) 
  (h_θ : arc_length θ = 2 * Real.pi / 3) :
  ∃ (end_x end_y : ℝ), 
    unit_circle end_x end_y ∧ 
    end_x = -1/2 ∧ 
    end_y = Real.sqrt 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_point_on_circle_after_rotation_l2759_275973


namespace NUMINAMATH_CALUDE_questionnaire_responses_l2759_275976

theorem questionnaire_responses (response_rate : ℝ) (min_questionnaires : ℕ) (responses_needed : ℕ) : 
  response_rate = 0.60 → 
  min_questionnaires = 370 → 
  responses_needed = ⌊response_rate * min_questionnaires⌋ →
  responses_needed = 222 := by
sorry

end NUMINAMATH_CALUDE_questionnaire_responses_l2759_275976


namespace NUMINAMATH_CALUDE_kimberly_store_visits_l2759_275965

/-- Represents the number of peanuts Kimberly buys each time she goes to the store. -/
def peanuts_per_visit : ℕ := 7

/-- Represents the total number of peanuts Kimberly bought last month. -/
def total_peanuts : ℕ := 21

/-- Represents the number of times Kimberly went to the store last month. -/
def store_visits : ℕ := total_peanuts / peanuts_per_visit

/-- Proves that Kimberly went to the store 3 times last month. -/
theorem kimberly_store_visits : store_visits = 3 := by
  sorry

end NUMINAMATH_CALUDE_kimberly_store_visits_l2759_275965


namespace NUMINAMATH_CALUDE_determinant_maximum_value_l2759_275934

open Real Matrix

theorem determinant_maximum_value (θ φ : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 1, 1 + sin θ, 1 + cos φ; 1 + cos θ, 1 + sin φ, 1]
  ∃ (θ' φ' : ℝ), ∀ (θ φ : ℝ), det A ≤ det (!![1, 1, 1; 1, 1 + sin θ', 1 + cos φ'; 1 + cos θ', 1 + sin φ', 1]) ∧
  det (!![1, 1, 1; 1, 1 + sin θ', 1 + cos φ'; 1 + cos θ', 1 + sin φ', 1]) = 1 :=
by sorry

end NUMINAMATH_CALUDE_determinant_maximum_value_l2759_275934


namespace NUMINAMATH_CALUDE_b_over_c_equals_one_l2759_275922

theorem b_over_c_equals_one (a b c d : ℕ) : 
  0 < a ∧ a < 4 ∧ 
  0 < b ∧ b < 4 ∧ 
  0 < c ∧ c < 4 ∧ 
  0 < d ∧ d < 4 ∧ 
  4^a + 3^b + 2^c + 1^d = 78 → 
  b / c = 1 := by
  sorry

end NUMINAMATH_CALUDE_b_over_c_equals_one_l2759_275922


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l2759_275992

/-- An equilateral triangle with height 9 and perimeter 36 has area 54 -/
theorem equilateral_triangle_area (h : ℝ) (p : ℝ) :
  h = 9 → p = 36 → (1/2) * (p/3) * h = 54 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l2759_275992


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l2759_275995

theorem arithmetic_geometric_inequality 
  (a b c d h k : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d) (pos_h : 0 < h) (pos_k : 0 < k)
  (arith_prog : ∃ t : ℝ, t > 0 ∧ a = d + 3*t ∧ b = d + 2*t ∧ c = d + t)
  (geom_prog : ∃ r : ℝ, r > 1 ∧ a = d * r^3 ∧ h = d * r^2 ∧ k = d * r) :
  b * c > h * k := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l2759_275995


namespace NUMINAMATH_CALUDE_smallest_x_value_l2759_275924

theorem smallest_x_value (x : ℝ) : 
  x ≠ 9 → x ≠ -7 → (x^2 - 5*x - 84) / (x - 9) = 4 / (x + 7) → 
  ∃ (y : ℝ), y = -8 ∧ (y^2 - 5*y - 84) / (y - 9) = 4 / (y + 7) ∧ 
  ∀ (z : ℝ), z ≠ 9 → z ≠ -7 → (z^2 - 5*z - 84) / (z - 9) = 4 / (z + 7) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2759_275924


namespace NUMINAMATH_CALUDE_stratified_sampling_l2759_275906

theorem stratified_sampling (total : ℕ) (sample_size : ℕ) (group_size : ℕ) 
  (h1 : total = 700) 
  (h2 : sample_size = 14) 
  (h3 : group_size = 300) :
  (group_size * sample_size) / total = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2759_275906


namespace NUMINAMATH_CALUDE_puppies_sold_l2759_275960

theorem puppies_sold (initial_puppies initial_kittens kittens_sold remaining_pets : ℕ) :
  initial_puppies = 7 →
  initial_kittens = 6 →
  kittens_sold = 3 →
  remaining_pets = 8 →
  initial_puppies + initial_kittens - kittens_sold - remaining_pets = 2 := by
  sorry

#check puppies_sold

end NUMINAMATH_CALUDE_puppies_sold_l2759_275960


namespace NUMINAMATH_CALUDE_ivy_stripping_l2759_275970

/-- The number of feet of ivy Cary strips daily -/
def daily_strip : ℝ := 6

/-- The initial ivy coverage in feet -/
def initial_coverage : ℝ := 40

/-- The number of days it takes to remove all ivy -/
def days_to_remove : ℝ := 10

/-- The number of feet the ivy grows each night -/
def nightly_growth : ℝ := 2

theorem ivy_stripping :
  daily_strip * days_to_remove - nightly_growth * days_to_remove = initial_coverage :=
sorry

end NUMINAMATH_CALUDE_ivy_stripping_l2759_275970


namespace NUMINAMATH_CALUDE_factorization_proof_l2759_275950

theorem factorization_proof (a x y : ℝ) : a * x - a * y = a * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2759_275950


namespace NUMINAMATH_CALUDE_triangle_angle_measures_l2759_275914

theorem triangle_angle_measures (A B C : ℝ) 
  (h1 : B - A = 5)
  (h2 : C - B = 20)
  (h3 : A + B + C = 180) : 
  A = 50 ∧ B = 55 ∧ C = 75 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measures_l2759_275914


namespace NUMINAMATH_CALUDE_sector_radius_l2759_275952

/-- Given a sector of a circle with perimeter 144 cm and central angle π/3 radians,
    prove that the radius of the circle is 432 / (6 + π) cm. -/
theorem sector_radius (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 144) (h2 : central_angle = π/3) :
  ∃ r : ℝ, r = 432 / (6 + π) ∧ perimeter = 2*r + r * central_angle := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_l2759_275952


namespace NUMINAMATH_CALUDE_simplify_expression_l2759_275928

theorem simplify_expression (n : ℕ) : (3^(n+3) - 3*(3^n)) / (3*(3^(n+2))) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2759_275928


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2759_275919

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_a1 : a 1 = 6)
  (h_a3 : a 3 = 2) :
  a 5 = -2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l2759_275919


namespace NUMINAMATH_CALUDE_max_ways_to_schedule_single_game_l2759_275908

/-- Represents a chess tournament between two teams -/
structure ChessTournament where
  team_size : Nat
  total_games : Nat
  games_per_day : Nat → Nat

/-- The specific tournament configuration -/
def tournament : ChessTournament :=
  { team_size := 15,
    total_games := 15 * 15,
    games_per_day := fun d => if d = 1 then 15 else 1 }

/-- The number of ways to schedule a single game -/
def ways_to_schedule_single_game (t : ChessTournament) : Nat :=
  t.total_games - t.team_size

theorem max_ways_to_schedule_single_game :
  ways_to_schedule_single_game tournament ≤ 120 :=
sorry

end NUMINAMATH_CALUDE_max_ways_to_schedule_single_game_l2759_275908


namespace NUMINAMATH_CALUDE_sum_of_common_elements_l2759_275966

-- Define the arithmetic progression
def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n

-- Define the geometric progression
def geometric_progression (k : ℕ) : ℕ := 10 * 2^k

-- Define the sequence of common elements
def common_elements (n : ℕ) : ℕ := 10 * 4^n

-- Theorem statement
theorem sum_of_common_elements : 
  (Finset.range 10).sum common_elements = 3495250 := by sorry

end NUMINAMATH_CALUDE_sum_of_common_elements_l2759_275966


namespace NUMINAMATH_CALUDE_base_prime_repr_360_l2759_275945

/-- Base prime representation of a natural number --/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- The base prime representation of 360 --/
theorem base_prime_repr_360 : base_prime_repr 360 = [3, 2, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_repr_360_l2759_275945


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l2759_275915

theorem complex_fraction_equals_i (m n : ℝ) (h : m + Complex.I = 1 + n * Complex.I) :
  (m + n * Complex.I) / (m - n * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l2759_275915


namespace NUMINAMATH_CALUDE_value_of_b_l2759_275985

theorem value_of_b (a b t : ℝ) 
  (eq1 : a - t / 6 * b = 20)
  (eq2 : a - t / 5 * b = -10)
  (t_val : t = 60) : b = 15 := by
  sorry

end NUMINAMATH_CALUDE_value_of_b_l2759_275985


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2759_275913

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^4 + 12 * b^4 + 50 * c^4 + 1 / (9 * a * b * c) ≥ 2 * Real.sqrt (20 / 3) :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (8 * a^4 + 12 * b^4 + 50 * c^4 + 1 / (9 * a * b * c) = 2 * Real.sqrt (20 / 3)) ↔
  (a = (3/2)^(1/4) * b ∧ b = (25/6)^(1/4) * c ∧ c = (4/25)^(1/4) * a) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l2759_275913


namespace NUMINAMATH_CALUDE_max_uncolored_cubes_l2759_275946

/-- Represents a rectangular prism --/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular prism --/
def volume (p : RectangularPrism) : ℕ := p.length * p.width * p.height

/-- Calculates the number of interior cubes in a rectangular prism --/
def interiorCubes (p : RectangularPrism) : ℕ :=
  (p.length - 2) * (p.width - 2) * (p.height - 2)

theorem max_uncolored_cubes (p : RectangularPrism) 
  (h_dim : p.length = 8 ∧ p.width = 8 ∧ p.height = 16) 
  (h_vol : volume p = 1024) :
  interiorCubes p = 504 := by
  sorry


end NUMINAMATH_CALUDE_max_uncolored_cubes_l2759_275946


namespace NUMINAMATH_CALUDE_min_sum_given_product_l2759_275900

theorem min_sum_given_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y - (x + y) = 1) :
  x + y ≥ 2 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_product_l2759_275900


namespace NUMINAMATH_CALUDE_percent_greater_l2759_275907

theorem percent_greater (w x y z : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x = 1.2 * y) (hyz : y = 1.2 * z) (hwx : w = 0.8 * x) :
  w = 1.152 * z := by
sorry

end NUMINAMATH_CALUDE_percent_greater_l2759_275907


namespace NUMINAMATH_CALUDE_quadratic_properties_l2759_275999

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  f : ℝ → ℝ
  quad : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  axis_sym : ∀ x, f (1 + x) = f (1 - x)
  vertex : f 1 = -4
  table_values : f (-2) = 5 ∧ f (-1) = 0 ∧ f 0 = -3 ∧ f 1 = -4 ∧ f 2 = -3 ∧ f 3 = 0

theorem quadratic_properties (f : QuadraticFunction) :
  (∃ a > 0, ∀ x, f.f x = a * x^2 + f.f 1 - a) ∧
  f.f 4 = 5 ∧
  f.f (-3) > f.f 2 ∧
  {x : ℝ | f.f x < 0} = {x : ℝ | -1 < x ∧ x < 3} ∧
  {x : ℝ | f.f x = 5} = {-2, 4} := by
  sorry


end NUMINAMATH_CALUDE_quadratic_properties_l2759_275999


namespace NUMINAMATH_CALUDE_determinant_equation_solution_l2759_275920

/-- Definition of a 2x2 determinant -/
def det (a b c d : ℚ) : ℚ := a * d - b * c

/-- Theorem: If |x-2 x+3; x+1 x-2| = 13, then x = -3/2 -/
theorem determinant_equation_solution :
  ∀ x : ℚ, det (x - 2) (x + 3) (x + 1) (x - 2) = 13 → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_equation_solution_l2759_275920


namespace NUMINAMATH_CALUDE_golden_rectangle_ratio_l2759_275933

theorem golden_rectangle_ratio (x y : ℝ) (h1 : x > y) (h2 : y > 0) : 
  (y / x = (x - y) / y) → (x / y = (1 + Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_golden_rectangle_ratio_l2759_275933


namespace NUMINAMATH_CALUDE_max_y_value_l2759_275979

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : y ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l2759_275979


namespace NUMINAMATH_CALUDE_log_xy_value_l2759_275967

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the theorem
theorem log_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : log (x * y^2) = 2) (h2 : log (x^3 * y) = 3) : 
  log (x * y) = 7/5 := by
  sorry


end NUMINAMATH_CALUDE_log_xy_value_l2759_275967


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2759_275930

theorem min_value_of_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  (a + b) / c + (b + c) / a + (c + a) / b + 3 ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2759_275930


namespace NUMINAMATH_CALUDE_a_greater_than_b_l2759_275942

theorem a_greater_than_b (A B : ℝ) (h1 : A ≠ 0) (h2 : B ≠ 0) (h3 : A * 4 = B * 5) : A > B := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l2759_275942


namespace NUMINAMATH_CALUDE_min_translation_overlap_l2759_275925

theorem min_translation_overlap (φ : Real) : 
  (φ > 0) →
  (∀ x, Real.sin (2 * (x + φ)) = Real.sin (2 * x - 2 * φ + Real.pi / 3)) →
  φ ≥ Real.pi / 12 :=
sorry

end NUMINAMATH_CALUDE_min_translation_overlap_l2759_275925


namespace NUMINAMATH_CALUDE_income_a_is_4000_l2759_275953

/-- Represents the financial situation of two individuals A and B -/
structure FinancialSituation where
  incomeRatio : Rat
  expenditureRatio : Rat
  savings : ℕ

/-- Calculates the income of individual A given the financial situation -/
def incomeA (fs : FinancialSituation) : ℕ := sorry

/-- Theorem stating that given the specific financial situation, the income of A is $4000 -/
theorem income_a_is_4000 :
  let fs : FinancialSituation := {
    incomeRatio := 5 / 4,
    expenditureRatio := 3 / 2,
    savings := 1600
  }
  incomeA fs = 4000 := by sorry

end NUMINAMATH_CALUDE_income_a_is_4000_l2759_275953


namespace NUMINAMATH_CALUDE_function_inequality_l2759_275941

theorem function_inequality (a b : ℝ) (h_a : a > 0) : 
  (∃ x : ℝ, x > 0 ∧ Real.log x - a * x - b ≥ 0) → a * b ≤ Real.exp (-2) := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2759_275941


namespace NUMINAMATH_CALUDE_equation_a_solution_equation_b_no_solution_l2759_275971

-- Part (a)
theorem equation_a_solution (x : ℚ) : 
  1 + 1 / (2 + 1 / ((4*x + 1) / (2*x + 1) - 1 / (2 + 1/x))) = 19/14 ↔ x = 1/2 :=
sorry

-- Part (b)
theorem equation_b_no_solution :
  ¬∃ (x : ℚ), ((2*x - 1)/2 + 4/3) / ((x - 1)/3 - 1/2 * (1 - 1/3)) - 
  (x + 4) / ((2*x + 1)/2 + 1/5 - 2 - 1/(1 + 1/(2 + 1/3))) = (9 - 2*x) / (2*x - 4) :=
sorry

end NUMINAMATH_CALUDE_equation_a_solution_equation_b_no_solution_l2759_275971


namespace NUMINAMATH_CALUDE_distance_between_points_l2759_275949

/-- The distance between two points when two vehicles move towards each other -/
theorem distance_between_points (v1 v2 t : ℝ) (h1 : v1 > 0) (h2 : v2 > 0) (h3 : t > 0) :
  let d := (v1 + v2) * t
  d = v1 * t + v2 * t :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_l2759_275949


namespace NUMINAMATH_CALUDE_floor_times_x_equals_88_l2759_275982

theorem floor_times_x_equals_88 (x : ℝ) (h1 : x > 0) (h2 : ⌊x⌋ * x = 88) : x = 88 / 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_x_equals_88_l2759_275982


namespace NUMINAMATH_CALUDE_multiplication_correction_l2759_275975

theorem multiplication_correction (n : ℕ) : 
  n * 987 = 559981 → 
  (∃ a b : ℕ, a ≠ 9 ∧ b ≠ 8 ∧ n * 987 = 5 * 100000 + a * 10000 + b * 1000 + 981) → 
  n * 987 = 559989 :=
by sorry

end NUMINAMATH_CALUDE_multiplication_correction_l2759_275975


namespace NUMINAMATH_CALUDE_evaluate_expression_l2759_275948

theorem evaluate_expression (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2759_275948


namespace NUMINAMATH_CALUDE_solution_to_system_l2759_275901

theorem solution_to_system (x y : ℝ) 
  (h1 : 9 * x^2 - 25 * y^2 = 0) 
  (h2 : x^2 + y^2 = 10) : 
  (x = 5 * Real.sqrt (45/17) / 3 ∨ x = -5 * Real.sqrt (45/17) / 3) ∧
  (y = Real.sqrt (45/17) ∨ y = -Real.sqrt (45/17)) := by
sorry


end NUMINAMATH_CALUDE_solution_to_system_l2759_275901


namespace NUMINAMATH_CALUDE_no_common_points_range_two_common_points_product_l2759_275958

noncomputable section

-- Define the functions f and g
def f (x : ℝ) := Real.log x
def g (a : ℝ) (x : ℝ) := a * x

-- Part I
theorem no_common_points_range (a : ℝ) :
  (∀ x > 0, f x ≠ g a x) → a > 1 / Real.exp 1 := by sorry

-- Part II
theorem two_common_points_product (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ → f x₁ = g a x₁ → f x₂ = g a x₂ → x₁ * x₂ > Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_no_common_points_range_two_common_points_product_l2759_275958


namespace NUMINAMATH_CALUDE_bankers_calculation_l2759_275918

/-- Proves that given specific banker's gain, banker's discount, and interest rate, the time period is 3 years -/
theorem bankers_calculation (bankers_gain : ℝ) (bankers_discount : ℝ) (interest_rate : ℝ) :
  bankers_gain = 270 →
  bankers_discount = 1020 →
  interest_rate = 0.12 →
  ∃ (time : ℝ), time = 3 ∧ bankers_discount = (bankers_discount - bankers_gain) * (1 + interest_rate * time) :=
by sorry

end NUMINAMATH_CALUDE_bankers_calculation_l2759_275918


namespace NUMINAMATH_CALUDE_billy_free_time_l2759_275956

/-- Proves that Billy has 16 hours of free time each day of the weekend given the specified conditions. -/
theorem billy_free_time (video_game_percentage : ℝ) (reading_percentage : ℝ)
  (pages_per_hour : ℕ) (pages_per_book : ℕ) (books_read : ℕ) :
  video_game_percentage = 0.75 →
  reading_percentage = 0.25 →
  pages_per_hour = 60 →
  pages_per_book = 80 →
  books_read = 3 →
  (books_read * pages_per_book : ℝ) / pages_per_hour / reading_percentage = 16 :=
by sorry

end NUMINAMATH_CALUDE_billy_free_time_l2759_275956


namespace NUMINAMATH_CALUDE_fourTangentCircles_l2759_275986

-- Define the circles C₁ and C₂
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the plane containing the circles
def Plane : Type := ℝ × ℝ

-- Define tangency between circles
def areTangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

-- Define the given conditions
def givenConditions (c1 c2 : Circle) : Prop :=
  c1.radius = 2 ∧ c2.radius = 2 ∧ areTangent c1 c2

-- Define a function to count tangent circles
def countTangentCircles (c1 c2 : Circle) : ℕ :=
  sorry

-- Theorem statement
theorem fourTangentCircles (c1 c2 : Circle) :
  givenConditions c1 c2 → countTangentCircles c1 c2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourTangentCircles_l2759_275986


namespace NUMINAMATH_CALUDE_min_perimeter_noncongruent_isosceles_triangles_l2759_275947

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- The area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base / 2 : ℝ) * Real.sqrt ((t.leg : ℝ)^2 - (t.base / 2 : ℝ)^2)

theorem min_perimeter_noncongruent_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    t1.base * 5 = t2.base * 6 ∧
    ∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      s1.base * 5 = s2.base * 6 →
      perimeter t1 ≤ perimeter s1 ∧
    perimeter t1 = 364 :=
by sorry

end NUMINAMATH_CALUDE_min_perimeter_noncongruent_isosceles_triangles_l2759_275947


namespace NUMINAMATH_CALUDE_b₁_value_l2759_275991

/-- The polynomial f(x) with 4 distinct real roots -/
def f (x : ℝ) : ℝ := 8 + 32*x - 12*x^2 - 4*x^3 + x^4

/-- The set of roots of f(x) -/
def roots_f : Set ℝ := {x | f x = 0}

/-- The polynomial g(x) with roots being squares of roots of f(x) -/
def g (b₀ b₁ b₂ b₃ : ℝ) (x : ℝ) : ℝ := b₀ + b₁*x + b₂*x^2 + b₃*x^3 + x^4

/-- The set of roots of g(x) -/
def roots_g (b₀ b₁ b₂ b₃ : ℝ) : Set ℝ := {x | g b₀ b₁ b₂ b₃ x = 0}

theorem b₁_value (b₀ b₁ b₂ b₃ : ℝ) :
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
    roots_f = {x₁, x₂, x₃, x₄} ∧ 
    roots_g b₀ b₁ b₂ b₃ = {x₁^2, x₂^2, x₃^2, x₄^2}) →
  b₁ = -1216 := by
sorry

end NUMINAMATH_CALUDE_b₁_value_l2759_275991


namespace NUMINAMATH_CALUDE_no_real_solutions_quadratic_inequality_l2759_275912

theorem no_real_solutions_quadratic_inequality :
  ¬∃ (x : ℝ), 3 * x^2 + 9 * x ≤ -12 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_quadratic_inequality_l2759_275912


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2759_275936

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 2) :
  (1 + (1 - x) / (x + 1)) / ((2 * x - 2) / (x^2 + 2 * x + 1)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2759_275936


namespace NUMINAMATH_CALUDE_chord_equation_parabola_l2759_275996

/-- Given a parabola y² = 4x and a chord AB with midpoint P(1,1), 
    the equation of the line containing chord AB is 2x - y - 1 = 0 -/
theorem chord_equation_parabola (A B : ℝ × ℝ) :
  let parabola := fun (p : ℝ × ℝ) ↦ p.2^2 = 4 * p.1
  let midpoint := (1, 1)
  let on_parabola := fun (p : ℝ × ℝ) ↦ parabola p
  let is_midpoint := fun (m p1 p2 : ℝ × ℝ) ↦ 
    m.1 = (p1.1 + p2.1) / 2 ∧ m.2 = (p1.2 + p2.2) / 2
  on_parabola A ∧ on_parabola B ∧ is_midpoint midpoint A B →
  ∃ (a b c : ℝ), a * A.1 + b * A.2 + c = 0 ∧
                  a * B.1 + b * B.2 + c = 0 ∧
                  (a, b, c) = (2, -1, -1) :=
by sorry

end NUMINAMATH_CALUDE_chord_equation_parabola_l2759_275996


namespace NUMINAMATH_CALUDE_solution_product_l2759_275951

theorem solution_product (p q : ℝ) : 
  (p - 7) * (2 * p + 11) = p^2 - 19 * p + 60 →
  (q - 7) * (2 * q + 11) = q^2 - 19 * q + 60 →
  p ≠ q →
  (p - 2) * (q - 2) = -55 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l2759_275951


namespace NUMINAMATH_CALUDE_not_always_true_l2759_275962

theorem not_always_true (r p q : ℝ) (hr : r > 0) (hpq : p * q ≠ 0) (hpqr : p * r > q * r) :
  ¬((-p > -q) ∨ (-p > q) ∨ (1 > -q/p) ∨ (1 < q/p)) := by
  sorry

end NUMINAMATH_CALUDE_not_always_true_l2759_275962


namespace NUMINAMATH_CALUDE_scientific_notation_of_0_00003_l2759_275997

theorem scientific_notation_of_0_00003 :
  ∃ (a : ℝ) (n : ℤ), 0.00003 = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3 ∧ n = -5 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_0_00003_l2759_275997


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l2759_275990

/-- Given a cylinder with height 15 cm and radius 5 cm, and a cone with the same radius
    and height one-third of the cylinder's, prove that the ratio of their volumes is 1/9. -/
theorem cone_cylinder_volume_ratio :
  let cylinder_height : ℝ := 15
  let cylinder_radius : ℝ := 5
  let cone_radius : ℝ := cylinder_radius
  let cone_height : ℝ := cylinder_height / 3
  let cylinder_volume := π * cylinder_radius^2 * cylinder_height
  let cone_volume := (1/3) * π * cone_radius^2 * cone_height
  cone_volume / cylinder_volume = 1/9 := by
sorry


end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l2759_275990


namespace NUMINAMATH_CALUDE_roots_sum_l2759_275959

theorem roots_sum (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 12*a*x - 13*b = 0 ↔ (x = c ∨ x = d)) →
  (∀ x : ℝ, x^2 - 12*c*x - 13*d = 0 ↔ (x = a ∨ x = b)) →
  a + b + c + d = 1716 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_l2759_275959


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l2759_275935

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l2759_275935


namespace NUMINAMATH_CALUDE_min_value_theorem_l2759_275993

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 3) :
  (2 / x + 1 / y) ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ = 3 ∧ 2 / x₀ + 1 / y₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2759_275993


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l2759_275994

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def number : ℕ := 3050000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation := {
  coefficient := 3.05,
  exponent := 6,
  is_valid := by sorry
}

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l2759_275994


namespace NUMINAMATH_CALUDE_cubic_fraction_simplification_l2759_275902

theorem cubic_fraction_simplification (a b : ℝ) (h : a = 6 ∧ b = 6) :
  (a^3 + b^3) / (a^2 - a*b + b^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_simplification_l2759_275902


namespace NUMINAMATH_CALUDE_tenth_pirate_coins_l2759_275968

/-- Represents the number of pirates --/
def num_pirates : ℕ := 10

/-- Represents the initial number of silver coins --/
def initial_silver : ℕ := 1050

/-- Represents the number of silver coins each pirate takes --/
def silver_per_pirate : ℕ := 100

/-- Calculates the remaining gold coins after k pirates have taken their share --/
def remaining_gold (initial_gold : ℕ) (k : ℕ) : ℚ :=
  (num_pirates - k : ℚ) / num_pirates * initial_gold

/-- Calculates the number of gold coins the 10th pirate receives --/
def gold_for_last_pirate (initial_gold : ℕ) : ℚ :=
  remaining_gold initial_gold (num_pirates - 1)

/-- Calculates the number of silver coins the 10th pirate receives --/
def silver_for_last_pirate : ℕ :=
  initial_silver - (num_pirates - 1) * silver_per_pirate

/-- Theorem stating that the 10th pirate receives 494 coins in total --/
theorem tenth_pirate_coins (initial_gold : ℕ) :
  ∃ (gold_coins : ℕ), gold_for_last_pirate initial_gold = gold_coins ∧
  gold_coins + silver_for_last_pirate = 494 :=
sorry

end NUMINAMATH_CALUDE_tenth_pirate_coins_l2759_275968


namespace NUMINAMATH_CALUDE_dihedral_angle_definition_inconsistency_l2759_275961

/-- Definition of a half-plane --/
def HalfPlane : Type := sorry

/-- Definition of a straight line --/
def StraightLine : Type := sorry

/-- Definition of a spatial figure --/
def SpatialFigure : Type := sorry

/-- Definition of a planar angle --/
def PlanarAngle : Type := sorry

/-- Incorrect definition of a dihedral angle --/
def IncorrectDihedralAngle : Type :=
  {angle : PlanarAngle // ∃ (hp1 hp2 : HalfPlane) (l : StraightLine),
    angle = sorry }

/-- Correct definition of a dihedral angle --/
def CorrectDihedralAngle : Type :=
  {sf : SpatialFigure // ∃ (hp1 hp2 : HalfPlane) (l : StraightLine),
    sf = sorry }

/-- Theorem stating that the incorrect definition is inconsistent with the 3D nature of dihedral angles --/
theorem dihedral_angle_definition_inconsistency :
  ¬(IncorrectDihedralAngle = CorrectDihedralAngle) :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_definition_inconsistency_l2759_275961


namespace NUMINAMATH_CALUDE_amy_treasures_first_level_l2759_275940

def points_per_treasure : ℕ := 4
def treasures_second_level : ℕ := 2
def total_score : ℕ := 32

def treasures_first_level : ℕ := (total_score - points_per_treasure * treasures_second_level) / points_per_treasure

theorem amy_treasures_first_level : treasures_first_level = 6 := by
  sorry

end NUMINAMATH_CALUDE_amy_treasures_first_level_l2759_275940


namespace NUMINAMATH_CALUDE_cousin_name_probability_l2759_275929

theorem cousin_name_probability :
  let total_cards : ℕ := 12
  let adrian_cards : ℕ := 7
  let bella_cards : ℕ := 5
  let prob_one_from_each : ℚ := 
    (adrian_cards / total_cards) * (bella_cards / (total_cards - 1)) +
    (bella_cards / total_cards) * (adrian_cards / (total_cards - 1))
  prob_one_from_each = 35 / 66 := by
sorry

end NUMINAMATH_CALUDE_cousin_name_probability_l2759_275929


namespace NUMINAMATH_CALUDE_polynomial_sum_equality_l2759_275910

-- Define the two polynomials
def p1 (x : ℝ) : ℝ := 3*x^4 + 2*x^3 - 5*x^2 + 9*x - 2
def p2 (x : ℝ) : ℝ := -3*x^4 - 5*x^3 + 7*x^2 - 9*x + 4

-- Define the sum of the polynomials
def sum_poly (x : ℝ) : ℝ := p1 x + p2 x

-- Define the result polynomial
def result (x : ℝ) : ℝ := -3*x^3 + 2*x^2 + 2

-- Theorem statement
theorem polynomial_sum_equality : 
  ∀ x : ℝ, sum_poly x = result x := by sorry

end NUMINAMATH_CALUDE_polynomial_sum_equality_l2759_275910


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_cost_of_dozen_pens_is_720_l2759_275987

/-- The cost of one dozen pens given the cost of 3 pens and 5 pencils and the ratio of pen to pencil cost -/
theorem cost_of_dozen_pens (total_cost : ℕ) (ratio_pen_pencil : ℕ) : ℕ :=
  let pen_cost := ratio_pen_pencil * (total_cost / (3 * ratio_pen_pencil + 5))
  12 * pen_cost

/-- Proof that the cost of one dozen pens is 720 given the conditions -/
theorem cost_of_dozen_pens_is_720 :
  cost_of_dozen_pens 240 5 = 720 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_cost_of_dozen_pens_is_720_l2759_275987


namespace NUMINAMATH_CALUDE_lecture_orderings_l2759_275983

/-- Represents the number of lecturers --/
def n : ℕ := 7

/-- Represents the number of lecturers with specific ordering constraints --/
def k : ℕ := 3

/-- Calculates the number of valid orderings for n lecturers with k lecturers having specific ordering constraints --/
def validOrderings (n k : ℕ) : ℕ :=
  Nat.factorial (n - k + 1)

/-- Theorem stating that the number of valid orderings for 7 lecturers with 3 having specific constraints is 120 --/
theorem lecture_orderings : validOrderings n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_lecture_orderings_l2759_275983


namespace NUMINAMATH_CALUDE_probability_of_white_and_black_l2759_275904

/-- The number of white balls in the bag -/
def num_white : ℕ := 2

/-- The number of black balls in the bag -/
def num_black : ℕ := 2

/-- The total number of balls in the bag -/
def total_balls : ℕ := num_white + num_black

/-- The number of balls drawn -/
def drawn : ℕ := 2

/-- The probability of drawing one white ball and one black ball -/
def prob_white_and_black : ℚ := 2 / 3

theorem probability_of_white_and_black :
  (num_white * num_black : ℚ) / (total_balls.choose drawn) = prob_white_and_black := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_and_black_l2759_275904


namespace NUMINAMATH_CALUDE_set_intersection_empty_implies_a_range_l2759_275974

theorem set_intersection_empty_implies_a_range (a : ℝ) : 
  let A := {x : ℝ | a - 1 < x ∧ x < 2*a + 1}
  let B := {x : ℝ | 0 < x ∧ x < 1}
  (A ∩ B = ∅) → (a ≤ -1/2 ∨ a ≥ 2) := by
sorry

end NUMINAMATH_CALUDE_set_intersection_empty_implies_a_range_l2759_275974


namespace NUMINAMATH_CALUDE_time_to_fill_leaking_tank_l2759_275917

/-- Time to fill a leaking tank -/
theorem time_to_fill_leaking_tank 
  (pump_fill_time : ℝ) 
  (leak_empty_time : ℝ) 
  (h1 : pump_fill_time = 6) 
  (h2 : leak_empty_time = 12) : 
  (pump_fill_time * leak_empty_time) / (leak_empty_time - pump_fill_time) = 12 := by
  sorry

#check time_to_fill_leaking_tank

end NUMINAMATH_CALUDE_time_to_fill_leaking_tank_l2759_275917


namespace NUMINAMATH_CALUDE_max_value_of_a_l2759_275980

theorem max_value_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = |x - 5/2| + |x - a|) →
  (∀ x, f x ≥ a) →
  ∃ a_max : ℝ, a_max = 5/4 ∧ ∀ a' : ℝ, (∀ x, |x - 5/2| + |x - a'| ≥ a') → a' ≤ a_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l2759_275980


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l2759_275998

theorem smallest_undefined_value (x : ℝ) :
  let f := fun x => (x - 3) / (6 * x^2 - 47 * x + 7)
  let smallest_x := (47 - Real.sqrt 2041) / 12
  (∀ y < smallest_x, f y ≠ 0⁻¹) ∧
  (f smallest_x = 0⁻¹) :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l2759_275998


namespace NUMINAMATH_CALUDE_tangent_line_to_cubic_curve_l2759_275939

theorem tangent_line_to_cubic_curve (a : ℝ) :
  (∃ x y : ℝ, y = 3 * x + 1 ∧ y = x^3 - a ∧ 3 * x^2 = 3) →
  (a = -3 ∨ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_cubic_curve_l2759_275939


namespace NUMINAMATH_CALUDE_simplify_expression_l2759_275944

theorem simplify_expression (x y : ℝ) : (3 * x^2 * y)^3 + (4 * x * y) * (y^4) = 27 * x^6 * y^3 + 4 * x * y^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2759_275944


namespace NUMINAMATH_CALUDE_x_fifth_minus_five_x_l2759_275964

theorem x_fifth_minus_five_x (x : ℝ) : x = 4 → x^5 - 5*x = 1004 := by
  sorry

end NUMINAMATH_CALUDE_x_fifth_minus_five_x_l2759_275964


namespace NUMINAMATH_CALUDE_faye_pencils_l2759_275963

/-- The number of rows of pencils -/
def num_rows : ℕ := 14

/-- The number of pencils in each row -/
def pencils_per_row : ℕ := 11

/-- The total number of pencils -/
def total_pencils : ℕ := num_rows * pencils_per_row

theorem faye_pencils : total_pencils = 154 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencils_l2759_275963


namespace NUMINAMATH_CALUDE_committee_formation_l2759_275909

theorem committee_formation (n m k : ℕ) (h1 : n = 12) (h2 : m = 5) (h3 : k = 4) :
  Nat.choose n m = 792 ∧ Nat.choose (n - k) m = 56 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_l2759_275909


namespace NUMINAMATH_CALUDE_exam_mistakes_l2759_275943

theorem exam_mistakes (bryan_score jen_score sammy_score total_points : ℕ) : 
  bryan_score = 20 →
  jen_score = bryan_score + 10 →
  sammy_score = jen_score - 2 →
  total_points = 35 →
  total_points - sammy_score = 7 :=
by sorry

end NUMINAMATH_CALUDE_exam_mistakes_l2759_275943


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_15_9_l2759_275938

theorem gcd_lcm_sum_15_9 : 
  Nat.gcd 15 9 + 2 * Nat.lcm 15 9 = 93 := by sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_15_9_l2759_275938


namespace NUMINAMATH_CALUDE_green_tea_profit_maximization_l2759_275926

/-- The profit function for a green tea company -/
def profit (x : ℝ) : ℝ := -2 * x^2 + 340 * x - 12000

/-- The selling price that maximizes profit -/
def max_profit_price : ℝ := 85

theorem green_tea_profit_maximization :
  /- The profit function is correct -/
  (∀ x : ℝ, profit x = -2 * x^2 + 340 * x - 12000) ∧
  /- The maximum profit occurs at x = 85 -/
  (∀ x : ℝ, profit x ≤ profit max_profit_price) := by
  sorry


end NUMINAMATH_CALUDE_green_tea_profit_maximization_l2759_275926


namespace NUMINAMATH_CALUDE_kombucha_bottle_cost_l2759_275927

/-- Represents the cost of a bottle of kombucha -/
def bottle_cost : ℝ := sorry

/-- Represents the number of bottles Henry drinks per month -/
def bottles_per_month : ℕ := 15

/-- Represents the cash refund per bottle in dollars -/
def refund_per_bottle : ℝ := 0.1

/-- Represents the number of months in a year -/
def months_in_year : ℕ := 12

/-- Represents the number of bottles that can be bought with the yearly refund -/
def bottles_bought_with_refund : ℕ := 6

theorem kombucha_bottle_cost :
  bottle_cost = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_kombucha_bottle_cost_l2759_275927


namespace NUMINAMATH_CALUDE_volume_of_solid_is_62pi_over_3_l2759_275905

/-- The region S in the coordinate plane -/
def region_S : Set (ℝ × ℝ) :=
  {p | p.2 ≤ p.1 + 2 ∧ p.2 ≤ -p.1 + 6 ∧ p.2 ≤ 4}

/-- The volume of the solid formed by revolving region S around the y-axis -/
noncomputable def volume_of_solid : ℝ := sorry

/-- Theorem stating that the volume of the solid is 62π/3 -/
theorem volume_of_solid_is_62pi_over_3 :
  volume_of_solid = 62 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_volume_of_solid_is_62pi_over_3_l2759_275905


namespace NUMINAMATH_CALUDE_largest_digit_sum_is_8_l2759_275984

/-- Represents a three-digit decimal as a fraction 1/y where y is an integer between 1 and 16 -/
def IsValidFraction (a b c : ℕ) (y : ℕ) : Prop :=
  a < 10 ∧ b < 10 ∧ c < 10 ∧ 
  1 < y ∧ y ≤ 16 ∧
  (100 * a + 10 * b + c : ℚ) / 1000 = 1 / y

/-- The sum of digits a, b, and c is at most 8 given the conditions -/
theorem largest_digit_sum_is_8 :
  ∀ a b c y : ℕ, IsValidFraction a b c y → a + b + c ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_sum_is_8_l2759_275984


namespace NUMINAMATH_CALUDE_ordering_of_trig_and_log_expressions_l2759_275921

theorem ordering_of_trig_and_log_expressions :
  let a := Real.sin (Real.cos 2)
  let b := Real.cos (Real.cos 2)
  let c := Real.log (Real.cos 1)
  c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ordering_of_trig_and_log_expressions_l2759_275921


namespace NUMINAMATH_CALUDE_angle_A_value_l2759_275937

noncomputable section

-- Define the triangle ABC
variable (A B C : Real)  -- Angles
variable (a b c : Real)  -- Side lengths

-- Define the conditions
axiom triangle : A + B + C = Real.pi  -- Sum of angles in a triangle
axiom side_a : a = Real.sqrt 3
axiom side_b : b = Real.sqrt 2
axiom angle_B : B = Real.pi / 4  -- 45° in radians

-- State the theorem
theorem angle_A_value : 
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 := by sorry

end NUMINAMATH_CALUDE_angle_A_value_l2759_275937


namespace NUMINAMATH_CALUDE_contractor_male_wage_l2759_275932

/-- Represents the daily wage structure and worker composition of a building contractor --/
structure ContractorData where
  male_workers : ℕ
  female_workers : ℕ
  child_workers : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the daily wage for male workers given the contractor's data --/
def male_wage (data : ContractorData) : ℕ :=
  let total_workers := data.male_workers + data.female_workers + data.child_workers
  let total_wage := total_workers * data.average_wage
  let female_total := data.female_workers * data.female_wage
  let child_total := data.child_workers * data.child_wage
  (total_wage - female_total - child_total) / data.male_workers

/-- Theorem stating that for the given contractor data, the male wage is 25 --/
theorem contractor_male_wage :
  male_wage {
    male_workers := 20,
    female_workers := 15,
    child_workers := 5,
    female_wage := 20,
    child_wage := 8,
    average_wage := 21
  } = 25 := by
  sorry


end NUMINAMATH_CALUDE_contractor_male_wage_l2759_275932


namespace NUMINAMATH_CALUDE_line_slope_through_points_l2759_275923

/-- The slope of a line passing through points (1, 0) and (2, √3) is √3. -/
theorem line_slope_through_points : 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (2, Real.sqrt 3)
  (B.2 - A.2) / (B.1 - A.1) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_through_points_l2759_275923


namespace NUMINAMATH_CALUDE_larger_number_l2759_275911

theorem larger_number (x y : ℝ) (h1 : x - y = 5) (h2 : x + y = 20) : max x y = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l2759_275911


namespace NUMINAMATH_CALUDE_violet_balloons_lost_l2759_275969

theorem violet_balloons_lost (initial_violet : ℕ) (remaining_violet : ℕ) 
  (h1 : initial_violet = 7) 
  (h2 : remaining_violet = 4) : 
  initial_violet - remaining_violet = 3 := by
sorry

end NUMINAMATH_CALUDE_violet_balloons_lost_l2759_275969


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l2759_275903

theorem power_fraction_simplification :
  (3^5 * 4^5) / 6^5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l2759_275903


namespace NUMINAMATH_CALUDE_simple_interest_sum_l2759_275978

/-- Given a sum of money with simple interest, prove that it equals 1700 --/
theorem simple_interest_sum (P r : ℝ) 
  (h1 : P * (1 + r) = 1717)
  (h2 : P * (1 + 2 * r) = 1734) :
  P = 1700 := by sorry

end NUMINAMATH_CALUDE_simple_interest_sum_l2759_275978
