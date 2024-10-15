import Mathlib

namespace NUMINAMATH_CALUDE_income_relationship_l1982_198269

theorem income_relationship (juan tim mart : ℝ) 
  (h1 : mart = 1.4 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mart = 0.84 * juan := by
sorry

end NUMINAMATH_CALUDE_income_relationship_l1982_198269


namespace NUMINAMATH_CALUDE_percentage_problem_l1982_198278

theorem percentage_problem (x : ℝ) (h : 0.05 * x = 8) : 0.25 * x = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1982_198278


namespace NUMINAMATH_CALUDE_chocolate_bar_cost_l1982_198202

/-- The cost of one chocolate bar given the conditions of the problem -/
theorem chocolate_bar_cost : 
  ∀ (scouts : ℕ) (smores_per_scout : ℕ) (smores_per_bar : ℕ) (total_cost : ℚ),
  scouts = 15 →
  smores_per_scout = 2 →
  smores_per_bar = 3 →
  total_cost = 15 →
  (total_cost / (scouts * smores_per_scout / smores_per_bar : ℚ)) = 3/2 := by
sorry

end NUMINAMATH_CALUDE_chocolate_bar_cost_l1982_198202


namespace NUMINAMATH_CALUDE_sum_base7_equals_650_l1982_198261

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 7 --/
def base10ToBase7 (n : ℕ) : ℕ := sorry

/-- The sum of three numbers in base 7 --/
def sumBase7 (a b c : ℕ) : ℕ :=
  base10ToBase7 (base7ToBase10 a + base7ToBase10 b + base7ToBase10 c)

theorem sum_base7_equals_650 :
  sumBase7 543 65 6 = 650 := by sorry

end NUMINAMATH_CALUDE_sum_base7_equals_650_l1982_198261


namespace NUMINAMATH_CALUDE_inscribed_square_and_circle_dimensions_l1982_198201

-- Define the right triangle DEF
def triangle_DEF (DE EF DF : ℝ) : Prop :=
  DE = 5 ∧ EF = 12 ∧ DF = 13 ∧ DE ^ 2 + EF ^ 2 = DF ^ 2

-- Define the inscribed square PQRS
def inscribed_square (s : ℝ) (DE EF DF : ℝ) : Prop :=
  triangle_DEF DE EF DF ∧
  ∃ (P Q R S : ℝ × ℝ),
    -- P and Q on DF, R on DE, S on EF
    (P.1 + Q.1 = DF) ∧ (R.2 = DE) ∧ (S.1 = EF) ∧
    -- PQRS is a square with side length s
    (Q.1 - P.1 = s) ∧ (R.2 - Q.2 = s) ∧ (S.1 - R.1 = s) ∧ (P.2 - S.2 = s)

-- Define the inscribed circle
def inscribed_circle (r : ℝ) (s : ℝ) : Prop :=
  r = s / 2

-- Theorem statement
theorem inscribed_square_and_circle_dimensions :
  ∀ (DE EF DF s r : ℝ),
    inscribed_square s DE EF DF →
    inscribed_circle r s →
    s = 780 / 169 ∧ r = 390 / 338 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_and_circle_dimensions_l1982_198201


namespace NUMINAMATH_CALUDE_scientific_notation_of_1_6_million_l1982_198239

theorem scientific_notation_of_1_6_million :
  ∃ (a : ℝ) (n : ℤ), 1600000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.6 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1_6_million_l1982_198239


namespace NUMINAMATH_CALUDE_always_positive_l1982_198281

-- Define a monotonically increasing odd function
def MonoIncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h_f : MonoIncreasingOddFunction f)
  (h_a : ArithmeticSequence a)
  (h_a3 : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_always_positive_l1982_198281


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1982_198289

/-- 
Given an isosceles right triangle that, when folded twice along the altitude to its hypotenuse, 
results in a smaller isosceles right triangle with leg length 2 cm, 
prove that the area of the original triangle is 4 square centimeters.
-/
theorem isosceles_right_triangle_area (a : ℝ) (h1 : a > 0) : 
  (a / Real.sqrt 2 = 2) → (1 / 2 * a * a = 4) := by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l1982_198289


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_opposing_l1982_198231

/-- A bag containing two red balls and two black balls -/
structure Bag :=
  (red_balls : ℕ := 2)
  (black_balls : ℕ := 2)

/-- The event of drawing exactly one black ball -/
def exactly_one_black (bag : Bag) : Set (Fin 2 → Bool) :=
  {draw | (draw 0 = true ∧ draw 1 = false) ∨ (draw 0 = false ∧ draw 1 = true)}

/-- The event of drawing exactly two black balls -/
def exactly_two_black (bag : Bag) : Set (Fin 2 → Bool) :=
  {draw | draw 0 = true ∧ draw 1 = true}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutually_exclusive (E F : Set (Fin 2 → Bool)) : Prop :=
  E ∩ F = ∅

/-- Two events are opposing if their union is the entire sample space -/
def opposing (E F : Set (Fin 2 → Bool)) : Prop :=
  E ∪ F = Set.univ

theorem mutually_exclusive_not_opposing (bag : Bag) :
  mutually_exclusive (exactly_one_black bag) (exactly_two_black bag) ∧
  ¬opposing (exactly_one_black bag) (exactly_two_black bag) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_opposing_l1982_198231


namespace NUMINAMATH_CALUDE_net_sales_effect_l1982_198243

/-- Calculates the net effect on sales after two consecutive price reductions and sales increases -/
theorem net_sales_effect (initial_price_reduction : ℝ) 
                         (initial_sales_increase : ℝ)
                         (second_price_reduction : ℝ)
                         (second_sales_increase : ℝ) :
  initial_price_reduction = 0.20 →
  initial_sales_increase = 0.80 →
  second_price_reduction = 0.15 →
  second_sales_increase = 0.60 →
  let first_quarter_sales := 1 + initial_sales_increase
  let second_quarter_sales := first_quarter_sales * (1 + second_sales_increase)
  let net_effect := (second_quarter_sales - 1) * 100
  net_effect = 188 := by
sorry

end NUMINAMATH_CALUDE_net_sales_effect_l1982_198243


namespace NUMINAMATH_CALUDE_max_value_of_f_l1982_198284

def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 + 2 * k * x + 1

theorem max_value_of_f (k : ℝ) : 
  (∀ x ∈ Set.Icc (-3 : ℝ) 2, f k x ≤ 4) ∧ 
  (∃ x ∈ Set.Icc (-3 : ℝ) 2, f k x = 4) →
  k = -3 ∨ k = 3/8 := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1982_198284


namespace NUMINAMATH_CALUDE_units_digit_of_product_of_first_four_composites_l1982_198212

def first_four_composite_numbers : List Nat := [4, 6, 8, 9]

def product_of_list (l : List Nat) : Nat :=
  l.foldl (·*·) 1

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_product_of_first_four_composites :
  units_digit (product_of_list first_four_composite_numbers) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_of_first_four_composites_l1982_198212


namespace NUMINAMATH_CALUDE_length_of_cd_l1982_198223

/-- Represents a point that divides a line segment in a given ratio -/
structure DividingPoint where
  ratio_left : ℚ
  ratio_right : ℚ

/-- Represents a line segment divided by two points -/
structure DividedSegment where
  length : ℝ
  point1 : DividingPoint
  point2 : DividingPoint
  distance_between_points : ℝ

/-- Theorem stating the length of CD given the conditions -/
theorem length_of_cd (cd : DividedSegment) : 
  cd.point1.ratio_left = 3 ∧ 
  cd.point1.ratio_right = 5 ∧ 
  cd.point2.ratio_left = 4 ∧ 
  cd.point2.ratio_right = 7 ∧ 
  cd.distance_between_points = 3 → 
  cd.length = 264 := by
  sorry

end NUMINAMATH_CALUDE_length_of_cd_l1982_198223


namespace NUMINAMATH_CALUDE_quadratic_function_property_quadratic_function_property_independent_of_b_l1982_198216

theorem quadratic_function_property (c d h b : ℝ) : 
  let f (x : ℝ) := c * x^2
  let x₁ := b - d - h
  let x₂ := b - d
  let x₃ := b + d
  let x₄ := b + d + h
  let y₁ := f x₁
  let y₂ := f x₂
  let y₃ := f x₃
  let y₄ := f x₄
  (y₁ + y₄) - (y₂ + y₃) = 2 * c * h * (2 * d + h) :=
by sorry

theorem quadratic_function_property_independent_of_b (c d h : ℝ) :
  ∀ b₁ b₂ : ℝ, 
  let f (x : ℝ) := c * x^2
  let x₁ (b : ℝ) := b - d - h
  let x₂ (b : ℝ) := b - d
  let x₃ (b : ℝ) := b + d
  let x₄ (b : ℝ) := b + d + h
  let y₁ (b : ℝ) := f (x₁ b)
  let y₂ (b : ℝ) := f (x₂ b)
  let y₃ (b : ℝ) := f (x₃ b)
  let y₄ (b : ℝ) := f (x₄ b)
  (y₁ b₁ + y₄ b₁) - (y₂ b₁ + y₃ b₁) = (y₁ b₂ + y₄ b₂) - (y₂ b₂ + y₃ b₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_property_quadratic_function_property_independent_of_b_l1982_198216


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1982_198236

def A : Set (ℝ × ℝ) := {p | p.2 = -p.1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1^2 - 2}

theorem intersection_of_A_and_B :
  A ∩ B = {(-2, 2), (1, -1)} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1982_198236


namespace NUMINAMATH_CALUDE_sum_of_five_unit_fractions_l1982_198265

theorem sum_of_five_unit_fractions :
  ∃ (a b c d e : ℕ+), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                       b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                       c ≠ d ∧ c ≠ e ∧ 
                       d ≠ e ∧
                       (1 : ℚ) = 1 / a + 1 / b + 1 / c + 1 / d + 1 / e :=
sorry

end NUMINAMATH_CALUDE_sum_of_five_unit_fractions_l1982_198265


namespace NUMINAMATH_CALUDE_line_perp_parallel_planes_l1982_198280

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- A line is perpendicular to a plane if they intersect at right angles -/
def line_perp_plane (m : Line) (α : Plane) : Prop := sorry

/-- Two planes are parallel if they do not intersect -/
def planes_parallel (α β : Plane) : Prop := sorry

theorem line_perp_parallel_planes (α β : Plane) (m : Line) :
  different_planes α β →
  line_perp_plane m β →
  planes_parallel α β →
  line_perp_plane m α :=
sorry

end NUMINAMATH_CALUDE_line_perp_parallel_planes_l1982_198280


namespace NUMINAMATH_CALUDE_parents_age_when_mark_born_l1982_198287

/-- Given the ages of Mark and John, and their relation to their parents' age, 
    proves the age of the parents when Mark was born. -/
theorem parents_age_when_mark_born (mark_age john_age parents_age : ℕ) : 
  mark_age = 18 →
  john_age = mark_age - 10 →
  parents_age = 5 * john_age →
  parents_age - mark_age = 22 :=
by sorry

end NUMINAMATH_CALUDE_parents_age_when_mark_born_l1982_198287


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1982_198275

/-- A rectangle with a circle inscribed such that the circle is tangent to three sides of the rectangle and its center lies on a diagonal of the rectangle. -/
structure InscribedCircleRectangle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The width of the rectangle -/
  w : ℝ
  /-- The height of the rectangle -/
  h : ℝ
  /-- The circle is tangent to three sides of the rectangle -/
  tangent_to_sides : w = 2 * r ∧ h = r
  /-- The center of the circle lies on a diagonal of the rectangle -/
  center_on_diagonal : True

/-- The area of a rectangle with an inscribed circle as described is equal to 2r^2 -/
theorem inscribed_circle_rectangle_area (rect : InscribedCircleRectangle) :
  rect.w * rect.h = 2 * rect.r^2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l1982_198275


namespace NUMINAMATH_CALUDE_total_highlighters_l1982_198288

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 6) (h2 : yellow = 2) (h3 : blue = 4) :
  pink + yellow + blue = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_highlighters_l1982_198288


namespace NUMINAMATH_CALUDE_parallel_vectors_iff_y_eq_3_l1982_198264

/-- Two vectors in ℝ² are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

/-- The problem statement -/
theorem parallel_vectors_iff_y_eq_3 :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (6, y)
  parallel a b ↔ y = 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_iff_y_eq_3_l1982_198264


namespace NUMINAMATH_CALUDE_ratio_problem_l1982_198285

theorem ratio_problem (x y z w : ℝ) 
  (h1 : 0.1 * x = 0.2 * y) 
  (h2 : 0.3 * y = 0.4 * z) 
  (h3 : 0.5 * z = 0.6 * w) : 
  ∃ (k : ℝ), k > 0 ∧ x = 8 * k ∧ y = 4 * k ∧ z = 3 * k ∧ w = 2.5 * k :=
sorry

end NUMINAMATH_CALUDE_ratio_problem_l1982_198285


namespace NUMINAMATH_CALUDE_sum_of_squares_l1982_198232

theorem sum_of_squares : 1000 ^ 2 + 1001 ^ 2 + 1002 ^ 2 + 1003 ^ 2 + 1004 ^ 2 = 5020030 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l1982_198232


namespace NUMINAMATH_CALUDE_library_visitors_theorem_l1982_198240

/-- Calculates the total number of visitors to a library in a week given specific conditions --/
theorem library_visitors_theorem (monday_visitors : ℕ) 
  (h1 : monday_visitors = 50)
  (h2 : ∃ tuesday_visitors : ℕ, tuesday_visitors = 2 * monday_visitors)
  (h3 : ∃ wednesday_visitors : ℕ, wednesday_visitors = 2 * monday_visitors)
  (h4 : ∃ thursday_visitors : ℕ, thursday_visitors = 3 * (2 * monday_visitors))
  (h5 : ∃ weekend_visitors : ℕ, weekend_visitors = 3 * 20) :
  monday_visitors + 
  (2 * monday_visitors) + 
  (2 * monday_visitors) + 
  (3 * (2 * monday_visitors)) + 
  (3 * 20) = 610 := by
    sorry

end NUMINAMATH_CALUDE_library_visitors_theorem_l1982_198240


namespace NUMINAMATH_CALUDE_max_value_theorem_l1982_198263

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 1) :
  2*a*b + 2*a*c*Real.sqrt 3 ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1982_198263


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l1982_198279

/-- The smallest integer b > 3 for which 34_b is a perfect square -/
theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 3 ∧ 
  (∀ (x : ℕ), x > 3 ∧ x < b → ¬∃ (y : ℕ), 3*x + 4 = y^2) ∧
  (∃ (y : ℕ), 3*b + 4 = y^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l1982_198279


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l1982_198209

/-- Calculates the distance traveled downstream by a boat given its speed in still water,
    the stream speed, and the time taken to travel downstream. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat with a speed of 16 km/hr in still water, traveling in a stream
    with a speed of 4 km/hr for 3 hours, will travel 60 km downstream. -/
theorem boat_downstream_distance :
  distance_downstream 16 4 3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_boat_downstream_distance_l1982_198209


namespace NUMINAMATH_CALUDE_f_properties_l1982_198245

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/a + 2/x

theorem f_properties (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 → f a x₁ < f a x₂) ∧
  (a < 0 → ∀ x : ℝ, x > 0 → f a x > 0) ∧
  (a > 0 → ∀ x : ℝ, 0 < x ∧ x < 2*a ↔ f a x > 0) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1982_198245


namespace NUMINAMATH_CALUDE_election_result_l1982_198228

theorem election_result (total_voters : ℝ) (rep_percent : ℝ) (dem_percent : ℝ) 
  (dem_x_vote_percent : ℝ) (x_win_margin : ℝ) :
  rep_percent / dem_percent = 3 / 2 →
  rep_percent + dem_percent = 100 →
  dem_x_vote_percent = 25 →
  x_win_margin = 16.000000000000014 →
  ∃ (rep_x_vote_percent : ℝ),
    rep_x_vote_percent * rep_percent + dem_x_vote_percent * dem_percent = 
    (100 + x_win_margin) / 2 ∧
    rep_x_vote_percent = 80 :=
by sorry

end NUMINAMATH_CALUDE_election_result_l1982_198228


namespace NUMINAMATH_CALUDE_bucket_fill_time_l1982_198205

/-- Given that it takes 2 minutes to fill two-thirds of a bucket,
    prove that it takes 3 minutes to fill the entire bucket. -/
theorem bucket_fill_time :
  let partial_time : ℚ := 2
  let partial_fill : ℚ := 2/3
  let full_time : ℚ := 3
  (partial_fill * full_time = partial_time) → full_time = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_bucket_fill_time_l1982_198205


namespace NUMINAMATH_CALUDE_riddle_guessing_probabilities_l1982_198203

-- Define the probabilities of A and B guessing correctly
def prob_A_correct : ℚ := 5/6
def prob_B_correct : ℚ := 3/5

-- Define the probability of A winning in one activity
def prob_A_wins_one : ℚ := prob_A_correct * (1 - prob_B_correct)

-- Define the probability of A winning at least 2 out of 3 activities
def prob_A_wins_two_out_of_three : ℚ :=
  3 * (prob_A_wins_one^2 * (1 - prob_A_wins_one)) + prob_A_wins_one^3

-- State the theorem
theorem riddle_guessing_probabilities :
  prob_A_wins_one = 1/3 ∧ prob_A_wins_two_out_of_three = 7/27 := by
  sorry

end NUMINAMATH_CALUDE_riddle_guessing_probabilities_l1982_198203


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1982_198221

theorem complex_equation_solution (Z : ℂ) : Z = Complex.I * (2 + Z) → Z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1982_198221


namespace NUMINAMATH_CALUDE_unique_root_sum_l1982_198230

-- Define the function f(x) = x^3 - x + 1
def f (x : ℝ) : ℝ := x^3 - x + 1

-- Theorem statement
theorem unique_root_sum (a b : ℤ) : 
  (∃! x : ℝ, a < x ∧ x < b ∧ f x = 0) →  -- Exactly one root in (a, b)
  (b - a = 1) →                          -- b - a = 1
  (a + b = -3) :=                        -- Conclusion: a + b = -3
by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_unique_root_sum_l1982_198230


namespace NUMINAMATH_CALUDE_cube_root_8000_l1982_198292

theorem cube_root_8000 (c d : ℕ+) : 
  (c : ℝ) * (d : ℝ)^(1/3 : ℝ) = 20 → 
  (∀ (c' d' : ℕ+), (c' : ℝ) * (d' : ℝ)^(1/3 : ℝ) = 20 → d ≤ d') → 
  c + d = 21 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_8000_l1982_198292


namespace NUMINAMATH_CALUDE_solve_for_a_l1982_198296

theorem solve_for_a (x a : ℝ) : 2 * x + a - 8 = 0 → x = 2 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1982_198296


namespace NUMINAMATH_CALUDE_t_integer_characterization_t_irreducible_characterization_l1982_198262

def t (n : ℤ) : ℚ := (5 * n + 9) / (n - 3)

def is_integer_t (n : ℤ) : Prop := ∃ (k : ℤ), t n = k

def is_irreducible_t (n : ℤ) : Prop :=
  ∃ (a b : ℤ), t n = a / b ∧ Int.gcd a b = 1

theorem t_integer_characterization (n : ℤ) (h : n > 3) :
  is_integer_t n ↔ n ∈ ({4, 5, 6, 7, 9, 11, 15, 27} : Set ℤ) :=
sorry

theorem t_irreducible_characterization (n : ℤ) (h : n > 3) :
  is_irreducible_t n ↔ (∃ (k : ℤ), k > 0 ∧ (n = 6 * k + 1 ∨ n = 6 * k + 5)) :=
sorry

end NUMINAMATH_CALUDE_t_integer_characterization_t_irreducible_characterization_l1982_198262


namespace NUMINAMATH_CALUDE_divisible_by_120_l1982_198271

theorem divisible_by_120 (n : ℕ) : ∃ k : ℤ, n * (n^2 - 1) * (n^2 - 5*n + 26) = 120 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_120_l1982_198271


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1982_198251

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_eq_3 : a + b + c = 3) : 
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l1982_198251


namespace NUMINAMATH_CALUDE_chipped_marbles_bag_l1982_198214

/-- Represents the number of marbles in each bag -/
def bags : List Nat := [15, 18, 22, 24, 30]

/-- Represents the total number of marbles -/
def total : Nat := bags.sum

/-- Predicate to check if a list of two numbers from the bags list sums to a given value -/
def hasTwoSum (s : Nat) : Prop := ∃ (a b : Nat), a ∈ bags ∧ b ∈ bags ∧ a ≠ b ∧ a + b = s

/-- The main theorem stating that the bag with chipped marbles contains 24 marbles -/
theorem chipped_marbles_bag : 
  ∃ (jane george : Nat), 
    jane ∈ bags ∧ 
    george ∈ bags ∧ 
    jane ≠ george ∧
    hasTwoSum jane ∧ 
    hasTwoSum george ∧ 
    jane = 3 * george ∧ 
    total - jane - george = 24 := by
  sorry

end NUMINAMATH_CALUDE_chipped_marbles_bag_l1982_198214


namespace NUMINAMATH_CALUDE_subtract_negative_one_three_l1982_198211

theorem subtract_negative_one_three : -1 - 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_one_three_l1982_198211


namespace NUMINAMATH_CALUDE_q_necessary_not_sufficient_l1982_198277

/-- A function f is monotonically increasing on an interval if for any two points x and y in that interval, x < y implies f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The function f(x) = x³ + 2x² + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + m*x + 1

/-- The statement p: f(x) is monotonically increasing in (-∞, +∞) -/
def p (m : ℝ) : Prop := MonotonicallyIncreasing (f m)

/-- The statement q: m > 4/3 -/
def q (m : ℝ) : Prop := m > 4/3

/-- Theorem stating that q is a necessary but not sufficient condition for p -/
theorem q_necessary_not_sufficient :
  (∀ m : ℝ, p m → q m) ∧ (∃ m : ℝ, q m ∧ ¬(p m)) := by sorry

end NUMINAMATH_CALUDE_q_necessary_not_sufficient_l1982_198277


namespace NUMINAMATH_CALUDE_function_property_l1982_198210

theorem function_property (f : ℤ → ℝ) 
  (h1 : ∀ x y : ℤ, f x * f y = f (x + y) + f (x - y))
  (h2 : f 1 = 5/2) :
  ∀ x : ℤ, f x = (5/2) ^ x := by sorry

end NUMINAMATH_CALUDE_function_property_l1982_198210


namespace NUMINAMATH_CALUDE_line_equation_proof_l1982_198293

-- Define the circle P
def circle_P (x y : ℝ) : Prop := x^2 + y^2 - 4*y = 0

-- Define the parabola S
def parabola_S (x y : ℝ) : Prop := y = x^2 / 8

-- Define a line passing through a point
def line_through_point (k m x y : ℝ) : Prop := y = k*x + m

-- Define the center of the circle
def circle_center : ℝ × ℝ := (0, 2)

-- Define the property of four points being in arithmetic sequence
def arithmetic_sequence (a b c d : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  let d2 := Real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2)
  let d3 := Real.sqrt ((c.1 - d.1)^2 + (c.2 - d.2)^2)
  d2 - d1 = d3 - d2

theorem line_equation_proof :
  ∀ (k m : ℝ) (a b c d : ℝ × ℝ),
    (∀ x y, line_through_point k m x y → (circle_P x y ∨ parabola_S x y)) →
    line_through_point k m circle_center.1 circle_center.2 →
    arithmetic_sequence a b c d →
    (k = -Real.sqrt 2 / 2 ∨ k = Real.sqrt 2 / 2) ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_proof_l1982_198293


namespace NUMINAMATH_CALUDE_general_term_formula_correct_l1982_198229

/-- Arithmetic sequence with first term 3 and common difference 2 -/
def arithmeticSequence (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- General term formula -/
def generalTerm (n : ℕ) : ℕ := 2 * n + 1

/-- Theorem stating that the general term formula is correct for the given arithmetic sequence -/
theorem general_term_formula_correct :
  ∀ n : ℕ, n > 0 → arithmeticSequence n = generalTerm n := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_correct_l1982_198229


namespace NUMINAMATH_CALUDE_two_year_growth_l1982_198282

/-- Calculates the final value after compound growth --/
def compound_growth (initial_value : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
  initial_value * (1 + growth_rate) ^ years

/-- Theorem: After two years of 1/8 annual growth, 32000 becomes 40500 --/
theorem two_year_growth :
  compound_growth 32000 (1/8) 2 = 40500 := by
  sorry

end NUMINAMATH_CALUDE_two_year_growth_l1982_198282


namespace NUMINAMATH_CALUDE_stream_speed_l1982_198217

theorem stream_speed (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  boat_speed = 30 →
  downstream_distance = 80 →
  upstream_distance = 40 →
  (downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) →
  x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1982_198217


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_A_l1982_198207

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define set A
def A : Set Nat := {2, 4, 5}

-- Define set B
def B : Set Nat := {2, 7}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {2, 4, 5, 7} := by sorry

-- Theorem for complement of A with respect to U
theorem complement_of_A : (U \ A) = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_A_l1982_198207


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1982_198268

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 24) : 
  max x (max (x + 1) (x + 2)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1982_198268


namespace NUMINAMATH_CALUDE_sum_inequality_l1982_198220

theorem sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (h_prod : a * b * c * d = 1) :
  1 / Real.sqrt (1/2 + a + a*b + a*b*c) +
  1 / Real.sqrt (1/2 + b + b*c + b*c*d) +
  1 / Real.sqrt (1/2 + c + c*d + c*d*a) +
  1 / Real.sqrt (1/2 + d + d*a + d*a*b) ≥ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sum_inequality_l1982_198220


namespace NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l1982_198276

theorem smallest_sum_of_a_and_b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∃ x : ℝ, x^2 + 2*a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 3*b*x + 2*a = 0) :
  a + b ≥ 2 * Real.sqrt 2 + 4/3 * Real.sqrt (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_a_and_b_l1982_198276


namespace NUMINAMATH_CALUDE_cylindrical_bucket_height_l1982_198273

/-- The height of a cylindrical bucket given its radius and the dimensions of a conical heap formed when emptied -/
theorem cylindrical_bucket_height (r_cylinder r_cone h_cone : ℝ) (h_cylinder : ℝ) : 
  r_cylinder = 21 →
  r_cone = 63 →
  h_cone = 12 →
  r_cylinder^2 * h_cylinder = (1/3) * r_cone^2 * h_cone →
  h_cylinder = 36 := by
  sorry

end NUMINAMATH_CALUDE_cylindrical_bucket_height_l1982_198273


namespace NUMINAMATH_CALUDE_food_drive_cans_l1982_198238

theorem food_drive_cans (rachel jaydon mark : ℕ) : 
  jaydon = 2 * rachel + 5 →
  mark = 4 * jaydon →
  rachel + jaydon + mark = 135 →
  mark = 100 := by
sorry

end NUMINAMATH_CALUDE_food_drive_cans_l1982_198238


namespace NUMINAMATH_CALUDE_always_odd_l1982_198266

theorem always_odd (n : ℤ) : ∃ k : ℤ, n^2 + n + 5 = 2*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_always_odd_l1982_198266


namespace NUMINAMATH_CALUDE_magnitude_sum_vectors_l1982_198286

/-- Given two planar vectors a and b, prove that |a + 2b| = 2√2 -/
theorem magnitude_sum_vectors (a b : ℝ × ℝ) :
  (a.1 = 2 ∧ a.2 = 0) →  -- a = (2, 0)
  ‖b‖ = 1 →             -- |b| = 1
  a • b = 0 →           -- angle between a and b is 90°
  ‖a + 2 • b‖ = 2 * Real.sqrt 2 := by
sorry


end NUMINAMATH_CALUDE_magnitude_sum_vectors_l1982_198286


namespace NUMINAMATH_CALUDE_initial_pencils_count_l1982_198260

/-- The number of pencils Sally took out of the drawer -/
def pencils_taken : ℕ := 4

/-- The number of pencils left in the drawer after Sally took some out -/
def pencils_left : ℕ := 5

/-- The initial number of pencils in the drawer -/
def initial_pencils : ℕ := pencils_taken + pencils_left

theorem initial_pencils_count : initial_pencils = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_pencils_count_l1982_198260


namespace NUMINAMATH_CALUDE_xiao_ming_score_l1982_198244

/-- Calculates the comprehensive score based on individual scores and weights -/
def comprehensive_score (written : ℝ) (practical : ℝ) (publicity : ℝ) 
  (written_weight : ℝ) (practical_weight : ℝ) (publicity_weight : ℝ) : ℝ :=
  written * written_weight + practical * practical_weight + publicity * publicity_weight

/-- Theorem stating that Xiao Ming's comprehensive score is 97 -/
theorem xiao_ming_score : 
  comprehensive_score 96 98 96 0.3 0.5 0.2 = 97 := by
  sorry

#eval comprehensive_score 96 98 96 0.3 0.5 0.2

end NUMINAMATH_CALUDE_xiao_ming_score_l1982_198244


namespace NUMINAMATH_CALUDE_convention_center_distance_l1982_198274

/-- The distance from Elena's home to the convention center -/
def distance : ℝ := sorry

/-- Elena's initial speed in miles per hour -/
def initial_speed : ℝ := 45

/-- The increase in Elena's speed for the rest of the journey -/
def speed_increase : ℝ := 20

/-- The time Elena would be late if she continued at the initial speed -/
def late_time : ℝ := 0.75

/-- The time Elena arrives early after increasing her speed -/
def early_time : ℝ := 0.25

/-- The actual time needed to arrive on time -/
def actual_time : ℝ := sorry

theorem convention_center_distance :
  (distance = initial_speed * (actual_time + late_time)) ∧
  (distance - initial_speed = (initial_speed + speed_increase) * (actual_time - 1 - early_time)) ∧
  (distance = 191.25) := by sorry

end NUMINAMATH_CALUDE_convention_center_distance_l1982_198274


namespace NUMINAMATH_CALUDE_min_value_abc_l1982_198204

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_abc : a * b * c = 27) :
  a^2 + 6*a*b + 9*b^2 + 3*c^2 ≥ 126 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 27 ∧ 
    a₀^2 + 6*a₀*b₀ + 9*b₀^2 + 3*c₀^2 = 126 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l1982_198204


namespace NUMINAMATH_CALUDE_solution_system1_solution_system2_l1982_198254

-- Define the first system of equations
def system1 (x y : ℝ) : Prop :=
  x + y + 3 = 10 ∧ 4 * (x + y) - y = 25

-- Define the second system of equations
def system2 (x y : ℝ) : Prop :=
  (2 * y - 4 * x) / 3 + 2 * x = 4 ∧ y - 2 * x + 3 = 6

-- Theorem for the first system
theorem solution_system1 : ∃ x y : ℝ, system1 x y ∧ x = 4 ∧ y = 3 := by
  sorry

-- Theorem for the second system
theorem solution_system2 : ∃ x y : ℝ, system2 x y ∧ x = 1 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_solution_system1_solution_system2_l1982_198254


namespace NUMINAMATH_CALUDE_expression_value_l1982_198218

theorem expression_value (x y : ℤ) (hx : x = 3) (hy : y = 4) : 3 * x - 2 * y + 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1982_198218


namespace NUMINAMATH_CALUDE_unique_solution_is_zero_function_l1982_198297

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 * y) = f (x * y) + y * f (f x + y)

/-- The theorem stating that the only function satisfying the equation is the zero function -/
theorem unique_solution_is_zero_function
  (f : ℝ → ℝ) (h : SatisfiesFunctionalEquation f) :
  ∀ y : ℝ, f y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_zero_function_l1982_198297


namespace NUMINAMATH_CALUDE_peter_erasers_l1982_198298

theorem peter_erasers (initial_erasers : ℕ) (received_erasers : ℕ) : 
  initial_erasers = 8 → received_erasers = 3 → initial_erasers + received_erasers = 11 := by
  sorry

end NUMINAMATH_CALUDE_peter_erasers_l1982_198298


namespace NUMINAMATH_CALUDE_square_area_ratio_l1982_198222

theorem square_area_ratio (y : ℝ) (h : y > 0) :
  (y ^ 2) / ((3 * y) ^ 2) = 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1982_198222


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l1982_198247

/-- Given a right triangle with consecutive whole number leg lengths and hypotenuse 29,
    prove that the sum of the leg lengths is 41. -/
theorem right_triangle_leg_sum : 
  ∃ (a b : ℕ), 
    a + 1 = b ∧                   -- legs are consecutive whole numbers
    a^2 + b^2 = 29^2 ∧            -- Pythagorean theorem for hypotenuse 29
    a + b = 41 :=                 -- sum of leg lengths is 41
by sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l1982_198247


namespace NUMINAMATH_CALUDE_percent_relation_l1982_198283

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) :
  4 * b / a * 100 = 1000 / 3 := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1982_198283


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1982_198227

theorem partial_fraction_decomposition :
  ∃! (P Q R : ℝ), ∀ (x : ℝ), x ≠ 4 ∧ x ≠ 2 →
    3 * x^2 + 2 * x = (x - 4) * (x - 2)^2 * (P / (x - 4) + Q / (x - 2) + R / (x - 2)^2) ∧
    P = 14 ∧ Q = -11 ∧ R = -8 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1982_198227


namespace NUMINAMATH_CALUDE_area_inside_Q_outside_P_R_l1982_198215

-- Define the circles
def circle_P : Real := 1
def circle_Q : Real := 2
def circle_R : Real := 1

-- Define the centers of the circles
def center_P : ℝ × ℝ := (0, 0)
def center_R : ℝ × ℝ := (2, 0)
def center_Q : ℝ × ℝ := (0, 0)

-- Define the tangency conditions
def Q_R_tangent : Prop := 
  (center_Q.1 - center_R.1)^2 + (center_Q.2 - center_R.2)^2 = (circle_Q + circle_R)^2

def R_P_tangent : Prop :=
  (center_R.1 - center_P.1)^2 + (center_R.2 - center_P.2)^2 = (circle_R + circle_P)^2

-- Theorem statement
theorem area_inside_Q_outside_P_R : 
  Q_R_tangent → R_P_tangent → 
  (π * circle_Q^2) - (π * circle_P^2) - (π * circle_R^2) = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_area_inside_Q_outside_P_R_l1982_198215


namespace NUMINAMATH_CALUDE_library_visitors_theorem_l1982_198299

/-- The average number of visitors on non-Sunday days in a library -/
def average_visitors_non_sunday (sunday_avg : ℕ) (total_days : ℕ) (month_avg : ℕ) : ℚ :=
  let sundays := total_days / 7 + 1
  let other_days := total_days - sundays
  (total_days * month_avg - sundays * sunday_avg) / other_days

/-- Theorem stating the average number of visitors on non-Sunday days -/
theorem library_visitors_theorem :
  average_visitors_non_sunday 630 30 305 = 240 := by
  sorry

#eval average_visitors_non_sunday 630 30 305

end NUMINAMATH_CALUDE_library_visitors_theorem_l1982_198299


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l1982_198249

theorem triangle_max_perimeter (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  2 * Real.cos C + c = 2 * b →
  a + b + c ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l1982_198249


namespace NUMINAMATH_CALUDE_f_not_differentiable_at_zero_l1982_198250

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.sin (x * Real.sin (3 / x)) else 0

theorem f_not_differentiable_at_zero :
  ¬ DifferentiableAt ℝ f 0 := by sorry

end NUMINAMATH_CALUDE_f_not_differentiable_at_zero_l1982_198250


namespace NUMINAMATH_CALUDE_smallest_w_value_l1982_198233

theorem smallest_w_value : ∃ (w : ℕ+),
  (∀ (x : ℕ+), 
    (2^6 ∣ 2547 * x) ∧ 
    (3^5 ∣ 2547 * x) ∧ 
    (5^4 ∣ 2547 * x) ∧ 
    (7^3 ∣ 2547 * x) ∧ 
    (13^4 ∣ 2547 * x) → 
    w ≤ x) ∧
  (2^6 ∣ 2547 * w) ∧
  (3^5 ∣ 2547 * w) ∧
  (5^4 ∣ 2547 * w) ∧
  (7^3 ∣ 2547 * w) ∧
  (13^4 ∣ 2547 * w) ∧
  w = 1592010000 :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_value_l1982_198233


namespace NUMINAMATH_CALUDE_train_car_speed_ratio_l1982_198237

/-- Given a bus, a train, and a car with the following properties:
  * The speed of the bus is 3/4 of the speed of the train
  * The bus travels 480 km in 8 hours
  * The car travels 450 km in 6 hours
  Prove that the ratio of the speed of the train to the speed of the car is 16:15 -/
theorem train_car_speed_ratio : 
  ∀ (bus_speed train_speed car_speed : ℝ),
  bus_speed = (3/4) * train_speed →
  bus_speed = 480 / 8 →
  car_speed = 450 / 6 →
  train_speed / car_speed = 16 / 15 := by
sorry

end NUMINAMATH_CALUDE_train_car_speed_ratio_l1982_198237


namespace NUMINAMATH_CALUDE_absolute_value_equation_product_l1982_198259

theorem absolute_value_equation_product (x : ℝ) : 
  (|18 / x + 4| = 3) → (∃ y : ℝ, (|18 / y + 4| = 3) ∧ x * y = 324 / 7) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_product_l1982_198259


namespace NUMINAMATH_CALUDE_function_satisfying_condition_l1982_198272

theorem function_satisfying_condition (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y = 1 + x * y + f (x + y)) →
  ((∀ x : ℝ, f x = 2 * x - 1) ∨ (∀ x : ℝ, f x = x^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_condition_l1982_198272


namespace NUMINAMATH_CALUDE_rational_equation_solution_l1982_198242

theorem rational_equation_solution : ∃ x : ℚ, 
  (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 3*x - 18) / (x^2 - 2*x - 24) ∧ 
  x = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l1982_198242


namespace NUMINAMATH_CALUDE_range_of_m_l1982_198224

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 1/x + 4/y = 1) (h2 : ∃ m : ℝ, x + y < m^2 - 8*m) : 
  ∃ m : ℝ, (m < -1 ∨ m > 9) ∧ x + y < m^2 - 8*m := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1982_198224


namespace NUMINAMATH_CALUDE_actual_spent_correct_l1982_198291

/-- Represents a project budget with monthly allocations -/
structure ProjectBudget where
  total : ℕ
  months : ℕ
  monthly_allocation : ℕ
  h_allocation : monthly_allocation * months = total

/-- Calculates the actual amount spent given a project budget and over-budget amount -/
def actual_spent (budget : ProjectBudget) (over_budget : ℕ) (months_elapsed : ℕ) : ℕ :=
  budget.monthly_allocation * months_elapsed + over_budget

/-- Proves that the actual amount spent is correct given the project conditions -/
theorem actual_spent_correct (budget : ProjectBudget) 
    (h_total : budget.total = 12600)
    (h_months : budget.months = 12)
    (h_over_budget : over_budget = 280)
    (h_months_elapsed : months_elapsed = 6) :
    actual_spent budget over_budget months_elapsed = 6580 := by
  sorry

#eval actual_spent ⟨12600, 12, 1050, rfl⟩ 280 6

end NUMINAMATH_CALUDE_actual_spent_correct_l1982_198291


namespace NUMINAMATH_CALUDE_birthday_cake_red_candles_l1982_198267

/-- The number of red candles on a birthday cake -/
def red_candles (total_candles yellow_candles blue_candles : ℕ) : ℕ :=
  total_candles - (yellow_candles + blue_candles)

/-- Theorem stating the number of red candles used for the birthday cake -/
theorem birthday_cake_red_candles :
  red_candles 79 27 38 = 14 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cake_red_candles_l1982_198267


namespace NUMINAMATH_CALUDE_solve_for_p_l1982_198219

theorem solve_for_p (n m p : ℚ) 
  (h1 : 5/6 = n/90)
  (h2 : 5/6 = (m + n)/105)
  (h3 : 5/6 = (p - m)/150) : 
  p = 137.5 := by sorry

end NUMINAMATH_CALUDE_solve_for_p_l1982_198219


namespace NUMINAMATH_CALUDE_seashells_count_l1982_198253

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The total number of seashells found by Tom and Fred -/
def total_seashells : ℕ := tom_seashells + fred_seashells

theorem seashells_count : total_seashells = 58 := by
  sorry

end NUMINAMATH_CALUDE_seashells_count_l1982_198253


namespace NUMINAMATH_CALUDE_four_digit_integers_with_five_or_seven_l1982_198295

theorem four_digit_integers_with_five_or_seven (total_four_digit : Nat) 
  (four_digit_without_five_or_seven : Nat) :
  total_four_digit = 9000 →
  four_digit_without_five_or_seven = 3584 →
  total_four_digit - four_digit_without_five_or_seven = 5416 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_integers_with_five_or_seven_l1982_198295


namespace NUMINAMATH_CALUDE_smallest_winning_number_for_bernardo_l1982_198258

theorem smallest_winning_number_for_bernardo :
  ∃ (N : ℕ), N = 22 ∧
  (∀ k : ℕ, k < N →
    (3*k ≤ 999 ∧
     3*k + 30 ≤ 999 ∧
     9*k + 90 ≤ 999 ∧
     9*k + 120 ≤ 999 ∧
     27*k + 360 ≤ 999)) ∧
  (3*N ≤ 999 ∧
   3*N + 30 ≤ 999 ∧
   9*N + 90 ≤ 999 ∧
   9*N + 120 ≤ 999 ∧
   27*N + 360 ≤ 999 ∧
   27*N + 390 > 999) :=
by sorry

end NUMINAMATH_CALUDE_smallest_winning_number_for_bernardo_l1982_198258


namespace NUMINAMATH_CALUDE_sliding_triangle_forms_ellipse_l1982_198225

/-- Triangle ABC with A and B on perpendicular lines -/
structure SlidingTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  perpendicular : A.2 = 0 ∧ B.1 = 0
  non_right_angle_at_C : ∀ (t : ℝ), (C.1 - A.1) * (C.2 - B.2) ≠ (C.2 - A.2) * (C.1 - B.1)

/-- The locus of point C forms an ellipse -/
def is_ellipse (locus : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), (x, y) ∈ locus ↔ (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The theorem statement -/
theorem sliding_triangle_forms_ellipse (triangle : SlidingTriangle) :
  ∃ (locus : Set (ℝ × ℝ)), is_ellipse locus ∧ ∀ (t : ℝ), triangle.C ∈ locus :=
sorry

end NUMINAMATH_CALUDE_sliding_triangle_forms_ellipse_l1982_198225


namespace NUMINAMATH_CALUDE_teal_color_survey_l1982_198213

theorem teal_color_survey (total : ℕ) (green : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 150)
  (h_green : green = 90)
  (h_both : both = 45)
  (h_neither : neither = 24) :
  ∃ blue : ℕ, blue = 81 ∧ blue = total - (green - both) - both - neither :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l1982_198213


namespace NUMINAMATH_CALUDE_addition_problem_l1982_198234

theorem addition_problem : (-5 : ℤ) + 8 + (-4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_addition_problem_l1982_198234


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1982_198290

/-- The speed of a train given the lengths of two trains, the speed of the other train, and the time they take to cross each other when moving in opposite directions. -/
theorem train_speed_calculation (length1 : ℝ) (length2 : ℝ) (speed2 : ℝ) (cross_time : ℝ) :
  length1 = 270 →
  length2 = 230 →
  speed2 = 80 →
  cross_time = 9 / 3600 →
  (length1 + length2) / 1000 / cross_time - speed2 = 120 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l1982_198290


namespace NUMINAMATH_CALUDE_may_savings_l1982_198200

def savings (month : Nat) : Nat :=
  match month with
  | 0 => 10  -- January (0-indexed)
  | n + 1 => 2 * savings n

theorem may_savings : savings 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_may_savings_l1982_198200


namespace NUMINAMATH_CALUDE_odd_function_domain_l1982_198294

-- Define the function f
def f (a : ℝ) : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_domain (a : ℝ) : 
  (∀ x, f a x ≠ 0 → x ∈ Set.Ioo (3 - 2*a) (a + 1)) →  -- Domain condition
  (∀ x, f a (x + 1) = -f a (-x - 1)) →                -- Odd function condition
  a = 2 := by sorry

end NUMINAMATH_CALUDE_odd_function_domain_l1982_198294


namespace NUMINAMATH_CALUDE_year_square_minus_product_l1982_198256

theorem year_square_minus_product (n : ℕ) : n^2 - (n - 1) * n = n :=
by sorry

end NUMINAMATH_CALUDE_year_square_minus_product_l1982_198256


namespace NUMINAMATH_CALUDE_function_sum_theorem_l1982_198246

/-- Given a function f(x) = a^x + a^(-x) where a > 0 and a ≠ 1, 
    if f(1) = 3, then f(0) + f(1) + f(2) = 12 -/
theorem function_sum_theorem (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x + a^(-x)
  f 1 = 3 → f 0 + f 1 + f 2 = 12 := by
sorry

end NUMINAMATH_CALUDE_function_sum_theorem_l1982_198246


namespace NUMINAMATH_CALUDE_angle_between_planes_exists_l1982_198248

-- Define the basic structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  normal : Point3D
  d : ℝ

-- Define the projection axis
def ProjectionAxis : Point3D → Point3D → Prop :=
  sorry

-- Define a point on the projection axis
def PointOnProjectionAxis (p : Point3D) : Prop :=
  ∃ (q : Point3D), ProjectionAxis p q

-- Define a plane passing through a point
def PlaneThroughPoint (plane : Plane) (point : Point3D) : Prop :=
  plane.normal.x * point.x + plane.normal.y * point.y + plane.normal.z * point.z + plane.d = 0

-- Define the angle between two planes
def AngleBetweenPlanes (plane1 plane2 : Plane) : ℝ :=
  sorry

-- Theorem statement
theorem angle_between_planes_exists :
  ∀ (p : Point3D) (plane1 plane2 : Plane),
    PointOnProjectionAxis p →
    PlaneThroughPoint plane1 p →
    ∃ (angle : ℝ), AngleBetweenPlanes plane1 plane2 = angle :=
  sorry

end NUMINAMATH_CALUDE_angle_between_planes_exists_l1982_198248


namespace NUMINAMATH_CALUDE_equation_solutions_l1982_198270

theorem equation_solutions :
  (∀ x : ℝ, 2 * x^2 + 1 = 3 * x ↔ x = 1 ∨ x = 1/2) ∧
  (∀ x : ℝ, (2*x - 1)^2 = (3 - x)^2 ↔ x = -2 ∨ x = 4/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1982_198270


namespace NUMINAMATH_CALUDE_lines_perpendicular_iff_b_eq_neg_ten_l1982_198241

-- Define the slopes of the two lines
def slope1 : ℚ := -1/2
def slope2 (b : ℚ) : ℚ := -b/5

-- Define the perpendicularity condition
def perpendicular (b : ℚ) : Prop := slope1 * slope2 b = -1

-- Theorem statement
theorem lines_perpendicular_iff_b_eq_neg_ten :
  ∀ b : ℚ, perpendicular b ↔ b = -10 := by sorry

end NUMINAMATH_CALUDE_lines_perpendicular_iff_b_eq_neg_ten_l1982_198241


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1982_198257

theorem sqrt_equation_solution (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 3 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 5 * Real.sqrt (x - 9)) - 1 →
  x ≥ 8 + Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1982_198257


namespace NUMINAMATH_CALUDE_smallest_k_for_cos_squared_one_l1982_198255

theorem smallest_k_for_cos_squared_one :
  ∃ k : ℕ+, 
    (∀ m : ℕ+, m < k → (Real.cos ((m.val ^ 2 + 7 ^ 2 : ℝ) * Real.pi / 180)) ^ 2 ≠ 1) ∧
    (Real.cos ((k.val ^ 2 + 7 ^ 2 : ℝ) * Real.pi / 180)) ^ 2 = 1 ∧
    k = 49 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_cos_squared_one_l1982_198255


namespace NUMINAMATH_CALUDE_exactlyOnePen_exactlyTwoPencils_mutually_exclusive_not_opposite_l1982_198252

/-- Represents the contents of a pencil case -/
structure PencilCase where
  pencils : ℕ
  pens : ℕ

/-- Represents the outcome of selecting two items from a pencil case -/
inductive Selection
  | TwoPencils
  | OnePencilOnePen
  | TwoPens

/-- Defines the pencil case with 2 pencils and 2 pens -/
def myPencilCase : PencilCase := { pencils := 2, pens := 2 }

/-- Event: Exactly 1 pen is selected -/
def exactlyOnePen (s : Selection) : Prop :=
  s = Selection.OnePencilOnePen

/-- Event: Exactly 2 pencils are selected -/
def exactlyTwoPencils (s : Selection) : Prop :=
  s = Selection.TwoPencils

/-- Two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : Selection → Prop) : Prop :=
  ∀ s, ¬(e1 s ∧ e2 s)

/-- Two events are opposite -/
def opposite (e1 e2 : Selection → Prop) : Prop :=
  ∀ s, e1 s ↔ ¬(e2 s)

theorem exactlyOnePen_exactlyTwoPencils_mutually_exclusive_not_opposite :
  mutuallyExclusive exactlyOnePen exactlyTwoPencils ∧
  ¬(opposite exactlyOnePen exactlyTwoPencils) :=
by sorry

end NUMINAMATH_CALUDE_exactlyOnePen_exactlyTwoPencils_mutually_exclusive_not_opposite_l1982_198252


namespace NUMINAMATH_CALUDE_number_of_girls_l1982_198206

theorem number_of_girls (total_students : ℕ) (prob_girl : ℚ) (num_girls : ℕ) : 
  total_students = 20 →
  prob_girl = 2/5 →
  num_girls = (total_students : ℚ) * prob_girl →
  num_girls = 8 := by
sorry

end NUMINAMATH_CALUDE_number_of_girls_l1982_198206


namespace NUMINAMATH_CALUDE_only_negative_number_l1982_198235

theorem only_negative_number (a b c d : ℤ) (h1 : a = 5) (h2 : b = 1) (h3 : c = -2) (h4 : d = 0) :
  (a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0) ∧ (c < 0) ∧ (a ≥ 0) ∧ (b ≥ 0) ∧ (d ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_number_l1982_198235


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l1982_198226

theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℝ),
  (3/4 * 12 * banana_value = 6 * orange_value) →
  (1/4 * 12 * banana_value = 2 * orange_value) :=
by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l1982_198226


namespace NUMINAMATH_CALUDE_real_part_of_complex_number_l1982_198208

theorem real_part_of_complex_number : 
  (1 + 2 / (Complex.I + 1)).re = 2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_number_l1982_198208
