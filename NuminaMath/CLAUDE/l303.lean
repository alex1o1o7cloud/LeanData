import Mathlib

namespace NUMINAMATH_CALUDE_solve_equation_l303_30331

theorem solve_equation (a : ℚ) (h : 2 * a + 2 * a / 4 = 4) : a = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l303_30331


namespace NUMINAMATH_CALUDE_f_derivative_at_fixed_point_l303_30390

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos x)))))))

theorem f_derivative_at_fixed_point (a : ℝ) (h : a = Real.cos a) :
  deriv f a = a^8 - 4*a^6 + 6*a^4 - 4*a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_fixed_point_l303_30390


namespace NUMINAMATH_CALUDE_count_dominoes_l303_30314

/-- The number of different (noncongruent) dominoes in an m × n array -/
def num_dominoes (m n : ℕ) : ℚ :=
  m * n - m^2 / 2 + m / 2 - 1

/-- Theorem: The number of different (noncongruent) dominoes in an m × n array -/
theorem count_dominoes (m n : ℕ) (h : 0 < m ∧ m ≤ n) :
  num_dominoes m n = m * n - m^2 / 2 + m / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_count_dominoes_l303_30314


namespace NUMINAMATH_CALUDE_product_negative_from_positive_sum_negative_quotient_l303_30323

theorem product_negative_from_positive_sum_negative_quotient
  (a b : ℝ) (h_sum : a + b > 0) (h_quotient : a / b < 0) :
  a * b < 0 :=
by sorry

end NUMINAMATH_CALUDE_product_negative_from_positive_sum_negative_quotient_l303_30323


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l303_30334

/-- The lateral surface area of a cone with base radius 5 and height 12 is 65π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 5
  let h : ℝ := 12
  let l : ℝ := Real.sqrt (r^2 + h^2)
  π * r * l = 65 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l303_30334


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l303_30388

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt x + Real.sqrt 243) / Real.sqrt 75 = 2.4 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l303_30388


namespace NUMINAMATH_CALUDE_pattern_A_cannot_fold_into_cube_l303_30377

/-- Represents a pattern of squares -/
inductive Pattern
  | A  -- Five squares in a cross shape
  | B  -- Four squares in a "T" shape
  | C  -- Six squares in a "T" shape with an additional square
  | D  -- Three squares in a straight line

/-- Number of squares in a pattern -/
def squareCount (p : Pattern) : Nat :=
  match p with
  | .A => 5
  | .B => 4
  | .C => 6
  | .D => 3

/-- Number of squares required to form a cube -/
def cubeSquareCount : Nat := 6

/-- Checks if a pattern can be folded into a cube -/
def canFoldIntoCube (p : Pattern) : Prop :=
  squareCount p = cubeSquareCount ∧ 
  (p ≠ Pattern.A) -- Pattern A cannot be closed even with 5 squares

/-- Theorem: Pattern A cannot be folded into a cube -/
theorem pattern_A_cannot_fold_into_cube : 
  ¬ (canFoldIntoCube Pattern.A) := by
  sorry


end NUMINAMATH_CALUDE_pattern_A_cannot_fold_into_cube_l303_30377


namespace NUMINAMATH_CALUDE_square_side_length_l303_30366

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 12) (h2 : area = side ^ 2) :
  side = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l303_30366


namespace NUMINAMATH_CALUDE_sqrt_inequality_l303_30373

theorem sqrt_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : c ≥ d) (h4 : d > 0) : 
  Real.sqrt (a / d) > Real.sqrt (b / c) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l303_30373


namespace NUMINAMATH_CALUDE_time_on_other_subjects_is_40_l303_30308

/-- Represents the time spent on homework for each subject -/
structure HomeworkTime where
  total : ℝ
  math : ℝ
  science : ℝ
  history : ℝ
  english : ℝ

/-- Calculates the time spent on other subjects -/
def timeOnOtherSubjects (hw : HomeworkTime) : ℝ :=
  hw.total - (hw.math + hw.science + hw.history + hw.english)

/-- Theorem stating the time spent on other subjects is 40 minutes -/
theorem time_on_other_subjects_is_40 (hw : HomeworkTime) : 
  hw.total = 150 ∧
  hw.math = 0.20 * hw.total ∧
  hw.science = 0.25 * hw.total ∧
  hw.history = 0.10 * hw.total ∧
  hw.english = 0.15 * hw.total ∧
  hw.history ≥ 20 ∧
  hw.science ≥ 20 →
  timeOnOtherSubjects hw = 40 := by
  sorry

#check time_on_other_subjects_is_40

end NUMINAMATH_CALUDE_time_on_other_subjects_is_40_l303_30308


namespace NUMINAMATH_CALUDE_smallest_sum_sequence_l303_30380

theorem smallest_sum_sequence (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (∃ d : ℤ, C - B = d ∧ B - A = d) →  -- A, B, C form an arithmetic sequence
  (∃ r : ℚ, C = r * B ∧ D = r * C) →  -- B, C, D form a geometric sequence
  C = (7 : ℚ) / 3 * B →  -- C/B = 7/3
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 →
    (∃ d : ℤ, C' - B' = d ∧ B' - A' = d) →
    (∃ r : ℚ, C' = r * B' ∧ D' = r * C') →
    C' = (7 : ℚ) / 3 * B' →
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 76 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_sequence_l303_30380


namespace NUMINAMATH_CALUDE_one_and_two_red_mutually_exclusive_not_opposing_l303_30360

/-- Represents the number of red balls drawn -/
inductive RedBallsDrawn
  | zero
  | one
  | two
  | three

/-- The probability of drawing exactly one red ball -/
def prob_one_red : ℝ := sorry

/-- The probability of drawing exactly two red balls -/
def prob_two_red : ℝ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 5

/-- The number of white balls in the bag -/
def white_balls : ℕ := 3

/-- The number of balls drawn -/
def drawn_balls : ℕ := 3

theorem one_and_two_red_mutually_exclusive_not_opposing :
  (prob_one_red * prob_two_red = 0) ∧ (prob_one_red + prob_two_red < 1) := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_red_mutually_exclusive_not_opposing_l303_30360


namespace NUMINAMATH_CALUDE_polynomial_transformation_l303_30365

theorem polynomial_transformation (g : ℝ → ℝ) :
  (∀ x, g (x^2 - 2) = x^4 - 6*x^2 + 8) →
  (∀ x, g (x^2 - 1) = x^4 - 4*x^2 + 7) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l303_30365


namespace NUMINAMATH_CALUDE_probability_of_two_red_books_l303_30358

-- Define the number of red and blue books
def red_books : ℕ := 4
def blue_books : ℕ := 4
def total_books : ℕ := red_books + blue_books

-- Define the number of books to be selected
def books_selected : ℕ := 2

-- Function to calculate combinations
def combination (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem statement
theorem probability_of_two_red_books :
  (combination red_books books_selected : ℚ) / (combination total_books books_selected) = 3 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_two_red_books_l303_30358


namespace NUMINAMATH_CALUDE_joshua_fruit_profit_l303_30326

/-- Calculates the total profit in cents for Joshua's fruit sales --/
def fruit_profit (orange_qty : ℕ) (apple_qty : ℕ) (banana_qty : ℕ)
  (orange_cost : ℚ) (apple_cost : ℚ) (banana_cost : ℚ)
  (orange_sell : ℚ) (apple_sell : ℚ) (banana_sell : ℚ)
  (discount_threshold : ℕ) (discount_rate : ℚ) : ℕ :=
  let orange_total_cost := orange_qty * orange_cost
  let apple_total_cost := if apple_qty ≥ discount_threshold
    then apple_qty * (apple_cost * (1 - discount_rate))
    else apple_qty * apple_cost
  let banana_total_cost := if banana_qty ≥ discount_threshold
    then banana_qty * (banana_cost * (1 - discount_rate))
    else banana_qty * banana_cost
  let total_cost := orange_total_cost + apple_total_cost + banana_total_cost
  let total_revenue := orange_qty * orange_sell + apple_qty * apple_sell + banana_qty * banana_sell
  let profit := total_revenue - total_cost
  (profit * 100).floor.toNat

/-- Theorem stating that Joshua's profit is 2035 cents --/
theorem joshua_fruit_profit :
  fruit_profit 25 40 50 0.5 0.65 0.25 0.6 0.75 0.45 30 0.1 = 2035 := by
  sorry

end NUMINAMATH_CALUDE_joshua_fruit_profit_l303_30326


namespace NUMINAMATH_CALUDE_max_value_of_expression_l303_30347

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x * y / (2 * y + 3 * x) = 1) : 
  x / 2 + y / 3 ≤ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 
    x₀ * y₀ / (2 * y₀ + 3 * x₀) = 1 ∧ x₀ / 2 + y₀ / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l303_30347


namespace NUMINAMATH_CALUDE_expression_equality_l303_30361

theorem expression_equality : (8 : ℕ)^6 * 27^6 * 8^18 * 27^18 = 216^24 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l303_30361


namespace NUMINAMATH_CALUDE_matt_darius_difference_l303_30324

/-- The scores of three friends in a table football game. -/
structure Scores where
  darius : ℕ
  matt : ℕ
  marius : ℕ

/-- The conditions of the table football game. -/
def game_conditions (s : Scores) : Prop :=
  s.darius = 10 ∧
  s.marius = s.darius + 3 ∧
  s.matt > s.darius ∧
  s.darius + s.matt + s.marius = 38

/-- The theorem stating the difference between Matt's and Darius's scores. -/
theorem matt_darius_difference (s : Scores) (h : game_conditions s) : 
  s.matt - s.darius = 5 := by
  sorry

end NUMINAMATH_CALUDE_matt_darius_difference_l303_30324


namespace NUMINAMATH_CALUDE_increasing_f_implies_a_in_closed_interval_l303_30349

/-- A function f : ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The cosine function -/
noncomputable def cos : ℝ → ℝ := Real.cos

/-- The function f(x) = x - a * cos(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * cos x

theorem increasing_f_implies_a_in_closed_interval :
  ∀ a : ℝ, IsIncreasing (f a) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_f_implies_a_in_closed_interval_l303_30349


namespace NUMINAMATH_CALUDE_exists_monochromatic_isosceles_triangle_l303_30338

-- Define a color type
inductive Color
  | Red
  | Green
  | Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define an isosceles triangle
def isIsoscelesTriangle (p q r : Point) : Prop := sorry

-- Theorem statement
theorem exists_monochromatic_isosceles_triangle :
  ∃ (p q r : Point), 
    isIsoscelesTriangle p q r ∧ 
    coloring p = coloring q ∧ 
    coloring q = coloring r := 
by sorry

end NUMINAMATH_CALUDE_exists_monochromatic_isosceles_triangle_l303_30338


namespace NUMINAMATH_CALUDE_bmw_sales_l303_30395

def total_cars : ℕ := 300
def ford_percentage : ℚ := 18 / 100
def nissan_percentage : ℚ := 25 / 100
def chevrolet_percentage : ℚ := 20 / 100

theorem bmw_sales : 
  let other_brands_percentage := ford_percentage + nissan_percentage + chevrolet_percentage
  let bmw_percentage := 1 - other_brands_percentage
  ↑⌊bmw_percentage * total_cars⌋ = 111 := by sorry

end NUMINAMATH_CALUDE_bmw_sales_l303_30395


namespace NUMINAMATH_CALUDE_smallest_sum_c_d_l303_30317

theorem smallest_sum_c_d (c d : ℝ) : 
  c > 0 → d > 0 → 
  (∃ x : ℝ, x^2 + c*x + 3*d = 0) → 
  (∃ x : ℝ, x^2 + 3*d*x + c = 0) → 
  c + d ≥ 16/3 ∧ 
  ∃ c₀ d₀ : ℝ, c₀ > 0 ∧ d₀ > 0 ∧ 
    (∃ x : ℝ, x^2 + c₀*x + 3*d₀ = 0) ∧ 
    (∃ x : ℝ, x^2 + 3*d₀*x + c₀ = 0) ∧ 
    c₀ + d₀ = 16/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_c_d_l303_30317


namespace NUMINAMATH_CALUDE_lucy_grocery_shopping_l303_30345

theorem lucy_grocery_shopping (total_packs noodle_packs : ℕ) 
  (h1 : total_packs = 28)
  (h2 : noodle_packs = 16)
  (h3 : ∃ cookie_packs : ℕ, total_packs = cookie_packs + noodle_packs) :
  ∃ cookie_packs : ℕ, cookie_packs = 12 ∧ total_packs = cookie_packs + noodle_packs :=
by
  sorry

end NUMINAMATH_CALUDE_lucy_grocery_shopping_l303_30345


namespace NUMINAMATH_CALUDE_compare_fractions_l303_30351

theorem compare_fractions : (-5/6 : ℚ) > -|(-8/9 : ℚ)| := by sorry

end NUMINAMATH_CALUDE_compare_fractions_l303_30351


namespace NUMINAMATH_CALUDE_matrix_property_l303_30313

theorem matrix_property (a b c d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.transpose = A⁻¹) → (Matrix.det A = 1) → (a^2 + b^2 + c^2 + d^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_matrix_property_l303_30313


namespace NUMINAMATH_CALUDE_building_shadow_length_l303_30391

/-- Given a flagstaff and a building under similar conditions, prove that the length of the shadow
    cast by the building is 28.75 m. -/
theorem building_shadow_length
  (flagstaff_height : ℝ)
  (flagstaff_shadow : ℝ)
  (building_height : ℝ)
  (h1 : flagstaff_height = 17.5)
  (h2 : flagstaff_shadow = 40.25)
  (h3 : building_height = 12.5)
  : (building_height * flagstaff_shadow) / flagstaff_height = 28.75 := by
  sorry

end NUMINAMATH_CALUDE_building_shadow_length_l303_30391


namespace NUMINAMATH_CALUDE_log_difference_times_sqrt10_l303_30320

-- Define the logarithm function (base 10)
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_difference_times_sqrt10 :
  (log10 (1/4) - log10 25) * (10 ^ (1/2 : ℝ)) = -10 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_times_sqrt10_l303_30320


namespace NUMINAMATH_CALUDE_donation_distribution_l303_30368

/-- Calculates the amount each organization receives when a company donates a portion of its funds to a foundation with multiple organizations. -/
theorem donation_distribution (total_amount : ℝ) (donation_percentage : ℝ) (num_organizations : ℕ) 
  (h1 : total_amount = 2500)
  (h2 : donation_percentage = 80 / 100)
  (h3 : num_organizations = 8) :
  (total_amount * donation_percentage) / num_organizations = 250 := by
  sorry

end NUMINAMATH_CALUDE_donation_distribution_l303_30368


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l303_30328

theorem divisibility_implies_equality (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h_div : (a^2 + a*b + 1) % (b^2 + a*b + 1) = 0) : a = b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l303_30328


namespace NUMINAMATH_CALUDE_base_eight_1423_equals_787_l303_30352

/-- Converts a base-8 digit to its base-10 equivalent -/
def baseEightDigitToBaseTen (d : Nat) : Nat :=
  if d < 8 then d else 0

/-- Converts a four-digit base-8 number to base-10 -/
def baseEightToBaseTen (a b c d : Nat) : Nat :=
  (baseEightDigitToBaseTen a) * 512 + 
  (baseEightDigitToBaseTen b) * 64 + 
  (baseEightDigitToBaseTen c) * 8 + 
  (baseEightDigitToBaseTen d)

theorem base_eight_1423_equals_787 : 
  baseEightToBaseTen 1 4 2 3 = 787 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_1423_equals_787_l303_30352


namespace NUMINAMATH_CALUDE_oil_price_reduction_l303_30322

theorem oil_price_reduction (original_price : ℝ) : 
  (original_price > 0) →
  (1100 = (1100 / original_price) * original_price) →
  (1100 = ((1100 / original_price) + 5) * (0.75 * original_price)) →
  (0.75 * original_price = 55) := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l303_30322


namespace NUMINAMATH_CALUDE_min_value_expression_l303_30379

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 20 - 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l303_30379


namespace NUMINAMATH_CALUDE_odd_square_octal_property_l303_30375

theorem odd_square_octal_property (n : ℤ) : 
  ∃ (m : ℤ), (2*n + 1)^2 % 8 = 1 ∧ ((2*n + 1)^2 - 1) / 8 = m * (m + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_odd_square_octal_property_l303_30375


namespace NUMINAMATH_CALUDE_quadratic_sum_l303_30336

/-- A quadratic function passing through specific points -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (QuadraticFunction a b c 0 = 7) →
  (QuadraticFunction a b c 1 = 4) →
  a + b + 2*c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l303_30336


namespace NUMINAMATH_CALUDE_square_area_ratio_l303_30325

/-- The ratio of areas between a smaller square and a larger square, given specific conditions -/
theorem square_area_ratio : ∀ (r : ℝ) (y : ℝ),
  r > 0 →  -- radius of circumscribed circle is positive
  r = 4 * Real.sqrt 2 →  -- radius of circumscribed circle
  y > 0 →  -- half side length of smaller square is positive
  y * (3 * y - 8 * Real.sqrt 2) = 0 →  -- condition for diagonal touching circle
  (2 * y)^2 / 8^2 = 8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l303_30325


namespace NUMINAMATH_CALUDE_root_problems_l303_30330

theorem root_problems :
  (∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ x = 2) ∧
  (∃ x y : ℝ, x^2 = 5 ∧ y^2 = 5 ∧ x = -y ∧ x ≠ 0) ∧
  (∃ x : ℝ, x^3 = -27 ∧ x = -3) :=
by sorry

end NUMINAMATH_CALUDE_root_problems_l303_30330


namespace NUMINAMATH_CALUDE_division_problem_l303_30392

theorem division_problem :
  let dividend : ℕ := 16698
  let divisor : ℝ := 187.46067415730337
  let quotient : ℕ := 89
  let remainder : ℕ := 14
  (dividend : ℝ) = divisor * quotient + remainder :=
by sorry

end NUMINAMATH_CALUDE_division_problem_l303_30392


namespace NUMINAMATH_CALUDE_equal_distribution_classroom_l303_30353

/-- Proves that given 4 classrooms, 56 boys, and 44 girls, with an equal distribution of boys and girls across all classrooms, the total number of students in each classroom is 25. -/
theorem equal_distribution_classroom (num_classrooms : ℕ) (num_boys : ℕ) (num_girls : ℕ) 
  (h1 : num_classrooms = 4)
  (h2 : num_boys = 56)
  (h3 : num_girls = 44)
  (h4 : num_boys % num_classrooms = 0)
  (h5 : num_girls % num_classrooms = 0) :
  (num_boys / num_classrooms) + (num_girls / num_classrooms) = 25 :=
by sorry

end NUMINAMATH_CALUDE_equal_distribution_classroom_l303_30353


namespace NUMINAMATH_CALUDE_quadratic_roots_theorem_l303_30371

/-- A quadratic equation with parameter m -/
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*(m+1)*x + m^2 + 5

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ := 8*m - 16

/-- The condition for real roots -/
def has_real_roots (m : ℝ) : Prop := discriminant m ≥ 0

/-- The relation between roots and m -/
def roots_relation (m : ℝ) (x₁ x₂ : ℝ) : Prop :=
  quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ (x₁ - 1)*(x₂ - 1) = 28

theorem quadratic_roots_theorem (m : ℝ) :
  has_real_roots m →
  (∃ x₁ x₂, roots_relation m x₁ x₂) →
  m ≥ 2 ∧ m = 6 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_theorem_l303_30371


namespace NUMINAMATH_CALUDE_problem_solution_l303_30363

theorem problem_solution (x y z : ℝ) 
  (h1 : 2 * x + y + z = 14)
  (h2 : 2 * x + y = 7)
  (h3 : x + 2 * y + Real.sqrt z = 10) :
  (x + y - z) / 3 = (-4 - Real.sqrt 7) / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l303_30363


namespace NUMINAMATH_CALUDE_BH_length_l303_30329

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CA := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB = 5 ∧ BC = 7 ∧ CA = 8

-- Define points G and H on ray AB
def points_on_ray (A B G H : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ > 1 ∧ t₂ > t₁ ∧
  G = (A.1 + t₁ * (B.1 - A.1), A.2 + t₁ * (B.2 - A.2)) ∧
  H = (A.1 + t₂ * (B.1 - A.1), A.2 + t₂ * (B.2 - A.2))

-- Define point I on the intersection of circumcircles
def point_on_circumcircles (A B C G H I : ℝ × ℝ) : Prop :=
  I ≠ C ∧
  ∃ r₁ r₂ : ℝ,
    (I.1 - A.1)^2 + (I.2 - A.2)^2 = r₁^2 ∧
    (G.1 - A.1)^2 + (G.2 - A.2)^2 = r₁^2 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = r₁^2 ∧
    (I.1 - B.1)^2 + (I.2 - B.2)^2 = r₂^2 ∧
    (H.1 - B.1)^2 + (H.2 - B.2)^2 = r₂^2 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = r₂^2

-- Define distances GI and HI
def distances (G H I : ℝ × ℝ) : Prop :=
  Real.sqrt ((G.1 - I.1)^2 + (G.2 - I.2)^2) = 3 ∧
  Real.sqrt ((H.1 - I.1)^2 + (H.2 - I.2)^2) = 8

-- Main theorem
theorem BH_length (A B C G H I : ℝ × ℝ) :
  triangle_ABC A B C →
  points_on_ray A B G H →
  point_on_circumcircles A B C G H I →
  distances G H I →
  Real.sqrt ((B.1 - H.1)^2 + (B.2 - H.2)^2) = (6 + 47 * Real.sqrt 2) / 9 := by
  sorry

end NUMINAMATH_CALUDE_BH_length_l303_30329


namespace NUMINAMATH_CALUDE_train_delay_l303_30385

/-- Proves that a train moving at 4/5 of its usual speed will be 30 minutes late on a journey that usually takes 2 hours -/
theorem train_delay (usual_speed : ℝ) (usual_time : ℝ) (h1 : usual_time = 2) :
  let reduced_speed := (4/5 : ℝ) * usual_speed
  let reduced_time := usual_time * (5/4 : ℝ)
  reduced_time - usual_time = 1/2 := by sorry

#check train_delay

end NUMINAMATH_CALUDE_train_delay_l303_30385


namespace NUMINAMATH_CALUDE_unique_line_count_for_p_2_l303_30357

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for lines in a plane
def Line : Type := Point → Point → Prop

-- Define a function to count the number of intersection points
def count_intersections (lines : List Line) : ℕ := sorry

-- Define a function to check if lines intersect at exactly p points
def intersect_at_p_points (lines : List Line) (p : ℕ) : Prop :=
  count_intersections lines = p

-- Theorem: When p = 2, there is a unique number of lines (3) that intersect at exactly p points
theorem unique_line_count_for_p_2 :
  ∃! n : ℕ, ∃ lines : List Line, intersect_at_p_points lines 2 ∧ lines.length = n :=
sorry

end NUMINAMATH_CALUDE_unique_line_count_for_p_2_l303_30357


namespace NUMINAMATH_CALUDE_angle_relation_l303_30311

theorem angle_relation (α β : Real) (x y : Real) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos (α + β) = -4/5)
  (h4 : Real.sin β = x)
  (h5 : Real.cos α = y)
  (h6 : 4/5 < x ∧ x < 1) :
  y = -4/5 * Real.sqrt (1 - x^2) + 3/5 * x := by
  sorry

#check angle_relation

end NUMINAMATH_CALUDE_angle_relation_l303_30311


namespace NUMINAMATH_CALUDE_quadruple_base_exponent_l303_30318

theorem quadruple_base_exponent (a b x y s : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hs : s > 0)
  (h1 : s = (4*a)^(4*b))
  (h2 : s = a^b * y^b)
  (h3 : y = 4*x) : 
  x = 64 * a^3 := by sorry

end NUMINAMATH_CALUDE_quadruple_base_exponent_l303_30318


namespace NUMINAMATH_CALUDE_problem_1_l303_30355

theorem problem_1 : 99 * (118 + 4/5) + 99 * (-1/5) - 99 * (18 + 3/5) = 9900 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l303_30355


namespace NUMINAMATH_CALUDE_quadratic_tangent_theorem_l303_30316

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determines if a quadratic function is tangent to the x-axis -/
def isTangentToXAxis (f : QuadraticFunction) : Prop :=
  f.b^2 - 4*f.a*f.c = 0

/-- Determines if the vertex of a quadratic function is a minimum point -/
def hasMinimumVertex (f : QuadraticFunction) : Prop :=
  f.a > 0

/-- The main theorem to be proved -/
theorem quadratic_tangent_theorem :
  ∀ (d : ℝ),
  let f : QuadraticFunction := ⟨3, 12, d⟩
  isTangentToXAxis f →
  d = 12 ∧ hasMinimumVertex f := by
  sorry

end NUMINAMATH_CALUDE_quadratic_tangent_theorem_l303_30316


namespace NUMINAMATH_CALUDE_twentyFifthInBase6_l303_30393

/-- Converts a natural number to its representation in base 6 --/
def toBase6 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- Converts a list of digits in base 6 to a natural number --/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => acc * 6 + d) 0

theorem twentyFifthInBase6 : fromBase6 [4, 1] = 25 := by
  sorry

#eval toBase6 25  -- Should output [4, 1]
#eval fromBase6 [4, 1]  -- Should output 25

end NUMINAMATH_CALUDE_twentyFifthInBase6_l303_30393


namespace NUMINAMATH_CALUDE_households_without_car_or_bike_l303_30333

theorem households_without_car_or_bike 
  (total : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : both = 20)
  (h3 : with_car = 44)
  (h4 : bike_only = 35) :
  total - (with_car + bike_only) = 11 :=
by sorry

end NUMINAMATH_CALUDE_households_without_car_or_bike_l303_30333


namespace NUMINAMATH_CALUDE_problem_statement_l303_30350

theorem problem_statement (x y z : ℝ) 
  (h1 : x ≤ y) (h2 : y ≤ z) 
  (h3 : x + y + z = 12) 
  (h4 : x^2 + y^2 + z^2 = 54) : 
  x ≤ 3 ∧ z ≥ 5 ∧ 
  9 ≤ x * y ∧ x * y ≤ 25 ∧ 
  9 ≤ y * z ∧ y * z ≤ 25 ∧ 
  9 ≤ z * x ∧ z * x ≤ 25 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l303_30350


namespace NUMINAMATH_CALUDE_cube_face_sum_l303_30337

/-- Represents the numbers on the faces of a cube -/
def CubeFaces := Fin 6 → ℕ

/-- The sum of a pair of opposite faces -/
def OppositeSum (faces : CubeFaces) (pair : Fin 3) : ℕ :=
  faces (2 * pair) + faces (2 * pair + 1)

theorem cube_face_sum (faces : CubeFaces) :
  (∃ (pair : Fin 3), OppositeSum faces pair = 11) →
  (∀ (pair : Fin 3), OppositeSum faces pair ≠ 9) ∧
  (∀ (i : Fin 6), faces i ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ)) ∧
  (∀ (i j : Fin 6), i ≠ j → faces i ≠ faces j) :=
by sorry

end NUMINAMATH_CALUDE_cube_face_sum_l303_30337


namespace NUMINAMATH_CALUDE_vegetable_ghee_ratio_l303_30340

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 950

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 850

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3640

/-- The volume of brand 'a' in the mixture -/
def volume_a : ℝ := 2.4

/-- The volume of brand 'b' in the mixture -/
def volume_b : ℝ := 1.6

/-- Theorem stating that the ratio of volumes of brand 'a' to brand 'b' is 1.5:1 -/
theorem vegetable_ghee_ratio :
  volume_a / volume_b = 1.5 ∧
  volume_a + volume_b = total_volume ∧
  weight_a * volume_a + weight_b * volume_b = total_weight :=
by sorry

end NUMINAMATH_CALUDE_vegetable_ghee_ratio_l303_30340


namespace NUMINAMATH_CALUDE_triangle_properties_l303_30370

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a = 2 ∧
  cos B = 3/5 →
  (b = 4 → sin A = 2/5) ∧
  (1/2 * a * c * sin B = 4 → b = Real.sqrt 17 ∧ c = 5) := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l303_30370


namespace NUMINAMATH_CALUDE_special_trapezoid_angle_l303_30341

/-- A trapezoid with special properties -/
structure SpecialTrapezoid where
  /-- The diagonals intersect at a right angle -/
  diagonals_right_angle : Bool
  /-- One diagonal is equal to the midsegment -/
  diagonal_equals_midsegment : Bool

/-- The angle formed by the special diagonal and the bases of the trapezoid -/
def diagonal_base_angle (t : SpecialTrapezoid) : Real :=
  sorry

/-- Theorem: In a special trapezoid, the angle between the special diagonal and the bases is 60° -/
theorem special_trapezoid_angle (t : SpecialTrapezoid) 
  (h1 : t.diagonals_right_angle = true) 
  (h2 : t.diagonal_equals_midsegment = true) : 
  diagonal_base_angle t = 60 := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_angle_l303_30341


namespace NUMINAMATH_CALUDE_product_real_implies_b_value_l303_30372

theorem product_real_implies_b_value (z₁ z₂ : ℂ) (b : ℝ) :
  z₁ = 1 + I →
  z₂ = 2 + b * I →
  (z₁ * z₂).im = 0 →
  b = -2 := by
sorry

end NUMINAMATH_CALUDE_product_real_implies_b_value_l303_30372


namespace NUMINAMATH_CALUDE_range_of_a_l303_30335

theorem range_of_a (a b c : ℝ) 
  (h1 : b^2 + c^2 = -a^2 + 14*a + 5) 
  (h2 : b*c = a^2 - 2*a + 10) : 
  1 ≤ a ∧ a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l303_30335


namespace NUMINAMATH_CALUDE_product_base_8_units_digit_l303_30344

def base_10_product : ℕ := 123 * 57

def base_8_units_digit (n : ℕ) : ℕ := n % 8

theorem product_base_8_units_digit :
  base_8_units_digit base_10_product = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_base_8_units_digit_l303_30344


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l303_30397

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality : 
  (¬ ∃ x : ℝ, x^2 + 4*x + 6 < 0) ↔ (∀ x : ℝ, x^2 + 4*x + 6 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_inequality_l303_30397


namespace NUMINAMATH_CALUDE_system_solution_l303_30381

theorem system_solution (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧ 
  x^2 + y * Real.sqrt (x * y) = 105 ∧
  y^2 + x * Real.sqrt (y * x) = 70 →
  x = 9 ∧ y = 4 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l303_30381


namespace NUMINAMATH_CALUDE_inequality_of_cubes_l303_30300

theorem inequality_of_cubes (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_cubes_l303_30300


namespace NUMINAMATH_CALUDE_sum_of_squares_2870_l303_30307

theorem sum_of_squares_2870 :
  ∃! (n : ℕ), n > 0 ∧ n * (n + 1) * (2 * n + 1) / 6 = 2870 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_2870_l303_30307


namespace NUMINAMATH_CALUDE_cubic_equation_value_l303_30306

theorem cubic_equation_value (x : ℝ) (h : 3 * x^2 - x = 1) :
  6 * x^3 + 7 * x^2 - 5 * x + 2008 = 2011 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_value_l303_30306


namespace NUMINAMATH_CALUDE_midpoint_chain_l303_30304

/-- Given a line segment XY, we define points G, H, I, and J as follows:
  G is the midpoint of XY
  H is the midpoint of XG
  I is the midpoint of XH
  J is the midpoint of XI
  If XJ = 4, then XY = 64 -/
theorem midpoint_chain (X Y G H I J : ℝ) : 
  (G = (X + Y) / 2) →  -- G is midpoint of XY
  (H = (X + G) / 2) →  -- H is midpoint of XG
  (I = (X + H) / 2) →  -- I is midpoint of XH
  (J = (X + I) / 2) →  -- J is midpoint of XI
  (J - X = 4) →        -- XJ = 4
  (Y - X = 64) :=      -- XY = 64
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l303_30304


namespace NUMINAMATH_CALUDE_undefined_values_l303_30383

theorem undefined_values (x : ℝ) :
  (x^2 - 21*x + 110 = 0) ↔ (x = 10 ∨ x = 11) := by sorry

end NUMINAMATH_CALUDE_undefined_values_l303_30383


namespace NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l303_30369

theorem power_product_equals_sum_of_exponents (a : ℝ) : a^4 * a = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_sum_of_exponents_l303_30369


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l303_30348

/-- Given two vectors a and b in ℝ², where a is parallel to b,
    prove that the magnitude of b is 2√5 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) :
  a = (1, 2) →
  b.1 = -2 →
  ∃ (k : ℝ), k ≠ 0 ∧ a = k • b →
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l303_30348


namespace NUMINAMATH_CALUDE_share_face_value_l303_30354

theorem share_face_value (dividend_rate : ℝ) (desired_return : ℝ) (market_value : ℝ) :
  dividend_rate = 0.09 →
  desired_return = 0.12 →
  market_value = 36.00000000000001 →
  (desired_return / dividend_rate) * market_value = 48.00000000000001 :=
by
  sorry

end NUMINAMATH_CALUDE_share_face_value_l303_30354


namespace NUMINAMATH_CALUDE_heidi_painting_fraction_l303_30374

/-- If a person can paint a wall in a given time, this function calculates
    the fraction of the wall they can paint in a shorter time. -/
def fractionPainted (totalTime minutes : ℕ) : ℚ :=
  minutes / totalTime

/-- Theorem stating that if Heidi can paint a wall in 60 minutes,
    she can paint 1/5 of the wall in 12 minutes. -/
theorem heidi_painting_fraction :
  fractionPainted 60 12 = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_heidi_painting_fraction_l303_30374


namespace NUMINAMATH_CALUDE_cubic_decreasing_l303_30346

-- Define the function f(x) = mx³ - x
def f (m : ℝ) (x : ℝ) : ℝ := m * x^3 - x

-- State the theorem
theorem cubic_decreasing (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x ≥ f m y) ↔ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_decreasing_l303_30346


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l303_30309

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 4 + 2 * Complex.I) :
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l303_30309


namespace NUMINAMATH_CALUDE_line_intersects_segment_midpoint_l303_30389

theorem line_intersects_segment_midpoint (b : ℝ) : 
  let p1 : ℝ × ℝ := (3, 2)
  let p2 : ℝ × ℝ := (7, 6)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (b = 9) ↔ (midpoint.1 + midpoint.2 = b) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_segment_midpoint_l303_30389


namespace NUMINAMATH_CALUDE_smallest_x_value_l303_30310

theorem smallest_x_value (x : ℝ) : 
  (((5*x - 20)/(4*x - 5))^3 + ((5*x - 20)/(4*x - 5))^2 - ((5*x - 20)/(4*x - 5)) - 15 = 0) → 
  (∀ y : ℝ, (((5*y - 20)/(4*y - 5))^3 + ((5*y - 20)/(4*y - 5))^2 - ((5*y - 20)/(4*y - 5)) - 15 = 0) → 
  x ≤ y) → 
  x = 10/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l303_30310


namespace NUMINAMATH_CALUDE_square_diagonal_ratio_l303_30312

theorem square_diagonal_ratio (a b : ℝ) (h : b^2 / a^2 = 4) :
  (b * Real.sqrt 2) / (a * Real.sqrt 2) = 2 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_ratio_l303_30312


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l303_30319

theorem polynomial_divisibility (C D : ℝ) :
  (∀ x : ℂ, x^2 + x + 1 = 0 → x^104 + C*x + D = 0) →
  C + D = 2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l303_30319


namespace NUMINAMATH_CALUDE_g_behavior_l303_30362

/-- The quadratic function g(x) = x^2 - 2x - 8 -/
def g (x : ℝ) : ℝ := x^2 - 2*x - 8

/-- The graph of g(x) goes up to the right and up to the left -/
theorem g_behavior :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x > M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) :=
sorry

end NUMINAMATH_CALUDE_g_behavior_l303_30362


namespace NUMINAMATH_CALUDE_max_xy_constraint_l303_30359

theorem max_xy_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_constraint : 4 * x + 9 * y = 6) :
  x * y ≤ 1 / 4 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4 * x₀ + 9 * y₀ = 6 ∧ x₀ * y₀ = 1 / 4 :=
sorry

end NUMINAMATH_CALUDE_max_xy_constraint_l303_30359


namespace NUMINAMATH_CALUDE_casting_theorem_l303_30315

def men : ℕ := 7
def women : ℕ := 5
def male_roles : ℕ := 3
def either_gender_roles : ℕ := 2
def total_roles : ℕ := male_roles + either_gender_roles

def casting_combinations : ℕ := (men.choose male_roles) * ((men + women - male_roles).choose either_gender_roles)

theorem casting_theorem : casting_combinations = 15120 := by sorry

end NUMINAMATH_CALUDE_casting_theorem_l303_30315


namespace NUMINAMATH_CALUDE_fraction_meaningfulness_l303_30305

theorem fraction_meaningfulness (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 1)) ↔ x ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningfulness_l303_30305


namespace NUMINAMATH_CALUDE_segments_form_triangle_l303_30387

/-- Triangle inequality theorem: the sum of the lengths of any two sides 
    of a triangle must be greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Function to check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem stating that line segments of lengths 13, 12, and 20 can form a triangle -/
theorem segments_form_triangle : can_form_triangle 13 12 20 := by
  sorry


end NUMINAMATH_CALUDE_segments_form_triangle_l303_30387


namespace NUMINAMATH_CALUDE_john_remaining_money_l303_30303

def base7_to_base10 (n : ℕ) : ℕ :=
  6 * 7^3 + 5 * 7^2 + 3 * 7^1 + 4 * 7^0

theorem john_remaining_money :
  let savings : ℕ := base7_to_base10 6534
  let ticket_cost : ℕ := 1200
  savings - ticket_cost = 1128 := by sorry

end NUMINAMATH_CALUDE_john_remaining_money_l303_30303


namespace NUMINAMATH_CALUDE_sandwich_theorem_l303_30356

def sandwich_problem (david_spent : ℝ) (ben_spent : ℝ) : Prop :=
  ben_spent = 1.5 * david_spent ∧
  david_spent = ben_spent - 15 ∧
  david_spent + ben_spent = 75

theorem sandwich_theorem :
  ∃ (david_spent : ℝ) (ben_spent : ℝ), sandwich_problem david_spent ben_spent :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_theorem_l303_30356


namespace NUMINAMATH_CALUDE_prob_draw_star_is_one_sixth_l303_30327

/-- A deck of cards with multiple suits and ranks -/
structure Deck :=
  (num_suits : ℕ)
  (num_ranks : ℕ)

/-- The probability of drawing a specific suit from a deck -/
def prob_draw_suit (d : Deck) : ℚ :=
  1 / d.num_suits

/-- Theorem: The probability of drawing a ★ card from a deck with 6 suits and 13 ranks is 1/6 -/
theorem prob_draw_star_is_one_sixth (d : Deck) 
  (h_suits : d.num_suits = 6)
  (h_ranks : d.num_ranks = 13) :
  prob_draw_suit d = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_draw_star_is_one_sixth_l303_30327


namespace NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l303_30301

/-- The amount Adam spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Adam spent 81 dollars on the ferris wheel ride -/
theorem adam_ferris_wheel_cost :
  ferris_wheel_cost 13 4 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l303_30301


namespace NUMINAMATH_CALUDE_smallest_n_square_fifth_power_l303_30367

theorem smallest_n_square_fifth_power : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 2 * n = k^2) ∧ 
  (∃ (m : ℕ), 5 * n = m^5) ∧ 
  (∀ (x : ℕ), x > 0 → x < n → 
    (¬∃ (y : ℕ), 2 * x = y^2) ∨ 
    (¬∃ (z : ℕ), 5 * x = z^5)) ∧
  n = 5000 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_square_fifth_power_l303_30367


namespace NUMINAMATH_CALUDE_clarinet_players_count_l303_30378

/-- Represents the number of people in an orchestra section -/
structure OrchestraSection where
  count : ℕ

/-- Represents the composition of an orchestra -/
structure Orchestra where
  total : ℕ
  percussion : OrchestraSection
  brass : OrchestraSection
  strings : OrchestraSection
  flutes : OrchestraSection
  maestro : OrchestraSection
  clarinets : OrchestraSection

/-- Given an orchestra with the specified composition, prove that the number of clarinet players is 3 -/
theorem clarinet_players_count (o : Orchestra) 
  (h1 : o.total = 21)
  (h2 : o.percussion.count = 1)
  (h3 : o.brass.count = 7)
  (h4 : o.strings.count = 5)
  (h5 : o.flutes.count = 4)
  (h6 : o.maestro.count = 1)
  (h7 : o.total = o.percussion.count + o.brass.count + o.strings.count + o.flutes.count + o.maestro.count + o.clarinets.count) :
  o.clarinets.count = 3 := by
  sorry

end NUMINAMATH_CALUDE_clarinet_players_count_l303_30378


namespace NUMINAMATH_CALUDE_train_speed_l303_30376

/-- Proves that a train with given length and time to cross a pole has a specific speed -/
theorem train_speed (length : Real) (time : Real) (speed : Real) : 
  length = 400.032 →
  time = 9 →
  speed = (length / 1000) / time * 3600 →
  speed = 160.0128 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l303_30376


namespace NUMINAMATH_CALUDE_work_completion_l303_30364

/-- Given that 36 men can complete a piece of work in 18 days, and a different number of men can
    complete the same work in 24 days, prove that the number of men in the second group is 27. -/
theorem work_completion (total_work : ℕ) (men_group1 men_group2 : ℕ) (days_group1 days_group2 : ℕ) :
  men_group1 = 36 →
  days_group1 = 18 →
  days_group2 = 24 →
  total_work = men_group1 * days_group1 →
  total_work = men_group2 * days_group2 →
  men_group2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_l303_30364


namespace NUMINAMATH_CALUDE_function_composition_property_l303_30342

def iteratedFunction (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, n => n
| (i + 1), n => f (iteratedFunction f i n)

theorem function_composition_property (k : ℕ) :
  (k ≥ 2) ↔
  (∃ (f g : ℕ → ℕ),
    (∀ (S : Set ℕ), (∃ n, g n ∉ S) → Set.Infinite S) ∧
    (∀ n, iteratedFunction f (g n) n = f n + k)) :=
sorry

end NUMINAMATH_CALUDE_function_composition_property_l303_30342


namespace NUMINAMATH_CALUDE_machine_depletion_rate_l303_30394

theorem machine_depletion_rate 
  (initial_value : ℝ) 
  (final_value : ℝ) 
  (time : ℝ) 
  (h1 : initial_value = 400)
  (h2 : final_value = 225)
  (h3 : time = 2) :
  ∃ (rate : ℝ), 
    final_value = initial_value * (1 - rate) ^ time ∧ 
    rate = 0.25 := by
sorry

end NUMINAMATH_CALUDE_machine_depletion_rate_l303_30394


namespace NUMINAMATH_CALUDE_number_calculation_l303_30339

theorem number_calculation (x : ℝ) : ((x + 1.4) / 3 - 0.7) * 9 = 5.4 ↔ x = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l303_30339


namespace NUMINAMATH_CALUDE_larger_number_in_ratio_l303_30302

theorem larger_number_in_ratio (a b : ℝ) : 
  a / b = 8 / 3 → a + b = 143 → max a b = 104 := by sorry

end NUMINAMATH_CALUDE_larger_number_in_ratio_l303_30302


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l303_30382

theorem sum_of_three_numbers (A B C : ℚ) : 
  B = 30 → 
  A / B = 2 / 3 → 
  B / C = 5 / 8 → 
  A + B + C = 98 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l303_30382


namespace NUMINAMATH_CALUDE_martin_tv_purchase_l303_30398

/-- The initial amount Martin decided to spend on a TV -/
def initial_amount : ℝ := 1000

/-- The discount amount applied before the percentage discount -/
def initial_discount : ℝ := 100

/-- The percentage discount applied after the initial discount -/
def percentage_discount : ℝ := 0.20

/-- The difference between the initial amount and the final price -/
def price_difference : ℝ := 280

theorem martin_tv_purchase :
  initial_amount = 1000 ∧
  initial_amount - (initial_amount - initial_discount - 
    percentage_discount * (initial_amount - initial_discount)) = price_difference := by
  sorry

end NUMINAMATH_CALUDE_martin_tv_purchase_l303_30398


namespace NUMINAMATH_CALUDE_circle_equation_l303_30386

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - y + 6 = 0

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency to coordinate axes
def tangent_to_axes (c : Circle) : Prop :=
  c.radius = |c.center.1| ∧ c.radius = |c.center.2|

-- Define the center being on the line
def center_on_line (c : Circle) : Prop :=
  line_equation c.center.1 c.center.2

-- Define the standard equation of a circle
def standard_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation :
  ∀ c : Circle,
  tangent_to_axes c →
  center_on_line c →
  (∃ x y : ℝ, standard_equation c x y) →
  (∀ x y : ℝ, standard_equation c x y ↔ 
    ((x + 2)^2 + (y - 2)^2 = 4 ∨ (x + 6)^2 + (y + 6)^2 = 36)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l303_30386


namespace NUMINAMATH_CALUDE_log_product_equals_one_l303_30399

theorem log_product_equals_one : Real.log 2 / Real.log 5 * (Real.log 25 / Real.log 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_product_equals_one_l303_30399


namespace NUMINAMATH_CALUDE_private_schools_in_B_l303_30321

/-- Represents the three types of schools -/
inductive SchoolType
  | Public
  | Parochial
  | PrivateIndependent

/-- Represents the three districts -/
inductive District
  | A
  | B
  | C

/-- The total number of high schools -/
def total_schools : ℕ := 50

/-- The number of public schools -/
def public_schools : ℕ := 25

/-- The number of parochial schools -/
def parochial_schools : ℕ := 16

/-- The number of private independent schools -/
def private_schools : ℕ := 9

/-- The number of schools in District A -/
def schools_in_A : ℕ := 18

/-- The number of schools in District B -/
def schools_in_B : ℕ := 17

/-- Function to calculate the number of schools in District C -/
def schools_in_C : ℕ := total_schools - schools_in_A - schools_in_B

/-- Function to calculate the number of each type of school in District C -/
def schools_per_type_in_C : ℕ := schools_in_C / 3

theorem private_schools_in_B : 
  private_schools - schools_per_type_in_C = 4 := by sorry

end NUMINAMATH_CALUDE_private_schools_in_B_l303_30321


namespace NUMINAMATH_CALUDE_simplify_expression_l303_30332

theorem simplify_expression (x : ℝ) : 1 - (2 - (1 + (2 - (1 - x)))) = 1 - x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l303_30332


namespace NUMINAMATH_CALUDE_sum_remainder_of_arithmetic_sequence_l303_30343

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : List ℕ :=
  let n := (aₙ - a₁) / d + 1
  List.range n |>.map (λ i => a₁ + i * d)

theorem sum_remainder_of_arithmetic_sequence : 
  (arithmetic_sequence 3 8 283).sum % 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_of_arithmetic_sequence_l303_30343


namespace NUMINAMATH_CALUDE_x_varies_as_z_two_thirds_l303_30396

/-- Given that x varies directly as the square of y, and y varies directly as the cube root of z,
    prove that x varies as z^(2/3). -/
theorem x_varies_as_z_two_thirds
  (x y z : ℝ)
  (h1 : ∃ k : ℝ, ∀ y, x = k * y^2)
  (h2 : ∃ j : ℝ, ∀ z, y = j * z^(1/3))
  : ∃ m : ℝ, x = m * z^(2/3) :=
sorry

end NUMINAMATH_CALUDE_x_varies_as_z_two_thirds_l303_30396


namespace NUMINAMATH_CALUDE_solution_set_f_geq_0_max_value_f_range_of_m_l303_30384

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 20| - |16 - x|

-- Theorem for the solution set of f(x) ≥ 0
theorem solution_set_f_geq_0 : 
  {x : ℝ | f x ≥ 0} = {x : ℝ | x ≥ -2} := by sorry

-- Theorem for the maximum value of f(x)
theorem max_value_f : 
  ∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = 36 := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), f x ≥ m) ↔ m ≤ 36 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_0_max_value_f_range_of_m_l303_30384
