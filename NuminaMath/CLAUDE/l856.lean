import Mathlib

namespace NUMINAMATH_CALUDE_log_64_4_l856_85612

theorem log_64_4 : Real.log 4 / Real.log 64 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_64_4_l856_85612


namespace NUMINAMATH_CALUDE_max_books_is_eight_l856_85610

/-- Represents the maximum number of books borrowed by a single student in a class with the given conditions. -/
def max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) (avg_books : ℕ) : ℕ :=
  let rest_students := total_students - (zero_books + one_book + two_books)
  let total_books := total_students * avg_books
  let known_books := one_book + 2 * two_books
  let rest_books := total_books - known_books
  let min_rest_books := (rest_students - 1) * 3
  rest_books - min_rest_books

/-- Theorem stating that under the given conditions, the maximum number of books borrowed by a single student is 8. -/
theorem max_books_is_eight :
  max_books_borrowed 20 2 8 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_books_is_eight_l856_85610


namespace NUMINAMATH_CALUDE_det_roots_matrix_l856_85627

-- Define the polynomial and its roots
def polynomial (m p q : ℝ) (x : ℝ) : ℝ := x^3 - m*x^2 + p*x + q

-- Define the roots a, b, c
def roots (m p q : ℝ) : ℝ × ℝ × ℝ := 
  let (a, b, c) := sorry
  (a, b, c)

-- Define the matrix
def matrix (m p q : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  let (a, b, c) := roots m p q
  !![a, b, c; b, c, a; c, a, b]

-- Theorem statement
theorem det_roots_matrix (m p q : ℝ) :
  let (a, b, c) := roots m p q
  polynomial m p q a = 0 ∧ 
  polynomial m p q b = 0 ∧ 
  polynomial m p q c = 0 →
  Matrix.det (matrix m p q) = -3*q - m^3 + 3*m*p := by sorry

end NUMINAMATH_CALUDE_det_roots_matrix_l856_85627


namespace NUMINAMATH_CALUDE_water_consumption_in_five_hours_l856_85661

/-- The number of glasses of water consumed in a given time period. -/
def glasses_consumed (rate : ℚ) (time : ℚ) : ℚ :=
  time / rate

/-- Theorem stating that drinking a glass of water every 20 minutes for 5 hours results in 15 glasses. -/
theorem water_consumption_in_five_hours :
  glasses_consumed (20 : ℚ) (5 * 60 : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_water_consumption_in_five_hours_l856_85661


namespace NUMINAMATH_CALUDE_mean_of_remaining_two_l856_85640

def numbers : List ℕ := [1870, 1996, 2022, 2028, 2112, 2124]

theorem mean_of_remaining_two (four_numbers : List ℕ) 
  (h1 : four_numbers.length = 4)
  (h2 : four_numbers.all (· ∈ numbers))
  (h3 : (four_numbers.sum : ℚ) / 4 = 2011) :
  let remaining_two := numbers.filter (λ x => x ∉ four_numbers)
  (remaining_two.sum : ℚ) / 2 = 2054 := by
sorry

end NUMINAMATH_CALUDE_mean_of_remaining_two_l856_85640


namespace NUMINAMATH_CALUDE_problem_solution_l856_85616

theorem problem_solution (x : ℝ) (h : x = 6) : (x^6 - 17*x^3 + 72) / (x^3 - 8) = 207 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l856_85616


namespace NUMINAMATH_CALUDE_bert_grocery_fraction_l856_85660

def bert_spending (initial_amount : ℚ) (hardware_fraction : ℚ) (dry_cleaner_amount : ℚ) (final_amount : ℚ) : Prop :=
  let hardware_spent := initial_amount * hardware_fraction
  let after_hardware := initial_amount - hardware_spent
  let after_dry_cleaner := after_hardware - dry_cleaner_amount
  let grocery_spent := after_dry_cleaner - final_amount
  grocery_spent / after_dry_cleaner = 1/2

theorem bert_grocery_fraction :
  bert_spending 44 (1/4) 9 12 :=
by
  sorry

end NUMINAMATH_CALUDE_bert_grocery_fraction_l856_85660


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l856_85606

/-- The point corresponding to the complex number (a^2 - 4a + 5) + (-b^2 + 2b - 6)i 
    is in the fourth quadrant for all real a and b. -/
theorem complex_number_in_fourth_quadrant (a b : ℝ) : 
  (a^2 - 4*a + 5 > 0) ∧ (-b^2 + 2*b - 6 < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l856_85606


namespace NUMINAMATH_CALUDE_factorial_sum_equals_36018_l856_85695

theorem factorial_sum_equals_36018 : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 3 * Nat.factorial 3 + Nat.factorial 5 = 36018 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equals_36018_l856_85695


namespace NUMINAMATH_CALUDE_tangent_line_minimum_b_l856_85674

/-- Given a > 0 and y = 2x + b is tangent to y = 2a ln x, the minimum value of b is -2 -/
theorem tangent_line_minimum_b (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, (2 * x + b = 2 * a * Real.log x) ∧ 
             (∀ y : ℝ, y ≠ x → 2 * y + b > 2 * a * Real.log y)) → 
  (∀ c : ℝ, (∃ x : ℝ, (2 * x + c = 2 * a * Real.log x) ∧ 
                       (∀ y : ℝ, y ≠ x → 2 * y + c > 2 * a * Real.log y)) → 
            c ≥ -2) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_minimum_b_l856_85674


namespace NUMINAMATH_CALUDE_range_of_a_l856_85697

/-- Proposition p -/
def p (x : ℝ) : Prop := (2*x - 1) / (x - 1) < 0

/-- Proposition q -/
def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0

/-- The main theorem -/
theorem range_of_a (a : ℝ) : 
  (∀ x, p x ↔ q x a) → 0 ≤ a ∧ a ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l856_85697


namespace NUMINAMATH_CALUDE_modulus_of_one_plus_three_i_l856_85656

theorem modulus_of_one_plus_three_i : Complex.abs (1 + 3 * Complex.I) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_one_plus_three_i_l856_85656


namespace NUMINAMATH_CALUDE_least_valid_k_l856_85676

def sum_of_digits (n : ℕ) : ℕ := sorry

def is_valid_k (k : ℤ) : Prop :=
  (0.00010101 * (10 : ℝ) ^ k > 100) ∧
  (sum_of_digits k.natAbs ≤ 15)

def exists_valid_m : Prop :=
  ∃ m : ℤ, 0.000515151 * (10 : ℝ) ^ m ≤ 500

theorem least_valid_k :
  is_valid_k 7 ∧ exists_valid_m ∧
  ∀ k : ℤ, k < 7 → ¬(is_valid_k k) :=
sorry

end NUMINAMATH_CALUDE_least_valid_k_l856_85676


namespace NUMINAMATH_CALUDE_nine_sided_figure_perimeter_l856_85652

/-- The perimeter of a regular polygon with n sides of length s is n * s -/
def perimeter (n : ℕ) (s : ℝ) : ℝ := n * s

theorem nine_sided_figure_perimeter :
  let n : ℕ := 9
  let s : ℝ := 2
  perimeter n s = 18 := by sorry

end NUMINAMATH_CALUDE_nine_sided_figure_perimeter_l856_85652


namespace NUMINAMATH_CALUDE_hyperbola_foci_l856_85645

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

-- Define the foci of the hyperbola
def foci : Set (ℝ × ℝ) :=
  {(5, 0), (-5, 0)}

-- Theorem statement
theorem hyperbola_foci :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ foci :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l856_85645


namespace NUMINAMATH_CALUDE_number_of_trailing_zeros_l856_85662

theorem number_of_trailing_zeros : ∃ n : ℕ, (10^100 * 100^10 : ℕ) = n * 10^120 ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_number_of_trailing_zeros_l856_85662


namespace NUMINAMATH_CALUDE_product_of_distinct_solutions_l856_85687

theorem product_of_distinct_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 2 / x = y + 2 / y) : x * y = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_solutions_l856_85687


namespace NUMINAMATH_CALUDE_max_eggs_per_basket_l856_85683

def red_eggs : ℕ := 15
def blue_eggs : ℕ := 30
def min_eggs_per_basket : ℕ := 3

def is_valid_distribution (eggs_per_basket : ℕ) : Prop :=
  eggs_per_basket ≥ min_eggs_per_basket ∧
  red_eggs % eggs_per_basket = 0 ∧
  blue_eggs % eggs_per_basket = 0

theorem max_eggs_per_basket :
  ∃ (max : ℕ), is_valid_distribution max ∧
    ∀ (n : ℕ), is_valid_distribution n → n ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_eggs_per_basket_l856_85683


namespace NUMINAMATH_CALUDE_divisors_of_prime_products_l856_85698

theorem divisors_of_prime_products (p q : ℕ) (m n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hpq : p ≠ q) :
  let num_divisors := fun x => (Nat.divisors x).card
  (num_divisors (p * q) = 4) ∧
  (num_divisors (p^2 * q) = 6) ∧
  (num_divisors (p^2 * q^2) = 9) ∧
  (num_divisors (p^m * q^n) = (m + 1) * (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_prime_products_l856_85698


namespace NUMINAMATH_CALUDE_parallelogram_area_l856_85629

theorem parallelogram_area (side1 side2 : ℝ) (angle : ℝ) :
  side1 = 7 →
  side2 = 12 →
  angle = Real.pi / 3 →
  side2 * side1 * Real.sin angle = 12 * 7 * Real.sin (Real.pi / 3) :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l856_85629


namespace NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l856_85639

-- Define the piecewise function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then a * x + 5
  else if x ≥ -2 ∧ x ≤ 2 then x - 7
  else 3 * x - b

-- State the theorem
theorem continuous_piecewise_function_sum (a b : ℝ) :
  Continuous (f a b) → a + b = -2 := by sorry

end NUMINAMATH_CALUDE_continuous_piecewise_function_sum_l856_85639


namespace NUMINAMATH_CALUDE_project_work_time_l856_85647

/-- Calculates the time spent working on a project given the project duration and nap information -/
def time_spent_working (project_days : ℕ) (num_naps : ℕ) (nap_duration : ℕ) : ℕ :=
  let total_hours := project_days * 24
  let total_nap_hours := num_naps * nap_duration
  total_hours - total_nap_hours

/-- Proves that given a 4-day project and 6 seven-hour naps, the time spent working is 54 hours -/
theorem project_work_time : time_spent_working 4 6 7 = 54 := by
  sorry

end NUMINAMATH_CALUDE_project_work_time_l856_85647


namespace NUMINAMATH_CALUDE_marks_candy_bars_l856_85681

def total_candy_bars (snickers mars butterfingers : ℕ) : ℕ :=
  snickers + mars + butterfingers

theorem marks_candy_bars : total_candy_bars 3 2 7 = 12 := by
  sorry

end NUMINAMATH_CALUDE_marks_candy_bars_l856_85681


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l856_85622

theorem smallest_x_absolute_value_equation : 
  (∃ x : ℝ, 2 * |x - 10| = 24) ∧ 
  (∀ x : ℝ, 2 * |x - 10| = 24 → x ≥ -2) ∧
  (2 * |-2 - 10| = 24) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l856_85622


namespace NUMINAMATH_CALUDE_binomial_100_100_l856_85657

theorem binomial_100_100 : Nat.choose 100 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_100_100_l856_85657


namespace NUMINAMATH_CALUDE_abs_five_minus_e_equals_five_minus_e_l856_85641

theorem abs_five_minus_e_equals_five_minus_e :
  |5 - Real.exp 1| = 5 - Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_abs_five_minus_e_equals_five_minus_e_l856_85641


namespace NUMINAMATH_CALUDE_probability_score_at_most_seven_l856_85659

/-- The probability of scoring at most 7 points when drawing 4 balls from a bag containing 4 red balls (1 point each) and 3 black balls (3 points each) -/
theorem probability_score_at_most_seven (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) (red_score : ℕ) (black_score : ℕ) :
  total_balls = red_balls + black_balls →
  red_balls = 4 →
  black_balls = 3 →
  drawn_balls = 4 →
  red_score = 1 →
  black_score = 3 →
  (Nat.choose total_balls drawn_balls : ℚ) * (13 : ℚ) / (35 : ℚ) = 
    (Nat.choose red_balls drawn_balls : ℚ) + 
    (Nat.choose red_balls (drawn_balls - 1) : ℚ) * (Nat.choose black_balls 1 : ℚ) :=
by sorry

#check probability_score_at_most_seven

end NUMINAMATH_CALUDE_probability_score_at_most_seven_l856_85659


namespace NUMINAMATH_CALUDE_polynomial_degree_bound_l856_85626

theorem polynomial_degree_bound (m n k : ℕ) (P : Polynomial ℤ) :
  m > 0 →
  n > 0 →
  k ≥ 2 →
  (∀ i, Odd (P.coeff i)) →
  P.degree = n →
  (X - 1 : Polynomial ℤ) ^ m ∣ P →
  m ≥ 2^k →
  n ≥ 2^(k+1) - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_bound_l856_85626


namespace NUMINAMATH_CALUDE_octahedron_sum_l856_85600

/-- Represents an octahedron -/
structure Octahedron where
  edges : ℕ
  vertices : ℕ
  faces : ℕ

/-- The sum of edges, vertices, and faces of an octahedron is 26 -/
theorem octahedron_sum : ∀ (o : Octahedron), o.edges + o.vertices + o.faces = 26 := by
  sorry

end NUMINAMATH_CALUDE_octahedron_sum_l856_85600


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l856_85625

theorem fraction_equation_solution :
  ∃ x : ℚ, x - 2 ≠ 0 ∧ (2 / (x - 2) = (1 + x) / (x - 2) + 1) ∧ x = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l856_85625


namespace NUMINAMATH_CALUDE_mem_not_zeige_l856_85635

-- Define our universes
variable (U : Type)

-- Define our sets
variable (Mem Enform Zeige : Set U)

-- State our premises
variable (h1 : Mem ⊆ Enform)
variable (h2 : Enform ∩ Zeige = ∅)

-- State our theorem
theorem mem_not_zeige :
  (∀ x, x ∈ Mem → x ∉ Zeige) ∧
  (Mem ∩ Zeige = ∅) :=
sorry

end NUMINAMATH_CALUDE_mem_not_zeige_l856_85635


namespace NUMINAMATH_CALUDE_parabola_directrix_l856_85631

/-- The directrix of the parabola y = -1/4 * x^2 is y = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  y = -1/4 * x^2 → 
  (∀ (x₀ y₀ : ℝ), y₀ = -1/4 * x₀^2 → 
    (x₀ - x)^2 + (y₀ - y)^2 = (y₀ - 1)^2) → 
  y = 1 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l856_85631


namespace NUMINAMATH_CALUDE_johny_travel_distance_l856_85605

theorem johny_travel_distance (S : ℝ) : 
  S ≥ 0 →
  S + (S + 20) + 2*(S + 20) = 220 →
  S = 40 := by
sorry

end NUMINAMATH_CALUDE_johny_travel_distance_l856_85605


namespace NUMINAMATH_CALUDE_problem_1_l856_85658

theorem problem_1 (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1) : 
  2 / (a + 1) - (a - 2) / (a^2 - 1) / ((a^2 - 2*a) / (a^2 - 2*a + 1)) = 1 / a := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l856_85658


namespace NUMINAMATH_CALUDE_total_pencils_l856_85632

/-- Given the ages and pencil counts of Asaf and Alexander, prove their total pencil count -/
theorem total_pencils (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ) : 
  asaf_age = 50 →
  asaf_age + alexander_age = 140 →
  alexander_age - asaf_age = asaf_pencils / 2 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 := by
sorry


end NUMINAMATH_CALUDE_total_pencils_l856_85632


namespace NUMINAMATH_CALUDE_arcsin_plus_arccos_eq_pi_sixth_l856_85677

theorem arcsin_plus_arccos_eq_pi_sixth (x : ℝ) :
  Real.arcsin x + Real.arccos (3 * x) = π / 6 → x = Real.sqrt (3 / 124) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_plus_arccos_eq_pi_sixth_l856_85677


namespace NUMINAMATH_CALUDE_polygon_interior_angles_l856_85673

theorem polygon_interior_angles (n : ℕ) (h1 : n > 0) : 
  (n - 2) * 180 = n * 177 → n = 120 := by sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_l856_85673


namespace NUMINAMATH_CALUDE_rachel_score_l856_85666

/-- Rachel's video game scoring system -/
structure GameScore where
  points_per_treasure : ℕ
  treasures_level1 : ℕ
  treasures_level2 : ℕ

/-- Calculate the total score for Rachel's game -/
def total_score (game : GameScore) : ℕ :=
  game.points_per_treasure * (game.treasures_level1 + game.treasures_level2)

/-- Theorem: Rachel's total score is 63 points -/
theorem rachel_score :
  ∀ (game : GameScore),
  game.points_per_treasure = 9 →
  game.treasures_level1 = 5 →
  game.treasures_level2 = 2 →
  total_score game = 63 :=
by
  sorry

end NUMINAMATH_CALUDE_rachel_score_l856_85666


namespace NUMINAMATH_CALUDE_equality_holds_iff_l856_85653

theorem equality_holds_iff (α : ℝ) : 
  Real.sqrt (1 + Real.sin (2 * α)) = Real.sin α + Real.cos α ↔ 
  -π/4 < α ∧ α < 3*π/4 :=
sorry

end NUMINAMATH_CALUDE_equality_holds_iff_l856_85653


namespace NUMINAMATH_CALUDE_line_equation_problem_l856_85699

/-- Two lines are the same if their coefficients are proportional -/
def same_line (a b c : ℝ) (d e f : ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ k * a = d ∧ k * b = e ∧ k * c = f

/-- The problem statement -/
theorem line_equation_problem (p q : ℝ) :
  same_line p 2 7 3 q 5 → p = 21/5 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_problem_l856_85699


namespace NUMINAMATH_CALUDE_luis_red_socks_l856_85692

/-- The number of pairs of red socks Luis bought -/
def red_socks : ℕ := sorry

/-- The number of pairs of blue socks Luis bought -/
def blue_socks : ℕ := 6

/-- The cost of each pair of red socks in dollars -/
def red_sock_cost : ℕ := 3

/-- The cost of each pair of blue socks in dollars -/
def blue_sock_cost : ℕ := 5

/-- The total amount Luis spent in dollars -/
def total_spent : ℕ := 42

/-- Theorem stating that Luis bought 4 pairs of red socks -/
theorem luis_red_socks : 
  red_socks * red_sock_cost + blue_socks * blue_sock_cost = total_spent → 
  red_socks = 4 := by sorry

end NUMINAMATH_CALUDE_luis_red_socks_l856_85692


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l856_85663

theorem radical_conjugate_sum_product (c d : ℝ) 
  (h1 : (c + Real.sqrt d) + (c - Real.sqrt d) = -6)
  (h2 : (c + Real.sqrt d) * (c - Real.sqrt d) = 1) :
  4 * c + d = -4 := by
  sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l856_85663


namespace NUMINAMATH_CALUDE_spring_spending_is_1_7_l856_85689

/-- The spending of Rivertown government in millions of dollars -/
structure RivertownSpending where
  /-- Total accumulated spending by the end of February -/
  february_end : ℝ
  /-- Total accumulated spending by the end of May -/
  may_end : ℝ

/-- The spending during March, April, and May -/
def spring_spending (s : RivertownSpending) : ℝ :=
  s.may_end - s.february_end

theorem spring_spending_is_1_7 (s : RivertownSpending) 
  (h_feb : s.february_end = 0.8)
  (h_may : s.may_end = 2.5) : 
  spring_spending s = 1.7 := by
  sorry

end NUMINAMATH_CALUDE_spring_spending_is_1_7_l856_85689


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l856_85624

theorem triangle_ABC_properties (A B C : ℝ) (p : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle ABC exists
  (∃ x y : ℝ, x^2 + (x+1)*p + 1 = 0 ∧ y^2 + (y+1)*p + 1 = 0 ∧ x = Real.tan A ∧ y = Real.tan B) →
  C = 3*π/4 ∧ p ∈ Set.Ioo (-2 : ℝ) (2 - 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l856_85624


namespace NUMINAMATH_CALUDE_cos_beta_minus_alpha_l856_85664

theorem cos_beta_minus_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < 2 * π)
  (h4 : 5 * Real.sin (α - π / 6) = 1) (h5 : 5 * Real.sin (β - π / 6) = 1) :
  Real.cos (β - α) = -23 / 25 := by
sorry

end NUMINAMATH_CALUDE_cos_beta_minus_alpha_l856_85664


namespace NUMINAMATH_CALUDE_minimum_excellent_all_exams_l856_85680

theorem minimum_excellent_all_exams (total_students : ℕ) 
  (excellent_first : ℕ) (excellent_second : ℕ) (excellent_third : ℕ) 
  (h_total : total_students = 200)
  (h_first : excellent_first = (80 : ℝ) / 100 * total_students)
  (h_second : excellent_second = (70 : ℝ) / 100 * total_students)
  (h_third : excellent_third = (59 : ℝ) / 100 * total_students) :
  ∃ (excellent_all : ℕ), 
    excellent_all ≥ 18 ∧ 
    (∀ (n : ℕ), n < excellent_all → 
      ∃ (m1 m2 m3 m12 m13 m23 : ℕ),
        m1 + m2 + m3 + m12 + m13 + m23 + n > total_students ∨
        m1 + m12 + m13 + n > excellent_first ∨
        m2 + m12 + m23 + n > excellent_second ∨
        m3 + m13 + m23 + n > excellent_third) :=
sorry

end NUMINAMATH_CALUDE_minimum_excellent_all_exams_l856_85680


namespace NUMINAMATH_CALUDE_friends_carrying_bananas_l856_85611

theorem friends_carrying_bananas (total_friends : ℕ) (pears oranges apples : ℕ) : 
  total_friends = 35 →
  pears = 14 →
  oranges = 8 →
  apples = 5 →
  total_friends = pears + oranges + apples + (total_friends - (pears + oranges + apples)) →
  total_friends - (pears + oranges + apples) = 8 :=
by sorry

end NUMINAMATH_CALUDE_friends_carrying_bananas_l856_85611


namespace NUMINAMATH_CALUDE_odd_product_minus_one_divisible_by_four_l856_85678

theorem odd_product_minus_one_divisible_by_four (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c) : 
  (4 ∣ a * b - 1) ∨ (4 ∣ b * c - 1) ∨ (4 ∣ c * a - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_product_minus_one_divisible_by_four_l856_85678


namespace NUMINAMATH_CALUDE_odd_function_sum_condition_l856_85617

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum_condition (f : ℝ → ℝ) (h : OddFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧
  ¬(∀ x₁ x₂ : ℝ, f x₁ + f x₂ = 0 → x₁ + x₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_sum_condition_l856_85617


namespace NUMINAMATH_CALUDE_integral_inequality_l856_85603

-- Define a non-decreasing function on [0,∞)
def NonDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

-- State the theorem
theorem integral_inequality
  (f : ℝ → ℝ)
  (h_nondec : NonDecreasing f)
  {x y z : ℝ}
  (h_x : 0 ≤ x)
  (h_xy : x < y)
  (h_yz : y < z) :
  (z - x) * ∫ u in y..z, f u ≥ (z - y) * ∫ u in x..z, f u :=
sorry

end NUMINAMATH_CALUDE_integral_inequality_l856_85603


namespace NUMINAMATH_CALUDE_sum_a_b_equals_14_l856_85679

theorem sum_a_b_equals_14 (a b c d : ℝ) 
  (h1 : b + c = 9) 
  (h2 : c + d = 3) 
  (h3 : a + d = 8) : 
  a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_14_l856_85679


namespace NUMINAMATH_CALUDE_prob_all_red_before_both_green_is_one_third_l856_85671

/-- The number of red chips in the hat -/
def num_red : ℕ := 4

/-- The number of green chips in the hat -/
def num_green : ℕ := 2

/-- The total number of chips in the hat -/
def total_chips : ℕ := num_red + num_green

/-- The probability of drawing all red chips before both green chips -/
def prob_all_red_before_both_green : ℚ :=
  (total_chips - 1).choose num_green / total_chips.choose num_green

theorem prob_all_red_before_both_green_is_one_third :
  prob_all_red_before_both_green = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_all_red_before_both_green_is_one_third_l856_85671


namespace NUMINAMATH_CALUDE_total_entertainment_cost_l856_85646

def computer_game_cost : ℕ := 66
def movie_ticket_cost : ℕ := 12
def number_of_tickets : ℕ := 3

theorem total_entertainment_cost : 
  computer_game_cost + number_of_tickets * movie_ticket_cost = 102 := by
  sorry

end NUMINAMATH_CALUDE_total_entertainment_cost_l856_85646


namespace NUMINAMATH_CALUDE_area_DEF_value_l856_85607

/-- Triangle ABC with sides 5, 12, and 13 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (side_a : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 5)
  (side_b : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 12)
  (side_c : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13)

/-- Parabola with focus F and directrix L -/
structure Parabola :=
  (F : ℝ × ℝ)
  (L : Set (ℝ × ℝ))

/-- Intersection points of parabolas with triangle sides -/
structure Intersections (t : Triangle) :=
  (A1 A2 B1 B2 C1 C2 : ℝ × ℝ)
  (on_parabola_A : Parabola → Prop)
  (on_parabola_B : Parabola → Prop)
  (on_parabola_C : Parabola → Prop)

/-- The area of triangle DEF formed by A1C2, B1A2, and C1B2 -/
def area_DEF (t : Triangle) (i : Intersections t) : ℝ := sorry

/-- Main theorem: The area of triangle DEF is 6728/3375 -/
theorem area_DEF_value (t : Triangle) (i : Intersections t) : 
  area_DEF t i = 6728 / 3375 := by sorry

end NUMINAMATH_CALUDE_area_DEF_value_l856_85607


namespace NUMINAMATH_CALUDE_square_diff_ratio_equals_one_third_l856_85619

theorem square_diff_ratio_equals_one_third :
  (2025^2 - 2018^2) / (2032^2 - 2011^2) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_square_diff_ratio_equals_one_third_l856_85619


namespace NUMINAMATH_CALUDE_wheel_spinner_probability_l856_85642

theorem wheel_spinner_probability (p_E p_F p_G p_H p_I : ℚ) : 
  p_E = 1/5 →
  p_F = 1/4 →
  p_G = p_H →
  p_E + p_F + p_G + p_H + p_I = 1 →
  p_H = 9/40 := by
sorry

end NUMINAMATH_CALUDE_wheel_spinner_probability_l856_85642


namespace NUMINAMATH_CALUDE_f_is_smallest_not_on_board_l856_85682

/-- The game function that represents the number left on the board after subtraction -/
def g (k : ℕ) (x : ℕ) : ℕ := x^2 - k

/-- The smallest integer a such that g_{2n}(a) - g_{2n}(a-1) ≥ 3 -/
def x (n : ℕ) : ℕ := 2*n + 2

/-- The function f(2n) representing the smallest positive integer not written on the board -/
def f (n : ℕ) : ℕ := (2*n + 1)^2 - 2*n

/-- Theorem stating that f(2n) is the smallest positive integer not written on the board -/
theorem f_is_smallest_not_on_board (n : ℕ) :
  f n = (2*n + 1)^2 - 2*n ∧
  ∀ m < f n, ∃ i ≤ x n, m = g (2*n) i ∨ m = g (2*n) (i+1) :=
sorry

end NUMINAMATH_CALUDE_f_is_smallest_not_on_board_l856_85682


namespace NUMINAMATH_CALUDE_triangle_inequality_minimum_l856_85615

theorem triangle_inequality_minimum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htri : a + b ≥ c ∧ b + c ≥ a ∧ c + a ≥ b) :
  c / (a + b) + b / c ≥ Real.sqrt 2 - 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_minimum_l856_85615


namespace NUMINAMATH_CALUDE_intersection_length_l856_85694

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 5
def circle_O₂ (x y m : ℝ) : Prop := (x + m)^2 + y^2 = 20

-- Define the intersection points
structure IntersectionPoints (m : ℝ) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h₁ : circle_O₁ A.1 A.2
  h₂ : circle_O₂ A.1 A.2 m
  h₃ : circle_O₁ B.1 B.2
  h₄ : circle_O₂ B.1 B.2 m

-- Define the perpendicular tangents condition
def perpendicular_tangents (m : ℝ) (A : ℝ × ℝ) : Prop :=
  circle_O₁ A.1 A.2 ∧ circle_O₂ A.1 A.2 m ∧
  (∃ t₁ t₂ : ℝ × ℝ, 
    (t₁.1 * t₂.1 + t₁.2 * t₂.2 = 0) ∧  -- Perpendicular condition
    (t₁.1 * A.1 + t₁.2 * A.2 = 0) ∧    -- Tangent to O₁
    (t₂.1 * (A.1 + m) + t₂.2 * A.2 = 0)) -- Tangent to O₂

-- Theorem statement
theorem intersection_length (m : ℝ) (points : IntersectionPoints m) :
  perpendicular_tangents m points.A →
  Real.sqrt ((points.A.1 - points.B.1)^2 + (points.A.2 - points.B.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_intersection_length_l856_85694


namespace NUMINAMATH_CALUDE_inequality_proof_l856_85614

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2)/(x^2 + y*z) + (y^2 + 2*z^2 + 2*x^2)/(y^2 + z*x) + (z^2 + 2*x^2 + 2*y^2)/(z^2 + x*y) > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l856_85614


namespace NUMINAMATH_CALUDE_constant_volume_l856_85613

/-- Represents a line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Represents the configuration of the tetrahedron with moving vertices -/
structure MovingTetrahedron where
  fixedEdge : Line3D
  movingVertex1 : Line3D
  movingVertex2 : Line3D
  initialTetrahedron : Tetrahedron

/-- Checks if three lines are parallel -/
def areLinesParallel (l1 l2 l3 : Line3D) : Prop := sorry

/-- Calculates the tetrahedron at a given time t -/
def tetrahedronAtTime (mt : MovingTetrahedron) (t : ℝ) : Tetrahedron := sorry

/-- Theorem stating that the volume remains constant -/
theorem constant_volume (mt : MovingTetrahedron) 
  (h : areLinesParallel mt.fixedEdge mt.movingVertex1 mt.movingVertex2) :
  ∀ t : ℝ, tetrahedronVolume (tetrahedronAtTime mt t) = tetrahedronVolume mt.initialTetrahedron :=
sorry

end NUMINAMATH_CALUDE_constant_volume_l856_85613


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l856_85650

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop :=
  y - (focus.2) = m * (x - (focus.1))

-- Define the theorem
theorem parabola_intersection_theorem 
  (A B C : ℝ × ℝ) 
  (m : ℝ) 
  (h1 : parabola A.1 A.2)
  (h2 : parabola B.1 B.2)
  (h3 : directrix C.1)
  (h4 : line_through_focus m A.1 A.2)
  (h5 : line_through_focus m B.1 B.2)
  (h6 : line_through_focus m C.1 C.2)
  (h7 : A.2 * C.2 ≥ 0)  -- A and C on the same side of x-axis
  (h8 : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 
        2 * Real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2)) :
  Real.sqrt ((B.1 - focus.1)^2 + (B.2 - focus.2)^2) = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l856_85650


namespace NUMINAMATH_CALUDE_interest_rate_multiple_l856_85618

theorem interest_rate_multiple (P r m : ℝ) 
  (h1 : P * r^2 = 40)
  (h2 : P * (m * r)^2 = 360) :
  m = 3 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_multiple_l856_85618


namespace NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_l856_85628

theorem sin_cos_sum_fifteen_seventyfive : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (75 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_fifteen_seventyfive_l856_85628


namespace NUMINAMATH_CALUDE_sector_area_l856_85630

/-- Given a sector with perimeter 8 and central angle 2 radians, its area is 4 -/
theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (radius : ℝ) (arc_length : ℝ) :
  perimeter = 8 →
  central_angle = 2 →
  perimeter = arc_length + 2 * radius →
  arc_length = radius * central_angle →
  (1 / 2) * radius * arc_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l856_85630


namespace NUMINAMATH_CALUDE_four_digit_number_divisibility_l856_85688

theorem four_digit_number_divisibility (a b c d : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let M := 1000 * a + 100 * b + 10 * c + d
  let N := 1000 * d + 100 * c + 10 * b + a
  (101 ∣ (M + N)) → a + d = b + c :=
by sorry

end NUMINAMATH_CALUDE_four_digit_number_divisibility_l856_85688


namespace NUMINAMATH_CALUDE_total_pages_in_paper_l856_85690

/-- Represents the number of days Stacy has to complete her history paper -/
def days_to_complete : ℕ := 7

/-- Represents the number of pages Stacy needs to write per day -/
def pages_per_day : ℕ := 9

/-- Theorem stating that the total number of pages in Stacy's history paper is 63 -/
theorem total_pages_in_paper : days_to_complete * pages_per_day = 63 := by
  sorry

end NUMINAMATH_CALUDE_total_pages_in_paper_l856_85690


namespace NUMINAMATH_CALUDE_triangle_circles_area_sum_l856_85609

/-- The sum of the areas of three mutually externally tangent circles 
    centered at the vertices of a 6-8-10 right triangle is 56π. -/
theorem triangle_circles_area_sum : 
  ∀ (r s t : ℝ), 
    r + s = 6 →
    r + t = 8 →
    s + t = 10 →
    r > 0 → s > 0 → t > 0 →
    π * (r^2 + s^2 + t^2) = 56 * π := by
  sorry

end NUMINAMATH_CALUDE_triangle_circles_area_sum_l856_85609


namespace NUMINAMATH_CALUDE_triangle_area_change_l856_85654

theorem triangle_area_change (base height : ℝ) (base_new height_new area area_new : ℝ) 
  (h1 : base_new = 1.4 * base) 
  (h2 : height_new = 0.6 * height) 
  (h3 : area = (base * height) / 2) 
  (h4 : area_new = (base_new * height_new) / 2) : 
  area_new = 0.42 * area := by
sorry

end NUMINAMATH_CALUDE_triangle_area_change_l856_85654


namespace NUMINAMATH_CALUDE_kathleen_savings_problem_l856_85638

/-- Kathleen's savings and spending problem -/
theorem kathleen_savings_problem (june july august clothes_cost remaining : ℕ)
  (h_june : june = 21)
  (h_july : july = 46)
  (h_august : august = 45)
  (h_clothes : clothes_cost = 54)
  (h_remaining : remaining = 46) :
  ∃ (school_supplies : ℕ),
    june + july + august = clothes_cost + school_supplies + remaining ∧ 
    school_supplies = 12 := by
  sorry

end NUMINAMATH_CALUDE_kathleen_savings_problem_l856_85638


namespace NUMINAMATH_CALUDE_fraction_transformation_l856_85693

theorem fraction_transformation (a b c d x : ℚ) 
  (h1 : a ≠ b) 
  (h2 : b ≠ 0) 
  (h3 : (a + x) / (b + x) = c / d) : 
  x = (a * d - b * c) / (c - d) := by
sorry

end NUMINAMATH_CALUDE_fraction_transformation_l856_85693


namespace NUMINAMATH_CALUDE_remainder_theorem_l856_85669

theorem remainder_theorem (n : ℤ) (k : ℤ) (h : n = 30 * k - 1) :
  (n^2 + 2*n + n^3 + 3) % 30 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l856_85669


namespace NUMINAMATH_CALUDE_prob_A_misses_at_least_once_prob_A_hits_twice_B_hits_thrice_l856_85633

-- Define the probabilities of hitting the target for A and B
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define the number of shots
def num_shots : ℕ := 4

-- Theorem for the first question
theorem prob_A_misses_at_least_once :
  1 - prob_A_hit ^ num_shots = 65/81 :=
sorry

-- Theorem for the second question
theorem prob_A_hits_twice_B_hits_thrice :
  (Nat.choose num_shots 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit)^(num_shots - 2) *
  (Nat.choose num_shots 3 : ℚ) * prob_B_hit^3 * (1 - prob_B_hit)^(num_shots - 3) = 1/8 :=
sorry

end NUMINAMATH_CALUDE_prob_A_misses_at_least_once_prob_A_hits_twice_B_hits_thrice_l856_85633


namespace NUMINAMATH_CALUDE_xy_squared_change_l856_85675

/-- Theorem: Change in xy^2 when x increases by 20% and y decreases by 30% --/
theorem xy_squared_change (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let x' := 1.2 * x
  let y' := 0.7 * y
  1 - (x' * y' * y') / (x * y * y) = 0.412 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_change_l856_85675


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l856_85672

theorem polynomial_identity_sum_of_squares : 
  ∀ (a b c d e f : ℤ), 
  (∀ x : ℝ, 216 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 2180 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_of_squares_l856_85672


namespace NUMINAMATH_CALUDE_customers_left_l856_85604

theorem customers_left (initial : ℕ) (additional : ℕ) (final : ℕ) : 
  initial = 47 → additional = 20 → final = 26 → initial - (initial - additional + final) = 41 := by
sorry

end NUMINAMATH_CALUDE_customers_left_l856_85604


namespace NUMINAMATH_CALUDE_weight_replacement_l856_85691

theorem weight_replacement (n : ℕ) (avg_increase weight_new : ℝ) :
  n = 9 ∧ 
  avg_increase = 5.5 ∧
  weight_new = 135.5 →
  (n * avg_increase + weight_new - n * avg_increase) = 86 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l856_85691


namespace NUMINAMATH_CALUDE_value_of_expression_l856_85623

theorem value_of_expression (a : ℝ) (h : a - 1/a = 1) : a^2 - a + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l856_85623


namespace NUMINAMATH_CALUDE_trajectory_of_moving_circle_l856_85602

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def C1 : Circle := { center := (0, -1), radius := 5 }
def C2 : Circle := { center := (0, 2), radius := 1 }

def is_tangent_inside (M : ℝ × ℝ) (C : Circle) : Prop :=
  (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = (C.radius - 1)^2

def is_tangent_outside (M : ℝ × ℝ) (C : Circle) : Prop :=
  (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = (C.radius + 1)^2

theorem trajectory_of_moving_circle (x y : ℝ) :
  is_tangent_inside (x, y) C1 → is_tangent_outside (x, y) C2 →
  y ≠ 3 → y^2/9 + x^2/5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_moving_circle_l856_85602


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l856_85686

theorem cubic_sum_theorem (x y : ℝ) (h1 : y^2 - 3 = (x - 3)^3) (h2 : x^2 - 3 = (y - 3)^2) (h3 : x ≠ y) :
  x^3 + y^3 = 27 + 3 * Real.sqrt 3 ∨ x^3 + y^3 = 27 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l856_85686


namespace NUMINAMATH_CALUDE_sqrt_ten_plus_three_squared_times_sqrt_ten_minus_three_l856_85665

theorem sqrt_ten_plus_three_squared_times_sqrt_ten_minus_three :
  (Real.sqrt 10 + 3)^2 * (Real.sqrt 10 - 3) = Real.sqrt 10 + 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ten_plus_three_squared_times_sqrt_ten_minus_three_l856_85665


namespace NUMINAMATH_CALUDE_equation_solution_l856_85643

theorem equation_solution :
  ∃ x : ℚ, (8 * x^2 + 150 * x + 2) / (3 * x + 50) = 4 * x + 2 ∧ x = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l856_85643


namespace NUMINAMATH_CALUDE_flippy_divisible_by_four_l856_85608

/-- A four-digit number is flippy if its digits alternate between two distinct digits from the set {4, 6} -/
def is_flippy (n : ℕ) : Prop :=
  (n ≥ 1000 ∧ n < 10000) ∧
  (∃ a b : ℕ, (a = 4 ∨ a = 6) ∧ (b = 4 ∨ b = 6) ∧ a ≠ b ∧
   ((n = 1000 * a + 100 * b + 10 * a + b) ∨
    (n = 1000 * b + 100 * a + 10 * b + a)))

theorem flippy_divisible_by_four :
  ∃! n : ℕ, is_flippy n ∧ n % 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_flippy_divisible_by_four_l856_85608


namespace NUMINAMATH_CALUDE_problem_1_l856_85651

theorem problem_1 : |-2| - 8 / (-2) / (-1/2) = -6 := by sorry

end NUMINAMATH_CALUDE_problem_1_l856_85651


namespace NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_contradictory_l856_85620

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The set of all possible outcomes when drawing two balls from the pocket -/
def allOutcomes : Finset DrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Black ∧ outcome.second = BallColor.Red) ∨
  (outcome.first = BallColor.Red ∧ outcome.second = BallColor.Black)

/-- The event of drawing exactly two black balls -/
def exactlyTwoBlack (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.Black ∧ outcome.second = BallColor.Black

/-- The theorem stating that "Exactly one black ball" and "Exactly two black balls" are mutually exclusive but not contradictory -/
theorem exactly_one_two_black_mutually_exclusive_not_contradictory :
  (∀ outcome : DrawOutcome, ¬(exactlyOneBlack outcome ∧ exactlyTwoBlack outcome)) ∧
  (∃ outcome : DrawOutcome, exactlyOneBlack outcome) ∧
  (∃ outcome : DrawOutcome, exactlyTwoBlack outcome) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_two_black_mutually_exclusive_not_contradictory_l856_85620


namespace NUMINAMATH_CALUDE_function_equation_solution_l856_85696

theorem function_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2) - f (y^2) + 2*x + 1 = f (x + y) * f (x - y)) : 
  (∀ x : ℝ, f x = x + 1) ∨ (∀ x : ℝ, f x = -x - 1) := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l856_85696


namespace NUMINAMATH_CALUDE_meaningful_range_for_sqrt_fraction_l856_85644

/-- The range of x for which the expression sqrt(x-1)/(x-3) is meaningful in the real number system. -/
theorem meaningful_range_for_sqrt_fraction (x : ℝ) :
  (∃ y : ℝ, y^2 = x - 1 ∧ x - 3 ≠ 0) ↔ x ≥ 1 ∧ x ≠ 3 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_range_for_sqrt_fraction_l856_85644


namespace NUMINAMATH_CALUDE_dalton_needs_four_more_l856_85634

def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def saved_allowance : ℕ := 6
def money_from_uncle : ℕ := 13

theorem dalton_needs_four_more :
  jump_rope_cost + board_game_cost + playground_ball_cost - (saved_allowance + money_from_uncle) = 4 := by
  sorry

end NUMINAMATH_CALUDE_dalton_needs_four_more_l856_85634


namespace NUMINAMATH_CALUDE_negative_two_times_inequality_l856_85655

theorem negative_two_times_inequality (m n : ℝ) (h : m > n) : -2 * m < -2 * n := by
  sorry

end NUMINAMATH_CALUDE_negative_two_times_inequality_l856_85655


namespace NUMINAMATH_CALUDE_charlies_subtraction_l856_85648

theorem charlies_subtraction (charlie_add : 41^2 = 40^2 + 81) : 39^2 = 40^2 - 79 := by
  sorry

end NUMINAMATH_CALUDE_charlies_subtraction_l856_85648


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l856_85668

theorem fixed_point_on_line (k : ℝ) : k * 2 + 0 - 2 * k = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l856_85668


namespace NUMINAMATH_CALUDE_characterize_solutions_l856_85649

/-- The functional equation satisfied by f and g -/
def functional_equation (f g : ℕ → ℕ) : Prop :=
  ∀ n, f n + f (n + g n) = f (n + 1)

/-- The trivial solution where f is identically zero -/
def trivial_solution (f g : ℕ → ℕ) : Prop :=
  ∀ n, f n = 0

/-- The non-trivial solution family -/
def non_trivial_solution (f g : ℕ → ℕ) : Prop :=
  ∃ n₀ c : ℕ,
    (∀ n < n₀, f n = 0) ∧
    (∀ n ≥ n₀, f n = c * 2^(n - n₀)) ∧
    (∀ n < n₀ - 1, g n < n₀ - n) ∧
    (g (n₀ - 1) = 1) ∧
    (∀ n ≥ n₀, g n = 0)

/-- The main theorem characterizing all solutions to the functional equation -/
theorem characterize_solutions (f g : ℕ → ℕ) :
  functional_equation f g → (trivial_solution f g ∨ non_trivial_solution f g) :=
sorry

end NUMINAMATH_CALUDE_characterize_solutions_l856_85649


namespace NUMINAMATH_CALUDE_reciprocal_equation_l856_85685

theorem reciprocal_equation (x : ℝ) : 
  (3 + 1 / (2 - x) = 2 * (1 / (2 - x))) → x = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_equation_l856_85685


namespace NUMINAMATH_CALUDE_p_20_equals_657_l856_85684

/-- A polynomial p(x) = 3x^2 + kx + 117 where k is a constant such that p(1) = p(10) -/
def p (k : ℚ) (x : ℚ) : ℚ := 3 * x^2 + k * x + 117

/-- The theorem stating that for the polynomial p(x) with the given properties, p(20) = 657 -/
theorem p_20_equals_657 :
  ∃ k : ℚ, (p k 1 = p k 10) ∧ (p k 20 = 657) := by sorry

end NUMINAMATH_CALUDE_p_20_equals_657_l856_85684


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l856_85601

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 15 →
  (n * original_mean - n * decrement) / n = 185 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l856_85601


namespace NUMINAMATH_CALUDE_isosceles_triangle_on_hyperbola_x_coordinate_range_l856_85637

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 6 - y^2 / 3 = 1

-- Define the isosceles triangle
structure IsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isIsosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Theorem statement
theorem isosceles_triangle_on_hyperbola_x_coordinate_range 
  (triangle : IsoscelesTriangle)
  (hA : hyperbola triangle.A.1 triangle.A.2)
  (hB : hyperbola triangle.B.1 triangle.B.2)
  (hC : triangle.C.2 = 0)
  (hAB_not_perpendicular : (triangle.A.2 - triangle.B.2) ≠ 0) :
  triangle.C.1 > (3/2) * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_on_hyperbola_x_coordinate_range_l856_85637


namespace NUMINAMATH_CALUDE_frog_jump_probability_l856_85621

/-- Represents a jump in a random direction -/
structure Jump where
  length : ℝ
  direction : ℝ × ℝ

/-- Represents the frog's journey -/
def FrogJourney := List Jump

/-- Calculate the final position of the frog after a series of jumps -/
def finalPosition (journey : FrogJourney) : ℝ × ℝ := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Generate a random journey for the frog -/
def randomJourney : FrogJourney := sorry

/-- Probability that the frog's final position is within 2 meters of the start -/
def probabilityWithinTwoMeters : ℝ := sorry

/-- Theorem stating the probability is approximately 1/10 -/
theorem frog_jump_probability :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |probabilityWithinTwoMeters - 1/10| < ε := by sorry

end NUMINAMATH_CALUDE_frog_jump_probability_l856_85621


namespace NUMINAMATH_CALUDE_total_apples_is_sixteen_l856_85636

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- The total number of apples picked -/
def total_apples : ℕ := mike_apples + nancy_apples + keith_apples

theorem total_apples_is_sixteen : total_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_total_apples_is_sixteen_l856_85636


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l856_85667

def set_A : Set ℝ := { x | (x - 2) / (x + 5) < 0 }

def set_B : Set ℝ := { x | x^2 - 2*x - 3 ≥ 0 }

theorem intersection_of_A_and_B :
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ -5 < x ∧ x ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l856_85667


namespace NUMINAMATH_CALUDE_sibling_of_five_sevenths_unique_parent_one_over_2008_descendant_of_one_l856_85670

-- Define the child relation
def is_child (x y : ℝ) : Prop :=
  (y = x + 1) ∨ (y = x / (x + 1))

-- Define the sibling relation
def is_sibling (x y : ℝ) : Prop :=
  ∃ z, is_child z x ∧ y = z + 1

-- Define the descendant relation
def is_descendant (x y : ℝ) : Prop :=
  ∃ n : ℕ, ∃ f : ℕ → ℝ,
    f 0 = x ∧ f n = y ∧
    ∀ i < n, is_child (f i) (f (i + 1))

theorem sibling_of_five_sevenths :
  is_sibling (5/7) (7/2) :=
sorry

theorem unique_parent (x y z : ℝ) (hx : x > 0) (hz : z > 0) :
  is_child x y → is_child z y → x = z :=
sorry

theorem one_over_2008_descendant_of_one :
  is_descendant 1 (1/2008) :=
sorry

end NUMINAMATH_CALUDE_sibling_of_five_sevenths_unique_parent_one_over_2008_descendant_of_one_l856_85670
