import Mathlib

namespace NUMINAMATH_CALUDE_work_completion_time_l3358_335862

theorem work_completion_time 
  (days_a : ℝ) 
  (days_b : ℝ) 
  (h1 : days_a = 12) 
  (h2 : days_b = 24) : 
  1 / (1 / days_a + 1 / days_b) = 8 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3358_335862


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3358_335843

theorem quadratic_function_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 - 4 * x₁ - 13 = 0 ∧ a * x₂^2 - 4 * x₂ - 13 = 0) →
  (a > -4/13 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3358_335843


namespace NUMINAMATH_CALUDE_equation_solution_exists_l3358_335867

-- Define the possible operations
inductive Operation
  | mul
  | div

-- Define a function to apply the operation
def apply_op (op : Operation) (a b : ℕ) : ℚ :=
  match op with
  | Operation.mul => (a * b : ℚ)
  | Operation.div => (a / b : ℚ)

theorem equation_solution_exists : 
  ∃ (op1 op2 : Operation), 
    (apply_op op1 9 1307 = 100) ∧ 
    (∃ (n : ℕ), apply_op op2 14 2 = apply_op op2 n 5 ∧ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l3358_335867


namespace NUMINAMATH_CALUDE_divisibility_condition_l3358_335831

theorem divisibility_condition (n p : ℕ) (h_prime : Nat.Prime p) (h_bound : n < 2 * p) :
  (((p - 1) ^ n + 1) % (n ^ (p - 1)) = 0) ↔ 
  ((n = 1 ∧ Nat.Prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3358_335831


namespace NUMINAMATH_CALUDE_hyperbola_parameters_l3358_335858

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if the product of the slopes of its two asymptotes is -2
    and its focal length is 6, then a² = 3 and b² = 6 -/
theorem hyperbola_parameters (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (b^2 / a^2 = 2) →  -- product of slopes of asymptotes is -2
  (6^2 = 4 * (a^2 + b^2)) →  -- focal length is 6
  a^2 = 3 ∧ b^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parameters_l3358_335858


namespace NUMINAMATH_CALUDE_equilateral_triangles_congruence_l3358_335851

/-- An equilateral triangle -/
structure EquilateralTriangle :=
  (side : ℝ)
  (side_positive : side > 0)

/-- Two triangles are congruent if all their corresponding sides are equal -/
def congruent (t1 t2 : EquilateralTriangle) : Prop :=
  t1.side = t2.side

theorem equilateral_triangles_congruence (t1 t2 : EquilateralTriangle) :
  congruent t1 t2 ↔ t1.side = t2.side :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangles_congruence_l3358_335851


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3358_335877

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := (15 * Complex.I) / (3 + 4 * Complex.I)
  Complex.im z = 9 / 5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3358_335877


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_l3358_335834

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_prime_roots : 
  ∃! k : ℕ, ∃ p q : ℕ, 
    is_prime p ∧ is_prime q ∧ 
    p + q = 50 ∧ 
    p * q = k ∧ 
    k = 141 :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_l3358_335834


namespace NUMINAMATH_CALUDE_log_2_base_10_bounds_l3358_335850

theorem log_2_base_10_bounds :
  (2^9 = 512) →
  (2^14 = 16384) →
  (10^3 = 1000) →
  (10^4 = 10000) →
  (2/7 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_log_2_base_10_bounds_l3358_335850


namespace NUMINAMATH_CALUDE_part1_part2_l3358_335866

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - a) * (x - a + 1) ≤ 0}
def B : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define the proposition p
def p (m : ℝ) : Prop := ∃ x ∈ B, x^2 + (2*m + 1)*x + m^2 - m > 8

-- Theorem for part 1
theorem part1 : 
  (∀ x, x ∈ A a → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A a) → 
  a > -1 ∧ a < 1 :=
sorry

-- Theorem for part 2
theorem part2 : 
  (¬ p m) → m ≥ -1 ∧ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_l3358_335866


namespace NUMINAMATH_CALUDE_square_coverage_l3358_335810

/-- A square can be covered by smaller squares if the total area of the smaller squares
    is greater than or equal to the area of the larger square. -/
def can_cover (large_side small_side : ℝ) (num_small_squares : ℕ) : Prop :=
  large_side^2 ≤ (small_side^2 * num_small_squares)

/-- Theorem stating that a square with side length 7 can be covered by 8 squares
    with side length 3. -/
theorem square_coverage : can_cover 7 3 8 := by
  sorry

end NUMINAMATH_CALUDE_square_coverage_l3358_335810


namespace NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_l3358_335813

theorem sqrt_plus_square_zero_implies_diff (a b : ℝ) : 
  Real.sqrt (a - 3) + (b + 1)^2 = 0 → a - b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_plus_square_zero_implies_diff_l3358_335813


namespace NUMINAMATH_CALUDE_problem_solution_l3358_335815

theorem problem_solution (a : ℝ) : 3 ∈ ({1, -a^2, a-1} : Set ℝ) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3358_335815


namespace NUMINAMATH_CALUDE_marshmallow_roasting_l3358_335879

/-- The number of marshmallows Joe's dad has -/
def dads_marshmallows : ℕ := 21

/-- The number of marshmallows Joe has -/
def joes_marshmallows : ℕ := 4 * dads_marshmallows

/-- The number of marshmallows Joe's dad roasts -/
def dads_roasted : ℕ := dads_marshmallows / 3

/-- The number of marshmallows Joe roasts -/
def joes_roasted : ℕ := joes_marshmallows / 2

/-- The total number of marshmallows roasted by Joe and his dad -/
def total_roasted : ℕ := dads_roasted + joes_roasted

theorem marshmallow_roasting :
  total_roasted = 49 := by
  sorry

end NUMINAMATH_CALUDE_marshmallow_roasting_l3358_335879


namespace NUMINAMATH_CALUDE_regression_line_equation_l3358_335870

/-- Given a slope and a point on a line, calculate the y-intercept -/
def calculate_y_intercept (slope : ℝ) (point : ℝ × ℝ) : ℝ :=
  point.2 - slope * point.1

/-- The regression line problem -/
theorem regression_line_equation (slope : ℝ) (point : ℝ × ℝ) 
  (h_slope : slope = 1.23)
  (h_point : point = (4, 5)) :
  calculate_y_intercept slope point = 0.08 := by
  sorry

end NUMINAMATH_CALUDE_regression_line_equation_l3358_335870


namespace NUMINAMATH_CALUDE_hyperbola_satisfies_equation_l3358_335889

/-- A hyperbola with given asymptotes and passing through a specific point -/
structure Hyperbola where
  -- The slope of the asymptotes
  asymptote_slope : ℝ
  -- The point through which the hyperbola passes
  point : ℝ × ℝ

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 / 14 - y^2 / 7 = 1

/-- Theorem stating that the given hyperbola satisfies the equation -/
theorem hyperbola_satisfies_equation (h : Hyperbola) 
  (h_asymptote : h.asymptote_slope = 1/2)
  (h_point : h.point = (4, Real.sqrt 2)) :
  hyperbola_equation h 4 (Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_satisfies_equation_l3358_335889


namespace NUMINAMATH_CALUDE_factorial_divisibility_iff_power_of_two_l3358_335824

theorem factorial_divisibility_iff_power_of_two (n : ℕ) :
  (∃ k : ℕ, n = 2^k) ↔ (2^(n-1) ∣ n!) := by
  sorry

end NUMINAMATH_CALUDE_factorial_divisibility_iff_power_of_two_l3358_335824


namespace NUMINAMATH_CALUDE_yellow_green_weight_difference_l3358_335857

/-- The weight difference between two blocks -/
def weight_difference (yellow_weight green_weight : Real) : Real :=
  yellow_weight - green_weight

/-- Theorem stating the weight difference between yellow and green blocks -/
theorem yellow_green_weight_difference :
  let yellow_weight : Real := 0.6
  let green_weight : Real := 0.4
  weight_difference yellow_weight green_weight = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_yellow_green_weight_difference_l3358_335857


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l3358_335853

theorem factor_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, (x + 6) ∣ (x^2 - m*x - 42)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l3358_335853


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3358_335808

def A : Set ℝ := {x | x^2 - 5*x - 6 < 0}
def B : Set ℝ := {x | -3 < x ∧ x < 2}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -3 < x ∧ x < 6} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3358_335808


namespace NUMINAMATH_CALUDE_last_digit_of_power_of_two_plus_one_l3358_335818

theorem last_digit_of_power_of_two_plus_one (n : ℕ) (h : n ≥ 2) :
  (2^(2^n) + 1) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_power_of_two_plus_one_l3358_335818


namespace NUMINAMATH_CALUDE_complex_division_simplification_l3358_335845

theorem complex_division_simplification :
  let i : ℂ := Complex.I
  (-3 + 2*i) / (1 + i) = -1/2 + 5/2*i := by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l3358_335845


namespace NUMINAMATH_CALUDE_area_of_inscribed_square_l3358_335832

/-- The area of a square inscribed in a circle, which is itself inscribed in an equilateral triangle with side length 6 cm. -/
theorem area_of_inscribed_square (triangle_side : ℝ) (h : triangle_side = 6) : 
  let circle_radius := triangle_side / (2 * Real.sqrt 3)
  let square_side := 2 * circle_radius / Real.sqrt 2
  square_side ^ 2 = 6 * (3 - Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_area_of_inscribed_square_l3358_335832


namespace NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3358_335809

theorem fraction_sum_equals_decimal : 2/10 + 4/100 + 6/1000 = 0.246 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_decimal_l3358_335809


namespace NUMINAMATH_CALUDE_probability_at_least_one_girl_pair_l3358_335869

/-- The number of ways to split 2n people into n pairs --/
def total_pairings (n : ℕ) : ℕ := (2 * n).factorial / (2^n * n.factorial)

/-- The number of ways to pair n boys with n girls --/
def boy_girl_pairings (n : ℕ) : ℕ := n.factorial

theorem probability_at_least_one_girl_pair (n : ℕ) (hn : n = 5) :
  (total_pairings n - boy_girl_pairings n) / total_pairings n = 23640 / 23760 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_girl_pair_l3358_335869


namespace NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3358_335882

theorem gcd_factorial_eight_ten : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_eight_ten_l3358_335882


namespace NUMINAMATH_CALUDE_eight_fifteen_div_sixtyfour_six_l3358_335846

theorem eight_fifteen_div_sixtyfour_six : 8^15 / 64^6 = 512 := by
  sorry

end NUMINAMATH_CALUDE_eight_fifteen_div_sixtyfour_six_l3358_335846


namespace NUMINAMATH_CALUDE_volume_surface_area_ratio_l3358_335830

/-- A structure formed by connecting eight unit cubes -/
structure CubeStructure where
  /-- The number of unit cubes in the structure -/
  num_cubes : ℕ
  /-- The volume of the structure in cubic units -/
  volume : ℕ
  /-- The surface area of the structure in square units -/
  surface_area : ℕ
  /-- The number of cubes is 8 -/
  cube_count : num_cubes = 8
  /-- The volume is equal to the number of cubes -/
  volume_def : volume = num_cubes
  /-- The surface area is 24 square units -/
  surface_area_def : surface_area = 24

/-- Theorem: The ratio of volume to surface area is 1/3 -/
theorem volume_surface_area_ratio (c : CubeStructure) :
  (c.volume : ℚ) / c.surface_area = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_surface_area_ratio_l3358_335830


namespace NUMINAMATH_CALUDE_bonus_is_ten_dollars_l3358_335883

/-- Represents the payment structure for Brady's transcription job -/
structure TranscriptionJob where
  base_pay : ℚ  -- Base pay per card in dollars
  cards_for_bonus : ℕ  -- Number of cards needed for a bonus
  total_cards : ℕ  -- Total number of cards transcribed
  total_pay : ℚ  -- Total pay including bonuses in dollars

/-- Calculates the bonus amount per bonus interval -/
def bonus_amount (job : TranscriptionJob) : ℚ :=
  let base_total := job.base_pay * job.total_cards
  let bonus_count := job.total_cards / job.cards_for_bonus
  (job.total_pay - base_total) / bonus_count

/-- Theorem stating that the bonus amount is $10 for every 100 cards -/
theorem bonus_is_ten_dollars (job : TranscriptionJob) 
  (h1 : job.base_pay = 70 / 100)
  (h2 : job.cards_for_bonus = 100)
  (h3 : job.total_cards = 200)
  (h4 : job.total_pay = 160) :
  bonus_amount job = 10 := by
  sorry

end NUMINAMATH_CALUDE_bonus_is_ten_dollars_l3358_335883


namespace NUMINAMATH_CALUDE_total_lines_through_centers_l3358_335878

/-- The size of the cube --/
def cube_size : Nat := 2008

/-- The number of lines parallel to the edges of the cube --/
def parallel_lines : Nat := cube_size * cube_size * 3

/-- The number of diagonal lines within the planes --/
def diagonal_lines : Nat := cube_size * 2 * 3

/-- The number of space diagonals of the cube --/
def space_diagonals : Nat := 4

/-- Theorem stating the total number of lines passing through the centers of exactly 2008 unit cubes in a 2008 x 2008 x 2008 cube --/
theorem total_lines_through_centers (cube_size : Nat) (h : cube_size = 2008) :
  parallel_lines + diagonal_lines + space_diagonals = 12115300 := by
  sorry

#eval parallel_lines + diagonal_lines + space_diagonals

end NUMINAMATH_CALUDE_total_lines_through_centers_l3358_335878


namespace NUMINAMATH_CALUDE_total_animals_legoland_animals_l3358_335897

/-- Given a ratio of kangaroos to koalas and the total number of kangaroos,
    calculate the total number of animals (koalas and kangaroos). -/
theorem total_animals (ratio : ℕ) (num_kangaroos : ℕ) : ℕ :=
  let num_koalas := num_kangaroos / ratio
  num_koalas + num_kangaroos

/-- Prove that given 5 kangaroos for each koala and 180 kangaroos in total,
    the total number of koalas and kangaroos is 216. -/
theorem legoland_animals : total_animals 5 180 = 216 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_legoland_animals_l3358_335897


namespace NUMINAMATH_CALUDE_g_neg_three_l3358_335854

def g (x : ℝ) : ℝ := 10 * x^3 - 4 * x^2 - 6 * x + 7

theorem g_neg_three : g (-3) = -281 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_three_l3358_335854


namespace NUMINAMATH_CALUDE_pencil_cost_l3358_335884

theorem pencil_cost (total_students : Nat) (total_cost : Nat) 
  (h1 : total_students = 36)
  (h2 : total_cost = 1881)
  (s : Nat) (c : Nat) (n : Nat)
  (h3 : s > total_students / 2)
  (h4 : c > n)
  (h5 : n > 1)
  (h6 : s * c * n = total_cost) :
  c = 17 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l3358_335884


namespace NUMINAMATH_CALUDE_zero_of_f_l3358_335868

-- Define the function f(x) = 2x + 7
def f (x : ℝ) : ℝ := 2 * x + 7

-- Theorem stating that the zero of f(x) is -7/2
theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = -7/2 := by
sorry

end NUMINAMATH_CALUDE_zero_of_f_l3358_335868


namespace NUMINAMATH_CALUDE_foreign_language_books_l3358_335802

theorem foreign_language_books (total : ℝ) 
  (h1 : total * (36 / 100) = total - (total * (27 / 100) + 185))
  (h2 : total * (27 / 100) = total * (36 / 100) * (75 / 100))
  (h3 : 185 = total - (total * (36 / 100) + total * (27 / 100))) :
  total = 500 := by sorry

end NUMINAMATH_CALUDE_foreign_language_books_l3358_335802


namespace NUMINAMATH_CALUDE_tan_pi_plus_2alpha_l3358_335842

theorem tan_pi_plus_2alpha (α : Real) 
  (h1 : α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi))
  (h2 : Real.sin (Real.pi / 2 + α) = 1 / 3) : 
  Real.tan (Real.pi + 2 * α) = 4 * Real.sqrt 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_plus_2alpha_l3358_335842


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3358_335847

theorem least_positive_integer_with_remainders : ∃! x : ℕ,
  x > 0 ∧
  x % 4 = 3 ∧
  x % 5 = 4 ∧
  x % 7 = 6 ∧
  ∀ y : ℕ, y > 0 ∧ y % 4 = 3 ∧ y % 5 = 4 ∧ y % 7 = 6 → x ≤ y :=
by
  use 139
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3358_335847


namespace NUMINAMATH_CALUDE_percentage_problem_l3358_335804

theorem percentage_problem (x : ℝ) : 
  (x / 100) * 25 + 5.4 = 9.15 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3358_335804


namespace NUMINAMATH_CALUDE_integral_x_squared_plus_4x_plus_3_cos_x_l3358_335822

theorem integral_x_squared_plus_4x_plus_3_cos_x : 
  ∫ (x : ℝ) in (-1)..0, (x^2 + 4*x + 3) * Real.cos x = 4 - 2 * Real.cos 1 - 2 * Real.sin 1 := by
  sorry

end NUMINAMATH_CALUDE_integral_x_squared_plus_4x_plus_3_cos_x_l3358_335822


namespace NUMINAMATH_CALUDE_flour_to_add_l3358_335859

/-- Given a cake recipe and partially added ingredients, calculate the remaining amount to be added -/
theorem flour_to_add (total_required : ℕ) (already_added : ℕ) (h : total_required ≥ already_added) :
  total_required - already_added = 8 - 4 :=
by
  sorry

#check flour_to_add

end NUMINAMATH_CALUDE_flour_to_add_l3358_335859


namespace NUMINAMATH_CALUDE_four_planes_max_parts_l3358_335885

/-- The maximum number of parts into which space can be divided by k planes -/
def max_parts (k : ℕ) : ℚ := (k^3 + 5*k + 6) / 6

/-- Theorem: The maximum number of parts into which space can be divided by four planes is 15 -/
theorem four_planes_max_parts : max_parts 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_four_planes_max_parts_l3358_335885


namespace NUMINAMATH_CALUDE_hexagons_in_100th_ring_hexagons_in_nth_ring_formula_l3358_335811

/-- Represents the number of hexagons in the nth ring of a hexagonal array -/
def hexagons_in_nth_ring (n : ℕ) : ℕ := 6 * n

/-- The hexagonal array satisfies the initial conditions -/
axiom first_ring : hexagons_in_nth_ring 1 = 6
axiom second_ring : hexagons_in_nth_ring 2 = 12

/-- Theorem: The number of hexagons in the 100th ring is 600 -/
theorem hexagons_in_100th_ring : hexagons_in_nth_ring 100 = 600 := by
  sorry

/-- Theorem: The number of hexagons in the nth ring is 6n -/
theorem hexagons_in_nth_ring_formula (n : ℕ) : hexagons_in_nth_ring n = 6 * n := by
  sorry

end NUMINAMATH_CALUDE_hexagons_in_100th_ring_hexagons_in_nth_ring_formula_l3358_335811


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l3358_335898

theorem trigonometric_expression_equals_one :
  let cos30 : ℝ := Real.sqrt 3 / 2
  let sin30 : ℝ := 1 / 2
  let cos60 : ℝ := 1 / 2
  let sin60 : ℝ := Real.sqrt 3 / 2
  (1 - 1 / cos30) * (1 + 1 / sin60) * (1 - 1 / sin30) * (1 + 1 / cos60) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_l3358_335898


namespace NUMINAMATH_CALUDE_mindmaster_codes_l3358_335840

/-- The number of colors available for the Mindmaster game -/
def num_colors : ℕ := 5

/-- The number of slots in the Mindmaster game -/
def num_slots : ℕ := 5

/-- The total number of possible secret codes in the Mindmaster game -/
def total_codes : ℕ := num_colors ^ num_slots

/-- Theorem stating that the total number of possible secret codes is 3125 -/
theorem mindmaster_codes : total_codes = 3125 := by
  sorry

end NUMINAMATH_CALUDE_mindmaster_codes_l3358_335840


namespace NUMINAMATH_CALUDE_num_divisors_8_factorial_is_96_l3358_335876

/-- The number of positive divisors of 8! -/
def num_divisors_8_factorial : ℕ :=
  let factorial_8 : ℕ := 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1
  -- Definition of the number of divisors function not provided, so we'll declare it
  sorry

/-- Theorem: The number of positive divisors of 8! is 96 -/
theorem num_divisors_8_factorial_is_96 :
  num_divisors_8_factorial = 96 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_8_factorial_is_96_l3358_335876


namespace NUMINAMATH_CALUDE_tan_ratio_given_sin_relation_l3358_335865

theorem tan_ratio_given_sin_relation (α : Real) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * (Real.pi / 180))) :
  Real.tan (α + Real.pi / 180) / Real.tan (α - Real.pi / 180) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_ratio_given_sin_relation_l3358_335865


namespace NUMINAMATH_CALUDE_equation_solution_l3358_335800

theorem equation_solution (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (eq1 : x = 2 + 1 / y) (eq2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3358_335800


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3358_335894

/-- A convex quadrilateral in a plane -/
structure ConvexQuadrilateral where
  -- We don't need to define the specific properties of a convex quadrilateral here
  -- as they are not directly used in the problem statement

/-- The theorem stating the relation between the area, sum of sides and diagonals
    in a specific convex quadrilateral -/
theorem quadrilateral_diagonal_length 
  (Q : ConvexQuadrilateral) 
  (area : ℝ) 
  (sum_sides_and_diagonal : ℝ) 
  (h1 : area = 32) 
  (h2 : sum_sides_and_diagonal = 16) : 
  ∃ (other_diagonal : ℝ), other_diagonal = 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l3358_335894


namespace NUMINAMATH_CALUDE_initial_oranges_l3358_335826

theorem initial_oranges (total : ℕ) 
  (h1 : total % 2 = 0)  -- Half of the oranges were ripe
  (h2 : (total / 2) % 4 = 0)  -- 1/4 of the ripe oranges were eaten
  (h3 : (total / 2) % 8 = 0)  -- 1/8 of the unripe oranges were eaten
  (h4 : total * 13 / 16 = 78)  -- 78 oranges were left uneaten in total
  : total = 96 := by
  sorry

end NUMINAMATH_CALUDE_initial_oranges_l3358_335826


namespace NUMINAMATH_CALUDE_max_difference_m_n_l3358_335899

theorem max_difference_m_n (m n : ℤ) (hm : m > 0) (h : m^2 = 4*n^2 - 5*n + 16) :
  ∃ (m' n' : ℤ), m' > 0 ∧ m'^2 = 4*n'^2 - 5*n' + 16 ∧ |m' - n'| ≤ 33 ∧
  ∀ (m'' n'' : ℤ), m'' > 0 → m''^2 = 4*n''^2 - 5*n'' + 16 → |m'' - n''| ≤ |m' - n'| :=
sorry

end NUMINAMATH_CALUDE_max_difference_m_n_l3358_335899


namespace NUMINAMATH_CALUDE_initial_honey_amount_honey_jar_problem_l3358_335819

/-- The amount of honey remaining after each extraction -/
def honey_remaining (initial_honey : ℝ) (num_extractions : ℕ) : ℝ :=
  initial_honey * (0.8 ^ num_extractions)

/-- Theorem stating the initial amount of honey given the final amount and number of extractions -/
theorem initial_honey_amount 
  (final_honey : ℝ) 
  (num_extractions : ℕ) 
  (h_final : honey_remaining initial_honey num_extractions = final_honey) : 
  initial_honey = final_honey / (0.8 ^ num_extractions) :=
by sorry

/-- The solution to the honey jar problem -/
theorem honey_jar_problem : 
  ∃ (initial_honey : ℝ), 
    honey_remaining initial_honey 4 = 512 ∧ 
    initial_honey = 1250 :=
by sorry

end NUMINAMATH_CALUDE_initial_honey_amount_honey_jar_problem_l3358_335819


namespace NUMINAMATH_CALUDE_bill_oranges_count_l3358_335887

/-- The number of oranges Betty picked -/
def betty_oranges : ℕ := 15

/-- The number of oranges Bill picked -/
def bill_oranges : ℕ := sorry

/-- The number of oranges Frank picked -/
def frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)

/-- The number of seeds Frank planted -/
def seeds_planted : ℕ := 2 * frank_oranges

/-- The number of oranges on each tree -/
def oranges_per_tree : ℕ := 5

/-- The total number of oranges Philip can pick -/
def philip_oranges : ℕ := 810

theorem bill_oranges_count : bill_oranges = 12 := by
  sorry

end NUMINAMATH_CALUDE_bill_oranges_count_l3358_335887


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l3358_335801

theorem ceiling_floor_sum : ⌈(7:ℝ)/3⌉ + ⌊-(7:ℝ)/3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l3358_335801


namespace NUMINAMATH_CALUDE_square_area_3_square_area_3_proof_l3358_335892

/-- The area of a square with side length 3 is 9 -/
theorem square_area_3 : Real → Prop :=
  fun area =>
    let side_length : Real := 3
    area = side_length ^ 2

#check square_area_3 9

/-- Proof of the theorem -/
theorem square_area_3_proof : square_area_3 9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_3_square_area_3_proof_l3358_335892


namespace NUMINAMATH_CALUDE_expression_value_l3358_335803

theorem expression_value : (16.25 / 0.25) + (8.4 / 3) - (0.75 / 0.05) = 52.8 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3358_335803


namespace NUMINAMATH_CALUDE_banana_groups_l3358_335841

theorem banana_groups (total_bananas : ℕ) (num_groups : ℕ) 
  (h1 : total_bananas = 392) 
  (h2 : num_groups = 196) : 
  total_bananas / num_groups = 2 := by
  sorry

end NUMINAMATH_CALUDE_banana_groups_l3358_335841


namespace NUMINAMATH_CALUDE_smallest_enclosing_circle_radius_l3358_335895

theorem smallest_enclosing_circle_radius (r : ℝ) : 
  (∃ (A B C O : ℝ × ℝ),
    -- Three unit circles touching each other
    dist A B = 2 ∧ dist B C = 2 ∧ dist C A = 2 ∧
    -- O is the center of the enclosing circle
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧
    -- r is the smallest possible radius
    ∀ (r' : ℝ), (∃ (O' : ℝ × ℝ), dist O' A ≤ r' ∧ dist O' B ≤ r' ∧ dist O' C ≤ r') → r ≤ r') →
  r = 1 + 2 / Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_enclosing_circle_radius_l3358_335895


namespace NUMINAMATH_CALUDE_coaches_in_conference_l3358_335816

theorem coaches_in_conference (rowers : ℕ) (votes_per_rower : ℕ) (votes_per_coach : ℕ) 
  (h1 : rowers = 60)
  (h2 : votes_per_rower = 3)
  (h3 : votes_per_coach = 5) :
  (rowers * votes_per_rower) / votes_per_coach = 36 :=
by sorry

end NUMINAMATH_CALUDE_coaches_in_conference_l3358_335816


namespace NUMINAMATH_CALUDE_smallest_valid_arrangement_l3358_335852

def is_valid_arrangement (n : ℕ) : Prop :=
  ∃ (a₁ a₂ a₃ a₄ : ℕ),
    a₁ = 15 ∧
    a₂ = n ∧
    a₃ = 1 ∧
    a₄ = 6 ∧
    n % a₁ = 0 ∧
    n % a₂ = 0 ∧
    n % a₃ = 0 ∧
    n % a₄ = 0 ∧
    ∀ (i j : ℕ), i ≠ j → (n / a₁) ≠ (n / a₂) ∧ (n / a₁) ≠ (n / a₃) ∧ (n / a₁) ≠ (n / a₄) ∧
                         (n / a₂) ≠ (n / a₃) ∧ (n / a₂) ≠ (n / a₄) ∧ (n / a₃) ≠ (n / a₄)

theorem smallest_valid_arrangement : 
  ∃ (n : ℕ), is_valid_arrangement n ∧ ∀ (m : ℕ), m < n → ¬is_valid_arrangement m :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_arrangement_l3358_335852


namespace NUMINAMATH_CALUDE_extremum_and_max_min_of_f_l3358_335886

def f (x : ℝ) := x^3 + 4*x^2 - 11*x + 16

theorem extremum_and_max_min_of_f :
  (∃ (x : ℝ), f x = 10 ∧ ∀ y, |y - 1| < |x - 1| → f y ≠ 10) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≤ 18) ∧
  (∃ x ∈ Set.Icc 0 2, f x = 18) ∧
  (∀ x ∈ Set.Icc 0 2, f x ≥ 10) ∧
  (∃ x ∈ Set.Icc 0 2, f x = 10) :=
sorry

end NUMINAMATH_CALUDE_extremum_and_max_min_of_f_l3358_335886


namespace NUMINAMATH_CALUDE_gwi_seed_count_l3358_335893

/-- The number of watermelon seeds Bom has -/
def bom_seeds : ℕ := 300

/-- The total number of watermelon seeds they have together -/
def total_seeds : ℕ := 1660

/-- The number of watermelon seeds Gwi has -/
def gwi_seeds : ℕ := 340

/-- The number of watermelon seeds Yeon has -/
def yeon_seeds : ℕ := 3 * gwi_seeds

theorem gwi_seed_count :
  bom_seeds < gwi_seeds ∧
  yeon_seeds = 3 * gwi_seeds ∧
  bom_seeds + gwi_seeds + yeon_seeds = total_seeds :=
by sorry

end NUMINAMATH_CALUDE_gwi_seed_count_l3358_335893


namespace NUMINAMATH_CALUDE_range_of_m_l3358_335823

theorem range_of_m (m : ℝ) : 
  (∀ (x y : ℝ), x > 0 → y > 0 → (2*x - y/Real.exp 1) * Real.log (y/x) ≤ x/(m*Real.exp 1)) ↔ 
  (m > 0 ∧ m ≤ 1/Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3358_335823


namespace NUMINAMATH_CALUDE_particle_acceleration_l3358_335890

-- Define the displacement function
def s (t : ℝ) : ℝ := t^2 - t + 6

-- Define the velocity function as the derivative of displacement
def v (t : ℝ) : ℝ := 2 * t - 1

-- Define the acceleration function as the derivative of velocity
def a (t : ℝ) : ℝ := 2

-- Theorem statement
theorem particle_acceleration (t : ℝ) (h : t ∈ Set.Icc 1 4) :
  a t = 2 := by
  sorry

end NUMINAMATH_CALUDE_particle_acceleration_l3358_335890


namespace NUMINAMATH_CALUDE_exists_solution_for_calendar_equation_l3358_335820

theorem exists_solution_for_calendar_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_for_calendar_equation_l3358_335820


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2010th_term_l3358_335896

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_2010th_term 
  (p q : ℝ) 
  (h1 : 9 = arithmetic_sequence p (2 * q) 2)
  (h2 : 3 * p - q = arithmetic_sequence p (2 * q) 3)
  (h3 : 3 * p + q = arithmetic_sequence p (2 * q) 4) :
  arithmetic_sequence p (2 * q) 2010 = 8041 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2010th_term_l3358_335896


namespace NUMINAMATH_CALUDE_central_angle_nairobi_lima_l3358_335848

/-- Represents a point on Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real

/-- Calculates the central angle between two points on Earth -/
def centralAngle (p1 p2 : EarthPoint) : Real :=
  |p1.longitude - p2.longitude|

theorem central_angle_nairobi_lima :
  let nairobi : EarthPoint := { latitude := -1, longitude := 36 }
  let lima : EarthPoint := { latitude := -12, longitude := -77 }
  centralAngle nairobi lima = 113 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_nairobi_lima_l3358_335848


namespace NUMINAMATH_CALUDE_tylers_sanctuary_pairs_l3358_335814

/-- Represents the animal sanctuary with three regions -/
structure AnimalSanctuary where
  bird_species : ℕ
  bird_pairs_per_species : ℕ
  marine_species : ℕ
  marine_pairs_per_species : ℕ
  mammal_species : ℕ
  mammal_pairs_per_species : ℕ

/-- Calculates the total number of pairs in the sanctuary -/
def total_pairs (sanctuary : AnimalSanctuary) : ℕ :=
  sanctuary.bird_species * sanctuary.bird_pairs_per_species +
  sanctuary.marine_species * sanctuary.marine_pairs_per_species +
  sanctuary.mammal_species * sanctuary.mammal_pairs_per_species

/-- Theorem stating that the total number of pairs in Tyler's sanctuary is 470 -/
theorem tylers_sanctuary_pairs :
  let tyler_sanctuary : AnimalSanctuary := {
    bird_species := 29,
    bird_pairs_per_species := 7,
    marine_species := 15,
    marine_pairs_per_species := 9,
    mammal_species := 22,
    mammal_pairs_per_species := 6
  }
  total_pairs tyler_sanctuary = 470 := by
  sorry

end NUMINAMATH_CALUDE_tylers_sanctuary_pairs_l3358_335814


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l3358_335839

theorem complex_power_magnitude : Complex.abs ((2 + 2*Complex.I)^6) = 512 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l3358_335839


namespace NUMINAMATH_CALUDE_algebraic_equality_l3358_335874

theorem algebraic_equality (a b c : ℝ) : a + b * c = (a + b) * (a + c) ↔ a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equality_l3358_335874


namespace NUMINAMATH_CALUDE_age_difference_l3358_335863

/-- Given two people A and B, where B is currently 39 years old, and in 10 years A will be twice as old as B was 10 years ago, this theorem proves that A is currently 9 years older than B. -/
theorem age_difference (A_age B_age : ℕ) : 
  B_age = 39 → 
  A_age + 10 = 2 * (B_age - 10) → 
  A_age - B_age = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3358_335863


namespace NUMINAMATH_CALUDE_chalk_boxes_l3358_335835

theorem chalk_boxes (total_chalk : ℕ) (chalk_per_box : ℕ) (h1 : total_chalk = 3484) (h2 : chalk_per_box = 18) :
  (total_chalk + chalk_per_box - 1) / chalk_per_box = 194 := by
  sorry

end NUMINAMATH_CALUDE_chalk_boxes_l3358_335835


namespace NUMINAMATH_CALUDE_erased_digit_greater_than_original_l3358_335806

-- Define the fraction
def fraction : Rat := 3 / 7

-- Define the number of digits after the decimal point
def num_digits : Nat := 1000

-- Define the position of the digit to be erased
def erased_position : Nat := 500

-- Function to get the nth digit after the decimal point
def nth_digit (n : Nat) : Nat :=
  sorry

-- Function to construct the number after erasing the 500th digit
def number_after_erasing : Rat :=
  sorry

-- Theorem statement
theorem erased_digit_greater_than_original :
  number_after_erasing > fraction :=
sorry

end NUMINAMATH_CALUDE_erased_digit_greater_than_original_l3358_335806


namespace NUMINAMATH_CALUDE_quadratic_expression_values_l3358_335828

theorem quadratic_expression_values (m n : ℤ) 
  (hm : |m| = 3) 
  (hn : |n| = 2) 
  (hmn : m < n) : 
  m^2 + m*n + n^2 = 7 ∨ m^2 + m*n + n^2 = 19 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_values_l3358_335828


namespace NUMINAMATH_CALUDE_john_games_l3358_335812

/-- Calculates the number of unique working games John ended up with -/
def unique_working_games (friend_games : ℕ) (friend_nonworking : ℕ) (garage_games : ℕ) (garage_nonworking : ℕ) (garage_duplicates : ℕ) : ℕ :=
  (friend_games - friend_nonworking) + (garage_games - garage_nonworking - garage_duplicates)

/-- Theorem stating that John ended up with 17 unique working games -/
theorem john_games : unique_working_games 25 12 15 8 3 = 17 := by
  sorry

end NUMINAMATH_CALUDE_john_games_l3358_335812


namespace NUMINAMATH_CALUDE_triangle_base_length_l3358_335855

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 6 → height = 4 → area = (base * height) / 2 → base = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_base_length_l3358_335855


namespace NUMINAMATH_CALUDE_small_portion_visible_implies_intersection_l3358_335807

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a point is above a line -/
def isAboveLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c > 0

/-- Predicate to check if a circle intersects a line -/
def circleIntersectsLine (c : Circle) (l : Line) : Prop :=
  ∃ (p : ℝ × ℝ), (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
                 l.a * p.1 + l.b * p.2 + l.c = 0

/-- Predicate to check if a small portion of a circle is visible above a line -/
def smallPortionVisible (c : Circle) (l : Line) : Prop :=
  ∃ (p q : ℝ × ℝ), 
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 ∧
    (q.1 - c.center.1)^2 + (q.2 - c.center.2)^2 = c.radius^2 ∧
    isAboveLine p l ∧
    isAboveLine q l ∧
    ∀ (r : ℝ × ℝ), (r.1 - c.center.1)^2 + (r.2 - c.center.2)^2 = c.radius^2 →
                   isAboveLine r l →
                   (r.1 ≥ min p.1 q.1 ∧ r.1 ≤ max p.1 q.1) ∧
                   (r.2 ≥ min p.2 q.2 ∧ r.2 ≤ max p.2 q.2)

theorem small_portion_visible_implies_intersection (c : Circle) (l : Line) :
  smallPortionVisible c l → circleIntersectsLine c l :=
by sorry

end NUMINAMATH_CALUDE_small_portion_visible_implies_intersection_l3358_335807


namespace NUMINAMATH_CALUDE_largest_prime_to_test_primality_l3358_335849

theorem largest_prime_to_test_primality (n : ℕ) : 
  900 ≤ n ∧ n ≤ 950 → 
  (∀ (p : ℕ), p.Prime → p ≤ 29 → (p ∣ n ↔ ¬n.Prime)) ∧
  (∀ (p : ℕ), p.Prime → p > 29 → (p ∣ n → ¬n.Prime)) :=
sorry

end NUMINAMATH_CALUDE_largest_prime_to_test_primality_l3358_335849


namespace NUMINAMATH_CALUDE_perpendicular_bisector_theorem_l3358_335891

/-- A structure representing a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- A function to construct the perpendicular bisector points A', B', and C' -/
def constructPerpendicularBisectorPoints (t : Triangle) : Triangle :=
  sorry

/-- A predicate to check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  sorry

/-- A predicate to check if a triangle has angles 30°, 30°, and 120° -/
def has30_30_120Angles (t : Triangle) : Prop :=
  sorry

/-- The main theorem -/
theorem perpendicular_bisector_theorem (t : Triangle) :
  let t' := constructPerpendicularBisectorPoints t
  isEquilateral t' ↔ (isEquilateral t ∨ has30_30_120Angles t) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_theorem_l3358_335891


namespace NUMINAMATH_CALUDE_divisor_problem_l3358_335817

theorem divisor_problem : ∃ (N D : ℕ), 
  (N % D = 6) ∧ 
  (N % 19 = 7) ∧ 
  (D = 39) := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3358_335817


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l3358_335836

theorem necessary_not_sufficient (a b : ℝ) : 
  (∀ a b : ℝ, b ≥ 0 → a^2 + b ≥ 0) ∧ 
  (∃ a b : ℝ, a^2 + b ≥ 0 ∧ b < 0) := by
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l3358_335836


namespace NUMINAMATH_CALUDE_tim_soda_cans_l3358_335875

/-- The number of soda cans Tim has at the end of the scenario -/
def final_cans (initial : ℕ) (taken : ℕ) : ℕ :=
  let remaining := initial - taken
  remaining + (remaining / 2)

/-- Theorem stating that Tim ends up with 24 cans -/
theorem tim_soda_cans : final_cans 22 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_tim_soda_cans_l3358_335875


namespace NUMINAMATH_CALUDE_range_of_m_plus_n_l3358_335861

/-- Given a function f(x) = me^x + x^2 + nx where the set of roots of f and f∘f are equal and non-empty,
    prove that the range of m + n is [0, 4). -/
theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x, m * Real.exp x + x^2 + n * x = 0) →
  {x | m * Real.exp x + x^2 + n * x = 0} = {x | m * Real.exp (m * Real.exp x + x^2 + n * x) + 
    (m * Real.exp x + x^2 + n * x)^2 + n * (m * Real.exp x + x^2 + n * x) = 0} →
  m + n ∈ Set.Icc 0 4 ∧ ¬(m + n = 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_plus_n_l3358_335861


namespace NUMINAMATH_CALUDE_total_flowers_in_gardens_l3358_335844

/-- Given 10 gardens, each with 544 pots, and each pot containing 32 flowers,
    prove that the total number of flowers in all gardens is 174,080. -/
theorem total_flowers_in_gardens : 
  let num_gardens : ℕ := 10
  let pots_per_garden : ℕ := 544
  let flowers_per_pot : ℕ := 32
  num_gardens * pots_per_garden * flowers_per_pot = 174080 :=
by sorry

end NUMINAMATH_CALUDE_total_flowers_in_gardens_l3358_335844


namespace NUMINAMATH_CALUDE_total_water_consumption_l3358_335825

def traveler_ounces : ℕ := 32
def camel_multiplier : ℕ := 7
def ounces_per_gallon : ℕ := 128

theorem total_water_consumption :
  (traveler_ounces + camel_multiplier * traveler_ounces) / ounces_per_gallon = 2 := by
  sorry

end NUMINAMATH_CALUDE_total_water_consumption_l3358_335825


namespace NUMINAMATH_CALUDE_f_property_P_implies_m_range_l3358_335872

/-- Property P(a) for a function f on domain D -/
def property_P (f : ℝ → ℝ) (D : Set ℝ) (a : ℝ) : Prop :=
  ∀ x₁ ∈ D, ∃ x₂ ∈ D, (x₁ + f x₂) / 2 = a

/-- The function f(x) = -x² + mx - 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + m*x - 3

/-- The domain of f(x) -/
def D : Set ℝ := {x : ℝ | x > 0}

theorem f_property_P_implies_m_range :
  ∀ m : ℝ, property_P (f m) D (1/2) → m ∈ {m : ℝ | m ≥ 4} := by sorry

end NUMINAMATH_CALUDE_f_property_P_implies_m_range_l3358_335872


namespace NUMINAMATH_CALUDE_valid_param_iff_l3358_335837

/-- A vector parameterization of a line --/
structure VectorParam where
  x₀ : ℝ
  y₀ : ℝ
  dx : ℝ
  dy : ℝ

/-- The line y = 3x - 4 --/
def line (x : ℝ) : ℝ := 3 * x - 4

/-- Predicate for a valid vector parameterization --/
def is_valid_param (p : VectorParam) : Prop :=
  p.y₀ = line p.x₀ ∧ p.dy = 3 * p.dx

/-- Theorem: A vector parameterization is valid iff it satisfies the conditions --/
theorem valid_param_iff (p : VectorParam) :
  is_valid_param p ↔ ∀ t : ℝ, line (p.x₀ + t * p.dx) = p.y₀ + t * p.dy :=
sorry

end NUMINAMATH_CALUDE_valid_param_iff_l3358_335837


namespace NUMINAMATH_CALUDE_spider_journey_l3358_335805

theorem spider_journey (r : ℝ) (third_leg : ℝ) (h1 : r = 50) (h2 : third_leg = 70) :
  let diameter := 2 * r
  let second_leg := Real.sqrt (diameter^2 - third_leg^2)
  diameter + third_leg + second_leg = 170 + Real.sqrt 5100 := by
sorry

end NUMINAMATH_CALUDE_spider_journey_l3358_335805


namespace NUMINAMATH_CALUDE_smallest_cube_for_pyramid_l3358_335860

/-- Represents a triangular pyramid with a given height and base side length -/
structure TriangularPyramid where
  height : ℝ
  baseSide : ℝ

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ :=
  c.sideLength ^ 3

/-- Checks if a cube can contain a triangular pyramid upright -/
def canContainPyramid (c : Cube) (p : TriangularPyramid) : Prop :=
  c.sideLength ≥ p.height ∧ c.sideLength ≥ p.baseSide

/-- The main theorem statement -/
theorem smallest_cube_for_pyramid (p : TriangularPyramid)
    (h1 : p.height = 15)
    (h2 : p.baseSide = 12) :
    ∃ (c : Cube), canContainPyramid c p ∧
    cubeVolume c = 3375 ∧
    ∀ (c' : Cube), canContainPyramid c' p → cubeVolume c' ≥ cubeVolume c := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_for_pyramid_l3358_335860


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3358_335829

/-- Given that 3/4 of 16 bananas are worth 10 oranges, 
    prove that 3/5 of 15 bananas are worth 7.5 oranges -/
theorem banana_orange_equivalence :
  (3 / 4 : ℚ) * 16 * (1 / 10 : ℚ) = 1 →
  (3 / 5 : ℚ) * 15 * (1 / 10 : ℚ) = (15 / 2 : ℚ) * (1 / 10 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3358_335829


namespace NUMINAMATH_CALUDE_matthews_cracker_distribution_l3358_335833

theorem matthews_cracker_distribution (total_crackers : ℕ) (crackers_per_friend : ℕ) (num_friends : ℕ) :
  total_crackers = 36 →
  crackers_per_friend = 6 →
  total_crackers = crackers_per_friend * num_friends →
  num_friends = 6 := by
sorry

end NUMINAMATH_CALUDE_matthews_cracker_distribution_l3358_335833


namespace NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3358_335880

/-- A quadratic expression is a square of a binomial if and only if its coefficients satisfy certain conditions -/
theorem quadratic_is_square_of_binomial (b : ℝ) : 
  (∃ (t u : ℝ), ∀ x, b * x^2 + 8 * x + 4 = (t * x + u)^2) ↔ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_square_of_binomial_l3358_335880


namespace NUMINAMATH_CALUDE_circular_table_dice_probability_l3358_335873

/-- The number of people sitting around the circular table -/
def num_people : ℕ := 5

/-- The number of sides on each die -/
def die_sides : ℕ := 8

/-- The probability that two adjacent people roll different numbers -/
def prob_different_adjacent : ℚ := 7 / 8

/-- The probability that no two adjacent people roll the same number -/
def prob_no_adjacent_same : ℚ := (prob_different_adjacent) ^ num_people

theorem circular_table_dice_probability :
  prob_no_adjacent_same = (7 / 8) ^ 5 :=
sorry

end NUMINAMATH_CALUDE_circular_table_dice_probability_l3358_335873


namespace NUMINAMATH_CALUDE_lost_ship_depth_l3358_335856

/-- The depth of a lost ship given the descent rate and time taken to reach it. -/
theorem lost_ship_depth (rate : ℝ) (time : ℝ) (h1 : rate = 32) (h2 : time = 200) :
  rate * time = 6400 := by
  sorry

end NUMINAMATH_CALUDE_lost_ship_depth_l3358_335856


namespace NUMINAMATH_CALUDE_one_third_percent_of_180_l3358_335838

theorem one_third_percent_of_180 : (1 / 3) * (1 / 100) * 180 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_one_third_percent_of_180_l3358_335838


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3358_335821

theorem triangle_perimeter : 
  let A : ℝ × ℝ := (2, -2)
  let B : ℝ × ℝ := (8, 4)
  let C : ℝ × ℝ := (2, 4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let perimeter := dist A B + dist B C + dist C A
  perimeter = 12 + 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3358_335821


namespace NUMINAMATH_CALUDE_triangle_side_length_l3358_335888

theorem triangle_side_length 
  (A B C : Real) 
  (a b c : Real) 
  (h_area : (1/2) * b * c * Real.sin A = Real.sqrt 3)
  (h_angle : B = 60 * π / 180)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3358_335888


namespace NUMINAMATH_CALUDE_solve_journey_problem_l3358_335871

def journey_problem (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : Prop :=
  let half_distance := (total_time * speed1 * speed2) / (speed1 + speed2)
  total_time = half_distance / speed1 + half_distance / speed2 →
  2 * half_distance = 240

theorem solve_journey_problem :
  journey_problem 20 10 15 := by
  sorry

end NUMINAMATH_CALUDE_solve_journey_problem_l3358_335871


namespace NUMINAMATH_CALUDE_function_properties_l3358_335827

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 + a

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x : ℝ, f a (x + π) = f a x) ∧
  (∃ x_min : ℝ, ∀ x : ℝ, f a x_min ≤ f a x) ∧
  (∃ x_min : ℝ, f a x_min = 0) →
  (a = 1) ∧
  (∀ x : ℝ, f a x ≤ 4) ∧
  (∃ k : ℤ, ∀ x : ℝ, f a x = f a (k * π / 2 + π / 6 - x)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3358_335827


namespace NUMINAMATH_CALUDE_correct_shirt_price_l3358_335864

-- Define the price of one shirt
def shirt_price : ℝ := 10

-- Define the cost of two shirts
def cost_two_shirts (p : ℝ) : ℝ := 1.5 * p

-- Define the cost of three shirts
def cost_three_shirts (p : ℝ) : ℝ := 1.9 * p

-- Define the savings when buying three shirts
def savings_three_shirts (p : ℝ) : ℝ := 3 * p - cost_three_shirts p

-- Theorem stating that the shirt price is correct
theorem correct_shirt_price :
  cost_two_shirts shirt_price = 1.5 * shirt_price ∧
  cost_three_shirts shirt_price = 1.9 * shirt_price ∧
  savings_three_shirts shirt_price = 11 :=
by sorry

end NUMINAMATH_CALUDE_correct_shirt_price_l3358_335864


namespace NUMINAMATH_CALUDE_table_leg_problem_l3358_335881

theorem table_leg_problem :
  ∀ (x y : ℕ),
    x ≥ 2 →
    y ≥ 2 →
    3 * x + 4 * y = 23 →
    x = 5 ∧ y = 2 :=
by sorry

end NUMINAMATH_CALUDE_table_leg_problem_l3358_335881
