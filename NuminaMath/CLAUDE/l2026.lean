import Mathlib

namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2026_202606

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : 
  (-1 + 3*x) * (-3*x - 1) = 1 - 9*x^2 := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) : 
  (x + 1)^2 - (1 - 3*x) * (1 + 3*x) = 10*x^2 + 2*x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l2026_202606


namespace NUMINAMATH_CALUDE_min_a4_value_l2026_202698

theorem min_a4_value (a : Fin 10 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_a2 : a 2 = a 1 + a 5)
  (h_a3 : a 3 = a 2 + a 6)
  (h_a4 : a 4 = a 3 + a 7)
  (h_a6 : a 6 = a 5 + a 8)
  (h_a7 : a 7 = a 6 + a 9)
  (h_a9 : a 9 = a 8 + a 10) :
  ∀ b : Fin 10 → ℕ+, 
    (∀ i j, i ≠ j → b i ≠ b j) →
    (b 2 = b 1 + b 5) →
    (b 3 = b 2 + b 6) →
    (b 4 = b 3 + b 7) →
    (b 6 = b 5 + b 8) →
    (b 7 = b 6 + b 9) →
    (b 9 = b 8 + b 10) →
    a 4 ≤ b 4 :=
by sorry

#check min_a4_value

end NUMINAMATH_CALUDE_min_a4_value_l2026_202698


namespace NUMINAMATH_CALUDE_square_difference_equality_l2026_202682

theorem square_difference_equality : 1005^2 - 995^2 - 1004^2 + 996^2 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2026_202682


namespace NUMINAMATH_CALUDE_g_neg_one_eq_zero_l2026_202692

/-- The function g(x) as defined in the problem -/
def g (s : ℝ) (x : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + 4 * x^2 - 5 * x + s

/-- Theorem stating that g(-1) = 0 when s = -14 -/
theorem g_neg_one_eq_zero :
  g (-14) (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_g_neg_one_eq_zero_l2026_202692


namespace NUMINAMATH_CALUDE_joe_rounding_threshold_l2026_202666

/-- A grade is a nonnegative rational number -/
def Grade := { x : ℚ // 0 ≤ x }

/-- Joe's rounding function -/
noncomputable def joeRound (x : Grade) : ℕ :=
  sorry

/-- The smallest rational number M such that any grade x ≥ M gets rounded to at least 90 -/
def M : ℚ := 805 / 9

theorem joe_rounding_threshold :
  ∀ (x : Grade), joeRound x ≥ 90 ↔ x.val ≥ M :=
sorry

end NUMINAMATH_CALUDE_joe_rounding_threshold_l2026_202666


namespace NUMINAMATH_CALUDE_redskins_win_streak_probability_l2026_202613

/-- The probability of arranging wins and losses in exactly three winning streaks -/
theorem redskins_win_streak_probability 
  (total_games : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (h1 : total_games = wins + losses)
  (h2 : wins = 10)
  (h3 : losses = 6) :
  (Nat.choose 9 2 * Nat.choose 7 3 : ℚ) / Nat.choose total_games losses = 45 / 286 := by
sorry

end NUMINAMATH_CALUDE_redskins_win_streak_probability_l2026_202613


namespace NUMINAMATH_CALUDE_no_self_referential_function_l2026_202616

theorem no_self_referential_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), n > 1 → f n = f (f (n - 1)) + f (f (n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_no_self_referential_function_l2026_202616


namespace NUMINAMATH_CALUDE_divide_negative_four_by_two_l2026_202699

theorem divide_negative_four_by_two : -4 / 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_divide_negative_four_by_two_l2026_202699


namespace NUMINAMATH_CALUDE_factorial_sum_unit_digit_l2026_202693

theorem factorial_sum_unit_digit : (Nat.factorial 25 + Nat.factorial 17 - Nat.factorial 18) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_unit_digit_l2026_202693


namespace NUMINAMATH_CALUDE_cistern_problem_l2026_202614

/-- Calculates the total wet surface area of a rectangular cistern -/
def cistern_wet_surface_area (length width depth : ℝ) : ℝ :=
  let bottom_area := length * width
  let long_sides_area := 2 * length * depth
  let short_sides_area := 2 * width * depth
  bottom_area + long_sides_area + short_sides_area

/-- Theorem stating that for a cistern with given dimensions, the wet surface area is 88 square meters -/
theorem cistern_problem : 
  cistern_wet_surface_area 12 4 1.25 = 88 := by
  sorry

#eval cistern_wet_surface_area 12 4 1.25

end NUMINAMATH_CALUDE_cistern_problem_l2026_202614


namespace NUMINAMATH_CALUDE_circle_equation_from_parabola_focus_l2026_202685

/-- Given a parabola y^2 = 4x and a circle with its center at the focus of the parabola
    passing through the origin, the equation of the circle is x^2 + y^2 - 2x = 0 -/
theorem circle_equation_from_parabola_focus (x y : ℝ) :
  (y^2 = 4*x) →  -- Parabola equation
  (∃ (h k r : ℝ), (h = 1 ∧ k = 0) ∧  -- Focus at (1, 0)
    ((0 - h)^2 + (0 - k)^2 = r^2) ∧  -- Circle passes through origin
    ((x - h)^2 + (y - k)^2 = r^2)) →  -- General circle equation
  x^2 + y^2 - 2*x = 0 :=  -- Resulting circle equation
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_parabola_focus_l2026_202685


namespace NUMINAMATH_CALUDE_problem_solution_l2026_202670

theorem problem_solution (w x y : ℝ) 
  (h1 : 6 / w + 6 / x = 6 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  x = 0.5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2026_202670


namespace NUMINAMATH_CALUDE_bart_tree_cutting_l2026_202624

/-- The number of pieces of firewood obtained from one tree -/
def pieces_per_tree : ℕ := 75

/-- The number of pieces of firewood Bart burns daily -/
def daily_burn_rate : ℕ := 5

/-- The number of days from November 1 through February 28 -/
def total_days : ℕ := 120

/-- The number of trees Bart needs to cut down -/
def trees_needed : ℕ := (daily_burn_rate * total_days) / pieces_per_tree

theorem bart_tree_cutting :
  trees_needed = 8 :=
sorry

end NUMINAMATH_CALUDE_bart_tree_cutting_l2026_202624


namespace NUMINAMATH_CALUDE_equation_solution_range_l2026_202694

theorem equation_solution_range (a : ℝ) (m : ℝ) :
  a > 0 ∧ a ≠ 1 →
  (∃ x : ℝ, a^(2*x) + (1 + 1/m)*a^x + 1 = 0) ↔
  -1/3 ≤ m ∧ m < 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2026_202694


namespace NUMINAMATH_CALUDE_range_of_b_l2026_202671

theorem range_of_b (b : ℝ) : 
  (¬ ∀ x : ℝ, x^2 - 4*b*x + 3*b > 0) ↔ (b ≤ 0 ∨ b ≥ 3/4) := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l2026_202671


namespace NUMINAMATH_CALUDE_sundae_price_l2026_202653

theorem sundae_price (ice_cream_bars sundaes : ℕ) (total_price ice_cream_price : ℚ) :
  ice_cream_bars = 200 →
  sundaes = 200 →
  total_price = 200 →
  ice_cream_price = 0.4 →
  (total_price - (ice_cream_bars : ℚ) * ice_cream_price) / (sundaes : ℚ) = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sundae_price_l2026_202653


namespace NUMINAMATH_CALUDE_fermat_little_theorem_l2026_202654

theorem fermat_little_theorem (N p : ℕ) (hp : Prime p) (hN : ¬ p ∣ N) :
  p ∣ (N^(p - 1) - 1) := by
  sorry

end NUMINAMATH_CALUDE_fermat_little_theorem_l2026_202654


namespace NUMINAMATH_CALUDE_max_value_abcd_l2026_202610

theorem max_value_abcd (a b c d : ℤ) (hb : b > 0) 
  (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) : 
  (∀ a' b' c' d' : ℤ, b' > 0 → a' + b' = c' → b' + c' = d' → c' + d' = a' → 
    a' - 2*b' + 3*c' - 4*d' ≤ a - 2*b + 3*c - 4*d) ∧ 
  (a - 2*b + 3*c - 4*d = -7) := by
  sorry

end NUMINAMATH_CALUDE_max_value_abcd_l2026_202610


namespace NUMINAMATH_CALUDE_factorial_6_equals_720_l2026_202612

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_6_equals_720 : factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_factorial_6_equals_720_l2026_202612


namespace NUMINAMATH_CALUDE_frank_problems_per_type_is_30_l2026_202665

/-- The number of math problems composed by Bill -/
def bill_problems : ℕ := 20

/-- The number of math problems composed by Ryan -/
def ryan_problems : ℕ := 2 * bill_problems

/-- The number of math problems composed by Frank -/
def frank_problems : ℕ := 3 * ryan_problems

/-- The number of different types of math problems -/
def problem_types : ℕ := 4

/-- The number of problems of each type that Frank composes -/
def frank_problems_per_type : ℕ := frank_problems / problem_types

theorem frank_problems_per_type_is_30 :
  frank_problems_per_type = 30 := by sorry

end NUMINAMATH_CALUDE_frank_problems_per_type_is_30_l2026_202665


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2026_202662

/-- Proves that for a hyperbola x²/a² - y²/b² = 1 with a > b, 
    if the angle between its asymptotes is 45°, then a/b = 1 + √2 -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) → 
  (Real.pi / 4 = Real.arctan ((b/a - (-b/a)) / (1 + (b/a) * (-b/a)))) →
  a / b = 1 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2026_202662


namespace NUMINAMATH_CALUDE_root_in_interval_l2026_202602

def f (x : ℝ) := 3 * x^2 + 3 * x - 8

theorem root_in_interval :
  (f 1.25 < 0) → (f 1.5 > 0) →
  ∃ x ∈ Set.Ioo 1.25 1.5, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l2026_202602


namespace NUMINAMATH_CALUDE_unique_solution_l2026_202632

/-- A is a 200-digit number starting with 89 and ending with 2525 -/
def A : ℕ := 89252525 -- (simplified for representation)

/-- B is a number of the form 444x18y27 where x and y are single digits -/
def B (x y : ℕ) : ℕ := 444 * x * 100000 + 18 * 1000 + y * 10 + 27

/-- Get the nth digit from the right of a number -/
def nthDigitFromRight (n : ℕ) (num : ℕ) : ℕ :=
  (num / (10^(n-1))) % 10

/-- The main theorem -/
theorem unique_solution :
  ∃! (x y : ℕ), x < 10 ∧ y < 10 ∧
  nthDigitFromRight 53 (A * B x y) = 1 ∧
  nthDigitFromRight 54 (A * B x y) = 0 ∧
  x = 4 ∧ y = 6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2026_202632


namespace NUMINAMATH_CALUDE_cos_angle_BHD_value_l2026_202652

structure RectangularSolid where
  angle_DHG : ℝ
  angle_FHB : ℝ

def cos_angle_BHD (solid : RectangularSolid) : ℝ := sorry

theorem cos_angle_BHD_value (solid : RectangularSolid) 
  (h1 : solid.angle_DHG = π/3)  -- 60 degrees in radians
  (h2 : solid.angle_FHB = π/4)  -- 45 degrees in radians
  : cos_angle_BHD solid = -Real.sqrt 30 / 12 := by sorry

end NUMINAMATH_CALUDE_cos_angle_BHD_value_l2026_202652


namespace NUMINAMATH_CALUDE_backyard_sod_coverage_l2026_202664

/-- Represents the dimensions of a rectangular section -/
structure Section where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular section -/
def sectionArea (s : Section) : ℕ := s.length * s.width

/-- Represents the dimensions of a sod square -/
structure SodSquare where
  side : ℕ

/-- Calculates the area of a sod square -/
def sodSquareArea (s : SodSquare) : ℕ := s.side * s.side

/-- Calculates the number of sod squares needed to cover a given area -/
def sodSquaresNeeded (totalArea : ℕ) (sodSquare : SodSquare) : ℕ :=
  totalArea / sodSquareArea sodSquare

theorem backyard_sod_coverage (section1 : Section) (section2 : Section) (sodSquare : SodSquare) :
  section1.length = 30 →
  section1.width = 40 →
  section2.length = 60 →
  section2.width = 80 →
  sodSquare.side = 2 →
  sodSquaresNeeded (sectionArea section1 + sectionArea section2) sodSquare = 1500 := by
  sorry

end NUMINAMATH_CALUDE_backyard_sod_coverage_l2026_202664


namespace NUMINAMATH_CALUDE_sqrt_x_plus_3_real_l2026_202673

theorem sqrt_x_plus_3_real (x : ℝ) : (∃ y : ℝ, y^2 = x + 3) ↔ x ≥ -3 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_3_real_l2026_202673


namespace NUMINAMATH_CALUDE_more_heads_probability_l2026_202636

def coin_prob : ℚ := 2/3

def num_flips : ℕ := 5

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

def more_heads_prob : ℚ :=
  binomial_probability num_flips 3 coin_prob +
  binomial_probability num_flips 4 coin_prob +
  binomial_probability num_flips 5 coin_prob

theorem more_heads_probability :
  more_heads_prob = 64/81 := by sorry

end NUMINAMATH_CALUDE_more_heads_probability_l2026_202636


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l2026_202644

theorem divisibility_by_twelve (m : Nat) : m ≤ 9 → (365 * 10 + m) % 12 = 0 ↔ m = 0 := by sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l2026_202644


namespace NUMINAMATH_CALUDE_matrix_multiplication_l2026_202650

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 0, 2]
def C : Matrix (Fin 2) (Fin 2) ℤ := !![15, -7; 20, -16]

theorem matrix_multiplication :
  A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l2026_202650


namespace NUMINAMATH_CALUDE_sphere_surface_area_of_inscribed_cuboid_l2026_202626

theorem sphere_surface_area_of_inscribed_cuboid (a b c : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 6) (h3 : c = 3) :
  let d := Real.sqrt (a^2 + b^2 + c^2)
  let r := d / 2
  4 * Real.pi * r^2 = 16 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_of_inscribed_cuboid_l2026_202626


namespace NUMINAMATH_CALUDE_nancy_crystal_beads_l2026_202680

/-- Proves that Nancy bought exactly 1 set of crystal beads given the problem conditions --/
theorem nancy_crystal_beads :
  ∀ (crystal_price metal_price total_spent metal_sets : ℕ) (crystal_sets : ℕ),
    crystal_price = 9 →
    metal_price = 10 →
    total_spent = 29 →
    metal_sets = 2 →
    crystal_price * crystal_sets + metal_price * metal_sets = total_spent →
    crystal_sets = 1 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crystal_beads_l2026_202680


namespace NUMINAMATH_CALUDE_a_investment_value_l2026_202647

/-- Represents the investment and profit distribution in a partnership business -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that given the conditions of the problem, A's investment is 8000 -/
theorem a_investment_value (p : Partnership)
  (hb : p.b_investment = 4000)
  (hc : p.c_investment = 2000)
  (hprofit : p.total_profit = 252000)
  (hshare : p.c_profit_share = 36000)
  : p.a_investment = 8000 := by
  sorry

end NUMINAMATH_CALUDE_a_investment_value_l2026_202647


namespace NUMINAMATH_CALUDE_jihye_marbles_l2026_202649

/-- Given a total number of marbles and the difference between two people's marbles,
    calculate the number of marbles the person with more marbles has. -/
def marblesWithMore (total : ℕ) (difference : ℕ) : ℕ :=
  (total + difference) / 2

/-- Theorem stating that given 85 total marbles and a difference of 11,
    the person with more marbles has 48 marbles. -/
theorem jihye_marbles : marblesWithMore 85 11 = 48 := by
  sorry

end NUMINAMATH_CALUDE_jihye_marbles_l2026_202649


namespace NUMINAMATH_CALUDE_hyperbola_angle_in_fourth_quadrant_l2026_202696

/-- Represents a hyperbola equation with angle α -/
def hyperbola_equation (x y α : ℝ) : Prop :=
  x^2 * Real.sin α + y^2 * Real.cos α = 1

/-- Indicates that the foci of the hyperbola are on the y-axis -/
def foci_on_y_axis (α : ℝ) : Prop :=
  Real.cos α > 0 ∧ Real.sin α < 0

/-- Indicates that an angle is in the fourth quadrant -/
def fourth_quadrant (α : ℝ) : Prop :=
  Real.cos α > 0 ∧ Real.sin α < 0

theorem hyperbola_angle_in_fourth_quadrant (α : ℝ) 
  (h1 : ∃ x y : ℝ, hyperbola_equation x y α)
  (h2 : foci_on_y_axis α) : 
  fourth_quadrant α :=
sorry

end NUMINAMATH_CALUDE_hyperbola_angle_in_fourth_quadrant_l2026_202696


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l2026_202627

theorem abs_sum_minimum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 7 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 7 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l2026_202627


namespace NUMINAMATH_CALUDE_f_monotonicity_l2026_202618

def f (m n : ℕ) (x : ℝ) : ℝ := x^(m/n)

theorem f_monotonicity (m n : ℕ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ → f m n x₁ < f m n x₂) ∧
  (n % 2 = 1 ∧ m % 2 = 0 → ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f m n x₁ > f m n x₂) ∧
  (n % 2 = 1 ∧ m % 2 = 1 → ∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 0 → f m n x₁ < f m n x₂) :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_l2026_202618


namespace NUMINAMATH_CALUDE_product_first_three_terms_l2026_202679

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The first term of the sequence
  a : ℝ
  -- The common difference between consecutive terms
  d : ℝ
  -- The seventh term is 20
  seventh_term : a + 6 * d = 20
  -- The common difference is 2
  common_diff : d = 2

/-- The product of the first three terms of the arithmetic sequence is 960 -/
theorem product_first_three_terms (seq : ArithmeticSequence) :
  seq.a * (seq.a + seq.d) * (seq.a + 2 * seq.d) = 960 := by
  sorry


end NUMINAMATH_CALUDE_product_first_three_terms_l2026_202679


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l2026_202620

theorem max_value_sum_of_roots (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧
  (Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ max) ∧
  (∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 1 ∧
    Real.sqrt (3 * a₀ + 1) + Real.sqrt (3 * b₀ + 1) + Real.sqrt (3 * c₀ + 1) = max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l2026_202620


namespace NUMINAMATH_CALUDE_three_piece_suit_cost_l2026_202648

/-- The cost of a jacket in pounds -/
def jacket_cost : ℝ := sorry

/-- The cost of a pair of trousers in pounds -/
def trousers_cost : ℝ := sorry

/-- The cost of a waistcoat in pounds -/
def waistcoat_cost : ℝ := sorry

/-- Two jackets and three pairs of trousers cost £380 -/
axiom two_jackets_three_trousers : 2 * jacket_cost + 3 * trousers_cost = 380

/-- A pair of trousers costs the same as two waistcoats -/
axiom trousers_equals_two_waistcoats : trousers_cost = 2 * waistcoat_cost

/-- The cost of a three-piece suit is £190 -/
theorem three_piece_suit_cost : jacket_cost + trousers_cost + waistcoat_cost = 190 := by
  sorry

end NUMINAMATH_CALUDE_three_piece_suit_cost_l2026_202648


namespace NUMINAMATH_CALUDE_implication_equivalence_l2026_202630

theorem implication_equivalence (P Q : Prop) : 
  (P → Q) ↔ (¬Q → ¬P) :=
sorry

end NUMINAMATH_CALUDE_implication_equivalence_l2026_202630


namespace NUMINAMATH_CALUDE_exactly_one_incorrect_statement_l2026_202660

/-- Represents a statement about regression analysis -/
inductive RegressionStatement
  | residualBand
  | scatterPlotCorrelation
  | regressionLineInterpretation
  | sumSquaredResiduals

/-- Determines if a given statement about regression analysis is correct -/
def isCorrect (statement : RegressionStatement) : Prop :=
  match statement with
  | .residualBand => True
  | .scatterPlotCorrelation => False
  | .regressionLineInterpretation => True
  | .sumSquaredResiduals => True

theorem exactly_one_incorrect_statement :
  ∃! (s : RegressionStatement), ¬(isCorrect s) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_incorrect_statement_l2026_202660


namespace NUMINAMATH_CALUDE_simplify_expression_l2026_202678

theorem simplify_expression (a b : ℝ) :
  (32 * a + 45 * b) + (15 * a + 36 * b) - (27 * a + 41 * b) = 20 * a + 40 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2026_202678


namespace NUMINAMATH_CALUDE_scientific_notation_58000_l2026_202631

theorem scientific_notation_58000 : 
  58000 = 5.8 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_58000_l2026_202631


namespace NUMINAMATH_CALUDE_total_selling_price_usd_l2026_202684

/-- Calculate the total selling price in USD for three articles given their purchase prices,
    exchange rates, and profit percentages. -/
theorem total_selling_price_usd 
  (purchase_price_eur : ℝ) (purchase_price_gbp : ℝ) (purchase_price_usd : ℝ)
  (initial_exchange_rate_eur : ℝ) (initial_exchange_rate_gbp : ℝ)
  (new_exchange_rate_eur : ℝ) (new_exchange_rate_gbp : ℝ)
  (profit_percent_1 : ℝ) (profit_percent_2 : ℝ) (profit_percent_3 : ℝ)
  (h1 : purchase_price_eur = 600)
  (h2 : purchase_price_gbp = 450)
  (h3 : purchase_price_usd = 750)
  (h4 : initial_exchange_rate_eur = 1.1)
  (h5 : initial_exchange_rate_gbp = 1.3)
  (h6 : new_exchange_rate_eur = 1.15)
  (h7 : new_exchange_rate_gbp = 1.25)
  (h8 : profit_percent_1 = 0.08)
  (h9 : profit_percent_2 = 0.10)
  (h10 : profit_percent_3 = 0.15) :
  let selling_price_1 := purchase_price_eur * (1 + profit_percent_1) * new_exchange_rate_eur
  let selling_price_2 := purchase_price_gbp * (1 + profit_percent_2) * new_exchange_rate_gbp
  let selling_price_3 := purchase_price_usd * (1 + profit_percent_3)
  selling_price_1 + selling_price_2 + selling_price_3 = 2225.85 := by
  sorry


end NUMINAMATH_CALUDE_total_selling_price_usd_l2026_202684


namespace NUMINAMATH_CALUDE_circumscribed_radius_of_sector_l2026_202672

/-- The radius of the circle circumscribed about a sector with a central angle of 120° cut from a circle of radius 8 is equal to 8√3/3. -/
theorem circumscribed_radius_of_sector (r : ℝ) (θ : ℝ) : 
  r = 8 → θ = 2 * π / 3 → (8 * Real.sqrt 3) / 3 = r / (2 * Real.sin (θ / 2)) := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_radius_of_sector_l2026_202672


namespace NUMINAMATH_CALUDE_test_score_for_three_hours_l2026_202661

/-- A model for a test score based on preparation time. -/
structure TestScore where
  maxPoints : ℝ
  scoreFunction : ℝ → ℝ
  knownScore : ℝ
  knownTime : ℝ

/-- Theorem: Given the conditions, prove that 3 hours of preparation results in a score of 202.5 -/
theorem test_score_for_three_hours 
  (test : TestScore)
  (h1 : test.maxPoints = 150)
  (h2 : ∀ t, test.scoreFunction t = (test.knownScore / test.knownTime^2) * t^2)
  (h3 : test.knownScore = 90)
  (h4 : test.knownTime = 2) :
  test.scoreFunction 3 = 202.5 := by
  sorry


end NUMINAMATH_CALUDE_test_score_for_three_hours_l2026_202661


namespace NUMINAMATH_CALUDE_range_of_a_l2026_202625

theorem range_of_a (a : ℝ) : 
  (∀ t : ℝ, t^2 - a*t - a ≥ 0) → -4 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2026_202625


namespace NUMINAMATH_CALUDE_square_property_of_natural_numbers_l2026_202628

theorem square_property_of_natural_numbers (a b : ℕ) 
  (h1 : ∃ k : ℕ, a * b = k^2)
  (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m^2) :
  ∃ n : ℕ, 
    2 < n ∧ 
    Even n ∧ 
    ∃ p : ℕ, (a + n) * (b + n) = p^2 := by
  sorry

end NUMINAMATH_CALUDE_square_property_of_natural_numbers_l2026_202628


namespace NUMINAMATH_CALUDE_unique_triple_solution_l2026_202634

theorem unique_triple_solution (a b c : ℝ) : 
  a > 5 ∧ b > 5 ∧ c > 5 ∧
  ((a + 3)^2 / (b + c - 5) + (b + 5)^2 / (c + a - 7) + (c + 7)^2 / (a + b - 9) = 49) →
  a = 13 ∧ b = 9 ∧ c = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_solution_l2026_202634


namespace NUMINAMATH_CALUDE_grace_age_calculation_l2026_202604

/-- Grace's age in years -/
def grace_age : ℕ := sorry

/-- Grace's mother's age in years -/
def mother_age : ℕ := 80

/-- Grace's grandmother's age in years -/
def grandmother_age : ℕ := sorry

/-- Theorem stating Grace's age based on the given conditions -/
theorem grace_age_calculation :
  (grace_age = 3 * grandmother_age / 8) ∧
  (grandmother_age = 2 * mother_age) ∧
  (mother_age = 80) →
  grace_age = 60 := by
    sorry

end NUMINAMATH_CALUDE_grace_age_calculation_l2026_202604


namespace NUMINAMATH_CALUDE_union_of_sets_l2026_202659

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {1, 3, 5}
  A ∪ B = {1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2026_202659


namespace NUMINAMATH_CALUDE_biquadratic_equation_with_given_root_l2026_202646

theorem biquadratic_equation_with_given_root (x : ℝ) :
  (2 + Real.sqrt 3 : ℝ) ^ 4 - 14 * (2 + Real.sqrt 3 : ℝ) ^ 2 + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_biquadratic_equation_with_given_root_l2026_202646


namespace NUMINAMATH_CALUDE_product_remainder_mod_17_l2026_202669

theorem product_remainder_mod_17 : (2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_17_l2026_202669


namespace NUMINAMATH_CALUDE_vasechkin_result_l2026_202688

def petrov_operation (x : ℚ) : ℚ := (x / 2) * 7 - 1001

def vasechkin_operation (x : ℚ) : ℚ := (x / 8)^2 - 1001

theorem vasechkin_result :
  ∃ x : ℚ, (∃ p : ℕ, Nat.Prime p ∧ petrov_operation x = ↑p) →
  vasechkin_operation x = 295 :=
sorry

end NUMINAMATH_CALUDE_vasechkin_result_l2026_202688


namespace NUMINAMATH_CALUDE_simplify_expression_l2026_202615

theorem simplify_expression (x y : ℝ) : 20 * (x + y) - 19 * (y + x) = x + y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2026_202615


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l2026_202681

theorem triangle_angle_sum (P Q R : ℝ) (h : P + Q = 60) : R = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l2026_202681


namespace NUMINAMATH_CALUDE_sum_of_integers_l2026_202656

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2026_202656


namespace NUMINAMATH_CALUDE_oplus_five_two_l2026_202638

-- Define the operation ⊕
def oplus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- Theorem statement
theorem oplus_five_two : oplus 5 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_oplus_five_two_l2026_202638


namespace NUMINAMATH_CALUDE_seventeenth_term_is_two_l2026_202674

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (a 1 + a n) / 2
  sum_13 : sum 13 = 78
  sum_7_12 : a 7 + a 12 = 10

/-- The 17th term of the arithmetic sequence is 2 -/
theorem seventeenth_term_is_two (seq : ArithmeticSequence) : seq.a 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_seventeenth_term_is_two_l2026_202674


namespace NUMINAMATH_CALUDE_ratio_of_x_intercepts_l2026_202607

/-- Two lines with the same non-zero y-intercept, one with slope 8 and x-intercept (s, 0),
    the other with slope 4 and x-intercept (t, 0), have s/t = 1/2 -/
theorem ratio_of_x_intercepts (b s t : ℝ) (hb : b ≠ 0) : 
  (0 = 8 * s + b) → (0 = 4 * t + b) → s / t = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_x_intercepts_l2026_202607


namespace NUMINAMATH_CALUDE_percentage_of_filled_holes_l2026_202633

theorem percentage_of_filled_holes (total_holes : ℕ) (unfilled_holes : ℕ) 
  (h1 : total_holes = 8) 
  (h2 : unfilled_holes = 2) 
  (h3 : unfilled_holes < total_holes) : 
  (((total_holes - unfilled_holes) : ℚ) / total_holes) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_filled_holes_l2026_202633


namespace NUMINAMATH_CALUDE_board_coverage_uncoverable_boards_l2026_202675

/-- Represents a rectangular board, possibly with one square removed -/
structure Board where
  rows : Nat
  cols : Nat
  removed : Bool

/-- Calculates the total number of squares on a board -/
def Board.totalSquares (b : Board) : Nat :=
  b.rows * b.cols - if b.removed then 1 else 0

/-- Predicate for whether a board can be covered by dominoes -/
def canBeCovered (b : Board) : Prop :=
  b.totalSquares % 2 = 0

/-- Main theorem: A board can be covered iff its total squares is even -/
theorem board_coverage (b : Board) :
  canBeCovered b ↔ b.totalSquares % 2 = 0 := by sorry

/-- Specific boards from the problem -/
def board_3x4 : Board := { rows := 3, cols := 4, removed := false }
def board_3x5 : Board := { rows := 3, cols := 5, removed := false }
def board_4x4_removed : Board := { rows := 4, cols := 4, removed := true }
def board_5x5 : Board := { rows := 5, cols := 5, removed := false }
def board_6x3 : Board := { rows := 6, cols := 3, removed := false }

/-- Theorem about which boards cannot be covered -/
theorem uncoverable_boards :
  ¬(canBeCovered board_3x5) ∧
  ¬(canBeCovered board_4x4_removed) ∧
  ¬(canBeCovered board_5x5) ∧
  (canBeCovered board_3x4) ∧
  (canBeCovered board_6x3) := by sorry

end NUMINAMATH_CALUDE_board_coverage_uncoverable_boards_l2026_202675


namespace NUMINAMATH_CALUDE_xy_commutativity_l2026_202622

theorem xy_commutativity (x y : ℝ) : 10 * x * y - 10 * y * x = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_commutativity_l2026_202622


namespace NUMINAMATH_CALUDE_neighborhood_cable_cost_l2026_202677

/-- Calculates the total cost of cable for a neighborhood given the street layout and cable requirements. -/
theorem neighborhood_cable_cost
  (east_west_streets : ℕ)
  (east_west_length : ℝ)
  (north_south_streets : ℕ)
  (north_south_length : ℝ)
  (cable_per_street_mile : ℝ)
  (cable_cost_per_mile : ℝ)
  (h1 : east_west_streets = 18)
  (h2 : east_west_length = 2)
  (h3 : north_south_streets = 10)
  (h4 : north_south_length = 4)
  (h5 : cable_per_street_mile = 5)
  (h6 : cable_cost_per_mile = 2000) :
  (east_west_streets * east_west_length + north_south_streets * north_south_length) *
  cable_per_street_mile * cable_cost_per_mile = 760000 := by
  sorry


end NUMINAMATH_CALUDE_neighborhood_cable_cost_l2026_202677


namespace NUMINAMATH_CALUDE_positive_f_one_l2026_202645

def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem positive_f_one (f : ℝ → ℝ) 
    (h_mono : MonoIncreasing f) (h_odd : OddFunction f) : 
    f 1 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_f_one_l2026_202645


namespace NUMINAMATH_CALUDE_count_rectangles_3x6_grid_l2026_202600

/-- The number of rectangles in a 3 × 6 grid with vertices at grid points -/
def num_rectangles : ℕ :=
  let horizontal_lines := 4
  let vertical_lines := 7
  let horizontal_vertical_rectangles := (horizontal_lines.choose 2) * (vertical_lines.choose 2)
  let diagonal_sqrt2 := 5 * 2
  let diagonal_2sqrt2 := 4 * 2
  let diagonal_sqrt5 := 4 * 2
  horizontal_vertical_rectangles + diagonal_sqrt2 + diagonal_2sqrt2 + diagonal_sqrt5

theorem count_rectangles_3x6_grid :
  num_rectangles = 152 :=
sorry

end NUMINAMATH_CALUDE_count_rectangles_3x6_grid_l2026_202600


namespace NUMINAMATH_CALUDE_share_distribution_l2026_202639

theorem share_distribution (total : ℝ) (y_share : ℝ) (x_to_y_ratio : ℝ) :
  total = 273 →
  y_share = 63 →
  x_to_y_ratio = 0.45 →
  ∃ (x_share z_share : ℝ),
    y_share = x_to_y_ratio * x_share ∧
    total = x_share + y_share + z_share ∧
    z_share / x_share = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_share_distribution_l2026_202639


namespace NUMINAMATH_CALUDE_fixed_fee_is_7_42_l2026_202621

/-- Represents the billing structure and usage for an online service provider -/
structure BillingInfo where
  fixedFee : ℝ
  hourlyCharge : ℝ
  decemberUsage : ℝ
  januaryUsage : ℝ

/-- Calculates the total bill based on fixed fee, hourly charge, and usage -/
def calculateBill (info : BillingInfo) (usage : ℝ) : ℝ :=
  info.fixedFee + info.hourlyCharge * usage

/-- Theorem stating that under given conditions, the fixed monthly fee is $7.42 -/
theorem fixed_fee_is_7_42 (info : BillingInfo) :
  calculateBill info info.decemberUsage = 12.48 →
  calculateBill info info.januaryUsage = 17.54 →
  info.januaryUsage = 2 * info.decemberUsage →
  info.fixedFee = 7.42 := by
  sorry

#eval (7.42 : Float)

end NUMINAMATH_CALUDE_fixed_fee_is_7_42_l2026_202621


namespace NUMINAMATH_CALUDE_even_function_domain_l2026_202609

/-- A function f is even if f(x) = f(-x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The function f(x) = ax^2 + bx + 3a + b -/
def f (a b : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + 3 * a + b

theorem even_function_domain (a b : ℝ) :
  (IsEven (f a b)) ∧ (Set.Icc (2 * a) (a - 1)).Nonempty → a + b = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_even_function_domain_l2026_202609


namespace NUMINAMATH_CALUDE_negative_sqrt_of_squared_negative_nine_equals_negative_nine_l2026_202637

theorem negative_sqrt_of_squared_negative_nine_equals_negative_nine :
  -Real.sqrt ((-9)^2) = -9 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_of_squared_negative_nine_equals_negative_nine_l2026_202637


namespace NUMINAMATH_CALUDE_vector_subtraction_l2026_202658

def a : Fin 3 → ℝ := ![-3, 4, 2]
def b : Fin 3 → ℝ := ![5, -1, 3]

theorem vector_subtraction :
  (fun i => a i - 2 * b i) = ![-13, 6, -4] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l2026_202658


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l2026_202640

/-- 
Given that the cost price is 89% of the selling price, 
prove that the profit percentage is (100/89 - 1) * 100.
-/
theorem profit_percentage_calculation (selling_price : ℝ) (cost_price : ℝ) 
  (h : cost_price = 0.89 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100/89 - 1) * 100 := by
  sorry

#eval (100/89 - 1) * 100 -- This will output approximately 12.36

end NUMINAMATH_CALUDE_profit_percentage_calculation_l2026_202640


namespace NUMINAMATH_CALUDE_min_green_chips_l2026_202603

/-- Given a basket of chips with three colors: green, yellow, and violet.
    This theorem proves that the minimum number of green chips is 120,
    given the conditions stated in the problem. -/
theorem min_green_chips (y v g : ℕ) : 
  v ≥ (2 : ℕ) * y / 3 →  -- violet chips are at least two-thirds of yellow chips
  v ≤ g / 4 →            -- violet chips are at most one-fourth of green chips
  y + v ≥ 75 →           -- sum of yellow and violet chips is at least 75
  g ≥ 120 :=             -- prove that the minimum number of green chips is 120
by sorry

end NUMINAMATH_CALUDE_min_green_chips_l2026_202603


namespace NUMINAMATH_CALUDE_weekly_payment_problem_l2026_202611

/-- The weekly payment problem -/
theorem weekly_payment_problem (n_pay m_pay total_pay : ℕ) : 
  n_pay = 250 →
  m_pay = (120 * n_pay) / 100 →
  total_pay = m_pay + n_pay →
  total_pay = 550 := by
  sorry

end NUMINAMATH_CALUDE_weekly_payment_problem_l2026_202611


namespace NUMINAMATH_CALUDE_angle_4_value_l2026_202695

theorem angle_4_value (angle1 angle2 angle3 angle4 : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = 2 * angle4 →
  angle1 = 50 →
  angle3 + angle4 = 130 →
  angle4 = 130 / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_4_value_l2026_202695


namespace NUMINAMATH_CALUDE_leftHandedJazzLoversCount_l2026_202689

/-- Represents a club with members having different characteristics -/
structure Club where
  total : ℕ
  leftHanded : ℕ
  jazzLovers : ℕ
  rightHandedNonJazz : ℕ

/-- The number of left-handed jazz lovers in the club -/
def leftHandedJazzLovers (c : Club) : ℕ :=
  c.total - (c.leftHanded + c.jazzLovers - c.rightHandedNonJazz)

/-- Theorem stating the number of left-handed jazz lovers in the given club -/
theorem leftHandedJazzLoversCount (c : Club) 
  (h1 : c.total = 20)
  (h2 : c.leftHanded = 8)
  (h3 : c.jazzLovers = 15)
  (h4 : c.rightHandedNonJazz = 2) :
  leftHandedJazzLovers c = 5 := by
  sorry

#eval leftHandedJazzLovers { total := 20, leftHanded := 8, jazzLovers := 15, rightHandedNonJazz := 2 }

end NUMINAMATH_CALUDE_leftHandedJazzLoversCount_l2026_202689


namespace NUMINAMATH_CALUDE_lemonade_stand_cost_l2026_202676

-- Define the given conditions
def total_profit : ℝ := 44
def lemonade_revenue : ℝ := 47
def lemonades_sold : ℕ := 50
def babysitting_income : ℝ := 31
def lemon_cost : ℝ := 0.20
def sugar_cost : ℝ := 0.15
def ice_cost : ℝ := 0.05
def sunhat_cost : ℝ := 10

-- Define the theorem
theorem lemonade_stand_cost :
  let variable_cost_per_lemonade := lemon_cost + sugar_cost + ice_cost
  let total_variable_cost := variable_cost_per_lemonade * lemonades_sold
  let total_cost := total_variable_cost + sunhat_cost
  total_cost = 30 := by sorry

end NUMINAMATH_CALUDE_lemonade_stand_cost_l2026_202676


namespace NUMINAMATH_CALUDE_all_points_on_same_circle_l2026_202697

-- Define a type for points in the plane
variable (Point : Type)

-- Define a type for circles in the plane
variable (Circle : Type)

-- Define a function to check if a point lies on a circle
variable (lies_on : Point → Circle → Prop)

-- Define a function to create a circle from four points
variable (circle_from_four_points : Point → Point → Point → Point → Circle)

theorem all_points_on_same_circle 
  (P : Set Point) 
  (h : ∀ (a b c d : Point), a ∈ P → b ∈ P → c ∈ P → d ∈ P → 
    ∃ (C : Circle), lies_on a C ∧ lies_on b C ∧ lies_on c C ∧ lies_on d C) :
  ∃ (C : Circle), ∀ (p : Point), p ∈ P → lies_on p C :=
sorry

end NUMINAMATH_CALUDE_all_points_on_same_circle_l2026_202697


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2026_202657

theorem regular_polygon_sides (n : ℕ) (h_exterior : (360 : ℝ) / n = 30) 
  (h_interior : (180 : ℝ) - 30 = 150) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2026_202657


namespace NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l2026_202641

def late_flights : ℕ := 1
def initial_on_time_flights : ℕ := 3

def on_time_rate (additional_on_time : ℕ) : ℚ :=
  (initial_on_time_flights + additional_on_time) / (late_flights + initial_on_time_flights + additional_on_time)

theorem phoenix_airport_on_time_rate :
  ∃ n : ℕ, n > 0 ∧ on_time_rate n > (2 : ℚ) / 5 :=
sorry

end NUMINAMATH_CALUDE_phoenix_airport_on_time_rate_l2026_202641


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_third_l2026_202655

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1/3

/-- The reciprocal of the common fraction form of 0.333... is 3 --/
theorem reciprocal_of_repeating_third : (repeating_third⁻¹ : ℚ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_third_l2026_202655


namespace NUMINAMATH_CALUDE_problem_statement_l2026_202687

theorem problem_statement (number : ℝ) (value : ℝ) : 
  number = 1.375 →
  0.6667 * number + 0.75 = value →
  value = 1.666675 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2026_202687


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2026_202663

-- Define the vectors a and b
def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

-- Define the dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the scalar multiplication for 2D vectors
def scalar_mult (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (t * v.1, t * v.2)

-- Define vector addition for 2D vectors
def vector_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Theorem statement
theorem perpendicular_vectors (t : ℝ) : 
  dot_product a (vector_add (scalar_mult t a) b) = 0 → t = -5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2026_202663


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percent_l2026_202667

theorem shopkeeper_loss_percent 
  (profit_rate : ℝ) 
  (theft_rate : ℝ) 
  (initial_value : ℝ) 
  (profit_rate_is_10_percent : profit_rate = 0.1)
  (theft_rate_is_60_percent : theft_rate = 0.6)
  (initial_value_positive : initial_value > 0) : 
  let remaining_goods := initial_value * (1 - theft_rate)
  let final_value := remaining_goods * (1 + profit_rate)
  let loss := initial_value - final_value
  let loss_percent := (loss / initial_value) * 100
  loss_percent = 56 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percent_l2026_202667


namespace NUMINAMATH_CALUDE_kyle_age_l2026_202690

/-- Given the ages of several people and their relationships, prove Kyle's age --/
theorem kyle_age (david sandra casey fiona julian shelley kyle frederick tyson : ℕ) 
  (h1 : shelley = kyle - 3)
  (h2 : shelley = julian + 4)
  (h3 : julian = frederick - 20)
  (h4 : julian = fiona + 5)
  (h5 : frederick = 2 * tyson)
  (h6 : tyson = 2 * casey)
  (h7 : casey = fiona - 2)
  (h8 : 2 * casey = sandra)
  (h9 : sandra = david + 4)
  (h10 : david = 16) : 
  kyle = 23 := by sorry

end NUMINAMATH_CALUDE_kyle_age_l2026_202690


namespace NUMINAMATH_CALUDE_zero_lt_x_lt_two_sufficient_not_necessary_for_x_lt_two_l2026_202601

theorem zero_lt_x_lt_two_sufficient_not_necessary_for_x_lt_two :
  (∃ x : ℝ, 0 < x ∧ x < 2 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(0 < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_zero_lt_x_lt_two_sufficient_not_necessary_for_x_lt_two_l2026_202601


namespace NUMINAMATH_CALUDE_plums_picked_equals_127_l2026_202619

/-- Calculates the total number of plums picked by Alyssa and Jason after three hours -/
def total_plums_picked (alyssa_rate : ℕ) (jason_rate : ℕ) : ℕ :=
  let first_hour := alyssa_rate + jason_rate
  let second_hour := (3 * alyssa_rate) + (jason_rate + (2 * jason_rate / 5))
  let third_hour_before_drop := alyssa_rate + (2 * jason_rate)
  let third_hour_after_drop := third_hour_before_drop - (third_hour_before_drop / 14)
  first_hour + second_hour + third_hour_after_drop

/-- Theorem stating that the total number of plums picked is 127 -/
theorem plums_picked_equals_127 :
  total_plums_picked 17 10 = 127 := by
  sorry

#eval total_plums_picked 17 10

end NUMINAMATH_CALUDE_plums_picked_equals_127_l2026_202619


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l2026_202668

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l2026_202668


namespace NUMINAMATH_CALUDE_line_points_k_value_l2026_202623

/-- Given a line with equation x = 2y + 5 and two points (m, n) and (m + 3, n + k) on this line, k = 3/2. -/
theorem line_points_k_value (m n k : ℝ) : 
  (m = 2 * n + 5) → 
  (m + 3 = 2 * (n + k) + 5) → 
  k = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_line_points_k_value_l2026_202623


namespace NUMINAMATH_CALUDE_combination_equality_l2026_202605

theorem combination_equality (x : ℕ) : 
  (Nat.choose 18 x = Nat.choose 18 (3*x - 6)) → (x = 3 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_combination_equality_l2026_202605


namespace NUMINAMATH_CALUDE_function_periodicity_l2026_202635

/-- A function satisfying the given functional equation is periodic with period 8. -/
theorem function_periodicity (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f (x + 1) + f (x - 1) = Real.sqrt 2 * f x) : 
  ∀ x : ℝ, f (x + 8) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l2026_202635


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_mean_50_l2026_202643

theorem max_ratio_two_digit_mean_50 :
  ∀ x y : ℕ,
  10 ≤ x ∧ x ≤ 99 →
  10 ≤ y ∧ y ≤ 99 →
  (x + y) / 2 = 50 →
  ∀ a b : ℕ,
  10 ≤ a ∧ a ≤ 99 →
  10 ≤ b ∧ b ≤ 99 →
  (a + b) / 2 = 50 →
  x / y ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_mean_50_l2026_202643


namespace NUMINAMATH_CALUDE_cyclist_heartbeats_l2026_202691

/-- The number of heartbeats during a cycling race -/
def heartbeats_during_race (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Theorem: The cyclist's heart beats 57600 times during the race -/
theorem cyclist_heartbeats :
  heartbeats_during_race 120 4 120 = 57600 := by
  sorry

#eval heartbeats_during_race 120 4 120

end NUMINAMATH_CALUDE_cyclist_heartbeats_l2026_202691


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2026_202642

/-- Given that:
  - a and b are opposite numbers
  - c and d are reciprocals
  - The distance from point m to the origin is 5
Prove that m^2 - 100a - 99b - bcd + |cd - 2| = -74 -/
theorem algebraic_expression_value 
  (a b c d m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : c * d = 1) 
  (h3 : m^2 = 25) : 
  m^2 - 100*a - 99*b - b*c*d + |c*d - 2| = -74 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2026_202642


namespace NUMINAMATH_CALUDE_sum_reciprocal_squared_ge_sum_squared_l2026_202617

theorem sum_reciprocal_squared_ge_sum_squared 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hsum : a + b + c = 3) : 
  1/a^2 + 1/b^2 + 1/c^2 ≥ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_squared_ge_sum_squared_l2026_202617


namespace NUMINAMATH_CALUDE_function_periodicity_l2026_202608

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem function_periodicity
  (f : ℝ → ℝ)
  (h1 : ∀ x, |f x| ≤ 1)
  (h2 : ∀ x, f (x + 13/42) + f x = f (x + 1/7) + f (x + 1/6)) :
  is_periodic f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_periodicity_l2026_202608


namespace NUMINAMATH_CALUDE_second_boy_probability_l2026_202686

/-- Represents a student in the classroom -/
inductive Student : Type
| Boy : Student
| Girl : Student

/-- The type of all possible orders in which students can leave -/
def LeaveOrder := List Student

/-- Generate all possible leave orders for 2 boys and 2 girls -/
def allLeaveOrders : List LeaveOrder :=
  sorry

/-- Check if the second student in a leave order is a boy -/
def isSecondBoy (order : LeaveOrder) : Bool :=
  sorry

/-- Count the number of leave orders where the second student is a boy -/
def countSecondBoy (orders : List LeaveOrder) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem second_boy_probability (orders : List LeaveOrder) 
  (h1 : orders = allLeaveOrders) 
  (h2 : orders.length = 6) : 
  (countSecondBoy orders : ℚ) / (orders.length : ℚ) = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_second_boy_probability_l2026_202686


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2026_202629

theorem isosceles_triangle_perimeter (x₁ x₂ : ℝ) : 
  x₁^2 - 9*x₁ + 18 = 0 →
  x₂^2 - 9*x₂ + 18 = 0 →
  x₁ ≠ x₂ →
  (x₁ + x₂ + max x₁ x₂ = 15) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2026_202629


namespace NUMINAMATH_CALUDE_subgroup_normal_iff_power_commute_l2026_202683

variables {G : Type*} [Group G]

theorem subgroup_normal_iff_power_commute : 
  (∀ (H : Subgroup G), H.Normal) ↔ 
  (∀ (a b : G), ∃ (m : ℤ), (a * b) ^ m = b * a) :=
by sorry

end NUMINAMATH_CALUDE_subgroup_normal_iff_power_commute_l2026_202683


namespace NUMINAMATH_CALUDE_sum_triangle_quadrilateral_sides_l2026_202651

/-- A triangle is a shape with 3 sides -/
def Triangle : Nat := 3

/-- A quadrilateral is a shape with 4 sides -/
def Quadrilateral : Nat := 4

/-- The sum of the sides of a triangle and a quadrilateral is 7 -/
theorem sum_triangle_quadrilateral_sides : Triangle + Quadrilateral = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_triangle_quadrilateral_sides_l2026_202651
