import Mathlib

namespace NUMINAMATH_CALUDE_freshman_class_size_l1535_153523

theorem freshman_class_size :
  ∃! n : ℕ, n < 400 ∧ n % 26 = 17 ∧ n % 24 = 6 :=
by
  use 379
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l1535_153523


namespace NUMINAMATH_CALUDE_sum_of_dimensions_l1535_153514

-- Define the dimensions of the rectangular box
variable (X Y Z : ℝ)

-- Define the surface areas of the faces
def surfaceArea1 : ℝ := 18
def surfaceArea2 : ℝ := 18
def surfaceArea3 : ℝ := 36
def surfaceArea4 : ℝ := 36
def surfaceArea5 : ℝ := 54
def surfaceArea6 : ℝ := 54

-- State the theorem
theorem sum_of_dimensions (h1 : X * Y = surfaceArea1)
                          (h2 : X * Z = surfaceArea5)
                          (h3 : Y * Z = surfaceArea3)
                          (h4 : X > 0) (h5 : Y > 0) (h6 : Z > 0) :
  X + Y + Z = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_dimensions_l1535_153514


namespace NUMINAMATH_CALUDE_integral_sqrt_x_2_minus_x_l1535_153585

theorem integral_sqrt_x_2_minus_x (f : ℝ → ℝ) :
  (∀ x, f x = Real.sqrt (x * (2 - x))) →
  ∫ x in (0 : ℝ)..1, f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_integral_sqrt_x_2_minus_x_l1535_153585


namespace NUMINAMATH_CALUDE_mikes_age_l1535_153543

theorem mikes_age (mike anna : ℝ) 
  (h1 : mike = 3 * anna - 20)
  (h2 : mike + anna = 70) : 
  mike = 47.5 := by
sorry

end NUMINAMATH_CALUDE_mikes_age_l1535_153543


namespace NUMINAMATH_CALUDE_max_m_inequality_l1535_153540

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ m : ℝ, ∀ a b : ℝ, a > 0 → b > 0 → 4/a + 1/b ≥ m/(a+4*b)) ∧
  (∀ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → 4/a + 1/b ≥ m/(a+4*b)) → m ≤ 16) :=
sorry

end NUMINAMATH_CALUDE_max_m_inequality_l1535_153540


namespace NUMINAMATH_CALUDE_chicken_nugget_cost_100_l1535_153580

/-- The cost of chicken nuggets given the total number of nuggets and the cost per box -/
def chicken_nugget_cost (total_nuggets : ℕ) (nuggets_per_box : ℕ) (cost_per_box : ℕ) : ℕ :=
  (total_nuggets / nuggets_per_box) * cost_per_box

/-- Theorem: The cost of 100 chicken nuggets is $20 when a box of 20 costs $4 -/
theorem chicken_nugget_cost_100 :
  chicken_nugget_cost 100 20 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_chicken_nugget_cost_100_l1535_153580


namespace NUMINAMATH_CALUDE_quadratic_expression_evaluation_l1535_153599

theorem quadratic_expression_evaluation :
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  2 * x^2 + 3 * y^2 - z^2 + 4 * x * y = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_evaluation_l1535_153599


namespace NUMINAMATH_CALUDE_add_1457_minutes_to_3pm_l1535_153569

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := totalMinutes / 60 % 24
  let newMinutes := totalMinutes % 60
  { hours := newHours, minutes := newMinutes }

/-- Theorem: Adding 1457 minutes to 3:00 p.m. results in 3:17 p.m. the next day -/
theorem add_1457_minutes_to_3pm (initial : Time) (final : Time) :
  initial = { hours := 15, minutes := 0 } →
  final = addMinutes initial 1457 →
  final = { hours := 15, minutes := 17 } :=
by sorry

end NUMINAMATH_CALUDE_add_1457_minutes_to_3pm_l1535_153569


namespace NUMINAMATH_CALUDE_b_share_yearly_profit_l1535_153533

/-- Investment proportions and profit distribution for partners A, B, C, and D --/
structure Partnership where
  b_invest : ℝ  -- B's investment (base unit)
  a_invest : ℝ := 2.5 * b_invest  -- A's investment
  c_invest : ℝ := 1.5 * b_invest  -- C's investment
  d_invest : ℝ := 1.25 * b_invest  -- D's investment
  total_invest : ℝ := a_invest + b_invest + c_invest  -- Total investment of A, B, and C
  profit_6months : ℝ := 6000  -- Profit for 6 months
  d_fixed_amount : ℝ := 500  -- D's fixed amount per 6 months
  profit_year : ℝ := 16900  -- Total profit for the year

/-- Theorem stating B's share of the yearly profit --/
theorem b_share_yearly_profit (p : Partnership) :
  (p.b_invest / p.total_invest) * (p.profit_year - 2 * p.d_fixed_amount) = 3180 := by
  sorry

end NUMINAMATH_CALUDE_b_share_yearly_profit_l1535_153533


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_point_P_theorem_l1535_153586

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define the tangent line with equal intercepts
def tangent_equal_intercepts (k : ℝ) : Prop :=
  k = 2 + Real.sqrt 6 ∨ k = 2 - Real.sqrt 6 ∨ 
  (∀ x y : ℝ, x + y + 1 = 0) ∨ (∀ x y : ℝ, x + y - 3 = 0)

-- Define the point P outside the circle
def point_P (x y : ℝ) : Prop :=
  ¬ circle_C x y ∧ 2*x - 4*y + 3 = 0 ∧ 2*x + y = 0

-- Theorem for the tangent lines
theorem tangent_lines_theorem :
  ∃ k : ℝ, tangent_equal_intercepts k := by sorry

-- Theorem for the point P
theorem point_P_theorem :
  ∃ x y : ℝ, point_P x y ∧ x = -3/10 ∧ y = 3/5 := by sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_point_P_theorem_l1535_153586


namespace NUMINAMATH_CALUDE_childrens_cookbook_cost_l1535_153545

theorem childrens_cookbook_cost 
  (dictionary_cost : ℕ)
  (dinosaur_book_cost : ℕ)
  (saved_amount : ℕ)
  (additional_amount_needed : ℕ)
  (h1 : dictionary_cost = 11)
  (h2 : dinosaur_book_cost = 19)
  (h3 : saved_amount = 8)
  (h4 : additional_amount_needed = 29) :
  saved_amount + additional_amount_needed - (dictionary_cost + dinosaur_book_cost) = 7 := by
  sorry

end NUMINAMATH_CALUDE_childrens_cookbook_cost_l1535_153545


namespace NUMINAMATH_CALUDE_average_age_problem_l1535_153597

theorem average_age_problem (a b c : ℕ) : 
  (a + b + c) / 3 = 27 →
  b = 23 →
  (a + c) / 2 = 29 := by
sorry

end NUMINAMATH_CALUDE_average_age_problem_l1535_153597


namespace NUMINAMATH_CALUDE_percentage_difference_l1535_153561

theorem percentage_difference (n : ℝ) (x y : ℝ) 
  (h1 : n = 160) 
  (h2 : x > y) 
  (h3 : (x / 100) * n - (y / 100) * n = 24) : 
  x - y = 15 := by
sorry

end NUMINAMATH_CALUDE_percentage_difference_l1535_153561


namespace NUMINAMATH_CALUDE_determinant_theorem_l1535_153559

theorem determinant_theorem (a b c d : ℝ) : 
  a * d - b * c = -3 → 
  (a + 2*b) * d - (2*b - d) * (3*c) = -3 - 5*b*c + 2*b*d + 3*c*d := by
sorry

end NUMINAMATH_CALUDE_determinant_theorem_l1535_153559


namespace NUMINAMATH_CALUDE_median_squares_ratio_l1535_153591

/-- Given a triangle with sides a, b, c and corresponding medians ma, mb, mc,
    the ratio of the sum of squares of medians to the sum of squares of sides is 3/4 -/
theorem median_squares_ratio (a b c ma mb mc : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hma : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (hmb : mb^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (hmc : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  (ma^2 + mb^2 + mc^2) / (a^2 + b^2 + c^2) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_median_squares_ratio_l1535_153591


namespace NUMINAMATH_CALUDE_number_of_pens_in_first_set_l1535_153573

-- Define the cost of items
def pencil_cost : ℚ := 0.1
def pen_cost : ℚ := 0.32

-- Define the equations
def equation1 (num_pens : ℕ) : Prop :=
  4 * pencil_cost + num_pens * pen_cost = 2

def equation2 : Prop :=
  3 * pencil_cost + 4 * pen_cost = 1.58

-- Theorem statement
theorem number_of_pens_in_first_set :
  ∃ (num_pens : ℕ), equation1 num_pens ∧ equation2 ∧ num_pens = 5 :=
sorry

end NUMINAMATH_CALUDE_number_of_pens_in_first_set_l1535_153573


namespace NUMINAMATH_CALUDE_rectangular_cube_length_l1535_153516

/-- The length of a rectangular cube with width 2 inches and height 0.5 inches, 
    whose surface area is equal to that of a 2-inch cube, is 4.6 inches. -/
theorem rectangular_cube_length : 
  ∀ (L : ℝ), 
    (5 * L + 1 = 6 * 2^2) → 
    L = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_cube_length_l1535_153516


namespace NUMINAMATH_CALUDE_min_value_w_z_cubes_l1535_153593

/-- Given complex numbers w and z satisfying |w + z| = 1 and |w² + z²| = 14,
    the smallest possible value of |w³ + z³| is 41/2. -/
theorem min_value_w_z_cubes (w z : ℂ) 
    (h1 : Complex.abs (w + z) = 1)
    (h2 : Complex.abs (w^2 + z^2) = 14) :
    ∃ (m : ℝ), m = 41/2 ∧ ∀ (x : ℝ), x ≥ m → Complex.abs (w^3 + z^3) ≤ x :=
by sorry

end NUMINAMATH_CALUDE_min_value_w_z_cubes_l1535_153593


namespace NUMINAMATH_CALUDE_sticker_problem_l1535_153541

theorem sticker_problem (bob tom dan : ℕ) 
  (h1 : dan = 2 * tom) 
  (h2 : tom = 3 * bob) 
  (h3 : dan = 72) : 
  bob = 12 := by
  sorry

end NUMINAMATH_CALUDE_sticker_problem_l1535_153541


namespace NUMINAMATH_CALUDE_b_range_l1535_153528

noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1
def g (x : ℝ) : ℝ := -x^2 + 4*x - 3

theorem b_range (a b : ℝ) (h : f a = g b) : 
  2 - Real.sqrt 2 < b ∧ b < 2 + Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_b_range_l1535_153528


namespace NUMINAMATH_CALUDE_average_problem_l1535_153522

theorem average_problem (x : ℝ) (h : (47 + x) / 2 = 53) : 
  x = 59 ∧ |x - 47| = 12 ∧ x + 47 = 106 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l1535_153522


namespace NUMINAMATH_CALUDE_inverse_variation_cube_and_sqrt_l1535_153509

/-- Given that x^3 varies inversely with √x, prove that y = 1/16384 when x = 64, 
    given that y = 16 when x = 4 -/
theorem inverse_variation_cube_and_sqrt (y : ℝ → ℝ) :
  (∀ x : ℝ, x > 0 → ∃ k : ℝ, y x * (x^3 * x.sqrt) = k) →
  y 4 = 16 →
  y 64 = 1 / 16384 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_and_sqrt_l1535_153509


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1535_153571

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x^2 - 1 = 0 → x^3 - x = 0) ∧ 
  (∃ x : ℝ, x^3 - x = 0 ∧ x^2 - 1 ≠ 0) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1535_153571


namespace NUMINAMATH_CALUDE_only_prop2_and_prop3_true_l1535_153572

-- Define the propositions
def proposition1 : Prop :=
  (∀ x : ℝ, (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)) →
  (∀ x : ℝ, (x^2 - 3*x + 2 = 0) → (x = 1 ∨ x = 2))

def proposition2 : Prop :=
  ¬(∃ x : ℝ, x^2 - 2*x + 1 < 0)

def proposition3 (m : ℝ) : Prop :=
  (m = 1/2) →
  ((m + 2) * (m - 2) + 3 * m * (m + 2) = 0)

def proposition4 (m n : ℝ) : Prop :=
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ x₁^2 - m*x₁ + n = 0 ∧ x₂^2 - m*x₂ + n = 0) →
  (m > 0 ∧ n > 0)

-- Theorem stating that only propositions 2 and 3 are true
theorem only_prop2_and_prop3_true :
  ¬proposition1 ∧ proposition2 ∧ (∃ m : ℝ, proposition3 m) ∧ ¬(∀ m n : ℝ, proposition4 m n) :=
sorry

end NUMINAMATH_CALUDE_only_prop2_and_prop3_true_l1535_153572


namespace NUMINAMATH_CALUDE_largest_frog_weight_l1535_153562

theorem largest_frog_weight (S L : ℝ) 
  (h1 : L = 10 * S) 
  (h2 : L = S + 108) : 
  L = 120 := by
sorry

end NUMINAMATH_CALUDE_largest_frog_weight_l1535_153562


namespace NUMINAMATH_CALUDE_bronze_ball_balance_l1535_153538

theorem bronze_ball_balance (a : Fin 10 → ℝ) : 
  ∃ (S : Finset (Fin 10)), 
    (S.sum (λ i => |a (i + 1) - a i|)) = 
    ((Finset.univ \ S).sum (λ i => |a (i + 1) - a i|)) := by
  sorry


end NUMINAMATH_CALUDE_bronze_ball_balance_l1535_153538


namespace NUMINAMATH_CALUDE_oil_demand_scientific_notation_l1535_153530

theorem oil_demand_scientific_notation :
  (735000000 : ℝ) = 7.35 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_oil_demand_scientific_notation_l1535_153530


namespace NUMINAMATH_CALUDE_downstream_distance_l1535_153555

/-- The distance swum downstream by a woman given certain conditions -/
theorem downstream_distance (t : ℝ) (d_up : ℝ) (v_still : ℝ) : 
  t > 0 ∧ d_up > 0 ∧ v_still > 0 →
  t = 6 ∧ d_up = 6 ∧ v_still = 5 →
  ∃ d_down : ℝ, d_down = 54 ∧ 
    d_down / (v_still + (d_up / t - v_still)) = t ∧
    d_up / (v_still - (d_up / t - v_still)) = t :=
by sorry


end NUMINAMATH_CALUDE_downstream_distance_l1535_153555


namespace NUMINAMATH_CALUDE_kureishi_ratio_l1535_153508

/-- Represents the number of workers in Palabras bookstore who have read certain books -/
structure BookReadership where
  total : ℕ
  saramago : ℕ
  kureishi : ℕ
  both : ℕ
  neither : ℕ

/-- The conditions of the Palabras bookstore problem -/
def palabras_conditions (r : BookReadership) : Prop :=
  r.total = 150 ∧
  r.saramago = r.total / 2 ∧
  r.both = 12 ∧
  r.neither = (r.saramago - r.both) - 1 ∧
  r.total = r.saramago + r.kureishi - r.both + r.neither

/-- The theorem stating the ratio of Kureishi readers to total workers -/
theorem kureishi_ratio (r : BookReadership) 
  (h : palabras_conditions r) : r.kureishi * 6 = r.total := by
  sorry

end NUMINAMATH_CALUDE_kureishi_ratio_l1535_153508


namespace NUMINAMATH_CALUDE_M_intersect_N_l1535_153567

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem M_intersect_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_l1535_153567


namespace NUMINAMATH_CALUDE_polynomial_coefficients_sum_l1535_153577

theorem polynomial_coefficients_sum (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x : ℝ, (2*x - 1)^5 + (x + 2)^4 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  |a₀| + |a₂| + |a₄| = 110 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficients_sum_l1535_153577


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1535_153506

theorem absolute_value_simplification : |-4^2 + 5 - 2| = 13 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1535_153506


namespace NUMINAMATH_CALUDE_newspaper_weeks_l1535_153544

/-- The cost of a weekday newspaper --/
def weekday_cost : ℚ := 1/2

/-- The cost of a Sunday newspaper --/
def sunday_cost : ℚ := 2

/-- The number of weekday newspapers bought per week --/
def weekday_papers : ℕ := 3

/-- The total amount spent on newspapers --/
def total_spent : ℚ := 28

/-- The number of weeks Hillary buys newspapers --/
def weeks_buying : ℕ := 8

theorem newspaper_weeks : 
  (weekday_papers * weekday_cost + sunday_cost) * weeks_buying = total_spent := by
  sorry

end NUMINAMATH_CALUDE_newspaper_weeks_l1535_153544


namespace NUMINAMATH_CALUDE_quotient_of_A_and_B_l1535_153546

/-- Given A and B as defined, prove that A / B = 31 -/
theorem quotient_of_A_and_B : 
  let A := 8 * 10 + 13 * 1
  let B := 30 - 9 - 9 - 9
  A / B = 31 := by
sorry

end NUMINAMATH_CALUDE_quotient_of_A_and_B_l1535_153546


namespace NUMINAMATH_CALUDE_probability_theorem_l1535_153568

def num_male_students : ℕ := 5
def num_female_students : ℕ := 2
def num_representatives : ℕ := 3

def probability_B_or_C_given_A (total_students : ℕ) (remaining_selections : ℕ) : ℚ :=
  let favorable_outcomes := (remaining_selections * (total_students - 3)) + 1
  let total_outcomes := Nat.choose (total_students - 1) remaining_selections
  favorable_outcomes / total_outcomes

theorem probability_theorem :
  probability_B_or_C_given_A (num_male_students + num_female_students) (num_representatives - 1) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_probability_theorem_l1535_153568


namespace NUMINAMATH_CALUDE_impossibleGrid_l1535_153501

/-- Represents a 6x6 grid filled with numbers from 1 to 6 -/
def Grid := Fin 6 → Fin 6 → Fin 6

/-- The sum of numbers in a 2x2 subgrid starting at (i, j) -/
def subgridSum (g : Grid) (i j : Fin 5) : ℕ :=
  g i j + g i (j + 1) + g (i + 1) j + g (i + 1) (j + 1)

/-- A predicate that checks if all 2x2 subgrids have different sums -/
def allSubgridSumsDifferent (g : Grid) : Prop :=
  ∀ i j k l : Fin 5, (i, j) ≠ (k, l) → subgridSum g i j ≠ subgridSum g k l

theorem impossibleGrid : ¬ ∃ g : Grid, allSubgridSumsDifferent g := by
  sorry

end NUMINAMATH_CALUDE_impossibleGrid_l1535_153501


namespace NUMINAMATH_CALUDE_surrounded_pentagon_n_gons_l1535_153596

/-- The number of sides of the central polygon -/
def m : ℕ := 5

/-- The number of surrounding polygons -/
def num_surrounding : ℕ := 5

/-- The interior angle of a regular polygon with k sides -/
def interior_angle (k : ℕ) : ℚ :=
  (k - 2 : ℚ) * 180 / k

/-- The exterior angle of a regular polygon with k sides -/
def exterior_angle (k : ℕ) : ℚ :=
  180 - interior_angle k

/-- Theorem stating that for a regular pentagon surrounded by 5 regular n-gons
    with no overlap and no gaps, n must equal 5 -/
theorem surrounded_pentagon_n_gons :
  ∃ (n : ℕ), n > 2 ∧ 
  exterior_angle m = 360 / n ∧
  num_surrounding * (360 / n) = 360 := by
  sorry

end NUMINAMATH_CALUDE_surrounded_pentagon_n_gons_l1535_153596


namespace NUMINAMATH_CALUDE_digit_puzzle_solution_l1535_153554

theorem digit_puzzle_solution (c o u n t s : ℕ) 
  (h1 : c + o = u)
  (h2 : u + n = t + 1)
  (h3 : t + c = s)
  (h4 : o + n + s = 15)
  (h5 : c ≠ 0 ∧ o ≠ 0 ∧ u ≠ 0 ∧ n ≠ 0 ∧ t ≠ 0 ∧ s ≠ 0)
  (h6 : c < 10 ∧ o < 10 ∧ u < 10 ∧ n < 10 ∧ t < 10 ∧ s < 10) :
  t = 7 := by sorry

end NUMINAMATH_CALUDE_digit_puzzle_solution_l1535_153554


namespace NUMINAMATH_CALUDE_twentyFourthDigitOfSum_l1535_153564

-- Define the decimal representation of a rational number
def decimalRepresentation (q : ℚ) : ℕ → ℕ := sorry

-- Define the sum of two decimal representations
def sumDecimalRepresentations (f g : ℕ → ℕ) : ℕ → ℕ := sorry

-- The main theorem
theorem twentyFourthDigitOfSum :
  let f := decimalRepresentation (1/9 : ℚ)
  let g := decimalRepresentation (1/4 : ℚ)
  let sum := sumDecimalRepresentations f g
  sum 24 = 1 := by sorry

end NUMINAMATH_CALUDE_twentyFourthDigitOfSum_l1535_153564


namespace NUMINAMATH_CALUDE_raise_doubles_earnings_l1535_153513

/-- Calculates the new weekly earnings after a percentage raise -/
def new_earnings (initial_earnings : ℕ) (percentage_raise : ℕ) : ℕ :=
  initial_earnings + initial_earnings * percentage_raise / 100

/-- Proves that a 100% raise on $40 results in $80 weekly earnings -/
theorem raise_doubles_earnings : new_earnings 40 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_raise_doubles_earnings_l1535_153513


namespace NUMINAMATH_CALUDE_compounded_ratio_is_two_to_one_l1535_153565

/-- The compounded ratio of three given ratios -/
def compounded_ratio (r1 r2 r3 : Rat × Rat) : Rat × Rat :=
  let (a1, b1) := r1
  let (a2, b2) := r2
  let (a3, b3) := r3
  (a1 * a2 * a3, b1 * b2 * b3)

/-- The given ratios -/
def ratio1 : Rat × Rat := (2, 3)
def ratio2 : Rat × Rat := (6, 11)
def ratio3 : Rat × Rat := (11, 2)

/-- The theorem stating that the compounded ratio of the given ratios is 2:1 -/
theorem compounded_ratio_is_two_to_one :
  compounded_ratio ratio1 ratio2 ratio3 = (2, 1) := by
  sorry

end NUMINAMATH_CALUDE_compounded_ratio_is_two_to_one_l1535_153565


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l1535_153536

theorem square_difference_divided_by_nine : (110^2 - 95^2) / 9 = 3075 / 9 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l1535_153536


namespace NUMINAMATH_CALUDE_min_shift_for_trig_transformation_l1535_153526

open Real

/-- The minimum positive shift required to transform sin(2x) + √3cos(2x) into 2sin(2x) -/
theorem min_shift_for_trig_transformation : ∃ (m : ℝ), m > 0 ∧
  (∀ (x : ℝ), sin (2*x) + Real.sqrt 3 * cos (2*x) = 2 * sin (2*(x + m))) ∧
  (∀ (m' : ℝ), m' > 0 → 
    (∀ (x : ℝ), sin (2*x) + Real.sqrt 3 * cos (2*x) = 2 * sin (2*(x + m'))) → 
    m ≤ m') ∧
  m = π / 6 := by
sorry

end NUMINAMATH_CALUDE_min_shift_for_trig_transformation_l1535_153526


namespace NUMINAMATH_CALUDE_alice_bob_sum_l1535_153588

theorem alice_bob_sum : ∀ (a b : ℕ),
  1 ≤ a ∧ a ≤ 50 ∧                     -- Alice's number is between 1 and 50
  1 ≤ b ∧ b ≤ 50 ∧                     -- Bob's number is between 1 and 50
  a ≠ b ∧                              -- Numbers are drawn without replacement
  a ≠ 1 ∧ a ≠ 50 ∧                     -- Alice can't tell who has the larger number
  b > a ∧                              -- Bob knows he has the larger number
  ∃ (d : ℕ), d > 1 ∧ d < b ∧ d ∣ b ∧   -- Bob's number is composite
  ∃ (k : ℕ), 50 * b + a = k * k →      -- 50 * Bob's number + Alice's number is a perfect square
  a + b = 29 := by
sorry

end NUMINAMATH_CALUDE_alice_bob_sum_l1535_153588


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l1535_153529

/-- The trajectory of the midpoint between a fixed point and a point on a parabola -/
theorem midpoint_trajectory (B : ℝ × ℝ) :
  (B.2^2 = 2 * B.1) →  -- B is on the parabola y^2 = 2x
  let A : ℝ × ℝ := (2, 4)  -- Fixed point A(2, 4)
  let P : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)  -- P is the midpoint of AB
  (P.2 - 2)^2 = P.1 - 1 :=
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l1535_153529


namespace NUMINAMATH_CALUDE_no_power_of_three_and_five_l1535_153592

def v : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 8 * v (n + 1) - v n

theorem no_power_of_three_and_five :
  ∀ n : ℕ, ∀ α β : ℕ+, v n ≠ (3 : ℤ)^(α : ℕ) * (5 : ℤ)^(β : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_no_power_of_three_and_five_l1535_153592


namespace NUMINAMATH_CALUDE_square_sum_value_l1535_153510

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l1535_153510


namespace NUMINAMATH_CALUDE_cape_may_has_24_sightings_l1535_153512

/-- The number of shark sightings in Cape May -/
def cape_may_sightings : ℕ := 24

/-- The number of shark sightings in Daytona Beach -/
def daytona_beach_sightings : ℕ := 16

/-- The total number of shark sightings in Cape May and Daytona Beach -/
def total_sightings : ℕ := 40

/-- Theorem stating that Cape May has 24 shark sightings given the conditions -/
theorem cape_may_has_24_sightings :
  cape_may_sightings = 24 ∧
  cape_may_sightings + daytona_beach_sightings = total_sightings ∧
  cape_may_sightings = 2 * daytona_beach_sightings - 8 :=
by sorry

end NUMINAMATH_CALUDE_cape_may_has_24_sightings_l1535_153512


namespace NUMINAMATH_CALUDE_cubic_expansion_coefficients_l1535_153598

theorem cubic_expansion_coefficients (a b : ℤ) : 
  (3 * b + 3 * a^2 = 99) ∧ (3 * a * b^2 = 162) → (a = 6 ∧ b = -3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_expansion_coefficients_l1535_153598


namespace NUMINAMATH_CALUDE_autumn_outing_problem_l1535_153500

/-- Autumn Outing Problem -/
theorem autumn_outing_problem 
  (bus_seats : ℕ) 
  (public_bus_seats : ℕ) 
  (bus_count : ℕ) 
  (teachers_per_bus : ℕ) 
  (extra_seats_buses : ℕ) 
  (extra_teachers_public : ℕ) 
  (h1 : bus_seats = 39)
  (h2 : public_bus_seats = 27)
  (h3 : bus_count + 2 = public_bus_count)
  (h4 : teachers_per_bus = 2)
  (h5 : extra_seats_buses = 3)
  (h6 : extra_teachers_public = 3)
  (h7 : bus_seats * bus_count = teachers_per_bus * bus_count + students + extra_seats_buses)
  (h8 : public_bus_seats * public_bus_count = teachers + students)
  (h9 : teachers = public_bus_count + extra_teachers_public) :
  teachers = 18 ∧ students = 330 := by
  sorry


end NUMINAMATH_CALUDE_autumn_outing_problem_l1535_153500


namespace NUMINAMATH_CALUDE_complex_simplification_l1535_153515

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification : 3 * (4 - 2*i) + 2*i * (3 + 2*i) = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l1535_153515


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1535_153521

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1535_153521


namespace NUMINAMATH_CALUDE_distance_between_stations_distance_is_65km_l1535_153578

/-- The distance between two stations given train travel information -/
theorem distance_between_stations : ℝ :=
let train_p_speed : ℝ := 20
let train_q_speed : ℝ := 25
let train_p_time : ℝ := 2
let train_q_time : ℝ := 1
let distance_p : ℝ := train_p_speed * train_p_time
let distance_q : ℝ := train_q_speed * train_q_time
distance_p + distance_q

/-- Proof that the distance between the stations is 65 km -/
theorem distance_is_65km : distance_between_stations = 65 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_stations_distance_is_65km_l1535_153578


namespace NUMINAMATH_CALUDE_first_reduction_percentage_l1535_153553

theorem first_reduction_percentage (x : ℝ) : 
  (1 - x / 100) * (1 - 0.1) = 0.81 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_reduction_percentage_l1535_153553


namespace NUMINAMATH_CALUDE_club_members_proof_l1535_153584

theorem club_members_proof (total : Nat) (left_handed : Nat) (rock_fans : Nat) (right_handed_non_rock : Nat) 
  (h1 : total = 30)
  (h2 : left_handed = 12)
  (h3 : rock_fans = 20)
  (h4 : right_handed_non_rock = 3)
  (h5 : ∀ x : Nat, x ≤ total → x = (left_handed + (total - left_handed)))
  : ∃ x : Nat, x = 5 ∧ 
    x + (left_handed - x) + (rock_fans - x) + right_handed_non_rock = total := by
  sorry


end NUMINAMATH_CALUDE_club_members_proof_l1535_153584


namespace NUMINAMATH_CALUDE_clips_sold_and_average_earning_l1535_153556

/-- Calculates the total number of clips sold and average earning per clip -/
theorem clips_sold_and_average_earning 
  (x : ℝ) -- number of clips sold in April
  (y : ℝ) -- number of clips sold in May
  (z : ℝ) -- number of clips sold in June
  (W : ℝ) -- total earnings
  (h1 : y = x / 2) -- May sales condition
  (h2 : z = y + 0.25 * y) -- June sales condition
  : (x + y + z = 2.125 * x) ∧ (W / (x + y + z) = W / (2.125 * x)) := by
  sorry

end NUMINAMATH_CALUDE_clips_sold_and_average_earning_l1535_153556


namespace NUMINAMATH_CALUDE_perpendicular_sequence_limit_l1535_153590

/-- An equilateral triangle ABC with a sequence of points Pₙ on AB defined by perpendicular constructions --/
structure PerpendicularSequence where
  /-- The side length of the equilateral triangle --/
  a : ℝ
  /-- The sequence of distances BPₙ --/
  bp : ℕ → ℝ
  /-- The initial point P₁ is on AB --/
  h_initial : 0 ≤ bp 1 ∧ bp 1 ≤ a
  /-- The recurrence relation for the sequence --/
  h_recurrence : ∀ n, bp (n + 1) = 3/4 * a - 1/8 * bp n

/-- The limit of the perpendicular sequence converges to 2/3 of the side length --/
theorem perpendicular_sequence_limit (ps : PerpendicularSequence) :
  ∃ L, L = 2/3 * ps.a ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, |ps.bp n - L| < ε :=
sorry

end NUMINAMATH_CALUDE_perpendicular_sequence_limit_l1535_153590


namespace NUMINAMATH_CALUDE_min_of_quadratic_l1535_153503

/-- The quadratic function f(x) = x^2 + px + 2q -/
def f (p q : ℝ) (x : ℝ) : ℝ := x^2 + p*x + 2*q

/-- Theorem stating that the minimum of f occurs at x = -p/2 -/
theorem min_of_quadratic (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  ∀ x : ℝ, f p q (-p/2) ≤ f p q x :=
sorry

end NUMINAMATH_CALUDE_min_of_quadratic_l1535_153503


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l1535_153535

/-- Proves that adding 1.5 liters of 90% alcohol solution to 6 liters of 40% alcohol solution 
    results in a final mixture that is 50% alcohol. -/
theorem alcohol_mixture_proof :
  let initial_volume : ℝ := 6
  let initial_concentration : ℝ := 0.4
  let added_concentration : ℝ := 0.9
  let added_volume : ℝ := 1.5
  let final_concentration : ℝ := 0.5
  let final_volume : ℝ := initial_volume + added_volume
  let initial_alcohol : ℝ := initial_volume * initial_concentration
  let added_alcohol : ℝ := added_volume * added_concentration
  let total_alcohol : ℝ := initial_alcohol + added_alcohol
  total_alcohol = final_volume * final_concentration :=
by
  sorry


end NUMINAMATH_CALUDE_alcohol_mixture_proof_l1535_153535


namespace NUMINAMATH_CALUDE_function_range_l1535_153519

/-- The range of the function f(x) = (e^(3x) - 2) / (e^(3x) + 2) is (-1, 1) -/
theorem function_range (x : ℝ) : 
  -1 < (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2) ∧ 
  (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l1535_153519


namespace NUMINAMATH_CALUDE_diesel_cost_per_gallon_l1535_153531

/-- The cost of diesel fuel per gallon, given weekly spending and bi-weekly usage -/
theorem diesel_cost_per_gallon 
  (weekly_spending : ℝ) 
  (biweekly_usage : ℝ) 
  (h1 : weekly_spending = 36) 
  (h2 : biweekly_usage = 24) : 
  weekly_spending * 2 / biweekly_usage = 3 := by
sorry

end NUMINAMATH_CALUDE_diesel_cost_per_gallon_l1535_153531


namespace NUMINAMATH_CALUDE_basketball_players_l1535_153563

theorem basketball_players (C B_and_C B_or_C : ℕ) 
  (h1 : C = 8)
  (h2 : B_and_C = 5)
  (h3 : B_or_C = 10)
  : ∃ B : ℕ, B = 7 ∧ B_or_C = B + C - B_and_C :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_players_l1535_153563


namespace NUMINAMATH_CALUDE_existence_of_function_with_properties_l1535_153552

theorem existence_of_function_with_properties : ∃ f : ℝ → ℝ, 
  (∀ x : ℝ, f (1 + x) = f (1 - x)) ∧ 
  (∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f y ≤ f x) ∧ 
  (∃ z : ℝ, f z < f 0) ∧
  (let g : ℝ → ℝ := fun x ↦ (x - 1)^2;
   (∀ x : ℝ, g (1 + x) = g (1 - x)) ∧ 
   (∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → g y ≤ g x) ∧ 
   (∃ z : ℝ, g z < g 0)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_function_with_properties_l1535_153552


namespace NUMINAMATH_CALUDE_movie_preference_related_to_gender_expected_value_correct_l1535_153581

/-- The total number of survey participants -/
def total_participants : ℕ := 200

/-- The number of male viewers who prefer foreign movies -/
def male_foreign : ℕ := 40

/-- The total number of male viewers -/
def total_male : ℕ := 100

/-- The number of female viewers who prefer domestic movies -/
def female_domestic : ℕ := 80

/-- The critical value for χ² at α = 0.005 -/
def critical_value : ℝ := 7.879

/-- Calculate the χ² value for the contingency table -/
def calculate_chi_square : ℝ := sorry

/-- The probability of selecting a female viewer who chose domestic movies -/
def p_female_domestic : ℚ := 4/7

/-- The number of random selections -/
def num_selections : ℕ := 3

/-- The expected value of X, where X is the number of female viewers in 3 random selections -/
def expected_value : ℚ := 12/7

theorem movie_preference_related_to_gender : 
  calculate_chi_square > critical_value := sorry

theorem expected_value_correct :
  expected_value = num_selections * p_female_domestic := sorry

end NUMINAMATH_CALUDE_movie_preference_related_to_gender_expected_value_correct_l1535_153581


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1535_153505

theorem complex_equation_solution (a : ℝ) (h : (1 + a * Complex.I) * Complex.I = 3 + Complex.I) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1535_153505


namespace NUMINAMATH_CALUDE_finite_subsequence_exists_infinite_subsequence_not_exists_l1535_153550

/-- The sequence 1, 1/2, 1/3, ... -/
def harmonic_sequence : ℕ → ℚ 
  | n => 1 / n

/-- A subsequence of the harmonic sequence -/
structure Subsequence :=
  (indices : ℕ → ℕ)
  (strictly_increasing : ∀ n, indices n < indices (n + 1))

/-- The property that each term from the third is the difference of the two preceding terms -/
def has_difference_property (s : Subsequence) : Prop :=
  ∀ k ≥ 3, harmonic_sequence (s.indices k) = 
    harmonic_sequence (s.indices (k - 2)) - harmonic_sequence (s.indices (k - 1))

theorem finite_subsequence_exists : ∃ s : Subsequence, 
  (∀ n, n ≤ 100 → s.indices n ≤ 100) ∧ has_difference_property s :=
sorry

theorem infinite_subsequence_not_exists : ¬∃ s : Subsequence, has_difference_property s :=
sorry

end NUMINAMATH_CALUDE_finite_subsequence_exists_infinite_subsequence_not_exists_l1535_153550


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1535_153558

theorem quadratic_root_problem (a : ℝ) :
  ((-1 : ℝ)^2 - 2*(-1) + a = 0) → 
  (∃ x : ℝ, x^2 - 2*x + a = 0 ∧ x ≠ -1 ∧ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1535_153558


namespace NUMINAMATH_CALUDE_fourth_term_is_one_tenth_l1535_153537

theorem fourth_term_is_one_tenth (a : ℕ → ℚ) :
  (∀ n : ℕ, a n = 2 / (n^2 + n)) →
  a 4 = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_is_one_tenth_l1535_153537


namespace NUMINAMATH_CALUDE_inequality_proof_l1535_153595

theorem inequality_proof (x : ℝ) : x > 4 → 3 * x + 5 < 5 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1535_153595


namespace NUMINAMATH_CALUDE_least_k_cube_divisible_by_120_l1535_153547

theorem least_k_cube_divisible_by_120 : 
  ∃ k : ℕ+, k.val = 30 ∧ 
  (∀ m : ℕ+, m.val < k.val → ¬(120 ∣ m.val^3)) ∧ 
  (120 ∣ k.val^3) := by
  sorry

end NUMINAMATH_CALUDE_least_k_cube_divisible_by_120_l1535_153547


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_l1535_153551

theorem imaginary_unit_sum (i : ℂ) : i * i = -1 → (i⁻¹ : ℂ) + i^2015 = -2*i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_l1535_153551


namespace NUMINAMATH_CALUDE_symmetry_about_x_axis_periodicity_symmetry_about_origin_l1535_153574

-- Define a real-valued function on reals
variable (f : ℝ → ℝ)

-- Statement 1
theorem symmetry_about_x_axis (x : ℝ) : 
  f (-1 - x) = f (-(x - 1)) := by sorry

-- Statement 2
theorem periodicity (x : ℝ) : 
  f (1 + x) = f (x - 1) → f (x + 2) = f x := by sorry

-- Statement 3
theorem symmetry_about_origin (x : ℝ) : 
  f (1 - x) = -f (x - 1) → f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_symmetry_about_x_axis_periodicity_symmetry_about_origin_l1535_153574


namespace NUMINAMATH_CALUDE_lice_check_time_l1535_153548

/-- Calculates the time required for lice checks in an elementary school -/
theorem lice_check_time (kindergarteners : ℕ) (first_graders : ℕ) (second_graders : ℕ) (third_graders : ℕ) 
  (time_per_check : ℕ) (h1 : kindergarteners = 26) (h2 : first_graders = 19) (h3 : second_graders = 20) 
  (h4 : third_graders = 25) (h5 : time_per_check = 2) : 
  (kindergarteners + first_graders + second_graders + third_graders) * time_per_check / 60 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lice_check_time_l1535_153548


namespace NUMINAMATH_CALUDE_f_is_smallest_f_is_minimal_l1535_153542

/-- 
For a given integer n ≥ 4, f(n) is the smallest integer such that 
any f(n)-element subset of {m, m+1, ..., m+n-1} contains at least 
3 pairwise coprime elements, where m is any positive integer.
-/
def f (n : ℕ) : ℕ :=
  (n + 1) / 2 + (n + 1) / 3 - (n + 1) / 6 + 1

/-- 
Theorem: For integers n ≥ 4, f(n) is the smallest integer such that 
any f(n)-element subset of {m, m+1, ..., m+n-1} contains at least 
3 pairwise coprime elements, where m is any positive integer.
-/
theorem f_is_smallest (n : ℕ) (h : n ≥ 4) : 
  ∀ (m : ℕ+), ∀ (S : Finset ℕ), 
    S.card = f n → 
    (∀ x ∈ S, ∃ k, x = m + k ∧ k < n) → 
    ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
      Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c :=
by
  sorry

/-- 
Corollary: There is no smaller integer than f(n) that satisfies 
the conditions for all n ≥ 4.
-/
theorem f_is_minimal (n : ℕ) (h : n ≥ 4) :
  ∀ g : ℕ → ℕ, (∀ k ≥ 4, g k < f k) → 
    ∃ (m : ℕ+) (S : Finset ℕ), 
      S.card = g n ∧
      (∀ x ∈ S, ∃ k, x = m + k ∧ k < n) ∧
      ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → 
        ¬(Nat.Coprime a b ∧ Nat.Coprime b c ∧ Nat.Coprime a c) :=
by
  sorry

end NUMINAMATH_CALUDE_f_is_smallest_f_is_minimal_l1535_153542


namespace NUMINAMATH_CALUDE_doubled_average_l1535_153504

theorem doubled_average (n : ℕ) (original_avg : ℚ) (h1 : n = 12) (h2 : original_avg = 50) :
  let total_marks := n * original_avg
  let doubled_marks := 2 * total_marks
  let new_avg := doubled_marks / n
  new_avg = 100 := by sorry

end NUMINAMATH_CALUDE_doubled_average_l1535_153504


namespace NUMINAMATH_CALUDE_absolute_value_of_five_minus_pi_plus_two_l1535_153539

theorem absolute_value_of_five_minus_pi_plus_two : |5 - Real.pi + 2| = 7 - Real.pi := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_five_minus_pi_plus_two_l1535_153539


namespace NUMINAMATH_CALUDE_aunt_age_l1535_153524

/-- Proves that given Cori is 3 years old today, and in 5 years she will be one-third the age of her aunt, her aunt's current age is 19 years. -/
theorem aunt_age (cori_age : ℕ) (aunt_age : ℕ) : 
  cori_age = 3 → 
  (cori_age + 5 : ℕ) = (aunt_age + 5) / 3 → 
  aunt_age = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_aunt_age_l1535_153524


namespace NUMINAMATH_CALUDE_increasing_function_a_bound_l1535_153570

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 4*a*x + 1

-- State the theorem
theorem increasing_function_a_bound (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 1 3 → y ∈ Set.Icc 1 3 → x < y → f a x < f a y) →
  a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_a_bound_l1535_153570


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1535_153579

theorem inequality_equivalence (a : ℝ) : (a + 1 < 0) ↔ (a < -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1535_153579


namespace NUMINAMATH_CALUDE_unique_pizza_combinations_l1535_153557

def num_toppings : ℕ := 8
def toppings_per_pizza : ℕ := 3

theorem unique_pizza_combinations :
  Nat.choose num_toppings toppings_per_pizza = 56 := by
  sorry

end NUMINAMATH_CALUDE_unique_pizza_combinations_l1535_153557


namespace NUMINAMATH_CALUDE_coefficient_implies_a_value_l1535_153560

theorem coefficient_implies_a_value (a : ℝ) : 
  (5 / 2) * a^3 = 20 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_implies_a_value_l1535_153560


namespace NUMINAMATH_CALUDE_quadratic_function_unique_a_l1535_153576

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_function_unique_a (f : QuadraticFunction) :
  f.eval 1 = 5 → f.eval 0 = 2 → f.a = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_unique_a_l1535_153576


namespace NUMINAMATH_CALUDE_f_min_value_h_unique_zero_l1535_153511

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 / Real.exp x
noncomputable def h (x : ℝ) : ℝ := g x - f (-1) x

-- Theorem for part 1
theorem f_min_value (a : ℝ) (h : a > 0) :
  ∃ (x : ℝ), x > 0 ∧ IsLocalMin (f a) x ∧ f a x = a - a * Real.log a :=
sorry

-- Theorem for part 2
theorem h_unique_zero :
  ∃! (x : ℝ), x ∈ Set.Ioo 0 1 ∧ h x = 0 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_h_unique_zero_l1535_153511


namespace NUMINAMATH_CALUDE_least_sum_m_n_l1535_153527

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (m.val.gcd (330 : ℕ) = 1) ∧ 
  (n.val.gcd (330 : ℕ) = 1) ∧
  ((m + n).val.gcd (330 : ℕ) = 1) ∧
  (∃ (k : ℕ), m.val^m.val = k * n.val^n.val) ∧
  (∀ (l : ℕ+), m.val ≠ l.val * n.val) ∧
  (∀ (p q : ℕ+), 
    (p.val.gcd (330 : ℕ) = 1) ∧ 
    (q.val.gcd (330 : ℕ) = 1) ∧
    ((p + q).val.gcd (330 : ℕ) = 1) ∧
    (∃ (r : ℕ), p.val^p.val = r * q.val^q.val) ∧
    (∀ (s : ℕ+), p.val ≠ s.val * q.val) →
    (m + n).val ≤ (p + q).val) ∧
  (m + n).val = 154 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_m_n_l1535_153527


namespace NUMINAMATH_CALUDE_total_tape_area_l1535_153587

/-- Calculate the total area of tape used for taping boxes -/
theorem total_tape_area (box1_length box1_width : ℕ) (box2_side : ℕ) (box3_length box3_width : ℕ)
  (box1_count box2_count box3_count : ℕ) (tape_width overlap : ℕ) :
  box1_length = 30 ∧ box1_width = 15 ∧ 
  box2_side = 40 ∧
  box3_length = 50 ∧ box3_width = 20 ∧
  box1_count = 5 ∧ box2_count = 2 ∧ box3_count = 3 ∧
  tape_width = 2 ∧ overlap = 2 →
  (box1_count * (box1_length + overlap + 2 * (box1_width + overlap)) +
   box2_count * (3 * (box2_side + overlap)) +
   box3_count * (box3_length + overlap + 2 * (box3_width + overlap))) * tape_width = 1740 := by
  sorry

end NUMINAMATH_CALUDE_total_tape_area_l1535_153587


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1535_153582

theorem imaginary_part_of_z (z : ℂ) : z * (1 - 2*I) = Complex.abs (3 + 4*I) → Complex.im z = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1535_153582


namespace NUMINAMATH_CALUDE_expression_1_expression_2_expression_3_expression_4_expression_5_l1535_153507

-- Expression 1
theorem expression_1 : 0.11 * 1.8 + 8.2 * 0.11 = 1.1 := by sorry

-- Expression 2
theorem expression_2 : 0.8 * (3.2 - 2.99 / 2.3) = 1.52 := by sorry

-- Expression 3
theorem expression_3 : 3.5 - 3.5 * 0.98 = 0.07 := by sorry

-- Expression 4
theorem expression_4 : 12.5 * 2.5 * 3.2 = 100 := by sorry

-- Expression 5
theorem expression_5 : (8.1 - 5.4) / 3.6 + 85.7 = 86.45 := by sorry

end NUMINAMATH_CALUDE_expression_1_expression_2_expression_3_expression_4_expression_5_l1535_153507


namespace NUMINAMATH_CALUDE_total_dimes_proof_l1535_153575

/-- Calculates the total number of dimes Tom has after receiving more from his dad. -/
def total_dimes (initial_dimes : ℕ) (dimes_from_dad : ℕ) : ℕ :=
  initial_dimes + dimes_from_dad

/-- Proves that the total number of dimes Tom has is the sum of his initial dimes and those given by his dad. -/
theorem total_dimes_proof (initial_dimes : ℕ) (dimes_from_dad : ℕ) :
  total_dimes initial_dimes dimes_from_dad = initial_dimes + dimes_from_dad := by
  sorry

#eval total_dimes 15 33  -- Should output 48

end NUMINAMATH_CALUDE_total_dimes_proof_l1535_153575


namespace NUMINAMATH_CALUDE_train_station_problem_l1535_153583

theorem train_station_problem :
  ∀ (x v : ℕ),
  v > 3 →
  x = (2 * v) / (v - 3) →
  x - 5 > 0 →
  x / v - (x - 5) / 3 = 1 →
  (x = 8 ∧ v = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_train_station_problem_l1535_153583


namespace NUMINAMATH_CALUDE_dalton_savings_proof_l1535_153589

/-- The amount of money Dalton saved from his allowance -/
def dalton_savings : ℕ := sorry

/-- The cost of all items Dalton wants to buy -/
def total_cost : ℕ := 23

/-- The amount Dalton's uncle gave him -/
def uncle_contribution : ℕ := 13

/-- The additional amount Dalton needs -/
def additional_needed : ℕ := 4

theorem dalton_savings_proof :
  dalton_savings = total_cost - uncle_contribution - additional_needed :=
by sorry

end NUMINAMATH_CALUDE_dalton_savings_proof_l1535_153589


namespace NUMINAMATH_CALUDE_no_positive_subtraction_l1535_153518

theorem no_positive_subtraction (x : ℝ) : x > 0 → 24 - x ≠ 34 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_subtraction_l1535_153518


namespace NUMINAMATH_CALUDE_least_distinct_values_in_list_l1535_153566

/-- Given a list of 2520 positive integers with a unique mode occurring exactly 12 times,
    the least number of distinct values in the list is 229. -/
theorem least_distinct_values_in_list :
  ∀ (L : List ℕ+) (mode : ℕ+),
    L.length = 2520 →
    (∃! x, x ∈ L ∧ L.count x = 12) →
    (∃ x, x ∈ L ∧ L.count x = 12) →
    (∀ x, x ∈ L → L.count x ≤ 12) →
    L.toFinset.card ≥ 229 :=
by sorry

end NUMINAMATH_CALUDE_least_distinct_values_in_list_l1535_153566


namespace NUMINAMATH_CALUDE_dark_lord_sword_distribution_l1535_153534

/-- Calculates the weight of swords each orc must carry given the total weight,
    number of squads, and orcs per squad. -/
def weight_per_orc (total_weight : ℕ) (num_squads : ℕ) (orcs_per_squad : ℕ) : ℚ :=
  total_weight / (num_squads * orcs_per_squad)

/-- Proves that given 1200 pounds of swords, 10 squads, and 8 orcs per squad,
    each orc must carry 15 pounds of swords. -/
theorem dark_lord_sword_distribution :
  weight_per_orc 1200 10 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_dark_lord_sword_distribution_l1535_153534


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1535_153525

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (5 * x + 9) = 12 → x = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1535_153525


namespace NUMINAMATH_CALUDE_erased_number_is_202_l1535_153549

-- Define the sequence of consecutive positive integers
def consecutive_sequence (n : ℕ) : List ℕ := List.range n

-- Define the function to calculate the sum of the first n positive integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define the function to calculate the average of the remaining numbers after erasing x
def average_after_erasing (n : ℕ) (x : ℕ) : ℚ :=
  (sum_first_n n - x) / (n - 1 : ℚ)

-- The theorem to prove
theorem erased_number_is_202 (n : ℕ) (x : ℕ) :
  x ∈ consecutive_sequence n →
  average_after_erasing n x = 151 / 3 →
  x = 202 := by
  sorry

end NUMINAMATH_CALUDE_erased_number_is_202_l1535_153549


namespace NUMINAMATH_CALUDE_perfect_square_condition_l1535_153520

theorem perfect_square_condition (a b k : ℝ) : 
  (∃ (c : ℝ), a^2 + 2*(k-3)*a*b + 9*b^2 = c^2) → (k = 0 ∨ k = 6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l1535_153520


namespace NUMINAMATH_CALUDE_repetend_of_five_thirteenths_l1535_153517

theorem repetend_of_five_thirteenths : ∃ (n : ℕ), 
  (5 : ℚ) / 13 = (384615 : ℚ) / (10^6 - 1) + n / (13 * (10^6 - 1)) := by
  sorry

end NUMINAMATH_CALUDE_repetend_of_five_thirteenths_l1535_153517


namespace NUMINAMATH_CALUDE_bicycle_owners_without_scooters_l1535_153502

theorem bicycle_owners_without_scooters (total : ℕ) (bicycle_owners : ℕ) (scooter_owners : ℕ) 
  (h_total : total = 500)
  (h_bicycle : bicycle_owners = 485)
  (h_scooter : scooter_owners = 150)
  (h_subset : bicycle_owners + scooter_owners ≥ total) :
  bicycle_owners - (bicycle_owners + scooter_owners - total) = 350 := by
  sorry

#check bicycle_owners_without_scooters

end NUMINAMATH_CALUDE_bicycle_owners_without_scooters_l1535_153502


namespace NUMINAMATH_CALUDE_range_of_a_l1535_153532

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- State the theorem
theorem range_of_a (h : ∀ x a : ℝ, (¬(p x) → ¬(q x a)) ∧ ∃ x, ¬(p x) ∧ (q x a)) :
  ∀ a : ℝ, a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1535_153532


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l1535_153594

theorem existence_of_special_integers : ∃ (m n : ℤ), 
  (∃ (k₁ : ℤ), n^2 = k₁ * m) ∧
  (∃ (k₂ : ℤ), m^3 = k₂ * n^2) ∧
  (∃ (k₃ : ℤ), n^4 = k₃ * m^3) ∧
  (∃ (k₄ : ℤ), m^5 = k₄ * n^4) ∧
  (∀ (k₅ : ℤ), n^6 ≠ k₅ * m^5) ∧
  m = 32 ∧ n = 16 := by
sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l1535_153594
