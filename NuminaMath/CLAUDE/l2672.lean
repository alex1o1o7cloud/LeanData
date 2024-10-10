import Mathlib

namespace fraction_equation_solution_l2672_267241

theorem fraction_equation_solution (n : ℚ) : 
  (2 / (n + 1) + 3 / (n + 1) + n / (n + 1) = 4) → n = 1/3 := by
sorry

end fraction_equation_solution_l2672_267241


namespace interior_angle_regular_octagon_l2672_267225

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: The measure of one interior angle of a regular octagon is 135 degrees -/
theorem interior_angle_regular_octagon :
  (sum_interior_angles octagon_sides) / octagon_sides = 135 := by
  sorry

end interior_angle_regular_octagon_l2672_267225


namespace polynomial_simplification_l2672_267214

theorem polynomial_simplification (x : ℝ) : 
  (5 * x^10 + 8 * x^9 + 2 * x^8) + (3 * x^10 + x^9 + 4 * x^8 + 7 * x^4 + 6 * x + 9) = 
  8 * x^10 + 9 * x^9 + 6 * x^8 + 7 * x^4 + 6 * x + 9 := by
  sorry

end polynomial_simplification_l2672_267214


namespace alien_attack_probability_l2672_267279

/-- The number of aliens attacking --/
def num_aliens : ℕ := 3

/-- The number of galaxies being attacked --/
def num_galaxies : ℕ := 4

/-- The number of days of the attack --/
def num_days : ℕ := 3

/-- The probability that a specific galaxy is not chosen by any alien on a given day --/
def prob_not_chosen_day : ℚ := (3/4)^num_aliens

/-- The probability that a specific galaxy is not destroyed over all days --/
def prob_not_destroyed : ℚ := prob_not_chosen_day^num_days

/-- The probability that at least one galaxy is not destroyed --/
def prob_at_least_one_not_destroyed : ℚ := num_galaxies * prob_not_destroyed

/-- The probability that all galaxies are destroyed --/
def prob_all_destroyed : ℚ := 1 - prob_at_least_one_not_destroyed

theorem alien_attack_probability : prob_all_destroyed = 45853/65536 := by
  sorry

end alien_attack_probability_l2672_267279


namespace even_function_property_l2672_267297

-- Define an even function f
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- State the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_neg : ∀ x < 0, f x = x * (x + 1)) :
  ∀ x > 0, f x = x * (x - 1) := by
sorry

end even_function_property_l2672_267297


namespace expression_simplification_l2672_267227

theorem expression_simplification (x : ℝ) : 
  (12 * x^12 - 3 * x^10 + 5 * x^9) + (-x^12 + 2 * x^10 + x^9 + 4 * x^4 + 6 * x^2 + 9) = 
  11 * x^12 - x^10 + 6 * x^9 + 4 * x^4 + 6 * x^2 + 9 := by
sorry

end expression_simplification_l2672_267227


namespace arithmetic_sequence_divisibility_l2672_267288

-- Define the arithmetic sequences
def arithmetic_seq (a₁ d : ℕ+) : ℕ → ℕ
  | n => a₁ + (n - 1) * d

theorem arithmetic_sequence_divisibility 
  (a₁ d_a b₁ d_b : ℕ+) 
  (h : ∃ (S : Set (ℕ × ℕ)), S.Infinite ∧ 
    ∀ (i j : ℕ), (i, j) ∈ S → 
      i ≤ j ∧ j ≤ i + 2021 ∧ 
      (arithmetic_seq a₁ d_a i) ∣ (arithmetic_seq b₁ d_b j)) :
  ∀ i : ℕ, ∃ j : ℕ, (arithmetic_seq a₁ d_a i) ∣ (arithmetic_seq b₁ d_b j) := by
  sorry

end arithmetic_sequence_divisibility_l2672_267288


namespace company_picnic_teams_l2672_267207

theorem company_picnic_teams (managers : ℕ) (employees : ℕ) (teams : ℕ) :
  managers = 3 →
  employees = 3 →
  teams = 3 →
  (managers + employees) / teams = 2 := by
sorry

end company_picnic_teams_l2672_267207


namespace ashley_amount_l2672_267203

theorem ashley_amount (ashley betty carlos dick elgin : ℕ) : 
  ashley + betty + carlos + dick + elgin = 86 →
  ashley = betty + 20 →
  (betty = carlos + 9 ∨ carlos = betty + 9) →
  (carlos = dick + 6 ∨ dick = carlos + 6) →
  (dick = elgin + 7 ∨ elgin = dick + 7) →
  elgin = ashley + 10 →
  ashley = 24 := by sorry

end ashley_amount_l2672_267203


namespace unique_positive_zero_implies_a_less_than_negative_two_l2672_267296

/-- Given a cubic function f(x) = ax^3 - 3x^2 + 1 with a unique positive zero,
    prove that the coefficient a must be less than -2. -/
theorem unique_positive_zero_implies_a_less_than_negative_two 
  (a : ℝ) (x₀ : ℝ) (h_unique : ∀ x : ℝ, a * x^3 - 3 * x^2 + 1 = 0 ↔ x = x₀) 
  (h_positive : x₀ > 0) : 
  a < -2 := by
  sorry

end unique_positive_zero_implies_a_less_than_negative_two_l2672_267296


namespace rectangle_dimensions_l2672_267240

/-- Given a rectangle with dimensions (x - 3) by (2x + 3) and area 4x - 9, prove that x = 7/2 -/
theorem rectangle_dimensions (x : ℝ) : 
  (x - 3) * (2 * x + 3) = 4 * x - 9 → x = 7 / 2 := by
  sorry

end rectangle_dimensions_l2672_267240


namespace difference_of_squares_divided_l2672_267209

theorem difference_of_squares_divided : (113^2 - 107^2) / 6 = 220 := by
  sorry

end difference_of_squares_divided_l2672_267209


namespace birdseed_mix_theorem_l2672_267256

/-- The percentage of millet in Brand B -/
def brand_b_millet : ℝ := 0.65

/-- The percentage of Brand A in the mix -/
def mix_brand_a : ℝ := 0.60

/-- The percentage of Brand B in the mix -/
def mix_brand_b : ℝ := 0.40

/-- The percentage of millet in the final mix -/
def final_mix_millet : ℝ := 0.50

/-- The percentage of millet in Brand A -/
def brand_a_millet : ℝ := 0.40

theorem birdseed_mix_theorem :
  mix_brand_a * brand_a_millet + mix_brand_b * brand_b_millet = final_mix_millet :=
by sorry

end birdseed_mix_theorem_l2672_267256


namespace simplify_expression_l2672_267299

theorem simplify_expression : (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 := by
  sorry

end simplify_expression_l2672_267299


namespace annual_rent_per_square_foot_l2672_267248

/-- Calculates the annual rent per square foot of a shop given its dimensions and monthly rent. -/
theorem annual_rent_per_square_foot 
  (length width : ℝ) 
  (monthly_rent : ℝ) 
  (h1 : length = 18) 
  (h2 : width = 20) 
  (h3 : monthly_rent = 3600) : 
  (monthly_rent * 12) / (length * width) = 120 := by
  sorry

end annual_rent_per_square_foot_l2672_267248


namespace sally_fruit_spending_l2672_267286

/-- The total amount Sally spent on fruit --/
def total_spent (peach_price_after_coupon : ℝ) (peach_coupon : ℝ) (cherry_price : ℝ) (apple_price : ℝ) (apple_discount_percent : ℝ) : ℝ :=
  let peach_price := peach_price_after_coupon + peach_coupon
  let peach_and_cherry := peach_price + cherry_price
  let apple_discount := apple_price * apple_discount_percent
  let apple_price_discounted := apple_price - apple_discount
  peach_and_cherry + apple_price_discounted

/-- Theorem stating the total amount Sally spent on fruit --/
theorem sally_fruit_spending :
  total_spent 12.32 3 11.54 20 0.15 = 43.86 := by
  sorry

#eval total_spent 12.32 3 11.54 20 0.15

end sally_fruit_spending_l2672_267286


namespace circle_inequality_theta_range_l2672_267244

theorem circle_inequality_theta_range :
  ∀ θ : ℝ,
  (0 ≤ θ ∧ θ ≤ 2 * Real.pi) →
  (∀ x y : ℝ, (x - 2 * Real.cos θ)^2 + (y - 2 * Real.sin θ)^2 = 1 → x ≤ y) →
  (5 * Real.pi / 12 ≤ θ ∧ θ ≤ 13 * Real.pi / 12) :=
by sorry

end circle_inequality_theta_range_l2672_267244


namespace quadratic_root_property_l2672_267220

theorem quadratic_root_property (m : ℝ) : 
  (∃ α β : ℝ, (3 * α^2 + m * α - 4 = 0) ∧ 
              (3 * β^2 + m * β - 4 = 0) ∧ 
              (α * β = 2 * (α^3 + β^3))) ↔ 
  (m = -1.5 ∨ m = 6 ∨ m = -2.4) :=
by sorry

end quadratic_root_property_l2672_267220


namespace abc_inequality_l2672_267280

theorem abc_inequality (a b c : ℝ) 
  (ha : a = (1/3)^(2/3))
  (hb : b = (1/5)^(2/3))
  (hc : c = (4/9)^(1/3)) :
  b < a ∧ a < c :=
by sorry

end abc_inequality_l2672_267280


namespace density_function_properties_l2672_267224

/-- A density function that satisfies specific integral properties --/
noncomputable def f (g f_ζ : ℝ → ℝ) (x : ℝ) : ℝ := (g (-x) + f_ζ x) / 2

/-- The theorem stating the properties of the density function --/
theorem density_function_properties
  (g f_ζ : ℝ → ℝ)
  (hg : ∀ x, g (-x) = -g x)  -- g is odd
  (hf_ζ : ∀ x, f_ζ (-x) = f_ζ x)  -- f_ζ is even
  (hf_density : ∀ x, f g f_ζ x ≥ 0 ∧ ∫ x, f g f_ζ x = 1)  -- f is a density function
  : (∃ x, f g f_ζ x ≠ f g f_ζ (-x))  -- f is not even
  ∧ (∀ n : ℕ, n ≥ 1 → ∫ x in Set.Ici 0, |x|^n * f g f_ζ x = ∫ x in Set.Iic 0, |x|^n * f g f_ζ x) :=
sorry

end density_function_properties_l2672_267224


namespace unique_three_digit_number_l2672_267217

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (∃ a b c : ℕ, a > 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
    n = 100 * a + 10 * b + c ∧
    n = 5 * a * b * c) ∧
  n = 175 := by
  sorry

end unique_three_digit_number_l2672_267217


namespace rectangular_to_polar_conversion_l2672_267238

theorem rectangular_to_polar_conversion :
  let x : ℝ := 2
  let y : ℝ := -2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 7 * π / 4
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π) ∧
  r = 2 * Real.sqrt 2 ∧
  θ = 7 * π / 4 ∧
  x = r * Real.cos θ ∧
  y = r * Real.sin θ :=
by sorry

end rectangular_to_polar_conversion_l2672_267238


namespace amp_eight_five_l2672_267216

-- Define the & operation
def amp (a b : ℤ) : ℤ := (a + b)^2 * (a - b)

-- State the theorem
theorem amp_eight_five : amp 8 5 = 507 := by
  sorry

end amp_eight_five_l2672_267216


namespace max_pieces_is_18_l2672_267252

/-- Represents the size of a square cake piece -/
inductive PieceSize
  | Small : PieceSize  -- 2" x 2"
  | Medium : PieceSize -- 4" x 4"
  | Large : PieceSize  -- 6" x 6"

/-- Represents a configuration of cake pieces -/
structure CakeConfiguration where
  small_pieces : Nat
  medium_pieces : Nat
  large_pieces : Nat

/-- Checks if a given configuration fits within a 20" x 20" cake -/
def fits_in_cake (config : CakeConfiguration) : Prop :=
  2 * config.small_pieces + 4 * config.medium_pieces + 6 * config.large_pieces ≤ 400

/-- The maximum number of pieces that can be cut from the cake -/
def max_pieces : Nat := 18

/-- Theorem stating that the maximum number of pieces is 18 -/
theorem max_pieces_is_18 :
  ∀ (config : CakeConfiguration),
    fits_in_cake config →
    config.small_pieces + config.medium_pieces + config.large_pieces ≤ max_pieces :=
by sorry

end max_pieces_is_18_l2672_267252


namespace golden_ratio_roots_l2672_267245

theorem golden_ratio_roots (r : ℝ) : r^2 = r + 1 → r^6 = 8*r + 5 := by
  sorry

end golden_ratio_roots_l2672_267245


namespace f_increasing_and_odd_l2672_267210

def f (x : ℝ) : ℝ := x^3

theorem f_increasing_and_odd :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end f_increasing_and_odd_l2672_267210


namespace gift_shop_combinations_l2672_267201

/-- The number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 10

/-- The number of ribbon colors -/
def ribbon_colors : ℕ := 5

/-- The number of gift card types -/
def gift_card_types : ℕ := 6

/-- The number of required ribbon colors (silver and gold) -/
def required_ribbon_colors : ℕ := 2

/-- The total number of possible combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * required_ribbon_colors * gift_card_types

theorem gift_shop_combinations :
  total_combinations = 120 :=
by sorry

end gift_shop_combinations_l2672_267201


namespace remainder_divisibility_l2672_267251

theorem remainder_divisibility (x : ℤ) : x % 72 = 19 → x % 8 = 3 := by
  sorry

end remainder_divisibility_l2672_267251


namespace ratio_u_to_x_l2672_267294

theorem ratio_u_to_x (u v x y : ℚ) 
  (h1 : u / v = 5 / 2)
  (h2 : x / y = 4 / 1)
  (h3 : v / y = 3 / 4) :
  u / x = 15 / 32 := by
  sorry

end ratio_u_to_x_l2672_267294


namespace least_k_value_l2672_267235

theorem least_k_value (k : ℤ) : ∀ n : ℤ, n ≥ 7 ↔ (0.00010101 * (10 : ℝ)^n > 1000) :=
by sorry

end least_k_value_l2672_267235


namespace quotient_sum_and_difference_l2672_267275

theorem quotient_sum_and_difference (a b : ℝ) (h : a / b = -1) : 
  (a + b = 0) ∧ (|a - b| = 2 * |b|) := by
  sorry

end quotient_sum_and_difference_l2672_267275


namespace least_multiple_of_25_greater_than_390_l2672_267204

theorem least_multiple_of_25_greater_than_390 :
  ∀ n : ℕ, n > 0 → 25 ∣ n → n > 390 → n ≥ 400 :=
by
  sorry

end least_multiple_of_25_greater_than_390_l2672_267204


namespace prime_factors_count_l2672_267269

theorem prime_factors_count : 
  let expression := [(2, 25), (3, 17), (5, 11), (7, 8), (11, 4), (13, 3)]
  (expression.map (λ (p : ℕ × ℕ) => p.2)).sum = 68 := by sorry

end prime_factors_count_l2672_267269


namespace equation_solution_l2672_267242

theorem equation_solution : ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end equation_solution_l2672_267242


namespace star_six_three_l2672_267285

def star (a b : ℝ) : ℝ := 4*a + 3*b - 2*a*b

theorem star_six_three : star 6 3 = -3 := by sorry

end star_six_three_l2672_267285


namespace train_cars_problem_l2672_267273

theorem train_cars_problem (total_cars engine_and_caboose passenger_cars cargo_cars : ℕ) :
  total_cars = 71 →
  engine_and_caboose = 2 →
  cargo_cars = passenger_cars / 2 + 3 →
  total_cars = passenger_cars + cargo_cars + engine_and_caboose →
  passenger_cars = 44 := by
sorry

end train_cars_problem_l2672_267273


namespace isosceles_triangles_not_necessarily_congruent_l2672_267274

/-- An isosceles triangle with acute angles -/
structure AcuteIsoscelesTriangle where
  /-- The length of the equal sides (legs) of the triangle -/
  legLength : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The angle at the apex of the triangle (in radians) -/
  apexAngle : ℝ
  /-- The apex angle is acute -/
  acuteAngle : apexAngle < Real.pi / 4
  /-- The leg length is positive -/
  legPositive : legLength > 0
  /-- The inradius is positive -/
  inradiusPositive : inradius > 0

/-- The theorem stating that two isosceles triangles with the same leg length and inradius
    are not necessarily congruent -/
theorem isosceles_triangles_not_necessarily_congruent :
  ∃ (t1 t2 : AcuteIsoscelesTriangle),
    t1.legLength = t2.legLength ∧
    t1.inradius = t2.inradius ∧
    t1.apexAngle ≠ t2.apexAngle :=
  sorry

end isosceles_triangles_not_necessarily_congruent_l2672_267274


namespace extreme_points_count_a_range_l2672_267233

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - a * x^2

def has_extreme_points (n : ℕ) (f : ℝ → ℝ) : Prop :=
  ∃ (S : Finset ℝ), S.card = n ∧ ∀ x ∈ S, (deriv f) x = 0 ∧
    ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y ≠ x → (deriv f) y ≠ 0

theorem extreme_points_count (a : ℝ) :
  (a ≤ 0 → has_extreme_points 1 (f a)) ∧
  (0 < a ∧ a < 1/2 → has_extreme_points 2 (f a)) ∧
  (a = 1/2 → has_extreme_points 0 (f a)) ∧
  (a > 1/2 → has_extreme_points 2 (f a)) :=
sorry

theorem a_range (a : ℝ) :
  (∀ x : ℝ, f a x + Real.exp x ≥ x^3 + x) → a ≤ Real.exp 1 - 2 :=
sorry

end extreme_points_count_a_range_l2672_267233


namespace parallel_vectors_y_value_l2672_267265

-- Define the vectors
def a : ℝ × ℝ := (2, 5)
def b : ℝ → ℝ × ℝ := λ y ↦ (1, y)

-- Define parallel vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- Theorem statement
theorem parallel_vectors_y_value :
  parallel a (b y) → y = 5/2 := by sorry

end parallel_vectors_y_value_l2672_267265


namespace total_students_count_l2672_267277

/-- Represents the arrangement of students in two rows -/
structure StudentArrangement where
  boys_count : ℕ
  girls_count : ℕ
  rajan_left_position : ℕ
  vinay_right_position : ℕ
  boys_between_rajan_vinay : ℕ
  deepa_left_position : ℕ

/-- The total number of students in both rows -/
def total_students (arrangement : StudentArrangement) : ℕ :=
  arrangement.boys_count + arrangement.girls_count

/-- The theorem stating the total number of students given the conditions -/
theorem total_students_count (arrangement : StudentArrangement) 
  (h1 : arrangement.boys_count = arrangement.girls_count)
  (h2 : arrangement.rajan_left_position = 6)
  (h3 : arrangement.vinay_right_position = 10)
  (h4 : arrangement.boys_between_rajan_vinay = 8)
  (h5 : arrangement.deepa_left_position = 5)
  : total_students arrangement = 48 := by
  sorry

end total_students_count_l2672_267277


namespace line_through_point_parallel_to_given_l2672_267211

-- Define the given line
def given_line : ℝ → ℝ → Prop :=
  λ x y => x + 2 * y - 1 = 0

-- Define the point that the desired line passes through
def point : ℝ × ℝ := (2, 0)

-- Define the equation of the desired line
def desired_line : ℝ → ℝ → Prop :=
  λ x y => x + 2 * y - 2 = 0

-- Theorem statement
theorem line_through_point_parallel_to_given :
  (desired_line point.1 point.2) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ∀ x y : ℝ, desired_line x y ↔ given_line (x + k) (y + k/2)) :=
by sorry

end line_through_point_parallel_to_given_l2672_267211


namespace roots_of_polynomial_l2672_267234

def p (x : ℝ) : ℝ := 8*x^4 + 26*x^3 - 66*x^2 + 24*x

theorem roots_of_polynomial :
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p (3/2) = 0) ∧ (p (-4) = 0) :=
by sorry

end roots_of_polynomial_l2672_267234


namespace rectangle_area_l2672_267284

/-- Given a rectangle with diagonal length x and length three times its width, 
    the area of the rectangle is 3x^2/10 -/
theorem rectangle_area (x : ℝ) : 
  ∃ (w : ℝ), w > 0 ∧ x^2 = (3*w)^2 + w^2 → 3*w^2 = (3/10) * x^2 := by
  sorry

end rectangle_area_l2672_267284


namespace cos_300_degrees_l2672_267298

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end cos_300_degrees_l2672_267298


namespace cone_base_circumference_l2672_267289

theorem cone_base_circumference 
  (V : ℝ) (l : ℝ) (θ : ℝ) (h : ℝ) (r : ℝ) (C : ℝ) :
  V = 27 * Real.pi ∧ 
  l = 6 ∧ 
  θ = Real.pi / 3 ∧ 
  h = l * Real.cos θ ∧ 
  V = 1/3 * Real.pi * r^2 * h ∧ 
  C = 2 * Real.pi * r
  → C = 6 * Real.sqrt 3 * Real.pi := by sorry

end cone_base_circumference_l2672_267289


namespace rhombus_diagonals_perpendicular_l2672_267208

/-- A rhombus is a quadrilateral with four equal sides. -/
structure Rhombus where
  sides : Fin 4 → ℝ
  sides_equal : ∀ (i j : Fin 4), sides i = sides j

/-- The diagonals of a rhombus. -/
def Rhombus.diagonals (r : Rhombus) : Fin 2 → ℝ × ℝ := sorry

/-- Two lines are perpendicular if their dot product is zero. -/
def perpendicular (l1 l2 : ℝ × ℝ) : Prop :=
  l1.1 * l2.1 + l1.2 * l2.2 = 0

/-- The diagonals of a rhombus are always perpendicular to each other. -/
theorem rhombus_diagonals_perpendicular (r : Rhombus) :
  perpendicular (r.diagonals 0) (r.diagonals 1) := by sorry

end rhombus_diagonals_perpendicular_l2672_267208


namespace three_blocks_selection_count_l2672_267200

-- Define the size of the grid
def grid_size : ℕ := 5

-- Define the number of blocks to select
def blocks_to_select : ℕ := 3

-- Theorem statement
theorem three_blocks_selection_count :
  (grid_size.choose blocks_to_select) * (grid_size.choose blocks_to_select) * (blocks_to_select.factorial) = 600 := by
  sorry

end three_blocks_selection_count_l2672_267200


namespace difference_of_ones_and_zeros_l2672_267262

-- Define the decimal number
def decimal_number : ℕ := 173

-- Define the binary representation as a list of bits
def binary_representation : List Bool := [true, false, true, false, true, true, false, true]

-- Define x as the number of zeros
def x : ℕ := binary_representation.filter (· = false) |>.length

-- Define y as the number of ones
def y : ℕ := binary_representation.filter (· = true) |>.length

-- Theorem to prove
theorem difference_of_ones_and_zeros : y - x = 4 := by
  sorry

end difference_of_ones_and_zeros_l2672_267262


namespace car_wash_earnings_ratio_l2672_267271

theorem car_wash_earnings_ratio (total : ℕ) (lisa tommy : ℕ) : 
  total = 60 →
  lisa = total / 2 →
  lisa = tommy + 15 →
  Nat.gcd tommy lisa = tommy →
  (tommy : ℚ) / lisa = 1 / 2 := by sorry

end car_wash_earnings_ratio_l2672_267271


namespace distance_to_midpoint_is_12_l2672_267268

/-- An isosceles triangle DEF with given side lengths -/
structure IsoscelesTriangleDEF where
  /-- The length of side DE -/
  de : ℝ
  /-- The length of side DF -/
  df : ℝ
  /-- The length of side EF -/
  ef : ℝ
  /-- DE and DF are equal -/
  de_eq_df : de = df
  /-- DE is 13 units -/
  de_is_13 : de = 13
  /-- EF is 10 units -/
  ef_is_10 : ef = 10

/-- The distance from D to the midpoint of EF in the isosceles triangle DEF -/
def distanceToMidpoint (t : IsoscelesTriangleDEF) : ℝ :=
  sorry

/-- Theorem stating that the distance from D to the midpoint of EF is 12 units -/
theorem distance_to_midpoint_is_12 (t : IsoscelesTriangleDEF) :
  distanceToMidpoint t = 12 := by
  sorry

end distance_to_midpoint_is_12_l2672_267268


namespace paco_cookies_l2672_267247

/-- Calculates the total number of cookies Paco has after buying cookies with a promotion --/
def total_cookies (initial : ℕ) (eaten : ℕ) (bought : ℕ) : ℕ :=
  let remaining := initial - eaten
  let free := 2 * bought
  let from_bakery := bought + free
  remaining + from_bakery

/-- Proves that Paco ends up with 149 cookies given the initial conditions --/
theorem paco_cookies : total_cookies 40 2 37 = 149 := by
  sorry

end paco_cookies_l2672_267247


namespace second_quadrant_trig_identity_l2672_267205

theorem second_quadrant_trig_identity (α : Real) 
  (h1 : π / 2 < α) (h2 : α < π) : 
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) + 
  Real.sqrt (1 - Real.sin α ^ 2) / Real.cos α = 1 := by
  sorry

end second_quadrant_trig_identity_l2672_267205


namespace inequality_holds_iff_a_less_than_negative_one_l2672_267293

theorem inequality_holds_iff_a_less_than_negative_one (a : ℝ) : 
  (∀ x : ℝ, |x| ≤ 1 → x^2 - (a + 1) * x + a + 1 > 0) ↔ a < -1 := by
  sorry

end inequality_holds_iff_a_less_than_negative_one_l2672_267293


namespace alex_upside_down_hours_l2672_267226

/-- The number of hours Alex needs to hang upside down each month to be tall enough for the roller coaster --/
def hours_upside_down (
  required_height : ℚ)
  (current_height : ℚ)
  (growth_rate_upside_down : ℚ)
  (natural_growth_rate : ℚ)
  (months_in_year : ℕ) : ℚ :=
  let height_difference := required_height - current_height
  let natural_growth := natural_growth_rate * months_in_year
  let additional_growth_needed := height_difference - natural_growth
  (additional_growth_needed / growth_rate_upside_down) / months_in_year

/-- Theorem stating that Alex needs to hang upside down for 2 hours each month --/
theorem alex_upside_down_hours :
  hours_upside_down 54 48 (1/12) (1/3) 12 = 2 := by
  sorry

end alex_upside_down_hours_l2672_267226


namespace bruce_purchase_amount_l2672_267292

/-- The total amount Bruce paid to the shopkeeper for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Bruce paid 1125 for his purchase -/
theorem bruce_purchase_amount :
  total_amount 9 70 9 55 = 1125 := by
  sorry

end bruce_purchase_amount_l2672_267292


namespace triangle_ratio_proof_l2672_267213

theorem triangle_ratio_proof (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let BD := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
  let DC := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  AB = 8 →
  BC = 13 →
  AC = 10 →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2))) →
  AD = 8 →
  BD / DC = 133 / 36 := by
sorry

end triangle_ratio_proof_l2672_267213


namespace final_temp_is_50_l2672_267278

/-- Represents the thermal equilibrium problem with two metal bars and water. -/
structure ThermalEquilibrium where
  initialWaterTemp : ℝ
  initialBarTemp : ℝ
  firstEquilibriumTemp : ℝ

/-- Calculates the final equilibrium temperature after adding the second metal bar. -/
def finalEquilibriumTemp (te : ThermalEquilibrium) : ℝ :=
  sorry

/-- Theorem stating that the final equilibrium temperature is 50°C. -/
theorem final_temp_is_50 (te : ThermalEquilibrium)
    (h1 : te.initialWaterTemp = 80)
    (h2 : te.initialBarTemp = 20)
    (h3 : te.firstEquilibriumTemp = 60) :
  finalEquilibriumTemp te = 50 :=
by sorry

end final_temp_is_50_l2672_267278


namespace min_value_theorem_l2672_267202

theorem min_value_theorem (x : ℝ) (hx : x > 0) :
  (x^2 + 6 - Real.sqrt (x^4 + 36)) / x ≥ 12 / (2 * (Real.sqrt 6 + Real.sqrt 3)) :=
sorry

end min_value_theorem_l2672_267202


namespace trapezoid_not_constructible_l2672_267237

/-- Represents a quadrilateral with sides a, b, c, d where a is parallel to c -/
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  a_parallel_c : True  -- We use True here as a placeholder for the parallel condition

/-- The triangle inequality: the sum of any two sides of a triangle must be greater than the third side -/
def triangle_inequality (x y z : ℝ) : Prop := x + y > z ∧ y + z > x ∧ z + x > y

/-- Theorem stating that a trapezoid with the given side lengths cannot be formed -/
theorem trapezoid_not_constructible : ¬ ∃ (t : Trapezoid), t.a = 16 ∧ t.b = 13 ∧ t.c = 10 ∧ t.d = 6 := by
  sorry


end trapezoid_not_constructible_l2672_267237


namespace solve_system_for_p_l2672_267282

theorem solve_system_for_p (p q : ℚ) 
  (eq1 : 3 * p + 4 * q = 15) 
  (eq2 : 4 * p + 3 * q = 18) : 
  p = 27 / 7 := by sorry

end solve_system_for_p_l2672_267282


namespace sqrt_8_simplification_l2672_267287

theorem sqrt_8_simplification :
  Real.sqrt 8 = 2 * Real.sqrt 2 := by sorry

end sqrt_8_simplification_l2672_267287


namespace triangle_side_length_l2672_267267

theorem triangle_side_length
  (A B C : ℝ)  -- Angles of the triangle
  (AB BC AC : ℝ)  -- Sides of the triangle
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)  -- Angle sum theorem
  (h5 : Real.cos (A + 2*C - B) + Real.sin (B + C - A) = 2)
  (h6 : AB = 2)
  : BC = 2 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l2672_267267


namespace sqrt_4_plus_2_inv_l2672_267249

theorem sqrt_4_plus_2_inv : Real.sqrt 4 + 2⁻¹ = 5/2 := by sorry

end sqrt_4_plus_2_inv_l2672_267249


namespace largest_square_area_l2672_267259

-- Define a right-angled triangle with squares on each side
structure RightTriangleWithSquares where
  xy : ℝ  -- Length of side XY
  yz : ℝ  -- Length of side YZ
  xz : ℝ  -- Length of hypotenuse XZ
  right_angle : xz^2 = xy^2 + yz^2  -- Pythagorean theorem

-- Theorem statement
theorem largest_square_area
  (t : RightTriangleWithSquares)
  (sum_of_squares : t.xy^2 + t.yz^2 + t.xz^2 = 450) :
  t.xz^2 = 225 := by
sorry

end largest_square_area_l2672_267259


namespace range_of_m_l2672_267250

-- Define the equations
def P (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- State the theorem
theorem range_of_m :
  (∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m)) →
  (∀ m : ℝ, m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
sorry

end range_of_m_l2672_267250


namespace angle_ratio_is_one_fourth_l2672_267266

-- Define the triangle ABC
variable (A B C : Point) (ABC : Triangle A B C)

-- Define the points P and Q
variable (P Q : Point)

-- Define the angles
def angle (X Y Z : Point) : ℝ := sorry

-- BP and BQ bisect angle ABC
axiom bp_bisects : angle A B P = angle P B C
axiom bq_bisects : angle A B Q = angle Q B C

-- BM is the bisector of angle PBQ
variable (M : Point)
axiom bm_bisects : angle P B M = angle M B Q

-- Theorem statement
theorem angle_ratio_is_one_fourth :
  (angle M B Q) / (angle A B Q) = 1 / 4 := by sorry

end angle_ratio_is_one_fourth_l2672_267266


namespace geometric_sequence_sum_l2672_267264

theorem geometric_sequence_sum (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
    (h_geometric : ∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = r * a n)
    (h_sum1 : a 1 + a 2 + a 3 = 2)
    (h_sum2 : a 3 + a 4 + a 5 = 8) :
  a 4 + a 5 + a 6 = 16 := by
sorry

end geometric_sequence_sum_l2672_267264


namespace complex_fraction_equals_four_l2672_267253

theorem complex_fraction_equals_four :
  1 + (1 / (1 - (1 / (1 + (1 / 2))))) = 4 := by
  sorry

end complex_fraction_equals_four_l2672_267253


namespace triangle_count_is_53_l2672_267290

/-- Represents a rectangle divided into triangles --/
structure TriangulatedRectangle where
  columns : Nat
  rows : Nat
  has_full_diagonals : Bool
  has_half_diagonals : Bool

/-- Counts the number of triangles in a TriangulatedRectangle --/
def count_triangles (rect : TriangulatedRectangle) : Nat :=
  sorry

/-- The specific rectangle described in the problem --/
def problem_rectangle : TriangulatedRectangle :=
  { columns := 6
  , rows := 3
  , has_full_diagonals := true
  , has_half_diagonals := true }

theorem triangle_count_is_53 : count_triangles problem_rectangle = 53 := by
  sorry

end triangle_count_is_53_l2672_267290


namespace square_floor_theorem_l2672_267221

/-- Represents a square floor tiled with congruent square tiles. -/
structure SquareFloor :=
  (side_length : ℕ)

/-- Calculates the number of black tiles on the diagonals of a square floor. -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- Calculates the total number of tiles on a square floor. -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem stating that a square floor with 101 black diagonal tiles has 2601 total tiles. -/
theorem square_floor_theorem (floor : SquareFloor) :
  black_tiles floor = 101 → total_tiles floor = 2601 :=
by sorry

end square_floor_theorem_l2672_267221


namespace positive_integer_solutions_of_inequality_l2672_267206

theorem positive_integer_solutions_of_inequality :
  {x : ℕ+ | (3 : ℝ) * x.val < x.val + 3} = {1} := by sorry

end positive_integer_solutions_of_inequality_l2672_267206


namespace function_is_identity_l2672_267230

/-- A function satisfying specific functional equations -/
def FunctionWithProperties (f : ℝ → ℝ) (c : ℝ) : Prop :=
  c ≠ 0 ∧ 
  (∀ x : ℝ, f (x + 1) = f x + c) ∧
  (∀ x : ℝ, f (x^2) = (f x)^2)

/-- Theorem stating that a function with the given properties is the identity function with c = 1 -/
theorem function_is_identity 
  {f : ℝ → ℝ} {c : ℝ} 
  (h : FunctionWithProperties f c) : 
  c = 1 ∧ ∀ x : ℝ, f x = x :=
sorry

end function_is_identity_l2672_267230


namespace fraction_simplification_l2672_267257

theorem fraction_simplification (x : ℝ) (h : x ≠ 1) :
  (x^2 / (x - 1)) - (1 / (x - 1)) = x + 1 := by
  sorry

end fraction_simplification_l2672_267257


namespace midpoint_y_coordinate_l2672_267243

theorem midpoint_y_coordinate (a : Real) (h1 : 0 < a) (h2 : a < π / 2) :
  let f := λ x : Real => Real.sin x
  let g := λ x : Real => Real.cos x
  let M := (a, f a)
  let N := (a, g a)
  abs (f a - g a) = 1/5 →
  (f a + g a) / 2 = 7/10 := by
sorry

end midpoint_y_coordinate_l2672_267243


namespace triangle_angle_calculation_l2672_267263

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_angle_calculation (t : Triangle) :
  t.A = Real.pi / 3 ∧ 
  t.a = 4 * Real.sqrt 3 ∧ 
  t.b = 4 * Real.sqrt 2 →
  t.B = Real.pi / 4 := by
  sorry

end triangle_angle_calculation_l2672_267263


namespace fraction_simplification_l2672_267246

theorem fraction_simplification (a : ℝ) (h : a ≠ 5) :
  (a^2 - 5*a) / (a - 5) = a := by sorry

end fraction_simplification_l2672_267246


namespace vincent_book_cost_l2672_267232

theorem vincent_book_cost (animal_books : ℕ) (space_books : ℕ) (train_books : ℕ) (cost_per_book : ℕ) :
  animal_books = 15 →
  space_books = 4 →
  train_books = 6 →
  cost_per_book = 26 →
  (animal_books + space_books + train_books) * cost_per_book = 650 :=
by
  sorry

end vincent_book_cost_l2672_267232


namespace clothing_distribution_l2672_267276

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 59)
  (h2 : first_load = 32)
  (h3 : num_small_loads = 9)
  (h4 : first_load < total) :
  (total - first_load) / num_small_loads = 3 := by
  sorry

end clothing_distribution_l2672_267276


namespace speed_calculation_l2672_267283

theorem speed_calculation (v : ℝ) (t : ℝ) (h1 : t > 0) :
  v * t = (v + 18) * (2/3 * t) → v = 36 :=
by sorry

end speed_calculation_l2672_267283


namespace hanna_erasers_count_l2672_267236

/-- The number of erasers Tanya has -/
def tanya_erasers : ℕ := 20

/-- The number of red erasers Tanya has -/
def tanya_red_erasers : ℕ := tanya_erasers / 2

/-- The number of erasers Rachel has -/
def rachel_erasers : ℕ := tanya_red_erasers / 2 - 3

/-- The number of erasers Hanna has -/
def hanna_erasers : ℕ := rachel_erasers * 2

theorem hanna_erasers_count : hanna_erasers = 4 := by
  sorry

end hanna_erasers_count_l2672_267236


namespace function_properties_l2672_267260

theorem function_properties (f : ℝ → ℝ) (h1 : f (-2) > f (-1)) (h2 : f (-1) < f 0) :
  ¬ (
    (∀ x y, -2 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≥ f y) ∧
    (∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≤ f y)
  ) ∧
  ¬ (
    (∀ x y, -2 ≤ x ∧ x ≤ y ∧ y ≤ -1 → f x ≤ f y) ∧
    (∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≥ f y)
  ) ∧
  ¬ (∀ x, -2 ≤ x ∧ x ≤ 0 → f x ≥ f (-1)) :=
by sorry

end function_properties_l2672_267260


namespace petes_number_l2672_267229

theorem petes_number (x : ℚ) : 4 * (2 * x + 10) = 120 → x = 10 := by
  sorry

end petes_number_l2672_267229


namespace smallest_y_squared_l2672_267261

/-- An isosceles trapezoid with a tangent circle --/
structure IsoscelesTrapezoidWithCircle where
  -- The length of the longer base
  pq : ℝ
  -- The length of the shorter base
  rs : ℝ
  -- The length of the equal sides
  y : ℝ
  -- Assumption that pq > rs
  h_pq_gt_rs : pq > rs

/-- The theorem stating the smallest possible value of y^2 --/
theorem smallest_y_squared (t : IsoscelesTrapezoidWithCircle) 
  (h_pq : t.pq = 120) (h_rs : t.rs = 25) :
  ∃ (n : ℝ), n^2 = 4350 ∧ ∀ (y : ℝ), y ≥ n → 
  ∃ (c : IsoscelesTrapezoidWithCircle), c.pq = 120 ∧ c.rs = 25 ∧ c.y = y :=
by sorry

end smallest_y_squared_l2672_267261


namespace exists_cheaper_a_l2672_267231

/-- Represents the charge for printing x copies from Company A -/
def company_a_charge (x : ℝ) : ℝ := 0.2 * x + 200

/-- Represents the charge for printing x copies from Company B -/
def company_b_charge (x : ℝ) : ℝ := 0.4 * x

/-- Theorem stating that there exists a number of copies where Company A is cheaper than Company B -/
theorem exists_cheaper_a : ∃ x : ℝ, company_a_charge x < company_b_charge x :=
sorry

end exists_cheaper_a_l2672_267231


namespace xy_plus_reciprocal_minimum_l2672_267295

theorem xy_plus_reciprocal_minimum (x y : ℝ) (hx : x < 0) (hy : y < 0) (hsum : x + y = -1) :
  ∀ z, z = x * y + 1 / (x * y) → z ≥ 17/4 :=
by sorry

end xy_plus_reciprocal_minimum_l2672_267295


namespace count_valid_arrangements_l2672_267272

/-- Represents a seating arrangement for two families in two cars -/
structure SeatingArrangement where
  audi : Finset (Fin 6)
  jetta : Finset (Fin 6)

/-- The set of all valid seating arrangements -/
def validArrangements : Finset SeatingArrangement :=
  sorry

/-- The number of adults in the group -/
def numAdults : Nat := 4

/-- The number of children in the group -/
def numChildren : Nat := 2

/-- The maximum capacity of each car -/
def maxCapacity : Nat := 4

/-- Theorem stating the number of valid seating arrangements -/
theorem count_valid_arrangements :
  Finset.card validArrangements = 48 := by
  sorry

end count_valid_arrangements_l2672_267272


namespace flower_bed_side_length_l2672_267239

/-- Given a rectangular flower bed with area 6a^2 - 4ab + 2a and one side of length 2a,
    the length of the other side is 3a - 2b + 1 -/
theorem flower_bed_side_length (a b : ℝ) :
  let area := 6 * a^2 - 4 * a * b + 2 * a
  let side1 := 2 * a
  area / side1 = 3 * a - 2 * b + 1 := by
sorry

end flower_bed_side_length_l2672_267239


namespace log_and_exp_problem_l2672_267218

theorem log_and_exp_problem :
  (Real.log 9 / Real.log 3 = 2) ∧
  (∀ a : ℝ, a = Real.log 3 / Real.log 4 → 2^a = Real.sqrt 3) := by
  sorry

end log_and_exp_problem_l2672_267218


namespace largest_base5_3digit_in_base10_l2672_267291

/-- The largest three-digit number in base-5 -/
def largest_base5_3digit : ℕ := 4 * 5^2 + 4 * 5^1 + 4 * 5^0

/-- Theorem: The largest three-digit number in base-5, when converted to base-10, is equal to 124 -/
theorem largest_base5_3digit_in_base10 : largest_base5_3digit = 124 := by
  sorry

end largest_base5_3digit_in_base10_l2672_267291


namespace range_of_omega_l2672_267258

/-- Given vectors a and b, and a function f, prove the range of ω -/
theorem range_of_omega (ω : ℝ) (x : ℝ) : 
  ω > 0 →
  let a := (Real.sin (ω/2 * x), Real.sin (ω * x))
  let b := (Real.sin (ω/2 * x), (1/2 : ℝ))
  let f := λ x => (a.1 * b.1 + a.2 * b.2) - 1/2
  (∀ x ∈ Set.Ioo π (2*π), f x ≠ 0) →
  ω ∈ Set.Ioc 0 (1/8) ∪ Set.Icc (1/4) (5/8) :=
sorry

end range_of_omega_l2672_267258


namespace max_value_is_120_l2672_267219

def is_valid_assignment (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  b ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  c ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) ∧
  d ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ)

def expression (a b c d : ℕ) : ℚ :=
  (a : ℚ) / ((b : ℚ) / ((c * d : ℚ)))

theorem max_value_is_120 :
  ∀ a b c d : ℕ, is_valid_assignment a b c d →
    expression a b c d ≤ 120 :=
by sorry

end max_value_is_120_l2672_267219


namespace set_equality_l2672_267228

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x < 1}

-- Define set N
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem set_equality : {x : ℝ | x ≥ 2} = (M ∪ N)ᶜ := by
  sorry

end set_equality_l2672_267228


namespace triple_composition_even_l2672_267270

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

-- State the theorem
theorem triple_composition_even
  (g : ℝ → ℝ)
  (h : EvenFunction g) :
  EvenFunction (fun x ↦ g (g (g x))) :=
sorry

end triple_composition_even_l2672_267270


namespace larger_number_problem_l2672_267281

theorem larger_number_problem (x y : ℝ) : 
  x - y = 5 → x + y = 37 → max x y = 21 := by sorry

end larger_number_problem_l2672_267281


namespace function_zeros_l2672_267215

def has_at_least_n_zeros (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : Prop :=
  ∃ (S : Finset ℝ), S.card ≥ n ∧ (∀ x ∈ S, a < x ∧ x ≤ b ∧ f x = 0)

theorem function_zeros
  (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_symmetry : ∀ x, f (2 + x) = f (2 - x))
  (h_zero_in_interval : ∃ x, 0 < x ∧ x < 4 ∧ f x = 0)
  (h_zero_at_origin : f 0 = 0) :
  has_at_least_n_zeros f (-8) 10 9 :=
sorry

end function_zeros_l2672_267215


namespace min_value_quadratic_expression_l2672_267223

theorem min_value_quadratic_expression (a b c : ℝ) :
  a < b →
  (∀ x, a * x^2 + b * x + c ≥ 0) →
  (a + b + c) / (b - a) ≥ 3 :=
sorry

end min_value_quadratic_expression_l2672_267223


namespace equidistant_point_count_l2672_267222

/-- A quadrilateral is a polygon with four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- A rectangle is a quadrilateral with four right angles. -/
def IsRectangle (q : Quadrilateral) : Prop := sorry

/-- A trapezoid is a quadrilateral with at least one pair of parallel sides. -/
def IsTrapezoid (q : Quadrilateral) : Prop := sorry

/-- A trapezoid has congruent base angles if the angles adjacent to each parallel side are congruent. -/
def HasCongruentBaseAngles (q : Quadrilateral) : Prop := sorry

/-- A point is equidistant from all vertices of a quadrilateral if its distance to each vertex is the same. -/
def HasEquidistantPoint (q : Quadrilateral) : Prop := sorry

/-- The theorem states that among rectangles and trapezoids with congruent base angles, 
    exactly two types of quadrilaterals have a point equidistant from all four vertices. -/
theorem equidistant_point_count :
  ∃ (q1 q2 : Quadrilateral),
    (IsRectangle q1 ∨ (IsTrapezoid q1 ∧ HasCongruentBaseAngles q1)) ∧
    (IsRectangle q2 ∨ (IsTrapezoid q2 ∧ HasCongruentBaseAngles q2)) ∧
    q1 ≠ q2 ∧
    HasEquidistantPoint q1 ∧
    HasEquidistantPoint q2 ∧
    (∀ q : Quadrilateral,
      (IsRectangle q ∨ (IsTrapezoid q ∧ HasCongruentBaseAngles q)) →
      HasEquidistantPoint q →
      (q = q1 ∨ q = q2)) :=
by
  sorry

end equidistant_point_count_l2672_267222


namespace more_red_polygons_l2672_267255

/-- Represents a set of points on a circle -/
structure PointSet where
  white : ℕ
  red : ℕ

/-- Counts the number of polygons that can be formed from a given set of points -/
def countPolygons (ps : PointSet) (includeRed : Bool) : ℕ :=
  sorry

/-- The given configuration of points -/
def circlePoints : PointSet :=
  { white := 1997, red := 1 }

theorem more_red_polygons :
  countPolygons circlePoints true > countPolygons circlePoints false :=
sorry

end more_red_polygons_l2672_267255


namespace decreasing_quadratic_condition_l2672_267254

/-- A function f(x) = ax^2 - b that is decreasing on (-∞, 0) -/
def DecreasingQuadratic (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 - b

/-- The property of being decreasing on (-∞, 0) -/
def IsDecreasingOnNegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y ∧ y < 0 → f x > f y

theorem decreasing_quadratic_condition (a b : ℝ) :
  IsDecreasingOnNegatives (DecreasingQuadratic a b) → a > 0 ∧ b ∈ Set.univ := by
  sorry

end decreasing_quadratic_condition_l2672_267254


namespace beads_per_necklace_l2672_267212

theorem beads_per_necklace (members : ℕ) (necklaces_per_member : ℕ) (total_beads : ℕ) :
  members = 9 →
  necklaces_per_member = 2 →
  total_beads = 900 →
  total_beads / (members * necklaces_per_member) = 50 := by
  sorry

end beads_per_necklace_l2672_267212
