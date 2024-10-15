import Mathlib

namespace NUMINAMATH_CALUDE_gcd_1234_2047_l875_87535

theorem gcd_1234_2047 : Nat.gcd 1234 2047 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1234_2047_l875_87535


namespace NUMINAMATH_CALUDE_quadrilateral_angle_sum_l875_87572

theorem quadrilateral_angle_sum (a b c d : ℝ) (α β γ δ : ℝ) 
  (ha : a = 15) (hb : b = 20) (hc : c = 25) (hd : d = 33)
  (hα : α = 100) (hβ : β = 80) (hγ : γ = 105) (hδ : δ = 75) :
  α + β + γ + δ = 360 := by
sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_sum_l875_87572


namespace NUMINAMATH_CALUDE_power_of_power_l875_87549

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l875_87549


namespace NUMINAMATH_CALUDE_fair_coin_heads_prob_equals_frequency_l875_87597

/-- Represents the outcome of a coin toss experiment -/
structure CoinTossExperiment where
  total_tosses : ℕ
  heads_count : ℕ
  heads_frequency : ℝ

/-- Defines what it means for an experiment to be valid -/
def is_valid_experiment (e : CoinTossExperiment) : Prop :=
  e.total_tosses > 0 ∧ 
  e.heads_count ≤ e.total_tosses ∧ 
  e.heads_frequency = (e.heads_count : ℝ) / (e.total_tosses : ℝ)

/-- The probability of a fair coin landing heads up -/
def fair_coin_heads_probability : ℝ := 0.5005

/-- Pearson's experiment data -/
def pearson_experiment : CoinTossExperiment := {
  total_tosses := 24000,
  heads_count := 12012,
  heads_frequency := 0.5005
}

/-- Theorem stating that the probability of a fair coin landing heads up
    is equal to the frequency observed in Pearson's large-scale experiment -/
theorem fair_coin_heads_prob_equals_frequency 
  (h_valid : is_valid_experiment pearson_experiment)
  (h_large : pearson_experiment.total_tosses ≥ 10000) :
  fair_coin_heads_probability = pearson_experiment.heads_frequency := by
  sorry


end NUMINAMATH_CALUDE_fair_coin_heads_prob_equals_frequency_l875_87597


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l875_87561

theorem cone_lateral_surface_area 
  (r : ℝ) 
  (V : ℝ) 
  (h : ℝ) 
  (l : ℝ) : 
  r = 6 →
  V = 30 * Real.pi →
  V = (1/3) * Real.pi * r^2 * h →
  l^2 = r^2 + h^2 →
  r * l * Real.pi = 39 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l875_87561


namespace NUMINAMATH_CALUDE_inverse_of_f_is_neg_g_neg_l875_87539

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the symmetry condition
def symmetric_wrt_x_plus_y_eq_0 (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g (-y) = -x

-- Define the inverse function
def has_inverse (f : ℝ → ℝ) : Prop :=
  ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x

-- Theorem statement
theorem inverse_of_f_is_neg_g_neg (hf : has_inverse f) (h_sym : symmetric_wrt_x_plus_y_eq_0 f g) :
  ∃ f_inv : ℝ → ℝ, (∀ x, f_inv (f x) = x ∧ f (f_inv x) = x) ∧ (∀ x, f_inv x = -g (-x)) :=
sorry

end NUMINAMATH_CALUDE_inverse_of_f_is_neg_g_neg_l875_87539


namespace NUMINAMATH_CALUDE_puzzle_solution_l875_87582

theorem puzzle_solution :
  ∀ (E H O Y A : ℕ),
    (10 ≤ E * 10 + H) ∧ (E * 10 + H < 100) ∧
    (10 ≤ O * 10 + Y) ∧ (O * 10 + Y < 100) ∧
    (10 ≤ A * 10 + Y) ∧ (A * 10 + Y < 100) ∧
    (10 ≤ O * 10 + H) ∧ (O * 10 + H < 100) ∧
    (E * 10 + H = 4 * (O * 10 + Y)) ∧
    (A * 10 + Y = 4 * (O * 10 + H)) →
    (E * 10 + H) + (O * 10 + Y) + (A * 10 + Y) + (O * 10 + H) = 150 :=
by sorry


end NUMINAMATH_CALUDE_puzzle_solution_l875_87582


namespace NUMINAMATH_CALUDE_second_store_unload_percentage_l875_87532

def initial_load : ℝ := 50000
def first_unload_percent : ℝ := 0.1
def remaining_after_deliveries : ℝ := 36000

theorem second_store_unload_percentage :
  let remaining_after_first := initial_load * (1 - first_unload_percent)
  let unloaded_at_second := remaining_after_first - remaining_after_deliveries
  (unloaded_at_second / remaining_after_first) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_store_unload_percentage_l875_87532


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l875_87585

/-- An arithmetic sequence with first term 5 and sum of first 31 terms equal to 390 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 5 ∧ 
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (Finset.sum (Finset.range 31) (λ i => a (i + 1)) = 390)

/-- The ratio of sum of odd-indexed terms to sum of even-indexed terms -/
def ratio (a : ℕ → ℚ) : ℚ :=
  (Finset.sum (Finset.filter (λ i => i % 2 = 1) (Finset.range 31)) (λ i => a (i + 1))) /
  (Finset.sum (Finset.filter (λ i => i % 2 = 0) (Finset.range 31)) (λ i => a (i + 1)))

theorem arithmetic_sequence_ratio (a : ℕ → ℚ) :
  arithmetic_sequence a → ratio a = 16 / 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l875_87585


namespace NUMINAMATH_CALUDE_bracket_mult_example_bracket_mult_equation_roots_l875_87573

-- Define the operation for real numbers
def bracket_mult (a b c d : ℝ) : ℝ := a * c - b * d

-- Theorem 1
theorem bracket_mult_example : bracket_mult (-4) 3 2 (-6) = 10 := by sorry

-- Theorem 2
theorem bracket_mult_equation_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ bracket_mult x (2*x - 1) (m*x + 1) m = 0) ↔ 
  (m ≤ 1/4 ∧ m ≠ 0) := by sorry

end NUMINAMATH_CALUDE_bracket_mult_example_bracket_mult_equation_roots_l875_87573


namespace NUMINAMATH_CALUDE_binary_101101_to_decimal_l875_87584

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (λ acc (i, b) => acc + if b then 2^i else 0) 0

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_to_decimal :
  binary_to_decimal binary_101101 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_to_decimal_l875_87584


namespace NUMINAMATH_CALUDE_commute_days_theorem_l875_87599

theorem commute_days_theorem (x : ℕ) 
  (morning_bus : ℕ) 
  (afternoon_train : ℕ) 
  (bike_commute : ℕ) 
  (h1 : morning_bus = 12) 
  (h2 : afternoon_train = 20) 
  (h3 : bike_commute = 10) 
  (h4 : x = morning_bus + afternoon_train - bike_commute) : x = 30 := by
  sorry

#check commute_days_theorem

end NUMINAMATH_CALUDE_commute_days_theorem_l875_87599


namespace NUMINAMATH_CALUDE_min_vertices_for_quadrilateral_l875_87552

theorem min_vertices_for_quadrilateral (n : ℕ) (hn : n ≥ 10) :
  let k := ⌊(3 * n : ℝ) / 4⌋ + 1
  ∀ S : Finset (Fin n),
    S.card ≥ k →
    ∃ (a b c d : Fin n), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
      ((b - a) % n = 1 ∨ (b - a) % n = n - 1) ∧
      ((c - b) % n = 1 ∨ (c - b) % n = n - 1) ∧
      ((d - c) % n = 1 ∨ (d - c) % n = n - 1) :=
by sorry

#check min_vertices_for_quadrilateral

end NUMINAMATH_CALUDE_min_vertices_for_quadrilateral_l875_87552


namespace NUMINAMATH_CALUDE_complex_calculation_equality_l875_87581

theorem complex_calculation_equality : 
  (2 * (15^2 + 35^2 + 21^2) - (3^4 + 5^4 + 7^4)) / (3 + 5 + 7) = 45 := by
  sorry

end NUMINAMATH_CALUDE_complex_calculation_equality_l875_87581


namespace NUMINAMATH_CALUDE_laran_sells_five_posters_l875_87514

/-- Represents the poster business model for Laran -/
structure PosterBusiness where
  large_posters_per_day : ℕ
  large_poster_price : ℕ
  large_poster_cost : ℕ
  small_poster_price : ℕ
  small_poster_cost : ℕ
  weekly_profit : ℕ
  school_days_per_week : ℕ

/-- Calculates the total number of posters sold per day -/
def total_posters_per_day (b : PosterBusiness) : ℕ :=
  b.large_posters_per_day + 
  ((b.weekly_profit / b.school_days_per_week - 
    (b.large_posters_per_day * (b.large_poster_price - b.large_poster_cost))) / 
   (b.small_poster_price - b.small_poster_cost))

/-- Theorem stating that Laran sells 5 posters per day -/
theorem laran_sells_five_posters (b : PosterBusiness) 
  (h1 : b.large_posters_per_day = 2)
  (h2 : b.large_poster_price = 10)
  (h3 : b.large_poster_cost = 5)
  (h4 : b.small_poster_price = 6)
  (h5 : b.small_poster_cost = 3)
  (h6 : b.weekly_profit = 95)
  (h7 : b.school_days_per_week = 5) :
  total_posters_per_day b = 5 := by
  sorry

end NUMINAMATH_CALUDE_laran_sells_five_posters_l875_87514


namespace NUMINAMATH_CALUDE_albert_more_than_joshua_l875_87511

/-- The number of rocks collected by Joshua, Jose, and Albert -/
def rock_collection (joshua jose albert : ℕ) : Prop :=
  (jose = joshua - 14) ∧ 
  (albert = jose + 20) ∧ 
  (joshua = 80)

/-- Theorem stating that Albert collected 6 more rocks than Joshua -/
theorem albert_more_than_joshua {joshua jose albert : ℕ} 
  (h : rock_collection joshua jose albert) : albert - joshua = 6 := by
  sorry

end NUMINAMATH_CALUDE_albert_more_than_joshua_l875_87511


namespace NUMINAMATH_CALUDE_intersection_point_after_rotation_l875_87504

theorem intersection_point_after_rotation (θ : Real) : 
  0 < θ ∧ θ < π / 2 → 
  (fun φ ↦ (Real.cos φ, Real.sin φ)) (θ + π / 2) = (-Real.sin θ, Real.cos θ) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_after_rotation_l875_87504


namespace NUMINAMATH_CALUDE_half_sufficient_not_necessary_l875_87500

/-- Two lines in the plane -/
structure TwoLines where
  a : ℝ
  line1 : ℝ → ℝ → Prop := λ x y => x + 2 * a * y - 1 = 0
  line2 : ℝ → ℝ → Prop := λ x y => (a - 1) * x - a * y - 1 = 0

/-- The lines are parallel -/
def are_parallel (l : TwoLines) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l.line1 x y ↔ l.line2 (k * x) (k * y)

/-- The statement that a = 1/2 is sufficient but not necessary for the lines to be parallel -/
theorem half_sufficient_not_necessary :
  (∃ l : TwoLines, l.a = 1/2 ∧ ¬are_parallel l) ∧
  (∃ l : TwoLines, l.a ≠ 1/2 ∧ are_parallel l) ∧
  (∀ l : TwoLines, l.a = 1/2 → are_parallel l) :=
sorry

end NUMINAMATH_CALUDE_half_sufficient_not_necessary_l875_87500


namespace NUMINAMATH_CALUDE_subtraction_of_large_numbers_l875_87506

theorem subtraction_of_large_numbers :
  1000000000000 - 888777888777 = 111222111223 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_large_numbers_l875_87506


namespace NUMINAMATH_CALUDE_parabola_sum_l875_87521

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℚ) : ℚ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) :
  p.x_coord (-6) = 7 →  -- vertex condition
  p.x_coord 0 = 1 →     -- point condition
  p.a + p.b + p.c = -43/6 := by
sorry

end NUMINAMATH_CALUDE_parabola_sum_l875_87521


namespace NUMINAMATH_CALUDE_max_value_of_min_expression_l875_87540

theorem max_value_of_min_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  min x (min (-1/y) (y + 1/x)) ≤ Real.sqrt 2 ∧
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ min x (min (-1/y) (y + 1/x)) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_min_expression_l875_87540


namespace NUMINAMATH_CALUDE_least_number_with_remainder_one_l875_87588

theorem least_number_with_remainder_one (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 115 → (m % 38 ≠ 1 ∨ m % 3 ≠ 1)) ∧ 
  (115 % 38 = 1 ∧ 115 % 3 = 1) := by
  sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_one_l875_87588


namespace NUMINAMATH_CALUDE_complex_magnitude_l875_87589

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 30)
  (h2 : Complex.abs (z + 3 * w) = 15)
  (h3 : Complex.abs (z - w) = 10) :
  Complex.abs z = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l875_87589


namespace NUMINAMATH_CALUDE_xy_value_l875_87593

theorem xy_value (x y : ℝ) (h : Real.sqrt (x - 3) + |y + 2| = 0) : x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l875_87593


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l875_87594

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 46 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = 100 / 11.5 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l875_87594


namespace NUMINAMATH_CALUDE_kozel_garden_problem_l875_87554

theorem kozel_garden_problem (x : ℕ) (y : ℕ) : 
  (y = 3 * x + 1) → 
  (y = 4 * (x - 1)) → 
  (x = 5 ∧ y = 16) :=
by sorry

end NUMINAMATH_CALUDE_kozel_garden_problem_l875_87554


namespace NUMINAMATH_CALUDE_smallest_factorizable_b_l875_87525

/-- 
A function that checks if a quadratic expression x^2 + bx + c 
can be factored into two binomials with integer coefficients
-/
def is_factorizable (b : ℤ) (c : ℤ) : Prop :=
  ∃ (r s : ℤ), c = r * s ∧ b = r + s

/-- 
The smallest positive integer b for which x^2 + bx + 1890 
factors into a product of two binomials with integer coefficients
-/
theorem smallest_factorizable_b : 
  (∀ b : ℤ, b > 0 ∧ b < 141 → ¬(is_factorizable b 1890)) ∧ 
  (is_factorizable 141 1890) := by
  sorry

#check smallest_factorizable_b

end NUMINAMATH_CALUDE_smallest_factorizable_b_l875_87525


namespace NUMINAMATH_CALUDE_negation_existence_statement_l875_87591

theorem negation_existence_statement :
  (¬ ∃ x : ℝ, x < -1 ∧ x^2 ≥ 1) ↔ (∀ x : ℝ, x < -1 → x^2 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_existence_statement_l875_87591


namespace NUMINAMATH_CALUDE_matching_color_probability_l875_87520

/-- Represents the number of jelly beans of each color for a person -/
structure JellyBeans where
  green : ℕ
  red : ℕ
  yellow : ℕ

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : ℕ :=
  jb.green + jb.red + jb.yellow

/-- Abe's jelly bean distribution -/
def abe : JellyBeans :=
  { green := 2, red := 2, yellow := 0 }

/-- Bob's jelly bean distribution -/
def bob : JellyBeans :=
  { green := 2, red := 3, yellow := 2 }

/-- Calculates the probability of selecting a specific color -/
def probColor (jb : JellyBeans) (color : ℕ) : ℚ :=
  color / jb.total

/-- Calculates the probability of both selecting the same color -/
def probMatchingColor (jb1 jb2 : JellyBeans) : ℚ :=
  probColor jb1 jb1.green * probColor jb2 jb2.green +
  probColor jb1 jb1.red * probColor jb2 jb2.red

theorem matching_color_probability :
  probMatchingColor abe bob = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_matching_color_probability_l875_87520


namespace NUMINAMATH_CALUDE_exists_acute_triangle_configuration_l875_87545

/-- A configuration of n points on a plane. -/
structure PointConfiguration (n : ℕ) where
  points : Fin n → ℝ × ℝ

/-- A triangle formed by three points. -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Predicate to check if a triangle is acute. -/
def is_acute (t : Triangle) : Prop :=
  sorry  -- Definition of acute triangle

/-- Function to get the i-th triangle from a point configuration. -/
def get_triangle (config : PointConfiguration n) (i : Fin n) : Triangle :=
  sorry  -- Definition to extract triangle from configuration

/-- Theorem stating the existence of a configuration with all acute triangles. -/
theorem exists_acute_triangle_configuration (n : ℕ) (h_odd : Odd n) (h_gt_3 : n > 3) :
  ∃ (config : PointConfiguration n), ∀ (i : Fin n), is_acute (get_triangle config i) :=
sorry

end NUMINAMATH_CALUDE_exists_acute_triangle_configuration_l875_87545


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l875_87513

theorem triangle_trig_identity (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C)
  (h4 : A + B + C = Real.pi) (h5 : A ≤ B) (h6 : B ≤ C)
  (h7 : (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) = Real.sqrt 3) :
  Real.sin B + Real.sin (2 * B) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l875_87513


namespace NUMINAMATH_CALUDE_f_f_has_four_distinct_roots_l875_87534

-- Define the function f
def f (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

-- State the theorem
theorem f_f_has_four_distinct_roots :
  ∃! d : ℝ, ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (∀ x : ℝ, f (f d x) = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧
    d = 2 :=
sorry

end NUMINAMATH_CALUDE_f_f_has_four_distinct_roots_l875_87534


namespace NUMINAMATH_CALUDE_sum_prime_factors_2_pow_22_minus_4_l875_87527

/-- SPF(n) denotes the sum of the prime factors of n, where the prime factors are not necessarily distinct. -/
def SPF (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of prime factors of 2^22 - 4 is 100. -/
theorem sum_prime_factors_2_pow_22_minus_4 : SPF (2^22 - 4) = 100 := by sorry

end NUMINAMATH_CALUDE_sum_prime_factors_2_pow_22_minus_4_l875_87527


namespace NUMINAMATH_CALUDE_range_of_m_l875_87515

theorem range_of_m (f : ℝ → ℝ → ℝ) (x₀ : ℝ) (h_nonzero : x₀ ≠ 0) :
  (∀ m : ℝ, f m x = 9*x - m) →
  f x₀ x₀ = f 0 x₀ →
  ∃ m : ℝ, 0 < m ∧ m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l875_87515


namespace NUMINAMATH_CALUDE_rectangle_length_from_square_perimeter_l875_87531

theorem rectangle_length_from_square_perimeter (square_side : ℝ) (rect_width : ℝ) :
  square_side = 12 →
  rect_width = 6 →
  4 * square_side = 2 * (rect_width + (18 : ℝ)) :=
by
  sorry

#check rectangle_length_from_square_perimeter

end NUMINAMATH_CALUDE_rectangle_length_from_square_perimeter_l875_87531


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l875_87596

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 - 3*x₁ - 1 = 0 → 
  x₂^2 - 3*x₂ - 1 = 0 → 
  x₁^2 * x₂ + x₁ * x₂^2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l875_87596


namespace NUMINAMATH_CALUDE_hyperbola_range_of_b_squared_l875_87503

/-- Given a hyperbola M: x^2 - y^2/b^2 = 1 (b > 0) with foci F1(-c, 0) and F2(c, 0),
    if a line parallel to one asymptote passes through F1 and intersects the other asymptote at P(-c/2, bc/2),
    and P is inside the circle x^2 + y^2 = 4b^2, then 7 - 4√3 < b^2 < 7 + 4√3 -/
theorem hyperbola_range_of_b_squared (b c : ℝ) (hb : b > 0) (hc : c^2 = b^2 + 1) :
  let P : ℝ × ℝ := (-c/2, b*c/2)
  (P.1^2 + P.2^2 < 4*b^2) → (7 - 4*Real.sqrt 3 < b^2 ∧ b^2 < 7 + 4*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_range_of_b_squared_l875_87503


namespace NUMINAMATH_CALUDE_hyperbola_focus_l875_87505

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  (x - 5)^2 / 9^2 - (y - 20)^2 / 15^2 = 1

def is_focus (x y : ℝ) : Prop :=
  hyperbola_equation x y ∧ 
  ∃ (x' y' : ℝ), hyperbola_equation x' y' ∧ 
  (x - 5)^2 + (y - 20)^2 = (x' - 5)^2 + (y' - 20)^2 ∧ 
  (x ≠ x' ∨ y ≠ y')

theorem hyperbola_focus :
  ∃ (x y : ℝ), is_focus x y ∧ 
  (∀ (x' y' : ℝ), is_focus x' y' → x' ≤ x) ∧
  x = 5 + Real.sqrt 306 ∧ y = 20 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l875_87505


namespace NUMINAMATH_CALUDE_max_triangles_is_eleven_l875_87538

/-- Represents an equilateral triangle with a line segment connecting midpoints of two sides -/
structure EquilateralTriangleWithMidline where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents the configuration of two overlapping equilateral triangles -/
structure OverlappingTriangles where
  triangle1 : EquilateralTriangleWithMidline
  triangle2 : EquilateralTriangleWithMidline
  overlap : ℝ -- Represents the degree of overlap between the triangles

/-- Counts the number of triangles formed in a given configuration -/
def countTriangles (config : OverlappingTriangles) : ℕ :=
  sorry

/-- Theorem stating that the maximum number of triangles is 11 -/
theorem max_triangles_is_eleven :
  ∃ (config : OverlappingTriangles),
    (∀ (other : OverlappingTriangles), countTriangles other ≤ countTriangles config) ∧
    countTriangles config = 11 :=
  sorry

end NUMINAMATH_CALUDE_max_triangles_is_eleven_l875_87538


namespace NUMINAMATH_CALUDE_profit_percentage_without_discount_l875_87564

theorem profit_percentage_without_discount
  (cost_price : ℝ)
  (discount_rate : ℝ)
  (profit_rate_with_discount : ℝ)
  (h_positive_cost : cost_price > 0)
  (h_discount : discount_rate = 0.05)
  (h_profit_with_discount : profit_rate_with_discount = 0.1875) :
  let selling_price_with_discount := cost_price * (1 - discount_rate)
  let profit_amount := cost_price * profit_rate_with_discount
  let selling_price_without_discount := cost_price + profit_amount
  let profit_rate_without_discount := (selling_price_without_discount - cost_price) / cost_price
  profit_rate_without_discount = profit_rate_with_discount :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_without_discount_l875_87564


namespace NUMINAMATH_CALUDE_find_divisor_l875_87548

theorem find_divisor : ∃ (x : ℕ), x > 0 ∧ 190 = 9 * x + 1 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_find_divisor_l875_87548


namespace NUMINAMATH_CALUDE_extra_red_pencil_packs_l875_87501

theorem extra_red_pencil_packs (total_packs : ℕ) (normal_red_per_pack : ℕ) (total_red_pencils : ℕ) :
  total_packs = 15 →
  normal_red_per_pack = 1 →
  total_red_pencils = 21 →
  ∃ (extra_packs : ℕ),
    extra_packs * 2 + total_packs * normal_red_per_pack = total_red_pencils ∧
    extra_packs = 3 :=
by sorry

end NUMINAMATH_CALUDE_extra_red_pencil_packs_l875_87501


namespace NUMINAMATH_CALUDE_line_equation_correct_l875_87541

/-- A line passing through a point with a given slope -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The equation of a line in slope-intercept form -/
def line_equation (l : Line) : ℝ → ℝ := fun x => l.slope * x + (l.point.2 - l.slope * l.point.1)

theorem line_equation_correct (l : Line) : 
  l.slope = 2 ∧ l.point = (0, 3) → line_equation l = fun x => 2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_correct_l875_87541


namespace NUMINAMATH_CALUDE_parabola_through_point_l875_87590

def is_parabola_equation (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, f x y ↔ y^2 = 4*a*x

theorem parabola_through_point (f : ℝ → ℝ → Prop) : 
  is_parabola_equation f →
  (∀ x y : ℝ, f x y → y^2 = x) →
  f 4 (-2) →
  ∀ x y : ℝ, f x y ↔ y^2 = x :=
sorry

end NUMINAMATH_CALUDE_parabola_through_point_l875_87590


namespace NUMINAMATH_CALUDE_trig_identity_l875_87547

theorem trig_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) :
  Real.cos (π / 6 + α / 2) ^ 2 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l875_87547


namespace NUMINAMATH_CALUDE_line_contains_point_l875_87598

/-- A line in the xy-plane is represented by the equation 2 - kx = -4y,
    where k is a real number. The line contains the point (3,-2) if and only if k = -2. -/
theorem line_contains_point (k : ℝ) : 2 - k * 3 = -4 * (-2) ↔ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l875_87598


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l875_87528

theorem min_value_squared_sum (a b t k : ℝ) (hk : k > 0) (ht : a + k * b = t) :
  a^2 + k^2 * b^2 ≥ ((1 + k^2) * t^2) / (1 + k)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l875_87528


namespace NUMINAMATH_CALUDE_matrix_multiplication_l875_87571

def A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -1; 4, 2]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![0, 6; -2, 3]

theorem matrix_multiplication :
  A * B = !![2, 15; -4, 30] := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_l875_87571


namespace NUMINAMATH_CALUDE_simple_interest_principal_l875_87537

/-- Simple interest calculation --/
theorem simple_interest_principal
  (interest : ℚ)
  (rate : ℚ)
  (time : ℚ)
  (h1 : interest = 160)
  (h2 : rate = 4 / 100)
  (h3 : time = 5) :
  interest = (800 * rate * time) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l875_87537


namespace NUMINAMATH_CALUDE_tangent_line_at_minus_one_l875_87530

def curve (x : ℝ) : ℝ := x^3

theorem tangent_line_at_minus_one : 
  let p : ℝ × ℝ := (-1, -1)
  let m : ℝ := 3 * p.1^2
  let tangent_line (x : ℝ) : ℝ := m * (x - p.1) + p.2
  ∀ x, tangent_line x = 3 * x + 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_minus_one_l875_87530


namespace NUMINAMATH_CALUDE_coplanar_vectors_lambda_l875_87524

/-- Given vectors a, b, and c in ℝ³, prove that if they are coplanar and have the specified coordinates, then the third component of c equals 65/7. -/
theorem coplanar_vectors_lambda (a b c : ℝ × ℝ × ℝ) : 
  a = (2, -1, 3) →
  b = (-1, 4, -2) →
  c.1 = 7 ∧ c.2.1 = 5 →
  (∃ (x y : ℝ), c = x • a + y • b) →
  c.2.2 = 65/7 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_vectors_lambda_l875_87524


namespace NUMINAMATH_CALUDE_candies_problem_l875_87519

theorem candies_problem (n : ℕ) (a : ℕ) (h1 : n > 0) (h2 : a > 1) 
  (h3 : ∀ i : Fin n, a = n * a - a - 7) : n * a = 21 := by
  sorry

end NUMINAMATH_CALUDE_candies_problem_l875_87519


namespace NUMINAMATH_CALUDE_fraction_order_l875_87566

theorem fraction_order : (25 : ℚ) / 21 < 23 / 19 ∧ 23 / 19 < 21 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l875_87566


namespace NUMINAMATH_CALUDE_crayon_selection_theorem_l875_87560

def total_crayons : ℕ := 15
def red_crayons : ℕ := 3
def non_red_crayons : ℕ := total_crayons - red_crayons
def selection_size : ℕ := 5

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem crayon_selection_theorem :
  choose total_crayons selection_size - choose non_red_crayons selection_size = 2211 :=
by sorry

end NUMINAMATH_CALUDE_crayon_selection_theorem_l875_87560


namespace NUMINAMATH_CALUDE_least_value_ba_l875_87510

/-- Given a number in the form 11,0ab that is divisible by 115, 
    the least possible value of b × a is 0 -/
theorem least_value_ba (a b : ℕ) : 
  a < 10 → b < 10 → (11000 + 100 * a + b) % 115 = 0 → 
  ∀ (c d : ℕ), c < 10 → d < 10 → (11000 + 100 * c + d) % 115 = 0 → 
  b * a ≤ d * c := by
  sorry

end NUMINAMATH_CALUDE_least_value_ba_l875_87510


namespace NUMINAMATH_CALUDE_ellipse_equation_l875_87580

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (e : ℝ) (P : ℝ × ℝ) : 
  e = Real.sqrt 5 / 5 →
  P = (-5, 4) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 45 + y^2 / 36 = 1) :=
by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l875_87580


namespace NUMINAMATH_CALUDE_equation_roots_and_expression_l875_87567

open Real

theorem equation_roots_and_expression (α m : ℝ) : 
  0 < α → α < π →
  (∃ x : ℝ, x^2 + 4 * x * sin (α/2) + m * tan (α/2) = 0 ∧ 
   ∀ y : ℝ, y^2 + 4 * y * sin (α/2) + m * tan (α/2) = 0 → y = x) →
  m + 2 * cos α = 4/3 →
  (0 < m ∧ m ≤ 2) ∧ 
  (1 + sin (2*α) - cos (2*α)) / (1 + tan α) = -5/9 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_and_expression_l875_87567


namespace NUMINAMATH_CALUDE_largest_prime_in_equation_l875_87533

theorem largest_prime_in_equation (x : ℤ) (n : ℕ) (p : ℕ) 
  (hp : Nat.Prime p) (heq : 7 * x^2 - 44 * x + 12 = p^n) :
  p ≤ 47 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_in_equation_l875_87533


namespace NUMINAMATH_CALUDE_systematic_sampling_problem_l875_87502

/-- Systematic sampling function -/
def systematic_sampling (population : ℕ) (sample_size : ℕ) : 
  (ℕ × ℕ × ℕ) :=
  let remaining := population % sample_size
  let eliminated := remaining
  let segment_size := (population - eliminated) / sample_size
  (eliminated, sample_size, segment_size)

/-- Theorem for the given systematic sampling problem -/
theorem systematic_sampling_problem :
  systematic_sampling 1650 35 = (5, 35, 47) := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_problem_l875_87502


namespace NUMINAMATH_CALUDE_max_bar_weight_example_l875_87550

/-- Calculates the maximum weight that can be put on a weight bench bar given the bench's maximum support weight, safety margin percentage, and the weights of two people using the bench. -/
def maxBarWeight (benchMax : ℝ) (safetyMargin : ℝ) (weight1 : ℝ) (weight2 : ℝ) : ℝ :=
  benchMax * (1 - safetyMargin) - (weight1 + weight2)

/-- Theorem stating that for a 1000-pound bench with 20% safety margin and two people weighing 250 and 180 pounds, the maximum weight on the bar is 370 pounds. -/
theorem max_bar_weight_example :
  maxBarWeight 1000 0.2 250 180 = 370 := by
  sorry

end NUMINAMATH_CALUDE_max_bar_weight_example_l875_87550


namespace NUMINAMATH_CALUDE_reflection_across_y_axis_l875_87583

/-- Given a point A with coordinates (-4, 8, 6), 
    prove that its reflection across the y-axis has coordinates (4, 8, 6) -/
theorem reflection_across_y_axis :
  let A : ℝ × ℝ × ℝ := (-4, 8, 6)
  let reflection : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ := fun (x, y, z) ↦ (-x, y, z)
  reflection A = (4, 8, 6) := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_y_axis_l875_87583


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l875_87536

-- Define a function to convert a base 7 number to base 10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

-- Define the given base 7 number
def base7Number : List Nat := [1, 2, 3, 5, 4]

-- Theorem to prove
theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 11481 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l875_87536


namespace NUMINAMATH_CALUDE_noodles_and_pirates_total_l875_87568

theorem noodles_and_pirates_total (pirates : ℕ) (noodles : ℕ) : 
  pirates = 45 → noodles = pirates - 7 → noodles + pirates = 83 :=
by sorry

end NUMINAMATH_CALUDE_noodles_and_pirates_total_l875_87568


namespace NUMINAMATH_CALUDE_range_of_a_l875_87587

/-- A linear function y = mx + b where m = -3a + 1 and b = a -/
def linear_function (a : ℝ) (x : ℝ) : ℝ := (-3 * a + 1) * x + a

/-- Condition that the function is increasing -/
def is_increasing (a : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > x₂ → linear_function a x₁ > linear_function a x₂

/-- Condition that the graph does not pass through the fourth quadrant -/
def not_in_fourth_quadrant (a : ℝ) : Prop :=
  ∀ x y, linear_function a x = y → (x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≤ 0)

/-- The main theorem stating the range of a -/
theorem range_of_a (a : ℝ) 
  (h1 : is_increasing a) 
  (h2 : not_in_fourth_quadrant a) : 
  0 ≤ a ∧ a < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l875_87587


namespace NUMINAMATH_CALUDE_noah_in_middle_chair_l875_87553

/- Define the friends as an enumeration -/
inductive Friend
| Liam
| Noah
| Olivia
| Emma
| Sophia

/- Define the seating arrangement as a function from chair number to Friend -/
def Seating := Fin 5 → Friend

def is_valid_seating (s : Seating) : Prop :=
  /- Sophia sits in the first chair -/
  s 1 = Friend.Sophia ∧
  /- Emma sits directly in front of Liam -/
  (∃ i : Fin 4, s i = Friend.Emma ∧ s (i + 1) = Friend.Liam) ∧
  /- Noah sits somewhere in front of Emma -/
  (∃ i j : Fin 5, i < j ∧ s i = Friend.Noah ∧ s j = Friend.Emma) ∧
  /- At least one person sits between Noah and Olivia -/
  (∃ i j k : Fin 5, i < j ∧ j < k ∧ s i = Friend.Noah ∧ s k = Friend.Olivia) ∧
  /- All friends are seated -/
  (∃ i : Fin 5, s i = Friend.Liam) ∧
  (∃ i : Fin 5, s i = Friend.Noah) ∧
  (∃ i : Fin 5, s i = Friend.Olivia) ∧
  (∃ i : Fin 5, s i = Friend.Emma) ∧
  (∃ i : Fin 5, s i = Friend.Sophia)

theorem noah_in_middle_chair (s : Seating) (h : is_valid_seating s) :
  s 3 = Friend.Noah :=
by sorry

end NUMINAMATH_CALUDE_noah_in_middle_chair_l875_87553


namespace NUMINAMATH_CALUDE_log_system_solution_l875_87508

theorem log_system_solution :
  ∀ x y : ℝ, x > 0 → y > 0 →
  (Real.log x / Real.log 4 - Real.log y / Real.log 2 = 0) →
  (x^2 - 5*y^2 + 4 = 0) →
  ((x = 1 ∧ y = 1) ∨ (x = 4 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_log_system_solution_l875_87508


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l875_87574

/-- The cost ratio of a muffin to a banana given purchase information -/
theorem muffin_banana_cost_ratio :
  ∀ (m b : ℝ), 
    m > 0 →  -- m is positive (cost of muffin)
    b > 0 →  -- b is positive (cost of banana)
    5 * m + 2 * b > 0 →  -- Susie's purchase is positive
    3 * (5 * m + 2 * b) = 4 * m + 10 * b →  -- Jason's purchase is 3 times Susie's
    m / b = 4 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l875_87574


namespace NUMINAMATH_CALUDE_festival_attendance_l875_87586

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h1 : total_students = 1500)
  (h2 : festival_attendees = 900)
  (h3 : ∃ (girls boys : ℕ), 
    girls + boys = total_students ∧ 
    (2 * girls) / 3 + boys / 2 = festival_attendees) :
  ∃ (girls : ℕ), (2 * girls) / 3 = 600 := by
  sorry

end NUMINAMATH_CALUDE_festival_attendance_l875_87586


namespace NUMINAMATH_CALUDE_rhombus_construction_exists_l875_87546

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry

-- Define a rhombus
structure Rhombus where
  vertices : Fin 4 → ℝ × ℝ
  is_rhombus : sorry

-- Define the property of sides being parallel to diagonals
def parallel_to_diagonals (r : Rhombus) (q : ConvexQuadrilateral) : Prop :=
  sorry

-- Define the property of vertices lying on the sides of the quadrilateral
def vertices_on_sides (r : Rhombus) (q : ConvexQuadrilateral) : Prop :=
  sorry

theorem rhombus_construction_exists (q : ConvexQuadrilateral) :
  ∃ (r : Rhombus), vertices_on_sides r q ∧ parallel_to_diagonals r q :=
sorry

end NUMINAMATH_CALUDE_rhombus_construction_exists_l875_87546


namespace NUMINAMATH_CALUDE_distinct_collections_l875_87557

/-- Represents the count of each letter in MATHEMATICSH -/
def letter_count : Finset (Char × ℕ) :=
  {('A', 3), ('E', 1), ('I', 1), ('T', 2), ('M', 2), ('H', 2), ('C', 1), ('S', 1)}

/-- The set of vowels in MATHEMATICSH -/
def vowels : Finset Char := {'A', 'E', 'I'}

/-- The set of consonants in MATHEMATICSH -/
def consonants : Finset Char := {'T', 'M', 'H', 'C', 'S'}

/-- The number of distinct vowel combinations -/
def vowel_combinations : ℕ := 5

/-- The number of distinct consonant combinations -/
def consonant_combinations : ℕ := 48

/-- Theorem stating the number of distinct possible collections -/
theorem distinct_collections :
  vowel_combinations * consonant_combinations = 240 :=
by sorry

end NUMINAMATH_CALUDE_distinct_collections_l875_87557


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l875_87544

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the number --/
def base7Number : List Nat := [0, 1, 2, 3, 4]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 10738 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l875_87544


namespace NUMINAMATH_CALUDE_chord_length_in_unit_circle_l875_87576

theorem chord_length_in_unit_circle (chord1 chord2 chord3 : Real) : 
  -- Unit circle condition
  ∀ (r : Real), r = 1 →
  -- Three distinct diameters
  ∃ (α θ : Real), α ≠ θ ∧ α + θ + (180 - α - θ) = 180 →
  -- One chord has length √2
  chord1 = Real.sqrt 2 →
  -- The other two chords have equal lengths
  chord2 = chord3 →
  -- Length of chord2 and chord3
  chord2 = Real.sqrt (2 - Real.sqrt 2) := by
sorry


end NUMINAMATH_CALUDE_chord_length_in_unit_circle_l875_87576


namespace NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_lt_zero_l875_87592

theorem solution_set_x_squared_minus_one_lt_zero :
  Set.Ioo (-1 : ℝ) 1 = {x : ℝ | x^2 - 1 < 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_lt_zero_l875_87592


namespace NUMINAMATH_CALUDE_acid_dilution_l875_87518

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 50 → 
  initial_concentration = 0.4 → 
  final_concentration = 0.25 → 
  water_added = 30 → 
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration := by
  sorry

#check acid_dilution

end NUMINAMATH_CALUDE_acid_dilution_l875_87518


namespace NUMINAMATH_CALUDE_rosie_pies_l875_87522

/-- Represents the number of pies Rosie can make given the number of apples and pears -/
def total_pies (apples : ℕ) (pears : ℕ) : ℕ :=
  let apple_pies := (apples / 9) * 2
  let pear_pies := (pears / 15) * 3
  apple_pies + pear_pies

/-- Theorem stating that Rosie can make 12 pies with 27 apples and 30 pears -/
theorem rosie_pies : total_pies 27 30 = 12 := by
  sorry

end NUMINAMATH_CALUDE_rosie_pies_l875_87522


namespace NUMINAMATH_CALUDE_min_three_digit_divisible_by_seven_l875_87507

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def remove_middle_digit (n : ℕ) : ℕ :=
  (n / 100) * 10 + (n % 10)

theorem min_three_digit_divisible_by_seven :
  ∃ (N : ℕ),
    is_three_digit N ∧
    N % 7 = 0 ∧
    (remove_middle_digit N) % 7 = 0 ∧
    (∀ (M : ℕ), 
      is_three_digit M ∧ 
      M % 7 = 0 ∧ 
      (remove_middle_digit M) % 7 = 0 → 
      N ≤ M) ∧
    N = 154 := by
  sorry

end NUMINAMATH_CALUDE_min_three_digit_divisible_by_seven_l875_87507


namespace NUMINAMATH_CALUDE_opposite_blue_is_black_l875_87578

-- Define the colors
inductive Color
| Blue | Yellow | Orange | Black | Silver | Gold

-- Define a cube
structure Cube where
  faces : Fin 6 → Color

-- Define the views
structure View where
  top : Color
  front : Color
  right : Color

-- Define the problem setup
def cube_problem (c : Cube) (v1 v2 v3 : View) : Prop :=
  -- All faces have different colors
  (∀ i j : Fin 6, i ≠ j → c.faces i ≠ c.faces j) ∧
  -- The views are consistent with the cube
  (v1.top = Color.Gold ∧ v1.front = Color.Black ∧ v1.right = Color.Orange) ∧
  (v2.top = Color.Gold ∧ v2.front = Color.Yellow ∧ v2.right = Color.Orange) ∧
  (v3.top = Color.Gold ∧ v3.front = Color.Silver ∧ v3.right = Color.Orange)

-- The theorem to prove
theorem opposite_blue_is_black (c : Cube) (v1 v2 v3 : View) 
  (h : cube_problem c v1 v2 v3) : 
  ∃ (i j : Fin 6), c.faces i = Color.Blue ∧ c.faces j = Color.Black ∧ 
  (i.val + j.val = 5 ∨ i.val + j.val = 7) :=
sorry

end NUMINAMATH_CALUDE_opposite_blue_is_black_l875_87578


namespace NUMINAMATH_CALUDE_division_problem_l875_87558

theorem division_problem (a b c : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2/5) : 
  c / a = 5/6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l875_87558


namespace NUMINAMATH_CALUDE_bubble_pass_probability_specific_l875_87555

def bubble_pass_probability (n : ℕ) (initial_pos : ℕ) (final_pos : ℕ) : ℚ :=
  if initial_pos < final_pos ∧ final_pos ≤ n then
    1 / (initial_pos * (final_pos - 1))
  else
    0

theorem bubble_pass_probability_specific :
  bubble_pass_probability 50 25 35 = 1 / 850 := by
  sorry

end NUMINAMATH_CALUDE_bubble_pass_probability_specific_l875_87555


namespace NUMINAMATH_CALUDE_constant_sum_of_powers_l875_87565

/-- S_n is constant for real x, y, z with xyz = 1 and x + y + z = 0 iff n = 1 or n = 3 -/
theorem constant_sum_of_powers (n : ℕ+) :
  (∀ x y z : ℝ, x * y * z = 1 → x + y + z = 0 → 
    ∃ c : ℝ, ∀ x' y' z' : ℝ, x' * y' * z' = 1 → x' + y' + z' = 0 → 
      x'^(n : ℕ) + y'^(n : ℕ) + z'^(n : ℕ) = c) ↔ 
  n = 1 ∨ n = 3 := by
sorry

end NUMINAMATH_CALUDE_constant_sum_of_powers_l875_87565


namespace NUMINAMATH_CALUDE_rectangle_height_twice_square_side_l875_87563

/-- Given a square with side length s and a rectangle with base s and area twice that of the square,
    prove that the height of the rectangle is 2s. -/
theorem rectangle_height_twice_square_side (s : ℝ) (h : s > 0) : 
  let square_area := s^2
  let rectangle_base := s
  let rectangle_area := 2 * square_area
  rectangle_area / rectangle_base = 2 * s := by
  sorry

end NUMINAMATH_CALUDE_rectangle_height_twice_square_side_l875_87563


namespace NUMINAMATH_CALUDE_min_value_expression_l875_87551

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 2/y = 1) :
  ∃ (x₀ y₀ : ℝ), 2*x₀*y₀ - 2*x₀ - y₀ = 8 ∧ ∀ x y, x > 0 → y > 0 → 1/x + 2/y = 1 → 2*x*y - 2*x - y ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l875_87551


namespace NUMINAMATH_CALUDE_rolles_theorem_application_l875_87575

-- Define the function f(x) = x^2 + 2x + 7
def f (x : ℝ) : ℝ := x^2 + 2*x + 7

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 2*x + 2

-- Theorem statement
theorem rolles_theorem_application :
  ∃ c ∈ Set.Ioo (-6 : ℝ) 4, f' c = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_rolles_theorem_application_l875_87575


namespace NUMINAMATH_CALUDE_circle_ratio_l875_87526

theorem circle_ratio (R r : ℝ) (h1 : R > 0) (h2 : r > 0) (h3 : R > r) :
  (π * R^2 - π * r^2) = 3 * (π * r^2) → R = 2 * r := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l875_87526


namespace NUMINAMATH_CALUDE_divisibility_of_quadratic_l875_87559

theorem divisibility_of_quadratic (n : ℤ) : 
  (∀ n, ¬(8 ∣ (n^2 - 6*n - 2))) ∧ 
  (∀ n, ¬(9 ∣ (n^2 - 6*n - 2))) ∧ 
  (∀ n, (11 ∣ (n^2 - 6*n - 2)) ↔ (n ≡ 3 [ZMOD 11])) ∧ 
  (∀ n, ¬(121 ∣ (n^2 - 6*n - 2))) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_quadratic_l875_87559


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l875_87509

/-- The line mx-y+2m+1=0 passes through the point (-2, 1) for all values of m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), m * (-2) - 1 + 2 * m + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l875_87509


namespace NUMINAMATH_CALUDE_red_peppers_weight_l875_87529

/-- The weight of red peppers at Dale's Vegetarian Restaurant -/
def weight_red_peppers : ℝ := 5.666666666666667 - 2.8333333333333335

/-- Theorem: The weight of red peppers is equal to the total weight of peppers minus the weight of green peppers -/
theorem red_peppers_weight :
  weight_red_peppers = 5.666666666666667 - 2.8333333333333335 := by
  sorry

end NUMINAMATH_CALUDE_red_peppers_weight_l875_87529


namespace NUMINAMATH_CALUDE_hypotenuse_length_l875_87577

/-- A point on the parabola y = -x^2 --/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y = -x^2

/-- Triangle ABO with A and B on the parabola y = -x^2 and ∠AOB = 45° --/
structure TriangleABO where
  A : ParabolaPoint
  B : ParabolaPoint
  angle_AOB : Real.pi / 4 = Real.arctan (A.y / A.x) + Real.arctan (B.y / B.x)

/-- The length of the hypotenuse of triangle ABO is 2 --/
theorem hypotenuse_length (t : TriangleABO) : 
  Real.sqrt ((t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l875_87577


namespace NUMINAMATH_CALUDE_inequality_range_l875_87556

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2*x - 1| + |x + 1| > a) → a < (3/2) := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l875_87556


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l875_87579

/-- Two vectors are parallel if their corresponding components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ (m : ℝ),
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (m, -1)
  are_parallel a b → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l875_87579


namespace NUMINAMATH_CALUDE_system_solution_unique_l875_87562

theorem system_solution_unique :
  ∃! (x y : ℚ), (3 * (x - 1) = y + 6) ∧ (x / 2 + y / 3 = 2) ∧ (x = 10 / 3) ∧ (y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l875_87562


namespace NUMINAMATH_CALUDE_pizza_fraction_l875_87542

theorem pizza_fraction (total_slices : ℕ) (whole_slices : ℕ) (shared_slice_fraction : ℚ) :
  total_slices = 16 →
  whole_slices = 2 →
  shared_slice_fraction = 1/6 →
  (whole_slices : ℚ) / total_slices + shared_slice_fraction / total_slices = 13/96 := by
  sorry

end NUMINAMATH_CALUDE_pizza_fraction_l875_87542


namespace NUMINAMATH_CALUDE_f_value_at_3_l875_87570

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^3 - b * x + 2

-- State the theorem
theorem f_value_at_3 (a b : ℝ) (h : f a b (-3) = -1) : f a b 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_3_l875_87570


namespace NUMINAMATH_CALUDE_identity_is_unique_solution_l875_87595

/-- The set of positive integers -/
def PositiveIntegers := {n : ℕ | n > 0}

/-- A function from positive integers to positive integers -/
def PositiveIntegerFunction := PositiveIntegers → PositiveIntegers

/-- The functional equation that f must satisfy -/
def SatisfiesEquation (f : PositiveIntegerFunction) : Prop :=
  ∀ m n : PositiveIntegers,
    Nat.gcd (f m) n + Nat.lcm m (f n) = Nat.gcd m (f n) + Nat.lcm (f m) n

/-- The theorem stating that the identity function is the only solution -/
theorem identity_is_unique_solution :
  ∃! f : PositiveIntegerFunction, SatisfiesEquation f ∧ (∀ n, f n = n) :=
sorry

end NUMINAMATH_CALUDE_identity_is_unique_solution_l875_87595


namespace NUMINAMATH_CALUDE_appropriate_mass_units_l875_87517

-- Define the mass units
inductive MassUnit
| Gram
| Ton
| Kilogram

-- Define a structure for an item with its mass value
structure MassItem where
  value : ℕ
  unit : MassUnit

-- Define the function to check if a mass unit is appropriate for a given item
def isAppropriateUnit (item : MassItem) : Prop :=
  match item with
  | ⟨1, MassUnit.Gram⟩ => true     -- Peanut kernel
  | ⟨8, MassUnit.Ton⟩ => true      -- Truck loading capacity
  | ⟨30, MassUnit.Kilogram⟩ => true -- Xiao Ming's weight
  | ⟨580, MassUnit.Gram⟩ => true   -- Basketball mass
  | _ => false

-- Theorem statement
theorem appropriate_mass_units :
  let peanut := MassItem.mk 1 MassUnit.Gram
  let truck := MassItem.mk 8 MassUnit.Ton
  let xiaoMing := MassItem.mk 30 MassUnit.Kilogram
  let basketball := MassItem.mk 580 MassUnit.Gram
  isAppropriateUnit peanut ∧
  isAppropriateUnit truck ∧
  isAppropriateUnit xiaoMing ∧
  isAppropriateUnit basketball :=
by sorry


end NUMINAMATH_CALUDE_appropriate_mass_units_l875_87517


namespace NUMINAMATH_CALUDE_four_tangent_circles_l875_87543

-- Define a line in 2D space
def Line2D := (ℝ × ℝ) → Prop

-- Define a circle in 2D space
structure Circle2D where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency between a circle and a line
def isTangent (c : Circle2D) (l : Line2D) : Prop := sorry

-- Main theorem
theorem four_tangent_circles 
  (l1 l2 : Line2D) 
  (intersect : ∃ p : ℝ × ℝ, l1 p ∧ l2 p) 
  (r : ℝ) 
  (h : r > 0) : 
  ∃! (cs : Finset Circle2D), 
    cs.card = 4 ∧ 
    (∀ c ∈ cs, c.radius = r ∧ isTangent c l1 ∧ isTangent c l2) :=
sorry

end NUMINAMATH_CALUDE_four_tangent_circles_l875_87543


namespace NUMINAMATH_CALUDE_basketball_rim_height_l875_87512

/-- Represents the height of a basketball rim above the ground -/
def rim_height : ℕ := sorry

/-- Represents the player's height in feet -/
def player_height_feet : ℕ := 6

/-- Represents the player's reach above their head in inches -/
def player_reach : ℕ := 22

/-- Represents the player's jump height in inches -/
def player_jump : ℕ := 32

/-- Represents how far above the rim the player can reach when jumping, in inches -/
def above_rim : ℕ := 6

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

theorem basketball_rim_height : 
  rim_height = player_height_feet * feet_to_inches + player_reach + player_jump - above_rim :=
by sorry

end NUMINAMATH_CALUDE_basketball_rim_height_l875_87512


namespace NUMINAMATH_CALUDE_inequality_solution_l875_87516

theorem inequality_solution (x : ℝ) : 
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 2) ↔ 
  (x < -2 ∨ (-1 < x ∧ x < 0) ∨ 1 < x) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l875_87516


namespace NUMINAMATH_CALUDE_games_lost_calculation_l875_87523

def total_games : ℕ := 12
def games_won : ℕ := 8

theorem games_lost_calculation : total_games - games_won = 4 := by
  sorry

end NUMINAMATH_CALUDE_games_lost_calculation_l875_87523


namespace NUMINAMATH_CALUDE_birch_tree_spacing_probability_l875_87569

def total_trees : ℕ := 15
def pine_trees : ℕ := 4
def maple_trees : ℕ := 5
def birch_trees : ℕ := 6

theorem birch_tree_spacing_probability :
  let non_birch_trees := pine_trees + maple_trees
  let total_arrangements := (total_trees.choose birch_trees : ℚ)
  let valid_arrangements := ((non_birch_trees + 1).choose birch_trees : ℚ)
  valid_arrangements / total_arrangements = 2 / 95 := by
sorry

end NUMINAMATH_CALUDE_birch_tree_spacing_probability_l875_87569
