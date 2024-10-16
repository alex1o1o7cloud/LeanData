import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3585_358553

def a : ℝ × ℝ := (2, 1)

def b (k : ℝ) : ℝ × ℝ := (1 - 2, k - 1)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem perpendicular_vectors_k_value :
  ∀ k : ℝ, perpendicular a (b k) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l3585_358553


namespace NUMINAMATH_CALUDE_problem_statement_l3585_358583

theorem problem_statement (h : 125 = 5^3) : (125 : ℝ)^(2/3) * 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3585_358583


namespace NUMINAMATH_CALUDE_triangle_properties_l3585_358586

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sinA : ℝ
  sinB : ℝ
  sinC : ℝ
  cosC : ℝ

def is_valid_triangle (t : Triangle) : Prop :=
  t.sinA > 0 ∧ t.sinB > 0 ∧ t.sinC > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

theorem triangle_properties (t : Triangle) 
  (h_valid : is_valid_triangle t)
  (h_arith_seq : 2 * t.sinB = t.sinA + t.sinC)
  (h_cosC : t.cosC = 1/3) :
  (t.b / t.a = 10/9) ∧ 
  (t.c = 11 → t.a * t.b * t.sinC / 2 = 30 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3585_358586


namespace NUMINAMATH_CALUDE_trapezoid_gh_length_l3585_358577

/-- Represents a trapezoid with given dimensions -/
structure Trapezoid where
  area : ℝ
  altitude : ℝ
  side_ef : ℝ
  side_gh : ℝ

/-- The area of a trapezoid is equal to the average of its parallel sides multiplied by its altitude -/
axiom trapezoid_area (t : Trapezoid) : t.area = (t.side_ef + t.side_gh) / 2 * t.altitude

theorem trapezoid_gh_length (t : Trapezoid) 
    (h_area : t.area = 250)
    (h_altitude : t.altitude = 10)
    (h_ef : t.side_ef = 15) :
    t.side_gh = 35 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_gh_length_l3585_358577


namespace NUMINAMATH_CALUDE_tan_value_for_given_point_l3585_358540

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_value_for_given_point (θ : Real) :
  (∃ (t : Real), t > 0 ∧ t * (-Real.sqrt 3 / 2) = Real.cos θ ∧ t * (1 / 2) = Real.sin θ) →
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_for_given_point_l3585_358540


namespace NUMINAMATH_CALUDE_sqrt_five_is_quadratic_radical_l3585_358593

/-- A number is non-negative if it's greater than or equal to zero. -/
def NonNegative (x : ℝ) : Prop := x ≥ 0

/-- A quadratic radical is an expression √x where x is non-negative. -/
def QuadraticRadical (x : ℝ) : Prop := NonNegative x

/-- Theorem: √5 is a quadratic radical. -/
theorem sqrt_five_is_quadratic_radical : QuadraticRadical 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_is_quadratic_radical_l3585_358593


namespace NUMINAMATH_CALUDE_complex_root_cubic_equation_l3585_358506

theorem complex_root_cubic_equation 
  (a b q r : ℝ) 
  (h_b : b ≠ 0) 
  (h_root : ∃ (z : ℂ), z^3 + q * z + r = 0 ∧ z = a + b * Complex.I) :
  q = b^2 - 3 * a^2 := by
sorry

end NUMINAMATH_CALUDE_complex_root_cubic_equation_l3585_358506


namespace NUMINAMATH_CALUDE_absolute_value_five_l3585_358515

theorem absolute_value_five (x : ℝ) : |x| = 5 ↔ x = 5 ∨ x = -5 := by sorry

end NUMINAMATH_CALUDE_absolute_value_five_l3585_358515


namespace NUMINAMATH_CALUDE_michelles_necklace_l3585_358508

/-- Problem: Michelle's Necklace Beads --/
theorem michelles_necklace (total_beads : ℕ) (blue_beads : ℕ) : 
  total_beads = 200 →
  blue_beads = 12 →
  let red_beads := 3 * blue_beads
  let white_beads := (3/2 : ℚ) * (blue_beads + red_beads)
  let green_beads := (1/2 : ℚ) * (blue_beads + red_beads + white_beads)
  let colored_beads := blue_beads + red_beads + white_beads + green_beads
  total_beads - colored_beads = 20 := by
  sorry


end NUMINAMATH_CALUDE_michelles_necklace_l3585_358508


namespace NUMINAMATH_CALUDE_liquid_rise_ratio_l3585_358545

/-- Represents a right circular cone filled with liquid -/
structure LiquidCone where
  radius : ℝ
  height : ℝ
  volume : ℝ

/-- Represents the scenario with two cones and a marble -/
structure TwoConesScenario where
  narrow_cone : LiquidCone
  wide_cone : LiquidCone
  marble_radius : ℝ

/-- The rise of liquid level in a cone after dropping the marble -/
def liquid_rise (cone : LiquidCone) (marble_volume : ℝ) : ℝ :=
  sorry

theorem liquid_rise_ratio (scenario : TwoConesScenario) :
  scenario.narrow_cone.radius = 4 ∧
  scenario.wide_cone.radius = 8 ∧
  scenario.narrow_cone.volume = scenario.wide_cone.volume ∧
  scenario.marble_radius = 1.5 →
  let marble_volume := (4/3) * Real.pi * scenario.marble_radius^3
  (liquid_rise scenario.narrow_cone marble_volume) /
  (liquid_rise scenario.wide_cone marble_volume) = 4 := by
  sorry

end NUMINAMATH_CALUDE_liquid_rise_ratio_l3585_358545


namespace NUMINAMATH_CALUDE_inscribed_prism_surface_area_l3585_358525

/-- The surface area of a right square prism inscribed in a sphere -/
theorem inscribed_prism_surface_area (r h : ℝ) (a : ℝ) :
  r = Real.sqrt 6 →
  h = 4 →
  2 * a^2 + h^2 = 4 * r^2 →
  2 * a^2 + 4 * a * h = 40 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_prism_surface_area_l3585_358525


namespace NUMINAMATH_CALUDE_nursery_seedling_price_l3585_358534

theorem nursery_seedling_price :
  ∀ (price_day2 : ℝ),
    (price_day2 > 0) →
    (2 * (8000 / (price_day2 - 5)) = 17000 / price_day2) →
    price_day2 = 85 := by
  sorry

end NUMINAMATH_CALUDE_nursery_seedling_price_l3585_358534


namespace NUMINAMATH_CALUDE_triangle_area_and_fixed_point_l3585_358504

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the area of the triangle
def triangle_area : ℝ := 8

-- Define the family of lines
def family_of_lines (m x y : ℝ) : Prop := m * x + y + m = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-1, 0)

theorem triangle_area_and_fixed_point :
  (∀ x y, line_equation x y → 
    (x = 0 ∨ y = 0) → triangle_area = 8) ∧
  (∀ m x y, family_of_lines m x y → 
    (x, y) = fixed_point) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_and_fixed_point_l3585_358504


namespace NUMINAMATH_CALUDE_chris_money_before_birthday_l3585_358578

/-- Represents the amount of money Chris had before his birthday. -/
def money_before_birthday : ℕ := sorry

/-- The amount Chris received from his grandmother. -/
def grandmother_gift : ℕ := 25

/-- The amount Chris received from his aunt and uncle. -/
def aunt_uncle_gift : ℕ := 20

/-- The amount Chris received from his parents. -/
def parents_gift : ℕ := 75

/-- The total amount Chris has after receiving all gifts. -/
def total_after_gifts : ℕ := 279

/-- Theorem stating that Chris had $159 before his birthday. -/
theorem chris_money_before_birthday :
  money_before_birthday = total_after_gifts - (grandmother_gift + aunt_uncle_gift + parents_gift) :=
by sorry

end NUMINAMATH_CALUDE_chris_money_before_birthday_l3585_358578


namespace NUMINAMATH_CALUDE_always_positive_product_l3585_358554

theorem always_positive_product (a b c : ℝ) (h : a > b ∧ b > c) : (a - b) * |c - b| > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_product_l3585_358554


namespace NUMINAMATH_CALUDE_triangle_square_sum_l3585_358589

theorem triangle_square_sum (triangle square : ℝ) 
  (h1 : 3 + triangle = 5) 
  (h2 : triangle + square = 7) : 
  triangle + triangle + triangle + square + square = 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_square_sum_l3585_358589


namespace NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3585_358522

theorem fourth_root_equation_solutions :
  ∀ x : ℝ, (x ^ (1/4) = 15 / (8 - x ^ (1/4))) ↔ (x = 625 ∨ x = 81) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_equation_solutions_l3585_358522


namespace NUMINAMATH_CALUDE_ben_mm_count_l3585_358510

theorem ben_mm_count (bryan_skittles : ℕ) (difference : ℕ) (ben_mm : ℕ) : 
  bryan_skittles = 50 →
  difference = 30 →
  bryan_skittles = ben_mm + difference →
  ben_mm = 20 := by
sorry

end NUMINAMATH_CALUDE_ben_mm_count_l3585_358510


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_specific_root_condition_l3585_358513

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + 4*x + m - 1
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) → m ≤ 5 :=
by sorry

theorem specific_root_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + 4*x + m - 1
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 2*(x₁ + x₂) + x₁*x₂ + 10 = 0) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_specific_root_condition_l3585_358513


namespace NUMINAMATH_CALUDE_time_reduction_percentage_l3585_358581

/-- Calculates the time reduction percentage when increasing speed from 60 km/h to 86 km/h for a journey that initially takes 30 minutes. -/
theorem time_reduction_percentage 
  (initial_speed : ℝ) 
  (initial_time : ℝ) 
  (new_speed : ℝ) 
  (h1 : initial_speed = 60) 
  (h2 : initial_time = 30) 
  (h3 : new_speed = 86) : 
  ∃ (reduction_percentage : ℝ), 
    (abs (reduction_percentage - 30.23) < 0.01) ∧ 
    (reduction_percentage = (1 - (initial_speed * initial_time) / (new_speed * initial_time)) * 100) :=
by sorry

end NUMINAMATH_CALUDE_time_reduction_percentage_l3585_358581


namespace NUMINAMATH_CALUDE_sticker_distribution_theorem_l3585_358576

/-- Number of stickers -/
def n : ℕ := 10

/-- Number of sheets -/
def k : ℕ := 5

/-- Number of color options for each sheet -/
def c : ℕ := 2

/-- The number of ways to distribute n identical objects into k distinct boxes -/
def ways_to_distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The total number of distinct arrangements considering both sticker counts and colors -/
def total_arrangements (n k c : ℕ) : ℕ := (ways_to_distribute n k) * (c^k)

/-- The main theorem stating the total number of distinct arrangements -/
theorem sticker_distribution_theorem : total_arrangements n k c = 32032 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_theorem_l3585_358576


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3585_358523

theorem greatest_divisor_with_remainders : Nat.gcd (1442 - 12) (1816 - 6) = 10 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3585_358523


namespace NUMINAMATH_CALUDE_a_over_b_equals_half_l3585_358595

theorem a_over_b_equals_half (a b : ℤ) (h : a + Real.sqrt b = Real.sqrt (15 + Real.sqrt 216)) : a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_a_over_b_equals_half_l3585_358595


namespace NUMINAMATH_CALUDE_triangle_inequality_l3585_358587

theorem triangle_inequality (a b c : ℝ) : 
  (0 < a ∧ 0 < b ∧ 0 < c) → (a + b > c ∧ b + c > a ∧ c + a > b) → 
  (3 = a ∧ 7 = b) → 4 < c ∧ c < 10 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3585_358587


namespace NUMINAMATH_CALUDE_steves_final_height_l3585_358579

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

/-- Calculates the final height in inches after growth -/
def final_height (initial_feet : ℕ) (initial_inches : ℕ) (growth : ℕ) : ℕ :=
  feet_inches_to_inches initial_feet initial_inches + growth

/-- Theorem: Steve's final height is 72 inches -/
theorem steves_final_height :
  final_height 5 6 6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_steves_final_height_l3585_358579


namespace NUMINAMATH_CALUDE_pencil_distribution_l3585_358537

theorem pencil_distribution (num_children : ℕ) (pencils_per_child : ℕ) (total_pencils : ℕ) : 
  num_children = 4 → 
  pencils_per_child = 2 → 
  total_pencils = num_children * pencils_per_child →
  total_pencils = 8 := by
sorry

end NUMINAMATH_CALUDE_pencil_distribution_l3585_358537


namespace NUMINAMATH_CALUDE_not_divides_power_plus_one_l3585_358568

theorem not_divides_power_plus_one (n : ℕ) (h : n > 1) : ¬(2^n ∣ 3^n + 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_plus_one_l3585_358568


namespace NUMINAMATH_CALUDE_initial_oranges_count_l3585_358562

/-- The number of oranges initially in the basket -/
def initial_oranges : ℕ := sorry

/-- The number of oranges taken from the basket -/
def oranges_taken : ℕ := 5

/-- The number of oranges remaining in the basket -/
def oranges_remaining : ℕ := 3

/-- Theorem stating that the initial number of oranges is 8 -/
theorem initial_oranges_count : initial_oranges = 8 := by sorry

end NUMINAMATH_CALUDE_initial_oranges_count_l3585_358562


namespace NUMINAMATH_CALUDE_circle_radius_l3585_358511

/-- Given a circle and a line passing through its center, prove the radius is 3 -/
theorem circle_radius (m : ℝ) : 
  (∀ x y, x^2 + y^2 - 2*x + m*y - 4 = 0 → (x - 1)^2 + (y + m/2)^2 = 9) ∧ 
  (2 * 1 + (-m/2) = 0) →
  ∃ r, r = 3 ∧ ∀ x y, (x - 1)^2 + (y + m/2)^2 = r^2 → x^2 + y^2 - 2*x + m*y - 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l3585_358511


namespace NUMINAMATH_CALUDE_imaginary_unit_power_l3585_358594

theorem imaginary_unit_power (i : ℂ) : i ^ 2 = -1 → i ^ 2023 = -i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_l3585_358594


namespace NUMINAMATH_CALUDE_no_isosceles_triangle_l3585_358532

/-- The set of stick lengths -/
def stickLengths : Set ℝ :=
  {x : ℝ | ∃ n : ℕ, n < 100 ∧ x = (0.9 : ℝ) ^ n}

/-- Definition of an isosceles triangle formed by three sticks -/
def isIsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ a = c) ∧ a + b > c ∧ b + c > a ∧ a + c > b

/-- Theorem stating the impossibility of forming an isosceles triangle -/
theorem no_isosceles_triangle :
  ¬ ∃ a b c : ℝ, a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧
    isIsoscelesTriangle a b c :=
sorry

end NUMINAMATH_CALUDE_no_isosceles_triangle_l3585_358532


namespace NUMINAMATH_CALUDE_initial_female_percent_calculation_l3585_358546

/-- Represents a company's workforce statistics -/
structure Workforce where
  initial_total : ℕ
  initial_female_percent : ℚ
  hired_male : ℕ
  final_total : ℕ
  final_female_percent : ℚ

/-- Theorem stating the conditions and the result to be proved -/
theorem initial_female_percent_calculation (w : Workforce) 
  (h1 : w.hired_male = 30)
  (h2 : w.final_total = 360)
  (h3 : w.final_female_percent = 55/100)
  (h4 : w.initial_total * w.initial_female_percent = w.final_total * w.final_female_percent) :
  w.initial_female_percent = 60/100 := by
  sorry

end NUMINAMATH_CALUDE_initial_female_percent_calculation_l3585_358546


namespace NUMINAMATH_CALUDE_alligator_coins_l3585_358561

def river_crossing (initial : ℚ) : ℚ := 
  ((((initial * 3 - 30) * 3 - 30) * 3 - 30) * 3 - 30)

theorem alligator_coins : 
  ∃ initial : ℚ, river_crossing initial = 10 ∧ initial = 1210 / 81 := by
sorry

end NUMINAMATH_CALUDE_alligator_coins_l3585_358561


namespace NUMINAMATH_CALUDE_bucket_weight_l3585_358502

/-- Given a bucket with unknown weight and unknown full water weight,
    if the total weight is p when it's three-quarters full and q when it's one-third full,
    then the total weight when it's completely full is (1/5)(8p - 3q). -/
theorem bucket_weight (p q : ℝ) : 
  (∃ (x y : ℝ), x + 3/4 * y = p ∧ x + 1/3 * y = q) → 
  (∃ (x y : ℝ), x + 3/4 * y = p ∧ x + 1/3 * y = q ∧ x + y = 1/5 * (8*p - 3*q)) :=
by sorry

end NUMINAMATH_CALUDE_bucket_weight_l3585_358502


namespace NUMINAMATH_CALUDE_seeds_per_flowerbed_is_four_l3585_358526

/-- The number of flowerbeds -/
def num_flowerbeds : ℕ := 8

/-- The total number of seeds planted -/
def total_seeds : ℕ := 32

/-- The number of seeds in each flowerbed -/
def seeds_per_flowerbed : ℕ := total_seeds / num_flowerbeds

/-- Theorem: The number of seeds per flowerbed is 4 -/
theorem seeds_per_flowerbed_is_four :
  seeds_per_flowerbed = 4 :=
by sorry

end NUMINAMATH_CALUDE_seeds_per_flowerbed_is_four_l3585_358526


namespace NUMINAMATH_CALUDE_domain_f_l3585_358524

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_plus_one : Set ℝ := Set.Icc (-2) 3

-- Theorem statement
theorem domain_f (h : ∀ x, f (x + 1) ∈ domain_f_plus_one ↔ x ∈ Set.Icc (-2) 3) :
  ∀ x, f x ∈ Set.Icc (-3) 2 ↔ x ∈ Set.Icc (-3) 2 :=
sorry

end NUMINAMATH_CALUDE_domain_f_l3585_358524


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l3585_358551

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def is_mersenne_prime (m : ℕ) : Prop :=
  ∃ n : ℕ, is_prime n ∧ m = 2^n - 1 ∧ is_prime m

theorem largest_mersenne_prime_under_1000 :
  ∀ m : ℕ, is_mersenne_prime m → m < 1000 → m ≤ 127 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_1000_l3585_358551


namespace NUMINAMATH_CALUDE_geometric_locus_definition_l3585_358569

-- Define a type for points in a space
variable {Point : Type*}

-- Define a predicate for the condition that points must satisfy
variable (condition : Point → Prop)

-- Define a predicate for points being on the locus
variable (on_locus : Point → Prop)

-- Statement A
def statement_A : Prop :=
  (∀ p, on_locus p → condition p) ∧ (∀ p, condition p → on_locus p)

-- Statement B
def statement_B : Prop :=
  (∀ p, ¬condition p → ¬on_locus p) ∧ ¬(∀ p, condition p → on_locus p)

-- Statement C
def statement_C : Prop :=
  ∀ p, on_locus p ↔ condition p

-- Statement D
def statement_D : Prop :=
  (∀ p, ¬on_locus p → ¬condition p) ∧ (∀ p, condition p → on_locus p)

-- Statement E
def statement_E : Prop :=
  (∀ p, on_locus p → condition p) ∧ ¬(∀ p, condition p → on_locus p)

theorem geometric_locus_definition :
  (statement_A condition on_locus ∧ 
   statement_C condition on_locus ∧ 
   statement_D condition on_locus) ∧
  (¬statement_B condition on_locus ∧ 
   ¬statement_E condition on_locus) :=
sorry

end NUMINAMATH_CALUDE_geometric_locus_definition_l3585_358569


namespace NUMINAMATH_CALUDE_square_root_computation_l3585_358567

theorem square_root_computation : (3 * Real.sqrt 15625 - 5)^2 = 136900 := by
  sorry

end NUMINAMATH_CALUDE_square_root_computation_l3585_358567


namespace NUMINAMATH_CALUDE_existence_of_symmetry_axes_l3585_358527

/-- A bounded planar figure. -/
structure BoundedPlanarFigure where
  -- Define properties of a bounded planar figure
  is_bounded : Bool
  is_planar : Bool

/-- An axis of symmetry for a bounded planar figure. -/
structure AxisOfSymmetry (F : BoundedPlanarFigure) where
  -- Define properties of an axis of symmetry

/-- The number of axes of symmetry for a bounded planar figure. -/
def num_axes_of_symmetry (F : BoundedPlanarFigure) : Nat :=
  sorry

/-- Theorem: There exist bounded planar figures with exactly three axes of symmetry,
    and there exist bounded planar figures with more than three axes of symmetry. -/
theorem existence_of_symmetry_axes :
  (∃ F : BoundedPlanarFigure, F.is_bounded ∧ F.is_planar ∧ num_axes_of_symmetry F = 3) ∧
  (∃ G : BoundedPlanarFigure, G.is_bounded ∧ G.is_planar ∧ num_axes_of_symmetry G > 3) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_symmetry_axes_l3585_358527


namespace NUMINAMATH_CALUDE_distribute_balls_count_l3585_358585

/-- The number of ways to distribute 6 balls into 3 boxes -/
def distribute_balls : ℕ :=
  3 * (Nat.choose 4 2)

/-- Theorem stating that the number of ways to distribute the balls is 18 -/
theorem distribute_balls_count : distribute_balls = 18 := by
  sorry

end NUMINAMATH_CALUDE_distribute_balls_count_l3585_358585


namespace NUMINAMATH_CALUDE_perfectSquareFactors_360_l3585_358528

/-- A function that returns the number of perfect square factors of a natural number -/
def perfectSquareFactors (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of perfect square factors of 360 is 4 -/
theorem perfectSquareFactors_360 : perfectSquareFactors 360 = 4 := by sorry

end NUMINAMATH_CALUDE_perfectSquareFactors_360_l3585_358528


namespace NUMINAMATH_CALUDE_least_four_digit_solution_l3585_358575

theorem least_four_digit_solution (x : ℕ) : x = 1163 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 15 [ZMOD 20] ∧
     3 * y + 10 ≡ 19 [ZMOD 14] ∧
     -3 * y + 4 ≡ 2 * y [ZMOD 35] ∧
     y + 1 ≡ 0 [ZMOD 11]) →
    x ≤ y) ∧
  (5 * x ≡ 15 [ZMOD 20]) ∧
  (3 * x + 10 ≡ 19 [ZMOD 14]) ∧
  (-3 * x + 4 ≡ 2 * x [ZMOD 35]) ∧
  (x + 1 ≡ 0 [ZMOD 11]) := by
  sorry

end NUMINAMATH_CALUDE_least_four_digit_solution_l3585_358575


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l3585_358573

/-- A right triangle with side lengths 6, 8, and 10 -/
structure RightTriangle :=
  (PQ : ℝ) (QR : ℝ) (PR : ℝ)
  (right_angle : PQ^2 + QR^2 = PR^2)
  (PQ_eq : PQ = 6)
  (QR_eq : QR = 8)
  (PR_eq : PR = 10)

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) :=
  (side_length : ℝ)
  (on_hypotenuse : side_length ≤ t.PR)
  (on_leg1 : side_length ≤ t.PQ)
  (on_leg2 : side_length ≤ t.QR)

/-- The side length of the inscribed square is 3 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l3585_358573


namespace NUMINAMATH_CALUDE_fixed_cost_calculation_l3585_358552

/-- The fixed cost to run the molding machine per week -/
def fixed_cost : ℝ := 7640

/-- The cost to mold each handle -/
def mold_cost : ℝ := 0.60

/-- The selling price per handle -/
def selling_price : ℝ := 4.60

/-- The number of handles needed to break even -/
def break_even_quantity : ℕ := 1910

/-- Theorem stating that the fixed cost is correct given the conditions -/
theorem fixed_cost_calculation :
  fixed_cost = (selling_price - mold_cost) * break_even_quantity := by
  sorry

end NUMINAMATH_CALUDE_fixed_cost_calculation_l3585_358552


namespace NUMINAMATH_CALUDE_output_for_input_12_l3585_358543

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 35 then
    step1 + 10
  else
    step1 - 7

theorem output_for_input_12 :
  function_machine 12 = 29 := by sorry

end NUMINAMATH_CALUDE_output_for_input_12_l3585_358543


namespace NUMINAMATH_CALUDE_B_max_at_45_l3585_358590

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

/-- B_k as defined in the problem -/
def B (k : ℕ) : ℝ := (binomial 500 k : ℝ) * (0.1 ^ k)

/-- Theorem stating that B_k is largest when k = 45 -/
theorem B_max_at_45 : ∀ k : ℕ, k ≤ 500 → B 45 ≥ B k := by sorry

end NUMINAMATH_CALUDE_B_max_at_45_l3585_358590


namespace NUMINAMATH_CALUDE_masha_room_number_l3585_358549

theorem masha_room_number 
  (total_rooms : ℕ) 
  (masha_room : ℕ) 
  (alina_room : ℕ) 
  (h1 : total_rooms = 10000)
  (h2 : 1 ≤ masha_room ∧ masha_room < alina_room ∧ alina_room ≤ total_rooms)
  (h3 : masha_room + alina_room = 2022)
  (h4 : (((alina_room - masha_room - 1) * (masha_room + alina_room)) / 2) = 3033) :
  masha_room = 1009 := by
sorry

end NUMINAMATH_CALUDE_masha_room_number_l3585_358549


namespace NUMINAMATH_CALUDE_polynomial_bound_l3585_358563

theorem polynomial_bound (a b c : ℝ) :
  (∀ x : ℝ, abs x ≤ 1 → abs (a * x^2 + b * x + c) ≤ 1) →
  (∀ x : ℝ, abs x ≤ 1 → abs (c * x^2 + b * x + a) ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_bound_l3585_358563


namespace NUMINAMATH_CALUDE_jesselton_orchestra_max_size_l3585_358538

theorem jesselton_orchestra_max_size :
  ∀ n m : ℕ,
  n = 30 * m →
  n % 32 = 7 →
  n < 1200 →
  (∀ k : ℕ, k = 30 * m ∧ k % 32 = 7 ∧ k < 1200 → k ≤ n) →
  n = 750 :=
by
  sorry

end NUMINAMATH_CALUDE_jesselton_orchestra_max_size_l3585_358538


namespace NUMINAMATH_CALUDE_bike_ride_distance_l3585_358556

/-- Calculates the total distance of a 3-hour bike ride given the conditions -/
theorem bike_ride_distance (second_hour : ℝ) 
  (h1 : second_hour = 12)
  (h2 : second_hour = 1.2 * (second_hour / 1.2))
  (h3 : second_hour * 1.25 = 15) : 
  (second_hour / 1.2) + second_hour + (second_hour * 1.25) = 37 := by
  sorry

end NUMINAMATH_CALUDE_bike_ride_distance_l3585_358556


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_empty_solution_l3585_358544

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 1|

-- Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≤ 4} = Set.Icc 0 4 := by sorry

-- Part II
theorem range_of_a_for_empty_solution :
  {a : ℝ | ∀ x, f a x ≥ 2} = Set.Iic (-1) ∪ Set.Ici 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_for_empty_solution_l3585_358544


namespace NUMINAMATH_CALUDE_product_104_96_l3585_358507

theorem product_104_96 : 104 * 96 = 9984 := by
  sorry

end NUMINAMATH_CALUDE_product_104_96_l3585_358507


namespace NUMINAMATH_CALUDE_min_plates_min_plates_achieved_l3585_358531

theorem min_plates (m n : ℕ) : 
  2 * m + n ≥ 15 ∧ 
  m + 2 * n ≥ 18 ∧ 
  m + 3 * n ≥ 27 →
  m + n ≥ 12 :=
by
  sorry

theorem min_plates_achieved : 
  ∃ (m n : ℕ), 
    2 * m + n ≥ 15 ∧ 
    m + 2 * n ≥ 18 ∧ 
    m + 3 * n ≥ 27 ∧
    m + n = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_min_plates_min_plates_achieved_l3585_358531


namespace NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l3585_358536

def total_trees : ℕ := 17
def maple_trees : ℕ := 4
def oak_trees : ℕ := 5
def birch_trees : ℕ := 6
def pine_trees : ℕ := 2

def non_birch_trees : ℕ := maple_trees + oak_trees + pine_trees

theorem birch_tree_arrangement_probability :
  let total_arrangements := Nat.choose total_trees birch_trees
  let valid_arrangements := Nat.choose (non_birch_trees + 1) birch_trees
  (valid_arrangements : ℚ) / total_arrangements = 21 / 283 := by
sorry

end NUMINAMATH_CALUDE_birch_tree_arrangement_probability_l3585_358536


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l3585_358550

/-- A Pythagorean triple is a set of three positive integers a, b, and c that satisfy a² + b² = c² -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- The set (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean_triple :
  isPythagoreanTriple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l3585_358550


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3585_358541

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x < 1 → x < 2) ∧ 
  (∃ x : ℝ, x < 2 ∧ ¬(x < 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3585_358541


namespace NUMINAMATH_CALUDE_percentage_of_day_l3585_358560

theorem percentage_of_day (hours_in_day : ℝ) (percentage : ℝ) (result : ℝ) : 
  hours_in_day = 24 →
  percentage = 29.166666666666668 →
  result = 7 →
  (percentage / 100) * hours_in_day = result :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_day_l3585_358560


namespace NUMINAMATH_CALUDE_no_solution_iff_m_geq_two_thirds_l3585_358566

theorem no_solution_iff_m_geq_two_thirds (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2*m < 0 ∧ x + m > 2)) ↔ m ≥ 2/3 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_geq_two_thirds_l3585_358566


namespace NUMINAMATH_CALUDE_min_m_bound_l3585_358574

theorem min_m_bound (a b : ℝ) (h1 : |a - b| ≤ 1) (h2 : |2 * a - 1| ≤ 1) :
  ∃ m : ℝ, (∀ a b : ℝ, |a - b| ≤ 1 → |2 * a - 1| ≤ 1 → |4 * a - 3 * b + 2| ≤ m) ∧
  (∀ m' : ℝ, (∀ a b : ℝ, |a - b| ≤ 1 → |2 * a - 1| ≤ 1 → |4 * a - 3 * b + 2| ≤ m') → m ≤ m') ∧
  m = 6 :=
sorry

end NUMINAMATH_CALUDE_min_m_bound_l3585_358574


namespace NUMINAMATH_CALUDE_jayden_coins_l3585_358542

theorem jayden_coins (jason_coins jayden_coins total_coins : ℕ) 
  (h1 : jason_coins = jayden_coins + 60)
  (h2 : jason_coins + jayden_coins = total_coins)
  (h3 : total_coins = 660) : 
  jayden_coins = 300 := by
sorry

end NUMINAMATH_CALUDE_jayden_coins_l3585_358542


namespace NUMINAMATH_CALUDE_smallest_base_representation_l3585_358596

/-- Given two bases a and b greater than 2, this function returns the base-10 
    representation of 21 in base a and 12 in base b. -/
def baseRepresentation (a b : ℕ) : ℕ := 2 * a + 1

/-- The smallest base-10 integer that can be represented as 21₍ₐ₎ in one base 
    and 12₍ᵦ₎ in another base, where a and b are any bases larger than 2. -/
def smallestInteger : ℕ := 7

theorem smallest_base_representation :
  ∀ a b : ℕ, a > 2 → b > 2 → 
  (baseRepresentation a b = baseRepresentation b a) → 
  (baseRepresentation a b ≥ smallestInteger) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_representation_l3585_358596


namespace NUMINAMATH_CALUDE_equation_solution_l3585_358572

theorem equation_solution : ∃ x : ℝ, (x - 1) / 2 = 1 - (x + 2) / 3 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3585_358572


namespace NUMINAMATH_CALUDE_workers_per_block_l3585_358501

/-- Proves that given a total budget of $4000, a cost of $4 per gift, and 10 blocks in the company,
    the number of workers in each block is 100. -/
theorem workers_per_block (total_budget : ℕ) (cost_per_gift : ℕ) (num_blocks : ℕ)
  (h1 : total_budget = 4000)
  (h2 : cost_per_gift = 4)
  (h3 : num_blocks = 10) :
  (total_budget / cost_per_gift) / num_blocks = 100 := by
sorry

#eval (4000 / 4) / 10  -- Should output 100

end NUMINAMATH_CALUDE_workers_per_block_l3585_358501


namespace NUMINAMATH_CALUDE_cake_distribution_l3585_358539

theorem cake_distribution (total_pieces : ℕ) (pieces_per_friend : ℕ) (num_friends : ℕ) :
  total_pieces = 150 →
  pieces_per_friend = 3 →
  total_pieces = pieces_per_friend * num_friends →
  num_friends = 50 := by
sorry

end NUMINAMATH_CALUDE_cake_distribution_l3585_358539


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_perpendicular_line_a_value_l3585_358509

-- Define the line l
def line_l (a : ℝ) (x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0

-- Theorem 1: Line l passes through the fixed point (1, -3) for all a
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line_l a 1 (-3) := by sorry

-- Theorem 2: When line l is perpendicular to the given line, a = 1/2
theorem perpendicular_line_a_value :
  (∀ x y : ℝ, line_l (1/2) x y → perp_line x y → (a + 1) * (-3/2) = -1) ∧
  (∀ a : ℝ, (∀ x y : ℝ, line_l a x y → perp_line x y → (a + 1) * (-3/2) = -1) → a = 1/2) := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_perpendicular_line_a_value_l3585_358509


namespace NUMINAMATH_CALUDE_quadratic_roots_l3585_358516

theorem quadratic_roots (a : ℝ) : 
  (3^2 - 2*3 + a = 0) → 
  ((-1)^2 - 2*(-1) + a = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3585_358516


namespace NUMINAMATH_CALUDE_min_distance_inverse_curves_l3585_358547

/-- The minimum distance between points on two inverse curves -/
theorem min_distance_inverse_curves :
  let f (x : ℝ) := (1/2) * Real.exp x
  let g (x : ℝ) := Real.log (2 * x)
  ∀ (x y : ℝ), x > 0 → y > 0 →
  let P := (x, f x)
  let Q := (y, g y)
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 * (1 - Real.log 2) ∧
    ∀ (x' y' : ℝ), x' > 0 → y' > 0 →
    let P' := (x', f x')
    let Q' := (y', g y')
    Real.sqrt ((x' - y')^2 + (f x' - g y')^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_inverse_curves_l3585_358547


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3585_358517

theorem quadratic_root_relation (p q : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ x - y = 2 ∧ x = 2*y) → p = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3585_358517


namespace NUMINAMATH_CALUDE_shortest_distance_between_circles_l3585_358505

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles :
  let circle1 := (fun (x y : ℝ) => x^2 - 2*x + y^2 + 6*y + 2 = 0)
  let circle2 := (fun (x y : ℝ) => x^2 + 6*x + y^2 - 2*y + 9 = 0)
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 - 1 ∧
  ∀ (p1 p2 : ℝ × ℝ),
    circle1 p1.1 p1.2 → circle2 p2.1 p2.2 →
    Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) ≥ d :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_between_circles_l3585_358505


namespace NUMINAMATH_CALUDE_total_ipods_l3585_358519

-- Define the initial number of iPods Emmy has
def emmy_initial : ℕ := 14

-- Define the number of iPods Emmy loses
def emmy_lost : ℕ := 6

-- Define Emmy's remaining iPods
def emmy_remaining : ℕ := emmy_initial - emmy_lost

-- Define Rosa's iPods in terms of Emmy's remaining
def rosa : ℕ := emmy_remaining / 2

-- Theorem to prove
theorem total_ipods : emmy_remaining + rosa = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_ipods_l3585_358519


namespace NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l3585_358503

theorem half_plus_five_equals_eleven (n : ℝ) : (1/2) * n + 5 = 11 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l3585_358503


namespace NUMINAMATH_CALUDE_volume_of_prism_with_inscribed_sphere_l3585_358570

/-- A regular triangular prism with an inscribed sphere -/
structure RegularTriangularPrism where
  -- The radius of the inscribed sphere
  sphere_radius : ℝ
  -- Assertion that the sphere is inscribed in the prism
  sphere_inscribed : sphere_radius > 0

/-- The volume of a regular triangular prism with an inscribed sphere -/
def prism_volume (p : RegularTriangularPrism) : ℝ :=
  -- Definition of volume calculation
  sorry

/-- Theorem: The volume of a regular triangular prism with an inscribed sphere of radius 2 is 48√3 -/
theorem volume_of_prism_with_inscribed_sphere :
  ∀ (p : RegularTriangularPrism), p.sphere_radius = 2 → prism_volume p = 48 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_prism_with_inscribed_sphere_l3585_358570


namespace NUMINAMATH_CALUDE_extreme_values_of_f_l3585_358591

def f (x : ℝ) := x^3 - 12*x + 12

theorem extreme_values_of_f :
  (∃ x, f x = -4 ∧ x = 2) ∧
  (∃ x, f x = 28) →
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, f x ≤ 28) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = -4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f x = 28) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_of_f_l3585_358591


namespace NUMINAMATH_CALUDE_max_abcd_is_one_l3585_358557

theorem max_abcd_is_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : (1 + a) * (1 + b) * (1 + c) * (1 + d) = 16) :
  abcd ≤ 1 ∧ ∃ (a' b' c' d' : ℝ), 0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧
    (1 + a') * (1 + b') * (1 + c') * (1 + d') = 16 ∧ a' * b' * c' * d' = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_abcd_is_one_l3585_358557


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3585_358584

theorem constant_term_expansion (x : ℝ) : 
  ∃ (f : ℝ → ℝ), (∀ x ≠ 0, f x = (1/x - x^(1/2))^6) ∧ 
  (∃ c : ℝ, ∀ x ≠ 0, f x = c + x * (f x - c) ∧ c = 15) :=
sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l3585_358584


namespace NUMINAMATH_CALUDE_ratio_equality_implies_fraction_value_l3585_358535

theorem ratio_equality_implies_fraction_value (x y z : ℚ) 
  (h : x / 3 = y / 5 ∧ y / 5 = z / 7) : 
  (y + z) / (3 * x - y) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_implies_fraction_value_l3585_358535


namespace NUMINAMATH_CALUDE_prob_neither_correct_l3585_358571

/-- Given probabilities for answering questions correctly, calculate the probability of answering neither correctly -/
theorem prob_neither_correct (P_A P_B P_AB : ℝ) 
  (h1 : P_A = 0.65)
  (h2 : P_B = 0.55)
  (h3 : P_AB = 0.40)
  (h4 : 0 ≤ P_A ∧ P_A ≤ 1)
  (h5 : 0 ≤ P_B ∧ P_B ≤ 1)
  (h6 : 0 ≤ P_AB ∧ P_AB ≤ 1) :
  1 - (P_A + P_B - P_AB) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_prob_neither_correct_l3585_358571


namespace NUMINAMATH_CALUDE_simplify_expression_square_root_of_expression_l3585_358599

-- Part 1
theorem simplify_expression (x : ℝ) (h : 1 < x ∧ x < 4) :
  Real.sqrt ((1 - x)^2) - abs (x - 5) = 2 * x - 6 := by sorry

-- Part 2
theorem square_root_of_expression (x y : ℝ) (h : y = 1 + Real.sqrt (2*x - 1) + Real.sqrt (1 - 2*x)) :
  Real.sqrt (2*x + 3*y) = 2 ∨ Real.sqrt (2*x + 3*y) = -2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_square_root_of_expression_l3585_358599


namespace NUMINAMATH_CALUDE_unique_solution_l3585_358518

/-- Calculates the cost per person based on the number of participants -/
def costPerPerson (n : ℕ) : ℕ :=
  if n ≤ 30 then 80
  else max 50 (80 - (n - 30))

/-- Calculates the total cost for a given number of participants -/
def totalCost (n : ℕ) : ℕ :=
  n * costPerPerson n

/-- States that there exists a unique number of employees that satisfies the problem conditions -/
theorem unique_solution : ∃! n : ℕ, n > 30 ∧ totalCost n = 2800 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3585_358518


namespace NUMINAMATH_CALUDE_davids_english_marks_l3585_358529

/-- Given David's marks in 4 subjects and the average of all 5 subjects, 
    prove that his marks in English are 70. -/
theorem davids_english_marks 
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℚ)
  (h1 : math_marks = 63)
  (h2 : physics_marks = 80)
  (h3 : chemistry_marks = 63)
  (h4 : biology_marks = 65)
  (h5 : average_marks = 68.2)
  (h6 : (math_marks + physics_marks + chemistry_marks + biology_marks + english_marks : ℚ) / 5 = average_marks) :
  english_marks = 70 :=
by sorry

end NUMINAMATH_CALUDE_davids_english_marks_l3585_358529


namespace NUMINAMATH_CALUDE_solution_set_l3585_358559

theorem solution_set (x : ℝ) :
  x > 9 →
  Real.sqrt (x - 4 * Real.sqrt (x - 9)) + 3 = Real.sqrt (x + 4 * Real.sqrt (x - 9)) - 3 →
  x ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l3585_358559


namespace NUMINAMATH_CALUDE_revenue_is_432_l3585_358512

/-- Represents the rental business scenario -/
structure RentalBusiness where
  canoe_price : ℕ
  kayak_price : ℕ
  canoe_kayak_ratio : ℚ
  canoe_kayak_difference : ℕ

/-- Calculates the total revenue for the day -/
def total_revenue (rb : RentalBusiness) : ℕ :=
  let kayaks := rb.canoe_kayak_difference * 3
  let canoes := kayaks + rb.canoe_kayak_difference
  kayaks * rb.kayak_price + canoes * rb.canoe_price

/-- Theorem stating that the total revenue for the given scenario is $432 -/
theorem revenue_is_432 (rb : RentalBusiness) 
  (h1 : rb.canoe_price = 9)
  (h2 : rb.kayak_price = 12)
  (h3 : rb.canoe_kayak_ratio = 4/3)
  (h4 : rb.canoe_kayak_difference = 6) :
  total_revenue rb = 432 := by
  sorry

#eval total_revenue { canoe_price := 9, kayak_price := 12, canoe_kayak_ratio := 4/3, canoe_kayak_difference := 6 }

end NUMINAMATH_CALUDE_revenue_is_432_l3585_358512


namespace NUMINAMATH_CALUDE_output_theorem_l3585_358592

/-- Represents the output of the program at each step -/
structure ProgramOutput :=
  (x : ℕ)
  (y : ℤ)

/-- The sequence of outputs from the program -/
def output_sequence : ℕ → ProgramOutput := sorry

/-- The theorem stating that when y = -10, x = 32 in the output sequence -/
theorem output_theorem :
  ∃ n : ℕ, (output_sequence n).y = -10 ∧ (output_sequence n).x = 32 := by
  sorry

end NUMINAMATH_CALUDE_output_theorem_l3585_358592


namespace NUMINAMATH_CALUDE_loss_equates_to_five_balls_l3585_358530

/-- Given the sale of 20 balls at Rs. 720 with a loss equal to the cost price of some balls,
    and the cost price of a ball being Rs. 48, prove that the loss equates to 5 balls. -/
theorem loss_equates_to_five_balls 
  (total_balls : ℕ) 
  (selling_price : ℕ) 
  (cost_price_per_ball : ℕ) 
  (h1 : total_balls = 20)
  (h2 : selling_price = 720)
  (h3 : cost_price_per_ball = 48) :
  (total_balls * cost_price_per_ball - selling_price) / cost_price_per_ball = 5 :=
by sorry

end NUMINAMATH_CALUDE_loss_equates_to_five_balls_l3585_358530


namespace NUMINAMATH_CALUDE_solve_for_m_l3585_358598

theorem solve_for_m : ∃ m : ℝ, (-1 : ℝ) - 2 * m = 9 → m = -5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l3585_358598


namespace NUMINAMATH_CALUDE_stars_permutations_l3585_358500

def word_length : ℕ := 5
def repeated_letter_count : ℕ := 2
def unique_letters_count : ℕ := 3

theorem stars_permutations :
  (word_length.factorial) / (repeated_letter_count.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_stars_permutations_l3585_358500


namespace NUMINAMATH_CALUDE_coprime_set_properties_l3585_358582

-- Define the set M
def M (a b : ℕ) : Set ℤ :=
  {z : ℤ | ∃ (x y : ℕ), z = a * x + b * y}

-- State the theorem
theorem coprime_set_properties (a b : ℕ) (h : Nat.Coprime a b) :
  -- Part 1: The largest integer not in M is ab - a - b
  (∀ z : ℤ, z ∉ M a b → z ≤ a * b - a - b) ∧
  (a * b - a - b : ℤ) ∉ M a b ∧
  -- Part 2: For any integer n, exactly one of n and (ab - a - b - n) is in M
  (∀ n : ℤ, (n ∈ M a b ↔ (a * b - a - b - n) ∉ M a b)) := by
  sorry

end NUMINAMATH_CALUDE_coprime_set_properties_l3585_358582


namespace NUMINAMATH_CALUDE_chemistry_is_other_subject_l3585_358565

/-- Represents the scores in three subjects -/
structure Scores where
  physics : ℝ
  chemistry : ℝ
  mathematics : ℝ

/-- The conditions of the problem -/
def satisfiesConditions (s : Scores) : Prop :=
  s.physics = 110 ∧
  (s.physics + s.chemistry + s.mathematics) / 3 = 70 ∧
  (s.physics + s.mathematics) / 2 = 90 ∧
  (s.physics + s.chemistry) / 2 = 70

/-- The theorem to be proved -/
theorem chemistry_is_other_subject (s : Scores) :
  satisfiesConditions s → (s.physics + s.chemistry) / 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_is_other_subject_l3585_358565


namespace NUMINAMATH_CALUDE_missile_time_equation_l3585_358555

/-- Represents the speed of the missile in Mach -/
def missile_speed : ℝ := 26

/-- Represents the conversion factor from Mach to meters per second -/
def mach_to_mps : ℝ := 340

/-- Represents the distance to the target in kilometers -/
def target_distance : ℝ := 12000

/-- Represents the time taken to reach the target in minutes -/
def time_to_target : ℝ → ℝ := λ x => x

/-- Theorem stating the equation for the time taken by the missile to reach the target -/
theorem missile_time_equation :
  ∀ x : ℝ, (missile_speed * mach_to_mps * 60 * time_to_target x) / 1000 = target_distance * 1000 :=
by sorry

end NUMINAMATH_CALUDE_missile_time_equation_l3585_358555


namespace NUMINAMATH_CALUDE_divisibility_rule_2701_l3585_358548

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_squares_of_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  tens * tens + ones * ones

theorem divisibility_rule_2701 :
  ∀ x : ℕ, is_two_digit x →
    (2701 % x = 0 ↔ sum_of_squares_of_digits x = 58) := by sorry

end NUMINAMATH_CALUDE_divisibility_rule_2701_l3585_358548


namespace NUMINAMATH_CALUDE_no_roots_lost_l3585_358521

theorem no_roots_lost (x : ℝ) : 
  (x^4 + x^3 + x^2 + x + 1 = 0) ↔ (x^2 + x + 1 + 1/x + 1/x^2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_roots_lost_l3585_358521


namespace NUMINAMATH_CALUDE_blankets_per_person_l3585_358580

/-- Proves that the number of blankets each person gave on the first day is 2 --/
theorem blankets_per_person (team_size : Nat) (last_day_blankets : Nat) (total_blankets : Nat) :
  team_size = 15 →
  last_day_blankets = 22 →
  total_blankets = 142 →
  ∃ (first_day_blankets : Nat),
    first_day_blankets * team_size + 3 * (first_day_blankets * team_size) + last_day_blankets = total_blankets ∧
    first_day_blankets = 2 := by
  sorry

#check blankets_per_person

end NUMINAMATH_CALUDE_blankets_per_person_l3585_358580


namespace NUMINAMATH_CALUDE_problem_statement_l3585_358514

/-- The set D of positive real pairs (x₁, x₂) that sum to k -/
def D (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 = k}

theorem problem_statement (k : ℝ) (hk : k > 0) :
  (∀ p ∈ D k, 0 < p.1 * p.2 ∧ p.1 * p.2 ≤ k^2 / 4) ∧
  (k ≥ 1 → ∀ p ∈ D k, (1 / p.1 - p.1) * (1 / p.2 - p.2) ≤ (k / 2 - 2 / k)^2) ∧
  (∀ p ∈ D k, (1 / p.1 - p.1) * (1 / p.2 - p.2) ≥ (k / 2 - 2 / k)^2 ↔ 0 < k^2 ∧ k^2 ≤ 4 * Real.sqrt 5 - 8) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3585_358514


namespace NUMINAMATH_CALUDE_two_pairs_satisfying_equation_l3585_358533

theorem two_pairs_satisfying_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℕ), 
    (2 * x₁^3 = y₁^4) ∧ 
    (2 * x₂^3 = y₂^4) ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) :=
by sorry

end NUMINAMATH_CALUDE_two_pairs_satisfying_equation_l3585_358533


namespace NUMINAMATH_CALUDE_zoo_count_l3585_358520

/-- Counts the total number of animals observed during a zoo trip --/
def count_animals (snakes : ℕ) (arctic_foxes : ℕ) (leopards : ℕ) : ℕ :=
  let bee_eaters := 10 * (snakes / 2 + 2 * leopards)
  let cheetahs := 4 * (arctic_foxes - leopards)
  let alligators := 3 * (snakes * arctic_foxes * leopards)
  snakes + arctic_foxes + leopards + bee_eaters + cheetahs + alligators

/-- Theorem stating the total number of animals counted during the zoo trip --/
theorem zoo_count : count_animals 100 80 20 = 481340 := by
  sorry

end NUMINAMATH_CALUDE_zoo_count_l3585_358520


namespace NUMINAMATH_CALUDE_lottery_probability_maximum_l3585_358564

/-- The probability of winning in one draw -/
def p₀ (n : ℕ) : ℚ := (10 * n) / ((n + 5) * (n + 4))

/-- The probability of exactly one win in three draws -/
def p (n : ℕ) : ℚ := 3 * p₀ n * (1 - p₀ n)^2

/-- The statement to prove -/
theorem lottery_probability_maximum (n : ℕ) (h : n > 1) :
  ∃ (max_n : ℕ) (max_p : ℚ),
    max_n > 1 ∧
    max_p = p max_n ∧
    ∀ m, m > 1 → p m ≤ max_p ∧
    max_n = 20 ∧
    max_p = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_lottery_probability_maximum_l3585_358564


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3585_358558

theorem quadratic_factorization (a x : ℝ) : a * x^2 - a = a * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3585_358558


namespace NUMINAMATH_CALUDE_clock_gain_per_hour_l3585_358588

theorem clock_gain_per_hour (total_gain : ℕ) (total_hours : ℕ) (gain_per_hour : ℚ) :
  total_gain = 40 ∧ total_hours = 8 →
  gain_per_hour = total_gain / total_hours →
  gain_per_hour = 5 := by
sorry

end NUMINAMATH_CALUDE_clock_gain_per_hour_l3585_358588


namespace NUMINAMATH_CALUDE_circle_area_tripled_l3585_358597

theorem circle_area_tripled (n : ℝ) (r : ℝ) (h_pos : r > 0) : 
  π * (r + n)^2 = 3 * π * r^2 → r = n * (Real.sqrt 3 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l3585_358597
