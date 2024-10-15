import Mathlib

namespace NUMINAMATH_CALUDE_initial_average_production_l1476_147687

/-- Given a company's production data, calculate the initial average daily production. -/
theorem initial_average_production
  (n : ℕ) -- number of days of initial production
  (today_production : ℕ) -- today's production in units
  (new_average : ℚ) -- new average including today's production
  (hn : n = 11)
  (ht : today_production = 110)
  (ha : new_average = 55)
  : (n : ℚ) * (n + 1 : ℚ) * new_average - (n + 1 : ℚ) * today_production = n * 50
  := by sorry

end NUMINAMATH_CALUDE_initial_average_production_l1476_147687


namespace NUMINAMATH_CALUDE_kira_downloaded_songs_l1476_147661

/-- The size of each song in megabytes -/
def song_size : ℕ := 5

/-- The total size of new songs in megabytes -/
def total_new_size : ℕ := 140

/-- The number of songs downloaded later on that day -/
def songs_downloaded : ℕ := total_new_size / song_size

theorem kira_downloaded_songs :
  songs_downloaded = 28 := by sorry

end NUMINAMATH_CALUDE_kira_downloaded_songs_l1476_147661


namespace NUMINAMATH_CALUDE_expansion_proofs_l1476_147648

theorem expansion_proofs (x a b c : ℝ) : 
  (3*(x+1)*(x-1) - (3*x+2)*(2-3*x) = 12*x^2 - 7) ∧ 
  ((a+2*b+3*c)*(a+2*b-3*c) = a^2 + 4*a*b + 4*b^2 - 9*c^2) := by
  sorry

end NUMINAMATH_CALUDE_expansion_proofs_l1476_147648


namespace NUMINAMATH_CALUDE_ball_catching_circle_l1476_147667

theorem ball_catching_circle (n : ℕ) (skip : ℕ) (h1 : n = 50) (h2 : skip = 6) :
  ∃ (m : ℕ), m = 25 ∧ m = n - (n.lcm skip / skip) :=
sorry

end NUMINAMATH_CALUDE_ball_catching_circle_l1476_147667


namespace NUMINAMATH_CALUDE_equation_solution_l1476_147645

theorem equation_solution (a : ℝ) : 
  (∀ x, 3*x + |a - 2| = -3 ↔ 3*x + 4 = 0) → 
  ((a - 2)^2010 - 2*a + 1 = -4 ∨ (a - 2)^2010 - 2*a + 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1476_147645


namespace NUMINAMATH_CALUDE_a_properties_l1476_147660

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 3 * a n + 2 * (Int.sqrt (2 * (a n)^2 - 1)).toNat

theorem a_properties :
  (∀ n : ℕ, a n > 0) ∧
  (∀ m : ℕ, ¬(2015 ∣ a m)) := by
  sorry

end NUMINAMATH_CALUDE_a_properties_l1476_147660


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1476_147652

/-- Given a hyperbola with asymptote equations x ± 2y = 0 and focal length 10,
    prove that its equation is either x²/20 - y²/5 = 1 or y²/5 - x²/20 = 1 -/
theorem hyperbola_equation (x y : ℝ) :
  (∃ k : ℝ, k * x + 2 * y = 0 ∧ k * x - 2 * y = 0) →
  (∃ c : ℝ, c^2 = 100) →
  (x^2 / 20 - y^2 / 5 = 1) ∨ (y^2 / 5 - x^2 / 20 = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1476_147652


namespace NUMINAMATH_CALUDE_marble_problem_l1476_147659

theorem marble_problem (total : ℕ) (white : ℕ) (remaining : ℕ) : 
  total = 50 → 
  white = 20 → 
  remaining = 40 → 
  ∃ (red blue removed : ℕ),
    red = blue ∧ 
    total = white + red + blue ∧
    removed = total - remaining ∧
    removed = 2 * (white - blue) :=
by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l1476_147659


namespace NUMINAMATH_CALUDE_vector_subtraction_l1476_147626

def a : Fin 3 → ℝ := ![5, -3, 2]
def b : Fin 3 → ℝ := ![-2, 4, 1]

theorem vector_subtraction :
  (fun i => a i - 2 * b i) = ![9, -11, 0] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1476_147626


namespace NUMINAMATH_CALUDE_expression_value_at_4_l1476_147632

theorem expression_value_at_4 (a b : ℤ) 
  (h : ∀ (n : ℤ), ∃ (k : ℤ), (2 * n^3 + 3 * n^2 + a * n + b) = k * (n^2 + 1)) :
  (2 * 4^3 + 3 * 4^2 + a * 4 + b) / (4^2 + 1) = 11 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_4_l1476_147632


namespace NUMINAMATH_CALUDE_factor_x6_minus_64_l1476_147681

theorem factor_x6_minus_64 (x : ℝ) : 
  x^6 - 64 = (x - 2) * (x + 2) * (x^4 + 4*x^2 + 16) := by
  sorry

end NUMINAMATH_CALUDE_factor_x6_minus_64_l1476_147681


namespace NUMINAMATH_CALUDE_kiwi_count_l1476_147672

theorem kiwi_count (initial_oranges : ℕ) (added_kiwis : ℕ) (orange_percentage : ℚ) : 
  initial_oranges = 24 →
  added_kiwis = 26 →
  orange_percentage = 30 / 100 →
  ∃ initial_kiwis : ℕ, 
    (initial_oranges : ℚ) = orange_percentage * ((initial_oranges : ℚ) + (initial_kiwis : ℚ) + (added_kiwis : ℚ)) →
    initial_kiwis = 30 :=
by sorry

end NUMINAMATH_CALUDE_kiwi_count_l1476_147672


namespace NUMINAMATH_CALUDE_two_times_two_thousand_fifteen_minus_two_thousand_fifteen_l1476_147688

theorem two_times_two_thousand_fifteen_minus_two_thousand_fifteen : 2 * 2015 - 2015 = 2015 := by
  sorry

end NUMINAMATH_CALUDE_two_times_two_thousand_fifteen_minus_two_thousand_fifteen_l1476_147688


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1476_147638

theorem arithmetic_sequence_sum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) (n : ℕ) :
  a₁ = 2/7 →
  aₙ = 20/7 →
  d = 2/7 →
  n * (a₁ + aₙ) / 2 = 110/7 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1476_147638


namespace NUMINAMATH_CALUDE_original_recipe_eggs_l1476_147695

/-- The number of eggs needed for an eight-person cake -/
def eggs_for_eight : ℕ := 3 + 1

/-- The number of people the original recipe serves -/
def original_servings : ℕ := 4

/-- The number of people Tyler wants to serve -/
def target_servings : ℕ := 8

/-- The number of eggs required for the original recipe -/
def eggs_for_original : ℕ := eggs_for_eight / 2

theorem original_recipe_eggs :
  eggs_for_original = 2 :=
sorry

end NUMINAMATH_CALUDE_original_recipe_eggs_l1476_147695


namespace NUMINAMATH_CALUDE_tangent_line_at_1_l1476_147690

-- Define the function f
def f (x : ℝ) : ℝ := -(x^3) + x^2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := -3*x^2 + 2*x

-- Theorem statement
theorem tangent_line_at_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = -x + 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_1_l1476_147690


namespace NUMINAMATH_CALUDE_complex_magnitude_power_eight_l1476_147602

theorem complex_magnitude_power_eight :
  Complex.abs ((5 / 13 : ℂ) + (12 / 13 : ℂ) * Complex.I) ^ 8 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_power_eight_l1476_147602


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l1476_147643

theorem discount_percentage_proof (jacket_price shirt_price : ℝ)
  (jacket_discount shirt_discount : ℝ) :
  jacket_price = 80 →
  shirt_price = 40 →
  jacket_discount = 0.4 →
  shirt_discount = 0.55 →
  (jacket_price * jacket_discount + shirt_price * shirt_discount) /
  (jacket_price + shirt_price) = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l1476_147643


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l1476_147663

/-- A quadratic function with specific properties -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x, p a b c x = a * x^2 + b * x + c) →   -- p is quadratic
  (p a b c 9 = 4) →                         -- p(9) = 4
  (∀ x, p a b c (18 - x) = p a b c x) →     -- axis of symmetry at x = 9
  (∃ n : ℤ, p a b c 0 = n) →                -- p(0) is an integer
  p a b c 18 = 1 :=                         -- prove p(18) = 1
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l1476_147663


namespace NUMINAMATH_CALUDE_sin_cos_105_degrees_l1476_147627

theorem sin_cos_105_degrees : Real.sin (105 * π / 180) * Real.cos (105 * π / 180) = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_105_degrees_l1476_147627


namespace NUMINAMATH_CALUDE_trig_identity_proof_l1476_147693

theorem trig_identity_proof :
  Real.cos (10 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l1476_147693


namespace NUMINAMATH_CALUDE_sum_x_y_is_12_l1476_147646

/-- An equilateral triangle with side lengths x + 5, y + 11, and 14 -/
structure EquilateralTriangle (x y : ℝ) : Prop where
  side1 : x + 5 = 14
  side2 : y + 11 = 14
  side3 : (14 : ℝ) = 14

/-- The sum of x and y in the equilateral triangle is 12 -/
theorem sum_x_y_is_12 {x y : ℝ} (t : EquilateralTriangle x y) : x + y = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_is_12_l1476_147646


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l1476_147606

theorem greatest_q_minus_r : ∃ (q r : ℕ+), 
  839 = 19 * q + r ∧ 
  ∀ (q' r' : ℕ+), 839 = 19 * q' + r' → (q - r : ℤ) ≥ (q' - r' : ℤ) ∧
  (q - r : ℤ) = 41 := by
  sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l1476_147606


namespace NUMINAMATH_CALUDE_flowerpot_problem_l1476_147684

/-- Given a row of flowerpots, calculates the number of pots between two specific pots. -/
def pots_between (total : ℕ) (a_from_right : ℕ) (b_from_left : ℕ) : ℕ :=
  b_from_left - (total - a_from_right + 1) - 1

/-- Theorem stating that there are 8 flowerpots between A and B under the given conditions. -/
theorem flowerpot_problem :
  pots_between 33 14 29 = 8 := by
  sorry

end NUMINAMATH_CALUDE_flowerpot_problem_l1476_147684


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1476_147601

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + x^2 = x * g y + g x

theorem functional_equation_solution (g : ℝ → ℝ) 
  (h1 : FunctionalEquation g) 
  (h2 : g (-1) = 7) : 
  g (-1001) = 6006013 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1476_147601


namespace NUMINAMATH_CALUDE_fashion_design_not_in_digital_china_l1476_147640

-- Define the concept of a service area
def ServiceArea : Type := String

-- Define Digital China as a structure with a set of service areas
structure DigitalChina :=
  (services : Set ServiceArea)

-- Define known service areas
def environmentalMonitoring : ServiceArea := "Environmental Monitoring"
def publicSecurity : ServiceArea := "Public Security"
def financialInfo : ServiceArea := "Financial Information"
def fashionDesign : ServiceArea := "Fashion Design"

-- Theorem: Fashion design is not a service area of Digital China
theorem fashion_design_not_in_digital_china 
  (dc : DigitalChina) 
  (h1 : environmentalMonitoring ∈ dc.services)
  (h2 : publicSecurity ∈ dc.services)
  (h3 : financialInfo ∈ dc.services) :
  fashionDesign ∉ dc.services := by
  sorry


end NUMINAMATH_CALUDE_fashion_design_not_in_digital_china_l1476_147640


namespace NUMINAMATH_CALUDE_best_fit_line_slope_l1476_147673

/-- Represents a temperature measurement at a specific time -/
structure Measurement where
  time : ℝ
  temp : ℝ

/-- Given three equally spaced time measurements with corresponding temperatures,
    the slope of the best-fit line is (T₃ - T₁) / (t₃ - t₁) -/
theorem best_fit_line_slope (m₁ m₂ m₃ : Measurement) (h : ℝ) 
    (h1 : m₂.time = m₁.time + h)
    (h2 : m₃.time = m₁.time + 2 * h) :
  (m₃.temp - m₁.temp) / (m₃.time - m₁.time) =
    ((m₁.time - (m₁.time + h)) * (m₁.temp - (m₁.temp + m₂.temp + m₃.temp) / 3) +
     (m₂.time - (m₁.time + h)) * (m₂.temp - (m₁.temp + m₂.temp + m₃.temp) / 3) +
     (m₃.time - (m₁.time + h)) * (m₃.temp - (m₁.temp + m₂.temp + m₃.temp) / 3)) /
    ((m₁.time - (m₁.time + h))^2 + (m₂.time - (m₁.time + h))^2 + (m₃.time - (m₁.time + h))^2) :=
by sorry

end NUMINAMATH_CALUDE_best_fit_line_slope_l1476_147673


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1476_147655

theorem fraction_to_decimal : (4 : ℚ) / 5 = (0.8 : ℚ) := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1476_147655


namespace NUMINAMATH_CALUDE_simplify_expression_l1476_147641

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 * b) / (a^2 - a * b) * (a / b - b / a) = a + b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1476_147641


namespace NUMINAMATH_CALUDE_unique_1x5x_divisible_by_36_l1476_147635

def is_form_1x5x (n : ℕ) : Prop :=
  ∃ x : ℕ, x < 10 ∧ n = 1000 + 100 * x + 50 + x

theorem unique_1x5x_divisible_by_36 :
  ∃! n : ℕ, is_form_1x5x n ∧ n % 36 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_1x5x_divisible_by_36_l1476_147635


namespace NUMINAMATH_CALUDE_expected_adjacent_pairs_l1476_147669

/-- The expected number of adjacent boy-girl pairs in a random permutation of boys and girls -/
theorem expected_adjacent_pairs (num_boys num_girls : ℕ) : 
  let total := num_boys + num_girls
  let prob_pair := (num_boys : ℚ) * num_girls / (total * (total - 1))
  let num_pairs := total - 1
  num_boys = 8 → num_girls = 12 → 2 * num_pairs * prob_pair = 912 / 95 := by
  sorry

end NUMINAMATH_CALUDE_expected_adjacent_pairs_l1476_147669


namespace NUMINAMATH_CALUDE_cow_count_is_twelve_l1476_147614

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (count : AnimalCount) : ℕ :=
  2 * count.ducks + 4 * count.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (count : AnimalCount) : ℕ :=
  count.ducks + count.cows

/-- Theorem stating that the number of cows is 12 -/
theorem cow_count_is_twelve :
  ∃ (count : AnimalCount), 
    totalLegs count = 2 * totalHeads count + 24 ∧ 
    count.cows = 12 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_is_twelve_l1476_147614


namespace NUMINAMATH_CALUDE_smallest_variance_most_stable_city_D_most_stable_l1476_147699

/-- Represents a city with its cabbage price variance -/
structure City where
  name : String
  variance : ℝ

/-- Defines stability of cabbage prices based on variance -/
def is_most_stable (cities : List City) (c : City) : Prop :=
  ∀ city ∈ cities, c.variance ≤ city.variance

/-- The theorem stating that the city with the smallest variance is the most stable -/
theorem smallest_variance_most_stable (cities : List City) (c : City) 
    (h₁ : c ∈ cities) 
    (h₂ : ∀ city ∈ cities, c.variance ≤ city.variance) : 
    is_most_stable cities c := by
  sorry

/-- The specific problem instance -/
def problem_instance : List City :=
  [⟨"A", 18.3⟩, ⟨"B", 17.4⟩, ⟨"C", 20.1⟩, ⟨"D", 12.5⟩]

/-- The theorem applied to the specific problem instance -/
theorem city_D_most_stable : 
    is_most_stable problem_instance ⟨"D", 12.5⟩ := by
  sorry

end NUMINAMATH_CALUDE_smallest_variance_most_stable_city_D_most_stable_l1476_147699


namespace NUMINAMATH_CALUDE_wire_ratio_proof_l1476_147656

theorem wire_ratio_proof (total_length : ℝ) (shorter_length : ℝ) :
  total_length = 50 →
  shorter_length = 14.285714285714285 →
  let longer_length := total_length - shorter_length
  shorter_length / longer_length = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_wire_ratio_proof_l1476_147656


namespace NUMINAMATH_CALUDE_regular_polygon_interior_twice_exterior_has_six_sides_l1476_147654

/-- A regular polygon where the sum of interior angles is twice the sum of exterior angles has 6 sides. -/
theorem regular_polygon_interior_twice_exterior_has_six_sides :
  ∀ n : ℕ,
  n > 2 →
  (n - 2) * 180 = 2 * 360 →
  n = 6 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_twice_exterior_has_six_sides_l1476_147654


namespace NUMINAMATH_CALUDE_two_distinct_roots_implies_b_value_l1476_147618

-- Define the polynomial function
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 1

-- State the theorem
theorem two_distinct_roots_implies_b_value (b : ℝ) :
  (∃ (x y : ℝ), x ≠ y ∧ f b x = 0 ∧ f b y = 0 ∧ 
   ∀ (z : ℝ), f b z = 0 → (z = x ∨ z = y)) →
  b = (3/2) * Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_implies_b_value_l1476_147618


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1476_147620

theorem fraction_multiplication (x y : ℝ) (h : x + y ≠ 0) :
  (3*x * 3*y) / (3*x + 3*y) = 3 * (x*y / (x+y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1476_147620


namespace NUMINAMATH_CALUDE_adam_apples_l1476_147676

theorem adam_apples (x : ℕ) : 
  x + 3 * x + 12 * x = 240 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_adam_apples_l1476_147676


namespace NUMINAMATH_CALUDE_fixed_point_of_quadratic_l1476_147692

/-- The quadratic function y = -x^2 + (m-1)x + m passes through the point (-1, 0) for all real m. -/
theorem fixed_point_of_quadratic (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ -x^2 + (m-1)*x + m
  f (-1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_quadratic_l1476_147692


namespace NUMINAMATH_CALUDE_sum_of_c_values_l1476_147629

theorem sum_of_c_values : ∃ (S : Finset ℤ),
  (∀ c ∈ S, c ≤ 30 ∧ 
    ∃ (x y : ℚ), y = x^2 - 11*x - c ∧ 
    ∀ z : ℚ, z^2 - 11*z - c = 0 ↔ (z = x ∨ z = y)) ∧
  (∀ c : ℤ, c ≤ 30 → 
    (∃ (x y : ℚ), y = x^2 - 11*x - c ∧ 
    ∀ z : ℚ, z^2 - 11*z - c = 0 ↔ (z = x ∨ z = y)) → 
    c ∈ S) ∧
  (S.sum id = 38) :=
sorry

end NUMINAMATH_CALUDE_sum_of_c_values_l1476_147629


namespace NUMINAMATH_CALUDE_platform_height_is_44_l1476_147613

/-- Represents the dimensions of a rectangular brick -/
structure Brick where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the experimental setup -/
structure Setup where
  platform_height : ℝ
  brick : Brick
  r : ℝ
  s : ℝ

/-- The main theorem stating the height of the platform -/
theorem platform_height_is_44 (setup : Setup) :
  setup.brick.length + setup.platform_height - 2 * setup.brick.width = setup.r ∧
  setup.brick.width + setup.platform_height - setup.brick.length = setup.s ∧
  setup.platform_height = 2 * setup.brick.width ∧
  setup.r = 36 ∧
  setup.s = 30 →
  setup.platform_height = 44 := by
sorry


end NUMINAMATH_CALUDE_platform_height_is_44_l1476_147613


namespace NUMINAMATH_CALUDE_cos_300_degrees_l1476_147634

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l1476_147634


namespace NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1476_147683

/-- The minimum distance from a point on the parabola y = x^2 to the line 2x - y - 11 = 0 is 2√5 -/
theorem min_distance_parabola_to_line : 
  let parabola := {(x, y) : ℝ × ℝ | y = x^2}
  let line := {(x, y) : ℝ × ℝ | 2*x - y - 11 = 0}
  ∃ d : ℝ, d = 2 * Real.sqrt 5 ∧ 
    (∀ p ∈ parabola, ∀ q ∈ line, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ p ∈ parabola, ∃ q ∈ line, d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) :=
by sorry


end NUMINAMATH_CALUDE_min_distance_parabola_to_line_l1476_147683


namespace NUMINAMATH_CALUDE_existence_of_square_with_no_visible_points_l1476_147621

/-- A point is visible from the origin if the greatest common divisor of its coordinates is 1 -/
def visible_from_origin (x y : ℤ) : Prop := Int.gcd x y = 1

/-- A point (x, y) is inside a square with bottom-left corner (a, b) and side length n if
    a < x < a + n and b < y < b + n -/
def inside_square (x y a b n : ℤ) : Prop :=
  a < x ∧ x < a + n ∧ b < y ∧ y < b + n

theorem existence_of_square_with_no_visible_points :
  ∀ n : ℕ, n > 0 → ∃ a b : ℤ,
    ∀ x y : ℤ, inside_square x y a b n → ¬(visible_from_origin x y) :=
sorry

end NUMINAMATH_CALUDE_existence_of_square_with_no_visible_points_l1476_147621


namespace NUMINAMATH_CALUDE_class_field_trip_budget_l1476_147694

/-- The class's budget for a field trip to the zoo --/
theorem class_field_trip_budget
  (bus_rental_cost : ℕ)
  (admission_cost_per_student : ℕ)
  (number_of_students : ℕ)
  (h1 : bus_rental_cost = 100)
  (h2 : admission_cost_per_student = 10)
  (h3 : number_of_students = 25) :
  bus_rental_cost + admission_cost_per_student * number_of_students = 350 :=
by sorry

end NUMINAMATH_CALUDE_class_field_trip_budget_l1476_147694


namespace NUMINAMATH_CALUDE_july_birth_percentage_l1476_147622

def total_athletes : ℕ := 120
def july_athletes : ℕ := 18

def percentage_born_in_july : ℚ := july_athletes / total_athletes * 100

theorem july_birth_percentage :
  percentage_born_in_july = 15 := by
  sorry

end NUMINAMATH_CALUDE_july_birth_percentage_l1476_147622


namespace NUMINAMATH_CALUDE_division_problem_l1476_147675

theorem division_problem (n : ℕ) : n % 12 = 1 ∧ n / 12 = 9 → n = 109 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1476_147675


namespace NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l1476_147631

/-- Given that the line y = 1/m is tangent to the curve y = xe^x, prove that m = -e -/
theorem tangent_line_to_exponential_curve (m : ℝ) : 
  (∃ n : ℝ, n * Real.exp n = 1/m ∧ 
   ∀ x : ℝ, x * Real.exp x ≤ 1/m ∧ 
   (x * Real.exp x = 1/m → x = n)) → 
  m = -Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_exponential_curve_l1476_147631


namespace NUMINAMATH_CALUDE_binomial_8_choose_5_l1476_147604

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_5_l1476_147604


namespace NUMINAMATH_CALUDE_year_2020_is_gengzi_l1476_147698

/-- Represents the Heavenly Stems in the Sexagenary Cycle -/
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

/-- Represents the Earthly Branches in the Sexagenary Cycle -/
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

/-- Represents a year in the Sexagenary Cycle -/
structure SexagenaryYear :=
  (stem : HeavenlyStem)
  (branch : EarthlyBranch)

/-- The Sexagenary Cycle system -/
def sexagenaryCycle : List SexagenaryYear := sorry

/-- Function to get the Sexagenary year for a given Gregorian year -/
def getSexagenaryYear (gregorianYear : Nat) : SexagenaryYear := sorry

/-- Theorem stating that 2020 corresponds to the GengZi year in the Sexagenary Cycle -/
theorem year_2020_is_gengzi :
  getSexagenaryYear 2020 = SexagenaryYear.mk HeavenlyStem.Geng EarthlyBranch.Zi :=
sorry

end NUMINAMATH_CALUDE_year_2020_is_gengzi_l1476_147698


namespace NUMINAMATH_CALUDE_count_numbers_with_remainder_l1476_147607

theorem count_numbers_with_remainder (n : ℕ) : 
  (Finset.filter (fun N : ℕ => N > 17 ∧ 2017 % N = 17) (Finset.range (2017 + 1))).card = 13 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_remainder_l1476_147607


namespace NUMINAMATH_CALUDE_final_quiz_score_for_a_l1476_147608

def number_of_quizzes : ℕ := 4
def average_score : ℚ := 92 / 100
def required_average : ℚ := 90 / 100

theorem final_quiz_score_for_a (final_score : ℚ) :
  (number_of_quizzes * average_score + final_score) / (number_of_quizzes + 1) ≥ required_average →
  final_score ≥ 82 / 100 :=
by sorry

end NUMINAMATH_CALUDE_final_quiz_score_for_a_l1476_147608


namespace NUMINAMATH_CALUDE_tall_min_voters_to_win_l1476_147686

/-- Represents the voting structure and results of the giraffe beauty contest -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat
  winner : String

/-- Calculates the minimum number of voters required for a giraffe to win the contest -/
def min_voters_to_win (contest : GiraffeContest) : Nat :=
  sorry

/-- The theorem stating the minimum number of voters required for Tall to win -/
theorem tall_min_voters_to_win (contest : GiraffeContest) 
  (h1 : contest.total_voters = 105)
  (h2 : contest.num_districts = 5)
  (h3 : contest.sections_per_district = 7)
  (h4 : contest.voters_per_section = 3)
  (h5 : contest.winner = "Tall") :
  min_voters_to_win contest = 24 := by
  sorry

#check tall_min_voters_to_win

end NUMINAMATH_CALUDE_tall_min_voters_to_win_l1476_147686


namespace NUMINAMATH_CALUDE_range_of_a_for_two_roots_l1476_147625

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then (1/4) * x + 1 else Real.log x

theorem range_of_a_for_two_roots :
  ∃ (a_min a_max : ℝ), a_min = (1/4) ∧ a_max = (1/Real.exp 1) ∧
  ∀ (a : ℝ), (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = a * x₁ ∧ f x₂ = a * x₂ ∧
              ∀ (x : ℝ), f x = a * x → (x = x₁ ∨ x = x₂)) ↔
              (a_min ≤ a ∧ a < a_max) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_two_roots_l1476_147625


namespace NUMINAMATH_CALUDE_worker_arrival_time_l1476_147649

theorem worker_arrival_time (S : ℝ) (D : ℝ) (h1 : D = S * 36) (h2 : S > 0) :
  D / (3/4 * S) - 36 = 12 := by
  sorry

end NUMINAMATH_CALUDE_worker_arrival_time_l1476_147649


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_intersection_product_l1476_147679

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary geometric relations
variable (is_convex_cyclic_quadrilateral : Point → Point → Point → Point → Prop)
variable (is_center_of_circumcircle : Point → Point → Point → Point → Point → Prop)
variable (is_on_circle : Point → Circle → Prop)
variable (circumcircle : Point → Point → Point → Circle)
variable (intersection_point : Circle → Circle → Point)

-- Define the distance function
variable (distance : Point → Point → ℝ)

theorem cyclic_quadrilateral_intersection_product
  (A B C D O Q : Point)
  (h1 : is_convex_cyclic_quadrilateral A B C D)
  (h2 : is_center_of_circumcircle O A B C D)
  (h3 : Q = intersection_point (circumcircle O A B) (circumcircle O C D))
  : distance Q A * distance Q B = distance Q C * distance Q D := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_intersection_product_l1476_147679


namespace NUMINAMATH_CALUDE_solve_system_l1476_147696

theorem solve_system (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1476_147696


namespace NUMINAMATH_CALUDE_investment_equalizes_profits_l1476_147628

/-- The investment amount in yuan that equalizes profits from two selling methods -/
def investment : ℝ := 20000

/-- The profit rate when selling at the beginning of the month -/
def early_profit_rate : ℝ := 0.15

/-- The profit rate for reinvestment -/
def reinvestment_profit_rate : ℝ := 0.10

/-- The profit rate when selling at the end of the month -/
def late_profit_rate : ℝ := 0.30

/-- The storage fee in yuan -/
def storage_fee : ℝ := 700

/-- Theorem stating that the investment amount equalizes profits from both selling methods -/
theorem investment_equalizes_profits :
  investment * (1 + early_profit_rate) * (1 + reinvestment_profit_rate) =
  investment * (1 + late_profit_rate) - storage_fee := by
  sorry

#eval investment -- Should output 20000

end NUMINAMATH_CALUDE_investment_equalizes_profits_l1476_147628


namespace NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l1476_147633

theorem arcsin_arccos_equation_solution (x : ℝ) :
  Real.arcsin (3 * x) + Real.arccos (2 * x) = π / 4 →
  x = 1 / Real.sqrt (11 - 2 * Real.sqrt 2) ∨
  x = -1 / Real.sqrt (11 - 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_arcsin_arccos_equation_solution_l1476_147633


namespace NUMINAMATH_CALUDE_multiple_of_smaller_number_l1476_147653

theorem multiple_of_smaller_number 
  (L S m : ℝ) 
  (h1 : L = 33) 
  (h2 : L + S = 51) 
  (h3 : L = m * S - 3) : 
  m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_smaller_number_l1476_147653


namespace NUMINAMATH_CALUDE_distinct_prime_factors_count_l1476_147611

def product : ℕ := 91 * 92 * 93 * 94

theorem distinct_prime_factors_count :
  (Nat.factors product).toFinset.card = 7 := by sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_count_l1476_147611


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l1476_147612

theorem right_triangle_side_length : ∃ (k : ℕ), 
  (5 * k : ℕ) > 0 ∧ 
  (12 * k : ℕ) > 0 ∧ 
  (13 * k : ℕ) > 0 ∧ 
  (5 * k)^2 + (12 * k)^2 = (13 * k)^2 ∧ 
  (13 * k = 91 ∨ 12 * k = 91 ∨ 5 * k = 91) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l1476_147612


namespace NUMINAMATH_CALUDE_rectangular_field_width_l1476_147677

/-- Given a rectangular field with perimeter 240 meters and perimeter equal to 3 times its length, prove that its width is 40 meters. -/
theorem rectangular_field_width (length width : ℝ) : 
  (2 * length + 2 * width = 240) →  -- Perimeter formula
  (240 = 3 * length) →              -- Perimeter is 3 times length
  width = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l1476_147677


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_power_imaginary_part_of_specific_complex_l1476_147600

theorem imaginary_part_of_complex_power (r θ : ℝ) (n : ℕ) :
  let z := (r * (Complex.cos θ + Complex.I * Complex.sin θ)) ^ n
  Complex.im z = r^n * Real.sin (n * θ) := by sorry

theorem imaginary_part_of_specific_complex (π : ℝ) :
  let z := (2 * (Complex.cos (π/4) + Complex.I * Complex.sin (π/4))) ^ 5
  Complex.im z = -16 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_power_imaginary_part_of_specific_complex_l1476_147600


namespace NUMINAMATH_CALUDE_shaded_squares_count_l1476_147639

/-- Represents the number of shaded squares in each column of the grid -/
def shaded_per_column : List Nat := [1, 3, 5, 4, 2, 0, 0, 0]

/-- The total number of squares in the grid -/
def total_squares : Nat := 30

/-- The number of columns in the grid -/
def num_columns : Nat := 8

theorem shaded_squares_count :
  (List.sum shaded_per_column = 15) ∧
  (List.sum shaded_per_column = total_squares / 2) ∧
  (List.length shaded_per_column = num_columns) := by
  sorry

end NUMINAMATH_CALUDE_shaded_squares_count_l1476_147639


namespace NUMINAMATH_CALUDE_arrangement_count_l1476_147605

/-- Represents the number of different books of each subject -/
structure BookCounts where
  math : Nat
  physics : Nat
  chemistry : Nat

/-- Calculates the number of arrangements given the book counts and constraints -/
def countArrangements (books : BookCounts) : Nat :=
  let totalBooks := books.math + books.physics + books.chemistry
  let mathUnit := 1  -- Treat math books as a single unit
  let nonMathBooks := books.physics + books.chemistry
  let totalUnits := mathUnit + nonMathBooks
  sorry

/-- The theorem to be proven -/
theorem arrangement_count :
  let books : BookCounts := { math := 3, physics := 2, chemistry := 1 }
  countArrangements books = 2592 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1476_147605


namespace NUMINAMATH_CALUDE_subtract_point_five_from_forty_three_point_two_l1476_147668

theorem subtract_point_five_from_forty_three_point_two :
  43.2 - 0.5 = 42.7 := by sorry

end NUMINAMATH_CALUDE_subtract_point_five_from_forty_three_point_two_l1476_147668


namespace NUMINAMATH_CALUDE_work_left_after_collaboration_l1476_147624

theorem work_left_after_collaboration (days_a days_b collab_days : ℕ) 
  (ha : days_a = 15) (hb : days_b = 20) (hc : collab_days = 3) : 
  1 - (collab_days * (1 / days_a + 1 / days_b)) = 13 / 20 := by
  sorry

end NUMINAMATH_CALUDE_work_left_after_collaboration_l1476_147624


namespace NUMINAMATH_CALUDE_smallest_b_value_l1476_147636

theorem smallest_b_value (a b : ℤ) (h1 : 29 < a ∧ a < 41) (h2 : b < 51) 
  (h3 : (40 : ℚ) / b - (30 : ℚ) / 50 = (2 : ℚ) / 5) : b ≥ 40 := by
  sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1476_147636


namespace NUMINAMATH_CALUDE_subset_condition_implies_m_range_l1476_147674

theorem subset_condition_implies_m_range (m : ℝ) : 
  (∀ x, -1 < x ∧ x < 2 → -1 < x ∧ x < m + 1) ∧ 
  (∃ y, -1 < y ∧ y < m + 1 ∧ ¬(-1 < y ∧ y < 2)) → 
  m > 1 := by sorry

end NUMINAMATH_CALUDE_subset_condition_implies_m_range_l1476_147674


namespace NUMINAMATH_CALUDE_different_color_probability_l1476_147630

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

theorem different_color_probability :
  let p_different := (blue_chips * (total_chips - blue_chips) +
                      red_chips * (total_chips - red_chips) +
                      yellow_chips * (total_chips - yellow_chips) +
                      green_chips * (total_chips - green_chips)) /
                     (total_chips * total_chips)
  p_different = 119 / 162 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l1476_147630


namespace NUMINAMATH_CALUDE_circle_angle_measure_l1476_147651

noncomputable def Circle := ℝ × ℝ → Prop

def diameter (c : Circle) (A B : ℝ × ℝ) : Prop := sorry

def parallel (A B C D : ℝ × ℝ) : Prop := sorry

def angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem circle_angle_measure 
  (c : Circle) (A B C D E : ℝ × ℝ) :
  diameter c E B →
  parallel E B D C →
  parallel A B E C →
  angle A E B = (3/7) * Real.pi →
  angle A B E = (4/7) * Real.pi →
  angle B D C = (900/7) * (Real.pi/180) :=
by sorry

end NUMINAMATH_CALUDE_circle_angle_measure_l1476_147651


namespace NUMINAMATH_CALUDE_right_angled_triangle_m_values_l1476_147609

/-- Given three lines that form a right-angled triangle, prove the possible values of m -/
theorem right_angled_triangle_m_values :
  ∀ (m : ℝ),
  (∃ (x y : ℝ), 3*x + 2*y + 6 = 0 ∧ 2*x - 3*m^2*y + 18 = 0 ∧ 2*m*x - 3*y + 12 = 0) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (3*x₁ + 2*y₁ + 6 = 0 ∧ 2*x₁ - 3*m^2*y₁ + 18 = 0) ∧
    (3*x₂ + 2*y₂ + 6 = 0 ∧ 2*m*x₂ - 3*y₂ + 12 = 0) ∧
    ((3*2 + 2*(-3*m^2) = 0) ∨ (3*(2*m) + 2*(-3) = 0) ∨ (2*(-3*m^2) + (-3)*(2*m) = 0))) →
  m = 0 ∨ m = -1 ∨ m = -4/9 :=
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_m_values_l1476_147609


namespace NUMINAMATH_CALUDE_complex_equal_parts_l1476_147642

theorem complex_equal_parts (a : ℝ) :
  let z : ℂ := a - 2 * Complex.I
  z.re = z.im → a = -2 := by sorry

end NUMINAMATH_CALUDE_complex_equal_parts_l1476_147642


namespace NUMINAMATH_CALUDE_rectangle_dimension_difference_l1476_147680

theorem rectangle_dimension_difference (x y : ℝ) 
  (perimeter : x + y = 10)  -- Half of the perimeter is 10
  (diagonal : x^2 + y^2 = 100)  -- Diagonal squared is 100
  : x - y = 10 := by sorry

end NUMINAMATH_CALUDE_rectangle_dimension_difference_l1476_147680


namespace NUMINAMATH_CALUDE_positive_sum_of_odd_monotonic_increasing_l1476_147685

-- Define a monotonic increasing function
def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem positive_sum_of_odd_monotonic_increasing 
  (f : ℝ → ℝ) 
  (a : ℕ → ℝ) 
  (h_mono : MonotonicIncreasing f) 
  (h_odd : OddFunction f) 
  (h_arith : ArithmeticSequence a) 
  (h_a3_pos : a 3 > 0) : 
  f (a 1) + f (a 3) + f (a 5) > 0 :=
by sorry

end NUMINAMATH_CALUDE_positive_sum_of_odd_monotonic_increasing_l1476_147685


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1476_147603

/-- The function f(x) = a^(x+1) - 2 has a fixed point at (-1, -1) for a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x + 1) - 2
  f (-1) = -1 ∧ ∀ x : ℝ, f x = x → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1476_147603


namespace NUMINAMATH_CALUDE_max_diff_of_squares_l1476_147691

theorem max_diff_of_squares (n : ℕ) (h1 : n > 0) (h2 : n + (n + 1) < 150) :
  (∃ (m : ℕ), m > 0 ∧ m + (m + 1) < 150 ∧ (m + 1)^2 - m^2 > (n + 1)^2 - n^2) →
  (n + 1)^2 - n^2 ≤ 149 :=
sorry

end NUMINAMATH_CALUDE_max_diff_of_squares_l1476_147691


namespace NUMINAMATH_CALUDE_fourth_grade_blue_count_l1476_147670

/-- Represents the number of students in each grade and uniform color combination -/
structure StudentCount where
  third_red_blue : ℕ
  third_white : ℕ
  fourth_red : ℕ
  fourth_white : ℕ
  fourth_blue : ℕ
  fifth_red_blue : ℕ
  fifth_white : ℕ

/-- The theorem stating the number of 4th grade students wearing blue uniforms -/
theorem fourth_grade_blue_count (s : StudentCount) : s.fourth_blue = 213 :=
  by
  have total_participants : s.third_red_blue + s.third_white + s.fourth_red + s.fourth_white + s.fourth_blue + s.fifth_red_blue + s.fifth_white = 2013 := by sorry
  have fourth_grade_total : s.fourth_red + s.fourth_white + s.fourth_blue = 600 := by sorry
  have fifth_grade_total : s.fifth_red_blue + s.fifth_white = 800 := by sorry
  have total_white : s.third_white + s.fourth_white + s.fifth_white = 800 := by sorry
  have third_red_blue : s.third_red_blue = 200 := by sorry
  have fourth_red : s.fourth_red = 200 := by sorry
  have fifth_white : s.fifth_white = 200 := by sorry
  sorry

end NUMINAMATH_CALUDE_fourth_grade_blue_count_l1476_147670


namespace NUMINAMATH_CALUDE_systematic_sampling_tenth_group_l1476_147610

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstDraw : ℕ) (n : ℕ) : ℕ :=
  firstDraw + (totalStudents / sampleSize) * (n - 1)

/-- Theorem: In a systematic sampling of 1000 students into 100 groups,
    if the number drawn from the first group is 6,
    then the number drawn from the tenth group is 96. -/
theorem systematic_sampling_tenth_group :
  systematicSample 1000 100 6 10 = 96 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_tenth_group_l1476_147610


namespace NUMINAMATH_CALUDE_no_valid_x_l1476_147664

theorem no_valid_x : ¬∃ (x : ℕ), x > 1 ∧ x ≠ 5 ∧ x ≠ 6 ∧ x ≠ 12 ∧ 
  184 % 5 = 4 ∧ 184 % 6 = 4 ∧ 184 % x = 4 ∧ 184 % 12 = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_x_l1476_147664


namespace NUMINAMATH_CALUDE_room_length_proof_l1476_147689

theorem room_length_proof (width : ℝ) (total_cost : ℝ) (cost_per_sqm : ℝ) :
  width = 3.75 →
  total_cost = 28875 →
  cost_per_sqm = 1400 →
  (total_cost / cost_per_sqm) / width = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_proof_l1476_147689


namespace NUMINAMATH_CALUDE_diagonal_intersections_12x17_l1476_147682

/-- Counts the number of intersection points between a diagonal and grid lines in an m × n grid -/
def countIntersections (m n : ℕ) : ℕ :=
  (n + 1) + (m + 1) - 2

/-- Theorem: In a 12 × 17 grid, the diagonal from A to B intersects the grid at 29 points -/
theorem diagonal_intersections_12x17 :
  countIntersections 12 17 = 29 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersections_12x17_l1476_147682


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l1476_147619

/-- A parabola with focus at (2, 0) that opens to the right has the standard equation y² = 8x -/
theorem parabola_standard_equation (f : ℝ × ℝ) (opens_right : Bool) :
  f = (2, 0) → opens_right = true → ∃ (x y : ℝ), y^2 = 8*x := by
  sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l1476_147619


namespace NUMINAMATH_CALUDE_gcd_180_450_l1476_147666

theorem gcd_180_450 : Nat.gcd 180 450 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcd_180_450_l1476_147666


namespace NUMINAMATH_CALUDE_cube_product_three_four_l1476_147662

theorem cube_product_three_four : (3 : ℕ)^3 * (4 : ℕ)^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_three_four_l1476_147662


namespace NUMINAMATH_CALUDE_factor_expression_l1476_147647

theorem factor_expression (x : ℝ) : 4*x*(x-5) + 6*(x-5) = (4*x+6)*(x-5) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1476_147647


namespace NUMINAMATH_CALUDE_equation_solution_l1476_147650

theorem equation_solution :
  ∃! x : ℝ, (2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) ∧ (x = -14) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1476_147650


namespace NUMINAMATH_CALUDE_sqrt_comparison_l1476_147678

theorem sqrt_comparison : 2 * Real.sqrt 7 < 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l1476_147678


namespace NUMINAMATH_CALUDE_malingerers_exposed_l1476_147671

/-- Represents a five-digit number where each digit is represented by a letter --/
structure CryptarithmNumber where
  a : Nat
  b : Nat
  c : Nat
  h_a_digit : a < 10
  h_b_digit : b < 10
  h_c_digit : c < 10
  h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

def draftees (n : CryptarithmNumber) : Nat :=
  10000 * n.a + 1000 * n.a + 100 * n.b + 10 * n.b + n.b

def malingerers (n : CryptarithmNumber) : Nat :=
  10000 * n.a + 1000 * n.b + 100 * n.c + 10 * n.c + n.c

theorem malingerers_exposed (n : CryptarithmNumber) :
  draftees n - 1 = malingerers n → malingerers n = 10999 := by
  sorry

#check malingerers_exposed

end NUMINAMATH_CALUDE_malingerers_exposed_l1476_147671


namespace NUMINAMATH_CALUDE_certain_number_proof_l1476_147623

theorem certain_number_proof : ∃ x : ℝ, x * 9 = 0.45 * 900 ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1476_147623


namespace NUMINAMATH_CALUDE_data_comparison_l1476_147615

def set1 (x₁ x₂ x₃ x₄ x₅ : ℝ) := [x₁, x₂, x₃, x₄, x₅]
def set2 (x₁ x₂ x₃ x₄ x₅ : ℝ) := [2*x₁+3, 2*x₂+3, 2*x₃+3, 2*x₄+3, 2*x₅+3]

def standardDeviation (xs : List ℝ) : ℝ := sorry
def median (xs : List ℝ) : ℝ := sorry
def mean (xs : List ℝ) : ℝ := sorry

theorem data_comparison (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (standardDeviation (set2 x₁ x₂ x₃ x₄ x₅) ≠ standardDeviation (set1 x₁ x₂ x₃ x₄ x₅)) ∧
  (median (set2 x₁ x₂ x₃ x₄ x₅) ≠ median (set1 x₁ x₂ x₃ x₄ x₅)) ∧
  (mean (set2 x₁ x₂ x₃ x₄ x₅) ≠ mean (set1 x₁ x₂ x₃ x₄ x₅)) := by
  sorry

end NUMINAMATH_CALUDE_data_comparison_l1476_147615


namespace NUMINAMATH_CALUDE_abc_inequality_l1476_147657

theorem abc_inequality (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0)
  (ha' : a > -3) (hb' : b > -3) (hc' : c > -3) : 
  a * b * c > -27 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1476_147657


namespace NUMINAMATH_CALUDE_find_c_l1476_147644

theorem find_c (a b c d e : ℝ) : 
  (a + b + c) / 3 = 16 →
  (c + d + e) / 3 = 26 →
  (a + b + c + d + e) / 5 = 20 →
  c = 26 := by
sorry

end NUMINAMATH_CALUDE_find_c_l1476_147644


namespace NUMINAMATH_CALUDE_p_and_not_q_is_true_l1476_147637

-- Define proposition p
def p : Prop := ∃ x : ℝ, x - 2 > Real.log x

-- Define proposition q
def q : Prop := ∀ x : ℝ, x^2 > 0

-- Theorem to prove
theorem p_and_not_q_is_true : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_and_not_q_is_true_l1476_147637


namespace NUMINAMATH_CALUDE_power_equation_solution_l1476_147616

theorem power_equation_solution (n : ℕ) : 
  2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 17 := by
sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1476_147616


namespace NUMINAMATH_CALUDE_square_shape_side_length_l1476_147665

theorem square_shape_side_length (x : ℝ) :
  x > 0 →
  x - 3 > 0 →
  (x + (x - 1)) = ((x - 2) + (x - 3) + 4) →
  1 = x - 3 →
  4 = (x + (x - 1)) - (2 * x - 5) := by
sorry

end NUMINAMATH_CALUDE_square_shape_side_length_l1476_147665


namespace NUMINAMATH_CALUDE_mass_of_man_sinking_boat_l1476_147697

/-- The mass of a man who causes a boat to sink in water -/
theorem mass_of_man_sinking_boat (length width sinkage : ℝ) (water_density : ℝ) : 
  length = 4 →
  width = 2 →
  sinkage = 0.01 →
  water_density = 1000 →
  length * width * sinkage * water_density = 80 := by
sorry

end NUMINAMATH_CALUDE_mass_of_man_sinking_boat_l1476_147697


namespace NUMINAMATH_CALUDE_bridge_length_proof_l1476_147617

/-- 
Given a train with length 120 meters crossing a bridge in 55 seconds at a speed of 39.27272727272727 m/s,
prove that the length of the bridge is 2040 meters.
-/
theorem bridge_length_proof (train_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) :
  train_length = 120 →
  crossing_time = 55 →
  train_speed = 39.27272727272727 →
  train_speed * crossing_time - train_length = 2040 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l1476_147617


namespace NUMINAMATH_CALUDE_quadratic_completion_square_l1476_147658

theorem quadratic_completion_square (x : ℝ) : 
  (2 * x^2 + 3 * x + 1 = 0) ↔ (2 * (x + 3/4)^2 - 1/8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_square_l1476_147658
