import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1233_123349

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l1233_123349


namespace NUMINAMATH_CALUDE_process_600_parts_l1233_123343

/-- Linear regression equation relating parts processed to time spent -/
def linear_regression (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem stating that processing 600 parts takes 6.5 hours -/
theorem process_600_parts : linear_regression 600 = 6.5 := by sorry

end NUMINAMATH_CALUDE_process_600_parts_l1233_123343


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l1233_123311

theorem binomial_coefficient_two (n : ℕ) (h : n ≥ 2) : Nat.choose n 2 = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l1233_123311


namespace NUMINAMATH_CALUDE_simplify_fraction_l1233_123342

theorem simplify_fraction : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1233_123342


namespace NUMINAMATH_CALUDE_second_business_owner_donation_l1233_123325

theorem second_business_owner_donation
  (num_cakes : ℕ)
  (slices_per_cake : ℕ)
  (price_per_slice : ℚ)
  (first_donation_per_slice : ℚ)
  (total_raised : ℚ)
  (h1 : num_cakes = 10)
  (h2 : slices_per_cake = 8)
  (h3 : price_per_slice = 1)
  (h4 : first_donation_per_slice = 1/2)
  (h5 : total_raised = 140) :
  let total_slices := num_cakes * slices_per_cake
  let sales_revenue := total_slices * price_per_slice
  let first_donation := total_slices * first_donation_per_slice
  let second_donation := total_raised - (sales_revenue + first_donation)
  second_donation / total_slices = 1/4 := by
sorry

end NUMINAMATH_CALUDE_second_business_owner_donation_l1233_123325


namespace NUMINAMATH_CALUDE_yoongi_score_l1233_123348

theorem yoongi_score (yoongi eunji yuna : ℕ) 
  (h1 : eunji = yoongi - 25)
  (h2 : yuna = eunji - 20)
  (h3 : yuna = 8) :
  yoongi = 53 := by
  sorry

end NUMINAMATH_CALUDE_yoongi_score_l1233_123348


namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l1233_123392

/-- 
Given a triangle with sides a, b, c, angles α, β, γ, semi-perimeter p, inradius r, and circumradius R,
this theorem states two trigonometric identities related to the triangle.
-/
theorem triangle_trigonometric_identities 
  (a b c : ℝ) 
  (α β γ : ℝ) 
  (p r R : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b)
  (h_angles : α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = Real.pi)
  (h_semi_perimeter : p = (a + b + c) / 2)
  (h_inradius : r > 0)
  (h_circumradius : R > 0) :
  (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = (p^2 - r^2 - 4*r*R) / (2*R^2) ∧
  4*R^2 * Real.cos α * Real.cos β * Real.cos γ = p^2 - (2*R + r)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l1233_123392


namespace NUMINAMATH_CALUDE_coffee_syrup_combinations_l1233_123308

theorem coffee_syrup_combinations :
  let coffee_types : ℕ := 5
  let syrup_types : ℕ := 7
  let syrup_choices : ℕ := 3
  coffee_types * (syrup_types.choose syrup_choices) = 175 :=
by sorry

end NUMINAMATH_CALUDE_coffee_syrup_combinations_l1233_123308


namespace NUMINAMATH_CALUDE_tyson_race_time_l1233_123359

/-- Calculates the total time Tyson spent in his races given his swimming speeds and race conditions. -/
theorem tyson_race_time (lake_speed ocean_speed : ℝ) (total_races : ℕ) (race_distance : ℝ) : 
  lake_speed = 3 → 
  ocean_speed = 2.5 → 
  total_races = 10 → 
  race_distance = 3 → 
  (total_races / 2 : ℝ) * (race_distance / lake_speed) + 
  (total_races / 2 : ℝ) * (race_distance / ocean_speed) = 11 := by
  sorry

#check tyson_race_time

end NUMINAMATH_CALUDE_tyson_race_time_l1233_123359


namespace NUMINAMATH_CALUDE_distribute_negative_three_l1233_123383

theorem distribute_negative_three (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_distribute_negative_three_l1233_123383


namespace NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l1233_123367

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 9 = 0

-- Define the ellipse C
def ellipse_C (x y θ : ℝ) : Prop := x = 2 * Real.sqrt 3 * Real.cos θ ∧ y = Real.sqrt 3 * Real.sin θ

-- Define the foci
def F₁ : ℝ × ℝ := (-3, 0)
def F₂ : ℝ × ℝ := (3, 0)

-- Define a point on the line l
def point_on_l (M : ℝ × ℝ) : Prop := line_l M.1 M.2

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 45 + y^2 / 36 = 1

-- Theorem statement
theorem shortest_major_axis_ellipse :
  ∀ (M : ℝ × ℝ), point_on_l M →
  ∀ (x y : ℝ), ellipse_equation x y →
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
    (x^2 / a^2 + y^2 / b^2 = 1) ∧
    (∀ (a' b' : ℝ), a' > b' ∧ b' > 0 →
      (x^2 / a'^2 + y^2 / b'^2 = 1) →
      point_on_l (x, y) →
      a ≤ a') :=
by sorry

end NUMINAMATH_CALUDE_shortest_major_axis_ellipse_l1233_123367


namespace NUMINAMATH_CALUDE_division_remainder_theorem_l1233_123371

theorem division_remainder_theorem (a b : ℕ) :
  (∃ (q r : ℕ), a^2 + b^2 = (a + b) * q + r ∧ q^2 + r = 1977) →
  ((a = 37 ∧ b = 50) ∨ (a = 50 ∧ b = 37) ∨ (a = 7 ∧ b = 50) ∨ (a = 50 ∧ b = 7)) :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_theorem_l1233_123371


namespace NUMINAMATH_CALUDE_set_A_equivalent_range_of_a_l1233_123354

-- Define set A
def A : Set ℝ := {x | (3*x - 5)/(x + 1) ≤ 1}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}

-- Theorem for part 1
theorem set_A_equivalent : A = {x : ℝ | -1 < x ∧ x ≤ 3} := by sorry

-- Theorem for part 2
theorem range_of_a (a : ℝ) : B a ∩ (Set.univ \ A) = B a → a ≤ -2 ∨ a > 4 := by sorry

end NUMINAMATH_CALUDE_set_A_equivalent_range_of_a_l1233_123354


namespace NUMINAMATH_CALUDE_quadratic_vertex_form_l1233_123358

theorem quadratic_vertex_form (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_form_l1233_123358


namespace NUMINAMATH_CALUDE_complex_number_sum_of_parts_l1233_123320

theorem complex_number_sum_of_parts (a : ℝ) :
  let z : ℂ := a / (2 + Complex.I) + (2 + Complex.I) / 5
  (z.re + z.im = 1) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_of_parts_l1233_123320


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1233_123369

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x ≠ 5 ∧ x^2 - 4*x - 5 = 0) ∧ 
  (∀ x : ℝ, x = 5 → x^2 - 4*x - 5 = 0) := by
  sorry

#check sufficient_not_necessary

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1233_123369


namespace NUMINAMATH_CALUDE_fraction_product_l1233_123384

theorem fraction_product : (3 : ℚ) / 7 * 5 / 8 * 9 / 13 * 11 / 17 = 1485 / 12376 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l1233_123384


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1233_123368

/-- Given a geometric sequence with first term b₁ = 2, 
    the minimum value of 3b₂ + 4b₃ is -9/8 -/
theorem min_value_geometric_sequence (b₁ b₂ b₃ : ℝ) :
  b₁ = 2 →
  (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) →
  (∀ c₂ c₃ : ℝ, (∃ s : ℝ, c₂ = 2 * s ∧ c₃ = 2 * s^2) → 
    3 * b₂ + 4 * b₃ ≤ 3 * c₂ + 4 * c₃) →
  3 * b₂ + 4 * b₃ = -9/8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1233_123368


namespace NUMINAMATH_CALUDE_complex_modulus_evaluation_l1233_123364

theorem complex_modulus_evaluation :
  Complex.abs (3 / 4 - 5 * Complex.I + (1 + 3 * Complex.I)) = Real.sqrt 113 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_evaluation_l1233_123364


namespace NUMINAMATH_CALUDE_valid_triples_equal_solution_set_l1233_123373

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(23, 24, 30), (12, 30, 31), (9, 18, 40), (9, 30, 32), (4, 15, 42), (15, 22, 36), (4, 30, 33)}

theorem valid_triples_equal_solution_set :
  {(a, b, c) | is_valid_triple a b c} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_valid_triples_equal_solution_set_l1233_123373


namespace NUMINAMATH_CALUDE_lcm_12_18_30_l1233_123317

theorem lcm_12_18_30 : Nat.lcm 12 (Nat.lcm 18 30) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_30_l1233_123317


namespace NUMINAMATH_CALUDE_gcd_of_779_209_589_l1233_123398

theorem gcd_of_779_209_589 : Nat.gcd 779 (Nat.gcd 209 589) = 19 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_779_209_589_l1233_123398


namespace NUMINAMATH_CALUDE_function_root_iff_a_range_l1233_123300

/-- The function f(x) = 2ax - a + 3 has a root in (-1, 1) if and only if a ∈ (-∞, -3) ∪ (1, +∞) -/
theorem function_root_iff_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-1) 1 ∧ 2 * a * x₀ - a + 3 = 0) ↔ 
  a ∈ Set.Iic (-3) ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_function_root_iff_a_range_l1233_123300


namespace NUMINAMATH_CALUDE_oil_change_cost_l1233_123352

/-- Calculates the cost of each oil change given the specified conditions. -/
theorem oil_change_cost
  (miles_per_month : ℕ)
  (miles_per_oil_change : ℕ)
  (free_oil_changes_per_year : ℕ)
  (yearly_oil_change_cost : ℕ)
  (h1 : miles_per_month = 1000)
  (h2 : miles_per_oil_change = 3000)
  (h3 : free_oil_changes_per_year = 1)
  (h4 : yearly_oil_change_cost = 150) :
  yearly_oil_change_cost / (miles_per_month * 12 / miles_per_oil_change - free_oil_changes_per_year) = 50 := by
  sorry

end NUMINAMATH_CALUDE_oil_change_cost_l1233_123352


namespace NUMINAMATH_CALUDE_quadratic_set_single_element_l1233_123329

theorem quadratic_set_single_element (k : ℝ) :
  (∃! x : ℝ, k * x^2 + 4 * x + 4 = 0) → k = 0 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_set_single_element_l1233_123329


namespace NUMINAMATH_CALUDE_consecutive_integers_divisibility_l1233_123345

theorem consecutive_integers_divisibility (a₁ a₂ a₃ : ℕ) :
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₂ = a₁ + 1 ∧ a₃ = a₂ + 1 →
  a₂^3 ∣ (a₁ * a₂ * a₃ + a₂) :=
by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisibility_l1233_123345


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1233_123330

theorem tangent_line_to_circle (m : ℝ) : 
  (∃ (x y : ℝ), x - Real.sqrt 3 * y + m = 0 ∧ x^2 + y^2 - 2*y - 2 = 0) →
  (m = -Real.sqrt 3 ∨ m = 3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1233_123330


namespace NUMINAMATH_CALUDE_complex_determinant_equation_l1233_123385

def determinant (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_determinant_equation : 
  ∃ z : ℂ, determinant z (-Complex.I) (1 - Complex.I) (1 + Complex.I) = 0 ∧ z = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_determinant_equation_l1233_123385


namespace NUMINAMATH_CALUDE_square_sum_lower_bound_l1233_123360

theorem square_sum_lower_bound (x y : ℝ) (h : |x - 2*y| = 5) : x^2 + y^2 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_lower_bound_l1233_123360


namespace NUMINAMATH_CALUDE_dogwood_trees_remaining_l1233_123326

theorem dogwood_trees_remaining (trees_part1 trees_part2 trees_to_cut : ℝ) 
  (h1 : trees_part1 = 5.0)
  (h2 : trees_part2 = 4.0)
  (h3 : trees_to_cut = 7.0) :
  trees_part1 + trees_part2 - trees_to_cut = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_trees_remaining_l1233_123326


namespace NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l1233_123363

/-- Represents the fraction of orange juice in a mixture of two pitchers -/
def orange_juice_fraction (capacity1 capacity2 : ℚ) (fraction1 fraction2 : ℚ) : ℚ :=
  (capacity1 * fraction1 + capacity2 * fraction2) / (capacity1 + capacity2)

/-- Theorem stating that the fraction of orange juice in the given mixture is 17/52 -/
theorem orange_juice_mixture_fraction :
  orange_juice_fraction 500 800 (1/4) (3/8) = 17/52 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_mixture_fraction_l1233_123363


namespace NUMINAMATH_CALUDE_water_depth_calculation_l1233_123355

def water_depth (ron_height dean_height_difference : ℝ) : ℝ :=
  let dean_height := ron_height - dean_height_difference
  2.5 * dean_height + 3

theorem water_depth_calculation (ron_height dean_height_difference : ℝ) 
  (h1 : ron_height = 14.2)
  (h2 : dean_height_difference = 8.3) :
  water_depth ron_height dean_height_difference = 17.75 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_calculation_l1233_123355


namespace NUMINAMATH_CALUDE_simultaneous_colonies_count_l1233_123302

/-- Represents the growth of bacteria colonies over time -/
def bacteriaGrowth (n : ℕ) (t : ℕ) : ℕ := n * 2^t

/-- The number of days it takes for a single colony to reach the habitat limit -/
def singleColonyLimit : ℕ := 25

/-- The number of days it takes for multiple colonies to reach the habitat limit -/
def multipleColoniesLimit : ℕ := 24

/-- Theorem stating that the number of simultaneously growing colonies is 2 -/
theorem simultaneous_colonies_count :
  ∃ (n : ℕ), n > 0 ∧ 
    bacteriaGrowth n multipleColoniesLimit = bacteriaGrowth 1 singleColonyLimit ∧ 
    n = 2 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_colonies_count_l1233_123302


namespace NUMINAMATH_CALUDE_square_diff_equality_l1233_123332

theorem square_diff_equality (x y A : ℝ) : 
  (2*x - y)^2 + A = (2*x + y)^2 → A = 8*x*y := by
  sorry

end NUMINAMATH_CALUDE_square_diff_equality_l1233_123332


namespace NUMINAMATH_CALUDE_power_equation_solution_l1233_123370

theorem power_equation_solution (n : Real) : 
  10^n = 10^4 * Real.sqrt (10^155 / 0.0001) → n = 83.5 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1233_123370


namespace NUMINAMATH_CALUDE_group_materials_calculation_l1233_123387

-- Define the given quantities
def teacher_materials : ℕ := 28
def total_products : ℕ := 93

-- Define the function to calculate group materials
def group_materials : ℕ := total_products - teacher_materials

-- Theorem statement
theorem group_materials_calculation :
  group_materials = 65 :=
sorry

end NUMINAMATH_CALUDE_group_materials_calculation_l1233_123387


namespace NUMINAMATH_CALUDE_triangle_property_triangle_area_l1233_123301

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  angleSum : A + B + C = π
  positiveSides : 0 < a ∧ 0 < b ∧ 0 < c
  positiveAngles : 0 < A ∧ 0 < B ∧ 0 < C

theorem triangle_property (t : Triangle) 
  (h : Real.sin t.B * (Real.tan t.A + Real.tan t.C) = Real.tan t.A * Real.tan t.C) :
  t.b^2 = t.a * t.c :=
sorry

theorem triangle_area (t : Triangle) (h1 : t.a = 2 * t.c) (h2 : t.a = 2) :
  (1/2 : ℝ) * t.a * t.c * Real.sin t.B = Real.sqrt 7 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_property_triangle_area_l1233_123301


namespace NUMINAMATH_CALUDE_divisors_of_216n4_l1233_123386

/-- Number of positive integer divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_216n4 (n : ℕ) (h : n > 0) (h240 : num_divisors (240 * n^3) = 240) : 
  num_divisors (216 * n^4) = 156 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_216n4_l1233_123386


namespace NUMINAMATH_CALUDE_longest_chord_implies_a_equals_one_l1233_123334

/-- The line equation ax + y - 2 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y - 2 = 0

/-- The circle equation (x-1)^2 + (y-a)^2 = 4 -/
def circle_equation (a x y : ℝ) : Prop := (x - 1)^2 + (y - a)^2 = 4

/-- A point (x, y) is on the circle -/
def point_on_circle (a x y : ℝ) : Prop := circle_equation a x y

/-- A point (x, y) is on the line -/
def point_on_line (a x y : ℝ) : Prop := line_equation a x y

/-- The theorem to be proved -/
theorem longest_chord_implies_a_equals_one (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    point_on_circle a x₁ y₁ ∧
    point_on_circle a x₂ y₂ ∧
    point_on_line a x₁ y₁ ∧
    point_on_line a x₂ y₂ ∧
    ∀ x y : ℝ, point_on_circle a x y → (x₂ - x₁)^2 + (y₂ - y₁)^2 ≥ (x - x₁)^2 + (y - y₁)^2) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_longest_chord_implies_a_equals_one_l1233_123334


namespace NUMINAMATH_CALUDE_equation_solution_l1233_123361

theorem equation_solution :
  ∃ x : ℚ, (2 / 3 + 1 / x = 7 / 9) ∧ (x = 9) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1233_123361


namespace NUMINAMATH_CALUDE_function_inequality_range_l1233_123316

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

theorem function_inequality_range (a : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → 
    |f a x₁ - f a x₂| ≤ a - 2) ↔ 
  a ∈ Set.Ici (Real.exp 2) :=
sorry

end NUMINAMATH_CALUDE_function_inequality_range_l1233_123316


namespace NUMINAMATH_CALUDE_recurrence_sequence_is_natural_l1233_123340

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ+ → ℝ) : Prop :=
  a 2 = 2 ∧ ∀ n : ℕ+, (n - 1 : ℝ) * a (n + 1) - n * a n + 1 = 0

/-- The theorem stating that the sequence is equal to the natural numbers -/
theorem recurrence_sequence_is_natural (a : ℕ+ → ℝ) (h : RecurrenceSequence a) :
  ∀ n : ℕ+, a n = n := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_is_natural_l1233_123340


namespace NUMINAMATH_CALUDE_train_speed_problem_l1233_123339

theorem train_speed_problem (initial_distance : ℝ) (speed1 : ℝ) (distance_before_meeting : ℝ) :
  initial_distance = 120 →
  speed1 = 30 →
  distance_before_meeting = 70 →
  ∃ speed2 : ℝ, 
    speed2 = 40 ∧
    initial_distance - distance_before_meeting = speed1 + speed2 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l1233_123339


namespace NUMINAMATH_CALUDE_property_value_calculation_l1233_123350

/-- Calculate the total value of a property with different types of buildings --/
theorem property_value_calculation (condo_price condo_area barn_price barn_area 
  detached_price detached_area garage_price garage_area : ℕ) : 
  condo_price = 98 → 
  condo_area = 2400 → 
  barn_price = 84 → 
  barn_area = 1200 → 
  detached_price = 102 → 
  detached_area = 3500 → 
  garage_price = 60 → 
  garage_area = 480 → 
  (condo_price * condo_area + barn_price * barn_area + 
   detached_price * detached_area + garage_price * garage_area) = 721800 := by
  sorry

end NUMINAMATH_CALUDE_property_value_calculation_l1233_123350


namespace NUMINAMATH_CALUDE_systematic_sampling_first_number_l1233_123351

theorem systematic_sampling_first_number 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (last_sample : ℕ) 
  (h1 : population_size = 2000)
  (h2 : sample_size = 100)
  (h3 : last_sample = 1994)
  (h4 : last_sample < population_size) :
  let interval := population_size / sample_size
  let first_sample := last_sample - (sample_size - 1) * interval
  first_sample = 14 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_first_number_l1233_123351


namespace NUMINAMATH_CALUDE_complement_P_union_Q_l1233_123347

def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x * (x - 2) < 0}

theorem complement_P_union_Q : 
  (U \ (P ∪ Q)) = {x : ℝ | x ≤ 0} := by sorry

end NUMINAMATH_CALUDE_complement_P_union_Q_l1233_123347


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1233_123372

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 3*x + 5) % (x - 1) = 3 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1233_123372


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l1233_123396

/-- 
Given two cylinders with the same height and radii in the ratio 1:3,
if the volume of the larger cylinder is 360 cc, then the volume of the smaller cylinder is 40 cc.
-/
theorem cylinder_volume_ratio (h : ℝ) (r : ℝ) : 
  h > 0 → r > 0 → π * (3 * r)^2 * h = 360 → π * r^2 * h = 40 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l1233_123396


namespace NUMINAMATH_CALUDE_correct_average_weight_l1233_123322

theorem correct_average_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) :
  n = 20 →
  initial_average = 58.4 →
  misread_weight = 56 →
  correct_weight = 61 →
  (n * initial_average + (correct_weight - misread_weight)) / n = 58.65 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_weight_l1233_123322


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1233_123315

/-- The largest single-digit number N such that 5678N is divisible by 6 is 4 -/
theorem largest_digit_divisible_by_six : 
  (∀ N : ℕ, N ≤ 9 → 56780 + N ≤ 56789 → (56780 + N) % 6 = 0 → N ≤ 4) ∧ 
  (56784 % 6 = 0) := by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1233_123315


namespace NUMINAMATH_CALUDE_at_least_one_geq_two_l1233_123362

theorem at_least_one_geq_two (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_geq_two_l1233_123362


namespace NUMINAMATH_CALUDE_minimum_agreement_for_budget_constraint_l1233_123323

/-- Represents a parliament budget allocation problem -/
structure ParliamentBudget where
  members : ℕ
  items : ℕ
  limit : ℝ

/-- Defines the minimum number of members required for agreement -/
def min_agreement (pb : ParliamentBudget) : ℕ := pb.members - pb.items + 1

/-- Theorem stating the minimum agreement required for the given problem -/
theorem minimum_agreement_for_budget_constraint 
  (pb : ParliamentBudget) 
  (h_members : pb.members = 2000) 
  (h_items : pb.items = 200) :
  min_agreement pb = 1991 := by
  sorry

#eval min_agreement { members := 2000, items := 200, limit := 0 }

end NUMINAMATH_CALUDE_minimum_agreement_for_budget_constraint_l1233_123323


namespace NUMINAMATH_CALUDE_farm_animals_relation_l1233_123307

/-- Given a farm with pigs, cows, and goats, prove the relationship between the number of goats and cows -/
theorem farm_animals_relation (pigs cows goats : ℕ) : 
  pigs = 10 →
  cows = 2 * pigs - 3 →
  pigs + cows + goats = 50 →
  goats = cows + 6 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_relation_l1233_123307


namespace NUMINAMATH_CALUDE_hypotenuse_squared_is_40_l1233_123305

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := 9 * x^2 + y^2 = 36

-- Define the right triangle
structure RightTriangle where
  A : ℝ × ℝ -- right angle vertex
  B : ℝ × ℝ -- vertex on x-axis
  C : ℝ × ℝ -- vertex on y-axis
  on_ellipse : ellipse B.1 B.2 ∧ ellipse C.1 C.2
  right_angle_at_A : A = (0, -6)
  B_on_x_axis : B.2 = 0
  C_on_y_axis : C.1 = 0

-- Theorem statement
theorem hypotenuse_squared_is_40 (t : RightTriangle) : 
  (t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_hypotenuse_squared_is_40_l1233_123305


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1233_123366

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 930 →
  margin = 360 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 62 / 100 := by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1233_123366


namespace NUMINAMATH_CALUDE_vector_equality_conditions_l1233_123365

theorem vector_equality_conditions (n : ℕ) :
  ∃ (a b : Fin n → ℝ),
    (norm a = norm b ∧ norm (a + b) ≠ norm (a - b)) ∧
    ∃ (c d : Fin n → ℝ),
      (norm (c + d) = norm (c - d) ∧ norm c ≠ norm d) :=
by sorry

end NUMINAMATH_CALUDE_vector_equality_conditions_l1233_123365


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1233_123336

/-- Given a geometric sequence with positive terms and common ratio q,
    where S_n denotes the sum of the first n terms, prove that
    if 2^10 * S_30 + S_10 = (2^10 + 1) * S_20, then q = 1/2 -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_positive : ∀ n, a n > 0)
  (h_geometric : ∀ n, a (n + 1) = q * a n)
  (S : ℕ → ℝ)
  (h_sum : ∀ n, S n = (a 0) * (1 - q^n) / (1 - q))
  (h_equation : 2^10 * S 30 + S 10 = (2^10 + 1) * S 20) :
  q = 1/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1233_123336


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l1233_123389

theorem complex_fraction_sum (a b : ℝ) : 
  (1 + 2*I) / (1 + I) = a + b*I → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l1233_123389


namespace NUMINAMATH_CALUDE_certain_number_problem_l1233_123306

theorem certain_number_problem :
  ∃! x : ℝ,
    (28 + x + 42 + 78 + 104) / 5 = 62 ∧
    (48 + 62 + 98 + 124 + x) / 5 = 78 ∧
    x = 58 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1233_123306


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_bounded_zeros_product_lt_one_l1233_123341

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x / x - Real.log x + x - a

theorem f_nonnegative_iff_a_bounded (a : ℝ) :
  (∀ x > 0, f a x ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
sorry

theorem zeros_product_lt_one (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (hz₁ : f a x₁ = 0) (hz₂ : f a x₂ = 0) :
  x₁ * x₂ < 1 :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_bounded_zeros_product_lt_one_l1233_123341


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1233_123319

theorem complex_equation_solution :
  ∃ (z : ℂ), ∃ (a b : ℝ),
    z = Complex.mk a b ∧
    z * (z + Complex.I) * (z + 2 * Complex.I) = 1800 * Complex.I ∧
    a = 20.75 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1233_123319


namespace NUMINAMATH_CALUDE_terminal_side_in_third_quadrant_l1233_123376

def angle_in_radians : ℝ := 4

theorem terminal_side_in_third_quadrant (angle : ℝ) (h : angle = angle_in_radians) :
  ∃ θ : ℝ, 180 < θ ∧ θ < 270 ∧ θ = angle * (180 / Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_terminal_side_in_third_quadrant_l1233_123376


namespace NUMINAMATH_CALUDE_correct_stratified_sample_size_l1233_123382

/-- Represents the student population in each year -/
structure StudentPopulation where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ

/-- Calculates the stratified sample size for each year -/
def stratifiedSampleSize (population : StudentPopulation) (total_sample : ℕ) : ℕ × ℕ × ℕ :=
  sorry

/-- Theorem stating that the calculated stratified sample sizes are correct -/
theorem correct_stratified_sample_size 
  (population : StudentPopulation)
  (h1 : population.first_year = 540)
  (h2 : population.second_year = 440)
  (h3 : population.third_year = 420)
  (total_sample : ℕ)
  (h4 : total_sample = 70) :
  stratifiedSampleSize population total_sample = (27, 22, 21) :=
sorry

end NUMINAMATH_CALUDE_correct_stratified_sample_size_l1233_123382


namespace NUMINAMATH_CALUDE_remainder_equality_l1233_123377

theorem remainder_equality (A B D S S' : ℕ) (hA : A > B) :
  A % D = S →
  B % D = S' →
  (A + B) % D = (S + S') % D :=
by sorry

end NUMINAMATH_CALUDE_remainder_equality_l1233_123377


namespace NUMINAMATH_CALUDE_evaluate_otimes_expression_l1233_123318

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := (a^2 + b^2) / (a - b)

-- State the theorem
theorem evaluate_otimes_expression :
  (otimes (otimes 5 3) 2) = 293/15 := by sorry

end NUMINAMATH_CALUDE_evaluate_otimes_expression_l1233_123318


namespace NUMINAMATH_CALUDE_candy_chocolate_cost_difference_l1233_123390

/-- The cost difference between a candy bar and a chocolate -/
def cost_difference (candy_bar_cost chocolate_cost : ℕ) : ℕ :=
  candy_bar_cost - chocolate_cost

/-- Theorem: The cost difference between a $7 candy bar and a $3 chocolate is $4 -/
theorem candy_chocolate_cost_difference :
  cost_difference 7 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_candy_chocolate_cost_difference_l1233_123390


namespace NUMINAMATH_CALUDE_smallest_m_plus_n_for_divisibility_existence_of_smallest_m_plus_n_l1233_123304

theorem smallest_m_plus_n_for_divisibility (m n : ℕ) : 
  m > n → n ≥ 1 → (1000 ∣ 1978^m - 1978^n) → m + n ≥ 106 :=
by sorry

theorem existence_of_smallest_m_plus_n : 
  ∃ m n : ℕ, m > n ∧ n ≥ 1 ∧ (1000 ∣ 1978^m - 1978^n) ∧ m + n = 106 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_plus_n_for_divisibility_existence_of_smallest_m_plus_n_l1233_123304


namespace NUMINAMATH_CALUDE_derivative_ln_over_x_l1233_123313

open Real

theorem derivative_ln_over_x (x : ℝ) (h : x > 0) :
  deriv (λ x => (log x) / x) x = (1 - log x) / x^2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_ln_over_x_l1233_123313


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1233_123379

theorem smallest_part_of_proportional_division (x y z : ℚ) :
  x + y + z = 64 ∧ 
  y = 2 * x ∧ 
  z = 3 * x →
  x = 32 / 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l1233_123379


namespace NUMINAMATH_CALUDE_team_win_percentage_l1233_123324

theorem team_win_percentage (total_games : ℕ) (wins_first_100 : ℕ) 
  (h1 : total_games ≥ 100)
  (h2 : wins_first_100 ≤ 100)
  (h3 : (wins_first_100 : ℝ) / 100 + (0.5 * (total_games - 100) : ℝ) / total_games = 0.7) :
  wins_first_100 = 70 := by
sorry

end NUMINAMATH_CALUDE_team_win_percentage_l1233_123324


namespace NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l1233_123353

theorem permutations_of_six_distinct_objects : Nat.factorial 6 = 720 := by
  sorry

end NUMINAMATH_CALUDE_permutations_of_six_distinct_objects_l1233_123353


namespace NUMINAMATH_CALUDE_sum_integer_chord_lengths_equals_40_l1233_123327

/-- A circle with center O and a point P inside it. -/
structure CircleWithPoint where
  O : Point    -- Center of the circle
  P : Point    -- Point inside the circle
  radius : ℝ   -- Radius of the circle
  OP : ℝ       -- Distance between O and P

/-- The sum of all possible integer chord lengths passing through P -/
def sumIntegerChordLengths (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem -/
theorem sum_integer_chord_lengths_equals_40 (c : CircleWithPoint) 
  (h_radius : c.radius = 5)
  (h_OP : c.OP = 4) :
  sumIntegerChordLengths c = 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_integer_chord_lengths_equals_40_l1233_123327


namespace NUMINAMATH_CALUDE_max_value_theorem_l1233_123391

theorem max_value_theorem (a b c : ℝ) (h : 9 * a^2 + 4 * b^2 + 25 * c^2 = 1) :
  (∃ x y z : ℝ, 9 * x^2 + 4 * y^2 + 25 * z^2 = 1 ∧ 8 * x + 3 * y + 5 * z > 8 * a + 3 * b + 5 * c) →
  8 * a + 3 * b + 5 * c ≤ Real.sqrt (373 / 36) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1233_123391


namespace NUMINAMATH_CALUDE_last_digit_of_sum_l1233_123312

theorem last_digit_of_sum (n : ℕ) : (2^1992 + 3^1992) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_sum_l1233_123312


namespace NUMINAMATH_CALUDE_least_consecutive_primes_l1233_123356

/-- Definition of the sequence x_n -/
def x (a b n : ℕ) : ℚ :=
  (a^n - 1) / (b^n - 1)

/-- Main theorem statement -/
theorem least_consecutive_primes (a b : ℕ) (h1 : a > b) (h2 : b > 1) :
  ∃ d : ℕ, d = 3 ∧
  (∀ n : ℕ, ¬(Prime (x a b n) ∧ Prime (x a b (n+1)) ∧ Prime (x a b (n+2)))) ∧
  (∀ d' : ℕ, d' < d →
    ∃ a' b' n' : ℕ, a' > b' ∧ b' > 1 ∧
      Prime (x a' b' n') ∧ Prime (x a' b' (n'+1)) ∧
      (d' = 2 → Prime (x a' b' (n'+2)))) :=
sorry

end NUMINAMATH_CALUDE_least_consecutive_primes_l1233_123356


namespace NUMINAMATH_CALUDE_melissa_total_points_l1233_123380

/-- The number of points Melissa scores in each game -/
def points_per_game : ℕ := 120

/-- The number of games played -/
def num_games : ℕ := 10

/-- The total points scored by Melissa in all games -/
def total_points : ℕ := points_per_game * num_games

theorem melissa_total_points : total_points = 1200 := by
  sorry

end NUMINAMATH_CALUDE_melissa_total_points_l1233_123380


namespace NUMINAMATH_CALUDE_interception_time_correct_l1233_123399

/-- Represents the time of day in hours and minutes -/
structure TimeOfDay where
  hours : Nat
  minutes : Nat

/-- Represents the naval pursuit scenario -/
structure NavalPursuit where
  initialDistance : Real
  initialTime : TimeOfDay
  destroyerInitialSpeed : Real
  cargoShipSpeed : Real
  speedChangeTime : Real
  destroyerReducedSpeed : Real

/-- Calculates the time of interception given the naval pursuit scenario -/
def timeOfInterception (scenario : NavalPursuit) : Real :=
  sorry

/-- Theorem stating that the time of interception is 3 hours and 40 minutes after the initial time -/
theorem interception_time_correct (scenario : NavalPursuit) 
  (h1 : scenario.initialDistance = 20)
  (h2 : scenario.initialTime = ⟨9, 0⟩)
  (h3 : scenario.destroyerInitialSpeed = 16)
  (h4 : scenario.cargoShipSpeed = 10)
  (h5 : scenario.speedChangeTime = 3)
  (h6 : scenario.destroyerReducedSpeed = 13) :
  timeOfInterception scenario = 3 + 40 / 60 := by
  sorry

end NUMINAMATH_CALUDE_interception_time_correct_l1233_123399


namespace NUMINAMATH_CALUDE_fencing_required_for_rectangular_field_l1233_123375

/-- Given a rectangular field with one side uncovered, this theorem calculates the amount of fencing required. -/
theorem fencing_required_for_rectangular_field 
  (area : ℝ) 
  (uncovered_side : ℝ) 
  (h1 : area > 0) 
  (h2 : uncovered_side > 0) 
  (h3 : area = uncovered_side * (area / uncovered_side)) :
  2 * (area / uncovered_side) + uncovered_side = 
    (2 * area / uncovered_side) + uncovered_side := by
  sorry

#check fencing_required_for_rectangular_field

end NUMINAMATH_CALUDE_fencing_required_for_rectangular_field_l1233_123375


namespace NUMINAMATH_CALUDE_factorial_division_l1233_123337

theorem factorial_division (h : Nat.factorial 10 = 3628800) :
  Nat.factorial 10 / Nat.factorial 5 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l1233_123337


namespace NUMINAMATH_CALUDE_cityF_greatest_increase_l1233_123303

/-- Represents a city with population data for 1970 and 1980 --/
structure City where
  name : String
  pop1970 : Nat
  pop1980 : Nat

/-- Calculates the percentage increase in population from 1970 to 1980 --/
def percentageIncrease (city : City) : Rat :=
  (city.pop1980 - city.pop1970 : Rat) / city.pop1970 * 100

/-- The set of cities in the region --/
def cities : Finset City := sorry

/-- City F with its population data --/
def cityF : City := { name := "F", pop1970 := 30000, pop1980 := 45000 }

/-- City G with its population data --/
def cityG : City := { name := "G", pop1970 := 60000, pop1980 := 75000 }

/-- Combined City H (including I) with its population data --/
def cityH : City := { name := "H", pop1970 := 60000, pop1980 := 70000 }

/-- City J with its population data --/
def cityJ : City := { name := "J", pop1970 := 90000, pop1980 := 120000 }

/-- Theorem stating that City F had the greatest percentage increase --/
theorem cityF_greatest_increase : 
  ∀ city ∈ cities, percentageIncrease cityF ≥ percentageIncrease city :=
sorry

end NUMINAMATH_CALUDE_cityF_greatest_increase_l1233_123303


namespace NUMINAMATH_CALUDE_triangle_special_case_l1233_123314

/-- Given a triangle with sides a, b, and c satisfying (a + b + c)(a + b - c) = 4ab,
    the angle opposite side c is 0 or 2π. -/
theorem triangle_special_case (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  C = 0 ∨ C = 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_triangle_special_case_l1233_123314


namespace NUMINAMATH_CALUDE_expression_simplification_l1233_123374

theorem expression_simplification (x : ℝ) : 
  x * (x * (x * (x - 3) - 5) + 11) + 2 = x^4 - 3*x^3 - 5*x^2 + 11*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1233_123374


namespace NUMINAMATH_CALUDE_b_range_characterization_l1233_123331

theorem b_range_characterization :
  ∀ b : ℝ, (0 < b ∧ b ≤ 1/4) ↔ (b > 0 ∧ ∀ x : ℝ, |x - 5/4| < b → |x - 1| < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_b_range_characterization_l1233_123331


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1233_123321

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, x > 3 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1233_123321


namespace NUMINAMATH_CALUDE_sages_can_succeed_l1233_123388

/-- Represents the color of a hat -/
def HatColor := Fin 1000

/-- Represents the signal a sage can show (white or black card) -/
def Signal := Bool

/-- Represents the configuration of hats on the sages -/
def HatConfiguration := Fin 11 → HatColor

/-- A strategy is a function that takes the colors of the other hats and returns a signal -/
def Strategy := (Fin 10 → HatColor) → Signal

/-- The result of applying a strategy is a function that determines the hat color based on the signals of others -/
def StrategyResult := (Fin 10 → Signal) → HatColor

/-- A successful strategy correctly determines the hat color for all possible configurations -/
def SuccessfulStrategy (strategy : Fin 11 → Strategy) (result : Fin 11 → StrategyResult) : Prop :=
  ∀ (config : HatConfiguration),
    ∀ (i : Fin 11),
      result i (λ j => if j < i then strategy j (λ k => config (k.succ)) 
                       else strategy j.succ (λ k => if k < j then config k else config k.succ)) = config i

theorem sages_can_succeed : ∃ (strategy : Fin 11 → Strategy) (result : Fin 11 → StrategyResult),
  SuccessfulStrategy strategy result := by
  sorry

end NUMINAMATH_CALUDE_sages_can_succeed_l1233_123388


namespace NUMINAMATH_CALUDE_right_triangle_sides_from_median_perimeters_l1233_123328

/-- Given a right triangle with a median to the hypotenuse dividing it into two triangles
    with perimeters m and n, this theorem states the sides of the original triangle. -/
theorem right_triangle_sides_from_median_perimeters (m n : ℝ) 
  (h₁ : m > 0) (h₂ : n > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    ∃ (x : ℝ), x > 0 ∧
      x^2 = (a/2)^2 + (b/2)^2 ∧
      m = x + (c/2 - x) + b ∧
      n = x + (c/2 - x) + a ∧
      a = Real.sqrt (2*m*n) - m ∧
      b = Real.sqrt (2*m*n) - n ∧
      c = n + m - Real.sqrt (2*m*n) :=
by
  sorry


end NUMINAMATH_CALUDE_right_triangle_sides_from_median_perimeters_l1233_123328


namespace NUMINAMATH_CALUDE_fraction_unchanged_l1233_123357

theorem fraction_unchanged (x y : ℝ) : (5 * x) / (x + y) = (5 * (10 * x)) / ((10 * x) + (10 * y)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l1233_123357


namespace NUMINAMATH_CALUDE_home_run_multiple_l1233_123393

/-- The number of home runs hit by Hank Aaron -/
def hank_aaron_hr : ℕ := 755

/-- The number of home runs hit by Dave Winfield -/
def dave_winfield_hr : ℕ := 465

/-- The difference between Hank Aaron's home runs and the multiple of Dave Winfield's home runs -/
def hr_difference : ℕ := 175

/-- The theorem stating that the multiple of Dave Winfield's home runs that Hank Aaron's home runs are compared to is 2 -/
theorem home_run_multiple : ∃ (m : ℕ), m * dave_winfield_hr = hank_aaron_hr + hr_difference ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_home_run_multiple_l1233_123393


namespace NUMINAMATH_CALUDE_total_sessions_for_patients_l1233_123333

theorem total_sessions_for_patients : 
  let num_patients : ℕ := 4
  let first_patient_sessions : ℕ := 6
  let second_patient_sessions : ℕ := first_patient_sessions + 5
  let remaining_patients_sessions : ℕ := 8
  
  num_patients = 4 →
  first_patient_sessions + 
  second_patient_sessions + 
  (num_patients - 2) * remaining_patients_sessions = 33 := by
sorry

end NUMINAMATH_CALUDE_total_sessions_for_patients_l1233_123333


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1233_123309

/-- A circle inscribed in a convex polygon -/
structure InscribedCircle where
  /-- The circumference of the inscribed circle -/
  circle_circumference : ℝ
  /-- The perimeter of the convex polygon -/
  polygon_perimeter : ℝ
  /-- The area of the inscribed circle -/
  circle_area : ℝ
  /-- The area of the convex polygon -/
  polygon_area : ℝ

/-- Theorem stating that for a circle inscribed in a convex polygon with given circumference and perimeter,
    the ratio of the circle's area to the polygon's area is 2/3 -/
theorem inscribed_circle_area_ratio
  (ic : InscribedCircle)
  (h_circle_circumference : ic.circle_circumference = 10)
  (h_polygon_perimeter : ic.polygon_perimeter = 15) :
  ic.circle_area / ic.polygon_area = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l1233_123309


namespace NUMINAMATH_CALUDE_rectangle_strip_proof_l1233_123310

theorem rectangle_strip_proof (a b c : ℕ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a * b + a * c + a * (b - a) + a * a + a * (c - a) = 43 →
  (a = 1 ∧ b + c = 22) ∨ (a = 1 ∧ c + b = 22) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_strip_proof_l1233_123310


namespace NUMINAMATH_CALUDE_circle_triangle_area_relation_l1233_123378

theorem circle_triangle_area_relation :
  ∀ (A B C : ℝ),
  -- The triangle has sides 20, 21, and 29
  20^2 + 21^2 = 29^2 →
  -- A circle is circumscribed about the triangle
  -- A, B, and C are areas of non-triangular regions
  -- C is the largest area
  C ≥ A ∧ C ≥ B →
  -- The area of the triangle is 210
  (20 * 21) / 2 = 210 →
  -- The diameter of the circle is 29
  -- C is half the area of the circle
  C = (29^2 * π) / 8 →
  -- Prove that A + B + 210 = C
  A + B + 210 = C :=
by
  sorry

end NUMINAMATH_CALUDE_circle_triangle_area_relation_l1233_123378


namespace NUMINAMATH_CALUDE_dolls_count_l1233_123338

/-- Given that Hannah has 5 times as many dolls as her sister, and her sister has 8 dolls,
    prove that they have 48 dolls altogether. -/
theorem dolls_count (hannah_dolls : ℕ) (sister_dolls : ℕ) : 
  hannah_dolls = 5 * sister_dolls → sister_dolls = 8 → hannah_dolls + sister_dolls = 48 := by
  sorry

end NUMINAMATH_CALUDE_dolls_count_l1233_123338


namespace NUMINAMATH_CALUDE_distance_to_place_l1233_123344

/-- Calculates the distance to a place given rowing speed, current velocity, and round trip time -/
theorem distance_to_place (rowing_speed current_velocity : ℝ) (round_trip_time : ℝ) : 
  rowing_speed = 5 → 
  current_velocity = 1 → 
  round_trip_time = 1 → 
  (rowing_speed + current_velocity) * (rowing_speed - current_velocity) * round_trip_time / 
  (rowing_speed + current_velocity + rowing_speed - current_velocity) = 2.4 := by
sorry

end NUMINAMATH_CALUDE_distance_to_place_l1233_123344


namespace NUMINAMATH_CALUDE_chord_length_is_16_l1233_123394

-- Define the parabola C
def parabola_C (p : ℝ) (x y : ℝ) : Prop := x^2 = -2*p*y ∧ p > 0

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := y^2/8 + x^2/4 = 1

-- Define the focus F
def focus_F : ℝ × ℝ := (0, -2)

-- Define a line passing through F
def line_through_F (k : ℝ) (x y : ℝ) : Prop := y = k*x - 2

-- Define the intersection points A and B
def intersection_points (p k : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ),
    parabola_C p x1 y1 ∧ parabola_C p x2 y2 ∧
    line_through_F k x1 y1 ∧ line_through_F k x2 y2 ∧
    x1 ≠ x2

-- Define the tangent lines at A and B
def tangent_line (p x0 y0 x y : ℝ) : Prop :=
  y - y0 = -(x0 / (4*p)) * (x - x0)

-- Define the intersection point M of tangents
def intersection_M (p k : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 xM yM : ℝ),
    intersection_points p k ∧
    tangent_line p x1 y1 xM yM ∧ tangent_line p x2 y2 xM yM ∧
    xM = 4

-- Theorem statement
theorem chord_length_is_16 (p k : ℝ) :
  parabola_C p 0 (-2) →
  ellipse (focus_F.1) (focus_F.2) →
  intersection_M p k →
  ∃ (x1 y1 x2 y2 : ℝ),
    intersection_points p k ∧
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 16 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_16_l1233_123394


namespace NUMINAMATH_CALUDE_smallest_positive_solution_l1233_123346

theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x - 2 ∧
  ∀ (y : ℝ), y > 0 → Real.sqrt (3 * y) = 5 * y - 2 → x ≤ y ∧
  x = 4 / 25 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_solution_l1233_123346


namespace NUMINAMATH_CALUDE_hat_price_after_discounts_l1233_123335

/-- The final price of an item after two successive discounts --/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  originalPrice * (1 - discount1) * (1 - discount2)

/-- Theorem stating that a $20 item with 20% and 25% successive discounts results in a $12 final price --/
theorem hat_price_after_discounts :
  finalPrice 20 0.2 0.25 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hat_price_after_discounts_l1233_123335


namespace NUMINAMATH_CALUDE_range_of_k_equation_of_l_when_OB_twice_OA_l1233_123381

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + y^2 = 20

-- Define the line l
def line_l (k x y : ℝ) : Prop := y = k * x

-- Define the condition that line l intersects circle C at two distinct points
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂

-- Define the condition OB = 2OA
def OB_twice_OA (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₂^2 + y₂^2 = 4 * (x₁^2 + y₁^2)

-- Theorem for the range of k
theorem range_of_k (k : ℝ) :
  intersects_at_two_points k ↔ -Real.sqrt 5 / 2 < k ∧ k < Real.sqrt 5 / 2 :=
sorry

-- Theorem for the equation of line l when OB = 2OA
theorem equation_of_l_when_OB_twice_OA (k : ℝ) :
  (intersects_at_two_points k ∧
   ∃ (x₁ y₁ x₂ y₂ : ℝ), circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧ line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧ OB_twice_OA x₁ y₁ x₂ y₂)
  → k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_equation_of_l_when_OB_twice_OA_l1233_123381


namespace NUMINAMATH_CALUDE_power_division_rule_l1233_123395

theorem power_division_rule (a : ℝ) : a^7 / a^5 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1233_123395


namespace NUMINAMATH_CALUDE_prism_sides_plus_two_l1233_123397

/-- A prism is a polyhedron with two congruent parallel faces (bases) and rectangular sides. -/
structure Prism where
  sides : ℕ

/-- The number of edges in a prism. -/
def Prism.edges (p : Prism) : ℕ := 3 * p.sides

/-- The number of vertices in a prism. -/
def Prism.vertices (p : Prism) : ℕ := 2 * p.sides

/-- Theorem: For a prism where the sum of its edges and vertices is 30,
    the number of sides plus 2 equals 8. -/
theorem prism_sides_plus_two (p : Prism) 
    (h : p.edges + p.vertices = 30) : p.sides + 2 = 8 := by
  sorry


end NUMINAMATH_CALUDE_prism_sides_plus_two_l1233_123397
