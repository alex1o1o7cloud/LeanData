import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l1956_195609

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- Define the property of f not being monotonic
def not_monotonic (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, x < y ∧ y < z ∧ (f x < f y ∧ f y > f z ∨ f x > f y ∧ f y < f z)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  not_monotonic (f a) ↔ a < -Real.sqrt 3 ∨ a > Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1956_195609


namespace NUMINAMATH_CALUDE_xyz_product_l1956_195663

theorem xyz_product (x y z : ℚ) 
  (eq1 : x + y + z = 1)
  (eq2 : x + y - z = 2)
  (eq3 : x - y - z = 3) :
  x * y * z = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l1956_195663


namespace NUMINAMATH_CALUDE_frog_jump_expected_time_l1956_195664

/-- A dodecagon with vertices A₁ to A₁₂ -/
structure Dodecagon where
  vertices : Fin 12 → Point

/-- Represents the position of three frogs on the dodecagon -/
structure FrogPositions where
  frog1 : Fin 12
  frog2 : Fin 12
  frog3 : Fin 12

/-- The expected number of minutes until the frogs stop jumping -/
def expected_stop_time (d : Dodecagon) (initial : FrogPositions) : ℚ :=
  16/3

/-- Theorem stating the expected stop time for the given initial configuration -/
theorem frog_jump_expected_time 
  (d : Dodecagon) 
  (initial : FrogPositions) 
  (h1 : initial.frog1 = 4)
  (h2 : initial.frog2 = 8)
  (h3 : initial.frog3 = 12) :
  expected_stop_time d initial = 16/3 :=
sorry

end NUMINAMATH_CALUDE_frog_jump_expected_time_l1956_195664


namespace NUMINAMATH_CALUDE_popped_kernel_probability_l1956_195655

theorem popped_kernel_probability (white_ratio : ℚ) (yellow_ratio : ℚ) 
  (white_pop_prob : ℚ) (yellow_pop_prob : ℚ) 
  (h1 : white_ratio = 2/3) 
  (h2 : yellow_ratio = 1/3)
  (h3 : white_pop_prob = 1/2)
  (h4 : yellow_pop_prob = 2/3) :
  (white_ratio * white_pop_prob) / (white_ratio * white_pop_prob + yellow_ratio * yellow_pop_prob) = 3/5 := by
  sorry

#check popped_kernel_probability

end NUMINAMATH_CALUDE_popped_kernel_probability_l1956_195655


namespace NUMINAMATH_CALUDE_trigonometric_equation_l1956_195688

theorem trigonometric_equation (x : ℝ) :
  (1 / Real.cos (2022 * x) + Real.tan (2022 * x) = 1 / 2022) →
  (1 / Real.cos (2022 * x) - Real.tan (2022 * x) = 2022) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_l1956_195688


namespace NUMINAMATH_CALUDE_small_cubes_to_large_cube_l1956_195647

theorem small_cubes_to_large_cube (large_volume small_volume : ℕ) 
  (h : large_volume = 1000 ∧ small_volume = 8) : 
  (large_volume / small_volume : ℕ) = 125 := by
  sorry

end NUMINAMATH_CALUDE_small_cubes_to_large_cube_l1956_195647


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_mean60_l1956_195697

-- Define the set of two-digit positive integers
def TwoDigitPositiveInt : Set ℕ := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the mean of x and y
def meanIs60 (x y : ℕ) : Prop := (x + y) / 2 = 60

-- Theorem statement
theorem max_ratio_two_digit_mean60 :
  ∃ (x y : ℕ), x ∈ TwoDigitPositiveInt ∧ y ∈ TwoDigitPositiveInt ∧ meanIs60 x y ∧
  ∀ (a b : ℕ), a ∈ TwoDigitPositiveInt → b ∈ TwoDigitPositiveInt → meanIs60 a b →
  (a : ℚ) / b ≤ 33 / 7 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_mean60_l1956_195697


namespace NUMINAMATH_CALUDE_a_range_for_g_three_zeros_l1956_195699

open Real

noncomputable def f (a b x : ℝ) : ℝ := exp x - 2 * (a - 1) * x - b

noncomputable def g (a b x : ℝ) : ℝ := exp x - (a - 1) * x^2 - b * x - 1

theorem a_range_for_g_three_zeros (a b : ℝ) :
  (g a b 1 = 0) →
  (∃ x₁ x₂ x₃ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < 1 ∧
    g a b x₁ = 0 ∧ g a b x₂ = 0 ∧ g a b x₃ = 0) →
  (e - 1 < a ∧ a < 2) :=
by sorry

end NUMINAMATH_CALUDE_a_range_for_g_three_zeros_l1956_195699


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1956_195696

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 1 = 3 →
  a 1 + a 2 + a 3 = 21 →
  a 3 + a 4 + a 5 = 84 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sum_l1956_195696


namespace NUMINAMATH_CALUDE_cube_diagonal_l1956_195642

theorem cube_diagonal (s : ℝ) (h : s > 0) (eq : s^3 + 36*s = 12*s^2) : 
  Real.sqrt (3 * s^2) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_diagonal_l1956_195642


namespace NUMINAMATH_CALUDE_residue_theorem_l1956_195652

theorem residue_theorem (m k : ℕ) (hm : m > 0) (hk : k > 0) :
  (Nat.gcd m k = 1 →
    ∃ (a b : ℕ → ℕ),
      ∀ i j s t, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ k ∧
                 1 ≤ s ∧ s ≤ m ∧ 1 ≤ t ∧ t ≤ k ∧
                 (i ≠ s ∨ j ≠ t) →
                 (a i * b j) % (m * k) ≠ (a s * b t) % (m * k)) ∧
  (Nat.gcd m k > 1 →
    ∀ (a b : ℕ → ℕ),
      ∃ i j s t, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ k ∧
                 1 ≤ s ∧ s ≤ m ∧ 1 ≤ t ∧ t ≤ k ∧
                 (i ≠ s ∨ j ≠ t) ∧
                 (a i * b j) % (m * k) = (a s * b t) % (m * k)) :=
by sorry

end NUMINAMATH_CALUDE_residue_theorem_l1956_195652


namespace NUMINAMATH_CALUDE_chord_slope_l1956_195698

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x^2 = -2*y

/-- Point P on the parabola -/
def P : ℝ × ℝ := (2, -2)

/-- Complementary angles of inclination -/
def complementary_angles (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The theorem statement -/
theorem chord_slope : 
  ∃ (A B : ℝ × ℝ), 
    parabola A.1 A.2 ∧ 
    parabola B.1 B.2 ∧
    parabola P.1 P.2 ∧
    (∃ (m_PA m_PB : ℝ), complementary_angles m_PA m_PB) ∧
    (B.2 - A.2) / (B.1 - A.1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_chord_slope_l1956_195698


namespace NUMINAMATH_CALUDE_second_number_calculation_l1956_195612

theorem second_number_calculation (A B : ℝ) : 
  A = 700 → 
  0.3 * A = 0.6 * B + 120 → 
  B = 150 :=
by
  sorry

end NUMINAMATH_CALUDE_second_number_calculation_l1956_195612


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_volume_l1956_195614

theorem sphere_surface_area_from_volume :
  ∀ (r : ℝ),
  (4 / 3 : ℝ) * π * r^3 = 72 * π →
  4 * π * r^2 = 36 * π * 2^(2/3) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_volume_l1956_195614


namespace NUMINAMATH_CALUDE_factorial_sum_units_digit_l1956_195669

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem factorial_sum_units_digit :
  ∀ n ≥ 99, units_digit (factorial_sum n) = 7 :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_units_digit_l1956_195669


namespace NUMINAMATH_CALUDE_power_equality_l1956_195671

theorem power_equality (x y : ℕ) (h1 : 8^x = 2^y) (h2 : x = 3) : y = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1956_195671


namespace NUMINAMATH_CALUDE_fraction_equality_l1956_195615

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : s / u = 8 / 15) : 
  (2 * p * s - 3 * q * u) / (5 * q * u - 4 * p * s) = -5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1956_195615


namespace NUMINAMATH_CALUDE_b_minus_a_value_l1956_195689

theorem b_minus_a_value (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 2) (h3 : a + b > 0) :
  b - a = -1 ∨ b - a = -5 := by
  sorry

end NUMINAMATH_CALUDE_b_minus_a_value_l1956_195689


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l1956_195634

theorem sum_of_reciprocal_relations (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1 / x + 1 / y = 1) 
  (h4 : 1 / x - 1 / y = 5) : 
  x + y = -1/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_relations_l1956_195634


namespace NUMINAMATH_CALUDE_science_class_students_l1956_195681

theorem science_class_students :
  ∃! n : ℕ, 0 < n ∧ n < 60 ∧ n % 6 = 4 ∧ n % 8 = 5 ∧ n = 46 := by
sorry

end NUMINAMATH_CALUDE_science_class_students_l1956_195681


namespace NUMINAMATH_CALUDE_one_fourth_greater_than_one_fifth_of_successor_l1956_195622

theorem one_fourth_greater_than_one_fifth_of_successor :
  let N : ℝ := 24.000000000000004
  (1/4 : ℝ) * N - (1/5 : ℝ) * (N + 1) = 1.000000000000000 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_greater_than_one_fifth_of_successor_l1956_195622


namespace NUMINAMATH_CALUDE_polygon_sides_l1956_195680

theorem polygon_sides (sum_interior_angles : ℝ) (n : ℕ) : 
  sum_interior_angles = 1260 → (n - 2) * 180 = sum_interior_angles → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l1956_195680


namespace NUMINAMATH_CALUDE_wizard_elixir_combinations_l1956_195686

/-- The number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- The number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- The number of incompatible crystals. -/
def num_incompatible_crystals : ℕ := 2

/-- The number of herbs incompatible with some crystals. -/
def num_incompatible_herbs : ℕ := 3

/-- The number of valid combinations for the wizard's elixir. -/
def valid_combinations : ℕ := num_herbs * num_crystals - num_incompatible_crystals * num_incompatible_herbs

theorem wizard_elixir_combinations :
  valid_combinations = 18 :=
sorry

end NUMINAMATH_CALUDE_wizard_elixir_combinations_l1956_195686


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1956_195657

theorem binomial_expansion_coefficient (n : ℕ) : 
  (3^2 * (n.choose 2) = 54) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l1956_195657


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1956_195635

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - (a - 1)*x + (a - 1) > 0) ↔ (1 < a ∧ a < 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1956_195635


namespace NUMINAMATH_CALUDE_simplify_cube_roots_l1956_195656

theorem simplify_cube_roots : (512 : ℝ)^(1/3) * (125 : ℝ)^(1/3) = 40 := by sorry

end NUMINAMATH_CALUDE_simplify_cube_roots_l1956_195656


namespace NUMINAMATH_CALUDE_select_shoes_result_l1956_195692

/-- The number of ways to select 4 individual shoes from 5 pairs of shoes,
    such that exactly 1 pair is among the selected shoes -/
def select_shoes (total_pairs : ℕ) (shoes_to_select : ℕ) : ℕ :=
  total_pairs * (total_pairs - 1).choose 2 * 2 * 2

/-- Theorem stating that the number of ways to select 4 individual shoes
    from 5 pairs of shoes, such that exactly 1 pair is among them, is 120 -/
theorem select_shoes_result :
  select_shoes 5 4 = 120 := by sorry

end NUMINAMATH_CALUDE_select_shoes_result_l1956_195692


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1956_195679

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - 3 * x)}
def N : Set ℝ := {x | x^2 - 1 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x ≤ 1/3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1956_195679


namespace NUMINAMATH_CALUDE_otimes_property_implies_a_range_l1956_195685

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem otimes_property_implies_a_range :
  (∀ x : ℝ, otimes x (x + a) < 1) → -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_otimes_property_implies_a_range_l1956_195685


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l1956_195621

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_solution (a : ℕ → ℤ) (d : ℤ) :
  is_arithmetic_sequence a d →
  a 3 * a 7 = -16 →
  a 4 + a 6 = 0 →
  ((a 1 = -8 ∧ d = 2) ∨ (a 1 = 8 ∧ d = -2)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l1956_195621


namespace NUMINAMATH_CALUDE_train_distance_difference_l1956_195651

theorem train_distance_difference (v1 v2 d : ℝ) (hv1 : v1 = 20) (hv2 : v2 = 25) (hd : d = 495) :
  let t := d / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d2 - d1 = 55 := by sorry

end NUMINAMATH_CALUDE_train_distance_difference_l1956_195651


namespace NUMINAMATH_CALUDE_reading_time_proof_l1956_195605

/-- The number of days it took for Ryan and his brother to finish their books -/
def days_to_finish : ℕ := 7

/-- Ryan's total number of pages -/
def ryan_total_pages : ℕ := 2100

/-- Number of pages Ryan's brother reads per day -/
def brother_pages_per_day : ℕ := 200

/-- The difference in pages read per day between Ryan and his brother -/
def page_difference : ℕ := 100

theorem reading_time_proof :
  ryan_total_pages = (brother_pages_per_day + page_difference) * days_to_finish ∧
  ryan_total_pages % (brother_pages_per_day + page_difference) = 0 := by
  sorry

#check reading_time_proof

end NUMINAMATH_CALUDE_reading_time_proof_l1956_195605


namespace NUMINAMATH_CALUDE_parallelogram_area_and_perimeter_l1956_195676

/-- Represents a parallelogram EFGH -/
structure Parallelogram where
  base : ℝ
  height : ℝ
  side : ℝ

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := p.base * p.height

/-- Calculate the perimeter of a parallelogram with all sides equal -/
def perimeter (p : Parallelogram) : ℝ := 4 * p.side

/-- Theorem about the area and perimeter of a specific parallelogram -/
theorem parallelogram_area_and_perimeter :
  ∀ (p : Parallelogram),
  p.base = 6 → p.height = 3 → p.side = 5 →
  area p = 18 ∧ perimeter p = 20 := by sorry

end NUMINAMATH_CALUDE_parallelogram_area_and_perimeter_l1956_195676


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1956_195639

/-- An isosceles triangle with an angle bisector dividing the perimeter -/
structure IsoscelesTriangleWithBisector where
  /-- The length of one of the equal sides -/
  side : ℝ
  /-- The length of the base -/
  base : ℝ
  /-- The length of the angle bisector -/
  bisector : ℝ
  /-- The angle bisector divides the perimeter into parts of 63 and 35 -/
  perimeter_division : side + bisector = 63 ∧ side + base / 2 = 35
  /-- The triangle is isosceles -/
  isosceles : side > 0

/-- The length of the equal sides in the given isosceles triangle is not 26.4, 33, or 38.5 -/
theorem isosceles_triangle_side_length
  (t : IsoscelesTriangleWithBisector) :
  t.side ≠ 26.4 ∧ t.side ≠ 33 ∧ t.side ≠ 38.5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l1956_195639


namespace NUMINAMATH_CALUDE_orange_juice_amount_l1956_195694

theorem orange_juice_amount (total ingredients : ℝ) 
  (strawberries yogurt : ℝ) (h1 : total = 0.5) 
  (h2 : strawberries = 0.2) (h3 : yogurt = 0.1) :
  total - (strawberries + yogurt) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_amount_l1956_195694


namespace NUMINAMATH_CALUDE_green_garden_potato_yield_l1956_195632

/-- Represents Mr. Green's garden and potato yield calculation --/
theorem green_garden_potato_yield :
  let garden_length_steps : ℕ := 25
  let garden_width_steps : ℕ := 30
  let step_length_feet : ℕ := 3
  let non_productive_percentage : ℚ := 1/10
  let yield_per_square_foot : ℚ := 3/4

  let garden_length_feet : ℕ := garden_length_steps * step_length_feet
  let garden_width_feet : ℕ := garden_width_steps * step_length_feet
  let garden_area : ℕ := garden_length_feet * garden_width_feet
  let productive_area : ℚ := garden_area * (1 - non_productive_percentage)
  let total_yield : ℚ := productive_area * yield_per_square_foot

  total_yield = 4556.25 := by sorry

end NUMINAMATH_CALUDE_green_garden_potato_yield_l1956_195632


namespace NUMINAMATH_CALUDE_complement_A_B_l1956_195684

-- Define the set A
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (-2) 4, y = |x + 1|}

-- Define the set B
def B : Set ℝ := Set.Ici 2 ∩ Set.Iio 5

-- Theorem statement
theorem complement_A_B : 
  (Set.compl B) ∩ A = Set.Icc 0 2 ∪ {5} :=
sorry

end NUMINAMATH_CALUDE_complement_A_B_l1956_195684


namespace NUMINAMATH_CALUDE_election_votes_theorem_l1956_195661

/-- Represents an election between two candidates -/
structure Election where
  total_votes : ℕ
  winner_votes : ℕ
  loser_votes : ℕ

/-- The conditions of the election problem -/
def election_conditions (e : Election) : Prop :=
  e.winner_votes + e.loser_votes = e.total_votes ∧
  e.winner_votes - e.loser_votes = (e.total_votes : ℚ) * (1 / 10) ∧
  (e.winner_votes - 1500) - (e.loser_votes + 1500) = -(e.total_votes : ℚ) * (1 / 10)

/-- The theorem stating that under the given conditions, the total votes is 15000 -/
theorem election_votes_theorem (e : Election) :
  election_conditions e → e.total_votes = 15000 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l1956_195661


namespace NUMINAMATH_CALUDE_rose_mother_age_ratio_l1956_195654

/-- Represents the ratio of two ages -/
structure AgeRatio where
  numerator : ℕ
  denominator : ℕ

/-- Rose's age in years -/
def rose_age : ℕ := 25

/-- Rose's mother's age in years -/
def mother_age : ℕ := 75

/-- The ratio of Rose's age to her mother's age -/
def rose_to_mother_ratio : AgeRatio := ⟨1, 3⟩

/-- Theorem stating that the ratio of Rose's age to her mother's age is 1:3 -/
theorem rose_mother_age_ratio : 
  (rose_age : ℚ) / (mother_age : ℚ) = (rose_to_mother_ratio.numerator : ℚ) / (rose_to_mother_ratio.denominator : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_rose_mother_age_ratio_l1956_195654


namespace NUMINAMATH_CALUDE_gcd_459_357_l1956_195633

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1956_195633


namespace NUMINAMATH_CALUDE_sqrt_one_fourth_l1956_195637

theorem sqrt_one_fourth : Real.sqrt (1 / 4) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_one_fourth_l1956_195637


namespace NUMINAMATH_CALUDE_cds_on_shelf_l1956_195660

/-- The number of CDs that a single rack can hold -/
def cds_per_rack : ℕ := 8

/-- The number of racks that can fit on a shelf -/
def racks_per_shelf : ℕ := 4

/-- Theorem stating the total number of CDs that can fit on a shelf -/
theorem cds_on_shelf : cds_per_rack * racks_per_shelf = 32 := by
  sorry

end NUMINAMATH_CALUDE_cds_on_shelf_l1956_195660


namespace NUMINAMATH_CALUDE_interest_difference_implies_principal_l1956_195673

/-- Prove that for a given principal amount, if the difference between compound
    interest (compounded annually) and simple interest over 2 years at 4% per annum
    is 1, then the principal amount is 625. -/
theorem interest_difference_implies_principal (P : ℝ) : 
  P * (1 + 0.04)^2 - P - (P * 0.04 * 2) = 1 → P = 625 := by
  sorry

end NUMINAMATH_CALUDE_interest_difference_implies_principal_l1956_195673


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1956_195641

theorem sqrt_sum_equality (a b c : ℝ) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ a*b + b*c + c*a = 0 ∧ a + b + c ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1956_195641


namespace NUMINAMATH_CALUDE_square_difference_l1956_195625

theorem square_difference (a b : ℝ) (h1 : a + b = 10) (h2 : a - b = 4) : a^2 - b^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1956_195625


namespace NUMINAMATH_CALUDE_samara_oil_spending_l1956_195607

/-- The amount Alberto spent on his car -/
def alberto_spent : ℕ := 2457

/-- The amount Samara spent on tires -/
def samara_tires : ℕ := 467

/-- The amount Samara spent on detailing -/
def samara_detailing : ℕ := 79

/-- The difference between Alberto's and Samara's spending -/
def spending_difference : ℕ := 1886

/-- The amount Samara spent on oil -/
def samara_oil : ℕ := 25

theorem samara_oil_spending : 
  alberto_spent = samara_oil + samara_tires + samara_detailing + spending_difference :=
by sorry

end NUMINAMATH_CALUDE_samara_oil_spending_l1956_195607


namespace NUMINAMATH_CALUDE_quadratic_quotient_cubic_at_zero_l1956_195631

-- Define the set of integers from 1 to 5
def S : Set ℕ := {1, 2, 3, 4, 5}

-- Define the property that f(n) = n^3 for n in S
def cubic_on_S (f : ℚ → ℚ) : Prop :=
  ∀ n ∈ S, f n = n^3

-- Define the property that f is a quotient of two quadratic polynomials
def is_quadratic_quotient (f : ℚ → ℚ) : Prop :=
  ∃ (p q : ℚ → ℚ),
    (∀ x, ∃ a b c, p x = a*x^2 + b*x + c) ∧
    (∀ x, ∃ d e g, q x = d*x^2 + e*x + g) ∧
    (∀ x, q x ≠ 0) ∧
    (∀ x, f x = p x / q x)

-- The main theorem
theorem quadratic_quotient_cubic_at_zero
  (f : ℚ → ℚ)
  (h1 : is_quadratic_quotient f)
  (h2 : cubic_on_S f) :
  f 0 = 24/17 := by
sorry

end NUMINAMATH_CALUDE_quadratic_quotient_cubic_at_zero_l1956_195631


namespace NUMINAMATH_CALUDE_division_problem_l1956_195629

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 109)
  (h2 : divisor = 12)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 9 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1956_195629


namespace NUMINAMATH_CALUDE_factorial_not_equal_even_factorial_l1956_195650

theorem factorial_not_equal_even_factorial (n m : ℕ) (hn : n > 1) (hm : m > 1) :
  n.factorial ≠ 2^m * m.factorial := by
  sorry

end NUMINAMATH_CALUDE_factorial_not_equal_even_factorial_l1956_195650


namespace NUMINAMATH_CALUDE_inequality_theorem_l1956_195643

/-- A function f: ℝ⁺ → ℝ⁺ such that f(x)/x is increasing on ℝ⁺ -/
def IncreasingRatioFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → x < y → (f x) / x < (f y) / y

theorem inequality_theorem (f : ℝ → ℝ) (h : IncreasingRatioFunction f) 
    (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  2 * ((f a + f b) / (a + b) + (f b + f c) / (b + c) + (f c + f a) / (c + a)) ≥ 
  3 * ((f a + f b + f c) / (a + b + c)) + f a / a + f b / b + f c / c := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l1956_195643


namespace NUMINAMATH_CALUDE_distance_to_asymptote_l1956_195648

def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

def point : ℝ × ℝ := (3, 0)

theorem distance_to_asymptote :
  ∃ (a b c : ℝ), 
    (∀ (x y : ℝ), hyperbola x y → (a * x + b * y + c = 0 ∨ a * x + b * y - c = 0)) ∧
    (|a * point.1 + b * point.2 + c| / Real.sqrt (a^2 + b^2) = 9/5) :=
sorry

end NUMINAMATH_CALUDE_distance_to_asymptote_l1956_195648


namespace NUMINAMATH_CALUDE_g_of_one_eq_neg_25_l1956_195646

/-- g is a rational function satisfying the given equation for all non-zero x -/
def g_equation (g : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * g (2 / x) + 3 * g x / x = 2 * x^3 - x

/-- Theorem: If g satisfies the equation, then g(1) = -25 -/
theorem g_of_one_eq_neg_25 (g : ℚ → ℚ) (h : g_equation g) : g 1 = -25 := by
  sorry

end NUMINAMATH_CALUDE_g_of_one_eq_neg_25_l1956_195646


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1956_195666

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 > 0) ↔ (∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1956_195666


namespace NUMINAMATH_CALUDE_largest_m_for_factorization_l1956_195662

def is_valid_factorization (m A B : ℤ) : Prop :=
  A * B = 90 ∧ 5 * B + A = m

theorem largest_m_for_factorization :
  (∃ (m : ℤ), ∀ (A B : ℤ), is_valid_factorization m A B →
    ∀ (m' : ℤ), (∃ (A' B' : ℤ), is_valid_factorization m' A' B') → m' ≤ m) ∧
  (∃ (A B : ℤ), is_valid_factorization 451 A B) :=
sorry

end NUMINAMATH_CALUDE_largest_m_for_factorization_l1956_195662


namespace NUMINAMATH_CALUDE_abs_z_equals_5_sqrt_2_l1956_195608

theorem abs_z_equals_5_sqrt_2 (z : ℂ) (h : z^2 = -48 + 14*I) : Complex.abs z = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_z_equals_5_sqrt_2_l1956_195608


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l1956_195610

/-- An arithmetic sequence with 1990 terms -/
def ArithmeticSequence := Fin 1990 → ℝ

/-- The common difference of an arithmetic sequence -/
def commonDifference (a : ArithmeticSequence) : ℝ :=
  a 1 - a 0

/-- The condition that all terms in the sequence are positive -/
def allPositive (a : ArithmeticSequence) : Prop :=
  ∀ i j : Fin 1990, a i * a j > 0

/-- The b_k sequence defined in the problem -/
def b (a : ArithmeticSequence) (k : Fin 1990) : ℝ :=
  a k * a (1989 - k)

theorem arithmetic_sequence_max_product 
  (a : ArithmeticSequence) 
  (hd : commonDifference a ≠ 0) 
  (hp : allPositive a) : 
  (∀ k : Fin 1990, b a k ≤ b a 994 ∨ b a k ≤ b a 995) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_product_l1956_195610


namespace NUMINAMATH_CALUDE_product_abcd_l1956_195644

theorem product_abcd (a b c d : ℚ) : 
  (3 * a + 2 * b + 4 * c + 6 * d = 42) →
  (4 * d + 2 * c = b) →
  (4 * b - 2 * c = a) →
  (d + 2 = c) →
  (a * b * c * d = -(5 * 83 * 46 * 121) / (44 * 44 * 11 * 11)) := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_l1956_195644


namespace NUMINAMATH_CALUDE_quadratic_one_root_l1956_195690

theorem quadratic_one_root (m : ℝ) : 
  (∀ x : ℝ, x^2 - 8*m*x + 15*m = 0 → (∀ y : ℝ, y^2 - 8*m*y + 15*m = 0 → y = x)) → 
  m = 15/16 :=
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l1956_195690


namespace NUMINAMATH_CALUDE_inscribed_square_area_l1956_195675

/-- Given an equilateral triangle with side length a inscribed in a circle,
    the area of a square inscribed in the same circle is 2a^2/3 -/
theorem inscribed_square_area (a : ℝ) (ha : a > 0) :
  ∃ (R : ℝ), R > 0 ∧
  ∃ (s : ℝ), s > 0 ∧
  (a = R * Real.sqrt 3) ∧ 
  (s = R * Real.sqrt 2) ∧
  (s^2 = 2 * a^2 / 3) :=
sorry

end NUMINAMATH_CALUDE_inscribed_square_area_l1956_195675


namespace NUMINAMATH_CALUDE_correct_arrangement_count_l1956_195606

/-- The number of ways to arrange 7 people with specific adjacency conditions -/
def arrangement_count : ℕ := 960

/-- Proves that the number of arrangements is correct -/
theorem correct_arrangement_count : arrangement_count = 960 := by
  sorry

end NUMINAMATH_CALUDE_correct_arrangement_count_l1956_195606


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l1956_195667

/-- Fred's earnings from delivering newspapers -/
def newspaper_earnings : ℕ := 16

/-- Fred's earnings from washing cars -/
def car_wash_earnings : ℕ := 74

/-- Fred's total earnings over the weekend -/
def total_earnings : ℕ := newspaper_earnings + car_wash_earnings

/-- Theorem stating that Fred's total earnings over the weekend equal $90 -/
theorem fred_weekend_earnings : total_earnings = 90 := by
  sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l1956_195667


namespace NUMINAMATH_CALUDE_surjective_iff_coprime_l1956_195619

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- The function f(x) = x^x mod n -/
def f (n : ℕ) (x : ℕ+) : ZMod n :=
  (x : ZMod n) ^ (x : ℕ)

/-- Surjectivity of f -/
def is_surjective (n : ℕ) : Prop :=
  Function.Surjective (f n)

theorem surjective_iff_coprime (n : ℕ) (h : n > 0) :
  is_surjective n ↔ Nat.Coprime n (phi n) := by sorry

end NUMINAMATH_CALUDE_surjective_iff_coprime_l1956_195619


namespace NUMINAMATH_CALUDE_cinema_sampling_method_l1956_195682

/-- Represents a seating arrangement in a cinema --/
structure CinemaSeating where
  rows : ℕ
  seats_per_row : ℕ
  all_seats_filled : Bool

/-- Represents a sampling method --/
inductive SamplingMethod
  | LotteryMethod
  | RandomNumberTable
  | SystematicSampling
  | SamplingWithReplacement

/-- Defines the characteristics of systematic sampling --/
def is_systematic_sampling (seating : CinemaSeating) (selected_seat : ℕ) : Prop :=
  seating.all_seats_filled ∧
  selected_seat > 0 ∧
  selected_seat ≤ seating.seats_per_row ∧
  seating.rows > 1

/-- The main theorem to prove --/
theorem cinema_sampling_method (seating : CinemaSeating) (selected_seat : ℕ) :
  seating.rows = 50 →
  seating.seats_per_row = 60 →
  seating.all_seats_filled = true →
  selected_seat = 18 →
  is_systematic_sampling seating selected_seat →
  SamplingMethod.SystematicSampling = SamplingMethod.SystematicSampling :=
by
  sorry

end NUMINAMATH_CALUDE_cinema_sampling_method_l1956_195682


namespace NUMINAMATH_CALUDE_smallest_gcd_of_b_c_l1956_195620

theorem smallest_gcd_of_b_c (a b c x y : ℕ+) 
  (hab : Nat.gcd a b = 120)
  (hac : Nat.gcd a c = 1001)
  (hb : b = 120 * x)
  (hc : c = 1001 * y) :
  ∃ (b' c' : ℕ+), Nat.gcd b' c' = 1 ∧ 
    ∀ (b'' c'' : ℕ+), (∃ (x'' y'' : ℕ+), b'' = 120 * x'' ∧ c'' = 1001 * y'') → 
      Nat.gcd b'' c'' ≥ Nat.gcd b' c' :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_of_b_c_l1956_195620


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1956_195677

/-- A quadrilateral circumscribed around a circle -/
structure CircumscribedQuadrilateral where
  /-- The sum of two opposite sides -/
  opposite_sides_sum : ℝ
  /-- The area of the quadrilateral -/
  area : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ

/-- Theorem: If the sum of opposite sides is 10 and the area is 12, 
    then the radius of the inscribed circle is 6/5 -/
theorem inscribed_circle_radius 
  (q : CircumscribedQuadrilateral) 
  (h1 : q.opposite_sides_sum = 10) 
  (h2 : q.area = 12) : 
  q.inradius = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1956_195677


namespace NUMINAMATH_CALUDE_large_triangle_perimeter_l1956_195658

/-- An isosceles triangle with two sides of length 20 and one side of length 10 -/
structure SmallTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  is_isosceles : side2 = side3
  length_side1 : side1 = 10
  length_side2 : side2 = 20

/-- A triangle similar to SmallTriangle with shortest side of length 50 -/
structure LargeTriangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  shortest_side : side1 = 50
  similar_to_small : ∃ (k : ℝ), side1 = k * 10 ∧ side2 = k * 20 ∧ side3 = k * 20

/-- The perimeter of a triangle -/
def perimeter (t : LargeTriangle) : ℝ := t.side1 + t.side2 + t.side3

/-- Theorem stating that the perimeter of the larger triangle is 250 -/
theorem large_triangle_perimeter :
  ∀ (small : SmallTriangle) (large : LargeTriangle),
  perimeter large = 250 := by sorry

end NUMINAMATH_CALUDE_large_triangle_perimeter_l1956_195658


namespace NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l1956_195613

theorem sphere_in_cube_surface_area (cube_edge : ℝ) (h : cube_edge = 4) :
  let sphere_radius := cube_edge / 2
  let sphere_surface_area := 4 * Real.pi * sphere_radius ^ 2
  sphere_surface_area = 16 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_in_cube_surface_area_l1956_195613


namespace NUMINAMATH_CALUDE_right_triangle_bisector_inscribed_circle_l1956_195653

/-- 
Theorem: In a right triangle with an inscribed circle of radius ρ 
and an angle bisector of length f for one of its acute angles, 
the condition f > √(8ρ) must hold.
-/
theorem right_triangle_bisector_inscribed_circle 
  (ρ f : ℝ) 
  (h_positive_ρ : ρ > 0) 
  (h_positive_f : f > 0) 
  (h_right_triangle : ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a^2 + b^2 = c^2 ∧ 
    (a * b) / (a + b + c) = ρ ∧
    f = (2 * a * b) / (a + b)) :
  f > Real.sqrt (8 * ρ) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_bisector_inscribed_circle_l1956_195653


namespace NUMINAMATH_CALUDE_division_equality_l1956_195695

theorem division_equality (x : ℝ) (h : 2994 / x = 173) : x = 17.3 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l1956_195695


namespace NUMINAMATH_CALUDE_stack_b_tallest_l1956_195674

/-- Represents the height of a stack of wood blocks -/
def stack_height (num_pieces : ℕ) (block_height : ℝ) : ℝ :=
  (num_pieces : ℝ) * block_height

/-- Proves that stack B is the tallest among the three stacks of wood blocks -/
theorem stack_b_tallest (height_a height_b height_c : ℝ) 
  (h_height_a : height_a = 2)
  (h_height_b : height_b = 1.5)
  (h_height_c : height_c = 2.5) :
  stack_height 11 height_b > stack_height 8 height_a ∧ 
  stack_height 11 height_b > stack_height 6 height_c :=
by
  sorry

#check stack_b_tallest

end NUMINAMATH_CALUDE_stack_b_tallest_l1956_195674


namespace NUMINAMATH_CALUDE_paul_pencil_days_l1956_195645

/-- Calculates the number of days Paul makes pencils in a week -/
def pencil_making_days (
  pencils_per_day : ℕ) 
  (initial_stock : ℕ) 
  (pencils_sold : ℕ) 
  (final_stock : ℕ) : ℕ :=
  (final_stock + pencils_sold - initial_stock) / pencils_per_day

theorem paul_pencil_days : 
  pencil_making_days 100 80 350 230 = 5 := by sorry

end NUMINAMATH_CALUDE_paul_pencil_days_l1956_195645


namespace NUMINAMATH_CALUDE_consecutive_discounts_l1956_195670

theorem consecutive_discounts (original_price : ℝ) (h : original_price > 0) :
  let price_after_first_discount := original_price * (1 - 0.3)
  let price_after_second_discount := price_after_first_discount * (1 - 0.2)
  let final_price := price_after_second_discount * (1 - 0.1)
  (original_price - final_price) / original_price = 0.496 := by
sorry

end NUMINAMATH_CALUDE_consecutive_discounts_l1956_195670


namespace NUMINAMATH_CALUDE_number_problem_l1956_195627

theorem number_problem (x : ℚ) : (x = (3/8) * x + 40) → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1956_195627


namespace NUMINAMATH_CALUDE_rick_has_two_sisters_l1956_195683

/-- Calculates the number of Rick's sisters based on the given card distribution. -/
def number_of_sisters (total_cards : ℕ) (kept_cards : ℕ) (miguel_cards : ℕ) 
  (friends : ℕ) (cards_per_friend : ℕ) (cards_per_sister : ℕ) : ℕ :=
  let remaining_cards := total_cards - kept_cards - miguel_cards - (friends * cards_per_friend)
  remaining_cards / cards_per_sister

/-- Theorem stating that Rick has 2 sisters given the card distribution. -/
theorem rick_has_two_sisters :
  number_of_sisters 130 15 13 8 12 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_rick_has_two_sisters_l1956_195683


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l1956_195628

theorem z_in_first_quadrant (z : ℂ) (h : z * (1 - 2*I) = 3 - I) : 
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l1956_195628


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1956_195649

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℝ), 
    (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
      (-2 * x^2 + 5 * x - 7) / (x^3 - x) = P / x + (Q * x + R) / (x^2 - 1)) ∧
    P = 7 ∧ Q = -9 ∧ R = 5 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1956_195649


namespace NUMINAMATH_CALUDE_temp_difference_l1956_195659

/-- The highest temperature in Xiangyang City on March 7, 2023 -/
def highest_temp : ℝ := 26

/-- The lowest temperature in Xiangyang City on March 7, 2023 -/
def lowest_temp : ℝ := 14

/-- The theorem states that the difference between the highest and lowest temperatures is 12°C -/
theorem temp_difference : highest_temp - lowest_temp = 12 := by
  sorry

end NUMINAMATH_CALUDE_temp_difference_l1956_195659


namespace NUMINAMATH_CALUDE_min_length_intersection_l1956_195640

/-- The minimum length of the intersection of two sets M and N -/
theorem min_length_intersection (m n : ℝ) : 
  0 ≤ m → m + 3/4 ≤ 1 → 0 ≤ n - 1/3 → n ≤ 1 → 
  ∃ (a b : ℝ), 
    (∀ x, x ∈ (Set.Icc m (m + 3/4) ∩ Set.Icc (n - 1/3) n) → a ≤ x ∧ x ≤ b) ∧
    b - a = 1/12 ∧
    (∀ c d, (∀ x, x ∈ (Set.Icc m (m + 3/4) ∩ Set.Icc (n - 1/3) n) → c ≤ x ∧ x ≤ d) → 
      d - c ≥ 1/12) :=
sorry

end NUMINAMATH_CALUDE_min_length_intersection_l1956_195640


namespace NUMINAMATH_CALUDE_tangent_angle_inclination_l1956_195600

/-- The angle of inclination of the tangent to y = (1/3)x³ - 2 at (1, -5/3) is 45° --/
theorem tangent_angle_inclination (f : ℝ → ℝ) (x : ℝ) :
  f x = (1/3) * x^3 - 2 →
  (deriv f) x = x^2 →
  x = 1 →
  f x = -5/3 →
  Real.arctan ((deriv f) x) = π/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_angle_inclination_l1956_195600


namespace NUMINAMATH_CALUDE_smallest_multiple_l1956_195693

theorem smallest_multiple (n : ℕ) : n = 1050 ↔ 
  n > 0 ∧ 
  50 ∣ n ∧ 
  75 ∣ n ∧ 
  ¬(18 ∣ n) ∧ 
  7 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 50 ∣ m → 75 ∣ m → ¬(18 ∣ m) → 7 ∣ m → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l1956_195693


namespace NUMINAMATH_CALUDE_no_solution_implies_a_range_l1956_195623

theorem no_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(|x + 1| + |x - 2| < a)) → a ∈ Set.Ici 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_range_l1956_195623


namespace NUMINAMATH_CALUDE_katerina_weight_l1956_195616

theorem katerina_weight (total_weight : ℕ) (alexa_weight : ℕ) 
  (h1 : total_weight = 95)
  (h2 : alexa_weight = 46) :
  total_weight - alexa_weight = 49 := by
  sorry

end NUMINAMATH_CALUDE_katerina_weight_l1956_195616


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l1956_195617

/-- A linear function y = -3x + b -/
def linearFunction (x : ℝ) (b : ℝ) : ℝ := -3 * x + b

/-- Theorem: For a linear function y = -3x + b, if P₁(-3, y₁) and P₂(4, y₂) are points on the graph, then y₁ > y₂ -/
theorem y1_greater_than_y2 (b : ℝ) (y₁ y₂ : ℝ) 
  (h₁ : y₁ = linearFunction (-3) b) 
  (h₂ : y₂ = linearFunction 4 b) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l1956_195617


namespace NUMINAMATH_CALUDE_line_plane_relationship_l1956_195691

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (contained_in : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- State the theorem
theorem line_plane_relationship 
  (m n : Line) (α β : Plane) 
  (h1 : parallel α β) 
  (h2 : perpendicular m α) 
  (h3 : perpendicular_lines m n) :
  contained_in n β ∨ parallel_line_plane n β :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l1956_195691


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1956_195604

theorem p_necessary_not_sufficient_for_q :
  (∀ x : ℝ, x^2 - x - 2 < 0 → |x| < 2) ∧
  (∃ x : ℝ, |x| < 2 ∧ x^2 - x - 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1956_195604


namespace NUMINAMATH_CALUDE_f_monotone_and_no_min_l1956_195678

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a - 1) * Real.exp (x - 1) - (1/2) * x^2 + a * x

theorem f_monotone_and_no_min (x : ℝ) (hx : x > 0) :
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f 1 x₁ < f 1 x₂) ∧
  (∃ a₁ a₂ : ℤ, (∀ x > 0, ∃ y > x, f a₁ y < f a₁ x) ∧
                (∀ x > 0, ∃ y > x, f a₂ y < f a₂ x) ∧
                a₁ + a₂ = 3) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_and_no_min_l1956_195678


namespace NUMINAMATH_CALUDE_jiwon_walk_distance_l1956_195687

theorem jiwon_walk_distance 
  (sets_of_steps : ℕ) 
  (steps_per_set : ℕ) 
  (distance_per_step : ℝ) : 
  sets_of_steps = 13 → 
  steps_per_set = 90 → 
  distance_per_step = 0.45 → 
  (sets_of_steps * steps_per_set : ℝ) * distance_per_step = 526.5 := by
sorry

end NUMINAMATH_CALUDE_jiwon_walk_distance_l1956_195687


namespace NUMINAMATH_CALUDE_equal_area_parallelograms_locus_l1956_195618

/-- Given a triangle ABC and an interior point P, this theorem states that if the areas of
    parallelograms GPDC and FPEB (formed by lines parallel to the sides through P) are equal,
    then P lies on a specific line. -/
theorem equal_area_parallelograms_locus (a b c k l : ℝ) :
  let A : ℝ × ℝ := (0, a)
  let B : ℝ × ℝ := (-b, 0)
  let C : ℝ × ℝ := (c, 0)
  let P : ℝ × ℝ := (k, l)
  let E : ℝ × ℝ := (k - b*l/a, 0)
  let D : ℝ × ℝ := (k + l*c/a, 0)
  let F : ℝ × ℝ := (b*l/a - b, l)
  let G : ℝ × ℝ := (c - l*c/a, l)
  a > 0 ∧ b > 0 ∧ c > 0 ∧ k > -b ∧ k < c ∧ l > 0 ∧ l < a →
  abs (l/2 * (-c + 2*l*c/a)) = abs (l/2 * (-b + 2*l*b/a)) →
  2*a*k + (c - b)*l + a*(b - c) = 0 :=
sorry

end NUMINAMATH_CALUDE_equal_area_parallelograms_locus_l1956_195618


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_two_thirds_l1956_195626

/-- The infinite repeating decimal 0.666... -/
def repeating_decimal : ℚ := 0.6666666666666667

/-- The theorem stating that the repeating decimal 0.666... is equal to 2/3 -/
theorem repeating_decimal_equals_two_thirds : repeating_decimal = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_two_thirds_l1956_195626


namespace NUMINAMATH_CALUDE_double_sum_of_factors_17_l1956_195668

/-- The sum of positive factors of a natural number -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- The boxed notation representing the sum of positive factors -/
notation "⌈" n "⌉" => sum_of_factors n

/-- Theorem stating that the double application of sum_of_factors to 17 equals 39 -/
theorem double_sum_of_factors_17 : ⌈⌈17⌉⌉ = 39 := by sorry

end NUMINAMATH_CALUDE_double_sum_of_factors_17_l1956_195668


namespace NUMINAMATH_CALUDE_fraction_simplification_l1956_195672

theorem fraction_simplification (m : ℝ) (h : m^2 ≠ 1) :
  (m^2 - m) / (m^2 - 1) = m / (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1956_195672


namespace NUMINAMATH_CALUDE_spatial_vector_division_not_defined_l1956_195665

-- Define a spatial vector
structure SpatialVector where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define valid operations
def add (v w : SpatialVector) : SpatialVector :=
  { x := v.x + w.x, y := v.y + w.y, z := v.z + w.z }

def sub (v w : SpatialVector) : SpatialVector :=
  { x := v.x - w.x, y := v.y - w.y, z := v.z - w.z }

def scalarProduct (v w : SpatialVector) : ℝ :=
  v.x * w.x + v.y * w.y + v.z * w.z

-- Theorem stating that division is not well-defined for spatial vectors
theorem spatial_vector_division_not_defined :
  ¬ ∃ (f : SpatialVector → SpatialVector → SpatialVector),
    ∀ (v w : SpatialVector), w ≠ { x := 0, y := 0, z := 0 } →
      f v w = { x := v.x / w.x, y := v.y / w.y, z := v.z / w.z } :=
by
  sorry


end NUMINAMATH_CALUDE_spatial_vector_division_not_defined_l1956_195665


namespace NUMINAMATH_CALUDE_intersection_distance_l1956_195611

-- Define the line x = 4
def line (x : ℝ) : Prop := x = 4

-- Define the curve x = t², y = t³
def curve (t x y : ℝ) : Prop := x = t^2 ∧ y = t^3

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line A.1 ∧ curve 2 A.1 A.2 ∧
  line B.1 ∧ curve (-2) B.1 B.2

-- Theorem statement
theorem intersection_distance :
  ∀ A B : ℝ × ℝ, intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 :=
by sorry

end NUMINAMATH_CALUDE_intersection_distance_l1956_195611


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l1956_195601

/-- Proves that the ratio of downstream to upstream speed is 2:1 for a boat in a river --/
theorem boat_speed_ratio (v : ℝ) : 
  v > 3 →  -- Boat speed must be greater than river flow
  (4 / (v + 3) + 4 / (v - 3) = 1) →  -- Total travel time is 1 hour
  ((v + 3) / (v - 3) = 2) :=  -- Ratio of downstream to upstream speed
by
  sorry

#check boat_speed_ratio

end NUMINAMATH_CALUDE_boat_speed_ratio_l1956_195601


namespace NUMINAMATH_CALUDE_prism_sphere_surface_area_l1956_195630

/-- Right triangular prism with specified properties -/
structure RightTriangularPrism where
  -- Base triangle
  AB : ℝ
  AC : ℝ
  angleBAC : ℝ
  -- Prism properties
  volume : ℝ
  -- Ensure all vertices lie on the same spherical surface
  onSphere : Bool

/-- Theorem stating the surface area of the sphere containing the prism -/
theorem prism_sphere_surface_area (p : RightTriangularPrism) 
  (h1 : p.AB = 2)
  (h2 : p.AC = 1)
  (h3 : p.angleBAC = π / 3)  -- 60° in radians
  (h4 : p.volume = Real.sqrt 3)
  (h5 : p.onSphere = true) :
  ∃ (r : ℝ), 4 * π * r^2 = 8 * π := by
    sorry


end NUMINAMATH_CALUDE_prism_sphere_surface_area_l1956_195630


namespace NUMINAMATH_CALUDE_max_cylinder_volume_l1956_195602

/-- The maximum volume of a cylinder formed by rotating a rectangle with perimeter 20cm around one of its edges -/
theorem max_cylinder_volume : 
  ∃ (V : ℝ), V = (4000 / 27) * Real.pi ∧ 
  (∀ (x : ℝ), 0 < x → x < 10 → 
    π * x^2 * (10 - x) ≤ V) :=
by sorry

end NUMINAMATH_CALUDE_max_cylinder_volume_l1956_195602


namespace NUMINAMATH_CALUDE_sum_of_remainders_l1956_195638

theorem sum_of_remainders : Int.mod (Int.mod (5^(5^(5^5))) 500 + Int.mod (2^(2^(2^2))) 500) 500 = 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_remainders_l1956_195638


namespace NUMINAMATH_CALUDE_andrews_eggs_l1956_195636

-- Define the costs
def toast_cost : ℕ := 1
def egg_cost : ℕ := 3

-- Define Dale's breakfast
def dale_toast : ℕ := 2
def dale_eggs : ℕ := 2

-- Define Andrew's breakfast
def andrew_toast : ℕ := 1

-- Define the total cost
def total_cost : ℕ := 15

-- Theorem to prove
theorem andrews_eggs :
  ∃ (andrew_eggs : ℕ),
    toast_cost * (dale_toast + andrew_toast) +
    egg_cost * (dale_eggs + andrew_eggs) = total_cost ∧
    andrew_eggs = 2 := by
  sorry

end NUMINAMATH_CALUDE_andrews_eggs_l1956_195636


namespace NUMINAMATH_CALUDE_min_value_x_plus_3y_l1956_195624

theorem min_value_x_plus_3y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3*y + x*y = 9) :
  ∀ a b : ℝ, a > 0 → b > 0 → a + 3*b + a*b = 9 → x + 3*y ≤ a + 3*b :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_3y_l1956_195624


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1956_195603

theorem expand_and_simplify (x : ℝ) : 3 * (x - 3) * (x + 10) + 2 * x = 3 * x^2 + 23 * x - 90 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1956_195603
