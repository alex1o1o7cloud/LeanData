import Mathlib

namespace NUMINAMATH_CALUDE_dice_labeling_exists_l3845_384570

/-- Represents a 6-sided die with integer labels -/
def Die := Fin 6 → ℕ

/-- Checks if a given labeling of two dice produces all sums from 1 to 36 -/
def valid_labeling (d1 d2 : Die) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 36 → ∃ (i j : Fin 6), d1 i + d2 j = n

/-- There exists a labeling for two dice that produces all sums from 1 to 36 with equal probabilities -/
theorem dice_labeling_exists : ∃ (d1 d2 : Die), valid_labeling d1 d2 := by
  sorry

end NUMINAMATH_CALUDE_dice_labeling_exists_l3845_384570


namespace NUMINAMATH_CALUDE_equality_of_powers_l3845_384507

theorem equality_of_powers (a b c d : ℕ) :
  a^a * b^(a + b) = c^c * d^(c + d) →
  Nat.gcd a b = 1 →
  Nat.gcd c d = 1 →
  a = c ∧ b = d := by
  sorry

end NUMINAMATH_CALUDE_equality_of_powers_l3845_384507


namespace NUMINAMATH_CALUDE_largest_number_with_conditions_l3845_384540

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 5

def digits_sum_to_12 (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ : ℕ), n = 100 * d₁ + 10 * d₂ + d₃ ∧
    is_valid_digit d₁ ∧ is_valid_digit d₂ ∧ is_valid_digit d₃ ∧
    d₁ + d₂ + d₃ = 12

theorem largest_number_with_conditions : 
  ∀ n : ℕ, digits_sum_to_12 n → n ≤ 552 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_with_conditions_l3845_384540


namespace NUMINAMATH_CALUDE_not_all_F_zero_on_C_implies_exists_F_zero_not_on_C_l3845_384580

-- Define the curve C and the function F
variable (C : Set (ℝ × ℝ))
variable (F : ℝ → ℝ → ℝ)

-- Define the set of points satisfying F(x, y) = 0
def F_zero_set (F : ℝ → ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | F p.1 p.2 = 0}

-- State the theorem
theorem not_all_F_zero_on_C_implies_exists_F_zero_not_on_C
  (h : ¬(F_zero_set F ⊆ C)) :
  ∃ p : ℝ × ℝ, p ∉ C ∧ F p.1 p.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_not_all_F_zero_on_C_implies_exists_F_zero_not_on_C_l3845_384580


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l3845_384567

theorem sqrt_product_plus_one : 
  Real.sqrt ((41 : ℝ) * 40 * 39 * 38 + 1) = 1559 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l3845_384567


namespace NUMINAMATH_CALUDE_infinite_power_tower_four_l3845_384584

/-- The limit of the sequence defined by a_0 = x, a_(n+1) = x^(a_n) --/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := sorry

theorem infinite_power_tower_four (x : ℝ) :
  x > 0 → infinitePowerTower x = 4 → x = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_infinite_power_tower_four_l3845_384584


namespace NUMINAMATH_CALUDE_trapezoid_area_l3845_384595

theorem trapezoid_area (c : ℝ) (hc : c > 0) :
  let b := Real.sqrt c
  let shorter_base := b - 3
  let altitude := b
  let longer_base := b + 3
  let area := ((shorter_base + longer_base) / 2) * altitude
  area = c := by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l3845_384595


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l3845_384581

theorem half_abs_diff_squares_20_15 : 
  (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_20_15_l3845_384581


namespace NUMINAMATH_CALUDE_polynomial_property_l3845_384539

/-- A polynomial of the form 2x^3 - 30x^2 + cx -/
def P (c : ℤ) (x : ℤ) : ℤ := 2 * x^3 - 30 * x^2 + c * x

/-- The property that P(x) yields consecutive integers for consecutive integer inputs -/
def consecutive_values (c : ℤ) : Prop :=
  ∀ a : ℤ, ∃ k : ℤ, P c (a - 1) = k - 1 ∧ P c a = k ∧ P c (a + 1) = k + 1

theorem polynomial_property :
  ∀ c : ℤ, consecutive_values c → c = 149 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l3845_384539


namespace NUMINAMATH_CALUDE_cubic_root_inequality_l3845_384501

/-- Given a cubic polynomial with real coefficients and three real roots,
    prove the inequality involving the difference between the largest and smallest roots. -/
theorem cubic_root_inequality (a b c : ℝ) (α β γ : ℝ) : 
  let p : ℝ → ℝ := λ x => x^3 + a*x^2 + b*x + c
  (∀ x, p x = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  α < β →
  β < γ →
  Real.sqrt (a^2 - 3*b) < γ - α ∧ γ - α ≤ 2 * Real.sqrt ((a^2 / 3) - b) := by
sorry

end NUMINAMATH_CALUDE_cubic_root_inequality_l3845_384501


namespace NUMINAMATH_CALUDE_one_third_of_5_4_l3845_384575

theorem one_third_of_5_4 : (5.4 / 3 : ℚ) = 9 / 5 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_5_4_l3845_384575


namespace NUMINAMATH_CALUDE_range_of_m_l3845_384548

theorem range_of_m (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2)
  (h_ineq : ∀ m : ℝ, (4/a) + 1/(b-1) > m^2 + 8*m) :
  ∀ m : ℝ, (4/a) + 1/(b-1) > m^2 + 8*m → -9 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3845_384548


namespace NUMINAMATH_CALUDE_longest_side_of_special_rectangle_l3845_384544

/-- Given a rectangle with perimeter 240 feet and area equal to eight times its perimeter,
    the length of its longest side is 80 feet. -/
theorem longest_side_of_special_rectangle : 
  ∀ l w : ℝ,
  l > 0 → w > 0 →
  2 * l + 2 * w = 240 →
  l * w = 8 * (2 * l + 2 * w) →
  max l w = 80 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_special_rectangle_l3845_384544


namespace NUMINAMATH_CALUDE_abs_x_minus_sqrt_x_minus_one_squared_l3845_384592

theorem abs_x_minus_sqrt_x_minus_one_squared (x : ℝ) (h : x < 0) :
  |x - Real.sqrt ((x - 1)^2)| = 1 - 2*x := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_sqrt_x_minus_one_squared_l3845_384592


namespace NUMINAMATH_CALUDE_jade_stone_volume_sum_l3845_384526

-- Define the weights per cubic inch
def jade_weight_per_cubic_inch : ℝ := 7
def stone_weight_per_cubic_inch : ℝ := 6

-- Define the edge length of the cubic stone
def edge_length : ℝ := 3

-- Define the total weight in taels
def total_weight : ℝ := 176

-- Theorem statement
theorem jade_stone_volume_sum (x y : ℝ) 
  (h1 : x + y = total_weight) 
  (h2 : x ≥ 0) 
  (h3 : y ≥ 0) : 
  x / jade_weight_per_cubic_inch + y / stone_weight_per_cubic_inch = edge_length ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_jade_stone_volume_sum_l3845_384526


namespace NUMINAMATH_CALUDE_larger_integer_value_l3845_384560

theorem larger_integer_value (a b : ℕ+) 
  (h_quotient : (a : ℚ) / (b : ℚ) = 7 / 3)
  (h_product : (a : ℕ) * (b : ℕ) = 189) :
  a = 21 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_value_l3845_384560


namespace NUMINAMATH_CALUDE_number_of_possible_sums_l3845_384590

def bag_A : Finset ℕ := {1, 3, 5}
def bag_B : Finset ℕ := {2, 4, 6}

def possible_sums : Finset ℕ := (bag_A.product bag_B).image (fun p => p.1 + p.2)

theorem number_of_possible_sums : possible_sums.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_possible_sums_l3845_384590


namespace NUMINAMATH_CALUDE_soap_brand_ratio_l3845_384523

theorem soap_brand_ratio (total : ℕ) (neither : ℕ) (only_a : ℕ) (both : ℕ) 
  (h1 : total = 300)
  (h2 : neither = 80)
  (h3 : only_a = 60)
  (h4 : both = 40) :
  (total - neither - only_a - both) / both = 3 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_ratio_l3845_384523


namespace NUMINAMATH_CALUDE_min_value_of_a_l3845_384558

-- Define the set of x values
def X : Set ℝ := { x | 0 < x ∧ x ≤ 1/2 }

-- Define the inequality condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x ∈ X, x^2 + a*x + 1 ≥ 0

-- State the theorem
theorem min_value_of_a :
  (∃ a_min : ℝ, inequality_holds a_min ∧
    ∀ a : ℝ, inequality_holds a → a ≥ a_min) ∧
  (∀ a_min : ℝ, (inequality_holds a_min ∧
    ∀ a : ℝ, inequality_holds a → a ≥ a_min) →
    a_min = -5/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3845_384558


namespace NUMINAMATH_CALUDE_repeating_decimal_value_l3845_384559

/-- The decimal representation 0.7overline{23}15 as a rational number -/
def repeating_decimal : ℚ := 62519 / 66000

/-- Theorem stating that 0.7overline{23}15 is equal to 62519/66000 -/
theorem repeating_decimal_value : repeating_decimal = 0.7 + 0.015 + (23 : ℚ) / 990 := by
  sorry

#eval repeating_decimal

end NUMINAMATH_CALUDE_repeating_decimal_value_l3845_384559


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3845_384547

theorem sqrt_product_equality : Real.sqrt 72 * Real.sqrt 18 * Real.sqrt 8 = 72 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3845_384547


namespace NUMINAMATH_CALUDE_class_gender_ratio_l3845_384598

theorem class_gender_ratio :
  ∀ (girls boys : ℕ),
  girls + boys = 28 →
  girls = boys + 4 →
  (girls : ℚ) / (boys : ℚ) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_class_gender_ratio_l3845_384598


namespace NUMINAMATH_CALUDE_add_squared_terms_l3845_384556

theorem add_squared_terms (a : ℝ) : a^2 + 3*a^2 = 4*a^2 := by
  sorry

end NUMINAMATH_CALUDE_add_squared_terms_l3845_384556


namespace NUMINAMATH_CALUDE_aaron_brothers_count_l3845_384535

theorem aaron_brothers_count : ∃ (a : ℕ), a = 4 ∧ 6 = 2 * a - 2 := by
  sorry

end NUMINAMATH_CALUDE_aaron_brothers_count_l3845_384535


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3845_384588

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sixth_term_of_geometric_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 4)^2 - 8*(a 4) + 9 = 0 →
  (a 8)^2 - 8*(a 8) + 9 = 0 →
  a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l3845_384588


namespace NUMINAMATH_CALUDE_temp_difference_l3845_384563

-- Define the temperatures
def southern_temp : Int := -7
def northern_temp : Int := -15

-- State the theorem
theorem temp_difference : southern_temp - northern_temp = 8 := by
  sorry

end NUMINAMATH_CALUDE_temp_difference_l3845_384563


namespace NUMINAMATH_CALUDE_sequence_problem_l3845_384506

/-- Given a sequence {aₙ} that satisfies the recurrence relation
    aₙ₊₁/(n+1) = aₙ/n for all n, and a₅ = 15, prove that a₈ = 24. -/
theorem sequence_problem (a : ℕ → ℚ)
    (h1 : ∀ n, a (n + 1) / (n + 1) = a n / n)
    (h2 : a 5 = 15) :
    a 8 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sequence_problem_l3845_384506


namespace NUMINAMATH_CALUDE_box_surface_area_l3845_384589

/-- Calculates the surface area of the interior of an open box formed by removing square corners from a rectangular sheet and folding up the sides. -/
def interior_surface_area (sheet_length : ℕ) (sheet_width : ℕ) (corner_size : ℕ) : ℕ :=
  let modified_area := sheet_length * sheet_width
  let corner_area := corner_size * corner_size
  let total_removed_area := 4 * corner_area
  modified_area - total_removed_area

/-- Theorem stating that the surface area of the interior of the box is 804 square units. -/
theorem box_surface_area :
  interior_surface_area 25 40 7 = 804 := by
  sorry

#eval interior_surface_area 25 40 7

end NUMINAMATH_CALUDE_box_surface_area_l3845_384589


namespace NUMINAMATH_CALUDE_vacation_cost_difference_l3845_384515

theorem vacation_cost_difference (total_cost : ℕ) (initial_people : ℕ) (new_people : ℕ) : 
  total_cost = 375 → initial_people = 3 → new_people = 5 → 
  (total_cost / initial_people) - (total_cost / new_people) = 50 := by
sorry

end NUMINAMATH_CALUDE_vacation_cost_difference_l3845_384515


namespace NUMINAMATH_CALUDE_triangle_area_ratio_bounds_l3845_384564

theorem triangle_area_ratio_bounds (a b c r R : ℝ) (S S₁ : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 → R > 0 → S > 0 →
  6 * (a + b + c) * r^2 = a * b * c →
  R = 3 * r →
  S = (r * (a + b + c)) / 2 →
  ∃ (M : ℝ × ℝ), 
    (5 - 2 * Real.sqrt 3) / 36 ≤ S₁ / S ∧ 
    S₁ / S ≤ (5 + 2 * Real.sqrt 3) / 36 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_ratio_bounds_l3845_384564


namespace NUMINAMATH_CALUDE_complex_modulus_l3845_384597

theorem complex_modulus (z : ℂ) (h : z * (1 + Complex.I) = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3845_384597


namespace NUMINAMATH_CALUDE_simplified_expression_approximation_l3845_384519

theorem simplified_expression_approximation :
  let expr := Real.sqrt 5 * 5^(1/3) + 18 / (2^2) * 3 - 8^(3/2)
  ∃ ε > 0, |expr + 1.8| < ε ∧ ε < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_simplified_expression_approximation_l3845_384519


namespace NUMINAMATH_CALUDE_floor_sum_possibilities_l3845_384513

theorem floor_sum_possibilities (x y z : ℝ) 
  (hx : ⌊x⌋ = 5) (hy : ⌊y⌋ = -3) (hz : ⌊z⌋ = -2) : 
  ∃ (a b c : ℤ), (a < b ∧ b < c) ∧ 
    (∀ n : ℤ, ⌊x - y + z⌋ = n ↔ n = a ∨ n = b ∨ n = c) := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_possibilities_l3845_384513


namespace NUMINAMATH_CALUDE_green_hat_cost_is_seven_l3845_384541

/-- The cost of each green hat given the total number of hats, number of green hats,
    cost of blue hats, and total price of all hats -/
def green_hat_cost (total_hats : ℕ) (green_hats : ℕ) (blue_hat_cost : ℕ) (total_price : ℕ) : ℕ :=
  (total_price - blue_hat_cost * (total_hats - green_hats)) / green_hats

/-- Theorem stating that the cost of each green hat is 7 under the given conditions -/
theorem green_hat_cost_is_seven :
  green_hat_cost 85 38 6 548 = 7 := by
  sorry

end NUMINAMATH_CALUDE_green_hat_cost_is_seven_l3845_384541


namespace NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l3845_384500

theorem smallest_n_for_exact_tax : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → m < n → ¬∃ (x : ℕ), x > 0 ∧ 107 * x = 100 * m) ∧
  (∃ (x : ℕ), x > 0 ∧ 107 * x = 100 * n) ∧
  n = 107 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_exact_tax_l3845_384500


namespace NUMINAMATH_CALUDE_well_diameter_l3845_384522

/-- Proves that a circular well with given depth and volume has a specific diameter -/
theorem well_diameter (depth : ℝ) (volume : ℝ) (π : ℝ) :
  depth = 10 →
  volume = 31.41592653589793 →
  π = 3.141592653589793 →
  (2 * (volume / (π * depth)))^(1/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_well_diameter_l3845_384522


namespace NUMINAMATH_CALUDE_tom_marble_groups_l3845_384542

def red_marble : ℕ := 1
def green_marble : ℕ := 1
def blue_marble : ℕ := 1
def black_marble : ℕ := 1
def yellow_marbles : ℕ := 4

def total_marbles : ℕ := red_marble + green_marble + blue_marble + black_marble + yellow_marbles

def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem tom_marble_groups :
  let non_yellow_choices := choose_two (red_marble + green_marble + blue_marble + black_marble) - 1
  let yellow_combinations := choose_two yellow_marbles
  let color_with_yellow := red_marble + green_marble + blue_marble + black_marble
  non_yellow_choices + yellow_combinations + color_with_yellow = 10 :=
by sorry

end NUMINAMATH_CALUDE_tom_marble_groups_l3845_384542


namespace NUMINAMATH_CALUDE_minimize_reciprocal_sum_l3845_384587

theorem minimize_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 30) :
  (1 / a + 1 / b) ≥ 1 / 5 + 1 / 20 ∧
  (1 / a + 1 / b = 1 / 5 + 1 / 20 ↔ a = 5 ∧ b = 20) :=
by sorry

end NUMINAMATH_CALUDE_minimize_reciprocal_sum_l3845_384587


namespace NUMINAMATH_CALUDE_cubes_passed_in_specific_solid_l3845_384503

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_by_diagonal (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating the number of cubes an internal diagonal passes through
    in a 105 × 140 × 195 rectangular solid -/
theorem cubes_passed_in_specific_solid :
  cubes_passed_by_diagonal 105 140 195 = 395 := by
  sorry

end NUMINAMATH_CALUDE_cubes_passed_in_specific_solid_l3845_384503


namespace NUMINAMATH_CALUDE_first_video_length_l3845_384509

/-- Given information about Kimiko's YouTube watching --/
structure YoutubeWatching where
  total_time : ℕ
  second_video_length : ℕ
  last_video_length : ℕ

/-- The theorem stating the length of the first video --/
theorem first_video_length (info : YoutubeWatching)
  (h1 : info.total_time = 510)
  (h2 : info.second_video_length = 270)
  (h3 : info.last_video_length = 60) :
  510 - info.second_video_length - 2 * info.last_video_length = 120 := by
  sorry

#check first_video_length

end NUMINAMATH_CALUDE_first_video_length_l3845_384509


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3845_384537

def polynomial (b₂ b₁ : ℤ) (x : ℤ) : ℤ := x^3 + b₂ * x^2 + b₁ * x - 30

def divisors_of_30 : Set ℤ := {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30}

theorem integer_roots_of_polynomial (b₂ b₁ : ℤ) :
  {x : ℤ | polynomial b₂ b₁ x = 0} = divisors_of_30 := by sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l3845_384537


namespace NUMINAMATH_CALUDE_garden_length_is_140_l3845_384533

/-- Represents a rectangular garden -/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.breadth)

/-- Theorem: A rectangular garden with perimeter 480 m and breadth 100 m has length 140 m -/
theorem garden_length_is_140
  (g : RectangularGarden)
  (h1 : perimeter g = 480)
  (h2 : g.breadth = 100) :
  g.length = 140 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_is_140_l3845_384533


namespace NUMINAMATH_CALUDE_second_number_difference_l3845_384561

theorem second_number_difference (first second : ℤ) : 
  first + second = 56 → second = 30 → second - first = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_number_difference_l3845_384561


namespace NUMINAMATH_CALUDE_unique_intersection_l3845_384571

/-- Three lines in the 2D plane -/
structure ThreeLines where
  line1 : ℝ → ℝ → ℝ
  line2 : ℝ → ℝ → ℝ
  line3 : ℝ → ℝ → ℝ

/-- The intersection point of three lines -/
def intersection (lines : ThreeLines) (k : ℝ) : Set (ℝ × ℝ) :=
  {p | lines.line1 p.1 p.2 = 0 ∧ lines.line2 p.1 p.2 = 0 ∧ lines.line3 p.1 p.2 = 0}

/-- The theorem stating that k = -1/2 is the unique value for which the given lines intersect at a single point -/
theorem unique_intersection : ∃! k : ℝ, 
  let lines := ThreeLines.mk
    (fun x y => x + k * y)
    (fun x y => 2 * x + 3 * y + 8)
    (fun x y => x - y - 1)
  (∃! p : ℝ × ℝ, p ∈ intersection lines k) ∧ k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_intersection_l3845_384571


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3845_384553

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Theorem: A point M with coordinates (a, b), where a < 0 and b > 0, is in the second quadrant -/
theorem point_in_second_quadrant (a b : ℝ) (ha : a < 0) (hb : b > 0) :
  SecondQuadrant ⟨a, b⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3845_384553


namespace NUMINAMATH_CALUDE_inequality_properties_l3845_384566

theorem inequality_properties (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  (1 / a > 1 / b) ∧
  (a^(1/5 : ℝ) < b^(1/5 : ℝ)) ∧
  (Real.sqrt (a^2 - a) > Real.sqrt (b^2 - b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_properties_l3845_384566


namespace NUMINAMATH_CALUDE_mothers_age_l3845_384583

theorem mothers_age (daughter_age_in_3_years : ℕ) 
  (h1 : daughter_age_in_3_years = 26) 
  (h2 : ∃ (mother_age_5_years_ago daughter_age_5_years_ago : ℕ), 
    mother_age_5_years_ago = 2 * daughter_age_5_years_ago) : 
  ∃ (mother_current_age : ℕ), mother_current_age = 41 := by
sorry

end NUMINAMATH_CALUDE_mothers_age_l3845_384583


namespace NUMINAMATH_CALUDE_discount_percentage_proof_l3845_384536

theorem discount_percentage_proof (num_toys : ℕ) (cost_per_toy : ℚ) (total_paid : ℚ) :
  num_toys = 5 →
  cost_per_toy = 3 →
  total_paid = 12 →
  (1 - total_paid / (num_toys * cost_per_toy)) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_proof_l3845_384536


namespace NUMINAMATH_CALUDE_prob_two_heads_and_three_l3845_384502

-- Define the probability of getting heads on a fair coin
def prob_heads : ℚ := 1/2

-- Define the probability of rolling a 3 on a fair six-sided die
def prob_three : ℚ := 1/6

-- State the theorem
theorem prob_two_heads_and_three (h1 : prob_heads = 1/2) (h2 : prob_three = 1/6) : 
  prob_heads * prob_heads * prob_three = 1/24 := by
  sorry


end NUMINAMATH_CALUDE_prob_two_heads_and_three_l3845_384502


namespace NUMINAMATH_CALUDE_divisor_problem_l3845_384524

theorem divisor_problem (D : ℚ) : 
  (1280 + 720) / 125 = 7392 / D → D = 462 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l3845_384524


namespace NUMINAMATH_CALUDE_line_transformation_l3845_384529

/-- The analytical expression of a line after transformation -/
def transformed_line (a b : ℝ) (dx dy : ℝ) : ℝ → ℝ := fun x ↦ a * (x + dx) + b + dy

/-- The original line y = 2x - 1 -/
def original_line : ℝ → ℝ := fun x ↦ 2 * x - 1

theorem line_transformation :
  transformed_line 2 (-1) 1 (-2) = original_line := by sorry

end NUMINAMATH_CALUDE_line_transformation_l3845_384529


namespace NUMINAMATH_CALUDE_prob_at_least_one_man_l3845_384528

/-- The probability of selecting at least one man when choosing 5 people at random from a group of 12 men and 8 women -/
theorem prob_at_least_one_man (total_people : ℕ) (men : ℕ) (women : ℕ) (selection_size : ℕ) :
  total_people = men + women →
  men = 12 →
  women = 8 →
  selection_size = 5 →
  (1 : ℚ) - (women.choose selection_size : ℚ) / (total_people.choose selection_size : ℚ) = 687 / 692 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_man_l3845_384528


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l3845_384573

/-- The equations of the asymptotes for the hyperbola x²/16 - y²/9 = 1 are y = ±(3/4)x -/
theorem hyperbola_asymptotes :
  let h : ℝ → ℝ → Prop := fun x y ↦ x^2/16 - y^2/9 = 1
  ∃ (f g : ℝ → ℝ), (∀ x, f x = (3/4) * x) ∧ (∀ x, g x = -(3/4) * x) ∧
    (∀ ε > 0, ∃ M > 0, ∀ x y, h x y → (|x| > M → |y - f x| < ε ∨ |y - g x| < ε)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l3845_384573


namespace NUMINAMATH_CALUDE_square_diff_and_product_l3845_384549

theorem square_diff_and_product (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (diff_eq : a - b = 4) 
  (sum_squares_eq : a^2 + b^2 = 58) : 
  a^2 - b^2 = 40 ∧ a * b = 21 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_and_product_l3845_384549


namespace NUMINAMATH_CALUDE_fraction_always_nonnegative_l3845_384578

theorem fraction_always_nonnegative (x : ℝ) : (x^2 + 2*x + 1) / (x^2 + 4*x + 8) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_always_nonnegative_l3845_384578


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3845_384552

-- Define the original expression
def original_expr (a b : ℝ) : ℝ := 3*a^2*b - (2*a^2*b - (2*a*b - a^2*b) - 4*a^2) - a*b

-- Define the simplified expression
def simplified_expr (a b : ℝ) : ℝ := a*b + 4*a^2

-- Theorem statement
theorem expression_simplification_and_evaluation :
  (∀ a b : ℝ, original_expr a b = simplified_expr a b) ∧
  (original_expr (-3) (-2) = 22) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3845_384552


namespace NUMINAMATH_CALUDE_profit_percentage_doubling_l3845_384555

theorem profit_percentage_doubling (cost_price : ℝ) (original_selling_price : ℝ) :
  original_selling_price = cost_price * 1.3 →
  let double_price := original_selling_price * 2
  let new_profit_percentage := (double_price - cost_price) / cost_price * 100
  new_profit_percentage = 160 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_doubling_l3845_384555


namespace NUMINAMATH_CALUDE_monthly_average_production_l3845_384517

/-- The daily average TV production for an entire month, given production rates for different periods. -/
theorem monthly_average_production
  (days_in_month : ℕ)
  (first_period_days : ℕ)
  (second_period_days : ℕ)
  (first_period_avg : ℝ)
  (second_period_avg : ℝ)
  (h_total_days : days_in_month = first_period_days + second_period_days)
  (h_first_period : first_period_days = 25)
  (h_second_period : second_period_days = 5)
  (h_first_avg : first_period_avg = 70)
  (h_second_avg : second_period_avg = 58) :
  (first_period_avg * first_period_days + second_period_avg * second_period_days) / days_in_month = 68 :=
sorry

end NUMINAMATH_CALUDE_monthly_average_production_l3845_384517


namespace NUMINAMATH_CALUDE_extreme_points_inequality_l3845_384565

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x - 1 - a * Real.log x

theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ →
  (∀ x, x > 0 → (deriv (f a)) x = 0 ↔ x = x₁ ∨ x = x₂) →
  (f a x₁) / x₂ > -7/2 - Real.log 2 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_inequality_l3845_384565


namespace NUMINAMATH_CALUDE_log_equality_l3845_384531

theorem log_equality (x k : ℝ) :
  (Real.log 3 / Real.log 4 = x) →
  (Real.log 9 / Real.log 2 = k * x) →
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_log_equality_l3845_384531


namespace NUMINAMATH_CALUDE_ninth_term_of_sequence_l3845_384577

/-- The nth term of a geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The 9th term of the geometric sequence with first term 4 and common ratio 1 is 4 -/
theorem ninth_term_of_sequence : geometric_sequence 4 1 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_of_sequence_l3845_384577


namespace NUMINAMATH_CALUDE_tub_ratio_is_one_third_l3845_384508

/-- Represents the number of tubs in various categories -/
structure TubCounts where
  total : ℕ
  storage : ℕ
  usual_vendor : ℕ

/-- Calculates the ratio of tubs bought from new vendor to usual vendor -/
def tub_ratio (t : TubCounts) : Rat :=
  let new_vendor := t.total - t.storage - t.usual_vendor
  (new_vendor : Rat) / t.usual_vendor

/-- Theorem stating the ratio of tubs bought from new vendor to usual vendor -/
theorem tub_ratio_is_one_third (t : TubCounts) 
  (h_total : t.total = 100)
  (h_storage : t.storage = 20)
  (h_usual : t.usual_vendor = 60) :
  tub_ratio t = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tub_ratio_is_one_third_l3845_384508


namespace NUMINAMATH_CALUDE_trapezoid_segment_length_l3845_384574

/-- Represents a rectangle with given dimensions -/
structure Rectangle where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a trapezoid formed after removing segments from a rectangle -/
structure Trapezoid where
  longBase : ℝ
  shortBase : ℝ
  height : ℝ

/-- Calculates the total length of segments in the trapezoid -/
def totalLength (t : Trapezoid) : ℝ :=
  t.longBase + t.shortBase + t.height

/-- Theorem stating that the total length of segments in the resulting trapezoid is 19 units -/
theorem trapezoid_segment_length 
  (r : Rectangle)
  (t : Trapezoid)
  (h1 : r.length = 11)
  (h2 : r.width = 3)
  (h3 : r.height = 12)
  (h4 : t.longBase = 8)
  (h5 : t.shortBase = r.width)
  (h6 : t.height = r.height - 4) :
  totalLength t = 19 := by
    sorry


end NUMINAMATH_CALUDE_trapezoid_segment_length_l3845_384574


namespace NUMINAMATH_CALUDE_distance_between_points_l3845_384504

theorem distance_between_points : 
  ∀ (A B : ℝ), A = -4 ∧ B = 2 → |B - A| = |2 - (-4)| := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l3845_384504


namespace NUMINAMATH_CALUDE_coin_problem_l3845_384596

theorem coin_problem (n d h : ℕ) : 
  n + d + h = 150 →
  5*n + 10*d + 50*h = 1250 →
  ∃ (d_min d_max : ℕ), 
    (∃ (n' h' : ℕ), n' + d_min + h' = 150 ∧ 5*n' + 10*d_min + 50*h' = 1250) ∧
    (∃ (n'' h'' : ℕ), n'' + d_max + h'' = 150 ∧ 5*n'' + 10*d_max + 50*h'' = 1250) ∧
    d_max - d_min = 99 :=
by sorry

end NUMINAMATH_CALUDE_coin_problem_l3845_384596


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3845_384516

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x - 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x - 3 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3845_384516


namespace NUMINAMATH_CALUDE_triangle_area_l3845_384599

theorem triangle_area (a b c : ℝ) (h1 : a = 13) (h2 : b = 14) (h3 : c = 15) : 
  let S := (1/2) * a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)
  S = 84 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3845_384599


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l3845_384593

theorem solve_quadratic_equation (B : ℝ) : 3 * B^2 + 3 * B + 2 = 29 →
  B = (-1 + Real.sqrt 37) / 2 ∨ B = (-1 - Real.sqrt 37) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l3845_384593


namespace NUMINAMATH_CALUDE_triangle_construction_l3845_384534

/-- Given a triangle ABC with vertices A(-1, 0), B(1, 0), and C(3a, 3b),
    this theorem proves that it satisfies the specified conditions. -/
theorem triangle_construction (a b : ℝ) (OH_length AB_length : ℝ) :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 0)
  let C : ℝ × ℝ := (3*a, 3*b)
  let O : ℝ × ℝ := (0, b)  -- Circumcenter
  let H : ℝ × ℝ := (3*a, b)  -- Orthocenter
  -- Distance between O and H
  (O.1 - H.1)^2 + (O.2 - H.2)^2 = OH_length^2 ∧
  -- OH parallel to AB
  (O.2 - H.2) * (A.1 - B.1) = (O.1 - H.1) * (A.2 - B.2) ∧
  -- Length of AB
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB_length^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_construction_l3845_384534


namespace NUMINAMATH_CALUDE_new_person_weight_l3845_384521

theorem new_person_weight (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  n = 7 ∧ avg_increase = 3.5 ∧ old_weight = 75 →
  (n : ℝ) * avg_increase + old_weight = 99.5 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3845_384521


namespace NUMINAMATH_CALUDE_negative_number_identification_l3845_384525

theorem negative_number_identification (a b c d : ℝ) 
  (ha : a = -6) (hb : b = 0) (hc : c = 0.2) (hd : d = 3) :
  a < 0 ∧ b ≥ 0 ∧ c > 0 ∧ d > 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_number_identification_l3845_384525


namespace NUMINAMATH_CALUDE_train_length_l3845_384579

theorem train_length (t_platform : ℝ) (t_pole : ℝ) (l_platform : ℝ) 
  (h1 : t_platform = 36)
  (h2 : t_pole = 18)
  (h3 : l_platform = 300) :
  ∃ l_train : ℝ, l_train = 300 ∧ l_train / t_pole = (l_train + l_platform) / t_platform :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l3845_384579


namespace NUMINAMATH_CALUDE_second_quarter_profit_l3845_384510

theorem second_quarter_profit 
  (annual_profit : ℕ)
  (first_quarter_profit : ℕ)
  (third_quarter_profit : ℕ)
  (fourth_quarter_profit : ℕ)
  (h1 : annual_profit = 8000)
  (h2 : first_quarter_profit = 1500)
  (h3 : third_quarter_profit = 3000)
  (h4 : fourth_quarter_profit = 2000) :
  annual_profit - (first_quarter_profit + third_quarter_profit + fourth_quarter_profit) = 1500 :=
by
  sorry

end NUMINAMATH_CALUDE_second_quarter_profit_l3845_384510


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3845_384514

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 3) * (x + 2) = m + 3 * x) ↔ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3845_384514


namespace NUMINAMATH_CALUDE_johns_annual_profit_l3845_384554

/-- Calculates the annual profit from subletting an apartment --/
def annual_profit (num_subletters : ℕ) (subletter_rent : ℕ) (apartment_rent : ℕ) : ℕ :=
  (num_subletters * subletter_rent * 12) - (apartment_rent * 12)

/-- Theorem: John's annual profit from subletting his apartment is $3600 --/
theorem johns_annual_profit :
  annual_profit 3 400 900 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_johns_annual_profit_l3845_384554


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l3845_384527

def physics_students : ℕ := 200
def biology_students : ℕ := physics_students / 2
def boys_in_biology : ℕ := 25

def girls_in_biology : ℕ := biology_students - boys_in_biology

theorem girls_to_boys_ratio :
  girls_in_biology / boys_in_biology = 3 := by sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l3845_384527


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3845_384530

def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x : ℝ | x > 3}

theorem union_of_M_and_N : M ∪ N = {x : ℝ | x > -3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3845_384530


namespace NUMINAMATH_CALUDE_no_natural_square_diff_2014_l3845_384557

theorem no_natural_square_diff_2014 : ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_square_diff_2014_l3845_384557


namespace NUMINAMATH_CALUDE_hydropump_volume_l3845_384538

/-- Represents the rate of water pumping in gallons per hour -/
def pump_rate : ℝ := 600

/-- Represents the time in hours -/
def pump_time : ℝ := 1.5

/-- Represents the volume of water pumped in gallons -/
def water_volume : ℝ := pump_rate * pump_time

theorem hydropump_volume : water_volume = 900 := by
  sorry

end NUMINAMATH_CALUDE_hydropump_volume_l3845_384538


namespace NUMINAMATH_CALUDE_license_plate_theorem_l3845_384505

def vowels : Nat := 5
def consonants : Nat := 21
def odd_digits : Nat := 5
def even_digits : Nat := 5

def license_plate_count : Nat :=
  (vowels^2 + consonants^2) * odd_digits * even_digits^2

theorem license_plate_theorem :
  license_plate_count = 58250 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_theorem_l3845_384505


namespace NUMINAMATH_CALUDE_smallest_integer_solution_minus_four_is_smallest_l3845_384569

theorem smallest_integer_solution (x : ℤ) : (7 - 3 * x < 22) ↔ (x ≥ -4) :=
  sorry

theorem minus_four_is_smallest : ∀ y : ℤ, (7 - 3 * y < 22) → y ≥ -4 :=
  sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_minus_four_is_smallest_l3845_384569


namespace NUMINAMATH_CALUDE_f_properties_l3845_384591

def f (x : ℝ) : ℝ := (x - 2)^2

theorem f_properties :
  (∀ x, f (x + 2) = f (-x + 2)) ∧ 
  (∀ x y, x < y → x < 2 → f x > f y) ∧
  (∀ x y, x < y → y > 2 → f x < f y) ∧
  (∀ x y, x < y → f (x + 2) - f x < f (y + 2) - f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l3845_384591


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3845_384551

def S (n : ℕ) : ℕ := 11 * (10^n - 1) / 9

def arithmetic_mean (nums : List ℕ) : ℚ :=
  (nums.sum : ℚ) / nums.length

theorem arithmetic_mean_of_special_set :
  let nums := List.range 9
  let special_set := nums.map (fun i => S (i + 1))
  let mean := arithmetic_mean special_set
  ∃ (n : ℕ),
    n = ⌊mean⌋ ∧
    n ≥ 100000000 ∧ n < 1000000000 ∧
    (List.range 10).all (fun d => d ≠ 0 → (n / 10^d % 10 ≠ n / 10^(d+1) % 10)) ∧
    n % 10 ≠ 0 ∧
    (n / 10 % 10) ≠ 0 ∧
    (n / 100 % 10) ≠ 0 ∧
    (n / 1000 % 10) ≠ 0 ∧
    (n / 10000 % 10) ≠ 0 ∧
    (n / 100000 % 10) ≠ 0 ∧
    (n / 1000000 % 10) ≠ 0 ∧
    (n / 10000000 % 10) ≠ 0 ∧
    (n / 100000000 % 10) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3845_384551


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l3845_384518

/-- Given a cylinder with original volume of 15 cubic feet, prove that tripling its radius and halving its height results in a new volume of 67.5 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) : 
  π * r^2 * h = 15 → π * (3*r)^2 * (h/2) = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l3845_384518


namespace NUMINAMATH_CALUDE_chord_line_equation_l3845_384572

/-- The equation of a line containing a chord of an ellipse, given the ellipse equation and the midpoint of the chord. -/
theorem chord_line_equation (a b c : ℝ) (x₀ y₀ : ℝ) :
  (∀ x y, x^2 + a*y^2 = b) →  -- Ellipse equation
  (∃ x₁ y₁ x₂ y₂ : ℝ,  -- Existence of chord endpoints
    x₁^2 + a*y₁^2 = b ∧
    x₂^2 + a*y₂^2 = b ∧
    x₀ = (x₁ + x₂) / 2 ∧
    y₀ = (y₁ + y₂) / 2) →
  (∃ k m : ℝ, ∀ x y, (x - x₀) + k*(y - y₀) = 0 ↔ x + k*y = m) →
  (a = 4 ∧ b = 36 ∧ x₀ = 4 ∧ y₀ = 2 ∧ c = 8) →
  (∀ x y, x + 2*y - c = 0 ↔ (x - x₀) + 2*(y - y₀) = 0) :=
by sorry

#check chord_line_equation

end NUMINAMATH_CALUDE_chord_line_equation_l3845_384572


namespace NUMINAMATH_CALUDE_second_grade_years_l3845_384586

/-- Given information about Mrs. Randall's teaching career -/
def total_teaching_years : ℕ := 26
def third_grade_years : ℕ := 18

/-- Theorem stating the number of years Mrs. Randall taught second grade -/
theorem second_grade_years : total_teaching_years - third_grade_years = 8 := by
  sorry

end NUMINAMATH_CALUDE_second_grade_years_l3845_384586


namespace NUMINAMATH_CALUDE_odd_prime_square_difference_l3845_384550

theorem odd_prime_square_difference (d : ℕ) : 
  Nat.Prime d → 
  d % 2 = 1 → 
  ∃ m : ℕ, 89 - (d + 3)^2 = m^2 → 
  d = 5 := by sorry

end NUMINAMATH_CALUDE_odd_prime_square_difference_l3845_384550


namespace NUMINAMATH_CALUDE_cannot_reach_2000_l3845_384576

theorem cannot_reach_2000 (a b : ℕ) : a * 12 + b * 17 ≠ 2000 := by
  sorry

end NUMINAMATH_CALUDE_cannot_reach_2000_l3845_384576


namespace NUMINAMATH_CALUDE_division_problem_l3845_384594

theorem division_problem (n : ℕ) : n / 4 = 12 → n / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3845_384594


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3845_384545

/-- A quadratic function that opens upwards and passes through (0,1) -/
def QuadraticFunction (a b : ℝ) (h : a > 0) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + 1

theorem quadratic_function_properties (a b : ℝ) (h : a > 0) :
  (QuadraticFunction a b h) 0 = 1 ∧
  ∀ x y : ℝ, x < y → (QuadraticFunction a b h) x < (QuadraticFunction a b h) y :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3845_384545


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l3845_384512

def is_factor (a b : ℕ) : Prop := b % a = 0

def are_consecutive (a b : ℕ) : Prop := b = a + 1 ∨ a = b + 1

theorem smallest_non_factor_product_of_48 (x y : ℕ) :
  x ≠ y →
  x > 0 →
  y > 0 →
  is_factor x 48 →
  is_factor y 48 →
  ¬ are_consecutive x y →
  ¬ is_factor (x * y) 48 →
  ∀ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ is_factor a 48 ∧ is_factor b 48 ∧ ¬ are_consecutive a b ∧ ¬ is_factor (a * b) 48 →
  x * y ≤ a * b →
  x * y = 18 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_of_48_l3845_384512


namespace NUMINAMATH_CALUDE_farmer_tomatoes_l3845_384532

def initial_tomatoes (picked_yesterday : ℕ) (picked_today : ℕ) (remaining : ℕ) : ℕ :=
  picked_yesterday + picked_today + remaining

theorem farmer_tomatoes : initial_tomatoes 134 30 7 = 171 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomatoes_l3845_384532


namespace NUMINAMATH_CALUDE_min_snakes_owned_l3845_384543

/-- Represents the number of people owning a specific combination of pets -/
structure PetOwnership where
  total : ℕ
  onlyDogs : ℕ
  onlyCats : ℕ
  catsAndDogs : ℕ
  allThree : ℕ

/-- The given pet ownership data -/
def givenData : PetOwnership :=
  { total := 59
  , onlyDogs := 15
  , onlyCats := 10
  , catsAndDogs := 5
  , allThree := 3 }

/-- The minimum number of snakes owned -/
def minSnakes : ℕ := givenData.allThree

theorem min_snakes_owned (data : PetOwnership) : 
  data.allThree ≤ minSnakes := by sorry

end NUMINAMATH_CALUDE_min_snakes_owned_l3845_384543


namespace NUMINAMATH_CALUDE_milk_chocolate_caramel_percentage_l3845_384511

/-- The percentage of milk chocolate with caramel bars in a box of chocolates -/
theorem milk_chocolate_caramel_percentage
  (milk : ℕ)
  (dark : ℕ)
  (milk_almond : ℕ)
  (white : ℕ)
  (milk_caramel : ℕ)
  (h_milk : milk = 36)
  (h_dark : dark = 21)
  (h_milk_almond : milk_almond = 40)
  (h_white : white = 15)
  (h_milk_caramel : milk_caramel = 28) :
  (milk_caramel : ℚ) / (milk + dark + milk_almond + white + milk_caramel) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_milk_chocolate_caramel_percentage_l3845_384511


namespace NUMINAMATH_CALUDE_square_diagonals_equal_l3845_384582

-- Define the basic shapes
class Rectangle where
  diagonals_equal : Bool

class Square extends Rectangle

-- Define the properties
axiom rectangle_diagonals_equal : ∀ (r : Rectangle), r.diagonals_equal = true

-- Theorem to prove
theorem square_diagonals_equal (s : Square) : s.diagonals_equal = true := by
  sorry

end NUMINAMATH_CALUDE_square_diagonals_equal_l3845_384582


namespace NUMINAMATH_CALUDE_power_seven_145_mod_12_l3845_384520

theorem power_seven_145_mod_12 : 7^145 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_seven_145_mod_12_l3845_384520


namespace NUMINAMATH_CALUDE_triangle_with_integer_altitudes_and_prime_inradius_l3845_384546

/-- Represents a triangle with given side lengths -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Calculates the semi-perimeter of a triangle -/
def semiPerimeter (t : Triangle) : ℚ :=
  (t.a.val + t.b.val + t.c.val) / 2

/-- Calculates the area of a triangle using Heron's formula -/
def area (t : Triangle) : ℚ :=
  let s := semiPerimeter t
  (s * (s - t.a.val) * (s - t.b.val) * (s - t.c.val)).sqrt

/-- Calculates the inradius of a triangle -/
def inradius (t : Triangle) : ℚ :=
  area t / semiPerimeter t

/-- Calculates the altitude to side a of a triangle -/
def altitudeA (t : Triangle) : ℚ :=
  2 * area t / t.a.val

/-- Calculates the altitude to side b of a triangle -/
def altitudeB (t : Triangle) : ℚ :=
  2 * area t / t.b.val

/-- Calculates the altitude to side c of a triangle -/
def altitudeC (t : Triangle) : ℚ :=
  2 * area t / t.c.val

/-- States that a number is prime -/
def isPrime (n : ℕ) : Prop :=
  Nat.Prime n

theorem triangle_with_integer_altitudes_and_prime_inradius :
  ∃ (t : Triangle),
    t.a = 13 ∧ t.b = 14 ∧ t.c = 15 ∧
    (altitudeA t).isInt ∧ (altitudeB t).isInt ∧ (altitudeC t).isInt ∧
    ∃ (r : ℕ), (inradius t) = r ∧ isPrime r :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_integer_altitudes_and_prime_inradius_l3845_384546


namespace NUMINAMATH_CALUDE_solution_set_abs_inequality_l3845_384562

theorem solution_set_abs_inequality :
  {x : ℝ | |1 - 2*x| < 3} = Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_abs_inequality_l3845_384562


namespace NUMINAMATH_CALUDE_circumcircle_radius_from_centroid_distance_l3845_384585

/-- Given a triangle ABC with sides a, b, c, where c = AB, prove that if
    (b - c) / (a + c) = (c - a) / (b + c), then the radius R of the circumcircle
    satisfies R² = d² + c²/3, where d is the distance from the circumcircle
    center to the centroid of the triangle. -/
theorem circumcircle_radius_from_centroid_distance (a b c d : ℝ) :
  (b - c) / (a + c) = (c - a) / (b + c) →
  ∃ (R : ℝ), R > 0 ∧ R^2 = d^2 + c^2 / 3 :=
sorry

end NUMINAMATH_CALUDE_circumcircle_radius_from_centroid_distance_l3845_384585


namespace NUMINAMATH_CALUDE_unique_quadratic_root_condition_l3845_384568

theorem unique_quadratic_root_condition (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 1/b) * x + c = 0) ↔ 
  c = 3/2 := by
sorry

end NUMINAMATH_CALUDE_unique_quadratic_root_condition_l3845_384568
