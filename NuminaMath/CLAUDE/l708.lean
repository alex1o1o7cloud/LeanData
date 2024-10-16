import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l708_70834

theorem equation_solution (x y : ℚ) : 
  19 * (x + y) + 17 = 19 * (-x + y) - 21 ↔ x = -2/19 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l708_70834


namespace NUMINAMATH_CALUDE_sum_of_cubes_divisible_l708_70830

theorem sum_of_cubes_divisible (a : ℤ) : 
  ∃ k : ℤ, (a - 1)^3 + a^3 + (a + 1)^3 = 3 * a * k := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_divisible_l708_70830


namespace NUMINAMATH_CALUDE_earth_surface_usage_l708_70844

/-- The fraction of the Earth's surface that is land -/
def land_fraction : ℚ := 1/3

/-- The fraction of land that is inhabitable -/
def inhabitable_fraction : ℚ := 2/3

/-- The fraction of inhabitable land used for agriculture and urban development -/
def used_fraction : ℚ := 3/4

/-- The fraction of the Earth's surface used for agriculture or urban purposes -/
def agriculture_urban_fraction : ℚ := land_fraction * inhabitable_fraction * used_fraction

theorem earth_surface_usage :
  agriculture_urban_fraction = 1/6 := by sorry

end NUMINAMATH_CALUDE_earth_surface_usage_l708_70844


namespace NUMINAMATH_CALUDE_arthur_wallet_problem_l708_70822

theorem arthur_wallet_problem (initial_amount : ℝ) (spent_fraction : ℝ) (remaining_amount : ℝ) : 
  initial_amount = 200 →
  spent_fraction = 4/5 →
  remaining_amount = initial_amount - (spent_fraction * initial_amount) →
  remaining_amount = 40 := by
sorry

end NUMINAMATH_CALUDE_arthur_wallet_problem_l708_70822


namespace NUMINAMATH_CALUDE_school_population_l708_70848

/-- Given a school with boys, girls, and teachers, prove that the total number of people is 61t, where t is the number of teachers. -/
theorem school_population (b g t : ℕ) (h1 : b = 4 * g) (h2 : g = 12 * t) : 
  b + g + t = 61 * t := by
  sorry

end NUMINAMATH_CALUDE_school_population_l708_70848


namespace NUMINAMATH_CALUDE_tobys_friends_l708_70893

theorem tobys_friends (total_friends : ℕ) (boy_friends : ℕ) (girl_friends : ℕ) : 
  (boy_friends : ℚ) / total_friends = 55 / 100 →
  boy_friends = 33 →
  girl_friends = 27 :=
by sorry

end NUMINAMATH_CALUDE_tobys_friends_l708_70893


namespace NUMINAMATH_CALUDE_sum_perfect_square_values_l708_70853

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_perfect_square_values :
  ∀ K : ℕ, K > 0 → K < 150 →
    (∃ N : ℕ, N < 150 ∧ sum_first_n K = N * N) ↔ K = 8 ∨ K = 49 ∨ K = 59 := by
  sorry

end NUMINAMATH_CALUDE_sum_perfect_square_values_l708_70853


namespace NUMINAMATH_CALUDE_ngon_triangle_division_l708_70874

/-- 
Given an n-gon divided into k triangles, prove that k ≥ n-2.
-/
theorem ngon_triangle_division (n k : ℕ) (h1 : n ≥ 3) (h2 : k > 0) : k ≥ n - 2 := by
  sorry


end NUMINAMATH_CALUDE_ngon_triangle_division_l708_70874


namespace NUMINAMATH_CALUDE_ellipse_equation_l708_70894

/-- Given an ellipse with equation (x²/a²) + (y²/b²) = 1, where a > b > 0,
    if the minimum value of |k₁| + |k₂| is 1 (where k₁ and k₂ are slopes of lines 
    from any point P on the ellipse to the left and right vertices respectively)
    and the ellipse passes through the point (√3, 1/2), 
    then the equation of the ellipse is x²/4 + y² = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → 
    ∃ k₁ k₂ : ℝ, k₁ * k₂ ≠ 0 ∧ 
    (∀ k₁' k₂' : ℝ, |k₁'| + |k₂'| ≥ |k₁| + |k₂|) ∧
    |k₁| + |k₂| = 1) →
  3 / a^2 + (1/4) / b^2 = 1 →
  ∀ x y : ℝ, x^2 / 4 + y^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l708_70894


namespace NUMINAMATH_CALUDE_tangent_perpendicular_and_inequality_l708_70890

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((4*x + a) * Real.log x) / (3*x + 1)

theorem tangent_perpendicular_and_inequality (a : ℝ) :
  (∃ m : ℝ, ∀ x : ℝ, x ≥ 1 → f a x ≤ m * (x - 1)) →
  (a = 0 ∧ ∀ m : ℝ, (∀ x : ℝ, x ≥ 1 → f a x ≤ m * (x - 1)) → m ≥ 1) :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_and_inequality_l708_70890


namespace NUMINAMATH_CALUDE_ellipse_equation_and_sum_l708_70827

theorem ellipse_equation_and_sum (t : ℝ) :
  let x := (3 * (Real.sin t - 2)) / (3 - Real.cos t)
  let y := (4 * (Real.cos t - 6)) / (3 - Real.cos t)
  ∃ (A B C D E F : ℤ),
    (144 : ℝ) * x^2 - 96 * x * y + 25 * y^2 + 192 * x - 400 * y + 400 = 0 ∧
    Int.gcd A (Int.gcd B (Int.gcd C (Int.gcd D (Int.gcd E F)))) = 1 ∧
    Int.natAbs A + Int.natAbs B + Int.natAbs C + Int.natAbs D + Int.natAbs E + Int.natAbs F = 1257 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_sum_l708_70827


namespace NUMINAMATH_CALUDE_range_of_m_l708_70870

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h_eq : 1/x + 4/y = 1) :
  (∀ x y, x > 0 → y > 0 → 1/x + 4/y = 1 → x + y > m^2 + 8*m) ↔ -9 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l708_70870


namespace NUMINAMATH_CALUDE_joan_missed_games_l708_70819

/-- The number of baseball games Joan missed -/
def games_missed (total_games attended_games : ℕ) : ℕ :=
  total_games - attended_games

/-- Proof that Joan missed 469 baseball games -/
theorem joan_missed_games : games_missed 864 395 = 469 := by
  sorry

end NUMINAMATH_CALUDE_joan_missed_games_l708_70819


namespace NUMINAMATH_CALUDE_smallest_dual_representation_l708_70888

/-- Represents a number in a given base -/
def represent_in_base (n : ℕ) (base : ℕ) : List ℕ := sorry

/-- Converts a number from a given base to base 10 -/
def to_base_10 (digits : List ℕ) (base : ℕ) : ℕ := sorry

/-- Checks if a number can be represented as 13 in base c and 31 in base d -/
def is_valid_representation (n : ℕ) (c : ℕ) (d : ℕ) : Prop :=
  (represent_in_base n c = [1, 3]) ∧ (represent_in_base n d = [3, 1])

theorem smallest_dual_representation :
  ∃ (n : ℕ) (c : ℕ) (d : ℕ),
    c > 3 ∧ d > 3 ∧
    is_valid_representation n c d ∧
    (∀ (m : ℕ) (c' : ℕ) (d' : ℕ),
      c' > 3 → d' > 3 → is_valid_representation m c' d' → n ≤ m) ∧
    n = 13 := by sorry

#check smallest_dual_representation

end NUMINAMATH_CALUDE_smallest_dual_representation_l708_70888


namespace NUMINAMATH_CALUDE_total_tiles_count_l708_70889

def room_length : ℕ := 24
def room_width : ℕ := 18
def border_tile_size : ℕ := 2
def inner_tile_size : ℕ := 3

def border_tiles : ℕ :=
  2 * (room_length / border_tile_size + room_width / border_tile_size) - 4

def inner_area : ℕ :=
  (room_length - 2 * border_tile_size) * (room_width - 2 * border_tile_size)

def inner_tiles : ℕ :=
  (inner_area + inner_tile_size^2 - 1) / inner_tile_size^2

theorem total_tiles_count :
  border_tiles + inner_tiles = 70 := by sorry

end NUMINAMATH_CALUDE_total_tiles_count_l708_70889


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l708_70891

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a_3 = 8 and a_6 = 5, a_9 = 2 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a3 : a 3 = 8) 
  (h_a6 : a 6 = 5) : 
  a 9 = 2 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l708_70891


namespace NUMINAMATH_CALUDE_eight_digit_divisible_by_nine_l708_70833

theorem eight_digit_divisible_by_nine (n : Nat) : 
  n ≤ 9 →
  (854 * 10^7 + n * 10^6 + 5 * 10^5 + 2 * 10^4 + 6 * 10^3 + 8 * 10^2 + 6 * 10 + 8) % 9 = 0 →
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_divisible_by_nine_l708_70833


namespace NUMINAMATH_CALUDE_sum_after_transformation_l708_70824

theorem sum_after_transformation (a b S : ℝ) (h : a + b = S) :
  2 * ((a + 3) + (b + 3)) = 2 * S + 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_transformation_l708_70824


namespace NUMINAMATH_CALUDE_smallest_list_size_l708_70828

theorem smallest_list_size (n a b : ℕ) (h1 : n = a + b) (h2 : 89 * n = 73 * a + 111 * b) : n ≥ 19 := by
  sorry

end NUMINAMATH_CALUDE_smallest_list_size_l708_70828


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l708_70861

/-- The side length of an equilateral triangle with perimeter 2 meters is 2/3 meters. -/
theorem equilateral_triangle_side_length : 
  ∀ (side_length : ℝ), 
    (side_length > 0) →
    (3 * side_length = 2) →
    side_length = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l708_70861


namespace NUMINAMATH_CALUDE_height_difference_petronas_empire_l708_70807

/-- The height difference between two buildings -/
def height_difference (h1 h2 : ℝ) : ℝ := |h1 - h2|

/-- The Empire State Building is 443 m tall -/
def empire_state_height : ℝ := 443

/-- The Petronas Towers is 452 m tall -/
def petronas_towers_height : ℝ := 452

/-- Theorem: The height difference between the Petronas Towers and the Empire State Building is 9 meters -/
theorem height_difference_petronas_empire : 
  height_difference petronas_towers_height empire_state_height = 9 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_petronas_empire_l708_70807


namespace NUMINAMATH_CALUDE_no_refuel_needed_l708_70882

-- Define the parameters
def total_distance : ℕ := 156
def distance_driven : ℕ := 48
def gas_added : ℕ := 12
def fuel_consumption : ℕ := 24

-- Define the remaining distance
def remaining_distance : ℕ := total_distance - distance_driven

-- Define the range with added gas
def range_with_added_gas : ℕ := gas_added * fuel_consumption

-- Theorem statement
theorem no_refuel_needed : range_with_added_gas ≥ remaining_distance := by
  sorry

end NUMINAMATH_CALUDE_no_refuel_needed_l708_70882


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l708_70840

theorem complex_expression_simplification :
  let a : ℂ := 3 - I
  let b : ℂ := 2 + I
  let c : ℂ := -1 + 2 * I
  3 * a + 4 * b - 2 * c = 19 := by sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l708_70840


namespace NUMINAMATH_CALUDE_triangle_properties_l708_70859

/-- Given a triangle ABC with the following properties:
    - a, b, c are sides opposite to angles A, B, C respectively
    - a = 2√3
    - A = π/3
    - Area S = 2√3
    - sin(C-B) = sin(2B) - sin(A)
    Prove the properties of sides b, c and the shape of the triangle -/
theorem triangle_properties (a b c A B C S : Real) : 
  a = 2 * Real.sqrt 3 →
  A = π / 3 →
  S = 2 * Real.sqrt 3 →
  S = 1/2 * b * c * Real.sin A →
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A →
  Real.sin (C - B) = Real.sin (2*B) - Real.sin A →
  ((b = 2 ∧ c = 4) ∨ (b = 4 ∧ c = 2)) ∧
  (B = π / 2 ∨ C = B) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l708_70859


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l708_70876

/-- The eccentricity of an ellipse with specific properties -/
theorem ellipse_eccentricity (a b : ℝ) (h_ab : a > b) (h_b_pos : b > 0) : 
  let e := Real.sqrt (1 - b^2 / a^2)
  let c := e * a
  (∃ (P : ℝ × ℝ), 
    P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧ 
    P.2 = 0 ∧ 
    Real.sqrt ((P.1 + c)^2 + P.2^2) = 3/4 * (a + c)) →
  e = 1/4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l708_70876


namespace NUMINAMATH_CALUDE_brownie_count_l708_70809

def tray_length : ℕ := 24
def tray_width : ℕ := 15
def brownie_side : ℕ := 3

theorem brownie_count : 
  (tray_length * tray_width) / (brownie_side * brownie_side) = 40 := by
  sorry

end NUMINAMATH_CALUDE_brownie_count_l708_70809


namespace NUMINAMATH_CALUDE_only_zero_solution_for_diophantine_equation_l708_70801

theorem only_zero_solution_for_diophantine_equation :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_only_zero_solution_for_diophantine_equation_l708_70801


namespace NUMINAMATH_CALUDE_dice_probability_l708_70818

def num_dice : ℕ := 6
def sides_per_die : ℕ := 18
def one_digit_numbers : ℕ := 9
def two_digit_numbers : ℕ := 9

theorem dice_probability :
  let p_one_digit : ℚ := one_digit_numbers / sides_per_die
  let p_two_digit : ℚ := two_digit_numbers / sides_per_die
  let choose_three : ℕ := Nat.choose num_dice 3
  choose_three * (p_one_digit ^ 3 * p_two_digit ^ 3) = 5 / 16 := by
sorry

end NUMINAMATH_CALUDE_dice_probability_l708_70818


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l708_70867

def complex_equation (z : ℂ) : Prop :=
  z * ((1 + Complex.I) ^ 2) / 2 = 1 + 2 * Complex.I

theorem imaginary_part_of_z (z : ℂ) (h : complex_equation z) :
  z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l708_70867


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l708_70875

theorem no_prime_roots_for_quadratic : ¬∃ k : ℤ, ∃ p q : ℕ, 
  Prime p ∧ Prime q ∧ p ≠ q ∧ 
  (p : ℤ) + q = 90 ∧ 
  (p : ℤ) * q = k ∧
  ∀ x : ℤ, x^2 - 90*x + k = 0 ↔ x = p ∨ x = q :=
by sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l708_70875


namespace NUMINAMATH_CALUDE_problem_solution_l708_70898

/-- The set A as defined in the problem -/
def A : Set ℝ := {x | 12 - 5*x - 2*x^2 > 0}

/-- The set B as defined in the problem -/
def B (a b : ℝ) : Set ℝ := {x | x^2 - a*x + b ≤ 0}

/-- The theorem statement -/
theorem problem_solution :
  ∃ (a b : ℝ),
    (A ∩ B a b = ∅) ∧
    (A ∪ B a b = Set.Ioo (-4) 8) ∧
    (a = 19/2) ∧
    (b = 12) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l708_70898


namespace NUMINAMATH_CALUDE_intersection_points_max_distance_values_l708_70826

-- Define the line l
def line_l (a t : ℝ) : ℝ × ℝ := (a + 2*t, 1 - t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Part 1: Intersection points
theorem intersection_points :
  ∃ (t₁ t₂ : ℝ),
    let (x₁, y₁) := line_l (-2) t₁
    let (x₂, y₂) := line_l (-2) t₂
    curve_C x₁ y₁ ∧ curve_C x₂ y₂ ∧
    x₁ = -4*Real.sqrt 5/5 ∧ y₁ = 2*Real.sqrt 5/5 ∧
    x₂ = 4*Real.sqrt 5/5 ∧ y₂ = -2*Real.sqrt 5/5 :=
sorry

-- Part 2: Values of a
theorem max_distance_values :
  ∀ (a : ℝ),
    (∀ (x y : ℝ), curve_C x y →
      (|x + 2*y - 2 - a| / Real.sqrt 5 ≤ 2 * Real.sqrt 5)) ∧
    (∃ (x y : ℝ), curve_C x y ∧
      |x + 2*y - 2 - a| / Real.sqrt 5 = 2 * Real.sqrt 5) →
    (a = 8 - 2*Real.sqrt 5 ∨ a = 2*Real.sqrt 5 - 12) :=
sorry

end NUMINAMATH_CALUDE_intersection_points_max_distance_values_l708_70826


namespace NUMINAMATH_CALUDE_right_triangle_third_side_square_l708_70814

theorem right_triangle_third_side_square (a b c : ℝ) : 
  (a = 3 ∧ b = 4 ∧ a^2 + b^2 = c^2) ∨ (a = 3 ∧ c = 4 ∧ a^2 + b^2 = c^2) →
  c^2 = 25 ∨ b^2 = 7 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_square_l708_70814


namespace NUMINAMATH_CALUDE_probability_range_for_event_A_l708_70843

theorem probability_range_for_event_A (p : ℝ) : 
  (0 ≤ p ∧ p < 1) →
  (4 * p * (1 - p)^3 ≤ 6 * p^2 * (1 - p)^2) →
  0.4 ≤ p ∧ p < 1 :=
sorry

end NUMINAMATH_CALUDE_probability_range_for_event_A_l708_70843


namespace NUMINAMATH_CALUDE_mean_problem_l708_70816

theorem mean_problem (x : ℝ) :
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (128 + 255 + 511 + 1023 + x) / 5 = 398.2 :=
by
  sorry

end NUMINAMATH_CALUDE_mean_problem_l708_70816


namespace NUMINAMATH_CALUDE_smallest_undefined_inverse_seven_undefined_inverse_smallest_a_is_seven_l708_70823

theorem smallest_undefined_inverse (a : ℕ) : a > 0 ∧ 
  ¬ (∃ x : ℕ, x * a ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ y : ℕ, y * a ≡ 1 [MOD 77]) →
  a ≥ 7 :=
by sorry

theorem seven_undefined_inverse : 
  ¬ (∃ x : ℕ, x * 7 ≡ 1 [MOD 70]) ∧ 
  ¬ (∃ y : ℕ, y * 7 ≡ 1 [MOD 77]) :=
by sorry

theorem smallest_a_is_seven : 
  ∃ a : ℕ, a > 0 ∧
  ¬ (∃ x : ℕ, x * a ≡ 1 [MOD 70]) ∧
  ¬ (∃ y : ℕ, y * a ≡ 1 [MOD 77]) ∧
  ∀ b : ℕ, b > 0 ∧ 
    ¬ (∃ x : ℕ, x * b ≡ 1 [MOD 70]) ∧ 
    ¬ (∃ y : ℕ, y * b ≡ 1 [MOD 77]) →
    b ≥ a :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_inverse_seven_undefined_inverse_smallest_a_is_seven_l708_70823


namespace NUMINAMATH_CALUDE_original_expenditure_is_420_l708_70810

/-- Calculates the original expenditure of a student mess given the following conditions:
  * There were initially 35 students
  * 7 new students were admitted
  * The total expenses increased by 42 rupees per day
  * The average expenditure per head decreased by 1 rupee
-/
def calculate_original_expenditure (initial_students : ℕ) (new_students : ℕ) 
  (expense_increase : ℕ) (average_decrease : ℕ) : ℕ :=
  let total_students := initial_students + new_students
  let x := (expense_increase + total_students * average_decrease) / (total_students - initial_students)
  initial_students * x

/-- Theorem stating that under the given conditions, the original expenditure was 420 rupees per day -/
theorem original_expenditure_is_420 :
  calculate_original_expenditure 35 7 42 1 = 420 := by
  sorry

end NUMINAMATH_CALUDE_original_expenditure_is_420_l708_70810


namespace NUMINAMATH_CALUDE_sin_150_minus_sin_30_equals_zero_l708_70878

theorem sin_150_minus_sin_30_equals_zero :
  Real.sin (150 * π / 180) - Real.sin (30 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_minus_sin_30_equals_zero_l708_70878


namespace NUMINAMATH_CALUDE_gas_station_candy_boxes_l708_70897

theorem gas_station_candy_boxes : 
  let chocolate_boxes : ℕ := 2
  let sugar_boxes : ℕ := 5
  let gum_boxes : ℕ := 2
  chocolate_boxes + sugar_boxes + gum_boxes = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_gas_station_candy_boxes_l708_70897


namespace NUMINAMATH_CALUDE_eunji_class_size_l708_70815

/-- The number of students who can play instrument (a) -/
def students_a : ℕ := 24

/-- The number of students who can play instrument (b) -/
def students_b : ℕ := 17

/-- The number of students who can play both instruments -/
def students_both : ℕ := 8

/-- The total number of students in Eunji's class -/
def total_students : ℕ := students_a + students_b - students_both

theorem eunji_class_size :
  total_students = 33 ∧
  students_a = 24 ∧
  students_b = 17 ∧
  students_both = 8 ∧
  total_students = students_a + students_b - students_both :=
by sorry

end NUMINAMATH_CALUDE_eunji_class_size_l708_70815


namespace NUMINAMATH_CALUDE_firecrackers_confiscated_l708_70812

theorem firecrackers_confiscated (initial : ℕ) (remaining : ℕ) : 
  initial = 48 →
  remaining < initial →
  (1 : ℚ) / 6 * remaining = remaining - (2 * 15) →
  initial - remaining = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_firecrackers_confiscated_l708_70812


namespace NUMINAMATH_CALUDE_yuko_wins_l708_70871

/-- The minimum value of Yuko's last die to be ahead of Yuri -/
def min_value_to_win (yuri_dice : Fin 3 → Nat) (yuko_dice : Fin 2 → Nat) : Nat :=
  (yuri_dice 0 + yuri_dice 1 + yuri_dice 2) - (yuko_dice 0 + yuko_dice 1) + 1

theorem yuko_wins (yuri_dice : Fin 3 → Nat) (yuko_dice : Fin 2 → Nat) :
  yuri_dice 0 = 2 → yuri_dice 1 = 4 → yuri_dice 2 = 5 →
  yuko_dice 0 = 1 → yuko_dice 1 = 5 →
  min_value_to_win yuri_dice yuko_dice = 6 := by
  sorry

#eval min_value_to_win (![2, 4, 5]) (![1, 5])

end NUMINAMATH_CALUDE_yuko_wins_l708_70871


namespace NUMINAMATH_CALUDE_constant_term_of_expansion_l708_70835

/-- The constant term in the expansion of (3x + 2/x)^8 is 90720 -/
theorem constant_term_of_expansion (x : ℝ) (x_ne_zero : x ≠ 0) : 
  (Finset.range 9).sum (λ k => Nat.choose 8 k * (3^(8-k) * 2^k * x^(8-2*k))) = 90720 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_of_expansion_l708_70835


namespace NUMINAMATH_CALUDE_exam_mean_score_l708_70820

theorem exam_mean_score (morning_mean : ℝ) (afternoon_mean : ℝ) (class_ratio : ℚ) : 
  morning_mean = 82 →
  afternoon_mean = 68 →
  class_ratio = 4/5 →
  let total_students := class_ratio + 1
  let total_score := morning_mean * class_ratio + afternoon_mean
  total_score / total_students = 74 := by
sorry

end NUMINAMATH_CALUDE_exam_mean_score_l708_70820


namespace NUMINAMATH_CALUDE_polynomial_factorization_l708_70880

theorem polynomial_factorization (a x : ℝ) : a * x^2 - a * x - 2 * a = a * (x - 2) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l708_70880


namespace NUMINAMATH_CALUDE_certain_number_minus_fifteen_l708_70885

theorem certain_number_minus_fifteen (x : ℝ) : x / 10 = 6 → x - 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_minus_fifteen_l708_70885


namespace NUMINAMATH_CALUDE_incircle_tangent_smaller_triangle_perimeter_l708_70873

/-- Given a triangle with sides a, b, c and an inscribed incircle, 
    the perimeter of the smaller triangle formed by a tangent to the incircle 
    intersecting the two longer sides is equal to 2 * (semiperimeter - shortest_side) -/
theorem incircle_tangent_smaller_triangle_perimeter 
  (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sides : a = 6 ∧ b = 10 ∧ c = 12) : 
  let p := (a + b + c) / 2
  2 * (p - min a (min b c)) = 28 := by sorry

end NUMINAMATH_CALUDE_incircle_tangent_smaller_triangle_perimeter_l708_70873


namespace NUMINAMATH_CALUDE_scatter_plot_placement_l708_70808

/-- Represents a variable in a scatter plot --/
inductive Variable
| Explanatory
| Forecast

/-- Represents an axis in a scatter plot --/
inductive Axis
| X
| Y

/-- Defines the relationship between variables and their roles in regression analysis --/
def is_independent (v : Variable) : Prop :=
  match v with
  | Variable.Explanatory => true
  | Variable.Forecast => false

/-- Defines the correct placement of variables on axes in a scatter plot --/
def correct_placement (v : Variable) (a : Axis) : Prop :=
  (v = Variable.Explanatory ∧ a = Axis.X) ∨ (v = Variable.Forecast ∧ a = Axis.Y)

/-- Theorem stating the correct placement of variables in a scatter plot for regression analysis --/
theorem scatter_plot_placement :
  ∀ (v : Variable) (a : Axis),
    is_independent v ↔ correct_placement v Axis.X :=
by sorry

end NUMINAMATH_CALUDE_scatter_plot_placement_l708_70808


namespace NUMINAMATH_CALUDE_solution_set_implies_a_range_l708_70832

theorem solution_set_implies_a_range :
  (∀ x : ℝ, |x - 1| + |x - 2| > a^2 + a + 1) → a ∈ Set.Ioo (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_range_l708_70832


namespace NUMINAMATH_CALUDE_lottery_probability_l708_70879

/-- The probability of winning in a lottery with 10 balls labeled 1 to 10, 
    where winning occurs if the selected number is not less than 6. -/
theorem lottery_probability : 
  let total_balls : ℕ := 10
  let winning_balls : ℕ := 5
  let probability := winning_balls / total_balls
  probability = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_lottery_probability_l708_70879


namespace NUMINAMATH_CALUDE_square_formation_proof_l708_70802

theorem square_formation_proof :
  ∃ (s a b : ℝ),
    s > 0 ∧ a > 0 ∧ b > 0 ∧
    2 * s^2 + 10 * 24 + a * b = 24^2 ∧
    (s = 12 ∧ a = 2 ∧ b = 24) ∨
    (s = 12 ∧ a = 19 ∧ b = 24) ∨
    (s = 14 ∧ a = 34 ∧ b = 10) ∨
    (s = 10 ∧ a = 34 ∧ b = 44) ∨
    (s = 17 ∧ a = 14 ∧ b = 24) ∨
    (s = 19 ∧ a = 14 ∧ b = 17) ∨
    (s = 10 ∧ a = 24 ∧ b = 38) :=
by sorry

end NUMINAMATH_CALUDE_square_formation_proof_l708_70802


namespace NUMINAMATH_CALUDE_number_of_boys_l708_70892

/-- Given a school with girls and boys, prove the number of boys. -/
theorem number_of_boys (girls boys : ℕ) 
  (h1 : girls = 635)
  (h2 : boys = girls + 510) : 
  boys = 1145 := by
  sorry

end NUMINAMATH_CALUDE_number_of_boys_l708_70892


namespace NUMINAMATH_CALUDE_rationalize_sqrt_three_eighths_l708_70842

theorem rationalize_sqrt_three_eighths : 
  Real.sqrt (3 / 8) = Real.sqrt 6 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_sqrt_three_eighths_l708_70842


namespace NUMINAMATH_CALUDE_min_sum_quadratic_coeff_l708_70857

theorem min_sum_quadratic_coeff (a b c : ℕ+) 
  (root_condition : ∃ x₁ x₂ : ℝ, (a:ℝ) * x₁^2 + (b:ℝ) * x₁ + (c:ℝ) = 0 ∧ 
                                (a:ℝ) * x₂^2 + (b:ℝ) * x₂ + (c:ℝ) = 0 ∧
                                x₁ ≠ x₂ ∧ 
                                abs x₁ < (1:ℝ)/3 ∧ 
                                abs x₂ < (1:ℝ)/3) : 
  (a:ℕ) + b + c ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_quadratic_coeff_l708_70857


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l708_70865

theorem complex_fraction_equality : 
  let i : ℂ := Complex.I
  (7 + i) / (3 + 4*i) = 1 - i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l708_70865


namespace NUMINAMATH_CALUDE_number_of_shortest_paths_is_54_l708_70841

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ

/-- Represents the grid configuration -/
structure Grid where
  squareSize : ℕ  -- Side length of each square in km
  refuelDistance : ℕ  -- Distance the car can travel before refueling in km

/-- Calculates the number of shortest paths between two points on the grid -/
def numberOfShortestPaths (g : Grid) (start finish : GridPoint) : ℕ :=
  sorry

/-- The specific grid configuration for the problem -/
def problemGrid : Grid :=
  { squareSize := 10
  , refuelDistance := 30 }

/-- The start point A -/
def pointA : GridPoint :=
  { x := 0, y := 0 }

/-- The end point B -/
def pointB : GridPoint :=
  { x := 6, y := 6 }  -- Assuming a 6x6 grid based on the problem description

theorem number_of_shortest_paths_is_54 :
  numberOfShortestPaths problemGrid pointA pointB = 54 :=
by sorry

end NUMINAMATH_CALUDE_number_of_shortest_paths_is_54_l708_70841


namespace NUMINAMATH_CALUDE_bees_flew_in_l708_70825

theorem bees_flew_in (initial_bees final_bees : ℕ) (h1 : initial_bees = 16) (h2 : final_bees = 23) :
  final_bees - initial_bees = 7 := by
sorry

end NUMINAMATH_CALUDE_bees_flew_in_l708_70825


namespace NUMINAMATH_CALUDE_F_difference_l708_70872

/-- Represents the infinite repeating decimal 0.726726726... -/
def F : ℚ := 726 / 999

/-- The fraction representation of F in lowest terms -/
def F_reduced : ℚ := 242 / 333

theorem F_difference : (F_reduced.den : ℤ) - (F_reduced.num : ℤ) = 91 := by sorry

end NUMINAMATH_CALUDE_F_difference_l708_70872


namespace NUMINAMATH_CALUDE_bug_probability_l708_70850

/-- Probability of the bug being at vertex A after n steps -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - P n)

/-- The probability of the bug being at vertex A after 7 steps is 182/729 -/
theorem bug_probability : P 7 = 182 / 729 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_l708_70850


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l708_70863

theorem arithmetic_geometric_mean_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l708_70863


namespace NUMINAMATH_CALUDE_nancy_antacid_consumption_l708_70813

/-- Calculates the number of antacids Nancy takes per month based on her eating habits. -/
def antacids_per_month (indian_antacids : ℕ) (mexican_antacids : ℕ) (other_antacids : ℕ)
  (indian_freq : ℕ) (mexican_freq : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) : ℕ :=
  let other_days := days_per_week - indian_freq - mexican_freq
  let weekly_antacids := indian_antacids * indian_freq + mexican_antacids * mexican_freq + other_antacids * other_days
  weekly_antacids * weeks_per_month

/-- Theorem stating that Nancy takes 60 antacids per month given her eating habits. -/
theorem nancy_antacid_consumption :
  antacids_per_month 3 2 1 3 2 7 4 = 60 := by
  sorry

#eval antacids_per_month 3 2 1 3 2 7 4

end NUMINAMATH_CALUDE_nancy_antacid_consumption_l708_70813


namespace NUMINAMATH_CALUDE_sqrt_equation_proof_l708_70868

theorem sqrt_equation_proof (y : ℝ) : 
  (Real.sqrt 1.21 / Real.sqrt y) + (Real.sqrt 1.00 / Real.sqrt 0.49) = 2.650793650793651 → 
  y = 0.81 := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_proof_l708_70868


namespace NUMINAMATH_CALUDE_correct_train_process_l708_70884

-- Define the actions as an inductive type
inductive TrainAction
  | BuyTicket
  | WaitForTrain
  | CheckTicket
  | BoardTrain

-- Define a type for a sequence of actions
def ActionSequence := List TrainAction

-- Define the correct sequence
def correctSequence : ActionSequence :=
  [TrainAction.BuyTicket, TrainAction.WaitForTrain, TrainAction.CheckTicket, TrainAction.BoardTrain]

-- Define a predicate for a valid train-taking process
def isValidProcess (sequence : ActionSequence) : Prop :=
  sequence = correctSequence

-- Theorem statement
theorem correct_train_process :
  isValidProcess correctSequence :=
sorry

end NUMINAMATH_CALUDE_correct_train_process_l708_70884


namespace NUMINAMATH_CALUDE_base_conversion_count_l708_70829

theorem base_conversion_count : 
  ∃! n : ℕ, n = (Finset.filter (fun c : ℕ => c ≥ 2 ∧ c^2 ≤ 256 ∧ 256 < c^3) (Finset.range 257)).card ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_count_l708_70829


namespace NUMINAMATH_CALUDE_unique_solution_cubic_equation_l708_70854

theorem unique_solution_cubic_equation :
  ∃! x : ℝ, x ≠ -1 ∧ (x^3 - x^2) / (x^2 + 2*x + 1) + 2*x = -4 :=
by
  use 4/3
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_equation_l708_70854


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l708_70836

theorem smallest_number_divisible (n : ℕ) : n ≥ 1015 ∧ 
  (∀ m : ℕ, m < 1015 → ¬(12 ∣ (m - 7) ∧ 16 ∣ (m - 7) ∧ 18 ∣ (m - 7) ∧ 21 ∣ (m - 7) ∧ 28 ∣ (m - 7))) →
  (12 ∣ (n - 7) ∧ 16 ∣ (n - 7) ∧ 18 ∣ (n - 7) ∧ 21 ∣ (n - 7) ∧ 28 ∣ (n - 7)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l708_70836


namespace NUMINAMATH_CALUDE_fraction_simplification_l708_70895

theorem fraction_simplification : (3 : ℚ) / (1 - 2 / 5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l708_70895


namespace NUMINAMATH_CALUDE_intersection_A_B_l708_70804

-- Define the sets A and B
def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | (x - 1) * (x - 3) < 0}

-- State the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l708_70804


namespace NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l708_70845

/-- The problem of Jack, Christina, and Lindy meeting --/
theorem jack_christina_lindy_meeting 
  (initial_distance : ℝ) 
  (christina_speed : ℝ) 
  (lindy_speed : ℝ) 
  (lindy_total_distance : ℝ) 
  (h1 : initial_distance = 240)
  (h2 : christina_speed = 3)
  (h3 : lindy_speed = 9)
  (h4 : lindy_total_distance = 270) :
  ∃ (jack_speed : ℝ), 
    jack_speed = 5 ∧ 
    (lindy_total_distance / lindy_speed) * jack_speed + 
    (lindy_total_distance / lindy_speed) * christina_speed = 
    initial_distance := by
  sorry


end NUMINAMATH_CALUDE_jack_christina_lindy_meeting_l708_70845


namespace NUMINAMATH_CALUDE_train_y_completion_time_l708_70839

/-- Represents the time it takes for Train Y to complete the trip -/
def train_y_time (route_length : ℝ) (train_x_time : ℝ) (train_x_distance : ℝ) : ℝ :=
  4

/-- Theorem stating that Train Y takes 4 hours to complete the trip under the given conditions -/
theorem train_y_completion_time 
  (route_length : ℝ) 
  (train_x_time : ℝ) 
  (train_x_distance : ℝ)
  (h1 : route_length = 180)
  (h2 : train_x_time = 5)
  (h3 : train_x_distance = 80) :
  train_y_time route_length train_x_time train_x_distance = 4 := by
  sorry

#check train_y_completion_time

end NUMINAMATH_CALUDE_train_y_completion_time_l708_70839


namespace NUMINAMATH_CALUDE_farmer_brown_sheep_l708_70899

/-- The number of chickens Farmer Brown fed -/
def num_chickens : ℕ := 7

/-- The total number of legs among all animals Farmer Brown fed -/
def total_legs : ℕ := 34

/-- The number of legs a chicken has -/
def chicken_legs : ℕ := 2

/-- The number of legs a sheep has -/
def sheep_legs : ℕ := 4

/-- The number of sheep Farmer Brown fed -/
def num_sheep : ℕ := (total_legs - num_chickens * chicken_legs) / sheep_legs

theorem farmer_brown_sheep : num_sheep = 5 := by
  sorry

end NUMINAMATH_CALUDE_farmer_brown_sheep_l708_70899


namespace NUMINAMATH_CALUDE_f_2011_eq_sin_l708_70869

noncomputable def f : ℕ → (ℝ → ℝ)
| 0 => Real.cos
| (n + 1) => deriv (f n)

theorem f_2011_eq_sin : f 2011 = Real.sin := by sorry

end NUMINAMATH_CALUDE_f_2011_eq_sin_l708_70869


namespace NUMINAMATH_CALUDE_jasons_football_games_l708_70805

theorem jasons_football_games (this_month next_month total : ℕ) 
  (h1 : this_month = 11)
  (h2 : next_month = 16)
  (h3 : total = 44) :
  ∃ last_month : ℕ, last_month + this_month + next_month = total ∧ last_month = 17 := by
  sorry

end NUMINAMATH_CALUDE_jasons_football_games_l708_70805


namespace NUMINAMATH_CALUDE_pizza_combinations_l708_70837

theorem pizza_combinations (n : ℕ) (h : n = 8) : 
  n + n.choose 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_l708_70837


namespace NUMINAMATH_CALUDE_max_sum_after_adding_pyramid_l708_70806

/-- A rectangular prism -/
structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- Properties of the resulting solid after adding a square pyramid -/
structure ResultingSolid :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)

/-- Function to calculate the resulting solid properties -/
def add_pyramid (prism : RectangularPrism) : ResultingSolid :=
  { faces := prism.faces - 1 + 4,
    edges := prism.edges + 4,
    vertices := prism.vertices + 1 }

/-- Theorem stating the maximum sum of faces, edges, and vertices -/
theorem max_sum_after_adding_pyramid (prism : RectangularPrism)
  (h1 : prism.faces = 6)
  (h2 : prism.edges = 12)
  (h3 : prism.vertices = 8) :
  let resulting := add_pyramid prism
  resulting.faces + resulting.edges + resulting.vertices = 34 :=
sorry

end NUMINAMATH_CALUDE_max_sum_after_adding_pyramid_l708_70806


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l708_70860

/-- A number is composite if it has more than two distinct positive divisors. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

/-- A natural number can be expressed as the sum of two composite numbers. -/
def IsSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, IsComposite a ∧ IsComposite b ∧ a + b = n

/-- 11 is the largest natural number that cannot be expressed as the sum of two composite numbers. -/
theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → IsSumOfTwoComposites n) ∧
  ¬IsSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l708_70860


namespace NUMINAMATH_CALUDE_fraction_multiplication_l708_70803

theorem fraction_multiplication : (2 : ℚ) / 3 * 3 / 5 * 4 / 7 * 5 / 8 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l708_70803


namespace NUMINAMATH_CALUDE_curve_C_properties_l708_70866

-- Define the curve C in polar coordinates
def C (ρ θ : ℝ) : Prop :=
  ρ^2 - 4*ρ*(Real.cos θ) - 6*ρ*(Real.sin θ) + 12 = 0

-- Define the rectangular coordinates of a point on C
def point_on_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

-- Define the distance |PM| + |PN|
def PM_PN (x y : ℝ) : ℝ :=
  y + (x + 1)

-- Theorem statement
theorem curve_C_properties :
  (∀ ρ θ : ℝ, C ρ θ ↔ point_on_C (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∃ max : ℝ, max = 6 + Real.sqrt 2 ∧
    ∀ x y : ℝ, point_on_C x y → PM_PN x y ≤ max) :=
sorry

end NUMINAMATH_CALUDE_curve_C_properties_l708_70866


namespace NUMINAMATH_CALUDE_number_divided_by_004_equals_25_l708_70856

theorem number_divided_by_004_equals_25 : ∃ x : ℝ, x / 0.04 = 25 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_004_equals_25_l708_70856


namespace NUMINAMATH_CALUDE_map_scale_l708_70858

/-- Given a map where 15 cm represents 90 km, prove that 20 cm represents 120 km -/
theorem map_scale (map_cm : ℝ) (map_km : ℝ) (actual_cm : ℝ)
  (h1 : map_cm = 15)
  (h2 : map_km = 90)
  (h3 : actual_cm = 20) :
  (actual_cm / map_cm) * map_km = 120 := by
  sorry

end NUMINAMATH_CALUDE_map_scale_l708_70858


namespace NUMINAMATH_CALUDE_quadratic_decreasing_interval_l708_70851

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_decreasing_interval (b c : ℝ) :
  (f b c 1 = 0) → (f b c 3 = 0) →
  ∃ (x : ℝ), ∀ (y : ℝ), y < x → (∀ (z : ℝ), y < z → f b c y > f b c z) ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_interval_l708_70851


namespace NUMINAMATH_CALUDE_crayon_count_is_44_l708_70887

/-- The number of crayons in the drawer after a series of additions and removals. -/
def final_crayon_count (initial : ℝ) (benny_add : ℝ) (lucy_remove : ℝ) (sam_add : ℝ) : ℝ :=
  initial + benny_add - lucy_remove + sam_add

/-- Theorem stating that the final number of crayons is 44 given the initial count and actions. -/
theorem crayon_count_is_44 :
  final_crayon_count 25 15.5 8.75 12.25 = 44 := by
  sorry

end NUMINAMATH_CALUDE_crayon_count_is_44_l708_70887


namespace NUMINAMATH_CALUDE_rectangle_length_is_16_l708_70817

/-- Represents a rectangle with given perimeter and length-width relationship -/
structure Rectangle where
  perimeter : ℝ
  width : ℝ
  length : ℝ
  perimeter_eq : perimeter = 2 * (length + width)
  length_eq : length = 2 * width

/-- Theorem: For a rectangle with perimeter 48 and length twice the width, the length is 16 -/
theorem rectangle_length_is_16 (rect : Rectangle) 
  (h_perimeter : rect.perimeter = 48) : rect.length = 16 := by
  sorry

#check rectangle_length_is_16

end NUMINAMATH_CALUDE_rectangle_length_is_16_l708_70817


namespace NUMINAMATH_CALUDE_exterior_angle_bisector_theorem_l708_70821

/-- Given a triangle ABC with interior angles α, β, γ, and a triangle formed by 
    the bisectors of its exterior angles with angles α₁, β₁, γ₁, prove that 
    α = 180° - 2α₁, β = 180° - 2β₁, and γ = 180° - 2γ₁ --/
theorem exterior_angle_bisector_theorem 
  (α β γ α₁ β₁ γ₁ : Real) 
  (h_triangle : α + β + γ = 180) 
  (h_exterior_bisector : α₁ + β₁ + γ₁ = 180) : 
  α = 180 - 2*α₁ ∧ β = 180 - 2*β₁ ∧ γ = 180 - 2*γ₁ := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_bisector_theorem_l708_70821


namespace NUMINAMATH_CALUDE_mangoes_purchased_is_nine_l708_70800

/-- The amount of mangoes purchased, given the conditions of the problem -/
def mangoes_purchased (apple_kg : ℕ) (apple_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  (total_paid - apple_kg * apple_rate) / mango_rate

/-- Theorem stating that the amount of mangoes purchased is 9 kg -/
theorem mangoes_purchased_is_nine :
  mangoes_purchased 8 70 45 965 = 9 := by sorry

end NUMINAMATH_CALUDE_mangoes_purchased_is_nine_l708_70800


namespace NUMINAMATH_CALUDE_angle_properties_l708_70862

/-- Given a point P on the unit circle, determine the quadrant and smallest positive angle -/
theorem angle_properties (α : Real) : 
  (∃ P : Real × Real, P.1 = Real.sin (5 * Real.pi / 6) ∧ P.2 = Real.cos (5 * Real.pi / 6) ∧ 
   P.1 = Real.sin α ∧ P.2 = Real.cos α) →
  (α > 3 * Real.pi / 2 ∧ α < 2 * Real.pi) ∧
  (∃ β : Real, β = 5 * Real.pi / 3 ∧ 
   Real.sin β = Real.sin α ∧ Real.cos β = Real.cos α ∧
   ∀ γ : Real, 0 < γ ∧ γ < β → 
   Real.sin γ ≠ Real.sin α ∨ Real.cos γ ≠ Real.cos α) :=
by sorry

end NUMINAMATH_CALUDE_angle_properties_l708_70862


namespace NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l708_70877

theorem quadratic_roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 5*x₁ - 6 = 0 → 
  x₂^2 - 5*x₂ - 6 = 0 → 
  x₁ ≠ x₂ → 
  (1/x₁) + (1/x₂) = -5/6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_reciprocal_sum_l708_70877


namespace NUMINAMATH_CALUDE_octagon_area_l708_70846

/-- Given a square with area 16 and a regular octagon with equal perimeter to the square,
    the area of the octagon is 8(1+√2) -/
theorem octagon_area (s : ℝ) (t : ℝ) : 
  s^2 = 16 →                        -- Square area is 16
  4*s = 8*t →                       -- Equal perimeters
  2*(1+Real.sqrt 2)*t^2 = 8*(1+Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_octagon_area_l708_70846


namespace NUMINAMATH_CALUDE_jason_total_money_l708_70831

/-- Represents the value of different coin types in dollars -/
def coin_value : Fin 3 → ℚ
  | 0 => 0.25  -- Quarter
  | 1 => 0.10  -- Dime
  | 2 => 0.05  -- Nickel
  | _ => 0     -- Unreachable case

/-- Calculates the total value of coins given their quantities -/
def total_value (quarters dimes nickels : ℕ) : ℚ :=
  quarters * coin_value 0 + dimes * coin_value 1 + nickels * coin_value 2

/-- Jason's initial coin quantities -/
def initial_coins : Fin 3 → ℕ
  | 0 => 49  -- Quarters
  | 1 => 32  -- Dimes
  | 2 => 18  -- Nickels
  | _ => 0   -- Unreachable case

/-- Additional coins given by Jason's dad -/
def additional_coins : Fin 3 → ℕ
  | 0 => 25  -- Quarters
  | 1 => 15  -- Dimes
  | 2 => 10  -- Nickels
  | _ => 0   -- Unreachable case

/-- Theorem stating that Jason's total money is $24.60 -/
theorem jason_total_money :
  total_value (initial_coins 0 + additional_coins 0)
              (initial_coins 1 + additional_coins 1)
              (initial_coins 2 + additional_coins 2) = 24.60 := by
  sorry

end NUMINAMATH_CALUDE_jason_total_money_l708_70831


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l708_70896

def first_2015_primes : List Nat := sorry

def m : Nat := List.prod first_2015_primes

theorem no_positive_integer_solution :
  ∀ x y z : Nat, (2 * x - y - z) * (2 * y - z - x) * (2 * z - x - y) ≠ m :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l708_70896


namespace NUMINAMATH_CALUDE_sharon_coffee_cost_l708_70864

/-- Calculates the total cost of coffee pods for a vacation. -/
def coffee_cost (days : ℕ) (pods_per_day : ℕ) (pods_per_box : ℕ) (cost_per_box : ℚ) : ℚ :=
  let total_pods := days * pods_per_day
  let boxes_needed := (total_pods + pods_per_box - 1) / pods_per_box  -- Ceiling division
  boxes_needed * cost_per_box

/-- Proves that Sharon's coffee cost for her vacation is $32.00 -/
theorem sharon_coffee_cost :
  coffee_cost 40 3 30 8 = 32 :=
by
  sorry

#eval coffee_cost 40 3 30 8

end NUMINAMATH_CALUDE_sharon_coffee_cost_l708_70864


namespace NUMINAMATH_CALUDE_sequence_inequality_l708_70847

theorem sequence_inequality (a : ℕ+ → ℝ) 
  (h : ∀ (k m : ℕ+), |a (k + m) - a k - a m| ≤ 1) :
  ∀ (k m : ℕ+), |a k / k.val - a m / m.val| < 1 / k.val + 1 / m.val := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l708_70847


namespace NUMINAMATH_CALUDE_emilys_cards_l708_70883

theorem emilys_cards (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 63)
  (h2 : final_cards = 70) :
  final_cards - initial_cards = 7 := by
  sorry

end NUMINAMATH_CALUDE_emilys_cards_l708_70883


namespace NUMINAMATH_CALUDE_shooting_game_cost_l708_70855

theorem shooting_game_cost (jen_plays : ℕ) (russel_rides : ℕ) (carousel_cost : ℕ) (total_tickets : ℕ) :
  jen_plays = 2 →
  russel_rides = 3 →
  carousel_cost = 3 →
  total_tickets = 19 →
  ∃ (shooting_cost : ℕ), jen_plays * shooting_cost + russel_rides * carousel_cost = total_tickets ∧ shooting_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_shooting_game_cost_l708_70855


namespace NUMINAMATH_CALUDE_percentage_of_x_minus_y_l708_70838

theorem percentage_of_x_minus_y (x y : ℝ) (P : ℝ) :
  (P / 100) * (x - y) = (20 / 100) * (x + y) →
  y = (20 / 100) * x →
  P = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_x_minus_y_l708_70838


namespace NUMINAMATH_CALUDE_f_max_value_l708_70852

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def f (n : ℕ) : ℚ := S n / ((n + 32) * S (n + 1))

theorem f_max_value :
  (∀ n : ℕ, n > 0 → f n ≤ 1/50) ∧ (∃ n : ℕ, n > 0 ∧ f n = 1/50) :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l708_70852


namespace NUMINAMATH_CALUDE_tim_took_25_rulers_l708_70881

/-- The number of rulers Tim took from the drawer -/
def rulers_taken (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proof that Tim took 25 rulers from the drawer -/
theorem tim_took_25_rulers :
  let initial_rulers : ℕ := 46
  let remaining_rulers : ℕ := 21
  rulers_taken initial_rulers remaining_rulers = 25 := by
  sorry

end NUMINAMATH_CALUDE_tim_took_25_rulers_l708_70881


namespace NUMINAMATH_CALUDE_max_value_of_expr_l708_70811

def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

def expr (a b c : ℕ) : ℚ := 1 / (a + 2010 / (b + 1 / c))

theorem max_value_of_expr (a b c : ℕ) 
  (ha : is_nonzero_digit a) 
  (hb : is_nonzero_digit b) 
  (hc : is_nonzero_digit c) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  expr a b c ≤ 1 / 203 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_expr_l708_70811


namespace NUMINAMATH_CALUDE_soccer_points_for_win_l708_70886

theorem soccer_points_for_win (total_games : ℕ) (wins : ℕ) (losses : ℕ) (total_points : ℕ)
  (h_total_games : total_games = 20)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_total_points : total_points = 46)
  (h_games_balance : total_games = wins + losses + (total_games - wins - losses)) :
  ∃ (points_for_win : ℕ),
    points_for_win * wins + (total_games - wins - losses) = total_points ∧ 
    points_for_win = 3 := by
sorry

end NUMINAMATH_CALUDE_soccer_points_for_win_l708_70886


namespace NUMINAMATH_CALUDE_char_coeff_pair_example_char_poly_sum_example_char_poly_diff_example_l708_70849

-- Define the characteristic coefficient pair
def char_coeff_pair (a b c : ℝ) : ℝ × ℝ × ℝ := (a, b, c)

-- Define the characteristic polynomial
def char_poly (p : ℝ × ℝ × ℝ) (x : ℝ) : ℝ :=
  let (a, b, c) := p
  a * x^2 + b * x + c

theorem char_coeff_pair_example : char_coeff_pair 3 4 1 = (3, 4, 1) := by sorry

theorem char_poly_sum_example : 
  char_poly (2, 1, 2) x + char_poly (2, -1, 2) x = 4 * x^2 + 4 := by sorry

theorem char_poly_diff_example (m n : ℝ) : 
  (char_poly (1, 2, m) x - char_poly (2, n, 3) x = -x^2 + x - 1) → m * n = 2 := by sorry

end NUMINAMATH_CALUDE_char_coeff_pair_example_char_poly_sum_example_char_poly_diff_example_l708_70849
