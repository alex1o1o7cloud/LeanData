import Mathlib

namespace NUMINAMATH_CALUDE_representation_of_2019_representation_of_any_integer_l3007_300756

theorem representation_of_2019 : ∃ (a b c : ℤ), 2019 = a^2 + b^2 - c^2 := by sorry

theorem representation_of_any_integer : ∀ (n : ℤ), ∃ (a b c d : ℤ), n = a^2 + b^2 + c^2 - d^2 := by sorry

end NUMINAMATH_CALUDE_representation_of_2019_representation_of_any_integer_l3007_300756


namespace NUMINAMATH_CALUDE_equal_solution_implies_k_value_l3007_300787

theorem equal_solution_implies_k_value :
  ∀ (k : ℚ), 
  (∃ (x : ℚ), 3 * x - 6 = 0 ∧ 2 * x - 5 * k = 11) →
  (∀ (x : ℚ), 3 * x - 6 = 0 ↔ 2 * x - 5 * k = 11) →
  k = -7/5 := by
sorry

end NUMINAMATH_CALUDE_equal_solution_implies_k_value_l3007_300787


namespace NUMINAMATH_CALUDE_shaanxi_temp_difference_l3007_300730

/-- The temperature difference between two regions -/
def temperature_difference (temp1 : ℝ) (temp2 : ℝ) : ℝ :=
  temp1 - temp2

/-- The highest temperature in Shaanxi South -/
def shaanxi_south_temp : ℝ := 6

/-- The highest temperature in Shaanxi North -/
def shaanxi_north_temp : ℝ := -3

/-- Theorem: The temperature difference between Shaanxi South and Shaanxi North is 9°C -/
theorem shaanxi_temp_difference :
  temperature_difference shaanxi_south_temp shaanxi_north_temp = 9 := by
  sorry

end NUMINAMATH_CALUDE_shaanxi_temp_difference_l3007_300730


namespace NUMINAMATH_CALUDE_complex_number_equality_l3007_300768

theorem complex_number_equality (z : ℂ) : (1 - Complex.I) * z = Complex.abs (1 + Complex.I * Real.sqrt 3) → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l3007_300768


namespace NUMINAMATH_CALUDE_cassidy_poster_collection_l3007_300726

theorem cassidy_poster_collection (current_posters : ℕ) : current_posters = 22 :=
  by
  have two_years_ago : ℕ := 14
  have after_summer : ℕ := current_posters + 6
  have double_two_years_ago : after_summer = 2 * two_years_ago := by sorry
  sorry

end NUMINAMATH_CALUDE_cassidy_poster_collection_l3007_300726


namespace NUMINAMATH_CALUDE_odd_function_extension_l3007_300742

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem odd_function_extension :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x > 0, f x = x^2 - 2*x) →  -- f(x) = x^2 - 2x for x > 0
  (∀ x < 0, f x = -x^2 - 2*x) :=  -- f(x) = -x^2 - 2x for x < 0
by sorry

end NUMINAMATH_CALUDE_odd_function_extension_l3007_300742


namespace NUMINAMATH_CALUDE_weight_system_l3007_300721

/-- Represents the weight of birds in jin -/
structure BirdWeight where
  sparrow : ℝ
  swallow : ℝ

/-- The conditions of the sparrow and swallow weight problem -/
def weightProblem (w : BirdWeight) : Prop :=
  (5 * w.sparrow + 6 * w.swallow = 1) ∧
  (w.sparrow > w.swallow) ∧
  (4 * w.sparrow + 7 * w.swallow = 5 * w.sparrow + 6 * w.swallow)

/-- The system of equations representing the sparrow and swallow weight problem -/
theorem weight_system (w : BirdWeight) (h : weightProblem w) :
  (5 * w.sparrow + 6 * w.swallow = 1) ∧
  (4 * w.sparrow + 7 * w.swallow = 5 * w.sparrow + 6 * w.swallow) :=
by sorry

end NUMINAMATH_CALUDE_weight_system_l3007_300721


namespace NUMINAMATH_CALUDE_initial_red_marbles_l3007_300788

theorem initial_red_marbles (r g : ℕ) : 
  r * 3 = g * 5 → 
  (r - 18) * 4 = g + 27 → 
  r = 29 := by
sorry

end NUMINAMATH_CALUDE_initial_red_marbles_l3007_300788


namespace NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_interval_l3007_300779

theorem cubic_three_distinct_roots_in_interval 
  (p q : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    (-2 < x₁ ∧ x₁ < 4) ∧ (-2 < x₂ ∧ x₂ < 4) ∧ (-2 < x₃ ∧ x₃ < 4) ∧
    x₁^3 + p*x₁ + q = 0 ∧ x₂^3 + p*x₂ + q = 0 ∧ x₃^3 + p*x₃ + q = 0) ↔ 
  (4*p^3 + 27*q^2 < 0 ∧ 2*p + 8 < q ∧ q < -4*p - 64) :=
sorry

end NUMINAMATH_CALUDE_cubic_three_distinct_roots_in_interval_l3007_300779


namespace NUMINAMATH_CALUDE_trapezoid_height_l3007_300735

/-- The height of a trapezoid with specific properties -/
theorem trapezoid_height (a b : ℝ) (h_ab : a < b) : ∃ h : ℝ,
  h = a * b / (b - a) ∧
  ∃ (AB CD : ℝ) (angle_diagonals angle_sides : ℝ),
    AB = a ∧
    CD = b ∧
    angle_diagonals = 90 ∧
    angle_sides = 45 ∧
    h > 0 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_height_l3007_300735


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3007_300745

/-- The set of real numbers x that satisfy (x+2)/(x-4) ≥ 3 is exactly the interval (4, 7]. -/
theorem solution_set_equivalence (x : ℝ) : (x + 2) / (x - 4) ≥ 3 ↔ x ∈ Set.Ioo 4 7 ∪ {7} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3007_300745


namespace NUMINAMATH_CALUDE_two_sin_45_equals_sqrt_2_l3007_300719

theorem two_sin_45_equals_sqrt_2 (α : Real) (h : α = Real.pi / 4) : 
  2 * Real.sin α = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_two_sin_45_equals_sqrt_2_l3007_300719


namespace NUMINAMATH_CALUDE_equal_length_different_turns_l3007_300705

/-- Represents a point in the triangular grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction in the triangular grid -/
inductive Direction
  | Up
  | UpRight
  | DownRight
  | Down
  | DownLeft
  | UpLeft

/-- Represents a route in the triangular grid -/
structure Route where
  start : Point
  steps : List Direction
  leftTurns : ℕ

/-- Calculates the length of a route -/
def routeLength (r : Route) : ℕ := r.steps.length

/-- Theorem: There exist two routes in a triangular grid with different numbers of left turns but equal length -/
theorem equal_length_different_turns :
  ∃ (start finish : Point) (route1 route2 : Route),
    route1.start = start ∧
    route2.start = start ∧
    (routeLength route1 = routeLength route2) ∧
    route1.leftTurns = 4 ∧
    route2.leftTurns = 1 :=
  sorry

end NUMINAMATH_CALUDE_equal_length_different_turns_l3007_300705


namespace NUMINAMATH_CALUDE_line_length_after_erasing_l3007_300763

/-- Given a line of 1.5 meters with 15.25 centimeters erased, the resulting length is 134.75 centimeters. -/
theorem line_length_after_erasing (original_length : Real) (erased_length : Real) :
  original_length = 1.5 ∧ erased_length = 15.25 →
  original_length * 100 - erased_length = 134.75 :=
by sorry

end NUMINAMATH_CALUDE_line_length_after_erasing_l3007_300763


namespace NUMINAMATH_CALUDE_test_probabilities_l3007_300791

/-- Probability of A passing the test -/
def prob_A : ℝ := 0.8

/-- Probability of B passing the test -/
def prob_B : ℝ := 0.6

/-- Probability of C passing the test -/
def prob_C : ℝ := 0.5

/-- Probability that all three pass the test -/
def prob_all_pass : ℝ := prob_A * prob_B * prob_C

/-- Probability that at least one passes the test -/
def prob_at_least_one_pass : ℝ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

theorem test_probabilities :
  prob_all_pass = 0.24 ∧ prob_at_least_one_pass = 0.96 := by
  sorry

end NUMINAMATH_CALUDE_test_probabilities_l3007_300791


namespace NUMINAMATH_CALUDE_garden_walkway_area_l3007_300764

theorem garden_walkway_area :
  let flower_bed_width : ℕ := 4
  let flower_bed_height : ℕ := 3
  let flower_bed_rows : ℕ := 4
  let flower_bed_columns : ℕ := 3
  let walkway_width : ℕ := 2
  let pond_width : ℕ := 3
  let pond_height : ℕ := 2

  let total_width : ℕ := flower_bed_width * flower_bed_columns + walkway_width * (flower_bed_columns + 1)
  let total_height : ℕ := flower_bed_height * flower_bed_rows + walkway_width * (flower_bed_rows + 1)
  let total_area : ℕ := total_width * total_height

  let pond_area : ℕ := pond_width * pond_height
  let adjusted_area : ℕ := total_area - pond_area

  let flower_bed_area : ℕ := flower_bed_width * flower_bed_height
  let total_flower_bed_area : ℕ := flower_bed_area * flower_bed_rows * flower_bed_columns

  let walkway_area : ℕ := adjusted_area - total_flower_bed_area

  walkway_area = 290 := by sorry

end NUMINAMATH_CALUDE_garden_walkway_area_l3007_300764


namespace NUMINAMATH_CALUDE_man_rowing_downstream_speed_l3007_300732

/-- The speed of a man rowing downstream, given his speed in still water and the speed of the stream. -/
def speed_downstream (speed_still_water : ℝ) (speed_stream : ℝ) : ℝ :=
  speed_still_water + speed_stream

/-- Theorem: The speed of the man rowing downstream is 18 kmph. -/
theorem man_rowing_downstream_speed :
  let speed_still_water : ℝ := 12
  let speed_stream : ℝ := 6
  speed_downstream speed_still_water speed_stream = 18 := by
  sorry

end NUMINAMATH_CALUDE_man_rowing_downstream_speed_l3007_300732


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l3007_300798

def N : Matrix (Fin 2) (Fin 2) ℝ := !![1, -10; 0, 1]

theorem matrix_equation_proof :
  N^3 - 3 * N^2 + 2 * N = !![5, 10; 0, 5] := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l3007_300798


namespace NUMINAMATH_CALUDE_pascal_21st_number_23_row_l3007_300776

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of elements in the nth row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

theorem pascal_21st_number_23_row : 
  let row := 22
  let position := 21
  pascal_row_length row = 23 → binomial row (row + 1 - position) = 231 := by
  sorry

end NUMINAMATH_CALUDE_pascal_21st_number_23_row_l3007_300776


namespace NUMINAMATH_CALUDE_employee_discount_percentage_l3007_300743

theorem employee_discount_percentage
  (wholesale_cost : ℝ)
  (markup_percentage : ℝ)
  (employee_paid_price : ℝ)
  (h1 : wholesale_cost = 200)
  (h2 : markup_percentage = 20)
  (h3 : employee_paid_price = 180) :
  let retail_price := wholesale_cost * (1 + markup_percentage / 100)
  let discount_amount := retail_price - employee_paid_price
  let discount_percentage := (discount_amount / retail_price) * 100
  discount_percentage = 25 := by sorry

end NUMINAMATH_CALUDE_employee_discount_percentage_l3007_300743


namespace NUMINAMATH_CALUDE_least_three_digit_12_heavy_is_105_12_heavy_is_105_three_digit_least_three_digit_12_heavy_is_105_l3007_300761

/-- A number is 12-heavy if its remainder when divided by 12 is greater than 8. -/
def is_12_heavy (n : ℕ) : Prop := n % 12 > 8

/-- The set of three-digit natural numbers. -/
def three_digit_numbers : Set ℕ := {n : ℕ | 100 ≤ n ∧ n ≤ 999}

theorem least_three_digit_12_heavy : 
  ∀ n ∈ three_digit_numbers, is_12_heavy n → n ≥ 105 :=
by sorry

theorem is_105_12_heavy : is_12_heavy 105 :=
by sorry

theorem is_105_three_digit : 105 ∈ three_digit_numbers :=
by sorry

/-- 105 is the least three-digit 12-heavy whole number. -/
theorem least_three_digit_12_heavy_is_105 : 
  ∃ n ∈ three_digit_numbers, is_12_heavy n ∧ ∀ m ∈ three_digit_numbers, is_12_heavy m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_12_heavy_is_105_12_heavy_is_105_three_digit_least_three_digit_12_heavy_is_105_l3007_300761


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l3007_300762

theorem sum_of_roots_equation (x : ℝ) : 
  (10 = (x^3 - 5*x^2 - 10*x) / (x + 2)) → 
  (∃ (y z : ℝ), x + y + z = 5 ∧ 
    10 = (y^3 - 5*y^2 - 10*y) / (y + 2) ∧
    10 = (z^3 - 5*z^2 - 10*z) / (z + 2)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l3007_300762


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3007_300744

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 - 2*x^2 + 3 = (x^2 - 4*x + 7) * q + (28*x - 46) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3007_300744


namespace NUMINAMATH_CALUDE_divisible_by_64_l3007_300752

theorem divisible_by_64 (n : ℕ) (h : n > 0) : ∃ k : ℤ, 3^(2*n+2) - 8*n - 9 = 64*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_64_l3007_300752


namespace NUMINAMATH_CALUDE_sum_of_abs_roots_of_P_l3007_300782

/-- The polynomial P(x) = x^3 - 6x^2 + 5x + 12 -/
def P (x : ℝ) : ℝ := x^3 - 6*x^2 + 5*x + 12

/-- Theorem: The sum of the absolute values of the roots of P(x) is 8 -/
theorem sum_of_abs_roots_of_P :
  ∃ (x₁ x₂ x₃ : ℝ),
    (P x₁ = 0) ∧ (P x₂ = 0) ∧ (P x₃ = 0) ∧
    (∀ x, P x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    |x₁| + |x₂| + |x₃| = 8 :=
sorry

end NUMINAMATH_CALUDE_sum_of_abs_roots_of_P_l3007_300782


namespace NUMINAMATH_CALUDE_slope_angle_MN_l3007_300785

/-- Given points M(1, 2) and N(0, 1), the slope angle of line MN is π/4. -/
theorem slope_angle_MN : 
  let M : ℝ × ℝ := (1, 2)
  let N : ℝ × ℝ := (0, 1)
  let slope : ℝ := (M.2 - N.2) / (M.1 - N.1)
  let slope_angle : ℝ := Real.arctan slope
  slope_angle = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_MN_l3007_300785


namespace NUMINAMATH_CALUDE_largest_less_than_point_seven_l3007_300780

theorem largest_less_than_point_seven : 
  let numbers : List ℝ := [0.8, 1/2, 0.9]
  let target : ℝ := 0.7
  (∀ x ∈ numbers, x ≤ target → x ≤ (1/2 : ℝ)) ∧ 
  ((1/2 : ℝ) ∈ numbers) ∧ 
  ((1/2 : ℝ) < target) := by
  sorry

end NUMINAMATH_CALUDE_largest_less_than_point_seven_l3007_300780


namespace NUMINAMATH_CALUDE_binary_decimal_base5_conversion_l3007_300760

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to base 5
def decimal_to_base5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_decimal_base5_conversion :
  let binary : List Bool := [true, true, false, false, true, true]
  let decimal : Nat := 51
  let base5 : List Nat := [2, 0, 1]
  binary_to_decimal binary = decimal ∧ decimal_to_base5 decimal = base5 := by
  sorry


end NUMINAMATH_CALUDE_binary_decimal_base5_conversion_l3007_300760


namespace NUMINAMATH_CALUDE_different_rhetorical_device_l3007_300754

-- Define the rhetorical devices
inductive RhetoricalDevice
| Metaphor
| Personification

-- Define a function to assign rhetorical devices to options
def assignRhetoricalDevice (option : Char) : RhetoricalDevice :=
  match option with
  | 'A' => RhetoricalDevice.Metaphor
  | _ => RhetoricalDevice.Personification

-- Theorem statement
theorem different_rhetorical_device :
  ∀ (x : Char), x ≠ 'A' →
  assignRhetoricalDevice 'A' ≠ assignRhetoricalDevice x :=
by
  sorry

#check different_rhetorical_device

end NUMINAMATH_CALUDE_different_rhetorical_device_l3007_300754


namespace NUMINAMATH_CALUDE_remaining_popsicle_sticks_l3007_300736

theorem remaining_popsicle_sticks 
  (initial : ℝ) 
  (given_to_lisa : ℝ) 
  (given_to_peter : ℝ) 
  (given_to_you : ℝ) 
  (h1 : initial = 63.5) 
  (h2 : given_to_lisa = 18.2) 
  (h3 : given_to_peter = 21.7) 
  (h4 : given_to_you = 10.1) : 
  initial - (given_to_lisa + given_to_peter + given_to_you) = 13.5 := by
sorry

end NUMINAMATH_CALUDE_remaining_popsicle_sticks_l3007_300736


namespace NUMINAMATH_CALUDE_no_real_roots_l3007_300778

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem no_real_roots :
  ¬∃ x : ℝ, 1 - log10 (Real.sin x) = Real.cos x :=
sorry

end NUMINAMATH_CALUDE_no_real_roots_l3007_300778


namespace NUMINAMATH_CALUDE_sum_of_roots_l3007_300767

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1716 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l3007_300767


namespace NUMINAMATH_CALUDE_wendy_unrecycled_bags_l3007_300748

/-- Proves that Wendy did not recycle 2 bags given the problem conditions --/
theorem wendy_unrecycled_bags :
  ∀ (total_bags : ℕ) (points_per_bag : ℕ) (total_possible_points : ℕ),
    total_bags = 11 →
    points_per_bag = 5 →
    total_possible_points = 45 →
    total_bags - (total_possible_points / points_per_bag) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_wendy_unrecycled_bags_l3007_300748


namespace NUMINAMATH_CALUDE_stairs_climbed_together_l3007_300772

/-- The number of stairs Samir climbed -/
def samir_stairs : ℕ := 318

/-- The number of stairs Veronica climbed -/
def veronica_stairs : ℕ := samir_stairs / 2 + 18

/-- The total number of stairs Samir and Veronica climbed together -/
def total_stairs : ℕ := samir_stairs + veronica_stairs

theorem stairs_climbed_together : total_stairs = 495 := by
  sorry

end NUMINAMATH_CALUDE_stairs_climbed_together_l3007_300772


namespace NUMINAMATH_CALUDE_colorNGon_correct_l3007_300775

/-- The number of ways to color exactly k vertices of an n-gon in red,
    such that no two consecutive vertices are red. -/
def colorNGon (n k : ℕ) : ℕ :=
  Nat.choose (n - k - 1) (k - 1) + Nat.choose (n - k) k

/-- Theorem stating that the number of ways to color exactly k vertices of an n-gon in red,
    such that no two consecutive vertices are red, is equal to ⁽ⁿ⁻ᵏ⁻¹ᵏ⁻¹⁾ + ⁽ⁿ⁻ᵏᵏ⁾. -/
theorem colorNGon_correct (n k : ℕ) (h1 : n > k) (h2 : k > 0) :
  colorNGon n k = Nat.choose (n - k - 1) (k - 1) + Nat.choose (n - k) k := by
  sorry

#eval colorNGon 5 2  -- Example usage

end NUMINAMATH_CALUDE_colorNGon_correct_l3007_300775


namespace NUMINAMATH_CALUDE_double_xy_doubles_fraction_l3007_300797

/-- Given a fraction xy/(2x+y), prove that doubling both x and y results in doubling the fraction -/
theorem double_xy_doubles_fraction (x y : ℝ) (h : 2 * x + y ≠ 0) :
  (2 * x * 2 * y) / (2 * (2 * x) + 2 * y) = 2 * (x * y / (2 * x + y)) := by
  sorry

end NUMINAMATH_CALUDE_double_xy_doubles_fraction_l3007_300797


namespace NUMINAMATH_CALUDE_hot_dog_bun_distribution_l3007_300734

/-- Hot dog bun distribution problem -/
theorem hot_dog_bun_distribution
  (buns_per_package : ℕ)
  (packages_bought : ℕ)
  (num_classes : ℕ)
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages_bought = 30)
  (h3 : num_classes = 4)
  (h4 : students_per_class = 30) :
  (buns_per_package * packages_bought) / (num_classes * students_per_class) = 2 :=
sorry

end NUMINAMATH_CALUDE_hot_dog_bun_distribution_l3007_300734


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3007_300766

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 5 ∧ 1 / (x₀ + 2) + 1 / (y₀ + 2) = 4 / 9 :=
by sorry


end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3007_300766


namespace NUMINAMATH_CALUDE_drug_price_reduction_equation_l3007_300753

/-- Represents the price reduction scenario for a drug -/
def price_reduction (initial_price final_price : ℝ) (num_reductions : ℕ) (reduction_percentage : ℝ) : Prop :=
  initial_price * (1 - reduction_percentage) ^ num_reductions = final_price

/-- Theorem stating the equation for the drug price reduction scenario -/
theorem drug_price_reduction_equation :
  ∃ (x : ℝ), price_reduction 144 81 2 x :=
sorry

end NUMINAMATH_CALUDE_drug_price_reduction_equation_l3007_300753


namespace NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l3007_300790

theorem gcd_lcm_product_30_75 : Nat.gcd 30 75 * Nat.lcm 30 75 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_30_75_l3007_300790


namespace NUMINAMATH_CALUDE_square_difference_635_615_l3007_300714

theorem square_difference_635_615 : 635^2 - 615^2 = 25000 := by sorry

end NUMINAMATH_CALUDE_square_difference_635_615_l3007_300714


namespace NUMINAMATH_CALUDE_amanda_remaining_money_l3007_300727

/-- Calculates the remaining money after purchases -/
def remaining_money (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Proves that given the specific amounts in the problem, the remaining money is 7 -/
theorem amanda_remaining_money :
  remaining_money 50 9 2 25 = 7 := by
  sorry

end NUMINAMATH_CALUDE_amanda_remaining_money_l3007_300727


namespace NUMINAMATH_CALUDE_minimum_employees_science_bureau_hiring_l3007_300706

theorem minimum_employees (water : ℕ) (air : ℕ) (both : ℕ) : ℕ :=
  let total := water + air - both
  total

theorem science_bureau_hiring : 
  minimum_employees 98 89 34 = 153 := by sorry

end NUMINAMATH_CALUDE_minimum_employees_science_bureau_hiring_l3007_300706


namespace NUMINAMATH_CALUDE_walking_days_problem_l3007_300794

/-- 
Given:
- Jackie walks 2 miles per day
- Jessie walks 1.5 miles per day
- Over d days, Jackie walks 3 miles more than Jessie

Prove that d = 6
-/
theorem walking_days_problem (d : ℝ) 
  (h1 : 2 * d = 1.5 * d + 3) : d = 6 := by
  sorry

end NUMINAMATH_CALUDE_walking_days_problem_l3007_300794


namespace NUMINAMATH_CALUDE_decreasing_condition_l3007_300799

/-- The quadratic function f(x) = 2(x-1)^2 - 3 -/
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 - 3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4 * (x - 1)

theorem decreasing_condition (x : ℝ) : 
  x < 1 → f' x < 0 :=
sorry

end NUMINAMATH_CALUDE_decreasing_condition_l3007_300799


namespace NUMINAMATH_CALUDE_cube_root_four_solution_l3007_300758

theorem cube_root_four_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_four_solution_l3007_300758


namespace NUMINAMATH_CALUDE_weight_difference_l3007_300792

theorem weight_difference (jim steve stan : ℕ) 
  (h1 : steve < stan)
  (h2 : steve = jim - 8)
  (h3 : jim = 110)
  (h4 : stan + steve + jim = 319) :
  stan - steve = 5 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l3007_300792


namespace NUMINAMATH_CALUDE_cos_value_from_tan_sin_relation_l3007_300793

theorem cos_value_from_tan_sin_relation (θ : Real) 
  (h1 : 6 * Real.tan θ = 5 * Real.sin θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.cos θ = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_cos_value_from_tan_sin_relation_l3007_300793


namespace NUMINAMATH_CALUDE_marble_capacity_l3007_300771

theorem marble_capacity (v₁ v₂ : ℝ) (m₁ : ℕ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) :
  v₁ = 36 → m₁ = 180 → v₂ = 108 →
  (v₂ / v₁ * m₁ : ℝ) = 540 := by sorry

end NUMINAMATH_CALUDE_marble_capacity_l3007_300771


namespace NUMINAMATH_CALUDE_statement_IV_must_be_false_l3007_300733

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  value : Nat
  is_two_digit : 10 ≤ value ∧ value ≤ 99

/-- Represents the four statements about the number -/
structure Statements (n : TwoDigitNumber) where
  I : Bool
  II : Bool
  III : Bool
  IV : Bool
  I_def : I ↔ n.value = 12
  II_def : II ↔ n.value % 10 ≠ 2
  III_def : III ↔ n.value = 35
  IV_def : IV ↔ n.value % 10 ≠ 5
  three_true : I + II + III + IV = 3

theorem statement_IV_must_be_false (n : TwoDigitNumber) (s : Statements n) :
  s.IV = false :=
sorry

end NUMINAMATH_CALUDE_statement_IV_must_be_false_l3007_300733


namespace NUMINAMATH_CALUDE_all_sections_clearance_l3007_300701

/-- Represents the percentage of candidates who cleared a specific number of sections -/
structure SectionClearance where
  zero : ℝ
  one : ℝ
  two : ℝ
  three : ℝ
  four : ℝ
  five : ℝ

/-- Theorem stating the percentage of candidates who cleared all 5 sections -/
theorem all_sections_clearance 
  (total_candidates : ℕ) 
  (three_section_candidates : ℕ) 
  (clearance : SectionClearance) :
  total_candidates = 1200 →
  three_section_candidates = 300 →
  clearance.zero = 5 →
  clearance.one = 25 →
  clearance.two = 24.5 →
  clearance.four = 20 →
  clearance.three = (three_section_candidates : ℝ) / (total_candidates : ℝ) * 100 →
  clearance.five = 0.5 :=
by sorry

end NUMINAMATH_CALUDE_all_sections_clearance_l3007_300701


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l3007_300749

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ
  height : ℝ
  is_trapezoid : True
  AB_eq_CD : AB = CD
  BC_eq_10 : BC = 10
  AD_eq_22 : AD = 22
  height_eq_5 : height = 5

/-- The perimeter of the trapezoid ABCD is 2√61 + 32 -/
theorem trapezoid_perimeter (t : Trapezoid) : 
  t.AB + t.BC + t.CD + t.AD = 2 * Real.sqrt 61 + 32 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_perimeter_l3007_300749


namespace NUMINAMATH_CALUDE_everton_calculator_count_l3007_300755

/-- Represents the order of calculators by Everton college -/
structure CalculatorOrder where
  totalCost : ℕ
  scientificCost : ℕ
  graphingCost : ℕ
  scientificCount : ℕ

/-- Calculates the total number of calculators in an order -/
def totalCalculators (order : CalculatorOrder) : ℕ :=
  let graphingCount := (order.totalCost - order.scientificCount * order.scientificCost) / order.graphingCost
  order.scientificCount + graphingCount

/-- Theorem: The total number of calculators in Everton college's order is 45 -/
theorem everton_calculator_count :
  let order : CalculatorOrder := {
    totalCost := 1625,
    scientificCost := 10,
    graphingCost := 57,
    scientificCount := 20
  }
  totalCalculators order = 45 := by
  sorry

end NUMINAMATH_CALUDE_everton_calculator_count_l3007_300755


namespace NUMINAMATH_CALUDE_Q_value_l3007_300783

-- Define the relationship between P, Q, and U
def varies_directly_inversely (P Q U : ℚ) : Prop :=
  ∃ k : ℚ, P = k * Q / U

-- Define the initial conditions
def initial_conditions (P Q U : ℚ) : Prop :=
  P = 12 ∧ Q = 1/2 ∧ U = 16/25

-- Define the final conditions
def final_conditions (P U : ℚ) : Prop :=
  P = 27 ∧ U = 9/49

-- Theorem statement
theorem Q_value :
  ∀ P Q U : ℚ,
  varies_directly_inversely P Q U →
  initial_conditions P Q U →
  final_conditions P U →
  Q = 225/696 :=
by sorry

end NUMINAMATH_CALUDE_Q_value_l3007_300783


namespace NUMINAMATH_CALUDE_special_number_satisfies_conditions_special_number_unique_l3007_300713

/-- A two-digit number that satisfies the given conditions -/
def special_number : ℕ := 50

/-- The property that defines our special number -/
def is_special_number (a : ℕ) : Prop :=
  (a ≥ 10 ∧ a ≤ 99) ∧  -- Two-digit number
  (∃ (q r : ℚ), 
    (101 * a - a^2) / (0.04 * a^2) = q + r ∧
    q = a / 2 ∧
    r = a / (0.04 * a^2))

theorem special_number_satisfies_conditions : 
  is_special_number special_number :=
sorry

theorem special_number_unique : 
  ∀ (n : ℕ), is_special_number n → n = special_number :=
sorry

end NUMINAMATH_CALUDE_special_number_satisfies_conditions_special_number_unique_l3007_300713


namespace NUMINAMATH_CALUDE_negation_of_exp_positive_forall_l3007_300739

theorem negation_of_exp_positive_forall :
  (¬ ∀ x : ℝ, Real.exp x > 0) ↔ (∃ x : ℝ, Real.exp x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exp_positive_forall_l3007_300739


namespace NUMINAMATH_CALUDE_lottery_comparison_l3007_300777

-- Define the number of red and white balls
def red_balls : ℕ := 4
def white_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := red_balls + white_balls

-- Define the probability of drawing two red balls
def prob_two_red : ℚ := (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1))

-- Define the probability of rolling at least one four with two dice
def prob_at_least_one_four : ℚ := 1 - (5/6) * (5/6)

-- Theorem to prove
theorem lottery_comparison : prob_two_red > prob_at_least_one_four := by
  sorry


end NUMINAMATH_CALUDE_lottery_comparison_l3007_300777


namespace NUMINAMATH_CALUDE_same_color_probability_l3007_300712

def total_balls : ℕ := 13 + 7
def green_balls : ℕ := 13
def red_balls : ℕ := 7

theorem same_color_probability :
  (green_balls / total_balls) ^ 3 + (red_balls / total_balls) ^ 3 = 127 / 400 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l3007_300712


namespace NUMINAMATH_CALUDE_complementary_angles_difference_l3007_300781

theorem complementary_angles_difference (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a = 3 * b →   -- ratio of angles is 3:1
  |a - b| = 45  -- positive difference is 45°
:= by sorry

end NUMINAMATH_CALUDE_complementary_angles_difference_l3007_300781


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l3007_300795

theorem factorial_equation_solutions :
  ∀ (x y : ℕ) (z : ℤ),
    (Odd z) →
    (Nat.factorial x + Nat.factorial y = 24 * z + 2017) →
    ((x = 1 ∧ y = 4 ∧ z = -83) ∨
     (x = 4 ∧ y = 1 ∧ z = -83) ∨
     (x = 1 ∧ y = 5 ∧ z = -79) ∨
     (x = 5 ∧ y = 1 ∧ z = -79)) :=
by sorry


end NUMINAMATH_CALUDE_factorial_equation_solutions_l3007_300795


namespace NUMINAMATH_CALUDE_equation_a_is_linear_l3007_300757

/-- Definition of a linear equation -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- The equation x = 1 -/
def equation_a (x : ℝ) : ℝ := x - 1

theorem equation_a_is_linear : is_linear_equation equation_a := by
  sorry

#check equation_a_is_linear

end NUMINAMATH_CALUDE_equation_a_is_linear_l3007_300757


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_72_8712_l3007_300709

theorem gcd_lcm_sum_72_8712 : Nat.gcd 72 8712 + Nat.lcm 72 8712 = 26160 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_72_8712_l3007_300709


namespace NUMINAMATH_CALUDE_mangoes_rate_per_kg_l3007_300789

/-- Given the conditions of Harkamal's purchase, prove that the rate per kg of mangoes is 55. -/
theorem mangoes_rate_per_kg (grapes_quantity : ℕ) (grapes_rate : ℕ) (mangoes_quantity : ℕ) (total_paid : ℕ) :
  grapes_quantity = 8 →
  grapes_rate = 80 →
  mangoes_quantity = 9 →
  total_paid = 1135 →
  (total_paid - grapes_quantity * grapes_rate) / mangoes_quantity = 55 := by
  sorry

#eval (1135 - 8 * 80) / 9  -- This should evaluate to 55

end NUMINAMATH_CALUDE_mangoes_rate_per_kg_l3007_300789


namespace NUMINAMATH_CALUDE_range_of_m_l3007_300723

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |x - m| ≤ 2) →
  -1 ≤ m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3007_300723


namespace NUMINAMATH_CALUDE_blocks_eaten_l3007_300720

theorem blocks_eaten (initial_blocks remaining_blocks : ℕ) 
  (h1 : initial_blocks = 55)
  (h2 : remaining_blocks = 26) :
  initial_blocks - remaining_blocks = 29 := by
  sorry

end NUMINAMATH_CALUDE_blocks_eaten_l3007_300720


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3007_300786

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 - 17*x + 8 = 0 → ∃ s₁ s₂ : ℝ, s₁ + s₂ = 17 ∧ s₁ * s₂ = 8 ∧ s₁^2 + s₂^2 = 273 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3007_300786


namespace NUMINAMATH_CALUDE_two_pencils_length_l3007_300770

def pencil_length : ℕ := 12

theorem two_pencils_length : pencil_length + pencil_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_two_pencils_length_l3007_300770


namespace NUMINAMATH_CALUDE_jerrys_pool_length_l3007_300738

/-- Represents the problem of calculating Jerry's pool length --/
theorem jerrys_pool_length :
  ∀ (total_water drinking_cooking_water shower_water num_showers pool_width pool_height : ℝ),
    total_water = 1000 →
    drinking_cooking_water = 100 →
    shower_water = 20 →
    num_showers = 15 →
    pool_width = 10 →
    pool_height = 6 →
    ∃ (pool_length : ℝ),
      pool_length = 10 ∧
      pool_length * pool_width * pool_height = 
        total_water - (drinking_cooking_water + shower_water * num_showers) :=
by sorry

end NUMINAMATH_CALUDE_jerrys_pool_length_l3007_300738


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l3007_300717

/-- Represents the partnership profit calculation problem --/
theorem partnership_profit_calculation
  (p q r : ℕ) -- Initial capitals
  (h_ratio : p / q = 3 / 2 ∧ q / r = 4 / 3) -- Initial capital ratio
  (h_p_withdraw : ℕ) -- Amount p withdraws after 2 months
  (h_q_share : ℕ) -- q's share of profit in rupees
  (h_duration : ℕ) -- Total duration of partnership in months
  (h_p_withdraw_time : ℕ) -- Time after which p withdraws half capital
  (h_p_withdraw_half : h_p_withdraw = p / 2) -- p withdraws half of initial capital
  (h_duration_val : h_duration = 12) -- Total duration is 12 months
  (h_p_withdraw_time_val : h_p_withdraw_time = 2) -- p withdraws after 2 months
  (h_q_share_val : h_q_share = 144) -- q's share is Rs 144
  : ∃ (total_profit : ℕ), total_profit = 486 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l3007_300717


namespace NUMINAMATH_CALUDE_stock_percentage_sold_l3007_300724

/-- Proves that the percentage of stock sold is 0.25% given the specified conditions --/
theorem stock_percentage_sold (cash_realized : ℝ) (brokerage_rate : ℝ) (net_amount : ℝ)
  (h1 : cash_realized = 108.25)
  (h2 : brokerage_rate = 1 / 4 / 100)
  (h3 : net_amount = 108) :
  let brokerage_fee := cash_realized * brokerage_rate
  let percentage_sold := brokerage_fee / cash_realized * 100
  percentage_sold = 0.25 := by sorry

end NUMINAMATH_CALUDE_stock_percentage_sold_l3007_300724


namespace NUMINAMATH_CALUDE_prob_two_red_from_bag_l3007_300746

/-- The probability of picking two red balls from a bag -/
def probability_two_red_balls (red blue green : ℕ) : ℚ :=
  let total := red + blue + green
  (red : ℚ) / total * ((red - 1) : ℚ) / (total - 1)

/-- Theorem: The probability of picking two red balls from a bag with 3 red, 2 blue, and 4 green balls is 1/12 -/
theorem prob_two_red_from_bag : probability_two_red_balls 3 2 4 = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_from_bag_l3007_300746


namespace NUMINAMATH_CALUDE_expression_factorization_l3007_300747

theorem expression_factorization (x : ℝ) :
  (3 * x^3 + 70 * x^2 - 5) - (-4 * x^3 + 2 * x^2 - 5) = 7 * x^2 * (x + 68/7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l3007_300747


namespace NUMINAMATH_CALUDE_square_ratio_sum_l3007_300796

theorem square_ratio_sum (p q r : ℕ) : 
  (75 : ℚ) / 128 = (p * Real.sqrt q / r) ^ 2 → p + q + r = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_ratio_sum_l3007_300796


namespace NUMINAMATH_CALUDE_unchanged_total_plates_l3007_300716

/-- Represents the number of elements in each set of letters for license plates --/
structure LicensePlateSets :=
  (first : Nat)
  (second : Nat)
  (third : Nat)

/-- Calculates the total number of possible license plates --/
def totalPlates (sets : LicensePlateSets) : Nat :=
  sets.first * sets.second * sets.third

/-- The original configuration of letter sets --/
def originalSets : LicensePlateSets :=
  { first := 5, second := 3, third := 4 }

/-- The new configuration after moving one letter from the first to the third set --/
def newSets : LicensePlateSets :=
  { first := 4, second := 3, third := 5 }

/-- Theorem stating that the total number of license plates remains unchanged --/
theorem unchanged_total_plates :
  totalPlates originalSets = totalPlates newSets :=
by sorry

end NUMINAMATH_CALUDE_unchanged_total_plates_l3007_300716


namespace NUMINAMATH_CALUDE_zCoordinate_when_x_is_seven_l3007_300703

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- Calculate the z-coordinate for a given x-coordinate on the line -/
def zCoordinate (l : Line3D) (x : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for the given line, when x = 7, z = -4 -/
theorem zCoordinate_when_x_is_seven :
  let l : Line3D := { point1 := (1, 3, 2), point2 := (4, 4, -1) }
  zCoordinate l 7 = -4 := by
  sorry

end NUMINAMATH_CALUDE_zCoordinate_when_x_is_seven_l3007_300703


namespace NUMINAMATH_CALUDE_employee_payment_percentage_l3007_300740

theorem employee_payment_percentage (total_payment y_payment : ℚ) 
  (h1 : total_payment = 616)
  (h2 : y_payment = 280) : 
  (total_payment - y_payment) / y_payment * 100 = 120 := by
  sorry

end NUMINAMATH_CALUDE_employee_payment_percentage_l3007_300740


namespace NUMINAMATH_CALUDE_interest_rate_equality_l3007_300731

theorem interest_rate_equality (I : ℝ) (r : ℝ) : 
  I = 1000 * 0.12 * 2 → 
  I = 200 * r * 12 → 
  r = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equality_l3007_300731


namespace NUMINAMATH_CALUDE_coloring_exists_l3007_300769

/-- A coloring of numbers from 1 to 2n -/
def Coloring (n : ℕ) := Fin (2*n) → Fin n

/-- Predicate to check if a coloring is valid -/
def ValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  (∀ color : Fin n, ∃! (a b : Fin (2*n)), c a = color ∧ c b = color ∧ a ≠ b) ∧
  (∀ diff : Fin n, ∃! (a b : Fin (2*n)), c a = c b ∧ a ≠ b ∧ a.val - b.val = diff.val + 1)

/-- The sequence of n for which the coloring is possible -/
def ColoringSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 3 * ColoringSequence n + 1

theorem coloring_exists (m : ℕ) : ∃ c : Coloring (ColoringSequence m), ValidColoring (ColoringSequence m) c := by
  sorry

end NUMINAMATH_CALUDE_coloring_exists_l3007_300769


namespace NUMINAMATH_CALUDE_probability_is_two_ninety_one_l3007_300765

/-- Represents the number of jellybeans of each color in the basket -/
structure JellyBeanBasket where
  red : Nat
  blue : Nat
  yellow : Nat

/-- Calculates the probability of picking exactly 2 red and 2 blue jellybeans -/
def probability_two_red_two_blue (basket : JellyBeanBasket) : Rat :=
  let total := basket.red + basket.blue + basket.yellow
  let favorable := Nat.choose basket.red 2 * Nat.choose basket.blue 2
  let total_combinations := Nat.choose total 4
  favorable / total_combinations

/-- The main theorem stating the probability is 2/91 -/
theorem probability_is_two_ninety_one :
  probability_two_red_two_blue ⟨5, 3, 7⟩ = 2 / 91 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_two_ninety_one_l3007_300765


namespace NUMINAMATH_CALUDE_unique_polynomial_reconstruction_l3007_300708

/-- A polynomial with non-negative integer coefficients -/
def NonNegIntPolynomial (P : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, ∀ k > n, P k = 0

/-- The polynomial is non-constant -/
def NonConstant (P : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, P k ≠ P 0

theorem unique_polynomial_reconstruction
  (P : ℕ → ℕ)
  (h_non_neg : NonNegIntPolynomial P)
  (h_non_const : NonConstant P) :
  ∀ Q : ℕ → ℕ,
    NonNegIntPolynomial Q →
    NonConstant Q →
    P 2 = Q 2 →
    P (P 2) = Q (Q 2) →
    ∀ x, P x = Q x :=
sorry

end NUMINAMATH_CALUDE_unique_polynomial_reconstruction_l3007_300708


namespace NUMINAMATH_CALUDE_circle_with_parallel_tangents_l3007_300710

-- Define the type for points in 2D space
def Point := ℝ × ℝ

-- Define three non-collinear points
variable (A B C : Point)

-- Define the property of non-collinearity
def NonCollinear (A B C : Point) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (x₂ - x₁) * (y₃ - y₁) ≠ (y₂ - y₁) * (x₃ - x₁)

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a tangent line to a circle
def IsTangent (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define parallel lines
def Parallel (l₁ l₂ : Point → Prop) : Prop :=
  ∀ (p q : Point), l₁ p ∧ l₂ q → ∃ (k : ℝ), k ≠ 0 ∧ 
    (p.1 - q.1) * k = (p.2 - q.2)

-- Theorem statement
theorem circle_with_parallel_tangents 
  (h : NonCollinear A B C) : 
  ∃ (c : Circle), c.center = C ∧ 
    ∃ (t₁ t₂ : Point → Prop), 
      IsTangent A c ∧ IsTangent B c ∧ 
      Parallel t₁ t₂ :=
sorry

end NUMINAMATH_CALUDE_circle_with_parallel_tangents_l3007_300710


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l3007_300700

/-- The probability of drawing two chips of different colors from a bag with replacement -/
theorem different_color_chips_probability
  (total_chips : ℕ)
  (blue_chips : ℕ)
  (yellow_chips : ℕ)
  (h_total : total_chips = blue_chips + yellow_chips)
  (h_blue : blue_chips = 5)
  (h_yellow : yellow_chips = 3) :
  (blue_chips : ℚ) / total_chips * (yellow_chips : ℚ) / total_chips +
  (yellow_chips : ℚ) / total_chips * (blue_chips : ℚ) / total_chips =
  15 / 32 :=
sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l3007_300700


namespace NUMINAMATH_CALUDE_total_tanks_needed_l3007_300773

/-- Calculates the minimum number of tanks needed to fill all balloons --/
def minTanksNeeded (smallBalloons mediumBalloons largeBalloons : Nat)
  (smallCapacity mediumCapacity largeCapacity : Nat)
  (heliumTankCapacity hydrogenTankCapacity mixtureTankCapacity : Nat) : Nat :=
  let heliumNeeded := smallBalloons * smallCapacity
  let hydrogenNeeded := mediumBalloons * mediumCapacity
  let mixtureNeeded := largeBalloons * largeCapacity
  let heliumTanks := (heliumNeeded + heliumTankCapacity - 1) / heliumTankCapacity
  let hydrogenTanks := (hydrogenNeeded + hydrogenTankCapacity - 1) / hydrogenTankCapacity
  let mixtureTanks := (mixtureNeeded + mixtureTankCapacity - 1) / mixtureTankCapacity
  heliumTanks + hydrogenTanks + mixtureTanks

theorem total_tanks_needed :
  minTanksNeeded 5000 5000 5000 20 30 50 1000 1200 1500 = 392 := by
  sorry

#eval minTanksNeeded 5000 5000 5000 20 30 50 1000 1200 1500

end NUMINAMATH_CALUDE_total_tanks_needed_l3007_300773


namespace NUMINAMATH_CALUDE_triangle_area_with_perimeter_12_l3007_300725

/-- A triangle with integral sides and perimeter 12 has an area of 6 -/
theorem triangle_area_with_perimeter_12 :
  ∀ a b c : ℕ,
  a + b + c = 12 →
  a + b > c →
  b + c > a →
  c + a > b →
  (a * b : ℝ) / 2 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_area_with_perimeter_12_l3007_300725


namespace NUMINAMATH_CALUDE_norm_scalar_multiple_l3007_300741

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem norm_scalar_multiple (v : E) (h : ‖v‖ = 7) : ‖(5 : ℝ) • v‖ = 35 := by
  sorry

end NUMINAMATH_CALUDE_norm_scalar_multiple_l3007_300741


namespace NUMINAMATH_CALUDE_weight_loss_difference_equals_303_l3007_300737

/-- Calculates the total weight loss difference between Luca and Kim combined, and Barbi -/
def weight_loss_difference : ℝ :=
  let barbi_monthly_loss : ℝ := 1.5
  let barbi_months : ℕ := 2 * 12
  let luca_yearly_loss : ℝ := 9
  let luca_years : ℕ := 15
  let kim_first_year_monthly_loss : ℝ := 2
  let kim_remaining_monthly_loss : ℝ := 3
  let kim_remaining_months : ℕ := 5 * 12

  let barbi_total_loss := barbi_monthly_loss * barbi_months
  let luca_total_loss := luca_yearly_loss * luca_years
  let kim_first_year_loss := kim_first_year_monthly_loss * 12
  let kim_remaining_loss := kim_remaining_monthly_loss * kim_remaining_months
  let kim_total_loss := kim_first_year_loss + kim_remaining_loss

  (luca_total_loss + kim_total_loss) - barbi_total_loss

theorem weight_loss_difference_equals_303 : weight_loss_difference = 303 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_difference_equals_303_l3007_300737


namespace NUMINAMATH_CALUDE_square_remainder_mod_nine_l3007_300707

theorem square_remainder_mod_nine (n : ℤ) : 
  (n % 9 = 1 ∨ n % 9 = 8) → (n^2) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_remainder_mod_nine_l3007_300707


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_M_l3007_300728

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {x | x*(x-1)*(x-2) = 0}

theorem M_intersect_N_equals_M : M ∩ N = M := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_M_l3007_300728


namespace NUMINAMATH_CALUDE_calculate_expression_l3007_300722

theorem calculate_expression : |-7| + Real.sqrt 16 - (-3)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3007_300722


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l3007_300774

theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_diameter : ℝ)
  (hw : rectangle_width = 15)
  (hh : rectangle_height = 17)
  (hd : circle_diameter = 7) :
  let inner_width := rectangle_width - circle_diameter
  let inner_height := rectangle_height - circle_diameter
  Real.sqrt (inner_width ^ 2 + inner_height ^ 2) = Real.sqrt 164 :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l3007_300774


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_four_digit_numbers_exist_l3007_300718

def A (n m : ℕ) := n.factorial / (n - m).factorial

theorem four_digit_numbers_count : ℕ → Prop :=
  fun count => (count = A 5 4 - A 4 3) ∧ 
               (count = A 4 1 * A 4 3) ∧ 
               (count = A 4 4 + 3 * A 4 3) ∧ 
               (count ≠ A 5 4 - A 4 4)

theorem four_digit_numbers_exist : ∃ count : ℕ, four_digit_numbers_count count := by
  sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_four_digit_numbers_exist_l3007_300718


namespace NUMINAMATH_CALUDE_trig_expression_equality_l3007_300751

theorem trig_expression_equality : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_expression_equality_l3007_300751


namespace NUMINAMATH_CALUDE_periodic_last_digit_triangular_perfect_square_between_sums_l3007_300711

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

def last_digit (n : ℕ) : ℕ := n % 10

def sum_triangular_numbers (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

theorem periodic_last_digit_triangular :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, last_digit (triangular_number n) = last_digit (triangular_number (n + k)) :=
sorry

theorem perfect_square_between_sums (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℕ, sum_triangular_numbers (n - 1) < k * k ∧ k * k < sum_triangular_numbers n :=
sorry

end NUMINAMATH_CALUDE_periodic_last_digit_triangular_perfect_square_between_sums_l3007_300711


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3007_300759

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + 2*x*y + 2*y^2 ≥ 0 ∧ (x^2 + 2*x*y + 2*y^2 = 0 ↔ x = 0 ∧ y = 0) :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3007_300759


namespace NUMINAMATH_CALUDE_max_pigs_buyable_l3007_300702

def budget : ℕ := 1300
def pig_cost : ℕ := 21
def duck_cost : ℕ := 23
def min_ducks : ℕ := 20

theorem max_pigs_buyable :
  ∀ p d : ℕ,
    p > 0 →
    d ≥ min_ducks →
    pig_cost * p + duck_cost * d ≤ budget →
    p ≤ 40 ∧
    ∃ p' d' : ℕ, p' = 40 ∧ d' ≥ min_ducks ∧ pig_cost * p' + duck_cost * d' = budget :=
by sorry

end NUMINAMATH_CALUDE_max_pigs_buyable_l3007_300702


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3007_300750

theorem max_books_borrowed (total_students : Nat) (zero_books : Nat) (one_book : Nat) 
  (two_books : Nat) (three_books : Nat) (avg_books : Nat) (max_books : Nat) :
  total_students = 50 →
  zero_books = 4 →
  one_book = 15 →
  two_books = 9 →
  three_books = 7 →
  avg_books = 3 →
  max_books = 10 →
  ∃ (max_single : Nat),
    max_single ≤ max_books ∧
    max_single = 40 ∧
    (total_students * avg_books - (one_book + 2 * two_books + 3 * three_books)) % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l3007_300750


namespace NUMINAMATH_CALUDE_afternoon_snowfall_l3007_300704

theorem afternoon_snowfall (total : ℝ) (morning : ℝ) (afternoon : ℝ)
  (h1 : total = 0.625)
  (h2 : morning = 0.125)
  (h3 : total = morning + afternoon) :
  afternoon = 0.500 := by
sorry

end NUMINAMATH_CALUDE_afternoon_snowfall_l3007_300704


namespace NUMINAMATH_CALUDE_cosine_in_acute_triangle_l3007_300715

theorem cosine_in_acute_triangle (A B C : Real) (a b c : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π/2 →
  (1/2) * a * b * Real.sin C = 5 →
  a = 3 →
  b = 4 →
  Real.cos C = Real.sqrt 11 / 6 := by
sorry

end NUMINAMATH_CALUDE_cosine_in_acute_triangle_l3007_300715


namespace NUMINAMATH_CALUDE_hockey_cards_count_l3007_300784

theorem hockey_cards_count (hockey : ℕ) (football : ℕ) (baseball : ℕ) : 
  baseball = football - 50 →
  football = 4 * hockey →
  hockey + football + baseball = 1750 →
  hockey = 200 := by
sorry

end NUMINAMATH_CALUDE_hockey_cards_count_l3007_300784


namespace NUMINAMATH_CALUDE_rectangles_with_equal_areas_have_reciprocal_proportions_l3007_300729

theorem rectangles_with_equal_areas_have_reciprocal_proportions 
  (a b c d : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : d > 0) 
  (h5 : a * b = c * d) : 
  a / c = d / b := by
sorry


end NUMINAMATH_CALUDE_rectangles_with_equal_areas_have_reciprocal_proportions_l3007_300729
