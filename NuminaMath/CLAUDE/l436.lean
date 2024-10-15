import Mathlib

namespace NUMINAMATH_CALUDE_volume_three_triangular_pyramids_l436_43691

/-- The volume of three identical triangular pyramids -/
theorem volume_three_triangular_pyramids 
  (base_measurement : ℝ) 
  (base_height : ℝ) 
  (pyramid_height : ℝ) 
  (h1 : base_measurement = 40) 
  (h2 : base_height = 20) 
  (h3 : pyramid_height = 30) : 
  3 * (1/3 * (1/2 * base_measurement * base_height) * pyramid_height) = 12000 := by
  sorry

#check volume_three_triangular_pyramids

end NUMINAMATH_CALUDE_volume_three_triangular_pyramids_l436_43691


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l436_43601

theorem min_perimeter_triangle (a b c : ℕ) (A B C : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  Real.cos A = 3/5 →
  Real.cos B = 5/13 →
  Real.cos C = -1/3 →
  a + b > c ∧ b + c > a ∧ c + a > b →
  (∀ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 →
    Real.cos A = 3/5 →
    Real.cos B = 5/13 →
    Real.cos C = -1/3 →
    x + y > z ∧ y + z > x ∧ z + x > y →
    a + b + c ≤ x + y + z) →
  a + b + c = 192 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l436_43601


namespace NUMINAMATH_CALUDE_max_value_implies_a_l436_43661

/-- Given a function f(x) = 2x^3 - 3x^2 + a, prove that if its maximum value is 6, then a = 6 -/
theorem max_value_implies_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = 2 * x^3 - 3 * x^2 + a)
  (h2 : ∃ M, M = 6 ∧ ∀ x, f x ≤ M) : 
  a = 6 := by sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l436_43661


namespace NUMINAMATH_CALUDE_scale_model_height_l436_43695

/-- The scale ratio of the model to the actual skyscraper -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the skyscraper in feet -/
def actual_height : ℕ := 1250

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- The height of the scale model in inches -/
def model_height_inches : ℕ := 600

/-- Theorem stating that the height of the scale model in inches is 600 -/
theorem scale_model_height :
  (actual_height : ℚ) * scale_ratio * inches_per_foot = model_height_inches := by
  sorry

end NUMINAMATH_CALUDE_scale_model_height_l436_43695


namespace NUMINAMATH_CALUDE_pentagon_area_sum_l436_43602

/-- Represents a pentagon with given side lengths and angle -/
structure Pentagon where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DE : ℝ
  EA : ℝ
  angleCDE : ℝ
  ABparallelDE : Prop

/-- Represents the area of a pentagon in the form √a + b·√c -/
structure PentagonArea where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Function to calculate the area of a pentagon -/
noncomputable def calculatePentagonArea (p : Pentagon) : ℝ := sorry

/-- Function to express the pentagon area in the form √a + b·√c -/
noncomputable def expressAreaAsSum (area : ℝ) : PentagonArea := sorry

theorem pentagon_area_sum (p : Pentagon) 
  (h1 : p.AB = 8)
  (h2 : p.BC = 4)
  (h3 : p.CD = 10)
  (h4 : p.DE = 7)
  (h5 : p.EA = 10)
  (h6 : p.angleCDE = π / 3)  -- 60° in radians
  (h7 : p.ABparallelDE) :
  let area := calculatePentagonArea p
  let expression := expressAreaAsSum area
  expression.a + expression.b + expression.c = 39 := by sorry

end NUMINAMATH_CALUDE_pentagon_area_sum_l436_43602


namespace NUMINAMATH_CALUDE_total_remaining_apples_l436_43618

def tree_A : ℕ := 200
def tree_B : ℕ := 250
def tree_C : ℕ := 300

def picked_A : ℕ := tree_A / 5
def picked_B : ℕ := 2 * picked_A
def picked_C : ℕ := picked_A + 20

def remaining_A : ℕ := tree_A - picked_A
def remaining_B : ℕ := tree_B - picked_B
def remaining_C : ℕ := tree_C - picked_C

theorem total_remaining_apples :
  remaining_A + remaining_B + remaining_C = 570 := by
  sorry

end NUMINAMATH_CALUDE_total_remaining_apples_l436_43618


namespace NUMINAMATH_CALUDE_complement_of_B_l436_43621

-- Define the set B
def B : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem complement_of_B : 
  (Set.univ : Set ℝ) \ B = {x | x < -2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_B_l436_43621


namespace NUMINAMATH_CALUDE_yoojung_notebooks_l436_43693

theorem yoojung_notebooks :
  ∀ (initial : ℕ), 
  (initial ≥ 5) →
  (initial - 5) % 2 = 0 →
  ((initial - 5) / 2 - (initial - 5) / 2 / 2 = 4) →
  initial = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_yoojung_notebooks_l436_43693


namespace NUMINAMATH_CALUDE_expression_simplification_l436_43608

theorem expression_simplification :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l436_43608


namespace NUMINAMATH_CALUDE_binomial_square_constant_l436_43670

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 300*x + c = (x + a)^2) → c = 22500 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l436_43670


namespace NUMINAMATH_CALUDE_ben_egg_count_l436_43626

/-- Given that Ben has 7 trays of eggs and each tray contains 10 eggs,
    prove that the total number of eggs Ben examined is 70. -/
theorem ben_egg_count (num_trays : ℕ) (eggs_per_tray : ℕ) :
  num_trays = 7 → eggs_per_tray = 10 → num_trays * eggs_per_tray = 70 := by
  sorry

end NUMINAMATH_CALUDE_ben_egg_count_l436_43626


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l436_43619

/-- Given three non-zero real numbers x, y, and z forming a geometric sequence
    x(y-z), y(z-x), and z(y-x), prove that the common ratio q satisfies q^2 - q - 1 = 0 -/
theorem geometric_sequence_common_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hseq : ∃ q : ℝ, q ≠ 0 ∧ y * (z - x) = q * (x * (y - z)) ∧ z * (y - x) = q * (y * (z - x))) :
  ∃ q : ℝ, q^2 - q - 1 = 0 ∧ y * (z - x) = q * (x * (y - z)) ∧ z * (y - x) = q * (y * (z - x)) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l436_43619


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l436_43698

theorem no_positive_integer_solution :
  ¬ ∃ (x y z t : ℕ+), (x^2 + 5*y^2 = z^2) ∧ (5*x^2 + y^2 = t^2) := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l436_43698


namespace NUMINAMATH_CALUDE_base_n_representation_of_b_l436_43663

/-- Represents a number in base n --/
structure BaseN (n : ℕ) where
  value : ℕ
  is_valid : value < n^2

/-- Convert a decimal number to base n --/
def to_base_n (n : ℕ) (x : ℕ) : BaseN n :=
  ⟨x % (n^2), by sorry⟩

/-- Convert a base n number to decimal --/
def from_base_n {n : ℕ} (x : BaseN n) : ℕ :=
  x.value

theorem base_n_representation_of_b (n m a b : ℕ) : 
  n > 9 →
  n^2 - a*n + b = 0 →
  m^2 - a*m + b = 0 →
  to_base_n n a = ⟨21, by sorry⟩ →
  to_base_n n (n + m) = ⟨30, by sorry⟩ →
  to_base_n n b = ⟨200, by sorry⟩ := by
  sorry


end NUMINAMATH_CALUDE_base_n_representation_of_b_l436_43663


namespace NUMINAMATH_CALUDE_ellipse_slope_at_pi_third_l436_43657

/-- The slope of the line connecting the origin to a point on an ellipse --/
theorem ellipse_slope_at_pi_third :
  let x (t : Real) := 2 * Real.cos t
  let y (t : Real) := 4 * Real.sin t
  let t₀ : Real := Real.pi / 3
  let x₀ : Real := x t₀
  let y₀ : Real := y t₀
  (y₀ - 0) / (x₀ - 0) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_slope_at_pi_third_l436_43657


namespace NUMINAMATH_CALUDE_max_value_x_plus_inverse_l436_43620

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (∀ y : ℝ, y > 0 → 13 = y^2 + 1/y^2 → x + 1/x ≥ y + 1/y) ∧ x + 1/x = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_plus_inverse_l436_43620


namespace NUMINAMATH_CALUDE_inequality_multiplication_l436_43603

theorem inequality_multiplication (x y : ℝ) (h : x < y) : 2 * x < 2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l436_43603


namespace NUMINAMATH_CALUDE_fair_coin_toss_is_fair_l436_43689

-- Define a fair coin
def fair_coin (outcome : Bool) : ℝ :=
  if outcome then 0.5 else 0.5

-- Define fairness of a decision method
def is_fair (decision_method : Bool → ℝ) : Prop :=
  decision_method true = decision_method false

-- Theorem statement
theorem fair_coin_toss_is_fair :
  is_fair fair_coin :=
sorry

end NUMINAMATH_CALUDE_fair_coin_toss_is_fair_l436_43689


namespace NUMINAMATH_CALUDE_complex_to_exponential_l436_43614

theorem complex_to_exponential (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → z = 2 * Complex.exp (Complex.I * (Real.pi / 3)) := by
  sorry

end NUMINAMATH_CALUDE_complex_to_exponential_l436_43614


namespace NUMINAMATH_CALUDE_multiply_powers_of_x_l436_43686

theorem multiply_powers_of_x (x : ℝ) : 2 * x * (3 * x^2) = 6 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_x_l436_43686


namespace NUMINAMATH_CALUDE_power_of_power_l436_43625

theorem power_of_power : (3^3)^2 = 729 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l436_43625


namespace NUMINAMATH_CALUDE_rent_expenditure_l436_43631

theorem rent_expenditure (x : ℝ) 
  (h1 : x + 0.7 * x + 32 = 100) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_rent_expenditure_l436_43631


namespace NUMINAMATH_CALUDE_inequality_proof_l436_43662

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  a * b * (a - b) + b * c * (b - c) + c * d * (c - d) + d * a * (d - a) ≤ 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l436_43662


namespace NUMINAMATH_CALUDE_last_element_value_l436_43635

/-- Represents a triangular number table -/
def TriangularTable (n : ℕ) : Type :=
  Fin n → Fin n → ℕ

/-- The first row of the table contains the first n positive integers -/
def FirstRowCorrect (t : TriangularTable 100) : Prop :=
  ∀ i : Fin 100, t 0 i = i.val + 1

/-- Each element (except in the first row) is the sum of two elements above it -/
def ElementSum (t : TriangularTable 100) : Prop :=
  ∀ (i : Fin 99) (j : Fin (99 - i.val)), 
    t (i + 1) j = t i j + t i (j + 1)

/-- The last row contains only one element -/
def LastRowSingleton (t : TriangularTable 100) : Prop :=
  ∀ j : Fin 100, j.val > 0 → t 99 j = 0

/-- The main theorem: given the conditions, the last element is 101 * 2^98 -/
theorem last_element_value (t : TriangularTable 100) 
  (h1 : FirstRowCorrect t) 
  (h2 : ElementSum t)
  (h3 : LastRowSingleton t) : 
  t 99 0 = 101 * 2^98 := by
  sorry

end NUMINAMATH_CALUDE_last_element_value_l436_43635


namespace NUMINAMATH_CALUDE_function_behavior_l436_43672

theorem function_behavior (f : ℝ → ℝ) (h : ∀ x : ℝ, f x < f (x + 1)) :
  (∃ a b : ℝ, a < b ∧ ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y) ∧
  (∃ g : ℝ → ℝ, (∀ x : ℝ, g x < g (x + 1)) ∧
    ∀ a b : ℝ, a < b → ∃ x y, a ≤ x ∧ x < y ∧ y ≤ b ∧ g x ≥ g y) :=
by sorry

end NUMINAMATH_CALUDE_function_behavior_l436_43672


namespace NUMINAMATH_CALUDE_f_g_f_3_equals_1360_l436_43676

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x + 4
def g (x : ℝ) : ℝ := x^2 + 5 * x + 3

-- State the theorem
theorem f_g_f_3_equals_1360 : f (g (f 3)) = 1360 := by
  sorry

end NUMINAMATH_CALUDE_f_g_f_3_equals_1360_l436_43676


namespace NUMINAMATH_CALUDE_square_root_two_minus_one_squared_plus_two_times_plus_three_l436_43640

theorem square_root_two_minus_one_squared_plus_two_times_plus_three (x : ℝ) :
  x = Real.sqrt 2 - 1 → x^2 + 2*x + 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_two_minus_one_squared_plus_two_times_plus_three_l436_43640


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_to_2017_l436_43655

theorem last_four_digits_of_5_to_2017 :
  ∃ n : ℕ, 5^2017 ≡ 3125 [ZMOD 10000] :=
by
  -- We define the cycle of last four digits
  let cycle := [3125, 5625, 8125, 0625]
  
  -- We state that 5^5, 5^6, and 5^7 match the first three elements of the cycle
  have h1 : 5^5 ≡ cycle[0] [ZMOD 10000] := by sorry
  have h2 : 5^6 ≡ cycle[1] [ZMOD 10000] := by sorry
  have h3 : 5^7 ≡ cycle[2] [ZMOD 10000] := by sorry
  
  -- We state that the cycle repeats every 4 terms
  have h_cycle : ∀ k : ℕ, 5^(k+4) ≡ 5^k [ZMOD 10000] := by sorry
  
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_last_four_digits_of_5_to_2017_l436_43655


namespace NUMINAMATH_CALUDE_sqrt_sum_difference_l436_43685

theorem sqrt_sum_difference (x : ℝ) : 
  Real.sqrt 8 + Real.sqrt 18 - 4 * Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_difference_l436_43685


namespace NUMINAMATH_CALUDE_complex_roots_quadratic_l436_43681

theorem complex_roots_quadratic (b c : ℝ) : 
  (Complex.I + 1) ^ 2 + b * (Complex.I + 1) + c = 0 →
  (b = -2 ∧ c = 2) ∧ 
  ((Complex.I - 1) ^ 2 + b * (Complex.I - 1) + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_roots_quadratic_l436_43681


namespace NUMINAMATH_CALUDE_inequality_solution_range_l436_43653

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) → 
  (a < -2 ∨ a ≥ 6/5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l436_43653


namespace NUMINAMATH_CALUDE_group_size_calculation_l436_43610

theorem group_size_calculation (initial_avg : ℝ) (new_person_age : ℝ) (new_avg : ℝ) : 
  initial_avg = 15 → new_person_age = 37 → new_avg = 17 → 
  ∃ n : ℕ, (n : ℝ) * initial_avg + new_person_age = (n + 1) * new_avg ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_group_size_calculation_l436_43610


namespace NUMINAMATH_CALUDE_height_of_C_l436_43642

/-- Given three people A, B, and C with heights hA, hB, and hC respectively (in cm),
    prove that C's height is 143 cm under the following conditions:
    1. The average height of A, B, and C is 143 cm.
    2. A's height increased by 4.5 cm becomes the average height of B and C.
    3. B is 3 cm taller than C. -/
theorem height_of_C (hA hB hC : ℝ) : 
  (hA + hB + hC) / 3 = 143 →
  hA + 4.5 = (hB + hC) / 2 →
  hB = hC + 3 →
  hC = 143 := by sorry

end NUMINAMATH_CALUDE_height_of_C_l436_43642


namespace NUMINAMATH_CALUDE_square_sides_theorem_l436_43607

theorem square_sides_theorem (total_length : ℝ) (area_difference : ℝ) 
  (h1 : total_length = 20)
  (h2 : area_difference = 120) :
  ∃ (x y : ℝ), x + y = total_length ∧ x^2 - y^2 = area_difference ∧ x = 13 ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_square_sides_theorem_l436_43607


namespace NUMINAMATH_CALUDE_unique_pair_with_single_solution_l436_43609

theorem unique_pair_with_single_solution :
  ∃! p : ℕ × ℕ, 
    let b := p.1
    let c := p.2
    b > 0 ∧ c > 0 ∧
    (∃! x : ℝ, x^2 + b*x + c = 0) ∧
    (∃! x : ℝ, x^2 + c*x + b = 0) :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_with_single_solution_l436_43609


namespace NUMINAMATH_CALUDE_complex_number_location_l436_43680

theorem complex_number_location (z : ℂ) (h : (1 : ℂ) + Complex.I = Complex.I / z) :
  0 < z.re ∧ 0 < z.im := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l436_43680


namespace NUMINAMATH_CALUDE_smallest_square_arrangement_l436_43651

theorem smallest_square_arrangement : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬ ∃ k : ℕ+, m * (1^2 + 2^2 + 3^2) = k^2) ∧
  (∃ k : ℕ+, n * (1^2 + 2^2 + 3^2) = k^2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_square_arrangement_l436_43651


namespace NUMINAMATH_CALUDE_function_characterization_l436_43652

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f n) - f m = n

-- Theorem statement
theorem function_characterization :
  ∀ f : ℤ → ℤ, SatisfiesProperty f →
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l436_43652


namespace NUMINAMATH_CALUDE_stock_price_calculation_abc_stock_price_l436_43650

theorem stock_price_calculation (initial_price : ℝ) 
  (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let price_after_second_year := price_after_first_year * (1 - second_year_decrease)
  price_after_second_year

theorem abc_stock_price : 
  stock_price_calculation 100 0.5 0.3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_abc_stock_price_l436_43650


namespace NUMINAMATH_CALUDE_not_adjacent_2010_2011_l436_43617

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Checks if two natural numbers are consecutive -/
def are_consecutive (a b : ℕ) : Prop := b = a + 1

/-- Checks if a natural number is within a sequence of 100 consecutive numbers starting from start -/
def in_sequence (n start : ℕ) : Prop := start ≤ n ∧ n < start + 100

theorem not_adjacent_2010_2011 (start : ℕ) : 
  ¬(in_sequence 2010 start ∧ in_sequence 2011 start ∧
    (∀ (x y : ℕ), in_sequence x start → in_sequence y start →
      (digit_sum x < digit_sum y ∨ (digit_sum x = digit_sum y ∧ x < y)) →
      x < y) →
    are_consecutive 2010 2011) :=
sorry

end NUMINAMATH_CALUDE_not_adjacent_2010_2011_l436_43617


namespace NUMINAMATH_CALUDE_man_rowing_speed_l436_43665

/-- Given a man's downstream speed and speed in still water, calculate his upstream speed -/
theorem man_rowing_speed (downstream_speed still_water_speed : ℝ) 
  (h1 : downstream_speed = 31)
  (h2 : still_water_speed = 28) :
  still_water_speed - (downstream_speed - still_water_speed) = 25 := by
  sorry

end NUMINAMATH_CALUDE_man_rowing_speed_l436_43665


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l436_43699

theorem cube_volume_from_surface_area :
  ∀ s : ℝ,
  s > 0 →
  6 * s^2 = 864 →
  s^3 = 1728 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l436_43699


namespace NUMINAMATH_CALUDE_geometric_mean_of_3_and_12_l436_43644

theorem geometric_mean_of_3_and_12 : 
  ∃ (x : ℝ), x > 0 ∧ x^2 = 3 * 12 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_3_and_12_l436_43644


namespace NUMINAMATH_CALUDE_elberta_has_41_l436_43622

/-- The amount of money Granny Smith has -/
def granny_smith_amount : ℕ := 72

/-- The amount of money Anjou has -/
def anjou_amount : ℕ := granny_smith_amount / 4

/-- The amount of money Elberta has -/
def elberta_amount : ℕ := 2 * anjou_amount + 5

/-- Theorem stating that Elberta has $41 -/
theorem elberta_has_41 : elberta_amount = 41 := by
  sorry

end NUMINAMATH_CALUDE_elberta_has_41_l436_43622


namespace NUMINAMATH_CALUDE_range_of_a_l436_43632

def f (x : ℝ) : ℝ := -x^5 - 3*x^3 - 5*x + 3

theorem range_of_a (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l436_43632


namespace NUMINAMATH_CALUDE_a_range_l436_43611

-- Define the linear equation
def linear_equation (a x : ℝ) : ℝ := a * x + x + 4

-- Define the condition that the root is within [-2, 1]
def root_in_interval (a : ℝ) : Prop :=
  ∃ x, x ∈ Set.Icc (-2) 1 ∧ linear_equation a x = 0

-- State the theorem
theorem a_range (a : ℝ) : 
  root_in_interval a ↔ a ∈ Set.Ioi 1 ∪ Set.Iio (-5) :=
sorry

end NUMINAMATH_CALUDE_a_range_l436_43611


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l436_43659

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_third_term : a 3 = 12 / 5)
  (h_seventh_term : a 7 = 48) :
  a 5 = 12 / 5 :=
sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l436_43659


namespace NUMINAMATH_CALUDE_pond_and_field_dimensions_l436_43656

/-- Given a square field with a circular pond inside, this theorem proves
    the diameter of the pond and the side length of the field. -/
theorem pond_and_field_dimensions :
  ∀ (pond_diameter field_side : ℝ),
    pond_diameter > 0 →
    field_side > pond_diameter →
    (field_side^2 - (pond_diameter/2)^2 * 3) = 13.75 * 240 →
    field_side - pond_diameter = 40 →
    pond_diameter = 20 ∧ field_side = 60 := by
  sorry

end NUMINAMATH_CALUDE_pond_and_field_dimensions_l436_43656


namespace NUMINAMATH_CALUDE_complement_M_correct_l436_43675

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 4 ≤ 0}

-- Define the complement of M in U
def complement_M : Set ℝ := {x : ℝ | x < -2 ∨ x > 2}

-- Theorem statement
theorem complement_M_correct : 
  U \ M = complement_M := by sorry

end NUMINAMATH_CALUDE_complement_M_correct_l436_43675


namespace NUMINAMATH_CALUDE_average_age_decrease_l436_43600

theorem average_age_decrease (initial_size : ℕ) (replaced_age new_age : ℕ) : 
  initial_size = 10 → replaced_age = 42 → new_age = 12 → 
  (replaced_age - new_age) / initial_size = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_age_decrease_l436_43600


namespace NUMINAMATH_CALUDE_theater_popcorn_packages_l436_43634

/-- The number of popcorn buckets needed by the theater -/
def total_buckets : ℕ := 426

/-- The number of buckets in each package -/
def buckets_per_package : ℕ := 8

/-- The minimum number of packages required -/
def min_packages : ℕ := 54

theorem theater_popcorn_packages :
  min_packages = (total_buckets + buckets_per_package - 1) / buckets_per_package :=
by sorry

end NUMINAMATH_CALUDE_theater_popcorn_packages_l436_43634


namespace NUMINAMATH_CALUDE_angle_terminal_side_l436_43683

theorem angle_terminal_side (α : Real) (x : Real) :
  (∃ P : Real × Real, P = (x, 4) ∧ P.1 = x * Real.cos α ∧ P.2 = 4 * Real.sin α) →
  Real.sin α = 4/5 →
  x = 3 ∨ x = -3 := by
sorry

end NUMINAMATH_CALUDE_angle_terminal_side_l436_43683


namespace NUMINAMATH_CALUDE_zoo_ticket_price_l436_43669

theorem zoo_ticket_price (regular_price : ℝ) (discount_percentage : ℝ) (discounted_price : ℝ) : 
  regular_price = 15 →
  discount_percentage = 40 →
  discounted_price = regular_price * (1 - discount_percentage / 100) →
  discounted_price = 9 := by
sorry

end NUMINAMATH_CALUDE_zoo_ticket_price_l436_43669


namespace NUMINAMATH_CALUDE_total_students_is_17_l436_43679

/-- Represents the total number of students in a class with various sports preferences. -/
def total_students : ℕ :=
  let baseball_and_football := 7
  let only_baseball := 3
  let only_football := 4
  let basketball_as_well := 2
  let basketball_and_football_not_baseball := 1
  let all_three_sports := 2
  let no_sports := 5
  let only_basketball := basketball_as_well - basketball_and_football_not_baseball - all_three_sports

  (baseball_and_football - all_three_sports) + 
  only_baseball + 
  only_football + 
  basketball_and_football_not_baseball + 
  all_three_sports + 
  no_sports + 
  only_basketball

/-- Theorem stating that the total number of students in the class is 17. -/
theorem total_students_is_17 : total_students = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_17_l436_43679


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l436_43684

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2*a - 1 < x ∧ x < 2*a + 3}

-- Theorem 1: A ⊆ B iff a ∈ [-1/2, 0]
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ a ∈ Set.Icc (-1/2) 0 := by sorry

-- Theorem 2: A ∩ B = ∅ iff a ∈ (-∞, -2] ∪ [3/2, +∞)
theorem disjoint_condition (a : ℝ) : A ∩ B a = ∅ ↔ a ∈ Set.Iic (-2) ∪ Set.Ici (3/2) := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l436_43684


namespace NUMINAMATH_CALUDE_complex_equation_system_l436_43629

theorem complex_equation_system (p q r s t u : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0) (hu : u ≠ 0)
  (eq1 : p = (q + r) / (s - 3))
  (eq2 : q = (p + r) / (t - 3))
  (eq3 : r = (p + q) / (u - 3))
  (eq4 : s * t + s * u + t * u = 7)
  (eq5 : s + t + u = 4) :
  s * t * u = 6 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_system_l436_43629


namespace NUMINAMATH_CALUDE_angle_terminal_side_set_l436_43605

/-- 
Given an angle α whose terminal side, when rotated counterclockwise by 30°, 
coincides with the terminal side of 120°, the set of all angles β that have 
the same terminal side as α is {β | β = k × 360° + 90°, k ∈ ℤ}.
-/
theorem angle_terminal_side_set (α : Real) 
  (h : α + 30 = 120 + 360 * (⌊(α + 30 - 120) / 360⌋ : ℤ)) :
  {β : Real | ∃ k : ℤ, β = k * 360 + 90} = 
  {β : Real | ∃ k : ℤ, β = k * 360 + α} :=
by sorry


end NUMINAMATH_CALUDE_angle_terminal_side_set_l436_43605


namespace NUMINAMATH_CALUDE_power_calculation_l436_43645

theorem power_calculation : 8^15 / 64^7 * 16 = 512 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l436_43645


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l436_43623

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area is 2√3, angle B = π/3, and a² + c² = 3ac, then b = 4 -/
theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) :
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 3 →
  B = π / 3 →
  a^2 + c^2 = 3 * a * c →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l436_43623


namespace NUMINAMATH_CALUDE_smallest_divisible_by_18_and_24_l436_43692

theorem smallest_divisible_by_18_and_24 : 
  ∃ n : ℕ, (n > 0 ∧ n % 18 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, (m > 0 ∧ m % 18 = 0 ∧ m % 24 = 0) → n ≤ m) ∧ n = 72 :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_18_and_24_l436_43692


namespace NUMINAMATH_CALUDE_like_terms_sum_of_exponents_l436_43697

/-- Given two terms 5a^m * b^4 and -4a^3 * b^(n+2) are like terms, prove that m + n = 5 -/
theorem like_terms_sum_of_exponents (m n : ℕ) : 
  (∃ (a b : ℝ), 5 * a^m * b^4 = -4 * a^3 * b^(n+2)) → m + n = 5 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_of_exponents_l436_43697


namespace NUMINAMATH_CALUDE_intersection_point_m_value_l436_43630

theorem intersection_point_m_value (m : ℝ) :
  (∃ y : ℝ, -3 * (-6) + y = m ∧ 2 * (-6) + y = 28) →
  m = 58 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_m_value_l436_43630


namespace NUMINAMATH_CALUDE_tax_free_items_cost_l436_43636

/-- Calculates the cost of tax-free items given total spend, sales tax, and tax rate -/
theorem tax_free_items_cost 
  (total_spend : ℝ) 
  (sales_tax : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_spend = 25)
  (h2 : sales_tax = 0.30)
  (h3 : tax_rate = 0.05) :
  total_spend - sales_tax / tax_rate = 19 := by
  sorry

#check tax_free_items_cost

end NUMINAMATH_CALUDE_tax_free_items_cost_l436_43636


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l436_43660

theorem multiplication_addition_equality : 12 * 24 + 36 * 12 = 720 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l436_43660


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_l436_43627

/-- The diameter of the inscribed circle in a triangle with sides 11, 6, and 7 is √10 -/
theorem inscribed_circle_diameter (a b c : ℝ) (h1 : a = 11) (h2 : b = 6) (h3 : c = 7) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  2 * area / s = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_l436_43627


namespace NUMINAMATH_CALUDE_edward_money_theorem_l436_43674

def edward_money_problem (initial_money spent1 spent2 : ℕ) : Prop :=
  let total_spent := spent1 + spent2
  let remaining_money := initial_money - total_spent
  remaining_money = 17

theorem edward_money_theorem :
  edward_money_problem 34 9 8 := by
  sorry

end NUMINAMATH_CALUDE_edward_money_theorem_l436_43674


namespace NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l436_43664

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_times_sqrt_12_l436_43664


namespace NUMINAMATH_CALUDE_percentage_problem_l436_43682

theorem percentage_problem (x : ℝ) (p : ℝ) : 
  (0.5 * x = 200) → (p * x = 160) → p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l436_43682


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l436_43646

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Median AM in triangle ABC --/
def median_AM (t : Triangle) : ℝ := sorry

theorem triangle_ABC_properties (t : Triangle) 
  (h1 : t.a^2 - (t.b - t.c)^2 = (2 - Real.sqrt 3) * t.b * t.c)
  (h2 : Real.sin t.A * Real.sin t.B = (Real.cos (t.C / 2))^2)
  (h3 : median_AM t = Real.sqrt 7) :
  t.A = π / 6 ∧ t.B = π / 6 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ABC_properties_l436_43646


namespace NUMINAMATH_CALUDE_equation_result_l436_43658

theorem equation_result : (88320 : ℤ) + 1315 + 9211 - 1569 = 97277 := by
  sorry

end NUMINAMATH_CALUDE_equation_result_l436_43658


namespace NUMINAMATH_CALUDE_solve_linear_equation_l436_43616

theorem solve_linear_equation (x : ℝ) : 2*x + 3*x + 4*x = 12 + 9 + 6 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l436_43616


namespace NUMINAMATH_CALUDE_circles_intersect_l436_43694

def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

theorem circles_intersect : ∃ (x y : ℝ), circle_C1 x y ∧ circle_C2 x y := by
  sorry

end NUMINAMATH_CALUDE_circles_intersect_l436_43694


namespace NUMINAMATH_CALUDE_carly_grill_capacity_l436_43696

/-- The number of burgers Carly can fit on the grill at once -/
def burgers_on_grill (guests : ℕ) (cooking_time_per_burger : ℕ) (total_cooking_time : ℕ) : ℕ :=
  let total_burgers := guests / 2 * 2 + guests / 2 * 1
  total_burgers * cooking_time_per_burger / total_cooking_time

theorem carly_grill_capacity :
  burgers_on_grill 30 8 72 = 5 := by
  sorry

end NUMINAMATH_CALUDE_carly_grill_capacity_l436_43696


namespace NUMINAMATH_CALUDE_paul_bought_six_chocolate_boxes_l436_43637

/-- Represents the number of boxes of chocolate candy Paul bought. -/
def chocolate_boxes : ℕ := sorry

/-- Represents the number of boxes of caramel candy Paul bought. -/
def caramel_boxes : ℕ := 4

/-- Represents the number of pieces of candy in each box. -/
def pieces_per_box : ℕ := 9

/-- Represents the total number of candies Paul had. -/
def total_candies : ℕ := 90

/-- Theorem stating that Paul bought 6 boxes of chocolate candy. -/
theorem paul_bought_six_chocolate_boxes :
  chocolate_boxes = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_paul_bought_six_chocolate_boxes_l436_43637


namespace NUMINAMATH_CALUDE_language_knowledge_distribution_l436_43628

/-- Given the distribution of language knowledge among students, prove that
    among those who know both German and French, more than 90% know English. -/
theorem language_knowledge_distribution (a b c d : ℝ) 
    (h1 : a + b ≥ 0.9 * (a + b + c + d))
    (h2 : a + c ≥ 0.9 * (a + b + c + d))
    (h3 : a ≥ 0) (h4 : b ≥ 0) (h5 : c ≥ 0) (h6 : d ≥ 0) : 
    a ≥ 9 * d := by
  sorry


end NUMINAMATH_CALUDE_language_knowledge_distribution_l436_43628


namespace NUMINAMATH_CALUDE_senate_committee_arrangement_l436_43604

/-- The number of ways to arrange senators around a circular table. -/
def arrange_senators (num_democrats : ℕ) (num_republicans : ℕ) : ℕ :=
  if num_democrats = num_republicans ∧ num_democrats > 0 then
    (num_democrats.factorial) * ((num_democrats - 1).factorial)
  else
    0

/-- Theorem: The number of ways to arrange 6 Democrats and 6 Republicans
    around a circular table, with Democrats and Republicans alternating,
    is equal to 86,400. -/
theorem senate_committee_arrangement :
  arrange_senators 6 6 = 86400 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_arrangement_l436_43604


namespace NUMINAMATH_CALUDE_poverty_alleviation_rate_l436_43654

theorem poverty_alleviation_rate (initial_population final_population : ℕ) 
  (years : ℕ) (decrease_rate : ℝ) : 
  initial_population = 90000 →
  final_population = 10000 →
  years = 2 →
  final_population = initial_population * (1 - decrease_rate) ^ years →
  9 * (1 - decrease_rate) ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_poverty_alleviation_rate_l436_43654


namespace NUMINAMATH_CALUDE_toaster_customers_l436_43647

/-- Represents the inverse proportionality between customers and cost -/
def inverse_prop (k : ℝ) (p c : ℝ) : Prop := p * c = k

/-- Applies a discount to a given price -/
def apply_discount (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

theorem toaster_customers : 
  ∀ (k : ℝ),
  inverse_prop k 12 600 →
  (∃ (p : ℝ), 
    inverse_prop k p (apply_discount (2 * 400) 0.1) ∧ 
    p = 10) := by
sorry

end NUMINAMATH_CALUDE_toaster_customers_l436_43647


namespace NUMINAMATH_CALUDE_books_before_sale_l436_43648

theorem books_before_sale (books_bought : ℕ) (total_books : ℕ) 
  (h1 : books_bought = 56) 
  (h2 : total_books = 91) : 
  total_books - books_bought = 35 := by
  sorry

end NUMINAMATH_CALUDE_books_before_sale_l436_43648


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l436_43673

theorem complex_arithmetic_equation : 
  -1^4 + (4 - (3/8 + 1/6 - 3/4) * 24) / 5 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l436_43673


namespace NUMINAMATH_CALUDE_solve_refrigerator_problem_l436_43633

def refrigerator_problem (refrigerator_price : ℝ) : Prop :=
  let mobile_price : ℝ := 8000
  let refrigerator_sold : ℝ := refrigerator_price * 0.96
  let mobile_sold : ℝ := mobile_price * 1.1
  let total_bought : ℝ := refrigerator_price + mobile_price
  let total_sold : ℝ := refrigerator_sold + mobile_sold
  let profit : ℝ := 200
  total_sold = total_bought + profit

theorem solve_refrigerator_problem :
  ∃ (price : ℝ), refrigerator_problem price ∧ price = 15000 := by
  sorry

end NUMINAMATH_CALUDE_solve_refrigerator_problem_l436_43633


namespace NUMINAMATH_CALUDE_red_knights_magical_swords_fraction_l436_43668

/-- Represents the color of a knight -/
inductive KnightColor
  | Red
  | Blue
  | Green

/-- Represents the total number of knights -/
def totalKnights : ℕ := 40

/-- The fraction of knights that are red -/
def redFraction : ℚ := 3/8

/-- The fraction of knights that are blue -/
def blueFraction : ℚ := 1/4

/-- The fraction of knights that are green -/
def greenFraction : ℚ := 1 - redFraction - blueFraction

/-- The fraction of all knights that wield magical swords -/
def magicalSwordsFraction : ℚ := 1/5

/-- The ratio of red knights with magical swords to blue knights with magical swords -/
def redToBlueMagicalRatio : ℚ := 3/2

/-- The ratio of red knights with magical swords to green knights with magical swords -/
def redToGreenMagicalRatio : ℚ := 2

theorem red_knights_magical_swords_fraction :
  ∃ (redMagicalFraction : ℚ),
    redMagicalFraction = 48/175 ∧
    redMagicalFraction * redFraction * totalKnights +
    (redMagicalFraction / redToBlueMagicalRatio) * blueFraction * totalKnights +
    (redMagicalFraction / redToGreenMagicalRatio) * greenFraction * totalKnights =
    magicalSwordsFraction * totalKnights :=
by sorry

end NUMINAMATH_CALUDE_red_knights_magical_swords_fraction_l436_43668


namespace NUMINAMATH_CALUDE_max_intersections_circle_square_l436_43612

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A square in a plane -/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

/-- The number of intersection points between a circle and a square -/
def intersection_points (c : Circle) (s : Square) : ℕ :=
  sorry

/-- The maximum number of intersection points between any circle and any square -/
def max_intersection_points : ℕ := sorry

/-- Theorem: The maximum number of intersection points between a circle and a square is 8 -/
theorem max_intersections_circle_square : max_intersection_points = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_circle_square_l436_43612


namespace NUMINAMATH_CALUDE_system_equations_properties_l436_43638

theorem system_equations_properties (a b : ℝ) 
  (eq1 : 2 * a + b = 7) 
  (eq2 : a - b = 2) : 
  (b = 7 - 2 * a) ∧ 
  (a = b + 2) ∧ 
  (3 * a = 9) ∧ 
  (3 * b = 3) := by
  sorry

end NUMINAMATH_CALUDE_system_equations_properties_l436_43638


namespace NUMINAMATH_CALUDE_f_is_odd_l436_43643

def f (x : ℝ) : ℝ := x^(1/3)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

end NUMINAMATH_CALUDE_f_is_odd_l436_43643


namespace NUMINAMATH_CALUDE_inequality_proof_l436_43688

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l436_43688


namespace NUMINAMATH_CALUDE_student_calculation_l436_43615

theorem student_calculation (chosen_number : ℕ) : 
  chosen_number = 155 → 
  chosen_number * 2 - 200 = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l436_43615


namespace NUMINAMATH_CALUDE_cupcakes_remaining_l436_43666

def cupcake_problem (packages : ℕ) (cupcakes_per_package : ℕ) (eaten : ℕ) : Prop :=
  let total := packages * cupcakes_per_package
  let remaining := total - eaten
  remaining = 7

theorem cupcakes_remaining :
  cupcake_problem 3 4 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_remaining_l436_43666


namespace NUMINAMATH_CALUDE_visited_neither_country_l436_43649

theorem visited_neither_country (total : ℕ) (visited_iceland : ℕ) (visited_norway : ℕ) (visited_both : ℕ) :
  total = 90 →
  visited_iceland = 55 →
  visited_norway = 33 →
  visited_both = 51 →
  total - (visited_iceland + visited_norway - visited_both) = 53 := by
sorry

end NUMINAMATH_CALUDE_visited_neither_country_l436_43649


namespace NUMINAMATH_CALUDE_vector_operation_result_l436_43678

def a : ℝ × ℝ × ℝ := (3, 5, 1)
def b : ℝ × ℝ × ℝ := (2, 2, 3)
def c : ℝ × ℝ × ℝ := (4, -1, -3)

theorem vector_operation_result :
  (2 : ℝ) • a - (3 : ℝ) • b + (4 : ℝ) • c = (16, 0, -19) := by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l436_43678


namespace NUMINAMATH_CALUDE_candy_distribution_contradiction_l436_43641

theorem candy_distribution_contradiction (N : ℕ) : 
  (∃ (x : ℕ), N = 2 * x) →
  (∃ (y : ℕ), N = 3 * y) →
  (∃ (z : ℕ), N / 3 = 2 * z + 3) →
  False :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_contradiction_l436_43641


namespace NUMINAMATH_CALUDE_initial_toy_cost_l436_43690

theorem initial_toy_cost (total_toys : ℕ) (total_cost : ℕ) (teddy_bears : ℕ) (teddy_cost : ℕ) (initial_toys : ℕ) :
  total_toys = initial_toys + teddy_bears →
  total_cost = teddy_bears * teddy_cost + initial_toys * 10 →
  teddy_bears = 20 →
  teddy_cost = 15 →
  initial_toys = 28 →
  total_cost = 580 →
  10 = total_cost / total_toys - (teddy_bears * teddy_cost) / initial_toys :=
by sorry

end NUMINAMATH_CALUDE_initial_toy_cost_l436_43690


namespace NUMINAMATH_CALUDE_escalator_travel_time_l436_43613

/-- Proves that a person walking on a moving escalator takes 10 seconds to cover the entire length -/
theorem escalator_travel_time
  (escalator_speed : ℝ)
  (escalator_length : ℝ)
  (person_speed : ℝ)
  (h1 : escalator_speed = 11)
  (h2 : escalator_length = 140)
  (h3 : person_speed = 3) :
  escalator_length / (escalator_speed + person_speed) = 10 := by
  sorry


end NUMINAMATH_CALUDE_escalator_travel_time_l436_43613


namespace NUMINAMATH_CALUDE_sum_of_square_and_pentagon_angles_l436_43624

theorem sum_of_square_and_pentagon_angles : 
  let square_angle := 180 * (4 - 2) / 4
  let pentagon_angle := 180 * (5 - 2) / 5
  square_angle + pentagon_angle = 198 := by
sorry

end NUMINAMATH_CALUDE_sum_of_square_and_pentagon_angles_l436_43624


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_11_plus_1_divisible_by_13_l436_43667

theorem smallest_number_divisible_by_11_plus_1_divisible_by_13 :
  ∃ n : ℕ, n = 77 ∧
  (∀ m : ℕ, m < n → ¬(11 ∣ m ∧ 13 ∣ (m + 1))) ∧
  11 ∣ n ∧ 13 ∣ (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_11_plus_1_divisible_by_13_l436_43667


namespace NUMINAMATH_CALUDE_line_parameterization_l436_43606

/-- Given a line y = 2x - 3 parameterized as (x, y) = (-8, s) + t(l, -7),
    prove that s = -19 and l = -7/2 -/
theorem line_parameterization (s l : ℝ) : 
  (∀ (x y t : ℝ), y = 2*x - 3 ↔ (x, y) = (-8, s) + t • (l, -7)) →
  s = -19 ∧ l = -7/2 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l436_43606


namespace NUMINAMATH_CALUDE_place_value_difference_power_l436_43687

/-- Given a natural number, returns the count of a specific digit in it. -/
def countDigit (n : ℕ) (digit : ℕ) : ℕ := sorry

/-- Given a natural number, returns a list of place values for specific digits. -/
def getPlaceValues (n : ℕ) (digits : List ℕ) : List ℕ := sorry

/-- Calculates the sum of differences between consecutive place values. -/
def sumOfDifferences (placeValues : List ℕ) : ℕ := sorry

/-- The main theorem to prove. -/
theorem place_value_difference_power (n : ℕ) (h : n = 58219435) :
  let placeValues := getPlaceValues n [1, 5, 8]
  let diffSum := sumOfDifferences placeValues
  let numTwos := countDigit n 2
  diffSum ^ numTwos = 420950000 := by sorry

end NUMINAMATH_CALUDE_place_value_difference_power_l436_43687


namespace NUMINAMATH_CALUDE_square_difference_l436_43677

theorem square_difference (a b : ℝ) : a^2 - 2*a*b + b^2 = (a - b)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l436_43677


namespace NUMINAMATH_CALUDE_player_B_more_consistent_l436_43671

def player_A_scores : List ℕ := [9, 7, 8, 7, 8, 10, 7, 9, 8, 7]
def player_B_scores : List ℕ := [7, 8, 9, 8, 7, 8, 9, 8, 9, 7]

def mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def variance (scores : List ℕ) : ℚ :=
  let m := mean scores
  (scores.map (fun x => ((x : ℚ) - m) ^ 2)).sum / scores.length

theorem player_B_more_consistent :
  mean player_A_scores = mean player_B_scores ∧
  variance player_B_scores < variance player_A_scores := by
  sorry

#eval mean player_A_scores
#eval mean player_B_scores
#eval variance player_A_scores
#eval variance player_B_scores

end NUMINAMATH_CALUDE_player_B_more_consistent_l436_43671


namespace NUMINAMATH_CALUDE_tangent_and_sin_cos_product_l436_43639

theorem tangent_and_sin_cos_product (α : Real) 
  (h : Real.tan (π / 4 + α) = 3) : 
  Real.tan α = 1 / 2 ∧ Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tangent_and_sin_cos_product_l436_43639
