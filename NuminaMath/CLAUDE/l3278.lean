import Mathlib

namespace value_of_expression_l3278_327889

theorem value_of_expression (α : Real) (h : 4 * Real.sin α - 3 * Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin (2 * α)) = 25 / 64 := by
  sorry

end value_of_expression_l3278_327889


namespace opposite_sign_pairs_l3278_327864

theorem opposite_sign_pairs : 
  ¬((-2^3) * ((-2)^3) < 0) ∧
  ¬((|-4|) * (-(-4)) < 0) ∧
  ((-3^4) * ((-3)^4) < 0) ∧
  ¬((10^2) * (2^10) < 0) := by
sorry

end opposite_sign_pairs_l3278_327864


namespace log_range_incorrect_l3278_327814

-- Define the logarithm function
noncomputable def log (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

-- State the theorem
theorem log_range_incorrect (b : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : b > 1) 
  (h2 : y = log b x) 
  (h3 : Real.sqrt b < x) 
  (h4 : x < b) : 
  ¬ (0.5 < y ∧ y < 1.5) :=
sorry

end log_range_incorrect_l3278_327814


namespace factor_implies_m_value_l3278_327890

theorem factor_implies_m_value (m : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, x^2 - m*x - 40 = (x + 5) * k) → m = 13 := by
  sorry

end factor_implies_m_value_l3278_327890


namespace cubic_equation_value_l3278_327891

theorem cubic_equation_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2*a^2 + 2 = 3 := by
  sorry

end cubic_equation_value_l3278_327891


namespace mario_expected_doors_l3278_327878

/-- The expected number of doors Mario will pass before reaching Bowser's level -/
def expected_doors (d r : ℕ) : ℚ :=
  (d * (d^r - 1)) / (d - 1)

/-- Theorem stating the expected number of doors Mario will pass -/
theorem mario_expected_doors (d r : ℕ) (hd : d > 1) (hr : r > 0) :
  let E := expected_doors d r
  ∀ k : ℕ, k ≤ r → 
    (∃ Ek : ℚ, Ek = E ∧ 
      Ek = 1 + (d - 1) / d * E + 1 / d * expected_doors d (r - k)) :=
by sorry

end mario_expected_doors_l3278_327878


namespace all_rationals_same_color_l3278_327899

-- Define a color type
def Color := Nat

-- Define a coloring function
def coloring : ℚ → Color := sorry

-- Define the main theorem
theorem all_rationals_same_color (n : Nat) 
  (h : ∀ a b : ℚ, coloring a ≠ coloring b → 
       coloring ((a + b) / 2) ≠ coloring a ∧ 
       coloring ((a + b) / 2) ≠ coloring b) : 
  ∀ x y : ℚ, coloring x = coloring y := by sorry

end all_rationals_same_color_l3278_327899


namespace cubic_monotone_increasing_l3278_327833

/-- A cubic function f(x) = ax³ - x² + x - 5 is monotonically increasing on ℝ if and only if a ≥ 1/3 -/
theorem cubic_monotone_increasing (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => a * x^3 - x^2 + x - 5) (3 * a * x^2 - 2 * x + 1) x) →
  (∀ x y : ℝ, x < y → (a * x^3 - x^2 + x - 5) < (a * y^3 - y^2 + y - 5)) ↔
  a ≥ 1/3 := by
  sorry

end cubic_monotone_increasing_l3278_327833


namespace fruit_seller_apples_l3278_327802

theorem fruit_seller_apples : ∀ (original : ℕ),
  (original : ℝ) * 0.6 = 420 → original = 700 := by
  sorry

end fruit_seller_apples_l3278_327802


namespace intersection_nonempty_intersection_equals_A_l3278_327877

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x > 5}

-- Theorem 1
theorem intersection_nonempty (a : ℝ) : 
  (A a ∩ B).Nonempty ↔ a < -1 ∨ a > 2 := by sorry

-- Theorem 2
theorem intersection_equals_A (a : ℝ) : 
  A a ∩ B = A a ↔ a < -4 ∨ a > 5 := by sorry

end intersection_nonempty_intersection_equals_A_l3278_327877


namespace ratio_of_sum_to_difference_l3278_327887

theorem ratio_of_sum_to_difference (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (h : x + y = 7 * (x - y)) : x / y = 2 := by
  sorry

end ratio_of_sum_to_difference_l3278_327887


namespace horner_method_f_3_l3278_327807

def f (x : ℝ) : ℝ := x^5 - 2*x^3 + 3*x^2 - x + 1

def horner_v3 (x : ℝ) : ℝ := ((((x + 0)*x - 2)*x + 3)*x - 1)*x + 1

theorem horner_method_f_3 : horner_v3 3 = 24 := by sorry

end horner_method_f_3_l3278_327807


namespace furniture_assembly_time_l3278_327858

def chairs : ℕ := 2
def tables : ℕ := 2
def time_per_piece : ℕ := 8

def total_pieces : ℕ := chairs + tables

def total_time : ℕ := total_pieces * time_per_piece

theorem furniture_assembly_time : total_time = 32 := by
  sorry

end furniture_assembly_time_l3278_327858


namespace cut_rectangle_perimeter_example_l3278_327862

/-- The perimeter of a rectangle with squares cut from its corners -/
def cut_rectangle_perimeter (length width cut : ℝ) : ℝ :=
  2 * (length + width)

/-- Theorem: The perimeter of a 12x5 cm rectangle with 2x2 cm squares cut from each corner is 34 cm -/
theorem cut_rectangle_perimeter_example :
  cut_rectangle_perimeter 12 5 2 = 34 := by
  sorry

end cut_rectangle_perimeter_example_l3278_327862


namespace roots_sum_squared_plus_double_plus_other_l3278_327880

theorem roots_sum_squared_plus_double_plus_other (a b : ℝ) : 
  a^2 + a - 2023 = 0 → b^2 + b - 2023 = 0 → a^2 + 2*a + b = 2022 := by
  sorry

end roots_sum_squared_plus_double_plus_other_l3278_327880


namespace reciprocal_counterexample_l3278_327812

theorem reciprocal_counterexample : ∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x > y ∧ (1 / x) > (1 / y) := by
  sorry

end reciprocal_counterexample_l3278_327812


namespace house_cost_l3278_327879

theorem house_cost (total : ℝ) (interest_rate1 : ℝ) (interest_rate2 : ℝ) (total_interest : ℝ) 
  (h1 : total = 120000)
  (h2 : interest_rate1 = 0.04)
  (h3 : interest_rate2 = 0.05)
  (h4 : total_interest = 3920) :
  ∃ (house_cost : ℝ),
    house_cost = 36000 ∧
    (1/3 * (total - house_cost) * interest_rate1 + 2/3 * (total - house_cost) * interest_rate2 = total_interest) :=
by
  sorry

end house_cost_l3278_327879


namespace quadratic_roots_condition_l3278_327865

theorem quadratic_roots_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 - 2*m*x + 4 = 0 ∧ y^2 - 2*m*y + 4 = 0 ∧ x ≠ y ∧ x > 2 ∧ y < 2) → 
  m > 2 :=
sorry

end quadratic_roots_condition_l3278_327865


namespace sum_of_squares_of_roots_l3278_327895

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 9 * x₁ + 7 = 0) → 
  (2 * x₂^2 - 9 * x₂ + 7 = 0) → 
  x₁^2 + x₂^2 = 53/4 := by
sorry

end sum_of_squares_of_roots_l3278_327895


namespace water_height_in_aquarium_l3278_327868

/-- Proves that 10 litres of water in an aquarium with dimensions 50 cm length
and 20 cm breadth will rise to a height of 10 cm. -/
theorem water_height_in_aquarium (length : ℝ) (breadth : ℝ) (volume_litres : ℝ) :
  length = 50 →
  breadth = 20 →
  volume_litres = 10 →
  (volume_litres * 1000) / (length * breadth) = 10 := by
  sorry

end water_height_in_aquarium_l3278_327868


namespace sara_quarters_proof_l3278_327894

/-- The number of quarters Sara's dad gave her -/
def quarters_from_dad (initial_quarters final_quarters : ℕ) : ℕ :=
  final_quarters - initial_quarters

/-- Proof that Sara's dad gave her 49 quarters -/
theorem sara_quarters_proof (initial_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 21)
  (h2 : final_quarters = 70) :
  quarters_from_dad initial_quarters final_quarters = 49 := by
  sorry

end sara_quarters_proof_l3278_327894


namespace no_real_solution_condition_l3278_327892

theorem no_real_solution_condition (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, a^x ≠ x) ↔ a > Real.exp (1 / Real.exp 1) := by
  sorry

end no_real_solution_condition_l3278_327892


namespace fraction_multiplication_l3278_327843

theorem fraction_multiplication : (1 / 3 : ℚ) * (3 / 4 : ℚ) * (4 / 5 : ℚ) = 1 / 5 := by
  sorry

end fraction_multiplication_l3278_327843


namespace christina_total_driving_time_l3278_327806

-- Define the total journey distance
def total_distance : ℝ := 210

-- Define the speed limits for each segment
def speed_limit_1 : ℝ := 30
def speed_limit_2 : ℝ := 40
def speed_limit_3 : ℝ := 50
def speed_limit_4 : ℝ := 60

-- Define the distances covered in the second and third segments
def distance_2 : ℝ := 120
def distance_3 : ℝ := 50

-- Define the time spent in the second and third segments
def time_2 : ℝ := 3
def time_3 : ℝ := 1

-- Define Christina's driving time function
def christina_driving_time : ℝ := by sorry

-- Theorem statement
theorem christina_total_driving_time :
  christina_driving_time = 100 / 60 := by sorry

end christina_total_driving_time_l3278_327806


namespace sum_of_two_primes_24_l3278_327883

theorem sum_of_two_primes_24 : ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p + q = 24 := by
  sorry

end sum_of_two_primes_24_l3278_327883


namespace imaginary_part_of_z_l3278_327831

theorem imaginary_part_of_z (z : ℂ) (h : Complex.abs (z + 2 * Complex.I) = Complex.abs z) :
  z.im = -1 := by sorry

end imaginary_part_of_z_l3278_327831


namespace least_sum_of_exponential_equality_l3278_327852

theorem least_sum_of_exponential_equality (x y z : ℕ+) 
  (h : (2 : ℕ)^(x : ℕ) = (5 : ℕ)^(y : ℕ) ∧ (5 : ℕ)^(y : ℕ) = (8 : ℕ)^(z : ℕ)) : 
  (∀ a b c : ℕ+, (2 : ℕ)^(a : ℕ) = (5 : ℕ)^(b : ℕ) ∧ (5 : ℕ)^(b : ℕ) = (8 : ℕ)^(c : ℕ) → 
    (x : ℕ) + (y : ℕ) + (z : ℕ) ≤ (a : ℕ) + (b : ℕ) + (c : ℕ)) ∧
  (x : ℕ) + (y : ℕ) + (z : ℕ) = 33 :=
by sorry

end least_sum_of_exponential_equality_l3278_327852


namespace min_value_on_circle_l3278_327829

theorem min_value_on_circle (x y : ℝ) :
  x^2 + y^2 - 4*x - 6*y + 12 = 0 →
  ∃ (min_val : ℝ), min_val = 14 - 2 * Real.sqrt 13 ∧
    ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 6*y' + 12 = 0 →
      x'^2 + y'^2 ≥ min_val :=
by sorry

end min_value_on_circle_l3278_327829


namespace car_travel_time_ratio_l3278_327811

theorem car_travel_time_ratio : 
  let distance : ℝ := 504
  let original_time : ℝ := 6
  let new_speed : ℝ := 56
  let new_time := distance / new_speed
  new_time / original_time = 3 / 2 := by
sorry

end car_travel_time_ratio_l3278_327811


namespace find_b_l3278_327897

/-- Given the conditions, prove that b = -2 --/
theorem find_b (a : ℕ) (b : ℝ) : 
  (2 * (a.choose 2) - (a.choose 1 - 1) * 6 = 0) →  -- Condition 1
  (b ≠ 0) →                                        -- Condition 3
  (a.choose 1 * b = -12) →                         -- Condition 2 (simplified)
  b = -2 := by
  sorry

end find_b_l3278_327897


namespace rabbit_position_final_position_l3278_327809

theorem rabbit_position (n : ℕ) : 
  1 + n * (n + 1) / 2 = (n + 1) * (n + 2) / 2 := by sorry

theorem final_position : 
  (2020 + 1) * (2020 + 2) / 2 = 2041211 := by sorry

end rabbit_position_final_position_l3278_327809


namespace expression_simplification_l3278_327847

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ -1) :
  (a - (2*a - 1) / a) + (1 - a^2) / (a^2 + a) = (a^2 - 3*a + 2) / a := by
  sorry

end expression_simplification_l3278_327847


namespace video_game_cost_l3278_327804

/-- If two identical video games cost $50 in total, then seven of these video games will cost $175. -/
theorem video_game_cost (cost_of_two : ℝ) (h : cost_of_two = 50) :
  7 * (cost_of_two / 2) = 175 := by
  sorry

end video_game_cost_l3278_327804


namespace alex_sweaters_l3278_327826

/-- The number of shirts Alex has to wash -/
def num_shirts : ℕ := 18

/-- The number of pants Alex has to wash -/
def num_pants : ℕ := 12

/-- The number of jeans Alex has to wash -/
def num_jeans : ℕ := 13

/-- The maximum number of items the washing machine can wash per cycle -/
def items_per_cycle : ℕ := 15

/-- The duration of each washing cycle in minutes -/
def cycle_duration : ℕ := 45

/-- The total time needed to wash all clothes in minutes -/
def total_wash_time : ℕ := 180

/-- The theorem stating that Alex has 17 sweaters to wash -/
theorem alex_sweaters : 
  ∃ (num_sweaters : ℕ), 
    (num_shirts + num_pants + num_jeans + num_sweaters) = 
    (total_wash_time / cycle_duration * items_per_cycle) ∧ 
    num_sweaters = 17 := by
  sorry

end alex_sweaters_l3278_327826


namespace one_minus_repeating_third_equals_two_thirds_l3278_327875

def repeating_decimal_one_third : ℚ := 1/3

theorem one_minus_repeating_third_equals_two_thirds :
  1 - repeating_decimal_one_third = 2/3 := by sorry

end one_minus_repeating_third_equals_two_thirds_l3278_327875


namespace problem_statement_l3278_327801

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + x^2 + (a - 1) * x + 1

/-- The theorem statement -/
theorem problem_statement (a : ℝ) :
  (∀ x > 0, Real.exp x > x + 1) →
  (∀ x > 0, f a x ≤ x^4 + Real.exp x) →
  a ≤ Real.exp 1 - 1 := by
  sorry

end problem_statement_l3278_327801


namespace expression_value_l3278_327845

theorem expression_value :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (2 - 4 + 6 - 8 + 10 - 12 + 14) = 3 / 4 := by
sorry

end expression_value_l3278_327845


namespace sapling_growth_l3278_327818

/-- The height of a sapling after n years -/
def sapling_height (n : ℕ) : ℝ :=
  1.5 + 0.2 * n

/-- Theorem: The height of the sapling after n years is 1.5 + 0.2n meters -/
theorem sapling_growth (n : ℕ) :
  sapling_height n = 1.5 + 0.2 * n := by
  sorry

end sapling_growth_l3278_327818


namespace final_numbers_correct_l3278_327832

/-- The number of elements in the initial sequence -/
def n : ℕ := 2022

/-- The number of operations performed -/
def operations : ℕ := (n - 2) / 2

/-- The arithmetic mean operation on squares -/
def arithmetic_mean_operation (x : ℕ) : ℕ := x^2 + 1

/-- The final two numbers after applying the arithmetic mean operation -/
def final_numbers : Fin 2 → ℕ
| 0 => arithmetic_mean_operation 1011 + operations
| 1 => arithmetic_mean_operation 1012 + operations

theorem final_numbers_correct :
  final_numbers 0 = 1023131 ∧ final_numbers 1 = 1025154 := by sorry

end final_numbers_correct_l3278_327832


namespace matrix_commutation_result_l3278_327888

/-- Given two 2x2 matrices A and B, where A is fixed and B has variable entries,
    prove that if AB = BA and 4y ≠ z, then (x - w) / (z - 4y) = 0. -/
theorem matrix_commutation_result (x y z w : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![x, y; z, w]
  (A * B = B * A) → (4 * y ≠ z) → (x - w) / (z - 4 * y) = 0 := by
  sorry


end matrix_commutation_result_l3278_327888


namespace rectangle_perimeter_l3278_327815

theorem rectangle_perimeter (w : ℝ) (h1 : w > 0) :
  let l := 3 * w
  let d := 8 * Real.sqrt 10
  d^2 = l^2 + w^2 →
  2 * l + 2 * w = 64 := by
sorry

end rectangle_perimeter_l3278_327815


namespace solve_system_l3278_327808

theorem solve_system (x y : ℤ) (h1 : x + y = 260) (h2 : x - y = 200) : y = 30 := by
  sorry

end solve_system_l3278_327808


namespace even_digits_512_base5_l3278_327838

/-- Converts a natural number from base 10 to base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers --/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- The number of even digits in the base-5 representation of 512 is 3 --/
theorem even_digits_512_base5 : countEvenDigits (toBase5 512) = 3 := by
  sorry

end even_digits_512_base5_l3278_327838


namespace calculation_proofs_l3278_327881

theorem calculation_proofs :
  (∃ (x : ℝ), x = (1/2 * Real.sqrt 24 - 2 * Real.sqrt 2 * Real.sqrt 3) ∧ x = -Real.sqrt 6) ∧
  (∃ (y : ℝ), y = ((Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) + Real.sqrt 8 - Real.sqrt (9/2)) ∧ y = -1 + Real.sqrt 2 / 2) :=
by sorry

end calculation_proofs_l3278_327881


namespace expand_expression_l3278_327821

theorem expand_expression (x y : ℝ) : (2 * x - 5) * (3 * y + 15) = 6 * x * y + 30 * x - 15 * y - 75 := by
  sorry

end expand_expression_l3278_327821


namespace det_A_equals_two_l3278_327816

open Matrix

theorem det_A_equals_two (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A + 2 * A⁻¹ = 0) : 
  det A = 2 := by
  sorry

end det_A_equals_two_l3278_327816


namespace boy_squirrel_walnuts_l3278_327857

theorem boy_squirrel_walnuts (initial_walnuts : ℕ) (boy_gathered : ℕ) (girl_brought : ℕ) (girl_ate : ℕ) (final_walnuts : ℕ) :
  initial_walnuts = 12 →
  girl_brought = 5 →
  girl_ate = 2 →
  final_walnuts = 20 →
  final_walnuts = initial_walnuts + boy_gathered - 1 + girl_brought - girl_ate →
  boy_gathered = 6 := by
sorry

end boy_squirrel_walnuts_l3278_327857


namespace bridge_length_specific_bridge_length_l3278_327805

/-- The length of a bridge that a train can cross, given the train's length, speed, and crossing time. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Proof that a 500m train traveling at 42 km/h crosses a bridge of approximately 200.2m in 60 seconds. -/
theorem specific_bridge_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |bridge_length 500 42 60 - 200.2| < ε :=
sorry

end bridge_length_specific_bridge_length_l3278_327805


namespace even_sum_odd_vertices_l3278_327822

/-- Represents a country on the spherical map -/
structure Country where
  color : Fin 4  -- 0: red, 1: yellow, 2: blue, 3: green
  vertices : ℕ

/-- Represents the spherical map -/
structure SphericalMap where
  countries : List Country
  neighbor_relation : Country → Country → Prop

/-- The number of countries with odd vertices for a given color -/
def num_odd_vertices (m : SphericalMap) (c : Fin 4) : ℕ :=
  (m.countries.filter (λ country => country.color = c ∧ country.vertices % 2 = 1)).length

theorem even_sum_odd_vertices (m : SphericalMap) :
  (num_odd_vertices m 0 + num_odd_vertices m 2) % 2 = 0 :=
sorry

end even_sum_odd_vertices_l3278_327822


namespace cubic_inequality_l3278_327844

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 + 4*x < 0 ↔ x < 0 := by
  sorry

end cubic_inequality_l3278_327844


namespace cryptarithmetic_puzzle_solution_l3278_327848

theorem cryptarithmetic_puzzle_solution :
  ∃ (A B C D E F : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    A * B + C * D = 10 * E + F ∧
    B + C + D ≠ A ∧
    A = 2 * D ∧
    F = 8 :=
by sorry

end cryptarithmetic_puzzle_solution_l3278_327848


namespace arithmetic_sequence_tenth_term_l3278_327861

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_tenth_term
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + a 19 = -18) :
  a 10 = -9 :=
sorry

end arithmetic_sequence_tenth_term_l3278_327861


namespace non_negative_xy_l3278_327867

theorem non_negative_xy (x y : ℝ) :
  |x^2 + y^2 - 4*x - 4*y + 5| = |2*x + 2*y - 4| → x ≥ 0 ∧ y ≥ 0 := by
  sorry

end non_negative_xy_l3278_327867


namespace least_number_with_remainder_4_l3278_327872

def is_valid_divisor (n : ℕ) : Prop := n > 0 ∧ 252 % n = 0

theorem least_number_with_remainder_4 : 
  (∀ x : ℕ, is_valid_divisor x → 256 % x = 4) ∧ 
  (∀ n : ℕ, n < 256 → ∃ y : ℕ, is_valid_divisor y ∧ n % y ≠ 4) :=
sorry

end least_number_with_remainder_4_l3278_327872


namespace inequality_solution_l3278_327851

/-- The numerator of the inequality -/
def numerator (x : ℝ) : ℝ := |3*x^2 + 8*x - 3| + |3*x^4 + 2*x^3 - 10*x^2 + 30*x - 9|

/-- The denominator of the inequality -/
def denominator (x : ℝ) : ℝ := |x-2| - 2*x - 1

/-- The inequality function -/
def inequality (x : ℝ) : Prop := numerator x / denominator x ≤ 0

/-- The solution set of the inequality -/
def solution_set : Set ℝ := {x | x < 1/3 ∨ x > 1/3}

theorem inequality_solution : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set :=
sorry

end inequality_solution_l3278_327851


namespace andy_position_after_2021_moves_l3278_327817

-- Define the ant's position as a pair of integers
def Position := ℤ × ℤ

-- Define the direction as an enumeration
inductive Direction
| North
| East
| South
| West

-- Define the initial position and direction
def initial_position : Position := (10, -10)
def initial_direction : Direction := Direction.North

-- Define a function to calculate the movement distance for a given move number
def movement_distance (move_number : ℕ) : ℕ :=
  (move_number / 4 : ℕ) + 1

-- Define a function to update the direction after a right turn
def turn_right (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

-- Define a function to update the position based on direction and distance
def move (pos : Position) (dir : Direction) (dist : ℤ) : Position :=
  match dir with
  | Direction.North => (pos.1, pos.2 + dist)
  | Direction.East => (pos.1 + dist, pos.2)
  | Direction.South => (pos.1, pos.2 - dist)
  | Direction.West => (pos.1 - dist, pos.2)

-- Define a function to simulate the ant's movement for a given number of moves
def simulate_movement (num_moves : ℕ) : Position :=
  sorry -- Actual implementation would go here

-- State the theorem
theorem andy_position_after_2021_moves :
  simulate_movement 2021 = (10, 496) := by sorry

end andy_position_after_2021_moves_l3278_327817


namespace supplement_of_complement_of_35_l3278_327896

/-- The complement of an angle in degrees -/
def complement (α : ℝ) : ℝ := 90 - α

/-- The supplement of an angle in degrees -/
def supplement (α : ℝ) : ℝ := 180 - α

/-- The degree measure of the supplement of the complement of a 35-degree angle is 125 degrees -/
theorem supplement_of_complement_of_35 :
  supplement (complement 35) = 125 := by sorry

end supplement_of_complement_of_35_l3278_327896


namespace monotonic_increasing_interval_of_f_l3278_327874

-- Define the function f(x) = |x+2|
def f (x : ℝ) : ℝ := |x + 2|

-- State the theorem
theorem monotonic_increasing_interval_of_f :
  {x : ℝ | ∀ y, x ≤ y → f x ≤ f y} = {x : ℝ | x ≥ -2} := by sorry

end monotonic_increasing_interval_of_f_l3278_327874


namespace tangent_line_equation_l3278_327860

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (3*x + y - 7 = 0) :=
by sorry

end tangent_line_equation_l3278_327860


namespace infinite_product_of_a_l3278_327825

noncomputable def a : ℕ → ℚ
  | 0 => 1/3
  | n + 1 => 1 + (a n - 1)^3

theorem infinite_product_of_a : ∏' n, a n = 3/5 := by sorry

end infinite_product_of_a_l3278_327825


namespace taxi_average_speed_l3278_327884

/-- The average speed of a taxi that travels 100 kilometers in 1 hour and 15 minutes is 80 kilometers per hour. -/
theorem taxi_average_speed :
  let distance : ℝ := 100 -- distance in kilometers
  let time : ℝ := 1.25 -- time in hours (1 hour and 15 minutes = 1.25 hours)
  let average_speed := distance / time
  average_speed = 80 := by sorry

end taxi_average_speed_l3278_327884


namespace circle_radius_from_area_l3278_327871

theorem circle_radius_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 64 * π → r = 8 := by
  sorry

end circle_radius_from_area_l3278_327871


namespace triangular_grid_4_has_17_triangles_l3278_327850

/-- Represents a triangular grid with n rows -/
structure TriangularGrid (n : ℕ) where
  rows : Fin n → ℕ
  row_content : ∀ i : Fin n, rows i = i.val + 1

/-- Counts the number of triangles in a triangular grid -/
def count_triangles (grid : TriangularGrid n) : ℕ :=
  sorry

theorem triangular_grid_4_has_17_triangles :
  ∃ (grid : TriangularGrid 4), count_triangles grid = 17 :=
sorry

end triangular_grid_4_has_17_triangles_l3278_327850


namespace ab_max_and_inverse_sum_min_l3278_327876

theorem ab_max_and_inverse_sum_min (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + 4*b = 4) : 
  (∀ x y, x > 0 → y > 0 → x + 4*y = 4 → a*b ≥ x*y) ∧ 
  (∀ x y, x > 0 → y > 0 → x + 4*y = 4 → 1/a + 4/b ≤ 1/x + 4/y) ∧
  (a*b = 1) ∧ (1/a + 4/b = 25/4) :=
sorry

end ab_max_and_inverse_sum_min_l3278_327876


namespace derivative_even_implies_a_equals_three_l3278_327842

/-- Given a function f(x) = x³ + (a-3)x² + αx, prove that if its derivative f'(x) is an even function, then a = 3 -/
theorem derivative_even_implies_a_equals_three (a α : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^3 + (a - 3) * x^2 + α * x
  let f' : ℝ → ℝ := λ x ↦ deriv f x
  (∀ x, f' (-x) = f' x) → a = 3 := by
  sorry

end derivative_even_implies_a_equals_three_l3278_327842


namespace cube_root_equation_solution_l3278_327827

theorem cube_root_equation_solution :
  ∃! x : ℚ, Real.rpow (5 + x) (1/3 : ℝ) = 4/3 :=
by
  use -71/27
  sorry

end cube_root_equation_solution_l3278_327827


namespace odd_even_intersection_empty_l3278_327840

def odd_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
def even_integers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem odd_even_intersection_empty : odd_integers ∩ even_integers = ∅ := by
  sorry

end odd_even_intersection_empty_l3278_327840


namespace smallest_abc_cba_divisible_by_11_l3278_327837

/-- Represents a six-digit number in the form ABC,CBA -/
def AbcCba (a b c : Nat) : Nat :=
  100000 * a + 10000 * b + 1000 * c + 100 * c + 10 * b + a

theorem smallest_abc_cba_divisible_by_11 :
  ∀ a b c : Nat,
    a ≠ b ∧ b ≠ c ∧ a ≠ c →
    0 < a ∧ a < 10 ∧ b < 10 ∧ c < 10 →
    AbcCba a b c ≥ 123321 ∨ ¬(AbcCba a b c % 11 = 0) :=
by sorry

end smallest_abc_cba_divisible_by_11_l3278_327837


namespace unique_fraction_decomposition_l3278_327873

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (n m : ℕ), n ≠ m ∧ 2 / p = 1 / n + 1 / m ∧
  ((n = (p + 1) / 2 ∧ m = p * (p + 1) / 2) ∨
   (m = (p + 1) / 2 ∧ n = p * (p + 1) / 2)) :=
by sorry

end unique_fraction_decomposition_l3278_327873


namespace arithmetic_sequence_formula_l3278_327893

/-- An arithmetic sequence with first term a-1 and common difference 2 has the general formula a_n = a + 2n - 3 -/
theorem arithmetic_sequence_formula (a : ℝ) :
  let a_n := fun (n : ℕ) => a - 1 + 2 * (n - 1)
  ∀ n : ℕ, a_n n = a + 2 * n - 3 :=
by sorry

end arithmetic_sequence_formula_l3278_327893


namespace correct_answers_for_86_points_min_correct_for_first_prize_l3278_327813

/-- Represents a math competition with given parameters -/
structure MathCompetition where
  total_questions : ℕ
  full_score : ℕ
  correct_points : ℕ
  wrong_points : ℤ
  unanswered_points : ℕ

/-- Theorem for part (1) of the problem -/
theorem correct_answers_for_86_points (comp : MathCompetition)
    (h1 : comp.total_questions = 25)
    (h2 : comp.full_score = 100)
    (h3 : comp.correct_points = 4)
    (h4 : comp.wrong_points = -1)
    (h5 : comp.unanswered_points = 0)
    (h6 : ∃ (x : ℕ), x * comp.correct_points + (comp.total_questions - 1 - x) * comp.wrong_points = 86) :
    ∃ (x : ℕ), x = 22 ∧ x * comp.correct_points + (comp.total_questions - 1 - x) * comp.wrong_points = 86 :=
  sorry

/-- Theorem for part (2) of the problem -/
theorem min_correct_for_first_prize (comp : MathCompetition)
    (h1 : comp.total_questions = 25)
    (h2 : comp.full_score = 100)
    (h3 : comp.correct_points = 4)
    (h4 : comp.wrong_points = -1)
    (h5 : comp.unanswered_points = 0) :
    ∃ (x : ℕ), x ≥ 23 ∧ ∀ (y : ℕ), y * comp.correct_points + (comp.total_questions - y) * comp.wrong_points ≥ 90 → y ≥ x :=
  sorry

end correct_answers_for_86_points_min_correct_for_first_prize_l3278_327813


namespace rectangular_solid_diagonal_l3278_327841

/-- The length of the diagonal of a rectangular solid with edges of length 2, 3, and 4 is √29. -/
theorem rectangular_solid_diagonal : 
  let a : ℝ := 2
  let b : ℝ := 3
  let c : ℝ := 4
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 :=
by sorry

end rectangular_solid_diagonal_l3278_327841


namespace unique_solution_l3278_327849

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (a + b) - f (a * b) = f a * f b - 1

/-- The theorem stating that the only function satisfying the equation is f(n) = n + 1 -/
theorem unique_solution (f : ℤ → ℤ) (h : SatisfiesEquation f) :
  ∀ n : ℤ, f n = n + 1 := by
  sorry

end unique_solution_l3278_327849


namespace pattern_continuation_l3278_327853

theorem pattern_continuation (h1 : 1 = 6) (h2 : 2 = 12) (h3 : 3 = 18) (h4 : 4 = 24) (h5 : 5 = 30) : 6 = 36 := by
  sorry

end pattern_continuation_l3278_327853


namespace quadratic_inequality_implies_range_l3278_327810

theorem quadratic_inequality_implies_range (x : ℝ) :
  x^2 - 5*x + 4 < 0 → 10 < x^2 + 4*x + 5 ∧ x^2 + 4*x + 5 < 37 := by
  sorry

end quadratic_inequality_implies_range_l3278_327810


namespace divisibility_of_fifth_powers_l3278_327800

theorem divisibility_of_fifth_powers (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end divisibility_of_fifth_powers_l3278_327800


namespace return_trip_time_l3278_327886

/-- Calculates the return trip time given the outbound trip details -/
theorem return_trip_time (outbound_time : ℝ) (outbound_speed : ℝ) (speed_increase : ℝ) : 
  outbound_time = 6 →
  outbound_speed = 60 →
  speed_increase = 12 →
  (outbound_time * outbound_speed) / (outbound_speed + speed_increase) = 5 := by
  sorry

end return_trip_time_l3278_327886


namespace division_problem_l3278_327839

theorem division_problem (divisor quotient remainder : ℕ) 
  (h1 : divisor = 30)
  (h2 : quotient = 9)
  (h3 : remainder = 1) :
  divisor * quotient + remainder = 271 :=
by sorry

end division_problem_l3278_327839


namespace eliminate_cycles_in_complete_digraph_l3278_327830

/-- A complete directed graph with 32 vertices -/
def CompleteDigraph : Type := Fin 32 → Fin 32 → Prop

/-- The property that a graph contains no directed cycles -/
def NoCycles (g : CompleteDigraph) : Prop := sorry

/-- A step that changes the direction of a single edge -/
def Step (g₁ g₂ : CompleteDigraph) : Prop := sorry

/-- The theorem stating that it's possible to eliminate all cycles in at most 208 steps -/
theorem eliminate_cycles_in_complete_digraph :
  ∃ (sequence : Fin 209 → CompleteDigraph),
    (∀ i : Fin 208, Step (sequence i) (sequence (i + 1))) ∧
    NoCycles (sequence 208) :=
  sorry

end eliminate_cycles_in_complete_digraph_l3278_327830


namespace cube_volume_from_surface_area_l3278_327869

/-- Given a cube with surface area 150 cm², prove its volume is 125 cm³ -/
theorem cube_volume_from_surface_area :
  ∀ (side_length : ℝ),
  (6 : ℝ) * side_length^2 = 150 →
  side_length^3 = 125 :=
by
  sorry

end cube_volume_from_surface_area_l3278_327869


namespace polynomial_factorization_l3278_327836

theorem polynomial_factorization (a b m n : ℝ) : 
  (3 * a^2 - 6 * a * b + 3 * b^2 = 3 * (a - b)^2) ∧ 
  (4 * m^2 - 9 * n^2 = (2 * m - 3 * n) * (2 * m + 3 * n)) := by
  sorry

end polynomial_factorization_l3278_327836


namespace b_income_less_than_others_l3278_327819

structure Income where
  c : ℝ
  a : ℝ
  b_salary : ℝ
  b_commission : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

def Income.b_total (i : Income) : ℝ := i.b_salary + i.b_commission

def Income.others_total (i : Income) : ℝ := i.a + i.c + i.d + i.e + i.f

def valid_income (i : Income) : Prop :=
  i.a = i.c * 1.2 ∧
  i.b_salary = i.a * 1.25 ∧
  i.b_commission = (i.a + i.c) * 0.05 ∧
  i.d = i.b_total * 0.85 ∧
  i.e = i.c * 1.1 ∧
  i.f = (i.b_total + i.e) / 2

theorem b_income_less_than_others (i : Income) (h : valid_income i) :
  i.b_total < i.others_total ∧ i.b_commission = i.c * 0.11 :=
sorry

end b_income_less_than_others_l3278_327819


namespace b_completion_time_l3278_327859

/-- The number of days A needs to complete the entire work -/
def a_total_days : ℚ := 15

/-- The number of days A actually works -/
def a_worked_days : ℚ := 5

/-- The number of days B needs to complete the entire work -/
def b_total_days : ℚ := 9/2

/-- The fraction of work completed by A -/
def a_work_completed : ℚ := a_worked_days / a_total_days

/-- The fraction of work B needs to complete -/
def b_work_to_complete : ℚ := 1 - a_work_completed

/-- The fraction of work B completes per day -/
def b_work_per_day : ℚ := 1 / b_total_days

/-- The number of days B needs to complete the remaining work -/
def b_days_needed : ℚ := b_work_to_complete / b_work_per_day

theorem b_completion_time : b_days_needed = 3 := by
  sorry

end b_completion_time_l3278_327859


namespace unique_solution_for_k_l3278_327870

/-- The equation has exactly one solution when k = -3/4 -/
theorem unique_solution_for_k (k : ℝ) : 
  (∃! x, (x + 3) / (k * x - 2) = x) ↔ k = -3/4 := by
  sorry

end unique_solution_for_k_l3278_327870


namespace sum_medians_gt_four_times_circumradius_l3278_327866

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define properties of a triangle
def Triangle.isNonObtuse (t : Triangle) : Prop := sorry

def Triangle.medians (t : Triangle) : ℝ × ℝ × ℝ := sorry

def Triangle.circumradius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem sum_medians_gt_four_times_circumradius 
  (t : Triangle) (h : t.isNonObtuse) : 
  let (m₁, m₂, m₃) := t.medians
  m₁ + m₂ + m₃ > 4 * t.circumradius :=
by
  sorry

end sum_medians_gt_four_times_circumradius_l3278_327866


namespace max_distance_unit_circle_l3278_327820

/-- The maximum distance between any two points on the unit circle is 2 -/
theorem max_distance_unit_circle : 
  ∀ (α β : ℝ), 
  let P := (Real.cos α, Real.sin α)
  let Q := (Real.cos β, Real.sin β)
  ∃ (maxDist : ℝ), maxDist = 2 ∧ 
    ∀ (α' β' : ℝ), 
    let P' := (Real.cos α', Real.sin α')
    let Q' := (Real.cos β', Real.sin β')
    Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≤ maxDist :=
by sorry

end max_distance_unit_circle_l3278_327820


namespace triangle_inequality_l3278_327882

-- Define a structure for a triangle
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z
  triangle_ineq : x + y > z ∧ y + z > x ∧ z + x > y

-- State the theorem
theorem triangle_inequality (t : Triangle) :
  |t.x^2 * (t.y - t.z) + t.y^2 * (t.z - t.x) + t.z^2 * (t.x - t.y)| < t.x * t.y * t.z :=
by sorry

end triangle_inequality_l3278_327882


namespace fraction_zero_implies_a_negative_two_l3278_327803

theorem fraction_zero_implies_a_negative_two (a : ℝ) : 
  (a^2 - 4) / (a - 2) = 0 → a = -2 := by
  sorry

end fraction_zero_implies_a_negative_two_l3278_327803


namespace total_travel_time_travel_time_calculation_l3278_327855

/-- Calculates the total travel time between two towns given specific conditions -/
theorem total_travel_time (total_distance : ℝ) (initial_fraction : ℝ) (lunch_time : ℝ) 
  (second_fraction : ℝ) (pit_stop_time : ℝ) (speed_increase : ℝ) : ℝ :=
  let initial_distance := initial_fraction * total_distance
  let initial_speed := initial_distance
  let remaining_distance := total_distance - initial_distance
  let second_distance := second_fraction * remaining_distance
  let final_distance := remaining_distance - second_distance
  let final_speed := initial_speed + speed_increase
  initial_fraction + lunch_time + (second_distance / initial_speed) + 
  pit_stop_time + (final_distance / final_speed)

/-- The total travel time between the two towns is 5.25 hours -/
theorem travel_time_calculation : 
  total_travel_time 200 (1/4) 1 (1/2) (1/2) 10 = 5.25 := by
  sorry

end total_travel_time_travel_time_calculation_l3278_327855


namespace four_students_seven_seats_l3278_327824

/-- The number of ways to arrange students in seats with adjacent empty seats -/
def seating_arrangements (total_seats : ℕ) (students : ℕ) (adjacent_empty : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 480 ways to arrange 4 students in 7 seats with 2 adjacent empty seats -/
theorem four_students_seven_seats : seating_arrangements 7 4 2 = 480 := by
  sorry

end four_students_seven_seats_l3278_327824


namespace certain_number_existence_and_uniqueness_l3278_327885

theorem certain_number_existence_and_uniqueness :
  ∃! x : ℚ, x / 3 + x + 3 = 63 :=
by sorry

end certain_number_existence_and_uniqueness_l3278_327885


namespace fraction_addition_l3278_327835

theorem fraction_addition : (2 : ℚ) / 3 + (1 : ℚ) / 6 = (5 : ℚ) / 6 := by sorry

end fraction_addition_l3278_327835


namespace circle_radius_is_five_thirds_l3278_327863

/-- An isosceles triangle with a circle constructed on its base -/
structure IsoscelesTriangleWithCircle where
  /-- The base length of the isosceles triangle -/
  base : ℝ
  /-- The height of the isosceles triangle -/
  height : ℝ
  /-- The radius of the circle constructed on the base -/
  radius : ℝ

/-- The radius of the circle in an isosceles triangle with given base and height -/
def circleRadius (triangle : IsoscelesTriangleWithCircle) : ℝ :=
  triangle.radius

/-- Theorem: The radius of the circle is 5/3 given the specified conditions -/
theorem circle_radius_is_five_thirds (triangle : IsoscelesTriangleWithCircle)
    (h1 : triangle.base = 8)
    (h2 : triangle.height = 3) :
    circleRadius triangle = 5/3 := by
  sorry

end circle_radius_is_five_thirds_l3278_327863


namespace sum_is_linear_l3278_327898

/-- The original parabola function -/
def original_parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

/-- The function f(x) derived from the original parabola -/
def f (a h k x : ℝ) : ℝ := -a * (x - h - 3)^2 - k

/-- The function g(x) derived from the original parabola -/
def g (a h k x : ℝ) : ℝ := a * (x - h + 7)^2 + k

/-- The sum of f(x) and g(x) -/
def f_plus_g (a h k x : ℝ) : ℝ := f a h k x + g a h k x

theorem sum_is_linear (a h k : ℝ) (ha : a ≠ 0) :
  ∃ m b : ℝ, (∀ x : ℝ, f_plus_g a h k x = m * x + b) ∧ m ≠ 0 :=
sorry

end sum_is_linear_l3278_327898


namespace cricket_team_average_age_l3278_327828

/-- Represents a cricket team -/
structure CricketTeam where
  numPlayers : Nat
  captainAge : Nat
  captainBattingAvg : Nat
  wicketKeeperAge : Nat
  wicketKeeperBattingAvg : Nat
  youngestPlayerBattingAvg : Nat

/-- Calculate the average age of the team -/
def averageTeamAge (team : CricketTeam) : Rat :=
  sorry

theorem cricket_team_average_age 
  (team : CricketTeam)
  (h1 : team.numPlayers = 11)
  (h2 : team.captainAge = 25)
  (h3 : team.captainBattingAvg = 45)
  (h4 : team.wicketKeeperAge = team.captainAge + 5)
  (h5 : team.wicketKeeperBattingAvg = 35)
  (h6 : team.youngestPlayerBattingAvg = 42)
  (h7 : ∀ (remainingPlayersAvgAge : Rat),
        remainingPlayersAvgAge = (averageTeamAge team - 1) ∧
        (team.captainAge + team.wicketKeeperAge + remainingPlayersAvgAge * (team.numPlayers - 2)) / team.numPlayers = averageTeamAge team)
  (h8 : ∃ (youngestPlayerAge : Nat),
        youngestPlayerAge ≤ team.wicketKeeperAge - 15 ∧
        youngestPlayerAge > 0) :
  averageTeamAge team = 23 :=
sorry

end cricket_team_average_age_l3278_327828


namespace three_hundred_thousand_squared_minus_million_l3278_327823

theorem three_hundred_thousand_squared_minus_million : (300000 * 300000) - 1000000 = 89990000000 := by
  sorry

end three_hundred_thousand_squared_minus_million_l3278_327823


namespace problem_G6_1_l3278_327846

theorem problem_G6_1 (p : ℝ) : 
  p = (21^3 - 11^3) / (21^2 + 21*11 + 11^2) → p = 10 := by
  sorry


end problem_G6_1_l3278_327846


namespace max_fold_length_less_than_eight_l3278_327834

theorem max_fold_length_less_than_eight (length width : ℝ) 
  (h_length : length = 6) (h_width : width = 5) : 
  Real.sqrt (length^2 + width^2) < 8 := by
  sorry

end max_fold_length_less_than_eight_l3278_327834


namespace two_copy_machines_output_l3278_327856

/-- Calculates the total number of copies made by two copy machines in a given time -/
def total_copies (rate1 rate2 time : ℕ) : ℕ :=
  rate1 * time + rate2 * time

/-- Proves that two copy machines with given rates produce 3300 copies in 30 minutes -/
theorem two_copy_machines_output : total_copies 35 75 30 = 3300 := by
  sorry

end two_copy_machines_output_l3278_327856


namespace no_solution_for_floor_equation_l3278_327854

theorem no_solution_for_floor_equation :
  ∀ x : ℝ, ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345 := by
  sorry

end no_solution_for_floor_equation_l3278_327854
