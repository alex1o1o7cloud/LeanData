import Mathlib

namespace sum_product_ratio_l1078_107863

theorem sum_product_ratio (x y z : ℝ) (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z) (hsum : x + y + z = 1) :
  (x*y + y*z + z*x) / (x^2 + y^2 + z^2) = (1 - (x^2 + y^2 + z^2)) / (2*(x^2 + y^2 + z^2)) := by
  sorry

end sum_product_ratio_l1078_107863


namespace f_zero_points_iff_k_range_l1078_107883

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then k * x + 2 else (1/2) ^ x

def has_three_zero_points (k : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ x, f k (f k x) - 3/2 = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)

theorem f_zero_points_iff_k_range :
  ∀ k, has_three_zero_points k ↔ -1/2 < k ∧ k ≤ -1/4 :=
sorry

end f_zero_points_iff_k_range_l1078_107883


namespace first_year_after_2021_with_sum_15_l1078_107864

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_first_year_after_2021_with_sum_15 (year : ℕ) : Prop :=
  year > 2021 ∧
  sum_of_digits year = 15 ∧
  ∀ y : ℕ, 2021 < y ∧ y < year → sum_of_digits y ≠ 15

theorem first_year_after_2021_with_sum_15 :
  is_first_year_after_2021_with_sum_15 2049 := by
  sorry

end first_year_after_2021_with_sum_15_l1078_107864


namespace grade12_sample_size_l1078_107845

/-- Represents the number of grade 12 students in a stratified sample -/
def grade12InSample (totalStudents gradeStudents sampleSize : ℕ) : ℚ :=
  (sampleSize : ℚ) * (gradeStudents : ℚ) / (totalStudents : ℚ)

/-- Theorem: The number of grade 12 students in the sample is 140 -/
theorem grade12_sample_size :
  grade12InSample 2000 700 400 = 140 := by sorry

end grade12_sample_size_l1078_107845


namespace tape_circle_length_l1078_107859

/-- The total length of a circle formed by overlapping tape pieces -/
def circle_length (num_pieces : ℕ) (piece_length : ℝ) (overlap : ℝ) : ℝ :=
  num_pieces * (piece_length - overlap)

/-- Theorem stating the total length of the circle-shaped colored tapes -/
theorem tape_circle_length :
  circle_length 16 10.4 3.5 = 110.4 := by
  sorry

end tape_circle_length_l1078_107859


namespace largest_x_value_largest_x_exists_l1078_107815

theorem largest_x_value (x y : ℝ) : 
  (|x - 3| = 15 ∧ x + y = 10) → x ≤ 18 := by
  sorry

theorem largest_x_exists : 
  ∃ x y : ℝ, |x - 3| = 15 ∧ x + y = 10 ∧ x = 18 := by
  sorry

end largest_x_value_largest_x_exists_l1078_107815


namespace fraction_simplification_l1078_107836

theorem fraction_simplification : 
  let numerator := (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400)
  let denominator := (6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400)
  ∀ x : ℕ, x^4 + 400 = (x^2 - 10*x + 20) * (x^2 + 10*x + 20) →
  numerator / denominator = 995 := by
  sorry

end fraction_simplification_l1078_107836


namespace complementary_event_equivalence_l1078_107827

/-- The number of products in the sample -/
def sample_size : ℕ := 10

/-- Event A: at least 2 defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- Complementary event of A -/
def comp_A (defective : ℕ) : Prop := ¬(event_A defective)

/-- At most 1 defective product -/
def at_most_one_defective (defective : ℕ) : Prop := defective ≤ 1

/-- At least 2 non-defective products -/
def at_least_two_non_defective (defective : ℕ) : Prop := sample_size - defective ≥ 2

theorem complementary_event_equivalence :
  ∀ defective : ℕ, defective ≤ sample_size →
    (comp_A defective ↔ at_most_one_defective defective) ∧
    (comp_A defective ↔ at_least_two_non_defective defective) :=
by sorry

end complementary_event_equivalence_l1078_107827


namespace orange_crayon_boxes_l1078_107884

theorem orange_crayon_boxes (total_crayons : ℕ) 
  (orange_per_box blue_boxes blue_per_box red_boxes red_per_box : ℕ) : 
  total_crayons = 94 →
  orange_per_box = 8 →
  blue_boxes = 7 →
  blue_per_box = 5 →
  red_boxes = 1 →
  red_per_box = 11 →
  (total_crayons - (blue_boxes * blue_per_box + red_boxes * red_per_box)) / orange_per_box = 6 :=
by
  sorry

#check orange_crayon_boxes

end orange_crayon_boxes_l1078_107884


namespace opposite_zero_l1078_107807

/-- The opposite of a real number -/
def opposite (x : ℝ) : ℝ := -x

/-- The property that defines an opposite -/
axiom opposite_def (x : ℝ) : x + opposite x = 0

/-- Theorem: The opposite of 0 is 0 -/
theorem opposite_zero : opposite 0 = 0 := by
  sorry

end opposite_zero_l1078_107807


namespace quadrilateral_problem_l1078_107872

/-- Prove that for a quadrilateral PQRS with specific vertex coordinates,
    consecutive integer side lengths, and an area of 50,
    the product of the odd integer scale factor and the sum of side lengths is 5. -/
theorem quadrilateral_problem (a b k : ℤ) : 
  a > b ∧ b > 0 ∧  -- a and b are consecutive integers with a > b > 0
  ∃ n : ℤ, a = b + 1 ∧  -- a and b are consecutive integers
  ∃ m : ℤ, k = 2 * m + 1 ∧  -- k is an odd integer
  2 * k^2 * (a - b) * (a + b) = 50 →  -- area of PQRS is 50
  k * (a + b) = 5 := by
sorry

end quadrilateral_problem_l1078_107872


namespace average_difference_l1078_107873

theorem average_difference (x : ℝ) : (10 + 60 + x) / 3 = (20 + 40 + 60) / 3 - 5 ↔ x = 35 := by
  sorry

end average_difference_l1078_107873


namespace unique_triplet_solution_l1078_107887

theorem unique_triplet_solution :
  ∀ x y p : ℕ+,
  p.Prime →
  (x.val * y.val^3 : ℚ) / (x.val + y.val) = p.val →
  x = 14 ∧ y = 2 ∧ p = 7 :=
by sorry

end unique_triplet_solution_l1078_107887


namespace roots_of_quadratic_sum_l1078_107842

theorem roots_of_quadratic_sum (α β : ℝ) : 
  (α^2 - 3*α - 4 = 0) → (β^2 - 3*β - 4 = 0) → 4*α^3 + 9*β^2 = -72 := by
  sorry

end roots_of_quadratic_sum_l1078_107842


namespace fraction_meaningful_l1078_107888

theorem fraction_meaningful (x : ℝ) : 
  IsRegular (4 / (x + 2)) ↔ x ≠ -2 :=
sorry

end fraction_meaningful_l1078_107888


namespace sqrt_simplification_l1078_107882

theorem sqrt_simplification (x : ℝ) :
  1 + x ≥ 0 → -1 - x ≥ 0 → Real.sqrt (1 + x) - Real.sqrt (-1 - x) = 0 := by
  sorry

end sqrt_simplification_l1078_107882


namespace repair_cost_is_288_l1078_107816

/-- The amount spent on repairs for a scooter, given the purchase price, selling price, and gain percentage. -/
def repair_cost (purchase_price selling_price : ℚ) (gain_percentage : ℚ) : ℚ :=
  selling_price * (1 - gain_percentage / 100) - purchase_price

/-- Theorem stating that the repair cost is $288 given the specific conditions. -/
theorem repair_cost_is_288 :
  repair_cost 900 1320 10 = 288 := by
  sorry

end repair_cost_is_288_l1078_107816


namespace sine_cosine_equation_l1078_107817

theorem sine_cosine_equation (x y : ℝ) 
  (h : (Real.sin x ^ 2 - Real.cos x ^ 2 + Real.cos x ^ 2 * Real.cos y ^ 2 - Real.sin x ^ 2 * Real.sin y ^ 2) / Real.sin (x + y) = 1) :
  ∃ k : ℤ, x - y = 2 * k * Real.pi + Real.pi / 2 := by
sorry

end sine_cosine_equation_l1078_107817


namespace solution_set_of_inequality_l1078_107820

open Set

theorem solution_set_of_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_deriv : ∀ x, f x > deriv f x) (h_init : f 0 = 2) :
  {x : ℝ | f x < 2 * Real.exp x} = {x : ℝ | x > 0} := by
sorry

end solution_set_of_inequality_l1078_107820


namespace bedroom_renovation_time_l1078_107868

theorem bedroom_renovation_time :
  ∀ (bedroom_time : ℝ),
    bedroom_time > 0 →
    (3 * bedroom_time) +                                -- Time for 3 bedrooms
    (1.5 * bedroom_time) +                              -- Time for kitchen (50% longer than a bedroom)
    (2 * ((3 * bedroom_time) + (1.5 * bedroom_time))) = -- Time for living room (twice as everything else)
    54 →                                                -- Total renovation time
    bedroom_time = 4 := by
  sorry

end bedroom_renovation_time_l1078_107868


namespace common_chord_length_l1078_107822

theorem common_chord_length (r : ℝ) (d : ℝ) (h1 : r = 12) (h2 : d = 8) :
  let chord_length := 2 * Real.sqrt (r^2 - (d/2)^2)
  chord_length = 16 * Real.sqrt 2 := by
  sorry

end common_chord_length_l1078_107822


namespace tape_division_l1078_107821

theorem tape_division (total_tape : ℚ) (num_packages : ℕ) :
  total_tape = 7 / 12 ∧ num_packages = 5 →
  total_tape / num_packages = 7 / 60 := by
  sorry

end tape_division_l1078_107821


namespace remainder_three_to_seventeen_mod_five_l1078_107852

theorem remainder_three_to_seventeen_mod_five : 3^17 % 5 = 3 := by
  sorry

end remainder_three_to_seventeen_mod_five_l1078_107852


namespace notebook_cost_l1078_107814

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buying_students : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
  total_students = 50 ∧
  total_cost = 2739 ∧
  buying_students > total_students / 2 ∧
  notebooks_per_student % 2 = 1 ∧
  notebooks_per_student > 1 ∧
  cost_per_notebook > notebooks_per_student ∧
  buying_students * notebooks_per_student * cost_per_notebook = total_cost ∧
  cost_per_notebook = 7 := by
  sorry

end notebook_cost_l1078_107814


namespace ellipse_equation_l1078_107885

/-- Given an ellipse with focal distance 4 passing through (√2, √3), prove its equation is x²/8 + y²/4 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ c : ℝ, c = 2 ∧ a^2 - b^2 = c^2) → 
  (2 / a^2 + 3 / b^2 = 1) → 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 8 + y^2 / 4 = 1) :=
by sorry

end ellipse_equation_l1078_107885


namespace function_derivative_equality_l1078_107898

/-- Given a function f(x) = x(2017 + ln x), prove that if f'(x₀) = 2018, then x₀ = 1 -/
theorem function_derivative_equality (x₀ : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x * (2017 + Real.log x)
  (deriv f x₀ = 2018) → x₀ = 1 := by
  sorry

end function_derivative_equality_l1078_107898


namespace quadratic_inequality_solution_sets_l1078_107824

/-- Given that the solution set of ax^2 + bx + c ≤ 0 is {x | x ≤ -1/3 ∨ x ≥ 2},
    prove that the solution set of cx^2 + bx + a > 0 is {x | x < -3 ∨ x > 1/2} -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : ∀ x, ax^2 + b*x + c ≤ 0 ↔ x ≤ -1/3 ∨ x ≥ 2) :
  ∀ x, c*x^2 + b*x + a > 0 ↔ x < -3 ∨ x > 1/2 := by
  sorry

end quadratic_inequality_solution_sets_l1078_107824


namespace perpendicular_lines_k_value_l1078_107875

/-- Theorem: Given two lines with direction vectors perpendicular to each other, 
    we can determine the value of k in the second line equation. -/
theorem perpendicular_lines_k_value (k : ℝ) 
  (line1 : ℝ × ℝ → Prop) 
  (line2 : ℝ × ℝ → Prop)
  (dir1 : ℝ × ℝ) 
  (dir2 : ℝ × ℝ) :
  (∀ x y, line1 (x, y) ↔ x + 3*y - 7 = 0) →
  (∀ x y, line2 (x, y) ↔ k*x - y - 2 = 0) →
  (dir1 = (1, -3)) →  -- Direction vector of line1
  (dir2 = (k, 1))  →  -- Direction vector of line2
  (dir1.1 * dir2.1 + dir1.2 * dir2.2 = 0) →  -- Dot product = 0
  k = 3 := by
sorry

end perpendicular_lines_k_value_l1078_107875


namespace point_division_and_linear_combination_l1078_107878

/-- Given a line segment AB and a point P on it, prove that P divides AB in the ratio 4:1 
    and can be expressed as a linear combination of A and B -/
theorem point_division_and_linear_combination (A B P : ℝ × ℝ) : 
  A = (1, 2) →
  B = (4, 3) →
  (P.1 - A.1) / (B.1 - P.1) = 4 →
  (P.2 - A.2) / (B.2 - P.2) = 4 →
  ∃ (t u : ℝ), P = (t * A.1 + u * B.1, t * A.2 + u * B.2) ∧ t = 1/5 ∧ u = 4/5 :=
by sorry

end point_division_and_linear_combination_l1078_107878


namespace compare_log_and_sqrt_l1078_107833

theorem compare_log_and_sqrt : 2 + Real.log 6 / Real.log 2 > 2 * Real.sqrt 5 := by sorry

end compare_log_and_sqrt_l1078_107833


namespace oplus_2_4_1_3_l1078_107831

-- Define the # operation
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Define the ⊕ operation
def oplus (a b c d : ℝ) : ℝ := hash a (b + d) c - hash a b c

-- Theorem statement
theorem oplus_2_4_1_3 : oplus 2 4 1 3 = 33 := by
  sorry

end oplus_2_4_1_3_l1078_107831


namespace solution_satisfies_system_l1078_107834

/-- The system of equations --/
def system (x y : ℝ) : Prop :=
  x^2 * y + x * y^2 - 2*x - 2*y + 10 = 0 ∧
  x^3 * y - x * y^3 - 2*x^2 + 2*y^2 - 30 = 0

/-- The solution to the system of equations --/
def solution : ℝ × ℝ := (-4, -1)

/-- Theorem stating that the solution satisfies the system of equations --/
theorem solution_satisfies_system :
  let (x, y) := solution
  system x y := by sorry

end solution_satisfies_system_l1078_107834


namespace population_approximation_l1078_107829

def initial_population : ℝ := 14999.999999999998
def first_year_change : ℝ := 0.12
def second_year_change : ℝ := 0.12

def population_after_two_years : ℝ :=
  initial_population * (1 + first_year_change) * (1 - second_year_change)

theorem population_approximation :
  ∃ ε > 0, |population_after_two_years - 14784| < ε :=
sorry

end population_approximation_l1078_107829


namespace first_brand_price_l1078_107869

/-- The regular price of pony jeans -/
def pony_price : ℝ := 18

/-- The total savings on 5 pairs of jeans -/
def total_savings : ℝ := 8.55

/-- The sum of the two discount rates -/
def sum_discount_rates : ℝ := 0.22

/-- The discount rate on pony jeans -/
def pony_discount_rate : ℝ := 0.15

/-- The number of pairs of the first brand of jeans -/
def num_first_brand : ℕ := 3

/-- The number of pairs of pony jeans -/
def num_pony : ℕ := 2

/-- Theorem stating that the regular price of the first brand of jeans is $15 -/
theorem first_brand_price : ∃ (price : ℝ),
  price = 15 ∧
  (price * num_first_brand * (sum_discount_rates - pony_discount_rate) +
   pony_price * num_pony * pony_discount_rate = total_savings) :=
sorry

end first_brand_price_l1078_107869


namespace max_utilization_rate_square_plate_l1078_107808

/-- Given a square steel plate with side length 4 and a rusted corner defined by AF = 2 and BF = 1,
    prove that the maximum utilization rate is 50%. -/
theorem max_utilization_rate_square_plate (side_length : ℝ) (af bf : ℝ) :
  side_length = 4 ∧ af = 2 ∧ bf = 1 →
  ∃ (rect_area : ℝ),
    rect_area ≤ side_length * side_length ∧
    rect_area = side_length * (side_length - af) ∧
    (rect_area / (side_length * side_length)) * 100 = 50 := by
  sorry

end max_utilization_rate_square_plate_l1078_107808


namespace congruence_properties_l1078_107840

theorem congruence_properties : ∀ n : ℤ,
  (n ≡ 0 [ZMOD 2] → ∃ k : ℤ, n = 2 * k) ∧
  (n ≡ 1 [ZMOD 2] → ∃ k : ℤ, n = 2 * k + 1) ∧
  (n ≡ 2018 [ZMOD 2] → ∃ k : ℤ, n = 2 * k) :=
by sorry

end congruence_properties_l1078_107840


namespace flag_arrangement_count_remainder_mod_1000_l1078_107865

/-- The number of red flags -/
def red_flags : ℕ := 11

/-- The number of white flags -/
def white_flags : ℕ := 6

/-- The total number of flags -/
def total_flags : ℕ := red_flags + white_flags

/-- The number of distinguishable flagpoles -/
def flagpoles : ℕ := 2

/-- Represents a valid flag arrangement -/
structure FlagArrangement where
  arrangement : List Bool
  red_count : ℕ
  white_count : ℕ
  no_adjacent_white : Bool
  at_least_one_per_pole : Bool

/-- The number of valid distinguishable arrangements -/
def valid_arrangements : ℕ := 10164

theorem flag_arrangement_count :
  (∃ (arrangements : List FlagArrangement),
    (∀ a ∈ arrangements,
      a.red_count = red_flags ∧
      a.white_count = white_flags ∧
      a.no_adjacent_white = true ∧
      a.at_least_one_per_pole = true) ∧
    arrangements.length = valid_arrangements) :=
sorry

theorem remainder_mod_1000 :
  valid_arrangements % 1000 = 164 :=
sorry

end flag_arrangement_count_remainder_mod_1000_l1078_107865


namespace tysons_swimming_problem_l1078_107893

/-- Tyson's swimming problem -/
theorem tysons_swimming_problem 
  (lake_speed : ℝ) 
  (ocean_speed : ℝ) 
  (total_races : ℕ) 
  (total_time : ℝ) 
  (h1 : lake_speed = 3)
  (h2 : ocean_speed = 2.5)
  (h3 : total_races = 10)
  (h4 : total_time = 11)
  (h5 : total_races % 2 = 0) -- Ensures even number of races for equal distribution
  : ∃ (race_distance : ℝ), 
    race_distance = 3 ∧ 
    (total_races / 2 : ℝ) * (race_distance / lake_speed + race_distance / ocean_speed) = total_time :=
by sorry

end tysons_swimming_problem_l1078_107893


namespace linear_regression_intercept_l1078_107879

/-- Linear regression model parameters -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Mean values of x and y -/
structure MeanValues where
  x_mean : ℝ
  y_mean : ℝ

/-- Theorem: Given a linear regression model and mean values, prove the intercept -/
theorem linear_regression_intercept 
  (model : LinearRegression) 
  (means : MeanValues) 
  (h_slope : model.slope = -12/5) 
  (h_x_mean : means.x_mean = -4) 
  (h_y_mean : means.y_mean = 25) : 
  model.intercept = 77/5 := by
  sorry

#check linear_regression_intercept

end linear_regression_intercept_l1078_107879


namespace trigonometric_simplification_l1078_107870

theorem trigonometric_simplification :
  (Real.sin (11 * π / 180) * Real.cos (15 * π / 180) + 
   Real.sin (15 * π / 180) * Real.cos (11 * π / 180)) / 
  (Real.sin (18 * π / 180) * Real.cos (12 * π / 180) + 
   Real.sin (12 * π / 180) * Real.cos (18 * π / 180)) = 
  2 * Real.sin (26 * π / 180) := by
  sorry

end trigonometric_simplification_l1078_107870


namespace greatest_integer_fraction_l1078_107819

theorem greatest_integer_fraction (x : ℤ) : 
  (8 : ℚ) / 11 > (x : ℚ) / 15 ↔ x ≤ 10 :=
by sorry

end greatest_integer_fraction_l1078_107819


namespace sqrt_two_simplification_l1078_107874

theorem sqrt_two_simplification : 3 * Real.sqrt 2 - Real.sqrt 2 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_two_simplification_l1078_107874


namespace pure_imaginary_product_imaginary_part_quotient_l1078_107818

-- Define complex numbers z₁ and z₂
def z₁ (m : ℝ) : ℂ := m + Complex.I
def z₂ (m : ℝ) : ℂ := 2 + m * Complex.I

-- Theorem 1
theorem pure_imaginary_product (m : ℝ) :
  (z₁ m * z₂ m).re = 0 → m = 0 := by sorry

-- Theorem 2
theorem imaginary_part_quotient (m : ℝ) :
  z₁ m ^ 2 - 2 * z₁ m + 2 = 0 →
  (z₂ m / z₁ m).im = -1/2 := by sorry

end pure_imaginary_product_imaginary_part_quotient_l1078_107818


namespace billy_is_45_l1078_107809

/-- Billy's age -/
def B : ℕ := sorry

/-- Joe's age -/
def J : ℕ := sorry

/-- Billy's age is three times Joe's age -/
axiom billy_age : B = 3 * J

/-- The sum of their ages is 60 -/
axiom total_age : B + J = 60

/-- Prove that Billy is 45 years old -/
theorem billy_is_45 : B = 45 := by sorry

end billy_is_45_l1078_107809


namespace polynomial_difference_independent_of_x_l1078_107851

theorem polynomial_difference_independent_of_x (m n : ℝ) : 
  (∀ x y : ℝ, ∃ k : ℝ, (x^2 + m*x - 2*y + n) - (n*x^2 - 3*x + 4*y - 7) = k) →
  n - m = 4 := by
sorry

end polynomial_difference_independent_of_x_l1078_107851


namespace estimate_sqrt_19_l1078_107800

theorem estimate_sqrt_19 : 6 < 2 + Real.sqrt 19 ∧ 2 + Real.sqrt 19 < 7 := by
  sorry

end estimate_sqrt_19_l1078_107800


namespace union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_less_than_8_l1078_107862

-- Define the sets A, B, C, and U
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 6}
def C (a : ℝ) : Set ℝ := {x | x > a}
def U : Set ℝ := Set.univ

-- Theorem 1: A ∪ B = {x | 1 ≤ x ∧ x ≤ 8}
theorem union_A_B : A ∪ B = {x : ℝ | 1 ≤ x ∧ x ≤ 8} := by sorry

-- Theorem 2: (∁ₐA) ∩ B = {x | 1 ≤ x ∧ x < 2}
theorem complement_A_intersect_B : (Aᶜ) ∩ B = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a < 8
theorem intersection_A_C_nonempty_implies_a_less_than_8 (a : ℝ) :
  (A ∩ C a).Nonempty → a < 8 := by sorry

end union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_less_than_8_l1078_107862


namespace person_age_l1078_107823

/-- The age of a person satisfying a specific equation is 32 years old. -/
theorem person_age : ∃ (age : ℕ), 4 * (age + 4) - 4 * (age - 4) = age ∧ age = 32 := by
  sorry

end person_age_l1078_107823


namespace diophantine_equation_solution_l1078_107838

theorem diophantine_equation_solution (x y : ℤ) :
  x^2 = 2 + 6*y^2 + y^4 ↔ (x = 3 ∧ y = 1) ∨ (x = -3 ∧ y = 1) ∨ (x = 3 ∧ y = -1) ∨ (x = -3 ∧ y = -1) := by
  sorry

end diophantine_equation_solution_l1078_107838


namespace sum_of_roots_l1078_107854

theorem sum_of_roots (h : ℝ) (x₁ x₂ : ℝ) :
  x₁ ≠ x₂ →
  (6 * x₁^2 - 5 * h * x₁ - 4 * h = 0) →
  (6 * x₂^2 - 5 * h * x₂ - 4 * h = 0) →
  x₁ + x₂ = 5 * h / 6 := by
sorry

end sum_of_roots_l1078_107854


namespace line_exists_l1078_107881

/-- The parabola y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- The line l1: x + 5y - 5 = 0 -/
def line_l1 (x y : ℝ) : Prop := x + 5*y - 5 = 0

/-- The line l: 25x - 5y - 21 = 0 -/
def line_l (x y : ℝ) : Prop := 25*x - 5*y - 21 = 0

/-- Two points are distinct -/
def distinct (x1 y1 x2 y2 : ℝ) : Prop := x1 ≠ x2 ∨ y1 ≠ y2

/-- A line perpendicularly bisects a segment -/
def perpendicularly_bisects (x1 y1 x2 y2 : ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (xm ym : ℝ), line xm ym ∧ 
    xm = (x1 + x2) / 2 ∧ 
    ym = (y1 + y2) / 2 ∧
    (y2 - y1) * (x2 - xm) = (x2 - x1) * (y2 - ym)

theorem line_exists : ∃ (x1 y1 x2 y2 : ℝ),
  parabola x1 y1 ∧ parabola x2 y2 ∧
  line_l x1 y1 ∧ line_l x2 y2 ∧
  distinct x1 y1 x2 y2 ∧
  perpendicularly_bisects x1 y1 x2 y2 line_l1 :=
sorry

end line_exists_l1078_107881


namespace racecourse_length_l1078_107889

/-- Racecourse problem -/
theorem racecourse_length
  (speed_a speed_b : ℝ)
  (head_start : ℝ)
  (h1 : speed_a = 2 * speed_b)
  (h2 : head_start = 64)
  (h3 : speed_a > 0)
  (h4 : speed_b > 0) :
  ∃ (length : ℝ), 
    length > 0 ∧
    length / speed_a = (length - head_start) / speed_b ∧
    length = 128 := by
  sorry

end racecourse_length_l1078_107889


namespace factory_production_equation_l1078_107846

/-- Given the production data of an agricultural machinery factory,
    this theorem states the equation that the average monthly growth rate satisfies. -/
theorem factory_production_equation (x : ℝ) : 
  (500000 : ℝ) = 500000 ∧ 
  (1820000 : ℝ) = 1820000 → 
  50 + 50*(1+x) + 50*(1+x)^2 = 182 :=
by sorry

end factory_production_equation_l1078_107846


namespace binomial_expansion_coefficient_equality_l1078_107805

theorem binomial_expansion_coefficient_equality (n : ℕ) (h1 : n ≥ 6) :
  (Nat.choose n 5 * 3^5 = Nat.choose n 6 * 3^6) → n = 7 := by
  sorry

end binomial_expansion_coefficient_equality_l1078_107805


namespace max_q_minus_r_l1078_107876

theorem max_q_minus_r (q r : ℕ+) (h : 961 = 23 * q + r) : q - r ≤ 23 := by
  sorry

end max_q_minus_r_l1078_107876


namespace complex_abs_ratio_bounds_l1078_107803

theorem complex_abs_ratio_bounds (x y : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (m M : ℝ),
    (∀ z w : ℂ, z ≠ 0 → w ≠ 0 → m ≤ Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ∧
                                 Complex.abs (z + w) / (Complex.abs z + Complex.abs w) ≤ M) ∧
    m = 0 ∧
    M = 1 ∧
    M - m = 1 :=
by sorry

end complex_abs_ratio_bounds_l1078_107803


namespace sixtieth_term_of_arithmetic_sequence_l1078_107835

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence as a function from ℕ to ℚ
  d : ℚ      -- The common difference
  h : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence with a₁ = 7 and a₁₅ = 37,
    prove that a₆₀ = 134.5 -/
theorem sixtieth_term_of_arithmetic_sequence
  (seq : ArithmeticSequence)
  (h1 : seq.a 1 = 7)
  (h15 : seq.a 15 = 37) :
  seq.a 60 = 134.5 := by
  sorry

end sixtieth_term_of_arithmetic_sequence_l1078_107835


namespace chipmunk_families_count_l1078_107855

theorem chipmunk_families_count (families_left families_went_away : ℕ) 
  (h1 : families_left = 21)
  (h2 : families_went_away = 65) :
  families_left + families_went_away = 86 := by
  sorry

end chipmunk_families_count_l1078_107855


namespace sin_cos_difference_special_angle_l1078_107866

theorem sin_cos_difference_special_angle : 
  Real.sin (80 * π / 180) * Real.cos (20 * π / 180) - 
  Real.cos (80 * π / 180) * Real.sin (20 * π / 180) = 
  Real.sqrt 3 / 2 := by
sorry

end sin_cos_difference_special_angle_l1078_107866


namespace ellipse_focal_length_l1078_107891

-- Define the ellipse equation
def ellipse_equation (x y m : ℝ) : Prop :=
  x^2 / (m - 1) + y^2 / m = 1

-- Theorem statement
theorem ellipse_focal_length (m : ℝ) (h1 : m > m - 1) 
  (h2 : ∀ x y : ℝ, ellipse_equation x y m) : 
  ∃ (a b c : ℝ), a^2 = m ∧ b^2 = m - 1 ∧ c^2 = 1 ∧ 2 * c = 2 :=
sorry

end ellipse_focal_length_l1078_107891


namespace log_sum_inequality_l1078_107877

theorem log_sum_inequality (a b : ℝ) (h1 : 2^a = Real.pi) (h2 : 5^b = Real.pi) :
  1/a + 1/b > 2 := by
  sorry

end log_sum_inequality_l1078_107877


namespace mystic_four_calculator_theorem_l1078_107867

/-- Represents the possible operations on the Mystic Four Calculator --/
inductive Operation
| replace_one
| divide_two
| subtract_three
| multiply_four

/-- Represents the state of the Mystic Four Calculator --/
structure CalculatorState where
  display : ℕ

/-- Applies an operation to the calculator state --/
def apply_operation (state : CalculatorState) (op : Operation) : CalculatorState :=
  match op with
  | Operation.replace_one => CalculatorState.mk 1
  | Operation.divide_two => 
      if state.display % 2 = 0 then CalculatorState.mk (state.display / 2)
      else state
  | Operation.subtract_three => 
      if state.display ≥ 3 then CalculatorState.mk (state.display - 3)
      else state
  | Operation.multiply_four => 
      if state.display * 4 < 10000 then CalculatorState.mk (state.display * 4)
      else state

/-- Applies a sequence of operations to the calculator state --/
def apply_sequence (initial : CalculatorState) (ops : List Operation) : CalculatorState :=
  ops.foldl apply_operation initial

theorem mystic_four_calculator_theorem :
  (¬ ∃ (ops : List Operation), (apply_sequence (CalculatorState.mk 0) ops).display = 2007) ∧
  (∃ (ops : List Operation), (apply_sequence (CalculatorState.mk 0) ops).display = 2008) :=
sorry

end mystic_four_calculator_theorem_l1078_107867


namespace mountain_loop_trail_length_l1078_107860

/-- The Mountain Loop Trail Theorem -/
theorem mountain_loop_trail_length 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h1 : x1 + x2 = 36)
  (h2 : x2 + x3 = 40)
  (h3 : x3 + x4 + x5 = 45)
  (h4 : x1 + x4 = 38) :
  x1 + x2 + x3 + x4 + x5 = 81 := by
  sorry

#check mountain_loop_trail_length

end mountain_loop_trail_length_l1078_107860


namespace reflection_matrix_squared_is_identity_l1078_107802

/-- Reflection matrix over a non-zero vector -/
def reflection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  sorry

/-- Theorem: The square of a reflection matrix is the identity matrix -/
theorem reflection_matrix_squared_is_identity (v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  (reflection_matrix v) ^ 2 = !![1, 0; 0, 1] :=
sorry

end reflection_matrix_squared_is_identity_l1078_107802


namespace theta_max_ratio_l1078_107806

/-- Represents a participant's scores in the competition -/
structure Participant where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ
  day3_score : ℕ
  day3_total : ℕ

/-- The competition setup and conditions -/
def Competition (omega theta : Participant) : Prop :=
  omega.day1_score = 200 ∧
  omega.day1_total = 400 ∧
  omega.day2_score + omega.day3_score = 150 ∧
  omega.day2_total + omega.day3_total = 200 ∧
  omega.day1_total + omega.day2_total + omega.day3_total = 600 ∧
  theta.day1_total + theta.day2_total + theta.day3_total = 600 ∧
  theta.day1_score > 0 ∧ theta.day2_score > 0 ∧ theta.day3_score > 0 ∧
  (theta.day1_score : ℚ) / theta.day1_total < (omega.day1_score : ℚ) / omega.day1_total ∧
  (theta.day2_score : ℚ) / theta.day2_total < (omega.day2_score : ℚ) / omega.day2_total ∧
  (theta.day3_score : ℚ) / theta.day3_total < (omega.day3_score : ℚ) / omega.day3_total

/-- Theta's overall success ratio -/
def ThetaRatio (theta : Participant) : ℚ :=
  (theta.day1_score + theta.day2_score + theta.day3_score : ℚ) /
  (theta.day1_total + theta.day2_total + theta.day3_total)

/-- The main theorem stating Theta's maximum possible success ratio -/
theorem theta_max_ratio (omega theta : Participant) 
  (h : Competition omega theta) : ThetaRatio theta ≤ 56 / 75 := by
  sorry


end theta_max_ratio_l1078_107806


namespace problem_solution_l1078_107825

theorem problem_solution (x y : ℝ) : 
  x / y = 15 / 5 → y = 25 → x = 75 := by sorry

end problem_solution_l1078_107825


namespace min_value_of_expression_l1078_107890

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) :
  1/a + 2/b ≥ 9 := by
  sorry

end min_value_of_expression_l1078_107890


namespace complex_number_magnitude_l1078_107837

theorem complex_number_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 30)
  (h2 : Complex.abs (z + 2 * w) = 5)
  (h3 : Complex.abs (z + w) = 2) :
  Complex.abs z = Real.sqrt (19 / 8) := by
  sorry

end complex_number_magnitude_l1078_107837


namespace total_height_is_148_inches_l1078_107895

-- Define the heights of sculptures in feet and inches
def sculpture1_feet : ℕ := 2
def sculpture1_inches : ℕ := 10
def sculpture2_feet : ℕ := 3
def sculpture2_inches : ℕ := 5
def sculpture3_feet : ℕ := 4
def sculpture3_inches : ℕ := 7

-- Define the heights of bases in inches
def base1_inches : ℕ := 4
def base2_inches : ℕ := 6
def base3_inches : ℕ := 8

-- Define the number of inches in a foot
def inches_per_foot : ℕ := 12

-- Function to convert feet and inches to total inches
def to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * inches_per_foot + inches

-- Theorem statement
theorem total_height_is_148_inches :
  to_inches sculpture1_feet sculpture1_inches + base1_inches +
  to_inches sculpture2_feet sculpture2_inches + base2_inches +
  to_inches sculpture3_feet sculpture3_inches + base3_inches = 148 := by
  sorry


end total_height_is_148_inches_l1078_107895


namespace sum_of_absolute_values_l1078_107886

def S (n : ℕ+) : ℤ := n^2 + 6*n + 1

def a (n : ℕ+) : ℤ := S n - S (n-1)

theorem sum_of_absolute_values : |a 1| + |a 2| + |a 3| + |a 4| = 41 := by
  sorry

end sum_of_absolute_values_l1078_107886


namespace sum_of_coefficients_l1078_107861

theorem sum_of_coefficients (b₆ b₅ b₄ b₃ b₂ b₁ b₀ : ℝ) :
  (∀ x : ℝ, (5*x - 2)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 729 := by
sorry

end sum_of_coefficients_l1078_107861


namespace record_breaking_time_l1078_107810

/-- The number of jumps in the record -/
def record : ℕ := 54000

/-- The number of jumps Mark can do per second -/
def jumps_per_second : ℕ := 3

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The time required to break the record in hours -/
def time_to_break_record : ℚ :=
  (record / jumps_per_second) / seconds_per_hour

theorem record_breaking_time :
  time_to_break_record = 5 := by sorry

end record_breaking_time_l1078_107810


namespace inequality_solution_set_l1078_107813

theorem inequality_solution_set (m n : ℝ) : 
  (∀ x, mx - n > 0 ↔ x < 1/3) →
  (∀ x, (m + n) * x < n - m ↔ x > -1/2) :=
by sorry

end inequality_solution_set_l1078_107813


namespace rug_area_proof_l1078_107847

/-- Given three rugs covering a floor area, prove their combined area -/
theorem rug_area_proof (total_covered_area single_layer_area double_layer_area triple_layer_area : ℝ) 
  (h1 : total_covered_area = 140)
  (h2 : double_layer_area = 24)
  (h3 : triple_layer_area = 20)
  (h4 : single_layer_area = total_covered_area - double_layer_area - triple_layer_area) :
  single_layer_area + 2 * double_layer_area + 3 * triple_layer_area = 204 := by
  sorry

end rug_area_proof_l1078_107847


namespace min_a_for_p_half_ge_p_23_value_l1078_107857

def p (a : ℕ) : ℚ :=
  (Nat.choose (41 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose 50 2

theorem min_a_for_p_half_ge :
  ∀ a : ℕ, 1 ≤ a → a ≤ 40 → (∀ b : ℕ, 1 ≤ b → b < a → p b < 1/2) → p a ≥ 1/2 → a = 23 :=
sorry

theorem p_23_value : p 23 = 34/49 :=
sorry

end min_a_for_p_half_ge_p_23_value_l1078_107857


namespace exactly_three_sequences_l1078_107858

/-- Represents a sequence of 10 positive integers -/
def Sequence := Fin 10 → ℕ+

/-- Checks if a sequence satisfies the recurrence relation -/
def satisfies_recurrence (s : Sequence) : Prop :=
  ∀ n : Fin 8, s (n.succ.succ) = s (n.succ) + s n

/-- Checks if a sequence has the required last term -/
def has_correct_last_term (s : Sequence) : Prop :=
  s 9 = 2002

/-- The main theorem stating that there are exactly 3 valid sequences -/
theorem exactly_three_sequences :
  ∃! (sequences : Finset Sequence),
    sequences.card = 3 ∧
    ∀ s ∈ sequences, satisfies_recurrence s ∧ has_correct_last_term s :=
sorry

end exactly_three_sequences_l1078_107858


namespace distance_is_sqrt_5_l1078_107841

/-- A right triangle with sides of length 6, 8, and 10 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- The distance between the centers of the inscribed and circumscribed circles -/
def distance_between_centers (t : RightTriangle) : ℝ := sorry

theorem distance_is_sqrt_5 (t : RightTriangle) :
  distance_between_centers t = Real.sqrt 5 := by sorry

end distance_is_sqrt_5_l1078_107841


namespace triangle_projection_inequality_l1078_107899

/-- Given a triangle ABC with sides a, b, c and projections satisfying certain conditions,
    prove that a specific inequality holds. -/
theorem triangle_projection_inequality 
  (a b c : ℝ) 
  (t r μ : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_t : ∃ (A C₁ : ℝ), A > 0 ∧ C₁ > 0 ∧ C₁ = 2 * t * c) 
  (h_r : ∃ (B A₁ : ℝ), B > 0 ∧ A₁ > 0 ∧ A₁ = 2 * r * a) 
  (h_μ : ∃ (C B₁ : ℝ), C > 0 ∧ B₁ > 0 ∧ B₁ = 2 * μ * b) :
  (a^2 / b^2) * (t / (1 - 2*t))^2 + 
  (b^2 / c^2) * (r / (1 - 2*r))^2 + 
  (c^2 / a^2) * (μ / (1 - 2*μ))^2 + 
  16 * t * r * μ ≥ 1 := by
  sorry

end triangle_projection_inequality_l1078_107899


namespace right_triangle_line_equation_l1078_107832

/-- Given a right triangle in the first quadrant with vertices at (0, 0), (a, 0), and (0, b),
    where the area of the triangle is T, prove that the equation of the line passing through
    (0, b) and (a, 0) in its standard form is 2Tx - a²y + 2Ta = 0. -/
theorem right_triangle_line_equation (a b T : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : T = (1/2) * a * b) :
  ∃ (A B C : ℝ), A * a + B * b + C = 0 ∧ 
                 (∀ x y : ℝ, A * x + B * y + C = 0 ↔ 2 * T * x - a^2 * y + 2 * T * a = 0) :=
sorry

end right_triangle_line_equation_l1078_107832


namespace garden_area_calculation_l1078_107896

/-- The area of a rectangular garden plot -/
def garden_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of a rectangular garden plot with length 1.2 meters and width 0.5 meters is 0.6 square meters -/
theorem garden_area_calculation :
  garden_area 1.2 0.5 = 0.6 := by
  sorry

end garden_area_calculation_l1078_107896


namespace rectangle_area_l1078_107828

/-- Given a rectangle with perimeter 28 cm and width 6 cm, its area is 48 square centimeters. -/
theorem rectangle_area (perimeter width : ℝ) (h_perimeter : perimeter = 28) (h_width : width = 6) :
  let length := (perimeter - 2 * width) / 2
  width * length = 48 :=
by sorry

end rectangle_area_l1078_107828


namespace final_sum_after_operations_l1078_107880

theorem final_sum_after_operations (S a b : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by
  sorry

end final_sum_after_operations_l1078_107880


namespace store_products_theorem_l1078_107853

-- Define the universe of products
variable (Product : Type)

-- Define a predicate for products displayed in the store
variable (displayed : Product → Prop)

-- Define a predicate for products that are for sale
variable (for_sale : Product → Prop)

-- Theorem stating that if not all displayed products are for sale,
-- then some displayed products are not for sale and not all displayed products are for sale
theorem store_products_theorem (h : ¬∀ (p : Product), displayed p → for_sale p) :
  (∃ (p : Product), displayed p ∧ ¬for_sale p) ∧
  (¬∀ (p : Product), displayed p → for_sale p) :=
by sorry

end store_products_theorem_l1078_107853


namespace smallest_valid_number_last_four_digits_l1078_107839

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 5

def contains_2_and_5 (n : ℕ) : Prop :=
  2 ∈ n.digits 10 ∧ 5 ∈ n.digits 10

theorem smallest_valid_number_last_four_digits :
  ∃ m : ℕ,
    m > 0 ∧
    m % 6 = 0 ∧
    m % 5 = 0 ∧
    is_valid_number m ∧
    contains_2_and_5 m ∧
    (∀ n : ℕ, n > 0 → n % 6 = 0 → n % 5 = 0 → is_valid_number n → contains_2_and_5 n → m ≤ n) ∧
    m % 10000 = 5220 :=
by sorry

end smallest_valid_number_last_four_digits_l1078_107839


namespace patrick_savings_ratio_l1078_107894

theorem patrick_savings_ratio :
  ∀ (bicycle_cost initial_savings current_savings lent_amount : ℕ),
    bicycle_cost = 150 →
    lent_amount = 50 →
    current_savings = 25 →
    initial_savings = current_savings + lent_amount →
    (initial_savings : ℚ) / bicycle_cost = 1 / 2 := by
  sorry

end patrick_savings_ratio_l1078_107894


namespace square_root_of_one_fourth_l1078_107897

theorem square_root_of_one_fourth : 
  {x : ℝ | x^2 = (1/4 : ℝ)} = {-(1/2 : ℝ), (1/2 : ℝ)} := by sorry

end square_root_of_one_fourth_l1078_107897


namespace dinner_bill_contribution_l1078_107892

theorem dinner_bill_contribution (num_friends : ℕ) 
  (num_18_meals num_24_meals num_30_meals : ℕ)
  (cost_18_meal cost_24_meal cost_30_meal : ℚ)
  (num_appetizers : ℕ) (cost_appetizer : ℚ)
  (tip_percentage : ℚ)
  (h1 : num_friends = 8)
  (h2 : num_18_meals = 4)
  (h3 : num_24_meals = 2)
  (h4 : num_30_meals = 2)
  (h5 : cost_18_meal = 18)
  (h6 : cost_24_meal = 24)
  (h7 : cost_30_meal = 30)
  (h8 : num_appetizers = 3)
  (h9 : cost_appetizer = 12)
  (h10 : tip_percentage = 12 / 100) :
  let total_cost := num_18_meals * cost_18_meal + 
                    num_24_meals * cost_24_meal + 
                    num_30_meals * cost_30_meal + 
                    num_appetizers * cost_appetizer
  let total_with_tip := total_cost + total_cost * tip_percentage
  let contribution_per_person := total_with_tip / num_friends
  contribution_per_person = 30.24 := by
sorry

end dinner_bill_contribution_l1078_107892


namespace shaded_square_area_ratio_l1078_107848

theorem shaded_square_area_ratio : 
  let shaded_square_side : ℝ := Real.sqrt 2
  let grid_side : ℝ := 6
  (shaded_square_side ^ 2) / (grid_side ^ 2) = 1 / 18 := by
  sorry

end shaded_square_area_ratio_l1078_107848


namespace simplify_fraction_product_l1078_107804

theorem simplify_fraction_product : 5 * (14 / 3) * (27 / (-35)) * (9 / 7) = -6 := by sorry

end simplify_fraction_product_l1078_107804


namespace journey_speed_fraction_l1078_107830

/-- Proves that if a person travels part of a journey at 5 mph and the rest at 15 mph,
    with an average speed of 10 mph for the entire journey,
    then the fraction of time spent traveling at 15 mph is 1/2. -/
theorem journey_speed_fraction (t₅ t₁₅ : ℝ) (h₁ : t₅ > 0) (h₂ : t₁₅ > 0) :
  (5 * t₅ + 15 * t₁₅) / (t₅ + t₁₅) = 10 →
  t₁₅ / (t₅ + t₁₅) = 1 / 2 := by
sorry

end journey_speed_fraction_l1078_107830


namespace sqrt_seven_fraction_inequality_l1078_107871

theorem sqrt_seven_fraction_inequality (m n : ℤ) 
  (h1 : m ≥ 1) (h2 : n ≥ 1) (h3 : Real.sqrt 7 - (m : ℝ) / n > 0) : 
  Real.sqrt 7 - (m : ℝ) / n > 1 / (m * n) := by
  sorry

end sqrt_seven_fraction_inequality_l1078_107871


namespace quadratic_inequality_solution_l1078_107811

theorem quadratic_inequality_solution (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, 0 < a ∧ a < 1 → (a * x^2 - (a + 1) * x + 1 < 0 ↔ 1 < x ∧ x < 1/a)) ∧
  (∀ x : ℝ, a > 1 → (a * x^2 - (a + 1) * x + 1 < 0 ↔ 1/a < x ∧ x < 1)) ∧
  (a = 1 → ¬∃ x : ℝ, a * x^2 - (a + 1) * x + 1 < 0) :=
by sorry

end quadratic_inequality_solution_l1078_107811


namespace original_bacteria_count_l1078_107843

theorem original_bacteria_count (current : ℕ) (increase : ℕ) (original : ℕ)
  (h1 : current = 8917)
  (h2 : increase = 8317)
  (h3 : current = original + increase) :
  original = 600 := by
  sorry

end original_bacteria_count_l1078_107843


namespace factorize_x_squared_minus_one_l1078_107856

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by sorry

end factorize_x_squared_minus_one_l1078_107856


namespace arithmetic_sequence_11_terms_l1078_107801

theorem arithmetic_sequence_11_terms (a₁ : ℕ) (d : ℕ) (n : ℕ) (aₙ : ℕ) :
  a₁ = 12 →
  d = 6 →
  n = 11 →
  aₙ = a₁ + (n - 1) * d →
  aₙ = 72 :=
by sorry

end arithmetic_sequence_11_terms_l1078_107801


namespace math_books_in_same_box_l1078_107844

def total_textbooks : ℕ := 15
def math_textbooks : ℕ := 3
def box_capacities : List ℕ := [3, 5, 7]

def probability_all_math_in_same_box : ℚ := 25 / 242

theorem math_books_in_same_box :
  let total_arrangements := (total_textbooks.choose box_capacities[0]!) *
    ((total_textbooks - box_capacities[0]!).choose box_capacities[1]!) *
    ((total_textbooks - box_capacities[0]! - box_capacities[1]!).choose box_capacities[2]!)
  let favorable_outcomes := 
    (total_textbooks - math_textbooks).choose box_capacities[0]! +
    ((total_textbooks - math_textbooks).choose (box_capacities[1]! - math_textbooks)) * 
      ((total_textbooks - box_capacities[1]!).choose box_capacities[0]!) +
    ((total_textbooks - math_textbooks).choose (box_capacities[2]! - math_textbooks)) * 
      ((total_textbooks - box_capacities[2]!).choose box_capacities[0]!)
  probability_all_math_in_same_box = favorable_outcomes / total_arrangements :=
by sorry

end math_books_in_same_box_l1078_107844


namespace absolute_value_inequality_solution_set_l1078_107826

theorem absolute_value_inequality_solution_set : 
  {x : ℝ | |x - 2| ≤ 1} = Set.Icc 1 3 := by sorry

end absolute_value_inequality_solution_set_l1078_107826


namespace unique_number_l1078_107849

theorem unique_number : ∃! (n : ℕ), n > 0 ∧ n^2 + n = 217 ∧ 3 ∣ n ∧ n = 15 := by
  sorry

end unique_number_l1078_107849


namespace johns_weight_l1078_107812

/-- Given that Roy weighs 4 pounds and John is 77 pounds heavier than Roy,
    prove that John weighs 81 pounds. -/
theorem johns_weight (roy_weight : ℕ) (weight_difference : ℕ) :
  roy_weight = 4 →
  weight_difference = 77 →
  roy_weight + weight_difference = 81 :=
by sorry

end johns_weight_l1078_107812


namespace trigonometric_identities_l1078_107850

theorem trigonometric_identities (α : Real) (h : Real.tan (α / 2) = 3) :
  (Real.tan (α + Real.pi / 3) = (48 - 4 * Real.sqrt 3) / 11) ∧
  ((Real.sin α + 2 * Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = -5 / 17) := by
  sorry

end trigonometric_identities_l1078_107850
