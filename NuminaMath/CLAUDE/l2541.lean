import Mathlib

namespace max_value_theorem_l2541_254139

theorem max_value_theorem (x y : ℝ) (h : x * y > 0) :
  ∃ (max : ℝ), max = 4 - 2 * Real.sqrt 2 ∧
  ∀ (z : ℝ), z = x / (x + y) + 2 * y / (x + 2 * y) → z ≤ max :=
sorry

end max_value_theorem_l2541_254139


namespace hexagon_angle_measure_l2541_254100

theorem hexagon_angle_measure (F I U R G E : ℝ) : 
  -- Hexagon angle sum is 720°
  F + I + U + R + G + E = 720 →
  -- Four angles are congruent
  F = I ∧ F = U ∧ F = R →
  -- G and E are supplementary
  G + E = 180 →
  -- Prove that E is 45°
  E = 45 := by
sorry

end hexagon_angle_measure_l2541_254100


namespace louis_oranges_l2541_254146

/-- Given the fruit distribution among Louis, Samantha, and Marley, prove that Louis has 5 oranges. -/
theorem louis_oranges :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples marley_oranges marley_apples : ℕ),
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 →
  louis_oranges = 5 := by
sorry


end louis_oranges_l2541_254146


namespace average_of_three_numbers_l2541_254143

theorem average_of_three_numbers (N : ℝ) : 
  9 ≤ N ∧ N ≤ 17 →
  ∃ k : ℕ, (6 + 10 + N) / 3 = 2 * k →
  (6 + 10 + N) / 3 = 10 := by
sorry

end average_of_three_numbers_l2541_254143


namespace mary_fruits_left_l2541_254109

/-- The number of fruits Mary has left after buying and eating some -/
def fruits_left (apples oranges blueberries : ℕ) : ℕ :=
  (apples + oranges + blueberries) - 3

theorem mary_fruits_left : fruits_left 14 9 6 = 26 := by
  sorry

end mary_fruits_left_l2541_254109


namespace fourth_root_simplification_l2541_254169

theorem fourth_root_simplification : Real.sqrt (Real.sqrt (2^8 * 3^4 * 11^0)) = 12 := by
  sorry

end fourth_root_simplification_l2541_254169


namespace quadratic_equation_k_l2541_254152

/-- The equation (k-1)x^(|k|+1)-x+5=0 is quadratic in x -/
def is_quadratic (k : ℝ) : Prop :=
  (k - 1 ≠ 0) ∧ (|k| + 1 = 2)

theorem quadratic_equation_k (k : ℝ) :
  is_quadratic k → k = -1 := by
  sorry

end quadratic_equation_k_l2541_254152


namespace min_cups_to_fill_cylinder_l2541_254135

def cylinder_capacity : ℚ := 980
def cup_capacity : ℚ := 80

theorem min_cups_to_fill_cylinder :
  ⌈cylinder_capacity / cup_capacity⌉ = 13 := by
  sorry

end min_cups_to_fill_cylinder_l2541_254135


namespace product_of_numbers_l2541_254175

theorem product_of_numbers (x y : ℝ) 
  (sum_eq : x + y = 22) 
  (sum_squares_eq : x^2 + y^2 = 404) : 
  x * y = 40 := by sorry

end product_of_numbers_l2541_254175


namespace flour_spill_ratio_l2541_254127

def initial_flour : ℕ := 500
def used_flour : ℕ := 240
def needed_flour : ℕ := 370

theorem flour_spill_ratio :
  let flour_after_baking := initial_flour - used_flour
  let flour_after_spill := initial_flour - needed_flour
  let spilled_flour := flour_after_baking - flour_after_spill
  (spilled_flour : ℚ) / flour_after_baking = 1 / 2 := by sorry

end flour_spill_ratio_l2541_254127


namespace functional_equation_solution_l2541_254173

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, f (x * y + z) = f x * f y + f z

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, SatisfiesProperty f →
    (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end functional_equation_solution_l2541_254173


namespace no_integer_solution_x4_plus_6_eq_y3_l2541_254151

theorem no_integer_solution_x4_plus_6_eq_y3 :
  ∀ (x y : ℤ), (x^4 + 6) % 13 ≠ y^3 % 13 := by
  sorry

end no_integer_solution_x4_plus_6_eq_y3_l2541_254151


namespace geometric_sequence_product_l2541_254161

theorem geometric_sequence_product (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n = a 1 * q ^ (n - 1)) →  -- Definition of geometric sequence
  a 1 = 1 →                        -- First term is 1
  a 5 = 16 →                       -- Last term is 16
  a 2 * a 3 * a 4 = 64 :=           -- Product of middle three terms is 64
by
  sorry

end geometric_sequence_product_l2541_254161


namespace arithmetic_calculations_l2541_254187

theorem arithmetic_calculations : 
  (23 - 17 - (-6) + (-16) = -4) ∧ 
  (0 - 32 / ((-2)^3 - (-4)) = 8) := by sorry

end arithmetic_calculations_l2541_254187


namespace simplify_polynomial_l2541_254142

theorem simplify_polynomial (y : ℝ) : 
  3 * y^3 - 7 * y^2 + 12 * y + 5 - (2 * y^3 - 4 + 3 * y^2 - 9 * y) = y^3 - 10 * y^2 + 21 * y + 9 := by
  sorry

end simplify_polynomial_l2541_254142


namespace y_minus_x_value_l2541_254168

theorem y_minus_x_value (x y : ℝ) (hx : |x| = 5) (hy : |y| = 9) (hxy : x < y) :
  y - x = 4 ∨ y - x = 14 := by
  sorry

end y_minus_x_value_l2541_254168


namespace greatest_common_multiple_under_150_l2541_254155

theorem greatest_common_multiple_under_150 :
  ∃ (n : ℕ), n = 120 ∧ 
  n % 15 = 0 ∧ 
  n % 20 = 0 ∧ 
  n < 150 ∧ 
  ∀ (m : ℕ), m % 15 = 0 → m % 20 = 0 → m < 150 → m ≤ n :=
by sorry

end greatest_common_multiple_under_150_l2541_254155


namespace lukes_trip_time_l2541_254165

/-- Calculates the total trip time for Luke's journey to London --/
theorem lukes_trip_time :
  let bus_time : ℚ := 75 / 60
  let walk_time : ℚ := 15 / 60
  let wait_time : ℚ := 2 * walk_time
  let train_time : ℚ := 6
  bus_time + walk_time + wait_time + train_time = 8 :=
by sorry

end lukes_trip_time_l2541_254165


namespace vector_equation_holds_l2541_254132

def vector2D := ℝ × ℝ

def dot_product (v w : vector2D) : ℝ :=
  v.1 * w.1 + v.2 * w.2

def scale_vector (s : ℝ) (v : vector2D) : vector2D :=
  (s * v.1, s * v.2)

theorem vector_equation_holds (a c : vector2D) (b : vector2D) : 
  a = (1, 1) → c = (2, 2) → 
  scale_vector (dot_product a b) c = scale_vector (dot_product b c) a := by
  sorry

end vector_equation_holds_l2541_254132


namespace circles_intersect_distance_between_centers_l2541_254153

/-- Given two circles M and N, prove that they intersect --/
theorem circles_intersect : ∀ (a : ℝ),
  a > 0 →
  (∃ (x y : ℝ), x^2 + y^2 - 2*a*y = 0 ∧ x + y = 0 ∧ (x - (-x))^2 = 4) →
  a = Real.sqrt 2 →
  ∃ (x y : ℝ), 
    x^2 + (y - a)^2 = a^2 ∧
    (x - 1)^2 + (y - 1)^2 = 1 :=
by
  sorry

/-- The distance between the centers of the circles is between |R-r| and R+r --/
theorem distance_between_centers (a : ℝ) (h : a = Real.sqrt 2) :
  Real.sqrt 2 - 1 < Real.sqrt (1 + (Real.sqrt 2 - 1)^2) ∧
  Real.sqrt (1 + (Real.sqrt 2 - 1)^2) < Real.sqrt 2 + 1 :=
by
  sorry

end circles_intersect_distance_between_centers_l2541_254153


namespace fifth_number_is_24_l2541_254171

/-- Definition of the sequence function -/
def f (n : ℕ) : ℕ := n^2 - 1

/-- Theorem stating that the fifth number in the sequence is 24 -/
theorem fifth_number_is_24 : f 5 = 24 := by
  sorry

end fifth_number_is_24_l2541_254171


namespace sunzi_wood_measurement_l2541_254170

/-- Proves the equation for the wood measurement problem from "The Mathematical Classic of Sunzi" -/
theorem sunzi_wood_measurement (x : ℝ) 
  (h1 : ∃ (rope_length : ℝ), rope_length = x + 4.5) 
  (h2 : ∃ (half_rope : ℝ), half_rope = x - 1 ∧ half_rope = (x + 4.5) / 2) : 
  (x + 4.5) / 2 = x - 1 := by
  sorry

end sunzi_wood_measurement_l2541_254170


namespace other_root_of_equation_l2541_254167

theorem other_root_of_equation (m : ℤ) : 
  (∃ x : ℤ, x^2 - 3*x - m = 0 ∧ x = ⌊Real.sqrt 6⌋) →
  (∃ y : ℤ, y^2 - 3*y - m = 0 ∧ y ≠ ⌊Real.sqrt 6⌋ ∧ y = 1) :=
by sorry

end other_root_of_equation_l2541_254167


namespace ball_returns_to_start_l2541_254177

/-- The number of girls in the circle -/
def n : ℕ := 13

/-- The number of positions to advance in each throw -/
def k : ℕ := 5

/-- The function that determines the next girl to receive the ball -/
def next (x : ℕ) : ℕ := (x + k) % n

/-- The sequence of girls who receive the ball, starting from position 1 -/
def ball_sequence : ℕ → ℕ
  | 0 => 1
  | i + 1 => next (ball_sequence i)

theorem ball_returns_to_start :
  ∃ m : ℕ, m > 0 ∧ ball_sequence m = 1 ∧ ∀ i < m, ball_sequence i ≠ 1 :=
sorry

end ball_returns_to_start_l2541_254177


namespace employee_selection_distribution_l2541_254118

theorem employee_selection_distribution 
  (total_employees : ℕ) 
  (under_35 : ℕ) 
  (between_35_49 : ℕ) 
  (over_50 : ℕ) 
  (selected : ℕ) 
  (h1 : total_employees = 500) 
  (h2 : under_35 = 125) 
  (h3 : between_35_49 = 280) 
  (h4 : over_50 = 95) 
  (h5 : selected = 100) 
  (h6 : total_employees = under_35 + between_35_49 + over_50) :
  let select_under_35 := (under_35 * selected) / total_employees
  let select_between_35_49 := (between_35_49 * selected) / total_employees
  let select_over_50 := (over_50 * selected) / total_employees
  select_under_35 = 25 ∧ select_between_35_49 = 56 ∧ select_over_50 = 19 := by
  sorry

end employee_selection_distribution_l2541_254118


namespace final_price_correct_l2541_254149

/-- The final selling price of an item after two discounts -/
def final_price (m : ℝ) : ℝ :=
  0.8 * m - 10

/-- Theorem stating the correctness of the final price calculation -/
theorem final_price_correct (m : ℝ) :
  let first_discount := 0.2
  let second_discount := 10
  let price_after_first := m * (1 - first_discount)
  let final_price := price_after_first - second_discount
  final_price = 0.8 * m - 10 :=
by sorry

end final_price_correct_l2541_254149


namespace vector_sum_magnitude_l2541_254133

theorem vector_sum_magnitude (a b : ℝ × ℝ) : 
  ‖a - 2 • b‖ = 1 → a • b = 1 → ‖a + 2 • b‖ = 3 := by
  sorry

end vector_sum_magnitude_l2541_254133


namespace value_of_b_l2541_254148

theorem value_of_b (a c b : ℝ) : 
  a = 105 → 
  c = 70 → 
  a^4 = 21 * 25 * 15 * b * c^3 → 
  b = 0.045 := by
  sorry

end value_of_b_l2541_254148


namespace purely_imaginary_modulus_l2541_254176

theorem purely_imaginary_modulus (a : ℝ) :
  let z : ℂ := a + Complex.I
  (z.re = 0) → Complex.abs z = 1 := by
  sorry

end purely_imaginary_modulus_l2541_254176


namespace correlation_relationships_l2541_254184

/-- Represents a relationship between two factors --/
inductive Relationship
| TeacherStudent
| SphereVolumeRadius
| AppleProductionClimate
| CrowsCawingOmen
| TreeDiameterHeight
| StudentIDNumber

/-- Defines whether a relationship has a correlation --/
def has_correlation (r : Relationship) : Prop :=
  match r with
  | Relationship.TeacherStudent => true
  | Relationship.SphereVolumeRadius => false
  | Relationship.AppleProductionClimate => true
  | Relationship.CrowsCawingOmen => false
  | Relationship.TreeDiameterHeight => true
  | Relationship.StudentIDNumber => false

/-- Theorem stating which relationships have correlations --/
theorem correlation_relationships :
  (has_correlation Relationship.TeacherStudent) ∧
  (has_correlation Relationship.AppleProductionClimate) ∧
  (has_correlation Relationship.TreeDiameterHeight) ∧
  (¬ has_correlation Relationship.SphereVolumeRadius) ∧
  (¬ has_correlation Relationship.CrowsCawingOmen) ∧
  (¬ has_correlation Relationship.StudentIDNumber) := by
  sorry


end correlation_relationships_l2541_254184


namespace johnny_fish_count_l2541_254162

theorem johnny_fish_count (total : ℕ) (sony_multiplier : ℕ) (johnny_count : ℕ) : 
  total = 40 →
  sony_multiplier = 4 →
  total = johnny_count + sony_multiplier * johnny_count →
  johnny_count = 8 := by
sorry

end johnny_fish_count_l2541_254162


namespace work_efficiency_ratio_l2541_254158

-- Define the work efficiencies of A and B
def work_efficiency_A : ℚ := 1 / 45
def work_efficiency_B : ℚ := 1 / 22.5

-- Define the combined work time
def combined_work_time : ℚ := 15

-- Define B's individual work time
def B_work_time : ℚ := 22.5

-- Theorem statement
theorem work_efficiency_ratio :
  (work_efficiency_A / work_efficiency_B) = 45 / 2 := by
  sorry

end work_efficiency_ratio_l2541_254158


namespace special_quad_integer_area_iff_conditions_l2541_254104

/-- A quadrilateral ABCD with special properties -/
structure SpecialQuad where
  AB : ℝ
  CD : ℝ
  -- AB ⊥ BC and BC ⊥ CD
  perpendicular : True
  -- BC is tangent to a circle centered at O
  tangent : True
  -- AD is the diameter of the circle
  diameter : True

/-- The area of the special quadrilateral is an integer -/
def has_integer_area (q : SpecialQuad) : Prop :=
  ∃ n : ℕ, (q.AB + q.CD) * Real.sqrt (q.AB * q.CD) = n

/-- The product of AB and CD is a perfect square -/
def is_perfect_square_product (q : SpecialQuad) : Prop :=
  ∃ m : ℕ, q.AB * q.CD = m^2

theorem special_quad_integer_area_iff_conditions (q : SpecialQuad) :
  has_integer_area q ↔ is_perfect_square_product q ∧ has_integer_area q :=
sorry

end special_quad_integer_area_iff_conditions_l2541_254104


namespace square_diagonal_l2541_254110

/-- The diagonal length of a square with area 338 square meters is 26 meters. -/
theorem square_diagonal (area : ℝ) (diagonal : ℝ) : 
  area = 338 → diagonal = 26 := by
  sorry

end square_diagonal_l2541_254110


namespace f_greater_g_iff_a_geq_half_l2541_254156

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a - Real.log x

noncomputable def g (x : ℝ) : ℝ := 1 / x - 1 / Real.exp (x - 1)

theorem f_greater_g_iff_a_geq_half (a : ℝ) :
  (∀ x > 1, f a x > g x) ↔ a ≥ 1/2 := by
  sorry

end f_greater_g_iff_a_geq_half_l2541_254156


namespace positive_reals_inequality_l2541_254114

theorem positive_reals_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) : x^2 + y^2 < 1 := by
  sorry

end positive_reals_inequality_l2541_254114


namespace joan_oranges_l2541_254154

theorem joan_oranges (total_oranges sara_oranges : ℕ) 
  (h1 : total_oranges = 47) 
  (h2 : sara_oranges = 10) : 
  total_oranges - sara_oranges = 37 := by
  sorry

end joan_oranges_l2541_254154


namespace total_applications_eq_600_l2541_254196

def in_state_applications : ℕ := 200

def out_state_applications : ℕ := 2 * in_state_applications

def total_applications : ℕ := in_state_applications + out_state_applications

theorem total_applications_eq_600 : total_applications = 600 := by
  sorry

end total_applications_eq_600_l2541_254196


namespace largest_even_odd_two_digit_l2541_254112

-- Define the set of two-digit numbers
def TwoDigitNumbers : Set Nat := {n : Nat | 10 ≤ n ∧ n ≤ 99}

-- Define even numbers
def IsEven (n : Nat) : Prop := ∃ k : Nat, n = 2 * k

-- Define odd numbers
def IsOdd (n : Nat) : Prop := ∃ k : Nat, n = 2 * k + 1

-- Theorem statement
theorem largest_even_odd_two_digit :
  (∀ n ∈ TwoDigitNumbers, IsEven n → n ≤ 98) ∧
  (∃ n ∈ TwoDigitNumbers, IsEven n ∧ n = 98) ∧
  (∀ n ∈ TwoDigitNumbers, IsOdd n → n ≤ 99) ∧
  (∃ n ∈ TwoDigitNumbers, IsOdd n ∧ n = 99) :=
sorry

end largest_even_odd_two_digit_l2541_254112


namespace smallest_factor_l2541_254121

def is_divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem smallest_factor (w n : ℕ) : 
  w > 0 → 
  n > 0 → 
  (∀ w' : ℕ, w' ≥ w → is_divisible (w' * n) (2^5)) →
  (∀ w' : ℕ, w' ≥ w → is_divisible (w' * n) (3^3)) →
  (∀ w' : ℕ, w' ≥ w → is_divisible (w' * n) (10^2)) →
  w = 120 →
  n ≥ 180 :=
sorry

end smallest_factor_l2541_254121


namespace two_digit_square_sum_l2541_254136

/-- Two-digit integer -/
def TwoDigitInt (x : ℕ) : Prop := 10 ≤ x ∧ x < 100

/-- Reverse digits of a two-digit integer -/
def reverseDigits (x : ℕ) : ℕ := 
  let tens := x / 10
  let ones := x % 10
  10 * ones + tens

theorem two_digit_square_sum (x y n : ℕ) : 
  TwoDigitInt x → TwoDigitInt y → y = reverseDigits x → x^2 + y^2 = n^2 → x + y + n = 264 := by
  sorry

end two_digit_square_sum_l2541_254136


namespace spoonfuls_per_bowl_l2541_254147

/-- Proves that the number of spoonfuls in each bowl is 25 -/
theorem spoonfuls_per_bowl
  (clusters_per_spoonful : ℕ)
  (clusters_per_box : ℕ)
  (bowls_per_box : ℕ)
  (h1 : clusters_per_spoonful = 4)
  (h2 : clusters_per_box = 500)
  (h3 : bowls_per_box = 5) :
  clusters_per_box / (bowls_per_box * clusters_per_spoonful) = 25 := by
  sorry

end spoonfuls_per_bowl_l2541_254147


namespace non_parallel_diagonals_32gon_l2541_254129

/-- The number of diagonals not parallel to any side in a regular n-gon -/
def non_parallel_diagonals (n : ℕ) : ℕ :=
  let total_diagonals := n * (n - 3) / 2
  let parallel_pairs := n / 2
  let diagonals_per_pair := (n - 4) / 2
  let parallel_diagonals := parallel_pairs * diagonals_per_pair
  total_diagonals - parallel_diagonals

/-- Theorem: In a regular 32-gon, the number of diagonals not parallel to any of its sides is 240 -/
theorem non_parallel_diagonals_32gon :
  non_parallel_diagonals 32 = 240 := by
  sorry


end non_parallel_diagonals_32gon_l2541_254129


namespace complex_number_real_part_l2541_254107

theorem complex_number_real_part : 
  ∀ (z : ℂ) (a : ℝ), 
  (z / (2 + a * Complex.I) = 2 / (1 + Complex.I)) → 
  (z.im = -3) → 
  (z.re = 1) := by
sorry

end complex_number_real_part_l2541_254107


namespace angle_P_measure_l2541_254125

-- Define the triangle PQR
structure Triangle :=
  (P Q R : Real)

-- Define the properties of the triangle
def valid_triangle (t : Triangle) : Prop :=
  t.P > 0 ∧ t.Q > 0 ∧ t.R > 0 ∧ t.P + t.Q + t.R = 180

-- Define the theorem
theorem angle_P_measure (t : Triangle) 
  (h1 : valid_triangle t) 
  (h2 : t.Q = 3 * t.R) 
  (h3 : t.R = 18) : 
  t.P = 108 := by
  sorry

end angle_P_measure_l2541_254125


namespace alcohol_bottle_problem_l2541_254181

/-- The amount of alcohol originally in the bottle -/
def original_amount : ℝ := 750

/-- The amount poured back in after the first pour -/
def amount_added : ℝ := 40

/-- The amount poured out in the third pour -/
def third_pour : ℝ := 180

/-- The amount remaining after all pours -/
def final_amount : ℝ := 60

theorem alcohol_bottle_problem :
  let first_pour := original_amount * (1/3)
  let after_first_pour := original_amount - first_pour + amount_added
  let second_pour := after_first_pour * (5/9)
  let after_second_pour := after_first_pour - second_pour
  after_second_pour - third_pour = final_amount :=
sorry


end alcohol_bottle_problem_l2541_254181


namespace equation_solution_l2541_254103

theorem equation_solution : 
  ∃ x : ℚ, (3 / 4 - 2 / 5 : ℚ) = 1 / x ∧ x = 20 / 7 := by
  sorry

end equation_solution_l2541_254103


namespace discriminant_of_specific_quadratic_l2541_254106

theorem discriminant_of_specific_quadratic (a b c : ℝ) : 
  a = 1 → b = -2 → c = 1 → b^2 - 4*a*c = 0 := by
  sorry

end discriminant_of_specific_quadratic_l2541_254106


namespace inverse_variation_example_l2541_254141

-- Define the inverse variation relationship
def inverse_variation (p q : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ p * q = k

-- State the theorem
theorem inverse_variation_example :
  ∀ p q : ℝ,
  inverse_variation p q →
  (p = 1500 → q = 0.25) →
  (p = 3000 → q = 0.125) :=
by sorry

end inverse_variation_example_l2541_254141


namespace sin_2x_derivative_l2541_254150

theorem sin_2x_derivative (f : ℝ → ℝ) (x : ℝ) :
  (∀ x, f x = Real.sin (2 * x)) →
  (deriv f) x = 2 * Real.cos (2 * x) := by
sorry

end sin_2x_derivative_l2541_254150


namespace initial_depth_calculation_l2541_254123

theorem initial_depth_calculation (men_initial : ℕ) (hours_initial : ℕ) (men_extra : ℕ) (hours_final : ℕ) (depth_final : ℕ) :
  men_initial = 75 →
  hours_initial = 8 →
  men_extra = 65 →
  hours_final = 6 →
  depth_final = 70 →
  ∃ (depth_initial : ℕ), 
    (men_initial * hours_initial * depth_final = (men_initial + men_extra) * hours_final * depth_initial) ∧
    depth_initial = 50 := by
  sorry

#check initial_depth_calculation

end initial_depth_calculation_l2541_254123


namespace tan_fifteen_ratio_equals_sqrt_three_l2541_254122

theorem tan_fifteen_ratio_equals_sqrt_three :
  (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
sorry

end tan_fifteen_ratio_equals_sqrt_three_l2541_254122


namespace remaining_money_after_shopping_l2541_254115

/-- The amount of money remaining after spending 30% of $500 is $350. -/
theorem remaining_money_after_shopping (initial_amount : ℝ) (spent_percentage : ℝ) 
  (h1 : initial_amount = 500)
  (h2 : spent_percentage = 0.30) :
  initial_amount - (spent_percentage * initial_amount) = 350 := by
  sorry

end remaining_money_after_shopping_l2541_254115


namespace workshop_average_salary_l2541_254186

theorem workshop_average_salary
  (num_technicians : ℕ)
  (num_total_workers : ℕ)
  (avg_salary_technicians : ℚ)
  (avg_salary_others : ℚ)
  (h1 : num_technicians = 7)
  (h2 : num_total_workers = 56)
  (h3 : avg_salary_technicians = 12000)
  (h4 : avg_salary_others = 6000) :
  let num_other_workers := num_total_workers - num_technicians
  let total_salary := num_technicians * avg_salary_technicians + num_other_workers * avg_salary_others
  total_salary / num_total_workers = 6750 :=
sorry

end workshop_average_salary_l2541_254186


namespace smallest_n_cookie_boxes_l2541_254101

theorem smallest_n_cookie_boxes : ∃ (n : ℕ), n > 0 ∧ 12 ∣ (17 * n - 1) ∧ ∀ (m : ℕ), m > 0 ∧ 12 ∣ (17 * m - 1) → n ≤ m :=
by sorry

end smallest_n_cookie_boxes_l2541_254101


namespace julian_needs_80_more_legos_l2541_254174

/-- The number of legos Julian has -/
def julian_legos : ℕ := 400

/-- The number of airplane models Julian wants to make -/
def num_airplanes : ℕ := 2

/-- The number of legos required for each airplane model -/
def legos_per_airplane : ℕ := 240

/-- The number of additional legos Julian needs -/
def additional_legos : ℕ := num_airplanes * legos_per_airplane - julian_legos

theorem julian_needs_80_more_legos : additional_legos = 80 :=
by sorry

end julian_needs_80_more_legos_l2541_254174


namespace sqrt_equation_solution_l2541_254159

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (3 - 4 * z) = 7 :=
by
  use -23/2
  sorry

end sqrt_equation_solution_l2541_254159


namespace square_sum_division_theorem_l2541_254182

theorem square_sum_division_theorem (a b : ℕ+) :
  let q : ℕ := (a.val^2 + b.val^2) / (a.val + b.val)
  let r : ℕ := (a.val^2 + b.val^2) % (a.val + b.val)
  q^2 + r = 1977 →
  ((a.val = 50 ∧ b.val = 37) ∨
   (a.val = 37 ∧ b.val = 50) ∨
   (a.val = 50 ∧ b.val = 7) ∨
   (a.val = 7 ∧ b.val = 50)) :=
by sorry

end square_sum_division_theorem_l2541_254182


namespace customer_satisfaction_probability_l2541_254178

-- Define the probability of a customer being satisfied
def p : ℝ := sorry

-- Define the conditions
def dissatisfied_review_rate : ℝ := 0.80
def satisfied_review_rate : ℝ := 0.15
def angry_reviews : ℕ := 60
def positive_reviews : ℕ := 20

-- Theorem statement
theorem customer_satisfaction_probability :
  dissatisfied_review_rate * (1 - p) * (angry_reviews + positive_reviews) = angry_reviews ∧
  satisfied_review_rate * p * (angry_reviews + positive_reviews) = positive_reviews →
  p = 0.64 := by
  sorry

end customer_satisfaction_probability_l2541_254178


namespace three_digit_power_ending_l2541_254134

theorem three_digit_power_ending (N : ℕ) : 
  (100 ≤ N ∧ N < 1000) → 
  (∀ k : ℕ, k > 0 → N^k ≡ N [ZMOD 1000]) → 
  (N = 625 ∨ N = 376) :=
sorry

end three_digit_power_ending_l2541_254134


namespace ratio_equality_l2541_254180

theorem ratio_equality (a b c x y z : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
  (h_abc : a^2 + b^2 + c^2 = 49)
  (h_xyz : x^2 + y^2 + z^2 = 64)
  (h_dot : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end ratio_equality_l2541_254180


namespace lauras_average_speed_l2541_254179

def first_distance : ℝ := 420
def first_time : ℝ := 6.5
def second_distance : ℝ := 480
def second_time : ℝ := 8.25

def total_distance : ℝ := first_distance + second_distance
def total_time : ℝ := first_time + second_time

theorem lauras_average_speed :
  total_distance / total_time = 900 / 14.75 := by sorry

end lauras_average_speed_l2541_254179


namespace diophantine_equation_solvability_l2541_254124

theorem diophantine_equation_solvability (m : ℤ) :
  ∃ (k : ℕ+) (a b c d : ℕ+), a * b - c * d = m := by
  sorry

end diophantine_equation_solvability_l2541_254124


namespace partnership_profit_difference_l2541_254194

/-- Given a partnership scenario with specific investments and profit-sharing rules, 
    calculate the difference in profit shares between two partners. -/
theorem partnership_profit_difference 
  (john_investment mike_investment : ℚ)
  (total_profit : ℚ)
  (effort_share investment_share : ℚ)
  (h1 : john_investment = 700)
  (h2 : mike_investment = 300)
  (h3 : total_profit = 3000.0000000000005)
  (h4 : effort_share = 1/3)
  (h5 : investment_share = 2/3)
  (h6 : effort_share + investment_share = 1) :
  let total_investment := john_investment + mike_investment
  let john_investment_ratio := john_investment / total_investment
  let mike_investment_ratio := mike_investment / total_investment
  let john_share := (effort_share * total_profit / 2) + 
                    (investment_share * total_profit * john_investment_ratio)
  let mike_share := (effort_share * total_profit / 2) + 
                    (investment_share * total_profit * mike_investment_ratio)
  john_share - mike_share = 800.0000000000001 := by
  sorry


end partnership_profit_difference_l2541_254194


namespace gcd_8_factorial_12_factorial_l2541_254140

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_8_factorial_12_factorial :
  Nat.gcd (factorial 8) (factorial 12) = factorial 8 := by
  sorry

end gcd_8_factorial_12_factorial_l2541_254140


namespace min_value_expression_l2541_254183

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) :
  x^2 + 4*x*y + 4*y^2 + 4*z^2 ≥ 192 ∧ ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 64 ∧ x₀^2 + 4*x₀*y₀ + 4*y₀^2 + 4*z₀^2 = 192 :=
sorry

end min_value_expression_l2541_254183


namespace exists_u_floor_power_minus_n_even_l2541_254108

theorem exists_u_floor_power_minus_n_even :
  ∃ u : ℝ, u > 0 ∧ ∀ n : ℕ, n > 0 → ∃ k : ℤ, 
    (Int.floor (u ^ n) : ℤ) - n = 2 * k :=
by sorry

end exists_u_floor_power_minus_n_even_l2541_254108


namespace power_difference_over_sum_l2541_254197

theorem power_difference_over_sum : (3^2016 - 3^2014) / (3^2016 + 3^2014) = 4/5 := by
  sorry

end power_difference_over_sum_l2541_254197


namespace matrix_multiplication_example_l2541_254160

theorem matrix_multiplication_example :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; -1, 4]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![2, 5; 0, -3]
  A * B = !![6, 21; -2, -17] := by
  sorry

end matrix_multiplication_example_l2541_254160


namespace article_pricing_loss_l2541_254190

/-- Proves that for an article with a given cost price, selling at 216 results in a 20% profit,
    and selling at 153 results in a 15% loss. -/
theorem article_pricing_loss (CP : ℝ) : 
  CP * 1.2 = 216 → (CP - 153) / CP * 100 = 15 := by
  sorry

end article_pricing_loss_l2541_254190


namespace complex_modulus_l2541_254189

theorem complex_modulus (r : ℝ) (z : ℂ) (hr : |r| < 1) (hz : z - 1/z = r) :
  Complex.abs z = Real.sqrt (1 + r^2/2) := by
  sorry

end complex_modulus_l2541_254189


namespace sin_cos_equation_solution_l2541_254120

theorem sin_cos_equation_solution (x : Real) 
  (h1 : x ∈ Set.Icc 0 Real.pi) 
  (h2 : Real.sin (x + Real.sin x) = Real.cos (x - Real.cos x)) : 
  x = Real.pi / 4 := by
  sorry

end sin_cos_equation_solution_l2541_254120


namespace mobile_profit_percentage_l2541_254188

-- Define the given values
def grinder_cost : ℝ := 15000
def mobile_cost : ℝ := 8000
def grinder_loss_percentage : ℝ := 0.02
def overall_profit : ℝ := 500

-- Define the theorem
theorem mobile_profit_percentage :
  let grinder_selling_price := grinder_cost * (1 - grinder_loss_percentage)
  let total_cost := grinder_cost + mobile_cost
  let total_selling_price := total_cost + overall_profit
  let mobile_selling_price := total_selling_price - grinder_selling_price
  let mobile_profit := mobile_selling_price - mobile_cost
  (mobile_profit / mobile_cost) * 100 = 10 := by
sorry

end mobile_profit_percentage_l2541_254188


namespace circle_radius_range_l2541_254193

/-- Given points P and C in a 2D Cartesian coordinate system, 
    if there exist two distinct points A and B on the circle centered at C with radius r, 
    such that PA - 2AB = 0, then r is in the range [1, 5). -/
theorem circle_radius_range (P C A B : ℝ × ℝ) (r : ℝ) : 
  P = (2, 2) →
  C = (5, 6) →
  A ≠ B →
  (∃ (A B : ℝ × ℝ), 
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = r^2 ∧ 
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = r^2 ∧
    (A.1 - P.1, A.2 - P.2) = 2 • (B.1 - A.1, B.2 - A.2)) →
  r ∈ Set.Icc 1 5 ∧ r ≠ 5 :=
by sorry

end circle_radius_range_l2541_254193


namespace apple_distribution_l2541_254137

theorem apple_distribution (x : ℕ) : 
  x > 0 →
  x - x / 5 - x / 12 - x / 8 - x / 20 - x / 4 - x / 7 - x / 30 - 4 * (x / 30) - 300 ≤ 50 →
  x = 3360 := by
sorry

end apple_distribution_l2541_254137


namespace grisha_hat_color_l2541_254199

/-- Represents the color of a hat -/
inductive HatColor
| White
| Black

/-- Represents a person in the game -/
structure Person where
  name : String
  hatColor : HatColor
  canSee : List String

/-- The game setup -/
structure GameSetup where
  totalHats : Nat
  whiteHats : Nat
  blackHats : Nat
  persons : List Person
  remainingHats : Nat

/-- Predicate to check if a person can determine their hat color -/
def canDetermineColor (setup : GameSetup) (person : Person) : Prop := sorry

/-- The main theorem -/
theorem grisha_hat_color (setup : GameSetup) 
  (h1 : setup.totalHats = 5)
  (h2 : setup.whiteHats = 2)
  (h3 : setup.blackHats = 3)
  (h4 : setup.remainingHats = 2)
  (h5 : setup.persons.length = 3)
  (h6 : ∃ zhenya ∈ setup.persons, zhenya.name = "Zhenya" ∧ zhenya.canSee = ["Lyova", "Grisha"])
  (h7 : ∃ lyova ∈ setup.persons, lyova.name = "Lyova" ∧ lyova.canSee = ["Grisha"])
  (h8 : ∃ grisha ∈ setup.persons, grisha.name = "Grisha" ∧ grisha.canSee = [])
  (h9 : ∃ zhenya ∈ setup.persons, zhenya.name = "Zhenya" ∧ ¬canDetermineColor setup zhenya)
  (h10 : ∃ lyova ∈ setup.persons, lyova.name = "Lyova" ∧ ¬canDetermineColor setup lyova) :
  ∃ grisha ∈ setup.persons, grisha.name = "Grisha" ∧ grisha.hatColor = HatColor.Black ∧ canDetermineColor setup grisha :=
sorry

end grisha_hat_color_l2541_254199


namespace lcm_factor_42_l2541_254198

theorem lcm_factor_42 (A B : ℕ+) : 
  Nat.gcd A B = 42 → 
  max A B = 840 → 
  42 ∣ Nat.lcm A B :=
by
  sorry

end lcm_factor_42_l2541_254198


namespace least_multiple_of_25_greater_than_450_l2541_254185

theorem least_multiple_of_25_greater_than_450 : 
  ∀ n : ℕ, n * 25 > 450 → n * 25 ≥ 475 :=
sorry

end least_multiple_of_25_greater_than_450_l2541_254185


namespace exists_n_power_half_eq_twenty_l2541_254157

theorem exists_n_power_half_eq_twenty :
  ∃ n : ℝ, n > 0 ∧ n^(n/2) = 20 := by
  sorry

end exists_n_power_half_eq_twenty_l2541_254157


namespace equation_linear_iff_a_eq_neg_two_l2541_254163

/-- The equation (a-2)x^(|a|^(-1)+3) = 0 is linear in x if and only if a = -2 -/
theorem equation_linear_iff_a_eq_neg_two (a : ℝ) :
  (∀ x, ∃ b c : ℝ, (a - 2) * x^(|a|⁻¹ + 3) = b * x + c) ↔ a = -2 :=
sorry

end equation_linear_iff_a_eq_neg_two_l2541_254163


namespace fraction_sum_equals_one_l2541_254128

theorem fraction_sum_equals_one (x y : ℝ) 
  (h1 : 3 * x + 2 * y ≠ 0) (h2 : 3 * x - 2 * y ≠ 0) : 
  (7 * x - 5 * y) / (3 * x + 2 * y) + 
  (5 * x - 8 * y) / (3 * x - 2 * y) - 
  (x - 9 * y) / (3 * x + 2 * y) - 
  (8 * x - 10 * y) / (3 * x - 2 * y) = 1 := by
sorry

end fraction_sum_equals_one_l2541_254128


namespace union_of_M_and_N_l2541_254105

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by
  sorry

end union_of_M_and_N_l2541_254105


namespace regular_polygon_exterior_angle_l2541_254119

/-- For a regular polygon with n sides, if each exterior angle measures 45°, then n = 8. -/
theorem regular_polygon_exterior_angle (n : ℕ) : n > 2 → (360 : ℝ) / n = 45 → n = 8 := by
  sorry

end regular_polygon_exterior_angle_l2541_254119


namespace circle_symmetry_l2541_254111

/-- Definition of the first circle C₁ -/
def circle_C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

/-- Definition of the second circle C₂ -/
def circle_C2 (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

/-- Definition of the line of symmetry -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Function to check if two points are symmetric with respect to the line -/
def symmetric_points (x1 y1 x2 y2 : ℝ) : Prop :=
  symmetry_line ((x1 + x2) / 2) ((y1 + y2) / 2) ∧
  x2 - x1 = y2 - y1

/-- Theorem stating that C₂ is symmetric to C₁ with respect to the given line -/
theorem circle_symmetry :
  ∀ (x1 y1 x2 y2 : ℝ),
    circle_C1 x1 y1 →
    circle_C2 x2 y2 →
    symmetric_points x1 y1 x2 y2 :=
by
  sorry


end circle_symmetry_l2541_254111


namespace students_with_average_age_16_l2541_254166

theorem students_with_average_age_16 (total_students : ℕ) (total_avg_age : ℕ) 
  (students_avg_14 : ℕ) (age_15th_student : ℕ) :
  total_students = 15 →
  total_avg_age = 15 →
  students_avg_14 = 5 →
  age_15th_student = 11 →
  ∃ (students_avg_16 : ℕ),
    students_avg_16 = 9 ∧
    students_avg_16 * 16 = total_students * total_avg_age - students_avg_14 * 14 - age_15th_student :=
by sorry

end students_with_average_age_16_l2541_254166


namespace inverse_variation_cube_and_sqrt_l2541_254191

theorem inverse_variation_cube_and_sqrt (k : ℝ) :
  (∀ x > 0, x^3 * Real.sqrt x = k) →
  (4^3 * Real.sqrt 4 = 2 * k) →
  (16^3 * Real.sqrt 16 = 128 * k) := by
sorry

end inverse_variation_cube_and_sqrt_l2541_254191


namespace classroom_desks_l2541_254102

theorem classroom_desks :
  ∀ N y : ℕ,
  (3 * N = 4 * y) →  -- After 1/4 of students leave, 3/4N = 4/7y simplifies to 3N = 4y
  y ≤ 30 →
  y = 21 :=
by
  sorry

end classroom_desks_l2541_254102


namespace like_terms_ratio_l2541_254195

theorem like_terms_ratio (m n : ℕ) : 
  (∃ (x y : ℝ), 2 * x^(m-2) * y^3 = -1/2 * x^2 * y^(2*n-1)) → 
  m / n = 2 := by
  sorry

end like_terms_ratio_l2541_254195


namespace always_odd_l2541_254138

theorem always_odd (p m : ℤ) (h_p : Odd p) : Odd (p^3 + 3*p*m^2 + 2*m) := by
  sorry

end always_odd_l2541_254138


namespace fraction_evaluation_l2541_254116

theorem fraction_evaluation : (3106 - 2935 + 17)^2 / 121 = 292 := by
  sorry

end fraction_evaluation_l2541_254116


namespace evaluate_expression_l2541_254145

theorem evaluate_expression : 5 * (9 - 3) + 8 = 38 := by
  sorry

end evaluate_expression_l2541_254145


namespace quadratic_inequality_solution_l2541_254130

theorem quadratic_inequality_solution (x : ℝ) :
  -10 * x^2 + 6 * x + 8 < 0 ↔ -0.64335 < x ∧ x < 1.24335 := by sorry

end quadratic_inequality_solution_l2541_254130


namespace multivariable_jensen_inequality_l2541_254144

/-- A function F: ℝⁿ → ℝ is convex if for any two points x and y in ℝⁿ and weights q₁, q₂ ≥ 0 with q₁ + q₂ = 1,
    F(q₁x + q₂y) ≤ q₁F(x) + q₂F(y) -/
def IsConvex (n : ℕ) (F : (Fin n → ℝ) → ℝ) : Prop :=
  ∀ (x y : Fin n → ℝ) (q₁ q₂ : ℝ), q₁ ≥ 0 → q₂ ≥ 0 → q₁ + q₂ = 1 →
    F (fun i => q₁ * x i + q₂ * y i) ≤ q₁ * F x + q₂ * F y

/-- Jensen's inequality for multivariable convex functions -/
theorem multivariable_jensen_inequality {n : ℕ} (F : (Fin n → ℝ) → ℝ) (h_convex : IsConvex n F)
    (x y : Fin n → ℝ) (q₁ q₂ : ℝ) (hq₁ : q₁ ≥ 0) (hq₂ : q₂ ≥ 0) (hsum : q₁ + q₂ = 1) :
    F (fun i => q₁ * x i + q₂ * y i) ≤ q₁ * F x + q₂ * F y := by
  sorry

end multivariable_jensen_inequality_l2541_254144


namespace infinite_power_tower_four_equals_sqrt_two_l2541_254113

/-- The infinite power tower function -/
noncomputable def infinitePowerTower (x : ℝ) : ℝ := 
  Real.log x / Real.log (Real.log x)

/-- Theorem: If the infinite power tower of x equals 4, then x equals √2 -/
theorem infinite_power_tower_four_equals_sqrt_two :
  ∀ x : ℝ, x > 0 → infinitePowerTower x = 4 → x = Real.sqrt 2 := by
  sorry

end infinite_power_tower_four_equals_sqrt_two_l2541_254113


namespace square_perimeters_l2541_254117

theorem square_perimeters (a b : ℝ) (h1 : a = 3 * b) 
  (h2 : a ^ 2 + b ^ 2 = 130) (h3 : a ^ 2 - b ^ 2 = 108) : 
  4 * a + 4 * b = 16 * Real.sqrt 13 :=
sorry

end square_perimeters_l2541_254117


namespace new_average_weight_l2541_254164

theorem new_average_weight 
  (initial_students : ℕ) 
  (initial_average : ℚ) 
  (new_student_weight : ℚ) : 
  initial_students = 29 →
  initial_average = 28 →
  new_student_weight = 13 →
  (initial_students * initial_average + new_student_weight) / (initial_students + 1) = 27.5 := by
  sorry

end new_average_weight_l2541_254164


namespace intersection_of_P_and_Q_l2541_254131

-- Define the sets P and Q
def P : Set ℝ := {x | x < 1}
def Q : Set ℝ := {x | x^2 < 4}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | -2 < x ∧ x < 1} := by sorry

end intersection_of_P_and_Q_l2541_254131


namespace solve_for_q_l2541_254126

theorem solve_for_q (k l q : ℚ) : 
  (2/3 : ℚ) = k/45 ∧ (2/3 : ℚ) = (k+l)/75 ∧ (2/3 : ℚ) = (q-l)/105 → q = 90 := by
  sorry

end solve_for_q_l2541_254126


namespace triangle_area_l2541_254172

theorem triangle_area (a b c : ℝ) (h1 : a = 21) (h2 : b = 72) (h3 : c = 75) : 
  (1/2 : ℝ) * a * b = 756 := by
  sorry

end triangle_area_l2541_254172


namespace square_side_length_l2541_254192

theorem square_side_length (s : ℝ) : s^2 + s - 4*s = 4 → s = 4 := by
  sorry

end square_side_length_l2541_254192
