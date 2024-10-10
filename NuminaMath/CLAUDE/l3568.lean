import Mathlib

namespace inequality_proof_l3568_356811

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (a * b + 1) * (b + 1) ≥ (27 / 4) * a * b^2 := by
  sorry

end inequality_proof_l3568_356811


namespace medication_price_reduction_l3568_356801

theorem medication_price_reduction (P : ℝ) (r : ℝ) : 
  P * (1 - r)^2 = 100 →
  P * (1 - r)^2 = P * 0.81 →
  0 < r →
  r < 1 →
  P * (1 - r)^3 = 90 := by
sorry

end medication_price_reduction_l3568_356801


namespace area_of_four_presentable_set_l3568_356861

/-- A complex number is four-presentable if there exists a complex number w 
    with absolute value 4 such that z = w - 1/w -/
def FourPresentable (z : ℂ) : Prop :=
  ∃ w : ℂ, Complex.abs w = 4 ∧ z = w - 1 / w

/-- The set of all four-presentable complex numbers -/
def U : Set ℂ :=
  {z : ℂ | FourPresentable z}

/-- The area of a set in the complex plane -/
noncomputable def Area (S : Set ℂ) : ℝ := sorry

theorem area_of_four_presentable_set :
  Area U = 255 / 16 * Real.pi := by sorry

end area_of_four_presentable_set_l3568_356861


namespace linear_function_properties_l3568_356812

def f (x : ℝ) : ℝ := -2 * x + 2

theorem linear_function_properties :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ f x = y) ∧  -- First quadrant
  (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x = y) ∧  -- Second quadrant
  (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ f x = y) ∧  -- Fourth quadrant
  (f 1 = 0) ∧                               -- x-intercept
  (∀ x > 0, f x < 2) ∧                      -- y < 2 when x > 0
  (∀ x1 x2, x1 < x2 → f x1 > f x2)          -- y decreases as x increases
  := by sorry

end linear_function_properties_l3568_356812


namespace scooter_price_l3568_356807

/-- The total price of a scooter given the upfront payment and the percentage it represents -/
theorem scooter_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 ∧ 
  upfront_percentage = 20 ∧ 
  upfront_payment = (upfront_percentage / 100) * total_price →
  total_price = 1200 := by
sorry

end scooter_price_l3568_356807


namespace gcd_38_23_l3568_356817

theorem gcd_38_23 : Nat.gcd 38 23 = 1 := by
  sorry

end gcd_38_23_l3568_356817


namespace count_true_propositions_l3568_356886

/-- The number of true propositions among the original, converse, inverse, and contrapositive
    of the statement "For real numbers a, b, c, and d, if a=b and c=d, then a+c=b+d" -/
def num_true_propositions : ℕ := 2

/-- The original proposition -/
def original_prop (a b c d : ℝ) : Prop :=
  (a = b ∧ c = d) → (a + c = b + d)

theorem count_true_propositions :
  (∀ a b c d : ℝ, original_prop a b c d) ∧
  (∃ a b c d : ℝ, ¬(a + c = b + d → a = b ∧ c = d)) ∧
  num_true_propositions = 2 := by
  sorry

end count_true_propositions_l3568_356886


namespace weekly_income_proof_l3568_356867

/-- Proves that a weekly income of $500 satisfies the given conditions -/
theorem weekly_income_proof (income : ℝ) : 
  income - 0.2 * income - 55 = 345 → income = 500 := by
  sorry

end weekly_income_proof_l3568_356867


namespace sum_of_digits_8_pow_2004_l3568_356839

/-- The sum of the tens digit and the units digit of 8^2004 in its decimal representation -/
def sum_of_digits : ℕ :=
  let n := 8^2004
  let tens_digit := (n / 10) % 10
  let units_digit := n % 10
  tens_digit + units_digit

/-- Theorem stating that the sum of the tens digit and the units digit of 8^2004 is 7 -/
theorem sum_of_digits_8_pow_2004 : sum_of_digits = 7 := by
  sorry

end sum_of_digits_8_pow_2004_l3568_356839


namespace no_solution_exists_l3568_356897

theorem no_solution_exists (k m : ℕ) : k.factorial + 48 ≠ 48 * (k + 1) ^ m := by
  sorry

end no_solution_exists_l3568_356897


namespace complement_A_intersect_B_l3568_356853

def U : Set Nat := {0, 1, 2, 3, 4}
def A : Set Nat := {0, 1, 3}
def B : Set Nat := {2, 3, 4}

theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2, 4} := by sorry

end complement_A_intersect_B_l3568_356853


namespace cube_root_product_simplification_l3568_356831

theorem cube_root_product_simplification :
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 112 ^ (1/3) := by
  sorry

end cube_root_product_simplification_l3568_356831


namespace haley_candy_eaten_l3568_356881

/-- Given Haley's initial candy count, the amount her sister gave her, and her final candy count,
    calculate how many pieces of candy Haley ate on the first night. -/
theorem haley_candy_eaten (initial : ℕ) (sister_gave : ℕ) (final : ℕ) : 
  initial = 33 → sister_gave = 19 → final = 35 → initial - (final - sister_gave) = 17 := by
  sorry

end haley_candy_eaten_l3568_356881


namespace expected_digits_is_31_20_l3568_356896

/-- A fair 20-sided die numbered from 1 to 20 -/
def icosahedralDie : Finset ℕ := Finset.range 20

/-- The number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 2

/-- The expected number of digits when rolling the die -/
def expectedDigits : ℚ :=
  (icosahedralDie.sum fun i => numDigits (i + 1)) / icosahedralDie.card

/-- Theorem stating the expected number of digits -/
theorem expected_digits_is_31_20 : expectedDigits = 31 / 20 := by
  sorry

end expected_digits_is_31_20_l3568_356896


namespace james_weekly_earnings_l3568_356855

/-- Calculates the weekly earnings from car rental -/
def weekly_earnings (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hourly_rate * hours_per_day * days_per_week

/-- Proves that James' weekly earnings from car rental are $640 -/
theorem james_weekly_earnings :
  weekly_earnings 20 8 4 = 640 := by
  sorry

end james_weekly_earnings_l3568_356855


namespace kelly_apples_l3568_356820

theorem kelly_apples (initial_apples target_apples : ℕ) 
  (h1 : initial_apples = 128) 
  (h2 : target_apples = 250) : 
  target_apples - initial_apples = 122 := by
  sorry

end kelly_apples_l3568_356820


namespace point_coordinates_l3568_356808

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the third quadrant -/
def isInThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (p : Point) 
  (h1 : isInThirdQuadrant p)
  (h2 : distanceToXAxis p = 2)
  (h3 : distanceToYAxis p = 5) :
  p = Point.mk (-5) (-2) := by
  sorry


end point_coordinates_l3568_356808


namespace cylinder_volume_theorem_l3568_356826

/-- The volume of a cylinder with a rectangular net of dimensions 2a and a -/
def cylinder_volume (a : ℝ) : Set ℝ :=
  {v | v = a^3 / Real.pi ∨ v = a^3 / (2 * Real.pi)}

/-- Theorem stating that the volume of the cylinder is either a³/π or a³/(2π) -/
theorem cylinder_volume_theorem (a : ℝ) (h : a > 0) :
  ∀ v ∈ cylinder_volume a, v = a^3 / Real.pi ∨ v = a^3 / (2 * Real.pi) :=
by sorry

end cylinder_volume_theorem_l3568_356826


namespace factorization_of_quadratic_l3568_356829

theorem factorization_of_quadratic (x : ℝ) : x^2 - 5*x = x*(x - 5) := by
  sorry

end factorization_of_quadratic_l3568_356829


namespace probability_different_tens_digits_value_l3568_356835

def range_start : ℕ := 10
def range_end : ℕ := 59
def num_chosen : ℕ := 5

def probability_different_tens_digits : ℚ :=
  (10 ^ num_chosen : ℚ) / (Nat.choose (range_end - range_start + 1) num_chosen)

theorem probability_different_tens_digits_value :
  probability_different_tens_digits = 2500 / 52969 := by sorry

end probability_different_tens_digits_value_l3568_356835


namespace inverse_square_relation_l3568_356884

/-- A function that varies inversely as the square of its input -/
noncomputable def f (y : ℝ) : ℝ := 4 / y^2

theorem inverse_square_relation (y₀ : ℝ) :
  f 6 = 0.1111111111111111 →
  (∃ y, f y = 1) →
  f y₀ = 1 →
  y₀ = 2 := by
sorry

end inverse_square_relation_l3568_356884


namespace window_purchase_savings_l3568_356836

/-- Calculates the cost of purchasing windows under the given offer -/
def cost_with_offer (num_windows : ℕ) : ℕ :=
  ((num_windows + 4) / 7 * 5 + (num_windows + 4) % 7) * 100

/-- Represents the window purchase problem -/
theorem window_purchase_savings (dave_windows doug_windows : ℕ) 
  (h1 : dave_windows = 10) (h2 : doug_windows = 11) : 
  (dave_windows + doug_windows) * 100 - cost_with_offer (dave_windows + doug_windows) = 
  (dave_windows * 100 - cost_with_offer dave_windows) + 
  (doug_windows * 100 - cost_with_offer doug_windows) :=
sorry

end window_purchase_savings_l3568_356836


namespace lengths_form_triangle_l3568_356803

/-- A function that checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the lengths 4, 6, and 9 can form a triangle -/
theorem lengths_form_triangle : can_form_triangle 4 6 9 := by
  sorry

end lengths_form_triangle_l3568_356803


namespace function_inequality_l3568_356834

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property that f is differentiable
variable (hf : Differentiable ℝ f)

-- Define the condition f'(x) + f(x) < 0
variable (hf' : ∀ x, HasDerivAt f (f x) x → (deriv f x + f x < 0))

-- Define m as a real number
variable (m : ℝ)

-- State the theorem
theorem function_inequality :
  f (m - m^2) > Real.exp (m^2 - m + 1) * f 1 :=
sorry

end function_inequality_l3568_356834


namespace rationalize_denominator_l3568_356898

theorem rationalize_denominator : 
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end rationalize_denominator_l3568_356898


namespace fencing_requirement_l3568_356805

/-- Given a rectangular field with area 210 sq. feet and one side 20 feet,
    prove that the sum of the other three sides is 41 feet. -/
theorem fencing_requirement (area : ℝ) (length : ℝ) (width : ℝ) : 
  area = 210 →
  length = 20 →
  area = length * width →
  2 * width + length = 41 := by
  sorry

end fencing_requirement_l3568_356805


namespace rectangular_field_dimension_exists_unique_l3568_356800

theorem rectangular_field_dimension_exists_unique (area : ℝ) :
  ∃! m : ℝ, m > 0 ∧ (3 * m + 8) * (m - 3) = area :=
by sorry

end rectangular_field_dimension_exists_unique_l3568_356800


namespace mean_median_difference_l3568_356844

theorem mean_median_difference (x : ℕ) : 
  let set := [x, x + 2, x + 4, x + 7, x + 27]
  let median := x + 4
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 27)) / 5
  mean = median + 4 := by
  sorry

end mean_median_difference_l3568_356844


namespace monotonic_sequence_divisor_property_l3568_356877

def divisor_count (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

def is_monotonic_increasing (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < j → a i < a j

theorem monotonic_sequence_divisor_property (a : ℕ → ℕ) :
  is_monotonic_increasing a →
  (∀ i j : ℕ, divisor_count (i + j) = divisor_count (a i + a j)) →
  ∀ n : ℕ, a n = n :=
sorry

end monotonic_sequence_divisor_property_l3568_356877


namespace solve_equation_l3568_356859

theorem solve_equation (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (9 * x) * Real.sqrt (12 * x) * Real.sqrt (4 * x) * Real.sqrt (18 * x) = 36) :
  x = Real.sqrt (9 / 22) :=
by sorry

end solve_equation_l3568_356859


namespace cubeRoot_of_negative_eight_eq_negative_two_l3568_356850

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cubeRoot_of_negative_eight_eq_negative_two :
  cubeRoot (-8) = -2 := by sorry

end cubeRoot_of_negative_eight_eq_negative_two_l3568_356850


namespace inequality_proof_l3568_356890

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/2 + z/3 ≤ 1/3 := by
sorry

end inequality_proof_l3568_356890


namespace scholarship_difference_l3568_356865

theorem scholarship_difference (nina kelly wendy : ℕ) : 
  nina < kelly →
  kelly = 2 * wendy →
  wendy = 20000 →
  nina + kelly + wendy = 92000 →
  kelly - nina = 8000 := by
sorry

end scholarship_difference_l3568_356865


namespace trivia_team_grouping_l3568_356819

theorem trivia_team_grouping (total_students : ℕ) (students_not_picked : ℕ) (num_groups : ℕ)
  (h1 : total_students = 120)
  (h2 : students_not_picked = 22)
  (h3 : num_groups = 14)
  : (total_students - students_not_picked) / num_groups = 7 := by
  sorry

end trivia_team_grouping_l3568_356819


namespace always_real_roots_discriminant_one_implies_m_two_l3568_356882

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ :=
  2 * m * x^2 - (5 * m - 1) * x + 3 * m - 1

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ :=
  (5 * m - 1)^2 - 4 * 2 * m * (3 * m - 1)

-- Theorem stating that the equation always has real roots
theorem always_real_roots (m : ℝ) :
  ∃ x : ℝ, quadratic_equation m x = 0 :=
sorry

-- Theorem stating that when the discriminant is 1, m = 2
theorem discriminant_one_implies_m_two :
  ∀ m : ℝ, discriminant m = 1 → m = 2 :=
sorry

end always_real_roots_discriminant_one_implies_m_two_l3568_356882


namespace range_of_m_l3568_356862

def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 11

def q (x m : ℝ) : Prop := 1 - 3*m ≤ x ∧ x ≤ 3 + m

theorem range_of_m (h : ∀ x m : ℝ, m > 0 → (¬(p x) → ¬(q x m)) ∧ ∃ x', ¬(q x' m) ∧ p x') :
  ∀ m : ℝ, m ∈ Set.Ici 8 :=
sorry

end range_of_m_l3568_356862


namespace cost_of_pencils_l3568_356883

/-- Given that 100 pencils cost $30, prove that 1500 pencils cost $450. -/
theorem cost_of_pencils :
  (∃ (cost_per_100 : ℝ), cost_per_100 = 30 ∧ 
   (1500 / 100) * cost_per_100 = 450) :=
by sorry

end cost_of_pencils_l3568_356883


namespace lilly_fish_count_l3568_356824

/-- The number of fish Rosy has -/
def rosy_fish : ℕ := 9

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 19

/-- The number of fish Lilly has -/
def lilly_fish : ℕ := total_fish - rosy_fish

theorem lilly_fish_count : lilly_fish = 10 := by
  sorry

end lilly_fish_count_l3568_356824


namespace trillion_equals_ten_to_sixteen_l3568_356891

-- Define the relationships between numbers
def ten_thousand : ℕ := 10^4
def million : ℕ := 10^6
def billion : ℕ := 10^8
def trillion : ℕ := ten_thousand * million * billion

-- Theorem statement
theorem trillion_equals_ten_to_sixteen : trillion = 10^16 := by
  sorry

end trillion_equals_ten_to_sixteen_l3568_356891


namespace percentage_problem_l3568_356813

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : 3 * (x / 100 * x) = 18) : x = 10 * Real.sqrt 6 := by
  sorry

end percentage_problem_l3568_356813


namespace weekend_rain_probability_l3568_356842

theorem weekend_rain_probability
  (p_friday : ℝ)
  (p_saturday_given_friday : ℝ)
  (p_saturday_given_not_friday : ℝ)
  (p_sunday : ℝ)
  (h1 : p_friday = 0.3)
  (h2 : p_saturday_given_friday = 0.6)
  (h3 : p_saturday_given_not_friday = 0.25)
  (h4 : p_sunday = 0.4) :
  1 - (1 - p_friday) * (1 - p_saturday_given_not_friday * (1 - p_friday)) * (1 - p_sunday) = 0.685 := by
sorry

end weekend_rain_probability_l3568_356842


namespace tangent_sum_problem_l3568_356825

theorem tangent_sum_problem (x y m : ℝ) :
  x^3 + Real.sin (2*x) = m →
  y^3 + Real.sin (2*y) = -m →
  x ∈ Set.Ioo (-π/4) (π/4) →
  y ∈ Set.Ioo (-π/4) (π/4) →
  Real.tan (x + y + π/3) = Real.sqrt 3 := by
  sorry

end tangent_sum_problem_l3568_356825


namespace circles_tangent_internally_l3568_356845

def circle_O₁_center : ℝ × ℝ := (2, 0)
def circle_O₁_radius : ℝ := 1
def circle_O₂_center : ℝ × ℝ := (-1, 0)
def circle_O₂_radius : ℝ := 3

theorem circles_tangent_internally :
  let d := Real.sqrt ((circle_O₂_center.1 - circle_O₁_center.1)^2 + (circle_O₂_center.2 - circle_O₁_center.2)^2)
  d = circle_O₂_radius ∧ d > circle_O₁_radius :=
by sorry

end circles_tangent_internally_l3568_356845


namespace ed_remaining_money_l3568_356887

-- Define the hotel rates
def night_rate : ℝ := 1.50
def morning_rate : ℝ := 2

-- Define Ed's initial money
def initial_money : ℝ := 80

-- Define the duration of stay
def night_hours : ℝ := 6
def morning_hours : ℝ := 4

-- Theorem to prove
theorem ed_remaining_money :
  let night_cost := night_rate * night_hours
  let morning_cost := morning_rate * morning_hours
  let total_cost := night_cost + morning_cost
  let remaining_money := initial_money - total_cost
  remaining_money = 63 := by sorry

end ed_remaining_money_l3568_356887


namespace stratified_sampling_most_representative_l3568_356893

-- Define a type for sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

-- Define a type for high school grades
inductive HighSchoolGrade
  | First
  | Second
  | Third

-- Define a population with subgroups
structure Population where
  subgroups : List HighSchoolGrade

-- Define a characteristic being studied
structure Characteristic where
  name : String
  hasSignificantDifferences : Bool

-- Define a function to determine the most representative sampling method
def mostRepresentativeSamplingMethod (pop : Population) (char : Characteristic) : SamplingMethod :=
  if char.hasSignificantDifferences then
    SamplingMethod.Stratified
  else
    SamplingMethod.SimpleRandom

-- Theorem statement
theorem stratified_sampling_most_representative 
  (pop : Population) 
  (char : Characteristic) 
  (h1 : pop.subgroups = [HighSchoolGrade.First, HighSchoolGrade.Second, HighSchoolGrade.Third]) 
  (h2 : char.name = "Understanding of Jingma") 
  (h3 : char.hasSignificantDifferences = true) :
  mostRepresentativeSamplingMethod pop char = SamplingMethod.Stratified :=
by sorry

end stratified_sampling_most_representative_l3568_356893


namespace system_of_equations_solution_l3568_356849

theorem system_of_equations_solution (x y z : ℝ) 
  (eq1 : 4 * x - 6 * y - 2 * z = 0)
  (eq2 : 2 * x + 6 * y - 28 * z = 0)
  (z_nonzero : z ≠ 0) :
  (x^2 - 6*x*y) / (y^2 + 4*z^2) = -5 := by
  sorry

end system_of_equations_solution_l3568_356849


namespace sqrt_meaningful_range_l3568_356814

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) → x ≥ 1 := by
  sorry

end sqrt_meaningful_range_l3568_356814


namespace x_intercepts_count_l3568_356871

-- Define the polynomial
def f (x : ℝ) : ℝ := (x - 5) * (x^2 + 8*x + 12)

-- State the theorem
theorem x_intercepts_count : 
  ∃ (a b c : ℝ), (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) := by
  sorry

end x_intercepts_count_l3568_356871


namespace infinitely_many_primes_3_mod_4_infinitely_many_primes_5_mod_6_l3568_356868

-- Part 1: Infinitely many primes congruent to 3 modulo 4
theorem infinitely_many_primes_3_mod_4 : 
  ∀ S : Finset Nat, ∃ p : Nat, p ∉ S ∧ Prime p ∧ p % 4 = 3 := by sorry

-- Part 2: Infinitely many primes congruent to 5 modulo 6
theorem infinitely_many_primes_5_mod_6 : 
  ∀ S : Finset Nat, ∃ p : Nat, p ∉ S ∧ Prime p ∧ p % 6 = 5 := by sorry

end infinitely_many_primes_3_mod_4_infinitely_many_primes_5_mod_6_l3568_356868


namespace student_seat_occupancy_l3568_356889

/-- Proves that the fraction of occupied student seats is 4/5 --/
theorem student_seat_occupancy
  (total_chairs : ℕ)
  (rows : ℕ)
  (chairs_per_row : ℕ)
  (awardee_rows : ℕ)
  (admin_teacher_rows : ℕ)
  (parent_rows : ℕ)
  (vacant_student_seats : ℕ)
  (h1 : total_chairs = rows * chairs_per_row)
  (h2 : rows = 10)
  (h3 : chairs_per_row = 15)
  (h4 : awardee_rows = 1)
  (h5 : admin_teacher_rows = 2)
  (h6 : parent_rows = 2)
  (h7 : vacant_student_seats = 15) :
  let student_rows := rows - (awardee_rows + admin_teacher_rows + parent_rows)
  let student_chairs := student_rows * chairs_per_row
  let occupied_student_chairs := student_chairs - vacant_student_seats
  (occupied_student_chairs : ℚ) / student_chairs = 4 / 5 := by
  sorry

end student_seat_occupancy_l3568_356889


namespace complex_number_in_second_quadrant_l3568_356837

/-- The complex number z = (-2-3i)/i is in the second quadrant of the complex plane -/
theorem complex_number_in_second_quadrant : 
  let z : ℂ := (-2 - 3*Complex.I) / Complex.I
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end complex_number_in_second_quadrant_l3568_356837


namespace complex_sum_of_powers_l3568_356833

theorem complex_sum_of_powers (i : ℂ) : i^2 = -1 → i + i^2 + i^3 + i^4 = 0 := by
  sorry

end complex_sum_of_powers_l3568_356833


namespace min_value_x_l3568_356858

theorem min_value_x (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x - 2*y = (x + 16*y) / (2*x*y)) : 
  x ≥ 4 ∧ ∃ (x₀ y₀ : ℝ), x₀ = 4 ∧ y₀ > 0 ∧ x₀ - 2*y₀ = (x₀ + 16*y₀) / (2*x₀*y₀) := by
  sorry

#check min_value_x

end min_value_x_l3568_356858


namespace min_point_of_translated_graph_l3568_356802

-- Define the function
def f (x : ℝ) : ℝ := |x - 1|^2 - 7

-- State the theorem
theorem min_point_of_translated_graph :
  ∃! p : ℝ × ℝ, p.1 = 1 ∧ p.2 = -7 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
sorry

end min_point_of_translated_graph_l3568_356802


namespace percentage_fraction_proof_l3568_356873

theorem percentage_fraction_proof (P : ℚ) : 
  P < 35 → (P / 100) * 180 = 42 → P / 100 = 7 / 30 := by
  sorry

end percentage_fraction_proof_l3568_356873


namespace consecutive_odd_integers_sum_l3568_356895

theorem consecutive_odd_integers_sum (n : ℤ) : 
  (n % 2 = 1) → -- n is odd
  (n + (n + 4) = 150) → -- sum of first and third is 150
  (n + (n + 2) + (n + 4) = 225) -- sum of all three is 225
:= by sorry

end consecutive_odd_integers_sum_l3568_356895


namespace range_of_g_l3568_356832

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_g :
  Set.range g = {π/12, -π/12} := by sorry

end range_of_g_l3568_356832


namespace divisor_inequality_l3568_356879

theorem divisor_inequality (d d' n : ℕ) (h1 : d' > d) (h2 : d ∣ n) (h3 : d' ∣ n) :
  d' > d + d^2 / n :=
by sorry

end divisor_inequality_l3568_356879


namespace inverse_sum_l3568_356872

def f (a b x : ℝ) : ℝ := a * x^2 + b * x

def f_inv (a b x : ℝ) : ℝ := b * x^2 + a * x

theorem inverse_sum (a b : ℝ) :
  (∀ x, f a b (f_inv a b x) = x) → a + b = -2 := by
  sorry

end inverse_sum_l3568_356872


namespace katie_candy_problem_l3568_356860

theorem katie_candy_problem (x : ℕ) : 
  x + 6 - 9 = 7 → x = 10 := by sorry

end katie_candy_problem_l3568_356860


namespace average_bacon_calculation_l3568_356815

-- Define the price per pound of bacon
def price_per_pound : ℝ := 6

-- Define the revenue from a half-size pig
def revenue_from_half_pig : ℝ := 60

-- Define the average amount of bacon from a pig
def average_bacon_amount : ℝ := 20

-- Theorem statement
theorem average_bacon_calculation :
  price_per_pound * (average_bacon_amount / 2) = revenue_from_half_pig :=
by sorry

end average_bacon_calculation_l3568_356815


namespace lollipop_ratio_l3568_356894

theorem lollipop_ratio : 
  ∀ (alison henry diane : ℕ),
    alison = 60 →
    henry = alison + 30 →
    alison + henry + diane = 45 * 6 →
    (alison : ℚ) / diane = 1 / 2 := by
  sorry

end lollipop_ratio_l3568_356894


namespace I_max_min_zero_l3568_356840

noncomputable def f (x : ℝ) : ℝ := x^2 + 3

noncomputable def g (a x : ℝ) : ℝ := a * x + 3

noncomputable def I (a : ℝ) : ℝ := 3 * ∫ x in (-1)..(1), |f x - g a x|

theorem I_max_min_zero :
  (∀ a : ℝ, I a ≤ 0) ∧ (∃ a : ℝ, I a = 0) :=
sorry

end I_max_min_zero_l3568_356840


namespace fraction_simplification_l3568_356863

theorem fraction_simplification : (150 : ℚ) / 4500 = 1 / 30 := by
  sorry

end fraction_simplification_l3568_356863


namespace total_points_is_65_l3568_356810

/-- Represents the types of enemies in the game -/
inductive EnemyType
  | A
  | B
  | C

/-- The number of points earned for defeating each type of enemy -/
def pointsForEnemy (t : EnemyType) : ℕ :=
  match t with
  | EnemyType.A => 10
  | EnemyType.B => 15
  | EnemyType.C => 20

/-- The total number of enemies in the level -/
def totalEnemies : ℕ := 8

/-- The number of each type of enemy in the level -/
def enemyCount (t : EnemyType) : ℕ :=
  match t with
  | EnemyType.A => 3
  | EnemyType.B => 2
  | EnemyType.C => 3

/-- The number of enemies defeated for each type -/
def defeatedCount (t : EnemyType) : ℕ :=
  match t with
  | EnemyType.A => 3  -- All Type A enemies
  | EnemyType.B => 1  -- Half of Type B enemies
  | EnemyType.C => 1  -- One Type C enemy

/-- Calculates the total points earned -/
def totalPointsEarned : ℕ :=
  (defeatedCount EnemyType.A * pointsForEnemy EnemyType.A) +
  (defeatedCount EnemyType.B * pointsForEnemy EnemyType.B) +
  (defeatedCount EnemyType.C * pointsForEnemy EnemyType.C)

/-- Theorem stating that the total points earned is 65 -/
theorem total_points_is_65 : totalPointsEarned = 65 := by
  sorry

end total_points_is_65_l3568_356810


namespace binomial_expansion_problem_l3568_356870

theorem binomial_expansion_problem (n : ℕ) :
  (∀ k, 0 ≤ k ∧ k ≤ n → Nat.choose n 6 ≥ Nat.choose n k) ∧
  (∀ k, k ≠ 6 → Nat.choose n 6 > Nat.choose n k) →
  n = 12 ∧ 2^(n+4) % 7 = 2 := by
sorry

end binomial_expansion_problem_l3568_356870


namespace pizza_combinations_l3568_356875

theorem pizza_combinations (n k : ℕ) (h1 : n = 8) (h2 : k = 5) : 
  Nat.choose n k = 56 := by
  sorry

end pizza_combinations_l3568_356875


namespace retailer_profit_percent_l3568_356804

/-- Calculates the profit percent given purchase price, overhead expenses, and selling price -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percent for the given values is 45.83% -/
theorem retailer_profit_percent :
  profit_percent 225 15 350 = 45.83 := by
  sorry

end retailer_profit_percent_l3568_356804


namespace litter_bag_weight_l3568_356822

theorem litter_bag_weight (gina_bags : ℕ) (neighborhood_multiplier : ℕ) (total_weight : ℕ) :
  gina_bags = 2 →
  neighborhood_multiplier = 82 →
  total_weight = 664 →
  ∃ (bag_weight : ℕ), 
    bag_weight = 4 ∧ 
    (gina_bags + neighborhood_multiplier * gina_bags) * bag_weight = total_weight :=
by sorry

end litter_bag_weight_l3568_356822


namespace problem_solution_l3568_356876

-- Define the function f(x) = x^3 + ax^2 - x
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - 1

theorem problem_solution (a : ℝ) (h : f' a 1 = 4) :
  a = 1 ∧
  ∃ (m b : ℝ), m = 4 ∧ b = -3 ∧ ∀ x y, y = f a x → (y - f a 1 = m * (x - 1) ↔ m*x - y - b = 0) ∧
  ∃ (lower upper : ℝ), lower = -5/27 ∧ upper = 10 ∧
    (∀ x, x ∈ Set.Icc 0 2 → f a x ∈ Set.Icc lower upper) ∧
    (∃ x₁ x₂, x₁ ∈ Set.Icc 0 2 ∧ x₂ ∈ Set.Icc 0 2 ∧ f a x₁ = lower ∧ f a x₂ = upper) :=
by
  sorry


end problem_solution_l3568_356876


namespace min_value_of_function_l3568_356874

theorem min_value_of_function (θ a b : ℝ) (n : ℕ) 
  (h1 : 0 < θ) (h2 : θ < π/2) (h3 : a > 0) (h4 : b > 0) (h5 : n > 0) :
  let f := fun θ => a / (Real.sin θ)^n + b / (Real.cos θ)^n
  ∃ (θ_min : ℝ), ∀ θ', 0 < θ' ∧ θ' < π/2 → 
    f θ' ≥ f θ_min ∧ f θ_min = (a^(2/(n+2:ℝ)) + b^(2/(n+2:ℝ)))^((n+2)/2) :=
by sorry

end min_value_of_function_l3568_356874


namespace product_80641_9999_l3568_356830

theorem product_80641_9999 : 80641 * 9999 = 806329359 := by
  sorry

end product_80641_9999_l3568_356830


namespace talking_birds_l3568_356851

theorem talking_birds (total : ℕ) (non_talking : ℕ) (talking : ℕ) : 
  total = 77 → non_talking = 13 → talking = total - non_talking → talking = 64 := by
sorry

end talking_birds_l3568_356851


namespace carbon_copies_invariant_l3568_356818

/-- Represents a stack of sheets with carbon paper -/
structure CarbonPaperStack :=
  (num_sheets : ℕ)
  (carbons_between : ℕ)

/-- Calculates the number of carbon copies produced by a stack -/
def carbon_copies (stack : CarbonPaperStack) : ℕ :=
  max 0 (stack.num_sheets - 1)

/-- Represents a folding operation on the stack -/
inductive FoldOperation
  | UpperLower
  | LeftRight
  | BackFront

/-- Applies a sequence of folding operations to a stack -/
def apply_folds (stack : CarbonPaperStack) (folds : List FoldOperation) : CarbonPaperStack :=
  stack

theorem carbon_copies_invariant (initial_stack : CarbonPaperStack) (folds : List FoldOperation) :
  initial_stack.num_sheets = 6 ∧ initial_stack.carbons_between = 2 →
  carbon_copies initial_stack = carbon_copies (apply_folds initial_stack folds) ∧
  carbon_copies initial_stack = 5 :=
sorry

end carbon_copies_invariant_l3568_356818


namespace triangle_cosine_double_angle_l3568_356838

theorem triangle_cosine_double_angle 
  (A B C : Real) (a b c : Real) (S : Real) :
  c = 5 →
  B = 2 * Real.pi / 3 →
  S = 15 * Real.sqrt 3 / 4 →
  S = 1/2 * a * c * Real.sin B →
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  Real.sin A / a = Real.sin B / b →
  Real.cos (2*A) = 71/98 :=
by
  sorry

end triangle_cosine_double_angle_l3568_356838


namespace no_three_numbers_with_special_property_l3568_356846

theorem no_three_numbers_with_special_property : 
  ¬ (∃ (a b c : ℕ), 
    a > 1 ∧ b > 1 ∧ c > 1 ∧
    b ∣ (a^2 - 1) ∧ c ∣ (a^2 - 1) ∧
    a ∣ (b^2 - 1) ∧ c ∣ (b^2 - 1) ∧
    a ∣ (c^2 - 1) ∧ b ∣ (c^2 - 1)) :=
by sorry

end no_three_numbers_with_special_property_l3568_356846


namespace farm_output_growth_equation_l3568_356821

/-- Represents the relationship between initial value, final value, and growth rate over two years -/
theorem farm_output_growth_equation (initial_value final_value : ℝ) (growth_rate : ℝ) : 
  initial_value = 80 → final_value = 96.8 → 
  initial_value * (1 + growth_rate)^2 = final_value :=
by
  sorry

#check farm_output_growth_equation

end farm_output_growth_equation_l3568_356821


namespace symmetrical_line_intersection_l3568_356843

/-- Given points A and B, if the line symmetrical to AB about y=a intersects
    the circle (x+3)^2 + (y+2)^2 = 1, then 1/3 ≤ a ≤ 3/2 -/
theorem symmetrical_line_intersection (a : ℝ) : 
  let A : ℝ × ℝ := (-2, 3)
  let B : ℝ × ℝ := (0, a)
  let symmetrical_line (x y : ℝ) := (3 - a) * x - 2 * y + 2 * a = 0
  let circle (x y : ℝ) := (x + 3)^2 + (y + 2)^2 = 1
  (∃ x y, symmetrical_line x y ∧ circle x y) → 1/3 ≤ a ∧ a ≤ 3/2 := by
  sorry


end symmetrical_line_intersection_l3568_356843


namespace arithmetic_calculation_l3568_356841

theorem arithmetic_calculation : -1^4 * 8 - 2^3 / (-4) * (-7 + 5) = -12 := by
  sorry

end arithmetic_calculation_l3568_356841


namespace example_is_quadratic_l3568_356869

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x² = 2 + 3x is a quadratic equation -/
theorem example_is_quadratic : is_quadratic_equation (λ x => x^2 - 3*x - 2) := by
  sorry

end example_is_quadratic_l3568_356869


namespace car_sale_profit_l3568_356888

theorem car_sale_profit (P : ℝ) (h : P > 0) :
  let buying_price := 0.80 * P
  let selling_price := 1.3600000000000001 * P
  let percentage_increase := (selling_price / buying_price - 1) * 100
  percentage_increase = 70.00000000000002 := by
sorry

end car_sale_profit_l3568_356888


namespace taylor_books_l3568_356864

theorem taylor_books (candice amanda kara patricia taylor : ℕ) : 
  candice = 3 * amanda →
  kara = amanda / 2 →
  patricia = 7 * kara →
  taylor = (candice + amanda + kara + patricia) / 4 →
  candice = 18 →
  taylor = 12 := by
sorry

end taylor_books_l3568_356864


namespace max_right_triangle_area_in_rectangle_l3568_356823

theorem max_right_triangle_area_in_rectangle :
  ∀ (a b : ℝ),
  a = 12 ∧ b = 5 →
  (∀ (x y : ℝ),
    x ≤ a ∧ y ≤ b →
    x * y / 2 ≤ 30) ∧
  ∃ (x y : ℝ),
    x ≤ a ∧ y ≤ b ∧
    x * y / 2 = 30 :=
by sorry

end max_right_triangle_area_in_rectangle_l3568_356823


namespace bus_trip_distance_l3568_356866

/-- Given a bus trip with specific conditions, prove that the trip distance is 550 miles. -/
theorem bus_trip_distance (speed : ℝ) (distance : ℝ) : 
  speed = 50 →  -- The actual average speed is 50 mph
  distance / speed = distance / (speed + 5) + 1 →  -- The trip would take 1 hour less if speed increased by 5 mph
  distance = 550 := by
sorry

end bus_trip_distance_l3568_356866


namespace all_representable_l3568_356827

def powers_of_three : List ℕ := [1, 3, 9, 27, 81, 243, 729]

def can_represent (n : ℕ) (powers : List ℕ) : Prop :=
  ∃ (subset : List ℕ) (signs : List Bool),
    subset ⊆ powers ∧
    signs.length = subset.length ∧
    (List.zip subset signs).foldl
      (λ acc (p, sign) => if sign then acc + p else acc - p) 0 = n

theorem all_representable :
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 1093 → can_represent n powers_of_three :=
sorry

end all_representable_l3568_356827


namespace pyramid_levels_theorem_l3568_356880

/-- Represents a pyramid of blocks -/
structure BlockPyramid where
  firstRowBlocks : ℕ
  decreaseRate : ℕ
  totalBlocks : ℕ

/-- Calculate the number of levels in a BlockPyramid -/
def pyramidLevels (p : BlockPyramid) : ℕ :=
  sorry

/-- Theorem: A pyramid with 25 total blocks, 9 blocks in the first row,
    and decreasing by 2 blocks in each row has 5 levels -/
theorem pyramid_levels_theorem (p : BlockPyramid) 
  (h1 : p.firstRowBlocks = 9)
  (h2 : p.decreaseRate = 2)
  (h3 : p.totalBlocks = 25) :
  pyramidLevels p = 5 :=
  sorry

end pyramid_levels_theorem_l3568_356880


namespace imaginary_part_of_1_plus_2i_l3568_356885

theorem imaginary_part_of_1_plus_2i : Complex.im (1 + 2*Complex.I) = 2 := by
  sorry

end imaginary_part_of_1_plus_2i_l3568_356885


namespace eventB_mutually_exclusive_not_complementary_to_eventA_l3568_356899

/-- Represents the possible outcomes when drawing balls from a bag -/
inductive BallDraw
  | TwoBlack
  | ThreeBlack
  | OneBlack
  | NoBlack

/-- The total number of balls in the bag -/
def totalBalls : ℕ := 6

/-- The number of black balls in the bag -/
def blackBalls : ℕ := 3

/-- The number of red balls in the bag -/
def redBalls : ℕ := 3

/-- The number of balls drawn -/
def ballsDrawn : ℕ := 3

/-- Event A: At least 2 black balls are drawn -/
def eventA : Set BallDraw := {BallDraw.TwoBlack, BallDraw.ThreeBlack}

/-- Event B: Exactly 1 black ball is drawn -/
def eventB : Set BallDraw := {BallDraw.OneBlack}

/-- Two events are mutually exclusive if their intersection is empty -/
def mutuallyExclusive (S T : Set BallDraw) : Prop := S ∩ T = ∅

/-- Two events are complementary if their union is the entire sample space -/
def complementary (S T : Set BallDraw) : Prop := S ∪ T = Set.univ

theorem eventB_mutually_exclusive_not_complementary_to_eventA :
  mutuallyExclusive eventA eventB ∧ ¬complementary eventA eventB := by sorry

end eventB_mutually_exclusive_not_complementary_to_eventA_l3568_356899


namespace smallest_block_size_block_with_336_cubes_exists_l3568_356847

/-- Represents the dimensions of a rectangular block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of invisible cubes when viewed from a corner -/
def invisibleCubes (d : BlockDimensions) : ℕ :=
  (d.length - 1) * (d.width - 1) * (d.height - 1)

/-- Calculates the total number of cubes in the block -/
def totalCubes (d : BlockDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Theorem stating the smallest possible number of cubes in the block -/
theorem smallest_block_size (d : BlockDimensions) :
  invisibleCubes d = 143 → totalCubes d ≥ 336 := by
  sorry

/-- Theorem proving the existence of a block with 336 cubes and 143 invisible cubes -/
theorem block_with_336_cubes_exists :
  ∃ d : BlockDimensions, invisibleCubes d = 143 ∧ totalCubes d = 336 := by
  sorry

end smallest_block_size_block_with_336_cubes_exists_l3568_356847


namespace smallest_square_longest_ending_sequence_l3568_356856

/-- A function that returns the length of the longest sequence of the same non-zero digit at the end of a number -/
def longestEndingSequence (n : ℕ) : ℕ :=
  sorry

/-- A function that checks if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  sorry

/-- The theorem stating that 1444 is the smallest square with the longest ending sequence of same non-zero digits -/
theorem smallest_square_longest_ending_sequence :
  ∀ n : ℕ, isPerfectSquare n → n ≠ 1444 → longestEndingSequence n ≤ longestEndingSequence 1444 :=
sorry

end smallest_square_longest_ending_sequence_l3568_356856


namespace parallel_vectors_m_l3568_356828

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem parallel_vectors_m (m : ℝ) : 
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (-2, m)
  are_parallel a (a.1 + 2 * b.1, a.2 + 2 * b.2) → m = -6 := by
sorry

end parallel_vectors_m_l3568_356828


namespace inequality_proof_l3568_356806

theorem inequality_proof :
  (∀ x : ℝ, |x - 1| + |x - 2| ≥ 1) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
    (1 / a + 1 / (2 * b) + 1 / (3 * c) = 1) →
    a + 2 * b + 3 * c ≥ 9) := by
  sorry

end inequality_proof_l3568_356806


namespace f_range_l3568_356852

-- Define the function
def f (x : ℝ) : ℝ := 3 * x - x^3

-- State the theorem
theorem f_range :
  ∀ y ∈ Set.range f,
  ∃ x ∈ Set.Icc (0 : ℝ) 3,
  f x = y ∧ -18 ≤ y ∧ y ≤ 2 :=
by sorry

end f_range_l3568_356852


namespace complex_equation_solution_l3568_356854

theorem complex_equation_solution (a : ℂ) :
  a / (1 - Complex.I) = (1 + Complex.I) / Complex.I → a = -2 * Complex.I :=
by sorry

end complex_equation_solution_l3568_356854


namespace monday_sales_l3568_356848

/-- Represents the sales and pricing of a shoe store -/
structure ShoeStore where
  shoe_price : ℕ
  boot_price : ℕ
  monday_shoe_sales : ℕ
  monday_boot_sales : ℕ
  tuesday_shoe_sales : ℕ
  tuesday_boot_sales : ℕ
  tuesday_total_sales : ℕ

/-- The conditions of the problem -/
def store_conditions (s : ShoeStore) : Prop :=
  s.boot_price = s.shoe_price + 15 ∧
  s.monday_shoe_sales = 22 ∧
  s.monday_boot_sales = 16 ∧
  s.tuesday_shoe_sales = 8 ∧
  s.tuesday_boot_sales = 32 ∧
  s.tuesday_total_sales = 560 ∧
  s.tuesday_shoe_sales * s.shoe_price + s.tuesday_boot_sales * s.boot_price = s.tuesday_total_sales

/-- The theorem to be proved -/
theorem monday_sales (s : ShoeStore) (h : store_conditions s) : 
  s.monday_shoe_sales * s.shoe_price + s.monday_boot_sales * s.boot_price = 316 := by
  sorry


end monday_sales_l3568_356848


namespace base9_perfect_square_multiple_of_3_l3568_356809

/-- Represents a number in base 9 of the form ab4c -/
structure Base9Number where
  a : ℕ
  b : ℕ
  c : ℕ
  a_nonzero : a ≠ 0

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.b + 36 + n.c

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem base9_perfect_square_multiple_of_3 (n : Base9Number) 
  (h1 : isPerfectSquare (toDecimal n))
  (h2 : (toDecimal n) % 3 = 0) :
  n.c = 0 := by
  sorry

end base9_perfect_square_multiple_of_3_l3568_356809


namespace final_sign_is_minus_l3568_356878

/-- Represents the two types of signs on the board -/
inductive Sign
| Plus
| Minus

/-- Represents the state of the board -/
structure Board :=
  (plus_count : ℕ)
  (minus_count : ℕ)

/-- Performs one operation on the board -/
def perform_operation (b : Board) : Board :=
  sorry

/-- Performs n operations on the board -/
def perform_n_operations (b : Board) (n : ℕ) : Board :=
  sorry

/-- The main theorem to prove -/
theorem final_sign_is_minus :
  let initial_board : Board := ⟨10, 15⟩
  let final_board := perform_n_operations initial_board 24
  final_board.plus_count = 0 ∧ final_board.minus_count = 1 :=
sorry

end final_sign_is_minus_l3568_356878


namespace modular_congruence_l3568_356892

theorem modular_congruence (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ 99 * n ≡ 65 [ZMOD 103] → n ≡ 68 [ZMOD 103] := by
  sorry

end modular_congruence_l3568_356892


namespace negative_one_greater_than_negative_sqrt_two_l3568_356857

theorem negative_one_greater_than_negative_sqrt_two :
  -1 > -Real.sqrt 2 := by
  sorry

end negative_one_greater_than_negative_sqrt_two_l3568_356857


namespace least_sum_m_n_l3568_356816

theorem least_sum_m_n : ∃ (m n : ℕ+), 
  (Nat.gcd (m + n) 231 = 1) ∧ 
  (∃ (k : ℕ), m ^ m.val = k * (n ^ n.val)) ∧ 
  (∀ (k : ℕ+), m ≠ k * n) ∧
  (m + n = 75) ∧
  (∀ (m' n' : ℕ+), 
    (Nat.gcd (m' + n') 231 = 1) → 
    (∃ (k : ℕ), m' ^ m'.val = k * (n' ^ n'.val)) → 
    (∀ (k : ℕ+), m' ≠ k * n') → 
    (m' + n' ≥ 75)) :=
sorry

end least_sum_m_n_l3568_356816
