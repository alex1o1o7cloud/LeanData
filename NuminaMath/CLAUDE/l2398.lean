import Mathlib

namespace NUMINAMATH_CALUDE_geometric_locus_l2398_239818

-- Define the conditions
def condition1 (x y : ℝ) : Prop := y^2 - x^2 = 0
def condition2 (x y : ℝ) : Prop := x^2 + y^2 = 4*(y - 1)
def condition3 (x : ℝ) : Prop := x^2 - 2*x + 1 = 0
def condition4 (x y : ℝ) : Prop := x^2 - 2*x*y + y^2 = -1

-- Define the theorem
theorem geometric_locus :
  (∀ x y : ℝ, condition1 x y ↔ (y = x ∨ y = -x)) ∧
  (∀ x y : ℝ, condition2 x y ↔ (x = 0 ∧ y = 2)) ∧
  (∀ x : ℝ, condition3 x ↔ x = 1) ∧
  (¬∃ x y : ℝ, condition4 x y) :=
by sorry

end NUMINAMATH_CALUDE_geometric_locus_l2398_239818


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l2398_239830

theorem negation_of_quadratic_inequality (p : Prop) : 
  (p ↔ ∀ x : ℝ, x^2 + x + 1 ≠ 0) → 
  (¬p ↔ ∃ x : ℝ, x^2 + x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l2398_239830


namespace NUMINAMATH_CALUDE_range_when_p_range_when_p_or_q_l2398_239870

/-- Proposition p: The range of the function y=log(x^2+2ax+2-a) is ℝ -/
def p (a : ℝ) : Prop :=
  ∀ y : ℝ, ∃ x : ℝ, y = Real.log (x^2 + 2*a*x + 2 - a)

/-- Proposition q: ∀x ∈ [0,1], x^2+2x+a ≥ 0 -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 + 2*x + a ≥ 0

/-- If p is true, then a ≤ -2 or a ≥ 1 -/
theorem range_when_p (a : ℝ) : p a → a ≤ -2 ∨ a ≥ 1 := by
  sorry

/-- If p ∨ q is true, then a ≤ -2 or a ≥ 0 -/
theorem range_when_p_or_q (a : ℝ) : p a ∨ q a → a ≤ -2 ∨ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_when_p_range_when_p_or_q_l2398_239870


namespace NUMINAMATH_CALUDE_total_legs_is_22_l2398_239806

/-- The number of legs for each type of animal -/
def dog_legs : ℕ := 4
def bird_legs : ℕ := 2
def insect_legs : ℕ := 6

/-- The number of each type of animal -/
def num_dogs : ℕ := 3
def num_birds : ℕ := 2
def num_insects : ℕ := 2

/-- The total number of legs -/
def total_legs : ℕ := num_dogs * dog_legs + num_birds * bird_legs + num_insects * insect_legs

theorem total_legs_is_22 : total_legs = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_is_22_l2398_239806


namespace NUMINAMATH_CALUDE_solve_equation_l2398_239816

-- Define the function F
def F (a b c : ℚ) : ℚ := a * b^3 + c

-- Theorem statement
theorem solve_equation :
  ∃ a : ℚ, F a 3 8 = F a 5 12 ∧ a = -2/49 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2398_239816


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2398_239877

theorem complex_modulus_problem (z : ℂ) : z = (1 - Complex.I) / (1 + Complex.I) + 2 * Complex.I → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2398_239877


namespace NUMINAMATH_CALUDE_mary_total_spending_l2398_239850

-- Define the amounts spent on each item
def shirt_cost : ℚ := 13.04
def jacket_cost : ℚ := 12.27

-- Define the total cost
def total_cost : ℚ := shirt_cost + jacket_cost

-- Theorem to prove
theorem mary_total_spending : total_cost = 25.31 := by
  sorry

end NUMINAMATH_CALUDE_mary_total_spending_l2398_239850


namespace NUMINAMATH_CALUDE_vector_magnitude_range_l2398_239821

theorem vector_magnitude_range (a b : EuclideanSpace ℝ (Fin 3)) :
  (norm b = 2) → (norm a = 2 * norm (b - a)) → (4/3 : ℝ) ≤ norm a ∧ norm a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_range_l2398_239821


namespace NUMINAMATH_CALUDE_triangle_proof_l2398_239856

theorem triangle_proof (a b c : ℝ) (ha : a = 18) (hb : b = 24) (hc : c = 30) :
  (a + b > c ∧ a + c > b ∧ b + c > a) ∧  -- Triangle inequality
  (a^2 + b^2 = c^2) ∧                    -- Right triangle
  (1/2 * a * b = 216) :=                 -- Area
by sorry

end NUMINAMATH_CALUDE_triangle_proof_l2398_239856


namespace NUMINAMATH_CALUDE_ellipse_focal_chord_properties_l2398_239829

/-- An ellipse with eccentricity e and a line segment PQ passing through its left focus -/
structure EllipseWithFocalChord where
  e : ℝ
  b : ℝ
  hb : b > 0
  pq_not_vertical : True  -- Represents that PQ is not perpendicular to x-axis
  equilateral_exists : True  -- Represents that there exists R making PQR equilateral

/-- The range of eccentricity and slope of PQ for an ellipse with a special focal chord -/
theorem ellipse_focal_chord_properties (E : EllipseWithFocalChord) :
  E.e > Real.sqrt 3 / 3 ∧ E.e < 1 ∧
  ∃ (k : ℝ), (k = 1 / Real.sqrt (3 * E.e^2 - 1) ∨ k = -1 / Real.sqrt (3 * E.e^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_chord_properties_l2398_239829


namespace NUMINAMATH_CALUDE_square_difference_l2398_239891

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) :
  (x - y)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2398_239891


namespace NUMINAMATH_CALUDE_inequality_proof_l2398_239824

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) : 
  a * b * (a - b) + b * c * (b - c) + c * d * (c - d) + d * a * (d - a) ≤ 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2398_239824


namespace NUMINAMATH_CALUDE_f_has_two_zeros_l2398_239899

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x * Real.sin x

theorem f_has_two_zeros :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  x₁ ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧
  x₂ ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧
  f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) ∧ f x = 0 → (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_f_has_two_zeros_l2398_239899


namespace NUMINAMATH_CALUDE_inequality_solution_l2398_239864

theorem inequality_solution : 
  let x : ℝ := 3
  (1/3 - x/3 : ℝ) < -1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2398_239864


namespace NUMINAMATH_CALUDE_binomial_coefficient_7_4_l2398_239857

theorem binomial_coefficient_7_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_7_4_l2398_239857


namespace NUMINAMATH_CALUDE_max_value_constraint_l2398_239834

theorem max_value_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4*x + 3*y < 60) :
  xy*(60 - 4*x - 3*y) ≤ 2000/3 ∧ 
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 4*x₀ + 3*y₀ < 60 ∧ x₀*y₀*(60 - 4*x₀ - 3*y₀) = 2000/3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_constraint_l2398_239834


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2398_239839

def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n : ℕ, Prime n ∧ p = 2^n - 1 ∧ Prime p

theorem largest_mersenne_prime_under_500 :
  (∀ m : ℕ, is_mersenne_prime m ∧ m < 500 → m ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 500 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l2398_239839


namespace NUMINAMATH_CALUDE_school_class_average_difference_l2398_239887

theorem school_class_average_difference :
  let total_students : ℕ := 200
  let total_teachers : ℕ := 5
  let class_sizes : List ℕ := [80, 60, 40, 15, 5]
  
  let t : ℚ := (class_sizes.sum : ℚ) / total_teachers
  
  let s : ℚ := (class_sizes.map (λ size => size * size)).sum / total_students
  
  t - s = -19.25 := by sorry

end NUMINAMATH_CALUDE_school_class_average_difference_l2398_239887


namespace NUMINAMATH_CALUDE_percy_swimming_hours_l2398_239854

/-- Percy's swimming schedule and total hours over 4 weeks -/
theorem percy_swimming_hours :
  let weekday_hours : ℕ := 2 -- 1 hour before school + 1 hour after school
  let weekdays_per_week : ℕ := 5
  let weekend_hours : ℕ := 3
  let weekend_days : ℕ := 2
  let weeks : ℕ := 4
  
  let total_hours_per_week : ℕ := weekday_hours * weekdays_per_week + weekend_hours * weekend_days
  let total_hours_four_weeks : ℕ := total_hours_per_week * weeks
  
  total_hours_four_weeks = 64
  := by sorry

end NUMINAMATH_CALUDE_percy_swimming_hours_l2398_239854


namespace NUMINAMATH_CALUDE_valid_queues_count_l2398_239861

/-- Represents the amount a customer has: 
    1 for 50 cents (exact change), -1 for one dollar (needs change) -/
inductive CustomerMoney : Type
  | exact : CustomerMoney
  | needsChange : CustomerMoney

/-- A queue of customers -/
def CustomerQueue := List CustomerMoney

/-- The nth Catalan number -/
def catalanNumber (n : ℕ) : ℕ := sorry

/-- Checks if a queue is valid (cashier can always give change) -/
def isValidQueue (queue : CustomerQueue) : Prop := sorry

/-- Counts the number of valid queues for 2n customers -/
def countValidQueues (n : ℕ) : ℕ := sorry

/-- Theorem: The number of valid queues for 2n customers 
    (n with exact change, n needing change) is the nth Catalan number -/
theorem valid_queues_count (n : ℕ) : 
  countValidQueues n = catalanNumber n := by sorry

end NUMINAMATH_CALUDE_valid_queues_count_l2398_239861


namespace NUMINAMATH_CALUDE_max_value_of_f_in_interval_l2398_239873

def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 6

theorem max_value_of_f_in_interval :
  ∃ (m : ℝ), m = 11 ∧ 
  (∀ (x : ℝ), -4 ≤ x ∧ x ≤ 4 → f x ≤ m) ∧
  (∃ (x : ℝ), -4 ≤ x ∧ x ≤ 4 ∧ f x = m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_in_interval_l2398_239873


namespace NUMINAMATH_CALUDE_distinct_real_numbers_inequality_l2398_239841

theorem distinct_real_numbers_inequality (x y z : ℝ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hxz : x ≠ z)
  (eq1 : x^2 - x = y*z)
  (eq2 : y^2 - y = z*x)
  (eq3 : z^2 - z = x*y) :
  -1/3 < x ∧ x < 1 ∧ -1/3 < y ∧ y < 1 ∧ -1/3 < z ∧ z < 1 := by
sorry

end NUMINAMATH_CALUDE_distinct_real_numbers_inequality_l2398_239841


namespace NUMINAMATH_CALUDE_savings_calculation_l2398_239804

theorem savings_calculation (savings : ℚ) (tv_cost : ℚ) : 
  (1 / 4 : ℚ) * savings = tv_cost → 
  tv_cost = 300 → 
  savings = 1200 :=
by sorry

end NUMINAMATH_CALUDE_savings_calculation_l2398_239804


namespace NUMINAMATH_CALUDE_existence_of_integer_combination_l2398_239845

theorem existence_of_integer_combination (a b c : ℝ) 
  (hab : ∃ (q : ℚ), a * b = q)
  (hbc : ∃ (q : ℚ), b * c = q)
  (hca : ∃ (q : ℚ), c * a = q) :
  ∃ (x y z : ℤ), (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_integer_combination_l2398_239845


namespace NUMINAMATH_CALUDE_abs_squared_minus_two_abs_minus_fifteen_solution_set_l2398_239853

theorem abs_squared_minus_two_abs_minus_fifteen_solution_set :
  {x : ℝ | |x|^2 - 2*|x| - 15 > 0} = {x : ℝ | x < -5 ∨ x > 5} := by
  sorry

end NUMINAMATH_CALUDE_abs_squared_minus_two_abs_minus_fifteen_solution_set_l2398_239853


namespace NUMINAMATH_CALUDE_area_intersection_approx_l2398_239815

/-- The elliptical region D₁ -/
def D₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 ≤ 1

/-- The circular region D₂ -/
def D₂ (x y : ℝ) : Prop := x^2 + y^2 ≤ 2

/-- The intersection of D₁ and D₂ -/
def D_intersection (x y : ℝ) : Prop := D₁ x y ∧ D₂ x y

/-- The area of the intersection of D₁ and D₂ -/
noncomputable def area_intersection : ℝ := sorry

theorem area_intersection_approx :
  abs (area_intersection - 5.88) < 0.01 := by sorry

end NUMINAMATH_CALUDE_area_intersection_approx_l2398_239815


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisor_l2398_239849

def is_valid_seven_digit_number (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧
  (n / 100 % 10 = 2 * (n / 1000000)) ∧
  (n / 10 % 10 = 2 * (n / 100000 % 10)) ∧
  (n % 10 = 2 * (n / 10000 % 10)) ∧
  (n / 1000 % 10 = 0)

theorem smallest_five_digit_divisor :
  ∃ (n : ℕ), is_valid_seven_digit_number n ∧ n % 10002 = 0 ∧
  ∀ (m : ℕ), 10000 ≤ m ∧ m < 10002 → ¬(∃ (k : ℕ), is_valid_seven_digit_number k ∧ k % m = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisor_l2398_239849


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l2398_239896

theorem polynomial_division_quotient (z : ℝ) : 
  ((5/4 : ℝ) * z^4 - (23/16 : ℝ) * z^3 + (129/64 : ℝ) * z^2 - (353/256 : ℝ) * z + 949/1024) * (4 * z + 1) = 
  5 * z^5 - 3 * z^4 + 4 * z^3 - 7 * z^2 + 9 * z - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l2398_239896


namespace NUMINAMATH_CALUDE_expression_simplification_l2398_239837

theorem expression_simplification (x y : ℚ) 
  (hx : x = -3/8) (hy : y = 4) : 
  (x - 2*y)^2 + (x - 2*y)*(x + 2*y) - 2*x*(x - y) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2398_239837


namespace NUMINAMATH_CALUDE_power_division_equality_l2398_239895

theorem power_division_equality : (3 : ℕ)^16 / (81 : ℕ)^2 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l2398_239895


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l2398_239846

def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  (∀ a : ℝ, C a ⊆ (A ∪ B) → 2 ≤ a ∧ a ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l2398_239846


namespace NUMINAMATH_CALUDE_art_count_l2398_239835

/-- The number of Asian art pieces seen -/
def asian_art : ℕ := 465

/-- The number of Egyptian art pieces seen -/
def egyptian_art : ℕ := 527

/-- The total number of art pieces seen -/
def total_art : ℕ := asian_art + egyptian_art

theorem art_count : total_art = 992 := by
  sorry

end NUMINAMATH_CALUDE_art_count_l2398_239835


namespace NUMINAMATH_CALUDE_greatest_integer_problem_l2398_239833

theorem greatest_integer_problem (n : ℕ) : n < 50 ∧
  (∃ a : ℤ, n = 6 * a - 1) ∧
  (∃ b : ℤ, n = 8 * b - 5) ∧
  (∃ c : ℤ, n = 3 * c + 2) ∧
  (∀ m : ℕ, m < 50 →
    (∃ a : ℤ, m = 6 * a - 1) →
    (∃ b : ℤ, m = 8 * b - 5) →
    (∃ c : ℤ, m = 3 * c + 2) →
    m ≤ n) →
  n = 41 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_problem_l2398_239833


namespace NUMINAMATH_CALUDE_focus_of_specific_parabola_l2398_239871

/-- A parabola is defined by the equation y^2 = -4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = -4 * p.1}

/-- The focus of a parabola is a point from which all points on the parabola are equidistant -/
def FocusOfParabola (p : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

theorem focus_of_specific_parabola :
  FocusOfParabola Parabola = (-1, 0) := by sorry

end NUMINAMATH_CALUDE_focus_of_specific_parabola_l2398_239871


namespace NUMINAMATH_CALUDE_shirt_original_price_l2398_239814

/-- Calculates the original price of an item given its discounted price and discount percentage. -/
def originalPrice (discountedPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  discountedPrice / (1 - discountPercentage / 100)

/-- Theorem stating that if a shirt is sold at Rs. 780 after a 20% discount, 
    then the original price of the shirt was Rs. 975. -/
theorem shirt_original_price : 
  originalPrice 780 20 = 975 := by
  sorry

end NUMINAMATH_CALUDE_shirt_original_price_l2398_239814


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l2398_239820

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

theorem relay_race_arrangements : permutations 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l2398_239820


namespace NUMINAMATH_CALUDE_amp_composition_l2398_239855

-- Define the operations
def amp (x : ℤ) : ℤ := 10 - x
def amp_prefix (x : ℤ) : ℤ := x - 10

-- State the theorem
theorem amp_composition : amp_prefix (amp 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_amp_composition_l2398_239855


namespace NUMINAMATH_CALUDE_one_four_one_not_reappear_l2398_239825

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

def next_numbers (n : ℕ) : Set ℕ :=
  {n + digit_product n, n - digit_product n}

def reachable_numbers (start : ℕ) : Set ℕ :=
  {n | ∃ (seq : ℕ → ℕ), seq 0 = start ∧ ∀ i, seq (i + 1) ∈ next_numbers (seq i)}

theorem one_four_one_not_reappear : 141 ∉ reachable_numbers 141 \ {141} := by
  sorry

end NUMINAMATH_CALUDE_one_four_one_not_reappear_l2398_239825


namespace NUMINAMATH_CALUDE_second_concert_attendance_l2398_239884

theorem second_concert_attendance 
  (first_concert : ℕ) 
  (attendance_increase : ℕ) 
  (h1 : first_concert = 65899)
  (h2 : attendance_increase = 119) :
  first_concert + attendance_increase = 66018 :=
by sorry

end NUMINAMATH_CALUDE_second_concert_attendance_l2398_239884


namespace NUMINAMATH_CALUDE_governors_addresses_l2398_239874

/-- The number of commencement addresses given by Governor Sandoval -/
def sandoval_addresses : ℕ := 12

/-- The number of commencement addresses given by Governor Hawkins -/
def hawkins_addresses : ℕ := sandoval_addresses / 2

/-- The number of commencement addresses given by Governor Sloan -/
def sloan_addresses : ℕ := sandoval_addresses + 10

/-- The total number of commencement addresses given by all three governors -/
def total_addresses : ℕ := sandoval_addresses + hawkins_addresses + sloan_addresses

theorem governors_addresses : total_addresses = 40 := by
  sorry

end NUMINAMATH_CALUDE_governors_addresses_l2398_239874


namespace NUMINAMATH_CALUDE_compare_with_negative_three_sevenths_l2398_239890

theorem compare_with_negative_three_sevenths :
  let a : ℚ := 1
  let b : ℚ := -8/21
  let c : ℚ := 0
  let d : ℚ := -43/100
  let target : ℚ := -3/7
  (a > target) ∧ (b > target) ∧ (c > target) ∧ (d < target) :=
by sorry

end NUMINAMATH_CALUDE_compare_with_negative_three_sevenths_l2398_239890


namespace NUMINAMATH_CALUDE_problem_solution_l2398_239826

theorem problem_solution (x y : ℝ) :
  y = (Real.sqrt (x^2 - 4) + Real.sqrt (4 - x^2) + 1) / (x - 2) →
  3 * x + 4 * y = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2398_239826


namespace NUMINAMATH_CALUDE_music_class_participation_l2398_239809

theorem music_class_participation (jacob_total : ℕ) (jacob_participating : ℕ) (steve_total : ℕ)
  (h1 : jacob_total = 27)
  (h2 : jacob_participating = 18)
  (h3 : steve_total = 45) :
  (jacob_participating * steve_total) / jacob_total = 30 := by
  sorry

end NUMINAMATH_CALUDE_music_class_participation_l2398_239809


namespace NUMINAMATH_CALUDE_largest_integer_inequality_l2398_239885

theorem largest_integer_inequality (x : ℤ) : x ≤ 4 ↔ x / 3 + 3 / 4 < 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_inequality_l2398_239885


namespace NUMINAMATH_CALUDE_total_toys_l2398_239894

theorem total_toys (jaxon_toys gabriel_toys jerry_toys : ℕ) : 
  jaxon_toys = 15 →
  gabriel_toys = 2 * jaxon_toys →
  jerry_toys = gabriel_toys + 8 →
  jaxon_toys + gabriel_toys + jerry_toys = 83 := by
sorry

end NUMINAMATH_CALUDE_total_toys_l2398_239894


namespace NUMINAMATH_CALUDE_radius_is_3_sqrt_13_l2398_239892

/-- Represents a circular sector with an inscribed rectangle -/
structure CircularSectorWithRectangle where
  /-- Radius of the circle -/
  radius : ℝ
  /-- Central angle of the sector in radians -/
  centralAngle : ℝ
  /-- Length of the shorter side of the rectangle -/
  shortSide : ℝ
  /-- Length of the longer side of the rectangle -/
  longSide : ℝ
  /-- The longer side is 3 units longer than the shorter side -/
  sideDifference : longSide = shortSide + 3
  /-- The area of the rectangle is 18 -/
  rectangleArea : shortSide * longSide = 18
  /-- The central angle is 45 degrees (π/4 radians) -/
  angleIs45Degrees : centralAngle = Real.pi / 4

/-- The main theorem stating that the radius is 3√13 -/
theorem radius_is_3_sqrt_13 (sector : CircularSectorWithRectangle) :
  sector.radius = 3 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_radius_is_3_sqrt_13_l2398_239892


namespace NUMINAMATH_CALUDE_average_age_after_leaving_l2398_239803

def initial_average : ℝ := 40
def initial_count : ℕ := 8
def leaving_age : ℝ := 25
def final_count : ℕ := 7

theorem average_age_after_leaving :
  let initial_total_age := initial_average * initial_count
  let remaining_total_age := initial_total_age - leaving_age
  let final_average := remaining_total_age / final_count
  final_average = 42 := by sorry

end NUMINAMATH_CALUDE_average_age_after_leaving_l2398_239803


namespace NUMINAMATH_CALUDE_divisible_by_seventeen_l2398_239823

theorem divisible_by_seventeen (k : ℕ) : 
  17 ∣ (2^(2*k + 3) + 3^(k + 2) * 7^k) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_seventeen_l2398_239823


namespace NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l2398_239800

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- Number of cubic yards we're converting -/
def cubic_yards : ℝ := 4

/-- Theorem stating that 4 cubic yards equals 108 cubic feet -/
theorem cubic_yards_to_cubic_feet : 
  cubic_yards * (yards_to_feet ^ 3) = 108 := by sorry

end NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l2398_239800


namespace NUMINAMATH_CALUDE_no_functions_satisfying_condition_l2398_239880

theorem no_functions_satisfying_condition :
  ¬ (∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), x ≠ y → |f x - f y| + |g x - g y| > 1) :=
by sorry

end NUMINAMATH_CALUDE_no_functions_satisfying_condition_l2398_239880


namespace NUMINAMATH_CALUDE_unique_function_exists_l2398_239876

theorem unique_function_exists : ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y + 1) = x + y := by
  sorry

end NUMINAMATH_CALUDE_unique_function_exists_l2398_239876


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2398_239881

theorem quadratic_equations_solutions :
  (∀ x : ℝ, x^2 - 4*x - 1 = 0 ↔ x = 2 - Real.sqrt 5 ∨ x = 2 + Real.sqrt 5) ∧
  (∀ x : ℝ, 3*x^2 - 5*x + 1 = 0 ↔ x = (5 - Real.sqrt 13) / 6 ∨ x = (5 + Real.sqrt 13) / 6) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2398_239881


namespace NUMINAMATH_CALUDE_non_congruent_triangles_count_l2398_239822

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A 2x4 array of points -/
def PointArray : Array (Array Point) :=
  #[#[{x := 0, y := 0}, {x := 1, y := 0}, {x := 2, y := 0}, {x := 3, y := 0}],
    #[{x := 0, y := 1}, {x := 1, y := 1}, {x := 2, y := 1}, {x := 3, y := 1}]]

/-- Check if two triangles are congruent -/
def are_congruent (t1 t2 : Array Point) : Prop := sorry

/-- Count non-congruent triangles in the point array -/
def count_non_congruent_triangles (arr : Array (Array Point)) : ℕ := sorry

/-- Theorem: The number of non-congruent triangles in the given 2x4 array is 3 -/
theorem non_congruent_triangles_count :
  count_non_congruent_triangles PointArray = 3 := by sorry

end NUMINAMATH_CALUDE_non_congruent_triangles_count_l2398_239822


namespace NUMINAMATH_CALUDE_airport_gate_probability_l2398_239844

/-- The number of gates in the airport --/
def num_gates : ℕ := 15

/-- The distance between adjacent gates in feet --/
def gate_distance : ℕ := 90

/-- The maximum walking distance in feet --/
def max_distance : ℕ := 450

/-- The probability of selecting two gates within the maximum distance --/
def probability : ℚ := 10 / 21

theorem airport_gate_probability :
  let total_pairs := num_gates * (num_gates - 1)
  let valid_pairs := (num_gates - max_distance / gate_distance) * (max_distance / gate_distance)
    + 2 * (max_distance / gate_distance * (max_distance / gate_distance + 1) / 2)
  (valid_pairs : ℚ) / total_pairs = probability := by sorry

end NUMINAMATH_CALUDE_airport_gate_probability_l2398_239844


namespace NUMINAMATH_CALUDE_fraction_sum_equation_l2398_239862

theorem fraction_sum_equation (x y : ℝ) (h1 : x ≠ y) 
  (h2 : x / y + (x + 6 * y) / (y + 6 * x) = 3) : 
  x / y = (8 + Real.sqrt 46) / 6 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_equation_l2398_239862


namespace NUMINAMATH_CALUDE_total_jumps_l2398_239842

def hattie_first_round : ℕ := 180

def lorelei_first_round : ℕ := (3 * hattie_first_round) / 4

def hattie_second_round : ℕ := (2 * hattie_first_round) / 3

def lorelei_second_round : ℕ := hattie_second_round + 50

def hattie_third_round : ℕ := hattie_second_round + hattie_second_round / 3

def lorelei_third_round : ℕ := (4 * lorelei_first_round) / 5

theorem total_jumps :
  hattie_first_round + lorelei_first_round +
  hattie_second_round + lorelei_second_round +
  hattie_third_round + lorelei_third_round = 873 := by
  sorry

end NUMINAMATH_CALUDE_total_jumps_l2398_239842


namespace NUMINAMATH_CALUDE_marion_score_l2398_239828

theorem marion_score (total_items : Nat) (ella_incorrect : Nat) (marion_additional : Nat) :
  total_items = 40 →
  ella_incorrect = 4 →
  marion_additional = 6 →
  (total_items - ella_incorrect) / 2 + marion_additional = 24 := by
  sorry

end NUMINAMATH_CALUDE_marion_score_l2398_239828


namespace NUMINAMATH_CALUDE_min_triangles_for_G_2008_l2398_239831

/-- Represents a point in a 2D grid --/
structure GridPoint where
  x : Nat
  y : Nat

/-- Defines the grid G_n --/
def G (n : Nat) : Set GridPoint :=
  {p : GridPoint | p.x ≥ 1 ∧ p.x ≤ n ∧ p.y ≥ 1 ∧ p.y ≤ n}

/-- Minimum number of triangles needed to cover a grid --/
def minTriangles (n : Nat) : Nat :=
  if n = 2 then 1
  else if n = 3 then 2
  else (n * n) / 3 * 2

/-- Theorem stating the minimum number of triangles needed to cover G_2008 --/
theorem min_triangles_for_G_2008 :
  minTriangles 2008 = 1338 :=
sorry

end NUMINAMATH_CALUDE_min_triangles_for_G_2008_l2398_239831


namespace NUMINAMATH_CALUDE_triangle_area_l2398_239852

/-- The area of a triangle with base 2 and height 3 is 3 -/
theorem triangle_area : 
  let base : ℝ := 2
  let height : ℝ := 3
  let area := (base * height) / 2
  area = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l2398_239852


namespace NUMINAMATH_CALUDE_equidistant_point_y_axis_l2398_239875

theorem equidistant_point_y_axis (y : ℝ) : 
  (∀ (x : ℝ), x = 0 → 
    (x - 3)^2 + y^2 = (x - 5)^2 + (y - 6)^2) → 
  y = 13/3 := by sorry

end NUMINAMATH_CALUDE_equidistant_point_y_axis_l2398_239875


namespace NUMINAMATH_CALUDE_position_determination_in_plane_l2398_239867

theorem position_determination_in_plane :
  ∀ (P : ℝ × ℝ), ∃! (θ : ℝ) (r : ℝ), 
    P.1 = r * Real.cos θ ∧ P.2 = r * Real.sin θ ∧ r ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_position_determination_in_plane_l2398_239867


namespace NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l2398_239808

/-- Represents the number of tickets Tom has -/
structure TomTickets :=
  (yellow : ℕ)
  (red : ℕ)
  (blue : ℕ)

/-- Calculates the number of additional blue tickets needed to win the Bible -/
def additional_blue_tickets_needed (tickets : TomTickets) : ℕ :=
  let yellow_to_blue := 100
  let red_to_blue := 10
  let total_blue_needed := 10 * yellow_to_blue
  let blue_from_yellow := tickets.yellow * yellow_to_blue
  let blue_from_red := tickets.red * red_to_blue
  let blue_total := blue_from_yellow + blue_from_red + tickets.blue
  total_blue_needed - blue_total

/-- Theorem stating that Tom needs 163 more blue tickets to win the Bible -/
theorem tom_needs_163_blue_tickets :
  additional_blue_tickets_needed ⟨8, 3, 7⟩ = 163 := by
  sorry


end NUMINAMATH_CALUDE_tom_needs_163_blue_tickets_l2398_239808


namespace NUMINAMATH_CALUDE_melted_sphere_radius_l2398_239819

theorem melted_sphere_radius (r : ℝ) : 
  r > 0 → (4 / 3 * Real.pi * r^3 = 8 * (4 / 3 * Real.pi * 1^3)) → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_melted_sphere_radius_l2398_239819


namespace NUMINAMATH_CALUDE_area_of_overlapping_squares_area_of_overlapping_squares_is_216_l2398_239813

/-- The area of the region covered by two congruent squares with side length 12 units,
    where the center of one square coincides with a vertex of the other square. -/
theorem area_of_overlapping_squares : ℝ :=
  let square_side_length : ℝ := 12
  let square_area : ℝ := square_side_length ^ 2
  let total_area : ℝ := 2 * square_area
  let overlap_area : ℝ := square_area / 2
  total_area - overlap_area

/-- The area of the region covered by two congruent squares with side length 12 units,
    where the center of one square coincides with a vertex of the other square, is 216 square units. -/
theorem area_of_overlapping_squares_is_216 : area_of_overlapping_squares = 216 := by
  sorry

end NUMINAMATH_CALUDE_area_of_overlapping_squares_area_of_overlapping_squares_is_216_l2398_239813


namespace NUMINAMATH_CALUDE_five_digit_number_formation_l2398_239860

/-- A two-digit number is between 10 and 99, inclusive -/
def TwoDigitNumber (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A three-digit number is between 100 and 999, inclusive -/
def ThreeDigitNumber (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The five-digit number formed by placing x to the left of y -/
def FiveDigitNumber (x y : ℕ) : ℕ := 1000 * x + y

theorem five_digit_number_formation (x y : ℕ) 
  (hx : TwoDigitNumber x) (hy : ThreeDigitNumber y) :
  FiveDigitNumber x y = 1000 * x + y := by
  sorry

end NUMINAMATH_CALUDE_five_digit_number_formation_l2398_239860


namespace NUMINAMATH_CALUDE_cos_18_deg_l2398_239888

theorem cos_18_deg (h : Real.cos (72 * π / 180) = (Real.sqrt 5 - 1) / 4) :
  Real.cos (18 * π / 180) = Real.sqrt (5 + Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_18_deg_l2398_239888


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_l2398_239868

/-- A point P with coordinates (m, 4+2m) is in the third quadrant if and only if m < -2 -/
theorem point_in_third_quadrant (m : ℝ) : 
  (m < 0 ∧ 4 + 2*m < 0) ↔ m < -2 :=
sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_l2398_239868


namespace NUMINAMATH_CALUDE_gcf_of_90_and_135_l2398_239805

theorem gcf_of_90_and_135 : Nat.gcd 90 135 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_90_and_135_l2398_239805


namespace NUMINAMATH_CALUDE_total_situps_is_110_l2398_239811

/-- The number of situps Diana did -/
def diana_situps : ℕ := 40

/-- The rate at which Diana did situps (situps per minute) -/
def diana_rate : ℕ := 4

/-- The difference in situps per minute between Hani and Diana -/
def hani_extra_rate : ℕ := 3

/-- Theorem stating that the total number of situps Hani and Diana did together is 110 -/
theorem total_situps_is_110 : 
  diana_situps + (diana_rate + hani_extra_rate) * (diana_situps / diana_rate) = 110 := by
  sorry

end NUMINAMATH_CALUDE_total_situps_is_110_l2398_239811


namespace NUMINAMATH_CALUDE_circle_diameter_endpoint_l2398_239866

/-- Given a circle with center (1, -2) and one endpoint of a diameter at (4, 3),
    the other endpoint of the diameter is at (7, 3). -/
theorem circle_diameter_endpoint :
  let center : ℝ × ℝ := (1, -2)
  let endpoint1 : ℝ × ℝ := (4, 3)
  let endpoint2 : ℝ × ℝ := (7, 3)
  (endpoint1.1 - center.1 = center.1 - endpoint2.1 ∧
   endpoint1.2 - center.2 = center.2 - endpoint2.2) :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_endpoint_l2398_239866


namespace NUMINAMATH_CALUDE_soda_preference_l2398_239893

/-- Given a survey of 520 people and a central angle of 144° for "Soda" preference
    in a pie chart, prove that 208 people favor "Soda". -/
theorem soda_preference (total : ℕ) (angle : ℝ) (h1 : total = 520) (h2 : angle = 144) :
  (angle / 360 : ℝ) * total = 208 := by
  sorry

end NUMINAMATH_CALUDE_soda_preference_l2398_239893


namespace NUMINAMATH_CALUDE_same_color_pair_count_l2398_239802

/-- The number of ways to choose a pair of socks of the same color -/
def choose_same_color_pair (green red purple : ℕ) : ℕ :=
  Nat.choose green 2 + Nat.choose red 2 + Nat.choose purple 2

/-- Theorem stating that choosing a pair of socks of the same color from 
    5 green, 6 red, and 4 purple socks results in 31 possibilities -/
theorem same_color_pair_count : choose_same_color_pair 5 6 4 = 31 := by
  sorry

end NUMINAMATH_CALUDE_same_color_pair_count_l2398_239802


namespace NUMINAMATH_CALUDE_three_number_set_range_l2398_239801

theorem three_number_set_range (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- Ascending order
  a = 2 ∧  -- Smallest number is 2
  b = 5 ∧  -- Median is 5
  (a + b + c) / 3 = 5 →  -- Mean is 5
  c - a = 6 :=  -- Range is 6
by sorry

end NUMINAMATH_CALUDE_three_number_set_range_l2398_239801


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l2398_239832

def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_value_of_f_on_interval :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧ f x = 5 ∧ ∀ (y : ℝ), y ∈ Set.Icc 0 3 → f y ≤ f x :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l2398_239832


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2398_239879

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) :
  z.im = (Real.sqrt 2 + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2398_239879


namespace NUMINAMATH_CALUDE_triangle_cos_C_l2398_239869

theorem triangle_cos_C (A B C : Real) (h1 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h2 : A + B + C = Real.pi) (h3 : Real.sin A = 4/5) (h4 : Real.cos B = 3/5) : 
  Real.cos C = 7/25 := by
sorry

end NUMINAMATH_CALUDE_triangle_cos_C_l2398_239869


namespace NUMINAMATH_CALUDE_min_ties_for_twelve_pairs_l2398_239812

/-- Represents the minimum number of ties needed to guarantee a certain number of pairs -/
def min_ties_for_pairs (num_pairs : ℕ) : ℕ :=
  5 + 2 * (num_pairs - 1)

/-- Theorem stating the minimum number of ties needed for 12 pairs -/
theorem min_ties_for_twelve_pairs :
  min_ties_for_pairs 12 = 27 :=
by sorry

end NUMINAMATH_CALUDE_min_ties_for_twelve_pairs_l2398_239812


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l2398_239810

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 21.875

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 35

/-- Theorem stating that the weight of one bowling ball is 21.875 pounds -/
theorem bowling_ball_weight_proof :
  (8 * bowling_ball_weight = 5 * canoe_weight) ∧
  (4 * canoe_weight = 140) →
  bowling_ball_weight = 21.875 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l2398_239810


namespace NUMINAMATH_CALUDE_water_experiment_proof_l2398_239889

/-- Calculates the remaining amount of water after an experiment -/
def remaining_water (initial : ℚ) (used : ℚ) : ℚ :=
  initial - used

/-- Proves that given 3 gallons of water and using 5/4 gallons, the remaining amount is 7/4 gallons -/
theorem water_experiment_proof :
  remaining_water 3 (5/4) = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_water_experiment_proof_l2398_239889


namespace NUMINAMATH_CALUDE_mark_paid_54_l2398_239883

/-- The total amount Mark paid for hiring a singer -/
def total_paid (hours : ℕ) (rate : ℚ) (tip_percentage : ℚ) : ℚ :=
  let base_cost := hours * rate
  let tip := base_cost * tip_percentage
  base_cost + tip

/-- Theorem stating that Mark paid $54 for hiring the singer -/
theorem mark_paid_54 :
  total_paid 3 15 (20 / 100) = 54 := by
  sorry

end NUMINAMATH_CALUDE_mark_paid_54_l2398_239883


namespace NUMINAMATH_CALUDE_airport_distance_is_130_l2398_239847

/-- Represents the problem of calculating the distance to the airport --/
def AirportDistance (initial_speed : ℝ) (speed_increase : ℝ) (initial_delay : ℝ) (actual_early : ℝ) : Prop :=
  ∃ (distance : ℝ) (time : ℝ),
    distance = initial_speed * (time + 1) ∧
    distance - initial_speed = (initial_speed + speed_increase) * (time - actual_early) ∧
    distance = 130

/-- The theorem stating that the distance to the airport is 130 miles --/
theorem airport_distance_is_130 :
  AirportDistance 40 20 1 0.25 := by
  sorry

end NUMINAMATH_CALUDE_airport_distance_is_130_l2398_239847


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2398_239838

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  populationSize : Nat
  sampleSize : Nat
  knownSamples : List Nat

/-- Calculates the sampling interval for a systematic sample -/
def samplingInterval (s : SystematicSample) : Nat :=
  s.populationSize / s.sampleSize

/-- Checks if a given number is part of the systematic sample -/
def isInSample (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, n = (s.knownSamples.head!) + k * samplingInterval s

/-- The theorem to be proved -/
theorem systematic_sample_fourth_element 
  (s : SystematicSample) 
  (h1 : s.populationSize = 56)
  (h2 : s.sampleSize = 4)
  (h3 : s.knownSamples = [6, 34, 48]) :
  isInSample s 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_element_l2398_239838


namespace NUMINAMATH_CALUDE_no_solution_exists_l2398_239897

theorem no_solution_exists : ¬∃ (x a z b : ℕ), 
  (0 < x) ∧ (x < 10) ∧ 
  (0 < a) ∧ (a < 10) ∧ 
  (0 < z) ∧ (z < 10) ∧ 
  (0 < b) ∧ (b < 10) ∧ 
  (4 * x = a) ∧ 
  (4 * z = b) ∧ 
  (x^2 + a^2 = z^2 + b^2) ∧ 
  ((x + a)^3 > (z + b)^3) :=
sorry

end NUMINAMATH_CALUDE_no_solution_exists_l2398_239897


namespace NUMINAMATH_CALUDE_r_fourth_plus_inverse_l2398_239807

theorem r_fourth_plus_inverse (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_r_fourth_plus_inverse_l2398_239807


namespace NUMINAMATH_CALUDE_train_passengers_l2398_239878

/-- The number of people on a train after three stops -/
def people_after_three_stops (initial : ℕ) 
  (stop1_off stop1_on : ℕ) 
  (stop2_off stop2_on : ℕ) 
  (stop3_off stop3_on : ℕ) : ℕ :=
  initial - stop1_off + stop1_on - stop2_off + stop2_on - stop3_off + stop3_on

/-- Theorem stating the number of people on the train after three stops -/
theorem train_passengers : 
  people_after_three_stops 48 12 7 15 9 6 11 = 42 := by
  sorry

end NUMINAMATH_CALUDE_train_passengers_l2398_239878


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l2398_239886

theorem mango_rate_calculation (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (total_paid : ℕ) : 
  grape_quantity = 8 →
  grape_rate = 70 →
  mango_quantity = 8 →
  total_paid = 1000 →
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 := by
sorry

#eval (1000 - 8 * 70) / 8  -- Expected output: 55

end NUMINAMATH_CALUDE_mango_rate_calculation_l2398_239886


namespace NUMINAMATH_CALUDE_rachel_envelope_stuffing_l2398_239827

/-- Rachel's envelope stuffing problem -/
theorem rachel_envelope_stuffing
  (total_hours : ℕ)
  (total_envelopes : ℕ)
  (second_hour_envelopes : ℕ)
  (required_rate : ℕ)
  (h1 : total_hours = 8)
  (h2 : total_envelopes = 1500)
  (h3 : second_hour_envelopes = 141)
  (h4 : required_rate = 204) :
  total_envelopes - (required_rate * (total_hours - 2)) - second_hour_envelopes = 135 := by
  sorry

#check rachel_envelope_stuffing

end NUMINAMATH_CALUDE_rachel_envelope_stuffing_l2398_239827


namespace NUMINAMATH_CALUDE_anthony_total_pencils_l2398_239858

def initial_pencils : ℕ := 9
def gifted_pencils : ℕ := 56

theorem anthony_total_pencils : 
  initial_pencils + gifted_pencils = 65 := by
  sorry

end NUMINAMATH_CALUDE_anthony_total_pencils_l2398_239858


namespace NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l2398_239863

noncomputable def f (a b x : ℝ) : ℝ := (x - 2) * (a * x + b)

theorem solution_set_of_even_increasing_function 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_incr : ∀ x y, 0 < x → x < y → f a b x < f a b y) :
  ∀ x, f a b (2 - x) > 0 ↔ x < 0 ∨ x > 4 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l2398_239863


namespace NUMINAMATH_CALUDE_number_of_arrangements_l2398_239865

-- Define the number of male volunteers
def num_male : Nat := 4

-- Define the number of female volunteers
def num_female : Nat := 2

-- Define the number of elderly people
def num_elderly : Nat := 2

-- Define the total number of people
def total_people : Nat := num_male + num_female + num_elderly

-- Define the function to calculate the number of arrangements
def calculate_arrangements (n_male : Nat) (n_female : Nat) (n_elderly : Nat) : Nat :=
  -- Treat elderly people as one unit
  let n_units := n_male + 1
  -- Calculate arrangements of units
  let unit_arrangements := Nat.factorial n_units
  -- Calculate arrangements of elderly people themselves
  let elderly_arrangements := Nat.factorial n_elderly
  -- Calculate arrangements of female volunteers in the spaces between and around other people
  let female_arrangements := (n_units + 1) * n_units
  unit_arrangements * elderly_arrangements * female_arrangements

-- Theorem statement
theorem number_of_arrangements :
  calculate_arrangements num_male num_female num_elderly = 7200 := by
  sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l2398_239865


namespace NUMINAMATH_CALUDE_multiplication_puzzle_l2398_239872

theorem multiplication_puzzle (c d : ℕ) : 
  c < 10 → d < 10 → (30 + c) * (10 * d + 4) = 146 → c + d = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_puzzle_l2398_239872


namespace NUMINAMATH_CALUDE_inequality_proof_l2398_239898

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + b * c + c * a + 2 * a * b * c = 1) :
  Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2398_239898


namespace NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l2398_239817

/-- Given a square with side length s ≥ 4 containing a 2x2 square, 
    a 2x4 rectangle, and a non-overlapping rectangle R, 
    the area of R is exactly 4. -/
theorem area_of_inscribed_rectangle (s : ℝ) (h_s : s ≥ 4) : 
  s^2 - (2 * 2 + 2 * 4) = 4 := by sorry

end NUMINAMATH_CALUDE_area_of_inscribed_rectangle_l2398_239817


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_5_l2398_239859

theorem sqrt_meaningful_iff_x_geq_5 (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 5) ↔ x ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_5_l2398_239859


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2398_239836

theorem complex_number_in_first_quadrant : 
  let z : ℂ := (3 - I) / (1 - I)
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2398_239836


namespace NUMINAMATH_CALUDE_total_road_cost_l2398_239851

/-- Represents the dimensions of a rectangular lawn -/
structure LawnDimensions where
  length : ℕ
  width : ℕ

/-- Represents a road segment with its length and cost per square meter -/
structure RoadSegment where
  length : ℕ
  cost_per_sqm : ℕ

/-- Calculates the total cost of a road given its segments and width -/
def road_cost (segments : List RoadSegment) (width : ℕ) : ℕ :=
  segments.foldl (fun acc segment => acc + segment.length * segment.cost_per_sqm * width) 0

/-- The main theorem stating the total cost of traveling the two roads -/
theorem total_road_cost (lawn : LawnDimensions)
  (length_road : List RoadSegment) (breadth_road : List RoadSegment) (road_width : ℕ) :
  lawn.length = 100 ∧ lawn.width = 60 ∧
  road_width = 10 ∧
  length_road = [⟨30, 4⟩, ⟨40, 5⟩, ⟨30, 6⟩] ∧
  breadth_road = [⟨20, 3⟩, ⟨40, 2⟩] →
  road_cost length_road road_width + road_cost breadth_road road_width = 6400 := by
  sorry

end NUMINAMATH_CALUDE_total_road_cost_l2398_239851


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l2398_239840

theorem triangle_perimeter_bound : 
  ∀ s : ℝ, s > 0 → 7 + s > 25 → 25 + s > 7 → 7 + 25 + s < 65 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l2398_239840


namespace NUMINAMATH_CALUDE_least_integer_x_l2398_239882

theorem least_integer_x (x : ℤ) : (∀ y : ℤ, |3 * y + 5| ≤ 21 → y ≥ -8) ∧ |3 * (-8) + 5| ≤ 21 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_x_l2398_239882


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l2398_239848

/-- Represents a painted cube that can be cut into smaller cubes -/
structure PaintedCube where
  edge : ℕ  -- Edge length of the large cube
  small_edge : ℕ  -- Edge length of the smaller cubes

/-- Counts the number of smaller cubes with exactly one painted face -/
def count_one_face_painted (cube : PaintedCube) : ℕ :=
  6 * (cube.edge - 2) * (cube.edge - 2)

/-- Counts the number of smaller cubes with exactly two painted faces -/
def count_two_faces_painted (cube : PaintedCube) : ℕ :=
  12 * (cube.edge - 2)

theorem painted_cube_theorem (cube : PaintedCube) 
  (h1 : cube.edge = 10) 
  (h2 : cube.small_edge = 1) : 
  count_one_face_painted cube = 384 ∧ count_two_faces_painted cube = 96 := by
  sorry

#eval count_one_face_painted ⟨10, 1⟩
#eval count_two_faces_painted ⟨10, 1⟩

end NUMINAMATH_CALUDE_painted_cube_theorem_l2398_239848


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l2398_239843

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_square_one_plus_i : (1 + i)^2 = 2*i := by sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l2398_239843
