import Mathlib

namespace NUMINAMATH_CALUDE_exists_maximal_element_l2387_238787

/-- A family of subsets of ℕ satisfying the chain condition -/
structure ChainFamily where
  C : Set (Set ℕ)
  chain_condition : ∀ (chain : ℕ → Set ℕ), (∀ n m, n ≤ m → chain n ⊆ chain m) →
    (∀ n, chain n ∈ C) → ∃ S ∈ C, ∀ n, chain n ⊆ S

/-- The existence of a maximal element in a chain family -/
theorem exists_maximal_element (F : ChainFamily) :
  ∃ S ∈ F.C, ∀ T ∈ F.C, S ⊆ T → S = T := by sorry

end NUMINAMATH_CALUDE_exists_maximal_element_l2387_238787


namespace NUMINAMATH_CALUDE_college_students_count_l2387_238772

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 135) :
  boys + girls = 351 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l2387_238772


namespace NUMINAMATH_CALUDE_mushroom_collection_proof_l2387_238797

theorem mushroom_collection_proof :
  ∃ (x₁ x₂ x₃ x₄ : ℕ),
    x₁ + x₂ = 6 ∧
    x₁ + x₃ = 7 ∧
    x₁ + x₄ = 9 ∧
    x₂ + x₃ = 9 ∧
    x₂ + x₄ = 11 ∧
    x₃ + x₄ = 12 ∧
    x₁ = 2 ∧
    x₂ = 4 ∧
    x₃ = 5 ∧
    x₄ = 7 := by
  sorry

end NUMINAMATH_CALUDE_mushroom_collection_proof_l2387_238797


namespace NUMINAMATH_CALUDE_equal_numbers_product_l2387_238737

theorem equal_numbers_product (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 25 →
  c = 18 →
  d = e →
  d * e = 506.25 := by
  sorry

end NUMINAMATH_CALUDE_equal_numbers_product_l2387_238737


namespace NUMINAMATH_CALUDE_power_58_digits_l2387_238792

theorem power_58_digits (n : ℤ) :
  ¬ (10^63 ≤ n^58 ∧ n^58 < 10^64) ∧
  ∀ k : ℕ, k ≤ 81 → ¬ (10^(k-1) ≤ n^58 ∧ n^58 < 10^k) ∧
  ∃ m : ℤ, 10^81 ≤ m^58 ∧ m^58 < 10^82 :=
by sorry

end NUMINAMATH_CALUDE_power_58_digits_l2387_238792


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2387_238765

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 10 → y = 68 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2387_238765


namespace NUMINAMATH_CALUDE_solution_set_of_inequalities_l2387_238716

theorem solution_set_of_inequalities :
  let S := {x : ℝ | x - 1 < 0 ∧ |x| < 2}
  S = {x : ℝ | -2 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequalities_l2387_238716


namespace NUMINAMATH_CALUDE_donation_problem_l2387_238719

theorem donation_problem (total_donation_A total_donation_B : ℝ)
  (percent_more : ℝ) (diff_avg_donation : ℝ)
  (h1 : total_donation_A = 1200)
  (h2 : total_donation_B = 1200)
  (h3 : percent_more = 0.2)
  (h4 : diff_avg_donation = 5) :
  ∃ (students_A students_B : ℕ),
    students_A = 48 ∧ 
    students_B = 40 ∧
    students_A = (1 + percent_more) * students_B ∧
    (total_donation_B / students_B) - (total_donation_A / students_A) = diff_avg_donation :=
by
  sorry


end NUMINAMATH_CALUDE_donation_problem_l2387_238719


namespace NUMINAMATH_CALUDE_smallest_odd_five_primes_l2387_238788

def is_prime (n : ℕ) : Prop := sorry

def has_exactly_five_prime_factors (n : ℕ) : Prop := sorry

def smallest_odd_with_five_prime_factors : ℕ := 15015

theorem smallest_odd_five_primes :
  has_exactly_five_prime_factors smallest_odd_with_five_prime_factors ∧
  ∀ m : ℕ, m < smallest_odd_with_five_prime_factors →
    ¬(has_exactly_five_prime_factors m ∧ Odd m) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_five_primes_l2387_238788


namespace NUMINAMATH_CALUDE_power_of_three_expression_equals_zero_l2387_238769

theorem power_of_three_expression_equals_zero :
  3^2003 - 5 * 3^2002 + 6 * 3^2001 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_expression_equals_zero_l2387_238769


namespace NUMINAMATH_CALUDE_sixth_term_of_special_sequence_l2387_238759

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem sixth_term_of_special_sequence :
  ∀ (a : ℕ → ℝ),
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 = 2 →
  a 3 = 2 →
  a 4 = 2 →
  a 5 = 2 →
  a 6 = 2 :=
by sorry

end NUMINAMATH_CALUDE_sixth_term_of_special_sequence_l2387_238759


namespace NUMINAMATH_CALUDE_birthday_crayons_proof_l2387_238727

/-- The number of crayons Paul got for his birthday. -/
def birthday_crayons : ℕ := 253

/-- The number of crayons Paul lost or gave away. -/
def lost_crayons : ℕ := 70

/-- The number of crayons Paul had left by the end of the school year. -/
def remaining_crayons : ℕ := 183

/-- Theorem stating that the number of crayons Paul got for his birthday
    is equal to the sum of lost crayons and remaining crayons. -/
theorem birthday_crayons_proof :
  birthday_crayons = lost_crayons + remaining_crayons :=
by sorry

end NUMINAMATH_CALUDE_birthday_crayons_proof_l2387_238727


namespace NUMINAMATH_CALUDE_greatest_integer_difference_l2387_238713

theorem greatest_integer_difference (x y : ℝ) (hx : 3 < x ∧ x < 6) (hy : 6 < y ∧ y < 8) :
  ∃ (n : ℕ), n = 3 ∧ ∀ (m : ℕ), (∃ (a b : ℝ), 3 < a ∧ a < 6 ∧ 6 < b ∧ b < 8 ∧ m = ⌊b - a⌋) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_difference_l2387_238713


namespace NUMINAMATH_CALUDE_complex_fraction_real_l2387_238750

theorem complex_fraction_real (a : ℝ) : 
  (((1 : ℂ) + a * I) / ((2 : ℂ) + I)).im = 0 → a = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_real_l2387_238750


namespace NUMINAMATH_CALUDE_total_driving_hours_l2387_238739

/-- Carl's driving schedule --/
structure DrivingSchedule :=
  (mon : ℕ) (tue : ℕ) (wed : ℕ) (thu : ℕ) (fri : ℕ)

/-- Calculate total hours for a week --/
def weeklyHours (s : DrivingSchedule) : ℕ :=
  s.mon + s.tue + s.wed + s.thu + s.fri

/-- Carl's normal schedule --/
def normalSchedule : DrivingSchedule :=
  ⟨2, 3, 4, 2, 5⟩

/-- Carl's schedule after promotion --/
def promotedSchedule : DrivingSchedule :=
  ⟨3, 5, 7, 6, 5⟩

/-- Carl's schedule for the second week with two days off --/
def secondWeekSchedule : DrivingSchedule :=
  ⟨3, 5, 0, 0, 5⟩

theorem total_driving_hours :
  weeklyHours promotedSchedule + weeklyHours secondWeekSchedule = 39 := by
  sorry

#eval weeklyHours promotedSchedule + weeklyHours secondWeekSchedule

end NUMINAMATH_CALUDE_total_driving_hours_l2387_238739


namespace NUMINAMATH_CALUDE_equation_represents_two_hyperbolas_l2387_238786

-- Define the equation
def equation (x y : ℝ) : Prop :=
  y^4 - 6*x^4 = 3*y^2 - 2

-- Define what a hyperbola equation looks like
def is_hyperbola_equation (a b c : ℝ) (x y : ℝ) : Prop :=
  y^2 - a*x^2 = c ∧ b ≠ 0

-- Theorem statement
theorem equation_represents_two_hyperbolas :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x y : ℝ, equation x y ↔ 
      (is_hyperbola_equation a₁ b₁ c₁ x y ∨ is_hyperbola_equation a₂ b₂ c₂ x y)) ∧
    b₁ ≠ 0 ∧ b₂ ≠ 0 ∧ (a₁ ≠ a₂ ∨ c₁ ≠ c₂) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_hyperbolas_l2387_238786


namespace NUMINAMATH_CALUDE_no_valid_pairs_l2387_238780

theorem no_valid_pairs : 
  ¬∃ (a b x y : ℤ), 
    (a * x + b * y = 3) ∧ 
    (x^2 + y^2 = 85) ∧ 
    (3 * a - 5 * b = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_pairs_l2387_238780


namespace NUMINAMATH_CALUDE_problem_statement_l2387_238747

theorem problem_statement (f : ℝ → ℝ) : 
  (∀ x, f x = (x^4 + 2*x^3 + 4*x - 5)^2004 + 2004) →
  f (Real.sqrt 3 - 1) = 2005 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2387_238747


namespace NUMINAMATH_CALUDE_pencil_distribution_l2387_238740

theorem pencil_distribution (total_pencils : ℕ) (num_students : ℕ) 
  (h1 : total_pencils = 125) (h2 : num_students = 25) :
  total_pencils / num_students = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l2387_238740


namespace NUMINAMATH_CALUDE_power_of_power_three_l2387_238708

theorem power_of_power_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_l2387_238708


namespace NUMINAMATH_CALUDE_loan_interest_period_l2387_238705

/-- The problem of determining the number of years for B's gain --/
theorem loan_interest_period (principal : ℝ) (rate_A rate_C : ℝ) (gain : ℝ) : 
  principal = 1500 →
  rate_A = 0.10 →
  rate_C = 0.115 →
  gain = 67.5 →
  (rate_C - rate_A) * principal * 3 = gain :=
by sorry

end NUMINAMATH_CALUDE_loan_interest_period_l2387_238705


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l2387_238725

/-- If the solution set of x² - mx - 6n < 0 is {x | -3 < x < 6}, then m + n = 6 -/
theorem solution_set_implies_sum (m n : ℝ) : 
  (∀ x, x^2 - m*x - 6*n < 0 ↔ -3 < x ∧ x < 6) → 
  m + n = 6 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l2387_238725


namespace NUMINAMATH_CALUDE_major_axis_length_l2387_238782

/-- Represents an ellipse formed by a plane intersecting a right circular cylinder -/
structure CylinderEllipse where
  cylinder_radius : ℝ
  major_axis : ℝ
  minor_axis : ℝ

/-- The theorem stating the length of the major axis given the conditions -/
theorem major_axis_length (e : CylinderEllipse) 
  (h1 : e.cylinder_radius = 2)
  (h2 : e.minor_axis = 2 * e.cylinder_radius)
  (h3 : e.major_axis = e.minor_axis * 1.6) :
  e.major_axis = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_l2387_238782


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l2387_238704

theorem sum_of_cubes_zero (x y : ℝ) (h1 : x + y = 0) (h2 : x * y = -1) : x^3 + y^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l2387_238704


namespace NUMINAMATH_CALUDE_combined_population_l2387_238702

/-- The combined population of Port Perry and Lazy Harbor given the specified conditions -/
theorem combined_population (wellington_pop : ℕ) (port_perry_pop : ℕ) (lazy_harbor_pop : ℕ) 
  (h1 : port_perry_pop = 7 * wellington_pop)
  (h2 : port_perry_pop = lazy_harbor_pop + 800)
  (h3 : wellington_pop = 900) : 
  port_perry_pop + lazy_harbor_pop = 11800 := by
  sorry

end NUMINAMATH_CALUDE_combined_population_l2387_238702


namespace NUMINAMATH_CALUDE_concyclic_roots_l2387_238715

theorem concyclic_roots (m : ℝ) : 
  (∀ x : ℂ, (x^2 - 2*x + 2 = 0 ∨ x^2 + 2*m*x + 1 = 0) → 
    (∃ (a b r : ℝ), ∀ y : ℂ, (y^2 - 2*y + 2 = 0 ∨ y^2 + 2*m*y + 1 = 0) → 
      (y.re - a)^2 + (y.im - b)^2 = r^2)) ↔ 
  (-1 < m ∧ m < 1) ∨ m = -3/2 := by
sorry

end NUMINAMATH_CALUDE_concyclic_roots_l2387_238715


namespace NUMINAMATH_CALUDE_consecutive_interior_equal_parallel_false_l2387_238707

-- Define the concept of lines
variable (Line : Type)

-- Define the concept of angles
variable (Angle : Type)

-- Define what it means for lines to be parallel
variable (parallel : Line → Line → Prop)

-- Define what it means for angles to be consecutive interior angles
variable (consecutive_interior : Angle → Angle → Line → Line → Prop)

-- Define what it means for angles to be equal
variable (angle_equal : Angle → Angle → Prop)

-- Statement to be proven false
theorem consecutive_interior_equal_parallel_false :
  ¬(∀ (l1 l2 : Line) (a1 a2 : Angle), 
    consecutive_interior a1 a2 l1 l2 → angle_equal a1 a2 → parallel l1 l2) :=
sorry

end NUMINAMATH_CALUDE_consecutive_interior_equal_parallel_false_l2387_238707


namespace NUMINAMATH_CALUDE_square_triangle_area_equality_l2387_238761

theorem square_triangle_area_equality (square_perimeter : ℝ) (triangle_height : ℝ) (x : ℝ) :
  square_perimeter = 64 →
  triangle_height = 48 →
  (square_perimeter / 4) ^ 2 = (1 / 2) * triangle_height * x →
  x = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_square_triangle_area_equality_l2387_238761


namespace NUMINAMATH_CALUDE_a_sixth_bounds_l2387_238736

-- Define the condition
def condition (a : ℝ) : Prop := a^5 - a^3 + a = 2

-- State the theorem
theorem a_sixth_bounds {a : ℝ} (h : condition a) : 3 < a^6 ∧ a^6 < 4 := by
  sorry

end NUMINAMATH_CALUDE_a_sixth_bounds_l2387_238736


namespace NUMINAMATH_CALUDE_arithmetic_error_correction_l2387_238756

theorem arithmetic_error_correction : ∃! x : ℝ, 3 * x - 4 = x / 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_error_correction_l2387_238756


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2387_238744

/-- The range of m for which the quadratic inequality mx^2 - mx + 1 < 0 has a non-empty solution set -/
theorem quadratic_inequality_solution_range :
  {m : ℝ | ∃ x, m * x^2 - m * x + 1 < 0} = {m | m < 0 ∨ m > 4} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l2387_238744


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_120_and_m_l2387_238714

theorem greatest_common_divisor_of_120_and_m (m : ℕ) : 
  (∃ d₁ d₂ d₃ d₄ : ℕ, d₁ < d₂ ∧ d₂ < d₃ ∧ d₃ < d₄ ∧ 
    {d : ℕ | d ∣ 120 ∧ d ∣ m} = {d₁, d₂, d₃, d₄}) →
  Nat.gcd 120 m = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_120_and_m_l2387_238714


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2387_238754

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : ∀ x, quadratic_function a b c (-x + 1) = quadratic_function a b c (x + 1))
  (h2 : quadratic_function a b c 2 = 0)
  (h3 : ∃! x, quadratic_function a b c x = x) :
  a = -1/2 ∧ b = 1 ∧ c = 0 ∧
  ∃ m n : ℝ, m = -4 ∧ n = 0 ∧
    (∀ x, m ≤ x ∧ x ≤ n → 3*m ≤ quadratic_function a b c x ∧ quadratic_function a b c x ≤ 3*n) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2387_238754


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2387_238701

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value < 1000

/-- Returns the two-digit number formed by removing the first digit -/
def remove_first_digit (n : ThreeDigitNumber) : ℕ :=
  n.value % 100

/-- Checks if a three-digit number satisfies the division condition -/
def satisfies_division_condition (n : ThreeDigitNumber) : Prop :=
  let two_digit := remove_first_digit n
  n.value / two_digit = 8 ∧ n.value % two_digit = 6

theorem unique_three_digit_number :
  ∃! n : ThreeDigitNumber, satisfies_division_condition n ∧ n.value = 342 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2387_238701


namespace NUMINAMATH_CALUDE_sequence_formula_l2387_238799

-- Define the sequence a_n
def a : ℕ → ℝ := sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ := 2 * n^2 + n

-- Theorem statement
theorem sequence_formula (n : ℕ) : a n = 4 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l2387_238799


namespace NUMINAMATH_CALUDE_sum_of_integers_with_lcm_gcd_l2387_238766

theorem sum_of_integers_with_lcm_gcd (m n : ℕ) : 
  m > 50 → 
  n > 50 → 
  Nat.lcm m n = 480 → 
  Nat.gcd m n = 12 → 
  m + n = 156 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_with_lcm_gcd_l2387_238766


namespace NUMINAMATH_CALUDE_circle_center_coordinates_l2387_238774

/-- The center of a circle satisfying given conditions -/
theorem circle_center_coordinates : ∃ (x y : ℝ),
  (x - 2 * y = 0) ∧
  (3 * x + 4 * y = 10) ∧
  (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_l2387_238774


namespace NUMINAMATH_CALUDE_modulo_thirteen_equivalence_l2387_238722

theorem modulo_thirteen_equivalence : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 13 ∧ 52801 ≡ n [ZMOD 13] ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_modulo_thirteen_equivalence_l2387_238722


namespace NUMINAMATH_CALUDE_probability_of_one_hit_l2387_238721

/-- Represents a single shot result -/
inductive Shot
| Hit
| Miss

/-- Represents the result of three shots -/
structure ThreeShots :=
  (first second third : Shot)

/-- Counts the number of hits in a ThreeShots -/
def count_hits (shots : ThreeShots) : Nat :=
  match shots with
  | ⟨Shot.Hit, Shot.Hit, Shot.Hit⟩ => 3
  | ⟨Shot.Hit, Shot.Hit, Shot.Miss⟩ => 2
  | ⟨Shot.Hit, Shot.Miss, Shot.Hit⟩ => 2
  | ⟨Shot.Miss, Shot.Hit, Shot.Hit⟩ => 2
  | ⟨Shot.Hit, Shot.Miss, Shot.Miss⟩ => 1
  | ⟨Shot.Miss, Shot.Hit, Shot.Miss⟩ => 1
  | ⟨Shot.Miss, Shot.Miss, Shot.Hit⟩ => 1
  | ⟨Shot.Miss, Shot.Miss, Shot.Miss⟩ => 0

/-- Converts a digit to a Shot -/
def digit_to_shot (d : Nat) : Shot :=
  if d ∈ [1, 2, 3, 4] then Shot.Hit else Shot.Miss

/-- Converts a three-digit number to ThreeShots -/
def number_to_three_shots (n : Nat) : ThreeShots :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  ⟨digit_to_shot d1, digit_to_shot d2, digit_to_shot d3⟩

theorem probability_of_one_hit (data : List Nat) : 
  data.length = 20 →
  (data.filter (fun n => count_hits (number_to_three_shots n) = 1)).length = 9 →
  (data.filter (fun n => count_hits (number_to_three_shots n) = 1)).length / data.length = 9 / 20 :=
sorry

end NUMINAMATH_CALUDE_probability_of_one_hit_l2387_238721


namespace NUMINAMATH_CALUDE_cubic_polynomial_root_l2387_238731

theorem cubic_polynomial_root (a b : ℝ) : 
  (∃ (x : ℂ), x^3 + a*x^2 - x + b = 0 ∧ x = 2 - 3*I) → 
  (a = 7.5 ∧ b = -45.5) := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_root_l2387_238731


namespace NUMINAMATH_CALUDE_two_inequalities_l2387_238748

theorem two_inequalities :
  (∀ a b c : ℝ, a^2 + b^2 + c^2 ≥ a*b + a*c + b*c) ∧
  (Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_two_inequalities_l2387_238748


namespace NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_unit_cube_l2387_238703

/-- The surface area of a sphere that circumscribes a cube with edge length 1 is 3π. -/
theorem sphere_surface_area_circumscribing_unit_cube : 
  ∃ (S : ℝ), S = 3 * Real.pi ∧ 
  (∃ (r : ℝ), r > 0 ∧ 
    -- The radius is half the length of the cube's space diagonal
    r = (Real.sqrt 3) / 2 ∧ 
    -- The surface area formula
    S = 4 * Real.pi * r^2) := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_circumscribing_unit_cube_l2387_238703


namespace NUMINAMATH_CALUDE_min_value_xy_plus_x_squared_l2387_238700

theorem min_value_xy_plus_x_squared (x y : ℝ) (h1 : x * y > 0) (h2 : x^2 * y = 2) :
  x * y + x^2 ≥ 4 ∧ (x * y + x^2 = 4 ↔ y = 1 ∧ x = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_xy_plus_x_squared_l2387_238700


namespace NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2387_238733

theorem geometric_arithmetic_sequence_problem 
  (a b : ℕ → ℝ)
  (h_geometric : ∀ n : ℕ, a (n + 2) / a (n + 1) = a (n + 1) / a n)
  (h_arithmetic : ∀ n : ℕ, b (n + 2) - b (n + 1) = b (n + 1) - b n)
  (h_a_product : a 1 * a 5 * a 9 = -8)
  (h_b_sum : b 2 + b 5 + b 8 = 6 * Real.pi)
  : Real.cos ((b 4 + b 6) / (1 - a 3 * a 7)) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_arithmetic_sequence_problem_l2387_238733


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2387_238711

theorem complex_equation_solution : ∃ (a : ℝ) (b c : ℂ),
  a + b + c = 5 ∧
  a * b + b * c + c * a = 7 ∧
  a * b * c = 3 ∧
  (a = 1 ∨ a = 3) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2387_238711


namespace NUMINAMATH_CALUDE_system_solution_l2387_238778

/-- Given a system of equations and a partial solution, prove the complete solution -/
theorem system_solution (a : ℝ) :
  (∃ x y : ℝ, 2*x + y = a ∧ 2*x - y = 12 ∧ x = 5) →
  (∃ x y : ℝ, 2*x + y = a ∧ 2*x - y = 12 ∧ x = 5 ∧ y = -2 ∧ a = 8) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2387_238778


namespace NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l2387_238757

/-- Represents a 24-hour digital clock with a minute display error -/
structure ErrorClock where
  /-- The number of hours in a day -/
  hours_per_day : ℕ
  /-- The number of minutes in an hour -/
  minutes_per_hour : ℕ
  /-- The number of minutes with display error per hour -/
  error_minutes_per_hour : ℕ

/-- The fraction of the day the clock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : ℚ :=
  (clock.hours_per_day * (clock.minutes_per_hour - clock.error_minutes_per_hour)) /
  (clock.hours_per_day * clock.minutes_per_hour)

/-- Theorem stating the correct time fraction for the given clock -/
theorem error_clock_correct_time_fraction :
  let clock : ErrorClock := {
    hours_per_day := 24,
    minutes_per_hour := 60,
    error_minutes_per_hour := 1
  }
  correct_time_fraction clock = 59 / 60 := by
  sorry

end NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l2387_238757


namespace NUMINAMATH_CALUDE_juice_fraction_is_one_fourth_l2387_238785

/-- Represents the contents of a cup -/
structure CupContents where
  milk : ℚ
  juice : ℚ

/-- Represents the state of both cups -/
structure TwoCups where
  cup1 : CupContents
  cup2 : CupContents

def initial_state : TwoCups := {
  cup1 := { milk := 6, juice := 0 },
  cup2 := { milk := 0, juice := 6 }
}

def transfer_milk (state : TwoCups) : TwoCups := {
  cup1 := { milk := state.cup1.milk * 2/3, juice := state.cup1.juice },
  cup2 := { milk := state.cup2.milk + state.cup1.milk * 1/3, juice := state.cup2.juice }
}

def transfer_mixture (state : TwoCups) : TwoCups :=
  let total2 := state.cup2.milk + state.cup2.juice
  let transfer_amount := total2 * 1/4
  let milk_fraction := state.cup2.milk / total2
  let juice_fraction := state.cup2.juice / total2
  {
    cup1 := {
      milk := state.cup1.milk + transfer_amount * milk_fraction,
      juice := state.cup1.juice + transfer_amount * juice_fraction
    },
    cup2 := {
      milk := state.cup2.milk - transfer_amount * milk_fraction,
      juice := state.cup2.juice - transfer_amount * juice_fraction
    }
  }

def final_state : TwoCups :=
  transfer_mixture (transfer_milk initial_state)

theorem juice_fraction_is_one_fourth :
  (final_state.cup1.juice) / (final_state.cup1.milk + final_state.cup1.juice) = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_juice_fraction_is_one_fourth_l2387_238785


namespace NUMINAMATH_CALUDE_x_range_l2387_238796

theorem x_range (x : ℝ) : (1 / x < 4 ∧ 1 / x > -2) → (x < -1/2 ∨ x > 1/4) := by
  sorry

end NUMINAMATH_CALUDE_x_range_l2387_238796


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2387_238770

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3) ∧
  (∃ x : ℝ, 2*x^2 - 7*x + 5 = 0 ↔ x = 5/2 ∨ x = 1) ∧
  (∃ x : ℝ, (x + 3)^2 - 2*(x + 3) = 0 ↔ x = -3 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2387_238770


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2387_238798

theorem expression_simplification_and_evaluation (a : ℝ) 
  (h1 : a ≠ 1) (h2 : a ≠ 2) (h3 : a ≠ -2) :
  (1 - 3 / (a + 2)) / ((a^2 - 2*a + 1) / (a^2 - 4)) = (a - 2) / (a - 1) ∧
  (0 - 2) / (0 - 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l2387_238798


namespace NUMINAMATH_CALUDE_multiple_of_z_l2387_238793

theorem multiple_of_z (x y z k : ℕ+) : 
  (3 * x.val = 4 * y.val) → 
  (3 * x.val = k * z.val) → 
  (x.val - y.val + z.val = 19) → 
  (∀ (x' y' z' : ℕ+), 3 * x'.val = 4 * y'.val → 3 * x'.val = k * z'.val → x'.val - y'.val + z'.val ≥ 19) →
  k = 12 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_z_l2387_238793


namespace NUMINAMATH_CALUDE_monkey_escape_time_l2387_238760

/-- Proves that a monkey running at 15 feet/second for t seconds, then swinging at 10 feet/second for 10 seconds, covering 175 feet total, ran for 5 seconds. -/
theorem monkey_escape_time (run_speed : ℝ) (swing_speed : ℝ) (swing_time : ℝ) (total_distance : ℝ) :
  run_speed = 15 →
  swing_speed = 10 →
  swing_time = 10 →
  total_distance = 175 →
  ∃ t : ℝ, t * run_speed + swing_time * swing_speed = total_distance ∧ t = 5 :=
by
  sorry

#check monkey_escape_time

end NUMINAMATH_CALUDE_monkey_escape_time_l2387_238760


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l2387_238728

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 12) / (Nat.factorial 5)) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l2387_238728


namespace NUMINAMATH_CALUDE_baseball_game_attendance_difference_proof_baseball_game_attendance_difference_l2387_238724

theorem baseball_game_attendance_difference : ℕ → Prop :=
  fun difference =>
    ∀ (second_game_attendance : ℕ)
      (first_game_attendance : ℕ)
      (third_game_attendance : ℕ)
      (last_week_total : ℕ),
    second_game_attendance = 80 →
    first_game_attendance = second_game_attendance - 20 →
    third_game_attendance = second_game_attendance + 15 →
    last_week_total = 200 →
    difference = (first_game_attendance + second_game_attendance + third_game_attendance) - last_week_total →
    difference = 35

-- The proof of the theorem
theorem proof_baseball_game_attendance_difference : 
  baseball_game_attendance_difference 35 := by
  sorry

end NUMINAMATH_CALUDE_baseball_game_attendance_difference_proof_baseball_game_attendance_difference_l2387_238724


namespace NUMINAMATH_CALUDE_difference_of_reciprocals_l2387_238718

theorem difference_of_reciprocals (p q : ℚ) : 
  3 / p = 6 → 3 / q = 18 → p - q = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_reciprocals_l2387_238718


namespace NUMINAMATH_CALUDE_copper_alloy_percentages_l2387_238768

theorem copper_alloy_percentages
  (x y : ℝ)  -- Percentages of copper in first and second alloys
  (m₁ m₂ : ℝ)  -- Masses of first and second alloys
  (h₁ : y = x + 40)  -- First alloy's copper percentage is 40% less than the second
  (h₂ : x * m₁ / 100 = 6)  -- First alloy contains 6 kg of copper
  (h₃ : y * m₂ / 100 = 12)  -- Second alloy contains 12 kg of copper
  (h₄ : 36 * (m₁ + m₂) / 100 = 18)  -- Mixture contains 36% copper
  : x = 20 ∧ y = 60 := by
  sorry

end NUMINAMATH_CALUDE_copper_alloy_percentages_l2387_238768


namespace NUMINAMATH_CALUDE_degree_of_polynomial_l2387_238741

-- Define the polynomial
def p (a b : ℝ) : ℝ := 3 * a^2 - a * b^2 + 2 * a^2 - 3^4

-- Theorem statement
theorem degree_of_polynomial :
  ∃ (n : ℕ), n = 3 ∧ 
  (∀ (m : ℕ), (∃ (a b : ℝ), p a b ≠ 0 ∧ 
    (∀ (c d : ℝ), a^m * b^(n-m) = c^m * d^(n-m) → p a b = p c d)) →
  (∀ (k : ℕ), k > n → 
    (∀ (a b : ℝ), ∃ (c d : ℝ), a^k * b^(n-k) = c^k * d^(n-k) ∧ p a b = p c d))) :=
sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_l2387_238741


namespace NUMINAMATH_CALUDE_inequality_proof_l2387_238783

theorem inequality_proof (k m n : ℕ) (hk : k > 0) (hm : m > 0) (hn : n > 0) 
  (hkm : k ≠ m) (hkn : k ≠ n) (hmn : m ≠ n) : 
  (k - 1 / k) * (m - 1 / m) * (n - 1 / n) ≤ k * m * n - (k + m + n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2387_238783


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2387_238753

/-- A line in the form kx - y + 1 = 3k passes through the point (3, 1) for all values of k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), k * 3 - 1 + 1 = 3 * k := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l2387_238753


namespace NUMINAMATH_CALUDE_yang_hui_theorem_l2387_238763

theorem yang_hui_theorem (a b : ℝ) 
  (sum : a + b = 3)
  (product : a * b = 1)
  (sum_squares : a^2 + b^2 = 7)
  (sum_cubes : a^3 + b^3 = 18)
  (sum_fourth_powers : a^4 + b^4 = 47) :
  a^5 + b^5 = 123 := by sorry

end NUMINAMATH_CALUDE_yang_hui_theorem_l2387_238763


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_l2387_238791

/-- Given an angle α whose terminal side passes through the point P(4a,-3a) where a < 0,
    prove that 2sin α + cos α = 2/5 -/
theorem angle_terminal_side_point (a : ℝ) (α : ℝ) (h : a < 0) :
  let x : ℝ := 4 * a
  let y : ℝ := -3 * a
  let r : ℝ := Real.sqrt (x^2 + y^2)
  2 * (y / r) + (x / r) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_l2387_238791


namespace NUMINAMATH_CALUDE_max_integer_k_l2387_238717

theorem max_integer_k (x y k : ℝ) : 
  x - 4*y = k - 1 →
  2*x + y = k →
  x - y ≤ 0 →
  ∀ m : ℤ, m ≤ k → m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_k_l2387_238717


namespace NUMINAMATH_CALUDE_eighth_triangular_number_l2387_238771

/-- Triangular number function -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 8th triangular number is 36 -/
theorem eighth_triangular_number : triangular_number 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_eighth_triangular_number_l2387_238771


namespace NUMINAMATH_CALUDE_unique_magnitude_of_complex_roots_l2387_238746

theorem unique_magnitude_of_complex_roots : 
  ∃! r : ℝ, ∃ z : ℂ, z^2 - 4*z + 29 = 0 ∧ Complex.abs z = r :=
sorry

end NUMINAMATH_CALUDE_unique_magnitude_of_complex_roots_l2387_238746


namespace NUMINAMATH_CALUDE_parabola_directrix_l2387_238773

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y = (x^2 - 2*x + 1) / 8

/-- The directrix equation -/
def directrix (y : ℝ) : Prop := y = -2

/-- Theorem: The directrix of the given parabola is y = -2 -/
theorem parabola_directrix : ∀ x y : ℝ, parabola x y → ∃ d : ℝ, directrix d ∧ 
  (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    (p.1 - x)^2 + (p.2 - y)^2 = (p.1 - x)^2 + (p.2 - d)^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l2387_238773


namespace NUMINAMATH_CALUDE_regular_triangle_counts_l2387_238777

/-- Regular triangle with sides divided into n segments -/
structure RegularTriangle (n : ℕ) where
  -- Add any necessary fields

/-- Count of regular triangles in a RegularTriangle -/
def countRegularTriangles (t : RegularTriangle n) : ℕ :=
  if n % 2 = 0 then
    (n * (n + 2) * (2 * n + 1)) / 8
  else
    ((n + 1) * (2 * n^2 + 3 * n - 1)) / 8

/-- Count of rhombuses in a RegularTriangle -/
def countRhombuses (t : RegularTriangle n) : ℕ :=
  if n % 2 = 0 then
    (n * (n + 2) * (2 * n - 1)) / 8
  else
    ((n - 1) * (n + 1) * (2 * n + 3)) / 8

/-- Theorem stating the counts are correct -/
theorem regular_triangle_counts (n : ℕ) (t : RegularTriangle n) :
  (countRegularTriangles t = if n % 2 = 0 then (n * (n + 2) * (2 * n + 1)) / 8
                             else ((n + 1) * (2 * n^2 + 3 * n - 1)) / 8) ∧
  (countRhombuses t = if n % 2 = 0 then (n * (n + 2) * (2 * n - 1)) / 8
                      else ((n - 1) * (n + 1) * (2 * n + 3)) / 8) :=
by sorry

end NUMINAMATH_CALUDE_regular_triangle_counts_l2387_238777


namespace NUMINAMATH_CALUDE_white_balls_count_l2387_238729

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  green = 30 →
  yellow = 10 →
  red = 7 →
  purple = 3 →
  prob_not_red_purple = 9/10 →
  total - (green + yellow + red + purple) = 50 := by
  sorry

#check white_balls_count

end NUMINAMATH_CALUDE_white_balls_count_l2387_238729


namespace NUMINAMATH_CALUDE_rhombus_not_necessarily_planar_l2387_238794

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A shape in 3D space -/
class Shape where
  vertices : List Point3D

/-- A triangle is always planar -/
def Triangle (a b c : Point3D) : Shape :=
  { vertices := [a, b, c] }

/-- A trapezoid is always planar -/
def Trapezoid (a b c d : Point3D) : Shape :=
  { vertices := [a, b, c, d] }

/-- A parallelogram is always planar -/
def Parallelogram (a b c d : Point3D) : Shape :=
  { vertices := [a, b, c, d] }

/-- A rhombus (quadrilateral with equal sides) -/
def Rhombus (a b c d : Point3D) : Shape :=
  { vertices := [a, b, c, d] }

/-- Predicate to check if a shape is planar -/
def isPlanar (s : Shape) : Prop :=
  sorry

/-- Theorem stating that a rhombus is not necessarily planar -/
theorem rhombus_not_necessarily_planar :
  ∃ (a b c d : Point3D), ¬(isPlanar (Rhombus a b c d)) ∧
    (∀ (x y z : Point3D), isPlanar (Triangle x y z)) ∧
    (∀ (w x y z : Point3D), isPlanar (Trapezoid w x y z)) ∧
    (∀ (w x y z : Point3D), isPlanar (Parallelogram w x y z)) :=
  sorry

end NUMINAMATH_CALUDE_rhombus_not_necessarily_planar_l2387_238794


namespace NUMINAMATH_CALUDE_circles_properties_l2387_238734

-- Define the circles O and M
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_M (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 2*y + 1 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_M A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_M B.1 B.2 ∧
  A ≠ B

-- Define the theorem
theorem circles_properties 
  (A B : ℝ × ℝ) 
  (h : intersection_points A B) :
  (∃ (T1 T2 : ℝ × ℝ), T1 ≠ T2 ∧ 
    (∀ x y, circle_O x y → (x - T1.1) * T1.1 + (y - T1.2) * T1.2 = 0) ∧
    (∀ x y, circle_M x y → (x - T1.1) * T1.1 + (y - T1.2) * T1.2 = 0) ∧
    (∀ x y, circle_O x y → (x - T2.1) * T2.1 + (y - T2.2) * T2.2 = 0) ∧
    (∀ x y, circle_M x y → (x - T2.1) * T2.1 + (y - T2.2) * T2.2 = 0)) ∧
  (∀ x y, circle_O x y ↔ circle_M (2*A.1 - x) (2*A.2 - y)) ∧
  (∃ E F : ℝ × ℝ, circle_O E.1 E.2 ∧ circle_M F.1 F.2 ∧
    ∀ E' F' : ℝ × ℝ, circle_O E'.1 E'.2 → circle_M F'.1 F'.2 →
      (E.1 - F.1)^2 + (E.2 - F.2)^2 ≥ (E'.1 - F'.1)^2 + (E'.2 - F'.2)^2) ∧
  (∃ E F : ℝ × ℝ, circle_O E.1 E.2 ∧ circle_M F.1 F.2 ∧
    (E.1 - F.1)^2 + (E.2 - F.2)^2 = (4 + Real.sqrt 5)^2) :=
sorry

end NUMINAMATH_CALUDE_circles_properties_l2387_238734


namespace NUMINAMATH_CALUDE_casey_nail_coats_l2387_238735

/-- The time it takes to apply and dry one coat of nail polish -/
def coat_time : ℕ := 20 + 20

/-- The total time spent on decorating nails -/
def total_time : ℕ := 120

/-- The number of coats applied to each nail -/
def num_coats : ℕ := total_time / coat_time

theorem casey_nail_coats : num_coats = 3 := by
  sorry

end NUMINAMATH_CALUDE_casey_nail_coats_l2387_238735


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2387_238720

theorem smallest_integer_satisfying_inequality :
  ∀ y : ℤ, (y : ℚ) / 4 + 3 / 7 > 4 / 7 ↔ y ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l2387_238720


namespace NUMINAMATH_CALUDE_absolute_value_plus_power_l2387_238738

theorem absolute_value_plus_power : |-5| + 2^0 = 6 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_plus_power_l2387_238738


namespace NUMINAMATH_CALUDE_smallest_distance_between_circle_and_ellipse_l2387_238784

theorem smallest_distance_between_circle_and_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let ellipse := {p : ℝ × ℝ | ((p.1 - 2)^2 / 9) + (p.2^2 / 16) = 1}
  ∃ (d : ℝ), d = (Real.sqrt 35 - 2) / 2 ∧
    (∀ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ circle → p₂ ∈ ellipse →
      Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) ≥ d) ∧
    (∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ), p₁ ∈ circle ∧ p₂ ∈ ellipse ∧
      Real.sqrt ((p₁.1 - p₂.1)^2 + (p₁.2 - p₂.2)^2) = d) :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circle_and_ellipse_l2387_238784


namespace NUMINAMATH_CALUDE_product_adjacent_faces_is_144_l2387_238723

/-- Represents a face of the cube --/
structure Face :=
  (number : Nat)

/-- Represents the cube formed from the numbered net --/
structure Cube :=
  (faces : List Face)
  (adjacent_to_one : List Face)
  (h_adjacent : adjacent_to_one.length = 4)

/-- The product of the numbers on the faces adjacent to face 1 --/
def product_adjacent_faces (c : Cube) : Nat :=
  c.adjacent_to_one.map Face.number |>.foldl (· * ·) 1

/-- Theorem stating that the product of numbers on faces adjacent to face 1 is 144 --/
theorem product_adjacent_faces_is_144 (c : Cube) 
  (h_adjacent_numbers : c.adjacent_to_one.map Face.number = [2, 3, 4, 6]) :
  product_adjacent_faces c = 144 := by
  sorry

end NUMINAMATH_CALUDE_product_adjacent_faces_is_144_l2387_238723


namespace NUMINAMATH_CALUDE_impossible_partition_l2387_238712

theorem impossible_partition : ¬ ∃ (A B C : Finset ℕ),
  (A ∪ B ∪ C = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (A ∩ B = ∅) ∧ (B ∩ C = ∅) ∧ (A ∩ C = ∅) ∧
  (Finset.card A = 3) ∧ (Finset.card B = 3) ∧ (Finset.card C = 3) ∧
  (∃ (a₁ a₂ a₃ : ℕ), A = {a₁, a₂, a₃} ∧ max a₁ (max a₂ a₃) = a₁ + a₂ + a₃ - max a₁ (max a₂ a₃)) ∧
  (∃ (b₁ b₂ b₃ : ℕ), B = {b₁, b₂, b₃} ∧ max b₁ (max b₂ b₃) = b₁ + b₂ + b₃ - max b₁ (max b₂ b₃)) ∧
  (∃ (c₁ c₂ c₃ : ℕ), C = {c₁, c₂, c₃} ∧ max c₁ (max c₂ c₃) = c₁ + c₂ + c₃ - max c₁ (max c₂ c₃)) :=
by
  sorry


end NUMINAMATH_CALUDE_impossible_partition_l2387_238712


namespace NUMINAMATH_CALUDE_nonagon_coloring_theorem_l2387_238755

/-- A type representing the colors used to color the nonagon vertices -/
inductive Color
| A
| B
| C

/-- A type representing the vertices of a regular nonagon -/
inductive Vertex
| One | Two | Three | Four | Five | Six | Seven | Eight | Nine

/-- A function type representing a coloring of the nonagon -/
def Coloring := Vertex → Color

/-- Predicate to check if two vertices are adjacent in a regular nonagon -/
def adjacent (v1 v2 : Vertex) : Prop := sorry

/-- Predicate to check if three vertices form an equilateral triangle in a regular nonagon -/
def equilateralTriangle (v1 v2 v3 : Vertex) : Prop := sorry

/-- Predicate to check if a coloring is valid according to the given conditions -/
def validColoring (c : Coloring) : Prop :=
  (∀ v1 v2, adjacent v1 v2 → c v1 ≠ c v2) ∧
  (∀ v1 v2 v3, equilateralTriangle v1 v2 v3 → c v1 ≠ c v2 ∧ c v1 ≠ c v3 ∧ c v2 ≠ c v3)

/-- The minimum number of colors needed for a valid coloring -/
def m : Nat := 3

/-- The total number of valid colorings using m colors -/
def n : Nat := 18

/-- The main theorem stating that the product of m and n is 54 -/
theorem nonagon_coloring_theorem : m * n = 54 := by sorry

end NUMINAMATH_CALUDE_nonagon_coloring_theorem_l2387_238755


namespace NUMINAMATH_CALUDE_max_non_managers_l2387_238730

theorem max_non_managers (managers : ℕ) (non_managers : ℕ) : 
  managers = 8 →
  (managers : ℚ) / non_managers > 5 / 24 →
  non_managers ≤ 38 :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_l2387_238730


namespace NUMINAMATH_CALUDE_sqrt_3_irrational_l2387_238776

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_irrational_l2387_238776


namespace NUMINAMATH_CALUDE_rain_ratio_proof_l2387_238743

/-- Proves that the ratio of rain time on the third day to the second day is 2:1 -/
theorem rain_ratio_proof (first_day : ℕ) (second_day : ℕ) (total_time : ℕ) :
  first_day = 10 →
  second_day = first_day + 2 →
  total_time = 46 →
  ∃ (third_day : ℕ), 
    first_day + second_day + third_day = total_time ∧
    third_day = 2 * second_day :=
by
  sorry

end NUMINAMATH_CALUDE_rain_ratio_proof_l2387_238743


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l2387_238752

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_perimeter_range (t : Triangle) 
  (h1 : Real.sin (3 * t.B / 2 + π / 4) = Real.sqrt 2 / 2)
  (h2 : t.a + t.c = 2) :
  3 ≤ t.a + t.b + t.c ∧ t.a + t.b + t.c < 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l2387_238752


namespace NUMINAMATH_CALUDE_passengers_proportion_ge_cars_proportion_passengers_proportion_not_lt_cars_proportion_l2387_238764

/-- Represents the distribution of passenger cars -/
structure CarDistribution where
  total : ℕ
  overcrowded : ℕ
  passengers : ℕ
  passengers_overcrowded : ℕ

/-- Definition of an overcrowded car (60 or more passengers) -/
def is_overcrowded (passengers : ℕ) : Prop := passengers ≥ 60

/-- The proportion of overcrowded cars -/
def proportion_overcrowded (d : CarDistribution) : ℚ :=
  d.overcrowded / d.total

/-- The proportion of passengers in overcrowded cars -/
def proportion_passengers_overcrowded (d : CarDistribution) : ℚ :=
  d.passengers_overcrowded / d.passengers

/-- Theorem: The proportion of passengers in overcrowded cars is always
    greater than or equal to the proportion of overcrowded cars -/
theorem passengers_proportion_ge_cars_proportion (d : CarDistribution) :
  proportion_passengers_overcrowded d ≥ proportion_overcrowded d := by
  sorry

/-- Corollary: The proportion of passengers in overcrowded cars cannot be
    less than the proportion of overcrowded cars -/
theorem passengers_proportion_not_lt_cars_proportion (d : CarDistribution) :
  ¬(proportion_passengers_overcrowded d < proportion_overcrowded d) := by
  sorry

end NUMINAMATH_CALUDE_passengers_proportion_ge_cars_proportion_passengers_proportion_not_lt_cars_proportion_l2387_238764


namespace NUMINAMATH_CALUDE_fraction_decomposition_l2387_238790

theorem fraction_decomposition (n : ℕ) (h : n ≥ 3) :
  (2 : ℚ) / (2 * n - 1) = 1 / n + 1 / (n * (2 * n - 1)) := by
  sorry

#check fraction_decomposition

end NUMINAMATH_CALUDE_fraction_decomposition_l2387_238790


namespace NUMINAMATH_CALUDE_modulus_of_complex_quotient_l2387_238749

theorem modulus_of_complex_quotient :
  Complex.abs (Complex.I / (1 + 2 * Complex.I)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_quotient_l2387_238749


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2387_238789

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x + a > 0) → (0 ≤ a ∧ a ≤ 4) ∧
  ¬(0 ≤ a ∧ a ≤ 4 → ∀ x : ℝ, x^2 + a*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2387_238789


namespace NUMINAMATH_CALUDE_actual_distance_calculation_l2387_238709

/-- Calculates the actual distance between two towns given map distance, scale, and conversion factor. -/
theorem actual_distance_calculation (map_distance : ℝ) (scale : ℝ) (mile_to_km : ℝ) : 
  map_distance = 20 →
  scale = 5 →
  mile_to_km = 1.60934 →
  map_distance * scale * mile_to_km = 160.934 := by
sorry

end NUMINAMATH_CALUDE_actual_distance_calculation_l2387_238709


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l2387_238779

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_divisibility (m : ℕ) :
  ∃ k : ℕ, m ∣ (fibonacci k)^4 - (fibonacci k) - 2 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l2387_238779


namespace NUMINAMATH_CALUDE_circle_center_l2387_238726

/-- The center of the circle given by the equation x^2 + 10x + y^2 - 14y + 25 = 0 is (-5, 7) -/
theorem circle_center (x y : ℝ) : 
  (x^2 + 10*x + y^2 - 14*y + 25 = 0) → 
  (∃ r : ℝ, (x + 5)^2 + (y - 7)^2 = r^2) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l2387_238726


namespace NUMINAMATH_CALUDE_bowling_team_weight_problem_l2387_238775

theorem bowling_team_weight_problem (original_players : ℕ) (original_avg_weight : ℝ)
  (new_player1_weight : ℝ) (new_avg_weight : ℝ) :
  original_players = 7 →
  original_avg_weight = 121 →
  new_player1_weight = 110 →
  new_avg_weight = 113 →
  ∃ new_player2_weight : ℝ,
    new_player2_weight = 60 ∧
    (original_players : ℝ) * original_avg_weight + new_player1_weight + new_player2_weight =
      ((original_players : ℝ) + 2) * new_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_bowling_team_weight_problem_l2387_238775


namespace NUMINAMATH_CALUDE_sixteen_right_triangles_l2387_238745

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a right-angled triangle
structure RightTriangle where
  vertex1 : ℝ × ℝ
  vertex2 : ℝ × ℝ
  vertex3 : ℝ × ℝ

-- Function to check if two circles do not intersect
def nonIntersecting (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 > (c1.radius + c2.radius)^2

-- Function to check if a line is tangent to a circle
def isTangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (circle : Circle) : Prop :=
  ∃ p : ℝ × ℝ, line p p ∧ 
    let (x, y) := p
    let (cx, cy) := circle.center
    (x - cx)^2 + (y - cy)^2 = circle.radius^2

-- Function to check if a line is a common external tangent
def isCommonExternalTangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (c1 c2 : Circle) : Prop :=
  isTangent line c1 ∧ isTangent line c2

-- Function to check if a line is a common internal tangent
def isCommonInternalTangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (c1 c2 : Circle) : Prop :=
  isTangent line c1 ∧ isTangent line c2

-- Main theorem
theorem sixteen_right_triangles (c1 c2 : Circle) :
  nonIntersecting c1 c2 →
  ∃! (triangles : Finset RightTriangle),
    triangles.card = 16 ∧
    ∀ t ∈ triangles,
      ∃ (hypotenuse leg1 leg2 internalTangent : ℝ × ℝ → ℝ × ℝ → Prop),
        isCommonExternalTangent hypotenuse c1 c2 ∧
        isTangent leg1 c1 ∧
        isTangent leg2 c2 ∧
        isCommonInternalTangent internalTangent c1 c2 ∧
        (∃ p : ℝ × ℝ, internalTangent p p ∧ leg1 p p ∧ leg2 p p) :=
by
  sorry

end NUMINAMATH_CALUDE_sixteen_right_triangles_l2387_238745


namespace NUMINAMATH_CALUDE_max_remainder_problem_l2387_238795

theorem max_remainder_problem :
  ∃ (n : ℕ) (r : ℕ),
    2013 ≤ n ∧ n ≤ 2156 ∧
    n % 5 = r ∧ n % 11 = r ∧ n % 13 = r ∧
    r ≤ 4 ∧
    ∀ (m : ℕ) (s : ℕ),
      2013 ≤ m ∧ m ≤ 2156 ∧
      m % 5 = s ∧ m % 11 = s ∧ m % 13 = s →
      s ≤ r :=
by sorry

end NUMINAMATH_CALUDE_max_remainder_problem_l2387_238795


namespace NUMINAMATH_CALUDE_cube_root_of_negative_two_sqrt_two_l2387_238710

theorem cube_root_of_negative_two_sqrt_two (x : ℝ) :
  x = ((-2 : ℝ) ^ (1/2 : ℝ)) → x = ((-2 * (2 ^ (1/2 : ℝ))) ^ (1/3 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_two_sqrt_two_l2387_238710


namespace NUMINAMATH_CALUDE_domain_of_f_l2387_238706

-- Define the function f
def f (x : ℝ) : ℝ := (x - 5) ^ (1/4) + (x - 6) ^ (1/5) + (x - 7) ^ (1/2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ 7} :=
by
  sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_domain_of_f_l2387_238706


namespace NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l2387_238732

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem stating that 499 is the smallest prime whose digits sum to 23 -/
theorem smallest_prime_digit_sum_23 : 
  (is_prime 499 ∧ digit_sum 499 = 23) ∧ 
  ∀ n : ℕ, n < 499 → ¬(is_prime n ∧ digit_sum n = 23) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_digit_sum_23_l2387_238732


namespace NUMINAMATH_CALUDE_optimal_viewing_distance_l2387_238742

/-- The optimal distance from which to view a painting -/
theorem optimal_viewing_distance (a b : ℝ) (ha : a > 0) (hb : b > a) :
  ∃ x : ℝ, x > 0 ∧ ∀ y : ℝ, y > 0 → 
    (b - a) / (y + a * b / y) ≤ (b - a) / (x + a * b / x) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_optimal_viewing_distance_l2387_238742


namespace NUMINAMATH_CALUDE_symmetric_sine_graph_l2387_238758

theorem symmetric_sine_graph (φ : Real) : 
  (-Real.pi / 2 < φ ∧ φ < Real.pi / 2) →
  (∀ x, Real.sin (2 * x + φ) = Real.sin (2 * (2 * Real.pi / 3 - x) + φ)) →
  φ = -Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_symmetric_sine_graph_l2387_238758


namespace NUMINAMATH_CALUDE_lasagna_ratio_is_two_to_one_l2387_238781

/-- Represents the ratio of noodles to beef in Tom's lasagna recipe -/
def lasagna_ratio (beef_amount : ℕ) (initial_noodles : ℕ) (package_weight : ℕ) (packages_needed : ℕ) : ℚ :=
  let total_noodles := initial_noodles + package_weight * packages_needed
  (total_noodles : ℚ) / beef_amount

/-- The ratio of noodles to beef in Tom's lasagna recipe is 2:1 -/
theorem lasagna_ratio_is_two_to_one :
  lasagna_ratio 10 4 2 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_lasagna_ratio_is_two_to_one_l2387_238781


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2387_238762

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 2) :
  let set := List.replicate (n - 2) 2 ++ [1 - 2 / n, 1 - 2 / n]
  (List.sum set) / n = 2 - 2 / n - 4 / n^2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l2387_238762


namespace NUMINAMATH_CALUDE_driver_speed_problem_l2387_238767

theorem driver_speed_problem (v : ℝ) : 
  (v * 1 = (v + 18) * (2/3)) → v = 36 :=
by sorry

end NUMINAMATH_CALUDE_driver_speed_problem_l2387_238767


namespace NUMINAMATH_CALUDE_gcd_108_45_is_9_l2387_238751

theorem gcd_108_45_is_9 : Nat.gcd 108 45 = 9 := by
  -- Euclidean algorithm
  have h1 : 108 = 2 * 45 + 18 := by sorry
  have h2 : 45 = 2 * 18 + 9 := by sorry
  have h3 : 18 = 2 * 9 := by sorry

  -- Method of successive subtraction
  have s1 : 108 - 45 = 63 := by sorry
  have s2 : 63 - 45 = 18 := by sorry
  have s3 : 45 - 18 = 27 := by sorry
  have s4 : 27 - 18 = 9 := by sorry
  have s5 : 18 - 9 = 9 := by sorry

  sorry -- Proof to be completed

end NUMINAMATH_CALUDE_gcd_108_45_is_9_l2387_238751
