import Mathlib

namespace NUMINAMATH_CALUDE_B_equals_C_l3357_335796

def A : Set Int := {-1, 1}

def B : Set Int := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = x + y}

def C : Set Int := {z | ∃ x y, x ∈ A ∧ y ∈ A ∧ z = x - y}

theorem B_equals_C : B = C := by sorry

end NUMINAMATH_CALUDE_B_equals_C_l3357_335796


namespace NUMINAMATH_CALUDE_distance_between_points_l3357_335711

def point1 : ℝ × ℝ := (0, 3)
def point2 : ℝ × ℝ := (4, -5)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3357_335711


namespace NUMINAMATH_CALUDE_pancakes_and_honey_cost_l3357_335778

theorem pancakes_and_honey_cost (x y : ℕ) : 25 * x + 340 * y ≤ 2000 :=
by sorry

end NUMINAMATH_CALUDE_pancakes_and_honey_cost_l3357_335778


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3357_335713

theorem fraction_equation_solution : 
  let x : ℚ := 24
  (1 : ℚ) / 3 + (1 : ℚ) / 2 + (1 : ℚ) / x = (7 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3357_335713


namespace NUMINAMATH_CALUDE_range_of_alpha_minus_beta_l3357_335770

theorem range_of_alpha_minus_beta (α β : Real) 
  (h1 : -π ≤ α) (h2 : α ≤ β) (h3 : β ≤ π/2) :
  ∀ x, x ∈ Set.Icc (-3*π/2) 0 ↔ ∃ α' β', 
    -π ≤ α' ∧ α' ≤ β' ∧ β' ≤ π/2 ∧ x = α' - β' :=
by sorry

end NUMINAMATH_CALUDE_range_of_alpha_minus_beta_l3357_335770


namespace NUMINAMATH_CALUDE_trig_identity_l3357_335709

theorem trig_identity (α : Real) 
  (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3357_335709


namespace NUMINAMATH_CALUDE_eighth_fibonacci_is_21_l3357_335763

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem eighth_fibonacci_is_21 : fibonacci 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eighth_fibonacci_is_21_l3357_335763


namespace NUMINAMATH_CALUDE_double_root_equations_l3357_335723

/-- A quadratic equation ax^2 + bx + c = 0 is a double root equation if it has two real roots and one root is twice the other. -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ (x₂ = 2 * x₁ ∨ x₁ = 2 * x₂)

theorem double_root_equations :
  (is_double_root_equation 1 (-3) 2) ∧ 
  (∀ m n : ℝ, is_double_root_equation 1 m n → 4 * m^2 + 5 * m * n + n^2 = 0) ∧
  (∀ p q : ℝ, q = 2 / p → is_double_root_equation p 3 q) :=
by sorry

end NUMINAMATH_CALUDE_double_root_equations_l3357_335723


namespace NUMINAMATH_CALUDE_max_positive_terms_is_seven_l3357_335756

/-- An arithmetic sequence with a positive first term where the sum of the first 3 terms 
    equals the sum of the first 11 terms. -/
structure ArithmeticSequence where
  a₁ : ℝ
  d : ℝ
  first_term_positive : 0 < a₁
  sum_equality : 3 * (2 * a₁ + 2 * d) = 11 * (2 * a₁ + 10 * d)

/-- The maximum number of terms that can be summed before reaching a non-positive term -/
def max_positive_terms (seq : ArithmeticSequence) : ℕ :=
  7

/-- Theorem stating that the maximum number of terms is correct -/
theorem max_positive_terms_is_seven (seq : ArithmeticSequence) :
  (max_positive_terms seq = 7) ∧
  (∀ n : ℕ, n ≤ 7 → seq.a₁ + (n - 1) * seq.d > 0) ∧
  (seq.a₁ + 7 * seq.d ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_max_positive_terms_is_seven_l3357_335756


namespace NUMINAMATH_CALUDE_f_expression_and_g_monotonicity_l3357_335718

/-- A linear function f that is increasing on ℝ and satisfies f(f(x)) = 16x + 5 -/
def f : ℝ → ℝ :=
  sorry

/-- g is defined as g(x) = f(x)(x+m) -/
def g (m : ℝ) : ℝ → ℝ :=
  λ x ↦ f x * (x + m)

theorem f_expression_and_g_monotonicity :
  (∀ x y, x < y → f x < f y) ∧  -- f is increasing
  (∀ x, f (f x) = 16 * x + 5) →  -- f(f(x)) = 16x + 5
  (f = λ x ↦ 4 * x + 1) ∧  -- f(x) = 4x + 1
  (∀ m, (∀ x y, 1 < x ∧ x < y → g m x < g m y) → -9/4 ≤ m)  -- If g is increasing on (1,+∞), then m ≥ -9/4
  := by sorry

end NUMINAMATH_CALUDE_f_expression_and_g_monotonicity_l3357_335718


namespace NUMINAMATH_CALUDE_fraction_product_equality_l3357_335788

theorem fraction_product_equality : (3 + 5 + 7) / (2 + 4 + 6) * (4 + 8 + 12) / (1 + 3 + 5) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l3357_335788


namespace NUMINAMATH_CALUDE_simplify_expression_l3357_335787

theorem simplify_expression (x y : ℝ) : (5 - 4*y) - (6 + 5*y - 2*x) = -1 - 9*y + 2*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3357_335787


namespace NUMINAMATH_CALUDE_larger_number_proof_l3357_335725

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 55) (h2 : x - y = 15) : x = 35 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l3357_335725


namespace NUMINAMATH_CALUDE_complex_sum_to_polar_l3357_335773

theorem complex_sum_to_polar : 
  15 * Complex.exp (Complex.I * Real.pi / 7) + 15 * Complex.exp (Complex.I * 5 * Real.pi / 7) = 
  (30 * Real.cos (3 * Real.pi / 14) * Real.cos (Real.pi / 14)) * Complex.exp (Complex.I * 3 * Real.pi / 7) := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_to_polar_l3357_335773


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l3357_335716

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l3357_335716


namespace NUMINAMATH_CALUDE_wife_walking_speed_l3357_335730

/-- Proves that given a circular track of 726 m circumference, if two people walk in opposite
    directions starting from the same point, with one person walking at 4.5 km/hr and they
    meet after 5.28 minutes, then the other person's walking speed is 3.75 km/hr. -/
theorem wife_walking_speed
  (track_circumference : ℝ)
  (suresh_speed : ℝ)
  (meeting_time : ℝ)
  (h1 : track_circumference = 726 / 1000) -- Convert 726 m to km
  (h2 : suresh_speed = 4.5)
  (h3 : meeting_time = 5.28 / 60) -- Convert 5.28 minutes to hours
  : ∃ (wife_speed : ℝ), wife_speed = 3.75 := by
  sorry

#check wife_walking_speed

end NUMINAMATH_CALUDE_wife_walking_speed_l3357_335730


namespace NUMINAMATH_CALUDE_sequence_convergence_l3357_335781

theorem sequence_convergence (a : ℕ → ℚ) 
  (h : ∀ n : ℕ, a (n + 1)^2 - a (n + 1) = a n) : 
  a 1 = 0 ∨ a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_convergence_l3357_335781


namespace NUMINAMATH_CALUDE_johns_actual_marks_l3357_335789

theorem johns_actual_marks (total : ℝ) (n : ℕ) (wrong_mark : ℝ) (increase : ℝ) :
  n = 80 →
  wrong_mark = 82 →
  increase = 1/2 →
  (total + wrong_mark) / n = (total + johns_mark) / n + increase →
  johns_mark = 42 :=
by sorry

end NUMINAMATH_CALUDE_johns_actual_marks_l3357_335789


namespace NUMINAMATH_CALUDE_expected_different_faces_formula_l3357_335724

/-- The number of sides on a fair die -/
def numSides : ℕ := 6

/-- The number of times the die is rolled -/
def numRolls : ℕ := 6

/-- The probability of a specific face not appearing in a single roll -/
def probNotAppear : ℚ := (numSides - 1) / numSides

/-- The expected number of different faces that will appear when rolling a fair die -/
def expectedDifferentFaces : ℚ := numSides * (1 - probNotAppear ^ numRolls)

/-- Theorem: The expected number of different faces is equal to (6^6 - 5^6) / 6^5 -/
theorem expected_different_faces_formula :
  expectedDifferentFaces = (numSides ^ numRolls - (numSides - 1) ^ numRolls) / (numSides ^ (numRolls - 1)) :=
by sorry

end NUMINAMATH_CALUDE_expected_different_faces_formula_l3357_335724


namespace NUMINAMATH_CALUDE_no_infinite_harmonic_mean_sequence_l3357_335720

theorem no_infinite_harmonic_mean_sequence :
  ¬ ∃ (a : ℕ → ℕ), 
    (∃ i j, a i ≠ a j) ∧ 
    (∀ n : ℕ, n ≥ 2 → a n = (2 * a (n-1) * a (n+1)) / (a (n-1) + a (n+1))) :=
by sorry

end NUMINAMATH_CALUDE_no_infinite_harmonic_mean_sequence_l3357_335720


namespace NUMINAMATH_CALUDE_base_conversion_l3357_335706

/-- Given that the base 6 number 123₆ is equal to the base b number 203ᵦ,
    prove that the positive value of b is 2√6. -/
theorem base_conversion (b : ℝ) (h : b > 0) : 
  (1 * 6^2 + 2 * 6 + 3 : ℝ) = 2 * b^2 + 3 → b = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l3357_335706


namespace NUMINAMATH_CALUDE_oscar_christina_age_ratio_l3357_335795

def christina_age : ℕ := sorry
def oscar_age : ℕ := 6

theorem oscar_christina_age_ratio :
  (oscar_age + 15) / christina_age = 3 / 5 :=
by
  have h1 : christina_age + 5 = 80 / 2 := by sorry
  sorry

end NUMINAMATH_CALUDE_oscar_christina_age_ratio_l3357_335795


namespace NUMINAMATH_CALUDE_bert_stamp_ratio_l3357_335703

def stamps_before (total_after purchase : ℕ) : ℕ := total_after - purchase

def ratio_simplify (a b : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd a b
  (a / gcd, b / gcd)

theorem bert_stamp_ratio : 
  let purchase := 300
  let total_after := 450
  let before := stamps_before total_after purchase
  ratio_simplify before purchase = (1, 2) := by
sorry

end NUMINAMATH_CALUDE_bert_stamp_ratio_l3357_335703


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l3357_335758

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 99 → n ≥ 10 → n % 17 = 0 → n ≤ 85 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l3357_335758


namespace NUMINAMATH_CALUDE_min_lcm_a_c_l3357_335712

theorem min_lcm_a_c (a b c : ℕ+) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 21) :
  ∃ (a' c' : ℕ+), Nat.lcm a' c' = 420 ∧ ∀ (x y : ℕ+), Nat.lcm x b = 20 → Nat.lcm b y = 21 → Nat.lcm a' c' ≤ Nat.lcm x y :=
sorry

end NUMINAMATH_CALUDE_min_lcm_a_c_l3357_335712


namespace NUMINAMATH_CALUDE_tromino_tiling_l3357_335702

/-- An L-shaped tromino covers exactly 3 unit squares. -/
def Tromino : ℕ := 3

/-- Represents whether an m×n grid can be tiled with L-shaped trominoes. -/
def can_tile (m n : ℕ) : Prop := 6 ∣ (m * n)

/-- 
Theorem: An m×n grid can be tiled with L-shaped trominoes if and only if 6 divides mn.
-/
theorem tromino_tiling (m n : ℕ) : can_tile m n ↔ 6 ∣ (m * n) := by sorry

end NUMINAMATH_CALUDE_tromino_tiling_l3357_335702


namespace NUMINAMATH_CALUDE_pizza_cheese_calories_pizza_cheese_calories_proof_l3357_335728

theorem pizza_cheese_calories : ℝ → Prop :=
  fun cheese_calories =>
    let lettuce_calories : ℝ := 50
    let carrot_calories : ℝ := 2 * lettuce_calories
    let dressing_calories : ℝ := 210
    let salad_calories : ℝ := lettuce_calories + carrot_calories + dressing_calories
    let crust_calories : ℝ := 600
    let pepperoni_calories : ℝ := (1 / 3) * crust_calories
    let pizza_calories : ℝ := crust_calories + pepperoni_calories + cheese_calories
    let jackson_salad_portion : ℝ := 1 / 4
    let jackson_pizza_portion : ℝ := 1 / 5
    let jackson_consumed_calories : ℝ := 330
    jackson_salad_portion * salad_calories + jackson_pizza_portion * pizza_calories = jackson_consumed_calories →
    cheese_calories = 400

-- Proof
theorem pizza_cheese_calories_proof : pizza_cheese_calories 400 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cheese_calories_pizza_cheese_calories_proof_l3357_335728


namespace NUMINAMATH_CALUDE_non_black_cows_l3357_335776

theorem non_black_cows (total : ℕ) (black : ℕ) (h1 : total = 18) (h2 : black = total / 2 + 5) :
  total - black = 4 := by
sorry

end NUMINAMATH_CALUDE_non_black_cows_l3357_335776


namespace NUMINAMATH_CALUDE_factor_expression_l3357_335755

theorem factor_expression (x y z : ℝ) :
  (y^2 - z^2) * (1 + x*y) * (1 + x*z) + 
  (z^2 - x^2) * (1 + y*z) * (1 + x*y) + 
  (x^2 - y^2) * (1 + y*z) * (1 + x*z) = 
  (y - z) * (z - x) * (x - y) * (x*y*z + x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3357_335755


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l3357_335786

/-- Represents the number of triangles of each color in one half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs of each type when the figure is folded -/
structure CoincidingPairs where
  red_red : ℕ
  blue_blue : ℕ
  red_white : ℕ
  white_white : ℕ

/-- Theorem stating that given the conditions, 7 white pairs must coincide -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 4 ∧ 
  counts.blue = 6 ∧ 
  counts.white = 10 ∧
  pairs.red_red = 3 ∧
  pairs.blue_blue = 4 ∧
  pairs.red_white = 2 →
  pairs.white_white = 7 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l3357_335786


namespace NUMINAMATH_CALUDE_horner_method_v3_l3357_335733

def f (x : ℝ) : ℝ := 3*x^5 - 2*x^4 + 2*x^3 - 4*x^2 - 7

def horner_v3 (a : ℝ → ℝ) (x : ℝ) : ℝ :=
  let v0 := 3
  let v1 := v0 * x - 2
  let v2 := v1 * x + 2
  v2 * x - 4

theorem horner_method_v3 :
  horner_v3 f 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_v3_l3357_335733


namespace NUMINAMATH_CALUDE_soccer_field_kids_l3357_335734

/-- The number of kids initially on the soccer field -/
def initial_kids : ℕ := 14

/-- The number of kids who decided to join -/
def joining_kids : ℕ := 22

/-- The total number of kids on the soccer field after new kids join -/
def total_kids : ℕ := initial_kids + joining_kids

theorem soccer_field_kids : total_kids = 36 := by
  sorry

end NUMINAMATH_CALUDE_soccer_field_kids_l3357_335734


namespace NUMINAMATH_CALUDE_cos_equality_integer_l3357_335750

theorem cos_equality_integer (n : ℤ) : 
  -180 ≤ n ∧ n ≤ 180 ∧ Real.cos (↑n * π / 180) = Real.cos (430 * π / 180) →
  n = 70 ∨ n = -70 := by
sorry

end NUMINAMATH_CALUDE_cos_equality_integer_l3357_335750


namespace NUMINAMATH_CALUDE_initial_workers_correct_l3357_335798

/-- Represents the initial number of workers -/
def initial_workers : ℕ := 120

/-- Represents the total number of days to complete the wall -/
def total_days : ℕ := 50

/-- Represents the number of days after which progress is measured -/
def progress_days : ℕ := 25

/-- Represents the fraction of work completed after progress_days -/
def work_completed : ℚ := 2/5

/-- Represents the additional workers needed to complete on time -/
def additional_workers : ℕ := 30

/-- Proves that the initial number of workers is correct given the conditions -/
theorem initial_workers_correct :
  initial_workers * total_days = (initial_workers + additional_workers) * 
    (total_days * work_completed + progress_days * (1 - work_completed)) :=
by sorry

end NUMINAMATH_CALUDE_initial_workers_correct_l3357_335798


namespace NUMINAMATH_CALUDE_no_primes_between_factorial_plus_n_and_factorial_plus_2n_l3357_335772

theorem no_primes_between_factorial_plus_n_and_factorial_plus_2n (n : ℕ) (hn : n > 1) :
  ∀ p, Nat.Prime p → ¬(n! + n < p ∧ p < n! + 2*n) :=
sorry

end NUMINAMATH_CALUDE_no_primes_between_factorial_plus_n_and_factorial_plus_2n_l3357_335772


namespace NUMINAMATH_CALUDE_system_solution_l3357_335731

theorem system_solution : 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ = 0 ∧ y₁ = 1/3) ∧ 
    (x₂ = 19/2 ∧ y₂ = -6) ∧
    (∀ x y : ℝ, (5*x*(y + 6) = 0 ∧ 2*x + 3*y = 1) → 
      ((x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3357_335731


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3357_335736

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_power_sum : i^25 + i^125 = 2*i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3357_335736


namespace NUMINAMATH_CALUDE_condition_relationship_l3357_335782

theorem condition_relationship : 
  ∀ x : ℝ, (x > 3 → x > 2) ∧ ¬(x > 2 → x > 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3357_335782


namespace NUMINAMATH_CALUDE_number_problem_l3357_335743

theorem number_problem : ∃ x : ℝ, (0.3 * x = 0.6 * 50 + 30) ∧ (x = 200) := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3357_335743


namespace NUMINAMATH_CALUDE_train_length_l3357_335748

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) (bridge_length : ℝ) : 
  speed_kmph = 36 → 
  time_seconds = 23.998080153587715 → 
  bridge_length = 140 → 
  (speed_kmph * 1000 / 3600) * time_seconds - bridge_length = 99.98080153587715 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l3357_335748


namespace NUMINAMATH_CALUDE_factorization_of_four_a_squared_minus_one_l3357_335799

theorem factorization_of_four_a_squared_minus_one (a : ℝ) : 4 * a^2 - 1 = (2*a - 1) * (2*a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_four_a_squared_minus_one_l3357_335799


namespace NUMINAMATH_CALUDE_combined_mean_l3357_335779

theorem combined_mean (set1_count : ℕ) (set1_mean : ℝ) (set2_count : ℕ) (set2_mean : ℝ) :
  set1_count = 8 →
  set2_count = 10 →
  set1_mean = 17 →
  set2_mean = 23 →
  let total_count := set1_count + set2_count
  let combined_mean := (set1_count * set1_mean + set2_count * set2_mean) / total_count
  combined_mean = (8 * 17 + 10 * 23) / 18 :=
by
  sorry

end NUMINAMATH_CALUDE_combined_mean_l3357_335779


namespace NUMINAMATH_CALUDE_arrangements_count_l3357_335764

/-- The number of possible arrangements for 5 male students and 3 female students
    standing in a row, where the female students must stand together. -/
def num_arrangements : ℕ :=
  let num_male_students : ℕ := 5
  let num_female_students : ℕ := 3
  num_male_students.factorial * (num_male_students + 1) * num_female_students.factorial

theorem arrangements_count : num_arrangements = 720 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_count_l3357_335764


namespace NUMINAMATH_CALUDE_student_bicycle_speed_l3357_335768

theorem student_bicycle_speed
  (distance : ℝ)
  (speed_ratio : ℝ)
  (time_difference : ℝ)
  (h_distance : distance = 12)
  (h_speed_ratio : speed_ratio = 1.2)
  (h_time_difference : time_difference = 1/6) :
  ∃ (speed_B : ℝ), speed_B = 12 ∧
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference :=
by sorry

end NUMINAMATH_CALUDE_student_bicycle_speed_l3357_335768


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3357_335707

theorem binomial_coefficient_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₂ + a₄ + a₆)^2 - (a₁ + a₃ + a₅ + a₇)^2 = -2187 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3357_335707


namespace NUMINAMATH_CALUDE_element_in_union_l3357_335769

theorem element_in_union (M N : Set ℕ) (a : ℕ) 
  (h1 : M ∪ N = {1, 2, 3})
  (h2 : M ∩ N = {a}) : 
  a ∈ M ∪ N := by
  sorry

end NUMINAMATH_CALUDE_element_in_union_l3357_335769


namespace NUMINAMATH_CALUDE_sandy_nickels_theorem_sandy_specific_case_l3357_335741

/-- The number of nickels Sandy has after her dad borrows some -/
def nickels_remaining (initial_nickels borrowed_nickels : ℕ) : ℕ :=
  initial_nickels - borrowed_nickels

/-- Theorem stating that Sandy's remaining nickels is the difference between initial and borrowed -/
theorem sandy_nickels_theorem (initial_nickels borrowed_nickels : ℕ) 
  (h : borrowed_nickels ≤ initial_nickels) :
  nickels_remaining initial_nickels borrowed_nickels = initial_nickels - borrowed_nickels :=
by
  sorry

/-- Sandy's specific case -/
theorem sandy_specific_case :
  nickels_remaining 31 20 = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_nickels_theorem_sandy_specific_case_l3357_335741


namespace NUMINAMATH_CALUDE_square_minus_four_times_product_specific_calculation_l3357_335762

theorem square_minus_four_times_product (a b : ℕ) : 
  (a + b) ^ 2 - 4 * a * b = (a - b) ^ 2 :=
by sorry

theorem specific_calculation : 
  (476 + 424) ^ 2 - 4 * 476 * 424 = 5776 :=
by sorry

end NUMINAMATH_CALUDE_square_minus_four_times_product_specific_calculation_l3357_335762


namespace NUMINAMATH_CALUDE_fg_squared_value_l3357_335746

-- Define the functions g and f
def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 11

-- State the theorem
theorem fg_squared_value : (f (g 6))^2 = 26569 := by sorry

end NUMINAMATH_CALUDE_fg_squared_value_l3357_335746


namespace NUMINAMATH_CALUDE_probability_three_face_cards_different_suits_value_l3357_335774

/-- A standard deck of cards. -/
def StandardDeck : ℕ := 52

/-- The number of face cards in a standard deck. -/
def FaceCards : ℕ := 12

/-- The number of suits in a standard deck. -/
def Suits : ℕ := 4

/-- The number of face cards per suit. -/
def FaceCardsPerSuit : ℕ := FaceCards / Suits

/-- The probability of selecting three face cards of different suits from a standard deck without replacement. -/
def probability_three_face_cards_different_suits : ℚ :=
  (FaceCards : ℚ) / StandardDeck *
  (FaceCards - FaceCardsPerSuit : ℚ) / (StandardDeck - 1) *
  (FaceCards - 2 * FaceCardsPerSuit : ℚ) / (StandardDeck - 2)

theorem probability_three_face_cards_different_suits_value :
  probability_three_face_cards_different_suits = 4 / 915 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_face_cards_different_suits_value_l3357_335774


namespace NUMINAMATH_CALUDE_fair_attendance_ratio_l3357_335752

theorem fair_attendance_ratio :
  let this_year : ℕ := 600
  let total_three_years : ℕ := 2800
  let next_year : ℕ := (total_three_years - this_year + 200) / 2
  let last_year : ℕ := next_year - 200
  (next_year : ℚ) / this_year = 2 :=
by sorry

end NUMINAMATH_CALUDE_fair_attendance_ratio_l3357_335752


namespace NUMINAMATH_CALUDE_math_preference_gender_related_l3357_335797

/-- Represents the survey data and critical value for the chi-square test -/
structure SurveyData where
  total_students : Nat
  male_percentage : Rat
  total_math_liking : Nat
  female_math_liking : Nat
  critical_value : Rat

/-- Calculates the chi-square statistic for the given survey data -/
def calculate_chi_square (data : SurveyData) : Rat :=
  sorry

/-- Theorem stating that the calculated chi-square value exceeds the critical value -/
theorem math_preference_gender_related (data : SurveyData) :
  data.total_students = 100 ∧
  data.male_percentage = 55/100 ∧
  data.total_math_liking = 40 ∧
  data.female_math_liking = 20 ∧
  data.critical_value = 7879/1000 →
  calculate_chi_square data > data.critical_value :=
sorry

end NUMINAMATH_CALUDE_math_preference_gender_related_l3357_335797


namespace NUMINAMATH_CALUDE_inequality_proof_l3357_335704

theorem inequality_proof (x : ℝ) :
  (x > 0 → x + 1/x ≥ 2) ∧
  (x > 0 → (x + 1/x = 2 ↔ x = 1)) ∧
  (x < 0 → x + 1/x ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3357_335704


namespace NUMINAMATH_CALUDE_unique_rebus_solution_l3357_335732

/-- Represents a four-digit number ABCD where A, B, C, D are distinct non-zero digits. -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  d_nonzero : d ≠ 0
  all_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- The rebus equation ABCA = 182 * CD -/
def rebusEquation (n : FourDigitNumber) : Prop :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.a = 182 * (10 * n.c + n.d)

/-- Theorem stating that 2916 is the only solution to the rebus equation -/
theorem unique_rebus_solution :
  ∃! n : FourDigitNumber, rebusEquation n ∧ n.a = 2 ∧ n.b = 9 ∧ n.c = 1 ∧ n.d = 6 :=
sorry

end NUMINAMATH_CALUDE_unique_rebus_solution_l3357_335732


namespace NUMINAMATH_CALUDE_school_population_l3357_335790

/-- The number of students that each classroom holds -/
def students_per_classroom : ℕ := 30

/-- The number of classrooms needed -/
def number_of_classrooms : ℕ := 13

/-- The total number of students in the school -/
def total_students : ℕ := students_per_classroom * number_of_classrooms

theorem school_population : total_students = 390 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l3357_335790


namespace NUMINAMATH_CALUDE_divisibility_of_n_squared_plus_n_plus_two_l3357_335705

theorem divisibility_of_n_squared_plus_n_plus_two :
  (∀ n : ℕ, 2 ∣ (n^2 + n + 2)) ∧
  (∃ n : ℕ, ¬(5 ∣ (n^2 + n + 2))) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_n_squared_plus_n_plus_two_l3357_335705


namespace NUMINAMATH_CALUDE_special_polynomial_value_l3357_335794

/-- A polynomial of the form ± x^6 ± x^5 ± x^4 ± x^3 ± x^2 ± x ± 1 -/
def SpecialPolynomial (P : ℝ → ℝ) : Prop :=
  ∃ (a b c d e f g : ℤ), ∀ x,
    P x = a * x^6 + b * x^5 + c * x^4 + d * x^3 + e * x^2 + f * x + g ∧
    (a = 1 ∨ a = -1) ∧ (b = 1 ∨ b = -1) ∧ (c = 1 ∨ c = -1) ∧
    (d = 1 ∨ d = -1) ∧ (e = 1 ∨ e = -1) ∧ (f = 1 ∨ f = -1) ∧ (g = 1 ∨ g = -1)

theorem special_polynomial_value (P : ℝ → ℝ) :
  SpecialPolynomial P → P 2 = 27 → P 3 = 439 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_value_l3357_335794


namespace NUMINAMATH_CALUDE_variance_scaling_l3357_335759

/-- Given a list of 8 real numbers, compute its variance -/
def variance (xs : List ℝ) : ℝ := sorry

theorem variance_scaling (xs : List ℝ) (h : variance xs = 3) :
  variance (xs.map (· * 2)) = 12 := by sorry

end NUMINAMATH_CALUDE_variance_scaling_l3357_335759


namespace NUMINAMATH_CALUDE_polynomial_property_l3357_335738

/-- Given a polynomial P(x) = x^4 + ax^3 + bx^2 + cx + d where a, b, c, d are constants,
    if P(1) = 1993, P(2) = 3986, and P(3) = 5979, then 1/4[P(11) + P(-7)] = 4693. -/
theorem polynomial_property (a b c d : ℝ) (P : ℝ → ℝ) 
    (h1 : ∀ x, P x = x^4 + a*x^3 + b*x^2 + c*x + d)
    (h2 : P 1 = 1993)
    (h3 : P 2 = 3986)
    (h4 : P 3 = 5979) :
    (1/4) * (P 11 + P (-7)) = 4693 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l3357_335738


namespace NUMINAMATH_CALUDE_fib_inequality_l3357_335735

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Proof of the inequality for Fibonacci numbers -/
theorem fib_inequality (n : ℕ) (hn : n > 0) :
  (fib (n + 2) : ℝ) ^ (1 / n : ℝ) ≥ 1 + 1 / ((fib (n + 1) : ℝ) ^ (1 / n : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_fib_inequality_l3357_335735


namespace NUMINAMATH_CALUDE_correct_setup_is_valid_l3357_335737

-- Define the structure for an experimental setup
structure ExperimentalSetup :=
  (num_plates : Nat)
  (bacteria_counts : List Nat)
  (average_count : Nat)

-- Define the conditions for a valid experimental setup
def is_valid_setup (setup : ExperimentalSetup) : Prop :=
  setup.num_plates ≥ 3 ∧
  setup.bacteria_counts.length = setup.num_plates ∧
  setup.average_count = setup.bacteria_counts.sum / setup.num_plates ∧
  setup.bacteria_counts.all (λ count => 
    setup.bacteria_counts.all (λ other_count => 
      (count : Int) - other_count ≤ 50 ∧ other_count - count ≤ 50))

-- Define the correct setup (option D)
def correct_setup : ExperimentalSetup :=
  { num_plates := 3,
    bacteria_counts := [210, 240, 250],
    average_count := 233 }

-- Theorem to prove
theorem correct_setup_is_valid :
  is_valid_setup correct_setup :=
sorry

end NUMINAMATH_CALUDE_correct_setup_is_valid_l3357_335737


namespace NUMINAMATH_CALUDE_negation_of_all_cats_not_pets_l3357_335740

-- Define the universe of discourse
variable (U : Type)

-- Define predicates for "is a cat" and "is a pet"
variable (Cat : U → Prop)
variable (Pet : U → Prop)

-- Define the original statement "All cats are not pets"
def all_cats_not_pets : Prop := ∀ x, Cat x → ¬(Pet x)

-- Define the negation "Some cats are pets"
def some_cats_are_pets : Prop := ∃ x, Cat x ∧ Pet x

-- Theorem statement
theorem negation_of_all_cats_not_pets :
  ¬(all_cats_not_pets U Cat Pet) ↔ some_cats_are_pets U Cat Pet :=
sorry

end NUMINAMATH_CALUDE_negation_of_all_cats_not_pets_l3357_335740


namespace NUMINAMATH_CALUDE_computer_profit_profit_function_max_profit_l3357_335726

/-- Profit from selling computers -/
theorem computer_profit (profit_A profit_B : ℚ) : 
  (10 * profit_A + 20 * profit_B = 4000) →
  (20 * profit_A + 10 * profit_B = 3500) →
  (profit_A = 100 ∧ profit_B = 150) :=
sorry

/-- Functional relationship between total profit and number of type A computers -/
theorem profit_function (x y : ℚ) :
  (x ≥ 0 ∧ x ≤ 100) →
  (y = 100 * x + 150 * (100 - x)) →
  (y = -50 * x + 15000) :=
sorry

/-- Maximum profit when purchasing at least 20 units of type A -/
theorem max_profit (x y : ℚ) :
  (x ≥ 20 ∧ x ≤ 100) →
  (y = -50 * x + 15000) →
  (∀ z, z ≥ 20 ∧ z ≤ 100 → -50 * z + 15000 ≤ 14000) :=
sorry

end NUMINAMATH_CALUDE_computer_profit_profit_function_max_profit_l3357_335726


namespace NUMINAMATH_CALUDE_arccos_sum_equals_arcsin_l3357_335710

theorem arccos_sum_equals_arcsin (x : ℝ) : 
  Real.arccos x + Real.arccos (1 - x) = Real.arcsin x →
  (x = 0 ∨ x = 1 ∨ x = (1 + Real.sqrt 5) / 2) := by
sorry

end NUMINAMATH_CALUDE_arccos_sum_equals_arcsin_l3357_335710


namespace NUMINAMATH_CALUDE_unknown_denomination_is_500_l3357_335700

/-- Represents the denomination problem with given conditions --/
structure DenominationProblem where
  total_amount : ℕ
  known_denomination : ℕ
  total_notes : ℕ
  known_denomination_count : ℕ
  (total_amount_check : total_amount = 10350)
  (known_denomination_check : known_denomination = 50)
  (total_notes_check : total_notes = 54)
  (known_denomination_count_check : known_denomination_count = 37)

/-- Theorem stating that the unknown denomination is 500 --/
theorem unknown_denomination_is_500 (p : DenominationProblem) : 
  (p.total_amount - p.known_denomination * p.known_denomination_count) / (p.total_notes - p.known_denomination_count) = 500 :=
sorry

end NUMINAMATH_CALUDE_unknown_denomination_is_500_l3357_335700


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l3357_335708

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {-1, 0, 2}

theorem complement_A_intersect_B : 
  (Set.univ \ A) ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l3357_335708


namespace NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l3357_335766

theorem coefficient_x6_in_expansion : ∃ c : ℤ, c = -10 ∧ 
  (Polynomial.coeff ((1 + X + X^2) * (1 - X)^6) 6 = c) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x6_in_expansion_l3357_335766


namespace NUMINAMATH_CALUDE_oliver_seashell_collection_l3357_335767

-- Define the number of seashells collected on each day
def monday_shells : ℕ := 2
def tuesday_shells : ℕ := 2

-- Define the total number of seashells
def total_shells : ℕ := monday_shells + tuesday_shells

-- Theorem statement
theorem oliver_seashell_collection : total_shells = 4 := by
  sorry

end NUMINAMATH_CALUDE_oliver_seashell_collection_l3357_335767


namespace NUMINAMATH_CALUDE_parallel_condition_l3357_335717

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes and lines
variable (plane_parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)

-- Define the "contained in" relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem parallel_condition 
  (α β : Plane) 
  (m : Line) 
  (h_distinct : α ≠ β) 
  (h_m_in_α : line_in_plane m α) : 
  (line_parallel_plane m β → plane_parallel α β) ∧ 
  ¬(plane_parallel α β → line_parallel_plane m β) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l3357_335717


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3357_335749

/-- Given that the solution set of ax^2 + bx + c > 0 is {x | -1 < x < 2},
    prove that the solution set of a(x^2 + 1) + b(x - 1) + c > 2ax is {x | 0 < x < 3} -/
theorem solution_set_inequality (a b c : ℝ) :
  (∀ x, ax^2 + b*x + c > 0 ↔ -1 < x ∧ x < 2) →
  (∀ x, a*(x^2 + 1) + b*(x - 1) + c > 2*a*x ↔ 0 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3357_335749


namespace NUMINAMATH_CALUDE_solve_for_k_l3357_335745

theorem solve_for_k (k : ℚ) : 
  (∃ x : ℚ, 3 * x + (2 * k - 1) = x - 6 * (3 * k + 2)) ∧ 
  (3 * 1 + (2 * k - 1) = 1 - 6 * (3 * k + 2)) → 
  k = -13/20 := by
sorry

end NUMINAMATH_CALUDE_solve_for_k_l3357_335745


namespace NUMINAMATH_CALUDE_fidos_yard_area_l3357_335701

theorem fidos_yard_area (s : ℝ) (h : s > 0) :
  let r := s / 2
  let area_circle := π * r^2
  let area_square := s^2
  let fraction := area_circle / area_square
  ∃ (a b : ℝ), fraction = (Real.sqrt a / b) * π ∧ a * b = 0 :=
by sorry

end NUMINAMATH_CALUDE_fidos_yard_area_l3357_335701


namespace NUMINAMATH_CALUDE_cody_marbles_l3357_335715

def initial_marbles : ℕ := 12
def marbles_given : ℕ := 5

theorem cody_marbles : initial_marbles - marbles_given = 7 := by
  sorry

end NUMINAMATH_CALUDE_cody_marbles_l3357_335715


namespace NUMINAMATH_CALUDE_unit_intervals_have_continuum_cardinality_l3357_335714

-- Define the cardinality of the continuum
def continuum_cardinality := Cardinal.mk ℝ

-- Define the open interval (0,1)
def open_unit_interval := Set.Ioo (0 : ℝ) 1

-- Define the closed interval [0,1]
def closed_unit_interval := Set.Icc (0 : ℝ) 1

-- Theorem statement
theorem unit_intervals_have_continuum_cardinality :
  (Cardinal.mk open_unit_interval = continuum_cardinality) ∧
  (Cardinal.mk closed_unit_interval = continuum_cardinality) := by
  sorry

end NUMINAMATH_CALUDE_unit_intervals_have_continuum_cardinality_l3357_335714


namespace NUMINAMATH_CALUDE_game_boxes_needed_l3357_335719

theorem game_boxes_needed (initial_games : ℕ) (sold_games : ℕ) (games_per_box : ℕ) : 
  initial_games = 76 → sold_games = 46 → games_per_box = 5 → 
  (initial_games - sold_games) / games_per_box = 6 := by
  sorry

end NUMINAMATH_CALUDE_game_boxes_needed_l3357_335719


namespace NUMINAMATH_CALUDE_total_interest_is_350_l3357_335722

/-- Calculate the total interest amount for two loans over a specified period. -/
def totalInterest (loan1Amount : ℝ) (loan1Rate : ℝ) (loan2Amount : ℝ) (loan2Rate : ℝ) (years : ℝ) : ℝ :=
  (loan1Amount * loan1Rate * years) + (loan2Amount * loan2Rate * years)

/-- Theorem stating that the total interest for the given loans and time period is 350. -/
theorem total_interest_is_350 :
  totalInterest 1000 0.03 1200 0.05 3.888888888888889 = 350 := by
  sorry

#eval totalInterest 1000 0.03 1200 0.05 3.888888888888889

end NUMINAMATH_CALUDE_total_interest_is_350_l3357_335722


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3357_335780

/-- A right triangle with side lengths a, b, and c (a < b < c) -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  a_lt_b : a < b
  b_lt_c : b < c
  pythagoras : a^2 + b^2 = c^2

/-- The condition a:b:c = 3:4:5 -/
def is_345_ratio (t : RightTriangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t.a = 3*k ∧ t.b = 4*k ∧ t.c = 5*k

/-- The condition that a, b, c form an arithmetic progression -/
def is_arithmetic_progression (t : RightTriangle) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ t.b - t.a = d ∧ t.c - t.b = d

theorem sufficient_not_necessary :
  (∀ t : RightTriangle, is_345_ratio t → is_arithmetic_progression t) ∧
  (∃ t : RightTriangle, is_arithmetic_progression t ∧ ¬is_345_ratio t) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3357_335780


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3357_335742

/-- For positive real numbers a, b, c ≤ √2 with abc = 2, 
    prove √2 ∑(ab + 3c)/(3ab + c) ≥ a + b + c -/
theorem cyclic_sum_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha2 : a ≤ Real.sqrt 2) (hb2 : b ≤ Real.sqrt 2) (hc2 : c ≤ Real.sqrt 2)
  (habc : a * b * c = 2) :
  Real.sqrt 2 * (((a * b + 3 * c) / (3 * a * b + c)) +
                 ((b * c + 3 * a) / (3 * b * c + a)) +
                 ((c * a + 3 * b) / (3 * c * a + b))) ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3357_335742


namespace NUMINAMATH_CALUDE_juniors_percentage_l3357_335727

/-- Represents the composition of students in a high school sample. -/
structure StudentSample where
  total : ℕ
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ
  seniors : ℕ

/-- Calculates the percentage of a part relative to the total. -/
def percentage (part : ℕ) (total : ℕ) : ℚ :=
  (part : ℚ) / (total : ℚ) * 100

/-- Theorem stating the percentage of juniors in the given student sample. -/
theorem juniors_percentage (sample : StudentSample) : 
  sample.total = 800 ∧ 
  sample.seniors = 160 ∧
  sample.sophomores = sample.total / 4 ∧
  sample.freshmen = sample.sophomores + 24 ∧
  sample.total = sample.freshmen + sample.sophomores + sample.juniors + sample.seniors →
  percentage sample.juniors sample.total = 27 := by
  sorry

end NUMINAMATH_CALUDE_juniors_percentage_l3357_335727


namespace NUMINAMATH_CALUDE_reduce_to_single_digit_l3357_335791

/-- Represents a single operation on a natural number as described in the problem -/
def Operation (n : ℕ) : ℕ := sorry

/-- Predicate that checks if a number is a single digit -/
def IsSingleDigit (n : ℕ) : Prop := n < 10

/-- Theorem stating that any natural number can be reduced to a single digit in at most 15 steps -/
theorem reduce_to_single_digit (N : ℕ) : 
  ∃ (sequence : Fin 15 → ℕ), 
    sequence 0 = N ∧ 
    (∀ i : Fin 14, sequence (i + 1) = Operation (sequence i)) ∧
    IsSingleDigit (sequence 14) :=
sorry

end NUMINAMATH_CALUDE_reduce_to_single_digit_l3357_335791


namespace NUMINAMATH_CALUDE_monotone_increasing_iff_a_in_range_l3357_335771

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1/2 then a^x else (2*a - 1)*x

theorem monotone_increasing_iff_a_in_range (a : ℝ) :
  Monotone (f a) ↔ a ∈ Set.Ici ((2 + Real.sqrt 3) / 2) :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_iff_a_in_range_l3357_335771


namespace NUMINAMATH_CALUDE_parabola_passes_through_points_parabola_general_form_l3357_335739

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem statement
theorem parabola_passes_through_points :
  (parabola (-1) = 0) ∧ (parabola 3 = 0) :=
by
  sorry

-- Verify the general form
theorem parabola_general_form (x : ℝ) :
  ∃ (b c : ℝ), parabola x = x^2 - b*x + c :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_passes_through_points_parabola_general_form_l3357_335739


namespace NUMINAMATH_CALUDE_costs_equal_at_60_l3357_335777

/-- Represents the pricing and discount options for appliances -/
structure AppliancePricing where
  washing_machine_price : ℕ
  cooker_price : ℕ
  option1_free_cookers : ℕ
  option2_discount : ℚ

/-- Calculates the cost for Option 1 -/
def option1_cost (p : AppliancePricing) (washing_machines : ℕ) (cookers : ℕ) : ℕ :=
  p.washing_machine_price * washing_machines + p.cooker_price * (cookers - p.option1_free_cookers)

/-- Calculates the cost for Option 2 -/
def option2_cost (p : AppliancePricing) (washing_machines : ℕ) (cookers : ℕ) : ℚ :=
  (p.washing_machine_price * washing_machines + p.cooker_price * cookers : ℚ) * p.option2_discount

/-- Theorem: Costs of Option 1 and Option 2 are equal when x = 60 -/
theorem costs_equal_at_60 (p : AppliancePricing) 
    (h1 : p.washing_machine_price = 800)
    (h2 : p.cooker_price = 200)
    (h3 : p.option1_free_cookers = 10)
    (h4 : p.option2_discount = 9/10) :
    (option1_cost p 10 60 : ℚ) = option2_cost p 10 60 := by
  sorry

end NUMINAMATH_CALUDE_costs_equal_at_60_l3357_335777


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3357_335775

theorem complex_modulus_problem (z : ℂ) : z = (1 - 3*I) / (1 + 2*I) → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3357_335775


namespace NUMINAMATH_CALUDE_football_team_analysis_l3357_335792

/-- Represents a football team's performance in a season -/
structure FootballTeam where
  total_matches : ℕ
  matches_played : ℕ
  matches_lost : ℕ
  points : ℕ

/-- Calculates the number of wins given the team's current state -/
def wins (team : FootballTeam) : ℕ :=
  (team.points - (team.matches_played - team.matches_lost)) / 2

/-- Calculates the maximum possible points after all matches -/
def max_points (team : FootballTeam) : ℕ :=
  team.points + (team.total_matches - team.matches_played) * 3

/-- Calculates the minimum number of wins needed in remaining matches to reach a goal -/
def min_wins_needed (team : FootballTeam) (goal : ℕ) : ℕ :=
  ((goal - team.points) + 2) / 3

theorem football_team_analysis (team : FootballTeam) 
  (h1 : team.total_matches = 16)
  (h2 : team.matches_played = 9)
  (h3 : team.matches_lost = 2)
  (h4 : team.points = 19) :
  wins team = 6 ∧ 
  max_points team = 40 ∧ 
  min_wins_needed team 34 = 4 := by
  sorry

end NUMINAMATH_CALUDE_football_team_analysis_l3357_335792


namespace NUMINAMATH_CALUDE_proposition_analysis_l3357_335754

-- Define the propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem statement
theorem proposition_analysis :
  p ∧ 
  ¬q ∧ 
  (p ∨ q) ∧ 
  (p ∧ ¬q) ∧ 
  ¬(p ∧ q) ∧ 
  ¬(¬p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_proposition_analysis_l3357_335754


namespace NUMINAMATH_CALUDE_min_value_sum_l3357_335751

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + 2*a*b + 4*b*c + 2*c*a = 16) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 →
    x^2 + 2*x*y + 4*y*z + 2*z*x = 16 → x + y + z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l3357_335751


namespace NUMINAMATH_CALUDE_focal_length_of_specific_conic_l3357_335765

/-- A conic section centered at the origin with coordinate axes as its axes of symmetry -/
structure ConicSection where
  /-- The eccentricity of the conic section -/
  eccentricity : ℝ
  /-- A point that the conic section passes through -/
  point : ℝ × ℝ

/-- The focal length of a conic section -/
def focalLength (c : ConicSection) : ℝ := sorry

/-- Theorem: The focal length of the specified conic section is 6√2 -/
theorem focal_length_of_specific_conic :
  let c : ConicSection := { eccentricity := Real.sqrt 2, point := (5, 4) }
  focalLength c = 6 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_focal_length_of_specific_conic_l3357_335765


namespace NUMINAMATH_CALUDE_tropical_storm_sally_rainfall_l3357_335760

theorem tropical_storm_sally_rainfall (day1 day2 day3 : ℝ) : 
  day2 = 5 * day1 →
  day3 = day1 + day2 - 6 →
  day3 = 18 →
  day1 = 4 := by
sorry

end NUMINAMATH_CALUDE_tropical_storm_sally_rainfall_l3357_335760


namespace NUMINAMATH_CALUDE_lcm_count_l3357_335783

theorem lcm_count : 
  ∃! (n : ℕ), n > 0 ∧ 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ k ∈ S, k > 0 ∧ Nat.lcm (9^9) (Nat.lcm (12^12) k) = 18^18) ∧
    (∀ k ∉ S, k > 0 → Nat.lcm (9^9) (Nat.lcm (12^12) k) ≠ 18^18)) :=
sorry

end NUMINAMATH_CALUDE_lcm_count_l3357_335783


namespace NUMINAMATH_CALUDE_total_tile_cost_l3357_335757

def courtyard_length : ℝ := 10
def courtyard_width : ℝ := 25
def tiles_per_sqft : ℝ := 4
def green_tile_percentage : ℝ := 0.4
def green_tile_cost : ℝ := 3
def red_tile_cost : ℝ := 1.5

theorem total_tile_cost : 
  let area := courtyard_length * courtyard_width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * green_tile_percentage
  let red_tiles := total_tiles * (1 - green_tile_percentage)
  green_tiles * green_tile_cost + red_tiles * red_tile_cost = 2100 := by
sorry

end NUMINAMATH_CALUDE_total_tile_cost_l3357_335757


namespace NUMINAMATH_CALUDE_average_trees_planted_l3357_335747

theorem average_trees_planted (trees_A trees_B trees_C : ℕ) : 
  trees_A = 225 →
  trees_B = trees_A + 48 →
  trees_C = trees_A - 24 →
  (trees_A + trees_B + trees_C) / 3 = 233 := by
  sorry

end NUMINAMATH_CALUDE_average_trees_planted_l3357_335747


namespace NUMINAMATH_CALUDE_car_driving_east_when_sun_setting_in_mirror_l3357_335744

-- Define the direction type
inductive Direction
| East
| West
| North
| South

-- Define the position of the sun
inductive SunPosition
| Setting
| Rising
| Overhead

-- Define the view of the sun
structure SunView where
  position : SunPosition
  throughMirror : Bool

-- Define the state of the car
structure CarState where
  direction : Direction
  sunView : SunView

-- Theorem statement
theorem car_driving_east_when_sun_setting_in_mirror 
  (car : CarState) : 
  car.sunView.position = SunPosition.Setting ∧ 
  car.sunView.throughMirror = true → 
  car.direction = Direction.East :=
sorry

end NUMINAMATH_CALUDE_car_driving_east_when_sun_setting_in_mirror_l3357_335744


namespace NUMINAMATH_CALUDE_distinct_bracelets_count_l3357_335729

/-- Represents a bracelet configuration -/
structure Bracelet :=
  (red : Nat)
  (blue : Nat)
  (green : Nat)

/-- Defines the specific bracelet configuration in the problem -/
def problem_bracelet : Bracelet :=
  { red := 1, blue := 2, green := 2 }

/-- Calculates the total number of beads in a bracelet -/
def total_beads (b : Bracelet) : Nat :=
  b.red + b.blue + b.green

/-- Represents the number of distinct bracelets -/
def distinct_bracelets (b : Bracelet) : Nat :=
  (Nat.factorial (total_beads b)) / 
  (Nat.factorial b.red * Nat.factorial b.blue * Nat.factorial b.green * 
   (total_beads b) * 2)

/-- Theorem stating that the number of distinct bracelets for the given configuration is 4 -/
theorem distinct_bracelets_count :
  distinct_bracelets problem_bracelet = 4 := by
  sorry

end NUMINAMATH_CALUDE_distinct_bracelets_count_l3357_335729


namespace NUMINAMATH_CALUDE_largest_c_for_negative_five_in_range_l3357_335761

theorem largest_c_for_negative_five_in_range : 
  ∀ c : ℝ, (∃ x : ℝ, x^2 + 4*x + c = -5) ↔ c ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_largest_c_for_negative_five_in_range_l3357_335761


namespace NUMINAMATH_CALUDE_correct_minutes_for_ninth_day_l3357_335785

/-- The number of minutes Julia needs to read on the 9th day to achieve the target average -/
def minutes_to_read_on_ninth_day (days_reading_80_min : ℕ) (days_reading_100_min : ℕ) (target_average : ℕ) (total_days : ℕ) : ℕ :=
  let total_minutes_read := days_reading_80_min * 80 + days_reading_100_min * 100
  let target_total_minutes := total_days * target_average
  target_total_minutes - total_minutes_read

/-- Theorem stating the correct number of minutes Julia needs to read on the 9th day -/
theorem correct_minutes_for_ninth_day :
  minutes_to_read_on_ninth_day 6 2 95 9 = 175 := by
  sorry

end NUMINAMATH_CALUDE_correct_minutes_for_ninth_day_l3357_335785


namespace NUMINAMATH_CALUDE_monitor_pixel_count_l3357_335784

/-- Calculates the total number of pixels on a monitor given its dimensions and pixel density. -/
def total_pixels (width : ℕ) (height : ℕ) (pixel_density : ℕ) : ℕ :=
  (width * pixel_density) * (height * pixel_density)

/-- Theorem: A monitor that is 21 inches wide and 12 inches tall with a pixel density of 100 dots per inch has 2,520,000 pixels. -/
theorem monitor_pixel_count :
  total_pixels 21 12 100 = 2520000 := by
  sorry

end NUMINAMATH_CALUDE_monitor_pixel_count_l3357_335784


namespace NUMINAMATH_CALUDE_b_performance_conditions_l3357_335793

/-- Represents a person's shooting performance -/
structure ShootingPerformance where
  average : ℝ
  variance : ℝ

/-- The shooting performances of A, C, and D -/
def performances : List ShootingPerformance := [
  ⟨9.7, 0.25⟩,  -- A
  ⟨9.3, 0.28⟩,  -- C
  ⟨9.6, 0.27⟩   -- D
]

/-- B's performance is the best and most stable -/
def b_is_best (m n : ℝ) : Prop :=
  ∀ p ∈ performances, m > p.average ∧ n < p.variance

/-- Theorem stating the conditions for B's performance -/
theorem b_performance_conditions (m n : ℝ) 
  (h : b_is_best m n) : m > 9.7 ∧ n < 0.25 := by
  sorry

#check b_performance_conditions

end NUMINAMATH_CALUDE_b_performance_conditions_l3357_335793


namespace NUMINAMATH_CALUDE_shooting_probabilities_l3357_335753

/-- Probability of shooter A hitting the target -/
def prob_A : ℝ := 0.7

/-- Probability of shooter B hitting the target -/
def prob_B : ℝ := 0.6

/-- Probability of shooter C hitting the target -/
def prob_C : ℝ := 0.5

/-- Probability that at least one person hits the target -/
def prob_at_least_one : ℝ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- Probability that exactly two people hit the target -/
def prob_exactly_two : ℝ := 
  prob_A * prob_B * (1 - prob_C) + 
  prob_A * (1 - prob_B) * prob_C + 
  (1 - prob_A) * prob_B * prob_C

theorem shooting_probabilities : 
  prob_at_least_one = 0.94 ∧ prob_exactly_two = 0.44 := by
  sorry

end NUMINAMATH_CALUDE_shooting_probabilities_l3357_335753


namespace NUMINAMATH_CALUDE_carnival_sales_proof_l3357_335721

/-- Represents the daily sales of popcorn in dollars -/
def daily_popcorn_sales : ℝ := 50

/-- Represents the daily sales of cotton candy in dollars -/
def daily_cotton_candy_sales : ℝ := 3 * daily_popcorn_sales

/-- Duration of the carnival in days -/
def carnival_duration : ℕ := 5

/-- Total expenses for rent and ingredients in dollars -/
def total_expenses : ℝ := 105

/-- Net earnings after expenses in dollars -/
def net_earnings : ℝ := 895

theorem carnival_sales_proof :
  daily_popcorn_sales * carnival_duration +
  daily_cotton_candy_sales * carnival_duration -
  total_expenses = net_earnings :=
by sorry

end NUMINAMATH_CALUDE_carnival_sales_proof_l3357_335721
