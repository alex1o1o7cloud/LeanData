import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_equation_solution_l3509_350950

theorem complex_modulus_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ Complex.abs (5 - 3 * Complex.I * x) = 7 ∧ x = Real.sqrt (8/3) := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equation_solution_l3509_350950


namespace NUMINAMATH_CALUDE_total_boys_across_grades_l3509_350996

/-- Represents the number of students in each grade level -/
structure GradeLevel where
  girls : ℕ
  boys : ℕ

/-- Calculates the total number of boys across all grade levels -/
def totalBoys (gradeA gradeB gradeC : GradeLevel) : ℕ :=
  gradeA.boys + gradeB.boys + gradeC.boys

/-- Theorem stating the total number of boys across three grade levels -/
theorem total_boys_across_grades (gradeA gradeB gradeC : GradeLevel) 
  (hA : gradeA.girls = 256 ∧ gradeA.girls = gradeA.boys + 52)
  (hB : gradeB.girls = 360 ∧ gradeB.boys = gradeB.girls - 40)
  (hC : gradeC.girls = 168 ∧ gradeC.boys = gradeC.girls) : 
  totalBoys gradeA gradeB gradeC = 692 := by
  sorry


end NUMINAMATH_CALUDE_total_boys_across_grades_l3509_350996


namespace NUMINAMATH_CALUDE_haleys_trees_l3509_350946

theorem haleys_trees (initial_trees : ℕ) : 
  (initial_trees - 4 + 5 = 10) → initial_trees = 9 := by
  sorry

end NUMINAMATH_CALUDE_haleys_trees_l3509_350946


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l3509_350904

/-- The number of ways to distribute candy among children with restrictions -/
def distribute_candy (total_candy : ℕ) (num_children : ℕ) (min_candy : ℕ) (max_candy : ℕ) : ℕ :=
  sorry

/-- Theorem stating the specific case of candy distribution -/
theorem candy_distribution_theorem :
  distribute_candy 40 3 2 19 = 171 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l3509_350904


namespace NUMINAMATH_CALUDE_length_AE_is_seven_l3509_350974

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (13, 14, 15)

-- Define the altitude from A
def altitude_A (t : Triangle) : ℝ × ℝ := sorry

-- Define point D where altitude intersects BC
def point_D (t : Triangle) : ℝ × ℝ := sorry

-- Define incircles of ABD and ACD
def incircle_ABD (t : Triangle) : Circle := sorry
def incircle_ACD (t : Triangle) : Circle := sorry

-- Define the common external tangent
def common_external_tangent (c1 c2 : Circle) : Line := sorry

-- Define point E where the common external tangent intersects AD
def point_E (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of AE
def length_AE (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem length_AE_is_seven (t : Triangle) :
  side_lengths t = (13, 14, 15) →
  length_AE t = 7 := by sorry

end NUMINAMATH_CALUDE_length_AE_is_seven_l3509_350974


namespace NUMINAMATH_CALUDE_floor_abs_plus_const_l3509_350961

theorem floor_abs_plus_const : 
  ⌊|(-47.3 : ℝ)| + 0.7⌋ = 48 := by
  sorry

end NUMINAMATH_CALUDE_floor_abs_plus_const_l3509_350961


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l3509_350981

/-- Theorem: Tangent line intersection for two circles
    Given two circles:
    - Circle 1 with radius 3 and center (0, 0)
    - Circle 2 with radius 5 and center (12, 0)
    The x-coordinate of the point where a line tangent to both circles
    intersects the x-axis (to the right of the origin) is 9/2.
-/
theorem tangent_line_intersection (x : ℚ) : 
  (∃ y : ℚ, (x^2 + y^2 = 3^2 ∧ ((x - 12)^2 + y^2 = 5^2))) → x = 9/2 := by
  sorry

#check tangent_line_intersection

end NUMINAMATH_CALUDE_tangent_line_intersection_l3509_350981


namespace NUMINAMATH_CALUDE_bamboo_nine_sections_l3509_350948

theorem bamboo_nine_sections (a : ℕ → ℚ) (d : ℚ) :
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → a n = a 1 + (n - 1) * d) →
  a 1 + a 2 + a 3 + a 4 = 3 →
  a 7 + a 8 + a 9 = 4 →
  a 1 = 13 / 22 :=
sorry

end NUMINAMATH_CALUDE_bamboo_nine_sections_l3509_350948


namespace NUMINAMATH_CALUDE_ball_max_height_l3509_350932

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 10

/-- The maximum height reached by the ball -/
def max_height : ℝ := 135

/-- Theorem stating that the maximum height reached by the ball is 135 meters -/
theorem ball_max_height : 
  ∀ t : ℝ, h t ≤ max_height :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l3509_350932


namespace NUMINAMATH_CALUDE_ivanov_exaggerating_l3509_350935

-- Define the probabilities of machine breakdowns
def p1 : ℝ := 0.4
def p2 : ℝ := 0.3
def p3 : ℝ := 0

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the expected number of breakdowns per day
def expected_breakdowns_per_day : ℝ := p1 + p2 + p3

-- Define the expected number of breakdowns per week
def expected_breakdowns_per_week : ℝ := expected_breakdowns_per_day * days_in_week

-- Theorem statement
theorem ivanov_exaggerating : expected_breakdowns_per_week < 12 := by
  sorry

end NUMINAMATH_CALUDE_ivanov_exaggerating_l3509_350935


namespace NUMINAMATH_CALUDE_line_through_point_l3509_350947

theorem line_through_point (k : ℚ) : 
  (2 - 3 * k * (-3) = 5 * 1) → k = 1/3 := by sorry

end NUMINAMATH_CALUDE_line_through_point_l3509_350947


namespace NUMINAMATH_CALUDE_complex_real_condition_l3509_350920

theorem complex_real_condition (a : ℝ) : 
  let Z : ℂ := (a - 5) / (a^2 + 4*a - 5) + (a^2 + 2*a - 15) * Complex.I
  (Z.im = 0 ∧ (a^2 + 4*a - 5) ≠ 0) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3509_350920


namespace NUMINAMATH_CALUDE_two_distinct_roots_root_one_case_l3509_350997

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - (m - 3) * x - m

-- Theorem stating that the equation has two distinct real roots for all m
theorem two_distinct_roots (m : ℝ) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0 :=
sorry

-- Theorem for the case when one root is 1
theorem root_one_case :
  ∃ m : ℝ, quadratic_equation m 1 = 0 ∧ 
  (∃ x : ℝ, x ≠ 1 ∧ quadratic_equation m x = 0) ∧
  m = 2 ∧
  (∃ x : ℝ, x = -2 ∧ quadratic_equation m x = 0) :=
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_root_one_case_l3509_350997


namespace NUMINAMATH_CALUDE_unique_acute_prime_angled_triangle_l3509_350943

-- Define a structure for a triangle with three angles
structure Triangle where
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define what it means for a triangle to be acute
def isAcute (t : Triangle) : Prop := t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90

-- Define what it means for a triangle to have prime angles
def hasPrimeAngles (t : Triangle) : Prop := 
  isPrime t.angle1 ∧ isPrime t.angle2 ∧ isPrime t.angle3

-- Define what it means for a triangle to be valid (sum of angles is 180°)
def isValidTriangle (t : Triangle) : Prop := t.angle1 + t.angle2 + t.angle3 = 180

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop := 
  t.angle1 = t.angle2 ∨ t.angle2 = t.angle3 ∨ t.angle3 = t.angle1

-- Theorem statement
theorem unique_acute_prime_angled_triangle : 
  ∃! t : Triangle, isAcute t ∧ hasPrimeAngles t ∧ isValidTriangle t ∧
  t.angle1 = 2 ∧ t.angle2 = 89 ∧ t.angle3 = 89 ∧ isIsosceles t :=
sorry

end NUMINAMATH_CALUDE_unique_acute_prime_angled_triangle_l3509_350943


namespace NUMINAMATH_CALUDE_game_size_proof_l3509_350975

/-- Given a game download scenario where:
  * 310 MB has already been downloaded
  * The remaining download speed is 3 MB/minute
  * It takes 190 more minutes to finish the download
  Prove that the total size of the game is 880 MB -/
theorem game_size_proof (already_downloaded : ℕ) (download_speed : ℕ) (remaining_time : ℕ) :
  already_downloaded = 310 →
  download_speed = 3 →
  remaining_time = 190 →
  already_downloaded + download_speed * remaining_time = 880 :=
by sorry

end NUMINAMATH_CALUDE_game_size_proof_l3509_350975


namespace NUMINAMATH_CALUDE_increasing_odd_sum_nonpositive_l3509_350939

/-- A function f: ℝ → ℝ is increasing if for all x, y ∈ ℝ, x < y implies f(x) < f(y) -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- A function f: ℝ → ℝ is odd if for all x ∈ ℝ, f(-x) = -f(x) -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- Theorem: If f is an increasing and odd function on ℝ, and a and b are real numbers
    such that a + b ≤ 0, then f(a) + f(b) ≤ 0 -/
theorem increasing_odd_sum_nonpositive
  (f : ℝ → ℝ) (hf_inc : IsIncreasing f) (hf_odd : IsOdd f)
  (a b : ℝ) (hab : a + b ≤ 0) :
  f a + f b ≤ 0 := by
  sorry


end NUMINAMATH_CALUDE_increasing_odd_sum_nonpositive_l3509_350939


namespace NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l3509_350941

theorem sqrt_x_minus_3_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_3_real_l3509_350941


namespace NUMINAMATH_CALUDE_cookie_ratio_anna_to_tim_l3509_350949

/-- Represents the distribution of cookies among recipients --/
structure CookieDistribution where
  total : Nat
  tim : Nat
  mike : Nat
  fridge : Nat

/-- Calculates the number of cookies given to Anna --/
def cookiesForAnna (d : CookieDistribution) : Nat :=
  d.total - (d.tim + d.mike + d.fridge)

/-- Represents a ratio as a pair of natural numbers --/
structure Ratio where
  numerator : Nat
  denominator : Nat

/-- Theorem stating the ratio of cookies given to Anna to cookies given to Tim --/
theorem cookie_ratio_anna_to_tim (d : CookieDistribution)
  (h1 : d.total = 256)
  (h2 : d.tim = 15)
  (h3 : d.mike = 23)
  (h4 : d.fridge = 188) :
  Ratio.mk (cookiesForAnna d) d.tim = Ratio.mk 2 1 := by
  sorry

#check cookie_ratio_anna_to_tim

end NUMINAMATH_CALUDE_cookie_ratio_anna_to_tim_l3509_350949


namespace NUMINAMATH_CALUDE_solve_colored_paper_problem_l3509_350968

def colored_paper_problem (initial : ℕ) (given_per_friend : ℕ) (num_friends : ℕ) (bought : ℕ) (current : ℕ) : Prop :=
  initial + bought - (given_per_friend * num_friends) = current

theorem solve_colored_paper_problem :
  ∃ initial : ℕ, colored_paper_problem initial 11 2 27 63 ∧ initial = 58 := by
  sorry

end NUMINAMATH_CALUDE_solve_colored_paper_problem_l3509_350968


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3509_350900

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes forms a 30° angle with the x-axis,
    then its eccentricity is 2√3/3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.tan (π / 6)) :
  let e := Real.sqrt (1 + (b / a)^2)
  e = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3509_350900


namespace NUMINAMATH_CALUDE_chinese_sexagenary_cycle_properties_l3509_350965

/-- Represents the Chinese sexagenary cycle -/
structure SexagenaryCycle where
  heavenly_stems : Fin 10
  earthly_branches : Fin 12

/-- Calculates the next year with the same combination in the sexagenary cycle -/
def next_same_year (year : Int) : Int :=
  year + 60

/-- Calculates the previous year with the same combination in the sexagenary cycle -/
def prev_same_year (year : Int) : Int :=
  year - 60

/-- Calculates a year with a specific offset in the cycle -/
def year_with_offset (base_year : Int) (offset : Int) : Int :=
  base_year + offset

theorem chinese_sexagenary_cycle_properties :
  let ren_wu_2002 : SexagenaryCycle := ⟨9, 7⟩ -- Ren (9th stem), Wu (7th branch)
  -- 1. Next Ren Wu year
  (next_same_year 2002 = 2062) ∧
  -- 2. Jiawu War year (Jia Wu)
  (year_with_offset 2002 (-108) = 1894) ∧
  -- 3. Wuxu Reform year (Wu Xu)
  (year_with_offset 2002 (-104) = 1898) ∧
  -- 4. Geng Shen years in the 20th century
  (year_with_offset 2002 (-82) = 1920) ∧
  (year_with_offset 2002 (-22) = 1980) := by
  sorry

end NUMINAMATH_CALUDE_chinese_sexagenary_cycle_properties_l3509_350965


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3509_350903

theorem quadratic_no_real_roots :
  ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := by
  sorry

#check quadratic_no_real_roots

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3509_350903


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l3509_350917

/-- Given that ax^2 + 5x - 2 > 0 has solution set {x | 1/2 < x < 2}, prove:
    1. a = -2
    2. The solution set of ax^2 - 5x + a^2 - 1 > 0 is {x | -3 < x < 1/2} -/
theorem quadratic_inequality_problem 
  (h : ∀ x, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) :
  (a = -2) ∧ 
  (∀ x, a*x^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l3509_350917


namespace NUMINAMATH_CALUDE_calculate_otimes_expression_l3509_350940

-- Define the ⊗ operation
def otimes (a b : ℚ) : ℚ := (a + b) / (a - b)

-- The main theorem to prove
theorem calculate_otimes_expression :
  (otimes (otimes 8 6) (otimes 2 1)) = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_otimes_expression_l3509_350940


namespace NUMINAMATH_CALUDE_normal_vector_to_ellipsoid_l3509_350993

/-- The ellipsoid equation -/
def F (x y z : ℝ) : ℝ := x^2 + 2*y^2 + 3*z^2 - 6

/-- The point on the ellipsoid -/
def M₀ : ℝ × ℝ × ℝ := (1, -1, 1)

/-- The proposed normal vector -/
def n : ℝ × ℝ × ℝ := (2, -4, 6)

theorem normal_vector_to_ellipsoid :
  let (x₀, y₀, z₀) := M₀
  F x₀ y₀ z₀ = 0 ∧ 
  n = (2*x₀, 4*y₀, 6*z₀) :=
by sorry

end NUMINAMATH_CALUDE_normal_vector_to_ellipsoid_l3509_350993


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l3509_350929

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : ℕ := 
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)

theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l3509_350929


namespace NUMINAMATH_CALUDE_salary_increase_proof_l3509_350938

/-- Proves that given an employee's new annual salary of $90,000 after a 38.46153846153846% increase, the amount of the salary increase is $25,000. -/
theorem salary_increase_proof (new_salary : ℝ) (percent_increase : ℝ) 
  (h1 : new_salary = 90000)
  (h2 : percent_increase = 38.46153846153846) : 
  new_salary - (new_salary / (1 + percent_increase / 100)) = 25000 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l3509_350938


namespace NUMINAMATH_CALUDE_tempo_insured_fraction_l3509_350911

/-- Represents the insurance details of a tempo --/
structure TempoInsurance where
  premium_rate : Rat
  premium_amount : Rat
  original_value : Rat

/-- Calculates the fraction of the original value that is insured --/
def insured_fraction (insurance : TempoInsurance) : Rat :=
  (insurance.premium_amount / insurance.premium_rate) / insurance.original_value

/-- Theorem stating that for the given insurance details, the insured fraction is 5/7 --/
theorem tempo_insured_fraction :
  let insurance : TempoInsurance := {
    premium_rate := 3 / 100,
    premium_amount := 300,
    original_value := 14000
  }
  insured_fraction insurance = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_tempo_insured_fraction_l3509_350911


namespace NUMINAMATH_CALUDE_problem_statement_l3509_350927

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ 
  (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3509_350927


namespace NUMINAMATH_CALUDE_linear_function_shift_l3509_350978

/-- 
A linear function y = -2x + b is shifted 3 units upwards.
This theorem proves that if the shifted function passes through the point (2, 0),
then b = 1.
-/
theorem linear_function_shift (b : ℝ) : 
  (∀ x y : ℝ, y = -2 * x + b + 3 → (x = 2 ∧ y = 0) → b = 1) := by
sorry

end NUMINAMATH_CALUDE_linear_function_shift_l3509_350978


namespace NUMINAMATH_CALUDE_system_solution_l3509_350944

theorem system_solution :
  ∃ (x y : ℚ), (7 * x = -5 - 3 * y) ∧ (4 * x = 5 * y - 36) ∧ (x = -41/11) ∧ (y = 232/33) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3509_350944


namespace NUMINAMATH_CALUDE_temperature_conversion_l3509_350933

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 68 → t = 20 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l3509_350933


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3509_350964

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  ∀ y : ℝ, y = 1/a + 4/b → y ≥ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3509_350964


namespace NUMINAMATH_CALUDE_unique_k_value_l3509_350983

/-- The polynomial expression -/
def polynomial (k : ℚ) (x y : ℚ) : ℚ := x^2 + 4*x*y + 2*x + k*y - 3*k

/-- Condition for integer factorization -/
def has_integer_factorization (k : ℚ) : Prop :=
  ∃ (A B C D E F : ℤ), 
    ∀ (x y : ℚ), polynomial k x y = (A*x + B*y + C) * (D*x + E*y + F)

/-- Condition for non-negative discriminant of the quadratic part -/
def has_nonnegative_discriminant (k : ℚ) : Prop :=
  (4:ℚ)^2 - 4*1*0 ≥ 0

/-- The main theorem -/
theorem unique_k_value : 
  (∃! k : ℚ, has_integer_factorization k ∧ has_nonnegative_discriminant k) ∧
  (∀ k : ℚ, has_integer_factorization k ∧ has_nonnegative_discriminant k → k = 0) :=
sorry

end NUMINAMATH_CALUDE_unique_k_value_l3509_350983


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l3509_350956

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- State the theorem
theorem f_even_and_increasing :
  (∀ x, f x = f (-x)) ∧  -- f is an even function
  (∀ x y, 0 < x → x < y → f x < f y) -- f is monotonically increasing on (0,+∞)
  := by sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l3509_350956


namespace NUMINAMATH_CALUDE_yard_trees_l3509_350971

/-- The number of trees in a yard with given length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  yard_length / tree_spacing + 1

/-- Theorem: In a 400-meter yard with trees spaced 16 meters apart, there are 26 trees -/
theorem yard_trees : num_trees 400 16 = 26 := by
  sorry

end NUMINAMATH_CALUDE_yard_trees_l3509_350971


namespace NUMINAMATH_CALUDE_football_team_girls_l3509_350921

theorem football_team_girls (total : ℕ) (attended : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 30 →
  attended = 18 →
  girls + boys = total →
  attended = boys + (girls / 3) →
  girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_football_team_girls_l3509_350921


namespace NUMINAMATH_CALUDE_problem_solution_l3509_350987

def f (x : ℝ) := |x + 1| + |x - 3|

theorem problem_solution :
  (∀ x : ℝ, f x < 6 ↔ -2 < x ∧ x < 4) ∧
  (∀ a : ℝ, (∃ x : ℝ, f x = |a - 2|) → (a ≥ 6 ∨ a ≤ -2)) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3509_350987


namespace NUMINAMATH_CALUDE_dormitory_students_count_unique_solution_l3509_350957

/-- Represents the number of students in the dormitory -/
def n : ℕ := 6

/-- Represents the number of administrators -/
def m : ℕ := 3

/-- The total number of greeting cards used -/
def total_cards : ℕ := 51

/-- Theorem stating that the number of students in the dormitory is 6 -/
theorem dormitory_students_count :
  (n * (n - 1)) / 2 + n * m + m = total_cards :=
by sorry

/-- Theorem stating that n is the unique solution for the given conditions -/
theorem unique_solution (k : ℕ) :
  (k * (k - 1)) / 2 + k * m + m = total_cards → k = n :=
by sorry

end NUMINAMATH_CALUDE_dormitory_students_count_unique_solution_l3509_350957


namespace NUMINAMATH_CALUDE_roots_have_different_signs_l3509_350910

/-- A quadratic polynomial f(x) = ax^2 + bx + c -/
def quadraticPolynomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem roots_have_different_signs (a b c : ℝ) (ha : a ≠ 0) :
  (quadraticPolynomial a b c (1/a)) * (quadraticPolynomial a b c c) < 0 →
  ∃ x₁ x₂ : ℝ, x₁ * x₂ < 0 ∧ ∀ x, quadraticPolynomial a b c x = 0 ↔ x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_roots_have_different_signs_l3509_350910


namespace NUMINAMATH_CALUDE_initial_bird_families_l3509_350945

/-- The number of bird families that flew away for winter. -/
def flew_away : ℕ := 7

/-- The difference between the number of bird families that stayed and those that flew away. -/
def difference : ℕ := 73

/-- The total number of bird families initially living near the mountain. -/
def total_families : ℕ := flew_away + (flew_away + difference)

theorem initial_bird_families :
  total_families = 87 :=
sorry

end NUMINAMATH_CALUDE_initial_bird_families_l3509_350945


namespace NUMINAMATH_CALUDE_hike_remaining_distance_l3509_350960

/-- Calculates the remaining distance of a hike given the total distance and distance already hiked. -/
def remaining_distance (total : ℕ) (hiked : ℕ) : ℕ :=
  total - hiked

/-- Proves that for a 36-mile hike with 9 miles already hiked, 27 miles remain. -/
theorem hike_remaining_distance :
  remaining_distance 36 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_hike_remaining_distance_l3509_350960


namespace NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l3509_350966

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem largest_perfect_square_factor_of_1800 :
  ∃ (n : ℕ), is_perfect_square n ∧ is_factor n 1800 ∧
  ∀ (m : ℕ), is_perfect_square m → is_factor m 1800 → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_largest_perfect_square_factor_of_1800_l3509_350966


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3509_350930

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|) ∧
  ¬(|x + 1| + |x - 1| = 2 * |x| → x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3509_350930


namespace NUMINAMATH_CALUDE_caras_cat_catch_proof_l3509_350905

/-- The number of animals Cara's cat catches given Martha's cat's catch -/
def caras_cat_catch (marthas_rats : ℕ) (marthas_birds : ℕ) : ℕ :=
  5 * (marthas_rats + marthas_birds) - 3

theorem caras_cat_catch_proof :
  caras_cat_catch 3 7 = 47 := by
  sorry

end NUMINAMATH_CALUDE_caras_cat_catch_proof_l3509_350905


namespace NUMINAMATH_CALUDE_cos_pi_half_minus_two_alpha_l3509_350972

theorem cos_pi_half_minus_two_alpha (α : ℝ) (h : Real.tan α = 2) : 
  Real.cos (π / 2 - 2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_half_minus_two_alpha_l3509_350972


namespace NUMINAMATH_CALUDE_percent_problem_l3509_350986

theorem percent_problem (x : ℝ) : 0.01 = (10 / 100) * x → x = 0.1 := by sorry

end NUMINAMATH_CALUDE_percent_problem_l3509_350986


namespace NUMINAMATH_CALUDE_loan_repayment_equality_l3509_350994

/-- Represents the loan scenario described in the problem -/
structure LoanScenario where
  M : ℝ  -- Initial loan amount in million yuan
  x : ℝ  -- Monthly repayment amount in million yuan
  r : ℝ  -- Monthly interest rate (as a decimal)
  n : ℕ  -- Number of months for repayment

/-- The theorem representing the loan repayment equality -/
theorem loan_repayment_equality (scenario : LoanScenario) 
  (h_r : scenario.r = 0.05)
  (h_n : scenario.n = 20) : 
  scenario.n * scenario.x = scenario.M * (1 + scenario.r) ^ scenario.n :=
sorry

end NUMINAMATH_CALUDE_loan_repayment_equality_l3509_350994


namespace NUMINAMATH_CALUDE_function_max_min_implies_m_range_l3509_350916

/-- The function f(x) = x^2 - 2x + 3 on [0, m] with max 3 and min 2 implies m ∈ [1, 2] -/
theorem function_max_min_implies_m_range 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 - 2*x + 3) 
  (m : ℝ) 
  (h_max : ∃ x ∈ Set.Icc 0 m, ∀ y ∈ Set.Icc 0 m, f y ≤ f x)
  (h_min : ∃ x ∈ Set.Icc 0 m, ∀ y ∈ Set.Icc 0 m, f x ≤ f y)
  (h_max_val : ∃ x ∈ Set.Icc 0 m, f x = 3)
  (h_min_val : ∃ x ∈ Set.Icc 0 m, f x = 2) :
  m ∈ Set.Icc 1 2 :=
sorry

end NUMINAMATH_CALUDE_function_max_min_implies_m_range_l3509_350916


namespace NUMINAMATH_CALUDE_base8_to_base10_conversion_l3509_350954

/-- Converts a base 8 number to base 10 -/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base 8 representation of 246₈ -/
def base8Number : List Nat := [6, 4, 2]

theorem base8_to_base10_conversion :
  base8ToBase10 base8Number = 166 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_conversion_l3509_350954


namespace NUMINAMATH_CALUDE_point_P_in_quadrant_III_l3509_350908

def point_in_quadrant_III (x y : ℝ) : Prop :=
  x < 0 ∧ y < 0

theorem point_P_in_quadrant_III :
  point_in_quadrant_III (-1 : ℝ) (-2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_point_P_in_quadrant_III_l3509_350908


namespace NUMINAMATH_CALUDE_function_and_intersection_points_l3509_350922

noncomputable def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem function_and_intersection_points 
  (b c d : ℝ) 
  (h1 : f b c d 0 = 2) 
  (h2 : (6 : ℝ) * (-1) - f b c d (-1) + 7 = 0) 
  (h3 : (6 : ℝ) = (3 * (-1)^2 + 2*b*(-1) + c)) :
  (∀ x, f b c d x = x^3 - 3*x^2 - 3*x + 2) ∧
  (∀ a, (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    f b c d x₁ = (3/2)*x₁^2 - 9*x₁ + a + 2 ∧
    f b c d x₂ = (3/2)*x₂^2 - 9*x₂ + a + 2 ∧
    f b c d x₃ = (3/2)*x₃^2 - 9*x₃ + a + 2) →
  2 < a ∧ a < 5/2) :=
by sorry

end NUMINAMATH_CALUDE_function_and_intersection_points_l3509_350922


namespace NUMINAMATH_CALUDE_henry_trays_capacity_l3509_350909

/-- The number of trays Henry picked up from the first table -/
def trays_table1 : ℕ := 29

/-- The number of trays Henry picked up from the second table -/
def trays_table2 : ℕ := 52

/-- The total number of trips Henry made -/
def total_trips : ℕ := 9

/-- The number of trays Henry could carry at a time -/
def trays_per_trip : ℕ := (trays_table1 + trays_table2) / total_trips

theorem henry_trays_capacity : trays_per_trip = 9 := by
  sorry

end NUMINAMATH_CALUDE_henry_trays_capacity_l3509_350909


namespace NUMINAMATH_CALUDE_intersection_area_is_525_l3509_350976

/-- A cube with edge length 30 units -/
def Cube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 30}

/-- Point A of the cube -/
def A : Fin 3 → ℝ := λ _ ↦ 0

/-- Point B of the cube -/
def B : Fin 3 → ℝ := λ i ↦ if i = 0 then 30 else 0

/-- Point C of the cube -/
def C : Fin 3 → ℝ := λ i ↦ if i = 2 then 30 else B i

/-- Point D of the cube -/
def D : Fin 3 → ℝ := λ _ ↦ 30

/-- Point P on edge AB -/
def P : Fin 3 → ℝ := λ i ↦ if i = 0 then 10 else 0

/-- Point Q on edge BC -/
def Q : Fin 3 → ℝ := λ i ↦ if i = 0 then 30 else if i = 2 then 20 else 0

/-- Point R on edge CD -/
def R : Fin 3 → ℝ := λ i ↦ if i = 1 then 15 else 30

/-- The plane PQR -/
def PlanePQR : Set (Fin 3 → ℝ) :=
  {x | 3 * x 0 + 2 * x 1 - 3 * x 2 = 30}

/-- The intersection of the cube and the plane PQR -/
def Intersection : Set (Fin 3 → ℝ) :=
  Cube ∩ PlanePQR

/-- The area of the intersection -/
noncomputable def IntersectionArea : ℝ := sorry

theorem intersection_area_is_525 :
  IntersectionArea = 525 := by sorry

end NUMINAMATH_CALUDE_intersection_area_is_525_l3509_350976


namespace NUMINAMATH_CALUDE_initial_average_customers_l3509_350913

theorem initial_average_customers (x : ℕ) (today_customers : ℕ) (new_average : ℕ) 
  (h1 : x = 1)
  (h2 : today_customers = 120)
  (h3 : new_average = 90)
  : ∃ initial_average : ℕ, initial_average = 60 ∧ 
    (initial_average * x + today_customers) / (x + 1) = new_average :=
by
  sorry

end NUMINAMATH_CALUDE_initial_average_customers_l3509_350913


namespace NUMINAMATH_CALUDE_right_triangle_acute_angle_ratio_l3509_350985

theorem right_triangle_acute_angle_ratio (α β : ℝ) : 
  α > 0 ∧ β > 0 ∧  -- Angles are positive
  α + β = 90 ∧     -- Sum of acute angles in a right triangle is 90°
  β = 5 * α →      -- One angle is 5 times the other
  β = 75 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angle_ratio_l3509_350985


namespace NUMINAMATH_CALUDE_sin_negative_135_degrees_l3509_350919

theorem sin_negative_135_degrees : Real.sin (-(135 * π / 180)) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_135_degrees_l3509_350919


namespace NUMINAMATH_CALUDE_beanie_babies_total_l3509_350926

/-- The number of beanie babies Lori has -/
def lori_beanie_babies : ℕ := 300

/-- The number of beanie babies Sydney has -/
def sydney_beanie_babies : ℕ := lori_beanie_babies / 15

/-- The initial number of beanie babies Jake has -/
def jake_initial_beanie_babies : ℕ := 2 * sydney_beanie_babies

/-- The number of additional beanie babies Jake gained -/
def jake_additional_beanie_babies : ℕ := (jake_initial_beanie_babies * 20) / 100

/-- The total number of beanie babies Jake has after gaining more -/
def jake_total_beanie_babies : ℕ := jake_initial_beanie_babies + jake_additional_beanie_babies

/-- The total number of beanie babies all three have -/
def total_beanie_babies : ℕ := lori_beanie_babies + sydney_beanie_babies + jake_total_beanie_babies

theorem beanie_babies_total : total_beanie_babies = 368 := by
  sorry

end NUMINAMATH_CALUDE_beanie_babies_total_l3509_350926


namespace NUMINAMATH_CALUDE_price_difference_proof_l3509_350912

def original_price : ℝ := 120.00
def tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.25

def amy_total : ℝ := original_price * (1 + tax_rate) * (1 - discount_rate)
def bob_total : ℝ := original_price * (1 - discount_rate) * (1 + tax_rate)
def carla_total : ℝ := original_price * (1 + tax_rate) * (1 - discount_rate) * (1 + tax_rate)

theorem price_difference_proof :
  carla_total - amy_total = 6.744 ∧ carla_total - bob_total = 6.744 :=
by sorry

end NUMINAMATH_CALUDE_price_difference_proof_l3509_350912


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3509_350924

/-- The sum of an arithmetic sequence with first term 5, common difference 3, and 15 terms -/
def arithmetic_sum : ℕ := 
  let a₁ : ℕ := 5  -- first term
  let d : ℕ := 3   -- common difference
  let n : ℕ := 15  -- number of terms
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum : arithmetic_sum = 390 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3509_350924


namespace NUMINAMATH_CALUDE_milk_container_problem_l3509_350901

theorem milk_container_problem (A B C : ℝ) : 
  A > 0 →  -- A is positive (container capacity)
  B = 0.375 * A →  -- B is 62.5% less than A
  C = A - B →  -- C contains the rest of the milk
  C - 152 = B + 152 →  -- After transfer, B and C are equal
  A = 608 :=
by
  sorry

end NUMINAMATH_CALUDE_milk_container_problem_l3509_350901


namespace NUMINAMATH_CALUDE_diana_wins_probability_l3509_350989

/-- The number of sides on Apollo's die -/
def apollo_sides : ℕ := 8

/-- The number of sides on Diana's die -/
def diana_sides : ℕ := 5

/-- The probability that Diana's roll is larger than Apollo's roll -/
def probability_diana_wins : ℚ := 1/4

/-- Theorem stating that the probability of Diana winning is 1/4 -/
theorem diana_wins_probability : 
  probability_diana_wins = 1/4 := by sorry

end NUMINAMATH_CALUDE_diana_wins_probability_l3509_350989


namespace NUMINAMATH_CALUDE_expression_simplification_l3509_350928

theorem expression_simplification (x y : ℝ) : 
  x^2*y - 3*x*y^2 + 2*y*x^2 - y^2*x = 3*x^2*y - 4*x*y^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3509_350928


namespace NUMINAMATH_CALUDE_extremum_and_range_l3509_350942

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - a*x + b

-- Theorem statement
theorem extremum_and_range :
  ∀ a b : ℝ,
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f a b x ≥ f a b 2) ∧
  (f a b 2 = -8) →
  (a = 12 ∧ b = 8) ∧
  (∀ x ∈ Set.Icc (-3) 3, -8 ≤ f 12 8 x ∧ f 12 8 x ≤ 24) :=
by sorry

end NUMINAMATH_CALUDE_extremum_and_range_l3509_350942


namespace NUMINAMATH_CALUDE_compute_expression_l3509_350998

theorem compute_expression : 12 - 4 * (5 - 10)^3 = 512 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l3509_350998


namespace NUMINAMATH_CALUDE_inequality_proof_l3509_350984

theorem inequality_proof (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos₁ : a₁ > 0) (h_pos₂ : a₂ > 0) (h_pos₃ : a₃ > 0) (h_pos₄ : a₄ > 0)
  (h_distinct₁₂ : a₁ ≠ a₂) (h_distinct₁₃ : a₁ ≠ a₃) (h_distinct₁₄ : a₁ ≠ a₄)
  (h_distinct₂₃ : a₂ ≠ a₃) (h_distinct₂₄ : a₂ ≠ a₄) (h_distinct₃₄ : a₃ ≠ a₄) :
  a₁^3 / (a₂ - a₃)^2 + a₂^3 / (a₃ - a₄)^2 + a₃^3 / (a₄ - a₁)^2 + a₄^3 / (a₁ - a₂)^2 
  > a₁ + a₂ + a₃ + a₄ :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3509_350984


namespace NUMINAMATH_CALUDE_car_meeting_points_distance_prove_car_meeting_points_distance_l3509_350958

/-- Given two cars starting from points A and B, if they meet at a point 108 km from B, 
    then continue to each other's starting points and return, meeting again at a point 84 km from A, 
    the distance between their two meeting points is 48 km. -/
theorem car_meeting_points_distance : ℝ → Prop :=
  fun d =>
    let first_meeting := d - 108
    let second_meeting := 84
    first_meeting - second_meeting = 48

/-- Proof of the theorem -/
theorem prove_car_meeting_points_distance : ∃ d : ℝ, car_meeting_points_distance d :=
sorry

end NUMINAMATH_CALUDE_car_meeting_points_distance_prove_car_meeting_points_distance_l3509_350958


namespace NUMINAMATH_CALUDE_convention_handshakes_count_l3509_350925

/-- The number of handshakes at the Interregional Mischief Convention --/
def convention_handshakes (n_gremlins n_imps n_disagreeing_imps n_affected_gremlins : ℕ) : ℕ :=
  let gremlin_handshakes := n_gremlins * (n_gremlins - 1) / 2
  let normal_imp_gremlin_handshakes := (n_imps - n_disagreeing_imps) * n_gremlins
  let affected_imp_gremlin_handshakes := n_disagreeing_imps * (n_gremlins - n_affected_gremlins)
  gremlin_handshakes + normal_imp_gremlin_handshakes + affected_imp_gremlin_handshakes

/-- Theorem stating the number of handshakes at the convention --/
theorem convention_handshakes_count : convention_handshakes 30 20 5 10 = 985 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_count_l3509_350925


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l3509_350914

/-- A type representing a circular arrangement of 60 numbers -/
def CircularArrangement := Fin 60 → ℕ

/-- Predicate checking if a given arrangement satisfies all conditions -/
def SatisfiesConditions (arr : CircularArrangement) : Prop :=
  (∀ i : Fin 60, (arr i + arr ((i + 2) % 60)) % 2 = 0) ∧
  (∀ i : Fin 60, (arr i + arr ((i + 3) % 60)) % 3 = 0) ∧
  (∀ i : Fin 60, (arr i + arr ((i + 7) % 60)) % 7 = 0)

/-- Predicate checking if an arrangement is a permutation of 1 to 60 -/
def IsValidArrangement (arr : CircularArrangement) : Prop :=
  (∀ n : ℕ, n ∈ Finset.range 60 → ∃ i : Fin 60, arr i = n + 1) ∧
  (∀ i j : Fin 60, arr i = arr j → i = j)

/-- Theorem stating the impossibility of the arrangement -/
theorem no_valid_arrangement :
  ¬ ∃ arr : CircularArrangement, IsValidArrangement arr ∧ SatisfiesConditions arr :=
sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l3509_350914


namespace NUMINAMATH_CALUDE_square_equation_result_l3509_350991

theorem square_equation_result (a b c : ℤ) 
  (h : (a + 3)^2 + (b + 4)^2 - (c + 5)^2 = a^2 + b^2 - c^2) :
  ∃ k : ℤ, a^2 + b^2 - c^2 = k^2 / 25 := by
sorry

end NUMINAMATH_CALUDE_square_equation_result_l3509_350991


namespace NUMINAMATH_CALUDE_employee_pay_l3509_350970

theorem employee_pay (total_pay : ℝ) (a_pay : ℝ) (b_pay : ℝ) :
  total_pay = 550 →
  a_pay = 1.5 * b_pay →
  a_pay + b_pay = total_pay →
  b_pay = 220 := by
sorry

end NUMINAMATH_CALUDE_employee_pay_l3509_350970


namespace NUMINAMATH_CALUDE_robotics_club_enrollment_l3509_350988

theorem robotics_club_enrollment (total : ℕ) (engineering : ℕ) (computer_science : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : engineering = 45)
  (h3 : computer_science = 35)
  (h4 : both = 25) :
  total - (engineering + computer_science - both) = 25 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_enrollment_l3509_350988


namespace NUMINAMATH_CALUDE_parabola_directrix_l3509_350995

/-- A parabola is defined by its equation in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The directrix of a parabola is a line parallel to the x-axis -/
structure Directrix where
  y : ℝ

/-- Given a parabola y = (x^2 - 8x + 16) / 8, its directrix is y = -1/2 -/
theorem parabola_directrix (p : Parabola) (d : Directrix) :
  p.a = 1/8 ∧ p.b = -1 ∧ p.c = 2 → d.y = -1/2 := by
  sorry

#check parabola_directrix

end NUMINAMATH_CALUDE_parabola_directrix_l3509_350995


namespace NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l3509_350934

def first_seven_primes : List Nat := [2, 3, 5, 7, 11, 13, 17]
def eighth_prime : Nat := 19

theorem sum_first_seven_primes_mod_eighth_prime :
  (first_seven_primes.sum % eighth_prime) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_seven_primes_mod_eighth_prime_l3509_350934


namespace NUMINAMATH_CALUDE_a_greater_than_zero_when_a_greater_than_b_l3509_350967

theorem a_greater_than_zero_when_a_greater_than_b (a b : ℝ) 
  (h1 : a^2 > b^2) (h2 : a > b) : a > 0 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_zero_when_a_greater_than_b_l3509_350967


namespace NUMINAMATH_CALUDE_num_quadrilaterals_equals_choose_12_4_l3509_350962

/-- The number of ways to choose 4 items from 12 items -/
def choose_12_4 : ℕ := 495

/-- The number of distinct points on the circle -/
def num_points : ℕ := 12

/-- The number of vertices in a quadrilateral -/
def vertices_per_quadrilateral : ℕ := 4

/-- Theorem: The number of different convex quadrilaterals formed by selecting 4 vertices 
    from 12 distinct points on the circumference of a circle is equal to choose_12_4 -/
theorem num_quadrilaterals_equals_choose_12_4 : 
  choose_12_4 = Nat.choose num_points vertices_per_quadrilateral := by
  sorry

#eval choose_12_4  -- This should output 495
#eval Nat.choose num_points vertices_per_quadrilateral  -- This should also output 495

end NUMINAMATH_CALUDE_num_quadrilaterals_equals_choose_12_4_l3509_350962


namespace NUMINAMATH_CALUDE_inverse_of_7_mod_45_l3509_350907

theorem inverse_of_7_mod_45 : ∃ x : ℤ, 0 ≤ x ∧ x < 45 ∧ (7 * x) % 45 = 1 :=
  by
  use 32
  sorry

end NUMINAMATH_CALUDE_inverse_of_7_mod_45_l3509_350907


namespace NUMINAMATH_CALUDE_age_difference_l3509_350992

theorem age_difference (x y z : ℕ) (h : z = x - 15) :
  (x + y) - (y + z) = 15 := by sorry

end NUMINAMATH_CALUDE_age_difference_l3509_350992


namespace NUMINAMATH_CALUDE_incorrect_expression_l3509_350963

theorem incorrect_expression (a b : ℝ) (h1 : a < b) (h2 : b < 0) : ¬(b / a > 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l3509_350963


namespace NUMINAMATH_CALUDE_problem_statement_l3509_350969

theorem problem_statement : (3^1 - 2 + 6^2 - 1)^0 * 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3509_350969


namespace NUMINAMATH_CALUDE_pool_surface_area_l3509_350982

/-- A rectangular swimming pool with given dimensions. -/
structure RectangularPool where
  length : ℝ
  width : ℝ

/-- Calculate the surface area of a rectangular pool. -/
def surfaceArea (pool : RectangularPool) : ℝ :=
  pool.length * pool.width

/-- Theorem: The surface area of a rectangular pool with length 20 meters and width 15 meters is 300 square meters. -/
theorem pool_surface_area :
  let pool : RectangularPool := { length := 20, width := 15 }
  surfaceArea pool = 300 := by
  sorry

end NUMINAMATH_CALUDE_pool_surface_area_l3509_350982


namespace NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l3509_350952

theorem students_taking_neither_music_nor_art 
  (total : ℕ) (music : ℕ) (art : ℕ) (both : ℕ) :
  total = 500 →
  music = 30 →
  art = 20 →
  both = 10 →
  total - (music + art - both) = 460 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_neither_music_nor_art_l3509_350952


namespace NUMINAMATH_CALUDE_coefficient_x4_proof_l3509_350923

/-- The expression to be simplified -/
def expression (x : ℝ) : ℝ := 2 * (x^4 - 2*x^3) + 3 * (2*x^2 - 3*x^4 + x^6) - (5*x^6 - 2*x^4)

/-- The coefficient of x^4 in the simplified expression -/
def coefficient_x4 : ℝ := -5

theorem coefficient_x4_proof : ∃ (f : ℝ → ℝ), ∀ x, expression x = f x + coefficient_x4 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_proof_l3509_350923


namespace NUMINAMATH_CALUDE_red_from_white_impossible_l3509_350999

/-- Represents the color of a ball -/
inductive Color
| White
| Red

/-- Represents a bag of balls -/
structure Bag where
  balls : List Color

/-- Defines an impossible event -/
def impossible (event : Prop) : Prop :=
  ¬ event

/-- The bag contains only white balls -/
def only_white_balls (b : Bag) : Prop :=
  ∀ ball ∈ b.balls, ball = Color.White

/-- Drawing a red ball from the bag -/
def draw_red_ball (b : Bag) : Prop :=
  ∃ ball ∈ b.balls, ball = Color.Red

/-- Theorem: Drawing a red ball from a bag containing only white balls is an impossible event -/
theorem red_from_white_impossible (b : Bag) (h : only_white_balls b) :
  impossible (draw_red_ball b) := by
  sorry


end NUMINAMATH_CALUDE_red_from_white_impossible_l3509_350999


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l3509_350951

/-- The length of the pan in inches -/
def pan_length : ℕ := 24

/-- The width of the pan in inches -/
def pan_width : ℕ := 15

/-- The side length of a square brownie piece in inches -/
def piece_side : ℕ := 3

/-- The number of brownie pieces that can be cut from the pan -/
def num_pieces : ℕ := (pan_length * pan_width) / (piece_side * piece_side)

theorem brownie_pieces_count : num_pieces = 40 := by
  sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l3509_350951


namespace NUMINAMATH_CALUDE_min_value_problem_l3509_350980

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 27) :
  a^2 + 9*a*b + 9*b^2 + 3*c^2 ≥ 60 ∧
  (a^2 + 9*a*b + 9*b^2 + 3*c^2 = 60 ↔ a = 6 ∧ b = 2 ∧ c = 3) :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l3509_350980


namespace NUMINAMATH_CALUDE_student_arrangement_l3509_350979

/-- Given a rectangular arrangement of students, prove that the total number of students is 399 -/
theorem student_arrangement (rows columns : ℕ) (yoongi_left yoongi_right yoongi_front yoongi_back : ℕ) : 
  yoongi_left + yoongi_right = rows + 1 →
  yoongi_front + yoongi_back = columns + 1 →
  yoongi_left = 7 →
  yoongi_right = 13 →
  yoongi_front = 8 →
  yoongi_back = 14 →
  rows * columns = 399 := by
  sorry

#check student_arrangement

end NUMINAMATH_CALUDE_student_arrangement_l3509_350979


namespace NUMINAMATH_CALUDE_museum_entrance_cost_l3509_350936

/-- The total cost of entrance tickets for a group of students and teachers -/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : ℕ :=
  (num_students + num_teachers) * ticket_price

/-- Theorem: The total cost for 20 students and 3 teachers with $5 tickets is $115 -/
theorem museum_entrance_cost : total_cost 20 3 5 = 115 := by
  sorry

end NUMINAMATH_CALUDE_museum_entrance_cost_l3509_350936


namespace NUMINAMATH_CALUDE_square_difference_305_295_l3509_350931

theorem square_difference_305_295 : 305^2 - 295^2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_305_295_l3509_350931


namespace NUMINAMATH_CALUDE_complex_product_l3509_350990

/-- Given complex numbers Q, E, and D, prove their product is 116i -/
theorem complex_product (Q E D : ℂ) : 
  Q = 7 + 3*I ∧ E = 2*I ∧ D = 7 - 3*I → Q * E * D = 116*I := by
  sorry

end NUMINAMATH_CALUDE_complex_product_l3509_350990


namespace NUMINAMATH_CALUDE_range_of_f_l3509_350955

noncomputable def f (x : ℝ) : ℝ := 3 * (x - 2)

theorem range_of_f :
  Set.range f = {y : ℝ | y < -21 ∨ y > -21} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3509_350955


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_AD_l3509_350915

/-- Rectangle ABCD with given properties -/
structure Rectangle where
  AB : ℝ
  AD : ℝ
  E : ℝ
  ac_be_perp : Bool

/-- Conditions for the rectangle -/
def rectangle_conditions (rect : Rectangle) : Prop :=
  rect.AB = 80 ∧
  rect.E = (1/3) * rect.AD ∧
  rect.ac_be_perp = true

/-- Theorem statement -/
theorem greatest_integer_less_than_AD (rect : Rectangle) 
  (h : rectangle_conditions rect) : 
  ⌊rect.AD⌋ = 138 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_AD_l3509_350915


namespace NUMINAMATH_CALUDE_car_initial_payment_l3509_350902

/-- Calculates the initial payment for a car purchase given the total cost,
    monthly payment, and number of months. -/
def initial_payment (total_cost monthly_payment num_months : ℕ) : ℕ :=
  total_cost - monthly_payment * num_months

theorem car_initial_payment :
  initial_payment 13380 420 19 = 5400 := by
  sorry

end NUMINAMATH_CALUDE_car_initial_payment_l3509_350902


namespace NUMINAMATH_CALUDE_reciprocal_of_2023_l3509_350973

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Theorem statement
theorem reciprocal_of_2023 : reciprocal 2023 = 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2023_l3509_350973


namespace NUMINAMATH_CALUDE_cube_sum_equals_diff_implies_square_sum_less_than_one_l3509_350937

theorem cube_sum_equals_diff_implies_square_sum_less_than_one 
  (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) : 
  x^2 + y^2 < 1 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_equals_diff_implies_square_sum_less_than_one_l3509_350937


namespace NUMINAMATH_CALUDE_problem_solution_l3509_350918

theorem problem_solution (a b : ℝ) (h1 : a * b = 7) (h2 : a - b = 5) :
  a^2 - 6*a*b + b^2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3509_350918


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_bound_l3509_350953

/-- A convex polygon -/
structure ConvexPolygon where
  -- Define properties of a convex polygon
  area : ℝ
  is_convex : Bool

/-- A line in 2D space -/
structure Line where
  -- Define properties of a line

/-- A triangle inscribed in a polygon -/
structure InscribedTriangle (M : ConvexPolygon) where
  -- Define properties of an inscribed triangle
  area : ℝ
  side_parallel_to : Line

/-- Theorem statement -/
theorem inscribed_triangle_area_bound (M : ConvexPolygon) (l : Line) :
  (∃ T : InscribedTriangle M, T.side_parallel_to = l ∧ T.area ≥ 3/8 * M.area) ∧
  (∃ M' : ConvexPolygon, ∃ l' : Line, 
    ∀ T : InscribedTriangle M', T.side_parallel_to = l' → T.area ≤ 3/8 * M'.area) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_bound_l3509_350953


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l3509_350906

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = π) →
  -- Given condition
  (2 * b * Real.cos C = 2 * a + c) →
  -- Additional condition for part 2
  (2 * Real.sqrt 3 * Real.sin (A / 2 + π / 6) * Real.cos (A / 2 + π / 6) - 
   2 * Real.sin (A / 2 + π / 6) ^ 2 = 11 / 13) →
  -- Conclusions to prove
  (B = 2 * π / 3 ∧ 
   Real.cos C = (12 + 5 * Real.sqrt 3) / 26) := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l3509_350906


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3509_350959

/-- Represents a geometric sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function

/-- Conditions for the geometric sequence -/
def GeometricSequenceConditions (seq : GeometricSequence) : Prop :=
  seq.a 3 = 3/2 ∧ seq.S 3 = 9/2

/-- The value m forms a geometric sequence with a₃ and S₃ -/
def FormsGeometricSequence (seq : GeometricSequence) (m : ℝ) : Prop :=
  ∃ q : ℝ, seq.a 3 * q = m ∧ m * q = seq.S 3

theorem geometric_sequence_problem (seq : GeometricSequence) 
  (h : GeometricSequenceConditions seq) :
  (∀ m : ℝ, FormsGeometricSequence seq m → m = 3*Real.sqrt 3/2 ∨ m = -3*Real.sqrt 3/2) ∧
  (seq.a 1 = 3/2 ∨ seq.a 1 = 6) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3509_350959


namespace NUMINAMATH_CALUDE_smallest_multiple_with_factors_l3509_350977

theorem smallest_multiple_with_factors : 
  ∀ n : ℕ+, 
    (936 * n : ℕ) % 2^5 = 0 ∧ 
    (936 * n : ℕ) % 3^3 = 0 ∧ 
    (936 * n : ℕ) % 11^2 = 0 → 
    n ≥ 4356 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_factors_l3509_350977
