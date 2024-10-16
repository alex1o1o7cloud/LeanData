import Mathlib

namespace NUMINAMATH_CALUDE_milan_bill_cost_l2009_200939

/-- The total cost of a long distance phone bill given the monthly fee, per-minute cost, and minutes used. -/
def total_cost (monthly_fee : ℚ) (per_minute_cost : ℚ) (minutes_used : ℕ) : ℚ :=
  monthly_fee + per_minute_cost * minutes_used

/-- Proof that Milan's long distance bill is $23.36 -/
theorem milan_bill_cost :
  let monthly_fee : ℚ := 2
  let per_minute_cost : ℚ := 12 / 100
  let minutes_used : ℕ := 178
  total_cost monthly_fee per_minute_cost minutes_used = 2336 / 100 := by
  sorry

#eval total_cost 2 (12/100) 178

end NUMINAMATH_CALUDE_milan_bill_cost_l2009_200939


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l2009_200903

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l2009_200903


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2009_200959

theorem inequality_equivalence (x : ℝ) :
  (x - 2) * (2 * x + 3) ≠ 0 →
  ((10 * x^3 - x^2 - 38 * x + 40) / ((x - 2) * (2 * x + 3)) < 2) ↔ (x < 4/5) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2009_200959


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l2009_200950

/-- Given a zoo with penguins and polar bears, prove the ratio of polar bears to penguins -/
theorem zoo_animal_ratio (num_penguins num_total : ℕ) 
  (h1 : num_penguins = 21)
  (h2 : num_total = 63) :
  (num_total - num_penguins) / num_penguins = 2 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l2009_200950


namespace NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l2009_200902

theorem lattice_points_on_hyperbola : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1^2 - p.2^2 = 65) ∧ 
    s.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l2009_200902


namespace NUMINAMATH_CALUDE_triangle_problem_l2009_200932

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_problem (t : Triangle) 
  (h1 : t.b = Real.sqrt 3)
  (h2 : t.a + t.c = 4)
  (h3 : Real.cos t.B / Real.cos t.C = -t.b / (2 * t.a + t.c)) :
  t.B = 2 * Real.pi / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B : ℝ) = 13 * Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l2009_200932


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2009_200999

def A : Set ℝ := {x | -3 ≤ x ∧ x < 4}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2009_200999


namespace NUMINAMATH_CALUDE_larger_number_problem_l2009_200944

theorem larger_number_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 4 * y = 6 * x) (h4 : x + y = 50) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2009_200944


namespace NUMINAMATH_CALUDE_simplify_and_express_negative_exponents_l2009_200949

theorem simplify_and_express_negative_exponents 
  (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = 2 * x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_express_negative_exponents_l2009_200949


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l2009_200930

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The bouncing ball theorem -/
theorem bouncing_ball_distance :
  totalDistance 160 (3/4) 4 = 816.25 := by sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l2009_200930


namespace NUMINAMATH_CALUDE_range_of_a_l2009_200933

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x + 2/x + a ≥ 0) → a ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2009_200933


namespace NUMINAMATH_CALUDE_smallest_product_is_690_l2009_200985

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def smallest_three_digit_product (m : ℕ) : Prop :=
  ∃ a b : ℕ,
    m = a * b * (10*a + b) * (a + b) ∧
    a < 10 ∧ b < 10 ∧
    is_prime a ∧ is_prime b ∧
    is_prime (10*a + b) ∧ is_prime (a + b) ∧
    (a + b) % 5 = 1 ∧
    m ≥ 100 ∧ m < 1000 ∧
    ∀ n : ℕ, n ≥ 100 ∧ n < m → ¬(smallest_three_digit_product n)

theorem smallest_product_is_690 :
  smallest_three_digit_product 690 :=
sorry

end NUMINAMATH_CALUDE_smallest_product_is_690_l2009_200985


namespace NUMINAMATH_CALUDE_probability_one_red_one_white_l2009_200954

/-- The probability of selecting one red ball and one white ball from a bag -/
theorem probability_one_red_one_white (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 3 →
  white_balls = 2 →
  (red_balls.choose 1 * white_balls.choose 1 : ℚ) / total_balls.choose 2 = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_probability_one_red_one_white_l2009_200954


namespace NUMINAMATH_CALUDE_well_depth_and_rope_length_l2009_200973

theorem well_depth_and_rope_length :
  ∃! (x y : ℝ),
    x / 4 - 3 = y ∧
    x / 5 + 1 = y ∧
    x = 80 ∧
    y = 17 := by
  sorry

end NUMINAMATH_CALUDE_well_depth_and_rope_length_l2009_200973


namespace NUMINAMATH_CALUDE_part_one_part_two_l2009_200940

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x + a|
def g (x : ℝ) : ℝ := |x + 3| - x

-- Define the set M
def M (a : ℝ) : Set ℝ := {x | f a x < g x}

-- Statement for part (1)
theorem part_one (a : ℝ) : (a - 3) ∈ M a → a ∈ Set.Ioo 0 3 := by sorry

-- Statement for part (2)
theorem part_two (a : ℝ) : Set.Icc (-1) 1 ⊆ M a → a ∈ Set.Ioo (-2) 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2009_200940


namespace NUMINAMATH_CALUDE_arrangements_of_distinct_letters_l2009_200925

-- Define the number of distinct letters
def num_distinct_letters : ℕ := 7

-- Define the function to calculate the number of arrangements
def num_arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Theorem statement
theorem arrangements_of_distinct_letters : 
  num_arrangements num_distinct_letters = 5040 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_of_distinct_letters_l2009_200925


namespace NUMINAMATH_CALUDE_hexagon_area_from_triangle_l2009_200952

/-- Given a triangle XYZ with circumcircle radius R and perimeter P, 
    the area of the hexagon formed by the intersection points of 
    the perpendicular bisectors with the circumcircle is (P * R) / 4 -/
theorem hexagon_area_from_triangle (R P : ℝ) (hR : R = 10) (hP : P = 45) :
  let hexagon_area := (P * R) / 4
  hexagon_area = 112.5 := by sorry

end NUMINAMATH_CALUDE_hexagon_area_from_triangle_l2009_200952


namespace NUMINAMATH_CALUDE_study_group_formation_l2009_200910

def number_of_ways (n : ℕ) (g2 : ℕ) (g3 : ℕ) : ℕ :=
  (Nat.choose n 3 * Nat.choose (n - 3) 3) / 2 *
  ((Nat.choose (n - 6) 2 * Nat.choose (n - 8) 2 * Nat.choose (n - 10) 2) / 6)

theorem study_group_formation :
  number_of_ways 12 3 2 = 138600 := by
  sorry

end NUMINAMATH_CALUDE_study_group_formation_l2009_200910


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2009_200962

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, x > 2 → x^2 - 2*x > 0) ∧
  (∃ x : ℝ, x^2 - 2*x > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l2009_200962


namespace NUMINAMATH_CALUDE_tomorrow_sunny_is_uncertain_l2009_200927

-- Define the type for events
inductive Event : Type
  | certain : Event
  | impossible : Event
  | inevitable : Event
  | uncertain : Event

-- Define the event "Tomorrow will be sunny"
def tomorrow_sunny : Event := Event.uncertain

-- Define the properties of events
def is_guaranteed (e : Event) : Prop :=
  e = Event.certain ∨ e = Event.inevitable

def cannot_happen (e : Event) : Prop :=
  e = Event.impossible

def is_not_guaranteed (e : Event) : Prop :=
  e = Event.uncertain

-- Theorem statement
theorem tomorrow_sunny_is_uncertain :
  is_not_guaranteed tomorrow_sunny ∧
  ¬is_guaranteed tomorrow_sunny ∧
  ¬cannot_happen tomorrow_sunny :=
by sorry

end NUMINAMATH_CALUDE_tomorrow_sunny_is_uncertain_l2009_200927


namespace NUMINAMATH_CALUDE_specific_triangle_area_l2009_200911

/-- RightTriangle represents a right triangle with specific properties -/
structure RightTriangle where
  AB : ℝ  -- Length of hypotenuse
  median_CA : ℝ → ℝ  -- Equation of median to side CA
  median_CB : ℝ → ℝ  -- Equation of median to side CB

/-- Calculate the area of the right triangle -/
def triangle_area (t : RightTriangle) : ℝ := sorry

/-- Theorem stating the area of the specific right triangle -/
theorem specific_triangle_area :
  let t : RightTriangle := {
    AB := 60,
    median_CA := λ x => x + 3,
    median_CB := λ x => 2 * x + 4
  }
  triangle_area t = 400 := by sorry

end NUMINAMATH_CALUDE_specific_triangle_area_l2009_200911


namespace NUMINAMATH_CALUDE_jar_problem_l2009_200965

theorem jar_problem (total_jars small_jars : ℕ) 
  (small_capacity large_capacity : ℕ) (total_capacity : ℕ) :
  total_jars = 100 →
  small_jars = 62 →
  small_capacity = 3 →
  large_capacity = 5 →
  total_capacity = 376 →
  ∃ large_jars : ℕ, 
    small_jars + large_jars = total_jars ∧
    small_jars * small_capacity + large_jars * large_capacity = total_capacity ∧
    large_jars = 38 :=
by sorry

end NUMINAMATH_CALUDE_jar_problem_l2009_200965


namespace NUMINAMATH_CALUDE_chopped_cube_height_l2009_200915

theorem chopped_cube_height (cube_side_length : ℝ) (h_side : cube_side_length = 2) :
  let chopped_corner_height := cube_side_length - (1 / Real.sqrt 3)
  chopped_corner_height = (5 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_chopped_cube_height_l2009_200915


namespace NUMINAMATH_CALUDE_constant_values_l2009_200920

theorem constant_values (C A : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 5 → (C * x - 10) / ((x - 4) * (x - 5)) = A / (x - 4) + 2 / (x - 5)) →
  C = 12/5 ∧ A = 2/5 := by sorry

end NUMINAMATH_CALUDE_constant_values_l2009_200920


namespace NUMINAMATH_CALUDE_overall_class_average_l2009_200938

-- Define the percentages of each group
def group1_percent : Real := 0.20
def group2_percent : Real := 0.50
def group3_percent : Real := 1 - group1_percent - group2_percent

-- Define the test averages for each group
def group1_average : Real := 80
def group2_average : Real := 60
def group3_average : Real := 40

-- Define the overall class average
def class_average : Real :=
  group1_percent * group1_average +
  group2_percent * group2_average +
  group3_percent * group3_average

-- Theorem statement
theorem overall_class_average :
  class_average = 58 := by
  sorry

end NUMINAMATH_CALUDE_overall_class_average_l2009_200938


namespace NUMINAMATH_CALUDE_max_writers_is_fifty_l2009_200908

/-- Represents the number of people at a newspaper conference --/
structure ConferenceAttendees where
  total : Nat
  editors : Nat
  both : Nat
  neither : Nat
  hTotal : total = 90
  hEditors : editors > 38
  hNeither : neither = 2 * both
  hBothMax : both ≤ 6

/-- The maximum number of writers at the conference --/
def maxWriters (c : ConferenceAttendees) : Nat :=
  c.total - c.editors - c.both

/-- Theorem stating that the maximum number of writers is 50 --/
theorem max_writers_is_fifty (c : ConferenceAttendees) : maxWriters c ≤ 50 ∧ ∃ c', maxWriters c' = 50 := by
  sorry

#eval maxWriters { total := 90, editors := 39, both := 1, neither := 2, hTotal := rfl, hEditors := by norm_num, hNeither := rfl, hBothMax := by norm_num }

end NUMINAMATH_CALUDE_max_writers_is_fifty_l2009_200908


namespace NUMINAMATH_CALUDE_congruence_characterization_l2009_200953

/-- Euler's totient function -/
def phi (n : ℕ) : ℕ := sorry

/-- Characterization of integers satisfying the given congruence -/
theorem congruence_characterization (n : ℕ) (h : n > 2) :
  (phi n / 2) % 6 = 1 ↔ 
    n = 3 ∨ n = 4 ∨ n = 6 ∨ 
    (∃ (p k : ℕ), p.Prime ∧ p % 12 = 11 ∧ (n = p^(2*k) ∨ n = 2 * p^(2*k))) :=
  sorry

end NUMINAMATH_CALUDE_congruence_characterization_l2009_200953


namespace NUMINAMATH_CALUDE_percentage_difference_l2009_200926

theorem percentage_difference : 
  (38 / 100 : ℚ) * 80 - (12 / 100 : ℚ) * 160 = 11.2 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_l2009_200926


namespace NUMINAMATH_CALUDE_eleven_divides_six_digit_repeating_l2009_200948

/-- A 6-digit positive integer where the first three digits are the same as its last three digits -/
def SixDigitRepeating (z : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    0 < a ∧ a ≤ 9 ∧ 
    b ≤ 9 ∧ 
    c ≤ 9 ∧ 
    z = 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c

theorem eleven_divides_six_digit_repeating (z : ℕ) (h : SixDigitRepeating z) : 
  11 ∣ z := by
  sorry

end NUMINAMATH_CALUDE_eleven_divides_six_digit_repeating_l2009_200948


namespace NUMINAMATH_CALUDE_factorization_equality_l2009_200946

theorem factorization_equality (m n : ℝ) : m^2 - n^2 + 2*m - 2*n = (m-n)*(m+n+2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2009_200946


namespace NUMINAMATH_CALUDE_factory_machines_capping_l2009_200951

/-- Represents a machine in the factory -/
structure Machine where
  capping_rate : ℕ  -- bottles capped per minute
  working_time : ℕ  -- working time in minutes

/-- Calculates the total number of bottles capped by a machine -/
def total_capped (m : Machine) : ℕ := m.capping_rate * m.working_time

theorem factory_machines_capping (machine_a machine_b machine_c machine_d machine_e : Machine) :
  machine_a.capping_rate = 24 ∧
  machine_a.working_time = 10 ∧
  machine_b.capping_rate = machine_a.capping_rate - 3 ∧
  machine_b.working_time = 12 ∧
  machine_c.capping_rate = machine_b.capping_rate + 6 ∧
  machine_c.working_time = 15 ∧
  machine_d.capping_rate = machine_c.capping_rate - 4 ∧
  machine_d.working_time = 8 ∧
  machine_e.capping_rate = machine_d.capping_rate + 5 ∧
  machine_e.working_time = 5 →
  total_capped machine_a = 240 ∧
  total_capped machine_b = 252 ∧
  total_capped machine_c = 405 ∧
  total_capped machine_d = 184 ∧
  total_capped machine_e = 140 := by
  sorry

#check factory_machines_capping

end NUMINAMATH_CALUDE_factory_machines_capping_l2009_200951


namespace NUMINAMATH_CALUDE_distinct_arrangements_l2009_200958

/-- The number of distinct arrangements of 6 indistinguishable objects of one type
    and 4 indistinguishable objects of another type in a row of 10 positions -/
def arrangement_count : ℕ := Nat.choose 10 4

/-- Theorem stating that the number of distinct arrangements is 210 -/
theorem distinct_arrangements :
  arrangement_count = 210 := by sorry

end NUMINAMATH_CALUDE_distinct_arrangements_l2009_200958


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2009_200966

theorem coefficient_x_squared_in_expansion : 
  let n : ℕ := 6
  let k : ℕ := 2
  let a : ℤ := 1
  let b : ℤ := -2
  (n.choose k) * b^k * a^(n-k) = 60 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l2009_200966


namespace NUMINAMATH_CALUDE_airplane_passengers_virginia_l2009_200935

/-- Calculates the number of people landing in Virginia given the flight conditions -/
theorem airplane_passengers_virginia
  (initial_passengers : ℕ)
  (texas_off texas_on : ℕ)
  (nc_off nc_on : ℕ)
  (crew : ℕ)
  (h1 : initial_passengers = 124)
  (h2 : texas_off = 58)
  (h3 : texas_on = 24)
  (h4 : nc_off = 47)
  (h5 : nc_on = 14)
  (h6 : crew = 10) :
  initial_passengers - texas_off + texas_on - nc_off + nc_on + crew = 67 :=
by sorry

end NUMINAMATH_CALUDE_airplane_passengers_virginia_l2009_200935


namespace NUMINAMATH_CALUDE_prime_or_composite_a4_3a2_9_l2009_200984

theorem prime_or_composite_a4_3a2_9 (a : ℕ) :
  (a = 1 ∨ a = 2 → Nat.Prime (a^4 - 3*a^2 + 9)) ∧
  (a > 2 → ¬Nat.Prime (a^4 - 3*a^2 + 9)) :=
by sorry

end NUMINAMATH_CALUDE_prime_or_composite_a4_3a2_9_l2009_200984


namespace NUMINAMATH_CALUDE_bisection_method_calculations_l2009_200917

theorem bisection_method_calculations (a b : Real) (accuracy : Real) :
  a = 1.4 →
  b = 1.5 →
  accuracy = 0.001 →
  ∃ n : ℕ, (((b - a) / (2 ^ n : Real)) < accuracy) ∧ 
    (∀ m : ℕ, m < n → ((b - a) / (2 ^ m : Real)) ≥ accuracy) ∧
    n = 7 :=
by sorry

end NUMINAMATH_CALUDE_bisection_method_calculations_l2009_200917


namespace NUMINAMATH_CALUDE_two_digit_number_proof_l2009_200987

theorem two_digit_number_proof : ∃! n : ℕ, 
  (n ≥ 10 ∧ n < 100) ∧ 
  (n / 10 = 2 * (n % 10)) ∧ 
  (∃ m : ℕ, n + (n / 10)^2 = m^2) ∧
  n = 21 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_proof_l2009_200987


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l2009_200977

theorem inverse_proposition_false (a b : ℝ) (h : a < b) :
  ∃ (f : ℝ → ℝ), Continuous f ∧ 
  (∃ x ∈ Set.Ioo a b, f x = 0) ∧ 
  f a * f b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l2009_200977


namespace NUMINAMATH_CALUDE_degree_of_composed_and_multiplied_polynomials_l2009_200912

/-- The degree of a polynomial -/
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

/-- Polynomial composition -/
def polyComp (p q : Polynomial ℝ) : Polynomial ℝ := sorry

/-- Polynomial multiplication -/
def polyMul (p q : Polynomial ℝ) : Polynomial ℝ := sorry

theorem degree_of_composed_and_multiplied_polynomials 
  (f g : Polynomial ℝ) 
  (hf : degree f = 3) 
  (hg : degree g = 7) : 
  degree (polyMul (polyComp f (Polynomial.X^4)) (polyComp g (Polynomial.X^3))) = 33 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_composed_and_multiplied_polynomials_l2009_200912


namespace NUMINAMATH_CALUDE_students_who_left_zoo_l2009_200967

/-- Proves the number of students who left the zoo given the initial conditions and remaining individuals --/
theorem students_who_left_zoo 
  (initial_students : Nat) 
  (initial_chaperones : Nat) 
  (initial_teachers : Nat) 
  (remaining_individuals : Nat) 
  (chaperones_who_left : Nat)
  (h1 : initial_students = 20)
  (h2 : initial_chaperones = 5)
  (h3 : initial_teachers = 2)
  (h4 : remaining_individuals = 15)
  (h5 : chaperones_who_left = 2) :
  initial_students - (remaining_individuals - chaperones_who_left - initial_teachers) = 9 := by
  sorry


end NUMINAMATH_CALUDE_students_who_left_zoo_l2009_200967


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2009_200928

/-- The x-intercept of the line 4x + 6y = 24 is (6, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  4 * x + 6 * y = 24 → y = 0 → x = 6 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2009_200928


namespace NUMINAMATH_CALUDE_pregnant_cows_l2009_200924

theorem pregnant_cows (total_cows : ℕ) (female_ratio : ℚ) (pregnant_ratio : ℚ) :
  total_cows = 44 →
  female_ratio = 1/2 →
  pregnant_ratio = 1/2 →
  ⌊(total_cows : ℚ) * female_ratio * pregnant_ratio⌋ = 11 := by
  sorry

end NUMINAMATH_CALUDE_pregnant_cows_l2009_200924


namespace NUMINAMATH_CALUDE_arnolds_mileage_calculation_l2009_200904

/-- Calculates the total monthly driving mileage for Arnold given his car efficiencies and gas spending --/
def arnolds_mileage (efficiency1 efficiency2 efficiency3 : ℚ) (gas_price : ℚ) (monthly_spend : ℚ) : ℚ :=
  let total_cars := 3
  let inverse_efficiency := (1 / efficiency1 + 1 / efficiency2 + 1 / efficiency3) / total_cars
  monthly_spend / (gas_price * inverse_efficiency)

/-- Theorem stating that Arnold's total monthly driving mileage is (56 * 450) / 43 miles --/
theorem arnolds_mileage_calculation :
  let efficiency1 := 50
  let efficiency2 := 10
  let efficiency3 := 15
  let gas_price := 2
  let monthly_spend := 56
  arnolds_mileage efficiency1 efficiency2 efficiency3 gas_price monthly_spend = 56 * 450 / 43 := by
  sorry

end NUMINAMATH_CALUDE_arnolds_mileage_calculation_l2009_200904


namespace NUMINAMATH_CALUDE_decimal_sum_l2009_200992

theorem decimal_sum : (0.08 : ℚ) + (0.003 : ℚ) + (0.0070 : ℚ) = (0.09 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l2009_200992


namespace NUMINAMATH_CALUDE_odd_function_iff_a_b_zero_l2009_200995

def f (x a b : ℝ) : ℝ := x * abs (x - a) + b

theorem odd_function_iff_a_b_zero (a b : ℝ) :
  (∀ x, f x a b = -f (-x) a b) ↔ a^2 + b^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_iff_a_b_zero_l2009_200995


namespace NUMINAMATH_CALUDE_polynomial_identity_result_l2009_200964

theorem polynomial_identity_result : 
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ : ℝ),
  (∀ x : ℝ, (x^2 - x + 1)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                               a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10 + a₁₁*x^11 + a₁₂*x^12) →
  (a₀ + a₂ + a₄ + a₆ + a₈ + a₁₀ + a₁₂)^2 - (a₁ + a₃ + a₅ + a₇ + a₉ + a₁₁)^2 = 729 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_identity_result_l2009_200964


namespace NUMINAMATH_CALUDE_polynomial_as_sum_of_squares_l2009_200980

theorem polynomial_as_sum_of_squares (x : ℝ) :
  x^4 - 2*x^3 + 6*x^2 - 2*x + 1 = (x^2 - x)^2 + (x - 1)^2 + (2*x)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_as_sum_of_squares_l2009_200980


namespace NUMINAMATH_CALUDE_last_nonzero_digit_factorial_not_periodic_l2009_200936

/-- The last nonzero digit of a natural number -/
def lastNonzeroDigit (n : ℕ) : ℕ := sorry

/-- The sequence of last nonzero digits of factorials -/
def a (n : ℕ) : ℕ := lastNonzeroDigit (n.factorial)

/-- A sequence is eventually periodic if there exists some point after which it repeats with a fixed period -/
def EventuallyPeriodic (f : ℕ → ℕ) : Prop :=
  ∃ (N p : ℕ), p > 0 ∧ ∀ n ≥ N, f (n + p) = f n

theorem last_nonzero_digit_factorial_not_periodic :
  ¬ EventuallyPeriodic a := sorry

end NUMINAMATH_CALUDE_last_nonzero_digit_factorial_not_periodic_l2009_200936


namespace NUMINAMATH_CALUDE_total_stamps_l2009_200941

theorem total_stamps (stamps_AJ : ℕ) (stamps_KJ : ℕ) (stamps_CJ : ℕ) : 
  stamps_AJ = 370 →
  stamps_KJ = stamps_AJ / 2 →
  stamps_CJ = 2 * stamps_KJ + 5 →
  stamps_AJ + stamps_KJ + stamps_CJ = 930 :=
by
  sorry

end NUMINAMATH_CALUDE_total_stamps_l2009_200941


namespace NUMINAMATH_CALUDE_race_difference_l2009_200974

/-- Given a race where A and B run 110 meters, with A finishing in 20 seconds
    and B finishing in 25 seconds, prove that A beats B by 22 meters. -/
theorem race_difference (race_distance : ℝ) (a_time b_time : ℝ) 
  (h_distance : race_distance = 110)
  (h_a_time : a_time = 20)
  (h_b_time : b_time = 25) :
  race_distance - (race_distance / b_time) * a_time = 22 :=
by sorry

end NUMINAMATH_CALUDE_race_difference_l2009_200974


namespace NUMINAMATH_CALUDE_rectangle_equal_diagonals_converse_is_false_contrapositive_is_true_l2009_200998

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Define equal diagonals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry

-- Theorem for the original proposition
theorem rectangle_equal_diagonals (q : Quadrilateral) :
  is_rectangle q → has_equal_diagonals q := sorry

-- Theorem for the converse (which is false)
theorem converse_is_false : ¬(∀ q : Quadrilateral, has_equal_diagonals q → is_rectangle q) := sorry

-- Theorem for the contrapositive (which is true)
theorem contrapositive_is_true :
  ∀ q : Quadrilateral, ¬has_equal_diagonals q → ¬is_rectangle q := sorry

end NUMINAMATH_CALUDE_rectangle_equal_diagonals_converse_is_false_contrapositive_is_true_l2009_200998


namespace NUMINAMATH_CALUDE_vector_properties_l2009_200945

noncomputable section

def a (x : ℝ) : ℝ × ℝ := (Real.cos (3/2 * x), Real.sin (3/2 * x))
def b (x : ℝ) : ℝ × ℝ := (Real.cos (x/2), Real.sin (x/2))

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vector_sum (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

def vector_magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def f (m x : ℝ) : ℝ := m * vector_magnitude (vector_sum (a x) (b x)) - dot_product (a x) (b x)

theorem vector_properties (m : ℝ) :
  (dot_product (a (π/4)) (b (π/4)) = Real.sqrt 2 / 2) ∧
  (vector_magnitude (vector_sum (a (π/4)) (b (π/4))) = Real.sqrt (2 + Real.sqrt 2)) ∧
  (∀ x ∈ Set.Icc 0 π, 
    (m > 2 → f m x ≤ 2*m - 3) ∧
    (0 ≤ m ∧ m ≤ 2 → f m x ≤ m^2/2 - 1) ∧
    (m < 0 → f m x ≤ -1)) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l2009_200945


namespace NUMINAMATH_CALUDE_cow_chicken_leg_excess_l2009_200957

/-- Represents the number of legs more than twice the number of heads in a group of cows and chickens -/
def excess_legs (num_chickens : ℕ) : ℕ :=
  (4 * 10 + 2 * num_chickens) - 2 * (10 + num_chickens)

theorem cow_chicken_leg_excess :
  ∀ num_chickens : ℕ, excess_legs num_chickens = 20 := by
  sorry

end NUMINAMATH_CALUDE_cow_chicken_leg_excess_l2009_200957


namespace NUMINAMATH_CALUDE_base12_addition_l2009_200905

/-- Represents a digit in base 12 --/
inductive Digit12 : Type
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B

/-- Represents a number in base 12 as a list of digits --/
def Base12 := List Digit12

/-- Convert a base 12 number to its decimal representation --/
def toDecimal (n : Base12) : Nat := sorry

/-- Convert a decimal number to its base 12 representation --/
def fromDecimal (n : Nat) : Base12 := sorry

/-- Addition operation for base 12 numbers --/
def addBase12 (a b : Base12) : Base12 := sorry

/-- The main theorem --/
theorem base12_addition :
  let n1 : Base12 := [Digit12.D5, Digit12.A, Digit12.D3]
  let n2 : Base12 := [Digit12.D2, Digit12.B, Digit12.D8]
  addBase12 n1 n2 = [Digit12.D8, Digit12.D9, Digit12.D6] := by sorry

end NUMINAMATH_CALUDE_base12_addition_l2009_200905


namespace NUMINAMATH_CALUDE_floor_division_equality_l2009_200913

theorem floor_division_equality (α : ℝ) (d : ℕ) (h_α : α > 0) :
  ⌊α / d⌋ = ⌊⌊α⌋ / d⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_division_equality_l2009_200913


namespace NUMINAMATH_CALUDE_stream_speed_l2009_200969

/-- Proves that the speed of the stream is 4 km/hr given the boat's speed in still water and its downstream travel details -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  boat_speed = 24 →
  distance = 140 →
  time = 5 →
  (boat_speed + (distance / time - boat_speed)) = 4 := by
sorry


end NUMINAMATH_CALUDE_stream_speed_l2009_200969


namespace NUMINAMATH_CALUDE_cafe_tables_l2009_200921

theorem cafe_tables (outdoor_tables : ℕ) (indoor_chairs : ℕ) (outdoor_chairs : ℕ) (total_chairs : ℕ) :
  outdoor_tables = 11 →
  indoor_chairs = 10 →
  outdoor_chairs = 3 →
  total_chairs = 123 →
  ∃ indoor_tables : ℕ, indoor_tables * indoor_chairs + outdoor_tables * outdoor_chairs = total_chairs ∧ indoor_tables = 9 :=
by sorry

end NUMINAMATH_CALUDE_cafe_tables_l2009_200921


namespace NUMINAMATH_CALUDE_chris_cookies_l2009_200918

theorem chris_cookies (total_cookies : ℕ) (chris_fraction : ℚ) (eaten_fraction : ℚ) : 
  total_cookies = 84 →
  chris_fraction = 1/3 →
  eaten_fraction = 3/4 →
  (↑total_cookies * chris_fraction * eaten_fraction : ℚ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_chris_cookies_l2009_200918


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2009_200907

def setA : Set ℤ := {x | |x| < 3}
def setB : Set ℤ := {x | |x| > 1}

theorem intersection_of_A_and_B :
  setA ∩ setB = {-2, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2009_200907


namespace NUMINAMATH_CALUDE_gear_speed_ratio_l2009_200914

/-- Represents the number of teeth on a gear -/
structure Gear where
  teeth : ℕ

/-- Represents the angular speed of a gear in revolutions per minute -/
structure AngularSpeed where
  rpm : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear

/-- The theorem stating the ratio of angular speeds for four meshed gears -/
theorem gear_speed_ratio (system : GearSystem) 
  (ωA ωB ωC ωD : AngularSpeed) :
  ωA.rpm * system.A.teeth = ωB.rpm * system.B.teeth ∧
  ωB.rpm * system.B.teeth = ωC.rpm * system.C.teeth ∧
  ωC.rpm * system.C.teeth = ωD.rpm * system.D.teeth →
  ∃ (k : ℝ), k > 0 ∧
    ωA.rpm = k * (system.B.teeth * system.C.teeth * system.D.teeth) ∧
    ωB.rpm = k * (system.A.teeth * system.C.teeth * system.D.teeth) ∧
    ωC.rpm = k * (system.A.teeth * system.B.teeth * system.D.teeth) ∧
    ωD.rpm = k * (system.A.teeth * system.B.teeth * system.C.teeth) :=
by sorry

end NUMINAMATH_CALUDE_gear_speed_ratio_l2009_200914


namespace NUMINAMATH_CALUDE_average_decrease_l2009_200986

theorem average_decrease (n : ℕ) (old_avg new_obs : ℚ) : 
  n = 6 →
  old_avg = 12 →
  new_obs = 5 →
  (n * old_avg + new_obs) / (n + 1) = old_avg - 1 := by
  sorry

end NUMINAMATH_CALUDE_average_decrease_l2009_200986


namespace NUMINAMATH_CALUDE_intersection_M_N_l2009_200968

def M : Set ℝ := {0, 1, 3}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2009_200968


namespace NUMINAMATH_CALUDE_mardi_gras_necklaces_l2009_200900

/-- Proves that the total number of necklaces caught is 49 given the problem conditions --/
theorem mardi_gras_necklaces : 
  ∀ (boudreaux rhonda latch cecilia : ℕ),
  boudreaux = 12 →
  rhonda = boudreaux / 2 →
  latch = 3 * rhonda - 4 →
  cecilia = latch + 3 →
  ∃ (k : ℕ), boudreaux + rhonda + latch + cecilia = 7 * k →
  boudreaux + rhonda + latch + cecilia = 49 := by
sorry

end NUMINAMATH_CALUDE_mardi_gras_necklaces_l2009_200900


namespace NUMINAMATH_CALUDE_max_value_sin_cos_l2009_200990

theorem max_value_sin_cos (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  (∀ z w : ℝ, Real.sin z + Real.sin w = 1/3 → 
    Real.sin y - Real.cos x ^ 2 ≤ Real.sin w - Real.cos z ^ 2) →
  Real.sin y - Real.cos x ^ 2 = 4/9 :=
sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_l2009_200990


namespace NUMINAMATH_CALUDE_max_baggies_l2009_200906

def chocolate_chip_cookies : ℕ := 2
def oatmeal_cookies : ℕ := 16
def cookies_per_bag : ℕ := 3

theorem max_baggies : 
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_bag = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_baggies_l2009_200906


namespace NUMINAMATH_CALUDE_company_male_employees_l2009_200997

theorem company_male_employees (m f : ℕ) : 
  m / f = 7 / 8 →
  (m + 3) / f = 8 / 9 →
  m = 189 := by
sorry

end NUMINAMATH_CALUDE_company_male_employees_l2009_200997


namespace NUMINAMATH_CALUDE_megan_homework_pages_l2009_200961

def remaining_pages (total_problems completed_problems problems_per_page : ℕ) : ℕ :=
  ((total_problems - completed_problems) + problems_per_page - 1) / problems_per_page

theorem megan_homework_pages : remaining_pages 40 26 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_megan_homework_pages_l2009_200961


namespace NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l2009_200955

/-- The manufacturing cost of a shoe given transportation cost, selling price, and profit margin. -/
theorem manufacturing_cost_of_shoe (transportation_cost : ℚ) (selling_price : ℚ) (profit_margin : ℚ) :
  transportation_cost = 5 →
  selling_price = 234 →
  profit_margin = 1/5 →
  ∃ (manufacturing_cost : ℚ), 
    selling_price = (manufacturing_cost + transportation_cost) * (1 + profit_margin) ∧
    manufacturing_cost = 190 :=
by sorry

end NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l2009_200955


namespace NUMINAMATH_CALUDE_martin_fruits_l2009_200991

/-- Represents the number of fruits Martin initially had -/
def initial_fruits : ℕ := 150

/-- Represents the number of oranges Martin has after eating -/
def remaining_oranges : ℕ := 50

/-- Represents the fraction of fruits Martin ate -/
def eaten_fraction : ℚ := 1/2

theorem martin_fruits :
  (initial_fruits : ℚ) * (1 - eaten_fraction) = remaining_oranges * 3 :=
sorry

end NUMINAMATH_CALUDE_martin_fruits_l2009_200991


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l2009_200901

theorem cubic_sum_over_product (a b c : ℂ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 30) (h5 : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 33 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l2009_200901


namespace NUMINAMATH_CALUDE_connected_with_short_paths_l2009_200929

/-- A directed graph with n vertices -/
structure DirectedGraph (n : ℕ) where
  edges : Fin n → Fin n → Prop

/-- A path of length at most 2 between two vertices -/
def hasPathOfLengthAtMostTwo (G : DirectedGraph n) (u v : Fin n) : Prop :=
  G.edges u v ∨ ∃ w : Fin n, G.edges u w ∧ G.edges w v

/-- The main theorem statement -/
theorem connected_with_short_paths (n : ℕ) (h : n ≥ 5) :
  ∃ G : DirectedGraph n, ∀ u v : Fin n, hasPathOfLengthAtMostTwo G u v :=
sorry

end NUMINAMATH_CALUDE_connected_with_short_paths_l2009_200929


namespace NUMINAMATH_CALUDE_oh_squared_equals_526_l2009_200942

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter O and orthocenter H
def circumcenter (t : Triangle) : ℝ × ℝ := sorry
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumradius R
def circumradius (t : Triangle) : ℝ := sorry

-- Define side lengths a, b, c
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := sorry

theorem oh_squared_equals_526 (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  let R := circumradius t
  let (a, b, c) := side_lengths t
  R = 8 →
  2 * a^2 + b^2 + c^2 = 50 →
  (O.1 - H.1)^2 + (O.2 - H.2)^2 = 526 := by
  sorry

end NUMINAMATH_CALUDE_oh_squared_equals_526_l2009_200942


namespace NUMINAMATH_CALUDE_total_profit_is_5400_l2009_200979

/-- Represents the profit sharing scenario between Tom and Jose -/
structure ProfitSharing where
  tom_investment : ℕ
  tom_months : ℕ
  jose_investment : ℕ
  jose_months : ℕ
  jose_profit : ℕ

/-- Calculates the total profit earned by Tom and Jose -/
def total_profit (ps : ProfitSharing) : ℕ :=
  sorry

/-- Theorem stating that the total profit is 5400 given the specified conditions -/
theorem total_profit_is_5400 (ps : ProfitSharing) 
  (h1 : ps.tom_investment = 3000)
  (h2 : ps.tom_months = 12)
  (h3 : ps.jose_investment = 4500)
  (h4 : ps.jose_months = 10)
  (h5 : ps.jose_profit = 3000) :
  total_profit ps = 5400 :=
sorry

end NUMINAMATH_CALUDE_total_profit_is_5400_l2009_200979


namespace NUMINAMATH_CALUDE_num_spiders_is_one_l2009_200982

/-- The number of spiders in a pet shop. -/
def num_spiders : ℕ :=
  let num_birds : ℕ := 3
  let num_dogs : ℕ := 5
  let num_snakes : ℕ := 4
  let total_legs : ℕ := 34
  let bird_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let snake_legs : ℕ := 0
  let spider_legs : ℕ := 8
  (total_legs - (num_birds * bird_legs + num_dogs * dog_legs + num_snakes * snake_legs)) / spider_legs

theorem num_spiders_is_one : num_spiders = 1 := by
  sorry

end NUMINAMATH_CALUDE_num_spiders_is_one_l2009_200982


namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l2009_200994

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 3 ∧ 
  (∀ (x y z : ℝ), (x + y + z)^2 ≤ n * (x^2 + y^2 + z^2)) ∧
  (∀ (m : ℕ), m < n → ∃ (x y z : ℝ), (x + y + z)^2 > m * (x^2 + y^2 + z^2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l2009_200994


namespace NUMINAMATH_CALUDE_total_marbles_is_76_l2009_200978

/-- The total number of marbles in an arrangement where 9 rows have 8 marbles each and 1 row has 4 marbles -/
def total_marbles : ℕ := 
  let rows_with_eight := 9
  let marbles_per_row_eight := 8
  let rows_with_four := 1
  let marbles_per_row_four := 4
  rows_with_eight * marbles_per_row_eight + rows_with_four * marbles_per_row_four

/-- Theorem stating that the total number of marbles is 76 -/
theorem total_marbles_is_76 : total_marbles = 76 := by
  sorry

end NUMINAMATH_CALUDE_total_marbles_is_76_l2009_200978


namespace NUMINAMATH_CALUDE_power_multiplication_l2009_200972

theorem power_multiplication (a : ℝ) : a^3 * a = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l2009_200972


namespace NUMINAMATH_CALUDE_tiangong_survey_method_l2009_200960

/-- Represents the types of survey methods --/
inductive SurveyMethod
  | Comprehensive
  | Sampling

/-- Represents the requirements for the survey --/
structure SurveyRequirements where
  high_precision : Bool
  no_errors_allowed : Bool

/-- Determines the appropriate survey method based on the given requirements --/
def appropriate_survey_method (requirements : SurveyRequirements) : SurveyMethod :=
  if requirements.high_precision && requirements.no_errors_allowed then
    SurveyMethod.Comprehensive
  else
    SurveyMethod.Sampling

theorem tiangong_survey_method :
  let requirements : SurveyRequirements := ⟨true, true⟩
  appropriate_survey_method requirements = SurveyMethod.Comprehensive :=
by sorry

end NUMINAMATH_CALUDE_tiangong_survey_method_l2009_200960


namespace NUMINAMATH_CALUDE_angle_is_rational_multiple_of_360_degrees_l2009_200989

/-- A point moving on two intersecting lines -/
structure JumpingPoint where
  angle : ℝ  -- The angle between the lines in radians
  position : ℕ × Bool  -- The position as (jump number, which line)

/-- The condition that the point returns to its starting position -/
def returnsToStart (jp : JumpingPoint) (n : ℕ) : Prop :=
  ∃ k : ℕ, n * jp.angle = k * (2 * Real.pi)

/-- The main theorem -/
theorem angle_is_rational_multiple_of_360_degrees 
  (jp : JumpingPoint) 
  (returns : ∃ n : ℕ, returnsToStart jp n) 
  (h_angle : 0 < jp.angle ∧ jp.angle < 2 * Real.pi) :
  ∃ q : ℚ, jp.angle = q * (2 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_angle_is_rational_multiple_of_360_degrees_l2009_200989


namespace NUMINAMATH_CALUDE_amusement_park_spending_l2009_200963

theorem amusement_park_spending (initial_amount snack_cost : ℕ) : 
  initial_amount = 100 →
  snack_cost = 20 →
  initial_amount - (snack_cost + 3 * snack_cost) = 20 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_spending_l2009_200963


namespace NUMINAMATH_CALUDE_emily_gave_cards_l2009_200937

/-- The number of cards Martha starts with -/
def initial_cards : ℕ := 3

/-- The number of cards Martha ends up with -/
def final_cards : ℕ := 79

/-- The number of cards Emily gave to Martha -/
def cards_from_emily : ℕ := final_cards - initial_cards

theorem emily_gave_cards : cards_from_emily = 76 := by
  sorry

end NUMINAMATH_CALUDE_emily_gave_cards_l2009_200937


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l2009_200943

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 50 ∧ 
  wrong_value = 23 ∧ 
  correct_value = 43 ∧ 
  corrected_mean = 36.5 →
  (n : ℝ) * ((n * corrected_mean - (correct_value - wrong_value)) / n) = 36.1 * n :=
by sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l2009_200943


namespace NUMINAMATH_CALUDE_gcd_1443_999_l2009_200993

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by sorry

end NUMINAMATH_CALUDE_gcd_1443_999_l2009_200993


namespace NUMINAMATH_CALUDE_tangent_circles_exist_l2009_200981

-- Define the circle k
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define a ray
def Ray (origin : ℝ × ℝ) (direction : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, t ≥ 0 ∧ p = (origin.1 + t * direction.1, origin.2 + t * direction.2)}

-- Define tangency between a circle and a ray
def IsTangent (c : Set (ℝ × ℝ)) (r : Set (ℝ × ℝ)) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ c ∧ p ∈ r ∧ ∀ q : ℝ × ℝ, q ∈ c ∩ r → q = p

-- Main theorem
theorem tangent_circles_exist
  (k : Set (ℝ × ℝ))
  (O : ℝ × ℝ)
  (r : ℝ)
  (A : ℝ × ℝ)
  (e f : Set (ℝ × ℝ))
  (hk : k = Circle O r)
  (hA : A ∈ k)
  (he : e = Ray A (1, 0))  -- Arbitrary direction for e
  (hf : f = Ray A (0, 1))  -- Arbitrary direction for f
  (hef : e ≠ f) :
  ∃ c : Set (ℝ × ℝ), ∃ center : ℝ × ℝ, ∃ radius : ℝ,
    c = Circle center radius ∧
    IsTangent c k ∧
    IsTangent c e ∧
    IsTangent c f :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_exist_l2009_200981


namespace NUMINAMATH_CALUDE_roots_equal_opposite_sign_l2009_200934

theorem roots_equal_opposite_sign (a b c d m : ℝ) : 
  (∀ x, (x^2 - 2*b*x + d) / (3*a*x - 4*c) = (m - 2) / (m + 2)) →
  (∃ r, (r^2 - 2*b*r + d) / (3*a*r - 4*c) = (m - 2) / (m + 2) ∧ 
        (-r^2 + 2*b*r - d) / (-3*a*r + 4*c) = (m - 2) / (m + 2)) →
  m = 4*b / (3*a - 2*b) :=
by sorry

end NUMINAMATH_CALUDE_roots_equal_opposite_sign_l2009_200934


namespace NUMINAMATH_CALUDE_percentage_reduction_proof_price_increase_proof_l2009_200909

-- Define the initial price
def initial_price : ℝ := 50

-- Define the final price after two reductions
def final_price : ℝ := 32

-- Define the profit per kilogram before price increase
def initial_profit : ℝ := 10

-- Define the initial daily sales volume
def initial_sales : ℝ := 500

-- Define the target daily profit
def target_profit : ℝ := 6000

-- Define the maximum allowed price increase
def max_price_increase : ℝ := 8

-- Define the sales volume decrease per yuan of price increase
def sales_decrease_rate : ℝ := 20

-- Theorem for the percentage reduction
theorem percentage_reduction_proof :
  ∃ x : ℝ, x > 0 ∧ x < 1 ∧ initial_price * (1 - x)^2 = final_price ∧ x = 1/5 :=
sorry

-- Theorem for the required price increase
theorem price_increase_proof :
  ∃ y : ℝ, 0 < y ∧ y ≤ max_price_increase ∧
  (initial_profit + y) * (initial_sales - sales_decrease_rate * y) = target_profit ∧
  y = 5 :=
sorry

end NUMINAMATH_CALUDE_percentage_reduction_proof_price_increase_proof_l2009_200909


namespace NUMINAMATH_CALUDE_cut_prism_surface_area_l2009_200975

/-- Represents a rectangular prism with a cube cut out from one corner. -/
structure CutPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  cutSize : ℝ

/-- Calculates the surface area of a CutPrism. -/
def surfaceArea (p : CutPrism) : ℝ :=
  2 * (p.length * p.width + p.width * p.height + p.length * p.height)

/-- Theorem: The surface area of a 4 by 2 by 2 rectangular prism with a 1 by 1 by 1 cube
    cut out from one corner is equal to 40 square units. -/
theorem cut_prism_surface_area :
  let p : CutPrism := { length := 4, width := 2, height := 2, cutSize := 1 }
  surfaceArea p = 40 := by
  sorry

end NUMINAMATH_CALUDE_cut_prism_surface_area_l2009_200975


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l2009_200919

theorem mixed_number_calculation : (3 + 3 / 4) * 1.3 + 3 / (2 + 2 / 3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l2009_200919


namespace NUMINAMATH_CALUDE_acid_solution_volume_l2009_200976

theorem acid_solution_volume (V : ℝ) : 
  (V > 0) →                              -- Initial volume is positive
  (0.2 * V - 4 + 20 = 0.4 * V) →         -- Equation representing the acid balance
  (V = 80) :=                            -- Conclusion: initial volume is 80 ml
by
  sorry

end NUMINAMATH_CALUDE_acid_solution_volume_l2009_200976


namespace NUMINAMATH_CALUDE_selling_price_proof_l2009_200956

/-- The selling price that results in the same profit as the loss -/
def selling_price : ℕ := 66

/-- The price at which the article is sold for a loss -/
def loss_price : ℕ := 52

/-- The cost price of the article -/
def cost_price : ℕ := 59

/-- The profit is the same as the loss when selling at the selling price -/
axiom profit_equals_loss : selling_price - cost_price = cost_price - loss_price

theorem selling_price_proof : selling_price = 66 := by
  sorry

end NUMINAMATH_CALUDE_selling_price_proof_l2009_200956


namespace NUMINAMATH_CALUDE_ratio_problem_l2009_200947

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7) :
  d / a = 4 / 35 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2009_200947


namespace NUMINAMATH_CALUDE_slope_60_degrees_l2009_200931

/-- The slope of a line with an angle of inclination of 60° is equal to √3 -/
theorem slope_60_degrees :
  let angle_of_inclination : ℝ := 60 * π / 180
  let slope : ℝ := Real.tan angle_of_inclination
  slope = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_60_degrees_l2009_200931


namespace NUMINAMATH_CALUDE_point_on_transformed_graph_l2009_200922

/-- Given a function g where g(3) = 8, there exists a point (x, y) on the graph of 
    y = 4g(3x-1) + 6 such that x + y = 40 -/
theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 3 = 8) :
  ∃ x y : ℝ, 4 * g (3 * x - 1) + 6 = y ∧ x + y = 40 := by
  sorry

end NUMINAMATH_CALUDE_point_on_transformed_graph_l2009_200922


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l2009_200923

/-- In a pentagon WORDS where ∠W ≅ ∠O ≅ ∠D and ∠R is supplementary to ∠S, the measure of ∠D is 120°. -/
theorem pentagon_angle_measure (W O R D S : ℝ) : 
  -- Pentagon WORDS
  W + O + R + D + S = 540 →
  -- ∠W ≅ ∠O ≅ ∠D
  W = O ∧ O = D →
  -- ∠R is supplementary to ∠S
  R + S = 180 →
  -- The measure of ∠D is 120°
  D = 120 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l2009_200923


namespace NUMINAMATH_CALUDE_harish_paint_time_l2009_200988

/-- The time it takes Harish to paint the wall alone -/
def harish_time : ℝ := 3

/-- The time it takes Ganpat to paint the wall alone -/
def ganpat_time : ℝ := 6

/-- The time it takes Harish and Ganpat to paint the wall together -/
def combined_time : ℝ := 2

theorem harish_paint_time :
  (1 / harish_time + 1 / ganpat_time = 1 / combined_time) →
  harish_time = 3 := by
sorry

end NUMINAMATH_CALUDE_harish_paint_time_l2009_200988


namespace NUMINAMATH_CALUDE_triangle_side_length_l2009_200983

/-- Given two triangles ABC and DEF with specified side lengths and angles,
    prove that the length of EF is 3.75 units when the area of DEF is half that of ABC. -/
theorem triangle_side_length (AB DE AC DF : ℝ) (angleBAC angleEDF : ℝ) :
  AB = 5 →
  DE = 2 →
  AC = 6 →
  DF = 3 →
  angleBAC = 30 * π / 180 →
  angleEDF = 45 * π / 180 →
  (1 / 2 * DE * DF * Real.sin angleEDF) = (1 / 4 * AB * AC * Real.sin angleBAC) →
  ∃ (EF : ℝ), EF = 3.75 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2009_200983


namespace NUMINAMATH_CALUDE_sum_of_A_and_B_l2009_200971

/-- The number of four-digit odd numbers divisible by 3 -/
def A : ℕ := sorry

/-- The number of four-digit multiples of 7 -/
def B : ℕ := sorry

/-- The sum of A and B is 2786 -/
theorem sum_of_A_and_B : A + B = 2786 := by sorry

end NUMINAMATH_CALUDE_sum_of_A_and_B_l2009_200971


namespace NUMINAMATH_CALUDE_largest_prime_divisor_l2009_200996

/-- Converts a base-5 number (represented as a list of digits) to decimal --/
def base5ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

/-- The base-5 representation of the number in question --/
def base5Number : List Nat := [1, 2, 0, 1, 0, 2, 0, 1]

/-- The decimal representation of the number --/
def decimalNumber : Nat := base5ToDecimal base5Number

/-- Proposition: The largest prime divisor of the given number is 139 --/
theorem largest_prime_divisor :
  ∃ (d : Nat), d.Prime ∧ d ∣ decimalNumber ∧ d = 139 ∧ ∀ (p : Nat), p.Prime → p ∣ decimalNumber → p ≤ d :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_l2009_200996


namespace NUMINAMATH_CALUDE_square_difference_l2009_200916

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 3/8) 
  (h2 : x - y = 1/8) : 
  x^2 - y^2 = 3/64 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2009_200916


namespace NUMINAMATH_CALUDE_john_uber_earnings_l2009_200970

/-- Calculates the total money made from Uber before considering depreciation -/
def total_money_before_depreciation (initial_car_value trade_in_value profit_after_depreciation : ℕ) : ℕ :=
  profit_after_depreciation + (initial_car_value - trade_in_value)

/-- Theorem stating that John's total money made from Uber before depreciation is $30,000 -/
theorem john_uber_earnings :
  let initial_car_value : ℕ := 18000
  let trade_in_value : ℕ := 6000
  let profit_after_depreciation : ℕ := 18000
  total_money_before_depreciation initial_car_value trade_in_value profit_after_depreciation = 30000 := by
  sorry

end NUMINAMATH_CALUDE_john_uber_earnings_l2009_200970
