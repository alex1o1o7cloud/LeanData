import Mathlib

namespace NUMINAMATH_CALUDE_second_class_sample_size_l1029_102917

/-- Calculates the number of items to be sampled from a specific class in stratified sampling -/
def stratifiedSampleSize (totalPopulation : ℕ) (classPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  (classPopulation * sampleSize) / totalPopulation

theorem second_class_sample_size :
  let totalPopulation : ℕ := 200
  let secondClassPopulation : ℕ := 60
  let sampleSize : ℕ := 40
  stratifiedSampleSize totalPopulation secondClassPopulation sampleSize = 12 := by
sorry

end NUMINAMATH_CALUDE_second_class_sample_size_l1029_102917


namespace NUMINAMATH_CALUDE_distance_between_points_on_parabola_l1029_102963

/-- The distance between two points on a parabola y = mx^2 + k -/
theorem distance_between_points_on_parabola
  (m k a b c d : ℝ) 
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  Real.sqrt ((c - a)^2 + (d - b)^2) = |c - a| * Real.sqrt (1 + m^2 * (c + a)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_between_points_on_parabola_l1029_102963


namespace NUMINAMATH_CALUDE_difference_of_solutions_l1029_102997

def f (n : ℕ) : ℕ := (Finset.filter (fun (x, y, z) => 4*x + 3*y + 2*z = n) (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card

theorem difference_of_solutions : f 2009 - f 2000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_solutions_l1029_102997


namespace NUMINAMATH_CALUDE_grade_assignment_count_l1029_102908

def num_students : ℕ := 12
def num_grades : ℕ := 4

theorem grade_assignment_count :
  num_grades ^ num_students = 16777216 := by
  sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l1029_102908


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1029_102990

theorem inequality_system_solution (x : ℝ) :
  (1 / x < 3) ∧ (1 / x > -4) ∧ (x^2 - 3*x + 2 < 0) → (1 < x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1029_102990


namespace NUMINAMATH_CALUDE_holly_insulin_pills_l1029_102901

/-- Represents the number of pills Holly takes per day for each type of medication -/
structure DailyPills where
  insulin : ℕ
  blood_pressure : ℕ
  anticonvulsant : ℕ

/-- Calculates the total number of pills Holly takes in a week -/
def weekly_total (d : DailyPills) : ℕ :=
  7 * (d.insulin + d.blood_pressure + d.anticonvulsant)

/-- Holly's daily pill regimen satisfies the given conditions -/
def holly_pills : DailyPills :=
  { insulin := 2,
    blood_pressure := 3,
    anticonvulsant := 6 }

theorem holly_insulin_pills :
  holly_pills.insulin = 2 ∧
  holly_pills.blood_pressure = 3 ∧
  holly_pills.anticonvulsant = 2 * holly_pills.blood_pressure ∧
  weekly_total holly_pills = 77 := by
  sorry

end NUMINAMATH_CALUDE_holly_insulin_pills_l1029_102901


namespace NUMINAMATH_CALUDE_book_page_digits_l1029_102946

/-- The total number of digits used to number pages in a book -/
def totalDigits (n : ℕ) : ℕ :=
  (min n 9) +
  2 * (min n 99 - min n 9) +
  3 * (n - min n 99)

/-- Theorem: The total number of digits used in numbering the pages of a book with 366 pages is 990 -/
theorem book_page_digits : totalDigits 366 = 990 := by
  sorry

end NUMINAMATH_CALUDE_book_page_digits_l1029_102946


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_a_minus_b_l1029_102909

theorem quadratic_solution_implies_a_minus_b (a b : ℝ) : 
  (4^2 + 4*a - 4*b = 0) → (a - b = -4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_a_minus_b_l1029_102909


namespace NUMINAMATH_CALUDE_correct_num_license_plates_l1029_102914

/-- The number of distinct license plates with 5 digits and two consecutive letters -/
def num_license_plates : ℕ :=
  let num_digits : ℕ := 10  -- 0 to 9
  let num_uppercase : ℕ := 26
  let num_lowercase : ℕ := 26
  let num_digit_positions : ℕ := 5
  let num_letter_pair_positions : ℕ := 6  -- The letter pair can start in any of the first 6 positions
  let num_letter_pair_arrangements : ℕ := 2  -- uppercase-lowercase or lowercase-uppercase

  num_uppercase * num_lowercase *
  num_letter_pair_arrangements *
  num_letter_pair_positions *
  num_digits ^ num_digit_positions

theorem correct_num_license_plates : num_license_plates = 809280000 := by
  sorry

end NUMINAMATH_CALUDE_correct_num_license_plates_l1029_102914


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l1029_102949

/-- The number of ways to distribute n indistinguishable items among k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors that can be created by combining 5 scoops from 4 basic flavors -/
def ice_cream_flavors : ℕ := distribute 5 4

theorem ice_cream_flavors_count : ice_cream_flavors = 56 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l1029_102949


namespace NUMINAMATH_CALUDE_crystal_mass_ratio_l1029_102925

theorem crystal_mass_ratio : 
  ∀ (x y a : ℝ),
  (a = 0.04 * x) →
  ((3/7) * a = 0.05 * y) →
  (x / y = 35 / 12) :=
by sorry

end NUMINAMATH_CALUDE_crystal_mass_ratio_l1029_102925


namespace NUMINAMATH_CALUDE_binomial_15_13_l1029_102911

theorem binomial_15_13 : Nat.choose 15 13 = 105 := by
  sorry

end NUMINAMATH_CALUDE_binomial_15_13_l1029_102911


namespace NUMINAMATH_CALUDE_k_upper_bound_l1029_102924

theorem k_upper_bound (k : ℝ) : 
  (∀ x : ℝ, 3^(2*x) - (k+1)*3^x + 2 > 0) → k < 2*Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_k_upper_bound_l1029_102924


namespace NUMINAMATH_CALUDE_odometer_problem_l1029_102962

theorem odometer_problem (a b c : ℕ) : 
  a < 10 → b < 10 → c < 10 →  -- Digits are less than 10
  a ≠ b → b ≠ c → a ≠ c →     -- Digits are distinct
  a + b + c ≤ 9 →             -- Sum of digits is at most 9
  (100 * c + 10 * b + a) - (100 * a + 10 * b + c) % 60 = 0 → -- Difference divisible by 60
  a^2 + b^2 + c^2 = 35 :=
by sorry

end NUMINAMATH_CALUDE_odometer_problem_l1029_102962


namespace NUMINAMATH_CALUDE_min_value_problem_l1029_102916

theorem min_value_problem (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h_eq : x + 2*y = 1) :
  ∃ (min : ℝ), min = 2 ∧ ∀ (x' y' : ℝ), x' ≥ 0 → y' ≥ 0 → x' + 2*y' = 1 → 2*x' + 3*(y'^2) ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_problem_l1029_102916


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_l1029_102980

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a^2| + |x - a - 1|

-- Theorem 1: f(x) ≥ 3/4 for all x and a
theorem f_lower_bound (x a : ℝ) : f x a ≥ 3/4 := by
  sorry

-- Theorem 2: If f(4) < 13, then -2 < a < 3
theorem a_range (a : ℝ) : f 4 a < 13 → -2 < a ∧ a < 3 := by
  sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_l1029_102980


namespace NUMINAMATH_CALUDE_intersection_distance_squared_example_l1029_102958

/-- Circle represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Calculate the square of the distance between intersection points of two circles -/
def intersection_distance_squared (c1 c2 : Circle) : ℝ :=
  let x1 := c1.center.1
  let y1 := c1.center.2
  let x2 := c2.center.1
  let y2 := c2.center.2
  let r1 := c1.radius
  let r2 := c2.radius
  -- Calculate the square of the distance between intersection points
  sorry

theorem intersection_distance_squared_example : 
  let c1 : Circle := ⟨(3, -2), 5⟩
  let c2 : Circle := ⟨(3, 4), Real.sqrt 13⟩
  intersection_distance_squared c1 c2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_example_l1029_102958


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1029_102933

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 1 → x > 0) ∧ 
  (∃ x : ℝ, x > 0 ∧ ¬(x > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1029_102933


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1029_102964

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 1/y) * (x + 1/y - 2024) + (y + 1/x) * (y + 1/x - 2024) + 2024 ≥ -2050208 :=
by sorry

theorem min_value_achievable :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧
    (x + 1/y) * (x + 1/y - 2024) + (y + 1/x) * (y + 1/x - 2024) + 2024 = -2050208 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l1029_102964


namespace NUMINAMATH_CALUDE_factorization_sum_l1029_102954

theorem factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 20 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 14 := by
sorry

end NUMINAMATH_CALUDE_factorization_sum_l1029_102954


namespace NUMINAMATH_CALUDE_distance_to_focus_l1029_102993

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = 2 * x^2

-- Define the focus of the parabola
def focus (F : ℝ × ℝ) : Prop := F.2 = 1/4

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2 ∧ P.1 = 1

-- Theorem statement
theorem distance_to_focus (F P : ℝ × ℝ) :
  focus F → point_on_parabola P → |P.1 - F.1| + |P.2 - F.2| = 17/8 := by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l1029_102993


namespace NUMINAMATH_CALUDE_M_equals_four_l1029_102998

theorem M_equals_four : 
  let M := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / 
           Real.sqrt (Real.sqrt 8 + 2) - 
           Real.sqrt (5 - 2 * Real.sqrt 6)
  M = 4 := by sorry

end NUMINAMATH_CALUDE_M_equals_four_l1029_102998


namespace NUMINAMATH_CALUDE_planes_parallel_if_lines_perpendicular_and_parallel_l1029_102935

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_lines_perpendicular_and_parallel
  (m n : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : perpendicular n β)
  (h3 : parallel_lines m n) :
  parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_lines_perpendicular_and_parallel_l1029_102935


namespace NUMINAMATH_CALUDE_negation_equivalence_l1029_102974

-- Define the original proposition
def original_prop (a b : ℝ) : Prop :=
  a^2 + b^2 = 0 → a = 0 ∧ b = 0

-- Define the negation we want to prove
def negation (a b : ℝ) : Prop :=
  a^2 + b^2 = 0 → ¬(a = 0 ∧ b = 0)

-- Theorem stating that the negation of the original proposition
-- is equivalent to our defined negation
theorem negation_equivalence :
  (¬ ∀ a b : ℝ, original_prop a b) ↔ (∀ a b : ℝ, negation a b) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1029_102974


namespace NUMINAMATH_CALUDE_bulb_arrangement_theorem_l1029_102971

def blue_bulbs : ℕ := 7
def red_bulbs : ℕ := 7
def white_bulbs : ℕ := 12

def total_non_white_bulbs : ℕ := blue_bulbs + red_bulbs
def total_slots : ℕ := total_non_white_bulbs + 1

def arrangement_count : ℕ := Nat.choose total_non_white_bulbs blue_bulbs * Nat.choose total_slots white_bulbs

theorem bulb_arrangement_theorem :
  arrangement_count = 1561560 :=
by sorry

end NUMINAMATH_CALUDE_bulb_arrangement_theorem_l1029_102971


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1029_102937

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ := seq.a 2 - seq.a 1

theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h : 2 * seq.S 3 = 3 * seq.S 2 + 6) : 
  common_difference seq = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1029_102937


namespace NUMINAMATH_CALUDE_range_of_a_l1029_102902

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≤ 1 → 1 + 2^x + 4^x * a > 0) ↔ a > -1/4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1029_102902


namespace NUMINAMATH_CALUDE_circle_symmetry_l1029_102951

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 5

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 5

-- Define symmetry with respect to the origin
def symmetric_wrt_origin (f g : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ g (-x) (-y)

-- Theorem statement
theorem circle_symmetry :
  symmetric_wrt_origin original_circle symmetric_circle :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1029_102951


namespace NUMINAMATH_CALUDE_remainder_equality_l1029_102996

theorem remainder_equality (P P' Q D : ℕ) (R R' S : ℕ) 
  (h1 : P > P') (h2 : P' > Q) 
  (h3 : R = P % D) (h4 : R' = P' % D) (h5 : S = Q % D) : 
  (P * P' * Q) % D = (R * R' * S) % D :=
sorry

end NUMINAMATH_CALUDE_remainder_equality_l1029_102996


namespace NUMINAMATH_CALUDE_quadratic_equation_property_l1029_102913

theorem quadratic_equation_property (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   3 * x^2 + 5 * x + k = 0 ∧ 
   3 * y^2 + 5 * y + k = 0 ∧
   |x - y| = x^2 + y^2) ↔ 
  (k = (70 + 10 * Real.sqrt 33) / 8 ∨ k = (70 - 10 * Real.sqrt 33) / 8) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_property_l1029_102913


namespace NUMINAMATH_CALUDE_invalid_vote_percentage_l1029_102995

theorem invalid_vote_percentage
  (total_votes : ℕ)
  (candidate_a_percentage : ℝ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 0.85)
  (h3 : candidate_a_votes = 404600) :
  (1 - (candidate_a_votes : ℝ) / (candidate_a_percentage * total_votes)) * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_invalid_vote_percentage_l1029_102995


namespace NUMINAMATH_CALUDE_ball_bounce_height_l1029_102932

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (h_target : ℝ) (k : ℕ) 
  (h_initial : h₀ = 1500)
  (h_ratio : r = 2/3)
  (h_target_def : h_target = 2) :
  (∀ n : ℕ, n < k → h₀ * r^n ≥ h_target) ∧ 
  (h₀ * r^k < h_target) ↔ 
  k = 19 := by
sorry

end NUMINAMATH_CALUDE_ball_bounce_height_l1029_102932


namespace NUMINAMATH_CALUDE_keith_card_spending_l1029_102945

/-- The amount Keith spent on cards -/
def total_spent (digimon_packs : ℕ) (digimon_price : ℚ) (baseball_price : ℚ) : ℚ :=
  digimon_packs * digimon_price + baseball_price

/-- Proof that Keith spent $23.86 on cards -/
theorem keith_card_spending :
  total_spent 4 (445/100) (606/100) = 2386/100 := by
  sorry

end NUMINAMATH_CALUDE_keith_card_spending_l1029_102945


namespace NUMINAMATH_CALUDE_max_sum_squares_roots_l1029_102953

/-- 
For a quadratic equation x^2 + 2ax + 2a^2 + 4a + 3 = 0 with parameter a,
the sum of squares of its roots is maximized when a = -3, and the maximum sum is 18.
-/
theorem max_sum_squares_roots (a : ℝ) : 
  let f := fun x : ℝ => x^2 + 2*a*x + 2*a^2 + 4*a + 3
  let sum_squares := (- (2*a))^2 - 2*(2*a^2 + 4*a + 3)
  (∀ b : ℝ, sum_squares ≤ (-8 * (-3) - 6)) ∧ 
  sum_squares = 18 ↔ a = -3 := by sorry

end NUMINAMATH_CALUDE_max_sum_squares_roots_l1029_102953


namespace NUMINAMATH_CALUDE_quarters_percentage_is_fifty_percent_l1029_102903

/-- The number of dimes -/
def num_dimes : ℕ := 50

/-- The number of quarters -/
def num_quarters : ℕ := 20

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of all coins in cents -/
def total_value : ℕ := num_dimes * dime_value + num_quarters * quarter_value

/-- The value of quarters in cents -/
def quarters_value : ℕ := num_quarters * quarter_value

/-- Theorem stating that the percentage of the total value in quarters is 50% -/
theorem quarters_percentage_is_fifty_percent :
  (quarters_value : ℚ) / (total_value : ℚ) * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_quarters_percentage_is_fifty_percent_l1029_102903


namespace NUMINAMATH_CALUDE_min_value_expression_l1029_102928

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  3 * a^2 + 1 / (a * (a - b)) + 1 / (a * b) - 6 * a * c + 9 * c^2 ≥ 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1029_102928


namespace NUMINAMATH_CALUDE_five_twos_to_one_to_five_l1029_102936

theorem five_twos_to_one_to_five :
  ∃ (a b c d e : ℕ → ℕ → ℕ → ℕ → ℕ → ℚ),
    (∀ x y z w v, x = 2 ∧ y = 2 ∧ z = 2 ∧ w = 2 ∧ v = 2 →
      a x y z w v = 1 ∧
      b x y z w v = 2 ∧
      c x y z w v = 3 ∧
      d x y z w v = 4 ∧
      e x y z w v = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_five_twos_to_one_to_five_l1029_102936


namespace NUMINAMATH_CALUDE_sequence_max_value_l1029_102918

/-- The sequence a_n defined by -2n^2 + 29n + 3 for positive integers n has a maximum value of 108 -/
theorem sequence_max_value :
  ∃ (M : ℕ), ∀ (n : ℕ), n > 0 → (-2 * n^2 + 29 * n + 3 : ℤ) ≤ M ∧
  ∃ (k : ℕ), k > 0 ∧ (-2 * k^2 + 29 * k + 3 : ℤ) = M ∧ M = 108 :=
by
  sorry


end NUMINAMATH_CALUDE_sequence_max_value_l1029_102918


namespace NUMINAMATH_CALUDE_root_difference_equation_l1029_102948

theorem root_difference_equation (r s : ℝ) : 
  ((r - 5) * (r + 5) = 25 * r - 125) →
  ((s - 5) * (s + 5) = 25 * s - 125) →
  r ≠ s →
  r > s →
  r - s = 15 := by
sorry

end NUMINAMATH_CALUDE_root_difference_equation_l1029_102948


namespace NUMINAMATH_CALUDE_dots_erased_l1029_102955

/-- Checks if a number contains the digit 2 in its base-3 representation -/
def containsTwo (n : Nat) : Bool :=
  let rec aux (m : Nat) : Bool :=
    if m = 0 then false
    else if m % 3 = 2 then true
    else aux (m / 3)
  aux n

/-- Counts the number of integers from 0 to n (inclusive) whose base-3 representation contains at least one digit '2' -/
def countNumbersWithTwo (n : Nat) : Nat :=
  (List.range (n + 1)).filter containsTwo |>.length

theorem dots_erased (total_dots : Nat) : 
  total_dots = 1000 → countNumbersWithTwo (total_dots - 1) = 895 := by sorry

end NUMINAMATH_CALUDE_dots_erased_l1029_102955


namespace NUMINAMATH_CALUDE_vectors_collinear_l1029_102987

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- State the theorem
theorem vectors_collinear (h : ¬ ∃ (r : ℝ), e₁ = r • e₂) :
  ∃ (k : ℝ), (3 : ℝ) • e₁ - (2 : ℝ) • e₂ = k • ((4 : ℝ) • e₂ - (6 : ℝ) • e₁) :=
sorry

end NUMINAMATH_CALUDE_vectors_collinear_l1029_102987


namespace NUMINAMATH_CALUDE_flour_qualification_l1029_102973

def is_qualified (weight : ℝ) : Prop :=
  24.75 ≤ weight ∧ weight ≤ 25.25

theorem flour_qualification (weight : ℝ) :
  weight = 24.80 → is_qualified weight :=
by
  sorry

#check flour_qualification

end NUMINAMATH_CALUDE_flour_qualification_l1029_102973


namespace NUMINAMATH_CALUDE_james_remaining_milk_l1029_102926

/-- Calculates the remaining amount of milk in ounces after drinking some -/
def remaining_milk (initial_gallons : ℕ) (ounces_per_gallon : ℕ) (ounces_drunk : ℕ) : ℕ :=
  initial_gallons * ounces_per_gallon - ounces_drunk

/-- Proves that given the initial conditions, James has 371 ounces of milk left -/
theorem james_remaining_milk :
  remaining_milk 3 128 13 = 371 := by
  sorry

end NUMINAMATH_CALUDE_james_remaining_milk_l1029_102926


namespace NUMINAMATH_CALUDE_fourth_composition_is_even_l1029_102923

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statement
theorem fourth_composition_is_even
  (f : ℝ → ℝ) (h : OddFunction f) :
  ∀ x, f (f (f (f x))) = f (f (f (f (-x)))) :=
sorry

end NUMINAMATH_CALUDE_fourth_composition_is_even_l1029_102923


namespace NUMINAMATH_CALUDE_parabola_vertex_l1029_102907

/-- The vertex of the parabola y = 24x^2 - 48 has coordinates (0, -48) -/
theorem parabola_vertex (x y : ℝ) : 
  y = 24 * x^2 - 48 → (0, -48) = (x, y) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l1029_102907


namespace NUMINAMATH_CALUDE_intersection_complement_equals_specific_set_l1029_102904

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 3, 4, 5}
def B : Set ℕ := {2, 3, 6, 7}

theorem intersection_complement_equals_specific_set :
  B ∩ (U \ A) = {6, 7} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_specific_set_l1029_102904


namespace NUMINAMATH_CALUDE_jordans_unripe_mangoes_l1029_102944

/-- Proves that Jordan kept 16 unripe mangoes given the conditions of the problem -/
theorem jordans_unripe_mangoes (total_mangoes : ℕ) (ripe_fraction : ℚ) (unripe_fraction : ℚ)
  (mangoes_per_jar : ℕ) (jars_made : ℕ) :
  total_mangoes = 54 →
  ripe_fraction = 1/3 →
  unripe_fraction = 2/3 →
  mangoes_per_jar = 4 →
  jars_made = 5 →
  (unripe_fraction * total_mangoes : ℚ).num - mangoes_per_jar * jars_made = 16 :=
by sorry

end NUMINAMATH_CALUDE_jordans_unripe_mangoes_l1029_102944


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1029_102960

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 2 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1029_102960


namespace NUMINAMATH_CALUDE_ben_total_items_is_27_1_l1029_102910

/-- The number of new clothing items for Ben -/
def ben_total_items (alex_shirts alex_pants alex_shoes alex_hats alex_jackets : ℝ)
  (joe_shirts_diff joe_pants_diff joe_hats_diff joe_jackets_diff : ℝ)
  (ben_shirts_diff ben_pants_diff ben_shoes_diff ben_hats_diff ben_jackets_diff : ℝ) : ℝ :=
  let joe_shirts := alex_shirts + joe_shirts_diff
  let joe_pants := alex_pants + joe_pants_diff
  let joe_shoes := alex_shoes
  let joe_hats := alex_hats + joe_hats_diff
  let joe_jackets := alex_jackets + joe_jackets_diff
  let ben_shirts := joe_shirts + ben_shirts_diff
  let ben_pants := alex_pants + ben_pants_diff
  let ben_shoes := joe_shoes + ben_shoes_diff
  let ben_hats := alex_hats + ben_hats_diff
  let ben_jackets := joe_jackets + ben_jackets_diff
  ben_shirts + ben_pants + ben_shoes + ben_hats + ben_jackets

/-- Theorem stating that Ben has 27.1 total new clothing items -/
theorem ben_total_items_is_27_1 :
  ben_total_items 4.5 3 2.5 1.5 2 3.5 (-2.5) 0.3 (-1) 5.3 5.5 (-1.7) 0.5 1.5 = 27.1 := by
  sorry

end NUMINAMATH_CALUDE_ben_total_items_is_27_1_l1029_102910


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1029_102965

theorem necessary_but_not_sufficient (p q : Prop) :
  (p ∧ q → p) ∧ ¬(p → p ∧ q) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1029_102965


namespace NUMINAMATH_CALUDE_total_volume_of_cubes_l1029_102988

/-- The volume of a cube with side length s -/
def cube_volume (s : ℕ) : ℕ := s^3

/-- The total volume of n cubes with side length s -/
def total_volume (n : ℕ) (s : ℕ) : ℕ := n * (cube_volume s)

/-- Carl's cubes -/
def carl_cubes : ℕ × ℕ := (3, 3)

/-- Kate's cubes -/
def kate_cubes : ℕ × ℕ := (4, 4)

theorem total_volume_of_cubes :
  total_volume carl_cubes.1 carl_cubes.2 + total_volume kate_cubes.1 kate_cubes.2 = 337 := by
  sorry

end NUMINAMATH_CALUDE_total_volume_of_cubes_l1029_102988


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1029_102976

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal shift to a parabola -/
def horizontal_shift (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := 2 * p.a * h + p.b,
    c := p.a * h^2 + p.b * h + p.c }

/-- Applies a vertical shift to a parabola -/
def vertical_shift (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a,
    b := p.b,
    c := p.c + v }

theorem parabola_shift_theorem (p : Parabola) :
  let p1 := horizontal_shift p (-1)
  let p2 := vertical_shift p1 (-3)
  p.a = -2 ∧ p.b = 0 ∧ p.c = 0 →
  p2.a = -2 ∧ p2.b = -4 ∧ p2.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1029_102976


namespace NUMINAMATH_CALUDE_domain_of_g_l1029_102992

-- Define the function f with domain (0,1)
def f : {x : ℝ | 0 < x ∧ x < 1} → ℝ := sorry

-- Define the function g(x) = f(2x-1)
def g (x : ℝ) : ℝ := f ⟨2*x - 1, sorry⟩

-- Theorem stating that the domain of g is (1/2, 1)
theorem domain_of_g : 
  ∀ x : ℝ, (∃ y, g x = y) ↔ (1/2 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l1029_102992


namespace NUMINAMATH_CALUDE_tom_net_calories_l1029_102977

/-- Calculates the net calories consumed from candy bars in a week -/
def netCaloriesFromCandyBars (caloriesPerBar : ℕ) (barsPerWeek : ℕ) (caloriesBurned : ℕ) : ℕ :=
  caloriesPerBar * barsPerWeek - caloriesBurned

/-- Proves that Tom consumes 1082 net calories from candy bars in a week -/
theorem tom_net_calories : 
  netCaloriesFromCandyBars 347 6 1000 = 1082 := by
  sorry

#eval netCaloriesFromCandyBars 347 6 1000

end NUMINAMATH_CALUDE_tom_net_calories_l1029_102977


namespace NUMINAMATH_CALUDE_consecutive_composites_l1029_102939

theorem consecutive_composites (a n : ℕ) (ha : a ≥ 2) (hn : n > 0) :
  ∃ k : ℕ, k > 0 ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ d : ℕ, 1 < d ∧ d < a^k + i ∧ (a^k + i) % d = 0 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_composites_l1029_102939


namespace NUMINAMATH_CALUDE_safflower_percentage_in_brand_b_l1029_102994

/-- Represents the composition of a birdseed brand -/
structure BirdseedBrand where
  millet : ℝ
  sunflower : ℝ
  safflower : ℝ

/-- Represents the mix of two birdseed brands -/
structure BirdseedMix where
  brandA : BirdseedBrand
  brandB : BirdseedBrand
  proportionA : ℝ

/-- The theorem stating the percentage of safflower in Brand B -/
theorem safflower_percentage_in_brand_b 
  (brandA : BirdseedBrand)
  (brandB : BirdseedBrand)
  (mix : BirdseedMix)
  (h1 : brandA.millet = 0.4)
  (h2 : brandA.sunflower = 0.6)
  (h3 : brandB.millet = 0.65)
  (h4 : mix.proportionA = 0.6)
  (h5 : mix.proportionA * brandA.millet + (1 - mix.proportionA) * brandB.millet = 0.5)
  : brandB.safflower = 0.35 := by
  sorry


end NUMINAMATH_CALUDE_safflower_percentage_in_brand_b_l1029_102994


namespace NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_91_l1029_102969

theorem min_sum_of_squares_with_diff_91 :
  ∃ (x y : ℕ), x > y ∧ x^2 - y^2 = 91 ∧
  ∀ (a b : ℕ), a > b → a^2 - b^2 = 91 → x^2 + y^2 ≤ a^2 + b^2 ∧
  x^2 + y^2 = 109 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_with_diff_91_l1029_102969


namespace NUMINAMATH_CALUDE_arithmetic_mean_reciprocal_l1029_102978

theorem arithmetic_mean_reciprocal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  b = (a + c) / 2 →
  (2 / b = 1 / a + 1 / c ∨ 2 / a = 1 / b + 1 / c ∨ 2 / c = 1 / a + 1 / b) →
  (∃ x : ℝ, x ≠ 0 ∧ (a = x ∧ b = x ∧ c = x ∨ a = -4*x ∧ b = -x ∧ c = 2*x)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_reciprocal_l1029_102978


namespace NUMINAMATH_CALUDE_souvenir_sales_profit_l1029_102984

/-- Represents the profit function for souvenir sales -/
def profit_function (x : ℝ) : ℝ :=
  (x - 5) * (32 - 4 * (x - 9))

/-- Theorem stating the properties of the profit function -/
theorem souvenir_sales_profit :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ profit_function x₁ = 140 ∧ profit_function x₂ = 140 ∧ 
    (x₁ = 10 ∨ x₁ = 12) ∧ (x₂ = 10 ∨ x₂ = 12)) ∧ 
  (∃ x_max : ℝ, x_max = 11 ∧ 
    ∀ x : ℝ, profit_function x ≤ profit_function x_max) ∧
  profit_function 11 = 144 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_sales_profit_l1029_102984


namespace NUMINAMATH_CALUDE_largest_base5_five_digit_to_base10_l1029_102934

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ (digits.length - 1 - i))) 0

/-- The largest five-digit number in base 5 --/
def largestBase5FiveDigit : List Nat := [4, 4, 4, 4, 4]

theorem largest_base5_five_digit_to_base10 :
  base5ToBase10 largestBase5FiveDigit = 3124 := by
  sorry

#eval base5ToBase10 largestBase5FiveDigit

end NUMINAMATH_CALUDE_largest_base5_five_digit_to_base10_l1029_102934


namespace NUMINAMATH_CALUDE_line_intersections_l1029_102931

/-- The line equation 2y + 5x = 10 -/
def line_equation (x y : ℝ) : Prop := 2 * y + 5 * x = 10

/-- X-axis intersection point -/
def x_intersection : ℝ × ℝ := (2, 0)

/-- Y-axis intersection point -/
def y_intersection : ℝ × ℝ := (0, 5)

/-- Theorem stating that the line intersects the x-axis and y-axis at the given points -/
theorem line_intersections :
  (line_equation x_intersection.1 x_intersection.2) ∧
  (line_equation y_intersection.1 y_intersection.2) ∧
  (x_intersection.2 = 0) ∧
  (y_intersection.1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_intersections_l1029_102931


namespace NUMINAMATH_CALUDE_tan_double_angle_l1029_102967

theorem tan_double_angle (α : Real) (h : Real.tan α = 1/3) : Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l1029_102967


namespace NUMINAMATH_CALUDE_sum_and_equal_numbers_l1029_102961

/-- Given three numbers x, y, and z satisfying certain conditions, prove that y equals 688/9 -/
theorem sum_and_equal_numbers (x y z : ℚ) 
  (h1 : x + y + z = 150)
  (h2 : x + 7 = y - 12)
  (h3 : x + 7 = 4 * z) : 
  y = 688 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_and_equal_numbers_l1029_102961


namespace NUMINAMATH_CALUDE_count_non_similar_triangles_l1029_102930

/-- A regular decagon with all diagonals drawn -/
structure RegularDecagonWithDiagonals where
  /-- The number of vertices in a decagon -/
  num_vertices : ℕ
  /-- The central angle of a regular decagon -/
  central_angle : ℝ
  /-- The internal angle of a regular decagon -/
  internal_angle : ℝ
  /-- The smallest angle increment between diagonals -/
  diagonal_angle_increment : ℝ
  /-- Assertion that the number of vertices is 10 -/
  vertices_eq : num_vertices = 10
  /-- Assertion that the central angle is 36° -/
  central_angle_eq : central_angle = 36
  /-- Assertion that the internal angle is 144° -/
  internal_angle_eq : internal_angle = 144
  /-- Assertion that the diagonal angle increment is 18° -/
  diagonal_angle_increment_eq : diagonal_angle_increment = 18

/-- The number of pairwise non-similar triangles in a regular decagon with all diagonals drawn -/
def num_non_similar_triangles (d : RegularDecagonWithDiagonals) : ℕ := 8

/-- Theorem stating that the number of pairwise non-similar triangles in a regular decagon with all diagonals drawn is 8 -/
theorem count_non_similar_triangles (d : RegularDecagonWithDiagonals) :
  num_non_similar_triangles d = 8 := by
  sorry

end NUMINAMATH_CALUDE_count_non_similar_triangles_l1029_102930


namespace NUMINAMATH_CALUDE_isabellaPaintableArea_l1029_102991

/-- Calculates the total paintable wall area in multiple bedrooms -/
def totalPaintableArea (
  numBedrooms : ℕ
  ) (length width height : ℝ
  ) (unpaintableArea : ℝ
  ) : ℝ := by
  sorry

/-- Theorem stating the total paintable wall area for the given conditions -/
theorem isabellaPaintableArea :
  totalPaintableArea 3 12 10 8 60 = 876 := by
  sorry

end NUMINAMATH_CALUDE_isabellaPaintableArea_l1029_102991


namespace NUMINAMATH_CALUDE_fraction_value_l1029_102929

theorem fraction_value (a b : ℝ) (h1 : a * b ≠ 0) (h2 : 3 * a + 2 * b = 0) :
  (2 * a + b) / b = -1/3 := by sorry

end NUMINAMATH_CALUDE_fraction_value_l1029_102929


namespace NUMINAMATH_CALUDE_rick_sean_money_ratio_l1029_102959

theorem rick_sean_money_ratio :
  ∀ (fritz_money sean_money rick_money : ℝ),
    fritz_money = 40 →
    sean_money = fritz_money / 2 + 4 →
    rick_money + sean_money = 96 →
    rick_money / sean_money = 3 := by
  sorry

end NUMINAMATH_CALUDE_rick_sean_money_ratio_l1029_102959


namespace NUMINAMATH_CALUDE_ab_max_and_sum_squares_min_l1029_102906

theorem ab_max_and_sum_squares_min (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 1) : 
  (∀ x y, 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → a * b ≥ x * y) ∧
  (∀ x y, 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → 4 * a^2 + b^2 ≤ 4 * x^2 + y^2) ∧
  a * b = 1/8 ∧
  4 * a^2 + b^2 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ab_max_and_sum_squares_min_l1029_102906


namespace NUMINAMATH_CALUDE_equation_solution_l1029_102905

theorem equation_solution (y : ℕ) : 9^10 + 9^10 + 9^10 = 3^y → y = 21 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1029_102905


namespace NUMINAMATH_CALUDE_f_increasing_l1029_102966

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x - Real.sin x else x^3 + 1

theorem f_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_f_increasing_l1029_102966


namespace NUMINAMATH_CALUDE_triangle_probability_l1029_102922

def segment_lengths : List ℝ := [1, 3, 5, 7, 9]

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def valid_triangle_count : ℕ := 3

def total_combinations : ℕ := 10

theorem triangle_probability : 
  (valid_triangle_count : ℚ) / (total_combinations : ℚ) = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_triangle_probability_l1029_102922


namespace NUMINAMATH_CALUDE_function_property_l1029_102981

-- Define the function type
def FunctionQ := ℚ → ℚ

-- Define the property that the function must satisfy
def SatisfiesProperty (f : FunctionQ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1

-- Theorem statement
theorem function_property (f : FunctionQ) (h : SatisfiesProperty f) :
  ∀ x : ℚ, f x = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1029_102981


namespace NUMINAMATH_CALUDE_perpendicular_sum_limit_l1029_102979

/-- Given two distinct straight lines and alternating perpendiculars between them,
    prove that the sum of perpendicular lengths converges to a²/(a - b) -/
theorem perpendicular_sum_limit (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  let r := b / a
  ∃ (S : ℝ), (∑' n, a * r^n) = S ∧ S = a^2 / (a - b) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_sum_limit_l1029_102979


namespace NUMINAMATH_CALUDE_count_triangle_points_l1029_102915

/-- The number of integer points in the triangle OAB -/
def triangle_points : ℕ :=
  (Finset.range 99).sum (fun k => 2 * k - 1)

/-- The theorem stating the number of integer points in the triangle OAB -/
theorem count_triangle_points :
  triangle_points = 9801 := by sorry

end NUMINAMATH_CALUDE_count_triangle_points_l1029_102915


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l1029_102957

theorem complex_exponential_sum (γ δ : ℝ) : 
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = -5/8 + 9/10 * Complex.I → 
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = -5/8 - 9/10 * Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l1029_102957


namespace NUMINAMATH_CALUDE_average_of_three_l1029_102943

theorem average_of_three (q₁ q₂ q₃ q₄ q₅ : ℝ) : 
  (q₁ + q₂ + q₃ + q₄ + q₅) / 5 = 12 →
  (q₄ + q₅) / 2 = 24 →
  (q₁ + q₂ + q₃) / 3 = 4 := by
sorry

end NUMINAMATH_CALUDE_average_of_three_l1029_102943


namespace NUMINAMATH_CALUDE_harvest_duration_l1029_102941

/-- The number of weeks of harvest given total earnings and weekly earnings -/
def harvest_weeks (total_earnings weekly_earnings : ℕ) : ℕ :=
  total_earnings / weekly_earnings

/-- Theorem: The harvest lasted for 76 weeks -/
theorem harvest_duration :
  harvest_weeks 1216 16 = 76 := by
  sorry

end NUMINAMATH_CALUDE_harvest_duration_l1029_102941


namespace NUMINAMATH_CALUDE_marks_speed_l1029_102952

/-- Given information about Mark and Chris's journey to school, prove Mark's speed -/
theorem marks_speed (chris_speed : ℝ) (school_distance : ℝ) (mark_initial_distance : ℝ) (time_difference : ℝ) 
  (h1 : chris_speed = 3)
  (h2 : school_distance = 9)
  (h3 : mark_initial_distance = 3)
  (h4 : time_difference = 2) :
  let chris_time := school_distance / chris_speed
  let mark_total_distance := mark_initial_distance * 2 + school_distance
  let mark_time := chris_time + time_difference
  mark_total_distance / mark_time = 3 := by sorry

end NUMINAMATH_CALUDE_marks_speed_l1029_102952


namespace NUMINAMATH_CALUDE_no_real_solutions_l1029_102938

theorem no_real_solutions : 
  ∀ x : ℝ, (x - 3*x + 7)^2 + 1 ≠ -abs x := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1029_102938


namespace NUMINAMATH_CALUDE_donut_distribution_l1029_102940

/-- The number of ways to distribute n indistinguishable objects into k distinguishable categories,
    with at least one object in each category. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem stating that there are 35 ways to distribute 8 donuts into 5 varieties
    with at least one donut of each variety. -/
theorem donut_distribution : distribute 8 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_donut_distribution_l1029_102940


namespace NUMINAMATH_CALUDE_triangle_third_angle_l1029_102968

theorem triangle_third_angle (a b c : ℝ) (ha : a = 70) (hb : b = 50) 
  (sum_of_angles : a + b + c = 180) : c = 60 := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_angle_l1029_102968


namespace NUMINAMATH_CALUDE_chebyshev_birth_year_l1029_102986

/-- Represents a year in the 19th century -/
structure Year1800s where
  tens : Nat
  units : Nat

/-- Checks if the given year satisfies all the conditions for P.L. Chebyshev's birth year -/
def is_chebyshev_birth_year (y : Year1800s) : Prop :=
  -- Sum of digits in hundreds and thousands (1 + 8 = 9) is 3 times the sum of digits in tens and units
  9 = 3 * (y.tens + y.units) ∧
  -- Digit in tens place is greater than digit in units place
  y.tens > y.units ∧
  -- Born and died in the same century (19th century)
  1800 + 10 * y.tens + y.units + 73 < 1900

/-- Theorem stating that 1821 is the unique year satisfying all conditions -/
theorem chebyshev_birth_year : 
  ∃! (y : Year1800s), is_chebyshev_birth_year y ∧ 1800 + 10 * y.tens + y.units = 1821 :=
sorry

end NUMINAMATH_CALUDE_chebyshev_birth_year_l1029_102986


namespace NUMINAMATH_CALUDE_three_x_plus_four_l1029_102999

theorem three_x_plus_four (x : ℝ) : x = 5 → 3 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_three_x_plus_four_l1029_102999


namespace NUMINAMATH_CALUDE_max_value_of_e_l1029_102920

theorem max_value_of_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  e ≤ 16/5 ∧ ∃ a b c d, a + b + c + d + 16/5 = 8 ∧ a^2 + b^2 + c^2 + d^2 + (16/5)^2 = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_e_l1029_102920


namespace NUMINAMATH_CALUDE_correct_derivatives_l1029_102970

theorem correct_derivatives :
  ∀ x : ℝ,
  (deriv (λ x => x / (x + 1) - 2^x)) x = 1 / (x + 1)^2 - 2^x * Real.log 2 ∧
  (deriv (λ x => x^2 / Real.exp x)) x = (2*x - x^2) / Real.exp x :=
by sorry

end NUMINAMATH_CALUDE_correct_derivatives_l1029_102970


namespace NUMINAMATH_CALUDE_stratified_sampling_results_l1029_102919

theorem stratified_sampling_results (junior_students senior_students sample_size : ℕ) 
  (h1 : junior_students = 400)
  (h2 : senior_students = 200)
  (h3 : sample_size = 60) :
  let junior_sample := (junior_students * sample_size) / (junior_students + senior_students)
  let senior_sample := sample_size - junior_sample
  Nat.choose junior_students junior_sample * Nat.choose senior_students senior_sample =
  Nat.choose 400 40 * Nat.choose 200 20 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_results_l1029_102919


namespace NUMINAMATH_CALUDE_removed_cone_height_l1029_102900

-- Define the frustum
structure Frustum where
  height : ℝ
  lowerBaseArea : ℝ
  upperBaseArea : ℝ

-- Define the theorem
theorem removed_cone_height (f : Frustum) (h1 : f.height = 30)
  (h2 : f.lowerBaseArea = 400 * Real.pi) (h3 : f.upperBaseArea = 100 * Real.pi) :
  ∃ (removedHeight : ℝ), removedHeight = f.height := by
  sorry

end NUMINAMATH_CALUDE_removed_cone_height_l1029_102900


namespace NUMINAMATH_CALUDE_chess_tournament_schedules_l1029_102942

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  players_per_school : Nat
  games_per_player : Nat
  games_per_round : Nat

/-- Calculates the total number of games in the tournament -/
def total_games (t : ChessTournament) : Nat :=
  t.players_per_school * t.players_per_school * t.games_per_player

/-- Calculates the number of rounds in the tournament -/
def num_rounds (t : ChessTournament) : Nat :=
  total_games t / t.games_per_round

/-- Theorem stating the number of ways to schedule the tournament -/
theorem chess_tournament_schedules (t : ChessTournament) 
  (h1 : t.players_per_school = 4)
  (h2 : t.games_per_player = 2)
  (h3 : t.games_per_round = 4) :
  Nat.factorial (num_rounds t) = 40320 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_schedules_l1029_102942


namespace NUMINAMATH_CALUDE_robbery_participants_l1029_102950

theorem robbery_participants (A B V G : Prop) 
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G ∧ ¬V := by sorry

end NUMINAMATH_CALUDE_robbery_participants_l1029_102950


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_bound_l1029_102972

theorem right_triangle_leg_sum_bound (a b c : ℝ) (h : a^2 + b^2 = c^2) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + b ≤ Real.sqrt 2 * c := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_bound_l1029_102972


namespace NUMINAMATH_CALUDE_tom_annual_lease_cost_l1029_102956

/-- Represents Tom's car lease scenario -/
structure CarLease where
  short_drive_days : Nat
  short_drive_miles : Nat
  long_drive_miles : Nat
  cost_per_mile : Rat
  weekly_fee : Nat

/-- Calculates the total annual cost for Tom's car lease -/
def annual_cost (lease : CarLease) : Rat :=
  let days_in_week : Nat := 7
  let weeks_in_year : Nat := 52
  let long_drive_days : Nat := days_in_week - lease.short_drive_days
  let weekly_mileage : Nat := lease.short_drive_days * lease.short_drive_miles + long_drive_days * lease.long_drive_miles
  let weekly_mileage_cost : Rat := (weekly_mileage : Rat) * lease.cost_per_mile
  let total_weekly_cost : Rat := weekly_mileage_cost + (lease.weekly_fee : Rat)
  total_weekly_cost * (weeks_in_year : Rat)

/-- Theorem stating that Tom's annual car lease cost is $7800 -/
theorem tom_annual_lease_cost :
  let tom_lease : CarLease := {
    short_drive_days := 4
    short_drive_miles := 50
    long_drive_miles := 100
    cost_per_mile := 1/10
    weekly_fee := 100
  }
  annual_cost tom_lease = 7800 := by sorry

end NUMINAMATH_CALUDE_tom_annual_lease_cost_l1029_102956


namespace NUMINAMATH_CALUDE_negative_seven_to_fourth_power_l1029_102983

theorem negative_seven_to_fourth_power : (-7 : ℤ) ^ 4 = (-7) * (-7) * (-7) * (-7) := by
  sorry

end NUMINAMATH_CALUDE_negative_seven_to_fourth_power_l1029_102983


namespace NUMINAMATH_CALUDE_x_plus_y_plus_2009_l1029_102982

theorem x_plus_y_plus_2009 (x y : ℝ) 
  (h1 : |x| + x + 5*y = 2) 
  (h2 : |y| - y + x = 7) : 
  x + y + 2009 = 2012 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_plus_2009_l1029_102982


namespace NUMINAMATH_CALUDE_daves_initial_files_l1029_102912

theorem daves_initial_files (initial_apps : ℕ) (final_apps : ℕ) (final_files : ℕ) (deleted_files : ℕ) : 
  initial_apps = 17 → 
  final_apps = 3 → 
  final_files = 7 → 
  deleted_files = 14 → 
  ∃ initial_files : ℕ, initial_files = 21 ∧ initial_files = final_files + deleted_files :=
by sorry

end NUMINAMATH_CALUDE_daves_initial_files_l1029_102912


namespace NUMINAMATH_CALUDE_parallel_lines_m_values_l1029_102921

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_lines_m_values :
  ∀ m : ℝ,
  let l1 : Line := { a := 3, b := m, c := -1 }
  let l2 : Line := { a := m + 2, b := -(m - 2), c := 2 }
  parallel l1 l2 → m = 1 ∨ m = -6 := by
    sorry

end NUMINAMATH_CALUDE_parallel_lines_m_values_l1029_102921


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1029_102927

/-- Given a quadratic inequality mx^2 + 8mx + 28 < 0 with solution set {x | -7 < x < -1},
    prove that m = 4 -/
theorem quadratic_inequality_solution (m : ℝ) 
  (h : ∀ x, mx^2 + 8*m*x + 28 < 0 ↔ -7 < x ∧ x < -1) : 
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1029_102927


namespace NUMINAMATH_CALUDE_concert_ticket_problem_l1029_102975

/-- Represents the number of student tickets sold -/
def student_tickets : ℕ := sorry

/-- Represents the number of non-student tickets sold -/
def non_student_tickets : ℕ := sorry

/-- The price of a student ticket in dollars -/
def student_price : ℕ := 9

/-- The price of a non-student ticket in dollars -/
def non_student_price : ℕ := 11

/-- The total number of tickets sold -/
def total_tickets : ℕ := 2000

/-- The total revenue from ticket sales in dollars -/
def total_revenue : ℕ := 20960

theorem concert_ticket_problem :
  (student_tickets + non_student_tickets = total_tickets) ∧
  (student_tickets * student_price + non_student_tickets * non_student_price = total_revenue) →
  student_tickets = 520 := by sorry

end NUMINAMATH_CALUDE_concert_ticket_problem_l1029_102975


namespace NUMINAMATH_CALUDE_placard_distribution_l1029_102989

theorem placard_distribution (total_placards : ℕ) (total_people : ℕ) 
  (h1 : total_placards = 823) 
  (h2 : total_people = 412) :
  (total_placards : ℚ) / total_people = 2 := by
sorry

end NUMINAMATH_CALUDE_placard_distribution_l1029_102989


namespace NUMINAMATH_CALUDE_divisor_of_p_l1029_102947

theorem divisor_of_p (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 40)
  (h2 : Nat.gcd q.val r.val = 50)
  (h3 : Nat.gcd r.val s.val = 60)
  (h4 : 80 < Nat.gcd s.val p.val ∧ Nat.gcd s.val p.val < 120) :
  13 ∣ p.val :=
by sorry

end NUMINAMATH_CALUDE_divisor_of_p_l1029_102947


namespace NUMINAMATH_CALUDE_trapezoid_segment_ratio_l1029_102985

/-- A trapezoid with specific properties -/
structure Trapezoid where
  upperLength : ℝ
  lowerLength : ℝ
  smallSegment : ℝ
  largeSegment : ℝ
  upperEquation : 3 * smallSegment + largeSegment = upperLength
  lowerEquation : 2 * largeSegment + 6 * smallSegment = lowerLength

/-- The ratio of the largest to smallest segment in a specific trapezoid is 2 -/
theorem trapezoid_segment_ratio (t : Trapezoid) 
    (h1 : t.upperLength = 1) 
    (h2 : t.lowerLength = 2) : 
  t.largeSegment / t.smallSegment = 2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_ratio_l1029_102985
