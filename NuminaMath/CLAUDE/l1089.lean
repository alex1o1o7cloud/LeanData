import Mathlib

namespace NUMINAMATH_CALUDE_dice_roll_probability_l1089_108947

def is_valid_roll (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6

def meets_conditions (a b c : ℕ) : Prop :=
  a * b * c = 72 ∧ a + b + c = 13

def total_outcomes : ℕ := 6 * 6 * 6

def favorable_outcomes : ℕ := 6

theorem dice_roll_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 36 :=
sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1089_108947


namespace NUMINAMATH_CALUDE_oil_division_l1089_108949

/-- Proves that given 12.4 liters of oil divided into two bottles, where the large bottle can hold 2.6 liters more than the small bottle, the large bottle will hold 7.5 liters. -/
theorem oil_division (total_oil : ℝ) (difference : ℝ) (large_bottle : ℝ) : 
  total_oil = 12.4 →
  difference = 2.6 →
  large_bottle = (total_oil + difference) / 2 →
  large_bottle = 7.5 :=
by
  sorry

end NUMINAMATH_CALUDE_oil_division_l1089_108949


namespace NUMINAMATH_CALUDE_base_conversion_2345_to_base_7_l1089_108972

theorem base_conversion_2345_to_base_7 :
  (2345 : ℕ) = 6 * (7 : ℕ)^3 + 5 * (7 : ℕ)^2 + 6 * (7 : ℕ)^1 + 0 * (7 : ℕ)^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_2345_to_base_7_l1089_108972


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1089_108987

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the third term is 20 and the sixth term is 26,
    prove that the ninth term is 32. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third_term : a 3 = 20)
  (h_sixth_term : a 6 = 26) :
  a 9 = 32 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1089_108987


namespace NUMINAMATH_CALUDE_units_digit_of_sum_64_75_base_8_l1089_108936

/-- Represents a number in base 8 --/
def OctalNum := Nat

/-- Converts a base 10 number to its base 8 representation --/
def toOctal (n : Nat) : OctalNum := sorry

/-- Adds two numbers in base 8 --/
def octalAdd (a b : OctalNum) : OctalNum := sorry

/-- Gets the units digit of a number in base 8 --/
def unitsDigit (n : OctalNum) : Nat := sorry

theorem units_digit_of_sum_64_75_base_8 :
  unitsDigit (octalAdd (toOctal 64) (toOctal 75)) = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sum_64_75_base_8_l1089_108936


namespace NUMINAMATH_CALUDE_escalator_length_is_160_l1089_108901

/-- The length of an escalator given its speed, a person's walking speed on it, and the time taken to cover the entire length. -/
def escalatorLength (escalatorSpeed personSpeed : ℝ) (timeTaken : ℝ) : ℝ :=
  (escalatorSpeed + personSpeed) * timeTaken

/-- Theorem stating that the length of the escalator is 160 feet under the given conditions. -/
theorem escalator_length_is_160 :
  escalatorLength 12 8 8 = 160 := by
  sorry

end NUMINAMATH_CALUDE_escalator_length_is_160_l1089_108901


namespace NUMINAMATH_CALUDE_tangent_line_a_value_l1089_108959

/-- A line in polar coordinates tangent to a circle. -/
structure PolarLineTangentToCircle where
  a : ℝ
  tangent_line : ℝ → ℝ → Prop
  circle : ℝ → ℝ → Prop
  a_positive : a > 0
  is_tangent : ∀ θ ρ, tangent_line ρ θ ↔ ρ * (Real.cos θ + Real.sin θ) = a
  circle_eq : ∀ θ ρ, circle ρ θ ↔ ρ = 2 * Real.cos θ

/-- The value of 'a' for a line tangent to the given circle is 1 + √2. -/
theorem tangent_line_a_value (h : PolarLineTangentToCircle) : h.a = 1 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_a_value_l1089_108959


namespace NUMINAMATH_CALUDE_solve_for_z_l1089_108925

theorem solve_for_z : ∃ z : ℝ, (Real.sqrt 27 + Real.sqrt 243) / Real.sqrt z = 2.4 → z = 75 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_z_l1089_108925


namespace NUMINAMATH_CALUDE_parabola_area_ratio_l1089_108943

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Line passing through a point -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- Intersection of a line with x = -2 -/
def intersectionWithM (l : Line) : ℝ × ℝ :=
  (-2, l.slope * (-3 - l.point.1) + l.point.2)

/-- Area of a triangle -/
noncomputable def triangleArea (a b c : ℝ × ℝ) : ℝ :=
  sorry

/-- Main theorem -/
theorem parabola_area_ratio 
  (C : Parabola)
  (F : ℝ × ℝ)
  (L : Line)
  (A B : ℝ × ℝ)
  (O : ℝ × ℝ := (0, 0))
  (M N : ℝ × ℝ) :
  C.p = 2 →
  F = (1, 0) →
  L.point = F →
  C.equation A.1 A.2 →
  C.equation B.1 B.2 →
  M = intersectionWithM ⟨O, (A.2 - O.2) / (A.1 - O.1)⟩ →
  N = intersectionWithM ⟨O, (B.2 - O.2) / (B.1 - O.1)⟩ →
  (triangleArea A B O) / (triangleArea M N O) = 1/4 :=
sorry

end NUMINAMATH_CALUDE_parabola_area_ratio_l1089_108943


namespace NUMINAMATH_CALUDE_unique_solution_triple_sqrt_plus_four_l1089_108965

theorem unique_solution_triple_sqrt_plus_four :
  ∃! x : ℝ, x > 0 ∧ x = 3 * Real.sqrt x + 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_triple_sqrt_plus_four_l1089_108965


namespace NUMINAMATH_CALUDE_will_earnings_l1089_108928

def hourly_wage : ℕ := 8
def monday_hours : ℕ := 8
def tuesday_hours : ℕ := 2

theorem will_earnings : 
  hourly_wage * monday_hours + hourly_wage * tuesday_hours = 80 := by
  sorry

end NUMINAMATH_CALUDE_will_earnings_l1089_108928


namespace NUMINAMATH_CALUDE_total_distance_walked_l1089_108929

/-- The distance Spencer walked from his house to the library -/
def distance_house_to_library : ℝ := 0.3

/-- The distance Spencer walked from the library to the post office -/
def distance_library_to_post_office : ℝ := 0.1

/-- The distance Spencer walked from the post office back home -/
def distance_post_office_to_house : ℝ := 0.4

/-- The theorem stating that the total distance Spencer walked is 0.8 miles -/
theorem total_distance_walked :
  distance_house_to_library + distance_library_to_post_office + distance_post_office_to_house = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l1089_108929


namespace NUMINAMATH_CALUDE_root_in_interval_l1089_108914

theorem root_in_interval (a : ℤ) : 
  (∃ x : ℝ, x > a ∧ x < a + 1 ∧ Real.log x + x - 4 = 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1089_108914


namespace NUMINAMATH_CALUDE_inequality_solution_l1089_108916

def solution_set : Set ℝ := {x : ℝ | -5 < x ∧ x < 1 ∨ x > 6}

theorem inequality_solution :
  {x : ℝ | (x - 1) / (x^2 - x - 30) > 0} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1089_108916


namespace NUMINAMATH_CALUDE_average_sequence_l1089_108994

theorem average_sequence (x : ℚ) : 
  (List.sum (List.range 149) + x) / 150 = 50 * x → x = 11175 / 7499 := by
  sorry

end NUMINAMATH_CALUDE_average_sequence_l1089_108994


namespace NUMINAMATH_CALUDE_student_calculation_l1089_108966

theorem student_calculation (x : ℤ) (h : x = 110) : 3 * x - 220 = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_l1089_108966


namespace NUMINAMATH_CALUDE_arc_length_for_120_degrees_l1089_108933

-- Define the radius of the circle
def radius : ℝ := 6

-- Define the central angle in degrees
def central_angle : ℝ := 120

-- Define pi as a real number (since Lean doesn't have a built-in pi constant)
noncomputable def π : ℝ := Real.pi

-- State the theorem
theorem arc_length_for_120_degrees (r : ℝ) (θ : ℝ) :
  r = radius → θ = central_angle →
  (θ / 360) * (2 * π * r) = 4 * π :=
by sorry

end NUMINAMATH_CALUDE_arc_length_for_120_degrees_l1089_108933


namespace NUMINAMATH_CALUDE_only_solutions_for_equation_l1089_108953

theorem only_solutions_for_equation (x y : ℕ) : 
  33^x + 31 = 2^y ↔ (x = 0 ∧ y = 5) ∨ (x = 1 ∧ y = 6) := by
  sorry

end NUMINAMATH_CALUDE_only_solutions_for_equation_l1089_108953


namespace NUMINAMATH_CALUDE_triangle_area_345_l1089_108975

theorem triangle_area_345 (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) :
  (1/2 : ℝ) * a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_345_l1089_108975


namespace NUMINAMATH_CALUDE_train_length_is_60_l1089_108910

/-- Two trains with equal length on parallel tracks -/
structure TrainSystem where
  train_length : ℝ
  fast_speed : ℝ
  slow_speed : ℝ
  passing_time : ℝ

/-- The train system satisfies the given conditions -/
def valid_train_system (ts : TrainSystem) : Prop :=
  ts.fast_speed = 72 * (5/18) ∧  -- 72 km/h in m/s
  ts.slow_speed = 54 * (5/18) ∧  -- 54 km/h in m/s
  ts.passing_time = 24

/-- Theorem stating that the length of each train is 60 meters -/
theorem train_length_is_60 (ts : TrainSystem) 
  (h : valid_train_system ts) : ts.train_length = 60 := by
  sorry

#check train_length_is_60

end NUMINAMATH_CALUDE_train_length_is_60_l1089_108910


namespace NUMINAMATH_CALUDE_unique_fraction_decomposition_l1089_108948

theorem unique_fraction_decomposition (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (m n : ℕ), m ≠ n ∧ m > 0 ∧ n > 0 ∧ (2 : ℚ) / p = 1 / n + 1 / m ∧
  n = (p + 1) / 2 ∧ m = p * (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_fraction_decomposition_l1089_108948


namespace NUMINAMATH_CALUDE_inequality_proof_l1089_108971

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 1) : 
  (a^4 + b^4)/(a^6 + b^6) + (b^4 + c^4)/(b^6 + c^6) + (c^4 + a^4)/(c^6 + a^6) ≤ 1/(a*b*c) := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l1089_108971


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1089_108938

/-- The perimeter of a trapezoid JKLM with given coordinates -/
theorem trapezoid_perimeter : 
  let J : ℝ × ℝ := (-2, -4)
  let K : ℝ × ℝ := (-2, 1)
  let L : ℝ × ℝ := (6, 7)
  let M : ℝ × ℝ := (6, -4)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist J K + dist K L + dist L M + dist M J = 34 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1089_108938


namespace NUMINAMATH_CALUDE_task_completion_time_l1089_108983

/-- Given two workers can complete a task in 35 days, and one worker can complete it in 60 days,
    prove that the other worker can complete the task in 84 days. -/
theorem task_completion_time (total_time : ℝ) (worker1_time : ℝ) (worker2_time : ℝ) : 
  (1 / total_time = 1 / worker1_time + 1 / worker2_time) →
  (total_time = 35) →
  (worker1_time = 60) →
  (worker2_time = 84) := by
sorry

end NUMINAMATH_CALUDE_task_completion_time_l1089_108983


namespace NUMINAMATH_CALUDE_hyperbola_condition_l1089_108915

theorem hyperbola_condition (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (m + 2) - y^2 / (m - 1) = 1) → (m < -2 ∨ m > 1) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l1089_108915


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_three_l1089_108995

theorem sum_of_roots_equals_three : ∃ (P Q : ℝ), P + Q = 3 ∧ 3 * P^2 - 9 * P + 6 = 0 ∧ 3 * Q^2 - 9 * Q + 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_three_l1089_108995


namespace NUMINAMATH_CALUDE_total_oil_needed_l1089_108997

/-- Represents the oil requirements for a bicycle -/
structure BikeOil where
  wheel : ℕ  -- Oil needed for one wheel
  chain : ℕ  -- Oil needed for the chain
  pedals : ℕ -- Oil needed for the pedals
  brakes : ℕ -- Oil needed for the brakes

/-- Calculates the total oil needed for a bicycle -/
def totalOilForBike (bike : BikeOil) : ℕ :=
  2 * bike.wheel + bike.chain + bike.pedals + bike.brakes

/-- The oil requirements for the first bicycle -/
def bike1 : BikeOil := 
  { wheel := 20, chain := 15, pedals := 8, brakes := 10 }

/-- The oil requirements for the second bicycle -/
def bike2 : BikeOil := 
  { wheel := 25, chain := 18, pedals := 10, brakes := 12 }

/-- The oil requirements for the third bicycle -/
def bike3 : BikeOil := 
  { wheel := 30, chain := 20, pedals := 12, brakes := 15 }

/-- Theorem stating the total oil needed for all three bicycles -/
theorem total_oil_needed : 
  totalOilForBike bike1 + totalOilForBike bike2 + totalOilForBike bike3 = 270 := by
  sorry

end NUMINAMATH_CALUDE_total_oil_needed_l1089_108997


namespace NUMINAMATH_CALUDE_good_number_proof_l1089_108908

theorem good_number_proof :
  ∃! n : ℕ, n ∈ Finset.range 2016 ∧
  (Finset.sum (Finset.range 2016) id - n) % 2016 = 0 ∧
  n = 1008 := by
sorry

end NUMINAMATH_CALUDE_good_number_proof_l1089_108908


namespace NUMINAMATH_CALUDE_alpha_beta_inequality_l1089_108924

theorem alpha_beta_inequality (α β : ℝ) : α > β ↔ α - β > Real.sin α - Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_alpha_beta_inequality_l1089_108924


namespace NUMINAMATH_CALUDE_salary_problem_l1089_108977

theorem salary_problem (A_salary B_salary : ℝ) 
  (h1 : A_salary = 4500)
  (h2 : A_salary * 0.05 = B_salary * 0.15)
  : A_salary + B_salary = 6000 := by
  sorry

end NUMINAMATH_CALUDE_salary_problem_l1089_108977


namespace NUMINAMATH_CALUDE_quotient_base4_l1089_108911

/-- Converts a base 4 number represented as a list of digits to its decimal equivalent -/
def base4ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a decimal number to its base 4 representation -/
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec helper (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else helper (m / 4) ((m % 4) :: acc)
  helper n []

/-- Theorem: The quotient of 1213₄ divided by 13₄ is equal to 32₄ -/
theorem quotient_base4 :
  let a := base4ToDecimal [3, 1, 2, 1]  -- 1213₄
  let b := base4ToDecimal [3, 1]        -- 13₄
  decimalToBase4 (a / b) = [2, 3]       -- 32₄
  := by sorry

end NUMINAMATH_CALUDE_quotient_base4_l1089_108911


namespace NUMINAMATH_CALUDE_nathan_warmth_increase_l1089_108920

def blankets_in_closet : ℕ := 14
def warmth_per_blanket : ℕ := 3

def warmth_increase (blankets_used : ℕ) : ℕ :=
  blankets_used * warmth_per_blanket

theorem nathan_warmth_increase :
  warmth_increase (blankets_in_closet / 2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_nathan_warmth_increase_l1089_108920


namespace NUMINAMATH_CALUDE_intersection_A_B_l1089_108980

def A : Set ℤ := {-1, 0, 1, 5, 8}
def B : Set ℤ := {x | x > 1}

theorem intersection_A_B : A ∩ B = {5, 8} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l1089_108980


namespace NUMINAMATH_CALUDE_ln_abs_properties_l1089_108945

-- Define the function f(x) = ln|x|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem ln_abs_properties :
  (∀ x ≠ 0, f (-x) = f x) ∧  -- f is even
  (∀ x y, 0 < x → x < y → f x < f y) :=  -- f is increasing on (0, +∞)
by sorry

end NUMINAMATH_CALUDE_ln_abs_properties_l1089_108945


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1089_108992

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

theorem quadratic_coefficient (a b c : ℤ) :
  (∀ x, QuadraticFunction a b c x = a * (x - 2)^2 + 5) →
  QuadraticFunction a b c 1 = 2 →
  a = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1089_108992


namespace NUMINAMATH_CALUDE_green_face_probability_octahedral_die_l1089_108968

/-- An octahedral die with green and yellow faces -/
structure OctahedralDie where
  total_faces : Nat
  green_faces : Nat
  yellow_faces : Nat

/-- The probability of rolling a green face on an octahedral die -/
def green_face_probability (die : OctahedralDie) : Rat :=
  die.green_faces / die.total_faces

/-- Theorem: The probability of rolling a green face on an octahedral die
    with 5 green faces and 3 yellow faces is 5/8 -/
theorem green_face_probability_octahedral_die :
  let die : OctahedralDie := {
    total_faces := 8,
    green_faces := 5,
    yellow_faces := 3
  }
  green_face_probability die = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_green_face_probability_octahedral_die_l1089_108968


namespace NUMINAMATH_CALUDE_ellipse_properties_l1089_108990

/-- Ellipse C passing through points A(2,0) and B(0,1) -/
def ellipse_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Line on which point P lies -/
def line_P (x y : ℝ) : Prop := x + y = 4

/-- Point Q on ellipse C -/
def point_Q (x y : ℝ) : Prop := ellipse_C x y

/-- Parallelogram condition for PAQB -/
def is_parallelogram (px py qx qy : ℝ) : Prop :=
  px + qx = 2 ∧ py + qy = 1

theorem ellipse_properties :
  ∃ (e : ℝ),
    (∀ x y, ellipse_C x y ↔ x^2 / 4 + y^2 = 1) ∧
    e = Real.sqrt 3 / 2 ∧
    ∃ px py qx qy,
      line_P px py ∧
      point_Q qx qy ∧
      is_parallelogram px py qx qy ∧
      ((px = 18/5 ∧ py = 2/5) ∨ (px = 2 ∧ py = 2)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1089_108990


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l1089_108903

def is_valid_cryptarithm (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ 
  ∃ (a b c d : ℕ), n * n = 1000 * c + 100 * d + n ∧ 
  c ≠ 0

theorem cryptarithm_solution :
  ∃! (S : Set ℕ), 
    (∀ n ∈ S, is_valid_cryptarithm n ∧ Odd n) ∧ 
    (∀ n, is_valid_cryptarithm n → Odd n → n ∈ S) ∧
    (∃ m, m ∈ S) ∧
    (∀ T : Set ℕ, (∀ n ∈ T, is_valid_cryptarithm n ∧ Even n) → T.Nonempty → ¬(∃! x, x ∈ T)) :=
by sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l1089_108903


namespace NUMINAMATH_CALUDE_f_image_is_zero_to_eight_l1089_108946

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the domain
def D : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }

-- Theorem statement
theorem f_image_is_zero_to_eight :
  Set.image f D = { y | 0 ≤ y ∧ y ≤ 8 } := by
  sorry

end NUMINAMATH_CALUDE_f_image_is_zero_to_eight_l1089_108946


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1089_108993

/-- The equation of a circle in the form (x - h)^2 + (y - k)^2 = r^2 -/
def CircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

theorem circle_center_and_radius :
  ∃ (h k r : ℝ), 
    (∀ x y : ℝ, CircleEquation h k r x y ↔ (x - 2)^2 + (y + 3)^2 = 2) ∧
    h = 2 ∧ k = -3 ∧ r = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l1089_108993


namespace NUMINAMATH_CALUDE_sqrt_product_difference_l1089_108926

theorem sqrt_product_difference (x y z w : ℝ) : 
  x = Real.sqrt 108 → 
  y = Real.sqrt 128 → 
  z = Real.sqrt 6 → 
  w = Real.sqrt 18 → 
  x * y * z - w = 288 - 3 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_difference_l1089_108926


namespace NUMINAMATH_CALUDE_trig_equality_l1089_108934

theorem trig_equality (a b : ℝ) (θ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (Real.sin θ ^ 6 / a ^ 2 + Real.cos θ ^ 6 / b ^ 2 = 1 / (a ^ 2 + b ^ 2)) →
  (Real.sin θ ^ 12 / a ^ 5 + Real.cos θ ^ 12 / b ^ 5 = 1 / a ^ 5 + 1 / b ^ 5) :=
by sorry

end NUMINAMATH_CALUDE_trig_equality_l1089_108934


namespace NUMINAMATH_CALUDE_baking_contest_votes_l1089_108979

theorem baking_contest_votes (witch_votes dragon_votes unicorn_votes : ℕ) : 
  witch_votes = 7 →
  unicorn_votes = 3 * witch_votes →
  dragon_votes > witch_votes →
  witch_votes + unicorn_votes + dragon_votes = 60 →
  dragon_votes - witch_votes = 25 := by
sorry

end NUMINAMATH_CALUDE_baking_contest_votes_l1089_108979


namespace NUMINAMATH_CALUDE_range_of_n_l1089_108988

theorem range_of_n (m n : ℝ) : (m^2 - 2*m)^2 + 4*m^2 - 8*m + 6 - n = 0 → n ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_n_l1089_108988


namespace NUMINAMATH_CALUDE_equation_solution_l1089_108905

theorem equation_solution (x : ℝ) : 
  (x / 6) / 3 = 9 / (x / 3) → x = 9 * Real.sqrt 6 ∨ x = -9 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1089_108905


namespace NUMINAMATH_CALUDE_stone_197_is_5_and_prime_l1089_108969

/-- The number of stones in the line -/
def num_stones : ℕ := 13

/-- The length of one full cycle in the counting pattern -/
def cycle_length : ℕ := 24

/-- The count we're interested in -/
def target_count : ℕ := 197

/-- Function to determine which stone corresponds to a given count -/
def stone_for_count (count : ℕ) : ℕ :=
  (count - 1) % cycle_length + 1

/-- Primality check -/
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

theorem stone_197_is_5_and_prime :
  stone_for_count target_count = 5 ∧ is_prime 5 := by
  sorry


end NUMINAMATH_CALUDE_stone_197_is_5_and_prime_l1089_108969


namespace NUMINAMATH_CALUDE_equation_solutions_l1089_108963

theorem equation_solutions : 
  ∃! (s : Set ℝ), s = {x : ℝ | (50 - 3*x)^(1/4) + (30 + 3*x)^(1/4) = 4} ∧ s = {16, -14} :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1089_108963


namespace NUMINAMATH_CALUDE_m_mod_1000_l1089_108930

/-- The set of integers from 1 to 12 -/
def T : Finset ℕ := Finset.range 12

/-- The number of sets of two non-empty disjoint subsets of T -/
def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

/-- Theorem stating that the remainder of m divided by 1000 is 625 -/
theorem m_mod_1000 : m % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_m_mod_1000_l1089_108930


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l1089_108954

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the interval [-3, 3/2]
def interval : Set ℝ := Set.Icc (-3) (3/2)

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 2 ∧ min = -18 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l1089_108954


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1089_108922

/-- Parabola defined by y = x^2 + 5 -/
def P : ℝ → ℝ := λ x ↦ x^2 + 5

/-- Point Q -/
def Q : ℝ × ℝ := (10, 10)

/-- Line through Q with slope m -/
def line (m : ℝ) : ℝ → ℝ := λ x ↦ m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line m x

theorem parabola_line_intersection :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 40 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1089_108922


namespace NUMINAMATH_CALUDE_complex_number_classification_real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l1089_108913

def complex_number (m : ℝ) : ℂ := (1 + Complex.I) * m^2 - m * (5 + 3 * Complex.I) + 6

theorem complex_number_classification (m : ℝ) :
  (complex_number m).im = 0 ∨ 
  ((complex_number m).re ≠ 0 ∧ (complex_number m).im ≠ 0) ∨ 
  ((complex_number m).re = 0 ∧ (complex_number m).im ≠ 0) :=
by
  sorry

theorem real_number_condition (m : ℝ) :
  (complex_number m).im = 0 ↔ m = 0 ∨ m = 3 :=
by
  sorry

theorem imaginary_number_condition (m : ℝ) :
  ((complex_number m).re ≠ 0 ∧ (complex_number m).im ≠ 0) ↔ m ≠ 0 ∧ m ≠ 3 :=
by
  sorry

theorem pure_imaginary_number_condition (m : ℝ) :
  ((complex_number m).re = 0 ∧ (complex_number m).im ≠ 0) ↔ m = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_number_classification_real_number_condition_imaginary_number_condition_pure_imaginary_number_condition_l1089_108913


namespace NUMINAMATH_CALUDE_y_derivative_l1089_108909

noncomputable def y (x : ℝ) : ℝ := -2 * Real.exp x * Real.sin x

theorem y_derivative (x : ℝ) : 
  deriv y x = -2 * Real.exp x * (Real.cos x + Real.sin x) := by sorry

end NUMINAMATH_CALUDE_y_derivative_l1089_108909


namespace NUMINAMATH_CALUDE_solution_to_equation_l1089_108939

theorem solution_to_equation : ∃! (x : ℝ), x ≠ 0 ∧ (6 * x) ^ 5 = (12 * x) ^ 4 ∧ x = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1089_108939


namespace NUMINAMATH_CALUDE_nell_total_cards_l1089_108927

def initial_cards : ℝ := 304.5
def received_cards : ℝ := 276.25

theorem nell_total_cards : initial_cards + received_cards = 580.75 := by
  sorry

end NUMINAMATH_CALUDE_nell_total_cards_l1089_108927


namespace NUMINAMATH_CALUDE_songs_downloaded_l1089_108957

theorem songs_downloaded (internet_speed : ℕ) (song_size : ℕ) (download_time : ℕ) : 
  internet_speed = 20 → 
  song_size = 5 → 
  download_time = 1800 → 
  (internet_speed * download_time) / song_size = 7200 :=
by
  sorry

end NUMINAMATH_CALUDE_songs_downloaded_l1089_108957


namespace NUMINAMATH_CALUDE_tom_climbing_time_l1089_108902

/-- Tom and Elizabeth's hill climbing competition -/
theorem tom_climbing_time (elizabeth_time : ℕ) (tom_factor : ℕ) : elizabeth_time = 30 → tom_factor = 4 → (elizabeth_time * tom_factor) / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tom_climbing_time_l1089_108902


namespace NUMINAMATH_CALUDE_incorrect_value_calculation_l1089_108958

theorem incorrect_value_calculation (n : ℕ) (initial_mean correct_mean correct_value : ℝ) 
  (h1 : n = 25)
  (h2 : initial_mean = 190)
  (h3 : correct_mean = 191.4)
  (h4 : correct_value = 165) :
  ∃ incorrect_value : ℝ,
    incorrect_value = n * correct_mean - (n - 1) * initial_mean - correct_value ∧
    incorrect_value = 200 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_value_calculation_l1089_108958


namespace NUMINAMATH_CALUDE_polyhedron_property_l1089_108970

/-- A convex polyhedron with specific properties -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  euler_formula : V - E + F = 2
  face_count : F = 42
  face_types : t + h = F
  edge_formula : E = (3 * t + 6 * h) / 2
  vertex_face_relation : 3 * t + 2 * h = V

/-- The main theorem to be proved -/
theorem polyhedron_property (p : ConvexPolyhedron) : 
  100 * 2 + 10 * 3 + p.V = 328 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_property_l1089_108970


namespace NUMINAMATH_CALUDE_complete_square_property_l1089_108907

/-- A function to represent a quadratic expression of the form (p + qx)² + (r + sx)² -/
def quadraticExpression (p q r s : ℝ) (x : ℝ) : ℝ :=
  (p + q * x)^2 + (r + s * x)^2

/-- Predicate to check if a quadratic expression is a complete square -/
def isCompleteSquare (f : ℝ → ℝ) : Prop :=
  ∃ (k l : ℝ), ∀ x, f x = (k * x + l)^2

theorem complete_square_property 
  (a b c a' b' c' : ℝ) 
  (h1 : isCompleteSquare (quadraticExpression a b a' b'))
  (h2 : isCompleteSquare (quadraticExpression a c a' c')) :
  isCompleteSquare (quadraticExpression b c b' c') := by
  sorry

end NUMINAMATH_CALUDE_complete_square_property_l1089_108907


namespace NUMINAMATH_CALUDE_volume_is_1250_l1089_108964

/-- The volume of the solid bounded by the given surfaces -/
def volume_of_solid : ℝ :=
  let surface1 := {(x, y, z) : ℝ × ℝ × ℝ | x^2 / 27 + y^2 / 25 = 1}
  let surface2 := {(x, y, z) : ℝ × ℝ × ℝ | z = y / Real.sqrt 3}
  let surface3 := {(x, y, z) : ℝ × ℝ × ℝ | z = 0}
  let constraint := {(x, y, z) : ℝ × ℝ × ℝ | y ≥ 0}
  1250 -- placeholder for the actual volume

/-- Theorem stating that the volume of the solid is 1250 -/
theorem volume_is_1250 : volume_of_solid = 1250 := by
  sorry

end NUMINAMATH_CALUDE_volume_is_1250_l1089_108964


namespace NUMINAMATH_CALUDE_minimum_eccentricity_sum_l1089_108904

/-- Given two points F₁ and F₂ that are common foci of an ellipse and a hyperbola,
    and P is their common point. -/
structure CommonFociConfig where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ

/-- The eccentricity of the ellipse -/
def eccentricity_ellipse (config : CommonFociConfig) : ℝ := sorry

/-- The eccentricity of the hyperbola -/
def eccentricity_hyperbola (config : CommonFociConfig) : ℝ := sorry

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem minimum_eccentricity_sum (config : CommonFociConfig) 
  (h1 : distance config.P config.F₂ > distance config.P config.F₁)
  (h2 : distance config.P config.F₁ = distance config.F₁ config.F₂) :
  (∀ e₁ e₂ : ℝ, e₁ = eccentricity_ellipse config → e₂ = eccentricity_hyperbola config →
    3 / e₁ + e₂ / 3 ≥ 8) ∧ 
  (∃ e₁ e₂ : ℝ, e₁ = eccentricity_ellipse config ∧ e₂ = eccentricity_hyperbola config ∧
    3 / e₁ + e₂ / 3 = 8) :=
sorry

end NUMINAMATH_CALUDE_minimum_eccentricity_sum_l1089_108904


namespace NUMINAMATH_CALUDE_popcorn_probability_l1089_108982

theorem popcorn_probability (total : ℝ) (h_total_pos : 0 < total) : 
  let white := (3/4 : ℝ) * total
  let yellow := (1/4 : ℝ) * total
  let white_popped := (3/5 : ℝ) * white
  let yellow_popped := (3/4 : ℝ) * yellow
  let total_popped := white_popped + yellow_popped
  (white_popped / total_popped) = (12/17 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_popcorn_probability_l1089_108982


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l1089_108900

theorem fraction_equals_zero (x : ℝ) : (x + 2) / (x - 3) = 0 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l1089_108900


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l1089_108999

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) 
  (h4 : a + b = 7 * (a - b) + 14) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l1089_108999


namespace NUMINAMATH_CALUDE_smallest_multiples_sum_l1089_108985

theorem smallest_multiples_sum : ∃ c d : ℕ,
  (c ≥ 10 ∧ c < 100 ∧ c % 5 = 0 ∧ ∀ x : ℕ, x ≥ 10 ∧ x < 100 ∧ x % 5 = 0 → c ≤ x) ∧
  (d ≥ 100 ∧ d < 1000 ∧ d % 7 = 0 ∧ ∀ y : ℕ, y ≥ 100 ∧ y < 1000 ∧ y % 7 = 0 → d ≤ y) ∧
  c + d = 115 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiples_sum_l1089_108985


namespace NUMINAMATH_CALUDE_circle_intersection_equality_l1089_108984

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary relations and functions
variable (on_circle : Point → Circle → Prop)
variable (center : Circle → Point)
variable (intersect : Circle → Circle → Point × Point)
variable (line_intersect : Point → Point → Circle → Point)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem circle_intersection_equality 
  (circle1 circle2 : Circle) 
  (O P Q C A B : Point) :
  on_circle O circle1 ∧ 
  center circle2 = O ∧
  intersect circle1 circle2 = (P, Q) ∧
  on_circle C circle1 ∧
  line_intersect C P circle2 = A ∧
  line_intersect C Q circle2 = B →
  distance A B = distance P Q :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_equality_l1089_108984


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1089_108935

theorem min_value_of_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  let A : ℝ × ℝ := (-4, 0)
  let B : ℝ × ℝ := (-1, 0)
  let P : ℝ × ℝ := (a, b)
  (‖P - A‖ = 2 * ‖P - B‖) →
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 
    let Q : ℝ × ℝ := (x, y)
    (‖Q - A‖ = 2 * ‖Q - B‖) → 
    (4 / a^2 + 1 / b^2 ≤ 4 / x^2 + 1 / y^2)) →
  4 / a^2 + 1 / b^2 = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1089_108935


namespace NUMINAMATH_CALUDE_trailing_zeroes_sum_factorials_l1089_108917

def trailing_zeroes (n : ℕ) : ℕ := sorry

def factorial (n : ℕ) : ℕ := sorry

theorem trailing_zeroes_sum_factorials :
  trailing_zeroes (factorial 60 + factorial 120) = trailing_zeroes (factorial 60) :=
sorry

end NUMINAMATH_CALUDE_trailing_zeroes_sum_factorials_l1089_108917


namespace NUMINAMATH_CALUDE_star_seven_three_l1089_108981

-- Define the ⋆ operation
def star (a b : ℤ) : ℤ := 4*a + 3*b - a*b

-- State the theorem
theorem star_seven_three : star 7 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_star_seven_three_l1089_108981


namespace NUMINAMATH_CALUDE_alexander_pencils_per_picture_l1089_108978

theorem alexander_pencils_per_picture
  (first_exhibition_pictures : ℕ)
  (new_galleries : ℕ)
  (pictures_per_new_gallery : ℕ)
  (signing_pencils_per_exhibition : ℕ)
  (total_pencils_used : ℕ)
  (h1 : first_exhibition_pictures = 9)
  (h2 : new_galleries = 5)
  (h3 : pictures_per_new_gallery = 2)
  (h4 : signing_pencils_per_exhibition = 2)
  (h5 : total_pencils_used = 88) :
  (total_pencils_used - (signing_pencils_per_exhibition * (new_galleries + 1))) /
  (first_exhibition_pictures + new_galleries * pictures_per_new_gallery) = 4 := by
sorry

end NUMINAMATH_CALUDE_alexander_pencils_per_picture_l1089_108978


namespace NUMINAMATH_CALUDE_line_passes_through_circle_center_l1089_108932

-- Define the line equation
def line_equation (x y a : ℝ) : Prop := 3 * x + y + a = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y = 0

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem line_passes_through_circle_center (a : ℝ) :
  line_equation (circle_center.1) (circle_center.2) a →
  circle_equation (circle_center.1) (circle_center.2) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_circle_center_l1089_108932


namespace NUMINAMATH_CALUDE_fourteenSidedFigure_area_l1089_108989

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A polygon defined by a list of vertices -/
def Polygon := List Point

/-- The fourteen-sided figure described in the problem -/
def fourteenSidedFigure : Polygon := [
  ⟨1, 2⟩, ⟨2, 3⟩, ⟨4, 3⟩, ⟨5, 4⟩, ⟨5, 6⟩, ⟨6, 7⟩, ⟨7, 6⟩, ⟨7, 4⟩,
  ⟨6, 3⟩, ⟨4, 3⟩, ⟨3, 2⟩, ⟨3, 1⟩, ⟨2, 1⟩, ⟨1, 2⟩
]

/-- Calculate the area of a polygon -/
def calculateArea (p : Polygon) : ℝ :=
  sorry -- Actual implementation would go here

/-- Theorem: The area of the fourteen-sided figure is 14 cm² -/
theorem fourteenSidedFigure_area :
  calculateArea fourteenSidedFigure = 14 := by
  sorry -- Proof would go here

end NUMINAMATH_CALUDE_fourteenSidedFigure_area_l1089_108989


namespace NUMINAMATH_CALUDE_rollercoaster_time_interval_l1089_108923

theorem rollercoaster_time_interval
  (total_students : ℕ)
  (total_time : ℕ)
  (group_size : ℕ)
  (h1 : total_students = 21)
  (h2 : total_time = 15)
  (h3 : group_size = 7)
  : (total_time / (total_students / group_size) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rollercoaster_time_interval_l1089_108923


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1089_108906

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + (initial * percentage / 100) :=
by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 :=
by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1089_108906


namespace NUMINAMATH_CALUDE_correct_percentage_calculation_l1089_108919

/-- Calculates the overall percentage of correct answers across multiple tests -/
def overallPercentage (testSizes : List Nat) (scores : List Rat) : Rat :=
  sorry

/-- Rounds a rational number to the nearest whole number -/
def roundToNearest (x : Rat) : Nat :=
  sorry

theorem correct_percentage_calculation :
  let testSizes : List Nat := [40, 30, 20]
  let scores : List Rat := [65/100, 85/100, 75/100]
  roundToNearest (overallPercentage testSizes scores * 100) = 74 :=
by sorry

end NUMINAMATH_CALUDE_correct_percentage_calculation_l1089_108919


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1089_108918

/-- Given an arithmetic sequence with the first four terms as specified,
    prove that the fifth term has the given form. -/
theorem arithmetic_sequence_fifth_term
  (x y : ℝ)
  (seq : ℕ → ℝ)
  (h1 : seq 0 = x + y^2)
  (h2 : seq 1 = x + 2*y)
  (h3 : seq 2 = x*y^2)
  (h4 : seq 3 = x/y^2)
  (h_arithmetic : ∀ n : ℕ, seq (n + 1) - seq n = seq 1 - seq 0) :
  seq 4 = (y^6 - 2*y^5 + 4*y) / (y^4 + y^2) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1089_108918


namespace NUMINAMATH_CALUDE_johnny_money_left_l1089_108937

def savings_september : ℝ := 30
def savings_october : ℝ := 49
def savings_november : ℝ := 46
def savings_december : ℝ := 55
def january_savings_increase : ℝ := 0.15
def video_game_cost : ℝ := 58
def book_cost : ℝ := 25
def birthday_present_cost : ℝ := 40

def total_savings : ℝ :=
  savings_september + savings_october + savings_november + savings_december +
  (savings_december * (1 + january_savings_increase))

def total_expenses : ℝ :=
  video_game_cost + book_cost + birthday_present_cost

theorem johnny_money_left :
  total_savings - total_expenses = 120.25 := by
  sorry

end NUMINAMATH_CALUDE_johnny_money_left_l1089_108937


namespace NUMINAMATH_CALUDE_number_difference_l1089_108996

theorem number_difference (a b : ℕ) (h1 : a + b = 72) (h2 : a = 30) (h3 : b = 42) :
  b - a = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l1089_108996


namespace NUMINAMATH_CALUDE_people_in_room_l1089_108942

theorem people_in_room (total_chairs : ℕ) (seated_people : ℕ) (total_people : ℕ) : 
  (3 : ℚ) / 5 * total_people = seated_people →
  (4 : ℚ) / 5 * total_chairs = seated_people →
  total_chairs - seated_people = 5 →
  total_people = 33 := by
sorry

end NUMINAMATH_CALUDE_people_in_room_l1089_108942


namespace NUMINAMATH_CALUDE_water_formed_hcl_nahco3_l1089_108952

/-- Represents a chemical compound in a reaction -/
structure Compound where
  name : String
  moles : ℚ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Compound
  products : List Compound

/-- The balanced chemical equation for the reaction of HCl and NaHCO₃ -/
def hcl_nahco3_reaction : Reaction :=
  { reactants := [
      { name := "HCl", moles := 1 },
      { name := "NaHCO₃", moles := 1 }
    ],
    products := [
      { name := "NaCl", moles := 1 },
      { name := "CO₂", moles := 1 },
      { name := "H₂O", moles := 1 }
    ]
  }

/-- Calculate the amount of a specific product formed in a reaction -/
def amount_formed (reaction : Reaction) (product_name : String) (limiting_reagent_moles : ℚ) : ℚ :=
  let product := reaction.products.find? (fun c => c.name = product_name)
  match product with
  | some p => p.moles * limiting_reagent_moles
  | none => 0

/-- Theorem: The amount of water formed when 2 moles of HCl react with 2 moles of NaHCO₃ is 2 moles -/
theorem water_formed_hcl_nahco3 :
  amount_formed hcl_nahco3_reaction "H₂O" 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_formed_hcl_nahco3_l1089_108952


namespace NUMINAMATH_CALUDE_unique_solution_l1089_108944

theorem unique_solution : ∃! (x y z : ℕ), 
  2 ≤ x ∧ x ≤ y ∧ y ≤ z ∧
  (x * y) % z = 1 ∧
  (x * z) % y = 1 ∧
  (y * z) % x = 1 ∧
  x = 2 ∧ y = 3 ∧ z = 5 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l1089_108944


namespace NUMINAMATH_CALUDE_balls_without_holes_l1089_108991

theorem balls_without_holes 
  (total_soccer : ℕ) 
  (total_basketball : ℕ) 
  (soccer_with_holes : ℕ) 
  (basketball_with_holes : ℕ) 
  (h1 : total_soccer = 40) 
  (h2 : total_basketball = 15) 
  (h3 : soccer_with_holes = 30) 
  (h4 : basketball_with_holes = 7) : 
  (total_soccer - soccer_with_holes) + (total_basketball - basketball_with_holes) = 18 :=
by
  sorry


end NUMINAMATH_CALUDE_balls_without_holes_l1089_108991


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1089_108967

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) :
  (1 / x + 1 / y) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1089_108967


namespace NUMINAMATH_CALUDE_mod_power_thirteen_six_eleven_l1089_108951

theorem mod_power_thirteen_six_eleven : 13^6 % 11 = 9 := by
  sorry

end NUMINAMATH_CALUDE_mod_power_thirteen_six_eleven_l1089_108951


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l1089_108961

/-- Given a line y = kx + b tangent to the curve y = x³ + ax + 1 at (2,3), prove b = -15 -/
theorem tangent_line_b_value (k a b : ℝ) : 
  (3 = 2 * k + b) →  -- Line equation at (2,3)
  (3 = 2^3 + 2*a + 1) →  -- Curve equation at (2,3)
  (k = 3 * 2^2 + a) →  -- Slope equality condition for tangency
  (b = -15) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l1089_108961


namespace NUMINAMATH_CALUDE_grapes_distribution_l1089_108931

theorem grapes_distribution (total_grapes : ℕ) (num_kids : ℕ) 
  (h1 : total_grapes = 50)
  (h2 : num_kids = 7) :
  total_grapes % num_kids = 1 := by
  sorry

end NUMINAMATH_CALUDE_grapes_distribution_l1089_108931


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1089_108973

theorem quadratic_inequality_solution (n : ℕ) (x : ℝ) :
  (∀ n : ℕ, n^2 * x^2 - (2*n^2 + n) * x + n^2 + n - 6 ≤ 0) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1089_108973


namespace NUMINAMATH_CALUDE_greatest_possible_average_speed_l1089_108974

/-- Represents a palindromic number --/
def IsPalindrome (n : ℕ) : Prop := sorry

/-- Calculates the next palindrome after a given number --/
def NextPalindrome (n : ℕ) : ℕ := sorry

/-- Represents the maximum speed limit in miles per hour --/
def MaxSpeedLimit : ℕ := 80

/-- Represents the trip duration in hours --/
def TripDuration : ℕ := 4

/-- Represents the initial odometer reading --/
def InitialReading : ℕ := 12321

theorem greatest_possible_average_speed :
  ∃ (finalReading : ℕ),
    IsPalindrome InitialReading ∧
    IsPalindrome finalReading ∧
    finalReading > InitialReading ∧
    finalReading ≤ InitialReading + MaxSpeedLimit * TripDuration ∧
    (∀ (n : ℕ),
      IsPalindrome n ∧
      n > InitialReading ∧
      n ≤ InitialReading + MaxSpeedLimit * TripDuration →
      n ≤ finalReading) ∧
    (finalReading - InitialReading) / TripDuration = 75 :=
by sorry

end NUMINAMATH_CALUDE_greatest_possible_average_speed_l1089_108974


namespace NUMINAMATH_CALUDE_bicycle_store_promotion_correct_l1089_108941

/-- Represents the promotion rules and sales data for a bicycle store. -/
structure BicycleStore where
  single_clamps : ℕ  -- Number of clamps given for a single bicycle purchase
  single_helmet : ℕ  -- Number of helmets given for a single bicycle purchase
  discount_rate : ℚ  -- Discount rate on the 3rd bicycle for a 3-bicycle purchase
  morning_single : ℕ  -- Number of single bicycle purchases in the morning
  morning_triple : ℕ  -- Number of 3-bicycle purchases in the morning
  afternoon_single : ℕ  -- Number of single bicycle purchases in the afternoon
  afternoon_triple : ℕ  -- Number of 3-bicycle purchases in the afternoon

/-- Calculates the total number of bike clamps given away. -/
def total_clamps (store : BicycleStore) : ℕ :=
  (store.morning_single + store.afternoon_single) * store.single_clamps +
  (store.morning_triple + store.afternoon_triple) * store.single_clamps

/-- Calculates the total number of helmets given away. -/
def total_helmets (store : BicycleStore) : ℕ :=
  (store.morning_single + store.afternoon_single) * store.single_helmet +
  (store.morning_triple + store.afternoon_triple) * store.single_helmet

/-- Calculates the overall discount value in terms of full-price bicycles. -/
def discount_value (store : BicycleStore) : ℚ :=
  (store.morning_triple + store.afternoon_triple) * store.discount_rate

/-- Theorem stating the correctness of the calculations based on the given data. -/
theorem bicycle_store_promotion_correct (store : BicycleStore) 
  (h1 : store.single_clamps = 2)
  (h2 : store.single_helmet = 1)
  (h3 : store.discount_rate = 1/5)
  (h4 : store.morning_single = 12)
  (h5 : store.morning_triple = 7)
  (h6 : store.afternoon_single = 24)
  (h7 : store.afternoon_triple = 3) :
  total_clamps store = 92 ∧ 
  total_helmets store = 46 ∧ 
  discount_value store = 2 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_store_promotion_correct_l1089_108941


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l1089_108986

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - 2)^2 = 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem circle_tangent_to_x_axis :
  -- The circle equation represents a circle with the given center
  (∀ x y : ℝ, circle_equation x y ↔ ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 4)) ∧
  -- The circle is tangent to the x-axis
  (∃ x : ℝ, circle_equation x 0 ∧ ∀ y : ℝ, y ≠ 0 → ¬ circle_equation x y) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l1089_108986


namespace NUMINAMATH_CALUDE_average_weight_increase_problem_solution_l1089_108960

/-- The increase in average weight when replacing a person in a group -/
theorem average_weight_increase (n : ℕ) (old_weight new_weight : ℝ) : 
  n > 0 → (new_weight - old_weight) / n = (new_weight - old_weight) / n := by
  sorry

/-- The specific case of the problem -/
theorem problem_solution : 
  let n : ℕ := 10
  let old_weight : ℝ := 65
  let new_weight : ℝ := 137
  (new_weight - old_weight) / n = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_increase_problem_solution_l1089_108960


namespace NUMINAMATH_CALUDE_final_position_is_37_steps_behind_l1089_108940

/-- Represents the walking challenge rules -/
def walkingChallenge (n : ℕ) : ℤ :=
  if n = 1 then 0
  else if Nat.Prime n then 2
  else -3

/-- The final position after completing all 30 moves -/
def finalPosition : ℤ :=
  -(Finset.sum (Finset.range 30) (fun i => walkingChallenge (i + 1)))

/-- Theorem stating the final position is 37 steps behind the starting point -/
theorem final_position_is_37_steps_behind :
  finalPosition = -37 := by sorry

end NUMINAMATH_CALUDE_final_position_is_37_steps_behind_l1089_108940


namespace NUMINAMATH_CALUDE_window_width_is_ten_l1089_108921

-- Define the window parameters
def window_length : ℝ := 6
def window_area : ℝ := 60

-- Theorem statement
theorem window_width_is_ten :
  ∃ w : ℝ, w * window_length = window_area ∧ w = 10 :=
by sorry

end NUMINAMATH_CALUDE_window_width_is_ten_l1089_108921


namespace NUMINAMATH_CALUDE_popsicle_sticks_per_boy_l1089_108956

theorem popsicle_sticks_per_boy (num_boys num_girls : ℕ) (sticks_per_girl : ℕ) (diff : ℕ) :
  num_boys = 10 →
  num_girls = 12 →
  sticks_per_girl = 12 →
  num_girls * sticks_per_girl + diff = num_boys * (num_girls * sticks_per_girl + diff) / num_boys →
  diff = 6 →
  (num_girls * sticks_per_girl + diff) / num_boys = 15 :=
by sorry

end NUMINAMATH_CALUDE_popsicle_sticks_per_boy_l1089_108956


namespace NUMINAMATH_CALUDE_meeting_time_is_lcm_l1089_108962

/-- The lap times of the four friends in minutes -/
def lap_times : List Nat := [5, 8, 9, 12]

/-- The time in minutes after 10:00 AM when all friends meet -/
def meeting_time : Nat := 360

/-- Theorem stating that the meeting time is the LCM of the lap times -/
theorem meeting_time_is_lcm : 
  meeting_time = Nat.lcm (Nat.lcm (Nat.lcm (lap_times.get! 0) (lap_times.get! 1)) (lap_times.get! 2)) (lap_times.get! 3) :=
by sorry

end NUMINAMATH_CALUDE_meeting_time_is_lcm_l1089_108962


namespace NUMINAMATH_CALUDE_expression_simplification_l1089_108976

theorem expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 * (a^2)^(1/3) * b^(1/2) * (-6 * a^(1/3) * b^(1/3))^2) / (-3 * (a*b^5)^(1/6)) = -24 * a^(7/6) * b^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1089_108976


namespace NUMINAMATH_CALUDE_function_relationship_l1089_108955

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y)
variable (h2 : ∀ x, f (x + 2) = f (-x + 2))

-- State the theorem
theorem function_relationship : f 2.5 > f 1 ∧ f 1 > f 3.5 :=
sorry

end NUMINAMATH_CALUDE_function_relationship_l1089_108955


namespace NUMINAMATH_CALUDE_snack_distribution_probability_l1089_108950

/-- The number of students and snack types -/
def n : ℕ := 4

/-- The total number of snacks -/
def total_snacks : ℕ := n * n

/-- The number of ways to distribute snacks to one student -/
def ways_per_student (k : ℕ) : ℕ := n^n

/-- The number of ways to choose snacks for one student from remaining snacks -/
def choose_from_remaining (k : ℕ) : ℕ := Nat.choose (total_snacks - (k - 1) * n) n

/-- The probability of correct distribution for the k-th student -/
def prob_for_student (k : ℕ) : ℚ := ways_per_student k / choose_from_remaining k

/-- The probability that each student gets one of each type of snack -/
def prob_correct_distribution : ℚ :=
  prob_for_student 1 * prob_for_student 2 * prob_for_student 3

theorem snack_distribution_probability :
  prob_correct_distribution = 64 / 1225 :=
sorry

end NUMINAMATH_CALUDE_snack_distribution_probability_l1089_108950


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1089_108912

/-- The x-intercept of the line 4x + 7y = 28 is the point (7,0) -/
theorem x_intercept_of_line (x y : ℚ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1089_108912


namespace NUMINAMATH_CALUDE_circumcircle_area_l1089_108998

theorem circumcircle_area (a b c : ℝ) (A B C : ℝ) (R : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2 ∧
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  c / Real.sin C = 2 * R ∧
  A = 5 * π / 12 ∧
  B = π / 4 ∧
  c = 4 →
  π * R^2 = 16 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_circumcircle_area_l1089_108998
