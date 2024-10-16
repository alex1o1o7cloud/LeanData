import Mathlib

namespace NUMINAMATH_CALUDE_equal_probability_sums_l780_78088

def num_dice : ℕ := 8
def min_face_value : ℕ := 1
def max_face_value : ℕ := 6

def min_sum : ℕ := num_dice * min_face_value
def max_sum : ℕ := num_dice * max_face_value

def symmetric_sum (s : ℕ) : ℕ := 2 * ((min_sum + max_sum) / 2) - s

theorem equal_probability_sums :
  symmetric_sum 11 = 45 :=
sorry

end NUMINAMATH_CALUDE_equal_probability_sums_l780_78088


namespace NUMINAMATH_CALUDE_pet_food_price_l780_78003

theorem pet_food_price (regular_discount_min regular_discount_max additional_discount lowest_price : Real) 
  (h1 : 0.1 ≤ regular_discount_min ∧ regular_discount_min ≤ regular_discount_max ∧ regular_discount_max ≤ 0.3)
  (h2 : additional_discount = 0.2)
  (h3 : lowest_price = 25.2)
  : ∃ (original_price : Real),
    original_price * (1 - regular_discount_max) * (1 - additional_discount) = lowest_price ∧
    original_price = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_pet_food_price_l780_78003


namespace NUMINAMATH_CALUDE_job_total_amount_l780_78038

/-- Calculates the total amount earned for a job given the time taken by two workers and one worker's share. -/
theorem job_total_amount 
  (rahul_days : ℚ) 
  (rajesh_days : ℚ) 
  (rahul_share : ℚ) : 
  rahul_days = 3 → 
  rajesh_days = 2 → 
  rahul_share = 142 → 
  ∃ (total_amount : ℚ), total_amount = 355 := by
  sorry

#check job_total_amount

end NUMINAMATH_CALUDE_job_total_amount_l780_78038


namespace NUMINAMATH_CALUDE_a_value_equation_solution_l780_78016

-- Define the positive number whose square root is both a+6 and 2a-9
def positive_number (a : ℝ) : Prop := ∃ n : ℝ, n > 0 ∧ (a + 6 = Real.sqrt n) ∧ (2*a - 9 = Real.sqrt n)

-- Theorem 1: Prove that a = 15
theorem a_value (a : ℝ) (h : positive_number a) : a = 15 := by sorry

-- Theorem 2: Prove that the solution to ax³-64=0 is x = 4 when a = 15
theorem equation_solution (x : ℝ) : 15 * x^3 - 64 = 0 ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_a_value_equation_solution_l780_78016


namespace NUMINAMATH_CALUDE_cube_difference_factorization_l780_78009

theorem cube_difference_factorization (a b : ℝ) :
  a^3 - 8*b^3 = (a - 2*b)*(a^2 + 2*a*b + 4*b^2) := by
  sorry

end NUMINAMATH_CALUDE_cube_difference_factorization_l780_78009


namespace NUMINAMATH_CALUDE_zeljko_distance_l780_78045

/-- Calculates the total distance travelled given two segments of a journey -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Proves that Zeljko's total distance travelled is 20 km -/
theorem zeljko_distance : 
  let speed1 : ℝ := 30  -- km/h
  let time1  : ℝ := 20 / 60  -- 20 minutes in hours
  let speed2 : ℝ := 20  -- km/h
  let time2  : ℝ := 30 / 60  -- 30 minutes in hours
  total_distance speed1 time1 speed2 time2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_zeljko_distance_l780_78045


namespace NUMINAMATH_CALUDE_other_person_age_l780_78002

/-- Given two people where one (Marco) is 1 year older than twice the age of the other,
    and the sum of their ages is 37, prove that the younger person is 12 years old. -/
theorem other_person_age (x : ℕ) : x + (2 * x + 1) = 37 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_other_person_age_l780_78002


namespace NUMINAMATH_CALUDE_min_value_sum_of_products_l780_78035

theorem min_value_sum_of_products (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  (x*y/z + y*z/x + z*x/y) ≥ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_of_products_l780_78035


namespace NUMINAMATH_CALUDE_bicycle_final_price_l780_78025

/-- The final selling price of a bicycle given initial cost and profit margins -/
theorem bicycle_final_price (a_cost : ℝ) (a_profit_percent : ℝ) (b_profit_percent : ℝ) : 
  a_cost = 112.5 → 
  a_profit_percent = 60 → 
  b_profit_percent = 25 → 
  a_cost * (1 + a_profit_percent / 100) * (1 + b_profit_percent / 100) = 225 := by
sorry

end NUMINAMATH_CALUDE_bicycle_final_price_l780_78025


namespace NUMINAMATH_CALUDE_mean_temperature_l780_78021

theorem mean_temperature (temperatures : List ℝ) : 
  temperatures = [75, 77, 76, 80, 82] → 
  (temperatures.sum / temperatures.length : ℝ) = 78 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_l780_78021


namespace NUMINAMATH_CALUDE_ball_attendance_l780_78010

theorem ball_attendance :
  ∀ n m : ℕ,
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  (∃ k : ℕ, n = 20 * k ∧ m = 21 * k) →
  n + m = 41 :=
λ n m h1 h2 h3 =>
  sorry

end NUMINAMATH_CALUDE_ball_attendance_l780_78010


namespace NUMINAMATH_CALUDE_cube_third_yellow_faces_l780_78049

/-- Represents a cube with side length n -/
structure Cube where
  n : ℕ

/-- Total number of faces of unit cubes after division -/
def totalFaces (c : Cube) : ℕ := 6 * c.n^3

/-- Number of yellow faces after division -/
def yellowFaces (c : Cube) : ℕ := 6 * c.n^2

/-- Condition for exactly one-third of faces being yellow -/
def oneThirdYellow (c : Cube) : Prop :=
  3 * yellowFaces c = totalFaces c

/-- Theorem stating that n = 3 satisfies the condition -/
theorem cube_third_yellow_faces :
  ∃ (c : Cube), c.n = 3 ∧ oneThirdYellow c :=
sorry

end NUMINAMATH_CALUDE_cube_third_yellow_faces_l780_78049


namespace NUMINAMATH_CALUDE_inequality_preservation_l780_78093

theorem inequality_preservation (a b : ℝ) (h : a > b) :
  ∀ x : ℝ, a * (2 : ℝ)^x > b * (2 : ℝ)^x := by
sorry


end NUMINAMATH_CALUDE_inequality_preservation_l780_78093


namespace NUMINAMATH_CALUDE_not_divisible_by_1955_l780_78087

theorem not_divisible_by_1955 : ∀ n : ℕ, ¬(1955 ∣ (n^2 + n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_1955_l780_78087


namespace NUMINAMATH_CALUDE_plates_needed_is_38_l780_78052

/-- The number of plates needed for a week given the specified eating patterns -/
def plates_needed : ℕ :=
  let days_with_son := 3
  let days_with_parents := 7 - days_with_son
  let people_with_son := 2
  let people_with_parents := 4
  let plates_per_person_with_son := 1
  let plates_per_person_with_parents := 2
  
  (days_with_son * people_with_son * plates_per_person_with_son) +
  (days_with_parents * people_with_parents * plates_per_person_with_parents)

theorem plates_needed_is_38 : plates_needed = 38 := by
  sorry

end NUMINAMATH_CALUDE_plates_needed_is_38_l780_78052


namespace NUMINAMATH_CALUDE_cookie_sharing_l780_78050

theorem cookie_sharing (total_cookies : ℕ) (cookies_per_person : ℕ) (h1 : total_cookies = 24) (h2 : cookies_per_person = 4) :
  total_cookies / cookies_per_person = 6 :=
by sorry

end NUMINAMATH_CALUDE_cookie_sharing_l780_78050


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l780_78012

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote: y = x + 2 -/
  asymptote1 : ℝ → ℝ
  /-- Second asymptote: y = 4 - x -/
  asymptote2 : ℝ → ℝ
  /-- The hyperbola passes through the point (4, 4) -/
  passes_through : ℝ × ℝ
  /-- Conditions for the asymptotes -/
  h_asymptote1 : ∀ x, asymptote1 x = x + 2
  h_asymptote2 : ∀ x, asymptote2 x = 4 - x
  h_passes_through : passes_through = (4, 4)

/-- The distance between the foci of the hyperbola -/
def foci_distance (h : Hyperbola) : ℝ := 8

/-- Theorem stating that the distance between the foci of the given hyperbola is 8 -/
theorem hyperbola_foci_distance (h : Hyperbola) :
  foci_distance h = 8 := by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l780_78012


namespace NUMINAMATH_CALUDE_drama_club_ratio_l780_78046

theorem drama_club_ratio (girls boys : ℝ) (h : boys = 0.8 * girls) :
  girls = 1.25 * boys := by
  sorry

end NUMINAMATH_CALUDE_drama_club_ratio_l780_78046


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l780_78047

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (a b : Line) (α : Plane) :
  perp a α → perp b α → parallel a b :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l780_78047


namespace NUMINAMATH_CALUDE_debate_team_group_size_l780_78074

/-- Proves that the number of students in each group is 9,
    given the number of boys, girls, and total groups. -/
theorem debate_team_group_size
  (boys : ℕ)
  (girls : ℕ)
  (total_groups : ℕ)
  (h1 : boys = 26)
  (h2 : girls = 46)
  (h3 : total_groups = 8) :
  (boys + girls) / total_groups = 9 :=
by sorry

end NUMINAMATH_CALUDE_debate_team_group_size_l780_78074


namespace NUMINAMATH_CALUDE_robbery_participants_l780_78075

-- Define the suspects
variable (Alexey Boris Veniamin Grigory : Prop)

-- Define the conditions
axiom condition1 : ¬Grigory → (Boris ∧ ¬Alexey)
axiom condition2 : Veniamin → (¬Alexey ∧ ¬Boris)
axiom condition3 : Grigory → Boris
axiom condition4 : Boris → (Alexey ∨ Veniamin)

-- Theorem to prove
theorem robbery_participants :
  Alexey ∧ Boris ∧ Grigory ∧ ¬Veniamin :=
sorry

end NUMINAMATH_CALUDE_robbery_participants_l780_78075


namespace NUMINAMATH_CALUDE_quadratic_factorization_l780_78080

theorem quadratic_factorization (x : ℝ) : x^2 - x - 42 = (x + 6) * (x - 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l780_78080


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l780_78061

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}
def Q : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Define the union of P and Q
def PUnionQ : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem union_of_P_and_Q : P ∪ Q = PUnionQ := by
  sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l780_78061


namespace NUMINAMATH_CALUDE_zlatoust_miass_distance_l780_78098

/-- The distance between Zlatoust and Miass -/
def distance : ℝ := sorry

/-- The speed of GAZ truck -/
def speed_gaz : ℝ := sorry

/-- The speed of MAZ truck -/
def speed_maz : ℝ := sorry

/-- The speed of KAMAZ truck -/
def speed_kamaz : ℝ := sorry

theorem zlatoust_miass_distance :
  (distance + 18) / speed_kamaz = (distance - 18) / speed_maz ∧
  (distance + 25) / speed_kamaz = (distance - 25) / speed_gaz ∧
  (distance + 8) / speed_maz = (distance - 8) / speed_gaz →
  distance = 60 := by sorry

end NUMINAMATH_CALUDE_zlatoust_miass_distance_l780_78098


namespace NUMINAMATH_CALUDE_invalid_altitudes_l780_78079

/-- A triple of positive real numbers represents valid altitudes of a triangle if and only if
    the sum of the reciprocals of any two is greater than the reciprocal of the third. -/
def ValidAltitudes (h₁ h₂ h₃ : ℝ) : Prop :=
  h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧
  1/h₁ + 1/h₂ > 1/h₃ ∧
  1/h₂ + 1/h₃ > 1/h₁ ∧
  1/h₃ + 1/h₁ > 1/h₂

/-- The triple (5, 12, 13) cannot be the lengths of the three altitudes of a triangle. -/
theorem invalid_altitudes : ¬ ValidAltitudes 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_invalid_altitudes_l780_78079


namespace NUMINAMATH_CALUDE_units_digit_of_7_pow_6_squared_l780_78029

def units_digit_pattern : List Nat := [7, 9, 3, 1]

def units_digit (n : Nat) : Nat :=
  n % 10

theorem units_digit_of_7_pow_6_squared : 
  units_digit (7^(6^2)) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_pow_6_squared_l780_78029


namespace NUMINAMATH_CALUDE_f_range_implies_a_range_l780_78065

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |4*x + 1| - |4*x + a|

-- State the theorem
theorem f_range_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, f a x ≤ -5) → a ∈ Set.Iic (-4) ∪ Set.Ici 6 :=
by sorry

end NUMINAMATH_CALUDE_f_range_implies_a_range_l780_78065


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l780_78069

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℚ) : Prop :=
  ∃ m : ℚ, m * m = n

-- Define a function to check if a quadratic radical is in its simplest form
def is_simplest_quadratic_radical (n : ℚ) : Prop :=
  n > 0 ∧ ¬is_perfect_square n ∧ (∀ m : ℕ, m > 1 → ¬is_perfect_square (n / (m * m : ℚ)))

-- Theorem statement
theorem simplest_quadratic_radical :
  is_simplest_quadratic_radical 7 ∧
  ¬is_simplest_quadratic_radical 9 ∧
  ¬is_simplest_quadratic_radical 20 ∧
  ¬is_simplest_quadratic_radical (1/3) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l780_78069


namespace NUMINAMATH_CALUDE_sin_2x_minus_pi_3_zeros_min_distance_l780_78062

open Real

theorem sin_2x_minus_pi_3_zeros_min_distance (f : ℝ → ℝ) (h : ∀ x, f x = sin (2 * x - π / 3)) :
  ∀ a b : ℝ, a ≠ b → f a = 0 → f b = 0 → |a - b| ≥ π / 2 ∧ ∃ c d : ℝ, c ≠ d ∧ f c = 0 ∧ f d = 0 ∧ |c - d| = π / 2 :=
sorry

end NUMINAMATH_CALUDE_sin_2x_minus_pi_3_zeros_min_distance_l780_78062


namespace NUMINAMATH_CALUDE_circle_equal_circumference_area_l780_78063

theorem circle_equal_circumference_area (r : ℝ) : 
  2 * Real.pi * r = Real.pi * r^2 → 2 * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equal_circumference_area_l780_78063


namespace NUMINAMATH_CALUDE_parabola_c_value_l780_78097

/-- A parabola with equation x = ay^2 + by + c, vertex at (4, 3), and passing through (2, 5) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 4
  vertex_y : ℝ := 3
  point_x : ℝ := 2
  point_y : ℝ := 5
  eq_vertex : 4 = a * 3^2 + b * 3 + c
  eq_point : 2 = a * 5^2 + b * 5 + c

/-- The value of c for the given parabola is -1/2 -/
theorem parabola_c_value (p : Parabola) : p.c = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l780_78097


namespace NUMINAMATH_CALUDE_A_is_integer_l780_78023

def n : ℤ := 8795685

def A : ℚ :=
  (((n + 4) * (n + 3) * (n + 2) * (n + 1)) - ((n - 1) * (n - 2) * (n - 3) * (n - 4))) /
  ((n + 3)^2 + (n + 1)^2 + (n - 1)^2 + (n - 3)^2)

theorem A_is_integer : ∃ (k : ℤ), A = k := by
  sorry

end NUMINAMATH_CALUDE_A_is_integer_l780_78023


namespace NUMINAMATH_CALUDE_binary_1011001_equals_base5_324_l780_78070

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem binary_1011001_equals_base5_324 : 
  decimal_to_base5 (binary_to_decimal [true, false, false, true, true, false, true]) = [3, 2, 4] := by
  sorry

end NUMINAMATH_CALUDE_binary_1011001_equals_base5_324_l780_78070


namespace NUMINAMATH_CALUDE_tamika_always_greater_l780_78076

def tamika_set : Set ℕ := {6, 7, 8}
def carlos_set : Set ℕ := {2, 4, 5}

def tamika_product (a b : ℕ) : Prop := a ∈ tamika_set ∧ b ∈ tamika_set ∧ a ≠ b
def carlos_product (c d : ℕ) : Prop := c ∈ carlos_set ∧ d ∈ carlos_set ∧ c ≠ d

theorem tamika_always_greater :
  ∀ (a b c d : ℕ), tamika_product a b → carlos_product c d →
    a * b > c * d :=
sorry

end NUMINAMATH_CALUDE_tamika_always_greater_l780_78076


namespace NUMINAMATH_CALUDE_shooting_stars_count_difference_l780_78043

theorem shooting_stars_count_difference (bridget_count reginald_count sam_count : ℕ) : 
  bridget_count = 14 →
  reginald_count = bridget_count - 2 →
  sam_count > reginald_count →
  sam_count = (bridget_count + reginald_count + sam_count) / 3 + 2 →
  sam_count - reginald_count = 4 := by
  sorry

end NUMINAMATH_CALUDE_shooting_stars_count_difference_l780_78043


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l780_78089

/-- A geometric progression is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricProgression (a : ℝ) (r : ℝ) := fun (n : ℕ) => a * r ^ (n - 1)

theorem geometric_progression_first_term
  (a : ℝ) (r : ℝ) (h1 : r ≠ 0)
  (h2 : GeometricProgression a r 2 = 5)
  (h3 : GeometricProgression a r 3 = 1) :
  a = 25 := by
  sorry

#check geometric_progression_first_term

end NUMINAMATH_CALUDE_geometric_progression_first_term_l780_78089


namespace NUMINAMATH_CALUDE_complement_of_M_l780_78011

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x | x^2 - 4 ≤ 0}

-- State the theorem
theorem complement_of_M :
  Set.compl M = {x : ℝ | x < -2 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l780_78011


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l780_78040

theorem sqrt_equation_solution (x y : ℝ) : 
  Real.sqrt (10 + 3 * x - y) = 7 → y = 3 * x - 39 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l780_78040


namespace NUMINAMATH_CALUDE_integer_roots_of_polynomial_l780_78054

/-- Represents a polynomial of degree 4 with rational coefficients -/
structure Polynomial4 where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Checks if a given number is a root of the polynomial -/
def isRoot (p : Polynomial4) (x : ℝ) : Prop :=
  x^4 + p.a * x^2 + p.b * x + p.c = 0

theorem integer_roots_of_polynomial (p : Polynomial4) :
  isRoot p (2 - Real.sqrt 5) →
  (∃ (r₁ r₂ : ℤ), isRoot p (r₁ : ℝ) ∧ isRoot p (r₂ : ℝ)) →
  ∃ (r : ℤ), isRoot p (r : ℝ) ∧ r = -2 :=
sorry

end NUMINAMATH_CALUDE_integer_roots_of_polynomial_l780_78054


namespace NUMINAMATH_CALUDE_odd_m_triple_g_eq_5_l780_78094

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n - 4 else n / 3

theorem odd_m_triple_g_eq_5 (m : ℤ) (h1 : m % 2 = 1) (h2 : g (g (g m)) = 5) : m = 17 := by
  sorry

end NUMINAMATH_CALUDE_odd_m_triple_g_eq_5_l780_78094


namespace NUMINAMATH_CALUDE_field_length_calculation_l780_78055

theorem field_length_calculation (width length : ℝ) (pond_side : ℝ) : 
  length = 2 * width →
  pond_side = 8 →
  pond_side^2 = (1/8) * (length * width) →
  length = 32 := by
  sorry

end NUMINAMATH_CALUDE_field_length_calculation_l780_78055


namespace NUMINAMATH_CALUDE_combined_height_theorem_l780_78051

/-- The conversion factor from inches to centimeters -/
def inch_to_cm : ℝ := 2.54

/-- Maria's height in inches -/
def maria_height_inches : ℝ := 54

/-- Ben's height in inches -/
def ben_height_inches : ℝ := 72

/-- Combined height in centimeters -/
def combined_height_cm : ℝ := (maria_height_inches + ben_height_inches) * inch_to_cm

theorem combined_height_theorem :
  combined_height_cm = 320.04 := by sorry

end NUMINAMATH_CALUDE_combined_height_theorem_l780_78051


namespace NUMINAMATH_CALUDE_two_numbers_problem_l780_78072

theorem two_numbers_problem : ∃ (x y : ℕ), 
  x = y + 75 ∧ 
  x * y = (227 * y + 113) + 1000 ∧ 
  x > y ∧ 
  x = 234 ∧ 
  y = 159 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l780_78072


namespace NUMINAMATH_CALUDE_salad_dressing_composition_l780_78053

theorem salad_dressing_composition (vinegar_p : ℝ) (oil_p : ℝ) (vinegar_q : ℝ) (oil_q : ℝ) 
  (ratio_p : ℝ) (ratio_q : ℝ) (vinegar_new : ℝ) :
  vinegar_p = 0.3 →
  vinegar_p + oil_p = 1 →
  vinegar_q = 0.1 →
  oil_q = 0.9 →
  ratio_p = 0.1 →
  ratio_q = 0.9 →
  ratio_p + ratio_q = 1 →
  vinegar_new = 0.12 →
  ratio_p * vinegar_p + ratio_q * vinegar_q = vinegar_new →
  oil_p = 0.7 := by
sorry

end NUMINAMATH_CALUDE_salad_dressing_composition_l780_78053


namespace NUMINAMATH_CALUDE_like_terms_imply_expression_value_l780_78024

theorem like_terms_imply_expression_value :
  ∀ (a b : ℤ),
  (2 : ℤ) = 1 - a →
  (5 : ℤ) = 3 * b - 1 →
  5 * a * b^2 - (6 * a^2 * b - 3 * (a * b^2 + 2 * a^2 * b)) = -32 :=
by sorry

end NUMINAMATH_CALUDE_like_terms_imply_expression_value_l780_78024


namespace NUMINAMATH_CALUDE_polynomial_equality_l780_78068

theorem polynomial_equality : 2090^3 + 2089 * 2090^2 - 2089^2 * 2090 + 2089^3 = 4179 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l780_78068


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l780_78036

theorem quadratic_equation_roots (a b c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  let sum_roots := -b / a
  let prod_roots := c / a
  let new_sum := sum_roots + prod_roots
  let new_prod := sum_roots * prod_roots
  f 0 = 0 →
  (∃ x y : ℝ, x + y = new_sum ∧ x * y = new_prod) →
  ∃ k : ℝ, k ≠ 0 ∧ f = λ x => k * (x^2 - new_sum * x + new_prod) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l780_78036


namespace NUMINAMATH_CALUDE_parallel_lines_a_value_l780_78057

-- Define the lines l₁ and l₂
def l₁ (x y a : ℝ) : Prop := 3 * x + 2 * a * y - 5 = 0
def l₂ (x y a : ℝ) : Prop := (3 * a - 1) * x - a * y - 2 = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ x y, l₁ x y a ↔ l₂ x y a

-- Theorem statement
theorem parallel_lines_a_value (a : ℝ) :
  parallel a → (a = 0 ∨ a = -1/6) :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_a_value_l780_78057


namespace NUMINAMATH_CALUDE_number_of_students_in_class_l780_78006

/-- Proves that the number of students in a class is 23 given certain grade conditions --/
theorem number_of_students_in_class 
  (recorded_biology : ℝ) 
  (recorded_chemistry : ℝ)
  (actual_biology : ℝ) 
  (actual_chemistry : ℝ)
  (subject_weight : ℝ)
  (class_average_increase : ℝ)
  (initial_class_average : ℝ)
  (h1 : recorded_biology = 83)
  (h2 : recorded_chemistry = 85)
  (h3 : actual_biology = 70)
  (h4 : actual_chemistry = 75)
  (h5 : subject_weight = 0.5)
  (h6 : class_average_increase = 0.5)
  (h7 : initial_class_average = 80) :
  ∃ n : ℕ, n = 23 ∧ n * class_average_increase = (recorded_biology * subject_weight + recorded_chemistry * subject_weight) - (actual_biology * subject_weight + actual_chemistry * subject_weight) := by
  sorry

end NUMINAMATH_CALUDE_number_of_students_in_class_l780_78006


namespace NUMINAMATH_CALUDE_log_identity_l780_78056

theorem log_identity : (Real.log 2 / Real.log 10) ^ 2 + (Real.log 5 / Real.log 10) ^ 2 + 2 * (Real.log 2 / Real.log 10) * (Real.log 5 / Real.log 10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l780_78056


namespace NUMINAMATH_CALUDE_no_divisible_by_seven_l780_78014

/-- The sequence defined by a(n) = 9^n + 1 -/
def a (n : ℕ) : ℕ := 9^n + 1

/-- Theorem stating that none of the first 1200 terms of the sequence are divisible by 7 -/
theorem no_divisible_by_seven : ∀ n : ℕ, n ≤ 1200 → ¬(7 ∣ a n) := by sorry

end NUMINAMATH_CALUDE_no_divisible_by_seven_l780_78014


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l780_78086

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) :
  angle_sum = 720 → (n - 2) * 180 = angle_sum → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l780_78086


namespace NUMINAMATH_CALUDE_earnings_for_55_hours_l780_78085

/-- Calculates the earnings for a given number of hours based on the described pay rate pattern -/
def earnings (hours : ℕ) : ℕ :=
  let cycleEarnings := (List.range 10).map (· + 1) |> List.sum
  let completeCycles := hours / 10
  completeCycles * cycleEarnings

/-- Proves that working for 55 hours with the given pay rate results in earning $275 -/
theorem earnings_for_55_hours :
  earnings 55 = 275 := by
  sorry

end NUMINAMATH_CALUDE_earnings_for_55_hours_l780_78085


namespace NUMINAMATH_CALUDE_polynomial_factorization_l780_78015

theorem polynomial_factorization (a b c x y z : ℝ) : 
  (a^2*(b-c) + b^2*(c-a) + c^2*(a-b) = 0 ∨ 
   b^2*(c-a) + c^2*(a-b) + a^2*(b-c) = 0 ∨ 
   c^2*(a-b) + a^2*(b-c) + b^2*(c-a) = 0) ∧
  (a^2*(b-c) + b^2*(c-a) + c^2*(a-b) = -(a-b)*(b-c)*(c-a)) ∧
  ((x+y+z)^3 - x^3 - y^3 - z^3 = 3*(x+y)*(y+z)*(z+x)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l780_78015


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l780_78067

theorem quadratic_equation_solution :
  let equation := fun x : ℝ => 2 * x^2 + 6 * x - 1
  let solution1 := -3/2 + Real.sqrt 11 / 2
  let solution2 := -3/2 - Real.sqrt 11 / 2
  equation solution1 = 0 ∧ equation solution2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l780_78067


namespace NUMINAMATH_CALUDE_team_formation_ways_l780_78095

/-- Represents the number of people who know a specific pair of subjects -/
structure SubjectKnowledge where
  math_physics : Nat
  physics_chemistry : Nat
  chemistry_math : Nat
  physics_biology : Nat

/-- Calculates the total number of people -/
def total_people (sk : SubjectKnowledge) : Nat :=
  sk.math_physics + sk.physics_chemistry + sk.chemistry_math + sk.physics_biology

/-- Calculates the number of ways to choose 3 people from n people -/
def choose_3_from_n (n : Nat) : Nat :=
  n * (n - 1) * (n - 2) / 6

/-- Calculates the number of invalid selections (all 3 from the same group) -/
def invalid_selections (sk : SubjectKnowledge) : Nat :=
  choose_3_from_n sk.math_physics +
  choose_3_from_n sk.physics_chemistry +
  choose_3_from_n sk.chemistry_math +
  choose_3_from_n sk.physics_biology

/-- The main theorem to prove -/
theorem team_formation_ways (sk : SubjectKnowledge) 
  (h1 : sk.math_physics = 7)
  (h2 : sk.physics_chemistry = 6)
  (h3 : sk.chemistry_math = 3)
  (h4 : sk.physics_biology = 4) :
  choose_3_from_n (total_people sk) - invalid_selections sk = 1080 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_ways_l780_78095


namespace NUMINAMATH_CALUDE_range_of_f_domain1_range_of_f_domain2_l780_78092

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x + 9

-- Define the domains
def domain1 : Set ℝ := {x | 3 < x ∧ x ≤ 8}
def domain2 : Set ℝ := {x | -3 < x ∧ x ≤ 2}

-- State the theorems
theorem range_of_f_domain1 :
  f '' domain1 = Set.Ioc 12 57 := by sorry

theorem range_of_f_domain2 :
  f '' domain2 = Set.Ico 8 24 := by sorry

end NUMINAMATH_CALUDE_range_of_f_domain1_range_of_f_domain2_l780_78092


namespace NUMINAMATH_CALUDE_shopping_tax_theorem_l780_78019

/-- Calculates the total tax percentage given spending percentages and tax rates -/
def totalTaxPercentage (clothingPercent : ℝ) (foodPercent : ℝ) (otherPercent : ℝ)
                       (clothingTaxRate : ℝ) (foodTaxRate : ℝ) (otherTaxRate : ℝ) : ℝ :=
  clothingPercent * clothingTaxRate + foodPercent * foodTaxRate + otherPercent * otherTaxRate

theorem shopping_tax_theorem :
  totalTaxPercentage 0.4 0.3 0.3 0.04 0 0.08 = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_shopping_tax_theorem_l780_78019


namespace NUMINAMATH_CALUDE_queenie_earnings_l780_78090

/-- Calculates the total earnings for a part-time clerk with overtime -/
def total_earnings (daily_rate : ℕ) (overtime_rate : ℕ) (days_worked : ℕ) (overtime_hours : ℕ) : ℕ :=
  daily_rate * days_worked + overtime_rate * overtime_hours

/-- Proves that Queenie's total earnings are $770 -/
theorem queenie_earnings : total_earnings 150 5 5 4 = 770 := by
  sorry

end NUMINAMATH_CALUDE_queenie_earnings_l780_78090


namespace NUMINAMATH_CALUDE_union_complement_equality_l780_78099

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3}

-- Define set M
def M : Set Nat := {0, 1, 2}

-- Define set N
def N : Set Nat := {0, 2, 3}

-- Theorem statement
theorem union_complement_equality : M ∪ (U \ N) = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_union_complement_equality_l780_78099


namespace NUMINAMATH_CALUDE_bike_shop_profit_l780_78007

/-- Jim's bike shop problem -/
theorem bike_shop_profit (tire_repair_price : ℕ) (tire_repair_cost : ℕ) (tire_repairs : ℕ)
  (complex_repair_price : ℕ) (complex_repair_cost : ℕ) (complex_repairs : ℕ)
  (fixed_expenses : ℕ) (total_profit : ℕ) :
  tire_repair_price = 20 →
  tire_repair_cost = 5 →
  tire_repairs = 300 →
  complex_repair_price = 300 →
  complex_repair_cost = 50 →
  complex_repairs = 2 →
  fixed_expenses = 4000 →
  total_profit = 3000 →
  (tire_repairs * (tire_repair_price - tire_repair_cost) +
   complex_repairs * (complex_repair_price - complex_repair_cost) -
   fixed_expenses + 2000) = total_profit :=
by sorry

end NUMINAMATH_CALUDE_bike_shop_profit_l780_78007


namespace NUMINAMATH_CALUDE_equation_solution_l780_78073

theorem equation_solution (x : ℝ) : 
  2 - 1 / (3 - x) = 1 / (2 + x) → 
  x = (1 + Real.sqrt 15) / 2 ∨ x = (1 - Real.sqrt 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l780_78073


namespace NUMINAMATH_CALUDE_number_of_fives_l780_78048

theorem number_of_fives (x y : ℕ) : 
  x + y = 20 →
  3 * x + 5 * y = 94 →
  y = 17 := by
sorry

end NUMINAMATH_CALUDE_number_of_fives_l780_78048


namespace NUMINAMATH_CALUDE_labourer_savings_labourer_savings_specific_l780_78082

theorem labourer_savings (monthly_income : ℕ) (initial_expense : ℕ) (reduced_expense : ℕ) 
  (initial_months : ℕ) (reduced_months : ℕ) : ℕ :=
  let initial_total_expense := initial_months * initial_expense
  let initial_total_income := initial_months * monthly_income
  let debt := if initial_total_expense > initial_total_income 
    then initial_total_expense - initial_total_income 
    else 0
  let reduced_total_expense := reduced_months * reduced_expense
  let reduced_total_income := reduced_months * monthly_income
  let savings := reduced_total_income - (reduced_total_expense + debt)
  savings

theorem labourer_savings_specific : 
  labourer_savings 72 75 60 6 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_labourer_savings_labourer_savings_specific_l780_78082


namespace NUMINAMATH_CALUDE_calculate_expression_l780_78026

theorem calculate_expression : 
  (Real.sqrt 2 / 2) * (2 * Real.sqrt 12 / (4 * Real.sqrt (1/8)) - 3 * Real.sqrt 48) = 
  2 * Real.sqrt 3 - 6 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_calculate_expression_l780_78026


namespace NUMINAMATH_CALUDE_three_diamonds_balance_one_emerald_l780_78083

/-- The weight of a single diamond -/
def diamond_weight : ℝ := sorry

/-- The weight of a single emerald -/
def emerald_weight : ℝ := sorry

/-- The balance condition for the initial state of the scale -/
axiom initial_balance : 9 * diamond_weight = 4 * emerald_weight

/-- The balance condition after adding one emerald to the diamond side -/
axiom final_balance : 9 * diamond_weight + emerald_weight = 4 * emerald_weight

/-- Theorem stating that 3 diamonds balance one emerald -/
theorem three_diamonds_balance_one_emerald : 
  3 * diamond_weight = emerald_weight := by sorry

end NUMINAMATH_CALUDE_three_diamonds_balance_one_emerald_l780_78083


namespace NUMINAMATH_CALUDE_chris_soccer_cards_l780_78004

/-- Chris has some soccer cards. His friend, Charlie, has 32 cards. 
    Chris has 14 fewer cards than Charlie. -/
theorem chris_soccer_cards 
  (charlie_cards : ℕ) 
  (chris_fewer : ℕ)
  (h1 : charlie_cards = 32)
  (h2 : chris_fewer = 14) :
  charlie_cards - chris_fewer = 18 := by
  sorry

end NUMINAMATH_CALUDE_chris_soccer_cards_l780_78004


namespace NUMINAMATH_CALUDE_total_slices_is_seven_l780_78031

/-- The number of slices of pie sold yesterday -/
def slices_yesterday : ℕ := 5

/-- The number of slices of pie served today -/
def slices_today : ℕ := 2

/-- The total number of slices of pie sold -/
def total_slices : ℕ := slices_yesterday + slices_today

theorem total_slices_is_seven : total_slices = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_is_seven_l780_78031


namespace NUMINAMATH_CALUDE_take_home_pay_l780_78020

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem take_home_pay :
  annual_salary * (1 - tax_rate - healthcare_rate) - union_dues = 27200 := by
  sorry

end NUMINAMATH_CALUDE_take_home_pay_l780_78020


namespace NUMINAMATH_CALUDE_hourglass_problem_l780_78066

/-- Given two hourglasses that can measure exactly 15 minutes, 
    where one measures 7 minutes, the other measures 2 minutes. -/
theorem hourglass_problem :
  ∀ (x : ℕ), 
    (∃ (n m k : ℕ), n * 7 + m * x + k * (x - 1) = 15 ∧ 
                     n > 0 ∧ m ≥ 0 ∧ k ≥ 0 ∧ 
                     (m = 0 ∨ k = 0)) → 
    x = 2 :=
by sorry

end NUMINAMATH_CALUDE_hourglass_problem_l780_78066


namespace NUMINAMATH_CALUDE_frank_reading_speed_l780_78001

/-- Given a book with a certain number of pages and the number of days to read it,
    calculate the number of pages read per day. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

/-- Theorem stating that for a book with 249 pages read in 3 days,
    the number of pages read per day is 83. -/
theorem frank_reading_speed :
  pages_per_day 249 3 = 83 := by
  sorry

end NUMINAMATH_CALUDE_frank_reading_speed_l780_78001


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l780_78032

/-- Given two parallel 2D vectors a and b, where a = (2, 3) and b = (x, 6), prove that x = 4. -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![2, 3]
  let b : Fin 2 → ℝ := ![x, 6]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l780_78032


namespace NUMINAMATH_CALUDE_smallest_composite_with_large_factors_l780_78037

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_with_large_factors : 
  (is_composite 323) ∧ 
  (has_no_small_prime_factors 323) ∧ 
  (∀ m : ℕ, m < 323 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_with_large_factors_l780_78037


namespace NUMINAMATH_CALUDE_smallest_number_with_digit_sum_2017_properties_l780_78033

/-- The smallest natural number with digit sum 2017 -/
def smallest_number_with_digit_sum_2017 : ℕ :=
  1 * 10^224 + (10^224 - 1)

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  sorry

theorem smallest_number_with_digit_sum_2017_properties :
  digit_sum smallest_number_with_digit_sum_2017 = 2017 ∧
  num_digits smallest_number_with_digit_sum_2017 = 225 ∧
  smallest_number_with_digit_sum_2017 / 10^224 = 1 ∧
  ∀ m : ℕ, m < smallest_number_with_digit_sum_2017 → digit_sum m ≠ 2017 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_digit_sum_2017_properties_l780_78033


namespace NUMINAMATH_CALUDE_min_bound_sqrt_two_l780_78058

theorem min_bound_sqrt_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  min x (min (y + 1/x) (1/y)) ≤ Real.sqrt 2 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ min x (min (y + 1/x) (1/y)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_bound_sqrt_two_l780_78058


namespace NUMINAMATH_CALUDE_certain_number_exists_l780_78077

theorem certain_number_exists : ∃ N : ℝ, (7/13) * N = (5/16) * N + 500 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_exists_l780_78077


namespace NUMINAMATH_CALUDE_bouquet_count_l780_78081

/-- The number of narcissus flowers available -/
def narcissus : ℕ := 75

/-- The number of chrysanthemums available -/
def chrysanthemums : ℕ := 90

/-- The number of flowers in each bouquet -/
def flowers_per_bouquet : ℕ := 5

/-- The total number of bouquets that can be made -/
def total_bouquets : ℕ := (narcissus / flowers_per_bouquet) + (chrysanthemums / flowers_per_bouquet)

theorem bouquet_count : total_bouquets = 33 := by
  sorry

end NUMINAMATH_CALUDE_bouquet_count_l780_78081


namespace NUMINAMATH_CALUDE_odd_function_symmetry_l780_78030

-- Define a real-valued function
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define symmetry about the y-axis for the absolute value of a function
def IsSymmetricAboutYAxis (f : RealFunction) : Prop :=
  ∀ x : ℝ, |f (-x)| = |f x|

-- Theorem statement
theorem odd_function_symmetry :
  (∀ f : RealFunction, IsOdd f → IsSymmetricAboutYAxis f) ∧
  (∃ f : RealFunction, IsSymmetricAboutYAxis f ∧ ¬IsOdd f) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_symmetry_l780_78030


namespace NUMINAMATH_CALUDE_greatest_k_value_l780_78028

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 72) →
  k ≤ 2 * Real.sqrt 26 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l780_78028


namespace NUMINAMATH_CALUDE_license_plate_combinations_l780_78027

/-- The number of ways to choose 2 items from n items -/
def choose (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions on the license plate -/
def letter_positions : ℕ := 4

/-- The number of digit positions on the license plate -/
def digit_positions : ℕ := 3

/-- The maximum starting digit to allow for 3 consecutive increasing digits -/
def max_start_digit : ℕ := 7

theorem license_plate_combinations :
  (choose alphabet_size 2) * (choose letter_positions 2) * (max_start_digit + 1) = 15600 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l780_78027


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l780_78064

theorem quadratic_always_positive (m : ℝ) :
  (∀ x : ℝ, x^2 + m*x + m + 3 > 0) ↔ (-2 < m ∧ m < 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l780_78064


namespace NUMINAMATH_CALUDE_divisibility_of_quotient_l780_78013

theorem divisibility_of_quotient (a b n : ℕ) (h1 : a ≠ b) (h2 : n ∣ (a^n - b^n)) :
  n ∣ ((a^n - b^n) / (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_quotient_l780_78013


namespace NUMINAMATH_CALUDE_four_bb_two_divisible_by_nine_l780_78041

theorem four_bb_two_divisible_by_nine :
  ∃! (B : ℕ), B < 10 ∧ (4000 + 100 * B + 10 * B + 2) % 9 = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_four_bb_two_divisible_by_nine_l780_78041


namespace NUMINAMATH_CALUDE_raisin_cost_fraction_l780_78018

/-- The fraction of the total cost that is the cost of raisins in a mixture of raisins and nuts -/
theorem raisin_cost_fraction (raisin_pounds : ℝ) (nut_pounds : ℝ) (raisin_cost : ℝ) :
  raisin_pounds = 3 →
  nut_pounds = 4 →
  raisin_cost > 0 →
  (raisin_pounds * raisin_cost) / ((raisin_pounds * raisin_cost) + (nut_pounds * (2 * raisin_cost))) = 3 / 11 := by
  sorry

#check raisin_cost_fraction

end NUMINAMATH_CALUDE_raisin_cost_fraction_l780_78018


namespace NUMINAMATH_CALUDE_correct_marble_distribution_l780_78091

/-- Represents the distribution of marbles among three boys -/
structure MarbleDistribution where
  x : ℕ
  first_boy : ℕ := 5 * x + 2
  second_boy : ℕ := 2 * x - 1
  third_boy : ℕ := x + 3

/-- The theorem stating the correct distribution of marbles -/
theorem correct_marble_distribution :
  ∃ (d : MarbleDistribution),
    d.first_boy + d.second_boy + d.third_boy = 60 ∧
    d.first_boy = 37 ∧
    d.second_boy = 13 ∧
    d.third_boy = 10 := by
  sorry

end NUMINAMATH_CALUDE_correct_marble_distribution_l780_78091


namespace NUMINAMATH_CALUDE_value_of_a_l780_78005

theorem value_of_a (x y z a : ℝ) 
  (h1 : 2 * x^2 + 3 * y^2 + 6 * z^2 = a) 
  (h2 : a > 0)
  (h3 : ∀ (x' y' z' : ℝ), 2 * x'^2 + 3 * y'^2 + 6 * z'^2 = a → x' + y' + z' ≤ 1) 
  (h4 : ∃ (x' y' z' : ℝ), 2 * x'^2 + 3 * y'^2 + 6 * z'^2 = a ∧ x' + y' + z' = 1) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_value_of_a_l780_78005


namespace NUMINAMATH_CALUDE_max_participants_answering_A_l780_78022

theorem max_participants_answering_A (total : ℕ) (a b c : ℕ) : 
  total = 39 →
  a + b + c + (a + 3*b - 5) + 3*b + (total - (2*a + 6*b - 5)) = total →
  a = b + c →
  2*(total - (2*a + 6*b - 5)) = 3*b →
  (2*a + 9*b = 44 ∧ a ≥ 0 ∧ b ≥ 0) →
  (∃ max_A : ℕ, max_A = 2*a + 3*b - 5 ∧ max_A ≤ 23 ∧ 
   ∀ other_A : ℕ, other_A = 2*a' + 3*b' - 5 → 
   (2*a' + 9*b' = 44 ∧ a' ≥ 0 ∧ b' ≥ 0) → other_A ≤ max_A) :=
by sorry

end NUMINAMATH_CALUDE_max_participants_answering_A_l780_78022


namespace NUMINAMATH_CALUDE_regions_in_circle_l780_78078

/-- The number of regions created by radii and concentric circles inside a larger circle -/
def num_regions (num_radii : ℕ) (num_circles : ℕ) : ℕ :=
  (num_circles + 1) * num_radii

/-- Theorem stating that 16 radii and 10 concentric circles create 176 regions -/
theorem regions_in_circle (num_radii : ℕ) (num_circles : ℕ) 
    (h1 : num_radii = 16) (h2 : num_circles = 10) : 
    num_regions num_radii num_circles = 176 := by
  sorry

#eval num_regions 16 10

end NUMINAMATH_CALUDE_regions_in_circle_l780_78078


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l780_78071

/-- Proves that for a rectangle with length 4x and width x+7, if the area is twice the perimeter, then x = 1 -/
theorem rectangle_area_perimeter_relation (x : ℝ) :
  (4 * x) * (x + 7) = 2 * (2 * (4 * x) + 2 * (x + 7)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_relation_l780_78071


namespace NUMINAMATH_CALUDE_f_is_even_f_is_increasing_on_positive_l780_78060

-- Define the function f(x) = |x| + 1
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem stating that f is an even function
theorem f_is_even : ∀ x : ℝ, f x = f (-x) := by sorry

-- Theorem stating that f is monotonically increasing on (0, +∞)
theorem f_is_increasing_on_positive : ∀ x y : ℝ, 0 < x → x < y → f x < f y := by sorry

end NUMINAMATH_CALUDE_f_is_even_f_is_increasing_on_positive_l780_78060


namespace NUMINAMATH_CALUDE_soccer_game_theorem_l780_78039

def soccer_game (team_a_first_half : ℕ) (team_b_second_half : ℕ) (total_goals : ℕ) : Prop :=
  let team_b_first_half := team_a_first_half / 2
  let first_half_total := team_a_first_half + team_b_first_half
  let second_half_total := total_goals - first_half_total
  let team_a_second_half := second_half_total - team_b_second_half
  (team_a_first_half = 8) ∧
  (team_b_second_half = team_a_first_half) ∧
  (total_goals = 26) ∧
  (team_b_second_half > team_a_second_half) ∧
  (team_b_second_half - team_a_second_half = 2)

theorem soccer_game_theorem :
  ∃ (team_a_first_half team_b_second_half total_goals : ℕ),
    soccer_game team_a_first_half team_b_second_half total_goals :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_game_theorem_l780_78039


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l780_78044

/-- The number of ways to distribute n volunteers into k groups with size constraints -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to assign k groups to k different areas -/
def assign (k : ℕ) : ℕ := sorry

/-- The total number of allocation schemes -/
def total_schemes (n : ℕ) (k : ℕ) : ℕ :=
  distribute n k * assign k

theorem allocation_schemes_count :
  total_schemes 6 4 = 1080 := by sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l780_78044


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l780_78059

-- Define the function f(x) = |x + 2|
def f (x : ℝ) : ℝ := |x + 2|

-- State the theorem
theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ →
  (x₁ - x₂) * (f x₁ - f x₂) > 0 :=
by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l780_78059


namespace NUMINAMATH_CALUDE_smallest_gcd_l780_78084

theorem smallest_gcd (a b c : ℕ+) 
  (h1 : Nat.gcd a.val b.val = 360)
  (h2 : Nat.gcd a.val c.val = 1170)
  (h3 : ∃ k : ℕ, b.val = 5 * k)
  (h4 : ∃ m : ℕ, c.val = 13 * m) :
  Nat.gcd b.val c.val ≥ 90 ∧ ∃ (a' b' c' : ℕ+), 
    Nat.gcd a'.val b'.val = 360 ∧ 
    Nat.gcd a'.val c'.val = 1170 ∧ 
    (∃ k : ℕ, b'.val = 5 * k) ∧
    (∃ m : ℕ, c'.val = 13 * m) ∧
    Nat.gcd b'.val c'.val = 90 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_l780_78084


namespace NUMINAMATH_CALUDE_rectangle_area_l780_78008

/-- Rectangle ABCD with specific properties -/
structure Rectangle where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side AD -/
  AD : ℝ
  /-- AD is 9 units longer than AB -/
  length_diff : AD = AB + 9
  /-- Area of trapezoid ABCE is 5 times the area of triangle ADE -/
  area_ratio : AB * AD = 6 * ((1/2) * AB * (1/3 * AD))
  /-- Perimeter difference between trapezoid ABCE and triangle ADE -/
  perimeter_diff : AB + (2/3 * AB) - (1/3 * AB) = 68

/-- The area of the rectangle ABCD is 3060 square units -/
theorem rectangle_area (rect : Rectangle) : rect.AB * rect.AD = 3060 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_area_l780_78008


namespace NUMINAMATH_CALUDE_coconut_trips_l780_78017

/-- The number of trips needed to move coconuts -/
def num_trips (total_coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : ℕ :=
  total_coconuts / (barbie_capacity + bruno_capacity)

/-- Theorem stating that 12 trips are needed to move 144 coconuts -/
theorem coconut_trips : num_trips 144 4 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_coconut_trips_l780_78017


namespace NUMINAMATH_CALUDE_cos_value_for_special_angle_l780_78042

theorem cos_value_for_special_angle (θ : Real) 
  (h1 : 6 * Real.tan θ = 2 * Real.sin θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.cos θ = -1 := by
  sorry

end NUMINAMATH_CALUDE_cos_value_for_special_angle_l780_78042


namespace NUMINAMATH_CALUDE_heat_required_for_temperature_change_l780_78000

/-- Specific heat capacity as a function of temperature -/
def specific_heat_capacity (c₀ α t : ℝ) : ℝ := c₀ * (1 + α * t)

/-- Amount of heat required to change temperature -/
def heat_required (m c_avg Δt : ℝ) : ℝ := m * c_avg * Δt

theorem heat_required_for_temperature_change 
  (m : ℝ) 
  (c₀ : ℝ) 
  (α : ℝ) 
  (t_initial t_final : ℝ) 
  (h_m : m = 3) 
  (h_c₀ : c₀ = 200) 
  (h_α : α = 0.05) 
  (h_t_initial : t_initial = 30) 
  (h_t_final : t_final = 80) :
  heat_required m 
    ((specific_heat_capacity c₀ α t_initial + specific_heat_capacity c₀ α t_final) / 2) 
    (t_final - t_initial) = 112500 := by
  sorry

#check heat_required_for_temperature_change

end NUMINAMATH_CALUDE_heat_required_for_temperature_change_l780_78000


namespace NUMINAMATH_CALUDE_coffee_cheesecake_set_price_l780_78034

/-- Calculates the discounted price of a coffee and cheesecake set --/
def discounted_set_price (coffee_price cheesecake_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_price := coffee_price + cheesecake_price
  let discount_amount := discount_rate * total_price
  total_price - discount_amount

/-- Proves that the final price of a coffee and cheesecake set with a 25% discount is $12 --/
theorem coffee_cheesecake_set_price :
  discounted_set_price 6 10 (25 / 100) = 12 :=
by sorry

end NUMINAMATH_CALUDE_coffee_cheesecake_set_price_l780_78034


namespace NUMINAMATH_CALUDE_square_side_length_l780_78096

theorem square_side_length (area : ℝ) (side : ℝ) (h1 : area = 9 / 16) (h2 : side * side = area) : side = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l780_78096
