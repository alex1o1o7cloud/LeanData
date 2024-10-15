import Mathlib

namespace NUMINAMATH_CALUDE_work_completion_time_l293_29315

/-- 
If two workers can complete a job together in a certain time, 
and one worker can complete it alone in a known time, 
we can determine how long it takes the other worker to complete the job alone.
-/
theorem work_completion_time 
  (total_work : ℝ) 
  (time_together time_a time_b : ℝ) 
  (h1 : time_together > 0)
  (h2 : time_a > 0)
  (h3 : time_b > 0)
  (h4 : total_work / time_together = total_work / time_a + total_work / time_b)
  (h5 : time_together = 5)
  (h6 : time_a = 10) :
  time_b = 10 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l293_29315


namespace NUMINAMATH_CALUDE_farm_section_area_l293_29379

/-- Given a farm with a total area of 300 acres divided into 5 equal sections,
    prove that the area of each section is 60 acres. -/
theorem farm_section_area (total_area : ℝ) (num_sections : ℕ) (section_area : ℝ) :
  total_area = 300 ∧ num_sections = 5 ∧ section_area * num_sections = total_area →
  section_area = 60 := by
  sorry

end NUMINAMATH_CALUDE_farm_section_area_l293_29379


namespace NUMINAMATH_CALUDE_weight_of_new_person_l293_29314

theorem weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 40 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 60 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l293_29314


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l293_29316

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*m*x + 3*m^2
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (m > 0 → (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ |x₁ - x₂| = 2) → m = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l293_29316


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l293_29303

theorem necessary_not_sufficient_condition : 
  (∀ x : ℝ, x > 2 → x > 1) ∧ 
  (∃ x : ℝ, x > 1 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l293_29303


namespace NUMINAMATH_CALUDE_castle_provisions_theorem_l293_29359

/-- Represents the number of days provisions last given initial conditions and a change in population -/
def days_until_food_runs_out (initial_people : ℕ) (initial_days : ℕ) (days_passed : ℕ) (people_left : ℕ) : ℕ :=
  let remaining_days := initial_days - days_passed
  let new_duration := (remaining_days * initial_people) / people_left
  new_duration

/-- Theorem stating that under given conditions, food lasts for 90 more days after population change -/
theorem castle_provisions_theorem (initial_people : ℕ) (initial_days : ℕ) 
  (days_passed : ℕ) (people_left : ℕ) :
  initial_people = 300 ∧ initial_days = 90 ∧ days_passed = 30 ∧ people_left = 200 →
  days_until_food_runs_out initial_people initial_days days_passed people_left = 90 :=
by
  sorry

#eval days_until_food_runs_out 300 90 30 200

end NUMINAMATH_CALUDE_castle_provisions_theorem_l293_29359


namespace NUMINAMATH_CALUDE_profit_percentage_l293_29397

theorem profit_percentage (C S : ℝ) (h : C > 0) : 
  20 * C = 16 * S → (S - C) / C * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l293_29397


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_34_l293_29356

/-- A trapezoid with specific side lengths -/
structure Trapezoid :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)
  (h_AB_eq_CD : AB = CD)
  (h_AB : AB = 8)
  (h_CD : CD = 16)
  (h_BC_eq_DA : BC = DA)
  (h_BC : BC = 5)

/-- The perimeter of a trapezoid is the sum of its sides -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.BC + t.CD + t.DA

/-- Theorem: The perimeter of the specified trapezoid is 34 -/
theorem trapezoid_perimeter_is_34 (t : Trapezoid) : perimeter t = 34 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_34_l293_29356


namespace NUMINAMATH_CALUDE_power_sum_of_i_l293_29338

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^11 + i^111 = -2 * i :=
  sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l293_29338


namespace NUMINAMATH_CALUDE_min_convex_division_rotated_ngon_l293_29386

/-- A regular n-gon. -/
structure RegularNGon (n : ℕ) where
  -- Add necessary fields here

/-- Rotate a regular n-gon by an angle around its center. -/
def rotate (M : RegularNGon n) (angle : ℝ) : RegularNGon n :=
  sorry

/-- The union of two regular n-gons. -/
def union (M M' : RegularNGon n) : Set (ℝ × ℝ) :=
  sorry

/-- A convex polygon. -/
structure ConvexPolygon where
  -- Add necessary fields here

/-- The minimum number of convex polygons needed to divide a set. -/
def minConvexDivision (S : Set (ℝ × ℝ)) : ℕ :=
  sorry

theorem min_convex_division_rotated_ngon (n : ℕ) (M : RegularNGon n) :
  minConvexDivision (union M (rotate M (π / n))) = n + 1 :=
sorry

end NUMINAMATH_CALUDE_min_convex_division_rotated_ngon_l293_29386


namespace NUMINAMATH_CALUDE_line_point_k_value_l293_29309

/-- Given three points on a line, calculate the value of k -/
theorem line_point_k_value (k : ℝ) : 
  (∃ (m b : ℝ), 7 = m * 3 + b ∧ k = m * 5 + b ∧ 15 = m * 11 + b) → k = 9 := by
  sorry

end NUMINAMATH_CALUDE_line_point_k_value_l293_29309


namespace NUMINAMATH_CALUDE_f_derivative_f_at_one_f_equality_l293_29372

/-- A function f satisfying f'(x) = 4x^3 for all x and f(1) = -1 -/
def f : ℝ → ℝ :=
  sorry

theorem f_derivative (x : ℝ) : deriv f x = 4 * x^3 :=
  sorry

theorem f_at_one : f 1 = -1 :=
  sorry

theorem f_equality (x : ℝ) : f x = x^4 - 2 :=
  sorry

end NUMINAMATH_CALUDE_f_derivative_f_at_one_f_equality_l293_29372


namespace NUMINAMATH_CALUDE_unique_prime_solution_l293_29324

theorem unique_prime_solution :
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (7 * p * q^2 + p = q^3 + 43 * p^3 + 1) → 
    (p = 2 ∧ q = 7) := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l293_29324


namespace NUMINAMATH_CALUDE_claudia_water_amount_l293_29380

/-- The amount of water in ounces that Claudia had initially -/
def initial_water : ℕ := 122

/-- The capacity of a 5-ounce glass in ounces -/
def five_ounce_glass : ℕ := 5

/-- The capacity of an 8-ounce glass in ounces -/
def eight_ounce_glass : ℕ := 8

/-- The capacity of a 4-ounce glass in ounces -/
def four_ounce_glass : ℕ := 4

/-- The number of 5-ounce glasses filled -/
def num_five_ounce : ℕ := 6

/-- The number of 8-ounce glasses filled -/
def num_eight_ounce : ℕ := 4

/-- The number of 4-ounce glasses that can be filled with the remaining water -/
def num_four_ounce : ℕ := 15

theorem claudia_water_amount :
  initial_water = 
    num_five_ounce * five_ounce_glass + 
    num_eight_ounce * eight_ounce_glass + 
    num_four_ounce * four_ounce_glass := by
  sorry

end NUMINAMATH_CALUDE_claudia_water_amount_l293_29380


namespace NUMINAMATH_CALUDE_alpha_value_l293_29318

theorem alpha_value (α β : ℂ) 
  (h1 : (α + 2*β).re > 0)
  (h2 : (Complex.I * (α - 3*β)).re > 0)
  (h3 : β = 2 + 3*Complex.I) : 
  α = 6 - 6*Complex.I := by sorry

end NUMINAMATH_CALUDE_alpha_value_l293_29318


namespace NUMINAMATH_CALUDE_age_difference_l293_29388

theorem age_difference (A B : ℕ) : B = 48 → A + 10 = 2 * (B - 10) → A - B = 18 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l293_29388


namespace NUMINAMATH_CALUDE_first_number_value_l293_29393

theorem first_number_value (y x : ℚ) : 
  (y + 76 + x) / 3 = 5 → x = -63 → y = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l293_29393


namespace NUMINAMATH_CALUDE_floor_sqrt_225_l293_29399

theorem floor_sqrt_225 : ⌊Real.sqrt 225⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_225_l293_29399


namespace NUMINAMATH_CALUDE_divisibility_condition_l293_29312

theorem divisibility_condition (x y : ℕ+) :
  (xy^2 + y + 7 ∣ x^2*y + x + y) ↔ 
  (∃ t : ℕ+, x = 7*t^2 ∧ y = 7*t) ∨ (x = 11 ∧ y = 1) ∨ (x = 49 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l293_29312


namespace NUMINAMATH_CALUDE_two_ones_in_twelve_dice_l293_29302

def probability_two_ones (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem two_ones_in_twelve_dice :
  let n : ℕ := 12
  let k : ℕ := 2
  let p : ℚ := 1/6
  probability_two_ones n k p = 66 * (1/36) * (9765625/60466176) :=
sorry

end NUMINAMATH_CALUDE_two_ones_in_twelve_dice_l293_29302


namespace NUMINAMATH_CALUDE_min_clients_theorem_exists_solution_with_101_min_clients_is_101_l293_29398

/-- Represents a repunit number with n ones -/
def repunit (n : ℕ) : ℕ :=
  (10^n - 1) / 9

/-- The property that needs to be satisfied for the group of clients -/
def satisfies_property (m k : ℕ) : Prop :=
  ∃ n : ℕ, n > k ∧ k > 1 ∧ repunit n = repunit k * m

/-- The main theorem stating the minimum number of clients -/
theorem min_clients_theorem :
  ∀ m : ℕ, m > 1 → (satisfies_property m 2) → m ≥ 101 :=
by sorry

/-- The existence theorem proving there is a solution with 101 clients -/
theorem exists_solution_with_101 :
  satisfies_property 101 2 :=
by sorry

/-- The final theorem proving 101 is the minimum number of clients -/
theorem min_clients_is_101 :
  ∀ m : ℕ, m > 1 → satisfies_property m 2 → m ≥ 101 ∧ satisfies_property 101 2 :=
by sorry

end NUMINAMATH_CALUDE_min_clients_theorem_exists_solution_with_101_min_clients_is_101_l293_29398


namespace NUMINAMATH_CALUDE_square_side_length_l293_29369

theorem square_side_length (k : ℝ) (s d : ℝ) (h1 : s > 0) (h2 : d > 0) (h3 : s + d = k) (h4 : d = s * Real.sqrt 2) :
  s = k / (1 + Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l293_29369


namespace NUMINAMATH_CALUDE_new_students_count_l293_29366

/-- Calculates the number of new students who came to school during the year. -/
def new_students (initial : ℕ) (left : ℕ) (final : ℕ) : ℕ :=
  final - (initial - left)

/-- Proves that the number of new students who came to school during the year is 42. -/
theorem new_students_count :
  new_students 4 3 43 = 42 := by
  sorry

end NUMINAMATH_CALUDE_new_students_count_l293_29366


namespace NUMINAMATH_CALUDE_smallest_third_side_of_right_triangle_l293_29394

theorem smallest_third_side_of_right_triangle (a b c : ℝ) :
  a = 5 →
  b = 4 →
  c > 0 →
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 →
  3 ≤ c :=
by sorry

end NUMINAMATH_CALUDE_smallest_third_side_of_right_triangle_l293_29394


namespace NUMINAMATH_CALUDE_point_B_coordinates_l293_29396

def point := ℝ × ℝ

def vector := ℝ × ℝ

def point_A : point := (-1, 5)

def vector_AB : vector := (6, 9)

def point_B : point := (5, 14)

def vector_between (p q : point) : vector :=
  (q.1 - p.1, q.2 - p.2)

theorem point_B_coordinates :
  vector_between point_A point_B = vector_AB :=
by sorry

end NUMINAMATH_CALUDE_point_B_coordinates_l293_29396


namespace NUMINAMATH_CALUDE_friend_lunch_cost_l293_29322

/-- Proves that given a total lunch cost of $17 and one person spending $3 more than the other,
    the person who spent more paid $10. -/
theorem friend_lunch_cost (total : ℝ) (difference : ℝ) (friend_cost : ℝ) : 
  total = 17 → difference = 3 → friend_cost = total / 2 + difference / 2 → friend_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_friend_lunch_cost_l293_29322


namespace NUMINAMATH_CALUDE_omega_cube_root_unity_l293_29336

theorem omega_cube_root_unity (ω : ℂ) : 
  ω = -1/2 + (Complex.I * Real.sqrt 3)/2 → ω^2 + ω + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_omega_cube_root_unity_l293_29336


namespace NUMINAMATH_CALUDE_sum_of_coordinates_reflection_l293_29368

/-- Given a point C with coordinates (3, y) and its reflection D over the x-axis,
    the sum of all coordinates of C and D is 6. -/
theorem sum_of_coordinates_reflection (y : ℝ) : 
  let C : ℝ × ℝ := (3, y)
  let D : ℝ × ℝ := (3, -y)  -- reflection of C over x-axis
  C.1 + C.2 + D.1 + D.2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_reflection_l293_29368


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_seven_l293_29307

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n % sum_of_digits n = 0

def is_multiple_of_seven (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

theorem least_non_lucky_multiple_of_seven :
  (14 % sum_of_digits 14 ≠ 0) ∧
  is_multiple_of_seven 14 ∧
  ∀ n : ℕ, 0 < n ∧ n < 14 ∧ is_multiple_of_seven n → is_lucky n :=
by sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_seven_l293_29307


namespace NUMINAMATH_CALUDE_exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l293_29334

/-- The measure of an exterior angle formed by a regular pentagon and a regular octagon sharing a side -/
theorem exterior_angle_pentagon_octagon : ℝ :=
  let pentagon_interior_angle : ℝ := (180 * (5 - 2)) / 5
  let octagon_interior_angle : ℝ := (180 * (8 - 2)) / 8
  360 - (pentagon_interior_angle + octagon_interior_angle)

/-- The exterior angle formed by a regular pentagon and a regular octagon sharing a side is 117 degrees -/
theorem exterior_angle_pentagon_octagon_is_117 :
  exterior_angle_pentagon_octagon = 117 := by sorry

end NUMINAMATH_CALUDE_exterior_angle_pentagon_octagon_exterior_angle_pentagon_octagon_is_117_l293_29334


namespace NUMINAMATH_CALUDE_ratio_of_40_to_8_l293_29352

theorem ratio_of_40_to_8 (certain_number : ℚ) (h : certain_number = 40) : 
  certain_number / 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_40_to_8_l293_29352


namespace NUMINAMATH_CALUDE_prob_two_green_apples_l293_29371

/-- The probability of selecting 2 green apples from a set of 7 apples, where 3 are green -/
theorem prob_two_green_apples (total : ℕ) (green : ℕ) (choose : ℕ) 
  (h_total : total = 7) 
  (h_green : green = 3) 
  (h_choose : choose = 2) :
  (Nat.choose green choose : ℚ) / (Nat.choose total choose : ℚ) = 1 / 7 := by
  sorry

#check prob_two_green_apples

end NUMINAMATH_CALUDE_prob_two_green_apples_l293_29371


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l293_29347

theorem complex_simplification_and_multiplication :
  ((-5 + 3 * Complex.I) - (2 - 7 * Complex.I)) * (1 + 2 * Complex.I) = -27 - 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l293_29347


namespace NUMINAMATH_CALUDE_quadratic_exponent_condition_l293_29391

theorem quadratic_exponent_condition (a : ℝ) : 
  (∀ x, ∃ p q r : ℝ, x^(a^2 - 7) - 3*x - 2 = p*x^2 + q*x + r) → 
  (a = 3 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_exponent_condition_l293_29391


namespace NUMINAMATH_CALUDE_friends_receiving_balls_l293_29300

/-- The number of ping pong balls Eunji has -/
def total_balls : ℕ := 44

/-- The number of ping pong balls given to each friend -/
def balls_per_friend : ℕ := 4

/-- The number of friends who will receive ping pong balls -/
def num_friends : ℕ := total_balls / balls_per_friend

theorem friends_receiving_balls : num_friends = 11 := by
  sorry

end NUMINAMATH_CALUDE_friends_receiving_balls_l293_29300


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l293_29301

def set_A : Set ℝ := {x | x^2 = 1}
def set_B : Set ℝ := {x | x^2 - 2*x - 3 = 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l293_29301


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l293_29383

def A : Set ℝ := {x : ℝ | x^2 - x - 2 = 0}

def B : Set ℝ := {y : ℝ | ∃ x ∈ A, y = x + 3}

theorem union_of_A_and_B : A ∪ B = {-1, 2, 5} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l293_29383


namespace NUMINAMATH_CALUDE_brothers_age_fraction_l293_29339

theorem brothers_age_fraction :
  let younger_age : ℕ := 27
  let total_age : ℕ := 46
  let older_age : ℕ := total_age - younger_age
  ∃ f : ℚ, younger_age = f * older_age + 10 ∧ f = 17 / 19 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_fraction_l293_29339


namespace NUMINAMATH_CALUDE_four_false_statements_l293_29329

/-- Represents a statement on the card -/
inductive Statement
| one
| two
| three
| four
| all

/-- The truth value of a statement -/
def isFalse : Statement → Bool
| Statement.one => true
| Statement.two => true
| Statement.three => true
| Statement.four => false
| Statement.all => true

/-- The claim made by each statement -/
def claim : Statement → Nat
| Statement.one => 1
| Statement.two => 2
| Statement.three => 3
| Statement.four => 4
| Statement.all => 5

/-- The total number of false statements -/
def totalFalse : Nat := 
  (Statement.one :: Statement.two :: Statement.three :: Statement.four :: Statement.all :: []).filter isFalse |>.length

/-- Theorem stating that exactly 4 statements are false -/
theorem four_false_statements : totalFalse = 4 ∧ 
  ∀ s : Statement, isFalse s = true ↔ claim s ≠ totalFalse :=
  sorry


end NUMINAMATH_CALUDE_four_false_statements_l293_29329


namespace NUMINAMATH_CALUDE_expression_factorization_l293_29305

variable (x : ℝ)

theorem expression_factorization :
  (12 * x^3 + 27 * x^2 + 90 * x - 9) - (-3 * x^3 + 9 * x^2 - 15 * x - 9) =
  3 * x * (5 * x^2 + 6 * x + 35) := by sorry

end NUMINAMATH_CALUDE_expression_factorization_l293_29305


namespace NUMINAMATH_CALUDE_volleyball_lineup_combinations_l293_29362

def team_size : ℕ := 12
def starting_lineup_size : ℕ := 6
def non_libero_positions : ℕ := 5

theorem volleyball_lineup_combinations :
  (team_size) * (Nat.choose (team_size - 1) non_libero_positions) = 5544 :=
sorry

end NUMINAMATH_CALUDE_volleyball_lineup_combinations_l293_29362


namespace NUMINAMATH_CALUDE_perpendicular_radii_intercept_l293_29328

/-- Given a circle and a line intersecting it, if the radii to the intersection points are perpendicular, then the y-intercept of the line has specific values. -/
theorem perpendicular_radii_intercept (b : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*x = 0}
  let line := {(x, y) : ℝ × ℝ | y = x + b}
  let C := (2, 0)
  ∃ (M N : ℝ × ℝ), 
    M ∈ circle ∧ M ∈ line ∧
    N ∈ circle ∧ N ∈ line ∧
    M ≠ N ∧
    (M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2) = 0 →
    b = 0 ∨ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_radii_intercept_l293_29328


namespace NUMINAMATH_CALUDE_intersecting_lines_k_value_l293_29325

/-- Given two lines that intersect at a specific point, prove the value of k -/
theorem intersecting_lines_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + 5) →  -- Line p equation
  (∀ x y : ℝ, y = k * x + 3) →  -- Line q equation
  -7 = 3 * (-4) + 5 →           -- Point (-4, -7) satisfies line p equation
  -7 = k * (-4) + 3 →           -- Point (-4, -7) satisfies line q equation
  k = 2.5 := by
sorry

end NUMINAMATH_CALUDE_intersecting_lines_k_value_l293_29325


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l293_29330

theorem modular_arithmetic_problem (m : ℕ) : 
  13^5 % 7 = m → 0 ≤ m → m < 7 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l293_29330


namespace NUMINAMATH_CALUDE_parabola_chord_constant_sum_l293_29373

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = 2x^2 -/
def parabola (p : Point) : Prop :=
  p.y = 2 * p.x^2

/-- Distance squared between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: For the parabola y = 2x^2, if there exists a constant c such that
    for any chord AB passing through the point (0,c), the value
    t = 1/AC^2 + 1/BC^2 is constant, then c = 1/4 and t = 8 -/
theorem parabola_chord_constant_sum (c : ℝ) :
  (∃ t : ℝ, ∀ A B : Point,
    parabola A ∧ parabola B ∧
    (∃ m : ℝ, A.y = m * A.x + c ∧ B.y = m * B.x + c) →
    1 / distanceSquared A ⟨0, c⟩ + 1 / distanceSquared B ⟨0, c⟩ = t) →
  c = 1/4 ∧ t = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_chord_constant_sum_l293_29373


namespace NUMINAMATH_CALUDE_walnut_chestnut_cost_l293_29345

/-- The total cost of buying walnuts and chestnuts -/
def total_cost (m n : ℝ) : ℝ :=
  2 * m + 3 * n

/-- Theorem: The total cost of buying 2 kg of walnuts at m yuan/kg and 3 kg of chestnuts at n yuan/kg is (2m + 3n) yuan -/
theorem walnut_chestnut_cost (m n : ℝ) :
  total_cost m n = 2 * m + 3 * n :=
by sorry

end NUMINAMATH_CALUDE_walnut_chestnut_cost_l293_29345


namespace NUMINAMATH_CALUDE_complex_equation_sum_l293_29384

theorem complex_equation_sum (a b : ℝ) : (1 + 2*I)*I = a + b*I → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l293_29384


namespace NUMINAMATH_CALUDE_grade_assignments_l293_29310

/-- The number of students in the class -/
def num_students : ℕ := 8

/-- The number of distinct grades available -/
def num_grades : ℕ := 4

/-- Theorem: The number of ways to assign grades to students -/
theorem grade_assignments :
  num_grades ^ num_students = 65536 := by sorry

end NUMINAMATH_CALUDE_grade_assignments_l293_29310


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l293_29387

theorem least_addition_for_divisibility : 
  (∃ (n : ℕ), 25 ∣ (1019 + n) ∧ ∀ (m : ℕ), m < n → ¬(25 ∣ (1019 + m))) ∧ 
  (∃ (n : ℕ), n = 6 ∧ 25 ∣ (1019 + n) ∧ ∀ (m : ℕ), m < n → ¬(25 ∣ (1019 + m))) :=
by sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l293_29387


namespace NUMINAMATH_CALUDE_side_length_of_five_cubes_l293_29390

/-- Given five equal cubes placed adjacent to each other forming a new solid with volume 625 cm³,
    prove that the side length of each cube is 5 cm. -/
theorem side_length_of_five_cubes (n : ℕ) (v : ℝ) (s : ℝ) : 
  n = 5 → v = 625 → v = n * s^3 → s = 5 := by sorry

end NUMINAMATH_CALUDE_side_length_of_five_cubes_l293_29390


namespace NUMINAMATH_CALUDE_max_sum_constrained_length_l293_29350

/-- The length of an integer is the number of positive prime factors (not necessarily distinct) whose product equals the integer -/
def length (n : ℕ) : ℕ := sorry

/-- The theorem states that given the conditions, the maximum value of x + 3y is 49156 -/
theorem max_sum_constrained_length (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h_length_sum : length x + length y ≤ 16) :
  x + 3 * y ≤ 49156 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_constrained_length_l293_29350


namespace NUMINAMATH_CALUDE_no_return_after_12_jumps_all_return_after_13_jumps_l293_29376

/-- Represents a point on a circle -/
structure CirclePoint where
  position : ℕ

/-- The number of points on the circle -/
def n : ℕ := 12

/-- The jump function that moves a point to the next clockwise midpoint -/
def jump (p : CirclePoint) : CirclePoint :=
  ⟨(p.position + 1) % n⟩

/-- Applies the jump function k times -/
def jumpK (p : CirclePoint) (k : ℕ) : CirclePoint :=
  match k with
  | 0 => p
  | k + 1 => jump (jumpK p k)

theorem no_return_after_12_jumps :
  ∀ p : CirclePoint, jumpK p 12 ≠ p :=
sorry

theorem all_return_after_13_jumps :
  ∀ p : CirclePoint, jumpK p 13 = p :=
sorry

end NUMINAMATH_CALUDE_no_return_after_12_jumps_all_return_after_13_jumps_l293_29376


namespace NUMINAMATH_CALUDE_factors_imply_value_l293_29361

/-- The polynomial p(x) = 3x^3 - mx + n -/
def p (m n : ℝ) (x : ℝ) : ℝ := 3 * x^3 - m * x + n

theorem factors_imply_value (m n : ℝ) 
  (h1 : p m n 3 = 0)  -- x-3 is a factor
  (h2 : p m n (-4) = 0)  -- x+4 is a factor
  : |3*m - 2*n| = 33 := by
  sorry

end NUMINAMATH_CALUDE_factors_imply_value_l293_29361


namespace NUMINAMATH_CALUDE_inequality_holds_l293_29354

/-- A quadratic function with the given symmetry property -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

/-- The symmetry property of f -/
axiom symmetry_at_3 (b c : ℝ) : ∀ t : ℝ, f b c (3 + t) = f b c (3 - t)

/-- The main theorem stating the inequality -/
theorem inequality_holds (b c : ℝ) : f b c 3 < f b c 1 ∧ f b c 1 < f b c 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l293_29354


namespace NUMINAMATH_CALUDE_rectangle_area_l293_29321

/-- The area of a rectangle with vertices at (-3, 6), (1, 1), and (1, -6), 
    where (1, -6) is 7 units away from (1, 1), is equal to 7√41. -/
theorem rectangle_area : 
  let v1 : ℝ × ℝ := (-3, 6)
  let v2 : ℝ × ℝ := (1, 1)
  let v3 : ℝ × ℝ := (1, -6)
  let side1 := Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2)
  let side2 := |v2.2 - v3.2|
  side2 = 7 →
  side1 * side2 = 7 * Real.sqrt 41 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l293_29321


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l293_29327

/-- An arithmetic sequence with given first three terms -/
def arithmetic_sequence (x : ℝ) (n : ℕ) : ℝ :=
  let a₁ := x - 1
  let a₂ := x + 1
  let a₃ := 2 * x + 3
  let d := a₂ - a₁  -- common difference
  a₁ + (n - 1) * d

/-- Theorem stating the general formula for the given arithmetic sequence -/
theorem arithmetic_sequence_formula (x : ℝ) (n : ℕ) :
  arithmetic_sequence x n = 2 * n - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l293_29327


namespace NUMINAMATH_CALUDE_remainder_of_n_l293_29385

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 7 = 1) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_n_l293_29385


namespace NUMINAMATH_CALUDE_solution_theorem_l293_29344

-- Define the function f(x) = x^2023 + x
def f (x : ℝ) := x^2023 + x

-- State the theorem
theorem solution_theorem (x y : ℝ) :
  (3*x + y)^2023 + x^2023 + 4*x + y = 0 → 4*x + y = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_theorem_l293_29344


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l293_29381

theorem quadratic_root_relation (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 + m*x₁ + 5 = 0 ∧ 
                x₂^2 + m*x₂ + 5 = 0 ∧ 
                x₁ = 2*|x₂| - 3) → 
  m = -9/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l293_29381


namespace NUMINAMATH_CALUDE_stations_between_hyderabad_and_bangalore_l293_29323

theorem stations_between_hyderabad_and_bangalore : 
  ∃ (n : ℕ), n > 2 ∧ (n * (n - 1)) / 2 = 306 ∧ n - 2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_stations_between_hyderabad_and_bangalore_l293_29323


namespace NUMINAMATH_CALUDE_gcd_of_sides_gt_one_l293_29332

/-- A triangle with integer sides -/
structure IntegerTriangle where
  a : ℕ  -- side BC
  b : ℕ  -- side CA
  c : ℕ  -- side AB
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem to be proved -/
theorem gcd_of_sides_gt_one
  (t : IntegerTriangle)
  (side_order : t.c < t.b)
  (tangent_intersect : ℕ)  -- AD, the intersection of tangent at A with BC
  : Nat.gcd t.b t.c > 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_sides_gt_one_l293_29332


namespace NUMINAMATH_CALUDE_alcohol_distribution_correct_l293_29340

/-- Represents a container with an alcohol solution -/
structure Container where
  volume : ℝ
  concentration : ℝ

/-- Calculates the amount of pure alcohol needed to achieve the desired concentration -/
def pureAlcoholNeeded (c : Container) (desiredConcentration : ℝ) : ℝ :=
  c.volume * desiredConcentration - c.volume * c.concentration

/-- Theorem: The calculated amounts of pure alcohol will result in 50% solutions -/
theorem alcohol_distribution_correct 
  (containerA containerB containerC : Container)
  (pureAlcoholA pureAlcoholB pureAlcoholC : ℝ)
  (h1 : containerA = { volume := 8, concentration := 0.25 })
  (h2 : containerB = { volume := 10, concentration := 0.40 })
  (h3 : containerC = { volume := 6, concentration := 0.30 })
  (h4 : pureAlcoholA = pureAlcoholNeeded containerA 0.5)
  (h5 : pureAlcoholB = pureAlcoholNeeded containerB 0.5)
  (h6 : pureAlcoholC = pureAlcoholNeeded containerC 0.5)
  (h7 : pureAlcoholA + pureAlcoholB + pureAlcoholC ≤ 12) :
  pureAlcoholA = 2 ∧ 
  pureAlcoholB = 1 ∧ 
  pureAlcoholC = 1.2 ∧
  (containerA.volume * containerA.concentration + pureAlcoholA) / (containerA.volume + pureAlcoholA) = 0.5 ∧
  (containerB.volume * containerB.concentration + pureAlcoholB) / (containerB.volume + pureAlcoholB) = 0.5 ∧
  (containerC.volume * containerC.concentration + pureAlcoholC) / (containerC.volume + pureAlcoholC) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_distribution_correct_l293_29340


namespace NUMINAMATH_CALUDE_total_production_is_29621_l293_29335

/-- Represents the production numbers for a specific region -/
structure RegionProduction where
  sedans : Nat
  suvs : Nat
  pickups : Nat

/-- Calculates the total production for a region -/
def total_region_production (r : RegionProduction) : Nat :=
  r.sedans + r.suvs + r.pickups

/-- Represents the production data for all regions -/
structure GlobalProduction where
  north_america : RegionProduction
  europe : RegionProduction
  asia : RegionProduction
  south_america : RegionProduction

/-- Calculates the total global production -/
def total_global_production (g : GlobalProduction) : Nat :=
  total_region_production g.north_america +
  total_region_production g.europe +
  total_region_production g.asia +
  total_region_production g.south_america

/-- The production data for the 5-month period -/
def production_data : GlobalProduction := {
  north_america := { sedans := 3884, suvs := 2943, pickups := 1568 }
  europe := { sedans := 2871, suvs := 2145, pickups := 643 }
  asia := { sedans := 5273, suvs := 3881, pickups := 2338 }
  south_america := { sedans := 1945, suvs := 1365, pickups := 765 }
}

/-- Theorem stating that the total global production equals 29621 -/
theorem total_production_is_29621 :
  total_global_production production_data = 29621 := by
  sorry

end NUMINAMATH_CALUDE_total_production_is_29621_l293_29335


namespace NUMINAMATH_CALUDE_no_square_with_two_or_three_ones_l293_29337

/-- Represents a number in base-10 using only 0 and 1 digits -/
def IsBaseOneZero (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- Counts the number of ones in the base-10 representation of a number -/
def CountOnes (n : ℕ) : ℕ :=
  (n.digits 10).filter (· = 1) |>.length

/-- Main theorem: No square number exists with only 0 and 1 digits and exactly 2 or 3 ones -/
theorem no_square_with_two_or_three_ones :
  ¬ ∃ n : ℕ, IsBaseOneZero (n^2) ∧ (CountOnes (n^2) = 2 ∨ CountOnes (n^2) = 3) :=
sorry

end NUMINAMATH_CALUDE_no_square_with_two_or_three_ones_l293_29337


namespace NUMINAMATH_CALUDE_platform_length_calculation_l293_29375

/-- Calculates the length of a platform given train parameters -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 39 →
  time_pole = 18 →
  ∃ platform_length : ℝ,
    (platform_length > 348) ∧ (platform_length < 349) ∧
    (train_length + platform_length) / time_platform = train_length / time_pole :=
by
  sorry

#check platform_length_calculation

end NUMINAMATH_CALUDE_platform_length_calculation_l293_29375


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l293_29349

theorem max_value_theorem (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) ≤ Real.sqrt 29 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 2) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achievable_l293_29349


namespace NUMINAMATH_CALUDE_probability_sum_five_l293_29389

/-- The probability of obtaining a sum of 5 when rolling two dice of different sizes simultaneously -/
theorem probability_sum_five (total_outcomes : ℕ) (favorable_outcomes : ℕ) : 
  total_outcomes = 36 → favorable_outcomes = 4 → (favorable_outcomes : ℚ) / total_outcomes = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_five_l293_29389


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l293_29353

def U : Finset Nat := {1,2,3,4,5,6}
def A : Finset Nat := {1,3,5}
def B : Finset Nat := {1,4}

theorem intersection_A_complement_B : A ∩ (U \ B) = {3,5} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l293_29353


namespace NUMINAMATH_CALUDE_faye_bought_30_songs_l293_29367

/-- The number of songs Faye bought -/
def total_songs (country_albums pop_albums songs_per_album : ℕ) : ℕ :=
  (country_albums + pop_albums) * songs_per_album

/-- Proof that Faye bought 30 songs -/
theorem faye_bought_30_songs :
  total_songs 2 3 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_faye_bought_30_songs_l293_29367


namespace NUMINAMATH_CALUDE_solution_sets_equivalence_l293_29317

open Set

-- Define the solution set of the first inequality
def solution_set_1 : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 3}

-- Define the coefficients a, b, c based on the given conditions
def a : ℝ := -1  -- Assume a = -1 for simplicity, since we know a < 0
def b : ℝ := -2 * a
def c : ℝ := -3 * a

-- Define the solution set of the second inequality
def solution_set_2 : Set ℝ := {x : ℝ | -1/3 < x ∧ x < 1}

-- Theorem statement
theorem solution_sets_equivalence : 
  (∀ x : ℝ, x ∈ solution_set_1 ↔ a * x^2 + b * x + c ≤ 0) →
  (∀ x : ℝ, x ∈ solution_set_2 ↔ c * x^2 - b * x + a < 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_sets_equivalence_l293_29317


namespace NUMINAMATH_CALUDE_greatest_distance_C_D_l293_29320

def C : Set ℂ := {z : ℂ | z^3 = 1}

def D : Set ℂ := {z : ℂ | z^3 - 27*z^2 + 27*z - 1 = 0}

theorem greatest_distance_C_D : 
  ∃ (c : ℂ) (d : ℂ), c ∈ C ∧ d ∈ D ∧ 
    ∀ (c' : ℂ) (d' : ℂ), c' ∈ C → d' ∈ D → 
      Complex.abs (c - d) ≥ Complex.abs (c' - d') ∧
      Complex.abs (c - d) = Real.sqrt (184.5 + 60 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_greatest_distance_C_D_l293_29320


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l293_29341

/-- An isosceles right triangle with hypotenuse 68 and leg 48 -/
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  leg : ℝ
  hypotenuse_eq : hypotenuse = 68
  leg_eq : leg = 48

/-- A circle inscribed in the right angle of the triangle -/
structure InscribedCircle where
  radius : ℝ
  radius_eq : radius = 12

/-- A circle externally tangent to the inscribed circle and inscribed in the remaining space -/
structure TangentCircle where
  radius : ℝ

/-- The main theorem stating that the radius of the tangent circle is 8 -/
theorem tangent_circle_radius 
  (triangle : IsoscelesRightTriangle) 
  (inscribed : InscribedCircle) 
  (tangent : TangentCircle) : tangent.radius = 8 := by
  sorry


end NUMINAMATH_CALUDE_tangent_circle_radius_l293_29341


namespace NUMINAMATH_CALUDE_no_real_roots_l293_29342

theorem no_real_roots (k : ℝ) (h : 12 - 3 * k < 0) : 
  ∀ x : ℝ, x^2 + 4*x + k ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_l293_29342


namespace NUMINAMATH_CALUDE_ellipse_equation_l293_29370

/-- The standard equation of an ellipse with given foci and major axis length -/
theorem ellipse_equation (a b c : ℝ) (h1 : c^2 = 5) (h2 : a = 5) (h3 : b^2 = a^2 - c^2) :
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 25) + (y^2 / 20) = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l293_29370


namespace NUMINAMATH_CALUDE_collinear_vector_combinations_l293_29308

/-- Given two non-zero vectors in a real vector space that are not collinear,
    if a linear combination of these vectors with scalar k is collinear with
    another linear combination of the same vectors where k's role is swapped,
    then k must be either 1 or -1. -/
theorem collinear_vector_combinations (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (e₁ e₂ : V) (k : ℝ) 
  (h_nonzero₁ : e₁ ≠ 0)
  (h_nonzero₂ : e₂ ≠ 0)
  (h_not_collinear : ¬ ∃ (c : ℝ), e₁ = c • e₂)
  (h_collinear : ∃ (t : ℝ), k • e₁ + e₂ = t • (e₁ + k • e₂)) :
  k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_collinear_vector_combinations_l293_29308


namespace NUMINAMATH_CALUDE_maries_school_students_maries_school_students_proof_l293_29319

theorem maries_school_students : ℕ → ℕ → Prop :=
  fun m c =>
    m = 4 * c ∧ m + c = 2500 → m = 2000

-- The proof is omitted
theorem maries_school_students_proof : maries_school_students 2000 500 := by
  sorry

end NUMINAMATH_CALUDE_maries_school_students_maries_school_students_proof_l293_29319


namespace NUMINAMATH_CALUDE_exponent_division_l293_29346

theorem exponent_division (a : ℝ) (m n : ℕ) (h : m > n) :
  a^m / a^n = a^(m - n) := by sorry

end NUMINAMATH_CALUDE_exponent_division_l293_29346


namespace NUMINAMATH_CALUDE_road_repair_hours_proof_l293_29343

/-- The number of hours the first group works per day to repair a road -/
def hours_per_day : ℕ := 5

/-- The number of people in the first group -/
def people_group1 : ℕ := 39

/-- The number of days the first group works -/
def days_group1 : ℕ := 12

/-- The number of people in the second group -/
def people_group2 : ℕ := 30

/-- The number of days the second group works -/
def days_group2 : ℕ := 13

/-- The number of hours per day the second group works -/
def hours_group2 : ℕ := 6

theorem road_repair_hours_proof :
  people_group1 * days_group1 * hours_per_day = people_group2 * days_group2 * hours_group2 :=
by sorry

end NUMINAMATH_CALUDE_road_repair_hours_proof_l293_29343


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l293_29313

theorem two_digit_number_problem (x : ℕ) : x ≥ 10 ∧ x ≤ 99 → 500 + x = 9 * x - 12 → x = 64 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l293_29313


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l293_29355

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℝ, (x + 3) * (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x - 6)) → 
  p = -31 ∧ q = -71 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l293_29355


namespace NUMINAMATH_CALUDE_coordinates_of_M_l293_29360

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Check if a point is on the angle bisector of the first and third quadrants -/
def isOnAngleBisector (p : Point) : Prop := p.x = p.y

/-- Given point M with coordinates (2-m, 1+2m) -/
def M (m : ℝ) : Point := ⟨2 - m, 1 + 2*m⟩

theorem coordinates_of_M (m : ℝ) :
  (distanceToYAxis (M m) = 3 → (M m = ⟨3, -1⟩ ∨ M m = ⟨-3, 11⟩)) ∧
  (isOnAngleBisector (M m) → M m = ⟨5/3, 5/3⟩) := by
  sorry

end NUMINAMATH_CALUDE_coordinates_of_M_l293_29360


namespace NUMINAMATH_CALUDE_y_value_proof_l293_29392

theorem y_value_proof : ∀ y : ℚ, (1/4 - 1/5 = 4/y) → y = 80 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l293_29392


namespace NUMINAMATH_CALUDE_code_deciphering_probability_l293_29331

theorem code_deciphering_probability 
  (prob_A : ℚ) 
  (prob_B : ℚ) 
  (h_A : prob_A = 2 / 3) 
  (h_B : prob_B = 3 / 5) : 
  1 - (1 - prob_A) * (1 - prob_B) = 13 / 15 := by
  sorry

end NUMINAMATH_CALUDE_code_deciphering_probability_l293_29331


namespace NUMINAMATH_CALUDE_box_width_calculation_l293_29351

theorem box_width_calculation (length depth : ℕ) (total_cubes : ℕ) (width : ℕ) : 
  length = 49 → 
  depth = 14 → 
  total_cubes = 84 → 
  (∃ (cube_side : ℕ), 
    cube_side > 0 ∧ 
    length % cube_side = 0 ∧ 
    depth % cube_side = 0 ∧ 
    width % cube_side = 0 ∧
    (length / cube_side) * (depth / cube_side) * (width / cube_side) = total_cubes) →
  width = 42 := by
sorry

end NUMINAMATH_CALUDE_box_width_calculation_l293_29351


namespace NUMINAMATH_CALUDE_tea_house_payment_l293_29304

theorem tea_house_payment (t k b : ℕ+) (h : 11 ∣ (3 * t + 4 * k + 5 * b)) :
  11 ∣ (9 * t + k + 4 * b) := by
  sorry

end NUMINAMATH_CALUDE_tea_house_payment_l293_29304


namespace NUMINAMATH_CALUDE_fraction_meaningful_condition_l293_29363

theorem fraction_meaningful_condition (x : ℝ) : 
  (∃ y : ℝ, y = (x + 2) / (x - 1)) ↔ x ≠ 1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_condition_l293_29363


namespace NUMINAMATH_CALUDE_max_value_of_g_l293_29357

def S : Set Int := {-3, -2, 1, 2, 3, 4}

def g (a b : Int) : ℚ := -((a - b)^2 : ℚ) / 4

theorem max_value_of_g :
  ∃ (max : ℚ), max = -1/4 ∧
  ∀ (a b : Int), a ∈ S → b ∈ S → a ≠ b → g a b ≤ max ∧
  ∃ (a₀ b₀ : Int), a₀ ∈ S ∧ b₀ ∈ S ∧ a₀ ≠ b₀ ∧ g a₀ b₀ = max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l293_29357


namespace NUMINAMATH_CALUDE_inequality_proof_l293_29348

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) :
  1 + a + b > 3 * Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l293_29348


namespace NUMINAMATH_CALUDE_round_trip_distance_l293_29395

/-- Calculates the distance of a round trip given upstream speed, downstream speed, and total time -/
theorem round_trip_distance 
  (upstream_speed : ℝ) 
  (downstream_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : upstream_speed > 0)
  (h2 : downstream_speed > 0)
  (h3 : total_time > 0)
  (h4 : upstream_speed = 3)
  (h5 : downstream_speed = 9)
  (h6 : total_time = 8) :
  (let distance := (upstream_speed * downstream_speed * total_time) / (upstream_speed + downstream_speed)
   distance = 18) := by
  sorry

end NUMINAMATH_CALUDE_round_trip_distance_l293_29395


namespace NUMINAMATH_CALUDE_translation_product_l293_29377

/-- Given a point P(-3, y) translated 3 units down and 2 units left to obtain point Q(x, -1), 
    the product xy equals -10. -/
theorem translation_product (y : ℝ) : 
  let x : ℝ := -3 - 2
  let y' : ℝ := y - 3
  x * y = -10 ∧ y' = -1 := by sorry

end NUMINAMATH_CALUDE_translation_product_l293_29377


namespace NUMINAMATH_CALUDE_two_crayons_per_color_per_box_l293_29382

/-- Represents a crayon factory with given production parameters -/
structure CrayonFactory where
  colors : ℕ
  boxes_per_hour : ℕ
  total_crayons : ℕ
  total_hours : ℕ

/-- Calculates the number of crayons of each color per box -/
def crayons_per_color_per_box (factory : CrayonFactory) : ℕ :=
  factory.total_crayons / (factory.boxes_per_hour * factory.total_hours * factory.colors)

/-- Theorem stating that for the given factory parameters, there are 2 crayons of each color per box -/
theorem two_crayons_per_color_per_box (factory : CrayonFactory) 
  (h1 : factory.colors = 4)
  (h2 : factory.boxes_per_hour = 5)
  (h3 : factory.total_crayons = 160)
  (h4 : factory.total_hours = 4) :
  crayons_per_color_per_box factory = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_crayons_per_color_per_box_l293_29382


namespace NUMINAMATH_CALUDE_cookie_averages_l293_29311

def brand_x_packages : List ℕ := [6, 8, 9, 11, 13]
def brand_y_packages : List ℕ := [14, 15, 18, 20]

theorem cookie_averages :
  let x_total := brand_x_packages.sum
  let y_total := brand_y_packages.sum
  let x_avg : ℚ := x_total / brand_x_packages.length
  let y_avg : ℚ := y_total / brand_y_packages.length
  x_avg = 47 / 5 ∧ y_avg = 67 / 4 := by
  sorry

end NUMINAMATH_CALUDE_cookie_averages_l293_29311


namespace NUMINAMATH_CALUDE_polygon_sides_l293_29378

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1260 → ∃ n : ℕ, n = 9 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l293_29378


namespace NUMINAMATH_CALUDE_rectangle_perimeter_problem_l293_29374

theorem rectangle_perimeter_problem :
  ∀ (a b : ℕ),
    a ≠ b →
    a * b = 2 * (2 * a + 2 * b) →
    2 * (a + b) = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_problem_l293_29374


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l293_29358

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = 4) 
  (h_9 : a 9 = 10) : 
  a 15 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l293_29358


namespace NUMINAMATH_CALUDE_factorial_ratio_l293_29326

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l293_29326


namespace NUMINAMATH_CALUDE_kolya_purchase_l293_29364

/-- Represents the cost of an item in kopecks -/
def item_cost (rubles : ℕ) : ℕ := 100 * rubles + 99

/-- Represents the total purchase cost in kopecks -/
def total_cost : ℕ := 200 * 100 + 83

/-- Predicate to check if a given number of items is a valid solution -/
def is_valid_solution (n : ℕ) : Prop :=
  ∃ (rubles : ℕ), n * (item_cost rubles) = total_cost

theorem kolya_purchase :
  ∀ n : ℕ, is_valid_solution n ↔ n = 17 ∨ n = 117 :=
sorry

end NUMINAMATH_CALUDE_kolya_purchase_l293_29364


namespace NUMINAMATH_CALUDE_cookie_problem_l293_29306

/-- Cookie problem statement -/
theorem cookie_problem (alyssa_cookies aiyanna_cookies brady_cookies : ℕ) 
  (h1 : alyssa_cookies = 1523)
  (h2 : aiyanna_cookies = 3720)
  (h3 : brady_cookies = 2265) :
  (aiyanna_cookies - alyssa_cookies = 2197) ∧ 
  (aiyanna_cookies - brady_cookies = 1455) ∧ 
  (brady_cookies - alyssa_cookies = 742) := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l293_29306


namespace NUMINAMATH_CALUDE_negation_forall_squared_gt_neg_one_negation_exists_squared_leq_nine_abs_gt_not_necessary_for_gt_m_lt_zero_iff_one_positive_one_negative_root_l293_29333

-- Statement 1
theorem negation_forall_squared_gt_neg_one :
  (¬ ∀ x : ℝ, x^2 > -1) ↔ (∃ x : ℝ, x^2 ≤ -1) := by sorry

-- Statement 2
theorem negation_exists_squared_leq_nine :
  (¬ ∃ x : ℝ, x > -3 ∧ x^2 ≤ 9) ↔ (∀ x : ℝ, x > -3 → x^2 > 9) := by sorry

-- Statement 3
theorem abs_gt_not_necessary_for_gt :
  ∃ x y : ℝ, (abs x > abs y) ∧ (x ≤ y) := by sorry

-- Statement 4
theorem m_lt_zero_iff_one_positive_one_negative_root :
  ∀ m : ℝ, (m < 0) ↔ 
    (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - 2*x + m = 0 ∧ y^2 - 2*y + m = 0 ∧ 
      (∀ z : ℝ, z^2 - 2*z + m = 0 → z = x ∨ z = y)) := by sorry

end NUMINAMATH_CALUDE_negation_forall_squared_gt_neg_one_negation_exists_squared_leq_nine_abs_gt_not_necessary_for_gt_m_lt_zero_iff_one_positive_one_negative_root_l293_29333


namespace NUMINAMATH_CALUDE_perpendicular_slope_is_five_thirds_l293_29365

/-- The slope of a line perpendicular to the line containing points (3, 5) and (-2, 8) is 5/3 -/
theorem perpendicular_slope_is_five_thirds :
  let point1 : ℝ × ℝ := (3, 5)
  let point2 : ℝ × ℝ := (-2, 8)
  let slope_original := (point2.2 - point1.2) / (point2.1 - point1.1)
  let slope_perpendicular := -1 / slope_original
  slope_perpendicular = 5/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_slope_is_five_thirds_l293_29365
