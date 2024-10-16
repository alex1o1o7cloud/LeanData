import Mathlib

namespace NUMINAMATH_CALUDE_employee_assignment_l154_15434

/-- The number of ways to assign employees to workshops -/
def assign_employees (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  (n.choose (k - 1)) * (k.factorial)

/-- Theorem: Assigning 5 employees to 3 workshops with constraints -/
theorem employee_assignment :
  let total_employees : ℕ := 5
  let workshops : ℕ := 3
  let effective_employees : ℕ := total_employees - 1  -- Considering A and B as one entity
  assign_employees effective_employees workshops workshops = 36 := by
  sorry


end NUMINAMATH_CALUDE_employee_assignment_l154_15434


namespace NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l154_15417

theorem diagonal_not_parallel_to_sides (n : ℕ) (h : n > 1) :
  n * (2 * n - 3) > 2 * n * (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_not_parallel_to_sides_l154_15417


namespace NUMINAMATH_CALUDE_complement_union_eq_specific_set_l154_15405

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_union_eq_specific_set :
  (U \ (M ∪ N)) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_eq_specific_set_l154_15405


namespace NUMINAMATH_CALUDE_stability_comparison_l154_15407

/-- Represents the variance of a student's performance -/
structure StudentVariance where
  value : ℝ
  positive : value > 0

/-- Defines the concept of stability based on variance -/
def more_stable (a b : StudentVariance) : Prop :=
  a.value < b.value

theorem stability_comparison 
  (variance_A variance_B : StudentVariance)
  (h1 : variance_A.value = 0.05)
  (h2 : variance_B.value = 0.06) :
  more_stable variance_A variance_B :=
sorry

end NUMINAMATH_CALUDE_stability_comparison_l154_15407


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l154_15492

/-- The function f(x) = 4x^2 - kx - 8 is monotonic on [5, 8] iff k ∈ (-∞, 40] ∪ [64, +∞) -/
theorem monotonic_quadratic_function (k : ℝ) :
  (∀ x ∈ Set.Icc 5 8, Monotone (fun x => 4 * x^2 - k * x - 8)) ↔ 
  (k ≤ 40 ∨ k ≥ 64) := by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l154_15492


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l154_15438

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 6}
def B : Set ℕ := {2, 4, 5, 6}

theorem intersection_complement_equals_set (h : Set ℕ) : A ∩ (U \ B) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l154_15438


namespace NUMINAMATH_CALUDE_trig_equation_solution_l154_15418

theorem trig_equation_solution (x : ℝ) : 
  1 - Real.cos (6 * x) = Real.tan (3 * x) ↔ 
  (∃ k : ℤ, x = k * Real.pi / 3) ∨ 
  (∃ k : ℤ, x = Real.pi / 12 * (4 * k + 1)) := by
sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l154_15418


namespace NUMINAMATH_CALUDE_inequality_solutions_l154_15475

theorem inequality_solutions :
  (∀ x : ℝ, 2*x - 1 > x - 3 ↔ x > -2) ∧
  (∀ x : ℝ, x - 3*(x - 2) ≥ 4 ∧ (x - 1)/5 < (x + 1)/2 ↔ -7/3 < x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solutions_l154_15475


namespace NUMINAMATH_CALUDE_expansion_properties_l154_15412

def sum_of_coefficients (n : ℕ) : ℝ := (3 + 1)^n

def sum_of_binomial_coefficients (n : ℕ) : ℝ := 2^n

theorem expansion_properties (n : ℕ) 
  (h : sum_of_coefficients n - sum_of_binomial_coefficients n = 240) :
  n = 4 ∧ 
  ∃ (a b c : ℝ), a = 81 ∧ b = 54 ∧ c = 1 ∧
  (∀ (x : ℝ), (3*x + x^(1/2))^n = a*x^4 + b*x^3 + c*x^2 + x^(7/2) + 6*x^(5/2) + 4*x^(3/2) + x^(1/2)) :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l154_15412


namespace NUMINAMATH_CALUDE_seats_per_row_is_eight_l154_15457

/-- The number of people that fit in a row on an airplane, given the specified conditions. -/
def seats_per_row : ℕ := by
  sorry

theorem seats_per_row_is_eight :
  let total_rows : ℕ := 12
  let occupancy_rate : ℚ := 3/4
  let unoccupied_seats : ℕ := 24
  seats_per_row = 8 := by
  sorry

end NUMINAMATH_CALUDE_seats_per_row_is_eight_l154_15457


namespace NUMINAMATH_CALUDE_sam_has_five_dimes_l154_15414

/-- Represents the number of dimes Sam has at different stages -/
structure DimeCount where
  initial : ℕ
  after_sister_borrows : ℕ
  after_friend_borrows : ℕ
  after_sister_returns : ℕ
  after_friend_returns : ℕ

/-- Calculates the final number of dimes Sam has -/
def final_dime_count (d : DimeCount) : ℕ :=
  d.initial - 4 - 2 + 2 + 1

/-- Theorem stating that Sam ends up with 5 dimes -/
theorem sam_has_five_dimes (d : DimeCount) 
  (h_initial : d.initial = 8)
  (h_sister_borrows : d.after_sister_borrows = d.initial - 4)
  (h_friend_borrows : d.after_friend_borrows = d.after_sister_borrows - 2)
  (h_sister_returns : d.after_sister_returns = d.after_friend_borrows + 2)
  (h_friend_returns : d.after_friend_returns = d.after_sister_returns + 1) :
  final_dime_count d = 5 := by
  sorry

end NUMINAMATH_CALUDE_sam_has_five_dimes_l154_15414


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l154_15444

-- Define the function f
def f (x m : ℝ) : ℝ := |2 * x - m|

-- State the theorem
theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, f x m ≤ 6 ↔ -2 ≤ x ∧ x ≤ 4) →
  (m = 2 ∧
   ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 2 →
     (∀ x : ℝ, f x m + f ((1/2) * x + 3) m ≤ 8/a + 2/b ↔ -3 ≤ x ∧ x ≤ 7/3)) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l154_15444


namespace NUMINAMATH_CALUDE_orthogonal_centers_eq_radical_axis_l154_15454

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the orthogonality condition for circles
def is_orthogonal (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = c1.radius^2 + c2.radius^2

-- Define the radical axis of two circles
def radical_axis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               let (x1, y1) := c1.center
               let (x2, y2) := c2.center
               (x - x1)^2 + (y - y1)^2 - c1.radius^2 = 
               (x - x2)^2 + (y - y2)^2 - c2.radius^2}

-- Define the set of centers of circles orthogonal to both given circles
def orthogonal_centers (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (r : ℝ), is_orthogonal (Circle.mk p r) c1 ∧
                           is_orthogonal (Circle.mk p r) c2}

-- Define the common chord of two intersecting circles
def common_chord (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               let (x1, y1) := c1.center
               let (x2, y2) := c2.center
               (x - x1)^2 + (y - y1)^2 = c1.radius^2 ∧
               (x - x2)^2 + (y - y2)^2 = c2.radius^2}

-- Theorem statement
theorem orthogonal_centers_eq_radical_axis (c1 c2 : Circle) 
  (h : c1.center ≠ c2.center) : 
  orthogonal_centers c1 c2 = radical_axis c1 c2 \ common_chord c1 c2 :=
by sorry

end NUMINAMATH_CALUDE_orthogonal_centers_eq_radical_axis_l154_15454


namespace NUMINAMATH_CALUDE_chord_length_and_circle_M_equation_l154_15453

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 8

-- Define point P0
def P0 : ℝ × ℝ := (-1, 2)

-- Define point C
def C : ℝ × ℝ := (3, 0)

-- Define the angle of inclination
def alpha : ℝ := 135

-- Define the chord AB
def chord_AB (x y : ℝ) : Prop :=
  y = -x + 1 ∧ circle_equation x y

-- Define circle M
def circle_M (x y : ℝ) : Prop :=
  (x - 1/4)^2 + (y + 1/2)^2 = 125/16

theorem chord_length_and_circle_M_equation :
  (∃ A B : ℝ × ℝ, 
    chord_AB A.1 A.2 ∧ 
    chord_AB B.1 B.2 ∧ 
    P0 = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 30) ∧
  (∀ x y : ℝ, circle_M x y ↔ 
    (∃ A B : ℝ × ℝ, 
      chord_AB A.1 A.2 ∧ 
      chord_AB B.1 B.2 ∧ 
      P0 = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
      circle_M C.1 C.2 ∧
      (∀ t : ℝ, circle_M (A.1 + t*(B.1 - A.1)) (A.2 + t*(B.2 - A.2)) → t = 1/2))) := by
  sorry

end NUMINAMATH_CALUDE_chord_length_and_circle_M_equation_l154_15453


namespace NUMINAMATH_CALUDE_second_most_frequent_is_23_l154_15429

-- Define the function m(i) which represents the number of drawings where i appears in the second position
def m (i : ℕ) : ℕ := 
  if 2 ≤ i ∧ i ≤ 87 then
    (i - 1) * (90 - i).choose 3
  else
    0

-- Define the lottery parameters
def lotterySize : ℕ := 6
def lotteryRange : ℕ := 90

-- Theorem statement
theorem second_most_frequent_is_23 : 
  ∀ i, 2 ≤ i ∧ i ≤ 87 → m i ≤ m 23 :=
sorry

end NUMINAMATH_CALUDE_second_most_frequent_is_23_l154_15429


namespace NUMINAMATH_CALUDE_initial_boarders_l154_15433

/-- Proves that the initial number of boarders was 150 given the conditions of the problem -/
theorem initial_boarders (B D : ℕ) (h1 : B * 12 = D * 5) 
  (h2 : (B + 30) * 2 = D * 1) : B = 150 := by
  sorry

end NUMINAMATH_CALUDE_initial_boarders_l154_15433


namespace NUMINAMATH_CALUDE_binary_1011011_equals_base7_160_l154_15462

/-- Converts a binary number represented as a list of bits to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a natural number to its representation in base 7. -/
def nat_to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: nat_to_base7 (n / 7)

/-- The binary representation of 1011011. -/
def binary_1011011 : List Bool :=
  [true, false, true, true, false, true, true]

/-- The base 7 representation of 160. -/
def base7_160 : List ℕ :=
  [0, 6, 1]

theorem binary_1011011_equals_base7_160 :
  nat_to_base7 (binary_to_nat binary_1011011) = base7_160 := by
  sorry

#eval binary_to_nat binary_1011011
#eval nat_to_base7 (binary_to_nat binary_1011011)

end NUMINAMATH_CALUDE_binary_1011011_equals_base7_160_l154_15462


namespace NUMINAMATH_CALUDE_inequality_proof_l154_15490

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a + c) * (b + d)) ≥ Real.sqrt (a * b) + Real.sqrt (c * d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l154_15490


namespace NUMINAMATH_CALUDE_solution_distribution_l154_15478

theorem solution_distribution (num_tubes : ℕ) (num_beakers : ℕ) (beaker_volume : ℚ) :
  num_tubes = 6 →
  num_beakers = 3 →
  beaker_volume = 14 →
  (num_beakers * beaker_volume) / num_tubes = 7 :=
by sorry

end NUMINAMATH_CALUDE_solution_distribution_l154_15478


namespace NUMINAMATH_CALUDE_circle_graph_percentage_l154_15443

theorem circle_graph_percentage (sector_degrees : ℝ) (total_degrees : ℝ) 
  (h1 : sector_degrees = 18)
  (h2 : total_degrees = 360) :
  (sector_degrees / total_degrees) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_circle_graph_percentage_l154_15443


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l154_15425

theorem complex_fraction_equality : 2 + (3 / (4 + (5/6))) = 76/29 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l154_15425


namespace NUMINAMATH_CALUDE_twentyFifthBaseSum4_l154_15451

/-- Converts a natural number to its base 4 representation --/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

/-- Calculates the sum of digits in a list --/
def sumDigits (l : List ℕ) : ℕ :=
  l.sum

theorem twentyFifthBaseSum4 :
  let base4Rep := toBase4 25
  base4Rep = [1, 2, 1] ∧ sumDigits base4Rep = 4 := by sorry

end NUMINAMATH_CALUDE_twentyFifthBaseSum4_l154_15451


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l154_15476

/-- Properties of an acute triangle ABC with specific conditions -/
theorem acute_triangle_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π / 2 ∧
  0 < B ∧ B < π / 2 ∧
  0 < C ∧ C < π / 2 ∧
  -- Sum of angles in a triangle is π
  A + B + C = π ∧
  -- Side lengths are positive
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  -- Given condition for b
  b = 2 * Real.sqrt 6 ∧
  -- Sine rule
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  -- Angle bisector theorem
  (Real.sqrt 3 * a * c) / (a + c) = b * Real.sin (B / 2) / Real.sin B →
  -- Conclusions to prove
  π / 6 < C ∧ C < π / 2 ∧
  2 * Real.sqrt 2 < c ∧ c < 4 * Real.sqrt 2 ∧
  16 < a * c ∧ a * c ≤ 24 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l154_15476


namespace NUMINAMATH_CALUDE_line_through_points_l154_15406

theorem line_through_points (a : ℝ) : 
  a > 0 ∧ 
  (∃ m b : ℝ, m = 2 ∧ b = 0 ∧ 
    (5 = m * a + b) ∧ 
    (a = m * 2 + b)) →
  a = 3 :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l154_15406


namespace NUMINAMATH_CALUDE_tangent_line_value_l154_15491

/-- A line passing through point P(1, 2) is tangent to the circle x^2 + y^2 = 4 
    and perpendicular to the line ax - y + 1 = 0. The value of a is -3/4. -/
theorem tangent_line_value (a : ℝ) : 
  (∃ (m : ℝ), 
    -- Line equation: y - 2 = m(x - 1)
    (∀ x y : ℝ, y - 2 = m * (x - 1) → 
      -- Point P(1, 2) satisfies the line equation
      (1 : ℝ) - 1 = 0 ∧ 2 - 2 = 0 ∧
      -- Line is tangent to the circle
      ((x - 0)^2 + (y - 0)^2 = 4 → 
        (y - 2 = m * (x - 1) → x^2 + y^2 ≥ 4)) ∧
      -- Line is perpendicular to ax - y + 1 = 0
      m * a = -1)) →
  a = -3/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_value_l154_15491


namespace NUMINAMATH_CALUDE_complex_division_simplification_l154_15477

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_division_simplification :
  (2 + i) / i = 1 - 2*i :=
by sorry

end NUMINAMATH_CALUDE_complex_division_simplification_l154_15477


namespace NUMINAMATH_CALUDE_factorization_problem_1_l154_15423

theorem factorization_problem_1 (a x : ℝ) : 3*a*x^2 - 6*a*x + 3*a = 3*a*(x-1)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_l154_15423


namespace NUMINAMATH_CALUDE_zoo_admission_solution_l154_15468

/-- Represents the zoo admission problem for two classes -/
structure ZooAdmission where
  ticket_price : ℕ
  class_a_students : ℕ
  class_b_students : ℕ

/-- Calculates the number of free tickets given the total number of students -/
def free_tickets (n : ℕ) : ℕ :=
  n / 5

/-- Calculates the total cost for a class given the number of students and ticket price -/
def class_cost (students : ℕ) (price : ℕ) : ℕ :=
  (students - free_tickets students) * price

/-- The main theorem representing the zoo admission problem -/
theorem zoo_admission_solution :
  ∃ (za : ZooAdmission),
    za.ticket_price > 0 ∧
    class_cost za.class_a_students za.ticket_price = 1995 ∧
    class_cost (za.class_a_students + za.class_b_students) za.ticket_price = 4410 ∧
    za.class_a_students = 23 ∧
    za.class_b_students = 29 ∧
    za.ticket_price = 105 :=
  sorry

end NUMINAMATH_CALUDE_zoo_admission_solution_l154_15468


namespace NUMINAMATH_CALUDE_circle_area_decrease_l154_15487

/-- Given a circle with initial area 25π, prove that a 10% decrease in diameter results in a 19% decrease in area -/
theorem circle_area_decrease (π : ℝ) (h_π : π > 0) : 
  let initial_area : ℝ := 25 * π
  let initial_radius : ℝ := (initial_area / π).sqrt
  let new_radius : ℝ := initial_radius * 0.9
  let new_area : ℝ := π * new_radius^2
  (initial_area - new_area) / initial_area = 0.19 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_decrease_l154_15487


namespace NUMINAMATH_CALUDE_quadratic_range_on_interval_l154_15420

/-- The range of a quadratic function on a closed interval -/
theorem quadratic_range_on_interval (a b c : ℝ) (h : a < 0) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c
  let vertex_x : ℝ := -b / (2 * a)
  let range : Set ℝ := Set.range (fun x ↦ f x)
  (Set.Icc 0 2).image f =
    if 0 ≤ vertex_x ∧ vertex_x ≤ 2 then
      Set.Icc (4 * a + 2 * b + c) (-b^2 / (4 * a) + c)
    else
      Set.Icc (4 * a + 2 * b + c) c := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_on_interval_l154_15420


namespace NUMINAMATH_CALUDE_star_divided_by_square_equals_sixteen_l154_15481

-- Define the symbols as real numbers
variable (triangle circle square star : ℝ)

-- State the conditions
axiom triangle_plus_triangle : triangle + triangle = star
axiom circle_equals_square_plus_square : circle = square + square
axiom triangle_equals_four_circles : triangle = circle + circle + circle + circle

-- State the theorem to be proved
theorem star_divided_by_square_equals_sixteen :
  star / square = 16 := by sorry

end NUMINAMATH_CALUDE_star_divided_by_square_equals_sixteen_l154_15481


namespace NUMINAMATH_CALUDE_emails_evening_l154_15404

def emails_problem (afternoon evening morning total : ℕ) : Prop :=
  afternoon = 3 ∧ morning = 6 ∧ total = 10 ∧ afternoon + evening + morning = total

theorem emails_evening : ∃ evening : ℕ, emails_problem 3 evening 6 10 ∧ evening = 1 :=
  sorry

end NUMINAMATH_CALUDE_emails_evening_l154_15404


namespace NUMINAMATH_CALUDE_binomial_26_6_l154_15430

theorem binomial_26_6 (h1 : Nat.choose 24 5 = 42504) (h2 : Nat.choose 24 6 = 134596) :
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l154_15430


namespace NUMINAMATH_CALUDE_therapy_hours_is_five_l154_15445

/-- Represents the pricing structure and charges for therapy sessions -/
structure TherapyPricing where
  firstHourPrice : ℕ
  additionalHourPrice : ℕ
  firstPatientTotalCharge : ℕ
  threeHourCharge : ℕ

/-- Calculates the number of therapy hours for the first patient -/
def calculateTherapyHours (pricing : TherapyPricing) : ℕ :=
  sorry

/-- Theorem stating that the calculated number of therapy hours is 5 -/
theorem therapy_hours_is_five (pricing : TherapyPricing) 
  (h1 : pricing.firstHourPrice = pricing.additionalHourPrice + 30)
  (h2 : pricing.threeHourCharge = 252)
  (h3 : pricing.firstPatientTotalCharge = 400) : 
  calculateTherapyHours pricing = 5 := by
  sorry

end NUMINAMATH_CALUDE_therapy_hours_is_five_l154_15445


namespace NUMINAMATH_CALUDE_andrew_final_share_l154_15408

def total_stickers : ℕ := 2800

def ratio_sum : ℚ := 3/5 + 1 + 3/4 + 1/2 + 7/4

def andrew_initial_share : ℚ := (1 : ℚ) * total_stickers / ratio_sum

def sam_initial_share : ℚ := (3/4 : ℚ) * total_stickers / ratio_sum

def sam_to_andrew : ℚ := 0.4 * sam_initial_share

theorem andrew_final_share :
  ⌊andrew_initial_share + sam_to_andrew⌋ = 791 :=
sorry

end NUMINAMATH_CALUDE_andrew_final_share_l154_15408


namespace NUMINAMATH_CALUDE_arcade_spending_l154_15459

theorem arcade_spending (allowance : ℝ) (f : ℝ) : 
  allowance = 3.75 →
  (1 - f) * allowance - (1/3) * ((1 - f) * allowance) = 1 →
  f = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_arcade_spending_l154_15459


namespace NUMINAMATH_CALUDE_first_month_sale_l154_15480

def sales_4_months : List Int := [6927, 6855, 7230, 6562]
def average_6_months : Int := 6500
def sale_6th_month : Int := 4691
def num_months : Int := 6

theorem first_month_sale (sales_4_months : List Int) 
                         (average_6_months : Int) 
                         (sale_6th_month : Int) 
                         (num_months : Int) : 
  sales_4_months = [6927, 6855, 7230, 6562] →
  average_6_months = 6500 →
  sale_6th_month = 4691 →
  num_months = 6 →
  (List.sum sales_4_months + sale_6th_month + 6735) / num_months = average_6_months :=
by sorry

end NUMINAMATH_CALUDE_first_month_sale_l154_15480


namespace NUMINAMATH_CALUDE_parabola_properties_l154_15437

def f (x : ℝ) := -(x + 1)^2 + 3

theorem parabola_properties :
  (∀ x y : ℝ, f x ≤ f y → (x - (-1))^2 ≥ (y - (-1))^2) ∧ 
  (∀ x y : ℝ, x + y = -2 → f x = f y) ∧
  (f (-1) = 3 ∧ ∀ x : ℝ, f x ≤ f (-1)) ∧
  (∀ x y : ℝ, x > 1 ∧ y > 1 ∧ x > y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l154_15437


namespace NUMINAMATH_CALUDE_same_team_probability_l154_15426

/-- The probability of two students choosing the same team out of three teams -/
theorem same_team_probability (num_teams : ℕ) (h : num_teams = 3) :
  (num_teams : ℚ) / (num_teams^2 : ℚ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_same_team_probability_l154_15426


namespace NUMINAMATH_CALUDE_collinear_points_sum_l154_15495

/-- Three points in 3D space are collinear if they lie on the same line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop := sorry

/-- The main theorem: If the given points are collinear, then c + d = 6. -/
theorem collinear_points_sum (c d : ℝ) : 
  collinear (2, c, d) (c, 3, d) (c, d, 4) → c + d = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l154_15495


namespace NUMINAMATH_CALUDE_savings_proof_l154_15441

/-- Calculates savings given income and expenditure ratio -/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that savings are 4000 given the conditions -/
theorem savings_proof (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
  (h1 : income = 20000)
  (h2 : income_ratio = 5)
  (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 4000 := by
  sorry

#eval calculate_savings 20000 5 4

end NUMINAMATH_CALUDE_savings_proof_l154_15441


namespace NUMINAMATH_CALUDE_max_a_minus_b_is_seven_l154_15483

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

-- Define the theorem
theorem max_a_minus_b_is_seven :
  ∃ (a b : ℝ),
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, -3 ≤ a * f x + b ∧ a * f x + b ≤ 3) ∧
    (a - b = 7) ∧
    (∀ a' b' : ℝ, (∀ x ∈ Set.Icc (-1 : ℝ) 2, -3 ≤ a' * f x + b' ∧ a' * f x + b' ≤ 3) → a' - b' ≤ 7) :=
by sorry

end NUMINAMATH_CALUDE_max_a_minus_b_is_seven_l154_15483


namespace NUMINAMATH_CALUDE_fraction_equality_l154_15415

theorem fraction_equality (a b : ℝ) (h : (a - b) / a = 2 / 3) : b / a = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l154_15415


namespace NUMINAMATH_CALUDE_product_sum_inequality_l154_15484

theorem product_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  (a + b) * (b + c) * (c + a) ≥ 4 * (a + b + c - 1) := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l154_15484


namespace NUMINAMATH_CALUDE_rectangle_width_equals_six_l154_15400

theorem rectangle_width_equals_six (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ) : 
  square_side = 12 →
  rect_length = 24 →
  square_side * square_side = rect_length * rect_width →
  rect_width = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_equals_six_l154_15400


namespace NUMINAMATH_CALUDE_discount_percentage_retailer_discount_approx_25_percent_l154_15485

/-- Calculates the discount percentage given markup and profit percentages -/
theorem discount_percentage (markup : ℝ) (actual_profit : ℝ) : ℝ :=
  let marked_price := 1 + markup
  let actual_selling_price := 1 + actual_profit
  let discount := marked_price - actual_selling_price
  (discount / marked_price) * 100

/-- Proves that the discount percentage is approximately 25% given the specified markup and profit -/
theorem retailer_discount_approx_25_percent :
  ∀ (ε : ℝ), ε > 0 →
  abs (discount_percentage 0.60 0.20000000000000018 - 25) < ε :=
sorry

end NUMINAMATH_CALUDE_discount_percentage_retailer_discount_approx_25_percent_l154_15485


namespace NUMINAMATH_CALUDE_speed_limit_proof_l154_15435

/-- Prove that the speed limit is 50 mph given the conditions of Natasha's travel --/
theorem speed_limit_proof (natasha_speed : ℝ) (speed_limit : ℝ) (time : ℝ) (distance : ℝ) :
  natasha_speed = speed_limit + 10 →
  time = 1 →
  distance = 60 →
  natasha_speed = distance / time →
  speed_limit = 50 := by
sorry

end NUMINAMATH_CALUDE_speed_limit_proof_l154_15435


namespace NUMINAMATH_CALUDE_line_contains_point_l154_15469

/-- Proves that k = 11 for the line 3 - 3kx = 4y containing the point (1/3, -2) -/
theorem line_contains_point (k : ℝ) : 
  (3 - 3 * k * (1/3) = 4 * (-2)) → k = 11 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l154_15469


namespace NUMINAMATH_CALUDE_line_segment_param_sum_squares_l154_15432

/-- Given a line segment from (1,2) to (6,9) parameterized by x = pt + q and y = rt + s,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1,2), prove that p^2 + q^2 + r^2 + s^2 = 79 -/
theorem line_segment_param_sum_squares :
  ∀ (p q r s : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    p * t + q = 1 + 5 * t ∧ 
    r * t + s = 2 + 7 * t) →
  p^2 + q^2 + r^2 + s^2 = 79 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_sum_squares_l154_15432


namespace NUMINAMATH_CALUDE_market_equilibrium_and_subsidy_effect_l154_15428

-- Define the supply function
def supply (p : ℝ) : ℝ := 2 + 8 * p

-- Define the demand function (to be derived)
def demand (p : ℝ) : ℝ := -2 * p + 12

-- Define equilibrium
def is_equilibrium (p : ℝ) : Prop := supply p = demand p

-- Define the subsidy amount
def subsidy : ℝ := 1

-- Define the new supply function with subsidy
def supply_with_subsidy (p : ℝ) : ℝ := supply (p + subsidy)

-- Define the new equilibrium with subsidy
def is_equilibrium_with_subsidy (p : ℝ) : Prop := supply_with_subsidy p = demand p

theorem market_equilibrium_and_subsidy_effect :
  -- Original equilibrium
  (∃ p q : ℝ, p = 1 ∧ q = 10 ∧ is_equilibrium p ∧ supply p = q) ∧
  -- Effect of subsidy
  (∃ p' q' : ℝ, is_equilibrium_with_subsidy p' ∧ supply_with_subsidy p' = q' ∧ q' - 10 = 1.6) :=
by sorry

end NUMINAMATH_CALUDE_market_equilibrium_and_subsidy_effect_l154_15428


namespace NUMINAMATH_CALUDE_profit_division_l154_15488

/-- Given a profit divided between X and Y in the ratio 1/2 : 1/3, where the difference
    between their profit shares is 200, prove that the total profit amount is 1000. -/
theorem profit_division (profit_x profit_y : ℝ) 
    (h_ratio : profit_x / profit_y = (1 : ℝ) / 2 / ((1 : ℝ) / 3))
    (h_diff : profit_x - profit_y = 200) :
    profit_x + profit_y = 1000 := by
  sorry

end NUMINAMATH_CALUDE_profit_division_l154_15488


namespace NUMINAMATH_CALUDE_intersection_M_N_l154_15422

def M : Set ℤ := {1, 2, 3, 4, 5, 6}
def N : Set ℤ := {x | -2 < x ∧ x < 5}

theorem intersection_M_N : M ∩ N = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l154_15422


namespace NUMINAMATH_CALUDE_complex_equation_solution_l154_15455

theorem complex_equation_solution (z : ℂ) :
  z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l154_15455


namespace NUMINAMATH_CALUDE_new_rectangle_area_comparison_l154_15479

theorem new_rectangle_area_comparison (a : ℝ) (h : a > 0) :
  let original_diagonal := Real.sqrt (4 * a^2 + 9 * a^2)
  let new_base := original_diagonal + 3 * a
  let new_height := 9 * a - (1/2) * original_diagonal
  let new_area := new_base * new_height
  let original_area := 2 * a * 3 * a
  new_area > 2 * original_area := by sorry

end NUMINAMATH_CALUDE_new_rectangle_area_comparison_l154_15479


namespace NUMINAMATH_CALUDE_distance_between_locations_l154_15471

/-- Represents a car with its speed -/
structure Car where
  speed : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  carA : Car
  carB : Car
  meetingTime : ℝ
  additionalTime : ℝ
  finalDistanceA : ℝ
  finalDistanceB : ℝ

/-- The theorem stating the distance between locations A and B -/
theorem distance_between_locations (setup : ProblemSetup)
  (h1 : setup.meetingTime = 5)
  (h2 : setup.additionalTime = 3)
  (h3 : setup.finalDistanceA = 130)
  (h4 : setup.finalDistanceB = 160) :
  setup.carA.speed * setup.meetingTime + setup.carB.speed * setup.meetingTime = 290 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_locations_l154_15471


namespace NUMINAMATH_CALUDE_lengthXY_is_six_l154_15452

/-- An isosceles triangle with given properties -/
structure IsoscelesTriangle where
  -- The area of the triangle
  area : ℝ
  -- The length of the altitude from P
  altitude : ℝ
  -- The area of the trapezoid formed by dividing line XY
  trapezoidArea : ℝ

/-- The length of XY in the given isosceles triangle -/
def lengthXY (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating the length of XY is 6 inches for the given conditions -/
theorem lengthXY_is_six (t : IsoscelesTriangle) 
    (h1 : t.area = 180)
    (h2 : t.altitude = 30)
    (h3 : t.trapezoidArea = 135) : 
  lengthXY t = 6 := by
  sorry

end NUMINAMATH_CALUDE_lengthXY_is_six_l154_15452


namespace NUMINAMATH_CALUDE_shoes_outside_library_l154_15461

/-- The number of people in the group -/
def num_people : ℕ := 10

/-- The number of shoes each person has -/
def shoes_per_person : ℕ := 2

/-- The total number of shoes kept outside the library -/
def total_shoes : ℕ := num_people * shoes_per_person

theorem shoes_outside_library :
  total_shoes = 20 :=
by sorry

end NUMINAMATH_CALUDE_shoes_outside_library_l154_15461


namespace NUMINAMATH_CALUDE_equation_solution_l154_15499

theorem equation_solution : ∃ x : ℝ, x ≠ 2 ∧ x + 2 = 2 / (x - 2) ↔ x = Real.sqrt 6 ∨ x = -Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l154_15499


namespace NUMINAMATH_CALUDE_smallest_n_divisible_l154_15496

theorem smallest_n_divisible (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 12 → (¬(72 ∣ m^2) ∨ ¬(1728 ∣ m^3))) ∧ 
  (72 ∣ 12^2) ∧ (1728 ∣ 12^3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_l154_15496


namespace NUMINAMATH_CALUDE_percentage_material_B_in_solution_Y_l154_15421

/-- Given two solutions X and Y, and their mixture, this theorem proves
    the percentage of material B in solution Y. -/
theorem percentage_material_B_in_solution_Y
  (percent_A_X : ℝ) (percent_B_X : ℝ) (percent_A_Y : ℝ)
  (percent_X_in_mixture : ℝ) (percent_A_in_mixture : ℝ)
  (h1 : percent_A_X = 0.20)
  (h2 : percent_B_X = 0.80)
  (h3 : percent_A_Y = 0.30)
  (h4 : percent_X_in_mixture = 0.80)
  (h5 : percent_A_in_mixture = 0.22)
  (h6 : percent_X_in_mixture * percent_A_X + (1 - percent_X_in_mixture) * percent_A_Y = percent_A_in_mixture) :
  1 - percent_A_Y = 0.70 := by
sorry

end NUMINAMATH_CALUDE_percentage_material_B_in_solution_Y_l154_15421


namespace NUMINAMATH_CALUDE_factorial_simplification_l154_15439

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_simplification :
  (factorial 12) / ((factorial 10) + 3 * (factorial 9)) = 132 / 13 :=
by sorry

end NUMINAMATH_CALUDE_factorial_simplification_l154_15439


namespace NUMINAMATH_CALUDE_family_theater_cost_l154_15486

/-- Represents the cost of a theater ticket --/
structure TicketCost where
  full : ℝ
  senior : ℝ
  student : ℝ

/-- Calculates the total cost of tickets for a family group --/
def totalCost (t : TicketCost) : ℝ :=
  3 * t.senior + 3 * t.full + 3 * t.student

/-- Theorem: Given the specified discounts and senior ticket cost, 
    the total cost for all family members is $90 --/
theorem family_theater_cost : 
  ∀ (t : TicketCost), 
    t.senior = 10 ∧ 
    t.senior = 0.8 * t.full ∧ 
    t.student = 0.6 * t.full → 
    totalCost t = 90 := by
  sorry


end NUMINAMATH_CALUDE_family_theater_cost_l154_15486


namespace NUMINAMATH_CALUDE_ryan_chinese_learning_time_l154_15446

/-- Given Ryan's learning schedule, prove the time spent on Chinese daily -/
theorem ryan_chinese_learning_time
  (english_daily : ℕ)
  (days : ℕ)
  (total_time : ℕ)
  (h_english_daily : english_daily = 6)
  (h_days : days = 5)
  (h_total_time : total_time = 65) :
  (total_time - english_daily * days) / days = 7 := by
  sorry

end NUMINAMATH_CALUDE_ryan_chinese_learning_time_l154_15446


namespace NUMINAMATH_CALUDE_p_squared_minus_q_squared_l154_15497

theorem p_squared_minus_q_squared (p q : ℝ) 
  (h1 : p + q = 10) 
  (h2 : p - q = 4) : 
  p^2 - q^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_p_squared_minus_q_squared_l154_15497


namespace NUMINAMATH_CALUDE_savings_to_earnings_ratio_l154_15447

/-- Proves that the ratio of monthly savings to total monthly earnings is 1/2 -/
theorem savings_to_earnings_ratio 
  (car_wash_earnings : ℕ) 
  (dog_walking_earnings : ℕ) 
  (total_savings : ℕ) 
  (saving_months : ℕ) 
  (h1 : car_wash_earnings = 20)
  (h2 : dog_walking_earnings = 40)
  (h3 : total_savings = 150)
  (h4 : saving_months = 5) :
  (total_savings / saving_months) / (car_wash_earnings + dog_walking_earnings) = 1 / 2 := by
  sorry

#check savings_to_earnings_ratio

end NUMINAMATH_CALUDE_savings_to_earnings_ratio_l154_15447


namespace NUMINAMATH_CALUDE_season_games_count_l154_15466

/-- Represents the total number of games played by the team -/
def total_games : ℕ := sorry

/-- Represents the number of remaining games after the first 100 -/
def remaining_games : ℕ := sorry

/-- The number of games won in the first 100 games -/
def first_100_wins : ℕ := 65

/-- The number of games won in the remaining games -/
def remaining_wins : ℕ := remaining_games / 2

/-- The total number of games won in the entire season -/
def total_wins : ℕ := (total_games * 7) / 10

theorem season_games_count : 
  first_100_wins + remaining_wins = total_wins ∧
  total_games = 100 + remaining_games ∧
  total_games = 125 := by sorry

end NUMINAMATH_CALUDE_season_games_count_l154_15466


namespace NUMINAMATH_CALUDE_smallest_n_99n_all_threes_l154_15410

def all_threes (n : ℕ) : Prop :=
  ∀ d, d ∈ (n.digits 10) → d = 3

theorem smallest_n_99n_all_threes :
  ∃ (N : ℕ), (N = 3367 ∧ all_threes (99 * N) ∧ ∀ n < N, ¬ all_threes (99 * n)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_99n_all_threes_l154_15410


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l154_15401

theorem part_to_whole_ratio (N P : ℚ) (h1 : N = 160) (h2 : (1/5) * N + 4 = P - 4) :
  (P - 4) / N = 9 / 40 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l154_15401


namespace NUMINAMATH_CALUDE_equation_solutions_l154_15470

/-- The equation from the original problem -/
def original_equation (x : ℝ) : Prop :=
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48

/-- The theorem stating the solutions to the equation -/
theorem equation_solutions :
  ∀ x : ℝ, original_equation x ↔ (x = 6 ∨ x = 8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l154_15470


namespace NUMINAMATH_CALUDE_factorial_equation_solution_l154_15402

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem factorial_equation_solution (N : ℕ) (h : N > 0) :
  factorial 5 * factorial 9 = 12 * factorial N → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solution_l154_15402


namespace NUMINAMATH_CALUDE_band_sections_fraction_l154_15448

theorem band_sections_fraction (trumpet_fraction trombone_fraction : ℚ) 
  (h1 : trumpet_fraction = 1/2)
  (h2 : trombone_fraction = 1/8) :
  trumpet_fraction + trombone_fraction = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_band_sections_fraction_l154_15448


namespace NUMINAMATH_CALUDE_maryann_work_time_l154_15463

theorem maryann_work_time (total_time calling_time accounting_time report_time : ℕ) : 
  total_time = 1440 ∧
  accounting_time = 2 * calling_time ∧
  report_time = 3 * accounting_time ∧
  total_time = calling_time + accounting_time + report_time →
  report_time = 960 := by
  sorry

end NUMINAMATH_CALUDE_maryann_work_time_l154_15463


namespace NUMINAMATH_CALUDE_mod_sum_powers_l154_15403

theorem mod_sum_powers (n : ℕ) : (36^1724 + 18^1724) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_sum_powers_l154_15403


namespace NUMINAMATH_CALUDE_table_tennis_matches_l154_15431

theorem table_tennis_matches (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 ∧ (n * (n - 1)) / 2 ≠ 10 := by
  sorry

#check table_tennis_matches

end NUMINAMATH_CALUDE_table_tennis_matches_l154_15431


namespace NUMINAMATH_CALUDE_P_Q_disjoint_l154_15413

def P : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
def Q : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 2}

theorem P_Q_disjoint : P ∩ Q = ∅ := by
  sorry

end NUMINAMATH_CALUDE_P_Q_disjoint_l154_15413


namespace NUMINAMATH_CALUDE_marcy_spear_count_l154_15493

/-- Represents the number of spears that can be made from different resources --/
structure SpearYield where
  sapling : ℕ
  log : ℕ
  branches : ℕ
  trunk : ℕ

/-- Represents the exchange rates between resources --/
structure ExchangeRates where
  saplings_to_logs : ℕ × ℕ
  branches_to_trunk : ℕ × ℕ

/-- Represents the initial resources Marcy has --/
structure InitialResources where
  saplings : ℕ
  logs : ℕ
  branches : ℕ

/-- Calculates the maximum number of spears Marcy can make --/
def max_spears (yield : SpearYield) (rates : ExchangeRates) (initial : InitialResources) : ℕ :=
  sorry

/-- Theorem stating that Marcy can make 81 spears given the problem conditions --/
theorem marcy_spear_count :
  let yield : SpearYield := { sapling := 3, log := 9, branches := 7, trunk := 15 }
  let rates : ExchangeRates := { saplings_to_logs := (5, 2), branches_to_trunk := (3, 1) }
  let initial : InitialResources := { saplings := 12, logs := 1, branches := 6 }
  max_spears yield rates initial = 81 := by
  sorry

end NUMINAMATH_CALUDE_marcy_spear_count_l154_15493


namespace NUMINAMATH_CALUDE_intersection_properties_l154_15460

/-- A line passing through point (1,1) with an angle of inclination π/4 -/
def line_l : Set (ℝ × ℝ) :=
  {p | p.2 - 1 = p.1 - 1}

/-- A parabola defined by y² = x + 1 -/
def parabola : Set (ℝ × ℝ) :=
  {p | p.2^2 = p.1 + 1}

/-- The intersection points of the line and parabola -/
def intersection_points : Set (ℝ × ℝ) :=
  line_l ∩ parabola

/-- Point P -/
def P : ℝ × ℝ := (1, 1)

/-- Theorem stating the properties of the intersection -/
theorem intersection_properties :
  ∃ (A B : ℝ × ℝ) (M : ℝ × ℝ),
    A ∈ intersection_points ∧
    B ∈ intersection_points ∧
    A ≠ B ∧
    (‖A - P‖ * ‖B - P‖ = Real.sqrt 10) ∧
    (‖A - B‖ = Real.sqrt 10) ∧
    (M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) ∧
    (M = (1/2, 1/2)) := by
  sorry


end NUMINAMATH_CALUDE_intersection_properties_l154_15460


namespace NUMINAMATH_CALUDE_original_number_proof_l154_15458

theorem original_number_proof (y : ℝ) (h : 1 - 1/y = 1/5) : y = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l154_15458


namespace NUMINAMATH_CALUDE_max_unpainted_cubes_l154_15440

/-- Represents a 3D coordinate in a 3x3x3 cube arrangement -/
structure Coordinate where
  x : Fin 3
  y : Fin 3
  z : Fin 3

/-- Represents a cube in the 3x3x3 arrangement -/
structure Cube where
  coord : Coordinate
  painted : Bool

/-- Represents the 3x3x3 cube arrangement -/
def CubeArrangement : Type := Array Cube

/-- Checks if a cube is on the surface of the 3x3x3 arrangement -/
def isOnSurface (c : Coordinate) : Bool :=
  c.x = 0 || c.x = 2 || c.y = 0 || c.y = 2 || c.z = 0 || c.z = 2

/-- Counts the number of unpainted cubes in the arrangement -/
def countUnpaintedCubes (arr : CubeArrangement) : Nat :=
  arr.foldl (fun count cube => if !cube.painted then count + 1 else count) 0

/-- The main theorem stating the maximum number of unpainted cubes -/
theorem max_unpainted_cubes (arr : CubeArrangement) :
  arr.size = 27 → countUnpaintedCubes arr ≤ 15 := by sorry

end NUMINAMATH_CALUDE_max_unpainted_cubes_l154_15440


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l154_15449

def f (x : ℝ) := -x^2

theorem f_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, x < y ∧ y ≤ 0 → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l154_15449


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_plus_2_range_f_geq_expr_l154_15472

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for the solution set of f(x) ≤ x + 2
theorem solution_set_f_leq_x_plus_2 :
  {x : ℝ | f x ≤ x + 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for the range of x satisfying f(x) ≥ (|a+1| - |2a-1|)/|a| for all non-zero real a
theorem range_f_geq_expr :
  {x : ℝ | ∀ a : ℝ, a ≠ 0 → f x ≥ (|a + 1| - |2*a - 1|) / |a|} =
  {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_plus_2_range_f_geq_expr_l154_15472


namespace NUMINAMATH_CALUDE_average_equation_solution_l154_15467

theorem average_equation_solution (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 79 → a = 30 := by
sorry

end NUMINAMATH_CALUDE_average_equation_solution_l154_15467


namespace NUMINAMATH_CALUDE_complex_simplification_l154_15474

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Simplification of a complex expression -/
theorem complex_simplification : 7 * (4 - 2*i) + 4*i * (7 - 2*i) = 36 + 14*i := by sorry

end NUMINAMATH_CALUDE_complex_simplification_l154_15474


namespace NUMINAMATH_CALUDE_alok_payment_l154_15473

def chapati_quantity : ℕ := 16
def rice_quantity : ℕ := 5
def vegetable_quantity : ℕ := 7
def icecream_quantity : ℕ := 6

def chapati_price : ℕ := 6
def rice_price : ℕ := 45
def vegetable_price : ℕ := 70

def total_cost : ℕ := chapati_quantity * chapati_price + 
                       rice_quantity * rice_price + 
                       vegetable_quantity * vegetable_price

theorem alok_payment : total_cost = 811 := by
  sorry

end NUMINAMATH_CALUDE_alok_payment_l154_15473


namespace NUMINAMATH_CALUDE_joe_hvac_cost_per_vent_l154_15442

/-- The cost per vent of an HVAC system -/
def cost_per_vent (total_cost : ℕ) (num_zones : ℕ) (vents_per_zone : ℕ) : ℚ :=
  total_cost / (num_zones * vents_per_zone)

/-- Theorem: The cost per vent of Joe's HVAC system is $2,000 -/
theorem joe_hvac_cost_per_vent :
  cost_per_vent 20000 2 5 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_joe_hvac_cost_per_vent_l154_15442


namespace NUMINAMATH_CALUDE_prime_product_divisible_by_seven_l154_15456

theorem prime_product_divisible_by_seven (C D : ℕ+) 
  (hC : Nat.Prime C)
  (hD : Nat.Prime D)
  (hCmD : Nat.Prime (C - D))
  (hCpD : Nat.Prime (C + D)) :
  7 ∣ (C - D) * C * D * (C + D) := by
sorry

end NUMINAMATH_CALUDE_prime_product_divisible_by_seven_l154_15456


namespace NUMINAMATH_CALUDE_megans_eggs_per_meal_megans_eggs_problem_l154_15494

theorem megans_eggs_per_meal (initial_eggs : ℕ) (neighbor_eggs : ℕ) 
  (omelet_eggs : ℕ) (cake_eggs : ℕ) (meals : ℕ) : ℕ :=
  let total_eggs := initial_eggs + neighbor_eggs
  let used_eggs := omelet_eggs + cake_eggs
  let remaining_eggs := total_eggs - used_eggs
  let eggs_after_aunt := remaining_eggs / 2
  let final_eggs := eggs_after_aunt
  final_eggs / meals

theorem megans_eggs_problem :
  megans_eggs_per_meal 12 12 2 4 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_megans_eggs_per_meal_megans_eggs_problem_l154_15494


namespace NUMINAMATH_CALUDE_solution_to_equation_l154_15424

theorem solution_to_equation :
  ∃! (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ (5 * x)^10 = (10 * y)^5 - 25 * x ∧ x = 1/5 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l154_15424


namespace NUMINAMATH_CALUDE_sequence_sum_2000_is_zero_l154_15489

def sequence_sum (n : ℕ) : ℤ :=
  let group_sum (k : ℕ) : ℤ := (4*k + 1) - (4*k + 2) - (4*k + 3) + (4*k + 4)
  (Finset.range (n/4)).sum (λ k => group_sum k)

theorem sequence_sum_2000_is_zero : sequence_sum 500 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_2000_is_zero_l154_15489


namespace NUMINAMATH_CALUDE_constant_term_value_l154_15465

theorem constant_term_value (x y z k : ℤ) : 
  4 * x + y + z = 80 →
  3 * x + y - z = 20 →
  x = 20 →
  2 * x - y - z = k →
  k = 40 := by
sorry

end NUMINAMATH_CALUDE_constant_term_value_l154_15465


namespace NUMINAMATH_CALUDE_pascals_triangle_51_numbers_l154_15411

theorem pascals_triangle_51_numbers (n : ℕ) : n + 1 = 51 → Nat.choose n 2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_pascals_triangle_51_numbers_l154_15411


namespace NUMINAMATH_CALUDE_correct_operation_l154_15482

theorem correct_operation (a b : ℝ) : 2 * a^2 * b - a^2 * b = a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l154_15482


namespace NUMINAMATH_CALUDE_small_cubes_in_large_cube_l154_15436

/-- Converts decimeters to centimeters -/
def dm_to_cm (dm : ℕ) : ℕ := dm * 10

/-- Calculates the number of small cubes that fit in a large cube -/
def num_small_cubes (large_side_dm : ℕ) (small_side_cm : ℕ) : ℕ :=
  let large_side_cm := dm_to_cm large_side_dm
  let num_cubes_per_edge := large_side_cm / small_side_cm
  num_cubes_per_edge ^ 3

theorem small_cubes_in_large_cube :
  num_small_cubes 8 4 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_small_cubes_in_large_cube_l154_15436


namespace NUMINAMATH_CALUDE_complement_of_union_l154_15450

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 5}

theorem complement_of_union :
  (U \ (A ∪ B)) = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l154_15450


namespace NUMINAMATH_CALUDE_probability_five_consecutive_heads_eight_flips_l154_15416

/-- A sequence of coin flips -/
def CoinFlipSequence := List Bool

/-- The length of a coin flip sequence -/
def sequenceLength : CoinFlipSequence → Nat :=
  List.length

/-- Checks if a sequence has at least n consecutive heads -/
def hasConsecutiveHeads (n : Nat) : CoinFlipSequence → Bool :=
  sorry

/-- All possible outcomes of flipping a coin n times -/
def allOutcomes (n : Nat) : List CoinFlipSequence :=
  sorry

/-- Count of sequences with at least n consecutive heads -/
def countConsecutiveHeads (n : Nat) (totalFlips : Nat) : Nat :=
  sorry

/-- Probability of getting at least n consecutive heads in m flips -/
def probabilityConsecutiveHeads (n : Nat) (m : Nat) : Rat :=
  sorry

theorem probability_five_consecutive_heads_eight_flips :
  probabilityConsecutiveHeads 5 8 = 23 / 256 :=
sorry

end NUMINAMATH_CALUDE_probability_five_consecutive_heads_eight_flips_l154_15416


namespace NUMINAMATH_CALUDE_product_13_factor_l154_15427

theorem product_13_factor (w : ℕ+) (h1 : w ≥ 468) 
  (h2 : ∃ (k : ℕ), 2^4 * 3^3 * k = 1452 * w) : 
  (∃ (m : ℕ), 13^1 * m = 1452 * w) ∧ 
  (∀ (n : ℕ), n > 1 → ¬(∃ (m : ℕ), 13^n * m = 1452 * w)) :=
sorry

end NUMINAMATH_CALUDE_product_13_factor_l154_15427


namespace NUMINAMATH_CALUDE_students_taking_one_subject_l154_15409

theorem students_taking_one_subject (both : ℕ) (science : ℕ) (only_history : ℕ) 
  (h1 : both = 15)
  (h2 : science = 30)
  (h3 : only_history = 18) :
  science - both + only_history = 33 := by
sorry

end NUMINAMATH_CALUDE_students_taking_one_subject_l154_15409


namespace NUMINAMATH_CALUDE_intersection_m_complement_n_l154_15419

/-- The intersection of set M and the complement of set N in the real numbers -/
theorem intersection_m_complement_n :
  let U : Set ℝ := Set.univ
  let M : Set ℝ := {x | x^2 - 2*x < 0}
  let N : Set ℝ := {x | x ≥ 1}
  M ∩ (U \ N) = {x | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_m_complement_n_l154_15419


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l154_15498

theorem line_tangent_to_circle 
  (x₀ y₀ r : ℝ) 
  (h_outside : x₀^2 + y₀^2 > r^2) :
  ∃ (x y : ℝ), 
    x₀*x + y₀*y = r^2 ∧ 
    x^2 + y^2 = r^2 ∧
    ∀ (x' y' : ℝ), x₀*x' + y₀*y' = r^2 ∧ x'^2 + y'^2 = r^2 → (x', y') = (x, y) :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l154_15498


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l154_15464

-- Define the arithmetic sequence
def arithmetic_sequence (x y z : ℤ) : Prop :=
  ∃ d : ℤ, y = x + d ∧ z = y + d

-- Theorem statement
theorem arithmetic_sequence_problem (x y z w u : ℤ) 
  (h1 : arithmetic_sequence x y z)
  (h2 : x = 1370)
  (h3 : z = 1070)
  (h4 : w = -180)
  (h5 : u = -6430) :
  w^3 - u^2 + y^2 = -44200100 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l154_15464
