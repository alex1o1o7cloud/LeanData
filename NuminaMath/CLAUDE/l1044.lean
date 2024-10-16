import Mathlib

namespace NUMINAMATH_CALUDE_min_hours_is_eight_l1044_104479

/-- Represents Biff's expenses and earnings during the bus trip -/
structure BusTrip where
  ticket : ℕ
  snacks : ℕ
  headphones : ℕ
  lunch : ℕ
  dinner : ℕ
  accommodation : ℕ
  hourly_rate : ℕ
  day_wifi_rate : ℕ
  night_wifi_rate : ℕ

/-- Calculates the total fixed expenses for the trip -/
def total_fixed_expenses (trip : BusTrip) : ℕ :=
  trip.ticket + trip.snacks + trip.headphones + trip.lunch + trip.dinner + trip.accommodation

/-- Calculates the minimum number of hours needed to break even -/
def min_hours_to_break_even (trip : BusTrip) : ℕ :=
  (total_fixed_expenses trip + trip.night_wifi_rate - 1) / (trip.hourly_rate - trip.night_wifi_rate) + 1

/-- Theorem stating that the minimum number of hours to break even is 8 -/
theorem min_hours_is_eight (trip : BusTrip)
  (h1 : trip.ticket = 11)
  (h2 : trip.snacks = 3)
  (h3 : trip.headphones = 16)
  (h4 : trip.lunch = 8)
  (h5 : trip.dinner = 10)
  (h6 : trip.accommodation = 35)
  (h7 : trip.hourly_rate = 12)
  (h8 : trip.day_wifi_rate = 2)
  (h9 : trip.night_wifi_rate = 1) :
  min_hours_to_break_even trip = 8 := by
  sorry

#eval min_hours_to_break_even {
  ticket := 11,
  snacks := 3,
  headphones := 16,
  lunch := 8,
  dinner := 10,
  accommodation := 35,
  hourly_rate := 12,
  day_wifi_rate := 2,
  night_wifi_rate := 1
}

end NUMINAMATH_CALUDE_min_hours_is_eight_l1044_104479


namespace NUMINAMATH_CALUDE_problem_solution_l1044_104461

theorem problem_solution (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1044_104461


namespace NUMINAMATH_CALUDE_square_root_existence_l1044_104454

theorem square_root_existence : 
  (∃ x : ℝ, x^2 = (-3)^2) ∧ 
  (∃ x : ℝ, x^2 = 0) ∧ 
  (∃ x : ℝ, x^2 = 1/8) ∧ 
  (¬∃ x : ℝ, x^2 = -6^3) := by
  sorry

end NUMINAMATH_CALUDE_square_root_existence_l1044_104454


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1044_104414

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A point on the hyperbola -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) (P : Point) 
  (h1 : distance P h.F₂ = 2 * distance P h.F₁)
  (h2 : angle P h.F₁ h.F₂ = Real.pi / 3) : 
  eccentricity h = (1 + Real.sqrt 13) / 2 := sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1044_104414


namespace NUMINAMATH_CALUDE_a5_greater_than_b5_l1044_104487

-- Define the geometric sequence a_n
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ (n - 1)

-- Define the arithmetic sequence b_n
def arithmetic_sequence (b₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  b₁ + (n - 1) * d

theorem a5_greater_than_b5 
  (a₁ b₁ q d : ℝ)
  (h1 : a₁ = b₁)
  (h2 : a₁ > 0)
  (h3 : geometric_sequence a₁ q 3 = arithmetic_sequence b₁ d 3)
  (h4 : a₁ ≠ geometric_sequence a₁ q 3) :
  geometric_sequence a₁ q 5 > arithmetic_sequence b₁ d 5 := by
  sorry

end NUMINAMATH_CALUDE_a5_greater_than_b5_l1044_104487


namespace NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l1044_104482

/-- Given a rectangle of width y and height 2y divided into a square of side x
    and four congruent rectangles, the perimeter of one of the congruent rectangles
    is 3y - 2x. -/
theorem congruent_rectangle_perimeter
  (y : ℝ) (x : ℝ)
  (h1 : y > 0)
  (h2 : x > 0)
  (h3 : x < y)
  (h4 : x < 2*y) :
  ∃ (l w : ℝ),
    l > 0 ∧ w > 0 ∧
    x + 2*l = y ∧
    x + 2*w = 2*y ∧
    2*l + 2*w = 3*y - 2*x :=
by sorry

end NUMINAMATH_CALUDE_congruent_rectangle_perimeter_l1044_104482


namespace NUMINAMATH_CALUDE_triangle_area_l1044_104476

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if c² = a² + b² - 2ab + 6 and C = π/3, then the area of the triangle is 3√3/2 -/
theorem triangle_area (a b c : ℝ) (h1 : c^2 = a^2 + b^2 - 2*a*b + 6) (h2 : Real.pi / 3 = Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) :
  (1/2) * a * b * Real.sin (Real.pi / 3) = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1044_104476


namespace NUMINAMATH_CALUDE_slips_with_three_l1044_104425

/-- Given a bag with 15 slips, each having either 3 or 9, prove that if the expected value
    of a randomly drawn slip is 5, then 10 slips have 3 on them. -/
theorem slips_with_three (total : ℕ) (value_a value_b : ℕ) (expected : ℚ) : 
  total = 15 →
  value_a = 3 →
  value_b = 9 →
  expected = 5 →
  ∃ (count_a : ℕ), 
    count_a ≤ total ∧
    (count_a : ℚ) / total * value_a + (total - count_a : ℚ) / total * value_b = expected ∧
    count_a = 10 :=
by sorry

end NUMINAMATH_CALUDE_slips_with_three_l1044_104425


namespace NUMINAMATH_CALUDE_max_value_of_equation_l1044_104480

theorem max_value_of_equation (x : ℝ) : 
  (x^2 - x - 30) / (x - 5) = 2 / (x + 6) → x ≤ Real.sqrt 38 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_equation_l1044_104480


namespace NUMINAMATH_CALUDE_number_of_digits_l1044_104429

theorem number_of_digits (N : ℕ) : N = 2^12 * 5^8 → (Nat.digits 10 N).length = 10 := by sorry

end NUMINAMATH_CALUDE_number_of_digits_l1044_104429


namespace NUMINAMATH_CALUDE_last_digit_sum_powers_l1044_104428

theorem last_digit_sum_powers : (1993^2002 + 1995^2002) % 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_sum_powers_l1044_104428


namespace NUMINAMATH_CALUDE_product_of_multiples_of_three_l1044_104490

theorem product_of_multiples_of_three : ∃ (a b : ℕ), 
  a = 22 * 3 ∧ 
  b = 23 * 3 ∧ 
  a < 100 ∧ 
  b < 100 ∧ 
  a * b = 4554 := by
  sorry

end NUMINAMATH_CALUDE_product_of_multiples_of_three_l1044_104490


namespace NUMINAMATH_CALUDE_distance_center_to_point_l1044_104435

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y + 9

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 3)

-- Define the point
def point : ℝ × ℝ := (8, 3)

-- Theorem statement
theorem distance_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := point
  Real.sqrt ((cx - px)^2 + (cy - py)^2) = 6 :=
sorry

end NUMINAMATH_CALUDE_distance_center_to_point_l1044_104435


namespace NUMINAMATH_CALUDE_six_solved_only_b_l1044_104400

/-- Represents the number of students who solved specific combinations of problems -/
structure ProblemSolvers where
  a : ℕ  -- only A
  b : ℕ  -- only B
  c : ℕ  -- only C
  ab : ℕ  -- A and B
  bc : ℕ  -- B and C
  ca : ℕ  -- C and A
  abc : ℕ  -- all three

/-- The conditions of the math competition problem -/
def competition_conditions (s : ProblemSolvers) : Prop :=
  -- Total number of students is 25
  s.a + s.b + s.c + s.ab + s.bc + s.ca + s.abc = 25 ∧
  -- Among students who didn't solve A, those who solved B is twice those who solved C
  s.b + s.bc = 2 * (s.c + s.bc) ∧
  -- Among students who solved A, those who solved only A is one more than those who solved A and others
  s.a = (s.ab + s.ca + s.abc) + 1 ∧
  -- Among students who solved only one problem, half didn't solve A
  s.a = s.b + s.c

/-- The theorem stating that 6 students solved only problem B -/
theorem six_solved_only_b :
  ∃ (s : ProblemSolvers), competition_conditions s ∧ s.b = 6 :=
sorry

end NUMINAMATH_CALUDE_six_solved_only_b_l1044_104400


namespace NUMINAMATH_CALUDE_alice_bob_number_sum_l1044_104404

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

theorem alice_bob_number_sum :
  ∀ (A B : ℕ),
    A ≤ 50 ∧ B ≤ 50 ∧ A ≠ B →
    (¬(is_prime A) ∧ ¬(¬is_prime A)) →
    (¬(is_prime B) ∧ is_perfect_square B) →
    is_perfect_square (50 * B + A) →
    A + B = 43 := by
  sorry

end NUMINAMATH_CALUDE_alice_bob_number_sum_l1044_104404


namespace NUMINAMATH_CALUDE_fraction_power_product_l1044_104472

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by sorry

end NUMINAMATH_CALUDE_fraction_power_product_l1044_104472


namespace NUMINAMATH_CALUDE_matrix_commutator_similarity_l1044_104456

/-- Given n×n complex matrices A and B where A^2 = B^2, there exists an invertible n×n complex matrix S such that S(AB - BA) = (BA - AB)S. -/
theorem matrix_commutator_similarity {n : ℕ} (A B : Matrix (Fin n) (Fin n) ℂ) 
  (h : A ^ 2 = B ^ 2) : 
  ∃ S : Matrix (Fin n) (Fin n) ℂ, IsUnit S ∧ S * (A * B - B * A) = (B * A - A * B) * S := by
  sorry

end NUMINAMATH_CALUDE_matrix_commutator_similarity_l1044_104456


namespace NUMINAMATH_CALUDE_divides_power_difference_l1044_104471

theorem divides_power_difference (n : ℕ) : n ∣ 2^(2*n.factorial) - 2^(n.factorial) := by
  sorry

end NUMINAMATH_CALUDE_divides_power_difference_l1044_104471


namespace NUMINAMATH_CALUDE_b_and_d_know_grades_l1044_104494

-- Define the grade types
inductive Grade
| Excellent
| Good

-- Define the students
inductive Student
| A
| B
| C
| D

-- Function to represent the actual grade of a student
def actualGrade : Student → Grade := sorry

-- Function to represent what grades a student can see
def canSee : Student → Student → Prop := sorry

-- Theorem statement
theorem b_and_d_know_grades :
  -- There are 2 excellent grades and 2 good grades
  (∃ (s1 s2 s3 s4 : Student), s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
    actualGrade s1 = Grade.Excellent ∧ actualGrade s2 = Grade.Excellent ∧
    actualGrade s3 = Grade.Good ∧ actualGrade s4 = Grade.Good) →
  -- A, B, and C can see each other's grades
  (canSee Student.A Student.B ∧ canSee Student.A Student.C ∧
   canSee Student.B Student.A ∧ canSee Student.B Student.C ∧
   canSee Student.C Student.A ∧ canSee Student.C Student.B) →
  -- B and C can see each other's grades
  (canSee Student.B Student.C ∧ canSee Student.C Student.B) →
  -- D and A can see each other's grades
  (canSee Student.D Student.A ∧ canSee Student.A Student.D) →
  -- A doesn't know their own grade after seeing B and C's grades
  (∃ (g1 g2 : Grade), g1 ≠ g2 ∧
    ((actualGrade Student.B = g1 ∧ actualGrade Student.C = g2) ∨
     (actualGrade Student.B = g2 ∧ actualGrade Student.C = g1))) →
  -- B and D can know their own grades
  (∃ (gb gd : Grade),
    (actualGrade Student.B = gb ∧ ∀ g, actualGrade Student.B = g → g = gb) ∧
    (actualGrade Student.D = gd ∧ ∀ g, actualGrade Student.D = g → g = gd))
  := by sorry

end NUMINAMATH_CALUDE_b_and_d_know_grades_l1044_104494


namespace NUMINAMATH_CALUDE_system_solutions_l1044_104486

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  y = 2 * x^2 - 1 ∧ z = 2 * y^2 - 1 ∧ x = 2 * z^2 - 1

/-- The set of solutions to the system -/
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(1, 1, 1), (-1/2, -1/2, -1/2)} ∪
  {(Real.cos (2 * Real.pi / 9), Real.cos (4 * Real.pi / 9), -Real.cos (Real.pi / 9)),
   (Real.cos (4 * Real.pi / 9), -Real.cos (Real.pi / 9), Real.cos (2 * Real.pi / 9)),
   (-Real.cos (Real.pi / 9), Real.cos (2 * Real.pi / 9), Real.cos (4 * Real.pi / 9))} ∪
  {(Real.cos (2 * Real.pi / 7), -Real.cos (3 * Real.pi / 7), -Real.cos (Real.pi / 7)),
   (-Real.cos (3 * Real.pi / 7), -Real.cos (Real.pi / 7), Real.cos (2 * Real.pi / 7)),
   (-Real.cos (Real.pi / 7), Real.cos (2 * Real.pi / 7), -Real.cos (3 * Real.pi / 7))}

/-- Theorem stating that the solutions set contains all and only the solutions to the system -/
theorem system_solutions :
  ∀ x y z : ℝ, (x, y, z) ∈ solutions ↔ system x y z :=
sorry

end NUMINAMATH_CALUDE_system_solutions_l1044_104486


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1044_104481

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the foci coordinates
def foci : Set (ℝ × ℝ) := {(-4, 0), (4, 0)}

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_properties :
  (∀ x y, hyperbola x y → (x, y) ∈ foci ∨ (x, y) ∉ foci) ∧
  (∃ a b c : ℝ, a^2 = 4 ∧ b^2 = 12 ∧ c^2 = a^2 + b^2 ∧ eccentricity = c / a) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1044_104481


namespace NUMINAMATH_CALUDE_kingdom_animal_percentage_l1044_104415

/-- Represents the number of cats in the kingdom -/
def num_cats : ℕ := 25

/-- Represents the number of hogs in the kingdom -/
def num_hogs : ℕ := 75

/-- The relationship between hogs and cats -/
axiom hogs_cats_relation : num_hogs = 3 * num_cats

/-- The percentage we're looking for -/
def percentage : ℚ := 50

theorem kingdom_animal_percentage :
  (percentage / 100) * (num_cats - 5 : ℚ) = 10 :=
sorry

end NUMINAMATH_CALUDE_kingdom_animal_percentage_l1044_104415


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l1044_104439

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 682000000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 6.82
    exponent := 8
    is_valid := by sorry }

/-- Theorem stating that the proposed scientific notation correctly represents the original number -/
theorem scientific_notation_correct :
  (proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent) = original_number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l1044_104439


namespace NUMINAMATH_CALUDE_xy_reciprocal_problem_l1044_104499

theorem xy_reciprocal_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 1) (h2 : x / y = 36) : y = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_xy_reciprocal_problem_l1044_104499


namespace NUMINAMATH_CALUDE_P_roots_l1044_104440

def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | n + 1, x => x^(5 * (n + 1)) - P n x

theorem P_roots (n : ℕ) :
  (n % 2 = 1 → P n 1 = 0 ∧ ∀ x : ℝ, x ≠ 1 → P n x ≠ 0) ∧
  (n % 2 = 0 → ∀ x : ℝ, P n x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_P_roots_l1044_104440


namespace NUMINAMATH_CALUDE_compute_expression_l1044_104446

theorem compute_expression : 9 * (2/3)^4 = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l1044_104446


namespace NUMINAMATH_CALUDE_m_range_l1044_104402

-- Define propositions P and Q
def P (m : ℝ) : Prop := |m + 1| ≤ 2
def Q (m : ℝ) : Prop := ∃ x : ℝ, x^2 - m*x + 1 = 0

-- Define the theorem
theorem m_range :
  (∀ m : ℝ, ¬(¬(P m))) →
  (∀ m : ℝ, ¬(P m ∧ Q m)) →
  ∀ m : ℝ, (m > -2 ∧ m ≤ 1) ↔ (P m ∧ ¬(Q m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1044_104402


namespace NUMINAMATH_CALUDE_fence_painting_earnings_l1044_104448

/-- Calculate the total earnings from painting fences -/
theorem fence_painting_earnings
  (rate : ℝ)
  (num_fences : ℕ)
  (fence_length : ℝ)
  (h1 : rate = 0.20)
  (h2 : num_fences = 50)
  (h3 : fence_length = 500) :
  rate * (↑num_fences * fence_length) = 5000 := by
  sorry

end NUMINAMATH_CALUDE_fence_painting_earnings_l1044_104448


namespace NUMINAMATH_CALUDE_gigi_mushrooms_l1044_104457

/-- The number of pieces each mushroom is cut into -/
def pieces_per_mushroom : ℕ := 4

/-- The number of mushroom pieces Kenny used -/
def kenny_pieces : ℕ := 38

/-- The number of mushroom pieces Karla used -/
def karla_pieces : ℕ := 42

/-- The number of mushroom pieces left on the cutting board -/
def leftover_pieces : ℕ := 8

/-- The total number of mushroom pieces -/
def total_pieces : ℕ := kenny_pieces + karla_pieces + leftover_pieces

/-- The number of whole mushrooms GiGi cut up -/
def whole_mushrooms : ℕ := total_pieces / pieces_per_mushroom

theorem gigi_mushrooms : whole_mushrooms = 22 := by
  sorry

end NUMINAMATH_CALUDE_gigi_mushrooms_l1044_104457


namespace NUMINAMATH_CALUDE_daily_harvest_l1044_104419

/-- The number of sections in the orchard -/
def num_sections : ℕ := 8

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 45

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := num_sections * sacks_per_section

theorem daily_harvest : total_sacks = 360 := by
  sorry

end NUMINAMATH_CALUDE_daily_harvest_l1044_104419


namespace NUMINAMATH_CALUDE_square_of_negative_product_l1044_104488

theorem square_of_negative_product (a b : ℝ) : (-a^2 * b)^2 = a^4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l1044_104488


namespace NUMINAMATH_CALUDE_pi_approximation_proof_l1044_104433

theorem pi_approximation_proof :
  let π := 4 * Real.sin (52 * π / 180)
  (2 * π * Real.sqrt (16 - π^2) - 8 * Real.sin (44 * π / 180)) /
  (Real.sqrt 3 - 2 * Real.sqrt 3 * Real.sin (22 * π / 180)^2) = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_pi_approximation_proof_l1044_104433


namespace NUMINAMATH_CALUDE_translation_theorem_l1044_104441

/-- The original function -/
def f (x : ℝ) : ℝ := -(x - 1)^2 + 4

/-- The translated function -/
def g (x : ℝ) : ℝ := -(x + 1)^2 + 1

/-- Translation parameters -/
def left_shift : ℝ := 2
def down_shift : ℝ := 3

theorem translation_theorem :
  ∀ x : ℝ, g x = f (x + left_shift) - down_shift := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l1044_104441


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_15_18_l1044_104431

theorem smallest_divisible_by_12_15_18 : ∃ n : ℕ+, (∀ m : ℕ+, 12 ∣ m ∧ 15 ∣ m ∧ 18 ∣ m → n ≤ m) ∧ 12 ∣ n ∧ 15 ∣ n ∧ 18 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_15_18_l1044_104431


namespace NUMINAMATH_CALUDE_range_of_a_l1044_104420

theorem range_of_a (A B : Set ℝ) (a : ℝ) :
  A = {x : ℝ | x ≤ 1} →
  B = {x : ℝ | x ≥ a} →
  A ∪ B = Set.univ →
  Set.Iic 1 = {a | ∀ x, (x ∈ A ∪ B ↔ x ∈ Set.univ)} :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1044_104420


namespace NUMINAMATH_CALUDE_simplest_form_l1044_104473

theorem simplest_form (a b : ℝ) (h : a ≠ b ∧ a ≠ -b) : 
  ¬∃ (f g : ℝ → ℝ → ℝ), ∀ (x y : ℝ), 
    (x^2 + y^2) / (x^2 - y^2) = f x y / g x y ∧ 
    (f x y ≠ x^2 + y^2 ∨ g x y ≠ x^2 - y^2) :=
sorry

end NUMINAMATH_CALUDE_simplest_form_l1044_104473


namespace NUMINAMATH_CALUDE_binomial_coefficient_even_l1044_104468

theorem binomial_coefficient_even (n : ℕ) (h : Even n) (h2 : n > 0) : 
  Nat.choose n 2 = n * (n - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_even_l1044_104468


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1044_104475

theorem sum_of_x_and_y (x y : ℚ) 
  (hx : |x| = 5)
  (hy : |y| = 2)
  (hxy : |x - y| = x - y) :
  x + y = 7 ∨ x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1044_104475


namespace NUMINAMATH_CALUDE_meet_once_l1044_104459

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (m : Movement) : ℕ :=
  sorry

/-- The specific movement scenario described in the problem --/
def problem_scenario : Movement where
  michael_speed := 6
  truck_speed := 12
  pail_distance := 300
  truck_stop_time := 20
  initial_distance := 300

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once : number_of_meetings problem_scenario = 1 := by
  sorry

end NUMINAMATH_CALUDE_meet_once_l1044_104459


namespace NUMINAMATH_CALUDE_factorial_ratio_eq_120_l1044_104447

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_ratio_eq_120 :
  factorial 10 / (factorial 7 * factorial 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_eq_120_l1044_104447


namespace NUMINAMATH_CALUDE_p_squared_plus_36_composite_l1044_104418

theorem p_squared_plus_36_composite (p : ℕ) (h : Nat.Prime p) :
  ∃ (n : ℕ), n > 1 ∧ n ∣ (p^2 + 36) := by
  sorry

end NUMINAMATH_CALUDE_p_squared_plus_36_composite_l1044_104418


namespace NUMINAMATH_CALUDE_tan_value_from_trig_equation_l1044_104493

theorem tan_value_from_trig_equation (α : Real) 
  (h : (Real.sin α - Real.cos α) / (2 * Real.sin α + 3 * Real.cos α) = 1/5) : 
  Real.tan α = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_equation_l1044_104493


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1044_104445

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -5) : x^3 + 1/x^3 = -110 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l1044_104445


namespace NUMINAMATH_CALUDE_concentric_circles_area_ratio_l1044_104462

theorem concentric_circles_area_ratio :
  let d₁ : ℝ := 2  -- diameter of smaller circle
  let d₂ : ℝ := 6  -- diameter of larger circle
  let r₁ : ℝ := d₁ / 2  -- radius of smaller circle
  let r₂ : ℝ := d₂ / 2  -- radius of larger circle
  let A₁ : ℝ := Real.pi * r₁ ^ 2  -- area of smaller circle
  let A₂ : ℝ := Real.pi * r₂ ^ 2  -- area of larger circle
  (A₂ - A₁) / A₁ = 8 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_area_ratio_l1044_104462


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l1044_104450

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Condition that length is greater than width -/
def Rectangle.lengthGreaterThanWidth (r : Rectangle) : Prop :=
  r.length > r.width

/-- Perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ :=
  2 * (r.length + r.width)

/-- Area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Theorem stating the dimensions of the rectangle -/
theorem rectangle_dimensions (r : Rectangle) 
  (h1 : r.lengthGreaterThanWidth)
  (h2 : r.perimeter = 18)
  (h3 : r.area = 18) :
  r.length = 6 ∧ r.width = 3 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_dimensions_l1044_104450


namespace NUMINAMATH_CALUDE_shoes_mode_median_equal_l1044_104411

structure SalesData where
  sizes : List Float
  volumes : List Nat
  total_pairs : Nat

def mode (data : SalesData) : Float :=
  sorry

def median (data : SalesData) : Float :=
  sorry

theorem shoes_mode_median_equal (data : SalesData) :
  data.sizes = [23, 23.5, 24, 24.5, 25] ∧
  data.volumes = [1, 2, 2, 6, 2] ∧
  data.total_pairs = 15 →
  mode data = 24.5 ∧ median data = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_shoes_mode_median_equal_l1044_104411


namespace NUMINAMATH_CALUDE_opposite_sign_sum_l1044_104401

theorem opposite_sign_sum (x y : ℝ) :
  (|x + 2| + |y - 4| = 0) → (x + y - 3 = -1) := by
  sorry

end NUMINAMATH_CALUDE_opposite_sign_sum_l1044_104401


namespace NUMINAMATH_CALUDE_managers_salary_solve_manager_salary_problem_l1044_104436

/-- Calculates the manager's salary given the number of employees, their average salary,
    and the increase in average salary when the manager's salary is added. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℚ) (avg_increase : ℚ) : ℚ :=
  let total_salary := num_employees * avg_salary
  let new_avg_salary := avg_salary + avg_increase
  let new_total_salary := (num_employees + 1) * new_avg_salary
  new_total_salary - total_salary

/-- Proves that the manager's salary is 3300 given the problem conditions. -/
theorem solve_manager_salary_problem :
  managers_salary 20 1200 100 = 3300 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_solve_manager_salary_problem_l1044_104436


namespace NUMINAMATH_CALUDE_domino_distribution_l1044_104477

theorem domino_distribution (total_dominoes : Nat) (num_players : Nat) 
  (h1 : total_dominoes = 28) (h2 : num_players = 4) :
  total_dominoes / num_players = 7 := by
  sorry

#check domino_distribution

end NUMINAMATH_CALUDE_domino_distribution_l1044_104477


namespace NUMINAMATH_CALUDE_student_failed_marks_l1044_104422

theorem student_failed_marks (total_marks : ℕ) (passing_percentage : ℚ) (student_score : ℕ) : 
  total_marks = 600 → 
  passing_percentage = 33 / 100 → 
  student_score = 125 → 
  (total_marks * passing_percentage).floor - student_score = 73 := by
  sorry

end NUMINAMATH_CALUDE_student_failed_marks_l1044_104422


namespace NUMINAMATH_CALUDE_jason_newspaper_earnings_l1044_104460

/-- Proves that Jason's earnings from delivering newspapers equals $1.875 --/
theorem jason_newspaper_earnings 
  (fred_initial : ℝ) 
  (jason_initial : ℝ) 
  (emily_initial : ℝ) 
  (fred_increase : ℝ) 
  (jason_increase : ℝ) 
  (emily_increase : ℝ) 
  (h1 : fred_initial = 49) 
  (h2 : jason_initial = 3) 
  (h3 : emily_initial = 25) 
  (h4 : fred_increase = 1.5) 
  (h5 : jason_increase = 1.625) 
  (h6 : emily_increase = 1.4) :
  jason_initial * (jason_increase - 1) = 1.875 := by
  sorry

end NUMINAMATH_CALUDE_jason_newspaper_earnings_l1044_104460


namespace NUMINAMATH_CALUDE_range_of_a_l1044_104492

-- Define the inequality system
def inequality_system (a : ℝ) (x : ℝ) : Prop :=
  x > a ∧ x > 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  {x | x > 1}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  (∀ x, inequality_system a x ↔ x ∈ solution_set a) →
  a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1044_104492


namespace NUMINAMATH_CALUDE_f_properties_l1044_104467

noncomputable def f (a x : ℝ) : ℝ := Real.log x - (a + 2) * x + a * x^2

theorem f_properties (a : ℝ) :
  -- Part I: Tangent line equation when a = 0
  (∀ x y : ℝ, f 0 1 = -2 ∧ x + y + 1 = 0 ↔ y = f 0 x ∧ (x - 1) * (f 0 x - f 0 1) = (y - f 0 1) * (x - 1)) ∧
  -- Part II: Monotonicity intervals
  (∀ x : ℝ, 0 < x ∧ x < 1/2 → (∀ h : ℝ, h > 0 → f a (x + h) > f a x)) ∧
  (∀ x : ℝ, x > 1/2 → (∀ h : ℝ, h > 0 → f a (x + h) < f a x)) ∧
  -- Part III: Condition for exactly two zeros
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ (∀ x : ℝ, f a x = 0 → x = x₁ ∨ x = x₂) ↔ a < -4 * Real.log 2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1044_104467


namespace NUMINAMATH_CALUDE_quadratic_function_points_range_l1044_104444

theorem quadratic_function_points_range (m n y₁ y₂ : ℝ) : 
  y₁ = (m - 2)^2 + n → 
  y₂ = (m - 1)^2 + n → 
  y₁ < y₂ → 
  m > 3/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_points_range_l1044_104444


namespace NUMINAMATH_CALUDE_min_squares_cover_l1044_104496

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- Function to calculate the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℕ :=
  r.length * r.width

/-- Function to calculate the area of a square -/
def squareArea (s : Square) : ℕ :=
  s.side * s.side

/-- Function to check if a list of squares can cover a rectangle -/
def canCover (r : Rectangle) (squares : List Square) : Prop :=
  rectangleArea r = (squares.map squareArea).sum

/-- The main theorem to be proved -/
theorem min_squares_cover (r : Rectangle) (squares : List Square) :
  r.length = 10 ∧ r.width = 9 ∧ 
  (∀ s ∈ squares, s.side > 0) ∧
  canCover r squares →
  squares.length ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_squares_cover_l1044_104496


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l1044_104443

theorem min_value_of_sum_of_fractions (n : ℕ) (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
  (1 / (1 + a ^ n) + 1 / (1 + b ^ n)) ≥ 1 ∧ 
  (1 / (1 + 1 ^ n) + 1 / (1 + 1 ^ n) = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_fractions_l1044_104443


namespace NUMINAMATH_CALUDE_tan_BAC_equals_three_fourths_l1044_104463

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points D and E on sides AB and AC
structure TriangleWithDE extends Triangle :=
  (D : ℝ × ℝ)
  (E : ℝ × ℝ)
  (D_on_AB : D.1 = A.1 + t * (B.1 - A.1) ∧ D.2 = A.2 + t * (B.2 - A.2)) 
  (E_on_AC : E.1 = A.1 + s * (C.1 - A.1) ∧ E.2 = A.2 + s * (C.2 - A.2))
  (t s : ℝ)
  (t_range : 0 < t ∧ t < 1)
  (s_range : 0 < s ∧ s < 1)

-- Define the area of triangle ADE
def area_ADE (t : TriangleWithDE) : ℝ := sorry

-- Define the incircle of quadrilateral BDEC
structure Incircle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the point K where the incircle touches AB
def point_K (t : TriangleWithDE) (i : Incircle) : ℝ × ℝ := sorry

-- Define the function to calculate tan(BAC)
def tan_BAC (t : Triangle) : ℝ := sorry

-- Define the theorem
theorem tan_BAC_equals_three_fourths 
  (t : TriangleWithDE) 
  (i : Incircle) 
  (h1 : area_ADE t = 0.5)
  (h2 : point_K t i = (t.A.1 + 3, t.A.2))
  (h3 : (t.C.1 - t.B.1)^2 + (t.C.2 - t.B.2)^2 = 15^2)
  (h4 : ∃ (center : ℝ × ℝ) (radius : ℝ), 
        (t.B.1 - center.1)^2 + (t.B.2 - center.2)^2 = radius^2 ∧
        (t.D.1 - center.1)^2 + (t.D.2 - center.2)^2 = radius^2 ∧
        (t.E.1 - center.1)^2 + (t.E.2 - center.2)^2 = radius^2 ∧
        (t.C.1 - center.1)^2 + (t.C.2 - center.2)^2 = radius^2) :
  tan_BAC t.toTriangle = 3/4 := by sorry

end NUMINAMATH_CALUDE_tan_BAC_equals_three_fourths_l1044_104463


namespace NUMINAMATH_CALUDE_soap_brands_survey_l1044_104442

/-- The number of households that use both brands of soap -/
def households_both_brands : ℕ := 30

/-- The total number of households surveyed -/
def total_households : ℕ := 260

/-- The number of households that use neither brand A nor brand B -/
def households_neither_brand : ℕ := 80

/-- The number of households that use only brand A -/
def households_only_A : ℕ := 60

theorem soap_brands_survey :
  households_both_brands = 30 ∧
  total_households = households_neither_brand + households_only_A + households_both_brands + 3 * households_both_brands :=
by sorry

end NUMINAMATH_CALUDE_soap_brands_survey_l1044_104442


namespace NUMINAMATH_CALUDE_money_exchange_solution_money_exchange_unique_l1044_104485

/-- Represents the money exchange process between three friends -/
def money_exchange (a b c : ℕ) : Prop :=
  let step1_1 := a - b - c
  let step1_2 := 2 * b
  let step1_3 := 2 * c
  let step2_1 := 2 * (a - b - c)
  let step2_2 := 3 * b - a - 3 * c
  let step2_3 := 4 * c
  let step3_1 := 4 * (a - b - c)
  let step3_2 := 6 * b - 2 * a - 6 * c
  let step3_3 := 4 * c - 2 * (a - b - c) - (3 * b - a - 3 * c)
  step3_1 = 8 ∧ step3_2 = 8 ∧ step3_3 = 8

/-- Theorem stating that the initial amounts of 13, 7, and 4 écus result in each friend having 8 écus after the exchanges -/
theorem money_exchange_solution :
  money_exchange 13 7 4 :=
sorry

/-- Theorem stating that 13, 7, and 4 are the only initial amounts that result in each friend having 8 écus after the exchanges -/
theorem money_exchange_unique :
  ∀ a b c : ℕ, money_exchange a b c → (a = 13 ∧ b = 7 ∧ c = 4) :=
sorry

end NUMINAMATH_CALUDE_money_exchange_solution_money_exchange_unique_l1044_104485


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1044_104451

/-- Given two regular polygons with the same perimeter, where one has 50 sides
    and a side length three times that of the other, prove that the number of
    sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →                             -- Assume positive side length
  50 * (3 * s) = n * s →              -- Same perimeter condition
  n = 150 := by
sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1044_104451


namespace NUMINAMATH_CALUDE_sum_cubes_minus_product_l1044_104426

theorem sum_cubes_minus_product (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 15)
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 40) :
  a^3 + b^3 + c^3 + d^3 - 3*a*b*c*d = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_cubes_minus_product_l1044_104426


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1044_104406

theorem polynomial_division_remainder : ∃ q r : Polynomial ℚ, 
  (X : Polynomial ℚ)^4 = (X^2 + 4*X + 1) * q + r ∧ 
  r.degree < (X^2 + 4*X + 1).degree ∧ 
  r = -56*X - 15 := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1044_104406


namespace NUMINAMATH_CALUDE_product_of_numbers_l1044_104495

theorem product_of_numbers (x y : ℝ) 
  (h1 : (x + y) / (x - y) = 7)
  (h2 : (x * y) / (x - y) = 24) : 
  x * y = 48 := by
sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1044_104495


namespace NUMINAMATH_CALUDE_intersection_M_N_l1044_104497

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 ∪ {2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1044_104497


namespace NUMINAMATH_CALUDE_x_value_for_purely_imaginary_square_l1044_104403

-- Define a complex number
def complex (a b : ℝ) := a + b * Complex.I

-- Define what it means for a complex number to be purely imaginary
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem x_value_for_purely_imaginary_square (x : ℝ) :
  x > 0 → isPurelyImaginary ((x - complex 0 1) ^ 2) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_for_purely_imaginary_square_l1044_104403


namespace NUMINAMATH_CALUDE_nine_digit_repeat_gcd_l1044_104410

theorem nine_digit_repeat_gcd : 
  ∃ (d : ℕ), ∀ (n : ℕ), 100 ≤ n → n < 1000 → 
  Nat.gcd d (1001001 * n) = 1001001 ∧
  ∀ (m : ℕ), 100 ≤ m → m < 1000 → Nat.gcd d (1001001 * m) ∣ 1001001 :=
by sorry

end NUMINAMATH_CALUDE_nine_digit_repeat_gcd_l1044_104410


namespace NUMINAMATH_CALUDE_revenue_increase_when_doubled_l1044_104474

/-- Production function model -/
noncomputable def Q (A K L α₁ α₂ : ℝ) : ℝ := A * K^α₁ * L^α₂

/-- Theorem: When α₁ + α₂ > 1, doubling inputs more than doubles revenue -/
theorem revenue_increase_when_doubled
  (A K L α₁ α₂ : ℝ)
  (h_A : A > 0)
  (h_α₁ : 0 < α₁ ∧ α₁ < 1)
  (h_α₂ : 0 < α₂ ∧ α₂ < 1)
  (h_sum : α₁ + α₂ > 1) :
  Q A (2 * K) (2 * L) α₁ α₂ > 2 * Q A K L α₁ α₂ :=
sorry

end NUMINAMATH_CALUDE_revenue_increase_when_doubled_l1044_104474


namespace NUMINAMATH_CALUDE_problem_solution_l1044_104452

theorem problem_solution (x : ℝ) (h : x - 1/x = 5) : x^2 - 1/x^2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1044_104452


namespace NUMINAMATH_CALUDE_sqrt_five_minus_one_between_one_and_two_l1044_104498

theorem sqrt_five_minus_one_between_one_and_two :
  1 < Real.sqrt 5 - 1 ∧ Real.sqrt 5 - 1 < 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_five_minus_one_between_one_and_two_l1044_104498


namespace NUMINAMATH_CALUDE_middleAgedInPerformance_l1044_104430

/-- Represents the number of employees in each age group -/
structure EmployeeGroups where
  elderly : ℕ
  middleAged : ℕ
  young : ℕ

/-- Calculates the number of middle-aged employees selected in a stratified sample -/
def middleAgedSelected (total : ℕ) (groups : EmployeeGroups) (sampleSize : ℕ) : ℕ :=
  (sampleSize * groups.middleAged) / (groups.elderly + groups.middleAged + groups.young)

/-- Theorem: The number of middle-aged employees selected in the performance is 15 -/
theorem middleAgedInPerformance (total : ℕ) (groups : EmployeeGroups) (sampleSize : ℕ) 
    (h1 : total = 1200)
    (h2 : groups.elderly = 100)
    (h3 : groups.middleAged = 500)
    (h4 : groups.young = 600)
    (h5 : sampleSize = 36) :
  middleAgedSelected total groups sampleSize = 15 := by
  sorry

#eval middleAgedSelected 1200 ⟨100, 500, 600⟩ 36

end NUMINAMATH_CALUDE_middleAgedInPerformance_l1044_104430


namespace NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_smallest_positive_integer_congruence_l1044_104438

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 13 * n ≡ 567 [MOD 5] → n ≥ 4 :=
by sorry

theorem four_satisfies_congruence : 13 * 4 ≡ 567 [MOD 5] :=
by sorry

theorem four_is_smallest (m : ℕ) : m > 0 ∧ m < 4 → ¬(13 * m ≡ 567 [MOD 5]) :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃ (n : ℕ), n > 0 ∧ 13 * n ≡ 567 [MOD 5] ∧ 
  ∀ (m : ℕ), m > 0 ∧ 13 * m ≡ 567 [MOD 5] → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_four_satisfies_congruence_four_is_smallest_smallest_positive_integer_congruence_l1044_104438


namespace NUMINAMATH_CALUDE_power_equality_l1044_104407

theorem power_equality (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : b^a = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1044_104407


namespace NUMINAMATH_CALUDE_game_winnable_iff_k_leq_n_minus_one_l1044_104478

/-- Represents the game state -/
structure GameState :=
  (k : ℕ)
  (n : ℕ)
  (h1 : 2 ≤ k)
  (h2 : k ≤ n)

/-- Predicate to determine if the game is winnable -/
def is_winnable (g : GameState) : Prop :=
  g.k ≤ g.n - 1

/-- Theorem stating the condition for the game to be winnable -/
theorem game_winnable_iff_k_leq_n_minus_one (g : GameState) :
  is_winnable g ↔ g.k ≤ g.n - 1 :=
sorry

end NUMINAMATH_CALUDE_game_winnable_iff_k_leq_n_minus_one_l1044_104478


namespace NUMINAMATH_CALUDE_isosceles_triangle_third_vertex_y_coordinate_l1044_104437

/-- 
Given an isosceles triangle with:
- Base vertices at (3, 5) and (13, 5)
- Two equal sides of length 10 units
- Third vertex in the first quadrant

Prove that the y-coordinate of the third vertex is 5 + 5√3
-/
theorem isosceles_triangle_third_vertex_y_coordinate :
  ∀ (x y : ℝ),
  x > 0 →  -- First quadrant condition for x
  y > 5 →  -- First quadrant condition for y
  (x - 3)^2 + (y - 5)^2 = 100 →  -- Distance from (3, 5) is 10
  (x - 13)^2 + (y - 5)^2 = 100 →  -- Distance from (13, 5) is 10
  y = 5 + 5 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_third_vertex_y_coordinate_l1044_104437


namespace NUMINAMATH_CALUDE_equation_solution_l1044_104465

theorem equation_solution (a : ℝ) (h : a = 0.5) : 
  ∃ x : ℝ, x / (a - 3) = 3 / (a + 2) ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1044_104465


namespace NUMINAMATH_CALUDE_integer_solution_bound_l1044_104458

theorem integer_solution_bound (a : ℕ+) (x y : ℤ) 
  (h : x * (y^2 - 2*x^2) + x + y + a.val = 0) : 
  |x| ≤ a.val + Real.sqrt (2 * a.val^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_bound_l1044_104458


namespace NUMINAMATH_CALUDE_investment_interest_l1044_104408

/-- Calculates the compound interest earned given initial investment, interest rate, and time period. -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- The interest earned on a $2000 investment at 2% annual compound interest after 3 years is $122. -/
theorem investment_interest : 
  ∃ ε > 0, |compound_interest 2000 0.02 3 - 122| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_investment_interest_l1044_104408


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1044_104455

-- Define the function f
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 - b*x + c

-- State the theorem
theorem quadratic_inequality (b c : ℝ) :
  (∀ x, f b c (1 + x) = f b c (1 - x)) →
  f b c 0 = 3 →
  ∀ x, f b c (b^x) ≤ f b c (c^x) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1044_104455


namespace NUMINAMATH_CALUDE_expression_simplification_l1044_104427

theorem expression_simplification (a b : ℝ) (ha : a > 0) :
  a^(1/3) * b^(1/2) * (-3 * a^(1/2) * b^(1/3)) / ((1/3) * a^(1/6) * b^(5/6)) = -9 * a^(2/3) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1044_104427


namespace NUMINAMATH_CALUDE_runners_speed_difference_l1044_104470

/-- Given two runners starting at the same point, with one going north at 8 miles per hour
    and the other going east, if they are 5 miles apart after 1/2 hour,
    then the difference in their speeds is 2 miles per hour. -/
theorem runners_speed_difference (v : ℝ) : 
  (v ≥ 0) →  -- Ensuring non-negative speed
  ((8 * (1/2))^2 + (v * (1/2))^2 = 5^2) → 
  (8 - v = 2) :=
by sorry

end NUMINAMATH_CALUDE_runners_speed_difference_l1044_104470


namespace NUMINAMATH_CALUDE_simplify_expression_l1044_104464

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 + b^3 = a + b) :
  (a/b + b/a)^2 - 1/(a^2*b^2) = 2/(a*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1044_104464


namespace NUMINAMATH_CALUDE_savings_calculation_l1044_104432

def thomas_monthly_savings : ℕ := 40
def saving_years : ℕ := 6
def months_per_year : ℕ := 12
def joseph_savings_ratio : ℚ := 3 / 5  -- Joseph saves 2/5 less, so he saves 3/5 of what Thomas saves

def total_savings : ℕ := 4608

theorem savings_calculation :
  thomas_monthly_savings * saving_years * months_per_year +
  (thomas_monthly_savings * joseph_savings_ratio).floor * saving_years * months_per_year = total_savings :=
by sorry

end NUMINAMATH_CALUDE_savings_calculation_l1044_104432


namespace NUMINAMATH_CALUDE_sequence_20th_term_l1044_104409

/-- Given a sequence {aₙ} where a₁ = 1 and aₙ₊₁ = aₙ + 2 for n ∈ ℕ*, prove that a₂₀ = 39 -/
theorem sequence_20th_term (a : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2) : 
  a 20 = 39 := by
  sorry

end NUMINAMATH_CALUDE_sequence_20th_term_l1044_104409


namespace NUMINAMATH_CALUDE_house_sale_percentage_l1044_104483

theorem house_sale_percentage (market_value : ℝ) (num_people : ℕ) (after_tax_per_person : ℝ) (tax_rate : ℝ) :
  market_value = 500000 →
  num_people = 4 →
  after_tax_per_person = 135000 →
  tax_rate = 0.1 →
  ((num_people * after_tax_per_person / (1 - tax_rate) - market_value) / market_value) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_percentage_l1044_104483


namespace NUMINAMATH_CALUDE_monomial_count_l1044_104417

/-- A function that determines if an expression is a monomial -/
def isMonomial (expr : String) : Bool :=
  match expr with
  | "-1" => true
  | "-1/2*a^2" => true
  | "2/3*x^2*y" => true
  | "a*b^2/π" => true
  | "ab/c" => false
  | "3a-b" => false
  | "0" => true
  | "(x-1)/2" => false
  | _ => false

/-- The list of expressions to check -/
def expressions : List String :=
  ["-1", "-1/2*a^2", "2/3*x^2*y", "a*b^2/π", "ab/c", "3a-b", "0", "(x-1)/2"]

/-- Counts the number of monomials in the list of expressions -/
def countMonomials (exprs : List String) : Nat :=
  exprs.filter isMonomial |>.length

theorem monomial_count :
  countMonomials expressions = 5 := by sorry

end NUMINAMATH_CALUDE_monomial_count_l1044_104417


namespace NUMINAMATH_CALUDE_raine_steps_l1044_104449

/-- The number of steps Raine takes to walk to school -/
def steps_to_school : ℕ := 150

/-- The number of days Raine walks to and from school -/
def days : ℕ := 5

/-- The total number of steps Raine takes in five days -/
def total_steps : ℕ := 2 * steps_to_school * days

theorem raine_steps : total_steps = 1500 := by
  sorry

end NUMINAMATH_CALUDE_raine_steps_l1044_104449


namespace NUMINAMATH_CALUDE_tangent_ratio_given_sine_condition_l1044_104413

theorem tangent_ratio_given_sine_condition (α : Real) 
  (h : 5 * Real.sin (2 * α) = Real.sin (2 * Real.pi / 180)) : 
  Real.tan (α + Real.pi / 180) / Real.tan (α - Real.pi / 180) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_ratio_given_sine_condition_l1044_104413


namespace NUMINAMATH_CALUDE_fish_remaining_l1044_104412

theorem fish_remaining (initial : Float) (given_away : Float) : 
  initial = 47.0 → given_away = 22.5 → initial - given_away = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_fish_remaining_l1044_104412


namespace NUMINAMATH_CALUDE_logarithm_simplification_l1044_104416

theorem logarithm_simplification :
  (Real.log 2 / Real.log 6)^2 + (Real.log 2 / Real.log 6) * (Real.log 3 / Real.log 6) +
  2 * (Real.log 3 / Real.log 6) - 6^(Real.log 2 / Real.log 6) = -(Real.log 2 / Real.log 6) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l1044_104416


namespace NUMINAMATH_CALUDE_pages_copied_for_thirty_dollars_l1044_104484

/-- Given the rate of copying 5 pages for 8 cents, prove that $30 (3000 cents) will allow copying 1875 pages. -/
theorem pages_copied_for_thirty_dollars :
  let rate : ℚ := 5 / 8 -- pages per cent
  let total_cents : ℕ := 3000 -- $30 in cents
  (rate * total_cents : ℚ) = 1875 := by sorry

end NUMINAMATH_CALUDE_pages_copied_for_thirty_dollars_l1044_104484


namespace NUMINAMATH_CALUDE_triangle_3_4_5_l1044_104491

/-- Triangle inequality theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
axiom triangle_inequality (a b c : ℝ) : a > 0 ∧ b > 0 ∧ c > 0 → a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that determines if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that line segments of lengths 3, 4, and 5 can form a triangle -/
theorem triangle_3_4_5 : can_form_triangle 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_3_4_5_l1044_104491


namespace NUMINAMATH_CALUDE_multiplicative_inverse_201_mod_299_l1044_104489

theorem multiplicative_inverse_201_mod_299 :
  ∃! x : ℕ, x < 299 ∧ (201 * x) % 299 = 1 :=
by
  use 180
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_201_mod_299_l1044_104489


namespace NUMINAMATH_CALUDE_ellipse_equation_and_intersection_l1044_104421

/-- An ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ := (0, 0)
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- Theorem about the ellipse equation and intersection with a line -/
theorem ellipse_equation_and_intersection
  (e : Ellipse)
  (h1 : e.center = (0, 0))
  (h2 : e.foci_on_x_axis = true)
  (h3 : e.eccentricity = Real.sqrt 3 / 2)
  (h4 : e.passes_through = (4, 1)) :
  (∃ (a b : ℝ), ∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 20 + y^2 / 5 = 1)) ∧
  (∀ m : ℝ, (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁^2 / 20 + y₁^2 / 5 = 1) ∧ (y₁ = x₁ + m) ∧
    (x₂^2 / 20 + y₂^2 / 5 = 1) ∧ (y₂ = x₂ + m)) ↔
   (-5 < m ∧ m < 5)) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_and_intersection_l1044_104421


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1044_104453

/-- The distance between the vertices of the hyperbola x²/121 - y²/36 = 1 is 22 -/
theorem hyperbola_vertices_distance : 
  let a : ℝ := Real.sqrt 121
  let b : ℝ := Real.sqrt 36
  let hyperbola := fun (x y : ℝ) ↦ x^2 / 121 - y^2 / 36 = 1
  2 * a = 22 := by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1044_104453


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l1044_104469

/-- Given a rectangle ABCD with area a + 4√3, where the lines joining the centers of 
    circles inscribed in its corners form an equilateral triangle with side length 2, 
    prove that a = 8. -/
theorem rectangle_area_proof (a : ℝ) : 
  let triangle_side_length : ℝ := 2
  let rectangle_width : ℝ := triangle_side_length + triangle_side_length * Real.sqrt 3 / 2
  let rectangle_height : ℝ := 4
  let rectangle_area : ℝ := a + 4 * Real.sqrt 3
  rectangle_area = rectangle_width * rectangle_height → a = 8 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l1044_104469


namespace NUMINAMATH_CALUDE_xy_value_l1044_104466

theorem xy_value (x y : ℝ) (h : x^2 + 2*y^2 - 2*x*y + 4*y + 4 = 0) : x^y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1044_104466


namespace NUMINAMATH_CALUDE_inequality_for_positive_integers_l1044_104434

theorem inequality_for_positive_integers (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_positive_integers_l1044_104434


namespace NUMINAMATH_CALUDE_encode_decode_natural_numbers_l1044_104405

/-- Given a list of 100 natural numbers, we can encode them into a single number. -/
theorem encode_decode_natural_numbers :
  ∃ (encode : (Fin 100 → ℕ) → ℕ) (decode : ℕ → (Fin 100 → ℕ)),
    ∀ (nums : Fin 100 → ℕ), decode (encode nums) = nums :=
by sorry

end NUMINAMATH_CALUDE_encode_decode_natural_numbers_l1044_104405


namespace NUMINAMATH_CALUDE_smallest_with_16_divisors_l1044_104423

/-- The number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- n has exactly 16 different positive integer divisors -/
def has_16_divisors (n : ℕ) : Prop := num_divisors n = 16

theorem smallest_with_16_divisors : 
  ∃ (n : ℕ), has_16_divisors n ∧ ∀ (m : ℕ), has_16_divisors m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_with_16_divisors_l1044_104423


namespace NUMINAMATH_CALUDE_complex_exponential_form_l1044_104424

theorem complex_exponential_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ) (θ : ℝ), z = r * Complex.exp (Complex.I * θ) ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_form_l1044_104424
