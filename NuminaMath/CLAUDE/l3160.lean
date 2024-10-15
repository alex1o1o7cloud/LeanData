import Mathlib

namespace NUMINAMATH_CALUDE_course_selection_theorem_l3160_316043

def physical_education_courses : ℕ := 4
def art_courses : ℕ := 4
def total_courses : ℕ := physical_education_courses + art_courses

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def two_course_selections : ℕ := (choose physical_education_courses 1) * (choose art_courses 1)

def three_course_selections : ℕ := 
  (choose physical_education_courses 2) * (choose art_courses 1) +
  (choose physical_education_courses 1) * (choose art_courses 2)

def total_selections : ℕ := two_course_selections + three_course_selections

theorem course_selection_theorem : total_selections = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_theorem_l3160_316043


namespace NUMINAMATH_CALUDE_water_drinkers_l3160_316051

theorem water_drinkers (total : ℕ) (fruit_juice : ℕ) (h1 : fruit_juice = 140) 
  (h2 : (fruit_juice : ℚ) / total = 7 / 10) : 
  (total - fruit_juice : ℚ) = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_drinkers_l3160_316051


namespace NUMINAMATH_CALUDE_circle_condition_l3160_316066

/-- A circle in the xy-plane can be represented by an equation of the form
    (x - h)^2 + (y - k)^2 = r^2, where (h, k) is the center and r is the radius. --/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The main theorem: if x^2 + y^2 - x + y + m = 0 represents a circle, then m < 1/2 --/
theorem circle_condition (m : ℝ) 
  (h : is_circle (fun x y => x^2 + y^2 - x + y + m)) : 
  m < (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l3160_316066


namespace NUMINAMATH_CALUDE_fourth_root_of_12960000_l3160_316098

theorem fourth_root_of_12960000 : Real.sqrt (Real.sqrt 12960000) = 60 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_12960000_l3160_316098


namespace NUMINAMATH_CALUDE_work_completion_multiple_l3160_316099

/-- Given that some number of people can complete a work in 24 days,
    this theorem proves that 4 times that number of people
    can complete half the work in 6 days. -/
theorem work_completion_multiple :
  ∀ (P : ℕ) (W : ℝ),
  P > 0 →
  W > 0 →
  ∃ (m : ℕ),
    (P * 24 : ℝ) * W = (m * P * 6 : ℝ) * (W / 2) ∧
    m = 4 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_multiple_l3160_316099


namespace NUMINAMATH_CALUDE_ngon_division_formula_l3160_316009

/-- The number of parts into which the diagonals of an n-gon divide it,
    given that no three diagonals intersect at a single point. -/
def ngon_division (n : ℕ) : ℕ :=
  (n * (n - 1) * (n - 2) * (n - 3)) / 24 + (n * (n - 3)) / 2 + 1

/-- Theorem stating that the number of parts into which the diagonals
    of an n-gon divide it, given that no three diagonals intersect at
    a single point, is equal to the formula derived. -/
theorem ngon_division_formula (n : ℕ) (h : n ≥ 3) :
  ngon_division n = (n * (n - 1) * (n - 2) * (n - 3)) / 24 + (n * (n - 3)) / 2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_ngon_division_formula_l3160_316009


namespace NUMINAMATH_CALUDE_existence_of_n_for_k_l3160_316038

/-- f₂(n) is the number of divisors of n which are perfect squares -/
def f₂ (n : ℕ+) : ℕ := sorry

/-- f₃(n) is the number of divisors of n which are perfect cubes -/
def f₃ (n : ℕ+) : ℕ := sorry

/-- For all positive integers k, there exists a positive integer n such that f₂(n)/f₃(n) = k -/
theorem existence_of_n_for_k (k : ℕ+) : ∃ n : ℕ+, (f₂ n : ℚ) / (f₃ n : ℚ) = k := by sorry

end NUMINAMATH_CALUDE_existence_of_n_for_k_l3160_316038


namespace NUMINAMATH_CALUDE_area_above_line_is_two_thirds_l3160_316053

/-- A square in a 2D plane -/
structure Square where
  bottom_left : ℝ × ℝ
  top_right : ℝ × ℝ

/-- A line in a 2D plane defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Calculate the area of a square -/
def square_area (s : Square) : ℝ :=
  let (x1, y1) := s.bottom_left
  let (x2, y2) := s.top_right
  (x2 - x1) * (y2 - y1)

/-- Calculate the area of the region above a line in a square -/
noncomputable def area_above_line (s : Square) (l : Line) : ℝ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the area above the specified line is 2/3 of the square's area -/
theorem area_above_line_is_two_thirds (s : Square) (l : Line) : 
  s.bottom_left = (2, 1) ∧ 
  s.top_right = (5, 4) ∧ 
  l.point1 = (2, 1) ∧ 
  l.point2 = (5, 3) → 
  area_above_line s l = (2/3) * square_area s := by
  sorry


end NUMINAMATH_CALUDE_area_above_line_is_two_thirds_l3160_316053


namespace NUMINAMATH_CALUDE_trig_identity_l3160_316082

theorem trig_identity : Real.sin (18 * π / 180) * Real.sin (78 * π / 180) - 
  Real.cos (162 * π / 180) * Real.cos (78 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l3160_316082


namespace NUMINAMATH_CALUDE_min_max_m_l3160_316091

theorem min_max_m (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (eq1 : 3 * a + 2 * b + c = 5) (eq2 : 2 * a + b - 3 * c = 1) :
  let m := 3 * a + b - 7 * c
  ∃ (m_min m_max : ℝ), (∀ m', m' = m → m' ≥ m_min) ∧
                       (∀ m', m' = m → m' ≤ m_max) ∧
                       m_min = -5/7 ∧ m_max = -1/11 := by
  sorry

end NUMINAMATH_CALUDE_min_max_m_l3160_316091


namespace NUMINAMATH_CALUDE_xy_value_l3160_316033

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : x^2 + y^2 = 2) (h2 : x^4 + y^4 = 15/8) :
  x * y = Real.sqrt 17 / 4 := by
sorry

end NUMINAMATH_CALUDE_xy_value_l3160_316033


namespace NUMINAMATH_CALUDE_f_local_min_g_max_local_min_l3160_316013

noncomputable section

open Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := exp (x - 1) - log x

-- Define the function g(x) parameterized by a
def g (a : ℝ) (x : ℝ) : ℝ := f x - a * (x - 1)

-- Theorem for the local minimum of f(x)
theorem f_local_min : ∃ x₀ : ℝ, x₀ > 0 ∧ IsLocalMin f x₀ ∧ f x₀ = 1 := by sorry

-- Theorem for the maximum of the local minimum of g(x)
theorem g_max_local_min : 
  ∃ a₀ : ℝ, ∀ a : ℝ, 
    (∃ x₀ : ℝ, x₀ > 0 ∧ IsLocalMin (g a) x₀) → 
    (∃ x₁ : ℝ, x₁ > 0 ∧ IsLocalMin (g a₀) x₁ ∧ g a₀ x₁ ≥ g a x₀) ∧
    (∃ x₂ : ℝ, x₂ > 0 ∧ IsLocalMin (g a₀) x₂ ∧ g a₀ x₂ = 1) := by sorry

end

end NUMINAMATH_CALUDE_f_local_min_g_max_local_min_l3160_316013


namespace NUMINAMATH_CALUDE_solve_average_weight_l3160_316012

def average_weight_problem (weight_16 : ℝ) (weight_all : ℝ) (num_16 : ℕ) (num_8 : ℕ) : Prop :=
  let num_total : ℕ := num_16 + num_8
  let weight_8 : ℝ := (num_total * weight_all - num_16 * weight_16) / num_8
  weight_16 = 50.25 ∧ 
  weight_all = 48.55 ∧ 
  num_16 = 16 ∧ 
  num_8 = 8 ∧ 
  weight_8 = 45.15

theorem solve_average_weight : 
  ∃ (weight_16 weight_all : ℝ) (num_16 num_8 : ℕ), 
    average_weight_problem weight_16 weight_all num_16 num_8 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_average_weight_l3160_316012


namespace NUMINAMATH_CALUDE_fraction_problem_l3160_316088

theorem fraction_problem (f : ℚ) : f * 16 + 5 = 13 → f = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3160_316088


namespace NUMINAMATH_CALUDE_parabola_intersection_l3160_316062

theorem parabola_intersection (m : ℝ) (h : m > 0) :
  let f (x : ℝ) := x^2 + 2*m*x - (5/4)*m^2
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (∃ x₁ x₂, x₁ < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₂ - x₁ = 6 → m = 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l3160_316062


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_specific_perpendicular_line_l3160_316078

/-- A line passing through a point and perpendicular to another line -/
theorem perpendicular_line_through_point 
  (x₀ y₀ : ℝ) 
  (a b c : ℝ) 
  (h₁ : b ≠ 0) 
  (h₂ : a ≠ 0) :
  ∃ m k : ℝ, 
    (y₀ = m * x₀ + k) ∧ 
    (m = -a / b) ∧
    (k = y₀ - m * x₀) :=
sorry

/-- The specific line passing through (3, -5) and perpendicular to 2x - 6y + 15 = 0 -/
theorem specific_perpendicular_line : 
  ∃ m k : ℝ, 
    (-5 = m * 3 + k) ∧ 
    (m = -(2 : ℝ) / (-6 : ℝ)) ∧ 
    (k = -5 - m * 3) ∧
    (k = -4) ∧ 
    (m = 3) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_specific_perpendicular_line_l3160_316078


namespace NUMINAMATH_CALUDE_total_money_l3160_316090

def cecil_money : ℕ := 600

def catherine_money : ℕ := 2 * cecil_money - 250

def carmela_money : ℕ := 2 * cecil_money + 50

theorem total_money : cecil_money + catherine_money + carmela_money = 2800 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3160_316090


namespace NUMINAMATH_CALUDE_relay_team_permutations_l3160_316019

def team_size : ℕ := 4
def fixed_runner : String := "Lisa"
def fixed_lap : ℕ := 2

theorem relay_team_permutations :
  let remaining_runners := team_size - 1
  let free_laps := team_size - 1
  (remaining_runners.factorial : ℕ) = 6 := by sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l3160_316019


namespace NUMINAMATH_CALUDE_min_value_expression_l3160_316052

theorem min_value_expression (x y : ℝ) : 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) + y^2 ≥ -208.25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3160_316052


namespace NUMINAMATH_CALUDE_complement_of_M_l3160_316003

-- Define the universal set U as the set of real numbers
def U := Set ℝ

-- Define the set M
def M : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_of_M (x : ℝ) : 
  x ∈ (Set.univ \ M) ↔ x ≤ -1 ∨ 2 < x := by
  sorry

end NUMINAMATH_CALUDE_complement_of_M_l3160_316003


namespace NUMINAMATH_CALUDE_existence_of_large_subset_l3160_316080

/-- A family of 3-element subsets with at most one common element between any two subsets -/
def ValidFamily (I : Finset Nat) (A : Set (Finset Nat)) : Prop :=
  ∀ a ∈ A, a.card = 3 ∧ a ⊆ I ∧ ∀ b ∈ A, a ≠ b → (a ∩ b).card ≤ 1

/-- The theorem statement -/
theorem existence_of_large_subset (n : Nat) (I : Finset Nat) (hI : I.card = n) 
    (A : Set (Finset Nat)) (hA : ValidFamily I A) :
  ∃ X : Finset Nat, X ⊆ I ∧ 
    (∀ a ∈ A, ¬(a ⊆ X)) ∧ 
    X.card ≥ Nat.floor (Real.sqrt (2 * n)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_large_subset_l3160_316080


namespace NUMINAMATH_CALUDE_circle_area_increase_l3160_316071

theorem circle_area_increase (π : ℝ) (h : π > 0) : 
  π * 5^2 - π * 2^2 = 21 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_increase_l3160_316071


namespace NUMINAMATH_CALUDE_storage_volume_calculation_l3160_316014

/-- Converts cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet (yards : ℝ) : ℝ := 27 * yards

/-- Calculates the total volume in cubic feet -/
def total_volume (initial_yards : ℝ) (additional_feet : ℝ) : ℝ :=
  cubic_yards_to_cubic_feet initial_yards + additional_feet

/-- Theorem: The total volume is 180 cubic feet -/
theorem storage_volume_calculation :
  total_volume 5 45 = 180 := by
  sorry

end NUMINAMATH_CALUDE_storage_volume_calculation_l3160_316014


namespace NUMINAMATH_CALUDE_ring_arrangement_count_l3160_316035

def number_of_rings : ℕ := 10
def rings_to_arrange : ℕ := 6
def number_of_fingers : ℕ := 5

def ring_arrangements (n k : ℕ) : ℕ := (n.choose k) * k.factorial

def finger_distributions (m n : ℕ) : ℕ := (m + n - 1).choose n

theorem ring_arrangement_count :
  ring_arrangements number_of_rings rings_to_arrange * 
  finger_distributions (rings_to_arrange + number_of_fingers - 1) number_of_fingers = 31752000 :=
by sorry

end NUMINAMATH_CALUDE_ring_arrangement_count_l3160_316035


namespace NUMINAMATH_CALUDE_modulus_of_complex_l3160_316040

/-- Given that i is the imaginary unit and z is defined as z = (2+i)/i, prove that |z| = √5 -/
theorem modulus_of_complex (i : ℂ) (z : ℂ) :
  i * i = -1 →
  z = (2 + i) / i →
  Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l3160_316040


namespace NUMINAMATH_CALUDE_no_odd_total_students_l3160_316072

theorem no_odd_total_students (B : ℕ) (T : ℕ) : 
  (T = B + (7.25 * B : ℚ).floor) → 
  (50 ≤ T ∧ T ≤ 150) → 
  ¬(T % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_no_odd_total_students_l3160_316072


namespace NUMINAMATH_CALUDE_train_meeting_theorem_l3160_316073

/-- Represents the meeting point of three trains given their speeds and departure times. -/
structure TrainMeeting where
  speed_a : ℝ
  speed_b : ℝ
  speed_c : ℝ
  time_b_after_a : ℝ
  time_c_after_b : ℝ

/-- Calculates the meeting point of three trains. -/
def calculate_meeting_point (tm : TrainMeeting) : ℝ × ℝ := sorry

/-- Theorem stating the correct speed of Train C and the meeting distance. -/
theorem train_meeting_theorem (tm : TrainMeeting) 
  (h1 : tm.speed_a = 30)
  (h2 : tm.speed_b = 36)
  (h3 : tm.time_b_after_a = 2)
  (h4 : tm.time_c_after_b = 1) :
  let (speed_c, distance) := calculate_meeting_point tm
  speed_c = 45 ∧ distance = 180 := by sorry

end NUMINAMATH_CALUDE_train_meeting_theorem_l3160_316073


namespace NUMINAMATH_CALUDE_triangle_area_l3160_316041

theorem triangle_area (A B C : ℝ × ℝ) : 
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let angle_B := Real.arccos ((AC^2 + BC^2 - (A.1 - B.1)^2 - (A.2 - B.2)^2) / (2 * AC * BC))
  AC = Real.sqrt 7 ∧ BC = 2 ∧ angle_B = π/3 →
  (1/2) * AC * BC * Real.sin angle_B = (3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l3160_316041


namespace NUMINAMATH_CALUDE_power_equation_l3160_316068

theorem power_equation (a : ℝ) (m n k : ℤ) (h1 : a^m = 2) (h2 : a^n = 4) (h3 : a^k = 32) :
  a^(3*m + 2*n - k) = 4 := by
sorry

end NUMINAMATH_CALUDE_power_equation_l3160_316068


namespace NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l3160_316097

/-- The ratio of the volume of a cone to the volume of a cylinder, given specific proportions -/
theorem cone_cylinder_volume_ratio 
  (r h : ℝ) 
  (h_pos : h > 0) 
  (r_pos : r > 0) : 
  (1 / 3 * π * (r / 2)^2 * (h / 3)) / (π * r^2 * h) = 1 / 36 := by
  sorry

end NUMINAMATH_CALUDE_cone_cylinder_volume_ratio_l3160_316097


namespace NUMINAMATH_CALUDE_correct_meal_probability_l3160_316011

def number_of_people : ℕ := 12
def pasta_orders : ℕ := 5
def salad_orders : ℕ := 7

theorem correct_meal_probability : 
  let total_arrangements := Nat.factorial number_of_people
  let favorable_outcomes := 157410
  (favorable_outcomes : ℚ) / total_arrangements = 157410 / 479001600 := by
  sorry

end NUMINAMATH_CALUDE_correct_meal_probability_l3160_316011


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3160_316042

/-- Given a line L1 with equation 2x - 5y + 3 = 0 and a point P(2, -1),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 5x + 2y - 8 = 0 -/
theorem perpendicular_line_equation 
  (L1 : Set (ℝ × ℝ)) 
  (P : ℝ × ℝ) :
  (L1 = {(x, y) | 2 * x - 5 * y + 3 = 0}) →
  (P = (2, -1)) →
  (∃ L2 : Set (ℝ × ℝ), 
    (P ∈ L2) ∧ 
    (∀ (Q R : ℝ × ℝ), Q ∈ L1 → R ∈ L1 → Q ≠ R → 
      ∀ (S T : ℝ × ℝ), S ∈ L2 → T ∈ L2 → S ≠ T →
        ((Q.1 - R.1) * (S.1 - T.1) + (Q.2 - R.2) * (S.2 - T.2) = 0)) ∧
    (L2 = {(x, y) | 5 * x + 2 * y - 8 = 0})) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3160_316042


namespace NUMINAMATH_CALUDE_adjacent_product_geometric_sequence_l3160_316036

/-- Given a geometric sequence with common ratio q, prove that the sequence formed by
    the product of adjacent terms is a geometric sequence with common ratio q² -/
theorem adjacent_product_geometric_sequence (a : ℕ → ℝ) (q : ℝ) (hq : q ≠ 0) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  ∀ n : ℕ, (a (n + 1) * a (n + 2)) = q^2 * (a n * a (n + 1)) :=
by
  sorry

end NUMINAMATH_CALUDE_adjacent_product_geometric_sequence_l3160_316036


namespace NUMINAMATH_CALUDE_wrap_vs_sleeve_difference_l3160_316061

def raw_squat : ℝ := 600
def sleeve_addition : ℝ := 30
def wrap_percentage : ℝ := 0.25

theorem wrap_vs_sleeve_difference :
  (raw_squat * wrap_percentage) - sleeve_addition = 120 := by
  sorry

end NUMINAMATH_CALUDE_wrap_vs_sleeve_difference_l3160_316061


namespace NUMINAMATH_CALUDE_juggling_balls_needed_l3160_316087

/-- The number of balls needed for a juggling spectacle -/
theorem juggling_balls_needed (num_jugglers : ℕ) (balls_per_juggler : ℕ) : 
  num_jugglers = 5000 → balls_per_juggler = 12 → num_jugglers * balls_per_juggler = 60000 := by
  sorry

end NUMINAMATH_CALUDE_juggling_balls_needed_l3160_316087


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3160_316075

/-- Represents a hyperbola with equation y²/a² - x²/4 = 1 -/
structure Hyperbola where
  a : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  (1 / h.a^2 - 4 / 4 = 1) →  -- The hyperbola passes through (2, -1)
  eccentricity h = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3160_316075


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3160_316048

/-- 
Given an arithmetic sequence {a_n} with sum of first n terms S_n = n^2 - 3n,
prove that the general term formula is a_n = 2n - 4.
-/
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = n^2 - 3*n) : 
  ∀ n, a n = 2*n - 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l3160_316048


namespace NUMINAMATH_CALUDE_permutation_count_mod_1000_l3160_316069

/-- The number of permutations of a 15-character string with 4 A's, 5 B's, and 6 C's -/
def N : ℕ := sorry

/-- Condition: None of the first four letters is an A -/
axiom cond1 : sorry

/-- Condition: None of the next five letters is a B -/
axiom cond2 : sorry

/-- Condition: None of the last six letters is a C -/
axiom cond3 : sorry

/-- Theorem: The number of permutations N satisfying the conditions is congruent to 320 modulo 1000 -/
theorem permutation_count_mod_1000 : N ≡ 320 [MOD 1000] := by sorry

end NUMINAMATH_CALUDE_permutation_count_mod_1000_l3160_316069


namespace NUMINAMATH_CALUDE_certain_value_problem_l3160_316020

theorem certain_value_problem (n x : ℝ) : n = 5 ∧ n = 5 * (n - x) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_problem_l3160_316020


namespace NUMINAMATH_CALUDE_knowledge_group_theorem_l3160_316059

/-- A group of people where some know each other -/
structure KnowledgeGroup (k : ℕ) where
  knows : Fin k → Fin k → Prop
  symm : ∀ i j, knows i j ↔ knows j i

/-- For any n people, there's an (n+1)-th person who knows them all -/
def HasKnowledgeable (n : ℕ) (g : KnowledgeGroup k) : Prop :=
  ∀ (s : Finset (Fin k)), s.card = n → 
    ∃ i, i ∉ s ∧ ∀ j ∈ s, g.knows i j

theorem knowledge_group_theorem (n : ℕ) :
  (∃ (g : KnowledgeGroup (2*n + 1)), HasKnowledgeable n g → 
    ∃ i, ∀ j, g.knows i j) ∧
  (∃ (g : KnowledgeGroup (2*n + 2)), HasKnowledgeable n g ∧ 
    ∀ i, ∃ j, ¬g.knows i j) := by
  sorry

end NUMINAMATH_CALUDE_knowledge_group_theorem_l3160_316059


namespace NUMINAMATH_CALUDE_integer_solution_exists_iff_n_eq_one_l3160_316029

theorem integer_solution_exists_iff_n_eq_one (n : ℕ+) :
  (∃ x : ℤ, x^(n : ℕ) + (2 + x)^(n : ℕ) + (2 - x)^(n : ℕ) = 0) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_exists_iff_n_eq_one_l3160_316029


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3160_316005

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 7 ways to distribute 5 indistinguishable balls into 4 indistinguishable boxes -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3160_316005


namespace NUMINAMATH_CALUDE_train_crossing_time_l3160_316076

/-- The time taken for a train to cross an electric pole -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) : 
  train_length = 700 ∧ train_speed_kmh = 125.99999999999999 →
  (train_length / (train_speed_kmh * (1000 / 3600))) = 20 := by
  sorry


end NUMINAMATH_CALUDE_train_crossing_time_l3160_316076


namespace NUMINAMATH_CALUDE_ellipse_sum_l3160_316004

/-- For an ellipse with center (h, k), semi-major axis length a, and semi-minor axis length b,
    prove that h + k + a + b = 4 when the center is (3, -5), a = 4, and b = 2. -/
theorem ellipse_sum (h k a b : ℝ) : 
  h = 3 → k = -5 → a = 4 → b = 2 → h + k + a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_l3160_316004


namespace NUMINAMATH_CALUDE_quadratic_minimum_less_than_neg_six_l3160_316083

/-- A quadratic function satisfying specific point conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧
  f (-2) = 6 ∧ f 0 = -4 ∧ f 1 = -6 ∧ f 3 = -4

/-- The theorem stating that the minimum value of the quadratic function is less than -6 -/
theorem quadratic_minimum_less_than_neg_six (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  ∃ x : ℝ, ∀ y : ℝ, f y ≥ f x ∧ f x < -6 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_less_than_neg_six_l3160_316083


namespace NUMINAMATH_CALUDE_wooden_toy_price_is_20_l3160_316067

/-- The original price of each wooden toy -/
def wooden_toy_price : ℝ := 20

/-- The number of paintings bought -/
def num_paintings : ℕ := 10

/-- The original price of each painting -/
def painting_price : ℝ := 40

/-- The number of wooden toys bought -/
def num_toys : ℕ := 8

/-- The discount rate for paintings -/
def painting_discount : ℝ := 0.1

/-- The discount rate for wooden toys -/
def toy_discount : ℝ := 0.15

/-- The total loss from the sale -/
def total_loss : ℝ := 64

theorem wooden_toy_price_is_20 :
  (num_paintings * painting_price + num_toys * wooden_toy_price) -
  (num_paintings * painting_price * (1 - painting_discount) +
   num_toys * wooden_toy_price * (1 - toy_discount)) = total_loss :=
by sorry

end NUMINAMATH_CALUDE_wooden_toy_price_is_20_l3160_316067


namespace NUMINAMATH_CALUDE_total_puppies_l3160_316063

theorem total_puppies (female_puppies male_puppies : ℕ) 
  (h1 : female_puppies = 2)
  (h2 : male_puppies = 10)
  (h3 : (female_puppies : ℚ) / male_puppies = 0.2) :
  female_puppies + male_puppies = 12 := by
sorry

end NUMINAMATH_CALUDE_total_puppies_l3160_316063


namespace NUMINAMATH_CALUDE_committee_age_difference_l3160_316058

/-- Proves that the age difference between an old and new member in a committee is 40 years,
    given specific conditions about the committee's average age over time. -/
theorem committee_age_difference (n : ℕ) (A : ℝ) (O N : ℝ) : 
  n = 10 → -- The committee has 10 members
  n * A = n * A + n * 4 - (O - N) → -- The total age after 4 years minus the age difference equals the original total age
  O - N = 40 := by
  sorry

end NUMINAMATH_CALUDE_committee_age_difference_l3160_316058


namespace NUMINAMATH_CALUDE_total_age_is_23_l3160_316034

/-- Proves that the total combined age of Ryanne, Hezekiah, and Jamison is 23 years -/
theorem total_age_is_23 (hezekiah_age : ℕ) 
  (ryanne_older : ryanne_age = hezekiah_age + 7)
  (sum_ryanne_hezekiah : ryanne_age + hezekiah_age = 15)
  (jamison_twice : jamison_age = 2 * hezekiah_age) :
  hezekiah_age + ryanne_age + jamison_age = 23 :=
by
  sorry

#check total_age_is_23

end NUMINAMATH_CALUDE_total_age_is_23_l3160_316034


namespace NUMINAMATH_CALUDE_restaurant_bill_example_l3160_316070

/-- Calculates the total cost for a group at a restaurant with specific pricing and discount rules. -/
def restaurant_bill (total_people : ℕ) (num_kids : ℕ) (num_upgrades : ℕ) 
  (adult_meal_cost : ℚ) (upgrade_cost : ℚ) (adult_drink_cost : ℚ) (kid_drink_cost : ℚ) 
  (discount_rate : ℚ) : ℚ :=
  let num_adults := total_people - num_kids
  let meal_cost := num_adults * adult_meal_cost
  let upgrade_total := num_upgrades * upgrade_cost
  let drink_cost := num_adults * adult_drink_cost + num_kids * kid_drink_cost
  let subtotal := meal_cost + upgrade_total + drink_cost
  let discount := subtotal * discount_rate
  subtotal - discount

/-- Theorem stating that the total cost for the given group is $97.20 -/
theorem restaurant_bill_example : 
  restaurant_bill 11 2 4 8 4 2 1 (1/10) = 97.2 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_example_l3160_316070


namespace NUMINAMATH_CALUDE_volunteer_selection_probability_l3160_316089

theorem volunteer_selection_probability 
  (total_students : ℕ) 
  (eliminated : ℕ) 
  (selected : ℕ) 
  (h1 : total_students = 2018) 
  (h2 : eliminated = 18) 
  (h3 : selected = 50) :
  (selected : ℚ) / total_students = 25 / 1009 := by
sorry

end NUMINAMATH_CALUDE_volunteer_selection_probability_l3160_316089


namespace NUMINAMATH_CALUDE_square_equation_solution_l3160_316031

theorem square_equation_solution (b c x : ℝ) : 
  x^2 + c^2 = (b - x)^2 → x = (b^2 - c^2) / (2 * b) :=
by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l3160_316031


namespace NUMINAMATH_CALUDE_output_for_input_8_l3160_316001

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 > 22 then
    step1 - 7
  else
    step1 + 10

theorem output_for_input_8 : function_machine 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_output_for_input_8_l3160_316001


namespace NUMINAMATH_CALUDE_problem_1_l3160_316024

theorem problem_1 (x y : ℝ) : (2*x + y)^2 - 8*(2*x + y) - 9 = 0 → 2*x + y = 9 ∨ 2*x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3160_316024


namespace NUMINAMATH_CALUDE_partnership_investment_time_l3160_316018

/-- A partnership problem with three partners A, B, and C --/
theorem partnership_investment_time (x : ℝ) : 
  let total_investment := x * 12 + 2 * x * 6 + 3 * x * (12 - m)
  let m := 12 - (36 * x - 24 * x) / (3 * x)
  x > 0 → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_partnership_investment_time_l3160_316018


namespace NUMINAMATH_CALUDE_fraction_count_l3160_316060

theorem fraction_count : ∃ (S : Finset ℕ), 
  (∀ a ∈ S, a > 1 ∧ a < 7) ∧ 
  (∀ a ∉ S, a ≤ 1 ∨ a ≥ 7 ∨ (a - 1) / a ≥ 6 / 7) ∧
  S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_count_l3160_316060


namespace NUMINAMATH_CALUDE_junior_score_l3160_316015

theorem junior_score (n : ℝ) (junior_score : ℝ) : 
  n > 0 →
  0.2 * n * junior_score + 0.8 * n * 84 = n * 85 →
  junior_score = 89 := by
sorry

end NUMINAMATH_CALUDE_junior_score_l3160_316015


namespace NUMINAMATH_CALUDE_provisions_duration_l3160_316092

theorem provisions_duration (initial_soldiers : ℕ) (initial_consumption : ℚ)
  (additional_soldiers : ℕ) (new_consumption : ℚ) (new_duration : ℕ) :
  initial_soldiers = 1200 →
  initial_consumption = 3 →
  additional_soldiers = 528 →
  new_consumption = 5/2 →
  new_duration = 25 →
  (↑initial_soldiers * initial_consumption * ↑new_duration =
   ↑(initial_soldiers + additional_soldiers) * new_consumption * ↑new_duration) →
  (↑initial_soldiers * initial_consumption * (1080000 / 3600 : ℚ) =
   ↑initial_soldiers * initial_consumption * 300) :=
by sorry

end NUMINAMATH_CALUDE_provisions_duration_l3160_316092


namespace NUMINAMATH_CALUDE_expression_evaluation_l3160_316093

theorem expression_evaluation : -2^3 + (18 - (-3)^2) / (-3) = -11 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3160_316093


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_is_three_l3160_316079

/-- Right triangle PQR with inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Angle R is a right angle -/
  angle_r_is_right : True

/-- Calculate the radius of the inscribed circle in a right triangle -/
def inscribedCircleRadius (t : RightTriangleWithInscribedCircle) : ℝ :=
  sorry

/-- Theorem: The radius of the inscribed circle in the given right triangle is 3 -/
theorem inscribed_circle_radius_is_three :
  let t : RightTriangleWithInscribedCircle := ⟨15, 8, trivial⟩
  inscribedCircleRadius t = 3 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_is_three_l3160_316079


namespace NUMINAMATH_CALUDE_discount_calculation_l3160_316047

theorem discount_calculation (original_price discount_percentage : ℝ) 
  (h1 : original_price = 10)
  (h2 : discount_percentage = 10) :
  original_price * (1 - discount_percentage / 100) = 9 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l3160_316047


namespace NUMINAMATH_CALUDE_square_difference_division_problem_solution_l3160_316010

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
by sorry

theorem problem_solution : (144^2 - 121^2) / 23 = 265 :=
by sorry

end NUMINAMATH_CALUDE_square_difference_division_problem_solution_l3160_316010


namespace NUMINAMATH_CALUDE_cube_root_abs_sqrt_equality_l3160_316074

theorem cube_root_abs_sqrt_equality : 
  (64 : ℝ)^(1/3) - |Real.sqrt 3 - 3| + Real.sqrt 36 = 7 + Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_abs_sqrt_equality_l3160_316074


namespace NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_l3160_316028

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the parallelism relation
variable (parallel : Plane → Plane → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)

-- Define skew lines
variable (skew : Line → Line → Prop)

-- Theorem statement
theorem planes_parallel_from_skew_lines 
  (α β : Plane) (l m : Line) :
  skew l m →
  lineParallelToPlane l α →
  lineParallelToPlane l β →
  lineParallelToPlane m α →
  lineParallelToPlane m β →
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_l3160_316028


namespace NUMINAMATH_CALUDE_trigonometric_equality_l3160_316037

theorem trigonometric_equality (α : ℝ) :
  1 + Real.sin (3 * (α + π / 2)) * Real.cos (2 * α) +
  2 * Real.sin (3 * α) * Real.cos (3 * π - α) * Real.sin (α - π) =
  2 * (Real.sin (5 * α / 2))^2 := by sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l3160_316037


namespace NUMINAMATH_CALUDE_palindrome_with_five_percentage_l3160_316007

/-- A function that checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Bool :=
  sorry

/-- A function that checks if a natural number contains the digit 5 -/
def containsFive (n : ℕ) : Bool :=
  sorry

/-- The set of palindromes between 100 and 1000 (inclusive) -/
def palindromes : Finset ℕ :=
  sorry

/-- The set of palindromes between 100 and 1000 (inclusive) containing at least one 5 -/
def palindromesWithFive : Finset ℕ :=
  sorry

theorem palindrome_with_five_percentage :
  (palindromesWithFive.card : ℚ) / palindromes.card * 100 = 37 / 180 * 100 :=
sorry

end NUMINAMATH_CALUDE_palindrome_with_five_percentage_l3160_316007


namespace NUMINAMATH_CALUDE_midsize_to_fullsize_ratio_l3160_316026

/-- Proves that the ratio of the mid-size model's length to the full-size mustang's length is 1:10 -/
theorem midsize_to_fullsize_ratio :
  let full_size : ℝ := 240
  let smallest_size : ℝ := 12
  let mid_size : ℝ := 2 * smallest_size
  (mid_size / full_size) = (1 / 10 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_midsize_to_fullsize_ratio_l3160_316026


namespace NUMINAMATH_CALUDE_complex_cube_inequality_l3160_316085

theorem complex_cube_inequality (z : ℂ) (h : Complex.abs (z + 1) > 2) :
  Complex.abs (z^3 + 1) > 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_inequality_l3160_316085


namespace NUMINAMATH_CALUDE_boat_purchase_problem_l3160_316056

theorem boat_purchase_problem (a b c d e : ℝ) : 
  a + b + c + d + e = 120 ∧
  a = (1/3) * (b + c + d + e) ∧
  b = (1/4) * (a + c + d + e) ∧
  c = (1/5) * (a + b + d + e) ∧
  d = (1/6) * (a + b + c + e) →
  e = 40 := by sorry

end NUMINAMATH_CALUDE_boat_purchase_problem_l3160_316056


namespace NUMINAMATH_CALUDE_modular_congruence_iff_divisibility_l3160_316081

theorem modular_congruence_iff_divisibility (a n k : ℕ) (ha : a ≥ 2) :
  a ^ k ≡ 1 [MOD a ^ n - 1] ↔ n ∣ k :=
sorry

end NUMINAMATH_CALUDE_modular_congruence_iff_divisibility_l3160_316081


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l3160_316022

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflect_x (p : Point) : Point :=
  ⟨p.x, -p.y⟩

theorem reflection_across_x_axis :
  let P : Point := ⟨-3, 2⟩
  reflect_x P = ⟨-3, -2⟩ := by
  sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l3160_316022


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l3160_316095

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 6^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 11 * p * q) →
  (∀ r : ℕ, Prime r ∧ r ∣ x → r = 11 ∨ r = p ∨ r = q) →
  x = 7777 := by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l3160_316095


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3160_316065

theorem sqrt_product_equality : Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3160_316065


namespace NUMINAMATH_CALUDE_max_value_theorem_l3160_316032

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x*y - 1)^2 = (5*y + 2)*(y - 2)) : 
  x + 1/(2*y) ≤ -1 + (3*Real.sqrt 2)/2 := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3160_316032


namespace NUMINAMATH_CALUDE_decimal_to_binary_53_l3160_316030

theorem decimal_to_binary_53 : 
  (53 : ℕ) = 
  (1 * 2^5 + 1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 : ℕ) :=
by sorry

end NUMINAMATH_CALUDE_decimal_to_binary_53_l3160_316030


namespace NUMINAMATH_CALUDE_final_output_is_218_l3160_316017

def machine_transform (a : ℕ) : ℕ :=
  if a % 2 = 1 then a + 3 else a + 5

def repeated_transform (a : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => a
  | n + 1 => machine_transform (repeated_transform a n)

theorem final_output_is_218 :
  repeated_transform 15 51 = 218 := by
  sorry

end NUMINAMATH_CALUDE_final_output_is_218_l3160_316017


namespace NUMINAMATH_CALUDE_min_value_of_function_l3160_316008

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  (x + 5) * (x + 2) / (x + 1) ≥ 9 ∧
  (x + 5) * (x + 2) / (x + 1) = 9 ↔ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3160_316008


namespace NUMINAMATH_CALUDE_smallest_positive_root_of_g_l3160_316096

open Real

theorem smallest_positive_root_of_g : ∃ s : ℝ,
  s > 0 ∧
  sin s + 3 * cos s + 4 * tan s = 0 ∧
  (∀ x, 0 < x → x < s → sin x + 3 * cos x + 4 * tan x ≠ 0) ∧
  ⌊s⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_root_of_g_l3160_316096


namespace NUMINAMATH_CALUDE_jessica_cut_two_roses_l3160_316023

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_roses final_roses : ℕ) : ℕ :=
  final_roses - initial_roses

/-- Theorem stating that Jessica cut 2 roses -/
theorem jessica_cut_two_roses : roses_cut 15 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_two_roses_l3160_316023


namespace NUMINAMATH_CALUDE_vector_perpendicular_l3160_316021

/-- Given two vectors a and b in R², prove that if a + b is perpendicular to a,
    then the second component of b is -7/2. -/
theorem vector_perpendicular (a b : ℝ × ℝ) (h : a = (1, 2)) (h' : b.1 = 2) :
  (a + b) • a = 0 → b.2 = -7/2 := by sorry

end NUMINAMATH_CALUDE_vector_perpendicular_l3160_316021


namespace NUMINAMATH_CALUDE_nth_equation_l3160_316039

/-- The product of consecutive integers from n+1 to n+n -/
def leftSide (n : ℕ) : ℕ := (n + 1).factorial / n.factorial

/-- The product of odd numbers from 1 to 2n-1 -/
def oddProduct (n : ℕ) : ℕ := 
  Finset.prod (Finset.range n) (fun i => 2 * i + 1)

/-- The statement of the equality to be proved -/
theorem nth_equation (n : ℕ) : 
  leftSide n = 2^n * oddProduct n := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l3160_316039


namespace NUMINAMATH_CALUDE_shoe_price_calculation_l3160_316049

theorem shoe_price_calculation (initial_price : ℝ) (friday_increase : ℝ) (monday_decrease : ℝ) : 
  initial_price = 50 → 
  friday_increase = 0.20 → 
  monday_decrease = 0.15 → 
  initial_price * (1 + friday_increase) * (1 - monday_decrease) = 51 := by
sorry

end NUMINAMATH_CALUDE_shoe_price_calculation_l3160_316049


namespace NUMINAMATH_CALUDE_shipwreck_year_conversion_l3160_316086

/-- Converts an octal number to its decimal equivalent -/
def octal_to_decimal (octal : Nat) : Nat :=
  let hundreds := octal / 100
  let tens := (octal / 10) % 10
  let ones := octal % 10
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The octal year of the shipwreck -/
def shipwreck_year_octal : Nat := 536

theorem shipwreck_year_conversion :
  octal_to_decimal shipwreck_year_octal = 350 := by
  sorry

end NUMINAMATH_CALUDE_shipwreck_year_conversion_l3160_316086


namespace NUMINAMATH_CALUDE_factor_implies_m_value_l3160_316050

theorem factor_implies_m_value (m : ℝ) : 
  (∃ k : ℝ, ∀ x : ℝ, x^2 - m*x - 40 = (x + 5) * k) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_m_value_l3160_316050


namespace NUMINAMATH_CALUDE_non_pine_trees_l3160_316006

theorem non_pine_trees (total : ℕ) (pine_percentage : ℚ) (non_pine : ℕ) : 
  total = 350 → pine_percentage = 70 / 100 → 
  non_pine = total - (pine_percentage * total).floor → non_pine = 105 :=
by sorry

end NUMINAMATH_CALUDE_non_pine_trees_l3160_316006


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l3160_316094

def batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) : ℕ :=
  let previous_average := (total_innings - 1) * (average_increase + (last_innings_score / total_innings))
  let new_total_score := previous_average + last_innings_score
  new_total_score / total_innings

theorem batsman_average_theorem :
  batsman_average 17 80 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_theorem_l3160_316094


namespace NUMINAMATH_CALUDE_sachin_age_l3160_316077

theorem sachin_age : 
  ∀ (s r : ℕ), 
  r = s + 8 →  -- Sachin is younger than Rahul by 8 years
  s * 9 = r * 7 →  -- The ratio of their ages is 7 : 9
  s = 28 :=  -- Sachin's age is 28 years
by
  sorry

end NUMINAMATH_CALUDE_sachin_age_l3160_316077


namespace NUMINAMATH_CALUDE_triangle_side_length_l3160_316002

theorem triangle_side_length (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 2 * Real.sqrt 3) (h3 : B = π / 6) :
  ∃ b : ℝ, b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3160_316002


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3160_316064

theorem quadratic_equations_solutions :
  (∀ x, 2 * x^2 - 4 * x = 0 ↔ x = 0 ∨ x = 2) ∧
  (∀ x, x^2 - 6 * x - 6 = 0 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3160_316064


namespace NUMINAMATH_CALUDE_rectangular_plot_dimensions_l3160_316045

theorem rectangular_plot_dimensions (length width : ℝ) : 
  length = 58 →
  (4 * width + 2 * length) * 26.5 = 5300 →
  length - width = 37 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_dimensions_l3160_316045


namespace NUMINAMATH_CALUDE_min_sales_to_break_even_l3160_316000

def old_salary : ℕ := 75000
def new_base_salary : ℕ := 45000
def commission_rate : ℚ := 15 / 100
def sale_amount : ℕ := 750

theorem min_sales_to_break_even :
  let difference := old_salary - new_base_salary
  let commission_per_sale := commission_rate * sale_amount
  let min_sales := (difference : ℚ) / commission_per_sale
  ⌈min_sales⌉ = 267 := by sorry

end NUMINAMATH_CALUDE_min_sales_to_break_even_l3160_316000


namespace NUMINAMATH_CALUDE_smallest_m_for_probability_l3160_316027

def probability_condition (m : ℕ) : Prop :=
  (m - 1)^4 > (3/4) * m^4

theorem smallest_m_for_probability : 
  probability_condition 17 ∧ 
  ∀ k : ℕ, k < 17 → ¬ probability_condition k :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_probability_l3160_316027


namespace NUMINAMATH_CALUDE_incorrect_inequality_transformation_l3160_316016

theorem incorrect_inequality_transformation (x y : ℝ) (h : x < y) : ¬(-2*x < -2*y) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_transformation_l3160_316016


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3160_316044

theorem unique_solution_for_equation : 
  ∀ (n k : ℕ), 2023 + 2^n = k^2 ↔ n = 1 ∧ k = 45 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3160_316044


namespace NUMINAMATH_CALUDE_ratio_to_percentage_increase_l3160_316084

theorem ratio_to_percentage_increase (A B : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : A / B = 1/6 / (1/5)) :
  (B - A) / A * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_percentage_increase_l3160_316084


namespace NUMINAMATH_CALUDE_range_of_ratio_l3160_316057

theorem range_of_ratio (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) : 
  ∃ (k : ℝ), k = |y / (x + 1)| ∧ k ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_ratio_l3160_316057


namespace NUMINAMATH_CALUDE_raffle_donation_calculation_l3160_316046

theorem raffle_donation_calculation (num_tickets : ℕ) (ticket_price : ℚ) 
  (total_raised : ℚ) (fixed_donation : ℚ) :
  num_tickets = 25 →
  ticket_price = 2 →
  total_raised = 100 →
  fixed_donation = 20 →
  ∃ (equal_donation : ℚ),
    equal_donation * 2 + fixed_donation = total_raised - (num_tickets : ℚ) * ticket_price ∧
    equal_donation = 15 := by
  sorry

end NUMINAMATH_CALUDE_raffle_donation_calculation_l3160_316046


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l3160_316025

theorem complex_modulus_problem (z : ℂ) (h : 3 + z * Complex.I = z - 3 * Complex.I) : 
  Complex.abs z = 3 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l3160_316025


namespace NUMINAMATH_CALUDE_square_sum_equals_two_l3160_316054

theorem square_sum_equals_two (a b : ℝ) (h1 : a * b = -1) (h2 : a - b = 2) : a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_two_l3160_316054


namespace NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3160_316055

/-- A Mersenne number is of the form 2^n - 1 --/
def mersenne_number (n : ℕ) : ℕ := 2^n - 1

/-- A Mersenne prime is a Mersenne number that is prime --/
def is_mersenne_prime (p : ℕ) : Prop :=
  ∃ n, Prime n ∧ p = mersenne_number n ∧ Prime p

/-- The largest Mersenne prime less than 500 is 127 --/
theorem largest_mersenne_prime_under_500 :
  (∀ p, is_mersenne_prime p → p < 500 → p ≤ 127) ∧
  is_mersenne_prime 127 :=
sorry

end NUMINAMATH_CALUDE_largest_mersenne_prime_under_500_l3160_316055
