import Mathlib

namespace NUMINAMATH_CALUDE_rationalize_denominator_l1330_133051

theorem rationalize_denominator :
  7 / (2 * Real.sqrt 50) = (7 * Real.sqrt 2) / 20 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1330_133051


namespace NUMINAMATH_CALUDE_max_minute_hands_l1330_133049

/-- Represents the number of coincidences per hour for a pair of hands moving in opposite directions -/
def coincidences_per_pair : ℕ := 120

/-- Represents the total number of coincidences observed in one hour -/
def total_coincidences : ℕ := 54

/-- Proves that the maximum number of minute hands is 28 given the conditions -/
theorem max_minute_hands : 
  ∃ (m n : ℕ), 
    m * n = total_coincidences / 2 ∧ 
    m + n ≤ 28 ∧ 
    ∀ (k l : ℕ), k * l = total_coincidences / 2 → k + l ≤ m + n :=
by sorry

end NUMINAMATH_CALUDE_max_minute_hands_l1330_133049


namespace NUMINAMATH_CALUDE_initial_percent_problem_l1330_133053

theorem initial_percent_problem (x : ℝ) : 
  (x / 100) * (5 / 100) = 60 / 100 → x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_initial_percent_problem_l1330_133053


namespace NUMINAMATH_CALUDE_adams_balls_l1330_133096

theorem adams_balls (red : ℕ) (blue : ℕ) (pink : ℕ) (orange : ℕ) : 
  red = 20 → 
  blue = 10 → 
  pink = 3 * orange → 
  orange = 5 → 
  red + blue + pink + orange = 50 := by
sorry

end NUMINAMATH_CALUDE_adams_balls_l1330_133096


namespace NUMINAMATH_CALUDE_solution_set_when_a_eq_1_range_when_a_ge_1_l1330_133055

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 2|

-- Part 1: Solution set when a = 1
theorem solution_set_when_a_eq_1 :
  {x : ℝ | f 1 x ≤ 5} = Set.Icc (-3) 2 := by sorry

-- Part 2: Range of f(x) when a ≥ 1
theorem range_when_a_ge_1 (a : ℝ) (h : a ≥ 1) :
  Set.range (f a) = Set.Ici (a + 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_eq_1_range_when_a_ge_1_l1330_133055


namespace NUMINAMATH_CALUDE_f_inequality_l1330_133033

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * exp x - k * x^2

theorem f_inequality (k : ℝ) (h1 : k > 1/2) 
  (h2 : ∀ x > 0, f k x + (log (2*k))^2 + 2*k * log (exp 1 / (2*k)) > 0) :
  f k (k - 1 + log 2) < f k k := by
sorry

end NUMINAMATH_CALUDE_f_inequality_l1330_133033


namespace NUMINAMATH_CALUDE_train_length_l1330_133072

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 10 → speed * time * (1000 / 3600) = 250 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l1330_133072


namespace NUMINAMATH_CALUDE_chord_bisector_l1330_133076

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem chord_bisector :
  ellipse P.1 P.2 →
  ∃ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    P = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
    (∀ (x y : ℝ), line_equation x y ↔ ∃ t : ℝ, x = A.1 + t*(B.1 - A.1) ∧ y = A.2 + t*(B.2 - A.2)) :=
sorry

end NUMINAMATH_CALUDE_chord_bisector_l1330_133076


namespace NUMINAMATH_CALUDE_complement_A_union_B_a_lower_bound_l1330_133095

-- Define the sets A, B, and C
def A : Set ℝ := {x | 1 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem for part I
theorem complement_A_union_B :
  (Set.univ \ A) ∪ B = {x : ℝ | x < 1 ∨ x > 2} := by sorry

-- Theorem for part II
theorem a_lower_bound (h : A ⊆ C a) : a ≥ 7 := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_a_lower_bound_l1330_133095


namespace NUMINAMATH_CALUDE_product_closure_l1330_133004

def A : Set ℤ := {z | ∃ a b : ℤ, z = a^2 + 4*a*b + b^2}

theorem product_closure (x y : ℤ) (hx : x ∈ A) (hy : y ∈ A) : x * y ∈ A := by
  sorry

end NUMINAMATH_CALUDE_product_closure_l1330_133004


namespace NUMINAMATH_CALUDE_discretionary_income_ratio_l1330_133014

/-- Represents Jill's financial situation --/
structure JillFinances where
  netSalary : ℝ
  discretionaryIncome : ℝ
  vacationFundPercent : ℝ
  savingsPercent : ℝ
  socializingPercent : ℝ
  giftsAmount : ℝ

/-- The conditions of Jill's finances --/
def jillFinancesConditions (j : JillFinances) : Prop :=
  j.netSalary = 3300 ∧
  j.vacationFundPercent = 0.3 ∧
  j.savingsPercent = 0.2 ∧
  j.socializingPercent = 0.35 ∧
  j.giftsAmount = 99 ∧
  j.giftsAmount = (1 - (j.vacationFundPercent + j.savingsPercent + j.socializingPercent)) * j.discretionaryIncome

/-- The theorem stating the ratio of discretionary income to net salary --/
theorem discretionary_income_ratio (j : JillFinances) 
  (h : jillFinancesConditions j) : 
  j.discretionaryIncome / j.netSalary = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_discretionary_income_ratio_l1330_133014


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l1330_133045

/-- Calculates the upstream speed of a man given his downstream speed and the stream speed. -/
def upstreamSpeed (downstreamSpeed streamSpeed : ℝ) : ℝ :=
  downstreamSpeed - 2 * streamSpeed

/-- Theorem stating that given a downstream speed of 10 kmph and a stream speed of 1 kmph, 
    the upstream speed is 8 kmph. -/
theorem upstream_speed_calculation :
  upstreamSpeed 10 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_upstream_speed_calculation_l1330_133045


namespace NUMINAMATH_CALUDE_states_fraction_l1330_133035

/-- Given 30 total states and 15 states joining during a specific decade,
    prove that the fraction of states joining in that decade is 1/2. -/
theorem states_fraction (total_states : ℕ) (decade_states : ℕ) 
    (h1 : total_states = 30) 
    (h2 : decade_states = 15) : 
    (decade_states : ℚ) / total_states = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_states_fraction_l1330_133035


namespace NUMINAMATH_CALUDE_yard_length_theorem_l1330_133009

/-- The length of a yard with equally spaced trees -/
def yard_length (num_trees : ℕ) (tree_distance : ℝ) : ℝ :=
  (num_trees - 1) * tree_distance

/-- Theorem: The length of a yard with 14 trees planted at equal distances, 
    with one tree at each end, and a distance of 21 meters between consecutive trees, 
    is equal to 273 meters. -/
theorem yard_length_theorem : 
  yard_length 14 21 = 273 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_theorem_l1330_133009


namespace NUMINAMATH_CALUDE_quadratic_rational_root_parity_l1330_133040

theorem quadratic_rational_root_parity (a b c : ℤ) (x : ℚ) : 
  a ≠ 0 → 
  a * x^2 + b * x + c = 0 → 
  ¬(Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_parity_l1330_133040


namespace NUMINAMATH_CALUDE_square_equals_cube_root_16_l1330_133075

theorem square_equals_cube_root_16 : ∃! x : ℝ, x > 0 ∧ x^2 = (Real.sqrt 16)^3 := by
  sorry

end NUMINAMATH_CALUDE_square_equals_cube_root_16_l1330_133075


namespace NUMINAMATH_CALUDE_stating_six_suitcases_attempts_stating_ten_suitcases_attempts_l1330_133050

/-- 
Given n suitcases and n keys, where it is unknown which key opens which suitcase,
this function calculates the minimum number of attempts needed to ensure all suitcases are opened.
-/
def minAttempts (n : ℕ) : ℕ := (n - 1) * n / 2

/-- 
Theorem stating that for 6 suitcases and 6 keys, the minimum number of attempts is 15.
-/
theorem six_suitcases_attempts : minAttempts 6 = 15 := by sorry

/-- 
Theorem stating that for 10 suitcases and 10 keys, the minimum number of attempts is 45.
-/
theorem ten_suitcases_attempts : minAttempts 10 = 45 := by sorry

end NUMINAMATH_CALUDE_stating_six_suitcases_attempts_stating_ten_suitcases_attempts_l1330_133050


namespace NUMINAMATH_CALUDE_student_selection_methods_l1330_133013

/-- Represents the number of ways to select students by gender from a group -/
def select_students (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to select 4 students by gender from a group of 8 students (6 boys and 2 girls) is 40 -/
theorem student_selection_methods :
  select_students 8 6 2 4 = 40 :=
sorry

end NUMINAMATH_CALUDE_student_selection_methods_l1330_133013


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l1330_133016

theorem max_sum_on_circle : 
  ∀ x y : ℤ, 
  x^2 + y^2 = 169 → 
  x ≥ y → 
  x + y ≤ 21 := by
sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l1330_133016


namespace NUMINAMATH_CALUDE_adventure_team_probabilities_l1330_133032

def team_size : ℕ := 8
def medical_staff : ℕ := 3
def group_size : ℕ := 4

def probability_one_medical_in_one_group : ℚ := 6/7
def probability_at_least_two_medical_in_group : ℚ := 1/2
def expected_medical_in_group : ℚ := 3/2

theorem adventure_team_probabilities :
  (team_size = 8) →
  (medical_staff = 3) →
  (group_size = 4) →
  (probability_one_medical_in_one_group = 6/7) ∧
  (probability_at_least_two_medical_in_group = 1/2) ∧
  (expected_medical_in_group = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_adventure_team_probabilities_l1330_133032


namespace NUMINAMATH_CALUDE_prism_tetrahedron_surface_area_ratio_l1330_133079

/-- The ratio of surface areas of a rectangular prism to a tetrahedron --/
theorem prism_tetrahedron_surface_area_ratio :
  let prism_dimensions : Fin 3 → ℝ := ![2, 3, 4]
  let prism_surface_area := 2 * (prism_dimensions 0 * prism_dimensions 1 + 
                                 prism_dimensions 1 * prism_dimensions 2 + 
                                 prism_dimensions 0 * prism_dimensions 2)
  let tetrahedron_edge_length := Real.sqrt 13
  let tetrahedron_surface_area := Real.sqrt 3 * tetrahedron_edge_length ^ 2
  prism_surface_area / tetrahedron_surface_area = 4 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prism_tetrahedron_surface_area_ratio_l1330_133079


namespace NUMINAMATH_CALUDE_c_rent_share_is_72_l1330_133010

/-- Represents a person renting the pasture -/
structure Renter where
  oxen : ℕ
  months : ℕ

/-- Calculates the share of a renter in ox-months -/
def share (r : Renter) : ℕ := r.oxen * r.months

/-- Represents the pasture rental scenario -/
structure PastureRental where
  a : Renter
  b : Renter
  c : Renter
  totalRent : ℕ

/-- Calculates the total share of all renters -/
def totalShare (pr : PastureRental) : ℕ :=
  share pr.a + share pr.b + share pr.c

/-- Calculates the rent share for a specific renter -/
def rentShare (pr : PastureRental) (r : Renter) : ℚ :=
  (share r : ℚ) / (totalShare pr : ℚ) * pr.totalRent

theorem c_rent_share_is_72 (pr : PastureRental) : 
  pr.a = { oxen := 10, months := 7 } →
  pr.b = { oxen := 12, months := 5 } →
  pr.c = { oxen := 15, months := 3 } →
  pr.totalRent = 280 →
  rentShare pr pr.c = 72 := by
  sorry

#check c_rent_share_is_72

end NUMINAMATH_CALUDE_c_rent_share_is_72_l1330_133010


namespace NUMINAMATH_CALUDE_ice_cream_ratio_l1330_133088

theorem ice_cream_ratio (sunday : ℕ) (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) 
  (h1 : sunday = 4)
  (h2 : monday = 3 * sunday)
  (h3 : tuesday = monday / 3)
  (h4 : wednesday = 18)
  (h5 : sunday + monday + tuesday = wednesday + (sunday + monday + tuesday - wednesday)) :
  (sunday + monday + tuesday - wednesday) / tuesday = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_ratio_l1330_133088


namespace NUMINAMATH_CALUDE_abc_sqrt_problem_l1330_133080

theorem abc_sqrt_problem (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_abc_sqrt_problem_l1330_133080


namespace NUMINAMATH_CALUDE_train_passenger_problem_l1330_133098

theorem train_passenger_problem (P : ℚ) : 
  (((P * (2/3) + 280) * (1/2) + 12) = 242) → P = 270 := by
  sorry

end NUMINAMATH_CALUDE_train_passenger_problem_l1330_133098


namespace NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l1330_133093

/-- Represents the family structure of Marilyn and her descendants -/
structure FamilyStructure where
  daughters : ℕ
  granddaughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of daughters each daughter with children has -/
def daughters_per_mother : ℕ := 5

/-- Axioms representing the given conditions -/
axiom marilyn : FamilyStructure
axiom marilyn_daughters : marilyn.daughters = 10
axiom marilyn_total : marilyn.total_descendants = 40
axiom marilyn_granddaughters : marilyn.granddaughters = marilyn.total_descendants - marilyn.daughters
axiom marilyn_daughters_with_children : 
  marilyn.daughters_with_children * daughters_per_mother = marilyn.granddaughters

/-- The main theorem to prove -/
theorem daughters_and_granddaughters_without_children : 
  marilyn.granddaughters + (marilyn.daughters - marilyn.daughters_with_children) = 34 := by
  sorry

end NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l1330_133093


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1330_133092

theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 3) (hb : b = 6) (hc : c = 12) :
  let r := (1 / a + 1 / b + 1 / c + 2 * Real.sqrt (1 / (a * b) + 1 / (a * c) + 1 / (b * c)))⁻¹
  r = 12 / (7 + 2 * Real.sqrt 14) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1330_133092


namespace NUMINAMATH_CALUDE_annes_age_l1330_133099

theorem annes_age (maude emile anne : ℕ) 
  (h1 : anne = 2 * emile)
  (h2 : emile = 6 * maude)
  (h3 : maude = 8) :
  anne = 96 := by
  sorry

end NUMINAMATH_CALUDE_annes_age_l1330_133099


namespace NUMINAMATH_CALUDE_annas_car_rental_cost_l1330_133047

/-- Calculates the total cost of a car rental given the daily rate, per-mile rate, 
    number of days, and miles driven. -/
def carRentalCost (dailyRate : ℚ) (perMileRate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  dailyRate * days + perMileRate * miles

/-- Proves that Anna's car rental cost is $275 given the specified conditions. -/
theorem annas_car_rental_cost :
  carRentalCost 30 0.25 5 500 = 275 := by
  sorry

end NUMINAMATH_CALUDE_annas_car_rental_cost_l1330_133047


namespace NUMINAMATH_CALUDE_zoo_visitors_l1330_133069

theorem zoo_visitors (adult_price kid_price total_sales num_kids : ℕ) 
  (h1 : adult_price = 28)
  (h2 : kid_price = 12)
  (h3 : total_sales = 3864)
  (h4 : num_kids = 203) :
  ∃ (num_adults : ℕ), 
    adult_price * num_adults + kid_price * num_kids = total_sales ∧
    num_adults + num_kids = 254 := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_l1330_133069


namespace NUMINAMATH_CALUDE_set_operations_and_range_l1330_133058

-- Define the sets A, B, and C
def A : Set ℝ := {x | 2 < x ∧ x < 9}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}

-- Theorem statement
theorem set_operations_and_range :
  (A ∩ B = {x | 2 < x ∧ x ≤ 5}) ∧
  (B ∪ (Set.univ \ A) = {x | x ≤ 5 ∨ x ≥ 9}) ∧
  (∀ a : ℝ, C a ⊆ (Set.univ \ B) → (a < -4 ∨ a > 5)) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_and_range_l1330_133058


namespace NUMINAMATH_CALUDE_custom_equation_solution_l1330_133059

-- Define the custom operation *
def star (a b : ℝ) : ℝ := 2 * a - b

-- State the theorem
theorem custom_equation_solution :
  ∃! x : ℝ, star 2 (star 6 x) = 2 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_custom_equation_solution_l1330_133059


namespace NUMINAMATH_CALUDE_basketball_game_points_l1330_133048

/-- The total points scored by three players in a basketball game. -/
def total_points (jon_points jack_points tom_points : ℕ) : ℕ :=
  jon_points + jack_points + tom_points

/-- Theorem stating the total points scored by Jon, Jack, and Tom. -/
theorem basketball_game_points : ∃ (jack_points tom_points : ℕ),
  let jon_points := 3
  jack_points = jon_points + 5 ∧
  tom_points = (jon_points + jack_points) - 4 ∧
  total_points jon_points jack_points tom_points = 18 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_points_l1330_133048


namespace NUMINAMATH_CALUDE_complex_number_second_quadrant_l1330_133046

theorem complex_number_second_quadrant (a : ℝ) : 
  let z : ℂ := (a + 3*Complex.I)/Complex.I + a*Complex.I
  (z.re = 0) ∧ (z.im < 0) ∧ (z.re < 0) → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_second_quadrant_l1330_133046


namespace NUMINAMATH_CALUDE_constant_product_l1330_133065

-- Define the circle and points
variable (Circle : Type) (A B C D : Point)
variable (diameter : Circle → Point → Point → Prop)
variable (tangent : Circle → Point → Prop)
variable (on_circle : Circle → Point → Prop)
variable (on_tangent : Circle → Point → Prop)
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem constant_product 
  (circle : Circle)
  (h1 : diameter circle A B)
  (h2 : tangent circle B)
  (h3 : on_circle circle C)
  (h4 : on_tangent circle D)
  : distance A C * distance A D = distance A B * distance A B :=
sorry

end NUMINAMATH_CALUDE_constant_product_l1330_133065


namespace NUMINAMATH_CALUDE_solution_correctness_l1330_133057

theorem solution_correctness : ∀ x : ℝ,
  (((x^2 - 1)^2 - 5*(x^2 - 1) + 4 = 0) ↔ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2 ∨ x = Real.sqrt 5 ∨ x = -Real.sqrt 5)) ∧
  ((x^4 - x^2 - 6 = 0) ↔ (x = Real.sqrt 3 ∨ x = -Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_solution_correctness_l1330_133057


namespace NUMINAMATH_CALUDE_inequality_proof_l1330_133090

theorem inequality_proof (b c : ℝ) (hb : b > 0) (hc : c > 0) :
  (b - c)^2011 * (b + c)^2011 * (c - b)^2011 ≥ (b^2011 - c^2011) * (b^2011 + c^2011) * (c^2011 - b^2011) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1330_133090


namespace NUMINAMATH_CALUDE_triangle_properties_l1330_133074

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the given condition
def given_condition (t : Triangle) : Prop :=
  2 * t.a * Real.cos t.A = t.b * Real.cos t.C + t.c * Real.cos t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : given_condition t) : 
  t.A = π / 3 ∧ 
  ∀ x, x ∈ Set.Icc (-1 : ℝ) (-1/2) ↔ 
    ∃ (B C : ℝ), t.B = B ∧ t.C = C ∧ x = Real.cos B - Real.sqrt 3 * Real.sin C :=
sorry


end NUMINAMATH_CALUDE_triangle_properties_l1330_133074


namespace NUMINAMATH_CALUDE_target_digit_is_seven_l1330_133034

/-- The decimal representation of 13/481 -/
def decimal_rep : ℚ := 13 / 481

/-- The length of the repeating sequence in the decimal representation -/
def repeat_length : ℕ := 3

/-- The position of the digit we're looking for -/
def target_position : ℕ := 222

/-- The function that returns the nth digit after the decimal point -/
noncomputable def nth_digit (n : ℕ) : ℕ := 
  sorry

theorem target_digit_is_seven : nth_digit target_position = 7 := by
  sorry

end NUMINAMATH_CALUDE_target_digit_is_seven_l1330_133034


namespace NUMINAMATH_CALUDE_amoeba_bacteria_ratio_l1330_133000

theorem amoeba_bacteria_ratio (a₁ b₁ : ℕ) (h : a₁ > 0 ∧ b₁ > 0) :
  (∀ n : ℕ, n > 0 → 2^(n-1) * (b₁ - a₁) = 0) → a₁ = b₁ :=
sorry

end NUMINAMATH_CALUDE_amoeba_bacteria_ratio_l1330_133000


namespace NUMINAMATH_CALUDE_sams_money_l1330_133044

theorem sams_money (s b : ℕ) : b = 2 * s - 25 → s + b = 200 → s = 75 := by
  sorry

end NUMINAMATH_CALUDE_sams_money_l1330_133044


namespace NUMINAMATH_CALUDE_inequality_proof_l1330_133082

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a^2 - a*b + b^2) + Real.sqrt (b^2 - b*c + c^2) ≥ Real.sqrt (a^2 + a*c + c^2) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1330_133082


namespace NUMINAMATH_CALUDE_coeff_x2y2_in_expansion_l1330_133063

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^a * y^b in (1+x)^m * (1+y)^n
def coeff (m n a b : ℕ) : ℕ := binomial m a * binomial n b

-- Theorem statement
theorem coeff_x2y2_in_expansion : coeff 3 4 2 2 = 18 := by sorry

end NUMINAMATH_CALUDE_coeff_x2y2_in_expansion_l1330_133063


namespace NUMINAMATH_CALUDE_distribution_law_l1330_133001

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  h_x_lt : x₁ < x₂
  h_p_bound : 0 ≤ p₁ ∧ p₁ ≤ 1

/-- Expectation of a DiscreteRV -/
def expectation (X : DiscreteRV) : ℝ := X.x₁ * X.p₁ + X.x₂ * (1 - X.p₁)

/-- Variance of a DiscreteRV -/
def variance (X : DiscreteRV) : ℝ :=
  X.p₁ * (X.x₁ - expectation X)^2 + (1 - X.p₁) * (X.x₂ - expectation X)^2

/-- Theorem stating the distribution law of the given discrete random variable -/
theorem distribution_law (X : DiscreteRV)
  (h_p₁ : X.p₁ = 0.5)
  (h_expectation : expectation X = 3.5)
  (h_variance : variance X = 0.25) :
  X.x₁ = 3 ∧ X.x₂ = 4 :=
sorry

end NUMINAMATH_CALUDE_distribution_law_l1330_133001


namespace NUMINAMATH_CALUDE_roots_are_irrational_l1330_133037

theorem roots_are_irrational (k : ℝ) : 
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 3*k*x + 2*k^2 - 1 = 0 ∧ y^2 - 3*k*y + 2*k^2 - 1 = 0) →
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 3*k*x + 2*k^2 - 1 = 0 ∧ y^2 - 3*k*y + 2*k^2 - 1 = 0 ∧ 
   (¬∃ m n : ℤ, x = m / n ∨ y = m / n)) :=
by sorry

end NUMINAMATH_CALUDE_roots_are_irrational_l1330_133037


namespace NUMINAMATH_CALUDE_calculation_proof_l1330_133086

theorem calculation_proof :
  (2 * Real.sqrt 18 - 3 * Real.sqrt 2 - Real.sqrt (1/2) = (5 * Real.sqrt 2) / 2) ∧
  ((Real.sqrt 3 - 1)^2 - (Real.sqrt 2 + Real.sqrt 3) * (Real.sqrt 3 - Real.sqrt 2) = 3 - 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l1330_133086


namespace NUMINAMATH_CALUDE_percentage_calculation_l1330_133042

theorem percentage_calculation (x : ℝ) (h : 0.3 * 0.4 * x = 36) : 0.5 * 0.2 * x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1330_133042


namespace NUMINAMATH_CALUDE_cake_volume_and_icing_sum_l1330_133031

/-- Represents a point in 3D space -/
structure Point3D where
  x : Real
  y : Real
  z : Real

/-- Represents a triangular piece of cake -/
structure CakePiece where
  corner : Point3D
  midpoint1 : Point3D
  midpoint2 : Point3D

/-- Calculates the volume of the triangular cake piece -/
def volume (piece : CakePiece) : Real :=
  sorry

/-- Calculates the area of icing on the triangular cake piece -/
def icingArea (piece : CakePiece) : Real :=
  sorry

/-- The main theorem to prove -/
theorem cake_volume_and_icing_sum (cubeEdgeLength : Real) (piece : CakePiece) : 
  cubeEdgeLength = 3 →
  piece.corner = ⟨0, 0, 0⟩ →
  piece.midpoint1 = ⟨3, 3, 1.5⟩ →
  piece.midpoint2 = ⟨1.5, 3, 3⟩ →
  volume piece + icingArea piece = 24 :=
sorry

end NUMINAMATH_CALUDE_cake_volume_and_icing_sum_l1330_133031


namespace NUMINAMATH_CALUDE_set_equality_l1330_133022

theorem set_equality (M : Set ℕ) : M ∪ {1} = {1, 2, 3} → M = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l1330_133022


namespace NUMINAMATH_CALUDE_davids_physics_marks_l1330_133003

/-- Given David's marks in various subjects and his average, prove his marks in Physics --/
theorem davids_physics_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (total_subjects : ℕ)
  (h1 : english_marks = 86)
  (h2 : math_marks = 85)
  (h3 : chemistry_marks = 87)
  (h4 : biology_marks = 85)
  (h5 : average_marks = 85)
  (h6 : total_subjects = 5)
  : ∃ (physics_marks : ℕ),
    physics_marks = average_marks * total_subjects - (english_marks + math_marks + chemistry_marks + biology_marks) ∧
    physics_marks = 82 :=
by sorry

end NUMINAMATH_CALUDE_davids_physics_marks_l1330_133003


namespace NUMINAMATH_CALUDE_unique_solution_mod_125_l1330_133028

theorem unique_solution_mod_125 :
  ∃! x : ℕ, x < 125 ∧ (x^3 - 2*x + 6) % 125 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_mod_125_l1330_133028


namespace NUMINAMATH_CALUDE_carly_to_lisa_tshirt_ratio_l1330_133024

def lisa_tshirts : ℚ := 40
def lisa_jeans : ℚ := lisa_tshirts / 2
def lisa_coats : ℚ := lisa_tshirts * 2

def carly_jeans : ℚ := lisa_jeans * 3
def carly_coats : ℚ := lisa_coats / 4

def total_spending : ℚ := 230

theorem carly_to_lisa_tshirt_ratio :
  ∃ (carly_tshirts : ℚ),
    lisa_tshirts + lisa_jeans + lisa_coats + carly_tshirts + carly_jeans + carly_coats = total_spending ∧
    carly_tshirts / lisa_tshirts = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_carly_to_lisa_tshirt_ratio_l1330_133024


namespace NUMINAMATH_CALUDE_min_four_digit_number_l1330_133061

/-- Represents a four-digit number ABCD -/
structure FourDigitNumber where
  value : ℕ
  is_four_digit : 1000 ≤ value ∧ value ≤ 9999

/-- Returns the first two digits (AB) of a four-digit number -/
def first_two_digits (n : FourDigitNumber) : ℕ :=
  n.value / 100

/-- Returns the last two digits (CD) of a four-digit number -/
def last_two_digits (n : FourDigitNumber) : ℕ :=
  n.value % 100

/-- The property that ABCD + AB × CD is a multiple of 1111 -/
def satisfies_condition (n : FourDigitNumber) : Prop :=
  ∃ k : ℕ, n.value + (first_two_digits n) * (last_two_digits n) = 1111 * k

theorem min_four_digit_number :
  ∀ n : FourDigitNumber, satisfies_condition n → n.value ≥ 1729 :=
by sorry

end NUMINAMATH_CALUDE_min_four_digit_number_l1330_133061


namespace NUMINAMATH_CALUDE_divisibility_condition_l1330_133019

theorem divisibility_condition (a : ℤ) : 
  5 ∣ (a^3 + 3*a + 1) ↔ a % 5 = 1 ∨ a % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l1330_133019


namespace NUMINAMATH_CALUDE_flag_combinations_l1330_133067

def num_colors : ℕ := 2
def num_stripes : ℕ := 3

theorem flag_combinations : (num_colors ^ num_stripes : ℕ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_flag_combinations_l1330_133067


namespace NUMINAMATH_CALUDE_simplify_expression_l1330_133020

theorem simplify_expression : 
  (625 : ℝ) ^ (1/4 : ℝ) * (343 : ℝ) ^ (1/3 : ℝ) = 35 := by
  sorry

-- Additional definitions to match the problem conditions
def condition1 : (625 : ℝ) = 5^4 := by sorry
def condition2 : (343 : ℝ) = 7^3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1330_133020


namespace NUMINAMATH_CALUDE_geometric_sequence_b_value_l1330_133097

theorem geometric_sequence_b_value (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 180) (h₂ : a₃ = 64/25) (h₃ : a₂ > 0) 
  (h₄ : ∃ r : ℝ, a₂ = a₁ * r ∧ a₃ = a₂ * r) : a₂ = 21.6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_b_value_l1330_133097


namespace NUMINAMATH_CALUDE_circles_tangent_to_ellipse_l1330_133006

theorem circles_tangent_to_ellipse (r : ℝ) : 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x-r)^2 + y^2 = r^2) ∧ 
  (∃ (x y : ℝ), x^2 + 4*y^2 = 5 ∧ (x+r)^2 + y^2 = r^2) →
  r = Real.sqrt 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_circles_tangent_to_ellipse_l1330_133006


namespace NUMINAMATH_CALUDE_a_minus_b_value_l1330_133054

theorem a_minus_b_value (a b : ℝ) (h1 : |a| = 5) (h2 : b^2 = 64) (h3 : a * b > 0) :
  a - b = 3 ∨ a - b = -3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_value_l1330_133054


namespace NUMINAMATH_CALUDE_gerbil_weight_difference_gerbil_weight_difference_proof_l1330_133041

/-- The weight difference between Scruffy and Muffy given the conditions of the gerbil problem -/
theorem gerbil_weight_difference : ℝ → Prop :=
  fun weight_difference =>
    ∃ (muffy_weight : ℝ),
      let puffy_weight := muffy_weight + 5
      let scruffy_weight := 12
      puffy_weight + muffy_weight = 23 ∧
      weight_difference = scruffy_weight - muffy_weight ∧
      weight_difference = 3

/-- Proof of the gerbil weight difference theorem -/
theorem gerbil_weight_difference_proof : gerbil_weight_difference 3 := by
  sorry

end NUMINAMATH_CALUDE_gerbil_weight_difference_gerbil_weight_difference_proof_l1330_133041


namespace NUMINAMATH_CALUDE_money_exchange_solution_l1330_133094

/-- Represents the money exchange scenario between A, B, and C -/
def MoneyExchange (a b c : ℕ) : Prop :=
  let a₁ := a - 3*b - 3*c
  let b₁ := 4*b
  let c₁ := 4*c
  let a₂ := 4*a₁
  let b₂ := b₁ - 3*a₁ - 3*c₁
  let c₂ := 4*c₁
  let a₃ := 4*a₂
  let b₃ := 4*b₂
  let c₃ := c₂ - 3*a₂ - 3*b₂
  a₃ = 27 ∧ b₃ = 27 ∧ c₃ = 27 ∧ a + b + c = 81

theorem money_exchange_solution :
  ∃ (b c : ℕ), MoneyExchange 52 b c :=
sorry

end NUMINAMATH_CALUDE_money_exchange_solution_l1330_133094


namespace NUMINAMATH_CALUDE_cube_cutting_l1330_133018

theorem cube_cutting (n : ℕ) : 
  (∃ s : ℕ, n > s ∧ n^3 - s^3 = 152) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_l1330_133018


namespace NUMINAMATH_CALUDE_quadrilateral_offset_l1330_133087

/-- Given a quadrilateral with one diagonal of length d, two offsets h1 and h2,
    and area A, this theorem states that if d = 30, h2 = 6, and A = 240,
    then h1 = 10. -/
theorem quadrilateral_offset (d h1 h2 A : ℝ) :
  d = 30 → h2 = 6 → A = 240 → A = (1/2) * d * (h1 + h2) → h1 = 10 := by
  sorry

#check quadrilateral_offset

end NUMINAMATH_CALUDE_quadrilateral_offset_l1330_133087


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1330_133078

/-- Geometric sequence with common ratio 2, 5 terms, and sum 62 has first term equal to 2 -/
theorem geometric_sequence_first_term (a : ℕ → ℝ) (q n : ℕ) (S : ℝ) : 
  (∀ k, a (k + 1) = 2 * a k) →  -- geometric sequence with common ratio 2
  q = 2 →
  n = 5 →
  S = (a 1) * (1 - 2^5) / (1 - 2) →
  S = 62 →
  a 1 = 2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1330_133078


namespace NUMINAMATH_CALUDE_soft_drink_cost_l1330_133038

/-- Proves that the cost of each soft drink is $4 given the conditions of Benny's purchase. -/
theorem soft_drink_cost (num_soft_drinks : ℕ) (num_candy_bars : ℕ) (total_spent : ℚ) (candy_bar_cost : ℚ) :
  num_soft_drinks = 2 →
  num_candy_bars = 5 →
  total_spent = 28 →
  candy_bar_cost = 4 →
  ∃ (soft_drink_cost : ℚ), 
    soft_drink_cost * num_soft_drinks + candy_bar_cost * num_candy_bars = total_spent ∧
    soft_drink_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_soft_drink_cost_l1330_133038


namespace NUMINAMATH_CALUDE_cubic_root_sum_l1330_133068

theorem cubic_root_sum (α β γ : ℂ) : 
  (α^3 - α - 1 = 0) → 
  (β^3 - β - 1 = 0) → 
  (γ^3 - γ - 1 = 0) → 
  ((1 + α) / (1 - α) + (1 + β) / (1 - β) + (1 + γ) / (1 - γ) = -7) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l1330_133068


namespace NUMINAMATH_CALUDE_min_value_theorem_l1330_133030

theorem min_value_theorem (x : ℝ) (h1 : x > 0) (h2 : Real.log x + 1 ≤ x) :
  (x^2 - Real.log x + x) / x ≥ 2 ∧
  ((x^2 - Real.log x + x) / x = 2 ↔ x = 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1330_133030


namespace NUMINAMATH_CALUDE_min_value_expression_l1330_133007

theorem min_value_expression (x y z k : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hk : k > 0) : 
  (6 * z) / (x + 2 * y + k) + (6 * x) / (2 * z + y + k) + (3 * y) / (x + z + k) ≥ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1330_133007


namespace NUMINAMATH_CALUDE_billy_coins_l1330_133089

theorem billy_coins (quarter_piles dime_piles coins_per_pile : ℕ) 
  (h1 : quarter_piles = 2)
  (h2 : dime_piles = 3)
  (h3 : coins_per_pile = 4) :
  quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 20 :=
by sorry

end NUMINAMATH_CALUDE_billy_coins_l1330_133089


namespace NUMINAMATH_CALUDE_third_derivative_y_l1330_133084

noncomputable def y (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_y (x : ℝ) :
  (deriv^[3] y) x = 4 / (1 + x^2)^2 := by sorry

end NUMINAMATH_CALUDE_third_derivative_y_l1330_133084


namespace NUMINAMATH_CALUDE_f_range_l1330_133060

noncomputable def f (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem f_range : ∀ x : ℝ, f x = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_f_range_l1330_133060


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1330_133091

theorem inverse_proportion_problem (k : ℝ) (a b : ℝ → ℝ) :
  (∀ x, a x * (b x)^2 = k) →  -- Inverse proportion relationship
  (∃ x, a x = 40) →           -- a = 40 for some value of b
  (a (b 10) = 10) →           -- When a = 10
  b 10 = 2                    -- b = 2
:= by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1330_133091


namespace NUMINAMATH_CALUDE_jed_speeding_fine_jed_speed_l1330_133015

theorem jed_speeding_fine (fine_per_mph : ℕ) (total_fine : ℕ) (speed_limit : ℕ) : ℕ :=
  let speed_over_limit := total_fine / fine_per_mph
  let total_speed := speed_limit + speed_over_limit
  total_speed

theorem jed_speed : jed_speeding_fine 16 256 50 = 66 := by
  sorry

end NUMINAMATH_CALUDE_jed_speeding_fine_jed_speed_l1330_133015


namespace NUMINAMATH_CALUDE_triangle_side_length_l1330_133081

theorem triangle_side_length (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) →
  (0 < a ∧ 0 < b ∧ 0 < c) →
  -- Given conditions
  (Real.cos A = Real.sqrt 5 / 5) →
  (Real.cos B = Real.sqrt 10 / 10) →
  (c = Real.sqrt 2) →
  -- Sine rule
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (c / Real.sin C = a / Real.sin A) →
  -- Prove
  a = 4 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1330_133081


namespace NUMINAMATH_CALUDE_inequality_range_l1330_133036

theorem inequality_range (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ t : ℝ, (∀ x y : ℝ, a * x^2 + t * y^2 ≥ (a * x + t * y)^2) ↔ (0 ≤ t ∧ t ≤ 1 - a) :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_l1330_133036


namespace NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l1330_133021

theorem angle_sum_is_pi_over_two 
  (α β γ : Real) 
  (h_sin_α : Real.sin α = 1/3)
  (h_sin_β : Real.sin β = 1/(3*Real.sqrt 11))
  (h_sin_γ : Real.sin γ = 3/Real.sqrt 11)
  (h_acute_α : 0 < α ∧ α < π/2)
  (h_acute_β : 0 < β ∧ β < π/2)
  (h_acute_γ : 0 < γ ∧ γ < π/2) :
  α + β + γ = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_is_pi_over_two_l1330_133021


namespace NUMINAMATH_CALUDE_centers_regular_iff_original_affinely_regular_l1330_133064

open Complex

/-- Definition of an n-gon as a list of complex numbers -/
def NGon (n : ℕ) := List ℂ

/-- A convex n-gon -/
def ConvexNGon (n : ℕ) (A : NGon n) : Prop := sorry

/-- Centers of regular n-gons constructed on sides of an n-gon -/
def CentersOfExternalNGons (n : ℕ) (A : NGon n) : NGon n := sorry

/-- Check if an n-gon is regular -/
def IsRegularNGon (n : ℕ) (B : NGon n) : Prop := sorry

/-- Check if an n-gon is affinely regular -/
def IsAffinelyRegularNGon (n : ℕ) (A : NGon n) : Prop := sorry

/-- Main theorem: The centers form a regular n-gon iff the original n-gon is affinely regular -/
theorem centers_regular_iff_original_affinely_regular 
  (n : ℕ) (A : NGon n) (h : ConvexNGon n A) :
  IsRegularNGon n (CentersOfExternalNGons n A) ↔ IsAffinelyRegularNGon n A :=
sorry

end NUMINAMATH_CALUDE_centers_regular_iff_original_affinely_regular_l1330_133064


namespace NUMINAMATH_CALUDE_exists_larger_area_same_perimeter_l1330_133017

-- Define a convex figure
structure ConvexFigure where
  perimeter : ℝ
  area : ℝ

-- Define a property for a figure to be a circle
def isCircle (f : ConvexFigure) : Prop := sorry

-- Theorem statement
theorem exists_larger_area_same_perimeter 
  (Φ : ConvexFigure) 
  (h_not_circle : ¬ isCircle Φ) : 
  ∃ (Ψ : ConvexFigure), 
    Ψ.perimeter = Φ.perimeter ∧ 
    Ψ.area > Φ.area := by
  sorry

end NUMINAMATH_CALUDE_exists_larger_area_same_perimeter_l1330_133017


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1330_133039

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- State the theorem
theorem quadratic_function_properties
  (a b : ℝ) (h_a : a ≠ 0)
  (h_min : ∀ x, f a b x ≥ f a b 1)
  (h_zero : f a b 1 = 0) :
  -- 1. f(x) = x² - 2x + 1
  (∀ x, f a b x = x^2 - 2*x + 1) ∧
  -- 2. f(x) is decreasing on (-∞, 1] and increasing on [1, +∞)
  (∀ x y, x ≤ 1 → y ≤ 1 → x ≤ y → f a b x ≥ f a b y) ∧
  (∀ x y, 1 ≤ x → 1 ≤ y → x ≤ y → f a b x ≤ f a b y) ∧
  -- 3. If f(x) > x + k for all x ∈ [1, 3], then k < -5/4
  (∀ k, (∀ x, 1 ≤ x → x ≤ 3 → f a b x > x + k) → k < -5/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1330_133039


namespace NUMINAMATH_CALUDE_power_division_rule_l1330_133012

theorem power_division_rule (a : ℝ) (h : a ≠ 0) : a^4 / a = a^3 := by
  sorry

end NUMINAMATH_CALUDE_power_division_rule_l1330_133012


namespace NUMINAMATH_CALUDE_quadratic_symmetry_inequality_l1330_133070

/-- Given real numbers a, b, c, and a quadratic function f(x) = ax^2 + bx + c
    that is symmetric about x = 1, prove that f(1-a) < f(1-2a) < f(1) is impossible. -/
theorem quadratic_symmetry_inequality (a b c : ℝ) 
    (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 + b * x + c) 
    (h_sym : ∀ x, f x = f (2 - x)) : 
  ¬(f (1 - a) < f (1 - 2*a) ∧ f (1 - 2*a) < f 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_inequality_l1330_133070


namespace NUMINAMATH_CALUDE_ball_bounce_height_l1330_133026

theorem ball_bounce_height (h₀ : ℝ) (r : ℝ) (h₁ : h₀ = 1000) (h₂ : r = 1/2) :
  ∃ k : ℕ, k > 0 ∧ h₀ * r^k < 1 ∧ ∀ j : ℕ, 0 < j → j < k → h₀ * r^j ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_ball_bounce_height_l1330_133026


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1330_133002

theorem quadratic_coefficient (α : ℝ) (p q : ℝ) : 
  (∀ x, x^2 - (α - 2)*x - α - 1 = 0 ↔ x = p ∨ x = q) →
  (∀ a b, a^2 + b^2 ≥ 5 ∧ (a = p ∧ b = q ∨ a = q ∧ b = p) → p^2 + q^2 ≥ 5) →
  p^2 + q^2 = 5 →
  α - 2 = -1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1330_133002


namespace NUMINAMATH_CALUDE_missing_number_proof_l1330_133056

theorem missing_number_proof (x : ℝ) : 
  11 + Real.sqrt (-4 + 6 * 4 / x) = 13 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l1330_133056


namespace NUMINAMATH_CALUDE_michaels_estimate_greater_l1330_133029

theorem michaels_estimate_greater (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x > y) : 
  3 * ((x + z) - (y - 2 * z)) > 3 * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_michaels_estimate_greater_l1330_133029


namespace NUMINAMATH_CALUDE_triangle_angle_B_l1330_133085

theorem triangle_angle_B (a b : ℝ) (A B : ℝ) : 
  a = 1 → b = Real.sqrt 2 → A = 30 * π / 180 → 
  (B = 45 * π / 180 ∨ B = 135 * π / 180) ↔ 
  (Real.sin B = b * Real.sin A / a) := by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l1330_133085


namespace NUMINAMATH_CALUDE_highest_power_of_two_in_50_factorial_l1330_133027

theorem highest_power_of_two_in_50_factorial (n : ℕ) : 
  (∀ k : ℕ, k ≤ n → (50 : ℕ).factorial % 2^k = 0) ∧ 
  (50 : ℕ).factorial % 2^(n + 1) ≠ 0 → 
  n = 47 := by
sorry

end NUMINAMATH_CALUDE_highest_power_of_two_in_50_factorial_l1330_133027


namespace NUMINAMATH_CALUDE_larger_number_l1330_133011

theorem larger_number (x y : ℝ) (h1 : x + y = 28) (h2 : x - y = 4) : max x y = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_l1330_133011


namespace NUMINAMATH_CALUDE_fence_cost_calculation_l1330_133052

/-- The cost of building a fence around a rectangular plot -/
def fence_cost (length width length_price width_price : ℕ) : ℕ :=
  2 * (length * length_price + width * width_price)

/-- Theorem stating the total cost of building the fence -/
theorem fence_cost_calculation :
  fence_cost 35 25 60 50 = 6700 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_calculation_l1330_133052


namespace NUMINAMATH_CALUDE_volume_of_enlarged_box_l1330_133008

/-- Represents a rectangular box with length l, width w, and height h -/
structure Box where
  l : ℝ
  w : ℝ
  h : ℝ

/-- Theorem: Volume of enlarged box -/
theorem volume_of_enlarged_box (box : Box) 
  (volume_eq : box.l * box.w * box.h = 5000)
  (surface_area_eq : 2 * (box.l * box.w + box.w * box.h + box.l * box.h) = 1800)
  (edge_sum_eq : 4 * (box.l + box.w + box.h) = 210) :
  (box.l + 2) * (box.w + 2) * (box.h + 2) = 7018 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_enlarged_box_l1330_133008


namespace NUMINAMATH_CALUDE_circles_are_tangent_l1330_133066

/-- Represents a circle in the 2D plane -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- Checks if two circles are tangent to each other -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let x1 := -c1.b / 2
  let y1 := -c1.c / 2
  let r1 := Real.sqrt (x1^2 + y1^2 - c1.e)
  let x2 := -c2.b / 2
  let y2 := -c2.c / 2
  let r2 := Real.sqrt (x2^2 + y2^2 - c2.e)
  let d := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)
  d = r1 + r2 ∨ d = abs (r1 - r2)

theorem circles_are_tangent : 
  let c1 : Circle := ⟨1, -6, 4, 1, 12⟩
  let c2 : Circle := ⟨1, -14, -2, 1, 14⟩
  are_tangent c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_circles_are_tangent_l1330_133066


namespace NUMINAMATH_CALUDE_current_rate_l1330_133043

/-- The rate of the current in a river, given boat speed and downstream travel information -/
theorem current_rate (boat_speed : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) :
  boat_speed = 15 →
  downstream_distance = 7.2 →
  downstream_time = 24 / 60 →
  ∃ (current_rate : ℝ), 
    downstream_distance = (boat_speed + current_rate) * downstream_time ∧
    current_rate = 3 :=
by sorry

end NUMINAMATH_CALUDE_current_rate_l1330_133043


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l1330_133077

/-- The number of arrangements of four people in a line. -/
def total_arrangements : Nat := 24

/-- The number of arrangements where A is at the far left or B is at the far right. -/
def excluded_arrangements : Nat := 12

/-- The number of arrangements where A is at the far left and B is at the far right simultaneously. -/
def double_counted_arrangements : Nat := 2

/-- The number of valid arrangements where A is not at the far left and B is not at the far right. -/
def valid_arrangements : Nat := total_arrangements - excluded_arrangements + double_counted_arrangements

theorem count_valid_arrangements :
  valid_arrangements = 14 :=
by sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l1330_133077


namespace NUMINAMATH_CALUDE_strawberry_ratio_l1330_133005

def strawberry_problem (betty_strawberries matthew_strawberries natalie_strawberries : ℕ)
  (strawberries_per_jar jar_price total_revenue : ℕ) : Prop :=
  betty_strawberries = 16 ∧
  matthew_strawberries = betty_strawberries + 20 ∧
  matthew_strawberries = natalie_strawberries ∧
  strawberries_per_jar = 7 ∧
  jar_price = 4 ∧
  total_revenue = 40 ∧
  (matthew_strawberries : ℚ) / natalie_strawberries = 1

theorem strawberry_ratio :
  ∀ (betty_strawberries matthew_strawberries natalie_strawberries : ℕ)
    (strawberries_per_jar jar_price total_revenue : ℕ),
  strawberry_problem betty_strawberries matthew_strawberries natalie_strawberries
    strawberries_per_jar jar_price total_revenue →
  (matthew_strawberries : ℚ) / natalie_strawberries = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_strawberry_ratio_l1330_133005


namespace NUMINAMATH_CALUDE_solve_apple_dealer_problem_l1330_133073

/-- Represents the apple dealer problem -/
def apple_dealer_problem (cost_per_bushel : ℚ) (apples_per_bushel : ℕ) (profit : ℚ) (apples_sold : ℕ) : Prop :=
  let cost_per_apple : ℚ := cost_per_bushel / apples_per_bushel
  let total_cost : ℚ := cost_per_apple * apples_sold
  let total_revenue : ℚ := total_cost + profit
  let price_per_apple : ℚ := total_revenue / apples_sold
  price_per_apple = 40 / 100

/-- Theorem stating the solution to the apple dealer problem -/
theorem solve_apple_dealer_problem :
  apple_dealer_problem 12 48 15 100 := by
  sorry

end NUMINAMATH_CALUDE_solve_apple_dealer_problem_l1330_133073


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequences_l1330_133023

theorem smallest_sum_of_sequences (A B C D : ℤ) : 
  A > 0 → B > 0 → C > 0 →  -- A, B, C are positive integers
  (C - B = B - A) →  -- A, B, C form an arithmetic sequence
  (C * C = B * D) →  -- B, C, D form a geometric sequence
  (C = (7 * B) / 4) →  -- C/B = 7/4
  (∀ A' B' C' D' : ℤ, 
    A' > 0 → B' > 0 → C' > 0 → 
    (C' - B' = B' - A') → 
    (C' * C' = B' * D') → 
    (C' = (7 * B') / 4) → 
    A + B + C + D ≤ A' + B' + C' + D') →
  A + B + C + D = 97 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequences_l1330_133023


namespace NUMINAMATH_CALUDE_abc_sum_zero_product_nonpositive_l1330_133071

theorem abc_sum_zero_product_nonpositive (a b c : ℝ) (h : a + b + c = 0) :
  (∀ x ≤ 0, ∃ a b c : ℝ, a + b + c = 0 ∧ a * b + a * c + b * c = x) ∧
  (∀ a b c : ℝ, a + b + c = 0 → a * b + a * c + b * c ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_zero_product_nonpositive_l1330_133071


namespace NUMINAMATH_CALUDE_prob_not_both_odd_is_five_sixths_l1330_133062

/-- The set of numbers to choose from -/
def S : Finset ℕ := {1, 2, 3, 4}

/-- The set of odd numbers in S -/
def odd_numbers : Finset ℕ := {1, 3}

/-- The probability of selecting two numbers without replacement from S such that not both are odd -/
def prob_not_both_odd : ℚ :=
  1 - (Finset.card odd_numbers).choose 2 / (Finset.card S).choose 2

theorem prob_not_both_odd_is_five_sixths : 
  prob_not_both_odd = 5/6 := by sorry

end NUMINAMATH_CALUDE_prob_not_both_odd_is_five_sixths_l1330_133062


namespace NUMINAMATH_CALUDE_union_equals_reals_l1330_133083

def S : Set ℝ := {x | (x - 2)^2 > 9}
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8}

theorem union_equals_reals (a : ℝ) : S ∪ T a = Set.univ ↔ a ∈ Set.Ioo (-3) (-1) := by
  sorry

end NUMINAMATH_CALUDE_union_equals_reals_l1330_133083


namespace NUMINAMATH_CALUDE_pie_sugar_percentage_l1330_133025

/-- Given a pie weighing 200 grams with 50 grams of sugar, 
    prove that 75% of the pie is not sugar. -/
theorem pie_sugar_percentage (total_weight : ℝ) (sugar_weight : ℝ) 
    (h1 : total_weight = 200) 
    (h2 : sugar_weight = 50) : 
    (total_weight - sugar_weight) / total_weight * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_pie_sugar_percentage_l1330_133025
