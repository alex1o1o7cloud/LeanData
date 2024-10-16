import Mathlib

namespace NUMINAMATH_CALUDE_distance_between_specific_planes_l896_89672

/-- Represents a plane in 3D space defined by ax + by + cz = d -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two planes -/
def distance_between_planes (p1 p2 : Plane) : ℝ :=
  sorry

/-- The first plane: 2x + 4y - 2z = 10 -/
def plane1 : Plane := ⟨2, 4, -2, 10⟩

/-- The second plane: x + 2y - z = -3 -/
def plane2 : Plane := ⟨1, 2, -1, -3⟩

theorem distance_between_specific_planes :
  distance_between_planes plane1 plane2 = Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_specific_planes_l896_89672


namespace NUMINAMATH_CALUDE_line_parameter_range_l896_89656

/-- Given two points on opposite sides of a line, prove the range of the line's parameter. -/
theorem line_parameter_range (m : ℝ) : 
  (∀ (x y : ℝ), 2*x + y + m = 0 → 
    ((x = 1 ∧ y = 3) ∨ (x = -4 ∧ y = -2)) →
    (2*1 + 3 + m) * (2*(-4) + (-2) + m) < 0) →
  -5 < m ∧ m < 10 :=
sorry

end NUMINAMATH_CALUDE_line_parameter_range_l896_89656


namespace NUMINAMATH_CALUDE_remainder_sum_l896_89681

theorem remainder_sum (a b : ℤ) (h1 : a % 84 = 78) (h2 : b % 120 = 114) :
  (a + b) % 42 = 24 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l896_89681


namespace NUMINAMATH_CALUDE_units_digit_of_G_1009_l896_89664

-- Define G_n
def G (n : ℕ) : ℕ := 3^(2^n) + 1

-- Define the function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_of_G_1009 : unitsDigit (G 1009) = 4 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_G_1009_l896_89664


namespace NUMINAMATH_CALUDE_ratio_sum_to_last_l896_89623

theorem ratio_sum_to_last {a b c : ℝ} (h : a / c = 3 / 7 ∧ b / c = 4 / 7) :
  (a + b + c) / c = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_to_last_l896_89623


namespace NUMINAMATH_CALUDE_sequence_theorem_l896_89685

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, Real.sqrt (a (n + 1)) - Real.sqrt (a n) = d * n + (Real.sqrt (a 1) - Real.sqrt (a 0))

theorem sequence_theorem (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n : ℕ, Real.sqrt (a (n + 1)) - Real.sqrt (a n) = 2 * n - 2) →
  a 1 = 1 →
  a 3 = 9 →
  ∀ n : ℕ, a n = (n^2 - 3*n + 3)^2 :=
by sorry

end NUMINAMATH_CALUDE_sequence_theorem_l896_89685


namespace NUMINAMATH_CALUDE_integral_equals_two_plus_half_pi_l896_89693

open Set
open MeasureTheory
open Interval

theorem integral_equals_two_plus_half_pi :
  ∫ x in (Icc (-1) 1), (1 + x + Real.sqrt (1 - x^2)) = 2 + π / 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_two_plus_half_pi_l896_89693


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l896_89660

theorem quadratic_two_real_roots (b c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ ∀ z : ℝ, z^2 + b*z + c = 0 ↔ z = x ∨ z = y) ↔ b^2 - 4*c ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l896_89660


namespace NUMINAMATH_CALUDE_total_students_l896_89698

theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 6) (h2 : girls = 200) :
  boys + girls = 440 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l896_89698


namespace NUMINAMATH_CALUDE_absolute_value_sum_difference_l896_89668

theorem absolute_value_sum_difference (a b c : ℚ) :
  a = -1/4 → b = -2 → c = -11/4 → |a| + |b| - |c| = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_difference_l896_89668


namespace NUMINAMATH_CALUDE_earliest_82_degrees_l896_89610

/-- The temperature function modeling the temperature in Denver, CO -/
def temperature (t : ℝ) : ℝ := -2 * t^2 + 16 * t + 40

/-- Theorem stating that the earliest non-negative time when the temperature reaches 82 degrees is 3 hours past noon -/
theorem earliest_82_degrees :
  ∀ t : ℝ, t ≥ 0 → temperature t = 82 → t ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_earliest_82_degrees_l896_89610


namespace NUMINAMATH_CALUDE_square_of_85_l896_89649

theorem square_of_85 : (85 : ℕ) ^ 2 = 7225 := by
  sorry

end NUMINAMATH_CALUDE_square_of_85_l896_89649


namespace NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l896_89690

/-- Given a geometric sequence where the 3rd term is 5 and the 6th term is 40,
    the 9th term is 320. -/
theorem geometric_sequence_ninth_term : ∀ (a : ℕ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * (a 4 / a 3)) →  -- Geometric sequence condition
  a 3 = 5 →                                   -- 3rd term is 5
  a 6 = 40 →                                  -- 6th term is 40
  a 9 = 320 :=                                -- 9th term is 320
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_ninth_term_l896_89690


namespace NUMINAMATH_CALUDE_olivia_wallet_balance_l896_89634

theorem olivia_wallet_balance (initial amount_collected amount_spent : ℕ) :
  initial = 100 →
  amount_collected = 148 →
  amount_spent = 89 →
  initial + amount_collected - amount_spent = 159 :=
by sorry

end NUMINAMATH_CALUDE_olivia_wallet_balance_l896_89634


namespace NUMINAMATH_CALUDE_teacher_age_l896_89665

/-- Given a class of students and their teacher, this theorem proves the teacher's age
    based on how the average age changes when including the teacher. -/
theorem teacher_age (num_students : ℕ) (student_avg_age teacher_age : ℝ) 
    (h1 : num_students = 25)
    (h2 : student_avg_age = 26)
    (h3 : (num_students * student_avg_age + teacher_age) / (num_students + 1) = student_avg_age + 1) :
  teacher_age = 52 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l896_89665


namespace NUMINAMATH_CALUDE_simplify_fraction_l896_89615

theorem simplify_fraction (x : ℝ) (h : x ≠ 1) :
  (x^2 + 1) / (x - 1) - 2 * x / (x - 1) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l896_89615


namespace NUMINAMATH_CALUDE_ken_change_l896_89676

/-- Represents the grocery purchase and payment scenario --/
def grocery_purchase (steak_price : ℕ) (steak_quantity : ℕ) (eggs_price : ℕ) 
  (milk_price : ℕ) (bagels_price : ℕ) (bill_20 : ℕ) (bill_10 : ℕ) 
  (bill_5 : ℕ) (coin_1 : ℕ) : Prop :=
  let total_cost := steak_price * steak_quantity + eggs_price + milk_price + bagels_price
  let total_paid := 20 * bill_20 + 10 * bill_10 + 5 * bill_5 + coin_1
  total_paid - total_cost = 16

/-- Theorem stating that Ken will receive $16 in change --/
theorem ken_change : grocery_purchase 7 2 3 4 6 1 1 2 3 := by
  sorry

end NUMINAMATH_CALUDE_ken_change_l896_89676


namespace NUMINAMATH_CALUDE_sally_payment_l896_89632

/-- Calculates the amount Sally needs to pay out of pocket for books -/
def sally_out_of_pocket (given_amount : ℕ) (book_cost : ℕ) (num_students : ℕ) : ℕ :=
  max 0 (book_cost * num_students - given_amount)

/-- Proves that Sally needs to pay $205 out of pocket -/
theorem sally_payment : sally_out_of_pocket 320 15 35 = 205 := by
  sorry

end NUMINAMATH_CALUDE_sally_payment_l896_89632


namespace NUMINAMATH_CALUDE_simplify_expression_1_l896_89639

theorem simplify_expression_1 : 
  Real.sqrt 8 + Real.sqrt (1/3) - 2 * Real.sqrt 2 = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_l896_89639


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l896_89663

theorem condition_necessary_not_sufficient :
  (∀ x : ℝ, x = 0 → (2*x - 1)*x = 0) ∧
  (∃ x : ℝ, (2*x - 1)*x = 0 ∧ x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l896_89663


namespace NUMINAMATH_CALUDE_a_most_stable_l896_89691

/-- Represents a person's shooting performance data -/
structure ShootingData where
  name : String
  variance : Real

/-- Defines stability of shooting performance based on variance -/
def isMoreStable (a b : ShootingData) : Prop :=
  a.variance < b.variance

/-- Theorem: A has the most stable shooting performance -/
theorem a_most_stable (a b c d : ShootingData)
  (ha : a.name = "A" ∧ a.variance = 0.6)
  (hb : b.name = "B" ∧ b.variance = 1.1)
  (hc : c.name = "C" ∧ c.variance = 0.9)
  (hd : d.name = "D" ∧ d.variance = 1.2) :
  isMoreStable a b ∧ isMoreStable a c ∧ isMoreStable a d :=
sorry

end NUMINAMATH_CALUDE_a_most_stable_l896_89691


namespace NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l896_89675

theorem condition_neither_sufficient_nor_necessary
  (m n : ℕ+) :
  ¬(∀ a b : ℝ, a > b → (a^(m:ℕ) - b^(m:ℕ)) * (a^(n:ℕ) - b^(n:ℕ)) > 0) ∧
  ¬(∀ a b : ℝ, (a^(m:ℕ) - b^(m:ℕ)) * (a^(n:ℕ) - b^(n:ℕ)) > 0 → a > b) :=
by sorry

end NUMINAMATH_CALUDE_condition_neither_sufficient_nor_necessary_l896_89675


namespace NUMINAMATH_CALUDE_school_play_seating_l896_89637

/-- Given a school play seating arrangement, prove the number of unoccupied seats. -/
theorem school_play_seating (rows : ℕ) (chairs_per_row : ℕ) (occupied_seats : ℕ) 
  (h1 : rows = 40)
  (h2 : chairs_per_row = 20)
  (h3 : occupied_seats = 790) :
  rows * chairs_per_row - occupied_seats = 10 := by
  sorry

end NUMINAMATH_CALUDE_school_play_seating_l896_89637


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l896_89662

/-- Calculate the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 40 →
  passing_time = 25.2 →
  ∃ (bridge_length : ℝ), bridge_length = 160 ∧
    bridge_length = train_speed_kmh * 1000 / 3600 * passing_time - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l896_89662


namespace NUMINAMATH_CALUDE_sin_cos_equality_implies_ten_degrees_l896_89661

theorem sin_cos_equality_implies_ten_degrees (x : ℝ) :
  Real.sin (4 * x * π / 180) * Real.sin (5 * x * π / 180) = 
  Real.cos (4 * x * π / 180) * Real.cos (5 * x * π / 180) →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equality_implies_ten_degrees_l896_89661


namespace NUMINAMATH_CALUDE_cos_five_pi_thirds_equals_one_half_l896_89619

theorem cos_five_pi_thirds_equals_one_half : 
  Real.cos (5 * π / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_five_pi_thirds_equals_one_half_l896_89619


namespace NUMINAMATH_CALUDE_division_problem_l896_89652

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 507 → divisor = 8 → remainder = 19 → 
  dividend = divisor * quotient + remainder →
  quotient = 61 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l896_89652


namespace NUMINAMATH_CALUDE_fraction_problem_l896_89686

theorem fraction_problem (n d : ℚ) : 
  n / (d + 1) = 1 / 2 → (n + 1) / d = 1 → n / d = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l896_89686


namespace NUMINAMATH_CALUDE_smallest_n_sum_all_digits_same_l896_89642

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Checks if a number has all digits the same -/
def all_digits_same (n : ℕ) : Prop :=
  ∃ d : ℕ, d < 10 ∧ n = d * 111

/-- The smallest n such that sum_first_n(n) is a three-digit number with all digits the same -/
theorem smallest_n_sum_all_digits_same :
  ∃ n : ℕ, 
    (∀ m : ℕ, m < n → ¬(all_digits_same (sum_first_n m))) ∧
    (all_digits_same (sum_first_n n)) ∧
    n = 36 := by sorry

end NUMINAMATH_CALUDE_smallest_n_sum_all_digits_same_l896_89642


namespace NUMINAMATH_CALUDE_middle_term_coefficient_2x_plus_1_power_8_l896_89607

theorem middle_term_coefficient_2x_plus_1_power_8 :
  let n : ℕ := 8
  let k : ℕ := n / 2
  let coeff : ℕ := Nat.choose n k * (2^k)
  coeff = 1120 :=
by sorry

end NUMINAMATH_CALUDE_middle_term_coefficient_2x_plus_1_power_8_l896_89607


namespace NUMINAMATH_CALUDE_vacant_seats_l896_89628

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) (h1 : total_seats = 600) (h2 : filled_percentage = 1/2) :
  (total_seats : ℚ) * (1 - filled_percentage) = 300 := by
  sorry

end NUMINAMATH_CALUDE_vacant_seats_l896_89628


namespace NUMINAMATH_CALUDE_chessboard_rearrangement_3x3_no_chessboard_rearrangement_8x8_l896_89689

/-- Represents a chessboard of size N × N -/
structure Chessboard (N : ℕ) where
  size : ℕ
  size_eq : size = N

/-- Represents a position on the chessboard -/
structure Position (N : ℕ) where
  row : Fin N
  col : Fin N

/-- Knight's move distance between two positions -/
def knightDistance (N : ℕ) (p1 p2 : Position N) : ℕ :=
  sorry

/-- King's move distance between two positions -/
def kingDistance (N : ℕ) (p1 p2 : Position N) : ℕ :=
  sorry

/-- Represents a rearrangement of checkers on the board -/
def Rearrangement (N : ℕ) := Position N → Position N

/-- Checks if a rearrangement satisfies the problem condition -/
def isValidRearrangement (N : ℕ) (r : Rearrangement N) : Prop :=
  ∀ p1 p2 : Position N, knightDistance N p1 p2 = 1 → kingDistance N (r p1) (r p2) = 1

theorem chessboard_rearrangement_3x3 :
  ∃ (r : Rearrangement 3), isValidRearrangement 3 r :=
sorry

theorem no_chessboard_rearrangement_8x8 :
  ¬ ∃ (r : Rearrangement 8), isValidRearrangement 8 r :=
sorry

end NUMINAMATH_CALUDE_chessboard_rearrangement_3x3_no_chessboard_rearrangement_8x8_l896_89689


namespace NUMINAMATH_CALUDE_garden_fencing_cost_l896_89654

/-- The cost of fencing a rectangular garden -/
theorem garden_fencing_cost
  (garden_width : ℝ)
  (playground_length playground_width : ℝ)
  (fencing_price : ℝ)
  (h1 : garden_width = 12)
  (h2 : playground_length = 16)
  (h3 : playground_width = 12)
  (h4 : fencing_price = 15)
  (h5 : garden_width * (playground_length * playground_width / garden_width) = playground_length * playground_width) :
  2 * (garden_width + (playground_length * playground_width / garden_width)) * fencing_price = 840 :=
by sorry

end NUMINAMATH_CALUDE_garden_fencing_cost_l896_89654


namespace NUMINAMATH_CALUDE_blue_pill_cost_proof_l896_89609

def total_cost : ℚ := 430
def days : ℕ := 10
def blue_red_diff : ℚ := 3

def blue_pill_cost : ℚ := 23

theorem blue_pill_cost_proof :
  (blue_pill_cost * days + (blue_pill_cost - blue_red_diff) * days = total_cost) ∧
  (blue_pill_cost > 0) ∧
  (blue_pill_cost - blue_red_diff > 0) :=
by sorry

end NUMINAMATH_CALUDE_blue_pill_cost_proof_l896_89609


namespace NUMINAMATH_CALUDE_solve_necklace_cost_l896_89645

def necklace_cost_problem (necklace_cost book_cost total_cost spending_limit overspend : ℚ) : Prop :=
  book_cost = necklace_cost + 5 ∧
  spending_limit = 70 ∧
  overspend = 3 ∧
  total_cost = necklace_cost + book_cost ∧
  total_cost = spending_limit + overspend ∧
  necklace_cost = 34

theorem solve_necklace_cost :
  ∃ (necklace_cost book_cost total_cost spending_limit overspend : ℚ),
    necklace_cost_problem necklace_cost book_cost total_cost spending_limit overspend :=
by sorry

end NUMINAMATH_CALUDE_solve_necklace_cost_l896_89645


namespace NUMINAMATH_CALUDE_unique_solution_condition_l896_89678

theorem unique_solution_condition (k : ℝ) : 
  (∃! x y : ℝ, y = x^2 + k ∧ y = 3*x) ↔ k = 9/4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l896_89678


namespace NUMINAMATH_CALUDE_always_positive_l896_89620

theorem always_positive (x : ℝ) : 3 * x^2 - 6 * x + 3.5 > 0 := by
  sorry

end NUMINAMATH_CALUDE_always_positive_l896_89620


namespace NUMINAMATH_CALUDE_truthful_dwarfs_l896_89617

theorem truthful_dwarfs (n : ℕ) (h_n : n = 10) 
  (raised_vanilla raised_chocolate raised_fruit : ℕ)
  (h_vanilla : raised_vanilla = n)
  (h_chocolate : raised_chocolate = n / 2)
  (h_fruit : raised_fruit = 1) :
  ∃ (truthful liars : ℕ),
    truthful + liars = n ∧
    truthful + 2 * liars = raised_vanilla + raised_chocolate + raised_fruit ∧
    truthful = 4 ∧
    liars = 6 := by
  sorry

end NUMINAMATH_CALUDE_truthful_dwarfs_l896_89617


namespace NUMINAMATH_CALUDE_min_value_expression_l896_89611

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + b^2 + 1 / (a + b)^3 ≥ 1 / (4^(1/5 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l896_89611


namespace NUMINAMATH_CALUDE_triangle_property_l896_89643

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    proves that under certain conditions, angle A is π/3 and the area is 3√3. -/
theorem triangle_property (a b c A B C : ℝ) : 
  0 < A ∧ A < π ∧   -- A is in (0, π)
  0 < B ∧ B < π ∧   -- B is in (0, π)
  0 < C ∧ C < π ∧   -- C is in (0, π)
  A + B + C = π ∧   -- Sum of angles in a triangle
  a > 0 ∧ b > 0 ∧ c > 0 ∧   -- Positive side lengths
  a * Real.sin C = Real.sqrt 3 * c * Real.cos A ∧   -- Given condition
  a = Real.sqrt 13 ∧   -- Given value of a
  c = 3 →   -- Given value of c
  A = π / 3 ∧   -- Angle A is 60°
  (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3   -- Area of triangle ABC
  := by sorry

end NUMINAMATH_CALUDE_triangle_property_l896_89643


namespace NUMINAMATH_CALUDE_flour_mass_acceptance_l896_89631

-- Define the labeled mass and uncertainty
def labeled_mass : ℝ := 35
def uncertainty : ℝ := 0.25

-- Define the acceptable range
def min_acceptable : ℝ := labeled_mass - uncertainty
def max_acceptable : ℝ := labeled_mass + uncertainty

-- Define the masses of the flour bags
def mass_A : ℝ := 34.70
def mass_B : ℝ := 34.80
def mass_C : ℝ := 35.30
def mass_D : ℝ := 35.51

-- Theorem to prove
theorem flour_mass_acceptance :
  (min_acceptable ≤ mass_B ∧ mass_B ≤ max_acceptable) ∧
  (mass_A < min_acceptable ∨ mass_A > max_acceptable) ∧
  (mass_C < min_acceptable ∨ mass_C > max_acceptable) ∧
  (mass_D < min_acceptable ∨ mass_D > max_acceptable) := by
  sorry

end NUMINAMATH_CALUDE_flour_mass_acceptance_l896_89631


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l896_89614

/-- Given a right triangle with area 24 cm² and hypotenuse 10 cm, 
    prove that the radius of its inscribed circle is 2 cm. -/
theorem inscribed_circle_radius 
  (S : ℝ) 
  (c : ℝ) 
  (h1 : S = 24) 
  (h2 : c = 10) : 
  let a := Real.sqrt ((c^2 / 2) + Real.sqrt ((c^4 / 4) - S^2))
  let b := Real.sqrt ((c^2 / 2) - Real.sqrt ((c^4 / 4) - S^2))
  (a + b - c) / 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l896_89614


namespace NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l896_89635

theorem coefficient_x4_in_expansion : 
  let expansion := (fun x => (2 * x + 1) * (x - 3)^5)
  ∃ (a b c d e f : ℤ), 
    (∀ x, expansion x = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f) ∧
    b = 165 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_in_expansion_l896_89635


namespace NUMINAMATH_CALUDE_range_of_m_l896_89622

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, (9 : ℝ)^x - m*(3 : ℝ)^x + 4 ≤ 0) → m ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l896_89622


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l896_89653

theorem quadratic_coefficient_sum (m n : ℤ) : 
  (∀ x : ℤ, (x + 2) * (x - 1) = x^2 + m*x + n) → m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l896_89653


namespace NUMINAMATH_CALUDE_janes_homework_l896_89618

theorem janes_homework (x y z : ℝ) 
  (h1 : x - (y + z) = 15) 
  (h2 : x - y + z = 7) : 
  x - y = 11 := by
sorry

end NUMINAMATH_CALUDE_janes_homework_l896_89618


namespace NUMINAMATH_CALUDE_polynomial_product_identity_l896_89674

theorem polynomial_product_identity (x z : ℝ) :
  (3 * x^4 - 4 * z^3) * (9 * x^8 + 12 * x^4 * z^3 + 16 * z^6) = 27 * x^12 - 64 * z^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_identity_l896_89674


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l896_89659

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of blocks that can fit in a box -/
def maxBlocksFit (box : BoxDimensions) (block : BlockDimensions) : ℕ :=
  (box.length / block.length) * (box.width / block.width) * (box.height / block.height)

theorem max_blocks_in_box :
  let box : BoxDimensions := ⟨5, 4, 3⟩
  let block : BlockDimensions := ⟨1, 2, 2⟩
  maxBlocksFit box block = 12 := by
  sorry


end NUMINAMATH_CALUDE_max_blocks_in_box_l896_89659


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l896_89600

theorem rectangle_area_increase (x y : ℝ) (h1 : x > 0) (h2 : y > 0) :
  let original_area := x * y
  let new_length := 1.2 * x
  let new_width := 1.1 * y
  let new_area := new_length * new_width
  (new_area - original_area) / original_area = 0.32 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l896_89600


namespace NUMINAMATH_CALUDE_merchant_markup_percentage_l896_89670

theorem merchant_markup_percentage (C : ℝ) (M : ℝ) : 
  C > 0 →
  ((1 + M / 100) * C - 0.4 * ((1 + M / 100) * C) = 1.05 * C) →
  M = 75 := by
sorry

end NUMINAMATH_CALUDE_merchant_markup_percentage_l896_89670


namespace NUMINAMATH_CALUDE_second_number_value_l896_89669

theorem second_number_value : 
  ∀ (a b c d : ℝ),
  a + b + c + d = 280 →
  a = 2 * b →
  c = (1/3) * a →
  d = b + c →
  b = 52.5 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l896_89669


namespace NUMINAMATH_CALUDE_intersection_distance_l896_89640

-- Define the curves and ray
def C₁ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1
def C₂ (x y θ : ℝ) : Prop := x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sin θ
def ray (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * x ∧ x ≥ 0

-- Define the intersection points
def point_A (x y : ℝ) : Prop := C₁ x y ∧ ray x y
def point_B (x y θ : ℝ) : Prop := C₂ x y θ ∧ ray x y

-- Theorem statement
theorem intersection_distance :
  ∀ (x₁ y₁ x₂ y₂ θ : ℝ),
  point_A x₁ y₁ → point_B x₂ y₂ θ →
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = Real.sqrt 3 - 2 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_l896_89640


namespace NUMINAMATH_CALUDE_park_diameter_l896_89629

/-- Given a circular park with a fountain, garden, and walking path, 
    calculate the diameter of the outer boundary of the walking path. -/
theorem park_diameter (fountain_diameter garden_width path_width : ℝ) 
    (h1 : fountain_diameter = 12)
    (h2 : garden_width = 10)
    (h3 : path_width = 6) :
    fountain_diameter + 2 * garden_width + 2 * path_width = 44 :=
by sorry

end NUMINAMATH_CALUDE_park_diameter_l896_89629


namespace NUMINAMATH_CALUDE_no_single_digit_A_with_integer_solutions_l896_89606

theorem no_single_digit_A_with_integer_solutions : 
  ∀ A : ℕ, 1 ≤ A ∧ A ≤ 9 → 
  ¬∃ x : ℕ, x > 0 ∧ x^2 - 2*A*x + A*10 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_single_digit_A_with_integer_solutions_l896_89606


namespace NUMINAMATH_CALUDE_cinema_seat_unique_point_only_cinema_seat_determines_point_l896_89605

/-- A description of a location --/
inductive LocationDescription
| SchoolLibrary : LocationDescription
| CinemaSeat : Nat → Nat → LocationDescription
| TrainCarriage : Nat → LocationDescription
| Direction : Real → LocationDescription

/-- Predicate to check if a location description determines a unique point --/
def determines_unique_point (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.CinemaSeat _ _ => True
  | _ => False

/-- Theorem stating that only the cinema seat description determines a unique point --/
theorem cinema_seat_unique_point (desc : LocationDescription) :
  determines_unique_point desc ↔ ∃ (row seat : Nat), desc = LocationDescription.CinemaSeat row seat :=
sorry

/-- The main theorem proving that among the given options, only the cinema seat determines a unique point --/
theorem only_cinema_seat_determines_point :
  ∃! (desc : LocationDescription), determines_unique_point desc ∧
  (desc = LocationDescription.SchoolLibrary ∨
   ∃ (row seat : Nat), desc = LocationDescription.CinemaSeat row seat ∨
   ∃ (n : Nat), desc = LocationDescription.TrainCarriage n ∨
   ∃ (angle : Real), desc = LocationDescription.Direction angle) :=
sorry

end NUMINAMATH_CALUDE_cinema_seat_unique_point_only_cinema_seat_determines_point_l896_89605


namespace NUMINAMATH_CALUDE_intersection_equality_range_l896_89627

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2 * a + 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem statement
theorem intersection_equality_range (a : ℝ) :
  A a ∩ B = A a ↔ a ∈ Set.Iic (-4) ∪ Set.Icc (-1) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_range_l896_89627


namespace NUMINAMATH_CALUDE_square_roots_problem_l896_89648

theorem square_roots_problem (n : ℝ) (h_pos : n > 0) :
  (∃ x : ℝ, (x + 1)^2 = n ∧ (4 - 2*x)^2 = n) → n = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l896_89648


namespace NUMINAMATH_CALUDE_game_show_probability_l896_89687

theorem game_show_probability : 
  let num_tables : ℕ := 3
  let boxes_per_table : ℕ := 3
  let zonk_boxes_per_table : ℕ := 1
  let prob_no_zonk_per_table : ℚ := (boxes_per_table - zonk_boxes_per_table) / boxes_per_table
  (prob_no_zonk_per_table ^ num_tables : ℚ) = 8 / 27 := by sorry

end NUMINAMATH_CALUDE_game_show_probability_l896_89687


namespace NUMINAMATH_CALUDE_min_value_of_expression_l896_89625

theorem min_value_of_expression (x y : ℝ) 
  (h1 : x > -1) 
  (h2 : y > 0) 
  (h3 : x + 2*y = 2) : 
  ∃ (m : ℝ), m = 3 ∧ ∀ (a b : ℝ), a > -1 → b > 0 → a + 2*b = 2 → 1/(a+1) + 2/b ≥ m :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l896_89625


namespace NUMINAMATH_CALUDE_total_laundry_cost_l896_89612

def laundry_cost (washer_cost : ℝ) (dryer_cost_per_10_min : ℝ) (loads : ℕ) 
  (special_soap_cost : ℝ) (num_dryers : ℕ) (dryer_time : ℕ) (membership_fee : ℝ) : ℝ :=
  let washing_cost := washer_cost * loads + special_soap_cost
  let dryer_cost := (↑num_dryers * ↑(dryer_time / 10 + 1)) * dryer_cost_per_10_min
  washing_cost + dryer_cost + membership_fee

theorem total_laundry_cost :
  laundry_cost 4 0.25 3 2.5 4 45 10 = 29.5 := by
  sorry

end NUMINAMATH_CALUDE_total_laundry_cost_l896_89612


namespace NUMINAMATH_CALUDE_expression_evaluation_l896_89688

theorem expression_evaluation : 
  let a : ℚ := -1/2
  (a - 2) * (a + 2) - (a + 1) * (a - 3) = -2 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l896_89688


namespace NUMINAMATH_CALUDE_kaleb_books_l896_89679

theorem kaleb_books (initial_books : ℕ) : 
  initial_books - 17 + 7 = 24 → initial_books = 34 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_books_l896_89679


namespace NUMINAMATH_CALUDE_expression_value_l896_89680

theorem expression_value (x y z : ℝ) 
  (eq1 : 2*x - 3*y - z = 0)
  (eq2 : x + 3*y - 14*z = 0)
  (h : z ≠ 0) :
  (x^2 - x*y) / (y^2 + 2*z^2) = 10/11 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l896_89680


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l896_89667

-- Equation 1
theorem equation_one_solution (x : ℝ) : 9 * x^2 = 27 ↔ x = Real.sqrt 3 ∨ x = -Real.sqrt 3 := by sorry

-- Equation 2
theorem equation_two_solution (x : ℝ) : -2 * (x - 3)^3 + 16 = 0 ↔ x = 5 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l896_89667


namespace NUMINAMATH_CALUDE_exists_valid_coloring_l896_89603

-- Define the color type
inductive Color
  | White
  | Red
  | Black

-- Define the coloring function type
def ColoringFunction := ℤ × ℤ → Color

-- Define what it means for a color to appear on infinitely many lines
def AppearsOnInfinitelyManyLines (f : ColoringFunction) (c : Color) : Prop :=
  ∀ n : ℕ, ∃ y : ℤ, y > n ∧ (∀ m : ℕ, ∃ x : ℤ, x > m ∧ f (x, y) = c)

-- Define what it means to be a parallelogram
def IsParallelogram (A B C D : ℤ × ℤ) : Prop :=
  B.1 - A.1 = D.1 - C.1 ∧ B.2 - A.2 = D.2 - C.2

-- Main theorem
theorem exists_valid_coloring : ∃ f : ColoringFunction,
  (AppearsOnInfinitelyManyLines f Color.White) ∧
  (AppearsOnInfinitelyManyLines f Color.Red) ∧
  (AppearsOnInfinitelyManyLines f Color.Black) ∧
  (∀ A B C : ℤ × ℤ, f A = Color.White → f B = Color.Red → f C = Color.Black →
    ∃ D : ℤ × ℤ, f D = Color.Red ∧ IsParallelogram A B C D) :=
sorry

end NUMINAMATH_CALUDE_exists_valid_coloring_l896_89603


namespace NUMINAMATH_CALUDE_range_of_m_for_equation_l896_89650

theorem range_of_m_for_equation (P : Prop) 
  (h : P ↔ ∀ x : ℝ, ∃ m : ℝ, 4^x - 2^(x+1) + m = 0) : 
  P → ∀ m : ℝ, (∃ x : ℝ, 4^x - 2^(x+1) + m = 0) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_equation_l896_89650


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l896_89695

theorem decimal_to_fraction : 
  (2.36 : ℚ) = 59 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l896_89695


namespace NUMINAMATH_CALUDE_min_value_theorem_l896_89666

/-- Triangle ABC with area 2 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (area : ℝ)
  (area_eq : area = 2)

/-- Function f mapping a point to areas of subtriangles -/
def f (T : Triangle) (P : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the minimum value of 1/x + 4/y -/
theorem min_value_theorem (T : Triangle) :
  ∀ P : ℝ × ℝ, 
  ∀ x y : ℝ,
  f T P = (1, x, y) →
  (∀ a b : ℝ, f T (a, b) = (1, x, y) → 1/x + 4/y ≥ 9) ∧ 
  (∃ a b : ℝ, f T (a, b) = (1, x, y) ∧ 1/x + 4/y = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l896_89666


namespace NUMINAMATH_CALUDE_proof_method_characteristics_proof_method_relationship_l896_89697

/-- Represents a proof method -/
inductive ProofMethod
| Synthetic
| Analytic

/-- Represents the direction of reasoning -/
inductive ReasoningDirection
| KnownToConclusion
| ConclusionToKnown

/-- Defines the characteristics of a proof method -/
structure ProofMethodCharacteristics where
  method : ProofMethod
  direction : ReasoningDirection

/-- Defines the relationship between two proof methods -/
structure ProofMethodRelationship where
  method1 : ProofMethod
  method2 : ProofMethod
  oppositeThoughtProcess : Bool
  inverseProcedures : Bool

/-- Theorem stating the characteristics of synthetic and analytic methods -/
theorem proof_method_characteristics :
  ∃ (synthetic analytic : ProofMethodCharacteristics),
    synthetic.method = ProofMethod.Synthetic ∧
    synthetic.direction = ReasoningDirection.KnownToConclusion ∧
    analytic.method = ProofMethod.Analytic ∧
    analytic.direction = ReasoningDirection.ConclusionToKnown :=
  sorry

/-- Theorem stating the relationship between synthetic and analytic methods -/
theorem proof_method_relationship :
  ∃ (relationship : ProofMethodRelationship),
    relationship.method1 = ProofMethod.Synthetic ∧
    relationship.method2 = ProofMethod.Analytic ∧
    relationship.oppositeThoughtProcess = true ∧
    relationship.inverseProcedures = true :=
  sorry

end NUMINAMATH_CALUDE_proof_method_characteristics_proof_method_relationship_l896_89697


namespace NUMINAMATH_CALUDE_profit_percentage_calculation_l896_89630

theorem profit_percentage_calculation (selling_price profit : ℝ) :
  selling_price = 850 →
  profit = 215 →
  let cost_price := selling_price - profit
  let profit_percentage := (profit / cost_price) * 100
  ∃ ε > 0, abs (profit_percentage - 33.86) < ε :=
by sorry

end NUMINAMATH_CALUDE_profit_percentage_calculation_l896_89630


namespace NUMINAMATH_CALUDE_dog_food_insufficient_l896_89684

/-- Proves that the amount of dog food remaining after two weeks is negative -/
theorem dog_food_insufficient (num_dogs : ℕ) (food_per_meal : ℚ) (meals_per_day : ℕ) 
  (initial_food : ℚ) (days : ℕ) :
  num_dogs = 5 →
  food_per_meal = 3/4 →
  meals_per_day = 3 →
  initial_food = 45 →
  days = 14 →
  initial_food - (num_dogs * food_per_meal * meals_per_day * days) < 0 :=
by sorry

end NUMINAMATH_CALUDE_dog_food_insufficient_l896_89684


namespace NUMINAMATH_CALUDE_max_students_before_third_wave_l896_89613

/-- The total number of students in the class -/
def total_students : ℕ := 35

/-- A function that checks if a number is prime -/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- The theorem to be proved -/
theorem max_students_before_third_wave :
  ∃ (a b c : ℕ),
    is_prime a ∧ is_prime b ∧ is_prime c ∧
    a + b + c = total_students ∧
    ∀ (x y z : ℕ),
      is_prime x ∧ is_prime y ∧ is_prime z ∧
      x + y + z = total_students →
      total_students - (a + b) ≥ total_students - (x + y) :=
sorry

end NUMINAMATH_CALUDE_max_students_before_third_wave_l896_89613


namespace NUMINAMATH_CALUDE_quadratic_completion_of_square_l896_89641

theorem quadratic_completion_of_square (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 7*x + 6 = a*(x - h)^2 + k ∧ k = -25/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_of_square_l896_89641


namespace NUMINAMATH_CALUDE_victor_score_l896_89657

/-- 
Given a maximum mark and a percentage score, calculate the actual score.
-/
def calculateScore (maxMark : ℕ) (percentage : ℚ) : ℚ :=
  percentage * maxMark

theorem victor_score :
  let maxMark : ℕ := 300
  let percentage : ℚ := 80 / 100
  calculateScore maxMark percentage = 240 := by
  sorry

end NUMINAMATH_CALUDE_victor_score_l896_89657


namespace NUMINAMATH_CALUDE_modified_binomial_coefficient_integrality_l896_89621

theorem modified_binomial_coefficient_integrality 
  (k n : ℕ) (h1 : 1 ≤ k) (h2 : k < n) : 
  ∃ m : ℤ, (n - 3 * k - 2 : ℤ) * (n.factorial) = 
    (k + 2 : ℤ) * m * (k.factorial) * ((n - k).factorial) := by
  sorry

end NUMINAMATH_CALUDE_modified_binomial_coefficient_integrality_l896_89621


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l896_89624

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (9 ∣ (n + 7)) ∧ 
  (6 ∣ (n - 9)) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (9 ∣ (m + 7)) ∧ (6 ∣ (m - 9))) → False) ∧
  n = 101 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_conditions_l896_89624


namespace NUMINAMATH_CALUDE_parabola_through_three_points_l896_89673

/-- A parabola with equation y = x^2 + bx + c passing through (-1, -11), (3, 17), and (2, 5) has b = 13/3 and c = -5 -/
theorem parabola_through_three_points :
  ∀ b c : ℚ,
  ((-1)^2 + b*(-1) + c = -11) →
  (3^2 + b*3 + c = 17) →
  (2^2 + b*2 + c = 5) →
  (b = 13/3 ∧ c = -5) :=
by sorry

end NUMINAMATH_CALUDE_parabola_through_three_points_l896_89673


namespace NUMINAMATH_CALUDE_unique_solution_system_l896_89601

/-- Given positive real numbers a, b, c satisfying √a + √b + √c = √π/2,
    prove that there exists a unique triple (x, y, z) of real numbers
    satisfying the system of equations:
    √(y-a) + √(z-a) = 1
    √(z-b) + √(x-b) = 1
    √(x-c) + √(y-c) = 1 -/
theorem unique_solution_system (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = Real.sqrt (π / 2)) :
  ∃! x y z : ℝ,
    Real.sqrt (y - a) + Real.sqrt (z - a) = 1 ∧
    Real.sqrt (z - b) + Real.sqrt (x - b) = 1 ∧
    Real.sqrt (x - c) + Real.sqrt (y - c) = 1 :=
by sorry


end NUMINAMATH_CALUDE_unique_solution_system_l896_89601


namespace NUMINAMATH_CALUDE_tom_rides_l896_89626

/-- Given the total number of tickets, tickets spent, and cost per ride,
    calculate the number of rides Tom can go on. -/
def number_of_rides (total_tickets spent_tickets cost_per_ride : ℕ) : ℕ :=
  (total_tickets - spent_tickets) / cost_per_ride

/-- Theorem stating that Tom can go on 3 rides given the specific conditions. -/
theorem tom_rides : number_of_rides 40 28 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tom_rides_l896_89626


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l896_89692

theorem quadratic_equation_roots (a b c : ℝ) (h : a = 1 ∧ b = -8 ∧ c = 16) :
  ∃! x : ℝ, x^2 - 8*x + 16 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l896_89692


namespace NUMINAMATH_CALUDE_tetrahedron_cross_section_l896_89677

noncomputable def cross_section_area (V : ℝ) (d : ℝ) : ℝ :=
  3 * V / (5 * d)

theorem tetrahedron_cross_section 
  (V : ℝ) 
  (d : ℝ) 
  (h_V : V = 5) 
  (h_d : d = 1) :
  cross_section_area V d = 3 := by
sorry

end NUMINAMATH_CALUDE_tetrahedron_cross_section_l896_89677


namespace NUMINAMATH_CALUDE_not_q_is_false_l896_89638

theorem not_q_is_false (p q : Prop) (hp : ¬p) (hq : q) : ¬(¬q) := by
  sorry

end NUMINAMATH_CALUDE_not_q_is_false_l896_89638


namespace NUMINAMATH_CALUDE_circular_tablecloth_radius_increase_l896_89696

theorem circular_tablecloth_radius_increase :
  let initial_circumference : ℝ := 50
  let final_circumference : ℝ := 64
  let initial_radius : ℝ := initial_circumference / (2 * Real.pi)
  let final_radius : ℝ := final_circumference / (2 * Real.pi)
  final_radius - initial_radius = 7 / Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circular_tablecloth_radius_increase_l896_89696


namespace NUMINAMATH_CALUDE_fraction_sum_and_multiply_l896_89608

theorem fraction_sum_and_multiply :
  ((2 : ℚ) / 9 + 4 / 11) * 3 / 5 = 58 / 165 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_and_multiply_l896_89608


namespace NUMINAMATH_CALUDE_apartment_occupancy_l896_89644

theorem apartment_occupancy (stories : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ) 
  (h1 : stories = 25)
  (h2 : apartments_per_floor = 4)
  (h3 : total_people = 200) :
  total_people / (stories * apartments_per_floor) = 2 := by
sorry

end NUMINAMATH_CALUDE_apartment_occupancy_l896_89644


namespace NUMINAMATH_CALUDE_quadratic_expression_equality_l896_89647

theorem quadratic_expression_equality (x y : ℝ) 
  (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 
  10 * x^2 + 19 * x * y + 10 * y^2 = 153 := by sorry

end NUMINAMATH_CALUDE_quadratic_expression_equality_l896_89647


namespace NUMINAMATH_CALUDE_mollys_brothers_children_l896_89636

/-- The number of children each of Molly's brothers has -/
def children_per_brother : ℕ := 2

theorem mollys_brothers_children :
  let cost_per_package : ℕ := 5
  let num_parents : ℕ := 2
  let num_brothers : ℕ := 3
  let total_cost : ℕ := 70
  let immediate_family : ℕ := num_parents + num_brothers + num_brothers -- includes spouses
  (cost_per_package * (immediate_family + num_brothers * children_per_brother) = total_cost) ∧
  (children_per_brother > 0) :=
by sorry

end NUMINAMATH_CALUDE_mollys_brothers_children_l896_89636


namespace NUMINAMATH_CALUDE_small_animal_weight_l896_89651

def bear_weight_gain (total_weight : ℝ) (berry_fraction : ℝ) (acorn_multiplier : ℝ) (salmon_fraction : ℝ) : ℝ :=
  let berry_weight := total_weight * berry_fraction
  let acorn_weight := berry_weight * acorn_multiplier
  let remaining_weight := total_weight - (berry_weight + acorn_weight)
  let salmon_weight := remaining_weight * salmon_fraction
  total_weight - (berry_weight + acorn_weight + salmon_weight)

theorem small_animal_weight :
  bear_weight_gain 1000 (1/5) 2 (1/2) = 200 := by
  sorry

end NUMINAMATH_CALUDE_small_animal_weight_l896_89651


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocal_sin_squared_l896_89683

theorem min_value_sum_reciprocal_sin_squared (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C → -- Angles are positive
  A + B + C = π → -- Sum of angles in a triangle
  C = π / 2 → -- Right angle condition
  (∀ x y : ℝ, 0 < x → 0 < y → x + y + π/2 = π → 4 / (Real.sin x)^2 + 9 / (Real.sin y)^2 ≥ 25) ∧ 
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y + π/2 = π ∧ 4 / (Real.sin x)^2 + 9 / (Real.sin y)^2 = 25) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocal_sin_squared_l896_89683


namespace NUMINAMATH_CALUDE_handshake_count_total_handshakes_l896_89671

theorem handshake_count : ℕ :=
  let twin_sets : ℕ := 12
  let triplet_sets : ℕ := 8
  let twins : ℕ := twin_sets * 2
  let triplets : ℕ := triplet_sets * 3
  let twin_handshakes : ℕ := (twins * (twins - 2)) / 2
  let triplet_handshakes : ℕ := (triplets * (triplets - 3)) / 2
  let cross_handshakes : ℕ := twins * (2 * triplets / 3)
  twin_handshakes + triplet_handshakes + cross_handshakes

theorem total_handshakes : handshake_count = 900 := by
  sorry

end NUMINAMATH_CALUDE_handshake_count_total_handshakes_l896_89671


namespace NUMINAMATH_CALUDE_f_of_tan_squared_l896_89658

noncomputable def f (x : ℝ) : ℝ := 1 / ((x / (x - 1)))

theorem f_of_tan_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/2) :
  f (Real.tan t ^ 2) = Real.tan t ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_f_of_tan_squared_l896_89658


namespace NUMINAMATH_CALUDE_inequality_system_solution_range_l896_89604

theorem inequality_system_solution_range (a : ℝ) : 
  (∃! (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x - a ≥ 0 ∧ 2*x < 4))) → 
  (-2 < a ∧ a ≤ -1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_range_l896_89604


namespace NUMINAMATH_CALUDE_waiter_customers_l896_89694

/-- Represents the number of customers a waiter had at lunch -/
def lunch_customers (non_tipping : ℕ) (tip_amount : ℕ) (total_tips : ℕ) : ℕ :=
  non_tipping + (total_tips / tip_amount)

/-- Theorem stating the number of customers the waiter had at lunch -/
theorem waiter_customers :
  lunch_customers 4 9 27 = 7 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l896_89694


namespace NUMINAMATH_CALUDE_bed_sheet_problem_l896_89646

/-- Calculates the length of a bed-sheet in meters given the total cutting time, 
    time per cut, and length of each piece. -/
def bed_sheet_length (total_time : ℕ) (time_per_cut : ℕ) (piece_length : ℕ) : ℚ :=
  (total_time / time_per_cut) * piece_length / 100

/-- Proves that a bed-sheet cut into 20cm pieces, taking 5 minutes per cut and 
    245 minutes total, is 9.8 meters long. -/
theorem bed_sheet_problem : bed_sheet_length 245 5 20 = 9.8 := by
  sorry

#eval bed_sheet_length 245 5 20

end NUMINAMATH_CALUDE_bed_sheet_problem_l896_89646


namespace NUMINAMATH_CALUDE_final_water_fraction_l896_89633

/-- Represents the fraction of water in a mixture after a certain number of replacements -/
def water_fraction (initial_water : ℚ) (total_volume : ℚ) (replacement_volume : ℚ) (num_replacements : ℕ) : ℚ :=
  initial_water * ((total_volume - replacement_volume) / total_volume) ^ num_replacements

/-- The fraction of water in the final mixture after three replacements -/
theorem final_water_fraction : 
  water_fraction (18/20) 20 5 3 = 243/640 := by
  sorry

#eval water_fraction (18/20) 20 5 3

end NUMINAMATH_CALUDE_final_water_fraction_l896_89633


namespace NUMINAMATH_CALUDE_cost_for_3150_pencils_l896_89699

/-- Calculates the total cost of pencils with a bulk discount --/
def total_cost_with_discount (pencils_per_box : ℕ) (regular_price : ℚ) 
  (discount_price : ℚ) (discount_threshold : ℕ) (total_pencils : ℕ) : ℚ :=
  let boxes := (total_pencils + pencils_per_box - 1) / pencils_per_box
  let price_per_box := if total_pencils > discount_threshold then discount_price else regular_price
  boxes * price_per_box

/-- Theorem stating the total cost for 3150 pencils --/
theorem cost_for_3150_pencils : 
  total_cost_with_discount 150 40 35 2000 3150 = 735 := by
  sorry

end NUMINAMATH_CALUDE_cost_for_3150_pencils_l896_89699


namespace NUMINAMATH_CALUDE_function_extrema_condition_l896_89602

def f (a x : ℝ) : ℝ := x^3 + (a+1)*x^2 + (a+1)*x + a

theorem function_extrema_condition (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) ↔ (a < -1 ∨ a > 2) :=
by sorry

end NUMINAMATH_CALUDE_function_extrema_condition_l896_89602


namespace NUMINAMATH_CALUDE_unique_solution_condition_l896_89682

theorem unique_solution_condition (p q : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + p = q * x + 2) ↔ q ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l896_89682


namespace NUMINAMATH_CALUDE_polynomial_root_implies_h_value_l896_89616

theorem polynomial_root_implies_h_value :
  ∀ h : ℝ, ((-2 : ℝ)^3 + h * (-2) - 12 = 0) → h = -10 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_h_value_l896_89616


namespace NUMINAMATH_CALUDE_fence_cost_per_foot_l896_89655

/-- The cost per foot of building a fence around a square plot -/
theorem fence_cost_per_foot (area : ℝ) (total_cost : ℝ) : area = 289 → total_cost = 4080 → (total_cost / (4 * Real.sqrt area)) = 60 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_per_foot_l896_89655
