import Mathlib

namespace NUMINAMATH_CALUDE_set_of_values_for_a_l1410_141082

theorem set_of_values_for_a (a : ℝ) : 
  (2 ∉ {x : ℝ | x - a < 0}) ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_set_of_values_for_a_l1410_141082


namespace NUMINAMATH_CALUDE_finite_minimal_elements_l1410_141019

def is_minimal {n : ℕ} (A : Set (Fin n → ℕ+)) (a : Fin n → ℕ+) : Prop :=
  a ∈ A ∧ ∀ b ∈ A, (∀ i, b i ≤ a i) → b = a

theorem finite_minimal_elements {n : ℕ} (A : Set (Fin n → ℕ+)) :
  Set.Finite {a ∈ A | is_minimal A a} := by
  sorry

end NUMINAMATH_CALUDE_finite_minimal_elements_l1410_141019


namespace NUMINAMATH_CALUDE_complex_modulus_equality_l1410_141001

theorem complex_modulus_equality : Complex.abs (1/3 - 3*I) = Real.sqrt 82 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_equality_l1410_141001


namespace NUMINAMATH_CALUDE_line_slope_l1410_141023

/-- The slope of a line given by the equation 3y - (1/2)x = 9 is 1/6 -/
theorem line_slope (x y : ℝ) : 3 * y - (1/2) * x = 9 → (y - 3) / x = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1410_141023


namespace NUMINAMATH_CALUDE_raisin_distribution_l1410_141013

/-- The number of raisins Bryce received -/
def bryce_raisins : ℕ := 16

/-- The number of raisins Carter received -/
def carter_raisins : ℕ := bryce_raisins - 8

theorem raisin_distribution :
  (bryce_raisins = carter_raisins + 8) ∧
  (carter_raisins = bryce_raisins / 2) :=
by sorry

#check raisin_distribution

end NUMINAMATH_CALUDE_raisin_distribution_l1410_141013


namespace NUMINAMATH_CALUDE_scientific_notation_3930_billion_l1410_141009

-- Define billion as 10^9
def billion : ℕ := 10^9

-- Theorem to prove the equality
theorem scientific_notation_3930_billion :
  (3930 : ℝ) * billion = 3.93 * (10 : ℝ)^12 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_3930_billion_l1410_141009


namespace NUMINAMATH_CALUDE_inequality_proof_l1410_141045

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≤ 1) : x^6 - y^6 + 2*y^3 < π/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1410_141045


namespace NUMINAMATH_CALUDE_mothers_age_problem_l1410_141075

theorem mothers_age_problem (x : ℕ) : 
  x + 3 * x = 40 → x = 10 := by sorry

end NUMINAMATH_CALUDE_mothers_age_problem_l1410_141075


namespace NUMINAMATH_CALUDE_right_triangle_area_l1410_141071

theorem right_triangle_area (a b c : ℝ) (h1 : a = 12) (h2 : c = 15) (h3 : a^2 + b^2 = c^2) : 
  (a * b) / 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1410_141071


namespace NUMINAMATH_CALUDE_concert_duration_is_80_minutes_l1410_141059

/-- Calculates the total duration of a concert given the number of songs, 
    duration of regular songs, duration of the special song, and intermission time. -/
def concertDuration (numSongs : ℕ) (regularSongDuration : ℕ) (specialSongDuration : ℕ) (intermissionTime : ℕ) : ℕ :=
  (numSongs - 1) * regularSongDuration + specialSongDuration + intermissionTime

/-- Proves that the concert duration is 80 minutes given the specified conditions. -/
theorem concert_duration_is_80_minutes :
  concertDuration 13 5 10 10 = 80 := by
  sorry

end NUMINAMATH_CALUDE_concert_duration_is_80_minutes_l1410_141059


namespace NUMINAMATH_CALUDE_cubic_inequality_l1410_141048

theorem cubic_inequality (x : ℝ) : x^3 - 9*x^2 + 36*x > -16*x ↔ x > 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1410_141048


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1410_141050

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 1 / y) ≥ 4 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 1 / y = 4 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1410_141050


namespace NUMINAMATH_CALUDE_star_operation_value_l1410_141037

def star_operation (a b : ℚ) : ℚ := 1 / a + 1 / b

theorem star_operation_value (a b : ℚ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a + b = 15) (h4 : a * b = 36) :
  star_operation a b = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_star_operation_value_l1410_141037


namespace NUMINAMATH_CALUDE_ones_digit_largest_power_of_three_dividing_factorial_ones_digit_largest_power_of_three_dividing_27_factorial_l1410_141078

theorem ones_digit_largest_power_of_three_dividing_factorial : ℕ → Prop :=
  fun n => 
    let factorial := Nat.factorial n
    let largest_power := Nat.log 3 factorial
    (3^largest_power % 10 = 3 ∧ n = 27)

-- The proof
theorem ones_digit_largest_power_of_three_dividing_27_factorial :
  ones_digit_largest_power_of_three_dividing_factorial 27 := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_largest_power_of_three_dividing_factorial_ones_digit_largest_power_of_three_dividing_27_factorial_l1410_141078


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l1410_141038

theorem fraction_zero_implies_x_equals_one (x : ℝ) : (x - 1) / (x + 2) = 0 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_one_l1410_141038


namespace NUMINAMATH_CALUDE_additional_bags_capacity_plane_capacity_proof_l1410_141051

/-- Calculates the number of additional maximum-weight bags an airplane can hold -/
theorem additional_bags_capacity 
  (num_people : ℕ) 
  (bags_per_person : ℕ) 
  (bag_weight : ℕ) 
  (plane_capacity : ℕ) : ℕ :=
  let total_bags := num_people * bags_per_person
  let total_weight := total_bags * bag_weight
  let remaining_capacity := plane_capacity - total_weight
  remaining_capacity / bag_weight

/-- Proves that given the specific conditions, the plane can hold 90 more bags -/
theorem plane_capacity_proof :
  additional_bags_capacity 6 5 50 6000 = 90 := by
  sorry

end NUMINAMATH_CALUDE_additional_bags_capacity_plane_capacity_proof_l1410_141051


namespace NUMINAMATH_CALUDE_common_roots_imply_a_b_values_l1410_141073

-- Define the two cubic polynomials
def p (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 11*x + 6
def q (b : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + 14*x + 8

-- Define a predicate for having two distinct common roots
def has_two_distinct_common_roots (a b : ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ s ∧ p a r = 0 ∧ p a s = 0 ∧ q b r = 0 ∧ q b s = 0

-- State the theorem
theorem common_roots_imply_a_b_values :
  ∀ a b : ℝ, has_two_distinct_common_roots a b → (a = 6 ∧ b = 7) :=
by sorry

end NUMINAMATH_CALUDE_common_roots_imply_a_b_values_l1410_141073


namespace NUMINAMATH_CALUDE_person_b_hit_six_shots_l1410_141092

/-- A shooting competition between two people -/
structure ShootingCompetition where
  hits_points : ℕ     -- Points gained for each hit
  miss_points : ℕ     -- Points deducted for each miss
  total_shots : ℕ     -- Total number of shots per person
  total_score : ℕ     -- Combined score of both persons
  score_diff  : ℕ     -- Score difference between person A and B

/-- The number of shots hit by person B in the competition -/
def person_b_hits (comp : ShootingCompetition) : ℕ := 
  sorry

/-- Theorem stating that person B hit 6 shots in the given competition -/
theorem person_b_hit_six_shots 
  (comp : ShootingCompetition) 
  (h1 : comp.hits_points = 20)
  (h2 : comp.miss_points = 12)
  (h3 : comp.total_shots = 10)
  (h4 : comp.total_score = 208)
  (h5 : comp.score_diff = 64) : 
  person_b_hits comp = 6 := by
  sorry

end NUMINAMATH_CALUDE_person_b_hit_six_shots_l1410_141092


namespace NUMINAMATH_CALUDE_difference_of_cubes_factorization_l1410_141063

theorem difference_of_cubes_factorization (t : ℝ) : t^3 - 125 = (t - 5) * (t^2 + 5*t + 25) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_cubes_factorization_l1410_141063


namespace NUMINAMATH_CALUDE_function_composition_equality_l1410_141017

theorem function_composition_equality 
  (m n p q : ℝ) 
  (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = m * x + n) 
  (hg : ∀ x, g x = p * x + q) : 
  (∀ x, f (g x) = g (f x)) ↔ m + q = n + p := by
sorry

end NUMINAMATH_CALUDE_function_composition_equality_l1410_141017


namespace NUMINAMATH_CALUDE_crayons_per_day_l1410_141083

def boxes_per_day : ℕ := 45
def crayons_per_box : ℕ := 7

theorem crayons_per_day : boxes_per_day * crayons_per_box = 315 := by
  sorry

end NUMINAMATH_CALUDE_crayons_per_day_l1410_141083


namespace NUMINAMATH_CALUDE_range_of_a_l1410_141085

/-- The function f(x) = a - x² -/
def f (a : ℝ) (x : ℝ) : ℝ := a - x^2

/-- The function g(x) = x + 2 -/
def g (x : ℝ) : ℝ := x + 2

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) : 
  (∃ x y : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f a x = -g y) → 
  -2 ≤ a ∧ a ≤ 0 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l1410_141085


namespace NUMINAMATH_CALUDE_salary_change_calculation_salary_decrease_percentage_l1410_141057

/-- Given an initial salary increase followed by a decrease, 
    calculate the percentage of the decrease. -/
theorem salary_change_calculation (initial_increase : ℝ) (net_increase : ℝ) : ℝ :=
  let final_factor := 1 + net_increase / 100
  let increase_factor := 1 + initial_increase / 100
  100 * (1 - final_factor / increase_factor)

/-- The percentage decrease in salary after an initial 10% increase,
    resulting in a net 1% increase, is approximately 8.18%. -/
theorem salary_decrease_percentage : 
  ∃ ε > 0, |salary_change_calculation 10 1 - 8.18| < ε :=
sorry

end NUMINAMATH_CALUDE_salary_change_calculation_salary_decrease_percentage_l1410_141057


namespace NUMINAMATH_CALUDE_max_area_rectangle_142_perimeter_l1410_141064

/-- Represents a rectangle with integer side lengths. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle. -/
def Rectangle.area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem stating the maximum area of a rectangle with a perimeter of 142 feet. -/
theorem max_area_rectangle_142_perimeter :
  ∃ (r : Rectangle), r.perimeter = 142 ∧
    ∀ (s : Rectangle), s.perimeter = 142 → s.area ≤ r.area ∧
    r.area = 1260 := by
  sorry


end NUMINAMATH_CALUDE_max_area_rectangle_142_perimeter_l1410_141064


namespace NUMINAMATH_CALUDE_eggs_per_person_l1410_141025

theorem eggs_per_person (mark_eggs : ℕ) (siblings : ℕ) : 
  mark_eggs = 2 * 12 → siblings = 3 → (mark_eggs / (siblings + 1) : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_person_l1410_141025


namespace NUMINAMATH_CALUDE_panteleimon_twos_count_l1410_141033

/-- Represents the grades of a student -/
structure Grades :=
  (fives : ℕ)
  (fours : ℕ)
  (threes : ℕ)
  (twos : ℕ)

/-- The total number of grades for each student -/
def total_grades : ℕ := 20

/-- Calculates the average grade -/
def average_grade (g : Grades) : ℚ :=
  (5 * g.fives + 4 * g.fours + 3 * g.threes + 2 * g.twos : ℚ) / total_grades

theorem panteleimon_twos_count 
  (p g : Grades) -- Panteleimon's and Gerasim's grades
  (h1 : p.fives + p.fours + p.threes + p.twos = total_grades)
  (h2 : g.fives + g.fours + g.threes + g.twos = total_grades)
  (h3 : p.fives = g.fours)
  (h4 : p.fours = g.threes)
  (h5 : p.threes = g.twos)
  (h6 : p.twos = g.fives)
  (h7 : average_grade p = average_grade g) :
  p.twos = 5 := by
  sorry

end NUMINAMATH_CALUDE_panteleimon_twos_count_l1410_141033


namespace NUMINAMATH_CALUDE_second_term_base_l1410_141043

theorem second_term_base (x y : ℕ) (base : ℝ) : 
  3^x * base^y = 19683 → x - y = 9 → x = 9 → base = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_base_l1410_141043


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1410_141016

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) :
  x₁^2 - 5*x₁ + 4 = 0 →
  x₂^2 - 5*x₂ + 4 = 0 →
  x₁ + x₂ = 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1410_141016


namespace NUMINAMATH_CALUDE_fifth_girl_siblings_l1410_141034

def number_set : List ℕ := [1, 6, 10, 4, 3, 11, 3, 10]

theorem fifth_girl_siblings (mean : ℚ) (h1 : mean = 57/10) 
  (h2 : (number_set.sum + x) / 9 = mean) : x = 3 :=
sorry

end NUMINAMATH_CALUDE_fifth_girl_siblings_l1410_141034


namespace NUMINAMATH_CALUDE_jeff_donation_proof_l1410_141035

/-- The percentage of pencils Jeff donated -/
def jeff_donation_percentage : ℝ := 0.3

theorem jeff_donation_proof :
  let jeff_initial : ℕ := 300
  let vicki_initial : ℕ := 2 * jeff_initial
  let vicki_donation : ℝ := 3/4 * vicki_initial
  let total_remaining : ℕ := 360
  (jeff_initial - jeff_initial * jeff_donation_percentage) +
    (vicki_initial - vicki_donation) = total_remaining :=
by sorry

end NUMINAMATH_CALUDE_jeff_donation_proof_l1410_141035


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1410_141065

-- Define the universal set U as the real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | -3 < x ∧ x < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | x < -1}

-- Theorem statement
theorem intersection_A_complement_B : A ∩ (U \ B) = {x : ℝ | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1410_141065


namespace NUMINAMATH_CALUDE_complement_of_angle_A_l1410_141066

-- Define the angle A
def angle_A : ℝ := 42

-- Define the complement of an angle
def complement (x : ℝ) : ℝ := 90 - x

-- Theorem statement
theorem complement_of_angle_A :
  complement angle_A = 48 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle_A_l1410_141066


namespace NUMINAMATH_CALUDE_square_root_squared_sqrt_2023_squared_l1410_141095

theorem square_root_squared (x : ℝ) (h : x > 0) : (Real.sqrt x)^2 = x := by sorry

theorem sqrt_2023_squared : (Real.sqrt 2023)^2 = 2023 := by
  apply square_root_squared
  norm_num

end NUMINAMATH_CALUDE_square_root_squared_sqrt_2023_squared_l1410_141095


namespace NUMINAMATH_CALUDE_number_of_positive_divisors_of_M_l1410_141010

def M : ℕ := 49^6 + 6*49^5 + 15*49^4 + 20*49^3 + 15*49^2 + 6*49 + 1

theorem number_of_positive_divisors_of_M : (Finset.filter (· ∣ M) (Finset.range (M + 1))).card = 91 := by
  sorry

end NUMINAMATH_CALUDE_number_of_positive_divisors_of_M_l1410_141010


namespace NUMINAMATH_CALUDE_proportion_problem_l1410_141098

theorem proportion_problem (x y : ℚ) 
  (h1 : (3/4 : ℚ) / x = 5 / 7)
  (h2 : y / 19 = 11 / 3) :
  x = 21/20 ∧ y = 209/3 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l1410_141098


namespace NUMINAMATH_CALUDE_right_triangle_altitude_reciprocal_square_sum_l1410_141077

/-- Given a right triangle with legs a and b, hypotenuse c, and altitude h drawn to the hypotenuse,
    prove that 1/h^2 = 1/a^2 + 1/b^2. -/
theorem right_triangle_altitude_reciprocal_square_sum 
  (a b c h : ℝ) 
  (h_positive : h > 0)
  (a_positive : a > 0)
  (b_positive : b > 0)
  (right_triangle : a^2 + b^2 = c^2)
  (altitude_formula : h * c = a * b) : 
  1 / h^2 = 1 / a^2 + 1 / b^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_reciprocal_square_sum_l1410_141077


namespace NUMINAMATH_CALUDE_complement_union_equals_set_l1410_141018

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_union_equals_set : 
  (U \ (M ∪ N)) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_set_l1410_141018


namespace NUMINAMATH_CALUDE_debbys_candy_l1410_141079

theorem debbys_candy (sister_candy : ℕ) (eaten_candy : ℕ) (remaining_candy : ℕ)
  (h1 : sister_candy = 42)
  (h2 : eaten_candy = 35)
  (h3 : remaining_candy = 39) :
  ∃ (debby_candy : ℕ), debby_candy + sister_candy - eaten_candy = remaining_candy ∧ debby_candy = 32 :=
by sorry

end NUMINAMATH_CALUDE_debbys_candy_l1410_141079


namespace NUMINAMATH_CALUDE_wall_length_proof_l1410_141040

/-- Given a square mirror and a rectangular wall, prove the wall's length. -/
theorem wall_length_proof (mirror_side : ℕ) (wall_width : ℕ) : 
  mirror_side = 54 →
  wall_width = 68 →
  (mirror_side * mirror_side : ℕ) * 2 = wall_width * (wall_width * 2 - wall_width % 2) →
  wall_width * 2 - wall_width % 2 = 86 :=
by
  sorry

end NUMINAMATH_CALUDE_wall_length_proof_l1410_141040


namespace NUMINAMATH_CALUDE_library_interval_proof_l1410_141024

def dance_interval : ℕ := 6
def karate_interval : ℕ := 12
def next_common_day : ℕ := 36

theorem library_interval_proof (x : ℕ) 
  (h1 : x > 0)
  (h2 : x ∣ next_common_day)
  (h3 : x ≠ dance_interval)
  (h4 : x ≠ karate_interval)
  (h5 : ∀ y : ℕ, y > 0 → y ∣ next_common_day → y ≠ dance_interval → y ≠ karate_interval → y ≤ x) :
  x = 18 := by sorry

end NUMINAMATH_CALUDE_library_interval_proof_l1410_141024


namespace NUMINAMATH_CALUDE_robert_interest_l1410_141091

/-- Calculates the total interest earned in a year given an inheritance amount,
    two interest rates, and the amount invested at the higher rate. -/
def total_interest (inheritance : ℝ) (rate1 : ℝ) (rate2 : ℝ) (amount_at_rate2 : ℝ) : ℝ :=
  let amount_at_rate1 := inheritance - amount_at_rate2
  let interest1 := amount_at_rate1 * rate1
  let interest2 := amount_at_rate2 * rate2
  interest1 + interest2

/-- Theorem stating that given Robert's inheritance and investment conditions,
    the total interest earned in a year is $227. -/
theorem robert_interest :
  total_interest 4000 0.05 0.065 1800 = 227 := by
  sorry

end NUMINAMATH_CALUDE_robert_interest_l1410_141091


namespace NUMINAMATH_CALUDE_product_of_tripled_numbers_l1410_141090

theorem product_of_tripled_numbers (x : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ + 1/x₁ = 3*x₁ ∧ x₂ + 1/x₂ = 3*x₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_product_of_tripled_numbers_l1410_141090


namespace NUMINAMATH_CALUDE_cats_after_sale_l1410_141021

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_after_sale (siamese : ℕ) (house : ℕ) (sold : ℕ) : 
  siamese = 15 → house = 49 → sold = 19 → siamese + house - sold = 45 := by
  sorry

end NUMINAMATH_CALUDE_cats_after_sale_l1410_141021


namespace NUMINAMATH_CALUDE_rhombus_in_quadrilateral_l1410_141002

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Represents a rhombus -/
structure Rhombus :=
  (X Y Z V : Point)

/-- Checks if two line segments are parallel -/
def are_parallel (P1 P2 Q1 Q2 : Point) : Prop :=
  (P2.x - P1.x) * (Q2.y - Q1.y) = (P2.y - P1.y) * (Q2.x - Q1.x)

/-- Checks if a point is inside a quadrilateral -/
def is_inside (P : Point) (Q : Quadrilateral) : Prop :=
  sorry -- Definition of a point being inside a quadrilateral

/-- Main theorem: There exists a rhombus within a given quadrilateral
    such that its sides are parallel to the quadrilateral's diagonals -/
theorem rhombus_in_quadrilateral (ABCD : Quadrilateral) :
  ∃ (XYZV : Rhombus),
    (is_inside XYZV.X ABCD) ∧ (is_inside XYZV.Y ABCD) ∧
    (is_inside XYZV.Z ABCD) ∧ (is_inside XYZV.V ABCD) ∧
    (are_parallel XYZV.X XYZV.Y ABCD.A ABCD.C) ∧
    (are_parallel XYZV.X XYZV.Z ABCD.B ABCD.D) ∧
    (are_parallel XYZV.Y XYZV.Z ABCD.A ABCD.C) ∧
    (are_parallel XYZV.V XYZV.Y ABCD.B ABCD.D) :=
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_rhombus_in_quadrilateral_l1410_141002


namespace NUMINAMATH_CALUDE_smallest_a_for_equation_l1410_141099

theorem smallest_a_for_equation : 
  ∀ a : ℕ, a ≥ 2 → 
  (∃ (p : ℕ) (b : ℕ), Prime p ∧ b ≥ 2 ∧ (a^p - a) / p = b^2) → 
  a ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_equation_l1410_141099


namespace NUMINAMATH_CALUDE_parallel_lines_from_perpendicular_to_parallel_planes_l1410_141004

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the parallel relation for lines
variable (parallelLines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicularLinePlane : Line → Plane → Prop)

-- Define the property of two lines being non-coincident
variable (nonCoincident : Line → Line → Prop)

-- Theorem statement
theorem parallel_lines_from_perpendicular_to_parallel_planes 
  (α β : Plane) (a b : Line) :
  parallelPlanes α β →
  nonCoincident a b →
  perpendicularLinePlane a α →
  perpendicularLinePlane b β →
  parallelLines a b :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_from_perpendicular_to_parallel_planes_l1410_141004


namespace NUMINAMATH_CALUDE_volume_ratio_of_cubes_l1410_141000

/-- The ratio of volumes of two cubes -/
theorem volume_ratio_of_cubes (inches_per_foot : ℚ) (edge_length_small : ℚ) (edge_length_large : ℚ) :
  inches_per_foot = 12 →
  edge_length_small = 4 →
  edge_length_large = 2 * inches_per_foot →
  (edge_length_small ^ 3) / (edge_length_large ^ 3) = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_of_cubes_l1410_141000


namespace NUMINAMATH_CALUDE_negation_equivalence_l1410_141052

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ > 3 - x₀) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≤ 3 - x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1410_141052


namespace NUMINAMATH_CALUDE_triangle_existence_l1410_141072

/-- Represents a triangle with side lengths a, b, c and angles α, β, γ. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Theorem stating the existence of a triangle satisfying given conditions. -/
theorem triangle_existence (d β γ : ℝ) : 
  ∃ (t : Triangle), 
    t.b + t.c - t.a = d ∧ 
    t.β = β ∧ 
    t.γ = γ ∧
    t.α + t.β + t.γ = π :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l1410_141072


namespace NUMINAMATH_CALUDE_complex_abs_sum_l1410_141060

theorem complex_abs_sum : Complex.abs (2 - 4 * Complex.I) + Complex.abs (2 + 4 * Complex.I) = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_sum_l1410_141060


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1410_141014

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 1711 ∧ ∀ x, 8 * x^2 - 24 * x + 1729 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1410_141014


namespace NUMINAMATH_CALUDE_gcd_459_357_l1410_141012

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l1410_141012


namespace NUMINAMATH_CALUDE_total_sheets_l1410_141069

-- Define the number of brown and yellow sheets
def brown_sheets : ℕ := 28
def yellow_sheets : ℕ := 27

-- Theorem to prove
theorem total_sheets : brown_sheets + yellow_sheets = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_sheets_l1410_141069


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1410_141026

/-- A quadratic function is a polynomial function of degree 2. -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The main theorem: if f is quadratic and satisfies the given condition,
    then it has the specific form x^2 - 4x + 4. -/
theorem quadratic_function_theorem (f : ℝ → ℝ) 
    (h1 : IsQuadratic f)
    (h2 : ∀ x, f x + f (x + 1) = 2 * x^2 - 6 * x + 5) :
    ∀ x, f x = x^2 - 4 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1410_141026


namespace NUMINAMATH_CALUDE_eric_quarters_count_l1410_141041

/-- The number of dimes Cindy tosses -/
def cindy_dimes : ℕ := 5

/-- The number of nickels Garrick throws -/
def garrick_nickels : ℕ := 8

/-- The number of pennies Ivy drops -/
def ivy_pennies : ℕ := 60

/-- The total amount in the pond in cents -/
def total_cents : ℕ := 200

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of quarters Eric flipped into the pond -/
def eric_quarters : ℕ := 2

theorem eric_quarters_count :
  eric_quarters * quarter_value = 
    total_cents - (cindy_dimes * dime_value + garrick_nickels * nickel_value + ivy_pennies * penny_value) :=
by sorry

end NUMINAMATH_CALUDE_eric_quarters_count_l1410_141041


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1410_141046

theorem log_sum_equals_two : 2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1410_141046


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1410_141031

theorem binomial_coefficient_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_19_l1410_141031


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_12_with_digit_sum_24_l1410_141020

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_12_with_digit_sum_24 :
  ∀ n : ℕ, is_three_digit n → n.mod 12 = 0 → digit_sum n = 24 → n ≤ 996 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_12_with_digit_sum_24_l1410_141020


namespace NUMINAMATH_CALUDE_chocolate_candies_cost_l1410_141097

theorem chocolate_candies_cost (box_size : ℕ) (box_cost : ℚ) (total_candies : ℕ) : 
  box_size = 30 → box_cost = 7.5 → total_candies = 450 → 
  (total_candies / box_size : ℚ) * box_cost = 112.5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_candies_cost_l1410_141097


namespace NUMINAMATH_CALUDE_difference_quotient_of_f_l1410_141080

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 1

-- State the theorem
theorem difference_quotient_of_f (Δx : ℝ) :
  let y1 := f 1
  let y2 := f (1 + Δx)
  (y2 - y1) / Δx = 4 + 2 * Δx := by
  sorry

end NUMINAMATH_CALUDE_difference_quotient_of_f_l1410_141080


namespace NUMINAMATH_CALUDE_weather_conditions_on_july_15_l1410_141044

/-- Represents the weather conditions at the beach --/
structure WeatherCondition where
  temperature : ℝ
  sunny : Bool
  windSpeed : ℝ

/-- Predicate to determine if the beach is crowded based on weather conditions --/
def isCrowded (w : WeatherCondition) : Prop :=
  w.temperature ≥ 85 ∧ w.sunny ∧ w.windSpeed < 10

/-- Theorem: Given that the beach is not crowded on July 15, prove that the weather conditions
    must satisfy: temperature < 85°F or not sunny or wind speed ≥ 10 mph --/
theorem weather_conditions_on_july_15 (w : WeatherCondition) 
  (h : ¬isCrowded w) : 
  w.temperature < 85 ∨ ¬w.sunny ∨ w.windSpeed ≥ 10 := by
  sorry


end NUMINAMATH_CALUDE_weather_conditions_on_july_15_l1410_141044


namespace NUMINAMATH_CALUDE_workshop_workers_l1410_141070

theorem workshop_workers (total_average : ℕ) (tech_count : ℕ) (tech_average : ℕ) (nontech_average : ℕ) :
  total_average = 8000 →
  tech_count = 10 →
  tech_average = 12000 →
  nontech_average = 6000 →
  ∃ (total_workers : ℕ), 
    total_workers * total_average = tech_count * tech_average + (total_workers - tech_count) * nontech_average ∧
    total_workers = 30 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l1410_141070


namespace NUMINAMATH_CALUDE_A_share_is_175_l1410_141005

/-- Calculates the share of profit for partner A in a business partnership --/
def calculate_share_A (initial_A initial_B change_A change_B total_profit : ℚ) : ℚ :=
  let investment_months_A := initial_A * 8 + (initial_A + change_A) * 4
  let investment_months_B := initial_B * 8 + (initial_B + change_B) * 4
  let total_investment_months := investment_months_A + investment_months_B
  (investment_months_A / total_investment_months) * total_profit

/-- Theorem stating that A's share of the profit is 175 given the specified conditions --/
theorem A_share_is_175 :
  calculate_share_A 2000 4000 (-1000) 1000 630 = 175 := by
  sorry

end NUMINAMATH_CALUDE_A_share_is_175_l1410_141005


namespace NUMINAMATH_CALUDE_fish_dog_lifespan_difference_l1410_141096

/-- The difference between a fish's lifespan and a dog's lifespan is 2 years -/
theorem fish_dog_lifespan_difference :
  let hamster_lifespan : ℝ := 2.5
  let dog_lifespan : ℝ := 4 * hamster_lifespan
  let fish_lifespan : ℝ := 12
  fish_lifespan - dog_lifespan = 2 := by
  sorry

end NUMINAMATH_CALUDE_fish_dog_lifespan_difference_l1410_141096


namespace NUMINAMATH_CALUDE_fraction_power_division_l1410_141086

theorem fraction_power_division :
  (1 / 3 : ℚ)^4 / (1 / 5 : ℚ) = 5 / 81 := by sorry

end NUMINAMATH_CALUDE_fraction_power_division_l1410_141086


namespace NUMINAMATH_CALUDE_induction_base_case_not_always_one_l1410_141022

/-- In mathematical induction, the base case is not always n = 1. -/
theorem induction_base_case_not_always_one : ∃ (P : ℕ → Prop) (n₀ : ℕ), 
  n₀ ≠ 1 ∧ (∀ n ≥ n₀, P n → P (n + 1)) → (∀ n ≥ n₀, P n) :=
sorry

end NUMINAMATH_CALUDE_induction_base_case_not_always_one_l1410_141022


namespace NUMINAMATH_CALUDE_inequality_proof_l1410_141032

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_condition : |Real.sqrt (a * d) - Real.sqrt (b * c)| ≤ 1) : 
  (a * e + b / e) * (c * e + d / e) ≥ 
  (a^2 * f^2 - b^2 / f^2) * (d^2 / f^2 - c^2 * f^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1410_141032


namespace NUMINAMATH_CALUDE_circle_radius_three_inches_l1410_141027

theorem circle_radius_three_inches 
  (r : ℝ) 
  (h : r > 0) 
  (h_eq : 3 * (2 * Real.pi * r) = 2 * (Real.pi * r^2)) : 
  r = 3 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_three_inches_l1410_141027


namespace NUMINAMATH_CALUDE_partitionWays_10_l1410_141054

/-- The number of ways to partition n ordered elements into 1 to n non-empty subsets,
    where the elements within each subset are contiguous. -/
def partitionWays (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => Nat.choose (n - 1) k)

/-- Theorem stating that for 10 elements, the number of partition ways is 512. -/
theorem partitionWays_10 : partitionWays 10 = 512 := by
  sorry

end NUMINAMATH_CALUDE_partitionWays_10_l1410_141054


namespace NUMINAMATH_CALUDE_smallest_with_14_divisors_l1410_141036

/-- Count the number of positive divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := sorry

/-- Check if a natural number has exactly 14 positive divisors -/
def has_14_divisors (n : ℕ) : Prop :=
  count_divisors n = 14

/-- The theorem stating that 192 is the smallest positive integer with exactly 14 positive divisors -/
theorem smallest_with_14_divisors :
  (∀ m : ℕ, m > 0 → m < 192 → ¬(has_14_divisors m)) ∧ has_14_divisors 192 := by sorry

end NUMINAMATH_CALUDE_smallest_with_14_divisors_l1410_141036


namespace NUMINAMATH_CALUDE_average_salary_feb_to_may_l1410_141029

theorem average_salary_feb_to_may (
  avg_jan_to_apr : ℝ) 
  (salary_may : ℝ)
  (salary_jan : ℝ)
  (h1 : avg_jan_to_apr = 8000)
  (h2 : salary_may = 6500)
  (h3 : salary_jan = 4700) :
  (4 * avg_jan_to_apr - salary_jan + salary_may) / 4 = 8450 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_feb_to_may_l1410_141029


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1410_141094

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 5) = 10 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1410_141094


namespace NUMINAMATH_CALUDE_min_faces_two_dice_l1410_141030

structure Dice where
  faces : ℕ
  min_faces : ℕ
  distinct_numbering : faces ≥ min_faces

def probability_sum (a b : Dice) (sum : ℕ) : ℚ :=
  (Finset.filter (fun (x, y) => x + y = sum) (Finset.product (Finset.range a.faces) (Finset.range b.faces))).card /
  (a.faces * b.faces : ℚ)

theorem min_faces_two_dice (a b : Dice) : 
  a.min_faces = 7 → 
  b.min_faces = 5 → 
  probability_sum a b 13 = 2 * probability_sum a b 8 →
  probability_sum a b 16 = 1/20 →
  a.faces + b.faces ≥ 24 ∧ 
  ∀ (a' b' : Dice), a'.faces + b'.faces < 24 → 
    (a'.min_faces = 7 ∧ b'.min_faces = 5 ∧ 
     probability_sum a' b' 13 = 2 * probability_sum a' b' 8 ∧
     probability_sum a' b' 16 = 1/20) → False :=
by sorry

end NUMINAMATH_CALUDE_min_faces_two_dice_l1410_141030


namespace NUMINAMATH_CALUDE_probability_two_red_crayons_l1410_141015

/-- The probability of selecting 2 red crayons from a jar containing 3 red, 2 blue, and 1 green crayon -/
theorem probability_two_red_crayons (total : ℕ) (red : ℕ) (blue : ℕ) (green : ℕ) :
  total = red + blue + green →
  total = 6 →
  red = 3 →
  blue = 2 →
  green = 1 →
  (Nat.choose red 2 : ℚ) / (Nat.choose total 2) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_crayons_l1410_141015


namespace NUMINAMATH_CALUDE_line_through_circle_center_l1410_141087

/-- A line intersecting a circle -/
structure LineIntersectingCircle where
  /-- The slope of the line -/
  k : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- The line intersects the circle at two points -/
  intersects : True
  /-- The distance between the intersection points -/
  chord_length : ℝ

/-- The theorem stating the value of k for a specific configuration -/
theorem line_through_circle_center (config : LineIntersectingCircle)
    (h1 : config.b = 2)
    (h2 : config.center = (1, 1))
    (h3 : config.radius = Real.sqrt 2)
    (h4 : config.chord_length = 2 * Real.sqrt 2) :
    config.k = -1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l1410_141087


namespace NUMINAMATH_CALUDE_pages_left_to_write_l1410_141093

/-- Calculates the remaining pages to write given the daily page counts and total book length -/
theorem pages_left_to_write (total_pages day1 day2 day3 day4 day5 : ℝ) : 
  total_pages = 750 →
  day1 = 30 →
  day2 = 1.5 * day1 →
  day3 = 0.5 * day2 →
  day4 = 2.5 * day3 →
  day5 = 15 →
  total_pages - (day1 + day2 + day3 + day4 + day5) = 581.25 := by
  sorry

end NUMINAMATH_CALUDE_pages_left_to_write_l1410_141093


namespace NUMINAMATH_CALUDE_language_class_selection_probability_l1410_141006

/-- The probability of selecting two students from different language classes -/
theorem language_class_selection_probability
  (total_students : ℕ)
  (french_students : ℕ)
  (spanish_students : ℕ)
  (no_language_students : ℕ)
  (h1 : total_students = 30)
  (h2 : french_students = 22)
  (h3 : spanish_students = 24)
  (h4 : no_language_students = 2)
  (h5 : french_students + spanish_students - (total_students - no_language_students) + no_language_students = total_students) :
  let both_classes := french_students + spanish_students - (total_students - no_language_students)
  let only_french := french_students - both_classes
  let only_spanish := spanish_students - both_classes
  let total_combinations := total_students.choose 2
  let undesirable_outcomes := (only_french.choose 2) + (only_spanish.choose 2)
  (1 : ℚ) - (undesirable_outcomes : ℚ) / (total_combinations : ℚ) = 14 / 15 :=
by sorry

end NUMINAMATH_CALUDE_language_class_selection_probability_l1410_141006


namespace NUMINAMATH_CALUDE_modulo_residue_problem_l1410_141067

theorem modulo_residue_problem : (325 + 3 * 66 + 8 * 187 + 6 * 23) % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulo_residue_problem_l1410_141067


namespace NUMINAMATH_CALUDE_parallel_line_slope_parallel_line_y_intercept_exists_l1410_141028

/-- Given a line parallel to 3x - 6y = 12, prove its slope is 1/2 --/
theorem parallel_line_slope (a b c : ℝ) (h : ∃ k : ℝ, a * x + b * y = c ∧ k ≠ 0 ∧ 3 * (a / b) = -1 / 2) :
  a / b = 1 / 2 := by sorry

/-- The y-intercept of a line parallel to 3x - 6y = 12 can be any real number --/
theorem parallel_line_y_intercept_exists : ∀ k : ℝ, ∃ (a b c : ℝ), a * x + b * y = c ∧ a / b = 1 / 2 ∧ c / b = k := by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_parallel_line_y_intercept_exists_l1410_141028


namespace NUMINAMATH_CALUDE_solve_inequality_when_a_is_5_range_of_a_for_always_positive_l1410_141049

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 6

-- Statement for part I
theorem solve_inequality_when_a_is_5 :
  {x : ℝ | f 5 x < 0} = {x : ℝ | -3 < x ∧ x < -2} := by sorry

-- Statement for part II
theorem range_of_a_for_always_positive :
  ∀ a : ℝ, (∀ x : ℝ, f a x > 0) ↔ a ∈ Set.Ioo (-2 * Real.sqrt 6) (2 * Real.sqrt 6) := by sorry

end NUMINAMATH_CALUDE_solve_inequality_when_a_is_5_range_of_a_for_always_positive_l1410_141049


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1410_141011

def A : Set ℝ := {-1, 0, 3, 5}
def B : Set ℝ := {x : ℝ | x - 2 > 0}

theorem intersection_of_A_and_B : A ∩ B = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1410_141011


namespace NUMINAMATH_CALUDE_functional_equation_problem_l1410_141058

theorem functional_equation_problem (f : ℝ → ℝ) 
  (h : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3)
  (h1 : f 1 = 1)
  (h4 : f 4 = 7) :
  f 2022 = 4043 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l1410_141058


namespace NUMINAMATH_CALUDE_triangle_with_squares_sum_l1410_141008

/-- A right-angled triangle with two inscribed squares -/
structure TriangleWithSquares where
  -- The side lengths of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The areas of the inscribed squares
  area_s1 : ℝ
  area_s2 : ℝ
  -- Conditions
  right_angle : c^2 = a^2 + b^2
  inscribed_square1 : area_s1 = 40 * b + 1
  inscribed_square2 : area_s2 = 40 * b
  sum_sides : c = a + b

theorem triangle_with_squares_sum (t : TriangleWithSquares) : t.c = 462 := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_squares_sum_l1410_141008


namespace NUMINAMATH_CALUDE_village_population_l1410_141068

theorem village_population (initial_population : ℕ) 
  (death_rate : ℚ) (leaving_rate : ℚ) : 
  initial_population = 4500 →
  death_rate = 1/10 →
  leaving_rate = 1/5 →
  (initial_population - initial_population * death_rate) * (1 - leaving_rate) = 3240 :=
by sorry

end NUMINAMATH_CALUDE_village_population_l1410_141068


namespace NUMINAMATH_CALUDE_square_side_length_l1410_141076

/-- Given a ribbon of length 78 cm used to make a triangle and a square,
    with the triangle having a perimeter of 46 cm,
    prove that the length of one side of the square is 8 cm. -/
theorem square_side_length (total_ribbon : ℝ) (triangle_perimeter : ℝ) (square_side : ℝ) :
  total_ribbon = 78 ∧ 
  triangle_perimeter = 46 ∧ 
  square_side * 4 = total_ribbon - triangle_perimeter → 
  square_side = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1410_141076


namespace NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l1410_141089

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fourth_term_of_arithmetic_sequence 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_first : a 0 = 25) 
  (h_last : a 5 = 57) : 
  a 3 = 41 := by
sorry

end NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l1410_141089


namespace NUMINAMATH_CALUDE_solution_set_theorem_range_of_b_theorem_l1410_141007

-- Define the real number a
variable (a : ℝ)

-- Define the condition that the solution set of (1-a)x^2 - 4x + 6 > 0 is (-3, 1)
def solution_set_condition : Prop :=
  ∀ x : ℝ, ((1 - a) * x^2 - 4 * x + 6 > 0) ↔ (-3 < x ∧ x < 1)

-- Theorem for the first question
theorem solution_set_theorem (h : solution_set_condition a) :
  ∀ x : ℝ, (2 * x^2 + (2 - a) * x - a > 0) ↔ (x < -1 ∨ x > 3/2) :=
sorry

-- Theorem for the second question
theorem range_of_b_theorem (h : solution_set_condition a) :
  ∀ b : ℝ, (∀ x : ℝ, a * x^2 + b * x + 3 ≥ 0) ↔ (-6 ≤ b ∧ b ≤ 6) :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_range_of_b_theorem_l1410_141007


namespace NUMINAMATH_CALUDE_total_profit_is_35000_l1410_141084

/-- Represents the subscription amounts and profit for a business venture -/
structure BusinessVenture where
  total_subscription : ℕ
  a_extra : ℕ
  b_extra : ℕ
  a_profit : ℕ

/-- Calculates the total profit given a BusinessVenture -/
def calculate_total_profit (bv : BusinessVenture) : ℕ :=
  sorry

/-- Theorem stating that for the given business venture, the total profit is 35000 -/
theorem total_profit_is_35000 : 
  let bv : BusinessVenture := {
    total_subscription := 50000,
    a_extra := 4000,
    b_extra := 5000,
    a_profit := 14700
  }
  calculate_total_profit bv = 35000 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_35000_l1410_141084


namespace NUMINAMATH_CALUDE_africa_passenger_fraction_l1410_141053

theorem africa_passenger_fraction :
  let total_passengers : ℕ := 108
  let north_america_fraction : ℚ := 1 / 12
  let europe_fraction : ℚ := 1 / 4
  let asia_fraction : ℚ := 1 / 6
  let other_continents : ℕ := 42
  let africa_fraction : ℚ := 1 - north_america_fraction - europe_fraction - asia_fraction - (other_continents : ℚ) / total_passengers
  africa_fraction = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_africa_passenger_fraction_l1410_141053


namespace NUMINAMATH_CALUDE_logical_judgment_structures_l1410_141039

-- Define the basic structures of algorithms
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop

-- Define a property for structures that require logical judgment
def RequiresLogicalJudgment (s : AlgorithmStructure) : Prop :=
  match s with
  | AlgorithmStructure.Conditional => True
  | AlgorithmStructure.Loop => True
  | _ => False

-- Theorem statement
theorem logical_judgment_structures :
  ∀ s : AlgorithmStructure,
    RequiresLogicalJudgment s ↔ (s = AlgorithmStructure.Conditional ∨ s = AlgorithmStructure.Loop) :=
by sorry

end NUMINAMATH_CALUDE_logical_judgment_structures_l1410_141039


namespace NUMINAMATH_CALUDE_number_of_carnations_solve_carnation_problem_l1410_141088

/-- Proves the number of carnations given the problem conditions --/
theorem number_of_carnations : ℕ → Prop :=
  fun c =>
    let vase_capacity : ℕ := 9
    let num_roses : ℕ := 23
    let num_vases : ℕ := 3
    (c + num_roses = num_vases * vase_capacity) → c = 4

/-- The theorem statement --/
theorem solve_carnation_problem : number_of_carnations 4 := by
  sorry

end NUMINAMATH_CALUDE_number_of_carnations_solve_carnation_problem_l1410_141088


namespace NUMINAMATH_CALUDE_last_score_entered_last_score_is_95_l1410_141081

def scores : List ℕ := [75, 81, 85, 87, 95]

def is_integer_average (subset : List ℕ) : Prop :=
  ∃ n : ℕ, n * subset.length = subset.sum

theorem last_score_entered (last : ℕ) : Prop :=
  last ∈ scores ∧
  ∀ subset : List ℕ, subset ⊆ scores → last ∈ subset →
    is_integer_average subset

theorem last_score_is_95 : 
  ∃ last : ℕ, last_score_entered last ∧ last = 95 := by
  sorry

end NUMINAMATH_CALUDE_last_score_entered_last_score_is_95_l1410_141081


namespace NUMINAMATH_CALUDE_vivian_daily_songs_l1410_141047

/-- The number of songs Vivian plays each day -/
def vivian_songs : ℕ := sorry

/-- The number of songs Clara plays each day -/
def clara_songs : ℕ := sorry

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The number of weekend days in June -/
def weekend_days : ℕ := 8

/-- The total number of songs both listened to in June -/
def total_songs : ℕ := 396

theorem vivian_daily_songs :
  (vivian_songs = 10) ∧
  (clara_songs = vivian_songs - 2) ∧
  (total_songs = (june_days - weekend_days) * (vivian_songs + clara_songs)) := by
  sorry

end NUMINAMATH_CALUDE_vivian_daily_songs_l1410_141047


namespace NUMINAMATH_CALUDE_triangle_side_length_l1410_141055

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin (2 * x) + 2, Real.cos x)
noncomputable def n (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem triangle_side_length 
  (A B C : ℝ) 
  (h1 : f A = 4) 
  (h2 : 0 < A ∧ A < π) 
  (h3 : Real.sin A * 1 / 2 = Real.sqrt 3 / 2) :
  ∃ (a : ℝ), a^2 = 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1410_141055


namespace NUMINAMATH_CALUDE_slope_height_calculation_l1410_141042

theorem slope_height_calculation (slope_ratio : Real) (distance : Real) (height : Real) : 
  slope_ratio = 1 / 2.4 →
  distance = 130 →
  height ^ 2 + (height * 2.4) ^ 2 = distance ^ 2 →
  height = 50 := by
sorry

end NUMINAMATH_CALUDE_slope_height_calculation_l1410_141042


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_n_99996_satisfies_condition_n_99996_is_largest_l1410_141061

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 100000 ∧ 
  (9 * (n - 3)^6 - 3 * n^3 + 21 * n - 42) % 7 = 0 →
  n ≤ 99996 :=
by sorry

theorem n_99996_satisfies_condition : 
  (9 * (99996 - 3)^6 - 3 * 99996^3 + 21 * 99996 - 42) % 7 = 0 :=
by sorry

theorem n_99996_is_largest :
  ∀ m : ℕ, m < 100000 ∧ 
  (9 * (m - 3)^6 - 3 * m^3 + 21 * m - 42) % 7 = 0 →
  m ≤ 99996 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_n_99996_satisfies_condition_n_99996_is_largest_l1410_141061


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1410_141056

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (θ : ℝ) (h_d : d = 12 * Real.sqrt 2) (h_θ : θ = π / 4) :
  let r := d / 4
  (4 / 3) * π * r^3 = 288 * π := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1410_141056


namespace NUMINAMATH_CALUDE_am_gm_difference_bound_l1410_141062

theorem am_gm_difference_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  (x + y) / 2 - Real.sqrt (x * y) ≥ (x - y)^2 / (8 * x) := by
  sorry

end NUMINAMATH_CALUDE_am_gm_difference_bound_l1410_141062


namespace NUMINAMATH_CALUDE_blue_ball_probability_l1410_141003

noncomputable def bag_probabilities (p_red p_yellow p_blue : ℝ) : Prop :=
  p_red + p_yellow + p_blue = 1 ∧ 0 ≤ p_red ∧ 0 ≤ p_yellow ∧ 0 ≤ p_blue

theorem blue_ball_probability :
  ∀ (p_red p_yellow p_blue : ℝ),
    bag_probabilities p_red p_yellow p_blue →
    p_red = 0.48 →
    p_yellow = 0.35 →
    p_blue = 0.17 :=
by sorry

end NUMINAMATH_CALUDE_blue_ball_probability_l1410_141003


namespace NUMINAMATH_CALUDE_bullseye_mean_hits_l1410_141074

/-- The mean number of hits in a series of independent Bernoulli trials -/
def meanHits (p : ℝ) (n : ℕ) : ℝ := n * p

/-- The probability of hitting the bullseye -/
def bullseyeProbability : ℝ := 0.9

/-- The number of consecutive shots -/
def numShots : ℕ := 10

theorem bullseye_mean_hits :
  meanHits bullseyeProbability numShots = 9 := by
  sorry

end NUMINAMATH_CALUDE_bullseye_mean_hits_l1410_141074
