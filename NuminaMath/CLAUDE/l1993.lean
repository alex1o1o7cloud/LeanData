import Mathlib

namespace abc_sum_sqrt_l1993_199362

theorem abc_sum_sqrt (a b c : ℝ) 
  (h1 : b + c = 17)
  (h2 : c + a = 20)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 168 := by
sorry

end abc_sum_sqrt_l1993_199362


namespace parallel_condition_l1993_199342

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The vector a as a function of k -/
def a (k : ℝ) : ℝ × ℝ := (k^2, k + 1)

/-- The vector b as a function of k -/
def b (k : ℝ) : ℝ × ℝ := (k, 4)

/-- Theorem stating the conditions for parallelism of vectors a and b -/
theorem parallel_condition (k : ℝ) : 
  are_parallel (a k) (b k) ↔ k = 0 ∨ k = 1/3 := by sorry

end parallel_condition_l1993_199342


namespace complex_eighth_power_sum_l1993_199357

theorem complex_eighth_power_sum : (((1 : ℂ) + Complex.I * Real.sqrt 3) / 2) ^ 8 + 
  (((1 : ℂ) - Complex.I * Real.sqrt 3) / 2) ^ 8 = -1 := by
  sorry

end complex_eighth_power_sum_l1993_199357


namespace wrapping_paper_area_l1993_199377

/-- The area of wrapping paper required to wrap a box on a pedestal -/
theorem wrapping_paper_area (w h p : ℝ) (hw : w > 0) (hh : h > 0) (hp : p > 0) :
  let paper_area := 4 * w * (p + h)
  paper_area = 4 * w * (p + h) :=
by sorry

end wrapping_paper_area_l1993_199377


namespace rectangle_dg_length_l1993_199382

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

theorem rectangle_dg_length :
  ∀ (r1 r2 r3 : Rectangle),
  area r1 = area r2 ∧ area r2 = area r3 ∧   -- Equal areas
  r1.width = 23 ∧                           -- BC = 23
  r2.width = r1.height ∧                    -- DE = AB
  r3.width = r1.height - r2.height ∧        -- CE = AB - DE
  r3.height = r1.width →                    -- CH = BC
  r2.height = 552                           -- DG = 552
  := by sorry

end rectangle_dg_length_l1993_199382


namespace right_triangle_area_l1993_199371

theorem right_triangle_area (h : ℝ) (h_positive : h > 0) :
  let a := h * Real.sqrt 2
  let b := h * Real.sqrt 2
  let c := 2 * h * Real.sqrt 2
  h = 4 →
  (1 / 2 : ℝ) * c * h = 16 * Real.sqrt 2 := by sorry

end right_triangle_area_l1993_199371


namespace coefficient_x_squared_is_135_l1993_199322

/-- The coefficient of x^2 in the expansion of (3x-1)^6 -/
def coefficient_x_squared : ℕ :=
  let n : ℕ := 6
  let k : ℕ := 4
  let binomial_coefficient : ℕ := n.choose k
  let power_of_three : ℕ := 3^(n - k)
  binomial_coefficient * power_of_three

/-- Theorem stating that the coefficient of x^2 in (3x-1)^6 is 135 -/
theorem coefficient_x_squared_is_135 : coefficient_x_squared = 135 := by
  sorry

#eval coefficient_x_squared

end coefficient_x_squared_is_135_l1993_199322


namespace longest_altitudes_sum_is_31_l1993_199350

/-- A right triangle with sides 7, 24, and 25 -/
structure RightTriangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (side_a : a = 7)
  (side_b : b = 24)
  (side_c : c = 25)

/-- The sum of the lengths of the two longest altitudes in the right triangle -/
def longest_altitudes_sum (t : RightTriangle) : ℝ :=
  t.a + t.b

/-- Theorem: The sum of the lengths of the two longest altitudes in the given right triangle is 31 -/
theorem longest_altitudes_sum_is_31 (t : RightTriangle) :
  longest_altitudes_sum t = 31 := by
  sorry

end longest_altitudes_sum_is_31_l1993_199350


namespace y_intercept_of_line_l1993_199334

/-- The y-intercept of the line 2x - 3y = 6 is -2 -/
theorem y_intercept_of_line (x y : ℝ) : 2 * x - 3 * y = 6 → x = 0 → y = -2 := by
  sorry

end y_intercept_of_line_l1993_199334


namespace solve_for_a_l1993_199338

theorem solve_for_a : ∃ a : ℝ, (3 * 2 + 2 * a = 0) ∧ (a = -3) := by
  sorry

end solve_for_a_l1993_199338


namespace arithmetic_sequence_k_equals_4_l1993_199323

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_k_equals_4
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 1)
  (h_diff : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d)
  (h_k : ∃ k : ℕ, a k = 7) :
  ∃ k : ℕ, a k = 7 ∧ k = 4 :=
sorry

end arithmetic_sequence_k_equals_4_l1993_199323


namespace no_real_solution_system_l1993_199364

theorem no_real_solution_system :
  ¬∃ (x y z : ℝ), (x + y - 2 - 4*x*y = 0) ∧ 
                  (y + z - 2 - 4*y*z = 0) ∧ 
                  (z + x - 2 - 4*z*x = 0) :=
by sorry

end no_real_solution_system_l1993_199364


namespace trigonometric_identities_l1993_199397

theorem trigonometric_identities :
  (∃ (tan25 tan35 : ℝ),
    tan25 = Real.tan (25 * π / 180) ∧
    tan35 = Real.tan (35 * π / 180) ∧
    tan25 + tan35 + Real.sqrt 3 * tan25 * tan35 = Real.sqrt 3) ∧
  (Real.sin (10 * π / 180))⁻¹ - (Real.sqrt 3) * (Real.cos (10 * π / 180))⁻¹ = 4 := by
  sorry

end trigonometric_identities_l1993_199397


namespace test_ways_count_l1993_199345

/-- Represents the number of genuine items in the test. -/
def genuine_items : ℕ := 5

/-- Represents the number of defective items in the test. -/
def defective_items : ℕ := 4

/-- Represents the total number of tests conducted. -/
def total_tests : ℕ := 5

/-- Calculates the number of ways to conduct the test under the given conditions. -/
def test_ways : ℕ := sorry

/-- Theorem stating that the number of ways to conduct the test is 480. -/
theorem test_ways_count : test_ways = 480 := by sorry

end test_ways_count_l1993_199345


namespace inequality_proof_l1993_199326

theorem inequality_proof (x y z w : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0)
  (hxy : x + y ≠ 0) (hzw : z + w ≠ 0) (hxyzw : x * y + z * w ≥ 0) :
  ((x + y) / (z + w) + (z + w) / (x + y))⁻¹ + 1 / 2 ≥ 
  (x / z + z / x)⁻¹ + (y / w + w / y)⁻¹ := by
  sorry

end inequality_proof_l1993_199326


namespace division_by_fraction_fifteen_divided_by_two_thirds_result_is_twentytwo_point_five_l1993_199337

theorem division_by_fraction (a b c : ℚ) (hb : b ≠ 0) (hc : c ≠ 0) :
  a / (b / c) = (a * c) / b :=
by sorry

theorem fifteen_divided_by_two_thirds :
  15 / (2 / 3) = 45 / 2 :=
by sorry

theorem result_is_twentytwo_point_five :
  15 / (2 / 3) = 22.5 :=
by sorry

end division_by_fraction_fifteen_divided_by_two_thirds_result_is_twentytwo_point_five_l1993_199337


namespace fifth_term_geometric_progression_l1993_199385

theorem fifth_term_geometric_progression :
  let x : ℝ := -1 + Real.sqrt 5
  let r : ℝ := (1 + Real.sqrt 5) / (-1 + Real.sqrt 5)
  let a₁ : ℝ := x
  let a₂ : ℝ := x + 2
  let a₃ : ℝ := 2 * x + 6
  let a₅ : ℝ := r^4 * a₁
  (a₂ / a₁ = r) ∧ (a₃ / a₂ = r) →
  a₅ = ((1 + Real.sqrt 5) / (-1 + Real.sqrt 5)) * (4 + 2 * Real.sqrt 5) :=
by sorry

end fifth_term_geometric_progression_l1993_199385


namespace no_positive_integer_solutions_l1993_199316

theorem no_positive_integer_solutions
  (p : ℕ) (n : ℕ) (hp : Nat.Prime p) (hn : n > 0) :
  ¬∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x * (x + 1) = p^(2*n) * y * (y + 1) :=
by sorry

end no_positive_integer_solutions_l1993_199316


namespace fuel_tank_capacity_l1993_199379

theorem fuel_tank_capacity : ∃ (C : ℝ), C = 204 ∧ C > 0 := by
  -- Define the ethanol content of fuels A and B
  let ethanol_A : ℝ := 0.12
  let ethanol_B : ℝ := 0.16

  -- Define the volume of fuel A added
  let volume_A : ℝ := 66

  -- Define the total ethanol volume in the full tank
  let total_ethanol : ℝ := 30

  -- The capacity C satisfies the equation:
  -- ethanol_A * volume_A + ethanol_B * (C - volume_A) = total_ethanol
  
  sorry

end fuel_tank_capacity_l1993_199379


namespace poodle_bark_count_l1993_199383

/-- The number of times the terrier's owner says "hush" -/
def hush_count : ℕ := 6

/-- The ratio of poodle barks to terrier barks -/
def poodle_terrier_ratio : ℕ := 2

/-- The number of barks in a poodle bark set -/
def poodle_bark_set : ℕ := 5

/-- The number of times the terrier barks -/
def terrier_barks : ℕ := hush_count * 2

/-- The number of times the poodle barks -/
def poodle_barks : ℕ := terrier_barks * poodle_terrier_ratio

theorem poodle_bark_count : poodle_barks = 24 := by
  sorry

end poodle_bark_count_l1993_199383


namespace candle_count_l1993_199378

def total_candles (bedroom_candles : ℕ) (additional_candles : ℕ) : ℕ :=
  bedroom_candles + (bedroom_candles / 2) + additional_candles

theorem candle_count : total_candles 20 20 = 50 := by
  sorry

end candle_count_l1993_199378


namespace fraction_inequality_l1993_199396

theorem fraction_inequality (a b m : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hm : 0 < m) (hab : a < b) : 
  (a + m) / (b + m) > a / b := by
  sorry

end fraction_inequality_l1993_199396


namespace probability_theorem_l1993_199353

/-- The number of boys in the group -/
def num_boys : ℕ := 5

/-- The number of girls in the group -/
def num_girls : ℕ := 3

/-- The number of students to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly one girl -/
def prob_one_girl : ℚ := 15 / 28

/-- The probability of selecting exactly one girl given that at least one girl is selected -/
def prob_one_girl_given_at_least_one : ℚ := 5 / 6

/-- Theorem stating the probabilities for the given scenario -/
theorem probability_theorem :
  (prob_one_girl = 15 / 28) ∧
  (prob_one_girl_given_at_least_one = 5 / 6) :=
sorry

end probability_theorem_l1993_199353


namespace plane_intersection_line_properties_l1993_199310

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (intersect_at : Plane → Plane → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (intersects : Line → Line → Point → Prop)
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem plane_intersection_line_properties
  (α β : Plane) (l m : Line) (P : Point)
  (h1 : intersect_at α β l)
  (h2 : contains α m)
  (h3 : intersects m l P) :
  (∃ (n : Line), contains β n ∧ perpendicular m n) ∧
  (¬∃ (k : Line), contains β k ∧ parallel m k) :=
sorry

end plane_intersection_line_properties_l1993_199310


namespace triangle_side_length_l1993_199359

/-- Given a triangle ABC with ∠A = 40°, ∠B = 90°, and AC = 6, prove that BC = 6 * sin(40°) -/
theorem triangle_side_length (A B C : ℝ × ℝ) : 
  let angle (P Q R : ℝ × ℝ) := Real.arccos ((Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2)) / 
    (Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2) * Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2))
  let dist (P Q : ℝ × ℝ) := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  angle B A C = Real.pi / 4.5 →  -- 40°
  angle A B C = Real.pi / 2 →    -- 90°
  dist A C = 6 →
  dist B C = 6 * Real.sin (Real.pi / 4.5) := by
sorry

end triangle_side_length_l1993_199359


namespace forty_fifth_turn_turning_position_1978_to_2010_l1993_199347

-- Define the sequence of turning positions
def turningPosition (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    (1 + n / 2) * (n / 2) + 1
  else
    ((n + 1) / 2)^2 + 1

-- Theorem for the 45th turning position
theorem forty_fifth_turn : turningPosition 45 = 530 := by
  sorry

-- Theorem for the turning position between 1978 and 2010
theorem turning_position_1978_to_2010 :
  ∃ n : ℕ, turningPosition n = 1981 ∧
    1978 < turningPosition n ∧ turningPosition n < 2010 ∧
    ∀ m : ℕ, m ≠ n →
      (1978 < turningPosition m → turningPosition m ≥ 2010) ∨
      (turningPosition m ≤ 1978) := by
  sorry

end forty_fifth_turn_turning_position_1978_to_2010_l1993_199347


namespace consecutive_sum_26_l1993_199341

theorem consecutive_sum_26 (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 26 → n = 5 := by
  sorry

end consecutive_sum_26_l1993_199341


namespace smallest_first_term_divisible_by_11_l1993_199332

-- Define the arithmetic sequence
def arithmeticSequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

-- Define the sum of seven consecutive terms starting from k
def sumSevenTerms (k : ℕ) : ℤ := 
  (arithmeticSequence k) + 
  (arithmeticSequence (k + 1)) + 
  (arithmeticSequence (k + 2)) + 
  (arithmeticSequence (k + 3)) + 
  (arithmeticSequence (k + 4)) + 
  (arithmeticSequence (k + 5)) + 
  (arithmeticSequence (k + 6))

-- The theorem to prove
theorem smallest_first_term_divisible_by_11 :
  ∃ k : ℕ, (sumSevenTerms k) % 11 = 0 ∧ 
  ∀ m : ℕ, m < k → (sumSevenTerms m) % 11 ≠ 0 ∧
  arithmeticSequence k = 13 :=
sorry

end smallest_first_term_divisible_by_11_l1993_199332


namespace tan_value_second_quadrant_l1993_199328

/-- Given that α is an angle in the second quadrant and sin(π - α) = 3/5, prove that tan(α) = -3/4 -/
theorem tan_value_second_quadrant (α : Real) 
  (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
  (h2 : Real.sin (π - α) = 3/5) : 
  Real.tan α = -3/4 := by
  sorry

end tan_value_second_quadrant_l1993_199328


namespace cuboid_edge_length_l1993_199315

/-- Given a cuboid with edges x, 5, and 6, and volume 120, prove x = 4 -/
theorem cuboid_edge_length (x : ℝ) : x * 5 * 6 = 120 → x = 4 := by
  sorry

end cuboid_edge_length_l1993_199315


namespace choir_composition_l1993_199343

theorem choir_composition (initial_total : ℕ) : 
  let initial_girls : ℕ := (6 * initial_total) / 10
  let final_total : ℕ := initial_total + 6 - 4 - 2
  let final_girls : ℕ := initial_girls - 4
  (2 * final_girls = final_total) → initial_girls = 24 := by
sorry

end choir_composition_l1993_199343


namespace min_x_over_y_for_system_l1993_199325

/-- Given a system of equations, this theorem states that the minimum value of x/y
    for all solutions (x, y) is equal to (-1 - √217) / 12. -/
theorem min_x_over_y_for_system (x y : ℝ) :
  x^3 + 3*y^3 = 11 →
  x^2*y + x*y^2 = 6 →
  ∃ (min_val : ℝ), (∀ (x' y' : ℝ), x'^3 + 3*y'^3 = 11 → x'^2*y' + x'*y'^2 = 6 → x' / y' ≥ min_val) ∧
                   min_val = (-1 - Real.sqrt 217) / 12 :=
sorry

end min_x_over_y_for_system_l1993_199325


namespace cubic_equation_roots_l1993_199340

theorem cubic_equation_roots (P : ℤ) : 
  (∃ x y z : ℤ, 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x^3 - 10*x^2 + P*x - 30 = 0 ∧
    y^3 - 10*y^2 + P*y - 30 = 0 ∧
    z^3 - 10*z^2 + P*z - 30 = 0 ∧
    x ≠ y ∧ y ≠ z ∧ x ≠ z) →
  P = 31 :=
by sorry

end cubic_equation_roots_l1993_199340


namespace difference_1500th_1504th_term_l1993_199394

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem difference_1500th_1504th_term : 
  let a₁ := 3
  let d := 6
  |arithmetic_sequence a₁ d 1504 - arithmetic_sequence a₁ d 1500| = 24 := by
  sorry

end difference_1500th_1504th_term_l1993_199394


namespace common_root_equations_l1993_199349

theorem common_root_equations (p : ℝ) (h_p : p > 0) : 
  (∃ x : ℝ, 3 * x^2 - 4 * p * x + 9 = 0 ∧ x^2 - 2 * p * x + 5 = 0) ↔ p = 3 :=
by sorry

end common_root_equations_l1993_199349


namespace total_caffeine_consumption_l1993_199352

/-- Calculates the total caffeine consumption given the specifications of three drinks and a pill -/
theorem total_caffeine_consumption
  (drink1_oz : ℝ)
  (drink1_caffeine : ℝ)
  (drink2_oz : ℝ)
  (drink2_caffeine_multiplier : ℝ)
  (drink3_caffeine_per_ml : ℝ)
  (drink3_ml_consumed : ℝ) :
  drink1_oz = 12 →
  drink1_caffeine = 250 →
  drink2_oz = 8 →
  drink2_caffeine_multiplier = 3 →
  drink3_caffeine_per_ml = 18 →
  drink3_ml_consumed = 150 →
  let drink2_caffeine := (drink1_caffeine / drink1_oz) * drink2_caffeine_multiplier * drink2_oz
  let drink3_caffeine := drink3_caffeine_per_ml * drink3_ml_consumed
  let pill_caffeine := drink1_caffeine + drink2_caffeine + drink3_caffeine
  drink1_caffeine + drink2_caffeine + drink3_caffeine + pill_caffeine = 6900 := by
  sorry


end total_caffeine_consumption_l1993_199352


namespace basketball_team_cutoff_l1993_199388

/-- The number of students who didn't make the cut for the basketball team -/
theorem basketball_team_cutoff (girls boys callback : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : callback = 26) :
  girls + boys - callback = 17 := by
  sorry

end basketball_team_cutoff_l1993_199388


namespace sum_of_digits_M_l1993_199381

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- M is defined as the square root of 36^49 * 49^36 -/
def M : ℕ := sorry

/-- Theorem stating that the sum of digits of M is 37 -/
theorem sum_of_digits_M : sum_of_digits M = 37 := by sorry

end sum_of_digits_M_l1993_199381


namespace age_sum_proof_l1993_199301

theorem age_sum_proof (p q : ℕ) : 
  (p : ℚ) / q = 3 / 4 →
  p - 8 = (q - 8) / 2 →
  p + q = 28 := by
sorry

end age_sum_proof_l1993_199301


namespace graduation_ceremony_chairs_l1993_199339

/-- The number of graduates at a ceremony -/
def graduates : ℕ := 50

/-- The number of parents per graduate -/
def parents_per_graduate : ℕ := 2

/-- The number of teachers attending -/
def teachers : ℕ := 20

/-- The number of administrators attending -/
def administrators : ℕ := teachers / 2

/-- The total number of chairs available -/
def total_chairs : ℕ := 180

theorem graduation_ceremony_chairs :
  graduates + graduates * parents_per_graduate + teachers + administrators = total_chairs :=
sorry

end graduation_ceremony_chairs_l1993_199339


namespace valid_sets_count_l1993_199336

/-- Represents a family tree with 4 generations -/
structure FamilyTree :=
  (root : Unit)
  (gen1 : Fin 3)
  (gen2 : Fin 6)
  (gen3 : Fin 6)

/-- Represents a set of women from the family tree -/
def WomenSet := FamilyTree → Bool

/-- Checks if a set is valid (no woman and her daughter are both in the set) -/
def is_valid_set (s : WomenSet) : Bool :=
  sorry

/-- Counts the number of valid sets -/
def count_valid_sets : Nat :=
  sorry

/-- The main theorem to prove -/
theorem valid_sets_count : count_valid_sets = 793 :=
  sorry

end valid_sets_count_l1993_199336


namespace original_number_of_people_l1993_199344

theorem original_number_of_people (x : ℕ) : 
  (x / 3 : ℚ) = 18 → x = 54 := by sorry

end original_number_of_people_l1993_199344


namespace two_digit_division_problem_l1993_199321

theorem two_digit_division_problem :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 
  (∃ q r : ℕ, q = 9 ∧ r = 6 ∧ n = q * (n % 10) + r) :=
by sorry

end two_digit_division_problem_l1993_199321


namespace remainder_theorem_l1993_199327

/-- The polynomial f(x) = x^5 - 8x^4 + 15x^3 + 20x^2 - 5x - 20 -/
def f (x : ℝ) : ℝ := x^5 - 8*x^4 + 15*x^3 + 20*x^2 - 5*x - 20

/-- The theorem statement -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, f = fun x ↦ (x - 4) * q x + 216 := by sorry

end remainder_theorem_l1993_199327


namespace sqrt_31_between_5_and_6_l1993_199317

theorem sqrt_31_between_5_and_6 : 5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 := by
  sorry

end sqrt_31_between_5_and_6_l1993_199317


namespace cow_chicken_problem_l1993_199386

theorem cow_chicken_problem (c h : ℕ) : 
  4 * c + 2 * h = 2 * (c + h) + 20 → c = 10 := by
  sorry

end cow_chicken_problem_l1993_199386


namespace bowling_ball_surface_area_l1993_199320

theorem bowling_ball_surface_area :
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 81 * Real.pi := by
  sorry

end bowling_ball_surface_area_l1993_199320


namespace bird_watching_average_l1993_199311

theorem bird_watching_average : 
  let marcus_birds : ℕ := 7
  let humphrey_birds : ℕ := 11
  let darrel_birds : ℕ := 9
  let isabella_birds : ℕ := 15
  let total_birds : ℕ := marcus_birds + humphrey_birds + darrel_birds + isabella_birds
  let num_watchers : ℕ := 4
  (total_birds : ℚ) / num_watchers = 10.5 := by sorry

end bird_watching_average_l1993_199311


namespace exam_scoring_l1993_199319

theorem exam_scoring (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℕ) 
  (h1 : total_questions = 80)
  (h2 : correct_answers = 40)
  (h3 : total_marks = 120) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 := by
  sorry

end exam_scoring_l1993_199319


namespace special_integers_l1993_199309

def is_special (n : ℕ) : Prop :=
  (∃ d1 d2 : ℕ, 1 < d1 ∧ d1 < n ∧ d1 ∣ n ∧
                1 < d2 ∧ d2 < n ∧ d2 ∣ n ∧
                d1 ≠ d2) ∧
  (∀ d1 d2 : ℕ, 1 < d1 ∧ d1 < n ∧ d1 ∣ n →
                1 < d2 ∧ d2 < n ∧ d2 ∣ n →
                (d1 - d2) ∣ n ∨ (d2 - d1) ∣ n)

theorem special_integers :
  ∀ n : ℕ, is_special n ↔ n = 6 ∨ n = 8 ∨ n = 12 :=
by sorry

end special_integers_l1993_199309


namespace circle_equation_proof_l1993_199354

/-- The circle C with equation x^2 + y^2 + 10x + 10y = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 10*x + 10*y = 0

/-- The point A with coordinates (0, 6) -/
def point_A : ℝ × ℝ := (0, 6)

/-- The desired circle passing through A and tangent to C at the origin -/
def desired_circle (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 18

theorem circle_equation_proof :
  ∀ x y : ℝ,
  circle_C 0 0 ∧  -- C passes through the origin
  desired_circle (point_A.1) (point_A.2) ∧  -- Desired circle passes through A
  (∃ t : ℝ, t ≠ 0 ∧ 
    (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧ 
      ∀ x' y' : ℝ, 
      (x' - 0)^2 + (y' - 0)^2 < δ^2 → 
      (circle_C x' y' ∧ desired_circle x' y') ∨ 
      (¬circle_C x' y' ∧ ¬desired_circle x' y'))) →  -- Tangency condition
  desired_circle x y  -- The equation of the desired circle
:= by sorry

end circle_equation_proof_l1993_199354


namespace task_selection_count_l1993_199313

def num_males : ℕ := 3
def num_females : ℕ := 3
def total_students : ℕ := num_males + num_females
def num_selected : ℕ := 4

def num_single_person_tasks : ℕ := 2
def num_two_person_tasks : ℕ := 1

def selection_methods : ℕ := 144

theorem task_selection_count :
  (num_males = 3) →
  (num_females = 3) →
  (total_students = num_males + num_females) →
  (num_selected = 4) →
  (num_single_person_tasks = 2) →
  (num_two_person_tasks = 1) →
  selection_methods = 144 := by
  sorry

end task_selection_count_l1993_199313


namespace hannahs_tshirts_l1993_199391

theorem hannahs_tshirts (sweatshirt_count : ℕ) (sweatshirt_price : ℕ) (tshirt_price : ℕ) (total_spent : ℕ) :
  sweatshirt_count = 3 →
  sweatshirt_price = 15 →
  tshirt_price = 10 →
  total_spent = 65 →
  (total_spent - sweatshirt_count * sweatshirt_price) / tshirt_price = 2 := by
sorry

end hannahs_tshirts_l1993_199391


namespace triangle_preserving_characterization_l1993_199392

/-- A function satisfying the triangle property -/
def TrianglePreserving (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c →
    (a + b > c ∧ b + c > a ∧ c + a > b ↔ f a + f b > f c ∧ f b + f c > f a ∧ f c + f a > f b)

/-- Main theorem: Characterization of triangle-preserving functions -/
theorem triangle_preserving_characterization (f : ℝ → ℝ) 
    (h₁ : ∀ x, 0 < x → 0 < f x) 
    (h₂ : TrianglePreserving f) :
    ∃ c : ℝ, c > 0 ∧ ∀ x, 0 < x → f x = c * x :=
  sorry

end triangle_preserving_characterization_l1993_199392


namespace quadratic_inequality_solution_l1993_199312

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, -x^2 + c*x + 10 < 0 ↔ x < 2 ∨ x > 8) → c = 10 := by
  sorry

end quadratic_inequality_solution_l1993_199312


namespace percentage_increase_l1993_199374

theorem percentage_increase (x : ℝ) (h1 : x > 40) (h2 : x = 48) :
  (x - 40) / 40 * 100 = 20 := by
  sorry

end percentage_increase_l1993_199374


namespace smallest_integer_y_l1993_199308

theorem smallest_integer_y (y : ℤ) : (∀ z : ℤ, z < y → 3 * z - 6 ≥ 15) ∧ 3 * y - 6 < 15 ↔ y = 6 := by
  sorry

end smallest_integer_y_l1993_199308


namespace variety_show_arrangements_l1993_199355

def dance_song_count : ℕ := 3
def comedy_skit_count : ℕ := 2
def cross_talk_count : ℕ := 1

def non_adjacent_arrangements (ds c ct : ℕ) : ℕ :=
  ds.factorial * (2 * ds.factorial + c.choose 1 * (ds - 1).factorial * (ds - 1).factorial)

theorem variety_show_arrangements :
  non_adjacent_arrangements dance_song_count comedy_skit_count cross_talk_count = 120 := by
  sorry

end variety_show_arrangements_l1993_199355


namespace sandy_paint_area_l1993_199302

/-- The area to be painted on Sandy's bedroom wall -/
def areaToPaint (wallHeight wallLength window1Height window1Width window2Height window2Width : ℝ) : ℝ :=
  wallHeight * wallLength - (window1Height * window1Width + window2Height * window2Width)

/-- Theorem: The area Sandy needs to paint is 131 square feet -/
theorem sandy_paint_area :
  areaToPaint 10 15 3 5 2 2 = 131 := by
  sorry

end sandy_paint_area_l1993_199302


namespace composite_sum_of_power_l1993_199314

theorem composite_sum_of_power (n : ℕ) (h : n ≥ 2) : 
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 4^n = a * b :=
sorry

end composite_sum_of_power_l1993_199314


namespace expression_simplification_l1993_199361

theorem expression_simplification (a : ℝ) (ha : a ≥ 0) :
  (((2 * (a + 1) + 2 * Real.sqrt (a^2 + 2*a)) / (3*a + 1 - 2 * Real.sqrt (a^2 + 2*a)))^(1/2 : ℝ)) -
  ((Real.sqrt (2*a + 1) - Real.sqrt a)⁻¹ * Real.sqrt (a + 2)) =
  Real.sqrt a / (Real.sqrt (2*a + 1) - Real.sqrt a) := by
sorry

end expression_simplification_l1993_199361


namespace caroline_lassis_l1993_199306

/-- Represents the number of lassis that can be made with given ingredients -/
def max_lassis (initial_lassis initial_mangoes initial_coconuts available_mangoes available_coconuts : ℚ) : ℚ :=
  min 
    (available_mangoes * (initial_lassis / initial_mangoes))
    (available_coconuts * (initial_lassis / initial_coconuts))

/-- Theorem stating that Caroline can make 55 lassis with the given ingredients -/
theorem caroline_lassis : 
  max_lassis 11 2 4 12 20 = 55 := by
  sorry

#eval max_lassis 11 2 4 12 20

end caroline_lassis_l1993_199306


namespace average_and_variance_after_adding_datapoint_l1993_199303

def initial_average : ℝ := 4
def initial_variance : ℝ := 2
def initial_count : ℕ := 7
def new_datapoint : ℝ := 4
def new_count : ℕ := initial_count + 1

def new_average (x : ℝ) : Prop :=
  x = (initial_count * initial_average + new_datapoint) / new_count

def new_variance (s : ℝ) : Prop :=
  s = (initial_count * initial_variance + (new_datapoint - initial_average)^2) / new_count

theorem average_and_variance_after_adding_datapoint :
  ∃ (x s : ℝ), new_average x ∧ new_variance s ∧ x = initial_average ∧ s < initial_variance :=
sorry

end average_and_variance_after_adding_datapoint_l1993_199303


namespace ball_game_attendance_l1993_199324

/-- The number of children at a ball game -/
def num_children : ℕ :=
  let num_adults : ℕ := 10
  let adult_ticket_price : ℕ := 8
  let child_ticket_price : ℕ := 4
  let total_bill : ℕ := 124
  let adult_cost : ℕ := num_adults * adult_ticket_price
  let child_cost : ℕ := total_bill - adult_cost
  child_cost / child_ticket_price

theorem ball_game_attendance : num_children = 11 := by
  sorry

end ball_game_attendance_l1993_199324


namespace half_of_five_bananas_worth_l1993_199360

-- Define the worth of bananas in terms of oranges
def banana_orange_ratio : ℚ := 8 / (2/3 * 10)

-- Theorem statement
theorem half_of_five_bananas_worth (banana_orange_ratio : ℚ) :
  banana_orange_ratio = 8 / (2/3 * 10) →
  (1/2 * 5) * banana_orange_ratio = 3 := by
  sorry

end half_of_five_bananas_worth_l1993_199360


namespace min_omega_value_l1993_199329

theorem min_omega_value (f : ℝ → ℝ) (ω φ T : ℝ) : 
  (∀ x, f x = Real.cos (ω * x + φ)) →
  ω > 0 →
  0 < φ ∧ φ < π →
  T > 0 →
  (∀ t > 0, (∀ x, f (x + t) = f x) → T ≤ t) →
  f T = Real.sqrt 3 / 2 →
  f (π / 9) = 0 →
  3 ≤ ω ∧ ∀ ω' ≥ 0, (
    (∀ x, Real.cos (ω' * x + φ) = Real.cos (ω * x + φ)) →
    (Real.cos (ω' * T + φ) = Real.sqrt 3 / 2) →
    (Real.cos (ω' * π / 9 + φ) = 0) →
    ω' ≥ 3
  ) := by sorry


end min_omega_value_l1993_199329


namespace sequence_problem_l1993_199393

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An arithmetic sequence -/
def IsArithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The problem statement -/
theorem sequence_problem (a : Sequence) 
  (h_arith : IsArithmetic a)
  (h_geom : IsGeometric (fun n => a (n + 1)))
  (h_a5 : a 5 = 1) :
  a 10 = 1 := by
  sorry

end sequence_problem_l1993_199393


namespace polynomial_value_l1993_199384

theorem polynomial_value (x : ℝ) (h : x^2 + 3*x = 1) : 3*x^2 + 9*x - 1 = 2 := by
  sorry

end polynomial_value_l1993_199384


namespace age_difference_l1993_199367

theorem age_difference (louis_age jerica_age matilda_age : ℕ) : 
  louis_age = 14 →
  jerica_age = 2 * louis_age →
  matilda_age = 35 →
  matilda_age - jerica_age = 7 :=
by
  sorry

end age_difference_l1993_199367


namespace scientific_notation_of_43300000_l1993_199307

theorem scientific_notation_of_43300000 : 
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 43300000 = a * (10 : ℝ) ^ n ∧ a = 4.33 ∧ n = 7 :=
by sorry

end scientific_notation_of_43300000_l1993_199307


namespace transform_to_zero_y_l1993_199390

-- Define the set M
def M : Set (ℤ × ℤ) := Set.univ

-- Define transformation S
def S (p : ℤ × ℤ) : ℤ × ℤ := (p.1 + p.2, p.2)

-- Define transformation T
def T (p : ℤ × ℤ) : ℤ × ℤ := (-p.2, p.1)

-- Define the type of transformations
inductive Transform
| S : Transform
| T : Transform

-- Define the application of a sequence of transformations
def applyTransforms : List Transform → ℤ × ℤ → ℤ × ℤ
| [], p => p
| (Transform.S :: ts), p => applyTransforms ts (S p)
| (Transform.T :: ts), p => applyTransforms ts (T p)

-- The main theorem
theorem transform_to_zero_y (p : ℤ × ℤ) : 
  ∃ (ts : List Transform) (g : ℤ), applyTransforms ts p = (g, 0) := by
  sorry


end transform_to_zero_y_l1993_199390


namespace greatest_x_value_l1993_199363

theorem greatest_x_value (x : ℤ) (h : (6.1 : ℝ) * (10 : ℝ) ^ (x : ℝ) < 620) :
  x ≤ 2 ∧ ∃ y : ℤ, y > 2 → (6.1 : ℝ) * (10 : ℝ) ^ (y : ℝ) ≥ 620 :=
by sorry

end greatest_x_value_l1993_199363


namespace simplify_expression_l1993_199398

theorem simplify_expression (y : ℝ) : 3*y + 4*y^2 - 2 - (8 - 3*y - 4*y^2) = 8*y^2 + 6*y - 10 := by
  sorry

end simplify_expression_l1993_199398


namespace abs_diff_of_sum_and_product_l1993_199335

theorem abs_diff_of_sum_and_product (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (prod_eq : x * y = 221) : 
  |x - y| = 4 := by
sorry

end abs_diff_of_sum_and_product_l1993_199335


namespace one_meeting_l1993_199389

/-- Represents the movement and meeting of a jogger and an aid vehicle --/
structure JoggerVehicleSystem where
  jogger_speed : ℝ
  vehicle_speed : ℝ
  station_distance : ℝ
  vehicle_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between the jogger and the vehicle --/
def number_of_meetings (sys : JoggerVehicleSystem) : ℕ :=
  sorry

/-- The specific scenario described in the problem --/
def problem_scenario : JoggerVehicleSystem :=
  { jogger_speed := 6
  , vehicle_speed := 12
  , station_distance := 300
  , vehicle_stop_time := 20
  , initial_distance := 300 }

/-- Theorem stating that in the given scenario, there is exactly one meeting --/
theorem one_meeting :
  number_of_meetings problem_scenario = 1 :=
sorry

end one_meeting_l1993_199389


namespace det_A_eq_46_l1993_199305

def A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 0, -1; 7, 4, -3; 2, 2, 5]

theorem det_A_eq_46 : A.det = 46 := by sorry

end det_A_eq_46_l1993_199305


namespace not_p_and_q_implies_at_least_one_false_l1993_199366

theorem not_p_and_q_implies_at_least_one_false (p q : Prop) :
  ¬(p ∧ q) → (¬p ∨ ¬q) := by sorry

end not_p_and_q_implies_at_least_one_false_l1993_199366


namespace marys_characters_l1993_199375

theorem marys_characters (total : ℕ) (a b c d e f : ℕ) : 
  total = 120 →
  a = total / 3 →
  b = (total - a) / 4 →
  c = (total - a - b) / 5 →
  d + e + f = total - a - b - c →
  d = 3 * e →
  e = f / 2 →
  d = 24 := by sorry

end marys_characters_l1993_199375


namespace cookies_with_seven_cups_l1993_199380

/-- The number of cookies Lee can make with a given number of cups of flour -/
def cookies_made (cups : ℕ) : ℝ :=
  if cups ≤ 4 then 36
  else cookies_made (cups - 1) * 1.5

/-- The theorem stating the number of cookies Lee can make with 7 cups of flour -/
theorem cookies_with_seven_cups :
  cookies_made 7 = 121.5 := by
  sorry

end cookies_with_seven_cups_l1993_199380


namespace A_intersect_B_l1993_199395

def A : Set ℝ := {x | x^2 - x - 6 < 0}
def B : Set ℝ := {x | |x - 2| < 2}

theorem A_intersect_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 3} := by sorry

end A_intersect_B_l1993_199395


namespace choose_five_from_ten_l1993_199356

theorem choose_five_from_ten : Nat.choose 10 5 = 252 := by sorry

end choose_five_from_ten_l1993_199356


namespace polynomial_equation_odd_degree_l1993_199330

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- The statement of the theorem -/
theorem polynomial_equation_odd_degree (d : ℕ) :
  (d > 0 ∧ ∃ (P Q : RealPolynomial), 
    (Polynomial.degree P = d) ∧ 
    (∀ x : ℝ, P.eval x ^ 2 + 1 = (x^2 + 1) * Q.eval x ^ 2)) ↔ 
  Odd d :=
sorry

end polynomial_equation_odd_degree_l1993_199330


namespace cos_thirty_degrees_l1993_199358

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by sorry

end cos_thirty_degrees_l1993_199358


namespace discount_rate_calculation_l1993_199318

def marked_price : ℝ := 240
def selling_price : ℝ := 120

theorem discount_rate_calculation : 
  (marked_price - selling_price) / marked_price * 100 = 50 := by sorry

end discount_rate_calculation_l1993_199318


namespace question_mark_value_l1993_199372

theorem question_mark_value (x : ℝ) : (x * 74) / 30 = 1938.8 → x = 786 := by
  sorry

end question_mark_value_l1993_199372


namespace geometric_sequence_sum_l1993_199351

/-- A geometric sequence with the given properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧
  a 1 + a 2 + a 3 = 1 ∧
  a 2 + a 3 + a 4 = 2

/-- The sum of the 6th, 7th, and 8th terms equals 32 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 6 + a 7 + a 8 = 32 := by
  sorry

end geometric_sequence_sum_l1993_199351


namespace set_operations_l1993_199399

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}

def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}

def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5/2}

theorem set_operations :
  (A ∩ B = {x | -1 < x ∧ x < 2}) ∧
  ((U \ B) ∪ P = {x | x ≤ 0 ∨ x ≥ 5/2}) ∧
  ((A ∩ B) ∩ (U \ P) = {x | 0 < x ∧ x < 2}) := by
  sorry

end set_operations_l1993_199399


namespace simplify_and_rationalize_l1993_199369

theorem simplify_and_rationalize (x : ℝ) : 
  (1 : ℝ) / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end simplify_and_rationalize_l1993_199369


namespace permutation_equation_solution_l1993_199376

/-- Permutation function -/
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

/-- The problem statement -/
theorem permutation_equation_solution :
  ∃! (x : ℕ), x > 0 ∧ x ≤ 5 ∧ A 5 x = 2 * A 6 (x - 1) :=
by sorry

end permutation_equation_solution_l1993_199376


namespace swimmer_passes_l1993_199368

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  delay : ℝ

/-- Calculates the number of times swimmers pass each other --/
def count_passes (pool_length : ℝ) (time : ℝ) (swimmer_a : Swimmer) (swimmer_b : Swimmer) : ℕ :=
  sorry

/-- Theorem stating the number of passes in the given scenario --/
theorem swimmer_passes :
  let pool_length : ℝ := 120
  let total_time : ℝ := 900
  let swimmer_a : Swimmer := { speed := 3, delay := 0 }
  let swimmer_b : Swimmer := { speed := 4, delay := 10 }
  count_passes pool_length total_time swimmer_a swimmer_b = 38 := by
  sorry

end swimmer_passes_l1993_199368


namespace triangular_array_sum_recurrence_l1993_199333

def triangular_array_sum (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | n+1 => 2 * triangular_array_sum n + 2 * n

theorem triangular_array_sum_recurrence (n : ℕ) (h : n ≥ 2) :
  triangular_array_sum n = 2 * triangular_array_sum (n-1) + 2 * (n-1) :=
by sorry

#eval triangular_array_sum 20

end triangular_array_sum_recurrence_l1993_199333


namespace scientific_notation_correct_l1993_199387

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_bounds : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number we want to express in scientific notation -/
def target_number : ℝ := 318000000

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 3.18
    exponent := 8
    coeff_bounds := by sorry }

/-- Theorem stating that the proposed notation correctly represents the target number -/
theorem scientific_notation_correct :
  target_number = proposed_notation.coefficient * (10 : ℝ) ^ proposed_notation.exponent :=
sorry

end scientific_notation_correct_l1993_199387


namespace downstream_distance_is_100_l1993_199304

/-- Represents the properties of a boat traveling in a stream -/
structure BoatTravel where
  downstream_time : ℝ
  upstream_distance : ℝ
  upstream_time : ℝ
  stream_speed : ℝ

/-- Calculates the downstream distance given boat travel properties -/
def downstream_distance (bt : BoatTravel) : ℝ :=
  sorry

/-- Theorem stating that the downstream distance is 100 km given specific conditions -/
theorem downstream_distance_is_100 (bt : BoatTravel) 
  (h1 : bt.downstream_time = 10)
  (h2 : bt.upstream_distance = 200)
  (h3 : bt.upstream_time = 25)
  (h4 : bt.stream_speed = 1) :
  downstream_distance bt = 100 := by
  sorry

end downstream_distance_is_100_l1993_199304


namespace right_triangle_distance_theorem_l1993_199348

theorem right_triangle_distance_theorem (a b : ℝ) (ha : a = 9) (hb : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let area := a * b / 2
  let r := area / s
  let centroid_dist := 2 * c / 3
  1 = centroid_dist - r :=
by sorry

end right_triangle_distance_theorem_l1993_199348


namespace max_value_inequality_max_value_achievable_l1993_199346

theorem max_value_inequality (a b c d : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) (hd : d ≥ 0)
  (hsum : a + b + c + d = 100) : 
  (a / (b + 7))^(1/3) + (b / (c + 7))^(1/3) + (c / (d + 7))^(1/3) + (d / (a + 7))^(1/3) ≤ 2 * 25^(1/3) :=
by sorry

theorem max_value_achievable : 
  ∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 100 ∧
  (a / (b + 7))^(1/3) + (b / (c + 7))^(1/3) + (c / (d + 7))^(1/3) + (d / (a + 7))^(1/3) = 2 * 25^(1/3) :=
by sorry

end max_value_inequality_max_value_achievable_l1993_199346


namespace carlos_laundry_loads_l1993_199365

theorem carlos_laundry_loads (wash_time_per_load : ℕ) (dry_time : ℕ) (total_time : ℕ) 
  (h1 : wash_time_per_load = 45)
  (h2 : dry_time = 75)
  (h3 : total_time = 165) :
  ∃ n : ℕ, n * wash_time_per_load + dry_time = total_time ∧ n = 2 :=
by
  sorry

end carlos_laundry_loads_l1993_199365


namespace regular_ticket_cost_l1993_199300

theorem regular_ticket_cost (total_tickets : ℕ) (senior_ticket_cost : ℕ) (total_sales : ℕ) (regular_tickets_sold : ℕ) :
  total_tickets = 65 →
  senior_ticket_cost = 10 →
  total_sales = 855 →
  regular_tickets_sold = 41 →
  ∃ (regular_ticket_cost : ℕ),
    regular_ticket_cost * regular_tickets_sold + senior_ticket_cost * (total_tickets - regular_tickets_sold) = total_sales ∧
    regular_ticket_cost = 15 :=
by
  sorry

end regular_ticket_cost_l1993_199300


namespace different_color_chips_probability_l1993_199373

/-- The probability of drawing two chips of different colors from a bag containing
    7 blue chips and 5 yellow chips, with replacement after the first draw. -/
theorem different_color_chips_probability :
  let blue_chips : ℕ := 7
  let yellow_chips : ℕ := 5
  let total_chips : ℕ := blue_chips + yellow_chips
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_yellow : ℚ := yellow_chips / total_chips
  let prob_different_colors : ℚ := prob_blue * prob_yellow + prob_yellow * prob_blue
  prob_different_colors = 35 / 72 := by
sorry

end different_color_chips_probability_l1993_199373


namespace unique_n_reaches_two_l1993_199331

def g (n : ℤ) : ℤ := 
  if n % 2 = 1 then n^2 - 2*n + 2 else 2*n

def iterateG (n : ℤ) (k : ℕ) : ℤ :=
  match k with
  | 0 => n
  | k+1 => g (iterateG n k)

theorem unique_n_reaches_two :
  ∃! n : ℤ, 1 ≤ n ∧ n ≤ 100 ∧ ∃ k : ℕ, iterateG n k = 2 :=
sorry

end unique_n_reaches_two_l1993_199331


namespace sqrt_equation_solution_l1993_199370

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 15) = 12 → x = 129 := by
  sorry

end sqrt_equation_solution_l1993_199370
