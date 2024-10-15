import Mathlib

namespace NUMINAMATH_CALUDE_smallest_sum_of_two_primes_above_70_l1081_108148

theorem smallest_sum_of_two_primes_above_70 : 
  ∃ (p q : Nat), 
    Prime p ∧ 
    Prime q ∧ 
    p > 70 ∧ 
    q > 70 ∧ 
    p ≠ q ∧ 
    p + q = 144 ∧ 
    (∀ (r s : Nat), Prime r → Prime s → r > 70 → s > 70 → r ≠ s → r + s ≥ 144) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_two_primes_above_70_l1081_108148


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l1081_108186

open Matrix

/-- Given a 2x2 matrix B with its inverse, prove that the inverse of B^3 is equal to B^(-1) -/
theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B⁻¹ = ![![3, 4], ![-2, -3]]) : 
  (B^3)⁻¹ = ![![3, 4], ![-2, -3]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l1081_108186


namespace NUMINAMATH_CALUDE_line_equation_l1081_108120

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Checks if a given equation represents the line -/
def is_equation_of_line (l : Line) (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, f x y ↔ y - l.point.snd = l.slope * (x - l.point.fst)

theorem line_equation (l : Line) :
  l.slope = 3 ∧ l.point = (1, 3) →
  is_equation_of_line l (fun x y ↦ y - 3 = 3 * (x - 1)) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1081_108120


namespace NUMINAMATH_CALUDE_game_cost_l1081_108180

theorem game_cost (initial_amount allowance final_amount : ℕ) : 
  initial_amount = 5 → 
  allowance = 26 → 
  final_amount = 29 → 
  initial_amount + allowance - final_amount = 2 := by
sorry

end NUMINAMATH_CALUDE_game_cost_l1081_108180


namespace NUMINAMATH_CALUDE_number_puzzle_l1081_108143

theorem number_puzzle : ∃ x : ℝ, (20 * (x / 5) = 40) ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1081_108143


namespace NUMINAMATH_CALUDE_rainfall_problem_l1081_108100

/-- Rainfall problem -/
theorem rainfall_problem (monday_hours : ℝ) (monday_rate : ℝ) (tuesday_rate : ℝ)
  (wednesday_hours : ℝ) (total_rainfall : ℝ)
  (h1 : monday_hours = 7)
  (h2 : monday_rate = 1)
  (h3 : tuesday_rate = 2)
  (h4 : wednesday_hours = 2)
  (h5 : total_rainfall = 23)
  (h6 : wednesday_rate = 2 * tuesday_rate) :
  ∃ tuesday_hours : ℝ,
    tuesday_hours = 4 ∧
    total_rainfall = monday_hours * monday_rate +
                     tuesday_hours * tuesday_rate +
                     wednesday_hours * wednesday_rate :=
by sorry

end NUMINAMATH_CALUDE_rainfall_problem_l1081_108100


namespace NUMINAMATH_CALUDE_triangle_negative_five_sixths_one_half_l1081_108198

/-- The triangle operation on rational numbers -/
def triangle (a b : ℚ) : ℚ := b - a

theorem triangle_negative_five_sixths_one_half :
  triangle (-5/6) (1/2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_negative_five_sixths_one_half_l1081_108198


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l1081_108161

theorem complete_square_quadratic (a b c : ℝ) (h : a = 1 ∧ b = 6 ∧ c = 5) :
  ∃ (k : ℝ), (x + k)^2 - (x^2 + b*x + c) = 4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l1081_108161


namespace NUMINAMATH_CALUDE_w_coordinate_of_point_on_line_l1081_108159

/-- A 4D point -/
structure Point4D where
  x : ℝ
  y : ℝ
  z : ℝ
  w : ℝ

/-- Definition of the line passing through two points -/
def line_through (p q : Point4D) (t : ℝ) : Point4D :=
  { x := p.x + t * (q.x - p.x),
    y := p.y + t * (q.y - p.y),
    z := p.z + t * (q.z - p.z),
    w := p.w + t * (q.w - p.w) }

/-- The theorem to be proved -/
theorem w_coordinate_of_point_on_line : 
  let p1 : Point4D := {x := 3, y := 3, z := 2, w := 1}
  let p2 : Point4D := {x := 6, y := 2, z := 1, w := -1}
  ∃ t : ℝ, 
    let point := line_through p1 p2 t
    point.y = 4 ∧ point.w = 3 := by
  sorry

end NUMINAMATH_CALUDE_w_coordinate_of_point_on_line_l1081_108159


namespace NUMINAMATH_CALUDE_problem_statement_l1081_108197

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x / Real.log x - a * x

theorem problem_statement :
  (∃ (a : ℝ), ∀ (x y : ℝ), 1 < x ∧ x < y → f a y ≤ f a x) ∧
  (∃ (a : ℝ), ∀ (x₁ x₂ : ℝ), Real.exp 1 ≤ x₁ ∧ x₁ ≤ Real.exp 2 ∧
                              Real.exp 1 ≤ x₂ ∧ x₂ ≤ Real.exp 2 →
                              f a x₁ ≤ (deriv (f a)) x₂ + a) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1081_108197


namespace NUMINAMATH_CALUDE_sibling_ages_l1081_108142

/-- Represents the ages of the siblings -/
structure SiblingAges where
  maria : ℕ
  ann : ℕ
  david : ℕ
  ethan : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : SiblingAges) : Prop :=
  ages.maria = ages.ann - 3 ∧
  ages.maria - 4 = (ages.ann - 4) / 2 ∧
  ages.david = ages.maria + 2 ∧
  (ages.david - 2) + (ages.ann - 2) = 3 * (ages.maria - 2) ∧
  ages.ethan = ages.david - ages.maria ∧
  ages.ann - ages.ethan = 8

/-- The theorem stating the ages of the siblings -/
theorem sibling_ages : 
  ∃ (ages : SiblingAges), satisfiesConditions ages ∧ 
    ages.maria = 7 ∧ ages.ann = 10 ∧ ages.david = 9 ∧ ages.ethan = 2 := by
  sorry

end NUMINAMATH_CALUDE_sibling_ages_l1081_108142


namespace NUMINAMATH_CALUDE_arithmetic_geometric_progression_sine_l1081_108185

theorem arithmetic_geometric_progression_sine (x y z : ℝ) :
  let α := Real.arccos (-1/5)
  (∃ d, x = y - d ∧ z = y + d ∧ d = α) →
  (∃ r ≠ 1, (2 + Real.sin x) * (2 + Real.sin z) = (2 + Real.sin y)^2 ∧ 
             (2 + Real.sin y) = r * (2 + Real.sin x) ∧
             (2 + Real.sin z) = r * (2 + Real.sin y)) →
  Real.sin y = -1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_progression_sine_l1081_108185


namespace NUMINAMATH_CALUDE_negation_equivalence_l1081_108137

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 + Real.sin x + 1 < 0) ↔ (∃ x : ℝ, x^2 + Real.sin x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1081_108137


namespace NUMINAMATH_CALUDE_triangle_dot_product_l1081_108179

-- Define the triangle ABC
theorem triangle_dot_product (A B C : ℝ × ℝ) :
  -- Given conditions
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let S := abs ((A.1 - C.1) * (B.2 - C.2) - (A.2 - C.2) * (B.1 - C.1)) / 2
  -- Hypothesis
  AC = 8 →
  BC = 5 →
  S = 10 * Real.sqrt 3 →
  -- Conclusion
  ((B.1 - C.1) * (C.1 - A.1) + (B.2 - C.2) * (C.2 - A.2) = 20 ∨
   (B.1 - C.1) * (C.1 - A.1) + (B.2 - C.2) * (C.2 - A.2) = -20) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_dot_product_l1081_108179


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1081_108182

theorem quadratic_roots_range (m : ℝ) : 
  (¬ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + m = 0 ∧ x₂^2 + 2*x₂ + m = 0) →
  (5 - 2*m > 1) →
  1 ≤ m ∧ m < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1081_108182


namespace NUMINAMATH_CALUDE_bryden_quarter_sale_l1081_108130

/-- The amount a collector pays for state quarters as a percentage of face value -/
def collector_offer_percentage : ℚ := 2500

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 5

/-- The face value of a single state quarter in dollars -/
def quarter_face_value : ℚ := 1/4

/-- The amount Bryden will receive for his quarters in dollars -/
def bryden_received_amount : ℚ := 31.25

theorem bryden_quarter_sale :
  (collector_offer_percentage / 100) * (bryden_quarters : ℚ) * quarter_face_value = bryden_received_amount := by
  sorry

end NUMINAMATH_CALUDE_bryden_quarter_sale_l1081_108130


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1081_108177

theorem point_in_fourth_quadrant (a : ℝ) (h : a < -1) :
  let x := a^2 - 2*a - 1
  let y := (a + 1) / |a + 1|
  x > 0 ∧ y < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1081_108177


namespace NUMINAMATH_CALUDE_uniform_transformation_l1081_108108

theorem uniform_transformation (a₁ : ℝ) : 
  a₁ ∈ Set.Icc 0 1 → (8 * a₁ - 2) ∈ Set.Icc (-2) 6 := by
  sorry

end NUMINAMATH_CALUDE_uniform_transformation_l1081_108108


namespace NUMINAMATH_CALUDE_triangle_area_72_l1081_108117

/-- 
Given a right triangle with vertices (0, 0), (x, 3x), and (x, 0),
prove that if its area is 72 square units and x > 0, then x = 4√3.
-/
theorem triangle_area_72 (x : ℝ) (h1 : x > 0) : 
  (1/2 : ℝ) * x * (3*x) = 72 → x = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_72_l1081_108117


namespace NUMINAMATH_CALUDE_division_remainder_problem_l1081_108101

theorem division_remainder_problem (L S : ℕ) : 
  L - S = 1365 → 
  L = 1637 → 
  ∃ (q : ℕ), q = 6 ∧ L = q * S + (L % S) → 
  L % S = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_problem_l1081_108101


namespace NUMINAMATH_CALUDE_cube_root_problem_l1081_108168

theorem cube_root_problem (a : ℕ) : a^3 = 21 * 25 * 45 * 49 → a = 105 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_problem_l1081_108168


namespace NUMINAMATH_CALUDE_polygon_properties_l1081_108187

theorem polygon_properties :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    -- Condition: Each interior angle is 30° more than four times its adjacent exterior angle
    (180 : ℝ) = exterior_angle + 4 * exterior_angle + 30 →
    -- Condition: Sum of exterior angles is always 360°
    (n : ℝ) * exterior_angle = 360 →
    -- Conclusions
    n = 12 ∧
    (n - 2 : ℝ) * 180 = 1800 ∧
    n * (n - 3) / 2 = 54 :=
by
  sorry


end NUMINAMATH_CALUDE_polygon_properties_l1081_108187


namespace NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l1081_108119

-- Case 1
theorem triangle_case1 (AB AD HM : ℝ) (h1 : AB = 10) (h2 : AD = 4) (h3 : HM = 6/5) :
  let BD := Real.sqrt (AB^2 - AD^2)
  let DH := (4 * Real.sqrt 21) / 5
  let DC := BD - HM
  DC = (8 * Real.sqrt 21 - 12) / 5 := by sorry

-- Case 2
theorem triangle_case2 (AB AD HM : ℝ) (h1 : AB = 8 * Real.sqrt 2) (h2 : AD = 4) (h3 : HM = Real.sqrt 2) :
  let BD := Real.sqrt (AB^2 - AD^2)
  let DH := Real.sqrt 14
  let DC := BD - HM
  DC = 2 * Real.sqrt 14 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_triangle_case1_triangle_case2_l1081_108119


namespace NUMINAMATH_CALUDE_age_difference_in_decades_l1081_108134

/-- Given that the sum of x's and y's ages is 18 years greater than the sum of y's and z's ages,
    prove that z is 1.8 decades younger than x. -/
theorem age_difference_in_decades (x y z : ℕ) (h : x + y = y + z + 18) :
  (x - z : ℚ) / 10 = 1.8 := by sorry

end NUMINAMATH_CALUDE_age_difference_in_decades_l1081_108134


namespace NUMINAMATH_CALUDE_product_of_squares_l1081_108150

theorem product_of_squares (x : ℝ) :
  (Real.sqrt (7 + x) + Real.sqrt (28 - x) = 9) →
  (7 + x) * (28 - x) = 529 := by
sorry

end NUMINAMATH_CALUDE_product_of_squares_l1081_108150


namespace NUMINAMATH_CALUDE_integer_solution_exists_l1081_108191

theorem integer_solution_exists (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (∃ ℓ : ℤ, a = 7 * ℓ + 1 ∨ a = 7 * ℓ - 1) :=
by sorry

end NUMINAMATH_CALUDE_integer_solution_exists_l1081_108191


namespace NUMINAMATH_CALUDE_speed_ratio_l1081_108158

/-- The race scenario where A and B run at different speeds and finish at the same time -/
structure RaceScenario where
  speed_A : ℝ
  speed_B : ℝ
  distance_A : ℝ
  distance_B : ℝ
  finish_time : ℝ

/-- The conditions of the race -/
def race_conditions (r : RaceScenario) : Prop :=
  r.distance_A = 84 ∧ 
  r.distance_B = 42 ∧ 
  r.finish_time = r.distance_A / r.speed_A ∧
  r.finish_time = r.distance_B / r.speed_B

/-- The theorem stating the ratio of A's speed to B's speed -/
theorem speed_ratio (r : RaceScenario) (h : race_conditions r) : 
  r.speed_A / r.speed_B = 2 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_l1081_108158


namespace NUMINAMATH_CALUDE_purely_imaginary_solution_l1081_108181

theorem purely_imaginary_solution (z : ℂ) :
  (∃ b : ℝ, z = Complex.I * b) →
  (∃ c : ℝ, (z - 2)^2 - Complex.I * 8 = Complex.I * c) →
  z = Complex.I * 2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_solution_l1081_108181


namespace NUMINAMATH_CALUDE_product_equality_l1081_108189

theorem product_equality (x y : ℝ) :
  (3 * x^4 - 7 * y^3) * (9 * x^12 + 21 * x^8 * y^3 + 21 * x^4 * y^6 + 49 * y^9) =
  81 * x^16 - 2401 * y^12 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1081_108189


namespace NUMINAMATH_CALUDE_complete_square_quadratic_l1081_108127

theorem complete_square_quadratic : ∀ x : ℝ, x^2 - 4*x + 2 = 0 ↔ (x - 2)^2 = 2 := by sorry

end NUMINAMATH_CALUDE_complete_square_quadratic_l1081_108127


namespace NUMINAMATH_CALUDE_convention_handshakes_count_l1081_108124

/-- Represents the number of handshakes at a convention with twins and triplets -/
def convention_handshakes (twin_sets triplet_sets : ℕ) : ℕ :=
  let twins := twin_sets * 2
  let triplets := triplet_sets * 3
  let twin_handshakes := twins * (twins - 2) / 2
  let triplet_handshakes := triplets * (triplets - 3) / 2
  let cross_handshakes := twins * (2 * triplets / 3)
  twin_handshakes + triplet_handshakes + cross_handshakes

/-- The number of handshakes at the convention is 900 -/
theorem convention_handshakes_count : convention_handshakes 12 8 = 900 := by
  sorry

#eval convention_handshakes 12 8

end NUMINAMATH_CALUDE_convention_handshakes_count_l1081_108124


namespace NUMINAMATH_CALUDE_unique_solution_square_equation_l1081_108196

theorem unique_solution_square_equation :
  ∃! x : ℚ, (2015 + x)^2 = x^2 ∧ x = -2015/2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_square_equation_l1081_108196


namespace NUMINAMATH_CALUDE_toy_sword_cost_l1081_108140

theorem toy_sword_cost (total_spent : ℕ) (lego_cost : ℕ) (play_dough_cost : ℕ)
  (lego_sets : ℕ) (toy_swords : ℕ) (play_doughs : ℕ) :
  total_spent = 1940 →
  lego_cost = 250 →
  play_dough_cost = 35 →
  lego_sets = 3 →
  toy_swords = 7 →
  play_doughs = 10 →
  ∃ (sword_cost : ℕ),
    sword_cost = 120 ∧
    total_spent = lego_cost * lego_sets + sword_cost * toy_swords + play_dough_cost * play_doughs :=
by sorry

end NUMINAMATH_CALUDE_toy_sword_cost_l1081_108140


namespace NUMINAMATH_CALUDE_dinner_fraction_l1081_108114

theorem dinner_fraction (total_money : ℚ) (ice_cream_cost : ℚ) (money_left : ℚ) :
  total_money = 80 ∧ ice_cream_cost = 18 ∧ money_left = 2 →
  (total_money - ice_cream_cost - money_left) / total_money = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_dinner_fraction_l1081_108114


namespace NUMINAMATH_CALUDE_average_and_difference_l1081_108171

theorem average_and_difference (y : ℝ) : 
  (45 + y) / 2 = 53 → |y - 45| = 16 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l1081_108171


namespace NUMINAMATH_CALUDE_s_iff_q_r_iff_q_p_necessary_for_s_l1081_108118

-- Define the propositions
variable (p q r s : Prop)

-- Define the given conditions
axiom p_necessary_for_r : r → p
axiom q_necessary_for_r : r → q
axiom s_sufficient_for_r : s → r
axiom q_sufficient_for_s : q → s

-- Theorem statements
theorem s_iff_q : s ↔ q := by sorry

theorem r_iff_q : r ↔ q := by sorry

theorem p_necessary_for_s : s → p := by sorry

end NUMINAMATH_CALUDE_s_iff_q_r_iff_q_p_necessary_for_s_l1081_108118


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l1081_108149

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalStudents : ℕ
  groupSize : ℕ
  numGroups : ℕ
  sampleSize : ℕ
  initialSample : ℕ
  initialGroup : ℕ

/-- Given a systematic sampling scheme, calculate the sample from a specific group -/
def sampleFromGroup (s : SystematicSampling) (group : ℕ) : ℕ :=
  s.initialSample + s.groupSize * (group - s.initialGroup)

theorem systematic_sampling_theorem (s : SystematicSampling) 
  (h1 : s.totalStudents = 50)
  (h2 : s.groupSize = 5)
  (h3 : s.numGroups = 10)
  (h4 : s.sampleSize = 10)
  (h5 : s.initialSample = 12)
  (h6 : s.initialGroup = 3) :
  sampleFromGroup s 8 = 37 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l1081_108149


namespace NUMINAMATH_CALUDE_crayon_distribution_l1081_108126

/-- The problem of distributing crayons among Fred, Benny, Jason, and Sarah. -/
theorem crayon_distribution (total : ℕ) (fred benny jason sarah : ℕ) : 
  total = 96 →
  fred = 2 * benny →
  jason = 3 * sarah →
  benny = 12 →
  total = fred + benny + jason + sarah →
  fred = 24 ∧ benny = 12 ∧ jason = 45 ∧ sarah = 15 := by
  sorry

#check crayon_distribution

end NUMINAMATH_CALUDE_crayon_distribution_l1081_108126


namespace NUMINAMATH_CALUDE_right_triangle_tan_l1081_108188

theorem right_triangle_tan (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 13) (h3 : a = 5) :
  b = 12 ∧ a / b = 5 / 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_tan_l1081_108188


namespace NUMINAMATH_CALUDE_three_digit_integers_with_7_no_4_l1081_108146

/-- The set of digits excluding 0, 4, and 7 -/
def digits_no_047 : Finset Nat := Finset.filter (fun d => d ≠ 0 ∧ d ≠ 4 ∧ d ≠ 7) (Finset.range 10)

/-- The set of digits excluding 0 and 4 -/
def digits_no_04 : Finset Nat := Finset.filter (fun d => d ≠ 0 ∧ d ≠ 4) (Finset.range 10)

/-- The set of digits excluding 4 -/
def digits_no_4 : Finset Nat := Finset.filter (fun d => d ≠ 4) (Finset.range 10)

/-- The number of three-digit integers without 7 and 4 -/
def count_no_47 : Nat := digits_no_047.card * digits_no_4.card * digits_no_4.card

/-- The number of three-digit integers without 4 -/
def count_no_4 : Nat := digits_no_04.card * digits_no_4.card * digits_no_4.card

theorem three_digit_integers_with_7_no_4 :
  count_no_4 - count_no_47 = 200 := by sorry

end NUMINAMATH_CALUDE_three_digit_integers_with_7_no_4_l1081_108146


namespace NUMINAMATH_CALUDE_felix_drive_l1081_108155

theorem felix_drive (average_speed : ℝ) (drive_time : ℝ) : 
  average_speed = 66 → drive_time = 4 → (2 * average_speed) * drive_time = 528 := by
  sorry

end NUMINAMATH_CALUDE_felix_drive_l1081_108155


namespace NUMINAMATH_CALUDE_even_decreasing_implies_increasing_l1081_108113

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f y < f x

def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

theorem even_decreasing_implies_increasing
  (f : ℝ → ℝ) (h_even : is_even f) (h_decr : decreasing_on f (Set.Ici 0)) :
  increasing_on f (Set.Iic 0) :=
sorry

end NUMINAMATH_CALUDE_even_decreasing_implies_increasing_l1081_108113


namespace NUMINAMATH_CALUDE_median_salary_is_25000_l1081_108167

/-- Represents a position in the company with its title, number of employees, and salary. -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions and the total number of employees. -/
def medianSalary (positions : List Position) (totalEmployees : Nat) : Nat :=
  sorry

/-- The list of positions in the company. -/
def companyPositions : List Position := [
  { title := "President", count := 1, salary := 140000 },
  { title := "Vice-President", count := 4, salary := 95000 },
  { title := "Director", count := 11, salary := 78000 },
  { title := "Associate Director", count := 8, salary := 55000 },
  { title := "Administrative Specialist", count := 39, salary := 25000 }
]

/-- The total number of employees in the company. -/
def totalEmployees : Nat := 63

/-- Theorem stating that the median salary of the company is $25,000. -/
theorem median_salary_is_25000 : 
  medianSalary companyPositions totalEmployees = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_25000_l1081_108167


namespace NUMINAMATH_CALUDE_quadratic_root_conditions_l1081_108104

-- Define the quadratic equation
def quadratic (a x : ℝ) : ℝ := x^2 + (a^2 + 1) * x + a - 2

-- Define the roots of the quadratic equation
def roots (a : ℝ) : Set ℝ := {x : ℝ | quadratic a x = 0}

-- Theorem statement
theorem quadratic_root_conditions (a : ℝ) :
  (∃ x ∈ roots a, x > 1) ∧ (∃ y ∈ roots a, y < -1) → 0 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_conditions_l1081_108104


namespace NUMINAMATH_CALUDE_factorial_series_diverges_l1081_108162

/-- The series Σ(k!/(2^k)) for k from 1 to infinity -/
def factorial_series (k : ℕ) : ℚ := (Nat.factorial k : ℚ) / (2 ^ k : ℚ)

/-- The statement that the factorial series diverges -/
theorem factorial_series_diverges : ¬ Summable factorial_series := by
  sorry

end NUMINAMATH_CALUDE_factorial_series_diverges_l1081_108162


namespace NUMINAMATH_CALUDE_n_value_l1081_108178

-- Define the cubic polynomial
def cubic_poly (x m : ℝ) : ℝ := x^3 - 3*x^2 + m*x + 24

-- Define the quadratic polynomial
def quad_poly (x n : ℝ) : ℝ := x^2 + n*x - 6

theorem n_value (a b c m n : ℝ) : 
  (cubic_poly a m = 0) ∧ 
  (cubic_poly b m = 0) ∧ 
  (cubic_poly c m = 0) ∧
  (quad_poly (-a) n = 0) ∧ 
  (quad_poly (-b) n = 0) →
  n = -1 := by
sorry

end NUMINAMATH_CALUDE_n_value_l1081_108178


namespace NUMINAMATH_CALUDE_orange_distribution_l1081_108133

theorem orange_distribution (total_oranges : ℕ) (pieces_per_orange : ℕ) (pieces_per_friend : ℕ) : 
  total_oranges = 80 → pieces_per_orange = 10 → pieces_per_friend = 4 →
  (total_oranges * pieces_per_orange) / pieces_per_friend = 200 := by
sorry

end NUMINAMATH_CALUDE_orange_distribution_l1081_108133


namespace NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l1081_108190

/-- The units digit of m^2 + 3^m is 5, where m = 2023^2 + 3^2023 -/
theorem units_digit_of_m_squared_plus_three_to_m (m : ℕ) : 
  m = 2023^2 + 3^2023 → (m^2 + 3^m) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_m_squared_plus_three_to_m_l1081_108190


namespace NUMINAMATH_CALUDE_ned_good_games_l1081_108154

/-- Calculates the number of good games Ned ended up with -/
def good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (non_working_games : ℕ) : ℕ :=
  games_from_friend + games_from_garage_sale - non_working_games

/-- Theorem: Ned ended up with 14 good games -/
theorem ned_good_games : good_games 11 22 19 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ned_good_games_l1081_108154


namespace NUMINAMATH_CALUDE_max_rice_plates_l1081_108123

def chapati_count : ℕ := 16
def chapati_cost : ℕ := 6
def mixed_veg_count : ℕ := 7
def mixed_veg_cost : ℕ := 70
def ice_cream_count : ℕ := 6
def rice_cost : ℕ := 45
def total_paid : ℕ := 985

theorem max_rice_plates (rice_count : ℕ) : 
  rice_count * rice_cost + 
  chapati_count * chapati_cost + 
  mixed_veg_count * mixed_veg_cost ≤ total_paid →
  rice_count ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_max_rice_plates_l1081_108123


namespace NUMINAMATH_CALUDE_smallest_divisible_by_12_15_16_l1081_108153

theorem smallest_divisible_by_12_15_16 :
  ∃ (n : ℕ), n > 0 ∧ 12 ∣ n ∧ 15 ∣ n ∧ 16 ∣ n ∧
  ∀ (m : ℕ), m > 0 → 12 ∣ m → 15 ∣ m → 16 ∣ m → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_12_15_16_l1081_108153


namespace NUMINAMATH_CALUDE_green_dots_third_row_l1081_108139

/-- Represents a sequence of rows with green dots -/
def GreenDotSequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem green_dots_third_row
  (a : ℕ → ℕ)
  (seq : GreenDotSequence a)
  (h1 : a 1 = 3)
  (h2 : a 2 = 6)
  (h4 : a 4 = 12)
  (h5 : a 5 = 15) :
  a 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_green_dots_third_row_l1081_108139


namespace NUMINAMATH_CALUDE_unique_triple_l1081_108156

theorem unique_triple : 
  ∃! (a b c : ℕ), 
    (10 ≤ b ∧ b ≤ 99) ∧ 
    (10 ≤ c ∧ c ≤ 99) ∧ 
    (10^4 * a + 100 * b + c = (a + b + c)^3) ∧
    a = 9 ∧ b = 11 ∧ c = 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_l1081_108156


namespace NUMINAMATH_CALUDE_labyrinth_paths_count_l1081_108172

/-- Represents a point in the labyrinth --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents a direction of movement in the labyrinth --/
inductive Direction
  | Right
  | Down
  | Up

/-- Represents the labyrinth structure --/
structure Labyrinth where
  entrance : Point
  exit : Point
  branchPoints : List Point
  isValidMove : Point → Direction → Bool

/-- Counts the number of paths from a given point to the exit --/
def countPaths (lab : Labyrinth) (start : Point) : ℕ :=
  sorry

/-- The main theorem stating that there are 16 paths in the given labyrinth --/
theorem labyrinth_paths_count (lab : Labyrinth) : 
  countPaths lab lab.entrance = 16 :=
  sorry

end NUMINAMATH_CALUDE_labyrinth_paths_count_l1081_108172


namespace NUMINAMATH_CALUDE_petyas_coins_l1081_108165

theorem petyas_coins (total : ℕ) (not_two : ℕ) (not_ten : ℕ) (not_one : ℕ) 
  (h_total : total = 25)
  (h_not_two : not_two = 19)
  (h_not_ten : not_ten = 20)
  (h_not_one : not_one = 16) :
  total - ((total - not_two) + (total - not_ten) + (total - not_one)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_petyas_coins_l1081_108165


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1081_108125

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 21) :
  Real.sqrt a + Real.sqrt b < 2 * Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1081_108125


namespace NUMINAMATH_CALUDE_boat_speed_l1081_108135

/-- Given a boat that travels 11 km/h along a stream and 5 km/h against the same stream,
    its speed in still water is 8 km/h. -/
theorem boat_speed (along_stream : ℝ) (against_stream : ℝ) (still_water : ℝ)
    (h1 : along_stream = 11)
    (h2 : against_stream = 5)
    (h3 : along_stream = still_water + (along_stream - still_water))
    (h4 : against_stream = still_water - (along_stream - still_water)) :
    still_water = 8 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_l1081_108135


namespace NUMINAMATH_CALUDE_books_per_child_l1081_108132

theorem books_per_child (num_children : ℕ) (teacher_books : ℕ) (total_books : ℕ) :
  num_children = 10 →
  teacher_books = 8 →
  total_books = 78 →
  ∃ (books_per_child : ℕ), books_per_child * num_children + teacher_books = total_books ∧ books_per_child = 7 :=
by sorry

end NUMINAMATH_CALUDE_books_per_child_l1081_108132


namespace NUMINAMATH_CALUDE_point_on_line_m_range_l1081_108128

-- Define the function f
def f (x m n : ℝ) : ℝ := |x - m| + |x + n|

-- Part 1
theorem point_on_line (m n : ℝ) (h1 : m + n > 0) (h2 : ∀ x, f x m n ≥ 2) 
  (h3 : ∃ x, f x m n = 2) : m + n = 2 := by
  sorry

-- Part 2
theorem m_range (m : ℝ) (h : ∀ x ∈ Set.Icc 0 1, f x m 2 ≤ x + 5) : 
  m ∈ Set.Icc (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_m_range_l1081_108128


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1081_108194

theorem complex_modulus_problem (z : ℂ) (h : (3 + 4 * Complex.I) * z = 1) : Complex.abs z = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1081_108194


namespace NUMINAMATH_CALUDE_derivative_of_sin_over_x_l1081_108157

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / x

theorem derivative_of_sin_over_x :
  deriv f = fun x => (x * Real.cos x - Real.sin x) / (x^2) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_sin_over_x_l1081_108157


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_2010_l1081_108195

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_2010 :
  unitsDigit (sumFactorials 2010) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_2010_l1081_108195


namespace NUMINAMATH_CALUDE_quadrilateral_propositions_l1081_108144

-- Define a quadrilateral
structure Quadrilateral :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)
  (side4 : ℝ)

-- Define a property for quadrilaterals with four equal sides
def has_equal_sides (q : Quadrilateral) : Prop :=
  q.side1 = q.side2 ∧ q.side2 = q.side3 ∧ q.side3 = q.side4

-- Define a property for squares
def is_square (q : Quadrilateral) : Prop :=
  has_equal_sides q ∧ q.side1 = q.side2 -- This is a simplified definition

theorem quadrilateral_propositions :
  (∃ q : Quadrilateral, has_equal_sides q ∧ ¬is_square q) ∧
  (∀ q : Quadrilateral, is_square q → has_equal_sides q) ∧
  (∀ q : Quadrilateral, ¬is_square q → ¬has_equal_sides q) ∧
  (∃ q : Quadrilateral, ¬is_square q ∧ has_equal_sides q) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_propositions_l1081_108144


namespace NUMINAMATH_CALUDE_no_solution_for_2015_problems_l1081_108173

theorem no_solution_for_2015_problems : 
  ¬ ∃ (x y z : ℕ), (y - x = z - y) ∧ (x + y + z = 2015) := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_2015_problems_l1081_108173


namespace NUMINAMATH_CALUDE_sum_of_odd_coefficients_l1081_108166

theorem sum_of_odd_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₃ + a₅ = 122 := by
sorry

end NUMINAMATH_CALUDE_sum_of_odd_coefficients_l1081_108166


namespace NUMINAMATH_CALUDE_missing_number_equation_l1081_108183

theorem missing_number_equation (x : ℤ) : 1234562 - 12 * x * 2 = 1234490 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_equation_l1081_108183


namespace NUMINAMATH_CALUDE_jasons_music_store_spending_l1081_108129

/-- The problem of calculating Jason's total spending at the music store -/
theorem jasons_music_store_spending
  (flute_cost : ℝ)
  (music_stand_cost : ℝ)
  (song_book_cost : ℝ)
  (h1 : flute_cost = 142.46)
  (h2 : music_stand_cost = 8.89)
  (h3 : song_book_cost = 7.00) :
  flute_cost + music_stand_cost + song_book_cost = 158.35 := by
  sorry

end NUMINAMATH_CALUDE_jasons_music_store_spending_l1081_108129


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1081_108107

/-- Two lines ax + by + c = 0 and dx + ey + f = 0 are perpendicular if and only if ad + be = 0 -/
def are_perpendicular (a b c d e f : ℝ) : Prop :=
  a * d + b * e = 0

/-- The proposition "a = 2" is neither sufficient nor necessary for the line ax + 3y - 1 = 0
    to be perpendicular to the line 6x + 4y - 3 = 0 -/
theorem not_sufficient_not_necessary : 
  (∃ a : ℝ, a = 2 ∧ ¬(are_perpendicular a 3 (-1) 6 4 (-3))) ∧ 
  (∃ a : ℝ, are_perpendicular a 3 (-1) 6 4 (-3) ∧ a ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1081_108107


namespace NUMINAMATH_CALUDE_least_c_for_triple_f_l1081_108199

def f (x : ℤ) : ℤ :=
  if x % 2 = 1 then x + 5 else x / 2

def is_odd (n : ℤ) : Prop := n % 2 = 1

theorem least_c_for_triple_f (b : ℤ) :
  ∃ c : ℤ, is_odd c ∧ f (f (f c)) = b ∧ ∀ d : ℤ, is_odd d ∧ f (f (f d)) = b → c ≤ d :=
sorry

end NUMINAMATH_CALUDE_least_c_for_triple_f_l1081_108199


namespace NUMINAMATH_CALUDE_jessica_purchases_total_cost_l1081_108106

/-- The cost of Jessica's cat toy in dollars -/
def cat_toy_cost : ℚ := 10.22

/-- The cost of Jessica's cage in dollars -/
def cage_cost : ℚ := 11.73

/-- The total cost of Jessica's purchases in dollars -/
def total_cost : ℚ := cat_toy_cost + cage_cost

/-- Theorem stating that the total cost of Jessica's purchases is $21.95 -/
theorem jessica_purchases_total_cost : total_cost = 21.95 := by
  sorry

end NUMINAMATH_CALUDE_jessica_purchases_total_cost_l1081_108106


namespace NUMINAMATH_CALUDE_thirty_percent_less_eighty_forty_percent_more_l1081_108102

theorem thirty_percent_less_eighty_forty_percent_more (x : ℝ) : 
  (x + 0.4 * x = 80 - 0.3 * 80) → x = 40 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_eighty_forty_percent_more_l1081_108102


namespace NUMINAMATH_CALUDE_sequences_properties_l1081_108160

/-- Definition of the first sequence -/
def seq1 (n : ℕ) : ℤ := (-2)^n

/-- Definition of the second sequence -/
def seq2 (m : ℕ) : ℤ := (-2)^(m-1)

/-- Definition of the third sequence -/
def seq3 (m : ℕ) : ℤ := (-2)^(m-1) - 1

/-- Theorem stating the properties of the sequences -/
theorem sequences_properties :
  (∀ n : ℕ, seq1 n = (-2)^n) ∧
  (∀ m : ℕ, seq3 m = seq2 m - 1) ∧
  (seq1 2019 + seq2 2019 + seq3 2019 = -1) :=
by sorry

end NUMINAMATH_CALUDE_sequences_properties_l1081_108160


namespace NUMINAMATH_CALUDE_abc_base16_to_base4_l1081_108138

/-- Converts a base 16 digit to its decimal representation -/
def hexToDecimal (x : Char) : ℕ :=
  match x with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | _ => 0  -- This case should not occur for our specific problem

/-- Converts a decimal number to its base 4 representation -/
def decimalToBase4 (x : ℕ) : List ℕ :=
  [x / 4, x % 4]

/-- Converts a base 16 number to base 4 -/
def hexToBase4 (x : String) : List ℕ :=
  x.data.map hexToDecimal |>.bind decimalToBase4

theorem abc_base16_to_base4 :
  hexToBase4 "ABC" = [2, 2, 2, 3, 3, 0] := by sorry

end NUMINAMATH_CALUDE_abc_base16_to_base4_l1081_108138


namespace NUMINAMATH_CALUDE_valid_distribution_example_l1081_108164

def is_valid_distribution (probs : List ℚ) : Prop :=
  (probs.sum = 1) ∧ (∀ p ∈ probs, 0 < p ∧ p ≤ 1)

theorem valid_distribution_example : 
  is_valid_distribution [1/2, 1/3, 1/6] := by
  sorry

end NUMINAMATH_CALUDE_valid_distribution_example_l1081_108164


namespace NUMINAMATH_CALUDE_tan_alpha_half_implies_fraction_equals_negative_four_l1081_108103

theorem tan_alpha_half_implies_fraction_equals_negative_four (α : Real) 
  (h : Real.tan α = 1 / 2) : 
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = -4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_half_implies_fraction_equals_negative_four_l1081_108103


namespace NUMINAMATH_CALUDE_pool_filling_time_l1081_108147

/-- Represents the volume of water in a pool as a function of time -/
def water_volume (t : ℕ) : ℝ := sorry

/-- The full capacity of the pool -/
def full_capacity : ℝ := sorry

theorem pool_filling_time :
  (∀ t, water_volume (t + 1) = 2 * water_volume t) →  -- Volume doubles every hour
  (water_volume 8 = full_capacity) →                  -- Full capacity reached in 8 hours
  (water_volume 6 = full_capacity / 2) :=             -- Half capacity reached in 6 hours
by sorry

end NUMINAMATH_CALUDE_pool_filling_time_l1081_108147


namespace NUMINAMATH_CALUDE_wilson_payment_l1081_108145

def hamburger_price : ℕ := 5
def cola_price : ℕ := 2
def hamburger_quantity : ℕ := 2
def cola_quantity : ℕ := 3
def discount : ℕ := 4

def total_cost : ℕ := hamburger_price * hamburger_quantity + cola_price * cola_quantity - discount

theorem wilson_payment : total_cost = 12 := by
  sorry

end NUMINAMATH_CALUDE_wilson_payment_l1081_108145


namespace NUMINAMATH_CALUDE_matrix_power_four_l1081_108174

/-- The fourth power of a specific 2x2 matrix equals a specific result -/
theorem matrix_power_four : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -Real.sqrt 3; Real.sqrt 3, 1]
  A^4 = !![(-8 : ℝ), 8 * Real.sqrt 3; -8 * Real.sqrt 3, (-8 : ℝ)] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_four_l1081_108174


namespace NUMINAMATH_CALUDE_teacher_age_l1081_108112

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 15 →
  student_avg_age = 10 →
  new_avg_age = student_avg_age + 1 →
  (num_students : ℝ) * student_avg_age + (new_avg_age * (num_students + 1) - num_students * student_avg_age) = 26 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l1081_108112


namespace NUMINAMATH_CALUDE_range_of_m_l1081_108193

-- Define p as a proposition depending on m
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0

-- Define q as a proposition depending on m
def q (m : ℝ) : Prop := m > 2

-- Define the set of m satisfying the conditions
def S : Set ℝ := {m : ℝ | p m ∧ ¬(q m) ∧ ¬(¬(p m)) ∧ ¬(p m ∧ q m)}

-- Theorem statement
theorem range_of_m : S = {m : ℝ | 1 < m ∧ m ≤ 2} := by sorry

end NUMINAMATH_CALUDE_range_of_m_l1081_108193


namespace NUMINAMATH_CALUDE_expand_product_l1081_108136

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9*x + 18 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1081_108136


namespace NUMINAMATH_CALUDE_brady_record_theorem_l1081_108110

/-- The minimum average yards per game needed to beat the record -/
def min_avg_yards_per_game (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ) : ℚ :=
  (current_record + 1 - current_yards) / games_left

/-- Theorem stating the minimum average yards per game needed to beat the record -/
theorem brady_record_theorem (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ)
  (h1 : current_record = 5999)
  (h2 : current_yards = 4200)
  (h3 : games_left = 6) :
  min_avg_yards_per_game current_record current_yards games_left = 300 := by
  sorry

end NUMINAMATH_CALUDE_brady_record_theorem_l1081_108110


namespace NUMINAMATH_CALUDE_original_price_calculation_l1081_108122

theorem original_price_calculation (decreased_price : ℝ) (decrease_percentage : ℝ) :
  decreased_price = 620 ∧ decrease_percentage = 20 →
  (100 - decrease_percentage) / 100 * (100 / (100 - decrease_percentage) * decreased_price) = 775 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l1081_108122


namespace NUMINAMATH_CALUDE_quadratic_function_sum_l1081_108111

theorem quadratic_function_sum (a b c : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + 2 ≤ a*x^2 + b*x + c) ∧
  (∀ x : ℝ, a*x^2 + b*x + c ≤ 2*x^2 - 4*x + 3) →
  a + b + c = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_sum_l1081_108111


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1081_108192

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 1) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1081_108192


namespace NUMINAMATH_CALUDE_linear_function_properties_l1081_108176

-- Define the linear function
def f (k : ℝ) (x : ℝ) : ℝ := k * (x + 3)

-- Define the shifted function
def g (k : ℝ) (x : ℝ) : ℝ := f k x + 2

theorem linear_function_properties (k : ℝ) (h : k ≠ 0) :
  (f k (-3) = 0) ∧ 
  (g k 1 = -2 → k = -1) ∧
  (k < 0 → ∀ x₁ x₂ y₁ y₂ : ℝ, 
    f k x₁ = y₁ → f k x₂ = y₂ → y₁ < y₂ → x₁ > x₂) :=
by sorry


end NUMINAMATH_CALUDE_linear_function_properties_l1081_108176


namespace NUMINAMATH_CALUDE_nearest_town_distance_l1081_108169

theorem nearest_town_distance (d : ℝ) : 
  (¬ (d ≥ 8)) ∧ 
  (¬ (d ≤ 7)) ∧ 
  (¬ (d ≤ 6)) ∧ 
  (¬ (d ≥ 9)) →
  d ∈ Set.Ioo 7 8 :=
by sorry

end NUMINAMATH_CALUDE_nearest_town_distance_l1081_108169


namespace NUMINAMATH_CALUDE_electrolysis_mass_proportionality_l1081_108116

/-- Represents the mass of metal deposited during electrolysis -/
noncomputable def mass_deposited (current : ℝ) (time : ℝ) (ion_charge : ℝ) : ℝ :=
  sorry

/-- The mass deposited is directly proportional to the current -/
axiom mass_prop_current (time : ℝ) (ion_charge : ℝ) (k : ℝ) :
  ∀ current₁ current₂ : ℝ, mass_deposited (k * current₁) time ion_charge = k * mass_deposited current₂ time ion_charge

/-- The mass deposited is directly proportional to the time -/
axiom mass_prop_time (current : ℝ) (ion_charge : ℝ) (k : ℝ) :
  ∀ time₁ time₂ : ℝ, mass_deposited current (k * time₁) ion_charge = k * mass_deposited current time₂ ion_charge

/-- The mass deposited is inversely proportional to the ion charge -/
axiom mass_inv_prop_charge (current : ℝ) (time : ℝ) (k : ℝ) :
  ∀ charge₁ charge₂ : ℝ, charge₁ ≠ 0 → charge₂ ≠ 0 →
    mass_deposited current time (k * charge₁) = (1 / k) * mass_deposited current time charge₂

theorem electrolysis_mass_proportionality :
  (∀ k current time charge, mass_deposited (k * current) time charge = k * mass_deposited current time charge) ∧
  (∀ k current time charge, mass_deposited current (k * time) charge = k * mass_deposited current time charge) ∧
  ¬(∀ k current time charge, charge ≠ 0 → mass_deposited current time (k * charge) = k * mass_deposited current time charge) :=
by sorry

end NUMINAMATH_CALUDE_electrolysis_mass_proportionality_l1081_108116


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l1081_108115

theorem product_from_lcm_gcd (a b : ℤ) : 
  Int.lcm a b = 42 → Int.gcd a b = 7 → a * b = 294 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l1081_108115


namespace NUMINAMATH_CALUDE_expression_evaluation_l1081_108151

theorem expression_evaluation :
  let m : ℚ := -1/2
  let f (x : ℚ) := (5 / (x - 2) - x - 2) * ((2 * x - 4) / (3 - x))
  f m = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1081_108151


namespace NUMINAMATH_CALUDE_snake_length_problem_l1081_108184

theorem snake_length_problem (penny_snake : ℕ) (jake_snake : ℕ) : 
  jake_snake = penny_snake + 12 →
  jake_snake + penny_snake = 70 →
  jake_snake = 41 := by
sorry

end NUMINAMATH_CALUDE_snake_length_problem_l1081_108184


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1081_108152

-- Define the rates of the pipes
def rateA : ℚ := 1 / 12
def rateB : ℚ := 1 / 18
def rateC : ℚ := -(1 / 15)

-- Define the combined rate
def combinedRate : ℚ := rateA + rateB + rateC

-- Define the time to fill the cistern
def timeToFill : ℚ := 1 / combinedRate

-- Theorem statement
theorem cistern_fill_time :
  timeToFill = 180 / 13 :=
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1081_108152


namespace NUMINAMATH_CALUDE_fifteenth_prime_l1081_108109

theorem fifteenth_prime (p : ℕ → ℕ) (h : ∀ n, Prime (p n)) (h15 : p 7 = 15) : p 15 = 47 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l1081_108109


namespace NUMINAMATH_CALUDE_solve_fruit_salad_problem_l1081_108163

def fruit_salad_problem (alaya_salads : ℕ) (angel_multiplier : ℕ) : Prop :=
  let angel_salads := angel_multiplier * alaya_salads
  let total_salads := alaya_salads + angel_salads
  total_salads = 600

theorem solve_fruit_salad_problem :
  fruit_salad_problem 200 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_fruit_salad_problem_l1081_108163


namespace NUMINAMATH_CALUDE_quadratic_solution_l1081_108131

theorem quadratic_solution (x : ℝ) (h1 : x^2 - 3*x - 6 = 0) (h2 : x ≠ 0) :
  x = (3 + Real.sqrt 33) / 2 ∨ x = (3 - Real.sqrt 33) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1081_108131


namespace NUMINAMATH_CALUDE_horners_method_for_f_l1081_108141

def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

theorem horners_method_for_f :
  f 3 = 21324 := by
sorry

end NUMINAMATH_CALUDE_horners_method_for_f_l1081_108141


namespace NUMINAMATH_CALUDE_fred_change_l1081_108121

def movie_ticket_cost : ℝ := 5.92
def movie_tickets : ℕ := 3
def movie_rental : ℝ := 6.79
def snacks : ℝ := 10.50
def parking : ℝ := 3.25
def paid_amount : ℝ := 50

def total_cost : ℝ := movie_ticket_cost * movie_tickets + movie_rental + snacks + parking

def change : ℝ := paid_amount - total_cost

theorem fred_change : change = 11.70 := by sorry

end NUMINAMATH_CALUDE_fred_change_l1081_108121


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l1081_108170

theorem salary_reduction_percentage (initial_increase : Real) (net_increase : Real) (reduction : Real) : 
  initial_increase = 0.25 →
  net_increase = 0.0625 →
  (1 + initial_increase) * (1 - reduction) = 1 + net_increase →
  reduction = 0.15 := by
sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l1081_108170


namespace NUMINAMATH_CALUDE_trig_identity_l1081_108175

theorem trig_identity : 4 * Real.cos (50 * π / 180) - Real.tan (40 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l1081_108175


namespace NUMINAMATH_CALUDE_no_valid_numbers_l1081_108105

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧  -- 3-digit number
  (n / 100 + (n / 10) % 10 + n % 10 = 27) ∧  -- digit-sum is 27
  n % 2 = 0 ∧  -- even number
  n % 10 = 4  -- ends in 4

theorem no_valid_numbers : ¬∃ (n : ℕ), is_valid_number n := by
  sorry

end NUMINAMATH_CALUDE_no_valid_numbers_l1081_108105
