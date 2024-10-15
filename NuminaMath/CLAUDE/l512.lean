import Mathlib

namespace NUMINAMATH_CALUDE_milly_study_time_l512_51254

/-- Calculates the total study time for Milly given her homework durations. -/
theorem milly_study_time (math_time : ℕ) (math_time_eq : math_time = 60) :
  let geography_time := math_time / 2
  let science_time := (math_time + geography_time) / 2
  math_time + geography_time + science_time = 135 := by
  sorry

end NUMINAMATH_CALUDE_milly_study_time_l512_51254


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l512_51270

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The general term of an arithmetic sequence. -/
def arithmetic_general_term (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a 1 + (n - 1) * (a 2 - a 1)

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum1 : a 2 + a 6 = 8)
  (h_sum2 : a 3 + a 4 = 3) :
  ∀ n : ℕ, arithmetic_general_term a n = 5 * n - 16 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l512_51270


namespace NUMINAMATH_CALUDE_coffee_price_proof_l512_51237

/-- The regular price of coffee in dollars per pound -/
def regular_price : ℝ := 40

/-- The discount rate as a decimal -/
def discount_rate : ℝ := 0.6

/-- The price of a discounted quarter-pound package with a free chocolate bar -/
def discounted_quarter_pound_price : ℝ := 4

theorem coffee_price_proof :
  regular_price * (1 - discount_rate) / 4 = discounted_quarter_pound_price :=
by sorry

end NUMINAMATH_CALUDE_coffee_price_proof_l512_51237


namespace NUMINAMATH_CALUDE_average_of_five_quantities_l512_51229

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 14) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_of_five_quantities_l512_51229


namespace NUMINAMATH_CALUDE_inequalities_hold_l512_51213

theorem inequalities_hold (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  a^2 + b^2 ≥ 2 ∧ 1/a + 1/b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l512_51213


namespace NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l512_51247

/-- Given three sticks of lengths 9, 18, and 21 inches, this theorem proves that
    the minimum integral length that can be cut from each stick to prevent
    the remaining pieces from forming a triangle is 6 inches. -/
theorem min_cut_length_for_non_triangle : ∃ (x : ℕ),
  (∀ y : ℕ, y < x → (9 - y) + (18 - y) > 21 - y) ∧
  (9 - x) + (18 - x) ≤ 21 - x ∧
  x = 6 :=
sorry

end NUMINAMATH_CALUDE_min_cut_length_for_non_triangle_l512_51247


namespace NUMINAMATH_CALUDE_inverse_variation_proof_l512_51282

/-- Given that y^4 varies inversely with z^2 and y = 3 when z = 1, prove that y = √3 when z = 3 -/
theorem inverse_variation_proof (y z : ℝ) (h1 : ∃ k : ℝ, ∀ y z, y^4 * z^2 = k) 
  (h2 : ∃ y₀ z₀, y₀ = 3 ∧ z₀ = 1 ∧ y₀^4 * z₀^2 = (3 : ℝ)^4 * 1^2) :
  ∃ y₁, y₁^4 * 3^2 = 3^4 * 1^2 ∧ y₁ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_proof_l512_51282


namespace NUMINAMATH_CALUDE_sharp_value_theorem_l512_51228

/-- Define the function # -/
def sharp (k : ℚ) (p : ℚ) : ℚ := k * p + 20

/-- Main theorem -/
theorem sharp_value_theorem :
  ∀ k : ℚ, 
  (sharp k (sharp k (sharp k 18)) = -4) → 
  k = -4/3 := by
sorry

end NUMINAMATH_CALUDE_sharp_value_theorem_l512_51228


namespace NUMINAMATH_CALUDE_mary_screw_sections_l512_51277

def number_of_sections (initial_screws : ℕ) (multiplier : ℕ) (screws_per_section : ℕ) : ℕ :=
  (initial_screws + initial_screws * multiplier) / screws_per_section

theorem mary_screw_sections :
  number_of_sections 8 2 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_mary_screw_sections_l512_51277


namespace NUMINAMATH_CALUDE_system_of_equations_l512_51231

theorem system_of_equations (x y : ℚ) 
  (eq1 : 4 * x + y = 8) 
  (eq2 : 3 * x - 4 * y = 5) : 
  7 * x - 3 * y = 247 / 19 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l512_51231


namespace NUMINAMATH_CALUDE_circular_permutation_sum_l512_51208

def CircularPermutation (xs : List ℕ) : Prop :=
  xs.length = 6 ∧ xs.toFinset = {1, 2, 3, 4, 6}

def CircularProduct (xs : List ℕ) : ℕ :=
  (List.zip xs (xs.rotate 1)).map (λ (a, b) => a * b) |>.sum

def MaxCircularProduct : ℕ := sorry

def MaxCircularProductPermutations : ℕ := sorry

theorem circular_permutation_sum :
  MaxCircularProduct + MaxCircularProductPermutations = 96 := by sorry

end NUMINAMATH_CALUDE_circular_permutation_sum_l512_51208


namespace NUMINAMATH_CALUDE_susans_purchase_l512_51255

/-- Given Susan's purchase scenario, prove the number of 50-cent items -/
theorem susans_purchase (x y z : ℕ) : 
  x + y + z = 50 →  -- total number of items
  50 * x + 300 * y + 500 * z = 10000 →  -- total price in cents
  x = 40  -- number of 50-cent items
:= by sorry

end NUMINAMATH_CALUDE_susans_purchase_l512_51255


namespace NUMINAMATH_CALUDE_affected_days_in_factory_l512_51215

/-- Proves the number of affected days in a TV factory --/
theorem affected_days_in_factory (first_25_avg : ℝ) (overall_avg : ℝ) (affected_avg : ℝ)
  (h1 : first_25_avg = 60)
  (h2 : overall_avg = 58)
  (h3 : affected_avg = 48) :
  ∃ x : ℝ, x = 5 ∧ 25 * first_25_avg + x * affected_avg = (25 + x) * overall_avg :=
by sorry

end NUMINAMATH_CALUDE_affected_days_in_factory_l512_51215


namespace NUMINAMATH_CALUDE_closest_to_zero_l512_51292

def integers : List Int := [-1101, 1011, -1010, -1001, 1110]

theorem closest_to_zero (n : Int) (h : n ∈ integers) : 
  ∀ m ∈ integers, |n| ≤ |m| ↔ n = -1001 :=
by
  sorry

#check closest_to_zero

end NUMINAMATH_CALUDE_closest_to_zero_l512_51292


namespace NUMINAMATH_CALUDE_angle_bisector_length_l512_51244

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 9 ∧ BC = 12 ∧ AC = 15

-- Define the angle bisector CD
def is_angle_bisector (A B C D : ℝ × ℝ) : Prop :=
  let BD := Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  BD / AD = 12 / 15

-- Theorem statement
theorem angle_bisector_length (A B C D : ℝ × ℝ) :
  triangle_ABC A B C → is_angle_bisector A B C D →
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 4 * Real.sqrt 10 :=
by
  sorry


end NUMINAMATH_CALUDE_angle_bisector_length_l512_51244


namespace NUMINAMATH_CALUDE_angle_bisector_theorem_application_l512_51206

theorem angle_bisector_theorem_application (DE DF EF D₁F D₁E XY XZ YZ X₁Z X₁Y XX₁ : ℝ) : 
  DE = 13 →
  DF = 5 →
  EF = (DE^2 - DF^2).sqrt →
  D₁F / D₁E = DF / EF →
  D₁F + D₁E = EF →
  XY = D₁E →
  XZ = D₁F →
  YZ = (XY^2 - XZ^2).sqrt →
  X₁Z / X₁Y = XZ / XY →
  X₁Z + X₁Y = YZ →
  XX₁ = XZ - X₁Z →
  XX₁ = 0 := by
sorry

#eval "QED"

end NUMINAMATH_CALUDE_angle_bisector_theorem_application_l512_51206


namespace NUMINAMATH_CALUDE_solution_set_l512_51268

theorem solution_set (x : ℝ) : 
  (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Icc (-4) (-3/2) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_l512_51268


namespace NUMINAMATH_CALUDE_orange_juice_distribution_l512_51236

theorem orange_juice_distribution (C : ℝ) (h : C > 0) : 
  let juice_volume := (2 / 3) * C
  let num_cups := 6
  let juice_per_cup := juice_volume / num_cups
  juice_per_cup / C * 100 = 100 / 9 := by sorry

end NUMINAMATH_CALUDE_orange_juice_distribution_l512_51236


namespace NUMINAMATH_CALUDE_condition_type_l512_51241

theorem condition_type (a : ℝ) : 
  (∀ x : ℝ, x > 2 → x^2 > 2*x) ∧ 
  (∃ y : ℝ, y ≤ 2 ∧ y^2 > 2*y) :=
by sorry

end NUMINAMATH_CALUDE_condition_type_l512_51241


namespace NUMINAMATH_CALUDE_mod_congruence_l512_51246

theorem mod_congruence (m : ℕ) : 
  (65 * 90 * 111 ≡ m [ZMOD 20]) → 
  (0 ≤ m ∧ m < 20) → 
  m = 10 := by
  sorry

end NUMINAMATH_CALUDE_mod_congruence_l512_51246


namespace NUMINAMATH_CALUDE_sixth_root_of_12984301300421_l512_51232

theorem sixth_root_of_12984301300421 : 
  (12984301300421 : ℝ) ^ (1/6 : ℝ) = 51 := by sorry

end NUMINAMATH_CALUDE_sixth_root_of_12984301300421_l512_51232


namespace NUMINAMATH_CALUDE_part_one_part_two_l512_51279

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + 4*m*x - 5*m^2 < 0}

-- Part 1: Prove that when B = {x | -5 < x < 1}, m = 1
theorem part_one : 
  (B 1 = {x | -5 < x ∧ x < 1}) → 1 = 1 := by sorry

-- Part 2: Prove that when A ⊆ B, m ≤ -1 or m ≥ 4
theorem part_two (m : ℝ) : 
  A ⊆ B m → m ≤ -1 ∨ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l512_51279


namespace NUMINAMATH_CALUDE_min_intersection_size_l512_51287

theorem min_intersection_size (U B P : Finset ℕ) 
  (h1 : U.card = 25)
  (h2 : B ⊆ U)
  (h3 : P ⊆ U)
  (h4 : B.card = 15)
  (h5 : P.card = 18) :
  (B ∩ P).card ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_min_intersection_size_l512_51287


namespace NUMINAMATH_CALUDE_remainder_14_div_5_l512_51298

theorem remainder_14_div_5 : 14 % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_14_div_5_l512_51298


namespace NUMINAMATH_CALUDE_door_probability_l512_51286

/-- The probability of exactly k successes in n independent trials 
    with probability p for each trial -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem door_probability : 
  binomial_probability 5 2 (1/2) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_door_probability_l512_51286


namespace NUMINAMATH_CALUDE_tangencyTriangleAreaTheorem_l512_51207

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a triangle formed by the points of tangency of three circles -/
structure TangencyTriangle where
  c1 : Circle
  c2 : Circle
  c3 : Circle

/-- The area of the triangle formed by the points of tangency of three mutually externally tangent circles -/
def tangencyTriangleArea (t : TangencyTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of the triangle formed by the points of tangency
    of three mutually externally tangent circles with radii 1, 3, and 5 is 5/3 -/
theorem tangencyTriangleAreaTheorem :
  let c1 : Circle := { radius := 1 }
  let c2 : Circle := { radius := 3 }
  let c3 : Circle := { radius := 5 }
  let t : TangencyTriangle := { c1 := c1, c2 := c2, c3 := c3 }
  tangencyTriangleArea t = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_tangencyTriangleAreaTheorem_l512_51207


namespace NUMINAMATH_CALUDE_expression_equality_l512_51211

theorem expression_equality : 
  Real.sqrt 16 - 4 * (Real.sqrt 2 / 2) + abs (-Real.sqrt 3 * Real.sqrt 6) + (-1)^2023 = 3 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l512_51211


namespace NUMINAMATH_CALUDE_log_difference_equals_four_l512_51248

theorem log_difference_equals_four (a : ℝ) (h : a > 0) :
  Real.log (100 * a) / Real.log 10 - Real.log (a / 100) / Real.log 10 = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_four_l512_51248


namespace NUMINAMATH_CALUDE_circle_equation_k_range_l512_51258

theorem circle_equation_k_range (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 - 4*x + 4*y + 10 - k = 0) → k > 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_k_range_l512_51258


namespace NUMINAMATH_CALUDE_fourth_segment_length_l512_51251

/-- Represents an acute triangle with two altitudes dividing opposite sides -/
structure AcuteTriangleWithAltitudes where
  -- Lengths of segments created by altitudes
  segment1 : ℝ
  segment2 : ℝ
  segment3 : ℝ
  segment4 : ℝ
  -- Conditions
  acute : segment1 > 0 ∧ segment2 > 0 ∧ segment3 > 0 ∧ segment4 > 0
  segment1_eq : segment1 = 4
  segment2_eq : segment2 = 6
  segment3_eq : segment3 = 3

/-- Theorem stating that the fourth segment length is 3 -/
theorem fourth_segment_length (t : AcuteTriangleWithAltitudes) : t.segment4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_segment_length_l512_51251


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l512_51294

theorem sum_of_coefficients_equals_one (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^4 = a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₀ + a₁ + a₂ + a₃ + a₄ = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l512_51294


namespace NUMINAMATH_CALUDE_water_balloon_problem_l512_51261

theorem water_balloon_problem (janice randy cynthia : ℕ) : 
  cynthia = 4 * randy →
  randy = janice / 2 →
  cynthia + randy = janice + 12 →
  janice = 8 := by
sorry

end NUMINAMATH_CALUDE_water_balloon_problem_l512_51261


namespace NUMINAMATH_CALUDE_min_sum_squares_l512_51256

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 1) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b c : ℝ), a^3 + b^3 + c^3 - 3*a*b*c = 1 →
    x^2 + y^2 + z^2 ≥ m ∧ m ≤ a^2 + b^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l512_51256


namespace NUMINAMATH_CALUDE_ear_muffs_before_december_count_l512_51280

/-- The number of ear muffs bought before December -/
def ear_muffs_before_december (total : ℕ) (during_december : ℕ) : ℕ :=
  total - during_december

/-- Theorem stating that the number of ear muffs bought before December is 1346 -/
theorem ear_muffs_before_december_count :
  ear_muffs_before_december 7790 6444 = 1346 := by
  sorry

end NUMINAMATH_CALUDE_ear_muffs_before_december_count_l512_51280


namespace NUMINAMATH_CALUDE_total_gain_percentage_approx_l512_51216

/-- Calculates the total gain percentage for three items given their purchase and sale prices -/
def total_gain_percentage (cycle_cp cycle_sp scooter_cp scooter_sp skateboard_cp skateboard_sp : ℚ) : ℚ :=
  let total_gain := (cycle_sp - cycle_cp) + (scooter_sp - scooter_cp) + (skateboard_sp - skateboard_cp)
  let total_cost := cycle_cp + scooter_cp + skateboard_cp
  (total_gain / total_cost) * 100

/-- The total gain percentage for the given items is approximately 28.18% -/
theorem total_gain_percentage_approx :
  ∃ ε > 0, abs (total_gain_percentage 900 1260 4500 5400 1200 1800 - 2818/100) < ε :=
sorry

end NUMINAMATH_CALUDE_total_gain_percentage_approx_l512_51216


namespace NUMINAMATH_CALUDE_triangle_formation_constraint_l512_51252

/-- A line in 2D space represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines form a triangle -/
def form_triangle (l1 l2 l3 : Line) : Prop :=
  ∃ (x1 y1 x2 y2 x3 y3 : ℝ),
    (l1.a * x1 + l1.b * y1 = l1.c) ∧
    (l2.a * x2 + l2.b * y2 = l2.c) ∧
    (l3.a * x3 + l3.b * y3 = l3.c) ∧
    ((x1 ≠ x2) ∨ (y1 ≠ y2)) ∧
    ((x2 ≠ x3) ∨ (y2 ≠ y3)) ∧
    ((x3 ≠ x1) ∨ (y3 ≠ y1))

theorem triangle_formation_constraint (a : ℝ) :
  let l1 : Line := ⟨1, 1, 0⟩
  let l2 : Line := ⟨1, -1, 0⟩
  let l3 : Line := ⟨1, a, 3⟩
  form_triangle l1 l2 l3 → a ≠ 1 ∧ a ≠ -1 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_constraint_l512_51252


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l512_51218

def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {-2, 0, 1}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l512_51218


namespace NUMINAMATH_CALUDE_average_shirts_sold_per_day_l512_51281

theorem average_shirts_sold_per_day 
  (morning_day1 : ℕ) 
  (afternoon_day1 : ℕ) 
  (day2 : ℕ) 
  (h1 : morning_day1 = 250) 
  (h2 : afternoon_day1 = 20) 
  (h3 : day2 = 320) : 
  (morning_day1 + afternoon_day1 + day2) / 2 = 295 := by
sorry

end NUMINAMATH_CALUDE_average_shirts_sold_per_day_l512_51281


namespace NUMINAMATH_CALUDE_expand_product_l512_51275

theorem expand_product (x : ℝ) : (x - 3) * (x + 4) = x^2 + x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l512_51275


namespace NUMINAMATH_CALUDE_euler_most_prolific_l512_51230

/-- Represents a mathematician -/
structure Mathematician where
  name : String
  country : String
  published_volumes : ℕ

/-- The Swiss Society of Natural Sciences -/
def SwissSocietyOfNaturalSciences : Set Mathematician := sorry

/-- Leonhard Euler -/
def euler : Mathematician := {
  name := "Leonhard Euler",
  country := "Switzerland",
  published_volumes := 76  -- More than 75 volumes
}

/-- Predicate for being the most prolific mathematician -/
def most_prolific (m : Mathematician) : Prop :=
  ∀ n : Mathematician, n.published_volumes ≤ m.published_volumes

theorem euler_most_prolific :
  euler ∈ SwissSocietyOfNaturalSciences →
  euler.country = "Switzerland" →
  euler.published_volumes > 75 →
  most_prolific euler :=
sorry

end NUMINAMATH_CALUDE_euler_most_prolific_l512_51230


namespace NUMINAMATH_CALUDE_smallest_odd_five_primes_proof_l512_51203

/-- The smallest odd number with five different prime factors -/
def smallest_odd_five_primes : ℕ := 15015

/-- The list of prime factors of the smallest odd number with five different prime factors -/
def prime_factors : List ℕ := [3, 5, 7, 11, 13]

theorem smallest_odd_five_primes_proof :
  (smallest_odd_five_primes % 2 = 1) ∧
  (List.length prime_factors = 5) ∧
  (List.all prime_factors Nat.Prime) ∧
  (List.prod prime_factors = smallest_odd_five_primes) ∧
  (∀ n : ℕ, n < smallest_odd_five_primes →
    n % 2 = 1 →
    (∃ factors : List ℕ, List.all factors Nat.Prime ∧
      List.prod factors = n ∧
      List.length factors < 5)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_odd_five_primes_proof_l512_51203


namespace NUMINAMATH_CALUDE_unseen_corner_color_code_l512_51272

/-- Represents the colors of a Rubik's Cube -/
inductive Color
  | White
  | Yellow
  | Green
  | Blue
  | Orange
  | Red

/-- Represents a corner piece of a Rubik's Cube -/
structure Corner :=
  (c1 c2 c3 : Color)

/-- Assigns a numeric code to each color -/
def color_code (c : Color) : ℕ :=
  match c with
  | Color.White => 1
  | Color.Yellow => 2
  | Color.Green => 3
  | Color.Blue => 4
  | Color.Orange => 5
  | Color.Red => 6

/-- Represents the state of a Rubik's Cube -/
structure RubiksCube :=
  (corners : List Corner)

/-- Represents a solved Rubik's Cube -/
def solved_cube : RubiksCube := sorry

/-- Represents a scrambled Rubik's Cube with 7 visible corners -/
def scrambled_cube : RubiksCube := sorry

theorem unseen_corner_color_code :
  ∀ (cube : RubiksCube),
    (cube.corners.length = 8) →
    (∃ (visible_corners : List Corner), visible_corners.length = 7 ∧ visible_corners ⊆ cube.corners) →
    ∃ (unseen_corner : Corner),
      unseen_corner ∈ cube.corners ∧
      unseen_corner ∉ (visible_corners : List Corner) ∧
      color_code (unseen_corner.c1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_unseen_corner_color_code_l512_51272


namespace NUMINAMATH_CALUDE_triangle_shape_l512_51221

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = π) →
  (a^2 + b^2 + c^2 = 2 * Real.sqrt 3 * a * b * Real.sin C) →
  (a = b ∧ b = c ∧ A = B ∧ B = C ∧ C = π/3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_shape_l512_51221


namespace NUMINAMATH_CALUDE_fraction_subtraction_l512_51225

theorem fraction_subtraction : (18 : ℚ) / 45 - 3 / 8 = 1 / 40 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l512_51225


namespace NUMINAMATH_CALUDE_max_PXQ_value_l512_51240

def is_two_digit_with_equal_digits (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ (n / 10 = n % 10)

def is_one_digit (n : ℕ) : Prop :=
  0 < n ∧ n ≤ 9

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem max_PXQ_value :
  ∀ X XX PXQ : ℕ,
  is_two_digit_with_equal_digits XX →
  is_one_digit X →
  is_three_digit PXQ →
  XX * X = PXQ →
  PXQ ≤ 396 :=
sorry

end NUMINAMATH_CALUDE_max_PXQ_value_l512_51240


namespace NUMINAMATH_CALUDE_mother_daughter_age_ratio_l512_51259

/-- Given a mother who is 27 years older than her daughter and is currently 55 years old,
    prove that the ratio of their ages one year ago was 2:1. -/
theorem mother_daughter_age_ratio : 
  ∀ (mother_age daughter_age : ℕ),
  mother_age = 55 →
  mother_age = daughter_age + 27 →
  (mother_age - 1) / (daughter_age - 1) = 2 := by
sorry

end NUMINAMATH_CALUDE_mother_daughter_age_ratio_l512_51259


namespace NUMINAMATH_CALUDE_smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l512_51296

theorem smallest_integer_fraction (y : ℤ) : (8 : ℚ) / 11 < (y : ℚ) / 17 → y ≥ 13 :=
by
  sorry

theorem thirteen_satisfies (y : ℤ) : (8 : ℚ) / 11 < (13 : ℚ) / 17 :=
by
  sorry

theorem smallest_integer_is_thirteen : ∃ y : ℤ, ((8 : ℚ) / 11 < (y : ℚ) / 17) ∧ (∀ z : ℤ, (8 : ℚ) / 11 < (z : ℚ) / 17 → z ≥ y) ∧ y = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_fraction_thirteen_satisfies_smallest_integer_is_thirteen_l512_51296


namespace NUMINAMATH_CALUDE_stock_price_theorem_l512_51290

/-- The stock price after three years of changes -/
def stock_price_after_three_years (initial_price : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + 0.8)
  let price_after_second_year := price_after_first_year * (1 - 0.3)
  let price_after_third_year := price_after_second_year * (1 + 0.5)
  price_after_third_year

/-- Theorem stating that the stock price after three years is $226.8 -/
theorem stock_price_theorem :
  stock_price_after_three_years 120 = 226.8 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_theorem_l512_51290


namespace NUMINAMATH_CALUDE_pyramid_theorem_l512_51264

/-- A regular triangular pyramid with an inscribed sphere -/
structure RegularPyramidWithSphere where
  /-- The side length of the base triangle -/
  base_side : ℝ
  /-- The radius of the inscribed sphere -/
  sphere_radius : ℝ
  /-- The sphere is inscribed at the midpoint of the pyramid's height -/
  sphere_at_midpoint : True
  /-- The sphere touches the lateral faces of the pyramid -/
  sphere_touches_faces : True
  /-- A hemisphere supported by the inscribed circle in the base touches the sphere externally -/
  hemisphere_touches_sphere : True

/-- Properties of the regular triangular pyramid with inscribed sphere -/
def pyramid_properties (p : RegularPyramidWithSphere) : Prop :=
  p.sphere_radius = 1 ∧
  p.base_side = 2 * Real.sqrt 3 * (Real.sqrt 5 + 1)

/-- The lateral surface area of the pyramid -/
noncomputable def lateral_surface_area (p : RegularPyramidWithSphere) : ℝ :=
  3 * Real.sqrt 15 * (Real.sqrt 5 + 1)

/-- The angle between lateral faces of the pyramid -/
noncomputable def lateral_face_angle (p : RegularPyramidWithSphere) : ℝ :=
  Real.arccos (1 / Real.sqrt 5)

/-- Theorem stating the properties of the pyramid -/
theorem pyramid_theorem (p : RegularPyramidWithSphere) 
  (h : pyramid_properties p) :
  lateral_surface_area p = 3 * Real.sqrt 15 * (Real.sqrt 5 + 1) ∧
  lateral_face_angle p = Real.arccos (1 / Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_pyramid_theorem_l512_51264


namespace NUMINAMATH_CALUDE_f_order_l512_51289

def f (x : ℝ) : ℝ := -x^2 + 2

theorem f_order : f (-2) < f 1 ∧ f 1 < f 0 :=
  by sorry

end NUMINAMATH_CALUDE_f_order_l512_51289


namespace NUMINAMATH_CALUDE_douglas_weight_is_52_l512_51263

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- The difference in weight between Anne and Douglas in pounds -/
def weight_difference : ℕ := 15

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := anne_weight - weight_difference

/-- Theorem stating Douglas's weight -/
theorem douglas_weight_is_52 : douglas_weight = 52 := by
  sorry

end NUMINAMATH_CALUDE_douglas_weight_is_52_l512_51263


namespace NUMINAMATH_CALUDE_only_fourth_equation_has_real_roots_l512_51233

-- Define the discriminant function
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Define a function to check if a quadratic equation has real roots
def hasRealRoots (a b c : ℝ) : Prop := discriminant a b c ≥ 0

-- Theorem statement
theorem only_fourth_equation_has_real_roots :
  ¬(hasRealRoots 1 0 1) ∧
  ¬(hasRealRoots 1 1 1) ∧
  ¬(hasRealRoots 1 (-1) 1) ∧
  hasRealRoots 1 (-1) (-1) :=
sorry

end NUMINAMATH_CALUDE_only_fourth_equation_has_real_roots_l512_51233


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l512_51293

/-- An isosceles triangle with sides 4 and 9 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c =>
    (a = 4 ∨ a = 9) ∧  -- One side is either 4 or 9
    (b = a) ∧          -- The triangle is isosceles
    (c = if a = 4 then 9 else 4) ∧  -- The third side is whichever of 4 or 9 that a is not
    (a + b + c = 22)   -- The perimeter is 22

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof :
  ∃ a b c, isosceles_triangle_perimeter a b c :=
by
  sorry  -- The proof is omitted as per instructions

#check isosceles_triangle_perimeter
#check isosceles_triangle_perimeter_proof

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l512_51293


namespace NUMINAMATH_CALUDE_college_students_count_l512_51226

theorem college_students_count :
  ∀ (total : ℕ) (enrolled_percent : ℚ) (not_enrolled : ℕ),
    enrolled_percent = 1/2 →
    not_enrolled = 440 →
    (1 - enrolled_percent) * total = not_enrolled →
    total = 880 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l512_51226


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_k_value_when_x_is_two_l512_51260

/-- The quadratic equation x^2 - kx + k - 1 = 0 -/
def quadratic (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x + k - 1

theorem quadratic_always_real_roots (k : ℝ) :
  ∃ x : ℝ, quadratic k x = 0 :=
sorry

theorem k_value_when_x_is_two :
  ∃ k : ℝ, quadratic k 2 = 0 ∧ k = 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_k_value_when_x_is_two_l512_51260


namespace NUMINAMATH_CALUDE_ceiling_squared_fraction_l512_51212

theorem ceiling_squared_fraction : ⌈((-7/4 + 1/4) : ℚ)^2⌉ = 3 := by sorry

end NUMINAMATH_CALUDE_ceiling_squared_fraction_l512_51212


namespace NUMINAMATH_CALUDE_max_intersection_points_circle_triangle_l512_51239

/-- A circle in a plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A triangle in a plane --/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- The number of intersection points between a circle and a line segment --/
def intersectionPointsCircleLine (c : Circle) (p1 p2 : ℝ × ℝ) : ℕ := sorry

/-- The number of intersection points between a circle and a triangle --/
def intersectionPointsCircleTriangle (c : Circle) (t : Triangle) : ℕ :=
  (intersectionPointsCircleLine c (t.vertices 0) (t.vertices 1)) +
  (intersectionPointsCircleLine c (t.vertices 1) (t.vertices 2)) +
  (intersectionPointsCircleLine c (t.vertices 2) (t.vertices 0))

/-- The maximum number of intersection points between a circle and a triangle is 6 --/
theorem max_intersection_points_circle_triangle :
  ∃ (c : Circle) (t : Triangle), 
    (∀ (c' : Circle) (t' : Triangle), intersectionPointsCircleTriangle c' t' ≤ 6) ∧
    intersectionPointsCircleTriangle c t = 6 :=
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_circle_triangle_l512_51239


namespace NUMINAMATH_CALUDE_max_true_statements_l512_51219

theorem max_true_statements (a b : ℝ) : 
  0 < a ∧ 0 < b ∧ a < b →
  (1 / a < 1 / b) ∧ 
  (a^2 > b^2) ∧ 
  (a < b) ∧ 
  (a > 0) ∧ 
  (b > 0) := by
  sorry

end NUMINAMATH_CALUDE_max_true_statements_l512_51219


namespace NUMINAMATH_CALUDE_race_solution_l512_51267

/-- A race between two runners A and B -/
structure Race where
  /-- The total distance of the race in meters -/
  distance : ℝ
  /-- The time it takes runner A to complete the race in seconds -/
  time_A : ℝ
  /-- The difference in distance between A and B at the finish line in meters -/
  distance_diff : ℝ
  /-- The difference in time between A and B at the finish line in seconds -/
  time_diff : ℝ

/-- The theorem stating the properties of the race and its solution -/
theorem race_solution (race : Race)
  (h1 : race.time_A = 23)
  (h2 : race.distance_diff = 56 ∨ race.time_diff = 7) :
  race.distance = 56 := by
  sorry


end NUMINAMATH_CALUDE_race_solution_l512_51267


namespace NUMINAMATH_CALUDE_range_of_a_l512_51220

def A (a : ℝ) := {x : ℝ | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B := {x : ℝ | x < 0 ∨ x > 19}

theorem range_of_a (a : ℝ) : 
  (A a ⊆ (A a ∩ B)) → (a < 6 ∨ a > 9) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l512_51220


namespace NUMINAMATH_CALUDE_absent_student_grade_calculation_l512_51234

/-- Given a class where one student was initially absent for a test, prove that the absent student's grade can be determined from the class averages before and after including their score. -/
theorem absent_student_grade_calculation (total_students : ℕ) 
  (initial_students : ℕ) (initial_average : ℚ) (final_average : ℚ) 
  (h1 : total_students = 25) 
  (h2 : initial_students = 24)
  (h3 : initial_average = 82)
  (h4 : final_average = 84) :
  (total_students : ℚ) * final_average - (initial_students : ℚ) * initial_average = 132 := by
  sorry

end NUMINAMATH_CALUDE_absent_student_grade_calculation_l512_51234


namespace NUMINAMATH_CALUDE_ladder_problem_l512_51274

theorem ladder_problem (ladder_length height : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height = 12) :
  ∃ (base : ℝ), base^2 + height^2 = ladder_length^2 ∧ base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l512_51274


namespace NUMINAMATH_CALUDE_locus_and_max_area_l512_51222

noncomputable section

-- Define the points E, F, and D
def E : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (2, 0)
def D : ℝ × ℝ := (0, -2)

-- Define the moving point P
def P : ℝ × ℝ → Prop
  | (x, y) => (x + 2) * x + (y - 0) * y = 0 ∧ (x - 2) * x + (y - 0) * y = 0

-- Define the point M
def M : ℝ × ℝ → Prop
  | (x, y) => ∃ (px py : ℝ), P (px, py) ∧ px = x ∧ py = 2 * y

-- Define the locus C
def C : ℝ × ℝ → Prop
  | (x, y) => M (x, y)

-- Define the line l
def l (k : ℝ) : ℝ × ℝ → Prop
  | (x, y) => y = k * x - 2

-- Define the area of quadrilateral OANB
def area_OANB (k : ℝ) : ℝ := 
  8 * Real.sqrt ((4 * k^2 - 3) / (1 + 4 * k^2)^2)

-- Theorem statement
theorem locus_and_max_area :
  (∀ x y, C (x, y) ↔ x^2 / 4 + y^2 = 1) ∧
  (∃ k₁ k₂, k₁ ≠ k₂ ∧
    area_OANB k₁ = 2 ∧
    area_OANB k₂ = 2 ∧
    (∀ k, area_OANB k ≤ 2) ∧
    l k₁ = λ (x, y) => y = Real.sqrt 7 / 2 * x - 2 ∧
    l k₂ = λ (x, y) => y = -Real.sqrt 7 / 2 * x - 2) :=
by sorry

end NUMINAMATH_CALUDE_locus_and_max_area_l512_51222


namespace NUMINAMATH_CALUDE_condition_analysis_l512_51266

theorem condition_analysis :
  (∃ a b : ℝ, (1 / a > 1 / b ∧ a ≥ b) ∨ (1 / a ≤ 1 / b ∧ a < b)) ∧
  (∀ A B : Set α, A = ∅ → A ∩ B = ∅) ∧
  (∃ A B : Set α, A ∩ B = ∅ ∧ A ≠ ∅) ∧
  (∀ a b : ℝ, a^2 + b^2 ≠ 0 ↔ |a| + |b| ≠ 0) ∧
  (∃ a b : ℝ, ∃ n : ℕ, n ≥ 2 ∧ (a^n > b^n ∧ ¬(a > b ∧ b > 0))) :=
by sorry

end NUMINAMATH_CALUDE_condition_analysis_l512_51266


namespace NUMINAMATH_CALUDE_min_value_fraction_l512_51205

theorem min_value_fraction (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧ ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l512_51205


namespace NUMINAMATH_CALUDE_integral_tangent_sine_cosine_l512_51223

open Real MeasureTheory

theorem integral_tangent_sine_cosine :
  ∫ x in (Set.Icc 0 (π/4)), (7 + 3 * tan x) / (sin x + 2 * cos x)^2 = 3 * log (3/2) + 1/6 := by
  sorry

end NUMINAMATH_CALUDE_integral_tangent_sine_cosine_l512_51223


namespace NUMINAMATH_CALUDE_choose_with_mandatory_l512_51271

theorem choose_with_mandatory (n m k : ℕ) (h1 : n = 10) (h2 : m = 4) (h3 : k = 1) :
  (Nat.choose (n - k) (m - k)) = 84 :=
sorry

end NUMINAMATH_CALUDE_choose_with_mandatory_l512_51271


namespace NUMINAMATH_CALUDE_fourth_grade_students_l512_51276

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) : 
  initial = 10 → left = 4 → new = 42 → initial - left + new = 48 := by
sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l512_51276


namespace NUMINAMATH_CALUDE_square_field_area_l512_51214

theorem square_field_area (side_length : ℝ) (h : side_length = 16) : 
  side_length * side_length = 256 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l512_51214


namespace NUMINAMATH_CALUDE_exponent_rules_l512_51265

theorem exponent_rules :
  (∀ x : ℝ, x^5 * x^2 = x^7) ∧
  (∀ m : ℝ, (m^2)^4 = m^8) ∧
  (∀ x y : ℝ, (-2*x*y^2)^3 = -8*x^3*y^6) := by
  sorry

end NUMINAMATH_CALUDE_exponent_rules_l512_51265


namespace NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l512_51297

theorem sqrt_twelve_minus_sqrt_three_equals_sqrt_three : 
  Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_twelve_minus_sqrt_three_equals_sqrt_three_l512_51297


namespace NUMINAMATH_CALUDE_expression_equals_one_l512_51217

theorem expression_equals_one (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  ((((x + 2)^3 * (x^2 - 2*x + 2)^3) / (x^3 + 8)^3)^2 * 
   (((x - 2)^3 * (x^2 + 2*x + 2)^3) / (x^3 - 8)^3)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l512_51217


namespace NUMINAMATH_CALUDE_linear_function_characterization_l512_51242

/-- A function f: ℚ → ℚ satisfies the arithmetic progression property if
    f(x) + f(t) = f(y) + f(z) for all rational x < y < z < t in arithmetic progression -/
def ArithmeticProgressionProperty (f : ℚ → ℚ) : Prop :=
  ∀ (x y z t : ℚ), x < y ∧ y < z ∧ z < t ∧ (y - x = z - y) ∧ (z - y = t - z) →
    f x + f t = f y + f z

/-- The main theorem: if f satisfies the arithmetic progression property,
    then f is a linear function -/
theorem linear_function_characterization (f : ℚ → ℚ) 
  (h : ArithmeticProgressionProperty f) :
  ∃ (c b : ℚ), ∀ (q : ℚ), f q = c * q + b :=
sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l512_51242


namespace NUMINAMATH_CALUDE_special_cone_volume_l512_51235

/-- A cone with inscribed and circumscribed spheres having the same center -/
structure SpecialCone where
  /-- The radius of the inscribed sphere -/
  inscribed_radius : ℝ
  /-- The inscribed and circumscribed spheres have the same center -/
  spheres_same_center : Bool

/-- The volume of a SpecialCone -/
noncomputable def volume (cone : SpecialCone) : ℝ := sorry

/-- Theorem: The volume of a SpecialCone with inscribed radius 1 is 2π -/
theorem special_cone_volume (cone : SpecialCone) 
  (h1 : cone.inscribed_radius = 1) 
  (h2 : cone.spheres_same_center = true) : 
  volume cone = 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_special_cone_volume_l512_51235


namespace NUMINAMATH_CALUDE_increase_in_circumference_l512_51238

/-- The increase in circumference of a circle when its diameter increases by π units -/
theorem increase_in_circumference (d : ℝ) : 
  let original_circumference := π * d
  let new_circumference := π * (d + π)
  let increase := new_circumference - original_circumference
  increase = π^2 := by sorry

end NUMINAMATH_CALUDE_increase_in_circumference_l512_51238


namespace NUMINAMATH_CALUDE_correct_expression_l512_51227

/-- A type representing mathematical expressions --/
inductive MathExpression
  | DivideABC : MathExpression
  | MixedFraction : MathExpression
  | MultiplyAB : MathExpression
  | ThreeM : MathExpression

/-- A predicate that determines if an expression is correctly written --/
def is_correctly_written (e : MathExpression) : Prop :=
  match e with
  | MathExpression.ThreeM => True
  | _ => False

/-- The set of given expressions --/
def expression_set : Set MathExpression :=
  {MathExpression.DivideABC, MathExpression.MixedFraction, 
   MathExpression.MultiplyAB, MathExpression.ThreeM}

theorem correct_expression :
  ∃ (e : MathExpression), e ∈ expression_set ∧ is_correctly_written e :=
by sorry

end NUMINAMATH_CALUDE_correct_expression_l512_51227


namespace NUMINAMATH_CALUDE_seashells_given_to_sam_proof_l512_51243

/-- The number of seashells Joan initially found -/
def initial_seashells : ℕ := 70

/-- The number of seashells Joan has left -/
def remaining_seashells : ℕ := 27

/-- The number of seashells Joan gave to Sam -/
def seashells_given_to_sam : ℕ := initial_seashells - remaining_seashells

theorem seashells_given_to_sam_proof :
  seashells_given_to_sam = 43 := by sorry

end NUMINAMATH_CALUDE_seashells_given_to_sam_proof_l512_51243


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l512_51283

theorem stratified_sampling_problem (total_sample : ℕ) (school_A : ℕ) (school_B : ℕ) (school_C : ℕ) 
  (h1 : total_sample = 60)
  (h2 : school_A = 180)
  (h3 : school_B = 140)
  (h4 : school_C = 160) :
  (total_sample * school_C) / (school_A + school_B + school_C) = 20 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l512_51283


namespace NUMINAMATH_CALUDE_quadratic_equation_completion_square_l512_51284

theorem quadratic_equation_completion_square (x : ℝ) : 
  16 * x^2 - 32 * x - 512 = 0 → ∃ r s : ℝ, (x + r)^2 = s ∧ s = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_completion_square_l512_51284


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l512_51288

/-- Given that x and y are inversely proportional, prove that y = -16.875 when x = -10 -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
  (h2 : ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 30 ∧ x₀ = 3 * y₀ ∧ x₀ * y₀ = k) : 
  x = -10 → y = -16.875 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l512_51288


namespace NUMINAMATH_CALUDE_problem_solution_l512_51245

theorem problem_solution (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 16 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l512_51245


namespace NUMINAMATH_CALUDE_completing_square_result_l512_51269

theorem completing_square_result (x : ℝ) : 
  (x^2 - 4*x + 2 = 0) ↔ ((x - 2)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l512_51269


namespace NUMINAMATH_CALUDE_hash_2_3_4_l512_51299

-- Define the # operation
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c + b

-- Theorem statement
theorem hash_2_3_4 : hash 2 3 4 = -20 := by sorry

end NUMINAMATH_CALUDE_hash_2_3_4_l512_51299


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_l512_51295

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define a point on the left branch of the hyperbola
def left_branch_point (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ P.1 < 0

-- Define the distance from a point to the left focus
def dist_to_left_focus (P : ℝ × ℝ) : ℝ := 10

-- Theorem statement
theorem hyperbola_focus_distance (P : ℝ × ℝ) :
  left_branch_point P → dist_to_left_focus P = 10 →
  ∃ (dist_to_right_focus : ℝ), dist_to_right_focus = 18 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_l512_51295


namespace NUMINAMATH_CALUDE_evaluate_f_l512_51204

/-- The function f(x) = 2x^2 - 4x + 9 -/
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 9

/-- Theorem stating that 2f(3) + 3f(-3) = 147 -/
theorem evaluate_f : 2 * f 3 + 3 * f (-3) = 147 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l512_51204


namespace NUMINAMATH_CALUDE_inequality_theorem_l512_51209

open Set

-- Define the interval (0,+∞)
def openPositiveReals : Set ℝ := {x : ℝ | x > 0}

-- Define the properties of functions f and g
def hasContinuousDerivative (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ Differentiable ℝ f

-- Define the inequality condition
def satisfiesInequality (f g : ℝ → ℝ) : Prop :=
  ∀ x ∈ openPositiveReals, f x > x * (deriv f x) - x^2 * (deriv g x)

-- Theorem statement
theorem inequality_theorem (f g : ℝ → ℝ) 
  (hf : hasContinuousDerivative f) (hg : hasContinuousDerivative g)
  (h_ineq : satisfiesInequality f g) :
  2 * g 2 + 2 * f 1 > f 2 + 2 * g 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l512_51209


namespace NUMINAMATH_CALUDE_max_area_circular_sector_l512_51202

/-- Theorem: Maximum area of a circular sector with perimeter 16 --/
theorem max_area_circular_sector (r θ : ℝ) : 
  r > 0 → 
  θ > 0 → 
  2 * r + θ * r = 16 → 
  (1/2) * θ * r^2 ≤ 16 ∧ 
  (∃ (r₀ θ₀ : ℝ), r₀ > 0 ∧ θ₀ > 0 ∧ 2 * r₀ + θ₀ * r₀ = 16 ∧ (1/2) * θ₀ * r₀^2 = 16) :=
by sorry

end NUMINAMATH_CALUDE_max_area_circular_sector_l512_51202


namespace NUMINAMATH_CALUDE_correct_propositions_l512_51210

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the operations and relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (line_not_in_plane : Line → Plane → Prop)
variable (line_parallel_line : Line → Line → Prop)

-- Axioms for the properties of these operations
axiom perpendicular_sym {α β : Plane} : perpendicular α β → perpendicular β α
axiom parallel_sym {α β : Plane} : parallel α β → parallel β α
axiom line_parallel_plane_sym {l : Line} {α : Plane} : line_parallel_plane l α → line_parallel_plane l α

-- The theorem to be proved
theorem correct_propositions 
  (m n : Line) (α β γ : Plane) : 
  (perpendicular α β ∧ line_perpendicular_plane m β ∧ line_not_in_plane m α → line_parallel_plane m α) ∧
  (parallel α β ∧ line_in_plane m α → line_parallel_plane m β) ∧
  ¬(perpendicular α β ∧ line_parallel_line n m → line_parallel_plane n α ∧ line_parallel_plane n β) ∧
  ¬(perpendicular α β ∧ perpendicular α γ → parallel β γ) :=
sorry

end NUMINAMATH_CALUDE_correct_propositions_l512_51210


namespace NUMINAMATH_CALUDE_cookie_distribution_l512_51291

theorem cookie_distribution (total : ℚ) (blue green red : ℚ) : 
  blue + green + red = total →
  blue + green = 2/3 * total →
  blue = 1/4 * total →
  green / (blue + green) = 5/8 := by
sorry

end NUMINAMATH_CALUDE_cookie_distribution_l512_51291


namespace NUMINAMATH_CALUDE_max_value_inequality_max_value_attained_l512_51249

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + 5 * y < 75) :
  x * y * (75 - 2 * x - 5 * y) ≤ 1562.5 := by
  sorry

theorem max_value_attained (ε : ℝ) (hε : ε > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + 5 * y < 75 ∧
  x * y * (75 - 2 * x - 5 * y) > 1562.5 - ε := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_max_value_attained_l512_51249


namespace NUMINAMATH_CALUDE_home_theater_savings_l512_51224

def in_store_price : ℝ := 320
def in_store_discount : ℝ := 0.05
def website_monthly_payment : ℝ := 62
def website_num_payments : ℕ := 5
def website_shipping : ℝ := 10

theorem home_theater_savings :
  let website_total := website_monthly_payment * website_num_payments + website_shipping
  let in_store_discounted := in_store_price * (1 - in_store_discount)
  website_total - in_store_discounted = 16 := by sorry

end NUMINAMATH_CALUDE_home_theater_savings_l512_51224


namespace NUMINAMATH_CALUDE_collinear_points_m_value_l512_51262

/-- Two non-collinear vectors in a vector space -/
structure NonCollinearVectors (V : Type*) [AddCommGroup V] [Module ℝ V] where
  e₁ : V
  e₂ : V
  not_collinear : ∃ (a b : ℝ), a • e₁ + b • e₂ ≠ 0

/-- Three collinear points in a vector space -/
structure CollinearPoints (V : Type*) [AddCommGroup V] [Module ℝ V] where
  A : V
  B : V
  C : V
  collinear : ∃ (t : ℝ), C - A = t • (B - A)

/-- Theorem: If e₁ and e₂ are non-collinear vectors, AB = 2e₁ + me₂, BC = e₁ + 3e₂,
    and points A, B, C are collinear, then m = 6 -/
theorem collinear_points_m_value
  {V : Type*} [AddCommGroup V] [Module ℝ V]
  (ncv : NonCollinearVectors V)
  (cp : CollinearPoints V)
  (h₁ : cp.B - cp.A = 2 • ncv.e₁ + m • ncv.e₂)
  (h₂ : cp.C - cp.B = ncv.e₁ + 3 • ncv.e₂)
  : m = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_m_value_l512_51262


namespace NUMINAMATH_CALUDE_point_transformation_l512_51250

/-- Rotate a point (x,y) by 180° around (h,k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2*h - x, 2*k - y)

/-- Reflect a point (x,y) about the line y = -x -/
def reflectAboutNegativeX (x y : ℝ) : ℝ × ℝ :=
  (-y, -x)

theorem point_transformation (a b : ℝ) :
  let p₁ := rotate180 a b 2 4
  let p₂ := reflectAboutNegativeX p₁.1 p₁.2
  p₂ = (-1, 4) → a - b = -9 := by
sorry

end NUMINAMATH_CALUDE_point_transformation_l512_51250


namespace NUMINAMATH_CALUDE_bag_original_price_l512_51285

theorem bag_original_price (sale_price : ℝ) (discount_percent : ℝ) (original_price : ℝ) : 
  sale_price = 120 → 
  discount_percent = 50 → 
  sale_price = original_price * (1 - discount_percent / 100) → 
  original_price = 240 := by
sorry

end NUMINAMATH_CALUDE_bag_original_price_l512_51285


namespace NUMINAMATH_CALUDE_sum_of_fractions_l512_51273

theorem sum_of_fractions : 
  (251 : ℚ) / (2008 * 2009) + (251 : ℚ) / (2009 * 2010) = -1 / 8040 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l512_51273


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l512_51201

theorem quadratic_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l512_51201


namespace NUMINAMATH_CALUDE_fourth_term_is_two_l512_51278

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  a_1_eq : a 1 = 16
  a_6_eq : a 6 = 2 * a 5 * a 7

/-- The fourth term of the geometric sequence is 2 -/
theorem fourth_term_is_two (seq : GeometricSequence) : seq.a 4 = 2 := by
  sorry


end NUMINAMATH_CALUDE_fourth_term_is_two_l512_51278


namespace NUMINAMATH_CALUDE_codes_lost_calculation_l512_51253

/-- The number of digits in each code -/
def code_length : ℕ := 4

/-- The base of the number system (decimal) -/
def base : ℕ := 10

/-- The total number of possible codes with leading zeros -/
def total_codes : ℕ := base ^ code_length

/-- The number of possible codes without leading zeros -/
def codes_without_leading_zeros : ℕ := (base - 1) * (base ^ (code_length - 1))

/-- The number of codes lost when disallowing leading zeros -/
def codes_lost : ℕ := total_codes - codes_without_leading_zeros

theorem codes_lost_calculation :
  codes_lost = 1000 :=
sorry

end NUMINAMATH_CALUDE_codes_lost_calculation_l512_51253


namespace NUMINAMATH_CALUDE_samuel_coaching_fee_l512_51200

/-- Calculate the total coaching fee for Samuel --/
theorem samuel_coaching_fee :
  let days_in_period : ℕ := 307 -- Days from Jan 1 to Nov 4 in a non-leap year
  let holidays : ℕ := 5
  let daily_fee : ℕ := 23
  let discount_period : ℕ := 30
  let discount_rate : ℚ := 1 / 10

  let coaching_days : ℕ := days_in_period - holidays
  let full_discount_periods : ℕ := coaching_days / discount_period
  let base_fee : ℕ := coaching_days * daily_fee
  let discount_per_period : ℚ := (discount_period * daily_fee : ℚ) * discount_rate
  let total_discount : ℚ := discount_per_period * full_discount_periods
  
  (base_fee : ℚ) - total_discount = 6256 := by
  sorry

end NUMINAMATH_CALUDE_samuel_coaching_fee_l512_51200


namespace NUMINAMATH_CALUDE_game_ends_in_58_rounds_l512_51257

/-- Represents the state of the game at any point --/
structure GameState where
  playerA : Nat
  playerB : Nat
  playerC : Nat

/-- Simulates one round of the game --/
def playRound (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended --/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends --/
def countRounds (state : GameState) : Nat :=
  sorry

/-- Theorem stating that the game ends after 58 rounds --/
theorem game_ends_in_58_rounds :
  let initialState : GameState := { playerA := 20, playerB := 18, playerC := 15 }
  countRounds initialState = 58 := by
  sorry

end NUMINAMATH_CALUDE_game_ends_in_58_rounds_l512_51257
