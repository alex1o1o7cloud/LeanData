import Mathlib

namespace tan_22_5_deg_l1220_122071

theorem tan_22_5_deg (h1 : Real.pi / 4 = 2 * (22.5 * Real.pi / 180)) 
  (h2 : Real.tan (Real.pi / 4) = 1) :
  Real.tan (22.5 * Real.pi / 180) / (1 - Real.tan (22.5 * Real.pi / 180)^2) = 1/2 := by
  sorry

end tan_22_5_deg_l1220_122071


namespace twenty_knocks_to_knicks_l1220_122004

-- Define the units
variable (knick knack knock : ℚ)

-- Define the given conditions
axiom knicks_to_knacks : 8 * knick = 3 * knack
axiom knacks_to_knocks : 4 * knack = 5 * knock

-- State the theorem
theorem twenty_knocks_to_knicks : 
  20 * knock = 64 / 3 * knick :=
sorry

end twenty_knocks_to_knicks_l1220_122004


namespace wheel_rotations_per_block_l1220_122045

theorem wheel_rotations_per_block 
  (total_blocks : ℕ) 
  (initial_rotations : ℕ) 
  (additional_rotations : ℕ) : 
  total_blocks = 8 → 
  initial_rotations = 600 → 
  additional_rotations = 1000 → 
  (initial_rotations + additional_rotations) / total_blocks = 200 := by
sorry

end wheel_rotations_per_block_l1220_122045


namespace exists_non_prime_l1220_122088

/-- The recurrence relation for the sequence x_n -/
def recurrence (x₀ a b : ℕ) : ℕ → ℕ
| 0 => x₀
| n + 1 => recurrence x₀ a b n * a + b

/-- Theorem: There exists a non-prime number in the sequence defined by the recurrence relation -/
theorem exists_non_prime (x₀ a b : ℕ) : ∃ n : ℕ, ¬ Nat.Prime (recurrence x₀ a b n) := by
  sorry

end exists_non_prime_l1220_122088


namespace arithmetic_sequence_sum_l1220_122058

/-- 
Given an arithmetic sequence with first term a₁ = k^2 - k + 1 and common difference d = 1,
the sum of the first k + 2 terms is equal to k^3 + 2k^2 + k + 2.
-/
theorem arithmetic_sequence_sum (k : ℕ) : 
  let a₁ := k^2 - k + 1
  let d := 1
  let n := k + 2
  let Sn := n * (a₁ + (a₁ + (n - 1) * d)) / 2
  Sn = k^3 + 2*k^2 + k + 2 := by
sorry

end arithmetic_sequence_sum_l1220_122058


namespace legs_in_room_is_40_l1220_122080

/-- Calculates the total number of legs in a room with various furniture items. -/
def total_legs_in_room : ℕ :=
  let four_legged_items := 4 + 1 + 2  -- 4 tables, 1 sofa, 2 chairs
  let three_legged_tables := 3
  let one_legged_table := 1
  let two_legged_rocking_chair := 1
  
  4 * four_legged_items + 
  3 * three_legged_tables + 
  1 * one_legged_table + 
  2 * two_legged_rocking_chair

/-- Theorem stating that the total number of legs in the room is 40. -/
theorem legs_in_room_is_40 : total_legs_in_room = 40 := by
  sorry

end legs_in_room_is_40_l1220_122080


namespace base_conversion_403_6_to_8_l1220_122082

/-- Converts a number from base 6 to base 10 --/
def base6_to_decimal (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 10 to base 8 --/
def decimal_to_base8 (n : ℕ) : ℕ :=
  if n < 8 then n
  else (decimal_to_base8 (n / 8)) * 10 + (n % 8)

theorem base_conversion_403_6_to_8 :
  decimal_to_base8 (base6_to_decimal 403) = 223 := by
  sorry

end base_conversion_403_6_to_8_l1220_122082


namespace sin_2alpha_value_l1220_122046

theorem sin_2alpha_value (α : Real) 
  (h : Real.tan (α - π/4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 7/9 := by
  sorry

end sin_2alpha_value_l1220_122046


namespace third_speed_calculation_l1220_122084

/-- Prove that given the conditions, the third speed is 3 km/hr -/
theorem third_speed_calculation (total_time : ℝ) (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_time = 11 →
  total_distance = 900 →
  speed1 = 6 →
  speed2 = 9 →
  ∃ (speed3 : ℝ), speed3 = 3 ∧
    total_time = (total_distance / 3) / (speed1 * 1000 / 60) +
                 (total_distance / 3) / (speed2 * 1000 / 60) +
                 (total_distance / 3) / (speed3 * 1000 / 60) :=
by sorry


end third_speed_calculation_l1220_122084


namespace pencil_pen_cost_l1220_122015

/-- Given the costs of different combinations of pencils and pens, 
    calculate the cost of three pencils and three pens. -/
theorem pencil_pen_cost (pencil pen : ℝ) 
  (h1 : 3 * pencil + 2 * pen = 3.60)
  (h2 : 2 * pencil + 3 * pen = 3.15) :
  3 * pencil + 3 * pen = 4.05 := by
  sorry


end pencil_pen_cost_l1220_122015


namespace math_team_combinations_l1220_122072

theorem math_team_combinations : ℕ := by
  -- Define the total number of girls and boys in the math club
  let total_girls : ℕ := 5
  let total_boys : ℕ := 5
  
  -- Define the number of girls and boys needed for the team
  let team_girls : ℕ := 3
  let team_boys : ℕ := 3
  
  -- Define the total team size
  let team_size : ℕ := team_girls + team_boys
  
  -- Calculate the number of ways to choose the team
  let result := (total_girls.choose team_girls) * (total_boys.choose team_boys)
  
  -- Prove that the result is equal to 100
  have h : result = 100 := by sorry
  
  -- Return the result
  exact result

end math_team_combinations_l1220_122072


namespace mike_bird_feeding_l1220_122075

/-- The number of seeds Mike throws to the birds on the left -/
def seeds_left : ℕ := 20

/-- The total number of seeds Mike starts with -/
def total_seeds : ℕ := 120

/-- The number of additional seeds thrown -/
def additional_seeds : ℕ := 30

/-- The number of seeds left at the end -/
def remaining_seeds : ℕ := 30

theorem mike_bird_feeding :
  seeds_left + 2 * seeds_left + additional_seeds + remaining_seeds = total_seeds :=
by sorry

end mike_bird_feeding_l1220_122075


namespace commission_allocation_l1220_122027

theorem commission_allocation (commission_rate : ℚ) (total_sales : ℚ) (amount_saved : ℚ)
  (h1 : commission_rate = 12 / 100)
  (h2 : total_sales = 24000)
  (h3 : amount_saved = 1152) :
  (total_sales * commission_rate - amount_saved) / (total_sales * commission_rate) = 60 / 100 := by
  sorry

end commission_allocation_l1220_122027


namespace min_value_fraction_l1220_122085

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hsum : b + 2*a = 8) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 8 → 2/(x*y) ≥ 2/(a*b)) ∧ 2/(a*b) = 1/4 :=
sorry

end min_value_fraction_l1220_122085


namespace min_abs_alpha_plus_gamma_l1220_122094

theorem min_abs_alpha_plus_gamma :
  ∀ (α γ : ℂ),
  let g := λ (z : ℂ) => (3 + I) * z^2 + α * z + γ
  (g 1).im = 0 →
  (g I).im = 0 →
  ∃ (α₀ γ₀ : ℂ),
    (let g₀ := λ (z : ℂ) => (3 + I) * z^2 + α₀ * z + γ₀
     (g₀ 1).im = 0 ∧
     (g₀ I).im = 0 ∧
     Complex.abs α₀ + Complex.abs γ₀ = Real.sqrt 2 ∧
     ∀ (α' γ' : ℂ),
       (let g' := λ (z : ℂ) => (3 + I) * z^2 + α' * z + γ'
        (g' 1).im = 0 ∧
        (g' I).im = 0 →
        Complex.abs α' + Complex.abs γ' ≥ Real.sqrt 2)) :=
by sorry

end min_abs_alpha_plus_gamma_l1220_122094


namespace difference_is_2_5q_minus_15_l1220_122073

/-- The difference in dimes between two people's quarter amounts -/
def difference_in_dimes (q : ℝ) : ℝ :=
  let samantha_quarters : ℝ := 3 * q + 2
  let bob_quarters : ℝ := 2 * q + 8
  let quarter_to_dime : ℝ := 2.5
  quarter_to_dime * (samantha_quarters - bob_quarters)

/-- Theorem stating the difference in dimes -/
theorem difference_is_2_5q_minus_15 (q : ℝ) :
  difference_in_dimes q = 2.5 * q - 15 := by
  sorry

end difference_is_2_5q_minus_15_l1220_122073


namespace three_digit_self_repeating_powers_l1220_122006

theorem three_digit_self_repeating_powers : 
  {N : ℕ | 100 ≤ N ∧ N < 1000 ∧ ∀ k : ℕ, k ≥ 1 → N^k % 1000 = N} = {376, 625} := by
  sorry

end three_digit_self_repeating_powers_l1220_122006


namespace circle_plus_equality_l1220_122076

/-- Definition of the ⊕ operation -/
def circle_plus (a b : ℝ) : ℝ := a * b + a - b

/-- Theorem stating the equality to be proved -/
theorem circle_plus_equality (a b : ℝ) : 
  circle_plus a b + circle_plus (b - a) b = b^2 - b := by
  sorry

end circle_plus_equality_l1220_122076


namespace trillion_scientific_notation_l1220_122081

theorem trillion_scientific_notation :
  (10000 : ℝ) * 10000 * 10000 = 1 * (10 : ℝ)^12 := by sorry

end trillion_scientific_notation_l1220_122081


namespace sqrt_meaningful_range_l1220_122057

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3 - x) ↔ x ≤ 3 := by sorry

end sqrt_meaningful_range_l1220_122057


namespace function_property_l1220_122063

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_property (f : ℝ → ℝ) 
  (h1 : is_even_function f)
  (h2 : ∀ x, f (x - 3/2) = f (x + 1/2))
  (h3 : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
sorry

end function_property_l1220_122063


namespace residue_1237_mod_17_l1220_122070

theorem residue_1237_mod_17 : 1237 % 17 = 13 := by
  sorry

end residue_1237_mod_17_l1220_122070


namespace set_difference_N_M_l1220_122024

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {1, 2, 3, 7}

theorem set_difference_N_M : N \ M = {7} := by
  sorry

end set_difference_N_M_l1220_122024


namespace correct_num_students_l1220_122003

/-- The number of students in a class with incorrectly entered marks -/
def num_students : ℕ := by sorry

/-- The total increase in marks due to incorrect entry -/
def total_mark_increase : ℕ := 44

/-- The increase in class average due to incorrect entry -/
def average_increase : ℚ := 1/2

theorem correct_num_students :
  num_students = 88 ∧
  (total_mark_increase : ℚ) = num_students * average_increase := by sorry

end correct_num_students_l1220_122003


namespace count_special_numbers_is_360_l1220_122056

/-- A function that counts 4-digit numbers beginning with 1 and having exactly two identical digits -/
def count_special_numbers : ℕ :=
  let digits := Finset.range 10  -- digits from 0 to 9
  let non_one_digits := digits.erase 1  -- digits excluding 1
  let case1 := 3 * non_one_digits.card * (non_one_digits.card - 1)  -- case where one of the identical digits is 1
  let case2 := 2 * non_one_digits.card * digits.card  -- case where the identical digits are not 1
  case1 + case2

/-- Theorem stating that the count of special numbers is 360 -/
theorem count_special_numbers_is_360 : count_special_numbers = 360 := by
  sorry

end count_special_numbers_is_360_l1220_122056


namespace bugs_and_flowers_l1220_122010

theorem bugs_and_flowers (total_bugs : ℝ) (total_flowers : ℝ) 
  (h1 : total_bugs = 2.0) 
  (h2 : total_flowers = 3.0) : 
  total_flowers / total_bugs = 1.5 := by
  sorry

end bugs_and_flowers_l1220_122010


namespace min_value_quadratic_sum_l1220_122016

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + 2*y + z = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ ∀ (a b c : ℝ), a + 2*b + c = 1 → a^2 + 4*b^2 + c^2 ≥ m :=
sorry

end min_value_quadratic_sum_l1220_122016


namespace area_trapezoid_equals_rectangle_l1220_122002

-- Define the points
variable (P Q R S T : ℝ × ℝ)

-- Define the shapes
def rectangle_PQRS : Set (ℝ × ℝ) := sorry
def trapezoid_TQSR : Set (ℝ × ℝ) := sorry

-- Define the area function
noncomputable def area : Set (ℝ × ℝ) → ℝ := sorry

-- State the theorem
theorem area_trapezoid_equals_rectangle
  (h1 : area rectangle_PQRS = 20)
  (h2 : trapezoid_TQSR ⊆ rectangle_PQRS)
  (h3 : P = (0, 0))
  (h4 : Q = (5, 0))
  (h5 : R = (5, 4))
  (h6 : S = (0, 4))
  (h7 : T = (2, 4)) :
  area trapezoid_TQSR = area rectangle_PQRS :=
by sorry

end area_trapezoid_equals_rectangle_l1220_122002


namespace polynomial_equality_l1220_122026

theorem polynomial_equality (q : Polynomial ℝ) :
  (q + (2 * X^4 - 5 * X^2 + 8 * X + 3) = 10 * X^3 - 7 * X^2 + 15 * X + 6) →
  q = -2 * X^4 + 10 * X^3 - 2 * X^2 + 7 * X + 3 := by
  sorry

end polynomial_equality_l1220_122026


namespace system_solution_l1220_122092

theorem system_solution :
  let f (x y : ℝ) := y + Real.sqrt (y - 3*x) + 3*x = 12
  let g (x y : ℝ) := y^2 + y - 3*x - 9*x^2 = 144
  ∀ x y : ℝ, (f x y ∧ g x y) ↔ ((x = -4/3 ∧ y = 12) ∨ (x = -24 ∧ y = 72)) :=
by sorry

end system_solution_l1220_122092


namespace largest_angle_120_l1220_122068

-- Define the triangle PQR
structure Triangle :=
  (P Q R : ℝ)

-- Define properties of the triangle
def isObtuse (t : Triangle) : Prop :=
  t.P > 90 ∨ t.Q > 90 ∨ t.R > 90

def isIsosceles (t : Triangle) : Prop :=
  (t.P = t.Q) ∨ (t.Q = t.R) ∨ (t.P = t.R)

def angleP30 (t : Triangle) : Prop :=
  t.P = 30

-- Theorem statement
theorem largest_angle_120 (t : Triangle) 
  (h1 : isObtuse t) 
  (h2 : isIsosceles t) 
  (h3 : angleP30 t) : 
  max t.P (max t.Q t.R) = 120 := by
  sorry

end largest_angle_120_l1220_122068


namespace cake_cutting_l1220_122050

/-- Represents a square cake -/
structure Cake where
  side : ℕ
  pieces : ℕ

/-- The maximum number of pieces obtainable with a single straight cut -/
def max_pieces_single_cut (c : Cake) : ℕ := sorry

/-- The minimum number of straight cuts required to intersect all original pieces -/
def min_cuts_all_pieces (c : Cake) : ℕ := sorry

/-- The theorem statement -/
theorem cake_cutting (c : Cake) 
  (h1 : c.side = 4) 
  (h2 : c.pieces = 16) : 
  max_pieces_single_cut c = 23 ∧ min_cuts_all_pieces c = 3 := by sorry

end cake_cutting_l1220_122050


namespace andrews_age_l1220_122096

theorem andrews_age (a : ℕ) (g : ℕ) : 
  g = 12 * a →  -- Andrew's grandfather's age is twelve times Andrew's age
  g - a = 55 →  -- Andrew's grandfather was 55 years old when Andrew was born
  a = 5 :=       -- Andrew's age is 5 years
by sorry

end andrews_age_l1220_122096


namespace dot_product_special_vectors_l1220_122051

theorem dot_product_special_vectors :
  let a : ℝ × ℝ := (Real.sin (15 * π / 180), Real.sin (75 * π / 180))
  let b : ℝ × ℝ := (Real.cos (30 * π / 180), Real.sin (30 * π / 180))
  (a.1 * b.1 + a.2 * b.2) = Real.sqrt 2 / 2 := by
  sorry

end dot_product_special_vectors_l1220_122051


namespace intersection_theorem_l1220_122043

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x^2 < 4}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_theorem :
  A_intersect_B = {x | -1 < x ∧ x < 2} :=
by sorry

end intersection_theorem_l1220_122043


namespace shared_side_angle_measure_l1220_122054

-- Define the properties of the figure
def regular_pentagon (P : Set Point) : Prop := sorry

def equilateral_triangle (T : Set Point) : Prop := sorry

def share_side (P T : Set Point) : Prop := sorry

-- Define the angle we're interested in
def angle_at_vertex (T : Set Point) (v : Point) : ℝ := sorry

-- Theorem statement
theorem shared_side_angle_measure 
  (P T : Set Point) (v : Point) :
  regular_pentagon P → 
  equilateral_triangle T → 
  share_side P T → 
  angle_at_vertex T v = 6 := by sorry

end shared_side_angle_measure_l1220_122054


namespace female_contestant_probability_l1220_122018

theorem female_contestant_probability :
  let total_contestants : ℕ := 8
  let female_contestants : ℕ := 4
  let male_contestants : ℕ := 4
  let chosen_contestants : ℕ := 2
  
  (female_contestants.choose chosen_contestants : ℚ) / (total_contestants.choose chosen_contestants) = 3 / 14 := by
  sorry

end female_contestant_probability_l1220_122018


namespace base7_sum_property_l1220_122007

/-- A digit in base 7 is a natural number less than 7 -/
def Digit7 : Type := { n : ℕ // n < 7 }

/-- Convert a three-digit number in base 7 to its decimal representation -/
def toDecimal (d e f : Digit7) : ℕ := 49 * d.val + 7 * e.val + f.val

/-- The sum of three permutations of a three-digit number in base 7 -/
def sumPermutations (d e f : Digit7) : ℕ :=
  toDecimal d e f + toDecimal e f d + toDecimal f d e

theorem base7_sum_property (d e f : Digit7) 
  (h_distinct : d ≠ e ∧ d ≠ f ∧ e ≠ f) 
  (h_nonzero : d.val ≠ 0 ∧ e.val ≠ 0 ∧ f.val ≠ 0)
  (h_sum : sumPermutations d e f = 400 * d.val) :
  e.val + f.val = 6 :=
sorry

end base7_sum_property_l1220_122007


namespace right_triangle_arctan_sum_l1220_122078

/-- In a right-angled triangle ABC, the sum of two specific arctangent expressions equals π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  Real.arctan (a / (Real.sqrt b + Real.sqrt c)) + Real.arctan (b / (Real.sqrt a + Real.sqrt c)) = π/4 := by
  sorry

#check right_triangle_arctan_sum

end right_triangle_arctan_sum_l1220_122078


namespace cookies_per_day_l1220_122037

-- Define the problem parameters
def cookie_cost : ℕ := 19
def total_spent : ℕ := 2356
def days_in_march : ℕ := 31

-- Define the theorem
theorem cookies_per_day :
  (total_spent / cookie_cost) / days_in_march = 4 := by
  sorry


end cookies_per_day_l1220_122037


namespace min_perimeter_57_triangle_hexagon_exists_57_triangle_hexagon_with_perimeter_19_l1220_122040

/-- Represents a hexagon formed by unit equilateral triangles -/
structure TriangleHexagon where
  /-- The number of unit equilateral triangles used to form the hexagon -/
  num_triangles : ℕ
  /-- The perimeter of the hexagon -/
  perimeter : ℕ
  /-- Assertion that the hexagon is formed without gaps or overlaps -/
  no_gaps_or_overlaps : Prop
  /-- Assertion that all internal angles of the hexagon are not greater than 180 degrees -/
  angles_not_exceeding_180 : Prop

/-- Theorem stating the minimum perimeter of a hexagon formed by 57 unit equilateral triangles -/
theorem min_perimeter_57_triangle_hexagon :
  ∀ h : TriangleHexagon,
    h.num_triangles = 57 →
    h.no_gaps_or_overlaps →
    h.angles_not_exceeding_180 →
    h.perimeter ≥ 19 := by
  sorry

/-- Existence of a hexagon with perimeter 19 formed by 57 unit equilateral triangles -/
theorem exists_57_triangle_hexagon_with_perimeter_19 :
  ∃ h : TriangleHexagon,
    h.num_triangles = 57 ∧
    h.perimeter = 19 ∧
    h.no_gaps_or_overlaps ∧
    h.angles_not_exceeding_180 := by
  sorry

end min_perimeter_57_triangle_hexagon_exists_57_triangle_hexagon_with_perimeter_19_l1220_122040


namespace circle_radius_theorem_l1220_122031

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the circle
structure Circle :=
  (center : Point)
  (radius : ℝ)

-- Define the points E and F on the sides of the triangle
def E (triangle : Triangle) : Point := sorry
def F (triangle : Triangle) : Point := sorry

-- Define the angles
def angle_ABC (triangle : Triangle) : ℝ := sorry
def angle_AEC (triangle : Triangle) (circle : Circle) : ℝ := sorry
def angle_BAF (triangle : Triangle) (circle : Circle) : ℝ := sorry

-- Define the length of AC
def length_AC (triangle : Triangle) : ℝ := sorry

-- State the theorem
theorem circle_radius_theorem (triangle : Triangle) (circle : Circle) :
  angle_ABC triangle = 72 →
  angle_AEC triangle circle = 5 * angle_BAF triangle circle →
  length_AC triangle = 6 →
  circle.radius = 3 := by sorry

end circle_radius_theorem_l1220_122031


namespace range_of_m_l1220_122047

theorem range_of_m (x y m : ℝ) (h1 : 2/x + 1/y = 1) (h2 : x + y = 2 + 2*m) :
  -4 < m ∧ m < 2 :=
by sorry

end range_of_m_l1220_122047


namespace sequence_2013_value_l1220_122030

def is_valid_sequence (a : ℕ → ℤ) : Prop :=
  ∀ (p k : ℕ), Nat.Prime p → k > 0 → a (p * k + 1) = p * a k - 3 * a p + 13

theorem sequence_2013_value (a : ℕ → ℤ) (h : is_valid_sequence a) : a 2013 = 13 := by
  sorry

end sequence_2013_value_l1220_122030


namespace no_solution_base_conversion_l1220_122008

theorem no_solution_base_conversion : ¬∃ (d : ℕ), d ≤ 9 ∧ d * 5 + 2 = d * 9 + 7 := by
  sorry

end no_solution_base_conversion_l1220_122008


namespace original_water_amount_l1220_122032

/-- Proves that the original amount of water in a glass is 15 ounces, given the daily evaporation rate,
    evaporation period, and the percentage of water evaporated. -/
theorem original_water_amount
  (daily_evaporation : ℝ)
  (evaporation_period : ℕ)
  (evaporation_percentage : ℝ)
  (h1 : daily_evaporation = 0.05)
  (h2 : evaporation_period = 15)
  (h3 : evaporation_percentage = 0.05)
  (h4 : daily_evaporation * ↑evaporation_period = evaporation_percentage * original_amount) :
  original_amount = 15 :=
by
  sorry

#check original_water_amount

end original_water_amount_l1220_122032


namespace yellow_flowers_killed_correct_l1220_122098

/-- Represents the number of flowers of each color --/
structure FlowerCounts where
  red : ℕ
  yellow : ℕ
  orange : ℕ
  purple : ℕ

/-- Represents the problem parameters --/
structure BouquetProblem where
  seeds_per_color : ℕ
  flowers_per_bouquet : ℕ
  total_bouquets : ℕ
  killed_flowers : FlowerCounts

def yellow_flowers_killed (problem : BouquetProblem) : ℕ :=
  problem.seeds_per_color -
    (problem.total_bouquets * problem.flowers_per_bouquet -
      (problem.seeds_per_color - problem.killed_flowers.red +
       problem.seeds_per_color - problem.killed_flowers.orange +
       problem.seeds_per_color - problem.killed_flowers.purple))

theorem yellow_flowers_killed_correct (problem : BouquetProblem) :
  problem.seeds_per_color = 125 →
  problem.flowers_per_bouquet = 9 →
  problem.total_bouquets = 36 →
  problem.killed_flowers.red = 45 →
  problem.killed_flowers.orange = 30 →
  problem.killed_flowers.purple = 40 →
  yellow_flowers_killed problem = 61 := by
  sorry

#eval yellow_flowers_killed {
  seeds_per_color := 125,
  flowers_per_bouquet := 9,
  total_bouquets := 36,
  killed_flowers := {
    red := 45,
    yellow := 0,  -- This value doesn't affect the calculation
    orange := 30,
    purple := 40
  }
}

end yellow_flowers_killed_correct_l1220_122098


namespace function_property_l1220_122017

noncomputable section

def f (x : ℝ) : ℝ := Real.log ((1 - x) / (1 + x))

def domain (x : ℝ) : Prop := -1 < x ∧ x < 1

theorem function_property (f : ℝ → ℝ) (h : ∀ x y, domain x → domain y → f x + f y = f ((x + y) / (1 + x * y))) :
  ∀ x, domain x → f x = Real.log ((1 - x) / (1 + x)) →
  (∀ x, domain x → f (-x) = -f x) ∧
  ∃ a b, domain a ∧ domain b ∧ 
    f ((a + b) / (1 + a * b)) = 1 ∧ 
    f ((a - b) / (1 - a * b)) = 2 ∧
    f a = 3/2 ∧ f b = -1/2 :=
by sorry

end function_property_l1220_122017


namespace reciprocal_in_fourth_quadrant_l1220_122077

theorem reciprocal_in_fourth_quadrant (i : ℂ) (z : ℂ) :
  i * i = -1 →
  z = 1 + i →
  let w := 1 / z
  0 < w.re ∧ w.im < 0 := by
  sorry

end reciprocal_in_fourth_quadrant_l1220_122077


namespace f_composition_value_l1220_122014

noncomputable def f (x : ℝ) : ℝ :=
  if |x| ≤ 1 then |x - 1| - 2 else 1 / (1 + x^2)

theorem f_composition_value : f (f 3) = -11/10 := by
  sorry

end f_composition_value_l1220_122014


namespace square_area_with_four_circles_l1220_122038

theorem square_area_with_four_circles (r : ℝ) (h : r = 3) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 144 := by sorry

end square_area_with_four_circles_l1220_122038


namespace store_earnings_is_120_l1220_122097

/-- Represents the store's sales policy and outcomes -/
structure StoreSales where
  pencil_count : ℕ
  eraser_per_pencil : ℕ
  eraser_price : ℚ
  pencil_price : ℚ

/-- Calculates the total earnings from pencil and eraser sales -/
def total_earnings (s : StoreSales) : ℚ :=
  s.pencil_count * s.pencil_price + 
  s.pencil_count * s.eraser_per_pencil * s.eraser_price

/-- Theorem stating that the store's earnings are $120 given the specified conditions -/
theorem store_earnings_is_120 (s : StoreSales) 
  (h1 : s.eraser_per_pencil = 2)
  (h2 : s.eraser_price = 1)
  (h3 : s.pencil_price = 2 * s.eraser_per_pencil * s.eraser_price)
  (h4 : s.pencil_count = 20) : 
  total_earnings s = 120 := by
  sorry

#eval total_earnings { pencil_count := 20, eraser_per_pencil := 2, eraser_price := 1, pencil_price := 4 }

end store_earnings_is_120_l1220_122097


namespace min_value_of_function_l1220_122039

theorem min_value_of_function (x : ℝ) : 
  (x^2 + 5) / Real.sqrt (x^2 + 4) ≥ 5/2 ∧ 
  ∃ y : ℝ, (y^2 + 5) / Real.sqrt (y^2 + 4) = 5/2 :=
by sorry

end min_value_of_function_l1220_122039


namespace intersection_of_A_and_B_l1220_122005

def A : Set (ℝ × ℝ) := {p | p.1 + p.2 = 5}
def B : Set (ℝ × ℝ) := {p | p.1 - p.2 = 1}

theorem intersection_of_A_and_B : A ∩ B = {(3, 2)} := by
  sorry

end intersection_of_A_and_B_l1220_122005


namespace min_value_xy_l1220_122074

theorem min_value_xy (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h_geom : (Real.log x) * (Real.log y) = 1 / 16) : 
  x * y ≥ Real.sqrt 10 := by
  sorry

end min_value_xy_l1220_122074


namespace positive_t_value_l1220_122033

theorem positive_t_value (a b : ℂ) (t : ℝ) :
  (Complex.abs a = 3) →
  (Complex.abs b = 5) →
  (a * b = t - 3 * Complex.I) →
  (t > 0) →
  t = 6 * Real.sqrt 6 :=
by sorry

end positive_t_value_l1220_122033


namespace computer_sticker_price_l1220_122034

theorem computer_sticker_price : 
  ∀ (sticker_price : ℝ),
    (sticker_price * 0.85 - 90 = sticker_price * 0.75 - 15) →
    sticker_price = 750 := by
  sorry

end computer_sticker_price_l1220_122034


namespace cube_max_volume_l1220_122066

/-- A cuboid with side lengths a, b, and c. -/
structure Cuboid where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The surface area of a cuboid. -/
def surfaceArea (x : Cuboid) : ℝ :=
  2 * (x.a * x.b + x.b * x.c + x.a * x.c)

/-- The volume of a cuboid. -/
def volume (x : Cuboid) : ℝ :=
  x.a * x.b * x.c

/-- Given a fixed surface area S, the cube maximizes the volume among all cuboids. -/
theorem cube_max_volume (S : ℝ) (h : 0 < S) :
  ∀ x : Cuboid, surfaceArea x = S →
    ∃ y : Cuboid, surfaceArea y = S ∧ y.a = y.b ∧ y.b = y.c ∧
      ∀ z : Cuboid, surfaceArea z = S → volume z ≤ volume y :=
by sorry

end cube_max_volume_l1220_122066


namespace butter_for_original_recipe_l1220_122091

/-- Given a recipe where 12 ounces of butter is used for 28 cups of flour
    in a 4x version of the original recipe, prove that the amount of butter
    needed for the original recipe is 3 ounces. -/
theorem butter_for_original_recipe
  (butter_4x : ℝ) -- Amount of butter for 4x recipe
  (flour_4x : ℝ) -- Amount of flour for 4x recipe
  (scale_factor : ℕ) -- Factor by which the original recipe is scaled
  (h1 : butter_4x = 12) -- 12 ounces of butter used in 4x recipe
  (h2 : flour_4x = 28) -- 28 cups of flour used in 4x recipe
  (h3 : scale_factor = 4) -- The recipe is scaled by a factor of 4
  : butter_4x / scale_factor = 3 := by
  sorry

end butter_for_original_recipe_l1220_122091


namespace sum_of_squares_first_15_sum_of_squares_16_to_30_main_theorem_l1220_122012

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_first_15 :
  sum_of_squares 15 = 1240 :=
by sorry

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8205 :=
by sorry

theorem main_theorem :
  sum_of_squares 15 = 1240 :=
by sorry

end sum_of_squares_first_15_sum_of_squares_16_to_30_main_theorem_l1220_122012


namespace clock_angle_at_seven_l1220_122067

/-- The number of degrees in a full circle -/
def full_circle : ℕ := 360

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees per hour on a clock face -/
def degrees_per_hour : ℕ := full_circle / clock_hours

/-- The hour we're examining -/
def current_hour : ℕ := 7

/-- The smaller angle between the hour hand and 12 o'clock position -/
def smaller_angle : ℕ := min (current_hour * degrees_per_hour) ((clock_hours - current_hour) * degrees_per_hour)

theorem clock_angle_at_seven : smaller_angle = 150 := by
  sorry

end clock_angle_at_seven_l1220_122067


namespace kaleb_spring_earnings_l1220_122089

/-- Represents Kaleb's lawn mowing business earnings and expenses -/
structure LawnMowingBusiness where
  spring_earnings : ℤ
  summer_earnings : ℤ
  supplies_cost : ℤ
  final_amount : ℤ

/-- Theorem stating Kaleb's spring earnings given the other known values -/
theorem kaleb_spring_earnings (business : LawnMowingBusiness)
  (h1 : business.summer_earnings = 50)
  (h2 : business.supplies_cost = 4)
  (h3 : business.final_amount = 50) :
  business.spring_earnings = 4 := by
  sorry

end kaleb_spring_earnings_l1220_122089


namespace divisibility_equivalence_l1220_122013

theorem divisibility_equivalence (n : ℤ) : 
  let A := n % 1000
  let B := n / 1000
  let k := A - B
  (7 ∣ n ∨ 11 ∣ n ∨ 13 ∣ n) ↔ (7 ∣ k ∨ 11 ∣ k ∨ 13 ∣ k) := by
  sorry

end divisibility_equivalence_l1220_122013


namespace equilateral_triangle_area_perimeter_ratio_l1220_122021

/-- The ratio of the area to the perimeter of an equilateral triangle with side length 6 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 6
  let area : ℝ := (side_length^2 * Real.sqrt 3) / 4
  let perimeter : ℝ := 3 * side_length
  area / perimeter = Real.sqrt 3 / 2 := by
  sorry

end equilateral_triangle_area_perimeter_ratio_l1220_122021


namespace center_sum_l1220_122079

theorem center_sum (x y : ℝ) : 
  (∀ X Y : ℝ, X^2 + Y^2 + 4*X - 6*Y = 3 ↔ (X - x)^2 + (Y - y)^2 = 16) → 
  x + y = 1 := by
sorry

end center_sum_l1220_122079


namespace tablecloth_black_percentage_l1220_122023

/-- Represents a square tablecloth -/
structure Tablecloth :=
  (size : ℕ)
  (black_outer_ratio : ℚ)

/-- Calculates the percentage of black area on the tablecloth -/
def black_percentage (t : Tablecloth) : ℚ :=
  let total_squares := t.size * t.size
  let outer_squares := 4 * (t.size - 1)
  let black_squares := (outer_squares : ℚ) * t.black_outer_ratio
  (black_squares / total_squares) * 100

/-- Theorem stating that a 5x5 tablecloth with half of each outer square black is 32% black -/
theorem tablecloth_black_percentage :
  let t : Tablecloth := ⟨5, 1/2⟩
  black_percentage t = 32 := by
  sorry

end tablecloth_black_percentage_l1220_122023


namespace max_sum_of_product_48_l1220_122000

theorem max_sum_of_product_48 :
  ∃ (a b : ℕ), a * b = 48 ∧ a + b = 49 ∧
  ∀ (x y : ℕ), x * y = 48 → x + y ≤ 49 := by
  sorry

end max_sum_of_product_48_l1220_122000


namespace rachel_apples_l1220_122053

def initial_apples (num_trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ) : ℕ :=
  num_trees * apples_per_tree + remaining_apples

theorem rachel_apples : initial_apples 3 8 9 = 33 := by
  sorry

end rachel_apples_l1220_122053


namespace gears_rotating_when_gear1_rotates_l1220_122061

-- Define the state of a gear (rotating or stopped)
inductive GearState
| rotating
| stopped

-- Define the gearbox with 6 gears
structure Gearbox :=
  (gear1 gear2 gear3 gear4 gear5 gear6 : GearState)

-- Define the conditions of the gearbox operation
def validGearbox (gb : Gearbox) : Prop :=
  -- Condition 1
  (gb.gear1 = GearState.rotating → gb.gear2 = GearState.rotating ∧ gb.gear5 = GearState.stopped) ∧
  -- Condition 2
  ((gb.gear2 = GearState.rotating ∨ gb.gear5 = GearState.rotating) → gb.gear4 = GearState.stopped) ∧
  -- Condition 3
  (gb.gear3 = GearState.rotating ↔ gb.gear4 = GearState.rotating) ∧
  -- Condition 4
  (gb.gear5 = GearState.rotating ∨ gb.gear6 = GearState.rotating)

-- Theorem statement
theorem gears_rotating_when_gear1_rotates (gb : Gearbox) :
  validGearbox gb →
  gb.gear1 = GearState.rotating →
  gb.gear2 = GearState.rotating ∧ gb.gear3 = GearState.rotating ∧ gb.gear6 = GearState.rotating :=
by sorry

end gears_rotating_when_gear1_rotates_l1220_122061


namespace water_level_rise_l1220_122022

/-- The rise in water level when a cube is immersed in a rectangular vessel -/
theorem water_level_rise
  (cube_edge : ℝ)
  (vessel_length : ℝ)
  (vessel_width : ℝ)
  (h_cube_edge : cube_edge = 15)
  (h_vessel_length : vessel_length = 20)
  (h_vessel_width : vessel_width = 15) :
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 11.25 :=
by sorry

end water_level_rise_l1220_122022


namespace system_solution_l1220_122009

theorem system_solution (x y : ℝ) : 2*x - y = 5 ∧ x - 2*y = 1 → x - y = 2 := by
  sorry

end system_solution_l1220_122009


namespace tan_increasing_on_interval_l1220_122055

open Real

theorem tan_increasing_on_interval :
  StrictMonoOn tan (Set.Ioo (π / 2) π) := by
  sorry

end tan_increasing_on_interval_l1220_122055


namespace five_by_five_perimeter_l1220_122019

/-- The number of points on the perimeter of a square grid -/
def perimeterPoints (n : ℕ) : ℕ := 4 * n - 4

/-- Theorem: The number of points on the perimeter of a 5x5 grid is 16 -/
theorem five_by_five_perimeter : perimeterPoints 5 = 16 := by
  sorry

end five_by_five_perimeter_l1220_122019


namespace geometric_sequence_a3_l1220_122087

/-- A geometric sequence with a₁ = 1 and a₅ = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 5 = 16 ∧ ∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence with a₁ = 1 and a₅ = 16, a₃ = 4 -/
theorem geometric_sequence_a3 (a : ℕ → ℝ) (h : geometric_sequence a) : a 3 = 4 := by
  sorry

end geometric_sequence_a3_l1220_122087


namespace quadratic_inequality_empty_solution_set_l1220_122020

theorem quadratic_inequality_empty_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*x + a^2 > 0) ↔ (a < -2 ∨ a > 2) := by
  sorry

end quadratic_inequality_empty_solution_set_l1220_122020


namespace linear_system_fraction_sum_l1220_122041

theorem linear_system_fraction_sum (a b c x y z : ℝ) 
  (eq1 : 17 * x + b * y + c * z = 0)
  (eq2 : a * x + 29 * y + c * z = 0)
  (eq3 : a * x + b * y + 53 * z = 0)
  (ha : a ≠ 17)
  (hx : x ≠ 0) :
  a / (a - 17) + b / (b - 29) + c / (c - 53) = 1 := by
sorry

end linear_system_fraction_sum_l1220_122041


namespace optimal_discount_order_l1220_122086

/-- Proves that the optimal order of applying discounts results in an additional savings of 125 cents --/
theorem optimal_discount_order (initial_price : ℝ) (flat_discount : ℝ) (percent_discount : ℝ) :
  initial_price = 30 →
  flat_discount = 5 →
  percent_discount = 0.25 →
  ((initial_price - flat_discount) * (1 - percent_discount) - 
   (initial_price * (1 - percent_discount) - flat_discount)) * 100 = 125 := by
  sorry

end optimal_discount_order_l1220_122086


namespace somu_age_problem_l1220_122052

/-- Proves that Somu was one-fifth of his father's age 5 years ago -/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) :
  somu_age = 10 →
  somu_age = father_age / 3 →
  somu_age - years_ago = (father_age - years_ago) / 5 →
  years_ago = 5 := by
  sorry

#check somu_age_problem

end somu_age_problem_l1220_122052


namespace average_decrease_l1220_122099

theorem average_decrease (n : ℕ) (old_avg new_obs : ℚ) : 
  n = 6 → 
  old_avg = 14 → 
  new_obs = 7 → 
  old_avg - (n * old_avg + new_obs) / (n + 1) = 1 := by
sorry

end average_decrease_l1220_122099


namespace basketball_league_games_l1220_122049

/-- Calculates the total number of games in a basketball league -/
def total_games (n : ℕ) (regular_games_per_pair : ℕ) (knockout_games_per_team : ℕ) : ℕ :=
  let regular_season_games := (n * (n - 1) / 2) * regular_games_per_pair
  let knockout_games := n * knockout_games_per_team / 2
  regular_season_games + knockout_games

/-- Theorem: In a 12-team basketball league where each team plays 4 games against every other team
    and participates in 2 knockout matches, the total number of games is 276 -/
theorem basketball_league_games :
  total_games 12 4 2 = 276 := by
  sorry

end basketball_league_games_l1220_122049


namespace count_valid_configurations_l1220_122036

/-- Represents a configuration of 8's with + signs inserted -/
structure Configuration where
  singles : ℕ  -- number of individual 8's
  doubles : ℕ  -- number of 88's
  triples : ℕ  -- number of 888's

/-- The total number of 8's used in a configuration -/
def Configuration.total_eights (c : Configuration) : ℕ :=
  c.singles + 2 * c.doubles + 3 * c.triples

/-- The sum of a configuration -/
def Configuration.sum (c : Configuration) : ℕ :=
  8 * c.singles + 88 * c.doubles + 888 * c.triples

/-- A configuration is valid if its sum is 8880 -/
def Configuration.is_valid (c : Configuration) : Prop :=
  c.sum = 8880

theorem count_valid_configurations :
  (∃ (s : Finset ℕ), s.card = 119 ∧
    (∀ n, n ∈ s ↔ ∃ c : Configuration, c.is_valid ∧ c.total_eights = n)) := by
  sorry

end count_valid_configurations_l1220_122036


namespace corner_sum_is_164_l1220_122029

/-- Represents a 9x9 checkerboard filled with numbers 1 through 81 in order across rows. -/
def Checkerboard := Fin 9 → Fin 9 → Nat

/-- The value at position (i, j) on the checkerboard. -/
def checkerboardValue (i j : Fin 9) : Nat :=
  i.val * 9 + j.val + 1

/-- The sum of the values in the four corners of the checkerboard. -/
def cornerSum (board : Checkerboard) : Nat :=
  board 0 0 + board 0 8 + board 8 0 + board 8 8

/-- Theorem stating that the sum of the numbers in the four corners of the checkerboard is 164. -/
theorem corner_sum_is_164 (board : Checkerboard) :
  (∀ i j : Fin 9, board i j = checkerboardValue i j) →
  cornerSum board = 164 := by
  sorry

end corner_sum_is_164_l1220_122029


namespace quadratic_linear_intersection_l1220_122069

theorem quadratic_linear_intersection
  (a d : ℝ) (x₁ x₂ : ℝ) (h_a : a ≠ 0) (h_d : d ≠ 0) (h_x : x₁ ≠ x₂)
  (y₁ : ℝ → ℝ) (y₂ : ℝ → ℝ) (y : ℝ → ℝ)
  (h_y₁ : ∀ x, y₁ x = a * (x - x₁) * (x - x₂))
  (h_y₂ : ∃ e, ∀ x, y₂ x = d * x + e)
  (h_intersect : y₂ x₁ = 0)
  (h_single_root : ∃! x, y x = 0) :
  x₂ - x₁ = d / a := by sorry

end quadratic_linear_intersection_l1220_122069


namespace range_of_a_l1220_122062

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ≤ -2 ∨ a = 1 := by
sorry

end range_of_a_l1220_122062


namespace isolated_sets_intersection_empty_l1220_122025

def is_isolated_element (x : ℤ) (A : Set ℤ) : Prop :=
  x ∈ A ∧ (x - 1) ∉ A ∧ (x + 1) ∉ A

def isolated_set (A : Set ℤ) : Set ℤ :=
  {x | is_isolated_element x A}

def M : Set ℤ := {0, 1, 3}
def N : Set ℤ := {0, 3, 4}

theorem isolated_sets_intersection_empty :
  (isolated_set M) ∩ (isolated_set N) = ∅ := by
  sorry

end isolated_sets_intersection_empty_l1220_122025


namespace sugar_problem_l1220_122060

theorem sugar_problem (initial_sugar : ℝ) : 
  (initial_sugar / 4 * 3.5 = 21) → initial_sugar = 24 := by
  sorry

end sugar_problem_l1220_122060


namespace line_through_point_perpendicular_to_line_l1220_122028

/-- Given a point A and two lines l₁ and l₂, this theorem states that
    l₂ passes through A and is perpendicular to l₁. -/
theorem line_through_point_perpendicular_to_line
  (A : ℝ × ℝ)  -- Point A
  (l₁ : ℝ → ℝ → Prop)  -- Line l₁
  (l₂ : ℝ → ℝ → Prop)  -- Line l₂
  (h₁ : l₁ = fun x y ↦ 2 * x + 3 * y + 4 = 0)  -- Equation of l₁
  (h₂ : l₂ = fun x y ↦ 3 * x - 2 * y + 7 = 0)  -- Equation of l₂
  (h₃ : A = (-1, 2))  -- Coordinates of point A
  : (l₂ (A.1) (A.2)) ∧  -- l₂ passes through A
    (∀ (x y : ℝ), l₁ x y → l₂ x y → (2 * 3 + 3 * (-2) = 0)) :=  -- l₁ ⊥ l₂
by sorry

end line_through_point_perpendicular_to_line_l1220_122028


namespace cos_54_degrees_l1220_122001

theorem cos_54_degrees : Real.cos (54 * π / 180) = Real.sqrt ((3 + Real.sqrt 5) / 8) := by
  sorry

end cos_54_degrees_l1220_122001


namespace function_value_at_sine_l1220_122042

/-- Given a function f(x) = 4x² + 2x, prove that f(sin(7π/6)) = 0 -/
theorem function_value_at_sine (f : ℝ → ℝ) : 
  (∀ x, f x = 4 * x^2 + 2 * x) → f (Real.sin (7 * π / 6)) = 0 := by
  sorry

end function_value_at_sine_l1220_122042


namespace zoo_trip_buses_l1220_122059

theorem zoo_trip_buses (total_students : ℕ) (students_per_bus : ℕ) (car_students : ℕ) : 
  total_students = 375 → students_per_bus = 53 → car_students = 4 →
  ((total_students - car_students + students_per_bus - 1) / students_per_bus : ℕ) = 8 := by
sorry

end zoo_trip_buses_l1220_122059


namespace cubic_equation_with_geometric_progression_roots_l1220_122035

theorem cubic_equation_with_geometric_progression_roots (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 11*x^2 + a*x - 8 = 0 ∧
    y^3 - 11*y^2 + a*y - 8 = 0 ∧
    z^3 - 11*z^2 + a*z - 8 = 0 ∧
    ∃ q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ y = x*q ∧ z = y*q) →
  a = 22 :=
by sorry

end cubic_equation_with_geometric_progression_roots_l1220_122035


namespace y_coord_Q_l1220_122090

/-- A line passing through the origin with slope 0.8 -/
def line (x : ℝ) : ℝ := 0.8 * x

/-- The x-coordinate of point Q -/
def x_coord_Q : ℝ := 6

/-- Theorem: The y-coordinate of point Q is 4.8 -/
theorem y_coord_Q : line x_coord_Q = 4.8 := by
  sorry

end y_coord_Q_l1220_122090


namespace equation_negative_root_l1220_122048

theorem equation_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 4^x - 2^(x-1) + a = 0) ↔ -1/2 < a ∧ a ≤ 1/16 := by
  sorry

end equation_negative_root_l1220_122048


namespace line_intersection_point_l1220_122011

theorem line_intersection_point :
  ∃! p : ℝ × ℝ, 
    5 * p.1 - 3 * p.2 = 15 ∧ 
    4 * p.1 + 2 * p.2 = 14 :=
by
  -- The proof goes here
  sorry

end line_intersection_point_l1220_122011


namespace largest_multiple_of_15_under_500_l1220_122083

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n > 0 → 15 * n < 500 → 15 * n ≤ 495 := by
  sorry

end largest_multiple_of_15_under_500_l1220_122083


namespace bagel_cost_is_1_50_l1220_122095

/-- The cost of a cup of coffee -/
def coffee_cost : ℝ := sorry

/-- The cost of a bagel -/
def bagel_cost : ℝ := sorry

/-- Condition 1: 3 cups of coffee and 2 bagels cost $12.75 -/
axiom condition1 : 3 * coffee_cost + 2 * bagel_cost = 12.75

/-- Condition 2: 2 cups of coffee and 5 bagels cost $14.00 -/
axiom condition2 : 2 * coffee_cost + 5 * bagel_cost = 14.00

/-- Theorem: The cost of one bagel is $1.50 -/
theorem bagel_cost_is_1_50 : bagel_cost = 1.50 := by sorry

end bagel_cost_is_1_50_l1220_122095


namespace workshop_workers_l1220_122044

/-- The total number of workers in a workshop, given specific salary conditions -/
theorem workshop_workers (average_salary : ℚ) (technician_salary : ℚ) (other_salary : ℚ)
  (h1 : average_salary = 8000)
  (h2 : technician_salary = 18000)
  (h3 : other_salary = 6000) :
  ∃ (total_workers : ℕ), 
    (7 : ℚ) * technician_salary + (total_workers - 7 : ℚ) * other_salary = (total_workers : ℚ) * average_salary ∧
    total_workers = 42 := by
  sorry

end workshop_workers_l1220_122044


namespace success_permutations_l1220_122093

def word := "SUCCESS"

-- Define the counts of each letter
def s_count := 3
def c_count := 2
def u_count := 1
def e_count := 1

-- Define the total number of letters
def total_letters := s_count + c_count + u_count + e_count

-- Theorem statement
theorem success_permutations :
  (Nat.factorial total_letters) / 
  (Nat.factorial s_count * Nat.factorial c_count * Nat.factorial u_count * Nat.factorial e_count) = 420 := by
  sorry

end success_permutations_l1220_122093


namespace max_cross_section_area_l1220_122065

/-- Represents a regular tetrahedron -/
structure RegularTetrahedron where
  sideLength : ℝ
  sideLength_pos : 0 < sideLength

/-- Represents a plane that cuts the tetrahedron parallel to two opposite edges -/
structure CuttingPlane where
  tetrahedron : RegularTetrahedron
  distanceFromEdge : ℝ
  distance_nonneg : 0 ≤ distanceFromEdge
  distance_bound : distanceFromEdge ≤ tetrahedron.sideLength

/-- The area of the cross-section formed by the cutting plane -/
def crossSectionArea (plane : CuttingPlane) : ℝ :=
  plane.distanceFromEdge * (plane.tetrahedron.sideLength - plane.distanceFromEdge)

/-- The theorem stating that the maximum cross-section area is a²/4 -/
theorem max_cross_section_area (t : RegularTetrahedron) :
  ∃ (plane : CuttingPlane), plane.tetrahedron = t ∧
  ∀ (p : CuttingPlane), p.tetrahedron = t →
  crossSectionArea p ≤ crossSectionArea plane ∧
  crossSectionArea plane = t.sideLength^2 / 4 :=
sorry

end max_cross_section_area_l1220_122065


namespace sum_of_coefficients_factorization_l1220_122064

theorem sum_of_coefficients_factorization (x y : ℝ) : 
  (∃ a b c d e f g h i j : ℤ, 
    27 * x^9 - 512 * y^9 = (a*x + b*y) * (c*x^2 + d*x*y + e*y^2) * (f*x + g*y) * (h*x^2 + i*x*y + j*y^2) ∧
    a + b + c + d + e + f + g + h + i + j = 32) :=
by sorry

end sum_of_coefficients_factorization_l1220_122064
