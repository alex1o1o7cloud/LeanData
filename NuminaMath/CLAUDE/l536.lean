import Mathlib

namespace NUMINAMATH_CALUDE_age_puzzle_l536_53667

theorem age_puzzle (A : ℕ) (x : ℕ) (h1 : A = 18) (h2 : 3 * (A + x) - 3 * (A - 3) = A) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_age_puzzle_l536_53667


namespace NUMINAMATH_CALUDE_exist_three_permuted_numbers_l536_53679

/-- A function that checks if a number is a five-digit number in the decimal system -/
def isFiveDigit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A function that checks if two numbers are permutations of each other -/
def isPermutation (a b : ℕ) : Prop :=
  ∃ (digits_a digits_b : List ℕ),
    digits_a.length = 5 ∧
    digits_b.length = 5 ∧
    digits_a.toFinset = digits_b.toFinset ∧
    a = digits_a.foldl (fun acc d => acc * 10 + d) 0 ∧
    b = digits_b.foldl (fun acc d => acc * 10 + d) 0

/-- Theorem stating that there exist three five-digit numbers that are permutations of each other,
    where the sum of two equals twice the third -/
theorem exist_three_permuted_numbers :
  ∃ (a b c : ℕ),
    isFiveDigit a ∧ isFiveDigit b ∧ isFiveDigit c ∧
    isPermutation a b ∧ isPermutation b c ∧ isPermutation a c ∧
    a + b = 2 * c := by
  sorry

end NUMINAMATH_CALUDE_exist_three_permuted_numbers_l536_53679


namespace NUMINAMATH_CALUDE_proof_by_contradiction_method_l536_53629

-- Define what proof by contradiction means
def proof_by_contradiction (P : Prop) : Prop :=
  ∃ (proof : ¬P → False), P

-- State the theorem
theorem proof_by_contradiction_method :
  ¬(∀ (P Q : Prop), proof_by_contradiction P ↔ (¬P ∧ ¬Q → False)) :=
sorry

end NUMINAMATH_CALUDE_proof_by_contradiction_method_l536_53629


namespace NUMINAMATH_CALUDE_cube_root_nine_inequality_false_l536_53688

theorem cube_root_nine_inequality_false : 
  ¬(∀ n : ℤ, (n : ℝ) < (9 : ℝ)^(1/3) ∧ (9 : ℝ)^(1/3) < (n : ℝ) + 1 → n = 3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_nine_inequality_false_l536_53688


namespace NUMINAMATH_CALUDE_midpoint_hexagon_area_l536_53696

/-- A regular octagon -/
structure RegularOctagon where
  /-- The apothem of the octagon -/
  apothem : ℝ

/-- A hexagon formed by connecting midpoints of six consecutive sides of a regular octagon -/
def MidpointHexagon (octagon : RegularOctagon) : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- 
Given a regular octagon with apothem 3, the area of the hexagon formed by 
connecting the midpoints of six consecutive sides is 162√3 - 108√6
-/
theorem midpoint_hexagon_area (octagon : RegularOctagon) 
  (h : octagon.apothem = 3) : 
  area (MidpointHexagon octagon) = 162 * Real.sqrt 3 - 108 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_midpoint_hexagon_area_l536_53696


namespace NUMINAMATH_CALUDE_intercept_ratio_l536_53654

/-- Given two lines intersecting the y-axis at different points:
    - Line 1 has y-intercept 2, slope 5, and x-intercept (u, 0)
    - Line 2 has y-intercept 3, slope -7, and x-intercept (v, 0)
    The ratio of u to v is -14/15 -/
theorem intercept_ratio (u v : ℝ) : 
  (2 : ℝ) + 5 * u = 0 →  -- Line 1 equation at x-intercept
  (3 : ℝ) - 7 * v = 0 →  -- Line 2 equation at x-intercept
  u / v = -14 / 15 := by
  sorry

end NUMINAMATH_CALUDE_intercept_ratio_l536_53654


namespace NUMINAMATH_CALUDE_cloth_cost_per_meter_l536_53686

theorem cloth_cost_per_meter (total_length : ℝ) (total_cost : ℝ) 
  (h1 : total_length = 9.25)
  (h2 : total_cost = 434.75) :
  total_cost / total_length = 47 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_per_meter_l536_53686


namespace NUMINAMATH_CALUDE_integral_x_plus_exp_x_l536_53630

theorem integral_x_plus_exp_x : ∫ x in (0:ℝ)..2, (x + Real.exp x) = Real.exp 2 + 1 := by sorry

end NUMINAMATH_CALUDE_integral_x_plus_exp_x_l536_53630


namespace NUMINAMATH_CALUDE_binary_1011001100_equals_octal_5460_l536_53642

def binary_to_octal (b : ℕ) : ℕ :=
  sorry

theorem binary_1011001100_equals_octal_5460 :
  binary_to_octal 1011001100 = 5460 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011001100_equals_octal_5460_l536_53642


namespace NUMINAMATH_CALUDE_room_width_calculation_l536_53610

/-- Given a rectangular room with known length, paving cost per square meter, and total paving cost,
    calculate the width of the room. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : cost_per_sqm = 700)
    (h3 : total_cost = 14437.5) :
    total_cost / cost_per_sqm / length = 3.75 := by
  sorry

end NUMINAMATH_CALUDE_room_width_calculation_l536_53610


namespace NUMINAMATH_CALUDE_fountain_area_l536_53689

-- Define the fountain
structure Fountain :=
  (ab : ℝ)  -- Length of AB
  (dc : ℝ)  -- Length of DC
  (h_ab_positive : ab > 0)
  (h_dc_positive : dc > 0)
  (h_d_midpoint : True)  -- Represents that D is the midpoint of AB
  (h_c_center : True)    -- Represents that C is the center of the fountain

-- Define the theorem
theorem fountain_area (f : Fountain) (h_ab : f.ab = 20) (h_dc : f.dc = 12) : 
  (π * (f.ab / 2) ^ 2 + π * f.dc ^ 2) = 244 * π := by
  sorry


end NUMINAMATH_CALUDE_fountain_area_l536_53689


namespace NUMINAMATH_CALUDE_expression_value_l536_53690

theorem expression_value (a b c : ℤ) : 
  (-a = 2) → (abs b = 6) → (-c + b = -10) → (8 - a + b - c = 0) := by
sorry

end NUMINAMATH_CALUDE_expression_value_l536_53690


namespace NUMINAMATH_CALUDE_total_price_houses_l536_53606

/-- The total price of two houses, given the price of the first house and that the second house is twice as expensive. -/
def total_price (price_first_house : ℕ) : ℕ :=
  price_first_house + 2 * price_first_house

/-- Theorem stating that the total price of the two houses is $600,000 when the first house costs $200,000. -/
theorem total_price_houses : total_price 200000 = 600000 := by
  sorry

end NUMINAMATH_CALUDE_total_price_houses_l536_53606


namespace NUMINAMATH_CALUDE_speed_ratio_with_head_start_l536_53623

/-- The ratio of speeds between two runners in a race with a head start --/
theorem speed_ratio_with_head_start (va vb : ℝ) (h : va > 0 ∧ vb > 0) :
  (∃ k : ℝ, va = k * vb) →
  (va * (1 - 0.15625) = vb) →
  va / vb = 32 / 27 := by
sorry

end NUMINAMATH_CALUDE_speed_ratio_with_head_start_l536_53623


namespace NUMINAMATH_CALUDE_sum_of_digits_of_4444_power_4444_l536_53611

-- Define the sum of digits function
def S (n : ℕ) : ℕ := sorry

-- State the theorem
theorem sum_of_digits_of_4444_power_4444 :
  ∃ (S : ℕ → ℕ),
    (∀ n : ℕ, S n % 9 = n % 9) →
    S (S (S (4444^4444))) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_4444_power_4444_l536_53611


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l536_53695

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 30 * S) :
  (S - C) / C * 100 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l536_53695


namespace NUMINAMATH_CALUDE_sin_450_degrees_l536_53683

theorem sin_450_degrees : Real.sin (450 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_450_degrees_l536_53683


namespace NUMINAMATH_CALUDE_unique_number_theorem_l536_53664

def A₁ (n : ℕ) : Prop := n < 12
def A₂ (n : ℕ) : Prop := ¬(7 ∣ n)
def A₃ (n : ℕ) : Prop := 5 * n < 70

def B₁ (n : ℕ) : Prop := 12 * n > 1000
def B₂ (n : ℕ) : Prop := 10 ∣ n
def B₃ (n : ℕ) : Prop := n > 100

def C₁ (n : ℕ) : Prop := 4 ∣ n
def C₂ (n : ℕ) : Prop := 11 * n < 1000
def C₃ (n : ℕ) : Prop := 9 ∣ n

def D₁ (n : ℕ) : Prop := n < 20
def D₂ (n : ℕ) : Prop := Nat.Prime n
def D₃ (n : ℕ) : Prop := 7 ∣ n

def at_least_one_true (p q r : Prop) : Prop := p ∨ q ∨ r
def at_least_one_false (p q r : Prop) : Prop := ¬p ∨ ¬q ∨ ¬r

theorem unique_number_theorem (n : ℕ) : 
  (at_least_one_true (A₁ n) (A₂ n) (A₃ n)) ∧ 
  (at_least_one_false (A₁ n) (A₂ n) (A₃ n)) ∧
  (at_least_one_true (B₁ n) (B₂ n) (B₃ n)) ∧ 
  (at_least_one_false (B₁ n) (B₂ n) (B₃ n)) ∧
  (at_least_one_true (C₁ n) (C₂ n) (C₃ n)) ∧ 
  (at_least_one_false (C₁ n) (C₂ n) (C₃ n)) ∧
  (at_least_one_true (D₁ n) (D₂ n) (D₃ n)) ∧ 
  (at_least_one_false (D₁ n) (D₂ n) (D₃ n)) →
  n = 89 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_theorem_l536_53664


namespace NUMINAMATH_CALUDE_arc_angle_proof_l536_53635

/-- Given a circle with radius 3 cm and an arc length of π/2 cm, 
    prove that the corresponding central angle is 30°. -/
theorem arc_angle_proof (r : ℝ) (l : ℝ) (θ : ℝ) : 
  r = 3 → l = π / 2 → θ = (l * 180) / (π * r) → θ = 30 := by
  sorry

end NUMINAMATH_CALUDE_arc_angle_proof_l536_53635


namespace NUMINAMATH_CALUDE_circle_radius_with_tangents_l536_53613

/-- Given a circle with parallel tangents and a third tangent, prove the radius. -/
theorem circle_radius_with_tangents 
  (AB CD DE : ℝ) 
  (h_AB : AB = 7)
  (h_CD : CD = 12)
  (h_DE : DE = 3) : 
  ∃ (r : ℝ), r = 3 * Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_circle_radius_with_tangents_l536_53613


namespace NUMINAMATH_CALUDE_root_sum_cube_theorem_l536_53645

theorem root_sum_cube_theorem (a : ℝ) (x₁ x₂ x₃ : ℝ) : 
  (x₁^3 - 6*x₁^2 + a*x₁ + a = 0) →
  (x₂^3 - 6*x₂^2 + a*x₂ + a = 0) →
  (x₃^3 - 6*x₃^2 + a*x₃ + a = 0) →
  ((x₁ - 3)^3 + (x₂ - 3)^3 + (x₃ - 3)^3 = 0) →
  (a = 9) := by
sorry

end NUMINAMATH_CALUDE_root_sum_cube_theorem_l536_53645


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l536_53658

/-- Given Isabella's initial and final hair lengths, prove the amount of hair growth. -/
theorem isabellas_hair_growth 
  (initial_length : ℝ) 
  (final_length : ℝ) 
  (h1 : initial_length = 18) 
  (h2 : final_length = 24) : 
  final_length - initial_length = 6 := by
sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l536_53658


namespace NUMINAMATH_CALUDE_system_solution_product_l536_53680

theorem system_solution_product : 
  ∃ (a b c d : ℚ),
    (4*a + 2*b + 6*c + 8*d = 48) ∧
    (2*(d+c) = b) ∧
    (4*b + 2*c = a) ∧
    (c + 2 = d) ∧
    (a * b * c * d = -88807680/4879681) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_product_l536_53680


namespace NUMINAMATH_CALUDE_solution_set_inequality_l536_53620

theorem solution_set_inequality (x : ℝ) : 
  (x ≠ 2) → ((2 * x + 5) / (x - 2) < 1 ↔ -7 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l536_53620


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l536_53615

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) / a n = a 2 / a 1

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n, a n > 0) →
  a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25 →
  a 3 + a 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l536_53615


namespace NUMINAMATH_CALUDE_max_min_f_on_I_l536_53697

-- Define the function f(x) = x^3 - 3x + 1
def f (x : ℝ) : ℝ := x^3 - 3*x + 1

-- Define the closed interval [-3, 0]
def I : Set ℝ := Set.Icc (-3) 0

-- State the theorem
theorem max_min_f_on_I :
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f y ≤ f x) ∧
  (∃ (x : ℝ), x ∈ I ∧ ∀ (y : ℝ), y ∈ I → f x ≤ f y) ∧
  (∀ (x : ℝ), x ∈ I → f x ≤ 3) ∧
  (∀ (x : ℝ), x ∈ I → -17 ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_max_min_f_on_I_l536_53697


namespace NUMINAMATH_CALUDE_band_sections_fraction_l536_53634

theorem band_sections_fraction (trumpet_fraction trombone_fraction : ℝ) 
  (h1 : trumpet_fraction = 0.5)
  (h2 : trombone_fraction = 0.12) :
  trumpet_fraction + trombone_fraction = 0.62 := by
  sorry

end NUMINAMATH_CALUDE_band_sections_fraction_l536_53634


namespace NUMINAMATH_CALUDE_angle_difference_range_l536_53639

theorem angle_difference_range (α β : ℝ) 
  (h1 : -π/2 < α) (h2 : α < 0) (h3 : 0 < β) (h4 : β < π/3) : 
  -5*π/6 < α - β ∧ α - β < 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_difference_range_l536_53639


namespace NUMINAMATH_CALUDE_max_overlap_area_isosceles_triangles_l536_53600

/-- The maximal area of overlap between two congruent right-angled isosceles triangles -/
theorem max_overlap_area_isosceles_triangles :
  ∃ (overlap_area : ℝ),
    overlap_area = 2/9 ∧
    ∀ (x : ℝ),
      0 ≤ x ∧ x ≤ 1 →
      let triangle_area := 1/4 * (1 - x)^2
      let pentagon_area := 1/4 * (1 - x) * (3*x + 1)
      overlap_area ≥ max triangle_area pentagon_area :=
by sorry

end NUMINAMATH_CALUDE_max_overlap_area_isosceles_triangles_l536_53600


namespace NUMINAMATH_CALUDE_unique_base_solution_l536_53601

-- Define a function to convert a number from base h to decimal
def to_decimal (digits : List Nat) (h : Nat) : Nat :=
  digits.foldl (fun acc d => acc * h + d) 0

-- Define the equation in base h
def equation_holds (h : Nat) : Prop :=
  to_decimal [7, 3, 6, 4] h + to_decimal [8, 4, 2, 1] h = to_decimal [1, 7, 2, 8, 5] h

-- Theorem statement
theorem unique_base_solution :
  ∃! h : Nat, h > 1 ∧ equation_holds h :=
sorry

end NUMINAMATH_CALUDE_unique_base_solution_l536_53601


namespace NUMINAMATH_CALUDE_jacob_excess_calories_l536_53699

def jacob_calorie_problem (calorie_goal : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) : Prop :=
  calorie_goal < 1800 ∧
  breakfast = 400 ∧
  lunch = 900 ∧
  dinner = 1100 ∧
  (breakfast + lunch + dinner) - calorie_goal = 600

theorem jacob_excess_calories :
  ∃ (calorie_goal : ℕ), jacob_calorie_problem calorie_goal 400 900 1100 :=
by
  sorry

end NUMINAMATH_CALUDE_jacob_excess_calories_l536_53699


namespace NUMINAMATH_CALUDE_largest_quotient_l536_53647

def digits : List Nat := [4, 2, 8, 1, 9]

def is_valid_pair (a b : Nat) : Prop :=
  a ≥ 100 ∧ a < 1000 ∧ b ≥ 10 ∧ b < 100 ∧
  (∃ (d1 d2 d3 d4 d5 : Nat),
    d1 ∈ digits ∧ d2 ∈ digits ∧ d3 ∈ digits ∧ d4 ∈ digits ∧ d5 ∈ digits ∧
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d3 ≠ d4 ∧ d3 ≠ d5 ∧ d4 ≠ d5 ∧
    a = 100 * d1 + 10 * d2 + d3 ∧ b = 10 * d4 + d5)

theorem largest_quotient :
  ∀ (a b : Nat), is_valid_pair a b →
  ∃ (q : Nat), a / b = q ∧ q ≤ 82 ∧
  (∀ (c d : Nat), is_valid_pair c d → c / d ≤ q) :=
sorry

end NUMINAMATH_CALUDE_largest_quotient_l536_53647


namespace NUMINAMATH_CALUDE_market_equilibrium_and_subsidy_effect_l536_53655

/-- Supply function -/
def supply (p : ℝ) : ℝ := 2 + 8 * p

/-- Demand function (to be derived) -/
def demand (p : ℝ) : ℝ := 12 - 2 * p

/-- Equilibrium price -/
def equilibrium_price : ℝ := 1

/-- Equilibrium quantity -/
def equilibrium_quantity : ℝ := 10

/-- Subsidy amount -/
def subsidy : ℝ := 1

/-- New supply function after subsidy -/
def new_supply (p : ℝ) : ℝ := supply (p + subsidy)

/-- New equilibrium price after subsidy -/
def new_equilibrium_price : ℝ := 0.2

/-- New equilibrium quantity after subsidy -/
def new_equilibrium_quantity : ℝ := 11.6

theorem market_equilibrium_and_subsidy_effect :
  (demand 2 = 8) ∧
  (demand 3 = 6) ∧
  (supply equilibrium_price = demand equilibrium_price) ∧
  (supply equilibrium_price = equilibrium_quantity) ∧
  (new_supply new_equilibrium_price = demand new_equilibrium_price) ∧
  (new_equilibrium_quantity - equilibrium_quantity = 1.6) := by
  sorry

end NUMINAMATH_CALUDE_market_equilibrium_and_subsidy_effect_l536_53655


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l536_53676

-- Define a function to convert a base 7 number to base 10
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

-- Define the given base 7 number
def base7Number : List Nat := [1, 2, 3, 5, 4]

-- Theorem to prove
theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 11481 := by
  sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l536_53676


namespace NUMINAMATH_CALUDE_a_2021_eq_6_l536_53618

def a : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => 
    if n % 3 = 0 then a (n / 3)
    else a (n / 3) + 1

theorem a_2021_eq_6 : a 2021 = 6 := by
  sorry

end NUMINAMATH_CALUDE_a_2021_eq_6_l536_53618


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l536_53692

theorem degree_to_radian_conversion (angle_deg : ℝ) (angle_rad : ℝ) : 
  angle_deg = 15 → angle_rad = π / 12 := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l536_53692


namespace NUMINAMATH_CALUDE_fraction_order_l536_53691

theorem fraction_order : (25 : ℚ) / 21 < 23 / 19 ∧ 23 / 19 < 21 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l536_53691


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l536_53659

theorem trigonometric_simplification :
  (Real.sin (15 * π / 180) + Real.sin (45 * π / 180)) /
  (Real.cos (15 * π / 180) + Real.cos (45 * π / 180)) =
  Real.tan (30 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l536_53659


namespace NUMINAMATH_CALUDE_total_price_is_530_l536_53694

/-- The total price of hats given the number of hats, their prices, and the number of green hats. -/
def total_price (total_hats : ℕ) (blue_price green_price : ℕ) (green_hats : ℕ) : ℕ :=
  let blue_hats := total_hats - green_hats
  blue_price * blue_hats + green_price * green_hats

/-- Theorem stating that the total price of hats is $530 given the specific conditions. -/
theorem total_price_is_530 :
  total_price 85 6 7 20 = 530 :=
by sorry

end NUMINAMATH_CALUDE_total_price_is_530_l536_53694


namespace NUMINAMATH_CALUDE_angle_F_is_60_l536_53628

/-- A trapezoid with specific angle relationships -/
structure SpecialTrapezoid where
  -- Angles of the trapezoid
  angleE : ℝ
  angleF : ℝ
  angleG : ℝ
  angleH : ℝ
  -- Conditions given in the problem
  parallel_sides : True  -- Represents that EF and GH are parallel
  angle_E_triple_H : angleE = 3 * angleH
  angle_G_double_F : angleG = 2 * angleF
  -- Properties of a trapezoid
  sum_angles : angleE + angleF + angleG + angleH = 360
  opposite_angles_sum : angleF + angleG = 180

/-- Theorem stating that in the special trapezoid, angle F measures 60 degrees -/
theorem angle_F_is_60 (t : SpecialTrapezoid) : t.angleF = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_F_is_60_l536_53628


namespace NUMINAMATH_CALUDE_reader_group_size_l536_53653

theorem reader_group_size (S L B : ℕ) (h1 : S = 250) (h2 : L = 230) (h3 : B = 80) :
  S + L - B = 400 := by
  sorry

end NUMINAMATH_CALUDE_reader_group_size_l536_53653


namespace NUMINAMATH_CALUDE_shaded_area_of_square_with_rectangles_shaded_area_is_22_l536_53643

/-- The area of the shaded L-shaped region in a square with three rectangles removed -/
theorem shaded_area_of_square_with_rectangles (side_length : ℝ) 
  (rect1_length rect1_width : ℝ) 
  (rect2_length rect2_width : ℝ) 
  (rect3_length rect3_width : ℝ) : ℝ :=
  side_length * side_length - (rect1_length * rect1_width + rect2_length * rect2_width + rect3_length * rect3_width)

/-- The area of the shaded L-shaped region is 22 square units -/
theorem shaded_area_is_22 :
  shaded_area_of_square_with_rectangles 6 3 1 4 2 1 3 = 22 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_square_with_rectangles_shaded_area_is_22_l536_53643


namespace NUMINAMATH_CALUDE_intersection_point_after_rotation_l536_53666

theorem intersection_point_after_rotation (θ : Real) : 
  0 < θ ∧ θ < π / 2 → 
  (fun φ ↦ (Real.cos φ, Real.sin φ)) (θ + π / 2) = (-Real.sin θ, Real.cos θ) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_after_rotation_l536_53666


namespace NUMINAMATH_CALUDE_polynomial_roots_l536_53633

theorem polynomial_roots : 
  let p (x : ℝ) := 3 * x^4 - 2 * x^3 - 4 * x^2 - 2 * x + 3
  ∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = (-1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l536_53633


namespace NUMINAMATH_CALUDE_melissa_initial_oranges_l536_53616

/-- The number of oranges Melissa has initially -/
def initial_oranges : ℕ := sorry

/-- The number of oranges John takes away -/
def oranges_taken : ℕ := 19

/-- The number of oranges Melissa has left -/
def oranges_left : ℕ := 51

/-- Theorem stating that Melissa's initial number of oranges is 70 -/
theorem melissa_initial_oranges : 
  initial_oranges = oranges_taken + oranges_left :=
sorry

end NUMINAMATH_CALUDE_melissa_initial_oranges_l536_53616


namespace NUMINAMATH_CALUDE_basketball_shots_improvement_l536_53671

theorem basketball_shots_improvement (initial_shots : ℕ) (initial_success_rate : ℚ)
  (additional_shots : ℕ) (new_success_rate : ℚ) :
  initial_shots = 30 →
  initial_success_rate = 60 / 100 →
  additional_shots = 10 →
  new_success_rate = 62 / 100 →
  (↑(initial_shots * initial_success_rate.num / initial_success_rate.den +
    (new_success_rate * ↑(initial_shots + additional_shots)).num / (new_success_rate * ↑(initial_shots + additional_shots)).den -
    (initial_success_rate * ↑initial_shots).num / (initial_success_rate * ↑initial_shots).den) : ℚ) = 7 :=
by
  sorry

#check basketball_shots_improvement

end NUMINAMATH_CALUDE_basketball_shots_improvement_l536_53671


namespace NUMINAMATH_CALUDE_fraction_simplification_l536_53693

theorem fraction_simplification :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 3) = 177 / 182 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l536_53693


namespace NUMINAMATH_CALUDE_probability_of_a_l536_53668

theorem probability_of_a (a b : Set α) (p : Set α → ℝ) 
  (h1 : p b = 2/5)
  (h2 : p (a ∩ b) = p a * p b)
  (h3 : p (a ∩ b) = 0.16000000000000003) :
  p a = 0.4 := by
sorry

end NUMINAMATH_CALUDE_probability_of_a_l536_53668


namespace NUMINAMATH_CALUDE_largest_angle_hexagon_l536_53627

/-- The largest interior angle of a convex hexagon with six consecutive integer angles -/
def largest_hexagon_angle : ℝ := 122.5

/-- A hexagon with six consecutive integer angles -/
structure ConsecutiveAngleHexagon where
  -- The smallest angle of the hexagon
  base_angle : ℝ
  -- Predicate ensuring the angles are consecutive integers
  consecutive_integers : ∀ i : Fin 6, (base_angle + i) = ↑(⌊base_angle⌋ + i)

/-- Theorem stating that the largest angle in a convex hexagon with six consecutive integer angles is 122.5° -/
theorem largest_angle_hexagon (h : ConsecutiveAngleHexagon) : 
  (h.base_angle + 5) = largest_hexagon_angle := by
  sorry

/-- The sum of interior angles of a hexagon is 720° -/
axiom sum_hexagon_angles : ∀ (h : ConsecutiveAngleHexagon), 
  (h.base_angle * 6 + 15) = 720

end NUMINAMATH_CALUDE_largest_angle_hexagon_l536_53627


namespace NUMINAMATH_CALUDE_cos_120_degrees_l536_53660

theorem cos_120_degrees : Real.cos (2 * π / 3) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l536_53660


namespace NUMINAMATH_CALUDE_phi_value_l536_53661

/-- Given a function f(x) = 2sin(ωx + φ) with the following properties:
    - ω > 0
    - |φ| < π/2
    - x = 5π/8 is an axis of symmetry for y = f(x)
    - x = 11π/8 is a zero of f(x)
    - The smallest positive period of f(x) is greater than 2π
    Prove that φ = π/12 -/
theorem phi_value (ω φ : Real) (h1 : ω > 0) (h2 : |φ| < π/2)
  (h3 : ∀ x, 2 * Real.sin (ω * (5*π/4 - (x - 5*π/8)) + φ) = 2 * Real.sin (ω * x + φ))
  (h4 : 2 * Real.sin (ω * 11*π/8 + φ) = 0)
  (h5 : 2*π / ω > 2*π) : φ = π/12 := by
  sorry

end NUMINAMATH_CALUDE_phi_value_l536_53661


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l536_53607

theorem quadratic_equation_solution :
  ∃ x : ℝ, x^2 + 4*x + 3 = -(x + 3)*(x + 5) ∧ x = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l536_53607


namespace NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_lt_zero_l536_53678

theorem solution_set_x_squared_minus_one_lt_zero :
  Set.Ioo (-1 : ℝ) 1 = {x : ℝ | x^2 - 1 < 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_minus_one_lt_zero_l536_53678


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l536_53638

theorem equation_represents_two_lines :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    x^2 - 50*y^2 - 16*x + 64 = 0 ↔ (x = a*y + b ∨ x = -a*y + b) :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l536_53638


namespace NUMINAMATH_CALUDE_book_purchase_total_price_l536_53674

theorem book_purchase_total_price
  (total_books : ℕ)
  (math_books : ℕ)
  (math_book_price : ℕ)
  (history_book_price : ℕ)
  (h1 : total_books = 80)
  (h2 : math_books = 27)
  (h3 : math_book_price = 4)
  (h4 : history_book_price = 5) :
  let history_books := total_books - math_books
  let total_price := math_books * math_book_price + history_books * history_book_price
  total_price = 373 := by
sorry

end NUMINAMATH_CALUDE_book_purchase_total_price_l536_53674


namespace NUMINAMATH_CALUDE_boat_downstream_distance_l536_53621

/-- Proves that given a boat with a speed of 20 km/hr in still water and a stream with
    speed of 6 km/hr, if the boat travels the same time downstream as it does to
    travel 14 km upstream, then the distance traveled downstream is 26 km. -/
theorem boat_downstream_distance
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (upstream_distance : ℝ)
  (h1 : boat_speed = 20)
  (h2 : stream_speed = 6)
  (h3 : upstream_distance = 14)
  (h4 : (upstream_distance / (boat_speed - stream_speed)) =
        (downstream_distance / (boat_speed + stream_speed))) :
  downstream_distance = 26 :=
by
  sorry


end NUMINAMATH_CALUDE_boat_downstream_distance_l536_53621


namespace NUMINAMATH_CALUDE_circumcenter_equidistant_l536_53682

-- Define a triangle in 2D space
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter of a triangle
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a distance function between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem: The circumcenter is equidistant from all vertices
theorem circumcenter_equidistant (t : Triangle) :
  distance (circumcenter t) t.A = distance (circumcenter t) t.B ∧
  distance (circumcenter t) t.B = distance (circumcenter t) t.C :=
sorry

end NUMINAMATH_CALUDE_circumcenter_equidistant_l536_53682


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l536_53608

theorem solution_satisfies_equations :
  let x : ℚ := -5/7
  let y : ℚ := -18/7
  (6 * x + 3 * y = -12) ∧ (4 * x = 5 * y + 10) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l536_53608


namespace NUMINAMATH_CALUDE_simple_interest_principal_l536_53677

/-- Simple interest calculation --/
theorem simple_interest_principal
  (interest : ℚ)
  (rate : ℚ)
  (time : ℚ)
  (h1 : interest = 160)
  (h2 : rate = 4 / 100)
  (h3 : time = 5) :
  interest = (800 * rate * time) :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l536_53677


namespace NUMINAMATH_CALUDE_gcd_of_2535_5929_11629_l536_53648

theorem gcd_of_2535_5929_11629 : Nat.gcd 2535 (Nat.gcd 5929 11629) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_2535_5929_11629_l536_53648


namespace NUMINAMATH_CALUDE_sin_negative_120_degrees_l536_53646

theorem sin_negative_120_degrees : Real.sin (-(120 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_120_degrees_l536_53646


namespace NUMINAMATH_CALUDE_greatest_power_of_three_specific_case_l536_53624

theorem greatest_power_of_three (n : ℕ) : ∃ (k : ℕ), (3^n : ℤ) ∣ (6^n - 3^n) ∧ ¬(3^(n+1) : ℤ) ∣ (6^n - 3^n) :=
by
  sorry

theorem specific_case : ∃ (k : ℕ), (3^1503 : ℤ) ∣ (6^1503 - 3^1503) ∧ ¬(3^1504 : ℤ) ∣ (6^1503 - 3^1503) :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_specific_case_l536_53624


namespace NUMINAMATH_CALUDE_lcm_48_180_l536_53636

theorem lcm_48_180 : Nat.lcm 48 180 = 720 := by
  sorry

end NUMINAMATH_CALUDE_lcm_48_180_l536_53636


namespace NUMINAMATH_CALUDE_proposition_1_proposition_4_proposition_5_l536_53617

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations and operations
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (skew : Line → Line → Prop)
variable (point_not_on_line : Line → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_perpendicular_plane : Line → Plane → Prop)

-- Proposition 1
theorem proposition_1 (l m : Line) (α : Plane) :
  contains α m → contains α l → point_not_on_line m → skew l m :=
sorry

-- Proposition 4
theorem proposition_4 (l m : Line) (α : Plane) :
  line_perpendicular_plane m α → line_parallel_plane l α → perpendicular l m :=
sorry

-- Proposition 5
theorem proposition_5 (m n : Line) (α β : Plane) :
  skew m n → contains α m → line_parallel_plane m β → 
  contains β n → line_parallel_plane n α → parallel_plane α β :=
sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_4_proposition_5_l536_53617


namespace NUMINAMATH_CALUDE_rectangular_to_polar_l536_53685

theorem rectangular_to_polar :
  let x : ℝ := 2 * Real.sqrt 3
  let y : ℝ := -2
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := if x > 0 ∧ y < 0 then 2 * π + Real.arctan (y / x) else Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = 4 ∧ θ = 11 * π / 6 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_l536_53685


namespace NUMINAMATH_CALUDE_sequence_sum_l536_53631

theorem sequence_sum (a b c d : ℕ) 
  (h1 : 0 < a ∧ a < b ∧ b < c ∧ c < d)
  (h2 : b * a = c * a)
  (h3 : c - b = d - c)
  (h4 : d - a = 36) : 
  a + b + c + d = 1188 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_l536_53631


namespace NUMINAMATH_CALUDE_reverse_digit_numbers_base_9_11_l536_53603

def is_three_digit_base (n : ℕ) (base : ℕ) : Prop :=
  base ^ 2 ≤ n ∧ n < base ^ 3

def digits_base (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

def reverse_digits (l : List ℕ) : List ℕ :=
  sorry

theorem reverse_digit_numbers_base_9_11 :
  ∃! (S : Finset ℕ),
    (∀ n ∈ S,
      is_three_digit_base n 9 ∧
      is_three_digit_base n 11 ∧
      digits_base n 9 = reverse_digits (digits_base n 11)) ∧
    S.card = 2 ∧
    245 ∈ S ∧
    490 ∈ S :=
  sorry

end NUMINAMATH_CALUDE_reverse_digit_numbers_base_9_11_l536_53603


namespace NUMINAMATH_CALUDE_money_difference_l536_53656

/-- The problem statement about Isabella, Sam, and Giselle's money --/
theorem money_difference (isabella sam giselle : ℕ) : 
  isabella = sam + 45 →  -- Isabella has $45 more than Sam
  giselle = 120 →  -- Giselle has $120
  isabella + sam + giselle = 3 * 115 →  -- Total money shared equally among 3 shoppers
  isabella - giselle = 15 :=  -- Isabella has $15 more than Giselle
by sorry

end NUMINAMATH_CALUDE_money_difference_l536_53656


namespace NUMINAMATH_CALUDE_equation_equivalence_l536_53605

theorem equation_equivalence : ∃ (b c : ℝ), 
  (∀ x : ℝ, |x - 4| = 3 ↔ x = 1 ∨ x = 7) →
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 1 ∨ x = 7) →
  b = -8 ∧ c = 7 := by
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l536_53605


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l536_53641

-- Define the quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 + m*x + 1 = 0

-- Define the property of having two distinct real roots
def has_two_distinct_real_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation m x ∧ quadratic_equation m y

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  m < -2 ∨ m > 2

-- State the theorem
theorem quadratic_roots_range :
  ∀ m : ℝ, has_two_distinct_real_roots m ↔ m_range m :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l536_53641


namespace NUMINAMATH_CALUDE_flower_path_distance_l536_53651

/-- Given eight equally spaced flowers along a straight path, 
    where the distance between the first and fifth flower is 80 meters, 
    prove that the distance between the first and last flower is 140 meters. -/
theorem flower_path_distance :
  ∀ (flower_positions : ℕ → ℝ),
    (∀ i j : ℕ, i < j → flower_positions j - flower_positions i = (j - i : ℝ) * (flower_positions 1 - flower_positions 0)) →
    (flower_positions 4 - flower_positions 0 = 80) →
    (flower_positions 7 - flower_positions 0 = 140) :=
by sorry

end NUMINAMATH_CALUDE_flower_path_distance_l536_53651


namespace NUMINAMATH_CALUDE_origin_constructible_l536_53626

-- Define the points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the condition that A is above and to the left of B
def A_above_left_of_B : Prop :=
  A.1 < B.1 ∧ A.2 > B.2

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Theorem stating that the origin can be constructed
theorem origin_constructible (h : A_above_left_of_B) :
  ∃ (construction : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ)), construction A B = O :=
sorry

end NUMINAMATH_CALUDE_origin_constructible_l536_53626


namespace NUMINAMATH_CALUDE_bubble_pass_probability_specific_l536_53698

def bubble_pass_probability (n : ℕ) (initial_pos : ℕ) (final_pos : ℕ) : ℚ :=
  if initial_pos < final_pos ∧ final_pos ≤ n then
    1 / (initial_pos * (final_pos - 1))
  else
    0

theorem bubble_pass_probability_specific :
  bubble_pass_probability 50 25 35 = 1 / 850 := by
  sorry

end NUMINAMATH_CALUDE_bubble_pass_probability_specific_l536_53698


namespace NUMINAMATH_CALUDE_polygon_sides_l536_53652

/-- Theorem: For a polygon with n sides, if the sum of its interior angles is 180° less than three times the sum of its exterior angles, then n = 7. -/
theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 - 180 → n = 7 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l536_53652


namespace NUMINAMATH_CALUDE_exist_four_distinct_naturals_perfect_squares_l536_53612

theorem exist_four_distinct_naturals_perfect_squares :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∃ (m n : ℕ), a^2 + 2*c*d + b^2 = m^2 ∧ c^2 + 2*a*b + d^2 = n^2 :=
by
  sorry

end NUMINAMATH_CALUDE_exist_four_distinct_naturals_perfect_squares_l536_53612


namespace NUMINAMATH_CALUDE_complex_polygon_area_theorem_l536_53649

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side_length : ℝ)

/-- Represents the configuration of three overlapping sheets -/
structure SheetConfiguration :=
  (bottom : Sheet)
  (middle : Sheet)
  (top : Sheet)
  (middle_rotation : ℝ)
  (top_rotation : ℝ)
  (top_shift : ℝ)

/-- Calculates the area of the complex polygon formed by overlapping sheets -/
noncomputable def complex_polygon_area (config : SheetConfiguration) : ℝ :=
  sorry

/-- The main theorem stating the area of the complex polygon -/
theorem complex_polygon_area_theorem (config : SheetConfiguration) :
  config.bottom.side_length = 8 ∧
  config.middle.side_length = 8 ∧
  config.top.side_length = 8 ∧
  config.middle_rotation = 45 ∧
  config.top_rotation = 90 ∧
  config.top_shift = 4 →
  complex_polygon_area config = 144 :=
by sorry

end NUMINAMATH_CALUDE_complex_polygon_area_theorem_l536_53649


namespace NUMINAMATH_CALUDE_megan_initial_albums_l536_53602

/-- The number of albums Megan initially put in her shopping cart -/
def initial_albums : ℕ := sorry

/-- The number of albums Megan removed from her cart -/
def removed_albums : ℕ := 2

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of songs Megan bought -/
def total_songs : ℕ := 42

/-- Theorem stating that Megan initially put 8 albums in her shopping cart -/
theorem megan_initial_albums :
  initial_albums = 8 :=
by sorry

end NUMINAMATH_CALUDE_megan_initial_albums_l536_53602


namespace NUMINAMATH_CALUDE_range_of_a_l536_53673

/-- A linear function y = mx + b where m = -3a + 1 and b = a -/
def linear_function (a : ℝ) (x : ℝ) : ℝ := (-3 * a + 1) * x + a

/-- Condition that the function is increasing -/
def is_increasing (a : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > x₂ → linear_function a x₁ > linear_function a x₂

/-- Condition that the graph does not pass through the fourth quadrant -/
def not_in_fourth_quadrant (a : ℝ) : Prop :=
  ∀ x y, linear_function a x = y → (x ≥ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≥ 0) ∨ (x ≤ 0 ∧ y ≤ 0)

/-- The main theorem stating the range of a -/
theorem range_of_a (a : ℝ) 
  (h1 : is_increasing a) 
  (h2 : not_in_fourth_quadrant a) : 
  0 ≤ a ∧ a < 1/3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l536_53673


namespace NUMINAMATH_CALUDE_problem_solution_l536_53663

theorem problem_solution (x y : ℝ) (h : x^2 + y^2 = 12*x - 4*y - 40) :
  x * Real.cos (-23/3 * Real.pi) + y * Real.tan (-15/4 * Real.pi) = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l536_53663


namespace NUMINAMATH_CALUDE_problem_solution_l536_53625

theorem problem_solution (x y z : ℚ) 
  (eq1 : 102 * x - 5 * y = 25)
  (eq2 : 3 * y - x = 10)
  (eq3 : z^2 = y - x) : 
  x = 125 / 301 ∧ 10 - x = 2885 / 301 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l536_53625


namespace NUMINAMATH_CALUDE_digit_150_is_5_l536_53604

/-- The decimal expansion of 7/29 -/
def decimal_expansion : List Nat := [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7]

/-- The length of the repeating block in the decimal expansion of 7/29 -/
def repeat_length : Nat := decimal_expansion.length

/-- The 150th digit after the decimal point in the decimal expansion of 7/29 -/
def digit_150 : Nat := decimal_expansion[(150 - 1) % repeat_length]

theorem digit_150_is_5 : digit_150 = 5 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_5_l536_53604


namespace NUMINAMATH_CALUDE_rainfall_difference_l536_53632

def monday_count : ℕ := 10
def tuesday_count : ℕ := 12
def wednesday_count : ℕ := 8
def thursday_count : ℕ := 6

def monday_rain : ℝ := 1.25
def tuesday_rain : ℝ := 2.15
def wednesday_rain : ℝ := 1.60
def thursday_rain : ℝ := 2.80

theorem rainfall_difference :
  (tuesday_count * tuesday_rain + thursday_count * thursday_rain) -
  (monday_count * monday_rain + wednesday_count * wednesday_rain) = 17.3 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_difference_l536_53632


namespace NUMINAMATH_CALUDE_mrs_blue_garden_yield_l536_53662

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected tomato yield from a garden -/
def expectedTomatoYield (garden : GardenDimensions) (stepLength : ℚ) (yieldPerSqFt : ℚ) : ℚ :=
  (garden.length : ℚ) * stepLength * (garden.width : ℚ) * stepLength * yieldPerSqFt

/-- Theorem stating the expected tomato yield for Mrs. Blue's garden -/
theorem mrs_blue_garden_yield :
  let garden : GardenDimensions := { length := 18, width := 24 }
  let stepLength : ℚ := 3/2
  let yieldPerSqFt : ℚ := 2/3
  expectedTomatoYield garden stepLength yieldPerSqFt = 648 := by
  sorry

end NUMINAMATH_CALUDE_mrs_blue_garden_yield_l536_53662


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l536_53681

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  second_quadrant (-3) 2 := by sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l536_53681


namespace NUMINAMATH_CALUDE_factor_expression_l536_53622

theorem factor_expression (b : ℝ) : 45 * b^2 + 135 * b^3 = 45 * b^2 * (1 + 3 * b) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l536_53622


namespace NUMINAMATH_CALUDE_marble_count_l536_53650

theorem marble_count (total : ℕ) (blue : ℕ) (prob_red_or_white : ℚ) 
  (h1 : total = 50)
  (h2 : blue = 5)
  (h3 : prob_red_or_white = 9/10) :
  total - blue = 45 := by
sorry

end NUMINAMATH_CALUDE_marble_count_l536_53650


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l536_53670

/-- An arithmetic sequence with first term 5 and sum of first 31 terms equal to 390 -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 5 ∧ 
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (Finset.sum (Finset.range 31) (λ i => a (i + 1)) = 390)

/-- The ratio of sum of odd-indexed terms to sum of even-indexed terms -/
def ratio (a : ℕ → ℚ) : ℚ :=
  (Finset.sum (Finset.filter (λ i => i % 2 = 1) (Finset.range 31)) (λ i => a (i + 1))) /
  (Finset.sum (Finset.filter (λ i => i % 2 = 0) (Finset.range 31)) (λ i => a (i + 1)))

theorem arithmetic_sequence_ratio (a : ℕ → ℚ) :
  arithmetic_sequence a → ratio a = 16 / 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l536_53670


namespace NUMINAMATH_CALUDE_tangerines_remaining_l536_53687

/-- The number of tangerines remaining in Yuna's house after Yoo-jung ate some. -/
def remaining_tangerines (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem stating that the number of remaining tangerines is 9. -/
theorem tangerines_remaining :
  remaining_tangerines 12 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_remaining_l536_53687


namespace NUMINAMATH_CALUDE_tennis_tournament_matches_l536_53609

theorem tennis_tournament_matches (total_players : ℕ) (bye_players : ℕ) (first_round_players : ℕ) :
  total_players = 128 →
  bye_players = 40 →
  first_round_players = 88 →
  (total_players = bye_players + first_round_players) →
  (∃ (total_matches : ℕ), total_matches = 127 ∧
    total_matches = (first_round_players / 2) + (total_players - 1)) := by
  sorry

end NUMINAMATH_CALUDE_tennis_tournament_matches_l536_53609


namespace NUMINAMATH_CALUDE_laran_sells_five_posters_l536_53657

/-- Represents the poster business model for Laran -/
structure PosterBusiness where
  large_posters_per_day : ℕ
  large_poster_price : ℕ
  large_poster_cost : ℕ
  small_poster_price : ℕ
  small_poster_cost : ℕ
  weekly_profit : ℕ
  school_days_per_week : ℕ

/-- Calculates the total number of posters sold per day -/
def total_posters_per_day (b : PosterBusiness) : ℕ :=
  b.large_posters_per_day + 
  ((b.weekly_profit / b.school_days_per_week - 
    (b.large_posters_per_day * (b.large_poster_price - b.large_poster_cost))) / 
   (b.small_poster_price - b.small_poster_cost))

/-- Theorem stating that Laran sells 5 posters per day -/
theorem laran_sells_five_posters (b : PosterBusiness) 
  (h1 : b.large_posters_per_day = 2)
  (h2 : b.large_poster_price = 10)
  (h3 : b.large_poster_cost = 5)
  (h4 : b.small_poster_price = 6)
  (h5 : b.small_poster_cost = 3)
  (h6 : b.weekly_profit = 95)
  (h7 : b.school_days_per_week = 5) :
  total_posters_per_day b = 5 := by
  sorry

end NUMINAMATH_CALUDE_laran_sells_five_posters_l536_53657


namespace NUMINAMATH_CALUDE_triangle_sine_squared_ratio_l536_53640

theorem triangle_sine_squared_ratio (a b c : ℝ) (A B C : Real) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  S > 0 →
  S = (1/2) * a * b * Real.sin C →
  (a^2 + b^2) * Real.tan C = 8 * S →
  (Real.sin A)^2 + (Real.sin B)^2 = 2 * (Real.sin C)^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_squared_ratio_l536_53640


namespace NUMINAMATH_CALUDE_inverse_and_negation_of_union_subset_inverse_of_divisibility_negation_and_contrapositive_of_inequality_inverse_of_quadratic_inequality_l536_53669

-- Define the sets A and B
variable (A B : Set α)

-- Define the divisibility relation
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

-- 1. Inverse and negation of "If x ∈ (A ∪ B), then x ∈ B"
theorem inverse_and_negation_of_union_subset (x : α) :
  (x ∈ B → x ∈ A ∪ B) ∧ (x ∉ A ∪ B → x ∉ B) := by sorry

-- 2. Inverse of "If a natural number is divisible by 6, then it is divisible by 2"
theorem inverse_of_divisibility :
  ¬(∀ n : ℕ, divides 2 n → divides 6 n) := by sorry

-- 3. Negation and contrapositive of "If 0 < x < 5, then |x-2| < 3"
theorem negation_and_contrapositive_of_inequality (x : ℝ) :
  ¬(¬(0 < x ∧ x < 5) → |x - 2| ≥ 3) ∧
  (|x - 2| ≥ 3 → ¬(0 < x ∧ x < 5)) := by sorry

-- 4. Inverse of "If (a-2)x^2 + 2(a-2)x - 4 < 0 holds for all x ∈ ℝ, then a ∈ (-2, 2)"
theorem inverse_of_quadratic_inequality (a : ℝ) :
  a ∈ Set.Ioo (-2) 2 →
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) := by sorry

end NUMINAMATH_CALUDE_inverse_and_negation_of_union_subset_inverse_of_divisibility_negation_and_contrapositive_of_inequality_inverse_of_quadratic_inequality_l536_53669


namespace NUMINAMATH_CALUDE_percentage_of_female_employees_l536_53644

theorem percentage_of_female_employees (total_employees : ℕ) 
  (computer_literate_percentage : ℚ) (female_computer_literate : ℕ) 
  (male_computer_literate_percentage : ℚ) :
  total_employees = 1100 →
  computer_literate_percentage = 62 / 100 →
  female_computer_literate = 462 →
  male_computer_literate_percentage = 1 / 2 →
  (↑female_computer_literate + (male_computer_literate_percentage * ↑(total_employees - female_computer_literate / computer_literate_percentage))) / ↑total_employees = 3 / 5 := by
  sorry

#check percentage_of_female_employees

end NUMINAMATH_CALUDE_percentage_of_female_employees_l536_53644


namespace NUMINAMATH_CALUDE_hyperbola_range_of_b_squared_l536_53665

/-- Given a hyperbola M: x^2 - y^2/b^2 = 1 (b > 0) with foci F1(-c, 0) and F2(c, 0),
    if a line parallel to one asymptote passes through F1 and intersects the other asymptote at P(-c/2, bc/2),
    and P is inside the circle x^2 + y^2 = 4b^2, then 7 - 4√3 < b^2 < 7 + 4√3 -/
theorem hyperbola_range_of_b_squared (b c : ℝ) (hb : b > 0) (hc : c^2 = b^2 + 1) :
  let P : ℝ × ℝ := (-c/2, b*c/2)
  (P.1^2 + P.2^2 < 4*b^2) → (7 - 4*Real.sqrt 3 < b^2 ∧ b^2 < 7 + 4*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_range_of_b_squared_l536_53665


namespace NUMINAMATH_CALUDE_lights_at_top_point_l536_53614

/-- Represents the number of layers in the structure -/
def num_layers : ℕ := 7

/-- Represents the common ratio of the geometric sequence -/
def common_ratio : ℕ := 2

/-- Represents the total number of lights -/
def total_lights : ℕ := 381

/-- Theorem stating that the number of lights at the topmost point is 3 -/
theorem lights_at_top_point : 
  ∃ (a : ℕ), a * (common_ratio ^ num_layers - 1) / (common_ratio - 1) = total_lights ∧ a = 3 :=
sorry

end NUMINAMATH_CALUDE_lights_at_top_point_l536_53614


namespace NUMINAMATH_CALUDE_m_four_sufficient_not_necessary_l536_53619

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Define the two lines parameterized by m -/
def line1 (m : ℝ) : Line := ⟨2*m - 4, m + 1, 2⟩
def line2 (m : ℝ) : Line := ⟨m + 1, -m, 3⟩

/-- Main theorem -/
theorem m_four_sufficient_not_necessary :
  (∃ m : ℝ, m ≠ 4 ∧ are_perpendicular (line1 m) (line2 m)) ∧
  are_perpendicular (line1 4) (line2 4) := by
  sorry

end NUMINAMATH_CALUDE_m_four_sufficient_not_necessary_l536_53619


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l536_53675

theorem ellipse_eccentricity (k : ℝ) : 
  (∃ (x y : ℝ), x^2 / (k + 8) + y^2 / 9 = 1) →  -- Ellipse equation
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ (a^2 - b^2) / a^2 = 1/4) →  -- Eccentricity condition
  (k = 4 ∨ k = -5/4) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l536_53675


namespace NUMINAMATH_CALUDE_circle_radius_with_chords_l536_53684

/-- A circle with three parallel chords -/
structure CircleWithChords where
  -- The radius of the circle
  radius : ℝ
  -- The distance from the center to the closest chord
  x : ℝ
  -- The common distance between the chords
  y : ℝ
  -- Conditions on the chords
  chord_condition : radius^2 = x^2 + 100 ∧ 
                    radius^2 = (x + y)^2 + 64 ∧ 
                    radius^2 = (x + 2*y)^2 + 16

/-- The theorem stating that the radius of the circle with the given chord configuration is 5√22/2 -/
theorem circle_radius_with_chords (c : CircleWithChords) : c.radius = 5 * Real.sqrt 22 / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_with_chords_l536_53684


namespace NUMINAMATH_CALUDE_festival_attendance_l536_53672

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h1 : total_students = 1500)
  (h2 : festival_attendees = 900)
  (h3 : ∃ (girls boys : ℕ), 
    girls + boys = total_students ∧ 
    (2 * girls) / 3 + boys / 2 = festival_attendees) :
  ∃ (girls : ℕ), (2 * girls) / 3 = 600 := by
  sorry

end NUMINAMATH_CALUDE_festival_attendance_l536_53672


namespace NUMINAMATH_CALUDE_peaches_per_basket_proof_l536_53637

/-- The number of peaches in each basket originally -/
def peaches_per_basket : ℕ := 25

/-- The number of baskets -/
def num_baskets : ℕ := 5

/-- The number of peaches eaten by farmers -/
def eaten_peaches : ℕ := 5

/-- The number of peaches in each small box after packing -/
def peaches_per_box : ℕ := 15

/-- The number of small boxes after packing -/
def num_boxes : ℕ := 8

theorem peaches_per_basket_proof :
  peaches_per_basket * num_baskets = 
    num_boxes * peaches_per_box + eaten_peaches :=
by sorry

end NUMINAMATH_CALUDE_peaches_per_basket_proof_l536_53637
