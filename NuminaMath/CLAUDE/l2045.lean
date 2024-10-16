import Mathlib

namespace NUMINAMATH_CALUDE_somu_present_age_l2045_204568

/-- Somu's present age -/
def somu_age : ℕ := sorry

/-- Somu's father's present age -/
def father_age : ℕ := sorry

/-- Somu's age is one-third of his father's age -/
axiom current_age_ratio : somu_age = father_age / 3

/-- 9 years ago, Somu was one-fifth of his father's age -/
axiom past_age_ratio : somu_age - 9 = (father_age - 9) / 5

theorem somu_present_age : somu_age = 18 := by sorry

end NUMINAMATH_CALUDE_somu_present_age_l2045_204568


namespace NUMINAMATH_CALUDE_sum_of_valid_m_l2045_204525

def inequality_system (x m : ℤ) : Prop :=
  (x - 2) / 4 < (x - 1) / 3 ∧ 3 * x - m ≤ 3 - x

def equation_system (x y m : ℤ) : Prop :=
  m * x + y = 4 ∧ 3 * x - y = 0

theorem sum_of_valid_m :
  (∃ (s : Finset ℤ), 
    (∀ m ∈ s, 
      (∃! (a b : ℤ), inequality_system a m) ∧
      (∃ (x y : ℤ), equation_system x y m)) ∧
    (s.sum id = -3)) :=
sorry

end NUMINAMATH_CALUDE_sum_of_valid_m_l2045_204525


namespace NUMINAMATH_CALUDE_only_negative_number_l2045_204589

theorem only_negative_number (a b c d : ℝ) : 
  a = |(-2)| ∧ b = Real.sqrt 3 ∧ c = 0 ∧ d = -5 →
  (d < 0 ∧ a ≥ 0 ∧ b > 0 ∧ c = 0) := by sorry

end NUMINAMATH_CALUDE_only_negative_number_l2045_204589


namespace NUMINAMATH_CALUDE_derivative_at_one_l2045_204554

theorem derivative_at_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f x = 2 * x * f' 1 + x^2) :
  f' 1 = -2 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l2045_204554


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2045_204517

theorem arithmetic_mean_problem (x : ℝ) : 
  (12 + 18 + 24 + 36 + 6 + x) / 6 = 16 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2045_204517


namespace NUMINAMATH_CALUDE_common_factor_proof_l2045_204578

theorem common_factor_proof (a b : ℕ) : 
  (4 * a^2 * b^3).gcd (6 * a^3 * b) = 2 * a^2 * b :=
by sorry

end NUMINAMATH_CALUDE_common_factor_proof_l2045_204578


namespace NUMINAMATH_CALUDE_ashley_champagne_bottles_l2045_204511

/-- The number of bottles of champagne needed for a wedding toast -/
def bottles_needed (glasses_per_guest : ℕ) (num_guests : ℕ) (servings_per_bottle : ℕ) : ℕ :=
  (glasses_per_guest * num_guests + servings_per_bottle - 1) / servings_per_bottle

/-- Proof that Ashley needs 40 bottles of champagne for her wedding toast -/
theorem ashley_champagne_bottles :
  bottles_needed 2 120 6 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ashley_champagne_bottles_l2045_204511


namespace NUMINAMATH_CALUDE_total_seashells_l2045_204557

def mary_seashells : ℕ := 18
def jessica_seashells : ℕ := 41

theorem total_seashells : mary_seashells + jessica_seashells = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l2045_204557


namespace NUMINAMATH_CALUDE_bijective_function_exists_l2045_204534

/-- A function that maps elements of ℤm × ℤn to itself -/
def bijective_function (m n : ℕ+) : (Fin m × Fin n) → (Fin m × Fin n) := sorry

/-- Predicate to check if all f(v) + v are pairwise distinct -/
def all_distinct (m n : ℕ+) (f : (Fin m × Fin n) → (Fin m × Fin n)) : Prop := sorry

/-- Main theorem statement -/
theorem bijective_function_exists (m n : ℕ+) :
  (∃ f : (Fin m × Fin n) → (Fin m × Fin n), Function.Bijective f ∧ all_distinct m n f) ↔
  (m.val % 2 = n.val % 2) := by sorry

end NUMINAMATH_CALUDE_bijective_function_exists_l2045_204534


namespace NUMINAMATH_CALUDE_angle_from_point_l2045_204538

theorem angle_from_point (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) :
  (∃ (P : ℝ × ℝ), P.1 = Real.sin (3 * Real.pi / 4) ∧ 
                   P.2 = Real.cos (3 * Real.pi / 4) ∧ 
                   P.1 = Real.sin θ ∧ 
                   P.2 = Real.cos θ) →
  θ = 7 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_from_point_l2045_204538


namespace NUMINAMATH_CALUDE_abc_encodes_to_57_l2045_204502

/-- Represents the set of characters used in the encoding -/
inductive EncodingChar : Type
  | A | B | C | D

/-- Represents a base 4 number as a list of EncodingChar -/
def Base4Number := List EncodingChar

/-- Converts a Base4Number to its decimal (base 10) representation -/
def toDecimal (n : Base4Number) : ℕ :=
  sorry

/-- Checks if three Base4Numbers are consecutive encodings -/
def areConsecutiveEncodings (a b c : Base4Number) : Prop :=
  sorry

/-- Main theorem: Given the conditions, ABC encodes to 57 in base 10 -/
theorem abc_encodes_to_57 
  (h : areConsecutiveEncodings 
    [EncodingChar.B, EncodingChar.C, EncodingChar.D]
    [EncodingChar.B, EncodingChar.C, EncodingChar.C]
    [EncodingChar.B, EncodingChar.D, EncodingChar.A]) :
  toDecimal [EncodingChar.A, EncodingChar.B, EncodingChar.C] = 57 := by
  sorry

end NUMINAMATH_CALUDE_abc_encodes_to_57_l2045_204502


namespace NUMINAMATH_CALUDE_christmas_cards_count_l2045_204583

/-- The number of Christmas cards John sent -/
def christmas_cards : ℕ := 20

/-- The number of birthday cards John sent -/
def birthday_cards : ℕ := 15

/-- The cost of each card in dollars -/
def cost_per_card : ℕ := 2

/-- The total amount John spent on cards in dollars -/
def total_spent : ℕ := 70

/-- Theorem stating that the number of Christmas cards is 20 -/
theorem christmas_cards_count :
  christmas_cards = 20 ∧
  birthday_cards = 15 ∧
  cost_per_card = 2 ∧
  total_spent = 70 →
  christmas_cards * cost_per_card + birthday_cards * cost_per_card = total_spent :=
by sorry

end NUMINAMATH_CALUDE_christmas_cards_count_l2045_204583


namespace NUMINAMATH_CALUDE_six_sufficient_not_necessary_l2045_204516

-- Define the binomial expansion term
def binomialTerm (n : ℕ) (r : ℕ) : ℚ → ℚ := λ x => x^(2*n - 3*r)

-- Define the condition for a constant term
def hasConstantTerm (n : ℕ) : Prop := ∃ r : ℕ, 2*n = 3*r

-- Theorem stating that n=6 is sufficient but not necessary
theorem six_sufficient_not_necessary :
  (hasConstantTerm 6) ∧ (∃ m : ℕ, m ≠ 6 ∧ hasConstantTerm m) :=
sorry

end NUMINAMATH_CALUDE_six_sufficient_not_necessary_l2045_204516


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2045_204566

/-- Given two circles R and S, if the diameter of R is 80% of the diameter of S,
    then the area of R is 64% of the area of S. -/
theorem circle_area_ratio (R S : Real) (hdiameter : R = 0.8 * S) :
  (π * (R / 2)^2) / (π * (S / 2)^2) = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2045_204566


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l2045_204509

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (m : ℝ) : Set ℝ := {x | x^2 - m*x < 0}

-- State the theorem
theorem intersection_implies_m_equals_one :
  ∀ m : ℝ, (M ∩ N m) = {x | 0 < x ∧ x < 1} → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_one_l2045_204509


namespace NUMINAMATH_CALUDE_nail_decoration_theorem_l2045_204579

/-- The time it takes to decorate nails with three coats -/
def nail_decoration_time (application_time dry_time number_of_coats : ℕ) : ℕ :=
  (application_time + dry_time) * number_of_coats

/-- Theorem: The total time to apply and dry three coats on nails is 120 minutes -/
theorem nail_decoration_theorem :
  nail_decoration_time 20 20 3 = 120 :=
by sorry

end NUMINAMATH_CALUDE_nail_decoration_theorem_l2045_204579


namespace NUMINAMATH_CALUDE_custom_op_neg_four_six_l2045_204585

-- Define the custom operation ﹡
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b - 1

-- Theorem statement
theorem custom_op_neg_four_six :
  custom_op (-4) 6 = -9 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_neg_four_six_l2045_204585


namespace NUMINAMATH_CALUDE_bug_return_probability_l2045_204552

/-- Probability of the bug being at the starting corner after n moves -/
def Q : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - Q n)

/-- The probability of the bug returning to its starting corner on the eighth move -/
theorem bug_return_probability : Q 8 = 547 / 2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l2045_204552


namespace NUMINAMATH_CALUDE_abs_equation_unique_solution_l2045_204543

theorem abs_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x - 3| :=
by
  sorry

end NUMINAMATH_CALUDE_abs_equation_unique_solution_l2045_204543


namespace NUMINAMATH_CALUDE_modulus_of_complex_l2045_204575

theorem modulus_of_complex (m : ℝ) : 
  let z : ℂ := Complex.mk (m - 2) (m + 1)
  Complex.abs z = Real.sqrt (2 * m^2 - 2 * m + 5) := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l2045_204575


namespace NUMINAMATH_CALUDE_rotten_bananas_percentage_l2045_204510

theorem rotten_bananas_percentage
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (rotten_oranges_percent : ℚ)
  (good_fruits_percent : ℚ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : rotten_oranges_percent = 15 / 100)
  (h4 : good_fruits_percent = 89 / 100)
  : (1 : ℚ) - (good_fruits_percent * (total_oranges + total_bananas : ℚ) - (1 - rotten_oranges_percent) * total_oranges) / total_bananas = 5 / 100 := by
  sorry

end NUMINAMATH_CALUDE_rotten_bananas_percentage_l2045_204510


namespace NUMINAMATH_CALUDE_f_symmetric_property_l2045_204574

/-- Given a function f(x) = ax^4 + bx^2 + 2x - 8 where a and b are real constants,
    if f(-1) = 10, then f(1) = -26 -/
theorem f_symmetric_property (a b : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^4 + b * x^2 + 2 * x - 8
  f (-1) = 10 → f 1 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_symmetric_property_l2045_204574


namespace NUMINAMATH_CALUDE_plastic_bag_estimate_l2045_204536

def plastic_bag_data : List Nat := [33, 25, 28, 26, 25, 31]
def total_students : Nat := 45

theorem plastic_bag_estimate :
  let average := (plastic_bag_data.sum / plastic_bag_data.length)
  average * total_students = 1260 := by
  sorry

end NUMINAMATH_CALUDE_plastic_bag_estimate_l2045_204536


namespace NUMINAMATH_CALUDE_more_triangles_2003_l2045_204567

/-- A triangle with integer sides -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of triangles with integer sides and perimeter 2000 -/
def Triangles2000 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 2000}

/-- The set of triangles with integer sides and perimeter 2003 -/
def Triangles2003 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 2003}

/-- Function that maps a triangle with perimeter 2000 to a triangle with perimeter 2003 -/
def f (t : IntTriangle) : IntTriangle :=
  ⟨t.a + 1, t.b + 1, t.c + 1, sorry⟩

theorem more_triangles_2003 :
  ∃ (g : Triangles2000 → Triangles2003), Function.Injective g ∧
  ∃ (t : Triangles2003), t ∉ Set.range g :=
sorry

end NUMINAMATH_CALUDE_more_triangles_2003_l2045_204567


namespace NUMINAMATH_CALUDE_range_of_m_for_sufficient_not_necessary_condition_l2045_204503

/-- The range of m for which ¬p is a sufficient but not necessary condition for ¬q -/
theorem range_of_m_for_sufficient_not_necessary_condition 
  (p : ℝ → Prop) (q : ℝ → ℝ → Prop) (m : ℝ) : 
  (∀ x, p x ↔ x^2 - 8*x - 20 ≤ 0) →
  (∀ x, q x m ↔ x^2 - 2*x + 1 - m^2 ≤ 0) →
  m > 0 →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  0 < m ∧ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_for_sufficient_not_necessary_condition_l2045_204503


namespace NUMINAMATH_CALUDE_S_equals_T_l2045_204588

def S : Set ℤ := {x | ∃ n : ℤ, x = 2*n + 1}
def T : Set ℤ := {x | ∃ n : ℤ, x = 4*n + 1 ∨ x = 4*n - 1}

theorem S_equals_T : S = T := by sorry

end NUMINAMATH_CALUDE_S_equals_T_l2045_204588


namespace NUMINAMATH_CALUDE_line_intersects_segment_l2045_204531

/-- A line defined by the equation 2x + y - b = 0 intersects the line segment
    between points (1,0) and (-1,0) if and only if -2 ≤ b ≤ 2. -/
theorem line_intersects_segment (b : ℝ) :
  (∃ (x y : ℝ), 2*x + y - b = 0 ∧ 
    ((x = 1 ∧ y = 0) ∨ 
     (x = -1 ∧ y = 0) ∨ 
     (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ x = -1 + 2*t ∧ y = 0)))
  ↔ -2 ≤ b ∧ b ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_segment_l2045_204531


namespace NUMINAMATH_CALUDE_brass_players_count_l2045_204541

/-- Represents the composition of a marching band -/
structure MarchingBand where
  brass : ℕ
  woodwind : ℕ
  percussion : ℕ

/-- Checks if the given marching band composition is valid -/
def isValidBand (band : MarchingBand) : Prop :=
  band.woodwind = 2 * band.brass ∧
  band.percussion = 4 * band.woodwind ∧
  band.brass + band.woodwind + band.percussion = 110

theorem brass_players_count (band : MarchingBand) (h : isValidBand band) : band.brass = 10 := by
  sorry

#check brass_players_count

end NUMINAMATH_CALUDE_brass_players_count_l2045_204541


namespace NUMINAMATH_CALUDE_water_left_l2045_204521

theorem water_left (initial : ℚ) (used : ℚ) (left : ℚ) : 
  initial = 3 → used = 9/4 → left = initial - used → left = 3/4 := by sorry

end NUMINAMATH_CALUDE_water_left_l2045_204521


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l2045_204594

theorem tangent_line_to_circle (a : ℝ) : 
  (∀ x y : ℝ, ax + y + 1 = 0 → (x - 2)^2 + y^2 = 4) →
  (∃! x y : ℝ, ax + y + 1 = 0 ∧ (x - 2)^2 + y^2 = 4) →
  a = 3/4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l2045_204594


namespace NUMINAMATH_CALUDE_calculation_proof_l2045_204504

theorem calculation_proof : 5 * (-2) + Real.pi ^ 0 + (-1) ^ 2023 - 2 ^ 3 = -18 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2045_204504


namespace NUMINAMATH_CALUDE_celia_running_time_l2045_204550

/-- Given that Celia runs twice as fast as Lexie, and Lexie takes 20 minutes to run a mile,
    prove that Celia will take 300 minutes to run 30 miles. -/
theorem celia_running_time :
  ∀ (lexie_speed celia_speed : ℝ),
  celia_speed = 2 * lexie_speed →
  lexie_speed * 20 = 1 →
  celia_speed * 300 = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_celia_running_time_l2045_204550


namespace NUMINAMATH_CALUDE_sum_of_factors_l2045_204573

theorem sum_of_factors (a b c d e : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  (5 - a) * (5 - b) * (5 - c) * (5 - d) * (5 - e) = 120 →
  a + b + c + d + e = 13 := by
sorry

end NUMINAMATH_CALUDE_sum_of_factors_l2045_204573


namespace NUMINAMATH_CALUDE_line_segment_point_sum_l2045_204542

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := y = -5/6 * x + 10

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (12, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 10)

/-- Point T is on the line segment PQ -/
def T : ℝ × ℝ → Prop
  | (r, s) => ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- Area of triangle POQ -/
def area_POQ : ℝ := 60

/-- Area of triangle TOP -/
def area_TOP : ℝ := 15

/-- Theorem: If the given conditions are met, then r + s = 11.5 -/
theorem line_segment_point_sum (r s : ℝ) : 
  line_eq r s → T (r, s) → area_POQ = 4 * area_TOP → r + s = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_point_sum_l2045_204542


namespace NUMINAMATH_CALUDE_largest_of_three_consecutive_odd_integers_l2045_204563

theorem largest_of_three_consecutive_odd_integers (a b c : ℤ) : 
  (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1) →  -- a, b, c are odd
  (b = a + 2 ∧ c = b + 2) →              -- a, b, c are consecutive
  (a + b + c = -147) →                   -- sum is -147
  (max a (max b c) = -47) :=             -- largest is -47
by sorry

end NUMINAMATH_CALUDE_largest_of_three_consecutive_odd_integers_l2045_204563


namespace NUMINAMATH_CALUDE_ratio_sum_problem_l2045_204547

theorem ratio_sum_problem (a b c : ℕ) : 
  a + b + c = 1000 → 
  5 * b = a → 
  4 * b = c → 
  c = 400 := by sorry

end NUMINAMATH_CALUDE_ratio_sum_problem_l2045_204547


namespace NUMINAMATH_CALUDE_compute_expression_l2045_204549

theorem compute_expression : (85 * 1515 - 25 * 1515) + (48 * 1515) = 163620 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_l2045_204549


namespace NUMINAMATH_CALUDE_x_varies_as_three_fifths_of_z_l2045_204598

-- Define the relationships between x, y, and z
def varies_as_cube (x y : ℝ) : Prop := ∃ k : ℝ, x = k * y^3

def varies_as_fifth_root (y z : ℝ) : Prop := ∃ j : ℝ, y = j * z^(1/5)

def varies_as_power (x z : ℝ) (n : ℝ) : Prop := ∃ m : ℝ, x = m * z^n

-- State the theorem
theorem x_varies_as_three_fifths_of_z (x y z : ℝ) :
  varies_as_cube x y → varies_as_fifth_root y z → varies_as_power x z (3/5) :=
by sorry

end NUMINAMATH_CALUDE_x_varies_as_three_fifths_of_z_l2045_204598


namespace NUMINAMATH_CALUDE_angle_terminal_side_point_angle_terminal_side_point_with_sin_l2045_204580

-- Part 1
theorem angle_terminal_side_point (α : Real) :
  ∃ (P : ℝ × ℝ), P.1 = 4 ∧ P.2 = -3 →
  2 * Real.sin α + Real.cos α = -2/5 := by sorry

-- Part 2
theorem angle_terminal_side_point_with_sin (α : Real) (m : Real) :
  m ≠ 0 →
  ∃ (P : ℝ × ℝ), P.1 = -Real.sqrt 3 ∧ P.2 = m →
  Real.sin α = (Real.sqrt 2 * m) / 4 →
  (m = Real.sqrt 5 ∨ m = -Real.sqrt 5) ∧
  Real.cos α = -Real.sqrt 6 / 4 ∧
  (m > 0 → Real.tan α = -Real.sqrt 15 / 3) ∧
  (m < 0 → Real.tan α = Real.sqrt 15 / 3) := by sorry

end NUMINAMATH_CALUDE_angle_terminal_side_point_angle_terminal_side_point_with_sin_l2045_204580


namespace NUMINAMATH_CALUDE_base_three_digit_difference_l2045_204506

/-- The number of digits in the base-b representation of a positive integer n -/
def numDigits (n : ℕ+) (b : ℕ) : ℕ :=
  Nat.log b n + 1

/-- Theorem: The number of digits in the base-3 representation of 1500
    is exactly 1 more than the number of digits in the base-3 representation of 300 -/
theorem base_three_digit_difference :
  numDigits 1500 3 = numDigits 300 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_base_three_digit_difference_l2045_204506


namespace NUMINAMATH_CALUDE_cube_lateral_surface_area_l2045_204562

/-- The lateral surface area of a cube with side length 12 meters is 576 square meters. -/
theorem cube_lateral_surface_area : 
  let side_length : ℝ := 12
  let lateral_surface_area := 4 * side_length * side_length
  lateral_surface_area = 576 := by
sorry

end NUMINAMATH_CALUDE_cube_lateral_surface_area_l2045_204562


namespace NUMINAMATH_CALUDE_range_of_a_l2045_204513

def prop_p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def prop_q (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ 1 → y ≥ 1 → x ≤ y → (4 * x^2 - a * x) ≤ (4 * y^2 - a * y)

theorem range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) → ¬(prop_p a) → (a ≤ 0 ∨ (4 ≤ a ∧ a ≤ 8)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2045_204513


namespace NUMINAMATH_CALUDE_infimum_attained_by_uniform_distribution_l2045_204505

-- Define the set of Borel functions
def BorelFunction (f : ℝ → ℝ) : Prop := sorry

-- Define the property of being an increasing function
def Increasing (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y

-- Define a random variable
def RandomVariable (X : ℝ → ℝ) : Prop := sorry

-- Define the property of density not exceeding 1/2
def DensityNotExceedingHalf (X : ℝ → ℝ) : Prop := sorry

-- Define uniform distribution on [-1, 1]
def UniformDistributionOnUnitInterval (U : ℝ → ℝ) : Prop := sorry

-- Define expected value
def ExpectedValue (f : ℝ → ℝ) (X : ℝ → ℝ) : ℝ := sorry

-- Theorem statement
theorem infimum_attained_by_uniform_distribution
  (f : ℝ → ℝ) (X U : ℝ → ℝ) :
  BorelFunction f →
  Increasing f →
  RandomVariable X →
  DensityNotExceedingHalf X →
  UniformDistributionOnUnitInterval U →
  ExpectedValue (fun x => f (abs x)) X ≥ ExpectedValue (fun x => f (abs x)) U :=
sorry

end NUMINAMATH_CALUDE_infimum_attained_by_uniform_distribution_l2045_204505


namespace NUMINAMATH_CALUDE_decimal_rep_17_70_digit_150_of_17_70_l2045_204558

/-- The decimal representation of 17/70 has a repeating cycle of 6 digits -/
def decimal_cycle (n : ℕ) : ℕ := n % 6

/-- The digits in the repeating cycle of 17/70 -/
def cycle_digits : Fin 6 → ℕ
| 0 => 2
| 1 => 4
| 2 => 2
| 3 => 8
| 4 => 5
| 5 => 7

theorem decimal_rep_17_70 (n : ℕ) : 
  n > 0 → cycle_digits (decimal_cycle n) = 7 → n % 6 = 0 := by sorry

/-- The 150th digit after the decimal point in the decimal representation of 17/70 is 7 -/
theorem digit_150_of_17_70 : cycle_digits (decimal_cycle 150) = 7 := by sorry

end NUMINAMATH_CALUDE_decimal_rep_17_70_digit_150_of_17_70_l2045_204558


namespace NUMINAMATH_CALUDE_line_circle_intersection_l2045_204570

theorem line_circle_intersection (k : ℝ) : ∃ (x y : ℝ),
  y = k * x - k ∧ (x - 2)^2 + y^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l2045_204570


namespace NUMINAMATH_CALUDE_right_triangle_conditions_l2045_204530

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles A, B, C in radians

-- Define the conditions
def condition1 (t : Triangle) : Prop := t.A + t.C = t.B
def condition2 (t : Triangle) : Prop := ∃ (k : ℝ), t.A = k ∧ t.B = 2*k ∧ t.C = 3*k
def condition3 (t : Triangle) : Prop := ∃ (AB BC AC : ℝ), 3*AB = 4*BC ∧ 4*BC = 5*AC
def condition4 (t : Triangle) : Prop := t.A = t.B ∧ t.B = t.C

-- Define a right triangle
def is_right_triangle (t : Triangle) : Prop := t.A = Real.pi/2 ∨ t.B = Real.pi/2 ∨ t.C = Real.pi/2

-- Theorem statement
theorem right_triangle_conditions (t : Triangle) :
  (condition1 t → is_right_triangle t) ∧
  (condition2 t → is_right_triangle t) ∧
  ¬(condition3 t → is_right_triangle t) ∧
  ¬(condition4 t → is_right_triangle t) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_conditions_l2045_204530


namespace NUMINAMATH_CALUDE_distance_after_third_turn_l2045_204528

/-- Represents the distance traveled after each turn in the tunnel -/
structure TunnelDistance where
  after_first : ℝ
  after_second : ℝ
  after_third : ℝ
  after_fourth : ℝ
  total : ℝ

/-- Theorem: Given the conditions of the car's journey through the tunnel,
    the distance traveled after the 3rd turn is 10 meters. -/
theorem distance_after_third_turn (d : TunnelDistance) 
    (h1 : d.after_first = 5)
    (h2 : d.after_second = 8)
    (h3 : d.after_fourth = 0)
    (h4 : d.total = 23) :
    d.after_third = 10 := by
  sorry

#check distance_after_third_turn

end NUMINAMATH_CALUDE_distance_after_third_turn_l2045_204528


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2045_204548

theorem consecutive_odd_integers_sum (a : ℤ) : 
  (∃ n : ℤ, a = n ∧ 
    (∀ i : Fin 5, Odd (a + 2 * i.val)) ∧
    (a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = -365)) → 
  (a + 8 = -69) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l2045_204548


namespace NUMINAMATH_CALUDE_square_root_fraction_equality_l2045_204501

theorem square_root_fraction_equality : 
  let x : ℝ := Real.sqrt (7 - 4 * Real.sqrt 3)
  (x^2 - 4*x + 5) / (x^2 - 4*x + 3) = 2 := by sorry

end NUMINAMATH_CALUDE_square_root_fraction_equality_l2045_204501


namespace NUMINAMATH_CALUDE_final_jasmine_concentration_l2045_204507

/-- Calculates the final jasmine concentration after adding pure jasmine and water to a solution -/
theorem final_jasmine_concentration
  (initial_volume : ℝ)
  (initial_concentration : ℝ)
  (added_jasmine : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 80)
  (h2 : initial_concentration = 0.1)
  (h3 : added_jasmine = 5)
  (h4 : added_water = 15) :
  let initial_jasmine := initial_volume * initial_concentration
  let final_jasmine := initial_jasmine + added_jasmine
  let final_volume := initial_volume + added_jasmine + added_water
  final_jasmine / final_volume = 0.13 := by
sorry


end NUMINAMATH_CALUDE_final_jasmine_concentration_l2045_204507


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2045_204546

theorem fraction_subtraction : ((5 / 2) / (7 / 12)) - 4 / 9 = 242 / 63 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2045_204546


namespace NUMINAMATH_CALUDE_real_roots_of_polynomial_l2045_204571

theorem real_roots_of_polynomial (x : ℝ) :
  x^4 + 2*x^3 - x - 2 = 0 ↔ x = -2 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_real_roots_of_polynomial_l2045_204571


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l2045_204524

/-- Proves that mixing 250 mL of 10% alcohol solution with 750 mL of 30% alcohol solution results in a 25% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 250
  let y_volume : ℝ := 750
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.25
  let total_volume : ℝ := x_volume + y_volume
  let total_alcohol : ℝ := x_volume * x_concentration + y_volume * y_concentration
  total_alcohol / total_volume = target_concentration := by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l2045_204524


namespace NUMINAMATH_CALUDE_M_union_N_equals_R_l2045_204564

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set N
def N : Set ℝ := {x | |x| < Real.sqrt 5}

-- Theorem statement
theorem M_union_N_equals_R : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_M_union_N_equals_R_l2045_204564


namespace NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l2045_204508

/-- Calculates the number of paid windows given the total number of windows needed -/
def paidWindows (total : ℕ) : ℕ :=
  total - (total / 3)

/-- Calculates the cost of windows before any flat discount -/
def windowCost (paid : ℕ) : ℕ :=
  paid * 150

/-- Applies the flat discount if the cost is over 1000 -/
def applyDiscount (cost : ℕ) : ℕ :=
  if cost > 1000 then cost - 200 else cost

theorem no_savings_on_joint_purchase (dave_windows doug_windows : ℕ) 
  (h_dave : dave_windows = 9) (h_doug : doug_windows = 10) :
  let dave_cost := applyDiscount (windowCost (paidWindows dave_windows))
  let doug_cost := applyDiscount (windowCost (paidWindows doug_windows))
  let separate_cost := dave_cost + doug_cost
  let joint_windows := dave_windows + doug_windows
  let joint_cost := applyDiscount (windowCost (paidWindows joint_windows))
  separate_cost = joint_cost := by
  sorry

end NUMINAMATH_CALUDE_no_savings_on_joint_purchase_l2045_204508


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2045_204582

/-- An arithmetic sequence {aₙ} with a₇ = 4 and a₁₉ = 2a₉ has the general term formula aₙ = (n+1)/2 -/
theorem arithmetic_sequence_formula (a : ℕ → ℚ) 
  (h_arithmetic : ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) 
  (h_a7 : a 7 = 4)
  (h_a19 : a 19 = 2 * a 9) :
  ∀ n : ℕ, a n = (n + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l2045_204582


namespace NUMINAMATH_CALUDE_certification_cost_is_3000_l2045_204597

/-- The cost of certification for a seeing-eye dog --/
def certification_cost (adoption_fee training_cost_per_week training_weeks insurance_coverage out_of_pocket_cost : ℚ) : ℚ :=
  ((out_of_pocket_cost - (adoption_fee + training_cost_per_week * training_weeks)) / (1 - insurance_coverage)) * 10

/-- Theorem: The certification cost is $3000 given the specified conditions --/
theorem certification_cost_is_3000 :
  certification_cost 150 250 12 0.9 3450 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_certification_cost_is_3000_l2045_204597


namespace NUMINAMATH_CALUDE_emma_square_calculation_l2045_204569

theorem emma_square_calculation : 37^2 = 38^2 - 75 := by
  sorry

end NUMINAMATH_CALUDE_emma_square_calculation_l2045_204569


namespace NUMINAMATH_CALUDE_jellybean_count_jellybean_problem_l2045_204518

theorem jellybean_count (normal_class_size : ℕ) (absent_children : ℕ) 
  (jellybeans_per_child : ℕ) (remaining_jellybeans : ℕ) : ℕ :=
  let present_children := normal_class_size - absent_children
  let eaten_jellybeans := present_children * jellybeans_per_child
  eaten_jellybeans + remaining_jellybeans

theorem jellybean_problem : 
  jellybean_count 24 2 3 34 = 100 := by
  sorry

end NUMINAMATH_CALUDE_jellybean_count_jellybean_problem_l2045_204518


namespace NUMINAMATH_CALUDE_smallest_gcd_of_20m_25n_l2045_204519

theorem smallest_gcd_of_20m_25n (m n : ℕ+) (h : Nat.gcd m.val n.val = 18) :
  ∃ (m₀ n₀ : ℕ+), Nat.gcd m₀.val n₀.val = 18 ∧
    Nat.gcd (20 * m₀.val) (25 * n₀.val) = 90 ∧
    ∀ (m' n' : ℕ+), Nat.gcd m'.val n'.val = 18 →
      Nat.gcd (20 * m'.val) (25 * n'.val) ≥ 90 := by
  sorry

end NUMINAMATH_CALUDE_smallest_gcd_of_20m_25n_l2045_204519


namespace NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l2045_204587

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ x : ℚ, x > 0 ∧ total = a * x + b * x + c * x ∧ min (a * x) (min (b * x) (c * x)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_smallest_part_of_proportional_division_l2045_204587


namespace NUMINAMATH_CALUDE_fraction_inequality_l2045_204590

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  c / a > c / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2045_204590


namespace NUMINAMATH_CALUDE_minutes_to_year_l2045_204556

/-- Proves that 525,600 minutes is equivalent to 365 days (1 year) --/
theorem minutes_to_year (minutes_per_hour : ℕ) (hours_per_day : ℕ) (days_per_year : ℕ) : 
  minutes_per_hour = 60 → hours_per_day = 24 → days_per_year = 365 →
  525600 / (minutes_per_hour * hours_per_day) = days_per_year := by
  sorry

end NUMINAMATH_CALUDE_minutes_to_year_l2045_204556


namespace NUMINAMATH_CALUDE_cubic_expression_value_l2045_204529

theorem cubic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2*a^2 + 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l2045_204529


namespace NUMINAMATH_CALUDE_graces_mother_age_l2045_204561

theorem graces_mother_age :
  ∀ (grace_age grandmother_age mother_age : ℕ),
    grace_age = 60 →
    grace_age = (3 * grandmother_age) / 8 →
    grandmother_age = 2 * mother_age →
    mother_age = 80 := by
  sorry

end NUMINAMATH_CALUDE_graces_mother_age_l2045_204561


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2045_204527

/-- The function f(x) = (1/2)^x + 1 passes through the point (0, 2) -/
theorem function_passes_through_point :
  let f : ℝ → ℝ := fun x ↦ (1/2)^x + 1
  f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2045_204527


namespace NUMINAMATH_CALUDE_accurate_reading_is_10_30_l2045_204593

/-- Represents a scale reading with a lower bound, upper bound, and increment -/
structure ScaleReading where
  lowerBound : ℝ
  upperBound : ℝ
  increment : ℝ

/-- Represents the position of an arrow on the scale -/
structure ArrowPosition where
  value : ℝ
  beforeMidpoint : Bool

/-- Given a scale reading and an arrow position, determines the most accurate reading -/
def mostAccurateReading (scale : ScaleReading) (arrow : ArrowPosition) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the most accurate reading is 10.30 -/
theorem accurate_reading_is_10_30 :
  let scale := ScaleReading.mk 10.2 10.4 0.05
  let arrow := ArrowPosition.mk 10.33 true
  mostAccurateReading scale arrow = 10.30 := by
  sorry

end NUMINAMATH_CALUDE_accurate_reading_is_10_30_l2045_204593


namespace NUMINAMATH_CALUDE_currency_conversion_area_conversion_l2045_204595

-- Define the currency units
def yuan : ℝ := 1
def jiao : ℝ := 0.1
def fen : ℝ := 0.01

-- Define the area units
def hectare : ℝ := 10000
def square_meter : ℝ := 1

-- Theorem for currency conversion
theorem currency_conversion :
  6.89 * yuan = 6 * yuan + 8 * jiao + 9 * fen := by sorry

-- Theorem for area conversion
theorem area_conversion :
  2 * hectare + 60 * square_meter = 20060 * square_meter := by sorry

end NUMINAMATH_CALUDE_currency_conversion_area_conversion_l2045_204595


namespace NUMINAMATH_CALUDE_inequality_proof_l2045_204533

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a - b*c)/(a + b*c) + (b - c*a)/(b + c*a) + (c - a*b)/(c + a*b) ≤ 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2045_204533


namespace NUMINAMATH_CALUDE_circplus_square_sum_diff_l2045_204535

/-- Custom operation ⊕ for real numbers -/
def circplus (a b : ℝ) : ℝ := (a + b)^2

/-- Theorem stating the equality for (x+y)^2 ⊕ (x-y)^2 -/
theorem circplus_square_sum_diff (x y : ℝ) : 
  circplus ((x + y)^2) ((x - y)^2) = 4 * (x^2 + y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_circplus_square_sum_diff_l2045_204535


namespace NUMINAMATH_CALUDE_equation_solution_l2045_204537

theorem equation_solution (x : ℚ) : 
  (30 * x^2 + 17 = 47 * x - 6) →
  (x = 3/5 ∨ x = 23/36) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2045_204537


namespace NUMINAMATH_CALUDE_distribution_percent_below_mean_plus_std_dev_l2045_204532

-- Define a symmetric distribution with mean m and standard deviation d
def SymmetricDistribution (μ : ℝ) (σ : ℝ) (F : ℝ → ℝ) : Prop :=
  ∀ x, F (μ + x) + F (μ - x) = 1

-- Define the condition that 36% of the distribution lies within one standard deviation of the mean
def WithinOneStdDev (μ : ℝ) (σ : ℝ) (F : ℝ → ℝ) : Prop :=
  F (μ + σ) - F (μ - σ) = 0.36

-- Theorem statement
theorem distribution_percent_below_mean_plus_std_dev
  (μ σ : ℝ) (F : ℝ → ℝ) 
  (h_symmetric : SymmetricDistribution μ σ F)
  (h_within_one_std_dev : WithinOneStdDev μ σ F) :
  F (μ + σ) = 0.68 := by
  sorry

end NUMINAMATH_CALUDE_distribution_percent_below_mean_plus_std_dev_l2045_204532


namespace NUMINAMATH_CALUDE_inequality_preservation_l2045_204545

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a + c > b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2045_204545


namespace NUMINAMATH_CALUDE_maisy_current_wage_l2045_204596

/-- Represents Maisy's job options and wage calculations --/
structure JobOptions where
  current_hours : ℕ
  new_hours : ℕ
  new_wage : ℚ
  new_bonus : ℚ
  wage_difference : ℚ

/-- Calculates the wage per hour at Maisy's current job --/
def calculate_current_wage (options : JobOptions) : ℚ :=
  ((options.new_hours * options.new_wage + options.new_bonus) - options.wage_difference) / options.current_hours

/-- Theorem stating that Maisy's current wage is $10 per hour --/
theorem maisy_current_wage (options : JobOptions) 
  (h1 : options.current_hours = 8)
  (h2 : options.new_hours = 4)
  (h3 : options.new_wage = 15)
  (h4 : options.new_bonus = 35)
  (h5 : options.wage_difference = 15) :
  calculate_current_wage options = 10 := by
  sorry

end NUMINAMATH_CALUDE_maisy_current_wage_l2045_204596


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2045_204576

theorem sqrt_equation_solution (n : ℝ) : Real.sqrt (8 + n) = 9 → n = 73 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2045_204576


namespace NUMINAMATH_CALUDE_tangent_line_value_l2045_204526

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a circle -/
def isOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop :=
  sorry -- Definition of tangency

/-- Check if two lines are perpendicular -/
def arePerpendicular (l1 l2 : Line) : Prop :=
  sorry -- Definition of perpendicularity

theorem tangent_line_value (c : Circle) (l1 l2 : Line) (a : ℝ) :
  c.center = (1, 0) →
  c.radius^2 = 5 →
  isOnCircle (2, 2) c →
  isTangent l1 c →
  l2.a = a ∧ l2.b = -1 ∧ l2.c = 1 →
  arePerpendicular l1 l2 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_value_l2045_204526


namespace NUMINAMATH_CALUDE_unique_divisible_by_seven_l2045_204581

def is_valid_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 110000 ∧ n % 100 = 1 ∧ (n / 100) % 10 ≠ 0

theorem unique_divisible_by_seven :
  ∃! n : ℕ, is_valid_number n ∧ n % 7 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_divisible_by_seven_l2045_204581


namespace NUMINAMATH_CALUDE_complex_magnitude_l2045_204539

theorem complex_magnitude (z w : ℂ) 
  (h1 : Complex.abs (2 * z - w) = 25)
  (h2 : Complex.abs (z + 2 * w) = 5)
  (h3 : Complex.abs (z + w) = 2) : 
  Complex.abs z = 9 := by sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2045_204539


namespace NUMINAMATH_CALUDE_smallest_portion_is_ten_l2045_204555

/-- Represents the distribution of bread loaves -/
structure BreadDistribution where
  a : ℕ  -- smallest portion (first term of arithmetic sequence)
  d : ℕ  -- common difference of arithmetic sequence

/-- The problem of distributing bread loaves -/
def breadProblem (bd : BreadDistribution) : Prop :=
  -- Total sum is 100
  (5 * bd.a + 10 * bd.d = 100) ∧
  -- Sum of larger three portions is 1/3 of sum of smaller two portions
  (3 * bd.a + 9 * bd.d = (2 * bd.a + bd.d) / 3)

/-- Theorem stating the smallest portion is 10 -/
theorem smallest_portion_is_ten :
  ∃ (bd : BreadDistribution), breadProblem bd ∧ bd.a = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_portion_is_ten_l2045_204555


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l2045_204544

/-- The jumping contest between a grasshopper and a frog -/
theorem grasshopper_jump_distance 
  (frog_jump : ℕ) 
  (total_jump : ℕ) 
  (h1 : frog_jump = 35)
  (h2 : total_jump = 66) :
  total_jump - frog_jump = 31 := by
sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l2045_204544


namespace NUMINAMATH_CALUDE_new_house_cost_l2045_204560

def first_house_cost : ℝ := 100000

def value_increase_percentage : ℝ := 0.25

def new_house_down_payment_percentage : ℝ := 0.25

theorem new_house_cost (old_house_value : ℝ) (new_house_cost : ℝ) : 
  old_house_value = first_house_cost * (1 + value_increase_percentage) ∧
  old_house_value = new_house_cost * new_house_down_payment_percentage →
  new_house_cost = 500000 := by
  sorry

end NUMINAMATH_CALUDE_new_house_cost_l2045_204560


namespace NUMINAMATH_CALUDE_joan_initial_books_l2045_204500

/-- The number of books Joan initially gathered -/
def initial_books : ℕ := sorry

/-- The number of additional books Joan found -/
def additional_books : ℕ := 26

/-- The total number of books Joan has now -/
def total_books : ℕ := 59

/-- Theorem stating that the initial number of books is 33 -/
theorem joan_initial_books : 
  initial_books = total_books - additional_books :=
by sorry

end NUMINAMATH_CALUDE_joan_initial_books_l2045_204500


namespace NUMINAMATH_CALUDE_cubic_function_property_l2045_204515

/-- A cubic function passing through the point (-3, -2) -/
structure CubicFunction where
  p : ℝ
  q : ℝ
  r : ℝ
  s : ℝ
  passes_through : p * (-3)^3 + q * (-3)^2 + r * (-3) + s = -2

/-- Theorem: For a cubic function g(x) = px^3 + qx^2 + rx + s passing through (-3, -2),
    the expression 12p - 6q + 3r - s equals 2 -/
theorem cubic_function_property (g : CubicFunction) : 
  12 * g.p - 6 * g.q + 3 * g.r - g.s = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2045_204515


namespace NUMINAMATH_CALUDE_triangle_count_l2045_204572

/-- The number of triangles formed by three mutually intersecting line segments
    in a configuration of n points on a circle, where n ≥ 6 and
    any three line segments do not intersect at a single point inside the circle. -/
def num_triangles (n : ℕ) : ℕ :=
  Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6

/-- Theorem stating the number of triangles formed under the given conditions. -/
theorem triangle_count (n : ℕ) (h : n ≥ 6) :
  num_triangles n = Nat.choose n 3 + 4 * Nat.choose n 4 + 5 * Nat.choose n 5 + Nat.choose n 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_l2045_204572


namespace NUMINAMATH_CALUDE_last_digit_of_large_power_l2045_204592

theorem last_digit_of_large_power : ∃ (n1 n2 n3 : ℕ), 
  n1 = 99^9 ∧ 
  n2 = 999^n1 ∧ 
  n3 = 9999^n2 ∧ 
  99999^n3 % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_large_power_l2045_204592


namespace NUMINAMATH_CALUDE_median_of_class_distribution_l2045_204599

/-- Represents the distribution of weekly reading times for students -/
structure ReadingTimeDistribution where
  six_hours : Nat
  seven_hours : Nat
  eight_hours : Nat
  nine_hours : Nat

/-- Calculates the median of a given reading time distribution -/
def median (d : ReadingTimeDistribution) : Real :=
  sorry

/-- The specific distribution of reading times for the 30 students -/
def class_distribution : ReadingTimeDistribution :=
  { six_hours := 7
  , seven_hours := 8
  , eight_hours := 5
  , nine_hours := 10 }

/-- The theorem stating that the median of the given distribution is 7.5 -/
theorem median_of_class_distribution :
  median class_distribution = 7.5 := by sorry

end NUMINAMATH_CALUDE_median_of_class_distribution_l2045_204599


namespace NUMINAMATH_CALUDE_zoo_ticket_cost_zoo_ticket_cost_example_l2045_204514

/-- Calculate the total cost of zoo tickets for a group with a discount --/
theorem zoo_ticket_cost (num_children num_adults num_seniors : ℕ)
                        (child_price adult_price senior_price : ℚ)
                        (discount_rate : ℚ) : ℚ :=
  let total_before_discount := num_children * child_price +
                               num_adults * adult_price +
                               num_seniors * senior_price
  let discount_amount := discount_rate * total_before_discount
  let total_after_discount := total_before_discount - discount_amount
  total_after_discount

/-- Prove that the total cost of zoo tickets for the given group is $227.80 --/
theorem zoo_ticket_cost_example : zoo_ticket_cost 6 10 4 10 16 12 (15/100) = 227.8 := by
  sorry

end NUMINAMATH_CALUDE_zoo_ticket_cost_zoo_ticket_cost_example_l2045_204514


namespace NUMINAMATH_CALUDE_simplify_polynomial_l2045_204553

theorem simplify_polynomial (x : ℝ) :
  2 - 4*x - 6*x^2 + 8 + 10*x - 12*x^2 - 14 + 16*x + 18*x^2 = 22*x - 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l2045_204553


namespace NUMINAMATH_CALUDE_range_of_m_l2045_204591

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = x * y)
  (h_inequality : ∃ x y, x > 0 ∧ y > 0 ∧ x + y = x * y ∧ x + 4 * y < m^2 + 8 * m) :
  m < -9 ∨ m > 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2045_204591


namespace NUMINAMATH_CALUDE_ratio_problem_l2045_204565

theorem ratio_problem (a b x m : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a / b = 4 / 5) 
  (h4 : x = a + 0.75 * a) 
  (h5 : m = b - 0.80 * b) : 
  m / x = 1 / 7 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2045_204565


namespace NUMINAMATH_CALUDE_trig_identity_l2045_204584

theorem trig_identity : 
  (Real.sin (160 * π / 180) + Real.sin (40 * π / 180)) * 
  (Real.sin (140 * π / 180) + Real.sin (20 * π / 180)) + 
  (Real.sin (50 * π / 180) - Real.sin (70 * π / 180)) * 
  (Real.sin (130 * π / 180) - Real.sin (110 * π / 180)) = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l2045_204584


namespace NUMINAMATH_CALUDE_couscous_per_dish_l2045_204520

theorem couscous_per_dish 
  (shipment1 : ℕ) 
  (shipment2 : ℕ) 
  (shipment3 : ℕ) 
  (num_dishes : ℕ) 
  (h1 : shipment1 = 7)
  (h2 : shipment2 = 13)
  (h3 : shipment3 = 45)
  (h4 : num_dishes = 13) :
  (shipment1 + shipment2 + shipment3) / num_dishes = 5 := by
  sorry

#check couscous_per_dish

end NUMINAMATH_CALUDE_couscous_per_dish_l2045_204520


namespace NUMINAMATH_CALUDE_simplify_expressions_l2045_204577

theorem simplify_expressions :
  ∀ (x a b : ℝ),
    (x^2 + (3*x - 5) - (4*x - 1) = x^2 - x - 4) ∧
    (7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2045_204577


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2045_204586

def A : Set ℝ := {x | -2 < x ∧ x < 4}
def B : Set ℝ := {2, 3, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2045_204586


namespace NUMINAMATH_CALUDE_chess_tournament_theorem_l2045_204512

/-- Represents the number of participants from each city -/
structure Participants where
  moscow : ℕ
  saintPetersburg : ℕ
  kazan : ℕ

/-- Represents the number of games played between participants from different cities -/
structure Games where
  moscowSaintPetersburg : ℕ
  moscowKazan : ℕ
  saintPetersburgKazan : ℕ

/-- The theorem statement based on the chess tournament problem -/
theorem chess_tournament_theorem (p : Participants) (g : Games) : 
  (p.moscow * 9 = p.saintPetersburg * 6) ∧ 
  (p.saintPetersburg * 2 = p.kazan * 6) ∧ 
  (p.moscow * g.moscowKazan = p.kazan * 8) →
  g.moscowKazan = 4 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_theorem_l2045_204512


namespace NUMINAMATH_CALUDE_L_quotient_property_l2045_204540

/-- L(a,b) is defined as the exponent c such that a^c = b, for positive numbers a and b -/
noncomputable def L (a b : ℝ) : ℝ :=
  Real.log b / Real.log a

/-- Theorem: For positive real numbers a, m, and n, L(a, m/n) = L(a,m) - L(a,n) -/
theorem L_quotient_property (a m n : ℝ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  L a (m/n) = L a m - L a n := by
  sorry

end NUMINAMATH_CALUDE_L_quotient_property_l2045_204540


namespace NUMINAMATH_CALUDE_paint_theorem_l2045_204522

def paint_problem (initial_paint : ℚ) (first_day_fraction : ℚ) (second_day_fraction : ℚ) : Prop :=
  let remaining_after_first_day := initial_paint - (first_day_fraction * initial_paint)
  let used_second_day := second_day_fraction * remaining_after_first_day
  let remaining_after_second_day := remaining_after_first_day - used_second_day
  remaining_after_second_day = (4 : ℚ) / 9 * initial_paint

theorem paint_theorem : 
  paint_problem 1 (1/3) (1/3) := by sorry

end NUMINAMATH_CALUDE_paint_theorem_l2045_204522


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2045_204559

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 > 4}

def N : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem intersection_M_complement_N :
  M ∩ (U \ N) = {x : ℝ | x > 3 ∨ x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2045_204559


namespace NUMINAMATH_CALUDE_xiao_ming_math_grade_l2045_204551

/-- Calculates a student's semester math grade based on component scores and weights -/
def semesterMathGrade (routineStudyScore midTermScore finalExamScore : ℝ) : ℝ :=
  0.3 * routineStudyScore + 0.3 * midTermScore + 0.4 * finalExamScore

/-- Xiao Ming's semester math grade is 92.4 points -/
theorem xiao_ming_math_grade :
  semesterMathGrade 90 90 96 = 92.4 := by sorry

end NUMINAMATH_CALUDE_xiao_ming_math_grade_l2045_204551


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l2045_204523

theorem quadratic_always_nonnegative (m : ℝ) : 
  (∀ x : ℝ, x^2 - (m - 1) * x + 1 ≥ 0) ↔ m ∈ Set.Icc (-1 : ℝ) 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l2045_204523
