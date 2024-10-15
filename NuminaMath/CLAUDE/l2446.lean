import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_set_l2446_244620

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ ((m - 1) * x < Real.sqrt (4 * x) - x^2)) → 
  m = 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2446_244620


namespace NUMINAMATH_CALUDE_fraction_simplification_l2446_244696

theorem fraction_simplification : (1625^2 - 1618^2) / (1632^2 - 1611^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2446_244696


namespace NUMINAMATH_CALUDE_scale_division_l2446_244666

theorem scale_division (total_length : ℝ) (num_parts : ℕ) (part_length : ℝ) : 
  total_length = 90 → num_parts = 5 → part_length * num_parts = total_length → part_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l2446_244666


namespace NUMINAMATH_CALUDE_solution_set_for_negative_two_minimum_value_for_one_range_of_m_l2446_244642

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - 2*x - 1

-- Part 1
theorem solution_set_for_negative_two (x : ℝ) :
  f (-2) x ≤ 0 ↔ x ≥ 1 :=
sorry

-- Part 2
theorem minimum_value_for_one (x : ℝ) :
  f 1 x + |x + 2| ≥ 0 :=
sorry

-- Range of m
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, f 1 x + |x + 2| ≤ m) ↔ m ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_negative_two_minimum_value_for_one_range_of_m_l2446_244642


namespace NUMINAMATH_CALUDE_algebraic_multiplication_l2446_244645

theorem algebraic_multiplication (x y : ℝ) : 
  6 * x * y^2 * (-1/2 * x^3 * y^3) = -3 * x^4 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_multiplication_l2446_244645


namespace NUMINAMATH_CALUDE_distance_to_double_reflection_distance_C_to_C_l2446_244630

/-- The distance between a point and its reflection over both x and y axes --/
theorem distance_to_double_reflection (x y : ℝ) : 
  let C : ℝ × ℝ := (x, y)
  let C' : ℝ × ℝ := (-x, -y)
  Real.sqrt ((C'.1 - C.1)^2 + (C'.2 - C.2)^2) = Real.sqrt (4 * (x^2 + y^2)) :=
by sorry

/-- The specific case for point C(-3, 2) --/
theorem distance_C_to_C'_is_sqrt_52 : 
  let C : ℝ × ℝ := (-3, 2)
  let C' : ℝ × ℝ := (3, -2)
  Real.sqrt ((C'.1 - C.1)^2 + (C'.2 - C.2)^2) = Real.sqrt 52 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_double_reflection_distance_C_to_C_l2446_244630


namespace NUMINAMATH_CALUDE_divide_by_six_multiply_by_twelve_l2446_244634

theorem divide_by_six_multiply_by_twelve (x : ℝ) : (x / 6) * 12 = 2 * x := by
  sorry

end NUMINAMATH_CALUDE_divide_by_six_multiply_by_twelve_l2446_244634


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2446_244626

-- Define the complex number z
def z : ℂ := sorry

-- Define the condition
axiom z_condition : Complex.I^3 * z = 2 + Complex.I

-- Theorem to prove
theorem z_in_second_quadrant : 
  Real.sign (z.re) = -1 ∧ Real.sign (z.im) = 1 :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2446_244626


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2446_244619

theorem partial_fraction_decomposition :
  ∀ (A B C : ℝ),
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 →
    5 * x / ((x - 4) * (x - 2)^2) = A / (x - 4) + B / (x - 2) + C / (x - 2)^2) ↔
  A = 5 ∧ B = -5 ∧ C = -5 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2446_244619


namespace NUMINAMATH_CALUDE_proposition_equivalences_l2446_244669

theorem proposition_equivalences (a b c : ℝ) :
  (((c < 0 ∧ a * c > b * c) → a < b) ∧
   ((c < 0 ∧ a < b) → a * c > b * c) ∧
   ((c < 0 ∧ a * c ≤ b * c) → a ≥ b) ∧
   ((c < 0 ∧ a ≥ b) → a * c ≤ b * c) ∧
   ((a * b = 0) → (a = 0 ∨ b = 0)) ∧
   ((a = 0 ∨ b = 0) → a * b = 0) ∧
   ((a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0)) ∧
   ((a ≠ 0 ∧ b ≠ 0) → a * b ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_equivalences_l2446_244669


namespace NUMINAMATH_CALUDE_range_of_a_l2446_244622

open Set Real

noncomputable def f (x : ℝ) : ℝ := 4 * x / (3 * x^2 + 3)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - log x - a

def I : Set ℝ := Ioo 0 2
def J : Set ℝ := Icc 1 2

theorem range_of_a :
  {a : ℝ | ∀ x₁ ∈ I, ∃ x₂ ∈ J, f x₁ = g a x₂} = Icc (1/2) (4/3 - log 2) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2446_244622


namespace NUMINAMATH_CALUDE_inscribed_rectangle_epsilon_l2446_244695

-- Define the triangle
structure Triangle :=
  (MN NP PM : ℝ)

-- Define the rectangle
structure Rectangle :=
  (W X Y Z : ℝ × ℝ)

-- Define the area function
def rectangleArea (γ ε δ : ℝ) : ℝ := γ * δ - δ * ε^2

theorem inscribed_rectangle_epsilon (t : Triangle) (r : Rectangle) (γ ε : ℝ) :
  t.MN = 10 ∧ t.NP = 24 ∧ t.PM = 26 →
  (∃ δ, rectangleArea γ ε δ = 0) →
  (∃ δ, rectangleArea γ ε δ = 60) →
  ε = 5/12 := by
  sorry

#check inscribed_rectangle_epsilon

end NUMINAMATH_CALUDE_inscribed_rectangle_epsilon_l2446_244695


namespace NUMINAMATH_CALUDE_shared_focus_parabola_ellipse_l2446_244638

/-- Given a parabola and an ellipse that share a focus, prove the value of m in the ellipse equation -/
theorem shared_focus_parabola_ellipse (x y : ℝ) (m : ℝ) : 
  (x^2 = 2*y) →  -- Parabola equation
  (y^2/m + x^2/2 = 1) →  -- Ellipse equation
  (∃ f : ℝ × ℝ, f ∈ {p : ℝ × ℝ | p.1^2 = 2*p.2} ∩ {e : ℝ × ℝ | e.2^2/m + e.1^2/2 = 1}) →  -- Shared focus
  (m = 9/4) :=
by sorry

end NUMINAMATH_CALUDE_shared_focus_parabola_ellipse_l2446_244638


namespace NUMINAMATH_CALUDE_product_mod_seventeen_is_zero_l2446_244618

theorem product_mod_seventeen_is_zero :
  (2001 * 2002 * 2003 * 2004 * 2005 * 2006 * 2007) % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seventeen_is_zero_l2446_244618


namespace NUMINAMATH_CALUDE_exponent_equality_l2446_244652

theorem exponent_equality (a b : ℝ) : (-a * b^3)^2 = a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l2446_244652


namespace NUMINAMATH_CALUDE_percentage_increase_l2446_244617

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 1500 → final = 1800 → (final - initial) / initial * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2446_244617


namespace NUMINAMATH_CALUDE_total_tickets_sold_l2446_244687

/-- Given the number of tickets sold in section A and section B, prove that the total number of tickets sold is their sum. -/
theorem total_tickets_sold (section_a_tickets : ℕ) (section_b_tickets : ℕ) :
  section_a_tickets = 2900 →
  section_b_tickets = 1600 →
  section_a_tickets + section_b_tickets = 4500 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_sold_l2446_244687


namespace NUMINAMATH_CALUDE_circle_passes_through_fixed_point_circle_tangent_conditions_l2446_244697

/-- The equation of the given circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*(a - 1) = 0

/-- The equation of the fixed circle -/
def fixed_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem circle_passes_through_fixed_point :
  ∀ a : ℝ, circle_equation 4 (-2) a := by sorry

theorem circle_tangent_conditions :
  ∀ a : ℝ, (∃ x y : ℝ, circle_equation x y a ∧ fixed_circle_equation x y ∧
    (∀ x' y' : ℝ, circle_equation x' y' a ∧ fixed_circle_equation x' y' → (x', y') = (x, y))) ↔
  (a = 1 - Real.sqrt 5 ∨ a = 1 + Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_circle_passes_through_fixed_point_circle_tangent_conditions_l2446_244697


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_is_twenty_l2446_244604

/-- The coefficient of x^3 in the expansion of (2x + 1/(4x))^5 -/
def coefficient_x_cubed : ℚ :=
  let a := 2  -- coefficient of x
  let b := 1 / 4  -- coefficient of 1/x
  let n := 5  -- exponent
  let k := (n - 3) / 2  -- power of x is n - 2k, so n - 2k = 3
  (n.choose k) * (a ^ (n - k)) * (b ^ k)

/-- Theorem stating that the coefficient of x^3 in (2x + 1/(4x))^5 is 20 -/
theorem coefficient_x_cubed_is_twenty : coefficient_x_cubed = 20 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_is_twenty_l2446_244604


namespace NUMINAMATH_CALUDE_remainder_17_63_mod_7_l2446_244639

theorem remainder_17_63_mod_7 : 17^63 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_17_63_mod_7_l2446_244639


namespace NUMINAMATH_CALUDE_line_equation_l2446_244653

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

/-- Point M -/
def M : ℝ × ℝ := (4, 1)

/-- Line passing through two points -/
def line_through (p₁ p₂ : ℝ × ℝ) (x y : ℝ) : Prop :=
  (y - p₁.2) * (p₂.1 - p₁.1) = (x - p₁.1) * (p₂.2 - p₁.2)

/-- Midpoint of two points -/
def is_midpoint (m p₁ p₂ : ℝ × ℝ) : Prop :=
  m.1 = (p₁.1 + p₂.1) / 2 ∧ m.2 = (p₁.2 + p₂.2) / 2

theorem line_equation :
  ∃ (A B : ℝ × ℝ),
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    is_midpoint M A B ∧
    (∀ x y, line_through M (x, y) x y ↔ y = 8*x - 31) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_l2446_244653


namespace NUMINAMATH_CALUDE_root_equation_value_l2446_244625

theorem root_equation_value (b c : ℝ) : 
  (2 : ℝ)^2 - b * 2 + c = 0 → 4 * b - 2 * c + 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_value_l2446_244625


namespace NUMINAMATH_CALUDE_banana_bread_recipe_l2446_244640

/-- Banana bread recipe problem -/
theorem banana_bread_recipe 
  (bananas_per_mush : ℕ) 
  (total_bananas : ℕ) 
  (total_flour : ℕ) 
  (h1 : bananas_per_mush = 4)
  (h2 : total_bananas = 20)
  (h3 : total_flour = 15) :
  (total_flour : ℚ) / ((total_bananas : ℚ) / (bananas_per_mush : ℚ)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_banana_bread_recipe_l2446_244640


namespace NUMINAMATH_CALUDE_square_division_l2446_244637

theorem square_division (original_side : ℝ) (n : ℕ) (smaller_side : ℝ) : 
  original_side = 12 →
  n = 4 →
  smaller_side^2 * n = original_side^2 →
  smaller_side = 6 := by
sorry

end NUMINAMATH_CALUDE_square_division_l2446_244637


namespace NUMINAMATH_CALUDE_andrei_valentin_distance_at_finish_l2446_244635

/-- Represents the race scenario with three runners -/
structure RaceScenario where
  race_distance : ℝ
  andrei_boris_gap : ℝ
  boris_valentin_gap : ℝ

/-- Calculates the distance between Andrei and Valentin at Andrei's finish -/
def distance_andrei_valentin (scenario : RaceScenario) : ℝ :=
  scenario.race_distance - (scenario.race_distance - scenario.andrei_boris_gap - scenario.boris_valentin_gap)

/-- Theorem stating the distance between Andrei and Valentin when Andrei finishes -/
theorem andrei_valentin_distance_at_finish (scenario : RaceScenario) 
  (h1 : scenario.race_distance = 1000)
  (h2 : scenario.andrei_boris_gap = 100)
  (h3 : scenario.boris_valentin_gap = 50) :
  distance_andrei_valentin scenario = 145 := by
  sorry

#eval distance_andrei_valentin ⟨1000, 100, 50⟩

end NUMINAMATH_CALUDE_andrei_valentin_distance_at_finish_l2446_244635


namespace NUMINAMATH_CALUDE_max_sum_fraction_min_sum_fraction_l2446_244623

def Digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The maximum value of A/B + C/P given six different digits from Digits -/
theorem max_sum_fraction (A B C P Q R : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ P ∧ A ≠ Q ∧ A ≠ R ∧
                B ≠ C ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧
                C ≠ P ∧ C ≠ Q ∧ C ≠ R ∧
                P ≠ Q ∧ P ≠ R ∧
                Q ≠ R)
  (h_in_digits : A ∈ Digits ∧ B ∈ Digits ∧ C ∈ Digits ∧ 
                 P ∈ Digits ∧ Q ∈ Digits ∧ R ∈ Digits) :
  (A : ℚ) / B + (C : ℚ) / P ≤ 13 :=
sorry

/-- The minimum value of Q/R + P/C using the remaining digits -/
theorem min_sum_fraction (A B C P Q R : ℕ) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ P ∧ A ≠ Q ∧ A ≠ R ∧
                B ≠ C ∧ B ≠ P ∧ B ≠ Q ∧ B ≠ R ∧
                C ≠ P ∧ C ≠ Q ∧ C ≠ R ∧
                P ≠ Q ∧ P ≠ R ∧
                Q ≠ R)
  (h_in_digits : A ∈ Digits ∧ B ∈ Digits ∧ C ∈ Digits ∧ 
                 P ∈ Digits ∧ Q ∈ Digits ∧ R ∈ Digits) :
  (Q : ℚ) / R + (P : ℚ) / C ≥ 23 / 21 :=
sorry

end NUMINAMATH_CALUDE_max_sum_fraction_min_sum_fraction_l2446_244623


namespace NUMINAMATH_CALUDE_pentagon_angle_measure_l2446_244667

-- Define a pentagon
structure Pentagon where
  P : ℝ
  Q : ℝ
  R : ℝ
  S : ℝ
  T : ℝ

-- Define the theorem
theorem pentagon_angle_measure (PQRST : Pentagon) 
  (h1 : PQRST.P = PQRST.R ∧ PQRST.R = PQRST.T)  -- ∠P ≅ ∠R ≅ ∠T
  (h2 : PQRST.Q + PQRST.S = 180)  -- ∠Q is supplementary to ∠S
  (h3 : PQRST.P + PQRST.Q + PQRST.R + PQRST.S + PQRST.T = 540)  -- Sum of angles in a pentagon
  : PQRST.T = 120 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_measure_l2446_244667


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l2446_244647

theorem smallest_sum_of_sequence (E F G H : ℕ+) : 
  (∃ d : ℤ, (F : ℤ) - (E : ℤ) = d ∧ (G : ℤ) - (F : ℤ) = d) →  -- arithmetic sequence condition
  (∃ r : ℚ, (G : ℚ) / (F : ℚ) = r ∧ (H : ℚ) / (G : ℚ) = r) →  -- geometric sequence condition
  (G : ℚ) / (F : ℚ) = 4 / 3 →                                -- given ratio
  E + F + G + H ≥ 43 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l2446_244647


namespace NUMINAMATH_CALUDE_exponent_and_polynomial_identities_l2446_244670

variable (a b : ℝ)

theorem exponent_and_polynomial_identities : 
  ((a^2)^3 / (-a)^2 = a^4) ∧ 
  ((a+2*b)*(a+b)-3*a*(a+b) = -2*a^2 + 2*b^2) := by sorry

end NUMINAMATH_CALUDE_exponent_and_polynomial_identities_l2446_244670


namespace NUMINAMATH_CALUDE_logarithm_difference_equals_two_l2446_244641

theorem logarithm_difference_equals_two :
  (Real.log 80 / Real.log 2) / (Real.log 40 / Real.log 2) -
  (Real.log 160 / Real.log 2) / (Real.log 20 / Real.log 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_difference_equals_two_l2446_244641


namespace NUMINAMATH_CALUDE_lcm_12_18_l2446_244632

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_18_l2446_244632


namespace NUMINAMATH_CALUDE_possible_values_of_a_l2446_244694

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a = -2 ∨ a = -1 ∨ a = 0 ∨ a = 1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l2446_244694


namespace NUMINAMATH_CALUDE_bullying_instances_l2446_244655

def days_per_bullying : ℕ := 3
def typical_fingers_and_toes : ℕ := 20
def additional_suspension_days : ℕ := 14

def total_suspension_days : ℕ := 3 * typical_fingers_and_toes + additional_suspension_days

theorem bullying_instances : 
  (total_suspension_days / days_per_bullying : ℕ) = 24 := by
  sorry

end NUMINAMATH_CALUDE_bullying_instances_l2446_244655


namespace NUMINAMATH_CALUDE_no_rain_probability_l2446_244668

theorem no_rain_probability (p : ℝ) (h : p = 2/3) : (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l2446_244668


namespace NUMINAMATH_CALUDE_married_men_count_l2446_244683

theorem married_men_count (total : ℕ) (tv : ℕ) (radio : ℕ) (ac : ℕ) (all_and_married : ℕ) 
  (h_total : total = 100)
  (h_tv : tv = 75)
  (h_radio : radio = 85)
  (h_ac : ac = 70)
  (h_all_and_married : all_and_married = 12)
  (h_all_and_married_le_total : all_and_married ≤ total) :
  ∃ (married : ℕ), married ≥ all_and_married ∧ married ≤ total :=
by
  sorry

end NUMINAMATH_CALUDE_married_men_count_l2446_244683


namespace NUMINAMATH_CALUDE_triangle_problem_l2446_244615

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.C.cos = 2 * t.A.cos * (t.B - π/6).sin)
  (h2 : t.S = 2 * Real.sqrt 3)
  (h3 : t.b - t.c = 2) :
  t.A = π/3 ∧ t.a = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2446_244615


namespace NUMINAMATH_CALUDE_fruit_basket_ratio_l2446_244664

/-- The number of bananas in the blue basket -/
def blue_bananas : ℕ := 12

/-- The number of apples in the blue basket -/
def blue_apples : ℕ := 4

/-- The number of fruits in the red basket -/
def red_fruits : ℕ := 8

/-- The total number of fruits in the blue basket -/
def blue_total : ℕ := blue_bananas + blue_apples

/-- The ratio of fruits in the red basket to the blue basket -/
def fruit_ratio : ℚ := red_fruits / blue_total

theorem fruit_basket_ratio : fruit_ratio = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_ratio_l2446_244664


namespace NUMINAMATH_CALUDE_fathers_age_fathers_current_age_l2446_244691

theorem fathers_age (sons_age_next_year : ℕ) (father_age_ratio : ℕ) : ℕ :=
  let sons_current_age := sons_age_next_year - 1
  father_age_ratio * sons_current_age

theorem fathers_current_age :
  fathers_age 8 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_fathers_current_age_l2446_244691


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2446_244688

theorem inverse_sum_reciprocal (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a⁻¹ + b⁻¹ + c⁻¹)⁻¹ = (a * b * c) / (b * c + a * c + a * b) := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocal_l2446_244688


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l2446_244661

/-- Given a geometric sequence with positive terms and common ratio q > 0, q ≠ 1,
    the sum of the first and fourth terms is greater than the sum of the second and third terms. -/
theorem geometric_sequence_sum_inequality {a : ℕ → ℝ} {q : ℝ} 
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_pos : ∀ n, a n > 0)
  (h_q_pos : q > 0)
  (h_q_neq_1 : q ≠ 1) :
  a 1 + a 4 > a 2 + a 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_inequality_l2446_244661


namespace NUMINAMATH_CALUDE_percent_problem_l2446_244606

theorem percent_problem (x : ℝ) (h : 0.22 * x = 66) : x = 300 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l2446_244606


namespace NUMINAMATH_CALUDE_shipping_cost_correct_l2446_244673

/-- The cost function for shipping packages -/
def shipping_cost (W : ℕ) : ℝ :=
  5 + 4 * (W - 1)

/-- Theorem stating the correctness of the shipping cost formula -/
theorem shipping_cost_correct (W : ℕ) (h : W ≥ 2) :
  shipping_cost W = 5 + 4 * (W - 1) :=
by sorry

end NUMINAMATH_CALUDE_shipping_cost_correct_l2446_244673


namespace NUMINAMATH_CALUDE_star_two_three_l2446_244650

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 * b^2 - a + 1

-- Theorem statement
theorem star_two_three : star 2 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_star_two_three_l2446_244650


namespace NUMINAMATH_CALUDE_min_ping_pong_balls_l2446_244675

def is_valid_box_count (n : ℕ) : Prop :=
  n ≥ 11 ∧ n ≠ 17 ∧ n % 6 ≠ 0

def distinct_counts (counts : List ℕ) : Prop :=
  counts.Nodup

theorem min_ping_pong_balls :
  ∃ (counts : List ℕ),
    counts.length = 10 ∧
    (∀ n ∈ counts, is_valid_box_count n) ∧
    distinct_counts counts ∧
    counts.sum = 174 ∧
    (∀ (other_counts : List ℕ),
      other_counts.length = 10 →
      (∀ n ∈ other_counts, is_valid_box_count n) →
      distinct_counts other_counts →
      other_counts.sum ≥ 174) :=
by sorry

end NUMINAMATH_CALUDE_min_ping_pong_balls_l2446_244675


namespace NUMINAMATH_CALUDE_roots_of_equation_l2446_244681

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^2 - 4*x + 3)*(x - 5)*(x + 1)
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 3 ∨ x = 5 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2446_244681


namespace NUMINAMATH_CALUDE_sum_fractions_l2446_244680

theorem sum_fractions (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 9) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_sum_fractions_l2446_244680


namespace NUMINAMATH_CALUDE_bucket_weight_l2446_244611

/-- Given a bucket with weight p when three-quarters full and weight q when one-third full,
    prove that its weight when full is (8p - 7q) / 5 -/
theorem bucket_weight (p q : ℝ) : ℝ :=
  let x := (5 * q - 4 * p) / 5  -- weight of empty bucket
  let y := (12 * (p - q)) / 5   -- weight of water when bucket is full
  let weight_three_quarters := x + 3/4 * y
  let weight_one_third := x + 1/3 * y
  have h1 : weight_three_quarters = p := by sorry
  have h2 : weight_one_third = q := by sorry
  (8 * p - 7 * q) / 5

#check bucket_weight

end NUMINAMATH_CALUDE_bucket_weight_l2446_244611


namespace NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l2446_244659

theorem consecutive_integers_fourth_power_sum (n : ℤ) : 
  n * (n + 1) * (n + 2) = 12 * (3 * n + 3) → 
  n^4 + (n + 1)^4 + (n + 2)^4 = 7793 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_fourth_power_sum_l2446_244659


namespace NUMINAMATH_CALUDE_problem1_l2446_244662

theorem problem1 : |-3| - Real.sqrt 12 + 2 * Real.sin (30 * π / 180) + (-1) ^ 2021 = 3 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_problem1_l2446_244662


namespace NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l2446_244648

theorem sum_and_ratio_to_difference (x y : ℚ) 
  (sum_eq : x + y = 500)
  (ratio_eq : x / y = 4/5) :
  y - x = 500/9 := by
sorry

end NUMINAMATH_CALUDE_sum_and_ratio_to_difference_l2446_244648


namespace NUMINAMATH_CALUDE_sams_first_month_earnings_l2446_244685

/-- Sam's hourly rate for Math tutoring -/
def hourly_rate : ℕ := 10

/-- The difference in earnings between the second and first month -/
def second_month_increase : ℕ := 150

/-- Total hours spent tutoring over two months -/
def total_hours : ℕ := 55

/-- Sam's earnings in the first month -/
def first_month_earnings : ℕ := 200

/-- Theorem stating that Sam's earnings in the first month were $200 -/
theorem sams_first_month_earnings :
  first_month_earnings = (hourly_rate * total_hours - second_month_increase) / 2 :=
by sorry

end NUMINAMATH_CALUDE_sams_first_month_earnings_l2446_244685


namespace NUMINAMATH_CALUDE_extremum_and_derivative_not_equivalent_l2446_244682

-- Define a function type that represents real-valued functions of a real variable
def RealFunction := ℝ → ℝ

-- Define what it means for a function to have an extremum at a point
def has_extremum (f : RealFunction) (a : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - a| < ε → f a ≤ f x ∨ f a ≥ f x

-- Define the derivative of a function at a point
noncomputable def has_derivative_at (f : RealFunction) (a : ℝ) (f' : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x - f a - f' * (x - a)| ≤ ε * |x - a|

-- Theorem statement
theorem extremum_and_derivative_not_equivalent :
  ∃ (f : RealFunction) (a : ℝ),
    (has_extremum f a ∧ ¬(has_derivative_at f a 0)) ∧
    ∃ (g : RealFunction) (b : ℝ),
      (has_derivative_at g b 0 ∧ ¬(has_extremum g b)) :=
sorry

end NUMINAMATH_CALUDE_extremum_and_derivative_not_equivalent_l2446_244682


namespace NUMINAMATH_CALUDE_complex_modulus_range_l2446_244609

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z = Complex.mk a 1) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l2446_244609


namespace NUMINAMATH_CALUDE_gwen_spent_eight_dollars_l2446_244665

/-- The amount of money Gwen received for her birthday. -/
def initial_amount : ℕ := 14

/-- The amount of money Gwen has left. -/
def remaining_amount : ℕ := 6

/-- The amount of money Gwen spent. -/
def spent_amount : ℕ := initial_amount - remaining_amount

theorem gwen_spent_eight_dollars : spent_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_gwen_spent_eight_dollars_l2446_244665


namespace NUMINAMATH_CALUDE_square_side_length_l2446_244658

theorem square_side_length (r : ℝ) (h : r = 3) : 
  ∃ x : ℝ, 4 * x = 2 * π * r ∧ x = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l2446_244658


namespace NUMINAMATH_CALUDE_all_positive_rationals_in_X_l2446_244629

theorem all_positive_rationals_in_X (X : Set ℚ) 
  (h1 : ∀ x : ℚ, 2021 ≤ x ∧ x ≤ 2022 → x ∈ X) 
  (h2 : ∀ x y : ℚ, x ∈ X → y ∈ X → (x / y) ∈ X) :
  ∀ q : ℚ, 0 < q → q ∈ X := by
  sorry

end NUMINAMATH_CALUDE_all_positive_rationals_in_X_l2446_244629


namespace NUMINAMATH_CALUDE_product_of_largest_primes_l2446_244633

def largest_one_digit_primes : Finset Nat := {5, 7}
def largest_two_digit_prime : Nat := 97

theorem product_of_largest_primes : 
  (Finset.prod largest_one_digit_primes id) * largest_two_digit_prime = 3395 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_primes_l2446_244633


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l2446_244643

theorem fourth_rectangle_area (a b c : ℕ) (h1 : a + b + c = 350) : 
  ∃ d : ℕ, d = 300 - (a + b + c) ∧ d = 50 := by
  sorry

#check fourth_rectangle_area

end NUMINAMATH_CALUDE_fourth_rectangle_area_l2446_244643


namespace NUMINAMATH_CALUDE_intersection_and_chord_properties_l2446_244654

/-- Given two points M and N in a 2D Cartesian coordinate system -/
def M : ℝ × ℝ := (1, -3)
def N : ℝ × ℝ := (5, 1)

/-- Point C satisfies the given condition -/
def C (t : ℝ) : ℝ × ℝ :=
  (t * M.1 + (1 - t) * N.1, t * M.2 + (1 - t) * N.2)

/-- The parabola y^2 = 4x -/
def on_parabola (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

/-- Perpendicularity of two vectors -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Main theorem -/
theorem intersection_and_chord_properties :
  (∃ A B : ℝ × ℝ, 
    (∃ t : ℝ, C t = A) ∧ 
    (∃ t : ℝ, C t = B) ∧ 
    on_parabola A ∧ 
    on_parabola B ∧ 
    perpendicular A B) ∧
  (∃ P : ℝ × ℝ, P.1 = 4 ∧ P.2 = 0 ∧
    ∀ Q R : ℝ × ℝ, 
      on_parabola Q ∧ 
      on_parabola R ∧ 
      (Q.2 - P.2) * (R.1 - P.1) = (Q.1 - P.1) * (R.2 - P.2) →
      (Q.1 * R.1 + Q.2 * R.2 = 0)) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_chord_properties_l2446_244654


namespace NUMINAMATH_CALUDE_trig_identity_l2446_244699

/-- Prove that (cos 70° * cos 20°) / (1 - 2 * sin² 25°) = 1/2 -/
theorem trig_identity : 
  (Real.cos (70 * π / 180) * Real.cos (20 * π / 180)) / 
  (1 - 2 * Real.sin (25 * π / 180) ^ 2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2446_244699


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l2446_244636

theorem part_to_whole_ratio (N : ℝ) (P : ℝ) : 
  (1 / 4 : ℝ) * P = 10 →
  (40 / 100 : ℝ) * N = 120 →
  P / ((2 / 5 : ℝ) * N) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l2446_244636


namespace NUMINAMATH_CALUDE_prove_average_growth_rate_l2446_244684

-- Define the initial number of books borrowed in 2020
def initial_books : ℝ := 7500

-- Define the final number of books borrowed in 2022
def final_books : ℝ := 10800

-- Define the number of years between 2020 and 2022
def years : ℕ := 2

-- Define the average annual growth rate
def average_growth_rate : ℝ := 0.2

-- Theorem statement
theorem prove_average_growth_rate :
  initial_books * (1 + average_growth_rate) ^ years = final_books := by
  sorry

end NUMINAMATH_CALUDE_prove_average_growth_rate_l2446_244684


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2446_244690

def M : Set ℕ := {1, 2}

def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_of_M_and_N : M ∪ N = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2446_244690


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2446_244607

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  arithmetic_property : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_of_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) :
  sum_of_terms seq 9 = 54 → 2 + seq.a 4 + 9 = 307 / 27 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2446_244607


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2446_244686

theorem unique_three_digit_number : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 10 = 3 ∧ 
  (300 + n / 10) = 3 * n + 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2446_244686


namespace NUMINAMATH_CALUDE_power_of_negative_power_l2446_244610

theorem power_of_negative_power (a : ℝ) : (-2 * a^4)^3 = -8 * a^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_power_l2446_244610


namespace NUMINAMATH_CALUDE_square_root_div_five_l2446_244621

theorem square_root_div_five : Real.sqrt 625 / 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_div_five_l2446_244621


namespace NUMINAMATH_CALUDE_square_area_from_perimeter_l2446_244605

/-- The area in square centimeters of a square with perimeter 28 dm is 4900 -/
theorem square_area_from_perimeter : 
  let perimeter : ℝ := 28
  let side_length : ℝ := perimeter / 4
  let area_dm : ℝ := side_length ^ 2
  let area_cm : ℝ := area_dm * 100
  area_cm = 4900 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_perimeter_l2446_244605


namespace NUMINAMATH_CALUDE_chinese_character_sum_l2446_244627

theorem chinese_character_sum (a b c d : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  100 * a + 10 * b + c + 100 * c + 10 * b + d = 1000 * a + 100 * b + 10 * c + d →
  1000 * a + 100 * b + 10 * c + d = 18 := by
sorry

end NUMINAMATH_CALUDE_chinese_character_sum_l2446_244627


namespace NUMINAMATH_CALUDE_actual_miles_traveled_l2446_244651

/-- A function that counts the number of integers from 0 to n (inclusive) that contain the digit 3 --/
def countWithThree (n : ℕ) : ℕ := sorry

/-- The odometer reading --/
def odometerReading : ℕ := 3008

/-- Theorem stating that the actual miles traveled is 2465 when the odometer reads 3008 --/
theorem actual_miles_traveled :
  odometerReading - countWithThree odometerReading = 2465 := by sorry

end NUMINAMATH_CALUDE_actual_miles_traveled_l2446_244651


namespace NUMINAMATH_CALUDE_percent_difference_l2446_244614

theorem percent_difference (y u w z : ℝ) 
  (hw : w = 0.6 * u) 
  (hu : u = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  (z - w) / w = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_percent_difference_l2446_244614


namespace NUMINAMATH_CALUDE_abs_diff_bound_l2446_244689

theorem abs_diff_bound (a b c h : ℝ) (ha : |a - c| < h) (hb : |b - c| < h) : |a - b| < 2 * h := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_bound_l2446_244689


namespace NUMINAMATH_CALUDE_car_speed_problem_l2446_244602

/-- Proves that a car traveling for two hours with an average speed of 80 km/h
    and a second hour speed of 90 km/h must have a first hour speed of 70 km/h. -/
theorem car_speed_problem (first_hour_speed second_hour_speed average_speed : ℝ) :
  second_hour_speed = 90 →
  average_speed = 80 →
  average_speed = (first_hour_speed + second_hour_speed) / 2 →
  first_hour_speed = 70 := by
sorry


end NUMINAMATH_CALUDE_car_speed_problem_l2446_244602


namespace NUMINAMATH_CALUDE_smallest_a_for_sqrt_12a_integer_three_satisfies_condition_three_is_smallest_l2446_244603

theorem smallest_a_for_sqrt_12a_integer (a : ℕ) : 
  (∃ (n : ℕ), n > 0 ∧ n^2 = 12*a) → a ≥ 3 :=
sorry

theorem three_satisfies_condition : 
  ∃ (n : ℕ), n > 0 ∧ n^2 = 12*3 :=
sorry

theorem three_is_smallest : 
  ∀ (a : ℕ), a > 0 → (∃ (n : ℕ), n > 0 ∧ n^2 = 12*a) → a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_for_sqrt_12a_integer_three_satisfies_condition_three_is_smallest_l2446_244603


namespace NUMINAMATH_CALUDE_nested_square_root_equality_l2446_244663

theorem nested_square_root_equality : 
  Real.sqrt (1 + 2014 * Real.sqrt (1 + 2015 * Real.sqrt (1 + 2016 * 2018))) = 2015 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_equality_l2446_244663


namespace NUMINAMATH_CALUDE_necklaces_made_l2446_244649

def total_beads : ℕ := 52
def beads_per_necklace : ℕ := 2

theorem necklaces_made : total_beads / beads_per_necklace = 26 := by
  sorry

end NUMINAMATH_CALUDE_necklaces_made_l2446_244649


namespace NUMINAMATH_CALUDE_defective_units_percentage_l2446_244678

theorem defective_units_percentage
  (shipped_defective_ratio : Real)
  (total_shipped_defective_ratio : Real)
  (h1 : shipped_defective_ratio = 0.04)
  (h2 : total_shipped_defective_ratio = 0.0036) :
  ∃ (defective_ratio : Real),
    defective_ratio * shipped_defective_ratio = total_shipped_defective_ratio ∧
    defective_ratio = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l2446_244678


namespace NUMINAMATH_CALUDE_min_value_implications_l2446_244613

theorem min_value_implications (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hmin : ∀ x, |x + a| + |x - b| ≥ 2) : 
  (3 * a^2 + b^2 ≥ 3) ∧ (4 / (a + 1) + 1 / b ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_implications_l2446_244613


namespace NUMINAMATH_CALUDE_elf_goblin_theorem_l2446_244671

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Number of valid arrangements of elves and goblins -/
def elf_goblin_arrangements (n : ℕ) : ℕ := fib (n + 1)

/-- Theorem: The number of valid arrangements of n elves and n goblins,
    where no two goblins can be adjacent, is equal to the (n+2)th Fibonacci number -/
theorem elf_goblin_theorem (n : ℕ) :
  elf_goblin_arrangements n = fib (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_elf_goblin_theorem_l2446_244671


namespace NUMINAMATH_CALUDE_pascal_sum_29_l2446_244698

/-- Number of elements in a row of Pascal's Triangle -/
def pascal_row_count (n : ℕ) : ℕ := n + 1

/-- Sum of elements in Pascal's Triangle from row 0 to row n -/
def pascal_sum (n : ℕ) : ℕ :=
  (n + 1) * (n + 2) / 2

theorem pascal_sum_29 : pascal_sum 29 = 465 := by
  sorry

end NUMINAMATH_CALUDE_pascal_sum_29_l2446_244698


namespace NUMINAMATH_CALUDE_income_calculation_l2446_244628

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 3 = expenditure * 5 →
  income - expenditure = savings →
  savings = 4000 →
  income = 10000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l2446_244628


namespace NUMINAMATH_CALUDE_quadratic_product_equals_quadratic_l2446_244600

/-- A quadratic polynomial with integer coefficients -/
def QuadraticPolynomial (a b : ℤ) : ℤ → ℤ := fun x ↦ x^2 + a * x + b

theorem quadratic_product_equals_quadratic (a b n : ℤ) :
  ∃ M : ℤ, (QuadraticPolynomial a b n) * (QuadraticPolynomial a b (n + 1)) =
    QuadraticPolynomial a b M := by
  sorry

end NUMINAMATH_CALUDE_quadratic_product_equals_quadratic_l2446_244600


namespace NUMINAMATH_CALUDE_coefficient_x3y7_expansion_l2446_244677

theorem coefficient_x3y7_expansion :
  let n : ℕ := 10
  let k : ℕ := 3
  let coeff : ℚ := (n.choose k) * (2/3)^k * (-3/5)^(n-k)
  coeff = -256/257 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3y7_expansion_l2446_244677


namespace NUMINAMATH_CALUDE_triangle_angle_A_l2446_244672

theorem triangle_angle_A (a b c : ℝ) (A B C : ℝ) :
  a = 3 →
  b = 4 →
  Real.sin B = 2/3 →
  a < b →
  (Real.sin A) * b = a * (Real.sin B) →
  A = π/6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_A_l2446_244672


namespace NUMINAMATH_CALUDE_inequality_proof_l2446_244657

theorem inequality_proof (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_prod : a * b * c * d = 1) : 
  (1 / (a * (1 + b))) + (1 / (b * (1 + c))) + (1 / (c * (1 + d))) + (1 / (d * (1 + a))) ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2446_244657


namespace NUMINAMATH_CALUDE_soda_packing_l2446_244676

theorem soda_packing (total : ℕ) (regular : ℕ) (diet : ℕ) (pack_size : ℕ) :
  total = 200 →
  regular = 55 →
  diet = 40 →
  pack_size = 3 →
  let energy := total - regular - diet
  let complete_packs := energy / pack_size
  let leftover := energy % pack_size
  complete_packs = 35 ∧ leftover = 0 := by
  sorry

end NUMINAMATH_CALUDE_soda_packing_l2446_244676


namespace NUMINAMATH_CALUDE_group_size_calculation_l2446_244608

theorem group_size_calculation (average_increase : ℝ) (weight_difference : ℝ) : 
  average_increase = 3 ∧ weight_difference = 30 → 
  (weight_difference / average_increase : ℝ) = 10 := by
  sorry

#check group_size_calculation

end NUMINAMATH_CALUDE_group_size_calculation_l2446_244608


namespace NUMINAMATH_CALUDE_rotation_90_ccw_parabola_l2446_244656

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define the original function
def original_function (x : ℝ) : ℝ := x^2

-- Define the rotation operation
def rotate_90_ccw (p : Point) : Point :=
  { x := -p.y, y := p.x }

-- Define the rotated function
def rotated_function (y : ℝ) : ℝ := -y^2

-- Theorem statement
theorem rotation_90_ccw_parabola :
  ∀ (p : Point), p.y = original_function p.x →
  (rotate_90_ccw p).y = rotated_function (rotate_90_ccw p).x :=
sorry

end NUMINAMATH_CALUDE_rotation_90_ccw_parabola_l2446_244656


namespace NUMINAMATH_CALUDE_infinite_non_sum_of_three_cubes_l2446_244692

theorem infinite_non_sum_of_three_cubes :
  ∀ k : ℤ, ¬∃ a b c : ℤ, (9*k + 4 = a^3 + b^3 + c^3) ∧ ¬∃ a b c : ℤ, (9*k - 4 = a^3 + b^3 + c^3) :=
by
  sorry

end NUMINAMATH_CALUDE_infinite_non_sum_of_three_cubes_l2446_244692


namespace NUMINAMATH_CALUDE_intersection_determines_a_l2446_244612

theorem intersection_determines_a (A B : Set ℝ) (a : ℝ) :
  A = {1, 3, a} →
  B = {4, 5} →
  A ∩ B = {4} →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_determines_a_l2446_244612


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2446_244679

theorem weight_of_replaced_person (initial_count : ℕ) (avg_increase : ℚ) (new_weight : ℚ) :
  initial_count = 6 →
  avg_increase = 4.5 →
  new_weight = 102 →
  ∃ (old_weight : ℚ), old_weight = 75 ∧ new_weight = old_weight + initial_count * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2446_244679


namespace NUMINAMATH_CALUDE_inequalities_check_l2446_244646

theorem inequalities_check :
  (∀ x : ℝ, x^2 + 3 > 2*x) ∧
  (∃ a b : ℝ, a^5 + b^5 < a^3*b^2 + a^2*b^3) ∧
  (∀ a b : ℝ, a^2 + b^2 ≥ 2*(a - b - 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequalities_check_l2446_244646


namespace NUMINAMATH_CALUDE_waiter_problem_l2446_244674

theorem waiter_problem (initial_customers : ℕ) (left_customers : ℕ) (num_tables : ℕ) 
  (h1 : initial_customers = 62)
  (h2 : left_customers = 17)
  (h3 : num_tables = 5) :
  (initial_customers - left_customers) / num_tables = 9 := by
  sorry

end NUMINAMATH_CALUDE_waiter_problem_l2446_244674


namespace NUMINAMATH_CALUDE_correct_sample_ids_l2446_244624

/-- A function that generates the sample IDs based on the given conditions -/
def generateSampleIDs (populationSize : Nat) (sampleSize : Nat) : List Nat :=
  (List.range sampleSize).map (fun i => 6 * i + 3)

/-- The theorem stating that the generated sample IDs match the expected result -/
theorem correct_sample_ids :
  generateSampleIDs 60 10 = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57] := by
  sorry

#eval generateSampleIDs 60 10

end NUMINAMATH_CALUDE_correct_sample_ids_l2446_244624


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l2446_244631

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l2446_244631


namespace NUMINAMATH_CALUDE_ellipse_equation_l2446_244616

/-- Given an ellipse centered at the origin with eccentricity e = 1/2, 
    and one of its foci coinciding with the focus of the parabola y^2 = -4x,
    prove that the equation of this ellipse is x^2/4 + y^2/3 = 1 -/
theorem ellipse_equation (e : ℝ) (f : ℝ × ℝ) :
  e = (1 : ℝ) / 2 →
  f = (-1, 0) →
  ∀ (x y : ℝ), (x^2 / 4 + y^2 / 3 = 1) ↔ 
    (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
      x^2 / a^2 + y^2 / b^2 = 1 ∧
      e = (f.1^2 + f.2^2).sqrt / a) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2446_244616


namespace NUMINAMATH_CALUDE_distance_calculation_l2446_244601

theorem distance_calculation (A B C D : ℝ) 
  (h1 : A = 350)
  (h2 : A + B = 600)
  (h3 : A + B + C + D = 1500)
  (h4 : D = 275) :
  C = 625 ∧ B + C = 875 := by
  sorry

end NUMINAMATH_CALUDE_distance_calculation_l2446_244601


namespace NUMINAMATH_CALUDE_toucan_count_l2446_244660

/-- The number of toucans on the first limb initially -/
def initial_first_limb : ℕ := 3

/-- The number of toucans on the second limb initially -/
def initial_second_limb : ℕ := 4

/-- The number of toucans that join the first group -/
def join_first_limb : ℕ := 2

/-- The number of toucans that join the second group -/
def join_second_limb : ℕ := 3

/-- The total number of toucans after all changes -/
def total_toucans : ℕ := initial_first_limb + initial_second_limb + join_first_limb + join_second_limb

theorem toucan_count : total_toucans = 12 := by
  sorry

end NUMINAMATH_CALUDE_toucan_count_l2446_244660


namespace NUMINAMATH_CALUDE_jill_jack_distance_difference_l2446_244693

/-- The side length of the inner square (Jack's path) in feet -/
def inner_side_length : ℕ := 300

/-- The width of the street in feet -/
def street_width : ℕ := 15

/-- The side length of the outer square (Jill's path) in feet -/
def outer_side_length : ℕ := inner_side_length + 2 * street_width

/-- The difference in distance run by Jill and Jack -/
def distance_difference : ℕ := 4 * outer_side_length - 4 * inner_side_length

theorem jill_jack_distance_difference : distance_difference = 120 := by
  sorry

end NUMINAMATH_CALUDE_jill_jack_distance_difference_l2446_244693


namespace NUMINAMATH_CALUDE_function_inequality_function_inequality_bounded_l2446_244644

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  a * Real.sin x - (1/2) * Real.cos (2*x) + a - 3/a + 1/2

theorem function_inequality (a : ℝ) (h : a ≠ 0) :
  (∀ x, f a x ≤ 0) ↔ (0 < a ∧ a ≤ 1) :=
sorry

theorem function_inequality_bounded (a : ℝ) (h : a ≥ 2) :
  (∃ x, f a x ≤ 0) ↔ a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_function_inequality_bounded_l2446_244644
