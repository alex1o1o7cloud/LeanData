import Mathlib

namespace NUMINAMATH_CALUDE_average_wage_calculation_l299_29945

/-- Calculates the average wage per day paid by a contractor given the number of workers and their wages. -/
theorem average_wage_calculation
  (male_workers female_workers child_workers : ℕ)
  (male_wage female_wage child_wage : ℚ)
  (h_male : male_workers = 20)
  (h_female : female_workers = 15)
  (h_child : child_workers = 5)
  (h_male_wage : male_wage = 35)
  (h_female_wage : female_wage = 20)
  (h_child_wage : child_wage = 8) :
  (male_workers * male_wage + female_workers * female_wage + child_workers * child_wage) /
  (male_workers + female_workers + child_workers : ℚ) = 26 := by
  sorry

end NUMINAMATH_CALUDE_average_wage_calculation_l299_29945


namespace NUMINAMATH_CALUDE_union_of_sets_l299_29966

theorem union_of_sets : 
  let M : Set Int := {1, 0, -1}
  let N : Set Int := {1, 2}
  M ∪ N = {1, 2, 0, -1} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l299_29966


namespace NUMINAMATH_CALUDE_zoo_visitors_per_hour_l299_29993

/-- The number of hours the zoo is open in one day -/
def zoo_hours : ℕ := 8

/-- The percentage of total visitors who go to the gorilla exhibit -/
def gorilla_exhibit_percentage : ℚ := 80 / 100

/-- The number of visitors who go to the gorilla exhibit in one day -/
def gorilla_exhibit_visitors : ℕ := 320

/-- The number of new visitors entering the zoo every hour -/
def new_visitors_per_hour : ℕ := 50

theorem zoo_visitors_per_hour :
  new_visitors_per_hour = (gorilla_exhibit_visitors : ℚ) / gorilla_exhibit_percentage / zoo_hours := by
  sorry

end NUMINAMATH_CALUDE_zoo_visitors_per_hour_l299_29993


namespace NUMINAMATH_CALUDE_f_value_at_log_half_24_l299_29939

/-- An odd function with period 2 and specific definition on (0,1) -/
def f (x : ℝ) : ℝ :=
  sorry

theorem f_value_at_log_half_24 :
  ∀ (f : ℝ → ℝ),
  (∀ x, f (-x) = -f x) →  -- f is odd
  (∀ x, f (x + 1) = f (x - 1)) →  -- f has period 2
  (∀ x ∈ Set.Ioo 0 1, f x = 2^x - 2) →  -- definition on (0,1)
  f (Real.log 24 / Real.log (1/2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_log_half_24_l299_29939


namespace NUMINAMATH_CALUDE_exists_arrangement_with_more_than_five_holes_l299_29907

/-- Represents a strange ring, which is a circle with a square hole in the middle. -/
structure StrangeRing where
  circle_radius : ℝ
  square_side : ℝ
  center : ℝ × ℝ
  h_square_fits : square_side ≤ 2 * circle_radius

/-- Represents an arrangement of two strange rings on a table. -/
structure StrangeRingArrangement where
  ring1 : StrangeRing
  ring2 : StrangeRing
  placement : ℝ × ℝ  -- Relative placement of ring2 with respect to ring1

/-- Counts the number of holes in a given arrangement of strange rings. -/
def count_holes (arrangement : StrangeRingArrangement) : ℕ :=
  sorry

/-- Theorem stating that there exists an arrangement of two strange rings
    that results in more than 5 holes. -/
theorem exists_arrangement_with_more_than_five_holes :
  ∃ (arrangement : StrangeRingArrangement), count_holes arrangement > 5 :=
sorry

end NUMINAMATH_CALUDE_exists_arrangement_with_more_than_five_holes_l299_29907


namespace NUMINAMATH_CALUDE_six_balls_two_boxes_l299_29968

/-- The number of ways to distribute n distinguishable balls into 2 indistinguishable boxes -/
def distributionWays (n : ℕ) : ℕ :=
  (2^n) / 2 - 1

/-- Theorem: There are 31 ways to distribute 6 distinguishable balls into 2 indistinguishable boxes -/
theorem six_balls_two_boxes : distributionWays 6 = 31 := by
  sorry

end NUMINAMATH_CALUDE_six_balls_two_boxes_l299_29968


namespace NUMINAMATH_CALUDE_particle_max_height_l299_29917

/-- The height function of the particle -/
def h (t : ℝ) : ℝ := 180 * t - 18 * t^2

/-- The maximum height reached by the particle -/
def max_height : ℝ := 450

/-- Theorem stating that the maximum height reached by the particle is 450 meters -/
theorem particle_max_height :
  ∃ t : ℝ, h t = max_height ∧ ∀ s : ℝ, h s ≤ h t :=
sorry

end NUMINAMATH_CALUDE_particle_max_height_l299_29917


namespace NUMINAMATH_CALUDE_rational_cube_sum_ratio_l299_29938

theorem rational_cube_sum_ratio (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_sum_ratio_l299_29938


namespace NUMINAMATH_CALUDE_largest_common_term_l299_29948

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

def is_in_first_sequence (x : ℤ) : Prop :=
  ∃ n : ℕ, x = arithmetic_sequence 3 8 n

def is_in_second_sequence (x : ℤ) : Prop :=
  ∃ n : ℕ, x = arithmetic_sequence 5 9 n

theorem largest_common_term :
  ∀ x : ℤ, 1 ≤ x ∧ x ≤ 200 ∧ is_in_first_sequence x ∧ is_in_second_sequence x →
  x ≤ 131 ∧ is_in_first_sequence 131 ∧ is_in_second_sequence 131 :=
by sorry

end NUMINAMATH_CALUDE_largest_common_term_l299_29948


namespace NUMINAMATH_CALUDE_set_equality_and_range_of_a_l299_29905

-- Define the sets
def M : Set ℝ := {x | (x + 3)^2 ≤ 0}
def N : Set ℝ := {x | x^2 + x - 6 = 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 5 - a}
def A : Set ℝ := (Set.univ \ M) ∩ N

-- State the theorem
theorem set_equality_and_range_of_a :
  (A = {2}) ∧
  (∀ a : ℝ, (A ∪ B a = A) ↔ a ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_set_equality_and_range_of_a_l299_29905


namespace NUMINAMATH_CALUDE_half_of_three_fifths_of_120_l299_29981

theorem half_of_three_fifths_of_120 : (1/2 : ℚ) * ((3/5 : ℚ) * 120) = 36 := by
  sorry

end NUMINAMATH_CALUDE_half_of_three_fifths_of_120_l299_29981


namespace NUMINAMATH_CALUDE_mork_mindy_tax_rate_l299_29942

/-- Calculates the combined tax rate for Mork and Mindy -/
theorem mork_mindy_tax_rate (mork_income : ℝ) (mork_tax_rate : ℝ) (mindy_tax_rate : ℝ) :
  mork_tax_rate = 0.45 →
  mindy_tax_rate = 0.15 →
  let mindy_income := 4 * mork_income
  let combined_tax := mork_tax_rate * mork_income + mindy_tax_rate * mindy_income
  let combined_income := mork_income + mindy_income
  combined_tax / combined_income = 0.21 :=
by
  sorry

#check mork_mindy_tax_rate

end NUMINAMATH_CALUDE_mork_mindy_tax_rate_l299_29942


namespace NUMINAMATH_CALUDE_lindas_painting_area_l299_29985

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  width : ℝ
  length : ℝ
  height : ℝ

/-- Represents a rectangular opening in a wall -/
structure Opening where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangular surface -/
def rectangleArea (width : ℝ) (height : ℝ) : ℝ := width * height

/-- Calculates the total wall area of a room -/
def totalWallArea (room : RoomDimensions) : ℝ :=
  2 * (room.width * room.height + room.length * room.height)

/-- Calculates the area of an opening -/
def openingArea (opening : Opening) : ℝ :=
  rectangleArea opening.width opening.height

/-- Represents Linda's bedroom -/
def lindasBedroom : RoomDimensions := {
  width := 20,
  length := 20,
  height := 8
}

/-- Represents the doorway in Linda's bedroom -/
def doorway : Opening := {
  width := 3,
  height := 7
}

/-- Represents the window in Linda's bedroom -/
def window : Opening := {
  width := 6,
  height := 4
}

/-- Represents the closet doorway in Linda's bedroom -/
def closetDoorway : Opening := {
  width := 5,
  height := 7
}

/-- Theorem stating the total area of wall space Linda will have to paint -/
theorem lindas_painting_area :
  totalWallArea lindasBedroom -
  (openingArea doorway + openingArea window + openingArea closetDoorway) = 560 := by
  sorry

end NUMINAMATH_CALUDE_lindas_painting_area_l299_29985


namespace NUMINAMATH_CALUDE_b_plus_c_equals_seven_l299_29987

theorem b_plus_c_equals_seven (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : c + d = 5) 
  (h3 : a + d = 2) : 
  b + c = 7 := by sorry

end NUMINAMATH_CALUDE_b_plus_c_equals_seven_l299_29987


namespace NUMINAMATH_CALUDE_range_of_m_l299_29973

-- Define the propositions p and q
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2

def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the theorem
theorem range_of_m (m : ℝ) :
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x m)) →
  (∃ x, ¬(p x) ∧ (q x m)) →
  m ≥ 9 :=
by
  sorry

-- Define the final result
def result : Set ℝ := {m | m ≥ 9}

end NUMINAMATH_CALUDE_range_of_m_l299_29973


namespace NUMINAMATH_CALUDE_factors_of_N_l299_29997

/-- The number of natural-number factors of N, where N = 2^4 * 3^3 * 5^2 * 7^2 -/
def num_factors (N : Nat) : Nat :=
  if N = 2^4 * 3^3 * 5^2 * 7^2 then 180 else 0

/-- Theorem stating that the number of natural-number factors of N is 180 -/
theorem factors_of_N :
  ∃ N : Nat, N = 2^4 * 3^3 * 5^2 * 7^2 ∧ num_factors N = 180 :=
by
  sorry

#check factors_of_N

end NUMINAMATH_CALUDE_factors_of_N_l299_29997


namespace NUMINAMATH_CALUDE_dividend_calculation_l299_29964

theorem dividend_calculation (divisor quotient remainder : ℝ) 
  (h1 : divisor = 127.5)
  (h2 : quotient = 238)
  (h3 : remainder = 53.2) :
  divisor * quotient + remainder = 30398.2 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l299_29964


namespace NUMINAMATH_CALUDE_negation_equivalence_l299_29980

theorem negation_equivalence :
  (¬ ∃ a ∈ Set.Icc (-1 : ℝ) 2, ∃ x : ℝ, a * x^2 + 1 < 0) ↔
  (∀ a ∈ Set.Icc (-1 : ℝ) 2, ∀ x : ℝ, a * x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l299_29980


namespace NUMINAMATH_CALUDE_lune_area_specific_case_l299_29958

/-- Represents a semicircle with a given diameter -/
structure Semicircle where
  diameter : ℝ
  diameter_pos : diameter > 0

/-- Represents a lune formed by two semicircles -/
structure Lune where
  upper : Semicircle
  lower : Semicircle
  upper_on_lower : upper.diameter < lower.diameter

/-- Calculates the area of a lune -/
noncomputable def lune_area (l : Lune) : ℝ :=
  sorry

theorem lune_area_specific_case :
  let upper := Semicircle.mk 3 (by norm_num)
  let lower := Semicircle.mk 4 (by norm_num)
  let l := Lune.mk upper lower (by norm_num)
  lune_area l = (9 * Real.sqrt 3) / 4 - (55 / 24) * Real.pi :=
sorry

end NUMINAMATH_CALUDE_lune_area_specific_case_l299_29958


namespace NUMINAMATH_CALUDE_height_minus_twice_radius_l299_29915

/-- An equilateral triangle with an inscribed circle -/
structure EquilateralTriangleWithInscribedCircle where
  -- Side length of the equilateral triangle
  side_length : ℝ
  -- Radius of the inscribed circle
  radius : ℝ
  -- The triangle is equilateral with the given side length
  is_equilateral : side_length > 0
  -- The circle is inscribed in the triangle
  is_inscribed : radius > 0 ∧ radius < side_length / 2

/-- The theorem to be proved -/
theorem height_minus_twice_radius
  (triangle : EquilateralTriangleWithInscribedCircle)
  (h_side : triangle.side_length = 24) :
  let height := (Real.sqrt 3 / 2) * triangle.side_length
  height - 2 * triangle.radius = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_height_minus_twice_radius_l299_29915


namespace NUMINAMATH_CALUDE_intersection_with_complement_l299_29955

def U : Finset ℕ := {0, 1, 2, 3, 4}
def A : Finset ℕ := {1, 2, 3}
def B : Finset ℕ := {2, 0}

theorem intersection_with_complement : A ∩ (U \ B) = {1, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l299_29955


namespace NUMINAMATH_CALUDE_inequality_proof_l299_29956

theorem inequality_proof (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_ineq : ∀ x, f x > deriv f x) (a b : ℝ) (hab : a > b) :
  Real.exp a * f b > Real.exp b * f a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l299_29956


namespace NUMINAMATH_CALUDE_sin_tan_inequality_l299_29979

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f x = f (-x))
variable (h_mono : ∀ x y, -1 ≤ x ∧ x < y ∧ y ≤ 0 → f x < f y)

-- State the theorem
theorem sin_tan_inequality :
  f (Real.sin (π / 12)) > f (Real.tan (π / 12)) := by sorry

end NUMINAMATH_CALUDE_sin_tan_inequality_l299_29979


namespace NUMINAMATH_CALUDE_largest_number_l299_29998

theorem largest_number (a b c d : ℝ) (h1 : a = -3) (h2 : b = 0) (h3 : c = Real.sqrt 5) (h4 : d = 2) :
  c = max a (max b (max c d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l299_29998


namespace NUMINAMATH_CALUDE_rhombus_area_calculation_l299_29927

/-- Represents a rhombus -/
structure Rhombus where
  side_length : ℝ
  area : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  ABCD : Rhombus
  BAFC : Rhombus
  AF_parallel_BD : Prop

/-- Main theorem -/
theorem rhombus_area_calculation (setup : ProblemSetup) 
  (h1 : setup.ABCD.side_length = 13)
  (h2 : setup.BAFC.area = 65)
  : setup.ABCD.area = 120 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_calculation_l299_29927


namespace NUMINAMATH_CALUDE_intersection_solution_set_l299_29921

theorem intersection_solution_set (a b : ℝ) : 
  (∀ x, x^2 + a*x + b < 0 ↔ (x^2 - 2*x - 3 < 0 ∧ x^2 + x - 6 < 0)) →
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_intersection_solution_set_l299_29921


namespace NUMINAMATH_CALUDE_polynomial_equality_l299_29978

theorem polynomial_equality (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b = (x - 1)*(x + 4)) → a = 3 ∧ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l299_29978


namespace NUMINAMATH_CALUDE_limit_of_sequence_l299_29976

def a (n : ℕ) : ℚ := (3 * n - 2) / (2 * n - 1)

theorem limit_of_sequence : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/2| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_sequence_l299_29976


namespace NUMINAMATH_CALUDE_d_eq_4_sufficient_not_necessary_l299_29936

/-- An arithmetic sequence with first term 2 and common difference d -/
def arithmetic_seq (n : ℕ) (d : ℝ) : ℝ := 2 + (n - 1) * d

/-- Condition for a_1, a_2, a_5 to form a geometric sequence -/
def is_geometric (d : ℝ) : Prop :=
  (arithmetic_seq 2 d)^2 = (arithmetic_seq 1 d) * (arithmetic_seq 5 d)

/-- d = 4 is a sufficient but not necessary condition for a_1, a_2, a_5 to form a geometric sequence -/
theorem d_eq_4_sufficient_not_necessary :
  (∀ d : ℝ, d = 4 → is_geometric d) ∧
  ¬(∀ d : ℝ, is_geometric d → d = 4) :=
sorry

end NUMINAMATH_CALUDE_d_eq_4_sufficient_not_necessary_l299_29936


namespace NUMINAMATH_CALUDE_sum_of_first_10_terms_l299_29950

-- Define the sequence sum function
def S (n : ℕ) : ℕ := n^2 - 4*n + 1

-- Theorem statement
theorem sum_of_first_10_terms : S 10 = 61 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_10_terms_l299_29950


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l299_29982

theorem ceiling_floor_sum : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l299_29982


namespace NUMINAMATH_CALUDE_fourth_quadrant_properties_l299_29904

open Real

-- Define the fourth quadrant
def fourth_quadrant (α : ℝ) : Prop := 3 * π / 2 < α ∧ α < 2 * π

theorem fourth_quadrant_properties (α : ℝ) (h : fourth_quadrant α) :
  (∃ α, fourth_quadrant α ∧ cos (2 * α) > 0) ∧
  (∀ α, fourth_quadrant α → sin (2 * α) < 0) ∧
  (¬ ∃ α, fourth_quadrant α ∧ tan (α / 2) < 0) ∧
  (∃ α, fourth_quadrant α ∧ cos (α / 2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_fourth_quadrant_properties_l299_29904


namespace NUMINAMATH_CALUDE_sara_quarters_proof_l299_29970

/-- The number of quarters Sara's dad gave her -/
def quarters_from_dad (initial_quarters final_quarters : ℕ) : ℕ :=
  final_quarters - initial_quarters

/-- Proof that Sara's dad gave her 49 quarters -/
theorem sara_quarters_proof (initial_quarters final_quarters : ℕ) 
  (h1 : initial_quarters = 21)
  (h2 : final_quarters = 70) :
  quarters_from_dad initial_quarters final_quarters = 49 := by
  sorry

end NUMINAMATH_CALUDE_sara_quarters_proof_l299_29970


namespace NUMINAMATH_CALUDE_multiply_469158_and_9999_l299_29946

theorem multiply_469158_and_9999 : 469158 * 9999 = 4691176842 := by
  sorry

end NUMINAMATH_CALUDE_multiply_469158_and_9999_l299_29946


namespace NUMINAMATH_CALUDE_intersection_A_C_R_B_range_of_a_l299_29918

-- Define sets A, B, and C
def A : Set ℝ := {x | x^2 - x - 12 < 0}
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

-- Define the relative complement of B with respect to ℝ
def C_R_B : Set ℝ := {x | x ∉ B}

-- Theorem 1: A ∩ (C_R B) = {x | -3 < x ≤ 2}
theorem intersection_A_C_R_B : A ∩ C_R_B = {x : ℝ | -3 < x ∧ x ≤ 2} := by
  sorry

-- Theorem 2: If C ⊇ (A ∩ B), then 4/3 ≤ a ≤ 2
theorem range_of_a (a : ℝ) (h : a ≠ 0) :
  (C a ⊇ (A ∩ B)) → (4/3 ≤ a ∧ a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_C_R_B_range_of_a_l299_29918


namespace NUMINAMATH_CALUDE_point_not_in_second_quadrant_l299_29935

theorem point_not_in_second_quadrant : ¬ ((-Real.sqrt 2 < 0) ∧ (-Real.sqrt 3 > 0)) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_second_quadrant_l299_29935


namespace NUMINAMATH_CALUDE_circle_equation_given_conditions_l299_29924

/-- A circle C in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle given its center and radius -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

/-- A circle is tangent to the y-axis if its center's x-coordinate equals its radius -/
def tangent_to_y_axis (c : Circle) : Prop :=
  c.center.1 = c.radius ∨ c.center.1 = -c.radius

/-- A point (x, y) lies on the line x - 3y = 0 -/
def on_line (x y : ℝ) : Prop :=
  x - 3*y = 0

theorem circle_equation_given_conditions :
  ∀ (C : Circle),
    tangent_to_y_axis C →
    C.radius = 4 →
    on_line C.center.1 C.center.2 →
    ∀ (x y : ℝ),
      circle_equation C x y ↔ 
        ((x - 4)^2 + (y - 4/3)^2 = 16 ∨ (x + 4)^2 + (y + 4/3)^2 = 16) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_given_conditions_l299_29924


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l299_29906

theorem complex_fraction_simplification :
  (5 + 7 * Complex.I) / (2 + 3 * Complex.I) = 31/13 - 1/13 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l299_29906


namespace NUMINAMATH_CALUDE_current_rate_l299_29901

/-- Proves that given a man who can row 3.3 km/hr in still water, and it takes him twice as long
    to row upstream as to row downstream, the rate of the current is 1.1 km/hr. -/
theorem current_rate (still_water_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) :
  still_water_speed = 3.3 ∧ upstream_time = 2 * downstream_time →
  ∃ current_rate : ℝ,
    current_rate = 1.1 ∧
    (still_water_speed + current_rate) * downstream_time =
    (still_water_speed - current_rate) * upstream_time :=
by sorry

end NUMINAMATH_CALUDE_current_rate_l299_29901


namespace NUMINAMATH_CALUDE_quadratic_function_range_l299_29943

/-- A quadratic function with two distinct zeros -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  has_distinct_zeros : ∃ (x y : ℝ), x ≠ y ∧ x^2 + a*x + b = 0 ∧ y^2 + a*y + b = 0

/-- Four distinct roots in arithmetic progression -/
structure FourRoots where
  x₁ : ℝ
  x₂ : ℝ
  x₃ : ℝ
  x₄ : ℝ
  distinct : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄
  arithmetic : ∃ (d : ℝ), x₂ - x₁ = d ∧ x₃ - x₂ = d ∧ x₄ - x₃ = d

/-- The main theorem -/
theorem quadratic_function_range (f : QuadraticFunction) (roots : FourRoots) 
  (h : ∀ x, (x^2 + 2*x - 1)^2 + f.a*(x^2 + 2*x - 1) + f.b = 0 ↔ 
           x = roots.x₁ ∨ x = roots.x₂ ∨ x = roots.x₃ ∨ x = roots.x₄) :
  ∀ x, x ≤ 25/9 ∧ (∃ y, f.a - f.b = y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l299_29943


namespace NUMINAMATH_CALUDE_incorrect_subset_l299_29967

-- Define the sets
def set1 : Set ℕ := {1, 2, 3}
def set2 : Set ℕ := {1, 2}

-- Theorem statement
theorem incorrect_subset : ¬(set1 ⊆ set2) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_subset_l299_29967


namespace NUMINAMATH_CALUDE_circumscribed_sphere_radius_eq_side_length_l299_29931

/-- A regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  /-- The side length of the base -/
  baseSideLength : ℝ
  /-- The height of the pyramid -/
  height : ℝ

/-- The radius of a sphere circumscribed around a regular hexagonal pyramid -/
def circumscribedSphereRadius (p : RegularHexagonalPyramid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed sphere of a regular hexagonal pyramid
    with base side length a and height a is equal to a -/
theorem circumscribed_sphere_radius_eq_side_length
    (p : RegularHexagonalPyramid)
    (h1 : p.baseSideLength = p.height)
    (h2 : p.baseSideLength > 0) :
    circumscribedSphereRadius p = p.baseSideLength :=
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_radius_eq_side_length_l299_29931


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l299_29947

theorem closest_integer_to_cube_root : ∃ (n : ℤ), 
  n = 8 ∧ ∀ (m : ℤ), |m - (5^3 + 7^3)^(1/3)| ≥ |n - (5^3 + 7^3)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l299_29947


namespace NUMINAMATH_CALUDE_visible_percentage_for_given_prism_and_film_l299_29965

/-- Represents a regular triangular prism -/
structure RegularTriangularPrism where
  base_edge : ℝ
  height : ℝ

/-- Represents a checkerboard film -/
structure CheckerboardFilm where
  cell_size : ℝ

/-- Calculates the visible percentage of a prism's lateral surface when wrapped with a film -/
def visible_percentage (prism : RegularTriangularPrism) (film : CheckerboardFilm) : ℝ :=
  sorry

/-- Theorem stating the visible percentage for the given prism and film -/
theorem visible_percentage_for_given_prism_and_film :
  let prism := RegularTriangularPrism.mk 3.2 5
  let film := CheckerboardFilm.mk 1
  visible_percentage prism film = 28.75 := by
  sorry

end NUMINAMATH_CALUDE_visible_percentage_for_given_prism_and_film_l299_29965


namespace NUMINAMATH_CALUDE_decimal_equivalent_of_half_squared_l299_29903

theorem decimal_equivalent_of_half_squared : (1 / 2 : ℚ) ^ 2 = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_equivalent_of_half_squared_l299_29903


namespace NUMINAMATH_CALUDE_cary_earns_five_per_lawn_l299_29989

/-- The amount earned per lawn mowed --/
def amount_per_lawn (cost_of_shoes amount_saved lawns_per_weekend num_weekends : ℚ) : ℚ :=
  (cost_of_shoes - amount_saved) / (lawns_per_weekend * num_weekends)

/-- Theorem: Cary earns $5 per lawn mowed --/
theorem cary_earns_five_per_lawn :
  amount_per_lawn 120 30 3 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cary_earns_five_per_lawn_l299_29989


namespace NUMINAMATH_CALUDE_fathers_age_l299_29971

theorem fathers_age (M F : ℕ) : 
  M = (2 : ℕ) * F / (5 : ℕ) →
  M + 6 = (F + 6) / (2 : ℕ) →
  F = 30 := by
sorry

end NUMINAMATH_CALUDE_fathers_age_l299_29971


namespace NUMINAMATH_CALUDE_total_money_end_is_3933_33_l299_29940

/-- Calculates the total money after splitting and increasing the remainder -/
def totalMoneyAtEnd (cecilMoney : ℚ) : ℚ :=
  let catherineMoney := 2 * cecilMoney - 250
  let carmelaMoney := 2 * cecilMoney + 50
  let averageMoney := (cecilMoney + catherineMoney + carmelaMoney) / 3
  let carlosMoney := averageMoney + 200
  let totalMoney := cecilMoney + catherineMoney + carmelaMoney + carlosMoney
  let splitAmount := totalMoney / 7
  let remainingAmount := totalMoney - (splitAmount * 7)
  let increase := remainingAmount * (5 / 100)
  totalMoney + increase

/-- Theorem stating that the total money at the end is $3933.33 -/
theorem total_money_end_is_3933_33 :
  totalMoneyAtEnd 600 = 3933.33 := by sorry

end NUMINAMATH_CALUDE_total_money_end_is_3933_33_l299_29940


namespace NUMINAMATH_CALUDE_exam_correct_answers_l299_29949

/-- Given an exam with the following conditions:
  * Total number of questions is 150
  * Correct answers score 4 marks
  * Wrong answers score -2 marks
  * Total score is 420 marks
  Prove that the number of correct answers is 120 -/
theorem exam_correct_answers 
  (total_questions : ℕ) 
  (correct_score wrong_score total_score : ℤ) 
  (h1 : total_questions = 150)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -2)
  (h4 : total_score = 420) : 
  ∃ (correct_answers : ℕ), 
    correct_answers = 120 ∧ 
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score :=
by
  sorry

end NUMINAMATH_CALUDE_exam_correct_answers_l299_29949


namespace NUMINAMATH_CALUDE_shaded_to_white_ratio_l299_29925

/-- A square divided into smaller squares where the vertices of inner squares 
    are at the midpoints of the sides of the outer squares -/
structure NestedSquares :=
  (side : ℝ)
  (is_positive : side > 0)

/-- The area of the shaded part in a NestedSquares structure -/
def shaded_area (s : NestedSquares) : ℝ := sorry

/-- The area of the white part in a NestedSquares structure -/
def white_area (s : NestedSquares) : ℝ := sorry

/-- Theorem stating that the ratio of shaded area to white area is 5:3 -/
theorem shaded_to_white_ratio (s : NestedSquares) : 
  shaded_area s / white_area s = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_shaded_to_white_ratio_l299_29925


namespace NUMINAMATH_CALUDE_quaternary_201_equals_33_l299_29923

/-- Converts a quaternary (base-4) number to its decimal (base-10) equivalent -/
def quaternary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

theorem quaternary_201_equals_33 :
  quaternary_to_decimal [1, 0, 2] = 33 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_201_equals_33_l299_29923


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l299_29913

/-- The area of a rectangle given its width and height -/
def rectangleArea (width height : ℕ) : ℕ := width * height

/-- The sum of the areas of four rectangles -/
def totalArea (w1 h1 w2 h2 w3 h3 w4 h4 : ℕ) : ℕ :=
  rectangleArea w1 h1 + rectangleArea w2 h2 + rectangleArea w3 h3 + rectangleArea w4 h4

/-- Theorem stating that the sum of the areas of four specific rectangles is 56 -/
theorem sum_of_rectangle_areas :
  totalArea 7 6 3 2 3 1 5 1 = 56 := by
  sorry

#eval totalArea 7 6 3 2 3 1 5 1

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l299_29913


namespace NUMINAMATH_CALUDE_sqrt_n_factorial_inequality_l299_29996

theorem sqrt_n_factorial_inequality (n : ℕ) (hn : n > 0) :
  Real.sqrt n < (n.factorial : ℝ) ^ (1 / n : ℝ) ∧ (n.factorial : ℝ) ^ (1 / n : ℝ) < (n + 1 : ℝ) / 2 := by
  sorry

#check sqrt_n_factorial_inequality

end NUMINAMATH_CALUDE_sqrt_n_factorial_inequality_l299_29996


namespace NUMINAMATH_CALUDE_equal_distance_to_axes_l299_29928

theorem equal_distance_to_axes (m : ℝ) : 
  let P : ℝ × ℝ := (3*m + 1, 2*m - 5)
  (|P.1| = |P.2|) ↔ (m = -6 ∨ m = 4/5) := by
  sorry

end NUMINAMATH_CALUDE_equal_distance_to_axes_l299_29928


namespace NUMINAMATH_CALUDE_ball_box_distribution_l299_29922

def num_balls : ℕ := 5
def num_boxes : ℕ := 5

/-- The number of ways to put all balls into boxes -/
def total_ways : ℕ := num_boxes ^ num_balls

/-- The number of ways to put balls into boxes with exactly one box left empty -/
def one_empty : ℕ := Nat.choose num_boxes 2 * Nat.factorial (num_balls - 1)

/-- The number of ways to put balls into boxes with exactly two boxes left empty -/
def two_empty : ℕ := 
  (Nat.choose num_boxes 2 * Nat.choose 3 2 * Nat.factorial (num_balls - 2) +
   Nat.choose num_boxes 3 * Nat.choose 2 1 * Nat.factorial (num_balls - 2)) * 
  Nat.factorial num_boxes / (Nat.factorial 2)

theorem ball_box_distribution :
  total_ways = 3125 ∧ one_empty = 1200 ∧ two_empty = 1500 := by
  sorry

end NUMINAMATH_CALUDE_ball_box_distribution_l299_29922


namespace NUMINAMATH_CALUDE_store_refusal_illegal_l299_29951

/-- Represents a banknote --/
structure Banknote where
  damaged : Bool
  torn : Bool

/-- Represents the store's action --/
inductive StoreAction
  | Accept
  | Refuse

/-- Defines what constitutes legal tender in Russia --/
def is_legal_tender (b : Banknote) : Bool :=
  b.damaged && b.torn

/-- Determines if a store's action is legal based on the banknote --/
def is_legal_action (b : Banknote) (a : StoreAction) : Prop :=
  is_legal_tender b → a = StoreAction.Accept

/-- The main theorem stating that refusing a torn banknote is illegal --/
theorem store_refusal_illegal (b : Banknote) (h1 : b.damaged) (h2 : b.torn) :
  ¬(is_legal_action b StoreAction.Refuse) := by
  sorry


end NUMINAMATH_CALUDE_store_refusal_illegal_l299_29951


namespace NUMINAMATH_CALUDE_correct_equation_for_x_equals_one_l299_29953

theorem correct_equation_for_x_equals_one :
  (2 * 1 + 2 = 4) ∧
  ¬(1 + 1 = 0) ∧
  ¬(3 * 1 = -3) ∧
  ¬(1 - 1 = 2) := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_for_x_equals_one_l299_29953


namespace NUMINAMATH_CALUDE_lucy_grocery_problem_l299_29912

/-- Lucy's grocery shopping problem -/
theorem lucy_grocery_problem (total_packs cookies_packs noodles_packs : ℕ) :
  total_packs = 28 →
  cookies_packs = 12 →
  total_packs = cookies_packs + noodles_packs →
  noodles_packs = 16 := by
  sorry

end NUMINAMATH_CALUDE_lucy_grocery_problem_l299_29912


namespace NUMINAMATH_CALUDE_orange_crates_pigeonhole_l299_29975

theorem orange_crates_pigeonhole (total_crates : ℕ) (min_oranges max_oranges : ℕ) :
  total_crates = 200 →
  min_oranges = 100 →
  max_oranges = 130 →
  ∃ (n : ℕ), n ≥ 7 ∧ 
    ∃ (k : ℕ), min_oranges ≤ k ∧ k ≤ max_oranges ∧
      (∃ (crates : Finset (Fin total_crates)), crates.card = n ∧ 
        ∀ c ∈ crates, ∃ f : Fin total_crates → ℕ, f c = k) :=
by sorry

end NUMINAMATH_CALUDE_orange_crates_pigeonhole_l299_29975


namespace NUMINAMATH_CALUDE_expenditure_estimate_l299_29932

/-- Represents the annual income in billions of yuan -/
def annual_income : ℝ := 15

/-- Represents the relationship between income x and expenditure y -/
def expenditure_function (x : ℝ) : ℝ := 0.8 * x + 0.1

/-- The estimated annual expenditure based on the given income and relationship -/
def estimated_expenditure : ℝ := expenditure_function annual_income

theorem expenditure_estimate : estimated_expenditure = 12.1 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_estimate_l299_29932


namespace NUMINAMATH_CALUDE_expand_product_l299_29926

theorem expand_product (x : ℝ) : (x + 3) * (x - 4) * (x + 1) = x^3 - 13*x - 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l299_29926


namespace NUMINAMATH_CALUDE_complex_expression_equality_l299_29930

theorem complex_expression_equality : -(-1 - (-2*(-3-4) - 5 - 6*(-7-80))) - 9 = 523 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l299_29930


namespace NUMINAMATH_CALUDE_sheep_ratio_l299_29900

theorem sheep_ratio (total : ℕ) (beth_sheep : ℕ) (h1 : total = 608) (h2 : beth_sheep = 76) :
  (total - beth_sheep) / beth_sheep = 133 / 19 := by
  sorry

end NUMINAMATH_CALUDE_sheep_ratio_l299_29900


namespace NUMINAMATH_CALUDE_excel_manufacturing_company_women_percentage_l299_29941

theorem excel_manufacturing_company_women_percentage
  (total_employees : ℕ)
  (male_percentage : Real)
  (union_percentage : Real)
  (non_union_women_percentage : Real)
  (h1 : male_percentage = 0.46)
  (h2 : union_percentage = 0.60)
  (h3 : non_union_women_percentage = 0.90) :
  non_union_women_percentage = 0.90 := by
sorry

end NUMINAMATH_CALUDE_excel_manufacturing_company_women_percentage_l299_29941


namespace NUMINAMATH_CALUDE_cube_volume_equals_surface_area_l299_29929

/-- For a cube with side length s, if the volume is equal to the surface area, then s = 6. -/
theorem cube_volume_equals_surface_area (s : ℝ) (h : s > 0) :
  s^3 = 6 * s^2 → s = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_equals_surface_area_l299_29929


namespace NUMINAMATH_CALUDE_isosceles_triangle_attachment_l299_29992

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Checks if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop := sorry

/-- Checks if two triangles share a common side -/
def shareCommonSide (t1 t2 : Triangle) : Prop := sorry

/-- Checks if two triangles do not overlap -/
def noOverlap (t1 t2 : Triangle) : Prop := sorry

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Combines two triangles into a new triangle -/
def combineTriangles (t1 t2 : Triangle) : Triangle := sorry

theorem isosceles_triangle_attachment (t : Triangle) : 
  isRightTriangle t → 
  ∃ t2 : Triangle, 
    shareCommonSide t t2 ∧ 
    noOverlap t t2 ∧ 
    isIsosceles (combineTriangles t t2) := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_attachment_l299_29992


namespace NUMINAMATH_CALUDE_license_plate_count_l299_29983

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits (0-9) -/
def num_digits : ℕ := 10

/-- The total number of characters (letters + digits) -/
def num_chars : ℕ := num_letters + num_digits

/-- The number of possible license plates -/
def num_license_plates : ℕ := num_letters * num_chars * 1 * num_digits

theorem license_plate_count :
  num_license_plates = 9360 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_count_l299_29983


namespace NUMINAMATH_CALUDE_jenny_walking_distance_l299_29974

theorem jenny_walking_distance (ran_distance : ℝ) (extra_distance : ℝ) :
  ran_distance = 0.6 →
  extra_distance = 0.2 →
  ran_distance = (ran_distance - extra_distance) + extra_distance →
  ran_distance - extra_distance = 0.4 :=
by
  sorry

end NUMINAMATH_CALUDE_jenny_walking_distance_l299_29974


namespace NUMINAMATH_CALUDE_intersection_M_N_l299_29972

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x | x^2 + x ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l299_29972


namespace NUMINAMATH_CALUDE_taylor_family_reunion_tables_l299_29920

theorem taylor_family_reunion_tables (num_kids : ℕ) (num_adults : ℕ) (people_per_table : ℕ) : 
  num_kids = 45 → num_adults = 123 → people_per_table = 12 → 
  (num_kids + num_adults) / people_per_table = 14 := by
sorry

end NUMINAMATH_CALUDE_taylor_family_reunion_tables_l299_29920


namespace NUMINAMATH_CALUDE_solve_linear_system_l299_29984

theorem solve_linear_system (x y a : ℝ) 
  (eq1 : 4 * x + 3 * y = 1)
  (eq2 : a * x + (a - 1) * y = 3)
  (eq3 : x = y) :
  a = 11 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_system_l299_29984


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_845_l299_29954

theorem sqrt_expression_equals_sqrt_845 :
  Real.sqrt 80 - 3 * Real.sqrt 5 + Real.sqrt 720 / Real.sqrt 3 = Real.sqrt 845 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_sqrt_845_l299_29954


namespace NUMINAMATH_CALUDE_simplify_expression_l299_29909

theorem simplify_expression (a b : ℝ) :
  (-2 * a^2 * b)^3 / (-2 * a * b) * (1/3 * a^2 * b^3) = 4/3 * a^7 * b^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l299_29909


namespace NUMINAMATH_CALUDE_deer_distribution_l299_29902

theorem deer_distribution (a₁ : ℚ) (d : ℚ) :
  a₁ = 5/3 ∧ 
  5 * a₁ + (5 * 4)/2 * d = 5 →
  a₁ + 2*d = 1 :=
by sorry

end NUMINAMATH_CALUDE_deer_distribution_l299_29902


namespace NUMINAMATH_CALUDE_largest_power_of_five_factor_l299_29944

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_of_factorials : ℕ := factorial 102 + factorial 103 + factorial 104 + factorial 105

theorem largest_power_of_five_factor : 
  (∀ m : ℕ, 5^(24 + 1) ∣ sum_of_factorials → 5^m ∣ sum_of_factorials) ∧ 
  5^24 ∣ sum_of_factorials :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_five_factor_l299_29944


namespace NUMINAMATH_CALUDE_set_equality_l299_29999

theorem set_equality : 
  {z : ℤ | ∃ (x a : ℝ), z = x - a ∧ a - 1 ≤ x ∧ x ≤ a + 1} = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_l299_29999


namespace NUMINAMATH_CALUDE_cloth_selling_price_l299_29988

/-- Calculates the total selling price of cloth given the quantity, profit per meter, and cost price per meter. -/
def total_selling_price (quantity : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  quantity * (cost_price_per_meter + profit_per_meter)

/-- Proves that the total selling price of 85 meters of cloth with a profit of 20 Rs per meter and a cost price of 85 Rs per meter is 8925 Rs. -/
theorem cloth_selling_price :
  total_selling_price 85 20 85 = 8925 := by
  sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l299_29988


namespace NUMINAMATH_CALUDE_total_remaining_sand_first_truck_percentage_lost_second_truck_percentage_lost_third_truck_percentage_lost_fourth_truck_percentage_lost_l299_29957

/- Define the trucks and their properties -/
structure Truck where
  initial_sand : Float
  sand_lost : Float
  miles_driven : Float

/- Define the four trucks -/
def truck1 : Truck := { initial_sand := 4.1, sand_lost := 2.4, miles_driven := 20 }
def truck2 : Truck := { initial_sand := 5.7, sand_lost := 3.6, miles_driven := 15 }
def truck3 : Truck := { initial_sand := 8.2, sand_lost := 1.9, miles_driven := 25 }
def truck4 : Truck := { initial_sand := 10.5, sand_lost := 2.1, miles_driven := 30 }

/- Calculate remaining sand for a truck -/
def remaining_sand (t : Truck) : Float :=
  t.initial_sand - t.sand_lost

/- Calculate percentage of sand lost for a truck -/
def percentage_lost (t : Truck) : Float :=
  (t.sand_lost / t.initial_sand) * 100

/- Theorem: Total remaining sand is 18.5 pounds -/
theorem total_remaining_sand :
  remaining_sand truck1 + remaining_sand truck2 + remaining_sand truck3 + remaining_sand truck4 = 18.5 := by
  sorry

/- Theorem: Percentage of sand lost by the first truck is 58.54% -/
theorem first_truck_percentage_lost :
  percentage_lost truck1 = 58.54 := by
  sorry

/- Theorem: Percentage of sand lost by the second truck is 63.16% -/
theorem second_truck_percentage_lost :
  percentage_lost truck2 = 63.16 := by
  sorry

/- Theorem: Percentage of sand lost by the third truck is 23.17% -/
theorem third_truck_percentage_lost :
  percentage_lost truck3 = 23.17 := by
  sorry

/- Theorem: Percentage of sand lost by the fourth truck is 20% -/
theorem fourth_truck_percentage_lost :
  percentage_lost truck4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_remaining_sand_first_truck_percentage_lost_second_truck_percentage_lost_third_truck_percentage_lost_fourth_truck_percentage_lost_l299_29957


namespace NUMINAMATH_CALUDE_house_sale_profit_l299_29934

/-- Calculates the final profit for Mr. A after three house sales --/
theorem house_sale_profit (initial_value : ℝ) (profit1 profit2 profit3 : ℝ) : 
  initial_value = 120000 ∧ 
  profit1 = 0.2 ∧ 
  profit2 = -0.15 ∧ 
  profit3 = 0.05 → 
  let sale1 := initial_value * (1 + profit1)
  let sale2 := sale1 * (1 + profit2)
  let sale3 := sale2 * (1 + profit3)
  (sale1 - sale2) + (sale3 - sale2) = 27720 := by
  sorry

#check house_sale_profit

end NUMINAMATH_CALUDE_house_sale_profit_l299_29934


namespace NUMINAMATH_CALUDE_office_age_problem_l299_29990

theorem office_age_problem (total_persons : Nat) (group1_persons : Nat) (group2_persons : Nat)
  (total_avg_age : Nat) (group1_avg_age : Nat) (group2_avg_age : Nat)
  (h1 : total_persons = 19)
  (h2 : group1_persons = 5)
  (h3 : group2_persons = 9)
  (h4 : total_avg_age = 15)
  (h5 : group1_avg_age = 14)
  (h6 : group2_avg_age = 16) :
  total_persons * total_avg_age = 
    group1_persons * group1_avg_age + group2_persons * group2_avg_age + 71 := by
  sorry

#check office_age_problem

end NUMINAMATH_CALUDE_office_age_problem_l299_29990


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l299_29914

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 6

theorem quadratic_function_properties :
  (f (-1) = 0) ∧ 
  (f 3 = 0) ∧ 
  (f 1 = -8) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ -8) ∧
  (∀ x ∈ Set.Icc 0 3, f x ≤ 0) ∧
  (f 1 = -8) ∧
  (f 3 = 0) ∧
  (∀ x, f x ≥ 0 ↔ x ≤ -1 ∨ x ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l299_29914


namespace NUMINAMATH_CALUDE_quadratic_inequality_l299_29994

theorem quadratic_inequality (x : ℝ) : 
  10 * x^2 - 2 * x - 3 < 0 ↔ (1 - Real.sqrt 31) / 10 < x ∧ x < (1 + Real.sqrt 31) / 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l299_29994


namespace NUMINAMATH_CALUDE_xy_value_l299_29995

theorem xy_value (x y : ℝ) (h : x / 2 + 2 * y - 2 = Real.log x + Real.log y) : 
  x ^ y = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l299_29995


namespace NUMINAMATH_CALUDE_find_a_l299_29911

def f (x : ℝ) := 3 * (x - 1) + 2

theorem find_a : ∃ a : ℝ, f a = 5 ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_find_a_l299_29911


namespace NUMINAMATH_CALUDE_mercury_radius_scientific_notation_l299_29963

/-- Given a number in decimal notation, returns its scientific notation as a pair (a, n) where a is the coefficient and n is the exponent. -/
def toScientificNotation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem mercury_radius_scientific_notation :
  toScientificNotation 2440000 = (2.44, 6) :=
sorry

end NUMINAMATH_CALUDE_mercury_radius_scientific_notation_l299_29963


namespace NUMINAMATH_CALUDE_expression_evaluation_l299_29908

theorem expression_evaluation :
  let x : ℝ := 2 - Real.sqrt 3
  (7 + 4 * Real.sqrt 3) * x^2 - (2 + Real.sqrt 3) * x + Real.sqrt 3 = 2 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l299_29908


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l299_29962

theorem complex_sum_theorem (B Q R T : ℂ) : 
  B = 3 - 2*I ∧ Q = -5 + 3*I ∧ R = 2*I ∧ T = -1 + 2*I →
  B - Q + R + T = 7 - I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l299_29962


namespace NUMINAMATH_CALUDE_quinary_to_octal_conversion_polynomial_evaluation_l299_29910

-- Define the polynomial f(x)
def f (x : ℕ) : ℕ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

-- Define the quinary to decimal conversion function
def quinary_to_decimal (q : ℕ) : ℕ :=
  (q / 1000) * 5^3 + ((q / 100) % 10) * 5^2 + ((q / 10) % 10) * 5^1 + (q % 10)

-- Define the decimal to octal conversion function
def decimal_to_octal (d : ℕ) : ℕ :=
  (d / 64) * 100 + ((d / 8) % 8) * 10 + (d % 8)

theorem quinary_to_octal_conversion :
  decimal_to_octal (quinary_to_decimal 1234) = 302 := by sorry

theorem polynomial_evaluation :
  f 3 = 21324 := by sorry

end NUMINAMATH_CALUDE_quinary_to_octal_conversion_polynomial_evaluation_l299_29910


namespace NUMINAMATH_CALUDE_solution_set_a_eq_1_range_of_a_l299_29961

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |x + a|

-- Theorem 1: Solution set when a = 1
theorem solution_set_a_eq_1 :
  {x : ℝ | f 1 x < -2} = {x : ℝ | x > 3/2} := by sorry

-- Theorem 2: Range of 'a'
theorem range_of_a :
  {a : ℝ | ∀ x y : ℝ, -2 + f a y ≤ f a x ∧ f a x ≤ 2 + f a y} =
  {a : ℝ | -3 ≤ a ∧ a ≤ -1} := by sorry

end NUMINAMATH_CALUDE_solution_set_a_eq_1_range_of_a_l299_29961


namespace NUMINAMATH_CALUDE_circle_point_distance_range_l299_29991

/-- Given a circle C with equation (x-a)^2 + (y-a+2)^2 = 1 and a point A(0,2),
    if there exists a point M on C such that MA^2 + MO^2 = 10,
    then 0 ≤ a ≤ 3. -/
theorem circle_point_distance_range (a : ℝ) :
  (∃ x y : ℝ, (x - a)^2 + (y - a + 2)^2 = 1 ∧ x^2 + y^2 + x^2 + (y - 2)^2 = 10) →
  0 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_point_distance_range_l299_29991


namespace NUMINAMATH_CALUDE_intersection_of_lines_l299_29959

/-- The intersection point of two lines in 3D space --/
def intersection_point (A B C D : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

/-- Theorem: The intersection of lines AB and CD --/
theorem intersection_of_lines 
  (A : ℝ × ℝ × ℝ) 
  (B : ℝ × ℝ × ℝ) 
  (C : ℝ × ℝ × ℝ) 
  (D : ℝ × ℝ × ℝ) 
  (h1 : A = (6, -7, 7)) 
  (h2 : B = (15, -16, 11)) 
  (h3 : C = (0, 3, -6)) 
  (h4 : D = (2, -5, 10)) : 
  intersection_point A B C D = (144/27, -171/27, 181/27) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l299_29959


namespace NUMINAMATH_CALUDE_smallest_sum_with_conditions_l299_29933

theorem smallest_sum_with_conditions (a b : ℕ+) 
  (h1 : Nat.gcd (a + b) 330 = 1)
  (h2 : (a : ℕ)^(a : ℕ) % (b : ℕ)^(b : ℕ) = 0)
  (h3 : ¬(∃k : ℕ, b = k * a)) :
  (∀ c d : ℕ+, 
    Nat.gcd (c + d) 330 = 1 → 
    (c : ℕ)^(c : ℕ) % (d : ℕ)^(d : ℕ) = 0 → 
    ¬(∃k : ℕ, d = k * c) → 
    a + b ≤ c + d) ∧ 
  a + b = 147 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_with_conditions_l299_29933


namespace NUMINAMATH_CALUDE_xyz_sum_product_bounds_l299_29919

theorem xyz_sum_product_bounds (x y z : ℝ) : 
  5 * (x + y + z) = x^2 + y^2 + z^2 → 
  ∃ (M m : ℝ), 
    (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → 
      a * b + a * c + b * c ≤ M) ∧
    (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 → 
      m ≤ a * b + a * c + b * c) ∧
    M + 10 * m = 31 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_product_bounds_l299_29919


namespace NUMINAMATH_CALUDE_cube_pyramid_sum_l299_29952

/-- A solid figure formed by constructing a pyramid on one face of a cube -/
structure CubePyramid where
  cube_faces : ℕ := 6
  cube_edges : ℕ := 12
  cube_vertices : ℕ := 8
  pyramid_new_faces : ℕ := 4
  pyramid_new_edges : ℕ := 4
  pyramid_new_vertex : ℕ := 1

/-- The total number of exterior faces in the CubePyramid -/
def total_faces (cp : CubePyramid) : ℕ := cp.cube_faces - 1 + cp.pyramid_new_faces

/-- The total number of edges in the CubePyramid -/
def total_edges (cp : CubePyramid) : ℕ := cp.cube_edges + cp.pyramid_new_edges

/-- The total number of vertices in the CubePyramid -/
def total_vertices (cp : CubePyramid) : ℕ := cp.cube_vertices + cp.pyramid_new_vertex

theorem cube_pyramid_sum (cp : CubePyramid) : 
  total_faces cp + total_edges cp + total_vertices cp = 34 := by
  sorry

end NUMINAMATH_CALUDE_cube_pyramid_sum_l299_29952


namespace NUMINAMATH_CALUDE_tank_capacity_l299_29977

theorem tank_capacity (bucket_capacity : ℚ) : 
  (13 * 42 = 91 * bucket_capacity) → bucket_capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_tank_capacity_l299_29977


namespace NUMINAMATH_CALUDE_rectangle_area_comparison_l299_29916

theorem rectangle_area_comparison 
  (A B C D A' B' C' D' : ℝ) 
  (hA : 0 ≤ A) (hB : 0 ≤ B) (hC : 0 ≤ C) (hD : 0 ≤ D)
  (hA' : 0 ≤ A') (hB' : 0 ≤ B') (hC' : 0 ≤ C') (hD' : 0 ≤ D')
  (hAA' : A ≤ A') (hBB' : B ≤ B') (hCC' : C ≤ C') (hDB' : D ≤ B') :
  A + B + C + D ≤ A' + B' + C' + D' := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_comparison_l299_29916


namespace NUMINAMATH_CALUDE_sum_of_decimals_equals_fraction_l299_29986

theorem sum_of_decimals_equals_fraction :
  (∃ (x y : ℚ), x = 1/3 ∧ y = 7/9 ∧ x + y + (1/4 : ℚ) = 49/36) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_decimals_equals_fraction_l299_29986


namespace NUMINAMATH_CALUDE_equal_division_of_cakes_l299_29937

theorem equal_division_of_cakes (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) :
  total_cakes = 18 →
  num_children = 3 →
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_cakes_l299_29937


namespace NUMINAMATH_CALUDE_intersection_and_union_when_m_is_one_subset_condition_l299_29969

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 4}
def B : Set ℝ := {x | x < -5 ∨ x > 3}

-- Theorem for part 1
theorem intersection_and_union_when_m_is_one :
  (A 1 ∩ B = {x | 3 < x ∧ x ≤ 5}) ∧
  (A 1 ∪ B = {x | x < -5 ∨ x ≥ 1}) := by sorry

-- Theorem for part 2
theorem subset_condition :
  ∀ m : ℝ, A m ⊆ B ↔ m < -9 ∨ m > 3 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_m_is_one_subset_condition_l299_29969


namespace NUMINAMATH_CALUDE_cold_brew_time_per_batch_l299_29960

/-- Proves that the time to make one batch of cold brew coffee is 20 hours -/
theorem cold_brew_time_per_batch : 
  ∀ (batch_size : ℝ) (daily_consumption : ℝ) (total_time : ℝ) (total_days : ℕ),
    batch_size = 1.5 →  -- size of one batch in gallons
    daily_consumption = 48 →  -- 96 ounces every 2 days = 48 ounces per day
    total_time = 120 →  -- total hours spent making coffee
    total_days = 24 →  -- number of days
    (total_time / (total_days * daily_consumption / (batch_size * 128))) = 20 := by
  sorry


end NUMINAMATH_CALUDE_cold_brew_time_per_batch_l299_29960
