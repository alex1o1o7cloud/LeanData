import Mathlib

namespace NUMINAMATH_CALUDE_complex_number_magnitude_l4049_404969

theorem complex_number_magnitude (z w : ℂ) 
  (h1 : Complex.abs (3 * z - 2 * w) = 30)
  (h2 : Complex.abs (2 * z + 3 * w) = 19)
  (h3 : Complex.abs (z + w) = 5) :
  Complex.abs z = Real.sqrt 89 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l4049_404969


namespace NUMINAMATH_CALUDE_largest_B_term_l4049_404901

def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.1 ^ k)

theorem largest_B_term : 
  ∀ k ∈ Finset.range 501, B 45 ≥ B k :=
sorry

end NUMINAMATH_CALUDE_largest_B_term_l4049_404901


namespace NUMINAMATH_CALUDE_age_of_b_l4049_404945

theorem age_of_b (a b c : ℕ) 
  (avg_abc : (a + b + c) / 3 = 25)
  (avg_ac : (a + c) / 2 = 29) : 
  b = 17 := by sorry

end NUMINAMATH_CALUDE_age_of_b_l4049_404945


namespace NUMINAMATH_CALUDE_three_digit_numbers_problem_l4049_404949

theorem three_digit_numbers_problem : ∃ (a b : ℕ), 
  (100 ≤ a ∧ a < 1000) ∧ 
  (100 ≤ b ∧ b < 1000) ∧ 
  (a / 100 = b % 10) ∧ 
  (b / 100 = a % 10) ∧ 
  (a > b → a - b = 297) ∧ 
  (b > a → b - a = 297) ∧ 
  ((a < b → (a / 100 + (a / 10) % 10 + a % 10 = 23)) ∧ 
   (b < a → (b / 100 + (b / 10) % 10 + b % 10 = 23))) ∧ 
  ((a = 986 ∧ b = 689) ∨ (a = 689 ∧ b = 986)) := by
sorry

end NUMINAMATH_CALUDE_three_digit_numbers_problem_l4049_404949


namespace NUMINAMATH_CALUDE_remainder_of_base_12_num_div_9_l4049_404930

-- Define the base-12 number 1742₁₂
def base_12_num : ℕ := 1 * 12^3 + 7 * 12^2 + 4 * 12 + 2

-- Theorem statement
theorem remainder_of_base_12_num_div_9 :
  base_12_num % 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_base_12_num_div_9_l4049_404930


namespace NUMINAMATH_CALUDE_cone_height_l4049_404905

/-- Prove that a cone with base area 30 cm² and volume 60 cm³ has a height of 6 cm -/
theorem cone_height (base_area : ℝ) (volume : ℝ) (height : ℝ) : 
  base_area = 30 → volume = 60 → volume = (1/3) * base_area * height → height = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l4049_404905


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l4049_404920

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≤ 4}

-- Define set A
def A : Set ℝ := {x | |x - 1| ≤ 1}

-- Theorem statement
theorem complement_of_A_in_U :
  (U \ A) = {x : ℝ | -2 ≤ x ∧ x < 0} :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l4049_404920


namespace NUMINAMATH_CALUDE_black_bears_count_l4049_404964

/-- Represents the number of bears of each color in the park -/
structure BearPopulation where
  white : ℕ
  black : ℕ
  brown : ℕ

/-- Conditions for the bear population in the park -/
def validBearPopulation (p : BearPopulation) : Prop :=
  p.black = 2 * p.white ∧
  p.brown = p.black + 40 ∧
  p.white + p.black + p.brown = 190

/-- Theorem stating that under the given conditions, there are 60 black bears -/
theorem black_bears_count (p : BearPopulation) (h : validBearPopulation p) : p.black = 60 := by
  sorry

end NUMINAMATH_CALUDE_black_bears_count_l4049_404964


namespace NUMINAMATH_CALUDE_carols_age_difference_l4049_404995

theorem carols_age_difference (bob_age carol_age : ℕ) : 
  bob_age = 16 → carol_age = 50 → carol_age - 3 * bob_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_carols_age_difference_l4049_404995


namespace NUMINAMATH_CALUDE_stating_teacher_duty_arrangements_l4049_404998

/-- Represents the number of science teachers -/
def num_science_teachers : ℕ := 6

/-- Represents the number of liberal arts teachers -/
def num_liberal_arts_teachers : ℕ := 2

/-- Represents the number of days for duty arrangement -/
def num_days : ℕ := 3

/-- Represents the number of science teachers required per day -/
def science_teachers_per_day : ℕ := 2

/-- Represents the number of liberal arts teachers required per day -/
def liberal_arts_teachers_per_day : ℕ := 1

/-- Represents the minimum number of days a teacher should be on duty -/
def min_duty_days : ℕ := 1

/-- Represents the maximum number of days a teacher can be on duty -/
def max_duty_days : ℕ := 2

/-- 
Calculates the number of different arrangements for the teacher duty roster
given the specified conditions
-/
def num_arrangements : ℕ := 540

/-- 
Theorem stating that the number of different arrangements for the teacher duty roster
is equal to 540, given the specified conditions
-/
theorem teacher_duty_arrangements :
  num_arrangements = 540 :=
by sorry

end NUMINAMATH_CALUDE_stating_teacher_duty_arrangements_l4049_404998


namespace NUMINAMATH_CALUDE_sin_pi_over_two_minus_pi_over_six_l4049_404934

theorem sin_pi_over_two_minus_pi_over_six :
  Real.sin (π / 2 - π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_over_two_minus_pi_over_six_l4049_404934


namespace NUMINAMATH_CALUDE_fixed_point_on_line_fixed_point_coordinates_l4049_404928

/-- The fixed point through which the line ax + y + 1 = 0 always passes -/
def fixed_point : ℝ × ℝ := sorry

/-- The line equation ax + y + 1 = 0 -/
def line_equation (a x y : ℝ) : Prop := a * x + y + 1 = 0

/-- The theorem stating that the fixed point satisfies the line equation for all values of a -/
theorem fixed_point_on_line : ∀ a : ℝ, line_equation a (fixed_point.1) (fixed_point.2) := sorry

/-- The theorem proving that the fixed point is (0, -1) -/
theorem fixed_point_coordinates : fixed_point = (0, -1) := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_fixed_point_coordinates_l4049_404928


namespace NUMINAMATH_CALUDE_lcm_210_297_l4049_404919

theorem lcm_210_297 : Nat.lcm 210 297 = 20790 := by
  sorry

end NUMINAMATH_CALUDE_lcm_210_297_l4049_404919


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_product_product_equals_87526608_l4049_404916

theorem consecutive_even_numbers_product : Int → Prop :=
  fun n => (n - 2) * n * (n + 2) = 87526608

theorem product_equals_87526608 : consecutive_even_numbers_product 444 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_product_product_equals_87526608_l4049_404916


namespace NUMINAMATH_CALUDE_max_m_value_l4049_404956

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the inequality condition
def inequality_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x ≤ 1 → det (x + 1) x m (x - 1) ≥ -2

-- Theorem statement
theorem max_m_value :
  ∃ m : ℝ, inequality_condition m ∧ ∀ m' : ℝ, inequality_condition m' → m' ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l4049_404956


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l4049_404996

-- Define set A
def A : Set ℝ := {x | x^2 - x - 2 > 0}

-- Define set B
def B : Set ℝ := {x | |2*x - 3| < 3}

-- Theorem statement
theorem complement_A_inter_B :
  ∀ x : ℝ, x ∈ (A ∩ B)ᶜ ↔ x ≥ 3 ∨ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l4049_404996


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l4049_404987

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def C₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define external tangency
def externally_tangent (x y : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧
    ((x + 3)^2 + y^2 = (r + 1)^2) ∧
    ((x - 3)^2 + y^2 = (r + 3)^2)

-- Define the trajectory of the center of M
def trajectory (x y : ℝ) : Prop :=
  x < 0 ∧ x^2 - y^2/8 = 1

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ x y : ℝ, externally_tangent x y → trajectory x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l4049_404987


namespace NUMINAMATH_CALUDE_evaluate_expression_l4049_404914

theorem evaluate_expression : (900^2 : ℚ) / (200^2 - 196^2) = 511 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4049_404914


namespace NUMINAMATH_CALUDE_xy_max_value_l4049_404926

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 2) :
  ∃ (m : ℝ), m = 1/2 ∧ ∀ z, z = x*y → z ≤ m :=
sorry

end NUMINAMATH_CALUDE_xy_max_value_l4049_404926


namespace NUMINAMATH_CALUDE_subway_speed_problem_l4049_404980

theorem subway_speed_problem :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 7 ∧ (t^2 + 2*t) - (3^2 + 2*3) = 20 ∧ t = 5 := by
sorry

end NUMINAMATH_CALUDE_subway_speed_problem_l4049_404980


namespace NUMINAMATH_CALUDE_circle_C_is_correct_l4049_404959

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 6)^2 = 4

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := x - 2*y + 2 = 0

-- Define points A and B
def point_A : ℝ × ℝ := (-1, 0)
def point_B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem circle_C_is_correct :
  (∀ x y : ℝ, circle_C x y → tangent_line x y → False) ∧ 
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 := by
  sorry

end NUMINAMATH_CALUDE_circle_C_is_correct_l4049_404959


namespace NUMINAMATH_CALUDE_complement_of_union_l4049_404994

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def A : Set Int := {-1, 2}
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

theorem complement_of_union : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_l4049_404994


namespace NUMINAMATH_CALUDE_min_dot_product_sum_l4049_404979

/-- The ellipse on which point P moves --/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Circle E --/
def circle_E (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Circle F --/
def circle_F (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

/-- The dot product of vectors PA and PB plus the dot product of vectors PC and PD --/
def dot_product_sum (a b : ℝ) : ℝ := 2 * (a^2 + b^2)

theorem min_dot_product_sum :
  ∀ a b : ℝ, ellipse a b → 
  (∀ x y : ℝ, circle_E x y → ∀ u v : ℝ, circle_F u v → 
    dot_product_sum a b ≥ 6) ∧ 
  (∃ x y u v : ℝ, circle_E x y ∧ circle_F u v ∧ dot_product_sum a b = 6) :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_sum_l4049_404979


namespace NUMINAMATH_CALUDE_shortest_combined_track_length_l4049_404960

def melanie_pieces : List Nat := [8, 12]
def martin_pieces : List Nat := [20, 30]
def area_width : Nat := 100
def area_length : Nat := 200

theorem shortest_combined_track_length :
  let melanie_gcd := melanie_pieces.foldl Nat.gcd 0
  let martin_gcd := martin_pieces.foldl Nat.gcd 0
  let common_segment := Nat.lcm melanie_gcd martin_gcd
  let length_segments := area_length / common_segment
  let width_segments := area_width / common_segment
  let total_segments := 2 * (length_segments + width_segments)
  let single_track_length := total_segments * common_segment
  single_track_length * 2 = 1200 := by
sorry


end NUMINAMATH_CALUDE_shortest_combined_track_length_l4049_404960


namespace NUMINAMATH_CALUDE_suraj_average_increase_l4049_404902

/-- Represents a cricket player's innings record -/
structure InningsRecord where
  initial_innings : ℕ
  new_innings_score : ℕ
  new_average : ℚ

/-- Calculates the increase in average for a given innings record -/
def average_increase (record : InningsRecord) : ℚ :=
  record.new_average - (record.new_average * (record.initial_innings + 1) - record.new_innings_score) / record.initial_innings

/-- Theorem stating that Suraj's average increased by 6 runs -/
theorem suraj_average_increase :
  let suraj_record : InningsRecord := {
    initial_innings := 16,
    new_innings_score := 112,
    new_average := 16
  }
  average_increase suraj_record = 6 := by sorry

end NUMINAMATH_CALUDE_suraj_average_increase_l4049_404902


namespace NUMINAMATH_CALUDE_residue_mod_16_l4049_404958

theorem residue_mod_16 : 260 * 18 - 21 * 8 + 4 ≡ 4 [ZMOD 16] := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_16_l4049_404958


namespace NUMINAMATH_CALUDE_sum_of_cubes_product_l4049_404999

theorem sum_of_cubes_product (x y : ℤ) : x^3 + y^3 = 189 → x * y = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_product_l4049_404999


namespace NUMINAMATH_CALUDE_parabola_translation_l4049_404955

/-- Represents a vertical translation of a parabola -/
def verticalTranslation (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := λ x ↦ f x + k

/-- The original parabola function -/
def originalParabola : ℝ → ℝ := λ x ↦ x^2

theorem parabola_translation :
  verticalTranslation originalParabola 4 = λ x ↦ x^2 + 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_l4049_404955


namespace NUMINAMATH_CALUDE_tangent_beta_l4049_404937

theorem tangent_beta (a b : ℝ) (α β γ : Real) 
  (h1 : (a + b) / (a - b) = Real.tan ((α + β) / 2) / Real.tan ((α - β) / 2))
  (h2 : (α + β) / 2 = π / 2 - γ / 2)
  (h3 : (α - β) / 2 = π / 2 - (β + γ / 2)) :
  Real.tan β = (2 * b * Real.tan (γ / 2)) / ((a + b) * Real.tan (γ / 2)^2 + (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_beta_l4049_404937


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l4049_404986

theorem min_value_of_fraction (x : ℝ) (h : x ≥ 3/2) :
  (2*x^2 - 2*x + 1) / (x - 1) ≥ 2*Real.sqrt 2 + 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l4049_404986


namespace NUMINAMATH_CALUDE_system_solution_is_solution_set_l4049_404984

def system_solution (x y z : ℝ) : Prop :=
  x + y + z = 6 ∧ x*y + y*z + z*x = 11 ∧ x*y*z = 6

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)}

theorem system_solution_is_solution_set :
  ∀ x y z, system_solution x y z ↔ (x, y, z) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_is_solution_set_l4049_404984


namespace NUMINAMATH_CALUDE_min_value_of_function_l4049_404977

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  x + 4 / x^2 ≥ 3 ∧ ∀ ε > 0, ∃ x₀ > 0, x₀ + 4 / x₀^2 < 3 + ε :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l4049_404977


namespace NUMINAMATH_CALUDE_log_inequality_range_l4049_404903

-- Define the logarithm with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the range set
def range_set : Set ℝ := {x | x ∈ Set.Ioc 0 (1/8) ∪ Set.Ici 8}

-- State the theorem
theorem log_inequality_range :
  ∀ x > 0, Complex.abs (log_half x - (0 : ℝ) + 4*Complex.I) ≥ Complex.abs (3 + 4*Complex.I) ↔ x ∈ range_set :=
sorry

end NUMINAMATH_CALUDE_log_inequality_range_l4049_404903


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_l4049_404939

theorem greatest_multiple_of_four (x : ℕ) : x > 0 ∧ 4 ∣ x ∧ x^3 < 4096 → x ≤ 12 :=
sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_l4049_404939


namespace NUMINAMATH_CALUDE_max_value_expression_l4049_404965

theorem max_value_expression :
  ∃ (M : ℝ), M = 27 ∧
  ∀ (x y : ℝ),
    (Real.sqrt (36 - 4 * Real.sqrt 5) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 2) *
    (3 + 2 * Real.sqrt (10 - Real.sqrt 5) * Real.cos y - Real.cos (2 * y)) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l4049_404965


namespace NUMINAMATH_CALUDE_polynomial_simplification_l4049_404968

theorem polynomial_simplification (x : ℝ) :
  (2 * x^6 + 3 * x^5 + x^4 + 2 * x^2 + 15) - (x^6 + 4 * x^5 - 2 * x^4 + x^3 - 3 * x^2 + 18) =
  x^6 - x^5 + 3 * x^4 - x^3 + 5 * x^2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l4049_404968


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l4049_404938

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  /-- The length of two equal sides -/
  side_length : ℝ
  /-- The ratio of EJ to JF -/
  ej_jf_ratio : ℝ
  /-- side_length is positive -/
  side_length_pos : 0 < side_length
  /-- ej_jf_ratio is greater than 1 -/
  ej_jf_ratio_gt_one : 1 < ej_jf_ratio

/-- The theorem stating the length of the base of the isosceles triangle -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle)
    (h1 : t.side_length = 6)
    (h2 : t.ej_jf_ratio = 2) :
  ∃ (base_length : ℝ), base_length = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l4049_404938


namespace NUMINAMATH_CALUDE_even_quadratic_function_l4049_404972

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem even_quadratic_function (a b : ℝ) :
  let f : ℝ → ℝ := fun x ↦ a * x^2 + (b - 3) * x + 3
  IsEven f ∧ (∀ x, x ∈ Set.Icc (a^2 - 2) a → f x ∈ Set.range f) →
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_quadratic_function_l4049_404972


namespace NUMINAMATH_CALUDE_min_quotient_four_digit_number_l4049_404942

def is_digit (n : ℕ) : Prop := 0 < n ∧ n ≤ 9

theorem min_quotient_four_digit_number :
  ∃ (a b c d : ℕ),
    is_digit a ∧ is_digit b ∧ is_digit c ∧ is_digit d ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∀ (w x y z : ℕ),
      is_digit w → is_digit x → is_digit y → is_digit z →
      w ≠ x → w ≠ y → w ≠ z → x ≠ y → x ≠ z → y ≠ z →
      (1000 * a + 100 * b + 10 * c + d : ℚ) / (a + b + c + d) ≤
      (1000 * w + 100 * x + 10 * y + z : ℚ) / (w + x + y + z) :=
by sorry

end NUMINAMATH_CALUDE_min_quotient_four_digit_number_l4049_404942


namespace NUMINAMATH_CALUDE_spelling_contest_questions_spelling_contest_total_questions_l4049_404933

/-- Given a spelling contest with two competitors, Drew and Carla, prove the total number of questions asked. -/
theorem spelling_contest_questions (drew_correct : ℕ) (drew_wrong : ℕ) (carla_correct : ℕ) : ℕ :=
  let drew_total := drew_correct + drew_wrong
  let carla_wrong := 2 * drew_wrong
  let carla_total := carla_correct + carla_wrong
  drew_total + carla_total

/-- Prove that the total number of questions in the spelling contest is 52. -/
theorem spelling_contest_total_questions : spelling_contest_questions 20 6 14 = 52 := by
  sorry

end NUMINAMATH_CALUDE_spelling_contest_questions_spelling_contest_total_questions_l4049_404933


namespace NUMINAMATH_CALUDE_part_one_part_two_l4049_404971

-- Define polynomials A and B
def A (x y : ℝ) : ℝ := x^2 + x*y + 3*y
def B (x y : ℝ) : ℝ := x^2 - x*y

-- Part 1
theorem part_one (x y : ℝ) : (x - 2)^2 + |y + 5| = 0 → 2 * A x y - B x y = -56 := by
  sorry

-- Part 2
theorem part_two (x : ℝ) : (∀ y : ℝ, ∃ c : ℝ, 2 * A x y - B x y = c) ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4049_404971


namespace NUMINAMATH_CALUDE_sarah_borrowed_l4049_404906

/-- Calculates the earnings for a given number of hours based on the described wage structure --/
def earnings (hours : ℕ) : ℕ :=
  let fullCycles := hours / 8
  let remainingHours := hours % 8
  fullCycles * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8) + 
    (List.range remainingHours).sum.succ

/-- The amount Sarah borrowed is equal to her earnings for 40 hours of work --/
theorem sarah_borrowed (borrowedAmount : ℕ) : borrowedAmount = earnings 40 := by
  sorry

#eval earnings 40  -- Should output 180

end NUMINAMATH_CALUDE_sarah_borrowed_l4049_404906


namespace NUMINAMATH_CALUDE_four_digit_count_l4049_404967

/-- The count of four-digit numbers -/
def count_four_digit_numbers : ℕ := 9000

/-- The smallest four-digit number -/
def min_four_digit : ℕ := 1000

/-- The largest four-digit number -/
def max_four_digit : ℕ := 9999

/-- Theorem: The count of integers from the smallest to the largest four-digit number
    (inclusive) is equal to the count of four-digit numbers. -/
theorem four_digit_count :
  (Finset.range (max_four_digit - min_four_digit + 1)).card = count_four_digit_numbers := by
  sorry

end NUMINAMATH_CALUDE_four_digit_count_l4049_404967


namespace NUMINAMATH_CALUDE_sugar_water_dilution_l4049_404912

theorem sugar_water_dilution (initial_weight : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_weight = 300 →
  initial_concentration = 0.08 →
  final_concentration = 0.05 →
  initial_concentration * initial_weight = final_concentration * (initial_weight + water_added) →
  water_added = 180 := by
sorry

end NUMINAMATH_CALUDE_sugar_water_dilution_l4049_404912


namespace NUMINAMATH_CALUDE_order_of_magnitude_l4049_404943

theorem order_of_magnitude : 70.3 > 70.2 ∧ 70.2 > Real.log 0.3 := by
  have h : 0 < 0.3 ∧ 0.3 < 1 := by sorry
  sorry

end NUMINAMATH_CALUDE_order_of_magnitude_l4049_404943


namespace NUMINAMATH_CALUDE_wednesday_bags_raked_l4049_404924

theorem wednesday_bags_raked (charge_per_bag : ℕ) (monday_bags : ℕ) (tuesday_bags : ℕ) (total_money : ℕ) :
  charge_per_bag = 4 →
  monday_bags = 5 →
  tuesday_bags = 3 →
  total_money = 68 →
  ∃ wednesday_bags : ℕ, wednesday_bags = 9 ∧ 
    total_money = charge_per_bag * (monday_bags + tuesday_bags + wednesday_bags) :=
by sorry

end NUMINAMATH_CALUDE_wednesday_bags_raked_l4049_404924


namespace NUMINAMATH_CALUDE_proposition_evaluation_l4049_404963

theorem proposition_evaluation :
  (¬ ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0) ∧
  (¬ ∀ x y : ℝ, x + y > 2 → x > 1 ∧ y > 1) ∧
  (∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3) ∧
  (¬ ∀ a b c : ℝ, a ≠ 0 →
    (b^2 - 4*a*c > 0 ↔ 
      ∃ x y : ℝ, x < 0 ∧ y > 0 ∧ 
      a*x^2 + b*x + c = 0 ∧ 
      a*y^2 + b*y + c = 0)) := by
  sorry

end NUMINAMATH_CALUDE_proposition_evaluation_l4049_404963


namespace NUMINAMATH_CALUDE_ordering_from_log_half_inequalities_l4049_404973

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- State the theorem
theorem ordering_from_log_half_inequalities 
  (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : log_half b < log_half a) 
  (h5 : log_half a < log_half c) : 
  c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_ordering_from_log_half_inequalities_l4049_404973


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l4049_404941

theorem area_between_concentric_circles (r : ℝ) (h1 : r > 0) :
  let R := 3 * r
  R - r = 3 →
  π * R^2 - π * r^2 = 18 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l4049_404941


namespace NUMINAMATH_CALUDE_single_displacement_equivalent_l4049_404931

-- Define a type for plane figures
structure PlaneFigure where
  -- Add necessary properties for a plane figure

-- Define a function for parallel displacement
def parallelDisplacement (F : PlaneFigure) (v : ℝ × ℝ) : PlaneFigure :=
  sorry

-- Theorem statement
theorem single_displacement_equivalent (F : PlaneFigure) (v1 v2 : ℝ × ℝ) :
  ∃ v : ℝ × ℝ, parallelDisplacement F v = parallelDisplacement (parallelDisplacement F v1) v2 :=
sorry

end NUMINAMATH_CALUDE_single_displacement_equivalent_l4049_404931


namespace NUMINAMATH_CALUDE_car_speed_time_relation_l4049_404940

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- The distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

theorem car_speed_time_relation (m n : Car) 
  (h1 : n.speed = 2 * m.speed) 
  (h2 : distance n = distance m) : 
  n.time = m.time / 2 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_time_relation_l4049_404940


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l4049_404974

-- Problem 1
theorem problem_1 : 2 * (Real.sqrt 5 - 1) - Real.sqrt 5 = Real.sqrt 5 - 2 := by sorry

-- Problem 2
theorem problem_2 : Real.sqrt 3 * (Real.sqrt 3 + 4 / Real.sqrt 3) = 7 := by sorry

-- Problem 3
theorem problem_3 : |Real.sqrt 3 - 2| + 3 - 27 + Real.sqrt ((-5)^2) = 3 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l4049_404974


namespace NUMINAMATH_CALUDE_x_satisfies_equation_x_is_approximately_69_28_l4049_404909

/-- The number that satisfies the given equation -/
def x : ℝ := 69.28

/-- The given approximation of q -/
def q_approx : ℝ := 9.237333333333334

/-- Theorem stating that x satisfies the equation within a small margin of error -/
theorem x_satisfies_equation : 
  abs ((x * 0.004) / 0.03 - q_approx) < 0.000001 := by
  sorry

/-- Theorem stating that x is approximately equal to 69.28 -/
theorem x_is_approximately_69_28 : 
  abs (x - 69.28) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_x_satisfies_equation_x_is_approximately_69_28_l4049_404909


namespace NUMINAMATH_CALUDE_quadratic_root_transformation_l4049_404957

theorem quadratic_root_transformation (a b c r s : ℝ) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - b * x + a * c = 0 ↔ x = a * r + b ∨ x = a * s + b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_transformation_l4049_404957


namespace NUMINAMATH_CALUDE_train_journey_length_l4049_404982

theorem train_journey_length 
  (speed_on_time : ℝ) 
  (speed_late : ℝ) 
  (late_time : ℝ) 
  (h1 : speed_on_time = 100)
  (h2 : speed_late = 80)
  (h3 : late_time = 1/3)
  : ∃ (distance : ℝ), distance = 400/3 ∧ 
    distance / speed_on_time = distance / speed_late - late_time :=
by sorry

end NUMINAMATH_CALUDE_train_journey_length_l4049_404982


namespace NUMINAMATH_CALUDE_samanthas_last_name_has_seven_letters_l4049_404988

/-- The length of Jamie's last name -/
def jamies_last_name_length : ℕ := 4

/-- The length of Bobbie's last name -/
def bobbies_last_name_length : ℕ := 
  2 * jamies_last_name_length + 2

/-- The length of Samantha's last name -/
def samanthas_last_name_length : ℕ := 
  bobbies_last_name_length - 3

/-- Theorem stating that Samantha's last name has 7 letters -/
theorem samanthas_last_name_has_seven_letters : 
  samanthas_last_name_length = 7 := by
  sorry

end NUMINAMATH_CALUDE_samanthas_last_name_has_seven_letters_l4049_404988


namespace NUMINAMATH_CALUDE_smallest_value_for_y_between_zero_and_one_l4049_404961

theorem smallest_value_for_y_between_zero_and_one 
  (y : ℝ) (h1 : 0 < y) (h2 : y < 1) :
  y^3 ≤ min (2*y) (min (3*y) (min (y^(1/3)) (1/y))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_for_y_between_zero_and_one_l4049_404961


namespace NUMINAMATH_CALUDE_square_minus_three_times_l4049_404983

/-- The expression "square of a minus 3 times b" is equivalent to a^2 - 3*b -/
theorem square_minus_three_times (a b : ℝ) : (a^2 - 3*b) = (a^2 - 3*b) := by sorry

end NUMINAMATH_CALUDE_square_minus_three_times_l4049_404983


namespace NUMINAMATH_CALUDE_natashas_journey_l4049_404966

/-- Natasha's hill climbing problem -/
theorem natashas_journey (time_up : ℝ) (time_down : ℝ) (speed_up : ℝ) :
  time_up = 4 →
  time_down = 2 →
  speed_up = 2.25 →
  (speed_up * time_up * 2) / (time_up + time_down) = 3 :=
by sorry

end NUMINAMATH_CALUDE_natashas_journey_l4049_404966


namespace NUMINAMATH_CALUDE_quadratic_decreasing_before_vertex_l4049_404948

-- Define the quadratic function
def f (x : ℝ) : ℝ := 5 * (x - 3)^2 + 2

-- Theorem statement
theorem quadratic_decreasing_before_vertex :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 3 → f x₁ > f x₂ := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_before_vertex_l4049_404948


namespace NUMINAMATH_CALUDE_range_of_m_l4049_404990

theorem range_of_m (m : ℝ) : 
  (|m + 3| = m + 3) →
  (|3*m + 9| ≥ 4*m - 3 ↔ -3 ≤ m ∧ m ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l4049_404990


namespace NUMINAMATH_CALUDE_min_value_theorem_l4049_404929

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 4) :
  (1 / x + 4 / y) ≥ 9 / 4 ∧ 
  (1 / x + 4 / y = 9 / 4 ↔ y = 8 / 3 ∧ x = 4 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l4049_404929


namespace NUMINAMATH_CALUDE_count_valid_pairs_l4049_404970

def has_two_distinct_real_solutions (a b c : ℤ) : Prop :=
  b^2 - 4*a*c > 0

def valid_pair (b c : ℕ+) : Prop :=
  ¬(has_two_distinct_real_solutions 1 b c) ∧
  ¬(has_two_distinct_real_solutions 1 c b)

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ+ × ℕ+)), 
    (∀ (p : ℕ+ × ℕ+), p ∈ S ↔ valid_pair p.1 p.2) ∧
    Finset.card S = 6 := by sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l4049_404970


namespace NUMINAMATH_CALUDE_acceleration_at_two_l4049_404907

-- Define the distance function
def s (t : ℝ) : ℝ := 2 * t^3 - 5 * t^2

-- Define the velocity function as the derivative of the distance function
def v (t : ℝ) : ℝ := 6 * t^2 - 10 * t

-- Define the acceleration function as the derivative of the velocity function
def a (t : ℝ) : ℝ := 12 * t - 10

-- Theorem: The acceleration at t = 2 seconds is 14 units
theorem acceleration_at_two : a 2 = 14 := by
  sorry

-- Lemma: The velocity function is the derivative of the distance function
lemma velocity_is_derivative_of_distance (t : ℝ) : 
  deriv s t = v t := by
  sorry

-- Lemma: The acceleration function is the derivative of the velocity function
lemma acceleration_is_derivative_of_velocity (t : ℝ) : 
  deriv v t = a t := by
  sorry

end NUMINAMATH_CALUDE_acceleration_at_two_l4049_404907


namespace NUMINAMATH_CALUDE_initial_sum_proof_l4049_404900

def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem initial_sum_proof (P : ℝ) (r : ℝ) : 
  simple_interest P r 5 = 1500 ∧ 
  simple_interest P (r + 0.05) 5 = 1750 → 
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_initial_sum_proof_l4049_404900


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l4049_404921

theorem sqrt_difference_equality (p q : ℝ) 
  (h1 : p > 0) (h2 : 0 ≤ q) (h3 : q ≤ 5 * p) : 
  Real.sqrt (10 * p + 2 * Real.sqrt (25 * p^2 - q^2)) - 
  Real.sqrt (10 * p - 2 * Real.sqrt (25 * p^2 - q^2)) = 
  2 * Real.sqrt (5 * p - q) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l4049_404921


namespace NUMINAMATH_CALUDE_triangle_area_reduction_l4049_404954

theorem triangle_area_reduction (b h m : ℝ) (hb : b > 0) (hh : h > 0) (hm : m ≥ 0) :
  ∃ x : ℝ, 
    (1/2 : ℝ) * (b - x) * (h + m) = (1/2 : ℝ) * ((1/2 : ℝ) * b * h) ∧
    x = b * (2 * m + h) / (2 * (h + m)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_reduction_l4049_404954


namespace NUMINAMATH_CALUDE_total_athletes_l4049_404989

/-- Represents the number of athletes in each sport -/
structure Athletes :=
  (football : ℕ)
  (baseball : ℕ)
  (soccer : ℕ)
  (basketball : ℕ)

/-- The ratio of athletes in different sports -/
def ratio : Athletes := ⟨10, 7, 5, 4⟩

/-- The number of basketball players -/
def basketball_players : ℕ := 16

/-- Theorem stating the total number of athletes in the school -/
theorem total_athletes : 
  ∃ (k : ℕ), 
    k * ratio.football + 
    k * ratio.baseball + 
    k * ratio.soccer + 
    k * ratio.basketball = 104 ∧
    k * ratio.basketball = basketball_players :=
by sorry

end NUMINAMATH_CALUDE_total_athletes_l4049_404989


namespace NUMINAMATH_CALUDE_complex_square_l4049_404913

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_square : (1 + i)^2 = 2*i := by sorry

end NUMINAMATH_CALUDE_complex_square_l4049_404913


namespace NUMINAMATH_CALUDE_residue_negative_437_mod_13_l4049_404991

theorem residue_negative_437_mod_13 :
  ∃ (k : ℤ), -437 = 13 * k + 5 ∧ (0 : ℤ) ≤ 5 ∧ 5 < 13 := by
  sorry

end NUMINAMATH_CALUDE_residue_negative_437_mod_13_l4049_404991


namespace NUMINAMATH_CALUDE_malcolm_lights_theorem_l4049_404925

/-- The number of white lights Malcolm had initially --/
def initial_white_lights : ℕ := 59

/-- The number of red lights Malcolm bought --/
def red_lights : ℕ := 12

/-- The number of blue lights Malcolm bought --/
def blue_lights : ℕ := red_lights * 3

/-- The number of green lights Malcolm bought --/
def green_lights : ℕ := 6

/-- The number of colored lights Malcolm still needs to buy --/
def remaining_lights : ℕ := 5

/-- Theorem stating that the initial number of white lights is equal to 
    the sum of all colored lights bought and still to be bought --/
theorem malcolm_lights_theorem : 
  initial_white_lights = red_lights + blue_lights + green_lights + remaining_lights :=
by sorry

end NUMINAMATH_CALUDE_malcolm_lights_theorem_l4049_404925


namespace NUMINAMATH_CALUDE_block_tower_combinations_l4049_404951

theorem block_tower_combinations :
  let initial_blocks : ℕ := 35
  let final_blocks : ℕ := 65
  let additional_blocks : ℕ := final_blocks - initial_blocks
  ∃! n : ℕ, n = (additional_blocks + 1) ∧
    n = (Finset.filter (fun p : ℕ × ℕ => p.1 + p.2 = additional_blocks)
      (Finset.product (Finset.range (additional_blocks + 1)) (Finset.range (additional_blocks + 1)))).card :=
by sorry

end NUMINAMATH_CALUDE_block_tower_combinations_l4049_404951


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_projections_imply_frustum_of_cone_l4049_404985

/-- A solid object in 3D space. -/
structure Solid :=
  (shape : Type)

/-- Represents a view of a solid. -/
inductive View
  | Front
  | Side

/-- Represents the shape of a 2D projection. -/
inductive ProjectionShape
  | IsoscelesTrapezoid
  | Other

/-- Returns the shape of the projection of a solid from a given view. -/
def projection (s : Solid) (v : View) : ProjectionShape :=
  sorry

/-- Represents a frustum of a cone. -/
def FrustumOfCone : Solid :=
  sorry

/-- Theorem stating that a solid with isosceles trapezoid projections
    in both front and side views is a frustum of a cone. -/
theorem isosceles_trapezoid_projections_imply_frustum_of_cone
  (s : Solid)
  (h1 : projection s View.Front = ProjectionShape.IsoscelesTrapezoid)
  (h2 : projection s View.Side = ProjectionShape.IsoscelesTrapezoid) :
  s = FrustumOfCone :=
sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_projections_imply_frustum_of_cone_l4049_404985


namespace NUMINAMATH_CALUDE_lucas_age_probability_l4049_404950

def coin_sides : Finset ℕ := {5, 15}
def die_sides : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sum_probability (target : ℕ) : ℚ :=
  (coin_sides.filter (λ c => ∃ d ∈ die_sides, c + d = target)).card /
    (coin_sides.card * die_sides.card)

theorem lucas_age_probability :
  sum_probability 15 = 0 := by sorry

end NUMINAMATH_CALUDE_lucas_age_probability_l4049_404950


namespace NUMINAMATH_CALUDE_passes_through_first_and_fourth_quadrants_l4049_404993

-- Define a linear function
def f (x : ℝ) : ℝ := x - 1

-- Theorem statement
theorem passes_through_first_and_fourth_quadrants :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ f x = y) :=
sorry

end NUMINAMATH_CALUDE_passes_through_first_and_fourth_quadrants_l4049_404993


namespace NUMINAMATH_CALUDE_distinct_triangles_in_3x2_grid_l4049_404927

/-- Represents a grid of dots -/
structure Grid :=
  (rows : Nat)
  (cols : Nat)

/-- Calculates the total number of dots in the grid -/
def Grid.totalDots (g : Grid) : Nat :=
  g.rows * g.cols

/-- Calculates the number of collinear groups in the grid -/
def Grid.collinearGroups (g : Grid) : Nat :=
  g.rows + g.cols

/-- Theorem: In a 3x2 grid, the number of distinct triangles is 15 -/
theorem distinct_triangles_in_3x2_grid :
  let g : Grid := { rows := 3, cols := 2 }
  let totalCombinations := Nat.choose (g.totalDots) 3
  let validTriangles := totalCombinations - g.collinearGroups
  validTriangles = 15 := by sorry


end NUMINAMATH_CALUDE_distinct_triangles_in_3x2_grid_l4049_404927


namespace NUMINAMATH_CALUDE_discount_amount_l4049_404908

/-- The cost of a spiral notebook in dollars -/
def spiral_notebook_cost : ℕ := 15

/-- The cost of a personal planner in dollars -/
def personal_planner_cost : ℕ := 10

/-- The number of spiral notebooks purchased -/
def notebooks_purchased : ℕ := 4

/-- The number of personal planners purchased -/
def planners_purchased : ℕ := 8

/-- The total cost after discount in dollars -/
def discounted_total : ℕ := 112

/-- Theorem stating that the discount amount is $28 -/
theorem discount_amount : 
  (notebooks_purchased * spiral_notebook_cost + planners_purchased * personal_planner_cost) - discounted_total = 28 := by
  sorry

end NUMINAMATH_CALUDE_discount_amount_l4049_404908


namespace NUMINAMATH_CALUDE_work_completion_time_l4049_404932

theorem work_completion_time (T : ℝ) 
  (h1 : 100 * T = 200 * (T - 35)) 
  (h2 : T > 35) : T = 70 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l4049_404932


namespace NUMINAMATH_CALUDE_payment_difference_l4049_404953

/-- Represents the pizza with its properties and consumption details -/
structure Pizza :=
  (total_slices : ℕ)
  (plain_cost : ℚ)
  (mushroom_cost : ℚ)
  (mushroom_slices : ℕ)
  (alex_plain_slices : ℕ)
  (ally_plain_slices : ℕ)

/-- Calculates the total cost of the pizza -/
def total_cost (p : Pizza) : ℚ :=
  p.plain_cost + p.mushroom_cost

/-- Calculates the cost per slice -/
def cost_per_slice (p : Pizza) : ℚ :=
  total_cost p / p.total_slices

/-- Calculates Alex's payment -/
def alex_payment (p : Pizza) : ℚ :=
  cost_per_slice p * (p.mushroom_slices + p.alex_plain_slices)

/-- Calculates Ally's payment -/
def ally_payment (p : Pizza) : ℚ :=
  cost_per_slice p * p.ally_plain_slices

/-- Theorem stating the difference in payment between Alex and Ally -/
theorem payment_difference (p : Pizza) 
  (h1 : p.total_slices = 12)
  (h2 : p.plain_cost = 12)
  (h3 : p.mushroom_cost = 3)
  (h4 : p.mushroom_slices = 4)
  (h5 : p.alex_plain_slices = 4)
  (h6 : p.ally_plain_slices = 4)
  : alex_payment p - ally_payment p = 5 := by
  sorry


end NUMINAMATH_CALUDE_payment_difference_l4049_404953


namespace NUMINAMATH_CALUDE_roots_of_quadratic_expression_l4049_404981

theorem roots_of_quadratic_expression (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - x₁ - 2023 = 0) 
  (h₂ : x₂^2 - x₂ - 2023 = 0) : 
  x₁^3 - 2023*x₁ + x₂^2 = 4047 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_expression_l4049_404981


namespace NUMINAMATH_CALUDE_sharp_triple_30_l4049_404936

-- Define the function #
def sharp (N : ℝ) : ℝ := 0.6 * N + 2

-- Theorem statement
theorem sharp_triple_30 : sharp (sharp (sharp 30)) = 10.4 := by
  sorry

end NUMINAMATH_CALUDE_sharp_triple_30_l4049_404936


namespace NUMINAMATH_CALUDE_parallel_line_equation_l4049_404935

/-- The equation of a line passing through (-1, 2) and parallel to 2x + y - 5 = 0 is 2x + y = 0 -/
theorem parallel_line_equation :
  let P : ℝ × ℝ := (-1, 2)
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y - 5 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 2 * x + y = 0
  (∀ x y, l₂ x y ↔ (2 * P.1 + P.2 = 2 * x + y ∧ ∀ x₁ y₁ x₂ y₂, l₁ x₁ y₁ → l₁ x₂ y₂ → 
    2 * (x₂ - x₁) = y₁ - y₂)) := by
  sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l4049_404935


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l4049_404946

/-- Given a rectangle with perimeter 100 meters and length-to-width ratio 5:2,
    prove that its diagonal length is (5 * sqrt 290) / 7 meters. -/
theorem rectangle_diagonal (length width : ℝ) : 
  (2 * (length + width) = 100) →  -- Perimeter condition
  (length / width = 5 / 2) →      -- Ratio condition
  Real.sqrt (length^2 + width^2) = (5 * Real.sqrt 290) / 7 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l4049_404946


namespace NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l4049_404923

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x : ℝ | m - 1 ≤ x ∧ x ≤ 2*m + 1}
def B : Set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ 2}

-- Theorem 1: When m = 2, A ∩ B = [1, 2]
theorem intersection_when_m_is_two :
  A 2 ∩ B = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2: A ⊆ A ∩ B if and only if -2 ≤ m ≤ 1/2
theorem subset_condition (m : ℝ) :
  A m ⊆ A m ∩ B ↔ -2 ≤ m ∧ m ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_two_subset_condition_l4049_404923


namespace NUMINAMATH_CALUDE_min_total_weight_proof_l4049_404910

/-- The maximum number of crates the trailer can carry on a single trip -/
def max_crates : ℕ := 6

/-- The minimum weight of each crate in kilograms -/
def min_crate_weight : ℕ := 120

/-- The minimum total weight of crates on a single trip when carrying the maximum number of crates -/
def min_total_weight : ℕ := max_crates * min_crate_weight

theorem min_total_weight_proof :
  min_total_weight = 720 := by sorry

end NUMINAMATH_CALUDE_min_total_weight_proof_l4049_404910


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l4049_404952

def initial_flour : ℕ := 8
def additional_flour : ℕ := 2

theorem recipe_flour_amount : initial_flour + additional_flour = 10 := by sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l4049_404952


namespace NUMINAMATH_CALUDE_leadership_team_selection_l4049_404918

theorem leadership_team_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) : 
  Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_leadership_team_selection_l4049_404918


namespace NUMINAMATH_CALUDE_proposition_equivalence_implies_m_range_l4049_404922

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 3*x - 10 > 0
def q (x m : ℝ) : Prop := x > m^2 - m + 3

-- Define the range of m
def m_range (m : ℝ) : Prop := m ≤ -1 ∨ m ≥ 2

-- State the theorem
theorem proposition_equivalence_implies_m_range :
  (∀ x m : ℝ, (¬(p x) ↔ ¬(q x m))) → 
  (∀ m : ℝ, m_range m) :=
sorry

end NUMINAMATH_CALUDE_proposition_equivalence_implies_m_range_l4049_404922


namespace NUMINAMATH_CALUDE_max_vertex_sum_l4049_404915

/-- Represents the set of numbers to be assigned to cube faces -/
def cube_numbers : Finset ℕ := {7, 8, 9, 10, 11, 12}

/-- Represents a valid assignment of numbers to cube faces -/
def valid_assignment (assignment : Fin 6 → ℕ) : Prop :=
  ∀ i : Fin 6, assignment i ∈ cube_numbers ∧ (∀ j : Fin 6, i ≠ j → assignment i ≠ assignment j)

/-- Calculates the sum of products at vertices given a face assignment -/
def vertex_sum (assignment : Fin 6 → ℕ) : ℕ :=
  let opposite_pairs := [(0, 1), (2, 3), (4, 5)]
  (assignment 0 + assignment 1) * (assignment 2 + assignment 3) * (assignment 4 + assignment 5)

/-- Theorem stating the maximum sum of vertex products -/
theorem max_vertex_sum :
  ∃ assignment : Fin 6 → ℕ,
    valid_assignment assignment ∧
    vertex_sum assignment = 6859 ∧
    ∀ other : Fin 6 → ℕ, valid_assignment other → vertex_sum other ≤ 6859 := by
  sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l4049_404915


namespace NUMINAMATH_CALUDE_inscribed_triangle_angle_measure_l4049_404975

/-- Given a triangle PQR inscribed in a circle, if the measures of arcs PQ, QR, and RP
    are y + 60°, 2y + 40°, and 3y - 10° respectively, then the measure of interior angle Q
    is 62.5°. -/
theorem inscribed_triangle_angle_measure (y : ℝ) :
  let arc_PQ : ℝ := y + 60
  let arc_QR : ℝ := 2 * y + 40
  let arc_RP : ℝ := 3 * y - 10
  arc_PQ + arc_QR + arc_RP = 360 →
  (1 / 2 : ℝ) * arc_RP = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_angle_measure_l4049_404975


namespace NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l4049_404917

/-- Pascal's triangle as a function from row and column to value -/
def pascal (n k : ℕ) : ℕ := sorry

/-- Predicate to check if a number is four digits -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The set of all four-digit numbers in Pascal's triangle -/
def four_digit_pascal : Set ℕ :=
  {n | ∃ (i j : ℕ), pascal i j = n ∧ is_four_digit n}

/-- The third smallest element in a set of natural numbers -/
noncomputable def third_smallest (S : Set ℕ) : ℕ := sorry

theorem third_smallest_four_digit_pascal :
  third_smallest four_digit_pascal = 1002 := by sorry

end NUMINAMATH_CALUDE_third_smallest_four_digit_pascal_l4049_404917


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l4049_404904

theorem sqrt_sum_equals_seven (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_l4049_404904


namespace NUMINAMATH_CALUDE_line_slope_intercept_sum_l4049_404944

/-- Given a line passing through points (1,3) and (4,-2), prove that the sum of its slope and y-intercept is -1/3 -/
theorem line_slope_intercept_sum : 
  ∀ (m b : ℚ), 
  (3 : ℚ) = m * 1 + b →  -- Point (1,3) satisfies the equation
  (-2 : ℚ) = m * 4 + b → -- Point (4,-2) satisfies the equation
  m + b = -1/3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_intercept_sum_l4049_404944


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l4049_404911

/-- Calculates the molecular weight of a compound given the number of atoms and atomic weights -/
def molecularWeight (carbon_atoms : ℕ) (hydrogen_atoms : ℕ) (oxygen_atoms : ℕ) 
  (carbon_weight : ℝ) (hydrogen_weight : ℝ) (oxygen_weight : ℝ) : ℝ :=
  carbon_atoms * carbon_weight + hydrogen_atoms * hydrogen_weight + oxygen_atoms * oxygen_weight

/-- Theorem stating that the molecular weight of the given compound is 192.124 g/mol -/
theorem compound_molecular_weight :
  molecularWeight 6 8 7 12.01 1.008 16.00 = 192.124 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l4049_404911


namespace NUMINAMATH_CALUDE_remainder_problem_l4049_404962

theorem remainder_problem (n : ℤ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l4049_404962


namespace NUMINAMATH_CALUDE_race_speed_ratio_l4049_404992

theorem race_speed_ratio (k : ℝ) (v_B : ℝ) : 
  v_B > 0 →
  k * v_B * (20 / v_B) = 80 →
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l4049_404992


namespace NUMINAMATH_CALUDE_sinusoidal_function_properties_l4049_404997

/-- Proves that for a sinusoidal function y = A sin(Bx) - C with given properties, A = 2 and C = 1 -/
theorem sinusoidal_function_properties (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) 
  (hMax : A - C = 3) (hMin : -A - C = -1) : A = 2 ∧ C = 1 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_function_properties_l4049_404997


namespace NUMINAMATH_CALUDE_minimum_handshakes_l4049_404947

theorem minimum_handshakes (n : ℕ) (h : n = 30) :
  let handshakes_per_person := 3
  (n * handshakes_per_person) / 2 = 45 :=
by sorry

end NUMINAMATH_CALUDE_minimum_handshakes_l4049_404947


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l4049_404976

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 12 * (5 / 18))  -- Convert 12 km/hr to m/s
  (h2 : train_speed = 60 * (5 / 18))   -- Convert 60 km/hr to m/s
  (h3 : train_length = 300)
  (h4 : initial_distance = 300) :
  (train_length + initial_distance) / (train_speed - jogger_speed) = 15 := by
  sorry

#eval Float.ofScientific 15 0 1  -- Output: 15.0

end NUMINAMATH_CALUDE_train_passing_jogger_time_l4049_404976


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l4049_404978

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x - 1 = 0

-- Define the roots of the equation
def roots_of_equation (x₁ x₂ : ℝ) : Prop :=
  quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ x₁ ≠ x₂

-- Theorem statement
theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) :
  roots_of_equation x₁ x₂ → 1/x₁ + 1/x₂ = 3 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l4049_404978
