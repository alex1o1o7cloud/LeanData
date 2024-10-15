import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_coordinate_sum_l2910_291090

/-- A parallelogram with vertices P, Q, R, S in 2D space -/
structure Parallelogram where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- The sum of coordinates of a point -/
def sum_coordinates (point : ℝ × ℝ) : ℝ := point.1 + point.2

/-- Theorem: In a parallelogram PQRS with P(-3,-2), Q(1,-5), R(9,1), and P, R opposite vertices,
    the sum of coordinates of S is 9 -/
theorem parallelogram_coordinate_sum (PQRS : Parallelogram) 
    (h1 : PQRS.P = (-3, -2))
    (h2 : PQRS.Q = (1, -5))
    (h3 : PQRS.R = (9, 1))
    (h4 : PQRS.P.1 + PQRS.R.1 = PQRS.Q.1 + PQRS.S.1) 
    (h5 : PQRS.P.2 + PQRS.R.2 = PQRS.Q.2 + PQRS.S.2) :
    sum_coordinates PQRS.S = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_coordinate_sum_l2910_291090


namespace NUMINAMATH_CALUDE_biathlon_run_distance_l2910_291093

/-- A biathlon consisting of a bicycle race and a running race. -/
structure Biathlon where
  total_distance : ℝ
  bicycle_distance : ℝ
  run_velocity : ℝ
  bicycle_velocity : ℝ

/-- The theorem stating that for a specific biathlon, the running distance is 10 miles. -/
theorem biathlon_run_distance (b : Biathlon) 
  (h1 : b.total_distance = 155) 
  (h2 : b.bicycle_distance = 145) 
  (h3 : b.run_velocity = 10)
  (h4 : b.bicycle_velocity = 29) : 
  b.total_distance - b.bicycle_distance = 10 := by
  sorry

end NUMINAMATH_CALUDE_biathlon_run_distance_l2910_291093


namespace NUMINAMATH_CALUDE_fraction_integer_pairs_l2910_291087

theorem fraction_integer_pairs (m n : ℕ+) :
  (∃ h : ℕ+, (m.val^2 : ℚ) / (2 * m.val * n.val^2 - n.val^3 + 1) = h.val) ↔
  (∃ k : ℕ+, (m = 2 * k ∧ n = 1) ∨
             (m = k ∧ n = 2 * k) ∨
             (m = 8 * k.val^4 - k.val ∧ n = 2 * k)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_integer_pairs_l2910_291087


namespace NUMINAMATH_CALUDE_P_k_at_neg_half_is_zero_l2910_291077

/-- The unique polynomial P_k such that P_k(n) = 1^k + 2^k + 3^k + ... + n^k for each positive integer n -/
noncomputable def P_k (k : ℕ+) : ℝ → ℝ :=
  sorry

/-- For any positive integer k, P_k(-1/2) = 0 -/
theorem P_k_at_neg_half_is_zero (k : ℕ+) : P_k k (-1/2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_P_k_at_neg_half_is_zero_l2910_291077


namespace NUMINAMATH_CALUDE_constant_term_proof_l2910_291091

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the function to find the maximum coefficient term
def max_coeff_term (n : ℕ) : ℕ := sorry

-- Define the function to calculate the constant term
def constant_term (n : ℕ) : ℕ := sorry

theorem constant_term_proof (n : ℕ) :
  max_coeff_term n = 6 → constant_term n = 180 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_proof_l2910_291091


namespace NUMINAMATH_CALUDE_marbles_lost_calculation_specific_marbles_lost_l2910_291050

/-- Given an initial number of marbles and the current number of marbles in a bag,
    calculate the number of lost marbles. -/
def lost_marbles (initial : ℕ) (current : ℕ) : ℕ :=
  initial - current

/-- Theorem stating that the number of lost marbles is equal to
    the difference between the initial and current number of marbles. -/
theorem marbles_lost_calculation (initial current : ℕ) (h : current ≤ initial) :
  lost_marbles initial current = initial - current :=
by
  sorry

/-- The specific problem instance -/
def initial_marbles : ℕ := 8
def current_marbles : ℕ := 6

/-- Theorem for the specific problem instance -/
theorem specific_marbles_lost :
  lost_marbles initial_marbles current_marbles = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_calculation_specific_marbles_lost_l2910_291050


namespace NUMINAMATH_CALUDE_derivative_F_at_one_l2910_291060

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x^3 - 1) + f (1 - x^3)

theorem derivative_F_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) :
  deriv (F f) 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_derivative_F_at_one_l2910_291060


namespace NUMINAMATH_CALUDE_escalator_time_l2910_291044

/-- Time taken for a person to cover the length of a moving escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) 
  (h1 : escalator_speed = 15)
  (h2 : person_speed = 5)
  (h3 : escalator_length = 180) :
  escalator_length / (escalator_speed + person_speed) = 9 := by
  sorry

end NUMINAMATH_CALUDE_escalator_time_l2910_291044


namespace NUMINAMATH_CALUDE_sugar_concentration_mixture_l2910_291039

/-- Given two solutions with different sugar concentrations, calculate the sugar concentration of the resulting mixture --/
theorem sugar_concentration_mixture (original_concentration : ℝ) (replacement_concentration : ℝ)
  (replacement_fraction : ℝ) (h1 : original_concentration = 0.12)
  (h2 : replacement_concentration = 0.28000000000000004) (h3 : replacement_fraction = 0.25) :
  (1 - replacement_fraction) * original_concentration + replacement_fraction * replacement_concentration = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_sugar_concentration_mixture_l2910_291039


namespace NUMINAMATH_CALUDE_hyperbola_condition_l2910_291088

/-- Represents a curve defined by the equation x²/(4-t) + y²/(t-1) = 1 --/
def C (t : ℝ) := {(x, y) : ℝ × ℝ | x^2 / (4 - t) + y^2 / (t - 1) = 1}

/-- Defines when C is a hyperbola --/
def is_hyperbola (t : ℝ) : Prop := (4 - t) * (t - 1) < 0

/-- Theorem stating that C is a hyperbola iff t > 4 or t < 1 --/
theorem hyperbola_condition (t : ℝ) : 
  is_hyperbola t ↔ t > 4 ∨ t < 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_condition_l2910_291088


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2910_291041

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, 3 * x^2 + k * x + 16 = 0) ↔ k = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2910_291041


namespace NUMINAMATH_CALUDE_mary_marbles_l2910_291038

def dan_marbles : ℕ := 8
def mary_times_more : ℕ := 4

theorem mary_marbles : dan_marbles * mary_times_more = 32 := by
  sorry

end NUMINAMATH_CALUDE_mary_marbles_l2910_291038


namespace NUMINAMATH_CALUDE_square_area_and_perimeter_comparison_l2910_291047

theorem square_area_and_perimeter_comparison (a b : ℝ) :
  let square_I_diagonal := 2 * (a + b)
  let square_II_area := 4 * (square_I_diagonal^2 / 4)
  let square_II_perimeter := 4 * Real.sqrt square_II_area
  let rectangle_perimeter := 2 * (4 * (a + b) + (a + b))
  square_II_area = 8 * (a + b)^2 ∧ square_II_perimeter > rectangle_perimeter :=
by sorry

end NUMINAMATH_CALUDE_square_area_and_perimeter_comparison_l2910_291047


namespace NUMINAMATH_CALUDE_dessert_preference_l2910_291031

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ) :
  total = 50 →
  apple = 22 →
  chocolate = 20 →
  neither = 17 →
  ∃ (both : ℕ), both = apple + chocolate - (total - neither) :=
by sorry

end NUMINAMATH_CALUDE_dessert_preference_l2910_291031


namespace NUMINAMATH_CALUDE_sister_age_2021_l2910_291074

def kelsey_birth_year (kelsey_age_1999 : ℕ) : ℕ := 1999 - kelsey_age_1999

def sister_birth_year (kelsey_birth : ℕ) (age_difference : ℕ) : ℕ := kelsey_birth - age_difference

def current_age (birth_year : ℕ) (current_year : ℕ) : ℕ := current_year - birth_year

theorem sister_age_2021 (kelsey_age_1999 : ℕ) (age_difference : ℕ) (current_year : ℕ) :
  kelsey_age_1999 = 25 →
  age_difference = 3 →
  current_year = 2021 →
  current_age (sister_birth_year (kelsey_birth_year kelsey_age_1999) age_difference) current_year = 50 :=
by sorry

end NUMINAMATH_CALUDE_sister_age_2021_l2910_291074


namespace NUMINAMATH_CALUDE_johnnys_walk_legs_l2910_291086

/-- The number of legs for a given organism type -/
def legs_count (organism : String) : ℕ :=
  match organism with
  | "human" => 2
  | "dog" => 4
  | _ => 0

/-- The total number of legs for a group of organisms -/
def total_legs (humans : ℕ) (dogs : ℕ) : ℕ :=
  humans * legs_count "human" + dogs * legs_count "dog"

/-- Theorem stating that the total number of legs in Johnny's walking group is 12 -/
theorem johnnys_walk_legs :
  let humans : ℕ := 2  -- Johnny and his son
  let dogs : ℕ := 2    -- Johnny's two dogs
  total_legs humans dogs = 12 := by
  sorry

end NUMINAMATH_CALUDE_johnnys_walk_legs_l2910_291086


namespace NUMINAMATH_CALUDE_modulo_nine_equivalence_l2910_291032

theorem modulo_nine_equivalence : ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -2022 ≡ n [ZMOD 9] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modulo_nine_equivalence_l2910_291032


namespace NUMINAMATH_CALUDE_group_ratio_l2910_291020

theorem group_ratio (x : ℝ) (h1 : x > 0) (h2 : 1 - x > 0) : 
  15 * x + 21 * (1 - x) = 20 → x / (1 - x) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_group_ratio_l2910_291020


namespace NUMINAMATH_CALUDE_complex_magnitude_example_l2910_291083

theorem complex_magnitude_example : Complex.abs (-5 + (8/3)*Complex.I) = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_example_l2910_291083


namespace NUMINAMATH_CALUDE_work_completion_theorem_l2910_291001

/-- The number of boys in the first group that satisfies the work conditions -/
def num_boys_in_first_group : ℕ := 16

theorem work_completion_theorem (x : ℕ) 
  (h1 : 5 * (12 * 2 + x) = 4 * (13 * 2 + 24)) : 
  x = num_boys_in_first_group := by
  sorry

#check work_completion_theorem

end NUMINAMATH_CALUDE_work_completion_theorem_l2910_291001


namespace NUMINAMATH_CALUDE_art_gallery_theorem_l2910_291034

theorem art_gallery_theorem (T : ℕ) 
  (h1 : T / 3 = T - (2 * T / 3))  -- 1/3 of pieces are displayed
  (h2 : (T / 3) / 6 = T / 18)  -- 1/6 of displayed pieces are sculptures
  (h3 : (2 * T / 3) / 3 = 2 * T / 9)  -- 1/3 of non-displayed pieces are paintings
  (h4 : T / 18 + 400 = T / 18 + (T - (T / 3)) / 3)  -- 400 sculptures not on display
  (h5 : 3 * (T / 18) = T / 6)  -- 3 photographs for each displayed sculpture
  (h6 : 2 * (T / 18) = T / 9)  -- 2 installations for each displayed sculpture
  : T = 7200 := by
sorry

end NUMINAMATH_CALUDE_art_gallery_theorem_l2910_291034


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2910_291011

/-- Represents a tetrahedron with vertices P, Q, R, S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  PS : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is approximately 10.54 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 5.5,
    PR := 3.5,
    QR := 4,
    PS := 4.2,
    QS := 3.7,
    RS := 2.6
  }
  abs (tetrahedronVolume t - 10.54) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l2910_291011


namespace NUMINAMATH_CALUDE_line_through_point_l2910_291035

/-- Given a line equation -1/2 - 2kx = 5y that passes through the point (1/4, -6),
    prove that k = 59 is the unique solution. -/
theorem line_through_point (k : ℝ) : 
  (-1/2 : ℝ) - 2 * k * (1/4 : ℝ) = 5 * (-6 : ℝ) ↔ k = 59 := by
sorry

end NUMINAMATH_CALUDE_line_through_point_l2910_291035


namespace NUMINAMATH_CALUDE_woodworker_tables_l2910_291069

/-- Calculates the number of tables made given the total number of furniture legs,
    number of chairs, legs per chair, and legs per table. -/
def tables_made (total_legs : ℕ) (chairs : ℕ) (legs_per_chair : ℕ) (legs_per_table : ℕ) : ℕ :=
  (total_legs - chairs * legs_per_chair) / legs_per_table

/-- Theorem stating that given 40 total furniture legs, 6 chairs made,
    4 legs per chair, and 4 legs per table, the number of tables made is 4. -/
theorem woodworker_tables :
  tables_made 40 6 4 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_tables_l2910_291069


namespace NUMINAMATH_CALUDE_sum_of_integers_l2910_291002

theorem sum_of_integers (x y z : ℕ+) (h : 27 * x.val + 28 * y.val + 29 * z.val = 363) :
  10 * (x.val + y.val + z.val) = 130 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2910_291002


namespace NUMINAMATH_CALUDE_floor_sqrt_equation_solutions_l2910_291037

theorem floor_sqrt_equation_solutions : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (n + 1000) / 70 = ⌊Real.sqrt n⌋) ∧ 
    Finset.card S = 6 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_equation_solutions_l2910_291037


namespace NUMINAMATH_CALUDE_three_digit_divisibility_l2910_291082

theorem three_digit_divisibility (a b c : ℕ) (p : ℕ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0)
  (h_p : Nat.Prime p) (h_abc : p ∣ (100 * a + 10 * b + c)) (h_cba : p ∣ (100 * c + 10 * b + a)) :
  p ∣ (a + b + c) ∨ p ∣ (a - b + c) ∨ p ∣ (a - c) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_divisibility_l2910_291082


namespace NUMINAMATH_CALUDE_simplify_sqrt_product_simplify_log_product_l2910_291084

-- Part I
theorem simplify_sqrt_product (a : ℝ) (ha : 0 < a) :
  Real.sqrt (a^(1/4)) * Real.sqrt (a * Real.sqrt a) = Real.sqrt a := by sorry

-- Part II
theorem simplify_log_product :
  Real.log 3 / Real.log 2 * Real.log 5 / Real.log 3 * Real.log 4 / Real.log 5 = 2 := by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_product_simplify_log_product_l2910_291084


namespace NUMINAMATH_CALUDE_cube_sum_identity_l2910_291061

theorem cube_sum_identity (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h_sum : x + y + z = 0) :
  (x^3 + y^3 + z^3 + 3*x*y*z) / (x*y*z) = 6 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_identity_l2910_291061


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2910_291029

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 4}
def B : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 5}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2910_291029


namespace NUMINAMATH_CALUDE_one_zero_point_condition_l2910_291005

/-- A quadratic function with only one zero point -/
def has_one_zero_point (a : ℝ) : Prop :=
  ∃! x : ℝ, a * x^2 - x - 1 = 0

/-- The theorem stating the condition for a quadratic function to have only one zero point -/
theorem one_zero_point_condition (a : ℝ) :
  has_one_zero_point a ↔ a = 0 ∨ a = -1/4 :=
sorry

end NUMINAMATH_CALUDE_one_zero_point_condition_l2910_291005


namespace NUMINAMATH_CALUDE_perpendicular_foot_coordinates_l2910_291048

/-- Given a point P(1, √2, √3) in a 3-D Cartesian coordinate system and a perpendicular line PQ 
    drawn from P to the plane xOy with Q as the foot of the perpendicular, 
    prove that the coordinates of point Q are (1, √2, 0). -/
theorem perpendicular_foot_coordinates :
  let P : ℝ × ℝ × ℝ := (1, Real.sqrt 2, Real.sqrt 3)
  let xOy_plane : Set (ℝ × ℝ × ℝ) := {p | p.2.2 = 0}
  ∃ Q : ℝ × ℝ × ℝ, Q ∈ xOy_plane ∧ 
    (Q.1 = P.1 ∧ Q.2.1 = P.2.1 ∧ Q.2.2 = 0) ∧
    (∀ R ∈ xOy_plane, (P.1 - R.1)^2 + (P.2.1 - R.2.1)^2 + (P.2.2 - R.2.2)^2 ≥
                      (P.1 - Q.1)^2 + (P.2.1 - Q.2.1)^2 + (P.2.2 - Q.2.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_foot_coordinates_l2910_291048


namespace NUMINAMATH_CALUDE_S_not_union_of_finite_arithmetic_progressions_l2910_291070

-- Define the set S
def S : Set ℕ := {n : ℕ | ∀ p q : ℕ, (3 : ℚ) / n ≠ 1 / p + 1 / q}

-- Define what it means for a set to be the union of finitely many arithmetic progressions
def is_union_of_finite_arithmetic_progressions (T : Set ℕ) : Prop :=
  ∃ (k : ℕ) (a b : Fin k → ℕ), T = ⋃ i, {n : ℕ | ∃ m : ℕ, n = a i + m * b i}

-- State the theorem
theorem S_not_union_of_finite_arithmetic_progressions :
  ¬(is_union_of_finite_arithmetic_progressions S) := by
  sorry

end NUMINAMATH_CALUDE_S_not_union_of_finite_arithmetic_progressions_l2910_291070


namespace NUMINAMATH_CALUDE_equation_solutions_l2910_291096

def equation (x : ℝ) : Prop :=
  x ≠ 1 ∧ (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ (x = 3 ∨ x = -6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2910_291096


namespace NUMINAMATH_CALUDE_certain_number_proof_l2910_291036

theorem certain_number_proof (m : ℤ) (x : ℝ) (h1 : m = 6) (h2 : x^(2*m) = 2^(18 - m)) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2910_291036


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l2910_291051

/-- Given a hyperbola with equation x^2/a^2 - y^2/b^2 = 1, where a ≠ b, 
    if the angle between its asymptotes is 90°, then a/b = 1 -/
theorem hyperbola_asymptote_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ m₁ m₂ : ℝ, m₁ * m₂ = -1 ∧ m₁ = a/b ∧ m₂ = -a/b) →
  a / b = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_ratio_l2910_291051


namespace NUMINAMATH_CALUDE_proportion_check_l2910_291080

/-- A set of four line segments forms a proportion if the product of the means equals the product of the extremes. -/
def is_proportion (a b c d : ℝ) : Prop := b * c = a * d

/-- The given sets of line segments -/
def set_A : Fin 4 → ℝ := ![2, 3, 5, 6]
def set_B : Fin 4 → ℝ := ![1, 2, 3, 5]
def set_C : Fin 4 → ℝ := ![1, 3, 3, 7]
def set_D : Fin 4 → ℝ := ![3, 2, 4, 6]

theorem proportion_check :
  ¬ is_proportion (set_A 0) (set_A 1) (set_A 2) (set_A 3) ∧
  ¬ is_proportion (set_B 0) (set_B 1) (set_B 2) (set_B 3) ∧
  ¬ is_proportion (set_C 0) (set_C 1) (set_C 2) (set_C 3) ∧
  is_proportion (set_D 0) (set_D 1) (set_D 2) (set_D 3) := by
  sorry

end NUMINAMATH_CALUDE_proportion_check_l2910_291080


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l2910_291071

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -3
  let y : ℝ := 4
  second_quadrant x y :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l2910_291071


namespace NUMINAMATH_CALUDE_fraction_reciprocal_difference_l2910_291097

theorem fraction_reciprocal_difference : 
  let f : ℚ := 4/5
  let r : ℚ := 5/4  -- reciprocal of f
  r - f = 9/20 := by sorry

end NUMINAMATH_CALUDE_fraction_reciprocal_difference_l2910_291097


namespace NUMINAMATH_CALUDE_set_difference_proof_l2910_291055

def A : Set Int := {-1, 1, 3, 5, 7, 9}
def B : Set Int := {-1, 5, 7}

theorem set_difference_proof : A \ B = {1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_set_difference_proof_l2910_291055


namespace NUMINAMATH_CALUDE_polynomial_degree_of_product_l2910_291015

/-- The degree of the polynomial resulting from multiplying 
    x^5, x + 1/x, and 1 + 2/x + 3/x^2 -/
theorem polynomial_degree_of_product : ℕ := by
  sorry

end NUMINAMATH_CALUDE_polynomial_degree_of_product_l2910_291015


namespace NUMINAMATH_CALUDE_partnership_profit_calculation_l2910_291062

/-- Calculates the total profit given investments and one partner's profit share -/
def calculate_total_profit (investment_A investment_B investment_C profit_share_A : ℚ) : ℚ :=
  let total_investment := investment_A + investment_B + investment_C
  (profit_share_A * total_investment) / investment_A

theorem partnership_profit_calculation 
  (investment_A investment_B investment_C profit_share_A : ℚ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : profit_share_A = 3630) :
  calculate_total_profit investment_A investment_B investment_C profit_share_A = 12100 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_calculation_l2910_291062


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l2910_291028

/-- Given real numbers d, e, and f, prove that the matrix multiplication of 
    A = [[0, d, -e], [-d, 0, f], [e, -f, 0]] and 
    B = [[f^2, fd, fe], [fd, d^2, de], [fe, de, e^2]] 
    results in 
    C = [[d^2 - e^2, 2fd, 0], [0, f^2 - d^2, de-fe], [0, e^2 - d^2, fe-df]] -/
theorem matrix_multiplication_result (d e f : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, d, -e; -d, 0, f; e, -f, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![f^2, f*d, f*e; f*d, d^2, d*e; f*e, d*e, e^2]
  let C : Matrix (Fin 3) (Fin 3) ℝ := !![d^2 - e^2, 2*f*d, 0; 0, f^2 - d^2, d*e - f*e; 0, e^2 - d^2, f*e - d*f]
  A * B = C := by sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l2910_291028


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l2910_291017

theorem polynomial_division_quotient :
  let dividend : Polynomial ℤ := X^6 + 3*X^4 - 2*X^3 + X + 12
  let divisor : Polynomial ℤ := X - 2
  let quotient : Polynomial ℤ := X^5 + 2*X^4 + 6*X^3 + 10*X^2 + 18*X + 34
  dividend = divisor * quotient + 80 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l2910_291017


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l2910_291046

theorem quadratic_form_k_value : ∃ (a h : ℝ), ∀ x : ℝ, 
  x^2 - 6*x = a*(x - h)^2 + (-9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l2910_291046


namespace NUMINAMATH_CALUDE_mona_grouped_before_l2910_291045

/-- Represents the game groups Mona joined -/
structure GameGroups where
  totalGroups : ℕ
  playersPerGroup : ℕ
  uniquePlayers : ℕ
  knownPlayersInOneGroup : ℕ

/-- Calculates the number of players Mona had grouped with before in a specific group -/
def playersGroupedBefore (g : GameGroups) : ℕ :=
  g.totalGroups * g.playersPerGroup - g.uniquePlayers - g.knownPlayersInOneGroup

/-- Theorem stating the number of players Mona had grouped with before in a specific group -/
theorem mona_grouped_before (g : GameGroups) 
  (h1 : g.totalGroups = 9)
  (h2 : g.playersPerGroup = 4)
  (h3 : g.uniquePlayers = 33)
  (h4 : g.knownPlayersInOneGroup = 1) :
  playersGroupedBefore g = 2 := by
    sorry

end NUMINAMATH_CALUDE_mona_grouped_before_l2910_291045


namespace NUMINAMATH_CALUDE_class_average_score_l2910_291030

theorem class_average_score (total_students : Nat) (group1_students : Nat) (group1_average : ℚ)
  (score1 score2 score3 score4 : ℚ) :
  total_students = 30 →
  group1_students = 26 →
  group1_average = 82 →
  score1 = 90 →
  score2 = 85 →
  score3 = 88 →
  score4 = 80 →
  let group1_total := group1_students * group1_average
  let group2_total := score1 + score2 + score3 + score4
  let class_total := group1_total + group2_total
  class_total / total_students = 82.5 := by
sorry

end NUMINAMATH_CALUDE_class_average_score_l2910_291030


namespace NUMINAMATH_CALUDE_dog_weight_l2910_291073

theorem dog_weight (d l s : ℝ) 
  (total_weight : d + l + s = 36)
  (larger_comparison : d + l = 3 * s)
  (smaller_comparison : d + s = l) : 
  d = 9 := by
  sorry

end NUMINAMATH_CALUDE_dog_weight_l2910_291073


namespace NUMINAMATH_CALUDE_micks_to_macks_l2910_291079

/-- Given the conversion rates between micks, mocks, and macks, 
    prove that 200/3 micks equal 30 macks. -/
theorem micks_to_macks 
  (h1 : (8 : ℚ) * mick = 3 * mock) 
  (h2 : (5 : ℚ) * mock = 6 * mack) : 
  (200 : ℚ) / 3 * mick = 30 * mack :=
by
  sorry


end NUMINAMATH_CALUDE_micks_to_macks_l2910_291079


namespace NUMINAMATH_CALUDE_triple_solution_l2910_291099

theorem triple_solution (a b c : ℝ) :
  a^2 + 2*b^2 - 2*b*c = 16 ∧ 2*a*b - c^2 = 16 →
  (a = 4 ∧ b = 4 ∧ c = 4) ∨ (a = -4 ∧ b = -4 ∧ c = -4) := by
sorry

end NUMINAMATH_CALUDE_triple_solution_l2910_291099


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l2910_291019

theorem factorial_sum_equality : 
  ∃! (w x y z : ℕ), w > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  Nat.factorial w = Nat.factorial x + Nat.factorial y + Nat.factorial z ∧
  w = 3 ∧ x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l2910_291019


namespace NUMINAMATH_CALUDE_men_finished_race_l2910_291003

/-- The number of men who finished the race given the specified conditions -/
def men_who_finished (total_men : ℕ) : ℕ :=
  let tripped := total_men / 4
  let tripped_finished := tripped * 3 / 8
  let remaining_after_trip := total_men - tripped
  let dehydrated := remaining_after_trip * 2 / 9
  let dehydrated_not_finished := dehydrated * 11 / 14
  let remaining_after_dehydration := remaining_after_trip - dehydrated
  let lost := remaining_after_dehydration * 17 / 100
  let lost_finished := lost * 5 / 11
  let remaining_after_lost := remaining_after_dehydration - lost
  let obstacle := remaining_after_lost * 5 / 12
  let obstacle_finished := obstacle * 7 / 15
  let remaining_after_obstacle := remaining_after_lost - obstacle
  let cramps := remaining_after_obstacle * 3 / 7
  let cramps_finished := cramps * 4 / 5
  tripped_finished + lost_finished + obstacle_finished + cramps_finished

/-- Theorem stating that 25 men finished the race given the specified conditions -/
theorem men_finished_race : men_who_finished 80 = 25 := by
  sorry

end NUMINAMATH_CALUDE_men_finished_race_l2910_291003


namespace NUMINAMATH_CALUDE_binomial_square_condition_l2910_291053

theorem binomial_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l2910_291053


namespace NUMINAMATH_CALUDE_average_book_price_l2910_291025

/-- The average price of books bought from two shops -/
theorem average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) :
  books1 = 27 →
  books2 = 20 →
  price1 = 581 →
  price2 = 594 →
  (price1 + price2) / (books1 + books2 : ℚ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_average_book_price_l2910_291025


namespace NUMINAMATH_CALUDE_total_points_is_94_bonus_points_is_7_l2910_291027

/-- Represents the points system and creature counts in the video game --/
structure GameState where
  goblin_points : ℕ := 3
  troll_points : ℕ := 5
  dragon_points : ℕ := 10
  combo_bonus : ℕ := 7
  total_goblins : ℕ := 14
  total_trolls : ℕ := 15
  total_dragons : ℕ := 4
  defeated_goblins : ℕ := 9  -- 70% of 14 rounded down
  defeated_trolls : ℕ := 10  -- 2/3 of 15
  defeated_dragons : ℕ := 1

/-- Calculates the total points earned in the game --/
def calculate_points (state : GameState) : ℕ :=
  state.goblin_points * state.defeated_goblins +
  state.troll_points * state.defeated_trolls +
  state.dragon_points * state.defeated_dragons +
  state.combo_bonus * (min state.defeated_goblins (min state.defeated_trolls state.defeated_dragons))

/-- Theorem stating that the total points earned is 94 --/
theorem total_points_is_94 (state : GameState) : calculate_points state = 94 := by
  sorry

/-- Theorem stating that the bonus points earned is 7 --/
theorem bonus_points_is_7 (state : GameState) : 
  state.combo_bonus * (min state.defeated_goblins (min state.defeated_trolls state.defeated_dragons)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_points_is_94_bonus_points_is_7_l2910_291027


namespace NUMINAMATH_CALUDE_book_arrangement_l2910_291018

theorem book_arrangement (n m : ℕ) (h : n + m = 11) :
  Nat.choose (n + m) n = 462 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_l2910_291018


namespace NUMINAMATH_CALUDE_inequality_proof_l2910_291057

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  2 / (b * (a + b)) + 2 / (c * (b + c)) + 2 / (a * (c + a)) ≥ 27 / (a + b + c)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2910_291057


namespace NUMINAMATH_CALUDE_exist_three_numbers_not_exist_four_numbers_l2910_291042

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Theorem stating the existence of three different natural numbers satisfying the condition -/
theorem exist_three_numbers :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    is_perfect_square (a * b + 10) ∧
    is_perfect_square (a * c + 10) ∧
    is_perfect_square (b * c + 10) :=
sorry

/-- Theorem stating the non-existence of four different natural numbers satisfying the condition -/
theorem not_exist_four_numbers :
  ¬∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    is_perfect_square (a * b + 10) ∧
    is_perfect_square (a * c + 10) ∧
    is_perfect_square (a * d + 10) ∧
    is_perfect_square (b * c + 10) ∧
    is_perfect_square (b * d + 10) ∧
    is_perfect_square (c * d + 10) :=
sorry

end NUMINAMATH_CALUDE_exist_three_numbers_not_exist_four_numbers_l2910_291042


namespace NUMINAMATH_CALUDE_f_minimum_at_two_l2910_291014

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Theorem statement
theorem f_minimum_at_two :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = 2 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_at_two_l2910_291014


namespace NUMINAMATH_CALUDE_integer_root_quadratic_count_l2910_291026

theorem integer_root_quadratic_count :
  ∃! (S : Finset ℝ), (∀ a ∈ S, ∃ r s : ℤ, ∀ x : ℝ, x^2 + a*x + 9*a = 0 ↔ x = r ∨ x = s) ∧ Finset.card S = 10 :=
sorry

end NUMINAMATH_CALUDE_integer_root_quadratic_count_l2910_291026


namespace NUMINAMATH_CALUDE_floor_ceiling_solution_l2910_291081

theorem floor_ceiling_solution (c : ℝ) : 
  (∃ (n : ℤ), n = ⌊c⌋ ∧ 3 * (n : ℝ)^2 + 8 * (n : ℝ) - 35 = 0) ∧
  (let frac := c - ⌊c⌋; 4 * frac^2 - 12 * frac + 5 = 0 ∧ 0 ≤ frac ∧ frac < 1) →
  c = -9/2 := by
sorry

end NUMINAMATH_CALUDE_floor_ceiling_solution_l2910_291081


namespace NUMINAMATH_CALUDE_train_crossing_time_l2910_291054

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross : ℝ) :
  train_length = 110 →
  train_speed_kmh = 144 →
  time_to_cross = train_length / (train_speed_kmh * (1000 / 3600)) →
  time_to_cross = 2.75 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2910_291054


namespace NUMINAMATH_CALUDE_michelle_taxi_cost_l2910_291000

/-- Calculates the total cost of a taxi ride given the initial fee, distance, and per-mile charge. -/
def taxi_cost (initial_fee : ℝ) (distance : ℝ) (per_mile_charge : ℝ) : ℝ :=
  initial_fee + distance * per_mile_charge

/-- Theorem stating that for the given conditions, the total cost is $12. -/
theorem michelle_taxi_cost : taxi_cost 2 4 2.5 = 12 := by sorry

end NUMINAMATH_CALUDE_michelle_taxi_cost_l2910_291000


namespace NUMINAMATH_CALUDE_alchemerion_age_proof_l2910_291066

/-- Alchemerion's age in years -/
def alchemerion_age : ℕ := 360

/-- Alchemerion's son's age in years -/
def son_age : ℕ := alchemerion_age / 3

/-- Alchemerion's father's age in years -/
def father_age : ℕ := 2 * alchemerion_age + 40

theorem alchemerion_age_proof :
  (alchemerion_age = 3 * son_age) ∧
  (father_age = 2 * alchemerion_age + 40) ∧
  (alchemerion_age + son_age + father_age = 1240) :=
by sorry

end NUMINAMATH_CALUDE_alchemerion_age_proof_l2910_291066


namespace NUMINAMATH_CALUDE_power_division_equals_square_l2910_291092

theorem power_division_equals_square (a : ℝ) (h : a ≠ 0) : a^5 / a^3 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equals_square_l2910_291092


namespace NUMINAMATH_CALUDE_right_triangle_legs_sum_l2910_291049

theorem right_triangle_legs_sum (a b c : ℕ) : 
  a + 1 = b →                   -- legs are consecutive integers
  a^2 + b^2 = 41^2 →            -- Pythagorean theorem with hypotenuse 41
  a + b = 57 := by              -- sum of legs is 57
sorry

end NUMINAMATH_CALUDE_right_triangle_legs_sum_l2910_291049


namespace NUMINAMATH_CALUDE_fraction_equals_one_l2910_291007

theorem fraction_equals_one (x : ℝ) : x ≠ 3 → ((2 * x - 7) / (x - 3) = 1 ↔ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_one_l2910_291007


namespace NUMINAMATH_CALUDE_zero_sum_and_product_implies_all_zero_l2910_291065

theorem zero_sum_and_product_implies_all_zero (a b c d : ℝ) 
  (sum_zero : a + b + c + d = 0)
  (product_zero : a*b + c*d + a*c + b*c + a*d + b*d = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
sorry

end NUMINAMATH_CALUDE_zero_sum_and_product_implies_all_zero_l2910_291065


namespace NUMINAMATH_CALUDE_grocery_cost_l2910_291085

/-- The cost of groceries problem -/
theorem grocery_cost (mango_cost rice_cost flour_cost : ℝ)
  (h1 : 10 * mango_cost = 24 * rice_cost)
  (h2 : flour_cost = 2 * rice_cost)
  (h3 : flour_cost = 23) :
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 260.90 := by
  sorry

end NUMINAMATH_CALUDE_grocery_cost_l2910_291085


namespace NUMINAMATH_CALUDE_exists_tastrophic_function_l2910_291022

/-- A function is k-tastrophic if its k-th iteration raises its input to the k-th power. -/
def IsTastrophic (k : ℕ) (f : ℕ → ℕ) : Prop :=
  k > 1 ∧ ∀ n : ℕ, n > 0 → (f^[k] n = n^k)

/-- For every integer k > 1, there exists a k-tastrophic function. -/
theorem exists_tastrophic_function :
  ∀ k : ℕ, k > 1 → ∃ f : ℕ → ℕ, IsTastrophic k f :=
sorry

end NUMINAMATH_CALUDE_exists_tastrophic_function_l2910_291022


namespace NUMINAMATH_CALUDE_simplify_fraction_l2910_291043

theorem simplify_fraction (a : ℝ) (h : a ≠ -3) :
  (a^2 / (a + 3)) - (9 / (a + 3)) = a - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2910_291043


namespace NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l2910_291012

theorem largest_of_seven_consecutive_integers (n : ℕ) 
  (h1 : n > 0)
  (h2 : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 2222) : 
  n + 6 = 320 :=
sorry

end NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l2910_291012


namespace NUMINAMATH_CALUDE_series_sum_equals_one_l2910_291008

theorem series_sum_equals_one : 
  ∑' n : ℕ+, (1 : ℝ) / (n * (n + 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_one_l2910_291008


namespace NUMINAMATH_CALUDE_exponential_inequality_l2910_291076

theorem exponential_inequality (a b : ℝ) (h : a < b) :
  let f := fun (x : ℝ) => Real.exp x
  let A := f b - f a
  let B := (1/2) * (b - a) * (f a + f b)
  A < B := by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l2910_291076


namespace NUMINAMATH_CALUDE_f_value_l2910_291021

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (x + 1) = -f (-x + 1)
axiom f_periodic : ∀ x, f (x + 2) = f x
axiom f_def : ∀ x, -1 ≤ x → x ≤ 0 → f x = -2 * x * (x + 1)

-- State the theorem to be proved
theorem f_value : f (-3/2) = -1/2 := by sorry

end NUMINAMATH_CALUDE_f_value_l2910_291021


namespace NUMINAMATH_CALUDE_locus_is_parabolic_arc_l2910_291013

-- Define the semicircle
structure Semicircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency between a circle and a semicircle
def is_tangent_to_semicircle (c : Circle) (s : Semicircle) : Prop :=
  ∃ p : ℝ × ℝ, 
    (p.1 - s.center.1)^2 + (p.2 - s.center.2)^2 = s.radius^2 ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define tangency between a circle and a line (diameter)
def is_tangent_to_diameter (c : Circle) (s : Semicircle) : Prop :=
  ∃ p : ℝ × ℝ,
    p.2 = s.center.2 - s.radius ∧
    (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define a parabola
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ  -- y-coordinate of the directrix

-- Define a point being on a parabola
def on_parabola (p : ℝ × ℝ) (para : Parabola) : Prop :=
  (p.1 - para.focus.1)^2 + (p.2 - para.focus.2)^2 = (p.2 - para.directrix)^2

-- Main theorem
theorem locus_is_parabolic_arc (s : Semicircle) :
  ∀ c : Circle, 
    is_tangent_to_semicircle c s → 
    is_tangent_to_diameter c s → 
    ∃ para : Parabola, 
      para.focus = s.center ∧ 
      para.directrix = s.center.2 - 2 * s.radius ∧
      on_parabola c.center para ∧
      (c.center.1 - s.center.1)^2 + (c.center.2 - s.center.2)^2 < s.radius^2 :=
sorry

end NUMINAMATH_CALUDE_locus_is_parabolic_arc_l2910_291013


namespace NUMINAMATH_CALUDE_coconut_crab_goat_trade_l2910_291004

theorem coconut_crab_goat_trade (coconuts_per_crab : ℕ) (total_coconuts : ℕ) (final_goats : ℕ) :
  coconuts_per_crab = 3 →
  total_coconuts = 342 →
  final_goats = 19 →
  (total_coconuts / coconuts_per_crab) / final_goats = 6 :=
by sorry

end NUMINAMATH_CALUDE_coconut_crab_goat_trade_l2910_291004


namespace NUMINAMATH_CALUDE_calculator_game_sum_l2910_291024

/-- Represents the operations performed on the calculators. -/
def calculatorOperations (n : ℕ) (a b c : ℕ) : ℕ × ℕ × ℕ :=
  (a^3, b^2, c + 1)

/-- Applies the operations n times to the initial values. -/
def applyNTimes (n : ℕ) : ℕ × ℕ × ℕ :=
  match n with
  | 0 => (2, 1, 0)
  | m + 1 => calculatorOperations m (applyNTimes m).1 (applyNTimes m).2.1 (applyNTimes m).2.2

/-- The main theorem to be proved. -/
theorem calculator_game_sum :
  let (a, b, c) := applyNTimes 50
  a + b + c = 307 := by sorry

end NUMINAMATH_CALUDE_calculator_game_sum_l2910_291024


namespace NUMINAMATH_CALUDE_studio_audience_size_l2910_291033

theorem studio_audience_size :
  ∀ (total : ℕ) (envelope_ratio winner_ratio : ℚ) (winners : ℕ),
    envelope_ratio = 2/5 →
    winner_ratio = 1/5 →
    winners = 8 →
    (envelope_ratio * winner_ratio * total : ℚ) = winners →
    total = 100 := by
  sorry

end NUMINAMATH_CALUDE_studio_audience_size_l2910_291033


namespace NUMINAMATH_CALUDE_paco_salty_cookies_left_l2910_291058

/-- The number of salty cookies Paco had left -/
def salty_cookies_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

theorem paco_salty_cookies_left :
  salty_cookies_left 26 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_paco_salty_cookies_left_l2910_291058


namespace NUMINAMATH_CALUDE_infinite_series_sum_l2910_291016

/-- The sum of the infinite series ∑(n=1 to ∞) (3n - 2) / (n(n + 1)(n + 3)) is equal to -5/3 -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3)) = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l2910_291016


namespace NUMINAMATH_CALUDE_solve_scooter_problem_l2910_291056

def scooter_problem (purchase_price repair_cost gain_percentage : ℝ) : Prop :=
  let total_cost := purchase_price + repair_cost
  let gain := total_cost * (gain_percentage / 100)
  let selling_price := total_cost + gain
  (purchase_price = 4700) ∧ 
  (repair_cost = 1000) ∧ 
  (gain_percentage = 1.7543859649122806) →
  selling_price = 5800

theorem solve_scooter_problem :
  scooter_problem 4700 1000 1.7543859649122806 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_scooter_problem_l2910_291056


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2910_291040

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y ^ 2 = x + 3) → x ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2910_291040


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l2910_291094

/-- For a quadratic function y = ax^2 - 4ax + 1 where a ≠ 0, the axis of symmetry is x = 2 -/
theorem axis_of_symmetry (a : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 4 * a * x + 1
  (∀ x : ℝ, f (2 + x) = f (2 - x)) := by
sorry


end NUMINAMATH_CALUDE_axis_of_symmetry_l2910_291094


namespace NUMINAMATH_CALUDE_f_not_in_first_quadrant_l2910_291009

/-- A linear function defined by y = -3x + 2 -/
def f (x : ℝ) : ℝ := -3 * x + 2

/-- Definition of the first quadrant -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- Theorem stating that the function f does not pass through the first quadrant -/
theorem f_not_in_first_quadrant :
  ∀ x : ℝ, ¬(first_quadrant x (f x)) :=
sorry

end NUMINAMATH_CALUDE_f_not_in_first_quadrant_l2910_291009


namespace NUMINAMATH_CALUDE_change_distance_scientific_notation_l2910_291010

/-- Definition of scientific notation -/
def is_scientific_notation (n : ℝ) (a : ℝ) (p : ℤ) : Prop :=
  n = a * (10 : ℝ) ^ p ∧ 1 ≤ |a| ∧ |a| < 10

/-- The distance of Chang'e 1 from Earth in kilometers -/
def change_distance : ℝ := 380000

/-- Theorem stating that 380,000 is equal to 3.8 × 10^5 in scientific notation -/
theorem change_distance_scientific_notation :
  is_scientific_notation change_distance 3.8 5 :=
sorry

end NUMINAMATH_CALUDE_change_distance_scientific_notation_l2910_291010


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2910_291052

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : 2 * x + y = 2) :
  ∀ z : ℝ, (1 / x + 1 / y) ≥ z → z ≤ 3 / 2 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2910_291052


namespace NUMINAMATH_CALUDE_ad_square_area_l2910_291089

/-- Given two joined right triangles ABC and ACD with squares on their sides -/
structure JoinedTriangles where
  /-- Area of square on side AB -/
  ab_square_area : ℝ
  /-- Area of square on side BC -/
  bc_square_area : ℝ
  /-- Area of square on side CD -/
  cd_square_area : ℝ
  /-- ABC is a right triangle -/
  abc_right : True
  /-- ACD is a right triangle -/
  acd_right : True

/-- The theorem stating the area of the square on AD -/
theorem ad_square_area (t : JoinedTriangles)
  (h1 : t.ab_square_area = 36)
  (h2 : t.bc_square_area = 9)
  (h3 : t.cd_square_area = 16) :
  ∃ (ad_square_area : ℝ), ad_square_area = 61 := by
  sorry

end NUMINAMATH_CALUDE_ad_square_area_l2910_291089


namespace NUMINAMATH_CALUDE_students_taking_statistics_l2910_291006

/-- Given a group of students with the following properties:
  * There are 89 students in total
  * 36 students are taking history
  * 59 students are taking history or statistics or both
  * 27 students are taking history but not statistics
  Prove that 32 students are taking statistics -/
theorem students_taking_statistics
  (total : ℕ)
  (history : ℕ)
  (history_or_statistics : ℕ)
  (history_not_statistics : ℕ)
  (h_total : total = 89)
  (h_history : history = 36)
  (h_history_or_statistics : history_or_statistics = 59)
  (h_history_not_statistics : history_not_statistics = 27) :
  history_or_statistics - (history - history_not_statistics) = 32 := by
  sorry

#check students_taking_statistics

end NUMINAMATH_CALUDE_students_taking_statistics_l2910_291006


namespace NUMINAMATH_CALUDE_unique_value_not_in_range_l2910_291075

/-- A function f with specific properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- The theorem stating the properties of f and its unique value not in its range -/
theorem unique_value_not_in_range
  (a b c d : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (h19 : f a b c d 19 = 19)
  (h97 : f a b c d 97 = 97)
  (hinv : ∀ x, x ≠ -d/c → f a b c d (f a b c d x) = x) :
  ∃! y, ∀ x, f a b c d x ≠ y ∧ y = 58 :=
sorry

end NUMINAMATH_CALUDE_unique_value_not_in_range_l2910_291075


namespace NUMINAMATH_CALUDE_complex_inequality_l2910_291064

theorem complex_inequality (a b c : ℂ) (h : a * Complex.abs (b * c) + b * Complex.abs (c * a) + c * Complex.abs (a * b) = 0) :
  Complex.abs ((a - b) * (b - c) * (c - a)) ≥ 3 * Real.sqrt 3 * Complex.abs (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_l2910_291064


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2910_291068

theorem inequality_system_solution_set :
  let S : Set ℝ := {x | x - 2 ≥ -5 ∧ 3*x < x + 2}
  S = {x | -3 ≤ x ∧ x < 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2910_291068


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l2910_291078

theorem stationery_box_sheets : ∀ (S E : ℕ),
  S - E = 30 →  -- Ann's condition
  2 * E = S →   -- Bob's condition
  3 * E = S - 10 →  -- Sue's condition
  S = 40 := by
sorry

end NUMINAMATH_CALUDE_stationery_box_sheets_l2910_291078


namespace NUMINAMATH_CALUDE_box_width_is_twenty_l2910_291063

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Represents the properties of cubes filling a box -/
structure CubeFill where
  box : BoxDimensions
  cubeCount : ℕ
  cubeSideLength : ℕ

/-- Theorem stating that a box with given dimensions filled with 56 cubes has a width of 20 inches -/
theorem box_width_is_twenty
  (box : BoxDimensions)
  (fill : CubeFill)
  (h1 : box.length = 35)
  (h2 : box.depth = 10)
  (h3 : fill.box = box)
  (h4 : fill.cubeCount = 56)
  (h5 : fill.cubeSideLength * fill.cubeSideLength * fill.cubeSideLength * fill.cubeCount = box.length * box.width * box.depth)
  (h6 : fill.cubeSideLength ∣ box.length ∧ fill.cubeSideLength ∣ box.width ∧ fill.cubeSideLength ∣ box.depth)
  : box.width = 20 := by
  sorry

#check box_width_is_twenty

end NUMINAMATH_CALUDE_box_width_is_twenty_l2910_291063


namespace NUMINAMATH_CALUDE_factory_output_percentage_l2910_291098

theorem factory_output_percentage (may_output june_output : ℝ) : 
  may_output = june_output * (1 - 0.2) → 
  (june_output - may_output) / may_output = 0.25 := by
sorry

end NUMINAMATH_CALUDE_factory_output_percentage_l2910_291098


namespace NUMINAMATH_CALUDE_april_roses_unsold_l2910_291059

/-- Calculates the number of roses left unsold after a sale -/
def roses_left_unsold (initial_roses : ℕ) (price_per_rose : ℕ) (total_earned : ℕ) : ℕ :=
  initial_roses - (total_earned / price_per_rose)

/-- Proves that the number of roses left unsold is 4 -/
theorem april_roses_unsold :
  roses_left_unsold 13 4 36 = 4 := by
  sorry

end NUMINAMATH_CALUDE_april_roses_unsold_l2910_291059


namespace NUMINAMATH_CALUDE_angle_relation_l2910_291072

theorem angle_relation (α β : Real) (h1 : 0 < α) (h2 : α < π) (h3 : 0 < β) (h4 : β < π)
  (h5 : Real.tan (α - β) = 1/2) (h6 : Real.cos β = -7 * Real.sqrt 2 / 10) :
  2 * α - β = -3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_relation_l2910_291072


namespace NUMINAMATH_CALUDE_john_daily_calories_is_3275_l2910_291095

/-- Calculates John's total daily calorie intake based on given meal and shake information. -/
def johnDailyCalories : ℕ :=
  let breakfastCalories : ℕ := 500
  let lunchCalories : ℕ := breakfastCalories + (breakfastCalories / 4)
  let dinnerCalories : ℕ := 2 * lunchCalories
  let shakeCalories : ℕ := 3 * 300
  breakfastCalories + lunchCalories + dinnerCalories + shakeCalories

/-- Theorem stating that John's total daily calorie intake is 3275 calories. -/
theorem john_daily_calories_is_3275 : johnDailyCalories = 3275 := by
  sorry

end NUMINAMATH_CALUDE_john_daily_calories_is_3275_l2910_291095


namespace NUMINAMATH_CALUDE_two_different_color_chips_probability_l2910_291067

def blue_chips : ℕ := 5
def yellow_chips : ℕ := 3
def red_chips : ℕ := 4

def total_chips : ℕ := blue_chips + yellow_chips + red_chips

def probability_different_colors : ℚ :=
  (blue_chips * yellow_chips + blue_chips * red_chips + yellow_chips * red_chips) * 2 /
  (total_chips * total_chips)

theorem two_different_color_chips_probability :
  probability_different_colors = 47 / 72 := by
  sorry

end NUMINAMATH_CALUDE_two_different_color_chips_probability_l2910_291067


namespace NUMINAMATH_CALUDE_max_odd_sequence_length_l2910_291023

/-- The type of sequences where each term is obtained by adding the largest digit of the previous term --/
def DigitAddSequence := ℕ → ℕ

/-- The largest digit of a natural number --/
def largest_digit (n : ℕ) : ℕ := sorry

/-- The property that a sequence follows the digit addition rule --/
def is_digit_add_sequence (s : DigitAddSequence) : Prop :=
  ∀ n, s (n + 1) = s n + largest_digit (s n)

/-- The property that a number is odd --/
def is_odd (n : ℕ) : Prop := n % 2 = 1

/-- The length of a sequence of successive odd terms starting from a given index --/
def odd_sequence_length (s : DigitAddSequence) (start : ℕ) : ℕ := sorry

/-- The theorem stating that the maximal number of successive odd terms is 5 --/
theorem max_odd_sequence_length (s : DigitAddSequence) (h : is_digit_add_sequence s) :
  ∀ start, odd_sequence_length s start ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_odd_sequence_length_l2910_291023
