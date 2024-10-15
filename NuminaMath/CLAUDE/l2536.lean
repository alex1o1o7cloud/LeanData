import Mathlib

namespace NUMINAMATH_CALUDE_game_points_sequence_l2536_253637

theorem game_points_sequence (a : ℕ → ℕ) : 
  a 1 = 2 ∧ 
  a 3 = 5 ∧ 
  a 4 = 8 ∧ 
  a 5 = 12 ∧ 
  a 6 = 17 ∧ 
  (∀ n : ℕ, n > 1 → (a (n + 1) - a n) - (a n - a (n - 1)) = 1) →
  a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_game_points_sequence_l2536_253637


namespace NUMINAMATH_CALUDE_log_simplification_l2536_253673

-- Define variables
variable (a b c d x y : ℝ)
-- Assume all variables are positive to ensure logarithms are defined
variable (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hx : x > 0) (hy : y > 0)

-- Define the theorem
theorem log_simplification :
  Real.log (2*a/(3*b)) + Real.log (5*b/(4*c)) + Real.log (6*c/(7*d)) - Real.log (20*a*y/(21*d*x)) = Real.log (3*x/(4*y)) :=
by sorry

end NUMINAMATH_CALUDE_log_simplification_l2536_253673


namespace NUMINAMATH_CALUDE_unused_sector_angle_l2536_253600

/-- Given a cone with radius 10 cm and volume 250π cm³, 
    prove that the angle of the sector not used to form the cone is 72°. -/
theorem unused_sector_angle (r h : ℝ) (volume : ℝ) : 
  r = 10 → 
  volume = 250 * Real.pi → 
  (1/3) * Real.pi * r^2 * h = volume → 
  Real.sqrt (r^2 + h^2) = 12.5 → 
  360 - (360 * ((2 * Real.pi * r) / (2 * Real.pi * 12.5))) = 72 := by
  sorry

#check unused_sector_angle

end NUMINAMATH_CALUDE_unused_sector_angle_l2536_253600


namespace NUMINAMATH_CALUDE_complex_number_properties_l2536_253693

theorem complex_number_properties (z : ℂ) (h : Complex.abs z ^ 2 + 2 * z - Complex.I * 2 = 0) :
  z = -1 + Complex.I ∧ Complex.abs z + Complex.abs (z + 3 * Complex.I) > Complex.abs (2 * z + 3 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l2536_253693


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_fourths_l2536_253677

theorem reciprocal_of_negative_three_fourths :
  let x : ℚ := -3/4
  let y : ℚ := -4/3
  (x * y = 1) → (y = x⁻¹) := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_fourths_l2536_253677


namespace NUMINAMATH_CALUDE_teacher_number_game_l2536_253649

theorem teacher_number_game (x : ℤ) : 
  let ben_result := ((x + 3) * 2) - 2
  let sue_result := ((ben_result + 1) * 2) + 4
  sue_result = 2 * x + 30 := by
sorry

end NUMINAMATH_CALUDE_teacher_number_game_l2536_253649


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2536_253674

theorem contrapositive_equivalence (a b c d : ℝ) :
  ((a = b ∧ c = d) → (a + c = b + d)) ↔ ((a + c ≠ b + d) → (a ≠ b ∨ c ≠ d)) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2536_253674


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2536_253661

theorem quadratic_equation_coefficients :
  ∀ (a b c : ℝ), 
  (∀ x, 9 * x^2 = 4 * (3 * x - 1)) →
  (∀ x, a * x^2 + b * x + c = 0) →
  a = 9 ∧ b = -12 ∧ c = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l2536_253661


namespace NUMINAMATH_CALUDE_f_one_lower_bound_l2536_253633

/-- A function f(x) = 4x^2 - mx + 5 that is increasing on [-2, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The property that f is increasing on [-2, +∞) -/
def is_increasing_on_interval (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x → x < y → f m x < f m y

theorem f_one_lower_bound (m : ℝ) (h : is_increasing_on_interval m) : f m 1 ≥ 25 := by
  sorry

end NUMINAMATH_CALUDE_f_one_lower_bound_l2536_253633


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2536_253640

theorem arithmetic_mean_problem (x : ℝ) : 
  (x + 8 + 15 + 2*x + 13 + 2*x + 4) / 5 = 24 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2536_253640


namespace NUMINAMATH_CALUDE_min_difference_ln2_l2536_253628

/-- Given functions f and g, prove that the minimum value of x₁ - x₂ is ln(2) when f(x₁) = g(x₂) -/
theorem min_difference_ln2 (f g : ℝ → ℝ) (x₁ x₂ : ℝ) 
  (hf : f = fun x ↦ Real.log (x / 2) + 1 / 2)
  (hg : g = fun x ↦ Real.exp (x - 2))
  (hx₁ : x₁ > 0)
  (hequal : f x₁ = g x₂) :
  ∃ (min : ℝ), min = Real.log 2 ∧ ∀ y₁ y₂, f y₁ = g y₂ → y₁ - y₂ ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_difference_ln2_l2536_253628


namespace NUMINAMATH_CALUDE_function_equality_l2536_253602

theorem function_equality (f g : ℕ → ℕ) 
  (h_surj : Function.Surjective f)
  (h_inj : Function.Injective g)
  (h_ge : ∀ n : ℕ, f n ≥ g n) :
  ∀ n : ℕ, f n = g n := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2536_253602


namespace NUMINAMATH_CALUDE_percentage_problem_l2536_253699

theorem percentage_problem (P : ℝ) (number : ℝ) : 
  number = 40 →
  P = (0.5 * number) + 10 →
  P = 30 :=
by sorry

end NUMINAMATH_CALUDE_percentage_problem_l2536_253699


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2536_253610

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2536_253610


namespace NUMINAMATH_CALUDE_inverse_proportionality_l2536_253627

theorem inverse_proportionality (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * (-10) = k) :
  x = 10 / 3 → y = -15 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportionality_l2536_253627


namespace NUMINAMATH_CALUDE_integer_solution_problem_l2536_253622

theorem integer_solution_problem :
  let S : Set (ℤ × ℤ × ℤ) := {(a, b, c) | a + b + c = 15 ∧ (a - 3)^3 + (b - 5)^3 + (c - 7)^3 = 540}
  S = {(12, 0, 3), (-2, 14, 3), (-1, 0, 16), (-2, 1, 16)} := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_problem_l2536_253622


namespace NUMINAMATH_CALUDE_find_t_l2536_253645

theorem find_t : ∃ t : ℤ, (∃ s : ℤ, 12 * s + 7 * t = 173 ∧ s = t - 3) → t = 11 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l2536_253645


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2536_253613

theorem concentric_circles_ratio (s S : ℝ) (h : s > 0) (H : S > s) :
  (π * S^2 = 3/2 * (π * S^2 - π * s^2)) → S/s = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_ratio_l2536_253613


namespace NUMINAMATH_CALUDE_susan_bob_cat_difference_l2536_253642

/-- Proves that Susan has 6 more cats than Bob after all exchanges --/
theorem susan_bob_cat_difference : 
  let susan_initial : ℕ := 21
  let bob_initial : ℕ := 3
  let emma_initial : ℕ := 8
  let neighbor_to_susan : ℕ := 12
  let neighbor_to_bob : ℕ := 14
  let neighbor_to_emma : ℕ := 6
  let susan_to_bob : ℕ := 6
  let emma_to_susan : ℕ := 5
  let emma_to_bob : ℕ := 3

  let susan_final := susan_initial + neighbor_to_susan - susan_to_bob + emma_to_susan
  let bob_final := bob_initial + neighbor_to_bob + susan_to_bob + emma_to_bob

  susan_final - bob_final = 6 :=
by sorry

end NUMINAMATH_CALUDE_susan_bob_cat_difference_l2536_253642


namespace NUMINAMATH_CALUDE_abs_neg_2023_l2536_253621

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l2536_253621


namespace NUMINAMATH_CALUDE_quadratic_triple_root_relation_l2536_253650

/-- Given a quadratic equation ax^2 + bx + c = 0 where one root is triple the other,
    prove that 3b^2 = 16ac -/
theorem quadratic_triple_root_relation (a b c : ℝ) (h : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_triple_root_relation_l2536_253650


namespace NUMINAMATH_CALUDE_remainder_of_sequence_sum_l2536_253681

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def sequence_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a₁ + (n - 1) * d)) / 2

theorem remainder_of_sequence_sum :
  ∃ n : ℕ, 
    arithmetic_sequence 1 6 n = 403 ∧ 
    sequence_sum 1 6 n % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sequence_sum_l2536_253681


namespace NUMINAMATH_CALUDE_survey_analysis_l2536_253619

/-- Data from the survey --/
structure SurveyData where
  a : Nat  -- Females who understand
  b : Nat  -- Females who do not understand
  c : Nat  -- Males who understand
  d : Nat  -- Males who do not understand

/-- Chi-square calculation function --/
def chiSquare (data : SurveyData) : Rat :=
  let n := data.a + data.b + data.c + data.d
  n * (data.a * data.d - data.b * data.c)^2 / 
    ((data.a + data.b) * (data.c + data.d) * (data.a + data.c) * (data.b + data.d))

/-- Binomial probability calculation function --/
def binomialProb (n k : Nat) (p : Rat) : Rat :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- Main theorem --/
theorem survey_analysis (data : SurveyData) 
    (h_data : data.a = 140 ∧ data.b = 60 ∧ data.c = 180 ∧ data.d = 20) :
    chiSquare data = 25 ∧ 
    chiSquare data > (10828 : Rat) / 1000 ∧
    binomialProb 5 3 (4/5) = 128/625 := by
  sorry

#eval chiSquare ⟨140, 60, 180, 20⟩
#eval binomialProb 5 3 (4/5)

end NUMINAMATH_CALUDE_survey_analysis_l2536_253619


namespace NUMINAMATH_CALUDE_curve_C_parametric_equations_l2536_253609

/-- Given a curve C with polar equation ρ = 2cosθ, prove that its parametric equations are x = 1 + cosθ and y = sinθ -/
theorem curve_C_parametric_equations (θ : ℝ) :
  let ρ := 2 * Real.cos θ
  let x := 1 + Real.cos θ
  let y := Real.sin θ
  (x, y) ∈ {(x, y) : ℝ × ℝ | x^2 + y^2 = ρ^2 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ} :=
by sorry

end NUMINAMATH_CALUDE_curve_C_parametric_equations_l2536_253609


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l2536_253685

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Finds the point symmetric to a given point with respect to the xOz plane -/
def symmetricPointXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- The theorem stating that the point symmetric to (-3, -4, 5) with respect to the xOz plane is (-3, 4, 5) -/
theorem symmetric_point_theorem :
  let A : Point3D := { x := -3, y := -4, z := 5 }
  symmetricPointXOZ A = { x := -3, y := 4, z := 5 } := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_theorem_l2536_253685


namespace NUMINAMATH_CALUDE_angle_bisector_length_bound_l2536_253614

theorem angle_bisector_length_bound (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 15) (h3 : 0 < θ) (h4 : θ < π) :
  (2 * a * b * Real.cos (θ / 2)) / (a + b) < 12 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_length_bound_l2536_253614


namespace NUMINAMATH_CALUDE_existence_of_special_divisor_l2536_253660

theorem existence_of_special_divisor (n k : ℕ) (h1 : n > 1) (h2 : k = (Nat.factors n).card) :
  ∃ a : ℕ, 1 < a ∧ a < n / k + 1 ∧ n ∣ (a^2 - a) :=
sorry

end NUMINAMATH_CALUDE_existence_of_special_divisor_l2536_253660


namespace NUMINAMATH_CALUDE_mascot_purchase_equations_l2536_253620

/-- Represents the purchase of mascot dolls and keychains --/
structure MascotPurchase where
  dolls : ℕ
  keychains : ℕ
  total_cost : ℕ
  doll_price : ℕ
  keychain_price : ℕ

/-- The correct system of equations for the mascot purchase --/
def correct_equations (p : MascotPurchase) : Prop :=
  p.keychains = 2 * p.dolls ∧ 
  p.total_cost = p.doll_price * p.dolls + p.keychain_price * p.keychains

/-- Theorem stating the correct system of equations for the given conditions --/
theorem mascot_purchase_equations :
  ∀ (p : MascotPurchase), 
    p.total_cost = 5000 ∧ 
    p.doll_price = 60 ∧ 
    p.keychain_price = 20 →
    correct_equations p :=
by
  sorry


end NUMINAMATH_CALUDE_mascot_purchase_equations_l2536_253620


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2536_253647

theorem cos_150_degrees : Real.cos (150 * π / 180) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2536_253647


namespace NUMINAMATH_CALUDE_last_nonzero_digit_aperiodic_l2536_253652

/-- d_n is the last nonzero digit of n! -/
def last_nonzero_digit (n : ℕ) : ℕ :=
  sorry

/-- The sequence d_n is aperiodic -/
theorem last_nonzero_digit_aperiodic :
  ∀ T n₀ : ℕ, ∃ n : ℕ, n ≥ n₀ ∧ last_nonzero_digit (n + T) ≠ last_nonzero_digit n :=
sorry

end NUMINAMATH_CALUDE_last_nonzero_digit_aperiodic_l2536_253652


namespace NUMINAMATH_CALUDE_sum_le_product_plus_two_l2536_253624

theorem sum_le_product_plus_two (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) :
  x + y + z ≤ x * y * z + 2 := by
sorry

end NUMINAMATH_CALUDE_sum_le_product_plus_two_l2536_253624


namespace NUMINAMATH_CALUDE_hydra_disconnect_l2536_253690

/-- A graph representing a hydra -/
structure Hydra where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  vertex_count : vertices.card = 100

/-- Invert a vertex in the hydra -/
def invert_vertex (H : Hydra) (v : Nat) : Hydra :=
  sorry

/-- Check if the hydra is disconnected -/
def is_disconnected (H : Hydra) : Prop :=
  sorry

/-- Main theorem: Any 100-vertex hydra can be disconnected in at most 10 inversions -/
theorem hydra_disconnect (H : Hydra) :
  ∃ (inversions : List Nat), inversions.length ≤ 10 ∧
    is_disconnected (inversions.foldl invert_vertex H) :=
  sorry

end NUMINAMATH_CALUDE_hydra_disconnect_l2536_253690


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l2536_253655

/-- Given two rectangles S₁ and S₂ with specific vertices and equal areas, prove that 360x/y = 810 --/
theorem rectangle_area_ratio (x y : ℝ) (hx : x < 9) (hy : y < 4) 
  (h_equal_area : x * (4 - y) = y * (9 - x)) : 
  360 * x / y = 810 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l2536_253655


namespace NUMINAMATH_CALUDE_seaplane_speed_l2536_253666

theorem seaplane_speed (v : ℝ) (h1 : v > 0) : 
  (2 : ℝ) / ((1 / v) + (1 / 72)) = 91 → v = 6552 / 53 := by
  sorry

end NUMINAMATH_CALUDE_seaplane_speed_l2536_253666


namespace NUMINAMATH_CALUDE_tangent_and_inequality_l2536_253692

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - Real.log (x + 1) - a

theorem tangent_and_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀ = 0 ∧ f a x₀ = 0 ∧ (deriv (f a)) x₀ = 0) →
  (∀ x t : ℝ, x > t ∧ t ≥ 0 → Real.exp (x - t) + Real.log (t + 1) > Real.log (x + 1) + 1) ∧
  (f a = f 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_and_inequality_l2536_253692


namespace NUMINAMATH_CALUDE_no_real_solution_for_A_only_l2536_253653

theorem no_real_solution_for_A_only : 
  (¬ ∃ x : ℝ, (x - 3)^2 = -1) ∧ 
  (∃ x : ℝ, |x/2| - 6 = 0) ∧ 
  (∃ x : ℝ, x^2 + 8*x + 16 = 0) ∧ 
  (∃ x : ℝ, x + Real.sqrt (x - 5) = 0) ∧ 
  (∃ x : ℝ, Real.sqrt (-2*x - 10) = 3) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_for_A_only_l2536_253653


namespace NUMINAMATH_CALUDE_greatest_3digit_base9_div_by_7_l2536_253616

/-- Converts a base 9 number to base 10 -/
def base9ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 81 + ((n / 10) % 10) * 9 + (n % 10)

/-- Checks if a number is a 3-digit base 9 number -/
def isThreeDigitBase9 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 888

theorem greatest_3digit_base9_div_by_7 :
  ∀ n : ℕ, isThreeDigitBase9 n → (base9ToBase10 n) % 7 = 0 → n ≤ 888 :=
by sorry

end NUMINAMATH_CALUDE_greatest_3digit_base9_div_by_7_l2536_253616


namespace NUMINAMATH_CALUDE_triangle_problem_l2536_253617

-- Define a triangle with interior angles a, b, and x
structure Triangle where
  a : ℝ
  b : ℝ
  x : ℝ

-- Define the property of being an acute triangle
def isAcute (t : Triangle) : Prop :=
  t.a < 90 ∧ t.b < 90 ∧ t.x < 90

-- Theorem statement
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 60)
  (h2 : t.b = 70)
  (h3 : t.a + t.b + t.x = 180) : 
  t.x = 50 ∧ isAcute t := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2536_253617


namespace NUMINAMATH_CALUDE_find_A_minus_C_l2536_253638

theorem find_A_minus_C (A B C : ℕ) 
  (h1 : A + B = 84)
  (h2 : B + C = 60)
  (h3 : A = B + B + B + B + B + B)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C) :
  A - C = 24 := by
sorry

end NUMINAMATH_CALUDE_find_A_minus_C_l2536_253638


namespace NUMINAMATH_CALUDE_game_draw_fraction_l2536_253672

theorem game_draw_fraction (ben_win : ℚ) (tom_win : ℚ) (draw : ℚ) : 
  ben_win = 4/9 → tom_win = 1/3 → draw = 1 - (ben_win + tom_win) → draw = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_game_draw_fraction_l2536_253672


namespace NUMINAMATH_CALUDE_tangent_lines_through_point_l2536_253603

-- Define the curve
def f (x : ℝ) : ℝ := x^2

-- Define the point P
def P : ℝ × ℝ := (3, 5)

-- Define the two lines
def line1 (x : ℝ) : ℝ := 2*x - 1
def line2 (x : ℝ) : ℝ := 10*x - 25

theorem tangent_lines_through_point :
  ∀ m b : ℝ,
  (∃ x₀ : ℝ, 
    -- The line y = mx + b passes through P(3, 5)
    m * 3 + b = 5 ∧
    -- The line is tangent to the curve at some point (x₀, f(x₀))
    m * x₀ + b = f x₀ ∧
    m = 2 * x₀) →
  ((∀ x, m * x + b = line1 x) ∨ (∀ x, m * x + b = line2 x)) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_through_point_l2536_253603


namespace NUMINAMATH_CALUDE_circle_radius_squared_l2536_253618

theorem circle_radius_squared (r : ℝ) 
  (AB CD : ℝ) (angle_APD : ℝ) (BP : ℝ) : 
  AB = 10 → 
  CD = 7 → 
  angle_APD = 60 * π / 180 → 
  BP = 8 → 
  r^2 = 73 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_squared_l2536_253618


namespace NUMINAMATH_CALUDE_essay_time_theorem_l2536_253635

/-- Represents the time spent on various activities during essay writing -/
structure EssayWritingTime where
  wordsPerPage : ℕ
  timePerPageFirstDraft : ℕ
  researchTime : ℕ
  outlineTime : ℕ
  brainstormTime : ℕ
  firstDraftPages : ℕ
  timePerPageSecondDraft : ℕ
  breakTimePerPage : ℕ
  editingTime : ℕ
  proofreadingTime : ℕ

/-- Calculates the total time spent on writing the essay -/
def totalEssayTime (t : EssayWritingTime) : ℕ :=
  t.researchTime +
  t.outlineTime * 60 +
  t.brainstormTime +
  t.firstDraftPages * t.timePerPageFirstDraft +
  (t.firstDraftPages - 1) * t.breakTimePerPage +
  t.firstDraftPages * t.timePerPageSecondDraft +
  t.editingTime +
  t.proofreadingTime

/-- Theorem stating that the total time spent on the essay is 34900 seconds -/
theorem essay_time_theorem (t : EssayWritingTime)
  (h1 : t.wordsPerPage = 500)
  (h2 : t.timePerPageFirstDraft = 1800)
  (h3 : t.researchTime = 2700)
  (h4 : t.outlineTime = 15)
  (h5 : t.brainstormTime = 1200)
  (h6 : t.firstDraftPages = 6)
  (h7 : t.timePerPageSecondDraft = 1500)
  (h8 : t.breakTimePerPage = 600)
  (h9 : t.editingTime = 4500)
  (h10 : t.proofreadingTime = 1800) :
  totalEssayTime t = 34900 := by
  sorry

#eval totalEssayTime {
  wordsPerPage := 500,
  timePerPageFirstDraft := 1800,
  researchTime := 2700,
  outlineTime := 15,
  brainstormTime := 1200,
  firstDraftPages := 6,
  timePerPageSecondDraft := 1500,
  breakTimePerPage := 600,
  editingTime := 4500,
  proofreadingTime := 1800
}

end NUMINAMATH_CALUDE_essay_time_theorem_l2536_253635


namespace NUMINAMATH_CALUDE_third_number_proof_l2536_253669

theorem third_number_proof (a b c : ℝ) : 
  a = 6 → b = 16 → (a + b + c) / 3 = 13 → c = 17 := by
  sorry

end NUMINAMATH_CALUDE_third_number_proof_l2536_253669


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l2536_253634

theorem complex_exponential_sum (θ φ : ℝ) :
  Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1/3 : ℂ) + (2/5 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1/3 : ℂ) - (2/5 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l2536_253634


namespace NUMINAMATH_CALUDE_plane_stops_at_20_seconds_stop_time_unique_l2536_253684

/-- The distance function representing the plane's movement after landing -/
def s (t : ℝ) : ℝ := -1.5 * t^2 + 60 * t

/-- The time at which the plane stops -/
def stop_time : ℝ := 20

/-- Theorem stating that the plane stops at 20 seconds -/
theorem plane_stops_at_20_seconds :
  (∀ t : ℝ, t ≥ 0 → s t ≤ s stop_time) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ t, |t - stop_time| < δ → s t < s stop_time) := by
  sorry

/-- Corollary: The stop time is unique -/
theorem stop_time_unique (t : ℝ) :
  (∀ τ : ℝ, τ ≥ 0 → s τ ≤ s t) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ τ, |τ - t| < δ → s τ < s t) →
  t = stop_time := by
  sorry

end NUMINAMATH_CALUDE_plane_stops_at_20_seconds_stop_time_unique_l2536_253684


namespace NUMINAMATH_CALUDE_no_blonde_girls_added_l2536_253639

/-- The number of blonde girls added to a choir -/
def blonde_girls_added (initial_total : ℕ) (initial_blonde : ℕ) (black_haired : ℕ) : ℕ :=
  initial_total - initial_blonde - black_haired

/-- Theorem: Given the initial conditions, no blonde girls were added to the choir -/
theorem no_blonde_girls_added :
  blonde_girls_added 80 30 50 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_blonde_girls_added_l2536_253639


namespace NUMINAMATH_CALUDE_selection_theorem_l2536_253667

/-- The number of candidates --/
def n : ℕ := 5

/-- The number of languages --/
def k : ℕ := 3

/-- The number of candidates unwilling to study Hebrew --/
def m : ℕ := 2

/-- The number of ways to select students for the languages --/
def selection_methods : ℕ := (n - m) * (Nat.choose (n - 1) (k - 1)) * 2

theorem selection_theorem : selection_methods = 36 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l2536_253667


namespace NUMINAMATH_CALUDE_division_of_decimals_l2536_253689

theorem division_of_decimals : (0.36 : ℝ) / (0.004 : ℝ) = 90 := by sorry

end NUMINAMATH_CALUDE_division_of_decimals_l2536_253689


namespace NUMINAMATH_CALUDE_combination_equation_solution_l2536_253657

theorem combination_equation_solution (n : ℕ) : n ≥ 2 → (Nat.choose n 2 = Nat.choose (n - 1) 2 + Nat.choose (n - 1) 3) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l2536_253657


namespace NUMINAMATH_CALUDE_susan_chairs_l2536_253615

theorem susan_chairs (total : ℕ) (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 43 →
  red = 5 →
  yellow = 4 * red →
  total = red + yellow + blue →
  blue = 18 := by
sorry

end NUMINAMATH_CALUDE_susan_chairs_l2536_253615


namespace NUMINAMATH_CALUDE_origin_is_solution_l2536_253623

/-- The equation defining the set of points -/
def equation (x y : ℝ) : Prop :=
  x^2 * (y + y^2) = y^3 + x^4

/-- Theorem stating that (0, 0) is a solution to the equation -/
theorem origin_is_solution : equation 0 0 := by
  sorry

end NUMINAMATH_CALUDE_origin_is_solution_l2536_253623


namespace NUMINAMATH_CALUDE_average_score_is_1_9_l2536_253664

/-- Represents the score distribution of a class -/
structure ScoreDistribution where
  total_students : ℕ
  score_3_percent : ℚ
  score_2_percent : ℚ
  score_1_percent : ℚ
  score_0_percent : ℚ

/-- Calculates the average score of a class given its score distribution -/
def average_score (sd : ScoreDistribution) : ℚ :=
  (3 * sd.score_3_percent + 2 * sd.score_2_percent + sd.score_1_percent) * sd.total_students / 100

/-- The theorem stating that the average score for the given distribution is 1.9 -/
theorem average_score_is_1_9 (sd : ScoreDistribution)
  (h1 : sd.total_students = 30)
  (h2 : sd.score_3_percent = 30)
  (h3 : sd.score_2_percent = 40)
  (h4 : sd.score_1_percent = 20)
  (h5 : sd.score_0_percent = 10) :
  average_score sd = 19/10 := by sorry

end NUMINAMATH_CALUDE_average_score_is_1_9_l2536_253664


namespace NUMINAMATH_CALUDE_calculator_time_saved_l2536_253683

/-- Proves that using a calculator saves 150 minutes for Matt's math homework -/
theorem calculator_time_saved 
  (time_with_calc : ℕ) 
  (time_without_calc : ℕ) 
  (num_problems : ℕ) 
  (h1 : time_with_calc = 3)
  (h2 : time_without_calc = 8)
  (h3 : num_problems = 30) :
  time_without_calc * num_problems - time_with_calc * num_problems = 150 :=
by sorry

end NUMINAMATH_CALUDE_calculator_time_saved_l2536_253683


namespace NUMINAMATH_CALUDE_age_difference_l2536_253625

/-- Given information about Lexie and her siblings' ages, prove the age difference between her brother and sister. -/
theorem age_difference (lexie_age : ℕ) (brother_age_diff : ℕ) (sister_age_factor : ℕ) 
  (h1 : lexie_age = 8)
  (h2 : lexie_age = brother_age_diff + lexie_age - 6)
  (h3 : sister_age_factor * lexie_age = 2 * lexie_age) :
  2 * lexie_age - (lexie_age - 6) = 14 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2536_253625


namespace NUMINAMATH_CALUDE_percentage_of_75_to_125_l2536_253687

theorem percentage_of_75_to_125 : ∀ (x : ℝ), x = (75 : ℝ) / (125 : ℝ) * 100 → x = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_of_75_to_125_l2536_253687


namespace NUMINAMATH_CALUDE_determine_set_N_l2536_253671

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define subset M
def M : Set Nat := {1, 4}

-- Define the theorem
theorem determine_set_N (N : Set Nat) 
  (h1 : N ⊆ U)
  (h2 : M ∩ N = {1})
  (h3 : N ∩ (U \ M) = {3, 5}) :
  N = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_determine_set_N_l2536_253671


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l2536_253688

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_number_satisfying_conditions : 
  (∀ n : ℕ, n < 70 → ¬(is_multiple_of n 7 ∧ is_multiple_of n 5 ∧ is_prime (n + 9))) ∧ 
  (is_multiple_of 70 7 ∧ is_multiple_of 70 5 ∧ is_prime (70 + 9)) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l2536_253688


namespace NUMINAMATH_CALUDE_power_of_power_l2536_253696

theorem power_of_power (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2536_253696


namespace NUMINAMATH_CALUDE_number_of_factors_of_b_power_n_l2536_253663

def b : ℕ := 6
def n : ℕ := 15

theorem number_of_factors_of_b_power_n : 
  b ≤ 15 → n ≤ 15 → (Nat.factors (b^n)).length + 1 = 256 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_b_power_n_l2536_253663


namespace NUMINAMATH_CALUDE_smallest_linear_combination_l2536_253654

theorem smallest_linear_combination (m n : ℤ) : 
  ∃ (k : ℕ), k > 0 ∧ (∃ (a b : ℤ), k = 2024 * a + 48048 * b) ∧ 
  (∀ (l : ℕ) (c d : ℤ), l > 0 ∧ l = 2024 * c + 48048 * d → k ≤ l) ∧ 
  k = 88 := by
sorry

end NUMINAMATH_CALUDE_smallest_linear_combination_l2536_253654


namespace NUMINAMATH_CALUDE_parabola_segment_length_l2536_253644

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus and directrix
def focus : ℝ × ℝ := (2, 0)
def directrix : ℝ → Prop := λ x => x = -2

-- Define a point on the directrix
def point_on_directrix (P : ℝ × ℝ) : Prop :=
  directrix P.1

-- Define points on the parabola
def point_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

-- Define the condition PF = 3MF
def vector_condition (P M : ℝ × ℝ) : Prop :=
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 
  9 * ((M.1 - focus.1)^2 + (M.2 - focus.2)^2)

-- State the theorem
theorem parabola_segment_length 
  (P M N : ℝ × ℝ) 
  (h1 : point_on_directrix P) 
  (h2 : point_on_parabola M) 
  (h3 : point_on_parabola N) 
  (h4 : vector_condition P M) :
  let MN_length := Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2)
  MN_length = 32/3 := by sorry

end NUMINAMATH_CALUDE_parabola_segment_length_l2536_253644


namespace NUMINAMATH_CALUDE_article_price_reduction_l2536_253612

/-- Proves that given an article with an original cost of 50, sold at a 25% profit,
    if the selling price is reduced by 10.50 and the profit becomes 30%,
    then the reduction in the buying price is 20%. -/
theorem article_price_reduction (original_cost : ℝ) (original_profit_percent : ℝ)
  (price_reduction : ℝ) (new_profit_percent : ℝ) :
  original_cost = 50 →
  original_profit_percent = 25 →
  price_reduction = 10.50 →
  new_profit_percent = 30 →
  ∃ (buying_price_reduction : ℝ),
    buying_price_reduction = 20 ∧
    (original_cost * (1 + original_profit_percent / 100) - price_reduction) =
    (original_cost * (1 - buying_price_reduction / 100)) * (1 + new_profit_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_article_price_reduction_l2536_253612


namespace NUMINAMATH_CALUDE_f_maximum_and_a_range_l2536_253694

/-- The function f(x) = |x+1| - |x-4| - a -/
def f (x a : ℝ) : ℝ := |x + 1| - |x - 4| - a

theorem f_maximum_and_a_range :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x a ≤ f x_max a) ∧
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x a ≤ 5 - a) ∧
  (∃ (x : ℝ), f x a ≥ 4/a + 1 → (a = 2 ∨ a < 0)) := by
  sorry

end NUMINAMATH_CALUDE_f_maximum_and_a_range_l2536_253694


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2536_253608

open Set

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

-- Define the intersection of A and B
def A_intersect_B : Set ℝ := A ∩ B

-- Theorem statement
theorem intersection_of_A_and_B :
  A_intersect_B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2536_253608


namespace NUMINAMATH_CALUDE_abc_remainder_mod_9_l2536_253636

theorem abc_remainder_mod_9 (a b c : ℕ) 
  (ha : a < 9) (hb : b < 9) (hc : c < 9)
  (cong1 : a + 2*b + 3*c ≡ 0 [ZMOD 9])
  (cong2 : 2*a + 3*b + c ≡ 5 [ZMOD 9])
  (cong3 : 3*a + b + 2*c ≡ 5 [ZMOD 9]) :
  a * b * c ≡ 0 [ZMOD 9] := by
sorry

end NUMINAMATH_CALUDE_abc_remainder_mod_9_l2536_253636


namespace NUMINAMATH_CALUDE_constant_sum_product_l2536_253676

theorem constant_sum_product (n : Nat) (h : n = 15) : 
  ∃ k : Nat, ∀ (operations : List (Nat × Nat)), 
    operations.length = n - 1 → 
    (∀ (x y : Nat), (x, y) ∈ operations → x ≤ n ∧ y ≤ n) →
    (List.foldl (λ acc (x, y) => acc + x * y * (x + y)) 0 operations) = k ∧ k = 49140 := by
  sorry

end NUMINAMATH_CALUDE_constant_sum_product_l2536_253676


namespace NUMINAMATH_CALUDE_loop_execution_l2536_253648

theorem loop_execution (n α : ℕ) (β : ℚ) : 
  β = (n - 1) / 2^α →
  ∃ (ℓ m : ℕ → ℚ),
    (ℓ 0 = 0 ∧ m 0 = n - 1) ∧
    (∀ k, k < α → ℓ (k + 1) = ℓ k + 1 ∧ m (k + 1) = m k / 2) →
    ℓ α = α ∧ m α = β :=
sorry

end NUMINAMATH_CALUDE_loop_execution_l2536_253648


namespace NUMINAMATH_CALUDE_twenty_percent_women_without_plan_l2536_253629

/-- Represents a company with workers and their retirement plan status -/
structure Company where
  total_workers : ℕ
  workers_without_plan : ℕ
  men_with_plan : ℕ
  total_men : ℕ
  total_women : ℕ

/-- Conditions for the company -/
def company_conditions (c : Company) : Prop :=
  c.workers_without_plan = c.total_workers / 3 ∧
  c.men_with_plan = (c.total_workers - c.workers_without_plan) * 2 / 5 ∧
  c.total_men = 112 ∧
  c.total_women = 98 ∧
  c.total_workers = c.total_men + c.total_women

/-- The percentage of women without a retirement plan -/
def women_without_plan_percentage (c : Company) : ℚ :=
  let women_without_plan := c.workers_without_plan - (c.total_men - c.men_with_plan)
  (women_without_plan : ℚ) / c.workers_without_plan * 100

/-- Theorem stating that 20% of workers without a retirement plan are women -/
theorem twenty_percent_women_without_plan (c : Company) 
  (h : company_conditions c) : women_without_plan_percentage c = 20 := by
  sorry


end NUMINAMATH_CALUDE_twenty_percent_women_without_plan_l2536_253629


namespace NUMINAMATH_CALUDE_rocket_heights_l2536_253601

theorem rocket_heights (h1 : ℝ) (h2 : ℝ) (height1 : h1 = 500) (height2 : h2 = 2 * h1) :
  h1 + h2 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_rocket_heights_l2536_253601


namespace NUMINAMATH_CALUDE_bisection_method_termination_condition_l2536_253662

/-- The bisection method termination condition -/
def bisection_termination (x₁ x₂ ε : ℝ) : Prop :=
  |x₁ - x₂| < ε

/-- Theorem stating the correct termination condition for the bisection method -/
theorem bisection_method_termination_condition 
  (f : ℝ → ℝ) (a b x₁ x₂ ε : ℝ) 
  (hf : Continuous f) 
  (ha : f a < 0) 
  (hb : f b > 0) 
  (hε : ε > 0) 
  (hx₁ : x₁ ∈ Set.Icc a b) 
  (hx₂ : x₂ ∈ Set.Icc a b) :
  bisection_termination x₁ x₂ ε ↔ 
    (∃ x ∈ Set.Icc x₁ x₂, f x = 0) ∧ 
    (∀ y ∈ Set.Icc a b, f y = 0 → y ∈ Set.Icc x₁ x₂) := by
  sorry


end NUMINAMATH_CALUDE_bisection_method_termination_condition_l2536_253662


namespace NUMINAMATH_CALUDE_hilton_final_marbles_l2536_253604

/-- Calculates the final number of marbles Hilton has after a series of events -/
def hiltons_marbles (initial : ℕ) (found : ℕ) (lost : ℕ) : ℕ :=
  initial + found - lost + 2 * lost

/-- Theorem stating that given the initial conditions, Hilton ends up with 42 marbles -/
theorem hilton_final_marbles :
  hiltons_marbles 26 6 10 = 42 := by
  sorry

end NUMINAMATH_CALUDE_hilton_final_marbles_l2536_253604


namespace NUMINAMATH_CALUDE_train_length_l2536_253643

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 225 →
  train_speed * crossing_time - bridge_length = 150 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2536_253643


namespace NUMINAMATH_CALUDE_factorization_proof_l2536_253605

theorem factorization_proof (z : ℝ) : 
  75 * z^12 + 162 * z^24 + 27 = 3 * (9 + z^12 * (25 + 54 * z^12)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2536_253605


namespace NUMINAMATH_CALUDE_sum_25887_2014_not_even_l2536_253651

theorem sum_25887_2014_not_even : ¬ Even (25887 + 2014) := by
  sorry

end NUMINAMATH_CALUDE_sum_25887_2014_not_even_l2536_253651


namespace NUMINAMATH_CALUDE_geometric_sum_five_terms_l2536_253697

theorem geometric_sum_five_terms :
  (1 / 5 : ℚ) - (1 / 25 : ℚ) + (1 / 125 : ℚ) - (1 / 625 : ℚ) + (1 / 3125 : ℚ) = 521 / 3125 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sum_five_terms_l2536_253697


namespace NUMINAMATH_CALUDE_tan_function_property_l2536_253675

open Real

theorem tan_function_property (φ a : ℝ) (h1 : π / 2 < φ) (h2 : φ < 3 * π / 2) : 
  let f := fun x => tan (φ - x)
  (f 0 = 0) → (f (-a) = 1 / 2) → (f (a + π / 4) = -3) := by
  sorry

end NUMINAMATH_CALUDE_tan_function_property_l2536_253675


namespace NUMINAMATH_CALUDE_min_sum_squares_l2536_253632

theorem min_sum_squares (x₁ x₂ x₃ : ℝ) (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0) 
  (h_sum : x₁ + 2*x₂ + 3*x₃ = 120) : 
  ∀ y₁ y₂ y₃ : ℝ, y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0 → y₁ + 2*y₂ + 3*y₃ = 120 → 
  x₁^2 + x₂^2 + x₃^2 ≤ y₁^2 + y₂^2 + y₃^2 ∧ 
  ∃ x₁' x₂' x₃' : ℝ, x₁'^2 + x₂'^2 + x₃'^2 = 1400 ∧ 
                    x₁' > 0 ∧ x₂' > 0 ∧ x₃' > 0 ∧ 
                    x₁' + 2*x₂' + 3*x₃' = 120 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2536_253632


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2536_253680

theorem absolute_value_inequality_solution_set :
  {x : ℝ | 1 < |1 - x| ∧ |1 - x| ≤ 2} = Set.Icc (-1) 0 ∪ Set.Ioc 2 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2536_253680


namespace NUMINAMATH_CALUDE_set_A_determination_l2536_253665

universe u

def U : Set ℕ := {1, 2, 3, 4}

theorem set_A_determination (A : Set ℕ) 
  (h1 : A ⊆ U)
  (h2 : A ∩ {1, 2, 3} = {2})
  (h3 : A ∪ {1, 2, 3} = U) :
  A = {2, 4} := by
sorry


end NUMINAMATH_CALUDE_set_A_determination_l2536_253665


namespace NUMINAMATH_CALUDE_largest_last_digit_l2536_253679

/-- A string of digits satisfying the problem conditions -/
def ValidString : Type := 
  {s : List Nat // s.length = 2007 ∧ s.head! = 2 ∧ 
    ∀ i, i < 2006 → (s.get! i * 10 + s.get! (i+1)) % 23 = 0 ∨ 
                     (s.get! i * 10 + s.get! (i+1)) % 37 = 0}

/-- The theorem stating the largest possible last digit -/
theorem largest_last_digit (s : ValidString) : s.val.getLast! ≤ 9 :=
sorry

end NUMINAMATH_CALUDE_largest_last_digit_l2536_253679


namespace NUMINAMATH_CALUDE_westville_summer_retreat_soccer_percentage_l2536_253607

theorem westville_summer_retreat_soccer_percentage 
  (total : ℝ) 
  (soccer_percentage : ℝ) 
  (swim_percentage : ℝ) 
  (soccer_and_swim_percentage : ℝ) 
  (basketball_percentage : ℝ) 
  (basketball_soccer_no_swim_percentage : ℝ) 
  (h1 : soccer_percentage = 0.7) 
  (h2 : swim_percentage = 0.5) 
  (h3 : soccer_and_swim_percentage = 0.3 * soccer_percentage) 
  (h4 : basketball_percentage = 0.2) 
  (h5 : basketball_soccer_no_swim_percentage = 0.25 * basketball_percentage) : 
  (soccer_percentage * total - soccer_and_swim_percentage * total - basketball_soccer_no_swim_percentage * total) / 
  ((1 - swim_percentage) * total) = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_westville_summer_retreat_soccer_percentage_l2536_253607


namespace NUMINAMATH_CALUDE_rectangular_hall_area_l2536_253682

theorem rectangular_hall_area (length width : ℝ) : 
  width = length / 2 →
  length - width = 8 →
  length * width = 128 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_hall_area_l2536_253682


namespace NUMINAMATH_CALUDE_intersection_line_through_origin_l2536_253686

/-- Given two lines l₁ and l₂ in the plane, prove that the line passing through
    their intersection point and the origin has the equation x - 10y = 0. -/
theorem intersection_line_through_origin
  (l₁ : Set (ℝ × ℝ))
  (l₂ : Set (ℝ × ℝ))
  (h₁ : l₁ = {(x, y) | 2 * x + y = 3})
  (h₂ : l₂ = {(x, y) | x + 4 * y = 2})
  (P : ℝ × ℝ)
  (hP : P ∈ l₁ ∧ P ∈ l₂)
  (l : Set (ℝ × ℝ))
  (hl : l = {(x, y) | ∃ t : ℝ, x = t * P.1 ∧ y = t * P.2}) :
  l = {(x, y) | x - 10 * y = 0} :=
sorry

end NUMINAMATH_CALUDE_intersection_line_through_origin_l2536_253686


namespace NUMINAMATH_CALUDE_cost_per_serving_is_one_dollar_l2536_253656

/-- The cost of a serving of spaghetti and meatballs -/
def cost_per_serving (pasta_cost sauce_cost meatballs_cost : ℚ) (num_servings : ℕ) : ℚ :=
  (pasta_cost + sauce_cost + meatballs_cost) / num_servings

/-- Theorem: The cost per serving is $1.00 -/
theorem cost_per_serving_is_one_dollar :
  cost_per_serving 1 2 5 8 = 1 := by sorry

end NUMINAMATH_CALUDE_cost_per_serving_is_one_dollar_l2536_253656


namespace NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l2536_253626

/-- An arithmetic sequence with sum of first n terms Sn -/
structure ArithmeticSequence where
  S : ℕ → ℤ  -- Sum function

/-- Theorem: For an arithmetic sequence, if S_{m-1} = -2, S_m = 0, and S_{m+1} = 3, then m = 5 -/
theorem arithmetic_sequence_m_value (seq : ArithmeticSequence) (m : ℕ) 
  (h1 : seq.S (m - 1) = -2)
  (h2 : seq.S m = 0)
  (h3 : seq.S (m + 1) = 3) :
  m = 5 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_m_value_l2536_253626


namespace NUMINAMATH_CALUDE_max_value_trig_product_l2536_253678

theorem max_value_trig_product (x y z : ℝ) :
  (Real.sin (2*x) + Real.sin y + Real.sin (3*z)) * 
  (Real.cos (2*x) + Real.cos y + Real.cos (3*z)) ≤ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_product_l2536_253678


namespace NUMINAMATH_CALUDE_spending_ratio_l2536_253691

/-- Represents the spending of Lisa and Carly -/
structure Spending where
  lisa_tshirt : ℝ
  lisa_jeans : ℝ
  lisa_coat : ℝ
  carly_tshirt : ℝ
  carly_jeans : ℝ
  carly_coat : ℝ

/-- The theorem representing the problem -/
theorem spending_ratio (s : Spending) : 
  s.lisa_tshirt = 40 →
  s.lisa_jeans = s.lisa_tshirt / 2 →
  s.carly_tshirt = s.lisa_tshirt / 4 →
  s.carly_jeans = 3 * s.lisa_jeans →
  s.carly_coat = s.lisa_coat / 4 →
  s.lisa_tshirt + s.lisa_jeans + s.lisa_coat + 
  s.carly_tshirt + s.carly_jeans + s.carly_coat = 230 →
  s.lisa_coat / s.lisa_tshirt = 2 := by
  sorry

end NUMINAMATH_CALUDE_spending_ratio_l2536_253691


namespace NUMINAMATH_CALUDE_triangle_properties_l2536_253646

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.C)
  (h2 : t.a^2 - t.c^2 = 2 * t.b^2)
  (h3 : (1/2) * t.a * t.b * Real.sin t.C = 21 * Real.sqrt 3) :
  t.C = π/3 ∧ t.b = 2 * Real.sqrt 7 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l2536_253646


namespace NUMINAMATH_CALUDE_simplify_polynomial_subtraction_l2536_253631

theorem simplify_polynomial_subtraction (r : ℝ) : 
  (r^2 + 3*r - 2) - (r^2 + 7*r - 5) = -4*r + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_subtraction_l2536_253631


namespace NUMINAMATH_CALUDE_inequality_implication_l2536_253695

theorem inequality_implication (x y : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_ineq : 4 * Real.log x + 2 * Real.log (2 * y) ≥ x^2 + 8 * y - 4) : 
  x * y = Real.sqrt 2 / 4 ∧ x + 2 * y = 1 / 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l2536_253695


namespace NUMINAMATH_CALUDE_initial_fee_correct_l2536_253611

/-- The initial fee for Jim's taxi service -/
def initial_fee : ℝ := 2.25

/-- The charge per 2/5 mile segment -/
def charge_per_segment : ℝ := 0.35

/-- The length of a trip in miles -/
def trip_length : ℝ := 3.6

/-- The total charge for the trip -/
def total_charge : ℝ := 5.4

/-- Theorem stating that the initial fee is correct given the conditions -/
theorem initial_fee_correct : 
  initial_fee + (trip_length / (2/5) * charge_per_segment) = total_charge :=
by sorry

end NUMINAMATH_CALUDE_initial_fee_correct_l2536_253611


namespace NUMINAMATH_CALUDE_circle_arrangement_impossibility_l2536_253670

theorem circle_arrangement_impossibility :
  ¬ ∃ (a : Fin 7 → ℕ),
    (∀ i : Fin 7, ∃ j : Fin 7, (a j + a ((j + 1) % 7) + a ((j + 2) % 7) = i + 1)) ∧
    (∀ i j : Fin 7, i ≠ j → 
      (a i + a ((i + 1) % 7) + a ((i + 2) % 7)) ≠ (a j + a ((j + 1) % 7) + a ((j + 2) % 7))) :=
by sorry

end NUMINAMATH_CALUDE_circle_arrangement_impossibility_l2536_253670


namespace NUMINAMATH_CALUDE_remainder_problem_l2536_253630

theorem remainder_problem (n : ℤ) (h : n % 6 = 1) : (3 * (n + 1812)) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2536_253630


namespace NUMINAMATH_CALUDE_triangle_side_length_l2536_253659

theorem triangle_side_length (A B C : ℝ) (AB BC AC : ℝ) : 
  A = π / 3 →
  Real.tan B = 1 / 2 →
  AB = 2 * Real.sqrt 3 + 1 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  AB > 0 ∧ BC > 0 ∧ AC > 0 →
  AC / Real.sin B = AB / Real.sin C →
  BC / Real.sin A = AB / Real.sin C →
  BC = Real.sqrt 15 := by
sorry


end NUMINAMATH_CALUDE_triangle_side_length_l2536_253659


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2536_253698

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) =
  (x^2 + 6*x + 5) * (x^2 + 6*x + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2536_253698


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2536_253641

-- Define the function f
def f (b c x : ℝ) : ℝ := 2 * x^2 + b * x + c

-- State the theorem
theorem min_value_of_reciprocal_sum (b c : ℝ) (x₁ x₂ : ℝ) :
  (∃ (b c : ℝ), f b c (-10) = f b c 12) →  -- f(-10) = f(12)
  (x₁ > 0 ∧ x₂ > 0) →  -- x₁ and x₂ are positive
  (f b c x₁ = 0 ∧ f b c x₂ = 0) →  -- x₁ and x₂ are roots of f(x) = 0
  (∀ y z : ℝ, y > 0 ∧ z > 0 ∧ f b c y = 0 ∧ f b c z = 0 → 1/y + 1/z ≥ 1/x₁ + 1/x₂) →  -- x₁ and x₂ give the minimum value
  1/x₁ + 1/x₂ = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l2536_253641


namespace NUMINAMATH_CALUDE_calculation_proof_l2536_253668

theorem calculation_proof : 3.6 * 0.25 + 1.5 = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2536_253668


namespace NUMINAMATH_CALUDE_large_ball_uses_300_rubber_bands_l2536_253658

/-- Calculates the number of rubber bands used in a large ball -/
def large_ball_rubber_bands (total_rubber_bands : ℕ) (small_balls : ℕ) (rubber_bands_per_small : ℕ) (large_balls : ℕ) : ℕ :=
  (total_rubber_bands - small_balls * rubber_bands_per_small) / large_balls

/-- Proves that a large ball uses 300 rubber bands given the problem conditions -/
theorem large_ball_uses_300_rubber_bands :
  large_ball_rubber_bands 5000 22 50 13 = 300 := by
  sorry

end NUMINAMATH_CALUDE_large_ball_uses_300_rubber_bands_l2536_253658


namespace NUMINAMATH_CALUDE_alpha_value_l2536_253606

theorem alpha_value (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.tan (α + β) = 3 →
  Real.tan β = 1/2 →
  α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l2536_253606
