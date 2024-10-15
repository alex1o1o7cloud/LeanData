import Mathlib

namespace NUMINAMATH_CALUDE_min_value_expression_l2027_202734

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x * y = 1) :
  (x + 2 * y) * (2 * x + z) * (y + 2 * z) ≥ 48 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ = 1 ∧
    (x₀ + 2 * y₀) * (2 * x₀ + z₀) * (y₀ + 2 * z₀) = 48 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2027_202734


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2027_202750

theorem inequality_solution_set (a b m : ℝ) : 
  (∀ x, x^2 - a*x - 2 > 0 ↔ (x < -1 ∨ x > b)) →
  b > -1 →
  m > -1/2 →
  (a = 1 ∧ b = 2) ∧
  (
    (m > 0 → ∀ x, (m*x + a)*(x - b) > 0 ↔ (x < -1/m ∨ x > 2)) ∧
    (m = 0 → ∀ x, (m*x + a)*(x - b) > 0 ↔ x > 2) ∧
    (-1/2 < m ∧ m < 0 → ∀ x, (m*x + a)*(x - b) > 0 ↔ (2 < x ∧ x < -1/m))
  ) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2027_202750


namespace NUMINAMATH_CALUDE_one_positive_real_solution_l2027_202731

def f (x : ℝ) := x^4 + 8*x^3 + 16*x^2 + 2023*x - 2023

theorem one_positive_real_solution :
  ∃! x : ℝ, x > 0 ∧ x^10 + 8*x^9 + 16*x^8 + 2023*x^7 - 2023*x^6 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_one_positive_real_solution_l2027_202731


namespace NUMINAMATH_CALUDE_melted_spheres_radius_l2027_202710

theorem melted_spheres_radius (r : ℝ) : r > 0 → (4 / 3 * Real.pi * r^3 = 8 / 3 * Real.pi) → r = 2^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_melted_spheres_radius_l2027_202710


namespace NUMINAMATH_CALUDE_parabola_equation_l2027_202770

/-- A parabola is a set of points equidistant from a fixed point (focus) and a fixed line (directrix) -/
def Parabola (focus : ℝ × ℝ) (directrix : ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | dist p focus = |p.2 - directrix p.1|}

theorem parabola_equation (p : ℝ × ℝ) :
  p ∈ Parabola (0, -1) (fun _ ↦ 1) ↔ p.1^2 = -4 * p.2 := by
  sorry

#check parabola_equation

end NUMINAMATH_CALUDE_parabola_equation_l2027_202770


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2027_202708

/-- Represents a triangle divided into three triangles and one quadrilateral -/
structure DividedTriangle where
  -- Areas of the three triangles
  area1 : ℝ
  area2 : ℝ
  area3 : ℝ
  -- Area of the quadrilateral
  quad_area : ℝ

/-- Theorem stating that if the areas of the three triangles are 6, 9, and 15,
    then the area of the quadrilateral is 65 -/
theorem quadrilateral_area (t : DividedTriangle) :
  t.area1 = 6 ∧ t.area2 = 9 ∧ t.area3 = 15 → t.quad_area = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2027_202708


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2027_202744

-- Problem 1
theorem problem_1 (x : ℝ) : (-2*x)^2 + 3*x*x = 7*x^2 := by sorry

-- Problem 2
theorem problem_2 (m a b : ℝ) : m*a^2 - m*b^2 = m*(a - b)*(a + b) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2027_202744


namespace NUMINAMATH_CALUDE_remainder_problem_l2027_202738

theorem remainder_problem (N : ℤ) : 
  (∃ k : ℤ, N = 45 * k + 31) → (∃ m : ℤ, N = 15 * m + 1) :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l2027_202738


namespace NUMINAMATH_CALUDE_f_simplification_g_definition_g_value_at_pi_over_6_l2027_202716

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * sin (π - x) * sin x - (sin x - cos x)^2

noncomputable def g (x : ℝ) : ℝ := 2 * sin x + Real.sqrt 3 - 1

theorem f_simplification (x : ℝ) : f x = 2 * sin (2*x - π/3) + Real.sqrt 3 - 1 := by sorry

theorem g_definition (x : ℝ) : g x = 2 * sin x + Real.sqrt 3 - 1 := by sorry

theorem g_value_at_pi_over_6 : g (π/6) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_f_simplification_g_definition_g_value_at_pi_over_6_l2027_202716


namespace NUMINAMATH_CALUDE_cone_cylinder_equal_volume_l2027_202778

/-- Given a cylinder M with base radius 2 and height 2√3/3, and a cone N whose base diameter
    equals its slant height, if the volumes of M and N are equal, then the base radius of cone N is 2. -/
theorem cone_cylinder_equal_volume (r : ℝ) : 
  let cylinder_volume := π * 2^2 * (2 * Real.sqrt 3 / 3)
  let cone_volume := (1/3) * π * r^2 * (Real.sqrt 3 * r)
  (2 * r = Real.sqrt 3 * r) → (cylinder_volume = cone_volume) → r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_cylinder_equal_volume_l2027_202778


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l2027_202723

/-- Proves that two successive discounts are equivalent to a single discount --/
theorem successive_discounts_equivalence (original_price : ℝ) 
  (first_discount second_discount : ℝ) :
  original_price = 800 ∧ 
  first_discount = 0.15 ∧ 
  second_discount = 0.10 →
  let price_after_first := original_price * (1 - first_discount)
  let final_price := price_after_first * (1 - second_discount)
  let equivalent_discount := (original_price - final_price) / original_price
  equivalent_discount = 0.235 := by
  sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l2027_202723


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2027_202768

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x + 1

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 3

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 0)

/-- Theorem: The equation of the tangent line to y = 2x^3 - 3x + 1 at (1, 0) is 3x - y - 3 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (x, y) ∈ {(x, y) | 3 * x - y - 3 = 0} ↔
  y - point.2 = f' point.1 * (x - point.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2027_202768


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2027_202715

theorem unique_three_digit_number : ∃! (n : ℕ), 
  (100 ≤ n ∧ n < 1000) ∧ 
  (∃ (π b γ : ℕ), 
    π ≠ b ∧ π ≠ γ ∧ b ≠ γ ∧
    π < 10 ∧ b < 10 ∧ γ < 10 ∧
    n = 100 * π + 10 * b + γ ∧
    n = (π + b + γ) * (π + b + γ + 1)) ∧
  n = 156 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2027_202715


namespace NUMINAMATH_CALUDE_area_at_stage_4_is_1360_l2027_202773

/-- The area of the figure at stage n, given an initial square of side length 4 inches -/
def area_at_stage (n : ℕ) : ℕ :=
  let initial_side := 4
  let rec sum_areas (k : ℕ) (acc : ℕ) : ℕ :=
    if k = 0 then acc
    else sum_areas (k - 1) (acc + (initial_side * 2^(k - 1))^2)
  sum_areas n 0

/-- The theorem stating that the area at stage 4 is 1360 square inches -/
theorem area_at_stage_4_is_1360 : area_at_stage 4 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_4_is_1360_l2027_202773


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l2027_202764

theorem geometric_sequence_problem (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1/4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l2027_202764


namespace NUMINAMATH_CALUDE_weight_of_CaI2_l2027_202753

/-- The atomic weight of calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The number of calcium atoms in CaI2 -/
def num_Ca_atoms : ℕ := 1

/-- The number of iodine atoms in CaI2 -/
def num_I_atoms : ℕ := 2

/-- The number of moles of CaI2 -/
def num_moles : ℝ := 3

/-- The molecular weight of CaI2 in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca * num_Ca_atoms + atomic_weight_I * num_I_atoms

/-- The total weight of CaI2 in grams -/
def weight_CaI2 : ℝ := molecular_weight_CaI2 * num_moles

theorem weight_of_CaI2 : weight_CaI2 = 881.64 := by sorry

end NUMINAMATH_CALUDE_weight_of_CaI2_l2027_202753


namespace NUMINAMATH_CALUDE_smallest_x_y_sum_l2027_202795

def is_fourth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^4

def is_sixth_power (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^6

theorem smallest_x_y_sum :
  ∃ (x y : ℕ),
    (∀ x' : ℕ, is_fourth_power (180 * x') → x ≤ x') ∧
    (∀ y' : ℕ, is_sixth_power (180 * y') → y ≤ y') ∧
    is_fourth_power (180 * x) ∧
    is_sixth_power (180 * y) ∧
    x + y = 4054500 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_y_sum_l2027_202795


namespace NUMINAMATH_CALUDE_square_fence_perimeter_l2027_202771

theorem square_fence_perimeter 
  (num_posts : ℕ) 
  (post_width_inches : ℝ) 
  (gap_between_posts_feet : ℝ) : 
  num_posts = 36 →
  post_width_inches = 6 →
  gap_between_posts_feet = 8 →
  let posts_per_side : ℕ := num_posts / 4
  let gaps_per_side : ℕ := posts_per_side - 1
  let total_gap_length : ℝ := (gaps_per_side : ℝ) * gap_between_posts_feet
  let post_width_feet : ℝ := post_width_inches / 12
  let total_post_width : ℝ := (posts_per_side : ℝ) * post_width_feet
  let side_length : ℝ := total_gap_length + total_post_width
  let perimeter : ℝ := 4 * side_length
  perimeter = 242 := by
sorry

end NUMINAMATH_CALUDE_square_fence_perimeter_l2027_202771


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2027_202742

def is_divisible_by_all (n : ℕ) : Prop :=
  (n - 5) % 12 = 0 ∧
  (n - 5) % 16 = 0 ∧
  (n - 5) % 18 = 0 ∧
  (n - 5) % 21 = 0 ∧
  (n - 5) % 28 = 0

theorem smallest_number_divisible_by_all :
  ∀ m : ℕ, m < 1013 → ¬(is_divisible_by_all m) ∧ is_divisible_by_all 1013 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l2027_202742


namespace NUMINAMATH_CALUDE_fixed_point_power_function_l2027_202736

-- Define the conditions
variable (a : ℝ) (ha : a > 0) (hna : a ≠ 1)
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (hf1 : f (Real.sqrt 2) = 2)
variable (hf2 : ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α)

-- State the theorem
theorem fixed_point_power_function : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_power_function_l2027_202736


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2027_202761

theorem trigonometric_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + 2 * y) ^ 2 - 2 * Real.sin x * Real.sin (2 * y) * Real.cos (x + 2 * y) =
  2 * Real.sin x ^ 2 - Real.sin x ^ 2 * Real.cos y ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2027_202761


namespace NUMINAMATH_CALUDE_scale_division_l2027_202717

/-- Given a scale of length 188 inches divided into 8 equal parts, 
    the length of each part is 23.5 inches. -/
theorem scale_division (total_length : ℝ) (num_parts : ℕ) 
  (h1 : total_length = 188) 
  (h2 : num_parts = 8) :
  total_length / num_parts = 23.5 := by
  sorry

end NUMINAMATH_CALUDE_scale_division_l2027_202717


namespace NUMINAMATH_CALUDE_greatest_solution_quadratic_l2027_202703

theorem greatest_solution_quadratic : 
  ∃ (x : ℝ), x = 4/5 ∧ 5*x^2 - 3*x - 4 = 0 ∧ 
  ∀ (y : ℝ), 5*y^2 - 3*y - 4 = 0 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_greatest_solution_quadratic_l2027_202703


namespace NUMINAMATH_CALUDE_min_correct_answers_l2027_202766

theorem min_correct_answers (total : Nat) (a b c d : Nat)
  (h_total : total = 15)
  (h_a : a = 11)
  (h_b : b = 12)
  (h_c : c = 13)
  (h_d : d = 14)
  (h_a_le : a ≤ total)
  (h_b_le : b ≤ total)
  (h_c_le : c ≤ total)
  (h_d_le : d ≤ total) :
  ∃ (x : Nat), x = min a (min b (min c d)) ∧ x ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_min_correct_answers_l2027_202766


namespace NUMINAMATH_CALUDE_gcd_upper_bound_from_lcm_lower_bound_l2027_202796

theorem gcd_upper_bound_from_lcm_lower_bound 
  (a b : ℕ) 
  (ha : a < 10^7) 
  (hb : b < 10^7) 
  (hlcm : 10^11 ≤ Nat.lcm a b) : 
  Nat.gcd a b < 1000 := by
sorry

end NUMINAMATH_CALUDE_gcd_upper_bound_from_lcm_lower_bound_l2027_202796


namespace NUMINAMATH_CALUDE_existence_of_a_i_for_x_ij_l2027_202792

theorem existence_of_a_i_for_x_ij (n : ℕ) (x : Fin n → Fin n → ℝ)
  (h : ∀ (i j k : Fin n), x i j + x j k + x k i = 0) :
  ∃ (a : Fin n → ℝ), ∀ (i j : Fin n), x i j = a i - a j := by
  sorry

end NUMINAMATH_CALUDE_existence_of_a_i_for_x_ij_l2027_202792


namespace NUMINAMATH_CALUDE_singer_arrangement_count_l2027_202788

/-- The number of singers -/
def n : ℕ := 6

/-- The number of singers with specific arrangement requirements (A, B, C) -/
def k : ℕ := 3

/-- The number of valid arrangements of A, B, C (A-B-C, A-C-B, B-C-A, C-B-A) -/
def valid_abc_arrangements : ℕ := 4

/-- The total number of arrangements of n singers -/
def total_arrangements : ℕ := n.factorial

/-- The number of arrangements of k singers -/
def k_arrangements : ℕ := k.factorial

theorem singer_arrangement_count :
  (valid_abc_arrangements * total_arrangements / k_arrangements : ℕ) = 480 := by
  sorry

end NUMINAMATH_CALUDE_singer_arrangement_count_l2027_202788


namespace NUMINAMATH_CALUDE_ploughing_time_l2027_202746

/-- Given two workers R and S who can plough a field together in 10 hours,
    and R alone can plough the field in 15 hours,
    prove that S alone would take 30 hours to plough the same field. -/
theorem ploughing_time (r s : ℝ) : 
  (r + s = 1 / 10) →  -- R and S together take 10 hours
  (r = 1 / 15) →      -- R alone takes 15 hours
  (s = 1 / 30) :=     -- S alone takes 30 hours
by sorry

end NUMINAMATH_CALUDE_ploughing_time_l2027_202746


namespace NUMINAMATH_CALUDE_q_at_zero_l2027_202798

-- Define polynomials p, q, and r
variable (p q r : ℝ[X])

-- Define the relationship between r, p, and q
axiom r_eq_p_mul_q : r = p * q

-- Define the constant term of p(x)
axiom p_const_term : p.coeff 0 = 6

-- Define the constant term of r(x)
axiom r_const_term : r.coeff 0 = -18

-- The theorem to prove
theorem q_at_zero : q.eval 0 = -3 := by sorry

end NUMINAMATH_CALUDE_q_at_zero_l2027_202798


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_roots_difference_condition_l2027_202763

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (m + 3) * x + m + 1
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

theorem roots_difference_condition (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + (m + 3) * x + m + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ |x₁ - x₂| = 2 * Real.sqrt 2) →
  (m = 1 ∨ m = -3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_roots_difference_condition_l2027_202763


namespace NUMINAMATH_CALUDE_work_completion_time_l2027_202786

theorem work_completion_time 
  (ratio_a : ℚ) 
  (ratio_b : ℚ) 
  (combined_time : ℚ) 
  (h1 : ratio_a / ratio_b = 3 / 2) 
  (h2 : combined_time = 18) : 
  ratio_a / (ratio_a + ratio_b) * combined_time = 30 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2027_202786


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l2027_202755

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region with its number of outlets -/
structure Region where
  name : String
  outlets : Nat

/-- Represents an investigation with its sample size and population size -/
structure Investigation where
  sampleSize : Nat
  populationSize : Nat

/-- The company's sales outlet data -/
def companyData : List Region :=
  [⟨"A", 150⟩, ⟨"B", 120⟩, ⟨"C", 180⟩, ⟨"D", 150⟩]

/-- Total number of outlets -/
def totalOutlets : Nat := (companyData.map Region.outlets).sum

/-- Investigation ① -/
def investigation1 : Investigation :=
  ⟨100, totalOutlets⟩

/-- Investigation ② -/
def investigation2 : Investigation :=
  ⟨7, 10⟩

/-- Determines the appropriate sampling method for an investigation -/
def appropriateSamplingMethod (i : Investigation) : SamplingMethod :=
  sorry

theorem correct_sampling_methods :
  appropriateSamplingMethod investigation1 = SamplingMethod.StratifiedSampling ∧
  appropriateSamplingMethod investigation2 = SamplingMethod.SimpleRandomSampling :=
  sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l2027_202755


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l2027_202760

theorem triangle_angle_inequality (a b c α β γ : Real) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < α ∧ 0 < β ∧ 0 < γ ∧
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  α + β + γ = π →
  π / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < π / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l2027_202760


namespace NUMINAMATH_CALUDE_abs_nine_sqrt_l2027_202735

theorem abs_nine_sqrt : Real.sqrt (abs (-9)) = 3 := by sorry

end NUMINAMATH_CALUDE_abs_nine_sqrt_l2027_202735


namespace NUMINAMATH_CALUDE_health_run_distance_to_finish_l2027_202732

/-- The distance between a runner and the finish line in a health run event -/
def distance_to_finish (total_distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  total_distance - speed * time

/-- Theorem: In a 7.5 km health run, after running for 10 minutes at speed x km/min, 
    the distance to the finish line is 7.5 - 10x km -/
theorem health_run_distance_to_finish (x : ℝ) : 
  distance_to_finish 7.5 x 10 = 7.5 - 10 * x := by
  sorry

end NUMINAMATH_CALUDE_health_run_distance_to_finish_l2027_202732


namespace NUMINAMATH_CALUDE_division_remainder_proof_l2027_202756

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) 
  (h1 : dividend = 725)
  (h2 : divisor = 36)
  (h3 : quotient = 20) :
  dividend = divisor * quotient + 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l2027_202756


namespace NUMINAMATH_CALUDE_base_conversion_2458_to_base_7_l2027_202774

theorem base_conversion_2458_to_base_7 :
  2458 = 1 * 7^4 + 0 * 7^3 + 1 * 7^2 + 1 * 7^1 + 1 * 7^0 :=
by sorry

end NUMINAMATH_CALUDE_base_conversion_2458_to_base_7_l2027_202774


namespace NUMINAMATH_CALUDE_jane_apples_l2027_202713

theorem jane_apples (num_baskets : ℕ) (apples_taken : ℕ) (apples_remaining : ℕ) : 
  num_baskets = 4 → 
  apples_taken = 3 → 
  apples_remaining = 13 → 
  num_baskets * (apples_remaining + apples_taken) = 64 := by
sorry

end NUMINAMATH_CALUDE_jane_apples_l2027_202713


namespace NUMINAMATH_CALUDE_unique_k_is_zero_l2027_202707

/-- A function f: ℕ → ℕ satisfying f^n(n) = n + k for all n ∈ ℕ, where k is a non-negative integer -/
def SatisfiesCondition (f : ℕ → ℕ) (k : ℕ) : Prop :=
  ∀ n : ℕ, (f^[n] n) = n + k

/-- Theorem stating that if a function satisfies the condition, then k must be 0 -/
theorem unique_k_is_zero (f : ℕ → ℕ) (k : ℕ) (h : SatisfiesCondition f k) : k = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_is_zero_l2027_202707


namespace NUMINAMATH_CALUDE_function_periodic_l2027_202749

/-- A function satisfying the given conditions is periodic with period 1 -/
theorem function_periodic (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  ∀ x : ℝ, f (x + 1) = f x := by
  sorry

end NUMINAMATH_CALUDE_function_periodic_l2027_202749


namespace NUMINAMATH_CALUDE_sum_of_cubes_of_five_l2027_202701

theorem sum_of_cubes_of_five : 5^3 + 5^3 + 5^3 + 5^3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_of_five_l2027_202701


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2027_202712

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -1/2 := by sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2027_202712


namespace NUMINAMATH_CALUDE_fifty_percent_of_2002_l2027_202721

theorem fifty_percent_of_2002 : (50 : ℚ) / 100 * 2002 = 1001 := by
  sorry

end NUMINAMATH_CALUDE_fifty_percent_of_2002_l2027_202721


namespace NUMINAMATH_CALUDE_group_size_l2027_202781

theorem group_size (T : ℕ) (N : ℕ) (h1 : N > 0) : 
  (T : ℝ) / N - 3 = (T - 44 + 14 : ℝ) / N → N = 10 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l2027_202781


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l2027_202726

-- Define the variables and conditions
variable (x y a : ℝ)
variable (h1 : x * y ∈ Set.Icc (-π/4) (π/4))
variable (h2 : x^3 + Real.sin x - 2*a = 0)
variable (h3 : 4*y^3 + (1/2) * Real.sin (2*y) - a = 0)

-- State the theorem
theorem cos_x_plus_2y_equals_one : Real.cos (x + 2*y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l2027_202726


namespace NUMINAMATH_CALUDE_goose_survival_fraction_l2027_202797

theorem goose_survival_fraction (total_eggs : ℕ) 
  (hatch_rate : ℚ) (first_month_survival_rate : ℚ) (first_year_survivors : ℕ) :
  hatch_rate = 1/2 →
  first_month_survival_rate = 3/4 →
  first_year_survivors = 120 →
  (hatch_rate * first_month_survival_rate * total_eggs : ℚ) = first_year_survivors →
  (first_year_survivors : ℚ) / (hatch_rate * first_month_survival_rate * total_eggs) = 1 :=
by sorry

end NUMINAMATH_CALUDE_goose_survival_fraction_l2027_202797


namespace NUMINAMATH_CALUDE_selma_has_50_marbles_l2027_202745

/-- The number of marbles Selma has -/
def selma_marbles (merill_marbles elliot_marbles : ℕ) : ℕ :=
  merill_marbles + elliot_marbles + 5

/-- Theorem stating the number of marbles Selma has -/
theorem selma_has_50_marbles :
  ∀ (merill_marbles elliot_marbles : ℕ),
    merill_marbles = 30 →
    merill_marbles = 2 * elliot_marbles →
    selma_marbles merill_marbles elliot_marbles = 50 :=
by
  sorry

#check selma_has_50_marbles

end NUMINAMATH_CALUDE_selma_has_50_marbles_l2027_202745


namespace NUMINAMATH_CALUDE_x₁x₂_equals_1008_l2027_202789

noncomputable def x₁ : ℝ := Real.exp (Real.log 2 * 1008 / (Real.log 2 * Real.exp (Real.log 2 * 1008 / (Real.log 2 * Real.exp (Real.log 2 * 1008 / Real.log 2)))))

noncomputable def x₂ : ℝ := Real.log 2 * 1008 / (Real.log 2 * Real.exp (Real.log 2 * 1008 / Real.log 2))

theorem x₁x₂_equals_1008 :
  x₁ * Real.log x₁ / Real.log 2 = 1008 ∧
  x₂ * 2^x₂ = 1008 →
  x₁ * x₂ = 1008 := by
  sorry

end NUMINAMATH_CALUDE_x₁x₂_equals_1008_l2027_202789


namespace NUMINAMATH_CALUDE_parabola_sum_property_l2027_202751

/-- Represents a quadratic function of the form ax^2 + bx + c --/
structure Quadratic where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The function resulting from reflecting a quadratic about the y-axis --/
def reflect (q : Quadratic) : Quadratic :=
  { a := q.a, b := -q.b, c := q.c }

/-- Vertical translation of a quadratic function --/
def translate (q : Quadratic) (d : ℝ) : Quadratic :=
  { a := q.a, b := q.b, c := q.c + d }

/-- The sum of two quadratic functions --/
def add (q1 q2 : Quadratic) : Quadratic :=
  { a := q1.a + q2.a, b := q1.b + q2.b, c := q1.c + q2.c }

theorem parabola_sum_property (q : Quadratic) :
  let f := translate q 4
  let g := translate (reflect q) (-4)
  (add f g).b = 0 := by sorry

end NUMINAMATH_CALUDE_parabola_sum_property_l2027_202751


namespace NUMINAMATH_CALUDE_probability_product_eight_l2027_202709

/-- A standard 6-sided die -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sample space of rolling a standard 6-sided die twice -/
def TwoRollsSampleSpace : Finset (ℕ × ℕ) :=
  (StandardDie.product StandardDie)

/-- The favorable outcomes where the product of two rolls is 8 -/
def FavorableOutcomes : Finset (ℕ × ℕ) :=
  {(2, 4), (4, 2)}

/-- Probability of the product of two rolls being 8 -/
theorem probability_product_eight :
  (FavorableOutcomes.card : ℚ) / TwoRollsSampleSpace.card = 1 / 18 :=
sorry

end NUMINAMATH_CALUDE_probability_product_eight_l2027_202709


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_equals_volume_l2027_202706

theorem rectangular_parallelepiped_surface_area_equals_volume :
  ∃ (a b c : ℕ+), 2 * (a * b + b * c + a * c) = a * b * c :=
sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_equals_volume_l2027_202706


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l2027_202704

/-- Given a geometric sequence {aₙ} where a₁ + a₂ = 3 and a₂ + a₃ = 6, prove that a₃ = 4 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- a is a sequence of real numbers
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1))  -- a is a geometric sequence
  (h_sum1 : a 1 + a 2 = 3)  -- first condition
  (h_sum2 : a 2 + a 3 = 6)  -- second condition
  : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l2027_202704


namespace NUMINAMATH_CALUDE_henrys_score_l2027_202747

theorem henrys_score (june patty josh henry : ℕ) : 
  june = 97 → patty = 85 → josh = 100 →
  (june + patty + josh + henry) / 4 = 94 →
  henry = 94 := by
sorry

end NUMINAMATH_CALUDE_henrys_score_l2027_202747


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_factorial_sum_l2027_202739

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

theorem largest_prime_factor_of_factorial_sum :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (factorial 6 + factorial 7) ∧
  ∀ (q : ℕ), Nat.Prime q → q ∣ (factorial 6 + factorial 7) → q ≤ p :=
sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_factorial_sum_l2027_202739


namespace NUMINAMATH_CALUDE_meat_purchase_cost_l2027_202733

/-- Represents the cost and quantity of a type of meat -/
structure Meat where
  name : String
  cost : ℝ
  quantity : ℝ

/-- Calculates the total cost of a purchase given meat prices and quantities -/
def totalCost (meats : List Meat) : ℝ :=
  meats.map (fun m => m.cost * m.quantity) |>.sum

/-- Theorem stating the total cost of the meat purchase -/
theorem meat_purchase_cost :
  let pork_cost : ℝ := 6
  let chicken_cost : ℝ := pork_cost - 2
  let beef_cost : ℝ := chicken_cost + 4
  let lamb_cost : ℝ := pork_cost + 3
  let meats : List Meat := [
    { name := "Chicken", cost := chicken_cost, quantity := 3.5 },
    { name := "Pork", cost := pork_cost, quantity := 1.2 },
    { name := "Beef", cost := beef_cost, quantity := 2.3 },
    { name := "Lamb", cost := lamb_cost, quantity := 0.8 }
  ]
  totalCost meats = 46.8 := by
  sorry

end NUMINAMATH_CALUDE_meat_purchase_cost_l2027_202733


namespace NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2027_202700

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) * (2 * a - Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → z = -2 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_on_imaginary_axis_l2027_202700


namespace NUMINAMATH_CALUDE_no_negative_roots_l2027_202784

theorem no_negative_roots : 
  ∀ x : ℝ, x < 0 → x^4 - 4*x^3 - 6*x^2 - 3*x + 9 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_negative_roots_l2027_202784


namespace NUMINAMATH_CALUDE_solution_satisfies_equations_l2027_202779

theorem solution_satisfies_equations :
  ∃ (x y : ℚ), 
    (x = 5/2 ∧ y = 3) ∧
    (x + y + 1 = (6 - x) + (6 - y)) ∧
    (x - y + 2 = (x - 2) + (y - 2)) := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_equations_l2027_202779


namespace NUMINAMATH_CALUDE_inverse_of_5_mod_33_l2027_202777

theorem inverse_of_5_mod_33 : ∃ x : ℕ, x < 33 ∧ (5 * x) % 33 = 1 := by
  use 20
  sorry

end NUMINAMATH_CALUDE_inverse_of_5_mod_33_l2027_202777


namespace NUMINAMATH_CALUDE_distance_relationships_l2027_202702

structure Distance where
  r : ℝ
  a : ℝ
  b : ℝ

def perpendicular_to_x12 (d : Distance) : Prop := sorry
def parallel_to_H (d : Distance) : Prop := sorry
def parallel_to_P1 (d : Distance) : Prop := sorry
def perpendicular_to_H (d : Distance) : Prop := sorry
def perpendicular_to_P1 (d : Distance) : Prop := sorry
def parallel_to_x12 (d : Distance) : Prop := sorry

theorem distance_relationships (d : Distance) :
  (∃ α β : ℝ, d.a = d.r * Real.cos α ∧ d.b = d.r * Real.cos β) ∧
  (perpendicular_to_x12 d → d.a^2 + d.b^2 = d.r^2) ∧
  (parallel_to_H d → d.a = d.b) ∧
  (parallel_to_P1 d → d.a = d.r ∧ ∃ β : ℝ, d.b = d.a * Real.cos β) ∧
  (perpendicular_to_H d → d.a = d.b ∧ d.a = d.r * Real.sqrt 2 / 2) ∧
  (perpendicular_to_P1 d → d.a = 0 ∧ d.b = d.r) ∧
  (parallel_to_x12 d → d.a = d.b ∧ d.a = d.r) :=
by sorry

end NUMINAMATH_CALUDE_distance_relationships_l2027_202702


namespace NUMINAMATH_CALUDE_intersection_length_l2027_202737

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 5
def circle2 (x y m : ℝ) : Prop := (x - m)^2 + y^2 = 20

-- Define the intersection points
def intersectionPoints (m : ℝ) : Prop := ∃ (A B : ℝ × ℝ), 
  circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 A.1 A.2 m ∧ circle2 B.1 B.2 m

-- Define the perpendicular tangents condition
def perpendicularTangents (m : ℝ) (A : ℝ × ℝ) : Prop :=
  circle1 A.1 A.2 ∧ circle2 A.1 A.2 m ∧
  (A.1 * m = 5) -- This condition represents perpendicular tangents

-- Theorem statement
theorem intersection_length (m : ℝ) :
  intersectionPoints m →
  (∃ (A : ℝ × ℝ), perpendicularTangents m A) →
  (∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle1 B.1 B.2 ∧ 
                     circle2 A.1 A.2 m ∧ circle2 B.1 B.2 m ∧
                     ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16) :=
by sorry

end NUMINAMATH_CALUDE_intersection_length_l2027_202737


namespace NUMINAMATH_CALUDE_error_arrangement_probability_l2027_202719

/-- The number of letters in the word "error" -/
def word_length : Nat := 5

/-- The number of 'r's in the word "error" -/
def num_r : Nat := 3

/-- The number of ways to arrange the letters in "error" -/
def total_arrangements : Nat := 20

/-- The probability of incorrectly arranging the letters in "error" -/
def incorrect_probability : Rat := 19 / 20

/-- Theorem stating that the probability of incorrectly arranging the letters in "error" is 19/20 -/
theorem error_arrangement_probability :
  incorrect_probability = 19 / 20 :=
by sorry

end NUMINAMATH_CALUDE_error_arrangement_probability_l2027_202719


namespace NUMINAMATH_CALUDE_problem_solution_l2027_202720

theorem problem_solution (a b c : ℝ) 
  (h1 : ∀ x, (x - a) * (x - b) / (x - c) ≤ 0 ↔ x < -2 ∨ |x - 30| ≤ 2)
  (h2 : a < b) : 
  a + 2*b + 3*c = 86 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2027_202720


namespace NUMINAMATH_CALUDE_perimeter_of_cut_square_perimeter_of_specific_cut_square_l2027_202793

/-- The perimeter of a figure formed by cutting a square into two equal rectangles and placing them next to each other -/
theorem perimeter_of_cut_square (square_side : ℝ) : 
  square_side > 0 → 
  (3 * square_side + 4 * (square_side / 2)) = 5 * square_side := by
  sorry

/-- The perimeter of a figure formed by cutting a square with side length 100 into two equal rectangles and placing them next to each other is 500 -/
theorem perimeter_of_specific_cut_square : 
  (3 * 100 + 4 * (100 / 2)) = 500 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_cut_square_perimeter_of_specific_cut_square_l2027_202793


namespace NUMINAMATH_CALUDE_problem_statement_l2027_202724

theorem problem_statement (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) :
  (a^2)^(4*b) = a^(2*b) * x^(3*b) → x = a^2 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2027_202724


namespace NUMINAMATH_CALUDE_road_trip_total_hours_l2027_202757

/-- Calculates the total hours driven during a road trip -/
def total_hours_driven (days : ℕ) (hours_per_day_person1 : ℕ) (hours_per_day_person2 : ℕ) : ℕ :=
  days * (hours_per_day_person1 + hours_per_day_person2)

/-- Proves that the total hours driven in the given scenario is 42 -/
theorem road_trip_total_hours : total_hours_driven 3 8 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_total_hours_l2027_202757


namespace NUMINAMATH_CALUDE_josie_checkout_wait_time_l2027_202743

def total_shopping_time : ℕ := 90
def cart_wait_time : ℕ := 3
def employee_wait_time : ℕ := 13
def stocker_wait_time : ℕ := 14
def shopping_time : ℕ := 42

theorem josie_checkout_wait_time :
  total_shopping_time - shopping_time - (cart_wait_time + employee_wait_time + stocker_wait_time) = 18 := by
  sorry

end NUMINAMATH_CALUDE_josie_checkout_wait_time_l2027_202743


namespace NUMINAMATH_CALUDE_cube_digits_convergence_l2027_202791

/-- The function that cubes each digit of a natural number and sums the results -/
def cube_digits_sum (n : ℕ) : ℕ :=
  n.digits 10
    |>.map (fun d => d^3)
    |>.sum

/-- The sequence generated by repeatedly applying cube_digits_sum -/
def cube_digits_sequence (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => cube_digits_sum (cube_digits_sequence start n)

/-- The theorem stating that the sequence converges to 153 for multiples of 3 -/
theorem cube_digits_convergence (n : ℕ) (h : 3 ∣ n) :
  ∃ k, ∀ m ≥ k, cube_digits_sequence n m = 153 := by
  sorry

end NUMINAMATH_CALUDE_cube_digits_convergence_l2027_202791


namespace NUMINAMATH_CALUDE_weight_of_B_l2027_202785

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 47) :
  B = 39 := by
sorry

end NUMINAMATH_CALUDE_weight_of_B_l2027_202785


namespace NUMINAMATH_CALUDE_complex_simplification_l2027_202762

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem stating the equality of the complex expression and its simplified form -/
theorem complex_simplification :
  3 * (2 + i) - i * (3 - i) + 2 * (1 - 2*i) = 7 - 4*i :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2027_202762


namespace NUMINAMATH_CALUDE_modular_congruence_existence_l2027_202730

theorem modular_congruence_existence (a c : ℕ+) (b : ℤ) :
  ∃ x : ℕ+, (c : ℤ) ∣ ((a : ℤ)^(x : ℕ) + x - b) := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_existence_l2027_202730


namespace NUMINAMATH_CALUDE_base7_addition_theorem_l2027_202787

/-- Addition of numbers in base 7 -/
def base7_add (a b c : ℕ) : ℕ :=
  (a + b + c) % 7^4

/-- Conversion from base 7 to decimal -/
def base7_to_decimal (n : ℕ) : ℕ :=
  (n / 7^3) * 7^3 + ((n / 7^2) % 7) * 7^2 + ((n / 7) % 7) * 7 + (n % 7)

theorem base7_addition_theorem :
  base7_add (base7_to_decimal 256) (base7_to_decimal 463) (base7_to_decimal 132) =
  base7_to_decimal 1214 :=
sorry

end NUMINAMATH_CALUDE_base7_addition_theorem_l2027_202787


namespace NUMINAMATH_CALUDE_largest_solution_is_three_l2027_202758

theorem largest_solution_is_three :
  let f (x : ℝ) := (15 * x^2 - 40 * x + 18) / (4 * x - 3) + 7 * x
  ∃ (max : ℝ), max = 3 ∧ 
    (∀ x : ℝ, f x = 8 * x - 2 → x ≤ max) ∧
    (f max = 8 * max - 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_is_three_l2027_202758


namespace NUMINAMATH_CALUDE_largest_covered_range_l2027_202741

def is_monic_quadratic (p : ℤ → ℤ) : Prop :=
  ∃ a b : ℤ, ∀ x, p x = x^2 + a*x + b

def covers_range (p₁ p₂ p₃ : ℤ → ℤ) (n : ℕ) : Prop :=
  ∀ i ∈ Finset.range n, ∃ j ∈ [1, 2, 3], ∃ m : ℤ, 
    (if j = 1 then p₁ else if j = 2 then p₂ else p₃) m = i + 1

theorem largest_covered_range : 
  (∃ p₁ p₂ p₃ : ℤ → ℤ, 
    is_monic_quadratic p₁ ∧ 
    is_monic_quadratic p₂ ∧ 
    is_monic_quadratic p₃ ∧ 
    covers_range p₁ p₂ p₃ 9) ∧ 
  (∀ n > 9, ¬∃ p₁ p₂ p₃ : ℤ → ℤ, 
    is_monic_quadratic p₁ ∧ 
    is_monic_quadratic p₂ ∧ 
    is_monic_quadratic p₃ ∧ 
    covers_range p₁ p₂ p₃ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_covered_range_l2027_202741


namespace NUMINAMATH_CALUDE_minimum_number_of_boys_l2027_202725

theorem minimum_number_of_boys (k : ℕ) (n m : ℕ) : 
  (k > 0) →  -- total number of apples is positive
  (n > 0) →  -- there is at least one boy who collected 10 apples
  (m > 0) →  -- there is at least one boy who collected 10% of apples
  (100 * n + m * k = 10 * k) →  -- equation representing total apples collected
  (n + m ≥ 6) →  -- total number of boys is at least 6
  ∀ (n' m' : ℕ), (n' > 0) → (m' > 0) → 
    (∃ (k' : ℕ), k' > 0 ∧ 100 * n' + m' * k' = 10 * k') → 
    (n' + m' ≥ 6) :=
by
  sorry

#check minimum_number_of_boys

end NUMINAMATH_CALUDE_minimum_number_of_boys_l2027_202725


namespace NUMINAMATH_CALUDE_marina_olympiad_supplies_l2027_202748

/-- The cost of school supplies for Marina's olympiad participation. -/
def school_supplies_cost 
  (notebook : ℕ) 
  (pencil : ℕ) 
  (eraser : ℕ) 
  (ruler : ℕ) 
  (pen : ℕ) : Prop :=
  notebook = 15 ∧ 
  notebook + pencil + eraser = 47 ∧
  notebook + ruler + pen = 58 →
  notebook + pencil + eraser + ruler + pen = 90

theorem marina_olympiad_supplies : 
  ∃ (notebook pencil eraser ruler pen : ℕ), 
  school_supplies_cost notebook pencil eraser ruler pen :=
sorry

end NUMINAMATH_CALUDE_marina_olympiad_supplies_l2027_202748


namespace NUMINAMATH_CALUDE_business_profit_calculation_l2027_202767

-- Define the partners
inductive Partner
| Mary
| Mike
| Anna
| Ben

-- Define the investment amounts
def investment (p : Partner) : ℕ :=
  match p with
  | Partner.Mary => 800
  | Partner.Mike => 200
  | Partner.Anna => 600
  | Partner.Ben => 400

-- Define the profit sharing ratios for the last part
def profit_ratio (p : Partner) : ℕ :=
  match p with
  | Partner.Mary => 2
  | Partner.Mike => 1
  | Partner.Anna => 3
  | Partner.Ben => 4

-- Define the total investment
def total_investment : ℕ := 
  investment Partner.Mary + investment Partner.Mike + 
  investment Partner.Anna + investment Partner.Ben

-- Define the theorem
theorem business_profit_calculation (P : ℚ) : 
  (3 * P / 10 - 3 * P / 20 = 900) ∧ 
  (17 * P / 60 - 13 * P / 60 = 600) → 
  P = 6000 := by
sorry

end NUMINAMATH_CALUDE_business_profit_calculation_l2027_202767


namespace NUMINAMATH_CALUDE_intersection_A_B_l2027_202790

def U : Set Nat := {0, 1, 3, 7, 9}
def C_UA : Set Nat := {0, 5, 9}
def B : Set Nat := {3, 5, 7}
def A : Set Nat := U \ C_UA

theorem intersection_A_B : A ∩ B = {3, 7} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2027_202790


namespace NUMINAMATH_CALUDE_equilateral_triangle_intersection_l2027_202705

/-- Given a right triangular prism with base edges a, b, and c,
    when intersected by a plane to form an equilateral triangle with side length d,
    prove that d satisfies the equation: 3d^4 - 100d^2 + 576 = 0 -/
theorem equilateral_triangle_intersection (a b c d : ℝ) : 
  a = 3 → b = 4 → c = 5 → 
  3 * d^4 - 100 * d^2 + 576 = 0 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_intersection_l2027_202705


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l2027_202799

/-- Converts a list of digits in a given base to its decimal representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

/-- Checks if a list of digits represents 2017 in the given base -/
def is_2017 (digits : List Nat) (base : Nat) : Prop :=
  to_decimal digits base = 2017

/-- Checks if a list of digits can have one digit removed to represent 2017 in another base -/
def can_remove_digit_for_2017 (digits : List Nat) (new_base : Nat) : Prop :=
  ∃ (new_digits : List Nat), new_digits.length + 1 = digits.length ∧ 
    (∃ (i : Nat), i < digits.length ∧ new_digits = (digits.take i ++ digits.drop (i+1))) ∧
    is_2017 new_digits new_base

theorem base_conversion_theorem :
  ∃ (a b c : Nat),
    is_2017 [1, 3, 3, 2, 0, 1] a ∧
    can_remove_digit_for_2017 [1, 3, 3, 2, 0, 1] b ∧
    (∃ (digits : List Nat), digits.length = 5 ∧ 
      can_remove_digit_for_2017 digits c ∧
      is_2017 digits b) ∧
    a + b + c = 22 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l2027_202799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2027_202740

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 + a 2011 = 10) →
  (a 1 * a 2011 = 16) →
  a 2 + a 1006 + a 2010 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2027_202740


namespace NUMINAMATH_CALUDE_lunch_combinations_count_l2027_202783

/-- Represents the number of options for each lunch component -/
structure LunchOptions where
  mainCourses : Nat
  beverages : Nat
  snacks : Nat

/-- Calculates the total number of lunch combinations -/
def totalCombinations (options : LunchOptions) : Nat :=
  options.mainCourses * options.beverages * options.snacks

/-- The given lunch options in the cafeteria -/
def cafeteriaOptions : LunchOptions :=
  { mainCourses := 4
  , beverages := 3
  , snacks := 2 }

/-- Theorem stating that the number of lunch combinations is 24 -/
theorem lunch_combinations_count : totalCombinations cafeteriaOptions = 24 := by
  sorry

end NUMINAMATH_CALUDE_lunch_combinations_count_l2027_202783


namespace NUMINAMATH_CALUDE_smallest_square_sum_20_consecutive_l2027_202759

/-- The sum of an arithmetic sequence of 20 terms -/
def sum_20_consecutive (first : ℕ) : ℕ :=
  20 * (2 * first + 19) / 2

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem smallest_square_sum_20_consecutive :
  (∀ n : ℕ, n < 490 → ¬(is_perfect_square n ∧ ∃ k : ℕ, sum_20_consecutive k = n)) ∧
  (is_perfect_square 490 ∧ ∃ k : ℕ, sum_20_consecutive k = 490) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_sum_20_consecutive_l2027_202759


namespace NUMINAMATH_CALUDE_expression_evaluation_l2027_202752

theorem expression_evaluation (a b c : ℕ) (ha : a = 3) (hb : b = 2) (hc : c = 4) :
  ((a^b)^a - (b^a)^b) * c = 2660 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2027_202752


namespace NUMINAMATH_CALUDE_percent_of_y_l2027_202711

theorem percent_of_y (y : ℝ) (h : y > 0) : ((8 * y) / 20 + (3 * y) / 10) / y = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l2027_202711


namespace NUMINAMATH_CALUDE_square_sum_given_difference_l2027_202782

theorem square_sum_given_difference (a : ℝ) (h : a - 1/a = 3) : (a + 1/a)^2 = 13 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_l2027_202782


namespace NUMINAMATH_CALUDE_f_max_min_implies_m_range_l2027_202769

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- Theorem: If the maximum value of f(x) in [0,m] is 2 and the minimum is 1, then 1 ≤ m ≤ 2 -/
theorem f_max_min_implies_m_range (m : ℝ) 
  (h_max : ∀ x ∈ Set.Icc 0 m, f x ≤ 2) 
  (h_min : ∃ x ∈ Set.Icc 0 m, f x = 1) 
  (h_reaches_max : ∃ x ∈ Set.Icc 0 m, f x = 2) : 
  1 ≤ m ∧ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_implies_m_range_l2027_202769


namespace NUMINAMATH_CALUDE_two_points_l2027_202714

/-- The number of integer points satisfying the given equation and conditions -/
def num_points : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ =>
    let x := p.1
    let y := p.2
    x > 0 ∧ y > 0 ∧ x > y ∧ y * (x - 1) = 2 * x + 2018
  ) (Finset.product (Finset.range 10000) (Finset.range 10000))).card

/-- Theorem stating that there are exactly two points satisfying the conditions -/
theorem two_points : num_points = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_points_l2027_202714


namespace NUMINAMATH_CALUDE_division_problem_l2027_202775

theorem division_problem (h : (7125 : ℝ) / 1.25 = 5700) : 
  ∃ x : ℝ, 712.5 / x = 57 ∧ x = 12.5 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2027_202775


namespace NUMINAMATH_CALUDE_smallest_ends_in_9_divisible_by_13_l2027_202727

/-- A positive integer that ends in 9 -/
def EndsIn9 (n : ℕ) : Prop := n % 10 = 9 ∧ n > 0

/-- The smallest positive integer that ends in 9 and is divisible by 13 -/
def SmallestEndsIn9DivisibleBy13 : ℕ := 99

theorem smallest_ends_in_9_divisible_by_13 :
  EndsIn9 SmallestEndsIn9DivisibleBy13 ∧
  SmallestEndsIn9DivisibleBy13 % 13 = 0 ∧
  ∀ n : ℕ, EndsIn9 n ∧ n % 13 = 0 → n ≥ SmallestEndsIn9DivisibleBy13 := by
  sorry

end NUMINAMATH_CALUDE_smallest_ends_in_9_divisible_by_13_l2027_202727


namespace NUMINAMATH_CALUDE_grass_field_width_l2027_202772

/-- Represents the width of the grass field -/
def field_width : ℝ := sorry

/-- The length of the grass field in meters -/
def field_length : ℝ := 75

/-- The width of the path around the field in meters -/
def path_width : ℝ := 2.5

/-- The cost of constructing the path per square meter in Rupees -/
def cost_per_sqm : ℝ := 2

/-- The total cost of constructing the path in Rupees -/
def total_cost : ℝ := 1350

/-- Theorem stating that given the conditions, the width of the grass field is 55 meters -/
theorem grass_field_width : 
  field_width = 55 := by sorry

end NUMINAMATH_CALUDE_grass_field_width_l2027_202772


namespace NUMINAMATH_CALUDE_expression_evaluation_l2027_202729

theorem expression_evaluation :
  let a : ℤ := -1
  let b : ℤ := 3
  2 * a * b^2 - (3 * a^2 * b - 2 * (3 * a^2 * b - a * b^2 - 1)) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2027_202729


namespace NUMINAMATH_CALUDE_trig_identity_l2027_202722

theorem trig_identity : 
  Real.sin (47 * π / 180) * Real.cos (17 * π / 180) - 
  Real.cos (47 * π / 180) * Real.cos (73 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2027_202722


namespace NUMINAMATH_CALUDE_trinomial_cube_l2027_202765

theorem trinomial_cube (x : ℝ) :
  (x^2 - 2*x + 1)^3 = x^6 - 6*x^5 + 15*x^4 - 20*x^3 + 15*x^2 - 6*x + 1 :=
by sorry

end NUMINAMATH_CALUDE_trinomial_cube_l2027_202765


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2027_202728

/-- The number of positive integers satisfying the inequality -/
def count_satisfying_integers : ℕ := 8

/-- The inequality function -/
def inequality (n : ℤ) : Prop :=
  (n + 7) * (n - 4) * (n - 10) < 0

theorem count_integers_satisfying_inequality :
  (∃ (S : Finset ℤ), S.card = count_satisfying_integers ∧
    (∀ n ∈ S, n > 0 ∧ inequality n) ∧
    (∀ n : ℤ, n > 0 → inequality n → n ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l2027_202728


namespace NUMINAMATH_CALUDE_software_package_savings_l2027_202780

/-- Calculates the savings when choosing a more expensive software package with higher device coverage over a cheaper one with lower device coverage. -/
theorem software_package_savings
  (total_devices : ℕ)
  (package1_price package2_price : ℕ)
  (package1_coverage package2_coverage : ℕ)
  (h1 : total_devices = 50)
  (h2 : package1_price = 40)
  (h3 : package2_price = 60)
  (h4 : package1_coverage = 5)
  (h5 : package2_coverage = 10) :
  (total_devices / package1_coverage * package1_price) -
  (total_devices / package2_coverage * package2_price) = 100 :=
by
  sorry

#check software_package_savings

end NUMINAMATH_CALUDE_software_package_savings_l2027_202780


namespace NUMINAMATH_CALUDE_triangle_area_l2027_202776

/-- Given a triangle with perimeter 20 cm and inradius 2.5 cm, its area is 25 cm². -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
    (h1 : perimeter = 20) 
    (h2 : inradius = 2.5) 
    (h3 : area = inradius * (perimeter / 2)) : 
  area = 25 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2027_202776


namespace NUMINAMATH_CALUDE_sqrt_three_plus_one_over_two_lt_sqrt_two_l2027_202794

theorem sqrt_three_plus_one_over_two_lt_sqrt_two :
  (Real.sqrt 3 + 1) / 2 < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_plus_one_over_two_lt_sqrt_two_l2027_202794


namespace NUMINAMATH_CALUDE_tire_cost_calculation_l2027_202718

theorem tire_cost_calculation (total_cost : ℕ) (num_tires : ℕ) (h1 : total_cost = 240) (h2 : num_tires = 4) :
  total_cost / num_tires = 60 := by
  sorry

end NUMINAMATH_CALUDE_tire_cost_calculation_l2027_202718


namespace NUMINAMATH_CALUDE_inverse_undefined_at_one_l2027_202754

/-- Given a function g(x) = (x - 5) / (x - 6), prove that its inverse g⁻¹(x) is undefined when x = 1 -/
theorem inverse_undefined_at_one (g : ℝ → ℝ) (h : ∀ x, g x = (x - 5) / (x - 6)) :
  ¬∃ y, g y = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_undefined_at_one_l2027_202754
