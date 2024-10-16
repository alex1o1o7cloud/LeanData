import Mathlib

namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_sum_of_4th_and_6th_is_zero_l2314_231453

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ+) : ℤ := -n^2 + 9*n

/-- The n-th term of the sequence -/
def a (n : ℕ+) : ℤ := S n - S (n-1)

/-- The sequence {a_n} is arithmetic -/
theorem sequence_is_arithmetic : ∀ n : ℕ+, a (n+1) - a n = a (n+2) - a (n+1) :=
sorry

/-- The sum of the 4th and 6th terms is zero -/
theorem sum_of_4th_and_6th_is_zero : a 4 + a 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_sum_of_4th_and_6th_is_zero_l2314_231453


namespace NUMINAMATH_CALUDE_ratio_problem_l2314_231426

theorem ratio_problem (first_part : ℝ) (ratio_percent : ℝ) (second_part : ℝ) :
  first_part = 25 →
  ratio_percent = 50 →
  first_part / (first_part + second_part) * 100 = ratio_percent →
  second_part = 25 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2314_231426


namespace NUMINAMATH_CALUDE_residue_mod_35_l2314_231476

theorem residue_mod_35 : ∃ r : ℤ, 0 ≤ r ∧ r < 35 ∧ (-963 + 100) ≡ r [ZMOD 35] ∧ r = 12 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_35_l2314_231476


namespace NUMINAMATH_CALUDE_arithmetic_sequence_and_general_formula_l2314_231498

def sequence_a : ℕ → ℚ
  | 0 => 1
  | n + 1 => 2 * sequence_a n / (sequence_a n + 2)

def sequence_b (n : ℕ) : ℚ := 1 / sequence_a n

theorem arithmetic_sequence_and_general_formula :
  (∀ n : ℕ, ∃ d : ℚ, sequence_b (n + 1) - sequence_b n = d) ∧
  (∀ n : ℕ, sequence_a n = 2 / (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_and_general_formula_l2314_231498


namespace NUMINAMATH_CALUDE_range_of_k_when_q_range_of_a_when_p_necessary_not_sufficient_l2314_231470

-- Define propositions p and q
def p (k a : ℝ) : Prop := ∃ x y : ℝ, x^2/(k-1) + y^2/(7-a) = 1 ∧ k ≠ 1 ∧ a ≠ 7

def q (k : ℝ) : Prop := ¬∃ x y : ℝ, (4-k)*x^2 + (k-2)*y^2 = 1 ∧ (4-k)*(k-2) < 0

-- Theorem 1: Range of k when q is true
theorem range_of_k_when_q (k : ℝ) : q k → 2 ≤ k ∧ k ≤ 4 :=
sorry

-- Theorem 2: Range of a when p is a necessary but not sufficient condition for q
theorem range_of_a_when_p_necessary_not_sufficient (a : ℝ) :
  (∀ k : ℝ, q k → (∃ k', p k' a)) ∧ (∃ k : ℝ, p k a ∧ ¬q k) → a < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_k_when_q_range_of_a_when_p_necessary_not_sufficient_l2314_231470


namespace NUMINAMATH_CALUDE_linear_function_through_0_3_l2314_231400

/-- A linear function passing through (0,3) -/
def LinearFunctionThrough0_3 (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y - f 0) ∧ f 0 = 3

theorem linear_function_through_0_3 (f : ℝ → ℝ) (hf : LinearFunctionThrough0_3 f) :
  ∃ m : ℝ, ∀ x : ℝ, f x = m * x + 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_through_0_3_l2314_231400


namespace NUMINAMATH_CALUDE_cuboid_to_cube_l2314_231466

-- Define the dimensions of the original cuboid
def cuboid_length : ℝ := 27
def cuboid_width : ℝ := 18
def cuboid_height : ℝ := 12

-- Define the volume to be added
def added_volume : ℝ := 17.999999999999996

-- Define the edge length of the resulting cube in centimeters
def cube_edge_cm : ℕ := 1802

-- Theorem statement
theorem cuboid_to_cube :
  let original_volume := cuboid_length * cuboid_width * cuboid_height
  let total_volume := original_volume + added_volume
  let cube_edge_m := (total_volume ^ (1/3 : ℝ))
  ∃ (ε : ℝ), ε ≥ 0 ∧ ε < 1 ∧ cube_edge_cm = ⌊cube_edge_m * 100 + ε⌋ :=
sorry

end NUMINAMATH_CALUDE_cuboid_to_cube_l2314_231466


namespace NUMINAMATH_CALUDE_salary_change_l2314_231427

/-- Proves that when a salary is increased by 10% and then reduced by 10%, 
    the net change is a decrease of 1% of the original salary. -/
theorem salary_change (S : ℝ) : 
  (S + S * (10 / 100)) * (1 - 10 / 100) = S * 0.99 := by
  sorry

end NUMINAMATH_CALUDE_salary_change_l2314_231427


namespace NUMINAMATH_CALUDE_y_value_proof_l2314_231401

theorem y_value_proof : ∀ y : ℝ, (1/3 - 1/4 = 4/y) → y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_proof_l2314_231401


namespace NUMINAMATH_CALUDE_samia_walking_distance_l2314_231482

/-- Proves that Samia walked 4.0 km given the journey conditions --/
theorem samia_walking_distance :
  ∀ (total_distance : ℝ) (biking_distance : ℝ),
    -- Samia's average biking speed is 15 km/h
    -- Samia bikes for 30 minutes (0.5 hours)
    biking_distance = 15 * 0.5 →
    -- The entire journey took 90 minutes (1.5 hours)
    0.5 + ((total_distance - biking_distance) / 4) = 1.5 →
    -- Prove that the walking distance is 4.0 km
    total_distance - biking_distance = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_samia_walking_distance_l2314_231482


namespace NUMINAMATH_CALUDE_cube_root_last_three_digits_l2314_231442

theorem cube_root_last_three_digits :
  ∃ (n a b : ℕ), 
    n > 0 ∧
    n = 1000 * a + b ∧
    b ≤ 999 ∧
    a^3 = n ∧
    n = 32768 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_last_three_digits_l2314_231442


namespace NUMINAMATH_CALUDE_society_committee_selection_l2314_231455

theorem society_committee_selection (n : ℕ) (k : ℕ) : n = 20 ∧ k = 3 → Nat.choose n k = 1140 := by
  sorry

end NUMINAMATH_CALUDE_society_committee_selection_l2314_231455


namespace NUMINAMATH_CALUDE_percentage_calculation_l2314_231467

theorem percentage_calculation (P : ℝ) : 
  (0.05 * (P / 100 * 1600) = 20) → P = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2314_231467


namespace NUMINAMATH_CALUDE_cupcakes_eaten_equals_packaged_l2314_231433

/-- Proves that the number of cupcakes Todd ate is equal to the number of cupcakes used for packaging -/
theorem cupcakes_eaten_equals_packaged (initial_cupcakes : ℕ) (num_packages : ℕ) (cupcakes_per_package : ℕ)
  (h1 : initial_cupcakes = 71)
  (h2 : num_packages = 4)
  (h3 : cupcakes_per_package = 7) :
  initial_cupcakes - (initial_cupcakes - num_packages * cupcakes_per_package) = num_packages * cupcakes_per_package :=
by sorry

end NUMINAMATH_CALUDE_cupcakes_eaten_equals_packaged_l2314_231433


namespace NUMINAMATH_CALUDE_system_solution_l2314_231481

theorem system_solution :
  ∃! (x y : ℚ), (4 * x - 3 * y = -2) ∧ (8 * x + 5 * y = 7) ∧ x = 1/4 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2314_231481


namespace NUMINAMATH_CALUDE_jackie_has_six_apples_l2314_231492

/-- The number of apples Adam has -/
def adam_apples : ℕ := 9

/-- The difference between Adam's and Jackie's apples -/
def difference : ℕ := 3

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := adam_apples - difference

theorem jackie_has_six_apples : jackie_apples = 6 := by sorry

end NUMINAMATH_CALUDE_jackie_has_six_apples_l2314_231492


namespace NUMINAMATH_CALUDE_oblique_line_plane_angle_range_l2314_231459

-- Define the angle between an oblique line and a plane
def angle_oblique_line_plane (θ : Real) : Prop := 
  θ > 0 ∧ θ < Real.pi / 2

-- Theorem statement
theorem oblique_line_plane_angle_range :
  ∀ θ : Real, angle_oblique_line_plane θ ↔ 0 < θ ∧ θ < Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_oblique_line_plane_angle_range_l2314_231459


namespace NUMINAMATH_CALUDE_lcm_of_12_16_15_l2314_231404

theorem lcm_of_12_16_15 : Nat.lcm (Nat.lcm 12 16) 15 = 240 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_12_16_15_l2314_231404


namespace NUMINAMATH_CALUDE_number_problem_l2314_231494

theorem number_problem : ∃ x : ℚ, (35 / 100) * x = (40 / 100) * 50 ∧ x = 400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2314_231494


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2314_231473

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) : 
  a^3 + b^3 = 1008 := by sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2314_231473


namespace NUMINAMATH_CALUDE_square_area_from_circles_l2314_231448

/-- Given two circles where one passes through the center of and is tangent to the other,
    and the other is inscribed in a square, this theorem proves the area of the square
    given the area of the first circle. -/
theorem square_area_from_circles (circle_I circle_II : Real → Prop) (square : Real → Prop) : 
  (∃ r R s : Real,
    -- Circle I has area 9π
    circle_I r ∧ π * r^2 = 9 * π ∧
    -- Circle I passes through center of and is tangent to Circle II
    circle_II R ∧ R = 2 * r ∧
    -- Circle II is inscribed in the square
    square s ∧ s = 2 * R) →
  (∃ area : Real, square area ∧ area = 36) :=
by sorry

end NUMINAMATH_CALUDE_square_area_from_circles_l2314_231448


namespace NUMINAMATH_CALUDE_age_difference_l2314_231463

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 17) : A = C + 17 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2314_231463


namespace NUMINAMATH_CALUDE_sum_of_powers_of_three_l2314_231450

theorem sum_of_powers_of_three : (-3)^4 + (-3)^2 + (-3)^0 + 3^0 + 3^2 + 3^4 = 182 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_three_l2314_231450


namespace NUMINAMATH_CALUDE_parabola_vertex_and_point_l2314_231410

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y (p : Parabola) (x : ℚ) : ℚ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_vertex_and_point (p : Parabola) :
  p.y 2 = 1 → p.y 0 = 5 → p.a + p.b - p.c = -8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_and_point_l2314_231410


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l2314_231422

theorem price_decrease_percentage (initial_price : ℝ) (h : initial_price > 0) : 
  let increased_price := initial_price * (1 + 0.25)
  let decrease_percentage := (increased_price - initial_price) / increased_price * 100
  decrease_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l2314_231422


namespace NUMINAMATH_CALUDE_sequence_negative_term_l2314_231431

theorem sequence_negative_term
  (k : ℝ) (h_k : 0 < k ∧ k < 1)
  (a : ℕ → ℝ)
  (h_a : ∀ n : ℕ, n ≥ 1 → a (n + 1) ≤ (1 + k / n) * a n - 1) :
  ∃ t : ℕ, a t < 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_negative_term_l2314_231431


namespace NUMINAMATH_CALUDE_parabola_shift_l2314_231413

/-- A parabola shifted left and down -/
def shifted_parabola (x y : ℝ) : Prop :=
  y = -(x + 2)^2 - 3

/-- The original parabola -/
def original_parabola (x y : ℝ) : Prop :=
  y = -x^2

/-- Theorem stating that the shifted parabola is equivalent to
    the original parabola shifted 2 units left and 3 units down -/
theorem parabola_shift :
  ∀ x y : ℝ, shifted_parabola x y ↔ original_parabola (x + 2) (y + 3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_shift_l2314_231413


namespace NUMINAMATH_CALUDE_cassies_nail_cutting_l2314_231499

/-- The number of nails Cassie needs to cut -/
def total_nails (num_dogs : ℕ) (num_parrots : ℕ) (dog_feet : ℕ) (dog_nails_per_foot : ℕ) 
                (parrot_legs : ℕ) (parrot_claws_per_leg : ℕ) (extra_claw : ℕ) : ℕ :=
  num_dogs * dog_feet * dog_nails_per_foot + 
  (num_parrots - 1) * parrot_legs * parrot_claws_per_leg + 
  (parrot_legs * parrot_claws_per_leg + extra_claw)

/-- Theorem stating the total number of nails Cassie needs to cut -/
theorem cassies_nail_cutting : 
  total_nails 4 8 4 4 2 3 1 = 113 := by
  sorry

end NUMINAMATH_CALUDE_cassies_nail_cutting_l2314_231499


namespace NUMINAMATH_CALUDE_power_sum_equality_l2314_231408

theorem power_sum_equality (x : ℝ) : x^3 * x + x^2 * x^2 = 2 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2314_231408


namespace NUMINAMATH_CALUDE_square_difference_l2314_231483

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 9/20) 
  (h2 : x - y = 1/20) : 
  x^2 - y^2 = 9/400 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l2314_231483


namespace NUMINAMATH_CALUDE_solve_equation_l2314_231495

theorem solve_equation (x : ℚ) :
  (2 / (x + 2) + 4 / (x + 2) + (2 * x) / (x + 2) = 5) → x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2314_231495


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2314_231486

theorem arithmetic_sequence_middle_term (a : ℕ → ℝ) :
  (a 0 = 3^2) →
  (a 2 = 3^4) →
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →
  (a 1 = 45) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2314_231486


namespace NUMINAMATH_CALUDE_order_of_xyz_l2314_231421

-- Define the variables and their relationships
theorem order_of_xyz (a b c d : ℝ) 
  (h_order : a > b ∧ b > c ∧ c > d ∧ d > 0) 
  (x : ℝ) (hx : x = Real.sqrt (a * b) + Real.sqrt (c * d))
  (y : ℝ) (hy : y = Real.sqrt (a * c) + Real.sqrt (b * d))
  (z : ℝ) (hz : z = Real.sqrt (a * d) + Real.sqrt (b * c)) :
  x > y ∧ y > z :=
by sorry

end NUMINAMATH_CALUDE_order_of_xyz_l2314_231421


namespace NUMINAMATH_CALUDE_sum_of_max_min_xyz_l2314_231419

theorem sum_of_max_min_xyz (x y z : ℝ) 
  (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z ≥ 0) 
  (h4 : 6*x + 5*y + 4*z = 120) : 
  ∃ (max_sum min_sum : ℝ), 
    (∀ (a b c : ℝ), a ≥ b → b ≥ c → c ≥ 0 → 6*a + 5*b + 4*c = 120 → a + b + c ≤ max_sum) ∧
    (∀ (a b c : ℝ), a ≥ b → b ≥ c → c ≥ 0 → 6*a + 5*b + 4*c = 120 → a + b + c ≥ min_sum) ∧
    max_sum + min_sum = 44 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_xyz_l2314_231419


namespace NUMINAMATH_CALUDE_portfolio_growth_portfolio_growth_example_l2314_231491

theorem portfolio_growth (initial_investment : ℝ) (first_year_rate : ℝ) 
  (additional_investment : ℝ) (second_year_rate : ℝ) : ℝ :=
  let first_year_value := initial_investment * (1 + first_year_rate)
  let second_year_initial := first_year_value + additional_investment
  let final_value := second_year_initial * (1 + second_year_rate)
  final_value

theorem portfolio_growth_example : 
  portfolio_growth 80 0.15 28 0.10 = 132 := by
  sorry

end NUMINAMATH_CALUDE_portfolio_growth_portfolio_growth_example_l2314_231491


namespace NUMINAMATH_CALUDE_inequality_proof_l2314_231484

theorem inequality_proof (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h : x₁^2 + x₂^2 + x₃^2 ≤ 1) :
  (x₁*y₁ + x₂*y₂ + x₃*y₃ - 1)^2 ≥ (x₁^2 + x₂^2 + x₃^2 - 1)*(y₁^2 + y₂^2 + y₃^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2314_231484


namespace NUMINAMATH_CALUDE_exists_real_not_in_geometric_sequence_l2314_231430

/-- A geometric sequence is a sequence where the ratio of each term to its preceding term is constant (not zero) -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

/-- There exists a real number that cannot be a term in any geometric sequence -/
theorem exists_real_not_in_geometric_sequence :
  ∃ x : ℝ, ∀ a : ℕ → ℝ, IsGeometricSequence a → ∀ n : ℕ, a n ≠ x :=
sorry

end NUMINAMATH_CALUDE_exists_real_not_in_geometric_sequence_l2314_231430


namespace NUMINAMATH_CALUDE_equilateral_roots_ratio_l2314_231464

theorem equilateral_roots_ratio (a b z₁ z₂ : ℂ) : 
  z₁^2 + a*z₁ + b = 0 → 
  z₂^2 + a*z₂ + b = 0 → 
  z₂ = (Complex.exp (2*Real.pi*Complex.I/3)) * z₁ → 
  a^2 / b = 0 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_roots_ratio_l2314_231464


namespace NUMINAMATH_CALUDE_total_distance_via_intermediate_point_l2314_231414

/-- The total distance traveled from (2, 3) to (-3, 2) via (1, -1) is √17 + 5. -/
theorem total_distance_via_intermediate_point :
  let start : ℝ × ℝ := (2, 3)
  let intermediate : ℝ × ℝ := (1, -1)
  let end_point : ℝ × ℝ := (-3, 2)
  let distance (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  distance start intermediate + distance intermediate end_point = Real.sqrt 17 + 5 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_via_intermediate_point_l2314_231414


namespace NUMINAMATH_CALUDE_short_stack_customers_count_l2314_231405

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- The number of customers who ordered the big stack -/
def big_stack_customers : ℕ := 6

/-- The total number of pancakes made -/
def total_pancakes : ℕ := 57

/-- The number of customers who ordered the short stack -/
def short_stack_customers : ℕ := 9

theorem short_stack_customers_count :
  short_stack_customers * short_stack + big_stack_customers * big_stack = total_pancakes := by
  sorry

end NUMINAMATH_CALUDE_short_stack_customers_count_l2314_231405


namespace NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l2314_231468

/-- Slope angle of a line with given parametric equations -/
theorem slope_angle_of_parametric_line :
  ∀ (t : ℝ),
  let x := -3 + t
  let y := 1 + Real.sqrt 3 * t
  let k := (y - 1) / (x + 3)  -- Slope calculation
  let α := Real.arctan k      -- Angle calculation
  α = π / 3 := by sorry

end NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l2314_231468


namespace NUMINAMATH_CALUDE_f_lower_bound_a_range_when_f2_less_than_4_l2314_231477

noncomputable section

variable (a : ℝ)
variable (h : a > 0)

def f (x : ℝ) : ℝ := |x + 1/a| + |x - a|

theorem f_lower_bound : ∀ x : ℝ, f a x ≥ 2 :=
sorry

theorem a_range_when_f2_less_than_4 : 
  f a 2 < 4 → 1 < a ∧ a < 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_f_lower_bound_a_range_when_f2_less_than_4_l2314_231477


namespace NUMINAMATH_CALUDE_complex_power_sum_l2314_231452

theorem complex_power_sum (w : ℂ) (hw : w^2 - w + 1 = 0) :
  w^102 + w^103 + w^104 + w^105 + w^106 = 2*w + 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2314_231452


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2314_231462

theorem polynomial_remainder (x : ℝ) : 
  let f : ℝ → ℝ := λ x => x^4 - 3*x^3 + 2*x^2 + 11*x - 6
  (f x) % (x - 2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2314_231462


namespace NUMINAMATH_CALUDE_percentage_of_material_A_in_first_solution_l2314_231457

/-- Given two solutions and their mixture, proves the percentage of material A in the first solution -/
theorem percentage_of_material_A_in_first_solution 
  (x : ℝ) -- Percentage of material A in the first solution
  (h1 : x + 80 = 100) -- First solution composition
  (h2 : 30 + 70 = 100) -- Second solution composition
  (h3 : 0.8 * x + 0.2 * 30 = 22) -- Mixture composition
  : x = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_material_A_in_first_solution_l2314_231457


namespace NUMINAMATH_CALUDE_sum_of_digits_product_nines_fives_l2314_231461

/-- Represents a number with n repetitions of a digit --/
def repeatedDigit (digit : Nat) (n : Nat) : Nat :=
  digit * (10^n - 1) / 9

/-- Calculates the sum of digits of a natural number --/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved --/
theorem sum_of_digits_product_nines_fives :
  let nines := repeatedDigit 9 100
  let fives := repeatedDigit 5 100
  sumOfDigits (nines * fives) = 1800 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_product_nines_fives_l2314_231461


namespace NUMINAMATH_CALUDE_f_monotonicity_l2314_231402

noncomputable def f (x : ℝ) := 2 * x^3 - 6 * x^2 + 7

theorem f_monotonicity :
  (∀ x y, x < y ∧ y < 0 → f x < f y) ∧
  (∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x > f y) ∧
  (∀ x y, 2 < x ∧ x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_l2314_231402


namespace NUMINAMATH_CALUDE_correct_assignment_l2314_231440

-- Define the colors and labels
inductive Color : Type
| White : Color
| Red : Color
| Yellow : Color
| Green : Color

def Label := Color

-- Define a package as a pair of label and actual color
structure Package where
  label : Label
  actual : Color

-- Define the condition that no label matches its actual content
def labelMismatch (p : Package) : Prop := p.label ≠ p.actual

-- Define the set of all packages
def allPackages : Finset Package := sorry

-- Define the property that all labels are different
def allLabelsDifferent (packages : Finset Package) : Prop := sorry

-- Define the property that all actual colors are different
def allActualColorsDifferent (packages : Finset Package) : Prop := sorry

-- Main theorem
theorem correct_assignment :
  ∀ (packages : Finset Package),
    packages = allPackages →
    (∀ p ∈ packages, labelMismatch p) →
    allLabelsDifferent packages →
    allActualColorsDifferent packages →
    ∃! (w r y g : Package),
      w ∈ packages ∧ r ∈ packages ∧ y ∈ packages ∧ g ∈ packages ∧
      w.label = Color.Red ∧ w.actual = Color.White ∧
      r.label = Color.White ∧ r.actual = Color.Red ∧
      y.label = Color.Green ∧ y.actual = Color.Yellow ∧
      g.label = Color.Yellow ∧ g.actual = Color.Green :=
by sorry

end NUMINAMATH_CALUDE_correct_assignment_l2314_231440


namespace NUMINAMATH_CALUDE_hanna_has_zero_erasers_l2314_231424

/-- The number of erasers Tanya has -/
def tanya_total : ℕ := 30

/-- The number of red erasers Tanya has -/
def tanya_red : ℕ := tanya_total / 2

/-- The number of blue erasers Tanya has -/
def tanya_blue : ℕ := tanya_total / 3

/-- The number of yellow erasers Tanya has -/
def tanya_yellow : ℕ := tanya_total - tanya_red - tanya_blue

/-- Rachel's erasers in terms of Tanya's red erasers -/
def rachel_erasers : ℤ := tanya_red / 3 - 5

/-- Hanna's erasers in terms of Rachel's -/
def hanna_erasers : ℤ := 3 * rachel_erasers

theorem hanna_has_zero_erasers :
  tanya_yellow = 2 * tanya_blue → hanna_erasers = 0 := by sorry

end NUMINAMATH_CALUDE_hanna_has_zero_erasers_l2314_231424


namespace NUMINAMATH_CALUDE_fraction_addition_l2314_231485

theorem fraction_addition : (1 : ℚ) / 420 + 19 / 35 = 229 / 420 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2314_231485


namespace NUMINAMATH_CALUDE_max_length_valid_progression_l2314_231436

/-- An arithmetic progression of natural numbers. -/
structure ArithmeticProgression :=
  (first : ℕ)
  (diff : ℕ)
  (len : ℕ)

/-- Check if a natural number contains the digit 9. -/
def containsNine (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 9

/-- An arithmetic progression satisfying the given conditions. -/
def ValidProgression (ap : ArithmeticProgression) : Prop :=
  ap.diff ≠ 0 ∧
  ∀ i : ℕ, i < ap.len → ¬containsNine (ap.first + i * ap.diff)

/-- The main theorem: The maximum length of a valid progression is 72. -/
theorem max_length_valid_progression :
  ∀ ap : ArithmeticProgression, ValidProgression ap → ap.len ≤ 72 :=
sorry

end NUMINAMATH_CALUDE_max_length_valid_progression_l2314_231436


namespace NUMINAMATH_CALUDE_red_card_selections_count_l2314_231432

/-- A modified deck of cards with specific properties -/
structure ModifiedDeck :=
  (total_cards : Nat)
  (num_suits : Nat)
  (cards_per_suit : Nat)
  (red_suits : Nat)
  (black_suits : Nat)

/-- Conditions for the modified deck -/
def deck_conditions (d : ModifiedDeck) : Prop :=
  d.total_cards = 36 ∧
  d.num_suits = 3 ∧
  d.cards_per_suit = 12 ∧
  d.red_suits = 2 ∧
  d.black_suits = 1

/-- Number of ways to select two different cards from red suits -/
def red_card_selections (d : ModifiedDeck) : Nat :=
  (d.red_suits * d.cards_per_suit) * (d.red_suits * d.cards_per_suit - 1)

/-- Theorem: The number of ways to select two different cards from red suits is 552 -/
theorem red_card_selections_count (d : ModifiedDeck) :
  deck_conditions d → red_card_selections d = 552 := by
  sorry

end NUMINAMATH_CALUDE_red_card_selections_count_l2314_231432


namespace NUMINAMATH_CALUDE_interval_length_implies_difference_l2314_231439

theorem interval_length_implies_difference (a b : ℝ) : 
  (∃ x, 3 * a ≤ 4 * x + 6 ∧ 4 * x + 6 ≤ 3 * b) → 
  ((3 * b - 6) / 4 - (3 * a - 6) / 4 = 15) → 
  b - a = 20 := by
sorry

end NUMINAMATH_CALUDE_interval_length_implies_difference_l2314_231439


namespace NUMINAMATH_CALUDE_sin_double_angle_l2314_231412

theorem sin_double_angle (α : ℝ) (h : Real.sin (α - π/4) = 3/5) : 
  Real.sin (2 * α) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_l2314_231412


namespace NUMINAMATH_CALUDE_distance_to_gym_l2314_231407

theorem distance_to_gym (home_to_grocery : ℝ) (grocery_to_gym_speed : ℝ) 
  (time_difference : ℝ) :
  home_to_grocery = 200 →
  grocery_to_gym_speed = 2 →
  time_difference = 50 →
  grocery_to_gym_speed = 2 * (home_to_grocery / 200) →
  (200 / (home_to_grocery / 200)) - (200 / grocery_to_gym_speed) = time_difference →
  200 / grocery_to_gym_speed = 300 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_gym_l2314_231407


namespace NUMINAMATH_CALUDE_organization_member_count_l2314_231489

/-- Represents an organization with committees and members -/
structure Organization where
  num_committees : Nat
  num_members : Nat
  member_committee_count : Nat
  pair_common_member_count : Nat

/-- The specific organization described in the problem -/
def specific_org : Organization :=
  { num_committees := 5
  , num_members := 10
  , member_committee_count := 2
  , pair_common_member_count := 1
  }

/-- Theorem stating that the organization with the given properties has 10 members -/
theorem organization_member_count :
  ∀ (org : Organization),
    org.num_committees = 5 ∧
    org.member_committee_count = 2 ∧
    org.pair_common_member_count = 1 →
    org.num_members = 10 := by
  sorry

#check organization_member_count

end NUMINAMATH_CALUDE_organization_member_count_l2314_231489


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2314_231423

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
   x^2 - 2*(m+1)*x + m^2 + 2 = 0 ∧ 
   y^2 - 2*(m+1)*y + m^2 + 2 = 0 ∧ 
   (1/x + 1/y = 1)) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2314_231423


namespace NUMINAMATH_CALUDE_rectangle_area_l2314_231449

-- Define a rectangle type
structure Rectangle where
  width : ℝ
  length : ℝ
  diagonal : ℝ

-- Theorem statement
theorem rectangle_area (r : Rectangle) 
  (h1 : r.length = 2 * r.width) 
  (h2 : r.diagonal = 15 * Real.sqrt 2) : 
  r.width * r.length = 180 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2314_231449


namespace NUMINAMATH_CALUDE_probability_of_seven_in_three_eighths_l2314_231441

theorem probability_of_seven_in_three_eighths :
  let decimal_rep := (3 : ℚ) / 8
  let digits := [3, 7, 5]
  (digits.count 7 : ℚ) / digits.length = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_probability_of_seven_in_three_eighths_l2314_231441


namespace NUMINAMATH_CALUDE_distance_from_movements_l2314_231456

/-- The distance between two points given their net movements --/
theorem distance_from_movements (south west : ℝ) (south_nonneg : 0 ≤ south) (west_nonneg : 0 ≤ west) :
  Real.sqrt (south ^ 2 + west ^ 2) = 50 ↔ south = 30 ∧ west = 40 := by
sorry

end NUMINAMATH_CALUDE_distance_from_movements_l2314_231456


namespace NUMINAMATH_CALUDE_orange_crates_pigeonhole_l2314_231420

theorem orange_crates_pigeonhole :
  ∀ (crate_contents : Fin 150 → ℕ),
  (∀ i, 130 ≤ crate_contents i ∧ crate_contents i ≤ 150) →
  ∃ n : ℕ, 130 ≤ n ∧ n ≤ 150 ∧ (Finset.filter (λ i => crate_contents i = n) Finset.univ).card ≥ 8 :=
by sorry

end NUMINAMATH_CALUDE_orange_crates_pigeonhole_l2314_231420


namespace NUMINAMATH_CALUDE_class_representation_ratio_l2314_231434

theorem class_representation_ratio (boys girls : ℕ) 
  (h1 : boys + girls > 0)  -- ensure non-empty class
  (h2 : (boys : ℚ) / (boys + girls : ℚ) = 3/4 * (girls : ℚ) / (boys + girls : ℚ)) :
  (boys : ℚ) / (boys + girls : ℚ) = 3/7 := by
sorry

end NUMINAMATH_CALUDE_class_representation_ratio_l2314_231434


namespace NUMINAMATH_CALUDE_alan_ticket_count_l2314_231406

theorem alan_ticket_count (alan marcy : ℕ) 
  (total : alan + marcy = 150)
  (marcy_relation : marcy = 5 * alan - 6) :
  alan = 26 := by
sorry

end NUMINAMATH_CALUDE_alan_ticket_count_l2314_231406


namespace NUMINAMATH_CALUDE_function_f_properties_l2314_231458

/-- A function satisfying the given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ ≠ f x₂) ∧
  (∀ x y, f (x + y) = f x * f y)

/-- Theorem stating the properties of the function f -/
theorem function_f_properties (f : ℝ → ℝ) (hf : FunctionF f) :
  f 0 = 1 ∧ ∀ x, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_function_f_properties_l2314_231458


namespace NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_condition_l2314_231411

theorem smallest_positive_integer_satisfying_condition : 
  ∃ (x : ℕ+), (x : ℝ) + 1000 > 1000 * x ∧ 
  ∀ (y : ℕ+), ((y : ℝ) + 1000 > 1000 * y → x ≤ y) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_satisfying_condition_l2314_231411


namespace NUMINAMATH_CALUDE_x_value_l2314_231496

def M (x : ℝ) : Set ℝ := {2, 0, x}
def N : Set ℝ := {0, 1}

theorem x_value (h : N ⊆ M x) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2314_231496


namespace NUMINAMATH_CALUDE_product_of_extremes_is_cube_l2314_231480

theorem product_of_extremes_is_cube (a : Fin 2022 → ℕ)
  (h : ∀ i : Fin 2021, ∃ k : ℕ, a i * a (i.succ) = k^3) :
  ∃ m : ℕ, a 0 * a 2021 = m^3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_extremes_is_cube_l2314_231480


namespace NUMINAMATH_CALUDE_overlap_area_is_0_15_l2314_231425

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its vertices -/
structure Square where
  v1 : Point
  v2 : Point
  v3 : Point
  v4 : Point

/-- A triangle defined by its vertices -/
structure Triangle where
  v1 : Point
  v2 : Point
  v3 : Point

/-- The area of the overlapping region between a square and a triangle -/
def overlapArea (s : Square) (t : Triangle) : ℝ := sorry

/-- The theorem stating the area of overlap between the specific square and triangle -/
theorem overlap_area_is_0_15 :
  let s := Square.mk
    (Point.mk 0 0)
    (Point.mk 2 0)
    (Point.mk 2 2)
    (Point.mk 0 2)
  let t := Triangle.mk
    (Point.mk 3 0)
    (Point.mk 1 2)
    (Point.mk 2 1)
  overlapArea s t = 0.15 := by sorry

end NUMINAMATH_CALUDE_overlap_area_is_0_15_l2314_231425


namespace NUMINAMATH_CALUDE_preimage_of_one_seven_l2314_231479

/-- The mapping function from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- The theorem stating that (4, -3) is the preimage of (1, 7) under f -/
theorem preimage_of_one_seven :
  f (4, -3) = (1, 7) ∧ 
  ∀ p : ℝ × ℝ, f p = (1, 7) → p = (4, -3) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_one_seven_l2314_231479


namespace NUMINAMATH_CALUDE_boat_distance_theorem_l2314_231443

/-- Calculates the distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat traveling downstream for 5 hours covers 125 km -/
theorem boat_distance_theorem (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : stream_speed = 5)
  (h3 : time = 5) :
  distance_downstream boat_speed stream_speed time = 125 := by
  sorry

#check boat_distance_theorem

end NUMINAMATH_CALUDE_boat_distance_theorem_l2314_231443


namespace NUMINAMATH_CALUDE_earthworm_catches_centipede_l2314_231416

/-- The time it takes for an earthworm to catch up to a centipede under specific conditions -/
theorem earthworm_catches_centipede : 
  let centipede_speed : ℚ := 5 / 3  -- meters per minute
  let earthworm_speed : ℚ := 5 / 2  -- meters per minute
  let initial_distance : ℚ := 20   -- meters
  let relative_speed : ℚ := earthworm_speed - centipede_speed
  let catch_up_time : ℚ := initial_distance / relative_speed
  catch_up_time = 24 := by sorry

end NUMINAMATH_CALUDE_earthworm_catches_centipede_l2314_231416


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l2314_231415

theorem quadratic_equation_solutions :
  ∀ x : ℝ, x * (x - 1) = 1 - x ↔ x = 1 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l2314_231415


namespace NUMINAMATH_CALUDE_factorization_equality_l2314_231409

theorem factorization_equality (a b : ℝ) : 12 * b^3 - 3 * a^2 * b = 3 * b * (2*b + a) * (2*b - a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2314_231409


namespace NUMINAMATH_CALUDE_problem_solution_l2314_231446

theorem problem_solution : ∃ x : ℚ, x + (1/4 * x) = 90 - (30/100 * 90) ∧ x = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2314_231446


namespace NUMINAMATH_CALUDE_another_divisor_of_increased_number_l2314_231445

theorem another_divisor_of_increased_number : ∃ (n : ℕ), n ≠ 12 ∧ n ≠ 30 ∧ n ≠ 74 ∧ n ≠ 100 ∧ (44402 + 2) % n = 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_another_divisor_of_increased_number_l2314_231445


namespace NUMINAMATH_CALUDE_range_of_f_l2314_231469

def f (x : ℝ) := -x^2 + 2*x + 3

theorem range_of_f : 
  ∀ y ∈ Set.Icc (-5 : ℝ) 4, ∃ x ∈ Set.Icc (-2 : ℝ) 3, f x = y ∧
  ∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ∈ Set.Icc (-5 : ℝ) 4 :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_f_l2314_231469


namespace NUMINAMATH_CALUDE_sequence_consecutive_product_l2314_231444

/-- The nth term of the sequence, represented as n 1's followed by n 2's -/
def sequence_term (n : ℕ) : ℕ := 
  (10^n - 1) * (10^n + 2)

/-- The first factor of the product -/
def factor1 (n : ℕ) : ℕ := 
  (10^n - 1) / 3

/-- The second factor of the product -/
def factor2 (n : ℕ) : ℕ := 
  (10^n + 2) / 3

theorem sequence_consecutive_product (n : ℕ) : 
  sequence_term n = factor1 n * factor2 n ∧ factor2 n = factor1 n + 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_consecutive_product_l2314_231444


namespace NUMINAMATH_CALUDE_spending_difference_is_30_l2314_231438

-- Define the quantities and prices
def ice_cream_cartons : ℕ := 10
def yoghurt_cartons : ℕ := 4
def ice_cream_price : ℚ := 4
def yoghurt_price : ℚ := 1

-- Define the discount and tax rates
def ice_cream_discount : ℚ := 15 / 100
def sales_tax : ℚ := 5 / 100

-- Define the function to calculate the difference in spending
def difference_in_spending : ℚ :=
  let ice_cream_cost := ice_cream_cartons * ice_cream_price
  let ice_cream_discounted := ice_cream_cost * (1 - ice_cream_discount)
  let yoghurt_cost := yoghurt_cartons * yoghurt_price
  ice_cream_discounted - yoghurt_cost

-- Theorem statement
theorem spending_difference_is_30 : difference_in_spending = 30 := by
  sorry

end NUMINAMATH_CALUDE_spending_difference_is_30_l2314_231438


namespace NUMINAMATH_CALUDE_unique_square_solution_l2314_231493

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digits_match (abc adeff : ℕ) : Prop :=
  let abc_digits := [abc / 100, (abc / 10) % 10, abc % 10]
  let adeff_digits := [adeff / 10000, (adeff / 1000) % 10, (adeff / 100) % 10, (adeff / 10) % 10, adeff % 10]
  (abc_digits.head? = adeff_digits.head?) ∧
  (abc_digits.get? 2 = adeff_digits.get? 3) ∧
  (abc_digits.get? 2 = adeff_digits.get? 4)

theorem unique_square_solution :
  ∀ abc adeff : ℕ,
    is_three_digit abc →
    is_five_digit adeff →
    abc ^ 2 = adeff →
    digits_match abc adeff →
    abc = 138 ∧ adeff = 19044 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_solution_l2314_231493


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2314_231478

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 1/b ≥ 2 ∧ (1/a + 1/b = 2 ↔ a = 1 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2314_231478


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_l2314_231435

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

structure Line3D where
  point : Point3D
  direction : Point3D

-- Define perpendicularity between a line and a plane
def isPerpendicular (l : Line3D) (p : Plane) : Prop :=
  l.direction.x * p.a + l.direction.y * p.b + l.direction.z * p.c = 0

-- State the theorem
theorem unique_perpendicular_line 
  (P : Point3D) (π : Plane) : 
  ∃! l : Line3D, l.point = P ∧ isPerpendicular l π :=
sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_l2314_231435


namespace NUMINAMATH_CALUDE_tan_two_fifths_pi_plus_theta_l2314_231428

theorem tan_two_fifths_pi_plus_theta (θ : ℝ) 
  (h : Real.sin ((12 / 5) * Real.pi + θ) + 2 * Real.sin ((11 / 10) * Real.pi - θ) = 0) : 
  Real.tan ((2 / 5) * Real.pi + θ) = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_two_fifths_pi_plus_theta_l2314_231428


namespace NUMINAMATH_CALUDE_job_completion_proof_l2314_231471

/-- The number of days it takes for A to complete the job alone -/
def days_A : ℝ := 10

/-- The number of days A and B work together -/
def days_together : ℝ := 4

/-- The fraction of the job completed after A and B work together -/
def fraction_completed : ℝ := 0.6

/-- The number of days it takes for B to complete the job alone -/
def days_B : ℝ := 20

theorem job_completion_proof :
  (days_together * (1 / days_A + 1 / days_B) = fraction_completed) ∧
  (days_B = 20) := by sorry

end NUMINAMATH_CALUDE_job_completion_proof_l2314_231471


namespace NUMINAMATH_CALUDE_sequence_inequality_l2314_231417

-- Define a non-negative sequence
def non_negative_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, 0 ≤ a n

-- Define the condition for the sequence
def seq_condition (a : ℕ → ℝ) : Prop :=
  ∀ m n, a (m + n) ≤ a m + a n

-- State the theorem
theorem sequence_inequality (a : ℕ → ℝ) 
  (h_non_neg : non_negative_seq a) 
  (h_condition : seq_condition a) :
  ∀ m n, m ≤ n → a n ≤ m * a 1 + (n / m - 1) * a m :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l2314_231417


namespace NUMINAMATH_CALUDE_coloring_methods_difference_l2314_231437

/-- A four-sided pyramid -/
structure FourSidedPyramid :=
  (vertices : Fin 5 → Color)

/-- Colors available for coloring the pyramid -/
inductive Color
  | Red | Blue | Green | Yellow | Purple

/-- The number of coloring methods for a four-sided pyramid with n colors available -/
def coloringMethods (n : ℕ) : ℕ :=
  match n with
  | 4 => 72
  | 5 => 420
  | _ => 0

theorem coloring_methods_difference :
  coloringMethods 5 - coloringMethods 4 = 348 := by
  sorry

end NUMINAMATH_CALUDE_coloring_methods_difference_l2314_231437


namespace NUMINAMATH_CALUDE_elsa_token_count_l2314_231451

/-- The number of tokens Angus has -/
def angus_tokens : ℕ := 55

/-- The value of each token in dollars -/
def token_value : ℕ := 4

/-- The difference in dollar value between Elsa's and Angus's tokens -/
def value_difference : ℕ := 20

/-- The number of tokens Elsa has -/
def elsa_tokens : ℕ := 60

theorem elsa_token_count : elsa_tokens = 60 := by
  sorry

end NUMINAMATH_CALUDE_elsa_token_count_l2314_231451


namespace NUMINAMATH_CALUDE_complement_of_M_l2314_231460

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 < 2*x}

-- State the theorem
theorem complement_of_M :
  (Set.univ : Set ℝ) \ M = {x : ℝ | x ≤ 0} ∪ {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l2314_231460


namespace NUMINAMATH_CALUDE_stadium_problem_l2314_231487

theorem stadium_problem (total_start : ℕ) (total_end : ℕ) 
  (h1 : total_start = 600)
  (h2 : total_end = 480) :
  ∃ (boys girls : ℕ),
    boys + girls = total_start ∧
    boys - boys / 4 + girls - girls / 8 = total_end ∧
    girls = 240 := by
  sorry

end NUMINAMATH_CALUDE_stadium_problem_l2314_231487


namespace NUMINAMATH_CALUDE_special_collection_loans_l2314_231429

theorem special_collection_loans (initial_books final_books : ℕ) 
  (return_rate : ℚ) (loaned_books : ℕ) : 
  initial_books = 75 → 
  final_books = 57 → 
  return_rate = 7/10 →
  initial_books - final_books = (1 - return_rate) * loaned_books →
  loaned_books = 60 := by
  sorry

end NUMINAMATH_CALUDE_special_collection_loans_l2314_231429


namespace NUMINAMATH_CALUDE_small_cube_edge_length_l2314_231472

theorem small_cube_edge_length 
  (initial_volume : ℝ) 
  (remaining_volume : ℝ) 
  (num_small_cubes : ℕ) 
  (h1 : initial_volume = 1000) 
  (h2 : remaining_volume = 488) 
  (h3 : num_small_cubes = 8) :
  ∃ (edge_length : ℝ), 
    edge_length = 4 ∧ 
    initial_volume - num_small_cubes * edge_length ^ 3 = remaining_volume :=
by sorry

end NUMINAMATH_CALUDE_small_cube_edge_length_l2314_231472


namespace NUMINAMATH_CALUDE_circle_and_max_z_l2314_231490

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 2)^2 = 4}

-- Define the line on which the center lies
def center_line (x y : ℝ) : Prop :=
  x + y - 3 = 0

-- Define the function z
def z (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

-- Theorem statement
theorem circle_and_max_z :
  (∀ p : ℝ × ℝ, p ∈ circle_C → (p.1 = 1 ∧ p.2 = 4) ∨ (p.1 = 3 ∧ p.2 = 2)) ∧
  (∃ c : ℝ × ℝ, c ∈ circle_C ∧ center_line c.1 c.2) →
  (∀ p : ℝ × ℝ, p ∈ circle_C → (p.1 - 1)^2 + (p.2 - 2)^2 = 4) ∧
  (∀ p : ℝ × ℝ, p ∈ circle_C → z p ≤ 3 + 2 * Real.sqrt 2) ∧
  (∃ p : ℝ × ℝ, p ∈ circle_C ∧ z p = 3 + 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_circle_and_max_z_l2314_231490


namespace NUMINAMATH_CALUDE_range_of_a_l2314_231465

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 > 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, p x → q x a) →
  (∃ x : ℝ, q x a ∧ ¬p x) →
  (0 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2314_231465


namespace NUMINAMATH_CALUDE_smallest_coin_count_l2314_231475

def count_factors (n : ℕ) : ℕ := (Nat.divisors n).card

def count_proper_factors (n : ℕ) : ℕ := (count_factors n) - 2

theorem smallest_coin_count :
  ∀ m : ℕ, m > 0 →
    (count_factors m = 19 ∧ count_proper_factors m = 17) →
    m ≥ 786432 :=
by sorry

end NUMINAMATH_CALUDE_smallest_coin_count_l2314_231475


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2314_231488

theorem polynomial_simplification (p q : ℝ) :
  (4 * q^4 + 2 * p^3 - 7 * p + 8) + (3 * q^4 - 2 * p^3 + 3 * p^2 - 5 * p + 6) =
  7 * q^4 + 3 * p^2 - 12 * p + 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2314_231488


namespace NUMINAMATH_CALUDE_simplify_trig_fraction_l2314_231474

theorem simplify_trig_fraction (x : ℝ) : 
  (1 - Real.sin x - Real.cos x) / (1 - Real.sin x + Real.cos x) = -Real.tan (x / 2) :=
by sorry

end NUMINAMATH_CALUDE_simplify_trig_fraction_l2314_231474


namespace NUMINAMATH_CALUDE_soda_bottles_ordered_l2314_231403

/-- The number of bottles of soda ordered by a store owner in April and May -/
theorem soda_bottles_ordered (april_cases may_cases bottles_per_case : ℕ) 
  (h1 : april_cases = 20)
  (h2 : may_cases = 30)
  (h3 : bottles_per_case = 20) :
  (april_cases + may_cases) * bottles_per_case = 1000 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottles_ordered_l2314_231403


namespace NUMINAMATH_CALUDE_books_loaned_out_is_125_l2314_231418

/-- Represents the inter-library loan program between Library A and Library B -/
structure LibraryLoanProgram where
  initial_collection : ℕ -- Initial number of books in Library A's unique collection
  end_year_collection : ℕ -- Number of books from the unique collection in Library A at year end
  return_rate : ℚ -- Rate of return for books loaned out from Library A's unique collection
  same_year_return_rate : ℚ -- Rate of return within the same year for books from Library A's collection
  b_to_a_loan : ℕ -- Number of books loaned from Library B to Library A
  b_to_a_return_rate : ℚ -- Rate of return for books loaned from Library B to Library A

/-- Calculates the number of books loaned out from Library A's unique collection -/
def books_loaned_out (program : LibraryLoanProgram) : ℕ :=
  sorry

/-- Theorem stating that the number of books loaned out from Library A's unique collection is 125 -/
theorem books_loaned_out_is_125 (program : LibraryLoanProgram) 
  (h1 : program.initial_collection = 150)
  (h2 : program.end_year_collection = 100)
  (h3 : program.return_rate = 3/5)
  (h4 : program.same_year_return_rate = 3/10)
  (h5 : program.b_to_a_loan = 20)
  (h6 : program.b_to_a_return_rate = 1/2) :
  books_loaned_out program = 125 := by
  sorry

end NUMINAMATH_CALUDE_books_loaned_out_is_125_l2314_231418


namespace NUMINAMATH_CALUDE_maria_white_towels_l2314_231454

/-- The number of green towels Maria bought -/
def green_towels : ℕ := 40

/-- The number of towels Maria gave to her mother -/
def towels_given : ℕ := 65

/-- The number of towels Maria ended up with -/
def towels_left : ℕ := 19

/-- The number of white towels Maria bought -/
def white_towels : ℕ := green_towels + towels_given - towels_left

theorem maria_white_towels : white_towels = 44 := by
  sorry

end NUMINAMATH_CALUDE_maria_white_towels_l2314_231454


namespace NUMINAMATH_CALUDE_negation_equivalence_l2314_231497

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x + |x| < 0) ↔ (∀ x : ℝ, x + |x| ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2314_231497


namespace NUMINAMATH_CALUDE_base_13_conversion_l2314_231447

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a Base13Digit to its decimal value -/
def toDecimal (d : Base13Digit) : Nat :=
  match d with
  | Base13Digit.D0 => 0
  | Base13Digit.D1 => 1
  | Base13Digit.D2 => 2
  | Base13Digit.D3 => 3
  | Base13Digit.D4 => 4
  | Base13Digit.D5 => 5
  | Base13Digit.D6 => 6
  | Base13Digit.D7 => 7
  | Base13Digit.D8 => 8
  | Base13Digit.D9 => 9
  | Base13Digit.A => 10
  | Base13Digit.B => 11
  | Base13Digit.C => 12

/-- Converts a three-digit base 13 number to its decimal equivalent -/
def base13ToDecimal (d1 d2 d3 : Base13Digit) : Nat :=
  (toDecimal d1) * 169 + (toDecimal d2) * 13 + (toDecimal d3)

theorem base_13_conversion :
  base13ToDecimal Base13Digit.D1 Base13Digit.D2 Base13Digit.D1 = 196 := by
  sorry

end NUMINAMATH_CALUDE_base_13_conversion_l2314_231447
