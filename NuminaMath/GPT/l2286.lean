import Mathlib

namespace NUMINAMATH_GPT_total_marks_l2286_228692

theorem total_marks (Keith_marks Larry_marks Danny_marks : ℕ)
  (hK : Keith_marks = 3)
  (hL : Larry_marks = 3 * Keith_marks)
  (hD : Danny_marks = Larry_marks + 5) :
  Keith_marks + Larry_marks + Danny_marks = 26 := 
by
  sorry

end NUMINAMATH_GPT_total_marks_l2286_228692


namespace NUMINAMATH_GPT_exponentiation_multiplication_identity_l2286_228663

theorem exponentiation_multiplication_identity :
  (-4)^(2010) * (-0.25)^(2011) = -0.25 :=
by
  sorry

end NUMINAMATH_GPT_exponentiation_multiplication_identity_l2286_228663


namespace NUMINAMATH_GPT_pq_sum_l2286_228618

open Real

section Problem
variables (p q : ℝ)
  (hp : p^3 - 21 * p^2 + 35 * p - 105 = 0)
  (hq : 5 * q^3 - 35 * q^2 - 175 * q + 1225 = 0)

theorem pq_sum : p + q = 21 / 2 :=
sorry
end Problem

end NUMINAMATH_GPT_pq_sum_l2286_228618


namespace NUMINAMATH_GPT_original_number_of_workers_l2286_228628

theorem original_number_of_workers (W A : ℕ)
  (h1 : W * 75 = A)
  (h2 : (W + 10) * 65 = A) :
  W = 65 :=
by
  sorry

end NUMINAMATH_GPT_original_number_of_workers_l2286_228628


namespace NUMINAMATH_GPT_eval_f_at_800_l2286_228682

-- Given conditions in Lean 4:
def f : ℝ → ℝ := sorry -- placeholder for the function definition
axiom func_eqn (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom f_at_1000 : f 1000 = 4

-- The goal/proof statement:
theorem eval_f_at_800 : f 800 = 5 := sorry

end NUMINAMATH_GPT_eval_f_at_800_l2286_228682


namespace NUMINAMATH_GPT_cost_of_five_juices_l2286_228660

-- Given conditions as assumptions
variables {J S : ℝ}

axiom h1 : 2 * S = 6
axiom h2 : S + J = 5

-- Prove the statement
theorem cost_of_five_juices : 5 * J = 10 :=
sorry

end NUMINAMATH_GPT_cost_of_five_juices_l2286_228660


namespace NUMINAMATH_GPT_inverse_matrix_eigenvalues_l2286_228652

theorem inverse_matrix_eigenvalues 
  (c d : ℝ) 
  (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (eigenvalue1 eigenvalue2 : ℝ) 
  (eigenvector1 eigenvector2 : Fin 2 → ℝ) :
  A = ![![1, 2], ![c, d]] →
  eigenvalue1 = 2 →
  eigenvalue2 = 3 →
  eigenvector1 = ![2, 1] →
  eigenvector2 = ![1, 1] →
  (A.vecMul eigenvector1 = (eigenvalue1 • eigenvector1)) →
  (A.vecMul eigenvector2 = (eigenvalue2 • eigenvector2)) →
  A⁻¹ = ![![2 / 3, -1 / 3], ![1 / 6, 1 / 6]] :=
sorry

end NUMINAMATH_GPT_inverse_matrix_eigenvalues_l2286_228652


namespace NUMINAMATH_GPT_least_n_for_obtuse_triangle_l2286_228674

namespace obtuse_triangle

-- Define angles and n
def alpha (n : ℕ) : ℝ := 59 + n * 0.02
def beta : ℝ := 60
def gamma (n : ℕ) : ℝ := 61 - n * 0.02

-- Define condition for the triangle being obtuse
def is_obtuse_triangle (n : ℕ) : Prop :=
  alpha n > 90 ∨ gamma n > 90

-- Statement about the smallest n such that the triangle is obtuse
theorem least_n_for_obtuse_triangle : ∃ n : ℕ, n = 1551 ∧ is_obtuse_triangle n :=
by
  -- existence proof ends here, details for proof to be provided separately
  sorry

end obtuse_triangle

end NUMINAMATH_GPT_least_n_for_obtuse_triangle_l2286_228674


namespace NUMINAMATH_GPT_false_statement_l2286_228651

-- Define the geometrical conditions based on the problem statements
variable {A B C D: Type}

-- A rhombus with equal diagonals is a square
def rhombus_with_equal_diagonals_is_square (R : A) : Prop := 
  ∀ (a b : A), a = b → true

-- A rectangle with perpendicular diagonals is a square
def rectangle_with_perpendicular_diagonals_is_square (Rec : B) : Prop :=
  ∀ (a b : B), a = b → true

-- A parallelogram with perpendicular and equal diagonals is a square
def parallelogram_with_perpendicular_and_equal_diagonals_is_square (P : C) : Prop :=
  ∀ (a b : C), a = b → true

-- A quadrilateral with perpendicular and bisecting diagonals is a square
def quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square (Q : D) : Prop :=
  ∀ (a b : D), (a = b) → true 

-- The main theorem: Statement D is false
theorem false_statement (Q : D) : ¬ (quadrilateral_with_perpendicular_and_bisecting_diagonals_is_square Q) := 
  sorry

end NUMINAMATH_GPT_false_statement_l2286_228651


namespace NUMINAMATH_GPT_evaluate_polynomial_at_4_l2286_228680

-- Define the polynomial f
noncomputable def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

-- Given x = 4, prove that f(4) = 1559
theorem evaluate_polynomial_at_4 : f 4 = 1559 :=
  by
    sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_4_l2286_228680


namespace NUMINAMATH_GPT_brown_eggs_survived_l2286_228638

-- Conditions
variables (B : ℕ)  -- Number of brown eggs that survived

-- States that Linda had three times as many white eggs as brown eggs before the fall
def white_eggs_eq_3_times_brown : Prop := 3 * B + B = 12

-- Theorem statement
theorem brown_eggs_survived (h : white_eggs_eq_3_times_brown B) : B = 3 :=
sorry

end NUMINAMATH_GPT_brown_eggs_survived_l2286_228638


namespace NUMINAMATH_GPT_total_surface_area_correct_l2286_228609

def six_cubes_surface_area : ℕ :=
  let cube_edge := 1
  let cubes := 6
  let initial_surface_area := 6 * cubes -- six faces per cube, total initial surface area
  let hidden_faces := 10 -- determined by counting connections
  initial_surface_area - hidden_faces

theorem total_surface_area_correct : six_cubes_surface_area = 26 := by
  sorry

end NUMINAMATH_GPT_total_surface_area_correct_l2286_228609


namespace NUMINAMATH_GPT_sum_gcd_lcm_eight_twelve_l2286_228695

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end NUMINAMATH_GPT_sum_gcd_lcm_eight_twelve_l2286_228695


namespace NUMINAMATH_GPT_div_by_7_l2286_228656

theorem div_by_7 (n : ℕ) : (3 ^ (12 * n + 1) + 2 ^ (6 * n + 2)) % 7 = 0 := by
  sorry

end NUMINAMATH_GPT_div_by_7_l2286_228656


namespace NUMINAMATH_GPT_seating_arrangement_six_people_l2286_228685

theorem seating_arrangement_six_people : 
  ∃ (n : ℕ), n = 216 ∧ 
  (∀ (a b c d e f : ℕ),
    -- Alice, Bob, and Carla indexing
    1 ≤ a ∧ a ≤ 6 ∧ 
    1 ≤ b ∧ b ≤ 6 ∧ 
    1 ≤ c ∧ c ≤ 6 ∧ 
    a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
    (a ≠ b + 1 ∧ a ≠ b - 1) ∧
    (a ≠ c + 1 ∧ a ≠ c - 1) ∧
    
    -- Derek, Eric, and Fiona indexing
    1 ≤ d ∧ d ≤ 6 ∧ 
    1 ≤ e ∧ e ≤ 6 ∧ 
    1 ≤ f ∧ f ≤ 6 ∧ 
    d ≠ e ∧ d ≠ f ∧ e ≠ f ∧
    (d ≠ e + 1 ∧ d ≠ e - 1) ∧
    (d ≠ f + 1 ∧ d ≠ f - 1)) -> 
  n = 216 := 
sorry

end NUMINAMATH_GPT_seating_arrangement_six_people_l2286_228685


namespace NUMINAMATH_GPT_percent_decrease_l2286_228672

variable (OriginalPrice : ℝ) (SalePrice : ℝ)

theorem percent_decrease : 
  OriginalPrice = 100 → 
  SalePrice = 30 → 
  ((OriginalPrice - SalePrice) / OriginalPrice) * 100 = 70 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_percent_decrease_l2286_228672


namespace NUMINAMATH_GPT_largest_divisor_of_three_consecutive_even_integers_is_sixteen_l2286_228627

theorem largest_divisor_of_three_consecutive_even_integers_is_sixteen (n : ℕ) :
  ∃ d : ℕ, d = 16 ∧ 16 ∣ ((2 * n) * (2 * n + 2) * (2 * n + 4)) :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_three_consecutive_even_integers_is_sixteen_l2286_228627


namespace NUMINAMATH_GPT_goose_eggs_calculation_l2286_228622

theorem goose_eggs_calculation (E : ℝ) (hatch_fraction : ℝ) (survived_first_month_fraction : ℝ) 
(survived_first_year_fraction : ℝ) (survived_first_year : ℝ) (no_more_than_one_per_egg : Prop) 
(h_hatch : hatch_fraction = 1/3) 
(h_month_survival : survived_first_month_fraction = 3/4)
(h_year_survival : survived_first_year_fraction = 2/5)
(h_survived120 : survived_first_year = 120)
(h_no_more_than_one : no_more_than_one_per_egg) :
  E = 1200 :=
by
  -- Convert the information from conditions to formulate the equation
  sorry


end NUMINAMATH_GPT_goose_eggs_calculation_l2286_228622


namespace NUMINAMATH_GPT_probability_of_event_l2286_228684

def is_uniform (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1

theorem probability_of_event : 
  ∀ (a : ℝ), is_uniform a → ∀ (p : ℚ), (3 * a - 1 > 0) → p = 2 / 3 → 
  (∃ b, 0 ≤ b ∧ b ≤ 1 ∧ 3 * b - 1 > 0) := 
by
  intro a h_uniform p h_event h_prob
  sorry

end NUMINAMATH_GPT_probability_of_event_l2286_228684


namespace NUMINAMATH_GPT_min_cost_open_top_rectangular_pool_l2286_228611

theorem min_cost_open_top_rectangular_pool
  (volume : ℝ)
  (depth : ℝ)
  (cost_bottom_per_sqm : ℝ)
  (cost_walls_per_sqm : ℝ)
  (h1 : volume = 18)
  (h2 : depth = 2)
  (h3 : cost_bottom_per_sqm = 200)
  (h4 : cost_walls_per_sqm = 150) :
  ∃ (min_cost : ℝ), min_cost = 5400 :=
by
  sorry

end NUMINAMATH_GPT_min_cost_open_top_rectangular_pool_l2286_228611


namespace NUMINAMATH_GPT_simplify_to_quadratic_form_l2286_228634

noncomputable def simplify_expression (p : ℝ) : ℝ :=
  ((6 * p + 2) - 3 * p * 5) ^ 2 + (5 - 2 / 4) * (8 * p - 12)

theorem simplify_to_quadratic_form (p : ℝ) : simplify_expression p = 81 * p ^ 2 - 50 :=
sorry

end NUMINAMATH_GPT_simplify_to_quadratic_form_l2286_228634


namespace NUMINAMATH_GPT_max_cos_a_l2286_228650

theorem max_cos_a (a b : ℝ) (h : Real.cos (a - b) = Real.cos a - Real.cos b) : 
  Real.cos a ≤ 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_max_cos_a_l2286_228650


namespace NUMINAMATH_GPT_Q_contribution_l2286_228671

def P_contribution : ℕ := 4000
def P_months : ℕ := 12
def Q_months : ℕ := 8
def profit_ratio_PQ : ℚ := 2 / 3

theorem Q_contribution :
  ∃ X : ℕ, (P_contribution * P_months) / (X * Q_months) = profit_ratio_PQ → X = 9000 := 
by sorry

end NUMINAMATH_GPT_Q_contribution_l2286_228671


namespace NUMINAMATH_GPT_dice_composite_probability_l2286_228654

theorem dice_composite_probability :
  let total_outcomes := (8:ℕ)^6
  let non_composite_outcomes := 1 + 4 * 6 
  let composite_probability := 1 - (non_composite_outcomes / total_outcomes) 
  composite_probability = 262119 / 262144 := by
  sorry

end NUMINAMATH_GPT_dice_composite_probability_l2286_228654


namespace NUMINAMATH_GPT_factorize_expression_l2286_228698

-- Defining the variables x and y as real numbers.
variable (x y : ℝ)

-- Statement of the proof problem.
theorem factorize_expression : 
  x * y^2 - x = x * (y + 1) * (y - 1) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l2286_228698


namespace NUMINAMATH_GPT_largest_divisor_of_n_cube_minus_n_minus_six_l2286_228681

theorem largest_divisor_of_n_cube_minus_n_minus_six (n : ℤ) : 6 ∣ (n^3 - n - 6) :=
by sorry

end NUMINAMATH_GPT_largest_divisor_of_n_cube_minus_n_minus_six_l2286_228681


namespace NUMINAMATH_GPT_parabola_focus_at_centroid_l2286_228699

theorem parabola_focus_at_centroid (A B C : ℝ × ℝ) (a : ℝ) 
  (hA : A = (-1, 2))
  (hB : B = (3, 4))
  (hC : C = (4, -6))
  (h_focus : (a/4, 0) = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)) :
  a = 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_at_centroid_l2286_228699


namespace NUMINAMATH_GPT_polar_to_cartesian_l2286_228630

theorem polar_to_cartesian (θ ρ x y : ℝ) (h1 : ρ = 2 * Real.sin θ) (h2 : x = ρ * Real.cos θ) (h3 : y = ρ * Real.sin θ) :
  x^2 + (y - 1)^2 = 1 :=
sorry

end NUMINAMATH_GPT_polar_to_cartesian_l2286_228630


namespace NUMINAMATH_GPT_tangent_lines_parabola_through_point_l2286_228659

theorem tangent_lines_parabola_through_point :
  ∃ (m : ℝ), 
    (∀ (x y : ℝ), y = x ^ 2 + 1 → (y - 0) = m * (x - 0)) 
     ∧ ((m = 2 ∧ y = 2 * x) ∨ (m = -2 ∧ y = -2 * x)) :=
sorry

end NUMINAMATH_GPT_tangent_lines_parabola_through_point_l2286_228659


namespace NUMINAMATH_GPT_find_worst_competitor_l2286_228657

structure Competitor :=
  (name : String)
  (gender : String)
  (generation : String)

-- Define the competitors
def man : Competitor := ⟨"man", "male", "generation1"⟩
def wife : Competitor := ⟨"wife", "female", "generation1"⟩
def son : Competitor := ⟨"son", "male", "generation2"⟩
def sister : Competitor := ⟨"sister", "female", "generation1"⟩

-- Conditions
def opposite_genders (c1 c2 : Competitor) : Prop :=
  c1.gender ≠ c2.gender

def different_generations (c1 c2 : Competitor) : Prop :=
  c1.generation ≠ c2.generation

noncomputable def worst_competitor : Competitor :=
  sister

def is_sibling (c1 c2 : Competitor) : Prop :=
  (c1 = man ∧ c2 = sister) ∨ (c1 = sister ∧ c2 = man)

-- Theorem statement
theorem find_worst_competitor (best_competitor : Competitor) :
  (opposite_genders worst_competitor best_competitor) ∧
  (different_generations worst_competitor best_competitor) ∧
  ∃ (sibling : Competitor), (is_sibling worst_competitor sibling) :=
  sorry

end NUMINAMATH_GPT_find_worst_competitor_l2286_228657


namespace NUMINAMATH_GPT_function_divisibility_l2286_228647

theorem function_divisibility
    (f : ℤ → ℕ)
    (h_pos : ∀ x, 0 < f x)
    (h_div : ∀ m n : ℤ, (f m - f n) % f (m - n) = 0) :
    ∀ m n : ℤ, f m ≤ f n → f m ∣ f n :=
by sorry

end NUMINAMATH_GPT_function_divisibility_l2286_228647


namespace NUMINAMATH_GPT_max_height_reached_threat_to_object_at_70km_l2286_228604

noncomputable def initial_acceleration : ℝ := 20 -- m/s^2
noncomputable def duration : ℝ := 50 -- seconds
noncomputable def gravity : ℝ := 10 -- m/s^2
noncomputable def height_at_max_time : ℝ := 75000 -- meters (75km)

-- Proof that the maximum height reached is 75 km
theorem max_height_reached (a τ g : ℝ) (H : ℝ) (h₀: a = initial_acceleration) (h₁: τ = duration) (h₂: g = gravity) (h₃: H = height_at_max_time) :
  H = 75 * 1000 := 
sorry

-- Proof that the rocket poses a threat to an object located at 70 km
theorem threat_to_object_at_70km (a τ g : ℝ) (H : ℝ) (h₀: a = initial_acceleration) (h₁: τ = duration) (h₂: g = gravity) (h₃: H = height_at_max_time) :
  H > 70 * 1000 :=
sorry

end NUMINAMATH_GPT_max_height_reached_threat_to_object_at_70km_l2286_228604


namespace NUMINAMATH_GPT_find_ages_l2286_228610

theorem find_ages (M F S : ℕ) 
  (h1 : M = 2 * F / 5)
  (h2 : M + 10 = (F + 10) / 2)
  (h3 : S + 10 = 3 * (F + 10) / 4) :
  M = 20 ∧ F = 50 ∧ S = 35 := 
by
  sorry

end NUMINAMATH_GPT_find_ages_l2286_228610


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2286_228665

theorem necessary_but_not_sufficient (a : ℝ) : (a ≠ 1) → (a^2 ≠ 1) → (a ≠ 1) ∧ ¬((a ≠ 1) → (a^2 ≠ 1)) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2286_228665


namespace NUMINAMATH_GPT_fourth_quadrant_point_l2286_228639

theorem fourth_quadrant_point (a : ℤ) (h1 : 2 * a + 6 > 0) (h2 : 3 * a + 3 < 0) :
  (2 * a + 6, 3 * a + 3) = (2, -3) :=
sorry

end NUMINAMATH_GPT_fourth_quadrant_point_l2286_228639


namespace NUMINAMATH_GPT_cherry_tomatoes_ratio_l2286_228669

theorem cherry_tomatoes_ratio (T P B : ℕ) (M : ℕ := 3) (h1 : P = 4 * T) (h2 : B = 4 * P) (h3 : B / 3 = 32) :
  (T : ℚ) / M = 2 :=
by
  sorry

end NUMINAMATH_GPT_cherry_tomatoes_ratio_l2286_228669


namespace NUMINAMATH_GPT_gcd_gx_x_l2286_228620

noncomputable def g (x : ℤ) : ℤ :=
  (3 * x + 5) * (9 * x + 4) * (11 * x + 8) * (x + 11)

theorem gcd_gx_x (x : ℤ) (h : 34914 ∣ x) : Int.gcd (g x) x = 1760 :=
by
  sorry

end NUMINAMATH_GPT_gcd_gx_x_l2286_228620


namespace NUMINAMATH_GPT_f_monotonicity_l2286_228664

noncomputable def f (x : ℝ) : ℝ := abs (x^2 - 1)

theorem f_monotonicity :
  (∀ x y : ℝ, (-1 < x ∧ x < 0 ∧ x < y ∧ y < 0) → f x < f y) ∧
  (∀ x y : ℝ, (1 < x ∧ 1 < y ∧ x < y) → f x < f y) ∧
  (∀ x y : ℝ, (x < -1 ∧ y < -1 ∧ y < x) → f x < f y) ∧
  (∀ x y : ℝ, (0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ y < x) → f x < f y) :=
by
  sorry

end NUMINAMATH_GPT_f_monotonicity_l2286_228664


namespace NUMINAMATH_GPT_divisible_by_5886_l2286_228640

theorem divisible_by_5886 (r b c : ℕ) (h1 : (523000 + r * 1000 + b * 100 + c * 10) % 89 = 0) (h2 : r * b * c = 180) : 
  (523000 + r * 1000 + b * 100 + c * 10) % 5886 = 0 := 
sorry

end NUMINAMATH_GPT_divisible_by_5886_l2286_228640


namespace NUMINAMATH_GPT_number_of_students_l2286_228631

theorem number_of_students (n : ℕ) (h1 : 90 - n = n / 2) : n = 60 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l2286_228631


namespace NUMINAMATH_GPT_jump_rope_total_l2286_228648

theorem jump_rope_total :
  (56 * 3) + (35 * 4) = 308 :=
by
  sorry

end NUMINAMATH_GPT_jump_rope_total_l2286_228648


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l2286_228605

theorem sum_of_squares_of_roots :
  let a := 1
  let b := 8
  let c := -12
  let r1_r2_sum := -(b:ℝ) / a
  let r1_r2_product := (c:ℝ) / a
  (r1_r2_sum) ^ 2 - 2 * r1_r2_product = 88 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l2286_228605


namespace NUMINAMATH_GPT_q_sufficient_but_not_necessary_for_p_l2286_228653

variable (x : ℝ)

def p : Prop := (x - 2) ^ 2 ≤ 1
def q : Prop := 2 / (x - 1) ≥ 1

theorem q_sufficient_but_not_necessary_for_p : 
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) := 
by
  sorry

end NUMINAMATH_GPT_q_sufficient_but_not_necessary_for_p_l2286_228653


namespace NUMINAMATH_GPT_base6_subtraction_proof_l2286_228689

-- Define the operations needed
def base6_add (a b : Nat) : Nat := sorry
def base6_subtract (a b : Nat) : Nat := sorry

axiom base6_add_correct : ∀ (a b : Nat), base6_add a b = (a + b)
axiom base6_subtract_correct : ∀ (a b : Nat), base6_subtract a b = (if a ≥ b then a - b else 0)

-- Define the problem conditions in base 6
def a := 5*6^2 + 5*6^1 + 5*6^0
def b := 5*6^1 + 5*6^0
def c := 2*6^2 + 0*6^1 + 2*6^0

-- Define the expected result
def result := 6*6^2 + 1*6^1 + 4*6^0

-- State the proof problem
theorem base6_subtraction_proof : base6_subtract (base6_add a b) c = result :=
by
  rw [base6_add_correct, base6_subtract_correct]
  sorry

end NUMINAMATH_GPT_base6_subtraction_proof_l2286_228689


namespace NUMINAMATH_GPT_sufficient_m_value_l2286_228600

theorem sufficient_m_value (m : ℕ) : 
  ((8 = m ∨ 9 = m) → 
  (m^2 + m^4 + m^6 + m^8 ≥ 6^3 + 6^5 + 6^7 + 6^9)) := 
by 
  sorry

end NUMINAMATH_GPT_sufficient_m_value_l2286_228600


namespace NUMINAMATH_GPT_largest_x_FloorDiv7_eq_FloorDiv8_plus_1_l2286_228662

-- Definitions based on conditions
def floor_div_7 (x : ℕ) : ℕ := x / 7
def floor_div_8 (x : ℕ) : ℕ := x / 8

-- The statement of the problem
theorem largest_x_FloorDiv7_eq_FloorDiv8_plus_1 :
  ∃ x : ℕ, (floor_div_7 x = floor_div_8 x + 1) ∧ (∀ y : ℕ, floor_div_7 y = floor_div_8 y + 1 → y ≤ x) ∧ x = 104 :=
sorry

end NUMINAMATH_GPT_largest_x_FloorDiv7_eq_FloorDiv8_plus_1_l2286_228662


namespace NUMINAMATH_GPT_inequality_proof_l2286_228625

theorem inequality_proof {x y z : ℝ} (n : ℕ) 
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : x + y + z = 1)
  : (x^4 / (y * (1 - y^n))) + (y^4 / (z * (1 - z^n))) + (z^4 / (x * (1 - x^n))) 
    ≥ (3^n) / (3^(n - 2) - 9) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2286_228625


namespace NUMINAMATH_GPT_find_m_even_fn_l2286_228696

theorem find_m_even_fn (m : ℝ) (f : ℝ → ℝ) 
  (Hf : ∀ x : ℝ, f x = x * (10^x + m * 10^(-x))) 
  (Heven : ∀ x : ℝ, f (-x) = f x) : m = -1 := by
  sorry

end NUMINAMATH_GPT_find_m_even_fn_l2286_228696


namespace NUMINAMATH_GPT_range_of_x_l2286_228661

theorem range_of_x (x : ℝ) (hx1 : 1 / x ≤ 3) (hx2 : 1 / x ≥ -2) : x ≥ 1 / 3 := 
sorry

end NUMINAMATH_GPT_range_of_x_l2286_228661


namespace NUMINAMATH_GPT_ratio_proof_l2286_228612

-- Define x and y as real numbers
variables (x y : ℝ)
-- Define the given condition
def given_condition : Prop := (3 * x - 2 * y) / (2 * x + y) = 3 / 4
-- Define the result to prove
def result : Prop := x / y = 11 / 6

-- State the theorem
theorem ratio_proof (h : given_condition x y) : result x y :=
by 
  sorry

end NUMINAMATH_GPT_ratio_proof_l2286_228612


namespace NUMINAMATH_GPT_smaller_square_area_percentage_is_zero_l2286_228635

noncomputable def area_smaller_square_percentage (r : ℝ) : ℝ :=
  let side_length_larger_square := 2 * r
  let x := 0  -- Solution from the Pythagorean step
  let area_larger_square := side_length_larger_square ^ 2
  let area_smaller_square := x ^ 2
  100 * area_smaller_square / area_larger_square

theorem smaller_square_area_percentage_is_zero (r : ℝ) :
    area_smaller_square_percentage r = 0 :=
  sorry

end NUMINAMATH_GPT_smaller_square_area_percentage_is_zero_l2286_228635


namespace NUMINAMATH_GPT_gcd_of_78_and_104_l2286_228641

theorem gcd_of_78_and_104 : Int.gcd 78 104 = 26 := by
  sorry

end NUMINAMATH_GPT_gcd_of_78_and_104_l2286_228641


namespace NUMINAMATH_GPT_quadratic_function_correct_l2286_228601

-- Defining the quadratic function a
def quadratic_function (x : ℝ) : ℝ := 2 * x^2 - 14 * x + 20

-- Theorem stating that the quadratic function passes through the points (2, 0) and (5, 0)
theorem quadratic_function_correct : 
  quadratic_function 2 = 0 ∧ quadratic_function 5 = 0 := 
by
  -- these proofs are skipped with sorry for now
  sorry

end NUMINAMATH_GPT_quadratic_function_correct_l2286_228601


namespace NUMINAMATH_GPT_prime_integer_roots_l2286_228619

theorem prime_integer_roots (p : ℕ) (hp : Prime p) 
  (hroots : ∀ (x1 x2 : ℤ), x1 * x2 = -512 * p ∧ x1 + x2 = -p) : p = 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_prime_integer_roots_l2286_228619


namespace NUMINAMATH_GPT_range_of_expression_l2286_228668

noncomputable def expression (a b c d : ℝ) : ℝ :=
  Real.sqrt (a^2 + (2 - b)^2) + Real.sqrt (b^2 + (2 - c)^2) + 
  Real.sqrt (c^2 + (2 - d)^2) + Real.sqrt (d^2 + (2 - a)^2)

theorem range_of_expression (a b c d : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 2)
  (h3 : 0 ≤ b) (h4 : b ≤ 2) (h5 : 0 ≤ c) (h6 : c ≤ 2)
  (h7 : 0 ≤ d) (h8 : d ≤ 2) :
  4 * Real.sqrt 2 ≤ expression a b c d ∧ expression a b c d ≤ 16 :=
by
  sorry

end NUMINAMATH_GPT_range_of_expression_l2286_228668


namespace NUMINAMATH_GPT_find_first_number_l2286_228679

open Int

theorem find_first_number (A : ℕ) : 
  (Nat.lcm A 671 = 2310) ∧ (Nat.gcd A 671 = 61) → 
  A = 210 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_first_number_l2286_228679


namespace NUMINAMATH_GPT_partition_count_l2286_228637

noncomputable def count_partition (n : ℕ) : ℕ :=
  -- Function that counts the number of ways to partition n as per the given conditions
  n

theorem partition_count (n : ℕ) (h : n > 0) :
  count_partition n = n :=
sorry

end NUMINAMATH_GPT_partition_count_l2286_228637


namespace NUMINAMATH_GPT_cone_volume_ratio_l2286_228670

theorem cone_volume_ratio (r_C h_C r_D h_D : ℝ) (h_rC : r_C = 20) (h_hC : h_C = 40) 
  (h_rD : r_D = 40) (h_hD : h_D = 20) : 
  (1 / 3 * pi * r_C^2 * h_C) / (1 / 3 * pi * r_D^2 * h_D) = 1 / 2 :=
by
  rw [h_rC, h_hC, h_rD, h_hD]
  sorry

end NUMINAMATH_GPT_cone_volume_ratio_l2286_228670


namespace NUMINAMATH_GPT_maddie_watched_8_episodes_l2286_228606

def minutes_per_episode : ℕ := 44
def minutes_monday : ℕ := 138
def minutes_tuesday_wednesday : ℕ := 0
def minutes_thursday : ℕ := 21
def episodes_friday : ℕ := 2
def minutes_per_episode_friday := episodes_friday * minutes_per_episode
def minutes_weekend : ℕ := 105
def total_minutes := minutes_monday + minutes_tuesday_wednesday + minutes_thursday + minutes_per_episode_friday + minutes_weekend
def answer := total_minutes / minutes_per_episode

theorem maddie_watched_8_episodes : answer = 8 := by
  sorry

end NUMINAMATH_GPT_maddie_watched_8_episodes_l2286_228606


namespace NUMINAMATH_GPT_smallest_angle_in_trapezoid_l2286_228603

theorem smallest_angle_in_trapezoid 
  (a d : ℝ) 
  (h1 : a + 2 * d = 150) 
  (h2 : a + d + a + 2 * d = 180) : 
  a = 90 := 
sorry

end NUMINAMATH_GPT_smallest_angle_in_trapezoid_l2286_228603


namespace NUMINAMATH_GPT_proof_problem_l2286_228626

theorem proof_problem 
  (a b c : ℝ) 
  (h1 : ∀ x, (x < -4 ∨ (23 ≤ x ∧ x ≤ 27)) ↔ ((x - a) * (x - b) / (x - c) ≤ 0))
  (h2 : a < b) : 
  a + 2 * b + 3 * c = 65 :=
sorry

end NUMINAMATH_GPT_proof_problem_l2286_228626


namespace NUMINAMATH_GPT_polynomial_comparison_l2286_228690

theorem polynomial_comparison {x : ℝ} :
  let A := (x - 3) * (x - 2)
  let B := (x + 1) * (x - 6)
  A > B :=
by 
  sorry -- Proof is omitted.

end NUMINAMATH_GPT_polynomial_comparison_l2286_228690


namespace NUMINAMATH_GPT_crow_speed_l2286_228616

/-- Definitions from conditions -/
def distance_between_nest_and_ditch : ℝ := 250 -- in meters
def total_trips : ℕ := 15
def total_hours : ℝ := 1.5 -- hours

/-- The statement to be proved -/
theorem crow_speed :
  let distance_per_trip := 2 * distance_between_nest_and_ditch
  let total_distance := (total_trips : ℝ) * distance_per_trip / 1000 -- convert to kilometers
  let speed := total_distance / total_hours
  speed = 5 := by
  let distance_per_trip := 2 * distance_between_nest_and_ditch
  let total_distance := (total_trips : ℝ) * distance_per_trip / 1000
  let speed := total_distance / total_hours
  sorry

end NUMINAMATH_GPT_crow_speed_l2286_228616


namespace NUMINAMATH_GPT_potato_sales_l2286_228693

theorem potato_sales :
  let total_weight := 6500
  let damaged_weight := 150
  let bag_weight := 50
  let price_per_bag := 72
  let sellable_weight := total_weight - damaged_weight
  let num_bags := sellable_weight / bag_weight
  let total_revenue := num_bags * price_per_bag
  total_revenue = 9144 :=
by
  sorry

end NUMINAMATH_GPT_potato_sales_l2286_228693


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l2286_228694

theorem boat_speed_in_still_water (x : ℕ) 
  (h1 : x + 17 = 77) (h2 : x - 17 = 43) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l2286_228694


namespace NUMINAMATH_GPT_initial_pocket_money_l2286_228678

variable (P : ℝ)

-- Conditions
axiom chocolates_expenditure : P * (1/9) ≥ 0
axiom fruits_expenditure : P * (2/5) ≥ 0
axiom remaining_money : P * (22/45) = 220

-- Theorem statement
theorem initial_pocket_money : P = 450 :=
by
  have h₁ : P * (1/9) + P * (2/5) = P * (23/45) := by sorry
  have h₂ : P * (1 - 23/45) = P * (22/45) := by sorry
  have h₃ : P = 220 / (22/45) := by sorry
  have h₄ : P = 220 * (45/22) := by sorry
  have h₅ : P = 450 := by sorry
  exact h₅

end NUMINAMATH_GPT_initial_pocket_money_l2286_228678


namespace NUMINAMATH_GPT_molecular_weight_of_BaF2_l2286_228691

theorem molecular_weight_of_BaF2 (mw_6_moles : ℕ → ℕ) (h : mw_6_moles 6 = 1050) : mw_6_moles 1 = 175 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_BaF2_l2286_228691


namespace NUMINAMATH_GPT_snowman_volume_l2286_228632

noncomputable def volume_snowman (r₁ r₂ r₃ r_c h_c : ℝ) : ℝ :=
  (4 / 3 * Real.pi * r₁^3) + (4 / 3 * Real.pi * r₂^3) + (4 / 3 * Real.pi * r₃^3) + (Real.pi * r_c^2 * h_c)

theorem snowman_volume 
  : volume_snowman 4 6 8 3 5 = 1101 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_snowman_volume_l2286_228632


namespace NUMINAMATH_GPT_mode_of_list_is_five_l2286_228623

def list := [3, 4, 5, 5, 5, 5, 7, 11, 21]

def occurrence_count (l : List ℕ) (x : ℕ) : ℕ :=
  l.count x

def is_mode (l : List ℕ) (x : ℕ) : Prop :=
  ∀ y : ℕ, occurrence_count l x ≥ occurrence_count l y

theorem mode_of_list_is_five : is_mode list 5 := by
  sorry

end NUMINAMATH_GPT_mode_of_list_is_five_l2286_228623


namespace NUMINAMATH_GPT_wombats_count_l2286_228673

theorem wombats_count (W : ℕ) (H : 4 * W + 3 = 39) : W = 9 := 
sorry

end NUMINAMATH_GPT_wombats_count_l2286_228673


namespace NUMINAMATH_GPT_base_conversion_problem_l2286_228608

theorem base_conversion_problem 
  (b x y z : ℕ)
  (h1 : 1987 = x * b^2 + y * b + z)
  (h2 : x + y + z = 25) :
  b = 19 ∧ x = 5 ∧ y = 9 ∧ z = 11 := 
by
  sorry

end NUMINAMATH_GPT_base_conversion_problem_l2286_228608


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l2286_228643

theorem repeating_decimal_to_fraction (h : (0.0909090909 : ℝ) = 1 / 11) : (0.2727272727 : ℝ) = 3 / 11 :=
sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l2286_228643


namespace NUMINAMATH_GPT_probability_at_least_2_defective_is_one_third_l2286_228697

noncomputable def probability_at_least_2_defective (good defective : ℕ) (total_selected : ℕ) : ℚ :=
  let total_ways := Nat.choose (good + defective) total_selected
  let ways_2_defective_1_good := Nat.choose defective 2 * Nat.choose good 1
  let ways_3_defective := Nat.choose defective 3
  (ways_2_defective_1_good + ways_3_defective) / total_ways

theorem probability_at_least_2_defective_is_one_third :
  probability_at_least_2_defective 6 4 3 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_2_defective_is_one_third_l2286_228697


namespace NUMINAMATH_GPT_find_f4_l2286_228667

theorem find_f4 (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = -f (-x + 1)) 
  (h2 : ∀ x, f (x - 1) = f (-x - 1)) 
  (h3 : f 0 = 2) : 
  f 4 = -2 :=
sorry

end NUMINAMATH_GPT_find_f4_l2286_228667


namespace NUMINAMATH_GPT_find_pairs_l2286_228642

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ a b, (a, b) = (2, 2) ∨ (a, b) = (1, 3) ∨ (a, b) = (3, 3))
  ↔ (∃ a b, a > 0 ∧ b > 0 ∧ (a^3 * b - 1) % (a + 1) = 0 ∧ (b^3 * a + 1) % (b - 1) = 0) := by
  sorry

end NUMINAMATH_GPT_find_pairs_l2286_228642


namespace NUMINAMATH_GPT_max_min_diff_c_l2286_228675

variable (a b c : ℝ)

theorem max_min_diff_c (h1 : a + b + c = 6) (h2 : a^2 + b^2 + c^2 = 18) : 
  (4 - 0) = 4 :=
by
  sorry

end NUMINAMATH_GPT_max_min_diff_c_l2286_228675


namespace NUMINAMATH_GPT_c_a_plus_c_b_geq_a_a_plus_b_b_l2286_228621

theorem c_a_plus_c_b_geq_a_a_plus_b_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (c : ℚ) (h : c = (a^(a+1) + b^(b+1)) / (a^a + b^b)) :
  c^a + c^b ≥ a^a + b^b :=
sorry

end NUMINAMATH_GPT_c_a_plus_c_b_geq_a_a_plus_b_b_l2286_228621


namespace NUMINAMATH_GPT_maximum_rectangle_area_l2286_228607

-- Define the perimeter condition
def perimeter (rectangle : ℝ × ℝ) : ℝ :=
  2 * rectangle.fst + 2 * rectangle.snd

-- Define the area function
def area (rectangle : ℝ × ℝ) : ℝ :=
  rectangle.fst * rectangle.snd

-- Define the question statement in terms of Lean
theorem maximum_rectangle_area (length_width : ℝ × ℝ) (h : perimeter length_width = 32) : 
  area length_width ≤ 64 :=
sorry

end NUMINAMATH_GPT_maximum_rectangle_area_l2286_228607


namespace NUMINAMATH_GPT_cos_cofunction_identity_l2286_228617

theorem cos_cofunction_identity (α : ℝ) (h : Real.sin (30 * Real.pi / 180 + α) = Real.sqrt 3 / 2) :
  Real.cos (60 * Real.pi / 180 - α) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_cofunction_identity_l2286_228617


namespace NUMINAMATH_GPT_smallest_among_neg2_cube_neg3_square_neg_neg1_l2286_228629

def smallest_among (a b c : ℤ) : ℤ :=
if a < b then
  if a < c then a else c
else
  if b < c then b else c

theorem smallest_among_neg2_cube_neg3_square_neg_neg1 :
  smallest_among ((-2)^3) (-(3^2)) (-(-1)) = -(3^2) :=
by
  sorry

end NUMINAMATH_GPT_smallest_among_neg2_cube_neg3_square_neg_neg1_l2286_228629


namespace NUMINAMATH_GPT_sum_of_first_ten_terms_seq_l2286_228633

def a₁ : ℤ := -5
def d : ℤ := 6
def n : ℕ := 10

theorem sum_of_first_ten_terms_seq : (n * (a₁ + a₁ + (n - 1) * d)) / 2 = 220 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_ten_terms_seq_l2286_228633


namespace NUMINAMATH_GPT_triangle_area_arithmetic_sequence_l2286_228646

theorem triangle_area_arithmetic_sequence :
  ∃ (S_1 S_2 S_3 S_4 S_5 : ℝ) (d : ℝ),
  S_1 + S_2 + S_3 + S_4 + S_5 = 420 ∧
  S_2 = S_1 + d ∧
  S_3 = S_1 + 2 * d ∧
  S_4 = S_1 + 3 * d ∧
  S_5 = S_1 + 4 * d ∧
  S_5 = 112 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_arithmetic_sequence_l2286_228646


namespace NUMINAMATH_GPT_cake_pieces_per_sister_l2286_228649

theorem cake_pieces_per_sister (total_pieces : ℕ) (percentage_eaten : ℕ) (sisters : ℕ)
  (h1 : total_pieces = 240) (h2 : percentage_eaten = 60) (h3 : sisters = 3) :
  (total_pieces * (1 - percentage_eaten / 100)) / sisters = 32 :=
by
  sorry

end NUMINAMATH_GPT_cake_pieces_per_sister_l2286_228649


namespace NUMINAMATH_GPT_bodies_distance_apart_l2286_228644

def distance_fallen (t : ℝ) : ℝ := 4.9 * t^2

theorem bodies_distance_apart (t : ℝ) (h₁ : 220.5 = distance_fallen t - distance_fallen (t - 5)) : t = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_bodies_distance_apart_l2286_228644


namespace NUMINAMATH_GPT_discount_rate_l2286_228683

theorem discount_rate (marked_price selling_price discount_rate: ℝ) 
  (h₁: marked_price = 80)
  (h₂: selling_price = 68)
  (h₃: discount_rate = ((marked_price - selling_price) / marked_price) * 100) : 
  discount_rate = 15 :=
by
  sorry

end NUMINAMATH_GPT_discount_rate_l2286_228683


namespace NUMINAMATH_GPT_digit_in_2017th_place_l2286_228687

def digit_at_position (n : ℕ) : ℕ := sorry

theorem digit_in_2017th_place :
  digit_at_position 2017 = 7 :=
by sorry

end NUMINAMATH_GPT_digit_in_2017th_place_l2286_228687


namespace NUMINAMATH_GPT_sum_of_uv_l2286_228624

theorem sum_of_uv (u v : ℕ) (hu : 0 < u) (hv : 0 < v) (hv_lt_hu : v < u)
  (area_pent : 6 * u * v = 500) : u + v = 19 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_uv_l2286_228624


namespace NUMINAMATH_GPT_find_constants_l2286_228686

theorem find_constants (t s : ℤ) :
  (∀ x : ℤ, (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + s) = 15 * x^4 - 22 * x^3 + (41 + s) * x^2 - 34 * x + 9 * s) →
  t = -2 ∧ s = s :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_constants_l2286_228686


namespace NUMINAMATH_GPT_purchase_price_of_article_l2286_228658

theorem purchase_price_of_article (P : ℝ) (h : 45 = 0.20 * P + 12) : P = 165 :=
by
  sorry

end NUMINAMATH_GPT_purchase_price_of_article_l2286_228658


namespace NUMINAMATH_GPT_albert_needs_more_money_l2286_228614

def cost_paintbrush : Real := 1.50
def cost_paints : Real := 4.35
def cost_easel : Real := 12.65
def cost_canvas : Real := 7.95
def cost_palette : Real := 3.75
def money_albert_has : Real := 10.60
def total_cost : Real := cost_paintbrush + cost_paints + cost_easel + cost_canvas + cost_palette
def money_needed : Real := total_cost - money_albert_has

theorem albert_needs_more_money : money_needed = 19.60 := by
  sorry

end NUMINAMATH_GPT_albert_needs_more_money_l2286_228614


namespace NUMINAMATH_GPT_Patrick_fish_count_l2286_228666

variable (Angus Patrick Ollie : ℕ)

-- Conditions
axiom h1 : Ollie + 7 = Angus
axiom h2 : Angus = Patrick + 4
axiom h3 : Ollie = 5

-- Theorem statement
theorem Patrick_fish_count : Patrick = 8 := 
by
  sorry

end NUMINAMATH_GPT_Patrick_fish_count_l2286_228666


namespace NUMINAMATH_GPT_obtuse_angle_of_parallel_vectors_l2286_228655

noncomputable def is_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem obtuse_angle_of_parallel_vectors (θ : ℝ) :
  let a := (2, 1 - Real.cos θ)
  let b := (1 + Real.cos θ, 1 / 4)
  is_parallel a b → 90 < θ ∧ θ < 180 → θ = 135 :=
by
  intro ha hb
  sorry

end NUMINAMATH_GPT_obtuse_angle_of_parallel_vectors_l2286_228655


namespace NUMINAMATH_GPT_Kelly_current_baking_powder_l2286_228636

-- Definitions based on conditions
def yesterday_amount : ℝ := 0.4
def difference : ℝ := 0.1
def current_amount : ℝ := yesterday_amount - difference

-- Statement to prove the question == answer given the conditions
theorem Kelly_current_baking_powder : current_amount = 0.3 := 
by
  sorry

end NUMINAMATH_GPT_Kelly_current_baking_powder_l2286_228636


namespace NUMINAMATH_GPT_div_by_5_mul_diff_l2286_228613

theorem div_by_5_mul_diff (x y z : ℤ) (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) :
  5 ∣ ((x - y)^5 + (y - z)^5 + (z - x)^5) :=
by
  sorry

end NUMINAMATH_GPT_div_by_5_mul_diff_l2286_228613


namespace NUMINAMATH_GPT_sum_xyz_l2286_228676

variables {x y z : ℝ}

theorem sum_xyz (hx : x * y = 30) (hy : x * z = 60) (hz : y * z = 90) : 
  x + y + z = 11 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_sum_xyz_l2286_228676


namespace NUMINAMATH_GPT_derivative_value_at_pi_over_12_l2286_228688

open Real

theorem derivative_value_at_pi_over_12 :
  let f (x : ℝ) := cos (2 * x + π / 3)
  deriv f (π / 12) = -2 :=
by
  let f (x : ℝ) := cos (2 * x + π / 3)
  sorry

end NUMINAMATH_GPT_derivative_value_at_pi_over_12_l2286_228688


namespace NUMINAMATH_GPT_greatest_power_of_3_l2286_228615

theorem greatest_power_of_3 (n : ℕ) : 
  (n = 603) → 
  3^603 ∣ (15^n - 6^n + 3^n) ∧ ¬ (3^(603+1) ∣ (15^n - 6^n + 3^n)) :=
by
  intro hn
  cases hn
  sorry

end NUMINAMATH_GPT_greatest_power_of_3_l2286_228615


namespace NUMINAMATH_GPT_evaluate_expression_l2286_228602

theorem evaluate_expression (x : ℝ) (h1 : x^3 + 2 ≠ 0) (h2 : x^3 - 2 ≠ 0) :
  (( (x+2)^3 * (x^2-x+2)^3 / (x^3+2)^3 )^3 * ( (x-2)^3 * (x^2+x+2)^3 / (x^3-2)^3 )^3 ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2286_228602


namespace NUMINAMATH_GPT_product_of_powers_eq_nine_l2286_228677

variable (a : ℕ)

theorem product_of_powers_eq_nine : a^3 * a^6 = a^9 := 
by sorry

end NUMINAMATH_GPT_product_of_powers_eq_nine_l2286_228677


namespace NUMINAMATH_GPT_common_chord_of_circles_l2286_228645

theorem common_chord_of_circles : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 2*x = 0 ∧ x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := 
by 
  sorry

end NUMINAMATH_GPT_common_chord_of_circles_l2286_228645
