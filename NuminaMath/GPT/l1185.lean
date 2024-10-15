import Mathlib

namespace NUMINAMATH_GPT_lathes_equal_parts_processed_15_minutes_l1185_118591

variable (efficiencyA efficiencyB efficiencyC : ℝ)
variable (timeA timeB timeC : ℕ)

/-- Lathe A starts 10 minutes before lathe C -/
def start_time_relation_1 : Prop := timeA + 10 = timeC

/-- Lathe C starts 5 minutes before lathe B -/
def start_time_relation_2 : Prop := timeC + 5 = timeB

/-- After lathe B has been working for 10 minutes, B and C process the same number of parts -/
def parts_processed_relation_1 (efficiencyB efficiencyC : ℝ) : Prop :=
  10 * efficiencyB = (10 + 5) * efficiencyC

/-- After lathe C has been working for 30 minutes, A and C process the same number of parts -/
def parts_processed_relation_2 (efficiencyA efficiencyC : ℝ) : Prop :=
  (30 + 10) * efficiencyA = 30 * efficiencyC

/-- How many minutes after lathe B starts will it have processed the same number of standard parts as lathe A? -/
theorem lathes_equal_parts_processed_15_minutes
  (h₁ : start_time_relation_1 timeA timeC)
  (h₂ : start_time_relation_2 timeC timeB)
  (h₃ : parts_processed_relation_1 efficiencyB efficiencyC)
  (h₄ : parts_processed_relation_2 efficiencyA efficiencyC) :
  ∃ t : ℕ, (t = 15) ∧ ( (timeB + t) * efficiencyB = (timeA + (timeB + t - timeA)) * efficiencyA ) := sorry

end NUMINAMATH_GPT_lathes_equal_parts_processed_15_minutes_l1185_118591


namespace NUMINAMATH_GPT_problem_equivalent_l1185_118537

theorem problem_equivalent (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a + b = 6) (h₃ : a * (a - 6) = x) (h₄ : b * (b - 6) = x) : 
  x = -9 :=
by
  sorry

end NUMINAMATH_GPT_problem_equivalent_l1185_118537


namespace NUMINAMATH_GPT_problem_inequality_l1185_118541

theorem problem_inequality (a : ℝ) :
  (∀ x : ℝ, (1 ≤ x) → (x ≤ 2) → (x^2 + 2 + |x^3 - 2 * x| ≥ a * x)) ↔ (a ≤ 2 * Real.sqrt 2) := 
sorry

end NUMINAMATH_GPT_problem_inequality_l1185_118541


namespace NUMINAMATH_GPT_multiples_7_not_14_l1185_118596

theorem multiples_7_not_14 (n : ℕ) : (n < 500) → (n % 7 = 0) → (n % 14 ≠ 0) → ∃ k, (k = 36) :=
by {
  sorry
}

end NUMINAMATH_GPT_multiples_7_not_14_l1185_118596


namespace NUMINAMATH_GPT_divisibility_323_l1185_118512

theorem divisibility_323 (n : ℕ) : 
  (20^n + 16^n - 3^n - 1) % 323 = 0 ↔ Even n := 
sorry

end NUMINAMATH_GPT_divisibility_323_l1185_118512


namespace NUMINAMATH_GPT_infinite_sequence_no_square_factors_l1185_118549

/-
  Prove that there exist infinitely many positive integers \( n_1 < n_2 < \cdots \)
  such that for all \( i \neq j \), \( n_i + n_j \) has no square factors other than 1.
-/

theorem infinite_sequence_no_square_factors :
  ∃ (n : ℕ → ℕ), (∀ (i j : ℕ), i ≠ j → ∀ p : ℕ, p ≠ 1 → p^2 ∣ (n i + n j) → false) ∧
    ∀ k : ℕ, n k < n (k + 1) :=
sorry

end NUMINAMATH_GPT_infinite_sequence_no_square_factors_l1185_118549


namespace NUMINAMATH_GPT_planter_cost_l1185_118511

-- Define costs
def cost_palm_fern : ℝ := 15.00
def cost_creeping_jenny : ℝ := 4.00
def cost_geranium : ℝ := 3.50

-- Define quantities
def num_creeping_jennies : ℝ := 4
def num_geraniums : ℝ := 4
def num_corners : ℝ := 4

-- Define the total cost
def total_cost : ℝ :=
  (cost_palm_fern
   + (cost_creeping_jenny * num_creeping_jennies)
   + (cost_geranium * num_geraniums))
  * num_corners

-- Prove the total cost is $180.00
theorem planter_cost : total_cost = 180.00 :=
by
  sorry

end NUMINAMATH_GPT_planter_cost_l1185_118511


namespace NUMINAMATH_GPT_value_of_m_l1185_118558

variable (a m : ℝ)
variable (h1 : a > 0)
variable (h2 : -a*m^2 + 2*a*m + 3 = 3)
variable (h3 : m ≠ 0)

theorem value_of_m : m = 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_m_l1185_118558


namespace NUMINAMATH_GPT_calculate_total_students_l1185_118586

-- Define the conditions and state the theorem
theorem calculate_total_students (perc_bio : ℝ) (num_not_bio : ℝ) (perc_not_bio : ℝ) (T : ℝ) :
  perc_bio = 0.475 →
  num_not_bio = 462 →
  perc_not_bio = 1 - perc_bio →
  perc_not_bio * T = num_not_bio →
  T = 880 :=
by
  intros
  -- proof will be here
  sorry

end NUMINAMATH_GPT_calculate_total_students_l1185_118586


namespace NUMINAMATH_GPT_problem_statement_l1185_118577

theorem problem_statement (a b : ℝ) (h_domain : ∀ x, 1 ≤ x ∧ x ≤ b)
  (h_range : ∀ y, 1 ≤ y ∧ y ≤ b) (h_b_gt_1 : b > 1)
  (h1 : a = 1) (h2 : 1/2 * (b - 1)^2 + 1 = b) : a + b = 4 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1185_118577


namespace NUMINAMATH_GPT_average_of_x_y_z_l1185_118535

theorem average_of_x_y_z (x y z : ℝ) (h : (5 / 2) * (x + y + z) = 20) : 
  (x + y + z) / 3 = 8 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_average_of_x_y_z_l1185_118535


namespace NUMINAMATH_GPT_instrument_failure_probability_l1185_118582

noncomputable def probability_of_instrument_not_working (m : ℕ) (P : ℝ) : ℝ :=
  1 - (1 - P)^m

theorem instrument_failure_probability (m : ℕ) (P : ℝ) :
  0 ≤ P → P ≤ 1 → probability_of_instrument_not_working m P = 1 - (1 - P)^m :=
by
  intros _ _
  sorry

end NUMINAMATH_GPT_instrument_failure_probability_l1185_118582


namespace NUMINAMATH_GPT_probability_not_blue_l1185_118513

-- Definitions based on the conditions
def total_faces : ℕ := 12
def blue_faces : ℕ := 1
def non_blue_faces : ℕ := total_faces - blue_faces

-- Statement of the problem
theorem probability_not_blue : (non_blue_faces : ℚ) / total_faces = 11 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_blue_l1185_118513


namespace NUMINAMATH_GPT_functional_equation_identity_l1185_118570

def f : ℝ → ℝ := sorry

theorem functional_equation_identity (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x * f y + y) = f (f (x * y)) + y) : 
  ∀ y : ℝ, f y = y :=
sorry

end NUMINAMATH_GPT_functional_equation_identity_l1185_118570


namespace NUMINAMATH_GPT_exists_quadratic_sequence_for_any_b_c_smallest_n_for_quadratic_sequence_0_to_2021_l1185_118515

noncomputable def quadratic_sequence (n : ℕ) (a : ℕ → ℤ) :=
  ∀i : ℕ, 1 ≤ i ∧ i ≤ n → abs (a i - a (i - 1)) = i * i

theorem exists_quadratic_sequence_for_any_b_c (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ quadratic_sequence n a := by
  sorry

theorem smallest_n_for_quadratic_sequence_0_to_2021 :
  ∃ n : ℕ, 0 < n ∧ ∀ (a : ℕ → ℤ), a 0 = 0 → a n = 2021 → quadratic_sequence n a := by
  sorry

end NUMINAMATH_GPT_exists_quadratic_sequence_for_any_b_c_smallest_n_for_quadratic_sequence_0_to_2021_l1185_118515


namespace NUMINAMATH_GPT_three_digit_multiples_of_36_eq_25_l1185_118504

-- Definition: A three-digit number is between 100 and 999
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

-- Definition: A number is a multiple of both 4 and 9 if and only if it's a multiple of 36
def is_multiple_of_36 (n : ℕ) : Prop := n % 36 = 0

-- Definition: Count of three-digit integers that are multiples of 36
def count_multiples_of_36 : ℕ :=
  (999 / 36) - (100 / 36) + 1

-- Theorem: There are 25 three-digit integers that are multiples of 36
theorem three_digit_multiples_of_36_eq_25 : count_multiples_of_36 = 25 := by
  sorry

end NUMINAMATH_GPT_three_digit_multiples_of_36_eq_25_l1185_118504


namespace NUMINAMATH_GPT_proposition_p_proposition_not_q_proof_p_and_not_q_l1185_118547

variable (p : Prop)
variable (q : Prop)
variable (r : Prop)

theorem proposition_p : (∃ x0 : ℝ, x0 > 2) := sorry

theorem proposition_not_q : ¬ (∀ x : ℝ, x^3 > x^2) := sorry

theorem proof_p_and_not_q : (∃ x0 : ℝ, x0 > 2) ∧ ¬ (∀ x : ℝ, x^3 > x^2) :=
by
  exact ⟨proposition_p, proposition_not_q⟩

end NUMINAMATH_GPT_proposition_p_proposition_not_q_proof_p_and_not_q_l1185_118547


namespace NUMINAMATH_GPT_number_of_arrangements_l1185_118516

theorem number_of_arrangements (n : ℕ) (h : n = 7) :
  ∃ (arrangements : ℕ), arrangements = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_arrangements_l1185_118516


namespace NUMINAMATH_GPT_length_of_c_l1185_118529

theorem length_of_c (A B C : ℝ) (a b c : ℝ) (h1 : (π / 3) - A = B) (h2 : a = 3) (h3 : b = 5) : c = 7 :=
sorry

end NUMINAMATH_GPT_length_of_c_l1185_118529


namespace NUMINAMATH_GPT_geometric_sequence_properties_l1185_118560

noncomputable def geometric_sequence_sum (r a1 : ℝ) : Prop :=
  a1 * (r^3 + r^4) = 27 ∨ a1 * (r^3 + r^4) = -27

theorem geometric_sequence_properties (a1 r : ℝ) (h1 : a1 + a1 * r = 1) (h2 : a1 * r^2 + a1 * r^3 = 9) :
  geometric_sequence_sum r a1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l1185_118560


namespace NUMINAMATH_GPT_smallest_n_satisfying_mod_cond_l1185_118500

theorem smallest_n_satisfying_mod_cond (n : ℕ) : (15 * n - 3) % 11 = 0 ↔ n = 9 := by
  sorry

end NUMINAMATH_GPT_smallest_n_satisfying_mod_cond_l1185_118500


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1185_118566

open Set

noncomputable def M := {x : ℝ | ∃ y:ℝ, y = Real.log (2 - x)}
noncomputable def N := {x : ℝ | x^2 - 3*x - 4 ≤ 0 }
noncomputable def I := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem intersection_of_M_and_N : M ∩ N = I := 
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1185_118566


namespace NUMINAMATH_GPT_boa_constrictor_is_70_inches_l1185_118530

-- Definitions based on given problem conditions
def garden_snake_length : ℕ := 10
def boa_constrictor_length : ℕ := 7 * garden_snake_length

-- Statement to prove
theorem boa_constrictor_is_70_inches : boa_constrictor_length = 70 :=
by
  sorry

end NUMINAMATH_GPT_boa_constrictor_is_70_inches_l1185_118530


namespace NUMINAMATH_GPT_percentage_of_exceedance_l1185_118569

theorem percentage_of_exceedance (x p : ℝ) (h : x = (p / 100) * x + 52.8) (hx : x = 60) : p = 12 :=
by 
  sorry

end NUMINAMATH_GPT_percentage_of_exceedance_l1185_118569


namespace NUMINAMATH_GPT_part1_part2_l1185_118595

open Real

noncomputable def a_value := 2 * sqrt 2

noncomputable def line_cartesian_eqn (x y : ℝ) : Prop :=
  x + y - 4 = 0

noncomputable def point_on_line (ρ θ : ℝ) :=
  ρ * cos (θ - π / 4) = a_value

noncomputable def curve_param_eqns (θ : ℝ) : (ℝ × ℝ) :=
  (sqrt 3 * cos θ, sin θ)

noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y - 4) / sqrt 2

theorem part1 (P : ℝ × ℝ) (ρ θ : ℝ) : 
  P = (4, π / 2) ∧ point_on_line ρ θ → 
  a_value = 2 * sqrt 2 ∧ line_cartesian_eqn 4 (4 * tan (π / 4)) :=
sorry

theorem part2 :
  (∀ θ : ℝ, distance_to_line (sqrt 3 * cos θ) (sin θ) ≤ 3 * sqrt 2) ∧
  (∃ θ : ℝ, distance_to_line (sqrt 3 * cos θ) (sin θ) = 3 * sqrt 2) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1185_118595


namespace NUMINAMATH_GPT_tan_135_eq_neg1_l1185_118563

theorem tan_135_eq_neg1 : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_GPT_tan_135_eq_neg1_l1185_118563


namespace NUMINAMATH_GPT_total_weight_of_balls_l1185_118576

theorem total_weight_of_balls :
  let weight_blue := 6
  let weight_brown := 3.12
  weight_blue + weight_brown = 9.12 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_balls_l1185_118576


namespace NUMINAMATH_GPT_f_plus_g_eq_l1185_118561

variables {R : Type*} [CommRing R]

-- Define the odd function f
def f (x : R) : R := sorry

-- Define the even function g
def g (x : R) : R := sorry

-- Define that f is odd and g is even
axiom f_odd (x : R) : f (-x) = -f x
axiom g_even (x : R) : g (-x) = g x

-- Define the given equation
axiom f_minus_g_eq (x : R) : f x - g x = x ^ 2 + 9 * x + 12

-- Statement of the goal
theorem f_plus_g_eq (x : R) : f x + g x = -x ^ 2 + 9 * x - 12 := by
  sorry

end NUMINAMATH_GPT_f_plus_g_eq_l1185_118561


namespace NUMINAMATH_GPT_max_snowmen_l1185_118568

-- We define the conditions for the masses of the snowballs.
def masses (n : ℕ) := {i | 1 ≤ i ∧ i ≤ n}

-- Define the constraints for a valid snowman.
def valid_snowman (x y z : ℕ) : Prop :=
  x ≥ 2 * y ∧ y ≥ 2 * z

-- Prove the maximum number of snowmen that can be constructed under given conditions.
theorem max_snowmen : ∀ (n : ℕ), masses n = {i | 1 ≤ i ∧ i ≤ 99} →
  3 ∣ 99 →
  (∀ (x y z : ℕ), valid_snowman x y z → 
    x ∈ masses 99 ∧ y ∈ masses 99 ∧ z ∈ masses 99) →
  ∃ (m : ℕ), m = 24 :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_snowmen_l1185_118568


namespace NUMINAMATH_GPT_third_term_binomial_coefficient_l1185_118548

theorem third_term_binomial_coefficient :
  (∃ m : ℕ, m = 4 ∧ ∃ k : ℕ, k = 2 ∧ Nat.choose m k = 6) :=
by
  sorry

end NUMINAMATH_GPT_third_term_binomial_coefficient_l1185_118548


namespace NUMINAMATH_GPT_function_b_is_even_and_monotonically_increasing_l1185_118543

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃a b : ℝ⦄, a ∈ s → b ∈ s → a < b → f a ≤ f b

def f (x : ℝ) : ℝ := abs x + 1

theorem function_b_is_even_and_monotonically_increasing :
  is_even_function f ∧ is_monotonically_increasing_on f (Set.Ioi 0) :=
by
  sorry

end NUMINAMATH_GPT_function_b_is_even_and_monotonically_increasing_l1185_118543


namespace NUMINAMATH_GPT_tan_double_angle_l1185_118527

theorem tan_double_angle (θ : ℝ) (h1 : θ = Real.arctan (-2)) : Real.tan (2 * θ) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l1185_118527


namespace NUMINAMATH_GPT_gabi_final_prices_l1185_118555

theorem gabi_final_prices (x y : ℝ) (hx : 0.8 * x = 1.2 * y) (hl : (x - 0.8 * x) + (y - 1.2 * y) = 10) :
  x = 30 ∧ y = 20 := sorry

end NUMINAMATH_GPT_gabi_final_prices_l1185_118555


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1185_118597

theorem necessary_but_not_sufficient (x : Real)
  (p : Prop := x < 1) 
  (q : Prop := x^2 + x - 2 < 0) 
  : p -> (q <-> x > -2 ∧ x < 1) ∧ (q -> p) → ¬ (p -> q) ∧ (x > -2 -> p) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1185_118597


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1185_118510

theorem isosceles_triangle_perimeter (x y : ℝ) (h : 4 * x ^ 2 + 17 * y ^ 2 - 16 * x * y - 4 * y + 4 = 0):
  x = 4 ∧ y = 2 → 2 * x + y = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1185_118510


namespace NUMINAMATH_GPT_total_pies_bigger_event_l1185_118517

def pies_last_week := 16.5
def apple_pies_last_week := 14.25
def cherry_pies_last_week := 12.75

def pecan_multiplier := 4.3
def apple_multiplier := 3.5
def cherry_multiplier := 5.7

theorem total_pies_bigger_event :
  (pies_last_week * pecan_multiplier) + 
  (apple_pies_last_week * apple_multiplier) + 
  (cherry_pies_last_week * cherry_multiplier) = 193.5 :=
by
  sorry

end NUMINAMATH_GPT_total_pies_bigger_event_l1185_118517


namespace NUMINAMATH_GPT_artist_paint_usage_l1185_118508

def ounces_of_paint_used (extra_large: ℕ) (large: ℕ) (medium: ℕ) (small: ℕ) : ℕ :=
  4 * extra_large + 3 * large + 2 * medium + 1 * small

theorem artist_paint_usage : ounces_of_paint_used 3 5 6 8 = 47 := by
  sorry

end NUMINAMATH_GPT_artist_paint_usage_l1185_118508


namespace NUMINAMATH_GPT_largest_number_is_34_l1185_118519

theorem largest_number_is_34 (a b c : ℕ) (h1 : a + b + c = 82) (h2 : c - b = 8) (h3 : b - a = 4) : c = 34 := 
by 
  sorry

end NUMINAMATH_GPT_largest_number_is_34_l1185_118519


namespace NUMINAMATH_GPT_option_D_correct_l1185_118578

-- Defining the types for lines and planes
variables {Line Plane : Type}

-- Defining what's needed for perpendicularity and parallelism
variables (perp : Line → Plane → Prop)
variables (subset : Line → Plane → Prop)
variables (parallel : Line → Line → Prop)
variables (perp_planes : Plane → Plane → Prop)

-- Main theorem statement
theorem option_D_correct (a b : Line) (α β : Plane) :
  perp a α → subset b β → parallel a b → perp_planes α β :=
by
  sorry

end NUMINAMATH_GPT_option_D_correct_l1185_118578


namespace NUMINAMATH_GPT_quadratic_trinomial_neg_values_l1185_118592

theorem quadratic_trinomial_neg_values (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 7 * x + 4 * a < 0) ↔ a < -7/4 := by
sorry

end NUMINAMATH_GPT_quadratic_trinomial_neg_values_l1185_118592


namespace NUMINAMATH_GPT_map_distance_representation_l1185_118540

theorem map_distance_representation
  (d_map : ℕ) (d_actual : ℕ) (conversion_factor : ℕ) (final_length_map : ℕ):
  d_map = 10 →
  d_actual = 80 →
  conversion_factor = d_actual / d_map →
  final_length_map = 18 →
  (final_length_map * conversion_factor) = 144 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_map_distance_representation_l1185_118540


namespace NUMINAMATH_GPT_sequence_ln_l1185_118506

theorem sequence_ln (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) :
  ∀ n : ℕ, n ≥ 1 → a n = 1 + Real.log n := 
sorry

end NUMINAMATH_GPT_sequence_ln_l1185_118506


namespace NUMINAMATH_GPT_rectangular_prism_inequality_l1185_118565

variable {a b c l : ℝ}

theorem rectangular_prism_inequality (h_diag : l^2 = a^2 + b^2 + c^2) :
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := 
sorry

end NUMINAMATH_GPT_rectangular_prism_inequality_l1185_118565


namespace NUMINAMATH_GPT_common_chord_eq_l1185_118507

theorem common_chord_eq (x y : ℝ) :
  (x^2 + y^2 + 2*x + 8*y - 8 = 0) →
  (x^2 + y^2 - 4*x - 4*y - 2 = 0) →
  (x + 2*y - 1 = 0) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_common_chord_eq_l1185_118507


namespace NUMINAMATH_GPT_decompose_96_l1185_118580

theorem decompose_96 (a b : ℤ) (h1 : a * b = 96) (h2 : a^2 + b^2 = 208) : 
  (a = 8 ∧ b = 12) ∨ (a = 12 ∧ b = 8) ∨ (a = -8 ∧ b = -12) ∨ (a = -12 ∧ b = -8) :=
by
  sorry

end NUMINAMATH_GPT_decompose_96_l1185_118580


namespace NUMINAMATH_GPT_range_of_a_l1185_118583

noncomputable def f (a x : ℝ) := (a - Real.sin x) / Real.cos x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (π / 6 < x) → (x < π / 3) → (f a x) ≤ (f a (x + ε))) → 2 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1185_118583


namespace NUMINAMATH_GPT_vic_max_marks_l1185_118594

theorem vic_max_marks (M : ℝ) (h : 0.92 * M = 368) : M = 400 := 
sorry

end NUMINAMATH_GPT_vic_max_marks_l1185_118594


namespace NUMINAMATH_GPT_incorrect_games_leq_75_percent_l1185_118518

theorem incorrect_games_leq_75_percent (N : ℕ) (win_points : ℕ) (draw_points : ℚ) (loss_points : ℕ) (incorrect : (ℕ × ℕ) → Prop) :
  (win_points = 1) → (draw_points = 1 / 2) → (loss_points = 0) →
  ∀ (g : ℕ × ℕ), incorrect g → 
  ∃ (total_games incorrect_games : ℕ), 
    total_games = N * (N - 1) / 2 ∧
    incorrect_games ≤ 3 / 4 * total_games := sorry

end NUMINAMATH_GPT_incorrect_games_leq_75_percent_l1185_118518


namespace NUMINAMATH_GPT_find_f2_f_neg1_f_is_odd_f_monotonic_on_negatives_l1185_118532

def f : ℝ → ℝ :=
  sorry

noncomputable def f_properties : Prop :=
  (∀ x y : ℝ, x < 0 → f x < 0 → f x + f y = f (x * y) / f (x + y)) ∧ f 1 = 1

theorem find_f2_f_neg1 :
  f_properties →
  f 2 = 1 / 2 ∧ f (-1) = -1 :=
sorry

theorem f_is_odd :
  f_properties →
  ∀ x : ℝ, f x = -f (-x) :=
sorry

theorem f_monotonic_on_negatives :
  f_properties →
  ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 < x2 → f x1 > f x2 :=
sorry

end NUMINAMATH_GPT_find_f2_f_neg1_f_is_odd_f_monotonic_on_negatives_l1185_118532


namespace NUMINAMATH_GPT_travel_speed_l1185_118557

theorem travel_speed (distance : ℕ) (time : ℕ) (h_distance : distance = 160) (h_time : time = 8) :
  ∃ speed : ℕ, speed = distance / time ∧ speed = 20 :=
by
  sorry

end NUMINAMATH_GPT_travel_speed_l1185_118557


namespace NUMINAMATH_GPT_calc_value_l1185_118575

def f (x : ℤ) : ℤ := x^2 + 5 * x + 4
def g (x : ℤ) : ℤ := 2 * x - 3

theorem calc_value :
  f (g (-3)) - 2 * g (f 2) = -26 := by
  sorry

end NUMINAMATH_GPT_calc_value_l1185_118575


namespace NUMINAMATH_GPT_exists_projectile_time_l1185_118562

noncomputable def projectile_time := 
  ∃ t1 t2 : ℝ, (-4.9 * t1^2 + 31 * t1 - 40 = 0) ∧ ((abs (t1 - 1.8051) < 0.001) ∨ (abs (t2 - 4.5319) < 0.001))

theorem exists_projectile_time : projectile_time := 
sorry

end NUMINAMATH_GPT_exists_projectile_time_l1185_118562


namespace NUMINAMATH_GPT_lindas_average_speed_l1185_118503

theorem lindas_average_speed
  (dist1 : ℕ) (time1 : ℝ)
  (dist2 : ℕ) (time2 : ℝ)
  (h1 : dist1 = 450)
  (h2 : time1 = 7.5)
  (h3 : dist2 = 480)
  (h4 : time2 = 8) :
  (dist1 + dist2) / (time1 + time2) = 60 :=
by
  sorry

end NUMINAMATH_GPT_lindas_average_speed_l1185_118503


namespace NUMINAMATH_GPT_find_fraction_l1185_118528

-- Definition of the fractions and the given condition
def certain_fraction : ℚ := 1 / 2
def given_ratio : ℚ := 2 / 6
def target_fraction : ℚ := 1 / 3

-- The proof problem to verify
theorem find_fraction (X : ℚ) : (X / given_ratio) = 1 ↔ X = target_fraction :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l1185_118528


namespace NUMINAMATH_GPT_total_surfers_calculation_l1185_118599

def surfers_on_malibu_beach (m_sm : ℕ) (s_sm : ℕ) : ℕ := 2 * s_sm

def total_surfers (m_sm s_sm : ℕ) : ℕ := m_sm + s_sm

theorem total_surfers_calculation : total_surfers (surfers_on_malibu_beach 20 20) 20 = 60 := by
  sorry

end NUMINAMATH_GPT_total_surfers_calculation_l1185_118599


namespace NUMINAMATH_GPT_correct_propositions_l1185_118556

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * (Real.sin x + Real.cos x)

-- Proposition 2: Symmetry about the line x = -3π/4
def proposition_2 : Prop := ∀ x, f (x + 3 * Real.pi / 4) = f (-x)

-- Proposition 3: There exists φ ∈ ℝ, such that the graph of the function f(x + φ) is centrally symmetric about the origin
def proposition_3 : Prop := ∃ φ : ℝ, ∀ x, f (x + φ) = -f (-x)

theorem correct_propositions :
  (proposition_2 ∧ proposition_3) := by
  sorry

end NUMINAMATH_GPT_correct_propositions_l1185_118556


namespace NUMINAMATH_GPT_gcd_cubed_and_sum_l1185_118587

theorem gcd_cubed_and_sum (n : ℕ) (h_pos : 0 < n) (h_gt_square : n > 9) : 
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := 
sorry

end NUMINAMATH_GPT_gcd_cubed_and_sum_l1185_118587


namespace NUMINAMATH_GPT_cat_run_time_l1185_118514

/-- An electronic cat runs a lap on a circular track with a perimeter of 240 meters.
It runs at a speed of 5 meters per second for the first half of the time and 3 meters per second for the second half of the time.
Prove that the cat takes 36 seconds to run the last 120 meters. -/
theorem cat_run_time
  (perimeter : ℕ)
  (speed1 speed2 : ℕ)
  (half_perimeter : ℕ)
  (half_time : ℕ)
  (last_120m_time : ℕ) :
  perimeter = 240 →
  speed1 = 5 →
  speed2 = 3 →
  half_perimeter = perimeter / 2 →
  half_time = 60 / 2 →
  (5 * half_time - half_perimeter) / speed1 + (half_perimeter - (5 * half_time - half_perimeter)) / speed2 = 36 :=
by sorry

end NUMINAMATH_GPT_cat_run_time_l1185_118514


namespace NUMINAMATH_GPT_power_of_powers_eval_powers_l1185_118524

theorem power_of_powers (a m n : ℕ) : (a^m)^n = a^(m * n) :=
  by sorry

theorem eval_powers : (3^2)^4 = 6561 :=
  by
    rw [power_of_powers]
    -- further proof of (3^8 = 6561) would go here
    sorry

end NUMINAMATH_GPT_power_of_powers_eval_powers_l1185_118524


namespace NUMINAMATH_GPT_compute_complex_expression_l1185_118539

-- Define the expression we want to prove
def complex_expression : ℚ := 1 / (1 + (1 / (2 + (1 / (4^2)))))

-- The theorem stating the expression equals to the correct result
theorem compute_complex_expression : complex_expression = 33 / 49 :=
by sorry

end NUMINAMATH_GPT_compute_complex_expression_l1185_118539


namespace NUMINAMATH_GPT_value_of_y_l1185_118598

theorem value_of_y (x y : ℝ) (h1 : x^(2 * y) = 8) (h2 : x = 2) : y = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l1185_118598


namespace NUMINAMATH_GPT_ratio_of_sleep_l1185_118581

theorem ratio_of_sleep (connor_sleep : ℝ) (luke_extra : ℝ) (puppy_sleep : ℝ) 
    (h1 : connor_sleep = 6)
    (h2 : luke_extra = 2)
    (h3 : puppy_sleep = 16) :
    puppy_sleep / (connor_sleep + luke_extra) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_sleep_l1185_118581


namespace NUMINAMATH_GPT_isaiah_types_more_words_than_micah_l1185_118589

theorem isaiah_types_more_words_than_micah :
  let micah_speed := 20   -- Micah's typing speed in words per minute
  let isaiah_speed := 40  -- Isaiah's typing speed in words per minute
  let minutes_in_hour := 60  -- Number of minutes in an hour
  (isaiah_speed * minutes_in_hour) - (micah_speed * minutes_in_hour) = 1200 :=
by
  sorry

end NUMINAMATH_GPT_isaiah_types_more_words_than_micah_l1185_118589


namespace NUMINAMATH_GPT_pentagon_area_l1185_118567

noncomputable def square_area (side_length : ℤ) : ℤ :=
  side_length * side_length

theorem pentagon_area (CF : ℤ) (a b : ℤ) (CE : ℤ) (ED : ℤ) (EF : ℤ) :
  (CF = 5) →
  (a = CE + ED) →
  (b = EF) →
  (CE < ED) →
  CF * CF = CE * CE + EF * EF →
  square_area a + square_area b - (CE * EF / 2) = 71 :=
by
  intros hCF ha hb hCE_lt_ED hPythagorean
  sorry

end NUMINAMATH_GPT_pentagon_area_l1185_118567


namespace NUMINAMATH_GPT_convert_2e_15pi_i4_to_rectangular_form_l1185_118590

noncomputable def convert_to_rectangular_form (z : ℂ) : ℂ :=
  let θ := (15 * Real.pi) / 4
  let θ' := θ - 2 * Real.pi
  2 * Complex.exp (θ' * Complex.I)

theorem convert_2e_15pi_i4_to_rectangular_form :
  convert_to_rectangular_form (2 * Complex.exp ((15 * Real.pi) / 4 * Complex.I)) = (Real.sqrt 2 - Complex.I * Real.sqrt 2) :=
  sorry

end NUMINAMATH_GPT_convert_2e_15pi_i4_to_rectangular_form_l1185_118590


namespace NUMINAMATH_GPT_pascal_triangle_fifth_number_l1185_118509

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_pascal_triangle_fifth_number_l1185_118509


namespace NUMINAMATH_GPT_tangent_line_intercept_l1185_118538

theorem tangent_line_intercept :
  ∃ (m : ℚ) (b : ℚ), m > 0 ∧ b = 740 / 171 ∧
    ∀ (x y : ℚ), ((x - 1)^2 + (y - 3)^2 = 9 ∨ (x - 15)^2 + (y - 8)^2 = 100) →
                 (y = m * x + b) ↔ False := 
sorry

end NUMINAMATH_GPT_tangent_line_intercept_l1185_118538


namespace NUMINAMATH_GPT_find_s_when_t_eq_5_l1185_118571

theorem find_s_when_t_eq_5 (s : ℝ) (h : 5 = 8 * s^2 + 2 * s) :
  s = (-1 + Real.sqrt 41) / 8 ∨ s = (-1 - Real.sqrt 41) / 8 :=
by sorry

end NUMINAMATH_GPT_find_s_when_t_eq_5_l1185_118571


namespace NUMINAMATH_GPT_time_per_lawn_in_minutes_l1185_118502

def jason_lawns := 16
def total_hours_cutting := 8
def minutes_per_hour := 60

theorem time_per_lawn_in_minutes : 
  (total_hours_cutting / jason_lawns) * minutes_per_hour = 30 :=
by
  sorry

end NUMINAMATH_GPT_time_per_lawn_in_minutes_l1185_118502


namespace NUMINAMATH_GPT_function_equality_l1185_118523

theorem function_equality (f : ℝ → ℝ) (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  (f ( (x + 1) / x ) = (x^2 + 1) / x^2 + 1 / x) ↔ (f x = x^2 - x + 1) :=
by
  sorry

end NUMINAMATH_GPT_function_equality_l1185_118523


namespace NUMINAMATH_GPT_sodium_hydride_reaction_l1185_118521

theorem sodium_hydride_reaction (H2O NaH NaOH H2 : ℕ) 
  (balanced_eq : NaH + H2O = NaOH + H2) 
  (stoichiometry : NaH = H2O → NaOH = H2 → NaH = H2) 
  (h : H2O = 2) : NaH = 2 :=
sorry

end NUMINAMATH_GPT_sodium_hydride_reaction_l1185_118521


namespace NUMINAMATH_GPT_smallest_c_for_inverse_l1185_118545

def g (x : ℝ) : ℝ := -3 * (x - 1)^2 + 4

theorem smallest_c_for_inverse :
  ∃ c : ℝ, (∀ x y : ℝ, c ≤ x → c ≤ y → g x = g y → x = y) ∧ (∀ d : ℝ, (∀ x y : ℝ, d ≤ x → d ≤ y → g x = g y → x = y) → c ≤ d) :=
sorry

end NUMINAMATH_GPT_smallest_c_for_inverse_l1185_118545


namespace NUMINAMATH_GPT_karthik_weight_average_l1185_118593

noncomputable def average_probable_weight_of_karthik (weight : ℝ) : Prop :=
  (55 < weight ∧ weight < 62) ∧
  (50 < weight ∧ weight < 60) ∧
  (weight ≤ 58) →
  weight = 56.5

theorem karthik_weight_average :
  ∀ weight : ℝ, average_probable_weight_of_karthik weight :=
by
  sorry

end NUMINAMATH_GPT_karthik_weight_average_l1185_118593


namespace NUMINAMATH_GPT_balance_proof_l1185_118564

variables (a b c : ℝ)

theorem balance_proof (h1 : 4 * a + 2 * b = 12 * c) (h2 : 2 * a = b + 3 * c) : 3 * b = 4.5 * c :=
sorry

end NUMINAMATH_GPT_balance_proof_l1185_118564


namespace NUMINAMATH_GPT_point_transformation_correct_l1185_118554

-- Define the rectangular coordinate system O-xyz
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the point in the original coordinate system
def originalPoint : Point3D := { x := 1, y := -2, z := 3 }

-- Define the transformation function for the yOz plane
def transformToYOzPlane (p : Point3D) : Point3D :=
  { x := -p.x, y := p.y, z := p.z }

-- Define the expected transformed point
def transformedPoint : Point3D := { x := -1, y := -2, z := 3 }

-- State the theorem to be proved
theorem point_transformation_correct :
  transformToYOzPlane originalPoint = transformedPoint :=
by
  sorry

end NUMINAMATH_GPT_point_transformation_correct_l1185_118554


namespace NUMINAMATH_GPT_value_of_d_l1185_118526

theorem value_of_d (d : ℝ) (h : ∀ x : ℝ, 3 * (5 + d * x) = 15 * x + 15 → True) : d = 5 :=
sorry

end NUMINAMATH_GPT_value_of_d_l1185_118526


namespace NUMINAMATH_GPT_range_of_a_l1185_118550

-- Defining the core problem conditions in Lean
def prop_p (a : ℝ) : Prop := ∃ x₀ : ℝ, a * x₀^2 + 2 * a * x₀ + 1 < 0

-- The original proposition p is false, thus we need to show the range of a is 0 ≤ a ≤ 1
theorem range_of_a (a : ℝ) : ¬ prop_p a → 0 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1185_118550


namespace NUMINAMATH_GPT_combined_salaries_l1185_118585

-- Define the variables and constants corresponding to the conditions
variable (A B D E C : ℝ)
variable (avg_salary : ℝ)
variable (num_individuals : ℕ)

-- Given conditions translated into Lean definitions 
def salary_C : ℝ := 15000
def average_salary : ℝ := 8800
def number_of_individuals : ℕ := 5

-- Define the statement to prove
theorem combined_salaries (h1 : C = salary_C) (h2 : avg_salary = average_salary) (h3 : num_individuals = number_of_individuals) : 
  A + B + D + E = avg_salary * num_individuals - salary_C := 
by 
  -- Here the proof would involve calculating the total salary and subtracting C's salary
  sorry

end NUMINAMATH_GPT_combined_salaries_l1185_118585


namespace NUMINAMATH_GPT_part1_domain_of_f_part2_inequality_l1185_118522

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (abs (x + 1) + abs (x - 1) - 4)

theorem part1_domain_of_f : {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -2 ∨ x ≥ 2} :=
by 
  sorry

theorem part2_inequality (a b : ℝ) (h_a : -2 < a) (h_a' : a < 2) (h_b : -2 < b) (h_b' : b < 2) 
  : 2 * abs (a + b) < abs (4 + a * b) :=
by 
  sorry

end NUMINAMATH_GPT_part1_domain_of_f_part2_inequality_l1185_118522


namespace NUMINAMATH_GPT_percent_students_in_range_l1185_118542

theorem percent_students_in_range
    (n1 n2 n3 n4 n5 : ℕ)
    (h1 : n1 = 5)
    (h2 : n2 = 7)
    (h3 : n3 = 8)
    (h4 : n4 = 4)
    (h5 : n5 = 3) :
  ((n3 : ℝ) / (n1 + n2 + n3 + n4 + n5) * 100) = 29.63 :=
by
  sorry

end NUMINAMATH_GPT_percent_students_in_range_l1185_118542


namespace NUMINAMATH_GPT_two_digit_numbers_less_than_35_l1185_118505

theorem two_digit_numbers_less_than_35 : 
  let count : ℕ := (99 - 10 + 1) - (7 * 5)
  count = 55 :=
by
  -- definition of total number of two-digit numbers
  let total_two_digit_numbers : ℕ := 99 - 10 + 1
  -- definition of the number of unsuitable two-digit numbers
  let unsuitable_numbers : ℕ := 7 * 5
  -- definition of the count of suitable two-digit numbers
  let count : ℕ := total_two_digit_numbers - unsuitable_numbers
  -- verifying the final count
  exact rfl

end NUMINAMATH_GPT_two_digit_numbers_less_than_35_l1185_118505


namespace NUMINAMATH_GPT_journey_distance_l1185_118551

theorem journey_distance (t : ℝ) : 
  t = 20 →
  ∃ D : ℝ, (D / 20 + D / 30 = t) ∧ D = 240 :=
by
  sorry

end NUMINAMATH_GPT_journey_distance_l1185_118551


namespace NUMINAMATH_GPT_sequence_sum_correct_l1185_118544

theorem sequence_sum_correct :
  ∀ (r x y : ℝ),
  (x = 128 * r) →
  (y = x * r) →
  (2 * r = 1 / 2) →
  (x + y = 40) :=
by
  intros r x y hx hy hr
  sorry

end NUMINAMATH_GPT_sequence_sum_correct_l1185_118544


namespace NUMINAMATH_GPT_toys_in_stock_l1185_118574

theorem toys_in_stock (sold_first_week sold_second_week toys_left toys_initial: ℕ) :
  sold_first_week = 38 → 
  sold_second_week = 26 → 
  toys_left = 19 → 
  toys_initial = sold_first_week + sold_second_week + toys_left → 
  toys_initial = 83 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_toys_in_stock_l1185_118574


namespace NUMINAMATH_GPT_term_61_is_201_l1185_118588

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (a5 : ℤ)

-- Define the general formula for the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ :=
  a5 + (n - 5) * d

-- Given variables and conditions:
axiom h1 : a5 = 33
axiom h2 : d = 3

theorem term_61_is_201 :
  arithmetic_sequence a5 d 61 = 201 :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_term_61_is_201_l1185_118588


namespace NUMINAMATH_GPT_impossible_measure_1_liter_with_buckets_l1185_118546

theorem impossible_measure_1_liter_with_buckets :
  ¬ (∃ k l : ℤ, k * Real.sqrt 2 + l * (2 - Real.sqrt 2) = 1) :=
by
  sorry

end NUMINAMATH_GPT_impossible_measure_1_liter_with_buckets_l1185_118546


namespace NUMINAMATH_GPT_evaluate_expression_l1185_118553

theorem evaluate_expression : 27^(- (2 / 3 : ℝ)) + Real.log 4 / Real.log 8 = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1185_118553


namespace NUMINAMATH_GPT_probability_of_red_ball_and_removed_red_balls_l1185_118534

-- Conditions for the problem
def initial_red_balls : Nat := 10
def initial_yellow_balls : Nat := 2
def initial_blue_balls : Nat := 8
def total_balls : Nat := initial_red_balls + initial_yellow_balls + initial_blue_balls

-- Problem statement in Lean
theorem probability_of_red_ball_and_removed_red_balls :
  (initial_red_balls / total_balls = 1 / 2) ∧
  (∃ (x : Nat), -- Number of red balls removed
    ((initial_yellow_balls + x) / total_balls = 2 / 5) ∧
    (initial_red_balls - x = 10 - 6)) := 
by
  -- Lean will need the proofs here; we use sorry for now.
  sorry

end NUMINAMATH_GPT_probability_of_red_ball_and_removed_red_balls_l1185_118534


namespace NUMINAMATH_GPT_pages_used_l1185_118573

variable (n o c : ℕ)

theorem pages_used (h_n : n = 3) (h_o : o = 13) (h_c : c = 8) :
  (n + o) / c = 2 :=
  by
    sorry

end NUMINAMATH_GPT_pages_used_l1185_118573


namespace NUMINAMATH_GPT_charming_number_unique_l1185_118533

def is_charming (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≥ 1 ∧ a ≤ 9 ∧ b ≤ 9 ∧ n = 10 * a + b ∧ n = 2 * a + b^3

theorem charming_number_unique : ∃! n, 10 ≤ n ∧ n ≤ 99 ∧ is_charming n := by
  sorry

end NUMINAMATH_GPT_charming_number_unique_l1185_118533


namespace NUMINAMATH_GPT_interest_rate_per_annum_l1185_118531

-- Definitions for the given conditions
def SI : ℝ := 4016.25
def P : ℝ := 44625
def T : ℝ := 9

-- The interest rate R must be 1 according to the conditions
theorem interest_rate_per_annum : (SI * 100) / (P * T) = 1 := by
  sorry

end NUMINAMATH_GPT_interest_rate_per_annum_l1185_118531


namespace NUMINAMATH_GPT_distinct_ways_to_divide_books_l1185_118572

theorem distinct_ways_to_divide_books : 
  ∃ (ways : ℕ), ways = 5 := sorry

end NUMINAMATH_GPT_distinct_ways_to_divide_books_l1185_118572


namespace NUMINAMATH_GPT_find_sum_lent_l1185_118552

variable (P : ℝ)

/-- Given that the annual interest rate is 4%, and the interest earned in 8 years
amounts to Rs 340 less than the sum lent, prove that the sum lent is Rs 500. -/
theorem find_sum_lent
  (h1 : ∀ I, I = P - 340 → I = (P * 4 * 8) / 100) : 
  P = 500 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_lent_l1185_118552


namespace NUMINAMATH_GPT_cycle_selling_price_l1185_118501

theorem cycle_selling_price 
  (CP : ℝ) (gain_percent : ℝ) (SP : ℝ) 
  (h1 : CP = 840) 
  (h2 : gain_percent = 45.23809523809524 / 100)
  (h3 : SP = CP * (1 + gain_percent)) :
  SP = 1220 :=
sorry

end NUMINAMATH_GPT_cycle_selling_price_l1185_118501


namespace NUMINAMATH_GPT_ball_bounces_height_l1185_118536

theorem ball_bounces_height (initial_height : ℝ) (decay_factor : ℝ) (threshold : ℝ) (n : ℕ) :
  initial_height = 20 →
  decay_factor = 3/4 →
  threshold = 2 →
  n = 9 →
  initial_height * (decay_factor ^ n) < threshold :=
by
  intros
  sorry

end NUMINAMATH_GPT_ball_bounces_height_l1185_118536


namespace NUMINAMATH_GPT_lyle_payment_l1185_118559

def pen_cost : ℝ := 1.50

def notebook_cost : ℝ := 3 * pen_cost

def cost_for_4_notebooks : ℝ := 4 * notebook_cost

theorem lyle_payment : cost_for_4_notebooks = 18.00 :=
by
  sorry

end NUMINAMATH_GPT_lyle_payment_l1185_118559


namespace NUMINAMATH_GPT_find_c_l1185_118584

theorem find_c :
  ∃ c : ℝ, 0 < c ∧ ∀ line : ℝ, (∃ x y : ℝ, (x = 1 ∧ y = c) ∧ (x*x + y*y - 2*x - 2*y - 7 = 0)) ∧ (line = 1*x + 0 + y*c - 0) :=
sorry

end NUMINAMATH_GPT_find_c_l1185_118584


namespace NUMINAMATH_GPT_evaluate_g_g_g_25_l1185_118579

def g (x : ℤ) : ℤ :=
  if x < 10 then x^2 - 9 else x - 20

theorem evaluate_g_g_g_25 : g (g (g 25)) = -4 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_g_g_25_l1185_118579


namespace NUMINAMATH_GPT_energy_fraction_l1185_118520

-- Conditions
variables (E : ℝ → ℝ)
variable (x : ℝ)
variable (h : ∀ x, E (x + 1) = 31.6 * E x)

-- The statement to be proven
theorem energy_fraction (x : ℝ) (h : ∀ x, E (x + 1) = 31.6 * E x) : 
  E (x - 1) / E x = 1 / 31.6 :=
by
  sorry

end NUMINAMATH_GPT_energy_fraction_l1185_118520


namespace NUMINAMATH_GPT_diagonals_in_polygon_l1185_118525

-- Define the number of sides of the polygon
def n : ℕ := 30

-- Define the formula for the total number of diagonals in an n-sided polygon
def total_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Define the number of excluded diagonals for being parallel to one given side
def excluded_diagonals : ℕ := 1

-- Define the final count of valid diagonals after exclusion
def valid_diagonals : ℕ := total_diagonals n - excluded_diagonals

-- State the theorem to prove
theorem diagonals_in_polygon : valid_diagonals = 404 := by
  sorry


end NUMINAMATH_GPT_diagonals_in_polygon_l1185_118525
