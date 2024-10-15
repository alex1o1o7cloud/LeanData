import Mathlib

namespace NUMINAMATH_GPT_sequence_general_term_l977_97726

theorem sequence_general_term :
  ∀ n : ℕ, n > 0 → (∀ a: ℕ → ℝ,  a 1 = 4 ∧ (∀ n: ℕ, n > 0 → a (n + 1) = (3 * a n + 2) / (a n + 4))
  → a n = (2 ^ (n - 1) + 5 ^ (n - 1)) / (5 ^ (n - 1) - 2 ^ (n - 1))) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l977_97726


namespace NUMINAMATH_GPT_moles_of_CO2_formed_l977_97704

-- Definitions based on the conditions provided
def moles_HNO3 := 2
def moles_NaHCO3 := 2
def balanced_eq (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) : Prop :=
  HNO3 = NaHCO3 ∧ NaNO3 = NaHCO3 ∧ CO2 = NaHCO3 ∧ H2O = NaHCO3

-- Lean Proposition: Prove that 2 moles of CO2 are formed
theorem moles_of_CO2_formed :
  balanced_eq moles_HNO3 moles_NaHCO3 moles_HNO3 moles_HNO3 moles_HNO3 →
  ∃ CO2, CO2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_CO2_formed_l977_97704


namespace NUMINAMATH_GPT_mass_percentage_C_in_C6HxO6_indeterminate_l977_97714

-- Definition of conditions
def mass_percentage_C_in_C6H8O6 : ℚ := 40.91 / 100
def molar_mass_C : ℚ := 12.01
def molar_mass_H : ℚ := 1.01
def molar_mass_O : ℚ := 16.00

-- Formula for molar mass of C6H8O6
def molar_mass_C6H8O6 : ℚ := 6 * molar_mass_C + 8 * molar_mass_H + 6 * molar_mass_O

-- Mass of carbon in C6H8O6 is 40.91% of the total molar mass
def mass_of_C_in_C6H8O6 : ℚ := mass_percentage_C_in_C6H8O6 * molar_mass_C6H8O6

-- Hypothesis: mass percentage of carbon in C6H8O6 is given
axiom hyp_mass_percentage_C_in_C6H8O6 : mass_of_C_in_C6H8O6 = 72.06

-- Proof that we need the value of x to determine the mass percentage of C in C6HxO6
theorem mass_percentage_C_in_C6HxO6_indeterminate (x : ℚ) :
  (molar_mass_C6H8O6 = 176.14) → (mass_of_C_in_C6H8O6 = 72.06) → False :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_C_in_C6HxO6_indeterminate_l977_97714


namespace NUMINAMATH_GPT_greatest_integer_y_l977_97765

-- Define the fraction and inequality condition
def inequality_condition (y : ℤ) : Prop := 8 * 17 > 11 * y

-- Prove the greatest integer y satisfying the condition is 12
theorem greatest_integer_y : ∃ y : ℤ, inequality_condition y ∧ (∀ z : ℤ, inequality_condition z → z ≤ y) ∧ y = 12 :=
by
  exists 12
  sorry

end NUMINAMATH_GPT_greatest_integer_y_l977_97765


namespace NUMINAMATH_GPT_carla_games_won_l977_97792

theorem carla_games_won (F C : ℕ) (h1 : F + C = 30) (h2 : F = C / 2) : C = 20 :=
by
  sorry

end NUMINAMATH_GPT_carla_games_won_l977_97792


namespace NUMINAMATH_GPT_speed_of_boat_in_still_water_l977_97752

variable (b s : ℝ) -- Speed of the boat in still water and speed of the stream

-- Condition 1: The boat goes 9 km along the stream in 1 hour
def boat_along_stream := b + s = 9

-- Condition 2: The boat goes 5 km against the stream in 1 hour
def boat_against_stream := b - s = 5

-- Theorem to prove: The speed of the boat in still water is 7 km/hr
theorem speed_of_boat_in_still_water : boat_along_stream b s → boat_against_stream b s → b = 7 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_boat_in_still_water_l977_97752


namespace NUMINAMATH_GPT_ratio_AR_AU_l977_97756

-- Define the conditions in the problem as variables and constraints
variables (A B C P Q U R : Type)
variables (AP PB AQ QC : ℝ)
variables (angle_bisector_AU : A -> U)
variables (intersect_AU_PQ_at_R : A -> U -> P -> Q -> R)

-- Assuming the given distances
def conditions (AP PB AQ QC : ℝ) : Prop :=
  AP = 2 ∧ PB = 6 ∧ AQ = 4 ∧ QC = 5

-- The statement to prove
theorem ratio_AR_AU (h : conditions AP PB AQ QC) : 
  (AR / AU) = 108 / 289 :=
sorry

end NUMINAMATH_GPT_ratio_AR_AU_l977_97756


namespace NUMINAMATH_GPT_solve_for_x_l977_97729

noncomputable def proof (x : ℚ) : Prop :=
  (x + 6) / (x - 4) = (x - 7) / (x + 2)

theorem solve_for_x (x : ℚ) (h : proof x) : x = 16 / 19 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l977_97729


namespace NUMINAMATH_GPT_positive_difference_enrollment_l977_97774

theorem positive_difference_enrollment 
  (highest_enrollment : ℕ)
  (lowest_enrollment : ℕ)
  (h_highest : highest_enrollment = 2150)
  (h_lowest : lowest_enrollment = 980) :
  highest_enrollment - lowest_enrollment = 1170 :=
by {
  -- Proof to be added here
  sorry
}

end NUMINAMATH_GPT_positive_difference_enrollment_l977_97774


namespace NUMINAMATH_GPT_probability_girls_same_color_l977_97708

open Classical

noncomputable def probability_same_color_marbles : ℚ :=
(3/6) * (2/5) * (1/4) + (3/6) * (2/5) * (1/4)

theorem probability_girls_same_color :
  probability_same_color_marbles = 1/20 := by
  sorry

end NUMINAMATH_GPT_probability_girls_same_color_l977_97708


namespace NUMINAMATH_GPT_bar_weight_calc_l977_97777

variable (blue_weight green_weight num_blue_weights num_green_weights bar_weight total_weight : ℕ)

theorem bar_weight_calc
  (h1 : blue_weight = 2)
  (h2 : green_weight = 3)
  (h3 : num_blue_weights = 4)
  (h4 : num_green_weights = 5)
  (h5 : total_weight = 25)
  (weights_total := num_blue_weights * blue_weight + num_green_weights * green_weight)
  : bar_weight = total_weight - weights_total :=
by
  sorry

end NUMINAMATH_GPT_bar_weight_calc_l977_97777


namespace NUMINAMATH_GPT_realNumbersGreaterThan8IsSet_l977_97797

-- Definitions based on conditions:
def verySmallNumbers : Type := {x : ℝ // sorry} -- Need to define what very small numbers would be
def interestingBooks : Type := sorry -- Need to define what interesting books would be
def realNumbersGreaterThan8 : Set ℝ := { x : ℝ | x > 8 }
def tallPeople : Type := sorry -- Need to define what tall people would be

-- Main theorem: Real numbers greater than 8 can form a set
theorem realNumbersGreaterThan8IsSet : Set ℝ :=
  realNumbersGreaterThan8

end NUMINAMATH_GPT_realNumbersGreaterThan8IsSet_l977_97797


namespace NUMINAMATH_GPT_students_in_class_l977_97719

theorem students_in_class (b g : ℕ) 
  (h1 : b + g = 20)
  (h2 : (b : ℚ) / 20 = (3 : ℚ) / 4 * (g : ℚ) / 20) : 
  b = 12 ∧ g = 8 :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l977_97719


namespace NUMINAMATH_GPT_division_problem_l977_97769

theorem division_problem (A : ℕ) (h : 23 = (A * 3) + 2) : A = 7 :=
sorry

end NUMINAMATH_GPT_division_problem_l977_97769


namespace NUMINAMATH_GPT_linear_function_behavior_l977_97754

theorem linear_function_behavior (x y : ℝ) (h : y = -3 * x + 6) :
  ∀ x1 x2 : ℝ, x1 < x2 → (y = -3 * x1 + 6) → (y = -3 * x2 + 6) → -3 * (x1 - x2) > 0 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_behavior_l977_97754


namespace NUMINAMATH_GPT_baseEight_conversion_l977_97753

-- Base-eight number is given as 1563
def baseEight : Nat := 1563

-- Function to convert a base-eight number to base-ten
noncomputable def baseEightToBaseTen (n : Nat) : Nat :=
  let digit3 := (n / 1000) % 10
  let digit2 := (n / 100) % 10
  let digit1 := (n / 10) % 10
  let digit0 := n % 10
  digit3 * 8^3 + digit2 * 8^2 + digit1 * 8^1 + digit0 * 8^0

theorem baseEight_conversion :
  baseEightToBaseTen baseEight = 883 := by
  sorry

end NUMINAMATH_GPT_baseEight_conversion_l977_97753


namespace NUMINAMATH_GPT_inequality_solution_set_l977_97771

theorem inequality_solution_set :
  { x : ℝ | (3 * x + 1) / (x - 2) ≤ 0 } = { x : ℝ | -1/3 ≤ x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l977_97771


namespace NUMINAMATH_GPT_min_value_of_a2_b2_l977_97791

theorem min_value_of_a2_b2 (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 4) : 
  ∃ m : ℝ, (∀ x y, x > 0 → y > 0 → x + y = 4 → x^2 + y^2 ≥ m) ∧ m = 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a2_b2_l977_97791


namespace NUMINAMATH_GPT_find_positive_real_solutions_l977_97799

theorem find_positive_real_solutions (x : ℝ) (h1 : 0 < x) 
(h2 : 3 / 5 * (2 * x ^ 2 - 2) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4)) :
    x = (40 + Real.sqrt 1636) / 2 ∨ x = (-20 + Real.sqrt 388) / 2 := by
  sorry

end NUMINAMATH_GPT_find_positive_real_solutions_l977_97799


namespace NUMINAMATH_GPT_tangent_line_eq_extreme_values_range_of_a_l977_97764

noncomputable def f (x : ℝ) (a: ℝ) : ℝ := x^2 - a * Real.log x

-- (I) Proving the tangent line equation is y = x for a = 1 at x = 1.
theorem tangent_line_eq (h : ∀ x, f x 1 = x^2 - Real.log x) :
  ∃ y : (ℝ → ℝ), y = id ∧ y 1 = x :=
sorry

-- (II) Proving extreme values of the function f(x).
theorem extreme_values (a: ℝ) :
  (∃ x_min : ℝ, f x_min a = (a/2) - (a/2) * Real.log (a/2)) ∧ 
  (∀ x, ¬∃ x_max : ℝ, f x_max a > f x a) :=
sorry

-- (III) Proving the range of values for a.
theorem range_of_a :
  (∀ x, 2*x - (a/x) ≥ 0 → 2 < x) → a ≤ 8 :=
sorry

end NUMINAMATH_GPT_tangent_line_eq_extreme_values_range_of_a_l977_97764


namespace NUMINAMATH_GPT_abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0_l977_97742

theorem abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0 :
  (∀ x : ℝ, |x| < 2 → x^2 - x - 6 < 0) ∧ (¬ ∀ x : ℝ, x^2 - x - 6 < 0 → |x| < 2) :=
by
  sorry

end NUMINAMATH_GPT_abs_x_lt_2_sufficient_not_necessary_for_x_sq_minus_x_minus_6_lt_0_l977_97742


namespace NUMINAMATH_GPT_simplify_exponentiation_l977_97718

-- Define the exponents and the base
variables (t : ℕ)

-- Define the expression and expected result
def expr := t^5 * t^2
def expected := t^7

-- State the proof goal
theorem simplify_exponentiation : expr = expected := 
by sorry

end NUMINAMATH_GPT_simplify_exponentiation_l977_97718


namespace NUMINAMATH_GPT_part_a_l977_97747

theorem part_a (x : ℝ) : 1 + (1 / (2 + 1 / ((4 * x + 1) / (2 * x + 1) - 1 / (2 + 1 / x)))) = 19 / 14 ↔ x = 1 / 2 := sorry

end NUMINAMATH_GPT_part_a_l977_97747


namespace NUMINAMATH_GPT_fish_speed_in_still_water_l977_97750

theorem fish_speed_in_still_water (u d : ℕ) (v : ℕ) : 
  u = 35 → d = 55 → 2 * v = u + d → v = 45 := 
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end NUMINAMATH_GPT_fish_speed_in_still_water_l977_97750


namespace NUMINAMATH_GPT_roger_steps_time_l977_97736

theorem roger_steps_time (steps_per_30_min : ℕ := 2000) (time_for_2000_steps : ℕ := 30) (goal_steps : ℕ := 10000) : 
  (goal_steps * time_for_2000_steps) / steps_per_30_min = 150 :=
by 
  -- This is the statement. Proof is omitted as per instruction.
  sorry

end NUMINAMATH_GPT_roger_steps_time_l977_97736


namespace NUMINAMATH_GPT_negation_of_existence_proposition_l977_97720

theorem negation_of_existence_proposition :
  ¬ (∃ x : ℝ, x^2 + 2*x - 8 = 0) ↔ ∀ x : ℝ, x^2 + 2*x - 8 ≠ 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_existence_proposition_l977_97720


namespace NUMINAMATH_GPT_negation_proposition_l977_97794

theorem negation_proposition :
  (¬ (x ≠ 3 ∧ x ≠ 2) → ¬ (x ^ 2 - 5 * x + 6 ≠ 0)) =
  ((x = 3 ∨ x = 2) → (x ^ 2 - 5 * x + 6 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l977_97794


namespace NUMINAMATH_GPT_find_number_l977_97785

theorem find_number (x : ℤ) (h1 : x - 2 + 4 = 9) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l977_97785


namespace NUMINAMATH_GPT_a_n_geometric_sequence_b_n_general_term_l977_97723

theorem a_n_geometric_sequence (t : ℝ) (h : t ≠ 0 ∧ t ≠ 1) :
  (∀ n, ∃ r : ℝ, a_n = t^n) :=
sorry

theorem b_n_general_term (t : ℝ) (h1 : t ≠ 0 ∧ t ≠ 1) (h2 : ∀ n, a_n = t^n)
  (h3 : ∃ q : ℝ, q = (2 * t^2 + t) / 2) :
  (∀ n, b_n = (t^(n + 1) * (2 * t + 1)^(n - 1)) / 2^(n - 2)) :=
sorry

end NUMINAMATH_GPT_a_n_geometric_sequence_b_n_general_term_l977_97723


namespace NUMINAMATH_GPT_simplification_and_evaluation_l977_97768

theorem simplification_and_evaluation (a : ℚ) (h : a = -1 / 2) :
  (3 * a + 2) * (a - 1) - 4 * a * (a + 1) = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_simplification_and_evaluation_l977_97768


namespace NUMINAMATH_GPT_quadratic_eq_distinct_solutions_l977_97767

theorem quadratic_eq_distinct_solutions (b : ℤ) (k : ℤ) (h1 : 1 ≤ b ∧ b ≤ 100) :
  ∃ n : ℕ, n = 27 ∧ (x^2 + (2 * b + 3) * x + b^2 = 0 →
    12 * b + 9 = k^2 → 
    (∃ m n : ℤ, x = m ∧ x = n ∧ m ≠ n)) :=
sorry

end NUMINAMATH_GPT_quadratic_eq_distinct_solutions_l977_97767


namespace NUMINAMATH_GPT_find_inverse_value_l977_97733

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_function (x : ℝ) : f x = f (-x)
axiom periodic (x : ℝ) : f (x - 1) = f (x + 3)
axiom defined_interval (x : ℝ) (h : 4 ≤ x ∧ x ≤ 5) : f x = 2 ^ x + 1

noncomputable def f_inv : ℝ → ℝ := sorry
axiom inverse_defined (x : ℝ) (h : -2 ≤ x ∧ x ≤ 0) : f (f_inv x) = x

theorem find_inverse_value : f_inv 19 = 3 - 2 * (Real.log 3 / Real.log 2) := by
  sorry

end NUMINAMATH_GPT_find_inverse_value_l977_97733


namespace NUMINAMATH_GPT_remainder_196c_2008_mod_97_l977_97734

theorem remainder_196c_2008_mod_97 (c : ℤ) : ((196 * c) ^ 2008) % 97 = 44 := by
  sorry

end NUMINAMATH_GPT_remainder_196c_2008_mod_97_l977_97734


namespace NUMINAMATH_GPT_cross_section_area_correct_l977_97758

noncomputable def area_of_cross_section (a : ℝ) : ℝ :=
  (3 * a^2 * Real.sqrt 11) / 16

theorem cross_section_area_correct (a : ℝ) (h : 0 < a) :
  area_of_cross_section a = (3 * a^2 * Real.sqrt 11) / 16 := by
  sorry

end NUMINAMATH_GPT_cross_section_area_correct_l977_97758


namespace NUMINAMATH_GPT_ball_attendance_l977_97743

noncomputable def num_people_attending_ball (n m : ℕ) : ℕ := n + m 

theorem ball_attendance:
  ∀ (n m : ℕ), 
  n + m < 50 ∧ 
  (n - n / 4) = (5 * m / 7) →
  num_people_attending_ball n m = 41 :=
by 
  intros n m h
  have h1 : n + m < 50 := h.1
  have h2 : n - n / 4 = 5 * m / 7 := h.2
  sorry

end NUMINAMATH_GPT_ball_attendance_l977_97743


namespace NUMINAMATH_GPT_Rachel_plant_arrangement_l977_97706

-- We define Rachel's plants and lamps
inductive Plant : Type
| basil1
| basil2
| aloe
| cactus

inductive Lamp : Type
| white1
| white2
| red1
| red2

def arrangements (plants : List Plant) (lamps : List Lamp) : Nat :=
  -- This would be the function counting all valid arrangements
  -- I'm skipping the implementation
  sorry

def Rachel_arrangement_count : Nat :=
  arrangements [Plant.basil1, Plant.basil2, Plant.aloe, Plant.cactus]
                [Lamp.white1, Lamp.white2, Lamp.red1, Lamp.red2]

theorem Rachel_plant_arrangement : Rachel_arrangement_count = 22 := by
  sorry

end NUMINAMATH_GPT_Rachel_plant_arrangement_l977_97706


namespace NUMINAMATH_GPT_range_of_a_l977_97782

open Set

noncomputable def setA (a : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*x + a ≥ 0}

theorem range_of_a (a : ℝ) : (1 ∉ setA a) → a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l977_97782


namespace NUMINAMATH_GPT_james_hears_beats_per_week_l977_97770

theorem james_hears_beats_per_week
  (beats_per_minute : ℕ)
  (hours_per_day : ℕ)
  (days_per_week : ℕ)
  (H1 : beats_per_minute = 200)
  (H2 : hours_per_day = 2)
  (H3 : days_per_week = 7) :
  beats_per_minute * hours_per_day * 60 * days_per_week = 168000 := 
by
  -- sorry proof step placeholder
  sorry

end NUMINAMATH_GPT_james_hears_beats_per_week_l977_97770


namespace NUMINAMATH_GPT_notAlwaysTriangleInSecondQuadrantAfterReflection_l977_97710

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  P : Point
  Q : Point
  R : Point

def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

def reflectionOverYEqualsX (p : Point) : Point :=
  { x := p.y, y := p.x }

def reflectTriangleOverYEqualsX (T : Triangle) : Triangle :=
  { P := reflectionOverYEqualsX T.P,
    Q := reflectionOverYEqualsX T.Q,
    R := reflectionOverYEqualsX T.R }

def triangleInSecondQuadrant (T : Triangle) : Prop :=
  isInSecondQuadrant T.P ∧ isInSecondQuadrant T.Q ∧ isInSecondQuadrant T.R

theorem notAlwaysTriangleInSecondQuadrantAfterReflection
  (T : Triangle)
  (h : triangleInSecondQuadrant T)
  : ¬ (triangleInSecondQuadrant (reflectTriangleOverYEqualsX T)) := 
sorry -- Proof not required

end NUMINAMATH_GPT_notAlwaysTriangleInSecondQuadrantAfterReflection_l977_97710


namespace NUMINAMATH_GPT_transformation_correct_l977_97793

theorem transformation_correct (a b : ℝ) (h₁ : 3 * a = 2 * b) (h₂ : a ≠ 0) (h₃ : b ≠ 0) :
  a / 2 = b / 3 :=
sorry

end NUMINAMATH_GPT_transformation_correct_l977_97793


namespace NUMINAMATH_GPT_factorization_correct_l977_97778

theorem factorization_correct (x y : ℝ) : x^2 - 4 * y^2 = (x - 2 * y) * (x + 2 * y) :=
by sorry

end NUMINAMATH_GPT_factorization_correct_l977_97778


namespace NUMINAMATH_GPT_number_of_proper_subsets_of_P_l977_97725

theorem number_of_proper_subsets_of_P (P : Set ℝ) (hP : P = {x | x^2 = 1}) : 
  (∃ n, n = 2 ∧ ∃ k, k = 2 ^ n - 1 ∧ k = 3) :=
by
  sorry

end NUMINAMATH_GPT_number_of_proper_subsets_of_P_l977_97725


namespace NUMINAMATH_GPT_exists_acute_triangle_l977_97727

theorem exists_acute_triangle (a b c d e : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (h_triangle_abc : a + b > c) (h_triangle_abd : a + b > d) (h_triangle_abe : a + b > e)
  (h_triangle_bcd : b + c > d) (h_triangle_bce : b + c > e) (h_triangle_cde : c + d > e)
  (h_triangle_abc2 : a + c > b) (h_triangle_abd2 : a + d > b) (h_triangle_abe2 : a + e > b)
  (h_triangle_bcd2 : b + d > c) (h_triangle_bce2 : b + e > c) (h_triangle_cde2 : c + e > d)
  (h_triangle_abc3 : b + c > a) (h_triangle_abd3 : b + d > a) (h_triangle_abe3 : b + e > a)
  (h_triangle_bcd3 : b + d > a) (h_triangle_bce3 : c + e > a) (h_triangle_cde3 : d + e > c) :
  ∃ x y z : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
              (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧ 
              (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
              (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) ∧
              x + y > z ∧ 
              ¬ (x^2 + y^2 ≤ z^2) :=
by
  sorry

end NUMINAMATH_GPT_exists_acute_triangle_l977_97727


namespace NUMINAMATH_GPT_student_chose_number_l977_97759

theorem student_chose_number (x : ℤ) (h : 2 * x - 138 = 104) : x = 121 := by
  sorry

end NUMINAMATH_GPT_student_chose_number_l977_97759


namespace NUMINAMATH_GPT_article_cost_price_l977_97751

theorem article_cost_price (SP : ℝ) (CP : ℝ) (h1 : SP = 455) (h2 : SP = CP + 0.3 * CP) : CP = 350 :=
by sorry

end NUMINAMATH_GPT_article_cost_price_l977_97751


namespace NUMINAMATH_GPT_lisa_total_miles_flown_l977_97745

-- Definitions based on given conditions
def distance_per_trip : ℝ := 256.0
def number_of_trips : ℝ := 32.0
def total_miles_flown : ℝ := 8192.0

-- Lean statement asserting the equivalence
theorem lisa_total_miles_flown : 
    (distance_per_trip * number_of_trips = total_miles_flown) :=
by 
    sorry

end NUMINAMATH_GPT_lisa_total_miles_flown_l977_97745


namespace NUMINAMATH_GPT_friend_owns_10_bikes_l977_97712

theorem friend_owns_10_bikes (ignatius_bikes : ℕ) (tires_per_bike : ℕ) (unicycle_tires : ℕ) (tricycle_tires : ℕ) (friend_total_tires : ℕ) :
  ignatius_bikes = 4 →
  tires_per_bike = 2 →
  unicycle_tires = 1 →
  tricycle_tires = 3 →
  friend_total_tires = 3 * (ignatius_bikes * tires_per_bike) →
  (friend_total_tires - (unicycle_tires + tricycle_tires)) / tires_per_bike = 10 :=
by
  sorry

end NUMINAMATH_GPT_friend_owns_10_bikes_l977_97712


namespace NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_same_plane_l977_97773

-- Definitions and conditions
variables {Point : Type*} [MetricSpace Point]
variables {Line Plane : Type*}

def is_parallel (l₁ l₂ : Line) : Prop := sorry
def is_perpendicular (l : Line) (p : Plane) : Prop := sorry

variables (m n : Line) (α : Plane)

-- Theorem statement
theorem lines_parallel_if_perpendicular_to_same_plane :
  is_perpendicular m α → is_perpendicular n α → is_parallel m n :=
sorry

end NUMINAMATH_GPT_lines_parallel_if_perpendicular_to_same_plane_l977_97773


namespace NUMINAMATH_GPT_bird_stork_difference_l977_97798

theorem bird_stork_difference :
  let initial_birds := 3
  let initial_storks := 4
  let additional_birds := 2
  let total_birds := initial_birds + additional_birds
  total_birds - initial_storks = 1 := 
by
  let initial_birds := 3
  let initial_storks := 4
  let additional_birds := 2
  let total_birds := initial_birds + additional_birds
  show total_birds - initial_storks = 1
  sorry

end NUMINAMATH_GPT_bird_stork_difference_l977_97798


namespace NUMINAMATH_GPT_total_number_of_people_l977_97776

-- Definitions corresponding to conditions
variables (A C : ℕ)
variables (cost_adult cost_child total_revenue : ℝ)
variables (ratio_child_adult : ℝ)

-- Assumptions given in the problem
axiom cost_adult_def : cost_adult = 7
axiom cost_child_def : cost_child = 3
axiom total_revenue_def : total_revenue = 6000
axiom ratio_def : C = 3 * A
axiom revenue_eq : total_revenue = cost_adult * A + cost_child * C

-- The main statement to prove
theorem total_number_of_people : A + C = 1500 :=
by
  sorry  -- Proof of the theorem

end NUMINAMATH_GPT_total_number_of_people_l977_97776


namespace NUMINAMATH_GPT_quadratic_real_roots_k_leq_one_l977_97786

theorem quadratic_real_roots_k_leq_one (k : ℝ) : 
  (∃ x : ℝ, k * x^2 + 4 * x + 4 = 0) ↔ k ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_k_leq_one_l977_97786


namespace NUMINAMATH_GPT_line_intersection_x_value_l977_97707

theorem line_intersection_x_value :
  let line1 (x : ℝ) := 3 * x + 14
  let line2 (x : ℝ) (y : ℝ) := 5 * x - 2 * y = 40
  ∃ x : ℝ, ∃ y : ℝ, (line1 x = y) ∧ (line2 x y) ∧ (x = -68) :=
by
  sorry

end NUMINAMATH_GPT_line_intersection_x_value_l977_97707


namespace NUMINAMATH_GPT_MN_intersection_correct_l977_97789

-- Define the sets M and N
def setM : Set ℝ := {y | ∃ x ∈ (Set.univ : Set ℝ), y = x^2 + 2*x - 3}
def setN : Set ℝ := {x | |x - 2| ≤ 3}

-- Reformulated sets
def setM_reformulated : Set ℝ := {y | y ≥ -4}
def setN_reformulated : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

-- The intersection set
def MN_intersection : Set ℝ := {y | -1 ≤ y ∧ y ≤ 5}

-- The theorem stating the intersection of M and N equals MN_intersection
theorem MN_intersection_correct :
  {y | ∃ x ∈ setN_reformulated, y = x^2 + 2*x - 3} = MN_intersection :=
sorry  -- Proof not required as per instruction

end NUMINAMATH_GPT_MN_intersection_correct_l977_97789


namespace NUMINAMATH_GPT_remainder_of_m_div_1000_l977_97700

   -- Define the set T
   def T : Set ℕ := {n | 1 ≤ n ∧ n ≤ 12}

   -- Define the computation of m
   noncomputable def m : ℕ := (3^12 - 2 * 2^12 + 1) / 2

   -- Statement for the proof problem
   theorem remainder_of_m_div_1000 : m % 1000 = 625 := by
     sorry
   
end NUMINAMATH_GPT_remainder_of_m_div_1000_l977_97700


namespace NUMINAMATH_GPT_minimum_value_of_g_l977_97701

noncomputable def g (a b x : ℝ) : ℝ :=
  max (|x + a|) (|x + b|)

theorem minimum_value_of_g (a b : ℝ) (h : a < b) :
  ∃ x : ℝ, g a b x = (b - a) / 2 :=
by
  use - (a + b) / 2
  sorry

end NUMINAMATH_GPT_minimum_value_of_g_l977_97701


namespace NUMINAMATH_GPT_find_a_l977_97709

open Set

-- Define set A
def A : Set ℝ := {-1, 1, 3}

-- Define set B in terms of a
def B (a : ℝ) : Set ℝ := {a + 2, a^2 + 4}

-- State the theorem
theorem find_a (a : ℝ) (h : A ∩ B a = {3}) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l977_97709


namespace NUMINAMATH_GPT_no_factors_multiple_of_210_l977_97731

theorem no_factors_multiple_of_210 (n : ℕ) (h : n = 2^12 * 3^18 * 5^10) : ∀ d : ℕ, d ∣ n → ¬ (210 ∣ d) :=
by
  sorry

end NUMINAMATH_GPT_no_factors_multiple_of_210_l977_97731


namespace NUMINAMATH_GPT_exists_unique_root_in_interval_l977_97784

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem exists_unique_root_in_interval : 
  ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 :=
sorry

end NUMINAMATH_GPT_exists_unique_root_in_interval_l977_97784


namespace NUMINAMATH_GPT_inequality_bound_l977_97762

theorem inequality_bound 
  (a b c d : ℝ) 
  (ha : 0 ≤ a) (hb : a ≤ 1)
  (hb : 0 ≤ b) (hc : b ≤ 1)
  (hc : 0 ≤ c) (hd : c ≤ 1)
  (hd : 0 ≤ d) (ha2 : d ≤ 1) : 
  ab * (a - b) + bc * (b - c) + cd * (c - d) + da * (d - a) ≤ 8/27 := 
by
  sorry

end NUMINAMATH_GPT_inequality_bound_l977_97762


namespace NUMINAMATH_GPT_max_value_of_f_on_interval_l977_97796

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x - 2

theorem max_value_of_f_on_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_on_interval_l977_97796


namespace NUMINAMATH_GPT_Aiyanna_more_than_Alyssa_Brady_fewer_than_Aiyanna_Brady_more_than_Alyssa_l977_97716

-- Defining the number of cookies each person had
def Alyssa_cookies : ℕ := 1523
def Aiyanna_cookies : ℕ := 3720
def Brady_cookies : ℕ := 2265

-- Proving the statements
theorem Aiyanna_more_than_Alyssa : Aiyanna_cookies - Alyssa_cookies = 2197 := by
  sorry

theorem Brady_fewer_than_Aiyanna : Aiyanna_cookies - Brady_cookies = 1455 := by
  sorry

theorem Brady_more_than_Alyssa : Brady_cookies - Alyssa_cookies = 742 := by
  sorry

end NUMINAMATH_GPT_Aiyanna_more_than_Alyssa_Brady_fewer_than_Aiyanna_Brady_more_than_Alyssa_l977_97716


namespace NUMINAMATH_GPT_number_of_solutions_is_zero_l977_97722

theorem number_of_solutions_is_zero : 
  ∀ x : ℝ, (x ≠ 0 ∧ x ≠ 5) → (3 * x^2 - 15 * x) / (x^2 - 5 * x) ≠ x - 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_is_zero_l977_97722


namespace NUMINAMATH_GPT_average_speed_of_participant_l977_97740

noncomputable def average_speed (d : ℝ) : ℝ :=
  let total_distance := 4 * d
  let total_time := (d / 6) + (d / 12) + (d / 18) + (d / 24)
  total_distance / total_time

theorem average_speed_of_participant :
  ∀ (d : ℝ), d > 0 → average_speed d = 11.52 :=
by
  intros d hd
  unfold average_speed
  sorry

end NUMINAMATH_GPT_average_speed_of_participant_l977_97740


namespace NUMINAMATH_GPT_find_greater_number_l977_97755

-- Define the two numbers x and y
variables (x y : ℕ)

-- Conditions
theorem find_greater_number (h1 : x + y = 36) (h2 : x - y = 12) : x = 24 := 
by
  sorry

end NUMINAMATH_GPT_find_greater_number_l977_97755


namespace NUMINAMATH_GPT_sector_area_15deg_radius_6cm_l977_97732

noncomputable def sector_area (r : ℝ) (theta : ℝ) : ℝ :=
  0.5 * theta * r^2

theorem sector_area_15deg_radius_6cm :
  sector_area 6 (15 * Real.pi / 180) = 3 * Real.pi / 2 := by
  sorry

end NUMINAMATH_GPT_sector_area_15deg_radius_6cm_l977_97732


namespace NUMINAMATH_GPT_tiles_needed_l977_97717

def floor9ₓ12_ft : Type := {l : ℕ × ℕ // l = (9, 12)}
def tile4ₓ6_inch : Type := {l : ℕ × ℕ // l = (4, 6)}

theorem tiles_needed (floor : floor9ₓ12_ft) (tile : tile4ₓ6_inch) : 
  ∃ tiles : ℕ, tiles = 648 :=
sorry

end NUMINAMATH_GPT_tiles_needed_l977_97717


namespace NUMINAMATH_GPT_compare_exponents_product_of_roots_l977_97766

noncomputable def f (x : ℝ) (a : ℝ) := (Real.log x) / (x + a)

theorem compare_exponents : (2016 : ℝ) ^ 2017 > (2017 : ℝ) ^ 2016 :=
sorry

theorem product_of_roots (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : f x1 0 = k) (h3 : f x2 0 = k) : 
  x1 * x2 > Real.exp 2 :=
sorry

end NUMINAMATH_GPT_compare_exponents_product_of_roots_l977_97766


namespace NUMINAMATH_GPT_product_simplification_l977_97779

variables {a b c : ℝ}

theorem product_simplification (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ac) * ((ab)⁻¹ + (bc)⁻¹ + (ac)⁻¹)) = 
  ((ab + bc + ac)^2) / (abc) := 
sorry

end NUMINAMATH_GPT_product_simplification_l977_97779


namespace NUMINAMATH_GPT_no_base_makes_131b_square_l977_97728

theorem no_base_makes_131b_square : ∀ (b : ℤ), b > 3 → ∀ (n : ℤ), n * n ≠ b^2 + 3 * b + 1 :=
by
  intros b h_gt_3 n
  sorry

end NUMINAMATH_GPT_no_base_makes_131b_square_l977_97728


namespace NUMINAMATH_GPT_trajectory_of_moving_circle_l977_97715

-- Define the conditions
def passes_through (M : ℝ × ℝ) (A : ℝ × ℝ) : Prop :=
  M = A

def tangent_to_line (M : ℝ × ℝ) (l : ℝ) : Prop :=
  M.1 = -l

noncomputable def equation_of_trajectory (M : ℝ × ℝ) : Prop :=
  M.2 ^ 2 = 12 * M.1

theorem trajectory_of_moving_circle 
  (M : ℝ × ℝ)
  (A : ℝ × ℝ)
  (l : ℝ)
  (h1 : passes_through M (3, 0))
  (h2 : tangent_to_line M 3)
  : equation_of_trajectory M := 
sorry

end NUMINAMATH_GPT_trajectory_of_moving_circle_l977_97715


namespace NUMINAMATH_GPT_min_value_condition_l977_97761

theorem min_value_condition (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  3 * m + n = 1 → (1 / m + 2 / n) ≥ 5 + 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_condition_l977_97761


namespace NUMINAMATH_GPT_divisibility_of_3_pow_p_minus_2_pow_p_minus_1_l977_97790

theorem divisibility_of_3_pow_p_minus_2_pow_p_minus_1 (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  (3^p - 2^p - 1) % (42 * p) = 0 := 
by
  sorry

end NUMINAMATH_GPT_divisibility_of_3_pow_p_minus_2_pow_p_minus_1_l977_97790


namespace NUMINAMATH_GPT_problem_statement_l977_97795

def digit_sum (k : ℕ) : ℕ :=
  k.digits 10 |>.sum

theorem problem_statement :
  ∀ n : ℕ, (∃ a b : ℕ, n = digit_sum a ∧ n = digit_sum b ∧ n = digit_sum (a + b)) ↔ (∃ k : ℕ, n = 9 * k) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l977_97795


namespace NUMINAMATH_GPT_monomial_k_add_n_l977_97703

variable (k n : ℤ)

-- Conditions
def is_monomial_coefficient (k : ℤ) : Prop := -k = 5
def is_monomial_degree (n : ℤ) : Prop := n + 1 = 7

-- Theorem to prove
theorem monomial_k_add_n (hk : is_monomial_coefficient k) (hn : is_monomial_degree n) : k + n = 1 :=
by
  sorry

end NUMINAMATH_GPT_monomial_k_add_n_l977_97703


namespace NUMINAMATH_GPT_child_ticket_cost_l977_97760

noncomputable def cost_of_child_ticket : ℝ := 3.50

theorem child_ticket_cost
  (adult_ticket_price : ℝ)
  (total_tickets : ℕ)
  (total_cost : ℝ)
  (adult_tickets_bought : ℕ)
  (adult_ticket_price_eq : adult_ticket_price = 5.50)
  (total_tickets_bought_eq : total_tickets = 21)
  (total_cost_eq : total_cost = 83.50)
  (adult_tickets_count : adult_tickets_bought = 5) :
  cost_of_child_ticket = 3.50 :=
by
  sorry

end NUMINAMATH_GPT_child_ticket_cost_l977_97760


namespace NUMINAMATH_GPT_pieces_left_to_place_l977_97721

noncomputable def total_pieces : ℕ := 300
noncomputable def reyn_pieces : ℕ := 25
noncomputable def rhys_pieces : ℕ := 2 * reyn_pieces
noncomputable def rory_pieces : ℕ := 3 * reyn_pieces
noncomputable def placed_pieces : ℕ := reyn_pieces + rhys_pieces + rory_pieces
noncomputable def remaining_pieces : ℕ := total_pieces - placed_pieces

theorem pieces_left_to_place : remaining_pieces = 150 :=
by sorry

end NUMINAMATH_GPT_pieces_left_to_place_l977_97721


namespace NUMINAMATH_GPT_zero_function_unique_l977_97713

theorem zero_function_unique 
  (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), f (x ^ (42 ^ 42) + y) = f (x ^ 3 + 2 * y) + f (x ^ 12)) : 
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_function_unique_l977_97713


namespace NUMINAMATH_GPT_households_selected_l977_97741

theorem households_selected (H : ℕ) (M L S n h : ℕ)
  (h1 : H = 480)
  (h2 : M = 200)
  (h3 : L = 160)
  (h4 : H = M + L + S)
  (h5 : h = 6)
  (h6 : (h : ℚ) / n = (S : ℚ) / H) : n = 24 :=
by
  sorry

end NUMINAMATH_GPT_households_selected_l977_97741


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l977_97780

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l977_97780


namespace NUMINAMATH_GPT_percentage_increase_l977_97783

noncomputable def percentMoreThan (a b : ℕ) : ℕ :=
  ((a - b) * 100) / b

theorem percentage_increase (x y z : ℕ) (h1 : z = 300) (h2 : x = 5 * y / 4) (h3 : x + y + z = 1110) :
  percentMoreThan y z = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l977_97783


namespace NUMINAMATH_GPT_smallest_b_greater_than_1_l977_97763

def g (x : ℕ) : ℕ :=
  if x % 35 = 0 then x / 35
  else if x % 7 = 0 then 5 * x
  else if x % 5 = 0 then 7 * x
  else x + 5

def g_iter (n : ℕ) (x : ℕ) : ℕ := Nat.iterate g n x

theorem smallest_b_greater_than_1 (b : ℕ) :
  (b > 1) → 
  g_iter 1 3 = 8 ∧ g_iter b 3 = 8 →
  b = 21 := by
  sorry

end NUMINAMATH_GPT_smallest_b_greater_than_1_l977_97763


namespace NUMINAMATH_GPT_sequence_term_l977_97738

noncomputable def geometric_sum (n : ℕ) : ℝ :=
  2 * (1 - (1 / 2) ^ n) / (1 - 1 / 2)

theorem sequence_term (m n : ℕ) (h : n < m) : 
  let Sn := geometric_sum n
  let Sn_plus_1 := geometric_sum (n + 1)
  Sn - Sn_plus_1 = -(1 / 2 ^ (n - 1)) := sorry

end NUMINAMATH_GPT_sequence_term_l977_97738


namespace NUMINAMATH_GPT_marians_groceries_l977_97775

variables (G : ℝ)

theorem marians_groceries :
  let initial_balance := 126
  let returned_amount := 45
  let new_balance := 171
  let gas_expense := G / 2
  initial_balance + G + gas_expense - returned_amount = new_balance → G = 60 :=
sorry

end NUMINAMATH_GPT_marians_groceries_l977_97775


namespace NUMINAMATH_GPT_bobby_initial_candy_count_l977_97730

theorem bobby_initial_candy_count (x : ℕ) (h : x + 17 = 43) : x = 26 :=
by
  sorry

end NUMINAMATH_GPT_bobby_initial_candy_count_l977_97730


namespace NUMINAMATH_GPT_not_possible_coloring_l977_97735

def color : Nat → Option ℕ := sorry

def all_colors_used (f : Nat → Option ℕ) : Prop := 
  (∃ n, f n = some 0) ∧ (∃ n, f n = some 1) ∧ (∃ n, f n = some 2)

def valid_coloring (f : Nat → Option ℕ) : Prop :=
  ∀ (a b : Nat), 1 < a → 1 < b → f a ≠ f b → f (a * b) ≠ f a ∧ f (a * b) ≠ f b

theorem not_possible_coloring : ¬ (∃ f : Nat → Option ℕ, all_colors_used f ∧ valid_coloring f) := 
sorry

end NUMINAMATH_GPT_not_possible_coloring_l977_97735


namespace NUMINAMATH_GPT_square_side_length_l977_97739

theorem square_side_length (x : ℝ) (h : 4 * x = x^2) : x = 4 := 
by
  sorry

end NUMINAMATH_GPT_square_side_length_l977_97739


namespace NUMINAMATH_GPT_part1_part2_l977_97724

noncomputable def f : ℝ → ℝ := sorry

variable (x y : ℝ)
variable (hx0 : 0 < x)
variable (hy0 : 0 < y)
variable (hx12 : x < 1 → f x > 0)
variable (hf_half : f (1 / 2) = 1)
variable (hf_mul : f (x * y) = f x + f y)

theorem part1 : (∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2) := sorry

theorem part2 : (∀ x, 3 < x → x < 4 → f (x - 3) > f (1 / x) - 2) := sorry

end NUMINAMATH_GPT_part1_part2_l977_97724


namespace NUMINAMATH_GPT_solve_for_y_l977_97788

theorem solve_for_y 
  (x y : ℝ) 
  (h1 : 2 * x - 3 * y = 9) 
  (h2 : x + y = 8) : 
  y = 1.4 := 
sorry

end NUMINAMATH_GPT_solve_for_y_l977_97788


namespace NUMINAMATH_GPT_minimum_value_of_function_l977_97757

theorem minimum_value_of_function (x : ℝ) (h : x * Real.log 2 / Real.log 3 ≥ 1) : 
  ∃ t : ℝ, t = 2^x ∧ t ≥ 3 ∧ ∀ y : ℝ, y = t^2 - 2*t - 3 → y = (t-1)^2 - 4 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_function_l977_97757


namespace NUMINAMATH_GPT_perimeter_gt_sixteen_l977_97781

theorem perimeter_gt_sixteen (a b : ℝ) (h : a * b > 2 * a + 2 * b) : 2 * (a + b) > 16 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_gt_sixteen_l977_97781


namespace NUMINAMATH_GPT_valid_range_of_x_l977_97749

theorem valid_range_of_x (x : ℝ) : 3 * x + 5 ≥ 0 → x ≥ -5 / 3 := 
by
  sorry

end NUMINAMATH_GPT_valid_range_of_x_l977_97749


namespace NUMINAMATH_GPT_range_of_a_l977_97737

theorem range_of_a (a : ℝ) : |a - 1| + |a - 4| = 3 ↔ 1 ≤ a ∧ a ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l977_97737


namespace NUMINAMATH_GPT_even_n_of_even_Omega_P_l977_97705

-- Define the Omega function
def Omega (N : ℕ) : ℕ := 
  N.factors.length

-- Define the polynomial function P
def P (x : ℕ) (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  List.prod (List.map (λ i => x + a i) (List.range n))

theorem even_n_of_even_Omega_P (a : ℕ → ℕ) (n : ℕ)
  (H : ∀ k > 0, Even (Omega (P k a n))) : Even n :=
by
  sorry

end NUMINAMATH_GPT_even_n_of_even_Omega_P_l977_97705


namespace NUMINAMATH_GPT_ninety_eight_times_ninety_eight_l977_97711

theorem ninety_eight_times_ninety_eight : 98 * 98 = 9604 :=
by
  sorry

end NUMINAMATH_GPT_ninety_eight_times_ninety_eight_l977_97711


namespace NUMINAMATH_GPT_cost_price_of_a_ball_l977_97748

variables (C : ℝ) (selling_price : ℝ) (cost_price_20_balls : ℝ) (loss_on_20_balls : ℝ)

def cost_price_per_ball (C : ℝ) := (20 * C - 720 = 5 * C)

theorem cost_price_of_a_ball :
  (∃ C : ℝ, 20 * C - 720 = 5 * C) -> (C = 48) := 
by
  sorry

end NUMINAMATH_GPT_cost_price_of_a_ball_l977_97748


namespace NUMINAMATH_GPT_new_person_weight_is_90_l977_97702

-- Define the weight of the replaced person
def replaced_person_weight : ℝ := 40

-- Define the increase in average weight when the new person replaces the replaced person
def increase_in_average_weight : ℝ := 10

-- Define the increase in total weight as 5 times the increase in average weight
def increase_in_total_weight (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase

-- Define the weight of the new person
def new_person_weight (replaced_w : ℝ) (total_increase : ℝ) : ℝ := replaced_w + total_increase

-- Prove that the weight of the new person is 90 kg
theorem new_person_weight_is_90 :
  new_person_weight replaced_person_weight (increase_in_total_weight 5 increase_in_average_weight) = 90 := 
by 
  -- sorry will skip the proof, as required
  sorry

end NUMINAMATH_GPT_new_person_weight_is_90_l977_97702


namespace NUMINAMATH_GPT_num_outfits_l977_97787

def num_shirts := 6
def num_ties := 4
def num_pants := 3
def outfits : ℕ := num_shirts * num_pants * (num_ties + 1)

theorem num_outfits: outfits = 90 :=
by 
  -- sorry will be removed when proof is provided
  sorry

end NUMINAMATH_GPT_num_outfits_l977_97787


namespace NUMINAMATH_GPT_rent_percentage_l977_97746

variable (E : ℝ)
variable (last_year_rent : ℝ := 0.20 * E)
variable (this_year_earnings : ℝ := 1.20 * E)
variable (this_year_rent : ℝ := 0.30 * this_year_earnings)

theorem rent_percentage (E : ℝ) (h_last_year_rent : last_year_rent = 0.20 * E)
  (h_this_year_earnings : this_year_earnings = 1.20 * E)
  (h_this_year_rent : this_year_rent = 0.30 * this_year_earnings) : 
  this_year_rent / last_year_rent * 100 = 180 := by
  sorry

end NUMINAMATH_GPT_rent_percentage_l977_97746


namespace NUMINAMATH_GPT_example_theorem_l977_97744

def not_a_term : Prop := ∀ n : ℕ, ¬ (24 - 2 * n = 3)

theorem example_theorem : not_a_term :=
  by sorry

end NUMINAMATH_GPT_example_theorem_l977_97744


namespace NUMINAMATH_GPT_sub_neg_seven_eq_neg_fourteen_l977_97772

theorem sub_neg_seven_eq_neg_fourteen : (-7) - 7 = -14 := 
  by
  sorry

end NUMINAMATH_GPT_sub_neg_seven_eq_neg_fourteen_l977_97772
