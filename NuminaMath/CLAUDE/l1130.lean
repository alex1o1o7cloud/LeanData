import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_property_l1130_113084

theorem divisibility_property (n a b c d : ℤ) 
  (hn : n > 0)
  (h1 : n ∣ (a + b + c + d))
  (h2 : n ∣ (a^2 + b^2 + c^2 + d^2)) :
  n ∣ (a^4 + b^4 + c^4 + d^4 + 4*a*b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l1130_113084


namespace NUMINAMATH_CALUDE_system_solution_l1130_113070

theorem system_solution : 
  ∀ x y : ℝ, x + y = 3 ∧ x^5 + y^5 = 33 → (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1130_113070


namespace NUMINAMATH_CALUDE_line_parallel_from_perpendicular_to_parallel_planes_l1130_113003

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- Define the theorem
theorem line_parallel_from_perpendicular_to_parallel_planes
  (m n : Line) (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_β : perpendicular n β)
  (h_α_parallel_β : plane_parallel α β) :
  parallel m n :=
sorry

end NUMINAMATH_CALUDE_line_parallel_from_perpendicular_to_parallel_planes_l1130_113003


namespace NUMINAMATH_CALUDE_min_value_fraction_min_value_fraction_equality_l1130_113092

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x - 2*y + 3*z = 0) : 
  (y^2 / (x*z)) ≥ 3 := by
sorry

theorem min_value_fraction_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x - 2*y + 3*z = 0) : 
  (y^2 / (x*z) = 3) ↔ (x = 3*z) := by
sorry

end NUMINAMATH_CALUDE_min_value_fraction_min_value_fraction_equality_l1130_113092


namespace NUMINAMATH_CALUDE_photo_selection_choices_l1130_113049

theorem photo_selection_choices : ∀ n : ℕ, n = 5 →
  (Nat.choose n 3 + Nat.choose n 4 = 15) :=
by
  sorry

end NUMINAMATH_CALUDE_photo_selection_choices_l1130_113049


namespace NUMINAMATH_CALUDE_repair_center_solution_l1130_113041

/-- Represents a bonus distribution plan -/
structure BonusPlan where
  techBonus : ℕ
  assistBonus : ℕ

/-- Represents the repair center staff and bonus distribution -/
structure RepairCenter where
  techCount : ℕ
  assistCount : ℕ
  totalBonus : ℕ
  bonusPlans : List BonusPlan

/-- The conditions of the repair center problem -/
def repairCenterConditions (rc : RepairCenter) : Prop :=
  rc.techCount + rc.assistCount = 15 ∧
  rc.techCount = 2 * rc.assistCount ∧
  rc.totalBonus = 20000 ∧
  ∀ plan ∈ rc.bonusPlans,
    plan.techBonus ≥ plan.assistBonus ∧
    plan.assistBonus ≥ 800 ∧
    plan.techBonus % 100 = 0 ∧
    plan.assistBonus % 100 = 0 ∧
    rc.techCount * plan.techBonus + rc.assistCount * plan.assistBonus = rc.totalBonus

/-- The theorem stating the solution to the repair center problem -/
theorem repair_center_solution :
  ∃ (rc : RepairCenter),
    repairCenterConditions rc ∧
    rc.techCount = 10 ∧
    rc.assistCount = 5 ∧
    rc.bonusPlans = [
      { techBonus := 1600, assistBonus := 800 },
      { techBonus := 1500, assistBonus := 1000 },
      { techBonus := 1400, assistBonus := 1200 }
    ] :=
  sorry

end NUMINAMATH_CALUDE_repair_center_solution_l1130_113041


namespace NUMINAMATH_CALUDE_ball_distribution_equality_l1130_113012

theorem ball_distribution_equality (k : ℤ) : ∃ (n : ℕ), (19 + 6 * n) % 95 = 0 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ball_distribution_equality_l1130_113012


namespace NUMINAMATH_CALUDE_theater_line_permutations_l1130_113076

theorem theater_line_permutations : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_theater_line_permutations_l1130_113076


namespace NUMINAMATH_CALUDE_distance_difference_l1130_113014

def house_to_bank : ℕ := 800
def bank_to_pharmacy : ℕ := 1300
def pharmacy_to_school : ℕ := 1700

theorem distance_difference : 
  (house_to_bank + bank_to_pharmacy) - pharmacy_to_school = 400 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1130_113014


namespace NUMINAMATH_CALUDE_negation_of_log_inequality_l1130_113089

theorem negation_of_log_inequality (p : Prop) : 
  (p ↔ ∀ x : ℝ, Real.log x > 1) → 
  (¬p ↔ ∃ x₀ : ℝ, Real.log x₀ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_log_inequality_l1130_113089


namespace NUMINAMATH_CALUDE_cipher_decoding_probabilities_l1130_113010

-- Define the probabilities of success for each person
def p_A : ℝ := 0.4
def p_B : ℝ := 0.35
def p_C : ℝ := 0.3

-- Define the probability of exactly two successes
def prob_two_successes : ℝ :=
  p_A * p_B * (1 - p_C) + p_A * (1 - p_B) * p_C + (1 - p_A) * p_B * p_C

-- Define the probability of at least one success
def prob_at_least_one_success : ℝ :=
  1 - (1 - p_A) * (1 - p_B) * (1 - p_C)

-- Theorem statement
theorem cipher_decoding_probabilities :
  prob_two_successes = 0.239 ∧ prob_at_least_one_success = 0.727 := by
  sorry

end NUMINAMATH_CALUDE_cipher_decoding_probabilities_l1130_113010


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1130_113095

/-- A point in the fourth quadrant with given conditions has coordinates (7, -3) -/
theorem point_in_fourth_quadrant (x y : ℝ) (h1 : x > 0) (h2 : y < 0) 
  (h3 : |x| = 7) (h4 : y^2 = 9) : (x, y) = (7, -3) := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l1130_113095


namespace NUMINAMATH_CALUDE_magic_8_ball_probability_l1130_113081

theorem magic_8_ball_probability :
  let n : ℕ := 7  -- total number of questions
  let k : ℕ := 3  -- number of positive answers
  let p : ℚ := 1/3  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 :=
by sorry

end NUMINAMATH_CALUDE_magic_8_ball_probability_l1130_113081


namespace NUMINAMATH_CALUDE_slope_positive_for_a_in_open_unit_interval_l1130_113098

theorem slope_positive_for_a_in_open_unit_interval :
  ∀ a : ℝ, 0 < a ∧ a < 1 →
  let k := -(2^a - 1) / Real.log a
  k > 0 := by
sorry

end NUMINAMATH_CALUDE_slope_positive_for_a_in_open_unit_interval_l1130_113098


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l1130_113055

theorem weight_loss_challenge (original_weight : ℝ) (h : original_weight > 0) :
  let weight_after_loss := 0.87 * original_weight
  let final_measured_weight := 0.8874 * original_weight
  let clothes_weight := final_measured_weight - weight_after_loss
  clothes_weight / weight_after_loss = 0.02 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l1130_113055


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1130_113044

def U : Set ℕ := {2, 3, 4}
def A : Set ℕ := {2, 3}

theorem complement_of_A_in_U : (U \ A) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1130_113044


namespace NUMINAMATH_CALUDE_geometric_mean_proof_l1130_113007

theorem geometric_mean_proof (a b : ℝ) (hb : b ≠ 0) :
  Real.sqrt ((2 * (a^2 - a*b)) / (35*b) * (10*a) / (7*(a*b - b^2))) = 2*a / (7*b) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_proof_l1130_113007


namespace NUMINAMATH_CALUDE_polynomial_existence_l1130_113017

theorem polynomial_existence : 
  ∃ (P : ℝ → ℝ → ℝ → ℝ), ∀ (t : ℝ), P (t^1993) (t^1994) (t + t^1995) = t := by
  sorry

end NUMINAMATH_CALUDE_polynomial_existence_l1130_113017


namespace NUMINAMATH_CALUDE_parabola_properties_l1130_113061

/-- A parabola with focus on a given line -/
structure Parabola where
  p : ℝ
  focus_on_line : (p / 2) + (0 : ℝ) - 2 = 0

/-- The directrix of a parabola -/
def directrix (C : Parabola) : ℝ → Prop :=
  λ x => x = -(C.p / 2)

theorem parabola_properties (C : Parabola) :
  C.p = 4 ∧ directrix C = λ x => x = -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l1130_113061


namespace NUMINAMATH_CALUDE_shaded_design_area_l1130_113024

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a triangle in the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- The shaded design in the 7x7 grid -/
def shaded_design : List GridTriangle := sorry

/-- Calculates the area of a single triangle in the grid -/
def triangle_area (t : GridTriangle) : Rat := sorry

/-- Calculates the total area of the shaded design -/
def total_area (design : List GridTriangle) : Rat :=
  design.map triangle_area |>.sum

/-- The theorem stating that the area of the shaded design is 1.5 -/
theorem shaded_design_area :
  total_area shaded_design = 3/2 := by sorry

end NUMINAMATH_CALUDE_shaded_design_area_l1130_113024


namespace NUMINAMATH_CALUDE_lcm_of_54_and_16_l1130_113028

theorem lcm_of_54_and_16 : Nat.lcm 54 16 = 48 :=
by
  have h1 : Nat.gcd 54 16 = 18 := by sorry
  sorry

end NUMINAMATH_CALUDE_lcm_of_54_and_16_l1130_113028


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1130_113026

/-- Given a triangle with inradius 2.5 cm and area 40 cm², its perimeter is 32 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) : 
  r = 2.5 → A = 40 → A = r * (p / 2) → p = 32 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1130_113026


namespace NUMINAMATH_CALUDE_coin_arrangement_l1130_113033

theorem coin_arrangement (n : ℕ) : 
  n ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) → 
  (n * 4 = 12 ↔ n = 3) :=
by sorry

end NUMINAMATH_CALUDE_coin_arrangement_l1130_113033


namespace NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l1130_113047

/-- Joel's current age -/
def joel_current_age : ℕ := 8

/-- Joel's dad's current age -/
def dad_current_age : ℕ := 37

/-- The number of years until Joel's dad is twice Joel's age -/
def years_until_double : ℕ := dad_current_age - 2 * joel_current_age

/-- Joel's age when his dad is twice as old as him -/
def joel_future_age : ℕ := joel_current_age + years_until_double

theorem joel_age_when_dad_twice_as_old :
  joel_future_age = 29 :=
sorry

end NUMINAMATH_CALUDE_joel_age_when_dad_twice_as_old_l1130_113047


namespace NUMINAMATH_CALUDE_min_value_theorem_l1130_113025

theorem min_value_theorem (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) (h_prod : x * y * z = 1) :
  x^2 + 8*x*y + 9*y^2 + 8*y*z + 2*z^2 ≥ 18 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 1 ∧ 
    a^2 + 8*a*b + 9*b^2 + 8*b*c + 2*c^2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1130_113025


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1130_113034

/-- A geometric sequence with positive terms and common ratio greater than 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : ∀ n, a n > 0
  h_ratio : q > 1
  h_geometric : ∀ n, a (n + 1) = q * a n

/-- Theorem statement for the geometric sequence problem -/
theorem geometric_sequence_problem (seq : GeometricSequence)
  (h1 : seq.a 3 + seq.a 5 = 20)
  (h2 : seq.a 2 * seq.a 6 = 64) :
  seq.a 6 = 32 := by
    sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1130_113034


namespace NUMINAMATH_CALUDE_chair_cost_l1130_113045

theorem chair_cost (total_spent : ℕ) (table_cost : ℕ) (num_chairs : ℕ) :
  total_spent = 56 →
  table_cost = 34 →
  num_chairs = 2 →
  ∃ (chair_cost : ℕ), 
    chair_cost * num_chairs = total_spent - table_cost ∧
    chair_cost = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_chair_cost_l1130_113045


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_problem_l1130_113036

/-- Calculates the length of the second train given the speeds of two trains moving in opposite directions, the length of the first train, and the time taken to cross each other. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (length1 : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let total_distance := relative_speed * crossing_time / 3600
  total_distance - length1

/-- The length of the second train is 0.9 km given the specified conditions -/
theorem second_train_length_problem : 
  second_train_length 60 90 1.10 47.99999999999999 = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_problem_l1130_113036


namespace NUMINAMATH_CALUDE_f_decreasing_interval_l1130_113097

-- Define the function
def f (x : ℝ) : ℝ := (x - 3) * |x|

-- Define the property of being decreasing on an interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y ≤ f x

-- Theorem statement
theorem f_decreasing_interval :
  ∃ (a b : ℝ), a = 0 ∧ b = 3/2 ∧
  is_decreasing_on f a b ∧
  ∀ (c d : ℝ), c < a ∨ b < d → ¬(is_decreasing_on f c d) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_interval_l1130_113097


namespace NUMINAMATH_CALUDE_prism_minimum_characteristics_l1130_113018

/-- A prism is a polyhedron with two congruent parallel faces (bases) and all other faces (lateral faces) are parallelograms. -/
structure Prism where
  base_edges : ℕ
  height : ℝ
  height_pos : height > 0

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ := p.base_edges + 2

/-- The number of edges in a prism -/
def num_edges (p : Prism) : ℕ := 3 * p.base_edges

/-- The number of lateral edges in a prism -/
def num_lateral_edges (p : Prism) : ℕ := p.base_edges

/-- The number of vertices in a prism -/
def num_vertices (p : Prism) : ℕ := 2 * p.base_edges

/-- Theorem about the minimum characteristics of a prism -/
theorem prism_minimum_characteristics :
  (∀ p : Prism, num_faces p ≥ 5) ∧
  (∃ p : Prism, num_faces p = 5 ∧
                num_edges p = 9 ∧
                num_lateral_edges p = 3 ∧
                num_vertices p = 6) := by sorry

end NUMINAMATH_CALUDE_prism_minimum_characteristics_l1130_113018


namespace NUMINAMATH_CALUDE_expression_evaluation_l1130_113046

theorem expression_evaluation : 4 * (5^2 + 5^2 + 5^2 + 5^2) = 400 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1130_113046


namespace NUMINAMATH_CALUDE_parallel_condition_l1130_113059

/-- Two lines l₁ and l₂ in the plane -/
structure TwoLines where
  a : ℝ
  l₁ : ℝ → ℝ → ℝ := λ x y => a * x + (a + 2) * y + 1
  l₂ : ℝ → ℝ → ℝ := λ x y => x + a * y + 2

/-- The condition for two lines to be parallel -/
def parallel (lines : TwoLines) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ lines.a = k ∧ lines.a + 2 = k * lines.a

/-- The statement to be proved -/
theorem parallel_condition (lines : TwoLines) :
  (parallel lines → lines.a = -1) ∧ ¬(lines.a = -1 → parallel lines) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l1130_113059


namespace NUMINAMATH_CALUDE_b_range_l1130_113001

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then (x + 1) / x^2 else Real.log (x + 2)

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x - 4

-- State the theorem
theorem b_range (b : ℝ) :
  (∃ a : ℝ, f a + g b = 1) → b ∈ Set.Icc (-3/2) (7/2) :=
by sorry

end NUMINAMATH_CALUDE_b_range_l1130_113001


namespace NUMINAMATH_CALUDE_nearest_integer_to_power_l1130_113062

theorem nearest_integer_to_power : 
  ∃ (n : ℤ), n = 2654 ∧ 
  ∀ (m : ℤ), |((3 : ℝ) + Real.sqrt 5)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5)^6 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_power_l1130_113062


namespace NUMINAMATH_CALUDE_line_perp_to_parallel_planes_l1130_113030

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_to_parallel_planes
  (m : Line) (α β : Plane)
  (h1 : perpendicular m β)
  (h2 : parallel α β) :
  perpendicular m α :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_parallel_planes_l1130_113030


namespace NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l1130_113000

theorem min_value_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) :=
sorry

theorem min_value_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a / (3 * b) + b / (6 * c) + c / (9 * a) = 3 / Real.rpow 162 (1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_min_value_achievable_l1130_113000


namespace NUMINAMATH_CALUDE_compound_composition_l1130_113006

/-- The atomic weight of aluminum in g/mol -/
def aluminum_weight : ℝ := 26.98

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of the compound in g/mol -/
def compound_weight : ℝ := 132

/-- The number of chlorine atoms in the compound -/
def chlorine_atoms : ℕ := 3

theorem compound_composition :
  ∃ (n : ℕ), n = chlorine_atoms ∧
  compound_weight = aluminum_weight + n * chlorine_weight :=
sorry

end NUMINAMATH_CALUDE_compound_composition_l1130_113006


namespace NUMINAMATH_CALUDE_inequality_proof_l1130_113082

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1130_113082


namespace NUMINAMATH_CALUDE_father_age_is_27_l1130_113008

def father_son_ages (father_age son_age : ℕ) : Prop :=
  (father_age = 3 * son_age + 3) ∧
  (father_age + 3 = 2 * (son_age + 3) + 8)

theorem father_age_is_27 :
  ∃ (son_age : ℕ), father_son_ages 27 son_age :=
sorry

end NUMINAMATH_CALUDE_father_age_is_27_l1130_113008


namespace NUMINAMATH_CALUDE_smallest_all_ones_divisible_by_d_is_correct_l1130_113032

def d : ℕ := 3 * (10^100 - 1) / 9

def is_all_ones (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

def smallest_all_ones_divisible_by_d : ℕ := (10^300 - 1) / 9

theorem smallest_all_ones_divisible_by_d_is_correct :
  is_all_ones smallest_all_ones_divisible_by_d ∧
  smallest_all_ones_divisible_by_d % d = 0 ∧
  ∀ n : ℕ, is_all_ones n → n % d = 0 → n ≥ smallest_all_ones_divisible_by_d :=
by sorry

end NUMINAMATH_CALUDE_smallest_all_ones_divisible_by_d_is_correct_l1130_113032


namespace NUMINAMATH_CALUDE_product_equals_square_l1130_113058

theorem product_equals_square : 1000 * 1993 * 0.1993 * 10 = (1993 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l1130_113058


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1130_113042

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (a = Real.sqrt 3 ∧ b = 1 → Complex.abs ((1 + Complex.I * b) / (a + Complex.I)) = Real.sqrt 2 / 2) ∧
  (∃ (x y : ℝ), (x ≠ Real.sqrt 3 ∨ y ≠ 1) ∧ Complex.abs ((1 + Complex.I * y) / (x + Complex.I)) = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1130_113042


namespace NUMINAMATH_CALUDE_bus_ticket_solution_l1130_113054

/-- Represents the number and cost of bus tickets -/
structure BusTickets where
  total_tickets : ℕ
  total_cost : ℕ
  one_way_cost : ℕ
  round_trip_cost : ℕ

/-- Theorem stating the correct number of one-way and round-trip tickets -/
theorem bus_ticket_solution (tickets : BusTickets)
  (h1 : tickets.total_tickets = 99)
  (h2 : tickets.total_cost = 280)
  (h3 : tickets.one_way_cost = 2)
  (h4 : tickets.round_trip_cost = 3) :
  ∃ (one_way round_trip : ℕ),
    one_way + round_trip = tickets.total_tickets ∧
    one_way * tickets.one_way_cost + round_trip * tickets.round_trip_cost = tickets.total_cost ∧
    one_way = 17 ∧
    round_trip = 82 := by
  sorry

end NUMINAMATH_CALUDE_bus_ticket_solution_l1130_113054


namespace NUMINAMATH_CALUDE_arithmetic_mean_change_l1130_113011

theorem arithmetic_mean_change (a b c d : ℝ) : 
  (a + b + c + d) / 4 = 10 →
  b + c + d = 33 →
  a + c + d = 36 →
  a + b + d = 39 →
  (a + b + c) / 3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_change_l1130_113011


namespace NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l1130_113039

-- Part 1: Equation solution
theorem equation_solution :
  ∃ x : ℚ, (2 / (x + 3) - (x - 3) / (2*x + 6) = 1) ∧ x = 1/3 := by sorry

-- Part 2: System of inequalities solution
theorem inequalities_solution :
  ∀ x : ℚ, (2*x - 1 > 3*(x - 1) ∧ (5 - x)/2 < x + 4) ↔ (-1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_inequalities_solution_l1130_113039


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l1130_113066

/-- Two lines are distinct if they are not equal -/
def distinct_lines (l m : Line) : Prop := l ≠ m

/-- Two planes are distinct if they are not equal -/
def distinct_planes (α β : Plane) : Prop := α ≠ β

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (α : Plane) : Prop := sorry

/-- Two planes are parallel -/
def planes_parallel (α β : Plane) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perpendicular (l m : Line) : Prop := sorry

theorem perpendicular_lines_from_parallel_planes 
  (l m : Line) (α β : Plane) 
  (h1 : distinct_lines l m)
  (h2 : distinct_planes α β)
  (h3 : planes_parallel α β)
  (h4 : line_perp_plane l α)
  (h5 : line_parallel_plane m β) :
  lines_perpendicular l m := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l1130_113066


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l1130_113075

theorem max_value_sum_of_roots (a b c : ℝ) (h : a + b + c = 1) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧
  (Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1) ≤ max) ∧
  (∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 1 ∧
    Real.sqrt (3 * a₀ + 1) + Real.sqrt (3 * b₀ + 1) + Real.sqrt (3 * c₀ + 1) = max) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l1130_113075


namespace NUMINAMATH_CALUDE_ball_diameter_proof_l1130_113013

theorem ball_diameter_proof (h s d : ℝ) (h_pos : h > 0) (s_pos : s > 0) (d_pos : d > 0) :
  h / s = (h / s) / (1 + d / (h / s)) → h / s = 1.25 → s = 1 → d = 0.23 → h / s = 0.23 :=
by sorry

end NUMINAMATH_CALUDE_ball_diameter_proof_l1130_113013


namespace NUMINAMATH_CALUDE_unique_divisor_l1130_113038

def sum_even_two_digit : Nat := 2430

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

def digits_sum (n : Nat) : Nat := (n / 10) + (n % 10)

def reverse_digits (n : Nat) : Nat := (n % 10) * 10 + (n / 10)

theorem unique_divisor :
  ∃! n : Nat, is_two_digit n ∧ 
    sum_even_two_digit % n = 0 ∧
    sum_even_two_digit / n = reverse_digits n ∧
    digits_sum (sum_even_two_digit / n) = 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisor_l1130_113038


namespace NUMINAMATH_CALUDE_point_C_coordinates_l1130_113086

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line on which point C lies
def line_C (x : ℝ) : ℝ := 3 * x + 3

-- Define the area of the triangle
def triangle_area : ℝ := 10

-- Theorem statement
theorem point_C_coordinates :
  ∃ (C : ℝ × ℝ), 
    (C.2 = line_C C.1) ∧ 
    (abs ((C.1 - A.1) * (B.2 - A.2) - (B.1 - A.1) * (C.2 - A.2)) / 2 = triangle_area) ∧
    ((C = (-1, 0)) ∨ (C = (5/3, 8))) :=
sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l1130_113086


namespace NUMINAMATH_CALUDE_fish_distribution_l1130_113087

theorem fish_distribution (total_fish : ℕ) (num_bowls : ℕ) (fish_per_bowl : ℕ) :
  total_fish = 6003 →
  num_bowls = 261 →
  total_fish = num_bowls * fish_per_bowl →
  fish_per_bowl = 23 := by
  sorry

end NUMINAMATH_CALUDE_fish_distribution_l1130_113087


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l1130_113053

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (planeParallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_parallel_lines
  (a b : Line) (α β : Plane)
  (distinct_lines : a ≠ b)
  (distinct_planes : α ≠ β)
  (a_perp_α : perpendicular a α)
  (b_perp_β : perpendicular b β)
  (a_parallel_b : parallel a b) :
  planeParallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_parallel_lines_l1130_113053


namespace NUMINAMATH_CALUDE_one_and_quarter_of_what_is_forty_l1130_113090

theorem one_and_quarter_of_what_is_forty : ∃ x : ℝ, 1.25 * x = 40 ∧ x = 32 := by
  sorry

end NUMINAMATH_CALUDE_one_and_quarter_of_what_is_forty_l1130_113090


namespace NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1130_113035

theorem smallest_n_square_and_cube : 
  (∃ (n : ℕ), n > 0 ∧ 
   (∃ (k : ℕ), 5 * n = k^2) ∧ 
   (∃ (m : ℕ), 7 * n = m^3) ∧
   (∀ (x : ℕ), x > 0 → 
    (∃ (y : ℕ), 5 * x = y^2) → 
    (∃ (z : ℕ), 7 * x = z^3) → 
    x ≥ 1715)) ∧
  (∃ (k m : ℕ), 5 * 1715 = k^2 ∧ 7 * 1715 = m^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_square_and_cube_l1130_113035


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1130_113016

theorem simplify_and_evaluate (x : ℝ) :
  x = -2 →
  (1 - 2 / (2 - x)) / (x / (x^2 - 4*x + 4)) = x - 2 ∧
  x - 2 = -4 :=
by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1130_113016


namespace NUMINAMATH_CALUDE_twelfth_term_value_l1130_113031

/-- An arithmetic sequence with a₂ = -8 and common difference d = 2 -/
def arithmetic_seq (n : ℕ) : ℤ :=
  let a₁ : ℤ := -10  -- Derived from a₂ = -8 and d = 2
  a₁ + (n - 1) * 2

theorem twelfth_term_value : arithmetic_seq 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_value_l1130_113031


namespace NUMINAMATH_CALUDE_carla_liquid_consumption_l1130_113063

/-- The amount of water Carla drank in ounces -/
def water : ℝ := 15

/-- The amount of soda Carla drank in ounces -/
def soda : ℝ := 3 * water - 6

/-- The total amount of liquid Carla drank in ounces -/
def total_liquid : ℝ := water + soda

/-- Theorem stating the total amount of liquid Carla drank -/
theorem carla_liquid_consumption : total_liquid = 54 := by
  sorry

end NUMINAMATH_CALUDE_carla_liquid_consumption_l1130_113063


namespace NUMINAMATH_CALUDE_paint_combinations_l1130_113091

theorem paint_combinations (n m k : ℕ) (hn : n = 10) (hm : m = 3) (hk : k = 2) :
  (n.choose m) * k^m = 960 := by
  sorry

end NUMINAMATH_CALUDE_paint_combinations_l1130_113091


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l1130_113065

theorem tangent_line_to_parabola (x y : ℝ) :
  y = x^2 →                                    -- Condition: parabola equation
  (∃ k : ℝ, k * x - y + 4 = 0) →               -- Condition: parallel line exists
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧        -- Tangent line equation
               a / b = 2 ∧                     -- Parallel to given line
               (∃ x₀ y₀ : ℝ, y₀ = x₀^2 ∧       -- Point on parabola
                             a * x₀ + b * y₀ + c = 0 ∧  -- Point on tangent line
                             2 * x₀ = (y₀ - y) / (x₀ - x))) →  -- Derivative condition
  2 * x - y - 1 = 0 :=                         -- Conclusion: specific tangent line equation
by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l1130_113065


namespace NUMINAMATH_CALUDE_root_in_interval_l1130_113083

def f (x : ℝ) := x^3 + x - 1

theorem root_in_interval :
  (f 0.5 < 0) → (f 0.75 > 0) →
  ∃ x₀ ∈ Set.Ioo 0.5 0.75, f x₀ = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l1130_113083


namespace NUMINAMATH_CALUDE_regular_polygon_angle_relation_l1130_113093

theorem regular_polygon_angle_relation (n : ℕ) : n ≥ 3 →
  (120 : ℝ) = 5 * (360 / n) → n = 15 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_relation_l1130_113093


namespace NUMINAMATH_CALUDE_diagonal_path_cubes_3_4_5_l1130_113072

/-- The number of cubes a diagonal path crosses in a cuboid -/
def cubes_crossed (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd a c + Nat.gcd a (Nat.gcd b c)

/-- Theorem: In a 3 × 4 × 5 cuboid, a diagonal path from one corner to the opposite corner
    that doesn't intersect the edges of any small cube inside the cuboid passes through 10 small cubes -/
theorem diagonal_path_cubes_3_4_5 :
  cubes_crossed 3 4 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_path_cubes_3_4_5_l1130_113072


namespace NUMINAMATH_CALUDE_expected_checks_on_4x4_board_l1130_113057

/-- Represents a 4x4 chessboard -/
def Board := Fin 4 × Fin 4

/-- Calculates the number of ways a knight can check a king on a 4x4 board -/
def knight_check_positions (board : Board) : ℕ :=
  match board with
  | (0, 0) | (0, 3) | (3, 0) | (3, 3) => 2  -- corners
  | (0, 1) | (0, 2) | (1, 0) | (1, 3) | (2, 0) | (2, 3) | (3, 1) | (3, 2) => 3  -- edges
  | _ => 4  -- central squares

/-- The total number of possible knight-king pairs -/
def total_pairs : ℕ := 3 * 3

/-- The total number of ways to place a knight and a king on distinct squares -/
def total_placements : ℕ := 16 * 15

/-- The expected number of checks for a single knight-king pair -/
def expected_checks_per_pair : ℚ := 1 / 5

theorem expected_checks_on_4x4_board :
  (total_pairs : ℚ) * expected_checks_per_pair = 9 / 5 := by sorry

#check expected_checks_on_4x4_board

end NUMINAMATH_CALUDE_expected_checks_on_4x4_board_l1130_113057


namespace NUMINAMATH_CALUDE_flower_bed_area_is_35_l1130_113004

/-- The area of a rectangular flower bed -/
def flower_bed_area (width : ℝ) (length : ℝ) : ℝ := width * length

theorem flower_bed_area_is_35 :
  flower_bed_area 5 7 = 35 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_area_is_35_l1130_113004


namespace NUMINAMATH_CALUDE_eight_points_chords_l1130_113040

/-- The number of chords that can be drawn by connecting two points out of n points on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords that can be drawn by connecting two points out of eight points on the circumference of a circle is equal to 28 -/
theorem eight_points_chords : num_chords 8 = 28 := by
  sorry

end NUMINAMATH_CALUDE_eight_points_chords_l1130_113040


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1130_113002

theorem complex_equation_solution :
  ∃ z : ℂ, (4 - 3 * Complex.I * z = 1 + 5 * Complex.I * z) ∧ (z = -3/8 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1130_113002


namespace NUMINAMATH_CALUDE_max_pairs_after_loss_l1130_113099

/-- Represents the number of matching pairs of shoes after losing some shoes. -/
def MaxPairsAfterLoss (totalPairs : ℕ) (colors : ℕ) (sizes : ℕ) (shoesLost : ℕ) : ℕ :=
  min (totalPairs - shoesLost) (colors * sizes)

/-- Theorem stating the maximum number of matching pairs after losing shoes. -/
theorem max_pairs_after_loss :
  MaxPairsAfterLoss 23 6 3 9 = 14 := by
  sorry

#eval MaxPairsAfterLoss 23 6 3 9

end NUMINAMATH_CALUDE_max_pairs_after_loss_l1130_113099


namespace NUMINAMATH_CALUDE_number_of_arrangements_l1130_113027

/-- The number of foreign guests -/
def num_foreign_guests : ℕ := 4

/-- The number of security officers -/
def num_security_officers : ℕ := 2

/-- The total number of individuals -/
def total_individuals : ℕ := num_foreign_guests + num_security_officers

/-- The number of foreign guests that must be together -/
def num_guests_together : ℕ := 2

/-- The function to calculate the number of possible arrangements -/
def calculate_arrangements (n_foreign : ℕ) (n_security : ℕ) (n_together : ℕ) : ℕ :=
  sorry

/-- The theorem stating the number of possible arrangements -/
theorem number_of_arrangements :
  calculate_arrangements num_foreign_guests num_security_officers num_guests_together = 24 :=
sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l1130_113027


namespace NUMINAMATH_CALUDE_factorization_equality_l1130_113077

theorem factorization_equality (a x : ℝ) : a * x^2 - 4 * a * x + 4 * a = a * (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1130_113077


namespace NUMINAMATH_CALUDE_apple_baskets_proof_l1130_113052

/-- Given a total number of apples and apples per basket, calculate the number of full baskets -/
def fullBaskets (totalApples applesPerBasket : ℕ) : ℕ :=
  totalApples / applesPerBasket

theorem apple_baskets_proof :
  fullBaskets 495 25 = 19 := by
  sorry

end NUMINAMATH_CALUDE_apple_baskets_proof_l1130_113052


namespace NUMINAMATH_CALUDE_cradle_cup_d_score_l1130_113021

/-- Represents the scores of the five participants in the "Cradle Cup" math competition. -/
structure CradleCupScores where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ

/-- The conditions of the "Cradle Cup" math competition. -/
def CradleCupConditions (scores : CradleCupScores) : Prop :=
  scores.A = 94 ∧
  scores.B ≥ scores.A ∧ scores.B ≥ scores.C ∧ scores.B ≥ scores.D ∧ scores.B ≥ scores.E ∧
  scores.C = (scores.A + scores.D) / 2 ∧
  5 * scores.D = scores.A + scores.B + scores.C + scores.D + scores.E ∧
  scores.E = scores.C + 2 ∧
  scores.B ≤ 100 ∧ scores.C ≤ 100 ∧ scores.D ≤ 100 ∧ scores.E ≤ 100

/-- The theorem stating that given the conditions of the "Cradle Cup" math competition,
    participant D must have scored 96 points. -/
theorem cradle_cup_d_score (scores : CradleCupScores) :
  CradleCupConditions scores → scores.D = 96 :=
by sorry

end NUMINAMATH_CALUDE_cradle_cup_d_score_l1130_113021


namespace NUMINAMATH_CALUDE_volunteer_distribution_theorem_l1130_113015

/-- The number of ways to distribute n people into two activities with capacity constraints -/
def distributeVolunteers (n : ℕ) (maxPerActivity : ℕ) : ℕ :=
  -- We don't implement the function here, just declare it
  sorry

/-- Theorem: The number of ways to distribute 6 people into two activities,
    where each activity can accommodate no more than 4 people, is equal to 50 -/
theorem volunteer_distribution_theorem :
  distributeVolunteers 6 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_distribution_theorem_l1130_113015


namespace NUMINAMATH_CALUDE_total_is_100_l1130_113060

/-- Represents the shares of money for three individuals -/
structure Shares :=
  (a : ℚ)
  (b : ℚ)
  (c : ℚ)

/-- The conditions of the problem -/
def SatisfiesConditions (s : Shares) : Prop :=
  s.a = (1 / 4) * (s.b + s.c) ∧
  s.b = (3 / 5) * (s.a + s.c) ∧
  s.a = 20

/-- The theorem stating that the total amount is 100 -/
theorem total_is_100 (s : Shares) (h : SatisfiesConditions s) :
  s.a + s.b + s.c = 100 := by
  sorry

end NUMINAMATH_CALUDE_total_is_100_l1130_113060


namespace NUMINAMATH_CALUDE_product_sum_theorem_l1130_113029

theorem product_sum_theorem (x y : ℤ) : 
  y = x + 2 → x * y = 20400 → x + y = 286 := by
sorry

end NUMINAMATH_CALUDE_product_sum_theorem_l1130_113029


namespace NUMINAMATH_CALUDE_response_rate_increase_l1130_113096

/-- Calculate the percentage increase in response rate between two surveys -/
theorem response_rate_increase (customers1 customers2 respondents1 respondents2 : ℕ) :
  customers1 = 80 →
  customers2 = 63 →
  respondents1 = 7 →
  respondents2 = 9 →
  let rate1 := (respondents1 : ℝ) / customers1
  let rate2 := (respondents2 : ℝ) / customers2
  let increase := (rate2 - rate1) / rate1 * 100
  ∃ ε > 0, |increase - 63.24| < ε :=
by sorry

end NUMINAMATH_CALUDE_response_rate_increase_l1130_113096


namespace NUMINAMATH_CALUDE_sqrt_32_minus_cos_45_plus_one_minus_sqrt_2_squared_l1130_113037

theorem sqrt_32_minus_cos_45_plus_one_minus_sqrt_2_squared :
  Real.sqrt 32 - Real.cos (π / 4) + (1 - Real.sqrt 2) ^ 2 = 3 + (3 / 2) * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_32_minus_cos_45_plus_one_minus_sqrt_2_squared_l1130_113037


namespace NUMINAMATH_CALUDE_video_upload_total_l1130_113022

theorem video_upload_total (days_in_month : ℕ) (initial_daily_upload : ℕ) : 
  days_in_month = 30 →
  initial_daily_upload = 10 →
  (days_in_month / 2 * initial_daily_upload) + 
  (days_in_month / 2 * (2 * initial_daily_upload)) = 450 := by
sorry

end NUMINAMATH_CALUDE_video_upload_total_l1130_113022


namespace NUMINAMATH_CALUDE_geometric_sum_remainder_l1130_113056

theorem geometric_sum_remainder (n : ℕ) : 
  (((5^(n+1) - 1) / 4) % 500 = 31) ∧ (n = 1002) := by sorry

end NUMINAMATH_CALUDE_geometric_sum_remainder_l1130_113056


namespace NUMINAMATH_CALUDE_fraction_stayed_home_l1130_113019

theorem fraction_stayed_home (total : ℚ) (fun_fraction : ℚ) (youth_fraction : ℚ)
  (h1 : fun_fraction = 5 / 13)
  (h2 : youth_fraction = 4 / 13)
  (h3 : total = 1) :
  total - (fun_fraction + youth_fraction) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_fraction_stayed_home_l1130_113019


namespace NUMINAMATH_CALUDE_smoking_hospitalization_percentage_l1130_113079

theorem smoking_hospitalization_percentage 
  (total_students : ℕ) 
  (smoking_percentage : ℚ) 
  (non_hospitalized : ℕ) 
  (h1 : total_students = 300)
  (h2 : smoking_percentage = 2/5)
  (h3 : non_hospitalized = 36) :
  (total_students * smoking_percentage - non_hospitalized) / (total_students * smoking_percentage) = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_smoking_hospitalization_percentage_l1130_113079


namespace NUMINAMATH_CALUDE_max_triples_count_l1130_113064

def N (n : ℕ) : ℕ := sorry

theorem max_triples_count (n : ℕ) (h : n ≥ 2) :
  N n = ⌊(2 * n : ℚ) / 3 + 1⌋ :=
by sorry

end NUMINAMATH_CALUDE_max_triples_count_l1130_113064


namespace NUMINAMATH_CALUDE_number_of_digits_in_x_l1130_113068

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the problem statement
theorem number_of_digits_in_x (x y : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (x_gt_y : x > y)
  (prod_xy : x * y = 490)
  (log_eq : (log10 x - log10 7) * (log10 y - log10 7) = -143/4) :
  ∃ n : ℕ, n = 8 ∧ 10^(n-1) ≤ x ∧ x < 10^n := by
sorry

end NUMINAMATH_CALUDE_number_of_digits_in_x_l1130_113068


namespace NUMINAMATH_CALUDE_a1_plus_a3_equals_24_l1130_113069

theorem a1_plus_a3_equals_24 (x : ℝ) (a₀ a₁ a₂ a₃ a₄ : ℝ) 
  (h : (1 - 2/x)^4 = a₀ + a₁*(1/x) + a₂*(1/x)^2 + a₃*(1/x)^3 + a₄*(1/x)^4) :
  a₁ + a₃ = 24 := by
sorry

end NUMINAMATH_CALUDE_a1_plus_a3_equals_24_l1130_113069


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l1130_113048

theorem log_sum_equals_two (a : ℝ) (h : 1 + a^3 = 9) : 
  Real.log a / Real.log (1/4) + Real.log 8 / Real.log a = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l1130_113048


namespace NUMINAMATH_CALUDE_plums_picked_equals_127_l1130_113074

/-- Calculates the total number of plums picked by Alyssa and Jason after three hours -/
def total_plums_picked (alyssa_rate : ℕ) (jason_rate : ℕ) : ℕ :=
  let first_hour := alyssa_rate + jason_rate
  let second_hour := (3 * alyssa_rate) + (jason_rate + (2 * jason_rate / 5))
  let third_hour_before_drop := alyssa_rate + (2 * jason_rate)
  let third_hour_after_drop := third_hour_before_drop - (third_hour_before_drop / 14)
  first_hour + second_hour + third_hour_after_drop

/-- Theorem stating that the total number of plums picked is 127 -/
theorem plums_picked_equals_127 :
  total_plums_picked 17 10 = 127 := by
  sorry

#eval total_plums_picked 17 10

end NUMINAMATH_CALUDE_plums_picked_equals_127_l1130_113074


namespace NUMINAMATH_CALUDE_rhombus_parallel_sides_distance_l1130_113071

/-- The distance between parallel sides of a rhombus given its diagonals -/
theorem rhombus_parallel_sides_distance (AC BD : ℝ) (h1 : AC = 3) (h2 : BD = 4) :
  let area := (1 / 2) * AC * BD
  let side := Real.sqrt ((AC / 2)^2 + (BD / 2)^2)
  area / side = 12 / 5 := by sorry

end NUMINAMATH_CALUDE_rhombus_parallel_sides_distance_l1130_113071


namespace NUMINAMATH_CALUDE_equation_solution_l1130_113005

theorem equation_solution : 
  ∀ x : ℝ, x ≠ 5 → (x + 36 / (x - 5) = -9 ↔ x = -9 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1130_113005


namespace NUMINAMATH_CALUDE_imaginary_part_of_1_minus_2i_l1130_113067

theorem imaginary_part_of_1_minus_2i :
  Complex.im (1 - 2 * Complex.I) = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_1_minus_2i_l1130_113067


namespace NUMINAMATH_CALUDE_inlet_pipe_rate_l1130_113080

/-- Given a tank with the following properties:
  * Capacity of 6048 liters
  * Empties in 7 hours due to a leak
  * Empties in 12 hours when both the leak and an inlet pipe are open
  Prove that the rate at which the inlet pipe fills water is 360 liters per hour -/
theorem inlet_pipe_rate (tank_capacity : ℝ) (leak_empty_time : ℝ) (both_empty_time : ℝ)
  (h1 : tank_capacity = 6048)
  (h2 : leak_empty_time = 7)
  (h3 : both_empty_time = 12) :
  let leak_rate := tank_capacity / leak_empty_time
  let net_empty_rate := tank_capacity / both_empty_time
  leak_rate - (leak_rate - net_empty_rate) = 360 := by
sorry


end NUMINAMATH_CALUDE_inlet_pipe_rate_l1130_113080


namespace NUMINAMATH_CALUDE_B_power_200_l1130_113050

def B : Matrix (Fin 4) (Fin 4) ℤ :=
  !![0,0,0,1;
     1,0,0,0;
     0,1,0,0;
     0,0,1,0]

theorem B_power_200 : B ^ 200 = 1 := by sorry

end NUMINAMATH_CALUDE_B_power_200_l1130_113050


namespace NUMINAMATH_CALUDE_least_integer_square_98_more_than_double_l1130_113094

theorem least_integer_square_98_more_than_double : 
  ∃ x : ℤ, x^2 = 2*x + 98 ∧ ∀ y : ℤ, y^2 = 2*y + 98 → x ≤ y :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_integer_square_98_more_than_double_l1130_113094


namespace NUMINAMATH_CALUDE_product_in_base7_l1130_113085

/-- Converts a base-7 number to base-10 --/
def toBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a base-10 number to base-7 --/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else go (m / 7) ((m % 7) :: acc)
    go n []

/-- The product of 325₇ and 6₇ in base 7 is 2624₇ --/
theorem product_in_base7 :
  toBase7 (toBase10 [5, 2, 3] * toBase10 [6]) = [4, 2, 6, 2] := by
  sorry

end NUMINAMATH_CALUDE_product_in_base7_l1130_113085


namespace NUMINAMATH_CALUDE_repeating_decimal_denominator_l1130_113073

theorem repeating_decimal_denominator : ∃ (n : ℕ) (d : ℕ), d ≠ 0 ∧ (n / d : ℚ) = 0.6666666666666667 ∧ (∀ (n' : ℕ) (d' : ℕ), d' ≠ 0 → (n' / d' : ℚ) = (n / d : ℚ) → d' ≥ d) ∧ d = 3 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_denominator_l1130_113073


namespace NUMINAMATH_CALUDE_new_person_weight_l1130_113009

/-- Proves that if replacing a 65 kg person in a group of 4 people
    increases the average weight by 1.5 kg, then the weight of the new person is 71 kg. -/
theorem new_person_weight (initial_total : ℝ) :
  (initial_total - 65 + new_weight) / 4 = initial_total / 4 + 1.5 →
  new_weight = 71 :=
by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1130_113009


namespace NUMINAMATH_CALUDE_circle_symmetry_l1130_113078

-- Define the original circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y = 2

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 1

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y x' y' : ℝ),
    circle_C x y →
    line_l ((x + x') / 2) ((y + y') / 2) →
    symmetric_circle x' y' :=
by sorry

end NUMINAMATH_CALUDE_circle_symmetry_l1130_113078


namespace NUMINAMATH_CALUDE_quadratic_inequality_properties_l1130_113043

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h_solution_set : ∀ x, f a b c x < 0 ↔ -1 < x ∧ x < 2) :
  (∀ x, b * x + c > 0 ↔ x < -2) ∧
  (4 * a - 2 * b + c > 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_properties_l1130_113043


namespace NUMINAMATH_CALUDE_inequality_proof_l1130_113051

theorem inequality_proof (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_one : a + b + c = 1) : 
  a * (1 + b - c)^(1/3 : ℝ) + b * (1 + c - a)^(1/3 : ℝ) + c * (1 + a - b)^(1/3 : ℝ) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1130_113051


namespace NUMINAMATH_CALUDE_max_min_difference_c_l1130_113023

theorem max_min_difference_c (a b c : ℝ) 
  (sum_eq : a + b + c = 3) 
  (sum_squares_eq : a^2 + b^2 + c^2 = 20) : 
  ∃ (c_max c_min : ℝ), 
    (∀ x : ℝ, (∃ y z : ℝ, y + z + x = 3 ∧ y^2 + z^2 + x^2 = 20) → x ≤ c_max ∧ x ≥ c_min) ∧ 
    c_max - c_min = 2 * Real.sqrt 34 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_c_l1130_113023


namespace NUMINAMATH_CALUDE_tank_dimension_l1130_113020

/-- Given a rectangular tank with dimensions 3 feet, 7 feet, and x feet,
    if the total surface area is 82 square feet, then x = 2 feet. -/
theorem tank_dimension (x : ℝ) : 
  2 * (3 * 7 + 3 * x + 7 * x) = 82 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_tank_dimension_l1130_113020


namespace NUMINAMATH_CALUDE_paper_size_problem_l1130_113088

theorem paper_size_problem (L : ℝ) :
  (L > 0) →
  (2 * (L * 11) = 2 * (5.5 * 11) + 100) →
  L = 10 := by
sorry

end NUMINAMATH_CALUDE_paper_size_problem_l1130_113088
