import Mathlib

namespace NUMINAMATH_CALUDE_divisibility_by_35_l3941_394118

theorem divisibility_by_35 : 
  {a : ℕ | 1 ≤ a ∧ a ≤ 105 ∧ 35 ∣ (a^3 - 1)} = 
  {1, 11, 16, 36, 46, 51, 71, 81, 86} := by sorry

end NUMINAMATH_CALUDE_divisibility_by_35_l3941_394118


namespace NUMINAMATH_CALUDE_min_value_expression_l3941_394171

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2 * y = 1 → a^2 + 4 * b^2 + 1 / (a * b) ≤ x^2 + 4 * y^2 + 1 / (x * y)) ∧
  a^2 + 4 * b^2 + 1 / (a * b) = 17 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3941_394171


namespace NUMINAMATH_CALUDE_largest_x_value_l3941_394189

theorem largest_x_value (x : ℝ) : 
  (16 * x^2 - 40 * x + 15) / (4 * x - 3) + 7 * x = 8 * x - 2 →
  x ≤ 9/4 ∧ ∃ y, y > 9/4 → (16 * y^2 - 40 * y + 15) / (4 * y - 3) + 7 * y ≠ 8 * y - 2 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_value_l3941_394189


namespace NUMINAMATH_CALUDE_cost_per_crayon_is_two_l3941_394176

/-- The number of crayons in half a dozen -/
def half_dozen : ℕ := 6

/-- The number of half dozens Jamal bought -/
def number_of_half_dozens : ℕ := 4

/-- The total cost of the crayons in dollars -/
def total_cost : ℕ := 48

/-- The total number of crayons Jamal bought -/
def total_crayons : ℕ := number_of_half_dozens * half_dozen

/-- The cost per crayon in dollars -/
def cost_per_crayon : ℚ := total_cost / total_crayons

theorem cost_per_crayon_is_two : cost_per_crayon = 2 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_crayon_is_two_l3941_394176


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3941_394186

theorem complex_equation_solution (z : ℂ) 
  (h : 20 * Complex.abs z ^ 2 = 3 * Complex.abs (z + 3) ^ 2 + Complex.abs (z ^ 2 + 2) ^ 2 + 37) :
  z + 9 / z = -3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3941_394186


namespace NUMINAMATH_CALUDE_power_of_two_problem_l3941_394166

theorem power_of_two_problem (a b : ℕ+) 
  (h1 : (2 ^ a.val) ^ b.val = 2 ^ 2)
  (h2 : 2 ^ a.val * 2 ^ b.val = 8) :
  2 ^ b.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_problem_l3941_394166


namespace NUMINAMATH_CALUDE_quadratic_radical_sum_l3941_394193

theorem quadratic_radical_sum (m n : ℕ) : 
  (∃ k : ℕ, (m - 1 : ℕ) = 2 ∧ 7^k = 7) ∧ 
  (∃ l : ℕ, 4*n - 1 = 7^l) ∧
  (m - 1 : ℕ) = 2 → 
  m + n = 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_radical_sum_l3941_394193


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_l3941_394197

def ribbon_lengths : List ℕ := [8, 16, 20, 28]

theorem greatest_prime_divisor (lengths : List ℕ) : 
  ∃ (n : ℕ), n.Prime ∧ 
  (∀ m : ℕ, m.Prime → (∀ l ∈ lengths, l % m = 0) → m ≤ n) ∧
  (∀ l ∈ lengths, l % n = 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_l3941_394197


namespace NUMINAMATH_CALUDE_rabbit_weight_l3941_394153

theorem rabbit_weight (k r p : ℝ) 
  (total_weight : k + r + p = 39)
  (rabbit_parrot_weight : r + p = 3 * k)
  (rabbit_kitten_weight : r + k = 1.5 * p) :
  r = 13.65 := by
sorry

end NUMINAMATH_CALUDE_rabbit_weight_l3941_394153


namespace NUMINAMATH_CALUDE_last_stage_less_than_2014_l3941_394195

theorem last_stage_less_than_2014 :
  ∀ k : ℕ, k > 0 → (2 * k^2 - 2 * k + 1 < 2014) ↔ k ≤ 32 :=
by sorry

end NUMINAMATH_CALUDE_last_stage_less_than_2014_l3941_394195


namespace NUMINAMATH_CALUDE_implication_disjunction_equivalence_l3941_394133

theorem implication_disjunction_equivalence (A B : Prop) : (A → B) ↔ (¬A ∨ B) := by
  sorry

end NUMINAMATH_CALUDE_implication_disjunction_equivalence_l3941_394133


namespace NUMINAMATH_CALUDE_prime_triple_equation_l3941_394181

theorem prime_triple_equation (p q n : ℕ) : 
  p.Prime → q.Prime → p > 0 → q > 0 → n > 0 →
  p * (p + 1) + q * (q + 1) = n * (n + 1) →
  ((p = 5 ∧ q = 3 ∧ n = 6) ∨ (p = 3 ∧ q = 5 ∧ n = 6)) :=
by sorry

end NUMINAMATH_CALUDE_prime_triple_equation_l3941_394181


namespace NUMINAMATH_CALUDE_particle_movement_probability_l3941_394105

/-- The probability of a particle reaching (n,n) from (0,0) in exactly 2n+k tosses -/
def particle_probability (n k : ℕ) : ℚ :=
  (Nat.choose (2*n + k - 1) (n - 1) : ℚ) * (1 / 2 ^ (2*n + k - 1))

/-- Theorem stating the probability of the particle reaching (n,n) in 2n+k tosses -/
theorem particle_movement_probability (n k : ℕ) (h1 : n > 0) (h2 : k > 0) :
  particle_probability n k = (Nat.choose (2*n + k - 1) (n - 1) : ℚ) * (1 / 2 ^ (2*n + k - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_particle_movement_probability_l3941_394105


namespace NUMINAMATH_CALUDE_average_speed_calculation_l3941_394177

-- Define the distance traveled in the first and second hours
def distance_first_hour : ℝ := 98
def distance_second_hour : ℝ := 70

-- Define the total time
def total_time : ℝ := 2

-- Theorem statement
theorem average_speed_calculation :
  let total_distance := distance_first_hour + distance_second_hour
  (total_distance / total_time) = 84 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l3941_394177


namespace NUMINAMATH_CALUDE_top_tier_lamps_l3941_394143

/-- Represents the number of tiers in the tower -/
def n : ℕ := 7

/-- Represents the common ratio of the geometric sequence -/
def r : ℕ := 2

/-- Represents the total number of lamps in the tower -/
def total_lamps : ℕ := 381

/-- Calculates the sum of a geometric series -/
def geometric_sum (a₁ : ℕ) : ℕ := a₁ * (1 - r^n) / (1 - r)

/-- Theorem stating that the number of lamps on the top tier is 3 -/
theorem top_tier_lamps : ∃ (a₁ : ℕ), geometric_sum a₁ = total_lamps ∧ a₁ = 3 := by
  sorry

end NUMINAMATH_CALUDE_top_tier_lamps_l3941_394143


namespace NUMINAMATH_CALUDE_problem_statement_l3941_394111

theorem problem_statement (a b c d e : ℝ) 
  (h1 : (a + c) * (a + d) = e)
  (h2 : (b + c) * (b + d) = e)
  (h3 : e ≠ 0)
  (h4 : a ≠ b) :
  (a + c) * (b + c) - (a + d) * (b + d) = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3941_394111


namespace NUMINAMATH_CALUDE_equations_solution_l3941_394139

def satisfies_equations (a b c d : ℕ) : Prop :=
  a + b = c * d ∧ c + d = a * b

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(2,2,2,2), (1,2,3,5), (2,1,3,5), (1,2,5,3), (2,1,5,3), (3,5,1,2), (5,3,1,2), (3,5,2,1), (5,3,2,1)}

theorem equations_solution :
  ∀ (a b c d : ℕ), satisfies_equations a b c d ↔ (a, b, c, d) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equations_solution_l3941_394139


namespace NUMINAMATH_CALUDE_point_C_coordinates_l3941_394191

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angle bisector line
def AngleBisector : ℝ → ℝ := fun x ↦ 2 * x

-- Define the condition that y=2x is the angle bisector of ∠C
def IsAngleBisector (t : Triangle) : Prop :=
  AngleBisector (t.C.1) = t.C.2

theorem point_C_coordinates (t : Triangle) :
  t.A = (-4, 2) →
  t.B = (3, 1) →
  IsAngleBisector t →
  t.C = (2, 4) := by
  sorry


end NUMINAMATH_CALUDE_point_C_coordinates_l3941_394191


namespace NUMINAMATH_CALUDE_equation_solution_l3941_394160

theorem equation_solution : ∃ x : ℝ, 45 - (28 - (37 - (x - 20))) = 59 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3941_394160


namespace NUMINAMATH_CALUDE_total_employees_l3941_394180

theorem total_employees (part_time full_time : ℕ) 
  (h1 : part_time = 2041) 
  (h2 : full_time = 63093) : 
  part_time + full_time = 65134 := by
  sorry

end NUMINAMATH_CALUDE_total_employees_l3941_394180


namespace NUMINAMATH_CALUDE_perp_para_implies_perp_line_perp_para_planes_implies_perp_perp_two_planes_implies_para_l3941_394109

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Line → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (paraPlane : Plane → Plane → Prop)

-- Axioms for the relations
axiom perp_antisymm {l : Line} {p : Plane} : perp l p ↔ perp l p
axiom para_antisymm {l : Line} {p : Plane} : para l p ↔ para l p
axiom perpLine_antisymm {l1 l2 : Line} : perpLine l1 l2 ↔ perpLine l2 l1
axiom paraPlane_antisymm {p1 p2 : Plane} : paraPlane p1 p2 ↔ paraPlane p2 p1

-- Theorem 1
theorem perp_para_implies_perp_line {m n : Line} {α : Plane} 
  (h1 : perp m α) (h2 : para n α) : perpLine m n := by sorry

-- Theorem 2
theorem perp_para_planes_implies_perp {m : Line} {α β : Plane}
  (h1 : perp m α) (h2 : paraPlane α β) : perp m β := by sorry

-- Theorem 3
theorem perp_two_planes_implies_para {m : Line} {α β : Plane}
  (h1 : perp m α) (h2 : perp m β) : paraPlane α β := by sorry

end NUMINAMATH_CALUDE_perp_para_implies_perp_line_perp_para_planes_implies_perp_perp_two_planes_implies_para_l3941_394109


namespace NUMINAMATH_CALUDE_timothy_total_cost_l3941_394196

/-- The total cost of Timothy's purchases -/
def total_cost (land_acres : ℕ) (land_price_per_acre : ℕ) 
               (house_price : ℕ) 
               (cow_count : ℕ) (cow_price : ℕ) 
               (chicken_count : ℕ) (chicken_price : ℕ) 
               (solar_install_hours : ℕ) (solar_install_price_per_hour : ℕ) 
               (solar_equipment_price : ℕ) : ℕ :=
  land_acres * land_price_per_acre + 
  house_price + 
  cow_count * cow_price + 
  chicken_count * chicken_price + 
  solar_install_hours * solar_install_price_per_hour + 
  solar_equipment_price

/-- Theorem stating that Timothy's total cost is $147,700 -/
theorem timothy_total_cost : 
  total_cost 30 20 120000 20 1000 100 5 6 100 6000 = 147700 := by
  sorry

end NUMINAMATH_CALUDE_timothy_total_cost_l3941_394196


namespace NUMINAMATH_CALUDE_solve_equation_l3941_394187

theorem solve_equation : ∃ x : ℝ, x + Real.sqrt (-4 + 6 * 4 / 3) = 13 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3941_394187


namespace NUMINAMATH_CALUDE_min_value_of_difference_l3941_394178

theorem min_value_of_difference (x y z : ℝ) : 
  0 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ z ≤ 4 →
  y^2 - x^2 = 2 →
  z^2 - y^2 = 2 →
  (2 * (2 - Real.sqrt 3) : ℝ) ≤ |x - y| + |y - z| ∧ |x - y| + |y - z| ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_difference_l3941_394178


namespace NUMINAMATH_CALUDE_cube_root_of_2_plus_11i_l3941_394175

def complex_cube_root (z : ℂ) : Prop :=
  z^3 = (2 : ℂ) + Complex.I * 11

theorem cube_root_of_2_plus_11i :
  complex_cube_root (2 + Complex.I) ∧
  ∃ (z₁ z₂ : ℂ), 
    complex_cube_root z₁ ∧
    complex_cube_root z₂ ∧
    z₁ ≠ z₂ ∧
    z₁ ≠ (2 + Complex.I) ∧
    z₂ ≠ (2 + Complex.I) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_2_plus_11i_l3941_394175


namespace NUMINAMATH_CALUDE_solution_quadratic_equation_l3941_394112

theorem solution_quadratic_equation :
  ∀ x : ℝ, (x - 2)^2 = 3*(x - 2) ↔ x = 2 ∨ x = 5 := by sorry

end NUMINAMATH_CALUDE_solution_quadratic_equation_l3941_394112


namespace NUMINAMATH_CALUDE_inequality_proof_l3941_394157

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x * (x - z)^2 + y * (y - z)^2 ≥ (x - z) * (y - z) * (x + y - z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3941_394157


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l3941_394150

/-- A geometric sequence with third term 5 and sixth term 40 has first term 5/4 -/
theorem geometric_sequence_first_term (a : ℝ) (r : ℝ) : 
  a * r^2 = 5 → a * r^5 = 40 → a = 5/4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l3941_394150


namespace NUMINAMATH_CALUDE_henry_shells_l3941_394194

theorem henry_shells (broken_shells : ℕ) (perfect_non_spiral : ℕ) (spiral_difference : ℕ) :
  broken_shells = 52 →
  perfect_non_spiral = 12 →
  spiral_difference = 21 →
  ∃ (total_perfect : ℕ),
    total_perfect = (broken_shells / 2 - spiral_difference) + perfect_non_spiral ∧
    total_perfect = 17 := by
  sorry

end NUMINAMATH_CALUDE_henry_shells_l3941_394194


namespace NUMINAMATH_CALUDE_darnel_running_results_l3941_394192

/-- Represents Darnel's running activities --/
structure RunningActivity where
  sprint1 : Real
  sprint2 : Real
  jog1 : Real
  jog2 : Real
  walk : Real

/-- Calculates the total distance covered in all activities --/
def totalDistance (activity : RunningActivity) : Real :=
  activity.sprint1 + activity.sprint2 + activity.jog1 + activity.jog2 + activity.walk

/-- Calculates the additional distance sprinted compared to jogging and walking --/
def additionalSprint (activity : RunningActivity) : Real :=
  (activity.sprint1 + activity.sprint2) - (activity.jog1 + activity.jog2 + activity.walk)

/-- Theorem stating the total distance and additional sprint for Darnel's activities --/
theorem darnel_running_results (activity : RunningActivity)
  (h1 : activity.sprint1 = 0.88)
  (h2 : activity.sprint2 = 1.12)
  (h3 : activity.jog1 = 0.75)
  (h4 : activity.jog2 = 0.45)
  (h5 : activity.walk = 0.32) :
  totalDistance activity = 3.52 ∧ additionalSprint activity = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_darnel_running_results_l3941_394192


namespace NUMINAMATH_CALUDE_find_C_l3941_394131

theorem find_C (A B C : ℤ) (h1 : A = 509) (h2 : A = B + 197) (h3 : C = B - 125) : C = 187 := by
  sorry

end NUMINAMATH_CALUDE_find_C_l3941_394131


namespace NUMINAMATH_CALUDE_total_tax_percentage_l3941_394140

/-- Calculates the total tax percentage given spending percentages and tax rates -/
theorem total_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.2)
  (h3 : other_percent = 0.3)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.05)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.1) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.055 := by
sorry


end NUMINAMATH_CALUDE_total_tax_percentage_l3941_394140


namespace NUMINAMATH_CALUDE_total_spent_l3941_394185

def lunch_cost : ℝ := 60.50
def tip_percentage : ℝ := 20

theorem total_spent (lunch_cost : ℝ) (tip_percentage : ℝ) : 
  lunch_cost * (1 + tip_percentage / 100) = 72.60 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_l3941_394185


namespace NUMINAMATH_CALUDE_min_value_z_l3941_394163

theorem min_value_z (x y : ℝ) (h1 : x - y + 1 ≥ 0) (h2 : x + y - 1 ≥ 0) (h3 : x ≤ 3) :
  ∃ (z : ℝ), z = 2*x - 3*y ∧ z ≥ -6 ∧ (∀ (x' y' : ℝ), x' - y' + 1 ≥ 0 → x' + y' - 1 ≥ 0 → x' ≤ 3 → 2*x' - 3*y' ≥ z) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_z_l3941_394163


namespace NUMINAMATH_CALUDE_intersection_of_lines_l3941_394173

theorem intersection_of_lines :
  ∃! (x y : ℚ), (6 * x - 5 * y = 10) ∧ (8 * x + 2 * y = 20) ∧ (x = 30 / 13) ∧ (y = 10 / 13) := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_lines_l3941_394173


namespace NUMINAMATH_CALUDE_knitting_productivity_ratio_l3941_394119

/-- Represents the knitting productivity of a girl -/
structure Knitter where
  work_time : ℕ  -- Time spent working before a break
  break_time : ℕ -- Duration of the break

/-- Calculates the total cycle time for a knitter -/
def cycle_time (k : Knitter) : ℕ := k.work_time + k.break_time

/-- Calculates the actual working time within a given period -/
def working_time (k : Knitter) (period : ℕ) : ℕ :=
  (period / cycle_time k) * k.work_time

theorem knitting_productivity_ratio :
  let girl1 : Knitter := ⟨5, 1⟩
  let girl2 : Knitter := ⟨7, 1⟩
  let common_period := Nat.lcm (cycle_time girl1) (cycle_time girl2)
  (working_time girl2 common_period : ℚ) / (working_time girl1 common_period) = 20/21 :=
sorry

end NUMINAMATH_CALUDE_knitting_productivity_ratio_l3941_394119


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3941_394174

open Set

def M : Set ℝ := {x : ℝ | -1/2 < x ∧ x < 1/2}
def N : Set ℝ := {x : ℝ | x * (x - 1) ≤ 0}

theorem union_of_M_and_N : M ∪ N = Ioo (-1/2 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3941_394174


namespace NUMINAMATH_CALUDE_f_max_value_when_a_2_f_no_min_value_when_a_2_f_decreasing_when_a_leq_neg_quarter_f_decreasing_when_neg_quarter_lt_a_leq_zero_f_monotonicity_when_a_gt_zero_l3941_394106

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x - (1/2) * x^2

-- Theorem for the maximum value when a = 2
theorem f_max_value_when_a_2 :
  ∃ (x : ℝ), x > 0 ∧ f 2 x = -(3/2) ∧ ∀ (y : ℝ), y > 0 → f 2 y ≤ f 2 x :=
sorry

-- Theorem for no minimum value when a = 2
theorem f_no_min_value_when_a_2 :
  ∀ (M : ℝ), ∃ (x : ℝ), x > 0 ∧ f 2 x < M :=
sorry

-- Theorem for monotonicity when a ≤ -1/4
theorem f_decreasing_when_a_leq_neg_quarter (a : ℝ) (h : a ≤ -(1/4)) :
  ∀ (x y : ℝ), 0 < x → 0 < y → x < y → f a x > f a y :=
sorry

-- Theorem for monotonicity when -1/4 < a ≤ 0
theorem f_decreasing_when_neg_quarter_lt_a_leq_zero (a : ℝ) (h1 : -(1/4) < a) (h2 : a ≤ 0) :
  ∀ (x y : ℝ), 0 < x → 0 < y → x < y → f a x > f a y :=
sorry

-- Theorem for monotonicity when a > 0
theorem f_monotonicity_when_a_gt_zero (a : ℝ) (h : a > 0) :
  let x0 := (-1 + Real.sqrt (1 + 4*a)) / 2
  ∀ (x y : ℝ), 0 < x → x < y → y < x0 → f a x < f a y ∧
  ∀ (x y : ℝ), x0 < x → x < y → f a x > f a y :=
sorry

end

end NUMINAMATH_CALUDE_f_max_value_when_a_2_f_no_min_value_when_a_2_f_decreasing_when_a_leq_neg_quarter_f_decreasing_when_neg_quarter_lt_a_leq_zero_f_monotonicity_when_a_gt_zero_l3941_394106


namespace NUMINAMATH_CALUDE_jogger_distance_ahead_l3941_394149

/-- Calculates the distance a jogger is ahead of a train given their speeds and the time it takes for the train to pass the jogger. -/
theorem jogger_distance_ahead (jogger_speed train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  passing_time = 38 →
  (train_speed - jogger_speed) * passing_time - train_length = 260 :=
by sorry

end NUMINAMATH_CALUDE_jogger_distance_ahead_l3941_394149


namespace NUMINAMATH_CALUDE_max_area_right_triangle_pen_l3941_394134

/-- The maximum area of a right triangular pen with perimeter 60 feet is 450 square feet. -/
theorem max_area_right_triangle_pen (x y : ℝ) : 
  x > 0 → y > 0 → x + y + Real.sqrt (x^2 + y^2) = 60 → 
  (1/2) * x * y ≤ 450 := by
sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_pen_l3941_394134


namespace NUMINAMATH_CALUDE_ellipse_equation_l3941_394103

/-- Given a circle and an ellipse with specific properties, prove the equation of the ellipse -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (A B : ℝ × ℝ),
    -- Point (1, 1/2) is on the circle x^2 + y^2 = 1
    1^2 + (1/2)^2 = 1 ∧
    -- A and B are points on the circle
    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧
    -- Line AB passes through (1, 0) (focus) and (0, 2) (upper vertex)
    ∃ (m c : ℝ), (m * 1 + c = 0) ∧ (m * 0 + c = 2) ∧
    (m * A.1 + c = A.2) ∧ (m * B.1 + c = B.2) ∧
    -- Ellipse equation
    ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) →
  a^2 = 5 ∧ b^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3941_394103


namespace NUMINAMATH_CALUDE_negation_of_existential_l3941_394129

theorem negation_of_existential (p : Prop) :
  (¬∃ (x : ℝ), x^2 + 2*x = 3) ↔ (∀ (x : ℝ), x^2 + 2*x ≠ 3) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_l3941_394129


namespace NUMINAMATH_CALUDE_range_of_a_l3941_394138

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → 
  -1 ≤ a ∧ a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3941_394138


namespace NUMINAMATH_CALUDE_implicit_derivative_l3941_394147

-- Define the implicit function
def implicit_function (x y : ℝ) : Prop := x^2 - y^2 = 4

-- State the theorem
theorem implicit_derivative (x y : ℝ) (h : implicit_function x y) :
  ∃ (y' : ℝ), y' = x / y :=
sorry

end NUMINAMATH_CALUDE_implicit_derivative_l3941_394147


namespace NUMINAMATH_CALUDE_tea_mixture_ratio_l3941_394167

theorem tea_mixture_ratio (price_tea1 price_tea2 price_mixture : ℚ) 
  (h1 : price_tea1 = 62)
  (h2 : price_tea2 = 72)
  (h3 : price_mixture = 64.5) :
  ∃ (x y : ℚ), x > 0 ∧ y > 0 ∧ x / y = 3 ∧
  (x * price_tea1 + y * price_tea2) / (x + y) = price_mixture :=
by sorry

end NUMINAMATH_CALUDE_tea_mixture_ratio_l3941_394167


namespace NUMINAMATH_CALUDE_ratio_problem_l3941_394102

theorem ratio_problem (a b : ℝ) (h : (9*a - 4*b) / (12*a - 3*b) = 4/7) : 
  a / b = 16 / 15 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l3941_394102


namespace NUMINAMATH_CALUDE_sara_sold_oranges_l3941_394116

/-- Represents the number of oranges Joan picked initially -/
def initial_oranges : ℕ := 37

/-- Represents the number of oranges Joan is left with -/
def remaining_oranges : ℕ := 27

/-- Represents the number of oranges Sara sold -/
def sold_oranges : ℕ := initial_oranges - remaining_oranges

theorem sara_sold_oranges : sold_oranges = 10 := by
  sorry

end NUMINAMATH_CALUDE_sara_sold_oranges_l3941_394116


namespace NUMINAMATH_CALUDE_constant_value_c_l3941_394108

theorem constant_value_c (b c : ℚ) :
  (∀ x : ℚ, (x + 3) * (x + b) = x^2 + c*x + 8) →
  c = 17/3 := by
sorry

end NUMINAMATH_CALUDE_constant_value_c_l3941_394108


namespace NUMINAMATH_CALUDE_yoga_studio_average_weight_l3941_394117

theorem yoga_studio_average_weight 
  (num_men : ℕ) 
  (num_women : ℕ) 
  (avg_weight_men : ℝ) 
  (avg_weight_women : ℝ) 
  (h1 : num_men = 8) 
  (h2 : num_women = 6) 
  (h3 : avg_weight_men = 190) 
  (h4 : avg_weight_women = 120) :
  let total_people := num_men + num_women
  let total_weight := num_men * avg_weight_men + num_women * avg_weight_women
  total_weight / total_people = 160 := by
sorry

end NUMINAMATH_CALUDE_yoga_studio_average_weight_l3941_394117


namespace NUMINAMATH_CALUDE_field_length_is_48_l3941_394155

-- Define the field and pond
def field_width : ℝ := sorry
def field_length : ℝ := 2 * field_width
def pond_side : ℝ := 8

-- Define the areas
def field_area : ℝ := field_length * field_width
def pond_area : ℝ := pond_side^2

-- State the theorem
theorem field_length_is_48 :
  field_length = 2 * field_width ∧
  pond_side = 8 ∧
  pond_area = (1/18) * field_area →
  field_length = 48 := by sorry

end NUMINAMATH_CALUDE_field_length_is_48_l3941_394155


namespace NUMINAMATH_CALUDE_maximum_discount_rate_proof_l3941_394136

/-- Represents the maximum discount rate that can be applied to a product. -/
def max_discount_rate : ℝ := 8.8

/-- The cost price of the product in yuan. -/
def cost_price : ℝ := 4

/-- The original selling price of the product in yuan. -/
def original_selling_price : ℝ := 5

/-- The minimum required profit margin as a percentage. -/
def min_profit_margin : ℝ := 10

theorem maximum_discount_rate_proof :
  let discounted_price := original_selling_price * (1 - max_discount_rate / 100)
  let profit := discounted_price - cost_price
  let profit_margin := (profit / cost_price) * 100
  (profit_margin ≥ min_profit_margin) ∧
  (∀ x : ℝ, x > max_discount_rate →
    let new_discounted_price := original_selling_price * (1 - x / 100)
    let new_profit := new_discounted_price - cost_price
    let new_profit_margin := (new_profit / cost_price) * 100
    new_profit_margin < min_profit_margin) :=
by sorry

#check maximum_discount_rate_proof

end NUMINAMATH_CALUDE_maximum_discount_rate_proof_l3941_394136


namespace NUMINAMATH_CALUDE_rhombus_side_length_l3941_394141

/-- A rhombus with area L and longer diagonal three times the shorter diagonal has side length √(5L/3) -/
theorem rhombus_side_length (L : ℝ) (h : L > 0) : 
  ∃ (short_diag long_diag side : ℝ),
    short_diag > 0 ∧
    long_diag = 3 * short_diag ∧
    L = (1/2) * short_diag * long_diag ∧
    side^2 = (short_diag/2)^2 + (long_diag/2)^2 ∧
    side = Real.sqrt ((5 * L) / 3) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l3941_394141


namespace NUMINAMATH_CALUDE_certain_number_divisor_of_factorial_l3941_394144

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem certain_number_divisor_of_factorial :
  ∃! (n : ℕ), n > 0 ∧ (factorial 15) % (n^6) = 0 ∧ (factorial 15) % (n^7) ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_divisor_of_factorial_l3941_394144


namespace NUMINAMATH_CALUDE_part_one_part_two_l3941_394124

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {x | a * x^2 + b * x + 1 = 0}
def B : Set ℝ := {-1, 1}

-- Theorem for part I
theorem part_one (a b : ℝ) : B ⊆ A a b → a = -1 := by sorry

-- Theorem for part II
theorem part_two (a b : ℝ) : (A a b ∩ B).Nonempty → a^2 - b^2 + 2*a = -1 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3941_394124


namespace NUMINAMATH_CALUDE_impossible_inequalities_l3941_394152

theorem impossible_inequalities (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (hb₁ : b₁ > 0) (hb₂ : b₂ > 0) (hb₃ : b₃ > 0) :
  ¬(((a₁ * b₂) / (a₁ + b₂) < (a₂ * b₁) / (a₂ + b₁)) ∧
    ((a₂ * b₃) / (a₂ + b₃) > (a₃ * b₂) / (a₃ + b₂)) ∧
    ((a₃ * b₁) / (a₃ + b₁) > (a₁ * b₃) / (a₁ + b₃))) :=
by sorry

end NUMINAMATH_CALUDE_impossible_inequalities_l3941_394152


namespace NUMINAMATH_CALUDE_spending_ratio_l3941_394127

-- Define the amounts spent by each person
def akeno_spent : ℚ := 2985
def lev_spent : ℚ := 995  -- This is derived from the solution, but we'll use it as a given
def ambrocio_spent : ℚ := lev_spent - 177

-- State the theorem
theorem spending_ratio :
  -- Conditions
  (akeno_spent = lev_spent + ambrocio_spent + 1172) →
  (ambrocio_spent = lev_spent - 177) →
  -- Conclusion
  (lev_spent / akeno_spent = 1 / 3) := by
sorry


end NUMINAMATH_CALUDE_spending_ratio_l3941_394127


namespace NUMINAMATH_CALUDE_function_determination_l3941_394164

theorem function_determination (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 1) :
  ∀ x, f x = x^2 - 2*x := by
sorry

end NUMINAMATH_CALUDE_function_determination_l3941_394164


namespace NUMINAMATH_CALUDE_original_professors_count_l3941_394123

/-- The original number of professors in the DVEU Department of Mathematical Modeling. -/
def original_professors : ℕ := 5

/-- The number of failing grades given in the first academic year. -/
def first_year_grades : ℕ := 6480

/-- The number of failing grades given in the second academic year. -/
def second_year_grades : ℕ := 11200

/-- The increase in the number of professors in the second year. -/
def professor_increase : ℕ := 3

theorem original_professors_count :
  (first_year_grades % original_professors = 0) ∧
  (second_year_grades % (original_professors + professor_increase) = 0) ∧
  (first_year_grades / original_professors < second_year_grades / (original_professors + professor_increase)) ∧
  (∀ p : ℕ, p < original_professors →
    (first_year_grades % p = 0 ∧ 
     second_year_grades % (p + professor_increase) = 0) → 
    (first_year_grades / p ≥ second_year_grades / (p + professor_increase))) :=
by sorry

end NUMINAMATH_CALUDE_original_professors_count_l3941_394123


namespace NUMINAMATH_CALUDE_equation_equivalence_l3941_394132

theorem equation_equivalence :
  ∃ (m n p : ℤ), ∀ (a b x y : ℝ),
    (a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) ↔ 
    ((a^m*x - a^n) * (a^p*y - a^3) = a^5*b^5) :=
by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3941_394132


namespace NUMINAMATH_CALUDE_rectangle_area_l3941_394122

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3941_394122


namespace NUMINAMATH_CALUDE_motion_solution_correct_l3941_394151

/-- Two bodies moving towards each other with uniform acceleration -/
structure MotionProblem where
  initialDistance : ℝ
  initialVelocityA : ℝ
  accelerationA : ℝ
  initialVelocityB : ℝ
  accelerationB : ℝ

/-- Solution to the motion problem -/
structure MotionSolution where
  time : ℝ
  distanceA : ℝ
  distanceB : ℝ

/-- The function to solve the motion problem -/
def solveMotion (p : MotionProblem) : MotionSolution :=
  { time := 7,
    distanceA := 143.5,
    distanceB := 199.5 }

/-- Theorem stating that the solution is correct -/
theorem motion_solution_correct (p : MotionProblem) :
  p.initialDistance = 343 ∧
  p.initialVelocityA = 3 ∧
  p.accelerationA = 5 ∧
  p.initialVelocityB = 4 ∧
  p.accelerationB = 7 →
  let s := solveMotion p
  s.time = 7 ∧
  s.distanceA = 143.5 ∧
  s.distanceB = 199.5 ∧
  s.distanceA + s.distanceB = p.initialDistance :=
by
  sorry


end NUMINAMATH_CALUDE_motion_solution_correct_l3941_394151


namespace NUMINAMATH_CALUDE_base_n_representation_l3941_394115

/-- Represents a number in base n -/
def BaseN (n : ℕ) (x : ℕ) : Prop := ∃ (d₁ d₀ : ℕ), x = d₁ * n + d₀ ∧ d₀ < n

theorem base_n_representation (n a b : ℕ) : 
  n > 8 → 
  n^2 - a*n + b = 0 → 
  BaseN n a → 
  BaseN n 18 → 
  BaseN n b → 
  BaseN n 80 := by
  sorry

end NUMINAMATH_CALUDE_base_n_representation_l3941_394115


namespace NUMINAMATH_CALUDE_max_area_rectangle_l3941_394125

/-- Represents a rectangular enclosure --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of the rectangle is 400 feet --/
def isValidPerimeter (r : Rectangle) : Prop :=
  2 * r.length + 2 * r.width = 400

/-- The length is at least 90 feet --/
def hasValidLength (r : Rectangle) : Prop :=
  r.length ≥ 90

/-- The width is at least 50 feet --/
def hasValidWidth (r : Rectangle) : Prop :=
  r.width ≥ 50

/-- The area of the rectangle --/
def area (r : Rectangle) : ℝ :=
  r.length * r.width

/-- Theorem: The maximum area of a rectangle with perimeter 400 feet, 
    length ≥ 90 feet, and width ≥ 50 feet is 10,000 square feet --/
theorem max_area_rectangle :
  ∃ (r : Rectangle), isValidPerimeter r ∧ hasValidLength r ∧ hasValidWidth r ∧
    (∀ (s : Rectangle), isValidPerimeter s ∧ hasValidLength s ∧ hasValidWidth s →
      area s ≤ area r) ∧
    area r = 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l3941_394125


namespace NUMINAMATH_CALUDE_disprove_statement_l3941_394168

theorem disprove_statement : ∃ (a b c : ℤ), c < b ∧ b < a ∧ a * c < 0 ∧ a * b ≥ a * c := by
  sorry

end NUMINAMATH_CALUDE_disprove_statement_l3941_394168


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_triangle_area_l3941_394190

/-- Given a parallelogram with area 128 square meters, the area of a triangle formed by its diagonal is 64 square meters. -/
theorem parallelogram_diagonal_triangle_area (P : Real) (h : P = 128) : P / 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_triangle_area_l3941_394190


namespace NUMINAMATH_CALUDE_unique_solution_linear_system_l3941_394121

theorem unique_solution_linear_system (x y z : ℝ) :
  (3*x + 2*y + 2*z = 13) ∧
  (2*x + 3*y + 2*z = 14) ∧
  (2*x + 2*y + 3*z = 15) ↔
  (x = 1 ∧ y = 2 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_linear_system_l3941_394121


namespace NUMINAMATH_CALUDE_largest_initial_number_l3941_394113

theorem largest_initial_number :
  ∃ (a b c d e : ℕ),
    189 ∉ {x : ℕ | a ∣ x ∨ b ∣ x ∨ c ∣ x ∨ d ∣ x ∨ e ∣ x} ∧
    189 + a + b + c + d + e = 200 ∧
    ∀ (n : ℕ), n > 189 →
      ¬∃ (a' b' c' d' e' : ℕ),
        n ∉ {x : ℕ | a' ∣ x ∨ b' ∣ x ∨ c' ∣ x ∨ d' ∣ x ∨ e' ∣ x} ∧
        n + a' + b' + c' + d' + e' = 200 :=
by sorry

end NUMINAMATH_CALUDE_largest_initial_number_l3941_394113


namespace NUMINAMATH_CALUDE_waiter_tables_count_l3941_394198

/-- Calculates the number of tables a waiter has based on customer information -/
def waiterTables (initialCustomers leavingCustomers peoplePerTable : ℕ) : ℕ :=
  (initialCustomers - leavingCustomers) / peoplePerTable

/-- Theorem stating that under the given conditions, the waiter had 5 tables -/
theorem waiter_tables_count :
  waiterTables 62 17 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_count_l3941_394198


namespace NUMINAMATH_CALUDE_solve_travel_problem_l3941_394146

def travel_problem (train_distance : ℝ) : Prop :=
  let bus_distance := train_distance / 2
  let cab_distance := bus_distance / 3
  let total_distance := train_distance + bus_distance + cab_distance
  (train_distance = 300) → (total_distance = 500)

theorem solve_travel_problem : travel_problem 300 := by
  sorry

end NUMINAMATH_CALUDE_solve_travel_problem_l3941_394146


namespace NUMINAMATH_CALUDE_problem_solution_l3941_394179

theorem problem_solution (a b c : ℝ) 
  (h1 : a + b + c = 150)
  (h2 : a + 10 = b - 5)
  (h3 : b - 5 = c^2) :
  b = (1322 - 2 * Real.sqrt 1241) / 16 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3941_394179


namespace NUMINAMATH_CALUDE_withdrawal_recorded_as_negative_l3941_394137

-- Define the banking system
structure BankAccount where
  balance : ℤ

-- Define deposit and withdrawal operations
def deposit (account : BankAccount) (amount : ℕ) : BankAccount :=
  { balance := account.balance + amount }

def withdraw (account : BankAccount) (amount : ℕ) : BankAccount :=
  { balance := account.balance - amount }

-- Theorem statement
theorem withdrawal_recorded_as_negative (initial_balance : ℕ) (withdrawal_amount : ℕ) :
  (withdraw (BankAccount.mk initial_balance) withdrawal_amount).balance =
  initial_balance - withdrawal_amount :=
by sorry

end NUMINAMATH_CALUDE_withdrawal_recorded_as_negative_l3941_394137


namespace NUMINAMATH_CALUDE_probability_of_two_in_eight_elevenths_l3941_394184

/-- The decimal representation of a rational number -/
def decimal_rep (q : ℚ) : ℕ → ℕ := sorry

/-- The length of the repeating part in the decimal representation -/
def repeat_length (q : ℚ) : ℕ := sorry

/-- The count of a specific digit in the repeating part -/
def digit_count (q : ℚ) (d : ℕ) : ℕ := sorry

/-- The probability of randomly selecting a specific digit -/
def digit_probability (q : ℚ) (d : ℕ) : ℚ :=
  (digit_count q d : ℚ) / (repeat_length q : ℚ)

theorem probability_of_two_in_eight_elevenths :
  digit_probability (8 / 11) 2 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_of_two_in_eight_elevenths_l3941_394184


namespace NUMINAMATH_CALUDE_cones_problem_l3941_394107

-- Define the radii of the three cones
def r1 (r : ℝ) : ℝ := 2 * r
def r2 (r : ℝ) : ℝ := 3 * r
def r3 (r : ℝ) : ℝ := 10 * r

-- Define the radius of the smaller base of the truncated cone
def R : ℝ := 15

-- Define the distances between the centers of the bases of cones
def d12 (r : ℝ) : ℝ := 5 * r
def d13 (r : ℝ) : ℝ := 12 * r
def d23 (r : ℝ) : ℝ := 13 * r

-- Define the distances from the center of the truncated cone to the centers of the other cones
def dC1 (r : ℝ) : ℝ := r1 r + R
def dC2 (r : ℝ) : ℝ := r2 r + R
def dC3 (r : ℝ) : ℝ := r3 r + R

-- Theorem statement
theorem cones_problem (r : ℝ) (h_pos : r > 0) :
  225 * (r1 r + R)^2 = (30 * r - 10 * R)^2 + (30 * r - 3 * R)^2 → r = 29 := by
  sorry

end NUMINAMATH_CALUDE_cones_problem_l3941_394107


namespace NUMINAMATH_CALUDE_same_birthday_probability_is_one_over_365_l3941_394182

/-- The number of days in a year -/
def days_in_year : ℕ := 365

/-- The probability that two classmates have their birthdays on the same day -/
def same_birthday_probability : ℚ := 1 / days_in_year

/-- Theorem: The probability that two classmates have their birthdays on the same day is 1/365 -/
theorem same_birthday_probability_is_one_over_365 :
  same_birthday_probability = 1 / 365 := by sorry

end NUMINAMATH_CALUDE_same_birthday_probability_is_one_over_365_l3941_394182


namespace NUMINAMATH_CALUDE_not_enough_apples_for_pie_l3941_394154

theorem not_enough_apples_for_pie (tessa_initial : Real) (anita_gave : Real) (pie_requirement : Real) : 
  tessa_initial = 4.75 → anita_gave = 5.5 → pie_requirement = 12.25 → tessa_initial + anita_gave < pie_requirement :=
by
  sorry

end NUMINAMATH_CALUDE_not_enough_apples_for_pie_l3941_394154


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3941_394165

theorem greatest_divisor_with_remainders : 
  ∃ (n : ℕ), n > 0 ∧ 
  (178340 % n = 20) ∧ 
  (253785 % n = 35) ∧ 
  (375690 % n = 50) ∧ 
  (∀ m : ℕ, m > 0 → 
    (178340 % m = 20) → 
    (253785 % m = 35) → 
    (375690 % m = 50) → 
    m ≤ n) ∧
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l3941_394165


namespace NUMINAMATH_CALUDE_seven_lines_regions_l3941_394199

/-- The number of regions created by n lines in a plane, where no two lines are parallel and no three lines meet at a single point -/
def num_regions (n : ℕ) : ℕ := 1 + n + n * (n - 1) / 2

/-- Seven lines in a plane with the given conditions divide the plane into 29 regions -/
theorem seven_lines_regions : num_regions 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_seven_lines_regions_l3941_394199


namespace NUMINAMATH_CALUDE_complex_difference_magnitude_l3941_394183

theorem complex_difference_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 2)
  (h2 : Complex.abs z₂ = 2)
  (h3 : Complex.abs (z₁ + z₂) = 2 * Real.sqrt 3) :
  Complex.abs (z₁ - z₂) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_difference_magnitude_l3941_394183


namespace NUMINAMATH_CALUDE_increasing_cubic_function_l3941_394159

/-- A function f(x) = x^3 - ax^2 - 3x is increasing on [1, +∞) if and only if a ≤ 0 -/
theorem increasing_cubic_function (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (deriv (fun x => x^3 - a*x^2 - 3*x)) x ≥ 0) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_l3941_394159


namespace NUMINAMATH_CALUDE_number_plus_19_equals_47_l3941_394110

theorem number_plus_19_equals_47 (x : ℤ) : x + 19 = 47 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_number_plus_19_equals_47_l3941_394110


namespace NUMINAMATH_CALUDE_gcd_factorial_nine_eleven_l3941_394156

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem gcd_factorial_nine_eleven : 
  Nat.gcd (factorial 9) (factorial 11) = factorial 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_nine_eleven_l3941_394156


namespace NUMINAMATH_CALUDE_second_term_of_geometric_series_l3941_394142

/-- For an infinite geometric series with common ratio 1/4 and sum 16, the second term is 3 -/
theorem second_term_of_geometric_series : 
  ∀ (a : ℝ), 
  (a / (1 - (1/4 : ℝ)) = 16) →  -- Sum of infinite geometric series
  (a * (1/4 : ℝ) = 3) :=        -- Second term
by sorry

end NUMINAMATH_CALUDE_second_term_of_geometric_series_l3941_394142


namespace NUMINAMATH_CALUDE_sum_divided_non_negative_l3941_394145

theorem sum_divided_non_negative (x : ℝ) :
  ((x + 6) / 2 ≥ 0) ↔ (∃ y ≥ 0, y = (x + 6) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sum_divided_non_negative_l3941_394145


namespace NUMINAMATH_CALUDE_coffee_beans_per_cup_l3941_394188

/-- Represents the coffee consumption and cost scenario for Maddie's mom --/
structure CoffeeScenario where
  cups_per_day : ℕ
  coffee_bag_cost : ℚ
  coffee_bag_ounces : ℚ
  milk_gallons_per_week : ℚ
  milk_cost_per_gallon : ℚ
  weekly_coffee_expense : ℚ

/-- Calculates the ounces of coffee beans per cup --/
def ounces_per_cup (scenario : CoffeeScenario) : ℚ :=
  sorry

/-- Theorem stating that the ounces of coffee beans per cup is 1.5 --/
theorem coffee_beans_per_cup (scenario : CoffeeScenario) 
  (h1 : scenario.cups_per_day = 2)
  (h2 : scenario.coffee_bag_cost = 8)
  (h3 : scenario.coffee_bag_ounces = 21/2)
  (h4 : scenario.milk_gallons_per_week = 1/2)
  (h5 : scenario.milk_cost_per_gallon = 4)
  (h6 : scenario.weekly_coffee_expense = 18) :
  ounces_per_cup scenario = 3/2 :=
sorry

end NUMINAMATH_CALUDE_coffee_beans_per_cup_l3941_394188


namespace NUMINAMATH_CALUDE_bus_ride_duration_l3941_394104

/-- Calculates the bus ride time given the total trip time and other component times -/
def bus_ride_time (total_trip_time walk_time train_ride_time : ℕ) : ℕ :=
  let waiting_time := 2 * walk_time
  let total_trip_minutes := total_trip_time * 60
  let train_ride_minutes := train_ride_time * 60
  total_trip_minutes - (walk_time + waiting_time + train_ride_minutes)

/-- Theorem stating that given the specific trip components, the bus ride time is 75 minutes -/
theorem bus_ride_duration :
  bus_ride_time 8 15 6 = 75 := by
  sorry

#eval bus_ride_time 8 15 6

end NUMINAMATH_CALUDE_bus_ride_duration_l3941_394104


namespace NUMINAMATH_CALUDE_trapezoid_area_in_isosceles_triangle_l3941_394100

/-- Represents a triangle in a plane -/
structure Triangle where
  area : ℝ

/-- Represents a trapezoid in a plane -/
structure Trapezoid where
  area : ℝ

/-- The main theorem statement -/
theorem trapezoid_area_in_isosceles_triangle 
  (PQR : Triangle) 
  (smallest : Triangle)
  (RSQT : Trapezoid) :
  PQR.area = 72 ∧ 
  smallest.area = 2 ∧ 
  (∃ n : ℕ, n = 9 ∧ n * smallest.area = PQR.area) ∧
  (∃ m : ℕ, m = 3 ∧ m * smallest.area ≤ RSQT.area) →
  RSQT.area = 39 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_area_in_isosceles_triangle_l3941_394100


namespace NUMINAMATH_CALUDE_frank_brownies_columns_l3941_394161

/-- The number of columns Frank cut into the pan of brownies -/
def num_columns : ℕ := sorry

/-- The number of rows Frank cut into the pan of brownies -/
def num_rows : ℕ := 3

/-- The total number of people -/
def num_people : ℕ := 6

/-- The number of brownies each person can eat -/
def brownies_per_person : ℕ := 3

/-- The total number of brownies needed -/
def total_brownies : ℕ := num_people * brownies_per_person

theorem frank_brownies_columns :
  num_columns = total_brownies / num_rows :=
by sorry

end NUMINAMATH_CALUDE_frank_brownies_columns_l3941_394161


namespace NUMINAMATH_CALUDE_g_of_seven_equals_twentyone_l3941_394162

/-- Given that g(3x - 8) = 2x + 11 for all real x, prove that g(7) = 21 -/
theorem g_of_seven_equals_twentyone (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (3 * x - 8) = 2 * x + 11) : g 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_g_of_seven_equals_twentyone_l3941_394162


namespace NUMINAMATH_CALUDE_print_shop_charge_l3941_394114

/-- The charge per color copy at print shop X -/
def charge_x : ℚ := 1.25

/-- The number of color copies -/
def num_copies : ℕ := 60

/-- The additional charge at print shop Y for 60 copies -/
def additional_charge : ℚ := 90

/-- The charge per color copy at print shop Y -/
def charge_y : ℚ := 2.75

theorem print_shop_charge : 
  charge_y * num_copies = charge_x * num_copies + additional_charge := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charge_l3941_394114


namespace NUMINAMATH_CALUDE_pool_fill_time_l3941_394172

/-- The time required to fill a pool, given its capacity and the water supply rate. -/
def fillTime (poolCapacity : ℚ) (numHoses : ℕ) (flowRatePerHose : ℚ) : ℚ :=
  poolCapacity / (numHoses * flowRatePerHose * 60)

/-- Theorem stating that the time to fill the pool is 100/3 hours. -/
theorem pool_fill_time :
  fillTime 36000 6 3 = 100 / 3 := by
  sorry

end NUMINAMATH_CALUDE_pool_fill_time_l3941_394172


namespace NUMINAMATH_CALUDE_larger_integer_problem_l3941_394135

theorem larger_integer_problem (a b : ℕ+) : 
  a * b = 168 → 
  (a : ℤ) - (b : ℤ) = 4 ∨ (b : ℤ) - (a : ℤ) = 4 → 
  max a b = 14 := by
sorry

end NUMINAMATH_CALUDE_larger_integer_problem_l3941_394135


namespace NUMINAMATH_CALUDE_sandwich_change_calculation_l3941_394101

theorem sandwich_change_calculation (num_sandwiches : ℕ) (cost_per_sandwich : ℕ) (amount_paid : ℕ) : 
  num_sandwiches = 3 → cost_per_sandwich = 5 → amount_paid = 20 → 
  amount_paid - (num_sandwiches * cost_per_sandwich) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_change_calculation_l3941_394101


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3941_394120

/-- The solution set of the inequality -x^2 - 2x + 3 > 0 is the open interval (-3, 1) -/
theorem inequality_solution_set : 
  {x : ℝ | -x^2 - 2*x + 3 > 0} = Set.Ioo (-3) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3941_394120


namespace NUMINAMATH_CALUDE_coefficient_implies_a_squared_one_l3941_394158

/-- The coefficient of x in the expansion of (2x + a/x)^5 -/
def coefficient_of_x (a : ℝ) : ℝ := 80 * a^2

theorem coefficient_implies_a_squared_one (a : ℝ) :
  coefficient_of_x a = 80 → a^2 = 1 := by sorry

end NUMINAMATH_CALUDE_coefficient_implies_a_squared_one_l3941_394158


namespace NUMINAMATH_CALUDE_carter_reads_30_pages_l3941_394128

/-- The number of pages Oliver can read in 1 hour -/
def oliver_pages : ℕ := 40

/-- The number of pages Lucy can read in 1 hour -/
def lucy_pages : ℕ := oliver_pages + 20

/-- The number of pages Carter can read in 1 hour -/
def carter_pages : ℕ := lucy_pages / 2

/-- Theorem stating that Carter can read 30 pages in 1 hour -/
theorem carter_reads_30_pages : carter_pages = 30 := by
  sorry

end NUMINAMATH_CALUDE_carter_reads_30_pages_l3941_394128


namespace NUMINAMATH_CALUDE_fraction_sum_bounds_l3941_394170

theorem fraction_sum_bounds (a b c d : ℕ+) 
  (sum_num : a + c = 1000)
  (sum_denom : b + d = 1000) :
  (999 : ℚ) / 969 + 1 / 31 ≤ (a : ℚ) / b + (c : ℚ) / d ∧ 
  (a : ℚ) / b + (c : ℚ) / d ≤ 999 + 1 / 999 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_bounds_l3941_394170


namespace NUMINAMATH_CALUDE_total_subjects_theorem_l3941_394169

/-- The number of subjects taken by Monica, Marius, and Millie -/
def total_subjects (monica : ℕ) (marius_extra : ℕ) (millie_extra : ℕ) : ℕ :=
  monica + (monica + marius_extra) + (monica + marius_extra + millie_extra)

/-- Theorem stating the total number of subjects taken by the three students -/
theorem total_subjects_theorem :
  total_subjects 10 4 3 = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_subjects_theorem_l3941_394169


namespace NUMINAMATH_CALUDE_number_divided_by_004_l3941_394148

theorem number_divided_by_004 :
  ∃ x : ℝ, x / 0.04 = 100.9 ∧ x = 4.036 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_004_l3941_394148


namespace NUMINAMATH_CALUDE_intersection_M_N_l3941_394130

def M : Set ℝ := {-1, 0, 1, 2}
def N : Set ℝ := {x | x^2 - x ≤ 0}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3941_394130


namespace NUMINAMATH_CALUDE_right_triangle_altitude_condition_l3941_394126

theorem right_triangle_altitude_condition 
  (a b m : ℝ) 
  (h_positive : a > 0 ∧ b > 0) 
  (h_right_triangle : a^2 + b^2 = (a + b)^2 / 4) 
  (h_altitude : m = (1/5) * Real.sqrt (9*b^2 - 16*a^2)) : 
  (m = (1/5) * Real.sqrt (9*b^2 - 16*a^2)) ↔ b = 2*a := by
sorry

end NUMINAMATH_CALUDE_right_triangle_altitude_condition_l3941_394126
