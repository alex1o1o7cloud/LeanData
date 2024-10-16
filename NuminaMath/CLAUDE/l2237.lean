import Mathlib

namespace NUMINAMATH_CALUDE_triangle_trigonometric_identities_l2237_223777

/-- Given a triangle ABC with side lengths a, b, c and angles A, B, C, 
    prove two trigonometric identities. -/
theorem triangle_trigonometric_identities 
  (a b c : ℝ) (A B C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_angles : A > 0 ∧ B > 0 ∧ C > 0)
  (h_sum_angles : A + B + C = Real.pi)
  (h_cosine_law_a : a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
  (h_cosine_law_b : b^2 = a^2 + c^2 - 2*a*c*Real.cos B)
  (h_cosine_law_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)
  (h_sine_law : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C) :
  ((a^2 + b^2) / c^2 = (Real.sin A)^2 + (Real.sin B)^2 / (Real.sin C)^2) ∧
  (a^2 + b^2 + c^2 = 2*(b*c*Real.cos A + c*a*Real.cos B + a*b*Real.cos C)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_trigonometric_identities_l2237_223777


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2237_223733

theorem arithmetic_mean_problem : ∃ (x y : ℝ), 
  ((x + 12) + y + 3*x + 18 + (3*x + 6)) / 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2237_223733


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2237_223770

theorem inequality_solution_set : 
  {x : ℝ | x + 5 > -1} = {x : ℝ | x > -6} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2237_223770


namespace NUMINAMATH_CALUDE_odd_sum_is_odd_l2237_223707

theorem odd_sum_is_odd (a b : ℤ) (ha : Odd a) (hb : Odd b) : Odd (a + 2*b + 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_is_odd_l2237_223707


namespace NUMINAMATH_CALUDE_combined_weight_l2237_223751

/-- Given weights of John, Mary, and Jamison, prove their combined weight -/
theorem combined_weight 
  (mary_weight : ℝ) 
  (john_weight : ℝ) 
  (jamison_weight : ℝ)
  (h1 : john_weight = mary_weight * (5/4))
  (h2 : mary_weight = jamison_weight - 20)
  (h3 : mary_weight = 160) :
  mary_weight + john_weight + jamison_weight = 540 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_l2237_223751


namespace NUMINAMATH_CALUDE_log_sum_equals_three_l2237_223789

theorem log_sum_equals_three : Real.log 50 + Real.log 20 = 3 * Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_l2237_223789


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l2237_223781

/-- A parabola y = ax^2 + 6 is tangent to the line y = x if and only if a = 1/24 -/
theorem parabola_tangent_to_line (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 6 = x ∧ ∀ y : ℝ, y ≠ x → a * y^2 + 6 ≠ y) ↔ a = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l2237_223781


namespace NUMINAMATH_CALUDE_union_of_sets_l2237_223787

def A (a : ℝ) : Set ℝ := {1, 2^a}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ) :
  A a ∩ B a b = {1/4} → A a ∪ B a b = {-2, 1, 1/4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l2237_223787


namespace NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l2237_223702

theorem fourth_root_over_seventh_root_of_seven (x : ℝ) (h : x = 7) :
  (x^(1/4)) / (x^(1/7)) = x^(3/28) :=
by sorry

end NUMINAMATH_CALUDE_fourth_root_over_seventh_root_of_seven_l2237_223702


namespace NUMINAMATH_CALUDE_sum_of_same_sign_values_l2237_223765

theorem sum_of_same_sign_values (a b : ℝ) : 
  (abs a = 3) → (abs b = 1) → (a * b > 0) → (a + b = 4 ∨ a + b = -4) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_same_sign_values_l2237_223765


namespace NUMINAMATH_CALUDE_unique_real_roots_l2237_223736

def n : ℕ := 2016

-- Define geometric progression
def is_geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ i : ℕ, i < n → a (i + 1) = r * a i

-- Define arithmetic progression
def is_arithmetic_progression (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : ℕ, i < n → b (i + 1) = b i + d

-- Define quadratic polynomial
def P (a b : ℕ → ℝ) (i : ℕ) (x : ℝ) : ℝ :=
  x^2 + a i * x + b i

-- Define discriminant
def discriminant (a b : ℕ → ℝ) (i : ℕ) : ℝ :=
  (a i)^2 - 4 * b i

-- Theorem statement
theorem unique_real_roots
  (a : ℕ → ℝ) (b : ℕ → ℝ) (k : ℕ)
  (h_geo : is_geometric_progression a)
  (h_arith : is_arithmetic_progression b)
  (h_unique : ∀ i : ℕ, i ≤ n → i ≠ k → discriminant a b i < 0)
  (h_real : discriminant a b k ≥ 0) :
  k = 1 ∨ k = n := by sorry

end NUMINAMATH_CALUDE_unique_real_roots_l2237_223736


namespace NUMINAMATH_CALUDE_relationship_proof_l2237_223715

theorem relationship_proof (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 := by
  sorry

end NUMINAMATH_CALUDE_relationship_proof_l2237_223715


namespace NUMINAMATH_CALUDE_resulting_number_divisibility_l2237_223752

theorem resulting_number_divisibility : ∃ k : ℕ, (722425 + 335) = 30 * k := by
  sorry

end NUMINAMATH_CALUDE_resulting_number_divisibility_l2237_223752


namespace NUMINAMATH_CALUDE_chloe_carrot_problem_l2237_223728

theorem chloe_carrot_problem :
  ∀ (initial_carrots picked_next_day final_carrots thrown_out : ℕ),
    initial_carrots = 48 →
    picked_next_day = 42 →
    final_carrots = 45 →
    initial_carrots - thrown_out + picked_next_day = final_carrots →
    thrown_out = 45 := by
  sorry

end NUMINAMATH_CALUDE_chloe_carrot_problem_l2237_223728


namespace NUMINAMATH_CALUDE_rectangle_rotation_path_length_l2237_223740

/-- The length of the path traveled by point A of a rectangle ABCD when rotated as described -/
theorem rectangle_rotation_path_length (AB CD BC AD : ℝ) (h1 : AB = 4) (h2 : CD = 4) (h3 : BC = 8) (h4 : AD = 8) :
  let diagonal := Real.sqrt (AB ^ 2 + AD ^ 2)
  let first_rotation_arc := (π / 2) * diagonal
  let second_rotation_arc := (π / 2) * diagonal
  first_rotation_arc + second_rotation_arc = 4 * Real.sqrt 5 * π :=
by sorry

end NUMINAMATH_CALUDE_rectangle_rotation_path_length_l2237_223740


namespace NUMINAMATH_CALUDE_fran_required_speed_l2237_223734

/-- Calculates the required average speed for Fran to travel the same distance as Joann -/
theorem fran_required_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5) :
  (joann_speed * joann_time) / fran_time = 120 / 7 :=
by sorry

end NUMINAMATH_CALUDE_fran_required_speed_l2237_223734


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l2237_223794

theorem quadratic_root_relation (a b : ℝ) : 
  (3 : ℝ)^2 + 2*a*3 + 3*b = 0 → 2*a + b = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l2237_223794


namespace NUMINAMATH_CALUDE_average_marks_l2237_223778

theorem average_marks (n : ℕ) (avg_five : ℚ) (sixth_mark : ℚ) (h1 : n = 6) (h2 : avg_five = 74) (h3 : sixth_mark = 62) :
  (avg_five * (n - 1) + sixth_mark) / n = 72 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_l2237_223778


namespace NUMINAMATH_CALUDE_sum_x_y_is_three_sevenths_l2237_223746

theorem sum_x_y_is_three_sevenths (x y : ℚ) 
  (eq1 : 2 * x + y = 3)
  (eq2 : 3 * x - 2 * y = 12) : 
  x + y = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_is_three_sevenths_l2237_223746


namespace NUMINAMATH_CALUDE_smallest_whole_dollar_price_with_tax_l2237_223755

theorem smallest_whole_dollar_price_with_tax (n : ℕ) (x : ℕ) : n = 21 ↔ 
  n > 0 ∧ 
  x > 0 ∧
  (105 * x) % 100 = 0 ∧
  (105 * x) / 100 = n ∧
  ∀ m : ℕ, m > 0 → m < n → ¬∃ y : ℕ, y > 0 ∧ (105 * y) % 100 = 0 ∧ (105 * y) / 100 = m :=
sorry

end NUMINAMATH_CALUDE_smallest_whole_dollar_price_with_tax_l2237_223755


namespace NUMINAMATH_CALUDE_angle_bisector_inequalities_l2237_223764

/-- Given a triangle with side lengths a, b, and c, and semiperimeter p,
    prove properties about the lengths of its angle bisectors. -/
theorem angle_bisector_inequalities
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (p : ℝ) (hp : p = (a + b + c) / 2)
  (l_a l_b l_c : ℝ)
  (hl_a : l_a^2 ≤ p * (p - a))
  (hl_b : l_b^2 ≤ p * (p - b))
  (hl_c : l_c^2 ≤ p * (p - c)) :
  (l_a^2 + l_b^2 + l_c^2 ≤ p^2) ∧
  (l_a + l_b + l_c ≤ Real.sqrt 3 * p) := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_inequalities_l2237_223764


namespace NUMINAMATH_CALUDE_farmer_cages_solution_l2237_223743

/-- Represents the problem of determining the number of cages a farmer wants to fill -/
def farmer_cages_problem (initial_rabbits : ℕ) (additional_rabbits : ℕ) (total_rabbits : ℕ) : Prop :=
  ∃ (num_cages : ℕ) (rabbits_per_cage : ℕ),
    num_cages > 1 ∧
    initial_rabbits + additional_rabbits = total_rabbits ∧
    num_cages * rabbits_per_cage = total_rabbits

/-- The solution to the farmer's cage problem -/
theorem farmer_cages_solution :
  farmer_cages_problem 164 6 170 → ∃ (num_cages : ℕ), num_cages = 10 :=
by
  sorry

#check farmer_cages_solution

end NUMINAMATH_CALUDE_farmer_cages_solution_l2237_223743


namespace NUMINAMATH_CALUDE_milk_storage_calculation_l2237_223798

/-- Calculates the final amount of milk in a storage tank given initial amount,
    pumping out rate and duration, and adding rate and duration. -/
def final_milk_amount (initial : ℝ) (pump_rate : ℝ) (pump_duration : ℝ) 
                       (add_rate : ℝ) (add_duration : ℝ) : ℝ :=
  initial - pump_rate * pump_duration + add_rate * add_duration

/-- Theorem stating that given the specific conditions from the problem,
    the final amount of milk in the storage tank is 28,980 gallons. -/
theorem milk_storage_calculation :
  final_milk_amount 30000 2880 4 1500 7 = 28980 := by
  sorry

end NUMINAMATH_CALUDE_milk_storage_calculation_l2237_223798


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l2237_223783

theorem average_of_three_numbers (x : ℝ) (h1 : x = 33) : (x + 4*x + 2*x) / 3 = 77 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l2237_223783


namespace NUMINAMATH_CALUDE_solution_of_system_l2237_223762

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x^((Real.log y)^(Real.log (Real.log x))) = 10^(y^2)
def equation2 (x y : ℝ) : Prop := y^((Real.log x)^(Real.log (Real.log y))) = y^y

-- State the theorem
theorem solution_of_system :
  ∃ (x y : ℝ), x > 1 ∧ y > 1 ∧ equation1 x y ∧ equation2 x y ∧ x = 10^(10^10) ∧ y = 10^10 :=
sorry

end NUMINAMATH_CALUDE_solution_of_system_l2237_223762


namespace NUMINAMATH_CALUDE_perimeter_after_cut_l2237_223750

/-- The perimeter of the figure remaining after cutting a square corner from a larger square -/
def remaining_perimeter (original_side_length cut_side_length : ℝ) : ℝ :=
  2 * original_side_length + 3 * (original_side_length - cut_side_length)

/-- Theorem stating that the perimeter of the remaining figure is 17 -/
theorem perimeter_after_cut :
  remaining_perimeter 4 1 = 17 := by
  sorry

#eval remaining_perimeter 4 1

end NUMINAMATH_CALUDE_perimeter_after_cut_l2237_223750


namespace NUMINAMATH_CALUDE_max_radius_of_circle_l2237_223771

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the two given points
def point1 : ℝ × ℝ := (4, 0)
def point2 : ℝ × ℝ := (-4, 0)

-- Theorem statement
theorem max_radius_of_circle (C : ℝ × ℝ → ℝ → Set (ℝ × ℝ)) 
  (h1 : point1 ∈ C center radius) (h2 : point2 ∈ C center radius) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), radius ≤ 4 ∧ 
  (∀ (center' : ℝ × ℝ) (radius' : ℝ), 
    point1 ∈ C center' radius' → point2 ∈ C center' radius' → radius' ≤ radius) :=
sorry

end NUMINAMATH_CALUDE_max_radius_of_circle_l2237_223771


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2237_223796

/-- Two 2D vectors are parallel if and only if their cross product is zero -/
def parallel (v w : Fin 2 → ℝ) : Prop :=
  v 0 * w 1 = v 1 * w 0

theorem parallel_vectors_x_value :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![-3, x]
  parallel a b → x = -6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2237_223796


namespace NUMINAMATH_CALUDE_prob_select_B_is_one_fourth_prob_select_B_and_C_is_one_sixth_l2237_223760

-- Define the set of students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student

-- Define the total number of students
def total_students : ℕ := 4

-- Define the probability of selecting one student
def prob_select_one (s : Student) : ℚ :=
  1 / total_students

-- Define the probability of selecting two specific students
def prob_select_two (s1 s2 : Student) : ℚ :=
  2 / (total_students * (total_students - 1))

-- Theorem for part 1
theorem prob_select_B_is_one_fourth :
  prob_select_one Student.B = 1 / 4 := by sorry

-- Theorem for part 2
theorem prob_select_B_and_C_is_one_sixth :
  prob_select_two Student.B Student.C = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_select_B_is_one_fourth_prob_select_B_and_C_is_one_sixth_l2237_223760


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2237_223747

/-- Given two hyperbolas l and C, prove that the eccentricity of C is 3 -/
theorem hyperbola_eccentricity (k a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), k * x + y - Real.sqrt 2 * k = 0) →  -- Hyperbola l
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola C
  (abs k = b / a) →  -- Parallel asymptotes condition
  (Real.sqrt 2 * k / Real.sqrt (1 + k^2) = 4 / 3) →  -- Distance between asymptotes
  Real.sqrt (1 + b^2 / a^2) = 3 :=  -- Eccentricity of C
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2237_223747


namespace NUMINAMATH_CALUDE_base9_to_base5_conversion_l2237_223708

/-- Converts a base-9 number to its decimal (base-10) representation -/
def base9ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (9 ^ i)) 0

/-- Converts a decimal (base-10) number to its base-5 representation -/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- The base-9 representation of the number to be converted -/
def number_base9 : List Nat := [4, 2, 7]

theorem base9_to_base5_conversion :
  decimalToBase5 (base9ToDecimal number_base9) = [4, 3, 2, 4] :=
sorry

end NUMINAMATH_CALUDE_base9_to_base5_conversion_l2237_223708


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l2237_223775

theorem divisibility_equivalence (m n k : ℕ) (h : m > n) :
  (∃ a : ℤ, 4^m - 4^n = a * 3^(k+1)) ↔ (∃ b : ℤ, m - n = b * 3^k) :=
sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l2237_223775


namespace NUMINAMATH_CALUDE_hair_dye_cost_salon_hair_dye_cost_l2237_223731

/-- Calculates the cost of a box of hair dye based on salon revenue and expenses --/
theorem hair_dye_cost (haircut_price perm_price dye_job_price : ℕ)
  (haircuts perms dye_jobs : ℕ) (tips final_amount : ℕ) : ℕ :=
  let total_revenue := haircut_price * haircuts + perm_price * perms + dye_job_price * dye_jobs + tips
  let dye_cost := total_revenue - final_amount
  dye_cost / dye_jobs

/-- Proves that the cost of a box of hair dye is $10 given the problem conditions --/
theorem salon_hair_dye_cost : hair_dye_cost 30 40 60 4 1 2 50 310 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hair_dye_cost_salon_hair_dye_cost_l2237_223731


namespace NUMINAMATH_CALUDE_divisibility_by_three_l2237_223756

theorem divisibility_by_three (a b c : ℤ) (h : (9 : ℤ) ∣ (a^3 + b^3 + c^3)) :
  (3 : ℤ) ∣ a ∨ (3 : ℤ) ∣ b ∨ (3 : ℤ) ∣ c :=
by sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l2237_223756


namespace NUMINAMATH_CALUDE_x_needs_seven_days_l2237_223719

/-- The number of days X needs to finish the remaining work after Y leaves -/
def days_for_x_to_finish (x_days y_days y_worked_days : ℕ) : ℚ :=
  let x_rate : ℚ := 1 / x_days
  let y_rate : ℚ := 1 / y_days
  let work_done_by_y : ℚ := y_rate * y_worked_days
  let remaining_work : ℚ := 1 - work_done_by_y
  remaining_work / x_rate

/-- Theorem stating that X needs 7 days to finish the remaining work -/
theorem x_needs_seven_days :
  days_for_x_to_finish 21 15 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_needs_seven_days_l2237_223719


namespace NUMINAMATH_CALUDE_friend_money_pooling_l2237_223797

/-- Represents the money pooling problem with 4 friends --/
theorem friend_money_pooling
  (peter john quincy andrew : ℕ)  -- Money amounts for each friend
  (h1 : peter = 320)              -- Peter has $320
  (h2 : peter = 2 * john)         -- Peter has twice as much as John
  (h3 : quincy > peter)           -- Quincy has more than Peter
  (h4 : andrew = (115 * quincy) / 100)  -- Andrew has 15% more than Quincy
  (h5 : peter + john + quincy + andrew = 1211)  -- Total money after spending $1200
  : quincy - peter = 20 :=
by sorry

end NUMINAMATH_CALUDE_friend_money_pooling_l2237_223797


namespace NUMINAMATH_CALUDE_initial_blue_marbles_l2237_223716

/-- Proves that the initial number of blue marbles is 30 given the conditions of the problem -/
theorem initial_blue_marbles (initial_red : ℕ) (removed_red : ℕ) (total_left : ℕ)
  (h1 : initial_red = 20)
  (h2 : removed_red = 3)
  (h3 : total_left = 35) :
  initial_red + (total_left + removed_red + 4 * removed_red - (initial_red - removed_red)) = 30 := by
  sorry

end NUMINAMATH_CALUDE_initial_blue_marbles_l2237_223716


namespace NUMINAMATH_CALUDE_equation_solution_l2237_223767

theorem equation_solution (a : ℤ) : 
  (∃ x : ℕ, a * (x : ℤ) = 3) → (a = 1 ∨ a = 3) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2237_223767


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l2237_223739

theorem cubic_root_equation_solution :
  ∃ x : ℝ, x > 0 ∧ 3 * (2 + x)^(1/3) + 4 * (2 - x)^(1/3) = 6 ∧ |x - 2.096| < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l2237_223739


namespace NUMINAMATH_CALUDE_probability_theorem_l2237_223763

/-- Represents the number of buttons in a jar -/
structure JarContents where
  red : ℕ
  blue : ℕ

/-- Represents the state of both jars -/
structure JarState where
  jarA : JarContents
  jarB : JarContents

def initial_jarA : JarContents := { red := 6, blue := 14 }

def initial_jarB : JarContents := { red := 0, blue := 0 }

def initial_state : JarState := { jarA := initial_jarA, jarB := initial_jarB }

def buttons_removed (state : JarState) : ℕ :=
  initial_jarA.red + initial_jarA.blue - (state.jarA.red + state.jarA.blue)

def same_number_removed (state : JarState) : Prop :=
  state.jarB.red = state.jarB.blue

def fraction_remaining (state : JarState) : ℚ :=
  (state.jarA.red + state.jarA.blue) / (initial_jarA.red + initial_jarA.blue)

def probability_both_red (state : JarState) : ℚ :=
  (state.jarA.red / (state.jarA.red + state.jarA.blue)) *
  (state.jarB.red / (state.jarB.red + state.jarB.blue))

theorem probability_theorem (final_state : JarState) :
  buttons_removed final_state > 0 ∧
  same_number_removed final_state ∧
  fraction_remaining final_state = 5/7 →
  probability_both_red final_state = 3/28 := by
  sorry

#check probability_theorem

end NUMINAMATH_CALUDE_probability_theorem_l2237_223763


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l2237_223766

theorem quadratic_equation_result (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 10) : 
  (12 * y - 4)^2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l2237_223766


namespace NUMINAMATH_CALUDE_petya_friends_count_l2237_223795

/-- The number of friends Petya has -/
def num_friends : ℕ := 19

/-- The number of stickers Petya has -/
def total_stickers : ℕ := num_friends * 5 + 8

theorem petya_friends_count :
  (total_stickers = num_friends * 5 + 8) ∧
  (total_stickers = num_friends * 6 - 11) →
  num_friends = 19 := by
sorry

end NUMINAMATH_CALUDE_petya_friends_count_l2237_223795


namespace NUMINAMATH_CALUDE_tomato_basket_price_l2237_223758

-- Define the given values
def strawberry_plants : ℕ := 5
def tomato_plants : ℕ := 7
def strawberries_per_plant : ℕ := 14
def tomatoes_per_plant : ℕ := 16
def fruits_per_basket : ℕ := 7
def strawberry_basket_price : ℕ := 9
def total_revenue : ℕ := 186

-- Calculate total strawberries and tomatoes
def total_strawberries : ℕ := strawberry_plants * strawberries_per_plant
def total_tomatoes : ℕ := tomato_plants * tomatoes_per_plant

-- Calculate number of baskets
def strawberry_baskets : ℕ := total_strawberries / fruits_per_basket
def tomato_baskets : ℕ := total_tomatoes / fruits_per_basket

-- Define the theorem
theorem tomato_basket_price :
  (total_revenue - strawberry_baskets * strawberry_basket_price) / tomato_baskets = 6 :=
by sorry

end NUMINAMATH_CALUDE_tomato_basket_price_l2237_223758


namespace NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l2237_223774

/-- Given two planes α and β with normal vectors (1, 2, -2) and (-2, -4, k) respectively,
    if α is parallel to β, then k = 4. -/
theorem parallel_planes_normal_vectors (k : ℝ) :
  let nα : ℝ × ℝ × ℝ := (1, 2, -2)
  let nβ : ℝ × ℝ × ℝ := (-2, -4, k)
  (∃ (t : ℝ), t ≠ 0 ∧ nα = t • nβ) →
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_parallel_planes_normal_vectors_l2237_223774


namespace NUMINAMATH_CALUDE_distribution_count_l2237_223786

/-- Represents the number of ways to distribute items between two people. -/
def distribute (pencils notebooks pens : Nat) : Nat :=
  let pencil_distributions := 3  -- (1,3), (2,2), (3,1)
  let notebook_distributions := 1  -- (1,1)
  let pen_distributions := 2  -- (1,2), (2,1)
  pencil_distributions * notebook_distributions * pen_distributions

/-- Theorem stating that the number of ways to distribute the given items is 6. -/
theorem distribution_count :
  ∀ (erasers : Nat), erasers > 0 → distribute 4 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_l2237_223786


namespace NUMINAMATH_CALUDE_yoongi_has_more_points_l2237_223721

theorem yoongi_has_more_points : ∀ (yoongi_points jungkook_points : ℕ),
  yoongi_points = 4 →
  jungkook_points = 6 - 3 →
  yoongi_points > jungkook_points :=
by
  sorry

end NUMINAMATH_CALUDE_yoongi_has_more_points_l2237_223721


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2237_223741

def A : Set ℝ := {x | x / (x - 1) ≥ 0}

def B : Set ℝ := {y | ∃ x : ℝ, y = 3 * x^2 + 1}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2237_223741


namespace NUMINAMATH_CALUDE_base8_to_base10_547_l2237_223780

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The base-8 representation of the number --/
def base8Number : List Nat := [7, 4, 5]

theorem base8_to_base10_547 :
  base8ToBase10 base8Number = 359 := by
  sorry

end NUMINAMATH_CALUDE_base8_to_base10_547_l2237_223780


namespace NUMINAMATH_CALUDE_line_tangent_to_curve_l2237_223754

/-- Tangency condition for a line to a curve -/
theorem line_tangent_to_curve
  {m n u v : ℝ}
  (hm : m > 1)
  (hn : m⁻¹ + n⁻¹ = 1) :
  (∀ x y : ℝ, u * x + v * y = 1 →
    (∃ a : ℝ, x^m + y^m = a) →
    (∀ δ ε : ℝ, δ ≠ 0 ∨ ε ≠ 0 →
      (x + δ)^m + (y + ε)^m > a)) ↔
  u^n + v^n = 1 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_curve_l2237_223754


namespace NUMINAMATH_CALUDE_total_bones_l2237_223703

/-- The number of bones Xiao Qi has -/
def xiao_qi_bones : ℕ := sorry

/-- The number of bones Xiao Shi has -/
def xiao_shi_bones : ℕ := sorry

/-- The number of bones Xiao Ha has -/
def xiao_ha_bones : ℕ := sorry

/-- Xiao Ha has 2 more bones than twice the number of bones Xiao Shi has -/
axiom ha_shi_relation : xiao_ha_bones = 2 * xiao_shi_bones + 2

/-- Xiao Shi has 3 more bones than three times the number of bones Xiao Qi has -/
axiom shi_qi_relation : xiao_shi_bones = 3 * xiao_qi_bones + 3

/-- Xiao Ha has 5 fewer bones than seven times the number of bones Xiao Qi has -/
axiom ha_qi_relation : xiao_ha_bones = 7 * xiao_qi_bones - 5

/-- The total number of bones is 141 -/
theorem total_bones :
  xiao_qi_bones + xiao_shi_bones + xiao_ha_bones = 141 :=
sorry

end NUMINAMATH_CALUDE_total_bones_l2237_223703


namespace NUMINAMATH_CALUDE_fraction_unchanged_l2237_223744

theorem fraction_unchanged (a b : ℝ) : (2 * (7 * a)) / ((7 * a) + (7 * b)) = (2 * a) / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l2237_223744


namespace NUMINAMATH_CALUDE_range_of_m_given_one_root_l2237_223782

/-- The function f(x) defined in terms of x and m -/
def f (x m : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

/-- The property that f has exactly one root in [0, 1] -/
def has_one_root_in_unit_interval (m : ℝ) : Prop :=
  ∃! x, x ∈ Set.Icc 0 1 ∧ f x m = 0

/-- The theorem stating the range of m given the condition -/
theorem range_of_m_given_one_root :
  ∀ m, has_one_root_in_unit_interval m → m ∈ Set.Icc (-1) 0 ∪ Set.Icc 1 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_given_one_root_l2237_223782


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2237_223784

/-- The number of diagonals in a regular polygon with n sides -/
def diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2237_223784


namespace NUMINAMATH_CALUDE_climb_eight_steps_climb_ways_eq_fib_l2237_223749

/-- Fibonacci sequence starting with 1, 1 -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

/-- Number of ways to climb n steps -/
def climbWays : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => climbWays n + climbWays (n + 1)

theorem climb_eight_steps : climbWays 8 = 34 := by
  sorry

theorem climb_ways_eq_fib (n : ℕ) : climbWays n = fib n := by
  sorry

end NUMINAMATH_CALUDE_climb_eight_steps_climb_ways_eq_fib_l2237_223749


namespace NUMINAMATH_CALUDE_area_ratio_GHI_JKL_l2237_223757

/-- Triangle GHI with sides 7, 24, and 25 -/
def triangle_GHI : Set (ℝ × ℝ) := sorry

/-- Triangle JKL with sides 9, 40, and 41 -/
def triangle_JKL : Set (ℝ × ℝ) := sorry

/-- Area of a triangle -/
def area (triangle : Set (ℝ × ℝ)) : ℝ := sorry

/-- The ratio of the areas of triangle GHI to triangle JKL is 7/15 -/
theorem area_ratio_GHI_JKL : 
  (area triangle_GHI) / (area triangle_JKL) = 7 / 15 := by sorry

end NUMINAMATH_CALUDE_area_ratio_GHI_JKL_l2237_223757


namespace NUMINAMATH_CALUDE_building_floors_l2237_223753

/-- Given information about three buildings A, B, and C, prove that Building C has 59 floors. -/
theorem building_floors :
  let floors_A : ℕ := 4
  let floors_B : ℕ := floors_A + 9
  let floors_C : ℕ := 5 * floors_B - 6
  floors_C = 59 := by sorry

end NUMINAMATH_CALUDE_building_floors_l2237_223753


namespace NUMINAMATH_CALUDE_rectangle_area_l2237_223773

/-- Given a rectangle with length L and width W, if increasing the length by 10
    and decreasing the width by 6 doesn't change the area, and the perimeter is 76,
    then the area of the original rectangle is 360 square meters. -/
theorem rectangle_area (L W : ℝ) : 
  (L + 10) * (W - 6) = L * W → 2 * L + 2 * W = 76 → L * W = 360 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2237_223773


namespace NUMINAMATH_CALUDE_quadratic_vertex_property_l2237_223735

/-- Given a quadratic function y = -x^2 + 2x + n with vertex (m, 1), prove m - n = 1 -/
theorem quadratic_vertex_property (n m : ℝ) : 
  (∀ x, -x^2 + 2*x + n = -(x - m)^2 + 1) → m - n = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_vertex_property_l2237_223735


namespace NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l2237_223737

theorem largest_integer_negative_quadratic :
  ∃ (n : ℤ), n^2 - 13*n + 40 < 0 ∧
  ∀ (m : ℤ), m^2 - 13*m + 40 < 0 → m ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_negative_quadratic_l2237_223737


namespace NUMINAMATH_CALUDE_right_triangle_area_with_incircle_tangency_l2237_223793

/-- 
Given a right triangle with hypotenuse length c, where the incircle's point of tangency 
divides the hypotenuse in the ratio 4:9, the area of the triangle is (36/169) * c^2.
-/
theorem right_triangle_area_with_incircle_tangency (c : ℝ) (h : c > 0) : 
  ∃ (a b : ℝ), 
    a > 0 ∧ b > 0 ∧
    a^2 + b^2 = c^2 ∧  -- Pythagorean theorem for right triangle
    (4 / 13) * c * (9 / 13) * c = (1 / 2) * a * b ∧  -- Area calculation
    (1 / 2) * a * b = (36 / 169) * c^2  -- The final area formula
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_with_incircle_tangency_l2237_223793


namespace NUMINAMATH_CALUDE_investment_return_calculation_l2237_223729

theorem investment_return_calculation (total_investment : ℝ) (combined_return_rate : ℝ) 
  (investment_1 : ℝ) (return_rate_1 : ℝ) (investment_2 : ℝ) :
  total_investment = 2000 →
  combined_return_rate = 0.22 →
  investment_1 = 500 →
  return_rate_1 = 0.07 →
  investment_2 = 1500 →
  let total_return := combined_return_rate * total_investment
  let return_1 := return_rate_1 * investment_1
  let return_2 := total_return - return_1
  return_2 / investment_2 = 0.27 := by sorry

end NUMINAMATH_CALUDE_investment_return_calculation_l2237_223729


namespace NUMINAMATH_CALUDE_invalid_paper_percentage_l2237_223742

theorem invalid_paper_percentage (total_papers : ℕ) (valid_papers : ℕ) 
  (h1 : total_papers = 400)
  (h2 : valid_papers = 240) :
  (total_papers - valid_papers) * 100 / total_papers = 40 := by
  sorry

end NUMINAMATH_CALUDE_invalid_paper_percentage_l2237_223742


namespace NUMINAMATH_CALUDE_total_spent_with_tip_l2237_223717

def lunch_cost : ℝ := 60.50
def tip_percentage : ℝ := 0.20

theorem total_spent_with_tip : 
  lunch_cost * (1 + tip_percentage) = 72.60 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_with_tip_l2237_223717


namespace NUMINAMATH_CALUDE_fixed_point_satisfies_line_equation_fixed_point_is_unique_l2237_223745

/-- The line equation as a function of m, x, and y -/
def line_equation (m x y : ℝ) : ℝ := (3*m + 4)*x + (5 - 2*m)*y + 7*m - 6

/-- The fixed point through which all lines pass -/
def fixed_point : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the fixed point satisfies the line equation for all real m -/
theorem fixed_point_satisfies_line_equation :
  ∀ (m : ℝ), line_equation m fixed_point.1 fixed_point.2 = 0 := by sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_is_unique :
  ∀ (x y : ℝ), (∀ (m : ℝ), line_equation m x y = 0) → (x, y) = fixed_point := by sorry

end NUMINAMATH_CALUDE_fixed_point_satisfies_line_equation_fixed_point_is_unique_l2237_223745


namespace NUMINAMATH_CALUDE_total_birds_l2237_223718

def geese : ℕ := 58
def ducks : ℕ := 37

theorem total_birds : geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_l2237_223718


namespace NUMINAMATH_CALUDE_greatest_x_value_l2237_223712

theorem greatest_x_value (x : ℕ+) (y : ℕ) (b : ℚ) 
  (h1 : y.Prime)
  (h2 : y = 2)
  (h3 : b = 3.56)
  (h4 : (b * y^x.val : ℚ) < 600000) :
  x.val ≤ 17 ∧ ∃ (x' : ℕ+), x'.val = 17 ∧ (b * y^x'.val : ℚ) < 600000 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l2237_223712


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_minimized_l2237_223700

/-- The eccentricity of an ellipse passing through (3, 2) when a² + b² is minimized -/
theorem ellipse_eccentricity_minimized (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : (3:ℝ)^2 / a^2 + (2:ℝ)^2 / b^2 = 1) :
  let e := Real.sqrt (1 - b^2 / a^2)
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' > b' → (3:ℝ)^2 / a'^2 + (2:ℝ)^2 / b'^2 = 1 →
    a^2 + b^2 ≤ a'^2 + b'^2) →
  e = Real.sqrt 3 / 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_minimized_l2237_223700


namespace NUMINAMATH_CALUDE_square_of_1027_l2237_223726

theorem square_of_1027 : (1027 : ℕ)^2 = 1054729 := by
  sorry

end NUMINAMATH_CALUDE_square_of_1027_l2237_223726


namespace NUMINAMATH_CALUDE_shoes_savings_theorem_l2237_223701

/-- The number of weekends needed to save for shoes -/
def weekends_needed (shoe_cost : ℕ) (saved : ℕ) (earnings_per_lawn : ℕ) (lawns_per_weekend : ℕ) : ℕ :=
  let remaining := shoe_cost - saved
  let earnings_per_weekend := earnings_per_lawn * lawns_per_weekend
  (remaining + earnings_per_weekend - 1) / earnings_per_weekend

theorem shoes_savings_theorem (shoe_cost saved earnings_per_lawn lawns_per_weekend : ℕ) 
  (h1 : shoe_cost = 120)
  (h2 : saved = 30)
  (h3 : earnings_per_lawn = 5)
  (h4 : lawns_per_weekend = 3) :
  weekends_needed shoe_cost saved earnings_per_lawn lawns_per_weekend = 6 := by
  sorry

end NUMINAMATH_CALUDE_shoes_savings_theorem_l2237_223701


namespace NUMINAMATH_CALUDE_salary_left_unspent_l2237_223710

/-- The fraction of salary spent in the first week -/
def first_week_spending : ℚ := 1/4

/-- The fraction of salary spent in each of the following three weeks -/
def other_weeks_spending : ℚ := 1/5

/-- The number of weeks after the first week -/
def remaining_weeks : ℕ := 3

/-- Theorem: Given the spending conditions, the fraction of salary left unspent at the end of the month is 3/20 -/
theorem salary_left_unspent :
  1 - (first_week_spending + remaining_weeks * other_weeks_spending) = 3/20 := by
  sorry

end NUMINAMATH_CALUDE_salary_left_unspent_l2237_223710


namespace NUMINAMATH_CALUDE_quadratic_sum_of_constants_l2237_223711

theorem quadratic_sum_of_constants (x : ℝ) : ∃ (a b c : ℝ),
  (6 * x^2 + 48 * x + 162 = a * (x + b)^2 + c) ∧ (a + b + c = 76) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_constants_l2237_223711


namespace NUMINAMATH_CALUDE_abs_x_minus_two_integral_l2237_223730

theorem abs_x_minus_two_integral : ∫ x in (0)..(4), |x - 2| = 4 := by sorry

end NUMINAMATH_CALUDE_abs_x_minus_two_integral_l2237_223730


namespace NUMINAMATH_CALUDE_intersects_both_branches_iff_l2237_223714

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a line with slope k passing through a point -/
structure Line where
  k : ℝ

/-- Predicate indicating if a line intersects both branches of a hyperbola -/
def intersects_both_branches (h : Hyperbola) (l : Line) : Prop := sorry

/-- The necessary and sufficient condition for a line to intersect both branches of a hyperbola -/
theorem intersects_both_branches_iff (h : Hyperbola) (l : Line) :
  intersects_both_branches h l ↔ -h.b / h.a < l.k ∧ l.k < h.b / h.a := by sorry

end NUMINAMATH_CALUDE_intersects_both_branches_iff_l2237_223714


namespace NUMINAMATH_CALUDE_composite_sum_of_powers_l2237_223724

theorem composite_sum_of_powers (a b c d : ℕ) (h : a * b = c * d) :
  ∃ (x y : ℕ), x > 1 ∧ y > 1 ∧ x * y = a^2000 + b^2000 + c^2000 + d^2000 :=
by
  sorry

end NUMINAMATH_CALUDE_composite_sum_of_powers_l2237_223724


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2237_223723

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_4 + a_8 = 16, then a_2 + a_10 = 16 -/
theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ) (h_arithmetic : arithmetic_sequence a) (h_sum : a 4 + a 8 = 16) :
  a 2 + a 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l2237_223723


namespace NUMINAMATH_CALUDE_probability_of_mixed_team_l2237_223722

def num_girls : ℕ := 3
def num_boys : ℕ := 2
def team_size : ℕ := 2
def total_group_size : ℕ := num_girls + num_boys

def num_total_combinations : ℕ := (total_group_size.choose team_size)
def num_mixed_combinations : ℕ := num_girls * num_boys

theorem probability_of_mixed_team :
  (num_mixed_combinations : ℚ) / num_total_combinations = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_mixed_team_l2237_223722


namespace NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l2237_223790

theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 750) 
  (h_product : a * b = 18750) : 
  Nat.gcd a b = 25 := by
sorry

end NUMINAMATH_CALUDE_hcf_from_lcm_and_product_l2237_223790


namespace NUMINAMATH_CALUDE_apple_picking_multiple_l2237_223759

theorem apple_picking_multiple (K : ℕ) (M : ℕ) : 
  K + 274 = 340 → 
  274 = M * K + 10 →
  M = 4 := by sorry

end NUMINAMATH_CALUDE_apple_picking_multiple_l2237_223759


namespace NUMINAMATH_CALUDE_juan_marbles_l2237_223779

theorem juan_marbles (connie_marbles : ℕ) (juan_extra : ℕ) : 
  connie_marbles = 39 → juan_extra = 25 → connie_marbles + juan_extra = 64 := by
  sorry

end NUMINAMATH_CALUDE_juan_marbles_l2237_223779


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l2237_223761

theorem smallest_common_multiple_of_9_and_6 : ∃ n : ℕ+, (∀ m : ℕ+, 9 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 9 ∣ n ∧ 6 ∣ n := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_9_and_6_l2237_223761


namespace NUMINAMATH_CALUDE_evenBlueFaceCubesFor6x3x2_l2237_223748

/-- Represents a rectangular block with given dimensions -/
structure Block where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of cubes with an even number of blue faces -/
def evenBlueFaceCubes (b : Block) : ℕ :=
  let edgeCubes := 4 * (b.length + b.width + b.height - 6)
  let internalCubes := (b.length - 2) * (b.width - 2) * (b.height - 2)
  edgeCubes + internalCubes

/-- The main theorem stating that a 6x3x2 block has 20 cubes with an even number of blue faces -/
theorem evenBlueFaceCubesFor6x3x2 : 
  evenBlueFaceCubes { length := 6, width := 3, height := 2 } = 20 := by
  sorry

end NUMINAMATH_CALUDE_evenBlueFaceCubesFor6x3x2_l2237_223748


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l2237_223788

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I : ℂ) = (a + Complex.I) / (b + 2 * Complex.I) → a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l2237_223788


namespace NUMINAMATH_CALUDE_dacid_weighted_average_score_l2237_223732

/-- Calculates the weighted average score for a student given their marks and subject weightages --/
def weighted_average_score (
  english_mark : ℚ)
  (math_mark : ℚ)
  (physics_mark : ℚ)
  (chemistry_mark : ℚ)
  (biology_mark : ℚ)
  (cs_mark : ℚ)
  (sports_mark : ℚ)
  (english_weight : ℚ)
  (math_weight : ℚ)
  (physics_weight : ℚ)
  (chemistry_weight : ℚ)
  (biology_weight : ℚ)
  (cs_weight : ℚ)
  (sports_weight : ℚ) : ℚ :=
  english_mark * english_weight +
  math_mark * math_weight +
  physics_mark * physics_weight +
  chemistry_mark * chemistry_weight +
  biology_mark * biology_weight +
  (cs_mark * 100 / 150) * cs_weight +
  (sports_mark * 100 / 150) * sports_weight

/-- Theorem stating that Dacid's weighted average score is approximately 86.82 --/
theorem dacid_weighted_average_score :
  ∃ ε > 0, abs (weighted_average_score 96 95 82 97 95 88 83 0.25 0.20 0.10 0.15 0.10 0.15 0.05 - 86.82) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_dacid_weighted_average_score_l2237_223732


namespace NUMINAMATH_CALUDE_dannys_remaining_bottle_caps_l2237_223709

def initial_bottle_caps : ℕ := 91
def lost_bottle_caps : ℕ := 66

theorem dannys_remaining_bottle_caps :
  initial_bottle_caps - lost_bottle_caps = 25 := by
  sorry

end NUMINAMATH_CALUDE_dannys_remaining_bottle_caps_l2237_223709


namespace NUMINAMATH_CALUDE_subset_ratio_eight_elements_l2237_223772

theorem subset_ratio_eight_elements : 
  let n : ℕ := 8
  let total_subsets : ℕ := 2^n
  let three_elem_subsets : ℕ := n.choose 3
  (three_elem_subsets : ℚ) / total_subsets = 7/32 := by sorry

end NUMINAMATH_CALUDE_subset_ratio_eight_elements_l2237_223772


namespace NUMINAMATH_CALUDE_pet_ownership_l2237_223706

theorem pet_ownership (total_students : ℕ) 
  (dog_owners cat_owners other_pet_owners : ℕ)
  (no_pet_owners : ℕ)
  (only_dog_owners only_cat_owners only_other_pet_owners : ℕ) :
  total_students = 40 →
  dog_owners = total_students / 2 →
  cat_owners = total_students / 4 →
  other_pet_owners = 8 →
  no_pet_owners = 5 →
  only_dog_owners = 15 →
  only_cat_owners = 4 →
  only_other_pet_owners = 5 →
  ∃ (all_three_pets : ℕ),
    all_three_pets = 1 ∧
    dog_owners = only_dog_owners + (other_pet_owners - only_other_pet_owners) + 
                 (cat_owners - only_cat_owners) - all_three_pets + all_three_pets ∧
    cat_owners = only_cat_owners + (other_pet_owners - only_other_pet_owners) + 
                 (dog_owners - only_dog_owners) - all_three_pets + all_three_pets ∧
    other_pet_owners = only_other_pet_owners + (dog_owners - only_dog_owners) + 
                       (cat_owners - only_cat_owners) - all_three_pets + all_three_pets ∧
    total_students = dog_owners + cat_owners + other_pet_owners - 
                     (dog_owners - only_dog_owners) - (cat_owners - only_cat_owners) - 
                     (other_pet_owners - only_other_pet_owners) + all_three_pets + no_pet_owners :=
by
  sorry

end NUMINAMATH_CALUDE_pet_ownership_l2237_223706


namespace NUMINAMATH_CALUDE_evaluate_expression_l2237_223725

theorem evaluate_expression : 3 - 5 * (2^3 + 3) * 2 = -107 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2237_223725


namespace NUMINAMATH_CALUDE_tech_class_avg_age_l2237_223768

def avg_age_arts : ℝ := 21
def num_arts_classes : ℕ := 8
def num_tech_classes : ℕ := 5
def overall_avg_age : ℝ := 19.846153846153847

theorem tech_class_avg_age :
  let total_classes := num_arts_classes + num_tech_classes
  let total_age := overall_avg_age * total_classes
  let arts_total_age := avg_age_arts * num_arts_classes
  (total_age - arts_total_age) / num_tech_classes = 990.4000000000002 := by
sorry

end NUMINAMATH_CALUDE_tech_class_avg_age_l2237_223768


namespace NUMINAMATH_CALUDE_simple_interest_investment_l2237_223785

/-- Proves that an initial investment of $1000 with 10% simple interest over 3 years results in $1300 --/
theorem simple_interest_investment (P : ℝ) : 
  (P * (1 + 0.1 * 3) = 1300) → P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_investment_l2237_223785


namespace NUMINAMATH_CALUDE_average_after_removal_l2237_223799

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) :
  Finset.card numbers = 12 →
  sum / 12 = 72 →
  60 ∈ numbers →
  80 ∈ numbers →
  ((sum - 60 - 80) / 10 : ℝ) = 72.4 := by
  sorry

end NUMINAMATH_CALUDE_average_after_removal_l2237_223799


namespace NUMINAMATH_CALUDE_volleyball_team_selection_l2237_223704

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * Nat.factorial (n - k))

theorem volleyball_team_selection (total_players starters : ℕ) (twins : ℕ) : 
  total_players = 16 → 
  starters = 6 → 
  twins = 2 →
  (choose (total_players - twins) (starters - twins) + 
   choose (total_players - twins) starters) = 4004 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_selection_l2237_223704


namespace NUMINAMATH_CALUDE_smallest_congruent_to_zero_l2237_223776

theorem smallest_congruent_to_zero : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (k : ℕ), k > 0 → k < 10 → n % k = 0) ∧
  (∀ (m : ℕ), m > 0 → (∀ (k : ℕ), k > 0 → k < 10 → m % k = 0) → m ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_congruent_to_zero_l2237_223776


namespace NUMINAMATH_CALUDE_sqrt_three_subtraction_l2237_223791

theorem sqrt_three_subtraction : 2 * Real.sqrt 3 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_subtraction_l2237_223791


namespace NUMINAMATH_CALUDE_haris_capital_contribution_l2237_223713

/-- Represents the capital contribution of a business partner -/
structure Capital where
  amount : ℕ
  months : ℕ

/-- Calculates the effective capital based on the amount and months invested -/
def effectiveCapital (c : Capital) : ℕ := c.amount * c.months

/-- Represents the profit-sharing ratio between two partners -/
structure ProfitRatio where
  first : ℕ
  second : ℕ

theorem haris_capital_contribution 
  (praveens_capital : Capital)
  (haris_join_month : ℕ)
  (total_months : ℕ)
  (profit_ratio : ProfitRatio)
  (h1 : praveens_capital.amount = 3360)
  (h2 : praveens_capital.months = total_months)
  (h3 : haris_join_month = 5)
  (h4 : total_months = 12)
  (h5 : profit_ratio.first = 2)
  (h6 : profit_ratio.second = 3)
  : ∃ (haris_capital : Capital), 
    haris_capital.amount = 8640 ∧ 
    haris_capital.months = total_months - haris_join_month ∧
    effectiveCapital praveens_capital * profit_ratio.second = 
    effectiveCapital haris_capital * profit_ratio.first :=
sorry

end NUMINAMATH_CALUDE_haris_capital_contribution_l2237_223713


namespace NUMINAMATH_CALUDE_football_team_throwers_l2237_223720

theorem football_team_throwers :
  ∀ (total_players throwers right_handed : ℕ),
    total_players = 70 →
    throwers ≤ total_players →
    right_handed = 63 →
    3 * (right_handed - throwers) = 2 * (total_players - throwers) →
    throwers = 49 := by
  sorry

end NUMINAMATH_CALUDE_football_team_throwers_l2237_223720


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2237_223769

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ b : ℝ, a - (10 : ℂ) / (3 - Complex.I) = b * Complex.I) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2237_223769


namespace NUMINAMATH_CALUDE_translation_right_proof_l2237_223705

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation function
def translate_right (p : Point) (units : ℝ) : Point :=
  (p.1 + units, p.2)

-- Theorem statement
theorem translation_right_proof :
  let A : Point := (-4, 3)
  let A' : Point := translate_right A 2
  A' = (-2, 3) := by sorry

end NUMINAMATH_CALUDE_translation_right_proof_l2237_223705


namespace NUMINAMATH_CALUDE_exists_x_squared_plus_two_x_plus_one_nonpositive_l2237_223727

theorem exists_x_squared_plus_two_x_plus_one_nonpositive :
  ∃ x : ℝ, x^2 + 2*x + 1 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_exists_x_squared_plus_two_x_plus_one_nonpositive_l2237_223727


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2237_223738

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a b : ℕ → ℝ) :
  ArithmeticSequence a → ArithmeticSequence b →
  (a 1 + b 1 = 7) → (a 3 + b 3 = 21) →
  (a 5 + b 5 = 35) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2237_223738


namespace NUMINAMATH_CALUDE_even_number_induction_step_l2237_223792

theorem even_number_induction_step (P : ℕ → Prop) (k : ℕ) 
  (h_even : Even k) (h_ge_2 : k ≥ 2) (h_base : P 2) (h_k : P k) :
  (∀ n, Even n → n ≥ 2 → P n) ↔ 
  (P k → P (k + 2)) :=
sorry

end NUMINAMATH_CALUDE_even_number_induction_step_l2237_223792
