import Mathlib

namespace NUMINAMATH_GPT_exists_polyhedron_with_given_vertices_and_edges_l1751_175118

theorem exists_polyhedron_with_given_vertices_and_edges :
  ∃ (V : Finset (String)) (E : Finset (Finset (String))),
    V = { "A", "B", "C", "D", "E", "F", "G", "H" } ∧
    E = { { "A", "B" }, { "A", "C" }, { "A", "H" }, { "B", "C" },
          { "B", "D" }, { "C", "D" }, { "D", "E" }, { "E", "F" },
          { "E", "G" }, { "F", "G" }, { "F", "H" }, { "G", "H" } } ∧
    (V.card : ℤ) - (E.card : ℤ) + 6 = 2 :=
by
  sorry

end NUMINAMATH_GPT_exists_polyhedron_with_given_vertices_and_edges_l1751_175118


namespace NUMINAMATH_GPT_find_polynomial_parameters_and_minimum_value_l1751_175158

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_polynomial_parameters_and_minimum_value 
  (a b c : ℝ)
  (h1 : f (-1) a b c = 7)
  (h2 : 3 * (-1)^2 + 2 * a * (-1) + b = 0)
  (h3 : 3 * 3^2 + 2 * a * 3 + b = 0)
  (h4 : a = -3)
  (h5 : b = -9)
  (h6 : c = 2) :
  f 3 (-3) (-9) 2 = -25 :=
by
  sorry

end NUMINAMATH_GPT_find_polynomial_parameters_and_minimum_value_l1751_175158


namespace NUMINAMATH_GPT_age_difference_l1751_175103

theorem age_difference (x : ℕ) 
  (h_ratio : 4 * x + 3 * x + 7 * x = 126)
  (h_halima : 4 * x = 36)
  (h_beckham : 3 * x = 27) :
  4 * x - 3 * x = 9 :=
by sorry

end NUMINAMATH_GPT_age_difference_l1751_175103


namespace NUMINAMATH_GPT_arthur_initial_amount_l1751_175125

def initial_amount (X : ℝ) : Prop :=
  (1/5) * X = 40

theorem arthur_initial_amount (X : ℝ) (h : initial_amount X) : X = 200 :=
by
  sorry

end NUMINAMATH_GPT_arthur_initial_amount_l1751_175125


namespace NUMINAMATH_GPT_sphere_radius_in_cube_l1751_175168

theorem sphere_radius_in_cube (r : ℝ) (n : ℕ) (side_length : ℝ) 
  (h1 : side_length = 2) 
  (h2 : n = 16)
  (h3 : ∀ (i : ℕ), i < n → (center_distance : ℝ) = 2 * r)
  (h4: ∀ (i : ℕ), i < n → (face_distance : ℝ) = r) : 
  r = 1 :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_in_cube_l1751_175168


namespace NUMINAMATH_GPT_cheaper_fluid_cost_is_20_l1751_175110

variable (x : ℕ) -- Denote the cost per drum of the cheaper fluid as x

-- Given conditions:
variable (total_drums : ℕ) (cheaper_drums : ℕ) (expensive_cost : ℕ) (total_cost : ℕ)
variable (remaining_drums : ℕ) (total_expensive_cost : ℕ)

axiom total_drums_eq : total_drums = 7
axiom cheaper_drums_eq : cheaper_drums = 5
axiom expensive_cost_eq : expensive_cost = 30
axiom total_cost_eq : total_cost = 160
axiom remaining_drums_eq : remaining_drums = total_drums - cheaper_drums
axiom total_expensive_cost_eq : total_expensive_cost = remaining_drums * expensive_cost

-- The equation for the total cost:
axiom total_cost_eq2 : total_cost = cheaper_drums * x + total_expensive_cost

-- The goal: Prove that the cheaper fluid cost per drum is $20
theorem cheaper_fluid_cost_is_20 : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_cheaper_fluid_cost_is_20_l1751_175110


namespace NUMINAMATH_GPT_initial_maintenance_time_l1751_175111

theorem initial_maintenance_time (x : ℝ) 
  (h1 : (1 + (1 / 3)) * x = 60) : 
  x = 45 :=
by
  sorry

end NUMINAMATH_GPT_initial_maintenance_time_l1751_175111


namespace NUMINAMATH_GPT_rectangle_perimeter_of_triangle_area_l1751_175106

theorem rectangle_perimeter_of_triangle_area
  (h_right : ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a = 9 ∧ b = 12 ∧ c = 15)
  (rect_length : ℕ) 
  (rect_area_eq_triangle_area : ∃ (area : ℕ), area = 1/2 * 9 * 12 ∧ area = rect_length * rect_width ) 
  : ∃ (perimeter : ℕ), perimeter = 2 * (6 + rect_width) ∧ perimeter = 30 :=
sorry

end NUMINAMATH_GPT_rectangle_perimeter_of_triangle_area_l1751_175106


namespace NUMINAMATH_GPT_flour_amount_second_combination_l1751_175181

-- Define given conditions as parameters
variables {sugar_cost flour_cost : ℝ} (sugar_per_pound flour_per_pound : ℝ)
variable (cost1 cost2 : ℝ)

axiom cost1_eq :
  40 * sugar_per_pound + 16 * flour_per_pound = cost1

axiom cost2_eq :
  30 * sugar_per_pound + flour_cost = cost2

axiom sugar_rate :
  sugar_per_pound = 0.45

axiom flour_rate :
  flour_per_pound = 0.45

-- Define the target theorem
theorem flour_amount_second_combination : ∃ flour_amount : ℝ, flour_amount = 28 := by
  sorry

end NUMINAMATH_GPT_flour_amount_second_combination_l1751_175181


namespace NUMINAMATH_GPT_smallest_number_of_students_l1751_175183

theorem smallest_number_of_students (n9 n7 n8 : ℕ) (h7 : 9 * n7 = 7 * n9) (h8 : 5 * n8 = 9 * n9) :
  n9 + n7 + n8 = 134 :=
by
  -- Skipping proof with sorry
  sorry

end NUMINAMATH_GPT_smallest_number_of_students_l1751_175183


namespace NUMINAMATH_GPT_money_left_after_purchases_is_correct_l1751_175133

noncomputable def initial_amount : ℝ := 12.50
noncomputable def cost_pencil : ℝ := 1.25
noncomputable def cost_notebook : ℝ := 3.45
noncomputable def cost_pens : ℝ := 4.80

noncomputable def total_cost : ℝ := cost_pencil + cost_notebook + cost_pens
noncomputable def money_left : ℝ := initial_amount - total_cost

theorem money_left_after_purchases_is_correct : money_left = 3.00 :=
by
  -- proof goes here, skipping with sorry for now
  sorry

end NUMINAMATH_GPT_money_left_after_purchases_is_correct_l1751_175133


namespace NUMINAMATH_GPT_cara_between_friends_l1751_175144

theorem cara_between_friends (n : ℕ) (h : n = 6) : ∃ k : ℕ, k = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_cara_between_friends_l1751_175144


namespace NUMINAMATH_GPT_cole_drive_time_l1751_175145

theorem cole_drive_time (d : ℝ) (h1 : d / 75 + d / 105 = 1) : (d / 75) * 60 = 35 :=
by
  -- Using the given equation: d / 75 + d / 105 = 1
  -- We solve it step by step and finally show that the time it took to drive to work is 35 minutes.
  sorry

end NUMINAMATH_GPT_cole_drive_time_l1751_175145


namespace NUMINAMATH_GPT_range_of_m_l1751_175175

theorem range_of_m 
  (h : ∀ x, -1 < x ∧ x < 4 → x > 2 * (m: ℝ)^2 - 3)
  : ∀ (m: ℝ), -1 ≤ m ∧ m ≤ 1 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l1751_175175


namespace NUMINAMATH_GPT_repeating_decimal_mul_l1751_175119

theorem repeating_decimal_mul (x : ℝ) (hx : x = 0.3333333333333333) :
  x * 12 = 4 :=
sorry

end NUMINAMATH_GPT_repeating_decimal_mul_l1751_175119


namespace NUMINAMATH_GPT_total_girls_is_68_l1751_175141

-- Define the initial conditions
def track_length : ℕ := 100
def student_spacing : ℕ := 2
def girls_per_cycle : ℕ := 2
def cycle_length : ℕ := 3

-- Calculate the number of students on one side
def students_on_one_side : ℕ := track_length / student_spacing + 1

-- Number of cycles of three students
def num_cycles : ℕ := students_on_one_side / cycle_length

-- Number of girls on one side
def girls_on_one_side : ℕ := num_cycles * girls_per_cycle

-- Total number of girls on both sides
def total_girls : ℕ := girls_on_one_side * 2

theorem total_girls_is_68 : total_girls = 68 := by
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_total_girls_is_68_l1751_175141


namespace NUMINAMATH_GPT_beads_per_package_eq_40_l1751_175172

theorem beads_per_package_eq_40 (b r : ℕ) (x : ℕ) (total_beads : ℕ) 
(h1 : b = 3) (h2 : r = 5) (h3 : total_beads = 320) (h4 : total_beads = (b + r) * x) :
  x = 40 := by
  sorry

end NUMINAMATH_GPT_beads_per_package_eq_40_l1751_175172


namespace NUMINAMATH_GPT_solve_eq_l1751_175116

theorem solve_eq (a b : ℕ) : a * a = b * (b + 7) ↔ (a, b) = (0, 0) ∨ (a, b) = (12, 9) :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_l1751_175116


namespace NUMINAMATH_GPT_f_odd_and_minimum_period_pi_l1751_175163

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x

theorem f_odd_and_minimum_period_pi :
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + π) = f x) :=
  sorry

end NUMINAMATH_GPT_f_odd_and_minimum_period_pi_l1751_175163


namespace NUMINAMATH_GPT_odd_prime_power_condition_l1751_175179

noncomputable def is_power_of (a b : ℕ) : Prop :=
  ∃ t : ℕ, b = a ^ t

theorem odd_prime_power_condition (n p x y k : ℕ) (hn : 1 < n) (hp_prime : Prime p) 
  (hp_odd : p % 2 = 1) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (hx_odd : x % 2 ≠ 0) 
  (hy_odd : y % 2 ≠ 0) (h_eq : x^n + y^n = p^k) :
  is_power_of p n :=
sorry

end NUMINAMATH_GPT_odd_prime_power_condition_l1751_175179


namespace NUMINAMATH_GPT_length_of_X_l1751_175100

theorem length_of_X
  {X : ℝ}
  (h1 : 2 + 2 + X = 4 + X)
  (h2 : 3 + 4 + 1 = 8)
  (h3 : ∃ y : ℝ, y * (4 + X) = 29) : 
  X = 4 := sorry

end NUMINAMATH_GPT_length_of_X_l1751_175100


namespace NUMINAMATH_GPT_rectangle_area_l1751_175109

theorem rectangle_area :
  ∀ (width length : ℝ), (length = 3 * width) → (width = 5) → (length * width = 75) :=
by
  intros width length h1 h2
  rw [h2, h1]
  sorry

end NUMINAMATH_GPT_rectangle_area_l1751_175109


namespace NUMINAMATH_GPT_gcd_three_digit_palindromes_l1751_175193

theorem gcd_three_digit_palindromes : 
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) → 
  ∃ d : ℕ, d = 1 ∧ ∀ n m : ℕ, (n = 101 * a + 10 * b) → (m = 101 * a + 10 * b) → gcd n m = d := 
by sorry

end NUMINAMATH_GPT_gcd_three_digit_palindromes_l1751_175193


namespace NUMINAMATH_GPT_part1_part2_l1751_175198

def f (x a : ℝ) := |x - a| + 2 * |x + 1|

-- Part 1: Solve the inequality f(x) > 4 when a = 2
theorem part1 (x : ℝ) : f x 2 > 4 ↔ (x < -4/3 ∨ x > 0) := by
  sorry

-- Part 2: If the solution set of the inequality f(x) < 3x + 4 is {x | x > 2}, find the value of a.
theorem part2 (a : ℝ) : (∀ x : ℝ, (f x a < 3 * x + 4 ↔ x > 2)) → a = 6 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1751_175198


namespace NUMINAMATH_GPT_fraction_value_l1751_175164

theorem fraction_value : (4 * 5) / 10 = 2 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l1751_175164


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1751_175174

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b^2 = 2 * a^2) : ℝ :=
  (1 + b^2 / a^2) ^ (1/2)

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b^2 = 2 * a^2) :
  hyperbola_eccentricity a b h1 h2 h3 = Real.sqrt 3 := 
by
  unfold hyperbola_eccentricity
  rw [h3]
  simp
  sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1751_175174


namespace NUMINAMATH_GPT_min_value_y1_minus_4y2_l1751_175177

/-- 
Suppose a parabola C : y^2 = 4x intersects at points A(x1, y1) and B(x2, y2) with a line 
passing through its focus. Given that A is in the first quadrant, 
the minimum value of |y1 - 4y2| is 8.
--/
theorem min_value_y1_minus_4y2 (x1 y1 x2 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) 
  (h2 : y2^2 = 4 * x2)
  (h3 : x1 > 0) (h4 : y1 > 0) 
  (focus : (1, 0) ∈ {(x, y) | y^2 = 4 * x}) : 
  (|y1 - 4 * y2|) ≥ 8 :=
sorry

end NUMINAMATH_GPT_min_value_y1_minus_4y2_l1751_175177


namespace NUMINAMATH_GPT_rectangle_area_l1751_175138

theorem rectangle_area (l w : ℕ) (h_diagonal : l^2 + w^2 = 17^2) (h_perimeter : l + w = 23) : l * w = 120 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1751_175138


namespace NUMINAMATH_GPT_fraction_transformation_half_l1751_175167

theorem fraction_transformation_half (a b : ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  ((2 * a + 2 * b) / (4 * a^2 + 4 * b^2)) = (1 / 2) * ((a + b) / (a^2 + b^2)) :=
by sorry

end NUMINAMATH_GPT_fraction_transformation_half_l1751_175167


namespace NUMINAMATH_GPT_brad_money_l1751_175113

noncomputable def money_problem : Prop :=
  ∃ (B J D : ℝ), 
    J = 2 * B ∧
    J = (3/4) * D ∧
    B + J + D = 68 ∧
    B = 12

theorem brad_money : money_problem :=
by {
  -- Insert proof steps here if necessary
  sorry
}

end NUMINAMATH_GPT_brad_money_l1751_175113


namespace NUMINAMATH_GPT_probability_A_or_B_complement_l1751_175114

-- Define the sample space for rolling a die
def sample_space : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define Event A: the outcome is an even number not greater than 4
def event_A : Finset ℕ := {2, 4}

-- Define Event B: the outcome is less than 6
def event_B : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the complement of Event B
def event_B_complement : Finset ℕ := {6}

-- Mutually exclusive property of events A and B_complement
axiom mutually_exclusive (A B_complement: Finset ℕ) : A ∩ B_complement = ∅

-- Define the probability function
def probability (events: Finset ℕ) : ℚ := (events.card : ℚ) / (sample_space.card : ℚ)

-- Theorem stating the probability of event (A + B_complement)
theorem probability_A_or_B_complement : probability (event_A ∪ event_B_complement) = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_probability_A_or_B_complement_l1751_175114


namespace NUMINAMATH_GPT_age_ratio_l1751_175170

theorem age_ratio (A B C : ℕ) (h1 : A = B + 2) (h2 : A + B + C = 27) (h3 : B = 10) : B / C = 2 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_l1751_175170


namespace NUMINAMATH_GPT_smallest_A_is_144_l1751_175184

noncomputable def smallest_A (B : ℕ) := B * 28 + 4

theorem smallest_A_is_144 :
  ∃ (B : ℕ), smallest_A B = 144 ∧ ∀ (B' : ℕ), B' * 28 + 4 < 144 → false :=
by
  sorry

end NUMINAMATH_GPT_smallest_A_is_144_l1751_175184


namespace NUMINAMATH_GPT_find_constants_and_min_value_l1751_175173

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x * Real.log x
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b * Real.log x + b
noncomputable def g (a b x : ℝ) := f a b x - Real.exp 1 * x^2

theorem find_constants_and_min_value :
  (∀ (a b : ℝ),
    -- Condition for the derivative at x = 1 and the given tangent line slope
    (f' a b 1 = 2 * Real.exp 1) ∧
    -- Condition for the function value at x = 1
    (f a b 1 = Real.exp 1) →
    -- Expected results for a and b
    (a = 1 ∧ b = Real.exp 1)) ∧

  -- Evaluating the minimum value of the function g(x)
  (∀ (x : ℝ), 0 < x →
    -- Given the minimum occurs at x = 1
    g 1 (Real.exp 1) 1 = 0 ∧
    (∀ (x : ℝ), 0 < x →
      (g 1 (Real.exp 1) x ≥ 0))) :=
sorry

end NUMINAMATH_GPT_find_constants_and_min_value_l1751_175173


namespace NUMINAMATH_GPT_hotel_accommodation_arrangements_l1751_175194

theorem hotel_accommodation_arrangements :
  let triple_room := 1
  let double_rooms := 2
  let adults := 3
  let children := 2
  (∀ (triple_room : ℕ) (double_rooms : ℕ) (adults : ℕ) (children : ℕ),
    children ≤ adults ∧ double_rooms + triple_room ≥ 1 →
    (∃ (arrangements : ℕ),
      arrangements = 60)) :=
sorry

end NUMINAMATH_GPT_hotel_accommodation_arrangements_l1751_175194


namespace NUMINAMATH_GPT_number_of_possible_values_b_l1751_175169

theorem number_of_possible_values_b : 
  ∃ n : ℕ, n = 2 ∧ 
    (∀ b : ℕ, b ≥ 2 → (b^3 ≤ 256) ∧ (256 < b^4) ↔ (b = 5 ∨ b = 6)) :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_possible_values_b_l1751_175169


namespace NUMINAMATH_GPT_convert_degrees_to_radians_l1751_175112

theorem convert_degrees_to_radians (deg : ℝ) (deg_eq : deg = -300) : 
  deg * (π / 180) = - (5 * π) / 3 := 
by
  rw [deg_eq]
  sorry

end NUMINAMATH_GPT_convert_degrees_to_radians_l1751_175112


namespace NUMINAMATH_GPT_geometric_series_sum_l1751_175159

theorem geometric_series_sum :
  let a := (1/2 : ℚ)
  let r := (-1/3 : ℚ)
  let n := 7
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 547 / 1458 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1751_175159


namespace NUMINAMATH_GPT_pages_left_to_read_l1751_175128

-- Define the conditions
def total_pages : ℕ := 400
def percent_read : ℚ := 20 / 100
def pages_read := total_pages * percent_read

-- Define the question as a theorem
theorem pages_left_to_read (total_pages : ℕ) (percent_read : ℚ) (pages_read : ℚ) : ℚ :=
total_pages - pages_read

-- Assert the correct answer
example : pages_left_to_read total_pages percent_read pages_read = 320 := 
by
  sorry

end NUMINAMATH_GPT_pages_left_to_read_l1751_175128


namespace NUMINAMATH_GPT_pascal_triangle_ratios_l1751_175102
open Nat

theorem pascal_triangle_ratios :
  ∃ n r : ℕ, 
  (choose n r) * 4 = (choose n (r + 1)) * 3 ∧ 
  (choose n (r + 1)) * 3 = (choose n (r + 2)) * 4 ∧ 
  n = 34 :=
by
  sorry

end NUMINAMATH_GPT_pascal_triangle_ratios_l1751_175102


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1751_175121

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = Real.pi^0 + 1) :
  (1 - 2 / (x + 1)) / ((x^2 - 1) / (2 * x + 2)) = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1751_175121


namespace NUMINAMATH_GPT_ellipse_same_foci_l1751_175187

-- Definitions related to the problem
variables {x y p q : ℝ}

-- Condition
def represents_hyperbola (p q : ℝ) : Prop :=
  (p * q > 0) ∧ (∀ x y : ℝ, (x^2 / -p + y^2 / q = 1))

-- Proof Statement
theorem ellipse_same_foci (p q : ℝ) (hpq : p * q > 0)
  (h : ∀ x y : ℝ, x^2 / -p + y^2 / q = 1) :
  (∀ x y : ℝ, x^2 / (2*p + q) + y^2 / p = -1) :=
sorry -- Proof goes here

end NUMINAMATH_GPT_ellipse_same_foci_l1751_175187


namespace NUMINAMATH_GPT_Democrats_in_House_l1751_175155

-- Let D be the number of Democrats.
-- Let R be the number of Republicans.
-- Given conditions.

def Democrats (D R : ℕ) : Prop := 
  D + R = 434 ∧ R = D + 30

theorem Democrats_in_House : ∃ D, ∃ R, Democrats D R ∧ D = 202 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_Democrats_in_House_l1751_175155


namespace NUMINAMATH_GPT_min_value_expression_l1751_175124

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : x + y + z = 3) (h2 : z = (x + y) / 2) : 
  (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_min_value_expression_l1751_175124


namespace NUMINAMATH_GPT_no_real_roots_iff_k_gt_1_div_4_l1751_175130

theorem no_real_roots_iff_k_gt_1_div_4 (k : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - x + k = 0)) ↔ k > 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_iff_k_gt_1_div_4_l1751_175130


namespace NUMINAMATH_GPT_min_value_a_4b_l1751_175129

theorem min_value_a_4b (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 / (a - 1) + 1 / (b - 1) = 1) : a + 4 * b = 14 := 
sorry

end NUMINAMATH_GPT_min_value_a_4b_l1751_175129


namespace NUMINAMATH_GPT_train_speed_approx_72_km_hr_l1751_175126

noncomputable def train_length : ℝ := 150
noncomputable def bridge_length : ℝ := 132
noncomputable def crossing_time : ℝ := 14.098872090232781
noncomputable def total_distance : ℝ := train_length + bridge_length
noncomputable def speed_m_s : ℝ := total_distance / crossing_time
noncomputable def conversion_factor : ℝ := 3.6
noncomputable def speed_km_hr : ℝ := speed_m_s * conversion_factor

theorem train_speed_approx_72_km_hr : abs (speed_km_hr - 72) < 0.01 :=
sorry

end NUMINAMATH_GPT_train_speed_approx_72_km_hr_l1751_175126


namespace NUMINAMATH_GPT_total_string_length_l1751_175180

theorem total_string_length 
  (circumference1 : ℝ) (height1 : ℝ) (loops1 : ℕ)
  (circumference2 : ℝ) (height2 : ℝ) (loops2 : ℕ)
  (h1 : circumference1 = 6) (h2 : height1 = 20) (h3 : loops1 = 5)
  (h4 : circumference2 = 3) (h5 : height2 = 10) (h6 : loops2 = 3)
  : (loops1 * Real.sqrt (circumference1 ^ 2 + (height1 / loops1) ^ 2) + loops2 * Real.sqrt (circumference2 ^ 2 + (height2 / loops2) ^ 2)) = (5 * Real.sqrt 52 + 3 * Real.sqrt 19.89) := 
by {
  sorry
}

end NUMINAMATH_GPT_total_string_length_l1751_175180


namespace NUMINAMATH_GPT_measure_of_angle_D_l1751_175132

def angle_A := 95 -- Defined in step b)
def angle_B := angle_A
def angle_C := angle_A
def angle_D := angle_A + 50
def angle_E := angle_D
def angle_F := angle_D

theorem measure_of_angle_D (x : ℕ) (y : ℕ) :
  (angle_A = x) ∧ (angle_D = y) ∧ (y = x + 50) ∧ (3 * x + 3 * y = 720) → y = 145 :=
by
  intros
  sorry

end NUMINAMATH_GPT_measure_of_angle_D_l1751_175132


namespace NUMINAMATH_GPT_sum_sequence_l1751_175147

noncomputable def sum_first_n_minus_1_terms (n : ℕ) : ℕ :=
  (2^n - n - 1)

theorem sum_sequence (n : ℕ) : 
  sum_first_n_minus_1_terms n = (2^n - n - 1) :=
by
  sorry 

end NUMINAMATH_GPT_sum_sequence_l1751_175147


namespace NUMINAMATH_GPT_distinct_placements_of_two_pieces_l1751_175105

-- Definitions of the conditions
def grid_size : ℕ := 3
def cell_count : ℕ := grid_size * grid_size
def pieces_count : ℕ := 2

-- The theorem statement
theorem distinct_placements_of_two_pieces : 
  (number_of_distinct_placements : ℕ) = 10 := by
  -- Proof goes here with calculations and accounting for symmetry
  sorry

end NUMINAMATH_GPT_distinct_placements_of_two_pieces_l1751_175105


namespace NUMINAMATH_GPT_sum_squares_l1751_175176

theorem sum_squares (a b c : ℝ) (h1 : a + b + c = 22) (h2 : a * b + b * c + c * a = 116) : 
  (a^2 + b^2 + c^2 = 252) :=
by
  sorry

end NUMINAMATH_GPT_sum_squares_l1751_175176


namespace NUMINAMATH_GPT_sixth_power_of_sqrt_l1751_175137

variable (x : ℝ)
axiom h1 : x = Real.sqrt (2 + Real.sqrt 2)

theorem sixth_power_of_sqrt : x^6 = 16 + 10 * Real.sqrt 2 :=
by {
    sorry
}

end NUMINAMATH_GPT_sixth_power_of_sqrt_l1751_175137


namespace NUMINAMATH_GPT_quotient_remainder_difference_l1751_175196

theorem quotient_remainder_difference (N Q P R k : ℕ) (h1 : N = 75) (h2 : N = 5 * Q) (h3 : N = 34 * P + R) (h4 : Q = R + k) (h5 : k > 0) :
  Q - R = 8 :=
sorry

end NUMINAMATH_GPT_quotient_remainder_difference_l1751_175196


namespace NUMINAMATH_GPT_ratio_xy_half_l1751_175134

noncomputable def common_ratio_k (x y z : ℝ) (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (y + 9) / (z - 3) = (x + 5) / (z - 5)) : ℝ := sorry

theorem ratio_xy_half (x y z k : ℝ) (h : (x + 4) / 2 = (y + 9) / (z - 3) ∧ (y + 9) / (z - 3) = (x + 5) / (z - 5)) :
  ∃ k, (x + 4) = 2 * k ∧ (y + 9) = k * (z - 3) ∧ (x + 5) = k * (z - 5) → (x / y) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_xy_half_l1751_175134


namespace NUMINAMATH_GPT_price_for_70_cans_is_correct_l1751_175154

def regular_price_per_can : ℝ := 0.55
def discount_percentage : ℝ := 0.25
def purchase_quantity : ℕ := 70

def discount_per_can : ℝ := discount_percentage * regular_price_per_can
def discounted_price_per_can : ℝ := regular_price_per_can - discount_per_can

def price_for_72_cans : ℝ := 72 * discounted_price_per_can
def price_for_2_cans : ℝ := 2 * discounted_price_per_can

def final_price_for_70_cans : ℝ := price_for_72_cans - price_for_2_cans

theorem price_for_70_cans_is_correct
    (regular_price_per_can : ℝ := 0.55)
    (discount_percentage : ℝ := 0.25)
    (purchase_quantity : ℕ := 70)
    (disc_per_can : ℝ := discount_percentage * regular_price_per_can)
    (disc_price_per_can : ℝ := regular_price_per_can - disc_per_can)
    (price_72_cans : ℝ := 72 * disc_price_per_can)
    (price_2_cans : ℝ := 2 * disc_price_per_can):
    final_price_for_70_cans = 28.875 :=
by
  sorry

end NUMINAMATH_GPT_price_for_70_cans_is_correct_l1751_175154


namespace NUMINAMATH_GPT_find_first_term_arithmetic_progression_l1751_175195

theorem find_first_term_arithmetic_progression
  (a1 a2 a3 : ℝ)
  (h1 : a1 + a2 + a3 = 12)
  (h2 : a1 * a2 * a3 = 48)
  (h3 : a2 = a1 + d)
  (h4 : a3 = a1 + 2 * d)
  (h5 : a1 < a2 ∧ a2 < a3) :
  a1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_first_term_arithmetic_progression_l1751_175195


namespace NUMINAMATH_GPT_total_adults_wearing_hats_l1751_175148

theorem total_adults_wearing_hats (total_adults : ℕ) (men_percentage : ℝ) (men_hats_percentage : ℝ) 
  (women_hats_percentage : ℝ) (total_men_wearing_hats : ℕ) (total_women_wearing_hats : ℕ) : 
  (total_adults = 1200) ∧ (men_percentage = 0.60) ∧ (men_hats_percentage = 0.15) 
  ∧ (women_hats_percentage = 0.10)
     → total_men_wearing_hats + total_women_wearing_hats = 156 :=
by
  -- Definitions
  let total_men := total_adults * men_percentage
  let total_women := total_adults - total_men
  let men_wearing_hats := total_men * men_hats_percentage
  let women_wearing_hats := total_women * women_hats_percentage
  sorry

end NUMINAMATH_GPT_total_adults_wearing_hats_l1751_175148


namespace NUMINAMATH_GPT_jen_age_when_son_born_l1751_175161

theorem jen_age_when_son_born (S : ℕ) (Jen_present_age : ℕ) 
  (h1 : S = 16) (h2 : Jen_present_age = 3 * S - 7) : 
  Jen_present_age - S = 25 :=
by {
  sorry -- Proof would be here, but it is not required as per the instructions.
}

end NUMINAMATH_GPT_jen_age_when_son_born_l1751_175161


namespace NUMINAMATH_GPT_point_on_inverse_proportion_function_l1751_175152

theorem point_on_inverse_proportion_function :
  ∀ (x y k : ℝ), k ≠ 0 ∧ y = k / x ∧ (2, -3) = (2, -(3 : ℝ)) → (x, y) = (-2, 3) → (y = -6 / x) :=
sorry

end NUMINAMATH_GPT_point_on_inverse_proportion_function_l1751_175152


namespace NUMINAMATH_GPT_complement_correct_l1751_175135

universe u

-- We define sets A and B
def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {1, 3, 5}

-- Define the complement of B with respect to A
def complement (A B : Set ℕ) : Set ℕ := {x ∈ A | x ∉ B}

-- The theorem we need to prove
theorem complement_correct : complement A B = {2, 4} := 
  sorry

end NUMINAMATH_GPT_complement_correct_l1751_175135


namespace NUMINAMATH_GPT_symmetric_axis_of_g_l1751_175123

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6))

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - (Real.pi / 6))

theorem symmetric_axis_of_g :
  ∃ k : ℤ, (∃ x : ℝ, g x = 2 * Real.sin (k * Real.pi + (Real.pi / 2)) ∧ x = (k * Real.pi) / 2 + (Real.pi / 3)) :=
sorry

end NUMINAMATH_GPT_symmetric_axis_of_g_l1751_175123


namespace NUMINAMATH_GPT_percent_gain_on_transaction_l1751_175190

theorem percent_gain_on_transaction
  (c : ℝ) -- cost per sheep
  (price_750_sold : ℝ := 800 * c) -- price at which 750 sheep were sold in total
  (price_per_sheep_750 : ℝ := price_750_sold / 750)
  (price_per_sheep_50 : ℝ := 1.1 * price_per_sheep_750)
  (revenue_750 : ℝ := price_per_sheep_750 * 750)
  (revenue_50 : ℝ := price_per_sheep_50 * 50)
  (total_revenue : ℝ := revenue_750 + revenue_50)
  (total_cost : ℝ := 800 * c)
  (profit : ℝ := total_revenue - total_cost)
  (percent_gain : ℝ := (profit / total_cost) * 100) :
  percent_gain = 14 :=
sorry

end NUMINAMATH_GPT_percent_gain_on_transaction_l1751_175190


namespace NUMINAMATH_GPT_division_of_decimals_l1751_175101

theorem division_of_decimals : 0.18 / 0.003 = 60 :=
by
  sorry

end NUMINAMATH_GPT_division_of_decimals_l1751_175101


namespace NUMINAMATH_GPT_remainder_of_7529_div_by_9_is_not_divisible_by_11_l1751_175166

theorem remainder_of_7529_div_by_9 : 7529 % 9 = 5 := by
  sorry

theorem is_not_divisible_by_11 : ¬ (7529 % 11 = 0) := by
  sorry

end NUMINAMATH_GPT_remainder_of_7529_div_by_9_is_not_divisible_by_11_l1751_175166


namespace NUMINAMATH_GPT_find_number_l1751_175185

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 13) : x = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1751_175185


namespace NUMINAMATH_GPT_set_union_example_l1751_175165

open Set

/-- Given sets A = {1, 2, 3} and B = {-1, 1}, prove that A ∪ B = {-1, 1, 2, 3} -/
theorem set_union_example : 
  let A := ({1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  A ∪ B = ({-1, 1, 2, 3} : Set ℤ) :=
by
  let A := ({1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  show A ∪ B = ({-1, 1, 2, 3} : Set ℤ)
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_set_union_example_l1751_175165


namespace NUMINAMATH_GPT_graph_symmetry_l1751_175156

theorem graph_symmetry (f : ℝ → ℝ) : 
  ∀ x : ℝ, f (x - 1) = f (-(x - 1)) ↔ x = 1 :=
by 
  sorry

end NUMINAMATH_GPT_graph_symmetry_l1751_175156


namespace NUMINAMATH_GPT_possible_integer_roots_l1751_175182

-- Define the general polynomial
def polynomial (b2 b1 : ℤ) (x : ℤ) : ℤ := x ^ 3 + b2 * x ^ 2 + b1 * x - 30

-- Statement: Prove the set of possible integer roots includes exactly the divisors of -30
theorem possible_integer_roots (b2 b1 : ℤ) :
  {r : ℤ | polynomial b2 b1 r = 0} = 
  {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30} :=
sorry

end NUMINAMATH_GPT_possible_integer_roots_l1751_175182


namespace NUMINAMATH_GPT_solve_x4_minus_16_eq_0_l1751_175115

open Complex  -- Open the complex number notation

theorem solve_x4_minus_16_eq_0 :
  {x : ℂ | x^4 = 16} = {2, -2, 2 * Complex.I, -2 * Complex.I} :=
by sorry

end NUMINAMATH_GPT_solve_x4_minus_16_eq_0_l1751_175115


namespace NUMINAMATH_GPT_solution_set_l1751_175127

noncomputable def f : ℝ → ℝ := sorry
axiom f'_lt_one_third (x : ℝ) : deriv f x < 1 / 3
axiom f_at_two : f 2 = 1

theorem solution_set : {x : ℝ | 0 < x ∧ x < 4} = {x : ℝ | f (Real.logb 2 x) > (Real.logb 2 x + 1) / 3} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1751_175127


namespace NUMINAMATH_GPT_inversely_proportional_ratios_l1751_175104

theorem inversely_proportional_ratios (x y x₁ x₂ y₁ y₂ : ℝ) (hx_inv : ∀ x y, x * y = 1)
  (hx_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 :=
sorry

end NUMINAMATH_GPT_inversely_proportional_ratios_l1751_175104


namespace NUMINAMATH_GPT_intersection_proof_l1751_175120

-- Definitions based on conditions
def circle1 (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 10) ^ 2 = 50
def circle2 (x y : ℝ) : Prop := x ^ 2 + y ^ 2 + 2 * (x - y) - 18 = 0

-- Correct answer tuple
def intersection_points : (ℝ × ℝ) × (ℝ × ℝ) := ((3, 3), (-3, 5))

-- The goal statement to prove
theorem intersection_proof :
  (circle1 3 3 ∧ circle2 3 3) ∧ (circle1 (-3) 5 ∧ circle2 (-3) 5) :=
by
  sorry

end NUMINAMATH_GPT_intersection_proof_l1751_175120


namespace NUMINAMATH_GPT_least_addition_for_divisibility_least_subtraction_for_divisibility_least_addition_for_common_divisibility_l1751_175157

theorem least_addition_for_divisibility (n : ℕ) : (1100 + n) % 53 = 0 ↔ n = 9 := by
  sorry

theorem least_subtraction_for_divisibility (n : ℕ) : (1100 - n) % 71 = 0 ↔ n = 0 := by
  sorry

theorem least_addition_for_common_divisibility (X : ℕ) : (1100 + X) % (Nat.lcm 19 43) = 0 ∧ X = 534 := by
  sorry

end NUMINAMATH_GPT_least_addition_for_divisibility_least_subtraction_for_divisibility_least_addition_for_common_divisibility_l1751_175157


namespace NUMINAMATH_GPT_value_of_t_l1751_175162

theorem value_of_t (t : ℝ) (x y : ℝ) (h1 : x = 1 - 2 * t) (h2 : y = 2 * t - 2) (h3 : x = y) : t = 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_value_of_t_l1751_175162


namespace NUMINAMATH_GPT_cookies_per_bag_l1751_175186

-- Definitions of the given conditions
def c1 := 23  -- number of chocolate chip cookies
def c2 := 25  -- number of oatmeal cookies
def b := 8    -- number of baggies

-- Statement to prove
theorem cookies_per_bag : (c1 + c2) / b = 6 :=
by 
  sorry

end NUMINAMATH_GPT_cookies_per_bag_l1751_175186


namespace NUMINAMATH_GPT_parabola_equation_l1751_175189

-- Define the conditions and the claim
theorem parabola_equation (p : ℝ) (hp : p > 0) (h_symmetry : -p / 2 = -1 / 2) : 
  (∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = 2 * y) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_equation_l1751_175189


namespace NUMINAMATH_GPT_max_visible_cubes_from_point_l1751_175117

theorem max_visible_cubes_from_point (n : ℕ) (h : n = 12) :
  let total_cubes := n^3
  let face_cube_count := n * n
  let edge_count := n
  let visible_face_count := 3 * face_cube_count
  let double_counted_edges := 3 * (edge_count - 1)
  let corner_cube_count := 1
  visible_face_count - double_counted_edges + corner_cube_count = 400 := by
  sorry

end NUMINAMATH_GPT_max_visible_cubes_from_point_l1751_175117


namespace NUMINAMATH_GPT_woman_works_finish_days_l1751_175107

theorem woman_works_finish_days (M W : ℝ) 
  (hm_work : ∀ n : ℝ, n * M = 1 / 100)
  (hw_work : ∀ men women : ℝ, (10 * M + 15 * women) * 6 = 1) :
  W = 1 / 225 :=
by
  have man_work := hm_work 1
  have woman_work := hw_work 10 W
  sorry

end NUMINAMATH_GPT_woman_works_finish_days_l1751_175107


namespace NUMINAMATH_GPT_collinear_points_b_value_l1751_175153

theorem collinear_points_b_value :
  ∃ b : ℝ, (3 - (-2)) * (11 - b) = (8 - 3) * (1 - b) → b = -9 :=
by
  sorry

end NUMINAMATH_GPT_collinear_points_b_value_l1751_175153


namespace NUMINAMATH_GPT_isabella_initial_hair_length_l1751_175151

theorem isabella_initial_hair_length
  (final_length : ℕ)
  (growth_over_year : ℕ)
  (initial_length : ℕ)
  (h_final : final_length = 24)
  (h_growth : growth_over_year = 6)
  (h_initial : initial_length = 18) :
  initial_length + growth_over_year = final_length := 
by 
  sorry

end NUMINAMATH_GPT_isabella_initial_hair_length_l1751_175151


namespace NUMINAMATH_GPT_clients_using_radio_l1751_175160

theorem clients_using_radio (total_clients T R M TR TM RM TRM : ℕ)
  (h1 : total_clients = 180)
  (h2 : T = 115)
  (h3 : M = 130)
  (h4 : TR = 75)
  (h5 : TM = 85)
  (h6 : RM = 95)
  (h7 : TRM = 80) : R = 30 :=
by
  -- Using Inclusion-Exclusion Principle
  have h : total_clients = T + R + M - TR - TM - RM + TRM :=
    sorry  -- Proof of Inclusion-Exclusion principle for these sets
  rw [h1, h2, h3, h4, h5, h6, h7] at h
  -- Solve for R
  sorry

end NUMINAMATH_GPT_clients_using_radio_l1751_175160


namespace NUMINAMATH_GPT_max_dot_product_between_ellipses_l1751_175142

noncomputable def ellipse1 (x y : ℝ) : Prop := (x^2 / 25 + y^2 / 9 = 1)
noncomputable def ellipse2 (x y : ℝ) : Prop := (x^2 / 9 + y^2 / 9 = 1)

theorem max_dot_product_between_ellipses :
  ∀ (M N : ℝ × ℝ),
    ellipse1 M.1 M.2 →
    ellipse2 N.1 N.2 →
    ∃ θ φ : ℝ,
      M = (5 * Real.cos θ, 3 * Real.sin θ) ∧
      N = (3 * Real.cos φ, 3 * Real.sin φ) ∧
      (15 * Real.cos θ * Real.cos φ + 9 * Real.sin θ * Real.sin φ ≤ 15) :=
by
  sorry

end NUMINAMATH_GPT_max_dot_product_between_ellipses_l1751_175142


namespace NUMINAMATH_GPT_sum_zero_quotient_l1751_175150

   theorem sum_zero_quotient (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ x ≠ z) (h_sum_zero : x + y + z = 0) :
     (xy + yz + zx) / (x^2 + y^2 + z^2) = -1 / 2 :=
   by
     sorry
   
end NUMINAMATH_GPT_sum_zero_quotient_l1751_175150


namespace NUMINAMATH_GPT_no_solution_value_of_m_l1751_175139

theorem no_solution_value_of_m (m : ℤ) : ¬ ∃ x : ℤ, x ≠ 3 ∧ (x - 5) * (x - 3) = (m * (x - 3) + 2 * (x - 3) * (x - 3)) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_value_of_m_l1751_175139


namespace NUMINAMATH_GPT_quadratic_completion_l1751_175171

noncomputable def sum_of_r_s (r s : ℝ) : ℝ := r + s

theorem quadratic_completion (x r s : ℝ) (h : 16 * x^2 - 64 * x - 144 = 0) :
  ((x + r)^2 = s) → sum_of_r_s r s = -7 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_completion_l1751_175171


namespace NUMINAMATH_GPT_complex_roots_circle_radius_l1751_175188

theorem complex_roots_circle_radius (z : ℂ) (h : (z + 2)^4 = 16 * z^4) :
  ∃ r : ℝ, (∀ z, (z + 2)^4 = 16 * z^4 → (z - (2/3))^2 + y^2 = r) ∧ r = 1 :=
sorry

end NUMINAMATH_GPT_complex_roots_circle_radius_l1751_175188


namespace NUMINAMATH_GPT_years_taught_third_grade_l1751_175122

def total_years : ℕ := 26
def years_taught_second_grade : ℕ := 8

theorem years_taught_third_grade :
  total_years - years_taught_second_grade = 18 :=
by {
  -- Subtract the years taught second grade from the total years
  -- Exact the result
  sorry
}

end NUMINAMATH_GPT_years_taught_third_grade_l1751_175122


namespace NUMINAMATH_GPT_residue_mod_neg_935_mod_24_l1751_175178

theorem residue_mod_neg_935_mod_24 : (-935) % 24 = 1 :=
by
  sorry

end NUMINAMATH_GPT_residue_mod_neg_935_mod_24_l1751_175178


namespace NUMINAMATH_GPT_order_of_a_b_c_l1751_175146

noncomputable def a : ℝ := (Real.log (Real.sqrt 2)) / 2
noncomputable def b : ℝ := Real.log 3 / 6
noncomputable def c : ℝ := 1 / (2 * Real.exp 1)

theorem order_of_a_b_c : c > b ∧ b > a := by
  sorry

end NUMINAMATH_GPT_order_of_a_b_c_l1751_175146


namespace NUMINAMATH_GPT_find_constants_l1751_175199

theorem find_constants (a b c : ℝ) (h_neg : a < 0) (h_amp : |a| = 3) (h_period : b > 0 ∧ (2 * π / b) = 8 * π) : 
a = -3 ∧ b = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l1751_175199


namespace NUMINAMATH_GPT_susan_books_l1751_175149

theorem susan_books (S : ℕ) (h1 : S + 4 * S = 3000) : S = 600 :=
by 
  sorry

end NUMINAMATH_GPT_susan_books_l1751_175149


namespace NUMINAMATH_GPT_product_sequence_eq_l1751_175197

theorem product_sequence_eq :
  let seq := [ (1 : ℚ) / 2, 4 / 1, 1 / 8, 16 / 1, 1 / 32, 64 / 1,
               1 / 128, 256 / 1, 1 / 512, 1024 / 1, 1 / 2048, 4096 / 1 ]
  (seq.prod) * (3 / 4) = 1536 := by 
  -- expand and simplify the series of products
  sorry 

end NUMINAMATH_GPT_product_sequence_eq_l1751_175197


namespace NUMINAMATH_GPT_marbles_problem_l1751_175136

theorem marbles_problem :
  let red_marbles := 20
  let green_marbles := 3 * red_marbles
  let yellow_marbles := 0.20 * green_marbles
  let total_marbles := green_marbles + 3 * green_marbles
  total_marbles - (red_marbles + green_marbles + yellow_marbles) = 148 := by
  sorry

end NUMINAMATH_GPT_marbles_problem_l1751_175136


namespace NUMINAMATH_GPT_five_coins_all_heads_or_tails_l1751_175108

theorem five_coins_all_heads_or_tails : 
  (1 / 2) ^ 5 + (1 / 2) ^ 5 = 1 / 16 := 
by 
  sorry

end NUMINAMATH_GPT_five_coins_all_heads_or_tails_l1751_175108


namespace NUMINAMATH_GPT_p_6_is_126_l1751_175143

noncomputable def p (x : ℝ) : ℝ := sorry

axiom h1 : p 1 = 1
axiom h2 : p 2 = 2
axiom h3 : p 3 = 3
axiom h4 : p 4 = 4
axiom h5 : p 5 = 5

theorem p_6_is_126 : p 6 = 126 := sorry

end NUMINAMATH_GPT_p_6_is_126_l1751_175143


namespace NUMINAMATH_GPT_proof_problem_l1751_175140

noncomputable def problem : Prop :=
  ∃ (m n l : Type) (α β : Type) 
    (is_line : ∀ x, x = m ∨ x = n ∨ x = l)
    (is_plane : ∀ x, x = α ∨ x = β)
    (perpendicular : ∀ (l α : Type), Prop)
    (parallel : ∀ (l α : Type), Prop)
    (belongs_to : ∀ (l α : Type), Prop),
    (parallel l α → ∃ l', parallel l' α ∧ parallel l l') ∧
    (perpendicular m α ∧ perpendicular m β → parallel α β)

theorem proof_problem : problem :=
sorry

end NUMINAMATH_GPT_proof_problem_l1751_175140


namespace NUMINAMATH_GPT_family_ages_l1751_175192

theorem family_ages :
  ∃ (x j b m F M : ℕ), 
    (b = j - x) ∧
    (m = j - 2 * x) ∧
    (j * b = F) ∧
    (b * m = M) ∧
    (j + b + m + F + M = 90) ∧
    (F = M + x ∨ F = M - x) ∧
    (j = 6) ∧ 
    (b = 6) ∧ 
    (m = 6) ∧ 
    (F = 36) ∧ 
    (M = 36) :=
sorry

end NUMINAMATH_GPT_family_ages_l1751_175192


namespace NUMINAMATH_GPT_min_value_x_l1751_175191

theorem min_value_x (a b x : ℝ) (ha : 0 < a) (hb : 0 < b)
(hcond : 4 * a + b * (1 - a) = 0)
(hineq : ∀ a b, 0 < a → 0 < b → 4 * a + b * (1 - a) = 0 → (1 / (a ^ 2) + 16 / (b ^ 2) ≥ 1 + x / 2 - x ^ 2)) :
  x = 1 :=
sorry

end NUMINAMATH_GPT_min_value_x_l1751_175191


namespace NUMINAMATH_GPT_speed_in_first_hour_l1751_175131

variable (x : ℕ)
-- Conditions: 
-- The speed of the car in the second hour:
def speed_in_second_hour : ℕ := 30
-- The average speed of the car:
def average_speed : ℕ := 60
-- The total time traveled:
def total_time : ℕ := 2

-- Proof problem: Prove that the speed of the car in the first hour is 90 km/h.
theorem speed_in_first_hour : x + speed_in_second_hour = average_speed * total_time → x = 90 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_speed_in_first_hour_l1751_175131
