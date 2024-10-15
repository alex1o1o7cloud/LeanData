import Mathlib

namespace NUMINAMATH_CALUDE_therapy_hours_calculation_l1051_105198

/-- Represents the pricing structure and patient charges for a psychologist's therapy sessions -/
structure TherapyPricing where
  first_hour : ℕ  -- Cost of the first hour
  additional_hour : ℕ  -- Cost of each additional hour
  first_patient_total : ℕ  -- Total charge for the first patient
  second_patient_total : ℕ  -- Total charge for the second patient (3 hours)

/-- Calculates the number of therapy hours for the first patient given the pricing structure -/
def calculate_hours (pricing : TherapyPricing) : ℕ :=
  sorry

/-- Theorem stating the conditions and the result to be proved -/
theorem therapy_hours_calculation (pricing : TherapyPricing) 
  (h1 : pricing.first_hour = pricing.additional_hour + 20)
  (h2 : pricing.second_patient_total = pricing.first_hour + 2 * pricing.additional_hour)
  (h3 : pricing.second_patient_total = 188)
  (h4 : pricing.first_patient_total = 300) :
  calculate_hours pricing = 5 :=
sorry

end NUMINAMATH_CALUDE_therapy_hours_calculation_l1051_105198


namespace NUMINAMATH_CALUDE_xy_value_l1051_105166

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1051_105166


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l1051_105155

theorem min_sum_of_squares (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  ∃ (min : ℝ), min = 50 ∧ ∀ (a b : ℝ), (a + 5) * (b - 5) = 0 → a^2 + b^2 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l1051_105155


namespace NUMINAMATH_CALUDE_number_count_l1051_105110

theorem number_count (average : ℝ) (avg1 avg2 avg3 : ℝ) : 
  average = 6.40 →
  avg1 = 6.2 →
  avg2 = 6.1 →
  avg3 = 6.9 →
  (2 * avg1 + 2 * avg2 + 2 * avg3) / 6 = average →
  6 = (2 * avg1 + 2 * avg2 + 2 * avg3) / average :=
by sorry

end NUMINAMATH_CALUDE_number_count_l1051_105110


namespace NUMINAMATH_CALUDE_smallest_sum_with_lcm_2012_l1051_105104

theorem smallest_sum_with_lcm_2012 (a b c d e f g : ℕ) : 
  (Nat.lcm a (Nat.lcm b (Nat.lcm c (Nat.lcm d (Nat.lcm e (Nat.lcm f g)))))) = 2012 → 
  a + b + c + d + e + f + g ≥ 512 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_with_lcm_2012_l1051_105104


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1051_105102

theorem quadratic_root_problem (m : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - m * x + 6 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 2 * y^2 - m * y + 6 = 0 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1051_105102


namespace NUMINAMATH_CALUDE_train_speed_l1051_105160

/-- The speed of a train given crossing times and platform length -/
theorem train_speed (platform_length : ℝ) (platform_crossing_time : ℝ) (man_crossing_time : ℝ) :
  platform_length = 300 →
  platform_crossing_time = 33 →
  man_crossing_time = 18 →
  (platform_length / (platform_crossing_time - man_crossing_time)) * 3.6 = 72 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1051_105160


namespace NUMINAMATH_CALUDE_sum_of_roots_l1051_105106

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 + 5*a = 1) 
  (hb : b^3 - 3*b^2 + 5*b = 5) : 
  a + b = 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1051_105106


namespace NUMINAMATH_CALUDE_power_fraction_simplification_l1051_105186

theorem power_fraction_simplification :
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_simplification_l1051_105186


namespace NUMINAMATH_CALUDE_triangle_inequality_squared_l1051_105137

theorem triangle_inequality_squared (a b c : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_squared_l1051_105137


namespace NUMINAMATH_CALUDE_integer_fraction_triples_l1051_105162

theorem integer_fraction_triples :
  ∀ a b c : ℕ+,
    (a = 1 ∧ b = 20 ∧ c = 1) ∨
    (a = 1 ∧ b = 4 ∧ c = 1) ∨
    (a = 3 ∧ b = 4 ∧ c = 1) ↔
    ∃ k : ℤ, (32 * a.val + 3 * b.val + 48 * c.val) = 4 * k * a.val * b.val * c.val := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_triples_l1051_105162


namespace NUMINAMATH_CALUDE_discriminant_neither_sufficient_nor_necessary_l1051_105109

/-- The function f(x) = ax^2 + bx + c --/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that the graph of f is always above the x-axis --/
def always_above (a b c : ℝ) : Prop :=
  ∀ x, f a b c x > 0

/-- The discriminant condition --/
def discriminant_condition (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

/-- The main theorem --/
theorem discriminant_neither_sufficient_nor_necessary :
  ¬(∀ a b c : ℝ, discriminant_condition a b c → always_above a b c) ∧
  ¬(∀ a b c : ℝ, always_above a b c → discriminant_condition a b c) := by
  sorry

end NUMINAMATH_CALUDE_discriminant_neither_sufficient_nor_necessary_l1051_105109


namespace NUMINAMATH_CALUDE_train_speed_is_45_km_per_hour_l1051_105175

-- Define the given parameters
def train_length : ℝ := 140
def bridge_length : ℝ := 235
def crossing_time : ℝ := 30

-- Define the conversion factor
def meters_per_second_to_km_per_hour : ℝ := 3.6

-- Theorem statement
theorem train_speed_is_45_km_per_hour :
  let total_distance := train_length + bridge_length
  let speed_in_meters_per_second := total_distance / crossing_time
  let speed_in_km_per_hour := speed_in_meters_per_second * meters_per_second_to_km_per_hour
  speed_in_km_per_hour = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_is_45_km_per_hour_l1051_105175


namespace NUMINAMATH_CALUDE_degree_of_h_is_4_l1051_105159

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 1 - 12*x + 3*x^2 - 4*x^3 + 5*x^4

/-- The polynomial g(x) -/
def g (x : ℝ) : ℝ := 3 - 2*x - 6*x^3 + 8*x^4 + x^5

/-- The combined polynomial h(x) = f(x) + c*g(x) -/
def h (c : ℝ) (x : ℝ) : ℝ := f x + c * g x

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

theorem degree_of_h_is_4 : degree (h 0) = 4 := by sorry

end NUMINAMATH_CALUDE_degree_of_h_is_4_l1051_105159


namespace NUMINAMATH_CALUDE_positive_real_inequality_l1051_105192

theorem positive_real_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x / (x + 2*y + 3*z)) + (y / (y + 2*z + 3*x)) + (z / (z + 2*x + 3*y)) ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l1051_105192


namespace NUMINAMATH_CALUDE_sprocket_production_l1051_105117

theorem sprocket_production (machine_p machine_q machine_a : ℕ → ℕ) : 
  (∃ t_q : ℕ, 
    machine_p (t_q + 10) = 550 ∧ 
    machine_q t_q = 550 ∧ 
    (∀ t, machine_q t = (11 * machine_a t) / 10)) → 
  machine_a 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_sprocket_production_l1051_105117


namespace NUMINAMATH_CALUDE_plane_perp_from_line_perp_and_parallel_l1051_105188

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (linePerpPlane : Line → Plane → Prop)

-- State the theorem
theorem plane_perp_from_line_perp_and_parallel
  (α β : Plane) (l : Line)
  (h1 : linePerpPlane l α)
  (h2 : parallel l β) :
  perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_plane_perp_from_line_perp_and_parallel_l1051_105188


namespace NUMINAMATH_CALUDE_stratified_sampling_ratio_l1051_105124

-- Define the total number of male and female students
def total_male : ℕ := 500
def total_female : ℕ := 400

-- Define the number of male students selected
def selected_male : ℕ := 25

-- Define the function to calculate the number of female students to be selected
def female_to_select : ℕ := (selected_male * total_female) / total_male

-- Theorem statement
theorem stratified_sampling_ratio :
  female_to_select = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_ratio_l1051_105124


namespace NUMINAMATH_CALUDE_mittens_per_box_l1051_105170

theorem mittens_per_box (num_boxes : ℕ) (scarves_per_box : ℕ) (total_clothing : ℕ) : 
  num_boxes = 8 → 
  scarves_per_box = 4 → 
  total_clothing = 80 → 
  (total_clothing - num_boxes * scarves_per_box) / num_boxes = 6 := by
sorry

end NUMINAMATH_CALUDE_mittens_per_box_l1051_105170


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1051_105178

theorem trigonometric_identities (α : Real) (h : Real.tan α = 2) :
  (Real.cos (π/2 + α) * Real.sin (3*π/2 - α)) / Real.tan (-π + α) = 1/5 ∧
  (1 + 3*Real.sin α*Real.cos α) / (Real.sin α^2 - 2*Real.cos α^2) = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1051_105178


namespace NUMINAMATH_CALUDE_min_coins_blind_pew_l1051_105136

/-- Represents the pirate's trunk with chests, boxes, and gold coins. -/
structure PirateTrunk where
  num_chests : Nat
  boxes_per_chest : Nat
  coins_per_box : Nat
  num_locks_opened : Nat

/-- Calculates the minimum number of gold coins that can be taken. -/
def min_coins_taken (trunk : PirateTrunk) : Nat :=
  let remaining_locks := trunk.num_locks_opened - 1 - trunk.num_chests
  remaining_locks * trunk.coins_per_box

/-- Theorem stating the minimum number of gold coins Blind Pew could take. -/
theorem min_coins_blind_pew :
  let trunk : PirateTrunk := {
    num_chests := 5,
    boxes_per_chest := 4,
    coins_per_box := 10,
    num_locks_opened := 9
  }
  min_coins_taken trunk = 30 := by
  sorry


end NUMINAMATH_CALUDE_min_coins_blind_pew_l1051_105136


namespace NUMINAMATH_CALUDE_cylinder_lateral_area_l1051_105150

/-- Given a cylinder with height 4 and base area 9π, its lateral area is 24π. -/
theorem cylinder_lateral_area (h : ℝ) (base_area : ℝ) :
  h = 4 → base_area = 9 * Real.pi → 2 * Real.pi * (Real.sqrt (base_area / Real.pi)) * h = 24 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_area_l1051_105150


namespace NUMINAMATH_CALUDE_extreme_values_and_maximum_b_l1051_105189

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / x * Real.exp x

noncomputable def g (a b x : ℝ) : ℝ := a * (x - 1) * Real.exp x - f a b x

theorem extreme_values_and_maximum_b :
  (∀ x : ℝ, x ≠ 0 → f 2 1 x ≤ 1 / Real.exp 1) ∧
  (∀ x : ℝ, x ≠ 0 → f 2 1 x ≥ 4 * Real.sqrt (Real.exp 1)) ∧
  (∃ x : ℝ, x ≠ 0 ∧ f 2 1 x = 1 / Real.exp 1) ∧
  (∃ x : ℝ, x ≠ 0 ∧ f 2 1 x = 4 * Real.sqrt (Real.exp 1)) ∧
  (∀ b : ℝ, (∀ x : ℝ, x > 0 → g 1 b x ≥ 1) → b ≤ -1 - 1 / Real.exp 1) ∧
  (∀ x : ℝ, x > 0 → g 1 (-1 - 1 / Real.exp 1) x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_extreme_values_and_maximum_b_l1051_105189


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l1051_105195

-- Define the function f(x) = x³ + 4x + 5
def f (x : ℝ) : ℝ := x^3 + 4*x + 5

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 + 4

-- Theorem: The y-intercept of the tangent line to f(x) at x = 1 is (0, 3)
theorem tangent_line_y_intercept :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  let b : ℝ := y₀ - m * x₀
  b = 3 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l1051_105195


namespace NUMINAMATH_CALUDE_inequality_solution_l1051_105133

theorem inequality_solution (a x : ℝ) : ax^2 - ax + x > 0 ↔
  (a = 0 ∧ x > 0) ∨
  (a = 1 ∧ x ≠ 0) ∨
  (a < 0 ∧ 0 < x ∧ x < 1 - 1/a) ∨
  (a > 1 ∧ (x < 0 ∨ x > 1 - 1/a)) ∨
  (0 < a ∧ a < 1 ∧ (x < 1 - 1/a ∨ x > 0)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1051_105133


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1051_105199

theorem inequality_solution_set (a : ℝ) (h : a > 1) :
  let f := fun x : ℝ => (a - 1) * x^2 - a * x + 1
  (a = 2 → {x : ℝ | f x > 0} = {x : ℝ | x ≠ 1}) ∧
  (1 < a ∧ a < 2 → {x : ℝ | f x > 0} = {x : ℝ | x < 1 ∨ x > 1/(a-1)}) ∧
  (a > 2 → {x : ℝ | f x > 0} = {x : ℝ | x < 1/(a-1) ∨ x > 1}) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1051_105199


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l1051_105123

theorem quadratic_root_implies_k (k : ℝ) : 
  ((k - 1) * 1^2 + k^2 - k = 0) → 
  (k - 1 ≠ 0) → 
  (k = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l1051_105123


namespace NUMINAMATH_CALUDE_max_x_value_l1051_105173

theorem max_x_value (x y z : ℝ) 
  (sum_eq : x + y + z = 7) 
  (sum_prod_eq : x * y + x * z + y * z = 11) : 
  x ≤ (7 + Real.sqrt 34) / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l1051_105173


namespace NUMINAMATH_CALUDE_converse_x_gt_abs_y_implies_x_gt_y_l1051_105153

theorem converse_x_gt_abs_y_implies_x_gt_y : ∀ x y : ℝ, x > |y| → x > y := by
  sorry

end NUMINAMATH_CALUDE_converse_x_gt_abs_y_implies_x_gt_y_l1051_105153


namespace NUMINAMATH_CALUDE_base7_equals_base10_l1051_105147

/-- Converts a number from base 7 to base 10 -/
def base7To10 (n : ℕ) : ℕ := sorry

/-- Represents a base-10 digit (0-9) -/
def Digit := {d : ℕ // d < 10}

theorem base7_equals_base10 (c d : Digit) :
  base7To10 764 = 400 + 10 * c.val + d.val →
  (c.val * d.val) / 20 = 6 / 5 := by sorry

end NUMINAMATH_CALUDE_base7_equals_base10_l1051_105147


namespace NUMINAMATH_CALUDE_total_selection_schemes_l1051_105151

/-- The number of elective courses in each category (physical education and art) -/
def num_courses_per_category : ℕ := 4

/-- The minimum number of courses a student can choose -/
def min_courses : ℕ := 2

/-- The maximum number of courses a student can choose -/
def max_courses : ℕ := 3

/-- The number of categories (physical education and art) -/
def num_categories : ℕ := 2

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: The total number of different course selection schemes is 64 -/
theorem total_selection_schemes : 
  (choose num_courses_per_category 1 * choose num_courses_per_category 1) + 
  (choose num_courses_per_category 2 * choose num_courses_per_category 1 + 
   choose num_courses_per_category 1 * choose num_courses_per_category 2) = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_selection_schemes_l1051_105151


namespace NUMINAMATH_CALUDE_inequality_proof_l1051_105116

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_inequality : a * b + b * c + c * a ≥ 1) :
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ Real.sqrt 3 / (a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1051_105116


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1051_105125

theorem remainder_divisibility (n : ℤ) : 
  (2 * n) % 7 = 4 → n % 7 = 2 := by
sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1051_105125


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1051_105193

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0) →
  m ∈ Set.Ioi 2 ∪ Set.Iio (-2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1051_105193


namespace NUMINAMATH_CALUDE_string_cheese_cost_is_10_cents_l1051_105112

/-- The cost of each piece of string cheese in cents -/
def string_cheese_cost (num_packs : ℕ) (cheeses_per_pack : ℕ) (total_cost : ℚ) : ℚ :=
  (total_cost * 100) / (num_packs * cheeses_per_pack)

/-- Theorem: The cost of each piece of string cheese is 10 cents -/
theorem string_cheese_cost_is_10_cents :
  string_cheese_cost 3 20 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_string_cheese_cost_is_10_cents_l1051_105112


namespace NUMINAMATH_CALUDE_length_of_PC_l1051_105197

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the conditions
def is_right_triangle_with_internal_right_angle (t : Triangle) : Prop :=
  -- Right angle at B
  (t.A.1 - t.B.1) * (t.C.1 - t.B.1) + (t.A.2 - t.B.2) * (t.C.2 - t.B.2) = 0 ∧
  -- ∠BPC = 90°
  (t.B.1 - t.P.1) * (t.C.1 - t.P.1) + (t.B.2 - t.P.2) * (t.C.2 - t.P.2) = 0

def satisfies_length_conditions (t : Triangle) : Prop :=
  -- PA = 12
  Real.sqrt ((t.A.1 - t.P.1)^2 + (t.A.2 - t.P.2)^2) = 12 ∧
  -- PB = 8
  Real.sqrt ((t.B.1 - t.P.1)^2 + (t.B.2 - t.P.2)^2) = 8

-- The theorem to prove
theorem length_of_PC (t : Triangle) 
  (h1 : is_right_triangle_with_internal_right_angle t)
  (h2 : satisfies_length_conditions t) :
  Real.sqrt ((t.C.1 - t.P.1)^2 + (t.C.2 - t.P.2)^2) = Real.sqrt 464 :=
sorry

end NUMINAMATH_CALUDE_length_of_PC_l1051_105197


namespace NUMINAMATH_CALUDE_star_difference_l1051_105132

def star (x y : ℝ) : ℝ := x * y - 3 * x + y

theorem star_difference : (star 5 8) - (star 8 5) = 12 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l1051_105132


namespace NUMINAMATH_CALUDE_yellow_to_red_ratio_l1051_105190

/-- Represents the number of marbles Beth has initially -/
def total_marbles : ℕ := 72

/-- Represents the number of colors of marbles -/
def num_colors : ℕ := 3

/-- Represents the number of red marbles Beth loses -/
def red_marbles_lost : ℕ := 5

/-- Represents the number of marbles Beth has left after losing some of each color -/
def marbles_left : ℕ := 42

/-- Theorem stating the ratio of yellow marbles lost to red marbles lost -/
theorem yellow_to_red_ratio :
  let initial_per_color := total_marbles / num_colors
  let blue_marbles_lost := 2 * red_marbles_lost
  let yellow_marbles_lost := initial_per_color - (marbles_left - (2 * initial_per_color - red_marbles_lost - blue_marbles_lost))
  yellow_marbles_lost / red_marbles_lost = 3 := by sorry

end NUMINAMATH_CALUDE_yellow_to_red_ratio_l1051_105190


namespace NUMINAMATH_CALUDE_count_integers_satisfying_conditions_l1051_105105

theorem count_integers_satisfying_conditions : 
  ∃! (S : Finset ℤ), 
    (∀ x ∈ S, ⌊Real.sqrt x⌋ = 8 ∧ x % 5 = 3) ∧ 
    (∀ x : ℤ, ⌊Real.sqrt x⌋ = 8 ∧ x % 5 = 3 → x ∈ S) ∧
    S.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_conditions_l1051_105105


namespace NUMINAMATH_CALUDE_probability_is_three_twentyfifths_l1051_105126

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x_max : ℝ
  y_max : ℝ

/-- A point in the 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point satisfies the condition x > 5y --/
def satisfies_condition (p : Point) : Prop :=
  p.x > 5 * p.y

/-- Calculate the probability of a randomly chosen point satisfying the condition --/
def probability_satisfies_condition (r : Rectangle) : ℝ :=
  sorry

/-- The main theorem --/
theorem probability_is_three_twentyfifths :
  let r := Rectangle.mk 3000 2500
  probability_satisfies_condition r = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_three_twentyfifths_l1051_105126


namespace NUMINAMATH_CALUDE_chord_length_implies_a_values_point_m_existence_implies_a_range_l1051_105101

-- Define the circle C
def C (a : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - a - 1)^2 = 9

-- Define the line l
def l (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the point A
def A : ℝ × ℝ := (3, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Part 1
theorem chord_length_implies_a_values (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, C a x₁ y₁ ∧ C a x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4) →
  a = -1 ∨ a = 3 :=
sorry

-- Part 2
theorem point_m_existence_implies_a_range (a : ℝ) :
  (∃ x y : ℝ, C a x y ∧ (x - 3)^2 + y^2 = 4 * (x^2 + y^2)) →
  (-1 - 5 * Real.sqrt 2 / 2 ≤ a ∧ a ≤ -1 - Real.sqrt 2 / 2) ∨
  (-1 + Real.sqrt 2 / 2 ≤ a ∧ a ≤ -1 + 5 * Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_chord_length_implies_a_values_point_m_existence_implies_a_range_l1051_105101


namespace NUMINAMATH_CALUDE_inequality_proof_l1051_105127

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) :
  x + y/2 + z/3 ≤ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1051_105127


namespace NUMINAMATH_CALUDE_total_animals_l1051_105156

theorem total_animals (num_pigs num_giraffes : ℕ) : 
  num_pigs = 7 → num_giraffes = 6 → num_pigs + num_giraffes = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_animals_l1051_105156


namespace NUMINAMATH_CALUDE_intersection_equality_implies_m_equals_five_l1051_105142

def A (m : ℝ) : Set ℝ := {-1, 3, m}
def B : Set ℝ := {3, 5}

theorem intersection_equality_implies_m_equals_five (m : ℝ) :
  B ∩ A m = B → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_m_equals_five_l1051_105142


namespace NUMINAMATH_CALUDE_square_sum_inequality_l1051_105174

theorem square_sum_inequality (a b c : ℝ) : 
  a^2 + b^2 + a*b + b*c + c*a < 0 → a^2 + b^2 < c^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_inequality_l1051_105174


namespace NUMINAMATH_CALUDE_smallest_solution_congruence_l1051_105183

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (5 * x) % 31 = 14 % 31 ∧
  ∀ (y : ℕ), y > 0 ∧ (5 * y) % 31 = 14 % 31 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_congruence_l1051_105183


namespace NUMINAMATH_CALUDE_min_people_to_complete_task_l1051_105191

/-- Proves the minimum number of people needed to complete a task on time -/
theorem min_people_to_complete_task
  (total_days : ℕ)
  (days_worked : ℕ)
  (initial_people : ℕ)
  (work_completed : ℚ)
  (h1 : total_days = 40)
  (h2 : days_worked = 10)
  (h3 : initial_people = 12)
  (h4 : work_completed = 2 / 5)
  (h5 : days_worked < total_days) :
  let remaining_days := total_days - days_worked
  let remaining_work := 1 - work_completed
  let work_rate_per_day := work_completed / days_worked / initial_people
  ⌈(remaining_work / (work_rate_per_day * remaining_days))⌉ = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_people_to_complete_task_l1051_105191


namespace NUMINAMATH_CALUDE_prism_volume_l1051_105118

theorem prism_volume (x y z : Real) (h : Real) :
  x = Real.sqrt 9 →
  y = Real.sqrt 9 →
  h = 6 →
  (1 / 2 : Real) * x * y * h = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l1051_105118


namespace NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l1051_105184

theorem x_positive_necessary_not_sufficient :
  (∀ x : ℝ, (x - 2) * (x - 4) < 0 → x > 0) ∧
  (∃ x : ℝ, x > 0 ∧ (x - 2) * (x - 4) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_positive_necessary_not_sufficient_l1051_105184


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1051_105164

theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 2*x + 1 = 0 ∧ a * y^2 - 2*y + 1 = 0) ↔ 
  (a < 1 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1051_105164


namespace NUMINAMATH_CALUDE_flag_arrangement_modulo_l1051_105154

/-- The number of distinguishable arrangements of flags on two poles -/
def M : ℕ :=
  let total_flags := 17
  let blue_flags := 9
  let red_flags := 8
  let slots_for_red := blue_flags + 1
  let ways_to_place_red := Nat.choose slots_for_red red_flags
  let initial_arrangements := (blue_flags + 1) * ways_to_place_red
  let invalid_cases := 2 * Nat.choose blue_flags red_flags
  initial_arrangements - invalid_cases

/-- Theorem stating that M is congruent to 432 modulo 1000 -/
theorem flag_arrangement_modulo :
  M % 1000 = 432 := by sorry

end NUMINAMATH_CALUDE_flag_arrangement_modulo_l1051_105154


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1051_105148

theorem magnitude_of_z (z : ℂ) (h : (Complex.I - 1) * z = (Complex.I + 1)^2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1051_105148


namespace NUMINAMATH_CALUDE_solution_system_l1051_105194

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 12)
  (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 10344 / 169 := by
sorry

end NUMINAMATH_CALUDE_solution_system_l1051_105194


namespace NUMINAMATH_CALUDE_count_numbers_with_digit_product_180_l1051_105176

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 9

def is_five_digit_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000

def digit_product (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.prod

def count_valid_numbers : ℕ := sorry

theorem count_numbers_with_digit_product_180 :
  count_valid_numbers = 360 := by sorry

end NUMINAMATH_CALUDE_count_numbers_with_digit_product_180_l1051_105176


namespace NUMINAMATH_CALUDE_max_value_sine_sum_l1051_105185

theorem max_value_sine_sum : 
  ∀ x : ℝ, 3 * Real.sin (x + π/9) + 5 * Real.sin (x + 4*π/9) ≤ 7 ∧ 
  ∃ x : ℝ, 3 * Real.sin (x + π/9) + 5 * Real.sin (x + 4*π/9) = 7 :=
sorry

end NUMINAMATH_CALUDE_max_value_sine_sum_l1051_105185


namespace NUMINAMATH_CALUDE_total_sand_donation_l1051_105139

-- Define the amounts of sand for each city
def city_A : ℚ := 16 + 1/2
def city_B : ℕ := 26
def city_C : ℚ := 24 + 1/2
def city_D : ℕ := 28

-- Theorem statement
theorem total_sand_donation :
  city_A + city_B + city_C + city_D = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_sand_donation_l1051_105139


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l1051_105181

theorem sum_of_coefficients_zero 
  (a b c d : ℝ) 
  (h : ∀ x : ℝ, (1 + x)^2 * (1 - x) = a + b*x + c*x^2 + d*x^3) : 
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l1051_105181


namespace NUMINAMATH_CALUDE_border_area_is_144_l1051_105120

/-- The area of the border of a framed rectangular photograph -/
def border_area (photo_height photo_width border_width : ℝ) : ℝ :=
  (photo_height + 2 * border_width) * (photo_width + 2 * border_width) - photo_height * photo_width

/-- Theorem: The area of the border of a framed rectangular photograph is 144 square inches -/
theorem border_area_is_144 :
  border_area 8 10 3 = 144 := by
  sorry

end NUMINAMATH_CALUDE_border_area_is_144_l1051_105120


namespace NUMINAMATH_CALUDE_length_of_segment_AB_is_10_l1051_105140

/-- Given point A with coordinates (2, -3, 5) and point B symmetrical to A with respect to the xy-plane,
    prove that the length of line segment AB is 10. -/
theorem length_of_segment_AB_is_10 :
  let A : ℝ × ℝ × ℝ := (2, -3, 5)
  let B : ℝ × ℝ × ℝ := (2, -3, -5)  -- B is symmetrical to A with respect to xy-plane
  ‖A - B‖ = 10 := by
  sorry

end NUMINAMATH_CALUDE_length_of_segment_AB_is_10_l1051_105140


namespace NUMINAMATH_CALUDE_median_is_82_l1051_105180

/-- Represents the list where each integer n (1 ≤ n ≤ 100) appears 2n times -/
def special_list : List ℕ := sorry

/-- The total number of elements in the special list -/
def total_elements : ℕ := sorry

/-- The median of the special list -/
def median_of_special_list : ℚ := sorry

/-- Theorem stating that the median of the special list is 82 -/
theorem median_is_82 : median_of_special_list = 82 := by sorry

end NUMINAMATH_CALUDE_median_is_82_l1051_105180


namespace NUMINAMATH_CALUDE_one_third_complex_point_l1051_105143

theorem one_third_complex_point (z₁ z₂ z : ℂ) :
  z₁ = -5 + 6*I →
  z₂ = 7 - 4*I →
  z = (1 - 1/3) * z₁ + 1/3 * z₂ →
  z = -1 + 8/3 * I :=
by sorry

end NUMINAMATH_CALUDE_one_third_complex_point_l1051_105143


namespace NUMINAMATH_CALUDE_min_wednesday_birthdays_l1051_105171

/-- Represents the number of employees with birthdays on each day of the week -/
structure BirthdayDistribution where
  wednesday : ℕ
  other : ℕ

/-- The conditions of the problem -/
def validDistribution (d : BirthdayDistribution) : Prop :=
  d.wednesday > d.other ∧
  d.wednesday + 6 * d.other = 50

/-- The theorem to prove -/
theorem min_wednesday_birthdays :
  ∀ d : BirthdayDistribution,
  validDistribution d →
  d.wednesday ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_min_wednesday_birthdays_l1051_105171


namespace NUMINAMATH_CALUDE_markeesha_friday_sales_l1051_105134

/-- Proves that Markeesha sold 30 boxes on Friday given the conditions of the problem -/
theorem markeesha_friday_sales : ∀ (friday : ℕ), 
  (∃ (saturday sunday : ℕ),
    saturday = 2 * friday ∧
    sunday = saturday - 15 ∧
    friday + saturday + sunday = 135) →
  friday = 30 := by
sorry

end NUMINAMATH_CALUDE_markeesha_friday_sales_l1051_105134


namespace NUMINAMATH_CALUDE_truck_calculation_l1051_105161

/-- The number of trucks initially requested to transport 60 tons of cargo,
    where reducing each truck's capacity by 0.5 tons required 4 additional trucks. -/
def initial_trucks : ℕ := 20

/-- The total cargo to be transported in tons. -/
def total_cargo : ℝ := 60

/-- The reduction in capacity per truck in tons. -/
def capacity_reduction : ℝ := 0.5

/-- The number of additional trucks required after capacity reduction. -/
def additional_trucks : ℕ := 4

theorem truck_calculation :
  initial_trucks * (total_cargo / initial_trucks - capacity_reduction) = 
  (initial_trucks + additional_trucks) * ((total_cargo / initial_trucks) - capacity_reduction) ∧
  (initial_trucks + additional_trucks) * ((total_cargo / initial_trucks) - capacity_reduction) = total_cargo :=
sorry

end NUMINAMATH_CALUDE_truck_calculation_l1051_105161


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1051_105149

/-- Given a number c that forms a geometric sequence when added to 20, 50, and 100, 
    the common ratio of this sequence is 5/3 -/
theorem geometric_sequence_ratio (c : ℝ) : 
  (∃ r : ℝ, (50 + c) / (20 + c) = r ∧ (100 + c) / (50 + c) = r) → 
  (50 + c) / (20 + c) = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1051_105149


namespace NUMINAMATH_CALUDE_range_of_t_l1051_105172

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ -1}
def B (t : ℝ) : Set ℝ := {y : ℝ | y ≥ t}

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem range_of_t (t : ℝ) :
  (∀ x ∈ A, f x ∈ B t) → t ≤ 0 := by
  sorry

-- Define the final result
def result : Set ℝ := {t : ℝ | t ≤ 0}

end NUMINAMATH_CALUDE_range_of_t_l1051_105172


namespace NUMINAMATH_CALUDE_john_learning_alphabets_l1051_105196

/-- The number of alphabets John is learning in the first group -/
def alphabets_learned : ℕ := 15 / 3

/-- The number of days it takes John to learn one alphabet -/
def days_per_alphabet : ℕ := 3

/-- The total number of days John needs to finish learning the alphabets -/
def total_days : ℕ := 15

theorem john_learning_alphabets :
  alphabets_learned = 5 :=
by sorry

end NUMINAMATH_CALUDE_john_learning_alphabets_l1051_105196


namespace NUMINAMATH_CALUDE_dark_integer_characterization_l1051_105121

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- Checks if a positive integer is of the form a999...999 -/
def isA999Form (n : ℕ+) : Prop := sorry

/-- A positive integer is shiny if it can be written as the sum of two integers
    with the same sum of digits -/
def isShiny (n : ℕ+) : Prop :=
  ∃ a b : ℕ, n = a + b ∧ sumOfDigits ⟨a, sorry⟩ = sumOfDigits ⟨b, sorry⟩

theorem dark_integer_characterization (n : ℕ+) :
  ¬isShiny n ↔ isA999Form n ∧ Odd (sumOfDigits n) := by sorry

end NUMINAMATH_CALUDE_dark_integer_characterization_l1051_105121


namespace NUMINAMATH_CALUDE_students_not_enrolled_l1051_105145

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 87) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : both = 9) : 
  total - (french + german - both) = 33 :=
by sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l1051_105145


namespace NUMINAMATH_CALUDE_equivalent_statements_l1051_105163

-- Define the propositions
variable (P Q : Prop)

-- State the theorem
theorem equivalent_statements : 
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) :=
sorry

end NUMINAMATH_CALUDE_equivalent_statements_l1051_105163


namespace NUMINAMATH_CALUDE_somu_age_problem_l1051_105182

/-- Represents the problem of finding when Somu was one-fifth of his father's age --/
theorem somu_age_problem (somu_age : ℕ) (father_age : ℕ) (years_ago : ℕ) : 
  somu_age = 20 →
  3 * somu_age = father_age →
  5 * (somu_age - years_ago) = father_age - years_ago →
  years_ago = 10 := by
  sorry

#check somu_age_problem

end NUMINAMATH_CALUDE_somu_age_problem_l1051_105182


namespace NUMINAMATH_CALUDE_stacy_paper_completion_time_l1051_105119

/-- The number of days Stacy has to complete her paper -/
def days_to_complete : ℕ := 66 / 11

/-- The total number of pages in Stacy's paper -/
def total_pages : ℕ := 66

/-- The number of pages Stacy has to write per day -/
def pages_per_day : ℕ := 11

theorem stacy_paper_completion_time :
  days_to_complete = 6 :=
by sorry

end NUMINAMATH_CALUDE_stacy_paper_completion_time_l1051_105119


namespace NUMINAMATH_CALUDE_lyle_percentage_l1051_105146

/-- Given a total number of chips and a ratio for division, 
    calculate the percentage of chips the second person receives. -/
def calculate_percentage (total_chips : ℕ) (ratio1 ratio2 : ℕ) : ℚ :=
  let total_parts := ratio1 + ratio2
  let chips_per_part := total_chips / total_parts
  let second_person_chips := ratio2 * chips_per_part
  (second_person_chips : ℚ) / total_chips * 100

/-- Theorem stating that given 100 chips divided in a 4:6 ratio, 
    the person with the larger share has 60% of the total chips. -/
theorem lyle_percentage : calculate_percentage 100 4 6 = 60 := by
  sorry

end NUMINAMATH_CALUDE_lyle_percentage_l1051_105146


namespace NUMINAMATH_CALUDE_system_solution_l1051_105168

theorem system_solution (w x y z : ℚ) 
  (eq1 : 2*w + x + y + z = 1)
  (eq2 : w + 2*x + y + z = 2)
  (eq3 : w + x + 2*y + z = 2)
  (eq4 : w + x + y + 2*z = 1) :
  w = -1/5 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1051_105168


namespace NUMINAMATH_CALUDE_sin_phi_value_l1051_105144

theorem sin_phi_value (φ : ℝ) : 
  (∀ x, 2 * Real.sin x + Real.cos x = 2 * Real.sin (x - φ) - Real.cos (x - φ)) →
  Real.sin φ = 4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_phi_value_l1051_105144


namespace NUMINAMATH_CALUDE_max_y_intercept_of_even_function_l1051_105187

def f (x a b : ℝ) : ℝ := x^2 + (a^2 + b^2 - 1)*x + a^2 + 2*a*b - b^2

theorem max_y_intercept_of_even_function
  (h : ∀ x, f x a b = f (-x) a b) :
  ∃ C, (∀ a b, f 0 a b ≤ C) ∧ (∃ a b, f 0 a b = C) ∧ C = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_y_intercept_of_even_function_l1051_105187


namespace NUMINAMATH_CALUDE_driver_hourly_wage_l1051_105122

/-- Calculates the hourly wage of a driver after fuel costs --/
theorem driver_hourly_wage
  (speed : ℝ)
  (time : ℝ)
  (fuel_efficiency : ℝ)
  (income_per_mile : ℝ)
  (fuel_cost_per_gallon : ℝ)
  (h1 : speed = 60)
  (h2 : time = 2)
  (h3 : fuel_efficiency = 30)
  (h4 : income_per_mile = 0.5)
  (h5 : fuel_cost_per_gallon = 2)
  : (income_per_mile * speed * time - (speed * time / fuel_efficiency) * fuel_cost_per_gallon) / time = 26 := by
  sorry

end NUMINAMATH_CALUDE_driver_hourly_wage_l1051_105122


namespace NUMINAMATH_CALUDE_monotone_xfx_l1051_105100

open Real

theorem monotone_xfx (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, HasDerivAt f (f' x) x) 
  (h_ineq : ∀ x, x * f' x > -f x) (x₁ x₂ : ℝ) (h_lt : x₁ < x₂) : 
  x₁ * f x₁ < x₂ * f x₂ := by
  sorry

end NUMINAMATH_CALUDE_monotone_xfx_l1051_105100


namespace NUMINAMATH_CALUDE_age_difference_l1051_105152

/-- Given the ages of three people A, B, and C, prove that A is 2 years older than B. -/
theorem age_difference (A B C : ℕ) : 
  B = 18 →
  B = 2 * C →
  A + B + C = 47 →
  A = B + 2 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1051_105152


namespace NUMINAMATH_CALUDE_bookshop_inventory_problem_l1051_105107

/-- Represents the bookshop inventory problem -/
theorem bookshop_inventory_problem (S : ℕ) : 
  743 - (S + 128 + 2*S + (128 + 34)) + 160 = 502 → S = 37 := by
  sorry

end NUMINAMATH_CALUDE_bookshop_inventory_problem_l1051_105107


namespace NUMINAMATH_CALUDE_range_of_a_l1051_105135

def p (a : ℝ) : Prop := ∀ m ∈ Set.Icc (-1 : ℝ) 1, a^2 - 5*a - 3 ≥ Real.sqrt (m^2 + 8)

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + a*x + 2 < 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) →
  a ∈ Set.Icc (-Real.sqrt 8) (-1) ∪ Set.Ioo (Real.sqrt 8) 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1051_105135


namespace NUMINAMATH_CALUDE_subtract_like_terms_l1051_105115

theorem subtract_like_terms (a : ℝ) : 7 * a^2 - 4 * a^2 = 3 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_subtract_like_terms_l1051_105115


namespace NUMINAMATH_CALUDE_difference_C_D_l1051_105108

def C : ℤ := (Finset.range 20).sum (fun i => (2*i + 2) * (2*i + 3)) + 42

def D : ℤ := 2 + (Finset.range 20).sum (fun i => (2*i + 3) * (2*i + 4))

theorem difference_C_D : |C - D| = 400 := by sorry

end NUMINAMATH_CALUDE_difference_C_D_l1051_105108


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1051_105129

theorem min_value_of_expression (b : ℝ) (h : 8 * b^2 + 7 * b + 6 = 5) :
  ∃ (m : ℝ), (∀ b', 8 * b'^2 + 7 * b' + 6 = 5 → 3 * b' + 2 ≥ m) ∧ (3 * b + 2 = m) ∧ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1051_105129


namespace NUMINAMATH_CALUDE_min_distance_point_to_circle_through_reflection_l1051_105177

/-- The minimum distance from a point to a circle through a reflection point on the x-axis -/
theorem min_distance_point_to_circle_through_reflection (A B P : ℝ × ℝ) : 
  A = (-3, 3) →
  P.2 = 0 →
  (B.1 - 1)^2 + (B.2 - 1)^2 = 2 →
  ∃ (min_dist : ℝ), min_dist = 3 * Real.sqrt 2 ∧ 
    ∀ (P' : ℝ × ℝ), P'.2 = 0 → 
      Real.sqrt ((P'.1 - A.1)^2 + (P'.2 - A.2)^2) + 
      Real.sqrt ((B.1 - P'.1)^2 + (B.2 - P'.2)^2) ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_point_to_circle_through_reflection_l1051_105177


namespace NUMINAMATH_CALUDE_eva_patch_area_l1051_105111

/-- Represents a rectangular vegetable patch -/
structure VegetablePatch where
  short_side : ℕ  -- Number of posts on the shorter side
  long_side : ℕ   -- Number of posts on the longer side
  post_spacing : ℕ -- Distance between posts in yards

/-- Properties of Eva's vegetable patch -/
def eva_patch : VegetablePatch where
  short_side := 3
  long_side := 9
  post_spacing := 6

/-- Total number of posts -/
def total_posts (p : VegetablePatch) : ℕ :=
  2 * (p.short_side + p.long_side) - 4

/-- Relationship between short and long sides -/
def side_relationship (p : VegetablePatch) : Prop :=
  p.long_side = 3 * p.short_side

/-- Calculate the area of the vegetable patch -/
def patch_area (p : VegetablePatch) : ℕ :=
  (p.short_side - 1) * (p.long_side - 1) * p.post_spacing * p.post_spacing

/-- Theorem stating the area of Eva's vegetable patch -/
theorem eva_patch_area :
  total_posts eva_patch = 24 ∧
  side_relationship eva_patch ∧
  patch_area eva_patch = 576 := by
  sorry

#eval patch_area eva_patch

end NUMINAMATH_CALUDE_eva_patch_area_l1051_105111


namespace NUMINAMATH_CALUDE_sum_f_positive_l1051_105138

def f (x : ℝ) : ℝ := x + x^3

theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l1051_105138


namespace NUMINAMATH_CALUDE_perfect_square_m_l1051_105141

theorem perfect_square_m (n : ℕ) (m : ℤ) 
  (h1 : m = 2 + 2 * Int.sqrt (44 * n^2 + 1)) : 
  ∃ k : ℤ, m = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_m_l1051_105141


namespace NUMINAMATH_CALUDE_triangle_angle_sine_identity_l1051_105165

theorem triangle_angle_sine_identity 
  (A B C : ℝ) (n : ℤ) 
  (h_triangle : A + B + C = Real.pi) 
  (h_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  Real.sin (2 * n * A) + Real.sin (2 * n * B) + Real.sin (2 * n * C) = 
  (-1)^(n+1) * 4 * Real.sin (n * A) * Real.sin (n * B) * Real.sin (n * C) := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sine_identity_l1051_105165


namespace NUMINAMATH_CALUDE_triangle_problem_l1051_105128

/-- Given a triangle ABC with tanA = 1/4, tanB = 3/5, and AB = √17,
    prove that the measure of angle C is 3π/4 and the smallest side length is √2 -/
theorem triangle_problem (A B C : Real) (AB : Real) :
  0 < A ∧ A < π/2 →
  0 < B ∧ B < π/2 →
  0 < C ∧ C < π →
  Real.tan A = 1/4 →
  Real.tan B = 3/5 →
  AB = Real.sqrt 17 →
  C = 3*π/4 ∧ 
  (min AB (min (AB * Real.sin A / Real.sin C) (AB * Real.sin B / Real.sin C)) = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1051_105128


namespace NUMINAMATH_CALUDE_solution_is_two_lines_l1051_105130

-- Define the equation
def equation (x y : ℝ) : Prop := (x - 2*y)^2 = x^2 - 4*y^2

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | equation p.1 p.2}

-- Define the two lines
def x_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}
def diagonal_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 * p.2}

-- Theorem statement
theorem solution_is_two_lines :
  solution_set = x_axis ∪ diagonal_line :=
sorry

end NUMINAMATH_CALUDE_solution_is_two_lines_l1051_105130


namespace NUMINAMATH_CALUDE_derivative_sum_positive_l1051_105113

open Real

noncomputable def f (a b x : ℝ) : ℝ := a * log x - b * x - 1 / x

theorem derivative_sum_positive (a : ℝ) (h_a : a > 0) (x₁ x₂ : ℝ) 
  (h_x₁ : x₁ > 0) (h_x₂ : x₂ > 0) (h_neq : x₁ ≠ x₂) :
  ∃ b : ℝ, f a b x₁ = f a b x₂ → 
    (deriv (f a b) x₁ + deriv (f a b) x₂ > 0) := by
  sorry

end NUMINAMATH_CALUDE_derivative_sum_positive_l1051_105113


namespace NUMINAMATH_CALUDE_daniella_savings_l1051_105167

/-- Daniella's savings amount -/
def D : ℝ := 400

/-- Ariella's initial savings amount -/
def A : ℝ := D + 200

/-- Interest rate per annum (as a decimal) -/
def r : ℝ := 0.1

/-- Time period in years -/
def t : ℝ := 2

/-- Ariella's final amount after interest -/
def F : ℝ := 720

theorem daniella_savings : 
  (A + A * r * t = F) → D = 400 := by
  sorry

end NUMINAMATH_CALUDE_daniella_savings_l1051_105167


namespace NUMINAMATH_CALUDE_income_comparison_l1051_105157

theorem income_comparison (Tim Mary Juan : ℝ) 
  (h1 : Mary = 1.60 * Tim) 
  (h2 : Mary = 1.28 * Juan) : 
  Tim = 0.80 * Juan := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l1051_105157


namespace NUMINAMATH_CALUDE_f_2_3_4_equals_59_l1051_105103

def f (x y z : ℝ) : ℝ := 2 * x^3 + 3 * y^2 + z^2

theorem f_2_3_4_equals_59 : f 2 3 4 = 59 := by
  sorry

end NUMINAMATH_CALUDE_f_2_3_4_equals_59_l1051_105103


namespace NUMINAMATH_CALUDE_expense_difference_l1051_105169

theorem expense_difference (alice_paid bob_paid carol_paid : ℕ) 
  (h_alice : alice_paid = 120)
  (h_bob : bob_paid = 150)
  (h_carol : carol_paid = 210) : 
  let total := alice_paid + bob_paid + carol_paid
  let each_share := total / 3
  let alice_owes := each_share - alice_paid
  let bob_owes := each_share - bob_paid
  alice_owes - bob_owes = 30 := by
  sorry

end NUMINAMATH_CALUDE_expense_difference_l1051_105169


namespace NUMINAMATH_CALUDE_arithmetic_sum_odd_numbers_l1051_105131

theorem arithmetic_sum_odd_numbers : ∀ (a₁ aₙ d n : ℕ),
  a₁ = 1 →
  aₙ = 99 →
  d = 2 →
  aₙ = a₁ + (n - 1) * d →
  n * (a₁ + aₙ) / 2 = 2500 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_odd_numbers_l1051_105131


namespace NUMINAMATH_CALUDE_product_expansion_sum_l1051_105114

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  9 * a + 3 * b + c + d = 19 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_sum_l1051_105114


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l1051_105179

theorem polynomial_multiplication (a b : ℝ) :
  (3 * a^4 - 7 * b^3) * (9 * a^8 + 21 * a^4 * b^3 + 49 * b^6 + 6 * a^2 * b^2) =
  27 * a^12 + 18 * a^6 * b^2 - 42 * a^2 * b^5 - 343 * b^9 := by
sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l1051_105179


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_less_than_zero_negation_of_cubic_inequality_l1051_105158

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬(p x)) :=
by sorry

theorem negation_of_less_than_zero (x : ℝ) :
  ¬(x < 0) ↔ (x ≥ 0) :=
by sorry

theorem negation_of_cubic_inequality :
  (¬∀ x : ℝ, x^3 + 2 < 0) ↔ (∃ x : ℝ, x^3 + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_less_than_zero_negation_of_cubic_inequality_l1051_105158
