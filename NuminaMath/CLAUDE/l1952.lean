import Mathlib

namespace NUMINAMATH_CALUDE_cube_power_inequality_l1952_195241

theorem cube_power_inequality (a b c : ℕ+) :
  (a^(a:ℕ) * b^(b:ℕ) * c^(c:ℕ))^3 ≥ (a*b*c)^((a:ℕ)+(b:ℕ)+(c:ℕ)) := by
  sorry

end NUMINAMATH_CALUDE_cube_power_inequality_l1952_195241


namespace NUMINAMATH_CALUDE_inequality_proof_l1952_195254

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a^(1/3) * b^(1/3) + c^(1/3) * d^(1/3) ≤ (a+b+c)^(1/3) * (a+c+d)^(1/3) ∧
  (a^(1/3) * b^(1/3) + c^(1/3) * d^(1/3) = (a+b+c)^(1/3) * (a+c+d)^(1/3) ↔
   b = (a/c)*(a+c) ∧ d = (c/a)*(a+c)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1952_195254


namespace NUMINAMATH_CALUDE_weight_of_fresh_grapes_l1952_195260

/-- Given that fresh grapes contain 90% water by weight, dried grapes contain 20% water by weight,
    and the weight of dry grapes available is 3.125 kg, prove that the weight of fresh grapes is 78.125 kg. -/
theorem weight_of_fresh_grapes :
  let fresh_water_ratio : ℝ := 0.9
  let dried_water_ratio : ℝ := 0.2
  let dried_grapes_weight : ℝ := 3.125
  fresh_water_ratio * fresh_grapes_weight = dried_water_ratio * dried_grapes_weight + 
    (1 - dried_water_ratio) * dried_grapes_weight →
  fresh_grapes_weight = 78.125
  := by sorry

#check weight_of_fresh_grapes

end NUMINAMATH_CALUDE_weight_of_fresh_grapes_l1952_195260


namespace NUMINAMATH_CALUDE_function_point_coefficient_l1952_195232

/-- Given a function f(x) = ax³ - 2x that passes through the point (-1, 4), prove that a = -2 -/
theorem function_point_coefficient (a : ℝ) : 
  (fun x : ℝ => a * x^3 - 2*x) (-1) = 4 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_point_coefficient_l1952_195232


namespace NUMINAMATH_CALUDE_max_value_theorem_l1952_195243

theorem max_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (heq : a * (a + b + c) = b * c) : 
  a / (b + c) ≤ (Real.sqrt 2 - 1) / 2 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ 
    a₀ * (a₀ + b₀ + c₀) = b₀ * c₀ ∧ 
    a₀ / (b₀ + c₀) = (Real.sqrt 2 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1952_195243


namespace NUMINAMATH_CALUDE_class_size_l1952_195215

theorem class_size (total : ℚ) 
  (h1 : (2 : ℚ) / 3 * total = total * (2 : ℚ) / 3)  -- Two-thirds of the class have brown eyes
  (h2 : (1 : ℚ) / 2 * (total * (2 : ℚ) / 3) = total / 3)  -- Half of the students with brown eyes have black hair
  (h3 : (total / 3 : ℚ) = 6)  -- There are 6 students with brown eyes and black hair
  : total = 18 := by
sorry

end NUMINAMATH_CALUDE_class_size_l1952_195215


namespace NUMINAMATH_CALUDE_inequality_proof_l1952_195266

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c) / 3 ≥ ((a + b) * (b + c) * (c + a) / 8) ^ (1/3) ∧
  ((a + b) * (b + c) * (c + a) / 8) ^ (1/3) ≥ (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a)) / 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1952_195266


namespace NUMINAMATH_CALUDE_marble_picking_ways_l1952_195298

/-- The number of ways to pick up at least one marble from a set of marbles -/
def pick_marbles_ways (red green yellow black pink : ℕ) : ℕ :=
  ((red + 1) * (green + 1) * (yellow + 1) * (black + 1) * (pink + 1)) - 1

/-- Theorem: There are 95 ways to pick up at least one marble from a set
    containing 3 red marbles, 2 green marbles, and one each of yellow, black, and pink marbles -/
theorem marble_picking_ways :
  pick_marbles_ways 3 2 1 1 1 = 95 := by
  sorry

end NUMINAMATH_CALUDE_marble_picking_ways_l1952_195298


namespace NUMINAMATH_CALUDE_enter_exit_ways_count_l1952_195281

/-- The number of doors in the room -/
def num_doors : ℕ := 4

/-- The number of ways to enter and exit the room -/
def ways_to_enter_and_exit : ℕ := num_doors * num_doors

/-- Theorem: The number of different ways to enter and exit a room with four doors is 64 -/
theorem enter_exit_ways_count : ways_to_enter_and_exit = 64 := by
  sorry

end NUMINAMATH_CALUDE_enter_exit_ways_count_l1952_195281


namespace NUMINAMATH_CALUDE_triangle_incircle_path_length_l1952_195299

theorem triangle_incircle_path_length (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_sides : a = 6 ∧ b = 8 ∧ c = 10) : 
  let s := (a + b + c) / 2
  let r := (s * (s - a) * (s - b) * (s - c)).sqrt / s
  (a + b + c) - 2 * r = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_incircle_path_length_l1952_195299


namespace NUMINAMATH_CALUDE_complex_number_values_l1952_195251

theorem complex_number_values (z : ℂ) (a : ℝ) :
  z = (4 + 2*I) / (a + I) → Complex.abs z = Real.sqrt 10 →
  z = 3 - I ∨ z = -1 - 3*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_values_l1952_195251


namespace NUMINAMATH_CALUDE_tom_steps_when_matt_reaches_220_l1952_195212

/-- Proves that Tom reaches 275 steps when Matt reaches 220 steps, given their respective speeds -/
theorem tom_steps_when_matt_reaches_220 
  (matt_speed : ℕ) 
  (tom_speed_diff : ℕ) 
  (matt_steps : ℕ) 
  (h1 : matt_speed = 20)
  (h2 : tom_speed_diff = 5)
  (h3 : matt_steps = 220) :
  matt_steps + (matt_steps / matt_speed) * tom_speed_diff = 275 := by
  sorry

#check tom_steps_when_matt_reaches_220

end NUMINAMATH_CALUDE_tom_steps_when_matt_reaches_220_l1952_195212


namespace NUMINAMATH_CALUDE_subset_relationship_l1952_195238

def M : Set (ℝ × ℝ) := {p | |p.1| + |p.2| < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | |p.1 + p.2| < 1 ∧ |p.1| < 1 ∧ |p.2| < 1}

theorem subset_relationship : M ⊆ N ∧ N ⊆ P := by sorry

end NUMINAMATH_CALUDE_subset_relationship_l1952_195238


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l1952_195210

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_ratio_theorem (t : Triangle) 
  (h1 : t.A = 30 * π / 180)  -- A = 30°
  (h2 : t.a = Real.sqrt 3)   -- a = √3
  (h3 : t.A + t.B + t.C = π) -- Sum of angles in a triangle is π
  (h4 : t.a / Real.sin t.A = t.b / Real.sin t.B) -- Law of Sines
  (h5 : t.b / Real.sin t.B = t.c / Real.sin t.C) -- Law of Sines
  : (t.a + t.b + t.c) / (Real.sin t.A + Real.sin t.B + Real.sin t.C) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_ratio_theorem_l1952_195210


namespace NUMINAMATH_CALUDE_solve_system_l1952_195287

theorem solve_system (a b : ℝ) (h1 : 3 * a + 4 * b = 2) (h2 : a = 2 * b - 3) : 5 * b = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1952_195287


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l1952_195206

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 7/11 ∧ 
  ∀ (y : ℤ), ((1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 7/11) → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l1952_195206


namespace NUMINAMATH_CALUDE_triangle_angle_value_l1952_195214

theorem triangle_angle_value (A B C : ℝ) (a b c : ℝ) :
  0 < B → B < π →
  0 < C → C < π →
  b * Real.cos C + c * Real.sin B = 0 →
  C = 3 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_value_l1952_195214


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l1952_195291

-- Define the polynomials f, g, and h
def f (x : ℝ) : ℝ := -2 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -3 * x^2 + 4 * x - 7
def h (x : ℝ) : ℝ := 6 * x^2 + 3 * x + 2

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -2 * x^3 - x^2 + 9 * x - 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l1952_195291


namespace NUMINAMATH_CALUDE_set_operation_result_l1952_195269

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {1, 3, 5}
def C : Set ℤ := {0, 2, 4}

theorem set_operation_result : (A ∩ B) ∪ C = {0, 1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_operation_result_l1952_195269


namespace NUMINAMATH_CALUDE_equal_interest_rates_l1952_195264

/-- Proves that given two accounts with equal investments, if one account has an interest rate of 10%
    and both accounts earn the same interest at the end of the year, then the interest rate of the
    other account is also 10%. -/
theorem equal_interest_rates
  (investment : ℝ)
  (rate1 rate2 : ℝ)
  (h1 : investment > 0)
  (h2 : rate2 = 0.1)
  (h3 : investment * rate1 = investment * rate2) :
  rate1 = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_equal_interest_rates_l1952_195264


namespace NUMINAMATH_CALUDE_fraction_inequality_implies_inequality_l1952_195208

theorem fraction_inequality_implies_inequality (a b c : ℝ) (hc : c ≠ 0) :
  a / c^2 < b / c^2 → a < b :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_implies_inequality_l1952_195208


namespace NUMINAMATH_CALUDE_sum_of_multiples_l1952_195217

def smallest_two_digit_multiple_of_5 : ℕ := 10

def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_multiples : 
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_multiples_l1952_195217


namespace NUMINAMATH_CALUDE_x_minus_y_value_l1952_195242

theorem x_minus_y_value (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : x - y = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_value_l1952_195242


namespace NUMINAMATH_CALUDE_smallest_k_for_integer_product_l1952_195205

def a : ℕ → ℝ
  | 0 => 1
  | 1 => 3 ^ (1 / 17)
  | (n + 2) => a (n + 1) * (a n) ^ 2

def product_up_to (k : ℕ) : ℝ :=
  (List.range k).foldl (λ acc i => acc * a (i + 1)) 1

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem smallest_k_for_integer_product :
  (∀ k < 11, ¬ is_integer (product_up_to k)) ∧
  is_integer (product_up_to 11) := by sorry

end NUMINAMATH_CALUDE_smallest_k_for_integer_product_l1952_195205


namespace NUMINAMATH_CALUDE_friday_work_proof_l1952_195245

/-- The time Mr. Willson worked on Friday in minutes -/
def friday_work_minutes : ℚ := 75

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

theorem friday_work_proof (monday : ℚ) (tuesday : ℚ) (wednesday : ℚ) (thursday : ℚ) 
  (h_monday : monday = 3/4)
  (h_tuesday : tuesday = 1/2)
  (h_wednesday : wednesday = 2/3)
  (h_thursday : thursday = 5/6)
  (h_total : monday + tuesday + wednesday + thursday + friday_work_minutes / 60 = 4) :
  friday_work_minutes = 75 := by
  sorry


end NUMINAMATH_CALUDE_friday_work_proof_l1952_195245


namespace NUMINAMATH_CALUDE_perimeter_formula_and_maximum_l1952_195204

noncomputable section

open Real

/-- Triangle ABC with given properties -/
structure Triangle where
  A : ℝ
  BC : ℝ
  x : ℝ  -- Angle B
  y : ℝ  -- Perimeter
  h_A : A = π / 3
  h_BC : BC = 2 * sqrt 3
  h_x_pos : x > 0
  h_x_upper : x < 2 * π / 3

/-- Perimeter function -/
def perimeter (t : Triangle) : ℝ := 6 * sin t.x + 2 * sqrt 3 * cos t.x + 2 * sqrt 3

theorem perimeter_formula_and_maximum (t : Triangle) :
  t.y = perimeter t ∧ t.y ≤ 6 * sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_perimeter_formula_and_maximum_l1952_195204


namespace NUMINAMATH_CALUDE_trigonometric_product_l1952_195246

theorem trigonometric_product (cos60 sin60 cos30 sin30 : ℝ) : 
  cos60 = 1/2 →
  sin60 = Real.sqrt 3 / 2 →
  cos30 = Real.sqrt 3 / 2 →
  sin30 = 1/2 →
  (1 - 1/cos30) * (1 + 1/sin60) * (1 - 1/sin30) * (1 + 1/cos60) = -1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_l1952_195246


namespace NUMINAMATH_CALUDE_assembly_line_theorem_l1952_195257

/-- Represents the number of tasks in the assembly line -/
def num_tasks : ℕ := 6

/-- Represents the number of tasks that can be freely arranged -/
def free_tasks : ℕ := num_tasks - 1

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The number of ways to arrange the assembly line -/
def assembly_line_arrangements : ℕ := factorial free_tasks

theorem assembly_line_theorem : assembly_line_arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_assembly_line_theorem_l1952_195257


namespace NUMINAMATH_CALUDE_sum_of_roots_l1952_195259

theorem sum_of_roots (x : ℝ) : 
  (x^2 + 2023*x = 2024) → 
  (∃ y : ℝ, y^2 + 2023*y = 2024 ∧ x + y = -2023) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1952_195259


namespace NUMINAMATH_CALUDE_x_minus_y_equals_18_l1952_195263

theorem x_minus_y_equals_18 (x y : ℤ) (h1 : x + y = 10) (h2 : x = 14) : x - y = 18 := by
  sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_18_l1952_195263


namespace NUMINAMATH_CALUDE_union_eq_right_iff_complement_subset_l1952_195244

variable {U : Type*} -- Universal set
variable (A B : Set U) -- Sets A and B

theorem union_eq_right_iff_complement_subset :
  A ∪ B = B ↔ (Bᶜ : Set U) ⊆ (Aᶜ : Set U) := by sorry

end NUMINAMATH_CALUDE_union_eq_right_iff_complement_subset_l1952_195244


namespace NUMINAMATH_CALUDE_special_arrangement_count_l1952_195262

/-- The number of permutations of n distinct objects taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ := sorry

/-- The number of ways to arrange 6 people in a row with specific conditions -/
def special_arrangement : ℕ :=
  permutations 2 2 * permutations 4 4

theorem special_arrangement_count :
  special_arrangement = 48 :=
sorry

end NUMINAMATH_CALUDE_special_arrangement_count_l1952_195262


namespace NUMINAMATH_CALUDE_nikola_ant_farm_problem_l1952_195239

/-- Nikola's ant farm problem -/
theorem nikola_ant_farm_problem 
  (num_ants : ℕ) 
  (food_per_ant : ℕ) 
  (food_cost_per_oz : ℚ) 
  (leaf_cost : ℚ) 
  (num_leaves : ℕ) 
  (num_jobs : ℕ) : 
  num_ants = 400 →
  food_per_ant = 2 →
  food_cost_per_oz = 1/10 →
  leaf_cost = 1/100 →
  num_leaves = 6000 →
  num_jobs = 4 →
  (num_ants * food_per_ant * food_cost_per_oz - num_leaves * leaf_cost) / num_jobs = 5 :=
by sorry

end NUMINAMATH_CALUDE_nikola_ant_farm_problem_l1952_195239


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1952_195213

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∃ (n : ℕ), is_three_digit n ∧ n % 9 = 0 ∧ digit_sum n = 27 ∧
  ∀ (m : ℕ), is_three_digit m ∧ m % 9 = 0 ∧ digit_sum m = 27 → m ≤ n :=
by
  use 999
  sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_9_with_digit_sum_27_l1952_195213


namespace NUMINAMATH_CALUDE_max_value_cos_theta_l1952_195279

noncomputable def f (x : ℝ) : ℝ := Real.sin x - 2 * Real.cos x

theorem max_value_cos_theta :
  ∀ θ : ℝ, (∀ x : ℝ, f x ≤ f θ) → Real.cos θ = -2 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_theta_l1952_195279


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l1952_195295

theorem triangle_max_perimeter :
  ∀ x : ℕ,
    x > 0 →
    x < 18 →
    x + 2*x > 18 →
    x + 2*x + 18 ≤ 69 :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l1952_195295


namespace NUMINAMATH_CALUDE_teacher_health_survey_l1952_195207

theorem teacher_health_survey (total : ℕ) (high_bp : ℕ) (heart_trouble : ℕ) (both : ℕ)
  (h_total : total = 120)
  (h_high_bp : high_bp = 70)
  (h_heart_trouble : heart_trouble = 40)
  (h_both : both = 20) :
  (total - (high_bp + heart_trouble - both)) / total * 100 = 25 :=
by sorry

end NUMINAMATH_CALUDE_teacher_health_survey_l1952_195207


namespace NUMINAMATH_CALUDE_largest_divisor_of_product_l1952_195216

def product (n : ℕ) : ℕ := (n+2)*(n+4)*(n+6)*(n+8)*(n+10)

theorem largest_divisor_of_product (n : ℕ) (h : Odd n) :
  (∃ (k : ℕ), product n = 15 * k) ∧
  (∀ (m : ℕ), m > 15 → ¬(∀ (n : ℕ), Odd n → ∃ (k : ℕ), product n = m * k)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_product_l1952_195216


namespace NUMINAMATH_CALUDE_jogger_train_distance_l1952_195282

/-- Proves that a jogger is 240 meters ahead of a train given specific conditions -/
theorem jogger_train_distance (jogger_speed : ℝ) (train_speed : ℝ) (train_length : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 120 →
  passing_time = 36 →
  (train_speed - jogger_speed) * passing_time - train_length = 240 :=
by
  sorry

#check jogger_train_distance

end NUMINAMATH_CALUDE_jogger_train_distance_l1952_195282


namespace NUMINAMATH_CALUDE_power_of_two_contains_k_zeros_l1952_195237

theorem power_of_two_contains_k_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ+, ∃ a b : ℕ, a ≠ 0 ∧ b ≠ 0 ∧
  (∃ m : ℕ, (2 : ℕ) ^ (n : ℕ) = m * 10^(k+1) + a * 10^k + b) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_contains_k_zeros_l1952_195237


namespace NUMINAMATH_CALUDE_sqrt_sum_squares_plus_seven_l1952_195225

theorem sqrt_sum_squares_plus_seven (a b : ℝ) : 
  a = Real.sqrt 5 + 2 → b = Real.sqrt 5 - 2 → Real.sqrt (a^2 + b^2 + 7) = 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_squares_plus_seven_l1952_195225


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_l1952_195235

theorem lawrence_county_kids_count :
  let group_a_1week : ℕ := 175000
  let group_a_2week : ℕ := 107000
  let group_a_3week : ℕ := 35000
  let group_b_1week : ℕ := 100000
  let group_b_2week : ℕ := 70350
  let group_b_3week : ℕ := 19500
  let group_c_1week : ℕ := 45000
  let group_c_2week : ℕ := 87419
  let group_c_3week : ℕ := 14425
  let kids_staying_home : ℕ := 590796
  let kids_outside_county : ℕ := 22
  
  let total_group_a : ℕ := group_a_1week + group_a_2week + group_a_3week
  let total_group_b : ℕ := group_b_1week + group_b_2week + group_b_3week
  let total_group_c : ℕ := group_c_1week + group_c_2week + group_c_3week
  
  let total_kids_in_camp : ℕ := total_group_a + total_group_b + total_group_c
  
  total_kids_in_camp + kids_staying_home + kids_outside_county = 1244512 :=
by
  sorry

#check lawrence_county_kids_count

end NUMINAMATH_CALUDE_lawrence_county_kids_count_l1952_195235


namespace NUMINAMATH_CALUDE_french_english_speakers_l1952_195250

theorem french_english_speakers (total_students : ℕ) 
  (non_french_percentage : ℚ) (french_non_english : ℕ) : 
  total_students = 200 →
  non_french_percentage = 3/4 →
  french_non_english = 40 →
  (total_students : ℚ) * (1 - non_french_percentage) - french_non_english = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_french_english_speakers_l1952_195250


namespace NUMINAMATH_CALUDE_fourth_term_value_l1952_195284

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * a 2
  first_term : a 1 = 9
  fifth_term : a 5 = a 3 * (a 4)^2

/-- The fourth term of the geometric sequence is ± 1/3 -/
theorem fourth_term_value (seq : GeometricSequence) : 
  seq.a 4 = 1/3 ∨ seq.a 4 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_value_l1952_195284


namespace NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l1952_195219

/-- Represents the number of flies eaten daily in a swamp ecosystem -/
def flies_eaten_daily (num_gharials : ℕ) (fish_per_gharial : ℕ) (frogs_per_fish : ℕ) (flies_per_frog : ℕ) : ℕ :=
  num_gharials * fish_per_gharial * frogs_per_fish * flies_per_frog

/-- Theorem stating the number of flies eaten daily in the given swamp ecosystem -/
theorem swamp_ecosystem_flies_eaten :
  flies_eaten_daily 9 15 8 30 = 32400 := by
  sorry

#eval flies_eaten_daily 9 15 8 30

end NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l1952_195219


namespace NUMINAMATH_CALUDE_triangle_altitude_l1952_195226

theorem triangle_altitude (base : ℝ) (square_side : ℝ) (altitude : ℝ) : 
  base = 6 →
  square_side = 6 →
  (1/2) * base * altitude = square_side * square_side →
  altitude = 12 := by
sorry

end NUMINAMATH_CALUDE_triangle_altitude_l1952_195226


namespace NUMINAMATH_CALUDE_vectors_form_basis_l1952_195261

def e₁ : ℝ × ℝ := (-1, 2)
def e₂ : ℝ × ℝ := (5, 7)

theorem vectors_form_basis : LinearIndependent ℝ ![e₁, e₂] ∧ Submodule.span ℝ {e₁, e₂} = ⊤ := by
  sorry

end NUMINAMATH_CALUDE_vectors_form_basis_l1952_195261


namespace NUMINAMATH_CALUDE_proposition_2_proposition_4_l1952_195228

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the lines and planes
variable (m n : Line) (α β : Plane)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (subset : Line → Plane → Prop)

-- Axiom: m and n are different lines
axiom different_lines : m ≠ n

-- Axiom: α and β are different planes
axiom different_planes : α ≠ β

-- Proposition 2
theorem proposition_2 : 
  (perpendicular_line_line m n ∧ perpendicular_line_plane n α ∧ perpendicular_line_plane m β) → 
  perpendicular_plane_plane α β :=
sorry

-- Proposition 4
theorem proposition_4 : 
  (perpendicular_line_plane n β ∧ perpendicular_plane_plane α β) → 
  (parallel n α ∨ subset n α) :=
sorry

end NUMINAMATH_CALUDE_proposition_2_proposition_4_l1952_195228


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1952_195272

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  arith : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1 : ℕ) * seq.d) / 2

theorem arithmetic_sequence_first_term 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 3 + seq.a 6 = 40) 
  (h2 : S seq 2 = 10) : 
  seq.a 1 = 5/2 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1952_195272


namespace NUMINAMATH_CALUDE_circumference_difference_of_concentric_circles_l1952_195289

/-- Given two concentric circles with the specified properties, 
    prove the difference in their circumferences --/
theorem circumference_difference_of_concentric_circles 
  (r_inner : ℝ) (r_outer : ℝ) (h1 : r_outer = r_inner + 15) 
  (h2 : 2 * r_inner = 50) : 
  2 * π * r_outer - 2 * π * r_inner = 30 * π := by
  sorry

end NUMINAMATH_CALUDE_circumference_difference_of_concentric_circles_l1952_195289


namespace NUMINAMATH_CALUDE_factors_of_210_l1952_195265

theorem factors_of_210 : Finset.card (Nat.divisors 210) = 16 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_210_l1952_195265


namespace NUMINAMATH_CALUDE_simplify_expression_l1952_195275

theorem simplify_expression : 
  Real.sqrt 8 - 2 * Real.sqrt (1/2) + (2 - Real.sqrt 3) * (2 + Real.sqrt 3) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1952_195275


namespace NUMINAMATH_CALUDE_stewart_farm_horse_food_l1952_195293

/-- Calculates the total amount of horse food needed per day on a farm -/
def total_horse_food (num_sheep : ℕ) (sheep_ratio horse_ratio : ℕ) (food_per_horse : ℕ) : ℕ :=
  let num_horses := (num_sheep * horse_ratio) / sheep_ratio
  num_horses * food_per_horse

/-- Theorem: The Stewart farm needs 12,880 ounces of horse food per day -/
theorem stewart_farm_horse_food :
  total_horse_food 40 5 7 230 = 12880 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_horse_food_l1952_195293


namespace NUMINAMATH_CALUDE_cubic_polynomial_c_value_l1952_195292

theorem cubic_polynomial_c_value 
  (g : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, g x = x^3 + a*x^2 + b*x + c) 
  (h2 : ∃ r₁ r₂ r₃ : ℕ, (∀ i, Odd (r₁ + 2*i) ∧ g (r₁ + 2*i) = 0) ∧ 
                        (r₁ < r₂) ∧ (r₂ < r₃) ∧ 
                        g r₁ = 0 ∧ g r₂ = 0 ∧ g r₃ = 0)
  (h3 : a + b + c = -11) :
  c = -15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_c_value_l1952_195292


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l1952_195236

theorem unique_two_digit_number : ∃! n : ℕ,
  10 ≤ n ∧ n < 100 ∧  -- two-digit number
  n % 5 = 0 ∧         -- divisible by 5
  n % 3 ≠ 0 ∧         -- not divisible by 3
  n % 4 ≠ 0 ∧         -- not divisible by 4
  (97 * n) % 2 = 0 ∧  -- 97 times is even
  n / 10 ≥ 6 ∧        -- tens digit not less than 6
  n = 70              -- the number is 70
  := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l1952_195236


namespace NUMINAMATH_CALUDE_exists_periodic_nonconstant_sequence_l1952_195240

/-- A sequence is periodic if there exists a positive integer p such that
    x_{n+p} = x_n for all integers n -/
def IsPeriodic (x : ℤ → ℝ) : Prop :=
  ∃ p : ℕ+, ∀ n : ℤ, x (n + p) = x n

/-- A sequence is constant if all its terms are equal -/
def IsConstant (x : ℤ → ℝ) : Prop :=
  ∀ m n : ℤ, x m = x n

/-- The recurrence relation for the sequence -/
def SatisfiesRecurrence (x : ℤ → ℝ) : Prop :=
  ∀ n : ℤ, x (n + 1) = 3 * x n + 4 * x (n - 1)

theorem exists_periodic_nonconstant_sequence :
  ∃ x : ℤ → ℝ, SatisfiesRecurrence x ∧ IsPeriodic x ∧ ¬IsConstant x := by
  sorry

end NUMINAMATH_CALUDE_exists_periodic_nonconstant_sequence_l1952_195240


namespace NUMINAMATH_CALUDE_coefficient_a9_l1952_195230

theorem coefficient_a9 (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (fun x : ℝ => x^2 + x^10) = 
  (fun x : ℝ => a + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + 
                a₉*(x+1)^9 + a₁₀*(x+1)^10) →
  a₉ = -10 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a9_l1952_195230


namespace NUMINAMATH_CALUDE_range_of_a_l1952_195202

-- Define the propositions P and Q
def P (a : ℝ) : Prop := ∀ x, x^2 + 2*a*x + 4 > 0

def Q (a : ℝ) : Prop := ∀ x y, x < y → (5-2*a)^x < (5-2*a)^y

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (P a ∨ Q a) ∧ ¬(P a ∧ Q a) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1952_195202


namespace NUMINAMATH_CALUDE_jims_cousin_money_l1952_195227

-- Define the costs of items
def cheeseburger_cost : ℚ := 3
def milkshake_cost : ℚ := 5
def cheese_fries_cost : ℚ := 8

-- Define the number of items ordered
def num_cheeseburgers : ℕ := 2
def num_milkshakes : ℕ := 2
def num_cheese_fries : ℕ := 1

-- Define Jim's contribution
def jim_money : ℚ := 20

-- Define the percentage of combined money spent
def percentage_spent : ℚ := 80 / 100

-- Theorem to prove
theorem jims_cousin_money :
  let total_cost := num_cheeseburgers * cheeseburger_cost + 
                    num_milkshakes * milkshake_cost + 
                    num_cheese_fries * cheese_fries_cost
  let total_money := total_cost / percentage_spent
  let cousin_money := total_money - jim_money
  cousin_money = 10 := by sorry

end NUMINAMATH_CALUDE_jims_cousin_money_l1952_195227


namespace NUMINAMATH_CALUDE_part_1_part_2_part_3_part_3_unique_l1952_195220

-- Define the algebraic expression
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Part 1
theorem part_1 : f 2 = -1 := by sorry

-- Part 2
theorem part_2 : 
  ∃ x₁ x₂ : ℝ, f x₁ = 4 ∧ f x₂ = 4 ∧ x₁^2 + x₂^2 = 18 := by sorry

-- Part 3
theorem part_3 :
  ∃ m : ℕ, 
    (∃ n₁ n₂ : ℝ, 
      f (m + 2) = n₁ ∧ 
      f (2*m + 1) = n₂ ∧ 
      ∃ k : ℤ, n₂ / n₁ = k) ∧
    m = 3 := by sorry

-- Additional theorem to show the uniqueness of m in part 3
theorem part_3_unique :
  ∀ m : ℕ, 
    (∃ n₁ n₂ : ℝ, 
      f (m + 2) = n₁ ∧ 
      f (2*m + 1) = n₂ ∧ 
      ∃ k : ℤ, n₂ / n₁ = k) →
    m = 3 := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_part_3_part_3_unique_l1952_195220


namespace NUMINAMATH_CALUDE_relationship_xyz_l1952_195222

theorem relationship_xyz (x y z : ℝ) 
  (h1 : x - y > x + z) 
  (h2 : x + y < y + z) : 
  y < -z ∧ x < z := by
sorry

end NUMINAMATH_CALUDE_relationship_xyz_l1952_195222


namespace NUMINAMATH_CALUDE_bead_problem_solutions_l1952_195231

/-- Represents the possible total number of beads -/
def PossibleTotals : Set ℕ := {107, 109, 111, 113, 115, 117}

/-- Represents a solution to the bead problem -/
structure BeadSolution where
  x : ℕ -- number of 19-gram beads
  y : ℕ -- number of 17-gram beads

/-- Checks if a BeadSolution is valid -/
def isValidSolution (s : BeadSolution) : Prop :=
  19 * s.x + 17 * s.y = 2017 ∧ s.x + s.y ∈ PossibleTotals

/-- Theorem stating that there exist valid solutions for all possible totals -/
theorem bead_problem_solutions :
  ∀ n ∈ PossibleTotals, ∃ s : BeadSolution, isValidSolution s ∧ s.x + s.y = n :=
sorry

end NUMINAMATH_CALUDE_bead_problem_solutions_l1952_195231


namespace NUMINAMATH_CALUDE_system_solution_l1952_195256

theorem system_solution (x y z : ℝ) : 
  (x^2 - y*z = |y - z| + 1 ∧
   y^2 - z*x = |z - x| + 1 ∧
   z^2 - x*y = |x - y| + 1) ↔ 
  ((x = 5/3 ∧ y = -4/3 ∧ z = -4/3) ∨
   (x = 4/3 ∧ y = 4/3 ∧ z = -5/3) ∨
   (x = 5/3 ∧ y = -4/3 ∧ z = -4/3) ∨
   (x = -4/3 ∧ y = 5/3 ∧ z = -4/3) ∨
   (x = -4/3 ∧ y = -4/3 ∧ z = 5/3) ∨
   (x = 4/3 ∧ y = 4/3 ∧ z = -5/3) ∨
   (x = 4/3 ∧ y = -5/3 ∧ z = 4/3) ∨
   (x = -5/3 ∧ y = 4/3 ∧ z = 4/3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1952_195256


namespace NUMINAMATH_CALUDE_system_solution_correct_l1952_195290

theorem system_solution_correct (x y : ℚ) :
  x = 2 ∧ y = 1/2 → (x - 2*y = 1 ∧ 2*x + 2*y = 5) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_correct_l1952_195290


namespace NUMINAMATH_CALUDE_all_props_true_l1952_195278

-- Define the original proposition
def original_prop (x : ℝ) : Prop := x^2 - 3*x + 2 ≠ 0 → x ≠ 1 ∧ x ≠ 2

-- Define the converse proposition
def converse_prop (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ 2 → x^2 - 3*x + 2 ≠ 0

-- Define the inverse proposition
def inverse_prop (x : ℝ) : Prop := x^2 - 3*x + 2 = 0 → x = 1 ∨ x = 2

-- Define the contrapositive proposition
def contrapositive_prop (x : ℝ) : Prop := x = 1 ∨ x = 2 → x^2 - 3*x + 2 = 0

-- Theorem stating that all propositions are true
theorem all_props_true : 
  (∀ x : ℝ, original_prop x) ∧ 
  (∀ x : ℝ, converse_prop x) ∧ 
  (∀ x : ℝ, inverse_prop x) ∧ 
  (∀ x : ℝ, contrapositive_prop x) :=
sorry

end NUMINAMATH_CALUDE_all_props_true_l1952_195278


namespace NUMINAMATH_CALUDE_final_price_percentage_l1952_195296

/-- The price change over 5 years -/
def price_change (p : ℝ) : ℝ :=
  p * 1.3 * 0.8 * 1.25 * 0.9 * 1.15

/-- Theorem stating the final price is 134.55% of the original price -/
theorem final_price_percentage (p : ℝ) (hp : p > 0) :
  price_change p / p = 1.3455 := by
  sorry

end NUMINAMATH_CALUDE_final_price_percentage_l1952_195296


namespace NUMINAMATH_CALUDE_certain_number_equation_l1952_195294

theorem certain_number_equation (x : ℝ) : 15 * x + 16 * x + 19 * x + 11 = 161 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1952_195294


namespace NUMINAMATH_CALUDE_cubic_inequality_l1952_195218

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 + 27*x > 0 ↔ x ∈ Set.union (Set.Ioo 0 3) (Set.Ioi 9) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1952_195218


namespace NUMINAMATH_CALUDE_bedroom_set_final_price_l1952_195248

def original_price : ℝ := 2000
def gift_cards : ℝ := 200
def first_discount_rate : ℝ := 0.15
def second_discount_rate : ℝ := 0.10

def final_price : ℝ :=
  let price_after_first_discount := original_price * (1 - first_discount_rate)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount_rate)
  price_after_second_discount - gift_cards

theorem bedroom_set_final_price :
  final_price = 1330 := by sorry

end NUMINAMATH_CALUDE_bedroom_set_final_price_l1952_195248


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1952_195276

-- Define the curve
def f (x : ℝ) : ℝ := 2 * x^3 + x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 6 * x^2 + 1

-- Theorem statement
theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ 7 * x - y - 4 = 0 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1952_195276


namespace NUMINAMATH_CALUDE_problem_solution_l1952_195267

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x - m < 0}

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - 3*a) * (x - a - 2) < 0}

theorem problem_solution :
  (B = Set.Ioi 2) ∧
  ({a : ℝ | A a ⊆ B} = Set.Ici (2/3)) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1952_195267


namespace NUMINAMATH_CALUDE_team_b_city_a_matches_l1952_195273

/-- Represents a team in the tournament -/
structure Team where
  city : Fin 16
  isTeamA : Bool

/-- The number of matches played by a team -/
def matchesPlayed (t : Team) : ℕ := sorry

/-- The tournament satisfies the given conditions -/
axiom tournament_conditions :
  ∀ t1 t2 : Team,
    (t1 ≠ t2) →
    (t1.city ≠ t2.city ∨ t1.isTeamA ≠ t2.isTeamA) →
    (t1 ≠ ⟨0, true⟩) →
    (t2 ≠ ⟨0, true⟩) →
    matchesPlayed t1 ≠ matchesPlayed t2

/-- All teams except one have played between 0 and 30 matches -/
axiom matches_range :
  ∀ t : Team, t ≠ ⟨0, true⟩ → matchesPlayed t ≤ 30

/-- The theorem to be proved -/
theorem team_b_city_a_matches :
  matchesPlayed ⟨0, false⟩ = 15 := by sorry

end NUMINAMATH_CALUDE_team_b_city_a_matches_l1952_195273


namespace NUMINAMATH_CALUDE_area_of_closed_figure_l1952_195234

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3*x

theorem area_of_closed_figure : 
  ∫ x in (1/2)..1, (1/x + 2*x - 3) = 3/4 - Real.log 2 := by sorry

end NUMINAMATH_CALUDE_area_of_closed_figure_l1952_195234


namespace NUMINAMATH_CALUDE_ship_departure_theorem_l1952_195285

/-- Represents the travel times and expected delivery for a cargo shipment --/
structure CargoShipment where
  travelDays : ℕ        -- Days for ship travel
  customsDays : ℕ       -- Days for customs processing
  deliveryDays : ℕ      -- Days from port to warehouse
  expectedArrival : ℕ   -- Days until expected arrival at warehouse

/-- Calculates the number of days ago the ship should have departed --/
def departureDays (shipment : CargoShipment) : ℕ :=
  shipment.travelDays + shipment.customsDays + shipment.deliveryDays - shipment.expectedArrival

/-- Theorem stating that for the given conditions, the ship should have departed 30 days ago --/
theorem ship_departure_theorem (shipment : CargoShipment) 
  (h1 : shipment.travelDays = 21)
  (h2 : shipment.customsDays = 4)
  (h3 : shipment.deliveryDays = 7)
  (h4 : shipment.expectedArrival = 2) :
  departureDays shipment = 30 := by
  sorry

#eval departureDays { travelDays := 21, customsDays := 4, deliveryDays := 7, expectedArrival := 2 }

end NUMINAMATH_CALUDE_ship_departure_theorem_l1952_195285


namespace NUMINAMATH_CALUDE_sum_of_squares_and_products_l1952_195221

theorem sum_of_squares_and_products (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 52)
  (sum_of_products : x*y + y*z + z*x = 28) :
  x + y + z = 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_products_l1952_195221


namespace NUMINAMATH_CALUDE_road_trip_total_distance_l1952_195253

/-- Represents the road trip with given conditions -/
def RoadTrip (x : ℝ) : Prop :=
  let first_leg := x
  let second_leg := 2 * x
  let third_leg := 40
  let final_leg := 2 * (first_leg + second_leg + third_leg)
  (third_leg = x / 2) ∧
  (first_leg + second_leg + third_leg + final_leg = 840)

/-- Theorem stating the total distance of the road trip -/
theorem road_trip_total_distance : ∃ x : ℝ, RoadTrip x :=
  sorry

end NUMINAMATH_CALUDE_road_trip_total_distance_l1952_195253


namespace NUMINAMATH_CALUDE_ratio_between_zero_and_one_l1952_195271

theorem ratio_between_zero_and_one : 
  let A : ℕ := 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28
  let B : ℕ := 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20
  0 < (A : ℚ) / B ∧ (A : ℚ) / B < 1 :=
by sorry

end NUMINAMATH_CALUDE_ratio_between_zero_and_one_l1952_195271


namespace NUMINAMATH_CALUDE_tan70_cos10_expression_equals_one_l1952_195283

theorem tan70_cos10_expression_equals_one :
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (1 - Real.sqrt 3 * Real.tan (20 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan70_cos10_expression_equals_one_l1952_195283


namespace NUMINAMATH_CALUDE_bag_problem_l1952_195277

/-- The number of red balls in the bag -/
def red_balls (a : ℕ) : ℕ := a + 1

/-- The number of yellow balls in the bag -/
def yellow_balls (a : ℕ) : ℕ := a

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 1

/-- The total number of balls in the bag -/
def total_balls (a : ℕ) : ℕ := red_balls a + yellow_balls a + blue_balls

/-- The score earned by drawing a red ball -/
def red_score : ℕ := 1

/-- The score earned by drawing a yellow ball -/
def yellow_score : ℕ := 2

/-- The score earned by drawing a blue ball -/
def blue_score : ℕ := 3

/-- The expected value of the score when drawing a ball -/
def expected_value : ℚ := 5 / 3

theorem bag_problem (a : ℕ) :
  (a = 2) ∧
  (let p : ℚ := (Nat.choose 3 1 * Nat.choose 2 2 + Nat.choose 3 2 * Nat.choose 1 1) / Nat.choose 6 3
   p = 3 / 10) :=
by sorry

end NUMINAMATH_CALUDE_bag_problem_l1952_195277


namespace NUMINAMATH_CALUDE_base_4_of_185_l1952_195211

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (a b c d : ℕ) : ℕ :=
  a * (4^3) + b * (4^2) + c * (4^1) + d * (4^0)

/-- The base 4 representation of 185 (base 10) is 2321 --/
theorem base_4_of_185 : base4ToBase10 2 3 2 1 = 185 := by
  sorry

end NUMINAMATH_CALUDE_base_4_of_185_l1952_195211


namespace NUMINAMATH_CALUDE_notebook_words_per_page_l1952_195247

theorem notebook_words_per_page :
  ∀ (words_per_page : ℕ),
    words_per_page > 0 →
    words_per_page ≤ 150 →
    (180 * words_per_page) % 221 = 246 % 221 →
    words_per_page = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_words_per_page_l1952_195247


namespace NUMINAMATH_CALUDE_inequality_proof_l1952_195258

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_condition : a + b + c + d = 1) :
  a * b * c + b * c * d + c * d * a + d * a * b ≤ 1 / 27 + 176 / 27 * a * b * c * d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1952_195258


namespace NUMINAMATH_CALUDE_child_support_calculation_l1952_195249

def child_support_owed (base_salary : List ℝ) (bonuses : List ℝ) (rates : List ℝ) (paid : ℝ) : ℝ :=
  let incomes := List.zipWith (· + ·) base_salary bonuses
  let owed := List.sum (List.zipWith (· * ·) incomes rates)
  owed - paid

theorem child_support_calculation : 
  let base_salary := [30000, 30000, 30000, 36000, 36000, 36000, 36000]
  let bonuses := [2000, 3000, 4000, 5000, 6000, 7000, 8000]
  let rates := [0.3, 0.3, 0.3, 0.3, 0.3, 0.25, 0.25]
  let paid := 1200
  child_support_owed base_salary bonuses rates paid = 75150 := by
  sorry

#eval child_support_owed 
  [30000, 30000, 30000, 36000, 36000, 36000, 36000]
  [2000, 3000, 4000, 5000, 6000, 7000, 8000]
  [0.3, 0.3, 0.3, 0.3, 0.3, 0.25, 0.25]
  1200

end NUMINAMATH_CALUDE_child_support_calculation_l1952_195249


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_six_l1952_195270

theorem opposite_of_sqrt_six :
  ∀ x : ℝ, x = Real.sqrt 6 → -x = -(Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_six_l1952_195270


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l1952_195268

/-- The number of ways to arrange 6 rings out of 10 distinguishable rings on 5 fingers,
    where the order on each finger matters and no finger can have more than 3 rings. -/
def ring_arrangements : ℕ :=
  let total_rings : ℕ := 10
  let rings_used : ℕ := 6
  let num_fingers : ℕ := 5
  let max_rings_per_finger : ℕ := 3
  -- The actual calculation would go here, but we'll use the result directly
  145152000

/-- Theorem stating that the number of ring arrangements is 145,152,000 -/
theorem ring_arrangements_count : ring_arrangements = 145152000 := by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l1952_195268


namespace NUMINAMATH_CALUDE_adjacent_difference_at_least_six_l1952_195224

/-- A 9x9 table containing integers from 1 to 81 -/
def Table : Type := Fin 9 → Fin 9 → Fin 81

/-- Two cells are adjacent if they share a side -/
def adjacent (a b : Fin 9 × Fin 9) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ b.2 = a.2 + 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ b.1 = a.1 + 1))

/-- The table contains all integers from 1 to 81 exactly once -/
def valid_table (t : Table) : Prop :=
  ∀ n : Fin 81, ∃! (i j : Fin 9), t i j = n

theorem adjacent_difference_at_least_six (t : Table) (h : valid_table t) :
  ∃ (a b : Fin 9 × Fin 9), adjacent a b ∧ 
    ((t a.1 a.2).val + 6 ≤ (t b.1 b.2).val ∨ (t b.1 b.2).val + 6 ≤ (t a.1 a.2).val) :=
sorry

end NUMINAMATH_CALUDE_adjacent_difference_at_least_six_l1952_195224


namespace NUMINAMATH_CALUDE_line_y_intercept_l1952_195252

/-- Given a line with slope 4 passing through the point (50, 300), prove that its y-intercept is 100. -/
theorem line_y_intercept (m : ℝ) (x y b : ℝ) :
  m = 4 →
  x = 50 →
  y = 300 →
  y = m * x + b →
  b = 100 := by
sorry

end NUMINAMATH_CALUDE_line_y_intercept_l1952_195252


namespace NUMINAMATH_CALUDE_common_zero_implies_f0_or_f1_zero_l1952_195233

/-- A quadratic function f(x) = x^2 + px + q -/
def f (p q x : ℝ) : ℝ := x^2 + p * x + q

/-- The composition f(f(f(x))) -/
def triple_f (p q x : ℝ) : ℝ := f p q (f p q (f p q x))

/-- Theorem: If f and triple_f have a common zero, then f(0) = 0 or f(1) = 0 -/
theorem common_zero_implies_f0_or_f1_zero (p q : ℝ) :
  (∃ m, f p q m = 0 ∧ triple_f p q m = 0) →
  f p q 0 = 0 ∨ f p q 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_common_zero_implies_f0_or_f1_zero_l1952_195233


namespace NUMINAMATH_CALUDE_abs_value_difference_l1952_195229

theorem abs_value_difference (x y : ℝ) (hx : |x| = 2) (hy : |y| = 3) (hxy : x > y) :
  x - y = 5 ∨ x - y = 1 := by
sorry

end NUMINAMATH_CALUDE_abs_value_difference_l1952_195229


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l1952_195223

-- Define the polynomial
def p (x : ℝ) : ℝ := 3 * x^3 - 9 * x^2 + 12 * x - 12

-- Define the factors
def factor1 (x : ℝ) : ℝ := x - 2
def factor2 (x : ℝ) : ℝ := 3 * x^2 - 4

-- Theorem statement
theorem polynomial_divisibility :
  (∃ q1 q2 : ℝ → ℝ, ∀ x, p x = factor1 x * q1 x ∧ p x = factor2 x * q2 x) :=
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l1952_195223


namespace NUMINAMATH_CALUDE_baron_munchausen_claim_l1952_195255

-- Define a function to calculate the sum of squares of digits
def sumOfSquaresOfDigits (n : ℕ) : ℕ := sorry

-- Define the property for the numbers we're looking for
def satisfiesProperty (a b : ℕ) : Prop :=
  (a ≠ b) ∧ 
  (a ≥ 10^9) ∧ (a < 10^10) ∧ 
  (b ≥ 10^9) ∧ (b < 10^10) ∧ 
  (a % 10 ≠ 0) ∧ (b % 10 ≠ 0) ∧
  (a - sumOfSquaresOfDigits a = b - sumOfSquaresOfDigits b)

-- Theorem statement
theorem baron_munchausen_claim : ∃ a b : ℕ, satisfiesProperty a b := by sorry

end NUMINAMATH_CALUDE_baron_munchausen_claim_l1952_195255


namespace NUMINAMATH_CALUDE_max_b_value_l1952_195280

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 4*a^2 * log x + b

theorem max_b_value (a : ℝ) (h_a : a > 0) :
  (∃ x₀ : ℝ, f a x₀ = g a b x₀ ∧ deriv (f a) x₀ = deriv (g a b) x₀) →
  (∃ b : ℝ, ∀ b' : ℝ, (∃ x₀ : ℝ, f a x₀ = g a b' x₀ ∧ deriv (f a) x₀ = deriv (g a b') x₀) → b' ≤ b) →
  (∃ b : ℝ, (∃ x₀ : ℝ, f a x₀ = g a b x₀ ∧ deriv (f a) x₀ = deriv (g a b) x₀) ∧
             ∀ b' : ℝ, (∃ x₀ : ℝ, f a x₀ = g a b' x₀ ∧ deriv (f a) x₀ = deriv (g a b') x₀) → b' ≤ b) →
  b = 2 * sqrt e :=
by sorry

end NUMINAMATH_CALUDE_max_b_value_l1952_195280


namespace NUMINAMATH_CALUDE_probability_two_red_one_green_l1952_195274

def red_shoes : ℕ := 6
def green_shoes : ℕ := 8
def blue_shoes : ℕ := 5
def yellow_shoes : ℕ := 3

def total_shoes : ℕ := red_shoes + green_shoes + blue_shoes + yellow_shoes

def draw_count : ℕ := 3

theorem probability_two_red_one_green :
  (Nat.choose red_shoes 2 * Nat.choose green_shoes 1) / Nat.choose total_shoes draw_count = 6 / 77 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_one_green_l1952_195274


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1952_195286

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₄ = 1 and a₇ + a₉ = 16, prove that a₁₂ = 15 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) 
    (h_a4 : a 4 = 1) 
    (h_sum : a 7 + a 9 = 16) : 
  a 12 = 15 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1952_195286


namespace NUMINAMATH_CALUDE_max_product_sum_2020_l1952_195201

theorem max_product_sum_2020 : 
  (∃ (a b : ℤ), a + b = 2020 ∧ a * b = 1020100) ∧ 
  (∀ (x y : ℤ), x + y = 2020 → x * y ≤ 1020100) := by
sorry

end NUMINAMATH_CALUDE_max_product_sum_2020_l1952_195201


namespace NUMINAMATH_CALUDE_concert_tickets_sold_l1952_195297

theorem concert_tickets_sold (cost_A cost_B total_tickets total_revenue : ℚ)
  (h1 : cost_A = 8)
  (h2 : cost_B = 4.25)
  (h3 : total_tickets = 4500)
  (h4 : total_revenue = 30000)
  : ∃ (tickets_A tickets_B : ℚ),
    tickets_A + tickets_B = total_tickets ∧
    cost_A * tickets_A + cost_B * tickets_B = total_revenue ∧
    tickets_A = 2900 := by
  sorry

end NUMINAMATH_CALUDE_concert_tickets_sold_l1952_195297


namespace NUMINAMATH_CALUDE_parallelogram_angle_difference_parallelogram_angle_difference_holds_l1952_195200

/-- In a parallelogram, given that one angle measures 85 degrees, 
    the difference between this angle and its adjacent angle is 10 degrees. -/
theorem parallelogram_angle_difference : ℝ → Prop :=
  fun angle_difference : ℝ =>
    ∀ (smaller_angle larger_angle : ℝ),
      smaller_angle = 85 ∧
      smaller_angle + larger_angle = 180 →
      larger_angle - smaller_angle = angle_difference ∧
      angle_difference = 10

/-- The theorem holds for the given angle difference. -/
theorem parallelogram_angle_difference_holds : parallelogram_angle_difference 10 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_angle_difference_parallelogram_angle_difference_holds_l1952_195200


namespace NUMINAMATH_CALUDE_definite_integral_problem_l1952_195209

open Real MeasureTheory Interval

theorem definite_integral_problem :
  ∫ x in (-1 : ℝ)..1, x * cos x + (x^2)^(1/3) = 6/5 := by sorry

end NUMINAMATH_CALUDE_definite_integral_problem_l1952_195209


namespace NUMINAMATH_CALUDE_coronavirus_state_after_three_days_l1952_195288

/-- Represents the state of Coronavirus cases on a given day -/
structure CoronavirusState where
  positiveCase : ℕ
  hospitalizedCase : ℕ
  deaths : ℕ

/-- Calculates the next day's Coronavirus state based on the current state and rates -/
def nextDayState (state : CoronavirusState) (newCaseRate : ℝ) (recoveryRate : ℝ) 
                 (hospitalizationRate : ℝ) (hospitalizationIncreaseRate : ℝ)
                 (deathRate : ℝ) (deathIncreaseRate : ℝ) : CoronavirusState :=
  sorry

/-- Theorem stating the Coronavirus state after 3 days given initial conditions -/
theorem coronavirus_state_after_three_days 
  (initialState : CoronavirusState)
  (newCaseRate : ℝ)
  (recoveryRate : ℝ)
  (hospitalizationRate : ℝ)
  (hospitalizationIncreaseRate : ℝ)
  (deathRate : ℝ)
  (deathIncreaseRate : ℝ)
  (h1 : initialState.positiveCase = 2000)
  (h2 : newCaseRate = 0.15)
  (h3 : recoveryRate = 0.05)
  (h4 : hospitalizationRate = 0.03)
  (h5 : hospitalizationIncreaseRate = 0.10)
  (h6 : deathRate = 0.02)
  (h7 : deathIncreaseRate = 0.05) :
  let day1 := nextDayState initialState newCaseRate recoveryRate hospitalizationRate hospitalizationIncreaseRate deathRate deathIncreaseRate
  let day2 := nextDayState day1 newCaseRate recoveryRate hospitalizationRate hospitalizationIncreaseRate deathRate deathIncreaseRate
  let day3 := nextDayState day2 newCaseRate recoveryRate hospitalizationRate hospitalizationIncreaseRate deathRate deathIncreaseRate
  day3.positiveCase = 2420 ∧ day3.hospitalizedCase = 92 ∧ day3.deaths = 57 :=
sorry

end NUMINAMATH_CALUDE_coronavirus_state_after_three_days_l1952_195288


namespace NUMINAMATH_CALUDE_gcd_factorial_problem_l1952_195203

theorem gcd_factorial_problem : Nat.gcd (Nat.factorial 7) ((Nat.factorial 10) / (Nat.factorial 4)) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_gcd_factorial_problem_l1952_195203
