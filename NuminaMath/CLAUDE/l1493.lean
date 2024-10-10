import Mathlib

namespace set_difference_N_M_l1493_149385

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {1, 2, 3, 7}

theorem set_difference_N_M : N \ M = {7} := by
  sorry

end set_difference_N_M_l1493_149385


namespace fishing_ratio_l1493_149309

/-- Given that Tom caught 16 trout and Melanie caught 8 trout, 
    prove that the ratio of Tom's catch to Melanie's catch is 2. -/
theorem fishing_ratio (tom_catch melanie_catch : ℕ) 
  (h1 : tom_catch = 16) (h2 : melanie_catch = 8) : 
  (tom_catch : ℚ) / melanie_catch = 2 := by
  sorry

end fishing_ratio_l1493_149309


namespace work_earnings_equation_l1493_149308

theorem work_earnings_equation (t : ℚ) : (t + 2) * (4 * t - 5) = (2 * t + 1) * (2 * t + 3) + 3 ↔ t = -16/3 := by
  sorry

end work_earnings_equation_l1493_149308


namespace garrison_provision_theorem_l1493_149352

/-- Calculates the initial number of days provisions were supposed to last for a garrison --/
def initial_provision_days (initial_garrison : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) : ℕ :=
  (initial_garrison + reinforcement) * days_after_reinforcement / initial_garrison + days_before_reinforcement

theorem garrison_provision_theorem (initial_garrison : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) (days_after_reinforcement : ℕ) :
  initial_garrison = 2000 →
  reinforcement = 600 →
  days_before_reinforcement = 15 →
  days_after_reinforcement = 30 →
  initial_provision_days initial_garrison reinforcement days_before_reinforcement days_after_reinforcement = 39 :=
by
  sorry

#eval initial_provision_days 2000 600 15 30

end garrison_provision_theorem_l1493_149352


namespace binary_10110011_equals_179_l1493_149399

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_10110011_equals_179 :
  binary_to_decimal [true, true, false, false, true, true, false, true] = 179 := by
  sorry

end binary_10110011_equals_179_l1493_149399


namespace eugene_apples_proof_l1493_149375

def apples_from_eugene (initial_apples final_apples : ℝ) : ℝ :=
  final_apples - initial_apples

theorem eugene_apples_proof (initial_apples final_apples : ℝ) :
  apples_from_eugene initial_apples final_apples =
  final_apples - initial_apples :=
by
  sorry

#eval apples_from_eugene 20.0 27.0

end eugene_apples_proof_l1493_149375


namespace min_value_reciprocal_sum_l1493_149340

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 2) :
  (1/a + 2/b) ≥ 9/2 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2*b₀ = 2 ∧ 1/a₀ + 2/b₀ = 9/2 :=
by sorry

end min_value_reciprocal_sum_l1493_149340


namespace complex_equality_l1493_149339

theorem complex_equality (z : ℂ) : z = -1 + I →
  Complex.abs (z - 2) = Complex.abs (z + 4) ∧
  Complex.abs (z - 2) = Complex.abs (z + 2*I) := by
  sorry

end complex_equality_l1493_149339


namespace min_sum_product_2400_l1493_149371

theorem min_sum_product_2400 (x y z : ℕ+) (h : x * y * z = 2400) :
  x + y + z ≥ 43 := by
  sorry

end min_sum_product_2400_l1493_149371


namespace negation_equivalence_function_property_l1493_149331

-- Define the statement for the negation of the existential proposition
theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
sorry

-- Define the properties for functions f and g
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def positive_derivative_pos (h : ℝ → ℝ) : Prop := ∀ x > 0, deriv h x > 0

-- Theorem for the properties of functions f and g
theorem function_property (f g : ℝ → ℝ) 
  (hodd : odd_function f) (heven : even_function g)
  (hf_deriv : positive_derivative_pos f) (hg_deriv : positive_derivative_pos g) :
  ∀ x < 0, deriv f x > deriv g x :=
sorry

end negation_equivalence_function_property_l1493_149331


namespace complex_sum_argument_l1493_149338

theorem complex_sum_argument : 
  let z : ℂ := Complex.exp (11 * Real.pi * I / 60) + 
                Complex.exp (31 * Real.pi * I / 60) + 
                Complex.exp (51 * Real.pi * I / 60) + 
                Complex.exp (71 * Real.pi * I / 60) + 
                Complex.exp (91 * Real.pi * I / 60)
  ∃ (r : ℝ), z = r * Complex.exp (17 * Real.pi * I / 20) ∧ r > 0 :=
by sorry

end complex_sum_argument_l1493_149338


namespace arithmetic_sequence_sum_l1493_149380

/-- Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
    if S_m = 2 and S_2m = 10, then S_3m = 24. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (m : ℕ) :
  (∀ n, S n = (n : ℝ) * (a 1 + a n) / 2) →  -- Definition of S_n for arithmetic sequence
  (S m = 2) →
  (S (2 * m) = 10) →
  (S (3 * m) = 24) := by
sorry

end arithmetic_sequence_sum_l1493_149380


namespace simplify_expression_l1493_149355

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (4 * a^3 * b - 2 * a * b) / (2 * a * b) = 2 * a^2 - 1 := by
  sorry

end simplify_expression_l1493_149355


namespace side_face_area_is_288_l1493_149343

/-- Represents a rectangular box with length, width, and height. -/
structure RectangularBox where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular box. -/
def volume (box : RectangularBox) : ℝ :=
  box.length * box.width * box.height

/-- Calculates the area of the front face of a rectangular box. -/
def frontFaceArea (box : RectangularBox) : ℝ :=
  box.length * box.width

/-- Calculates the area of the top face of a rectangular box. -/
def topFaceArea (box : RectangularBox) : ℝ :=
  box.length * box.height

/-- Calculates the area of the side face of a rectangular box. -/
def sideFaceArea (box : RectangularBox) : ℝ :=
  box.width * box.height

/-- Theorem stating that given the conditions, the area of the side face is 288. -/
theorem side_face_area_is_288 (box : RectangularBox) 
  (h1 : frontFaceArea box = (1/2) * topFaceArea box)
  (h2 : topFaceArea box = (3/2) * sideFaceArea box)
  (h3 : volume box = 5184) :
  sideFaceArea box = 288 := by
  sorry

end side_face_area_is_288_l1493_149343


namespace pentagon_angle_measure_l1493_149368

/-- In a pentagon with angles 104°, 97°, x°, 2x°, and R°, where the sum of all angles is 540°, 
    the measure of angle R is 204°. -/
theorem pentagon_angle_measure (x : ℝ) (R : ℝ) : 
  104 + 97 + x + 2*x + R = 540 → R = 204 := by
  sorry

end pentagon_angle_measure_l1493_149368


namespace ruby_reading_homework_l1493_149390

theorem ruby_reading_homework (nina_math : ℕ) (nina_reading : ℕ) (ruby_math : ℕ) (ruby_reading : ℕ) :
  nina_math = 4 * ruby_math →
  nina_reading = 8 * ruby_reading →
  ruby_math = 6 →
  nina_math + nina_reading = 48 →
  ruby_reading = 3 := by
sorry

end ruby_reading_homework_l1493_149390


namespace roses_per_bouquet_l1493_149350

/-- Proves that the number of roses in each bouquet is 12 given the problem conditions -/
theorem roses_per_bouquet (total_bouquets : ℕ) (rose_bouquets : ℕ) (daisy_bouquets : ℕ) 
  (total_flowers : ℕ) (daisies_per_bouquet : ℕ) :
  total_bouquets = 20 →
  rose_bouquets = 10 →
  daisy_bouquets = 10 →
  rose_bouquets + daisy_bouquets = total_bouquets →
  total_flowers = 190 →
  daisies_per_bouquet = 7 →
  (total_flowers - daisy_bouquets * daisies_per_bouquet) / rose_bouquets = 12 :=
by sorry

end roses_per_bouquet_l1493_149350


namespace max_value_of_roots_expression_l1493_149320

theorem max_value_of_roots_expression (a : ℝ) (x₁ x₂ : ℝ) :
  x₁^2 + a*x₁ + a = 2 →
  x₂^2 + a*x₂ + a = 2 →
  x₁ ≠ x₂ →
  ∀ b : ℝ, (x₁ - 2*x₂)*(x₂ - 2*x₁) ≤ -63/8 :=
by sorry

end max_value_of_roots_expression_l1493_149320


namespace power_of_prime_exponent_l1493_149361

theorem power_of_prime_exponent (x y p n k : ℕ) 
  (h_n_gt_1 : n > 1)
  (h_n_odd : Odd n)
  (h_p_prime : Nat.Prime p)
  (h_p_odd : Odd p)
  (h_eq : x^n + y^n = p^k) :
  ∃ m : ℕ, n = p^m :=
sorry

end power_of_prime_exponent_l1493_149361


namespace remaining_seeds_l1493_149346

theorem remaining_seeds (initial_seeds : ℕ) (seeds_per_zone : ℕ) (num_zones : ℕ) : 
  initial_seeds = 54000 →
  seeds_per_zone = 3123 →
  num_zones = 7 →
  initial_seeds - (seeds_per_zone * num_zones) = 32139 :=
by
  sorry

end remaining_seeds_l1493_149346


namespace salt_concentration_increase_l1493_149306

/-- Given a 100 kg solution with 10% salt concentration and adding 20 kg of pure salt,
    the final salt concentration is 25%. -/
theorem salt_concentration_increase (initial_solution : ℝ) (initial_concentration : ℝ) 
    (added_salt : ℝ) (final_concentration : ℝ) : 
    initial_solution = 100 →
    initial_concentration = 0.1 →
    added_salt = 20 →
    final_concentration = (initial_solution * initial_concentration + added_salt) / 
                          (initial_solution + added_salt) →
    final_concentration = 0.25 := by
  sorry

end salt_concentration_increase_l1493_149306


namespace january_salary_l1493_149321

theorem january_salary (feb mar apr may : ℕ) 
  (h1 : (feb + mar + apr + may) / 4 = 8300)
  (h2 : may = 6500)
  (h3 : ∃ jan, (jan + feb + mar + apr) / 4 = 8000) :
  ∃ jan, (jan + feb + mar + apr) / 4 = 8000 ∧ jan = 5300 := by
  sorry

end january_salary_l1493_149321


namespace f_not_in_second_quadrant_l1493_149391

/-- A linear function f(x) = 2x - 3 -/
def f (x : ℝ) : ℝ := 2 * x - 3

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- Theorem: The graph of f(x) = 2x - 3 does not pass through the second quadrant -/
theorem f_not_in_second_quadrant :
  ∀ x y : ℝ, f x = y → ¬(second_quadrant x y) :=
sorry

end f_not_in_second_quadrant_l1493_149391


namespace linear_function_max_value_l1493_149315

/-- The maximum value of a linear function y = (5/3)x + 2 over the interval [-3, 3] is 7 -/
theorem linear_function_max_value (x : ℝ) :
  x ∈ Set.Icc (-3 : ℝ) 3 →
  (5/3 : ℝ) * x + 2 ≤ 7 ∧ ∃ x₀, x₀ ∈ Set.Icc (-3 : ℝ) 3 ∧ (5/3 : ℝ) * x₀ + 2 = 7 :=
by sorry

end linear_function_max_value_l1493_149315


namespace decimal_to_fraction_l1493_149326

theorem decimal_to_fraction : (2.35 : ℚ) = 47 / 20 := by sorry

end decimal_to_fraction_l1493_149326


namespace max_value_ab_l1493_149397

theorem max_value_ab (a b : ℕ) : 
  a > 1 → b > 1 → a^b * b^a + a^b + b^a = 5329 → a^b ≤ 64 := by
sorry

end max_value_ab_l1493_149397


namespace no_valid_a_l1493_149363

theorem no_valid_a : ∀ a : ℝ, a > 0 → ∃ x : ℝ, |Real.cos x| + |Real.cos (a * x)| ≤ Real.sin x + Real.sin (a * x) := by
  sorry

end no_valid_a_l1493_149363


namespace apples_joan_can_buy_l1493_149302

def total_budget : ℕ := 60
def hummus_containers : ℕ := 2
def hummus_price : ℕ := 5
def chicken_price : ℕ := 20
def bacon_price : ℕ := 10
def vegetables_price : ℕ := 10
def apple_price : ℕ := 2

theorem apples_joan_can_buy :
  (total_budget - (hummus_containers * hummus_price + chicken_price + bacon_price + vegetables_price)) / apple_price = 5 := by
  sorry

end apples_joan_can_buy_l1493_149302


namespace f_is_log_x_range_l1493_149304

noncomputable section

variable (a : ℝ) (f g : ℝ → ℝ)

-- Define g(x) = a^x
def g_def : g = fun x ↦ a^x := by sorry

-- Define f(x) as symmetric to g(x) with respect to y = x
def f_symmetric : ∀ x y, f x = y ↔ g y = x := by sorry

-- Part 1: Prove that f(x) = log_a x
theorem f_is_log : f = fun x ↦ Real.log x / Real.log a := by sorry

-- Part 2: Prove the range of x when a > 1 and f(x) < f(2)
theorem x_range (h : a > 1) : 
  ∀ x, f x < f 2 ↔ 0 < x ∧ x < a^2 := by sorry

end

end f_is_log_x_range_l1493_149304


namespace inequality_proof_l1493_149328

theorem inequality_proof (a b : ℝ) (h : 1/a > 1/b ∧ 1/b > 0) : 
  a^3 < b^3 ∧ Real.sqrt b - Real.sqrt a < Real.sqrt (b - a) := by
  sorry

end inequality_proof_l1493_149328


namespace real_part_of_z_l1493_149357

theorem real_part_of_z (z : ℂ) (h : z * (2 + Complex.I) = 1) : 
  Complex.re z = 2/5 := by
  sorry

end real_part_of_z_l1493_149357


namespace odd_three_digit_count_l1493_149313

/-- The set of digits that can be used for the first digit -/
def first_digit_set : Finset Nat := {0, 2}

/-- The set of digits that can be used for the second and third digits -/
def odd_digit_set : Finset Nat := {1, 3, 5}

/-- A function to check if a number is odd -/
def is_odd (n : Nat) : Bool := n % 2 = 1

/-- A function to check if a three-digit number has no repeating digits -/
def no_repeats (n : Nat) : Bool :=
  let d1 := n / 100
  let d2 := (n / 10) % 10
  let d3 := n % 10
  d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3

/-- The main theorem to be proved -/
theorem odd_three_digit_count : 
  (Finset.filter (λ n : Nat => 
    100 ≤ n ∧ n < 1000 ∧
    (n / 100) ∈ first_digit_set ∧
    ((n / 10) % 10) ∈ odd_digit_set ∧
    (n % 10) ∈ odd_digit_set ∧
    is_odd n ∧
    no_repeats n
  ) (Finset.range 1000)).card = 18 := by
  sorry

end odd_three_digit_count_l1493_149313


namespace cistern_emptying_rate_l1493_149323

-- Define the rates of the pipes
def rate_A : ℚ := 1 / 60
def rate_B : ℚ := 1 / 75
def rate_combined : ℚ := 1 / 50

-- Define the theorem
theorem cistern_emptying_rate :
  ∃ (rate_C : ℚ), 
    rate_A + rate_B - rate_C = rate_combined ∧ 
    rate_C = 1 / 100 := by
  sorry

end cistern_emptying_rate_l1493_149323


namespace line_tangent_to_curve_l1493_149372

/-- The line equation y = x + a is tangent to the curve y = x^3 - x^2 + 1 at the point (-1/3, 23/27) when a = 32/27 -/
theorem line_tangent_to_curve :
  let line (x : ℝ) := x + 32/27
  let curve (x : ℝ) := x^3 - x^2 + 1
  let tangent_point : ℝ × ℝ := (-1/3, 23/27)
  (∀ x, line x ≠ curve x ∨ x = tangent_point.1) ∧
  (line tangent_point.1 = curve tangent_point.1) ∧
  (HasDerivAt curve (line tangent_point.1) tangent_point.1) :=
by sorry


end line_tangent_to_curve_l1493_149372


namespace remainder_of_x_plus_one_power_2011_l1493_149334

theorem remainder_of_x_plus_one_power_2011 (x : ℤ) :
  (x + 1)^2011 ≡ x [ZMOD (x^2 - x + 1)] := by
  sorry

end remainder_of_x_plus_one_power_2011_l1493_149334


namespace arithmetic_sequence_before_one_l1493_149365

theorem arithmetic_sequence_before_one (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 100 → d = -7 → n = 15 →
  a₁ + (n - 1) * d = 1 ∧ n - 1 = 14 := by
  sorry

end arithmetic_sequence_before_one_l1493_149365


namespace distribute_5_3_l1493_149325

/-- The number of ways to distribute n identical objects into k identical containers,
    where at least one container must remain empty -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 26 ways to distribute 5 identical objects into 3 identical containers,
    where at least one container must remain empty -/
theorem distribute_5_3 : distribute 5 3 = 26 := by sorry

end distribute_5_3_l1493_149325


namespace invalid_set_l1493_149356

/-- A set of three positive real numbers representing the lengths of external diagonals of a right regular prism. -/
structure ExternalDiagonals where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0

/-- The condition for a valid set of external diagonals. -/
def is_valid (d : ExternalDiagonals) : Prop :=
  d.a^2 + d.b^2 > d.c^2 ∧ 
  d.b^2 + d.c^2 > d.a^2 ∧ 
  d.c^2 + d.a^2 > d.b^2

/-- The set {5,7,9} is not a valid set of external diagonals for a right regular prism. -/
theorem invalid_set : ¬ is_valid ⟨5, 7, 9, by norm_num, by norm_num, by norm_num⟩ := by
  sorry

end invalid_set_l1493_149356


namespace tan_ratio_inequality_l1493_149377

theorem tan_ratio_inequality (α β : Real) (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2) : 
  (Real.tan α) / α < (Real.tan β) / β := by
  sorry

end tan_ratio_inequality_l1493_149377


namespace iron_conducts_electricity_is_deductive_l1493_149318

-- Define the set of all substances
def Substance : Type := String

-- Define the property of conducting electricity
def conductsElectricity : Substance → Prop := sorry

-- Define the property of being a metal
def isMetal : Substance → Prop := sorry

-- Define iron as a substance
def iron : Substance := "iron"

-- Define the concept of deductive reasoning
def isDeductiveReasoning (premise1 premise2 conclusion : Prop) : Prop := sorry

-- Theorem statement
theorem iron_conducts_electricity_is_deductive :
  (∀ x, isMetal x → conductsElectricity x) →  -- All metals conduct electricity
  isMetal iron →                              -- Iron is a metal
  isDeductiveReasoning 
    (∀ x, isMetal x → conductsElectricity x)
    (isMetal iron)
    (conductsElectricity iron) :=
by
  sorry

end iron_conducts_electricity_is_deductive_l1493_149318


namespace solution_set_inequality_l1493_149382

theorem solution_set_inequality (x : ℝ) : 
  (x * (2 - x) > 0) ↔ (0 < x ∧ x < 2) :=
sorry

end solution_set_inequality_l1493_149382


namespace geometric_sequence_ratio_theorem_l1493_149394

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

def geometric_sum_reciprocals (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  (1 / a) * (1 - (1 / r)^n) / (1 - (1 / r))

theorem geometric_sequence_ratio_theorem :
  let a : ℚ := 1 / 4
  let r : ℚ := 2
  let n : ℕ := 10
  let S := geometric_sum a r n
  let S' := geometric_sum_reciprocals a r n
  S / S' = 32 := by sorry

end geometric_sequence_ratio_theorem_l1493_149394


namespace washing_machines_removed_per_box_l1493_149345

theorem washing_machines_removed_per_box :
  let num_crates : ℕ := 10
  let boxes_per_crate : ℕ := 6
  let initial_machines_per_box : ℕ := 4
  let total_machines_removed : ℕ := 60
  let total_boxes : ℕ := num_crates * boxes_per_crate
  let machines_removed_per_box : ℕ := total_machines_removed / total_boxes
  machines_removed_per_box = 1 := by
  sorry

end washing_machines_removed_per_box_l1493_149345


namespace solution_set_inequality_l1493_149367

/-- Given a function f: ℝ → ℝ satisfying certain conditions,
    prove that the solution set of f(x) + 1 > 2023 * exp(x) is (-∞, 0) -/
theorem solution_set_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x, (deriv f) x - f x < 1)
  (h2 : f 0 = 2022) :
  {x : ℝ | f x + 1 > 2023 * Real.exp x} = Set.Iio 0 := by
  sorry

end solution_set_inequality_l1493_149367


namespace model_shop_purchase_l1493_149305

theorem model_shop_purchase : ∃ (c t : ℕ), c > 0 ∧ t > 0 ∧ 5 * c + 8 * t = 31 ∧ c + t = 5 := by
  sorry

end model_shop_purchase_l1493_149305


namespace two_digit_sum_l1493_149358

theorem two_digit_sum (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → a ≠ 0 → a - b = a * b → 
  10 * a + b + (10 * b + a) = 22 := by
sorry

end two_digit_sum_l1493_149358


namespace basement_water_pumping_time_l1493_149337

/-- Proves that pumping out water from a flooded basement takes 450 minutes given specific conditions. -/
theorem basement_water_pumping_time : ∀ (length width depth : ℝ) 
  (num_pumps pump_rate : ℕ) (water_density : ℝ),
  length = 30 →
  width = 40 →
  depth = 24 / 12 →
  num_pumps = 4 →
  pump_rate = 10 →
  water_density = 7.5 →
  (length * width * depth * water_density) / (num_pumps * pump_rate) = 450 := by
  sorry

end basement_water_pumping_time_l1493_149337


namespace solve_for_y_l1493_149316

theorem solve_for_y (x y : ℤ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := by
  sorry

end solve_for_y_l1493_149316


namespace new_energy_vehicle_sales_growth_equation_l1493_149330

theorem new_energy_vehicle_sales_growth_equation 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (growth_period : ℕ) 
  (x : ℝ) 
  (h1 : initial_sales = 298) 
  (h2 : final_sales = 850) 
  (h3 : growth_period = 2) :
  initial_sales * (1 + x)^growth_period = final_sales :=
by sorry

end new_energy_vehicle_sales_growth_equation_l1493_149330


namespace height_range_l1493_149301

def heights : List ℕ := [153, 167, 148, 170, 154, 166, 149, 159, 167, 153]

theorem height_range :
  (List.maximum heights).map (λ max =>
    (List.minimum heights).map (λ min =>
      max - min
    )
  ) = some 22 := by
  sorry

end height_range_l1493_149301


namespace circle_equation_proof_l1493_149381

/-- Given two points P and Q in a 2D plane, we define a circle with PQ as its diameter. -/
def circle_with_diameter (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {point | (point.1 - (P.1 + Q.1) / 2)^2 + (point.2 - (P.2 + Q.2) / 2)^2 = ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / 4}

/-- The theorem states that for P(4,0) and Q(0,2), the equation of the circle with PQ as diameter is (x-2)^2 + (y-1)^2 = 5. -/
theorem circle_equation_proof :
  circle_with_diameter (4, 0) (0, 2) = {point : ℝ × ℝ | (point.1 - 2)^2 + (point.2 - 1)^2 = 5} :=
by sorry

end circle_equation_proof_l1493_149381


namespace sqrt_sum_inequality_l1493_149348

theorem sqrt_sum_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : a * b = c * d) (h2 : a + b > c + d) : 
  Real.sqrt a + Real.sqrt b > Real.sqrt c + Real.sqrt d :=
by sorry

end sqrt_sum_inequality_l1493_149348


namespace perimeter_area_ratio_bound_l1493_149362

/-- A shape in the plane formed by a union of finitely many unit squares -/
structure UnitSquareShape where
  squares : Finset (ℤ × ℤ)

/-- The perimeter of a UnitSquareShape -/
def perimeter (S : UnitSquareShape) : ℝ := sorry

/-- The area of a UnitSquareShape -/
def area (S : UnitSquareShape) : ℝ := S.squares.card

/-- The theorem stating that the ratio of perimeter to area is at most 8 -/
theorem perimeter_area_ratio_bound (S : UnitSquareShape) :
  perimeter S / area S ≤ 8 := by sorry

end perimeter_area_ratio_bound_l1493_149362


namespace password_probability_l1493_149347

-- Define the set of all possible symbols
def AllSymbols : Finset Char := {'!', '@', '#', '$', '%'}

-- Define the set of allowed symbols
def AllowedSymbols : Finset Char := {'!', '@', '#'}

-- Define the set of all possible single digits
def AllDigits : Finset Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define the set of even single digits
def EvenDigits : Finset Nat := {0, 2, 4, 6, 8}

-- Define the set of non-zero single digits
def NonZeroDigits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Theorem statement
theorem password_probability :
  (Finset.card EvenDigits : ℚ) / (Finset.card AllDigits) *
  (Finset.card AllowedSymbols : ℚ) / (Finset.card AllSymbols) *
  (Finset.card NonZeroDigits : ℚ) / (Finset.card AllDigits) =
  27 / 100 := by
sorry

end password_probability_l1493_149347


namespace total_trophies_is_950_l1493_149303

/-- The total number of trophies Jack and Michael will have after five years -/
def totalTrophies (michaelCurrent : ℕ) (michaelIncrease : ℕ) (jackMultiplier : ℕ) : ℕ :=
  (michaelCurrent + michaelIncrease) + (jackMultiplier * michaelCurrent)

/-- Proof that the total number of trophies is 950 -/
theorem total_trophies_is_950 :
  totalTrophies 50 150 15 = 950 := by
  sorry

end total_trophies_is_950_l1493_149303


namespace part_one_part_two_l1493_149392

-- Define the sets M and N
def M : Set ℝ := {x | (2*x - 2)/(x + 3) > 1}
def N (a : ℝ) : Set ℝ := {x | x^2 + (a - 8)*x - 8*a ≤ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ M
def q (a x : ℝ) : Prop := x ∈ N a

-- Part I: Relationship when a = -6
theorem part_one : 
  (∀ x, q (-6) x → p x) ∧ 
  (∃ x, p x ∧ ¬(q (-6) x)) := by sorry

-- Part II: Range of a where p is necessary but not sufficient for q
theorem part_two : 
  (∀ a, (∀ x, q a x → p x) ∧ (∃ x, p x ∧ ¬(q a x))) ↔ 
  a < -5 := by sorry

end part_one_part_two_l1493_149392


namespace vector_dot_product_problem_l1493_149319

theorem vector_dot_product_problem (a b : ℝ × ℝ) : 
  a = (0, 1) → b = (-1, 1) → (3 • a + 2 • b) • b = 7 := by sorry

end vector_dot_product_problem_l1493_149319


namespace right_triangle_perimeter_l1493_149322

theorem right_triangle_perimeter (area : ℝ) (leg : ℝ) (h1 : area = 180) (h2 : leg = 30) :
  ∃ (other_leg : ℝ) (hypotenuse : ℝ),
    area = (1 / 2) * leg * other_leg ∧
    hypotenuse^2 = leg^2 + other_leg^2 ∧
    leg + other_leg + hypotenuse = 42 + 2 * Real.sqrt 261 :=
by sorry

end right_triangle_perimeter_l1493_149322


namespace syrup_volume_proof_l1493_149335

def final_syrup_volume (original_volume : ℚ) (reduction_factor : ℚ) (sugar_added : ℚ) (cups_per_quart : ℚ) : ℚ :=
  original_volume * cups_per_quart * reduction_factor + sugar_added

theorem syrup_volume_proof (original_volume : ℚ) (reduction_factor : ℚ) (sugar_added : ℚ) (cups_per_quart : ℚ)
  (h1 : original_volume = 6)
  (h2 : reduction_factor = 1 / 12)
  (h3 : sugar_added = 1)
  (h4 : cups_per_quart = 4) :
  final_syrup_volume original_volume reduction_factor sugar_added cups_per_quart = 3 := by
  sorry

end syrup_volume_proof_l1493_149335


namespace quadratic_maximum_l1493_149312

/-- The quadratic function we're analyzing -/
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 16

/-- The point where the maximum occurs -/
def x_max : ℝ := 2

/-- The maximum value of the function -/
def y_max : ℝ := 24

theorem quadratic_maximum :
  (∀ x : ℝ, f x ≤ y_max) ∧ f x_max = y_max :=
sorry

end quadratic_maximum_l1493_149312


namespace number_plus_four_equals_six_l1493_149393

theorem number_plus_four_equals_six (x : ℤ) : x + 4 = 6 → x = 2 := by
  sorry

end number_plus_four_equals_six_l1493_149393


namespace circle_radius_is_five_l1493_149359

/-- A square with side length 10 -/
structure Square :=
  (side : ℝ)
  (is_ten : side = 10)

/-- A circle passing through two opposite vertices of the square and tangent to one side -/
structure Circle (s : Square) :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (passes_through_vertices : True)  -- This is a placeholder for the actual condition
  (tangent_to_side : True)  -- This is a placeholder for the actual condition

/-- The theorem stating that the radius of the circle is 5 -/
theorem circle_radius_is_five (s : Square) (c : Circle s) : c.radius = 5 := by
  sorry

end circle_radius_is_five_l1493_149359


namespace probability_of_identical_cubes_l1493_149342

/-- Represents the three possible colors for painting cube faces -/
inductive Color
  | Red
  | Blue
  | Green

/-- Represents a cube with six colored faces -/
def Cube := Fin 6 → Color

/-- The number of ways to paint a single cube -/
def waysToColorCube : ℕ := 729

/-- The total number of ways to paint three cubes -/
def totalWaysToPaintThreeCubes : ℕ := 387420489

/-- The number of ways to paint three cubes so they are rotationally identical -/
def waysToColorIdenticalCubes : ℕ := 633

/-- Checks if two cubes are rotationally identical -/
def areRotationallyIdentical (c1 c2 : Cube) : Prop := sorry

/-- The probability of three independently painted cubes being rotationally identical -/
theorem probability_of_identical_cubes :
  (waysToColorIdenticalCubes : ℚ) / totalWaysToPaintThreeCubes = 211 / 129140163 := by
  sorry

end probability_of_identical_cubes_l1493_149342


namespace quadratic_equations_one_common_root_l1493_149344

theorem quadratic_equations_one_common_root 
  (a b c d : ℝ) : 
  (∃! x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + c*x + d = 0) ↔ 
  ((a*d - b*c)*(c - a) = (b - d)^2 ∧ (a*d - b*c)*(c - a) ≠ 0) :=
by sorry

end quadratic_equations_one_common_root_l1493_149344


namespace total_breakfast_cost_l1493_149307

def breakfast_cost (muffin_price fruit_cup_price francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ) : ℕ :=
  (francis_muffins * muffin_price + francis_fruit_cups * fruit_cup_price) +
  (kiera_muffins * muffin_price + kiera_fruit_cups * fruit_cup_price)

theorem total_breakfast_cost :
  breakfast_cost 2 3 2 2 2 1 = 17 :=
by sorry

end total_breakfast_cost_l1493_149307


namespace inscribed_equiangular_triangle_exists_l1493_149311

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the concept of being inscribed in a circle
def isInscribed (t : Triangle) (c : Circle) : Prop :=
  sorry

-- Define the concept of two triangles being equiangular
def isEquiangular (t1 t2 : Triangle) : Prop :=
  sorry

theorem inscribed_equiangular_triangle_exists 
  (c : Circle) (reference : Triangle) : 
  ∃ (t : Triangle), isInscribed t c ∧ isEquiangular t reference := by
  sorry

end inscribed_equiangular_triangle_exists_l1493_149311


namespace daycare_count_l1493_149386

/-- The real number of toddlers in the daycare -/
def real_count (bill_count playground_count new_count double_counted missed : ℕ) : ℕ :=
  bill_count - double_counted + missed - playground_count + new_count

/-- Theorem stating the real number of toddlers given the conditions -/
theorem daycare_count : real_count 28 6 4 9 5 = 22 := by
  sorry

end daycare_count_l1493_149386


namespace gcf_factorial_seven_eight_l1493_149389

theorem gcf_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcf_factorial_seven_eight_l1493_149389


namespace problem_solution_l1493_149383

theorem problem_solution (x : ℝ) (h1 : x > 0) (h2 : x * ↑(⌊x⌋) = 72) : x = 9 := by
  sorry

end problem_solution_l1493_149383


namespace divisor_property_l1493_149396

theorem divisor_property (k : ℕ) (h : 5^k - k^5 = 1) : 15^k = 1 := by
  sorry

end divisor_property_l1493_149396


namespace volunteer_arrangement_count_l1493_149378

/-- The number of ways to arrange volunteers among events --/
def arrangeVolunteers (n : ℕ) (k : ℕ) : ℕ := k^n

/-- The number of ways to arrange volunteers among events, excluding one event --/
def arrangeVolunteersExcludeOne (n : ℕ) (k : ℕ) : ℕ := k * (k-1)^n

/-- The number of ways to arrange volunteers to only one event --/
def arrangeVolunteersToOne (n : ℕ) (k : ℕ) : ℕ := k

theorem volunteer_arrangement_count :
  let n : ℕ := 5  -- number of volunteers
  let k : ℕ := 3  -- number of events
  arrangeVolunteers n k - k * arrangeVolunteersExcludeOne n (k-1) + arrangeVolunteersToOne n k = 150 :=
by sorry

end volunteer_arrangement_count_l1493_149378


namespace rajesh_savings_amount_l1493_149360

/-- Calculates Rajesh's monthly savings based on his salary and spending habits -/
def rajesh_savings (monthly_salary : ℕ) : ℕ :=
  let food_expense := (40 * monthly_salary) / 100
  let medicine_expense := (20 * monthly_salary) / 100
  let remaining := monthly_salary - (food_expense + medicine_expense)
  (60 * remaining) / 100

/-- Theorem stating that Rajesh's monthly savings are 3600 given his salary and spending habits -/
theorem rajesh_savings_amount :
  rajesh_savings 15000 = 3600 := by
  sorry

end rajesh_savings_amount_l1493_149360


namespace apple_bag_price_apple_bag_price_is_8_l1493_149353

/-- Calculates the selling price of one bag of apples given the harvest and sales information. -/
theorem apple_bag_price (total_harvest : ℕ) (juice_amount : ℕ) (restaurant_amount : ℕ) 
  (bag_size : ℕ) (total_revenue : ℕ) : ℕ :=
  let remaining := total_harvest - juice_amount - restaurant_amount
  let num_bags := remaining / bag_size
  total_revenue / num_bags

/-- Proves that the selling price of one bag of apples is $8 given the specific harvest and sales information. -/
theorem apple_bag_price_is_8 :
  apple_bag_price 405 90 60 5 408 = 8 := by
  sorry

end apple_bag_price_apple_bag_price_is_8_l1493_149353


namespace fixed_points_equality_implies_a_bound_l1493_149300

/-- Given a function f(x) = x^2 - 2x + a, if the set of fixed points of f is equal to the set of fixed points of f ∘ f, then a is greater than or equal to 5/4. -/
theorem fixed_points_equality_implies_a_bound (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + a
  ({x : ℝ | f x = x} = {x : ℝ | f (f x) = x}) →
  a ≥ 5/4 := by
sorry

end fixed_points_equality_implies_a_bound_l1493_149300


namespace min_value_sum_l1493_149317

theorem min_value_sum (x₁ x₂ x₃ x₄ : ℝ) 
  (h_pos : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0) 
  (h_sum : x₁^2 + x₂^2 + x₃^2 + x₄^2 = 4) : 
  x₁ / (1 - x₁^2) + x₂ / (1 - x₂^2) + x₃ / (1 - x₃^2) + x₄ / (1 - x₄^2) ≥ 6 * Real.sqrt 3 := by
  sorry

end min_value_sum_l1493_149317


namespace circle_equation_holds_l1493_149379

/-- A circle in the Cartesian plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in general form --/
def CircleEquation (x y : ℝ) := x^2 + y^2 - 6*x = 0

/-- The circle represented by the equation x^2 + y^2 - 6x = 0 --/
def specificCircle : Circle := { center := (3, 0), radius := 3 }

/-- Theorem stating that the specificCircle satisfies the given equation --/
theorem circle_equation_holds (x y : ℝ) :
  CircleEquation x y ↔ (x - specificCircle.center.1)^2 + (y - specificCircle.center.2)^2 = specificCircle.radius^2 := by
  sorry

#check circle_equation_holds

end circle_equation_holds_l1493_149379


namespace distinct_collections_count_l1493_149366

/-- Represents the collection of letters in MATHEMATICAL -/
def mathematical : Finset Char := {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'A', 'L'}

/-- Represents the vowels in MATHEMATICAL -/
def vowels : Finset Char := {'A', 'E', 'I'}

/-- Represents the consonants in MATHEMATICAL -/
def consonants : Finset Char := {'M', 'T', 'H', 'C', 'L'}

/-- Function to count occurrences of a character in MATHEMATICAL -/
def count (c : Char) : Nat := (mathematical.filter (· = c)).card

/-- The number of distinct possible collections of letters in the bag -/
def distinct_collections : Nat := sorry

theorem distinct_collections_count : distinct_collections = 220 := by
  sorry

end distinct_collections_count_l1493_149366


namespace vertex_angle_is_40_l1493_149364

-- Define an isosceles triangle
structure IsoscelesTriangle where
  vertexAngle : ℝ
  baseAngle : ℝ
  sum_of_angles : vertexAngle + 2 * baseAngle = 180
  base_angle_relation : baseAngle = vertexAngle + 30

-- Theorem statement
theorem vertex_angle_is_40 (t : IsoscelesTriangle) : t.vertexAngle = 40 :=
by sorry

end vertex_angle_is_40_l1493_149364


namespace horner_v2_value_horner_method_correctness_l1493_149341

/-- Horner's method intermediate value -/
def v2 (x : ℝ) : ℝ := 2 * x - 3 * x + 1

/-- The polynomial function -/
def f (x : ℝ) : ℝ := 1 + 2*x + x^2 - 3*x^3 + 2*x^4

theorem horner_v2_value :
  v2 (-1) = 6 :=
by sorry

theorem horner_method_correctness (x : ℝ) :
  f x = (((2*x - 3)*x + 1)*x + 2)*x + 1 :=
by sorry

#check horner_v2_value
#check horner_method_correctness

end horner_v2_value_horner_method_correctness_l1493_149341


namespace functional_equation_solution_l1493_149369

theorem functional_equation_solution (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f (x + y) = f x + f y) : 
  ∃ a : ℤ, ∀ x : ℤ, f x = a * x := by
  sorry

end functional_equation_solution_l1493_149369


namespace garden_area_l1493_149395

/-- Represents a rectangular garden with specific properties -/
structure Garden where
  width : ℝ
  length : ℝ
  perimeter_minus_one_side : ℝ

/-- The properties of the garden as described in the problem -/
def garden_properties (g : Garden) : Prop :=
  g.perimeter_minus_one_side = 60 ∧
  g.length = 2 * g.width

/-- The theorem stating that a garden with the given properties has an area of 450 square meters -/
theorem garden_area (g : Garden) (h : garden_properties g) : g.width * g.length = 450 := by
  sorry


end garden_area_l1493_149395


namespace arccos_of_neg_one_eq_pi_l1493_149384

theorem arccos_of_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_of_neg_one_eq_pi_l1493_149384


namespace equal_roots_quadratic_l1493_149354

theorem equal_roots_quadratic (k C : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + 4 * k * x + C = 0) →
  (∃! r : ℝ, 2 * x^2 + 4 * x + C = 0) →
  C = 2 :=
by sorry

end equal_roots_quadratic_l1493_149354


namespace probability_12th_roll_last_proof_l1493_149388

/-- The probability of the 12th roll being the last roll when rolling a standard 
    eight-sided die until getting the same number on consecutive rolls -/
def probability_12th_roll_last : ℚ :=
  (7^10 : ℚ) / (8^11 : ℚ)

/-- The number of sides on the standard die -/
def num_sides : ℕ := 8

/-- The number of rolls -/
def num_rolls : ℕ := 12

theorem probability_12th_roll_last_proof :
  probability_12th_roll_last = (7^(num_rolls - 2) : ℚ) / (num_sides^(num_rolls - 1) : ℚ) :=
by sorry

end probability_12th_roll_last_proof_l1493_149388


namespace bill_due_time_l1493_149370

/-- Proves that given a bill with specified face value, true discount, and annual interest rate,
    the time until the bill is due is 9 months. -/
theorem bill_due_time (face_value : ℝ) (true_discount : ℝ) (annual_interest_rate : ℝ) :
  face_value = 2240 →
  true_discount = 240 →
  annual_interest_rate = 0.16 →
  (face_value / (face_value - true_discount) - 1) / annual_interest_rate * 12 = 9 := by
  sorry

end bill_due_time_l1493_149370


namespace average_cat_weight_l1493_149310

def cat_weights : List ℝ := [12, 12, 14.7, 9.3]

theorem average_cat_weight :
  (cat_weights.sum / cat_weights.length) = 12 := by
  sorry

end average_cat_weight_l1493_149310


namespace ackermann_2_1_l1493_149349

def A : ℕ → ℕ → ℕ
  | 0, n => n + 1
  | m + 1, 0 => A m 1
  | m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_2_1 : A 2 1 = 5 := by
  sorry

end ackermann_2_1_l1493_149349


namespace positive_intervals_l1493_149387

-- Define the expression
def f (x : ℝ) : ℝ := (x - 2) * (x + 3)

-- Theorem statement
theorem positive_intervals (x : ℝ) : 
  f x > 0 ↔ x < -3 ∨ x > 2 := by
  sorry

end positive_intervals_l1493_149387


namespace rectangle_dimension_solution_l1493_149332

theorem rectangle_dimension_solution (x : ℝ) : 
  (3*x - 5 > 0) → (x + 7 > 0) → ((3*x - 5) * (x + 7) = 14*x - 35) → x = 0 :=
by sorry

end rectangle_dimension_solution_l1493_149332


namespace david_biology_marks_l1493_149314

/-- Calculates David's marks in Biology given his marks in other subjects and his average -/
def davidsBiologyMarks (english : ℕ) (mathematics : ℕ) (physics : ℕ) (chemistry : ℕ) (average : ℕ) : ℕ :=
  5 * average - (english + mathematics + physics + chemistry)

theorem david_biology_marks :
  davidsBiologyMarks 51 65 82 67 70 = 85 := by
  sorry

end david_biology_marks_l1493_149314


namespace equality_condition_for_squared_sum_equals_product_sum_l1493_149398

theorem equality_condition_for_squared_sum_equals_product_sum (a b c : ℝ) :
  (a^2 + b^2 + c^2 = a*b + b*c + c*a) ↔ (a = b ∧ b = c) :=
sorry

end equality_condition_for_squared_sum_equals_product_sum_l1493_149398


namespace power_of_two_plus_five_l1493_149333

theorem power_of_two_plus_five : 2^5 + 5 = 37 := by
  sorry

end power_of_two_plus_five_l1493_149333


namespace tims_books_l1493_149374

theorem tims_books (sandy_books : ℕ) (benny_books : ℕ) (total_books : ℕ) :
  sandy_books = 10 →
  benny_books = 24 →
  total_books = 67 →
  ∃ tim_books : ℕ, tim_books = total_books - (sandy_books + benny_books) ∧ tim_books = 33 :=
by sorry

end tims_books_l1493_149374


namespace david_presents_l1493_149376

theorem david_presents (christmas : ℕ) (easter : ℕ) (birthday : ℕ) 
  (h1 : christmas = 60)
  (h2 : birthday = 3 * easter)
  (h3 : easter = christmas / 2 - 10) : 
  christmas + easter + birthday = 140 := by
  sorry

end david_presents_l1493_149376


namespace triangle_side_length_l1493_149327

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : Real) : 
  (1/2) * a * c * Real.sin B = Real.sqrt 3 →  -- Area condition
  B = Real.pi / 3 →  -- 60° in radians
  a^2 + c^2 = 3 * a * c →  -- Given equation
  b = 2 * Real.sqrt 2 := by
sorry

end triangle_side_length_l1493_149327


namespace evaluate_expression_l1493_149329

theorem evaluate_expression (x y : ℝ) (hx : x = 2) (hy : y = 3) :
  y * (y - 5 * x + 2) = -15 := by
  sorry

end evaluate_expression_l1493_149329


namespace binomial_coefficient_two_l1493_149373

theorem binomial_coefficient_two (n : ℕ+) : (n.val.choose 2) = n.val * (n.val - 1) / 2 := by
  sorry

end binomial_coefficient_two_l1493_149373


namespace tundra_electrification_l1493_149324

theorem tundra_electrification (x y : ℝ) : 
  x + y = 1 →                 -- Initial parts sum to 1
  2*x + 0.75*y = 1 →          -- Condition after changes
  0 ≤ x ∧ x ≤ 1 →             -- x is a fraction
  0 ≤ y ∧ y ≤ 1 →             -- y is a fraction
  y = 4/5 :=                  -- Conclusion: non-electrified part was 4/5
by sorry

end tundra_electrification_l1493_149324


namespace deceased_member_income_l1493_149351

/-- Proves that given a family of 3 earning members with an average monthly income of Rs. 735,
    if one member dies and the new average income becomes Rs. 650,
    then the income of the deceased member was Rs. 905. -/
theorem deceased_member_income
  (total_income : ℕ)
  (remaining_income : ℕ)
  (h1 : total_income / 3 = 735)
  (h2 : remaining_income / 2 = 650)
  (h3 : total_income > remaining_income) :
  total_income - remaining_income = 905 := by
sorry

end deceased_member_income_l1493_149351


namespace min_value_cubic_expression_l1493_149336

theorem min_value_cubic_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^3 + y^3 - 5*x*y ≥ -125/27 ∧ ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x^3 + y^3 - 5*x*y = -125/27 := by
  sorry

end min_value_cubic_expression_l1493_149336
