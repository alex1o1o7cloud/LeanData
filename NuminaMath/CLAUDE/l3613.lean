import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l3613_361345

def complex_number (a : ℝ) : ℂ := (1 - a * Complex.I) * (a + 2 * Complex.I)

def in_first_quadrant (z : ℂ) : Prop := Complex.re z > 0 ∧ Complex.im z > 0

theorem range_of_a (a : ℝ) :
  in_first_quadrant (complex_number a) → 0 < a ∧ a < Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3613_361345


namespace NUMINAMATH_CALUDE_rosemary_leaves_count_rosemary_leaves_solution_l3613_361308

theorem rosemary_leaves_count : ℕ → Prop :=
  fun r : ℕ =>
    let basil_pots : ℕ := 3
    let rosemary_pots : ℕ := 9
    let thyme_pots : ℕ := 6
    let basil_leaves_per_plant : ℕ := 4
    let thyme_leaves_per_plant : ℕ := 30
    let total_leaves : ℕ := 354
    
    basil_pots * basil_leaves_per_plant + 
    rosemary_pots * r + 
    thyme_pots * thyme_leaves_per_plant = total_leaves →
    r = 18

theorem rosemary_leaves_solution : rosemary_leaves_count 18 := by
  sorry

end NUMINAMATH_CALUDE_rosemary_leaves_count_rosemary_leaves_solution_l3613_361308


namespace NUMINAMATH_CALUDE_polynomial_sum_l3613_361316

theorem polynomial_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 11)
  (h3 : a * x^3 + b * y^3 = 24)
  (h4 : a * x^4 + b * y^4 = 58) :
  a * x^5 + b * y^5 = 262.88 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_l3613_361316


namespace NUMINAMATH_CALUDE_exists_unique_N_l3613_361365

theorem exists_unique_N : ∃ N : ℤ, N = 1719 ∧
  ∀ a b : ℤ, (N / 2 - a = b - N / 2) →
    ((∃ m n : ℕ+, a = 19 * m + 85 * n) ∨ (∃ m n : ℕ+, b = 19 * m + 85 * n)) ∧
    ¬((∃ m n : ℕ+, a = 19 * m + 85 * n) ∧ (∃ m n : ℕ+, b = 19 * m + 85 * n)) :=
by sorry

end NUMINAMATH_CALUDE_exists_unique_N_l3613_361365


namespace NUMINAMATH_CALUDE_two_parts_of_ten_l3613_361336

theorem two_parts_of_ten (x y : ℝ) : 
  x + y = 10 ∧ |x - y| = 5 → 
  (x = 7.5 ∧ y = 2.5) ∨ (x = 2.5 ∧ y = 7.5) := by
sorry

end NUMINAMATH_CALUDE_two_parts_of_ten_l3613_361336


namespace NUMINAMATH_CALUDE_sum_of_cubes_equals_ten_squared_l3613_361354

theorem sum_of_cubes_equals_ten_squared (h1 : 1 + 2 + 3 + 4 = 10) 
  (h2 : ∃ n : ℕ, 1^3 + 2^3 + 3^3 + 4^3 = 10^n) : 
  ∃ n : ℕ, 1^3 + 2^3 + 3^3 + 4^3 = 10^n ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_equals_ten_squared_l3613_361354


namespace NUMINAMATH_CALUDE_john_biking_distance_john_biking_distance_proof_l3613_361363

theorem john_biking_distance (bike_speed walking_speed : ℝ) 
  (walking_distance total_time : ℝ) : ℝ :=
  let total_biking_distance := 
    (total_time - walking_distance / walking_speed) * bike_speed + walking_distance
  total_biking_distance

#check john_biking_distance 15 4 3 (7/6) = 9.25

theorem john_biking_distance_proof :
  john_biking_distance 15 4 3 (7/6) = 9.25 := by
  sorry

end NUMINAMATH_CALUDE_john_biking_distance_john_biking_distance_proof_l3613_361363


namespace NUMINAMATH_CALUDE_green_balls_count_l3613_361350

/-- The number of green balls in a bag with specific conditions -/
def num_green_balls (total : ℕ) (white : ℕ) (yellow : ℕ) (red : ℕ) (purple : ℕ) 
    (prob_not_red_purple : ℚ) : ℕ :=
  total - (white + yellow + red + purple)

theorem green_balls_count :
  let total := 60
  let white := 22
  let yellow := 2
  let red := 15
  let purple := 3
  let prob_not_red_purple := 7/10
  num_green_balls total white yellow red purple prob_not_red_purple = 18 := by
  sorry

end NUMINAMATH_CALUDE_green_balls_count_l3613_361350


namespace NUMINAMATH_CALUDE_system_solution_l3613_361346

theorem system_solution : ∃! (x y z : ℝ),
  (3 * x - 2 * y + z = 7) ∧
  (9 * y - 6 * x - 3 * z = -21) ∧
  (x + y + z = 5) ∧
  (x = 1 ∧ y = 0 ∧ z = 4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3613_361346


namespace NUMINAMATH_CALUDE_sin_72_cos_18_plus_cos_72_sin_18_l3613_361340

theorem sin_72_cos_18_plus_cos_72_sin_18 : 
  Real.sin (72 * π / 180) * Real.cos (18 * π / 180) + 
  Real.cos (72 * π / 180) * Real.sin (18 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_72_cos_18_plus_cos_72_sin_18_l3613_361340


namespace NUMINAMATH_CALUDE_max_value_abcd_l3613_361351

theorem max_value_abcd (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a * b * c * d * (a + b + c + d)) / ((a + b)^2 * (c + d)^2) ≤ (1 : ℝ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abcd_l3613_361351


namespace NUMINAMATH_CALUDE_soda_filling_time_difference_l3613_361377

/-- Proves that the additional time needed to fill 12 barrels with a leak is 24 minutes -/
theorem soda_filling_time_difference 
  (normal_time : ℕ) 
  (leak_time : ℕ) 
  (barrel_count : ℕ) 
  (h1 : normal_time = 3)
  (h2 : leak_time = 5)
  (h3 : barrel_count = 12) :
  leak_time * barrel_count - normal_time * barrel_count = 24 := by
  sorry

end NUMINAMATH_CALUDE_soda_filling_time_difference_l3613_361377


namespace NUMINAMATH_CALUDE_compound_has_one_hydrogen_l3613_361384

/-- Represents the number of atoms of each element in the compound -/
structure Compound where
  hydrogen : ℕ
  bromine : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (h_weight o_weight br_weight : ℚ) : ℚ :=
  c.hydrogen * h_weight + c.bromine * br_weight + c.oxygen * o_weight

/-- The theorem stating that a compound with 1 Br, 3 O, and molecular weight 129 has 1 H atom -/
theorem compound_has_one_hydrogen :
  ∃ (c : Compound),
    c.bromine = 1 ∧
    c.oxygen = 3 ∧
    molecularWeight c 1 16 79.9 = 129 ∧
    c.hydrogen = 1 := by
  sorry


end NUMINAMATH_CALUDE_compound_has_one_hydrogen_l3613_361384


namespace NUMINAMATH_CALUDE_third_stack_difference_l3613_361385

/-- Represents the heights of five stacks of blocks -/
structure BlockStacks where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- The properties of the block stacks as described in the problem -/
def validBlockStacks (s : BlockStacks) : Prop :=
  s.first = 7 ∧
  s.second = s.first + 3 ∧
  s.third < s.second ∧
  s.fourth = s.third + 10 ∧
  s.fifth = 2 * s.second ∧
  s.first + s.second + s.third + s.fourth + s.fifth = 55

theorem third_stack_difference (s : BlockStacks) 
  (h : validBlockStacks s) : s.second - s.third = 1 := by
  sorry

end NUMINAMATH_CALUDE_third_stack_difference_l3613_361385


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3613_361378

theorem quadratic_function_properties (a b c : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ioo 1 3 ↔ a * x^2 + b * x + c > -2 * x) →
  (a < 0 ∧
   b = -4 * a - 2 ∧
   (∀ x : ℝ, (a * x^2 + b * x + c + 6 * a = 0 → 
    ∃! r : ℝ, a * r^2 + b * r + c + 6 * a = 0) → a = -1/5)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3613_361378


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3613_361371

theorem complex_magnitude_problem (z w : ℂ) 
  (h1 : Complex.abs (3 * z - w) = 15)
  (h2 : Complex.abs (z + 3 * w) = 9)
  (h3 : Complex.abs (z + w) = 6) :
  Complex.abs z = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3613_361371


namespace NUMINAMATH_CALUDE_sharons_harvest_l3613_361307

theorem sharons_harvest (greg_harvest : ℝ) (difference : ℝ) (sharon_harvest : ℝ) 
  (h1 : greg_harvest = 0.4)
  (h2 : greg_harvest = sharon_harvest + difference)
  (h3 : difference = 0.3) :
  sharon_harvest = 0.1 := by
sorry

end NUMINAMATH_CALUDE_sharons_harvest_l3613_361307


namespace NUMINAMATH_CALUDE_factorize_nine_minus_a_squared_l3613_361323

theorem factorize_nine_minus_a_squared (a : ℝ) : 9 - a^2 = (3 + a) * (3 - a) := by
  sorry

end NUMINAMATH_CALUDE_factorize_nine_minus_a_squared_l3613_361323


namespace NUMINAMATH_CALUDE_michael_crates_thursday_l3613_361369

/-- The number of crates Michael bought on Thursday -/
def crates_bought_thursday (initial_crates : ℕ) (crates_given : ℕ) (eggs_per_crate : ℕ) (final_eggs : ℕ) : ℕ :=
  (final_eggs - (initial_crates - crates_given) * eggs_per_crate) / eggs_per_crate

theorem michael_crates_thursday :
  crates_bought_thursday 6 2 30 270 = 5 := by
  sorry

end NUMINAMATH_CALUDE_michael_crates_thursday_l3613_361369


namespace NUMINAMATH_CALUDE_problem_solution_l3613_361303

def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0

def q (x m : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

theorem problem_solution (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, m = 4 → (p x ∧ q x m) → 4 < x ∧ x < 5) ∧
  ((∀ x : ℝ, ¬(q x m) → ¬(p x)) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m) → 5/3 ≤ m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3613_361303


namespace NUMINAMATH_CALUDE_team_average_weight_l3613_361379

theorem team_average_weight 
  (num_forwards : ℕ) 
  (num_defensemen : ℕ) 
  (avg_weight_forwards : ℝ) 
  (avg_weight_defensemen : ℝ) 
  (h1 : num_forwards = 8)
  (h2 : num_defensemen = 12)
  (h3 : avg_weight_forwards = 75)
  (h4 : avg_weight_defensemen = 82) :
  let total_players := num_forwards + num_defensemen
  let total_weight := num_forwards * avg_weight_forwards + num_defensemen * avg_weight_defensemen
  total_weight / total_players = 79.2 := by
  sorry

end NUMINAMATH_CALUDE_team_average_weight_l3613_361379


namespace NUMINAMATH_CALUDE_line_properties_l3613_361343

/-- The slope of the line sqrt(3)x - y - 1 = 0 is sqrt(3) and its inclination angle is 60° --/
theorem line_properties :
  let line := fun (x y : ℝ) => Real.sqrt 3 * x - y - 1 = 0
  ∃ (m θ : ℝ),
    (∀ x y, line x y → y = m * x - 1) ∧ 
    m = Real.sqrt 3 ∧
    θ = 60 * π / 180 ∧
    Real.tan θ = m :=
by sorry

end NUMINAMATH_CALUDE_line_properties_l3613_361343


namespace NUMINAMATH_CALUDE_euro_calculation_l3613_361324

-- Define the € operation
def euro (x y : ℕ) : ℕ := 2 * x * y

-- State the theorem
theorem euro_calculation : euro 7 (euro 4 5) = 560 := by
  sorry

end NUMINAMATH_CALUDE_euro_calculation_l3613_361324


namespace NUMINAMATH_CALUDE_roses_distribution_l3613_361341

def total_money : ℕ := 300
def jenna_price : ℕ := 2
def imma_price : ℕ := 3
def ravi_price : ℕ := 4
def leila_price : ℕ := 5

def jenna_budget : ℕ := 100
def imma_budget : ℕ := 100
def ravi_budget : ℕ := 50
def leila_budget : ℕ := 50

def jenna_fraction : ℚ := 1/3
def imma_fraction : ℚ := 1/4
def ravi_fraction : ℚ := 1/6

theorem roses_distribution (jenna_roses imma_roses ravi_roses leila_roses : ℕ) :
  jenna_roses = ⌊(jenna_fraction * (jenna_budget / jenna_price : ℚ))⌋ ∧
  imma_roses = ⌊(imma_fraction * (imma_budget / imma_price : ℚ))⌋ ∧
  ravi_roses = ⌊(ravi_fraction * (ravi_budget / ravi_price : ℚ))⌋ ∧
  leila_roses = leila_budget / leila_price →
  jenna_roses + imma_roses + ravi_roses + leila_roses = 36 := by
  sorry

end NUMINAMATH_CALUDE_roses_distribution_l3613_361341


namespace NUMINAMATH_CALUDE_orchid_rose_difference_l3613_361355

/-- Given the initial and final counts of roses and orchids in a vase,
    prove that there are 10 more orchids than roses after adding new flowers. -/
theorem orchid_rose_difference (initial_roses initial_orchids final_roses final_orchids : ℕ) : 
  initial_roses = 9 →
  initial_orchids = 6 →
  final_roses = 3 →
  final_orchids = 13 →
  final_orchids - final_roses = 10 := by
  sorry

end NUMINAMATH_CALUDE_orchid_rose_difference_l3613_361355


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3613_361367

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x - 1 ≤ 0) ↔ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3613_361367


namespace NUMINAMATH_CALUDE_value_of_M_l3613_361381

theorem value_of_M : ∃ M : ℝ, (0.25 * M = 0.35 * 1200) ∧ (M = 1680) := by
  sorry

end NUMINAMATH_CALUDE_value_of_M_l3613_361381


namespace NUMINAMATH_CALUDE_reflection_over_y_eq_neg_x_l3613_361327

/-- Reflects a point (x, y) over the line y = -x -/
def reflect_over_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.2), -(p.1))

/-- The original point -/
def original_point : ℝ × ℝ := (7, -3)

/-- The expected reflected point -/
def expected_reflected_point : ℝ × ℝ := (3, -7)

theorem reflection_over_y_eq_neg_x :
  reflect_over_y_eq_neg_x original_point = expected_reflected_point := by
  sorry

end NUMINAMATH_CALUDE_reflection_over_y_eq_neg_x_l3613_361327


namespace NUMINAMATH_CALUDE_derivative_sin_minus_exp_two_l3613_361356

theorem derivative_sin_minus_exp_two (x : ℝ) :
  deriv (fun x => Real.sin x - (2 : ℝ)^x) x = Real.cos x - (2 : ℝ)^x * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sin_minus_exp_two_l3613_361356


namespace NUMINAMATH_CALUDE_fixed_fee_is_7_42_l3613_361310

/-- Represents the monthly bill structure for an online service provider -/
structure Bill where
  fixed_fee : ℝ
  connect_time_charge : ℝ
  data_usage_charge_per_gb : ℝ

/-- The December bill without data usage -/
def december_bill (b : Bill) : ℝ :=
  b.fixed_fee + b.connect_time_charge

/-- The January bill with 3 GB data usage -/
def january_bill (b : Bill) : ℝ :=
  b.fixed_fee + b.connect_time_charge + 3 * b.data_usage_charge_per_gb

/-- Theorem stating that the fixed monthly fee is $7.42 -/
theorem fixed_fee_is_7_42 (b : Bill) : b.fixed_fee = 7.42 :=
  by
  have h1 : december_bill b = 18.50 := by sorry
  have h2 : january_bill b = 23.45 := by sorry
  have h3 : january_bill b - december_bill b = 3 * b.data_usage_charge_per_gb := by sorry
  sorry

end NUMINAMATH_CALUDE_fixed_fee_is_7_42_l3613_361310


namespace NUMINAMATH_CALUDE_gabby_fruit_count_l3613_361302

def watermelons : ℕ := 1

def peaches (w : ℕ) : ℕ := w + 12

def plums (p : ℕ) : ℕ := 3 * p

def total_fruits (w p l : ℕ) : ℕ := w + p + l

theorem gabby_fruit_count :
  total_fruits watermelons (peaches watermelons) (plums (peaches watermelons)) = 53 := by
  sorry

end NUMINAMATH_CALUDE_gabby_fruit_count_l3613_361302


namespace NUMINAMATH_CALUDE_new_range_after_percent_increase_l3613_361332

/-- Given a set of investments with a range of annual yields R last year,
    if each investment's yield increases by p percent this year,
    then the new range of annual yields is (1 + p/100) * R. -/
theorem new_range_after_percent_increase 
  (R : ℝ) -- Range of annual yields last year
  (p : ℝ) -- Percentage increase in yields
  (h : R > 0) -- Assumption that the original range is positive
  : ∃ (L H : ℝ), -- There exist lowest and highest yields L and H
    H - L = R ∧ -- Such that the original range is R
    (H * (1 + p / 100) - L * (1 + p / 100) = R * (1 + p / 100)) -- And the new range is R * (1 + p/100)
  := by sorry

end NUMINAMATH_CALUDE_new_range_after_percent_increase_l3613_361332


namespace NUMINAMATH_CALUDE_inequality_preservation_l3613_361305

theorem inequality_preservation (x y : ℝ) (h : x > y) : x + 5 > y + 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3613_361305


namespace NUMINAMATH_CALUDE_percentage_calculation_l3613_361359

theorem percentage_calculation (P : ℝ) : 
  (P / 100) * 1265 / 5.96 = 377.8020134228188 → P = 178 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3613_361359


namespace NUMINAMATH_CALUDE_math_class_students_count_l3613_361312

theorem math_class_students_count :
  ∃! n : ℕ, n < 50 ∧ n % 8 = 5 ∧ n % 6 = 3 ∧ n = 45 :=
by sorry

end NUMINAMATH_CALUDE_math_class_students_count_l3613_361312


namespace NUMINAMATH_CALUDE_remainder_is_perfect_square_l3613_361391

theorem remainder_is_perfect_square (n : ℕ+) : ∃ k : ℤ, (10^n.val - 1) % 37 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_is_perfect_square_l3613_361391


namespace NUMINAMATH_CALUDE_spliced_wire_length_l3613_361398

theorem spliced_wire_length 
  (num_pieces : ℕ) 
  (piece_length : ℝ) 
  (overlap : ℝ) 
  (h1 : num_pieces = 15) 
  (h2 : piece_length = 25) 
  (h3 : overlap = 0.5) : 
  (num_pieces * piece_length - (num_pieces - 1) * overlap) / 100 = 3.68 := by
sorry

end NUMINAMATH_CALUDE_spliced_wire_length_l3613_361398


namespace NUMINAMATH_CALUDE_f_decreasing_iff_a_range_l3613_361397

/-- A function f(x) = x^2 + 2(a-1)x + 2 that is decreasing on (-∞, 4] -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*(a-1)

theorem f_decreasing_iff_a_range :
  ∀ a : ℝ, (∀ x ≤ 4, f_deriv a x ≤ 0) ↔ a < -3 := by sorry


end NUMINAMATH_CALUDE_f_decreasing_iff_a_range_l3613_361397


namespace NUMINAMATH_CALUDE_triangle_formation_l3613_361314

theorem triangle_formation (a b c : ℝ) : 
  a = 4 ∧ b = 9 ∧ c = 9 → 
  a + b > c ∧ b + c > a ∧ c + a > b :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l3613_361314


namespace NUMINAMATH_CALUDE_difference_of_squares_1027_l3613_361372

theorem difference_of_squares_1027 : (1027 : ℤ) * 1027 - 1026 * 1028 = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_1027_l3613_361372


namespace NUMINAMATH_CALUDE_batsman_average_increase_l3613_361387

theorem batsman_average_increase (total_innings : ℕ) (last_innings_score : ℕ) (final_average : ℚ) :
  total_innings = 12 →
  last_innings_score = 65 →
  final_average = 43 →
  (total_innings * final_average - last_innings_score) / (total_innings - 1) = 41 →
  final_average - (total_innings * final_average - last_innings_score) / (total_innings - 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l3613_361387


namespace NUMINAMATH_CALUDE_count_valid_plans_l3613_361329

/-- Represents a teacher --/
inductive Teacher : Type
  | A | B | C | D | E

/-- Represents a remote area --/
inductive Area : Type
  | One | Two | Three

/-- A dispatch plan assigns teachers to areas --/
def DispatchPlan := Teacher → Area

/-- Checks if a dispatch plan is valid according to the given conditions --/
def isValidPlan (plan : DispatchPlan) : Prop :=
  (∀ a : Area, ∃ t : Teacher, plan t = a) ∧  -- Each area has at least 1 person
  (plan Teacher.A ≠ plan Teacher.B) ∧        -- A and B are not in the same area
  (plan Teacher.A = plan Teacher.C)          -- A and C are in the same area

/-- The number of valid dispatch plans --/
def numValidPlans : ℕ := sorry

/-- Theorem stating that the number of valid dispatch plans is 30 --/
theorem count_valid_plans : numValidPlans = 30 := by sorry

end NUMINAMATH_CALUDE_count_valid_plans_l3613_361329


namespace NUMINAMATH_CALUDE_increasing_function_composition_l3613_361399

theorem increasing_function_composition (f : ℝ → ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x > y → f x - f y > x - y) →
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x > y → f (x^2) - f (y^2) > x^6 - y^6) →
  (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x > y → f (x^3) - f (y^3) > (Real.sqrt 3 / 2) * (x^6 - y^6)) :=
by sorry

end NUMINAMATH_CALUDE_increasing_function_composition_l3613_361399


namespace NUMINAMATH_CALUDE_mrs_randall_teaching_years_l3613_361304

theorem mrs_randall_teaching_years (third_grade_years second_grade_years : ℕ) 
  (h1 : third_grade_years = 18) 
  (h2 : second_grade_years = 8) : 
  third_grade_years + second_grade_years = 26 := by
  sorry

end NUMINAMATH_CALUDE_mrs_randall_teaching_years_l3613_361304


namespace NUMINAMATH_CALUDE_inequality_solution_and_sum_of_roots_l3613_361328

-- Define the inequality
def inequality (m n x : ℝ) : Prop :=
  |x^2 + m*x + n| ≤ |3*x^2 - 6*x - 9|

-- Main theorem
theorem inequality_solution_and_sum_of_roots (m n : ℝ) 
  (h : ∀ x, inequality m n x) : 
  m = -2 ∧ n = -3 ∧ 
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a + b + c = m - n → 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
sorry


end NUMINAMATH_CALUDE_inequality_solution_and_sum_of_roots_l3613_361328


namespace NUMINAMATH_CALUDE_angle_conversion_l3613_361349

theorem angle_conversion (α k : ℤ) : 
  α = 195 ∧ k = -3 → 
  0 ≤ α ∧ α < 360 ∧ 
  -885 = α + k * 360 := by
  sorry

end NUMINAMATH_CALUDE_angle_conversion_l3613_361349


namespace NUMINAMATH_CALUDE_slope_equals_half_implies_y_eleven_l3613_361317

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is 1/2, then the y-coordinate of Q is 11. -/
theorem slope_equals_half_implies_y_eleven (x₁ y₁ x₂ y₂ : ℝ) : 
  x₁ = -3 → y₁ = 7 → x₂ = 5 → 
  (y₂ - y₁) / (x₂ - x₁) = 1/2 →
  y₂ = 11 := by
  sorry

#check slope_equals_half_implies_y_eleven

end NUMINAMATH_CALUDE_slope_equals_half_implies_y_eleven_l3613_361317


namespace NUMINAMATH_CALUDE_chord_length_problem_l3613_361376

/-- The length of the chord formed by the intersection of a line and a circle -/
def chord_length (line_point : ℝ × ℝ) (parallel_line : ℝ → ℝ → ℝ → Prop) 
  (circle_center : ℝ × ℝ) (circle_radius : ℝ) : ℝ :=
  sorry

/-- The problem statement -/
theorem chord_length_problem :
  let line_point := (1, 0)
  let parallel_line := λ x y c => x - Real.sqrt 2 * y + c = 0
  let circle_center := (6, Real.sqrt 2)
  let circle_radius := Real.sqrt 12
  chord_length line_point parallel_line circle_center circle_radius = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_problem_l3613_361376


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l3613_361306

theorem sum_of_three_squares (s t : ℚ) 
  (h1 : 3 * t + 2 * s = 27)
  (h2 : 2 * t + 3 * s = 25) :
  3 * s = 63 / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l3613_361306


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l3613_361344

theorem fraction_sum_equality (p q : ℚ) (h : p / q = 4 / 5) :
  11 / 7 + (2 * q - p) / (2 * q + p) = 2 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l3613_361344


namespace NUMINAMATH_CALUDE_equation_subtraction_result_l3613_361383

theorem equation_subtraction_result :
  let eq1 : ℝ → ℝ → ℝ := fun x y => 2*x + 5*y
  let eq2 : ℝ → ℝ → ℝ := fun x y => 2*x - 3*y
  let result : ℝ → ℝ := fun y => 8*y
  ∀ x y : ℝ, eq1 x y = 9 ∧ eq2 x y = 6 →
    result y = 3 :=
by sorry

end NUMINAMATH_CALUDE_equation_subtraction_result_l3613_361383


namespace NUMINAMATH_CALUDE_abs_equal_necessary_not_sufficient_l3613_361364

theorem abs_equal_necessary_not_sufficient :
  (∀ x y : ℝ, x = y → |x| = |y|) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_abs_equal_necessary_not_sufficient_l3613_361364


namespace NUMINAMATH_CALUDE_nested_sqrt_value_l3613_361393

theorem nested_sqrt_value : 
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_value_l3613_361393


namespace NUMINAMATH_CALUDE_probability_between_R_and_S_l3613_361320

/-- Given points P, Q, R, and S on a line segment PQ, where PQ = 4PR and PQ = 8QR,
    the probability of a randomly selected point on PQ being between R and S is 5/8. -/
theorem probability_between_R_and_S (P Q R S : ℝ) : 
  P < R ∧ R < S ∧ S < Q ∧ Q - P = 4 * (R - P) ∧ Q - P = 8 * (Q - R) →
  (S - R) / (Q - P) = 5 / 8 := by
sorry

end NUMINAMATH_CALUDE_probability_between_R_and_S_l3613_361320


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3613_361352

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 3}

-- Define the union of A and B
def AUnionB : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = AUnionB := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3613_361352


namespace NUMINAMATH_CALUDE_domain_of_g_l3613_361375

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the original domain of f
def original_domain : Set ℝ := Set.Icc 1 5

-- Define the new function g(x) = f(2x - 3)
def g (x : ℝ) : ℝ := f (2 * x - 3)

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x ∈ Set.range f} = Set.Icc 2 4 :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l3613_361375


namespace NUMINAMATH_CALUDE_savings_equality_l3613_361321

/-- Proves that A's savings equal B's savings given the problem conditions --/
theorem savings_equality (total_salary : ℝ) (a_salary : ℝ) (a_spend_rate : ℝ) (b_spend_rate : ℝ)
  (h1 : total_salary = 3000)
  (h2 : a_salary = 2250)
  (h3 : a_spend_rate = 0.95)
  (h4 : b_spend_rate = 0.85) :
  a_salary * (1 - a_spend_rate) = (total_salary - a_salary) * (1 - b_spend_rate) := by
  sorry

end NUMINAMATH_CALUDE_savings_equality_l3613_361321


namespace NUMINAMATH_CALUDE_station_distance_l3613_361361

theorem station_distance (d : ℝ) : 
  (d > 0) → 
  (∃ (x_speed y_speed : ℝ), x_speed > 0 ∧ y_speed > 0 ∧ 
    (d + 100) / x_speed = (d - 100) / y_speed ∧
    (2 * d + 300) / x_speed = (d + 400) / y_speed) →
  (2 * d = 600) := by sorry

end NUMINAMATH_CALUDE_station_distance_l3613_361361


namespace NUMINAMATH_CALUDE_lens_curve_properties_l3613_361366

/-- A lens-shaped curve consisting of two equal circular arcs -/
structure LensCurve where
  radius : ℝ
  arc_angle : ℝ
  h_positive_radius : 0 < radius
  h_arc_angle : arc_angle = 2 * Real.pi / 3

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side_length : ℝ
  h_positive_side : 0 < side_length

/-- Predicate to check if a curve is closed and non-self-intersecting -/
def is_closed_non_self_intersecting (curve : Type) : Prop := sorry

/-- Predicate to check if a curve is different from a circle -/
def is_not_circle (curve : Type) : Prop := sorry

/-- Predicate to check if a triangle can be moved inside a curve with vertices tracing the curve -/
def can_move_triangle_inside (curve : Type) (triangle : Type) : Prop := sorry

theorem lens_curve_properties (l : LensCurve) (t : EquilateralTriangle) 
  (h : l.radius = t.side_length) : 
  is_closed_non_self_intersecting LensCurve ∧ 
  is_not_circle LensCurve ∧ 
  can_move_triangle_inside LensCurve EquilateralTriangle := by
  sorry

end NUMINAMATH_CALUDE_lens_curve_properties_l3613_361366


namespace NUMINAMATH_CALUDE_sons_age_l3613_361315

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 18 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 16 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3613_361315


namespace NUMINAMATH_CALUDE_product_equals_fraction_l3613_361392

/-- The repeating decimal 0.456̄ -/
def repeating_decimal : ℚ := 456 / 999

/-- The product of 0.456̄ and 7 -/
def product : ℚ := repeating_decimal * 7

/-- Theorem stating that the product of 0.456̄ and 7 is equal to 1064/333 -/
theorem product_equals_fraction : product = 1064 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_fraction_l3613_361392


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l3613_361380

/-- Proves that 2370000 is equal to 2.37 × 10^6 in scientific notation -/
theorem scientific_notation_equivalence :
  2370000 = 2.37 * (10 : ℝ)^6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l3613_361380


namespace NUMINAMATH_CALUDE_harmonic_interval_k_range_l3613_361318

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

def is_harmonic_interval (k a b : ℝ) : Prop :=
  a ≤ b ∧ a ≥ 1 ∧ b ≥ 1 ∧
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∧
  f a = k * a ∧ f b = k * b

theorem harmonic_interval_k_range :
  {k : ℝ | ∃ a b, is_harmonic_interval k a b} = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_harmonic_interval_k_range_l3613_361318


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l3613_361368

theorem partial_fraction_decomposition_sum (p q r A B C : ℝ) : 
  (p ≠ q ∧ p ≠ r ∧ q ≠ r) →
  (∀ (x : ℝ), x^3 - 16*x^2 + 72*x - 27 = (x - p) * (x - q) * (x - r)) →
  (∀ (s : ℝ), s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 16*s^2 + 72*s - 27) = A / (s - p) + B / (s - q) + C / (s - r)) →
  A + B + C = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_sum_l3613_361368


namespace NUMINAMATH_CALUDE_lending_period_is_one_year_l3613_361313

/-- 
Given a person who:
- Borrows an amount at a certain interest rate
- Lends the same amount at a higher interest rate
- Makes a fixed gain per year

This theorem proves that the lending period is 1 year under specific conditions.
-/
theorem lending_period_is_one_year 
  (borrowed_amount : ℝ) 
  (borrowing_rate : ℝ) 
  (lending_rate : ℝ) 
  (gain_per_year : ℝ) 
  (h1 : borrowed_amount = 5000)
  (h2 : borrowing_rate = 0.04)
  (h3 : lending_rate = 0.06)
  (h4 : gain_per_year = 100)
  : ∃ t : ℝ, t = 1 ∧ borrowed_amount * lending_rate * t - borrowed_amount * borrowing_rate * t = gain_per_year :=
sorry

end NUMINAMATH_CALUDE_lending_period_is_one_year_l3613_361313


namespace NUMINAMATH_CALUDE_cubic_function_property_l3613_361353

/-- Given a cubic function f(x) = ax³ + bx + 8 where f(-2) = 10, prove that f(2) = 6 -/
theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x + 8
  f (-2) = 10 → f 2 = 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_property_l3613_361353


namespace NUMINAMATH_CALUDE_bucket_weight_l3613_361335

/-- Given a bucket with weight c when 1/4 full and weight d when 3/4 full,
    prove that its weight when full is (3d - 3c)/2 -/
theorem bucket_weight (c d : ℝ) 
  (h1 : ∃ x y : ℝ, x + (1/4) * y = c ∧ x + (3/4) * y = d) : 
  ∃ w : ℝ, w = (3*d - 3*c)/2 ∧ 
  (∃ x y : ℝ, x + y = w ∧ x + (1/4) * y = c ∧ x + (3/4) * y = d) :=
by sorry

end NUMINAMATH_CALUDE_bucket_weight_l3613_361335


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l3613_361331

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + 5*m - 2023 = 0) → (n^2 + 5*n - 2023 = 0) → m^2 + 7*m + 2*n = 2013 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l3613_361331


namespace NUMINAMATH_CALUDE_rebecca_eggs_count_l3613_361386

/-- Proves that Rebecca has 13 eggs given the problem conditions -/
theorem rebecca_eggs_count :
  ∀ (total_items : ℕ) (group_size : ℕ) (num_groups : ℕ) (num_marbles : ℕ),
    group_size = 2 →
    num_groups = 8 →
    num_marbles = 3 →
    total_items = group_size * num_groups →
    total_items = num_marbles + (total_items - num_marbles) →
    (total_items - num_marbles) = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_count_l3613_361386


namespace NUMINAMATH_CALUDE_franklin_valentines_l3613_361322

/-- Represents the number of Valentines Mrs. Franklin has -/
structure ValentinesCount where
  total : ℕ
  given : ℕ

/-- Calculates the remaining Valentines after giving some away -/
def remaining_valentines (v : ValentinesCount) : ℕ :=
  v.total - v.given

/-- Theorem stating that Mrs. Franklin has 16 Valentines left -/
theorem franklin_valentines : 
  let v := ValentinesCount.mk 58 42
  remaining_valentines v = 16 := by
  sorry

end NUMINAMATH_CALUDE_franklin_valentines_l3613_361322


namespace NUMINAMATH_CALUDE_arithmetic_progression_formula_l3613_361362

/-- An arithmetic progression with specific conditions -/
def ArithmeticProgression (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 3 + a 11 = 24 ∧
  a 4 = 3

/-- The general term formula for the arithmetic progression -/
def GeneralTermFormula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = 3 * n - 9

/-- Theorem stating that the given arithmetic progression has the specified general term formula -/
theorem arithmetic_progression_formula (a : ℕ → ℝ) :
  ArithmeticProgression a → GeneralTermFormula a := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_formula_l3613_361362


namespace NUMINAMATH_CALUDE_probability_age_less_than_20_l3613_361388

theorem probability_age_less_than_20 (total : ℕ) (age_over_30 : ℕ) (age_under_20 : ℕ) :
  total = 120 →
  age_over_30 = 90 →
  age_under_20 = total - age_over_30 →
  (age_under_20 : ℚ) / (total : ℚ) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_probability_age_less_than_20_l3613_361388


namespace NUMINAMATH_CALUDE_solution_value_a_l3613_361333

theorem solution_value_a (a x y : ℝ) : 
  a * x - 3 * y = 0 ∧ x + y = 1 ∧ 2 * x + y = 0 → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_a_l3613_361333


namespace NUMINAMATH_CALUDE_remaining_honey_l3613_361382

/-- Theorem: Remaining honey after bear consumption --/
theorem remaining_honey (total_honey : ℝ) (eaten_honey : ℝ) 
  (h1 : total_honey = 0.36)
  (h2 : eaten_honey = 0.05) : 
  total_honey - eaten_honey = 0.31 := by
sorry

end NUMINAMATH_CALUDE_remaining_honey_l3613_361382


namespace NUMINAMATH_CALUDE_harkamal_fruit_payment_l3613_361395

/-- Calculates the total amount Harkamal had to pay for fruits with given quantities, prices, discount, and tax rates. -/
def calculate_total_payment (grape_kg : ℝ) (grape_price : ℝ) (mango_kg : ℝ) (mango_price : ℝ)
                            (apple_kg : ℝ) (apple_price : ℝ) (orange_kg : ℝ) (orange_price : ℝ)
                            (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let grape_total := grape_kg * grape_price
  let mango_total := mango_kg * mango_price
  let apple_total := apple_kg * apple_price
  let orange_total := orange_kg * orange_price
  let total_before_discount := grape_total + mango_total + apple_total + orange_total
  let discount := discount_rate * (grape_total + apple_total)
  let price_after_discount := total_before_discount - discount
  let tax := tax_rate * price_after_discount
  price_after_discount + tax

/-- Theorem stating that the total payment for Harkamal's fruit purchase is $1507.32. -/
theorem harkamal_fruit_payment :
  calculate_total_payment 9 70 9 55 5 40 6 30 0.1 0.06 = 1507.32 := by
  sorry

end NUMINAMATH_CALUDE_harkamal_fruit_payment_l3613_361395


namespace NUMINAMATH_CALUDE_single_windows_upstairs_correct_number_of_single_windows_l3613_361342

theorem single_windows_upstairs 
  (double_windows : ℕ) 
  (panels_per_double : ℕ) 
  (panels_per_single : ℕ) 
  (total_panels : ℕ) : ℕ :=
  let downstairs_panels := double_windows * panels_per_double
  let upstairs_panels := total_panels - downstairs_panels
  upstairs_panels / panels_per_single

theorem correct_number_of_single_windows :
  single_windows_upstairs 6 4 4 80 = 14 := by
  sorry

end NUMINAMATH_CALUDE_single_windows_upstairs_correct_number_of_single_windows_l3613_361342


namespace NUMINAMATH_CALUDE_tomato_ratio_l3613_361339

def total_tomatoes : ℕ := 127
def eaten_by_birds : ℕ := 19
def tomatoes_left : ℕ := 54

theorem tomato_ratio :
  let picked := total_tomatoes - eaten_by_birds
  let given_to_friend := picked - tomatoes_left
  (given_to_friend : ℚ) / picked = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tomato_ratio_l3613_361339


namespace NUMINAMATH_CALUDE_f_explicit_formula_b_value_l3613_361301

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2 + 5

-- Define the function g
def g (b : ℝ) : ℝ → ℝ := fun x ↦ f x - b * x

-- Theorem for the first part
theorem f_explicit_formula : ∀ x : ℝ, f (x - 2) = x^2 - 4*x + 9 := by sorry

-- Theorem for the second part
theorem b_value : 
  ∃ b : ℝ, b = 1/2 ∧ 
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, g b x ≤ 11/2) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 1, g b x = 11/2) := by sorry

end NUMINAMATH_CALUDE_f_explicit_formula_b_value_l3613_361301


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3613_361389

-- Define the solution set based on the value of a
def solutionSet (a : ℝ) : Set ℝ :=
  if a > 0 then {x | x < -a/4 ∨ x > a/3}
  else if a = 0 then {x | x ≠ 0}
  else {x | x > -a/4 ∨ x < a/3}

-- Theorem statement
theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | 12 * x^2 - a * x > a^2} = solutionSet a := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3613_361389


namespace NUMINAMATH_CALUDE_trains_distance_before_meeting_l3613_361358

/-- The distance between two trains one hour before they meet -/
def distance_before_meeting (speed_A speed_B : ℝ) : ℝ :=
  speed_A + speed_B

theorem trains_distance_before_meeting 
  (speed_A speed_B total_distance : ℝ)
  (h1 : speed_A = 60)
  (h2 : speed_B = 40)
  (h3 : total_distance ≤ 250) :
  distance_before_meeting speed_A speed_B = 100 := by
  sorry

#check trains_distance_before_meeting

end NUMINAMATH_CALUDE_trains_distance_before_meeting_l3613_361358


namespace NUMINAMATH_CALUDE_remainder_problem_l3613_361337

theorem remainder_problem (n : ℤ) (h : n % 5 = 3) : (n + 1) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3613_361337


namespace NUMINAMATH_CALUDE_additional_calories_burnt_l3613_361325

def calories_per_hour : ℕ := 30

def calories_burnt (hours : ℕ) : ℕ := calories_per_hour * hours

theorem additional_calories_burnt : 
  calories_burnt 5 - calories_burnt 2 = 90 := by
  sorry

end NUMINAMATH_CALUDE_additional_calories_burnt_l3613_361325


namespace NUMINAMATH_CALUDE_cone_surface_area_l3613_361357

/-- The surface area of a cone with lateral surface as a sector of a circle 
    with radius 2 and central angle π/2 is 5π/4 -/
theorem cone_surface_area : 
  ∀ (cone : Real → Real → Real),
  (∀ r θ, cone r θ = 2 * π * r^2 * (θ / (2 * π)) + π * r^2) →
  cone 2 (π / 2) = 5 * π / 4 :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l3613_361357


namespace NUMINAMATH_CALUDE_min_value_quadratic_sum_l3613_361373

theorem min_value_quadratic_sum (x y z : ℝ) (h : x + y + z = 1) :
  2 * x^2 + 3 * y^2 + z^2 ≥ 6/11 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_sum_l3613_361373


namespace NUMINAMATH_CALUDE_age_problem_l3613_361311

theorem age_problem (a b : ℚ) : 
  (a = 2 * (b - (a - b))) →  -- Condition 1
  (a + (a - b) + b + (a - b) = 130) →  -- Condition 2
  (a = 57 + 7/9 ∧ b = 43 + 1/3) := by
sorry

end NUMINAMATH_CALUDE_age_problem_l3613_361311


namespace NUMINAMATH_CALUDE_complex_number_sum_l3613_361396

theorem complex_number_sum (z : ℂ) : z = (2 + Complex.I) / (1 - 2 * Complex.I) → 
  ∃ (a b : ℝ), z = a + b * Complex.I ∧ a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_l3613_361396


namespace NUMINAMATH_CALUDE_train_interval_l3613_361347

-- Define the train route times
def northern_route_time : ℝ := 17
def southern_route_time : ℝ := 11

-- Define the average time difference between counterclockwise and clockwise trains
def train_arrival_difference : ℝ := 1.25

-- Define the commute time difference
def commute_time_difference : ℝ := 1

-- Theorem statement
theorem train_interval (p : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 1) 
  (hcommute : southern_route_time * p + northern_route_time * (1 - p) + 1 = 
              northern_route_time * p + southern_route_time * (1 - p))
  (htrain_diff : (1 - p) * 3 = train_arrival_difference) : 
  3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_train_interval_l3613_361347


namespace NUMINAMATH_CALUDE_P_and_S_not_third_l3613_361394

-- Define the set of runners
inductive Runner : Type
| P | Q | R | S | T | U

-- Define the finish order relation
def finishes_before (a b : Runner) : Prop := sorry

-- Define the race conditions
axiom P_beats_Q : finishes_before Runner.P Runner.Q
axiom P_beats_R : finishes_before Runner.P Runner.R
axiom Q_beats_S : finishes_before Runner.Q Runner.S
axiom U_after_P_before_T : finishes_before Runner.P Runner.U ∧ finishes_before Runner.U Runner.T
axiom T_after_P_before_Q : finishes_before Runner.P Runner.T ∧ finishes_before Runner.T Runner.Q

-- Define a function to represent the finishing position of a runner
def finish_position (r : Runner) : ℕ := sorry

-- State the theorem
theorem P_and_S_not_third :
  ¬(finish_position Runner.P = 3 ∨ finish_position Runner.S = 3) :=
sorry

end NUMINAMATH_CALUDE_P_and_S_not_third_l3613_361394


namespace NUMINAMATH_CALUDE_triangle_formation_l3613_361330

/-- Triangle Inequality Theorem: A triangle can be formed if the sum of the lengths of any two sides
    is greater than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given three line segments with lengths 4cm, 5cm, and 6cm, they can form a triangle. -/
theorem triangle_formation : can_form_triangle 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_formation_l3613_361330


namespace NUMINAMATH_CALUDE_volume_ratio_cylinders_capacity_ratio_64_percent_l3613_361338

/-- The volume ratio of two right circular cylinders with the same height
    is equal to the square of the ratio of their circumferences. -/
theorem volume_ratio_cylinders (h C_A C_B : ℝ) (h_pos : h > 0) (C_A_pos : C_A > 0) (C_B_pos : C_B > 0) :
  (h * (C_A / (2 * Real.pi))^2) / (h * (C_B / (2 * Real.pi))^2) = (C_A / C_B)^2 := by
  sorry

/-- The capacity of a cylinder with circumference 8 is 64% of the capacity
    of a cylinder with circumference 10, given the same height. -/
theorem capacity_ratio_64_percent (h : ℝ) (h_pos : h > 0) :
  (h * (8 / (2 * Real.pi))^2) / (h * (10 / (2 * Real.pi))^2) = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_cylinders_capacity_ratio_64_percent_l3613_361338


namespace NUMINAMATH_CALUDE_trajectory_and_fixed_point_l3613_361300

-- Define the points and conditions
def A₁ : ℝ × ℝ := (-2, 0)
def A₂ : ℝ × ℝ := (2, 0)
def F₂ : ℝ × ℝ := (1, 0)

-- Define the function for the trajectory M
def M (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1 ∧ x ≠ 2 ∧ x ≠ -2

-- Define the line l
def l (k m : ℝ) (x y : ℝ) : Prop :=
  y = k * x + m

-- Theorem statement
theorem trajectory_and_fixed_point 
  (m n : ℝ) 
  (h_mn : m * n = 3)
  (k α β : ℝ) 
  (h_αβ : α + β = Real.pi) :
  (∃ x y : ℝ, M x y) ∧ 
  (∃ k m : ℝ, ∀ x y : ℝ, l k m x y → y = k * (x - 4)) :=
sorry

end NUMINAMATH_CALUDE_trajectory_and_fixed_point_l3613_361300


namespace NUMINAMATH_CALUDE_point_on_x_axis_l3613_361390

/-- A point P with coordinates (m+3, m-1) lies on the x-axis if and only if its coordinates are (4, 0) -/
theorem point_on_x_axis (m : ℝ) : 
  (m - 1 = 0 ∧ (m + 3, m - 1) = (m + 3, 0)) ↔ (m + 3, m - 1) = (4, 0) :=
by sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l3613_361390


namespace NUMINAMATH_CALUDE_ratio_calculation_l3613_361334

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by sorry

end NUMINAMATH_CALUDE_ratio_calculation_l3613_361334


namespace NUMINAMATH_CALUDE_complex_modulus_l3613_361360

theorem complex_modulus (i : ℂ) (h : i * i = -1) : 
  Complex.abs (5 * i / (2 - i)) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_l3613_361360


namespace NUMINAMATH_CALUDE_quadratic_form_k_value_l3613_361348

theorem quadratic_form_k_value : 
  ∃ (a h k : ℚ), ∀ x, x^2 - 7*x = a*(x - h)^2 + k ∧ k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_k_value_l3613_361348


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l3613_361326

theorem geometric_series_first_term 
  (a r : ℝ) 
  (sum_condition : a / (1 - r) = 20) 
  (sum_squares_condition : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l3613_361326


namespace NUMINAMATH_CALUDE_race_theorem_l3613_361309

/-- A race with 40 kids where some finish under 6 minutes, some under 8 minutes, and the rest take longer. -/
structure Race where
  total_kids : ℕ
  under_6_min : ℕ
  under_8_min : ℕ
  over_certain_min : ℕ

/-- The race satisfies the given conditions. -/
def race_conditions (r : Race) : Prop :=
  r.total_kids = 40 ∧
  r.under_6_min = (10 : ℕ) * r.total_kids / 100 ∧
  r.under_8_min = 3 * r.under_6_min ∧
  r.over_certain_min = 4 ∧
  r.over_certain_min = (r.total_kids - (r.under_6_min + r.under_8_min)) / 6

/-- The theorem stating that the number of kids who take more than a certain number of minutes is 4. -/
theorem race_theorem (r : Race) (h : race_conditions r) : r.over_certain_min = 4 := by
  sorry


end NUMINAMATH_CALUDE_race_theorem_l3613_361309


namespace NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l3613_361319

theorem inscribed_sphere_surface_area (cube_edge : ℝ) (sphere_area : ℝ) :
  cube_edge = 2 →
  sphere_area = 4 * Real.pi →
  sphere_area = (4 : ℝ) * Real.pi * (cube_edge / 2) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l3613_361319


namespace NUMINAMATH_CALUDE_triangle_right_angled_l3613_361370

theorem triangle_right_angled (A B C : Real) (a b c : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A + B + C = π →
  a = -c * Real.cos (A + C) →
  a^2 + b^2 = c^2 :=
sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l3613_361370


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l3613_361374

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := -8/21
  let a₃ : ℚ := 16/63
  let r : ℚ := a₂ / a₁
  (r = -2/3) ∧ (a₃ / a₂ = r) := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l3613_361374
