import Mathlib

namespace NUMINAMATH_CALUDE_min_box_value_l3862_386279

theorem min_box_value (a b Box : ℤ) : 
  (a ≠ b ∧ a ≠ Box ∧ b ≠ Box) →
  (∀ x, (a * x + b) * (b * x + a) = 31 * x^2 + Box * x + 31) →
  962 ≤ Box :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l3862_386279


namespace NUMINAMATH_CALUDE_largest_digit_sum_for_special_fraction_l3862_386277

/-- A digit is a natural number between 0 and 9 inclusive -/
def Digit := {n : ℕ // n ≤ 9}

/-- abc represents a three-digit number -/
def ThreeDigitNumber (a b c : Digit) : ℕ := 100 * a.val + 10 * b.val + c.val

theorem largest_digit_sum_for_special_fraction :
  ∃ (a b c : Digit) (y : ℕ),
    (10 ≤ y ∧ y ≤ 99) ∧
    (ThreeDigitNumber a b c : ℚ) / 1000 = 1 / y ∧
    ∀ (a' b' c' : Digit) (y' : ℕ),
      (10 ≤ y' ∧ y' ≤ 99) →
      (ThreeDigitNumber a' b' c' : ℚ) / 1000 = 1 / y' →
      a.val + b.val + c.val ≥ a'.val + b'.val + c'.val ∧
      a.val + b.val + c.val = 7 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_for_special_fraction_l3862_386277


namespace NUMINAMATH_CALUDE_square_root_difference_product_l3862_386296

theorem square_root_difference_product : (Real.sqrt 100 + Real.sqrt 9) * (Real.sqrt 100 - Real.sqrt 9) = 91 := by
  sorry

end NUMINAMATH_CALUDE_square_root_difference_product_l3862_386296


namespace NUMINAMATH_CALUDE_grasshopper_cannot_return_after_25_jumps_l3862_386237

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The position of the grasshopper after n jumps -/
def grasshopper_position (n : ℕ) : ℕ := sum_first_n n

theorem grasshopper_cannot_return_after_25_jumps :
  ∃ k : ℕ, grasshopper_position 25 = 2 * k + 1 :=
sorry

end NUMINAMATH_CALUDE_grasshopper_cannot_return_after_25_jumps_l3862_386237


namespace NUMINAMATH_CALUDE_map_distance_conversion_l3862_386293

/-- Proves that given a map scale where 312 inches represents 136 km,
    a point 25 inches away on the map corresponds to approximately 10.9 km
    in actual distance. -/
theorem map_distance_conversion
  (map_distance : ℝ) (actual_distance : ℝ) (point_on_map : ℝ)
  (h1 : map_distance = 312)
  (h2 : actual_distance = 136)
  (h3 : point_on_map = 25) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧
  abs ((actual_distance / map_distance) * point_on_map - 10.9) < ε :=
sorry

end NUMINAMATH_CALUDE_map_distance_conversion_l3862_386293


namespace NUMINAMATH_CALUDE_equal_chord_lengths_l3862_386231

/-- The ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 8 = 1

/-- The first line -/
def line1 (k x y : ℝ) : Prop := k * x + y - 2 = 0

/-- The second line -/
def line2 (k x y : ℝ) : Prop := y = k * x + 2

/-- Length of the chord intercepted by a line on the ellipse -/
noncomputable def chord_length (line : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem equal_chord_lengths (k : ℝ) :
  chord_length (line1 k) = chord_length (line2 k) :=
sorry

end NUMINAMATH_CALUDE_equal_chord_lengths_l3862_386231


namespace NUMINAMATH_CALUDE_dot_product_theorem_l3862_386206

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 2)

theorem dot_product_theorem (c : ℝ × ℝ) 
  (h : c = (3 * a.1 + 2 * b.1 - a.1, 3 * a.2 + 2 * b.2 - a.2)) :
  a.1 * c.1 + a.2 * c.2 = 4 := by sorry

end NUMINAMATH_CALUDE_dot_product_theorem_l3862_386206


namespace NUMINAMATH_CALUDE_count_special_numbers_is_279_l3862_386282

/-- A function that counts the number of positive integers less than 100,000 
    with at most two different digits, where one of the digits must be 1. -/
def count_special_numbers : ℕ :=
  let max_number := 100000
  let required_digit := 1
  -- Implementation details are omitted
  279

/-- Theorem stating that the count of special numbers is 279. -/
theorem count_special_numbers_is_279 : count_special_numbers = 279 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_is_279_l3862_386282


namespace NUMINAMATH_CALUDE_root_preservation_l3862_386208

/-- Given a polynomial P(x) = x^3 + ax^2 + bx + c with three distinct real roots,
    the polynomial Q(x) = x^3 + ax^2 + (1/4)(a^2 + b)x + (1/8)(ab - c) also has three distinct real roots. -/
theorem root_preservation (a b c : ℝ) 
  (h : ∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    (∀ x, x^3 + a*x^2 + b*x + c = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) :
  ∃ (y₁ y₂ y₃ : ℝ), y₁ ≠ y₂ ∧ y₂ ≠ y₃ ∧ y₁ ≠ y₃ ∧
    (∀ x, x^3 + a*x^2 + (1/4)*(a^2 + b)*x + (1/8)*(a*b - c) = 0 ↔ x = y₁ ∨ x = y₂ ∨ x = y₃) :=
by sorry

end NUMINAMATH_CALUDE_root_preservation_l3862_386208


namespace NUMINAMATH_CALUDE_orange_sale_savings_l3862_386278

/-- Calculates the total savings for a mother's birthday gift based on orange sales. -/
theorem orange_sale_savings 
  (liam_oranges : ℕ) 
  (liam_price : ℚ) 
  (claire_oranges : ℕ) 
  (claire_price : ℚ) 
  (h1 : liam_oranges = 40)
  (h2 : liam_price = 5/2)
  (h3 : claire_oranges = 30)
  (h4 : claire_price = 6/5)
  : ℚ :=
by
  sorry

#check orange_sale_savings

end NUMINAMATH_CALUDE_orange_sale_savings_l3862_386278


namespace NUMINAMATH_CALUDE_system_three_solutions_l3862_386238

def system (a : ℝ) (x y : ℝ) : Prop :=
  y = |x - Real.sqrt a| + Real.sqrt a - 2 ∧
  (|x| - 4)^2 + (|y| - 3)^2 = 25

def has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    system a x₁ y₁ ∧ system a x₂ y₂ ∧ system a x₃ y₃ ∧
    (∀ x y, system a x y → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃))

theorem system_three_solutions :
  ∀ a : ℝ, has_exactly_three_solutions a ↔ a = 1 ∨ a = 16 ∨ a = ((5 * Real.sqrt 2 + 1) / 2)^2 :=
sorry

end NUMINAMATH_CALUDE_system_three_solutions_l3862_386238


namespace NUMINAMATH_CALUDE_bird_cage_problem_l3862_386246

theorem bird_cage_problem (N : ℚ) : 
  (5/8 * (4/5 * (1/2 * N + 12) + 20) = 60) → N = 166 := by
  sorry

end NUMINAMATH_CALUDE_bird_cage_problem_l3862_386246


namespace NUMINAMATH_CALUDE_complement_of_intersection_main_theorem_l3862_386264

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2, 3}

-- Define set B
def B : Set Nat := {2, 3, 4}

-- Theorem statement
theorem complement_of_intersection (x : Nat) : 
  x ∈ (A ∩ B)ᶜ ↔ (x ∈ U ∧ x ∉ (A ∩ B)) :=
by
  sorry

-- Main theorem to prove
theorem main_theorem : (A ∩ B)ᶜ = {1, 4} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_of_intersection_main_theorem_l3862_386264


namespace NUMINAMATH_CALUDE_fermats_little_theorem_l3862_386209

theorem fermats_little_theorem (p a : ℕ) (hp : Prime p) (ha : ¬(p ∣ a)) :
  a^(p-1) ≡ 1 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_fermats_little_theorem_l3862_386209


namespace NUMINAMATH_CALUDE_prob_only_one_value_l3862_386262

/-- The probability that student A solves the problem -/
def prob_A : ℚ := 1/2

/-- The probability that student B solves the problem -/
def prob_B : ℚ := 1/3

/-- The probability that student C solves the problem -/
def prob_C : ℚ := 1/4

/-- The probability that only one student solves the problem -/
def prob_only_one : ℚ :=
  prob_A * (1 - prob_B) * (1 - prob_C) +
  prob_B * (1 - prob_A) * (1 - prob_C) +
  prob_C * (1 - prob_A) * (1 - prob_B)

theorem prob_only_one_value : prob_only_one = 11/24 := by
  sorry

end NUMINAMATH_CALUDE_prob_only_one_value_l3862_386262


namespace NUMINAMATH_CALUDE_complement_S_union_T_equals_interval_l3862_386207

open Set Real

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem complement_S_union_T_equals_interval :
  (𝒰 \ S) ∪ T = Iic 1 := by sorry

end NUMINAMATH_CALUDE_complement_S_union_T_equals_interval_l3862_386207


namespace NUMINAMATH_CALUDE_solve_salary_problem_l3862_386253

def salary_problem (salary_A salary_B : ℝ) : Prop :=
  salary_A + salary_B = 3000 ∧
  0.05 * salary_A = 0.15 * salary_B

theorem solve_salary_problem :
  ∃ (salary_A : ℝ), salary_problem salary_A (3000 - salary_A) ∧ salary_A = 2250 := by
  sorry

end NUMINAMATH_CALUDE_solve_salary_problem_l3862_386253


namespace NUMINAMATH_CALUDE_evaluate_expression_l3862_386224

theorem evaluate_expression (x y z : ℚ) (hx : x = 1/2) (hy : y = 1/3) (hz : z = -3) :
  (2*x)^2 * (y^2)^3 * z^2 = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3862_386224


namespace NUMINAMATH_CALUDE_sisters_name_length_sisters_name_length_is_five_l3862_386240

theorem sisters_name_length (jonathan_first_name_length : ℕ) 
                             (jonathan_surname_length : ℕ) 
                             (sister_surname_length : ℕ) 
                             (total_letters : ℕ) : ℕ :=
  let jonathan_full_name_length := jonathan_first_name_length + jonathan_surname_length
  let sister_first_name_length := total_letters - jonathan_full_name_length - sister_surname_length
  sister_first_name_length

theorem sisters_name_length_is_five : 
  sisters_name_length 8 10 10 33 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sisters_name_length_sisters_name_length_is_five_l3862_386240


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l3862_386221

theorem five_digit_divisible_by_nine :
  ∃! d : ℕ, d < 10 ∧ (34700 + 10 * d + 9) % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l3862_386221


namespace NUMINAMATH_CALUDE_sum_remainder_mod_9_l3862_386288

theorem sum_remainder_mod_9 : (9237 + 9238 + 9239 + 9240 + 9241) % 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_9_l3862_386288


namespace NUMINAMATH_CALUDE_stating_mans_downstream_speed_l3862_386225

/-- 
Given a man's upstream speed and the speed of a stream, 
this function calculates his downstream speed.
-/
def downstream_speed (upstream_speed stream_speed : ℝ) : ℝ :=
  (upstream_speed + stream_speed) + stream_speed

/-- 
Theorem stating that given the specific conditions of the problem,
the man's downstream speed is 11 kmph.
-/
theorem mans_downstream_speed : 
  downstream_speed 8 1.5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_stating_mans_downstream_speed_l3862_386225


namespace NUMINAMATH_CALUDE_smallest_integer_solution_system_of_inequalities_solution_l3862_386267

-- Part 1
theorem smallest_integer_solution (x : ℤ) :
  (5 * x + 15 > x - 1) ∧ (∀ y : ℤ, y < x → ¬(5 * y + 15 > y - 1)) ↔ x = -3 :=
sorry

-- Part 2
theorem system_of_inequalities_solution (x : ℝ) :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 4 * x) / 3 > x - 1) ↔ -4 < x ∧ x ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_system_of_inequalities_solution_l3862_386267


namespace NUMINAMATH_CALUDE_two_color_theorem_l3862_386226

/-- Represents a region in the plane --/
structure Region where
  id : Nat

/-- Represents the configuration of circles and lines --/
structure Configuration where
  regions : List Region
  adjacency : Region → Region → Bool

/-- Represents a coloring of regions --/
def Coloring := Region → Bool

/-- A valid coloring is one where adjacent regions have different colors --/
def is_valid_coloring (config : Configuration) (coloring : Coloring) : Prop :=
  ∀ r1 r2, config.adjacency r1 r2 → coloring r1 ≠ coloring r2

theorem two_color_theorem (config : Configuration) :
  ∃ (coloring : Coloring), is_valid_coloring config coloring := by
  sorry

end NUMINAMATH_CALUDE_two_color_theorem_l3862_386226


namespace NUMINAMATH_CALUDE_james_water_storage_l3862_386286

/-- Represents the water storage problem with different container types --/
structure WaterStorage where
  barrelCount : ℕ
  largeCaskCount : ℕ
  smallCaskCount : ℕ
  largeCaskCapacity : ℕ

/-- Calculates the total water storage capacity --/
def totalCapacity (storage : WaterStorage) : ℕ :=
  let barrelCapacity := 2 * storage.largeCaskCapacity + 3
  let smallCaskCapacity := storage.largeCaskCapacity / 2
  storage.barrelCount * barrelCapacity +
  storage.largeCaskCount * storage.largeCaskCapacity +
  storage.smallCaskCount * smallCaskCapacity

/-- Theorem stating that James' total water storage capacity is 282 gallons --/
theorem james_water_storage :
  let storage : WaterStorage := {
    barrelCount := 4,
    largeCaskCount := 3,
    smallCaskCount := 5,
    largeCaskCapacity := 20
  }
  totalCapacity storage = 282 := by
  sorry

end NUMINAMATH_CALUDE_james_water_storage_l3862_386286


namespace NUMINAMATH_CALUDE_fourth_grade_students_l3862_386281

theorem fourth_grade_students (initial_students : ℝ) (left_students : ℝ) (transferred_students : ℝ) :
  initial_students = 42.0 →
  left_students = 4.0 →
  transferred_students = 10.0 →
  initial_students - left_students - transferred_students = 28.0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_grade_students_l3862_386281


namespace NUMINAMATH_CALUDE_police_catch_time_police_catch_rogue_l3862_386201

/-- The time it takes for a police spaceship to catch up with a rogue spaceship -/
theorem police_catch_time (rogue_speed : ℝ) (head_start_minutes : ℝ) (police_speed_increase : ℝ) : ℝ :=
  let head_start_hours := head_start_minutes / 60
  let police_speed := rogue_speed * (1 + police_speed_increase)
  let distance_traveled := rogue_speed * head_start_hours
  let relative_speed := police_speed - rogue_speed
  let catch_up_time_hours := distance_traveled / relative_speed
  catch_up_time_hours * 60

/-- The police will catch up with the rogue spaceship in 450 minutes -/
theorem police_catch_rogue : 
  ∀ (rogue_speed : ℝ), rogue_speed > 0 → police_catch_time rogue_speed 54 0.12 = 450 :=
by
  sorry


end NUMINAMATH_CALUDE_police_catch_time_police_catch_rogue_l3862_386201


namespace NUMINAMATH_CALUDE_classroom_capacity_l3862_386290

/-- Calculates the total number of desks in a classroom with an arithmetic progression of desks per row -/
def totalDesks (rows : ℕ) (firstRowDesks : ℕ) (increment : ℕ) : ℕ :=
  rows * (2 * firstRowDesks + (rows - 1) * increment) / 2

/-- Theorem stating that a classroom with 8 rows, starting with 10 desks and increasing by 2 each row, can seat 136 students -/
theorem classroom_capacity :
  totalDesks 8 10 2 = 136 := by
  sorry

#eval totalDesks 8 10 2

end NUMINAMATH_CALUDE_classroom_capacity_l3862_386290


namespace NUMINAMATH_CALUDE_find_y_value_l3862_386260

theorem find_y_value (x y : ℝ) (h1 : x * y = 4) (h2 : x / y = 81) (h3 : x > 0) (h4 : y > 0) :
  y = 2 / 9 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l3862_386260


namespace NUMINAMATH_CALUDE_intersection_not_roots_l3862_386239

theorem intersection_not_roots : ∀ x : ℝ,
  (x^2 - 1 = x + 7) → (x^2 + x - 6 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_not_roots_l3862_386239


namespace NUMINAMATH_CALUDE_f_has_one_or_two_zeros_l3862_386294

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x - m^2

-- State the theorem
theorem f_has_one_or_two_zeros (m : ℝ) :
  ∃ (x₁ x₂ : ℝ), f m x₁ = 0 ∧ f m x₂ = 0 ∧ (x₁ = x₂ ∨ x₁ ≠ x₂) :=
sorry

end NUMINAMATH_CALUDE_f_has_one_or_two_zeros_l3862_386294


namespace NUMINAMATH_CALUDE_cube_edge_sum_l3862_386261

theorem cube_edge_sum (surface_area : ℝ) (h : surface_area = 150) :
  let side_length := Real.sqrt (surface_area / 6)
  12 * side_length = 60 := by sorry

end NUMINAMATH_CALUDE_cube_edge_sum_l3862_386261


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3862_386203

/-- An arithmetic sequence with first term a, common difference d, and index n -/
def arithmeticSequence (a d : ℚ) (n : ℕ) : ℚ := a + d * n

theorem arithmetic_sequence_solution :
  ∃ (x : ℚ),
    (arithmeticSequence (3/4) d 0 = 3/4) ∧
    (arithmeticSequence (3/4) d 1 = x + 1) ∧
    (arithmeticSequence (3/4) d 2 = 5*x) →
    x = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solution_l3862_386203


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3862_386248

theorem quadratic_factorization :
  ∀ x : ℝ, 12 * x^2 + 16 * x - 20 = 4 * (x - 1) * (3 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3862_386248


namespace NUMINAMATH_CALUDE_milk_dilution_l3862_386243

theorem milk_dilution (initial_volume : ℝ) (initial_milk_percentage : ℝ) (water_added : ℝ) :
  initial_volume = 60 →
  initial_milk_percentage = 0.84 →
  water_added = 18.75 →
  let initial_milk_volume := initial_volume * initial_milk_percentage
  let final_volume := initial_volume + water_added
  let final_milk_percentage := initial_milk_volume / final_volume
  final_milk_percentage = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_milk_dilution_l3862_386243


namespace NUMINAMATH_CALUDE_overall_percentage_increase_l3862_386247

def initial_price_A : ℝ := 300
def initial_price_B : ℝ := 150
def initial_price_C : ℝ := 50
def initial_price_D : ℝ := 100

def new_price_A : ℝ := 390
def new_price_B : ℝ := 180
def new_price_C : ℝ := 70
def new_price_D : ℝ := 110

def total_initial_price : ℝ := initial_price_A + initial_price_B + initial_price_C + initial_price_D
def total_new_price : ℝ := new_price_A + new_price_B + new_price_C + new_price_D

theorem overall_percentage_increase :
  (total_new_price - total_initial_price) / total_initial_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_overall_percentage_increase_l3862_386247


namespace NUMINAMATH_CALUDE_min_value_mn_l3862_386222

def f (a x : ℝ) : ℝ := |x - a|

theorem min_value_mn (a m n : ℝ) : 
  (∀ x, f a x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  m > 0 →
  n > 0 →
  1/m + 1/(2*n) = a →
  ∀ k, m * n ≤ k → 2 ≤ k :=
by sorry

end NUMINAMATH_CALUDE_min_value_mn_l3862_386222


namespace NUMINAMATH_CALUDE_reciprocal_equation_l3862_386223

theorem reciprocal_equation (x : ℝ) : 1 - 1 / (1 - x) = 1 / (1 - x) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equation_l3862_386223


namespace NUMINAMATH_CALUDE_division_remainder_proof_l3862_386280

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : dividend = 125)
  (h2 : divisor = 15)
  (h3 : quotient = 8)
  (h4 : dividend = divisor * quotient + remainder) :
  remainder = 5 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_proof_l3862_386280


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3862_386283

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 2*a - 2 = 0) → 
  (b^3 - 2*b - 2 = 0) → 
  (c^3 - 2*c - 2 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3862_386283


namespace NUMINAMATH_CALUDE_smallest_angle_in_3_4_5_ratio_triangle_l3862_386270

theorem smallest_angle_in_3_4_5_ratio_triangle (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- angles are positive
  a + b + c = 180 →  -- sum of angles in a triangle
  ∃ (k : ℝ), a = 3*k ∧ b = 4*k ∧ c = 5*k →  -- angles are in ratio 3:4:5
  min a (min b c) = 45 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_3_4_5_ratio_triangle_l3862_386270


namespace NUMINAMATH_CALUDE_min_shift_value_l3862_386202

open Real

noncomputable def f (x : ℝ) : ℝ := sin x * cos x - Real.sqrt 3 * cos x ^ 2

noncomputable def g (x : ℝ) : ℝ := sin (2 * x + π / 3) - Real.sqrt 3 / 2

theorem min_shift_value (k : ℝ) (h : k > 0) :
  (∀ x, f x = g (x - k)) ↔ k ≥ π / 3 :=
sorry

end NUMINAMATH_CALUDE_min_shift_value_l3862_386202


namespace NUMINAMATH_CALUDE_decimal_expansion_18_37_l3862_386265

/-- The decimal expansion of 18/37 has a repeating pattern of length 3 -/
def decimal_expansion_period (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧
  (18 : ℚ) / 37 = (a * 100 + b * 10 + c : ℚ) / 999

/-- The 123rd digit after the decimal point in the expansion of 18/37 -/
def digit_123 : ℕ := 6

theorem decimal_expansion_18_37 :
  decimal_expansion_period 3 ∧ digit_123 = 6 :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_18_37_l3862_386265


namespace NUMINAMATH_CALUDE_triangle_value_l3862_386272

theorem triangle_value (triangle p : ℝ) 
  (eq1 : 2 * triangle + p = 72)
  (eq2 : triangle + p + 2 * triangle = 128) :
  triangle = 56 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l3862_386272


namespace NUMINAMATH_CALUDE_smallest_c_value_l3862_386297

theorem smallest_c_value (c d : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x : ℝ, x^3 - c*x^2 + d*x - 2550 = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  r₁ * r₂ * r₃ = 2550 →
  c = r₁ + r₂ + r₃ →
  c ≥ 42 :=
by sorry

end NUMINAMATH_CALUDE_smallest_c_value_l3862_386297


namespace NUMINAMATH_CALUDE_infinite_solutions_l3862_386245

theorem infinite_solutions (b : ℝ) : 
  (∀ x, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_l3862_386245


namespace NUMINAMATH_CALUDE_omega_on_real_axis_l3862_386230

theorem omega_on_real_axis (z : ℂ) (h1 : z.re ≠ 0) (h2 : Complex.abs z = 1) :
  let ω := z + z⁻¹
  ω.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_omega_on_real_axis_l3862_386230


namespace NUMINAMATH_CALUDE_line_plane_relations_l3862_386215

/-- The direction vector of line l -/
def m (a b : ℝ) : ℝ × ℝ × ℝ := (1, a + b, a - b)

/-- The normal vector of plane α -/
def n : ℝ × ℝ × ℝ := (1, 2, 3)

/-- Line l is parallel to plane α -/
def is_parallel (a b : ℝ) : Prop :=
  let (x₁, y₁, z₁) := m a b
  let (x₂, y₂, z₂) := n
  x₁ * x₂ + y₁ * y₂ + z₁ * z₂ = 0

/-- Line l is perpendicular to plane α -/
def is_perpendicular (a b : ℝ) : Prop :=
  let (x₁, y₁, z₁) := m a b
  let (x₂, y₂, z₂) := n
  x₁ / x₂ = y₁ / y₂ ∧ x₁ / x₂ = z₁ / z₂

theorem line_plane_relations (a b : ℝ) :
  (is_parallel a b → 5 * a - b + 1 = 0) ∧
  (is_perpendicular a b → a + b - 2 = 0 ∧ a - b - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_line_plane_relations_l3862_386215


namespace NUMINAMATH_CALUDE_cafeteria_shirts_l3862_386229

theorem cafeteria_shirts (total : ℕ) (checkered : ℕ) (horizontal : ℕ) (vertical : ℕ) : 
  total = 40 →
  checkered = 7 →
  horizontal = 4 * checkered →
  vertical = total - (checkered + horizontal) →
  vertical = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_cafeteria_shirts_l3862_386229


namespace NUMINAMATH_CALUDE_smallest_solutions_l3862_386271

/-- The function that checks if a given positive integer k satisfies the equation cos²(k² + 6²)° = 1 --/
def satisfies_equation (k : ℕ+) : Prop :=
  (Real.cos ((k.val ^ 2 + 6 ^ 2 : ℕ) : ℝ) * Real.pi / 180) ^ 2 = 1

/-- Theorem stating that 12 and 18 are the two smallest positive integers satisfying the equation --/
theorem smallest_solutions : 
  (satisfies_equation 12) ∧ 
  (satisfies_equation 18) ∧ 
  (∀ k : ℕ+, k < 12 → ¬(satisfies_equation k)) ∧
  (∀ k : ℕ+, 12 < k → k < 18 → ¬(satisfies_equation k)) :=
sorry

end NUMINAMATH_CALUDE_smallest_solutions_l3862_386271


namespace NUMINAMATH_CALUDE_mikeys_leaves_l3862_386276

/-- The number of leaves that blew away -/
def leaves_blown_away (initial final : ℕ) : ℕ := initial - final

/-- Proof that 244 leaves blew away -/
theorem mikeys_leaves : leaves_blown_away 356 112 = 244 := by
  sorry

end NUMINAMATH_CALUDE_mikeys_leaves_l3862_386276


namespace NUMINAMATH_CALUDE_angle_equality_from_cofunctions_l3862_386295

-- Define a type for angles
variable {α : Type*} [AddCommGroup α]

-- Define a function for co-functions (abstract representation)
variable (cofunc : α → ℝ)

-- State the theorem
theorem angle_equality_from_cofunctions (θ₁ θ₂ : α) :
  (θ₁ = θ₂) ∨ (cofunc θ₁ = cofunc θ₂) → θ₁ = θ₂ := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_from_cofunctions_l3862_386295


namespace NUMINAMATH_CALUDE_dice_roll_sum_l3862_386252

theorem dice_roll_sum (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 360 →
  a + b + c + d ≠ 17 :=
by sorry

end NUMINAMATH_CALUDE_dice_roll_sum_l3862_386252


namespace NUMINAMATH_CALUDE_set_b_forms_triangle_l3862_386275

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function to check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (5, 6, 10) can form a triangle -/
theorem set_b_forms_triangle : can_form_triangle 5 6 10 := by
  sorry


end NUMINAMATH_CALUDE_set_b_forms_triangle_l3862_386275


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l3862_386284

/-- Given two points A and B on the natural logarithm curve, prove that the x-coordinate
    of the point E, where E is the intersection of a horizontal line through C
    (C divides AB in a 1:3 ratio) and the natural logarithm curve, is 16. -/
theorem intersection_x_coordinate (x₁ x₂ x₃ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) 
  (h₃ : x₁ = 2) (h₄ : x₂ = 32) : 
  let y₁ := Real.log x₁
  let y₂ := Real.log x₂
  let yC := (1 / 4 : ℝ) * y₁ + (3 / 4 : ℝ) * y₂
  x₃ = Real.exp yC → x₃ = 16 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l3862_386284


namespace NUMINAMATH_CALUDE_P_inf_zero_no_minimum_l3862_386269

/-- The function P from ℝ² to ℝ -/
def P : ℝ × ℝ → ℝ := fun (x₁, x₂) ↦ x₁^2 + (1 - x₁ * x₂)^2

theorem P_inf_zero_no_minimum :
  (∀ ε > 0, ∃ x : ℝ × ℝ, P x < ε) ∧
  ¬∃ x : ℝ × ℝ, ∀ y : ℝ × ℝ, P x ≤ P y :=
by sorry

end NUMINAMATH_CALUDE_P_inf_zero_no_minimum_l3862_386269


namespace NUMINAMATH_CALUDE_total_eggs_proof_l3862_386274

/-- The total number of eggs used by Molly's employees at the Wafting Pie Company -/
def total_eggs (morning_eggs afternoon_eggs : ℕ) : ℕ :=
  morning_eggs + afternoon_eggs

/-- Proof that the total number of eggs used is 1339 -/
theorem total_eggs_proof (morning_eggs afternoon_eggs : ℕ) 
  (h1 : morning_eggs = 816) 
  (h2 : afternoon_eggs = 523) : 
  total_eggs morning_eggs afternoon_eggs = 1339 := by
  sorry

#eval total_eggs 816 523

end NUMINAMATH_CALUDE_total_eggs_proof_l3862_386274


namespace NUMINAMATH_CALUDE_maintenance_model_correct_l3862_386298

/-- Linear regression model for device maintenance cost --/
structure MaintenanceModel where
  b : ℝ  -- Slope of the regression line
  a : ℝ  -- Y-intercept of the regression line

/-- Conditions for the maintenance cost model --/
class MaintenanceConditions (model : MaintenanceModel) where
  avg_point : 5.4 = 4 * model.b + model.a
  cost_diff : 8 * model.b + model.a - (7 * model.b + model.a) = 1.1

/-- Theorem stating the correctness of the derived model and its prediction --/
theorem maintenance_model_correct (model : MaintenanceModel) 
  [cond : MaintenanceConditions model] : 
  model.b = 0.55 ∧ model.a = 3.2 ∧ 
  (0.55 * 10 + 3.2 : ℝ) = 8.7 := by
  sorry

#check maintenance_model_correct

end NUMINAMATH_CALUDE_maintenance_model_correct_l3862_386298


namespace NUMINAMATH_CALUDE_no_integer_solution_for_equation_l3862_386249

theorem no_integer_solution_for_equation :
  ∀ x y : ℤ, x^2 - 3*y^2 ≠ 17 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_equation_l3862_386249


namespace NUMINAMATH_CALUDE_work_completion_time_l3862_386216

theorem work_completion_time (a_time b_time : ℝ) (a_share : ℝ) : 
  a_time = 10 →
  a_share = 3 / 5 →
  a_share = (1 / a_time) / ((1 / a_time) + (1 / b_time)) →
  b_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3862_386216


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3862_386234

theorem quadratic_function_properties (a b c : ℝ) (h1 : a ≠ 0) 
  (h2 : a^2 + 2*a*c + c^2 < b^2) 
  (h3 : ∀ t : ℝ, a*(t+2)^2 + b*(t+2) + c = a*(-t+2)^2 + b*(-t+2) + c) 
  (h4 : a*(-2)^2 + b*(-2) + c = 2) :
  (∃ axis : ℝ, axis = 2 ∧ 
    ∀ x : ℝ, a*x^2 + b*x + c = a*(2*axis - x)^2 + b*(2*axis - x) + c) ∧ 
  (2/15 < a ∧ a < 2/7) := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3862_386234


namespace NUMINAMATH_CALUDE_quadratic_root_geometric_sequence_l3862_386228

theorem quadratic_root_geometric_sequence (a b c : ℝ) : 
  a ≥ b ∧ b ≥ c ∧ c ≥ 0 →  -- Condition: a ≥ b ≥ c ≥ 0
  (∃ r : ℝ, b = a * r ∧ c = a * r^2) →  -- Condition: a, b, c form a geometric sequence
  (∃! x : ℝ, a * x^2 + b * x + c = 0) →  -- Condition: quadratic has exactly one root
  (∀ x : ℝ, a * x^2 + b * x + c = 0 → x = -1/8) :=  -- Conclusion: the root is -1/8
by sorry

end NUMINAMATH_CALUDE_quadratic_root_geometric_sequence_l3862_386228


namespace NUMINAMATH_CALUDE_monotonic_increasing_interval_l3862_386291

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem monotonic_increasing_interval (ω : ℝ) (h_pos : ω > 0) (h_period : ∀ x : ℝ, f ω (x + π / ω) = f ω x) :
  ∀ k : ℤ, StrictMonoOn (f ω) (Set.Icc (k * π - π / 12) (k * π + 5 * π / 12)) := by sorry

end NUMINAMATH_CALUDE_monotonic_increasing_interval_l3862_386291


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3862_386210

theorem exponent_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3862_386210


namespace NUMINAMATH_CALUDE_probability_of_overlap_l3862_386200

/-- Represents the duration of the entire time frame in minutes -/
def totalDuration : ℝ := 60

/-- Represents the waiting time of the train in minutes -/
def waitingTime : ℝ := 10

/-- Represents the area of the triangle in the graphical representation -/
def triangleArea : ℝ := 50

/-- Calculates the area of the parallelogram in the graphical representation -/
def parallelogramArea : ℝ := totalDuration * waitingTime

/-- Calculates the total area of overlap (favorable outcomes) -/
def overlapArea : ℝ := triangleArea + parallelogramArea

/-- Calculates the total area of all possible outcomes -/
def totalArea : ℝ := totalDuration * totalDuration

/-- Theorem stating the probability of Alex arriving while the train is at the station -/
theorem probability_of_overlap : overlapArea / totalArea = 11 / 72 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_overlap_l3862_386200


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l3862_386204

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n.choose 2 - n

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l3862_386204


namespace NUMINAMATH_CALUDE_proportion_solution_l3862_386257

theorem proportion_solution (n : ℝ) : n / 1.2 = 5 / 8 → n = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l3862_386257


namespace NUMINAMATH_CALUDE_johns_leisure_travel_l3862_386268

/-- Calculates the leisure travel distance for John given his car's efficiency,
    work commute details, and total gas consumption. -/
theorem johns_leisure_travel
  (efficiency : ℝ)  -- Car efficiency in miles per gallon
  (work_distance : ℝ)  -- One-way distance to work in miles
  (work_days : ℕ)  -- Number of work days per week
  (total_gas : ℝ)  -- Total gas used per week in gallons
  (h1 : efficiency = 30)  -- Car efficiency is 30 mpg
  (h2 : work_distance = 20)  -- Distance to work is 20 miles each way
  (h3 : work_days = 5)  -- Works 5 days a week
  (h4 : total_gas = 8)  -- Uses 8 gallons of gas per week
  : ℝ :=
  total_gas * efficiency - 2 * work_distance * work_days

#check johns_leisure_travel

end NUMINAMATH_CALUDE_johns_leisure_travel_l3862_386268


namespace NUMINAMATH_CALUDE_ratio_of_divisors_sums_l3862_386250

def M : ℕ := 75 * 75 * 140 * 343

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 6 := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisors_sums_l3862_386250


namespace NUMINAMATH_CALUDE_initial_cooking_time_is_45_l3862_386299

/-- The recommended cooking time in minutes -/
def recommended_time : ℕ := 5

/-- The remaining cooking time in seconds -/
def remaining_time : ℕ := 255

/-- Conversion factor from minutes to seconds -/
def minutes_to_seconds : ℕ := 60

/-- The initial cooking time in seconds -/
def initial_cooking_time : ℕ := recommended_time * minutes_to_seconds - remaining_time

theorem initial_cooking_time_is_45 : initial_cooking_time = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_cooking_time_is_45_l3862_386299


namespace NUMINAMATH_CALUDE_slide_boys_count_l3862_386244

/-- The number of boys who went down the slide initially -/
def initial_boys : ℕ := 22

/-- The number of additional boys who went down the slide -/
def additional_boys : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_boys : ℕ := initial_boys + additional_boys

theorem slide_boys_count : total_boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_slide_boys_count_l3862_386244


namespace NUMINAMATH_CALUDE_tory_has_six_games_l3862_386285

/-- The number of video games Theresa, Julia, and Tory have. -/
structure VideoGames where
  theresa : ℕ
  julia : ℕ
  tory : ℕ

/-- The conditions given in the problem. -/
def problem_conditions (vg : VideoGames) : Prop :=
  vg.theresa = 3 * vg.julia + 5 ∧
  vg.julia = vg.tory / 3 ∧
  vg.theresa = 11

/-- The theorem stating that Tory has 6 video games. -/
theorem tory_has_six_games (vg : VideoGames) (h : problem_conditions vg) : vg.tory = 6 := by
  sorry

end NUMINAMATH_CALUDE_tory_has_six_games_l3862_386285


namespace NUMINAMATH_CALUDE_angela_age_in_five_years_l3862_386214

/-- Given that Angela is four times as old as Beth, and five years ago the sum of their ages was 45 years, prove that Angela will be 49 years old in five years. -/
theorem angela_age_in_five_years (angela beth : ℕ) 
  (h1 : angela = 4 * beth) 
  (h2 : angela - 5 + beth - 5 = 45) : 
  angela + 5 = 49 := by
  sorry

end NUMINAMATH_CALUDE_angela_age_in_five_years_l3862_386214


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l3862_386211

theorem imaginary_part_of_i_times_one_plus_i (i : ℂ) : 
  i * i = -1 → Complex.im (i * (1 + i)) = 1 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l3862_386211


namespace NUMINAMATH_CALUDE_complex_square_i_positive_l3862_386235

theorem complex_square_i_positive (a : ℝ) 
  (h : (Complex.I * (a + Complex.I)^2).re > 0 ∧ (Complex.I * (a + Complex.I)^2).im = 0) : 
  a = -1 := by sorry

end NUMINAMATH_CALUDE_complex_square_i_positive_l3862_386235


namespace NUMINAMATH_CALUDE_rachel_age_problem_l3862_386217

/-- Rachel's age problem -/
theorem rachel_age_problem (rachel_age : ℕ) (grandfather_age : ℕ) (mother_age : ℕ) (father_age : ℕ) : 
  rachel_age = 12 →
  grandfather_age = 7 * rachel_age →
  mother_age = grandfather_age / 2 →
  father_age = mother_age + 5 →
  father_age + (25 - rachel_age) = 60 :=
by sorry

end NUMINAMATH_CALUDE_rachel_age_problem_l3862_386217


namespace NUMINAMATH_CALUDE_base5_product_correct_l3862_386233

/-- Converts a base 5 number to decimal --/
def toDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 5 --/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The first number in base 5 --/
def num1 : List Nat := [3, 0, 2]

/-- The second number in base 5 --/
def num2 : List Nat := [4, 1]

/-- The expected product in base 5 --/
def expected_product : List Nat := [2, 0, 4, 3]

theorem base5_product_correct :
  toBase5 (toDecimal num1 * toDecimal num2) = expected_product := by
  sorry

end NUMINAMATH_CALUDE_base5_product_correct_l3862_386233


namespace NUMINAMATH_CALUDE_cruise_liner_passengers_l3862_386266

theorem cruise_liner_passengers : ∃ n : ℕ, 
  (250 ≤ n ∧ n ≤ 400) ∧ 
  (∃ r : ℕ, n = 15 * r + 7) ∧
  (∃ s : ℕ, n = 25 * s - 8) ∧
  (n = 292 ∨ n = 367) := by
sorry

end NUMINAMATH_CALUDE_cruise_liner_passengers_l3862_386266


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3862_386289

theorem inequality_equivalence (x : ℝ) : 
  (|x + 3| + |1 - x|) / (x + 2016) < 1 ↔ x < -2016 ∨ (-1009 < x ∧ x < 1007) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3862_386289


namespace NUMINAMATH_CALUDE_power_of_256_three_fourths_l3862_386287

theorem power_of_256_three_fourths : (256 : ℝ) ^ (3/4) = 64 := by sorry

end NUMINAMATH_CALUDE_power_of_256_three_fourths_l3862_386287


namespace NUMINAMATH_CALUDE_intersection_M_N_l3862_386219

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 5}
def N : Set ℝ := {x | x * (x - 4) > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | (-1 < x ∧ x < 0) ∨ (4 < x ∧ x < 5)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3862_386219


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l3862_386273

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A convex nonagon has 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l3862_386273


namespace NUMINAMATH_CALUDE_empty_set_problem_l3862_386236

-- Define the sets
def set_A : Set ℝ := {x | x^2 - 4 = 0}
def set_B : Set ℝ := {x | x > 9 ∨ x < 3}
def set_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 0}
def set_D : Set ℝ := {x | x > 9 ∧ x < 3}

-- Theorem statement
theorem empty_set_problem :
  (set_A.Nonempty) ∧
  (set_B.Nonempty) ∧
  (set_C.Nonempty) ∧
  (set_D = ∅) :=
sorry

end NUMINAMATH_CALUDE_empty_set_problem_l3862_386236


namespace NUMINAMATH_CALUDE_sine_shifted_is_even_l3862_386254

/-- A function that reaches its maximum at x = 1 -/
def reaches_max_at_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f x ≤ f 1

/-- Definition of an even function -/
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- Main theorem -/
theorem sine_shifted_is_even
    (A ω φ : ℝ)
    (hA : A > 0)
    (hω : ω > 0)
    (h_max : reaches_max_at_one (fun x ↦ A * Real.sin (ω * x + φ))) :
    is_even (fun x ↦ A * Real.sin (ω * (x + 1) + φ)) := by
  sorry

end NUMINAMATH_CALUDE_sine_shifted_is_even_l3862_386254


namespace NUMINAMATH_CALUDE_correct_quotient_proof_l3862_386213

theorem correct_quotient_proof (D : ℕ) (h1 : D - 1000 = 1200 * 4900) : D / 2100 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_correct_quotient_proof_l3862_386213


namespace NUMINAMATH_CALUDE_smallest_disk_not_always_circumcircle_l3862_386220

/-- Three noncollinear points in the plane -/
structure ThreePoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  noncollinear : A ≠ B ∧ B ≠ C ∧ A ≠ C

/-- The radius of the smallest disk containing three points -/
def smallest_disk_radius (p : ThreePoints) : ℝ :=
  sorry

/-- The radius of the circumcircle of three points -/
def circumcircle_radius (p : ThreePoints) : ℝ :=
  sorry

/-- Theorem stating that the smallest disk is not always the circumcircle -/
theorem smallest_disk_not_always_circumcircle :
  ∃ p : ThreePoints, smallest_disk_radius p < circumcircle_radius p :=
sorry

end NUMINAMATH_CALUDE_smallest_disk_not_always_circumcircle_l3862_386220


namespace NUMINAMATH_CALUDE_quadratic_roots_isosceles_triangle_l3862_386212

theorem quadratic_roots_isosceles_triangle (b : ℝ) (α β : ℝ) :
  (∀ x, x^2 + b*x + 1 = 0 ↔ x = α ∨ x = β) →
  α > β →
  (α^2 + β^2 = 3*α - 3*β ∧ α^2 + β^2 = α*β) ∨
  (α^2 + β^2 = 3*α - 3*β ∧ 3*α - 3*β = α*β) ∨
  (3*α - 3*β = α*β ∧ α*β = α^2 + β^2) →
  b = Real.sqrt 5 ∨ b = -Real.sqrt 5 ∨ b = Real.sqrt 8 ∨ b = -Real.sqrt 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_isosceles_triangle_l3862_386212


namespace NUMINAMATH_CALUDE_inverse_sum_mod_25_l3862_386251

theorem inverse_sum_mod_25 :
  ∃ (a b c : ℤ), (7 * a) % 25 = 1 ∧ 
                 (7 * b) % 25 = a % 25 ∧ 
                 (7 * c) % 25 = b % 25 ∧ 
                 (a + b + c) % 25 = 9 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_mod_25_l3862_386251


namespace NUMINAMATH_CALUDE_combined_mix_dried_fruit_percentage_l3862_386218

/-- Represents a trail mix composition -/
structure TrailMix where
  nuts : ℚ
  dried_fruit : ℚ
  chocolate_chips : ℚ
  composition_sum : nuts + dried_fruit + chocolate_chips = 1

/-- The combined trail mix from two equal portions -/
def combined_mix (mix1 mix2 : TrailMix) : TrailMix :=
  { nuts := (mix1.nuts + mix2.nuts) / 2,
    dried_fruit := (mix1.dried_fruit + mix2.dried_fruit) / 2,
    chocolate_chips := (mix1.chocolate_chips + mix2.chocolate_chips) / 2,
    composition_sum := by sorry }

theorem combined_mix_dried_fruit_percentage
  (sue_mix : TrailMix)
  (jane_mix : TrailMix)
  (h_sue_nuts : sue_mix.nuts = 3/10)
  (h_sue_dried_fruit : sue_mix.dried_fruit = 7/10)
  (h_jane_nuts : jane_mix.nuts = 6/10)
  (h_jane_chocolate : jane_mix.chocolate_chips = 4/10)
  (h_combined_nuts : (combined_mix sue_mix jane_mix).nuts = 45/100) :
  (combined_mix sue_mix jane_mix).dried_fruit = 35/100 := by
sorry

end NUMINAMATH_CALUDE_combined_mix_dried_fruit_percentage_l3862_386218


namespace NUMINAMATH_CALUDE_puzzle_missing_pieces_l3862_386259

/-- Calculates the number of missing puzzle pieces. -/
def missing_pieces (total : ℕ) (border : ℕ) (trevor : ℕ) (joe_multiplier : ℕ) : ℕ :=
  total - (border + trevor + joe_multiplier * trevor)

/-- Proves that the number of missing puzzle pieces is 5. -/
theorem puzzle_missing_pieces :
  missing_pieces 500 75 105 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_puzzle_missing_pieces_l3862_386259


namespace NUMINAMATH_CALUDE_inequality_solution_l3862_386241

/-- Given that the solution of the inequality 2x^2 - 6x + 4 < 0 is 1 < x < b, prove that b = 2 -/
theorem inequality_solution (b : ℝ) 
  (h : ∀ x : ℝ, 1 < x ∧ x < b ↔ 2 * x^2 - 6 * x + 4 < 0) : 
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3862_386241


namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_27_and_n_plus_3_l3862_386227

theorem gcd_n_cube_plus_27_and_n_plus_3 (n : ℕ) (h : n > 9) :
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_27_and_n_plus_3_l3862_386227


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l3862_386258

theorem quadratic_roots_properties (r1 r2 : ℝ) : 
  r1 ≠ r2 → 
  r1^2 - 5*r1 + 6 = 0 → 
  r2^2 - 5*r2 + 6 = 0 → 
  (|r1 + r2| ≤ 6) ∧ 
  (|r1 * r2| ≤ 3 ∨ |r1 * r2| ≥ 8) ∧ 
  (r1 ≥ 0 ∨ r2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l3862_386258


namespace NUMINAMATH_CALUDE_problem_statement_l3862_386232

theorem problem_statement (a b : ℝ) : 
  ({a, 1, b/a} : Set ℝ) = {a + b, 0, a^2} → a^2016 + b^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3862_386232


namespace NUMINAMATH_CALUDE_equation_solution_l3862_386205

theorem equation_solution : ∃! x : ℝ, x + ((2 / 3 * 3 / 8) + 4) - 8 / 16 = 4.25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3862_386205


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3862_386242

theorem sqrt_equation_solution :
  ∀ y : ℚ, (Real.sqrt (8 * y) / Real.sqrt (6 * (y - 2)) = 3) → y = 54 / 23 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3862_386242


namespace NUMINAMATH_CALUDE_transform_458_to_14_l3862_386263

def double (n : ℕ) : ℕ := 2 * n

def eraseLast (n : ℕ) : ℕ := n / 10

inductive Operation
| Double
| EraseLast

def applyOperation (op : Operation) (n : ℕ) : ℕ :=
  match op with
  | Operation.Double => double n
  | Operation.EraseLast => eraseLast n

def applyOperations (ops : List Operation) (start : ℕ) : ℕ :=
  ops.foldl (fun n op => applyOperation op n) start

theorem transform_458_to_14 :
  ∃ (ops : List Operation), applyOperations ops 458 = 14 :=
sorry

end NUMINAMATH_CALUDE_transform_458_to_14_l3862_386263


namespace NUMINAMATH_CALUDE_function_sum_at_one_l3862_386255

-- Define f and g as functions from ℝ to ℝ
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem function_sum_at_one 
  (h1 : is_even f) 
  (h2 : is_odd g) 
  (h3 : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  f 1 + g 1 = 1 := by
sorry

end NUMINAMATH_CALUDE_function_sum_at_one_l3862_386255


namespace NUMINAMATH_CALUDE_fraction_representation_of_naturals_l3862_386292

theorem fraction_representation_of_naturals (n : ℕ) :
  ∃ x y : ℕ, n = x^3 / y^4 :=
sorry

end NUMINAMATH_CALUDE_fraction_representation_of_naturals_l3862_386292


namespace NUMINAMATH_CALUDE_smallest_c_value_l3862_386256

/-- Given a cosine function y = a cos(bx + c) with positive constants a, b, c,
    and maximum at x = 1, the smallest possible value of c is 0. -/
theorem smallest_c_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
    (h4 : ∀ x : ℝ, a * Real.cos (b * x + c) ≤ a * Real.cos (b * 1 + c)) :
    ∃ c' : ℝ, c' ≥ 0 ∧ c' ≤ c ∧ ∀ c'' : ℝ, c'' ≥ 0 → c'' ≤ c → c' ≤ c'' := by
  sorry

end NUMINAMATH_CALUDE_smallest_c_value_l3862_386256
