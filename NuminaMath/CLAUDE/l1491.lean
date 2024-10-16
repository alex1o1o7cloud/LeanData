import Mathlib

namespace NUMINAMATH_CALUDE_triangle_equilateral_from_cosine_product_l1491_149113

theorem triangle_equilateral_from_cosine_product (A B C : ℝ) 
  (triangle_condition : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (angle_sum : A + B + C = π) 
  (cosine_product : Real.cos (A - B) * Real.cos (B - C) * Real.cos (C - A) = 1) : 
  A = B ∧ B = C := by
  sorry

end NUMINAMATH_CALUDE_triangle_equilateral_from_cosine_product_l1491_149113


namespace NUMINAMATH_CALUDE_inequality_theorem_largest_constant_l1491_149103

theorem inequality_theorem (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 :=
by sorry

theorem largest_constant :
  ∀ m : ℝ, (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
    Real.sqrt (e / (a + b + c + d)) > m) → m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_theorem_largest_constant_l1491_149103


namespace NUMINAMATH_CALUDE_mobile_price_change_l1491_149102

theorem mobile_price_change (initial_price : ℝ) (decrease_percent : ℝ) : 
  (initial_price * 1.4 * (1 - decrease_percent / 100) = initial_price * 1.18999999999999993) →
  decrease_percent = 15 := by
sorry

end NUMINAMATH_CALUDE_mobile_price_change_l1491_149102


namespace NUMINAMATH_CALUDE_calculation_proof_l1491_149124

theorem calculation_proof : 
  71 * ((5 + 2/7) - (6 + 1/3)) / ((3 + 1/2) + (2 + 1/5)) = -(13 + 37/1197) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1491_149124


namespace NUMINAMATH_CALUDE_sixth_salary_proof_l1491_149153

theorem sixth_salary_proof (known_salaries : List ℝ) 
  (h1 : known_salaries = [1000, 2500, 3100, 3650, 2000])
  (h2 : (known_salaries.sum + x) / 6 = 2291.67) : x = 1500 := by
  sorry

end NUMINAMATH_CALUDE_sixth_salary_proof_l1491_149153


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_eq_neg_two_implies_result_l1491_149196

theorem tan_pi_minus_alpha_eq_neg_two_implies_result (α : ℝ) 
  (h : Real.tan (π - α) = -2) : 
  1 / (Real.cos (2 * α) + Real.cos α ^ 2) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_eq_neg_two_implies_result_l1491_149196


namespace NUMINAMATH_CALUDE_negation_true_l1491_149171

theorem negation_true : 
  ¬(∀ a : ℝ, a ≤ 3 → a^2 < 9) ↔ True :=
by sorry

end NUMINAMATH_CALUDE_negation_true_l1491_149171


namespace NUMINAMATH_CALUDE_rectangle_area_l1491_149160

theorem rectangle_area (b : ℝ) (h1 : b > 0) : 
  let l := 3 * b
  let perimeter := 2 * (l + b)
  perimeter = 112 → l * b = 588 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l1491_149160


namespace NUMINAMATH_CALUDE_man_work_time_l1491_149151

/-- The time taken by a man to complete a work given the following conditions:
    - A man, a woman, and a boy together complete the work in 3 days
    - A woman alone can do the work in 6 days
    - A boy alone can do the work in 18 days -/
theorem man_work_time (work : ℝ) (man_rate woman_rate boy_rate : ℝ) :
  work > 0 ∧
  man_rate > 0 ∧ woman_rate > 0 ∧ boy_rate > 0 ∧
  man_rate + woman_rate + boy_rate = work / 3 ∧
  woman_rate = work / 6 ∧
  boy_rate = work / 18 →
  work / man_rate = 9 := by
  sorry

#check man_work_time

end NUMINAMATH_CALUDE_man_work_time_l1491_149151


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l1491_149130

theorem quadratic_root_problem (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x - 7 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 + m * y - 7 = 0 ∧ y = -7/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l1491_149130


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1491_149150

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a > b ∧ b > 0 → a^2 > b^2) ∧
  ∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1491_149150


namespace NUMINAMATH_CALUDE_relay_race_time_difference_l1491_149152

def apple_distance : ℝ := 24
def apple_speed : ℝ := 3
def mac_distance : ℝ := 28
def mac_speed : ℝ := 4
def orange_distance : ℝ := 32
def orange_speed : ℝ := 5

def minutes_per_hour : ℝ := 60

theorem relay_race_time_difference :
  (apple_distance / apple_speed + mac_distance / mac_speed) * minutes_per_hour -
  (orange_distance / orange_speed * minutes_per_hour) = 516 := by
sorry

end NUMINAMATH_CALUDE_relay_race_time_difference_l1491_149152


namespace NUMINAMATH_CALUDE_missing_interior_angle_l1491_149112

theorem missing_interior_angle (n : ℕ) (sum_without_one : ℝ) (missing_angle : ℝ) :
  n = 18 →
  sum_without_one = 2750 →
  (n - 2) * 180 = sum_without_one + missing_angle →
  missing_angle = 130 :=
by sorry

end NUMINAMATH_CALUDE_missing_interior_angle_l1491_149112


namespace NUMINAMATH_CALUDE_solution_set_a_gt_1_solution_set_a_eq_1_solution_set_a_lt_1_a_range_subset_l1491_149117

-- Define the inequality function
def f (a x : ℝ) : ℝ := (a * x - (a - 2)) * (x + 1)

-- Define the solution set P
def P (a : ℝ) : Set ℝ := {x | f a x > 0}

-- Theorem for the solution set when a > 1
theorem solution_set_a_gt_1 (a : ℝ) (h : a > 1) :
  P a = {x | x < -1 ∨ x > (a - 2) / a} := by sorry

-- Theorem for the solution set when a = 1
theorem solution_set_a_eq_1 :
  P 1 = {x | x ≠ -1} := by sorry

-- Theorem for the solution set when 0 < a < 1
theorem solution_set_a_lt_1 (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  P a = {x | x < (a - 2) / a ∨ x > -1} := by sorry

-- Theorem for the range of a when {x | -3 < x < -1} ⊆ P
theorem a_range_subset (a : ℝ) (h : {x : ℝ | -3 < x ∧ x < -1} ⊆ P a) :
  a ∈ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_a_gt_1_solution_set_a_eq_1_solution_set_a_lt_1_a_range_subset_l1491_149117


namespace NUMINAMATH_CALUDE_average_yield_is_100_l1491_149161

/-- Calculates the average yield per tree given the number of trees and their yields. -/
def averageYield (x : ℕ) : ℚ :=
  let trees1 := x + 2
  let trees2 := x
  let trees3 := x - 2
  let yield1 := 30
  let yield2 := 120
  let yield3 := 180
  let totalTrees := trees1 + trees2 + trees3
  let totalNuts := trees1 * yield1 + trees2 * yield2 + trees3 * yield3
  totalNuts / totalTrees

/-- Theorem stating that the average yield per tree is 100 when x = 10. -/
theorem average_yield_is_100 : averageYield 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_average_yield_is_100_l1491_149161


namespace NUMINAMATH_CALUDE_shadow_height_ratio_michaels_height_l1491_149162

/-- Given a flagpole and a person casting shadows at the same time, 
    calculate the person's height using the ratio of heights to shadows. -/
theorem shadow_height_ratio 
  (h₁ : ℝ) (s₁ : ℝ) (s₂ : ℝ) 
  (h₁_pos : h₁ > 0) (s₁_pos : s₁ > 0) (s₂_pos : s₂ > 0) :
  ∃ h₂ : ℝ, h₂ = (h₁ * s₂) / s₁ := by
  sorry

/-- Michael's height calculation based on the shadow ratio -/
theorem michaels_height 
  (flagpole_height : ℝ) (flagpole_shadow : ℝ) (michael_shadow : ℝ)
  (flagpole_height_eq : flagpole_height = 50)
  (flagpole_shadow_eq : flagpole_shadow = 25)
  (michael_shadow_eq : michael_shadow = 5) :
  ∃ michael_height : ℝ, michael_height = 10 := by
  sorry

end NUMINAMATH_CALUDE_shadow_height_ratio_michaels_height_l1491_149162


namespace NUMINAMATH_CALUDE_triangle_formation_l1491_149135

/-- Checks if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem triangle_formation :
  can_form_triangle 8 6 3 ∧
  ¬can_form_triangle 2 2 4 ∧
  ¬can_form_triangle 2 6 3 ∧
  ¬can_form_triangle 11 4 6 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l1491_149135


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l1491_149128

theorem sum_of_roots_eq_fourteen : ∀ x₁ x₂ : ℝ, (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 → x₁ + x₂ = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l1491_149128


namespace NUMINAMATH_CALUDE_factorial_expression_equals_1584_l1491_149195

theorem factorial_expression_equals_1584 :
  (Nat.factorial 13 - Nat.factorial 12) / Nat.factorial 10 = 1584 := by
  sorry

end NUMINAMATH_CALUDE_factorial_expression_equals_1584_l1491_149195


namespace NUMINAMATH_CALUDE_opposite_unit_vector_l1491_149182

def a : ℝ × ℝ := (4, 3)

theorem opposite_unit_vector (a : ℝ × ℝ) :
  let magnitude := Real.sqrt (a.1^2 + a.2^2)
  let opposite_unit := (-a.1 / magnitude, -a.2 / magnitude)
  opposite_unit = (-4/5, -3/5) ∧
  opposite_unit.1^2 + opposite_unit.2^2 = 1 ∧
  a.1 * opposite_unit.1 + a.2 * opposite_unit.2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_opposite_unit_vector_l1491_149182


namespace NUMINAMATH_CALUDE_distribution_ratio_l1491_149174

def num_balls : ℕ := 20
def num_bins : ℕ := 5

def distribution_A : List ℕ := [3, 5, 4, 4, 4]
def distribution_B : List ℕ := [4, 4, 4, 4, 4]

def count_distributions (n : ℕ) (k : ℕ) (dist : List ℕ) : ℕ :=
  sorry

theorem distribution_ratio :
  (count_distributions num_balls num_bins distribution_A) /
  (count_distributions num_balls num_bins distribution_B) = 4 :=
by sorry

end NUMINAMATH_CALUDE_distribution_ratio_l1491_149174


namespace NUMINAMATH_CALUDE_union_equality_implies_m_value_l1491_149121

def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

theorem union_equality_implies_m_value (m : ℝ) :
  A m ∪ B m = A m → m = 0 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_implies_m_value_l1491_149121


namespace NUMINAMATH_CALUDE_sandy_watermelons_count_l1491_149142

/-- The number of watermelons Jason grew -/
def jason_watermelons : ℕ := 37

/-- The total number of watermelons grown by Jason and Sandy -/
def total_watermelons : ℕ := 48

/-- The number of watermelons Sandy grew -/
def sandy_watermelons : ℕ := total_watermelons - jason_watermelons

theorem sandy_watermelons_count : sandy_watermelons = 11 := by
  sorry

end NUMINAMATH_CALUDE_sandy_watermelons_count_l1491_149142


namespace NUMINAMATH_CALUDE_empty_solution_set_range_l1491_149118

theorem empty_solution_set_range (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 < 0) ↔ (-2 < a ∧ a ≤ 6/5) :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_range_l1491_149118


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l1491_149136

/-- If 9x^2 - 18x + a is the square of a binomial, then a = 9 -/
theorem perfect_square_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l1491_149136


namespace NUMINAMATH_CALUDE_complete_factorization_l1491_149172

theorem complete_factorization (x : ℝ) : 
  x^6 - 64 = (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
sorry

end NUMINAMATH_CALUDE_complete_factorization_l1491_149172


namespace NUMINAMATH_CALUDE_laura_cycling_distance_l1491_149189

def base_7_to_10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

theorem laura_cycling_distance : base_7_to_10 3 5 1 6 = 1287 := by
  sorry

end NUMINAMATH_CALUDE_laura_cycling_distance_l1491_149189


namespace NUMINAMATH_CALUDE_sum_of_digits_in_19_minutes_l1491_149175

def sum_of_digits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def time_to_minutes (hours minutes : Nat) : Nat :=
  (hours % 12) * 60 + minutes

def minutes_to_time (total_minutes : Nat) : (Nat × Nat) :=
  ((total_minutes / 60) % 12, total_minutes % 60)

theorem sum_of_digits_in_19_minutes 
  (current_hours current_minutes : Nat) 
  (h_valid_time : current_hours < 12 ∧ current_minutes < 60) 
  (h_sum_condition : 
    let (prev_hours, prev_minutes) := minutes_to_time (time_to_minutes current_hours current_minutes - 19)
    sum_of_digits prev_hours + sum_of_digits prev_minutes = 
      sum_of_digits current_hours + sum_of_digits current_minutes - 2) :
  let (future_hours, future_minutes) := minutes_to_time (time_to_minutes current_hours current_minutes + 19)
  sum_of_digits future_hours + sum_of_digits future_minutes = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_in_19_minutes_l1491_149175


namespace NUMINAMATH_CALUDE_four_statements_incorrect_l1491_149139

/-- The alternating sum from 1 to 2002 -/
def alternating_sum : ℕ → ℤ
  | 0 => 0
  | n + 1 => if n % 2 = 0 then alternating_sum n + (n + 1) else alternating_sum n - (n + 1)

/-- The sum of n consecutive natural numbers starting from k -/
def consec_sum (n k : ℕ) : ℕ := n * (2 * k + n - 1) / 2

theorem four_statements_incorrect : 
  (¬ Even (alternating_sum 2002)) ∧ 
  (∃ (a b c : ℤ), Odd a ∧ Odd b ∧ Odd c ∧ (a * b) * (c - b) ≠ a) ∧
  (¬ Even (consec_sum 2002 1)) ∧
  (¬ ∃ (a b : ℤ), (a + b) * (a - b) = 2002) :=
by sorry

end NUMINAMATH_CALUDE_four_statements_incorrect_l1491_149139


namespace NUMINAMATH_CALUDE_total_amount_paid_l1491_149123

def grape_quantity : ℕ := 8
def grape_price : ℚ := 70
def mango_quantity : ℕ := 8
def mango_price : ℚ := 55
def orange_quantity : ℕ := 5
def orange_price : ℚ := 40
def apple_quantity : ℕ := 10
def apple_price : ℚ := 30
def grape_discount : ℚ := 0.1
def mango_tax : ℚ := 0.05

theorem total_amount_paid : 
  (grape_quantity * grape_price * (1 - grape_discount) + 
   mango_quantity * mango_price * (1 + mango_tax) + 
   orange_quantity * orange_price + 
   apple_quantity * apple_price) = 1466 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_paid_l1491_149123


namespace NUMINAMATH_CALUDE_coopers_savings_l1491_149116

/-- Calculates the total savings given a daily savings amount and number of days -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Theorem: Cooper's savings after one year -/
theorem coopers_savings :
  totalSavings 34 365 = 12410 := by
  sorry

end NUMINAMATH_CALUDE_coopers_savings_l1491_149116


namespace NUMINAMATH_CALUDE_tetrahedron_smallest_faces_l1491_149194

/-- Represents the number of faces in a geometric shape. -/
def faces (shape : String) : ℕ :=
  match shape with
  | "Tetrahedron" => 4
  | "Quadrangular pyramid" => 5
  | "Triangular prism" => 5
  | "Triangular pyramid" => 4
  | _ => 0

/-- The list of shapes we're considering. -/
def shapes : List String :=
  ["Tetrahedron", "Quadrangular pyramid", "Triangular prism", "Triangular pyramid"]

/-- Theorem stating that the tetrahedron has the smallest number of faces among the given shapes. -/
theorem tetrahedron_smallest_faces :
    ∀ shape ∈ shapes, faces "Tetrahedron" ≤ faces shape := by
  sorry

#check tetrahedron_smallest_faces

end NUMINAMATH_CALUDE_tetrahedron_smallest_faces_l1491_149194


namespace NUMINAMATH_CALUDE_sum_of_powers_l1491_149186

theorem sum_of_powers (w : ℂ) (hw : w^2 - w + 1 = 0) :
  w^97 + w^98 + w^99 + w^100 + w^101 + w^102 = -2 + 2*w := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l1491_149186


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1491_149192

theorem opposite_of_2023 : 
  -(2023 : ℤ) = -2023 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1491_149192


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l1491_149145

theorem unique_solution_trigonometric_equation :
  ∃! (n k m : ℕ), 1 ≤ n ∧ n ≤ 5 ∧
                  1 ≤ k ∧ k ≤ 5 ∧
                  1 ≤ m ∧ m ≤ 5 ∧
                  (Real.sin (π * n / 12) * Real.sin (π * k / 12) * Real.sin (π * m / 12) = 1 / 8) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l1491_149145


namespace NUMINAMATH_CALUDE_monotonic_implies_not_even_but_not_conversely_l1491_149104

-- Define the properties of a function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsMonotonic (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y → f x ≤ f y

-- State the theorem
theorem monotonic_implies_not_even_but_not_conversely :
  (∃ f : ℝ → ℝ, IsMonotonic f → ¬IsEven f) ∧
  (∃ g : ℝ → ℝ, ¬IsEven g ∧ ¬IsMonotonic g) :=
sorry

end NUMINAMATH_CALUDE_monotonic_implies_not_even_but_not_conversely_l1491_149104


namespace NUMINAMATH_CALUDE_sum_of_distances_forms_ellipse_l1491_149157

/-- Definition of an ellipse based on the sum of distances to two foci -/
def is_ellipse (F₁ F₂ : ℝ × ℝ) (a : ℝ) (S : Set (ℝ × ℝ)) : Prop :=
  ∃ c : ℝ, F₁ = (c, 0) ∧ F₂ = (-c, 0) ∧ a > c ∧ c > 0 ∧
  S = {p : ℝ × ℝ | let (x, y) := p; Real.sqrt ((x - c)^2 + y^2) + Real.sqrt ((x + c)^2 + y^2) = 2 * a}

/-- Theorem: The set of points satisfying the sum of distances to two foci is an ellipse -/
theorem sum_of_distances_forms_ellipse (F₁ F₂ : ℝ × ℝ) (a : ℝ) (S : Set (ℝ × ℝ)) 
  (h : is_ellipse F₁ F₂ a S) : 
  ∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧ S = {p : ℝ × ℝ | let (x, y) := p; x^2 / a'^2 + y^2 / b'^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_forms_ellipse_l1491_149157


namespace NUMINAMATH_CALUDE_wheels_in_garage_is_39_l1491_149137

/-- The number of wheels in a garage with various vehicles and items -/
def total_wheels_in_garage (
  num_cars : ℕ)
  (num_lawnmowers : ℕ)
  (num_bicycles : ℕ)
  (num_tricycles : ℕ)
  (num_unicycles : ℕ)
  (num_skateboards : ℕ)
  (num_wheelbarrows : ℕ)
  (num_four_wheeled_wagons : ℕ)
  (num_two_wheeled_dollies : ℕ)
  (num_four_wheeled_shopping_carts : ℕ)
  (num_two_wheeled_scooters : ℕ) : ℕ :=
  num_cars * 4 +
  num_lawnmowers * 4 +
  num_bicycles * 2 +
  num_tricycles * 3 +
  num_unicycles * 1 +
  num_skateboards * 4 +
  num_wheelbarrows * 1 +
  num_four_wheeled_wagons * 4 +
  num_two_wheeled_dollies * 2 +
  num_four_wheeled_shopping_carts * 4 +
  num_two_wheeled_scooters * 2

/-- Theorem stating that the total number of wheels in the garage is 39 -/
theorem wheels_in_garage_is_39 :
  total_wheels_in_garage 2 1 3 1 1 1 1 1 1 1 1 = 39 := by
  sorry

end NUMINAMATH_CALUDE_wheels_in_garage_is_39_l1491_149137


namespace NUMINAMATH_CALUDE_polynomial_multiplication_l1491_149169

theorem polynomial_multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_l1491_149169


namespace NUMINAMATH_CALUDE_max_sum_ITEST_l1491_149148

theorem max_sum_ITEST (I T E S : ℕ+) : 
  I ≠ T ∧ I ≠ E ∧ I ≠ S ∧ T ≠ E ∧ T ≠ S ∧ E ≠ S →
  I * T * E * S * T = 2006 →
  (∀ (I' T' E' S' : ℕ+), 
    I' ≠ T' ∧ I' ≠ E' ∧ I' ≠ S' ∧ T' ≠ E' ∧ T' ≠ S' ∧ E' ≠ S' →
    I' * T' * E' * S' * T' = 2006 →
    I + T + E + S + T + 2006 ≥ I' + T' + E' + S' + T' + 2006) →
  I + T + E + S + T + 2006 = 2086 := by
sorry

end NUMINAMATH_CALUDE_max_sum_ITEST_l1491_149148


namespace NUMINAMATH_CALUDE_product_evaluation_l1491_149120

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = 5040 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l1491_149120


namespace NUMINAMATH_CALUDE_operations_for_106_triangles_l1491_149179

/-- The number of triangles after n operations -/
def num_triangles (n : ℕ) : ℕ := 4 + 3 * (n - 1)

theorem operations_for_106_triangles :
  ∃ n : ℕ, n > 0 ∧ num_triangles n = 106 ∧ n = 35 := by sorry

end NUMINAMATH_CALUDE_operations_for_106_triangles_l1491_149179


namespace NUMINAMATH_CALUDE_larger_solid_volume_is_seven_halves_l1491_149101

-- Define the rectangular prism
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the function to calculate the volume of the larger solid
def largerSolidVolume (prism : RectangularPrism) (plane : Plane3D) : ℝ := sorry

-- Theorem statement
theorem larger_solid_volume_is_seven_halves :
  let prism := RectangularPrism.mk 2 3 1
  let A := Point3D.mk 0 0 0
  let B := Point3D.mk 3 0 0
  let E := Point3D.mk 0 3 0
  let F := Point3D.mk 0 3 1
  let G := Point3D.mk 3 3 1
  let P := Point3D.mk 1.5 (3/2) (1/2)
  let Q := Point3D.mk 0 (3/2) (1/2)
  let plane := Plane3D.mk 1 1 1 0  -- Placeholder plane equation
  largerSolidVolume prism plane = 7/2 := by
  sorry


end NUMINAMATH_CALUDE_larger_solid_volume_is_seven_halves_l1491_149101


namespace NUMINAMATH_CALUDE_amoeba_population_after_five_days_l1491_149110

/-- The number of amoebas after n days, given an initial population and daily split rate --/
def amoeba_population (initial_population : ℕ) (split_rate : ℕ) (days : ℕ) : ℕ :=
  initial_population * split_rate ^ days

/-- The theorem stating that after 5 days, the amoeba population will be 486 --/
theorem amoeba_population_after_five_days :
  amoeba_population 2 3 5 = 486 := by
  sorry

#eval amoeba_population 2 3 5

end NUMINAMATH_CALUDE_amoeba_population_after_five_days_l1491_149110


namespace NUMINAMATH_CALUDE_numerical_expression_problem_l1491_149133

theorem numerical_expression_problem :
  ∃ (A B C D : ℕ),
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
    20180 ≤ 2018 * 10 + A ∧ 2018 * 10 + A < 20190 ∧
    100 ≤ B * 100 + C * 10 + D ∧ B * 100 + C * 10 + D < 1000 ∧
    (2018 * 10 + A) / (B * 100 + C * 10 + D) = 10 * A + A ∧
    A = 5 ∧ B = 3 ∧ C = 6 ∧ D = 7 :=
by sorry

end NUMINAMATH_CALUDE_numerical_expression_problem_l1491_149133


namespace NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1491_149185

theorem ratio_of_sum_and_difference (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y = 6 * (x - y)) : x / y = 7 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_sum_and_difference_l1491_149185


namespace NUMINAMATH_CALUDE_fish_per_multicolor_duck_l1491_149188

theorem fish_per_multicolor_duck 
  (white_fish_ratio : ℕ) 
  (black_fish_ratio : ℕ) 
  (white_ducks : ℕ) 
  (black_ducks : ℕ) 
  (multicolor_ducks : ℕ) 
  (total_fish : ℕ) 
  (h1 : white_fish_ratio = 5)
  (h2 : black_fish_ratio = 10)
  (h3 : white_ducks = 3)
  (h4 : black_ducks = 7)
  (h5 : multicolor_ducks = 6)
  (h6 : total_fish = 157) :
  (total_fish - (white_fish_ratio * white_ducks + black_fish_ratio * black_ducks)) / multicolor_ducks = 12 := by
sorry

end NUMINAMATH_CALUDE_fish_per_multicolor_duck_l1491_149188


namespace NUMINAMATH_CALUDE_plane_equation_theorem_l1491_149129

/-- The equation of a plane given its normal vector and a point on the plane -/
def plane_equation (normal : ℝ × ℝ × ℝ) (point : ℝ × ℝ × ℝ) : ℤ × ℤ × ℤ × ℤ :=
  sorry

/-- Check if the first coefficient is positive -/
def first_coeff_positive (coeffs : ℤ × ℤ × ℤ × ℤ) : Prop :=
  sorry

/-- Calculate the GCD of the absolute values of all coefficients -/
def gcd_of_coeffs (coeffs : ℤ × ℤ × ℤ × ℤ) : ℕ :=
  sorry

theorem plane_equation_theorem :
  let normal : ℝ × ℝ × ℝ := (10, -5, 6)
  let point : ℝ × ℝ × ℝ := (10, -5, 6)
  let coeffs := plane_equation normal point
  first_coeff_positive coeffs ∧ gcd_of_coeffs coeffs = 1 ∧ coeffs = (10, -5, 6, -161) :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_theorem_l1491_149129


namespace NUMINAMATH_CALUDE_infinitely_many_composites_in_sequence_l1491_149168

theorem infinitely_many_composites_in_sequence :
  ∀ N : ℕ, ∃ k : ℕ, k > N ∧ 
    (∃ m : ℕ, (10^(16*k+8) - 1) / 3 = 17 * m) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_composites_in_sequence_l1491_149168


namespace NUMINAMATH_CALUDE_bug_crawl_tiles_l1491_149166

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledFloor where
  length : Nat
  width : Nat
  tileSize : Nat
  totalTiles : Nat

/-- Calculates the number of tiles a bug crosses when crawling diagonally across the floor. -/
def tilesTraversed (floor : TiledFloor) : Nat :=
  floor.length + floor.width - 1

/-- Theorem stating the number of tiles crossed by a bug on a specific floor. -/
theorem bug_crawl_tiles (floor : TiledFloor) 
  (h1 : floor.length = 17)
  (h2 : floor.width = 10)
  (h3 : floor.tileSize = 1)
  (h4 : floor.totalTiles = 170) :
  tilesTraversed floor = 26 := by
  sorry

#eval tilesTraversed { length := 17, width := 10, tileSize := 1, totalTiles := 170 }

end NUMINAMATH_CALUDE_bug_crawl_tiles_l1491_149166


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1491_149155

/-- Given a quadratic function f(x) = ax^2 + bx, prove that if 1 ≤ f(-1) ≤ 2 and 2 ≤ f(1) ≤ 4, then 3 ≤ f(-2) ≤ 12. -/
theorem quadratic_function_range (a b : ℝ) :
  let f := fun x => a * x^2 + b * x
  (1 ≤ f (-1) ∧ f (-1) ≤ 2) ∧ (2 ≤ f 1 ∧ f 1 ≤ 4) →
  3 ≤ f (-2) ∧ f (-2) ≤ 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1491_149155


namespace NUMINAMATH_CALUDE_faster_person_speed_l1491_149111

/-- Given two towns 45 km apart and two people traveling towards each other,
    where one person travels 1 km/h faster than the other and they meet after 5 hours,
    prove that the faster person's speed is 5 km/h. -/
theorem faster_person_speed (distance : ℝ) (time : ℝ) (speed_diff : ℝ) :
  distance = 45 →
  time = 5 →
  speed_diff = 1 →
  ∃ (speed_slower : ℝ),
    speed_slower > 0 ∧
    speed_slower * time + (speed_slower + speed_diff) * time = distance ∧
    speed_slower + speed_diff = 5 := by
  sorry

end NUMINAMATH_CALUDE_faster_person_speed_l1491_149111


namespace NUMINAMATH_CALUDE_missing_digit_is_seven_l1491_149141

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

theorem missing_digit_is_seven :
  ∃ (d : ℕ), d < 10 ∧ is_divisible_by_9 (365000 + d * 100 + 42) ∧ d = 7 :=
by sorry

end NUMINAMATH_CALUDE_missing_digit_is_seven_l1491_149141


namespace NUMINAMATH_CALUDE_plane_equation_correct_l1491_149146

/-- Represents a plane in 3D space -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- Checks if a point lies on a plane -/
def Plane.contains (p : Plane) (x y z : ℤ) : Prop :=
  p.A * x + p.B * y + p.C * z + p.D = 0

/-- Checks if a vector is perpendicular to another vector -/
def isPerpendicular (x1 y1 z1 x2 y2 z2 : ℤ) : Prop :=
  x1 * x2 + y1 * y2 + z1 * z2 = 0

theorem plane_equation_correct : ∃ (p : Plane),
  p.contains 10 (-2) 5 ∧
  isPerpendicular p.A p.B p.C 10 (-2) 5 ∧
  p.A = 10 ∧ p.B = -2 ∧ p.C = 5 ∧ p.D = -129 := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l1491_149146


namespace NUMINAMATH_CALUDE_x_value_when_y_is_two_l1491_149198

theorem x_value_when_y_is_two (x y : ℝ) : 
  y = 1 / (4 * x + 2) → y = 2 → x = -3/8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_y_is_two_l1491_149198


namespace NUMINAMATH_CALUDE_count_scalene_triangles_l1491_149181

def is_valid_scalene_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ a + b + c < 16

theorem count_scalene_triangles :
  ∃! (triangles : Finset (ℕ × ℕ × ℕ)),
    triangles.card = 6 ∧
    ∀ (t : ℕ × ℕ × ℕ), t ∈ triangles ↔ is_valid_scalene_triangle t.1 t.2.1 t.2.2 :=
by sorry

end NUMINAMATH_CALUDE_count_scalene_triangles_l1491_149181


namespace NUMINAMATH_CALUDE_prime_divisibility_l1491_149187

theorem prime_divisibility (p q : ℕ) : 
  Prime p → Prime q → (p * q) ∣ (2^p + 2^q) → p = 2 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l1491_149187


namespace NUMINAMATH_CALUDE_exists_balanced_partition_l1491_149193

/-- An undirected graph represented by its vertex set and edge relation -/
structure Graph (V : Type) where
  edge : V → V → Prop
  symm : ∀ u v, edge u v → edge v u

/-- The neighborhood of a vertex v in a set S -/
def neighborhood {V : Type} (G : Graph V) (S : Set V) (v : V) : Set V :=
  {u ∈ S | G.edge v u}

/-- A partition of a set into two disjoint subsets -/
structure Partition (V : Type) where
  A : Set V
  B : Set V
  disjoint : A ∩ B = ∅
  complete : A ∪ B = Set.univ

/-- The main theorem statement -/
theorem exists_balanced_partition {V : Type} (G : Graph V) :
  ∃ (P : Partition V), 
    (∀ v ∈ P.A, (neighborhood G P.B v).ncard ≥ (neighborhood G P.A v).ncard) ∧
    (∀ v ∈ P.B, (neighborhood G P.A v).ncard ≥ (neighborhood G P.B v).ncard) := by
  sorry

end NUMINAMATH_CALUDE_exists_balanced_partition_l1491_149193


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_equals_71_l1491_149147

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 2*x^6 + 5*x^5 + 6*x^4 + 23*x^3 - 8*x^2 + 10*x - 3

-- Define Qin Jiushao's algorithm for calculating V₃
def qin_jiushao_v3 (x : ℝ) : ℝ :=
  let v0 := 2
  let v1 := v0 * x + 5
  let v2 := v1 * x + 6
  v2 * x + 23

-- Theorem statement
theorem qin_jiushao_v3_equals_71 :
  qin_jiushao_v3 2 = 71 :=
by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_equals_71_l1491_149147


namespace NUMINAMATH_CALUDE_remaining_digits_average_l1491_149154

theorem remaining_digits_average (digits : Finset ℕ) (subset : Finset ℕ) :
  Finset.card digits = 9 →
  (Finset.sum digits id) / 9 = 18 →
  Finset.card subset = 4 →
  subset ⊆ digits →
  (Finset.sum subset id) / 4 = 8 →
  let remaining := digits \ subset
  ((Finset.sum remaining id) / (Finset.card remaining) : ℚ) = 26 := by
sorry

end NUMINAMATH_CALUDE_remaining_digits_average_l1491_149154


namespace NUMINAMATH_CALUDE_quadratic_inequality_necessary_not_sufficient_l1491_149191

theorem quadratic_inequality_necessary_not_sufficient :
  (∃ x : ℝ, (|x - 2| < 1 ∧ ¬(x^2 - 5*x + 4 < 0))) ∧
  (∀ x : ℝ, (x^2 - 5*x + 4 < 0 → |x - 2| < 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_necessary_not_sufficient_l1491_149191


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equation_l1491_149164

theorem unique_function_satisfying_equation :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = x + y :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equation_l1491_149164


namespace NUMINAMATH_CALUDE_fraction_inequality_l1491_149100

theorem fraction_inequality (a b m : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : m > 0) (h4 : a + m > 0) :
  (b + m) / (a + m) > b / a :=
by sorry

end NUMINAMATH_CALUDE_fraction_inequality_l1491_149100


namespace NUMINAMATH_CALUDE_consecutive_sum_formula_l1491_149149

def consecutive_sum (n : ℤ) : ℤ := (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)

theorem consecutive_sum_formula (n : ℤ) : consecutive_sum n = 5 * n + 20 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_formula_l1491_149149


namespace NUMINAMATH_CALUDE_special_function_is_identity_l1491_149184

/-- A function satisfying certain conditions -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x ≤ x) ∧ (∀ x y, f (x + y) ≤ f x + f y)

/-- Theorem: If f is a SpecialFunction, then f(x) = x for all x in ℝ -/
theorem special_function_is_identity (f : ℝ → ℝ) (hf : SpecialFunction f) :
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_special_function_is_identity_l1491_149184


namespace NUMINAMATH_CALUDE_percentage_runs_from_running_approx_l1491_149180

def total_runs : ℕ := 120
def num_boundaries : ℕ := 5
def num_sixes : ℕ := 5
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

def runs_from_boundaries : ℕ := num_boundaries * runs_per_boundary
def runs_from_sixes : ℕ := num_sixes * runs_per_six
def runs_without_running : ℕ := runs_from_boundaries + runs_from_sixes
def runs_from_running : ℕ := total_runs - runs_without_running

theorem percentage_runs_from_running_approx (ε : ℚ) (h : ε > 0) :
  ∃ (p : ℚ), abs (p - (runs_from_running : ℚ) / (total_runs : ℚ) * 100) < ε ∧ 
             abs (p - 58.33) < ε :=
sorry

end NUMINAMATH_CALUDE_percentage_runs_from_running_approx_l1491_149180


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l1491_149127

/-- Represents a player's score in a chess competition --/
structure PlayerScore where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ

/-- Calculate the success ratio for a given day --/
def day_success_ratio (score : ℕ) (total : ℕ) : ℚ :=
  ↑score / ↑total

/-- Calculate the overall success ratio --/
def overall_success_ratio (player : PlayerScore) : ℚ :=
  ↑(player.day1_score + player.day2_score) / ↑(player.day1_total + player.day2_total)

theorem delta_max_success_ratio 
  (gamma : PlayerScore)
  (delta : PlayerScore)
  (h1 : gamma.day1_score = 180 ∧ gamma.day1_total = 360)
  (h2 : gamma.day2_score = 150 ∧ gamma.day2_total = 240)
  (h3 : delta.day1_total + delta.day2_total = 600)
  (h4 : delta.day1_total ≠ 360)
  (h5 : delta.day1_score > 0 ∧ delta.day2_score > 0)
  (h6 : day_success_ratio delta.day1_score delta.day1_total < day_success_ratio gamma.day1_score gamma.day1_total)
  (h7 : day_success_ratio delta.day2_score delta.day2_total < day_success_ratio gamma.day2_score gamma.day2_total)
  (h8 : overall_success_ratio gamma = 11/20) :
  overall_success_ratio delta ≤ 599/600 :=
sorry

end NUMINAMATH_CALUDE_delta_max_success_ratio_l1491_149127


namespace NUMINAMATH_CALUDE_fruit_basket_count_l1491_149108

def total_fruits (mangoes pears pawpaws kiwis lemons : ℕ) : ℕ :=
  mangoes + pears + pawpaws + kiwis + lemons

theorem fruit_basket_count : 
  ∀ (kiwis : ℕ),
  kiwis = 9 →
  total_fruits 18 10 12 kiwis 9 = 58 := by
sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l1491_149108


namespace NUMINAMATH_CALUDE_volume_of_specific_room_l1491_149158

/-- The volume of a rectangular room -/
def room_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a room with dimensions 100 m x 10 m x 10 m is 100,000 cubic meters -/
theorem volume_of_specific_room : 
  room_volume 100 10 10 = 100000 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_specific_room_l1491_149158


namespace NUMINAMATH_CALUDE_cost_of_450_candies_l1491_149125

/-- Represents the cost calculation for chocolate candies --/
def chocolate_cost (candies_per_box : ℕ) (discount_threshold : ℕ) (regular_price : ℚ) (discount_price : ℚ) (total_candies : ℕ) : ℚ :=
  let boxes := total_candies / candies_per_box
  if boxes ≥ discount_threshold then
    (boxes : ℚ) * discount_price
  else
    (boxes : ℚ) * regular_price

/-- Theorem stating the cost of 450 chocolate candies --/
theorem cost_of_450_candies :
  chocolate_cost 15 10 5 4 450 = 120 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_450_candies_l1491_149125


namespace NUMINAMATH_CALUDE_distance_sum_inequality_l1491_149107

theorem distance_sum_inequality (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 5| + |x - 7| < b) ↔ b > 2 := by sorry

end NUMINAMATH_CALUDE_distance_sum_inequality_l1491_149107


namespace NUMINAMATH_CALUDE_stating_last_seat_probability_is_reciprocal_seven_seats_probability_l1491_149115

/-- 
Represents the probability that the last passenger sits in their own seat 
in a seating arrangement problem with n seats and n passengers.
-/
def last_seat_probability (n : ℕ) : ℚ :=
  if n = 0 then 0
  else 1 / n

/-- 
Theorem stating that the probability of the last passenger sitting in their own seat
is 1/n for any number of seats n > 0.
-/
theorem last_seat_probability_is_reciprocal (n : ℕ) (h : n > 0) : 
  last_seat_probability n = 1 / n := by
  sorry

/-- 
Corollary for the specific case of 7 seats, as in the original problem.
-/
theorem seven_seats_probability : 
  last_seat_probability 7 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_stating_last_seat_probability_is_reciprocal_seven_seats_probability_l1491_149115


namespace NUMINAMATH_CALUDE_count_squares_with_six_black_l1491_149105

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  x : Nat
  y : Nat

/-- The size of the checkerboard -/
def boardSize : Nat := 10

/-- Checks if a square contains at least 6 black squares -/
def containsSixBlackSquares (s : Square) : Bool :=
  if s.size ≥ 5 then true
  else if s.size = 4 then (s.x + s.y) % 2 = 0
  else false

/-- Counts the number of squares containing at least 6 black squares -/
def countSquaresWithSixBlack : Nat :=
  let fourByFour := (boardSize - 3) * (boardSize - 3) / 2
  let fiveByFive := (boardSize - 4) * (boardSize - 4)
  let sixBySix := (boardSize - 5) * (boardSize - 5)
  let sevenBySeven := (boardSize - 6) * (boardSize - 6)
  let eightByEight := (boardSize - 7) * (boardSize - 7)
  let nineByNine := (boardSize - 8) * (boardSize - 8)
  let tenByTen := 1
  fourByFour + fiveByFive + sixBySix + sevenBySeven + eightByEight + nineByNine + tenByTen

theorem count_squares_with_six_black :
  countSquaresWithSixBlack = 115 := by
  sorry

end NUMINAMATH_CALUDE_count_squares_with_six_black_l1491_149105


namespace NUMINAMATH_CALUDE_households_with_only_bike_l1491_149106

-- Define the total number of households
def total_households : ℕ := 90

-- Define the number of households without car or bike
def households_without_car_or_bike : ℕ := 11

-- Define the number of households with both car and bike
def households_with_both : ℕ := 16

-- Define the number of households with a car
def households_with_car : ℕ := 44

-- Theorem to prove
theorem households_with_only_bike : 
  total_households - households_without_car_or_bike - households_with_car + households_with_both = 35 := by
  sorry

end NUMINAMATH_CALUDE_households_with_only_bike_l1491_149106


namespace NUMINAMATH_CALUDE_max_value_cyclic_sum_equality_condition_l1491_149131

theorem max_value_cyclic_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  (a / (a^3 + b^2 + c)) + (b / (b^3 + c^2 + a)) + (c / (c^3 + a^2 + b)) ≤ 1 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_3 : a + b + c = 3) : 
  (a / (a^3 + b^2 + c)) + (b / (b^3 + c^2 + a)) + (c / (c^3 + a^2 + b)) = 1 ↔ 
  a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cyclic_sum_equality_condition_l1491_149131


namespace NUMINAMATH_CALUDE_josh_remaining_money_l1491_149163

/-- Calculates the remaining money after Josh's shopping trip. -/
def remaining_money (initial_amount hat_cost pencil_cost cookie_cost cookie_count : ℚ) : ℚ :=
  initial_amount - (hat_cost + pencil_cost + cookie_cost * cookie_count)

/-- Theorem stating that Josh has $3 left after his shopping trip. -/
theorem josh_remaining_money :
  remaining_money 20 10 2 1.25 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l1491_149163


namespace NUMINAMATH_CALUDE_parrot_seed_consumption_l1491_149197

/-- Given a parrot that absorbs 40% of the seeds it consumes and absorbed 8 ounces of seeds,
    prove that the total amount of seeds consumed is 20 ounces and twice that amount is 40 ounces. -/
theorem parrot_seed_consumption (absorbed_percentage : ℝ) (absorbed_amount : ℝ) 
    (h1 : absorbed_percentage = 0.40)
    (h2 : absorbed_amount = 8) : 
  ∃ (total_consumed : ℝ), 
    total_consumed * absorbed_percentage = absorbed_amount ∧ 
    total_consumed = 20 ∧ 
    2 * total_consumed = 40 := by
  sorry


end NUMINAMATH_CALUDE_parrot_seed_consumption_l1491_149197


namespace NUMINAMATH_CALUDE_napkin_ratio_l1491_149122

/-- Proves the ratio of napkins Amelia gave to napkins Olivia gave -/
theorem napkin_ratio (william_initial : ℕ) (william_final : ℕ) (olivia_gave : ℕ) 
  (h1 : william_initial = 15)
  (h2 : william_final = 45)
  (h3 : olivia_gave = 10) :
  (william_final - william_initial - olivia_gave) / olivia_gave = 2 := by
  sorry

end NUMINAMATH_CALUDE_napkin_ratio_l1491_149122


namespace NUMINAMATH_CALUDE_cubic_fraction_evaluation_l1491_149156

theorem cubic_fraction_evaluation :
  let a : ℚ := 7
  let b : ℚ := 6
  let c : ℚ := 1
  (a^3 + b^3) / (a^2 - a*b + b^2 + c) = 559 / 44 := by
  sorry

end NUMINAMATH_CALUDE_cubic_fraction_evaluation_l1491_149156


namespace NUMINAMATH_CALUDE_science_books_count_l1491_149132

theorem science_books_count (total : ℕ) (storybooks science picture dictionaries : ℕ) :
  total = 35 →
  total = storybooks + science + picture + dictionaries →
  storybooks + science = 17 →
  science + picture = 16 →
  storybooks ≠ science →
  storybooks ≠ picture →
  storybooks ≠ dictionaries →
  science ≠ picture →
  science ≠ dictionaries →
  picture ≠ dictionaries →
  (storybooks = 9 ∨ science = 9 ∨ picture = 9 ∨ dictionaries = 9) →
  science = 9 :=
by sorry

end NUMINAMATH_CALUDE_science_books_count_l1491_149132


namespace NUMINAMATH_CALUDE_helen_hand_wash_frequency_l1491_149167

/-- The frequency of Helen's hand washing her pillowcases in weeks -/
def hand_wash_frequency (time_per_wash : ℕ) (total_time_per_year : ℕ) (weeks_per_year : ℕ) : ℕ :=
  weeks_per_year / (total_time_per_year / time_per_wash)

/-- Theorem stating that Helen hand washes her pillowcases every 4 weeks -/
theorem helen_hand_wash_frequency :
  hand_wash_frequency 30 390 52 = 4 := by
  sorry

end NUMINAMATH_CALUDE_helen_hand_wash_frequency_l1491_149167


namespace NUMINAMATH_CALUDE_union_A_B_equals_open_interval_l1491_149159

-- Define set A
def A : Set ℝ := {x | Real.log (x - 1) < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = 2^x - 1}

-- Theorem statement
theorem union_A_B_equals_open_interval :
  A ∪ B = Set.Ioo 1 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_equals_open_interval_l1491_149159


namespace NUMINAMATH_CALUDE_polynomial_division_l1491_149119

theorem polynomial_division (x : ℝ) (h : x ≠ 0) :
  (6 * x^4 - 4 * x^3 + 2 * x^2) / (2 * x^2) = 3 * x^2 - 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_l1491_149119


namespace NUMINAMATH_CALUDE_point_position_l1491_149109

def line (x y : ℝ) := x + 2 * y = 2

def point_below_left (P : ℝ × ℝ) : Prop :=
  P.1 + 2 * P.2 < 2

theorem point_position :
  let P : ℝ × ℝ := (1/12, 33/36)
  point_below_left P := by sorry

end NUMINAMATH_CALUDE_point_position_l1491_149109


namespace NUMINAMATH_CALUDE_white_square_area_l1491_149114

/-- Given a cube with edge length 10 feet and 300 square feet of paint used for borders,
    the area of the white square on each face is 50 square feet. -/
theorem white_square_area (cube_edge : ℝ) (paint_area : ℝ) : 
  cube_edge = 10 →
  paint_area = 300 →
  (6 * cube_edge^2 - paint_area) / 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_white_square_area_l1491_149114


namespace NUMINAMATH_CALUDE_max_min_distance_difference_l1491_149144

/-- Two unit squares with horizontal and vertical sides -/
structure UnitSquare where
  bottomLeft : ℝ × ℝ

/-- The minimum distance between two points -/
def minDistance (s1 s2 : UnitSquare) : ℝ :=
  sorry

/-- The maximum distance between two points -/
def maxDistance (s1 s2 : UnitSquare) : ℝ :=
  sorry

/-- Theorem: The difference between max and min possible y values is 5 - 3√2 -/
theorem max_min_distance_difference (s1 s2 : UnitSquare) 
  (h : minDistance s1 s2 = 5) :
  ∃ (yMin yMax : ℝ),
    yMin ≤ maxDistance s1 s2 ∧ 
    maxDistance s1 s2 ≤ yMax ∧
    yMax - yMin = 5 - 3 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_max_min_distance_difference_l1491_149144


namespace NUMINAMATH_CALUDE_subtraction_decimal_l1491_149199

theorem subtraction_decimal : (3.56 : ℝ) - (1.89 : ℝ) = 1.67 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_decimal_l1491_149199


namespace NUMINAMATH_CALUDE_intersection_point_condition_l1491_149190

theorem intersection_point_condition (α β : ℝ) : 
  (∃ x y : ℝ, 
    (x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1) ∧
    (x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1) ∧
    y = -x) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_condition_l1491_149190


namespace NUMINAMATH_CALUDE_hash_difference_l1491_149176

/-- Custom operation # -/
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

/-- Theorem stating the result of (5 # 3) - (3 # 5) -/
theorem hash_difference : hash 5 3 - hash 3 5 = -8 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l1491_149176


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l1491_149165

/-- A regular polygon with (2n-1) sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin (2*n-1) → ℝ × ℝ

/-- A subset of n vertices from a (2n-1)-gon -/
def VertexSubset (n : ℕ) := Fin n → Fin (2*n-1)

/-- Predicate to check if three vertices form an isosceles triangle -/
def IsIsosceles (p : RegularPolygon n) (a b c : Fin (2*n-1)) : Prop :=
  let va := p.vertices a
  let vb := p.vertices b
  let vc := p.vertices c
  (va.1 - vc.1)^2 + (va.2 - vc.2)^2 = (vb.1 - vc.1)^2 + (vb.2 - vc.2)^2

/-- Main theorem: In any subset of n vertices of a (2n-1)-gon, there exists an isosceles triangle -/
theorem isosceles_triangle_exists (n : ℕ) (h : n ≥ 3) (p : RegularPolygon n) (s : VertexSubset n) :
  ∃ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ IsIsosceles p (s a) (s b) (s c) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_exists_l1491_149165


namespace NUMINAMATH_CALUDE_extended_morse_code_symbols_l1491_149177

-- Define the function to calculate the number of sequences for a given length
def sequencesForLength (n : ℕ) : ℕ := 2^n

-- Define the total number of sequences for lengths 1 to 5
def totalSequences : ℕ :=
  (sequencesForLength 1) + (sequencesForLength 2) + (sequencesForLength 3) +
  (sequencesForLength 4) + (sequencesForLength 5)

-- Theorem statement
theorem extended_morse_code_symbols :
  totalSequences = 62 := by
  sorry

end NUMINAMATH_CALUDE_extended_morse_code_symbols_l1491_149177


namespace NUMINAMATH_CALUDE_three_digit_sum_divisibility_l1491_149126

theorem three_digit_sum_divisibility (a b : ℕ) : 
  (100 * 2 + 10 * a + 3) + 326 = (500 + 10 * b + 9) → 
  (500 + 10 * b + 9) % 9 = 0 →
  a + b = 6 := by sorry

end NUMINAMATH_CALUDE_three_digit_sum_divisibility_l1491_149126


namespace NUMINAMATH_CALUDE_factorial_fraction_equals_one_l1491_149143

theorem factorial_fraction_equals_one : (4 * Nat.factorial 7 + 28 * Nat.factorial 6) / Nat.factorial 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equals_one_l1491_149143


namespace NUMINAMATH_CALUDE_entree_cost_calculation_l1491_149173

/-- Given a total cost for an entree and dessert, where the entree costs $5 more than the dessert,
    this function calculates the cost of the entree. -/
def entree_cost (total : ℚ) (difference : ℚ) : ℚ :=
  (total + difference) / 2

theorem entree_cost_calculation :
  entree_cost 23 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_entree_cost_calculation_l1491_149173


namespace NUMINAMATH_CALUDE_polygon_properties_l1491_149178

-- Define the polygon
structure Polygon where
  n : ℕ  -- number of sides
  h : n > 2  -- a polygon must have at least 3 sides

-- Define the ratio of interior to exterior angles
def interiorToExteriorRatio (p : Polygon) : ℚ :=
  (p.n - 2) / 2

-- Theorem statement
theorem polygon_properties (p : Polygon) 
  (h : interiorToExteriorRatio p = 13 / 2) : 
  p.n = 15 ∧ (p.n * (p.n - 3)) / 2 = 90 := by
  sorry


end NUMINAMATH_CALUDE_polygon_properties_l1491_149178


namespace NUMINAMATH_CALUDE_pizza_toppings_theorem_l1491_149183

/-- Represents the number of distinct toppings on a pizza slice -/
def toppings_on_slice (n k : ℕ+) (t : Fin (2 * k)) : ℕ :=
  sorry

/-- The minimum number of distinct toppings on any slice -/
def min_toppings (n k : ℕ+) : ℕ :=
  sorry

/-- The maximum number of distinct toppings on any slice -/
def max_toppings (n k : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that the sum of minimum and maximum toppings equals the total number of toppings -/
theorem pizza_toppings_theorem (n k : ℕ+) :
    min_toppings n k + max_toppings n k = n :=
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_theorem_l1491_149183


namespace NUMINAMATH_CALUDE_correct_reading_growth_equation_l1491_149140

/-- Represents the growth of average reading amount per student over 2 years -/
def reading_growth (x : ℝ) : Prop :=
  let initial_amount : ℝ := 1
  let final_amount : ℝ := 1.21
  let growth_period : ℕ := 2
  100 * (1 + x)^growth_period = 121

/-- Proves that the equation correctly represents the reading growth -/
theorem correct_reading_growth_equation :
  ∃ x : ℝ, reading_growth x ∧ x > 0 := by sorry

end NUMINAMATH_CALUDE_correct_reading_growth_equation_l1491_149140


namespace NUMINAMATH_CALUDE_negation_of_proposition_l1491_149134

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l1491_149134


namespace NUMINAMATH_CALUDE_candy_distribution_l1491_149138

theorem candy_distribution (left right : ℕ) : 
  left + right = 27 →
  right - left = (left + left) + 3 →
  left = 6 ∧ right = 21 := by
sorry

end NUMINAMATH_CALUDE_candy_distribution_l1491_149138


namespace NUMINAMATH_CALUDE_player_percentage_of_team_points_l1491_149170

def three_point_goals : ℕ := 5
def two_point_goals : ℕ := 10
def team_total_points : ℕ := 70

def player_points : ℕ := three_point_goals * 3 + two_point_goals * 2

theorem player_percentage_of_team_points :
  (player_points : ℚ) / team_total_points * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_player_percentage_of_team_points_l1491_149170
