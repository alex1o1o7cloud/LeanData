import Mathlib

namespace NUMINAMATH_CALUDE_angles_with_same_terminal_side_as_45_l2410_241039

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β = α + k * 360

theorem angles_with_same_terminal_side_as_45 :
  ∀ θ : ℝ, -720 ≤ θ ∧ θ < 0 ∧ same_terminal_side 45 θ →
    θ = -675 ∨ θ = -315 := by sorry

end NUMINAMATH_CALUDE_angles_with_same_terminal_side_as_45_l2410_241039


namespace NUMINAMATH_CALUDE_intersection_with_complement_of_B_l2410_241073

open Set

theorem intersection_with_complement_of_B (U A B : Set ℕ) : 
  U = {1, 2, 3, 4, 5, 6} →
  A = {2, 4, 6} →
  B = {1, 3} →
  A ∩ (U \ B) = {2, 4, 6} := by
sorry

end NUMINAMATH_CALUDE_intersection_with_complement_of_B_l2410_241073


namespace NUMINAMATH_CALUDE_min_value_f_l2410_241026

def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem min_value_f (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 
    (if a < -1 then 2*a
     else if a ≤ 1 then -1 - a^2
     else -2*a)) ∧
  (∃ x ∈ Set.Icc (-1) 1, f a x = 
    (if a < -1 then 2*a
     else if a ≤ 1 then -1 - a^2
     else -2*a)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_f_l2410_241026


namespace NUMINAMATH_CALUDE_cookie_orange_cost_ratio_l2410_241024

/-- Cost of items in Susie and Calvin's purchases -/
structure ItemCosts where
  orange : ℚ
  muffin : ℚ
  cookie : ℚ

/-- Susie's purchase -/
def susie_purchase (costs : ItemCosts) : ℚ :=
  3 * costs.muffin + 5 * costs.orange

/-- Calvin's purchase -/
def calvin_purchase (costs : ItemCosts) : ℚ :=
  5 * costs.muffin + 10 * costs.orange + 4 * costs.cookie

theorem cookie_orange_cost_ratio :
  ∀ (costs : ItemCosts),
  costs.muffin = 2 * costs.orange →
  calvin_purchase costs = 3 * susie_purchase costs →
  costs.cookie = (13/4) * costs.orange :=
by sorry

end NUMINAMATH_CALUDE_cookie_orange_cost_ratio_l2410_241024


namespace NUMINAMATH_CALUDE_number_division_problem_l2410_241086

theorem number_division_problem (x : ℝ) : x / 0.04 = 200.9 → x = 8.036 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l2410_241086


namespace NUMINAMATH_CALUDE_second_half_speed_l2410_241023

/-- Proves that given a journey of 224 km completed in 10 hours, where the first half is traveled at 21 km/hr, the speed of the second half of the journey is 24 km/hr. -/
theorem second_half_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) 
    (h1 : total_distance = 224)
    (h2 : total_time = 10)
    (h3 : first_half_speed = 21) : 
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 24 := by
  sorry

#check second_half_speed

end NUMINAMATH_CALUDE_second_half_speed_l2410_241023


namespace NUMINAMATH_CALUDE_percentage_problem_l2410_241088

theorem percentage_problem (N : ℝ) : 
  (0.4 * N = 4/5 * 25 + 4) → N = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2410_241088


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_5_l2410_241021

theorem smallest_positive_integer_ending_in_3_divisible_by_5 : 
  ∃ n : ℕ, n > 0 ∧ n % 10 = 3 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 10 = 3 → m % 5 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_3_divisible_by_5_l2410_241021


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2410_241046

/-- Configuration of semicircles and inscribed circle -/
structure CircleConfiguration where
  R : ℝ  -- Radius of larger semicircle
  r : ℝ  -- Radius of smaller semicircle
  x : ℝ  -- Radius of inscribed circle

/-- Conditions for the circle configuration -/
def valid_configuration (c : CircleConfiguration) : Prop :=
  c.R = 18 ∧ c.r = 9 ∧ c.x > 0 ∧ c.x < c.R ∧
  (c.R - c.x)^2 - c.x^2 = (c.r + c.x)^2 - c.x^2

/-- Theorem stating that the radius of the inscribed circle is 8 -/
theorem inscribed_circle_radius (c : CircleConfiguration) 
  (h : valid_configuration c) : c.x = 8 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_l2410_241046


namespace NUMINAMATH_CALUDE_fencing_requirement_l2410_241041

/-- A rectangular field with one side of 20 feet and an area of 80 sq. feet requires 28 feet of fencing for the other three sides. -/
theorem fencing_requirement (length width : ℝ) : 
  length = 20 → 
  length * width = 80 → 
  length + 2 * width = 28 := by sorry

end NUMINAMATH_CALUDE_fencing_requirement_l2410_241041


namespace NUMINAMATH_CALUDE_final_milk_composition_l2410_241000

/-- The percentage of milk remaining after each replacement operation -/
def replacement_factor : ℝ := 0.7

/-- The number of replacement operations performed -/
def num_operations : ℕ := 5

/-- The final percentage of milk in the container after all operations -/
def final_milk_percentage : ℝ := replacement_factor ^ num_operations * 100

/-- Theorem stating the final percentage of milk after the operations -/
theorem final_milk_composition :
  ∃ ε > 0, |final_milk_percentage - 16.807| < ε :=
sorry

end NUMINAMATH_CALUDE_final_milk_composition_l2410_241000


namespace NUMINAMATH_CALUDE_tiangong_altitude_scientific_notation_l2410_241074

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem tiangong_altitude_scientific_notation :
  toScientificNotation 375000 = ScientificNotation.mk 3.75 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_tiangong_altitude_scientific_notation_l2410_241074


namespace NUMINAMATH_CALUDE_union_A_B_l2410_241065

def A : Set ℝ := {-1, 1}
def B : Set ℝ := {x | x^2 + x - 2 = 0}

theorem union_A_B : A ∪ B = {-2, -1, 1} := by
  sorry

end NUMINAMATH_CALUDE_union_A_B_l2410_241065


namespace NUMINAMATH_CALUDE_xy_value_l2410_241078

theorem xy_value (x y : ℂ) (h : (1 - Complex.I) * x + (1 + Complex.I) * y = 2) : x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2410_241078


namespace NUMINAMATH_CALUDE_count_valid_integers_l2410_241017

/-- The set of available digits -/
def available_digits : Finset ℕ := {1, 4, 7}

/-- The count of each digit in the available set -/
def digit_count : ℕ → ℕ
  | 1 => 2
  | 4 => 3
  | 7 => 1
  | _ => 0

/-- A valid three-digit integer formed from the available digits -/
structure ValidInteger where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  hundreds_in_set : hundreds ∈ available_digits
  tens_in_set : tens ∈ available_digits
  ones_in_set : ones ∈ available_digits
  valid_count : ∀ d ∈ available_digits,
    (if hundreds = d then 1 else 0) +
    (if tens = d then 1 else 0) +
    (if ones = d then 1 else 0) ≤ digit_count d

/-- The set of all valid three-digit integers -/
def valid_integers : Finset ValidInteger := sorry

theorem count_valid_integers :
  Finset.card valid_integers = 31 := by sorry

end NUMINAMATH_CALUDE_count_valid_integers_l2410_241017


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l2410_241013

/-- Represents the side lengths of squares in ascending order -/
structure SquareSides where
  b₁ : ℝ
  b₂ : ℝ
  b₃ : ℝ
  b₄ : ℝ
  b₅ : ℝ
  b₆ : ℝ
  h_order : 0 < b₁ ∧ b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < b₄ ∧ b₄ < b₅ ∧ b₅ < b₆

/-- Represents a rectangle partitioned into six squares -/
structure PartitionedRectangle where
  sides : SquareSides
  length : ℝ
  width : ℝ
  h_partition : length = sides.b₃ + sides.b₆ ∧ width = sides.b₁ + sides.b₅
  h_sum_smallest : sides.b₁ + sides.b₂ = sides.b₃
  h_longest_side : 2 * length = 3 * sides.b₆

theorem rectangle_perimeter (rect : PartitionedRectangle) :
    2 * (rect.length + rect.width) = 12 * rect.sides.b₆ := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_l2410_241013


namespace NUMINAMATH_CALUDE_triangle_count_on_circle_l2410_241054

theorem triangle_count_on_circle (n : ℕ) (h : n = 10) : 
  (n.choose 3) = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangle_count_on_circle_l2410_241054


namespace NUMINAMATH_CALUDE_symmetry_of_P_l2410_241004

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis. -/
def symmetry_x_axis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- The original point P. -/
def P : Point :=
  ⟨-2, -1⟩

theorem symmetry_of_P :
  symmetry_x_axis P = Point.mk (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_of_P_l2410_241004


namespace NUMINAMATH_CALUDE_x1_value_l2410_241056

theorem x1_value (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1/2) : 
  x1 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_x1_value_l2410_241056


namespace NUMINAMATH_CALUDE_factory_B_cheaper_for_200_copies_l2410_241018

/-- Cost calculation for Factory A -/
def cost_A (x : ℝ) : ℝ := 4.8 * x + 500

/-- Cost calculation for Factory B -/
def cost_B (x : ℝ) : ℝ := 6 * x + 200

/-- Theorem stating that Factory B has lower cost for 200 copies -/
theorem factory_B_cheaper_for_200_copies :
  cost_B 200 < cost_A 200 := by
  sorry

end NUMINAMATH_CALUDE_factory_B_cheaper_for_200_copies_l2410_241018


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l2410_241049

theorem sum_of_three_consecutive_integers (a : ℤ) (h : a = 29) :
  a + (a + 1) + (a + 2) = 90 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_integers_l2410_241049


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l2410_241003

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : Real) (time : Real) : Real :=
  let length_km := speed * (time / 3600)
  let length_m := length_km * 1000
  length_m

/-- Proves that a train with speed 60 km/hr crossing a pole in 15 seconds has a length of 250 meters -/
theorem train_length_proof :
  train_length 60 15 = 250 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l2410_241003


namespace NUMINAMATH_CALUDE_sin_2alpha_minus_cos_2alpha_l2410_241047

theorem sin_2alpha_minus_cos_2alpha (α : Real) (h : Real.tan α = 3) :
  Real.sin (2 * α) - Real.cos (2 * α) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_minus_cos_2alpha_l2410_241047


namespace NUMINAMATH_CALUDE_negation_equivalence_l2410_241058

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5*x₀ + 6 > 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2410_241058


namespace NUMINAMATH_CALUDE_difference_zero_iff_k_ge_five_l2410_241092

/-- Definition of the sequence u_n -/
def u (n : ℕ) : ℕ := n^4 + 2*n^2

/-- Definition of the first difference operator -/
def Δ₁ (f : ℕ → ℕ) (n : ℕ) : ℕ := f (n + 1) - f n

/-- Definition of the k-th difference operator -/
def Δ (k : ℕ) (f : ℕ → ℕ) : ℕ → ℕ :=
  match k with
  | 0 => f
  | k + 1 => Δ₁ (Δ k f)

/-- Theorem: The k-th difference of u_n is zero for all n if and only if k ≥ 5 -/
theorem difference_zero_iff_k_ge_five :
  ∀ k : ℕ, (∀ n : ℕ, Δ k u n = 0) ↔ k ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_difference_zero_iff_k_ge_five_l2410_241092


namespace NUMINAMATH_CALUDE_area_between_graphs_l2410_241055

-- Define the functions
def f (x : ℝ) : ℝ := |2 * x| - 3
def g (x : ℝ) : ℝ := |x|

-- Define the area enclosed by the graphs
def enclosed_area : ℝ := 9

-- Theorem statement
theorem area_between_graphs :
  (∃ (a b : ℝ), a < b ∧
    (∀ x ∈ Set.Icc a b, f x ≠ g x) ∧
    (∀ x ∈ Set.Ioi b, f x = g x) ∧
    (∀ x ∈ Set.Iio a, f x = g x)) →
  (∫ (x : ℝ) in Set.Icc (-3) 3, |f x - g x|) = enclosed_area :=
sorry

end NUMINAMATH_CALUDE_area_between_graphs_l2410_241055


namespace NUMINAMATH_CALUDE_problem_solution_l2410_241051

theorem problem_solution (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 152) : a = 50 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2410_241051


namespace NUMINAMATH_CALUDE_emily_sees_emerson_time_l2410_241014

def emily_speed : ℝ := 15
def emerson_speed : ℝ := 9
def initial_distance : ℝ := 1
def final_distance : ℝ := 1

theorem emily_sees_emerson_time : 
  let relative_speed := emily_speed - emerson_speed
  let time_to_catch := initial_distance / relative_speed
  let time_to_lose_sight := final_distance / relative_speed
  let total_time := time_to_catch + time_to_lose_sight
  (total_time * 60) = 20 := by sorry

end NUMINAMATH_CALUDE_emily_sees_emerson_time_l2410_241014


namespace NUMINAMATH_CALUDE_sixth_power_sum_l2410_241068

theorem sixth_power_sum (r : ℝ) (h : (r + 1/r)^4 = 17) : 
  r^6 + 1/r^6 = Real.sqrt 17 - 6 := by
  sorry

end NUMINAMATH_CALUDE_sixth_power_sum_l2410_241068


namespace NUMINAMATH_CALUDE_train_speed_l2410_241089

/-- Proves that a train of given length passing a point in a given time has a specific speed in kmph -/
theorem train_speed (train_length : Real) (time_to_pass : Real) (speed_kmph : Real) : 
  train_length = 20 →
  time_to_pass = 1.9998400127989762 →
  speed_kmph = (train_length / time_to_pass) * 3.6 →
  speed_kmph = 36.00287986320432 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l2410_241089


namespace NUMINAMATH_CALUDE_exponent_multiplication_l2410_241087

theorem exponent_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l2410_241087


namespace NUMINAMATH_CALUDE_pigeon_count_theorem_l2410_241028

theorem pigeon_count_theorem :
  ∃! n : ℕ,
    300 < n ∧ n < 900 ∧
    n % 2 = 1 ∧
    n % 3 = 2 ∧
    n % 4 = 3 ∧
    n % 5 = 4 ∧
    n % 6 = 5 ∧
    n % 7 = 0 ∧
    n = 539 := by
  sorry

end NUMINAMATH_CALUDE_pigeon_count_theorem_l2410_241028


namespace NUMINAMATH_CALUDE_boosters_club_average_sales_l2410_241009

/-- The average monthly sales for the Boosters Club candy sales --/
theorem boosters_club_average_sales :
  let sales : List ℕ := [90, 50, 70, 110, 80]
  let total_sales : ℕ := sales.sum
  let num_months : ℕ := sales.length
  (total_sales : ℚ) / num_months = 80 := by sorry

end NUMINAMATH_CALUDE_boosters_club_average_sales_l2410_241009


namespace NUMINAMATH_CALUDE_dandelion_seeds_l2410_241083

theorem dandelion_seeds (S : ℕ) : 
  (2/3 : ℚ) * (5/6 : ℚ) * (1/2 : ℚ) * S = 75 → S = 540 := by
  sorry

end NUMINAMATH_CALUDE_dandelion_seeds_l2410_241083


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l2410_241025

theorem cubic_root_ratio (r : ℝ) (h : r > 1) : 
  (∃ a b c : ℝ, 
    (81 * a^3 - 243 * a^2 + 216 * a - 64 = 0) ∧ 
    (81 * b^3 - 243 * b^2 + 216 * b - 64 = 0) ∧ 
    (81 * c^3 - 243 * c^2 + 216 * c - 64 = 0) ∧ 
    (b = a * r) ∧ 
    (c = b * r)) → 
  (c / a = r^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l2410_241025


namespace NUMINAMATH_CALUDE_product_354_78_base7_units_digit_l2410_241095

-- Define the multiplication of two numbers in base 10
def base10Multiply (a b : ℕ) : ℕ := a * b

-- Define the conversion of a number from base 10 to base 7
def toBase7 (n : ℕ) : ℕ := n

-- Define the units digit of a number in base 7
def unitsDigitBase7 (n : ℕ) : ℕ := n % 7

-- Theorem statement
theorem product_354_78_base7_units_digit :
  unitsDigitBase7 (toBase7 (base10Multiply 354 78)) = 4 := by sorry

end NUMINAMATH_CALUDE_product_354_78_base7_units_digit_l2410_241095


namespace NUMINAMATH_CALUDE_square_area_ratio_sqrt_l2410_241075

theorem square_area_ratio_sqrt (side_c side_d : ℝ) 
  (h1 : side_c = 45)
  (h2 : side_d = 60) : 
  Real.sqrt ((side_c ^ 2) / (side_d ^ 2)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_sqrt_l2410_241075


namespace NUMINAMATH_CALUDE_simplify_expression_l2410_241035

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) :
  Real.sqrt ((a - Real.pi) ^ 2) + |a - 2| = Real.pi - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2410_241035


namespace NUMINAMATH_CALUDE_rectangular_hall_area_l2410_241062

theorem rectangular_hall_area (length width : ℝ) : 
  width = (1/2) * length →
  length - width = 10 →
  length * width = 200 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_area_l2410_241062


namespace NUMINAMATH_CALUDE_new_person_weight_l2410_241007

theorem new_person_weight (n : ℕ) (initial_weight replaced_weight avg_increase : ℝ) :
  n = 8 ∧ 
  replaced_weight = 65 ∧
  avg_increase = 4 →
  n * avg_increase + replaced_weight = 97 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2410_241007


namespace NUMINAMATH_CALUDE_gorilla_to_cat_dog_ratio_l2410_241048

/-- Represents the lengths of animal videos and their ratio -/
structure AnimalVideos where
  cat_length : ℕ
  dog_length : ℕ
  total_time : ℕ
  gorilla_length : ℕ
  ratio : Rat

/-- Theorem stating the ratio of gorilla video length to combined cat and dog video length -/
theorem gorilla_to_cat_dog_ratio (v : AnimalVideos) 
  (h1 : v.cat_length = 4)
  (h2 : v.dog_length = 2 * v.cat_length)
  (h3 : v.total_time = 36)
  (h4 : v.gorilla_length = v.total_time - (v.cat_length + v.dog_length))
  (h5 : v.ratio = v.gorilla_length / (v.cat_length + v.dog_length)) :
  v.ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_gorilla_to_cat_dog_ratio_l2410_241048


namespace NUMINAMATH_CALUDE_water_per_drop_l2410_241020

/-- Given a faucet that drips 10 times per minute and wastes 30 mL of water in one hour,
    prove that each drop contains 0.05 mL of water. -/
theorem water_per_drop (drips_per_minute : ℕ) (water_wasted_per_hour : ℝ) :
  drips_per_minute = 10 →
  water_wasted_per_hour = 30 →
  (water_wasted_per_hour / (drips_per_minute * 60 : ℝ)) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_water_per_drop_l2410_241020


namespace NUMINAMATH_CALUDE_fundraising_theorem_l2410_241099

def fundraising_problem (goal ken_amount : ℕ) : Prop :=
  let mary_amount := 5 * ken_amount
  let scott_amount := mary_amount / 3
  let total_raised := ken_amount + mary_amount + scott_amount
  (mary_amount = 5 * ken_amount) ∧
  (mary_amount = 3 * scott_amount) ∧
  (ken_amount = 600) ∧
  (total_raised - goal = 600)

theorem fundraising_theorem : fundraising_problem 4000 600 := by
  sorry

end NUMINAMATH_CALUDE_fundraising_theorem_l2410_241099


namespace NUMINAMATH_CALUDE_exists_abs_neq_self_l2410_241037

theorem exists_abs_neq_self : ∃ a : ℝ, |a| ≠ a := by
  sorry

end NUMINAMATH_CALUDE_exists_abs_neq_self_l2410_241037


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2410_241036

theorem largest_prime_divisor_to_test (n : ℕ) : 
  1000 ≤ n ∧ n ≤ 1100 → 
  (∀ p : ℕ, p.Prime → p ≤ 31 → n % p ≠ 0) → 
  n.Prime ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l2410_241036


namespace NUMINAMATH_CALUDE_multiplication_equality_l2410_241029

theorem multiplication_equality : 469157 * 9999 = 4691116843 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_equality_l2410_241029


namespace NUMINAMATH_CALUDE_hannah_friday_distance_l2410_241067

/-- The distance Hannah ran on Monday in kilometers -/
def monday_km : ℝ := 9

/-- The distance Hannah ran on Wednesday in meters -/
def wednesday_m : ℝ := 4816

/-- The additional distance Hannah ran on Monday compared to Wednesday and Friday combined, in meters -/
def additional_m : ℝ := 2089

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

theorem hannah_friday_distance :
  ∃ (friday_m : ℝ),
    (monday_km * km_to_m = wednesday_m + friday_m + additional_m) ∧
    friday_m = 2095 := by
  sorry

end NUMINAMATH_CALUDE_hannah_friday_distance_l2410_241067


namespace NUMINAMATH_CALUDE_probability_is_one_twelfth_l2410_241064

/-- Represents the outcome of rolling two 6-sided dice -/
def DiceRoll := Fin 6 × Fin 6

/-- Calculates the sum of a dice roll -/
def sum_roll (roll : DiceRoll) : Nat :=
  (roll.1.val + 1) + (roll.2.val + 1)

/-- Represents the sample space of all possible dice rolls -/
def sample_space : Finset DiceRoll :=
  Finset.product (Finset.univ : Finset (Fin 6)) (Finset.univ : Finset (Fin 6))

/-- Checks if the area of a circle is less than its circumference given its diameter -/
def area_less_than_circumference (d : Nat) : Bool :=
  d * d < 4 * d

/-- The set of favorable outcomes -/
def favorable_outcomes : Finset DiceRoll :=
  sample_space.filter (λ roll => area_less_than_circumference (sum_roll roll))

/-- The probability of the area being less than the circumference -/
def probability : ℚ :=
  (favorable_outcomes.card : ℚ) / (sample_space.card : ℚ)

theorem probability_is_one_twelfth : probability = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_twelfth_l2410_241064


namespace NUMINAMATH_CALUDE_determinant_in_terms_of_coefficients_l2410_241071

theorem determinant_in_terms_of_coefficients 
  (s p q : ℝ) (a b c : ℝ) 
  (h1 : a^3 + s*a^2 + p*a + q = 0)
  (h2 : b^3 + s*b^2 + p*b + q = 0)
  (h3 : c^3 + s*c^2 + p*c + q = 0) :
  let M : Matrix (Fin 3) (Fin 3) ℝ := !![1+a, 1, 1; 1, 1+b, 1; 1, 1, 1+c]
  Matrix.det M = -q + p - s := by
  sorry

end NUMINAMATH_CALUDE_determinant_in_terms_of_coefficients_l2410_241071


namespace NUMINAMATH_CALUDE_largest_non_prime_consecutive_l2410_241006

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_integers (start : ℕ) (count : ℕ) : List ℕ :=
  List.range count |>.map (λ i => start + i)

theorem largest_non_prime_consecutive :
  ∃ (start : ℕ),
    start + 5 = 35 ∧
    start < 50 ∧
    (∀ n ∈ consecutive_integers start 6, n < 50 ∧ ¬(is_prime n)) ∧
    (∀ m : ℕ, m > start + 5 →
      ¬(∃ s : ℕ, s + 5 = m ∧
        s < 50 ∧
        (∀ n ∈ consecutive_integers s 6, n < 50 ∧ ¬(is_prime n)))) :=
sorry

end NUMINAMATH_CALUDE_largest_non_prime_consecutive_l2410_241006


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l2410_241084

theorem geometric_series_first_term
  (S : ℝ)
  (sum_first_two : ℝ)
  (h1 : S = 10)
  (h2 : sum_first_two = 7) :
  ∃ (a : ℝ), (a = 10 * (1 - Real.sqrt (3 / 10)) ∨ a = 10 * (1 + Real.sqrt (3 / 10))) ∧
             (∃ (r : ℝ), S = a / (1 - r) ∧ sum_first_two = a + a * r) :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l2410_241084


namespace NUMINAMATH_CALUDE_roots_expression_l2410_241002

theorem roots_expression (p q : ℝ) (α β γ δ : ℝ) 
  (hα : α^2 + p*α - 2 = 0)
  (hβ : β^2 + p*β - 2 = 0)
  (hγ : γ^2 + q*γ - 2 = 0)
  (hδ : δ^2 + q*δ - 2 = 0) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -2 * (q^2 - p^2) := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_l2410_241002


namespace NUMINAMATH_CALUDE_max_sum_roots_l2410_241079

/-- Given real numbers b and c, and function f(x) = x^2 + bx + c,
    if f(f(x)) = 0 has exactly three different real roots,
    then the maximum value of the sum of the roots of f(x) is 1/2. -/
theorem max_sum_roots (b c : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + b*x + c
  (∃! (r₁ r₂ r₃ : ℝ), r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃ ∧ f (f r₁) = 0 ∧ f (f r₂) = 0 ∧ f (f r₃) = 0) →
  (∃ (α β : ℝ), f α = 0 ∧ f β = 0 ∧ α + β = -b) →
  (∀ (x y : ℝ), f x = 0 ∧ f y = 0 → x + y ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_roots_l2410_241079


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2410_241072

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a + b + c = -a*b*c) :
  (a^2*b^2)/((a^2+b*c)*(b^2+a*c)) + (a^2*c^2)/((a^2+b*c)*(c^2+a*b)) +
  (b^2*c^2)/((b^2+a*c)*(c^2+a*b)) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l2410_241072


namespace NUMINAMATH_CALUDE_garden_ratio_l2410_241045

def garden_problem (table_price bench_price : ℕ) : Prop :=
  table_price + bench_price = 450 ∧
  ∃ k : ℕ, table_price = k * bench_price ∧
  bench_price = 150

theorem garden_ratio :
  ∀ table_price bench_price : ℕ,
  garden_problem table_price bench_price →
  table_price / bench_price = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l2410_241045


namespace NUMINAMATH_CALUDE_team_a_champion_probability_l2410_241080

/-- The probability of a team winning a single game -/
def game_win_prob : ℝ := 0.5

/-- The number of games Team A needs to win to become champion -/
def team_a_games_needed : ℕ := 1

/-- The number of games Team B needs to win to become champion -/
def team_b_games_needed : ℕ := 2

/-- The probability of Team A becoming the champion -/
def team_a_champion_prob : ℝ := 1 - game_win_prob ^ team_b_games_needed

theorem team_a_champion_probability :
  team_a_champion_prob = 0.75 := by sorry

end NUMINAMATH_CALUDE_team_a_champion_probability_l2410_241080


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l2410_241059

/-- The number of distinct arrangements of the letters in the word BANANA -/
def banana_arrangements : ℕ := 180

/-- The total number of letters in the word BANANA -/
def total_letters : ℕ := 6

/-- The number of A's in the word BANANA -/
def num_a : ℕ := 3

/-- The number of N's in the word BANANA -/
def num_n : ℕ := 2

/-- The number of B's in the word BANANA -/
def num_b : ℕ := 1

/-- Theorem stating that the number of distinct arrangements of the letters in BANANA is 180 -/
theorem banana_arrangements_count :
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial num_a * Nat.factorial num_n) :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l2410_241059


namespace NUMINAMATH_CALUDE_divisibility_conditions_l2410_241031

theorem divisibility_conditions (n : ℤ) :
  (∃ k : ℤ, 3 ∣ (5*n^2 + 10*n + 8) ↔ n = 2 + 3*k) ∧
  (∃ k : ℤ, 4 ∣ (5*n^2 + 10*n + 8) ↔ n = 2*k) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l2410_241031


namespace NUMINAMATH_CALUDE_reseat_ten_women_l2410_241015

def reseat_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | k + 3 => reseat_ways (k + 2) + reseat_ways (k + 1)

theorem reseat_ten_women :
  reseat_ways 10 = 89 :=
by sorry

end NUMINAMATH_CALUDE_reseat_ten_women_l2410_241015


namespace NUMINAMATH_CALUDE_parabola_properties_l2410_241034

-- Define the parabola
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ → Prop := fun x y ↦ y^2 = 2 * a * x

-- Define the properties of the parabola C
def C : Parabola where
  a := 1  -- This makes the equation y² = 2x

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through (2,2)
  C.equation 2 2 ∧
  -- The focus is on the x-axis at (1/2, 0)
  C.equation (1/2) 0 ∧
  -- The intersection with x - y - 1 = 0 gives |MN| = 2√6
  ∃ (x₁ x₂ : ℝ),
    C.equation x₁ (x₁ - 1) ∧
    C.equation x₂ (x₂ - 1) ∧
    x₁ ≠ x₂ ∧
    (x₂ - x₁)^2 + ((x₂ - 1) - (x₁ - 1))^2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2410_241034


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l2410_241070

theorem triangle_angle_measure (A B C : ℝ) (exterior_angle : ℝ) :
  -- An exterior angle of triangle ABC is 110°
  exterior_angle = 110 →
  -- ∠A = ∠B
  A = B →
  -- Triangle inequality (to ensure it's a valid triangle)
  A + B + C = 180 →
  -- Prove that ∠A is either 70° or 55°
  A = 70 ∨ A = 55 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l2410_241070


namespace NUMINAMATH_CALUDE_solution_to_equation_l2410_241060

theorem solution_to_equation : ∃ x : ℝ, 0.2 * x + (0.6 * 0.8) = 0.56 ∧ x = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l2410_241060


namespace NUMINAMATH_CALUDE_y_intercept_of_perpendicular_line_l2410_241010

-- Define line l
def line_l (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Define perpendicularity of two lines given their slopes
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Define a point on a line given its slope and a point it passes through
def point_on_line (m x₀ y₀ y : ℝ) (x : ℝ) : Prop := y - y₀ = m * (x - x₀)

-- Theorem statement
theorem y_intercept_of_perpendicular_line :
  ∃ (m : ℝ), 
    (∀ x y, line_l x y → y = (1/2) * x + (1/2)) →
    perpendicular (1/2) m →
    point_on_line m (-1) 0 0 0 →
    ∃ y, point_on_line m 0 y 0 0 ∧ y = -2 := by
  sorry

end NUMINAMATH_CALUDE_y_intercept_of_perpendicular_line_l2410_241010


namespace NUMINAMATH_CALUDE_function_composition_ratio_l2410_241011

def f (x : ℝ) : ℝ := 3 * x + 2

def g (x : ℝ) : ℝ := 2 * x - 3

theorem function_composition_ratio :
  f (g (f 2)) / g (f (g 2)) = 41 / 7 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_ratio_l2410_241011


namespace NUMINAMATH_CALUDE_least_integer_satisfying_inequality_negative_two_satisfies_inequality_least_integer_is_negative_two_l2410_241091

theorem least_integer_satisfying_inequality :
  ∀ x : ℤ, (3 * |2 * x - 1| + 6 < 24) → x ≥ -2 :=
by
  sorry

theorem negative_two_satisfies_inequality :
  3 * |2 * (-2) - 1| + 6 < 24 :=
by
  sorry

theorem least_integer_is_negative_two :
  ∃ x : ℤ, (3 * |2 * x - 1| + 6 < 24) ∧ (∀ y : ℤ, y < x → 3 * |2 * y - 1| + 6 ≥ 24) ∧ x = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_least_integer_satisfying_inequality_negative_two_satisfies_inequality_least_integer_is_negative_two_l2410_241091


namespace NUMINAMATH_CALUDE_probability_of_specific_hand_l2410_241022

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of cards drawn -/
def NumDraws : ℕ := 5

/-- Number of Aces in a standard deck -/
def NumAces : ℕ := 4

/-- Number of suits in a standard deck -/
def NumSuits : ℕ := 4

/-- Probability of the specific outcome -/
def SpecificOutcomeProbability : ℚ := 3 / 832

theorem probability_of_specific_hand :
  let prob_ace : ℚ := NumAces / StandardDeck
  let prob_non_ace_suit : ℚ := (StandardDeck - NumAces) / StandardDeck
  let prob_specific_suit : ℚ := (StandardDeck / NumSuits) / StandardDeck
  NumSuits * prob_ace * prob_non_ace_suit * prob_specific_suit * prob_specific_suit = SpecificOutcomeProbability :=
sorry

end NUMINAMATH_CALUDE_probability_of_specific_hand_l2410_241022


namespace NUMINAMATH_CALUDE_sum_of_products_l2410_241027

theorem sum_of_products (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 1) (h2 : a + b + c = 0) :
  a * b + a * c + b * c = -1/2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_l2410_241027


namespace NUMINAMATH_CALUDE_girl_scout_cookies_l2410_241096

theorem girl_scout_cookies (boxes_per_case : ℕ) (boxes_sold : ℕ) (unpacked_boxes : ℕ) :
  boxes_per_case = 12 →
  boxes_sold > 0 →
  unpacked_boxes = 7 →
  ∃ n : ℕ, boxes_sold = 12 * n + 7 :=
by sorry

end NUMINAMATH_CALUDE_girl_scout_cookies_l2410_241096


namespace NUMINAMATH_CALUDE_three_distinct_triangles60_l2410_241069

/-- A triangle with integer side lengths and one 60° angle -/
structure Triangle60 where
  a : ℕ
  b : ℕ
  c : ℕ
  coprime : Nat.gcd a (Nat.gcd b c) = 1
  angle60 : a^2 + b^2 + c^2 = 2 * max a (max b c)^2

/-- The existence of at least three distinct triangles with integer side lengths and one 60° angle -/
theorem three_distinct_triangles60 : ∃ (t1 t2 t3 : Triangle60),
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t2 ≠ t3 ∧
  (t1.a, t1.b, t1.c) ≠ (5, 7, 8) ∧
  (t2.a, t2.b, t2.c) ≠ (5, 7, 8) ∧
  (t3.a, t3.b, t3.c) ≠ (5, 7, 8) :=
sorry

end NUMINAMATH_CALUDE_three_distinct_triangles60_l2410_241069


namespace NUMINAMATH_CALUDE_least_froods_to_drop_l2410_241038

def score_dropping (n : ℕ) : ℕ := n * (n + 1) / 2

def score_eating (n : ℕ) : ℕ := 15 * n

theorem least_froods_to_drop : 
  ∀ k < 30, score_dropping k ≤ score_eating k ∧ 
  score_dropping 30 > score_eating 30 := by
  sorry

end NUMINAMATH_CALUDE_least_froods_to_drop_l2410_241038


namespace NUMINAMATH_CALUDE_log_product_l2410_241008

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_product (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  lg (m * n) = lg m + lg n :=
sorry

end NUMINAMATH_CALUDE_log_product_l2410_241008


namespace NUMINAMATH_CALUDE_replaced_girl_weight_l2410_241063

theorem replaced_girl_weight 
  (n : ℕ) 
  (new_weight : ℝ) 
  (avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : new_weight = 94)
  (h3 : avg_increase = 3) : 
  ∃ (old_weight : ℝ), 
    old_weight = new_weight - (n * avg_increase) ∧ 
    old_weight = 70 := by
  sorry

end NUMINAMATH_CALUDE_replaced_girl_weight_l2410_241063


namespace NUMINAMATH_CALUDE_bacteria_growth_example_l2410_241032

/-- The time needed for bacteria to reach a certain population -/
def bacteria_growth_time (initial_count : ℕ) (final_count : ℕ) (growth_factor : ℕ) (growth_time : ℕ) : ℕ :=
  let growth_cycles := (final_count / initial_count).log growth_factor
  growth_cycles * growth_time

/-- Theorem: The time needed for 200 bacteria to reach 145,800 bacteria, 
    given that they triple every 5 hours, is 30 hours. -/
theorem bacteria_growth_example : bacteria_growth_time 200 145800 3 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_example_l2410_241032


namespace NUMINAMATH_CALUDE_quadratic_minimum_minimizer_value_l2410_241081

theorem quadratic_minimum (c : ℝ) : 
  2 * c^2 - 7 * c + 4 ≥ 2 * (7/4)^2 - 7 * (7/4) + 4 := by
  sorry

theorem minimizer_value : 
  ∃ (c : ℝ), ∀ (x : ℝ), 2 * x^2 - 7 * x + 4 ≥ 2 * c^2 - 7 * c + 4 ∧ c = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_minimizer_value_l2410_241081


namespace NUMINAMATH_CALUDE_arcade_ticket_difference_l2410_241040

def arcade_tickets : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ := fun initial toys clothes food accessories =>
  let food_discounted := food * 0.85
  let combined := clothes + food_discounted + accessories
  combined - toys

theorem arcade_ticket_difference : arcade_tickets 250 58 85 60 45.5 = 123.5 := by
  sorry

end NUMINAMATH_CALUDE_arcade_ticket_difference_l2410_241040


namespace NUMINAMATH_CALUDE_weight_of_B_l2410_241077

/-- Given the weights of four people A, B, C, and D, prove that B weighs 50 kg. -/
theorem weight_of_B (W_A W_B W_C W_D : ℝ) : W_B = 50 :=
  by
  have h1 : W_A + W_B + W_C + W_D = 240 := by sorry
  have h2 : W_A + W_B = 110 := by sorry
  have h3 : W_B + W_C = 100 := by sorry
  have h4 : W_C + W_D = 130 := by sorry
  sorry

#check weight_of_B

end NUMINAMATH_CALUDE_weight_of_B_l2410_241077


namespace NUMINAMATH_CALUDE_abc_sum_and_squares_l2410_241030

theorem abc_sum_and_squares (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  (a*b + b*c + c*a = -1/2) ∧ (a^4 + b^4 + c^4 = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_abc_sum_and_squares_l2410_241030


namespace NUMINAMATH_CALUDE_range_of_m_l2410_241043

open Set

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x > m

def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*m*x + 2 - m ≤ 0

theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ Ioo (-2) (-1) ∪ Ici 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2410_241043


namespace NUMINAMATH_CALUDE_m_range_l2410_241066

/-- The function g(x) = mx + 2 -/
def g (m : ℝ) (x : ℝ) : ℝ := m * x + 2

/-- The function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The closed interval [-1, 2] -/
def I : Set ℝ := Set.Icc (-1) 2

theorem m_range :
  (∀ m : ℝ, (∀ x₁ ∈ I, ∃ x₀ ∈ I, g m x₁ = f x₀) ↔ m ∈ Set.Icc (-1) (1/2)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l2410_241066


namespace NUMINAMATH_CALUDE_ball_probabilities_l2410_241052

structure BallBag where
  red_balls : ℕ
  white_balls : ℕ

def initial_bag : BallBag := ⟨3, 2⟩

def total_balls (bag : BallBag) : ℕ := bag.red_balls + bag.white_balls

def P_A1 (bag : BallBag) : ℚ := bag.red_balls / total_balls bag
def P_A2 (bag : BallBag) : ℚ := bag.white_balls / total_balls bag

def P_B (bag : BallBag) : ℚ :=
  (P_A1 bag * (bag.red_balls - 1) / (total_balls bag - 1)) +
  (P_A2 bag * (bag.white_balls - 1) / (total_balls bag - 1))

def P_C_given_A2 (bag : BallBag) : ℚ := bag.red_balls / (total_balls bag - 1)

theorem ball_probabilities (bag : BallBag) :
  P_A1 bag + P_A2 bag = 1 ∧
  P_B initial_bag = 2/5 ∧
  P_C_given_A2 initial_bag = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_ball_probabilities_l2410_241052


namespace NUMINAMATH_CALUDE_eating_contest_l2410_241016

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ) 
  (noah_burgers jacob_pies mason_hotdogs : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  noah_burgers = 8 →
  mason_hotdogs * hot_dog_weight = 30 →
  mason_hotdogs = 15 := by
  sorry

end NUMINAMATH_CALUDE_eating_contest_l2410_241016


namespace NUMINAMATH_CALUDE_exactly_two_more_heads_probability_l2410_241082

/-- The number of coins being flipped -/
def num_coins : ℕ := 10

/-- The number of heads required to have exactly two more heads than tails -/
def required_heads : ℕ := (num_coins + 2) / 2

/-- The probability of getting heads on a single fair coin flip -/
def prob_heads : ℚ := 1 / 2

theorem exactly_two_more_heads_probability :
  (Nat.choose num_coins required_heads : ℚ) * prob_heads ^ required_heads * (1 - prob_heads) ^ (num_coins - required_heads) = 210 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_more_heads_probability_l2410_241082


namespace NUMINAMATH_CALUDE_percentage_problem_l2410_241097

theorem percentage_problem (x : ℝ) (h : 0.8 * x = 240) : 0.2 * x = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2410_241097


namespace NUMINAMATH_CALUDE_largest_two_digit_number_with_condition_l2410_241090

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  tens_valid : tens ≥ 1 ∧ tens ≤ 9
  units_valid : units ≥ 0 ∧ units ≤ 9

/-- Checks if a two-digit number satisfies the given condition -/
def satisfiesCondition (n : TwoDigitNumber) : Prop :=
  n.tens * n.units = n.tens + n.units + 17

/-- Theorem: 74 is the largest two-digit number satisfying the condition -/
theorem largest_two_digit_number_with_condition :
  ∀ n : TwoDigitNumber, satisfiesCondition n → n.tens * 10 + n.units ≤ 74 := by
  sorry

#check largest_two_digit_number_with_condition

end NUMINAMATH_CALUDE_largest_two_digit_number_with_condition_l2410_241090


namespace NUMINAMATH_CALUDE_aluminum_sulfide_weight_l2410_241098

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight (al_weight s_weight : ℝ) : ℝ :=
  2 * al_weight + 3 * s_weight

/-- The weight of a given number of moles of a compound -/
def molar_weight (moles molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

theorem aluminum_sulfide_weight :
  let al_weight : ℝ := 26.98
  let s_weight : ℝ := 32.06
  let moles : ℝ := 4
  molar_weight moles (molecular_weight al_weight s_weight) = 600.56 := by
sorry

end NUMINAMATH_CALUDE_aluminum_sulfide_weight_l2410_241098


namespace NUMINAMATH_CALUDE_line_intersects_circle_l2410_241019

/-- The line ax-y+2a=0 (a∈R) intersects the circle x^2+y^2=5 -/
theorem line_intersects_circle (a : ℝ) : 
  ∃ (x y : ℝ), (a * x - y + 2 * a = 0) ∧ (x^2 + y^2 = 5) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l2410_241019


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l2410_241076

theorem average_of_four_numbers (x y z w : ℝ) 
  (h : (5 / 2) * (x + y + z + w) = 25) : 
  (x + y + z + w) / 4 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l2410_241076


namespace NUMINAMATH_CALUDE_greater_solution_of_quadratic_l2410_241005

theorem greater_solution_of_quadratic (x : ℝ) :
  x^2 - 5*x - 84 = 0 → x ≤ 12 :=
by
  sorry

end NUMINAMATH_CALUDE_greater_solution_of_quadratic_l2410_241005


namespace NUMINAMATH_CALUDE_equation_solution_l2410_241001

theorem equation_solution (k : ℝ) : (∃ x : ℝ, 2 * x + k - 3 = 6 ∧ x = 3) → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2410_241001


namespace NUMINAMATH_CALUDE_orange_book_pages_l2410_241042

/-- Proves that the number of pages in each orange book is 510, given the specified conditions --/
theorem orange_book_pages : ℕ → Prop :=
  fun (x : ℕ) =>
    let purple_pages_per_book : ℕ := 230
    let purple_books_read : ℕ := 5
    let orange_books_read : ℕ := 4
    let extra_orange_pages : ℕ := 890
    (purple_pages_per_book * purple_books_read + extra_orange_pages = orange_books_read * x) →
    x = 510

/-- The proof of the theorem --/
lemma prove_orange_book_pages : orange_book_pages 510 := by
  sorry

end NUMINAMATH_CALUDE_orange_book_pages_l2410_241042


namespace NUMINAMATH_CALUDE_questionnaire_responses_l2410_241053

theorem questionnaire_responses (response_rate : ℝ) (min_questionnaires : ℝ) : 
  response_rate = 0.62 → min_questionnaires = 483.87 → 
  ⌊(⌈min_questionnaires⌉ : ℝ) * response_rate⌋ = 300 := by
sorry

end NUMINAMATH_CALUDE_questionnaire_responses_l2410_241053


namespace NUMINAMATH_CALUDE_amy_final_money_l2410_241050

-- Define the initial conditions
def initial_money : ℕ := 2
def num_neighbors : ℕ := 5
def chore_pay : ℕ := 13
def birthday_money : ℕ := 3
def toy_cost : ℕ := 12

-- Define the calculation steps
def money_after_chores : ℕ := initial_money + num_neighbors * chore_pay
def money_after_birthday : ℕ := money_after_chores + birthday_money
def money_after_toy : ℕ := money_after_birthday - toy_cost
def grandparents_gift : ℕ := 2 * money_after_toy

-- Theorem to prove
theorem amy_final_money :
  money_after_toy + grandparents_gift = 174 := by
  sorry


end NUMINAMATH_CALUDE_amy_final_money_l2410_241050


namespace NUMINAMATH_CALUDE_sum_bc_value_l2410_241057

theorem sum_bc_value (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 40)
  (h2 : a + d = 6)
  (h3 : a ≠ d) :
  b + c = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_bc_value_l2410_241057


namespace NUMINAMATH_CALUDE_journey_distance_l2410_241094

/-- Represents the journey of Jack and Peter -/
structure Journey where
  speed : ℝ
  distHomeToStore : ℝ
  distStoreToPeter : ℝ
  distPeterToStore : ℝ

/-- The total distance of the journey -/
def Journey.totalDistance (j : Journey) : ℝ :=
  j.distHomeToStore + j.distStoreToPeter + j.distPeterToStore

/-- Theorem stating the total distance of the journey -/
theorem journey_distance (j : Journey) 
  (h1 : j.speed > 0)
  (h2 : j.distStoreToPeter = 50)
  (h3 : j.distPeterToStore = 50)
  (h4 : j.distHomeToStore / j.speed = 2 * (j.distStoreToPeter / j.speed)) :
  j.totalDistance = 150 := by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_journey_distance_l2410_241094


namespace NUMINAMATH_CALUDE_triangle_circumcircle_radius_l2410_241033

theorem triangle_circumcircle_radius (a b c : ℝ) (h1 : a = 3) (h2 : b = 5) (h3 : c = 7) :
  let R := c / (2 * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2 * a * b))^2))
  R = 7 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_circumcircle_radius_l2410_241033


namespace NUMINAMATH_CALUDE_scientific_notation_of_70_62_million_l2410_241044

theorem scientific_notation_of_70_62_million :
  (70.62 : ℝ) * 1000000 = 7.062 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_70_62_million_l2410_241044


namespace NUMINAMATH_CALUDE_recipe_flour_amount_l2410_241085

/-- A baking recipe with flour and sugar -/
structure Recipe where
  flour : ℕ
  sugar : ℕ

/-- The amount of flour Mary has already added -/
def flour_added : ℕ := 2

/-- The amount of flour Mary still needs to add -/
def flour_to_add : ℕ := 7

/-- The recipe Mary is using -/
def marys_recipe : Recipe := {
  flour := flour_added + flour_to_add,
  sugar := 3
}

/-- Theorem: The total amount of flour in the recipe is equal to the sum of the flour already added and the flour to be added -/
theorem recipe_flour_amount : marys_recipe.flour = flour_added + flour_to_add := by
  sorry

end NUMINAMATH_CALUDE_recipe_flour_amount_l2410_241085


namespace NUMINAMATH_CALUDE_smallest_a_value_l2410_241012

/-- Given a polynomial x^3 - ax^2 + bx - 2550 with three positive integer roots,
    the smallest possible value of a is 62 -/
theorem smallest_a_value (a b : ℤ) (r s t : ℕ+) : 
  (∀ x, x^3 - a*x^2 + b*x - 2550 = (x - r.val)*(x - s.val)*(x - t.val)) →
  a = r.val + s.val + t.val →
  62 ≤ a :=
sorry

end NUMINAMATH_CALUDE_smallest_a_value_l2410_241012


namespace NUMINAMATH_CALUDE_number_difference_l2410_241061

theorem number_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) :
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2410_241061


namespace NUMINAMATH_CALUDE_expand_product_l2410_241093

theorem expand_product (x : ℝ) : (x^2 - 3*x + 3) * (x^2 + 3*x + 3) = x^4 - 3*x^2 + 9 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l2410_241093
