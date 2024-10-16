import Mathlib

namespace NUMINAMATH_CALUDE_room_length_calculation_l93_9316

/-- Given a rectangular room with width 12 m, surrounded by a 2 m wide veranda on all sides,
    and the area of the veranda being 148 m², the length of the room is 21 m. -/
theorem room_length_calculation (room_width : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_width = 12 →
  veranda_width = 2 →
  veranda_area = 148 →
  ∃ (room_length : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - 
    room_length * room_width = veranda_area ∧
    room_length = 21 :=
by sorry

end NUMINAMATH_CALUDE_room_length_calculation_l93_9316


namespace NUMINAMATH_CALUDE_product_of_base8_digits_9876_l93_9310

/-- Converts a natural number from base 10 to base 8 -/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers -/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem product_of_base8_digits_9876 :
  productOfList (toBase8 9876) = 96 :=
by sorry

end NUMINAMATH_CALUDE_product_of_base8_digits_9876_l93_9310


namespace NUMINAMATH_CALUDE_A_intersect_B_l93_9369

def A : Set ℕ := {1, 2, 3, 4}

def B : Set ℕ := {x | ∃ a ∈ A, x = 2*a - 1}

theorem A_intersect_B : A ∩ B = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_l93_9369


namespace NUMINAMATH_CALUDE_range_of_positive_values_l93_9390

def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem range_of_positive_values 
  (f : ℝ → ℝ) 
  (odd : OddFunction f)
  (incr_neg : ∀ x y, x < y ∧ y ≤ 0 → f x < f y)
  (f_neg_one_zero : f (-1) = 0) :
  {x : ℝ | f x > 0} = Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioi 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_positive_values_l93_9390


namespace NUMINAMATH_CALUDE_sin_plus_tan_special_angle_l93_9397

/-- 
If the terminal side of angle α passes through point (4,-3), 
then sin α + tan α = -27/20 
-/
theorem sin_plus_tan_special_angle (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) → 
  Real.sin α + Real.tan α = -27/20 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_tan_special_angle_l93_9397


namespace NUMINAMATH_CALUDE_max_value_on_circle_l93_9314

theorem max_value_on_circle : 
  ∀ (x y : ℝ), (x - 2)^2 + (y - 2)^2 = 2 → x + 2*y ≤ 6 + Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_on_circle_l93_9314


namespace NUMINAMATH_CALUDE_solution_set_eq_neg_reals_l93_9362

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Axiom: f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- Given conditions
axiom condition1 : ∀ x, f x + f' x < 1
axiom condition2 : f 0 = 2016

-- Define the solution set
def solution_set : Set ℝ := {x | Real.exp x * f x - Real.exp x > 2015}

-- Theorem statement
theorem solution_set_eq_neg_reals : solution_set f = Set.Iio 0 := by sorry

end NUMINAMATH_CALUDE_solution_set_eq_neg_reals_l93_9362


namespace NUMINAMATH_CALUDE_rectangular_field_area_l93_9380

theorem rectangular_field_area : 
  let length : ℝ := 5.9
  let width : ℝ := 3
  length * width = 17.7 := by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l93_9380


namespace NUMINAMATH_CALUDE_six_digit_number_representation_l93_9306

theorem six_digit_number_representation (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (1000 ≤ b ∧ b < 10000) →  -- b is a four-digit number
  (100000 ≤ 10000 * a + b ∧ 10000 * a + b < 1000000) →  -- result is a six-digit number
  10000 * a + b = 10000 * a + b :=  -- the representation is correct
by sorry

end NUMINAMATH_CALUDE_six_digit_number_representation_l93_9306


namespace NUMINAMATH_CALUDE_calculation_proof_l93_9331

theorem calculation_proof : 2⁻¹ + Real.sin (30 * π / 180) - (π - 3.14)^0 + abs (-3) - Real.sqrt 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l93_9331


namespace NUMINAMATH_CALUDE_hairs_to_grow_back_l93_9354

def hairs_lost_washing : ℕ := 32

def hairs_lost_brushing : ℕ := hairs_lost_washing / 2

def total_hairs_lost : ℕ := hairs_lost_washing + hairs_lost_brushing

theorem hairs_to_grow_back : total_hairs_lost + 1 = 49 := by sorry

end NUMINAMATH_CALUDE_hairs_to_grow_back_l93_9354


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l93_9344

theorem stratified_sampling_sample_size
  (ratio_10 : ℕ)
  (ratio_11 : ℕ)
  (ratio_12 : ℕ)
  (sample_12 : ℕ)
  (h_ratio : ratio_10 = 2 ∧ ratio_11 = 3 ∧ ratio_12 = 5)
  (h_sample_12 : sample_12 = 150)
  : ∃ (n : ℕ), n = 300 ∧ (ratio_12 : ℚ) / (ratio_10 + ratio_11 + ratio_12 : ℚ) = sample_12 / n :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l93_9344


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l93_9336

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 5 / 4 →  -- The ratio of the angles is 5:4
  b = 80 :=  -- The smaller angle is 80°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l93_9336


namespace NUMINAMATH_CALUDE_green_height_l93_9348

/-- The heights of the dwarves -/
structure DwarfHeights where
  blue : ℝ
  black : ℝ
  yellow : ℝ
  red : ℝ
  green : ℝ

/-- The conditions of the problem -/
def dwarfProblem (h : DwarfHeights) : Prop :=
  h.blue = 88 ∧
  h.black = 84 ∧
  h.yellow = 76 ∧
  (h.blue + h.black + h.yellow + h.red + h.green) / 5 = 81.6 ∧
  ((h.blue + h.black + h.yellow + h.green) / 4) = ((h.blue + h.black + h.yellow + h.red) / 4 - 6)

theorem green_height (h : DwarfHeights) (hc : dwarfProblem h) : h.green = 68 := by
  sorry

end NUMINAMATH_CALUDE_green_height_l93_9348


namespace NUMINAMATH_CALUDE_littering_citations_l93_9304

/-- Represents the number of citations for each category --/
structure Citations where
  littering : ℕ
  offLeash : ℕ
  smoking : ℕ
  parking : ℕ
  camping : ℕ

/-- Conditions for the park warden's citations --/
def citationConditions (c : Citations) : Prop :=
  c.littering = c.offLeash ∧
  c.littering = c.smoking + 5 ∧
  c.parking = 5 * (c.littering + c.offLeash + c.smoking) ∧
  c.camping = 10 ∧
  c.littering + c.offLeash + c.smoking + c.parking + c.camping = 150

/-- Theorem stating that under the given conditions, the number of littering citations is 9 --/
theorem littering_citations (c : Citations) (h : citationConditions c) : c.littering = 9 := by
  sorry


end NUMINAMATH_CALUDE_littering_citations_l93_9304


namespace NUMINAMATH_CALUDE_pet_ownership_l93_9385

theorem pet_ownership (total : ℕ) (dogs cats other_pets no_pets : ℕ) 
  (dogs_cats : ℕ) (dogs_other : ℕ) (cats_other : ℕ) :
  total = 32 →
  dogs = total / 2 →
  cats = total * 3 / 8 →
  other_pets = 6 →
  no_pets = 5 →
  dogs_cats = 10 →
  dogs_other = 2 →
  cats_other = 9 →
  ∃ (all_three : ℕ),
    all_three = 1 ∧
    dogs + cats + other_pets - dogs_cats - dogs_other - cats_other + all_three = total - no_pets :=
by sorry

end NUMINAMATH_CALUDE_pet_ownership_l93_9385


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l93_9363

/-- The solution set of the system of equations {x - 2y = 1, x^3 - 6xy - 8y^3 = 1} 
    is equivalent to the line y = (x-1)/2 -/
theorem solution_set_equivalence (x y : ℝ) : 
  (x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1) ↔ y = (x - 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l93_9363


namespace NUMINAMATH_CALUDE_distance_time_relationship_l93_9361

/-- The relationship between distance and time for a car traveling at 60 km/h -/
theorem distance_time_relationship (s t : ℝ) (h : s = 60 * t) :
  s = 60 * t :=
by sorry

end NUMINAMATH_CALUDE_distance_time_relationship_l93_9361


namespace NUMINAMATH_CALUDE_curve_equation_relationship_l93_9345

-- Define a type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a type for curves in 2D space
structure Curve where
  points : Set Point2D

-- Define a function type for equations in 2D space
def Equation2D := Point2D → Prop

-- Define the given condition
def satisfiesEquation (C : Curve) (f : Equation2D) : Prop :=
  ∀ p ∈ C.points, f p

-- Theorem statement
theorem curve_equation_relationship (C : Curve) (f : Equation2D) :
  satisfiesEquation C f →
  ¬ (∀ p : Point2D, f p ↔ p ∈ C.points) :=
by sorry

end NUMINAMATH_CALUDE_curve_equation_relationship_l93_9345


namespace NUMINAMATH_CALUDE_replaced_man_age_l93_9317

theorem replaced_man_age
  (total_men : Nat)
  (age_increase : Nat)
  (known_man_age : Nat)
  (women_avg_age : Nat)
  (h1 : total_men = 7)
  (h2 : age_increase = 4)
  (h3 : known_man_age = 26)
  (h4 : women_avg_age = 42) :
  ∃ (replaced_man_age : Nat),
    replaced_man_age = 30 ∧
    (∃ (initial_avg : ℚ),
      (total_men : ℚ) * initial_avg =
        (total_men - 2 : ℚ) * (initial_avg + age_increase) +
        2 * women_avg_age -
        (known_man_age + replaced_man_age : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_replaced_man_age_l93_9317


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l93_9365

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Checks if a point lies on a line -/
def pointOnLine (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- The given line 4x + 2y = 8 -/
def givenLine : Line :=
  { slope := -2, intercept := 4 }

/-- The line we need to prove -/
def parallelLine : Line :=
  { slope := -2, intercept := 1 }

theorem parallel_line_through_point :
  parallel parallelLine givenLine ∧
  pointOnLine parallelLine 0 1 :=
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l93_9365


namespace NUMINAMATH_CALUDE_albert_pizza_count_l93_9377

def pizza_problem (large_pizzas small_pizzas : ℕ) 
  (slices_per_large slices_per_small total_slices : ℕ) : Prop :=
  large_pizzas = 2 ∧ 
  slices_per_large = 16 ∧ 
  slices_per_small = 8 ∧ 
  total_slices = 48 ∧
  small_pizzas * slices_per_small = total_slices - (large_pizzas * slices_per_large)

theorem albert_pizza_count : 
  ∃ (large_pizzas small_pizzas slices_per_large slices_per_small total_slices : ℕ),
    pizza_problem large_pizzas small_pizzas slices_per_large slices_per_small total_slices ∧ 
    small_pizzas = 2 := by
  sorry

end NUMINAMATH_CALUDE_albert_pizza_count_l93_9377


namespace NUMINAMATH_CALUDE_karens_ferns_l93_9375

/-- Proves the number of ferns Karen hung, given the number of fronds per fern, 
    leaves per frond, and total number of leaves. -/
theorem karens_ferns (fronds_per_fern : ℕ) (leaves_per_frond : ℕ) (total_leaves : ℕ) 
    (h1 : fronds_per_fern = 7)
    (h2 : leaves_per_frond = 30)
    (h3 : total_leaves = 1260) :
  total_leaves / (fronds_per_fern * leaves_per_frond) = 6 := by
  sorry

#check karens_ferns

end NUMINAMATH_CALUDE_karens_ferns_l93_9375


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l93_9309

theorem right_triangle_inequality (a b c : ℝ) (h : c^2 = a^2 + b^2) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b ≤ c * Real.sqrt 2 ∧ (a + b = c * Real.sqrt 2 ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l93_9309


namespace NUMINAMATH_CALUDE_remaining_numbers_l93_9302

theorem remaining_numbers (total : ℕ) (total_avg : ℚ) (subset : ℕ) (subset_avg : ℚ) (remaining_avg : ℚ) :
  total = 9 →
  total_avg = 18 →
  subset = 4 →
  subset_avg = 8 →
  remaining_avg = 26 →
  total - subset = (total * total_avg - subset * subset_avg) / remaining_avg :=
by
  sorry

#eval (9 : ℕ) - 4  -- Expected output: 5

end NUMINAMATH_CALUDE_remaining_numbers_l93_9302


namespace NUMINAMATH_CALUDE_pepperoni_count_l93_9366

/-- Represents a pizza with pepperoni slices -/
structure Pizza :=
  (total_slices : ℕ)

/-- Represents a quarter of a pizza -/
def QuarterPizza := Pizza

theorem pepperoni_count (p : Pizza) (q : QuarterPizza) :
  (p.total_slices = 4 * q.total_slices) →
  (q.total_slices = 10) →
  (p.total_slices = 40) := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_count_l93_9366


namespace NUMINAMATH_CALUDE_min_value_theorem_l93_9346

theorem min_value_theorem (x : ℝ) (h : x > 1) :
  3 * x + 1 / (x - 1) ≥ 2 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l93_9346


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l93_9315

theorem line_intercepts_sum (c : ℚ) : 
  (∃ x y : ℚ, 4 * x + 7 * y + 3 * c = 0 ∧ x + y = 11) → c = -308/33 := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l93_9315


namespace NUMINAMATH_CALUDE_five_three_number_properties_l93_9382

/-- Definition of a "five-three number" -/
def is_five_three_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    n = a * 1000 + b * 100 + c * 10 + d ∧
    a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 ∧ d ≥ 0 ∧ d ≤ 9 ∧
    a = c + 5 ∧ b = d + 3

/-- Definition of M(A) -/
def M (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a + c + 2 * (b + d)

/-- Definition of N(A) -/
def N (n : ℕ) : ℤ :=
  (n / 100 % 10) - 3

theorem five_three_number_properties :
  (∃ (max min : ℕ),
    is_five_three_number max ∧
    is_five_three_number min ∧
    (∀ n, is_five_three_number n → n ≤ max ∧ n ≥ min) ∧
    max - min = 4646) ∧
  (∃ A : ℕ,
    is_five_three_number A ∧
    (M A) % (N A) = 0 ∧
    A = 5401) :=
sorry

end NUMINAMATH_CALUDE_five_three_number_properties_l93_9382


namespace NUMINAMATH_CALUDE_constant_function_shifted_l93_9329

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 5

-- State the theorem
theorem constant_function_shifted (x : ℝ) : f (x + 3) = 5 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_shifted_l93_9329


namespace NUMINAMATH_CALUDE_magnitude_of_z_l93_9395

theorem magnitude_of_z (z : ℂ) (h : z * (1 - 2*Complex.I) = 4 + 2*Complex.I) : 
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l93_9395


namespace NUMINAMATH_CALUDE_problem_solution_l93_9334

theorem problem_solution (m n : ℕ) 
  (h_pos_m : m > 0) 
  (h_pos_n : n > 0) 
  (h_inequality : m + 8 < n - 1) 
  (h_mean : (m + (m + 3) + (m + 8) + (n - 1) + (n + 3) + (2 * n - 2)) / 6 = n) 
  (h_median : (m + 8 + n - 1) / 2 = n) : 
  m + n = 47 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l93_9334


namespace NUMINAMATH_CALUDE_cubic_sum_ratio_l93_9364

theorem cubic_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_ratio_l93_9364


namespace NUMINAMATH_CALUDE_independent_x_implies_result_l93_9319

theorem independent_x_implies_result (m n : ℝ) :
  (∀ x y : ℝ, ∃ k : ℝ, (m*x^2 + 3*x - y) - (4*x^2 - (2*n + 3)*x + 3*y - 2) = k) →
  (m - n) + |m*n| = 19 := by
sorry

end NUMINAMATH_CALUDE_independent_x_implies_result_l93_9319


namespace NUMINAMATH_CALUDE_junior_count_in_club_l93_9381

theorem junior_count_in_club (total_students : ℕ) 
  (junior_chess_percent senior_chess_percent : ℚ)
  (h1 : total_students = 36)
  (h2 : junior_chess_percent = 2/5)
  (h3 : senior_chess_percent = 1/5)
  (h4 : ∃ (j s : ℕ), j + s = total_students ∧ 
    junior_chess_percent * j = senior_chess_percent * s) :
  ∃ (j : ℕ), j + (total_students - j) = total_students ∧ 
    junior_chess_percent * j = senior_chess_percent * (total_students - j) ∧
    j = 12 := by
sorry

end NUMINAMATH_CALUDE_junior_count_in_club_l93_9381


namespace NUMINAMATH_CALUDE_b_to_c_interest_rate_b_to_c_interest_rate_is_12_percent_l93_9358

/-- The interest rate at which B lent money to C, given the following conditions:
  * A lends Rs. 3500 to B at 10% per annum
  * B lends the same sum to C
  * B's gain over 3 years is Rs. 210
-/
theorem b_to_c_interest_rate : ℝ :=
  let principal : ℝ := 3500
  let a_to_b_rate : ℝ := 0.1
  let time : ℝ := 3
  let b_gain : ℝ := 210
  let a_to_b_interest : ℝ := principal * a_to_b_rate * time
  let total_interest_from_c : ℝ := a_to_b_interest + b_gain
  total_interest_from_c / (principal * time)

/-- Proof that the interest rate at which B lent money to C is 12% per annum -/
theorem b_to_c_interest_rate_is_12_percent : b_to_c_interest_rate = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_b_to_c_interest_rate_b_to_c_interest_rate_is_12_percent_l93_9358


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_expansion_l93_9330

/-- The repeating decimal 0.73246̅ expressed as a fraction with denominator 999900 -/
def repeating_decimal : ℚ :=
  731514 / 999900

/-- The repeating decimal 0.73246̅ as a real number -/
noncomputable def decimal_expansion : ℝ :=
  0.73 + (246 : ℝ) / 1000 * (1 / (1 - 1/1000))

theorem repeating_decimal_equals_expansion :
  (repeating_decimal : ℝ) = decimal_expansion :=
sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_expansion_l93_9330


namespace NUMINAMATH_CALUDE_theater_ticket_difference_l93_9341

theorem theater_ticket_difference :
  ∀ (orchestra_tickets balcony_tickets : ℕ),
  orchestra_tickets + balcony_tickets = 370 →
  12 * orchestra_tickets + 8 * balcony_tickets = 3320 →
  balcony_tickets - orchestra_tickets = 190 :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_difference_l93_9341


namespace NUMINAMATH_CALUDE_correct_calculation_l93_9376

theorem correct_calculation (a b : ℝ) : 3 * a * b - 2 * a * b = a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l93_9376


namespace NUMINAMATH_CALUDE_least_multiple_25_with_digit_product_125_l93_9360

def is_multiple_of_25 (n : ℕ) : Prop := ∃ k : ℕ, n = 25 * k

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  let digits := n.digits 10
  digits.prod

theorem least_multiple_25_with_digit_product_125 :
  ∀ n : ℕ, n > 0 → is_multiple_of_25 n → digit_product n = 125 → n ≥ 555 :=
sorry

end NUMINAMATH_CALUDE_least_multiple_25_with_digit_product_125_l93_9360


namespace NUMINAMATH_CALUDE_total_people_count_l93_9307

theorem total_people_count (num_students : ℕ) (ratio : ℕ) : 
  num_students = 37500 →
  ratio = 15 →
  num_students + (num_students / ratio) = 40000 := by
sorry

end NUMINAMATH_CALUDE_total_people_count_l93_9307


namespace NUMINAMATH_CALUDE_book_cost_problem_l93_9359

theorem book_cost_problem (total_cost : ℝ) (loss_percent : ℝ) (gain_percent : ℝ)
  (h_total : total_cost = 600)
  (h_loss : loss_percent = 15)
  (h_gain : gain_percent = 19) :
  ∃ (cost_loss cost_gain : ℝ),
    cost_loss + cost_gain = total_cost ∧
    cost_loss * (1 - loss_percent / 100) = cost_gain * (1 + gain_percent / 100) ∧
    cost_loss = 350 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_problem_l93_9359


namespace NUMINAMATH_CALUDE_sum_of_distances_constant_l93_9332

/-- A regular pyramid -/
structure RegularPyramid where
  base : Set (Fin 3 → ℝ)  -- Base of the pyramid as a set of points in ℝ³
  apex : Fin 3 → ℝ        -- Apex of the pyramid as a point in ℝ³
  is_regular : Bool       -- Property ensuring the pyramid is regular

/-- A point on the base of the pyramid -/
def BasePoint (pyramid : RegularPyramid) := { p : Fin 3 → ℝ // p ∈ pyramid.base }

/-- The perpendicular line from a point on the base to the base plane -/
def Perpendicular (pyramid : RegularPyramid) (p : BasePoint pyramid) : Set (Fin 3 → ℝ) :=
  sorry

/-- The intersection points of the perpendicular with the face planes -/
def IntersectionPoints (pyramid : RegularPyramid) (p : BasePoint pyramid) : Set (Fin 3 → ℝ) :=
  sorry

/-- The sum of distances from a base point to the intersection points -/
def SumOfDistances (pyramid : RegularPyramid) (p : BasePoint pyramid) : ℝ :=
  sorry

/-- Theorem: The sum of distances is constant for all points on the base -/
theorem sum_of_distances_constant (pyramid : RegularPyramid) :
  ∀ p q : BasePoint pyramid, SumOfDistances pyramid p = SumOfDistances pyramid q :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_constant_l93_9332


namespace NUMINAMATH_CALUDE_average_of_abc_is_three_l93_9374

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 1503 * C - 3006 * A = 6012)
  (eq2 : 1503 * B + 4509 * A = 7509) :
  (A + B + C) / 3 = 3 := by
sorry

end NUMINAMATH_CALUDE_average_of_abc_is_three_l93_9374


namespace NUMINAMATH_CALUDE_bottle_cap_distribution_l93_9370

theorem bottle_cap_distribution (initial : ℕ) (rebecca : ℕ) (siblings : ℕ) : 
  initial = 150 →
  rebecca = 42 →
  siblings = 5 →
  (initial + rebecca + 2 * rebecca) / (siblings + 1) = 46 :=
by sorry

end NUMINAMATH_CALUDE_bottle_cap_distribution_l93_9370


namespace NUMINAMATH_CALUDE_exponential_function_through_point_l93_9394

theorem exponential_function_through_point (f : ℝ → ℝ) :
  (∀ x : ℝ, ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ f x = a^x) →
  f 1 = 2 →
  f 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_exponential_function_through_point_l93_9394


namespace NUMINAMATH_CALUDE_bakery_bread_rolls_l93_9305

/-- Given a bakery with a total of 90 items, 19 croissants, and 22 bagels,
    prove that the number of bread rolls is 49. -/
theorem bakery_bread_rolls :
  let total_items : ℕ := 90
  let croissants : ℕ := 19
  let bagels : ℕ := 22
  let bread_rolls : ℕ := total_items - croissants - bagels
  bread_rolls = 49 := by
  sorry

end NUMINAMATH_CALUDE_bakery_bread_rolls_l93_9305


namespace NUMINAMATH_CALUDE_xy_value_l93_9301

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 18) : x * y = 18 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l93_9301


namespace NUMINAMATH_CALUDE_exists_strictly_increasing_set_function_l93_9357

-- Define the set of positive integers
def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

-- Define the power set of positive integers
def PowerSetOfPositiveIntegers : Set (Set ℕ) :=
  {X : Set ℕ | X ⊆ PositiveIntegers}

-- State the theorem
theorem exists_strictly_increasing_set_function :
  ∃ (f : ℝ → Set ℕ),
    (∀ x, f x ∈ PowerSetOfPositiveIntegers) ∧
    (∀ a b, a < b → f a ⊂ f b ∧ f a ≠ f b) :=
sorry

end NUMINAMATH_CALUDE_exists_strictly_increasing_set_function_l93_9357


namespace NUMINAMATH_CALUDE_division_sum_theorem_l93_9349

theorem division_sum_theorem (quotient divisor remainder : ℝ) :
  quotient = 450 →
  divisor = 350.7 →
  remainder = 287.9 →
  (divisor * quotient) + remainder = 158102.9 := by
  sorry

end NUMINAMATH_CALUDE_division_sum_theorem_l93_9349


namespace NUMINAMATH_CALUDE_car_train_distance_difference_l93_9386

theorem car_train_distance_difference :
  let train_speed : ℝ := 60
  let car_speed : ℝ := 2 * train_speed
  let travel_time : ℝ := 3
  let train_distance : ℝ := train_speed * travel_time
  let car_distance : ℝ := car_speed * travel_time
  car_distance - train_distance = 180 := by
  sorry

end NUMINAMATH_CALUDE_car_train_distance_difference_l93_9386


namespace NUMINAMATH_CALUDE_exists_polynomial_for_cosine_multiple_l93_9312

-- Define Chebyshev polynomials of the first kind
def chebyshev (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => λ _ => 1
  | 1 => λ x => x
  | n + 2 => λ x => 2 * x * chebyshev (n + 1) x - chebyshev n x

-- State the theorem
theorem exists_polynomial_for_cosine_multiple (n : ℕ) (hn : n > 0) :
  ∃ (p : ℝ → ℝ), ∀ x, p (2 * Real.cos x) = 2 * Real.cos (n * x) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_exists_polynomial_for_cosine_multiple_l93_9312


namespace NUMINAMATH_CALUDE_f_5_equals_207_l93_9318

def f (n : ℕ) : ℕ := n^3 + 2*n^2 + 3*n + 17

theorem f_5_equals_207 : f 5 = 207 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_207_l93_9318


namespace NUMINAMATH_CALUDE_min_questions_to_find_z_l93_9320

/-- Represents a person in the company -/
structure Person where
  id : Nat

/-- Represents the company with n people -/
structure Company where
  n : Nat
  people : Finset Person
  z : Person
  knows : Person → Person → Prop

/-- Axioms for the company structure -/
axiom company_size (c : Company) : c.people.card = c.n

axiom z_knows_all (c : Company) (p : Person) : 
  p ∈ c.people → p ≠ c.z → c.knows c.z p

axiom z_known_by_none (c : Company) (p : Person) : 
  p ∈ c.people → p ≠ c.z → ¬(c.knows p c.z)

/-- The main theorem to prove -/
theorem min_questions_to_find_z (c : Company) :
  ∃ (strategy : Nat → Person × Person),
    (∀ k, k < c.n - 1 → 
      (strategy k).1 ∈ c.people ∧ (strategy k).2 ∈ c.people) ∧
    (∀ result : Nat → Bool,
      ∃! p, p ∈ c.people ∧ 
        ∀ k, k < c.n - 1 → 
          result k = c.knows (strategy k).1 (strategy k).2 →
          p ≠ (strategy k).1 ∧ p ≠ (strategy k).2) ∧
  ¬∃ (strategy : Nat → Person × Person),
    (∀ k, k < c.n - 2 → 
      (strategy k).1 ∈ c.people ∧ (strategy k).2 ∈ c.people) ∧
    (∀ result : Nat → Bool,
      ∃! p, p ∈ c.people ∧ 
        ∀ k, k < c.n - 2 → 
          result k = c.knows (strategy k).1 (strategy k).2 →
          p ≠ (strategy k).1 ∧ p ≠ (strategy k).2) :=
by
  sorry

end NUMINAMATH_CALUDE_min_questions_to_find_z_l93_9320


namespace NUMINAMATH_CALUDE_factorization_of_polynomial_l93_9373

theorem factorization_of_polynomial (b : ℝ) :
  348 * b^2 + 87 * b + 261 = 87 * (4 * b^2 + b + 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_polynomial_l93_9373


namespace NUMINAMATH_CALUDE_no_consecutive_squares_arithmetic_sequence_l93_9396

theorem no_consecutive_squares_arithmetic_sequence :
  ∀ (x y z w : ℕ+), ¬∃ (d : ℝ),
    (y : ℝ)^2 = (x : ℝ)^2 + d ∧
    (z : ℝ)^2 = (y : ℝ)^2 + d ∧
    (w : ℝ)^2 = (z : ℝ)^2 + d :=
by sorry

end NUMINAMATH_CALUDE_no_consecutive_squares_arithmetic_sequence_l93_9396


namespace NUMINAMATH_CALUDE_arithmetic_progression_properties_l93_9342

/-- An arithmetic progression with given first and tenth terms -/
def ArithmeticProgression (a : ℕ → ℤ) : Prop :=
  a 1 = 21 ∧ a 10 = 3 ∧ ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_progression_properties (a : ℕ → ℤ) (h : ArithmeticProgression a) :
  (∀ n : ℕ, a n = -2 * n + 23) ∧
  (Finset.sum (Finset.range 11) a = 121) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_properties_l93_9342


namespace NUMINAMATH_CALUDE_min_value_theorem_l93_9343

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 2) :
  (2 / a) + (1 / b) ≥ 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l93_9343


namespace NUMINAMATH_CALUDE_f_eight_minus_f_four_l93_9388

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_eight_minus_f_four (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 5)
  (h1 : f 1 = 1)
  (h2 : f 2 = 3) :
  f 8 - f 4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_eight_minus_f_four_l93_9388


namespace NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_for_beta_l93_9333

-- Define the propositions
def α (x : ℝ) : Prop := |x - 1| ≤ 2
def β (x : ℝ) : Prop := (x - 3) / (x + 1) ≤ 0

-- Theorem statement
theorem alpha_necessary_not_sufficient_for_beta :
  (∀ x, β x → α x) ∧ ¬(∀ x, α x → β x) := by sorry

end NUMINAMATH_CALUDE_alpha_necessary_not_sufficient_for_beta_l93_9333


namespace NUMINAMATH_CALUDE_hundredth_term_is_9999_l93_9368

/-- The nth term of the sequence -/
def sequenceTerm (n : ℕ) : ℕ := n^2 - 1

/-- Theorem: The 100th term of the sequence is 9999 -/
theorem hundredth_term_is_9999 : sequenceTerm 100 = 9999 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_is_9999_l93_9368


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_of_a_for_two_distinct_fixed_points_l93_9383

def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

theorem fixed_points_for_specific_values :
  is_fixed_point (f 1 (-2)) (-1) ∧ is_fixed_point (f 1 (-2)) 3 :=
sorry

theorem range_of_a_for_two_distinct_fixed_points :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) →
  (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_of_a_for_two_distinct_fixed_points_l93_9383


namespace NUMINAMATH_CALUDE_fo_greater_than_di_l93_9379

-- Define the points
variable (F I D O : ℝ × ℝ)

-- Define the quadrilateral FIDO
def is_convex_quadrilateral (F I D O : ℝ × ℝ) : Prop := sorry

-- Define the length of a line segment
def length (A B : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two line segments
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem fo_greater_than_di 
  (h_convex : is_convex_quadrilateral F I D O)
  (h_equal_sides : length F I = length D O)
  (h_fi_greater : length F I > length D I)
  (h_equal_angles : angle F I O = angle D I O) :
  length F O > length D I :=
sorry

end NUMINAMATH_CALUDE_fo_greater_than_di_l93_9379


namespace NUMINAMATH_CALUDE_line_through_three_points_l93_9355

/-- Given a line passing through points (4, 10), (-3, m), and (-12, 5), prove that m = 125/16 -/
theorem line_through_three_points (m : ℚ) : 
  (let slope1 := (m - 10) / (-7 : ℚ)
   let slope2 := (5 - m) / (-9 : ℚ)
   slope1 = slope2) →
  m = 125 / 16 := by
sorry

end NUMINAMATH_CALUDE_line_through_three_points_l93_9355


namespace NUMINAMATH_CALUDE_cube_volume_problem_l93_9384

theorem cube_volume_problem (a : ℕ) : 
  (a - 2) * a * (a + 2) = a^3 - 14 → a^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l93_9384


namespace NUMINAMATH_CALUDE_cubic_real_root_l93_9372

/-- Given a cubic polynomial ax^3 + 3x^2 + bx - 65 = 0 where a and b are real numbers,
    and -2 - 3i is one of its roots, the real root of this polynomial is 5/2. -/
theorem cubic_real_root (a b : ℝ) :
  (∃ (z : ℂ), z = -2 - 3*I ∧ a * z^3 + 3 * z^2 + b * z - 65 = 0) →
  (∃ (x : ℝ), a * x^3 + 3 * x^2 + b * x - 65 = 0 ∧ x = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_real_root_l93_9372


namespace NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l93_9321

theorem sum_of_sqrt_inequality (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_sum : a + b + c = 1) : 
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) < 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l93_9321


namespace NUMINAMATH_CALUDE_square_ends_with_three_identical_nonzero_digits_l93_9350

theorem square_ends_with_three_identical_nonzero_digits : 
  ∃ n : ℤ, ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧ n^2 % 1000 = d * 100 + d * 10 + d :=
sorry

end NUMINAMATH_CALUDE_square_ends_with_three_identical_nonzero_digits_l93_9350


namespace NUMINAMATH_CALUDE_dani_pants_after_five_years_l93_9324

/-- Calculates the total number of pants after a given number of years -/
def totalPantsAfterYears (initialPants : ℕ) (pairsPerYear : ℕ) (pantsPerPair : ℕ) (years : ℕ) : ℕ :=
  initialPants + years * pairsPerYear * pantsPerPair

/-- Theorem: Given the initial conditions, Dani will have 90 pants after 5 years -/
theorem dani_pants_after_five_years :
  totalPantsAfterYears 50 4 2 5 = 90 := by
  sorry

#eval totalPantsAfterYears 50 4 2 5

end NUMINAMATH_CALUDE_dani_pants_after_five_years_l93_9324


namespace NUMINAMATH_CALUDE_multiples_of_nine_count_l93_9323

theorem multiples_of_nine_count (N : ℕ) : 
  (∃ (count : ℕ), count = (Nat.div N 9 - Nat.div 10 9 + 1) ∧ count = 1110) → N = 9989 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_nine_count_l93_9323


namespace NUMINAMATH_CALUDE_area_of_square_e_l93_9398

/-- Given a rectangle composed of squares a, b, c, d, and e, prove the area of square e. -/
theorem area_of_square_e (a b c d e : ℝ) : 
  a + b + c = 30 →
  a + b = 22 →
  2 * c + e = 22 →
  e^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_area_of_square_e_l93_9398


namespace NUMINAMATH_CALUDE_square_is_one_l93_9327

/-- Represents a digit in base-7 --/
def Base7Digit := Fin 7

/-- The addition problem in base-7 --/
def addition_problem (square : Base7Digit) : Prop :=
  ∃ (carry1 carry2 carry3 : Nat),
    (square.val + 1 + 3 + 2) % 7 = 0 ∧
    (carry1 + square.val + 5 + square.val + 1) % 7 = square.val ∧
    (carry2 + 4 + carry3) % 7 = 5 ∧
    carry1 = (square.val + 1 + 3 + 2) / 7 ∧
    carry2 = (carry1 + square.val + 5 + square.val + 1) / 7 ∧
    carry3 = (square.val + 5 + 1) / 7

theorem square_is_one :
  ∃ (square : Base7Digit), addition_problem square ∧ square.val = 1 := by sorry

end NUMINAMATH_CALUDE_square_is_one_l93_9327


namespace NUMINAMATH_CALUDE_f_minimum_and_tangents_l93_9351

noncomputable def f (x : ℝ) : ℝ := x * (Real.log x + 1)

theorem f_minimum_and_tangents 
  (a b : ℝ) 
  (h1 : 0 < b) (h2 : b < a * Real.log a + a) :
  (∃ (min : ℝ), min = -Real.exp (-2) ∧ ∀ x > 0, f x ≥ min) ∧
  (∃ (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    x₁ > Real.exp (-2) ∧ 
    x₂ > Real.exp (-2) ∧
    b - f x₁ = (Real.log x₁ + 2) * (a - x₁) ∧
    b - f x₂ = (Real.log x₂ + 2) * (a - x₂)) :=
sorry

end NUMINAMATH_CALUDE_f_minimum_and_tangents_l93_9351


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l93_9328

theorem square_circle_area_ratio :
  ∀ (r : ℝ) (s : ℝ),
    r > 0 →
    s > 0 →
    s = r * Real.sqrt 15 / 2 →
    (s^2) / (π * r^2) = 15 / (4 * π) := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l93_9328


namespace NUMINAMATH_CALUDE_min_coins_for_distribution_l93_9353

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed -/
theorem min_coins_for_distribution (num_friends : ℕ) (initial_coins : ℕ) 
  (h1 : num_friends = 15) (h2 : initial_coins = 63) :
  min_additional_coins num_friends initial_coins = 57 := by
  sorry

#eval min_additional_coins 15 63

end NUMINAMATH_CALUDE_min_coins_for_distribution_l93_9353


namespace NUMINAMATH_CALUDE_test_questions_l93_9313

theorem test_questions (total_points : ℕ) (five_point_questions : ℕ) (five_point_value : ℕ) (ten_point_value : ℕ) : 
  total_points = 200 →
  five_point_questions = 20 →
  five_point_value = 5 →
  ten_point_value = 10 →
  ∃ (ten_point_questions : ℕ),
    five_point_questions * five_point_value + ten_point_questions * ten_point_value = total_points ∧
    five_point_questions + ten_point_questions = 30 :=
by sorry

end NUMINAMATH_CALUDE_test_questions_l93_9313


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l93_9393

theorem system_of_inequalities_solution (x : ℝ) :
  (x - 1 ≤ x / 2 ∧ x + 2 > 3 * (x - 2)) ↔ x ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l93_9393


namespace NUMINAMATH_CALUDE_circle_coverage_l93_9387

-- Define a circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if one circle can cover another
def canCover (c1 c2 : Circle) : Prop :=
  ∀ p : ℝ × ℝ, (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 ≤ c2.radius^2 →
    (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 ≤ c1.radius^2

-- Theorem statement
theorem circle_coverage (M1 M2 : Circle) (h : M2.radius > M1.radius) :
  canCover M2 M1 ∧ ¬(canCover M1 M2) := by
  sorry

end NUMINAMATH_CALUDE_circle_coverage_l93_9387


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l93_9338

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  x₁ = 1 ∧ x₂ = 3 ∧ 
  (x₁^2 - 4*x₁ + 3 = 0) ∧ 
  (x₂^2 - 4*x₂ + 3 = 0) ∧
  (∀ x : ℝ, x^2 - 4*x + 3 = 0 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l93_9338


namespace NUMINAMATH_CALUDE_square_divisibility_l93_9347

theorem square_divisibility (n : ℤ) : ∃ k : ℤ, n^2 = 4*k ∨ n^2 = 4*k + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l93_9347


namespace NUMINAMATH_CALUDE_cosine_sine_square_difference_l93_9399

theorem cosine_sine_square_difference (α : Real) (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.cos α ^ 2 - Real.sin α ^ 2 = -(Real.sqrt 5 / 3) := by
  sorry

end NUMINAMATH_CALUDE_cosine_sine_square_difference_l93_9399


namespace NUMINAMATH_CALUDE_zeros_in_fraction_l93_9389

def count_leading_zeros (n : ℚ) : ℕ :=
  sorry

theorem zeros_in_fraction : count_leading_zeros (1 / (2^7 * 5^9)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_zeros_in_fraction_l93_9389


namespace NUMINAMATH_CALUDE_stationery_cost_l93_9303

def total_spent : ℝ := 32
def backpack_cost : ℝ := 15
def notebook_cost : ℝ := 3
def notebook_count : ℕ := 5

theorem stationery_cost :
  total_spent - (backpack_cost + notebook_cost * notebook_count) = 2 := by
  sorry

end NUMINAMATH_CALUDE_stationery_cost_l93_9303


namespace NUMINAMATH_CALUDE_fraction_simplification_l93_9300

theorem fraction_simplification (a b : ℕ) (h : b ≠ 0) : 
  (4 * a) / (4 * b) = a / b :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l93_9300


namespace NUMINAMATH_CALUDE_smallest_good_sequence_index_is_60_l93_9339

-- Define a good sequence
def GoodSequence (a : ℕ → ℝ) : Prop :=
  (∃ k : ℕ+, a 0 = k) ∧
  (∀ i : ℕ, (a (i + 1) = 2 * a i + 1) ∨ (a (i + 1) = a i / (a i + 2))) ∧
  (∃ k : ℕ+, a k = 2014)

-- Define the property we want to prove
def SmallestGoodSequenceIndex : Prop :=
  ∃ n : ℕ+, 
    (∃ a : ℕ → ℝ, GoodSequence a ∧ a n = 2014) ∧
    (∀ m : ℕ+, m < n → ¬∃ a : ℕ → ℝ, GoodSequence a ∧ a m = 2014)

-- The theorem to prove
theorem smallest_good_sequence_index_is_60 : 
  SmallestGoodSequenceIndex ∧ (∃ n : ℕ+, n = 60 ∧ ∃ a : ℕ → ℝ, GoodSequence a ∧ a n = 2014) :=
sorry

end NUMINAMATH_CALUDE_smallest_good_sequence_index_is_60_l93_9339


namespace NUMINAMATH_CALUDE_committee_meeting_attendance_committee_meeting_attendance_proof_l93_9391

/-- Proves that the total number of people present in the committee meeting is 7 -/
theorem committee_meeting_attendance : ℕ → ℕ → Prop :=
  fun (associate_profs assistant_profs : ℕ) =>
    -- Each associate professor brings 2 pencils and 1 chart
    -- Each assistant professor brings 1 pencil and 2 charts
    -- Total of 10 pencils and 11 charts brought to the meeting
    (2 * associate_profs + assistant_profs = 10) ∧
    (associate_profs + 2 * assistant_profs = 11) →
    -- The total number of people present is 7
    associate_profs + assistant_profs = 7

theorem committee_meeting_attendance_proof : ∃ (a b : ℕ), committee_meeting_attendance a b :=
  sorry

end NUMINAMATH_CALUDE_committee_meeting_attendance_committee_meeting_attendance_proof_l93_9391


namespace NUMINAMATH_CALUDE_angle_triple_complement_measure_l93_9352

theorem angle_triple_complement_measure :
  ∀ x : ℝ, 
    (x = 3 * (90 - x)) → 
    x = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_complement_measure_l93_9352


namespace NUMINAMATH_CALUDE_cubic_root_cubes_l93_9308

-- Define the polynomials h(x) and p(x)
def h (x : ℝ) : ℝ := x^3 - x^2 - 4*x + 4
def p (a b c x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_root_cubes (a b c : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ h x = 0 ∧ h y = 0 ∧ h z = 0) →
  (∀ s : ℝ, h s = 0 → p a b c (s^3) = 0) →
  a = 12 ∧ b = -13 ∧ c = -64 :=
sorry

end NUMINAMATH_CALUDE_cubic_root_cubes_l93_9308


namespace NUMINAMATH_CALUDE_runner_speed_ratio_l93_9340

theorem runner_speed_ratio :
  ∀ (v1 v2 : ℝ),
    v1 > v2 →
    v1 - v2 = 4 →
    v1 + v2 = 20 →
    v1 / v2 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_runner_speed_ratio_l93_9340


namespace NUMINAMATH_CALUDE_one_right_intersection_implies_negative_n_l93_9392

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a quadratic function has one intersection point with the x-axis to the right of the y-axis -/
def hasOneRightIntersection (f : QuadraticFunction) : Prop :=
  ∃ x : ℝ, x > 0 ∧ f.a * x^2 + f.b * x + f.c = 0 ∧
  ∀ y : ℝ, y ≠ x → f.a * y^2 + f.b * y + f.c ≠ 0

/-- Theorem: If a quadratic function y = x^2 + 3x + n has one intersection point
    with the x-axis to the right of the y-axis, then n < 0 -/
theorem one_right_intersection_implies_negative_n :
  ∀ n : ℝ, hasOneRightIntersection ⟨1, 3, n⟩ → n < 0 := by
  sorry


end NUMINAMATH_CALUDE_one_right_intersection_implies_negative_n_l93_9392


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_l93_9322

/-- Profit function given price x -/
def profit (x : ℝ) : ℝ :=
  let P := -750 * x + 15000
  x * P - 4 * P - 7000

/-- The optimal price that maximizes profit -/
def optimal_price : ℝ := 12

/-- The maximum profit achieved at the optimal price -/
def max_profit : ℝ := 41000

/-- Theorem stating that the optimal price maximizes profit -/
theorem optimal_price_maximizes_profit :
  profit optimal_price = max_profit ∧
  ∀ x : ℝ, profit x ≤ max_profit :=
sorry

/-- Theorem stating that the maximum profit is achieved at the optimal price -/
theorem max_profit_at_optimal_price :
  ∀ x : ℝ, x ≠ optimal_price → profit x < max_profit :=
sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_profit_max_profit_at_optimal_price_l93_9322


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l93_9325

theorem floor_negative_seven_fourths : ⌊(-7 : ℝ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l93_9325


namespace NUMINAMATH_CALUDE_man_travel_distance_l93_9326

theorem man_travel_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 2 → time = 39 → distance = speed * time → distance = 78 := by
  sorry

end NUMINAMATH_CALUDE_man_travel_distance_l93_9326


namespace NUMINAMATH_CALUDE_cosine_symmetry_center_l93_9311

/-- Given a cosine function with a phase shift, prove that under certain symmetry conditions,
    the symmetric center closest to the origin is at a specific point when the period is maximized. -/
theorem cosine_symmetry_center (ω : ℝ) (h_ω_pos : ω > 0) :
  let f : ℝ → ℝ := λ x => Real.cos (ω * x + 3 * Real.pi / 4)
  (∀ x : ℝ, f (π / 3 - x) = f (π / 3 + x)) →  -- Symmetry about x = π/6
  (∀ k : ℤ, ω ≠ 6 * k - 9 / 2 → ω > 6 * k - 9 / 2) →  -- ω is the smallest positive value
  (π / 6 : ℝ) ∈ { x : ℝ | ∃ k : ℤ, x = 2 / 3 * k * π - π / 6 } →  -- Symmetric center formula
  (-π / 6 : ℝ) ∈ { x : ℝ | ∃ k : ℤ, x = 2 / 3 * k * π - π / 6 } ∧  -- Closest symmetric center
  (∀ x : ℝ, x ∈ { y : ℝ | ∃ k : ℤ, y = 2 / 3 * k * π - π / 6 } → |x| ≥ |(-π / 6 : ℝ)|) :=
by
  sorry

end NUMINAMATH_CALUDE_cosine_symmetry_center_l93_9311


namespace NUMINAMATH_CALUDE_completing_square_result_l93_9371

theorem completing_square_result (x : ℝ) :
  (x^2 - 6*x + 5 = 0) ↔ ((x - 3)^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_result_l93_9371


namespace NUMINAMATH_CALUDE_max_value_of_a_l93_9337

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- State the theorem
theorem max_value_of_a :
  ∀ a : ℝ, (A a ∪ B a = Set.univ) → (∀ b : ℝ, (A b ∪ B b = Set.univ) → b ≤ a) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l93_9337


namespace NUMINAMATH_CALUDE_vehicle_value_last_year_l93_9378

/-- If a vehicle's value this year is 16000 dollars and is 0.8 times its value last year,
    then its value last year was 20000 dollars. -/
theorem vehicle_value_last_year 
  (value_this_year : ℝ) 
  (value_ratio : ℝ) 
  (h1 : value_this_year = 16000)
  (h2 : value_ratio = 0.8)
  (h3 : value_this_year = value_ratio * value_last_year) : 
  value_last_year = 20000 := by
  sorry


end NUMINAMATH_CALUDE_vehicle_value_last_year_l93_9378


namespace NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l93_9367

/-- The focus of a parabola y = ax^2 + k is at (0, k - 1/(4a)) when a ≠ 0 -/
theorem parabola_focus (a k : ℝ) (ha : a ≠ 0) :
  let f : ℝ × ℝ := (0, k - 1 / (4 * a))
  ∀ x y : ℝ, y = a * x^2 + k → (x - f.1)^2 + (y - f.2)^2 = (y - k + 1 / (4 * a))^2 / (4 * a^2) :=
sorry

/-- The focus of the parabola y = -2x^2 + 4 is at (0, 33/8) -/
theorem focus_of_specific_parabola :
  let f : ℝ × ℝ := (0, 33/8)
  ∀ x y : ℝ, y = -2 * x^2 + 4 → (x - f.1)^2 + (y - f.2)^2 = (y - 4 + 1/8)^2 / 16 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_focus_of_specific_parabola_l93_9367


namespace NUMINAMATH_CALUDE_solution_set_for_m_eq_2_m_range_for_solution_set_R_l93_9356

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m-1)*x - m

-- Part 1
theorem solution_set_for_m_eq_2 :
  {x : ℝ | f 2 x < 0} = {x : ℝ | -2 < x ∧ x < 1} := by sorry

-- Part 2
theorem m_range_for_solution_set_R :
  (∀ x, f m x ≥ -1) → m ∈ Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_m_eq_2_m_range_for_solution_set_R_l93_9356


namespace NUMINAMATH_CALUDE_correct_operation_l93_9335

theorem correct_operation (x : ℤ) : (x - 7) * 20 = -380 → (x * 7) - 20 = -104 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l93_9335
