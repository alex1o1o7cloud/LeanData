import Mathlib

namespace log_54883_between_consecutive_integers_l1734_173411

theorem log_54883_between_consecutive_integers :
  ∃ (c d : ℤ), c + 1 = d ∧ (c : ℝ) < Real.log 54883 / Real.log 10 ∧ Real.log 54883 / Real.log 10 < (d : ℝ) → c + d = 9 := by
  sorry

end log_54883_between_consecutive_integers_l1734_173411


namespace other_vehicle_wheels_l1734_173405

theorem other_vehicle_wheels (total_wheels : Nat) (four_wheelers : Nat) (h1 : total_wheels = 58) (h2 : four_wheelers = 14) :
  ∃ (other_wheels : Nat), other_wheels = 2 ∧ total_wheels = four_wheelers * 4 + other_wheels := by
sorry

end other_vehicle_wheels_l1734_173405


namespace rectangle_width_proof_l1734_173401

-- Define the original length of the rectangle
def original_length : ℝ := 140

-- Define the length increase factor
def length_increase : ℝ := 1.30

-- Define the width decrease factor
def width_decrease : ℝ := 0.8230769230769231

-- Define the approximate width we want to prove
def approximate_width : ℝ := 130.91

-- Theorem statement
theorem rectangle_width_proof :
  ∃ (original_width : ℝ),
    (original_length * original_width = original_length * length_increase * original_width * width_decrease) ∧
    (abs (original_width - approximate_width) < 0.01) :=
by sorry

end rectangle_width_proof_l1734_173401


namespace lowest_n_for_polynomial_property_l1734_173457

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Property that a polynomial takes value 2 for n distinct integers -/
def TakesValueTwoForNIntegers (P : IntPolynomial) (n : ℕ) : Prop :=
  ∃ (S : Finset ℤ), S.card = n ∧ ∀ x ∈ S, P x = 2

/-- Property that a polynomial never takes value 4 for any integer -/
def NeverTakesValueFour (P : IntPolynomial) : Prop :=
  ∀ x : ℤ, P x ≠ 4

/-- The main theorem statement -/
theorem lowest_n_for_polynomial_property : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m ≥ n → 
    ∀ (P : IntPolynomial), 
      TakesValueTwoForNIntegers P m → NeverTakesValueFour P) ∧
  (∀ (k : ℕ), 0 < k ∧ k < n → 
    ∃ (Q : IntPolynomial), 
      TakesValueTwoForNIntegers Q k ∧ ¬NeverTakesValueFour Q) ∧
  n = 4 := by
  sorry

end lowest_n_for_polynomial_property_l1734_173457


namespace complex_fraction_simplification_l1734_173459

theorem complex_fraction_simplification :
  (1 + 2 * Complex.I) / (1 - 2 * Complex.I) = -(3/5 : ℂ) + (4/5 : ℂ) * Complex.I := by
  sorry

end complex_fraction_simplification_l1734_173459


namespace commute_speed_ratio_l1734_173460

/-- Proves that the ratio of speeds for a commuter is 2:1 given specific conditions -/
theorem commute_speed_ratio 
  (distance : ℝ) 
  (total_time : ℝ) 
  (return_speed : ℝ) 
  (h1 : distance = 28) 
  (h2 : total_time = 6) 
  (h3 : return_speed = 14) : 
  return_speed / ((2 * distance) / total_time - return_speed) = 2 := by
  sorry

#check commute_speed_ratio

end commute_speed_ratio_l1734_173460


namespace pure_imaginary_implies_x_equals_one_l1734_173437

/-- A complex number z is pure imaginary if its real part is 0 and its imaginary part is not 0 -/
def IsPureImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_implies_x_equals_one :
  ∀ x : ℝ, IsPureImaginary ((x^2 - 1) + (x^2 + 3*x + 2)*I) → x = 1 :=
by sorry

end pure_imaginary_implies_x_equals_one_l1734_173437


namespace frog_final_position_probability_l1734_173446

noncomputable def frog_jump_probability : ℝ := 
  let n : ℕ := 4  -- number of jumps
  let jump_length : ℝ := 1  -- length of each jump
  let max_distance : ℝ := 1.5  -- maximum distance from starting point
  1/3  -- probability

theorem frog_final_position_probability :
  frog_jump_probability = 1/3 :=
sorry

end frog_final_position_probability_l1734_173446


namespace proportion_sum_l1734_173427

theorem proportion_sum (a b c d : ℚ) 
  (h1 : a/b = 3/2) 
  (h2 : c/d = 3/2) 
  (h3 : b + d ≠ 0) : 
  (a + c) / (b + d) = 3/2 := by
sorry

end proportion_sum_l1734_173427


namespace oil_mixture_price_l1734_173496

theorem oil_mixture_price (x y z : ℝ) 
  (volume_constraint : x + y + z = 23.5)
  (cost_constraint : 55 * x + 70 * y + 82 * z = 65 * 23.5) :
  (55 * x + 70 * y + 82 * z) / (x + y + z) = 65 := by
  sorry

end oil_mixture_price_l1734_173496


namespace double_factorial_properties_l1734_173404

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def units_digit (n : ℕ) : ℕ := n % 10

theorem double_factorial_properties :
  (double_factorial 2003 * double_factorial 2002 = Nat.factorial 2003) ∧
  (double_factorial 2002 = 2^1001 * Nat.factorial 1001) ∧
  (units_digit (double_factorial 2002) = 0) ∧
  (units_digit (double_factorial 2003) = 5) := by
  sorry

#check double_factorial_properties

end double_factorial_properties_l1734_173404


namespace function_inequality_implies_parameter_bound_l1734_173461

open Real

theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∀ x > 0, x^2 - x ≤ Real.exp x - a*x - 1) →
  a ≤ Real.exp 1 - 1 :=
by sorry

end function_inequality_implies_parameter_bound_l1734_173461


namespace cat_dog_food_difference_l1734_173486

/-- Represents the number of packages of cat food Adam bought. -/
def cat_food_packages : ℕ := 15

/-- Represents the number of packages of dog food Adam bought. -/
def dog_food_packages : ℕ := 10

/-- Represents the number of cans in each package of cat food. -/
def cans_per_cat_package : ℕ := 12

/-- Represents the number of cans in each package of dog food. -/
def cans_per_dog_package : ℕ := 8

/-- Theorem stating the difference between the total number of cans of cat food and dog food. -/
theorem cat_dog_food_difference :
  cat_food_packages * cans_per_cat_package - dog_food_packages * cans_per_dog_package = 100 := by
  sorry

end cat_dog_food_difference_l1734_173486


namespace prob_empty_mailbox_is_five_ninths_l1734_173454

/-- The number of different greeting cards -/
def num_cards : ℕ := 4

/-- The number of different mailboxes -/
def num_mailboxes : ℕ := 3

/-- The probability of at least one mailbox being empty when cards are randomly placed -/
def prob_empty_mailbox : ℚ := 5/9

/-- Theorem stating that the probability of at least one empty mailbox is 5/9 -/
theorem prob_empty_mailbox_is_five_ninths :
  prob_empty_mailbox = 5/9 :=
sorry

end prob_empty_mailbox_is_five_ninths_l1734_173454


namespace existence_of_triangle_with_divisible_side_lengths_l1734_173453

/-- Given an odd prime p, a positive integer n, and 8 distinct points with integer coordinates
    on a circle of diameter p^n, there exists a triangle formed by three of these points
    such that the square of its side lengths is divisible by p^(n+1). -/
theorem existence_of_triangle_with_divisible_side_lengths
  (p : ℕ) (n : ℕ) (h_p_prime : Nat.Prime p) (h_p_odd : Odd p) (h_n_pos : 0 < n)
  (points : Fin 8 → ℤ × ℤ)
  (h_distinct : Function.Injective points)
  (h_on_circle : ∀ i : Fin 8, (points i).1^2 + (points i).2^2 = (p^n)^2) :
  ∃ i j k : Fin 8, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (∃ m : ℕ, (((points i).1 - (points j).1)^2 + ((points i).2 - (points j).2)^2) * m = p^(n+1)) ∧
    (∃ m : ℕ, (((points j).1 - (points k).1)^2 + ((points j).2 - (points k).2)^2) * m = p^(n+1)) ∧
    (∃ m : ℕ, (((points k).1 - (points i).1)^2 + ((points k).2 - (points i).2)^2) * m = p^(n+1)) :=
by sorry

end existence_of_triangle_with_divisible_side_lengths_l1734_173453


namespace alley_width_l1734_173489

/-- The width of a narrow alley given a ladder's length and angles -/
theorem alley_width (b : ℝ) (h_b_pos : b > 0) : ∃ w : ℝ,
  w = b * (1 + Real.sqrt 3) / 2 ∧
  ∃ (x y : ℝ),
    x > 0 ∧ y > 0 ∧
    x = b * Real.cos (π / 3) ∧
    y = b * Real.cos (π / 6) ∧
    w = x + y :=
by sorry

end alley_width_l1734_173489


namespace painted_faces_count_correct_l1734_173469

/-- Represents the count of painted faces for unit cubes cut from a larger cube. -/
structure PaintedFacesCount where
  one_face : ℕ
  two_faces : ℕ
  three_faces : ℕ

/-- 
  Given a cube with side length a (where a is a natural number greater than 2),
  calculates the number of unit cubes with exactly one, two, and three faces painted
  when the cube is cut into unit cubes.
-/
def count_painted_faces (a : ℕ) : PaintedFacesCount :=
  { one_face := 6 * (a - 2)^2,
    two_faces := 12 * (a - 2),
    three_faces := 8 }

/-- Theorem stating the correct count of painted faces for unit cubes. -/
theorem painted_faces_count_correct (a : ℕ) (h : a > 2) :
  count_painted_faces a = { one_face := 6 * (a - 2)^2,
                            two_faces := 12 * (a - 2),
                            three_faces := 8 } := by
  sorry

end painted_faces_count_correct_l1734_173469


namespace spaceship_speed_halving_l1734_173458

/-- The number of additional people that cause the spaceship's speed to be halved -/
def additional_people : ℕ := sorry

/-- The speed of the spaceship given the number of people on board -/
def speed (people : ℕ) : ℝ := sorry

/-- Theorem: The number of additional people that cause the spaceship's speed to be halved is 100 -/
theorem spaceship_speed_halving :
  (speed 200 = 500) →
  (speed 400 = 125) →
  (∀ n : ℕ, speed (n + additional_people) = (speed n) / 2) →
  additional_people = 100 := by
  sorry

end spaceship_speed_halving_l1734_173458


namespace point_coordinates_wrt_origin_l1734_173410

/-- In a Cartesian coordinate system, the coordinates of a point (2, -3) with respect to the origin are (2, -3) -/
theorem point_coordinates_wrt_origin :
  let point : ℝ × ℝ := (2, -3)
  point = (2, -3) := by sorry

end point_coordinates_wrt_origin_l1734_173410


namespace function_properties_l1734_173463

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a / x + 1

theorem function_properties :
  (∃ (x : ℝ), ∀ (y : ℝ), y > 0 → f 1 x ≤ f 1 y) ∧
  (f 1 1 = 2) ∧
  (∀ (a : ℝ), (∃! (x : ℝ), x > Real.exp (-3) ∧ f a x = 0) ↔ 
    (a ≤ 2 / Real.exp 3 ∨ a = 1 / Real.exp 2)) :=
by sorry

end function_properties_l1734_173463


namespace polynomial_simplification_l1734_173491

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + r^2 + 5 * r - 4) - (r^3 + 3 * r^2 + 7 * r - 2) = r^3 - 2 * r^2 - 2 * r - 2 := by
  sorry

end polynomial_simplification_l1734_173491


namespace zero_has_square_and_cube_root_l1734_173426

/-- A number x is a square root of y if x * x = y -/
def is_square_root (x y : ℝ) : Prop := x * x = y

/-- A number x is a cube root of y if x * x * x = y -/
def is_cube_root (x y : ℝ) : Prop := x * x * x = y

/-- 0 has both a square root and a cube root -/
theorem zero_has_square_and_cube_root :
  ∃ (x y : ℝ), is_square_root x 0 ∧ is_cube_root y 0 :=
sorry

end zero_has_square_and_cube_root_l1734_173426


namespace smallest_integer_a_l1734_173494

theorem smallest_integer_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (Real.pi / 2), 
    Real.exp x - x * Real.cos x + Real.cos x * Real.log (Real.cos x) + a * x^2 ≥ 1) ↔ 
  a ≥ 1 := by
sorry

end smallest_integer_a_l1734_173494


namespace sector_angle_l1734_173406

/-- Given a circular sector with perimeter 8 and area 4, its central angle is 2 radians. -/
theorem sector_angle (R : ℝ) (α : ℝ) (h1 : 2 * R + R * α = 8) (h2 : 1/2 * α * R^2 = 4) : α = 2 := by
  sorry

end sector_angle_l1734_173406


namespace inequality_theorem_l1734_173472

theorem inequality_theorem (p q : ℝ) :
  q > 0 ∧ p ≥ 0 →
  ((4 * (p * q^2 + p^2 * q + 4 * q^2 + 4 * p * q)) / (p + q) > 3 * p^2 * q) ↔
  (0 ≤ p ∧ p < 4) :=
by sorry

end inequality_theorem_l1734_173472


namespace unique_intersection_implies_r_equals_three_l1734_173479

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def B (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

-- State the theorem
theorem unique_intersection_implies_r_equals_three 
  (r : ℝ) 
  (h_r_pos : r > 0) 
  (h_unique : ∃! p, p ∈ A ∩ B r) : 
  r = 3 := by
sorry

end unique_intersection_implies_r_equals_three_l1734_173479


namespace solution_set_when_a_is_2_range_of_a_l1734_173414

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_when_a_is_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end solution_set_when_a_is_2_range_of_a_l1734_173414


namespace A_subset_B_iff_a_geq_2_plus_sqrt5_l1734_173476

/-- Set A defined as a circle with center (2,1) and radius 1 -/
def A : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 - 1)^2 ≤ 1}

/-- Set B defined by the condition 2|x-1| + |y-1| ≤ a -/
def B (a : ℝ) : Set (ℝ × ℝ) := {p | 2 * |p.1 - 1| + |p.2 - 1| ≤ a}

/-- Theorem stating that A is a subset of B if and only if a ≥ 2 + √5 -/
theorem A_subset_B_iff_a_geq_2_plus_sqrt5 :
  ∀ a : ℝ, A ⊆ B a ↔ a ≥ 2 + Real.sqrt 5 := by sorry

end A_subset_B_iff_a_geq_2_plus_sqrt5_l1734_173476


namespace inequality_proof_l1734_173425

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : a + b + Real.sqrt 2 * c ≤ 2 := by
  sorry

end inequality_proof_l1734_173425


namespace pm2_5_diameter_scientific_notation_l1734_173431

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The diameter of PM2.5 particulate matter in meters -/
def pm2_5_diameter : ℝ := 0.0000025

/-- The scientific notation representation of the PM2.5 diameter -/
def pm2_5_scientific : ScientificNotation :=
  { coefficient := 2.5
    exponent := -6
    valid := by sorry }

theorem pm2_5_diameter_scientific_notation :
  pm2_5_diameter = pm2_5_scientific.coefficient * (10 : ℝ) ^ pm2_5_scientific.exponent :=
by sorry

end pm2_5_diameter_scientific_notation_l1734_173431


namespace money_sharing_l1734_173429

theorem money_sharing (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 2 * (total / 13) →
  ben = 3 * (total / 13) →
  carlos = 8 * (total / 13) →
  ben = 60 →
  total = 260 := by
sorry

end money_sharing_l1734_173429


namespace diagonal_length_count_l1734_173452

/-- Represents a quadrilateral ABCD with given side lengths and diagonal AC --/
structure Quadrilateral where
  ab : ℕ
  bc : ℕ
  cd : ℕ
  ad : ℕ
  ac : ℕ

/-- Checks if the triangle inequality holds for a triangle with given side lengths --/
def triangle_inequality (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem about the number of possible integer lengths for the diagonal --/
theorem diagonal_length_count (q : Quadrilateral) : 
  q.ab = 9 → q.bc = 11 → q.cd = 18 → q.ad = 14 →
  (∀ x : ℕ, 5 ≤ x → x ≤ 19 → 
    (q.ac = x → 
      triangle_inequality q.ab q.bc x ∧ 
      triangle_inequality q.cd q.ad x)) →
  (∀ x : ℕ, x < 5 ∨ x > 19 → 
    ¬(triangle_inequality q.ab q.bc x ∧ 
      triangle_inequality q.cd q.ad x)) →
  (Finset.range 15).card = 15 := by
  sorry

#check diagonal_length_count

end diagonal_length_count_l1734_173452


namespace triangle_side_length_l1734_173487

theorem triangle_side_length (a b c : ℝ) (A : ℝ) (S : ℝ) :
  A = π / 3 →  -- 60° in radians
  S = (3 * Real.sqrt 3) / 2 →  -- Area of the triangle
  b + c = 3 * Real.sqrt 3 →  -- Sum of sides b and c
  S = (1 / 2) * b * c * Real.sin A →  -- Area formula
  a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A →  -- Law of cosines
  a = 3 := by sorry

end triangle_side_length_l1734_173487


namespace root_equation_k_value_l1734_173438

theorem root_equation_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 - k*x - 6 = 0 ∧ x = 3) → k = 1 := by
  sorry

end root_equation_k_value_l1734_173438


namespace sufficient_not_necessary_condition_necessary_not_sufficient_condition_l1734_173450

def M : Set ℝ := {x | (x + 3) * (x - 5) > 0}

def P (a : ℝ) : Set ℝ := {x | x^2 + (a - 8) * x - 8 * a ≤ 0}

def target_set : Set ℝ := {x | 5 < x ∧ x ≤ 8}

theorem sufficient_not_necessary_condition :
  (∀ a, a = 0 → M ∩ P a = target_set) ∧
  ¬(∀ a, M ∩ P a = target_set → a = 0) := by sorry

theorem necessary_not_sufficient_condition :
  (∀ a, M ∩ P a = target_set → a ≤ 3) ∧
  ¬(∀ a, a ≤ 3 → M ∩ P a = target_set) := by sorry

end sufficient_not_necessary_condition_necessary_not_sufficient_condition_l1734_173450


namespace equidistant_point_l1734_173400

/-- The distance between two points in a 2D plane -/
def distance (x1 y1 x2 y2 : ℚ) : ℚ :=
  ((x2 - x1)^2 + (y2 - y1)^2).sqrt

/-- The point C with coordinates (3, 0) -/
def C : ℚ × ℚ := (3, 0)

/-- The point D with coordinates (5, 6) -/
def D : ℚ × ℚ := (5, 6)

/-- The y-coordinate of the point on the y-axis -/
def y : ℚ := 13/3

theorem equidistant_point : 
  distance 0 y C.1 C.2 = distance 0 y D.1 D.2 := by sorry

end equidistant_point_l1734_173400


namespace opposite_numbers_cube_inequality_l1734_173445

theorem opposite_numbers_cube_inequality (a b : ℝ) (h1 : a = -b) (h2 : a ≠ 0) : a^3 ≠ b^3 := by
  sorry

end opposite_numbers_cube_inequality_l1734_173445


namespace probability_two_segments_longer_than_one_l1734_173471

/-- The probability of exactly two segments being longer than 1 when a line segment 
    of length 3 is divided into three parts by randomly selecting two points -/
theorem probability_two_segments_longer_than_one (total_length : ℝ) 
  (h_total_length : total_length = 3) : ℝ := by
  sorry

end probability_two_segments_longer_than_one_l1734_173471


namespace parallelogram_side_length_l1734_173449

/-- Given a parallelogram with adjacent sides of lengths 3s and 4s units forming a 30-degree angle,
    if the area is 18√3 square units, then s = 3^(3/4). -/
theorem parallelogram_side_length (s : ℝ) : 
  s > 0 →  -- Ensuring s is positive for physical meaning
  (3 * s) * (4 * s) * Real.sin (π / 6) = 18 * Real.sqrt 3 →
  s = 3 ^ (3 / 4) := by
  sorry


end parallelogram_side_length_l1734_173449


namespace eggs_in_jar_l1734_173408

/-- The number of eggs left in a jar after some are removed -/
def eggs_left (original : ℕ) (removed : ℕ) : ℕ := original - removed

/-- Theorem: Given 27 original eggs and 7 removed eggs, 20 eggs are left -/
theorem eggs_in_jar : eggs_left 27 7 = 20 := by
  sorry

end eggs_in_jar_l1734_173408


namespace product_of_repeated_digits_l1734_173434

def number_of_3s : ℕ := 25
def number_of_6s : ℕ := 25

def number_of_2s : ℕ := 24
def number_of_7s : ℕ := 24

def first_number : ℕ := (3 * (10^number_of_3s - 1)) / 9
def second_number : ℕ := (6 * (10^number_of_6s - 1)) / 9

def result : ℕ := (2 * 10^49 + 10^48 + 7 * (10^24 - 1) / 9) * 10 + 8

theorem product_of_repeated_digits :
  first_number * second_number = result := by sorry

end product_of_repeated_digits_l1734_173434


namespace total_payment_equals_car_cost_l1734_173403

/-- Represents the car purchase scenario -/
structure CarPurchase where
  carCost : ℕ             -- Cost of the car in euros
  initialPayment : ℕ      -- Initial payment in euros
  installments : ℕ        -- Number of installments
  installmentAmount : ℕ   -- Amount per installment in euros

/-- Theorem stating that the total amount paid equals the car's cost -/
theorem total_payment_equals_car_cost (purchase : CarPurchase) 
  (h1 : purchase.carCost = 18000)
  (h2 : purchase.initialPayment = 3000)
  (h3 : purchase.installments = 6)
  (h4 : purchase.installmentAmount = 2500) :
  purchase.initialPayment + purchase.installments * purchase.installmentAmount = purchase.carCost :=
by sorry

end total_payment_equals_car_cost_l1734_173403


namespace tangent_pentagon_division_l1734_173440

/-- A pentagon with sides tangent to a circle -/
structure TangentPentagon where
  -- Sides of the pentagon
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ
  -- Ensure all sides are positive
  side1_pos : 0 < side1
  side2_pos : 0 < side2
  side3_pos : 0 < side3
  side4_pos : 0 < side4
  side5_pos : 0 < side5
  -- Condition for tangency to a circle
  tangent_condition : ∃ (x : ℝ), 
    x + (side2 - x) = side1 ∧
    (side2 - x) + (side3 - (side2 - x)) = side2 ∧
    (side3 - (side2 - x)) + (side4 - (side3 - (side2 - x))) = side3 ∧
    (side4 - (side3 - (side2 - x))) + (side5 - (side4 - (side3 - (side2 - x)))) = side4 ∧
    (side5 - (side4 - (side3 - (side2 - x)))) + x = side5

/-- Theorem about the division of the first side in a specific tangent pentagon -/
theorem tangent_pentagon_division (p : TangentPentagon) 
  (h1 : p.side1 = 5) (h2 : p.side2 = 6) (h3 : p.side3 = 7) (h4 : p.side4 = 8) (h5 : p.side5 = 9) :
  ∃ (x : ℝ), x = 3/2 ∧ p.side1 - x = 5/2 := by
  sorry

end tangent_pentagon_division_l1734_173440


namespace A_intersect_B_eq_zero_one_l1734_173402

-- Define set A
def A : Set ℕ := {0, 1, 2}

-- Define set B
def B : Set ℕ := {x : ℕ | (x + 1) / (x - 2 : ℝ) ≤ 0}

-- Theorem statement
theorem A_intersect_B_eq_zero_one : A ∩ B = {0, 1} := by sorry

end A_intersect_B_eq_zero_one_l1734_173402


namespace vector_equation_solution_l1734_173407

theorem vector_equation_solution (a b : ℝ × ℝ) (m n : ℝ) : 
  a = (2, 1) → 
  b = (1, -2) → 
  m • a + n • b = (9, -8) → 
  m - n = -3 := by sorry

end vector_equation_solution_l1734_173407


namespace mary_crayons_left_l1734_173499

/-- Represents the number of crayons Mary has left after giving some away -/
def crayons_left (initial_green initial_blue initial_yellow given_green given_blue given_yellow : ℕ) : ℕ :=
  (initial_green - given_green) + (initial_blue - given_blue) + (initial_yellow - given_yellow)

/-- Theorem stating that Mary has 14 crayons left after giving some away -/
theorem mary_crayons_left : 
  crayons_left 5 8 7 3 1 2 = 14 := by
  sorry

end mary_crayons_left_l1734_173499


namespace max_trig_function_ratio_l1734_173470

/-- Given a function f(x) = 3sin(x) + 4cos(x) that attains its maximum value when x = θ,
    prove that (sin(2θ) + cos²(θ) + 1) / cos(2θ) = 65/7 -/
theorem max_trig_function_ratio (θ : Real) 
    (h : ∀ x, 3 * Real.sin x + 4 * Real.cos x ≤ 3 * Real.sin θ + 4 * Real.cos θ) :
    (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 65 / 7 := by
  sorry

end max_trig_function_ratio_l1734_173470


namespace satisfying_function_is_identity_l1734_173495

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (f 1 = 1) ∧
  (∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2)

/-- Theorem stating that any function satisfying the conditions is the identity function -/
theorem satisfying_function_is_identity (f : ℝ → ℝ) (hf : SatisfyingFunction f) : 
  ∀ x : ℝ, f x = x := by
  sorry

end satisfying_function_is_identity_l1734_173495


namespace parabola_shift_right_one_unit_l1734_173497

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c }

theorem parabola_shift_right_one_unit :
  let original := Parabola.mk (-1/2) 0 0
  let shifted := shift_parabola original 1
  shifted = Parabola.mk (-1/2) 1 (-1/2) := by sorry

end parabola_shift_right_one_unit_l1734_173497


namespace complement_intersection_problem_l1734_173433

theorem complement_intersection_problem (U A B : Set ℕ) : 
  U = {0, 1, 2, 3} → 
  A = {1, 2} → 
  B = {3, 4} → 
  (U \ A) ∩ B = {3} := by
  sorry

end complement_intersection_problem_l1734_173433


namespace total_siblings_weight_l1734_173482

def antonio_weight : ℕ := 50
def sister_weight_diff : ℕ := 12
def antonio_backpack : ℕ := 5
def sister_backpack : ℕ := 3
def marco_weight : ℕ := 30
def stuffed_animal : ℕ := 2

theorem total_siblings_weight :
  (antonio_weight + (antonio_weight - sister_weight_diff) + marco_weight) +
  (antonio_backpack + sister_backpack + stuffed_animal) = 128 := by
  sorry

end total_siblings_weight_l1734_173482


namespace power_division_simplification_l1734_173468

theorem power_division_simplification : (1000 : ℕ)^7 / (10 : ℕ)^17 = 10000 := by
  sorry

end power_division_simplification_l1734_173468


namespace binomial_60_3_l1734_173441

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l1734_173441


namespace hotel_expenditure_l1734_173418

theorem hotel_expenditure (n : ℕ) (m : ℕ) (individual_cost : ℕ) (extra_cost : ℕ) 
  (h1 : n = 9)
  (h2 : m = 8)
  (h3 : individual_cost = 12)
  (h4 : extra_cost = 8) :
  m * individual_cost + (individual_cost + (m * individual_cost + individual_cost + extra_cost) / n) = 117 := by
  sorry

end hotel_expenditure_l1734_173418


namespace remaining_distance_proof_l1734_173484

def total_distance : ℝ := 369
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3
def anayet_speed : ℝ := 61
def anayet_time : ℝ := 2

theorem remaining_distance_proof :
  total_distance - (amoli_speed * amoli_time + anayet_speed * anayet_time) = 121 := by
  sorry

end remaining_distance_proof_l1734_173484


namespace relationship_abc_l1734_173428

theorem relationship_abc (a b c : ℝ) (ha : a = (0.4 : ℝ)^2) (hb : b = 2^(0.4 : ℝ)) (hc : c = Real.log 2 / Real.log 0.4) :
  c < a ∧ a < b :=
by sorry

end relationship_abc_l1734_173428


namespace sum_of_k_values_l1734_173435

theorem sum_of_k_values (a b c k : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_eq1 : a^2 / (1 - b) = k)
  (h_eq2 : b^2 / (1 - c) = k)
  (h_eq3 : c^2 / (1 - a) = k) :
  ∃ k1 k2 : ℝ, k = k1 ∨ k = k2 ∧ k1 + k2 = 1 := by
  sorry

end sum_of_k_values_l1734_173435


namespace union_equality_iff_a_in_range_l1734_173456

def A (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 1}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem union_equality_iff_a_in_range :
  ∀ a : ℝ, (A a ∪ B = B) ↔ (0 ≤ a ∧ a ≤ 2) :=
sorry

end union_equality_iff_a_in_range_l1734_173456


namespace system_of_equations_solution_l1734_173430

theorem system_of_equations_solution :
  ∃! (x y : ℝ), x + 2*y = 1 ∧ 3*x - 2*y = 7 ∧ x = 2 ∧ y = -1/2 := by
  sorry

end system_of_equations_solution_l1734_173430


namespace range_of_a_l1734_173444

theorem range_of_a (a : ℝ) : 
  (∀ x > 0, Real.exp x + Real.log a / a > Real.log x / a) ↔ a > Real.exp (-1) :=
sorry

end range_of_a_l1734_173444


namespace bus_stop_walking_time_l1734_173455

/-- The time taken to walk to the bus stop at the usual speed, given that walking at 4/5 of the usual speed results in arriving 8 minutes later than normal, is 32 minutes. -/
theorem bus_stop_walking_time : ∃ (T : ℝ), 
  (T > 0) ∧ 
  (4/5 * T + 8 = T) ∧ 
  (T = 32) := by
  sorry

end bus_stop_walking_time_l1734_173455


namespace bulbs_per_pack_l1734_173483

/-- The number of bulbs Sean needs to replace in each room --/
def bedroom_bulbs : ℕ := 2
def bathroom_bulbs : ℕ := 1
def kitchen_bulbs : ℕ := 1
def basement_bulbs : ℕ := 4

/-- The total number of bulbs Sean needs to replace in the rooms --/
def room_bulbs : ℕ := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs

/-- The number of bulbs Sean needs to replace in the garage --/
def garage_bulbs : ℕ := room_bulbs / 2

/-- The total number of bulbs Sean needs to replace --/
def total_bulbs : ℕ := room_bulbs + garage_bulbs

/-- The number of packs Sean will buy --/
def num_packs : ℕ := 6

/-- Theorem: The number of bulbs in each pack is 2 --/
theorem bulbs_per_pack : total_bulbs / num_packs = 2 := by
  sorry

end bulbs_per_pack_l1734_173483


namespace different_remainders_l1734_173466

theorem different_remainders (a b c p : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1)
  (hp : Nat.Prime p) (hsum : p = a * b + b * c + a * c) : 
  (a ^ 2 % p ≠ b ^ 2 % p ∧ a ^ 2 % p ≠ c ^ 2 % p ∧ b ^ 2 % p ≠ c ^ 2 % p) ∧
  (a ^ 3 % p ≠ b ^ 3 % p ∧ a ^ 3 % p ≠ c ^ 3 % p ∧ b ^ 3 % p ≠ c ^ 3 % p) := by
  sorry

end different_remainders_l1734_173466


namespace focus_of_our_parabola_l1734_173451

/-- A parabola is defined by its equation and opening direction -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  opens_downward : Bool

/-- The focus of a parabola is a point in the plane -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- Our specific parabola -/
def our_parabola : Parabola := {
  equation := fun x y => y = -1/4 * x^2,
  opens_downward := true
}

/-- Theorem stating that the focus of our parabola is at (0, -1) -/
theorem focus_of_our_parabola : focus our_parabola = (0, -1) := by sorry

end focus_of_our_parabola_l1734_173451


namespace quadratic_absolute_inequality_l1734_173422

theorem quadratic_absolute_inequality (a : ℝ) :
  (∀ x : ℝ, x^2 + a * |x| + 1 ≥ 0) ↔ a ≥ -2 := by sorry

end quadratic_absolute_inequality_l1734_173422


namespace sum_of_three_numbers_l1734_173492

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156) 
  (h2 : a*b + b*c + a*c = 50) : 
  a + b + c = 16 := by sorry

end sum_of_three_numbers_l1734_173492


namespace proposition_relationship_l1734_173474

theorem proposition_relationship (a b : ℝ) : 
  ¬(((a + b ≠ 4) → (a ≠ 1 ∧ b ≠ 3)) ∧ ((a ≠ 1 ∧ b ≠ 3) → (a + b ≠ 4))) :=
by sorry

end proposition_relationship_l1734_173474


namespace only_one_correct_proposition_l1734_173481

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Line → Prop)
variable (para_line : Line → Line → Prop)
variable (para_line_plane : Line → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the lines and planes
variable (a b c : Line) (α β : Plane)

-- State the theorem
theorem only_one_correct_proposition :
  (¬(∀ (a b c : Line) (α : Plane), 
    subset a α → subset b α → perp c a → perp c b → perp_line_plane c α)) ∧
  (¬(∀ (a b : Line) (α : Plane),
    subset b α → para_line a b → para_line_plane a α)) ∧
  (¬(∀ (a b : Line) (α β : Plane),
    para_line_plane a α → intersect α β b → para_line a b)) ∧
  (∀ (a b : Line) (α : Plane),
    perp_line_plane a α → perp_line_plane b α → para_line a b) ∧
  (¬(∀ (a b c : Line) (α β : Plane),
    ((subset a α → subset b α → perp c a → perp c b → perp_line_plane c α) ∨
     (subset b α → para_line a b → para_line_plane a α) ∨
     (para_line_plane a α → intersect α β b → para_line a b)) ∧
    (perp_line_plane a α → perp_line_plane b α → para_line a b))) :=
by sorry

end only_one_correct_proposition_l1734_173481


namespace negation_of_forall_positive_l1734_173416

theorem negation_of_forall_positive (f : ℝ → ℝ) :
  (¬ ∀ x > 0, f x > 0) ↔ (∃ x > 0, f x ≤ 0) := by
  sorry

end negation_of_forall_positive_l1734_173416


namespace population_size_l1734_173439

/-- Given a population with specific birth and death rates, prove the initial population size. -/
theorem population_size (birth_rate death_rate net_growth_rate : ℚ) : 
  birth_rate = 32 →
  death_rate = 11 →
  net_growth_rate = 21 / 1000 →
  (birth_rate - death_rate) / 1000 = net_growth_rate →
  1000 = (birth_rate - death_rate) / net_growth_rate :=
by sorry

end population_size_l1734_173439


namespace least_common_denominator_l1734_173493

theorem least_common_denominator : Nat.lcm 2 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 11))))) = 3960 := by
  sorry

end least_common_denominator_l1734_173493


namespace dogwood_trees_after_planting_l1734_173477

/-- The number of dogwood trees in the park after a week of planting -/
def total_trees (initial : ℕ) (monday tuesday wednesday thursday friday saturday sunday : ℕ) : ℕ :=
  initial + monday + tuesday + wednesday + thursday + friday + saturday + sunday

/-- Theorem stating the total number of dogwood trees after the week's planting -/
theorem dogwood_trees_after_planting :
  total_trees 7 3 2 5 1 6 4 3 = 31 := by
  sorry

end dogwood_trees_after_planting_l1734_173477


namespace compare_negative_fractions_l1734_173490

theorem compare_negative_fractions : -3/4 < -3/5 := by
  sorry

end compare_negative_fractions_l1734_173490


namespace complex_quadrant_l1734_173478

theorem complex_quadrant (z : ℂ) (h : (1 - I) * z = 3 + 5*I) : 
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end complex_quadrant_l1734_173478


namespace fraction_simplification_l1734_173473

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
sorry

end fraction_simplification_l1734_173473


namespace arithmetic_progression_x_value_l1734_173447

/-- An arithmetic progression with the first three terms 2x - 2, 2x + 2, and 4x + 6 has x = 0 --/
theorem arithmetic_progression_x_value :
  ∀ (x : ℝ), 
  let a₁ : ℝ := 2 * x - 2
  let a₂ : ℝ := 2 * x + 2
  let a₃ : ℝ := 4 * x + 6
  (a₂ - a₁ = a₃ - a₂) → x = 0 :=
by
  sorry

end arithmetic_progression_x_value_l1734_173447


namespace quadratic_equation_roots_l1734_173443

theorem quadratic_equation_roots (k : ℝ) :
  let f := fun x : ℝ => x^2 + (2*k - 1)*x + k^2
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0) →
  (k < 1/4 ∧
   (∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 = 0 → f x2 = 0 → x1 + x2 + x1*x2 - 1 = 0 → k = 0)) :=
by sorry

end quadratic_equation_roots_l1734_173443


namespace max_value_of_sum_over_square_n_l1734_173415

theorem max_value_of_sum_over_square_n (n : ℕ+) : 
  let S : ℕ+ → ℚ := fun k => (k * (k + 1)) / 2
  (S n) / (n^2 : ℚ) ≤ 9/16 := by sorry

end max_value_of_sum_over_square_n_l1734_173415


namespace grocer_coffee_stock_l1734_173480

/-- The amount of coffee initially in stock -/
def initial_stock : ℝ := 400

/-- The percentage of decaffeinated coffee in the initial stock -/
def initial_decaf_percent : ℝ := 0.20

/-- The amount of additional coffee purchased -/
def additional_coffee : ℝ := 100

/-- The percentage of decaffeinated coffee in the additional purchase -/
def additional_decaf_percent : ℝ := 0.60

/-- The final percentage of decaffeinated coffee after the purchase -/
def final_decaf_percent : ℝ := 0.28000000000000004

theorem grocer_coffee_stock :
  (initial_decaf_percent * initial_stock + additional_decaf_percent * additional_coffee) / 
  (initial_stock + additional_coffee) = final_decaf_percent := by
  sorry

end grocer_coffee_stock_l1734_173480


namespace counterexample_squared_inequality_l1734_173467

theorem counterexample_squared_inequality :
  ∃ (m n : ℝ), m > n ∧ m^2 ≤ n^2 := by sorry

end counterexample_squared_inequality_l1734_173467


namespace triangle_inequality_max_l1734_173485

theorem triangle_inequality_max (a b c x y z : ℝ) : 
  a > 0 → b > 0 → c > 0 → x > 0 → y > 0 → z > 0 →
  x + y + z = 1 →
  a * y * z + b * z * x + c * x * y ≤ 
    (a * b * c) / (-a^2 - b^2 - c^2 + 2 * (a * b + b * c + c * a)) :=
by sorry

end triangle_inequality_max_l1734_173485


namespace power_relation_l1734_173432

theorem power_relation (a m n : ℝ) (h1 : a^(m+n) = 8) (h2 : a^(m-n) = 2) : a^(2*n) = 4 := by
  sorry

end power_relation_l1734_173432


namespace circle_pattern_proof_l1734_173412

theorem circle_pattern_proof : 
  ∀ n : ℕ, (n * (n + 1)) / 2 ≤ 120 ∧ ((n + 1) * (n + 2)) / 2 > 120 → n = 14 :=
by sorry

end circle_pattern_proof_l1734_173412


namespace probability_no_brown_is_51_310_l1734_173417

def total_balls : ℕ := 32
def brown_balls : ℕ := 14
def non_brown_balls : ℕ := total_balls - brown_balls

def probability_no_brown : ℚ := (Nat.choose non_brown_balls 3 : ℚ) / (Nat.choose total_balls 3 : ℚ)

theorem probability_no_brown_is_51_310 : probability_no_brown = 51 / 310 := by
  sorry

end probability_no_brown_is_51_310_l1734_173417


namespace z_is_real_z_is_pure_imaginary_l1734_173488

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (m^2 - m - 2 : ℝ) + (m^2 + 3*m + 2 : ℝ) * Complex.I

-- Theorem for part (I)
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = -1 ∨ m = -2 := by sorry

-- Theorem for part (II)
theorem z_is_pure_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 := by sorry

end z_is_real_z_is_pure_imaginary_l1734_173488


namespace gigi_cookies_theorem_l1734_173462

/-- Represents the number of cups of flour per batch of cookies -/
def flour_per_batch : ℕ := 2

/-- Represents the initial amount of flour in cups -/
def initial_flour : ℕ := 20

/-- Represents the number of additional batches that can be made with remaining flour -/
def additional_batches : ℕ := 7

/-- Calculates the number of batches Gigi baked initially -/
def batches_baked : ℕ := (initial_flour - additional_batches * flour_per_batch) / flour_per_batch

theorem gigi_cookies_theorem : batches_baked = 3 := by
  sorry

end gigi_cookies_theorem_l1734_173462


namespace sum_of_three_numbers_l1734_173498

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 22 := by
sorry

end sum_of_three_numbers_l1734_173498


namespace ancient_chinese_math_problem_l1734_173436

theorem ancient_chinese_math_problem (a₁ : ℝ) : 
  (a₁ * (1 - (1/2)^6) / (1 - 1/2) = 378) →
  (a₁ * (1/2)^4 = 12) :=
by sorry

end ancient_chinese_math_problem_l1734_173436


namespace cindy_calculation_l1734_173464

theorem cindy_calculation (x : ℝ) (h : (x - 9) / 3 = 43) : (x - 3) / 9 = 15 := by
  sorry

end cindy_calculation_l1734_173464


namespace hannahs_quarters_l1734_173420

def is_valid_quarter_count (n : ℕ) : Prop :=
  40 < n ∧ n < 400 ∧
  n % 6 = 3 ∧
  n % 7 = 3 ∧
  n % 8 = 3

theorem hannahs_quarters :
  ∀ n : ℕ, is_valid_quarter_count n ↔ (n = 171 ∨ n = 339) :=
by sorry

end hannahs_quarters_l1734_173420


namespace isosceles_triangle_condition_l1734_173419

theorem isosceles_triangle_condition (a b : ℝ) (A B : ℝ) : 
  0 < a → 0 < b → 0 < A → A < π → 0 < B → B < π →
  a * Real.cos B = b * Real.cos A → A = B := by
  sorry

end isosceles_triangle_condition_l1734_173419


namespace fraction_value_l1734_173465

theorem fraction_value (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 4) :
  (x + y) / (x - y) = Real.sqrt 3 := by
  sorry

end fraction_value_l1734_173465


namespace solution_set_part1_range_of_b_part2_l1734_173448

-- Part 1
def quadratic_inequality (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c ≤ -1

theorem solution_set_part1 (a : ℝ) (h1 : a > 0) :
  let b := -2 * a - 2
  let c := 3
  (∀ x, quadratic_inequality a b c x ↔ 
    (0 < a ∧ a < 1 ∧ 2 ≤ x ∧ x ≤ 2/a) ∨
    (a = 1 ∧ x = 2) ∨
    (a > 1 ∧ 2/a ≤ x ∧ x ≤ 2)) := by sorry

-- Part 2
def quadratic_inequality_part2 (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c ≥ (3/2) * b * x

theorem range_of_b_part2 :
  ∃ b : ℝ, (∀ x, 1 ≤ x ∧ x ≤ 5 → quadratic_inequality_part2 1 b 2 x) ∧
    b ≤ 4 * Real.sqrt 2 := by sorry

end solution_set_part1_range_of_b_part2_l1734_173448


namespace parallel_vectors_tan_alpha_l1734_173424

/-- Given two parallel vectors a and b, prove that tan(α) = -1 -/
theorem parallel_vectors_tan_alpha (a b : ℝ × ℝ) (α : ℝ) :
  a = (Real.sqrt 2, -Real.sqrt 2) →
  b = (Real.cos α, Real.sin α) →
  (∃ (k : ℝ), a = k • b) →
  Real.tan α = -1 := by
  sorry

end parallel_vectors_tan_alpha_l1734_173424


namespace regular_polygon_150_degrees_has_12_sides_l1734_173409

/-- A regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides : 
  ∀ n : ℕ, 
  n > 2 →
  (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
  n = 12 := by
sorry

end regular_polygon_150_degrees_has_12_sides_l1734_173409


namespace baseball_team_grouping_l1734_173423

/-- Given the number of new players, returning players, and groups, 
    calculate the number of players in each group -/
def players_per_group (new_players returning_players groups : ℕ) : ℕ :=
  (new_players + returning_players) / groups

/-- Theorem stating that with 48 new players, 6 returning players, and 9 groups,
    there are 6 players in each group -/
theorem baseball_team_grouping :
  players_per_group 48 6 9 = 6 := by
  sorry

end baseball_team_grouping_l1734_173423


namespace mike_marbles_l1734_173421

theorem mike_marbles (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  initial = 8 → given = 4 → remaining = initial - given → remaining = 4 := by
  sorry

end mike_marbles_l1734_173421


namespace profit_maximization_l1734_173475

variable (x : ℝ)

def production_cost (x : ℝ) : ℝ := x^3 - 24*x^2 + 63*x + 10
def sales_revenue (x : ℝ) : ℝ := 18*x
def profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem profit_maximization (h : x > 0) :
  profit x = -x^3 + 24*x^2 - 45*x - 10 ∧
  ∃ (max_x : ℝ), max_x = 15 ∧
    ∀ (y : ℝ), y > 0 → profit y ≤ profit max_x ∧
    profit max_x = 1340 :=
by sorry

end profit_maximization_l1734_173475


namespace yoga_practice_mean_l1734_173442

/-- Represents the number of students practicing for each day --/
def practice_data : List (Nat × Nat) :=
  [(1, 2), (2, 4), (3, 5), (4, 3), (5, 2), (6, 1), (7, 3)]

/-- Calculates the total number of practice days --/
def total_days : Nat :=
  practice_data.foldl (fun acc (days, students) => acc + days * students) 0

/-- Calculates the total number of students --/
def total_students : Nat :=
  practice_data.foldl (fun acc (_, students) => acc + students) 0

/-- Calculates the mean number of practice days --/
def mean_practice_days : Rat :=
  total_days / total_students

theorem yoga_practice_mean :
  mean_practice_days = 37/10 := by sorry

end yoga_practice_mean_l1734_173442


namespace base_b_proof_l1734_173413

theorem base_b_proof (b : ℕ) (h : b > 1) :
  (7 * b^2 + 8 * b + 4 = (2 * b + 8)^2) → b = 10 := by
  sorry

end base_b_proof_l1734_173413
