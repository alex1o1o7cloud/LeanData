import Mathlib

namespace NUMINAMATH_CALUDE_trajectory_equation_l1908_190892

theorem trajectory_equation (x y : ℝ) (h : x ≠ 0) :
  (y + Real.sqrt 2) / x * (y - Real.sqrt 2) / x = -2 →
  y^2 / 2 + x^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_equation_l1908_190892


namespace NUMINAMATH_CALUDE_complex_modulus_product_l1908_190845

theorem complex_modulus_product : Complex.abs (5 - 3*Complex.I) * Complex.abs (5 + 3*Complex.I) = 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l1908_190845


namespace NUMINAMATH_CALUDE_bicycle_cost_price_l1908_190864

/-- Proves that the initial cost price of a bicycle is 112.5 given specific profit margins and final selling price -/
theorem bicycle_cost_price 
  (profit_a profit_b final_price : ℝ) 
  (h1 : profit_a = 0.6) 
  (h2 : profit_b = 0.25) 
  (h3 : final_price = 225) : 
  ∃ (initial_price : ℝ), 
    initial_price * (1 + profit_a) * (1 + profit_b) = final_price ∧ 
    initial_price = 112.5 := by
  sorry

#check bicycle_cost_price

end NUMINAMATH_CALUDE_bicycle_cost_price_l1908_190864


namespace NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_l1908_190873

theorem sphere_volume_equal_surface_area (r : ℝ) : 
  (4 / 3 : ℝ) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 := by
sorry

end NUMINAMATH_CALUDE_sphere_volume_equal_surface_area_l1908_190873


namespace NUMINAMATH_CALUDE_f_minimum_and_inequality_l1908_190870

def f (x : ℝ) := |2*x - 1| + |x - 3|

theorem f_minimum_and_inequality (x y : ℝ) :
  (∀ x, f x ≥ 5/2) ∧
  (∀ m, (∀ x y, f x > m * (|y + 1| - |y - 1|)) ↔ -5/4 < m ∧ m < 5/4) :=
by sorry

end NUMINAMATH_CALUDE_f_minimum_and_inequality_l1908_190870


namespace NUMINAMATH_CALUDE_difference_of_squares_2018_ways_l1908_190809

theorem difference_of_squares_2018_ways :
  ∃ (n : ℕ), n = 5^(2 * 2018) ∧
  (∃! (ways : Finset (ℕ × ℕ)), ways.card = 2018 ∧
    ∀ (a b : ℕ), (a, b) ∈ ways ↔ n = a^2 - b^2) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_squares_2018_ways_l1908_190809


namespace NUMINAMATH_CALUDE_paint_ratio_circular_signs_l1908_190823

theorem paint_ratio_circular_signs (d : ℝ) (h : d > 0) : 
  let D := 7 * d
  (π * (D / 2)^2) / (π * (d / 2)^2) = 49 := by sorry

end NUMINAMATH_CALUDE_paint_ratio_circular_signs_l1908_190823


namespace NUMINAMATH_CALUDE_terese_tuesday_run_l1908_190806

-- Define the days Terese runs
inductive RunDay
| monday
| tuesday
| wednesday
| thursday

-- Define a function that returns the distance run on each day
def distance_run (day : RunDay) : Real :=
  match day with
  | RunDay.monday => 4.2
  | RunDay.wednesday => 3.6
  | RunDay.thursday => 4.4
  | RunDay.tuesday => 3.8  -- This is what we want to prove

-- Define the average distance
def average_distance : Real := 4

-- Define the number of days Terese runs
def num_run_days : Nat := 4

-- Theorem statement
theorem terese_tuesday_run :
  (distance_run RunDay.monday +
   distance_run RunDay.tuesday +
   distance_run RunDay.wednesday +
   distance_run RunDay.thursday) / num_run_days = average_distance :=
by
  sorry


end NUMINAMATH_CALUDE_terese_tuesday_run_l1908_190806


namespace NUMINAMATH_CALUDE_correct_calculation_l1908_190879

theorem correct_calculation (a : ℝ) : -2*a + (2*a - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1908_190879


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l1908_190853

theorem price_reduction_percentage (initial_price final_price : ℝ) 
  (h1 : initial_price = 25)
  (h2 : final_price = 16)
  (h3 : initial_price > 0)
  (h4 : final_price > 0)
  (h5 : final_price < initial_price) :
  ∃ (x : ℝ), 
    x > 0 ∧ 
    x < 1 ∧ 
    initial_price * (1 - x)^2 = final_price ∧ 
    x = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l1908_190853


namespace NUMINAMATH_CALUDE_money_distribution_l1908_190828

theorem money_distribution (x : ℝ) (x_pos : x > 0) : 
  let total_money := 6*x + 5*x + 4*x + 3*x
  let ott_money := x + x + x + x
  ott_money / total_money = 2 / 9 := by
sorry


end NUMINAMATH_CALUDE_money_distribution_l1908_190828


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_neg_two_l1908_190859

theorem tan_pi_minus_alpha_neg_two (α : ℝ) (h : Real.tan (π - α) = -2) :
  (Real.cos (2 * π - α) + 2 * Real.cos ((3 * π) / 2 - α)) /
  (Real.sin (π - α) - Real.sin (-π / 2 - α)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_neg_two_l1908_190859


namespace NUMINAMATH_CALUDE_number_of_arrangements_l1908_190817

/-- Represents a checkout lane with two checkout points -/
structure CheckoutLane :=
  (point1 : Bool)
  (point2 : Bool)

/-- Represents the arrangement of checkout lanes -/
def Arrangement := List CheckoutLane

/-- The total number of checkout lanes -/
def totalLanes : Nat := 6

/-- The number of lanes to be selected -/
def selectedLanes : Nat := 3

/-- Checks if the lanes in an arrangement are non-adjacent -/
def areNonAdjacent (arr : Arrangement) : Bool :=
  sorry

/-- Checks if at least one checkout point is open in each lane -/
def hasOpenPoint (lane : CheckoutLane) : Bool :=
  lane.point1 || lane.point2

/-- Counts the number of valid arrangements -/
def countArrangements : Nat :=
  sorry

/-- The main theorem stating the number of valid arrangements -/
theorem number_of_arrangements :
  countArrangements = 108 :=
sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l1908_190817


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt11_l1908_190896

theorem consecutive_integers_around_sqrt11 (m n : ℤ) :
  (n = m + 1) →
  (m < Real.sqrt 11) →
  (Real.sqrt 11 < n) →
  m + n = 7 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt11_l1908_190896


namespace NUMINAMATH_CALUDE_soap_bubble_radius_l1908_190877

noncomputable def final_radius (α : ℝ) (r₀ : ℝ) (U : ℝ) (k : ℝ) : ℝ :=
  (U^2 * r₀^2 / (32 * k * Real.pi * α))^(1/3)

theorem soap_bubble_radius (α : ℝ) (r₀ : ℝ) (U : ℝ) (k : ℝ) 
  (h1 : α > 0) (h2 : r₀ > 0) (h3 : U ≠ 0) (h4 : k > 0) :
  ∃ (r : ℝ), r = final_radius α r₀ U k ∧ 
  r = (U^2 * r₀^2 / (32 * k * Real.pi * α))^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_soap_bubble_radius_l1908_190877


namespace NUMINAMATH_CALUDE_M_subset_P_l1908_190842

def U : Set ℝ := Set.univ

def M : Set ℝ := {x : ℝ | x > 1}

def P : Set ℝ := {x : ℝ | x^2 > 1}

theorem M_subset_P : M ⊆ P := by sorry

end NUMINAMATH_CALUDE_M_subset_P_l1908_190842


namespace NUMINAMATH_CALUDE_zoo_animal_count_l1908_190858

/-- The number of tiger enclosures in the zoo -/
def tiger_enclosures : ℕ := 4

/-- The number of zebra enclosures behind each tiger enclosure -/
def zebras_per_tiger : ℕ := 2

/-- The number of tigers in each tiger enclosure -/
def tigers_per_enclosure : ℕ := 4

/-- The number of zebras in each zebra enclosure -/
def zebras_per_enclosure : ℕ := 10

/-- The number of giraffes in each giraffe enclosure -/
def giraffes_per_enclosure : ℕ := 2

/-- The ratio of giraffe enclosures to zebra enclosures -/
def giraffe_to_zebra_ratio : ℕ := 3

/-- The total number of zebra enclosures in the zoo -/
def total_zebra_enclosures : ℕ := tiger_enclosures * zebras_per_tiger

/-- The total number of giraffe enclosures in the zoo -/
def total_giraffe_enclosures : ℕ := total_zebra_enclosures * giraffe_to_zebra_ratio

/-- The total number of animals in the zoo -/
def total_animals : ℕ := 
  tiger_enclosures * tigers_per_enclosure + 
  total_zebra_enclosures * zebras_per_enclosure + 
  total_giraffe_enclosures * giraffes_per_enclosure

theorem zoo_animal_count : total_animals = 144 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_count_l1908_190858


namespace NUMINAMATH_CALUDE_cube_face_perimeter_l1908_190890

-- Define the volume of the cube
def cube_volume : ℝ := 216

-- Define the function to calculate the side length of a cube given its volume
def side_length (volume : ℝ) : ℝ := volume ^ (1/3)

-- Define the function to calculate the perimeter of a square given its side length
def square_perimeter (side : ℝ) : ℝ := 4 * side

-- Theorem statement
theorem cube_face_perimeter :
  square_perimeter (side_length cube_volume) = 24 := by
  sorry

end NUMINAMATH_CALUDE_cube_face_perimeter_l1908_190890


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_equation_l1908_190852

theorem product_of_solutions_abs_equation : 
  ∃ (y₁ y₂ : ℝ), (|y₁| = 3*(|y₁| - 2)) ∧ (|y₂| = 3*(|y₂| - 2)) ∧ (y₁ ≠ y₂) ∧ (y₁ * y₂ = -9) :=
sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_equation_l1908_190852


namespace NUMINAMATH_CALUDE_egg_grouping_l1908_190841

theorem egg_grouping (total_eggs : ℕ) (group_size : ℕ) (h1 : total_eggs = 16) (h2 : group_size = 2) :
  total_eggs / group_size = 8 := by
  sorry

end NUMINAMATH_CALUDE_egg_grouping_l1908_190841


namespace NUMINAMATH_CALUDE_exponent_multiplication_l1908_190876

theorem exponent_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l1908_190876


namespace NUMINAMATH_CALUDE_bouncing_ball_distance_l1908_190803

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (reboundFactor : ℝ) (bounces : ℕ) : ℝ :=
  sorry

/-- The bouncing ball theorem -/
theorem bouncing_ball_distance :
  totalDistance 160 (3/4) 4 = 816.25 := by sorry

end NUMINAMATH_CALUDE_bouncing_ball_distance_l1908_190803


namespace NUMINAMATH_CALUDE_sum_ab_over_c_squared_plus_one_le_one_l1908_190848

theorem sum_ab_over_c_squared_plus_one_le_one 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (sum_eq_two : a + b + c = 2) :
  (a * b) / (c^2 + 1) + (b * c) / (a^2 + 1) + (c * a) / (b^2 + 1) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_ab_over_c_squared_plus_one_le_one_l1908_190848


namespace NUMINAMATH_CALUDE_train_length_l1908_190819

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 90 → time = 4 → speed * time * (1000 / 3600) = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1908_190819


namespace NUMINAMATH_CALUDE_equal_reciprocal_sum_l1908_190888

theorem equal_reciprocal_sum (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x = 2 + 1 / y) (h2 : y = 2 + 1 / x) : y = x := by
  sorry

end NUMINAMATH_CALUDE_equal_reciprocal_sum_l1908_190888


namespace NUMINAMATH_CALUDE_circle_equation_radius_l1908_190899

theorem circle_equation_radius (k : ℝ) :
  (∃ (h : ℝ) (v : ℝ),
    ∀ (x y : ℝ),
      x^2 + 14*x + y^2 + 8*y - k = 0 ↔ (x - h)^2 + (y - v)^2 = 10^2) ↔
  k = 35 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l1908_190899


namespace NUMINAMATH_CALUDE_simplify_expression_l1908_190808

theorem simplify_expression (a : ℝ) : (-a^2)^3 * 3*a = -3*a^7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1908_190808


namespace NUMINAMATH_CALUDE_remaining_oranges_l1908_190891

/-- The number of oranges Michaela needs to get full -/
def michaela_oranges : ℕ := 45

/-- The number of oranges Cassandra needs to get full -/
def cassandra_oranges : ℕ := 5 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 520

/-- The number of oranges remaining after Michaela and Cassandra have eaten until full -/
theorem remaining_oranges : total_oranges - (michaela_oranges + cassandra_oranges) = 250 := by
  sorry

end NUMINAMATH_CALUDE_remaining_oranges_l1908_190891


namespace NUMINAMATH_CALUDE_inequality_proof_l1908_190881

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  Real.sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (((a*b*c + a*b*d + a*c*d + b*c*d) / 4) ^ (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1908_190881


namespace NUMINAMATH_CALUDE_equation_comparison_l1908_190818

theorem equation_comparison : 
  (abs (-2))^3 = abs 2^3 ∧ 
  (-2)^2 = 2^2 ∧ 
  (-2)^3 = -(2^3) ∧ 
  (-2)^4 ≠ -(2^4) := by
sorry

end NUMINAMATH_CALUDE_equation_comparison_l1908_190818


namespace NUMINAMATH_CALUDE_fraction_problem_l1908_190878

theorem fraction_problem (x : ℝ) : 
  (0.60 * x * 100 = 36) → x = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l1908_190878


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1908_190867

theorem trigonometric_identities (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 = 1 - 2 * Real.cos A * Real.cos B * Real.cos C ∧
  (Real.sin A)^2 + (Real.sin B)^2 + (Real.sin C)^2 = 2 * Real.cos A * Real.cos B * Real.cos C :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1908_190867


namespace NUMINAMATH_CALUDE_combined_rectangle_perimeter_l1908_190866

/-- The perimeter of a rectangle formed by combining a square of side 8 cm
    with a rectangle of dimensions 8 cm x 4 cm is 48 cm. -/
theorem combined_rectangle_perimeter :
  let square_side : ℝ := 8
  let rect_length : ℝ := 8
  let rect_width : ℝ := 4
  let new_rect_length : ℝ := square_side + rect_length
  let new_rect_width : ℝ := square_side
  let perimeter : ℝ := 2 * (new_rect_length + new_rect_width)
  perimeter = 48 := by sorry

end NUMINAMATH_CALUDE_combined_rectangle_perimeter_l1908_190866


namespace NUMINAMATH_CALUDE_training_hours_per_day_l1908_190898

/-- 
Given a person who trains for a constant number of hours per day over a period of time,
this theorem proves that if the total training period is 42 days and the total training time
is 210 hours, then the person trains for 5 hours every day.
-/
theorem training_hours_per_day 
  (total_days : ℕ) 
  (total_hours : ℕ) 
  (hours_per_day : ℕ) 
  (h1 : total_days = 42) 
  (h2 : total_hours = 210) 
  (h3 : total_hours = total_days * hours_per_day) : 
  hours_per_day = 5 := by
  sorry

end NUMINAMATH_CALUDE_training_hours_per_day_l1908_190898


namespace NUMINAMATH_CALUDE_dice_probability_theorem_l1908_190836

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 6

/-- The total number of possible outcomes when rolling 6 dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (at least one pair but not a four-of-a-kind) -/
def favorableOutcomes : ℕ := 28800

/-- The probability of getting at least one pair but not a four-of-a-kind when rolling 6 dice -/
def probabilityPairNotFourOfAKind : ℚ := favorableOutcomes / totalOutcomes

theorem dice_probability_theorem : probabilityPairNotFourOfAKind = 25 / 81 := by
  sorry

end NUMINAMATH_CALUDE_dice_probability_theorem_l1908_190836


namespace NUMINAMATH_CALUDE_laptop_price_l1908_190840

theorem laptop_price : ∃ (x : ℝ), 
  (0.855 * x - 50) = (0.88 * x - 20) - 30 ∧ x = 2400 := by
  sorry

end NUMINAMATH_CALUDE_laptop_price_l1908_190840


namespace NUMINAMATH_CALUDE_order_cost_proof_l1908_190897

def english_book_cost : ℝ := 7.50
def geography_book_cost : ℝ := 10.50
def num_books : ℕ := 35

def total_cost : ℝ := num_books * english_book_cost + num_books * geography_book_cost

theorem order_cost_proof : total_cost = 630 := by
  sorry

end NUMINAMATH_CALUDE_order_cost_proof_l1908_190897


namespace NUMINAMATH_CALUDE_study_group_formation_l1908_190824

def number_of_ways (n : ℕ) (g2 : ℕ) (g3 : ℕ) : ℕ :=
  (Nat.choose n 3 * Nat.choose (n - 3) 3) / 2 *
  ((Nat.choose (n - 6) 2 * Nat.choose (n - 8) 2 * Nat.choose (n - 10) 2) / 6)

theorem study_group_formation :
  number_of_ways 12 3 2 = 138600 := by
  sorry

end NUMINAMATH_CALUDE_study_group_formation_l1908_190824


namespace NUMINAMATH_CALUDE_smallest_shift_l1908_190862

-- Define the function f with the given property
def f_periodic (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 12) = f x

-- Define the property for the shifted function
def shifted_f_equal (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f ((x - a) / 3) = f (x / 3)

-- Theorem statement
theorem smallest_shift (f : ℝ → ℝ) (h : f_periodic f) :
  (∃ a : ℝ, a > 0 ∧ shifted_f_equal f a ∧
    ∀ b : ℝ, b > 0 ∧ shifted_f_equal f b → a ≤ b) →
  ∃ a : ℝ, a = 36 ∧ shifted_f_equal f a ∧
    ∀ b : ℝ, b > 0 ∧ shifted_f_equal f b → a ≤ b :=
sorry

end NUMINAMATH_CALUDE_smallest_shift_l1908_190862


namespace NUMINAMATH_CALUDE_sum_first_five_eq_l1908_190807

/-- A geometric progression with given fourth and fifth terms -/
structure GeometricProgression where
  b₄ : ℚ  -- Fourth term
  b₅ : ℚ  -- Fifth term
  h₄ : b₄ = 1 / 25
  h₅ : b₅ = 1 / 125

/-- The sum of the first five terms of a geometric progression -/
def sum_first_five (gp : GeometricProgression) : ℚ :=
  -- Definition of the sum (to be proved)
  781 / 125

/-- Theorem stating that the sum of the first five terms is 781/125 -/
theorem sum_first_five_eq (gp : GeometricProgression) :
  sum_first_five gp = 781 / 125 := by
  sorry

#eval sum_first_five ⟨1/25, 1/125, rfl, rfl⟩

end NUMINAMATH_CALUDE_sum_first_five_eq_l1908_190807


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1908_190886

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.I : ℂ).re = 0 ∧ (Complex.I : ℂ).im = 1 →
  (((2 : ℂ) - a * Complex.I) / ((1 : ℂ) + Complex.I)).re = 0 →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1908_190886


namespace NUMINAMATH_CALUDE_age_difference_l1908_190810

theorem age_difference (x y z : ℕ) (h1 : z = x - 19) : x + y - (y + z) = 19 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1908_190810


namespace NUMINAMATH_CALUDE_arithmetic_sum_10_terms_l1908_190857

def arithmetic_sum (a : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sum_10_terms : arithmetic_sum (-2) 7 10 = 295 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_10_terms_l1908_190857


namespace NUMINAMATH_CALUDE_language_study_difference_l1908_190804

theorem language_study_difference (total : ℕ) (german_min german_max russian_min russian_max : ℕ) :
  total = 2500 →
  german_min = 1750 →
  german_max = 1875 →
  russian_min = 1000 →
  russian_max = 1125 →
  let m := german_min + russian_min - total
  let M := german_max + russian_max - total
  M - m = 250 := by sorry

end NUMINAMATH_CALUDE_language_study_difference_l1908_190804


namespace NUMINAMATH_CALUDE_product_of_squares_l1908_190821

theorem product_of_squares (a b : ℝ) 
  (h1 : a^2 - b^2 = 10) 
  (h2 : a^4 + b^4 = 228) : 
  a * b = 8 := by
sorry

end NUMINAMATH_CALUDE_product_of_squares_l1908_190821


namespace NUMINAMATH_CALUDE_digit_distribution_proof_l1908_190863

theorem digit_distribution_proof (n : ℕ) 
  (h1 : n / 2 = n * (1 / 2 : ℚ))  -- 1/2 of all digits are 1
  (h2 : n / 5 = n * (1 / 5 : ℚ))  -- proportion of 2 and 5 are 1/5 each
  (h3 : n / 10 = n * (1 / 10 : ℚ))  -- proportion of other digits is 1/10
  (h4 : (1 / 2 : ℚ) + (1 / 5 : ℚ) + (1 / 5 : ℚ) + (1 / 10 : ℚ) = 1)  -- sum of all proportions is 1
  : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_digit_distribution_proof_l1908_190863


namespace NUMINAMATH_CALUDE_white_dogs_count_l1908_190815

theorem white_dogs_count (total brown black : ℕ) 
  (h_total : total = 45)
  (h_brown : brown = 20)
  (h_black : black = 15) :
  total - (brown + black) = 10 := by
  sorry

end NUMINAMATH_CALUDE_white_dogs_count_l1908_190815


namespace NUMINAMATH_CALUDE_triangle_perimeter_in_divided_square_l1908_190868

/-- Given a square of side length z divided into a central rectangle and four congruent right-angled triangles,
    where the shorter side of the rectangle is x, the perimeter of one of the triangles is 3z/2. -/
theorem triangle_perimeter_in_divided_square (z x : ℝ) (hz : z > 0) (hx : 0 < x ∧ x < z) :
  let triangle_perimeter := (z - x) / 2 + (z + x) / 2 + z / 2
  triangle_perimeter = 3 * z / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_in_divided_square_l1908_190868


namespace NUMINAMATH_CALUDE_integer_sqrt_pair_l1908_190816

theorem integer_sqrt_pair : ∃! (x y : ℕ), 
  ((x = 88209 ∧ y = 90288) ∨
   (x = 82098 ∧ y = 89028) ∨
   (x = 28098 ∧ y = 89082) ∨
   (x = 90882 ∧ y = 28809)) ∧
  ∃ (z : ℕ), z^2 = x^2 + y^2 := by
sorry

end NUMINAMATH_CALUDE_integer_sqrt_pair_l1908_190816


namespace NUMINAMATH_CALUDE_base8_47_equals_39_l1908_190801

/-- Converts a two-digit base-8 number to base-10 --/
def base8_to_base10 (tens : Nat) (ones : Nat) : Nat :=
  tens * 8 + ones

/-- The base-8 number 47 is equal to 39 in base-10 --/
theorem base8_47_equals_39 : base8_to_base10 4 7 = 39 := by
  sorry

end NUMINAMATH_CALUDE_base8_47_equals_39_l1908_190801


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1908_190861

theorem magnitude_of_z (z : ℂ) : z + Complex.I = 3 → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1908_190861


namespace NUMINAMATH_CALUDE_existence_of_sequences_l1908_190893

theorem existence_of_sequences : ∃ (a b : ℕ → ℝ), 
  (∀ i : ℕ, 3 * Real.pi / 2 ≤ a i ∧ a i ≤ b i) ∧
  (∀ i : ℕ, ∀ x : ℝ, 0 < x ∧ x < 1 → Real.cos (a i * x) - Real.cos (b i * x) ≥ -1 / i) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_sequences_l1908_190893


namespace NUMINAMATH_CALUDE_coin_count_l1908_190884

theorem coin_count (total_value : ℕ) (nickel_value dime_value quarter_value : ℕ) :
  total_value = 360 →
  nickel_value = 5 →
  dime_value = 10 →
  quarter_value = 25 →
  ∃ (x : ℕ), x * (nickel_value + dime_value + quarter_value) = total_value ∧
              3 * x = 27 :=
by
  sorry

#check coin_count

end NUMINAMATH_CALUDE_coin_count_l1908_190884


namespace NUMINAMATH_CALUDE_deceased_member_income_l1908_190811

theorem deceased_member_income
  (initial_members : ℕ)
  (initial_average : ℚ)
  (final_members : ℕ)
  (final_average : ℚ)
  (h1 : initial_members = 4)
  (h2 : initial_average = 735)
  (h3 : final_members = 3)
  (h4 : final_average = 590)
  : (initial_members : ℚ) * initial_average - (final_members : ℚ) * final_average = 1170 :=
by sorry

end NUMINAMATH_CALUDE_deceased_member_income_l1908_190811


namespace NUMINAMATH_CALUDE_intersection_of_complex_equations_l1908_190869

open Complex

theorem intersection_of_complex_equations (k : ℝ) : 
  (∃! z : ℂ, (Complex.abs (z - 3) = 3 * Complex.abs (z + 3)) ∧ (Complex.abs z = k)) ↔ k = 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_complex_equations_l1908_190869


namespace NUMINAMATH_CALUDE_star_equation_solution_l1908_190812

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem star_equation_solution (h : ℝ) :
  star 8 h = 11 → h = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1908_190812


namespace NUMINAMATH_CALUDE_intersection_product_l1908_190894

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 - 4*x + y^2 - 6*y + 9 = 0
def circle2 (x y : ℝ) : Prop := x^2 - 8*x + y^2 - 6*y + 21 = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | circle1 p.1 p.2 ∧ circle2 p.1 p.2}

-- Theorem statement
theorem intersection_product : 
  ∀ p ∈ intersection_points, p.1 * p.2 = 12 := by sorry

end NUMINAMATH_CALUDE_intersection_product_l1908_190894


namespace NUMINAMATH_CALUDE_system_solution_l1908_190829

theorem system_solution :
  let f (x y : ℝ) := x * Real.sqrt (1 - y^2) = (Real.sqrt 3 + 1) / 4
  let g (x y : ℝ) := y * Real.sqrt (1 - x^2) = (Real.sqrt 3 - 1) / 4
  ∀ x y : ℝ, (f x y ∧ g x y) ↔ 
    ((x = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧ y = Real.sqrt 2 / 2) ∨
     (x = Real.sqrt 2 / 2 ∧ y = (Real.sqrt 6 - Real.sqrt 2) / 4)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1908_190829


namespace NUMINAMATH_CALUDE_mode_of_scores_l1908_190832

def Scores := List Nat

def count (n : Nat) (scores : Scores) : Nat :=
  scores.filter (· = n) |>.length

def isMode (n : Nat) (scores : Scores) : Prop :=
  ∀ m, count n scores ≥ count m scores

theorem mode_of_scores (scores : Scores) 
  (h1 : scores.all (· ≤ 120))
  (h2 : count 91 scores = 5)
  (h3 : ∀ n, n ≠ 91 → count n scores ≤ 5) :
  isMode 91 scores :=
sorry

end NUMINAMATH_CALUDE_mode_of_scores_l1908_190832


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1908_190835

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- Define the theorem
theorem quadratic_function_properties
  (b c : ℝ)
  (h1 : f b c 2 = f b c (-2))
  (h2 : f b c 1 = 0) :
  (∀ x, f b c x = x^2 - 1) ∧
  (∀ m : ℝ, (∀ x ≥ (1/2 : ℝ), ∃ y, 4*m*(f b c y) + f b c (y-1) = 4-4*m) →
    -1/4 < m ∧ m ≤ 19/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1908_190835


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l1908_190847

open Set

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {2, 3}

theorem intersection_complement_equality : M ∩ (U \ N) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l1908_190847


namespace NUMINAMATH_CALUDE_line_through_midpoint_l1908_190837

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := x - 3*y + 10 = 0
def l2 (x y : ℝ) : Prop := 2*x + y - 8 = 0

-- Define point P
def P : ℝ × ℝ := (0, 1)

-- Define the line l
def l (x y : ℝ) : Prop := x + 4*y - 4 = 0

-- Theorem statement
theorem line_through_midpoint (A B : ℝ × ℝ) :
  l A.1 A.2 →
  l B.1 B.2 →
  l1 A.1 A.2 →
  l2 B.1 B.2 →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∀ x y, l x y ↔ x + 4*y - 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_line_through_midpoint_l1908_190837


namespace NUMINAMATH_CALUDE_ellipse_min_area_l1908_190834

/-- An ellipse containing two specific circles has a minimum area of π -/
theorem ellipse_min_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → 
    ((x - 2)^2 + y^2 ≤ 4 ∧ (x + 2)^2 + y^2 ≤ 4)) → 
  π * a * b ≥ π := by sorry

end NUMINAMATH_CALUDE_ellipse_min_area_l1908_190834


namespace NUMINAMATH_CALUDE_isosceles_triangle_exists_l1908_190820

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A coloring of the vertices of a polygon with two colors -/
def Coloring (n : ℕ) := Fin n → Bool

/-- An isosceles triangle in a regular polygon -/
def IsIsoscelesTriangle (p : RegularPolygon n) (v1 v2 v3 : Fin n) : Prop :=
  let d12 := dist (p.vertices v1) (p.vertices v2)
  let d23 := dist (p.vertices v2) (p.vertices v3)
  let d31 := dist (p.vertices v3) (p.vertices v1)
  d12 = d23 ∨ d23 = d31 ∨ d31 = d12

/-- The main theorem -/
theorem isosceles_triangle_exists (p : RegularPolygon 13) (c : Coloring 13) :
  ∃ (v1 v2 v3 : Fin 13), c v1 = c v2 ∧ c v2 = c v3 ∧ IsIsoscelesTriangle p v1 v2 v3 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_exists_l1908_190820


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1908_190882

theorem fraction_evaluation (x : ℝ) (h : x = 6) : 3 / (2 - 3 / x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1908_190882


namespace NUMINAMATH_CALUDE_abs_sum_nonzero_iff_either_nonzero_l1908_190871

theorem abs_sum_nonzero_iff_either_nonzero (x y : ℝ) :
  (abs x + abs y ≠ 0) ↔ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_nonzero_iff_either_nonzero_l1908_190871


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l1908_190883

theorem similar_triangle_perimeter (a b c d e : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem for smaller triangle
  (d/a)^2 + (e/b)^2 = 1 →  -- Similar triangles condition
  2*c = 30 →  -- Hypotenuse of larger triangle
  d + e + 30 = 72 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l1908_190883


namespace NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1908_190830

theorem least_common_multiple_first_ten : ∃ n : ℕ, 
  (∀ i : ℕ, i > 0 ∧ i ≤ 10 → n % i = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (∀ i : ℕ, i > 0 ∧ i ≤ 10 → m % i = 0) → m ≥ n) ∧
  n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_least_common_multiple_first_ten_l1908_190830


namespace NUMINAMATH_CALUDE_jerrys_action_figures_l1908_190854

theorem jerrys_action_figures :
  ∀ (initial_figures initial_books added_figures : ℕ),
    initial_figures = 2 →
    initial_books = 10 →
    initial_books = (initial_figures + added_figures) + 4 →
    added_figures = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_jerrys_action_figures_l1908_190854


namespace NUMINAMATH_CALUDE_sanctuary_bird_pairs_l1908_190851

/-- The number of endangered bird species in Tyler's sanctuary -/
def num_species : ℕ := 29

/-- The number of pairs of birds per species -/
def pairs_per_species : ℕ := 7

/-- The total number of pairs of birds in Tyler's sanctuary -/
def total_pairs : ℕ := num_species * pairs_per_species

theorem sanctuary_bird_pairs : total_pairs = 203 := by
  sorry

end NUMINAMATH_CALUDE_sanctuary_bird_pairs_l1908_190851


namespace NUMINAMATH_CALUDE_percentage_relationship_l1908_190802

theorem percentage_relationship (a b c : ℝ) (h1 : 0.06 * a = 10) (h2 : c = b / a) :
  ∃ p : ℝ, p * b = 6 ∧ p * 100 = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relationship_l1908_190802


namespace NUMINAMATH_CALUDE_specific_triangle_area_l1908_190825

/-- RightTriangle represents a right triangle with specific properties -/
structure RightTriangle where
  AB : ℝ  -- Length of hypotenuse
  median_CA : ℝ → ℝ  -- Equation of median to side CA
  median_CB : ℝ → ℝ  -- Equation of median to side CB

/-- Calculate the area of the right triangle -/
def triangle_area (t : RightTriangle) : ℝ := sorry

/-- Theorem stating the area of the specific right triangle -/
theorem specific_triangle_area :
  let t : RightTriangle := {
    AB := 60,
    median_CA := λ x => x + 3,
    median_CB := λ x => 2 * x + 4
  }
  triangle_area t = 400 := by sorry

end NUMINAMATH_CALUDE_specific_triangle_area_l1908_190825


namespace NUMINAMATH_CALUDE_discount_markup_percentage_l1908_190838

theorem discount_markup_percentage (original_price : ℝ) (discount_rate : ℝ) (h1 : discount_rate = 0.2) :
  let discounted_price := original_price * (1 - discount_rate)
  let markup_rate := (original_price - discounted_price) / discounted_price
  markup_rate = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_discount_markup_percentage_l1908_190838


namespace NUMINAMATH_CALUDE_ellipse_properties_l1908_190822

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_minor_axis : b = Real.sqrt 3
  h_eccentricity : a / Real.sqrt (a^2 - b^2) = 2

/-- The standard form of the ellipse -/
def standard_form (e : Ellipse) : Prop :=
  e.a^2 = 4 ∧ e.b^2 = 3

/-- The maximum area of triangle F₁AB -/
def max_triangle_area (e : Ellipse) : ℝ := 3

/-- Main theorem stating the properties of the ellipse -/
theorem ellipse_properties (e : Ellipse) :
  standard_form e ∧ max_triangle_area e = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l1908_190822


namespace NUMINAMATH_CALUDE_dan_remaining_money_l1908_190885

/-- Given an initial amount and a spent amount, calculate the remaining amount --/
def remaining_amount (initial : ℚ) (spent : ℚ) : ℚ :=
  initial - spent

/-- Proof that Dan has $1 left --/
theorem dan_remaining_money :
  let initial_amount : ℚ := 4
  let spent_amount : ℚ := 3
  remaining_amount initial_amount spent_amount = 1 := by
  sorry

end NUMINAMATH_CALUDE_dan_remaining_money_l1908_190885


namespace NUMINAMATH_CALUDE_quadratic_equation_result_l1908_190860

theorem quadratic_equation_result (y : ℂ) : 
  3 * y^2 + 2 * y + 1 = 0 → (6 * y + 5)^2 = -7 + 12 * Complex.I * Real.sqrt 2 ∨ 
                              (6 * y + 5)^2 = -7 - 12 * Complex.I * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_result_l1908_190860


namespace NUMINAMATH_CALUDE_smallest_angle_sine_cosine_equality_l1908_190849

theorem smallest_angle_sine_cosine_equality : 
  ∃ x : ℝ, x > 0 ∧ x < (2 * Real.pi / 360) * 11 ∧
    Real.sin (4 * x) * Real.sin (5 * x) = Real.cos (4 * x) * Real.cos (5 * x) ∧
    ∀ y : ℝ, 0 < y ∧ y < x → 
      Real.sin (4 * y) * Real.sin (5 * y) ≠ Real.cos (4 * y) * Real.cos (5 * y) ∧
    x = (Real.pi / 18) := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_sine_cosine_equality_l1908_190849


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1908_190880

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 4, 5}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1908_190880


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1908_190800

theorem condition_sufficient_not_necessary :
  (∀ x : ℝ, x > 2 → x^2 - 2*x > 0) ∧
  (∃ x : ℝ, x^2 - 2*x > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l1908_190800


namespace NUMINAMATH_CALUDE_students_without_vision_assistance_l1908_190850

/-- Given a group of 40 students where 25% wear glasses and 40% wear contact lenses,
    prove that 14 students do not wear any vision assistance wear. -/
theorem students_without_vision_assistance (total_students : ℕ) (glasses_percent : ℚ) (contacts_percent : ℚ) :
  total_students = 40 →
  glasses_percent = 25 / 100 →
  contacts_percent = 40 / 100 →
  total_students - (glasses_percent * total_students + contacts_percent * total_students) = 14 := by
  sorry

end NUMINAMATH_CALUDE_students_without_vision_assistance_l1908_190850


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1908_190805

theorem rectangle_perimeter (area : ℝ) (length : ℝ) (h1 : area = 192) (h2 : length = 24) :
  2 * (length + area / length) = 64 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1908_190805


namespace NUMINAMATH_CALUDE_exists_n_ratio_f_g_eq_2012_l1908_190846

/-- The number of divisors of n which are perfect squares -/
def f (n : ℕ+) : ℕ := sorry

/-- The number of divisors of n which are perfect cubes -/
def g (n : ℕ+) : ℕ := sorry

/-- There exists a positive integer n such that f(n) / g(n) = 2012 -/
theorem exists_n_ratio_f_g_eq_2012 : ∃ n : ℕ+, (f n : ℚ) / (g n : ℚ) = 2012 := by sorry

end NUMINAMATH_CALUDE_exists_n_ratio_f_g_eq_2012_l1908_190846


namespace NUMINAMATH_CALUDE_prob_same_color_equal_one_l1908_190833

/-- Procedure A: Choose one card from k cards with equal probability 1/k and replace it with a different color card. -/
def procedureA (k : ℕ) : Unit := sorry

/-- The probability of reaching a state where all cards are of the same color after n repetitions of procedure A. -/
def probSameColor (k n : ℕ) : ℝ := sorry

theorem prob_same_color_equal_one (k n : ℕ) (h1 : k > 0) (h2 : n > 0) (h3 : k % 2 = 0) :
  probSameColor k n = 1 := by sorry

end NUMINAMATH_CALUDE_prob_same_color_equal_one_l1908_190833


namespace NUMINAMATH_CALUDE_pirate_treasure_l1908_190887

theorem pirate_treasure (m : ℕ) (n : ℕ) (u : ℕ) : 
  (2/3 * (2/3 * (2/3 * (m - 1) - 1) - 1) = 3 * n) →
  (110 ≤ 81 * u + 25) →
  (81 * u + 25 ≤ 200) →
  (m = 187 ∧ 
   1 + (187 - 1) / 3 + 18 = 81 ∧
   1 + (187 - (1 + (187 - 1) / 3) - 1) / 3 + 18 = 60 ∧
   1 + (187 - (1 + (187 - 1) / 3) - (1 + (187 - (1 + (187 - 1) / 3) - 1) / 3) - 1) / 3 + 18 = 46) :=
by sorry

end NUMINAMATH_CALUDE_pirate_treasure_l1908_190887


namespace NUMINAMATH_CALUDE_mn_pq_ratio_l1908_190826

-- Define the points on a real line
variable (A B C M N P Q : ℝ)

-- Define the conditions
variable (h1 : A ≤ B ∧ B ≤ C)  -- B is on line segment AC
variable (h2 : M = (A + B) / 2)  -- M is midpoint of AB
variable (h3 : N = (A + C) / 2)  -- N is midpoint of AC
variable (h4 : P = (N + A) / 2)  -- P is midpoint of NA
variable (h5 : Q = (M + A) / 2)  -- Q is midpoint of MA

-- State the theorem
theorem mn_pq_ratio :
  |N - M| / |P - Q| = 2 :=
sorry

end NUMINAMATH_CALUDE_mn_pq_ratio_l1908_190826


namespace NUMINAMATH_CALUDE_solution_correct_l1908_190865

def M : Matrix (Fin 2) (Fin 2) ℚ := !![5, 2; 4, 1]
def N : Matrix (Fin 2) (Fin 1) ℚ := !![5; 8]
def X : Matrix (Fin 2) (Fin 1) ℚ := !![11/3; -20/3]

theorem solution_correct : M * X = N := by sorry

end NUMINAMATH_CALUDE_solution_correct_l1908_190865


namespace NUMINAMATH_CALUDE_distinct_triangles_in_3x3_grid_l1908_190839

/-- The number of points in a row or column of the grid -/
def gridSize : Nat := 3

/-- The total number of points in the grid -/
def totalPoints : Nat := gridSize * gridSize

/-- The number of ways to choose 3 points from the total points -/
def totalCombinations : Nat := Nat.choose totalPoints 3

/-- The number of sets of collinear points in the grid -/
def collinearSets : Nat := 2 * gridSize + 2

/-- The number of distinct triangles in a 3x3 grid -/
def distinctTriangles : Nat := totalCombinations - collinearSets

theorem distinct_triangles_in_3x3_grid :
  distinctTriangles = 76 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_3x3_grid_l1908_190839


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1908_190814

/-- Given a hyperbola with the equation y²/a² - x²/b² = l, where a > 0 and b > 0,
    if the point (1, 2) lies on the hyperbola, then its eccentricity e is greater than √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  4/a^2 - 1/b^2 = 1 → ∃ e : ℝ, e > Real.sqrt 5 / 2 ∧ e^2 = (a^2 + b^2)/a^2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1908_190814


namespace NUMINAMATH_CALUDE_choose_four_from_ten_l1908_190827

theorem choose_four_from_ten : Nat.choose 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_ten_l1908_190827


namespace NUMINAMATH_CALUDE_first_investment_interest_rate_l1908_190856

/-- Prove that the annual simple interest rate of the first investment is 8.5% --/
theorem first_investment_interest_rate 
  (total_income : ℝ) 
  (total_investment : ℝ) 
  (first_investment : ℝ) 
  (second_rate : ℝ) 
  (h1 : total_income = 575)
  (h2 : total_investment = 8000)
  (h3 : first_investment = 3000)
  (h4 : second_rate = 0.064)
  (h5 : total_income = first_investment * x + (total_investment - first_investment) * second_rate) :
  x = 0.085 := by
sorry

end NUMINAMATH_CALUDE_first_investment_interest_rate_l1908_190856


namespace NUMINAMATH_CALUDE_moon_speed_in_km_per_hour_l1908_190872

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.04

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Converts speed from kilometers per second to kilometers per hour -/
def km_per_sec_to_km_per_hour (speed_km_per_sec : ℝ) : ℝ :=
  speed_km_per_sec * (seconds_per_hour : ℝ)

theorem moon_speed_in_km_per_hour :
  km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3744 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_in_km_per_hour_l1908_190872


namespace NUMINAMATH_CALUDE_cube_cutting_surface_area_l1908_190843

/-- Calculates the total surface area of pieces after cutting a cube -/
def total_surface_area_after_cutting (edge_length : ℝ) (horizontal_cuts : ℕ) (vertical_cuts : ℕ) : ℝ :=
  let original_surface_area := 6 * edge_length^2
  let new_horizontal_faces := 2 * edge_length^2 * (2 * horizontal_cuts : ℝ)
  let new_vertical_faces := 2 * edge_length^2 * (2 * vertical_cuts : ℝ)
  original_surface_area + new_horizontal_faces + new_vertical_faces

/-- Theorem: The total surface area of pieces after cutting a 2-decimeter cube 4 times horizontally and 5 times vertically is 96 square decimeters -/
theorem cube_cutting_surface_area :
  total_surface_area_after_cutting 2 4 5 = 96 := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_surface_area_l1908_190843


namespace NUMINAMATH_CALUDE_minimum_students_l1908_190889

theorem minimum_students (b g : ℕ) : 
  b > 0 → g > 0 → 
  (b / 2 : ℚ) = 2 * (2 * g / 3 : ℚ) → 
  b + g ≥ 11 := by
sorry

end NUMINAMATH_CALUDE_minimum_students_l1908_190889


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1908_190874

def A : Set ℤ := {0, 2}
def B : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1908_190874


namespace NUMINAMATH_CALUDE_johns_pictures_l1908_190831

/-- The number of pictures John drew and colored -/
def num_pictures : ℕ := 10

/-- The time it takes John to draw one picture (in hours) -/
def drawing_time : ℝ := 2

/-- The time it takes John to color one picture (in hours) -/
def coloring_time : ℝ := drawing_time * 0.7

/-- The total time John spent on all pictures (in hours) -/
def total_time : ℝ := 34

theorem johns_pictures :
  (drawing_time + coloring_time) * num_pictures = total_time := by sorry

end NUMINAMATH_CALUDE_johns_pictures_l1908_190831


namespace NUMINAMATH_CALUDE_area_circle_outside_square_l1908_190875

/-- The area inside a circle of radius 1 but outside a square of side length 2, when both share the same center, is equal to π - 2. -/
theorem area_circle_outside_square :
  let circle_radius : ℝ := 1
  let square_side : ℝ := 2
  let circle_area : ℝ := π * circle_radius ^ 2
  let square_area : ℝ := square_side ^ 2
  let area_difference : ℝ := circle_area - square_area
  area_difference = π - 2 := by sorry

end NUMINAMATH_CALUDE_area_circle_outside_square_l1908_190875


namespace NUMINAMATH_CALUDE_pool_capacity_is_12000_l1908_190844

/-- Represents the capacity of a pool and its filling rates. -/
structure PoolSystem where
  capacity : ℝ
  bothValvesTime : ℝ
  firstValveTime : ℝ
  secondValveExtraRate : ℝ

/-- Theorem stating that under given conditions, the pool capacity is 12000 cubic meters. -/
theorem pool_capacity_is_12000 (p : PoolSystem)
  (h1 : p.bothValvesTime = 48)
  (h2 : p.firstValveTime = 120)
  (h3 : p.secondValveExtraRate = 50)
  (h4 : p.capacity / p.firstValveTime + (p.capacity / p.firstValveTime + p.secondValveExtraRate) = p.capacity / p.bothValvesTime) :
  p.capacity = 12000 := by
  sorry

#check pool_capacity_is_12000

end NUMINAMATH_CALUDE_pool_capacity_is_12000_l1908_190844


namespace NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1908_190813

/-- Given a hyperbola with equation (y-4)^2/32 - (x+3)^2/18 = 1,
    the distance between its vertices is 8√2. -/
theorem hyperbola_vertices_distance :
  let k : ℝ := 4
  let h : ℝ := -3
  let a_squared : ℝ := 32
  let b_squared : ℝ := 18
  let hyperbola_eq := fun (x y : ℝ) => (y - k)^2 / a_squared - (x - h)^2 / b_squared = 1
  let vertices_distance := 2 * Real.sqrt a_squared
  hyperbola_eq x y → vertices_distance = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertices_distance_l1908_190813


namespace NUMINAMATH_CALUDE_fourth_student_in_sample_l1908_190855

/-- Represents a systematic sample from a class of students. -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_student : ℕ

/-- Checks if a student number is part of the systematic sample. -/
def is_in_sample (s : SystematicSample) (student : ℕ) : Prop :=
  ∃ k : ℕ, student = s.first_student + k * s.interval

/-- The main theorem to be proved. -/
theorem fourth_student_in_sample
  (s : SystematicSample)
  (h_class_size : s.class_size = 48)
  (h_sample_size : s.sample_size = 4)
  (h_interval : s.interval = s.class_size / s.sample_size)
  (h_6_in_sample : is_in_sample s 6)
  (h_30_in_sample : is_in_sample s 30)
  (h_42_in_sample : is_in_sample s 42)
  : is_in_sample s 18 :=
sorry

end NUMINAMATH_CALUDE_fourth_student_in_sample_l1908_190855


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1908_190895

theorem inequality_system_solution (a : ℝ) :
  (∃ x : ℝ, (1 + x > a) ∧ (2 * x - 4 ≤ 0)) ↔ (a < 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1908_190895
