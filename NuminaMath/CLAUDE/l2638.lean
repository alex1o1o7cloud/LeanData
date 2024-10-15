import Mathlib

namespace NUMINAMATH_CALUDE_b_geq_0_sufficient_not_necessary_for_a_squared_plus_b_geq_0_l2638_263819

theorem b_geq_0_sufficient_not_necessary_for_a_squared_plus_b_geq_0 :
  ∃ (a b : ℝ), (b ≥ 0 → a^2 + b ≥ 0) ∧ ¬(a^2 + b ≥ 0 → b ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_b_geq_0_sufficient_not_necessary_for_a_squared_plus_b_geq_0_l2638_263819


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l2638_263812

/-- Proves that given a cycle sold at a loss of 18% with a selling price of Rs. 1558, the original price of the cycle is Rs. 1900. -/
theorem cycle_price_calculation (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1558)
  (h2 : loss_percentage = 18) : 
  ∃ (original_price : ℝ), 
    original_price = 1900 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l2638_263812


namespace NUMINAMATH_CALUDE_sqrt_5184_div_18_eq_4_l2638_263814

theorem sqrt_5184_div_18_eq_4 : Real.sqrt 5184 / 18 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_5184_div_18_eq_4_l2638_263814


namespace NUMINAMATH_CALUDE_daves_phone_files_l2638_263859

theorem daves_phone_files :
  ∀ (initial_apps initial_files current_apps : ℕ),
    initial_apps = 24 →
    initial_files = 9 →
    current_apps = 12 →
    current_apps = (current_apps - 7) + 7 →
    current_apps - 7 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_daves_phone_files_l2638_263859


namespace NUMINAMATH_CALUDE_sum_six_smallest_multiples_of_12_l2638_263868

theorem sum_six_smallest_multiples_of_12 : 
  (Finset.range 6).sum (fun i => 12 * (i + 1)) = 252 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_smallest_multiples_of_12_l2638_263868


namespace NUMINAMATH_CALUDE_polygon_exterior_angles_l2638_263821

theorem polygon_exterior_angles (n : ℕ) (exterior_angle : ℝ) : 
  (n : ℝ) * exterior_angle = 360 → exterior_angle = 45 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_exterior_angles_l2638_263821


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2638_263864

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2638_263864


namespace NUMINAMATH_CALUDE_rational_coefficient_terms_count_l2638_263890

theorem rational_coefficient_terms_count :
  let expression := (x : ℝ) * (5 ^ (1/4 : ℝ)) + (y : ℝ) * (7 ^ (1/2 : ℝ))
  let power := 500
  let is_rational_coeff (k : ℕ) := (k % 4 = 0) ∧ ((power - k) % 2 = 0)
  (Finset.filter is_rational_coeff (Finset.range (power + 1))).card = 126 :=
by sorry

end NUMINAMATH_CALUDE_rational_coefficient_terms_count_l2638_263890


namespace NUMINAMATH_CALUDE_solve_linear_equation_l2638_263836

theorem solve_linear_equation (x : ℝ) (h : 4 * x + 12 = 48) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l2638_263836


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2638_263854

/-- An arithmetic sequence is a sequence where the difference between
    successive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_2 + a_4 = 16, prove a_3 = 8 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 2 + a 4 = 16) : 
  a 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2638_263854


namespace NUMINAMATH_CALUDE_parallelogram_area_l2638_263845

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 18) (h2 : b = 10) (h3 : θ = 150 * π / 180) :
  a * b * Real.sin (π - θ) = 90 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2638_263845


namespace NUMINAMATH_CALUDE_wedding_couples_theorem_l2638_263872

/-- The number of couples invited by the bride and groom to their wedding reception --/
def couples_invited (total_guests : ℕ) (friends : ℕ) : ℕ :=
  (total_guests - friends) / 2

theorem wedding_couples_theorem (total_guests : ℕ) (friends : ℕ) 
  (h1 : total_guests = 180) 
  (h2 : friends = 100) :
  couples_invited total_guests friends = 40 := by
  sorry

end NUMINAMATH_CALUDE_wedding_couples_theorem_l2638_263872


namespace NUMINAMATH_CALUDE_max_value_of_f_l2638_263817

open Real

noncomputable def f (x : ℝ) := Real.log (3 * x) - 3 * x

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (Real.exp 1) ∧
  (∀ x, x ∈ Set.Ioo 0 (Real.exp 1) → f x ≤ f c) ∧
  f c = -Real.log 3 - 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2638_263817


namespace NUMINAMATH_CALUDE_complement_of_union_is_multiples_of_three_l2638_263865

-- Define the set of integers
variable (U : Set Int)

-- Define sets A and B
def A : Set Int := {x | ∃ k : Int, x = 3 * k + 1}
def B : Set Int := {x | ∃ k : Int, x = 3 * k + 2}

-- State the theorem
theorem complement_of_union_is_multiples_of_three (hU : U = Set.univ) :
  (U \ (A ∪ B)) = {x : Int | ∃ k : Int, x = 3 * k} :=
sorry

end NUMINAMATH_CALUDE_complement_of_union_is_multiples_of_three_l2638_263865


namespace NUMINAMATH_CALUDE_total_potatoes_l2638_263849

theorem total_potatoes (nancy_potatoes sandy_potatoes andy_potatoes : ℕ)
  (h1 : nancy_potatoes = 6)
  (h2 : sandy_potatoes = 7)
  (h3 : andy_potatoes = 9) :
  nancy_potatoes + sandy_potatoes + andy_potatoes = 22 :=
by sorry

end NUMINAMATH_CALUDE_total_potatoes_l2638_263849


namespace NUMINAMATH_CALUDE_sum_of_distances_eq_three_halves_side_length_l2638_263881

/-- An equilateral triangle with a point inside it -/
structure EquilateralTriangleWithPoint where
  /-- The side length of the equilateral triangle -/
  a : ℝ
  /-- The point inside the triangle -/
  M : ℝ × ℝ
  /-- Assertion that the triangle is equilateral with side length a -/
  is_equilateral : a > 0
  /-- Assertion that M is inside the triangle -/
  M_inside : True  -- This is a simplification; in a real proof, we'd need to define this properly

/-- The sum of distances from a point to the sides of an equilateral triangle -/
def sum_of_distances (t : EquilateralTriangleWithPoint) : ℝ :=
  sorry  -- The actual calculation would go here

/-- Theorem: The sum of distances from any point inside an equilateral triangle
    to its sides is equal to 3/2 times the side length -/
theorem sum_of_distances_eq_three_halves_side_length (t : EquilateralTriangleWithPoint) :
  sum_of_distances t = 3/2 * t.a := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_eq_three_halves_side_length_l2638_263881


namespace NUMINAMATH_CALUDE_journey_time_l2638_263880

/-- Given a journey where:
  * The distance is 320 miles
  * The speed is 50 miles per hour
  * There is a 30-minute stopover
Prove that the total trip time is 6.9 hours -/
theorem journey_time (distance : ℝ) (speed : ℝ) (stopover : ℝ) :
  distance = 320 →
  speed = 50 →
  stopover = 0.5 →
  distance / speed + stopover = 6.9 :=
by sorry

end NUMINAMATH_CALUDE_journey_time_l2638_263880


namespace NUMINAMATH_CALUDE_prime_power_fraction_l2638_263884

theorem prime_power_fraction (u v : ℕ+) :
  (∃ (p : ℕ) (n : ℕ), Prime p ∧ (u.val * v.val^3 : ℚ) / (u.val^2 + v.val^2) = p^n) ↔
  (∃ (k : ℕ), k ≥ 1 ∧ u.val = 2^k ∧ v.val = 2^k) :=
by sorry

end NUMINAMATH_CALUDE_prime_power_fraction_l2638_263884


namespace NUMINAMATH_CALUDE_train_length_calculation_l2638_263810

/-- Calculates the length of two trains given their speeds and overtaking time -/
theorem train_length_calculation (v_fast v_slow : ℝ) (t : ℝ) (h1 : v_fast = 46) (h2 : v_slow = 36) (h3 : t = 72) :
  let v_rel := v_fast - v_slow
  let d := v_rel * t * (1000 / 3600)
  let L := d / 2
  L = 100 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2638_263810


namespace NUMINAMATH_CALUDE_composition_result_l2638_263838

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x
def g (x : ℝ) : ℝ := x^2

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x / 2
noncomputable def g_inv (x : ℝ) : ℝ := Real.sqrt x

-- State the theorem
theorem composition_result :
  f (g_inv (f_inv (f_inv (g (f 8))))) = 16 := by
  sorry

end NUMINAMATH_CALUDE_composition_result_l2638_263838


namespace NUMINAMATH_CALUDE_factorial15_base16_zeros_l2638_263889

/-- The number of trailing zeros in n when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- 15 factorial -/
def factorial15 : ℕ :=
  sorry

theorem factorial15_base16_zeros :
  trailingZeros factorial15 16 = 2 := by
  sorry

end NUMINAMATH_CALUDE_factorial15_base16_zeros_l2638_263889


namespace NUMINAMATH_CALUDE_equal_distribution_of_drawings_l2638_263806

theorem equal_distribution_of_drawings (total_drawings : ℕ) (num_neighbors : ℕ) 
  (h1 : total_drawings = 54) (h2 : num_neighbors = 6) :
  total_drawings / num_neighbors = 9 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_drawings_l2638_263806


namespace NUMINAMATH_CALUDE_circle_area_is_one_l2638_263837

theorem circle_area_is_one (r : ℝ) (h : r > 0) :
  (4 / (2 * Real.pi * r) = 2 * r) → Real.pi * r^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_is_one_l2638_263837


namespace NUMINAMATH_CALUDE_area_of_triangle_BEF_l2638_263888

-- Define the rectangle ABCD
structure Rectangle :=
  (a : ℝ) (b : ℝ)
  (area_eq : a * b = 30)

-- Define points E and F
structure Points (rect : Rectangle) :=
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)
  (E_on_AB : E.2 = 0 ∧ 0 ≤ E.1 ∧ E.1 ≤ rect.a)
  (F_on_BC : F.1 = rect.a ∧ 0 ≤ F.2 ∧ F.2 ≤ rect.b)

-- Define the theorem
theorem area_of_triangle_BEF
  (rect : Rectangle)
  (pts : Points rect)
  (area_CGF : ℝ)
  (area_EGF : ℝ)
  (h1 : area_CGF = 2)
  (h2 : area_EGF = 3) :
  (1/2) * pts.E.1 * pts.F.2 = 35/8 :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_BEF_l2638_263888


namespace NUMINAMATH_CALUDE_divisible_by_30_implies_x_is_0_l2638_263870

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

def four_digit_number (x : ℕ) : ℕ := x * 1000 + 240 + x

theorem divisible_by_30_implies_x_is_0 (x : ℕ) (h : x < 10) :
  is_divisible_by (four_digit_number x) 30 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_30_implies_x_is_0_l2638_263870


namespace NUMINAMATH_CALUDE_jude_current_age_l2638_263822

/-- Heath's age today -/
def heath_age_today : ℕ := 16

/-- Heath's age in 5 years -/
def heath_age_future : ℕ := heath_age_today + 5

/-- Jude's age in 5 years -/
def jude_age_future : ℕ := heath_age_future / 3

/-- Jude's age today -/
def jude_age_today : ℕ := jude_age_future - 5

/-- Theorem stating Jude's age today -/
theorem jude_current_age : jude_age_today = 2 := by
  sorry

end NUMINAMATH_CALUDE_jude_current_age_l2638_263822


namespace NUMINAMATH_CALUDE_valid_arrays_count_l2638_263875

/-- A 3x3 array with entries of 1 or -1 -/
def ValidArray : Type := Matrix (Fin 3) (Fin 3) Int

/-- Predicate to check if an entry is valid (1 or -1) -/
def isValidEntry (x : Int) : Prop := x = 1 ∨ x = -1

/-- Predicate to check if all entries in the array are valid -/
def hasValidEntries (arr : ValidArray) : Prop :=
  ∀ i j, isValidEntry (arr i j)

/-- Predicate to check if the sum of a row is zero -/
def rowSumZero (arr : ValidArray) (i : Fin 3) : Prop :=
  (arr i 0) + (arr i 1) + (arr i 2) = 0

/-- Predicate to check if the sum of a column is zero -/
def colSumZero (arr : ValidArray) (j : Fin 3) : Prop :=
  (arr 0 j) + (arr 1 j) + (arr 2 j) = 0

/-- Predicate to check if an array satisfies all conditions -/
def isValidArray (arr : ValidArray) : Prop :=
  hasValidEntries arr ∧
  (∀ i, rowSumZero arr i) ∧
  (∀ j, colSumZero arr j)

/-- The main theorem: there are exactly 6 valid arrays -/
theorem valid_arrays_count :
  ∃! (s : Finset ValidArray), (∀ arr ∈ s, isValidArray arr) ∧ s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_valid_arrays_count_l2638_263875


namespace NUMINAMATH_CALUDE_joey_study_time_l2638_263896

/-- Calculates the total study time for Joey's SAT exam preparation --/
theorem joey_study_time (weekday_hours : ℕ) (weekday_nights : ℕ) (weekend_hours : ℕ) (weekend_days : ℕ) (weeks : ℕ) : 
  weekday_hours = 2 →
  weekday_nights = 5 →
  weekend_hours = 3 →
  weekend_days = 2 →
  weeks = 6 →
  (weekday_hours * weekday_nights + weekend_hours * weekend_days) * weeks = 96 := by
  sorry

#check joey_study_time

end NUMINAMATH_CALUDE_joey_study_time_l2638_263896


namespace NUMINAMATH_CALUDE_map_distance_to_actual_l2638_263892

/-- Given a map scale and a distance on the map, calculate the actual distance in kilometers. -/
theorem map_distance_to_actual (scale : ℚ) (map_distance : ℚ) :
  scale = 200000 →
  map_distance = 3.5 →
  (map_distance * scale) / 100000 = 7 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_to_actual_l2638_263892


namespace NUMINAMATH_CALUDE_find_k_l2638_263883

theorem find_k (a b c d k : ℝ) 
  (h1 : a * b * c * d = 2007)
  (h2 : a = Real.sqrt (55 + Real.sqrt (k + a)))
  (h3 : b = Real.sqrt (55 - Real.sqrt (k + b)))
  (h4 : c = Real.sqrt (55 + Real.sqrt (k - c)))
  (h5 : d = Real.sqrt (55 - Real.sqrt (k - d))) :
  k = 1018 := by sorry

end NUMINAMATH_CALUDE_find_k_l2638_263883


namespace NUMINAMATH_CALUDE_student_age_problem_l2638_263833

theorem student_age_problem (total_students : ℕ) (total_avg_age : ℝ)
  (group1_size group2_size group3_size : ℕ) (group1_avg group2_avg group3_avg : ℝ) :
  total_students = 24 →
  total_avg_age = 18 →
  group1_size = 6 →
  group2_size = 10 →
  group3_size = 7 →
  group1_avg = 16 →
  group2_avg = 20 →
  group3_avg = 17 →
  ∃ (last_student_age : ℝ),
    (group1_size * group1_avg + group2_size * group2_avg + group3_size * group3_avg + last_student_age) / total_students = total_avg_age ∧
    last_student_age = 15 :=
by sorry

end NUMINAMATH_CALUDE_student_age_problem_l2638_263833


namespace NUMINAMATH_CALUDE_sum_of_absolute_ratios_l2638_263885

theorem sum_of_absolute_ratios (x y z : ℚ) 
  (sum_zero : x + y + z = 0) 
  (product_nonzero : x * y * z ≠ 0) : 
  (|x| / (y + z) + |y| / (x + z) + |z| / (x + y) = 1) ∨
  (|x| / (y + z) + |y| / (x + z) + |z| / (x + y) = -1) :=
sorry

end NUMINAMATH_CALUDE_sum_of_absolute_ratios_l2638_263885


namespace NUMINAMATH_CALUDE_joe_video_game_spending_l2638_263811

/-- Joe's video game spending problem -/
theorem joe_video_game_spending
  (initial_money : ℕ)
  (selling_price : ℕ)
  (months : ℕ)
  (h1 : initial_money = 240)
  (h2 : selling_price = 30)
  (h3 : months = 12)
  : ∃ (monthly_spending : ℕ),
    monthly_spending = 50 ∧
    initial_money = months * monthly_spending - months * selling_price :=
by sorry

end NUMINAMATH_CALUDE_joe_video_game_spending_l2638_263811


namespace NUMINAMATH_CALUDE_same_color_probability_is_five_eighteenths_l2638_263873

/-- Represents the number of jelly beans of each color that Abe has -/
structure AbeJellyBeans where
  green : Nat
  blue : Nat

/-- Represents the number of jelly beans of each color that Bob has -/
structure BobJellyBeans where
  green : Nat
  blue : Nat
  red : Nat

/-- Calculates the probability of both Abe and Bob showing the same color jelly bean -/
def probability_same_color (abe : AbeJellyBeans) (bob : BobJellyBeans) : Rat :=
  sorry

/-- The main theorem stating the probability of Abe and Bob showing the same color jelly bean -/
theorem same_color_probability_is_five_eighteenths 
  (abe : AbeJellyBeans) 
  (bob : BobJellyBeans) 
  (h1 : abe.green = 2)
  (h2 : abe.blue = 1)
  (h3 : bob.green = 2)
  (h4 : bob.blue = 1)
  (h5 : bob.red = 3) :
  probability_same_color abe bob = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_is_five_eighteenths_l2638_263873


namespace NUMINAMATH_CALUDE_median_mode_difference_l2638_263827

def data : List ℕ := [42, 44, 44, 45, 45, 45, 51, 51, 51, 53, 53, 53, 62, 64, 66, 66, 67, 68, 70, 74, 74, 75, 75, 76, 81, 82, 85, 88, 89, 89]

def mode (l : List ℕ) : ℕ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem median_mode_difference : 
  |median data - (mode data : ℚ)| = 23 := by sorry

end NUMINAMATH_CALUDE_median_mode_difference_l2638_263827


namespace NUMINAMATH_CALUDE_permutation_remainder_l2638_263858

/-- The number of characters in the string -/
def string_length : ℕ := 16

/-- The number of A's in the string -/
def count_A : ℕ := 4

/-- The number of B's in the string -/
def count_B : ℕ := 5

/-- The number of C's in the string -/
def count_C : ℕ := 5

/-- The number of D's in the string -/
def count_D : ℕ := 2

/-- The length of the first segment -/
def first_segment : ℕ := 5

/-- The length of the second segment -/
def second_segment : ℕ := 5

/-- The length of the third segment -/
def third_segment : ℕ := 6

/-- The function to calculate the number of valid permutations -/
def count_permutations : ℕ := sorry

theorem permutation_remainder :
  count_permutations % 1000 = 540 := by sorry

end NUMINAMATH_CALUDE_permutation_remainder_l2638_263858


namespace NUMINAMATH_CALUDE_additional_week_cost_is_eleven_l2638_263820

/-- The cost per day for additional weeks in a student youth hostel -/
def additional_week_cost (first_week_daily_rate : ℚ) (total_days : ℕ) (total_cost : ℚ) : ℚ :=
  let first_week_cost := 7 * first_week_daily_rate
  let additional_days := total_days - 7
  let additional_cost := total_cost - first_week_cost
  additional_cost / additional_days

theorem additional_week_cost_is_eleven :
  additional_week_cost 18 23 302 = 11 := by
sorry

end NUMINAMATH_CALUDE_additional_week_cost_is_eleven_l2638_263820


namespace NUMINAMATH_CALUDE_marble_leftover_l2638_263848

theorem marble_leftover (n m k : ℤ) : (7*n + 2 + 7*m + 5 + 7*k + 4) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_marble_leftover_l2638_263848


namespace NUMINAMATH_CALUDE_cube_collinear_groups_l2638_263828

/-- Represents a point in a cube structure -/
inductive CubePoint
  | Vertex
  | EdgeMidpoint
  | FaceCenter
  | Center

/-- Represents a group of three collinear points in the cube -/
structure CollinearGroup :=
  (points : Fin 3 → CubePoint)

/-- The cube structure with its points -/
structure Cube :=
  (vertices : Fin 8 → CubePoint)
  (edgeMidpoints : Fin 12 → CubePoint)
  (faceCenters : Fin 6 → CubePoint)
  (center : CubePoint)

/-- Function to count collinear groups in the cube -/
def countCollinearGroups (c : Cube) : Nat :=
  sorry

theorem cube_collinear_groups :
  ∀ c : Cube, countCollinearGroups c = 49 :=
sorry

end NUMINAMATH_CALUDE_cube_collinear_groups_l2638_263828


namespace NUMINAMATH_CALUDE_original_number_proof_l2638_263818

theorem original_number_proof (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 3 * (1 / x)) :
  x = Real.sqrt 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2638_263818


namespace NUMINAMATH_CALUDE_cot_thirty_degrees_l2638_263844

theorem cot_thirty_degrees : Real.cos (π / 6) / Real.sin (π / 6) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cot_thirty_degrees_l2638_263844


namespace NUMINAMATH_CALUDE_base_to_lateral_area_ratio_l2638_263800

/-- A cone whose lateral surface unfolds into a semicircle -/
structure SemicircleCone where
  r : ℝ  -- radius of the base
  l : ℝ  -- slant height
  h : 2 * π * r = π * l  -- condition for unfolding into a semicircle

theorem base_to_lateral_area_ratio (cone : SemicircleCone) :
  (π * cone.r^2) / ((1/2) * π * cone.l^2) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_base_to_lateral_area_ratio_l2638_263800


namespace NUMINAMATH_CALUDE_negation_equivalence_l2638_263894

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2638_263894


namespace NUMINAMATH_CALUDE_yuan_equality_l2638_263860

theorem yuan_equality : (3.00 : ℝ) = (3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_yuan_equality_l2638_263860


namespace NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l2638_263876

/-- The number of distinct arrangements of n distinct beads on a necklace, 
    considering rotations and reflections as identical -/
def necklace_arrangements (n : ℕ) : ℕ := Nat.factorial n / (n * 2)

/-- Theorem stating that for 8 distinct beads, the number of distinct necklace arrangements is 2520 -/
theorem eight_bead_necklace_arrangements : 
  necklace_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_necklace_arrangements_l2638_263876


namespace NUMINAMATH_CALUDE_arithmetic_sequence_condition_l2638_263808

def is_arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

theorem arithmetic_sequence_condition (a : ℕ+ → ℝ) :
  (∀ n : ℕ+, a n = 2 * (n : ℝ) + 1) → is_arithmetic_sequence a ∧
  ∃ b : ℕ+ → ℝ, is_arithmetic_sequence b ∧ ∃ m : ℕ+, b m ≠ 2 * (m : ℝ) + 1 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_condition_l2638_263808


namespace NUMINAMATH_CALUDE_abs_five_minus_e_l2638_263861

theorem abs_five_minus_e (e : ℝ) (h : e < 5) : |5 - e| = 5 - e := by sorry

end NUMINAMATH_CALUDE_abs_five_minus_e_l2638_263861


namespace NUMINAMATH_CALUDE_men_entered_room_l2638_263852

/-- Proves that 2 men entered the room given the initial and final conditions --/
theorem men_entered_room : 
  ∀ (initial_men initial_women : ℕ),
  initial_men / initial_women = 4 / 5 →
  ∃ (men_entered : ℕ),
  2 * (initial_women - 3) = 24 ∧
  initial_men + men_entered = 14 →
  men_entered = 2 := by
sorry

end NUMINAMATH_CALUDE_men_entered_room_l2638_263852


namespace NUMINAMATH_CALUDE_digit_789_of_7_29_l2638_263831

def decimal_representation_7_29 : List ℕ :=
  [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7]

def repeating_period : ℕ := 28

theorem digit_789_of_7_29 : 
  (decimal_representation_7_29[(789 % repeating_period) - 1]) = 6 := by sorry

end NUMINAMATH_CALUDE_digit_789_of_7_29_l2638_263831


namespace NUMINAMATH_CALUDE_positive_number_equation_solution_l2638_263846

theorem positive_number_equation_solution :
  ∃ n : ℝ, n > 0 ∧ 3 * n^2 + n = 219 ∧ abs (n - 8.38) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_equation_solution_l2638_263846


namespace NUMINAMATH_CALUDE_certain_amount_calculation_l2638_263871

theorem certain_amount_calculation (x : ℝ) (A : ℝ) (h1 : x = 190) (h2 : 0.65 * x = 0.20 * A) : A = 617.5 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_calculation_l2638_263871


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l2638_263825

theorem decimal_to_fraction :
  (0.36 : ℚ) = 9 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l2638_263825


namespace NUMINAMATH_CALUDE_base_3_of_121_l2638_263816

def base_3_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_3_of_121 :
  base_3_representation 121 = [1, 1, 1, 1, 1] :=
sorry

end NUMINAMATH_CALUDE_base_3_of_121_l2638_263816


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l2638_263879

theorem smallest_solution_abs_equation :
  ∀ x : ℝ, |x - 3| = 8 → x ≥ -5 ∧ |-5 - 3| = 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l2638_263879


namespace NUMINAMATH_CALUDE_total_wheels_is_132_l2638_263843

/-- The number of bicycles in the storage area -/
def num_bicycles : ℕ := 24

/-- The number of tricycles in the storage area -/
def num_tricycles : ℕ := 14

/-- The number of unicycles in the storage area -/
def num_unicycles : ℕ := 10

/-- The number of quadbikes in the storage area -/
def num_quadbikes : ℕ := 8

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

/-- The number of wheels on a unicycle -/
def wheels_per_unicycle : ℕ := 1

/-- The number of wheels on a quadbike -/
def wheels_per_quadbike : ℕ := 4

/-- The total number of wheels in the storage area -/
def total_wheels : ℕ := 
  num_bicycles * wheels_per_bicycle +
  num_tricycles * wheels_per_tricycle +
  num_unicycles * wheels_per_unicycle +
  num_quadbikes * wheels_per_quadbike

theorem total_wheels_is_132 : total_wheels = 132 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_132_l2638_263843


namespace NUMINAMATH_CALUDE_boys_left_to_girl_l2638_263874

/-- Represents a group of children standing in a circle. -/
structure CircleGroup where
  boys : ℕ
  girls : ℕ
  boys_right_to_girl : ℕ

/-- The main theorem to be proved. -/
theorem boys_left_to_girl (group : CircleGroup) 
  (h1 : group.boys = 40)
  (h2 : group.girls = 28)
  (h3 : group.boys_right_to_girl = 18) :
  group.boys - (group.boys - group.boys_right_to_girl) = 18 := by
  sorry

#check boys_left_to_girl

end NUMINAMATH_CALUDE_boys_left_to_girl_l2638_263874


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l2638_263842

theorem largest_divisor_of_n_squared_div_72 (n : ℕ) (h : n > 0) (h_div : 72 ∣ n^2) :
  ∃ (t : ℕ), t > 0 ∧ t ∣ n ∧ ∀ (k : ℕ), k > 0 → k ∣ n → k ≤ t :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_72_l2638_263842


namespace NUMINAMATH_CALUDE_ages_sum_l2638_263809

theorem ages_sum (kiana_age twin_age : ℕ) : 
  kiana_age > twin_age →
  kiana_age * twin_age * twin_age = 72 →
  kiana_age + twin_age + twin_age = 14 :=
by sorry

end NUMINAMATH_CALUDE_ages_sum_l2638_263809


namespace NUMINAMATH_CALUDE_power_sum_implications_l2638_263835

theorem power_sum_implications (a b c : ℝ) : 
  (¬ ((a^2013 + b^2013 + c^2013 = 0) → (a^2014 + b^2014 + c^2014 = 0))) ∧ 
  ((a^2014 + b^2014 + c^2014 = 0) → (a^2015 + b^2015 + c^2015 = 0)) ∧ 
  (¬ ((a^2013 + b^2013 + c^2013 = 0 ∧ a^2015 + b^2015 + c^2015 = 0) → (a^2014 + b^2014 + c^2014 = 0))) :=
by sorry

end NUMINAMATH_CALUDE_power_sum_implications_l2638_263835


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l2638_263801

/-- Proves that given a principal of 6000, if increasing the interest rate by 2%
    results in 360 more interest over the same time period, then the time period is 3 years. -/
theorem simple_interest_time_period 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : principal = 6000)
  (h2 : principal * (rate + 2) / 100 * time = principal * rate / 100 * time + 360) :
  time = 3 :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l2638_263801


namespace NUMINAMATH_CALUDE_sum_of_digit_differences_eq_495_l2638_263813

/-- The sum of the differences between the first and last digits of all natural numbers from 1 to 999 -/
def sum_of_digit_differences : ℕ :=
  (List.range 999).foldl (λ sum n =>
    let first_digit := (n + 1) / 100
    let last_digit := (n + 1) % 10
    sum + (first_digit - last_digit)) 0

/-- Theorem stating that the sum of the differences between the first and last digits
    of all natural numbers from 1 to 999 is equal to 495 -/
theorem sum_of_digit_differences_eq_495 :
  sum_of_digit_differences = 495 := by sorry

end NUMINAMATH_CALUDE_sum_of_digit_differences_eq_495_l2638_263813


namespace NUMINAMATH_CALUDE_dvds_bought_online_l2638_263840

theorem dvds_bought_online (total : ℕ) (store : ℕ) (online : ℕ) : 
  total = 10 → store = 8 → online = total - store → online = 2 := by
  sorry

end NUMINAMATH_CALUDE_dvds_bought_online_l2638_263840


namespace NUMINAMATH_CALUDE_polynomial_degree_theorem_l2638_263855

theorem polynomial_degree_theorem : 
  let p : Polynomial ℝ := (X^3 + 1)^5 * (X^4 + 1)^2
  Polynomial.degree p = 23 := by sorry

end NUMINAMATH_CALUDE_polynomial_degree_theorem_l2638_263855


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l2638_263805

theorem sqrt_product_simplification (x : ℝ) :
  Real.sqrt (96 * x^2) * Real.sqrt (50 * x) * Real.sqrt (28 * x^3) = 1260 * x^3 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l2638_263805


namespace NUMINAMATH_CALUDE_original_number_l2638_263802

theorem original_number (x : ℝ) : 3 * (2 * x + 5) = 111 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l2638_263802


namespace NUMINAMATH_CALUDE_monkey_doll_price_difference_is_two_l2638_263893

def monkey_doll_price_difference (total_spending : ℕ) (large_doll_price : ℕ) (extra_small_dolls : ℕ) : ℕ :=
  let large_dolls := total_spending / large_doll_price
  let small_dolls := large_dolls + extra_small_dolls
  let small_doll_price := total_spending / small_dolls
  large_doll_price - small_doll_price

theorem monkey_doll_price_difference_is_two :
  monkey_doll_price_difference 300 6 25 = 2 := by sorry

end NUMINAMATH_CALUDE_monkey_doll_price_difference_is_two_l2638_263893


namespace NUMINAMATH_CALUDE_rectangle_midpoint_distances_l2638_263867

theorem rectangle_midpoint_distances (a b : ℝ) (ha : a = 3) (hb : b = 5) :
  let vertex := (0 : ℝ × ℝ)
  let midpoints := [
    (a / 2, 0),
    (a, b / 2),
    (a / 2, b),
    (0, b / 2)
  ]
  (midpoints.map (λ m => Real.sqrt ((m.1 - vertex.1)^2 + (m.2 - vertex.2)^2))).sum = 13.1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_midpoint_distances_l2638_263867


namespace NUMINAMATH_CALUDE_binary_operation_theorem_l2638_263826

def binary_to_decimal (b : List Bool) : Nat :=
  b.reverse.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) (acc : List Bool) : List Bool :=
    if m = 0 then acc
    else aux (m / 2) ((m % 2 = 1) :: acc)
  aux n []

def binary_add_subtract (a b c d : List Bool) : List Bool :=
  let sum := binary_to_decimal a + binary_to_decimal b - binary_to_decimal c + binary_to_decimal d
  decimal_to_binary sum

theorem binary_operation_theorem :
  binary_add_subtract [true, true, false, true] [true, true, true] [true, false, true, false] [true, false, false, true] =
  [true, false, false, true, true] := by sorry

end NUMINAMATH_CALUDE_binary_operation_theorem_l2638_263826


namespace NUMINAMATH_CALUDE_chemistry_books_count_l2638_263856

theorem chemistry_books_count (biology_books : ℕ) (total_ways : ℕ) : 
  biology_books = 14 →
  total_ways = 2548 →
  (∃ chemistry_books : ℕ, 
    total_ways = (biology_books.choose 2) * (chemistry_books.choose 2)) →
  ∃ chemistry_books : ℕ, chemistry_books = 8 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_books_count_l2638_263856


namespace NUMINAMATH_CALUDE_simplify_expression_l2638_263899

theorem simplify_expression (n : ℕ) :
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+4)) = 29 / 48 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2638_263899


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2638_263877

theorem container_volume_ratio (A B C : ℚ) 
  (h1 : (3 : ℚ) / 4 * A = (2 : ℚ) / 3 * B) 
  (h2 : (2 : ℚ) / 3 * B = (1 : ℚ) / 2 * C) : 
  A / C = (2 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2638_263877


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l2638_263807

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 3

-- State the theorem
theorem tangent_line_triangle_area (a : ℝ) : 
  (f' a 1 = -6) →  -- Condition for perpendicularity
  (∃ b c : ℝ, 
    (∀ x : ℝ, -6 * x + b = c * x + f a 1) ∧  -- Equation of tangent line
    (b = 6) ∧  -- y-intercept of tangent line
    (c = -6)) →  -- Slope of tangent line
  (1/2 * 1 * 6 = 3) :=  -- Area of triangle
by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l2638_263807


namespace NUMINAMATH_CALUDE_janes_stick_length_l2638_263886

/-- Given information about Pat's stick and its relationship to Sarah's and Jane's sticks,
    prove that Jane's stick is 22 inches long. -/
theorem janes_stick_length :
  -- Pat's stick length
  ∀ (pat_stick : ℕ),
  -- Covered portion of Pat's stick
  ∀ (covered_portion : ℕ),
  -- Conversion factor from feet to inches
  ∀ (feet_to_inches : ℕ),
  -- Conditions
  pat_stick = 30 →
  covered_portion = 7 →
  feet_to_inches = 12 →
  -- Sarah's stick is twice as long as the uncovered portion of Pat's stick
  ∃ (sarah_stick : ℕ), sarah_stick = 2 * (pat_stick - covered_portion) →
  -- Jane's stick is two feet shorter than Sarah's stick
  ∃ (jane_stick : ℕ), jane_stick = sarah_stick - 2 * feet_to_inches →
  -- Conclusion: Jane's stick is 22 inches long
  jane_stick = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_janes_stick_length_l2638_263886


namespace NUMINAMATH_CALUDE_orange_seller_loss_percentage_l2638_263803

theorem orange_seller_loss_percentage :
  ∀ (cost_price selling_price_10 selling_price_6 : ℚ),
    cost_price > 0 →
    selling_price_10 = 1 / 10 →
    selling_price_6 = 1 / 6 →
    selling_price_6 = 3/2 * cost_price →
    (cost_price - selling_price_10) / cost_price * 100 = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_orange_seller_loss_percentage_l2638_263803


namespace NUMINAMATH_CALUDE_daves_total_expense_l2638_263815

/-- The amount Dave spent on books -/
def daves_book_expense (animal_books outer_space_books train_books book_price : ℕ) : ℕ :=
  (animal_books + outer_space_books + train_books) * book_price

/-- Theorem stating the total amount Dave spent on books -/
theorem daves_total_expense : 
  daves_book_expense 8 6 3 6 = 102 := by
  sorry

end NUMINAMATH_CALUDE_daves_total_expense_l2638_263815


namespace NUMINAMATH_CALUDE_last_three_digits_of_3_to_8000_l2638_263804

theorem last_three_digits_of_3_to_8000 (h : 3^400 ≡ 1 [ZMOD 800]) :
  3^8000 ≡ 1 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_last_three_digits_of_3_to_8000_l2638_263804


namespace NUMINAMATH_CALUDE_cap_production_l2638_263847

theorem cap_production (first_week second_week third_week total_target : ℕ) 
  (h1 : first_week = 320)
  (h2 : second_week = 400)
  (h3 : third_week = 300)
  (h4 : total_target = 1360) :
  total_target - (first_week + second_week + third_week) = 340 :=
by sorry

end NUMINAMATH_CALUDE_cap_production_l2638_263847


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l2638_263832

/- Define the imaginary unit -/
variable (i : ℂ)

/- Define real numbers m and n -/
variable (m n : ℝ)

/- State the theorem -/
theorem complex_fraction_equals_i
  (h1 : i * i = -1)
  (h2 : m * (1 + i) = 11 + n * i) :
  (m + n * i) / (m - n * i) = i :=
sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l2638_263832


namespace NUMINAMATH_CALUDE_star_sum_five_l2638_263853

def star (a b : ℕ) : ℕ := a^b - a*b

theorem star_sum_five :
  ∀ a b : ℕ,
  a ≥ 2 →
  b ≥ 2 →
  star a b = 2 →
  a + b = 5 :=
by sorry

end NUMINAMATH_CALUDE_star_sum_five_l2638_263853


namespace NUMINAMATH_CALUDE_certain_number_is_fifteen_l2638_263834

/-- The number of Doberman puppies -/
def doberman_puppies : ℕ := 20

/-- The number of Schnauzers -/
def schnauzers : ℕ := 55

/-- The certain number calculated from the given expression -/
def certain_number : ℤ := 3 * doberman_puppies - 5 + (doberman_puppies - schnauzers)

/-- Theorem stating that the certain number equals 15 -/
theorem certain_number_is_fifteen : certain_number = 15 := by sorry

end NUMINAMATH_CALUDE_certain_number_is_fifteen_l2638_263834


namespace NUMINAMATH_CALUDE_parallel_line_plane_condition_l2638_263891

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (subset_of : Line → Plane → Prop)

-- State the theorem
theorem parallel_line_plane_condition
  (m n : Line) (α : Plane)
  (h1 : subset_of n α)
  (h2 : ¬ subset_of m α) :
  (∀ m n, parallel_lines m n → parallel_line_plane m α) ∧
  ¬(∀ m α, parallel_line_plane m α → parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_condition_l2638_263891


namespace NUMINAMATH_CALUDE_complex_modulus_product_l2638_263895

theorem complex_modulus_product : 
  Complex.abs ((10 - 5 * Complex.I) * (7 + 24 * Complex.I)) = 125 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l2638_263895


namespace NUMINAMATH_CALUDE_worker_efficiency_l2638_263878

/-- 
Proves that if worker A is thrice as efficient as worker B, 
and A takes 10 days less than B to complete a job, 
then B alone takes 15 days to complete the job.
-/
theorem worker_efficiency (days_b : ℕ) : 
  (days_b / 3 = days_b - 10) → days_b = 15 := by
  sorry

end NUMINAMATH_CALUDE_worker_efficiency_l2638_263878


namespace NUMINAMATH_CALUDE_drink_ticket_cost_l2638_263850

/-- Proves that the cost of each drink ticket is $7 given Jenna's income and spending constraints -/
theorem drink_ticket_cost 
  (concert_ticket_cost : ℕ)
  (hourly_wage : ℕ)
  (weekly_hours : ℕ)
  (spending_percentage : ℚ)
  (num_drink_tickets : ℕ)
  (h1 : concert_ticket_cost = 181)
  (h2 : hourly_wage = 18)
  (h3 : weekly_hours = 30)
  (h4 : spending_percentage = 1/10)
  (h5 : num_drink_tickets = 5) :
  (((hourly_wage * weekly_hours * 4) * spending_percentage - concert_ticket_cost : ℚ) / num_drink_tickets : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_drink_ticket_cost_l2638_263850


namespace NUMINAMATH_CALUDE_unique_f_exists_and_power_of_two_property_l2638_263824

def is_valid_f (f : ℕ+ → ℕ+) : Prop :=
  f 1 = 1 ∧ f 2 = 1 ∧ ∀ n ≥ 3, f n = f (f (n-1)) + f (n - f (n-1))

theorem unique_f_exists_and_power_of_two_property :
  ∃! f : ℕ+ → ℕ+, is_valid_f f ∧ ∀ m : ℕ, m ≥ 1 → f (2^m) = 2^(m-1) :=
sorry

end NUMINAMATH_CALUDE_unique_f_exists_and_power_of_two_property_l2638_263824


namespace NUMINAMATH_CALUDE_election_majority_proof_l2638_263863

theorem election_majority_proof :
  ∀ (total_votes : ℕ) (winning_percentage : ℚ),
    total_votes = 470 →
    winning_percentage = 70 / 100 →
    ∃ (winning_votes losing_votes : ℕ),
      winning_votes = (winning_percentage * total_votes).floor ∧
      losing_votes = total_votes - winning_votes ∧
      winning_votes - losing_votes = 188 :=
by
  sorry

end NUMINAMATH_CALUDE_election_majority_proof_l2638_263863


namespace NUMINAMATH_CALUDE_car_distance_theorem_l2638_263898

/-- Calculates the total distance traveled by a car with increasing speed over a given number of hours -/
def totalDistance (initialDistance : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  hours * (2 * initialDistance + (hours - 1) * speedIncrease) / 2

theorem car_distance_theorem :
  totalDistance 55 2 12 = 792 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_theorem_l2638_263898


namespace NUMINAMATH_CALUDE_contractor_work_problem_l2638_263887

theorem contractor_work_problem (M : ℕ) : 
  (M * 6 = (M - 5) * 10) → M = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_work_problem_l2638_263887


namespace NUMINAMATH_CALUDE_total_chinese_hours_l2638_263857

/-- The number of hours Ryan spends learning Chinese daily -/
def chinese_hours_per_day : ℕ := 4

/-- The number of days Ryan learns -/
def learning_days : ℕ := 6

/-- Theorem: The total hours Ryan spends learning Chinese over 6 days is 24 hours -/
theorem total_chinese_hours : chinese_hours_per_day * learning_days = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_chinese_hours_l2638_263857


namespace NUMINAMATH_CALUDE_f_properties_l2638_263869

-- Define the function f
def f (x : ℝ) : ℝ := x * (2 * x^2 - 3 * x - 12) + 5

-- Define the interval
def interval : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }

-- State the theorem
theorem f_properties :
  -- 1. Tangent line at x = 1
  (∃ (m c : ℝ), ∀ x y, y = f x → (x - 1) * (f 1 - y) = m * (x - 1)^2 + c * (x - 1) 
                     ∧ m = -12 ∧ c = 0) ∧
  -- 2. Maximum value
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 5) ∧
  -- 3. Minimum value
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = -15) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2638_263869


namespace NUMINAMATH_CALUDE_problem_statement_l2638_263823

theorem problem_statement : 
  (∀ x : ℝ, x^2 - x + 1 > 0) ∨ ¬(∃ x : ℝ, x > 0 ∧ Real.sin x > 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2638_263823


namespace NUMINAMATH_CALUDE_point_transformation_quadrant_l2638_263830

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- A point in the third quadrant has a negative x-coordinate and a negative y-coordinate. -/
def third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- If point P (a, b) is in the second quadrant, then point P₁ (-b, a-1) is in the third quadrant. -/
theorem point_transformation_quadrant (a b : ℝ) :
  second_quadrant (a, b) → third_quadrant (-b, a - 1) := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_quadrant_l2638_263830


namespace NUMINAMATH_CALUDE_equation_solution_l2638_263866

theorem equation_solution :
  ∃ x : ℝ, x ≠ 4 ∧ (x - 3) / (4 - x) - 1 = 1 / (x - 4) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2638_263866


namespace NUMINAMATH_CALUDE_y_coordinate_is_three_l2638_263882

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant of the Cartesian coordinate system -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Theorem: If a point is in the second quadrant and its distance to the x-axis is 3, then its y-coordinate is 3 -/
theorem y_coordinate_is_three (P : Point) 
  (h1 : second_quadrant P) 
  (h2 : distance_to_x_axis P = 3) : 
  P.y = 3 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_is_three_l2638_263882


namespace NUMINAMATH_CALUDE_b_equals_one_b_non_negative_l2638_263851

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Theorem 1
theorem b_equals_one (b c : ℝ) :
  c = -3 →
  quadratic 2 b c (-1) = -2 →
  b = 1 := by sorry

-- Theorem 2
theorem b_non_negative (b c p : ℝ) :
  b + c = -2 →
  b > c →
  quadratic 2 b c p = -2 →
  b ≥ 0 := by sorry

end NUMINAMATH_CALUDE_b_equals_one_b_non_negative_l2638_263851


namespace NUMINAMATH_CALUDE_final_value_is_correct_l2638_263897

/-- Calculates the final value of sold games in USD given initial conditions and exchange rates -/
def final_value_usd (initial_value : ℝ) (usd_to_eur : ℝ) (eur_to_jpy : ℝ) (jpy_to_usd : ℝ) : ℝ :=
  let tripled_value := initial_value * 3
  let eur_value := tripled_value * usd_to_eur
  let jpy_value := eur_value * eur_to_jpy
  let sold_portion := 0.4
  let sold_value_jpy := jpy_value * sold_portion
  sold_value_jpy * jpy_to_usd

/-- Theorem stating that the final value of sold games is $225.42 given the initial conditions -/
theorem final_value_is_correct :
  final_value_usd 200 0.85 130 0.0085 = 225.42 := by
  sorry

#eval final_value_usd 200 0.85 130 0.0085

end NUMINAMATH_CALUDE_final_value_is_correct_l2638_263897


namespace NUMINAMATH_CALUDE_problem_statement_l2638_263839

theorem problem_statement (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  (a - c > 2 * b) ∧ (a^2 > b^2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2638_263839


namespace NUMINAMATH_CALUDE_rabbit_toy_cost_l2638_263841

theorem rabbit_toy_cost (total_cost pet_food_cost cage_cost found_money : ℚ)
  (h1 : total_cost = 24.81)
  (h2 : pet_food_cost = 5.79)
  (h3 : cage_cost = 12.51)
  (h4 : found_money = 1.00) :
  total_cost - (pet_food_cost + cage_cost) + found_money = 7.51 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_toy_cost_l2638_263841


namespace NUMINAMATH_CALUDE_paul_picked_72_cans_l2638_263862

/-- The total number of cans Paul picked up -/
def total_cans (saturday_bags : ℕ) (sunday_bags : ℕ) (cans_per_bag : ℕ) : ℕ :=
  (saturday_bags + sunday_bags) * cans_per_bag

/-- Theorem stating that Paul picked up 72 cans in total -/
theorem paul_picked_72_cans :
  total_cans 6 3 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_paul_picked_72_cans_l2638_263862


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l2638_263829

theorem concentric_circles_radii_difference
  (r R : ℝ) -- r and R are real numbers representing radii
  (h_positive : r > 0) -- r is positive
  (h_ratio : π * R^2 = 4 * π * r^2) -- area ratio is 1:4
  : R - r = r :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l2638_263829
