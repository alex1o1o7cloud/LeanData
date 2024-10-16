import Mathlib

namespace NUMINAMATH_CALUDE_M_remainder_l276_27683

/-- The number of positive integers less than or equal to 2010 whose base-2 representation has more 1's than 0's -/
def M : ℕ := 1162

/-- 2010 is less than 2^11 - 1 -/
axiom h1 : 2010 < 2^11 - 1

/-- The sum of binary numbers where the number of 1's is more than 0's up to 2^11 - 1 -/
def total_sum : ℕ := 2^11 - 1

/-- The number of binary numbers more than 2010 and ≤ 2047 -/
def excess : ℕ := 37

/-- The sum of center elements in Pascal's Triangle rows 0 to 5 -/
def center_sum : ℕ := 351

theorem M_remainder (h2 : M = (total_sum + center_sum) / 2 - excess) :
  M % 1000 = 162 := by sorry

end NUMINAMATH_CALUDE_M_remainder_l276_27683


namespace NUMINAMATH_CALUDE_power_sum_of_i_l276_27603

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^66 + i^103 = -1 - i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l276_27603


namespace NUMINAMATH_CALUDE_same_distance_different_speeds_l276_27625

/-- Proves that given Joann's average speed and time, Fran needs to ride at a specific speed to cover the same distance in her given time -/
theorem same_distance_different_speeds (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15) 
  (h2 : joann_time = 4)
  (h3 : fran_time = 5) :
  joann_speed * joann_time = (60 / fran_time) * fran_time :=
by sorry

end NUMINAMATH_CALUDE_same_distance_different_speeds_l276_27625


namespace NUMINAMATH_CALUDE_complex_equality_problem_l276_27610

theorem complex_equality_problem (x y : ℝ) 
  (h : (x + y : ℂ) + Complex.I = 3*x + (x - y)*Complex.I) : 
  x = -1 ∧ y = -2 := by
sorry

end NUMINAMATH_CALUDE_complex_equality_problem_l276_27610


namespace NUMINAMATH_CALUDE_cylinder_height_relationship_l276_27640

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 ∧ h₁ > 0 ∧ r₂ > 0 ∧ h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_relationship_l276_27640


namespace NUMINAMATH_CALUDE_f_monotone_iff_m_range_l276_27671

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - m) * x - m else Real.log x / Real.log m

-- State the theorem
theorem f_monotone_iff_m_range (m : ℝ) :
  (∀ x y : ℝ, x < y → f m x < f m y) ↔ (3/2 ≤ m ∧ m < 3) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_iff_m_range_l276_27671


namespace NUMINAMATH_CALUDE_area_equality_l276_27666

-- Define the points
variable (A B C D E F G H O : Point)

-- Define area functions
noncomputable def S_Quadrilateral (P Q R S : Point) : ℝ := sorry
noncomputable def S_Triangle (P Q R : Point) : ℝ := sorry
noncomputable def S_Shaded : ℝ := sorry

-- State the theorem
theorem area_equality 
  (h1 : S_Quadrilateral B H C G / S_Quadrilateral A G D H = 1 / 4)
  (h2 : S_Triangle A B G + S_Triangle D C G + S_Triangle D E H + S_Triangle A F H = 
        S_Triangle A O G + S_Triangle D O G + S_Triangle D O H + S_Triangle A O H)
  (h3 : S_Triangle A O G + S_Triangle D O G + S_Triangle D O H + S_Triangle A O H = S_Shaded)
  (h4 : S_Triangle E F H + S_Triangle B C G = S_Quadrilateral B H C G)
  (h5 : S_Quadrilateral B H C G = 1/4 * S_Shaded) :
  S_Quadrilateral A G D H = S_Shaded :=
by sorry

end NUMINAMATH_CALUDE_area_equality_l276_27666


namespace NUMINAMATH_CALUDE_ellipse_minimum_area_l276_27695

/-- An ellipse containing two specific circles has a minimum area -/
theorem ellipse_minimum_area (a b : ℝ) (h_positive_a : a > 0) (h_positive_b : b > 0) :
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → ((x - 2)^2 + y^2 = 4 ∨ (x + 2)^2 + y^2 = 4)) →
  a * b ≥ 8 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_minimum_area_l276_27695


namespace NUMINAMATH_CALUDE_sneakers_price_l276_27665

/-- Given a pair of sneakers with an unknown original price, if a $10 coupon is applied first,
    followed by a 10% membership discount, and the final price is $99,
    then the original price of the sneakers was $120. -/
theorem sneakers_price (original_price : ℝ) : 
  (original_price - 10) * 0.9 = 99 → original_price = 120 := by
  sorry

end NUMINAMATH_CALUDE_sneakers_price_l276_27665


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l276_27684

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : ∃ q : ℝ, geometric_sequence a q)
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 1/4) :
  ∃ q : ℝ, geometric_sequence a q ∧ q = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l276_27684


namespace NUMINAMATH_CALUDE_unique_triple_existence_l276_27638

theorem unique_triple_existence (p : ℕ) (hp : Prime p) 
  (h_prime : ∀ n : ℕ, 0 < n → n < p → Prime (n^2 - n + p)) :
  ∃! (a b c : ℤ), 
    b^2 - 4*a*c = 1 - 4*p ∧ 
    0 < a ∧ a ≤ c ∧ 
    -a ≤ b ∧ b < a ∧
    a = 1 ∧ b = -1 ∧ c = p := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_existence_l276_27638


namespace NUMINAMATH_CALUDE_neighborhood_total_l276_27623

/-- Represents the number of households in different categories -/
structure Neighborhood where
  neither : ℕ
  both : ℕ
  car : ℕ
  bikeOnly : ℕ

/-- Calculates the total number of households in the neighborhood -/
def totalHouseholds (n : Neighborhood) : ℕ :=
  n.neither + (n.car - n.both) + n.bikeOnly + n.both

/-- Theorem stating that the total number of households is 90 -/
theorem neighborhood_total (n : Neighborhood) 
  (h1 : n.neither = 11)
  (h2 : n.both = 16)
  (h3 : n.car = 44)
  (h4 : n.bikeOnly = 35) :
  totalHouseholds n = 90 := by
  sorry

#eval totalHouseholds { neither := 11, both := 16, car := 44, bikeOnly := 35 }

end NUMINAMATH_CALUDE_neighborhood_total_l276_27623


namespace NUMINAMATH_CALUDE_vegetables_used_l276_27631

def initial_beef : ℝ := 4
def unused_beef : ℝ := 1
def veg_to_beef_ratio : ℝ := 2

theorem vegetables_used : 
  let beef_used := initial_beef - unused_beef
  let vegetables_used := beef_used * veg_to_beef_ratio
  vegetables_used = 6 := by sorry

end NUMINAMATH_CALUDE_vegetables_used_l276_27631


namespace NUMINAMATH_CALUDE_smallest_square_factor_l276_27680

theorem smallest_square_factor (n : ℕ) (hn : n = 4410) :
  (∃ (y : ℕ), y > 0 ∧ ∃ (k : ℕ), n * y = k^2) ∧
  (∀ (z : ℕ), z > 0 → (∃ (k : ℕ), n * z = k^2) → z ≥ 10) := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_factor_l276_27680


namespace NUMINAMATH_CALUDE_necklace_beads_l276_27612

theorem necklace_beads (total : ℕ) (blue : ℕ) (h1 : total = 40) (h2 : blue = 5) : 
  let red := 2 * blue
  let white := blue + red
  let silver := total - (blue + red + white)
  silver = 10 := by
  sorry

end NUMINAMATH_CALUDE_necklace_beads_l276_27612


namespace NUMINAMATH_CALUDE_solution_difference_l276_27662

theorem solution_difference (x : ℝ) : 
  (∃ y : ℝ, (7 - y^2 / 4)^(1/3) = -3 ∧ y ≠ x ∧ (7 - x^2 / 4)^(1/3) = -3) → 
  |x - y| = 2 * Real.sqrt 136 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l276_27662


namespace NUMINAMATH_CALUDE_remi_water_spill_l276_27620

/-- Represents the amount of water Remi spilled the first time -/
def first_spill : ℕ := sorry

/-- The capacity of Remi's water bottle in ounces -/
def bottle_capacity : ℕ := 20

/-- The number of times Remi refills his bottle per day -/
def refills_per_day : ℕ := 3

/-- The number of days Remi drinks water -/
def days : ℕ := 7

/-- The amount of water Remi spilled the second time -/
def second_spill : ℕ := 8

/-- The total amount of water Remi actually drank in ounces -/
def total_drunk : ℕ := 407

theorem remi_water_spill :
  first_spill = 5 ∧
  bottle_capacity * refills_per_day * days - first_spill - second_spill = total_drunk :=
by sorry

end NUMINAMATH_CALUDE_remi_water_spill_l276_27620


namespace NUMINAMATH_CALUDE_furniture_assembly_time_l276_27642

/-- Given the number of chairs and tables, and the time spent on each piece,
    calculate the total time taken to assemble all furniture. -/
theorem furniture_assembly_time 
  (num_chairs : ℕ) 
  (num_tables : ℕ) 
  (time_per_piece : ℕ) 
  (h1 : num_chairs = 4) 
  (h2 : num_tables = 2) 
  (h3 : time_per_piece = 8) : 
  (num_chairs + num_tables) * time_per_piece = 48 := by
  sorry

end NUMINAMATH_CALUDE_furniture_assembly_time_l276_27642


namespace NUMINAMATH_CALUDE_crop_ratio_l276_27670

theorem crop_ratio (corn_rows : ℕ) (potato_rows : ℕ) (corn_per_row : ℕ) (potatoes_per_row : ℕ) (remaining_crops : ℕ) : 
  corn_rows = 10 →
  potato_rows = 5 →
  corn_per_row = 9 →
  potatoes_per_row = 30 →
  remaining_crops = 120 →
  (remaining_crops : ℚ) / ((corn_rows * corn_per_row + potato_rows * potatoes_per_row) : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_crop_ratio_l276_27670


namespace NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l276_27650

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 4, 6, 8}

-- Define set M
def M : Set ℕ := {0, 4, 6}

-- Define set N
def N : Set ℕ := {0, 1, 6}

-- Theorem statement
theorem intersection_of_M_and_complement_of_N :
  M ∩ (U \ N) = {4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l276_27650


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_2x_l276_27649

theorem integral_sqrt_minus_2x : 
  ∫ x in (0:ℝ)..1, (Real.sqrt (1 - (x - 1)^2) - 2*x) = π/4 - 1 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_2x_l276_27649


namespace NUMINAMATH_CALUDE_secret_santa_five_friends_l276_27692

/-- The number of derangements for n elements -/
def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

/-- The number of ways to distribute gifts in a Secret Santa game -/
def secretSantaDistributions (n : ℕ) : ℕ := derangement n

theorem secret_santa_five_friends :
  secretSantaDistributions 5 = 44 := by
  sorry

#eval secretSantaDistributions 5

end NUMINAMATH_CALUDE_secret_santa_five_friends_l276_27692


namespace NUMINAMATH_CALUDE_greatest_integer_x_cubed_le_27_l276_27633

theorem greatest_integer_x_cubed_le_27 :
  ∃ (x : ℕ), x > 0 ∧ (x^6 / x^3 : ℚ) ≤ 27 ∧ ∀ (y : ℕ), y > x → (y^6 / y^3 : ℚ) > 27 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_x_cubed_le_27_l276_27633


namespace NUMINAMATH_CALUDE_no_positive_roots_l276_27660

theorem no_positive_roots :
  ∀ x : ℝ, x^3 + 6*x^2 + 11*x + 6 = 0 → x ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_roots_l276_27660


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l276_27648

/-- The area of the shaded region in a square with quarter circles at each corner -/
theorem shaded_area_square_with_quarter_circles 
  (square_side : ℝ) 
  (circle_radius : ℝ) 
  (h1 : square_side = 15) 
  (h2 : circle_radius = 5) : 
  square_side ^ 2 - 4 * (π / 4 * circle_radius ^ 2) = 225 - 25 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l276_27648


namespace NUMINAMATH_CALUDE_boys_from_pine_l276_27624

theorem boys_from_pine (total_students : ℕ) (total_boys : ℕ) (total_girls : ℕ)
  (maple_students : ℕ) (pine_students : ℕ) (maple_girls : ℕ)
  (h1 : total_students = 150)
  (h2 : total_boys = 90)
  (h3 : total_girls = 60)
  (h4 : maple_students = 50)
  (h5 : pine_students = 100)
  (h6 : maple_girls = 30)
  (h7 : total_students = total_boys + total_girls)
  (h8 : total_students = maple_students + pine_students)
  (h9 : total_girls = maple_girls + (total_girls - maple_girls)) :
  pine_students - (total_girls - maple_girls) = 70 := by
  sorry

end NUMINAMATH_CALUDE_boys_from_pine_l276_27624


namespace NUMINAMATH_CALUDE_inequality_abc_l276_27690

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) ≤ 1 ∧
  ((a * b / (a^5 + b^5 + a * b)) + (b * c / (b^5 + c^5 + b * c)) + (c * a / (c^5 + a^5 + c * a)) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_abc_l276_27690


namespace NUMINAMATH_CALUDE_sons_age_l276_27622

/-- Given a man and his son, where the man is 18 years older than his son,
    and in two years the man's age will be twice the age of his son,
    prove that the present age of the son is 16 years. -/
theorem sons_age (man_age son_age : ℕ) : 
  man_age = son_age + 18 →
  man_age + 2 = 2 * (son_age + 2) →
  son_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l276_27622


namespace NUMINAMATH_CALUDE_hot_sauce_duration_l276_27634

-- Define the volume of a quart in ounces
def quart_volume : ℝ := 32

-- Define the size of the hot sauce container
def container_size : ℝ := quart_volume - 2

-- Define the size of one serving in ounces
def serving_size : ℝ := 0.5

-- Define the number of servings James uses per day
def servings_per_day : ℝ := 3

-- Define the amount of hot sauce James uses per day
def daily_usage : ℝ := serving_size * servings_per_day

-- Theorem: The hot sauce will last 20 days
theorem hot_sauce_duration :
  container_size / daily_usage = 20 := by sorry

end NUMINAMATH_CALUDE_hot_sauce_duration_l276_27634


namespace NUMINAMATH_CALUDE_club_members_remainder_l276_27669

theorem club_members_remainder (N : ℕ) : 
  50 < N → N < 80 → 
  N % 5 = 0 → (N % 8 = 0 ∨ N % 7 = 0) → 
  N % 9 = 6 ∨ N % 9 = 7 := by
sorry

end NUMINAMATH_CALUDE_club_members_remainder_l276_27669


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l276_27628

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l276_27628


namespace NUMINAMATH_CALUDE_complement_intersection_equals_set_l276_27688

def I : Finset Nat := {1,2,3,4,5,6,7,8}
def M : Finset Nat := {3,4,5}
def N : Finset Nat := {1,3,6}

theorem complement_intersection_equals_set : 
  (I \ M) ∩ (I \ N) = {2,7,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_set_l276_27688


namespace NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l276_27644

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 6) (h2 : x^2 + y^2 = 18) : x^3 + y^3 = 54 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_from_sum_and_square_sum_l276_27644


namespace NUMINAMATH_CALUDE_smallest_multiple_of_90_with_128_divisors_l276_27607

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ := sorry

-- Define the property of being a multiple of 90
def is_multiple_of_90 (n : ℕ) : Prop := ∃ k : ℕ, n = 90 * k

-- Define the main theorem
theorem smallest_multiple_of_90_with_128_divisors :
  ∃ n : ℕ, 
    (∀ m : ℕ, m < n → ¬(is_multiple_of_90 m ∧ num_divisors m = 128)) ∧
    is_multiple_of_90 n ∧
    num_divisors n = 128 ∧
    n / 90 = 1728 := by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_90_with_128_divisors_l276_27607


namespace NUMINAMATH_CALUDE_monotonicity_undetermined_l276_27689

-- Define the real numbers a, b, and c
variable (a b c : ℝ)

-- Assume a < b < c
variable (h1 : a < b) (h2 : b < c)

-- Define a function f from reals to reals
variable (f : ℝ → ℝ)

-- Define what it means for f to be increasing on an open interval
def IncreasingOn (f : ℝ → ℝ) (l r : ℝ) : Prop :=
  ∀ x y, l < x ∧ x < y ∧ y < r → f x < f y

-- State the theorem
theorem monotonicity_undetermined
  (h_ab : IncreasingOn f a b)
  (h_bc : IncreasingOn f b c) :
  ¬ (IncreasingOn f a c ∨ (∀ x y, a < x ∧ x < y ∧ y < c → f x > f y)) :=
sorry

end NUMINAMATH_CALUDE_monotonicity_undetermined_l276_27689


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l276_27613

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)  -- a is a sequence of real numbers indexed by natural numbers
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- a is an arithmetic sequence
  (h_sum1 : a 2 + a 6 = 8)  -- given condition
  (h_sum2 : a 3 + a 4 = 3)  -- given condition
  : ∃ d, ∀ n, a (n + 1) - a n = d ∧ d = 5 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l276_27613


namespace NUMINAMATH_CALUDE_function_value_at_negative_one_l276_27602

/-- Given a function f(x) = ax³ + b*sin(x) + 1 where f(1) = 5, prove that f(-1) = -3 -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * Real.sin x + 1) 
  (h2 : f 1 = 5) : 
  f (-1) = -3 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_one_l276_27602


namespace NUMINAMATH_CALUDE_problem_statement_l276_27694

theorem problem_statement : (2 * Real.sqrt 2 - 1)^2 + (1 + Real.sqrt 5) * (1 - Real.sqrt 5) = 5 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l276_27694


namespace NUMINAMATH_CALUDE_lena_calculation_l276_27667

def round_to_nearest_ten (n : ℤ) : ℤ :=
  10 * ((n + 5) / 10)

theorem lena_calculation : round_to_nearest_ten (63 + 2 * 29) = 120 := by
  sorry

end NUMINAMATH_CALUDE_lena_calculation_l276_27667


namespace NUMINAMATH_CALUDE_pareto_principle_implies_key_parts_l276_27675

/-- Represents the Pareto Principle applied to business management -/
structure ParetoPrinciple where
  core_business : ℝ
  total_business : ℝ
  core_result : ℝ
  total_result : ℝ
  efficiency_improvement : Bool
  core_business_ratio : core_business / total_business = 0.2
  result_ratio : core_result / total_result = 0.8
  focus_on_core : efficiency_improvement = true

/-- The conclusion drawn from the Pareto Principle -/
def emphasis_on_key_parts : Prop := True

/-- Theorem stating that the Pareto Principle implies emphasis on key parts -/
theorem pareto_principle_implies_key_parts (p : ParetoPrinciple) : 
  emphasis_on_key_parts :=
sorry

end NUMINAMATH_CALUDE_pareto_principle_implies_key_parts_l276_27675


namespace NUMINAMATH_CALUDE_log_xy_value_l276_27682

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 3) :
  Real.log (x * y) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_log_xy_value_l276_27682


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_PAQR_l276_27657

-- Define the points
variable (P A Q R B : ℝ × ℝ)

-- Define the distances
def AP : ℝ := 10
def PB : ℝ := 20
def PR : ℝ := 25

-- Define the right triangles
def is_right_triangle (X Y Z : ℝ × ℝ) : Prop :=
  (X.1 - Y.1) * (X.1 - Z.1) + (X.2 - Y.2) * (X.2 - Z.2) = 0

-- State the theorem
theorem area_of_quadrilateral_PAQR :
  is_right_triangle P A Q →
  is_right_triangle P B R →
  (let area_PAQ := (1/2) * ‖A - P‖ * ‖Q - P‖;
   let area_PBR := (1/2) * ‖B - P‖ * ‖R - B‖;
   area_PAQ + area_PBR = 174) :=
by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_PAQR_l276_27657


namespace NUMINAMATH_CALUDE_license_plate_count_l276_27658

def alphabet : ℕ := 26
def vowels : ℕ := 7  -- A, E, I, O, U, W, Y
def consonants : ℕ := alphabet - vowels
def even_digits : ℕ := 5  -- 0, 2, 4, 6, 8

def license_plate_combinations : ℕ := consonants * vowels * consonants * even_digits

theorem license_plate_count : license_plate_combinations = 12565 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l276_27658


namespace NUMINAMATH_CALUDE_train_length_problem_l276_27668

/-- The length of Train 2 given the following conditions:
    - Train 1 length is 290 meters
    - Train 1 speed is 120 km/h
    - Train 2 speed is 80 km/h
    - Trains are running in opposite directions
    - Time to cross each other is 9 seconds
-/
theorem train_length_problem (train1_length : ℝ) (train1_speed : ℝ) (train2_speed : ℝ) (crossing_time : ℝ) :
  train1_length = 290 →
  train1_speed = 120 →
  train2_speed = 80 →
  crossing_time = 9 →
  ∃ train2_length : ℝ,
    (train1_length + train2_length) / crossing_time = (train1_speed + train2_speed) * (1000 / 3600) ∧
    abs (train2_length - 209.95) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l276_27668


namespace NUMINAMATH_CALUDE_not_A_implies_not_all_right_l276_27663

-- Define the universe of discourse
variable (Student : Type)

-- Define predicates
variable (got_all_right : Student → Prop)
variable (received_A : Student → Prop)

-- State Ms. Carroll's promise
variable (carroll_promise : ∀ s : Student, got_all_right s → received_A s)

-- Theorem to prove
theorem not_A_implies_not_all_right :
  ∀ s : Student, ¬(received_A s) → ¬(got_all_right s) :=
sorry

end NUMINAMATH_CALUDE_not_A_implies_not_all_right_l276_27663


namespace NUMINAMATH_CALUDE_circle_area_ratio_l276_27617

theorem circle_area_ratio (R S : Real) (hR : R > 0) (hS : S > 0) 
  (h_diameter : R = 0.4 * S) : 
  (π * R^2) / (π * S^2) = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l276_27617


namespace NUMINAMATH_CALUDE_lcm_factor_problem_l276_27619

/-- Given two positive integers with specific properties, prove that the second factor of their LCM is 13 -/
theorem lcm_factor_problem (A B : ℕ+) (X : ℕ+) : 
  (Nat.gcd A B = 23) →
  (Nat.lcm A B = 23 * 12 * X) →
  (A = 299) →
  X = 13 := by
sorry

end NUMINAMATH_CALUDE_lcm_factor_problem_l276_27619


namespace NUMINAMATH_CALUDE_tan_x_equals_negative_seven_l276_27654

theorem tan_x_equals_negative_seven (x : ℝ) 
  (h1 : Real.sin (x + π/4) = 3/5)
  (h2 : Real.sin (x - π/4) = 4/5) : 
  Real.tan x = -7 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_equals_negative_seven_l276_27654


namespace NUMINAMATH_CALUDE_closest_to_95_l276_27641

def options : List ℝ := [90, 92, 95, 98, 100]

theorem closest_to_95 :
  let product := 2.1 * (45.5 - 0.25)
  ∀ x ∈ options, |product - 95| ≤ |product - x| :=
by
  sorry

end NUMINAMATH_CALUDE_closest_to_95_l276_27641


namespace NUMINAMATH_CALUDE_teacher_work_days_l276_27687

/-- Represents the number of days a teacher works in a month -/
def days_worked_per_month (periods_per_day : ℕ) (pay_per_period : ℕ) (months_worked : ℕ) (total_earnings : ℕ) : ℕ :=
  (total_earnings / months_worked) / (periods_per_day * pay_per_period)

/-- Theorem stating the number of days a teacher works in a month given specific conditions -/
theorem teacher_work_days :
  days_worked_per_month 5 5 6 3600 = 24 := by
  sorry

end NUMINAMATH_CALUDE_teacher_work_days_l276_27687


namespace NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l276_27611

theorem cube_minus_reciprocal_cube (x : ℝ) (h : x - 1/x = 4) : x^3 - 1/x^3 = 76 := by
  sorry

end NUMINAMATH_CALUDE_cube_minus_reciprocal_cube_l276_27611


namespace NUMINAMATH_CALUDE_first_equation_is_midpoint_second_equation_is_midpoint_iff_l276_27681

/-- Definition of a midpoint equation -/
def is_midpoint_equation (a b : ℚ) : Prop :=
  a ≠ 0 ∧ ((-b) / a = (a + b) / 2)

/-- First part of the problem -/
theorem first_equation_is_midpoint : is_midpoint_equation 4 (-8/3) := by
  sorry

/-- Second part of the problem -/
theorem second_equation_is_midpoint_iff (m : ℚ) : 
  is_midpoint_equation 5 (m - 1) ↔ m = -18/7 := by
  sorry

end NUMINAMATH_CALUDE_first_equation_is_midpoint_second_equation_is_midpoint_iff_l276_27681


namespace NUMINAMATH_CALUDE_simplify_square_roots_l276_27635

theorem simplify_square_roots : Real.sqrt (5 * 3) * Real.sqrt (3^3 * 5^3) = 225 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l276_27635


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_values_l276_27606

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℕ
  d : ℕ
  first_term : a 1 = 1
  nth_term : ∀ n : ℕ, n ≥ 3 → a n = 70
  common_diff : ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the possible values of n -/
theorem arithmetic_sequence_n_values (seq : ArithmeticSequence) :
  ∀ n : ℕ, n ≥ 3 ∧ seq.a n = 70 → n = 4 ∨ n = 24 ∨ n = 70 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_values_l276_27606


namespace NUMINAMATH_CALUDE_function_equality_l276_27655

theorem function_equality (x : ℝ) (h : x > 0) : 
  (Real.sqrt x)^2 / x = x / (Real.sqrt x)^2 ∧ 
  (Real.sqrt x)^2 / x = 1 ∧ 
  x / (Real.sqrt x)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_function_equality_l276_27655


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_l276_27674

theorem smallest_five_digit_divisible : ∃ n : ℕ, 
  (10000 ≤ n ∧ n < 100000) ∧ 
  (∀ m : ℕ, (10000 ≤ m ∧ m < 100000) ∧ 2 ∣ m ∧ 3 ∣ m ∧ 8 ∣ m ∧ 9 ∣ m → n ≤ m) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 8 ∣ n ∧ 9 ∣ n ∧
  n = 10008 := by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_l276_27674


namespace NUMINAMATH_CALUDE_relay_race_distance_l276_27636

/-- Represents the distance each team member runs in a relay race. -/
def distance_per_member (total_distance : ℕ) (team_size : ℕ) : ℚ :=
  total_distance / team_size

/-- Theorem stating that in a 150-meter relay race with 5 team members,
    each member runs 30 meters. -/
theorem relay_race_distance :
  distance_per_member 150 5 = 30 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_distance_l276_27636


namespace NUMINAMATH_CALUDE_fraction_positivity_l276_27693

theorem fraction_positivity (x : ℝ) : (x + 2) / ((x - 3)^3) > 0 ↔ x < -2 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_positivity_l276_27693


namespace NUMINAMATH_CALUDE_divisor_of_99_l276_27615

def reverse_digits (n : ℕ) : ℕ := sorry

theorem divisor_of_99 (k : ℕ) 
  (h : ∀ n : ℕ, k ∣ n → k ∣ reverse_digits n) : 
  k ∣ 99 := by sorry

end NUMINAMATH_CALUDE_divisor_of_99_l276_27615


namespace NUMINAMATH_CALUDE_inequality_system_solution_l276_27605

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x - m > 0 ∧ 2*x + 1 > 3) ↔ x > 1) → m ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l276_27605


namespace NUMINAMATH_CALUDE_crate_height_determination_l276_27679

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ
  height : ℝ

/-- Checks if a gas tank fits inside a crate -/
def tankFitsInCrate (tank : GasTank) (crate : CrateDimensions) : Prop :=
  2 * tank.radius ≤ min crate.length (min crate.width crate.height) ∧
  tank.height ≤ max crate.length (max crate.width crate.height)

theorem crate_height_determination
  (crate : CrateDimensions)
  (tank : GasTank)
  (h_crate_dims : crate.length = 6 ∧ crate.width = 8)
  (h_tank_radius : tank.radius = 4)
  (h_tank_fits : tankFitsInCrate tank crate)
  (h_max_volume : ∀ (other_tank : GasTank),
    tankFitsInCrate other_tank crate →
    tank.radius * tank.radius * tank.height ≥ other_tank.radius * other_tank.radius * other_tank.height) :
  crate.height = 6 :=
sorry

end NUMINAMATH_CALUDE_crate_height_determination_l276_27679


namespace NUMINAMATH_CALUDE_trapezoid_shorter_lateral_l276_27645

/-- Represents a trapezoid with specific properties -/
structure Trapezoid where
  longer_lateral : ℝ
  base_difference : ℝ
  right_angle_intersection : Bool

/-- 
  Theorem: In a trapezoid where the lines containing the lateral sides intersect at a right angle,
  if the longer lateral side is 8 and the difference between the bases is 10,
  then the shorter lateral side is 6.
-/
theorem trapezoid_shorter_lateral 
  (t : Trapezoid) 
  (h1 : t.longer_lateral = 8) 
  (h2 : t.base_difference = 10) 
  (h3 : t.right_angle_intersection = true) : 
  ∃ (shorter_lateral : ℝ), shorter_lateral = 6 := by
  sorry

#check trapezoid_shorter_lateral

end NUMINAMATH_CALUDE_trapezoid_shorter_lateral_l276_27645


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l276_27647

/-- An isosceles trapezoid with given base lengths and area -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- The length of the sides of an isosceles trapezoid -/
def side_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The side length of the given isosceles trapezoid is 5 -/
theorem isosceles_trapezoid_side_length :
  let t : IsoscelesTrapezoid := { base1 := 10, base2 := 16, area := 52 }
  side_length t = 5 := by sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_side_length_l276_27647


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l276_27659

-- Define a geometric sequence
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric a →
  (∀ n : ℕ+, a n > 0) →
  a 1 * a 3 + 2 * a 2 * a 5 + a 4 * a 6 = 36 →
  a 2 + a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l276_27659


namespace NUMINAMATH_CALUDE_death_rate_calculation_l276_27661

/-- The average birth rate in people per two seconds -/
def average_birth_rate : ℕ := 10

/-- The population net increase in one day -/
def population_net_increase : ℕ := 345600

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The average death rate in people per two seconds -/
def average_death_rate : ℕ := 2

theorem death_rate_calculation :
  average_birth_rate - average_death_rate = 
    2 * (population_net_increase / seconds_per_day) :=
by sorry

end NUMINAMATH_CALUDE_death_rate_calculation_l276_27661


namespace NUMINAMATH_CALUDE_jons_website_hours_l276_27686

theorem jons_website_hours (earnings_per_visit : ℚ) (visits_per_hour : ℕ) 
  (monthly_earnings : ℚ) (days_in_month : ℕ) 
  (h1 : earnings_per_visit = 1/10) 
  (h2 : visits_per_hour = 50) 
  (h3 : monthly_earnings = 3600) 
  (h4 : days_in_month = 30) : 
  (monthly_earnings / earnings_per_visit / visits_per_hour) / days_in_month = 24 := by
  sorry

end NUMINAMATH_CALUDE_jons_website_hours_l276_27686


namespace NUMINAMATH_CALUDE_triangle_area_from_circumradius_side_angle_l276_27616

/-- The area of a triangle given its circumradius, one side, and one angle. -/
theorem triangle_area_from_circumradius_side_angle 
  (R a β : ℝ) (h_R : R > 0) (h_a : a > 0) (h_β : 0 < β ∧ β < π) : 
  ∃ (t : ℝ), t = (a^2 * Real.sin (2*β) / 4) + (a * Real.sin β^2 / 2) * Real.sqrt (4*R^2 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_from_circumradius_side_angle_l276_27616


namespace NUMINAMATH_CALUDE_song_count_proof_l276_27685

def final_song_count (initial : ℕ) (deleted : ℕ) (added : ℕ) : ℕ :=
  initial - deleted + added

theorem song_count_proof (initial deleted added : ℕ) 
  (h1 : initial ≥ deleted) : 
  final_song_count initial deleted added = initial - deleted + added :=
by
  sorry

#eval final_song_count 34 14 44

end NUMINAMATH_CALUDE_song_count_proof_l276_27685


namespace NUMINAMATH_CALUDE_similar_radical_expressions_l276_27664

-- Define the concept of similar radical expressions
def are_similar_radical_expressions (x y : ℝ) : Prop :=
  ∃ (k : ℝ) (n : ℕ), k > 0 ∧ x = k * (y^(1/n))

theorem similar_radical_expressions :
  ∀ (a : ℝ), a > 0 →
  (are_similar_radical_expressions (a^(1/3) * (3^(1/3))) 3) ∧
  ¬(are_similar_radical_expressions a (3*a/2)) ∧
  ¬(are_similar_radical_expressions (2*a) (a^(1/2))) ∧
  ¬(are_similar_radical_expressions (2*a) ((3*a^2)^(1/2))) :=
by sorry

end NUMINAMATH_CALUDE_similar_radical_expressions_l276_27664


namespace NUMINAMATH_CALUDE_shopping_cost_calculation_l276_27637

/-- Calculates the total cost of a shopping trip, including discounts and sales tax -/
theorem shopping_cost_calculation 
  (tshirt_price sweater_price jacket_price : ℚ)
  (jacket_discount sales_tax : ℚ)
  (tshirt_quantity sweater_quantity jacket_quantity : ℕ)
  (h1 : tshirt_price = 8)
  (h2 : sweater_price = 18)
  (h3 : jacket_price = 80)
  (h4 : jacket_discount = 1/10)
  (h5 : sales_tax = 1/20)
  (h6 : tshirt_quantity = 6)
  (h7 : sweater_quantity = 4)
  (h8 : jacket_quantity = 5) :
  let subtotal := tshirt_price * tshirt_quantity + 
                  sweater_price * sweater_quantity + 
                  jacket_price * jacket_quantity * (1 - jacket_discount)
  let total_with_tax := subtotal * (1 + sales_tax)
  total_with_tax = 504 := by sorry


end NUMINAMATH_CALUDE_shopping_cost_calculation_l276_27637


namespace NUMINAMATH_CALUDE_loan_duration_l276_27677

/-- Given a loan split into two parts, prove the duration of the second part. -/
theorem loan_duration (total sum : ℕ) (second_part : ℕ) (first_rate second_rate : ℚ) (first_duration : ℕ) :
  total = 2691 →
  second_part = 1656 →
  first_rate = 3 / 100 →
  second_rate = 5 / 100 →
  first_duration = 8 →
  (total - second_part) * first_rate * first_duration = second_part * second_rate * 3 →
  3 = (total - second_part) * first_rate * first_duration / (second_part * second_rate) :=
by sorry

end NUMINAMATH_CALUDE_loan_duration_l276_27677


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l276_27627

/-- For an infinite geometric series with common ratio 1/4 and sum 80, the first term is 60. -/
theorem infinite_geometric_series_first_term : 
  ∀ (a : ℝ), 
  (∑' n, a * (1/4)^n) = 80 → 
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l276_27627


namespace NUMINAMATH_CALUDE_system_solution_l276_27626

theorem system_solution : 
  ∃! (x y : ℝ), x + y = 8 ∧ 2*x - y = 7 ∧ x = 5 ∧ y = 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l276_27626


namespace NUMINAMATH_CALUDE_cubic_kilometer_to_cubic_meters_l276_27651

/-- Given that one kilometer equals 1000 meters, prove that one cubic kilometer equals 1,000,000,000 cubic meters. -/
theorem cubic_kilometer_to_cubic_meters :
  (1 : ℝ) * (1000 : ℝ)^3 = (1000000000 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_cubic_kilometer_to_cubic_meters_l276_27651


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l276_27621

/-- A point in a 2D coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Predicate to check if a point is in the fourth quadrant -/
def isInFourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Theorem: If P(m, 1) is in the second quadrant, then B(-m+1, -1) is in the fourth quadrant -/
theorem point_in_fourth_quadrant (m : ℝ) :
  isInSecondQuadrant (Point.mk m 1) → isInFourthQuadrant (Point.mk (-m + 1) (-1)) :=
by sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l276_27621


namespace NUMINAMATH_CALUDE_river_distance_l276_27604

theorem river_distance (d : ℝ) : 
  (¬(d ≤ 12)) → (¬(d ≥ 15)) → (¬(d ≥ 10)) → (12 < d ∧ d < 15) := by
  sorry

end NUMINAMATH_CALUDE_river_distance_l276_27604


namespace NUMINAMATH_CALUDE_parallel_line_m_value_l276_27629

/-- Given a line passing through points A(-2, m) and B(m, 4) that is parallel to the line 2x + y - 1 = 0, prove that m = -8 -/
theorem parallel_line_m_value :
  ∀ m : ℝ,
  let A : ℝ × ℝ := (-2, m)
  let B : ℝ × ℝ := (m, 4)
  let slope_AB := (B.2 - A.2) / (B.1 - A.1)
  let slope_given := -2  -- Slope of 2x + y - 1 = 0
  slope_AB = slope_given → m = -8 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_m_value_l276_27629


namespace NUMINAMATH_CALUDE_midpoint_value_l276_27632

/-- Given two distinct points (m, n) and (p, q) on the curve x^2 - 5xy + 2y^2 + 7x - 6y + 3 = 0,
    where (m + 2, n + k) is the midpoint of the line segment connecting (m, n) and (p, q),
    and the line passing through (m, n) and (p, q) has the equation x - 5y + 1 = 0,
    prove that k = 2/5. -/
theorem midpoint_value (m n p q k : ℝ) : 
  (m ≠ p ∨ n ≠ q) →
  m^2 - 5*m*n + 2*n^2 + 7*m - 6*n + 3 = 0 →
  p^2 - 5*p*q + 2*q^2 + 7*p - 6*q + 3 = 0 →
  m + 2 = (m + p) / 2 →
  n + k = (n + q) / 2 →
  m - 5*n + 1 = 0 →
  p - 5*q + 1 = 0 →
  k = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_value_l276_27632


namespace NUMINAMATH_CALUDE_square_difference_pattern_l276_27698

theorem square_difference_pattern (n : ℕ+) : (n + 2)^2 - n^2 = 4 * (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_pattern_l276_27698


namespace NUMINAMATH_CALUDE_robin_total_pieces_l276_27673

/-- The number of pieces in a package of Type A gum -/
def type_a_gum_pieces : ℕ := 4

/-- The number of pieces in a package of Type B gum -/
def type_b_gum_pieces : ℕ := 8

/-- The number of pieces in a package of Type C gum -/
def type_c_gum_pieces : ℕ := 12

/-- The number of pieces in a package of Type X candy -/
def type_x_candy_pieces : ℕ := 6

/-- The number of pieces in a package of Type Y candy -/
def type_y_candy_pieces : ℕ := 10

/-- The number of packages of Type A gum Robin has -/
def robin_type_a_gum_packages : ℕ := 10

/-- The number of packages of Type B gum Robin has -/
def robin_type_b_gum_packages : ℕ := 5

/-- The number of packages of Type C gum Robin has -/
def robin_type_c_gum_packages : ℕ := 13

/-- The number of packages of Type X candy Robin has -/
def robin_type_x_candy_packages : ℕ := 8

/-- The number of packages of Type Y candy Robin has -/
def robin_type_y_candy_packages : ℕ := 6

/-- The total number of gum packages Robin has -/
def robin_total_gum_packages : ℕ := 28

/-- The total number of candy packages Robin has -/
def robin_total_candy_packages : ℕ := 14

theorem robin_total_pieces : 
  robin_type_a_gum_packages * type_a_gum_pieces +
  robin_type_b_gum_packages * type_b_gum_pieces +
  robin_type_c_gum_packages * type_c_gum_pieces +
  robin_type_x_candy_packages * type_x_candy_pieces +
  robin_type_y_candy_packages * type_y_candy_pieces = 344 :=
by sorry

end NUMINAMATH_CALUDE_robin_total_pieces_l276_27673


namespace NUMINAMATH_CALUDE_zane_picked_62_pounds_l276_27614

/-- The amount of garbage picked up by Daliah in pounds -/
def daliah_garbage : ℝ := 17.5

/-- The amount of garbage picked up by Dewei in pounds -/
def dewei_garbage : ℝ := daliah_garbage - 2

/-- The amount of garbage picked up by Zane in pounds -/
def zane_garbage : ℝ := 4 * dewei_garbage

/-- Theorem stating that Zane picked up 62 pounds of garbage -/
theorem zane_picked_62_pounds : zane_garbage = 62 := by sorry

end NUMINAMATH_CALUDE_zane_picked_62_pounds_l276_27614


namespace NUMINAMATH_CALUDE_angle_measure_when_sine_is_half_l276_27697

/-- If ∠A is an acute angle in a triangle and sin A = 1/2, then ∠A = 30°. -/
theorem angle_measure_when_sine_is_half (A : Real) (h_acute : 0 < A ∧ A < π / 2) 
  (h_sin : Real.sin A = 1 / 2) : A = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_when_sine_is_half_l276_27697


namespace NUMINAMATH_CALUDE_function_extrema_l276_27672

/-- Given constants a and b, if the function f(x) = ax^3 + b*ln(x + sqrt(1+x^2)) + 3
    has a maximum value of 10 on the interval (-∞, 0),
    then the minimum value of f(x) on the interval (0, +∞) is -4. -/
theorem function_extrema (a b : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^3 + b * Real.log (x + Real.sqrt (1 + x^2)) + 3
  (∃ (M : ℝ), M = 10 ∧ ∀ (x : ℝ), x < 0 → f x ≤ M) →
  ∃ (m : ℝ), m = -4 ∧ ∀ (x : ℝ), x > 0 → f x ≥ m :=
by sorry


end NUMINAMATH_CALUDE_function_extrema_l276_27672


namespace NUMINAMATH_CALUDE_remaining_candies_formula_l276_27678

/-- Represents the remaining number of candies after the first night -/
def remaining_candies (K S m n : ℕ) : ℚ :=
  (K + S : ℚ) * (1 - m / n)

/-- Theorem stating that the remaining number of candies after the first night
    is equal to (K + S) * (1 - m/n) -/
theorem remaining_candies_formula (K S m n : ℕ) (h : n ≠ 0) :
  remaining_candies K S m n = (K + S : ℚ) * (1 - m / n) :=
by sorry

end NUMINAMATH_CALUDE_remaining_candies_formula_l276_27678


namespace NUMINAMATH_CALUDE_ellipse_existence_and_uniqueness_l276_27618

/-- A structure representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A structure representing a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A structure representing an ellipse in a 2D plane -/
structure Ellipse where
  center : Point
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ
  rotation : ℝ

/-- Function to check if two lines are perpendicular -/
def arePerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Function to check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  sorry

/-- Function to check if an ellipse has its axes on given lines -/
def ellipseAxesOnLines (e : Ellipse) (l1 l2 : Line) : Prop :=
  sorry

/-- Theorem stating the existence and uniqueness of ellipses -/
theorem ellipse_existence_and_uniqueness 
  (l1 l2 : Line) (p1 p2 : Point) 
  (h_perp : arePerpendicular l1 l2) :
  (p1 ≠ p2 → ∃! e : Ellipse, pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ ellipseAxesOnLines e l1 l2) ∧
  (p1 = p2 → ∃ e : Ellipse, pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ ellipseAxesOnLines e l1 l2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_existence_and_uniqueness_l276_27618


namespace NUMINAMATH_CALUDE_opposite_values_l276_27691

theorem opposite_values (x y : ℝ) : 
  |x + y - 9| + (2*x - y + 3)^2 = 0 → x = 2 ∧ y = 7 := by
  sorry

end NUMINAMATH_CALUDE_opposite_values_l276_27691


namespace NUMINAMATH_CALUDE_eating_contest_l276_27630

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight pizza_weight sandwich_weight : ℕ)
  (jacob_pies noah_burgers jacob_pizzas jacob_sandwiches mason_hotdogs mason_sandwiches : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  pizza_weight = 15 →
  sandwich_weight = 3 →
  jacob_pies = noah_burgers - 3 →
  jacob_pizzas = jacob_sandwiches / 2 →
  mason_hotdogs = 3 * jacob_pies →
  mason_hotdogs = (3 : ℚ) / 2 * mason_sandwiches →
  noah_burgers = 8 →
  mason_hotdogs * hot_dog_weight = 30 := by sorry

end NUMINAMATH_CALUDE_eating_contest_l276_27630


namespace NUMINAMATH_CALUDE_chemical_mixture_problem_l276_27639

/-- Proves that adding 20 liters of chemical x to 80 liters of a mixture that is 30% chemical x
    results in a new mixture that is 44% chemical x. -/
theorem chemical_mixture_problem :
  let initial_volume : ℝ := 80
  let initial_concentration : ℝ := 0.30
  let added_volume : ℝ := 20
  let final_concentration : ℝ := 0.44
  (initial_volume * initial_concentration + added_volume) / (initial_volume + added_volume) = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_chemical_mixture_problem_l276_27639


namespace NUMINAMATH_CALUDE_train_crossing_time_l276_27676

/-- A train problem -/
theorem train_crossing_time
  (train_speed : ℝ)
  (platform_length : ℝ)
  (platform_crossing_time : ℝ)
  (h1 : train_speed = 20)
  (h2 : platform_length = 300)
  (h3 : platform_crossing_time = 30) :
  let train_length := train_speed * platform_crossing_time - platform_length
  (train_length / train_speed) = 15 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l276_27676


namespace NUMINAMATH_CALUDE_min_people_to_remove_l276_27609

def total_people : ℕ := 73

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def people_to_remove (n : ℕ) : ℕ := total_people - n

theorem min_people_to_remove :
  ∃ n : ℕ, is_square n ∧
    (∀ m : ℕ, is_square m → people_to_remove m ≥ people_to_remove n) ∧
    people_to_remove n = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_people_to_remove_l276_27609


namespace NUMINAMATH_CALUDE_tennis_players_l276_27643

theorem tennis_players (total : ℕ) (squash : ℕ) (neither : ℕ) (both : ℕ)
  (h1 : total = 38)
  (h2 : squash = 21)
  (h3 : neither = 10)
  (h4 : both = 12) :
  total - squash + both - neither = 19 :=
by sorry

end NUMINAMATH_CALUDE_tennis_players_l276_27643


namespace NUMINAMATH_CALUDE_min_value_theorem_l276_27696

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 1) (hab : a + b = 2) :
  ∃ (min : ℝ), min = 9/2 ∧ ∀ x, x = 1/(2*a) + 2/(b-1) → x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l276_27696


namespace NUMINAMATH_CALUDE_expression_equality_l276_27656

theorem expression_equality : |Real.sqrt 2 - 1| - (π + 1)^0 + Real.sqrt ((-3)^2) = Real.sqrt 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l276_27656


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l276_27699

theorem imaginary_part_of_z (z : ℂ) (h : (1 - Complex.I) * z = Complex.I) :
  z.im = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l276_27699


namespace NUMINAMATH_CALUDE_max_time_proof_l276_27600

/-- The number of digits in the lock combination -/
def num_digits : ℕ := 3

/-- The number of possible values for each digit (0 to 8, inclusive) -/
def digits_range : ℕ := 9

/-- The time in seconds required for each trial -/
def time_per_trial : ℕ := 3

/-- Calculates the maximum time in seconds required to try all combinations -/
def max_time_seconds : ℕ := digits_range ^ num_digits * time_per_trial

/-- Theorem: The maximum time required to try all combinations is 2187 seconds -/
theorem max_time_proof : max_time_seconds = 2187 := by
  sorry

end NUMINAMATH_CALUDE_max_time_proof_l276_27600


namespace NUMINAMATH_CALUDE_emilys_typing_speed_l276_27653

/-- Emily's typing speed problem -/
theorem emilys_typing_speed : 
  ∀ (words_typed : ℕ) (hours_taken : ℕ),
  words_typed = 10800 ∧ hours_taken = 3 →
  words_typed / (hours_taken * 60) = 60 :=
by sorry

end NUMINAMATH_CALUDE_emilys_typing_speed_l276_27653


namespace NUMINAMATH_CALUDE_angle_BDC_value_l276_27608

-- Define the angles in degrees
def angle_ABD : ℝ := 118
def angle_BCD : ℝ := 82

-- Define the theorem
theorem angle_BDC_value :
  ∀ (angle_BDC : ℝ),
  -- ABC is a straight line (implied by the exterior angle theorem)
  angle_ABD = angle_BCD + angle_BDC →
  angle_BDC = 36 := by
sorry

end NUMINAMATH_CALUDE_angle_BDC_value_l276_27608


namespace NUMINAMATH_CALUDE_complex_modulus_range_l276_27601

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z.re = a) (h4 : z.im = 1) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l276_27601


namespace NUMINAMATH_CALUDE_jellybean_problem_l276_27646

theorem jellybean_problem (x : ℕ) : 
  x ≥ 150 ∧ 
  x % 15 = 14 ∧ 
  x % 17 = 16 ∧ 
  (∀ y : ℕ, y ≥ 150 ∧ y % 15 = 14 ∧ y % 17 = 16 → x ≤ y) → 
  x = 254 := by
sorry

end NUMINAMATH_CALUDE_jellybean_problem_l276_27646


namespace NUMINAMATH_CALUDE_susie_fish_count_l276_27652

/-- The number of fish caught by each family member and the total number of filets --/
structure FishingTrip where
  ben_fish : ℕ
  judy_fish : ℕ
  billy_fish : ℕ
  jim_fish : ℕ
  susie_fish : ℕ
  thrown_back : ℕ
  total_filets : ℕ

/-- Theorem stating that Susie caught 3 fish given the conditions of the fishing trip --/
theorem susie_fish_count (trip : FishingTrip) 
  (h1 : trip.ben_fish = 4)
  (h2 : trip.judy_fish = 1)
  (h3 : trip.billy_fish = 3)
  (h4 : trip.jim_fish = 2)
  (h5 : trip.thrown_back = 3)
  (h6 : trip.total_filets = 24)
  (h7 : ∀ (fish : ℕ), fish * 2 = trip.total_filets → 
    fish = trip.ben_fish + trip.judy_fish + trip.billy_fish + trip.jim_fish + trip.susie_fish - trip.thrown_back) :
  trip.susie_fish = 3 := by
  sorry

end NUMINAMATH_CALUDE_susie_fish_count_l276_27652
