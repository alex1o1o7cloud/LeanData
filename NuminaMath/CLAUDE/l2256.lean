import Mathlib

namespace NUMINAMATH_CALUDE_binomial_expansion_sum_l2256_225604

theorem binomial_expansion_sum (n : ℕ) : 
  (∀ k : ℕ, k ≠ 2 → Nat.choose n 2 > Nat.choose n k) → 
  (1 - 2)^n = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_l2256_225604


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2256_225609

theorem triangle_abc_properties (A B C : ℝ) (h_obtuse : π / 2 < C ∧ C < π) 
  (h_sin_2c : Real.sin (2 * C) = Real.sqrt 3 * Real.cos C) 
  (h_b : Real.sqrt (A^2 + C^2 - 2*A*C*Real.cos B) = 6) 
  (h_area : 1/2 * Real.sqrt (A^2 + C^2 - 2*A*C*Real.cos B) * 
    Real.sqrt (B^2 + C^2 - 2*B*C*Real.cos A) * Real.sin C = 6 * Real.sqrt 3) : 
  C = 2 * π / 3 ∧ 
  Real.sqrt (A^2 + C^2 - 2*A*C*Real.cos B) + 
  Real.sqrt (B^2 + C^2 - 2*B*C*Real.cos A) + 
  Real.sqrt (A^2 + B^2 - 2*A*B*Real.cos C) = 10 + 2 * Real.sqrt 19 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l2256_225609


namespace NUMINAMATH_CALUDE_sector_angle_l2256_225676

theorem sector_angle (r : ℝ) (α : ℝ) 
  (h1 : 2 * r + α * r = 4)  -- circumference of sector is 4
  (h2 : (1 / 2) * α * r^2 = 1)  -- area of sector is 1
  : α = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_l2256_225676


namespace NUMINAMATH_CALUDE_doll_collection_increase_l2256_225619

/-- Proves that if adding 2 dolls to a collection increases it by 25%, then the final number of dolls in the collection is 10. -/
theorem doll_collection_increase (original : ℕ) : 
  (original + 2 : ℚ) = original * (1 + 1/4) → original + 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_doll_collection_increase_l2256_225619


namespace NUMINAMATH_CALUDE_alex_grocery_charge_percentage_l2256_225699

/-- The problem of determining Alex's grocery delivery charge percentage --/
theorem alex_grocery_charge_percentage :
  ∀ (car_cost savings_initial trip_charge trips_made grocery_total charge_percentage : ℚ),
  car_cost = 14600 →
  savings_initial = 14500 →
  trip_charge = (3/2) →
  trips_made = 40 →
  grocery_total = 800 →
  car_cost - savings_initial = trip_charge * trips_made + charge_percentage * grocery_total →
  charge_percentage = (1/20) := by
  sorry

end NUMINAMATH_CALUDE_alex_grocery_charge_percentage_l2256_225699


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2256_225661

def f (x : ℝ) := x^3 + x

theorem solution_set_of_inequality (x : ℝ) :
  x ∈ Set.Ioo (1/3 : ℝ) 3 ↔ 
  (x ∈ Set.Icc (-5 : ℝ) 5 ∧ f (2*x - 1) + f x > 0) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2256_225661


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2256_225649

theorem sqrt_equation_solution (y : ℝ) :
  Real.sqrt (3 + Real.sqrt (3 * y - 4)) = Real.sqrt 10 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2256_225649


namespace NUMINAMATH_CALUDE_solution_product_l2256_225696

/-- Given that p and q are the two distinct solutions of the equation
    (x - 5)(3x + 9) = x^2 - 16x + 55, prove that (p + 4)(q + 4) = -54 -/
theorem solution_product (p q : ℝ) : 
  (p - 5) * (3 * p + 9) = p^2 - 16 * p + 55 →
  (q - 5) * (3 * q + 9) = q^2 - 16 * q + 55 →
  p ≠ q →
  (p + 4) * (q + 4) = -54 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l2256_225696


namespace NUMINAMATH_CALUDE_minimum_days_to_find_poisoned_apple_l2256_225680

def number_of_apples : ℕ := 2021

theorem minimum_days_to_find_poisoned_apple :
  ∀ (n : ℕ), n = number_of_apples →
  (∃ (k : ℕ), 2^k ≥ n ∧ ∀ (m : ℕ), 2^m ≥ n → k ≤ m) →
  (∃ (k : ℕ), k = 11 ∧ 2^k ≥ n ∧ ∀ (m : ℕ), 2^m ≥ n → k ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_minimum_days_to_find_poisoned_apple_l2256_225680


namespace NUMINAMATH_CALUDE_woman_age_difference_l2256_225662

/-- The age of the son -/
def son_age : ℕ := 27

/-- The age of the woman -/
def woman_age : ℕ := 84 - son_age

/-- The difference between the woman's age and twice her son's age -/
def age_difference : ℕ := woman_age - 2 * son_age

theorem woman_age_difference : age_difference = 3 := by
  sorry

end NUMINAMATH_CALUDE_woman_age_difference_l2256_225662


namespace NUMINAMATH_CALUDE_diophantine_approximation_l2256_225694

theorem diophantine_approximation (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℕ), 1 ≤ b ∧ b ≤ n ∧ |a - b * Real.sqrt 2| ≤ 1 / n := by
  sorry

end NUMINAMATH_CALUDE_diophantine_approximation_l2256_225694


namespace NUMINAMATH_CALUDE_selectBooks_eq_1041_l2256_225684

/-- The number of ways to select books for three children -/
def selectBooks : ℕ :=
  let smallBooks := 6
  let largeBooks := 3
  let children := 3

  -- Case 1: All children take large books
  let case1 := largeBooks.factorial

  -- Case 2: 1 child takes small books, 2 take large books
  let case2 := Nat.choose children 1 * Nat.choose smallBooks 2 * Nat.choose largeBooks 2

  -- Case 3: 2 children take small books, 1 takes large book
  let case3 := Nat.choose children 2 * Nat.choose smallBooks 2 * Nat.choose (smallBooks - 2) 2 * Nat.choose largeBooks 1

  -- Case 4: All children take small books
  let case4 := Nat.choose smallBooks 2 * Nat.choose (smallBooks - 2) 2 * Nat.choose (smallBooks - 4) 2

  case1 + case2 + case3 + case4

theorem selectBooks_eq_1041 : selectBooks = 1041 := by
  sorry

end NUMINAMATH_CALUDE_selectBooks_eq_1041_l2256_225684


namespace NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2256_225639

theorem pure_imaginary_complex_number (a : ℝ) : 
  (∃ z : ℂ, z = Complex.mk (a^2 + a - 2) (a^2 - 3*a + 2) ∧ z.re = 0 ∧ z.im ≠ 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_complex_number_l2256_225639


namespace NUMINAMATH_CALUDE_complement_A_union_B_l2256_225632

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ (2 ≤ x ∧ x < 3)}
def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

-- State the theorem
theorem complement_A_union_B : 
  (Set.univ \ A) ∪ B = {x : ℝ | x ≥ -2} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l2256_225632


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l2256_225677

theorem right_triangle_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x ≤ x + y → x + y ≤ x + 3*y →
  (x + 3*y)^2 = x^2 + (x + y)^2 →
  x / y = 1 + Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l2256_225677


namespace NUMINAMATH_CALUDE_large_ball_radius_l2256_225636

theorem large_ball_radius (num_small_balls : ℕ) (small_radius : ℝ) (large_radius : ℝ) : 
  num_small_balls = 12 →
  small_radius = 2 →
  (4 / 3) * Real.pi * large_radius ^ 3 = num_small_balls * ((4 / 3) * Real.pi * small_radius ^ 3) →
  large_radius = (96 : ℝ) ^ (1 / 3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_large_ball_radius_l2256_225636


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l2256_225634

theorem number_satisfying_condition : ∃ x : ℝ, 0.05 * x = 0.20 * 650 + 190 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l2256_225634


namespace NUMINAMATH_CALUDE_one_valid_x_l2256_225621

def box_volume (x : ℤ) : ℤ := (x + 6) * (x - 6) * (x^2 + 36)

theorem one_valid_x : ∃! x : ℤ, 
  x > 0 ∧ 
  x - 6 > 0 ∧ 
  box_volume x < 800 :=
sorry

end NUMINAMATH_CALUDE_one_valid_x_l2256_225621


namespace NUMINAMATH_CALUDE_x_range_proof_l2256_225645

def S (n : ℕ) : ℝ := sorry

def a : ℕ → ℝ := sorry

theorem x_range_proof :
  (∀ n : ℕ, n ≥ 2 → S (n - 1) + S n = 2 * n^2 + 1) →
  (∀ n : ℕ, n ≥ 1 → a (n + 1) > a n) →
  a 1 = x →
  2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_CALUDE_x_range_proof_l2256_225645


namespace NUMINAMATH_CALUDE_product_of_valid_bases_l2256_225675

theorem product_of_valid_bases : ∃ (S : Finset ℕ), 
  (∀ b ∈ S, b ≥ 2 ∧ 
    (∃ (P : Finset ℕ), (∀ p ∈ P, Nat.Prime p) ∧ 
      Finset.card P = b ∧
      (b^6 - 1) / (b - 1) = Finset.prod P id)) ∧
  Finset.prod S id = 12 := by
  sorry

end NUMINAMATH_CALUDE_product_of_valid_bases_l2256_225675


namespace NUMINAMATH_CALUDE_problem_statement_l2256_225656

theorem problem_statement (x y : ℝ) (a : ℝ) :
  (x - a*y) * (x + a*y) = x^2 - 16*y^2 → a = 4 ∨ a = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2256_225656


namespace NUMINAMATH_CALUDE_product_of_numbers_l2256_225616

theorem product_of_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2256_225616


namespace NUMINAMATH_CALUDE_remaining_cube_volume_l2256_225651

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) :
  cube_side = 4 →
  cylinder_radius = 2 →
  (cube_side ^ 3) - (π * cylinder_radius ^ 2 * cube_side) = 64 - 16 * π :=
by sorry

end NUMINAMATH_CALUDE_remaining_cube_volume_l2256_225651


namespace NUMINAMATH_CALUDE_age_problem_l2256_225647

theorem age_problem (p q : ℕ) : 
  (p - 8 = (q - 8) / 2) →  -- 8 years ago, p was half of q's age
  (p * 4 = q * 3) →        -- The ratio of their present ages is 3:4
  (p + q = 28) :=          -- The total of their present ages is 28
by sorry

end NUMINAMATH_CALUDE_age_problem_l2256_225647


namespace NUMINAMATH_CALUDE_f_properties_l2256_225663

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x y : ℝ, x > -1 → y > -1 → x < y → f a x < f a y) ∧
  (∀ x : ℝ, x < 0 → f a x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2256_225663


namespace NUMINAMATH_CALUDE_ln_range_is_real_l2256_225641

-- Define the natural logarithm function
noncomputable def ln : ℝ → ℝ := Real.log

-- Statement: The range of the natural logarithm is all real numbers
theorem ln_range_is_real : Set.range ln = Set.univ := by sorry

end NUMINAMATH_CALUDE_ln_range_is_real_l2256_225641


namespace NUMINAMATH_CALUDE_first_discount_percentage_l2256_225620

/-- Proves that the first discount percentage is 15% given the original price,
    final price after two discounts, and the second discount percentage. -/
theorem first_discount_percentage
  (original_price : ℝ)
  (final_price : ℝ)
  (second_discount : ℝ)
  (h1 : original_price = 495)
  (h2 : final_price = 378.675)
  (h3 : second_discount = 10) :
  ∃ (first_discount : ℝ),
    first_discount = 15 ∧
    final_price = original_price * (100 - first_discount) / 100 * (100 - second_discount) / 100 :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l2256_225620


namespace NUMINAMATH_CALUDE_nine_points_chords_l2256_225653

/-- The number of different chords that can be drawn by connecting two points
    out of n points marked on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of different chords that can be drawn by connecting two points
    out of nine points marked on the circumference of a circle is equal to 36 -/
theorem nine_points_chords : num_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nine_points_chords_l2256_225653


namespace NUMINAMATH_CALUDE_smallest_exponent_divisibility_l2256_225683

theorem smallest_exponent_divisibility (x y z : ℕ+) 
  (h1 : x ∣ y^3) (h2 : y ∣ z^3) (h3 : z ∣ x^3) :
  (∀ n : ℕ, n < 13 → ¬(x * y * z ∣ (x + y + z)^n)) ∧
  (x * y * z ∣ (x + y + z)^13) := by
sorry

end NUMINAMATH_CALUDE_smallest_exponent_divisibility_l2256_225683


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l2256_225654

def binomial_sum (n : ℕ) : ℕ := 2^(n-1)

def rational_terms (n : ℕ) : List (ℕ × ℕ) :=
  [(5, 1), (4, 210)]

def coefficient_x_squared (n : ℕ) : ℕ :=
  (Finset.range (n - 2)).sum (λ k => Nat.choose (k + 3) 2)

theorem binomial_expansion_problem (n : ℕ) 
  (h : binomial_sum n = 512) : 
  n = 10 ∧ 
  rational_terms n = [(5, 1), (4, 210)] ∧
  coefficient_x_squared n = 164 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l2256_225654


namespace NUMINAMATH_CALUDE_oshea_basil_seeds_l2256_225630

/-- The number of basil seeds Oshea bought -/
def total_seeds : ℕ := sorry

/-- The number of large planters Oshea has -/
def large_planters : ℕ := 4

/-- The number of seeds each large planter can hold -/
def seeds_per_large_planter : ℕ := 20

/-- The number of seeds each small planter can hold -/
def seeds_per_small_planter : ℕ := 4

/-- The number of small planters needed to plant all the basil seeds -/
def small_planters : ℕ := 30

/-- Theorem stating that the total number of basil seeds Oshea bought is 200 -/
theorem oshea_basil_seeds : total_seeds = 200 := by sorry

end NUMINAMATH_CALUDE_oshea_basil_seeds_l2256_225630


namespace NUMINAMATH_CALUDE_fractions_not_both_integers_l2256_225681

theorem fractions_not_both_integers (n : ℤ) : 
  ¬(∃ (x y : ℤ), (n - 6 : ℤ) = 15 * x ∧ (n - 5 : ℤ) = 24 * y) := by
  sorry

end NUMINAMATH_CALUDE_fractions_not_both_integers_l2256_225681


namespace NUMINAMATH_CALUDE_movie_watchers_l2256_225611

theorem movie_watchers (total_seats empty_seats : ℕ) 
  (h1 : total_seats = 750)
  (h2 : empty_seats = 218) : 
  total_seats - empty_seats = 532 := by
sorry

end NUMINAMATH_CALUDE_movie_watchers_l2256_225611


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l2256_225625

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : (R^2 / r^2) = 4) :
  R - r = r :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l2256_225625


namespace NUMINAMATH_CALUDE_function_equal_to_inverse_is_identity_l2256_225642

-- Define an increasing function from R to R
def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Define the theorem
theorem function_equal_to_inverse_is_identity
  (f : ℝ → ℝ)
  (h_increasing : IncreasingFunction f)
  (h_inverse : ∀ x : ℝ, f x = Function.invFun f x) :
  ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_CALUDE_function_equal_to_inverse_is_identity_l2256_225642


namespace NUMINAMATH_CALUDE_percentage_problem_l2256_225692

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 800 = (20 / 100) * 650 + 190 → P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2256_225692


namespace NUMINAMATH_CALUDE_survey_participants_l2256_225646

theorem survey_participants (sample : ℕ) (percentage : ℚ) (total : ℕ) 
  (h1 : sample = 40)
  (h2 : percentage = 20 / 100)
  (h3 : sample = percentage * total) :
  total = 200 := by
sorry

end NUMINAMATH_CALUDE_survey_participants_l2256_225646


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2256_225686

theorem sum_of_solutions_is_zero (x : ℝ) (h : x^2 - 4 = 36) :
  ∃ y : ℝ, y^2 - 4 = 36 ∧ x + y = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l2256_225686


namespace NUMINAMATH_CALUDE_sum_is_zero_or_negative_two_l2256_225652

-- Define the conditions
def is_neither_positive_nor_negative (a : ℝ) : Prop := a = 0

def largest_negative_integer (b : ℤ) : Prop := b = -1

def reciprocal_is_self (c : ℝ) : Prop := c = 1 ∨ c = -1

-- Theorem statement
theorem sum_is_zero_or_negative_two 
  (a : ℝ) (b : ℤ) (c : ℝ) 
  (ha : is_neither_positive_nor_negative a)
  (hb : largest_negative_integer b)
  (hc : reciprocal_is_self c) :
  a + b + c = 0 ∨ a + b + c = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_is_zero_or_negative_two_l2256_225652


namespace NUMINAMATH_CALUDE_profit_share_difference_l2256_225695

/-- Represents the profit share of a party -/
structure ProfitShare where
  numerator : ℕ
  denominator : ℕ
  inv_pos : denominator > 0

/-- Calculates the profit for a given share and total profit -/
def calculate_profit (share : ProfitShare) (total_profit : ℚ) : ℚ :=
  total_profit * (share.numerator : ℚ) / (share.denominator : ℚ)

/-- The problem statement -/
theorem profit_share_difference 
  (total_profit : ℚ)
  (share_x share_y share_z : ProfitShare)
  (h_total : total_profit = 700)
  (h_x : share_x = ⟨1, 3, by norm_num⟩)
  (h_y : share_y = ⟨1, 4, by norm_num⟩)
  (h_z : share_z = ⟨1, 5, by norm_num⟩) :
  let profit_x := calculate_profit share_x total_profit
  let profit_y := calculate_profit share_y total_profit
  let profit_z := calculate_profit share_z total_profit
  let max_profit := max profit_x (max profit_y profit_z)
  let min_profit := min profit_x (min profit_y profit_z)
  ∃ (ε : ℚ), abs (max_profit - min_profit - 7148.93) < ε ∧ ε < 0.01 :=
sorry

end NUMINAMATH_CALUDE_profit_share_difference_l2256_225695


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_surface_area_l2256_225655

/-- A regular tetrahedron with given height and base edge length -/
structure RegularTetrahedron where
  height : ℝ
  base_edge : ℝ

/-- Volume of a regular tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ := sorry

/-- Surface area of a regular tetrahedron -/
def surface_area (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the volume and surface area of a specific regular tetrahedron -/
theorem tetrahedron_volume_and_surface_area :
  let t : RegularTetrahedron := ⟨1, 2 * Real.sqrt 6⟩
  volume t = 2 * Real.sqrt 3 ∧
  surface_area t = 9 * Real.sqrt 2 + 6 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_surface_area_l2256_225655


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implication_l2256_225689

-- Define a structure for a line in 3D space
structure Line3D where
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

-- Theorem statement
theorem perpendicular_parallel_implication (a b c : Line3D) 
  (h1 : perpendicular a b) (h2 : parallel b c) : perpendicular a c :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implication_l2256_225689


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l2256_225600

theorem smallest_integer_with_given_remainders : 
  ∃ (x : ℕ), x > 0 ∧ 
    x % 5 = 2 ∧ 
    x % 3 = 1 ∧ 
    x % 7 = 3 ∧
    (∀ y : ℕ, y > 0 → y % 5 = 2 → y % 3 = 1 → y % 7 = 3 → x ≤ y) ∧
    x = 22 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l2256_225600


namespace NUMINAMATH_CALUDE_decoration_cost_theorem_l2256_225667

/-- Calculates the total cost of decorations for a wedding reception. -/
def total_decoration_cost (num_tables : ℕ) 
                          (tablecloth_cost : ℕ) 
                          (place_setting_cost : ℕ) 
                          (place_settings_per_table : ℕ) 
                          (roses_per_centerpiece : ℕ) 
                          (rose_cost : ℕ) 
                          (lilies_per_centerpiece : ℕ) 
                          (lily_cost : ℕ) 
                          (daisies_per_centerpiece : ℕ) 
                          (daisy_cost : ℕ) 
                          (sunflowers_per_centerpiece : ℕ) 
                          (sunflower_cost : ℕ) 
                          (lighting_cost : ℕ) : ℕ :=
  let tablecloth_total := num_tables * tablecloth_cost
  let place_setting_total := num_tables * place_settings_per_table * place_setting_cost
  let centerpiece_cost := roses_per_centerpiece * rose_cost + 
                          lilies_per_centerpiece * lily_cost + 
                          daisies_per_centerpiece * daisy_cost + 
                          sunflowers_per_centerpiece * sunflower_cost
  let centerpiece_total := num_tables * centerpiece_cost
  tablecloth_total + place_setting_total + centerpiece_total + lighting_cost

theorem decoration_cost_theorem : 
  total_decoration_cost 30 25 12 6 15 6 20 5 5 3 3 4 450 = 9870 := by
  sorry

end NUMINAMATH_CALUDE_decoration_cost_theorem_l2256_225667


namespace NUMINAMATH_CALUDE_smallest_n_for_more_than_half_remaining_l2256_225628

def outer_layer_cubes (n : ℕ) : ℕ := 6 * n^2 - 12 * n + 8

def remaining_cubes (n : ℕ) : ℕ := n^3 - outer_layer_cubes n

theorem smallest_n_for_more_than_half_remaining : 
  (∀ k : ℕ, k < 10 → 2 * remaining_cubes k ≤ k^3) ∧
  (2 * remaining_cubes 10 > 10^3) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_more_than_half_remaining_l2256_225628


namespace NUMINAMATH_CALUDE_rectangular_paper_area_l2256_225612

/-- The area of a rectangular sheet of paper -/
def paper_area (width length : ℝ) : ℝ := width * length

theorem rectangular_paper_area :
  let width : ℝ := 25
  let length : ℝ := 20
  paper_area width length = 500 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_paper_area_l2256_225612


namespace NUMINAMATH_CALUDE_function_bound_on_unit_interval_l2256_225633

theorem function_bound_on_unit_interval 
  (f : Set.Icc 0 1 → ℝ)
  (h₁ : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h₂ : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₁ - f x₂| < |x₁.1 - x₂.1|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₁ - f x₂| < (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_function_bound_on_unit_interval_l2256_225633


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2256_225673

theorem complex_sum_problem (x y z w u v : ℝ) : 
  y = 2 ∧ 
  x = -z - u ∧ 
  (x + z + u) + (y + w + v) * I = 3 - 4 * I → 
  w + v = -6 := by sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2256_225673


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2256_225627

theorem max_gcd_13n_plus_4_8n_plus_3 :
  ∃ (k : ℕ), k > 0 ∧ gcd (13 * k + 4) (8 * k + 3) = 9 ∧
  ∀ (n : ℕ), n > 0 → gcd (13 * n + 4) (8 * n + 3) ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_4_8n_plus_3_l2256_225627


namespace NUMINAMATH_CALUDE_hungarian_deck_probabilities_l2256_225610

/-- Represents a Hungarian deck of cards -/
structure HungarianDeck :=
  (cards : Finset (Fin 32))
  (suits : Fin 4)
  (cardsPerSuit : Fin 8)

/-- Calculates the probability of drawing at least one ace given three cards of different suits -/
def probAceGivenDifferentSuits (deck : HungarianDeck) : ℚ :=
  169 / 512

/-- Calculates the probability of drawing at least one ace when drawing three cards -/
def probAceThreeCards (deck : HungarianDeck) : ℚ :=
  421 / 1240

/-- Calculates the probability of drawing three cards of different suits with at least one ace -/
def probDifferentSuitsWithAce (deck : HungarianDeck) : ℚ :=
  169 / 1240

/-- Main theorem stating the probabilities for the given scenarios -/
theorem hungarian_deck_probabilities (deck : HungarianDeck) :
  (probAceGivenDifferentSuits deck = 169 / 512) ∧
  (probAceThreeCards deck = 421 / 1240) ∧
  (probDifferentSuitsWithAce deck = 169 / 1240) :=
sorry

end NUMINAMATH_CALUDE_hungarian_deck_probabilities_l2256_225610


namespace NUMINAMATH_CALUDE_max_power_of_two_divides_l2256_225697

theorem max_power_of_two_divides (n : ℕ) (hn : n > 0) :
  (∃ m : ℕ, 3^(2*n+3) + 40*n - 27 = 2^6 * m) ∧
  (∃ n₀ : ℕ, n₀ > 0 ∧ ∀ m : ℕ, 3^(2*n₀+3) + 40*n₀ - 27 ≠ 2^7 * m) :=
sorry

end NUMINAMATH_CALUDE_max_power_of_two_divides_l2256_225697


namespace NUMINAMATH_CALUDE_hoonjeong_marbles_l2256_225602

theorem hoonjeong_marbles :
  ∀ (initial_marbles : ℝ),
    (initial_marbles * (1 - 0.2) * (1 - 0.35) = 130) →
    initial_marbles = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_hoonjeong_marbles_l2256_225602


namespace NUMINAMATH_CALUDE_bowl_cost_l2256_225640

theorem bowl_cost (sets : ℕ) (bowls_per_set : ℕ) (total_cost : ℕ) : 
  sets = 12 → bowls_per_set = 2 → total_cost = 240 → 
  (total_cost : ℚ) / (sets * bowls_per_set : ℚ) = 10 := by
  sorry

end NUMINAMATH_CALUDE_bowl_cost_l2256_225640


namespace NUMINAMATH_CALUDE_fraction_calls_team_B_value_l2256_225672

/-- Represents the fraction of calls processed by team B in a call center scenario -/
def fraction_calls_team_B (num_agents_A num_agents_B : ℚ) 
  (calls_per_agent_A calls_per_agent_B : ℚ) : ℚ :=
  (num_agents_B * calls_per_agent_B) / 
  (num_agents_A * calls_per_agent_A + num_agents_B * calls_per_agent_B)

/-- Theorem stating the fraction of calls processed by team B -/
theorem fraction_calls_team_B_value 
  (num_agents_A num_agents_B : ℚ) 
  (calls_per_agent_A calls_per_agent_B : ℚ) 
  (h1 : num_agents_A = (5 / 8) * num_agents_B)
  (h2 : calls_per_agent_A = (6 / 5) * calls_per_agent_B) :
  fraction_calls_team_B num_agents_A num_agents_B calls_per_agent_A calls_per_agent_B = 4 / 7 := by
  sorry


end NUMINAMATH_CALUDE_fraction_calls_team_B_value_l2256_225672


namespace NUMINAMATH_CALUDE_equal_sum_of_squares_l2256_225631

/-- Given a positive integer, return the sum of its digits -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The set of positive integers with at most n digits -/
def numbersWithAtMostNDigits (n : ℕ) : Set ℕ := sorry

/-- The set of positive integers with at most n digits and even digit sum -/
def evenDigitSumNumbers (n : ℕ) : Set ℕ := 
  {x ∈ numbersWithAtMostNDigits n | Even (digitSum x)}

/-- The set of positive integers with at most n digits and odd digit sum -/
def oddDigitSumNumbers (n : ℕ) : Set ℕ := 
  {x ∈ numbersWithAtMostNDigits n | Odd (digitSum x)}

/-- The sum of squares of elements in a set of natural numbers -/
def sumOfSquares (s : Set ℕ) : ℕ := sorry

theorem equal_sum_of_squares (n : ℕ) (h : n > 2) :
  sumOfSquares (evenDigitSumNumbers n) = sumOfSquares (oddDigitSumNumbers n) := by
  sorry

end NUMINAMATH_CALUDE_equal_sum_of_squares_l2256_225631


namespace NUMINAMATH_CALUDE_point_position_l2256_225614

theorem point_position (a : ℝ) : 
  (a < 0) → -- A is on the negative side of the origin
  (2 > 0) → -- B is on the positive side of the origin
  (|a + 3| = 4) → -- CO = 2BO, where BO = 2
  a = -7 := by
sorry

end NUMINAMATH_CALUDE_point_position_l2256_225614


namespace NUMINAMATH_CALUDE_curve_intersection_points_l2256_225665

-- Define the parametric equations of the curve
def x (t : ℝ) : ℝ := -2 + 5 * t
def y (t : ℝ) : ℝ := 1 - 2 * t

-- Theorem statement
theorem curve_intersection_points :
  (∃ t : ℝ, x t = 0 ∧ y t = 1/5) ∧
  (∃ t : ℝ, x t = 1/2 ∧ y t = 0) :=
by
  sorry


end NUMINAMATH_CALUDE_curve_intersection_points_l2256_225665


namespace NUMINAMATH_CALUDE_city_distance_l2256_225607

/-- The distance between Hallelujah City and San Pedro -/
def distance : ℝ := 1074

/-- The distance from San Pedro where the planes first meet -/
def first_meeting : ℝ := 437

/-- The distance from Hallelujah City where the planes meet on the return journey -/
def second_meeting : ℝ := 237

/-- The theorem stating the distance between the cities -/
theorem city_distance : 
  ∃ (v1 v2 : ℝ), v1 > v2 ∧ v1 > 0 ∧ v2 > 0 →
  first_meeting = v2 * (distance / (v1 + v2)) ∧
  second_meeting = v1 * (distance / (v1 + v2)) ∧
  distance = 1074 := by
sorry


end NUMINAMATH_CALUDE_city_distance_l2256_225607


namespace NUMINAMATH_CALUDE_speed_difference_l2256_225644

/-- The difference in average speeds between no traffic and heavy traffic conditions --/
theorem speed_difference (distance : ℝ) (time_heavy : ℝ) (time_no : ℝ)
  (h1 : distance = 200)
  (h2 : time_heavy = 5)
  (h3 : time_no = 4) :
  distance / time_no - distance / time_heavy = 10 := by
  sorry

end NUMINAMATH_CALUDE_speed_difference_l2256_225644


namespace NUMINAMATH_CALUDE_temperature_stats_l2256_225618

def temperatures : List ℝ := [12, 9, 10, 6, 11, 12, 17]

def median (l : List ℝ) : ℝ := sorry

def range (l : List ℝ) : ℝ := sorry

theorem temperature_stats :
  median temperatures = 11 ∧ range temperatures = 11 := by sorry

end NUMINAMATH_CALUDE_temperature_stats_l2256_225618


namespace NUMINAMATH_CALUDE_cunningham_lambs_count_l2256_225637

/-- Represents the total number of lambs owned by farmer Cunningham -/
def total_lambs : ℕ := 6048

/-- Represents the number of white lambs -/
def white_lambs : ℕ := 193

/-- Represents the number of black lambs -/
def black_lambs : ℕ := 5855

/-- Theorem stating that the total number of lambs is the sum of white and black lambs -/
theorem cunningham_lambs_count : total_lambs = white_lambs + black_lambs := by
  sorry

end NUMINAMATH_CALUDE_cunningham_lambs_count_l2256_225637


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l2256_225613

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem a_equals_one_sufficient_not_necessary (a : ℝ) :
  (a = 1 → is_purely_imaginary ((a - 1) * (a + 2) + (a + 3) * I)) ∧
  ¬(is_purely_imaginary ((a - 1) * (a + 2) + (a + 3) * I) → a = 1) :=
sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l2256_225613


namespace NUMINAMATH_CALUDE_max_total_profit_l2256_225635

/-- The fixed cost in million yuan -/
def fixed_cost : ℝ := 20

/-- The variable cost per unit in million yuan -/
def variable_cost_per_unit : ℝ := 10

/-- The total revenue function k(Q) in million yuan -/
def total_revenue (Q : ℝ) : ℝ := 40 * Q - Q^2

/-- The total cost function C(Q) in million yuan -/
def total_cost (Q : ℝ) : ℝ := fixed_cost + variable_cost_per_unit * Q

/-- The total profit function L(Q) in million yuan -/
def total_profit (Q : ℝ) : ℝ := total_revenue Q - total_cost Q

/-- The theorem stating that the maximum total profit is 205 million yuan -/
theorem max_total_profit : ∃ Q : ℝ, ∀ x : ℝ, total_profit Q ≥ total_profit x ∧ total_profit Q = 205 :=
sorry

end NUMINAMATH_CALUDE_max_total_profit_l2256_225635


namespace NUMINAMATH_CALUDE_division_inequality_l2256_225648

theorem division_inequality : ¬(∃ q r, 4900 = 600 * q + r ∧ r < 600 ∧ 49 = 6 * q + r ∧ r < 6) := by
  sorry

end NUMINAMATH_CALUDE_division_inequality_l2256_225648


namespace NUMINAMATH_CALUDE_compare_expressions_l2256_225666

theorem compare_expressions : 3 - Real.sqrt 2 > 4 - 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_compare_expressions_l2256_225666


namespace NUMINAMATH_CALUDE_digit_subtraction_result_l2256_225629

def three_digit_number (a b c : ℕ) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9

theorem digit_subtraction_result (a b c : ℕ) 
  (h1 : three_digit_number a b c) 
  (h2 : a = c + 2) : 
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) ≡ 8 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_digit_subtraction_result_l2256_225629


namespace NUMINAMATH_CALUDE_y_value_at_x_2_l2256_225687

/-- Given y₁ = x² - 7x + 6, y₂ = 7x - 3, and y = y₁ + xy₂, prove that when x = 2, y = 18. -/
theorem y_value_at_x_2 :
  let y₁ : ℝ → ℝ := λ x => x^2 - 7*x + 6
  let y₂ : ℝ → ℝ := λ x => 7*x - 3
  let y : ℝ → ℝ := λ x => y₁ x + x * y₂ x
  y 2 = 18 := by sorry

end NUMINAMATH_CALUDE_y_value_at_x_2_l2256_225687


namespace NUMINAMATH_CALUDE_total_earnings_theorem_l2256_225605

/-- Represents the investment and return ratios for three investors -/
structure InvestmentData where
  investment_ratio : Fin 3 → ℕ
  return_ratio : Fin 3 → ℕ

/-- Calculates the total earnings given investment data and the earnings difference between two investors -/
def calculate_total_earnings (data : InvestmentData) (earnings_diff : ℕ) : ℕ := sorry

/-- The main theorem stating the total earnings for the given scenario -/
theorem total_earnings_theorem (data : InvestmentData) (earnings_diff : ℕ) : 
  data.investment_ratio 0 = 3 ∧ 
  data.investment_ratio 1 = 4 ∧ 
  data.investment_ratio 2 = 5 ∧
  data.return_ratio 0 = 6 ∧ 
  data.return_ratio 1 = 5 ∧ 
  data.return_ratio 2 = 4 ∧
  earnings_diff = 120 →
  calculate_total_earnings data earnings_diff = 3480 := by sorry

end NUMINAMATH_CALUDE_total_earnings_theorem_l2256_225605


namespace NUMINAMATH_CALUDE_long_furred_brown_dogs_l2256_225657

theorem long_furred_brown_dogs 
  (total : ℕ) 
  (long_furred : ℕ) 
  (brown : ℕ) 
  (neither : ℕ) 
  (h1 : total = 45)
  (h2 : long_furred = 26)
  (h3 : brown = 30)
  (h4 : neither = 8) :
  long_furred + brown - (total - neither) = 19 := by
sorry

end NUMINAMATH_CALUDE_long_furred_brown_dogs_l2256_225657


namespace NUMINAMATH_CALUDE_four_rows_with_eight_people_l2256_225668

/-- Represents a seating arrangement with rows of 7 or 8 people -/
structure SeatingArrangement where
  total_people : ℕ
  rows_with_eight : ℕ
  rows_with_seven : ℕ

/-- Conditions for a valid seating arrangement -/
def is_valid_arrangement (s : SeatingArrangement) : Prop :=
  s.total_people = 46 ∧
  s.total_people = 8 * s.rows_with_eight + 7 * s.rows_with_seven

/-- Theorem stating that in a valid arrangement with 46 people, 
    there are exactly 4 rows with 8 people -/
theorem four_rows_with_eight_people 
  (s : SeatingArrangement) (h : is_valid_arrangement s) : 
  s.rows_with_eight = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_rows_with_eight_people_l2256_225668


namespace NUMINAMATH_CALUDE_team_total_points_l2256_225622

theorem team_total_points (player_points : ℕ) (percentage : ℚ) (h1 : player_points = 35) (h2 : percentage = 1/2) :
  player_points / percentage = 70 := by
  sorry

end NUMINAMATH_CALUDE_team_total_points_l2256_225622


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2256_225660

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2256_225660


namespace NUMINAMATH_CALUDE_range_of_f_less_than_one_l2256_225601

-- Define the function f
def f (x : ℝ) := x^3

-- State the theorem
theorem range_of_f_less_than_one :
  {x : ℝ | f x < 1} = Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_range_of_f_less_than_one_l2256_225601


namespace NUMINAMATH_CALUDE_rectangle_area_l2256_225688

/-- The area of a rectangle composed of 24 congruent squares arranged in a 6x4 grid, with a diagonal of 10 cm, is 600/13 square cm. -/
theorem rectangle_area (diagonal : ℝ) (rows columns : ℕ) (num_squares : ℕ) : 
  diagonal = 10 → 
  rows = 6 → 
  columns = 4 → 
  num_squares = 24 → 
  (rows * columns : ℝ) * (diagonal^2 / ((rows : ℝ)^2 + (columns : ℝ)^2)) = 600 / 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2256_225688


namespace NUMINAMATH_CALUDE_number_calculation_l2256_225658

theorem number_calculation (x : ℚ) : (x - 2) / 13 = 4 → (x - 5) / 7 = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_calculation_l2256_225658


namespace NUMINAMATH_CALUDE_work_completion_time_l2256_225674

/-- Given two workers A and B, where:
    - A and B together can complete a job in 6 days
    - A alone can complete the job in 14 days
    This theorem proves that B alone can complete the job in 10.5 days -/
theorem work_completion_time (work_rate_A : ℝ) (work_rate_B : ℝ) : 
  work_rate_A + work_rate_B = 1 / 6 →
  work_rate_A = 1 / 14 →
  1 / work_rate_B = 10.5 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l2256_225674


namespace NUMINAMATH_CALUDE_pencil_distribution_l2256_225679

theorem pencil_distribution (total_students : ℕ) (total_pencils : ℕ) 
  (h1 : total_students = 36)
  (h2 : total_pencils = 50)
  (h3 : ∃ (a b c : ℕ), a + b + c = total_students ∧ a + 2*b + 3*c = total_pencils ∧ a = 2*(b + c)) :
  ∃ (a b c : ℕ), a + b + c = total_students ∧ a + 2*b + 3*c = total_pencils ∧ a = 2*(b + c) ∧ b = 10 := by
  sorry

#check pencil_distribution

end NUMINAMATH_CALUDE_pencil_distribution_l2256_225679


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_nine_l2256_225693

theorem reciprocal_of_negative_nine (x : ℚ) : 
  (x * (-9) = 1) → x = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_nine_l2256_225693


namespace NUMINAMATH_CALUDE_total_animals_after_addition_l2256_225650

/-- Represents the number of animals on a farm --/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- Calculates the total number of animals on the farm --/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.pigs + farm.goats

/-- The initial number of animals on the farm --/
def initialFarm : FarmAnimals :=
  { cows := 2, pigs := 3, goats := 6 }

/-- The number of animals to be added to the farm --/
def addedAnimals : FarmAnimals :=
  { cows := 3, pigs := 5, goats := 2 }

/-- Theorem stating that the total number of animals after addition is 21 --/
theorem total_animals_after_addition :
  totalAnimals initialFarm + totalAnimals addedAnimals = 21 := by
  sorry


end NUMINAMATH_CALUDE_total_animals_after_addition_l2256_225650


namespace NUMINAMATH_CALUDE_sum_of_squares_l2256_225617

theorem sum_of_squares (a b c : ℝ) 
  (eq1 : a^2 + 3*b = 10)
  (eq2 : b^2 + 5*c = 0)
  (eq3 : c^2 + 7*a = -21) :
  a^2 + b^2 + c^2 = 83/4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2256_225617


namespace NUMINAMATH_CALUDE_complex_root_quadratic_equation_l2256_225623

theorem complex_root_quadratic_equation (q : ℝ) : 
  (2 * (Complex.mk (-3) 2)^2 + 12 * Complex.mk (-3) 2 + q = 0) → q = 26 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_quadratic_equation_l2256_225623


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l2256_225626

theorem algebraic_expression_equality (x : ℝ) : 
  x^2 + 2*x + 5 = 6 → 2*x^2 + 4*x + 15 = 17 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l2256_225626


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l2256_225690

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 4 / 5 →  -- The ratio of the angles is 4:5
  b = 100 :=  -- The larger angle is 100°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l2256_225690


namespace NUMINAMATH_CALUDE_power_twenty_equals_r_s_l2256_225603

theorem power_twenty_equals_r_s (a b : ℤ) (R S : ℝ) 
  (hR : R = 2^a) (hS : S = 5^b) : 
  R^(2*b) * S^a = 20^(a*b) := by
sorry

end NUMINAMATH_CALUDE_power_twenty_equals_r_s_l2256_225603


namespace NUMINAMATH_CALUDE_problem_solution_l2256_225659

/-- A function satisfying the given property for all real numbers -/
def satisfies_property (g : ℝ → ℝ) : Prop :=
  ∀ a c : ℝ, c^3 * g a = a^3 * g c

theorem problem_solution (g : ℝ → ℝ) (h1 : satisfies_property g) (h2 : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2256_225659


namespace NUMINAMATH_CALUDE_james_new_weight_l2256_225664

/-- Calculates the new weight after muscle and fat gain -/
def new_weight (initial_weight : ℝ) (muscle_gain_percentage : ℝ) (fat_gain_ratio : ℝ) : ℝ :=
  let muscle_gain := initial_weight * muscle_gain_percentage
  let fat_gain := muscle_gain * fat_gain_ratio
  initial_weight + muscle_gain + fat_gain

/-- Proves that James's new weight is 150 kg after gaining muscle and fat -/
theorem james_new_weight :
  new_weight 120 0.2 0.25 = 150 := by
  sorry

end NUMINAMATH_CALUDE_james_new_weight_l2256_225664


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l2256_225624

theorem min_value_trig_expression (α β : Real) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 215 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l2256_225624


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l2256_225608

theorem trigonometric_expression_evaluation (α : Real) (h : Real.tan α = 2) :
  (Real.cos (-π/2 - α) * Real.tan (π + α) - Real.sin (π/2 - α)) /
  (Real.cos (3*π/2 + α) + Real.cos (π - α)) = -5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l2256_225608


namespace NUMINAMATH_CALUDE_davids_remaining_money_l2256_225671

/-- The amount of money David has left after mowing lawns, buying shoes, and giving money to his mom. -/
def davidsRemainingMoney (hourlyRate : ℚ) (hoursPerDay : ℚ) (daysPerWeek : ℕ) : ℚ :=
  let totalEarned := hourlyRate * hoursPerDay * daysPerWeek
  let afterShoes := totalEarned / 2
  afterShoes / 2

theorem davids_remaining_money :
  davidsRemainingMoney 14 2 7 = 49 := by
  sorry

#eval davidsRemainingMoney 14 2 7

end NUMINAMATH_CALUDE_davids_remaining_money_l2256_225671


namespace NUMINAMATH_CALUDE_sams_pen_collection_l2256_225685

/-- The number of pens in Sam's collection -/
def total_pens (black blue red pencils : ℕ) : ℕ := black + blue + red

/-- The problem statement -/
theorem sams_pen_collection :
  ∀ (black blue red pencils : ℕ),
  black = blue + 10 →
  blue = 2 * pencils →
  pencils = 8 →
  red = pencils - 2 →
  total_pens black blue red pencils = 48 := by
sorry

end NUMINAMATH_CALUDE_sams_pen_collection_l2256_225685


namespace NUMINAMATH_CALUDE_jessicas_mothers_death_years_jessicas_mothers_death_years_proof_l2256_225638

/-- Prove that the number of years passed since Jessica's mother's death is 10 -/
theorem jessicas_mothers_death_years : ℕ :=
  let jessica_current_age : ℕ := 40
  let mother_hypothetical_age : ℕ := 70
  let years_passed : ℕ → Prop := λ x =>
    -- Jessica was half her mother's age when her mother died
    2 * (jessica_current_age - x) = jessica_current_age - x + x ∧
    -- Jessica's mother would be 70 if she were alive now
    jessica_current_age - x + x = mother_hypothetical_age
  10

theorem jessicas_mothers_death_years_proof :
  jessicas_mothers_death_years = 10 := by sorry

end NUMINAMATH_CALUDE_jessicas_mothers_death_years_jessicas_mothers_death_years_proof_l2256_225638


namespace NUMINAMATH_CALUDE_tourist_count_scientific_notation_l2256_225682

theorem tourist_count_scientific_notation :
  ∀ (n : ℝ), n = 15.276 * 1000000 → 
  ∃ (a : ℝ) (b : ℤ), n = a * (10 : ℝ) ^ b ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.5276 ∧ b = 7 :=
by sorry

end NUMINAMATH_CALUDE_tourist_count_scientific_notation_l2256_225682


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2256_225691

theorem probability_of_white_ball (total_balls : Nat) (red_balls white_balls : Nat) :
  total_balls = red_balls + white_balls + 1 →
  red_balls = 2 →
  white_balls = 3 →
  (white_balls : ℚ) / (total_balls - 1 : ℚ) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2256_225691


namespace NUMINAMATH_CALUDE_tag_sum_is_1000_l2256_225669

/-- The sum of the numbers tagged on four cards W, X, Y, Z -/
def total_tag_sum (w x y z : ℕ) : ℕ := w + x + y + z

/-- Theorem stating that the sum of the tagged numbers is 1000 -/
theorem tag_sum_is_1000 :
  ∀ (w x y z : ℕ),
  w = 200 →
  x = w / 2 →
  y = w + x →
  z = 400 →
  total_tag_sum w x y z = 1000 := by
  sorry

end NUMINAMATH_CALUDE_tag_sum_is_1000_l2256_225669


namespace NUMINAMATH_CALUDE_proposition_equivalence_l2256_225670

theorem proposition_equivalence (a : ℝ) :
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a ≥ 0) ↔ a ≥ -8 := by
  sorry

end NUMINAMATH_CALUDE_proposition_equivalence_l2256_225670


namespace NUMINAMATH_CALUDE_stair_step_24th_row_white_squares_l2256_225643

/-- Represents the number of squares in a row of the stair-step figure -/
def total_squares (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- Represents the number of white squares in a row of the stair-step figure -/
def white_squares (n : ℕ) : ℕ := (total_squares n - 2) / 2 + (total_squares n - 2) % 2

/-- Theorem stating that the 24th row of the stair-step figure contains 23 white squares -/
theorem stair_step_24th_row_white_squares :
  white_squares 24 = 23 := by sorry

end NUMINAMATH_CALUDE_stair_step_24th_row_white_squares_l2256_225643


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2256_225698

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 + a 3 = 5 →
  a 3 + a 5 = 20 →
  a 5 + a 7 = 80 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2256_225698


namespace NUMINAMATH_CALUDE_fraction_less_than_decimal_l2256_225606

theorem fraction_less_than_decimal : (7 : ℚ) / 24 < (3 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_less_than_decimal_l2256_225606


namespace NUMINAMATH_CALUDE_polyhedron_space_diagonals_l2256_225678

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - 2 * Q.quadrilateral_faces

/-- Theorem: A convex polyhedron Q with 30 vertices, 72 edges, 44 faces 
    (30 triangular and 14 quadrilateral) has 335 space diagonals -/
theorem polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 14
  }
  space_diagonals Q = 335 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_space_diagonals_l2256_225678


namespace NUMINAMATH_CALUDE_min_value_a_l2256_225615

theorem min_value_a (m n : ℝ) (h1 : 0 < n) (h2 : n < m) (h3 : m < 1/a) 
  (h4 : (n^(1/m)) / (m^(1/n)) > (n^a) / (m^a)) : 
  ∀ ε > 0, ∃ a : ℝ, a ≥ 1 ∧ a < 1 + ε := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l2256_225615
