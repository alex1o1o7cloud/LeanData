import Mathlib

namespace NUMINAMATH_CALUDE_unique_quadratic_trinomial_l3218_321897

theorem unique_quadratic_trinomial : ∃! (a b c : ℝ), 
  (∀ x : ℝ, (a + 1) * x^2 + b * x + c = 0 → (∃! y : ℝ, y = x)) ∧
  (∀ x : ℝ, a * x^2 + (b + 1) * x + c = 0 → (∃! y : ℝ, y = x)) ∧
  (∀ x : ℝ, a * x^2 + b * x + (c + 1) = 0 → (∃! y : ℝ, y = x)) ∧
  a = 1/8 ∧ b = -3/4 ∧ c = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_trinomial_l3218_321897


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3218_321884

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3218_321884


namespace NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l3218_321840

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := (x - 2) * (a * x + b)

-- State the theorem
theorem solution_set_of_even_increasing_function 
  (a b : ℝ) 
  (h_even : ∀ x, f a b x = f a b (-x)) 
  (h_increasing : ∀ x y, 0 < x → x < y → f a b x < f a b y) :
  {x : ℝ | f a b (2 - x) > 0} = {x : ℝ | x > 4 ∨ x < 0} := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_even_increasing_function_l3218_321840


namespace NUMINAMATH_CALUDE_common_divisor_sequence_l3218_321821

theorem common_divisor_sequence (n : ℕ) : n = 4190 →
  ∀ k ∈ Finset.range 21, ∃ d > 1, d ∣ (n + k) ∧ d ∣ 30030 := by
  sorry

#check common_divisor_sequence

end NUMINAMATH_CALUDE_common_divisor_sequence_l3218_321821


namespace NUMINAMATH_CALUDE_grocery_store_soda_l3218_321814

theorem grocery_store_soda (total : ℕ) (diet : ℕ) (regular : ℕ) : 
  total = 30 → diet = 2 → regular = total - diet → regular = 28 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_soda_l3218_321814


namespace NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l3218_321815

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l3218_321815


namespace NUMINAMATH_CALUDE_fencing_required_l3218_321808

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_required (area : ℝ) (uncovered_side : ℝ) (fencing : ℝ) : 
  area = 650 ∧ uncovered_side = 20 → fencing = 85 := by
  sorry

end NUMINAMATH_CALUDE_fencing_required_l3218_321808


namespace NUMINAMATH_CALUDE_largest_gold_coins_l3218_321899

theorem largest_gold_coins : 
  ∃ (n : ℕ), n = 146 ∧ 
  (∃ (k : ℕ), n = 13 * k + 3) ∧ 
  n < 150 ∧
  ∀ (m : ℕ), (∃ (j : ℕ), m = 13 * j + 3) → m < 150 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_gold_coins_l3218_321899


namespace NUMINAMATH_CALUDE_divisibility_property_l3218_321813

theorem divisibility_property :
  ∀ k : ℤ, k ≠ 2013 → (2013 - k) ∣ (2013^2014 - k^2014) := by
sorry

end NUMINAMATH_CALUDE_divisibility_property_l3218_321813


namespace NUMINAMATH_CALUDE_infinite_good_primes_infinite_non_good_primes_l3218_321867

/-- Definition of a good prime -/
def is_good_prime (p : ℕ) : Prop :=
  Prime p ∧ ∀ a b : ℕ, a > 0 → b > 0 → (a ≡ b [ZMOD p] ↔ a^3 ≡ b^3 [ZMOD p])

/-- The set of good primes is infinite -/
theorem infinite_good_primes : Set.Infinite {p : ℕ | is_good_prime p} :=
sorry

/-- The set of non-good primes is infinite -/
theorem infinite_non_good_primes : Set.Infinite {p : ℕ | Prime p ∧ ¬is_good_prime p} :=
sorry

end NUMINAMATH_CALUDE_infinite_good_primes_infinite_non_good_primes_l3218_321867


namespace NUMINAMATH_CALUDE_no_solution_to_diophantine_equation_l3218_321889

theorem no_solution_to_diophantine_equation :
  ¬ ∃ (x y z t : ℕ), 3 * x^4 + 5 * y^4 + 7 * z^4 = 11 * t^4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_diophantine_equation_l3218_321889


namespace NUMINAMATH_CALUDE_root_squares_sum_l3218_321801

theorem root_squares_sum (a b : ℝ) (a_ne_b : a ≠ b) : 
  ∃ (s t : ℝ), (a * s^2 + b * s + b = 0) ∧ 
                (a * t^2 + a * t + b = 0) ∧ 
                (s * t = 1) → 
                (s^2 + t^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_root_squares_sum_l3218_321801


namespace NUMINAMATH_CALUDE_factor_condition_l3218_321805

theorem factor_condition (n : ℕ) (hn : n ≥ 2) 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (k : ℕ), n < 2 * k + 1 ∧ 2 * k + 1 < 3 * n ∧
    a = (-2 * Real.cos ((2 * k + 1 : ℝ) * π / (2 * n : ℝ))) ^ (2 * n / (2 * n - 1 : ℝ)) ∧
    b = (2 * Real.cos ((2 * k + 1 : ℝ) * π / (2 * n : ℝ))) ^ (2 / (2 * n - 1 : ℝ))) ↔
  (∀ x : ℂ, (x ^ 2 + a * x + b = 0) → (a * x ^ (2 * n) + (a * x + b) ^ (2 * n) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_factor_condition_l3218_321805


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3218_321841

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 12) = 10 → x = 88 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3218_321841


namespace NUMINAMATH_CALUDE_ticket_sales_total_l3218_321854

/-- Calculates the total amount collected from ticket sales -/
def totalAmountCollected (adultPrice studentPrice : ℚ) (totalTickets studentTickets : ℕ) : ℚ :=
  let adultTickets := totalTickets - studentTickets
  adultPrice * adultTickets + studentPrice * studentTickets

/-- Proves that the total amount collected from ticket sales is 222.50 -/
theorem ticket_sales_total : 
  totalAmountCollected 4 (5/2) 59 9 = 445/2 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_total_l3218_321854


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_specific_roots_l3218_321822

theorem sum_of_squares_of_roots (a b c : ℚ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x₁^2 + b * x₁ + c = 0 ∧ 
  a * x₂^2 + b * x₂ + c = 0 →
  x₁^2 + x₂^2 = (b^2 / a^2) + 2 * (c / a) :=
by sorry

theorem sum_of_squares_of_specific_roots :
  let a : ℚ := 5
  let b : ℚ := -3
  let c : ℚ := -11
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x₁^2 + b * x₁ + c = 0 ∧ 
  a * x₂^2 + b * x₂ + c = 0 →
  x₁^2 + x₂^2 = 119 / 25 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_specific_roots_l3218_321822


namespace NUMINAMATH_CALUDE_storks_joined_equals_six_l3218_321853

/-- The number of storks that joined the birds on the fence -/
def num_storks_joined : ℕ := 6

/-- The initial number of birds on the fence -/
def initial_birds : ℕ := 3

/-- The number of birds that joined later -/
def birds_joined : ℕ := 2

theorem storks_joined_equals_six :
  let total_birds := initial_birds + birds_joined
  num_storks_joined = total_birds + 1 :=
by sorry

end NUMINAMATH_CALUDE_storks_joined_equals_six_l3218_321853


namespace NUMINAMATH_CALUDE_library_book_count_l3218_321829

/-- The number of books the library had before the grant -/
def initial_books : ℕ := 5935

/-- The number of books purchased with the grant -/
def purchased_books : ℕ := 2647

/-- The total number of books after the grant -/
def total_books : ℕ := initial_books + purchased_books

theorem library_book_count : total_books = 8582 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l3218_321829


namespace NUMINAMATH_CALUDE_road_section_last_point_location_l3218_321826

/-- Given a road section from start_point to end_point divided into num_sections equal parts,
    the location of the last point is equal to the end_point. -/
theorem road_section_last_point_location
  (start_point end_point : ℝ)
  (num_sections : ℕ)
  (h1 : start_point = 0.35)
  (h2 : end_point = 0.37)
  (h3 : num_sections = 4)
  (h4 : start_point < end_point) :
  start_point + num_sections * ((end_point - start_point) / num_sections) = end_point :=
by sorry

end NUMINAMATH_CALUDE_road_section_last_point_location_l3218_321826


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3218_321809

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (π / 4 - α) = -4 / 5) : 
  Real.sin (2 * α) = 7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3218_321809


namespace NUMINAMATH_CALUDE_black_area_after_seven_changes_l3218_321878

/-- Represents the fraction of black area remaining after a number of changes -/
def blackFraction (changes : ℕ) : ℚ :=
  (8/9) ^ changes

/-- The number of changes applied to the triangle -/
def numChanges : ℕ := 7

/-- Theorem stating the fraction of black area after seven changes -/
theorem black_area_after_seven_changes :
  blackFraction numChanges = 2097152/4782969 := by
  sorry

#eval blackFraction numChanges

end NUMINAMATH_CALUDE_black_area_after_seven_changes_l3218_321878


namespace NUMINAMATH_CALUDE_star_operation_result_l3218_321861

def star (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem star_operation_result :
  let M : Set ℕ := {1, 2, 3, 4, 5}
  let P : Set ℕ := {2, 3, 6}
  star P M = {6} := by sorry

end NUMINAMATH_CALUDE_star_operation_result_l3218_321861


namespace NUMINAMATH_CALUDE_julian_needs_more_legos_l3218_321847

/-- The number of legos Julian has -/
def julianLegos : ℕ := 400

/-- The number of legos required for one airplane model -/
def legosPerModel : ℕ := 240

/-- The number of airplane models Julian wants to make -/
def numModels : ℕ := 2

/-- The number of additional legos Julian needs -/
def additionalLegosNeeded : ℕ := 80

theorem julian_needs_more_legos : 
  julianLegos + additionalLegosNeeded = legosPerModel * numModels := by
  sorry

end NUMINAMATH_CALUDE_julian_needs_more_legos_l3218_321847


namespace NUMINAMATH_CALUDE_common_tangent_lines_l3218_321888

/-- Represents a circle with equation x^2 + y^2 - (4m + 2)x - 2my + 4m^2 + 4m + 1 = 0 -/
def Circle (m : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 - (4*m + 2)*x - 2*m*y + 4*m^2 + 4*m + 1 = 0}

/-- Checks if a line y = kx + b is tangent to a circle -/
def isTangentLine (k b : ℝ) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ (x y : ℝ), (x, y) ∈ c ∧ y = k*x + b ∧ 
  ∀ (x' y' : ℝ), (x', y') ∈ c → (y' = k*x' + b → (x', y') = (x, y))

theorem common_tangent_lines (m : ℝ) (h : m > 0) :
  (isTangentLine 0 0 (Circle m)) ∧ 
  (isTangentLine (4/3) (-4/3) (Circle m)) :=
sorry

end NUMINAMATH_CALUDE_common_tangent_lines_l3218_321888


namespace NUMINAMATH_CALUDE_total_owed_correct_l3218_321882

/-- Calculates the total amount owed after one year given monthly charges and interest rates -/
def totalOwed (jan_charge feb_charge mar_charge apr_charge : ℝ)
              (jan_rate feb_rate mar_rate apr_rate : ℝ) : ℝ :=
  let jan_total := jan_charge * (1 + jan_rate)
  let feb_total := feb_charge * (1 + feb_rate)
  let mar_total := mar_charge * (1 + mar_rate)
  let apr_total := apr_charge * (1 + apr_rate)
  jan_total + feb_total + mar_total + apr_total

/-- The theorem stating the total amount owed after one year -/
theorem total_owed_correct :
  totalOwed 35 45 55 25 0.05 0.07 0.04 0.06 = 168.60 := by
  sorry

end NUMINAMATH_CALUDE_total_owed_correct_l3218_321882


namespace NUMINAMATH_CALUDE_distance_B_C_is_250_l3218_321856

/-- Represents a city in a triangle of cities -/
structure City :=
  (name : String)

/-- Represents the distance between two cities -/
def distance (a b : City) : ℝ := sorry

/-- The theorem stating the distance between cities B and C -/
theorem distance_B_C_is_250 (A B C : City) 
  (h1 : distance A B = distance A C + distance B C - 200)
  (h2 : distance A C = distance A B + distance B C - 300) :
  distance B C = 250 := by sorry

end NUMINAMATH_CALUDE_distance_B_C_is_250_l3218_321856


namespace NUMINAMATH_CALUDE_second_fraction_l3218_321812

theorem second_fraction (n : ℚ) (h1 : n = 0.5833333333333333) (h2 : n = 1/3 + x) : x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_second_fraction_l3218_321812


namespace NUMINAMATH_CALUDE_triangle_perpendicular_segment_length_l3218_321857

-- Define the triangle XYZ
structure Triangle (X Y Z : ℝ × ℝ) : Prop where
  right_angle : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0
  xy_length : Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 5
  xz_length : Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2) = 12

-- Define the perpendicular segment LM
def perpendicular_segment (X Y Z M : ℝ × ℝ) : Prop :=
  (M.1 - X.1) * (Y.1 - X.1) + (M.2 - X.2) * (Y.2 - X.2) = 0

-- Theorem statement
theorem triangle_perpendicular_segment_length 
  (X Y Z M : ℝ × ℝ) (h : Triangle X Y Z) (h_perp : perpendicular_segment X Y Z M) :
  Real.sqrt ((M.1 - Y.1)^2 + (M.2 - Y.2)^2) = (5 * Real.sqrt 119) / 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perpendicular_segment_length_l3218_321857


namespace NUMINAMATH_CALUDE_square_root_sum_l3218_321803

theorem square_root_sum (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = 50) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 52 := by
sorry

end NUMINAMATH_CALUDE_square_root_sum_l3218_321803


namespace NUMINAMATH_CALUDE_f_intersects_axes_twice_l3218_321848

/-- The quadratic function f(x) = x^2 + 4x + 4 -/
def f (x : ℝ) : ℝ := x^2 + 4*x + 4

/-- The number of intersection points between f(x) and the coordinate axes -/
def num_intersections : ℕ := 2

/-- Theorem stating that f(x) intersects the coordinate axes at exactly two points -/
theorem f_intersects_axes_twice :
  (∃! x : ℝ, f x = 0) ∧ (∃! y : ℝ, f 0 = y) ∧ num_intersections = 2 :=
sorry

end NUMINAMATH_CALUDE_f_intersects_axes_twice_l3218_321848


namespace NUMINAMATH_CALUDE_lemon_orange_scaling_l3218_321871

def lemon_orange_drink (gallons : ℚ) (lemons : ℚ) (oranges : ℚ) : Prop :=
  gallons > 0 ∧ lemons > 0 ∧ oranges > 0 ∧
  lemons / gallons = 30 / 40 ∧ oranges / gallons = 20 / 40

theorem lemon_orange_scaling (g1 g2 l1 l2 o1 o2 : ℚ) :
  lemon_orange_drink g1 l1 o1 →
  lemon_orange_drink g2 l2 o2 →
  g2 = (100 : ℚ) / 40 * g1 →
  l2 = 75 ∧ o2 = 50 :=
sorry

end NUMINAMATH_CALUDE_lemon_orange_scaling_l3218_321871


namespace NUMINAMATH_CALUDE_fruit_basket_total_cost_l3218_321859

/-- Represents the cost of a fruit basket -/
def fruit_basket_cost (banana_price : ℚ) (apple_price : ℚ) (strawberry_price : ℚ) 
  (avocado_price : ℚ) (grape_price : ℚ) : ℚ :=
  4 * banana_price + 3 * apple_price + 24 * strawberry_price / 12 + 
  2 * avocado_price + 2 * grape_price

/-- Theorem stating the total cost of the fruit basket -/
theorem fruit_basket_total_cost : 
  fruit_basket_cost 1 2 (4/12) 3 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_total_cost_l3218_321859


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l3218_321851

/-- Geometric sequence with the given property -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- Arithmetic sequence with the given property -/
def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, b (n + 1) = b n + d

/-- Sum of first n terms of a sequence -/
def sum_of_terms (s : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum s

theorem sequence_sum_theorem (a b : ℕ → ℝ) :
  geometric_sequence a →
  arithmetic_sequence b →
  3 * a 5 - a 3 * a 7 = 0 →
  b 5 = a 5 →
  sum_of_terms b 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l3218_321851


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_20_l3218_321875

def digit_sum (n : Nat) : Nat :=
  Nat.digits 10 n |>.sum

def all_digits_different (n : Nat) : Prop :=
  (Nat.digits 10 n).Nodup

def no_zero_digit (n : Nat) : Prop :=
  0 ∉ Nat.digits 10 n

theorem largest_number_with_digit_sum_20 :
  ∀ n : Nat,
    (digit_sum n = 20 ∧
     all_digits_different n ∧
     no_zero_digit n) →
    n ≤ 9821 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_20_l3218_321875


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3218_321874

theorem sufficient_not_necessary_condition :
  (∀ a b : ℝ, a > 2 ∧ b > 2 → a + b > 4 ∧ a * b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ a * b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3218_321874


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3218_321842

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 2 + Complex.I → z = 1 - 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3218_321842


namespace NUMINAMATH_CALUDE_inequality_solution_l3218_321895

theorem inequality_solution (x : ℝ) : 
  (5 * x^2 + 10 * x - 34) / ((x - 2) * (3 * x + 5)) < 2 ↔ 
  x < -5/3 ∨ x > 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3218_321895


namespace NUMINAMATH_CALUDE_equation_solution_l3218_321863

theorem equation_solution :
  ∃ x : ℝ, (64 : ℝ) ^ (3 * x + 1) = (16 : ℝ) ^ (4 * x - 5) ∧ x = -13 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3218_321863


namespace NUMINAMATH_CALUDE_soda_cost_l3218_321802

theorem soda_cost (bill : ℕ) (change : ℕ) (num_sodas : ℕ) (h1 : bill = 20) (h2 : change = 14) (h3 : num_sodas = 3) :
  (bill - change) / num_sodas = 2 := by
sorry

end NUMINAMATH_CALUDE_soda_cost_l3218_321802


namespace NUMINAMATH_CALUDE_career_preference_theorem_l3218_321832

/-- Represents the number of degrees in a circle that should be allocated to a career
    preference in a class with a given boy-to-girl ratio and preference ratios. -/
def career_preference_degrees (boy_ratio girl_ratio : ℚ) 
                               (boy_preference girl_preference : ℚ) : ℚ :=
  let total_parts := boy_ratio + girl_ratio
  let boy_parts := (boy_preference * boy_ratio) / total_parts
  let girl_parts := (girl_preference * girl_ratio) / total_parts
  let preference_fraction := (boy_parts + girl_parts) / 1
  preference_fraction * 360

/-- Theorem stating that for a class with a 2:3 ratio of boys to girls, 
    where 1/3 of boys and 2/3 of girls prefer a career, 
    192 degrees should be used to represent this preference in a circle graph. -/
theorem career_preference_theorem : 
  career_preference_degrees 2 3 (1/3) (2/3) = 192 := by
  sorry

end NUMINAMATH_CALUDE_career_preference_theorem_l3218_321832


namespace NUMINAMATH_CALUDE_exactly_three_sets_l3218_321819

/-- A set of consecutive positive integers -/
structure ConsecutiveSet :=
  (start : ℕ)
  (length : ℕ)
  (length_ge_two : length ≥ 2)

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- Predicate for a valid set of consecutive integers summing to 150 -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  sum_consecutive s = 150

theorem exactly_three_sets : 
  ∃! (sets : Finset ConsecutiveSet), 
    (∀ s ∈ sets, is_valid_set s) ∧ 
    sets.card = 3 := by sorry

end NUMINAMATH_CALUDE_exactly_three_sets_l3218_321819


namespace NUMINAMATH_CALUDE_point_not_on_graph_l3218_321823

/-- A linear function passing through (1, 2) -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x

/-- The theorem stating that (1, -2) is not on the graph of the function -/
theorem point_not_on_graph (k : ℝ) (h1 : k ≠ 0) (h2 : f k 1 = 2) :
  f k 1 ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l3218_321823


namespace NUMINAMATH_CALUDE_smallest_seem_l3218_321831

/-- Represents a digit mapping for the puzzle MY + ROZH = SEEM -/
structure DigitMapping where
  m : Nat
  y : Nat
  r : Nat
  o : Nat
  z : Nat
  h : Nat
  s : Nat
  e : Nat
  unique : m ≠ y ∧ m ≠ r ∧ m ≠ o ∧ m ≠ z ∧ m ≠ h ∧ m ≠ s ∧ m ≠ e ∧
           y ≠ r ∧ y ≠ o ∧ y ≠ z ∧ y ≠ h ∧ y ≠ s ∧ y ≠ e ∧
           r ≠ o ∧ r ≠ z ∧ r ≠ h ∧ r ≠ s ∧ r ≠ e ∧
           o ≠ z ∧ o ≠ h ∧ o ≠ s ∧ o ≠ e ∧
           z ≠ h ∧ z ≠ s ∧ z ≠ e ∧
           h ≠ s ∧ h ≠ e ∧
           s ≠ e
  valid_digits : m < 10 ∧ y < 10 ∧ r < 10 ∧ o < 10 ∧ z < 10 ∧ h < 10 ∧ s < 10 ∧ e < 10
  s_greater_than_one : s > 1

/-- The equation MY + ROZH = SEEM holds for the given digit mapping -/
def equation_holds (d : DigitMapping) : Prop :=
  10 * d.m + d.y + 1000 * d.r + 100 * d.o + 10 * d.z + d.h = 1000 * d.s + 100 * d.e + 10 * d.e + d.m

/-- There exists a valid digit mapping for which the equation holds -/
def exists_valid_mapping : Prop :=
  ∃ d : DigitMapping, equation_holds d

/-- 2003 is the smallest four-digit number SEEM for which there exists a valid mapping -/
theorem smallest_seem : (∃ d : DigitMapping, d.s = 2 ∧ d.e = 0 ∧ d.m = 3 ∧ equation_holds d) ∧
  (∀ n : Nat, n < 2003 → ¬∃ d : DigitMapping, 1000 * d.s + 100 * d.e + 10 * d.e + d.m = n ∧ equation_holds d) :=
sorry

end NUMINAMATH_CALUDE_smallest_seem_l3218_321831


namespace NUMINAMATH_CALUDE_square_of_larger_number_l3218_321837

theorem square_of_larger_number (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : x^2 = 1156 := by
  sorry

end NUMINAMATH_CALUDE_square_of_larger_number_l3218_321837


namespace NUMINAMATH_CALUDE_max_player_salary_l3218_321855

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (max_team_salary : ℕ) :
  num_players = 23 →
  min_salary = 17000 →
  max_team_salary = 800000 →
  ∃ (max_single_salary : ℕ),
    max_single_salary = 426000 ∧
    (num_players - 1) * min_salary + max_single_salary = max_team_salary ∧
    ∀ (alternative_salary : ℕ),
      (num_players - 1) * min_salary + alternative_salary ≤ max_team_salary →
      alternative_salary ≤ max_single_salary :=
by
  sorry

end NUMINAMATH_CALUDE_max_player_salary_l3218_321855


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3218_321844

theorem solve_linear_equation :
  ∃ x : ℝ, x + 1 = 3 ∧ x = 2 := by
sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3218_321844


namespace NUMINAMATH_CALUDE_remainder_divisibility_l3218_321870

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 20) → (∃ m : ℤ, N = 13 * m + 7) := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l3218_321870


namespace NUMINAMATH_CALUDE_charity_show_girls_l3218_321835

theorem charity_show_girls (initial_total : ℕ) (initial_girls : ℕ) : 
  initial_girls = initial_total / 2 →
  (initial_girls - 3 : ℚ) / (initial_total + 1 : ℚ) = 2/5 →
  initial_girls = 17 := by
sorry

end NUMINAMATH_CALUDE_charity_show_girls_l3218_321835


namespace NUMINAMATH_CALUDE_even_times_odd_is_even_l3218_321896

/-- An integer is even if it's divisible by 2 -/
def IsEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

/-- An integer is odd if it's not divisible by 2 -/
def IsOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

/-- The product of an even integer and an odd integer is always even -/
theorem even_times_odd_is_even (a b : ℤ) (ha : IsEven a) (hb : IsOdd b) : IsEven (a * b) := by
  sorry


end NUMINAMATH_CALUDE_even_times_odd_is_even_l3218_321896


namespace NUMINAMATH_CALUDE_symmetry_probability_l3218_321865

/-- Represents a point in a 2D grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents a square with a grid of points -/
structure GridSquare where
  size : Nat
  points : List GridPoint

/-- Checks if a line through two points is a symmetry line for the square -/
def isSymmetryLine (square : GridSquare) (p q : GridPoint) : Bool :=
  sorry

/-- Counts the number of points that form symmetry lines with a given point -/
def countSymmetryPoints (square : GridSquare) (p : GridPoint) : Nat :=
  sorry

theorem symmetry_probability (square : GridSquare) (p : GridPoint) :
  square.size = 7 ∧
  square.points.length = 49 ∧
  p = ⟨3, 4⟩ →
  (countSymmetryPoints square p : Rat) / (square.points.length - 1 : Rat) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_probability_l3218_321865


namespace NUMINAMATH_CALUDE_sabina_college_loan_l3218_321843

theorem sabina_college_loan (college_cost savings grant_percentage : ℝ) : 
  college_cost = 30000 →
  savings = 10000 →
  grant_percentage = 0.4 →
  let remainder := college_cost - savings
  let grant_amount := grant_percentage * remainder
  let loan_amount := remainder - grant_amount
  loan_amount = 12000 := by
sorry

end NUMINAMATH_CALUDE_sabina_college_loan_l3218_321843


namespace NUMINAMATH_CALUDE_optimal_purchase_solution_max_basketballs_part2_l3218_321860

def basketball_price : ℕ := 100
def soccer_ball_price : ℕ := 80
def total_budget : ℕ := 5600
def total_items : ℕ := 60

theorem optimal_purchase_solution :
  ∃! (basketballs soccer_balls : ℕ),
    basketballs + soccer_balls = total_items ∧
    basketball_price * basketballs + soccer_ball_price * soccer_balls = total_budget ∧
    basketballs = 40 ∧
    soccer_balls = 20 :=
by sorry

theorem max_basketballs_part2 (new_budget : ℕ) (new_total_items : ℕ)
  (h1 : new_budget = 6890) (h2 : new_total_items = 80) :
  ∃ (max_basketballs : ℕ),
    max_basketballs ≤ new_total_items ∧
    basketball_price * max_basketballs + soccer_ball_price * (new_total_items - max_basketballs) ≤ new_budget ∧
    ∀ (basketballs : ℕ),
      basketballs ≤ new_total_items →
      basketball_price * basketballs + soccer_ball_price * (new_total_items - basketballs) ≤ new_budget →
      basketballs ≤ max_basketballs ∧
    max_basketballs = 24 :=
by sorry

end NUMINAMATH_CALUDE_optimal_purchase_solution_max_basketballs_part2_l3218_321860


namespace NUMINAMATH_CALUDE_not_perfect_square_with_mostly_fives_l3218_321836

/-- A function that checks if a list of digits represents a number with all but at most one digit being 5 -/
def allButOneAre5 (digits : List Nat) : Prop :=
  digits.length = 1000 ∧ (digits.filter (· ≠ 5)).length ≤ 1

/-- The theorem stating that a number with 1000 digits, all but at most one being 5, is not a perfect square -/
theorem not_perfect_square_with_mostly_fives (digits : List Nat) (h : allButOneAre5 digits) :
    ¬∃ (n : Nat), n * n = digits.foldl (fun acc d => acc * 10 + d) 0 := by
  sorry


end NUMINAMATH_CALUDE_not_perfect_square_with_mostly_fives_l3218_321836


namespace NUMINAMATH_CALUDE_max_a_value_l3218_321883

/-- The function f(x) = x^2 + 2ax - 1 -/
def f (a : ℝ) (x : ℝ) := x^2 + 2*a*x - 1

/-- The theorem stating the maximum value of a -/
theorem max_a_value :
  ∀ a : ℝ,
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ici 1 → x₂ ∈ Set.Ici 1 → x₁ < x₂ →
    x₂ * (f a x₁) - x₁ * (f a x₂) < a * (x₁ - x₂)) ↔
  a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l3218_321883


namespace NUMINAMATH_CALUDE_hot_dogs_leftover_l3218_321816

theorem hot_dogs_leftover : 36159782 % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_leftover_l3218_321816


namespace NUMINAMATH_CALUDE_drive_problem_solution_correct_l3218_321827

/-- Represents the problem of Sarah's drive to the conference center. -/
structure DriveProblem where
  initial_speed : ℝ  -- Speed in miles per hour
  initial_distance : ℝ  -- Distance covered in the first hour
  speed_increase : ℝ  -- Increase in speed for the rest of the journey
  late_time : ℝ  -- Time in hours Sarah would be late if continuing at initial speed
  early_time : ℝ  -- Time in hours Sarah arrives early with increased speed

/-- The solution to the drive problem. -/
def solve_drive_problem (p : DriveProblem) : ℝ :=
  p.initial_distance

/-- Theorem stating that the solution to the drive problem is correct. -/
theorem drive_problem_solution_correct (p : DriveProblem) 
  (h1 : p.initial_speed = 40)
  (h2 : p.initial_distance = 40)
  (h3 : p.speed_increase = 20)
  (h4 : p.late_time = 0.75)
  (h5 : p.early_time = 0.25) :
  solve_drive_problem p = 40 := by sorry

#check drive_problem_solution_correct

end NUMINAMATH_CALUDE_drive_problem_solution_correct_l3218_321827


namespace NUMINAMATH_CALUDE_pyramid_height_proof_l3218_321825

/-- The height of a square-based pyramid with base edge length 10 units,
    given that its volume is equal to the volume of a cube with edge length 5 units. -/
def pyramid_height : ℝ := 3.75

theorem pyramid_height_proof :
  let cube_edge := 5
  let cube_volume := cube_edge ^ 3
  let pyramid_base_edge := 10
  let pyramid_base_area := pyramid_base_edge ^ 2
  pyramid_height = (3 * cube_volume) / pyramid_base_area :=
by sorry

end NUMINAMATH_CALUDE_pyramid_height_proof_l3218_321825


namespace NUMINAMATH_CALUDE_james_passenger_count_l3218_321811

/-- Calculate the total number of passengers James has seen --/
def total_passengers (total_vehicles : ℕ) (num_trucks : ℕ) (num_buses : ℕ) (num_cars : ℕ)
  (truck_occupancy : ℕ) (bus_occupancy : ℕ) (taxi_occupancy : ℕ) (motorbike_occupancy : ℕ) (car_occupancy : ℕ) : ℕ :=
  let num_taxis := 2 * num_buses
  let num_motorbikes := total_vehicles - (num_trucks + num_buses + num_taxis + num_cars)
  (num_trucks * truck_occupancy) + (num_buses * bus_occupancy) + (num_taxis * taxi_occupancy) +
  (num_motorbikes * motorbike_occupancy) + (num_cars * car_occupancy)

theorem james_passenger_count :
  total_passengers 52 12 2 30 2 15 2 1 3 = 156 := by
  sorry

end NUMINAMATH_CALUDE_james_passenger_count_l3218_321811


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3218_321830

/-- A hyperbola with foci F₁ and F₂, and endpoints of conjugate axis B₁ and B₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  B₁ : ℝ × ℝ
  B₂ : ℝ × ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The angle B₂F₁B₁ in a hyperbola -/
def angle_B₂F₁B₁ (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity (h : Hyperbola) :
  angle_B₂F₁B₁ h = π/3 → eccentricity h = Real.sqrt 6 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3218_321830


namespace NUMINAMATH_CALUDE_percentage_increase_l3218_321845

theorem percentage_increase (x : ℝ) (h1 : x = 90.4) (h2 : ∃ p, x = 80 * (1 + p / 100)) : 
  ∃ p, x = 80 * (1 + p / 100) ∧ p = 13 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l3218_321845


namespace NUMINAMATH_CALUDE_population_growth_rate_l3218_321846

theorem population_growth_rate (initial_population : ℝ) (final_population : ℝ) (second_year_decrease : ℝ) :
  initial_population = 20000 →
  final_population = 18750 →
  second_year_decrease = 0.25 →
  ∃ first_year_increase : ℝ,
    first_year_increase = 0.25 ∧
    final_population = initial_population * (1 + first_year_increase) * (1 - second_year_decrease) :=
by sorry

end NUMINAMATH_CALUDE_population_growth_rate_l3218_321846


namespace NUMINAMATH_CALUDE_simplify_expression_l3218_321881

theorem simplify_expression (m n : ℝ) (h : m^2 + 3*m*n = 5) :
  5*m^2 - 3*m*n - (-9*m*n + 3*m^2) = 10 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l3218_321881


namespace NUMINAMATH_CALUDE_ribbon_distribution_l3218_321893

/-- Given total ribbon, number of gifts, and leftover ribbon, calculate ribbon per gift --/
def ribbon_per_gift (total_ribbon : ℕ) (num_gifts : ℕ) (leftover : ℕ) : ℚ :=
  (total_ribbon - leftover : ℚ) / num_gifts

theorem ribbon_distribution (total_ribbon num_gifts leftover : ℕ) 
  (h1 : total_ribbon = 18)
  (h2 : num_gifts = 6)
  (h3 : leftover = 6)
  (h4 : num_gifts > 0) :
  ribbon_per_gift total_ribbon num_gifts leftover = 2 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_distribution_l3218_321893


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_for_f_geq_x_minus_m_l3218_321804

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- Theorem for part I
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -2 ≤ x ∧ x ≤ 4/3} :=
sorry

-- Theorem for part II
theorem range_of_m_for_f_geq_x_minus_m :
  {m : ℝ | ∀ x, f x ≥ x - m} = {m : ℝ | m ≥ -3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_for_f_geq_x_minus_m_l3218_321804


namespace NUMINAMATH_CALUDE_parallelogram_with_right_angle_is_rectangle_l3218_321849

-- Define a parallelogram
structure Parallelogram :=
  (has_parallel_sides : Bool)

-- Define a rectangle
structure Rectangle extends Parallelogram :=
  (has_right_angle : Bool)

-- Theorem statement
theorem parallelogram_with_right_angle_is_rectangle 
  (p : Parallelogram) (h : Bool) : 
  (p.has_parallel_sides ∧ h) ↔ ∃ (r : Rectangle), r.has_right_angle ∧ r.has_parallel_sides = p.has_parallel_sides :=
sorry

end NUMINAMATH_CALUDE_parallelogram_with_right_angle_is_rectangle_l3218_321849


namespace NUMINAMATH_CALUDE_no_solution_exists_l3218_321872

theorem no_solution_exists : ¬∃ (a b : ℕ+), 
  a * b + 52 = 20 * Nat.lcm a b + 15 * Nat.gcd a b := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3218_321872


namespace NUMINAMATH_CALUDE_square_area_ratio_l3218_321876

theorem square_area_ratio : 
  let a : ℝ := 36
  let b : ℝ := 42
  let c : ℝ := 54
  (a^2 + b^2) / c^2 = 255 / 243 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3218_321876


namespace NUMINAMATH_CALUDE_opposite_reciprocal_sum_l3218_321880

theorem opposite_reciprocal_sum (a b c d : ℝ) (m : ℕ) : 
  b ≠ 0 →
  a = -b →
  c * d = 1 →
  m < 2 →
  (m : ℝ) - c * d + (a + b) / 2023 + a / b = -2 ∨ 
  (m : ℝ) - c * d + (a + b) / 2023 + a / b = -1 :=
by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_sum_l3218_321880


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l3218_321834

def choose (n k : Nat) : Nat :=
  if k > n then 0
  else (List.range k).foldl (fun m i => m * (n - i) / (i + 1)) 1

theorem sock_pair_combinations : 
  let total_socks : Nat := 18
  let white_socks : Nat := 8
  let brown_socks : Nat := 6
  let blue_socks : Nat := 4
  choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_combinations_l3218_321834


namespace NUMINAMATH_CALUDE_f_of_g_5_l3218_321873

def g (x : ℝ) : ℝ := 3 * x - 4

def f (x : ℝ) : ℝ := 2 * x + 5

theorem f_of_g_5 : f (g 5) = 27 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_5_l3218_321873


namespace NUMINAMATH_CALUDE_benjamin_walks_95_miles_l3218_321800

/-- Represents the total miles Benjamin walks in a week -/
def total_miles_walked : ℕ :=
  let work_distance := 6
  let dog_walk_distance := 2
  let friend_distance := 1
  let store_distance := 3
  let work_days := 5
  let dog_walks_per_day := 2
  let days_in_week := 7
  let store_visits := 2
  let friend_visits := 1

  (2 * work_distance * work_days) + 
  (dog_walk_distance * dog_walks_per_day * days_in_week) + 
  (2 * store_distance * store_visits) + 
  (2 * friend_distance * friend_visits)

theorem benjamin_walks_95_miles : total_miles_walked = 95 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_walks_95_miles_l3218_321800


namespace NUMINAMATH_CALUDE_base6_sum_is_6_l3218_321894

/-- Represents a single digit in base 6 -/
def Base6Digit := Fin 6

/-- The addition problem in base 6 -/
def base6_addition (X Y : Base6Digit) : Prop :=
  ∃ (carry : Nat),
    (3 * 6^2 + X.val * 6 + Y.val) + 24 = 
    6 * 6^2 + carry * 6 + X.val

/-- The main theorem to prove -/
theorem base6_sum_is_6 :
  ∀ X Y : Base6Digit,
    base6_addition X Y →
    (X.val : ℕ) + (Y.val : ℕ) = 6 := by sorry

end NUMINAMATH_CALUDE_base6_sum_is_6_l3218_321894


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3218_321862

theorem polynomial_division_remainder (x : ℂ) : 
  (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3218_321862


namespace NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l3218_321886

theorem right_triangle_increase_sides_acute (a b c x : ℝ) :
  a > 0 → b > 0 → c > 0 → x > 0 →
  c^2 = a^2 + b^2 →  -- right-angled triangle condition
  (a + x)^2 + (b + x)^2 > (c + x)^2  -- acute triangle condition
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_increase_sides_acute_l3218_321886


namespace NUMINAMATH_CALUDE_always_even_l3218_321864

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def change_sign (n : ℕ) (k : ℕ) : ℤ :=
  (sum_to_n n : ℤ) - 2 * k

theorem always_even (n : ℕ) (k : ℕ) (h1 : n = 1995) (h2 : k ≤ n) :
  Even (change_sign n k) := by
  sorry

end NUMINAMATH_CALUDE_always_even_l3218_321864


namespace NUMINAMATH_CALUDE_M_divisible_by_51_l3218_321833

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Definition of the function that concatenates numbers from 1 to n
  sorry

theorem M_divisible_by_51 :
  ∃ M : ℕ, M = concatenate_numbers 50 ∧ M % 51 = 0 :=
sorry

end NUMINAMATH_CALUDE_M_divisible_by_51_l3218_321833


namespace NUMINAMATH_CALUDE_min_a_value_for_common_points_l3218_321817

/-- Given two curves C₁ and C₂, where C₁ is y = ax² (a > 0) and C₂ is y = eˣ, 
    if they have common points in (0, +∞), then the minimum value of a is e²/4 -/
theorem min_a_value_for_common_points (a : ℝ) (h1 : a > 0) :
  (∃ x : ℝ, x > 0 ∧ a * x^2 = Real.exp x) → a ≥ Real.exp 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_a_value_for_common_points_l3218_321817


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3218_321887

theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, 1 < x ∧ x < 2 → Real.log x < 1) ∧
  (∃ x : ℝ, Real.log x < 1 ∧ ¬(1 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l3218_321887


namespace NUMINAMATH_CALUDE_sqrt_sqrt_81_l3218_321885

theorem sqrt_sqrt_81 : Real.sqrt (Real.sqrt 81) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sqrt_81_l3218_321885


namespace NUMINAMATH_CALUDE_function_inequality_l3218_321850

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the condition that f''(x) > f(x) for all x ∈ ℝ
variable (h : ∀ x : ℝ, (deriv (deriv f)) x > f x)

-- State the theorem to be proved
theorem function_inequality : f (Real.log 2015) > 2015 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l3218_321850


namespace NUMINAMATH_CALUDE_library_books_fraction_l3218_321820

theorem library_books_fraction (total : ℕ) (sold : ℕ) (h1 : total = 9900) (h2 : sold = 3300) :
  (total - sold : ℚ) / total = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_library_books_fraction_l3218_321820


namespace NUMINAMATH_CALUDE_coin_combination_difference_l3218_321858

/-- Represents the denominations of coins available -/
inductive Coin : Type
  | five : Coin
  | ten : Coin
  | twenty : Coin

/-- The value of a coin in cents -/
def coinValue : Coin → Nat
  | Coin.five => 5
  | Coin.ten => 10
  | Coin.twenty => 20

/-- A combination of coins -/
def CoinCombination := List Coin

/-- The total value of a coin combination in cents -/
def combinationValue (combo : CoinCombination) : Nat :=
  combo.map coinValue |>.sum

/-- Predicate for valid coin combinations that sum to 30 cents -/
def isValidCombination (combo : CoinCombination) : Prop :=
  combinationValue combo = 30

/-- The number of coins in a combination -/
def coinCount (combo : CoinCombination) : Nat :=
  combo.length

theorem coin_combination_difference :
  ∃ (minCombo maxCombo : CoinCombination),
    isValidCombination minCombo ∧
    isValidCombination maxCombo ∧
    (∀ c : CoinCombination, isValidCombination c → 
      coinCount c ≥ coinCount minCombo ∧
      coinCount c ≤ coinCount maxCombo) ∧
    coinCount maxCombo - coinCount minCombo = 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_combination_difference_l3218_321858


namespace NUMINAMATH_CALUDE_cousins_age_sum_l3218_321892

theorem cousins_age_sum (ages : Fin 5 → ℕ) 
  (mean_condition : (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 = 10)
  (median_condition : ages 2 = 7)
  (sorted : ∀ i j, i ≤ j → ages i ≤ ages j) :
  ages 0 + ages 4 = 29 := by
sorry

end NUMINAMATH_CALUDE_cousins_age_sum_l3218_321892


namespace NUMINAMATH_CALUDE_ellipse_problem_l3218_321838

noncomputable section

def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def FocalLength (c : ℝ) := 2 * Real.sqrt 3

def Eccentricity (e : ℝ) := Real.sqrt 2 / 2

def RightFocus (F : ℝ × ℝ) := F.1 > 0 ∧ F.2 = 0

def VectorDot (v w : ℝ × ℝ) := v.1 * w.1 + v.2 * w.2

def LineIntersection (k : ℝ) (N : Set (ℝ × ℝ)) := 
  {p : ℝ × ℝ | p.2 = k * (p.1 - 2) ∧ p ∈ N}

def VectorLength (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)

theorem ellipse_problem (a b c : ℝ) (C : Set (ℝ × ℝ)) (F B : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  C = Ellipse a b ∧
  FocalLength c = 2 * Real.sqrt 3 ∧
  Eccentricity (c / a) = Real.sqrt 2 / 2 ∧
  RightFocus F ∧
  B = (0, b) →
  (∃ A ∈ C, VectorDot (A.1 - B.1, A.2 - B.2) (F.1 - B.1, F.2 - B.2) = -6 →
    (∃ O r, (∀ p, p ∈ {q | (q.1 - O.1)^2 + (q.2 - O.2)^2 = r^2} ↔ 
      (p = A ∨ p = B ∨ p = F)) ∧
      (O = (0, 0) ∧ r = Real.sqrt 3 ∨
       O = (2 * Real.sqrt 3 / 3, 2 * Real.sqrt 3 / 3) ∧ r = Real.sqrt 15 / 3))) ∧
  (∀ k G H, G ∈ LineIntersection k (Ellipse a b) ∧ 
            H ∈ LineIntersection k (Ellipse a b) ∧ 
            G ≠ H ∧
            VectorLength (H.1 - G.1, H.2 - G.2) < 2 * Real.sqrt 5 / 3 →
    (-Real.sqrt 2 / 2 < k ∧ k < -1/2) ∨ (1/2 < k ∧ k < Real.sqrt 2 / 2)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_problem_l3218_321838


namespace NUMINAMATH_CALUDE_min_value_of_fraction_l3218_321898

theorem min_value_of_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 + y^2) / (x + y)^2 ≥ (1 : ℝ) / 2 ∧
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (a^2 + b^2) / (a + b)^2 = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_fraction_l3218_321898


namespace NUMINAMATH_CALUDE_quadratic_equation_1_quadratic_equation_2_quadratic_equation_3_quadratic_equation_4_l3218_321879

-- Problem 1
theorem quadratic_equation_1 (x : ℝ) : 
  (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3) → x^2 - 2*x - 2 = 0 := by sorry

-- Problem 2
theorem quadratic_equation_2 (x : ℝ) :
  (x = -4 ∨ x = 1) → (x + 4)^2 = 5*(x + 4) := by sorry

-- Problem 3
theorem quadratic_equation_3 (x : ℝ) :
  (x = (-3 + 2*Real.sqrt 6) / 3 ∨ x = (-3 - 2*Real.sqrt 6) / 3) → 3*x^2 + 6*x - 5 = 0 := by sorry

-- Problem 4
theorem quadratic_equation_4 (x : ℝ) :
  (x = (-1 + Real.sqrt 5) / 4 ∨ x = (-1 - Real.sqrt 5) / 4) → 4*x^2 + 2*x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_quadratic_equation_2_quadratic_equation_3_quadratic_equation_4_l3218_321879


namespace NUMINAMATH_CALUDE_jellybean_problem_l3218_321810

theorem jellybean_problem :
  ∃ (n : ℕ), n ≥ 150 ∧ n % 17 = 15 ∧ ∀ (m : ℕ), m ≥ 150 ∧ m % 17 = 15 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l3218_321810


namespace NUMINAMATH_CALUDE_binomial_sum_expectation_variance_l3218_321828

/-- A random variable X following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  X : ℝ

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV n p) : ℝ := n * p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV n p) : ℝ := n * p * (1 - p)

theorem binomial_sum_expectation_variance
  (X : BinomialRV 10 0.6) (Y : ℝ) 
  (h₁ : X.X + Y = 10) :
  expectation X = 6 ∧ 
  variance X = 2.4 ∧ 
  expectation X + Y = 10 → 
  Y = 4 ∧ variance X = 2.4 :=
sorry

end NUMINAMATH_CALUDE_binomial_sum_expectation_variance_l3218_321828


namespace NUMINAMATH_CALUDE_square_area_tripled_side_l3218_321839

theorem square_area_tripled_side (s : ℝ) (h : s > 0) :
  (3 * s)^2 = 9 * s^2 :=
by sorry

end NUMINAMATH_CALUDE_square_area_tripled_side_l3218_321839


namespace NUMINAMATH_CALUDE_new_dwelling_points_order_l3218_321807

open Real

-- Define the "new dwelling point" for each function
def α : ℝ := 1

-- β is implicitly defined by the equation ln(β+1) = 1/(β+1)
def β : ℝ := sorry

-- γ is implicitly defined by the equation cos γ = -sin γ, where γ ∈ (π/2, π)
noncomputable def γ : ℝ := sorry

axiom β_eq : log (β + 1) = 1 / (β + 1)
axiom γ_eq : cos γ = -sin γ
axiom γ_range : π / 2 < γ ∧ γ < π

-- Theorem statement
theorem new_dwelling_points_order : γ > α ∧ α > β := by sorry

end NUMINAMATH_CALUDE_new_dwelling_points_order_l3218_321807


namespace NUMINAMATH_CALUDE_equation_solution_l3218_321890

theorem equation_solution : ∃ x : ℝ, (2 / (x - 4) + 3 = (x - 2) / (4 - x)) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3218_321890


namespace NUMINAMATH_CALUDE_intersection_range_l3218_321869

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line
def line (x y m : ℝ) : Prop := y = 4*x + m

-- Define symmetry with respect to the line
def symmetric_points (x1 y1 x2 y2 m : ℝ) : Prop :=
  line ((x1 + x2)/2) ((y1 + y2)/2) m

-- Theorem statement
theorem intersection_range (m : ℝ) : 
  (∃ x1 y1 x2 y2 : ℝ, 
    x1 ≠ x2 ∧ 
    ellipse x1 y1 ∧ 
    ellipse x2 y2 ∧ 
    line x1 y1 m ∧ 
    line x2 y2 m ∧ 
    symmetric_points x1 y1 x2 y2 m) ↔ 
  -2 * Real.sqrt 13 / 13 < m ∧ m < 2 * Real.sqrt 13 / 13 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l3218_321869


namespace NUMINAMATH_CALUDE_xyz_mod_seven_l3218_321852

theorem xyz_mod_seven (x y z : ℕ) 
  (h_x : x < 7) (h_y : y < 7) (h_z : z < 7)
  (h1 : (x + 3*y + 2*z) % 7 = 0)
  (h2 : (3*x + 2*y + z) % 7 = 2)
  (h3 : (2*x + y + 3*z) % 7 = 3) :
  (x * y * z) % 7 = 1 := by
sorry

end NUMINAMATH_CALUDE_xyz_mod_seven_l3218_321852


namespace NUMINAMATH_CALUDE_solution_completeness_l3218_321877

def is_integer (q : ℚ) : Prop := ∃ n : ℤ, q = n

def satisfies_conditions (x y z : ℚ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x ≤ y ∧ y ≤ z ∧
  is_integer (x + y + z) ∧
  is_integer (1/x + 1/y + 1/z) ∧
  is_integer (x * y * z)

def solution_set : Set (ℚ × ℚ × ℚ) :=
  {(1, 1, 1), (1, 2, 2), (2, 3, 6), (2, 4, 4), (3, 3, 3)}

theorem solution_completeness :
  ∀ x y z : ℚ, satisfies_conditions x y z ↔ (x, y, z) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_solution_completeness_l3218_321877


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l3218_321868

theorem wire_cutting_problem (shorter_piece longer_piece total_length : ℝ) :
  shorter_piece = 40 →
  shorter_piece = (2 / 5) * longer_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 140 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l3218_321868


namespace NUMINAMATH_CALUDE_total_cost_price_l3218_321806

/-- Represents the cost and selling information for a fruit --/
structure Fruit where
  sellingPrice : ℚ
  lossRatio : ℚ

/-- Calculates the cost price of a fruit given its selling price and loss ratio --/
def costPrice (fruit : Fruit) : ℚ :=
  fruit.sellingPrice / (1 - fruit.lossRatio)

/-- The apple sold in the shop --/
def apple : Fruit := { sellingPrice := 30, lossRatio := 1/5 }

/-- The orange sold in the shop --/
def orange : Fruit := { sellingPrice := 45, lossRatio := 1/4 }

/-- The banana sold in the shop --/
def banana : Fruit := { sellingPrice := 15, lossRatio := 1/6 }

/-- Theorem stating the total cost price of all three fruits --/
theorem total_cost_price :
  costPrice apple + costPrice orange + costPrice banana = 115.5 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_price_l3218_321806


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3218_321824

theorem complex_fraction_simplification (N : ℕ) (h : N = 2^16) :
  (65533^3 + 65534^3 + 65535^3 + 65536^3 + 65537^3 + 65538^3 + 65539^3) / 
  (32765 * 32766 + 32767 * 32768 + 32768 * 32769 + 32770 * 32771 : ℕ) = 7 * N :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3218_321824


namespace NUMINAMATH_CALUDE_restaurant_expenditure_l3218_321866

theorem restaurant_expenditure (num_people : ℕ) (regular_cost : ℚ) (num_regular : ℕ) (extra_cost : ℚ) :
  num_people = 7 →
  regular_cost = 11 →
  num_regular = 6 →
  extra_cost = 6 →
  let total_regular := num_regular * regular_cost
  let average := (total_regular + (total_regular + extra_cost) / num_people) / num_people
  let total_cost := total_regular + (average + extra_cost)
  total_cost = 84 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_expenditure_l3218_321866


namespace NUMINAMATH_CALUDE_max_value_of_reciprocal_sum_l3218_321818

theorem max_value_of_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hax : a^x = 3) (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∀ x' y' a' b' : ℝ, a' > 1 → b' > 1 → a'^x' = 3 → b'^y' = 3 → a' + b' = 2 * Real.sqrt 3 → 
    1/x' + 1/y' ≤ 1/x + 1/y) ∧ 1/x + 1/y = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_reciprocal_sum_l3218_321818


namespace NUMINAMATH_CALUDE_border_area_l3218_321891

/-- The area of the border around a rectangular painting -/
theorem border_area (height width border_width : ℕ) : 
  height = 12 → width = 16 → border_width = 3 →
  (height + 2 * border_width) * (width + 2 * border_width) - height * width = 204 := by
  sorry

end NUMINAMATH_CALUDE_border_area_l3218_321891
