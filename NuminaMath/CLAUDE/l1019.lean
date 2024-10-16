import Mathlib

namespace NUMINAMATH_CALUDE_specific_box_surface_area_l1019_101979

/-- Calculates the surface area of an open box formed by removing squares from corners of a rectangle --/
def open_box_surface_area (length width corner_size : ℕ) : ℕ :=
  let original_area := length * width
  let corner_area := corner_size * corner_size
  let total_removed_area := 4 * corner_area
  original_area - total_removed_area

/-- Theorem stating that the surface area of the specific open box is 1379 square units --/
theorem specific_box_surface_area :
  open_box_surface_area 45 35 7 = 1379 :=
by sorry

end NUMINAMATH_CALUDE_specific_box_surface_area_l1019_101979


namespace NUMINAMATH_CALUDE_relationship_abc_l1019_101976

theorem relationship_abc (a b c : ℝ) : 
  a = Real.sqrt 2 → b = 2^(0.8 : ℝ) → c = 2 * Real.log 2 / Real.log 5 → c < a ∧ a < b := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1019_101976


namespace NUMINAMATH_CALUDE_max_abs_z_on_circle_l1019_101946

theorem max_abs_z_on_circle (z : ℂ) (h : Complex.abs (z - Complex.I * 2) = 1) :
  ∃ (z_max : ℂ), Complex.abs z_max = 3 ∧ 
  ∀ (w : ℂ), Complex.abs (w - Complex.I * 2) = 1 → Complex.abs w ≤ Complex.abs z_max :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_on_circle_l1019_101946


namespace NUMINAMATH_CALUDE_frac_less_one_necessary_not_sufficient_l1019_101915

theorem frac_less_one_necessary_not_sufficient (x : ℝ) :
  (∀ x, x > 1 → 1 / x < 1) ∧ 
  (∃ x, 1 / x < 1 ∧ ¬(x > 1)) :=
sorry

end NUMINAMATH_CALUDE_frac_less_one_necessary_not_sufficient_l1019_101915


namespace NUMINAMATH_CALUDE_matrix_pattern_l1019_101931

/-- Given a 2x2 matrix [[a, 2], [5, 6]] where a is unknown, 
    if (5 * 6) = (a * 2) * 3, then a = 5 -/
theorem matrix_pattern (a : ℝ) : (5 * 6 : ℝ) = (a * 2) * 3 → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_matrix_pattern_l1019_101931


namespace NUMINAMATH_CALUDE_weights_representation_l1019_101943

def weights : List ℤ := [1, 3, 9, 27]

def is_representable (n : ℤ) : Prop :=
  ∃ (a b c d : ℤ), 
    (a ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (b ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (c ∈ ({-1, 0, 1} : Set ℤ)) ∧
    (d ∈ ({-1, 0, 1} : Set ℤ)) ∧
    n = 27*a + 9*b + 3*c + d

theorem weights_representation :
  ∀ n : ℤ, 0 ≤ n → n < 41 → is_representable n :=
by sorry

end NUMINAMATH_CALUDE_weights_representation_l1019_101943


namespace NUMINAMATH_CALUDE_prove_a_value_l1019_101918

theorem prove_a_value (A B : Set ℤ) (a : ℤ) : 
  A = {0, 1} → 
  B = {-1, 0, a+3} → 
  A ⊆ B → 
  a = -2 := by sorry

end NUMINAMATH_CALUDE_prove_a_value_l1019_101918


namespace NUMINAMATH_CALUDE_jake_arrival_delay_l1019_101999

/-- Represents the problem of Austin and Jake descending a building --/
structure DescentProblem where
  floors : ℕ               -- Number of floors to descend
  steps_per_floor : ℕ      -- Number of steps per floor
  jake_speed : ℕ           -- Jake's speed in steps per second
  elevator_time : ℕ        -- Time taken by elevator in seconds

/-- Calculates the time difference between Jake's arrival and the elevator's arrival --/
def time_difference (p : DescentProblem) : ℤ :=
  let total_steps := p.floors * p.steps_per_floor
  let jake_time := total_steps / p.jake_speed
  jake_time - p.elevator_time

/-- The main theorem stating that Jake will arrive 20 seconds after the elevator --/
theorem jake_arrival_delay (p : DescentProblem) 
  (h1 : p.floors = 8)
  (h2 : p.steps_per_floor = 30)
  (h3 : p.jake_speed = 3)
  (h4 : p.elevator_time = 60) : 
  time_difference p = 20 := by
  sorry

#eval time_difference ⟨8, 30, 3, 60⟩

end NUMINAMATH_CALUDE_jake_arrival_delay_l1019_101999


namespace NUMINAMATH_CALUDE_total_borrowed_by_lunchtime_l1019_101927

/-- Represents the number of books on a shelf at different times of the day -/
structure ShelfState where
  initial : ℕ
  added : ℕ
  borrowed_morning : ℕ
  borrowed_afternoon : ℕ
  remaining : ℕ

/-- Calculates the number of books borrowed by lunchtime for a given shelf -/
def borrowed_by_lunchtime (shelf : ShelfState) : ℕ :=
  shelf.initial + shelf.added - (shelf.remaining + shelf.borrowed_afternoon)

/-- The state of shelf A -/
def shelf_a : ShelfState := {
  initial := 100,
  added := 40,
  borrowed_morning := 0,  -- Unknown, to be calculated
  borrowed_afternoon := 30,
  remaining := 60
}

/-- The state of shelf B -/
def shelf_b : ShelfState := {
  initial := 150,
  added := 20,
  borrowed_morning := 50,
  borrowed_afternoon := 0,  -- Not needed for the calculation
  remaining := 80
}

/-- The state of shelf C -/
def shelf_c : ShelfState := {
  initial := 200,
  added := 10,
  borrowed_morning := 0,  -- Unknown, to be calculated
  borrowed_afternoon := 45,
  remaining := 200 + 10 - 130  -- 130 is total borrowed throughout the day
}

/-- Theorem stating that the total number of books borrowed by lunchtime across all shelves is 165 -/
theorem total_borrowed_by_lunchtime :
  borrowed_by_lunchtime shelf_a + borrowed_by_lunchtime shelf_b + borrowed_by_lunchtime shelf_c = 165 := by
  sorry

end NUMINAMATH_CALUDE_total_borrowed_by_lunchtime_l1019_101927


namespace NUMINAMATH_CALUDE_count_equal_f_is_501_l1019_101921

/-- f(n) denotes the number of 1's in the base-2 representation of n -/
def f (n : ℕ) : ℕ := sorry

/-- Counts the number of integers n between 1 and 2002 (inclusive) where f(n) = f(n+1) -/
def count_equal_f : ℕ := sorry

theorem count_equal_f_is_501 : count_equal_f = 501 := by sorry

end NUMINAMATH_CALUDE_count_equal_f_is_501_l1019_101921


namespace NUMINAMATH_CALUDE_prob_same_gender_specific_schools_l1019_101933

/-- Represents a school with a certain number of male and female teachers -/
structure School :=
  (male_count : ℕ)
  (female_count : ℕ)

/-- The probability of selecting two teachers of the same gender from two schools -/
def prob_same_gender (school_a school_b : School) : ℚ :=
  let total_combinations := school_a.male_count * school_b.male_count + 
                            school_a.female_count * school_b.female_count
  let total_selections := (school_a.male_count + school_a.female_count) * 
                          (school_b.male_count + school_b.female_count)
  total_combinations / total_selections

theorem prob_same_gender_specific_schools :
  let school_a : School := ⟨2, 1⟩
  let school_b : School := ⟨1, 2⟩
  prob_same_gender school_a school_b = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_gender_specific_schools_l1019_101933


namespace NUMINAMATH_CALUDE_parking_lot_ratio_l1019_101992

/-- Proves that the ratio of full-sized car spaces to compact car spaces is 11:4 
    given the total number of spaces and the number of full-sized car spaces. -/
theorem parking_lot_ratio (total_spaces full_sized_spaces : ℕ) 
  (h1 : total_spaces = 450)
  (h2 : full_sized_spaces = 330) :
  (full_sized_spaces : ℚ) / (total_spaces - full_sized_spaces : ℚ) = 11 / 4 := by
  sorry

#check parking_lot_ratio

end NUMINAMATH_CALUDE_parking_lot_ratio_l1019_101992


namespace NUMINAMATH_CALUDE_triangle_side_length_l1019_101981

/-- In a triangle XYZ, if ∠Z = 30°, ∠Y = 60°, and XZ = 12 units, then XY = 24 units. -/
theorem triangle_side_length (X Y Z : ℝ × ℝ) :
  let angle (A B C : ℝ × ℝ) : ℝ := sorry
  let distance (A B : ℝ × ℝ) : ℝ := sorry
  angle Z X Y = π / 6 →  -- 30°
  angle X Y Z = π / 3 →  -- 60°
  distance X Z = 12 →
  distance X Y = 24 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1019_101981


namespace NUMINAMATH_CALUDE_system_solution_l1019_101982

theorem system_solution (p q u v : ℝ) : 
  (p * u + q * v = 2 * (p^2 - q^2)) ∧ 
  (v / (p - q) - u / (p + q) = (p^2 + q^2) / (p * q)) →
  ((p * q * (p^2 - q^2) ≠ 0 ∧ q ≠ 1 + Real.sqrt 2 ∧ q ≠ 1 - Real.sqrt 2) →
    (u = (p^2 - q^2) / p ∧ v = (p^2 - q^2) / q)) ∧
  ((u ≠ 0 ∧ v ≠ 0 ∧ u^2 ≠ v^2) →
    (p = u * v^2 / (v^2 - u^2) ∧ q = u^2 * v / (v^2 - u^2))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1019_101982


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1019_101989

/-- Acme T-Shirt Company's pricing function -/
def acme_price (x : ℕ) : ℝ := 60 + 11 * x

/-- Gamma T-Shirt Company's pricing function -/
def gamma_price (x : ℕ) : ℝ := 20 + 15 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_acme_cheaper : ℕ := 11

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_acme_cheaper < gamma_price min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper → acme_price n ≥ gamma_price n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l1019_101989


namespace NUMINAMATH_CALUDE_unique_multiplication_707_l1019_101997

theorem unique_multiplication_707 : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    ∃ (a b : ℕ), n = 100 * a + 70 + b ∧ 
    707 * n = 124432 := by
  sorry

end NUMINAMATH_CALUDE_unique_multiplication_707_l1019_101997


namespace NUMINAMATH_CALUDE_fraction_invariance_l1019_101977

theorem fraction_invariance (x y : ℝ) (square : ℝ) :
  (2 * x * y) / (x^2 + square) = (2 * (3*x) * (3*y)) / ((3*x)^2 + square) →
  square = y^2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_invariance_l1019_101977


namespace NUMINAMATH_CALUDE_x_less_than_one_necessary_not_sufficient_l1019_101911

theorem x_less_than_one_necessary_not_sufficient :
  (∀ x : ℝ, Real.log x < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ Real.log x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_one_necessary_not_sufficient_l1019_101911


namespace NUMINAMATH_CALUDE_exists_n_for_digit_sum_ratio_l1019_101951

/-- S(a) denotes the sum of the digits of the natural number a -/
def digit_sum (a : ℕ) : ℕ := sorry

/-- Theorem stating that for any natural number R, there exists a natural number n 
    such that the ratio of the digit sum of n^2 to the digit sum of n equals R -/
theorem exists_n_for_digit_sum_ratio (R : ℕ) : 
  ∃ n : ℕ, (digit_sum (n^2) : ℚ) / (digit_sum n : ℚ) = R := by sorry

end NUMINAMATH_CALUDE_exists_n_for_digit_sum_ratio_l1019_101951


namespace NUMINAMATH_CALUDE_sphere_area_and_volume_l1019_101958

theorem sphere_area_and_volume (d : ℝ) (h : d = 6) : 
  let r := d / 2
  (4 * Real.pi * r^2 = 36 * Real.pi) ∧ 
  ((4 / 3) * Real.pi * r^3 = 36 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sphere_area_and_volume_l1019_101958


namespace NUMINAMATH_CALUDE_floor_sqrt_120_l1019_101953

theorem floor_sqrt_120 : ⌊Real.sqrt 120⌋ = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_120_l1019_101953


namespace NUMINAMATH_CALUDE_hotel_expenditure_l1019_101908

/-- The total expenditure of a group of men, where most spend a fixed amount and one spends more than the average -/
def total_expenditure (n : ℕ) (m : ℕ) (fixed_spend : ℚ) (extra_spend : ℚ) : ℚ :=
  let avg := (m * fixed_spend + ((m * fixed_spend + extra_spend) / n)) / n
  n * avg

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

theorem hotel_expenditure :
  round_to_nearest (total_expenditure 9 8 3 5) = 33 := by
  sorry

end NUMINAMATH_CALUDE_hotel_expenditure_l1019_101908


namespace NUMINAMATH_CALUDE_suit_price_theorem_l1019_101994

theorem suit_price_theorem (original_price : ℝ) : 
  (original_price * 1.25 * 0.75 = 150) → original_price = 160 := by
  sorry

end NUMINAMATH_CALUDE_suit_price_theorem_l1019_101994


namespace NUMINAMATH_CALUDE_tom_candy_left_l1019_101967

def initial_candy : ℕ := 2
def friend_candy : ℕ := 7
def bought_candy : ℕ := 10

def total_candy : ℕ := initial_candy + friend_candy + bought_candy

def candy_left : ℕ := total_candy - (total_candy / 2)

theorem tom_candy_left : candy_left = 10 := by sorry

end NUMINAMATH_CALUDE_tom_candy_left_l1019_101967


namespace NUMINAMATH_CALUDE_trapezoid_area_theorem_l1019_101929

/-- Represents a trapezoid with given diagonals and height -/
structure Trapezoid where
  diagonal1 : ℝ
  diagonal2 : ℝ
  height : ℝ

/-- Calculates the possible areas of a trapezoid given its diagonals and height -/
def trapezoid_areas (t : Trapezoid) : Set ℝ :=
  {900, 780}

/-- Theorem stating that a trapezoid with diagonals 17 and 113, and height 15 has an area of either 900 or 780 -/
theorem trapezoid_area_theorem (t : Trapezoid) 
    (h1 : t.diagonal1 = 17) 
    (h2 : t.diagonal2 = 113) 
    (h3 : t.height = 15) : 
  ∃ (area : ℝ), area ∈ trapezoid_areas t ∧ (area = 900 ∨ area = 780) := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_area_theorem_l1019_101929


namespace NUMINAMATH_CALUDE_arcsin_sin_2x_solutions_l1019_101938

theorem arcsin_sin_2x_solutions (x : Real) :
  x ∈ Set.Icc (-π/2) (π/2) ∧ Real.arcsin (Real.sin (2*x)) = x ↔ x = 0 ∨ x = -π/3 ∨ x = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sin_2x_solutions_l1019_101938


namespace NUMINAMATH_CALUDE_quadratic_symmetry_point_l1019_101904

/-- A quadratic function f(x) with a parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x + 2*(a-1)

theorem quadratic_symmetry_point (a : ℝ) :
  (∀ x ≤ 4, (f_derivative a x ≤ 0)) ∧
  (∀ x ≥ 4, (f_derivative a x ≥ 0)) →
  a = -3 := by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_point_l1019_101904


namespace NUMINAMATH_CALUDE_triangle_height_ratio_l1019_101948

theorem triangle_height_ratio (a b c : ℝ) (ha hb hc : a > 0 ∧ b > 0 ∧ c > 0) 
  (side_ratio : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  ∃ (h₁ h₂ h₃ : ℝ), h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ 
    (a * h₁ = b * h₂) ∧ (b * h₂ = c * h₃) ∧
    h₁ / 20 = h₂ / 15 ∧ h₂ / 15 = h₃ / 12 :=
sorry

end NUMINAMATH_CALUDE_triangle_height_ratio_l1019_101948


namespace NUMINAMATH_CALUDE_product_B_percentage_l1019_101924

theorem product_B_percentage (X : ℝ) : 
  X ≥ 0 → X ≤ 100 →
  ∃ (total : ℕ), total ≥ 100 ∧
  ∃ (A B both neither : ℕ),
    A + B + neither = total ∧
    both ≤ A ∧ both ≤ B ∧
    (X : ℝ) = (A : ℝ) / total * 100 ∧
    (23 : ℝ) = (both : ℝ) / total * 100 ∧
    (23 : ℝ) = (neither : ℝ) / total * 100 →
  (B : ℝ) / total * 100 = 100 - X :=
by sorry

end NUMINAMATH_CALUDE_product_B_percentage_l1019_101924


namespace NUMINAMATH_CALUDE_trebled_resultant_l1019_101988

theorem trebled_resultant (x : ℕ) : x = 20 → 3 * ((2 * x) + 5) = 135 := by
  sorry

end NUMINAMATH_CALUDE_trebled_resultant_l1019_101988


namespace NUMINAMATH_CALUDE_domain_of_g_l1019_101952

/-- The domain of f(x) -/
def DomainF : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 6}

/-- The function g(x) -/
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2 * x) / (x - 2)

/-- The domain of g(x) -/
def DomainG : Set ℝ := {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)}

/-- Theorem: The domain of g(x) is correct given the domain of f(x) -/
theorem domain_of_g (f : ℝ → ℝ) (hf : ∀ x, x ∈ DomainF → f x ≠ 0) :
  ∀ x, x ∈ DomainG ↔ (2 * x ∈ DomainF ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_domain_of_g_l1019_101952


namespace NUMINAMATH_CALUDE_product_expansion_l1019_101901

theorem product_expansion (x : ℝ) : 
  (x^2 - 3*x + 3) * (x^2 + x + 1) = x^4 - 2*x^3 + x^2 + 3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l1019_101901


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_proof_1008_smallest_l1019_101974

theorem smallest_four_digit_multiple_of_18 : ℕ → Prop :=
  fun n => (n ≥ 1000) ∧ (n < 10000) ∧ (n % 18 = 0) ∧
    ∀ m : ℕ, (m ≥ 1000) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m

theorem proof_1008_smallest : smallest_four_digit_multiple_of_18 1008 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_proof_1008_smallest_l1019_101974


namespace NUMINAMATH_CALUDE_middle_numbers_average_l1019_101980

theorem middle_numbers_average (a b c d : ℕ+) : 
  a < b ∧ b < c ∧ c < d ∧  -- Four different positive integers
  (a + b + c + d : ℚ) / 4 = 5 ∧  -- Average is 5
  ∀ w x y z : ℕ+, w < x ∧ x < y ∧ y < z ∧ (w + x + y + z : ℚ) / 4 = 5 → (z - w : ℤ) ≤ (d - a : ℤ) →  -- Maximum possible difference
  (b + c : ℚ) / 2 = 5/2 :=
sorry

end NUMINAMATH_CALUDE_middle_numbers_average_l1019_101980


namespace NUMINAMATH_CALUDE_cricket_bat_profit_percentage_cricket_bat_profit_is_twenty_percent_l1019_101925

/-- Calculates the profit percentage for seller A given the conditions of the cricket bat sale --/
theorem cricket_bat_profit_percentage 
  (cost_price_A : ℝ) 
  (profit_percentage_B : ℝ) 
  (selling_price_C : ℝ) : ℝ :=
  let selling_price_B := selling_price_C / (1 + profit_percentage_B)
  let profit_A := selling_price_B - cost_price_A
  let profit_percentage_A := (profit_A / cost_price_A) * 100
  
  profit_percentage_A

/-- The profit percentage for A when selling the cricket bat to B is 20% --/
theorem cricket_bat_profit_is_twenty_percent : 
  cricket_bat_profit_percentage 152 0.25 228 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bat_profit_percentage_cricket_bat_profit_is_twenty_percent_l1019_101925


namespace NUMINAMATH_CALUDE_fourth_root_difference_l1019_101964

theorem fourth_root_difference : (81 : ℝ) ^ (1/4) - (1296 : ℝ) ^ (1/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_difference_l1019_101964


namespace NUMINAMATH_CALUDE_expression_evaluation_l1019_101947

theorem expression_evaluation :
  let x : ℚ := -1/2
  (x - 3)^2 + (x + 3)*(x - 3) - 2*x*(x - 2) + 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1019_101947


namespace NUMINAMATH_CALUDE_gcd_59_power_l1019_101944

theorem gcd_59_power : Nat.gcd (59^7 + 1) (59^7 + 59^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_59_power_l1019_101944


namespace NUMINAMATH_CALUDE_tom_green_marbles_l1019_101919

/-- The number of green marbles Sara has -/
def sara_green : ℕ := 3

/-- The total number of green marbles Sara and Tom have together -/
def total_green : ℕ := 7

/-- The number of green marbles Tom has -/
def tom_green : ℕ := total_green - sara_green

theorem tom_green_marbles : tom_green = 4 := by
  sorry

end NUMINAMATH_CALUDE_tom_green_marbles_l1019_101919


namespace NUMINAMATH_CALUDE_set_c_forms_triangle_l1019_101926

/-- Triangle inequality theorem for a set of three line segments -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: The set of line segments (4, 5, 6) can form a triangle -/
theorem set_c_forms_triangle : satisfies_triangle_inequality 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_set_c_forms_triangle_l1019_101926


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l1019_101910

theorem two_numbers_with_given_means (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  Real.sqrt (a * b) = Real.sqrt 5 ∧
  (a + b) / 2 = 5 →
  (a = 5 + 2 * Real.sqrt 5 ∧ b = 5 - 2 * Real.sqrt 5) ∨
  (a = 5 - 2 * Real.sqrt 5 ∧ b = 5 + 2 * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l1019_101910


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1019_101907

/-- Given a parabola and a hyperbola with specific properties, prove that the parameter p of the parabola is equal to 1. -/
theorem parabola_hyperbola_intersection (p a b : ℝ) (x₀ y₀ : ℝ) : 
  p > 0 → a > 0 → b > 0 → x₀ ≠ 0 →
  y₀^2 = 2 * p * x₀ →  -- Point A satisfies parabola equation
  x₀^2 / a^2 - y₀^2 / b^2 = 1 →  -- Point A satisfies hyperbola equation
  y₀ = 2 * x₀ →  -- Point A is on the asymptote y = 2x
  (x₀ - 0)^2 + y₀^2 = p^4 →  -- Distance from A to parabola's axis of symmetry is p²
  (a^2 + b^2) / a^2 = 5 →  -- Eccentricity of hyperbola is √5
  p = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l1019_101907


namespace NUMINAMATH_CALUDE_perp_necessary_not_sufficient_l1019_101936

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the "within" relation for a line in a plane
variable (within : Line → Plane → Prop)

theorem perp_necessary_not_sufficient
  (l m : Line) (α : Plane)
  (h_diff : l ≠ m)
  (h_within : within m α) :
  (∀ x : Line, within x α → perp_line_plane l α → perp_line_line l x) ∧
  (∃ β : Plane, perp_line_line l m ∧ ¬perp_line_plane l β ∧ within m β) :=
sorry

end NUMINAMATH_CALUDE_perp_necessary_not_sufficient_l1019_101936


namespace NUMINAMATH_CALUDE_extreme_value_and_range_l1019_101991

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - a * x) / Real.exp x

theorem extreme_value_and_range :
  (∃ x : ℝ, ∀ y : ℝ, f 1 y ≥ f 1 x ∧ f 1 x = -1 / Real.exp 2) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ≥ 0 → f a x ≥ 1 - 2 * x) ↔ a ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_extreme_value_and_range_l1019_101991


namespace NUMINAMATH_CALUDE_division_equality_l1019_101900

theorem division_equality : 250 / (5 + 12 * 3^2) = 250 / 113 := by
  sorry

end NUMINAMATH_CALUDE_division_equality_l1019_101900


namespace NUMINAMATH_CALUDE_max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l1019_101917

/-- A segment of natural numbers -/
structure Segment where
  start : ℕ
  length : ℕ

/-- The probability of a number in a segment being divisible by 10 -/
def probabilityDivisibleBy10 (s : Segment) : ℚ :=
  (s.length.div 10) / s.length

theorem max_probability_divisible_by_10 :
  ∃ s : Segment, probabilityDivisibleBy10 s = 1 ∧
  ∀ t : Segment, probabilityDivisibleBy10 t ≤ 1 :=
sorry

theorem min_nonzero_probability_divisible_by_10 :
  ∃ s : Segment, probabilityDivisibleBy10 s = 1/19 ∧
  ∀ t : Segment, probabilityDivisibleBy10 t = 0 ∨ probabilityDivisibleBy10 t ≥ 1/19 :=
sorry

end NUMINAMATH_CALUDE_max_probability_divisible_by_10_min_nonzero_probability_divisible_by_10_l1019_101917


namespace NUMINAMATH_CALUDE_room_length_calculation_l1019_101978

theorem room_length_calculation (area width : ℝ) (h1 : area = 10) (h2 : width = 2) :
  area / width = 5 := by
  sorry

end NUMINAMATH_CALUDE_room_length_calculation_l1019_101978


namespace NUMINAMATH_CALUDE_caitlin_age_is_24_l1019_101966

/-- The age of Aunt Anna in years -/
def aunt_anna_age : ℕ := 45

/-- The age of Brianna in years -/
def brianna_age : ℕ := (2 * aunt_anna_age) / 3

/-- The age difference between Brianna and Caitlin in years -/
def age_difference : ℕ := 6

/-- The age of Caitlin in years -/
def caitlin_age : ℕ := brianna_age - age_difference

/-- Theorem stating Caitlin's age -/
theorem caitlin_age_is_24 : caitlin_age = 24 := by sorry

end NUMINAMATH_CALUDE_caitlin_age_is_24_l1019_101966


namespace NUMINAMATH_CALUDE_gcd_fx_x_l1019_101996

def f (x : ℤ) : ℤ := (3*x+4)*(5*x+6)*(11*x+9)*(x+7)

theorem gcd_fx_x (x : ℤ) (h : ∃ k : ℤ, x = 35622 * k) : 
  Nat.gcd (Int.natAbs (f x)) (Int.natAbs x) = 378 := by
  sorry

end NUMINAMATH_CALUDE_gcd_fx_x_l1019_101996


namespace NUMINAMATH_CALUDE_reciprocal_of_complex_l1019_101945

/-- The reciprocal of the complex number -3 + 4i is -0.12 - 0.16i -/
theorem reciprocal_of_complex (G : ℂ) : 
  G = -3 + 4*I → 1 / G = -0.12 - 0.16*I := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_complex_l1019_101945


namespace NUMINAMATH_CALUDE_james_final_amount_proof_l1019_101942

/-- The amount of money owned by James after paying off Lucas' debt -/
def james_final_amount : ℝ := 170

/-- The total amount owned by Lucas, James, and Ali -/
def total_amount : ℝ := 300

/-- The amount of Lucas' debt -/
def lucas_debt : ℝ := 25

/-- The difference between James' and Ali's initial amounts -/
def james_ali_difference : ℝ := 40

theorem james_final_amount_proof :
  ∃ (ali james lucas : ℝ),
    ali + james + lucas = total_amount ∧
    james = ali + james_ali_difference ∧
    lucas = -lucas_debt ∧
    james - (lucas_debt / 2) = james_final_amount :=
sorry

end NUMINAMATH_CALUDE_james_final_amount_proof_l1019_101942


namespace NUMINAMATH_CALUDE_bald_eagle_dive_time_l1019_101998

/-- The time it takes for the bald eagle to dive to the ground given the specified conditions -/
theorem bald_eagle_dive_time : 
  ∀ (v_eagle : ℝ) (v_falcon : ℝ) (t_falcon : ℝ) (distance : ℝ),
  v_eagle > 0 →
  v_falcon = 2 * v_eagle →
  t_falcon = 15 →
  distance > 0 →
  distance = v_eagle * (2 * t_falcon) →
  distance = v_falcon * t_falcon →
  2 * t_falcon = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_bald_eagle_dive_time_l1019_101998


namespace NUMINAMATH_CALUDE_possible_square_values_l1019_101972

/-- Represents a tiling of a 9x7 rectangle using L-trominoes and 2x2 squares. -/
structure Tiling :=
  (num_squares : ℕ)
  (num_trominoes : ℕ)

/-- The area of the rectangle is 63. -/
axiom rectangle_area : 63 = 9 * 7

/-- The area of a 2x2 square is 4. -/
axiom square_area : 4 = 2 * 2

/-- The area of an L-tromino is 3. -/
axiom tromino_area : 3 = 3

/-- The total area covered by tiles equals the rectangle area. -/
axiom area_equation (t : Tiling) : 4 * t.num_squares + 3 * t.num_trominoes = 63

/-- The number of 2x2 squares is a multiple of 3. -/
axiom squares_multiple_of_three (t : Tiling) : ∃ k : ℕ, t.num_squares = 3 * k

/-- The number of 2x2 squares is at most 3. -/
axiom max_squares (t : Tiling) : t.num_squares ≤ 3

/-- The possible values for the number of 2x2 squares are 0 and 3. -/
theorem possible_square_values (t : Tiling) : t.num_squares = 0 ∨ t.num_squares = 3 :=
sorry

end NUMINAMATH_CALUDE_possible_square_values_l1019_101972


namespace NUMINAMATH_CALUDE_sandbox_sand_calculation_l1019_101973

/-- Calculates the amount of sand needed to fill a square sandbox -/
theorem sandbox_sand_calculation (side_length : ℝ) (sand_weight_per_section : ℝ) (area_per_section : ℝ) :
  side_length = 40 →
  sand_weight_per_section = 30 →
  area_per_section = 80 →
  (side_length ^ 2 / area_per_section) * sand_weight_per_section = 600 :=
by sorry

end NUMINAMATH_CALUDE_sandbox_sand_calculation_l1019_101973


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l1019_101956

theorem quadratic_roots_difference (R : ℝ) : 
  (∃ α β : ℝ, α^2 - 2*α - R = 0 ∧ β^2 - 2*β - R = 0 ∧ α - β = 12) → R = 35 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l1019_101956


namespace NUMINAMATH_CALUDE_smallest_coefficient_value_l1019_101928

-- Define the ratio condition
def ratio_condition (n : ℕ) : Prop :=
  (6^n) / (2^n) = 729

-- Define the function to get the coefficient of the term with the smallest coefficient
def smallest_coefficient (n : ℕ) : ℤ :=
  (-1)^(n - 3) * (Nat.choose n 3)

-- Theorem statement
theorem smallest_coefficient_value :
  ∃ n : ℕ, ratio_condition n ∧ smallest_coefficient n = -20 := by
  sorry

end NUMINAMATH_CALUDE_smallest_coefficient_value_l1019_101928


namespace NUMINAMATH_CALUDE_remaining_soup_feeds_20_adults_l1019_101923

/-- Represents the number of adults a can of soup can feed -/
def adults_per_can : ℕ := 4

/-- Represents the number of children a can of soup can feed -/
def children_per_can : ℕ := 6

/-- Represents the total number of cans of soup -/
def total_cans : ℕ := 8

/-- Represents the number of children fed -/
def children_fed : ℕ := 20

/-- Represents the fraction of soup left in a can after feeding children -/
def leftover_fraction : ℚ := 1/3

/-- Calculates the number of adults that can be fed with the remaining soup -/
def adults_fed (adults_per_can : ℕ) (children_per_can : ℕ) (total_cans : ℕ) (children_fed : ℕ) (leftover_fraction : ℚ) : ℕ :=
  sorry

/-- Theorem stating that the remaining soup can feed 20 adults -/
theorem remaining_soup_feeds_20_adults : 
  adults_fed adults_per_can children_per_can total_cans children_fed leftover_fraction = 20 :=
sorry

end NUMINAMATH_CALUDE_remaining_soup_feeds_20_adults_l1019_101923


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l1019_101920

theorem student_average_greater_than_true_average (x y z : ℝ) (h : x < z ∧ z < y) :
  (x + z) / 2 / 2 + y / 2 > (x + y + z) / 3 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l1019_101920


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1019_101993

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x^2 + 2 * x - 1 = 0
def equation2 (x : ℝ) : Prop := (x + 2) * (x - 1) = 2 - 2 * x

-- Theorem for the first equation
theorem solutions_equation1 : 
  ∃ x₁ x₂ : ℝ, x₁ = -1 ∧ x₂ = 1/3 ∧ equation1 x₁ ∧ equation1 x₂ ∧ 
  ∀ x : ℝ, equation1 x → x = x₁ ∨ x = x₂ := by sorry

-- Theorem for the second equation
theorem solutions_equation2 : 
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = -4 ∧ equation2 x₁ ∧ equation2 x₂ ∧ 
  ∀ x : ℝ, equation2 x → x = x₁ ∨ x = x₂ := by sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1019_101993


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1019_101903

theorem complex_equation_solution :
  ∃ z : ℂ, 4 + 2 * Complex.I * z = 3 - 5 * Complex.I * z ∧ z = Complex.I / 7 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1019_101903


namespace NUMINAMATH_CALUDE_odd_sum_of_squares_l1019_101955

theorem odd_sum_of_squares (n m : ℤ) (h : Odd (n^2 + m^2)) :
  ¬(Even n ∧ Even m) ∧ ¬(Even (n + m)) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_squares_l1019_101955


namespace NUMINAMATH_CALUDE_range_of_a_when_A_union_B_equals_A_l1019_101906

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x < 3-a}

-- State the theorem
theorem range_of_a_when_A_union_B_equals_A :
  ∀ a : ℝ, (A ∪ B a = A) → a ≥ (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_a_when_A_union_B_equals_A_l1019_101906


namespace NUMINAMATH_CALUDE_f_at_five_l1019_101950

def f (x : ℝ) : ℝ := 3 * x^4 - 22 * x^3 + 51 * x^2 - 58 * x + 24

theorem f_at_five : f 5 = 134 := by sorry

end NUMINAMATH_CALUDE_f_at_five_l1019_101950


namespace NUMINAMATH_CALUDE_probability_less_than_three_l1019_101968

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  topRight : ℝ × ℝ

/-- The probability that a randomly chosen point in the square satisfies a given condition --/
def probability (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

/-- The square with vertices (0,0), (0,2), (2,2), and (2,0) --/
def unitSquare : Square :=
  { bottomLeft := (0, 0), topRight := (2, 2) }

/-- The condition x + y < 3 --/
def lessThanThree (p : ℝ × ℝ) : Prop :=
  p.1 + p.2 < 3

theorem probability_less_than_three :
  probability unitSquare lessThanThree = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_three_l1019_101968


namespace NUMINAMATH_CALUDE_interval_intersection_l1019_101987

theorem interval_intersection (x : ℝ) : 
  (|4 - x| < 5 ∧ x^2 < 36) ↔ (-1 < x ∧ x < 6) := by
  sorry

end NUMINAMATH_CALUDE_interval_intersection_l1019_101987


namespace NUMINAMATH_CALUDE_remaining_chess_pieces_l1019_101957

/-- Represents the number of chess pieces a player has lost -/
structure LostPieces where
  pawns : ℕ
  knights : ℕ
  bishops : ℕ
  rooks : ℕ
  queens : ℕ

/-- Calculates the total number of pieces lost by a player -/
def totalLost (lost : LostPieces) : ℕ :=
  lost.pawns + lost.knights + lost.bishops + lost.rooks + lost.queens

/-- Theorem: The total remaining chess pieces is 22 -/
theorem remaining_chess_pieces
  (initial_pieces : ℕ)
  (kennedy_lost : LostPieces)
  (riley_lost : LostPieces)
  (h1 : initial_pieces = 16)
  (h2 : kennedy_lost = { pawns := 4, knights := 1, bishops := 2, rooks := 0, queens := 0 })
  (h3 : riley_lost = { pawns := 1, knights := 0, bishops := 0, rooks := 1, queens := 1 })
  : 2 * initial_pieces - (totalLost kennedy_lost + totalLost riley_lost) = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_chess_pieces_l1019_101957


namespace NUMINAMATH_CALUDE_ratio_of_divisors_sums_l1019_101914

def P : ℕ := 45 * 45 * 98 * 480

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry

def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_of_odd_divisors P) * 126 = sum_of_even_divisors P := by sorry

end NUMINAMATH_CALUDE_ratio_of_divisors_sums_l1019_101914


namespace NUMINAMATH_CALUDE_farthest_poles_distance_l1019_101983

/-- The number of utility poles -/
def num_poles : ℕ := 45

/-- The interval between each pole in meters -/
def interval : ℕ := 60

/-- The distance between the first and last pole in kilometers -/
def distance : ℚ := 2.64

theorem farthest_poles_distance :
  (((num_poles - 1) * interval) : ℚ) / 1000 = distance := by sorry

end NUMINAMATH_CALUDE_farthest_poles_distance_l1019_101983


namespace NUMINAMATH_CALUDE_obtuse_triangle_third_side_range_l1019_101965

/-- A triangle with side lengths a, b, and c is obtuse if and only if
    one of its squared side lengths is greater than the sum of the squares of the other two side lengths. -/
def IsObtuse (a b c : ℝ) : Prop :=
  a^2 > b^2 + c^2 ∨ b^2 > a^2 + c^2 ∨ c^2 > a^2 + b^2

/-- The range of the third side length x in an obtuse triangle with side lengths 2, 3, and x. -/
theorem obtuse_triangle_third_side_range :
  ∀ x : ℝ, x > 0 →
    (IsObtuse 2 3 x ↔ (1 < x ∧ x < Real.sqrt 5) ∨ (Real.sqrt 13 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_third_side_range_l1019_101965


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l1019_101963

-- Define the triangle ABC
theorem triangle_abc_theorem (a b c A B C : ℝ) 
  (h1 : a / Real.tan A = b / (2 * Real.sin B))
  (h2 : a = 6)
  (h3 : b = 2 * c)
  (h4 : 0 < A ∧ A < π)
  (h5 : 0 < B ∧ B < π)
  (h6 : 0 < C ∧ C < π)
  (h7 : A + B + C = π) : 
  A = π / 3 ∧ 
  (1/2 * b * c * Real.sin A : ℝ) = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l1019_101963


namespace NUMINAMATH_CALUDE_small_hotdogs_count_l1019_101934

theorem small_hotdogs_count (total : ℕ) (large : ℕ) (h1 : total = 79) (h2 : large = 21) :
  total - large = 58 := by
  sorry

end NUMINAMATH_CALUDE_small_hotdogs_count_l1019_101934


namespace NUMINAMATH_CALUDE_apple_theorem_l1019_101995

def apple_problem (initial_apples : ℕ) : ℕ :=
  let after_jill := initial_apples - (initial_apples * 30 / 100)
  let after_june := after_jill - (after_jill * 20 / 100)
  let after_friend := after_june - 2
  after_friend - (after_friend * 10 / 100)

theorem apple_theorem :
  apple_problem 150 = 74 := by sorry

end NUMINAMATH_CALUDE_apple_theorem_l1019_101995


namespace NUMINAMATH_CALUDE_space_geometry_statements_l1019_101954

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (lineParallelPlane : Line → Plane → Prop)
variable (linePerpendicularPlane : Line → Plane → Prop)
variable (planeParallelPlane : Plane → Plane → Prop)
variable (planePerpendicularPlane : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)

-- Define the theorem
theorem space_geometry_statements 
  (m n : Line) (α β : Plane) (A : Point) :
  (∀ l₁ l₂ p, parallel l₁ l₂ → lineParallelPlane l₂ p → lineParallelPlane l₁ p) ∧
  (parallel m n → linePerpendicularPlane n β → lineParallelPlane m α → planePerpendicularPlane α β) ∧
  (intersect m n → lineParallelPlane m α → lineParallelPlane m β → 
   lineParallelPlane n α → lineParallelPlane n β → planeParallelPlane α β) :=
by sorry

end NUMINAMATH_CALUDE_space_geometry_statements_l1019_101954


namespace NUMINAMATH_CALUDE_normal_dist_symmetry_l1019_101959

-- Define a random variable with normal distribution
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P (ξ : normal_dist 0 σ) (event : Set ℝ) : ℝ := sorry

-- Theorem statement
theorem normal_dist_symmetry (σ : ℝ) (ξ : normal_dist 0 σ) :
  P ξ {x | x < 2} = 0.8 → P ξ {x | x < -2} = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_normal_dist_symmetry_l1019_101959


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1019_101984

theorem sum_of_squares_of_roots (a b α β : ℝ) : 
  (∀ x, (x - a) * (x - b) = 1 ↔ x = α ∨ x = β) →
  (∃ x₁ x₂, (x₁ - α) * (x₁ - β) = -1 ∧ (x₂ - α) * (x₂ - β) = -1 ∧ x₁ ≠ x₂) →
  x₁^2 + x₂^2 = a^2 + b^2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1019_101984


namespace NUMINAMATH_CALUDE_trigonometric_product_equality_l1019_101939

theorem trigonometric_product_equality : 
  (1 - 1 / Real.cos (30 * π / 180)) * 
  (1 + 2 / Real.sin (60 * π / 180)) * 
  (1 - 1 / Real.sin (30 * π / 180)) * 
  (1 + 2 / Real.cos (60 * π / 180)) = 
  (25 - 10 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_trigonometric_product_equality_l1019_101939


namespace NUMINAMATH_CALUDE_parallelogram_height_l1019_101912

theorem parallelogram_height (area base height : ℝ) : 
  area = 612 ∧ base = 34 ∧ area = base * height → height = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1019_101912


namespace NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1019_101930

/-- Given two vectors a and b in ℝ³, prove that k = -2 when (k * a + b) is perpendicular to a. -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ × ℝ) (k : ℝ) : 
  a = (1, 1, 1) → 
  b = (1, 2, 3) → 
  (k * a.1 + b.1, k * a.2.1 + b.2.1, k * a.2.2 + b.2.2) • a = 0 → 
  k = -2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vector_scalar_l1019_101930


namespace NUMINAMATH_CALUDE_unique_phone_number_l1019_101909

-- Define the set of available digits
def available_digits : Finset Nat := {2, 3, 4, 5, 6, 7, 8}

-- Define a function to check if a list of digits is valid
def valid_phone_number (digits : List Nat) : Prop :=
  digits.length = 7 ∧
  digits.toFinset = available_digits ∧
  digits.Sorted (·<·)

-- Theorem statement
theorem unique_phone_number :
  ∃! digits : List Nat, valid_phone_number digits :=
sorry

end NUMINAMATH_CALUDE_unique_phone_number_l1019_101909


namespace NUMINAMATH_CALUDE_simplify_expression_l1019_101986

theorem simplify_expression (a : ℝ) : 3 * a^2 - a * (2 * a - 1) = a^2 + a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1019_101986


namespace NUMINAMATH_CALUDE_urn_operations_theorem_l1019_101975

/-- Represents the state of the urn with white and black marbles -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Represents the three possible operations on the urn -/
inductive Operation
  | removeBlackAddWhite
  | removeWhiteAddBoth
  | removeBothAddBoth

/-- Applies a single operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.removeBlackAddWhite => 
      ⟨state.white + 3, state.black - 2⟩
  | Operation.removeWhiteAddBoth => 
      ⟨state.white - 2, state.black + 1⟩
  | Operation.removeBothAddBoth => 
      ⟨state.white - 2, state.black⟩

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to the initial state -/
def applySequence (init : UrnState) (seq : OperationSequence) : UrnState :=
  seq.foldl applyOperation init

/-- The theorem to be proved -/
theorem urn_operations_theorem : 
  ∃ (seq : OperationSequence), 
    applySequence ⟨150, 50⟩ seq = ⟨148, 2⟩ :=
sorry

end NUMINAMATH_CALUDE_urn_operations_theorem_l1019_101975


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1019_101961

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1019_101961


namespace NUMINAMATH_CALUDE_pythagorean_triple_properties_l1019_101916

theorem pythagorean_triple_properties (a b c : ℕ) (h : a^2 + b^2 = c^2) :
  (Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1) ∧
  ((2 ∣ a ∧ 4 ∣ a) ∨ (2 ∣ b ∧ 4 ∣ b)) ∧
  (3 ∣ a ∨ 3 ∣ b) ∧
  (5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c) ∧
  (¬(2 ∣ c) ∧ c % 4 = 1 ∧ ¬(3 ∣ c) ∧ ¬(7 ∣ c) ∧ ¬(11 ∣ c)) :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_properties_l1019_101916


namespace NUMINAMATH_CALUDE_stamp_collection_ratio_l1019_101905

theorem stamp_collection_ratio : 
  ∀ (tom_original mike_gift harry_gift tom_final : ℕ),
    tom_original = 3000 →
    mike_gift = 17 →
    ∃ k : ℕ, harry_gift = k * mike_gift + 10 →
    tom_final = tom_original + mike_gift + harry_gift →
    tom_final = 3061 →
    harry_gift / mike_gift = 44 / 17 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_ratio_l1019_101905


namespace NUMINAMATH_CALUDE_smallest_b_value_l1019_101932

theorem smallest_b_value (a b : ℕ+) (h1 : a.val - b.val = 8) 
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 8) :
  ∀ k : ℕ+, k.val < b.val → ¬(∃ a' : ℕ+, a'.val - k.val = 8 ∧ 
    Nat.gcd ((a'.val^3 + k.val^3) / (a'.val + k.val)) (a'.val * k.val) = 8) :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l1019_101932


namespace NUMINAMATH_CALUDE_outer_circle_radius_l1019_101922

/-- Given a circular race track with an inner circumference of 440 meters and a width of 14 meters,
    the radius of the outer circle is equal to (440 / (2 * π)) + 14 meters. -/
theorem outer_circle_radius (inner_circumference : ℝ) (track_width : ℝ) 
    (h1 : inner_circumference = 440)
    (h2 : track_width = 14) : 
    (inner_circumference / (2 * Real.pi) + track_width) = (440 / (2 * Real.pi) + 14) := by
  sorry

#check outer_circle_radius

end NUMINAMATH_CALUDE_outer_circle_radius_l1019_101922


namespace NUMINAMATH_CALUDE_range_of_a_l1019_101941

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - 2*a)*x + 3*a else 2^(x - 1)

theorem range_of_a : 
  ∀ a : ℝ, (∀ x : ℝ, ∃ y : ℝ, f a x = y) ↔ (0 ≤ a ∧ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1019_101941


namespace NUMINAMATH_CALUDE_farm_work_hourly_rate_l1019_101962

theorem farm_work_hourly_rate 
  (total_amount : ℕ) 
  (tips : ℕ) 
  (hours_worked : ℕ) 
  (h1 : total_amount = 240)
  (h2 : tips = 50)
  (h3 : hours_worked = 19) :
  (total_amount - tips) / hours_worked = 10 := by
sorry

end NUMINAMATH_CALUDE_farm_work_hourly_rate_l1019_101962


namespace NUMINAMATH_CALUDE_ladder_problem_l1019_101940

theorem ladder_problem (ladder_length height base : ℝ) :
  ladder_length = 13 ∧ height = 12 ∧ ladder_length^2 = height^2 + base^2 → base = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l1019_101940


namespace NUMINAMATH_CALUDE_line_plane_relationships_l1019_101960

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relationships between lines and planes
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_planes (p1 p2 : Plane) : Prop := sorry
def perpendicular_lines (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem line_plane_relationships 
  (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) :
  ((perpendicular_line_plane m α ∧ 
    parallel_line_plane n β ∧ 
    parallel_planes α β) → 
   perpendicular_lines m n) ∧
  ((perpendicular_line_plane m α ∧ 
    perpendicular_line_plane n β ∧ 
    perpendicular_planes α β) → 
   perpendicular_lines m n) :=
by sorry

end NUMINAMATH_CALUDE_line_plane_relationships_l1019_101960


namespace NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l1019_101935

theorem necessary_and_sufficient_condition (a b : ℝ) : (a > b) ↔ (a * |a| > b * |b|) := by
  sorry

end NUMINAMATH_CALUDE_necessary_and_sufficient_condition_l1019_101935


namespace NUMINAMATH_CALUDE_inequality_proof_l1019_101937

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + b + c) * (a^2 + b^2 + c^2) ≤ 3 * (a^3 + b^3 + c^3) ∧
  a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1019_101937


namespace NUMINAMATH_CALUDE_kitten_puppy_difference_l1019_101985

theorem kitten_puppy_difference (kittens puppies : ℕ) : 
  kittens = 78 → puppies = 32 → kittens - 2 * puppies = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_kitten_puppy_difference_l1019_101985


namespace NUMINAMATH_CALUDE_total_weight_compounds_l1019_101969

/-- The atomic mass of Nitrogen in g/mol -/
def mass_N : ℝ := 14.01

/-- The atomic mass of Hydrogen in g/mol -/
def mass_H : ℝ := 1.01

/-- The atomic mass of Bromine in g/mol -/
def mass_Br : ℝ := 79.90

/-- The atomic mass of Magnesium in g/mol -/
def mass_Mg : ℝ := 24.31

/-- The atomic mass of Chlorine in g/mol -/
def mass_Cl : ℝ := 35.45

/-- The molar mass of Ammonium Bromide (NH4Br) in g/mol -/
def molar_mass_NH4Br : ℝ := mass_N + 4 * mass_H + mass_Br

/-- The molar mass of Magnesium Chloride (MgCl2) in g/mol -/
def molar_mass_MgCl2 : ℝ := mass_Mg + 2 * mass_Cl

/-- The number of moles of Ammonium Bromide -/
def moles_NH4Br : ℝ := 3.72

/-- The number of moles of Magnesium Chloride -/
def moles_MgCl2 : ℝ := 2.45

theorem total_weight_compounds : 
  moles_NH4Br * molar_mass_NH4Br + moles_MgCl2 * molar_mass_MgCl2 = 597.64 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_compounds_l1019_101969


namespace NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_7_l1019_101949

theorem remainder_11_pow_2023_mod_7 : 11^2023 % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_pow_2023_mod_7_l1019_101949


namespace NUMINAMATH_CALUDE_P_union_Q_eq_Q_l1019_101902

-- Define the sets P and Q
def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x^2 - x > 0}

-- State the theorem
theorem P_union_Q_eq_Q : P ∪ Q = Q := by
  sorry

end NUMINAMATH_CALUDE_P_union_Q_eq_Q_l1019_101902


namespace NUMINAMATH_CALUDE_intersection_parallel_perpendicular_line_l1019_101970

/-- The equation of a line passing through the intersection of two lines,
    parallel to one line, and perpendicular to another line. -/
theorem intersection_parallel_perpendicular_line 
  (l1 l2 l_parallel l_perpendicular : ℝ → ℝ → Prop) 
  (h_l1 : ∀ x y, l1 x y ↔ 2*x - 3*y + 10 = 0)
  (h_l2 : ∀ x y, l2 x y ↔ 3*x + 4*y - 2 = 0)
  (h_parallel : ∀ x y, l_parallel x y ↔ x - y + 1 = 0)
  (h_perpendicular : ∀ x y, l_perpendicular x y ↔ 3*x - y - 2 = 0)
  : ∃ l : ℝ → ℝ → Prop, 
    (∃ x y, l1 x y ∧ l2 x y ∧ l x y) ∧ 
    (∀ x y, l x y ↔ x - y + 4 = 0) ∧
    (∀ a b c d, l a b ∧ l c d → (c - a) * (1) + (-1) * (d - b) = 0) ∧
    (∀ a b c d, l a b ∧ l c d → (c - a) * (3) + (-1) * (d - b) = 0) :=
by sorry

end NUMINAMATH_CALUDE_intersection_parallel_perpendicular_line_l1019_101970


namespace NUMINAMATH_CALUDE_b_most_suitable_l1019_101990

/-- Represents a candidate in the competition -/
structure Candidate where
  name : String
  average_score : ℝ
  variance : ℝ

/-- The set of all candidates -/
def candidates : Set Candidate :=
  { ⟨"A", 92.5, 3.4⟩, ⟨"B", 92.5, 2.1⟩, ⟨"C", 92.5, 2.5⟩, ⟨"D", 92.5, 2.7⟩ }

/-- Definition of the most suitable candidate -/
def most_suitable (c : Candidate) : Prop :=
  c ∈ candidates ∧
  ∀ d ∈ candidates, c.variance ≤ d.variance

/-- Theorem stating that B is the most suitable candidate -/
theorem b_most_suitable :
  ∃ c ∈ candidates, c.name = "B" ∧ most_suitable c := by
  sorry

end NUMINAMATH_CALUDE_b_most_suitable_l1019_101990


namespace NUMINAMATH_CALUDE_total_spent_is_88_70_l1019_101913

-- Define the constants
def pizza_price : ℝ := 10
def pizza_quantity : ℕ := 5
def pizza_discount_threshold : ℕ := 3
def pizza_discount_rate : ℝ := 0.15

def soft_drink_price : ℝ := 1.5
def soft_drink_quantity : ℕ := 10

def hamburger_price : ℝ := 3
def hamburger_quantity : ℕ := 6
def hamburger_discount_threshold : ℕ := 5
def hamburger_discount_rate : ℝ := 0.1

-- Define the function to calculate the total spent
def total_spent : ℝ :=
  let robert_pizza_cost := 
    if pizza_quantity > pizza_discount_threshold
    then pizza_price * pizza_quantity * (1 - pizza_discount_rate)
    else pizza_price * pizza_quantity
  let robert_drinks_cost := soft_drink_price * soft_drink_quantity
  let teddy_hamburger_cost := 
    if hamburger_quantity > hamburger_discount_threshold
    then hamburger_price * hamburger_quantity * (1 - hamburger_discount_rate)
    else hamburger_price * hamburger_quantity
  let teddy_drinks_cost := soft_drink_price * soft_drink_quantity
  robert_pizza_cost + robert_drinks_cost + teddy_hamburger_cost + teddy_drinks_cost

-- Theorem statement
theorem total_spent_is_88_70 : total_spent = 88.70 := by sorry

end NUMINAMATH_CALUDE_total_spent_is_88_70_l1019_101913


namespace NUMINAMATH_CALUDE_cartesian_to_polar_conversion_l1019_101971

theorem cartesian_to_polar_conversion (x y ρ θ : ℝ) :
  x = -1 ∧ y = Real.sqrt 3 →
  ρ = 2 ∧ θ = 2 * Real.pi / 3 →
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ ρ^2 = x^2 + y^2 := by
  sorry

end NUMINAMATH_CALUDE_cartesian_to_polar_conversion_l1019_101971
