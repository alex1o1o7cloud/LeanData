import Mathlib

namespace NUMINAMATH_CALUDE_missing_sale_is_6088_l3526_352629

/-- Calculates the missing sale amount given the sales for five months and the average sale for six months. -/
def calculate_missing_sale (sale1 sale2 sale3 sale5 sale6 average_sale : ℕ) : ℕ :=
  6 * average_sale - (sale1 + sale2 + sale3 + sale5 + sale6)

/-- Theorem stating that the missing sale amount is 6088 given the specific sales and average. -/
theorem missing_sale_is_6088 :
  calculate_missing_sale 5921 5468 5568 6433 5922 5900 = 6088 := by
  sorry

#eval calculate_missing_sale 5921 5468 5568 6433 5922 5900

end NUMINAMATH_CALUDE_missing_sale_is_6088_l3526_352629


namespace NUMINAMATH_CALUDE_problem_solution_l3526_352671

theorem problem_solution : 
  (Real.sqrt 32 + 4 * Real.sqrt (1/2) - Real.sqrt 18 = 3 * Real.sqrt 2) ∧ 
  ((7 - 4 * Real.sqrt 3) * (7 + 4 * Real.sqrt 3) - (Real.sqrt 3 - 1)^2 + (1/3)⁻¹ = 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3526_352671


namespace NUMINAMATH_CALUDE_megans_files_l3526_352669

theorem megans_files (deleted_files : ℕ) (files_per_folder : ℕ) (num_folders : ℕ) :
  deleted_files = 21 →
  files_per_folder = 8 →
  num_folders = 9 →
  deleted_files + (files_per_folder * num_folders) = 93 :=
by sorry

end NUMINAMATH_CALUDE_megans_files_l3526_352669


namespace NUMINAMATH_CALUDE_milk_expense_l3526_352630

/-- Given Mr. Kishore's savings and expenses, prove the amount spent on milk -/
theorem milk_expense (savings : ℕ) (rent groceries education petrol misc : ℕ) 
  (h1 : savings = 2350)
  (h2 : rent = 5000)
  (h3 : groceries = 4500)
  (h4 : education = 2500)
  (h5 : petrol = 2000)
  (h6 : misc = 5650)
  (h7 : savings = (1 / 10 : ℚ) * (savings / (1 / 10 : ℚ))) :
  ∃ (milk : ℕ), milk = 1500 ∧ 
    (9 / 10 : ℚ) * (savings / (1 / 10 : ℚ)) = 
    (rent + groceries + education + petrol + misc + milk) :=
by sorry

end NUMINAMATH_CALUDE_milk_expense_l3526_352630


namespace NUMINAMATH_CALUDE_smallest_base_for_square_property_l3526_352655

theorem smallest_base_for_square_property : ∃ (b x y : ℕ), 
  (b ≥ 2) ∧ 
  (x < b) ∧ 
  (y < b) ∧ 
  (x ≠ 0) ∧ 
  (y ≠ 0) ∧ 
  ((x * b + x)^2 = y * b^3 + y * b^2 + y * b + y) ∧
  (∀ b' x' y' : ℕ, 
    (b' ≥ 2) ∧ 
    (x' < b') ∧ 
    (y' < b') ∧ 
    (x' ≠ 0) ∧ 
    (y' ≠ 0) ∧ 
    ((x' * b' + x')^2 = y' * b'^3 + y' * b'^2 + y' * b' + y') →
    (b ≤ b')) ∧
  (b = 7) ∧ 
  (x = 5) ∧ 
  (y = 4) := by
sorry

end NUMINAMATH_CALUDE_smallest_base_for_square_property_l3526_352655


namespace NUMINAMATH_CALUDE_unique_modular_congruence_l3526_352656

theorem unique_modular_congruence :
  ∃! n : ℤ, 0 ≤ n ∧ n < 31 ∧ -300 ≡ n [ZMOD 31] ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_congruence_l3526_352656


namespace NUMINAMATH_CALUDE_floor_length_percentage_l3526_352601

-- Define the parameters of the problem
def floor_length : ℝ := 18.9999683334125
def total_cost : ℝ := 361
def paint_rate : ℝ := 3.00001

-- Define the theorem
theorem floor_length_percentage (l b : ℝ) (h1 : l = floor_length) 
  (h2 : l * b = total_cost / paint_rate) : 
  (l - b) / b * 100 = 200 := by
  sorry

end NUMINAMATH_CALUDE_floor_length_percentage_l3526_352601


namespace NUMINAMATH_CALUDE_greatest_common_divisor_with_digit_sum_l3526_352662

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem greatest_common_divisor_with_digit_sum : 
  ∃ (n : ℕ), 
    n ∣ (6905 - 4665) ∧ 
    sum_of_digits n = 4 ∧ 
    ∀ (m : ℕ), m ∣ (6905 - 4665) ∧ sum_of_digits m = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_with_digit_sum_l3526_352662


namespace NUMINAMATH_CALUDE_smoking_lung_cancer_relationship_l3526_352691

-- Define the confidence level in the smoking-lung cancer relationship
def confidence_level : ℝ := 0.99

-- Define the probability of making a mistake in the conclusion
def error_probability : ℝ := 0.01

-- Define a sample size
def sample_size : ℕ := 100

-- Define a predicate for having lung cancer
def has_lung_cancer : (ℕ → Prop) := sorry

-- Define a predicate for being a smoker
def is_smoker : (ℕ → Prop) := sorry

-- Theorem stating that high confidence in the smoking-lung cancer relationship
-- does not preclude the possibility of a sample with no lung cancer cases
theorem smoking_lung_cancer_relationship 
  (h1 : confidence_level > 0.99) 
  (h2 : error_probability ≤ 0.01) :
  ∃ (sample : Finset ℕ), 
    (∀ i ∈ sample, is_smoker i) ∧ 
    (Finset.card sample = sample_size) ∧
    (∀ i ∈ sample, ¬has_lung_cancer i) := by
  sorry

end NUMINAMATH_CALUDE_smoking_lung_cancer_relationship_l3526_352691


namespace NUMINAMATH_CALUDE_expected_value_coin_flip_l3526_352673

/-- The expected value of winnings for a coin flip game -/
theorem expected_value_coin_flip :
  let p_heads : ℚ := 2 / 5
  let p_tails : ℚ := 3 / 5
  let win_heads : ℚ := 5
  let loss_tails : ℚ := 3
  p_heads * win_heads - p_tails * loss_tails = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_coin_flip_l3526_352673


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3526_352663

theorem complex_equation_solution (m : ℝ) (i : ℂ) : 
  i * i = -1 → (m + 2 * i) * (2 - i) = 4 + 3 * i → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3526_352663


namespace NUMINAMATH_CALUDE_prime_roots_range_l3526_352659

theorem prime_roots_range (p : ℕ) (h_prime : Nat.Prime p) 
  (h_roots : ∃ x y : ℤ, x^2 + p*x - 444*p = 0 ∧ y^2 + p*y - 444*p = 0) : 
  31 < p ∧ p ≤ 41 :=
sorry

end NUMINAMATH_CALUDE_prime_roots_range_l3526_352659


namespace NUMINAMATH_CALUDE_building_height_l3526_352641

/-- Prove that given a flagpole of height 18 meters casting a shadow of 45 meters,
    and a building casting a shadow of 70 meters under similar conditions,
    the height of the building is 28 meters. -/
theorem building_height (flagpole_height : ℝ) (flagpole_shadow : ℝ) (building_shadow : ℝ)
  (h1 : flagpole_height = 18)
  (h2 : flagpole_shadow = 45)
  (h3 : building_shadow = 70)
  : (flagpole_height / flagpole_shadow) * building_shadow = 28 :=
by sorry

end NUMINAMATH_CALUDE_building_height_l3526_352641


namespace NUMINAMATH_CALUDE_inequality_analysis_l3526_352603

theorem inequality_analysis (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hxy : x > y) (hz : z ≠ 0) :
  (∀ z, x + z > y + z) ∧
  (∀ z, x - z > y - z) ∧
  (∃ z, ¬(x * z > y * z)) ∧
  (∀ z, x / z^2 > y / z^2) ∧
  (∀ z, x * z^2 > y * z^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_analysis_l3526_352603


namespace NUMINAMATH_CALUDE_clock_hands_overlap_l3526_352638

/-- The angle traveled by the hour hand in one minute -/
def hour_hand_speed : ℝ := 0.5

/-- The angle traveled by the minute hand in one minute -/
def minute_hand_speed : ℝ := 6

/-- The initial angle of the hour hand at 4:10 -/
def initial_hour_angle : ℝ := 60 + 0.5 * 10

/-- The time in minutes after 4:10 when the hands overlap -/
def overlap_time : ℝ := 11

theorem clock_hands_overlap :
  ∃ (t : ℝ), t > 0 ∧ t ≤ overlap_time ∧
  initial_hour_angle + hour_hand_speed * t = minute_hand_speed * t :=
sorry

end NUMINAMATH_CALUDE_clock_hands_overlap_l3526_352638


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3526_352639

theorem polynomial_division_theorem (x : ℝ) : 
  (4*x^2 - 2*x + 3) * (2*x^2 + 5*x + 3) + (43*x + 36) = 8*x^4 + 16*x^3 - 7*x^2 + 4*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3526_352639


namespace NUMINAMATH_CALUDE_study_abroad_work_hours_l3526_352615

/-- Proves that working 28 hours per week for the remaining 10 weeks
    will meet the financial goal, given the initial plan and actual work done. -/
theorem study_abroad_work_hours
  (initial_hours_per_week : ℕ)
  (initial_weeks : ℕ)
  (goal_amount : ℕ)
  (actual_full_weeks : ℕ)
  (actual_reduced_weeks : ℕ)
  (reduced_hours_per_week : ℕ)
  (h_initial_hours : initial_hours_per_week = 25)
  (h_initial_weeks : initial_weeks = 15)
  (h_goal_amount : goal_amount = 4500)
  (h_actual_full_weeks : actual_full_weeks = 3)
  (h_actual_reduced_weeks : actual_reduced_weeks = 2)
  (h_reduced_hours : reduced_hours_per_week = 10)
  : ∃ (remaining_hours_per_week : ℕ),
    remaining_hours_per_week = 28 ∧
    (initial_hours_per_week * actual_full_weeks +
     reduced_hours_per_week * actual_reduced_weeks +
     remaining_hours_per_week * (initial_weeks - actual_full_weeks - actual_reduced_weeks)) *
    (goal_amount / (initial_hours_per_week * initial_weeks)) = goal_amount :=
by sorry

end NUMINAMATH_CALUDE_study_abroad_work_hours_l3526_352615


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l3526_352614

/-- A quadratic function passing through three specific points has a coefficient of 8/5 for its x² term. -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + c → 
    ((x = -3 ∧ y = 2) ∨ (x = 3 ∧ y = 2) ∨ (x = 2 ∧ y = -6))) → 
  a = 8/5 := by sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l3526_352614


namespace NUMINAMATH_CALUDE_system_properties_l3526_352692

-- Define the system of equations
def system (x y a : ℝ) : Prop :=
  (x + 3 * y = 4 - a) ∧ (x - y = 3 * a)

-- Theorem statement
theorem system_properties :
  ∀ (x y a : ℝ), system x y a →
    ((x + y = 0) → (a = -2)) ∧
    (x + 2 * y = 3) ∧
    (y = -x / 2 + 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_system_properties_l3526_352692


namespace NUMINAMATH_CALUDE_modular_congruence_13_pow_6_mod_11_l3526_352686

theorem modular_congruence_13_pow_6_mod_11 : 
  ∃ m : ℕ, 13^6 ≡ m [ZMOD 11] ∧ m < 11 → m = 9 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_13_pow_6_mod_11_l3526_352686


namespace NUMINAMATH_CALUDE_circle_symmetry_l3526_352617

/-- Given a circle with center (1,1) and radius √2, prove that if it's symmetric about the line y = kx + 3, then k = -2 -/
theorem circle_symmetry (k : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 2 → 
    ∃ x' y' : ℝ, (x' - 1)^2 + (y' - 1)^2 = 2 ∧ 
    ((x + x') / 2, (y + y') / 2) ∈ {(x, y) | y = k * x + 3}) →
  k = -2 :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3526_352617


namespace NUMINAMATH_CALUDE_injured_cats_count_l3526_352637

/-- The number of injured cats Jeff found on Tuesday -/
def injured_cats : ℕ :=
  let initial_cats : ℕ := 20
  let kittens_found : ℕ := 2
  let cats_adopted : ℕ := 3 * 2
  let final_cats : ℕ := 17
  final_cats - (initial_cats + kittens_found - cats_adopted)

theorem injured_cats_count : injured_cats = 1 := by
  sorry

end NUMINAMATH_CALUDE_injured_cats_count_l3526_352637


namespace NUMINAMATH_CALUDE_line_segment_parameter_sum_squares_l3526_352681

/-- Given a line segment connecting (1, -3) and (4, 9), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that a^2 + b^2 + c^2 + d^2 = 163 -/
theorem line_segment_parameter_sum_squares :
  ∀ (a b c d : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = 1 ∧ d = -3) →
  (a + b = 4 ∧ c + d = 9) →
  a^2 + b^2 + c^2 + d^2 = 163 := by
sorry

end NUMINAMATH_CALUDE_line_segment_parameter_sum_squares_l3526_352681


namespace NUMINAMATH_CALUDE_monotonic_at_most_one_zero_l3526_352688

/-- A function f: ℝ → ℝ is monotonic if for all x₁ < x₂, either f(x₁) ≤ f(x₂) or f(x₁) ≥ f(x₂) -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → (f x₁ ≤ f x₂ ∨ f x₁ ≥ f x₂)

/-- A real number x is a zero of f if f(x) = 0 -/
def IsZero (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

/-- The number of zeros of f is at most one -/
def AtMostOneZero (f : ℝ → ℝ) : Prop :=
  ∀ x y, IsZero f x → IsZero f y → x = y

theorem monotonic_at_most_one_zero (f : ℝ → ℝ) (h : Monotonic f) : AtMostOneZero f := by
  sorry

end NUMINAMATH_CALUDE_monotonic_at_most_one_zero_l3526_352688


namespace NUMINAMATH_CALUDE_sin_shift_equivalence_l3526_352610

theorem sin_shift_equivalence (x : ℝ) : 
  Real.sin (2 * x + Real.pi / 2) - 1 = Real.sin (2 * (x + Real.pi / 4)) - 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_shift_equivalence_l3526_352610


namespace NUMINAMATH_CALUDE_students_in_multiple_activities_l3526_352651

theorem students_in_multiple_activities 
  (total_students : ℕ) 
  (debate_only : ℕ) 
  (singing_only : ℕ) 
  (dance_only : ℕ) 
  (no_activity : ℕ) 
  (h1 : total_students = 55)
  (h2 : debate_only = 10)
  (h3 : singing_only = 18)
  (h4 : dance_only = 8)
  (h5 : no_activity = 5) :
  total_students - (debate_only + singing_only + dance_only + no_activity) = 14 := by
  sorry

end NUMINAMATH_CALUDE_students_in_multiple_activities_l3526_352651


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3526_352604

theorem inequality_solution_set (x : ℝ) : 
  (4 * x^2 - 3 * x > 5) ↔ (x < -5/4 ∨ x > 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3526_352604


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_l3526_352600

theorem imaginary_part_of_one_minus_i :
  Complex.im (1 - Complex.I) = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_l3526_352600


namespace NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l3526_352668

/-- A function f: ℝ → ℝ is odd -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ has period p -/
def HasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem min_zeros_odd_periodic_function
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_period : HasPeriod f 3)
  (h_f2 : f 2 = 0) :
  ∃ (S : Finset ℝ), S.card ≥ 7 ∧ (∀ x ∈ S, 0 < x ∧ x < 6 ∧ f x = 0) :=
sorry

end NUMINAMATH_CALUDE_min_zeros_odd_periodic_function_l3526_352668


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l3526_352626

-- Define the circular sector
def circular_sector (R : ℝ) (θ : ℝ) := {(x, y) : ℝ × ℝ | x^2 + y^2 ≤ R^2 ∧ 0 ≤ x ∧ y ≤ x * Real.tan θ}

-- Define the inscribed circle
def inscribed_circle (r : ℝ) (R : ℝ) (θ : ℝ) :=
  {(x, y) : ℝ × ℝ | (x - (R - r))^2 + (y - r)^2 = r^2}

-- Theorem statement
theorem inscribed_circle_radius :
  ∀ (R : ℝ), R > 0 →
  ∃ (r : ℝ), r > 0 ∧
  inscribed_circle r R (π/6) ⊆ circular_sector R (π/6) ∧
  r = 2 := by
sorry


end NUMINAMATH_CALUDE_inscribed_circle_radius_l3526_352626


namespace NUMINAMATH_CALUDE_fort_blocks_count_l3526_352618

/-- Represents the dimensions of a rectangular fort -/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed for a fort with given dimensions and wall thickness -/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) : ℕ :=
  d.length * d.width * d.height - 
  (d.length - 2 * wallThickness) * (d.width - 2 * wallThickness) * (d.height - wallThickness)

/-- Theorem stating that a fort with given dimensions requires 280 blocks -/
theorem fort_blocks_count : 
  blocksNeeded ⟨12, 10, 5⟩ 1 = 280 := by sorry

end NUMINAMATH_CALUDE_fort_blocks_count_l3526_352618


namespace NUMINAMATH_CALUDE_max_blocks_in_box_l3526_352680

/-- Represents the dimensions of a rectangular solid -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular solid given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- Represents the box dimensions -/
def box : Dimensions := ⟨3, 4, 2⟩

/-- Represents the block dimensions -/
def block : Dimensions := ⟨2, 1, 2⟩

/-- Theorem stating that the maximum number of blocks that can fit in the box is 6 -/
theorem max_blocks_in_box :
  ∃ (n : ℕ), n = 6 ∧ 
  n * volume block ≤ volume box ∧
  ∀ m : ℕ, m * volume block ≤ volume box → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_max_blocks_in_box_l3526_352680


namespace NUMINAMATH_CALUDE_inequalities_satisfied_l3526_352628

theorem inequalities_satisfied (a b c x y z : ℤ) 
  (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) : 
  (x^2*y + y^2*z + z^2*x ≤ a^2*b + b^2*c + c^2*a) ∧ 
  (x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2) ∧ 
  (x^2*y*z ≤ a^2*b*c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_satisfied_l3526_352628


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3526_352653

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 42*x + 400 ≤ 16 ↔ 16 ≤ x ∧ x ≤ 24 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3526_352653


namespace NUMINAMATH_CALUDE_jerry_bacon_strips_l3526_352609

/-- Represents the number of calories in Jerry's breakfast items and total breakfast --/
structure BreakfastCalories where
  pancakeCalories : ℕ
  baconCalories : ℕ
  cerealCalories : ℕ
  totalCalories : ℕ

/-- Calculates the number of bacon strips in Jerry's breakfast --/
def calculateBaconStrips (b : BreakfastCalories) : ℕ :=
  (b.totalCalories - (6 * b.pancakeCalories + b.cerealCalories)) / b.baconCalories

/-- Theorem stating that Jerry had 2 strips of bacon for breakfast --/
theorem jerry_bacon_strips :
  let b : BreakfastCalories := {
    pancakeCalories := 120,
    baconCalories := 100,
    cerealCalories := 200,
    totalCalories := 1120
  }
  calculateBaconStrips b = 2 := by
  sorry


end NUMINAMATH_CALUDE_jerry_bacon_strips_l3526_352609


namespace NUMINAMATH_CALUDE_xy_sum_is_two_l3526_352648

theorem xy_sum_is_two (x y : ℝ) 
  (hx : (x - 1)^3 + 1997*(x - 1) = -1) 
  (hy : (y - 1)^3 + 1997*(y - 1) = 1) : 
  x + y = 2 := by sorry

end NUMINAMATH_CALUDE_xy_sum_is_two_l3526_352648


namespace NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_perfect_square_l3526_352672

theorem consecutive_integers_product_plus_one_is_perfect_square (n : ℤ) :
  ∃ m : ℤ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_plus_one_is_perfect_square_l3526_352672


namespace NUMINAMATH_CALUDE_min_value_of_expression_l3526_352627

theorem min_value_of_expression (x y z : ℝ) : 
  (x*y - z)^2 + (x + y + z)^2 ≥ 0 ∧ 
  ∃ (a b c : ℝ), (a*b - c)^2 + (a + b + c)^2 = 0 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l3526_352627


namespace NUMINAMATH_CALUDE_herbert_age_next_year_l3526_352642

theorem herbert_age_next_year (kris_age : ℕ) (age_difference : ℕ) :
  kris_age = 24 →
  age_difference = 10 →
  kris_age - age_difference + 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_herbert_age_next_year_l3526_352642


namespace NUMINAMATH_CALUDE_prob_two_tails_proof_l3526_352661

/-- The probability of getting exactly 2 tails when tossing 3 fair coins -/
def prob_two_tails : ℚ := 3 / 8

/-- A fair coin has a probability of 1/2 for each outcome -/
def fair_coin (outcome : Bool) : ℚ := 1 / 2

/-- The number of possible outcomes when tossing 3 coins -/
def total_outcomes : ℕ := 2^3

/-- The number of outcomes with exactly 2 tails when tossing 3 coins -/
def favorable_outcomes : ℕ := 3

theorem prob_two_tails_proof :
  prob_two_tails = favorable_outcomes / total_outcomes :=
sorry

end NUMINAMATH_CALUDE_prob_two_tails_proof_l3526_352661


namespace NUMINAMATH_CALUDE_two_numbers_difference_l3526_352625

theorem two_numbers_difference (a b : ℕ) 
  (sum_eq : a + b = 24300)
  (b_divisible : 100 ∣ b)
  (b_div_100 : b / 100 = a) :
  b - a = 23760 :=
by sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l3526_352625


namespace NUMINAMATH_CALUDE_complex_sum_powers_l3526_352640

theorem complex_sum_powers (ζ₁ ζ₂ ζ₃ : ℂ) 
  (h1 : ζ₁ + ζ₂ + ζ₃ = 2)
  (h2 : ζ₁^2 + ζ₂^2 + ζ₃^2 = 5)
  (h3 : ζ₁^3 + ζ₂^3 + ζ₃^3 = 8) :
  ζ₁^8 + ζ₂^8 + ζ₃^8 = 451.625 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l3526_352640


namespace NUMINAMATH_CALUDE_like_terms_exponents_l3526_352612

theorem like_terms_exponents (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ), 3 * a^(2*m) * b^2 = k * (-1/2 * a^2 * b^(n+1))) →
  m + n = 2 := by
sorry

end NUMINAMATH_CALUDE_like_terms_exponents_l3526_352612


namespace NUMINAMATH_CALUDE_dad_second_half_speed_l3526_352621

-- Define the given conditions
def total_time : Real := 0.5  -- 30 minutes in hours
def first_half_speed : Real := 28
def jake_bike_speed : Real := 11
def jake_bike_time : Real := 2

-- Define the theorem
theorem dad_second_half_speed :
  let total_distance := jake_bike_speed * jake_bike_time
  let first_half_distance := first_half_speed * (total_time / 2)
  let second_half_distance := total_distance - first_half_distance
  let second_half_speed := second_half_distance / (total_time / 2)
  second_half_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_dad_second_half_speed_l3526_352621


namespace NUMINAMATH_CALUDE_girls_count_in_school_l3526_352674

theorem girls_count_in_school (total_students : ℕ) (boys_avg_age girls_avg_age school_avg_age : ℚ) : 
  total_students = 652 →
  boys_avg_age = 12 →
  girls_avg_age = 11 →
  school_avg_age = 11.75 →
  ∃ (girls_count : ℕ), 
    girls_count = 162 ∧ 
    (total_students - girls_count) * boys_avg_age + girls_count * girls_avg_age = total_students * school_avg_age :=
by sorry

end NUMINAMATH_CALUDE_girls_count_in_school_l3526_352674


namespace NUMINAMATH_CALUDE_field_trip_total_cost_l3526_352619

def field_trip_cost (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_price : ℚ) (teacher_ticket_price : ℚ) 
  (discount_rate : ℚ) (tour_price : ℚ) (bus_cost : ℚ) 
  (meal_cost : ℚ) : ℚ :=
  let total_people := num_students + num_teachers
  let ticket_cost := num_students * student_ticket_price + num_teachers * teacher_ticket_price
  let discounted_ticket_cost := ticket_cost * (1 - discount_rate)
  let tour_cost := total_people * tour_price
  let meal_cost_total := total_people * meal_cost
  discounted_ticket_cost + tour_cost + bus_cost + meal_cost_total

theorem field_trip_total_cost :
  field_trip_cost 25 6 1.5 4 0.2 3.5 100 7.5 = 490.2 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_total_cost_l3526_352619


namespace NUMINAMATH_CALUDE_bus_time_calculation_l3526_352677

def minutes_in_day : ℕ := 24 * 60

def time_to_minutes (hours minutes : ℕ) : ℕ :=
  hours * 60 + minutes

theorem bus_time_calculation (leave_home arrive_bus arrive_home class_duration num_classes other_activities : ℕ) :
  leave_home = time_to_minutes 7 0 →
  arrive_bus = time_to_minutes 7 45 →
  arrive_home = time_to_minutes 17 15 →
  class_duration = 55 →
  num_classes = 8 →
  other_activities = time_to_minutes 1 45 →
  arrive_home - leave_home - (class_duration * num_classes + other_activities) = 25 := by
  sorry

#check bus_time_calculation

end NUMINAMATH_CALUDE_bus_time_calculation_l3526_352677


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3526_352667

/-- An arithmetic sequence with first term 2 and the sum of the second and third terms equal to 13 -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ 
  a 2 + a 3 = 13 ∧ 
  ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) :
  a 4 + a 5 + a 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3526_352667


namespace NUMINAMATH_CALUDE_samantha_score_l3526_352675

/-- Calculates the score for a revised AMC 8 contest --/
def calculate_score (correct : ℕ) (incorrect : ℕ) (unanswered : ℕ) : ℤ :=
  2 * correct - incorrect

/-- Proves that Samantha's score is 25 given the problem conditions --/
theorem samantha_score :
  let correct : ℕ := 15
  let incorrect : ℕ := 5
  let unanswered : ℕ := 5
  let total_questions : ℕ := correct + incorrect + unanswered
  total_questions = 25 →
  calculate_score correct incorrect unanswered = 25 := by
  sorry

end NUMINAMATH_CALUDE_samantha_score_l3526_352675


namespace NUMINAMATH_CALUDE_min_triple_sum_bound_l3526_352695

def circle_arrangement (n : ℕ) := Fin n → ℕ

theorem min_triple_sum_bound (arr : circle_arrangement 10) :
  ∀ i : Fin 10, arr i ∈ Finset.range 11 →
  (∀ i j : Fin 10, i ≠ j → arr i ≠ arr j) →
  ∃ i : Fin 10, arr i + arr ((i + 1) % 10) + arr ((i + 2) % 10) ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_min_triple_sum_bound_l3526_352695


namespace NUMINAMATH_CALUDE_count_solutions_equation_l3526_352631

theorem count_solutions_equation : 
  ∃! (S : Finset ℕ), 
    (∀ n ∈ S, n > 0 ∧ (n + 500) / 50 = ⌊Real.sqrt (2 * n)⌋) ∧ 
    S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_solutions_equation_l3526_352631


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3526_352670

/-- Given a quadratic equation x^2 + (a+1)x + 4 = 0 with roots x₁ and x₂, where x₁ = 1 + √3i and a ∈ ℝ,
    prove that a = -3 and the distance between the points corresponding to x₁ and x₂ in the complex plane is 2√3. -/
theorem quadratic_equation_roots (a : ℝ) (x₁ x₂ : ℂ) : 
  x₁^2 + (a+1)*x₁ + 4 = 0 ∧ 
  x₂^2 + (a+1)*x₂ + 4 = 0 ∧
  x₁ = 1 + Complex.I * Real.sqrt 3 →
  a = -3 ∧ 
  Complex.abs (x₁ - x₂) = Real.sqrt 12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3526_352670


namespace NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l3526_352696

theorem least_positive_integer_divisible_by_four_primes : 
  ∃ (p₁ p₂ p₃ p₄ : Nat), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
    210 % p₁ = 0 ∧ 210 % p₂ = 0 ∧ 210 % p₃ = 0 ∧ 210 % p₄ = 0 ∧
    ∀ n : Nat, n > 0 ∧ n < 210 → 
      ¬∃ (q₁ q₂ q₃ q₄ : Nat), 
        Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧
        q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₃ ≠ q₄ ∧
        n % q₁ = 0 ∧ n % q₂ = 0 ∧ n % q₃ = 0 ∧ n % q₄ = 0 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_divisible_by_four_primes_l3526_352696


namespace NUMINAMATH_CALUDE_three_hour_charge_l3526_352689

/-- Represents the therapy pricing structure and calculates total charges --/
structure TherapyPricing where
  first_hour : ℝ
  subsequent_hour : ℝ
  service_fee_rate : ℝ
  first_hour_premium : ℝ
  eight_hour_total : ℝ

/-- Calculates the total charge for a given number of hours --/
def total_charge (p : TherapyPricing) (hours : ℕ) : ℝ :=
  let base_charge := p.first_hour + p.subsequent_hour * (hours - 1)
  base_charge * (1 + p.service_fee_rate)

/-- Theorem stating the total charge for 3 hours of therapy --/
theorem three_hour_charge (p : TherapyPricing) : 
  p.first_hour = p.subsequent_hour + p.first_hour_premium →
  p.service_fee_rate = 0.1 →
  p.first_hour_premium = 50 →
  total_charge p 8 = p.eight_hour_total →
  p.eight_hour_total = 900 →
  total_charge p 3 = 371.87 := by
  sorry

end NUMINAMATH_CALUDE_three_hour_charge_l3526_352689


namespace NUMINAMATH_CALUDE_total_treats_is_275_l3526_352658

/-- The total number of treats Mary, John, and Sue have -/
def total_treats (chewing_gums chocolate_bars lollipops cookies other_candies : ℕ) : ℕ :=
  chewing_gums + chocolate_bars + lollipops + cookies + other_candies

/-- Theorem stating that the total number of treats is 275 -/
theorem total_treats_is_275 :
  total_treats 60 55 70 50 40 = 275 := by
  sorry

end NUMINAMATH_CALUDE_total_treats_is_275_l3526_352658


namespace NUMINAMATH_CALUDE_blue_apples_count_l3526_352633

theorem blue_apples_count (b : ℕ) : 
  (3 * b : ℚ) - (3 * b : ℚ) / 5 = 12 → b = 5 := by sorry

end NUMINAMATH_CALUDE_blue_apples_count_l3526_352633


namespace NUMINAMATH_CALUDE_determinant_zero_l3526_352697

-- Define the cubic equation
def cubic_equation (x s t : ℝ) : Prop := x^3 + s*x^2 + t*x = 0

-- Define the determinant of the 3x3 matrix
def matrix_determinant (x y z : ℝ) : ℝ :=
  x * (z * y - x * x) - y * (y * y - x * z) + z * (y * z - z * x)

-- Theorem statement
theorem determinant_zero (x y z s t : ℝ) 
  (hx : cubic_equation x s t) 
  (hy : cubic_equation y s t) 
  (hz : cubic_equation z s t) : 
  matrix_determinant x y z = 0 := by sorry

end NUMINAMATH_CALUDE_determinant_zero_l3526_352697


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l3526_352679

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

-- Main theorem
theorem smallest_prime_12_less_than_square : 
  ∀ n : ℕ, n > 0 → is_prime n → (∃ m : ℕ, is_perfect_square m ∧ n = m - 12) → n ≥ 13 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_square_l3526_352679


namespace NUMINAMATH_CALUDE_water_flow_fraction_l3526_352647

/-- Given a water flow problem with the following conditions:
  * The original flow rate is 5 gallons per minute
  * The reduced flow rate is 2 gallons per minute
  * The reduced flow rate is 1 gallon per minute less than a fraction of the original flow rate
  Prove that the fraction of the original flow rate is 3/5 -/
theorem water_flow_fraction (original_rate reduced_rate : ℚ) 
  (h1 : original_rate = 5)
  (h2 : reduced_rate = 2) :
  ∃ f : ℚ, f * original_rate - 1 = reduced_rate ∧ f = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_water_flow_fraction_l3526_352647


namespace NUMINAMATH_CALUDE_line_segments_and_midpoints_l3526_352665

/-- The number of line segments that can be formed with n points on a line -/
def num_segments (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of unique midpoints of line segments formed by n points on a line -/
def num_midpoints (n : ℕ) : ℕ := 2 * n - 3

/-- The number of points on the line -/
def num_points : ℕ := 10

theorem line_segments_and_midpoints :
  num_segments num_points = 45 ∧ num_midpoints num_points = 17 := by
  sorry

end NUMINAMATH_CALUDE_line_segments_and_midpoints_l3526_352665


namespace NUMINAMATH_CALUDE_max_type_a_workers_l3526_352623

theorem max_type_a_workers (total : ℕ) (x : ℕ) : 
  total = 150 → 
  total - x ≥ 3 * x → 
  x ≤ 37 ∧ ∃ y : ℕ, y > 37 → total - y < 3 * y :=
sorry

end NUMINAMATH_CALUDE_max_type_a_workers_l3526_352623


namespace NUMINAMATH_CALUDE_race_length_l3526_352644

/-- Represents the state of the race -/
structure RaceState where
  alexLead : Int
  distanceLeft : Int

/-- Calculates the final race state after all lead changes -/
def finalRaceState : RaceState :=
  let s1 : RaceState := { alexLead := 0, distanceLeft := 0 }  -- Even start
  let s2 : RaceState := { alexLead := 300, distanceLeft := s1.distanceLeft }
  let s3 : RaceState := { alexLead := s2.alexLead - 170, distanceLeft := s2.distanceLeft }
  { alexLead := s3.alexLead + 440, distanceLeft := 3890 }

/-- The theorem stating the total length of the race track -/
theorem race_length : 
  finalRaceState.alexLead + finalRaceState.distanceLeft = 4460 := by
  sorry


end NUMINAMATH_CALUDE_race_length_l3526_352644


namespace NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3526_352664

theorem sum_of_solutions_is_zero : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (18 * x₁) / 27 = 7 / x₁ ∧ 
  (18 * x₂) / 27 = 7 / x₂ ∧ 
  x₁ + x₂ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_is_zero_l3526_352664


namespace NUMINAMATH_CALUDE_smallest_number_l3526_352690

theorem smallest_number (a b c d e : ℚ) : 
  a = 3.4 ∧ b = 7/2 ∧ c = 1.7 ∧ d = 27/10 ∧ e = 2.9 →
  c ≤ a ∧ c ≤ b ∧ c ≤ d ∧ c ≤ e := by
sorry

end NUMINAMATH_CALUDE_smallest_number_l3526_352690


namespace NUMINAMATH_CALUDE_diamond_two_three_l3526_352608

/-- The diamond operation defined for real numbers -/
def diamond (a b : ℝ) : ℝ := a * b^2 - b + 1

/-- Theorem stating that 2 ◇ 3 = 16 -/
theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_diamond_two_three_l3526_352608


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3526_352636

/-- The floor function, which returns the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The statement to be proved -/
theorem floor_equation_solution :
  let x : ℚ := 22 / 7
  x * (floor (x * (floor (x * (floor x))))) = 88 := by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3526_352636


namespace NUMINAMATH_CALUDE_garden_area_difference_l3526_352684

/-- Represents a rectangular garden -/
structure Garden where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden -/
def garden_area (g : Garden) : ℝ := g.length * g.width

/-- Represents a shed in the garden -/
structure Shed where
  length : ℝ
  width : ℝ

/-- Calculates the area of a shed -/
def shed_area (s : Shed) : ℝ := s.length * s.width

theorem garden_area_difference : 
  let karl_garden : Garden := { length := 30, width := 50 }
  let makenna_garden : Garden := { length := 35, width := 55 }
  let makenna_shed : Shed := { length := 5, width := 10 }
  (garden_area makenna_garden - shed_area makenna_shed) - garden_area karl_garden = 375 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_difference_l3526_352684


namespace NUMINAMATH_CALUDE_qin_jiushao_evaluation_l3526_352649

-- Define the polynomial coefficients
def a₀ : ℝ := 12
def a₁ : ℝ := 35
def a₂ : ℝ := -8
def a₃ : ℝ := 79
def a₄ : ℝ := 6
def a₅ : ℝ := 5
def a₆ : ℝ := 3

-- Define the evaluation point
def x : ℝ := -4

-- Define Qin Jiushao's algorithm
def qin_jiushao (a : Fin 7 → ℝ) (x : ℝ) : ℝ :=
  (((((a 6 * x + a 5) * x + a 4) * x + a 3) * x + a 2) * x + a 1) * x + a 0

-- Theorem statement
theorem qin_jiushao_evaluation :
  qin_jiushao (fun i => [a₀, a₁, a₂, a₃, a₄, a₅, a₆].get i) x = 220 := by
  sorry

end NUMINAMATH_CALUDE_qin_jiushao_evaluation_l3526_352649


namespace NUMINAMATH_CALUDE_cross_number_puzzle_l3526_352635

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem cross_number_puzzle :
  ∃ (m n : ℕ),
    is_three_digit (3^m) ∧
    is_three_digit (7^n) ∧
    (3^m / 10) % 10 = (7^n / 10) % 10 ∧
    (3^m / 10) % 10 = 4 :=
by sorry

end NUMINAMATH_CALUDE_cross_number_puzzle_l3526_352635


namespace NUMINAMATH_CALUDE_train_length_proof_l3526_352622

/-- Proves that a train with the given conditions has a length of 1800 meters -/
theorem train_length_proof (train_speed : ℝ) (crossing_time : ℝ) (train_length : ℝ) : 
  train_speed = 216 →
  crossing_time = 1 →
  train_length = 1800 :=
by
  sorry

#check train_length_proof

end NUMINAMATH_CALUDE_train_length_proof_l3526_352622


namespace NUMINAMATH_CALUDE_function_properties_l3526_352605

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2^x + a) / (2^x - 1)

theorem function_properties (a : ℝ) :
  (∀ x : ℝ, x ≠ 0 → f a x = -f a (-x)) →
  (∀ x : ℝ, x ≠ 0 → f a x = f a x) ∧
  (a = 1) ∧
  (∀ x y : ℝ, 0 < x → x < y → f a y < f a x) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l3526_352605


namespace NUMINAMATH_CALUDE_system_of_equations_l3526_352643

theorem system_of_equations (x y z : ℝ) 
  (eq1 : y + z = 15 - 4*x)
  (eq2 : x + z = -17 - 4*y)
  (eq3 : x + y = 9 - 4*z) :
  2*x + 2*y + 2*z = 7/3 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l3526_352643


namespace NUMINAMATH_CALUDE_sequence_a2_value_l3526_352632

theorem sequence_a2_value (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = 2 * (a n - 1)) : a 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a2_value_l3526_352632


namespace NUMINAMATH_CALUDE_function_composition_problem_l3526_352699

theorem function_composition_problem (k b : ℝ) (f : ℝ → ℝ) :
  (k < 0) →
  (∀ x, f x = k * x + b) →
  (∀ x, f (f x) = 4 * x + 1) →
  (∀ x, f x = -2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_function_composition_problem_l3526_352699


namespace NUMINAMATH_CALUDE_blocks_remaining_problem_l3526_352694

/-- Given a person with an initial number of blocks and a number of blocks used,
    calculate the remaining number of blocks. -/
def remaining_blocks (initial : ℕ) (used : ℕ) : ℕ :=
  initial - used

/-- Theorem stating that given 78 initial blocks and 19 used blocks,
    the remaining number of blocks is 59. -/
theorem blocks_remaining_problem :
  remaining_blocks 78 19 = 59 := by
  sorry

end NUMINAMATH_CALUDE_blocks_remaining_problem_l3526_352694


namespace NUMINAMATH_CALUDE_number_added_at_end_l3526_352682

theorem number_added_at_end (x : ℝ) : (26.3 * 12 * 20) / 3 + x = 2229 → x = 125 := by
  sorry

end NUMINAMATH_CALUDE_number_added_at_end_l3526_352682


namespace NUMINAMATH_CALUDE_unique_b_value_l3526_352660

theorem unique_b_value (b h a : ℕ) (hb_pos : 0 < b) (hh_pos : 0 < h) (hb_lt_h : b < h)
  (heq : b^2 + h^2 = b*(a + h) + a*h) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_b_value_l3526_352660


namespace NUMINAMATH_CALUDE_director_selection_probability_l3526_352650

def total_actors : ℕ := 5
def golden_rooster_winners : ℕ := 2
def hundred_flowers_winners : ℕ := 3

def probability_select_2_golden_1_hundred : ℚ := 3 / 10

theorem director_selection_probability :
  (golden_rooster_winners.choose 2 * hundred_flowers_winners) / 
  (total_actors.choose 3) = probability_select_2_golden_1_hundred := by
  sorry

end NUMINAMATH_CALUDE_director_selection_probability_l3526_352650


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3526_352666

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : arithmetic_sequence a d)
  (h_sum1 : a 3 + a 6 = 11)
  (h_sum2 : a 5 + a 8 = 39) :
  d = 7 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3526_352666


namespace NUMINAMATH_CALUDE_range_of_a_l3526_352607

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) →
  (∃ x : ℝ, x^2 + 2*a*x + (2 - a) = 0) →
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3526_352607


namespace NUMINAMATH_CALUDE_intersection_formula_l3526_352676

/-- Given complex numbers a and b on a circle centered at the origin,
    u is the intersection of tangents at a and b -/
def intersection_of_tangents (a b : ℂ) : ℂ := sorry

/-- a and b lie on a circle centered at the origin -/
def on_circle (a b : ℂ) : Prop := sorry

theorem intersection_formula {a b : ℂ} (h : on_circle a b) :
  intersection_of_tangents a b = 2 * a * b / (a + b) := by sorry

end NUMINAMATH_CALUDE_intersection_formula_l3526_352676


namespace NUMINAMATH_CALUDE_even_increasing_relation_l3526_352620

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x < y → f x < f y

theorem even_increasing_relation (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_incr : increasing_on_nonneg f) :
  f π > f (-3) ∧ f (-3) > f (-2) :=
sorry

end NUMINAMATH_CALUDE_even_increasing_relation_l3526_352620


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3526_352698

-- Define the equations
def equation1 (x : ℝ) : Prop := 7*x - 20 = 2*(3 - 3*x)
def equation2 (x : ℝ) : Prop := (2*x - 3)/5 = (3*x - 1)/2 + 1

-- Theorem for equation 1
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 2 := by sorry

-- Theorem for equation 2
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l3526_352698


namespace NUMINAMATH_CALUDE_negative_one_times_negative_three_l3526_352646

theorem negative_one_times_negative_three : (-1 : ℤ) * (-3 : ℤ) = (3 : ℤ) := by
  sorry

end NUMINAMATH_CALUDE_negative_one_times_negative_three_l3526_352646


namespace NUMINAMATH_CALUDE_correct_guess_and_multiply_l3526_352683

def coin_head_prob : ℚ := 2/3
def aaron_head_guess_prob : ℚ := 2/3

def correct_guess_prob : ℚ := 
  coin_head_prob * aaron_head_guess_prob + (1 - coin_head_prob) * (1 - aaron_head_guess_prob)

theorem correct_guess_and_multiply :
  correct_guess_prob = 5/9 ∧ 9000 * correct_guess_prob = 5000 := by sorry

end NUMINAMATH_CALUDE_correct_guess_and_multiply_l3526_352683


namespace NUMINAMATH_CALUDE_number_solution_l3526_352685

theorem number_solution : ∃ x : ℝ, (50 + 5 * 12 / (180 / x) = 51) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_solution_l3526_352685


namespace NUMINAMATH_CALUDE_second_platform_length_l3526_352654

/-- The length of the second platform given train and first platform details -/
theorem second_platform_length
  (train_length : ℝ)
  (first_platform_length : ℝ)
  (time_first_platform : ℝ)
  (time_second_platform : ℝ)
  (h1 : train_length = 150)
  (h2 : first_platform_length = 150)
  (h3 : time_first_platform = 15)
  (h4 : time_second_platform = 20) :
  (time_second_platform * (train_length + first_platform_length) / time_first_platform) - train_length = 250 :=
by sorry

end NUMINAMATH_CALUDE_second_platform_length_l3526_352654


namespace NUMINAMATH_CALUDE_concert_ticket_price_l3526_352602

theorem concert_ticket_price :
  ∃ (P : ℝ) (x : ℕ),
    x + 2 + 1 = 5 ∧
    x * P + (2 * 2.4 * P - 10) + 0.6 * P = 360 →
    P = 50 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l3526_352602


namespace NUMINAMATH_CALUDE_fundraising_contribution_l3526_352624

theorem fundraising_contribution (total_goal : ℕ) (already_raised : ℕ) (num_people : ℕ) :
  total_goal = 2400 →
  already_raised = 600 →
  num_people = 8 →
  (total_goal - already_raised) / num_people = 225 :=
by
  sorry

end NUMINAMATH_CALUDE_fundraising_contribution_l3526_352624


namespace NUMINAMATH_CALUDE_circle_equation_l3526_352613

/-- The equation of a circle passing through points A(1, -1) and B(-1, 1) with center on the line x + y - 2 = 0 -/
theorem circle_equation (x y : ℝ) : 
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (-1, 1)
  let center_line (p : ℝ × ℝ) := p.1 + p.2 - 2 = 0
  let circle_eq (p : ℝ × ℝ) := (p.1 - 1)^2 + (p.2 - 1)^2 = 4
  let on_circle (p : ℝ × ℝ) := circle_eq p
  ∃ (c : ℝ × ℝ), 
    center_line c ∧ 
    on_circle A ∧ 
    on_circle B ∧ 
    on_circle (x, y) ↔ circle_eq (x, y) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l3526_352613


namespace NUMINAMATH_CALUDE_police_speed_l3526_352657

/-- Proves that the speed of a police officer chasing a thief is 40 km/hr given specific conditions --/
theorem police_speed (thief_speed : ℝ) (police_station_distance : ℝ) (police_delay : ℝ) (catch_time : ℝ) :
  thief_speed = 20 →
  police_station_distance = 60 →
  police_delay = 1 →
  catch_time = 4 →
  (police_station_distance + thief_speed * (police_delay + catch_time)) / catch_time = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_police_speed_l3526_352657


namespace NUMINAMATH_CALUDE_problem_solution_l3526_352634

/-- Checks if a sequence of binomial coefficients forms an arithmetic sequence -/
def is_arithmetic_sequence (n : ℕ) (j : ℕ) (k : ℕ) : Prop :=
  ∀ i : ℕ, i < k - 1 → 2 * (n.choose (j + i + 1)) = (n.choose (j + i)) + (n.choose (j + i + 2))

/-- The value of k that satisfies the conditions of the problem -/
def k : ℕ := 4

/-- The condition (a) of the problem -/
def condition_a (k : ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → ∀ j : ℕ, j ≤ n - k + 1 → ¬(is_arithmetic_sequence n j k)

/-- The condition (b) of the problem -/
def condition_b (k : ℕ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ ∃ j : ℕ, j ≤ n - k + 2 ∧ is_arithmetic_sequence n j (k - 1)

/-- The form of n that satisfies condition (b) for k = 4 -/
def valid_n (m : ℕ) : ℕ := m^2 - 2

theorem problem_solution :
  condition_a k ∧
  condition_b k ∧
  (∀ n : ℕ, n > 0 → (∃ j : ℕ, j ≤ n - k + 2 ∧ is_arithmetic_sequence n j (k - 1))
                 ↔ (∃ m : ℕ, m ≥ 3 ∧ n = valid_n m)) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3526_352634


namespace NUMINAMATH_CALUDE_scientific_notation_42000_l3526_352678

theorem scientific_notation_42000 :
  (42000 : ℝ) = 4.2 * (10 : ℝ)^4 := by sorry

end NUMINAMATH_CALUDE_scientific_notation_42000_l3526_352678


namespace NUMINAMATH_CALUDE_weight_of_b_l3526_352616

/-- Given three weights a, b, and c, prove that b = 37 under the given conditions -/
theorem weight_of_b (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (a + b) / 2 = 40 →
  (b + c) / 2 = 46 →
  b = 37 := by
sorry


end NUMINAMATH_CALUDE_weight_of_b_l3526_352616


namespace NUMINAMATH_CALUDE_index_card_area_l3526_352645

theorem index_card_area (l w : ℕ) (h1 : l = 3) (h2 : w = 7) : 
  (∃ (a b : ℕ), (l - a) * (w - b) = 10 ∧ a + b = 3) → 
  (l - 1) * (w - 2) = 10 := by
sorry

end NUMINAMATH_CALUDE_index_card_area_l3526_352645


namespace NUMINAMATH_CALUDE_marble_count_l3526_352611

theorem marble_count (fabian kyle miles : ℕ) 
  (h1 : fabian = 15)
  (h2 : fabian = 3 * kyle)
  (h3 : fabian = 5 * miles) :
  kyle + miles = 8 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l3526_352611


namespace NUMINAMATH_CALUDE_angle_D_measure_l3526_352693

theorem angle_D_measure (A B C D : ℝ) : 
  -- ABCD is a convex quadrilateral
  0 < A ∧ A < π ∧
  0 < B ∧ B < π ∧
  0 < C ∧ C < π ∧
  0 < D ∧ D < π ∧
  A + B + C + D = 2 * π ∧
  -- ∠C = 57°
  C = 57 * π / 180 ∧
  -- sin ∠A + sin ∠B = √2
  Real.sin A + Real.sin B = Real.sqrt 2 ∧
  -- cos ∠A + cos ∠B = 2 - √2
  Real.cos A + Real.cos B = 2 - Real.sqrt 2
  -- Then ∠D = 168°
  → D = 168 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_D_measure_l3526_352693


namespace NUMINAMATH_CALUDE_wood_length_ratio_l3526_352606

def first_set_length : ℝ := 4
def second_set_length : ℝ := 20

theorem wood_length_ratio : second_set_length / first_set_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_wood_length_ratio_l3526_352606


namespace NUMINAMATH_CALUDE_max_toys_theorem_l3526_352652

def max_toys_purchasable (initial_amount : ℚ) (game_cost : ℚ) (tax_rate : ℚ) (toy_cost : ℚ) : ℕ :=
  let total_game_cost := game_cost * (1 + tax_rate)
  let remaining_money := initial_amount - total_game_cost
  (remaining_money / toy_cost).floor.toNat

theorem max_toys_theorem :
  max_toys_purchasable 57 27 (8/100) 6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_max_toys_theorem_l3526_352652


namespace NUMINAMATH_CALUDE_decimal_expansion_contains_all_digits_l3526_352687

theorem decimal_expansion_contains_all_digits (p : ℕ) (hp : p.Prime) (hp_large : p > 10^9) 
  (hq : (4*p + 1).Prime) : 
  ∀ d : Fin 10, ∃ n : ℕ, (10^n - 1) % (4*p + 1) = d.val * ((10^n - 1) / (4*p + 1)) :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_contains_all_digits_l3526_352687
