import Mathlib

namespace NUMINAMATH_CALUDE_document_word_count_l3478_347800

/-- Given Barbara's typing speeds and time, calculate the number of words in the document -/
theorem document_word_count 
  (original_speed : ℕ) 
  (speed_reduction : ℕ) 
  (typing_time : ℕ) 
  (h1 : original_speed = 212)
  (h2 : speed_reduction = 40)
  (h3 : typing_time = 20) : 
  (original_speed - speed_reduction) * typing_time = 3440 :=
by sorry

end NUMINAMATH_CALUDE_document_word_count_l3478_347800


namespace NUMINAMATH_CALUDE_trapezoid_with_perpendicular_bisector_quadrilateral_is_isosceles_l3478_347870

-- Define the trapezoid and quadrilateral
variable (A B C D K L M N : Point)

-- Define the trapezoid ABCD
def is_trapezoid (A B C D : Point) : Prop := sorry

-- Define the angle bisectors of the trapezoid
def angle_bisectors_intersect (A B C D K L M N : Point) : Prop := sorry

-- Define the quadrilateral KLMN formed by the intersection of angle bisectors
def quadrilateral_from_bisectors (A B C D K L M N : Point) : Prop := sorry

-- Define perpendicular diagonals of KLMN
def perpendicular_diagonals (K L M N : Point) : Prop := sorry

-- Define an isosceles trapezoid
def is_isosceles_trapezoid (A B C D : Point) : Prop := sorry

-- Theorem statement
theorem trapezoid_with_perpendicular_bisector_quadrilateral_is_isosceles 
  (h1 : is_trapezoid A B C D)
  (h2 : angle_bisectors_intersect A B C D K L M N)
  (h3 : quadrilateral_from_bisectors A B C D K L M N)
  (h4 : perpendicular_diagonals K L M N) :
  is_isosceles_trapezoid A B C D := by sorry

end NUMINAMATH_CALUDE_trapezoid_with_perpendicular_bisector_quadrilateral_is_isosceles_l3478_347870


namespace NUMINAMATH_CALUDE_solve_system_l3478_347885

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 8) (eq2 : 2 * x + 3 * y = 11) :
  x = 46 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l3478_347885


namespace NUMINAMATH_CALUDE_john_house_wall_planks_l3478_347877

/-- The number of planks John uses for the house wall -/
def num_planks : ℕ := 32 / 2

/-- Each plank needs 2 nails -/
def nails_per_plank : ℕ := 2

/-- The total number of nails needed for the wall -/
def total_nails : ℕ := 32

theorem john_house_wall_planks : num_planks = 16 := by
  sorry

end NUMINAMATH_CALUDE_john_house_wall_planks_l3478_347877


namespace NUMINAMATH_CALUDE_x_1997_value_l3478_347895

def x : ℕ → ℕ
  | 0 => 1
  | n + 1 => x n + (x n / (n + 1)) + 2

theorem x_1997_value : x 1996 = 23913 := by
  sorry

end NUMINAMATH_CALUDE_x_1997_value_l3478_347895


namespace NUMINAMATH_CALUDE_three_at_five_equals_neg_six_l3478_347849

-- Define the @ operation
def at_op (a b : ℤ) : ℤ := 3 * a - 3 * b

-- Theorem statement
theorem three_at_five_equals_neg_six : at_op 3 5 = -6 := by
  sorry

end NUMINAMATH_CALUDE_three_at_five_equals_neg_six_l3478_347849


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3478_347812

theorem greatest_prime_factor_of_expression : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (5^8 + 10^7) ∧ ∀ (q : ℕ), q.Prime → q ∣ (5^8 + 10^7) → q ≤ p :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_expression_l3478_347812


namespace NUMINAMATH_CALUDE_fort_food_duration_l3478_347820

/-- Calculates the initial number of days the food provision was meant to last given:
  * The initial number of men
  * The number of days after which some men left
  * The number of men who left
  * The number of days the food lasted after some men left
-/
def initialFoodDuration (initialMen : ℕ) (daysBeforeLeaving : ℕ) (menWhoLeft : ℕ) (remainingDays : ℕ) : ℕ :=
  (initialMen * daysBeforeLeaving + (initialMen - menWhoLeft) * remainingDays) / initialMen

theorem fort_food_duration :
  initialFoodDuration 150 10 25 42 = 45 := by
  sorry

#eval initialFoodDuration 150 10 25 42

end NUMINAMATH_CALUDE_fort_food_duration_l3478_347820


namespace NUMINAMATH_CALUDE_F_and_G_increasing_l3478_347873

-- Define the functions f, g, F, and G
variable (f g : ℝ → ℝ)
def F (x : ℝ) := f x + g x
def G (x : ℝ) := f x - g x

-- State the theorem
theorem F_and_G_increasing
  (h_incr : ∀ x y, x < y → f x < f y)
  (h_ineq : ∀ x y, x ≠ y → (f x - f y)^2 > (g x - g y)^2) :
  (∀ x y, x < y → F f g x < F f g y) ∧
  (∀ x y, x < y → G f g x < G f g y) :=
sorry

end NUMINAMATH_CALUDE_F_and_G_increasing_l3478_347873


namespace NUMINAMATH_CALUDE_sams_trip_length_l3478_347832

theorem sams_trip_length (total : ℚ) 
  (h1 : total / 4 + 24 + total / 6 = total) : total = 288 / 7 := by
  sorry

end NUMINAMATH_CALUDE_sams_trip_length_l3478_347832


namespace NUMINAMATH_CALUDE_no_real_solution_for_matrix_equation_l3478_347819

theorem no_real_solution_for_matrix_equation :
  (∀ a b : ℝ, Matrix.det !![a, 2*b; 2*a, b] = a*b - 4*a*b) →
  ¬∃ x : ℝ, Matrix.det !![3*x, 2; 6*x, x] = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_for_matrix_equation_l3478_347819


namespace NUMINAMATH_CALUDE_valid_fractions_characterization_l3478_347805

def is_valid_fraction (n d : ℕ) : Prop :=
  0 < d ∧ d < 10 ∧ (7:ℚ)/9 < (n:ℚ)/d ∧ (n:ℚ)/d < (8:ℚ)/9

def valid_fractions : Set (ℕ × ℕ) :=
  {(n, d) | is_valid_fraction n d}

theorem valid_fractions_characterization :
  valid_fractions = {(5, 6), (6, 7), (7, 8), (4, 5)} := by sorry

end NUMINAMATH_CALUDE_valid_fractions_characterization_l3478_347805


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_10_l3478_347852

theorem product_of_five_consecutive_integers_divisible_by_10 (n : ℕ) :
  ∃ k : ℕ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisible_by_10_l3478_347852


namespace NUMINAMATH_CALUDE_cube_volume_scaling_l3478_347874

theorem cube_volume_scaling (V : ℝ) (V_pos : V > 0) :
  let original_side := V ^ (1/3)
  let new_side := 2 * original_side
  let new_volume := new_side ^ 3
  new_volume = 8 * V := by sorry

end NUMINAMATH_CALUDE_cube_volume_scaling_l3478_347874


namespace NUMINAMATH_CALUDE_sunday_reading_time_is_46_l3478_347855

def book_a_assignment : ℕ := 60
def book_b_assignment : ℕ := 45
def friday_book_a : ℕ := 16
def saturday_book_a : ℕ := 28
def saturday_book_b : ℕ := 15

def sunday_reading_time : ℕ := 
  (book_a_assignment - (friday_book_a + saturday_book_a)) + 
  (book_b_assignment - saturday_book_b)

theorem sunday_reading_time_is_46 : sunday_reading_time = 46 := by
  sorry

end NUMINAMATH_CALUDE_sunday_reading_time_is_46_l3478_347855


namespace NUMINAMATH_CALUDE_quadratic_abs_value_analysis_l3478_347804

theorem quadratic_abs_value_analysis (x a : ℝ) :
  (x ≥ a → x^2 + 4*x - 2*|x - a| + 2 - a = x^2 + 2*x + a + 2) ∧
  (x < a → x^2 + 4*x - 2*|x - a| + 2 - a = x^2 + 6*x - 3*a + 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_abs_value_analysis_l3478_347804


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3478_347822

theorem inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | 12 * x^2 - a * x > a^2}
  S = if a > 0 then
        {x : ℝ | x < -a/4 ∨ x > a/3}
      else if a = 0 then
        {x : ℝ | x ≠ 0}
      else
        {x : ℝ | x < a/3 ∨ x > -a/4} :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3478_347822


namespace NUMINAMATH_CALUDE_set_intersection_example_l3478_347808

theorem set_intersection_example :
  let M : Set ℕ := {1, 2, 3}
  let N : Set ℕ := {2, 3, 4}
  M ∩ N = {2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_example_l3478_347808


namespace NUMINAMATH_CALUDE_unit_vector_magnitude_is_one_l3478_347889

variable {V : Type*} [NormedAddCommGroup V]

/-- The magnitude of a unit vector is equal to 1. -/
theorem unit_vector_magnitude_is_one (v : V) (h : ‖v‖ = 1) : ‖v‖ = 1 := by
  sorry

end NUMINAMATH_CALUDE_unit_vector_magnitude_is_one_l3478_347889


namespace NUMINAMATH_CALUDE_sunnydale_farm_arrangement_l3478_347802

/-- The number of ways to arrange animals in a row -/
def arrange_animals (chickens dogs cats rabbits : ℕ) : ℕ :=
  Nat.factorial 4 * Nat.factorial chickens * Nat.factorial dogs * Nat.factorial cats * Nat.factorial rabbits

/-- Theorem stating the number of arrangements for the given animal counts -/
theorem sunnydale_farm_arrangement :
  arrange_animals 5 3 4 3 = 2488320 :=
by sorry

end NUMINAMATH_CALUDE_sunnydale_farm_arrangement_l3478_347802


namespace NUMINAMATH_CALUDE_same_solution_implies_k_equals_one_l3478_347865

theorem same_solution_implies_k_equals_one :
  (∃ x : ℝ, x - 2 = 0 ∧ 1 - (x + k) / 3 = 0) → k = 1 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_k_equals_one_l3478_347865


namespace NUMINAMATH_CALUDE_cosine_identity_l3478_347898

theorem cosine_identity (α : ℝ) :
  3 - 4 * Real.cos (4 * α - 3 * Real.pi) - Real.cos (5 * Real.pi + 8 * α) = 8 * (Real.cos (2 * α))^4 := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l3478_347898


namespace NUMINAMATH_CALUDE_school_experiment_l3478_347846

theorem school_experiment (boys girls : ℕ) (h1 : boys = 100) (h2 : girls = 125) : 
  (girls - boys) / girls * 100 = 20 ∧ (girls - boys) / boys * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_school_experiment_l3478_347846


namespace NUMINAMATH_CALUDE_smallest_number_l3478_347891

theorem smallest_number (S : Set ℤ) (h : S = {0, -1, 1, -5}) : 
  ∃ x ∈ S, ∀ y ∈ S, x ≤ y ∧ x = -5 :=
sorry

end NUMINAMATH_CALUDE_smallest_number_l3478_347891


namespace NUMINAMATH_CALUDE_complex_abs_from_square_l3478_347821

theorem complex_abs_from_square (z : ℂ) (h : z^2 = 16 - 30*I) : Complex.abs z = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_from_square_l3478_347821


namespace NUMINAMATH_CALUDE_semicircle_to_cone_volume_l3478_347831

/-- The volume of a cone formed by rolling a semicircle of radius R --/
theorem semicircle_to_cone_volume (R : ℝ) (R_pos : R > 0) :
  let r : ℝ := R / 2
  let h : ℝ := R * (Real.sqrt 3) / 2
  (1 / 3) * Real.pi * r^2 * h = (Real.sqrt 3 / 24) * Real.pi * R^3 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_to_cone_volume_l3478_347831


namespace NUMINAMATH_CALUDE_line_through_circle_centers_l3478_347876

/-- Given two circles in polar coordinates, C1: ρ = 2cos θ and C2: ρ = 2sin θ,
    the polar equation of the line passing through their centers is θ = π/4 -/
theorem line_through_circle_centers (θ : Real) :
  let c1 : Real → Real := fun θ => 2 * Real.cos θ
  let c2 : Real → Real := fun θ => 2 * Real.sin θ
  ∃ (ρ : Real), (ρ * Real.cos (π/4) = 1 ∧ ρ * Real.sin (π/4) = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_through_circle_centers_l3478_347876


namespace NUMINAMATH_CALUDE_acute_angle_range_l3478_347880

theorem acute_angle_range (α : Real) (h_acute : 0 < α ∧ α < Real.pi / 2) :
  (∃ x : Real, 3 * x^2 * Real.sin α - 4 * x * Real.cos α + 2 = 0) →
  0 < α ∧ α ≤ Real.pi / 6 := by
sorry

end NUMINAMATH_CALUDE_acute_angle_range_l3478_347880


namespace NUMINAMATH_CALUDE_ln_neg_implies_a_less_than_one_a_less_than_one_not_sufficient_for_ln_neg_l3478_347835

theorem ln_neg_implies_a_less_than_one :
  ∀ a : ℝ, Real.log a < 0 → a < 1 :=
sorry

theorem a_less_than_one_not_sufficient_for_ln_neg :
  ∃ a : ℝ, a < 1 ∧ ¬(Real.log a < 0) :=
sorry

end NUMINAMATH_CALUDE_ln_neg_implies_a_less_than_one_a_less_than_one_not_sufficient_for_ln_neg_l3478_347835


namespace NUMINAMATH_CALUDE_sin_difference_bound_l3478_347815

theorem sin_difference_bound (N : ℕ) :
  ∃ (n k : ℕ), n ≠ k ∧ n ≤ N + 1 ∧ k ≤ N + 1 ∧ |Real.sin n - Real.sin k| < 2 / N :=
sorry

end NUMINAMATH_CALUDE_sin_difference_bound_l3478_347815


namespace NUMINAMATH_CALUDE_frequency_proportion_l3478_347862

theorem frequency_proportion (frequency sample_size : ℕ) 
  (h1 : frequency = 80) 
  (h2 : sample_size = 100) : 
  (frequency : ℚ) / sample_size = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_frequency_proportion_l3478_347862


namespace NUMINAMATH_CALUDE_daves_painted_area_l3478_347864

theorem daves_painted_area 
  (total_area : ℝ) 
  (cathy_ratio : ℝ) 
  (dave_ratio : ℝ) 
  (h1 : total_area = 330) 
  (h2 : cathy_ratio = 4) 
  (h3 : dave_ratio = 7) : 
  dave_ratio / (cathy_ratio + dave_ratio) * total_area = 210 := by
sorry

end NUMINAMATH_CALUDE_daves_painted_area_l3478_347864


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l3478_347845

theorem quadratic_one_solution_sum (a : ℝ) : 
  let f := fun x : ℝ => 3 * x^2 + a * x + 6 * x + 7
  (∃! x, f x = 0) → 
  ∃ a₁ a₂ : ℝ, a = a₁ ∨ a = a₂ ∧ a₁ + a₂ = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l3478_347845


namespace NUMINAMATH_CALUDE_polynomial_rational_difference_l3478_347853

theorem polynomial_rational_difference (f : ℝ → ℝ) :
  (∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c) →
  (∀ (x y : ℝ), ∃ (q : ℚ), x - y = q → ∃ (r : ℚ), f x - f y = r) →
  ∃ (b : ℚ) (c : ℝ), ∀ x, f x = b * x + c :=
by sorry

end NUMINAMATH_CALUDE_polynomial_rational_difference_l3478_347853


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l3478_347850

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 10)
  (h2 : average_speed = 35) :
  let speed_second_hour := 2 * average_speed - speed_first_hour
  speed_second_hour = 60 := by
  sorry

#check car_speed_second_hour

end NUMINAMATH_CALUDE_car_speed_second_hour_l3478_347850


namespace NUMINAMATH_CALUDE_min_value_theorem_l3478_347813

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  (2*a + 3*b + 4*c) * ((a + b)⁻¹ + (b + c)⁻¹ + (c + a)⁻¹) ≥ 4.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3478_347813


namespace NUMINAMATH_CALUDE_right_angled_triangle_l3478_347860

theorem right_angled_triangle (α β γ : Real) (h1 : 0 < α) (h2 : 0 < β) (h3 : 0 < γ)
  (h4 : α + β + γ = Real.pi) 
  (h5 : (Real.sin α + Real.sin β) / (Real.cos α + Real.cos β) = Real.sin γ) : 
  γ = Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l3478_347860


namespace NUMINAMATH_CALUDE_special_polynomial_root_l3478_347883

/-- A fourth degree polynomial with specific root properties -/
structure SpecialPolynomial where
  P : ℝ → ℝ
  degree_four : ∃ (a b c d e : ℝ), ∀ x, P x = a*x^4 + b*x^3 + c*x^2 + d*x + e
  root_one : P 1 = 0
  root_three : P 3 = 0
  root_five : P 5 = 0
  derivative_root_seven : (deriv P) 7 = 0

/-- The remaining root of a SpecialPolynomial is 89/11 -/
theorem special_polynomial_root (p : SpecialPolynomial) : 
  ∃ (x : ℝ), x ≠ 1 ∧ x ≠ 3 ∧ x ≠ 5 ∧ p.P x = 0 ∧ x = 89/11 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_root_l3478_347883


namespace NUMINAMATH_CALUDE_correct_sample_sizes_l3478_347878

def model1_production : ℕ := 1600
def model2_production : ℕ := 6000
def model3_production : ℕ := 2000
def total_sample_size : ℕ := 48

theorem correct_sample_sizes :
  let total_production := model1_production + model2_production + model3_production
  let sample1 := (model1_production * total_sample_size) / total_production
  let sample2 := (model2_production * total_sample_size) / total_production
  let sample3 := (model3_production * total_sample_size) / total_production
  sample1 = 8 ∧ sample2 = 30 ∧ sample3 = 10 :=
by sorry

end NUMINAMATH_CALUDE_correct_sample_sizes_l3478_347878


namespace NUMINAMATH_CALUDE_kelsey_watched_160_l3478_347807

/-- The number of videos watched by three friends satisfies the given conditions -/
structure VideoWatching where
  total : ℕ
  kelsey_more_than_ekon : ℕ
  uma_more_than_ekon : ℕ
  h_total : total = 411
  h_kelsey_more : kelsey_more_than_ekon = 43
  h_uma_more : uma_more_than_ekon = 17

/-- Given the conditions, prove that Kelsey watched 160 videos -/
theorem kelsey_watched_160 (vw : VideoWatching) : 
  ∃ (ekon uma kelsey : ℕ), 
    ekon + uma + kelsey = vw.total ∧ 
    kelsey = ekon + vw.kelsey_more_than_ekon ∧ 
    uma = ekon + vw.uma_more_than_ekon ∧
    kelsey = 160 := by
  sorry

end NUMINAMATH_CALUDE_kelsey_watched_160_l3478_347807


namespace NUMINAMATH_CALUDE_proposition_p_or_q_is_true_l3478_347842

theorem proposition_p_or_q_is_true : 
  (1 ∈ { x : ℝ | x^2 - 2*x + 1 ≤ 0 }) ∨ (∀ x ∈ (Set.Icc 0 1 : Set ℝ), x^2 - 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_proposition_p_or_q_is_true_l3478_347842


namespace NUMINAMATH_CALUDE_min_group_size_with_94_percent_boys_l3478_347803

theorem min_group_size_with_94_percent_boys (boys girls : ℕ) :
  boys > 0 →
  girls > 0 →
  (boys : ℚ) / (boys + girls : ℚ) > 94 / 100 →
  boys + girls ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_min_group_size_with_94_percent_boys_l3478_347803


namespace NUMINAMATH_CALUDE_power_of_two_consecutive_zeros_l3478_347827

/-- For any positive integer k, there exists a positive integer n such that
    the decimal representation of 2^n contains exactly k consecutive zeros. -/
theorem power_of_two_consecutive_zeros (k : ℕ) (hk : k ≥ 1) :
  ∃ n : ℕ, ∃ a b : ℕ, ∃ m : ℕ,
    (a ≠ 0) ∧ (b ≠ 0) ∧ (m > k) ∧
    (2^n : ℕ) = a * 10^m + b * 10^(m-k) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_consecutive_zeros_l3478_347827


namespace NUMINAMATH_CALUDE_fraction_range_l3478_347810

theorem fraction_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hineq : a ≤ 2*b ∧ 2*b ≤ 2*a + b) :
  (4/9 : ℝ) ≤ (2*a*b)/(a^2 + 2*b^2) ∧ (2*a*b)/(a^2 + 2*b^2) ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_range_l3478_347810


namespace NUMINAMATH_CALUDE_milk_processing_time_l3478_347875

/-- Given two milk processing plants with the following conditions:
    - They process equal amounts of milk
    - The second plant starts 'a' days later than the first
    - The second plant processes 'm' liters more per day than the first
    - After 5a/9 days of joint work, 1/3 of the task remains
    - The work finishes simultaneously
    - Each plant processes half of the total volume

    Prove that the total number of days required to complete the task is 2a
-/
theorem milk_processing_time (a m : ℝ) (a_pos : 0 < a) (m_pos : 0 < m) : 
  ∃ (n : ℝ), n > 0 ∧ 
  (∃ (x : ℝ), x > 0 ∧ 
    (n * x = (n - a) * (x + m)) ∧ 
    (a * x + (5 * a / 9) * (2 * x + m) = 2 / 3) ∧
    (n * x = 1 / 2)) ∧
  n = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_milk_processing_time_l3478_347875


namespace NUMINAMATH_CALUDE_temperature_change_l3478_347841

/-- The temperature change problem -/
theorem temperature_change (initial temp_rise temp_drop final : Int) : 
  initial = -12 → 
  temp_rise = 8 → 
  temp_drop = 10 → 
  final = initial + temp_rise - temp_drop → 
  final = -14 := by sorry

end NUMINAMATH_CALUDE_temperature_change_l3478_347841


namespace NUMINAMATH_CALUDE_product_sum_inequality_l3478_347814

theorem product_sum_inequality (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end NUMINAMATH_CALUDE_product_sum_inequality_l3478_347814


namespace NUMINAMATH_CALUDE_binary_sum_equals_decimal_l3478_347809

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_sum_equals_decimal : 
  let binary1 := [true, false, true, false, true, false, true]  -- 1010101₂
  let binary2 := [false, false, false, true, true, true]        -- 111000₂
  binaryToDecimal binary1 + binaryToDecimal binary2 = 141 := by
sorry

end NUMINAMATH_CALUDE_binary_sum_equals_decimal_l3478_347809


namespace NUMINAMATH_CALUDE_sum_remainder_three_l3478_347879

theorem sum_remainder_three (n : ℤ) : (5 - n + (n + 4)) % 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_three_l3478_347879


namespace NUMINAMATH_CALUDE_ball_max_height_l3478_347856

/-- The height function of the ball's trajectory -/
def f (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the ball -/
theorem ball_max_height : ∃ (t : ℝ), ∀ (s : ℝ), f s ≤ f t ∧ f t = 161 := by
  sorry

end NUMINAMATH_CALUDE_ball_max_height_l3478_347856


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3478_347811

theorem trigonometric_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (10 * π / 180) = 2 * (3 * Real.sqrt 3 + 4) / 9 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3478_347811


namespace NUMINAMATH_CALUDE_bible_yellow_tickets_l3478_347825

-- Define the conversion rates
def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10

-- Define Tom's current tickets
def tom_yellow : ℕ := 8
def tom_red : ℕ := 3
def tom_blue : ℕ := 7

-- Define the additional blue tickets needed
def additional_blue : ℕ := 163

-- Define the function to calculate the total blue tickets equivalent
def total_blue_equivalent (yellow red blue : ℕ) : ℕ :=
  yellow * yellow_to_red * red_to_blue + red * red_to_blue + blue

-- Theorem statement
theorem bible_yellow_tickets :
  ∃ (required_yellow : ℕ),
    required_yellow = 10 ∧
    total_blue_equivalent tom_yellow tom_red tom_blue + additional_blue =
    required_yellow * yellow_to_red * red_to_blue :=
by sorry

end NUMINAMATH_CALUDE_bible_yellow_tickets_l3478_347825


namespace NUMINAMATH_CALUDE_product_sum_equality_l3478_347871

theorem product_sum_equality : ∃ (p q r s : ℝ),
  (∀ x, (4 * x^2 - 6 * x + 5) * (8 - 3 * x) = p * x^3 + q * x^2 + r * x + s) →
  8 * p + 4 * q + 2 * r + s = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_equality_l3478_347871


namespace NUMINAMATH_CALUDE_cube_root_negative_eight_properties_l3478_347897

-- Define the cube root function for real numbers
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Main theorem
theorem cube_root_negative_eight_properties :
  let y := cubeRoot (-8)
  ∃ (z : ℝ),
    -- Statement A: y represents the cube root of -8
    z^3 = -8 ∧
    -- Statement B: y results in -2
    y = -2 ∧
    -- Statement C: y is equal to -cubeRoot(8)
    y = -(cubeRoot 8) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_negative_eight_properties_l3478_347897


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_15_l3478_347884

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression (α : Type*) [Add α] [SMul ℕ α] where
  a₁ : α
  d : α

variable {α : Type*} [LinearOrderedField α]

def term (ap : ArithmeticProgression α) (n : ℕ) : α :=
  ap.a₁ + (n - 1) • ap.d

def sum_n_terms (ap : ArithmeticProgression α) (n : ℕ) : α :=
  (n : α) * (ap.a₁ + term ap n) / 2

theorem arithmetic_progression_sum_15
  (ap : ArithmeticProgression α)
  (h_sum : term ap 3 + term ap 9 = 6)
  (h_prod : term ap 3 * term ap 9 = 135 / 16) :
  sum_n_terms ap 15 = 37.5 ∨ sum_n_terms ap 15 = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_15_l3478_347884


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3478_347851

theorem solution_set_of_inequality (x : ℝ) :
  (((1 : ℝ) / Real.pi) ^ (-x + 1) > ((1 : ℝ) / Real.pi) ^ (x^2 - x)) ↔ (x > 1 ∨ x < -1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3478_347851


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l3478_347836

theorem quadratic_roots_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 2*m = 0 ∧ x₂^2 + 2*x₂ + 2*m = 0) → m < (1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l3478_347836


namespace NUMINAMATH_CALUDE_white_bar_dimensions_l3478_347872

/-- Represents the dimensions of a bar in the cube -/
structure BarDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- The cube assembled from bars -/
structure Cube where
  edge_length : ℚ
  bar_count : ℕ
  gray_bar : BarDimensions
  white_bar : BarDimensions

/-- Theorem stating the dimensions of the white bar in the cube -/
theorem white_bar_dimensions (c : Cube) : 
  c.edge_length = 1 ∧ 
  c.bar_count = 8 ∧ 
  (c.gray_bar.length * c.gray_bar.width * c.gray_bar.height = 
   c.white_bar.length * c.white_bar.width * c.white_bar.height) →
  c.white_bar = ⟨7/10, 1/2, 5/14⟩ := by
  sorry

end NUMINAMATH_CALUDE_white_bar_dimensions_l3478_347872


namespace NUMINAMATH_CALUDE_C_is_rotated_X_l3478_347818

/-- A shape in a 2D plane -/
structure Shape :=
  (points : Set (ℝ × ℝ))

/-- Rotation of a shape by 90 degrees clockwise around its center -/
def rotate90 (s : Shape) : Shape := sorry

/-- Two shapes are superimposable if they have the same set of points -/
def superimposable (s1 s2 : Shape) : Prop :=
  s1.points = s2.points

/-- The original shape X -/
def X : Shape := sorry

/-- The alternative shapes -/
def A : Shape := sorry
def B : Shape := sorry
def C : Shape := sorry
def D : Shape := sorry
def E : Shape := sorry

/-- The theorem stating that C is the only shape superimposable with X after rotation -/
theorem C_is_rotated_X : 
  superimposable (rotate90 X) C ∧ 
  (¬superimposable (rotate90 X) A ∧
   ¬superimposable (rotate90 X) B ∧
   ¬superimposable (rotate90 X) D ∧
   ¬superimposable (rotate90 X) E) :=
sorry

end NUMINAMATH_CALUDE_C_is_rotated_X_l3478_347818


namespace NUMINAMATH_CALUDE_sine_cosine_sum_l3478_347881

theorem sine_cosine_sum (α : ℝ) (h : Real.sin (α - π/6) = 1/3) :
  Real.sin (2*α - π/6) + Real.cos (2*α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_sum_l3478_347881


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3478_347887

/-- A triangle with specific area and angles has a specific perimeter -/
theorem triangle_perimeter (A B C : ℝ) (h_area : A = 3 - Real.sqrt 3)
    (h_angle1 : B = 45 * π / 180) (h_angle2 : C = 60 * π / 180) (h_angle3 : A = 75 * π / 180) :
  let perimeter := Real.sqrt 2 * (3 + 2 * Real.sqrt 3 - Real.sqrt 6)
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = perimeter ∧
    (1/2) * a * b * Real.sin C = 3 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3478_347887


namespace NUMINAMATH_CALUDE_rudy_typing_speed_l3478_347867

def team_size : ℕ := 5
def team_average : ℕ := 80
def joyce_speed : ℕ := 76
def gladys_speed : ℕ := 91
def lisa_speed : ℕ := 80
def mike_speed : ℕ := 89

theorem rudy_typing_speed :
  ∃ (rudy_speed : ℕ),
    rudy_speed = team_size * team_average - (joyce_speed + gladys_speed + lisa_speed + mike_speed) :=
by sorry

end NUMINAMATH_CALUDE_rudy_typing_speed_l3478_347867


namespace NUMINAMATH_CALUDE_production_days_calculation_l3478_347892

theorem production_days_calculation (n : ℕ) : 
  (70 * n + 90 = 75 * (n + 1)) → n = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_production_days_calculation_l3478_347892


namespace NUMINAMATH_CALUDE_fraction_equality_l3478_347816

theorem fraction_equality : (3 : ℚ) / (1 + 3 / 5) = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3478_347816


namespace NUMINAMATH_CALUDE_benny_stored_bales_l3478_347894

/-- The number of bales Benny stored in the barn -/
def bales_stored (initial_bales current_bales : ℕ) : ℕ :=
  current_bales - initial_bales

/-- Theorem stating that Benny stored 35 bales in the barn -/
theorem benny_stored_bales : 
  let initial_bales : ℕ := 47
  let current_bales : ℕ := 82
  bales_stored initial_bales current_bales = 35 := by
sorry

end NUMINAMATH_CALUDE_benny_stored_bales_l3478_347894


namespace NUMINAMATH_CALUDE_parallel_line_slope_l3478_347824

/-- Given a line with equation 5x - 3y = 21, prove that the slope of any parallel line is 5/3 -/
theorem parallel_line_slope (x y : ℝ) :
  (5 * x - 3 * y = 21) → 
  (∃ (m : ℝ), m = 5 / 3 ∧ ∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ → 
    (5 * x₁ - 3 * y₁ = 21 ∧ 5 * x₂ - 3 * y₂ = 21) → 
    (y₂ - y₁) / (x₂ - x₁) = m) :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_slope_l3478_347824


namespace NUMINAMATH_CALUDE_current_velocity_l3478_347869

theorem current_velocity (rowing_speed : ℝ) (distance : ℝ) (total_time : ℝ) :
  rowing_speed = 10 ∧ distance = 48 ∧ total_time = 10 →
  ∃ v : ℝ, v = 2 ∧ 
    distance / (rowing_speed - v) + distance / (rowing_speed + v) = total_time :=
by sorry

end NUMINAMATH_CALUDE_current_velocity_l3478_347869


namespace NUMINAMATH_CALUDE_cube_eight_eq_two_power_ten_unique_solution_l3478_347817

theorem cube_eight_eq_two_power_ten :
  8^3 + 8^3 + 8^3 = 2^10 := by
sorry

theorem unique_solution (x : ℕ) :
  8^3 + 8^3 + 8^3 = 2^x → x = 10 := by
sorry

end NUMINAMATH_CALUDE_cube_eight_eq_two_power_ten_unique_solution_l3478_347817


namespace NUMINAMATH_CALUDE_max_additional_plates_l3478_347833

/-- Represents a set of letters for license plates --/
def LetterSet := List Char

/-- Calculate the number of license plates given three sets of letters --/
def calculatePlates (set1 set2 set3 : LetterSet) : Nat :=
  set1.length * set2.length * set3.length

/-- The initial sets of letters --/
def initialSet1 : LetterSet := ['C', 'H', 'L', 'P', 'R', 'S']
def initialSet2 : LetterSet := ['A', 'I', 'O', 'U']
def initialSet3 : LetterSet := ['D', 'M', 'N', 'T', 'V']

/-- The number of new letters to be added --/
def newLettersCount : Nat := 3

/-- Theorem: The maximum number of additional license plates is 96 --/
theorem max_additional_plates :
  (∀ newSet1 newSet2 newSet3 : LetterSet,
    newSet1.length + newSet2.length + newSet3.length = initialSet1.length + initialSet2.length + initialSet3.length + newLettersCount →
    calculatePlates newSet1 newSet2 newSet3 - calculatePlates initialSet1 initialSet2 initialSet3 ≤ 96) ∧
  (∃ newSet1 newSet2 newSet3 : LetterSet,
    newSet1.length + newSet2.length + newSet3.length = initialSet1.length + initialSet2.length + initialSet3.length + newLettersCount ∧
    calculatePlates newSet1 newSet2 newSet3 - calculatePlates initialSet1 initialSet2 initialSet3 = 96) := by
  sorry


end NUMINAMATH_CALUDE_max_additional_plates_l3478_347833


namespace NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l3478_347840

theorem three_digit_multiples_of_seven (n : ℕ) : 
  (∃ k : ℕ, n = 7 * k ∧ 100 ≤ n ∧ n ≤ 999) ↔ n ∈ Finset.range 128 ∧ n ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_three_digit_multiples_of_seven_l3478_347840


namespace NUMINAMATH_CALUDE_onion_harvest_weight_l3478_347886

-- Define the number of bags per trip
def bags_per_trip : ℕ := 10

-- Define the weight of each bag in kg
def weight_per_bag : ℕ := 50

-- Define the number of trips
def number_of_trips : ℕ := 20

-- Define the total weight of onions harvested
def total_weight : ℕ := bags_per_trip * weight_per_bag * number_of_trips

-- Theorem statement
theorem onion_harvest_weight :
  total_weight = 10000 := by sorry

end NUMINAMATH_CALUDE_onion_harvest_weight_l3478_347886


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3478_347859

/-- A quadratic function -/
noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_properties (a b c : ℝ) :
  (∀ p q : ℝ, p ≠ q → f a b c p = f a b c q → f a b c (p + q) = c) ∧
  (∀ p q : ℝ, p ≠ q → f a b c (p + q) = c → (p + q = 0 ∨ f a b c p = f a b c q)) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3478_347859


namespace NUMINAMATH_CALUDE_perpendicular_projection_vector_l3478_347899

/-- Two-dimensional vector -/
structure Vec2 where
  x : ℝ
  y : ℝ

/-- Line represented by a point and a direction vector -/
structure Line where
  point : Vec2
  dir : Vec2

def l : Line :=
  { point := { x := 2, y := 5 }
    dir := { x := 3, y := 2 } }

def m : Line :=
  { point := { x := 1, y := 3 }
    dir := { x := 2, y := 2 } }

def v : Vec2 :=
  { x := 1, y := -1 }

theorem perpendicular_projection_vector :
  (v.x * m.dir.x + v.y * m.dir.y = 0) ∧
  (2 * v.x - v.y = 3) := by sorry

end NUMINAMATH_CALUDE_perpendicular_projection_vector_l3478_347899


namespace NUMINAMATH_CALUDE_dice_divisible_by_seven_l3478_347834

/-- A die is represented as a function from face index to digit -/
def Die := Fin 6 → Fin 6

/-- Property that opposite faces of a die sum to 7 -/
def OppositeFacesSum7 (d : Die) : Prop :=
  ∀ i : Fin 3, d i + d (i + 3) = 7

/-- A set of six dice -/
def DiceSet := Fin 6 → Die

/-- Property that all dice in a set have opposite faces summing to 7 -/
def AllDiceOppositeFacesSum7 (ds : DiceSet) : Prop :=
  ∀ i : Fin 6, OppositeFacesSum7 (ds i)

/-- A configuration of dice is a function from die position to face index -/
def DiceConfiguration := Fin 6 → Fin 6

/-- The number formed by a dice configuration -/
def NumberFormed (ds : DiceSet) (dc : DiceConfiguration) : ℕ :=
  (ds 0 (dc 0)) * 100000 + (ds 1 (dc 1)) * 10000 + (ds 2 (dc 2)) * 1000 +
  (ds 3 (dc 3)) * 100 + (ds 4 (dc 4)) * 10 + (ds 5 (dc 5))

theorem dice_divisible_by_seven (ds : DiceSet) (h : AllDiceOppositeFacesSum7 ds) :
  ∃ dc : DiceConfiguration, NumberFormed ds dc % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_dice_divisible_by_seven_l3478_347834


namespace NUMINAMATH_CALUDE_constant_term_value_l3478_347857

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The sum of the first two binomial coefficients equals 10 -/
def sum_first_two_coefficients (n : ℕ) : Prop :=
  binomial n 0 + binomial n 1 = 10

/-- The constant term in the expansion -/
def constant_term (n : ℕ) : ℕ :=
  2^(n - 6) * binomial n 6

theorem constant_term_value (n : ℕ) :
  sum_first_two_coefficients n → constant_term n = 672 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l3478_347857


namespace NUMINAMATH_CALUDE_adams_earnings_l3478_347826

theorem adams_earnings (daily_wage : ℝ) (tax_rate : ℝ) (work_days : ℕ) :
  daily_wage = 40 →
  tax_rate = 0.1 →
  work_days = 30 →
  (daily_wage * (1 - tax_rate) * work_days : ℝ) = 1080 := by
  sorry

end NUMINAMATH_CALUDE_adams_earnings_l3478_347826


namespace NUMINAMATH_CALUDE_person_a_silver_cards_l3478_347828

/-- Represents the number of sheets of each type of card paper -/
structure CardPapers :=
  (red : ℕ)
  (gold : ℕ)
  (silver : ℕ)

/-- Represents the exchange rates between different types of card papers -/
structure ExchangeRates :=
  (red_to_gold : ℕ × ℕ)
  (gold_to_red_and_silver : ℕ × ℕ × ℕ)

/-- Function to perform exchanges and calculate the maximum number of silver cards obtainable -/
def max_silver_obtainable (initial : CardPapers) (rates : ExchangeRates) : ℕ :=
  sorry

/-- Theorem stating that person A can obtain 7 sheets of silver card paper -/
theorem person_a_silver_cards :
  let initial := CardPapers.mk 3 3 0
  let rates := ExchangeRates.mk (5, 2) (1, 1, 1)
  max_silver_obtainable initial rates = 7 :=
sorry

end NUMINAMATH_CALUDE_person_a_silver_cards_l3478_347828


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3478_347801

theorem inequality_solution_set (x : ℝ) : 4 * x - 2 ≤ 3 * (x - 1) ↔ x ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3478_347801


namespace NUMINAMATH_CALUDE_parkway_soccer_players_l3478_347838

theorem parkway_soccer_players (total_students : ℕ) (boys : ℕ) (girls_not_playing : ℕ) :
  total_students = 420 →
  boys = 312 →
  girls_not_playing = 53 →
  ∃ (soccer_players : ℕ),
    soccer_players = 250 ∧
    (soccer_players : ℚ) * (78 / 100) = boys - (total_students - boys - girls_not_playing) :=
by sorry

end NUMINAMATH_CALUDE_parkway_soccer_players_l3478_347838


namespace NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l3478_347839

theorem x_squared_plus_y_squared_equals_four (x y : ℝ) :
  (x^2 + y^2 + 2) * (x^2 + y^2 - 3) = 6 → x^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_y_squared_equals_four_l3478_347839


namespace NUMINAMATH_CALUDE_perimeter_ratio_after_folding_and_cutting_l3478_347837

theorem perimeter_ratio_after_folding_and_cutting (s : ℝ) (h : s > 0) :
  let original_square_perimeter := 4 * s
  let small_rectangle_perimeter := 2 * (s / 2 + s / 4)
  small_rectangle_perimeter / original_square_perimeter = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_perimeter_ratio_after_folding_and_cutting_l3478_347837


namespace NUMINAMATH_CALUDE_sequence_inequality_l3478_347848

/-- A sequence of positive real numbers satisfying the given inequality -/
def PositiveSequence (a : ℕ → ℝ) : Prop :=
  ∀ i : ℕ, i > 0 → a i > 0 ∧ i * (a i)^2 ≥ (i + 1) * (a (i - 1)) * (a (i + 1))

/-- Definition of the sequence b in terms of a -/
def b (a : ℕ → ℝ) (x y : ℝ) : ℕ → ℝ :=
  λ i => x * (a i) + y * (a (i - 1))

theorem sequence_inequality (a : ℕ → ℝ) (x y : ℝ) 
    (h_pos : PositiveSequence a) (h_x : x > 0) (h_y : y > 0) :
    ∀ i : ℕ, i ≥ 2 → i * (b a x y i)^2 > (i + 1) * (b a x y (i - 1)) * (b a x y (i + 1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3478_347848


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_ratio_l3478_347847

/-- An obtuse triangle with sides a, b, and c, where a is the longest side -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  a_longest : a ≥ b ∧ a ≥ c
  obtuse : a^2 > b^2 + c^2

/-- The ratio of the sum of squares of two shorter sides to the square of the longest side
    in an obtuse triangle is always greater than or equal to 1/2 -/
theorem obtuse_triangle_side_ratio (t : ObtuseTriangle) :
  (t.b^2 + t.c^2) / t.a^2 ≥ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_ratio_l3478_347847


namespace NUMINAMATH_CALUDE_book_pages_calculation_l3478_347829

/-- Given a book with a specific number of chapters and pages per chapter, 
    calculate the total number of pages in the book. -/
theorem book_pages_calculation (num_chapters : ℕ) (pages_per_chapter : ℕ) 
    (h1 : num_chapters = 31) (h2 : pages_per_chapter = 61) : 
    num_chapters * pages_per_chapter = 1891 := by
  sorry

#check book_pages_calculation

end NUMINAMATH_CALUDE_book_pages_calculation_l3478_347829


namespace NUMINAMATH_CALUDE_f_difference_l3478_347858

/-- Sum of positive divisors of n -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Function f(n) defined as the sum of all positive divisors of n divided by n -/
def f (n : ℕ+) : ℚ := (sigma n : ℚ) / n

/-- Theorem stating that f(640) - f(320) = 3/320 -/
theorem f_difference : f 640 - f 320 = 3 / 320 := by sorry

end NUMINAMATH_CALUDE_f_difference_l3478_347858


namespace NUMINAMATH_CALUDE_f_has_zero_in_interval_l3478_347844

def f (x : ℝ) := 3 * x - x^2

theorem f_has_zero_in_interval :
  ∃ x : ℝ, x ∈ Set.Icc (-1) 0 ∧ f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_has_zero_in_interval_l3478_347844


namespace NUMINAMATH_CALUDE_smallest_bench_configuration_l3478_347854

theorem smallest_bench_configuration (adults_per_bench children_per_bench : ℕ) 
  (adults_per_bench_pos : adults_per_bench > 0)
  (children_per_bench_pos : children_per_bench > 0)
  (adults_per_bench_def : adults_per_bench = 9)
  (children_per_bench_def : children_per_bench = 15) :
  ∃ (M : ℕ), M > 0 ∧ M * adults_per_bench = M * children_per_bench ∧
  ∀ (N : ℕ), N > 0 → N * adults_per_bench = N * children_per_bench → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_smallest_bench_configuration_l3478_347854


namespace NUMINAMATH_CALUDE_initial_water_percentage_in_milk_l3478_347861

/-- The initial percentage of water in milk, given that adding 15 liters of pure milk to 10 liters
    of the initial milk reduces the water content to 2%. -/
theorem initial_water_percentage_in_milk :
  ∀ (initial_water_percentage : ℝ),
    (initial_water_percentage ≥ 0) →
    (initial_water_percentage ≤ 100) →
    (10 * (100 - initial_water_percentage) / 100 + 15 = 0.98 * 25) →
    initial_water_percentage = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_percentage_in_milk_l3478_347861


namespace NUMINAMATH_CALUDE_binomial_17_5_l3478_347890

theorem binomial_17_5 (h1 : Nat.choose 15 3 = 455)
                      (h2 : Nat.choose 15 4 = 1365)
                      (h3 : Nat.choose 15 5 = 3003) :
  Nat.choose 17 5 = 6188 := by
  sorry

end NUMINAMATH_CALUDE_binomial_17_5_l3478_347890


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3478_347882

def A : Set ℝ := {x | x^2 - x ≤ 0}
def B : Set ℝ := {x | 2*x - 1 > 0}

theorem intersection_of_A_and_B : A ∩ B = Set.Ioo (1/2) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3478_347882


namespace NUMINAMATH_CALUDE_andrew_sandwiches_l3478_347888

/-- The number of friends coming over to Andrew's game night. -/
def num_friends : ℕ := 4

/-- The number of sandwiches Andrew made for each friend. -/
def sandwiches_per_friend : ℕ := 3

/-- The total number of sandwiches Andrew made. -/
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

/-- Theorem stating that the total number of sandwiches Andrew made is 12. -/
theorem andrew_sandwiches : total_sandwiches = 12 := by
  sorry

end NUMINAMATH_CALUDE_andrew_sandwiches_l3478_347888


namespace NUMINAMATH_CALUDE_at_least_three_babies_speak_l3478_347843

def probability_baby_speaks : ℚ := 1 / 3

def number_of_babies : ℕ := 6

def probability_at_least_three_speak (p : ℚ) (n : ℕ) : ℚ :=
  1 - (Nat.choose n 0 * (1 - p)^n + 
       Nat.choose n 1 * p * (1 - p)^(n-1) + 
       Nat.choose n 2 * p^2 * (1 - p)^(n-2))

theorem at_least_three_babies_speak : 
  probability_at_least_three_speak probability_baby_speaks number_of_babies = 353 / 729 := by
  sorry

end NUMINAMATH_CALUDE_at_least_three_babies_speak_l3478_347843


namespace NUMINAMATH_CALUDE_min_value_quadratic_l3478_347806

theorem min_value_quadratic : 
  ∀ x : ℝ, 3 * x^2 - 12 * x + 908 ≥ 896 ∧ 
  ∃ x₀ : ℝ, 3 * x₀^2 - 12 * x₀ + 908 = 896 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l3478_347806


namespace NUMINAMATH_CALUDE_sqrt_three_comparison_l3478_347823

theorem sqrt_three_comparison : 2 * Real.sqrt 3 > 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_comparison_l3478_347823


namespace NUMINAMATH_CALUDE_process_600_parts_l3478_347863

/-- The regression line equation for processing parts -/
def regression_line (x : ℝ) : ℝ := 0.01 * x + 0.5

/-- Theorem stating that processing 600 parts takes 6.5 hours -/
theorem process_600_parts : regression_line 600 = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_process_600_parts_l3478_347863


namespace NUMINAMATH_CALUDE_translation_theorem_l3478_347896

/-- Represents a point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in the 2D Cartesian coordinate system -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def applyTranslation (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

/-- Theorem: Given points A, B, and C, where AB is translated to CD,
    prove that D has the correct coordinates -/
theorem translation_theorem (A B C : Point)
    (h1 : A = { x := -1, y := 0 })
    (h2 : B = { x := 1, y := 2 })
    (h3 : C = { x := 1, y := -2 }) :
  let t : Translation := { dx := C.x - A.x, dy := C.y - A.y }
  let D : Point := applyTranslation B t
  D = { x := 3, y := 0 } := by
  sorry


end NUMINAMATH_CALUDE_translation_theorem_l3478_347896


namespace NUMINAMATH_CALUDE_intersection_A_B_l3478_347866

-- Define set A
def A : Set ℝ := {x | x^2 - 3*x < 0}

-- Define set B
def B : Set ℝ := {x | x > 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ioo 2 3 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3478_347866


namespace NUMINAMATH_CALUDE_closest_to_300_l3478_347893

def expression : ℝ := 3.25 * 9.252 * (6.22 + 3.78) - 10

def options : List ℝ := [250, 300, 350, 400, 450]

theorem closest_to_300 : 
  ∀ x ∈ options, x ≠ 300 → |expression - 300| < |expression - x| :=
sorry

end NUMINAMATH_CALUDE_closest_to_300_l3478_347893


namespace NUMINAMATH_CALUDE_grain_equations_correct_l3478_347830

/-- Represents the amount of grain in sheng that one bundle can produce -/
structure GrainBundle where
  amount : ℝ

/-- High-quality grain bundle -/
def high_quality : GrainBundle := sorry

/-- Low-quality grain bundle -/
def low_quality : GrainBundle := sorry

/-- Theorem stating that the system of equations correctly represents the grain problem -/
theorem grain_equations_correct :
  (5 * high_quality.amount - 11 = 7 * low_quality.amount) ∧
  (7 * high_quality.amount - 25 = 5 * low_quality.amount) := by
  sorry

end NUMINAMATH_CALUDE_grain_equations_correct_l3478_347830


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l3478_347868

theorem rectangle_area_difference : 
  let rect1_width : ℕ := 4
  let rect1_height : ℕ := 5
  let rect2_width : ℕ := 3
  let rect2_height : ℕ := 6
  let rect1_area := rect1_width * rect1_height
  let rect2_area := rect2_width * rect2_height
  rect1_area - rect2_area = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l3478_347868
