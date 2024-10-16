import Mathlib

namespace NUMINAMATH_CALUDE_five_segments_max_regions_l531_53148

/-- The maximum number of regions formed by n line segments in a plane -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of regions formed by 5 line segments in a plane is 16 -/
theorem five_segments_max_regions : max_regions 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_five_segments_max_regions_l531_53148


namespace NUMINAMATH_CALUDE_charts_brought_by_associate_prof_l531_53179

/-- Represents the number of charts brought by each associate professor -/
def charts_per_associate_prof : ℕ := sorry

/-- Represents the number of associate professors -/
def num_associate_profs : ℕ := sorry

/-- Represents the number of assistant professors -/
def num_assistant_profs : ℕ := sorry

theorem charts_brought_by_associate_prof :
  (2 * num_associate_profs + num_assistant_profs = 7) →
  (charts_per_associate_prof * num_associate_profs + 2 * num_assistant_profs = 11) →
  (num_associate_profs + num_assistant_profs = 6) →
  charts_per_associate_prof = 1 := by
    sorry

#check charts_brought_by_associate_prof

end NUMINAMATH_CALUDE_charts_brought_by_associate_prof_l531_53179


namespace NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l531_53193

theorem inverse_proportion_y_relationship (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ > x₂ → x₂ > 0 → y₁ = -3 / x₁ → y₂ = -3 / x₂ → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_y_relationship_l531_53193


namespace NUMINAMATH_CALUDE_function_value_at_negative_l531_53187

theorem function_value_at_negative (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = x^5 + x^3 + 1) →
  f m = 10 →
  f (-m) = -8 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_l531_53187


namespace NUMINAMATH_CALUDE_fifth_hexagon_dots_l531_53171

/-- The number of dots on each side of a hexagon layer -/
def dots_per_side (n : ℕ) : ℕ := n + 2

/-- The total number of dots in a single layer of a hexagon -/
def dots_in_layer (n : ℕ) : ℕ := 6 * (dots_per_side n)

/-- The total number of dots in a hexagon with n layers -/
def total_dots (n : ℕ) : ℕ := 
  if n = 0 then 0
  else total_dots (n - 1) + dots_in_layer n

/-- The fifth hexagon has 150 dots -/
theorem fifth_hexagon_dots : total_dots 5 = 150 := by
  sorry


end NUMINAMATH_CALUDE_fifth_hexagon_dots_l531_53171


namespace NUMINAMATH_CALUDE_soccer_team_starters_l531_53147

theorem soccer_team_starters (n : ℕ) (q : ℕ) (s : ℕ) (qc : ℕ) :
  n = 15 →  -- Total number of players
  q = 4 →   -- Number of quadruplets
  s = 7 →   -- Number of starters
  qc = 2 →  -- Number of quadruplets in starting lineup
  (Nat.choose q qc) * (Nat.choose (n - q) (s - qc)) = 2772 := by
sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l531_53147


namespace NUMINAMATH_CALUDE_negation_of_every_prime_is_odd_l531_53140

theorem negation_of_every_prime_is_odd :
  (¬ ∀ p : ℕ, Prime p → Odd p) ↔ (∃ p : ℕ, Prime p ∧ ¬ Odd p) :=
sorry

end NUMINAMATH_CALUDE_negation_of_every_prime_is_odd_l531_53140


namespace NUMINAMATH_CALUDE_tan_sum_equals_one_l531_53186

theorem tan_sum_equals_one (α β : ℝ) 
  (h1 : Real.tan (α - π / 6) = 3 / 7)
  (h2 : Real.tan (π / 6 + β) = 2 / 5) :
  Real.tan (α + β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_equals_one_l531_53186


namespace NUMINAMATH_CALUDE_estimate_students_in_range_l531_53165

/-- Given a histogram of student heights with two adjacent rectangles, 
    estimate the number of students in the combined range. -/
theorem estimate_students_in_range 
  (total_students : ℕ) 
  (rectangle_width : ℝ) 
  (height_a : ℝ) 
  (height_b : ℝ) 
  (h_total : total_students = 1500)
  (h_width : rectangle_width = 5) :
  (rectangle_width * height_a + rectangle_width * height_b) * total_students = 
    7500 * (height_a + height_b) := by
  sorry

end NUMINAMATH_CALUDE_estimate_students_in_range_l531_53165


namespace NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l531_53166

theorem smallest_solution_absolute_value_equation :
  let f : ℝ → ℝ := λ x => x * |x| - 3 * x + 2
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → x ≤ y ∧ x = (-3 - Real.sqrt 17) / 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_solution_absolute_value_equation_l531_53166


namespace NUMINAMATH_CALUDE_pipe_length_difference_l531_53120

theorem pipe_length_difference (total_length shorter_length : ℕ) : 
  total_length = 68 → 
  shorter_length = 28 → 
  shorter_length < total_length - shorter_length →
  total_length - shorter_length - shorter_length = 12 :=
by sorry

end NUMINAMATH_CALUDE_pipe_length_difference_l531_53120


namespace NUMINAMATH_CALUDE_weight_difference_is_one_black_dog_weight_conditions_are_consistent_l531_53109

-- Define the weights of the dogs
def brown_weight : ℝ := 4
def black_weight : ℝ := 5  -- This is derived from the solution, but we'll prove it
def white_weight : ℝ := 2 * brown_weight
def grey_weight : ℝ := black_weight - 2

-- Define the average weight
def average_weight : ℝ := 5

-- Define the number of dogs
def num_dogs : ℕ := 4

-- Theorem to prove
theorem weight_difference_is_one :
  black_weight - brown_weight = 1 :=
by
  -- The proof would go here
  sorry

-- Additional theorem to prove the black dog's weight
theorem black_dog_weight :
  black_weight = 5 :=
by
  -- This proof would use the given conditions to show that black_weight must be 5
  sorry

-- Theorem to show that the conditions are consistent
theorem conditions_are_consistent :
  (brown_weight + black_weight + white_weight + grey_weight) / num_dogs = average_weight :=
by
  -- This proof would show that the given weights satisfy the average weight condition
  sorry

end NUMINAMATH_CALUDE_weight_difference_is_one_black_dog_weight_conditions_are_consistent_l531_53109


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_zero_minus_three_inv_l531_53118

theorem sqrt_two_minus_one_zero_minus_three_inv :
  (Real.sqrt 2 - 1) ^ 0 - 3⁻¹ = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_zero_minus_three_inv_l531_53118


namespace NUMINAMATH_CALUDE_absolute_value_w_l531_53133

theorem absolute_value_w (s : ℝ) (w : ℂ) (h1 : |s| < 3) (h2 : w + 2 / w = s) : 
  Complex.abs w = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_w_l531_53133


namespace NUMINAMATH_CALUDE_tan_alpha_value_l531_53151

theorem tan_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin (2 * α) + 2 * Real.cos (2 * α) = 2) : 
  Real.tan α = 1/2 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l531_53151


namespace NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l531_53176

theorem half_plus_five_equals_eleven : 
  (12 / 2 : ℚ) + 5 = 11 := by sorry

end NUMINAMATH_CALUDE_half_plus_five_equals_eleven_l531_53176


namespace NUMINAMATH_CALUDE_function_existence_l531_53178

theorem function_existence (A B : Type) [Fintype A] [Fintype B]
  (hA : Fintype.card A = 2011^2) (hB : Fintype.card B = 2010) :
  ∃ f : A × A → B,
    (∀ x y : A, f (x, y) = f (y, x)) ∧
    (∀ g : A → B, ∃ a₁ a₂ : A, a₁ ≠ a₂ ∧ g a₁ = f (a₁, a₂) ∧ f (a₁, a₂) = g a₂) := by
  sorry

end NUMINAMATH_CALUDE_function_existence_l531_53178


namespace NUMINAMATH_CALUDE_best_fit_highest_r_squared_model2_best_fit_l531_53146

/-- Represents a regression model with its coefficient of determination (R²) -/
structure RegressionModel where
  name : String
  r_squared : Real

/-- Determines if a model has the best fit among a list of models -/
def has_best_fit (model : RegressionModel) (models : List RegressionModel) : Prop :=
  ∀ m ∈ models, model.r_squared ≥ m.r_squared

theorem best_fit_highest_r_squared (models : List RegressionModel) 
  (h_nonempty : models ≠ []) : 
  ∃ best_model ∈ models, has_best_fit best_model models :=
sorry

def model1 : RegressionModel := ⟨"Model 1", 0.75⟩
def model2 : RegressionModel := ⟨"Model 2", 0.90⟩
def model3 : RegressionModel := ⟨"Model 3", 0.45⟩
def model4 : RegressionModel := ⟨"Model 4", 0.65⟩

def all_models : List RegressionModel := [model1, model2, model3, model4]

theorem model2_best_fit : has_best_fit model2 all_models :=
sorry

end NUMINAMATH_CALUDE_best_fit_highest_r_squared_model2_best_fit_l531_53146


namespace NUMINAMATH_CALUDE_purple_jellybeans_count_l531_53196

theorem purple_jellybeans_count (total : ℕ) (blue : ℕ) (orange : ℕ) (red : ℕ) 
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_orange : orange = 40)
  (h_red : red = 120) :
  total - (blue + orange + red) = 26 := by
  sorry

end NUMINAMATH_CALUDE_purple_jellybeans_count_l531_53196


namespace NUMINAMATH_CALUDE_student_calculation_error_l531_53101

theorem student_calculation_error (x : ℤ) : 
  (x + 5) - (x - (-5)) = 10 :=
sorry

end NUMINAMATH_CALUDE_student_calculation_error_l531_53101


namespace NUMINAMATH_CALUDE_barbara_typing_time_l531_53116

/-- Calculates the time needed to type a document given the original typing speed,
    speed decrease, and document length. -/
def typing_time (original_speed : ℕ) (speed_decrease : ℕ) (document_length : ℕ) : ℕ :=
  document_length / (original_speed - speed_decrease)

/-- Proves that given the specific conditions, the typing time is 20 minutes. -/
theorem barbara_typing_time :
  typing_time 212 40 3440 = 20 := by
  sorry

end NUMINAMATH_CALUDE_barbara_typing_time_l531_53116


namespace NUMINAMATH_CALUDE_fraction_equality_l531_53181

theorem fraction_equality (x : ℚ) (c : ℚ) (h1 : c ≠ 0) (h2 : c ≠ 3) :
  (4 + x) / (5 + x) = c / (3 * c) → x = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l531_53181


namespace NUMINAMATH_CALUDE_trig_identity_l531_53161

theorem trig_identity (α : ℝ) : Real.sin α ^ 6 + Real.cos α ^ 6 + 3 * Real.sin α ^ 2 * Real.cos α ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l531_53161


namespace NUMINAMATH_CALUDE_range_of_expression_l531_53197

theorem range_of_expression (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -3/2 < 1/2 * α - β ∧ 1/2 * α - β < 11/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l531_53197


namespace NUMINAMATH_CALUDE_triangle_inequality_l531_53137

theorem triangle_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l531_53137


namespace NUMINAMATH_CALUDE_count_divisible_by_eight_l531_53183

theorem count_divisible_by_eight (n : ℕ) : 
  (150 < n ∧ n ≤ 400 ∧ n % 8 = 0) → 
  (Finset.filter (λ x => 150 < x ∧ x ≤ 400 ∧ x % 8 = 0) (Finset.range 401)).card = 31 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_by_eight_l531_53183


namespace NUMINAMATH_CALUDE_fraction_simplification_l531_53145

theorem fraction_simplification : (2 : ℚ) / 462 + 29 / 42 = 107 / 154 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l531_53145


namespace NUMINAMATH_CALUDE_roots_sum_reciprocals_l531_53157

theorem roots_sum_reciprocals (a b : ℝ) : 
  (a^2 - 3*a - 5 = 0) → 
  (b^2 - 3*b - 5 = 0) → 
  (a ≠ 0) →
  (b ≠ 0) →
  (1/a + 1/b = -3/5) := by
sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocals_l531_53157


namespace NUMINAMATH_CALUDE_num_a_animals_l531_53158

def total_animals : ℕ := 17
def num_b_animals : ℕ := 8

theorem num_a_animals : total_animals - num_b_animals = 9 := by
  sorry

end NUMINAMATH_CALUDE_num_a_animals_l531_53158


namespace NUMINAMATH_CALUDE_angle_of_inclination_negative_sqrt_three_line_l531_53135

theorem angle_of_inclination_negative_sqrt_three_line :
  let line : ℝ → ℝ := λ x ↦ -Real.sqrt 3 * x + 1
  let slope : ℝ := -Real.sqrt 3
  let angle_of_inclination : ℝ := Real.arctan (-Real.sqrt 3)
  (0 ≤ angle_of_inclination) ∧ (angle_of_inclination < π) →
  angle_of_inclination = 2 * π / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_angle_of_inclination_negative_sqrt_three_line_l531_53135


namespace NUMINAMATH_CALUDE_man_walking_distance_l531_53156

theorem man_walking_distance (speed : ℝ) (time : ℝ) : 
  speed > 0 →
  time > 0 →
  (speed + 1/3) * (5/6 * time) = speed * time →
  (speed - 1/3) * (time + 3.5) = speed * time →
  speed * time = 35/96 := by
  sorry

end NUMINAMATH_CALUDE_man_walking_distance_l531_53156


namespace NUMINAMATH_CALUDE_problem_solution_l531_53124

theorem problem_solution (a b : ℝ) 
  (h1 : 2 * a - 1 = 9) 
  (h2 : 3 * a + b - 1 = 16) : 
  a + 2 * b = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l531_53124


namespace NUMINAMATH_CALUDE_total_stickers_is_60_l531_53182

/-- Represents the number of folders --/
def num_folders : Nat := 3

/-- Represents the number of sheets in each folder --/
def sheets_per_folder : Nat := 10

/-- Represents the number of stickers per sheet in the red folder --/
def red_stickers : Nat := 3

/-- Represents the number of stickers per sheet in the green folder --/
def green_stickers : Nat := 2

/-- Represents the number of stickers per sheet in the blue folder --/
def blue_stickers : Nat := 1

/-- Calculates the total number of stickers used --/
def total_stickers : Nat :=
  sheets_per_folder * red_stickers +
  sheets_per_folder * green_stickers +
  sheets_per_folder * blue_stickers

/-- Theorem stating that the total number of stickers used is 60 --/
theorem total_stickers_is_60 : total_stickers = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_is_60_l531_53182


namespace NUMINAMATH_CALUDE_some_ounce_glass_size_l531_53159

/-- Proves that the size of the some-ounce glasses is 5 ounces given the problem conditions. -/
theorem some_ounce_glass_size (total_water : ℕ) (S : ℕ) 
  (h1 : total_water = 122)
  (h2 : 6 * S + 4 * 8 + 15 * 4 = total_water) : S = 5 := by
  sorry

#check some_ounce_glass_size

end NUMINAMATH_CALUDE_some_ounce_glass_size_l531_53159


namespace NUMINAMATH_CALUDE_simplify_expression_l531_53127

theorem simplify_expression : 8 * (15/4) * (-45/50) = -12/25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l531_53127


namespace NUMINAMATH_CALUDE_f_properties_l531_53126

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 12 * x - 10

/-- Theorem stating that f has the required properties -/
theorem f_properties :
  (f 1 = 0) ∧ (f 5 = 0) ∧ (f 3 = 8) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_f_properties_l531_53126


namespace NUMINAMATH_CALUDE_total_routes_bristol_to_carlisle_l531_53173

theorem total_routes_bristol_to_carlisle :
  let bristol_to_birmingham : ℕ := 8
  let birmingham_to_manchester : ℕ := 5
  let manchester_to_sheffield : ℕ := 4
  let sheffield_to_newcastle : ℕ := 3
  let newcastle_to_carlisle : ℕ := 2
  bristol_to_birmingham * birmingham_to_manchester * manchester_to_sheffield * sheffield_to_newcastle * newcastle_to_carlisle = 960 := by
  sorry

end NUMINAMATH_CALUDE_total_routes_bristol_to_carlisle_l531_53173


namespace NUMINAMATH_CALUDE_correction_is_11y_l531_53175

/-- The correction needed when y quarters are mistakenly counted as nickels
    and y pennies are mistakenly counted as dimes -/
def correction (y : ℕ) : ℤ :=
  let quarter_value : ℕ := 25
  let nickel_value : ℕ := 5
  let penny_value : ℕ := 1
  let dime_value : ℕ := 10
  let quarter_nickel_diff : ℕ := quarter_value - nickel_value
  let dime_penny_diff : ℕ := dime_value - penny_value
  (quarter_nickel_diff * y : ℤ) - (dime_penny_diff * y : ℤ)

theorem correction_is_11y (y : ℕ) : correction y = 11 * y :=
  sorry

end NUMINAMATH_CALUDE_correction_is_11y_l531_53175


namespace NUMINAMATH_CALUDE_choir_arrangement_theorem_l531_53130

theorem choir_arrangement_theorem (m : ℕ) : 
  (∃ y : ℕ, m = y^2 + 11) ∧ 
  (∃ n : ℕ, m = n * (n + 5)) ∧ 
  (∀ k : ℕ, (∃ y : ℕ, k = y^2 + 11) ∧ (∃ n : ℕ, k = n * (n + 5)) → k ≤ m) → 
  m = 300 :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_theorem_l531_53130


namespace NUMINAMATH_CALUDE_two_digit_number_exchange_l531_53160

theorem two_digit_number_exchange (A : ℕ) : 
  A < 10 →  -- Ensure A is a single digit
  (10 * A + 2) - (20 + A) = 9 →  -- Condition for digit exchange
  A = 3 := by sorry

end NUMINAMATH_CALUDE_two_digit_number_exchange_l531_53160


namespace NUMINAMATH_CALUDE_mixed_coffee_bag_weight_l531_53162

/-- Proves that the total weight of a mixed coffee bag is 102.8 pounds given specific conditions --/
theorem mixed_coffee_bag_weight 
  (colombian_price : ℝ) 
  (peruvian_price : ℝ) 
  (mixed_price : ℝ) 
  (colombian_weight : ℝ) 
  (h1 : colombian_price = 5.50)
  (h2 : peruvian_price = 4.25)
  (h3 : mixed_price = 4.60)
  (h4 : colombian_weight = 28.8) :
  ∃ (total_weight : ℝ), total_weight = 102.8 ∧ 
  (colombian_price * colombian_weight + peruvian_price * (total_weight - colombian_weight)) / total_weight = mixed_price :=
by
  sorry

#check mixed_coffee_bag_weight

end NUMINAMATH_CALUDE_mixed_coffee_bag_weight_l531_53162


namespace NUMINAMATH_CALUDE_mass_percentage_cl_in_mixture_mass_percentage_cl_approx_43_85_l531_53111

/-- Mass percentage of Cl in a mixture of NaClO and NaClO2 -/
theorem mass_percentage_cl_in_mixture (moles_NaClO moles_NaClO2 : ℝ) 
  (mass_Na mass_Cl mass_O : ℝ) : ℝ :=
  let molar_mass_NaClO := mass_Na + mass_Cl + mass_O
  let molar_mass_NaClO2 := mass_Na + mass_Cl + 2 * mass_O
  let mass_Cl_NaClO := moles_NaClO * mass_Cl
  let mass_Cl_NaClO2 := moles_NaClO2 * mass_Cl
  let total_mass_Cl := mass_Cl_NaClO + mass_Cl_NaClO2
  let total_mass_mixture := moles_NaClO * molar_mass_NaClO + moles_NaClO2 * molar_mass_NaClO2
  let mass_percentage_Cl := (total_mass_Cl / total_mass_mixture) * 100
  mass_percentage_Cl

/-- The mass percentage of Cl in the given mixture is approximately 43.85% -/
theorem mass_percentage_cl_approx_43_85 :
  abs (mass_percentage_cl_in_mixture 3 2 22.99 35.45 16 - 43.85) < 0.01 :=
sorry

end NUMINAMATH_CALUDE_mass_percentage_cl_in_mixture_mass_percentage_cl_approx_43_85_l531_53111


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l531_53113

theorem complex_fraction_simplification :
  (7 + 15 * Complex.I) / (3 - 4 * Complex.I) = -39/25 + (73/25) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l531_53113


namespace NUMINAMATH_CALUDE_train_crossing_time_l531_53184

/-- Given two trains moving in opposite directions, this theorem proves
    the time taken for them to cross each other. -/
theorem train_crossing_time
  (train_length : ℝ)
  (faster_speed : ℝ)
  (h1 : train_length = 100)
  (h2 : faster_speed = 48)
  (h3 : faster_speed > 0) :
  let slower_speed := faster_speed / 2
  let relative_speed := faster_speed + slower_speed
  let total_distance := 2 * train_length
  let time := total_distance / (relative_speed * (1000 / 3600))
  time = 10 := by sorry

end NUMINAMATH_CALUDE_train_crossing_time_l531_53184


namespace NUMINAMATH_CALUDE_ones_digit_of_7_to_53_l531_53180

theorem ones_digit_of_7_to_53 : (7^53 : ℕ) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_7_to_53_l531_53180


namespace NUMINAMATH_CALUDE_coplanar_vectors_m_l531_53149

/-- Three vectors in ℝ³ are coplanar if and only if their scalar triple product is zero -/
def coplanar (a b c : ℝ × ℝ × ℝ) : Prop :=
  let (a₁, a₂, a₃) := a
  let (b₁, b₂, b₃) := b
  let (c₁, c₂, c₃) := c
  a₁ * (b₂ * c₃ - b₃ * c₂) - a₂ * (b₁ * c₃ - b₃ * c₁) + a₃ * (b₁ * c₂ - b₂ * c₁) = 0

theorem coplanar_vectors_m (m : ℝ) : 
  coplanar (1, -1, 0) (-1, 2, 1) (2, 1, m) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_coplanar_vectors_m_l531_53149


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_I_l531_53174

/-- A linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Definition of Quadrant I in the Cartesian plane -/
def QuadrantI (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The main theorem stating that the given linear function does not pass through Quadrant I -/
theorem linear_function_not_in_quadrant_I (f : LinearFunction) 
  (h1 : f.m = -2)
  (h2 : f.b = -1) : 
  ¬ ∃ (x y : ℝ), y = f.m * x + f.b ∧ QuadrantI x y :=
sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_I_l531_53174


namespace NUMINAMATH_CALUDE_rectangle_length_l531_53102

/-- Given a rectangle with width 16 cm and perimeter 70 cm, prove its length is 19 cm. -/
theorem rectangle_length (width : ℝ) (perimeter : ℝ) (length : ℝ) : 
  width = 16 → 
  perimeter = 70 → 
  perimeter = 2 * (length + width) → 
  length = 19 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l531_53102


namespace NUMINAMATH_CALUDE_sector_area_l531_53144

/-- Given a circular sector with central angle 2 radians and circumference 4 cm, its area is 1 cm² -/
theorem sector_area (θ : ℝ) (c : ℝ) (A : ℝ) : 
  θ = 2 → c = 4 → A = 1 → A = (θ * c^2) / (8 * π) := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l531_53144


namespace NUMINAMATH_CALUDE_integral_reciprocal_x_from_one_over_e_to_e_l531_53167

open Real MeasureTheory

theorem integral_reciprocal_x_from_one_over_e_to_e :
  ∫ x in (1 / Real.exp 1)..(Real.exp 1), (1 / x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_x_from_one_over_e_to_e_l531_53167


namespace NUMINAMATH_CALUDE_iron_conducts_electricity_l531_53152

-- Define the universe of discourse
variable (Object : Type)

-- Define the predicates
variable (is_metal : Object → Prop)
variable (can_conduct_electricity : Object → Prop)

-- Define iron as a constant
variable (iron : Object)

-- Theorem statement
theorem iron_conducts_electricity 
  (all_metals_conduct : ∀ x, is_metal x → can_conduct_electricity x) 
  (iron_is_metal : is_metal iron) : 
  can_conduct_electricity iron := by
  sorry

end NUMINAMATH_CALUDE_iron_conducts_electricity_l531_53152


namespace NUMINAMATH_CALUDE_sequence_property_l531_53104

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem sequence_property (a : ℕ → ℝ) 
    (h : ∀ n : ℕ, a (n + 1) + 1 = a n) : 
  is_arithmetic_sequence a (-1) := by
sorry

end NUMINAMATH_CALUDE_sequence_property_l531_53104


namespace NUMINAMATH_CALUDE_regular_ngon_triangle_property_l531_53189

/-- A regular n-gon -/
structure RegularNGon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- Triangle type: acute, right, or obtuse -/
inductive TriangleType
  | Acute
  | Right
  | Obtuse

/-- Determine the type of a triangle given its vertices -/
def triangleType (A B C : ℝ × ℝ) : TriangleType :=
  sorry

/-- The main theorem -/
theorem regular_ngon_triangle_property (n : ℕ) (hn : n > 0) :
  ∀ (P : RegularNGon n) (σ : Fin n → Fin n),
  Function.Bijective σ →
  ∃ (i j k : Fin n),
    triangleType (P.vertices i) (P.vertices j) (P.vertices k) =
    triangleType (P.vertices (σ i)) (P.vertices (σ j)) (P.vertices (σ k)) :=
sorry

end NUMINAMATH_CALUDE_regular_ngon_triangle_property_l531_53189


namespace NUMINAMATH_CALUDE_dance_steps_l531_53199

theorem dance_steps (nancy_ratio : ℕ) (total_steps : ℕ) (jason_steps : ℕ) : 
  nancy_ratio = 3 →
  total_steps = 32 →
  jason_steps + nancy_ratio * jason_steps = total_steps →
  jason_steps = 8 := by
sorry

end NUMINAMATH_CALUDE_dance_steps_l531_53199


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l531_53170

/-- An isosceles triangle with one side length of 3 and perimeter of 7 has equal sides of length 3 or 2 -/
theorem isosceles_triangle_side_length (a b c : ℝ) : 
  a + b + c = 7 →  -- perimeter is 7
  (a = 3 ∨ b = 3 ∨ c = 3) →  -- one side length is 3
  ((a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (a = c ∧ a ≠ b)) →  -- isosceles condition
  (a = 3 ∧ b = 3) ∨ (a = 2 ∧ b = 2) ∨ (b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) ∨ (a = 3 ∧ c = 3) ∨ (a = 2 ∧ c = 2) := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l531_53170


namespace NUMINAMATH_CALUDE_angle_bisector_sum_l531_53154

/-- A triangle with vertices P, Q, and R -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The angle bisector equation of the form ax + 2y + c = 0 -/
structure AngleBisectorEq where
  a : ℝ
  c : ℝ

/-- Given a triangle PQR, returns the angle bisector equation of ∠P -/
def angleBisectorP (t : Triangle) : AngleBisectorEq := sorry

theorem angle_bisector_sum (t : Triangle) 
  (h : t.P = (-7, 4) ∧ t.Q = (-14, -20) ∧ t.R = (2, -8)) : 
  let eq := angleBisectorP t
  eq.a + eq.c = 40 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_sum_l531_53154


namespace NUMINAMATH_CALUDE_combined_mix_dried_fruit_percentage_l531_53132

/-- Represents a trail mix composition -/
structure TrailMix where
  nuts : ℚ
  dried_fruit : ℚ
  chocolate_chips : ℚ
  composition_sum : nuts + dried_fruit + chocolate_chips = 1

/-- The combined trail mix from two equal portions -/
def combined_mix (mix1 mix2 : TrailMix) : TrailMix :=
  { nuts := (mix1.nuts + mix2.nuts) / 2,
    dried_fruit := (mix1.dried_fruit + mix2.dried_fruit) / 2,
    chocolate_chips := (mix1.chocolate_chips + mix2.chocolate_chips) / 2,
    composition_sum := by sorry }

theorem combined_mix_dried_fruit_percentage
  (sue_mix : TrailMix)
  (jane_mix : TrailMix)
  (h_sue_nuts : sue_mix.nuts = 3/10)
  (h_sue_dried_fruit : sue_mix.dried_fruit = 7/10)
  (h_jane_nuts : jane_mix.nuts = 6/10)
  (h_jane_chocolate : jane_mix.chocolate_chips = 4/10)
  (h_combined_nuts : (combined_mix sue_mix jane_mix).nuts = 45/100) :
  (combined_mix sue_mix jane_mix).dried_fruit = 35/100 := by
sorry

end NUMINAMATH_CALUDE_combined_mix_dried_fruit_percentage_l531_53132


namespace NUMINAMATH_CALUDE_R_share_is_1295_l531_53123

/-- Represents the capital invested by a partner -/
structure Capital where
  amount : ℚ
  is_positive : amount > 0

/-- Represents the investment scenario of the four partners -/
structure InvestmentScenario where
  P : Capital
  Q : Capital
  R : Capital
  S : Capital
  ratio_PQ : 4 * P.amount = 6 * Q.amount
  ratio_QR : 6 * Q.amount = 10 * R.amount
  S_investment : S.amount = P.amount + Q.amount
  total_profit : ℚ
  profit_is_positive : total_profit > 0

/-- Calculates the share of profit for partner R -/
def calculate_R_share (scenario : InvestmentScenario) : ℚ :=
  let total_capital := scenario.P.amount + scenario.Q.amount + scenario.R.amount + scenario.S.amount
  (scenario.total_profit * scenario.R.amount) / total_capital

/-- Theorem stating that R's share of profit is 1295 given the investment scenario -/
theorem R_share_is_1295 (scenario : InvestmentScenario) (h : scenario.total_profit = 12090) :
  calculate_R_share scenario = 1295 := by
  sorry

end NUMINAMATH_CALUDE_R_share_is_1295_l531_53123


namespace NUMINAMATH_CALUDE_smallest_common_factor_l531_53100

theorem smallest_common_factor (m : ℕ) : m = 108 ↔ 
  (m > 0 ∧ 
   ∃ (k : ℕ), k > 1 ∧ k ∣ (11*m - 3) ∧ k ∣ (8*m + 5) ∧
   ∀ (n : ℕ), n < m → ¬(∃ (l : ℕ), l > 1 ∧ l ∣ (11*n - 3) ∧ l ∣ (8*n + 5))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l531_53100


namespace NUMINAMATH_CALUDE_number_comparisons_l531_53131

theorem number_comparisons :
  (π > 3.14) ∧ (-Real.sqrt 3 < -Real.sqrt 2) ∧ (2 < Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_number_comparisons_l531_53131


namespace NUMINAMATH_CALUDE_max_product_ab_l531_53143

theorem max_product_ab (a b : ℝ) 
  (h : ∀ x : ℝ, Real.exp x ≥ a * (x - 1) + b) : 
  a * b ≤ (1/2) * Real.exp 3 := by
sorry

end NUMINAMATH_CALUDE_max_product_ab_l531_53143


namespace NUMINAMATH_CALUDE_tomato_problem_l531_53139

/-- The number of tomatoes produced by the first plant -/
def first_plant : ℕ := 19

/-- The number of tomatoes produced by the second plant -/
def second_plant (x : ℕ) : ℕ := x / 2 + 5

/-- The number of tomatoes produced by the third plant -/
def third_plant (x : ℕ) : ℕ := second_plant x + 2

/-- The total number of tomatoes produced by all three plants -/
def total_tomatoes : ℕ := 60

theorem tomato_problem :
  first_plant + second_plant first_plant + third_plant first_plant = total_tomatoes :=
by sorry

end NUMINAMATH_CALUDE_tomato_problem_l531_53139


namespace NUMINAMATH_CALUDE_max_plain_cookies_is_20_l531_53119

/-- Represents the number of cookies with a specific ingredient -/
structure CookieCount where
  total : ℕ
  chocolate : ℕ
  nuts : ℕ
  raisins : ℕ
  sprinkles : ℕ

/-- The conditions of the cookie problem -/
def cookieProblem : CookieCount where
  total := 60
  chocolate := 20
  nuts := 30
  raisins := 40
  sprinkles := 15

/-- The maximum number of cookies without any of the specified ingredients -/
def maxPlainCookies (c : CookieCount) : ℕ :=
  c.total - max c.chocolate (max c.nuts (max c.raisins c.sprinkles))

/-- Theorem stating the maximum number of plain cookies in the given problem -/
theorem max_plain_cookies_is_20 :
  maxPlainCookies cookieProblem = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_plain_cookies_is_20_l531_53119


namespace NUMINAMATH_CALUDE_motorcycle_speeds_correct_l531_53141

/-- Two motorcyclists travel towards each other with uniform speed. -/
structure MotorcycleJourney where
  /-- Total distance between starting points A and B in km -/
  total_distance : ℝ
  /-- Distance traveled by the first motorcyclist when the second has traveled 200 km -/
  first_partial_distance : ℝ
  /-- Time difference in hours between arrivals -/
  time_difference : ℝ
  /-- Speed of the first motorcyclist in km/h -/
  speed_first : ℝ
  /-- Speed of the second motorcyclist in km/h -/
  speed_second : ℝ

/-- The speeds of the motorcyclists satisfy the given conditions -/
def satisfies_conditions (j : MotorcycleJourney) : Prop :=
  j.total_distance = 600 ∧
  j.first_partial_distance = 250 ∧
  j.time_difference = 3 ∧
  j.first_partial_distance / j.speed_first = 200 / j.speed_second ∧
  j.total_distance / j.speed_first + j.time_difference = j.total_distance / j.speed_second

/-- The theorem stating that the given speeds satisfy the conditions -/
theorem motorcycle_speeds_correct (j : MotorcycleJourney) :
  j.speed_first = 50 ∧ j.speed_second = 40 → satisfies_conditions j :=
by sorry

end NUMINAMATH_CALUDE_motorcycle_speeds_correct_l531_53141


namespace NUMINAMATH_CALUDE_min_additional_marbles_for_john_l531_53191

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed -/
theorem min_additional_marbles_for_john : 
  min_additional_marbles 15 60 = 60 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_marbles_for_john_l531_53191


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l531_53155

theorem inequality_system_solutions : 
  ∃! (s : Finset ℕ), 
    (∀ x ∈ s, (3 - 2*x > 0 ∧ 2*x - 7 ≤ 4*x + 7)) ∧ 
    (∀ x : ℕ, (3 - 2*x > 0 ∧ 2*x - 7 ≤ 4*x + 7) → x ∈ s) ∧
    Finset.card s = 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l531_53155


namespace NUMINAMATH_CALUDE_not_always_meaningful_regression_l531_53190

-- Define the variables and their properties
variable (x y : ℝ)
variable (scatter_points : Set (ℝ × ℝ))

-- Define the conditions
def are_correlated (x y : ℝ) : Prop := sorry

def roughly_linear_distribution (points : Set (ℝ × ℝ)) : Prop := sorry

def regression_equation_meaningful (points : Set (ℝ × ℝ)) : Prop := sorry

-- State the theorem
theorem not_always_meaningful_regression 
  (h1 : are_correlated x y)
  (h2 : roughly_linear_distribution scatter_points) :
  ¬ ∀ (data : Set (ℝ × ℝ)), regression_equation_meaningful data :=
sorry

end NUMINAMATH_CALUDE_not_always_meaningful_regression_l531_53190


namespace NUMINAMATH_CALUDE_equation_equivalence_l531_53122

theorem equation_equivalence (x : ℝ) : 1 - (x + 3) / 3 = x / 2 ↔ 6 - 2 * x - 6 = 3 * x := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l531_53122


namespace NUMINAMATH_CALUDE_parabola_focus_is_correct_l531_53110

/-- The focus of a parabola given by y = ax^2 + bx + c -/
def parabola_focus (a b c : ℝ) : ℝ × ℝ := sorry

/-- The equation of the parabola -/
def parabola_equation (x : ℝ) : ℝ := -3 * x^2 - 6 * x

theorem parabola_focus_is_correct :
  parabola_focus (-3) (-6) 0 = (-1, 35/12) := by sorry

end NUMINAMATH_CALUDE_parabola_focus_is_correct_l531_53110


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l531_53114

/-- Represents the price and volume of orangeade on a given day -/
structure OrangeadeDay where
  juice : ℝ
  water : ℝ
  price : ℝ

/-- Calculates the revenue for a given day -/
def revenue (day : OrangeadeDay) : ℝ :=
  (day.juice + day.water) * day.price

theorem orangeade_price_day2 (day1 day2 : OrangeadeDay) :
  day1.juice = day1.water →
  day2.juice = day1.juice →
  day2.water = 2 * day1.water →
  day1.price = 0.9 →
  revenue day1 = revenue day2 →
  day2.price = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l531_53114


namespace NUMINAMATH_CALUDE_count_symmetric_scanning_codes_l531_53112

/-- A symmetric scanning code is a 7x7 grid of black and white squares that is invariant under 90° rotations and reflections across diagonals and midlines. -/
def SymmetricScanningCode := Fin 7 → Fin 7 → Bool

/-- A scanning code is valid if it has at least one black and one white square. -/
def is_valid (code : SymmetricScanningCode) : Prop :=
  (∃ i j, code i j = true) ∧ (∃ i j, code i j = false)

/-- A scanning code is symmetric if it's invariant under 90° rotations and reflections. -/
def is_symmetric (code : SymmetricScanningCode) : Prop :=
  (∀ i j, code i j = code (6-j) i) ∧  -- 90° rotation
  (∀ i j, code i j = code j i) ∧      -- diagonal reflection
  (∀ i j, code i j = code (6-i) (6-j))  -- midline reflection

/-- The number of valid symmetric scanning codes -/
def num_valid_symmetric_codes : ℕ := sorry

theorem count_symmetric_scanning_codes :
  num_valid_symmetric_codes = 1022 :=
sorry

end NUMINAMATH_CALUDE_count_symmetric_scanning_codes_l531_53112


namespace NUMINAMATH_CALUDE_two_digit_three_digit_sum_l531_53136

theorem two_digit_three_digit_sum : ∃! (x y : ℕ), 
  10 ≤ x ∧ x < 100 ∧ 100 ≤ y ∧ y < 1000 ∧ 
  1000 * x + y = 11 * x * y ∧ 
  x + y = 919 := by sorry

end NUMINAMATH_CALUDE_two_digit_three_digit_sum_l531_53136


namespace NUMINAMATH_CALUDE_system_solution_l531_53185

theorem system_solution (x y z : ℝ) (eq1 : x + y + z = 0) (eq2 : 4 * x + 2 * y + z = 0) :
  y = -3 * x ∧ z = 2 * x := by
sorry

end NUMINAMATH_CALUDE_system_solution_l531_53185


namespace NUMINAMATH_CALUDE_left_square_side_length_l531_53105

theorem left_square_side_length :
  ∀ (x : ℝ),
  (∃ (y z : ℝ),
    y = x + 17 ∧
    z = y - 6 ∧
    x + y + z = 52) →
  x = 8 := by
  sorry

end NUMINAMATH_CALUDE_left_square_side_length_l531_53105


namespace NUMINAMATH_CALUDE_bookkeeper_arrangements_l531_53115

/-- The number of distinct arrangements of letters in a word with the given letter distribution -/
def distinctArrangements (totalLetters : Nat) (repeatedLetters : Nat) (repeatCount : Nat) : Nat :=
  Nat.factorial totalLetters / (Nat.factorial repeatCount ^ repeatedLetters)

/-- Theorem stating the number of distinct arrangements for the specific word structure -/
theorem bookkeeper_arrangements :
  distinctArrangements 10 4 2 = 226800 := by
  sorry

end NUMINAMATH_CALUDE_bookkeeper_arrangements_l531_53115


namespace NUMINAMATH_CALUDE_box_2_neg2_3_l531_53192

def box (a b c : ℤ) : ℚ := (a ^ (2 * b) : ℚ) - (b ^ (2 * c) : ℚ) + (c ^ (2 * a) : ℚ)

theorem box_2_neg2_3 : box 2 (-2) 3 = 273 / 16 := by sorry

end NUMINAMATH_CALUDE_box_2_neg2_3_l531_53192


namespace NUMINAMATH_CALUDE_segment_is_definition_l531_53142

-- Define the type for geometric statements
inductive GeometricStatement
  | TwoPointsLine
  | SegmentDefinition
  | ComplementaryAngles
  | AlternateInteriorAngles

-- Define a predicate to check if a statement is a definition
def isDefinition : GeometricStatement → Prop
  | GeometricStatement.SegmentDefinition => True
  | _ => False

-- Theorem statement
theorem segment_is_definition :
  (∃! s : GeometricStatement, isDefinition s) →
  isDefinition GeometricStatement.SegmentDefinition :=
by
  sorry

end NUMINAMATH_CALUDE_segment_is_definition_l531_53142


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_two_l531_53194

/-- Two lines in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Definition of the first line -/
def line1 (k : ℝ) : Line3D :=
  { point := (-2, 4, 2),
    direction := (1, -k, k) }

/-- Definition of the second line -/
def line2 : Line3D :=
  { point := (0, 2, 3),
    direction := (1, 2, -1) }

/-- Two lines are coplanar if their direction vectors and the vector connecting their points are linearly dependent -/
def are_coplanar (l1 l2 : Line3D) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∧
    a • l1.direction + b • l2.direction = c • (l2.point - l1.point)

/-- Theorem stating that the lines are coplanar if and only if k = -2 -/
theorem lines_coplanar_iff_k_eq_neg_two (k : ℝ) :
  are_coplanar (line1 k) line2 ↔ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_k_eq_neg_two_l531_53194


namespace NUMINAMATH_CALUDE_parabola_shift_left_two_l531_53169

/-- Represents a parabola in the xy-plane -/
structure Parabola where
  f : ℝ → ℝ

/-- Shifts a parabola horizontally -/
def shift_parabola (p : Parabola) (h : ℝ) : Parabola :=
  { f := fun x => p.f (x + h) }

/-- The standard parabola y = x^2 -/
def standard_parabola : Parabola :=
  { f := fun x => x^2 }

theorem parabola_shift_left_two :
  (shift_parabola standard_parabola 2).f = fun x => (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_left_two_l531_53169


namespace NUMINAMATH_CALUDE_tangent_line_triangle_area_l531_53134

/-- A line tangent to the unit circle with intercepts summing to √3 forms a triangle with area 3/2 --/
theorem tangent_line_triangle_area :
  ∀ (a b : ℝ),
  (a > 0 ∧ b > 0) →  -- Positive intercepts
  (a + b = Real.sqrt 3) →  -- Sum of intercepts
  (∃ (x y : ℝ), x^2 + y^2 = 1 ∧ a*y + b*x = a*b) →  -- Tangent to unit circle
  (1/2 * a * b = 3/2) :=  -- Area of triangle
by sorry

end NUMINAMATH_CALUDE_tangent_line_triangle_area_l531_53134


namespace NUMINAMATH_CALUDE_photographer_choices_l531_53103

theorem photographer_choices (n : ℕ) (k₁ k₂ : ℕ) (h₁ : n = 7) (h₂ : k₁ = 4) (h₃ : k₂ = 5) :
  Nat.choose n k₁ + Nat.choose n k₂ = 56 :=
by sorry

end NUMINAMATH_CALUDE_photographer_choices_l531_53103


namespace NUMINAMATH_CALUDE_ages_sum_l531_53150

theorem ages_sum (a b c : ℕ+) (h1 : a = b) (h2 : a > c) (h3 : a * b * c = 72) : a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l531_53150


namespace NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l531_53188

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def largestPerfectSquareDivisor (n : ℕ) : ℕ :=
  sorry

def sumOfPrimeFactorExponents (n : ℕ) : ℕ :=
  sorry

theorem sum_of_exponents_15_factorial :
  sumOfPrimeFactorExponents (largestPerfectSquareDivisor (factorial 15).sqrt) = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_15_factorial_l531_53188


namespace NUMINAMATH_CALUDE_total_milk_count_l531_53108

theorem total_milk_count (chocolate : ℕ) (strawberry : ℕ) (regular : ℕ)
  (h1 : chocolate = 2)
  (h2 : strawberry = 15)
  (h3 : regular = 3) :
  chocolate + strawberry + regular = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_milk_count_l531_53108


namespace NUMINAMATH_CALUDE_circle_equation_l531_53163

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangent line
def tangentLine (x : ℝ) : ℝ := 2 * x + 1

-- Define the properties of the circle
def circleProperties (c : Circle) : Prop :=
  -- The center is on the x-axis
  c.center.2 = 0 ∧
  -- The circle is tangent to the line y = 2x + 1 at point (0, 1)
  c.radius^2 = c.center.1^2 + 1 ∧
  -- The tangent line is perpendicular to the radius at the point of tangency
  2 * c.center.1 + c.center.2 - 1 = 0

-- Theorem statement
theorem circle_equation (c : Circle) (h : circleProperties c) :
  ∀ (x y : ℝ), (x - 2)^2 + y^2 = 5 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l531_53163


namespace NUMINAMATH_CALUDE_f_one_equals_one_l531_53172

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_one_equals_one
  (f : ℝ → ℝ)
  (h : is_odd_function (fun x ↦ f (x + 1) - 1)) :
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_one_equals_one_l531_53172


namespace NUMINAMATH_CALUDE_unique_solution_system_l531_53168

theorem unique_solution_system (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃! (x y z : ℝ), x + a * y + a^2 * z = 0 ∧
                   x + b * y + b^2 * z = 0 ∧
                   x + c * y + c^2 * z = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_system_l531_53168


namespace NUMINAMATH_CALUDE_complex_sixth_power_l531_53121

theorem complex_sixth_power (z : ℂ) : z = (-Real.sqrt 3 - I) / 2 → z^6 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sixth_power_l531_53121


namespace NUMINAMATH_CALUDE_all_propositions_false_l531_53106

-- Define the concept of skew lines
def are_skew (l1 l2 : Line3D) : Prop := sorry

-- Define the concept of perpendicular lines
def is_perpendicular (l1 l2 : Line3D) : Prop := sorry

-- Define the concept of intersecting lines
def intersect (l1 l2 : Line3D) : Prop := sorry

-- Define the concept of lines in different planes
def in_different_planes (l1 l2 : Line3D) : Prop := sorry

theorem all_propositions_false :
  (∀ l1 l2 : Line3D, in_different_planes l1 l2 → are_skew l1 l2) = False ∧
  (∃! l : Line3D, ∀ l1 l2 : Line3D, are_skew l1 l2 → is_perpendicular l l1 ∧ is_perpendicular l l2) = False ∧
  (∀ l1 l2 l3 l4 : Line3D, are_skew l1 l2 → intersect l3 l1 → intersect l3 l2 → intersect l4 l1 → intersect l4 l2 → are_skew l3 l4) = False ∧
  (∀ a b c : Line3D, are_skew a b → are_skew b c → are_skew a c) = False :=
by sorry

end NUMINAMATH_CALUDE_all_propositions_false_l531_53106


namespace NUMINAMATH_CALUDE_fermats_little_theorem_l531_53128

theorem fermats_little_theorem (p a : ℕ) (hp : Prime p) (ha : ¬(p ∣ a)) :
  a^(p-1) ≡ 1 [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_fermats_little_theorem_l531_53128


namespace NUMINAMATH_CALUDE_sequence_next_terms_l531_53107

def sequence1 : ℕ → ℕ
  | 0 => 2
  | n + 1 => sequence1 n + 2

def sequence2 : ℕ → ℕ
  | 0 => 3
  | n + 1 => sequence2 n * 2

def sequence3 : ℕ → ℕ
  | 0 => 36
  | 1 => 11
  | n + 2 => sequence3 n + 2

theorem sequence_next_terms :
  (sequence1 5 = 12 ∧ sequence1 6 = 14) ∧
  (sequence2 5 = 96) ∧
  (sequence3 8 = 44 ∧ sequence3 9 = 19) := by
  sorry

end NUMINAMATH_CALUDE_sequence_next_terms_l531_53107


namespace NUMINAMATH_CALUDE_ticket_sales_revenue_l531_53125

/-- The total money made from ticket sales given the conditions -/
def total_money_made (advance_price same_day_price total_tickets advance_tickets : ℕ) : ℕ :=
  advance_price * advance_tickets + same_day_price * (total_tickets - advance_tickets)

/-- Theorem stating that the total money made is $1600 under the given conditions -/
theorem ticket_sales_revenue : 
  total_money_made 20 30 60 20 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_ticket_sales_revenue_l531_53125


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l531_53138

noncomputable def f (x : ℝ) : ℝ := x * (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5)

theorem f_derivative_at_zero : 
  deriv f 0 = -120 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l531_53138


namespace NUMINAMATH_CALUDE_pizza_recipe_l531_53164

theorem pizza_recipe (water flour salt : ℚ) : 
  water = 10 ∧ 
  salt = (1/2) * flour ∧ 
  water + flour + salt = 34 →
  flour = 16 := by
sorry

end NUMINAMATH_CALUDE_pizza_recipe_l531_53164


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l531_53195

theorem complex_magnitude_problem (w z : ℂ) :
  w * z = 20 - 15 * I ∧ Complex.abs w = Real.sqrt 20 →
  Complex.abs z = (5 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l531_53195


namespace NUMINAMATH_CALUDE_find_regular_working_hours_l531_53153

/-- Represents the problem of finding regular working hours per day --/
theorem find_regular_working_hours
  (working_days_per_week : ℕ)
  (regular_pay_rate : ℚ)
  (overtime_pay_rate : ℚ)
  (total_earnings : ℚ)
  (total_hours : ℕ)
  (h1 : working_days_per_week = 6)
  (h2 : regular_pay_rate = 21/10)
  (h3 : overtime_pay_rate = 42/10)
  (h4 : total_earnings = 525)
  (h5 : total_hours = 245) :
  ∃ (regular_hours_per_day : ℕ),
    regular_hours_per_day = 10 ∧
    regular_hours_per_day * working_days_per_week * 4 ≤ total_hours ∧
    regular_pay_rate * (regular_hours_per_day * working_days_per_week * 4) +
    overtime_pay_rate * (total_hours - regular_hours_per_day * working_days_per_week * 4) =
    total_earnings :=
by sorry

end NUMINAMATH_CALUDE_find_regular_working_hours_l531_53153


namespace NUMINAMATH_CALUDE_range_of_m_l531_53117

theorem range_of_m (m : ℝ) :
  (∀ θ : ℝ, m^2 + (Real.cos θ^2 - 5) * m + 4 * Real.sin θ^2 ≥ 0) →
  (m ≤ 0 ∨ m ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l531_53117


namespace NUMINAMATH_CALUDE_exponent_multiplication_l531_53129

theorem exponent_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l531_53129


namespace NUMINAMATH_CALUDE_intersection_M_N_l531_53177

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l531_53177


namespace NUMINAMATH_CALUDE_zoo_theorem_l531_53198

def zoo_problem (tiger_enclosures : ℕ) (zebra_enclosures_per_tiger : ℕ) 
  (giraffe_enclosures_multiplier : ℕ) (tigers_per_enclosure : ℕ) 
  (zebras_per_enclosure : ℕ) (giraffes_per_enclosure : ℕ) : Prop :=
  let total_zebra_enclosures := tiger_enclosures * zebra_enclosures_per_tiger
  let total_giraffe_enclosures := total_zebra_enclosures * giraffe_enclosures_multiplier
  let total_tigers := tiger_enclosures * tigers_per_enclosure
  let total_zebras := total_zebra_enclosures * zebras_per_enclosure
  let total_giraffes := total_giraffe_enclosures * giraffes_per_enclosure
  let total_animals := total_tigers + total_zebras + total_giraffes
  total_animals = 144

theorem zoo_theorem : 
  zoo_problem 4 2 3 4 10 2 := by
  sorry

end NUMINAMATH_CALUDE_zoo_theorem_l531_53198
