import Mathlib

namespace NUMINAMATH_CALUDE_curve_tangent_values_l2968_296845

/-- The curve equation -/
def curve (x a b : ℝ) : ℝ := x^2 + a*x + b

/-- The tangent equation -/
def tangent (x y : ℝ) : Prop := x - y + 1 = 0

/-- Main theorem -/
theorem curve_tangent_values (a b : ℝ) :
  (∀ x y, curve x a b = y → tangent x y) →
  a = 1 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_curve_tangent_values_l2968_296845


namespace NUMINAMATH_CALUDE_candy_distribution_l2968_296894

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (candy_per_student : ℕ) : 
  total_candy = 18 → num_students = 9 → candy_per_student = total_candy / num_students → 
  candy_per_student = 2 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2968_296894


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2968_296804

def vector_a : ℝ × ℝ := (3, -1)
def vector_b (x : ℝ) : ℝ × ℝ := (1, x)

theorem perpendicular_vectors_x_value :
  ∀ x : ℝ, (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l2968_296804


namespace NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2968_296866

theorem inscribed_circle_rectangle_area (r : ℝ) (ratio : ℝ) : 
  r > 0 → 
  ratio > 0 → 
  let width := 2 * r
  let length := ratio * width
  let area := length * width
  r = 8 ∧ ratio = 3 → area = 768 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_rectangle_area_l2968_296866


namespace NUMINAMATH_CALUDE_negative_difference_l2968_296871

theorem negative_difference (m n : ℝ) : -(m - n) = -m + n := by
  sorry

end NUMINAMATH_CALUDE_negative_difference_l2968_296871


namespace NUMINAMATH_CALUDE_average_text_messages_l2968_296883

/-- Calculate the average number of text messages sent over 5 days -/
theorem average_text_messages 
  (day1 : ℕ) 
  (day2 : ℕ) 
  (day3_to_5 : ℕ) 
  (h1 : day1 = 220) 
  (h2 : day2 = day1 / 2) 
  (h3 : day3_to_5 = 50) :
  (day1 + day2 + 3 * day3_to_5) / 5 = 96 := by
  sorry

end NUMINAMATH_CALUDE_average_text_messages_l2968_296883


namespace NUMINAMATH_CALUDE_promotion_theorem_l2968_296846

/-- Calculates the maximum amount of goods that can be purchased given a promotion and initial spending. -/
def maxPurchaseAmount (promotionRate : Rat) (rewardRate : Rat) (initialSpend : ℕ) : ℕ :=
  sorry

/-- The promotion theorem -/
theorem promotion_theorem :
  let promotionRate : Rat := 100
  let rewardRate : Rat := 20
  let initialSpend : ℕ := 7020
  maxPurchaseAmount promotionRate rewardRate initialSpend = 8760 := by
  sorry

end NUMINAMATH_CALUDE_promotion_theorem_l2968_296846


namespace NUMINAMATH_CALUDE_dina_dolls_count_l2968_296851

theorem dina_dolls_count (ivy_total : ℕ) (ivy_collectors : ℕ) (h1 : ivy_collectors = 20) 
  (h2 : ivy_collectors = 2 * ivy_total / 3) (h3 : ivy_total > 0) : 
  2 * ivy_total = 60 := by
  sorry

end NUMINAMATH_CALUDE_dina_dolls_count_l2968_296851


namespace NUMINAMATH_CALUDE_library_repacking_l2968_296876

theorem library_repacking (total_books : Nat) (initial_boxes : Nat) (new_box_size : Nat) 
  (h1 : total_books = 1870)
  (h2 : initial_boxes = 55)
  (h3 : new_box_size = 36) :
  total_books % new_box_size = 34 := by
  sorry

end NUMINAMATH_CALUDE_library_repacking_l2968_296876


namespace NUMINAMATH_CALUDE_modulus_of_z_l2968_296850

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I - 1) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2968_296850


namespace NUMINAMATH_CALUDE_rectangular_fence_length_l2968_296892

/-- A rectangular fence with a perimeter of 30 meters and a length that is twice its width has a length of 10 meters. -/
theorem rectangular_fence_length (width : ℝ) (length : ℝ) : 
  width > 0 → 
  length > 0 → 
  length = 2 * width → 
  2 * length + 2 * width = 30 → 
  length = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_fence_length_l2968_296892


namespace NUMINAMATH_CALUDE_train_crossing_time_l2968_296884

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 160 →
  train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2968_296884


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l2968_296879

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) : 
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l2968_296879


namespace NUMINAMATH_CALUDE_quadratic_equation_completing_square_l2968_296809

theorem quadratic_equation_completing_square (x : ℝ) : 
  x^2 - 4*x + 3 = 0 → (x - 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_completing_square_l2968_296809


namespace NUMINAMATH_CALUDE_x_value_in_equation_l2968_296890

theorem x_value_in_equation (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 71) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_x_value_in_equation_l2968_296890


namespace NUMINAMATH_CALUDE_greatest_k_value_l2968_296839

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 145) →
  k ≤ Real.sqrt 177 :=
sorry

end NUMINAMATH_CALUDE_greatest_k_value_l2968_296839


namespace NUMINAMATH_CALUDE_parabola_y_relationship_l2968_296858

-- Define the parabola function
def f (x : ℝ) (m : ℝ) : ℝ := -3 * x^2 - 12 * x + m

-- Define the theorem
theorem parabola_y_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f (-3) m = y₁)
  (h₂ : f (-2) m = y₂)
  (h₃ : f 1 m = y₃) :
  y₂ > y₁ ∧ y₁ > y₃ :=
sorry

end NUMINAMATH_CALUDE_parabola_y_relationship_l2968_296858


namespace NUMINAMATH_CALUDE_distance_sum_squares_l2968_296803

theorem distance_sum_squares (z : ℂ) (h : Complex.abs (z - (3 - 3*I)) = 3) :
  let z' := 1 + I
  let z'' := 5 - 5*I  -- reflection of z' about 3 - 3i
  (Complex.abs (z - z'))^2 + (Complex.abs (z - z''))^2 = 101 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_squares_l2968_296803


namespace NUMINAMATH_CALUDE_train_crossing_time_l2968_296868

/-- Proves that a train with given length and speed takes a specific time to cross an electric pole. -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 320 →
  train_speed_kmh = 144 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  crossing_time = 8 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2968_296868


namespace NUMINAMATH_CALUDE_limsup_subset_l2968_296878

open Set

theorem limsup_subset {α : Type*} (A B : ℕ → Set α) (h : ∀ n, A n ⊆ B n) :
  (⋂ k, ⋃ n ≥ k, A n) ⊆ (⋂ k, ⋃ n ≥ k, B n) := by
  sorry

end NUMINAMATH_CALUDE_limsup_subset_l2968_296878


namespace NUMINAMATH_CALUDE_ones_digit_of_19_power_l2968_296874

theorem ones_digit_of_19_power (n : ℕ) : 19^(19 * (13^13)) ≡ 9 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_19_power_l2968_296874


namespace NUMINAMATH_CALUDE_sin_equality_condition_l2968_296861

theorem sin_equality_condition :
  (∀ A B : ℝ, A = B → Real.sin A = Real.sin B) ∧
  (∃ A B : ℝ, Real.sin A = Real.sin B ∧ A ≠ B) := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_condition_l2968_296861


namespace NUMINAMATH_CALUDE_line_segment_point_sum_l2968_296834

/-- The line equation y = (5/3)x - 15 -/
def line_equation (x y : ℝ) : Prop := y = (5/3) * x - 15

/-- Point P is where the line crosses the x-axis -/
def point_P (x : ℝ) : Prop := line_equation x 0

/-- Point Q is where the line crosses the y-axis -/
def point_Q (y : ℝ) : Prop := line_equation 0 y

/-- Point T(r, s) is on the line -/
def point_T (r s : ℝ) : Prop := line_equation r s

/-- T is between P and Q on the line segment -/
def T_between_P_Q (r s : ℝ) : Prop := 
  ∃ (px qy : ℝ), point_P px ∧ point_Q qy ∧ 
  ((0 ≤ r ∧ r ≤ px) ∨ (px ≤ r ∧ r ≤ 0)) ∧
  ((qy ≤ s ∧ s ≤ 0) ∨ (0 ≤ s ∧ s ≤ qy))

/-- Area of triangle POQ is twice the area of triangle TOQ -/
def area_condition (r s : ℝ) : Prop :=
  ∃ (px qy : ℝ), point_P px ∧ point_Q qy ∧
  (1/2 * px * abs qy) = 2 * (1/2 * px * abs (s - qy))

theorem line_segment_point_sum : 
  ∀ (r s : ℝ), point_T r s ∧ T_between_P_Q r s ∧ area_condition r s → r + s = -3 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_point_sum_l2968_296834


namespace NUMINAMATH_CALUDE_jakobs_class_size_l2968_296811

theorem jakobs_class_size :
  ∃! b : ℕ, 100 < b ∧ b < 200 ∧
    b % 4 = 2 ∧ b % 5 = 2 ∧ b % 6 = 2 ∧
    b = 122 := by sorry

end NUMINAMATH_CALUDE_jakobs_class_size_l2968_296811


namespace NUMINAMATH_CALUDE_sqrt_a_plus_b_equals_four_l2968_296855

theorem sqrt_a_plus_b_equals_four :
  ∀ a b : ℕ,
  (a = ⌊Real.sqrt 17⌋) →
  (b - 1 = Real.sqrt 121) →
  Real.sqrt (a + b) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_plus_b_equals_four_l2968_296855


namespace NUMINAMATH_CALUDE_eighteenth_power_digits_l2968_296802

/-- The function that returns the list of digits in the decimal representation of a natural number -/
def digits (n : ℕ) : List ℕ :=
  sorry

/-- Theorem stating that 18 is the positive integer whose sixth power's decimal representation
    consists of the digits 0, 1, 2, 2, 2, 3, 4, 4 -/
theorem eighteenth_power_digits :
  ∃! (n : ℕ), n > 0 ∧ digits (n^6) = [3, 4, 0, 1, 2, 2, 2, 4] ∧ n = 18 :=
sorry

end NUMINAMATH_CALUDE_eighteenth_power_digits_l2968_296802


namespace NUMINAMATH_CALUDE_star_perimeter_is_160_l2968_296889

/-- The radius of each circle in cm -/
def circle_radius : ℝ := 5

/-- The side length of the square in cm -/
def square_side_length : ℝ := 4 * circle_radius

/-- The number of sides in the star -/
def star_sides : ℕ := 8

/-- The perimeter of the star in cm -/
def star_perimeter : ℝ := star_sides * square_side_length

/-- Theorem stating that the perimeter of the star is 160 cm -/
theorem star_perimeter_is_160 : star_perimeter = 160 := by
  sorry

end NUMINAMATH_CALUDE_star_perimeter_is_160_l2968_296889


namespace NUMINAMATH_CALUDE_x_and_a_ranges_l2968_296841

theorem x_and_a_ranges (x m a : ℝ) 
  (h1 : x^2 - 4*a*x + 3*a^2 < 0)
  (h2 : x = (1/2)^(m-1))
  (h3 : 1 < m ∧ m < 2) :
  (a = 1/4 → 1/2 < x ∧ x < 3/4) ∧
  (1/3 ≤ a ∧ a ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_x_and_a_ranges_l2968_296841


namespace NUMINAMATH_CALUDE_sum_of_c_values_l2968_296856

theorem sum_of_c_values : ∃ (S : Finset ℤ),
  (∀ c ∈ S, c ≤ 30 ∧ 
    ∃ (x y : ℚ), x^2 - 9*x - c = 0 ∧ y^2 - 9*y - c = 0 ∧ x ≠ y) ∧
  (∀ c : ℤ, c ≤ 30 → 
    (∃ (x y : ℚ), x^2 - 9*x - c = 0 ∧ y^2 - 9*y - c = 0 ∧ x ≠ y) → 
    c ∈ S) ∧
  (S.sum id = 32) := by
sorry

end NUMINAMATH_CALUDE_sum_of_c_values_l2968_296856


namespace NUMINAMATH_CALUDE_inequality_proof_l2968_296886

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) :=
by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2968_296886


namespace NUMINAMATH_CALUDE_sachin_age_l2968_296853

/-- Proves that Sachin's age is 49 given the conditions -/
theorem sachin_age :
  ∀ (s r : ℕ),
  r = s + 14 →
  s * 9 = r * 7 →
  s = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_sachin_age_l2968_296853


namespace NUMINAMATH_CALUDE_intersecting_circles_properties_l2968_296880

/-- Two circles intersecting at two distinct points -/
structure IntersectingCircles where
  r : ℝ
  a : ℝ
  b : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  r_pos : r > 0
  on_C₁_A : x₁^2 + y₁^2 = r^2
  on_C₁_B : x₂^2 + y₂^2 = r^2
  on_C₂_A : (x₁ - a)^2 + (y₁ - b)^2 = r^2
  on_C₂_B : (x₂ - a)^2 + (y₂ - b)^2 = r^2
  distinct : (x₁, y₁) ≠ (x₂, y₂)

/-- Properties of intersecting circles -/
theorem intersecting_circles_properties (c : IntersectingCircles) :
  (c.a * (c.x₁ - c.x₂) + c.b * (c.y₁ - c.y₂) = 0) ∧
  (2 * c.a * c.x₁ + 2 * c.b * c.y₁ = c.a^2 + c.b^2) ∧
  (c.x₁ + c.x₂ = c.a ∧ c.y₁ + c.y₂ = c.b) := by
  sorry

end NUMINAMATH_CALUDE_intersecting_circles_properties_l2968_296880


namespace NUMINAMATH_CALUDE_sin_beta_value_l2968_296854

theorem sin_beta_value (α β : Real) (h_acute : 0 < α ∧ α < π / 2)
  (h1 : 2 * Real.tan (π - α) - 3 * Real.cos (π / 2 + β) + 5 = 0)
  (h2 : Real.tan (π + α) + 6 * Real.sin (π + β) = 1) :
  Real.sin β = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_beta_value_l2968_296854


namespace NUMINAMATH_CALUDE_graces_pool_filling_time_l2968_296829

/-- The problem of filling Grace's pool --/
theorem graces_pool_filling_time 
  (pool_capacity : ℝ) 
  (first_hose_rate : ℝ) 
  (second_hose_rate : ℝ) 
  (additional_time : ℝ) 
  (h : pool_capacity = 390) 
  (r1 : first_hose_rate = 50) 
  (r2 : second_hose_rate = 70) 
  (t : additional_time = 2) :
  ∃ (wait_time : ℝ), 
    wait_time * first_hose_rate + 
    additional_time * (first_hose_rate + second_hose_rate) = 
    pool_capacity ∧ wait_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_graces_pool_filling_time_l2968_296829


namespace NUMINAMATH_CALUDE_newlandia_density_l2968_296867

/-- Represents the population and area data for a country -/
structure CountryData where
  population : ℕ
  area_sq_miles : ℕ

/-- Calculates the average square feet per person in a country -/
def avg_sq_feet_per_person (country : CountryData) : ℚ :=
  (country.area_sq_miles * (5280 * 5280) : ℚ) / country.population

/-- Theorem stating the properties of Newlandia's population density -/
theorem newlandia_density (newlandia : CountryData) 
  (h1 : newlandia.population = 350000000)
  (h2 : newlandia.area_sq_miles = 4500000) :
  let density := avg_sq_feet_per_person newlandia
  (358000 : ℚ) < density ∧ density < (359000 : ℚ) ∧ density > 700 := by
  sorry

#eval avg_sq_feet_per_person ⟨350000000, 4500000⟩

end NUMINAMATH_CALUDE_newlandia_density_l2968_296867


namespace NUMINAMATH_CALUDE_earning_goal_proof_l2968_296833

/-- Calculates the total earnings for a salesperson given fixed earnings, commission rate, and sales amount. -/
def totalEarnings (fixedEarnings : ℝ) (commissionRate : ℝ) (sales : ℝ) : ℝ :=
  fixedEarnings + commissionRate * sales

/-- Proves that the earning goal is $500 given the specified conditions. -/
theorem earning_goal_proof :
  let fixedEarnings : ℝ := 190
  let commissionRate : ℝ := 0.04
  let minSales : ℝ := 7750
  totalEarnings fixedEarnings commissionRate minSales = 500 := by
  sorry

#eval totalEarnings 190 0.04 7750

end NUMINAMATH_CALUDE_earning_goal_proof_l2968_296833


namespace NUMINAMATH_CALUDE_heptagon_diagonals_l2968_296830

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon is a polygon with 7 sides -/
def is_heptagon (n : ℕ) : Prop := n = 7

theorem heptagon_diagonals (n : ℕ) (h : is_heptagon n) : num_diagonals n = 14 := by
  sorry

end NUMINAMATH_CALUDE_heptagon_diagonals_l2968_296830


namespace NUMINAMATH_CALUDE_coin_ratio_l2968_296824

theorem coin_ratio (pennies nickels dimes quarters : ℕ) 
  (h1 : nickels = 5 * dimes)
  (h2 : pennies = 3 * nickels)
  (h3 : pennies = 120)
  (h4 : pennies + 5 * nickels + 10 * dimes + 25 * quarters = 800) :
  quarters = 2 * dimes := by
  sorry

end NUMINAMATH_CALUDE_coin_ratio_l2968_296824


namespace NUMINAMATH_CALUDE_complex_power_modulus_l2968_296848

theorem complex_power_modulus : 
  Complex.abs ((2/3 : ℂ) + (1/3 : ℂ) * Complex.I) ^ 8 = 625/6561 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l2968_296848


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2968_296805

theorem quadratic_equation_solution : ∃ (a b : ℝ), 
  (a^2 - 6*a + 11 = 27) ∧ 
  (b^2 - 6*b + 11 = 27) ∧ 
  (a ≥ b) ∧ 
  (3*a - 2*b = 28) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2968_296805


namespace NUMINAMATH_CALUDE_max_value_p_l2968_296888

theorem max_value_p (p q r s t u v w : ℕ+) : 
  (p + q + r + s = 35) →
  (q + r + s + t = 35) →
  (r + s + t + u = 35) →
  (s + t + u + v = 35) →
  (t + u + v + w = 35) →
  (q + v = 14) →
  (∀ x : ℕ+, x ≤ p → 
    ∃ q' r' s' t' u' v' w' : ℕ+,
      (x + q' + r' + s' = 35) ∧
      (q' + r' + s' + t' = 35) ∧
      (r' + s' + t' + u' = 35) ∧
      (s' + t' + u' + v' = 35) ∧
      (t' + u' + v' + w' = 35) ∧
      (q' + v' = 14)) →
  p = 20 :=
by sorry

end NUMINAMATH_CALUDE_max_value_p_l2968_296888


namespace NUMINAMATH_CALUDE_sixth_term_value_l2968_296832

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem sixth_term_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l2968_296832


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2968_296885

theorem complex_equation_solution (z : ℂ) :
  z * (2 - 3*I) = 6 + 4*I → z = 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2968_296885


namespace NUMINAMATH_CALUDE_card_sequence_periodicity_l2968_296815

def planet_value : ℕ := 2010
def hegemon_value (planets : ℕ) : ℕ := 4 * planets

def card_choice (n : ℕ) : ℕ := 
  if n ≤ 503 then 0 else (n - 503) % 2

theorem card_sequence_periodicity :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), n ≥ 503 → card_choice (n + k) = card_choice n :=
sorry

end NUMINAMATH_CALUDE_card_sequence_periodicity_l2968_296815


namespace NUMINAMATH_CALUDE_volume_ratio_of_pyramids_l2968_296826

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular pyramid -/
structure TriangularPyramid where
  apex : Point3D
  base1 : Point3D
  base2 : Point3D
  base3 : Point3D

/-- Calculates the volume of a triangular pyramid -/
def volumeOfTriangularPyramid (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Given two points, returns a point that divides the line segment in a given ratio -/
def divideSegment (p1 p2 : Point3D) (ratio : ℝ) : Point3D :=
  sorry

theorem volume_ratio_of_pyramids (P A B C : Point3D) : 
  let PABC := TriangularPyramid.mk P A B C
  let M := divideSegment P C (1/3)
  let N := divideSegment P B (2/3)
  let PAMN := TriangularPyramid.mk P A M N
  (volumeOfTriangularPyramid PAMN) / (volumeOfTriangularPyramid PABC) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_volume_ratio_of_pyramids_l2968_296826


namespace NUMINAMATH_CALUDE_intersection_sum_l2968_296859

/-- Given two lines y = nx + 3 and y = 5x + c that intersect at (4, 11), prove that n + c = -7 -/
theorem intersection_sum (n c : ℝ) : 
  (4 * n + 3 = 11) → (5 * 4 + c = 11) → n + c = -7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2968_296859


namespace NUMINAMATH_CALUDE_truck_distance_truck_distance_proof_l2968_296896

/-- The distance traveled by a truck given specific conditions -/
theorem truck_distance (truck_time car_time : ℝ) 
  (speed_difference distance_difference : ℝ) : ℝ :=
  let truck_speed := (car_time * speed_difference + distance_difference) / (car_time - truck_time)
  truck_speed * truck_time

/-- Prove that the truck travels 296 km under the given conditions -/
theorem truck_distance_proof : 
  truck_distance 8 5.5 18 6.5 = 296 := by
  sorry

end NUMINAMATH_CALUDE_truck_distance_truck_distance_proof_l2968_296896


namespace NUMINAMATH_CALUDE_abs_neg_three_equals_three_l2968_296877

theorem abs_neg_three_equals_three : abs (-3 : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_equals_three_l2968_296877


namespace NUMINAMATH_CALUDE_total_seashells_equation_l2968_296835

/-- The number of seashells Fred found on the beach -/
def total_seashells : ℕ := sorry

/-- The number of seashells Fred gave to Jessica -/
def seashells_given : ℕ := 25

/-- The number of seashells Fred has left -/
def seashells_left : ℕ := 22

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem total_seashells_equation : total_seashells = seashells_given + seashells_left := by sorry

end NUMINAMATH_CALUDE_total_seashells_equation_l2968_296835


namespace NUMINAMATH_CALUDE_parabola_vertex_l2968_296800

/-- The equation of a parabola in the xy-plane. -/
def ParabolaEquation (x y : ℝ) : Prop :=
  y^2 - 4*y + x + 7 = 0

/-- The vertex of a parabola. -/
def Vertex : ℝ × ℝ := (-3, 2)

/-- Theorem stating that the vertex of the parabola defined by y^2 - 4y + x + 7 = 0 is (-3, 2). -/
theorem parabola_vertex :
  ∀ x y : ℝ, ParabolaEquation x y → (x, y) = Vertex :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l2968_296800


namespace NUMINAMATH_CALUDE_triangle_area_l2968_296852

/-- Given a triangle with perimeter 32 and inradius 2.5, prove its area is 40 -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) 
  (h1 : perimeter = 32) 
  (h2 : inradius = 2.5) 
  (h3 : area = inradius * (perimeter / 2)) : 
  area = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2968_296852


namespace NUMINAMATH_CALUDE_remaining_macaroons_count_l2968_296895

/-- The number of remaining macaroons after eating some -/
def remaining_macaroons (initial_red initial_green eaten_green : ℕ) : ℕ :=
  let eaten_red := 2 * eaten_green
  let remaining_red := initial_red - eaten_red
  let remaining_green := initial_green - eaten_green
  remaining_red + remaining_green

/-- Theorem stating that the number of remaining macaroons is 45 -/
theorem remaining_macaroons_count :
  remaining_macaroons 50 40 15 = 45 := by
  sorry

end NUMINAMATH_CALUDE_remaining_macaroons_count_l2968_296895


namespace NUMINAMATH_CALUDE_ben_win_probability_l2968_296808

theorem ben_win_probability (p_lose : ℚ) (h1 : p_lose = 3/7) (h2 : p_lose + p_win = 1) : p_win = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ben_win_probability_l2968_296808


namespace NUMINAMATH_CALUDE_pastries_count_l2968_296827

/-- The number of pastries made by Lola and Lulu -/
def total_pastries (lola_cupcakes lola_poptarts lola_pies lulu_cupcakes lulu_poptarts lulu_pies : ℕ) : ℕ :=
  lola_cupcakes + lola_poptarts + lola_pies + lulu_cupcakes + lulu_poptarts + lulu_pies

/-- Theorem stating the total number of pastries made by Lola and Lulu -/
theorem pastries_count : total_pastries 13 10 8 16 12 14 = 73 := by
  sorry

end NUMINAMATH_CALUDE_pastries_count_l2968_296827


namespace NUMINAMATH_CALUDE_remainder_97_37_mod_100_l2968_296843

theorem remainder_97_37_mod_100 : 97^37 % 100 = 77 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_37_mod_100_l2968_296843


namespace NUMINAMATH_CALUDE_petya_bonus_points_l2968_296812

def calculate_bonus (final_score : ℕ) : ℕ :=
  if final_score < 1000 then
    (final_score * 20) / 100
  else if final_score < 2000 then
    200 + ((final_score - 1000) * 30) / 100
  else
    200 + 300 + ((final_score - 2000) * 50) / 100

theorem petya_bonus_points :
  calculate_bonus 2370 = 685 := by sorry

end NUMINAMATH_CALUDE_petya_bonus_points_l2968_296812


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l2968_296863

theorem sqrt_expression_equality : 
  (Real.sqrt 2 - Real.sqrt 3) ^ 2020 * (Real.sqrt 2 + Real.sqrt 3) ^ 2021 = Real.sqrt 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l2968_296863


namespace NUMINAMATH_CALUDE_basketball_free_throws_l2968_296865

theorem basketball_free_throws (deshawn kayla annieka : ℕ) : 
  deshawn = 12 →
  annieka = 14 →
  annieka = kayla - 4 →
  (kayla - deshawn) / deshawn * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l2968_296865


namespace NUMINAMATH_CALUDE_sum_of_tenth_powers_l2968_296831

theorem sum_of_tenth_powers (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = -1) :
  a^10 + b^10 = 123 := by sorry

end NUMINAMATH_CALUDE_sum_of_tenth_powers_l2968_296831


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2968_296897

theorem largest_n_satisfying_inequality :
  ∀ n : ℤ, (1 : ℚ) / 3 + (n : ℚ) / 7 < 1 ↔ n ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2968_296897


namespace NUMINAMATH_CALUDE_election_votes_l2968_296842

theorem election_votes (total_votes : ℕ) : 
  (total_votes : ℚ) * (62 : ℚ) / 100 - (total_votes : ℚ) * (38 : ℚ) / 100 = 336 →
  ((total_votes : ℚ) * (62 : ℚ) / 100).floor = 868 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l2968_296842


namespace NUMINAMATH_CALUDE_equation_solution_l2968_296813

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (4*x+1)*(3*x+1)*(2*x+1)*(x+1) - 3*x^4
  ∀ x : ℝ, f x = 0 ↔ x = (-5 + Real.sqrt 13) / 6 ∨ x = (-5 - Real.sqrt 13) / 6 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2968_296813


namespace NUMINAMATH_CALUDE_adjusted_work_hours_sufficient_l2968_296807

/-- Proves that working 27 hours per week for 9 weeks will result in at least $3000 earnings,
    given the initial plan of 20 hours per week for 12 weeks to earn $3000. -/
theorem adjusted_work_hours_sufficient
  (initial_hours_per_week : ℕ)
  (initial_weeks : ℕ)
  (target_earnings : ℕ)
  (missed_weeks : ℕ)
  (adjusted_hours_per_week : ℕ)
  (h1 : initial_hours_per_week = 20)
  (h2 : initial_weeks = 12)
  (h3 : target_earnings = 3000)
  (h4 : missed_weeks = 3)
  (h5 : adjusted_hours_per_week = 27) :
  (adjusted_hours_per_week : ℚ) * (initial_weeks - missed_weeks) ≥ (target_earnings : ℚ) := by
  sorry

#check adjusted_work_hours_sufficient

end NUMINAMATH_CALUDE_adjusted_work_hours_sufficient_l2968_296807


namespace NUMINAMATH_CALUDE_min_pizzas_to_cover_costs_l2968_296820

def car_cost : ℕ := 8000
def earnings_per_pizza : ℕ := 12
def gas_cost_per_delivery : ℕ := 4
def monthly_maintenance : ℕ := 200

theorem min_pizzas_to_cover_costs : 
  ∃ (p : ℕ), p = 1025 ∧ 
  (p * (earnings_per_pizza - gas_cost_per_delivery) ≥ car_cost + monthly_maintenance) ∧
  ∀ (q : ℕ), q < p → q * (earnings_per_pizza - gas_cost_per_delivery) < car_cost + monthly_maintenance :=
sorry

end NUMINAMATH_CALUDE_min_pizzas_to_cover_costs_l2968_296820


namespace NUMINAMATH_CALUDE_certain_number_proof_l2968_296869

theorem certain_number_proof : ∃ x : ℝ, (20 + x + 60) / 3 = (20 + 60 + 25) / 3 + 5 :=
by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l2968_296869


namespace NUMINAMATH_CALUDE_octagon_area_l2968_296881

theorem octagon_area (r : ℝ) (h : r = 4) : 
  let octagon_area := 8 * (1/2 * r * r * Real.sin (π/4))
  octagon_area = 32 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_octagon_area_l2968_296881


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l2968_296828

theorem integer_pairs_satisfying_equation :
  ∀ (x y : ℤ), x^5 + y^5 = (x + y)^3 ↔
    ((x = 0 ∧ y = 1) ∨
     (x = 1 ∧ y = 0) ∨
     (x = 0 ∧ y = -1) ∨
     (x = -1 ∧ y = 0) ∨
     (x = 2 ∧ y = 2) ∨
     (x = -2 ∧ y = -2) ∨
     (∃ (a : ℤ), x = a ∧ y = -a)) :=
by sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l2968_296828


namespace NUMINAMATH_CALUDE_cab_speed_ratio_l2968_296821

/-- Proves that the ratio of a cab's current speed to its usual speed is 5:6 -/
theorem cab_speed_ratio : 
  ∀ (usual_time current_time usual_speed current_speed : ℝ),
  usual_time = 25 →
  current_time = usual_time + 5 →
  usual_speed * usual_time = current_speed * current_time →
  current_speed / usual_speed = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cab_speed_ratio_l2968_296821


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2968_296840

theorem quadratic_factorization (a b : ℤ) :
  (∀ x, 25 * x^2 - 195 * x - 198 = (5 * x + a) * (5 * x + b)) →
  a + 2 * b = -420 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2968_296840


namespace NUMINAMATH_CALUDE_sqrt_calculations_l2968_296836

theorem sqrt_calculations :
  (Real.sqrt 18 - Real.sqrt 32 + Real.sqrt 2 = 0) ∧
  ((2 * Real.sqrt 3 + Real.sqrt 6) * (2 * Real.sqrt 3 - Real.sqrt 6) = 6) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculations_l2968_296836


namespace NUMINAMATH_CALUDE_sqrt_20_less_than_5_l2968_296862

theorem sqrt_20_less_than_5 : Real.sqrt 20 < 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_20_less_than_5_l2968_296862


namespace NUMINAMATH_CALUDE_fern_leaves_count_l2968_296818

/-- The number of leaves on all ferns -/
def total_leaves (num_ferns : ℕ) (fronds_per_fern : ℕ) (leaves_per_frond : ℕ) : ℕ :=
  num_ferns * fronds_per_fern * leaves_per_frond

/-- Theorem stating the total number of leaves on all ferns -/
theorem fern_leaves_count :
  total_leaves 6 7 30 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_fern_leaves_count_l2968_296818


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2968_296860

/-- Given a class of 15 students with an average age of 15 years,
    where 8 students have an average age of 14 years and 6 students
    have an average age of 16 years, the age of the 15th student is 17 years. -/
theorem fifteenth_student_age
  (total_students : Nat)
  (total_average_age : ℚ)
  (group1_students : Nat)
  (group1_average_age : ℚ)
  (group2_students : Nat)
  (group2_average_age : ℚ)
  (h1 : total_students = 15)
  (h2 : total_average_age = 15)
  (h3 : group1_students = 8)
  (h4 : group1_average_age = 14)
  (h5 : group2_students = 6)
  (h6 : group2_average_age = 16) :
  (total_students * total_average_age) - (group1_students * group1_average_age) - (group2_students * group2_average_age) = 17 := by
  sorry


end NUMINAMATH_CALUDE_fifteenth_student_age_l2968_296860


namespace NUMINAMATH_CALUDE_divisors_not_mult_6_l2968_296847

/-- The smallest integer satisfying the given conditions -/
def n : ℕ := 2^30 * 3^15 * 5^25

/-- n/2 is a perfect square -/
axiom n_div_2_is_square : ∃ k : ℕ, n / 2 = k^2

/-- n/4 is a perfect cube -/
axiom n_div_4_is_cube : ∃ j : ℕ, n / 4 = j^3

/-- n/5 is a perfect fifth -/
axiom n_div_5_is_fifth : ∃ m : ℕ, n / 5 = m^5

/-- The number of divisors of n -/
def total_divisors : ℕ := (30 + 1) * (15 + 1) * (25 + 1)

/-- The number of divisors of n that are multiples of 2 -/
def divisors_mult_2 : ℕ := (15 + 1) * (25 + 1)

/-- The number of divisors of n that are multiples of 3 -/
def divisors_mult_3 : ℕ := (29 + 1) * (25 + 1)

/-- Theorem: The number of divisors of n that are not multiples of 6 is 11740 -/
theorem divisors_not_mult_6 : total_divisors - divisors_mult_2 - divisors_mult_3 = 11740 := by
  sorry

end NUMINAMATH_CALUDE_divisors_not_mult_6_l2968_296847


namespace NUMINAMATH_CALUDE_whole_milk_fat_percentage_l2968_296882

/-- The percentage of fat in low-fat milk -/
def low_fat_milk_percentage : ℝ := 3

/-- The percentage difference between low-fat and semi-skimmed milk -/
def low_fat_semi_skimmed_difference : ℝ := 25

/-- The percentage difference between semi-skimmed and whole milk -/
def semi_skimmed_whole_difference : ℝ := 20

/-- The percentage of fat in whole milk -/
def whole_milk_percentage : ℝ := 5

theorem whole_milk_fat_percentage :
  (low_fat_milk_percentage / (1 - low_fat_semi_skimmed_difference / 100)) / (1 - semi_skimmed_whole_difference / 100) = whole_milk_percentage := by
  sorry

end NUMINAMATH_CALUDE_whole_milk_fat_percentage_l2968_296882


namespace NUMINAMATH_CALUDE_correct_calculation_l2968_296872

theorem correct_calculation (x : ℝ) : x * 7 = 126 → x / 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2968_296872


namespace NUMINAMATH_CALUDE_pizza_size_relation_l2968_296873

theorem pizza_size_relation (r : ℝ) (h : r > 0) :
  let R := r * Real.sqrt (1 + 156 / 100)
  (R - r) / r * 100 = 60 := by sorry

end NUMINAMATH_CALUDE_pizza_size_relation_l2968_296873


namespace NUMINAMATH_CALUDE_min_value_cubic_expression_l2968_296825

theorem min_value_cubic_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 15 * x - y = 22) :
  x^3 + y^3 - x^2 - y^2 ≥ 1 ∧
  (x^3 + y^3 - x^2 - y^2 = 1 ↔ x = 3/2 ∧ y = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_cubic_expression_l2968_296825


namespace NUMINAMATH_CALUDE_first_player_wins_l2968_296898

def Game := List Nat → List Nat

def validMove (n : Nat) (history : List Nat) : Prop :=
  n ∣ 328 ∧ n ∉ history ∧ ∀ m ∈ history, ¬(n ∣ m)

def gameOver (history : List Nat) : Prop :=
  328 ∈ history

def winningStrategy (strategy : Game) : Prop :=
  ∀ history : List Nat,
    ¬gameOver history →
    ∃ move,
      validMove move history ∧
      ∀ opponent_move,
        validMove opponent_move (move :: history) →
        gameOver (opponent_move :: move :: history)

theorem first_player_wins :
  ∃ strategy : Game, winningStrategy strategy :=
sorry

end NUMINAMATH_CALUDE_first_player_wins_l2968_296898


namespace NUMINAMATH_CALUDE_trajectory_and_circle_properties_l2968_296875

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the fixed line l
def line_l (x : ℝ) : Prop := x = 4

-- Define point F as the intersection of parabola and line l
def point_F : ℝ × ℝ := (2, 4)

-- Define the condition for point P
def condition_P (P Q F : ℝ × ℝ) : Prop :=
  let PQ := (Q.1 - P.1, Q.2 - P.2)
  let PF := (F.1 - P.1, F.2 - P.2)
  (PQ.1 + Real.sqrt 2 * PF.1, PQ.2 + Real.sqrt 2 * PF.2) • (PQ.1 - Real.sqrt 2 * PF.1, PQ.2 - Real.sqrt 2 * PF.2) = 0

-- Define the trajectory C
def trajectory_C (x y : ℝ) : Prop := x^2/8 + y^2/4 = 1

-- Define circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 8/3

-- Define the range of |AB|
def range_AB (ab : ℝ) : Prop := 4 * Real.sqrt 6 / 3 ≤ ab ∧ ab ≤ 2 * Real.sqrt 3

theorem trajectory_and_circle_properties :
  ∀ (P : ℝ × ℝ),
  (∃ (Q : ℝ × ℝ), line_l Q.1 ∧ condition_P P Q point_F) →
  trajectory_C P.1 P.2 ∧
  (∀ (A B : ℝ × ℝ),
    (circle_O A.1 A.2 ∧ line_l A.1 ∧ trajectory_C A.1 A.2) →
    (circle_O B.1 B.2 ∧ line_l B.1 ∧ trajectory_C B.1 B.2) →
    A ≠ B →
    (let O := (0, 0); let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2);
     (A.1 * B.1 + A.2 * B.2 = 0) → range_AB AB)) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_circle_properties_l2968_296875


namespace NUMINAMATH_CALUDE_average_z_squared_l2968_296816

theorem average_z_squared (z : ℝ) : 
  (0 + 2 * z^2 + 4 * z^2 + 8 * z^2 + 16 * z^2) / 5 = 6 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_average_z_squared_l2968_296816


namespace NUMINAMATH_CALUDE_population_growth_percentage_l2968_296849

/-- Proves that given an initial population of 1000, a 10% increase in the first year,
    and a final population of 1320 after two years, the percentage increase in the second year is 20%. -/
theorem population_growth_percentage (initial_population : ℕ) (first_year_increase : ℚ) 
  (final_population : ℕ) (second_year_increase : ℚ) : 
  initial_population = 1000 →
  first_year_increase = 1/10 →
  final_population = 1320 →
  (initial_population * (1 + first_year_increase) * (1 + second_year_increase) : ℚ) = final_population →
  second_year_increase = 1/5 := by
sorry

#eval (1000 : ℕ) * (1 + 1/10) * (1 + 1/5) -- Should evaluate to 1320

end NUMINAMATH_CALUDE_population_growth_percentage_l2968_296849


namespace NUMINAMATH_CALUDE_expected_value_of_new_balls_l2968_296864

/-- Represents the outcome of drawing balls in a ping pong match -/
inductive BallDraw
  | zero
  | one
  | two

/-- The probability mass function for the number of new balls drawn -/
def prob_new_balls (draw : BallDraw) : ℚ :=
  match draw with
  | BallDraw.zero => 37/100
  | BallDraw.one  => 54/100
  | BallDraw.two  => 9/100

/-- The number of new balls for each outcome -/
def num_new_balls (draw : BallDraw) : ℕ :=
  match draw with
  | BallDraw.zero => 0
  | BallDraw.one  => 1
  | BallDraw.two  => 2

/-- The expected value of new balls in the second draw -/
def expected_value : ℚ :=
  (prob_new_balls BallDraw.zero * num_new_balls BallDraw.zero) +
  (prob_new_balls BallDraw.one  * num_new_balls BallDraw.one)  +
  (prob_new_balls BallDraw.two  * num_new_balls BallDraw.two)

theorem expected_value_of_new_balls :
  expected_value = 18/25 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_new_balls_l2968_296864


namespace NUMINAMATH_CALUDE_lines_intersect_l2968_296810

/-- Represents a line in the form Ax + By + C = 0 --/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Determines if two lines are intersecting --/
def are_intersecting (l1 l2 : Line) : Prop :=
  l1.A * l2.B ≠ l2.A * l1.B

theorem lines_intersect : 
  let line1 : Line := { A := 3, B := -2, C := 5 }
  let line2 : Line := { A := 1, B := 3, C := 10 }
  are_intersecting line1 line2 := by
  sorry

end NUMINAMATH_CALUDE_lines_intersect_l2968_296810


namespace NUMINAMATH_CALUDE_square_side_lengths_l2968_296899

theorem square_side_lengths (s t : ℝ) : 
  (4 * s = 5 * (4 * t)) → -- perimeter of one square is 5 times the other
  (s + t = 60) →          -- sum of side lengths is 60
  ((s = 50 ∧ t = 10) ∨ (s = 10 ∧ t = 50)) := by
sorry

end NUMINAMATH_CALUDE_square_side_lengths_l2968_296899


namespace NUMINAMATH_CALUDE_a_formula_S_formula_min_t_value_l2968_296806

-- Define the arithmetic sequence and its sum
def a (n : ℕ) : ℚ := sorry
def S (n : ℕ) : ℚ := sorry

-- Define conditions
axiom S_9 : S 9 = 90
axiom S_15 : S 15 = 240

-- Define b_n and its sum
def b (n : ℕ) : ℚ := 1 / (2 * n * (n + 1))
def S_b (n : ℕ) : ℚ := (1 / 2) * (1 - 1 / (n + 1))

-- Theorem statements
theorem a_formula (n : ℕ) : a n = 2 * n := sorry

theorem S_formula (n : ℕ) : S n = n * (n + 1) := sorry

theorem min_t_value : 
  ∀ t : ℚ, (∀ n : ℕ, n > 0 → S_b n < t) → t ≥ 1/2 := sorry

end NUMINAMATH_CALUDE_a_formula_S_formula_min_t_value_l2968_296806


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l2968_296823

/-- Given a parabola with equation x = 2py^2 where p > 0, its focus has coordinates (1/(8p), 0) -/
theorem parabola_focus_coordinates (p : ℝ) (hp : p > 0) :
  let parabola := {(x, y) : ℝ × ℝ | x = 2 * p * y^2}
  ∃ (focus : ℝ × ℝ), focus ∈ parabola ∧ focus = (1 / (8 * p), 0) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l2968_296823


namespace NUMINAMATH_CALUDE_inequality_solution_equation_solution_range_l2968_296817

-- Define the function f
def f (x : ℝ) : ℝ := |2*x - 4| + |x + 1|

-- Theorem for the first part of the problem
theorem inequality_solution :
  {x : ℝ | f x ≤ 9} = {x : ℝ | -2 ≤ x ∧ x ≤ 4} :=
sorry

-- Theorem for the second part of the problem
theorem equation_solution_range :
  {a : ℝ | ∃ x ∈ Set.Icc 0 2, f x = -x^2 + a} = Set.Icc (19/4) 7 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_equation_solution_range_l2968_296817


namespace NUMINAMATH_CALUDE_goods_train_speed_calculation_l2968_296887

/-- The speed of the man's train in km/h -/
def man_train_speed : ℝ := 40

/-- The time it takes for the goods train to pass the man in seconds -/
def passing_time : ℝ := 12

/-- The length of the goods train in meters -/
def goods_train_length : ℝ := 350

/-- The speed of the goods train in km/h -/
def goods_train_speed : ℝ := 65

/-- Theorem stating that the given conditions imply the correct speed of the goods train -/
theorem goods_train_speed_calculation :
  man_train_speed = 40 ∧
  passing_time = 12 ∧
  goods_train_length = 350 →
  goods_train_speed = 65 := by
  sorry

#check goods_train_speed_calculation

end NUMINAMATH_CALUDE_goods_train_speed_calculation_l2968_296887


namespace NUMINAMATH_CALUDE_statement_1_statement_4_l2968_296801

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contained : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Statement ①
theorem statement_1 (m n : Line) (α : Plane) :
  perpendicular m n → perpendicularLP m α → ¬contained n α → parallel n α :=
sorry

-- Statement ④
theorem statement_4 (m n : Line) (α β : Plane) :
  perpendicular m n → perpendicularLP m α → perpendicularLP n β → perpendicularPP α β :=
sorry

end NUMINAMATH_CALUDE_statement_1_statement_4_l2968_296801


namespace NUMINAMATH_CALUDE_largest_garden_difference_l2968_296844

/-- Represents a rectangular garden with length and width in feet. -/
structure Garden where
  length : ℕ
  width : ℕ

/-- Calculates the area of a garden in square feet. -/
def gardenArea (g : Garden) : ℕ := g.length * g.width

/-- Alice's garden -/
def aliceGarden : Garden := { length := 30, width := 50 }

/-- Bob's garden -/
def bobGarden : Garden := { length := 35, width := 45 }

/-- Candace's garden -/
def candaceGarden : Garden := { length := 40, width := 40 }

theorem largest_garden_difference :
  let gardens := [aliceGarden, bobGarden, candaceGarden]
  let areas := gardens.map gardenArea
  let maxArea := areas.maximum?
  let minArea := areas.minimum?
  ∀ max min, maxArea = some max → minArea = some min →
    max - min = 100 := by sorry

end NUMINAMATH_CALUDE_largest_garden_difference_l2968_296844


namespace NUMINAMATH_CALUDE_bounded_diff_sequence_has_infinite_divisible_pairs_l2968_296870

/-- A sequence of positive integers with bounded differences -/
def BoundedDiffSequence (a : ℕ → ℕ) : Prop :=
  ∀ n, 0 < a (n + 1) - a n ∧ a (n + 1) - a n ≤ 2001

/-- The property of having infinitely many divisible pairs -/
def InfinitelyManyDivisiblePairs (a : ℕ → ℕ) : Prop :=
  ∀ k, ∃ p q, p > q ∧ q > k ∧ a q ∣ a p

/-- The main theorem -/
theorem bounded_diff_sequence_has_infinite_divisible_pairs
  (a : ℕ → ℕ) (h : BoundedDiffSequence a) :
  InfinitelyManyDivisiblePairs a :=
sorry

end NUMINAMATH_CALUDE_bounded_diff_sequence_has_infinite_divisible_pairs_l2968_296870


namespace NUMINAMATH_CALUDE_care_package_weight_l2968_296822

def final_weight (initial_weight : ℝ) : ℝ :=
  let weight_after_chocolate := initial_weight * 1.4
  let weight_after_snacks := weight_after_chocolate + 0.6 - 0.35 + 0.85
  let weight_after_cookies := weight_after_snacks * 1.6
  let weight_after_brownie_removal := weight_after_cookies - 0.45
  5 * initial_weight

theorem care_package_weight :
  let initial_weight := 1.25 + 0.75 + 1.5
  final_weight initial_weight = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_care_package_weight_l2968_296822


namespace NUMINAMATH_CALUDE_negation_equivalence_l2968_296857

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ (∀ x : ℝ, -1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2968_296857


namespace NUMINAMATH_CALUDE_solution_composition_l2968_296893

theorem solution_composition (x : ℝ) : 
  -- First solution composition
  let solution1_A := 0.20
  let solution1_B := 0.80
  -- Second solution composition
  let solution2_A := x
  let solution2_B := 0.70
  -- Mixture composition
  let mixture_solution1 := 0.80
  let mixture_solution2 := 0.20
  -- Final mixture composition of material A
  let final_mixture_A := 0.22
  -- Equation for material A in the final mixture
  solution1_A * mixture_solution1 + solution2_A * mixture_solution2 = final_mixture_A
  →
  x = 0.30 := by
sorry

end NUMINAMATH_CALUDE_solution_composition_l2968_296893


namespace NUMINAMATH_CALUDE_trig_identity_second_quadrant_l2968_296891

theorem trig_identity_second_quadrant (α : Real) 
  (h : π / 2 < α ∧ α < π) : 
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) + 
  Real.sqrt (1 - Real.sin α ^ 2) / Real.cos α = 1 := by sorry

end NUMINAMATH_CALUDE_trig_identity_second_quadrant_l2968_296891


namespace NUMINAMATH_CALUDE_remainder_zero_l2968_296838

def nines : ℕ := 10^20089 - 1
def threes : ℕ := 3 * (10^20083 - 1) / 9

theorem remainder_zero :
  (nines^2007 - threes^2007) % 11 = 0 := by sorry

end NUMINAMATH_CALUDE_remainder_zero_l2968_296838


namespace NUMINAMATH_CALUDE_sequence_divisibility_implies_zero_l2968_296814

theorem sequence_divisibility_implies_zero (x : ℕ → ℤ) :
  (∀ i j : ℕ, i ≠ j → (i * j : ℤ) ∣ (x i + x j)) →
  ∀ n : ℕ, x n = 0 :=
by sorry

end NUMINAMATH_CALUDE_sequence_divisibility_implies_zero_l2968_296814


namespace NUMINAMATH_CALUDE_max_product_under_constraint_l2968_296819

theorem max_product_under_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_constraint : 3 * x + 8 * y = 48) : x * y ≤ 24 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 8 * y = 48 ∧ x * y = 24 := by
  sorry

end NUMINAMATH_CALUDE_max_product_under_constraint_l2968_296819


namespace NUMINAMATH_CALUDE_root_range_implies_a_range_l2968_296837

theorem root_range_implies_a_range :
  ∀ a : ℝ,
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ x^2 - 4*x + 3*a^2 - 2 = 0) →
  a ∈ Set.Icc (-Real.sqrt (5/3)) (Real.sqrt (5/3)) :=
by sorry

end NUMINAMATH_CALUDE_root_range_implies_a_range_l2968_296837
