import Mathlib

namespace NUMINAMATH_CALUDE_average_temperature_is_42_4_l996_99612

/-- The average daily low temperature in Addington from September 15th to 19th, 2008 -/
def average_temperature : ℚ :=
  let temperatures : List ℤ := [40, 47, 45, 41, 39]
  (temperatures.sum : ℚ) / temperatures.length

/-- Theorem stating that the average temperature is 42.4°F -/
theorem average_temperature_is_42_4 : 
  average_temperature = 424/10 := by sorry

end NUMINAMATH_CALUDE_average_temperature_is_42_4_l996_99612


namespace NUMINAMATH_CALUDE_f_lower_bound_l996_99659

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x - log x

theorem f_lower_bound (a : ℝ) (h : a ≤ -1 / Real.exp 2) :
  ∀ x > 0, f a x ≥ 2 * a * x - x * Real.exp (a * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_f_lower_bound_l996_99659


namespace NUMINAMATH_CALUDE_abs_value_sum_difference_l996_99614

theorem abs_value_sum_difference (a b : ℝ) : 
  (|a| = 2) → (|b| = 4) → (a + b < 0) → (a - b = 2 ∨ a - b = 6) := by
  sorry

end NUMINAMATH_CALUDE_abs_value_sum_difference_l996_99614


namespace NUMINAMATH_CALUDE_uneven_gender_probability_l996_99632

/-- The number of children in the family -/
def num_children : ℕ := 8

/-- The probability of a child being male (or female) -/
def gender_prob : ℚ := 1/2

/-- The total number of possible gender combinations -/
def total_combinations : ℕ := 2^num_children

/-- The number of combinations with an even split of genders -/
def even_split_combinations : ℕ := Nat.choose num_children (num_children / 2)

/-- The probability of having an uneven number of sons and daughters -/
def prob_uneven : ℚ := 1 - (even_split_combinations : ℚ) / total_combinations

theorem uneven_gender_probability :
  prob_uneven = 93/128 :=
sorry

end NUMINAMATH_CALUDE_uneven_gender_probability_l996_99632


namespace NUMINAMATH_CALUDE_negation_p_sufficient_not_necessary_for_negation_q_l996_99675

theorem negation_p_sufficient_not_necessary_for_negation_q :
  ∃ (x : ℝ),
    (∀ x, (|x + 1| > 0 → (5*x - 6 > x^2)) →
      (x = -1 → (x ≤ 2 ∨ x ≥ 3)) ∧
      ¬(x ≤ 2 ∨ x ≥ 3 → x = -1)) :=
by sorry

end NUMINAMATH_CALUDE_negation_p_sufficient_not_necessary_for_negation_q_l996_99675


namespace NUMINAMATH_CALUDE_sqrt_36_times_sqrt_16_l996_99631

theorem sqrt_36_times_sqrt_16 : Real.sqrt (36 * Real.sqrt 16) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_36_times_sqrt_16_l996_99631


namespace NUMINAMATH_CALUDE_drug_price_reduction_l996_99670

theorem drug_price_reduction (initial_price final_price : ℝ) 
  (h1 : initial_price = 50)
  (h2 : final_price = 40.5)
  (h3 : final_price = initial_price * (1 - x)^2)
  (h4 : 0 < x ∧ x < 1) :
  x = 0.1 := by sorry

end NUMINAMATH_CALUDE_drug_price_reduction_l996_99670


namespace NUMINAMATH_CALUDE_cost_price_calculation_l996_99604

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 600)
  (h2 : profit_percentage = 25) : 
  ∃ (cost_price : ℝ), 
    selling_price = cost_price * (1 + profit_percentage / 100) ∧ 
    cost_price = 480 :=
by
  sorry

#check cost_price_calculation

end NUMINAMATH_CALUDE_cost_price_calculation_l996_99604


namespace NUMINAMATH_CALUDE_photo_frame_border_area_l996_99628

theorem photo_frame_border_area :
  let photo_height : ℝ := 12
  let photo_width : ℝ := 14
  let frame_width : ℝ := 3
  let framed_height : ℝ := photo_height + 2 * frame_width
  let framed_width : ℝ := photo_width + 2 * frame_width
  let photo_area : ℝ := photo_height * photo_width
  let framed_area : ℝ := framed_height * framed_width
  let border_area : ℝ := framed_area - photo_area
  border_area = 192 := by sorry

end NUMINAMATH_CALUDE_photo_frame_border_area_l996_99628


namespace NUMINAMATH_CALUDE_octagon_triangle_area_ratio_l996_99678

theorem octagon_triangle_area_ratio (s_o s_t : ℝ) (h : s_o > 0) (h' : s_t > 0) :
  (2 * s_o^2 * (1 + Real.sqrt 2) = s_t^2 * Real.sqrt 3 / 4) →
  s_t / s_o = Real.sqrt (8 + 8 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_octagon_triangle_area_ratio_l996_99678


namespace NUMINAMATH_CALUDE_inequality_problem_l996_99617

theorem inequality_problem (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  (b / a > c / a) ∧
  (c * (b - a) > 0) ∧
  (a * c * (a - c) < 0) ∧
  ¬ (∀ b, c * b^2 < a * b^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l996_99617


namespace NUMINAMATH_CALUDE_runner_problem_l996_99672

theorem runner_problem (v : ℝ) (h1 : v > 0) :
  let t1 := 20 / v
  let t2 := 40 / v
  t2 = t1 + 4 →
  t2 = 8 := by
sorry

end NUMINAMATH_CALUDE_runner_problem_l996_99672


namespace NUMINAMATH_CALUDE_one_minus_repeating_six_eq_one_third_l996_99698

/-- The decimal 0.666... (repeating 6) --/
def repeating_six : ℚ := 2/3

/-- Proof that 1 - 0.666... (repeating 6) equals 1/3 --/
theorem one_minus_repeating_six_eq_one_third : 1 - repeating_six = (1 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_one_minus_repeating_six_eq_one_third_l996_99698


namespace NUMINAMATH_CALUDE_total_area_is_60_l996_99620

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The four rectangles that compose the figure -/
def rectangles : List Rectangle := [
  { width := 5, height := 5 },
  { width := 5, height := 3 },
  { width := 5, height := 2 },
  { width := 5, height := 2 }
]

/-- Theorem: The total area of the figure is 60 square units -/
theorem total_area_is_60 : 
  (rectangles.map Rectangle.area).sum = 60 := by sorry

end NUMINAMATH_CALUDE_total_area_is_60_l996_99620


namespace NUMINAMATH_CALUDE_closer_to_center_is_enclosed_by_bisectors_l996_99680

/-- A rectangle in a 2D plane -/
structure Rectangle where
  a : ℝ
  b : ℝ

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The set of points closer to the center of a rectangle than to any of its vertices -/
def CloserToCenter (r : Rectangle) : Set Point :=
  { p : Point | p.x^2 + p.y^2 < (p.x - r.a)^2 + (p.y - r.b)^2 ∧
                p.x^2 + p.y^2 < (p.x - r.a)^2 + (p.y + r.b)^2 ∧
                p.x^2 + p.y^2 < (p.x + r.a)^2 + (p.y + r.b)^2 ∧
                p.x^2 + p.y^2 < (p.x + r.a)^2 + (p.y - r.b)^2 }

/-- Theorem stating that the set of points closer to the center is enclosed by perpendicular bisectors -/
theorem closer_to_center_is_enclosed_by_bisectors (r : Rectangle) :
  ∃ (bisectors : Set Point), CloserToCenter r = bisectors :=
sorry

end NUMINAMATH_CALUDE_closer_to_center_is_enclosed_by_bisectors_l996_99680


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l996_99606

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l996_99606


namespace NUMINAMATH_CALUDE_jordana_age_proof_l996_99607

/-- Jennifer's age in ten years -/
def jennifer_future_age : ℕ := 30

/-- Number of years in the future we're considering -/
def years_ahead : ℕ := 10

/-- Jordana's age relative to Jennifer's in the future -/
def jordana_relative_age : ℕ := 3

/-- Calculate Jordana's current age -/
def jordana_current_age : ℕ :=
  jennifer_future_age * jordana_relative_age - years_ahead

theorem jordana_age_proof :
  jordana_current_age = 80 := by
  sorry

end NUMINAMATH_CALUDE_jordana_age_proof_l996_99607


namespace NUMINAMATH_CALUDE_tangent_slope_and_function_lower_bound_l996_99602

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.exp x - a * Real.log x

theorem tangent_slope_and_function_lower_bound 
  (a : ℝ) 
  (h1 : ∀ x > 0, HasDerivAt (f a) ((2 * x + x^2) * Real.exp x - a / x) x) 
  (h2 : HasDerivAt (f a) (3 * Real.exp 1 - 1) 1) :
  a = 1 ∧ ∀ x > 0, f a x > 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_and_function_lower_bound_l996_99602


namespace NUMINAMATH_CALUDE_min_bushes_for_pumpkins_l996_99644

/-- Represents the number of containers of raspberries per bush -/
def containers_per_bush : ℕ := 10

/-- Represents the number of containers of raspberries needed to trade for 3 pumpkins -/
def containers_per_trade : ℕ := 6

/-- Represents the number of pumpkins obtained from one trade -/
def pumpkins_per_trade : ℕ := 3

/-- Represents the target number of pumpkins -/
def target_pumpkins : ℕ := 72

/-- 
Proves that the minimum number of bushes needed to obtain at least the target number of pumpkins
is 15, given the defined ratios of containers per bush and pumpkins per trade.
-/
theorem min_bushes_for_pumpkins :
  ∃ (n : ℕ), n * containers_per_bush * pumpkins_per_trade ≥ target_pumpkins * containers_per_trade ∧
  ∀ (m : ℕ), m * containers_per_bush * pumpkins_per_trade ≥ target_pumpkins * containers_per_trade → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_bushes_for_pumpkins_l996_99644


namespace NUMINAMATH_CALUDE_ice_cream_consumption_l996_99677

theorem ice_cream_consumption (friday_amount saturday_amount total_amount : ℝ) :
  friday_amount = 3.25 →
  saturday_amount = 0.25 →
  total_amount = friday_amount + saturday_amount →
  total_amount = 3.50 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_consumption_l996_99677


namespace NUMINAMATH_CALUDE_sugar_spilled_calculation_l996_99600

/-- The amount of sugar Pamela bought, in ounces -/
def original_amount : ℝ := 9.8

/-- The amount of sugar Pamela has left, in ounces -/
def amount_left : ℝ := 4.6

/-- The amount of sugar Pamela spilled, in ounces -/
def amount_spilled : ℝ := original_amount - amount_left

theorem sugar_spilled_calculation :
  amount_spilled = 5.2 := by
  sorry

end NUMINAMATH_CALUDE_sugar_spilled_calculation_l996_99600


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l996_99640

/-- The correct quadratic equation given erroneous roots -/
theorem correct_quadratic_equation 
  (root1_student1 root2_student1 : ℝ)
  (root1_student2 root2_student2 : ℝ)
  (h1 : root1_student1 = 5 ∧ root2_student1 = 3)
  (h2 : root1_student2 = -12 ∧ root2_student2 = -4) :
  ∃ (a b c : ℝ), a ≠ 0 ∧ 
    (∀ x, a * x^2 + b * x + c = 0 ↔ x^2 - 8 * x + 48 = 0) :=
sorry

#check correct_quadratic_equation

end NUMINAMATH_CALUDE_correct_quadratic_equation_l996_99640


namespace NUMINAMATH_CALUDE_dinner_cost_calculation_l996_99655

/-- The total amount paid for a dinner, given the food cost, sales tax rate, and tip rate. -/
def total_dinner_cost (food_cost : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  food_cost + (food_cost * sales_tax_rate) + (food_cost * tip_rate)

/-- Theorem stating that the total dinner cost is $35.85 given the specified conditions. -/
theorem dinner_cost_calculation :
  total_dinner_cost 30 0.095 0.10 = 35.85 := by
  sorry

end NUMINAMATH_CALUDE_dinner_cost_calculation_l996_99655


namespace NUMINAMATH_CALUDE_m_range_l996_99605

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/2 + y^2/m = 1 ∧ m > 2

def q (m : ℝ) : Prop := ∃ (x y : ℝ), (m+4)*x^2 - (m+2)*y^2 = (m+4)*(m+2)

-- Define the range of m
def range_m (m : ℝ) : Prop := m < -4 ∨ (-2 < m ∧ m ≤ 2)

-- State the theorem
theorem m_range : 
  (∀ m : ℝ, ¬(p m ∧ q m)) → 
  (∀ m : ℝ, p m → q m) → 
  (∀ m : ℝ, range_m m ↔ (¬(p m) ∧ q m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l996_99605


namespace NUMINAMATH_CALUDE_distance_between_points_l996_99641

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (2, 5)
  let p2 : ℝ × ℝ := (5, 1)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l996_99641


namespace NUMINAMATH_CALUDE_power_of_128_four_sevenths_l996_99657

theorem power_of_128_four_sevenths : (128 : ℝ) ^ (4/7 : ℝ) = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_128_four_sevenths_l996_99657


namespace NUMINAMATH_CALUDE_exists_large_class_l996_99624

/-- A club of students -/
structure Club where
  students : Finset Nat
  classes : Nat → Finset Nat
  total_students : students.card = 60
  classmate_property : ∀ s : Finset Nat, s ⊆ students → s.card = 10 →
    ∃ c : Nat, (s ∩ classes c).card ≥ 3

/-- The main theorem -/
theorem exists_large_class (club : Club) :
  ∃ c : Nat, (club.students ∩ club.classes c).card ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_exists_large_class_l996_99624


namespace NUMINAMATH_CALUDE_ribbons_left_l996_99699

theorem ribbons_left (initial : ℕ) (morning : ℕ) (afternoon : ℕ) : 
  initial = 38 → morning = 14 → afternoon = 16 → initial - (morning + afternoon) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ribbons_left_l996_99699


namespace NUMINAMATH_CALUDE_largest_base6_5digit_in_base10_l996_99685

/-- The largest five-digit number in base 6 -/
def largest_base6_5digit : ℕ := 5 * 6^4 + 5 * 6^3 + 5 * 6^2 + 5 * 6^1 + 5 * 6^0

/-- Theorem: The largest five-digit number in base 6 equals 7775 in base 10 -/
theorem largest_base6_5digit_in_base10 : largest_base6_5digit = 7775 := by
  sorry

end NUMINAMATH_CALUDE_largest_base6_5digit_in_base10_l996_99685


namespace NUMINAMATH_CALUDE_complex_modulus_l996_99603

theorem complex_modulus (a b : ℝ) : 
  (1 + 2*Complex.I) / (Complex.mk a b) = 1 + Complex.I → 
  Complex.abs (Complex.mk a b) = Real.sqrt 10 / 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l996_99603


namespace NUMINAMATH_CALUDE_magnitude_relationship_l996_99676

theorem magnitude_relationship (a b c : ℝ) : 
  a = Real.sin (46 * π / 180) →
  b = Real.cos (46 * π / 180) →
  c = Real.cos (36 * π / 180) →
  c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_magnitude_relationship_l996_99676


namespace NUMINAMATH_CALUDE_circular_section_area_l996_99645

theorem circular_section_area (r : ℝ) (d : ℝ) (h : r = 5 ∧ d = 3) :
  let section_radius : ℝ := Real.sqrt (r^2 - d^2)
  π * section_radius^2 = 16 * π :=
by sorry

end NUMINAMATH_CALUDE_circular_section_area_l996_99645


namespace NUMINAMATH_CALUDE_max_value_problem_l996_99634

theorem max_value_problem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y + z = 1) :
  ∀ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a^2 + b + c = 1 → x + y^3 + z^4 ≤ a + b^3 + c^4 → x + y^3 + z^4 ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_problem_l996_99634


namespace NUMINAMATH_CALUDE_heesu_has_greatest_sum_l996_99682

-- Define the card values for each person
def sora_cards : List Nat := [4, 6]
def heesu_cards : List Nat := [7, 5]
def jiyeon_cards : List Nat := [3, 8]

-- Define a function to calculate the sum of cards
def sum_cards (cards : List Nat) : Nat :=
  cards.sum

-- Theorem statement
theorem heesu_has_greatest_sum :
  sum_cards heesu_cards > sum_cards sora_cards ∧
  sum_cards heesu_cards > sum_cards jiyeon_cards :=
by
  sorry


end NUMINAMATH_CALUDE_heesu_has_greatest_sum_l996_99682


namespace NUMINAMATH_CALUDE_two_point_distribution_p_values_l996_99696

/-- A two-point distribution random variable -/
structure TwoPointDistribution where
  p : ℝ
  prob_x_eq_one : p ∈ Set.Icc 0 1

/-- The variance of a two-point distribution -/
def variance (X : TwoPointDistribution) : ℝ := X.p - X.p^2

theorem two_point_distribution_p_values (X : TwoPointDistribution) 
  (h : variance X = 2/9) : X.p = 1/3 ∨ X.p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_two_point_distribution_p_values_l996_99696


namespace NUMINAMATH_CALUDE_apps_deleted_l996_99681

theorem apps_deleted (initial_apps new_apps remaining_apps : ℕ) : 
  initial_apps = 10 →
  new_apps = 11 →
  remaining_apps = 4 →
  initial_apps + new_apps - remaining_apps = 17 :=
by sorry

end NUMINAMATH_CALUDE_apps_deleted_l996_99681


namespace NUMINAMATH_CALUDE_roots_equation_l996_99609

theorem roots_equation (x₁ x₂ : ℝ) : 
  x₁^2 + x₁ - 4 = 0 → x₂^2 + x₂ - 4 = 0 → x₁^3 - 5*x₂^2 + 10 = -19 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_l996_99609


namespace NUMINAMATH_CALUDE_milk_jars_theorem_l996_99643

/-- Calculates the number of jars of milk good for sale given the conditions of Logan's father's milk business. -/
def good_milk_jars (normal_cartons : ℕ) (jars_per_carton : ℕ) (less_cartons : ℕ) 
  (damaged_cartons : ℕ) (damaged_jars_per_carton : ℕ) (totally_damaged_cartons : ℕ) : ℕ :=
  let received_cartons := normal_cartons - less_cartons
  let total_jars := received_cartons * jars_per_carton
  let partially_damaged_jars := damaged_cartons * damaged_jars_per_carton
  let totally_damaged_jars := totally_damaged_cartons * jars_per_carton
  let total_damaged_jars := partially_damaged_jars + totally_damaged_jars
  total_jars - total_damaged_jars

/-- Theorem stating that under the given conditions, the number of good milk jars for sale is 565. -/
theorem milk_jars_theorem : good_milk_jars 50 20 20 5 3 1 = 565 := by
  sorry

end NUMINAMATH_CALUDE_milk_jars_theorem_l996_99643


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l996_99683

-- Problem 1
theorem problem_1 : (-1)^10 * 2 + (-2)^3 / 4 = 0 := by sorry

-- Problem 2
theorem problem_2 : (-24) * (5/6 - 4/3 + 3/8) = 3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l996_99683


namespace NUMINAMATH_CALUDE_target_line_properties_l996_99669

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y + 4 = 0
def line2 (x y : ℝ) : Prop := x - y + 5 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y = 0
def target_line (x y : ℝ) : Prop := 2 * x + y - 8 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define perpendicularity
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem target_line_properties :
  ∃ (x y : ℝ),
    intersection_point x y ∧
    target_line x y ∧
    ∃ (m1 m2 : ℝ),
      (∀ (x y : ℝ), line3 x y ↔ y = m1 * x) ∧
      (∀ (x y : ℝ), target_line x y ↔ y = m2 * x + (y - m2 * x)) ∧
      perpendicular m1 m2 :=
sorry

end NUMINAMATH_CALUDE_target_line_properties_l996_99669


namespace NUMINAMATH_CALUDE_stickers_bought_from_store_l996_99697

/-- Calculates the number of stickers Mika bought from the store -/
theorem stickers_bought_from_store 
  (initial : ℝ) 
  (birthday : ℝ) 
  (from_sister : ℝ) 
  (from_mother : ℝ) 
  (total : ℝ) 
  (h1 : initial = 20.0)
  (h2 : birthday = 20.0)
  (h3 : from_sister = 6.0)
  (h4 : from_mother = 58.0)
  (h5 : total = 130.0) :
  total - (initial + birthday + from_sister + from_mother) = 46.0 := by
  sorry

end NUMINAMATH_CALUDE_stickers_bought_from_store_l996_99697


namespace NUMINAMATH_CALUDE_increasing_cubic_function_condition_l996_99626

/-- A function f(x) = x³ - ax - 1 is increasing for all real x if and only if a ≤ 0 -/
theorem increasing_cubic_function_condition (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => x^3 - a*x - 1)) ↔ a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_condition_l996_99626


namespace NUMINAMATH_CALUDE_min_value_inequality_l996_99673

theorem min_value_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (b + d) + 1 / (c + d)) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l996_99673


namespace NUMINAMATH_CALUDE_right_triangle_area_l996_99689

/-- 
Given a right triangle with hypotenuse c, where the projection of the right angle 
vertex onto the hypotenuse divides it into two segments x and (c-x) such that 
(c-x)/x = x/c, the area of the triangle is (c^2 * sqrt(sqrt(5) - 2)) / 2.
-/
theorem right_triangle_area (c : ℝ) (h : c > 0) : 
  ∃ x : ℝ, 0 < x ∧ x < c ∧ (c - x) / x = x / c ∧ 
  (c^2 * Real.sqrt (Real.sqrt 5 - 2)) / 2 = 
  (1 / 2) * c * Real.sqrt (c * x - x^2) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l996_99689


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_lower_bound_l996_99619

/-- Given plane vectors a, b, and c satisfying certain dot product conditions,
    prove that the magnitude of their sum is at least 4. -/
theorem vector_sum_magnitude_lower_bound
  (a b c : ℝ × ℝ)
  (ha : a.1 * a.1 + a.2 * a.2 = 1)
  (hab : a.1 * b.1 + a.2 * b.2 = 1)
  (hac : a.1 * c.1 + a.2 * c.2 = 2)
  (hbc : b.1 * c.1 + b.2 * c.2 = 1) :
  (a.1 + b.1 + c.1)^2 + (a.2 + b.2 + c.2)^2 ≥ 16 := by
  sorry

#check vector_sum_magnitude_lower_bound

end NUMINAMATH_CALUDE_vector_sum_magnitude_lower_bound_l996_99619


namespace NUMINAMATH_CALUDE_inverse_proportional_properties_l996_99664

/-- Given two inverse proportional functions y = k/x and y = 1/x, where k > 0,
    and a point P(a, k/a) on y = k/x, with a > 0, we define:
    C(a, 0), A(a, 1/a), D(0, k/a), B(a/k, k/a) -/
theorem inverse_proportional_properties (k a : ℝ) (hk : k > 0) (ha : a > 0) :
  let P := (a, k / a)
  let C := (a, 0)
  let A := (a, 1 / a)
  let D := (0, k / a)
  let B := (a / k, k / a)
  let triangle_area (p q r : ℝ × ℝ) := (abs ((p.1 - r.1) * (q.2 - r.2) - (q.1 - r.1) * (p.2 - r.2))) / 2
  let quadrilateral_area (p q r s : ℝ × ℝ) := triangle_area p q r + triangle_area p r s
  -- 1. The areas of triangles ODB and OCA are equal to 1/2
  (triangle_area (0, 0) D B = 1 / 2 ∧ triangle_area (0, 0) C A = 1 / 2) ∧
  -- 2. The area of quadrilateral OAPB is equal to k - 1
  (quadrilateral_area (0, 0) A P B = k - 1) ∧
  -- 3. If k = 2, then A is the midpoint of PC and B is the midpoint of PD
  (k = 2 → (A.2 - C.2 = P.2 - A.2 ∧ B.1 - D.1 = P.1 - B.1)) := by
  sorry


end NUMINAMATH_CALUDE_inverse_proportional_properties_l996_99664


namespace NUMINAMATH_CALUDE_servant_salary_l996_99651

/-- Calculates the money received by a servant, excluding the turban -/
theorem servant_salary (annual_salary : ℝ) (turban_price : ℝ) (months_worked : ℝ) : 
  annual_salary = 90 →
  turban_price = 10 →
  months_worked = 9 →
  (months_worked / 12) * (annual_salary + turban_price) - turban_price = 65 :=
by sorry

end NUMINAMATH_CALUDE_servant_salary_l996_99651


namespace NUMINAMATH_CALUDE_dodecahedron_coloring_count_l996_99654

/-- The number of faces in a regular dodecahedron -/
def num_faces : ℕ := 12

/-- The order of the rotational symmetry group of a regular dodecahedron -/
def dodecahedron_symmetry_order : ℕ := 60

/-- The number of distinguishable colorings of a regular dodecahedron 
    with different colors for each face, considering rotational symmetries -/
def distinguishable_colorings : ℕ := (Nat.factorial (num_faces - 1)) / dodecahedron_symmetry_order

theorem dodecahedron_coloring_count :
  distinguishable_colorings = 665280 := by sorry

end NUMINAMATH_CALUDE_dodecahedron_coloring_count_l996_99654


namespace NUMINAMATH_CALUDE_simplify_expression_l996_99658

theorem simplify_expression (x y : ℝ) : 3*x + 6*x + 9*x + 12*y + 15*y + 18 + 21 = 18*x + 27*y + 39 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l996_99658


namespace NUMINAMATH_CALUDE_seven_fifths_of_negative_eighteen_fourths_l996_99630

theorem seven_fifths_of_negative_eighteen_fourths :
  (7 : ℚ) / 5 * (-18 : ℚ) / 4 = (-63 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_seven_fifths_of_negative_eighteen_fourths_l996_99630


namespace NUMINAMATH_CALUDE_seating_arrangement_probability_l996_99635

/-- Represents the number of delegates --/
def total_delegates : ℕ := 12

/-- Represents the number of countries --/
def num_countries : ℕ := 3

/-- Represents the number of delegates per country --/
def delegates_per_country : ℕ := 4

/-- Calculates the probability of the seating arrangement --/
noncomputable def seating_probability : ℚ :=
  409 / 500

/-- Theorem stating the probability of the specific seating arrangement --/
theorem seating_arrangement_probability :
  let total_arrangements := (total_delegates.factorial) / (delegates_per_country.factorial ^ num_countries)
  let favorable_arrangements := total_arrangements - (num_countries * total_delegates * 
    ((total_delegates - delegates_per_country).factorial / (delegates_per_country.factorial ^ (num_countries - 1))) -
    (num_countries * (num_countries - 1) / 2 * total_delegates * (total_delegates - 2)) +
    (total_delegates * (num_countries - 1)))
  (favorable_arrangements : ℚ) / total_arrangements = seating_probability :=
sorry

end NUMINAMATH_CALUDE_seating_arrangement_probability_l996_99635


namespace NUMINAMATH_CALUDE_solve_system_l996_99639

theorem solve_system (x y : ℚ) (eq1 : 3 * x - 2 * y = 7) (eq2 : 2 * x + 3 * y = 8) : x = 37 / 13 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l996_99639


namespace NUMINAMATH_CALUDE_negation_of_not_even_numbers_l996_99690

theorem negation_of_not_even_numbers (a b : ℤ) : 
  ¬(¬Even a ∧ ¬Even b) ↔ (Even a ∨ Even b) :=
sorry

end NUMINAMATH_CALUDE_negation_of_not_even_numbers_l996_99690


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l996_99663

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a3 : a 3 = 4) 
  (h_d : ∃ d : ℝ, d = -2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 10 - 2 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l996_99663


namespace NUMINAMATH_CALUDE_smallest_k_and_largest_base_l996_99622

theorem smallest_k_and_largest_base : ∃ (b : ℕ), 
  (64 ^ 7 > b ^ 20) ∧ 
  (∀ (x : ℕ), x > b → 64 ^ 7 ≤ x ^ 20) ∧ 
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_and_largest_base_l996_99622


namespace NUMINAMATH_CALUDE_set_equals_open_interval_l996_99627

theorem set_equals_open_interval :
  {x : ℝ | -1 < x ∧ x < 1} = Set.Ioo (-1 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_set_equals_open_interval_l996_99627


namespace NUMINAMATH_CALUDE_negation_equal_area_congruent_is_true_l996_99666

-- Define a type for triangles
def Triangle : Type := sorry

-- Define a function for the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem stating that the negation of "Triangles with equal areas are congruent" is true
theorem negation_equal_area_congruent_is_true :
  ¬(∀ t1 t2 : Triangle, area t1 = area t2 → congruent t1 t2) :=
sorry

end NUMINAMATH_CALUDE_negation_equal_area_congruent_is_true_l996_99666


namespace NUMINAMATH_CALUDE_transistor_count_scientific_notation_l996_99667

/-- The number of transistors in a Huawei Kirin 990 processor -/
def transistor_count : ℝ := 12000000000

/-- The scientific notation representation of the transistor count -/
def scientific_notation : ℝ := 1.2 * (10 ^ 10)

theorem transistor_count_scientific_notation : 
  transistor_count = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_transistor_count_scientific_notation_l996_99667


namespace NUMINAMATH_CALUDE_water_bottles_needed_l996_99650

theorem water_bottles_needed (people : ℕ) (trip_hours : ℕ) (bottles_per_person_per_hour : ℚ) : 
  people = 10 → trip_hours = 24 → bottles_per_person_per_hour = 1/2 →
  (people : ℚ) * trip_hours * bottles_per_person_per_hour = 120 :=
by sorry

end NUMINAMATH_CALUDE_water_bottles_needed_l996_99650


namespace NUMINAMATH_CALUDE_current_speed_l996_99647

/-- Given a boat's upstream and downstream speeds, calculate the current's speed -/
theorem current_speed (upstream_time : ℝ) (downstream_time : ℝ) :
  upstream_time = 30 →
  downstream_time = 12 →
  let upstream_speed := 60 / upstream_time
  let downstream_speed := 60 / downstream_time
  (downstream_speed - upstream_speed) / 2 = 1.5 := by
  sorry

#check current_speed

end NUMINAMATH_CALUDE_current_speed_l996_99647


namespace NUMINAMATH_CALUDE_functional_equation_implies_identity_l996_99616

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2

/-- The main theorem: if f satisfies the equation, then f is the identity function -/
theorem functional_equation_implies_identity (f : ℝ → ℝ) 
  (h : SatisfiesEquation f) : ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_implies_identity_l996_99616


namespace NUMINAMATH_CALUDE_bart_monday_surveys_l996_99668

/-- The number of surveys Bart finished on Monday -/
def monday_surveys : ℕ := 3

/-- The amount earned per question in dollars -/
def earnings_per_question : ℚ := 1/5

/-- The number of questions in each survey -/
def questions_per_survey : ℕ := 10

/-- The number of surveys Bart finished on Tuesday -/
def tuesday_surveys : ℕ := 4

/-- The total amount Bart earned over Monday and Tuesday in dollars -/
def total_earnings : ℚ := 14

theorem bart_monday_surveys :
  monday_surveys * questions_per_survey * earnings_per_question +
  tuesday_surveys * questions_per_survey * earnings_per_question =
  total_earnings :=
sorry

end NUMINAMATH_CALUDE_bart_monday_surveys_l996_99668


namespace NUMINAMATH_CALUDE_problem_statement_l996_99621

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (a * b ≤ 1) ∧ (2^a + 2^b ≥ 2 * Real.sqrt 2) ∧ (1/a + 4/b ≥ 9/2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l996_99621


namespace NUMINAMATH_CALUDE_borgnine_leg_count_l996_99638

/-- The number of legs Borgnine wants to see at the zoo -/
def total_legs (chimps lions lizards tarantulas : ℕ) : ℕ :=
  2 * chimps + 4 * lions + 4 * lizards + 8 * tarantulas

/-- Theorem stating the total number of legs Borgnine wants to see -/
theorem borgnine_leg_count : total_legs 12 8 5 125 = 1076 := by
  sorry

end NUMINAMATH_CALUDE_borgnine_leg_count_l996_99638


namespace NUMINAMATH_CALUDE_empty_solution_set_l996_99692

theorem empty_solution_set (a : ℝ) :
  (∀ x : ℝ, ¬(|x - 1| - |x + 2| < a)) ↔ a ≤ -3 :=
by sorry

end NUMINAMATH_CALUDE_empty_solution_set_l996_99692


namespace NUMINAMATH_CALUDE_no_valid_area_codes_l996_99623

def is_valid_digit (d : ℕ) : Prop := d = 2 ∨ d = 4 ∨ d = 3 ∨ d = 5

def is_valid_area_code (code : Fin 4 → ℕ) : Prop :=
  ∀ i, is_valid_digit (code i)

def product_of_digits (code : Fin 4 → ℕ) : ℕ :=
  (code 0) * (code 1) * (code 2) * (code 3)

theorem no_valid_area_codes :
  ¬∃ (code : Fin 4 → ℕ), is_valid_area_code code ∧ 13 ∣ product_of_digits code := by
  sorry

end NUMINAMATH_CALUDE_no_valid_area_codes_l996_99623


namespace NUMINAMATH_CALUDE_almonds_in_trail_mix_l996_99649

/-- Given the amount of walnuts and the total amount of nuts in a trail mix,
    calculate the amount of almonds added. -/
theorem almonds_in_trail_mix (walnuts total : ℚ) (h1 : walnuts = 0.25) (h2 : total = 0.5) :
  total - walnuts = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_almonds_in_trail_mix_l996_99649


namespace NUMINAMATH_CALUDE_total_students_theorem_l996_99695

/-- Calculates the total number of students at the end of the year --/
def total_students_end_year (middle_initial : ℕ) : ℕ :=
  let elementary_initial := 4 * middle_initial - 3
  let high_initial := 2 * elementary_initial
  let elementary_end := (elementary_initial * 110 + 50) / 100
  let middle_end := (middle_initial * 95 + 50) / 100
  let high_end := (high_initial * 107 + 50) / 100
  elementary_end + middle_end + high_end

/-- Theorem stating that the total number of students at the end of the year is 687 --/
theorem total_students_theorem : total_students_end_year 50 = 687 := by
  sorry

end NUMINAMATH_CALUDE_total_students_theorem_l996_99695


namespace NUMINAMATH_CALUDE_complex_sum_powers_l996_99660

theorem complex_sum_powers (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 + i^4 + i^5 = i := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l996_99660


namespace NUMINAMATH_CALUDE_nine_special_integers_l996_99662

theorem nine_special_integers (m n : ℕ) (hm : m ≥ 16) (hn : n ≥ 24) :
  ∃ (a : Fin 9 → ℕ),
    (∀ k : Fin 9, a k = 2^(m + k.val) * 3^(n - k.val)) ∧
    (∀ k : Fin 9, 6 ∣ a k) ∧
    (∀ i j : Fin 9, i ≠ j → ¬(a i ∣ a j)) ∧
    (∀ i j : Fin 9, (a i)^3 ∣ (a j)^2) := by
  sorry

end NUMINAMATH_CALUDE_nine_special_integers_l996_99662


namespace NUMINAMATH_CALUDE_hallie_tuesday_hours_l996_99637

/-- Calculates the number of hours Hallie worked on Tuesday given her earnings and tips -/
def hours_worked_tuesday (hourly_rate : ℚ) (monday_hours : ℚ) (monday_tips : ℚ) 
  (tuesday_tips : ℚ) (wednesday_hours : ℚ) (wednesday_tips : ℚ) (total_earnings : ℚ) : ℚ :=
  let monday_earnings := hourly_rate * monday_hours + monday_tips
  let wednesday_earnings := hourly_rate * wednesday_hours + wednesday_tips
  let tuesday_earnings := total_earnings - monday_earnings - wednesday_earnings
  let tuesday_wage_earnings := tuesday_earnings - tuesday_tips
  tuesday_wage_earnings / hourly_rate

theorem hallie_tuesday_hours :
  hours_worked_tuesday 10 7 18 12 7 20 240 = 5 := by
  sorry

end NUMINAMATH_CALUDE_hallie_tuesday_hours_l996_99637


namespace NUMINAMATH_CALUDE_a_4_times_a_3_l996_99652

def a : ℕ → ℤ
  | n => if n % 2 = 1 then (-2)^n else n

theorem a_4_times_a_3 : a 4 * a 3 = -32 := by
  sorry

end NUMINAMATH_CALUDE_a_4_times_a_3_l996_99652


namespace NUMINAMATH_CALUDE_cabin_rental_duration_l996_99615

/-- Proves that the number of days for which the cabin is rented is 14, given the specified conditions. -/
theorem cabin_rental_duration :
  let daily_rate : ℚ := 125
  let pet_fee : ℚ := 100
  let service_fee_rate : ℚ := 0.2
  let security_deposit_rate : ℚ := 0.5
  let security_deposit : ℚ := 1110
  ∃ (days : ℕ), 
    security_deposit = security_deposit_rate * (daily_rate * days + pet_fee + service_fee_rate * (daily_rate * days + pet_fee)) ∧
    days = 14 := by
  sorry

end NUMINAMATH_CALUDE_cabin_rental_duration_l996_99615


namespace NUMINAMATH_CALUDE_perfect_square_implies_congruence_l996_99674

theorem perfect_square_implies_congruence (p a : ℕ) (h_prime : Nat.Prime p) :
  (∃ t : ℤ, ∃ k : ℤ, p * t + a = k^2) →
  a^((p - 1) / 2) ≡ 1 [ZMOD p] :=
sorry

end NUMINAMATH_CALUDE_perfect_square_implies_congruence_l996_99674


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l996_99608

/-- Represents the number of triangles of each color in each half of the figure -/
structure TriangleCounts where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs when the figure is folded -/
structure CoincidingPairs where
  red : ℕ
  blue : ℕ
  redWhite : ℕ
  white : ℕ

/-- The main theorem stating that given the conditions, 6 white pairs coincide -/
theorem white_pairs_coincide (counts : TriangleCounts) (pairs : CoincidingPairs) : 
  counts.red = 4 ∧ 
  counts.blue = 6 ∧ 
  counts.white = 10 ∧
  pairs.red = 2 ∧
  pairs.blue = 4 ∧
  pairs.redWhite = 3 →
  pairs.white = 6 := by
  sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l996_99608


namespace NUMINAMATH_CALUDE_find_number_l996_99656

theorem find_number (x : ℚ) : (55 + x / 78) * 78 = 4403 ↔ x = 1 := by sorry

end NUMINAMATH_CALUDE_find_number_l996_99656


namespace NUMINAMATH_CALUDE_line_x_intercept_m_values_l996_99618

theorem line_x_intercept_m_values (m : ℝ) : 
  (∃ y : ℝ, (2 * m^2 - m + 3) * 1 + (m^2 + 2*m) * y = 4*m + 1) → 
  (m = 2 ∨ m = 1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_line_x_intercept_m_values_l996_99618


namespace NUMINAMATH_CALUDE_fraction_simplification_l996_99691

theorem fraction_simplification (N : ℕ) :
  (Nat.factorial (N - 2) * (N - 1) * N) / Nat.factorial (N + 2) = 1 / ((N + 1) * (N + 2)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l996_99691


namespace NUMINAMATH_CALUDE_student_count_incorrect_l996_99688

theorem student_count_incorrect : ¬ ∃ k : ℕ, 18 + 17 * k = 2012 := by
  sorry

end NUMINAMATH_CALUDE_student_count_incorrect_l996_99688


namespace NUMINAMATH_CALUDE_min_sum_reciprocals_l996_99636

theorem min_sum_reciprocals (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : m + n = 1) :
  1/m + 1/n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_reciprocals_l996_99636


namespace NUMINAMATH_CALUDE_triangular_field_yield_l996_99686

/-- Proves that a triangular field with given dimensions and harvest yields 1 ton per hectare -/
theorem triangular_field_yield (base : ℝ) (height_factor : ℝ) (total_harvest : ℝ) :
  base = 200 →
  height_factor = 1.2 →
  total_harvest = 2.4 →
  let height := height_factor * base
  let area_sq_meters := (1 / 2) * base * height
  let area_hectares := area_sq_meters / 10000
  total_harvest / area_hectares = 1 := by sorry

end NUMINAMATH_CALUDE_triangular_field_yield_l996_99686


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l996_99611

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y, x ∈ Set.Ioo (-1 : ℝ) 1 → y ∈ Set.Ioo (-1 : ℝ) 1 → x < y → f x > f y := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l996_99611


namespace NUMINAMATH_CALUDE_two_in_M_l996_99679

def U : Set Nat := {1, 2, 3, 4, 5}

theorem two_in_M (M : Set Nat) (h : (U \ M) = {1, 3}) : 2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_two_in_M_l996_99679


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l996_99661

/-- Given a circle C with equation x^2 + 8x - 5y = -y^2 + 2x, 
    the sum of the x-coordinate and y-coordinate of its center along with its radius 
    is equal to (√61 - 1) / 2 -/
theorem circle_center_radius_sum (x y : ℝ) : 
  (x^2 + 8*x - 5*y = -y^2 + 2*x) → 
  ∃ (center_x center_y radius : ℝ), 
    (center_x + center_y + radius = (Real.sqrt 61 - 1) / 2) ∧
    ∀ (p_x p_y : ℝ), (p_x - center_x)^2 + (p_y - center_y)^2 = radius^2 ↔ 
      p_x^2 + 8*p_x - 5*p_y = -p_y^2 + 2*p_x :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l996_99661


namespace NUMINAMATH_CALUDE_integer_sum_of_fractions_l996_99625

theorem integer_sum_of_fractions (x y z : ℤ) (n : ℕ) 
  (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
  ∃ k : ℤ, x^n / ((x-y)*(x-z)) + y^n / ((y-x)*(y-z)) + z^n / ((z-x)*(z-y)) = k := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_of_fractions_l996_99625


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l996_99671

theorem fraction_sum_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b) := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l996_99671


namespace NUMINAMATH_CALUDE_triangle_perimeter_l996_99665

theorem triangle_perimeter (a b c : ℕ) (α β γ : ℝ) : 
  a > 0 ∧ b = a + 1 ∧ c = b + 1 →  -- Consecutive positive integer sides
  α > 0 ∧ β > 0 ∧ γ > 0 →  -- Positive angles
  α + β + γ = π →  -- Sum of angles in a triangle
  max γ (max α β) = 2 * min γ (min α β) →  -- Largest angle is twice the smallest
  a + b + c = 15 :=  -- Perimeter is 15
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l996_99665


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l996_99613

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- An arithmetic sequence -/
def is_arithmetic (a : Sequence) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property that a_{n+1}^2 = a_n * a_{n+2} for all n -/
def has_square_middle_property (a : Sequence) : Prop :=
  ∀ n : ℕ, (a (n + 1))^2 = a n * a (n + 2)

theorem arithmetic_sequence_property :
  (∀ a : Sequence, is_arithmetic a → has_square_middle_property a) ∧
  (∃ a : Sequence, has_square_middle_property a ∧ ¬is_arithmetic a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l996_99613


namespace NUMINAMATH_CALUDE_tamara_garden_walkway_area_l996_99610

/-- Represents the dimensions of a flower bed -/
structure FlowerBed where
  length : ℝ
  width : ℝ

/-- Represents the garden layout -/
structure Garden where
  rows : ℕ
  columns : ℕ
  bed : FlowerBed
  walkwayWidth : ℝ

/-- Calculates the total area of walkways in the garden -/
def walkwayArea (g : Garden) : ℝ :=
  let totalWidth := g.columns * g.bed.length + (g.columns + 1) * g.walkwayWidth
  let totalHeight := g.rows * g.bed.width + (g.rows + 1) * g.walkwayWidth
  let totalArea := totalWidth * totalHeight
  let bedArea := g.rows * g.columns * g.bed.length * g.bed.width
  totalArea - bedArea

/-- Theorem stating that the walkway area for Tamara's garden is 214 square feet -/
theorem tamara_garden_walkway_area :
  let g : Garden := {
    rows := 3,
    columns := 2,
    bed := { length := 7, width := 3 },
    walkwayWidth := 2
  }
  walkwayArea g = 214 := by
  sorry

end NUMINAMATH_CALUDE_tamara_garden_walkway_area_l996_99610


namespace NUMINAMATH_CALUDE_duck_count_l996_99694

theorem duck_count (total_legs : ℕ) (rabbit_count : ℕ) (rabbit_legs : ℕ) (duck_legs : ℕ) :
  total_legs = 48 →
  rabbit_count = 9 →
  rabbit_legs = 4 →
  duck_legs = 2 →
  (total_legs - rabbit_count * rabbit_legs) / duck_legs = 6 :=
by sorry

end NUMINAMATH_CALUDE_duck_count_l996_99694


namespace NUMINAMATH_CALUDE_flower_bouquet_row_length_l996_99648

theorem flower_bouquet_row_length 
  (num_students : ℕ) 
  (student_space : ℝ) 
  (gap_space : ℝ) 
  (h1 : num_students = 50) 
  (h2 : student_space = 0.4) 
  (h3 : gap_space = 0.5) : 
  num_students * student_space + (num_students - 1) * gap_space = 44.5 := by
  sorry

end NUMINAMATH_CALUDE_flower_bouquet_row_length_l996_99648


namespace NUMINAMATH_CALUDE_polynomial_subtraction_simplification_l996_99601

theorem polynomial_subtraction_simplification :
  ∀ x : ℝ, (2 * x^6 + x^5 + 3 * x^4 + x^2 + 15) - (x^6 + 2 * x^5 - x^4 + x^3 + 17) = 
            x^6 - x^5 + 4 * x^4 - x^3 + x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_subtraction_simplification_l996_99601


namespace NUMINAMATH_CALUDE_machine_work_time_l996_99693

/-- Proves that a machine making 6 shirts per minute worked for 23 minutes yesterday,
    given it made 14 shirts today and 156 shirts in total over two days. -/
theorem machine_work_time (shirts_per_minute : ℕ) (shirts_today : ℕ) (total_shirts : ℕ) :
  shirts_per_minute = 6 →
  shirts_today = 14 →
  total_shirts = 156 →
  (total_shirts - shirts_today) / shirts_per_minute = 23 :=
by sorry

end NUMINAMATH_CALUDE_machine_work_time_l996_99693


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_120_over_8_l996_99687

theorem largest_whole_number_less_than_120_over_8 :
  ∃ (x : ℕ), x = 14 ∧ (∀ y : ℕ, 8 * y < 120 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_120_over_8_l996_99687


namespace NUMINAMATH_CALUDE_square_preserves_geometric_sequence_sqrt_abs_preserves_geometric_sequence_l996_99653

-- Define the domain for the functions
def Domain : Set ℝ := {x : ℝ | x < 0 ∨ x > 0}

-- Define the property of being a geometric sequence
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the property of being a geometric sequence preserving function
def IsGeometricSequencePreserving (f : ℝ → ℝ) : Prop :=
  ∀ a : ℕ → ℝ, (∀ n : ℕ, a n ∈ Domain) →
    IsGeometricSequence a → IsGeometricSequence (f ∘ a)

-- State the theorem for f(x) = x^2
theorem square_preserves_geometric_sequence :
  IsGeometricSequencePreserving (fun x ↦ x^2) :=
sorry

-- State the theorem for f(x) = √|x|
theorem sqrt_abs_preserves_geometric_sequence :
  IsGeometricSequencePreserving (fun x ↦ Real.sqrt (abs x)) :=
sorry

end NUMINAMATH_CALUDE_square_preserves_geometric_sequence_sqrt_abs_preserves_geometric_sequence_l996_99653


namespace NUMINAMATH_CALUDE_fraction_equality_l996_99633

theorem fraction_equality : (8 : ℚ) / (5 * 46) = 0.8 / 23 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l996_99633


namespace NUMINAMATH_CALUDE_six_balls_three_boxes_l996_99646

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := sorry

/-- The theorem stating that there are 132 ways to distribute 6 distinguishable balls into 3 indistinguishable boxes -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 132 := by sorry

end NUMINAMATH_CALUDE_six_balls_three_boxes_l996_99646


namespace NUMINAMATH_CALUDE_a_plus_b_equals_neg_nine_l996_99642

def f (a b x : ℝ) : ℝ := a * x - b

def g (x : ℝ) : ℝ := -4 * x - 1

def h (a b x : ℝ) : ℝ := f a b (g x)

def h_inv (x : ℝ) : ℝ := x + 9

theorem a_plus_b_equals_neg_nine (a b : ℝ) :
  (∀ x, h a b x = h_inv⁻¹ x) → a + b = -9 := by
  sorry

end NUMINAMATH_CALUDE_a_plus_b_equals_neg_nine_l996_99642


namespace NUMINAMATH_CALUDE_roe_savings_aug_to_nov_l996_99684

def savings_jan_to_jul : ℕ := 10 * 7
def savings_dec : ℕ := 20
def total_savings : ℕ := 150
def months_aug_to_nov : ℕ := 4

theorem roe_savings_aug_to_nov :
  (total_savings - savings_jan_to_jul - savings_dec) / months_aug_to_nov = 15 := by
  sorry

end NUMINAMATH_CALUDE_roe_savings_aug_to_nov_l996_99684


namespace NUMINAMATH_CALUDE_arccos_cos_ten_l996_99629

theorem arccos_cos_ten :
  Real.arccos (Real.cos 10) = 10 - 4 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_ten_l996_99629
