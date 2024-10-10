import Mathlib

namespace greatest_common_multiple_15_20_under_125_l930_93054

theorem greatest_common_multiple_15_20_under_125 : 
  ∃ n : ℕ, n = 120 ∧ 
  (∀ m : ℕ, m < 125 ∧ 15 ∣ m ∧ 20 ∣ m → m ≤ n) ∧
  15 ∣ n ∧ 20 ∣ n ∧ n < 125 :=
by sorry

end greatest_common_multiple_15_20_under_125_l930_93054


namespace natural_number_equality_integer_absolute_equality_l930_93059

-- Part a
theorem natural_number_equality (x y n : ℕ) 
  (h : x^3 + 2^n * y = y^3 + 2^n * x) : x = y := by sorry

-- Part b
theorem integer_absolute_equality (x y : ℤ) (n : ℕ) 
  (hx : x ≠ 0) (hy : y ≠ 0)
  (h : x^3 + 2^n * y = y^3 + 2^n * x) : |x| = |y| := by sorry

end natural_number_equality_integer_absolute_equality_l930_93059


namespace quadratic_solution_property_l930_93083

theorem quadratic_solution_property (k : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 + 5 * x + k = 0 ∧ 3 * y^2 + 5 * y + k = 0 ∧ 
   |x + y| = x^2 + y^2) ↔ k = -10/3 := by
sorry

end quadratic_solution_property_l930_93083


namespace shopkeeper_bananas_l930_93071

theorem shopkeeper_bananas (oranges : ℕ) (bananas : ℕ) : 
  oranges = 600 →
  (oranges * 85 / 100 + bananas * 97 / 100 : ℚ) = (oranges + bananas) * 898 / 1000 →
  bananas = 400 := by
sorry

end shopkeeper_bananas_l930_93071


namespace congruence_solutions_count_l930_93096

theorem congruence_solutions_count : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ x < 200 ∧ (x + 17) % 52 = 75 % 52) 
    (Finset.range 200)).card = 4 := by
  sorry

end congruence_solutions_count_l930_93096


namespace card_area_problem_l930_93085

theorem card_area_problem (length width : ℝ) : 
  length = 5 ∧ width = 7 →
  (∃ (shortened_side : ℝ), (shortened_side = length - 2 ∨ shortened_side = width - 2) ∧
                           shortened_side * (if shortened_side = length - 2 then width else length) = 21) →
  (if length - 2 * width = 21 then (length * (width - 2) = 25) else ((length - 2) * width = 25)) :=
by sorry

end card_area_problem_l930_93085


namespace ellipse_with_foci_on_y_axis_l930_93095

theorem ellipse_with_foci_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  ∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1 :=
sorry

end ellipse_with_foci_on_y_axis_l930_93095


namespace sufficient_not_necessary_l930_93084

theorem sufficient_not_necessary : 
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧ 
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) := by
  sorry

end sufficient_not_necessary_l930_93084


namespace right_triangle_area_l930_93086

theorem right_triangle_area (leg1 leg2 : ℝ) (h1 : leg1 = 45) (h2 : leg2 = 48) :
  (1 / 2 : ℝ) * leg1 * leg2 = 1080 := by
  sorry

end right_triangle_area_l930_93086


namespace units_digit_fourth_power_not_seven_l930_93007

theorem units_digit_fourth_power_not_seven :
  ∀ n : ℕ, (n^4 % 10) ≠ 7 := by
  sorry

end units_digit_fourth_power_not_seven_l930_93007


namespace church_attendance_l930_93023

/-- Proves the number of female adults in a church given the number of children, male adults, and total people. -/
theorem church_attendance (children : ℕ) (male_adults : ℕ) (total_people : ℕ) 
  (h1 : children = 80)
  (h2 : male_adults = 60)
  (h3 : total_people = 200) :
  total_people - (children + male_adults) = 60 := by
  sorry

#check church_attendance

end church_attendance_l930_93023


namespace right_prism_surface_area_l930_93006

/-- A right prism with an isosceles trapezoid base -/
structure RightPrism where
  /-- Length of parallel sides AB and CD -/
  ab_cd : ℝ
  /-- Length of side BC -/
  bc : ℝ
  /-- Length of side AD -/
  ad : ℝ
  /-- Area of the diagonal cross-section -/
  diagonal_area : ℝ
  /-- Condition: AB and CD are equal -/
  ab_eq_cd : ab_cd > 0
  /-- Condition: BC is positive -/
  bc_pos : bc > 0
  /-- Condition: AD is positive -/
  ad_pos : ad > 0
  /-- Condition: AD > BC (trapezoid property) -/
  ad_gt_bc : ad > bc
  /-- Condition: Diagonal area is positive -/
  diagonal_area_pos : diagonal_area > 0

/-- Total surface area of the right prism -/
def totalSurfaceArea (p : RightPrism) : ℝ :=
  sorry

/-- Theorem: The total surface area of the specified right prism is 906 -/
theorem right_prism_surface_area :
  ∀ (p : RightPrism),
    p.ab_cd = 13 ∧ p.bc = 11 ∧ p.ad = 21 ∧ p.diagonal_area = 180 →
    totalSurfaceArea p = 906 := by
  sorry

end right_prism_surface_area_l930_93006


namespace min_value_reciprocal_sum_l930_93008

theorem min_value_reciprocal_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m + n = 1) :
  (1 / m + 1 / n) ≥ 4 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ m + n = 1 ∧ 1 / m + 1 / n = 4 :=
by sorry

end min_value_reciprocal_sum_l930_93008


namespace trig_identity_quadratic_solution_l930_93036

-- Part 1
theorem trig_identity : 
  Real.tan (π / 6) ^ 2 + 2 * Real.sin (π / 4) - 2 * Real.cos (π / 3) = (3 * Real.sqrt 2 - 2) / 3 := by
  sorry

-- Part 2
theorem quadratic_solution :
  let x₁ := (-2 + Real.sqrt 2) / 2
  let x₂ := (-2 - Real.sqrt 2) / 2
  2 * x₁ ^ 2 + 4 * x₁ + 1 = 0 ∧ 2 * x₂ ^ 2 + 4 * x₂ + 1 = 0 := by
  sorry

end trig_identity_quadratic_solution_l930_93036


namespace min_distance_to_line_l930_93004

/-- The minimum distance from the origin to a point on the line 3x + 4y - 20 = 0 is 4 -/
theorem min_distance_to_line : 
  ∀ a b : ℝ, (3 * a + 4 * b = 20) → (∀ x y : ℝ, (3 * x + 4 * y = 20) → (a^2 + b^2 ≤ x^2 + y^2)) → 
  Real.sqrt (a^2 + b^2) = 4 := by
sorry

end min_distance_to_line_l930_93004


namespace distance_first_to_last_l930_93072

-- Define the number of trees
def num_trees : ℕ := 8

-- Define the distance between first and fifth tree
def distance_1_to_5 : ℝ := 80

-- Theorem to prove
theorem distance_first_to_last :
  let distance_between_trees := distance_1_to_5 / 4
  let num_spaces := num_trees - 1
  distance_between_trees * num_spaces = 140 := by
sorry

end distance_first_to_last_l930_93072


namespace smallest_class_size_is_42_l930_93031

/-- Represents the number of students in a physical education class. -/
def ClassSize (n : ℕ) : ℕ := 5 * n + 2

/-- The smallest class size satisfying the given conditions -/
def SmallestClassSize : ℕ := 42

theorem smallest_class_size_is_42 :
  (∀ m : ℕ, ClassSize m > 40 → m ≥ SmallestClassSize) ∧
  (ClassSize (SmallestClassSize - 1) ≤ 40) ∧
  (ClassSize SmallestClassSize > 40) :=
sorry

end smallest_class_size_is_42_l930_93031


namespace shaded_area_is_three_point_five_l930_93066

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  L : ℝ × ℝ
  M : ℝ × ℝ
  N : ℝ × ℝ
  O : ℝ × ℝ
  Q : ℝ × ℝ
  P : ℝ × ℝ
  h_dimensions : M.1 - L.1 = 4 ∧ O.2 - M.2 = 5
  h_equal_segments : 
    (M.1 - L.1 = 1) ∧ 
    (Q.2 - M.2 = 1) ∧ 
    (P.1 - Q.1 = 1) ∧ 
    (O.2 - P.2 = 1)

/-- The area of the shaded region in the special rectangle -/
def shadedArea (r : SpecialRectangle) : ℝ := sorry

/-- Theorem stating that the shaded area is 3.5 -/
theorem shaded_area_is_three_point_five (r : SpecialRectangle) : 
  shadedArea r = 3.5 := by sorry

end shaded_area_is_three_point_five_l930_93066


namespace bookstore_problem_l930_93079

/-- Represents the number of magazine types at each price point -/
structure MagazineTypes :=
  (twoYuan : ℕ)
  (oneYuan : ℕ)

/-- Represents the total budget and purchasing constraints -/
structure PurchaseConstraints :=
  (budget : ℕ)
  (maxPerType : ℕ)

/-- Calculates the number of different purchasing methods -/
def purchasingMethods (types : MagazineTypes) (constraints : PurchaseConstraints) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem bookstore_problem :
  let types := MagazineTypes.mk 8 3
  let constraints := PurchaseConstraints.mk 10 1
  purchasingMethods types constraints = 266 := by
  sorry

end bookstore_problem_l930_93079


namespace chad_dog_food_packages_l930_93057

/-- Given Chad's purchase of cat and dog food, prove he bought 2 packages of dog food -/
theorem chad_dog_food_packages : 
  ∀ (cat_packages dog_packages : ℕ),
  cat_packages = 6 →
  ∀ (cat_cans_per_package dog_cans_per_package : ℕ),
  cat_cans_per_package = 9 →
  dog_cans_per_package = 3 →
  cat_packages * cat_cans_per_package = dog_packages * dog_cans_per_package + 48 →
  dog_packages = 2 :=
by sorry

end chad_dog_food_packages_l930_93057


namespace percent_of_y_l930_93000

theorem percent_of_y (y : ℝ) (h : y > 0) : ((1 / y) / 20 + (3 / y) / 10) / y = 35 / 100 := by
  sorry

end percent_of_y_l930_93000


namespace true_discount_proof_l930_93082

/-- Calculates the true discount given the banker's discount and sum due -/
def true_discount (bankers_discount : ℚ) (sum_due : ℚ) : ℚ :=
  let a : ℚ := 1
  let b : ℚ := sum_due
  let c : ℚ := -sum_due * bankers_discount
  (-b + (b^2 - 4*a*c).sqrt) / (2*a)

/-- Proves that the true discount is 246 given the banker's discount of 288 and sum due of 1440 -/
theorem true_discount_proof (bankers_discount sum_due : ℚ) 
  (h1 : bankers_discount = 288)
  (h2 : sum_due = 1440) : 
  true_discount bankers_discount sum_due = 246 := by
  sorry

#eval true_discount 288 1440

end true_discount_proof_l930_93082


namespace hyperbola_max_ratio_hyperbola_max_ratio_achievable_l930_93016

theorem hyperbola_max_ratio (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_focal : c^2 = a^2 + b^2) : 
  (a + b) / c ≤ Real.sqrt 2 :=
sorry

theorem hyperbola_max_ratio_achievable :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ c^2 = a^2 + b^2 ∧ (a + b) / c = Real.sqrt 2 :=
sorry

end hyperbola_max_ratio_hyperbola_max_ratio_achievable_l930_93016


namespace rosy_fish_count_l930_93025

/-- The number of fish Lilly has -/
def lillys_fish : ℕ := 10

/-- The total number of fish Lilly and Rosy have together -/
def total_fish : ℕ := 18

/-- The number of fish Rosy has -/
def rosys_fish : ℕ := total_fish - lillys_fish

theorem rosy_fish_count : rosys_fish = 8 := by sorry

end rosy_fish_count_l930_93025


namespace sams_eatery_meal_cost_l930_93065

/-- Calculates the cost of a meal at Sam's Eatery with a discount --/
def meal_cost (hamburger_price : ℚ) (fries_price : ℚ) (drink_price : ℚ) 
              (num_hamburgers : ℕ) (num_fries : ℕ) (num_drinks : ℕ) 
              (discount_percent : ℚ) : ℕ :=
  let total_before_discount := hamburger_price * num_hamburgers + 
                               fries_price * num_fries + 
                               drink_price * num_drinks
  let discount_amount := total_before_discount * (discount_percent / 100)
  let total_after_discount := total_before_discount - discount_amount
  (total_after_discount + 1/2).floor.toNat

/-- The cost of the meal at Sam's Eatery is 35 dollars --/
theorem sams_eatery_meal_cost : 
  meal_cost 5 3 2 3 4 6 10 = 35 := by
  sorry


end sams_eatery_meal_cost_l930_93065


namespace replacement_solution_percentage_l930_93058

theorem replacement_solution_percentage
  (original_percentage : ℝ)
  (replaced_portion : ℝ)
  (final_percentage : ℝ)
  (h1 : original_percentage = 85)
  (h2 : replaced_portion = 0.6923076923076923)
  (h3 : final_percentage = 40)
  (x : ℝ) :
  (original_percentage * (1 - replaced_portion) + x * replaced_portion = final_percentage) →
  x = 20 := by
  sorry

end replacement_solution_percentage_l930_93058


namespace max_quarters_proof_l930_93018

/-- Represents the number of coins of each type --/
structure CoinCount where
  quarters : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total value of coins in cents --/
def totalValue (coins : CoinCount) : ℕ :=
  coins.quarters * 25 + coins.nickels * 5 + coins.dimes * 10

/-- Checks if the coin count satisfies the problem conditions --/
def isValidCount (coins : CoinCount) : Prop :=
  coins.quarters = coins.nickels ∧ coins.dimes * 2 = coins.quarters

/-- The maximum number of quarters possible given the conditions --/
def maxQuarters : ℕ := 11

theorem max_quarters_proof :
  ∀ coins : CoinCount,
    isValidCount coins →
    totalValue coins = 400 →
    coins.quarters ≤ maxQuarters :=
by sorry

end max_quarters_proof_l930_93018


namespace cloth_woven_approx_15_meters_l930_93081

/-- The rate at which the loom weaves cloth in meters per second -/
def weaving_rate : ℝ := 0.127

/-- The time taken by the loom to weave the cloth in seconds -/
def weaving_time : ℝ := 118.11

/-- The amount of cloth woven in meters -/
def cloth_woven : ℝ := weaving_rate * weaving_time

/-- Theorem stating that the amount of cloth woven is approximately 15 meters -/
theorem cloth_woven_approx_15_meters : 
  ∃ ε > 0, |cloth_woven - 15| < ε := by sorry

end cloth_woven_approx_15_meters_l930_93081


namespace hundredthOddPositiveInteger_l930_93001

/-- The nth odd positive integer -/
def nthOddPositiveInteger (n : ℕ) : ℕ := 2 * n - 1

/-- Theorem: The 100th odd positive integer is 199 -/
theorem hundredthOddPositiveInteger : nthOddPositiveInteger 100 = 199 := by
  sorry

end hundredthOddPositiveInteger_l930_93001


namespace exp_convex_and_ln_concave_l930_93063

-- Define the exponential function
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Define the natural logarithm function
noncomputable def g (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem exp_convex_and_ln_concave :
  (∀ x y : ℝ, ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y) ∧
  (∀ x y : ℝ, x > 0 ∧ y > 0 → ∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    g (t * x + (1 - t) * y) ≥ t * g x + (1 - t) * g y) :=
by sorry

end exp_convex_and_ln_concave_l930_93063


namespace four_digit_permutations_2033_eq_18_l930_93040

/-- The number of unique four-digit permutations of the digits in 2033 -/
def four_digit_permutations_2033 : ℕ := 18

/-- The set of digits in 2033 -/
def digits_2033 : Finset ℕ := {0, 2, 3}

/-- The function to count valid permutations -/
def count_valid_permutations (digits : Finset ℕ) : ℕ :=
  sorry

theorem four_digit_permutations_2033_eq_18 :
  count_valid_permutations digits_2033 = four_digit_permutations_2033 :=
by sorry

end four_digit_permutations_2033_eq_18_l930_93040


namespace tan_neg_alpha_implies_expression_l930_93047

theorem tan_neg_alpha_implies_expression (α : Real) 
  (h : Real.tan (-α) = 3) : 
  (Real.sin α)^2 - Real.sin (2 * α) = (-15/8) * Real.cos (2 * α) := by
  sorry

end tan_neg_alpha_implies_expression_l930_93047


namespace f_properties_l930_93053

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.cos x + Real.sqrt 3 * Real.sin x)

theorem f_properties :
  ∃ (T : ℝ),
    (∀ x, f (x + T) = f x) ∧
    T = π ∧
    (∀ k : ℤ, StrictMonoOn f (Set.Ioo (↑k * π - π / 3) (↑k * π + π / 6))) ∧
    (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ 3) ∧
    (∃ x ∈ Set.Icc 0 (π / 2), f x = 3) := by
  sorry

end f_properties_l930_93053


namespace sqrt_of_four_is_plus_minus_two_l930_93062

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem sqrt_of_four_is_plus_minus_two : sqrt 4 = {2, -2} := by
  sorry

end sqrt_of_four_is_plus_minus_two_l930_93062


namespace closest_integer_to_cube_root_l930_93044

theorem closest_integer_to_cube_root : ∃ n : ℤ, 
  n = 10 ∧ ∀ m : ℤ, m ≠ n → |n - (7^3 + 9^3 + 3)^(1/3)| < |m - (7^3 + 9^3 + 3)^(1/3)| :=
by sorry

end closest_integer_to_cube_root_l930_93044


namespace angies_age_equation_l930_93028

theorem angies_age_equation (angie_age : ℕ) (result : ℕ) : 
  angie_age = 8 → result = 2 * angie_age + 4 → result = 20 := by
  sorry

end angies_age_equation_l930_93028


namespace world_grain_ratio_l930_93055

def world_grain_supply : ℝ := 1800000
def world_grain_demand : ℝ := 2400000

theorem world_grain_ratio : 
  world_grain_supply / world_grain_demand = 3 / 4 := by
  sorry

end world_grain_ratio_l930_93055


namespace cube_volume_from_diagonal_l930_93061

/-- Given a cube with space diagonal 5√3, prove its volume is 125 -/
theorem cube_volume_from_diagonal (s : ℝ) : 
  s * Real.sqrt 3 = 5 * Real.sqrt 3 → s^3 = 125 := by
  sorry

end cube_volume_from_diagonal_l930_93061


namespace harrys_family_age_ratio_l930_93070

/-- Given Harry's age, the age difference between Harry and his father, and his mother's age when she gave birth to him, 
    prove that the ratio of the age difference between Harry's parents to Harry's age is 1:25. -/
theorem harrys_family_age_ratio (harry_age : ℕ) (father_age_diff : ℕ) (mother_age_at_birth : ℕ)
  (h1 : harry_age = 50)
  (h2 : father_age_diff = 24)
  (h3 : mother_age_at_birth = 22) :
  (father_age_diff + harry_age - (mother_age_at_birth + harry_age)) / harry_age = 1 / 25 := by
  sorry

end harrys_family_age_ratio_l930_93070


namespace stickers_at_end_of_week_l930_93030

def initial_stickers : ℝ := 39.0
def given_away_stickers : ℝ := 22.0

theorem stickers_at_end_of_week : 
  initial_stickers - given_away_stickers = 17.0 := by
  sorry

end stickers_at_end_of_week_l930_93030


namespace tmobile_additional_line_cost_l930_93051

theorem tmobile_additional_line_cost 
  (tmobile_base : ℕ) 
  (mmobile_base : ℕ) 
  (mmobile_additional : ℕ) 
  (total_lines : ℕ) 
  (price_difference : ℕ) 
  (h1 : tmobile_base = 50)
  (h2 : mmobile_base = 45)
  (h3 : mmobile_additional = 14)
  (h4 : total_lines = 5)
  (h5 : price_difference = 11)
  (h6 : tmobile_base + (total_lines - 2) * x = 
        mmobile_base + (total_lines - 2) * mmobile_additional + price_difference) :
  x = 16 := by
  sorry

#check tmobile_additional_line_cost

end tmobile_additional_line_cost_l930_93051


namespace total_out_of_pocket_cost_l930_93019

/-- Calculates the total out-of-pocket cost for medical treatment --/
theorem total_out_of_pocket_cost 
  (doctor_visit_cost : ℕ) 
  (cast_cost : ℕ) 
  (initial_insurance_coverage : ℚ) 
  (pt_sessions : ℕ) 
  (pt_cost_per_session : ℕ) 
  (pt_insurance_coverage : ℚ) : 
  doctor_visit_cost = 300 →
  cast_cost = 200 →
  initial_insurance_coverage = 60 / 100 →
  pt_sessions = 8 →
  pt_cost_per_session = 100 →
  pt_insurance_coverage = 40 / 100 →
  (1 - initial_insurance_coverage) * (doctor_visit_cost + cast_cost) +
  (1 - pt_insurance_coverage) * (pt_sessions * pt_cost_per_session) = 680 := by
sorry

end total_out_of_pocket_cost_l930_93019


namespace mashas_measurements_impossible_l930_93068

/-- A pentagon inscribed in a circle with given interior angles -/
structure InscribedPentagon where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ

/-- The sum of interior angles of a pentagon is 540° -/
axiom pentagon_angle_sum (p : InscribedPentagon) :
  p.angle1 + p.angle2 + p.angle3 + p.angle4 + p.angle5 = 540

/-- Opposite angles in an inscribed quadrilateral sum to 180° -/
axiom inscribed_quadrilateral_opposite_angles (a b : ℝ) :
  a + b = 180 → ∃ (p : InscribedPentagon), p.angle1 = a ∧ p.angle3 = b

/-- Masha's measurements -/
def mashas_pentagon : InscribedPentagon := {
  angle1 := 80,
  angle2 := 90,
  angle3 := 100,
  angle4 := 130,
  angle5 := 140
}

/-- Theorem: Masha's measurements are impossible for a pentagon inscribed in a circle -/
theorem mashas_measurements_impossible : 
  ¬∃ (p : InscribedPentagon), p = mashas_pentagon :=
sorry

end mashas_measurements_impossible_l930_93068


namespace division_property_l930_93069

theorem division_property (n : ℕ) (hn : n > 0) :
  (5^(n-1) + 3^(n-1)) ∣ (5^n + 3^n) ↔ n = 1 := by
  sorry

end division_property_l930_93069


namespace right_triangle_hypotenuse_l930_93078

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 2 ∧ b = 3 ∧ c^2 = a^2 + b^2 → c = Real.sqrt 13 := by
  sorry

end right_triangle_hypotenuse_l930_93078


namespace first_player_wins_l930_93020

/-- Represents a rectangular grid --/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a player in the game --/
inductive Player
  | First
  | Second

/-- Represents the game state --/
structure GameState :=
  (grid : Grid)
  (currentPlayer : Player)
  (shadedCells : Set (ℕ × ℕ))

/-- Defines a valid move in the game --/
def ValidMove (state : GameState) (move : Set (ℕ × ℕ)) : Prop :=
  ∀ cell ∈ move,
    cell.1 ≤ state.grid.rows ∧
    cell.2 ≤ state.grid.cols ∧
    cell ∉ state.shadedCells

/-- Defines the winning condition --/
def IsWinningState (state : GameState) : Prop :=
  ∀ move : Set (ℕ × ℕ), ¬(ValidMove state move)

/-- Theorem: The first player has a winning strategy in a 19 × 94 grid game --/
theorem first_player_wins :
  ∃ (strategy : GameState → Set (ℕ × ℕ)),
    let initialState := GameState.mk (Grid.mk 19 94) Player.First ∅
    ∀ (game : ℕ → GameState),
      game 0 = initialState →
      (∀ n : ℕ,
        (game n).currentPlayer = Player.First →
        ValidMove (game n) (strategy (game n)) ∧
        (game (n + 1)).shadedCells = (game n).shadedCells ∪ (strategy (game n))) →
      (∀ n : ℕ,
        (game n).currentPlayer = Player.Second →
        ∃ move,
          ValidMove (game n) move ∧
          (game (n + 1)).shadedCells = (game n).shadedCells ∪ move) →
      ∃ m : ℕ, IsWinningState (game m) ∧ (game m).currentPlayer = Player.First :=
sorry

end first_player_wins_l930_93020


namespace quadratic_solution_l930_93087

/-- A quadratic function passing through (-3,0) and (4,0) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The equation we want to solve -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * (x - 1)^2 + c - (b - b * x)

theorem quadratic_solution (a b c : ℝ) (h1 : f a b c (-3) = 0) (h2 : f a b c 4 = 0) :
  (∀ x : ℝ, g a b c x = 0 ↔ x = -2 ∨ x = 5) :=
sorry

end quadratic_solution_l930_93087


namespace zeros_before_first_nonzero_is_correct_l930_93092

/-- The number of zeros after the decimal point and before the first non-zero digit
    in the terminating decimal representation of 1/(2^7 * 5^3) -/
def zeros_before_first_nonzero : ℕ :=
  4

/-- The fraction we're considering -/
def fraction : ℚ :=
  1 / (2^7 * 5^3)

/-- Theorem stating that the number of zeros before the first non-zero digit
    in the terminating decimal representation of our fraction is correct -/
theorem zeros_before_first_nonzero_is_correct :
  zeros_before_first_nonzero = 4 ∧
  ∃ (n : ℕ), fraction * 10^zeros_before_first_nonzero = n / 10^zeros_before_first_nonzero ∧
             n % 10 ≠ 0 :=
by sorry

end zeros_before_first_nonzero_is_correct_l930_93092


namespace probability_not_red_card_l930_93012

theorem probability_not_red_card (odds_red : ℚ) (h : odds_red = 5/7) :
  1 - odds_red / (1 + odds_red) = 7/12 := by sorry

end probability_not_red_card_l930_93012


namespace cone_volume_l930_93048

/-- Given a cone whose lateral surface is an arc of a sector with radius 2 and arc length 2π,
    prove that its volume is (√3 * π) / 3 -/
theorem cone_volume (r : Real) (h : Real) :
  (r = 1) →
  (h^2 + r^2 = 2^2) →
  (1/3 * π * r^2 * h = (Real.sqrt 3 * π) / 3) := by
  sorry

end cone_volume_l930_93048


namespace ratio_fraction_l930_93064

theorem ratio_fraction (x y : ℝ) (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 := by
  sorry

end ratio_fraction_l930_93064


namespace expression_simplification_l930_93013

theorem expression_simplification (d : ℝ) (h : d ≠ 0) :
  ∃ (a b c : ℤ),
    (15 : ℝ) * d + 16 + 17 * d^2 + (3 : ℝ) * d + 2 = (a : ℝ) * d + b + (c : ℝ) * d^2 ∧
    a + b + c = 53 := by
  sorry

end expression_simplification_l930_93013


namespace exists_infinite_set_satisfying_equation_l930_93075

/-- A function from positive integers to positive integers -/
def PositiveIntegerFunction : Type := ℕ+ → ℕ+

/-- The property that f(x) + f(x+2) ≤ 2f(x+1) for all x -/
def SatisfiesInequality (f : PositiveIntegerFunction) : Prop :=
  ∀ x : ℕ+, f x + f (x + 2) ≤ 2 * f (x + 1)

/-- The property that (i-j)f(k) + (j-k)f(i) + (k-i)f(j) = 0 for all i, j, k in a set M -/
def SatisfiesEquation (f : PositiveIntegerFunction) (M : Set ℕ+) : Prop :=
  ∀ i j k : ℕ+, i ∈ M → j ∈ M → k ∈ M →
    (i - j : ℤ) * (f k : ℤ) + (j - k : ℤ) * (f i : ℤ) + (k - i : ℤ) * (f j : ℤ) = 0

/-- The main theorem -/
theorem exists_infinite_set_satisfying_equation
  (f : PositiveIntegerFunction) (h : SatisfiesInequality f) :
  ∃ M : Set ℕ+, Set.Infinite M ∧ SatisfiesEquation f M := by
  sorry

end exists_infinite_set_satisfying_equation_l930_93075


namespace gravitational_force_in_orbit_l930_93017

/-- Gravitational force calculation -/
theorem gravitational_force_in_orbit 
  (surface_distance : ℝ) 
  (orbit_distance : ℝ) 
  (surface_force : ℝ) 
  (h1 : surface_distance = 6000)
  (h2 : orbit_distance = 36000)
  (h3 : surface_force = 800)
  (h4 : ∀ (d f : ℝ), f * d^2 = surface_force * surface_distance^2) :
  ∃ (orbit_force : ℝ), 
    orbit_force * orbit_distance^2 = surface_force * surface_distance^2 ∧ 
    orbit_force = 1 / 45 := by
  sorry

end gravitational_force_in_orbit_l930_93017


namespace store_buying_combinations_l930_93024

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of item choices for each student --/
def num_choices : ℕ := 2

/-- The total number of possible buying combinations --/
def total_combinations : ℕ := num_choices ^ num_students

/-- The number of valid buying combinations --/
def valid_combinations : ℕ := total_combinations - 1

theorem store_buying_combinations :
  valid_combinations = 15 := by sorry

end store_buying_combinations_l930_93024


namespace bisecting_circle_relation_l930_93034

/-- A circle that always bisects another circle -/
structure BisectingCircle where
  a : ℝ
  b : ℝ
  eq_bisecting : ∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = b^2 + 1
  eq_bisected : ∀ (x y : ℝ), (x + 1)^2 + (y + 1)^2 = 4
  bisects : ∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = b^2 + 1 → 
    ∃ (t : ℝ), (x + 1)^2 + (y + 1)^2 = 4 ∧ 
    ((1 - t) * x + t * (-1))^2 + ((1 - t) * y + t * (-1))^2 = 1

/-- The relationship between a and b in a bisecting circle -/
theorem bisecting_circle_relation (c : BisectingCircle) : 
  c.a^2 + 2*c.a + 2*c.b + 5 = 0 := by
  sorry

end bisecting_circle_relation_l930_93034


namespace parallel_lines_k_equals_negative_one_l930_93089

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.b ≠ 0

theorem parallel_lines_k_equals_negative_one :
  ∀ k : ℝ,
  let l1 : Line := ⟨k, -1, 1⟩
  let l2 : Line := ⟨1, -k, 1⟩
  parallel l1 l2 → k = -1 :=
by sorry

end parallel_lines_k_equals_negative_one_l930_93089


namespace b_alone_time_l930_93038

/-- The time (in days) it takes for A and B together to complete the work -/
def combined_time : ℚ := 12

/-- The time (in days) it takes for A alone to complete the work -/
def a_time : ℚ := 24

/-- The work rate of A (work per day) -/
def a_rate : ℚ := 1 / a_time

/-- The combined work rate of A and B (work per day) -/
def combined_rate : ℚ := 1 / combined_time

/-- The work rate of B (work per day) -/
def b_rate : ℚ := combined_rate - a_rate

/-- The time (in days) it takes for B alone to complete the work -/
def b_time : ℚ := 1 / b_rate

theorem b_alone_time : b_time = 24 := by
  sorry

end b_alone_time_l930_93038


namespace dennis_teaching_years_l930_93094

theorem dennis_teaching_years 
  (total_years : ℕ) 
  (virginia_adrienne_diff : ℕ) 
  (dennis_virginia_diff : ℕ) 
  (h1 : total_years = 102)
  (h2 : virginia_adrienne_diff = 9)
  (h3 : dennis_virginia_diff = 9)
  : ∃ (adrienne virginia dennis : ℕ),
    adrienne + virginia + dennis = total_years ∧
    virginia = adrienne + virginia_adrienne_diff ∧
    dennis = virginia + dennis_virginia_diff ∧
    dennis = 43 :=
by sorry

end dennis_teaching_years_l930_93094


namespace congruence_solution_l930_93099

theorem congruence_solution (n : ℤ) : 13 * 26 ≡ 8 [ZMOD 47] := by sorry

end congruence_solution_l930_93099


namespace sphere_cylinder_volume_l930_93014

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 7) (h_cylinder : r_cylinder = 4) :
  let v_sphere := (4 / 3) * π * r_sphere^3
  let h_cylinder := Real.sqrt (r_sphere^2 - r_cylinder^2)
  let v_cylinder := π * r_cylinder^2 * h_cylinder
  v_sphere - v_cylinder = ((1372 - 48 * Real.sqrt 33) / 3) * π := by
  sorry

end sphere_cylinder_volume_l930_93014


namespace josette_bought_three_bottles_l930_93037

/-- The number of bottles Josette bought for €1.50, given that 4 bottles cost €2 -/
def bottles_bought (cost_four_bottles : ℚ) (amount_spent : ℚ) : ℚ :=
  amount_spent / (cost_four_bottles / 4)

/-- Theorem stating that Josette bought 3 bottles -/
theorem josette_bought_three_bottles : 
  bottles_bought 2 (3/2) = 3 := by
  sorry

end josette_bought_three_bottles_l930_93037


namespace power_mod_seventeen_l930_93043

theorem power_mod_seventeen : 3^45 % 17 = 15 := by
  sorry

end power_mod_seventeen_l930_93043


namespace sum_of_reciprocals_squared_l930_93033

/-- Theorem: Sum of reciprocals squared --/
theorem sum_of_reciprocals_squared :
  let a := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15
  let b := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 15
  let c := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15
  let d := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 15
  (1/a + 1/b + 1/c + 1/d)^2 = 960/3481 := by
  sorry

end sum_of_reciprocals_squared_l930_93033


namespace marias_cupcakes_l930_93039

/-- 
Given that Maria made some cupcakes, sold 5, made 10 more, and ended up with 24 cupcakes,
this theorem proves that she initially made 19 cupcakes.
-/
theorem marias_cupcakes (x : ℕ) 
  (h : x - 5 + 10 = 24) : x = 19 := by
  sorry

end marias_cupcakes_l930_93039


namespace field_pond_area_ratio_l930_93091

/-- Given a rectangular field and a square pond, prove the ratio of their areas -/
theorem field_pond_area_ratio :
  ∀ (field_length field_width pond_side : ℝ),
  field_length = 2 * field_width →
  field_length = 32 →
  pond_side = 8 →
  (pond_side^2) / (field_length * field_width) = 1 / 8 := by
sorry

end field_pond_area_ratio_l930_93091


namespace basketball_points_difference_basketball_game_theorem_l930_93041

/-- The difference between the combined points of Tobee and Jay and Sean's points is 2 -/
theorem basketball_points_difference : ℕ → ℕ → ℕ → Prop :=
  fun tobee_points jay_points_diff total_team_points =>
    let jay_points := tobee_points + jay_points_diff
    let combined_points := tobee_points + jay_points
    let sean_points := total_team_points - combined_points
    combined_points - sean_points = 2

/-- Given the conditions of the basketball game -/
theorem basketball_game_theorem :
  basketball_points_difference 4 6 26 := by
  sorry

end basketball_points_difference_basketball_game_theorem_l930_93041


namespace root_sum_ratio_l930_93060

theorem root_sum_ratio (m₁ m₂ : ℝ) : 
  (∃ a b : ℝ, m₁ * (a^2 - 2*a) + 3*a + 7 = 0 ∧ 
              m₂ * (b^2 - 2*b) + 3*b + 7 = 0 ∧ 
              a/b + b/a = 9/10) →
  m₁/m₂ + m₂/m₁ = ((323/40)^2 * 4 - 18) / 9 :=
by sorry

end root_sum_ratio_l930_93060


namespace parabola_equation_l930_93042

/-- A parabola that opens downward with focus at (0, -2) -/
structure DownwardParabola where
  focus : ℝ × ℝ
  opens_downward : focus.2 < 0
  focus_y : focus.1 = 0 ∧ focus.2 = -2

/-- The hyperbola y²/3 - x² = 1 -/
def Hyperbola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 / 3 - p.1^2 = 1}

/-- The standard form of a downward-opening parabola -/
def ParabolaEquation (p : ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | q.1^2 = -2 * p * q.2}

theorem parabola_equation (C : DownwardParabola) 
    (h : C.focus ∈ Hyperbola) : 
    ParabolaEquation 4 = {q : ℝ × ℝ | q.1^2 = -8 * q.2} := by
  sorry

end parabola_equation_l930_93042


namespace package_weight_problem_l930_93003

theorem package_weight_problem (a b c : ℝ) 
  (hab : a + b = 108)
  (hbc : b + c = 132)
  (hca : c + a = 138) :
  a + b + c = 189 ∧ a ≥ 40 ∧ b ≥ 40 ∧ c ≥ 40 := by
  sorry

end package_weight_problem_l930_93003


namespace ab_value_l930_93077

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧
    Real.sqrt (2 * log a) = m ∧
    Real.sqrt (2 * log b) = n ∧
    log (Real.sqrt a) = (m^2 : ℝ) / 4 ∧
    log (Real.sqrt b) = (n^2 : ℝ) / 4 ∧
    m + n + (m^2 : ℝ) / 4 + (n^2 : ℝ) / 4 = 104) →
  a * b = 10^260 := by
sorry

end ab_value_l930_93077


namespace max_integer_value_of_function_l930_93011

theorem max_integer_value_of_function (x : ℝ) : 
  (4*x^2 + 8*x + 5 ≠ 0) → 
  ∃ (y : ℤ), y = 17 ∧ ∀ (z : ℤ), z ≤ (4*x^2 + 8*x + 21) / (4*x^2 + 8*x + 5) → z ≤ y :=
by sorry

end max_integer_value_of_function_l930_93011


namespace rope_cutting_problem_l930_93049

theorem rope_cutting_problem (rope1 rope2 rope3 rope4 : ℕ) 
  (h1 : rope1 = 30) (h2 : rope2 = 45) (h3 : rope3 = 60) (h4 : rope4 = 75) :
  Nat.gcd rope1 (Nat.gcd rope2 (Nat.gcd rope3 rope4)) = 15 :=
by
  sorry

end rope_cutting_problem_l930_93049


namespace water_tank_full_time_l930_93098

/-- Represents the state of a water tank system with three pipes -/
structure WaterTankSystem where
  capacity : ℕ
  pipeA_rate : ℕ
  pipeB_rate : ℕ
  pipeC_rate : ℕ

/-- Calculates the net water change after one cycle -/
def net_change_per_cycle (system : WaterTankSystem) : ℤ :=
  system.pipeA_rate + system.pipeB_rate - system.pipeC_rate

/-- Calculates the time needed to fill the tank -/
def time_to_fill (system : WaterTankSystem) : ℕ :=
  (system.capacity / (net_change_per_cycle system).natAbs) * 3

/-- Theorem stating that the given water tank system will be full after 48 minutes -/
theorem water_tank_full_time (system : WaterTankSystem) 
  (h1 : system.capacity = 800)
  (h2 : system.pipeA_rate = 40)
  (h3 : system.pipeB_rate = 30)
  (h4 : system.pipeC_rate = 20) : 
  time_to_fill system = 48 := by
  sorry

#eval time_to_fill { capacity := 800, pipeA_rate := 40, pipeB_rate := 30, pipeC_rate := 20 }

end water_tank_full_time_l930_93098


namespace floor_plus_x_eq_seventeen_fourths_l930_93046

theorem floor_plus_x_eq_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by
  sorry

end floor_plus_x_eq_seventeen_fourths_l930_93046


namespace forty_students_in_music_l930_93074

/-- Represents the number of students in various categories in a high school. -/
structure SchoolData where
  total : ℕ
  art : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the number of students taking music based on the given school data. -/
def studentsInMusic (data : SchoolData) : ℕ :=
  data.total - data.neither - (data.art - data.both)

/-- Theorem stating that given the specific school data, 40 students are taking music. -/
theorem forty_students_in_music :
  let data : SchoolData := {
    total := 500,
    art := 20,
    both := 10,
    neither := 450
  }
  studentsInMusic data = 40 := by
  sorry


end forty_students_in_music_l930_93074


namespace train_interval_l930_93015

/-- Represents a metro station -/
inductive Station : Type
| Taganskaya : Station
| Kievskaya : Station

/-- Represents a direction of travel -/
inductive Direction : Type
| Clockwise : Direction
| Counterclockwise : Direction

/-- Represents the metro system -/
structure MetroSystem where
  northern_route_time : ℝ
  southern_route_time : ℝ
  train_delay : ℝ
  trip_time_difference : ℝ

/-- Calculate the expected travel time between stations -/
def expected_travel_time (m : MetroSystem) (p : ℝ) : ℝ :=
  m.southern_route_time * p + m.northern_route_time * (1 - p)

/-- Theorem: The interval between trains in one direction is 3 minutes -/
theorem train_interval (m : MetroSystem) 
  (h1 : m.northern_route_time = 17)
  (h2 : m.southern_route_time = 11)
  (h3 : m.train_delay = 5/4)
  (h4 : m.trip_time_difference = 1)
  : ∃ (T : ℝ), T = 3 ∧ 
    ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧
    expected_travel_time m p = expected_travel_time m (1-p) - m.trip_time_difference ∧
    T * (1 - p) = m.train_delay := by
  sorry

end train_interval_l930_93015


namespace third_circle_radius_l930_93026

/-- Given two externally tangent circles and a third circle tangent to both and their common external tangent, prove the radius of the third circle is 5. -/
theorem third_circle_radius (P Q R : ℝ × ℝ) (r : ℝ) : 
  let d := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (d = 8) →  -- Distance between centers P and Q
  (∀ X : ℝ × ℝ, (X.1 - P.1)^2 + (X.2 - P.2)^2 = 3^2 → 
    (X.1 - Q.1)^2 + (X.2 - Q.2)^2 = 8^2) →  -- Circles are externally tangent
  (∀ Y : ℝ × ℝ, (Y.1 - P.1)^2 + (Y.2 - P.2)^2 = (3 + r)^2 ∧ 
    (Y.1 - Q.1)^2 + (Y.2 - Q.2)^2 = (5 - r)^2) →  -- Third circle is tangent to both circles
  (∃ Z : ℝ × ℝ, (Z.1 - R.1)^2 + (Z.2 - R.2)^2 = r^2 ∧ 
    ((Z.1 - P.1) * (Q.2 - P.2) = (Z.2 - P.2) * (Q.1 - P.1))) →  -- Third circle is tangent to common external tangent
  r = 5 := by
sorry

end third_circle_radius_l930_93026


namespace exterior_angle_measure_l930_93009

theorem exterior_angle_measure (n : ℕ) (h : n > 2) :
  (n - 2) * 180 = 1260 →
  360 / n = 40 := by
sorry

end exterior_angle_measure_l930_93009


namespace monkey_apple_problem_l930_93073

/-- Given a number of monkeys and apples, this function checks if they satisfy the conditions:
    1. If each monkey gets 3 apples, there will be 6 left.
    2. If each monkey gets 4 apples, the last monkey will get less than 4 apples. -/
def satisfies_conditions (monkeys : ℕ) (apples : ℕ) : Prop :=
  (apples = 3 * monkeys + 6) ∧ 
  (apples < 4 * monkeys) ∧ 
  (apples > 4 * (monkeys - 1))

/-- Theorem stating that the only solutions satisfying the conditions are
    (7 monkeys, 27 apples), (8 monkeys, 30 apples), or (9 monkeys, 33 apples) -/
theorem monkey_apple_problem :
  ∀ monkeys apples : ℕ, 
    satisfies_conditions monkeys apples ↔ 
    ((monkeys = 7 ∧ apples = 27) ∨ 
     (monkeys = 8 ∧ apples = 30) ∨ 
     (monkeys = 9 ∧ apples = 33)) :=
by sorry

end monkey_apple_problem_l930_93073


namespace two_numbers_sum_product_l930_93045

theorem two_numbers_sum_product (S P : ℝ) (h : S^2 ≥ 4*P) :
  ∃! (x y : ℝ × ℝ), (x.1 + x.2 = S ∧ x.1 * x.2 = P) ∧ (y.1 + y.2 = S ∧ y.1 * y.2 = P) ∧ x ≠ y :=
by
  sorry

end two_numbers_sum_product_l930_93045


namespace no_such_function_exists_l930_93032

theorem no_such_function_exists : 
  ∀ f : ℝ → ℝ, ∃ x y : ℝ, (f x + f y) / 2 < f ((x + y) / 2) + |x - y| := by
  sorry

end no_such_function_exists_l930_93032


namespace sum_of_42_odd_numbers_l930_93090

/-- The sum of the first n odd numbers -/
def sumOfOddNumbers (n : ℕ) : ℕ :=
  n * n

theorem sum_of_42_odd_numbers :
  sumOfOddNumbers 42 = 1764 := by
  sorry

#eval sumOfOddNumbers 42  -- This will output 1764

end sum_of_42_odd_numbers_l930_93090


namespace matrix_N_computation_l930_93088

theorem matrix_N_computation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec (![3, -2]) = ![4, 0])
  (h2 : N.mulVec (![(-4), 6]) = ![(-2), -2]) :
  N.mulVec (![7, 2]) = ![16, -4] := by
  sorry

end matrix_N_computation_l930_93088


namespace loan_interest_rate_l930_93052

theorem loan_interest_rate (principal : ℝ) (total_paid : ℝ) (time : ℝ) : 
  principal = 150 → 
  total_paid = 159 → 
  time = 1 → 
  (total_paid - principal) / (principal * time) = 0.06 := by
sorry

end loan_interest_rate_l930_93052


namespace isosceles_right_triangle_longest_side_l930_93050

-- Define the triangle
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define properties of the triangle
def isIsoscelesRight (t : Triangle) : Prop :=
  -- We don't implement the full definition, just declare it as a property
  sorry

def longestSide (t : Triangle) (side : ℝ) : Prop :=
  -- We don't implement the full definition, just declare it as a property
  sorry

def triangleArea (t : Triangle) : ℝ :=
  -- We don't implement the full calculation, just declare it as a function
  sorry

-- Main theorem
theorem isosceles_right_triangle_longest_side 
  (t : Triangle) 
  (h1 : isIsoscelesRight t) 
  (h2 : longestSide t (dist t.X t.Y)) 
  (h3 : triangleArea t = 49) : 
  dist t.X t.Y = 14 :=
sorry

-- Note: dist is a function that calculates the distance between two points

end isosceles_right_triangle_longest_side_l930_93050


namespace ellipse_hyperbola_same_foci_l930_93080

/-- Given an ellipse and a hyperbola with the same foci, prove that the semi-major axis of the ellipse is 4 -/
theorem ellipse_hyperbola_same_foci (a : ℝ) : a > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1) →
  (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1) →
  (∀ c : ℝ, c^2 = 7 → 
    (∀ x y : ℝ, x^2 / a^2 + y^2 / 9 = 1 → (x + c)^2 + y^2 = a^2 ∧ (x - c)^2 + y^2 = a^2) ∧
    (∀ x y : ℝ, x^2 / 4 - y^2 / 3 = 1 → (x + c)^2 - y^2 = 4 ∧ (x - c)^2 - y^2 = 4)) →
  a = 4 := by
sorry

end ellipse_hyperbola_same_foci_l930_93080


namespace similar_triangles_side_length_l930_93010

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle :=
  (a b c : ℝ)

/-- Checks if two triangles are similar -/
def are_similar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

theorem similar_triangles_side_length 
  (FGH IJK : Triangle)
  (h_similar : are_similar FGH IJK)
  (h_GH : FGH.c = 30)
  (h_FG : FGH.a = 24)
  (h_IJ : IJK.a = 20) :
  IJK.c = 25 := by
sorry

end similar_triangles_side_length_l930_93010


namespace quadratic_function_increasing_on_positive_x_l930_93002

theorem quadratic_function_increasing_on_positive_x 
  (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = x₁^2 - 1) 
  (h2 : y₂ = x₂^2 - 1) 
  (h3 : 0 < x₁) 
  (h4 : x₁ < x₂) : 
  y₁ < y₂ := by
sorry

end quadratic_function_increasing_on_positive_x_l930_93002


namespace number_of_girls_in_class_l930_93027

/-- Given a class where there are 3 more girls than boys and the total number of students is 41,
    prove that the number of girls in the class is 22. -/
theorem number_of_girls_in_class (boys girls : ℕ) : 
  girls = boys + 3 → 
  boys + girls = 41 → 
  girls = 22 := by
  sorry

end number_of_girls_in_class_l930_93027


namespace kitchen_length_l930_93093

/-- Calculates the length of a rectangular kitchen given its dimensions and painting information. -/
theorem kitchen_length (width height : ℝ) (total_area_painted : ℝ) : 
  width = 16 ∧ 
  height = 10 ∧ 
  total_area_painted = 1680 → 
  ∃ length : ℝ, length = 12 ∧ 
    total_area_painted / 3 = 2 * (length * height + width * height) :=
by sorry

end kitchen_length_l930_93093


namespace janes_coins_l930_93022

theorem janes_coins (q d : ℕ) : 
  q + d = 30 → 
  (10 * q + 25 * d) - (25 * q + 10 * d) = 150 →
  25 * q + 10 * d = 450 :=
by sorry

end janes_coins_l930_93022


namespace man_walking_problem_l930_93076

theorem man_walking_problem (x : ℝ) :
  let final_x := x - 6 * Real.sin (2 * Real.pi / 3)
  let final_y := 6 * Real.cos (2 * Real.pi / 3)
  final_x ^ 2 + final_y ^ 2 = 12 →
  x = 3 * Real.sqrt 3 + Real.sqrt 3 ∨ x = 3 * Real.sqrt 3 - Real.sqrt 3 :=
by sorry

end man_walking_problem_l930_93076


namespace backpack_price_equation_l930_93067

/-- Represents the price of a backpack after discounts -/
def discounted_price (x : ℝ) : ℝ := 0.8 * x - 10

/-- Theorem stating that the discounted price equals the final selling price -/
theorem backpack_price_equation (x : ℝ) : 
  discounted_price x = 90 ↔ 0.8 * x - 10 = 90 := by sorry

end backpack_price_equation_l930_93067


namespace intersection_M_N_l930_93005

def M : Set ℝ := {x | x ≥ -2}
def N : Set ℝ := {x | 1 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_M_N_l930_93005


namespace parabola_focus_for_x_squared_l930_93035

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := sorry

/-- A parabola is symmetric about the y-axis if its equation has no x term -/
def isSymmetricAboutYAxis (p : Parabola) : Prop := p.b = 0

theorem parabola_focus_for_x_squared (p : Parabola) 
  (h1 : p.a = 1) 
  (h2 : p.b = 0) 
  (h3 : p.c = 0) 
  (h4 : isSymmetricAboutYAxis p) : 
  focus p = (0, 1/4) := by sorry

end parabola_focus_for_x_squared_l930_93035


namespace soccer_penalty_kicks_l930_93097

/-- The number of penalty kicks in a soccer challenge --/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  goalies * (total_players - 1)

/-- Theorem: In a soccer team with 26 players, including 4 goalies, 
    where each player kicks against each goalie once, 
    the total number of penalty kicks is 100. --/
theorem soccer_penalty_kicks : 
  penalty_kicks 26 4 = 100 := by
sorry

#eval penalty_kicks 26 4

end soccer_penalty_kicks_l930_93097


namespace intersection_points_l930_93029

def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 10)^2 = 50

def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*(x - y) - 18 = 0

theorem intersection_points :
  (∃ (x y : ℝ), circle1 x y ∧ circle2 x y) ∧
  (circle1 3 3 ∧ circle2 3 3) ∧
  (circle1 (-3) 5 ∧ circle2 (-3) 5) ∧
  (∀ (x y : ℝ), circle1 x y ∧ circle2 x y → (x = 3 ∧ y = 3) ∨ (x = -3 ∧ y = 5)) :=
by sorry

end intersection_points_l930_93029


namespace exp_three_has_property_M_g_property_M_iff_l930_93056

-- Define property M
def has_property_M (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

-- Statement for f(x) = 3^x
theorem exp_three_has_property_M :
  ∃ x₀ : ℝ, (3 : ℝ)^(x₀ + 1) = (3 : ℝ)^x₀ + (3 : ℝ)^1 :=
sorry

-- Define g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := Real.log (a / (2 * x^2 + 1))

-- Statement for g(x)
theorem g_property_M_iff (a : ℝ) :
  (a > 0) →
  (has_property_M (g a) ↔ 6 - 3 * Real.sqrt 3 ≤ a ∧ a ≤ 6 + 3 * Real.sqrt 3) :=
sorry

end exp_three_has_property_M_g_property_M_iff_l930_93056


namespace integer_average_l930_93021

theorem integer_average (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  max a (max b (max c (max d e))) - min a (min b (min c (min d e))) = 10 →
  max a (max b (max c (max d e))) ≤ 68 →
  (a + b + c + d + e) / 5 = 60 :=
by
  sorry

end integer_average_l930_93021
