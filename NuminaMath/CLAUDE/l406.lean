import Mathlib

namespace age_problem_l406_40616

theorem age_problem (a b : ℕ) (h1 : 5 * b = 3 * a) (h2 : 7 * (b + 6) = 5 * (a + 6)) : a = 15 := by
  sorry

end age_problem_l406_40616


namespace y_derivative_l406_40694

open Real

noncomputable def y (x : ℝ) : ℝ :=
  Real.sqrt (9 * x^2 - 12 * x + 5) * arctan (3 * x - 2) - log (3 * x - 2 + Real.sqrt (9 * x^2 - 12 * x + 5))

theorem y_derivative (x : ℝ) :
  deriv y x = ((9 * x - 6) * arctan (3 * x - 2)) / Real.sqrt (9 * x^2 - 12 * x + 5) :=
by sorry

end y_derivative_l406_40694


namespace hyperbola_tangent_orthogonal_l406_40662

/-- Hyperbola C: 2x^2 - y^2 = 1 -/
def Hyperbola (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1

/-- Circle: x^2 + y^2 = 1 -/
def UnitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Line with slope k passing through (x, y) -/
def Line (k b x y : ℝ) : Prop := y = k * x + b

/-- Tangent condition for a line to the unit circle -/
def IsTangent (k b : ℝ) : Prop := b^2 = k^2 + 1

/-- Perpendicularity condition for two vectors -/
def IsOrthogonal (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

/-- Main theorem -/
theorem hyperbola_tangent_orthogonal (k b x1 y1 x2 y2 : ℝ) :
  |k| < Real.sqrt 2 →
  Hyperbola x1 y1 →
  Hyperbola x2 y2 →
  Line k b x1 y1 →
  Line k b x2 y2 →
  IsTangent k b →
  IsOrthogonal x1 y1 x2 y2 := by sorry

end hyperbola_tangent_orthogonal_l406_40662


namespace compare_expressions_l406_40688

theorem compare_expressions (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (a^2 + b^2) ≤ 2 * (a^3 + b^3) := by
  sorry

end compare_expressions_l406_40688


namespace smallest_four_digit_with_digit_product_12_l406_40646

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_product (n : ℕ) : ℕ :=
  (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

theorem smallest_four_digit_with_digit_product_12 :
  ∀ n : ℕ, is_four_digit n → digit_product n = 12 → 1126 ≤ n :=
by sorry

end smallest_four_digit_with_digit_product_12_l406_40646


namespace set1_equivalence_set2_equivalence_set3_equivalence_set4_equivalence_l406_40695

-- Define the sets of points for each condition
def set1 : Set (ℝ × ℝ) := {p | p.1 ≥ -2}
def set2 : Set (ℝ × ℝ) := {p | -2 < p.1 ∧ p.1 < 2}
def set3 : Set (ℝ × ℝ) := {p | |p.1| < 2}
def set4 : Set (ℝ × ℝ) := {p | |p.1| ≥ 2}

-- State the theorems to be proved
theorem set1_equivalence : set1 = {p : ℝ × ℝ | p.1 ≥ -2} := by sorry

theorem set2_equivalence : set2 = {p : ℝ × ℝ | -2 < p.1 ∧ p.1 < 2} := by sorry

theorem set3_equivalence : set3 = {p : ℝ × ℝ | -2 < p.1 ∧ p.1 < 2} := by sorry

theorem set4_equivalence : set4 = {p : ℝ × ℝ | p.1 ≤ -2 ∨ p.1 ≥ 2} := by sorry

end set1_equivalence_set2_equivalence_set3_equivalence_set4_equivalence_l406_40695


namespace crow_probability_l406_40667

theorem crow_probability (a b c d : ℕ) : 
  a + b = 50 →  -- Total crows on birch
  c + d = 50 →  -- Total crows on oak
  b ≥ a →       -- Black crows ≥ White crows on birch
  d ≥ c - 1 →   -- Black crows ≥ White crows - 1 on oak
  (b * (d + 1) + a * (c + 1)) / (50 * 51 : ℚ) > (b * c + a * d) / (50 * 51 : ℚ) :=
by sorry

end crow_probability_l406_40667


namespace sin_2alpha_value_l406_40605

theorem sin_2alpha_value (a α : ℝ) 
  (h : Real.sin (a + π/4) = Real.sqrt 2 * (Real.sin α + 2 * Real.cos α)) : 
  Real.sin (2 * α) = -3/5 := by
  sorry

end sin_2alpha_value_l406_40605


namespace abs_neg_three_l406_40644

theorem abs_neg_three : |(-3 : ℤ)| = 3 := by sorry

end abs_neg_three_l406_40644


namespace reduced_price_is_three_l406_40663

/-- Represents the price reduction and quantity increase for apples -/
structure ApplePriceReduction where
  reduction_percent : ℝ
  additional_apples : ℕ
  fixed_price : ℝ

/-- Calculates the reduced price per dozen apples given the price reduction information -/
def reduced_price_per_dozen (info : ApplePriceReduction) : ℝ :=
  sorry

/-- Theorem stating that for the given conditions, the reduced price per dozen is 3 Rs -/
theorem reduced_price_is_three (info : ApplePriceReduction) 
  (h1 : info.reduction_percent = 40)
  (h2 : info.additional_apples = 64)
  (h3 : info.fixed_price = 40) : 
  reduced_price_per_dozen info = 3 :=
sorry

end reduced_price_is_three_l406_40663


namespace complex_absolute_value_product_l406_40689

theorem complex_absolute_value_product : Complex.abs (3 - 2*Complex.I) * Complex.abs (3 + 2*Complex.I) = 13 := by
  sorry

end complex_absolute_value_product_l406_40689


namespace complex_parts_of_i_squared_plus_i_l406_40650

theorem complex_parts_of_i_squared_plus_i :
  let i : ℂ := Complex.I
  let z : ℂ := i^2 + i
  (z.re = -1) ∧ (z.im = 1) := by sorry

end complex_parts_of_i_squared_plus_i_l406_40650


namespace smallest_non_special_number_twenty_two_is_non_special_l406_40637

def triangle_number (k : ℕ) : ℕ := k * (k + 1) / 2

def is_prime_power (n : ℕ) : Prop :=
  ∃ (p k : ℕ), p.Prime ∧ k > 0 ∧ n = p ^ k

def is_prime_plus_one (n : ℕ) : Prop :=
  ∃ p : ℕ, p.Prime ∧ n = p + 1

theorem smallest_non_special_number :
  ∀ n : ℕ, n < 22 →
    (∃ k : ℕ, n = triangle_number k) ∨
    is_prime_power n ∨
    is_prime_plus_one n :=
  sorry

theorem twenty_two_is_non_special :
  ¬(∃ k : ℕ, 22 = triangle_number k) ∧
  ¬is_prime_power 22 ∧
  ¬is_prime_plus_one 22 :=
  sorry

end smallest_non_special_number_twenty_two_is_non_special_l406_40637


namespace complex_simplification_l406_40686

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_simplification :
  7 * (4 - 2*i) + 4*i * (3 - 2*i) = 36 - 2*i :=
by sorry

end complex_simplification_l406_40686


namespace arithmetic_sequence_properties_l406_40633

def arithmetic_sequence (a : ℕ → ℤ) := ∀ n, a (n + 1) - a n = a 1 - a 0

def sum_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ := 
  n * (a 0 + a (n - 1)) / 2

theorem arithmetic_sequence_properties 
  (a : ℕ → ℤ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum3 : sum_arithmetic_sequence a 3 = 42)
  (h_sum6 : sum_arithmetic_sequence a 6 = 57) :
  (∀ n, a n = 20 - 3 * n) ∧ 
  (∀ n, n ≤ 6 → sum_arithmetic_sequence a n ≤ sum_arithmetic_sequence a 6) :=
sorry

end arithmetic_sequence_properties_l406_40633


namespace max_take_home_pay_l406_40696

/-- The income that maximizes take-home pay given a specific tax rate and fee structure -/
theorem max_take_home_pay :
  let tax_rate (x : ℝ) := 2 * x / 100
  let admin_fee := 500
  let take_home_pay (x : ℝ) := 1000 * x - (tax_rate x * 1000 * x) - admin_fee
  ∃ (x : ℝ), ∀ (y : ℝ), take_home_pay x ≥ take_home_pay y ∧ x = 25 := by
sorry

end max_take_home_pay_l406_40696


namespace ellipse_focus_l406_40691

/-- An ellipse with semi-major axis 5 and semi-minor axis m has its left focus at (-3,0) -/
theorem ellipse_focus (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2 / 25 + y^2 / m^2 = 1) → (-3 : ℝ)^2 = 25 - m^2 → m = 4 :=
by sorry

end ellipse_focus_l406_40691


namespace christopher_age_l406_40600

theorem christopher_age (c g : ℕ) : 
  c = 2 * g →                  -- Christopher is 2 times as old as Gabriela now
  c - 9 = 5 * (g - 9) →        -- Nine years ago, Christopher was 5 times as old as Gabriela
  c = 24                       -- Christopher's current age is 24
  := by sorry

end christopher_age_l406_40600


namespace weight_of_b_l406_40651

theorem weight_of_b (a b c d : ℝ) 
  (h1 : (a + b + c + d) / 4 = 40)
  (h2 : (a + b) / 2 = 25)
  (h3 : (b + c) / 2 = 28)
  (h4 : (c + d) / 2 = 32) :
  b = 46 := by
sorry

end weight_of_b_l406_40651


namespace orange_bags_count_l406_40664

theorem orange_bags_count (weight_per_bag : ℕ) (total_weight : ℕ) (h1 : weight_per_bag = 23) (h2 : total_weight = 1035) :
  total_weight / weight_per_bag = 45 := by
  sorry

end orange_bags_count_l406_40664


namespace percentage_problem_l406_40636

theorem percentage_problem (p : ℝ) (h1 : 0.5 * 10 = p / 100 * 500 - 20) : p = 5 := by
  sorry

end percentage_problem_l406_40636


namespace tens_digit_of_difference_l406_40648

/-- Given a single digit t, prove that the tens digit of (6t5 - 5t6) is 9 -/
theorem tens_digit_of_difference (t : ℕ) (h : t < 10) : 
  (6 * 100 + t * 10 + 5) - (5 * 100 + t * 10 + 6) = 94 := by
  sorry

end tens_digit_of_difference_l406_40648


namespace negation_relationship_l406_40681

theorem negation_relationship (x : ℝ) :
  (¬(|x + 1| > 2) → ¬(5*x - 6 > x^2)) ∧
  ¬(¬(5*x - 6 > x^2) → ¬(|x + 1| > 2)) :=
by sorry

end negation_relationship_l406_40681


namespace boat_race_distance_l406_40683

/-- The distance between two points A and B traveled by two boats with different speeds and start times -/
theorem boat_race_distance 
  (a b d n : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hd : d ≥ 0) 
  (hn : n > 0) 
  (hab : a > b) :
  ∃ x : ℝ, x > 0 ∧ x = (a * (d + b * n)) / (a - b) ∧
    x / a + n = (x - d) / b :=
by sorry

end boat_race_distance_l406_40683


namespace restaurant_spirits_profit_l406_40665

/-- Calculates the profit made by a restaurant on a bottle of spirits -/
theorem restaurant_spirits_profit
  (bottle_cost : ℝ)
  (servings_per_bottle : ℕ)
  (price_per_serving : ℝ)
  (h1 : bottle_cost = 30)
  (h2 : servings_per_bottle = 16)
  (h3 : price_per_serving = 8) :
  servings_per_bottle * price_per_serving - bottle_cost = 98 :=
by sorry

end restaurant_spirits_profit_l406_40665


namespace profit_share_ratio_l406_40617

theorem profit_share_ratio (total_profit : ℚ) (difference : ℚ) 
  (h1 : total_profit = 800)
  (h2 : difference = 160) :
  ∃ (x y : ℚ), x + y = total_profit ∧ 
                |x - y| = difference ∧ 
                y / total_profit = 2 / 5 := by
  sorry

end profit_share_ratio_l406_40617


namespace competition_results_l406_40640

def seventh_grade_scores : List ℕ := [3, 6, 7, 6, 6, 8, 6, 9, 6, 10]
def eighth_grade_scores : List ℕ := [5, 6, 8, 7, 5, 8, 7, 9, 8, 8]

def mode (l : List ℕ) : ℕ := sorry

def average (l : List ℕ) : ℚ := sorry

def median (l : List ℕ) : ℚ := sorry

theorem competition_results :
  (mode seventh_grade_scores = 6) ∧
  (average eighth_grade_scores = 71 / 10) ∧
  (7 > median seventh_grade_scores) ∧
  (7 < median eighth_grade_scores) := by sorry

end competition_results_l406_40640


namespace min_value_sum_of_reciprocals_l406_40620

/-- Two circles with equations x^2 + y^2 + 2ax + a^2 - 4 = 0 and x^2 + y^2 - 4by - 1 + 4b^2 = 0 -/
def circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + 2*a*x + a^2 - 4 = 0
def circle2 (b : ℝ) (x y : ℝ) : Prop := x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The circles have exactly three common tangents -/
def have_three_common_tangents (a b : ℝ) : Prop := sorry

theorem min_value_sum_of_reciprocals (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) 
  (h3 : have_three_common_tangents a b) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), (1 / a^2 + 1 / b^2) ≥ m :=
sorry

end min_value_sum_of_reciprocals_l406_40620


namespace work_efficiency_l406_40692

theorem work_efficiency (sakshi_days tanya_days : ℝ) : 
  tanya_days = 16 →
  sakshi_days / 1.25 = tanya_days →
  sakshi_days = 20 := by
sorry

end work_efficiency_l406_40692


namespace enchiladas_and_tacos_price_l406_40624

-- Define the prices of enchiladas and tacos
noncomputable def enchilada_price : ℝ := sorry
noncomputable def taco_price : ℝ := sorry

-- Define the conditions
axiom condition1 : enchilada_price + 4 * taco_price = 3
axiom condition2 : 4 * enchilada_price + taco_price = 3.2

-- State the theorem
theorem enchiladas_and_tacos_price :
  4 * enchilada_price + 5 * taco_price = 5.55 := by sorry

end enchiladas_and_tacos_price_l406_40624


namespace real_solutions_condition_l406_40638

theorem real_solutions_condition (x : ℝ) :
  (∃ y : ℝ, y^2 + 6*x*y + x + 8 = 0) ↔ (x ≤ -8/9 ∨ x ≥ 1) :=
by sorry

end real_solutions_condition_l406_40638


namespace probability_within_four_rings_l406_40635

def P_first_ring : ℚ := 1 / 10
def P_second_ring : ℚ := 3 / 10
def P_third_ring : ℚ := 2 / 5
def P_fourth_ring : ℚ := 1 / 10

theorem probability_within_four_rings :
  P_first_ring + P_second_ring + P_third_ring + P_fourth_ring = 9 / 10 := by
  sorry

end probability_within_four_rings_l406_40635


namespace cut_scene_is_six_minutes_l406_40608

/-- The length of a cut scene from a movie --/
def cut_scene_length (original_length final_length : ℕ) : ℕ :=
  original_length - final_length

/-- Theorem: The length of the cut scene is 6 minutes --/
theorem cut_scene_is_six_minutes :
  let original_length : ℕ := 60  -- One hour in minutes
  let final_length : ℕ := 54
  cut_scene_length original_length final_length = 6 := by
  sorry

#eval cut_scene_length 60 54  -- This should output 6

end cut_scene_is_six_minutes_l406_40608


namespace smallest_binary_multiple_of_ten_l406_40601

def is_binary_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_binary_multiple_of_ten :
  ∃ X : ℕ, X > 0 ∧ 
  (∃ T : ℕ, T > 0 ∧ is_binary_number T ∧ T = 10 * X) ∧
  (∀ Y : ℕ, Y > 0 → 
    (∃ S : ℕ, S > 0 ∧ is_binary_number S ∧ S = 10 * Y) → 
    X ≤ Y) ∧
  X = 1 :=
sorry

end smallest_binary_multiple_of_ten_l406_40601


namespace elena_savings_theorem_l406_40682

/-- The amount Elena saves when buying binders with a discount and rebate -/
def elenaSavings (numBinders : ℕ) (pricePerBinder : ℚ) (discountRate : ℚ) (rebateThreshold : ℚ) (rebateAmount : ℚ) : ℚ :=
  let originalCost := numBinders * pricePerBinder
  let discountedPrice := originalCost * (1 - discountRate)
  let finalPrice := if originalCost > rebateThreshold then discountedPrice - rebateAmount else discountedPrice
  originalCost - finalPrice

/-- Theorem stating that Elena saves $10.25 under the given conditions -/
theorem elena_savings_theorem :
  elenaSavings 7 3 (25 / 100) 20 5 = (41 / 4) := by
  sorry

end elena_savings_theorem_l406_40682


namespace german_enrollment_l406_40697

theorem german_enrollment (total_students : ℕ) (both_subjects : ℕ) (only_english : ℕ) 
  (h1 : total_students = 32)
  (h2 : both_subjects = 12)
  (h3 : only_english = 10)
  (h4 : total_students = both_subjects + only_english + (total_students - (both_subjects + only_english))) :
  total_students - (both_subjects + only_english) + both_subjects = 22 := by
  sorry

#check german_enrollment

end german_enrollment_l406_40697


namespace milk_replacement_percentage_l406_40606

theorem milk_replacement_percentage (x : ℝ) : 
  (((100 - x) / 100) * ((100 - x) / 100) * ((100 - x) / 100)) * 100 = 51.20000000000001 → 
  x = 20 := by
sorry

end milk_replacement_percentage_l406_40606


namespace roots_equality_l406_40619

theorem roots_equality (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ ≠ x₂) 
  (h2 : x₁^2 - x₁ + a = 0) 
  (h3 : x₂^2 - x₂ + a = 0) : 
  |x₁^2 - x₂^2| = 1 ↔ |x₁^3 - x₂^3| = 1 := by
sorry

end roots_equality_l406_40619


namespace bob_sandwich_options_l406_40698

/-- Represents the number of different types of bread available. -/
def num_breads : ℕ := 5

/-- Represents the number of different types of meat available. -/
def num_meats : ℕ := 7

/-- Represents the number of different types of cheese available. -/
def num_cheeses : ℕ := 6

/-- Represents the number of sandwiches with turkey and mozzarella cheese. -/
def turkey_mozzarella_combos : ℕ := num_breads

/-- Represents the number of sandwiches with rye bread and beef. -/
def rye_beef_combos : ℕ := num_cheeses

/-- Calculates the total number of possible sandwich combinations. -/
def total_combos : ℕ := num_breads * num_meats * num_cheeses

/-- Theorem stating the number of different sandwiches Bob could order. -/
theorem bob_sandwich_options : 
  total_combos - turkey_mozzarella_combos - rye_beef_combos = 199 := by
  sorry

end bob_sandwich_options_l406_40698


namespace water_left_in_ml_l406_40670

-- Define the total amount of water in liters
def total_water : ℝ := 135.1

-- Define the size of each bucket in liters
def bucket_size : ℝ := 7

-- Define the conversion factor from liters to milliliters
def liters_to_ml : ℝ := 1000

-- Theorem statement
theorem water_left_in_ml :
  (total_water - bucket_size * ⌊total_water / bucket_size⌋) * liters_to_ml = 2100 := by
  sorry


end water_left_in_ml_l406_40670


namespace ellipse_hyperbola_foci_coincide_l406_40654

/-- Given an ellipse and a hyperbola, if the endpoints of the major axis of the ellipse
    coincide with the foci of the hyperbola, then m = 2 -/
theorem ellipse_hyperbola_foci_coincide (m : ℝ) : 
  (∀ x y : ℝ, x^2/3 + y^2/4 = 1 → 
    (∃ a : ℝ, a > 0 ∧ (x = 0 ∧ y = a ∨ x = 0 ∧ y = -a) ∧
    ∀ x' y' : ℝ, y'^2/2 - x'^2/m = 1 → 
      (x' = 0 ∧ y' = a ∨ x' = 0 ∧ y' = -a))) → 
  m = 2 := by
sorry

end ellipse_hyperbola_foci_coincide_l406_40654


namespace min_distance_theorem_l406_40699

/-- Line l in polar form -/
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = 2 * Real.sqrt 2

/-- Circle C in Cartesian form -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Scaling transformation -/
def scaling (x y x' y' : ℝ) : Prop := x' = 2 * x ∧ y' = 3 * y

/-- Curve C' after scaling transformation -/
def curve_C' (x' y' : ℝ) : Prop := x'^2 / 4 + y'^2 / 9 = 1

theorem min_distance_theorem :
  (∀ ρ θ, line_l ρ θ) →
  (∀ x y, circle_C x y) →
  (∃ d : ℝ, d = 2 * Real.sqrt 2 - 1 ∧
    (∀ x y, circle_C x y → (x + y - 4)^2 / 2 ≥ d^2)) ∧
  (∃ d' : ℝ, d' = 2 * Real.sqrt 2 - 2 ∧
    (∀ x' y', curve_C' x' y' → (x' + y' - 4)^2 / 2 ≥ d'^2)) := by
  sorry

end min_distance_theorem_l406_40699


namespace multiply_polynomials_l406_40612

theorem multiply_polynomials (x y : ℝ) :
  (3 * x^4 - 2 * y^3) * (9 * x^8 + 6 * x^4 * y^3 + 4 * y^6) = 27 * x^12 - 8 * y^9 := by
  sorry

end multiply_polynomials_l406_40612


namespace divide_people_eq_280_l406_40632

/-- The number of ways to divide 8 people into three groups -/
def divide_people : ℕ :=
  let total_people : ℕ := 8
  let group_1_size : ℕ := 3
  let group_2_size : ℕ := 3
  let group_3_size : ℕ := 2
  let ways_to_choose_group_1 := Nat.choose total_people group_1_size
  let ways_to_choose_group_2 := Nat.choose (total_people - group_1_size) group_2_size
  let ways_to_choose_group_3 := Nat.choose group_3_size group_3_size
  let arrangements_of_identical_groups : ℕ := 2  -- 2! for two identical groups of 3
  (ways_to_choose_group_1 * ways_to_choose_group_2 * ways_to_choose_group_3) / arrangements_of_identical_groups

theorem divide_people_eq_280 : divide_people = 280 := by
  sorry

end divide_people_eq_280_l406_40632


namespace imaginary_part_of_z_l406_40625

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = Complex.I) :
  z.im = 1/2 := by sorry

end imaginary_part_of_z_l406_40625


namespace throne_identity_l406_40659

/-- Represents the types of beings in this problem -/
inductive Being
| Knight  -- Always tells the truth
| Liar    -- Always lies
| Human   -- Can either tell the truth or lie
| Monkey  -- An animal

/-- Represents a statement made by a being -/
structure Statement where
  content : Prop
  speaker : Being

/-- The statement made by the being on the throne -/
def throneStatement : Statement :=
  { content := (∃ x : Being, x = Being.Liar ∧ x = Being.Monkey),
    speaker := Being.Human }

/-- Theorem stating that the being on the throne must be a human who is lying -/
theorem throne_identity :
  throneStatement.speaker = Being.Human ∧ 
  ¬throneStatement.content :=
sorry

end throne_identity_l406_40659


namespace fraction_equality_l406_40628

theorem fraction_equality (q r s t : ℚ) 
  (h1 : q / r = 8)
  (h2 : s / r = 12)
  (h3 : s / t = 4) :
  t / q = 3 / 8 := by
sorry

end fraction_equality_l406_40628


namespace ellipse_properties_l406_40666

/-- Definition of the ellipse M -/
def ellipse_M (x y : ℝ) (a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / 3 = 1 ∧ a > 0

/-- One focus of the ellipse is at (-1, 0) -/
def focus_F : ℝ × ℝ := (-1, 0)

/-- A line l passing through F -/
def line_l (k : ℝ) (x : ℝ) : ℝ := k * (x + 1)

/-- Theorem stating the main results -/
theorem ellipse_properties :
  ∃ (a : ℝ),
    -- 1. The equation of the ellipse
    (∀ x y : ℝ, ellipse_M x y a ↔ x^2 / 4 + y^2 / 3 = 1) ∧
    -- 2. Length of CD when l has a 45° angle
    (∃ C D : ℝ × ℝ,
      C.1 ≠ D.1 ∧
      ellipse_M C.1 C.2 a ∧
      ellipse_M D.1 D.2 a ∧
      C.2 = line_l 1 C.1 ∧
      D.2 = line_l 1 D.1 ∧
      Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 24 / 7) ∧
    -- 3. Maximum value of |S₁ - S₂|
    (∃ S_max : ℝ,
      S_max = Real.sqrt 3 ∧
      ∀ k : ℝ,
        ∃ C D : ℝ × ℝ,
          C.1 ≠ D.1 ∧
          ellipse_M C.1 C.2 a ∧
          ellipse_M D.1 D.2 a ∧
          C.2 = line_l k C.1 ∧
          D.2 = line_l k D.1 ∧
          |C.2 - D.2| ≤ S_max) := by
  sorry

end ellipse_properties_l406_40666


namespace variety_show_probability_variety_show_probability_proof_l406_40647

/-- The probability of selecting 2 dance performances out of 3 for the first 3 slots
    in a randomly arranged program of 8 performances (5 singing, 3 dance) -/
theorem variety_show_probability : ℚ :=
  let total_performances : ℕ := 8
  let singing_performances : ℕ := 5
  let dance_performances : ℕ := 3
  let first_slots : ℕ := 3
  let required_dance : ℕ := 2

  3 / 28

theorem variety_show_probability_proof :
  variety_show_probability = 3 / 28 := by
  sorry

end variety_show_probability_variety_show_probability_proof_l406_40647


namespace matrix_sum_proof_l406_40618

theorem matrix_sum_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 0, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; 7, -10]
  A + B = !![-2, 5; 7, -5] := by
  sorry

end matrix_sum_proof_l406_40618


namespace john_max_books_l406_40669

/-- The maximum number of books John can buy given his money and the price per book -/
def max_books_buyable (total_money : ℕ) (price_per_book : ℕ) : ℕ :=
  total_money / price_per_book

/-- Proof that John can buy at most 14 books -/
theorem john_max_books :
  let john_money : ℕ := 4575  -- 45 dollars and 75 cents in cents
  let book_price : ℕ := 325   -- 3 dollars and 25 cents in cents
  max_books_buyable john_money book_price = 14 := by
sorry

end john_max_books_l406_40669


namespace marcos_lap_time_improvement_l406_40642

/-- Represents the improvement in lap time for Marcos after training -/
theorem marcos_lap_time_improvement :
  let initial_laps : ℕ := 15
  let initial_time : ℕ := 45
  let final_laps : ℕ := 18
  let final_time : ℕ := 42
  let initial_lap_time := initial_time / initial_laps
  let final_lap_time := final_time / final_laps
  let improvement := initial_lap_time - final_lap_time
  improvement = 2 / 3 := by sorry

end marcos_lap_time_improvement_l406_40642


namespace rams_weight_increase_percentage_l406_40621

theorem rams_weight_increase_percentage
  (weight_ratio : ℚ) -- Ratio of Ram's weight to Shyam's weight
  (total_weight_after : ℝ) -- Total weight after increase
  (total_increase_percentage : ℝ) -- Total weight increase percentage
  (shyam_increase_percentage : ℝ) -- Shyam's weight increase percentage
  (h1 : weight_ratio = 4 / 5) -- Condition: weight ratio is 4:5
  (h2 : total_weight_after = 82.8) -- Condition: total weight after increase is 82.8 kg
  (h3 : total_increase_percentage = 15) -- Condition: total weight increase is 15%
  (h4 : shyam_increase_percentage = 19) -- Condition: Shyam's weight increased by 19%
  : ∃ (ram_increase_percentage : ℝ), ram_increase_percentage = 10 :=
by sorry

end rams_weight_increase_percentage_l406_40621


namespace gyroscope_spin_rate_doubling_time_l406_40614

/-- The time interval for which a gyroscope's spin rate doubles -/
theorem gyroscope_spin_rate_doubling_time (v₀ v t : ℝ) (h₁ : v₀ = 6.25) (h₂ : v = 400) (h₃ : t = 90) :
  ∃ T : ℝ, v = v₀ * 2^(t/T) ∧ T = 15 := by
  sorry

end gyroscope_spin_rate_doubling_time_l406_40614


namespace area_ratio_theorem_l406_40690

-- Define the triangles
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the properties of the triangles
def is_30_60_90 (t : Triangle) : Prop := sorry

def is_right_angled_isosceles (t : Triangle) : Prop := sorry

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the theorem
theorem area_ratio_theorem (FGH IFG EGH IEH : Triangle) :
  is_30_60_90 FGH →
  is_right_angled_isosceles EGH →
  (area IFG) / (area IEH) = 1 / 2 :=
sorry

end area_ratio_theorem_l406_40690


namespace part_one_part_two_l406_40639

/-- Definition of a midpoint equation -/
def is_midpoint_equation (a b : ℚ) : Prop :=
  a ≠ 0 ∧ (- b / a) = (a + b) / 2

/-- Part 1: Prove that 4x - 8/3 = 0 is a midpoint equation -/
theorem part_one : is_midpoint_equation 4 (-8/3) := by
  sorry

/-- Part 2: Prove that for 5x + m - 1 = 0 to be a midpoint equation, m = -18/7 -/
theorem part_two : ∃ m : ℚ, is_midpoint_equation 5 (m - 1) ↔ m = -18/7 := by
  sorry

end part_one_part_two_l406_40639


namespace multiply_98_98_l406_40676

theorem multiply_98_98 : 98 * 98 = 9604 := by
  sorry

end multiply_98_98_l406_40676


namespace angle_complement_quadrant_l406_40653

/-- An angle is in the first quadrant if it's between 0 and π/2 radians -/
def is_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < Real.pi / 2

/-- An angle is in the second quadrant if it's between π/2 and π radians -/
def is_second_quadrant (α : ℝ) : Prop := Real.pi / 2 < α ∧ α < Real.pi

theorem angle_complement_quadrant (α : ℝ) 
  (h : is_first_quadrant α) : is_second_quadrant (Real.pi - α) := by
  sorry

end angle_complement_quadrant_l406_40653


namespace work_completion_time_l406_40613

theorem work_completion_time (a b : ℝ) (h1 : a = 2 * b) (h2 : 1 / (a + b) = 1 / 20) :
  1 / a = 1 / 30 := by
  sorry

end work_completion_time_l406_40613


namespace sum_of_real_and_imag_parts_l406_40680

theorem sum_of_real_and_imag_parts (z : ℂ) (h : z * (2 + Complex.I) = 2 * Complex.I - 1) :
  z.re + z.im = 1/5 := by
  sorry

end sum_of_real_and_imag_parts_l406_40680


namespace pascal_triangle_specific_number_l406_40643

/-- The number of elements in the nth row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

/-- The kth number in the nth row of Pascal's triangle -/
def pascal_number (n k : ℕ) : ℕ := Nat.choose n (k - 1)

theorem pascal_triangle_specific_number :
  pascal_number 50 10 = 2586948580 := by sorry

end pascal_triangle_specific_number_l406_40643


namespace rational_inequality_solution_l406_40603

theorem rational_inequality_solution (x : ℝ) : 
  x ≠ -5 → ((x - 2) / (x + 5) ≥ 0 ↔ x < -5 ∨ x ≥ 2) :=
by sorry

end rational_inequality_solution_l406_40603


namespace sum_of_A_and_C_is_six_l406_40602

theorem sum_of_A_and_C_is_six (A B C D : ℕ) : 
  A ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  B ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  C ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  D ∈ ({1, 2, 3, 4, 5} : Set ℕ) →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  (A : ℚ) / B + (C : ℚ) / D = 2 →
  A + C = 6 := by
sorry

end sum_of_A_and_C_is_six_l406_40602


namespace reducible_fraction_implies_divisibility_l406_40684

theorem reducible_fraction_implies_divisibility 
  (a b c d l k p q : ℤ) 
  (h1 : a * l + b = k * p) 
  (h2 : c * l + d = k * q) : 
  k ∣ (a * d - b * c) := by
  sorry

end reducible_fraction_implies_divisibility_l406_40684


namespace monotonic_cos_plus_linear_monotonic_cos_plus_linear_converse_l406_40630

/-- A function f : ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = cos x + ax is monotonic, then a ∈ (-∞, -1] ∪ [1, +∞) -/
theorem monotonic_cos_plus_linear (a : ℝ) :
  Monotonic (fun x => Real.cos x + a * x) → a ≤ -1 ∨ a ≥ 1 := by
  sorry

/-- The converse: if a ∈ (-∞, -1] ∪ [1, +∞), then f(x) = cos x + ax is monotonic -/
theorem monotonic_cos_plus_linear_converse (a : ℝ) :
  (a ≤ -1 ∨ a ≥ 1) → Monotonic (fun x => Real.cos x + a * x) := by
  sorry

end monotonic_cos_plus_linear_monotonic_cos_plus_linear_converse_l406_40630


namespace square_of_negative_product_l406_40687

theorem square_of_negative_product (a b : ℝ) : (-3 * a^2 * b)^2 = 9 * a^4 * b^2 := by
  sorry

end square_of_negative_product_l406_40687


namespace price_decrease_percentage_l406_40627

theorem price_decrease_percentage (base_price : ℝ) (regular_price : ℝ) (promotional_price : ℝ) : 
  regular_price = base_price * (1 + 0.25) ∧ 
  promotional_price = base_price →
  (regular_price - promotional_price) / regular_price = 0.20 := by
sorry

end price_decrease_percentage_l406_40627


namespace water_consumption_ratio_l406_40626

theorem water_consumption_ratio (initial_volume : ℝ) (first_drink_fraction : ℝ) (final_volume : ℝ) :
  initial_volume = 4 →
  first_drink_fraction = 1/4 →
  final_volume = 1 →
  let remaining_after_first := initial_volume - first_drink_fraction * initial_volume
  let second_drink := remaining_after_first - final_volume
  (second_drink / remaining_after_first) = 2/3 := by sorry

end water_consumption_ratio_l406_40626


namespace jellybean_average_increase_l406_40671

theorem jellybean_average_increase (initial_bags : ℕ) (initial_average : ℚ) (additional_jellybeans : ℕ) : 
  initial_bags = 34 →
  initial_average = 117 →
  additional_jellybeans = 362 →
  (((initial_bags : ℚ) * initial_average + additional_jellybeans) / (initial_bags + 1 : ℚ)) - initial_average = 7 :=
by sorry

end jellybean_average_increase_l406_40671


namespace remainder_1493829_div_7_l406_40607

theorem remainder_1493829_div_7 : 1493829 % 7 = 1 := by sorry

end remainder_1493829_div_7_l406_40607


namespace find_b_value_l406_40678

theorem find_b_value (a b c : ℤ) : 
  a + b + c = 111 → 
  (a + 10 = b - 10) ∧ (b - 10 = 3 * c) → 
  b = 58 :=
by sorry

end find_b_value_l406_40678


namespace gum_sticks_in_twelve_boxes_l406_40631

/-- Represents the number of sticks of gum in a given number of brown boxes -/
def sticks_in_boxes (full_boxes : ℕ) (half_boxes : ℕ) : ℕ :=
  let sticks_per_pack : ℕ := 5
  let packs_per_carton : ℕ := 7
  let cartons_per_full_box : ℕ := 6
  let cartons_per_half_box : ℕ := 3
  let sticks_per_carton : ℕ := sticks_per_pack * packs_per_carton
  let sticks_per_full_box : ℕ := sticks_per_carton * cartons_per_full_box
  let sticks_per_half_box : ℕ := sticks_per_carton * cartons_per_half_box
  full_boxes * sticks_per_full_box + half_boxes * sticks_per_half_box

/-- Theorem stating that 12 brown boxes with 2 half-full boxes contain 2310 sticks of gum -/
theorem gum_sticks_in_twelve_boxes : sticks_in_boxes 10 2 = 2310 := by
  sorry

end gum_sticks_in_twelve_boxes_l406_40631


namespace rotation180_maps_points_and_is_isometry_l406_40661

-- Define the points
def A : ℝ × ℝ := (-2, 1)
def A' : ℝ × ℝ := (2, -1)
def B : ℝ × ℝ := (-1, 4)
def B' : ℝ × ℝ := (1, -4)

-- Define the rotation function
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement
theorem rotation180_maps_points_and_is_isometry :
  (rotate180 A = A') ∧ 
  (rotate180 B = B') ∧ 
  (∀ p q : ℝ × ℝ, dist p q = dist (rotate180 p) (rotate180 q)) := by
  sorry


end rotation180_maps_points_and_is_isometry_l406_40661


namespace max_vertex_coordinate_sum_l406_40634

/-- Given a parabola y = ax^2 + bx + c passing through (0,0), (2T,0), and (2T+1,35),
    where a and T are integers and T ≠ 0, the maximum sum of vertex coordinates is 34. -/
theorem max_vertex_coordinate_sum :
  ∀ (a T : ℤ) (b c : ℝ),
    T ≠ 0 →
    (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ 
      (x = 0 ∧ y = 0) ∨ 
      (x = 2 * T ∧ y = 0) ∨ 
      (x = 2 * T + 1 ∧ y = 35)) →
    (∃ (N : ℝ), N = T - a * T^2 ∧ 
      (∀ (N' : ℝ), (∃ (a' T' : ℤ) (b' c' : ℝ),
        T' ≠ 0 ∧
        (∀ x y : ℝ, y = a' * x^2 + b' * x + c' ↔ 
          (x = 0 ∧ y = 0) ∨ 
          (x = 2 * T' ∧ y = 0) ∨ 
          (x = 2 * T' + 1 ∧ y = 35)) ∧
        N' = T' - a' * T'^2) → N' ≤ N)) →
    (∃ (N : ℝ), N = 34 ∧
      (∀ (N' : ℝ), (∃ (a' T' : ℤ) (b' c' : ℝ),
        T' ≠ 0 ∧
        (∀ x y : ℝ, y = a' * x^2 + b' * x + c' ↔ 
          (x = 0 ∧ y = 0) ∨ 
          (x = 2 * T' ∧ y = 0) ∨ 
          (x = 2 * T' + 1 ∧ y = 35)) ∧
        N' = T' - a' * T'^2) → N' ≤ N)) :=
by sorry

end max_vertex_coordinate_sum_l406_40634


namespace work_completion_theorem_l406_40609

/-- Represents the number of days it takes for a person to complete the work alone -/
structure WorkRate :=
  (days : ℝ)
  (positive : days > 0)

/-- Represents the state of the work project -/
structure WorkProject :=
  (rate_a : WorkRate)
  (rate_b : WorkRate)
  (total_days : ℝ)
  (a_left_before : Bool)

/-- Calculate the number of days A left before completion -/
def days_a_left_before (project : WorkProject) : ℝ :=
  sorry

theorem work_completion_theorem (project : WorkProject) 
  (h1 : project.rate_a.days = 10)
  (h2 : project.rate_b.days = 20)
  (h3 : project.total_days = 10)
  (h4 : project.a_left_before = true) :
  days_a_left_before project = 5 := by
  sorry

end work_completion_theorem_l406_40609


namespace percentage_problem_l406_40668

theorem percentage_problem (x : ℝ) (p : ℝ) 
  (h1 : 0.2 * x = 80) 
  (h2 : p / 100 * x = 160) : 
  p = 40 := by
  sorry

end percentage_problem_l406_40668


namespace orange_picking_fraction_l406_40673

/-- Proves that the fraction of oranges picked from each tree is 2/5 --/
theorem orange_picking_fraction
  (num_trees : ℕ)
  (fruits_per_tree : ℕ)
  (remaining_fruits : ℕ)
  (h1 : num_trees = 8)
  (h2 : fruits_per_tree = 200)
  (h3 : remaining_fruits = 960)
  (h4 : remaining_fruits < num_trees * fruits_per_tree) :
  (num_trees * fruits_per_tree - remaining_fruits) / (num_trees * fruits_per_tree) = 2 / 5 :=
by sorry

end orange_picking_fraction_l406_40673


namespace equation_has_real_root_l406_40657

theorem equation_has_real_root (a b c : ℝ) : 
  ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 := by
  sorry

end equation_has_real_root_l406_40657


namespace bowling_ball_weight_l406_40685

theorem bowling_ball_weight :
  ∀ (b c : ℝ),
  (5 * b = 3 * c) →
  (2 * c = 56) →
  (b = 16.8) :=
by
  sorry

end bowling_ball_weight_l406_40685


namespace kahi_memorized_words_l406_40629

theorem kahi_memorized_words (total : ℕ) (yesterday_fraction : ℚ) (today_fraction : ℚ)
  (h_total : total = 810)
  (h_yesterday : yesterday_fraction = 1 / 9)
  (h_today : today_fraction = 1 / 4) :
  (total - yesterday_fraction * total) * today_fraction = 180 := by
  sorry

end kahi_memorized_words_l406_40629


namespace min_value_theorem_l406_40655

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧
  ∀ (z : ℝ), z = 2 / x + 1 / y → z ≥ min_val :=
by sorry

end min_value_theorem_l406_40655


namespace binary_multiplication_1111_111_l406_40652

theorem binary_multiplication_1111_111 :
  (0b1111 : Nat) * 0b111 = 0b1001111 := by sorry

end binary_multiplication_1111_111_l406_40652


namespace adjacent_knights_probability_l406_40658

/-- The number of knights at the round table -/
def total_knights : ℕ := 30

/-- The number of knights selected for the quest -/
def selected_knights : ℕ := 4

/-- The probability that at least two of the selected knights were sitting next to each other -/
def adjacent_probability : ℚ := 943 / 1023

/-- Theorem stating the probability of at least two out of four randomly selected knights 
    sitting next to each other in a round table of 30 knights -/
theorem adjacent_knights_probability : 
  let total_ways := Nat.choose total_knights selected_knights
  let non_adjacent_ways := (total_knights - selected_knights) * 
                           (total_knights - selected_knights - 3) * 
                           (total_knights - selected_knights - 6) * 
                           (total_knights - selected_knights - 9)
  (1 : ℚ) - (non_adjacent_ways : ℚ) / total_ways = adjacent_probability := by
  sorry

#eval adjacent_probability.num + adjacent_probability.den

end adjacent_knights_probability_l406_40658


namespace figurine_arrangement_l406_40675

/-- The number of ways to arrange n uniquely sized figurines in a line -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n uniquely sized figurines in a line,
    with two specific figurines at opposite ends -/
def arrangementsWithEndsFixed (n : ℕ) : ℕ := 2 * arrangements (n - 2)

theorem figurine_arrangement :
  arrangementsWithEndsFixed 9 = 10080 := by
  sorry

end figurine_arrangement_l406_40675


namespace cathy_commission_l406_40679

theorem cathy_commission (x : ℝ) : 
  0.15 * (x - 15) = 0.25 * (x - 25) → 
  0.1 * (x - 10) = 3 := by
sorry

end cathy_commission_l406_40679


namespace A_minus_2B_specific_value_A_minus_2B_independent_of_x_l406_40645

/-- The algebraic expression A -/
def A (x y : ℝ) : ℝ := 2 * x^2 + 3 * x * y + 2 * y

/-- The algebraic expression B -/
def B (x y : ℝ) : ℝ := x^2 - x * y + x

/-- Theorem 1: A - 2B equals -20 when x = -2 and y = 3 -/
theorem A_minus_2B_specific_value : A (-2) 3 - 2 * B (-2) 3 = -20 := by sorry

/-- Theorem 2: A - 2B is independent of x when y = 2/5 -/
theorem A_minus_2B_independent_of_x : 
  ∀ x : ℝ, A x (2/5) - 2 * B x (2/5) = A 0 (2/5) - 2 * B 0 (2/5) := by sorry

end A_minus_2B_specific_value_A_minus_2B_independent_of_x_l406_40645


namespace intersection_M_N_l406_40615

open Set Real

-- Define the sets M and N
def M : Set ℝ := {x | 2 - x > 0}
def N : Set ℝ := Icc 1 3

-- State the theorem
theorem intersection_M_N : M ∩ N = Ico 1 2 := by sorry

end intersection_M_N_l406_40615


namespace symmetric_line_problem_l406_40641

/-- Given a line with equation ax + by + c = 0, 
    returns the equation of the line symmetric to it with respect to y = x -/
def symmetric_line (a b c : ℝ) : ℝ × ℝ × ℝ := (b, a, c)

/-- The equation of the line symmetric to 3x-5y+1=0 with respect to y=x -/
theorem symmetric_line_problem : 
  symmetric_line 3 (-5) 1 = (5, -3, -1) := by sorry

end symmetric_line_problem_l406_40641


namespace water_in_final_mixture_l406_40677

/-- Given a mixture where x liters of 10% acid solution is added to 5 liters of pure acid,
    resulting in a final mixture that is 40% water, prove that the amount of water
    in the final mixture is 3.6 liters. -/
theorem water_in_final_mixture :
  ∀ x : ℝ,
  x > 0 →
  0.4 * (5 + x) = 0.9 * x →
  0.9 * x = 3.6 :=
by
  sorry

end water_in_final_mixture_l406_40677


namespace gcd_problem_l406_40604

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 953) :
  Int.gcd (3 * b^2 + 17 * b + 23) (b + 19) = 34 := by
  sorry

end gcd_problem_l406_40604


namespace complex_exponent_calculation_l406_40622

theorem complex_exponent_calculation : 
  ((-8 : ℂ) ^ (2/3 : ℂ)) * ((1 / Real.sqrt 2) ^ (-2 : ℂ)) * ((27 : ℂ) ^ (-1/3 : ℂ)) = 8/3 := by
  sorry

end complex_exponent_calculation_l406_40622


namespace min_value_x_plus_y_l406_40674

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  x + y ≥ 16 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 1 ∧ x + y = 16 := by
  sorry

end min_value_x_plus_y_l406_40674


namespace expand_expression_l406_40623

theorem expand_expression (x : ℝ) : (7*x + 11) * (3*x^2 + 2*x) = 21*x^3 + 47*x^2 + 22*x := by
  sorry

end expand_expression_l406_40623


namespace maryann_work_time_l406_40693

theorem maryann_work_time (total_time calling_time accounting_time report_time : ℕ) : 
  total_time = 1440 ∧
  accounting_time = 2 * calling_time ∧
  report_time = 3 * accounting_time ∧
  total_time = calling_time + accounting_time + report_time →
  report_time = 960 := by
  sorry

end maryann_work_time_l406_40693


namespace sam_paul_study_difference_l406_40672

def average_difference (differences : List Int) : Int :=
  (differences.sum / differences.length)

theorem sam_paul_study_difference : 
  let differences : List Int := [20, 5, -5, 0, 15, -10, 10]
  average_difference differences = 5 := by
  sorry

end sam_paul_study_difference_l406_40672


namespace principal_amount_proof_l406_40610

/-- Proves that given the specified conditions, the principal amount is 7200 --/
theorem principal_amount_proof (rate : ℝ) (time : ℝ) (diff : ℝ) (P : ℝ) 
  (h1 : rate = 5 / 100)
  (h2 : time = 2)
  (h3 : diff = 18)
  (h4 : P * (1 + rate)^time - P - (P * rate * time) = diff) :
  P = 7200 := by
  sorry

#check principal_amount_proof

end principal_amount_proof_l406_40610


namespace graph_transformation_l406_40656

-- Define the original function f
variable (f : ℝ → ℝ)

-- Define the symmetry operation with respect to x = 1
def symmetry_x1 (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (2 - x)

-- Define the left shift operation
def shift_left (g : ℝ → ℝ) (units : ℝ) : ℝ → ℝ := λ x => g (x + units)

-- Theorem statement
theorem graph_transformation (f : ℝ → ℝ) :
  shift_left (symmetry_x1 f) 2 = λ x => f (1 - x) := by sorry

end graph_transformation_l406_40656


namespace largest_valid_number_l406_40660

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n % 100 = (n / 100 % 10 + n / 1000) % 10 * 10 + (n / 10 % 10 + n / 100 % 10) % 10)

theorem largest_valid_number : 
  is_valid_number 9099 ∧ ∀ m : ℕ, is_valid_number m → m ≤ 9099 :=
sorry

end largest_valid_number_l406_40660


namespace solution_implies_a_value_l406_40611

theorem solution_implies_a_value (x y a : ℝ) : 
  x = -2 → y = 1 → 2 * x + a * y = 3 → a = 7 := by sorry

end solution_implies_a_value_l406_40611


namespace interest_rate_difference_l406_40649

/-- Proves that the difference in interest rates is 5% given the specified conditions -/
theorem interest_rate_difference
  (principal : ℝ)
  (time : ℝ)
  (interest_difference : ℝ)
  (h1 : principal = 800)
  (h2 : time = 10)
  (h3 : interest_difference = 400)
  : ∃ (r1 r2 : ℝ), r2 - r1 = 5 ∧ 
    principal * r2 * time / 100 = principal * r1 * time / 100 + interest_difference :=
sorry

end interest_rate_difference_l406_40649
