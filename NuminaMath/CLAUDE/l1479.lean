import Mathlib

namespace square_root_and_abs_simplification_l1479_147934

theorem square_root_and_abs_simplification :
  Real.sqrt ((-2)^2) + |Real.sqrt 2 - Real.sqrt 3| - |Real.sqrt 3 - 1| = 3 - Real.sqrt 2 := by
  sorry

end square_root_and_abs_simplification_l1479_147934


namespace measure_one_kg_cereal_l1479_147924

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | Balanced
  | Unbalanced

/-- Represents a weighing operation -/
def weighing (left right : ℕ) : WeighingResult :=
  if left = right then WeighingResult.Balanced else WeighingResult.Unbalanced

/-- Represents the process of measuring cereal -/
def measureCereal (totalCereal weight : ℕ) (maxWeighings : ℕ) : Prop :=
  ∃ (firstLeft firstRight secondLeft secondRight : ℕ),
    firstLeft + firstRight = totalCereal ∧
    secondLeft + secondRight ≤ firstLeft ∧
    weighing (firstLeft - secondLeft) (firstRight + weight) = WeighingResult.Balanced ∧
    weighing secondLeft weight = WeighingResult.Balanced ∧
    secondRight = 1 ∧
    2 ≤ maxWeighings

/-- Theorem stating that it's possible to measure 1 kg of cereal from 11 kg using a 3 kg weight in two weighings -/
theorem measure_one_kg_cereal :
  measureCereal 11 3 2 := by sorry

end measure_one_kg_cereal_l1479_147924


namespace book_pages_problem_l1479_147911

theorem book_pages_problem (x : ℕ) (h1 : x > 0) (h2 : x + (x + 1) = 125) : x + 1 = 63 := by
  sorry

end book_pages_problem_l1479_147911


namespace man_twice_son_age_l1479_147957

/-- Represents the number of years until a man's age is twice his son's age. -/
def yearsUntilTwiceAge (sonAge : ℕ) (ageDifference : ℕ) : ℕ :=
  2

theorem man_twice_son_age (sonAge : ℕ) (ageDifference : ℕ) 
  (h1 : sonAge = 25) 
  (h2 : ageDifference = 27) : 
  yearsUntilTwiceAge sonAge ageDifference = 2 := by
  sorry

end man_twice_son_age_l1479_147957


namespace store_promotion_probabilities_l1479_147902

/-- A store promotion event with three prizes -/
structure StorePromotion where
  p_first : ℝ  -- Probability of winning first prize
  p_second : ℝ  -- Probability of winning second prize
  p_third : ℝ  -- Probability of winning third prize
  h_first : 0 ≤ p_first ∧ p_first ≤ 1
  h_second : 0 ≤ p_second ∧ p_second ≤ 1
  h_third : 0 ≤ p_third ∧ p_third ≤ 1

/-- The probability of winning a prize in the store promotion -/
def prob_win_prize (sp : StorePromotion) : ℝ :=
  sp.p_first + sp.p_second + sp.p_third

/-- The probability of not winning any prize in the store promotion -/
def prob_no_prize (sp : StorePromotion) : ℝ :=
  1 - prob_win_prize sp

/-- Theorem stating the probabilities for a specific store promotion -/
theorem store_promotion_probabilities (sp : StorePromotion) 
  (h1 : sp.p_first = 0.1) (h2 : sp.p_second = 0.2) (h3 : sp.p_third = 0.4) : 
  prob_win_prize sp = 0.7 ∧ prob_no_prize sp = 0.3 := by
  sorry

end store_promotion_probabilities_l1479_147902


namespace product_of_sum_of_four_squares_l1479_147906

theorem product_of_sum_of_four_squares (a b : ℤ)
  (ha : ∃ x₁ x₂ x₃ x₄ : ℤ, a = x₁^2 + x₂^2 + x₃^2 + x₄^2)
  (hb : ∃ y₁ y₂ y₃ y₄ : ℤ, b = y₁^2 + y₂^2 + y₃^2 + y₄^2) :
  ∃ z₁ z₂ z₃ z₄ : ℤ, a * b = z₁^2 + z₂^2 + z₃^2 + z₄^2 :=
by sorry

end product_of_sum_of_four_squares_l1479_147906


namespace infinite_composite_values_l1479_147940

theorem infinite_composite_values (m n k : ℕ) :
  (∃ f : ℕ → ℕ, ∀ k ≥ 2, f k = 4 * k^4) ∧
  (∀ k ≥ 2, ∀ m, ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ m^4 + 4 * k^4 = a * b) :=
sorry

end infinite_composite_values_l1479_147940


namespace marilyn_bananas_count_l1479_147993

/-- The number of boxes Marilyn has for her bananas. -/
def num_boxes : ℕ := 8

/-- The number of bananas required in each box. -/
def bananas_per_box : ℕ := 5

/-- Theorem stating that Marilyn has 40 bananas in total. -/
theorem marilyn_bananas_count :
  num_boxes * bananas_per_box = 40 := by
  sorry

end marilyn_bananas_count_l1479_147993


namespace negation_of_universal_proposition_l1479_147908

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x - 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 - 2*x₀ - 1 ≤ 0) := by sorry

end negation_of_universal_proposition_l1479_147908


namespace imaginary_part_of_z_l1479_147966

theorem imaginary_part_of_z (z : ℂ) (h : (1 : ℂ) / z = 1 / (1 + 2*I) + 1 / (1 - I)) : 
  z.im = -(1 / 5 : ℝ) := by
  sorry

end imaginary_part_of_z_l1479_147966


namespace multiple_of_q_in_equation_l1479_147939

theorem multiple_of_q_in_equation (p q m : ℚ) 
  (h1 : p / q = 3 / 4)
  (h2 : 3 * p + m * q = 25 / 4) :
  m = 4 := by
sorry

end multiple_of_q_in_equation_l1479_147939


namespace hyperbola_standard_equation_l1479_147965

/-- Given a hyperbola C passing through the point (1,1) with asymptotes 2x+y=0 and 2x-y=0,
    its standard equation is 4x²/3 - y²/3 = 1. -/
theorem hyperbola_standard_equation (C : Set (ℝ × ℝ)) :
  (∀ x y, (x, y) ∈ C ↔ 4 * x^2 / 3 - y^2 / 3 = 1) ↔
  ((1, 1) ∈ C ∧
   (∀ x y, 2*x + y = 0 → (x, y) ∈ frontier C) ∧
   (∀ x y, 2*x - y = 0 → (x, y) ∈ frontier C)) :=
by sorry

end hyperbola_standard_equation_l1479_147965


namespace imaginary_part_of_z_l1479_147932

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2*I)*z = 3 - 2*I) : 
  z.im = -8/5 := by sorry

end imaginary_part_of_z_l1479_147932


namespace circle_division_theorem_l1479_147916

/-- The number of regions a circle is divided into by radii and concentric circles -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem: A circle with 16 radii and 10 concentric circles is divided into 176 regions -/
theorem circle_division_theorem :
  num_regions 16 10 = 176 := by
  sorry

end circle_division_theorem_l1479_147916


namespace triangles_in_polygon_l1479_147958

/-- The number of triangles formed by diagonals passing through one vertex of an n-sided polygon -/
def triangles_from_diagonals (n : ℕ) : ℕ :=
  n - 2

/-- Theorem stating that the number of triangles formed by diagonals passing through one vertex
    of an n-sided polygon is equal to (n-2) -/
theorem triangles_in_polygon (n : ℕ) (h : n ≥ 3) :
  triangles_from_diagonals n = n - 2 := by
  sorry

end triangles_in_polygon_l1479_147958


namespace square_calculation_identity_l1479_147914

theorem square_calculation_identity (x : ℝ) : ((x + 1)^3 - (x - 1)^3 - 2) / 6 = x^2 := by
  sorry

end square_calculation_identity_l1479_147914


namespace highlighters_count_l1479_147984

/-- The number of pink highlighters in the teacher's desk -/
def pink_highlighters : ℕ := 3

/-- The number of yellow highlighters in the teacher's desk -/
def yellow_highlighters : ℕ := 7

/-- The number of blue highlighters in the teacher's desk -/
def blue_highlighters : ℕ := 5

/-- The total number of highlighters in the teacher's desk -/
def total_highlighters : ℕ := pink_highlighters + yellow_highlighters + blue_highlighters

theorem highlighters_count : total_highlighters = 15 := by
  sorry

end highlighters_count_l1479_147984


namespace smallest_with_18_divisors_l1479_147903

/-- Number of positive integer divisors of n -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- n has exactly 18 positive integer divisors -/
def has_18_divisors (n : ℕ) : Prop := num_divisors n = 18

theorem smallest_with_18_divisors :
  ∃ (n : ℕ), has_18_divisors n ∧ ∀ m : ℕ, has_18_divisors m → n ≤ m :=
by sorry

end smallest_with_18_divisors_l1479_147903


namespace decimal_equivalent_of_one_fifth_squared_l1479_147933

theorem decimal_equivalent_of_one_fifth_squared :
  (1 / 5 : ℚ) ^ 2 = (4 : ℚ) / 100 := by
  sorry

end decimal_equivalent_of_one_fifth_squared_l1479_147933


namespace smaller_side_of_rearranged_rectangle_l1479_147942

/-- Represents a rectangle with given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the result of dividing and rearranging a rectangle -/
structure RearrangedRectangle where
  original : Rectangle
  new : Rectangle
  is_valid : original.width * original.height = new.width * new.height

/-- The theorem to be proved -/
theorem smaller_side_of_rearranged_rectangle 
  (r : RearrangedRectangle) 
  (h1 : r.original.width = 10) 
  (h2 : r.original.height = 25) :
  min r.new.width r.new.height = 10 := by
  sorry

#check smaller_side_of_rearranged_rectangle

end smaller_side_of_rearranged_rectangle_l1479_147942


namespace quadratic_inequality_solution_l1479_147969

theorem quadratic_inequality_solution (x : ℝ) :
  -x^2 - 2*x + 3 < 0 ↔ x < -3 ∨ x > 1 := by
  sorry

end quadratic_inequality_solution_l1479_147969


namespace trigonometric_expression_equality_l1479_147905

theorem trigonometric_expression_equality : 
  (Real.sin (7 * π / 180) + Real.sin (8 * π / 180) * Real.cos (15 * π / 180)) / 
  (Real.cos (7 * π / 180) - Real.sin (8 * π / 180) * Real.sin (15 * π / 180)) = 
  2 - Real.sqrt 3 := by
  sorry

end trigonometric_expression_equality_l1479_147905


namespace rhombus_area_l1479_147992

/-- The area of a rhombus with side length 4 cm and an interior angle of 30 degrees is 8 cm² -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 6) :
  s * s * Real.sin θ = 8 :=
sorry

end rhombus_area_l1479_147992


namespace supply_lasts_18_months_l1479_147928

/-- Represents the number of pills in a supply -/
def supply : ℕ := 60

/-- Represents the fraction of a pill taken per dose -/
def dose : ℚ := 1 / 3

/-- Represents the number of days between doses -/
def days_between_doses : ℕ := 3

/-- Represents the average number of days in a month -/
def days_per_month : ℕ := 30

/-- Calculates the number of months a supply of medicine will last -/
def months_supply_lasts : ℚ :=
  (supply : ℚ) * (days_between_doses : ℚ) / dose / days_per_month

theorem supply_lasts_18_months :
  months_supply_lasts = 18 := by sorry

end supply_lasts_18_months_l1479_147928


namespace cafe_combinations_l1479_147909

/-- The number of drinks on the menu -/
def menu_size : ℕ := 8

/-- Whether Yann orders coffee -/
def yann_orders_coffee : Bool := sorry

/-- The number of options available to Camille -/
def camille_options : ℕ :=
  if yann_orders_coffee then menu_size - 1 else menu_size

/-- The number of combinations when Yann orders coffee -/
def coffee_combinations : ℕ := 1 * (menu_size - 1)

/-- The number of combinations when Yann doesn't order coffee -/
def non_coffee_combinations : ℕ := (menu_size - 1) * menu_size

/-- The total number of different combinations of drinks Yann and Camille can order -/
def total_combinations : ℕ := coffee_combinations + non_coffee_combinations

theorem cafe_combinations : total_combinations = 63 := by sorry

end cafe_combinations_l1479_147909


namespace vinegar_percentage_second_brand_l1479_147925

/-- Calculates the vinegar percentage in the second brand of Italian dressing -/
theorem vinegar_percentage_second_brand 
  (total_volume : ℝ) 
  (desired_vinegar_percentage : ℝ) 
  (first_brand_volume : ℝ) 
  (second_brand_volume : ℝ) 
  (first_brand_vinegar_percentage : ℝ)
  (h1 : total_volume = 320)
  (h2 : desired_vinegar_percentage = 11)
  (h3 : first_brand_volume = 128)
  (h4 : second_brand_volume = 128)
  (h5 : first_brand_vinegar_percentage = 8) :
  ∃ (second_brand_vinegar_percentage : ℝ),
    second_brand_vinegar_percentage = 19.5 ∧
    (first_brand_volume * first_brand_vinegar_percentage / 100 + 
     second_brand_volume * second_brand_vinegar_percentage / 100) / total_volume * 100 = 
    desired_vinegar_percentage :=
by sorry

end vinegar_percentage_second_brand_l1479_147925


namespace min_value_theorem_l1479_147971

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geo_mean : Real.sqrt 3 = Real.sqrt (3^a * 3^(2*b))) : 
  2/a + 1/b ≥ 8 := by
sorry

end min_value_theorem_l1479_147971


namespace polygon_sides_l1479_147979

theorem polygon_sides (sum_interior_angles : ℝ) : sum_interior_angles = 1080 → ∃ n : ℕ, n = 8 ∧ (n - 2) * 180 = sum_interior_angles := by
  sorry

end polygon_sides_l1479_147979


namespace square_remainder_l1479_147954

theorem square_remainder (n : ℤ) : n % 5 = 3 → n^2 % 5 = 4 := by
  sorry

end square_remainder_l1479_147954


namespace trapezoid_larger_base_l1479_147968

/-- Given a trapezoid with base ratio 1:3 and midline length 24, 
    prove the larger base is 36 -/
theorem trapezoid_larger_base 
  (shorter_base longer_base midline : ℝ) 
  (h_ratio : longer_base = 3 * shorter_base) 
  (h_midline : midline = (shorter_base + longer_base) / 2) 
  (h_midline_length : midline = 24) : 
  longer_base = 36 := by
sorry


end trapezoid_larger_base_l1479_147968


namespace subtraction_problem_l1479_147947

theorem subtraction_problem (n : ℝ) (h : n = 5) : ∃! x : ℝ, 7 * n - x = 2 * n + 10 ∧ x = 15 := by
  sorry

end subtraction_problem_l1479_147947


namespace smaller_number_theorem_l1479_147995

theorem smaller_number_theorem (x y : ℝ) : 
  x + y = 15 → x * y = 36 → min x y = 3 := by
sorry

end smaller_number_theorem_l1479_147995


namespace complex_equality_implies_modulus_l1479_147952

theorem complex_equality_implies_modulus (x y : ℝ) :
  (Complex.I + 1) * Complex.mk x y = 2 →
  Complex.abs (Complex.mk (2*x) y) = Real.sqrt 5 := by
  sorry

end complex_equality_implies_modulus_l1479_147952


namespace arctg_sum_eq_pi_fourth_l1479_147919

theorem arctg_sum_eq_pi_fourth (x : ℝ) (h : x > -1) : 
  Real.arctan x + Real.arctan ((1 - x) / (1 + x)) = π / 4 := by
  sorry

end arctg_sum_eq_pi_fourth_l1479_147919


namespace trigonometric_equation_l1479_147915

theorem trigonometric_equation (α : Real) 
  (h : (Real.tan α - 3) * (Real.sin α + Real.cos α + 3) = 0) : 
  ((4 * Real.sin α + 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 1) ∧ 
  (2 + (2/3) * (Real.sin α)^2 + (1/4) * (Real.cos α)^2 = 21/8) := by
  sorry

end trigonometric_equation_l1479_147915


namespace lenny_grocery_expense_l1479_147981

/-- Proves the amount Lenny spent at the grocery store, given his initial amount, video game expense, and remaining amount. -/
theorem lenny_grocery_expense (initial : ℕ) (video_games : ℕ) (remaining : ℕ) 
  (h1 : initial = 84)
  (h2 : video_games = 24)
  (h3 : remaining = 39) :
  initial - video_games - remaining = 21 := by
  sorry

#check lenny_grocery_expense

end lenny_grocery_expense_l1479_147981


namespace modified_cube_painted_faces_l1479_147918

/-- Represents a cube with its 8 corner small cubes removed and its surface painted -/
structure ModifiedCube where
  size : ℕ
  corner_removed : Bool
  surface_painted : Bool

/-- Counts the number of small cubes with a given number of painted faces -/
def count_painted_faces (c : ModifiedCube) (n : ℕ) : ℕ :=
  sorry

/-- The main theorem about the number of painted faces in a modified cube -/
theorem modified_cube_painted_faces (c : ModifiedCube) 
  (h1 : c.size > 2) 
  (h2 : c.corner_removed = true) 
  (h3 : c.surface_painted = true) : 
  (count_painted_faces c 4 = 12) ∧ 
  (count_painted_faces c 1 = 6) ∧ 
  (count_painted_faces c 0 = 1) :=
sorry

end modified_cube_painted_faces_l1479_147918


namespace sandys_savings_difference_l1479_147949

/-- Calculates the difference in savings between two years given the initial salary,
    savings percentages, and salary increase. -/
def savings_difference (initial_salary : ℝ) (savings_percent_year1 : ℝ) 
                       (savings_percent_year2 : ℝ) (salary_increase_percent : ℝ) : ℝ :=
  let salary_year2 := initial_salary * (1 + salary_increase_percent)
  let savings_year1 := initial_salary * savings_percent_year1
  let savings_year2 := salary_year2 * savings_percent_year2
  savings_year1 - savings_year2

/-- The difference in Sandy's savings between two years is $925.20 -/
theorem sandys_savings_difference :
  savings_difference 45000 0.083 0.056 0.115 = 925.20 := by
  sorry

end sandys_savings_difference_l1479_147949


namespace square_last_digit_l1479_147990

theorem square_last_digit (n : ℕ) 
  (h : (n^2 / 10) % 10 = 7) : 
  n^2 % 10 = 6 := by
sorry

end square_last_digit_l1479_147990


namespace symmetric_lines_l1479_147985

/-- Given two lines in the xy-plane, this function returns true if they are symmetric about the line x = a -/
def are_symmetric_lines (line1 line2 : ℝ → ℝ → Prop) (a : ℝ) : Prop :=
  ∀ x y, line1 x y ↔ line2 (2*a - x) y

/-- The equation of the first line: 2x + y - 1 = 0 -/
def line1 (x y : ℝ) : Prop := 2*x + y - 1 = 0

/-- The equation of the second line: 2x - y - 3 = 0 -/
def line2 (x y : ℝ) : Prop := 2*x - y - 3 = 0

/-- The line of symmetry: x = 1 -/
def symmetry_line : ℝ := 1

theorem symmetric_lines : are_symmetric_lines line1 line2 symmetry_line := by
  sorry

end symmetric_lines_l1479_147985


namespace small_pump_fills_in_three_hours_l1479_147989

-- Define the filling rates for the pumps
def large_pump_rate : ℝ := 4 -- 1 / (1/4)
def combined_time : ℝ := 0.23076923076923078

-- Define the time it takes for the small pump to fill the tank
def small_pump_time : ℝ := 3

-- Theorem statement
theorem small_pump_fills_in_three_hours :
  let combined_rate := 1 / combined_time
  let small_pump_rate := combined_rate - large_pump_rate
  1 / small_pump_rate = small_pump_time := by sorry

end small_pump_fills_in_three_hours_l1479_147989


namespace principal_amount_l1479_147953

/-- Proves that given the conditions of the problem, the principal amount is 3000 --/
theorem principal_amount (P : ℝ) : 
  (P * 4 * 5) / 100 = P - 2400 → P = 3000 := by
  sorry

end principal_amount_l1479_147953


namespace exists_m_for_all_n_l1479_147959

theorem exists_m_for_all_n (n : ℕ+) : ∃ m : ℤ, (2^(2^n.val) - 1) ∣ (m^2 + 9) := by
  sorry

end exists_m_for_all_n_l1479_147959


namespace election_winner_votes_l1479_147910

theorem election_winner_votes (total_votes : ℕ) : 
  (total_votes : ℚ) * (58 / 100) - (total_votes : ℚ) * (42 / 100) = 288 →
  (total_votes : ℚ) * (58 / 100) = 1044 := by
  sorry

end election_winner_votes_l1479_147910


namespace inequality_range_l1479_147962

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) → 
  -2 < a ∧ a ≤ 2 :=
by sorry

end inequality_range_l1479_147962


namespace ac_eq_b_squared_necessary_not_sufficient_l1479_147938

/-- Definition of a geometric progression for three real numbers -/
def isGeometricProgression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

/-- The main theorem stating that ac = b^2 is necessary but not sufficient for a, b, c to be in geometric progression -/
theorem ac_eq_b_squared_necessary_not_sufficient :
  (∀ a b c : ℝ, isGeometricProgression a b c → a * c = b^2) ∧
  (∃ a b c : ℝ, a * c = b^2 ∧ ¬isGeometricProgression a b c) := by
  sorry

end ac_eq_b_squared_necessary_not_sufficient_l1479_147938


namespace identity_matrix_solution_l1479_147994

def matrix_equation (N : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  N^4 - 3 • N^3 + 3 • N^2 - N = !![5, 15; 0, 5]

theorem identity_matrix_solution :
  ∃! N : Matrix (Fin 2) (Fin 2) ℝ, matrix_equation N ∧ N = 1 := by sorry

end identity_matrix_solution_l1479_147994


namespace rectangle_least_area_l1479_147998

theorem rectangle_least_area :
  ∀ l w : ℕ,
  l = 3 * w →
  2 * (l + w) = 120 →
  ∀ l' w' : ℕ,
  l' = 3 * w' →
  2 * (l' + w') = 120 →
  l * w ≤ l' * w' →
  l * w = 675 :=
by sorry

end rectangle_least_area_l1479_147998


namespace steve_final_marbles_l1479_147951

theorem steve_final_marbles (steve_initial sam_initial sally_initial : ℕ) 
  (h1 : sam_initial = 2 * steve_initial)
  (h2 : sally_initial = sam_initial - 5)
  (h3 : sam_initial - 6 = 8) : 
  steve_initial + 3 = 10 := by
sorry

end steve_final_marbles_l1479_147951


namespace all_composites_reachable_l1479_147941

/-- A proper divisor of n is a positive integer that divides n and is not equal to 1 or n. -/
def ProperDivisor (d n : ℕ) : Prop :=
  d ∣ n ∧ d ≠ 1 ∧ d ≠ n

/-- The set of numbers that can be obtained by starting from 4 and repeatedly adding proper divisors. -/
inductive Reachable : ℕ → Prop
  | base : Reachable 4
  | step {n m : ℕ} : Reachable n → ProperDivisor m n → Reachable (n + m)

/-- A composite number is a natural number greater than 1 that is not prime. -/
def Composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬ Nat.Prime n

/-- Theorem: Any composite number can be reached by starting from 4 and repeatedly adding proper divisors. -/
theorem all_composites_reachable : ∀ n : ℕ, Composite n → Reachable n := by
  sorry

end all_composites_reachable_l1479_147941


namespace inequality_solution_l1479_147926

theorem inequality_solution (x : ℝ) : (x + 2) / (x + 4) ≤ 3 ↔ -5 < x ∧ x < -4 := by
  sorry

end inequality_solution_l1479_147926


namespace solution_set_abs_equation_l1479_147978

theorem solution_set_abs_equation (x : ℝ) :
  |x - 2| + |2*x - 3| = |3*x - 5| ↔ x ≤ 3/2 ∨ x ≥ 2 := by sorry

end solution_set_abs_equation_l1479_147978


namespace factorization_of_ax_squared_minus_9a_l1479_147960

theorem factorization_of_ax_squared_minus_9a (a x : ℝ) : a * x^2 - 9 * a = a * (x - 3) * (x + 3) := by
  sorry

end factorization_of_ax_squared_minus_9a_l1479_147960


namespace post_office_mailing_l1479_147912

def total_cost : ℚ := 449/100
def letter_cost : ℚ := 37/100
def package_cost : ℚ := 88/100
def num_letters : ℕ := 5

theorem post_office_mailing :
  ∃ (num_packages : ℕ),
    letter_cost * num_letters + package_cost * num_packages = total_cost ∧
    num_letters - num_packages = 2 :=
by sorry

end post_office_mailing_l1479_147912


namespace fraction_equality_l1479_147961

theorem fraction_equality (m n p q r : ℚ) 
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5)
  (h4 : m / r = 10) :
  r / q = 1 / 10 := by
  sorry

end fraction_equality_l1479_147961


namespace max_cities_in_network_l1479_147930

/-- Represents a city in the airline network -/
structure City where
  id : Nat

/-- Represents the airline network -/
structure AirlineNetwork where
  cities : Finset City
  connections : City → Finset City

/-- The maximum number of direct connections a city can have -/
def maxDirectConnections : Nat := 3

/-- Defines a valid airline network based on the given conditions -/
def isValidNetwork (network : AirlineNetwork) : Prop :=
  ∀ c ∈ network.cities,
    (network.connections c).card ≤ maxDirectConnections ∧
    ∀ d ∈ network.cities, 
      c ≠ d → (d ∈ network.connections c ∨ 
               ∃ e ∈ network.cities, e ∈ network.connections c ∧ d ∈ network.connections e)

/-- The theorem stating the maximum number of cities in a valid network -/
theorem max_cities_in_network (network : AirlineNetwork) 
  (h : isValidNetwork network) : network.cities.card ≤ 10 := by
  sorry

end max_cities_in_network_l1479_147930


namespace base_twelve_representation_l1479_147904

def is_three_digit (n : ℕ) (b : ℕ) : Prop :=
  b ^ 2 ≤ n ∧ n < b ^ 3

def has_odd_final_digit (n : ℕ) (b : ℕ) : Prop :=
  n % b % 2 = 1

theorem base_twelve_representation : 
  is_three_digit 125 12 ∧ has_odd_final_digit 125 12 ∧ 
  ∀ b : ℕ, b ≠ 12 → ¬(is_three_digit 125 b ∧ has_odd_final_digit 125 b) :=
sorry

end base_twelve_representation_l1479_147904


namespace hyperbola_equation_theorem_l1479_147983

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- The slope of the asymptotes -/
  asymptote_slope : ℝ
  /-- A point that the hyperbola passes through -/
  point : ℝ × ℝ

/-- The equation of a hyperbola given its asymptotes and a point it passes through -/
def hyperbola_equation (h : Hyperbola) : ℝ → ℝ → Prop :=
  fun x y => x^2 - (y^2 / (4 * h.asymptote_slope^2)) = 1

/-- Theorem stating that a hyperbola with asymptotes y = ±2x passing through (√2, 2) has the equation x² - y²/4 = 1 -/
theorem hyperbola_equation_theorem (h : Hyperbola) 
    (h_asymptote : h.asymptote_slope = 2)
    (h_point : h.point = (Real.sqrt 2, 2)) :
    hyperbola_equation h = fun x y => x^2 - y^2/4 = 1 := by
  sorry

#check hyperbola_equation_theorem

end hyperbola_equation_theorem_l1479_147983


namespace average_age_combined_l1479_147973

theorem average_age_combined (n_students : ℕ) (n_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  n_students = 33 →
  n_parents = 55 →
  avg_age_students = 11 →
  avg_age_parents = 33 →
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents : ℚ) = 24.75 := by
  sorry

end average_age_combined_l1479_147973


namespace rectangle_subdivision_l1479_147921

/-- A rectangle can be subdivided into n pairwise noncongruent rectangles similar to the original -/
def can_subdivide (n : ℕ) : Prop :=
  ∃ (r : ℝ) (h : r > 1), ∃ (rectangles : Fin n → ℝ × ℝ),
    (∀ i j, i ≠ j → rectangles i ≠ rectangles j) ∧
    (∀ i, (rectangles i).1 / (rectangles i).2 = r)

theorem rectangle_subdivision (n : ℕ) (h : n > 1) :
  can_subdivide n ↔ n ≥ 3 :=
sorry

end rectangle_subdivision_l1479_147921


namespace three_team_leads_per_supervisor_l1479_147900

/-- Represents the organizational structure of a company -/
structure Company where
  workers : ℕ
  team_leads : ℕ
  supervisors : ℕ
  worker_to_lead_ratio : ℕ

/-- Calculates the number of team leads per supervisor -/
def team_leads_per_supervisor (c : Company) : ℚ :=
  c.team_leads / c.supervisors

/-- Theorem: The number of team leads per supervisor is 3 -/
theorem three_team_leads_per_supervisor (c : Company) 
  (h1 : c.worker_to_lead_ratio = 10)
  (h2 : c.supervisors = 13)
  (h3 : c.workers = 390) :
  team_leads_per_supervisor c = 3 := by
  sorry

#check three_team_leads_per_supervisor

end three_team_leads_per_supervisor_l1479_147900


namespace pump_fill_time_l1479_147967

theorem pump_fill_time (P : ℝ) (h1 : P > 0) (h2 : 14 > 0) :
  1 / P - 1 / 14 = 1 / (7 / 3) → P = 2 := by
  sorry

end pump_fill_time_l1479_147967


namespace derivative_of_f_l1479_147987

/-- The function f(x) = 3x^2 -/
def f (x : ℝ) : ℝ := 3 * x^2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x

theorem derivative_of_f (x : ℝ) : deriv f x = f' x := by
  sorry

end derivative_of_f_l1479_147987


namespace min_sum_squares_l1479_147976

def S : Finset Int := {-9, -4, -3, 0, 1, 5, 8, 10}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  32 ≤ (p + q + r + s)^2 + (t + u + v + w)^2 :=
by
  sorry

#check min_sum_squares

end min_sum_squares_l1479_147976


namespace henrys_socks_l1479_147945

theorem henrys_socks (a b c : ℕ) : 
  a + b + c = 15 →
  2 * a + 3 * b + 5 * c = 36 →
  a ≥ 1 →
  b ≥ 1 →
  c ≥ 1 →
  a = 11 :=
by sorry

end henrys_socks_l1479_147945


namespace nails_per_plank_l1479_147999

theorem nails_per_plank (large_planks : ℕ) (additional_nails : ℕ) (total_nails : ℕ) :
  large_planks = 13 →
  additional_nails = 8 →
  total_nails = 229 →
  ∃ (nails_per_plank : ℕ), nails_per_plank * large_planks + additional_nails = total_nails ∧ nails_per_plank = 17 :=
by
  sorry

end nails_per_plank_l1479_147999


namespace three_digit_difference_divisible_by_nine_l1479_147913

theorem three_digit_difference_divisible_by_nine :
  ∀ (a b c : ℕ), 
  0 ≤ a ∧ a ≤ 9 →
  0 ≤ b ∧ b ≤ 9 →
  0 ≤ c ∧ c ≤ 9 →
  ∃ (k : ℤ), (100 * a + 10 * b + c) - (a + b + c) = 9 * k :=
by sorry

end three_digit_difference_divisible_by_nine_l1479_147913


namespace product_equals_fraction_l1479_147991

def product_term (n : ℕ) : ℚ :=
  (2 * (n^4 - 1)) / (2 * (n^4 + 1))

def product_result : ℚ :=
  (product_term 2) * (product_term 3) * (product_term 4) * 
  (product_term 5) * (product_term 6) * (product_term 7)

theorem product_equals_fraction : product_result = 4400 / 135 := by
  sorry

end product_equals_fraction_l1479_147991


namespace average_age_proof_l1479_147982

/-- Given three people a, b, and c, this theorem proves that if their average age is 28 years
    and the age of b is 26 years, then the average age of a and c is 29 years. -/
theorem average_age_proof (a b c : ℕ) : 
  (a + b + c) / 3 = 28 → b = 26 → (a + c) / 2 = 29 := by
  sorry

end average_age_proof_l1479_147982


namespace new_total_weight_l1479_147986

/-- Proves that the new total weight of Ram and Shyam is 13.8 times their original common weight factor -/
theorem new_total_weight (x : ℝ) (x_pos : x > 0) : 
  let ram_original := 7 * x
  let shyam_original := 5 * x
  let ram_new := ram_original * 1.1
  let shyam_new := shyam_original * 1.22
  let total_original := ram_original + shyam_original
  let total_new := ram_new + shyam_new
  total_new = total_original * 1.15 ∧ total_new = 13.8 * x :=
by sorry

end new_total_weight_l1479_147986


namespace cubic_arithmetic_progression_l1479_147972

/-- 
A cubic equation x^3 + ax^2 + bx + c = 0 has three real roots forming an arithmetic progression 
if and only if the following conditions are satisfied:
1) ab/3 - 2a^3/27 - c = 0
2) a^3/3 - b ≥ 0
-/
theorem cubic_arithmetic_progression (a b c : ℝ) : 
  (∃ x y z : ℝ, x < y ∧ y < z ∧ 
    (∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = x ∨ t = y ∨ t = z) ∧
    y - x = z - y) ↔ 
  (a*b/3 - 2*a^3/27 - c = 0 ∧ a^3/3 - b ≥ 0) :=
sorry

end cubic_arithmetic_progression_l1479_147972


namespace halloween_houses_per_hour_l1479_147937

theorem halloween_houses_per_hour 
  (num_children : ℕ) 
  (num_hours : ℕ) 
  (treats_per_child_per_house : ℕ) 
  (total_treats : ℕ) 
  (h1 : num_children = 3)
  (h2 : num_hours = 4)
  (h3 : treats_per_child_per_house = 3)
  (h4 : total_treats = 180) :
  total_treats / (num_children * num_hours * treats_per_child_per_house) = 5 := by
  sorry

end halloween_houses_per_hour_l1479_147937


namespace scout_troop_profit_l1479_147922

-- Define the problem parameters
def candy_bars : ℕ := 1500
def buy_price : ℚ := 1 / 3
def transport_cost : ℕ := 50
def sell_price : ℚ := 3 / 5

-- Define the net profit calculation
def net_profit : ℚ :=
  candy_bars * sell_price - (candy_bars * buy_price + transport_cost)

-- Theorem statement
theorem scout_troop_profit :
  net_profit = 350 := by
  sorry

end scout_troop_profit_l1479_147922


namespace sphere_radius_ratio_l1479_147943

theorem sphere_radius_ratio (V_large : ℝ) (V_small : ℝ) :
  V_large = 288 * Real.pi →
  V_small = 0.125 * V_large →
  ∃ (r_large r_small : ℝ),
    V_large = (4 / 3) * Real.pi * r_large^3 ∧
    V_small = (4 / 3) * Real.pi * r_small^3 ∧
    r_small / r_large = 1 / 2 :=
by sorry

end sphere_radius_ratio_l1479_147943


namespace older_brother_stamps_l1479_147944

theorem older_brother_stamps : 
  ∀ (younger older : ℕ), 
  younger + older = 25 → 
  older = 2 * younger + 1 → 
  older = 17 := by sorry

end older_brother_stamps_l1479_147944


namespace function_property_l1479_147935

def is_symmetric_about_one (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1) = f (1 - x)

def is_increasing_on_right_of_one (f : ℝ → ℝ) : Prop :=
  ∀ x y, 1 ≤ x → x < y → f x < f y

def satisfies_inequality (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc (1/2) 1 → f (a * x + 2) ≤ f (x - 1)

theorem function_property (f : ℝ → ℝ) (h1 : is_symmetric_about_one f)
    (h2 : is_increasing_on_right_of_one f) :
    {a : ℝ | satisfies_inequality f a} = Set.Icc (-2) 0 := by
  sorry

end function_property_l1479_147935


namespace vasya_irrational_sequence_l1479_147931

theorem vasya_irrational_sequence (r : ℚ) (hr : 0 < r) : 
  ∃ n : ℕ, ¬ (∃ q : ℚ, q = (λ x => Real.sqrt (x + 1))^[n] r) :=
sorry

end vasya_irrational_sequence_l1479_147931


namespace min_value_theorem_l1479_147956

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 3 * x + 2 * y = 1 → 2 / x + 3 / y ≥ 2 / a + 3 / b) →
  2 / a + 3 / b = 24 :=
by sorry

end min_value_theorem_l1479_147956


namespace cone_generatrix_property_cylinder_generatrix_parallel_l1479_147950

-- Define the necessary geometric objects
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Cone where
  vertex : Point3D
  base_center : Point3D
  base_radius : ℝ

structure Cylinder where
  base_center : Point3D
  height : ℝ
  radius : ℝ

-- Define what a generatrix is for a cone and a cylinder
def is_generatrix_of_cone (l : Set Point3D) (c : Cone) : Prop :=
  ∃ p : Point3D, p ∈ l ∧ 
    (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 = c.base_radius^2 ∧
    p.z = c.base_center.z ∧
    c.vertex ∈ l

def are_parallel (l1 l2 : Set Point3D) : Prop :=
  ∃ v : Point3D, ∀ p q : Point3D, p ∈ l1 ∧ q ∈ l2 → 
    ∃ t : ℝ, q.x - p.x = t * v.x ∧ q.y - p.y = t * v.y ∧ q.z - p.z = t * v.z

def is_generatrix_of_cylinder (l : Set Point3D) (c : Cylinder) : Prop :=
  ∃ p q : Point3D, p ∈ l ∧ q ∈ l ∧
    (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 = c.radius^2 ∧
    p.z = c.base_center.z ∧
    (q.x - c.base_center.x)^2 + (q.y - c.base_center.y)^2 = c.radius^2 ∧
    q.z = c.base_center.z + c.height

-- State the theorems to be proved
theorem cone_generatrix_property (c : Cone) (p : Point3D) :
  (p.x - c.base_center.x)^2 + (p.y - c.base_center.y)^2 = c.base_radius^2 ∧ 
  p.z = c.base_center.z →
  is_generatrix_of_cone {q : Point3D | ∃ t : ℝ, q = Point3D.mk 
    (c.vertex.x + t * (p.x - c.vertex.x))
    (c.vertex.y + t * (p.y - c.vertex.y))
    (c.vertex.z + t * (p.z - c.vertex.z))} c :=
sorry

theorem cylinder_generatrix_parallel (c : Cylinder) (l1 l2 : Set Point3D) :
  is_generatrix_of_cylinder l1 c ∧ is_generatrix_of_cylinder l2 c →
  are_parallel l1 l2 :=
sorry

end cone_generatrix_property_cylinder_generatrix_parallel_l1479_147950


namespace c_finishes_in_60_days_l1479_147948

/-- The number of days it takes for worker c to finish the job alone, given:
  * Workers a and b together finish the job in 15 days
  * Workers a, b, and c together finish the job in 12 days
-/
def days_for_c_alone : ℚ :=
  let rate_ab : ℚ := 1 / 15  -- Combined rate of a and b
  let rate_abc : ℚ := 1 / 12 -- Combined rate of a, b, and c
  let rate_c : ℚ := rate_abc - rate_ab -- Rate of c alone
  1 / rate_c -- Days for c to finish the job

/-- Theorem stating that worker c alone can finish the job in 60 days -/
theorem c_finishes_in_60_days : days_for_c_alone = 60 := by
  sorry


end c_finishes_in_60_days_l1479_147948


namespace fraction_value_l1479_147964

theorem fraction_value : (3020 - 2931)^2 / 121 = 64 := by
  sorry

end fraction_value_l1479_147964


namespace imaginary_part_of_z_l1479_147963

theorem imaginary_part_of_z (i : ℂ) (h : i^2 = -1) : 
  (i^2 * (1 + i)).im = -1 := by sorry

end imaginary_part_of_z_l1479_147963


namespace solve_system_l1479_147936

theorem solve_system (y z x : ℚ) 
  (h1 : (2 : ℚ) / 3 = y / 90)
  (h2 : (2 : ℚ) / 3 = (y + z) / 120)
  (h3 : (2 : ℚ) / 3 = (x - z) / 150) : 
  x = 120 := by sorry

end solve_system_l1479_147936


namespace stacys_farm_goats_l1479_147970

/-- Calculates the number of goats on Stacy's farm given the conditions --/
theorem stacys_farm_goats (chickens : ℕ) (piglets : ℕ) (sick_animals : ℕ) :
  chickens = 26 →
  piglets = 40 →
  sick_animals = 50 →
  (chickens + piglets + (34 : ℕ)) / 2 = sick_animals →
  34 = (2 * sick_animals) - chickens - piglets :=
by
  sorry

end stacys_farm_goats_l1479_147970


namespace mowing_time_calculation_l1479_147997

/-- Represents the dimensions of a rectangular section of the lawn -/
structure LawnSection where
  length : ℝ
  width : ℝ

/-- Represents the mower specifications -/
structure Mower where
  swath_width : ℝ
  overlap : ℝ

/-- Calculates the time required to mow an L-shaped lawn -/
def mowing_time (section1 : LawnSection) (section2 : LawnSection) (mower : Mower) (walking_rate : ℝ) : ℝ :=
  sorry

/-- Theorem stating the time required to mow the lawn -/
theorem mowing_time_calculation :
  let section1 : LawnSection := { length := 120, width := 50 }
  let section2 : LawnSection := { length := 70, width := 50 }
  let mower : Mower := { swath_width := 35 / 12, overlap := 5 / 12 }
  let walking_rate : ℝ := 4000
  mowing_time section1 section2 mower walking_rate = 0.95 :=
by sorry

end mowing_time_calculation_l1479_147997


namespace rectangles_in_grid_l1479_147975

/-- The number of different rectangles in a 3x5 grid -/
def num_rectangles : ℕ := 30

/-- The number of rows in the grid -/
def num_rows : ℕ := 3

/-- The number of columns in the grid -/
def num_cols : ℕ := 5

/-- Theorem stating that the number of rectangles in a 3x5 grid is 30 -/
theorem rectangles_in_grid :
  num_rectangles = (num_rows.choose 2) * (num_cols.choose 2) := by
  sorry

#eval num_rectangles -- This should output 30

end rectangles_in_grid_l1479_147975


namespace intersection_distance_l1479_147929

/-- The distance between the intersection points of the line x - y + 1 = 0 and the circle x² + y² = 2 is equal to √6. -/
theorem intersection_distance : 
  ∃ (A B : ℝ × ℝ), 
    (A.1 - A.2 + 1 = 0) ∧ (A.1^2 + A.2^2 = 2) ∧
    (B.1 - B.2 + 1 = 0) ∧ (B.1^2 + B.2^2 = 2) ∧
    A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 6 :=
by sorry

end intersection_distance_l1479_147929


namespace quadratic_minimum_value_l1479_147977

def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem quadratic_minimum_value :
  ∀ x : ℝ, f x ≥ 1 ∧ ∃ x₀ : ℝ, f x₀ = 1 :=
by sorry

end quadratic_minimum_value_l1479_147977


namespace arithmetic_sequence_fifth_term_l1479_147996

/-- Given an arithmetic sequence {a_n} with a_1 = 2 and common difference d = 3,
    prove that the fifth term a_5 equals 14. -/
theorem arithmetic_sequence_fifth_term (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 3) →  -- Common difference is 3
  a 1 = 2 →                    -- First term is 2
  a 5 = 14 :=                  -- Fifth term is 14
by sorry

end arithmetic_sequence_fifth_term_l1479_147996


namespace polynomial_division_theorem_l1479_147923

/-- Given that g(x) = ax^2 + bx + c divides f(x) = x^3 + px^2 + qx + r, 
    where a ≠ 0, b ≠ 0, c ≠ 0, prove that (ap - b) / a = (aq - c) / b = ar / c -/
theorem polynomial_division_theorem 
  (a b c p q r : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h_divides : ∃ k, x^3 + p*x^2 + q*x + r = (a*x^2 + b*x + c) * k) : 
  (a*p - b) / a = (a*q - c) / b ∧ (a*q - c) / b = a*r / c := by
  sorry

end polynomial_division_theorem_l1479_147923


namespace vasyas_birthday_l1479_147974

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def two_days_after (d : DayOfWeek) : DayOfWeek :=
  next_day (next_day d)

theorem vasyas_birthday (birthday : DayOfWeek) 
  (h1 : next_day birthday ≠ DayOfWeek.Sunday)
  (h2 : two_days_after (next_day birthday) = DayOfWeek.Sunday) :
  birthday = DayOfWeek.Thursday := by
  sorry

end vasyas_birthday_l1479_147974


namespace bears_in_shipment_bears_shipment_proof_l1479_147955

/-- The number of bears in a toy store shipment -/
theorem bears_in_shipment (initial_stock : ℕ) (shelves : ℕ) (bears_per_shelf : ℕ) : ℕ :=
  shelves * bears_per_shelf - initial_stock

/-- Proof that the number of bears in the shipment is 7 -/
theorem bears_shipment_proof :
  bears_in_shipment 5 2 6 = 7 := by
  sorry

end bears_in_shipment_bears_shipment_proof_l1479_147955


namespace sum_of_coefficients_l1479_147917

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (1 - 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + a₂ + a₃ + a₄ + a₅ = -2 := by
sorry

end sum_of_coefficients_l1479_147917


namespace fraction_sum_equality_l1479_147927

theorem fraction_sum_equality (a b : ℝ) (ha : a ≠ 0) :
  1 / (2 * a * b) + b / (4 * a) = (2 + b^2) / (4 * a * b) := by
  sorry

end fraction_sum_equality_l1479_147927


namespace money_distribution_l1479_147901

theorem money_distribution (a b c total : ℕ) : 
  (a + b + c = total) →
  (2 * b = 3 * a) →
  (4 * b = 3 * c) →
  (b = 1500) →
  (total = 4500) := by
sorry

end money_distribution_l1479_147901


namespace badminton_players_l1479_147980

/-- A sports club with members playing badminton and tennis -/
structure SportsClub where
  total_members : ℕ
  tennis_players : ℕ
  both_players : ℕ
  neither_players : ℕ

/-- Theorem stating the number of badminton players in the sports club -/
theorem badminton_players (club : SportsClub)
  (h1 : club.total_members = 30)
  (h2 : club.tennis_players = 19)
  (h3 : club.both_players = 9)
  (h4 : club.neither_players = 2) :
  club.total_members - club.tennis_players + club.both_players - club.neither_players = 18 :=
by sorry

end badminton_players_l1479_147980


namespace min_value_theorem_l1479_147988

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x + 3 * y = 4) :
  (∃ (m : ℝ), m = 3/8 + Real.sqrt 2/4 ∧
    ∀ (z : ℝ), z = 1 / (2 * x + 1) + 1 / (3 * y + 2) → z ≥ m) :=
by sorry

end min_value_theorem_l1479_147988


namespace smallest_positive_integer_with_given_remainders_l1479_147907

theorem smallest_positive_integer_with_given_remainders :
  ∃ n : ℕ, n > 0 ∧
    n % 3 = 1 ∧
    n % 4 = 2 ∧
    n % 5 = 3 ∧
    ∀ m : ℕ, m > 0 →
      m % 3 = 1 →
      m % 4 = 2 →
      m % 5 = 3 →
      n ≤ m :=
by
  use 58
  sorry

end smallest_positive_integer_with_given_remainders_l1479_147907


namespace complex_number_quadrant_l1479_147920

theorem complex_number_quadrant (z : ℂ) (m : ℝ) :
  z * Complex.I = Complex.I + m →
  z.im = 1 →
  z.re > 0 := by
  sorry

end complex_number_quadrant_l1479_147920


namespace flower_bed_length_l1479_147946

/-- A rectangular flower bed with given area and width has a specific length -/
theorem flower_bed_length (area width : ℝ) (h1 : area = 35) (h2 : width = 5) :
  area / width = 7 := by
  sorry

end flower_bed_length_l1479_147946
