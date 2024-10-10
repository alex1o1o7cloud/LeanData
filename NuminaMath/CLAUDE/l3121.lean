import Mathlib

namespace stacy_heather_walk_l3121_312119

/-- The problem of determining the time difference between Stacy and Heather's start times -/
theorem stacy_heather_walk (total_distance : ℝ) (heather_speed : ℝ) (stacy_speed : ℝ) 
  (heather_distance : ℝ) :
  total_distance = 25 →
  heather_speed = 5 →
  stacy_speed = heather_speed + 1 →
  heather_distance = 10.272727272727273 →
  ∃ (time_diff : ℝ), 
    time_diff * 60 = 24 ∧ 
    heather_distance / heather_speed = 
      (total_distance - heather_distance) / stacy_speed - time_diff :=
by sorry


end stacy_heather_walk_l3121_312119


namespace parallel_vectors_sum_l3121_312133

/-- Given two vectors a and b in ℝ³, where a is parallel to b, prove that m + n = 4 -/
theorem parallel_vectors_sum (a b : ℝ × ℝ × ℝ) (m n : ℝ) : 
  a = (2, -1, 3) → b = (4, m, n) → (∃ (k : ℝ), a = k • b) → m + n = 4 := by
  sorry

end parallel_vectors_sum_l3121_312133


namespace min_value_of_sum_roots_l3121_312186

theorem min_value_of_sum_roots (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hsum : a + b + c + d = 1) : 
  Real.sqrt (a^2 + 1/(8*a)) + Real.sqrt (b^2 + 1/(8*b)) + 
  Real.sqrt (c^2 + 1/(8*c)) + Real.sqrt (d^2 + 1/(8*d)) ≥ 3 := by
sorry

end min_value_of_sum_roots_l3121_312186


namespace marilyn_bottle_caps_l3121_312110

/-- Given that Marilyn starts with 51 bottle caps and shares 36 with Nancy, 
    prove that she ends up with 15 bottle caps. -/
theorem marilyn_bottle_caps 
  (start : ℕ) 
  (shared : ℕ) 
  (h1 : start = 51) 
  (h2 : shared = 36) : 
  start - shared = 15 := by
sorry

end marilyn_bottle_caps_l3121_312110


namespace magnitude_of_complex_product_l3121_312141

theorem magnitude_of_complex_product : 
  Complex.abs ((7 - 4*I) * (3 + 10*I)) = Real.sqrt 7085 := by
  sorry

end magnitude_of_complex_product_l3121_312141


namespace salary_problem_l3121_312125

/-- Salary problem -/
theorem salary_problem 
  (jan feb mar apr may : ℕ)  -- Salaries for each month
  (h1 : (jan + feb + mar + apr) / 4 = 8000)  -- Average for Jan-Apr
  (h2 : (feb + mar + apr + may) / 4 = 8800)  -- Average for Feb-May
  (h3 : may = 6500)  -- May's salary
  : jan = 3300 := by
sorry

end salary_problem_l3121_312125


namespace min_value_inequality_l3121_312194

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  Real.sqrt ((x^2 + y^2) * (5 * x^2 + y^2)) / (x * y) ≥ Real.sqrt 5 + 1 := by
  sorry

end min_value_inequality_l3121_312194


namespace parallelogram_base_length_l3121_312193

/-- For a parallelogram with given height and area, prove that its base length is as calculated. -/
theorem parallelogram_base_length (height area : ℝ) (h_height : height = 11) (h_area : area = 44) :
  area / height = 4 := by
  sorry

end parallelogram_base_length_l3121_312193


namespace partial_fraction_decomposition_l3121_312191

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 6 ∧ x ≠ -3 →
    5 * x - 3 = (x^2 - 3*x - 18) * (C / (x - 6) + D / (x + 3))) →
  C = 3 ∧ D = 2 := by
sorry

end partial_fraction_decomposition_l3121_312191


namespace max_intersections_sine_line_l3121_312154

theorem max_intersections_sine_line (φ : ℝ) : 
  ∃ (n : ℕ), n ≤ 4 ∧ 
  (∀ (m : ℕ), (∃ (S : Finset ℝ), S.card = m ∧ 
    (∀ x ∈ S, x ∈ Set.Icc 0 Real.pi ∧ 3 * Real.sin (3 * x + φ) = 2)) → m ≤ n) :=
sorry

end max_intersections_sine_line_l3121_312154


namespace volume_difference_l3121_312152

/-- The volume of space inside a sphere and outside a combined cylinder and cone -/
theorem volume_difference (r_sphere : ℝ) (r_base : ℝ) (h_cylinder : ℝ) (h_cone : ℝ) 
  (hr_sphere : r_sphere = 6)
  (hr_base : r_base = 4)
  (hh_cylinder : h_cylinder = 10)
  (hh_cone : h_cone = 5) :
  (4 / 3 * π * r_sphere^3) - (π * r_base^2 * h_cylinder + 1 / 3 * π * r_base^2 * h_cone) = 304 / 3 * π :=
sorry

end volume_difference_l3121_312152


namespace yoo_jeong_borrowed_nine_notebooks_l3121_312159

/-- The number of notebooks Min-young originally had -/
def original_notebooks : ℕ := 17

/-- The number of notebooks Min-young had left after lending -/
def remaining_notebooks : ℕ := 8

/-- The number of notebooks Yoo-jeong borrowed -/
def borrowed_notebooks : ℕ := original_notebooks - remaining_notebooks

theorem yoo_jeong_borrowed_nine_notebooks : borrowed_notebooks = 9 := by
  sorry

end yoo_jeong_borrowed_nine_notebooks_l3121_312159


namespace digit_150_is_5_l3121_312115

/-- The decimal expansion of 7/29 -/
def decimal_expansion : List Nat := [2, 4, 1, 3, 7, 9, 3, 1, 0, 3, 4, 4, 8, 2, 7, 5, 8, 6, 2, 0, 6, 8, 9, 6, 5, 5, 1, 7]

/-- The length of the repeating block in the decimal expansion of 7/29 -/
def repeat_length : Nat := decimal_expansion.length

/-- The 150th digit after the decimal point in the decimal expansion of 7/29 -/
def digit_150 : Nat := decimal_expansion[(150 - 1) % repeat_length]

theorem digit_150_is_5 : digit_150 = 5 := by sorry

end digit_150_is_5_l3121_312115


namespace model1_best_fit_l3121_312146

-- Define the coefficient of determination for each model
def R2_model1 : ℝ := 0.98
def R2_model2 : ℝ := 0.80
def R2_model3 : ℝ := 0.50
def R2_model4 : ℝ := 0.25

-- Define a function to compare R² values
def better_fit (a b : ℝ) : Prop := a > b

-- Theorem stating that Model 1 has the best fitting effect
theorem model1_best_fit :
  better_fit R2_model1 R2_model2 ∧
  better_fit R2_model1 R2_model3 ∧
  better_fit R2_model1 R2_model4 :=
by sorry

end model1_best_fit_l3121_312146


namespace quotient_less_than_dividend_l3121_312187

theorem quotient_less_than_dividend : 
  let a := (5 : ℚ) / 7
  let b := (5 : ℚ) / 4
  a / b < a :=
by sorry

end quotient_less_than_dividend_l3121_312187


namespace derivative_f_at_3_l3121_312144

def f (x : ℝ) := x^2

theorem derivative_f_at_3 : 
  deriv f 3 = 6 := by
  sorry

end derivative_f_at_3_l3121_312144


namespace sum_of_prime_factors_1729728_l3121_312175

theorem sum_of_prime_factors_1729728 : 
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (1729728 + 1))) id) = 36 := by sorry

end sum_of_prime_factors_1729728_l3121_312175


namespace grandfathers_age_l3121_312148

/-- Given the conditions about a family's ages, prove the grandfather's age 5 years ago. -/
theorem grandfathers_age (father_age : ℕ) (h1 : father_age = 58) :
  ∃ (son_age grandfather_age : ℕ),
    father_age - son_age = son_age ∧ 
    (son_age - 5) * 2 = grandfather_age ∧
    grandfather_age = 48 :=
by sorry

end grandfathers_age_l3121_312148


namespace jesses_room_carpet_area_l3121_312182

theorem jesses_room_carpet_area :
  let room_length : ℝ := 12
  let room_width : ℝ := 8
  let room_area := room_length * room_width
  room_area = 96 := by sorry

end jesses_room_carpet_area_l3121_312182


namespace small_triangles_in_large_triangle_l3121_312113

theorem small_triangles_in_large_triangle :
  let large_side : ℝ := 15
  let small_side : ℝ := 3
  let area (side : ℝ) := (Real.sqrt 3 / 4) * side^2
  let num_small_triangles := (area large_side) / (area small_side)
  num_small_triangles = 25 := by sorry

end small_triangles_in_large_triangle_l3121_312113


namespace work_completion_time_l3121_312100

theorem work_completion_time 
  (work_rate_b : ℝ) 
  (work_rate_combined : ℝ) 
  (days_b : ℝ) 
  (days_combined : ℝ) :
  work_rate_b = 1 / days_b →
  work_rate_combined = 1 / days_combined →
  days_b = 6 →
  days_combined = 3.75 →
  work_rate_combined = work_rate_b + 1 / 10 :=
by
  sorry

end work_completion_time_l3121_312100


namespace james_works_six_hours_l3121_312130

-- Define the cleaning times and number of rooms
def num_bedrooms : ℕ := 3
def num_bathrooms : ℕ := 2
def bedroom_cleaning_time : ℕ := 20 -- in minutes

-- Define the relationships between cleaning times
def living_room_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time
def bathroom_cleaning_time : ℕ := 2 * living_room_cleaning_time
def house_cleaning_time : ℕ := num_bedrooms * bedroom_cleaning_time + living_room_cleaning_time + num_bathrooms * bathroom_cleaning_time
def outside_cleaning_time : ℕ := 2 * house_cleaning_time
def total_cleaning_time : ℕ := house_cleaning_time + outside_cleaning_time
def num_siblings : ℕ := 3

-- Define James' working time
def james_working_time : ℚ := (total_cleaning_time / num_siblings) / 60

-- Theorem statement
theorem james_works_six_hours : james_working_time = 6 := by
  sorry

end james_works_six_hours_l3121_312130


namespace origin_constructible_l3121_312107

-- Define the points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the condition that A is above and to the left of B
def A_above_left_of_B : Prop :=
  A.1 < B.1 ∧ A.2 > B.2

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Theorem stating that the origin can be constructed
theorem origin_constructible (h : A_above_left_of_B) :
  ∃ (construction : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ)), construction A B = O :=
sorry

end origin_constructible_l3121_312107


namespace quadratic_function_properties_l3121_312142

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2

-- Define the properties of function f
def is_quadratic (f : ℝ → ℝ) : Prop := ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

def is_odd_sum (f : ℝ → ℝ) : Prop := ∀ x, f x + g x = -(f (-x) + g (-x))

def has_equal_roots (f : ℝ → ℝ) : Prop := ∃ x : ℝ, f x = 3 * x + 2 ∧ 
  ∀ y : ℝ, f y = 3 * y + 2 → y = x

-- Main theorem
theorem quadratic_function_properties (f : ℝ → ℝ) 
  (h1 : is_quadratic f)
  (h2 : is_odd_sum f)
  (h3 : has_equal_roots f) :
  (∀ x, f x = -x^2 + 3*x + 2) ∧ 
  (∀ x, (3 - Real.sqrt 41) / 4 < x ∧ x < (3 + Real.sqrt 41) / 4 → f x > g x) ∧
  (∃ m n : ℝ, m = -1 ∧ n = 17/8 ∧ 
    (∀ x, f x ∈ Set.Icc (-2) (247/64) ↔ x ∈ Set.Icc m n)) :=
by sorry

end quadratic_function_properties_l3121_312142


namespace polynomial_problems_l3121_312138

theorem polynomial_problems :
  (∀ x y, ∃ k, (2 - b) * x^2 + (a + 3) * x + (-6) * y + 7 = k) →
  (a - b)^2 = 25 ∧
  (∀ x y, ∃ k, (-1 - n) * x^2 + (-m + 6) * x + (-18) * y + 5 = k) →
  n = -1 ∧ m = 6 :=
by sorry

end polynomial_problems_l3121_312138


namespace trigonometric_simplification_max_value_cosine_function_l3121_312116

open Real

theorem trigonometric_simplification (α : ℝ) :
  (sin (2 * π - α) * tan (π - α) * cos (-π + α)) / (sin (5 * π + α) * sin (π / 2 + α)) = tan α :=
sorry

theorem max_value_cosine_function :
  let f : ℝ → ℝ := λ x ↦ 2 * cos x - cos (2 * x)
  ∃ (max_value : ℝ), max_value = 3 / 2 ∧
    ∀ x, f x ≤ max_value ∧
    ∀ k : ℤ, f (π / 3 + 2 * π * ↑k) = max_value ∧ f (-π / 3 + 2 * π * ↑k) = max_value :=
sorry

end trigonometric_simplification_max_value_cosine_function_l3121_312116


namespace odd_square_plus_four_odd_l3121_312160

theorem odd_square_plus_four_odd (p q : ℕ) 
  (hp : Odd p) (hq : Odd q) (hp_pos : 0 < p) (hq_pos : 0 < q) : 
  Odd (p^2 + 4*q) := by
  sorry

end odd_square_plus_four_odd_l3121_312160


namespace discounted_price_theorem_l3121_312131

/-- The selling price of two discounted items -/
def discounted_price (a : ℝ) : ℝ :=
  let original_price := a
  let markup_percentage := 0.5
  let discount_percentage := 0.2
  let marked_up_price := original_price * (1 + markup_percentage)
  let discounted_price := marked_up_price * (1 - discount_percentage)
  2 * discounted_price

/-- Theorem stating that the discounted price of two items is 2.4 times the original price -/
theorem discounted_price_theorem (a : ℝ) : discounted_price a = 2.4 * a := by
  sorry

end discounted_price_theorem_l3121_312131


namespace bc_distances_l3121_312173

/-- Represents a circular road with four gas stations -/
structure CircularRoad where
  circumference : ℝ
  distAB : ℝ
  distAC : ℝ
  distCD : ℝ
  distDA : ℝ

/-- Theorem stating the possible distances between B and C -/
theorem bc_distances (road : CircularRoad)
  (h_circ : road.circumference = 100)
  (h_ab : road.distAB = 50)
  (h_ac : road.distAC = 40)
  (h_cd : road.distCD = 25)
  (h_da : road.distDA = 35) :
  ∃ (d1 d2 : ℝ), d1 = 10 ∧ d2 = 90 ∧
  (∀ (d : ℝ), (d = d1 ∨ d = d2) ↔ 
    (d = road.distAB - road.distAC ∨ 
     d = road.circumference - (road.distAB + road.distAC))) :=
by sorry

end bc_distances_l3121_312173


namespace rectangular_field_perimeter_l3121_312176

theorem rectangular_field_perimeter (length width : ℝ) (area perimeter : ℝ) : 
  width = 0.6 * length →
  area = length * width →
  area = 37500 →
  perimeter = 2 * (length + width) →
  perimeter = 800 := by
sorry

end rectangular_field_perimeter_l3121_312176


namespace remainder_9876543210_mod_101_l3121_312117

theorem remainder_9876543210_mod_101 : 9876543210 % 101 = 31 := by
  sorry

end remainder_9876543210_mod_101_l3121_312117


namespace exists_valid_classification_l3121_312132

/-- Represents a team of students -/
structure Team :=
  (members : Finset Nat)
  (size_eq_six : members.card = 6)

/-- Classification of teams as GOOD or OK -/
def TeamClassification := Team → Bool

/-- Partition of students into teams -/
structure Partition :=
  (teams : Finset Team)
  (covers_all_students : (teams.biUnion Team.members).card = 24)
  (team_count_eq_four : teams.card = 4)

/-- Counts the number of GOOD teams in a partition -/
def countGoodTeams (c : TeamClassification) (p : Partition) : Nat :=
  (p.teams.filter (λ t => c t)).card

/-- Theorem stating the existence of a valid team classification -/
theorem exists_valid_classification : ∃ (c : TeamClassification),
  (∀ (p : Partition), countGoodTeams c p = 3 ∨ countGoodTeams c p = 1) ∧
  (∃ (p1 p2 : Partition), countGoodTeams c p1 = 3 ∧ countGoodTeams c p2 = 1) := by
  sorry

end exists_valid_classification_l3121_312132


namespace daughters_work_time_l3121_312185

/-- Given a man can do a piece of work in 4 days, and together with his daughter
    they can do it in 3 days, prove that the daughter can do the work alone in 12 days. -/
theorem daughters_work_time (man_time : ℕ) (combined_time : ℕ) (daughter_time : ℕ) :
  man_time = 4 →
  combined_time = 3 →
  (1 : ℚ) / man_time + (1 : ℚ) / daughter_time = (1 : ℚ) / combined_time →
  daughter_time = 12 := by
  sorry

end daughters_work_time_l3121_312185


namespace smallest_number_l3121_312172

def number_set : Set ℤ := {-3, 2, -2, 0}

theorem smallest_number : ∀ x ∈ number_set, -3 ≤ x := by sorry

end smallest_number_l3121_312172


namespace fraction_equality_l3121_312157

theorem fraction_equality (a b : ℝ) (ha : a ≠ 0) (h : (a + 2*b) / a = 4) : 
  a / (b - a) = 2 := by
sorry

end fraction_equality_l3121_312157


namespace star_five_three_l3121_312196

-- Define the ※ operation
def star (a b : ℝ) : ℝ := b^2 + 1

-- Theorem statement
theorem star_five_three : star 5 3 = 10 := by
  sorry

end star_five_three_l3121_312196


namespace find_a_over_b_l3121_312190

theorem find_a_over_b (a b c d e f : ℝ) 
  (h1 : a * b * c / (d * e * f) = 0.1875)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) :
  a / b = 1 / 3 := by
sorry

end find_a_over_b_l3121_312190


namespace pen_retailer_profit_percentage_pen_retailer_profit_is_ten_percent_l3121_312198

/-- Calculates the profit percentage for a pen retailer. -/
theorem pen_retailer_profit_percentage 
  (num_pens_bought : ℕ) 
  (num_pens_price : ℕ) 
  (discount_percent : ℚ) : ℚ :=
  let cost_price := num_pens_price
  let market_price_per_pen := 1
  let selling_price_per_pen := market_price_per_pen * (1 - discount_percent / 100)
  let total_selling_price := num_pens_bought * selling_price_per_pen
  let profit := total_selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage

/-- Proves that the profit percentage is 10% for the given scenario. -/
theorem pen_retailer_profit_is_ten_percent :
  pen_retailer_profit_percentage 40 36 1 = 10 := by
  sorry

end pen_retailer_profit_percentage_pen_retailer_profit_is_ten_percent_l3121_312198


namespace line_symmetry_l3121_312137

/-- Given a point (x, y) on a line, returns the x-coordinate of its symmetric point with respect to x = 1 -/
def symmetric_x (x : ℝ) : ℝ := 2 - x

/-- The original line -/
def original_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

/-- The symmetric line -/
def symmetric_line (x y : ℝ) : Prop := x + 2*y - 3 = 0

/-- Theorem stating that the symmetric_line is indeed symmetric to the original_line with respect to x = 1 -/
theorem line_symmetry :
  ∀ x y : ℝ, original_line x y → symmetric_line (symmetric_x x) y :=
sorry

end line_symmetry_l3121_312137


namespace max_principals_is_five_l3121_312166

/-- Represents the duration of a principal's term in years -/
def term_length : ℕ := 3

/-- Represents the total period in years we're considering -/
def total_period : ℕ := 10

/-- Calculates the maximum number of principals that can serve during the total period -/
def max_principals : ℕ := 
  (total_period / term_length) + 
  (if total_period % term_length > 0 then 2 else 1)

/-- Theorem stating the maximum number of principals during the given period -/
theorem max_principals_is_five : max_principals = 5 := by
  sorry

end max_principals_is_five_l3121_312166


namespace sally_peaches_l3121_312103

/-- The number of peaches Sally picked from the orchard -/
def picked_peaches : ℕ := 42

/-- The total number of peaches at the stand after picking -/
def total_peaches : ℕ := 55

/-- The number of peaches Sally had before picking more -/
def initial_peaches : ℕ := total_peaches - picked_peaches

theorem sally_peaches : initial_peaches = 13 := by
  sorry

end sally_peaches_l3121_312103


namespace librarians_work_schedule_l3121_312171

theorem librarians_work_schedule : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 14)) = 280 := by
  sorry

end librarians_work_schedule_l3121_312171


namespace expression_evaluation_l3121_312195

theorem expression_evaluation : (20 + 16 * 20) / (20 * 16) = 17 / 16 := by
  sorry

end expression_evaluation_l3121_312195


namespace probability_theorem_l3121_312139

def shirts : ℕ := 6
def pants : ℕ := 7
def socks : ℕ := 8
def total_articles : ℕ := shirts + pants + socks
def selected_articles : ℕ := 4

def probability_two_shirts_one_pant_one_sock : ℚ :=
  (Nat.choose shirts 2 * Nat.choose pants 1 * Nat.choose socks 1) /
  Nat.choose total_articles selected_articles

theorem probability_theorem :
  probability_two_shirts_one_pant_one_sock = 40 / 285 := by
  sorry

end probability_theorem_l3121_312139


namespace linda_savings_l3121_312165

theorem linda_savings (savings : ℝ) : (1 / 4 : ℝ) * savings = 220 → savings = 880 := by
  sorry

end linda_savings_l3121_312165


namespace circle_area_increase_l3121_312120

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by
  sorry

end circle_area_increase_l3121_312120


namespace convex_pentagon_exists_l3121_312184

-- Define the points
variable (A₁ A₂ A₃ A₄ A₅ A₆ A₇ A₈ A₉ : ℝ × ℝ)

-- Define the square
def is_square (A₁ A₂ A₃ A₄ : ℝ × ℝ) : Prop := sorry

-- Define the convex quadrilateral
def is_convex_quadrilateral (A₅ A₆ A₇ A₈ : ℝ × ℝ) : Prop := sorry

-- Define the point inside the quadrilateral
def point_inside_quadrilateral (A₉ A₅ A₆ A₇ A₈ : ℝ × ℝ) : Prop := sorry

-- Define the non-collinearity condition
def no_three_collinear (points : List (ℝ × ℝ)) : Prop := sorry

-- Define a convex pentagon
def is_convex_pentagon (points : List (ℝ × ℝ)) : Prop := sorry

theorem convex_pentagon_exists 
  (h1 : is_square A₁ A₂ A₃ A₄)
  (h2 : is_convex_quadrilateral A₅ A₆ A₇ A₈)
  (h3 : point_inside_quadrilateral A₉ A₅ A₆ A₇ A₈)
  (h4 : no_three_collinear [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉]) :
  ∃ (points : List (ℝ × ℝ)), points.length = 5 ∧ 
    (∀ p ∈ points, p ∈ [A₁, A₂, A₃, A₄, A₅, A₆, A₇, A₈, A₉]) ∧ 
    is_convex_pentagon points :=
sorry

end convex_pentagon_exists_l3121_312184


namespace emma_age_theorem_l3121_312163

def emma_age_problem (emma_current_age : ℕ) (age_difference : ℕ) (sister_future_age : ℕ) : Prop :=
  let sister_current_age := emma_current_age + age_difference
  let years_passed := sister_future_age - sister_current_age
  emma_current_age + years_passed = 47

theorem emma_age_theorem :
  emma_age_problem 7 9 56 :=
by
  sorry

end emma_age_theorem_l3121_312163


namespace remainder_of_division_l3121_312167

theorem remainder_of_division (n : ℕ) : 
  (2^224 + 104) % (2^112 + 2^56 + 1) = 103 := by
sorry

end remainder_of_division_l3121_312167


namespace at_least_one_red_ball_probability_l3121_312181

theorem at_least_one_red_ball_probability 
  (prob_red_A : ℝ) 
  (prob_red_B : ℝ) 
  (h1 : prob_red_A = 1/3) 
  (h2 : prob_red_B = 1/2) :
  1 - (1 - prob_red_A) * (1 - prob_red_B) = 2/3 := by
sorry

end at_least_one_red_ball_probability_l3121_312181


namespace function_value_at_specific_point_l3121_312135

/-- Given a function f(x) = ax^3 + b*sin(x) + 4 where a and b are real numbers,
    and f(lg(log_2(10))) = 5, prove that f(lg(lg(2))) = 3 -/
theorem function_value_at_specific_point
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^3 + b * Real.sin x + 4)
  (h2 : f (Real.log 10 / Real.log 2) = 5) :
  f (Real.log (Real.log 2) / Real.log 10) = 3 := by
  sorry

end function_value_at_specific_point_l3121_312135


namespace inequality_proof_l3121_312199

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 3) :
  (a / (1 + b^2 * c)) + (b / (1 + c^2 * a)) + (c / (1 + a^2 * b)) ≥ 3/2 := by
  sorry

end inequality_proof_l3121_312199


namespace investment_ratio_is_two_thirds_l3121_312129

/-- Represents the investments and profit shares of three partners A, B, and C. -/
structure Partnership where
  b_investment : ℝ
  c_investment : ℝ
  total_profit : ℝ
  b_profit_share : ℝ

/-- Theorem stating that under given conditions, the ratio of B's investment to C's investment is 2:3 -/
theorem investment_ratio_is_two_thirds (p : Partnership) 
  (h1 : p.b_investment > 0)
  (h2 : p.c_investment > 0)
  (h3 : p.total_profit = 4400)
  (h4 : p.b_profit_share = 800)
  (h5 : 3 * p.b_investment = p.b_investment + p.c_investment) :
  p.b_investment / p.c_investment = 2 / 3 := by
  sorry

#check investment_ratio_is_two_thirds

end investment_ratio_is_two_thirds_l3121_312129


namespace problem_solution_l3121_312106

theorem problem_solution (x y z : ℚ) 
  (eq1 : 102 * x - 5 * y = 25)
  (eq2 : 3 * y - x = 10)
  (eq3 : z^2 = y - x) : 
  x = 125 / 301 ∧ 10 - x = 2885 / 301 := by
sorry

end problem_solution_l3121_312106


namespace volume_change_specific_l3121_312111

/-- Represents the change in volume of a rectangular parallelepiped -/
def volume_change (a b c : ℝ) (da db dc : ℝ) : ℝ :=
  b * c * da + a * c * db + a * b * dc

/-- Theorem stating the change in volume for specific dimensions and changes -/
theorem volume_change_specific :
  let a : ℝ := 8
  let b : ℝ := 6
  let c : ℝ := 3
  let da : ℝ := 0.1
  let db : ℝ := 0.05
  let dc : ℝ := -0.15
  volume_change a b c da db dc = -4.2 := by
  sorry

#eval volume_change 8 6 3 0.1 0.05 (-0.15)

end volume_change_specific_l3121_312111


namespace dvd_discount_amount_l3121_312164

/-- The discount on a pack of DVDs -/
def discount (original_price discounted_price : ℝ) : ℝ :=
  original_price - discounted_price

/-- Theorem: The discount on each pack of DVDs is 25 dollars -/
theorem dvd_discount_amount : discount 76 51 = 25 := by
  sorry

end dvd_discount_amount_l3121_312164


namespace max_term_binomial_expansion_max_term_value_max_term_specific_case_l3121_312136

theorem max_term_binomial_expansion (n : ℕ) (x : ℝ) (h : x > 0) :
  ∃ k : ℕ, k ≤ n ∧
  ∀ j : ℕ, j ≤ n →
    (n.choose k : ℝ) * x^k ≥ (n.choose j : ℝ) * x^j :=
by sorry

theorem max_term_value (n : ℕ) (x : ℝ) (h : x > 0) :
  let k := ⌊n * x / (1 + x)⌋ + 1
  ∃ m : ℕ, m ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose m : ℝ) * x^m ≥ (n.choose j : ℝ) * x^j ∧
    (m = k ∨ m = k - 1) :=
by sorry

theorem max_term_specific_case :
  let n : ℕ := 210
  let x : ℝ := Real.sqrt 13
  let k : ℕ := 165
  ∃ m : ℕ, m ≤ n ∧
    ∀ j : ℕ, j ≤ n →
      (n.choose m : ℝ) * x^m ≥ (n.choose j : ℝ) * x^j ∧
    m = k :=
by sorry

end max_term_binomial_expansion_max_term_value_max_term_specific_case_l3121_312136


namespace unique_base_representation_l3121_312156

/-- The fraction we're considering -/
def fraction : ℚ := 8 / 65

/-- The repeating digits in the base-k representation -/
def repeating_digits : List ℕ := [2, 4]

/-- 
Given a positive integer k, this function should return true if and only if
the base-k representation of the fraction is 0.24242424...
-/
def is_correct_representation (k : ℕ) : Prop :=
  k > 0 ∧ 
  fraction = (2 / k + 4 / k^2) / (1 - 1 / k^2)

/-- The theorem to be proved -/
theorem unique_base_representation : 
  ∃! k : ℕ, is_correct_representation k ∧ k = 18 :=
sorry

end unique_base_representation_l3121_312156


namespace gnomes_taken_is_40_percent_l3121_312179

/-- The percentage of gnomes taken by the forest owner in Ravenswood forest --/
def gnomes_taken_percentage (westerville_gnomes : ℕ) (ravenswood_multiplier : ℕ) (remaining_gnomes : ℕ) : ℚ :=
  100 - (remaining_gnomes : ℚ) / ((westerville_gnomes * ravenswood_multiplier) : ℚ) * 100

/-- Theorem stating that the percentage of gnomes taken is 40% --/
theorem gnomes_taken_is_40_percent :
  gnomes_taken_percentage 20 4 48 = 40 := by
  sorry

end gnomes_taken_is_40_percent_l3121_312179


namespace xy_value_l3121_312134

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^(Real.sqrt y) = 27) (h2 : (Real.sqrt x)^y = 9) : 
  x * y = 16 * Real.rpow 3 (1/4) := by
  sorry

end xy_value_l3121_312134


namespace reciprocal_pairs_l3121_312174

def are_reciprocals (a b : ℚ) : Prop := a * b = 1

theorem reciprocal_pairs :
  ¬(are_reciprocals 1 (-1)) ∧
  ¬(are_reciprocals (-1/3) 3) ∧
  are_reciprocals (-5) (-1/5) ∧
  ¬(are_reciprocals (-3) (|(-3)|)) :=
by sorry

end reciprocal_pairs_l3121_312174


namespace charity_ticket_sales_l3121_312149

theorem charity_ticket_sales (total_tickets : ℕ) (total_revenue : ℕ) 
  (h : total_tickets = 180 ∧ total_revenue = 2800) :
  ∃ (full_price : ℕ) (half_price_count : ℕ),
    full_price > 0 ∧
    half_price_count + (total_tickets - half_price_count) = total_tickets ∧
    half_price_count * (full_price / 2) + (total_tickets - half_price_count) * full_price = total_revenue ∧
    half_price_count = 328 := by
  sorry

end charity_ticket_sales_l3121_312149


namespace unknown_blanket_rate_l3121_312188

/-- Given the prices and quantities of blankets, proves that the unknown rate is 285 when the average price is 162. -/
theorem unknown_blanket_rate (price1 price2 avg_price : ℚ) (qty1 qty2 qty_unknown : ℕ) : 
  price1 = 100 →
  price2 = 150 →
  qty1 = 3 →
  qty2 = 5 →
  qty_unknown = 2 →
  avg_price = 162 →
  (qty1 * price1 + qty2 * price2 + qty_unknown * (avg_price * (qty1 + qty2 + qty_unknown) - qty1 * price1 - qty2 * price2) / qty_unknown) / (qty1 + qty2 + qty_unknown) = avg_price →
  (avg_price * (qty1 + qty2 + qty_unknown) - qty1 * price1 - qty2 * price2) / qty_unknown = 285 := by
  sorry

#check unknown_blanket_rate

end unknown_blanket_rate_l3121_312188


namespace inequality_1_solution_inequality_2_solution_l3121_312105

-- Define the solution sets
def solution_set_1 : Set ℝ := {x | -2 < x ∧ x < 1}
def solution_set_2 : Set ℝ := {x | x < 1 ∨ x > 3}

-- Theorem for the first inequality
theorem inequality_1_solution (x : ℝ) : 
  |2*x + 1| < 3 ↔ x ∈ solution_set_1 :=
sorry

-- Theorem for the second inequality
theorem inequality_2_solution (x : ℝ) :
  |x - 2| + |x - 3| > 3 ↔ x ∈ solution_set_2 :=
sorry

end inequality_1_solution_inequality_2_solution_l3121_312105


namespace tan_product_simplification_l3121_312143

theorem tan_product_simplification :
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 := by
  sorry

end tan_product_simplification_l3121_312143


namespace smallest_inverse_domain_l3121_312126

-- Define the function g
def g (x : ℝ) : ℝ := (x + 1)^2 - 6

-- State the theorem
theorem smallest_inverse_domain (d : ℝ) :
  (∀ x y, x ∈ Set.Ici d → y ∈ Set.Ici d → g x = g y → x = y) ∧ 
  (∀ d' < d, ∃ x y, x ∈ Set.Ici d' → y ∈ Set.Ici d' → x ≠ y ∧ g x = g y) ↔ 
  d = -1 :=
sorry

end smallest_inverse_domain_l3121_312126


namespace expansion_coefficient_l3121_312121

/-- The coefficient of x^2 in the expansion of (1-ax)(1+x)^5 -/
def coefficient_x_squared (a : ℝ) : ℝ := 10 - 5*a

theorem expansion_coefficient (a : ℝ) : coefficient_x_squared a = 5 → a = 1 := by
  sorry

end expansion_coefficient_l3121_312121


namespace aziz_parents_move_year_l3121_312112

/-- The year Aziz's parents moved to America -/
def year_parents_moved (current_year : ℕ) (aziz_age : ℕ) (years_before_birth : ℕ) : ℕ :=
  current_year - aziz_age - years_before_birth

/-- Proof that Aziz's parents moved to America in 1982 -/
theorem aziz_parents_move_year :
  year_parents_moved 2021 36 3 = 1982 := by
  sorry

end aziz_parents_move_year_l3121_312112


namespace power_three_mod_eight_l3121_312147

theorem power_three_mod_eight : 3^20 % 8 = 1 := by
  sorry

end power_three_mod_eight_l3121_312147


namespace max_sum_under_constraints_max_sum_achieved_l3121_312169

theorem max_sum_under_constraints (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 7/3 := by
sorry

theorem max_sum_achieved (ε : ℝ) (hε : ε > 0) : 
  ∃ (x y : ℝ), 4 * x + 3 * y ≤ 9 ∧ 2 * x + 4 * y ≤ 8 ∧ x + y > 7/3 - ε := by
sorry

end max_sum_under_constraints_max_sum_achieved_l3121_312169


namespace line_parallel_to_plane_l3121_312124

-- Define the types for line and plane
variable (m : Line) (α : Plane)

-- Define the property of having no common points
def noCommonPoints (l : Line) (p : Plane) : Prop := sorry

-- Define the property of being parallel
def isParallel (l : Line) (p : Plane) : Prop := sorry

-- Theorem statement
theorem line_parallel_to_plane (h : noCommonPoints m α) : isParallel m α := by sorry

end line_parallel_to_plane_l3121_312124


namespace restaurant_hamburgers_l3121_312161

/-- The number of hamburgers served by the restaurant -/
def hamburgers_served : ℕ := 3

/-- The number of hamburgers left over -/
def hamburgers_leftover : ℕ := 6

/-- The total number of hamburgers made by the restaurant -/
def total_hamburgers : ℕ := hamburgers_served + hamburgers_leftover

/-- Theorem stating that the total number of hamburgers is 9 -/
theorem restaurant_hamburgers : total_hamburgers = 9 := by
  sorry

end restaurant_hamburgers_l3121_312161


namespace quadratic_factorization_l3121_312150

theorem quadratic_factorization (c d : ℕ) (h1 : c > d) 
  (h2 : ∀ x : ℝ, x^2 - 18*x + 72 = (x - c)*(x - d)) : c - 2*d = 0 := by
  sorry

end quadratic_factorization_l3121_312150


namespace major_selection_ways_l3121_312180

-- Define the total number of majors
def total_majors : ℕ := 10

-- Define the number of majors to be chosen
def chosen_majors : ℕ := 3

-- Define the number of incompatible majors
def incompatible_majors : ℕ := 2

-- Theorem statement
theorem major_selection_ways :
  (total_majors.choose chosen_majors) - 
  ((total_majors - incompatible_majors).choose (chosen_majors - 1)) = 672 := by
  sorry

end major_selection_ways_l3121_312180


namespace mom_tshirt_count_l3121_312145

/-- The number of t-shirts in each package -/
def shirts_per_package : ℕ := 6

/-- The number of packages Mom buys -/
def packages_bought : ℕ := 71

/-- The total number of t-shirts Mom will have -/
def total_shirts : ℕ := shirts_per_package * packages_bought

theorem mom_tshirt_count : total_shirts = 426 := by
  sorry

end mom_tshirt_count_l3121_312145


namespace exists_nonparallel_quadrilateral_from_identical_triangles_l3121_312158

/-- A triangle in 2D space --/
structure Triangle :=
  (a b c : ℝ × ℝ)

/-- A quadrilateral in 2D space --/
structure Quadrilateral :=
  (a b c d : ℝ × ℝ)

/-- Check if two line segments are parallel --/
def are_parallel (p1 p2 q1 q2 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := q1
  let (x4, y4) := q2
  (x2 - x1) * (y4 - y3) = (y2 - y1) * (x4 - x3)

/-- Check if a quadrilateral has parallel sides --/
def has_parallel_sides (q : Quadrilateral) : Prop :=
  are_parallel q.a q.b q.c q.d ∨ are_parallel q.a q.d q.b q.c

/-- Check if a quadrilateral is convex --/
def is_convex (q : Quadrilateral) : Prop := sorry

/-- Function to construct a quadrilateral from four triangles --/
def construct_quadrilateral (t1 t2 t3 t4 : Triangle) : Quadrilateral := sorry

/-- Theorem: There exists a convex quadrilateral formed by four identical triangles that does not have parallel sides --/
theorem exists_nonparallel_quadrilateral_from_identical_triangles :
  ∃ (t : Triangle) (q : Quadrilateral),
    q = construct_quadrilateral t t t t ∧
    is_convex q ∧
    ¬has_parallel_sides q :=
sorry

end exists_nonparallel_quadrilateral_from_identical_triangles_l3121_312158


namespace division_equation_l3121_312153

theorem division_equation : (786 * 74) / 30 = 1938.8 := by
  sorry

end division_equation_l3121_312153


namespace selected_students_in_range_l3121_312123

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  totalPopulation : ℕ
  sampleSize : ℕ
  initialSelection : ℕ
  rangeStart : ℕ
  rangeEnd : ℕ

/-- Calculates the number of selected items in a given range for a systematic sample -/
def selectedInRange (s : SystematicSample) : ℕ :=
  sorry

/-- Theorem stating the number of selected students in the given range -/
theorem selected_students_in_range :
  let s : SystematicSample := {
    totalPopulation := 100,
    sampleSize := 25,
    initialSelection := 4,
    rangeStart := 46,
    rangeEnd := 78
  }
  selectedInRange s = 8 := by sorry

end selected_students_in_range_l3121_312123


namespace worksheet_grading_problem_l3121_312162

theorem worksheet_grading_problem 
  (problems_per_worksheet : ℕ)
  (worksheets_graded : ℕ)
  (remaining_problems : ℕ)
  (h1 : problems_per_worksheet = 4)
  (h2 : worksheets_graded = 5)
  (h3 : remaining_problems = 16) :
  worksheets_graded + (remaining_problems / problems_per_worksheet) = 9 := by
  sorry

end worksheet_grading_problem_l3121_312162


namespace reverse_digit_numbers_base_9_11_l3121_312114

def is_three_digit_base (n : ℕ) (base : ℕ) : Prop :=
  base ^ 2 ≤ n ∧ n < base ^ 3

def digits_base (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

def reverse_digits (l : List ℕ) : List ℕ :=
  sorry

theorem reverse_digit_numbers_base_9_11 :
  ∃! (S : Finset ℕ),
    (∀ n ∈ S,
      is_three_digit_base n 9 ∧
      is_three_digit_base n 11 ∧
      digits_base n 9 = reverse_digits (digits_base n 11)) ∧
    S.card = 2 ∧
    245 ∈ S ∧
    490 ∈ S :=
  sorry

end reverse_digit_numbers_base_9_11_l3121_312114


namespace curve_to_line_equation_l3121_312109

/-- Proves that the curve parameterized by (x,y) = (3t + 6, 5t - 8) 
    can be expressed as the line equation y = (5/3)x - 18 -/
theorem curve_to_line_equation : 
  ∀ (t x y : ℝ), x = 3*t + 6 ∧ y = 5*t - 8 → y = (5/3)*x - 18 := by
  sorry

end curve_to_line_equation_l3121_312109


namespace teacher_distribution_l3121_312108

/-- The number of ways to distribute teachers to schools -/
def distribute_teachers (total_teachers : ℕ) (female_teachers : ℕ) (schools : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute teachers to schools with constraints -/
def distribute_teachers_constrained (total_teachers : ℕ) (female_teachers : ℕ) (schools : ℕ) : ℕ :=
  sorry

theorem teacher_distribution :
  distribute_teachers 4 2 3 = 36 ∧
  distribute_teachers_constrained 4 2 3 = 30 :=
sorry

end teacher_distribution_l3121_312108


namespace optimal_usage_time_l3121_312122

/-- Profit function for the yacht (in ten thousand yuan) -/
def profit (x : ℕ+) : ℚ := -x^2 + 22*x - 49

/-- Average annual profit function -/
def avgProfit (x : ℕ+) : ℚ := profit x / x

/-- Theorem stating that 7 years maximizes the average annual profit -/
theorem optimal_usage_time :
  ∀ x : ℕ+, avgProfit 7 ≥ avgProfit x :=
sorry

end optimal_usage_time_l3121_312122


namespace complex_function_minimum_on_unit_circle_l3121_312170

theorem complex_function_minimum_on_unit_circle
  (a : ℝ) (ha : 0 < a ∧ a < 1)
  (f : ℂ → ℂ) (hf : ∀ z, f z = z^2 - z + a) :
  ∀ z : ℂ, 1 ≤ Complex.abs z →
  ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ Complex.abs (f z₀) ≤ Complex.abs (f z) :=
by sorry

end complex_function_minimum_on_unit_circle_l3121_312170


namespace solve_system_of_equations_l3121_312177

theorem solve_system_of_equations (a x y : ℚ) 
  (eq1 : a * x + y = 8)
  (eq2 : 3 * x - 4 * y = 5)
  (eq3 : 7 * x - 3 * y = 23) :
  a = 25 / 16 := by
  sorry

end solve_system_of_equations_l3121_312177


namespace circle_radius_l3121_312168

theorem circle_radius (r : ℝ) (h : r > 0) :
  π * r^2 + 2 * π * r = 100 * π → r = 10 := by
sorry

end circle_radius_l3121_312168


namespace container_capacity_l3121_312102

theorem container_capacity : ∀ (C : ℝ),
  (C > 0) →  -- Ensure the capacity is positive
  (0.35 * C + 48 = 0.75 * C) →
  C = 120 := by
sorry

end container_capacity_l3121_312102


namespace cost_equalization_l3121_312104

theorem cost_equalization (X Y Z : ℝ) (h : X < Y ∧ Y < Z) :
  let E := (X + Y + Z) / 3
  let payment_to_bernardo := E - X - (Z - Y) / 2
  let payment_to_carlos := (Z - Y) / 2
  (X + payment_to_bernardo + payment_to_carlos = E) ∧
  (Y - payment_to_bernardo = E) ∧
  (Z - payment_to_carlos = E) := by
sorry

end cost_equalization_l3121_312104


namespace quadratic_minimum_l3121_312151

-- Define the quadratic function
def f (x : ℝ) : ℝ := -2 * x^2 + 8 * x + 5

-- State the theorem
theorem quadratic_minimum :
  ∃ (x_min : ℝ), (∀ (x : ℝ), f x ≥ f x_min) ∧ (x_min = 2) ∧ (f x_min = 13) := by
  sorry

end quadratic_minimum_l3121_312151


namespace provisions_problem_l3121_312101

/-- The number of days the provisions last for the initial group -/
def initial_days : ℕ := 20

/-- The number of additional men that join the group -/
def additional_men : ℕ := 200

/-- The number of days the provisions last after the additional men join -/
def final_days : ℕ := 16

/-- The initial number of men in the group -/
def initial_men : ℕ := 800

theorem provisions_problem :
  initial_men * initial_days = (initial_men + additional_men) * final_days :=
by sorry

end provisions_problem_l3121_312101


namespace sleeping_bag_wholesale_cost_l3121_312127

theorem sleeping_bag_wholesale_cost :
  ∀ (wholesale_cost selling_price : ℝ),
    selling_price = wholesale_cost * 1.12 →
    selling_price = 28 →
    wholesale_cost = 25 := by sorry

end sleeping_bag_wholesale_cost_l3121_312127


namespace friendly_pair_solution_l3121_312140

/-- Definition of a friendly number pair -/
def is_friendly_pair (m n : ℚ) : Prop :=
  m / 2 + n / 4 = (m + n) / (2 + 4)

/-- Theorem: If (a, 3) is a friendly number pair, then a = -3/4 -/
theorem friendly_pair_solution (a : ℚ) :
  is_friendly_pair a 3 → a = -3/4 := by
  sorry

end friendly_pair_solution_l3121_312140


namespace nice_function_property_l3121_312192

def nice (f : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, f^[a] b = f (a + b - 1)

theorem nice_function_property (g : ℕ → ℕ) (A : ℕ) 
  (hg : nice g)
  (hA : g (A + 2018) = g A + 1)
  (hA1 : g (A + 1) ≠ g (A + 1 + 2017^2017)) :
  ∀ n < A, g n = n + 1 := by sorry

end nice_function_property_l3121_312192


namespace sum_reciprocals_lower_bound_l3121_312128

theorem sum_reciprocals_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a^2 + b^2 = 1) :
  4 ≤ (1/a + 1/b) ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀^2 + b₀^2 = 1 ∧ 1/a₀ + 1/b₀ = 4 := by
  sorry

end sum_reciprocals_lower_bound_l3121_312128


namespace marsha_pay_per_mile_l3121_312155

/-- Calculates the pay per mile for a delivery driver given their daily pay and distances driven --/
def pay_per_mile (daily_pay : ℚ) (first_distance second_distance : ℚ) : ℚ :=
  let third_distance := second_distance / 2
  let total_distance := first_distance + second_distance + third_distance
  daily_pay / total_distance

/-- Proves that Marsha's pay per mile is $2 given the specified conditions --/
theorem marsha_pay_per_mile :
  pay_per_mile 104 10 28 = 2 := by
  sorry

end marsha_pay_per_mile_l3121_312155


namespace A_power_50_l3121_312189

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 2; -16, -6]

theorem A_power_50 : A^50 = !![(-301 : ℤ), -100; 800, 299] := by sorry

end A_power_50_l3121_312189


namespace supplementary_angles_problem_l3121_312178

theorem supplementary_angles_problem (A B : Real) : 
  A + B = 180 →  -- angles A and B are supplementary
  A = 7 * B →    -- measure of angle A is 7 times angle B
  A = 157.5 :=   -- prove that measure of angle A is 157.5°
by
  sorry

end supplementary_angles_problem_l3121_312178


namespace pattern_boundary_length_l3121_312118

theorem pattern_boundary_length (square_area : ℝ) (num_points : ℕ) : square_area = 144 ∧ num_points = 4 →
  ∃ (boundary_length : ℝ), boundary_length = 18 * Real.pi + 36 := by
  sorry

end pattern_boundary_length_l3121_312118


namespace range_of_a_minus_b_and_a_div_b_l3121_312197

theorem range_of_a_minus_b_and_a_div_b (a b : ℝ) 
  (ha : 12 < a ∧ a < 60) (hb : 15 < b ∧ b < 36) : 
  (-24 < a - b ∧ a - b < 45) ∧ (1/3 < a/b ∧ a/b < 4) := by
  sorry

end range_of_a_minus_b_and_a_div_b_l3121_312197


namespace positive_sum_from_positive_difference_l3121_312183

theorem positive_sum_from_positive_difference (a b : ℝ) (h : a - |b| > 0) : b + a > 0 := by
  sorry

end positive_sum_from_positive_difference_l3121_312183
