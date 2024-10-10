import Mathlib

namespace distance_AB_is_250_l3311_331127

/-- The distance between two points A and B, where two people walk towards each other and meet under specific conditions. -/
def distance_AB : ℝ :=
  let first_meeting_distance := 100 -- meters from B
  let second_meeting_distance := 50 -- meters from A
  let total_distance := first_meeting_distance + second_meeting_distance + 100
  total_distance

/-- Theorem stating that the distance between points A and B is 250 meters. -/
theorem distance_AB_is_250 : distance_AB = 250 := by
  sorry

end distance_AB_is_250_l3311_331127


namespace employee_not_on_first_day_l3311_331154

def num_employees : ℕ := 6
def num_days : ℕ := 3
def employees_per_day : ℕ := 2

def probability_not_on_first_day : ℚ :=
  2 / 3

theorem employee_not_on_first_day :
  let total_arrangements := (num_employees.choose employees_per_day) * 
                            ((num_employees - employees_per_day).choose employees_per_day) * 
                            ((num_employees - 2 * employees_per_day).choose employees_per_day)
  let arrangements_without_A := (num_employees - 1).choose 1 * 
                                ((num_employees - employees_per_day).choose employees_per_day) * 
                                ((num_employees - 2 * employees_per_day).choose employees_per_day)
  (arrangements_without_A : ℚ) / total_arrangements = probability_not_on_first_day :=
sorry

end employee_not_on_first_day_l3311_331154


namespace inequality_solution_range_l3311_331172

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) → 
  (a < -2 ∨ a ≥ 6/5) :=
by sorry

end inequality_solution_range_l3311_331172


namespace second_car_speed_l3311_331136

/-- Given two cars on a circular track, prove the speed of the second car. -/
theorem second_car_speed
  (track_length : ℝ)
  (first_car_speed : ℝ)
  (total_time : ℝ)
  (h1 : track_length = 150)
  (h2 : first_car_speed = 60)
  (h3 : total_time = 2)
  (h4 : ∃ (second_car_speed : ℝ),
    (first_car_speed + second_car_speed) * total_time = 2 * track_length) :
  ∃ (second_car_speed : ℝ), second_car_speed = 90 :=
by
  sorry


end second_car_speed_l3311_331136


namespace quadratic_minimum_value_l3311_331134

/-- Given a quadratic function f(x) = ax^2 + bx + c that is always non-negative
    and a < b, prove that (3a-2b+c)/(b-a) ≥ 1 -/
theorem quadratic_minimum_value (a b c : ℝ) 
    (h1 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0)
    (h2 : a < b) : 
    (3*a - 2*b + c) / (b - a) ≥ 1 := by
  sorry

end quadratic_minimum_value_l3311_331134


namespace power_calculation_l3311_331159

theorem power_calculation : 8^15 / 64^7 * 16 = 512 := by
  sorry

end power_calculation_l3311_331159


namespace radical_equality_l3311_331123

theorem radical_equality (a b c : ℤ) :
  Real.sqrt (a + b / c) = a * Real.sqrt (b / c) ↔ c = b * (a^2 - 1) / a :=
by sorry

end radical_equality_l3311_331123


namespace triangle_side_b_value_l3311_331196

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if the area is 2√3, angle B = π/3, and a² + c² = 3ac, then b = 4 -/
theorem triangle_side_b_value (a b c : ℝ) (A B C : ℝ) :
  (1/2) * a * c * Real.sin B = 2 * Real.sqrt 3 →
  B = π / 3 →
  a^2 + c^2 = 3 * a * c →
  b = 4 := by
  sorry

end triangle_side_b_value_l3311_331196


namespace last_element_value_l3311_331165

/-- Represents a triangular number table -/
def TriangularTable (n : ℕ) : Type :=
  Fin n → Fin n → ℕ

/-- The first row of the table contains the first n positive integers -/
def FirstRowCorrect (t : TriangularTable 100) : Prop :=
  ∀ i : Fin 100, t 0 i = i.val + 1

/-- Each element (except in the first row) is the sum of two elements above it -/
def ElementSum (t : TriangularTable 100) : Prop :=
  ∀ (i : Fin 99) (j : Fin (99 - i.val)), 
    t (i + 1) j = t i j + t i (j + 1)

/-- The last row contains only one element -/
def LastRowSingleton (t : TriangularTable 100) : Prop :=
  ∀ j : Fin 100, j.val > 0 → t 99 j = 0

/-- The main theorem: given the conditions, the last element is 101 * 2^98 -/
theorem last_element_value (t : TriangularTable 100) 
  (h1 : FirstRowCorrect t) 
  (h2 : ElementSum t)
  (h3 : LastRowSingleton t) : 
  t 99 0 = 101 * 2^98 := by
  sorry

end last_element_value_l3311_331165


namespace gcd_364_154_l3311_331148

theorem gcd_364_154 : Nat.gcd 364 154 = 14 := by
  sorry

end gcd_364_154_l3311_331148


namespace prime_power_plus_three_l3311_331191

theorem prime_power_plus_three (P : ℕ) : 
  Prime P → Prime (P^6 + 3) → P^10 + 3 = 1027 := by sorry

end prime_power_plus_three_l3311_331191


namespace perfect_square_trinomial_m_values_l3311_331117

/-- A trinomial ax^2 + bx + c is a perfect square if and only if b^2 - 4ac = 0 -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  b^2 = 4*a*c

theorem perfect_square_trinomial_m_values :
  ∀ m : ℝ, (is_perfect_square_trinomial 1 (-2*(m+3)) 9) → (m = 0 ∨ m = -6) :=
by sorry

end perfect_square_trinomial_m_values_l3311_331117


namespace missing_number_proof_l3311_331102

theorem missing_number_proof : ∃ x : ℝ, 0.72 * x + 0.12 * 0.34 = 0.3504 :=
  by
  use 0.43
  sorry

end missing_number_proof_l3311_331102


namespace tangent_line_equation_monotonic_increase_condition_l3311_331197

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/x + a) * Real.log (1 + x)

-- Theorem for the tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  f (-1) 1 = 0 ∧ 
  (deriv (f (-1))) 1 = -Real.log 2 →
  (Real.log 2 * x + y - Real.log 2 = 0) ↔ 
  y = (deriv (f (-1))) 1 * (x - 1) + f (-1) 1 :=
sorry

-- Theorem for monotonic increase condition
theorem monotonic_increase_condition (a : ℝ) :
  Monotone (f a) ↔ a ≥ (1/2 : ℝ) :=
sorry

end tangent_line_equation_monotonic_increase_condition_l3311_331197


namespace age_difference_is_24_l3311_331139

/-- Proves that the age difference between Ana and Claudia is 24 years --/
theorem age_difference_is_24 (A C : ℕ) (n : ℕ) : 
  A = C + n →                 -- Ana is n years older than Claudia
  A - 3 = 6 * (C - 3) →       -- Three years ago, Ana was 6 times as old as Claudia
  A = C^3 →                   -- This year Ana's age is the cube of Claudia's age
  n = 24 := by
sorry

end age_difference_is_24_l3311_331139


namespace fraction_equivalence_l3311_331173

theorem fraction_equivalence : 
  ∀ x : ℝ, x ≠ 0 → (x / (740/999)) * (5/9) = x / 1.4814814814814814 := by
  sorry

end fraction_equivalence_l3311_331173


namespace correct_divisor_l3311_331112

/-- Represents a person with their age in years -/
structure Person where
  name : String
  age : Nat

/-- The divisor that gives Gokul's age when (Arun's age - 6) is divided by it -/
def divisor (arun gokul : Person) : Nat :=
  (arun.age - 6) / gokul.age

theorem correct_divisor (arun madan gokul : Person) : 
  arun.name = "Arun" → 
  arun.age = 60 →
  madan.name = "Madan" → 
  madan.age = 5 →
  gokul.name = "Gokul" →
  gokul.age = madan.age - 2 →
  divisor arun gokul = 18 := by
  sorry

#check correct_divisor

end correct_divisor_l3311_331112


namespace f_is_odd_l3311_331141

def f (x : ℝ) : ℝ := x^(1/3)

theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

end f_is_odd_l3311_331141


namespace no_twelve_parallelepipeds_l3311_331114

/-- A rectangular parallelepiped with edges parallel to coordinate axes -/
structure RectParallelepiped where
  xRange : Set ℝ
  yRange : Set ℝ
  zRange : Set ℝ

/-- Two parallelepipeds intersect if their projections on all axes intersect -/
def intersect (p q : RectParallelepiped) : Prop :=
  (p.xRange ∩ q.xRange).Nonempty ∧
  (p.yRange ∩ q.yRange).Nonempty ∧
  (p.zRange ∩ q.zRange).Nonempty

/-- The condition for intersection based on indices -/
def shouldIntersect (i j : Fin 12) : Prop :=
  i ≠ j + 1 ∧ i ≠ j - 1

/-- The main theorem stating that 12 such parallelepipeds cannot exist -/
theorem no_twelve_parallelepipeds :
  ¬ ∃ (ps : Fin 12 → RectParallelepiped),
    ∀ (i j : Fin 12), intersect (ps i) (ps j) ↔ shouldIntersect i j :=
sorry

end no_twelve_parallelepipeds_l3311_331114


namespace geometric_mean_of_3_and_12_l3311_331142

theorem geometric_mean_of_3_and_12 : 
  ∃ (x : ℝ), x > 0 ∧ x^2 = 3 * 12 ∧ x = 6 := by
  sorry

end geometric_mean_of_3_and_12_l3311_331142


namespace theater_popcorn_packages_l3311_331164

/-- The number of popcorn buckets needed by the theater -/
def total_buckets : ℕ := 426

/-- The number of buckets in each package -/
def buckets_per_package : ℕ := 8

/-- The minimum number of packages required -/
def min_packages : ℕ := 54

theorem theater_popcorn_packages :
  min_packages = (total_buckets + buckets_per_package - 1) / buckets_per_package :=
by sorry

end theater_popcorn_packages_l3311_331164


namespace line_parameterization_l3311_331174

/-- The line y = 5x - 7 is parameterized by (x, y) = (r, 2) + t(3, k). 
    This theorem proves that r = 9/5 and k = 15. -/
theorem line_parameterization (x y r k t : ℝ) : 
  y = 5 * x - 7 ∧ 
  x = r + 3 * t ∧ 
  y = 2 + k * t → 
  r = 9 / 5 ∧ k = 15 := by
sorry

end line_parameterization_l3311_331174


namespace faucet_drip_properties_l3311_331155

/-- Represents the volume of water dripped from a faucet -/
def water_volume (time_minutes : ℝ) : ℝ :=
  6 * time_minutes

theorem faucet_drip_properties :
  (∀ x : ℝ, water_volume x = 6 * x) ∧
  (water_volume 50 = 300) := by
  sorry

end faucet_drip_properties_l3311_331155


namespace max_value_x_plus_inverse_l3311_331189

theorem max_value_x_plus_inverse (x : ℝ) (h : 13 = x^2 + 1/x^2) :
  (∀ y : ℝ, y > 0 → 13 = y^2 + 1/y^2 → x + 1/x ≥ y + 1/y) ∧ x + 1/x = Real.sqrt 15 := by
  sorry

end max_value_x_plus_inverse_l3311_331189


namespace min_value_of_expression_l3311_331169

theorem min_value_of_expression (a b : ℕ) (ha : 0 < a ∧ a ≤ 5) (hb : 0 < b ∧ b ≤ 5) :
  ∀ x y : ℕ, (0 < x ∧ x ≤ 5) → (0 < y ∧ y ≤ 5) → 
  a^2 - a*b + 2*b ≤ x^2 - x*y + 2*y ∧ 
  a^2 - a*b + 2*b = 4 :=
by sorry

end min_value_of_expression_l3311_331169


namespace smallest_m_chess_tournament_l3311_331185

theorem smallest_m_chess_tournament : ∃ (m : ℕ), m > 0 ∧ 
  (∀ (k : ℕ), k > 0 → (
    (∃ (x : ℕ), x > 0 ∧
      (4 * k * (4 * k - 1)) / 2 = 11 * x ∧
      8 * x + 3 * x = (4 * k * (4 * k - 1)) / 2
    ) → k ≥ m
  )) ∧ 
  (∃ (x : ℕ), x > 0 ∧
    (4 * m * (4 * m - 1)) / 2 = 11 * x ∧
    8 * x + 3 * x = (4 * m * (4 * m - 1)) / 2
  ) ∧
  m = 6 := by
  sorry

end smallest_m_chess_tournament_l3311_331185


namespace additional_sugar_needed_l3311_331105

/-- The amount of sugar needed for a cake recipe -/
def recipe_sugar : ℕ := 14

/-- The amount of sugar already added to the cake -/
def sugar_added : ℕ := 2

/-- The additional amount of sugar needed -/
def additional_sugar : ℕ := recipe_sugar - sugar_added

theorem additional_sugar_needed : additional_sugar = 12 := by
  sorry

end additional_sugar_needed_l3311_331105


namespace arithmetic_mean_lower_bound_l3311_331119

theorem arithmetic_mean_lower_bound (a₁ a₂ a₃ : ℝ) 
  (h_positive : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0) 
  (h_sum : 2*a₁ + 3*a₂ + a₃ = 1) : 
  (1/(a₁ + a₂) + 1/(a₂ + a₃)) / 2 ≥ (3 + 2*Real.sqrt 2) / 2 :=
by sorry

end arithmetic_mean_lower_bound_l3311_331119


namespace hundredth_ring_squares_l3311_331167

/-- The number of unit squares in the nth ring around a 2x3 rectangle -/
def ring_squares (n : ℕ) : ℕ := 4 * n + 8

/-- Theorem: The 100th ring contains 408 unit squares -/
theorem hundredth_ring_squares :
  ring_squares 100 = 408 := by sorry

end hundredth_ring_squares_l3311_331167


namespace stating_football_league_equation_l3311_331113

/-- 
The number of matches in a football league where each pair of classes plays a match,
given the number of class teams.
-/
def number_of_matches (x : ℕ) : ℕ := x * (x - 1) / 2

/-- 
Theorem stating that for a football league with x class teams, where each pair plays a match,
and there are 15 matches in total, the equation relating x to the number of matches is correct.
-/
theorem football_league_equation (x : ℕ) : 
  (number_of_matches x = 15) ↔ (x * (x - 1) / 2 = 15) := by
  sorry

end stating_football_league_equation_l3311_331113


namespace square_diff_divided_by_three_l3311_331176

theorem square_diff_divided_by_three : (123^2 - 120^2) / 3 = 243 := by sorry

end square_diff_divided_by_three_l3311_331176


namespace language_knowledge_distribution_l3311_331150

/-- Given the distribution of language knowledge among students, prove that
    among those who know both German and French, more than 90% know English. -/
theorem language_knowledge_distribution (a b c d : ℝ) 
    (h1 : a + b ≥ 0.9 * (a + b + c + d))
    (h2 : a + c ≥ 0.9 * (a + b + c + d))
    (h3 : a ≥ 0) (h4 : b ≥ 0) (h5 : c ≥ 0) (h6 : d ≥ 0) : 
    a ≥ 9 * d := by
  sorry


end language_knowledge_distribution_l3311_331150


namespace power_of_power_l3311_331140

theorem power_of_power : (3^3)^2 = 729 := by
  sorry

end power_of_power_l3311_331140


namespace multiplication_addition_equality_l3311_331163

theorem multiplication_addition_equality : 12 * 24 + 36 * 12 = 720 := by
  sorry

end multiplication_addition_equality_l3311_331163


namespace rent_expenditure_l3311_331199

theorem rent_expenditure (x : ℝ) 
  (h1 : x + 0.7 * x + 32 = 100) : x = 40 := by
  sorry

end rent_expenditure_l3311_331199


namespace tangent_and_sin_cos_product_l3311_331184

theorem tangent_and_sin_cos_product (α : Real) 
  (h : Real.tan (π / 4 + α) = 3) : 
  Real.tan α = 1 / 2 ∧ Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end tangent_and_sin_cos_product_l3311_331184


namespace student_calculation_l3311_331143

theorem student_calculation (chosen_number : ℕ) : 
  chosen_number = 155 → 
  chosen_number * 2 - 200 = 110 :=
by
  sorry

end student_calculation_l3311_331143


namespace stock_price_calculation_abc_stock_price_l3311_331178

theorem stock_price_calculation (initial_price : ℝ) 
  (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let price_after_second_year := price_after_first_year * (1 - second_year_decrease)
  price_after_second_year

theorem abc_stock_price : 
  stock_price_calculation 100 0.5 0.3 = 105 := by
  sorry

end stock_price_calculation_abc_stock_price_l3311_331178


namespace store_clearance_sale_profit_store_profit_is_3000_l3311_331130

/-- Calculates the money left after a store's clearance sale and paying creditors -/
theorem store_clearance_sale_profit (total_items : ℕ) (original_price : ℝ) 
  (discount_percent : ℝ) (sold_percent : ℝ) (owed_to_creditors : ℝ) : ℝ :=
  let sale_price := original_price * (1 - discount_percent)
  let items_sold := total_items * sold_percent
  let total_revenue := items_sold * sale_price
  let money_left := total_revenue - owed_to_creditors
  money_left

/-- Proves that the store has $3000 left after the clearance sale and paying creditors -/
theorem store_profit_is_3000 :
  store_clearance_sale_profit 2000 50 0.8 0.9 15000 = 3000 := by
  sorry

end store_clearance_sale_profit_store_profit_is_3000_l3311_331130


namespace melanie_dimes_problem_l3311_331126

/-- Calculates the number of dimes Melanie's mother gave her -/
def dimes_from_mother (initial : ℕ) (given_to_dad : ℕ) (final : ℕ) : ℕ :=
  final - (initial - given_to_dad)

theorem melanie_dimes_problem :
  dimes_from_mother 8 7 5 = 4 := by
  sorry

end melanie_dimes_problem_l3311_331126


namespace candy_distribution_contradiction_l3311_331157

theorem candy_distribution_contradiction (N : ℕ) : 
  (∃ (x : ℕ), N = 2 * x) →
  (∃ (y : ℕ), N = 3 * y) →
  (∃ (z : ℕ), N / 3 = 2 * z + 3) →
  False :=
by sorry

end candy_distribution_contradiction_l3311_331157


namespace package_weight_ratio_l3311_331132

theorem package_weight_ratio (x y : ℝ) (h1 : x > y) (h2 : x > 0) (h3 : y > 0) 
  (h4 : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end package_weight_ratio_l3311_331132


namespace cos_135_degrees_l3311_331135

theorem cos_135_degrees :
  Real.cos (135 * π / 180) = -Real.sqrt 2 / 2 := by sorry

end cos_135_degrees_l3311_331135


namespace coefficient_of_negative_2pi_ab_squared_l3311_331180

/-- The coefficient of a monomial is the numerical factor that multiplies the variable part. -/
def coefficient (m : ℝ) (x : String) : ℝ := sorry

/-- A monomial is an algebraic expression consisting of a single term. -/
def is_monomial (x : String) : Prop := sorry

theorem coefficient_of_negative_2pi_ab_squared :
  is_monomial "-2πab²" → coefficient (-2 * Real.pi) "ab²" = -2 * Real.pi := by sorry

end coefficient_of_negative_2pi_ab_squared_l3311_331180


namespace senate_committee_arrangement_l3311_331181

/-- The number of ways to arrange senators around a circular table. -/
def arrange_senators (num_democrats : ℕ) (num_republicans : ℕ) : ℕ :=
  if num_democrats = num_republicans ∧ num_democrats > 0 then
    (num_democrats.factorial) * ((num_democrats - 1).factorial)
  else
    0

/-- Theorem: The number of ways to arrange 6 Democrats and 6 Republicans
    around a circular table, with Democrats and Republicans alternating,
    is equal to 86,400. -/
theorem senate_committee_arrangement :
  arrange_senators 6 6 = 86400 := by
  sorry

end senate_committee_arrangement_l3311_331181


namespace crafts_club_necklaces_l3311_331110

theorem crafts_club_necklaces 
  (members : ℕ) 
  (beads_per_necklace : ℕ) 
  (total_beads : ℕ) 
  (h1 : members = 9)
  (h2 : beads_per_necklace = 50)
  (h3 : total_beads = 900) :
  total_beads / beads_per_necklace / members = 2 := by
sorry

end crafts_club_necklaces_l3311_331110


namespace cycle_selling_price_l3311_331186

/-- Given a cycle bought for Rs. 930 and sold with a gain of 30.107526881720432%,
    prove that the selling price is Rs. 1210. -/
theorem cycle_selling_price (cost_price : ℝ) (gain_percentage : ℝ) (selling_price : ℝ) :
  cost_price = 930 →
  gain_percentage = 30.107526881720432 →
  selling_price = cost_price * (1 + gain_percentage / 100) →
  selling_price = 1210 :=
by sorry

end cycle_selling_price_l3311_331186


namespace triangle_shape_determination_l3311_331149

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of two sides and the included angle of a triangle -/
def ratio_two_sides_and_angle (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The ratios of the three angle bisectors of a triangle -/
def ratio_angle_bisectors (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The ratios of the three medians of a triangle -/
def ratio_medians (t : Triangle) : ℝ × ℝ × ℝ := sorry

/-- The ratio of the circumradius to the inradius of a triangle -/
def ratio_circumradius_to_inradius (t : Triangle) : ℝ := sorry

/-- Two angles of a triangle -/
def two_angles (t : Triangle) : ℝ × ℝ := sorry

/-- Two triangles are similar -/
def are_similar (t1 t2 : Triangle) : Prop := sorry

/-- The shape of a triangle is uniquely determined by a given property
    if any two triangles with the same property are similar -/
def uniquely_determines_shape (f : Triangle → α) : Prop :=
  ∀ t1 t2 : Triangle, f t1 = f t2 → are_similar t1 t2

theorem triangle_shape_determination :
  uniquely_determines_shape ratio_two_sides_and_angle ∧
  uniquely_determines_shape ratio_angle_bisectors ∧
  uniquely_determines_shape ratio_medians ∧
  ¬ uniquely_determines_shape ratio_circumradius_to_inradius ∧
  uniquely_determines_shape two_angles := by sorry

end triangle_shape_determination_l3311_331149


namespace inequality_holds_iff_a_in_range_l3311_331116

theorem inequality_holds_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, x ≤ 1 → (1 + 2^x + 4^x * a) / (a^2 - a + 1) > 0) ↔ a > -3/4 :=
by sorry

end inequality_holds_iff_a_in_range_l3311_331116


namespace height_of_C_l3311_331158

/-- Given three people A, B, and C with heights hA, hB, and hC respectively (in cm),
    prove that C's height is 143 cm under the following conditions:
    1. The average height of A, B, and C is 143 cm.
    2. A's height increased by 4.5 cm becomes the average height of B and C.
    3. B is 3 cm taller than C. -/
theorem height_of_C (hA hB hC : ℝ) : 
  (hA + hB + hC) / 3 = 143 →
  hA + 4.5 = (hB + hC) / 2 →
  hB = hC + 3 →
  hC = 143 := by sorry

end height_of_C_l3311_331158


namespace y_min_at_a_or_b_l3311_331104

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^2 * (x - b)^2

/-- Theorem stating that the minimum of y occurs at x = a or x = b -/
theorem y_min_at_a_or_b (a b : ℝ) :
  ∃ (x : ℝ), ∀ (z : ℝ), y z a b ≥ y x a b ∧ (x = a ∨ x = b) :=
sorry

end y_min_at_a_or_b_l3311_331104


namespace tax_free_items_cost_l3311_331187

/-- Calculates the cost of tax-free items given total spend, sales tax, and tax rate -/
theorem tax_free_items_cost 
  (total_spend : ℝ) 
  (sales_tax : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_spend = 25)
  (h2 : sales_tax = 0.30)
  (h3 : tax_rate = 0.05) :
  total_spend - sales_tax / tax_rate = 19 := by
  sorry

#check tax_free_items_cost

end tax_free_items_cost_l3311_331187


namespace sum_of_x_solutions_l3311_331107

theorem sum_of_x_solutions (y : ℝ) (x : ℝ → Prop) : 
  y = 5 → 
  (∀ x', x x' ↔ x'^2 + y^2 + 2*x' - 4*y = 80) → 
  (∃ a b, (x a ∧ x b) ∧ (∀ c, x c → (c = a ∨ c = b)) ∧ (a + b = -2)) :=
by sorry

end sum_of_x_solutions_l3311_331107


namespace nonreal_cubic_root_sum_l3311_331124

/-- Given ω is a nonreal cubic root of unity, 
    prove that (2 - ω + 2ω^2)^6 + (2 + ω - 2ω^2)^6 = 38908 -/
theorem nonreal_cubic_root_sum (ω : ℂ) : 
  ω^3 = 1 → ω ≠ (1 : ℂ) → (2 - ω + 2*ω^2)^6 + (2 + ω - 2*ω^2)^6 = 38908 := by
  sorry

end nonreal_cubic_root_sum_l3311_331124


namespace intersection_point_m_value_l3311_331198

theorem intersection_point_m_value (m : ℝ) :
  (∃ y : ℝ, -3 * (-6) + y = m ∧ 2 * (-6) + y = 28) →
  m = 58 := by
sorry

end intersection_point_m_value_l3311_331198


namespace equation_solutions_l3311_331133

theorem equation_solutions :
  ∀ a b : ℤ, 3 * a^2 * b^2 + b^2 = 517 + 30 * a^2 ↔ 
  ((a = 2 ∧ b = 7) ∨ (a = -2 ∧ b = 7) ∨ (a = 2 ∧ b = -7) ∨ (a = -2 ∧ b = -7)) :=
by sorry

end equation_solutions_l3311_331133


namespace elberta_has_41_l3311_331153

/-- The amount of money Granny Smith has -/
def granny_smith_amount : ℕ := 72

/-- The amount of money Anjou has -/
def anjou_amount : ℕ := granny_smith_amount / 4

/-- The amount of money Elberta has -/
def elberta_amount : ℕ := 2 * anjou_amount + 5

/-- Theorem stating that Elberta has $41 -/
theorem elberta_has_41 : elberta_amount = 41 := by
  sorry

end elberta_has_41_l3311_331153


namespace sin_sum_of_complex_exponentials_l3311_331137

theorem sin_sum_of_complex_exponentials (γ δ : ℝ) :
  Complex.exp (γ * Complex.I) = 4/5 + 3/5 * Complex.I →
  Complex.exp (δ * Complex.I) = -5/13 + 12/13 * Complex.I →
  Real.sin (γ + δ) = 33/65 := by
  sorry

end sin_sum_of_complex_exponentials_l3311_331137


namespace sum_of_coefficients_l3311_331118

theorem sum_of_coefficients (a b c d e : ℤ) : 
  (∀ x : ℚ, 1000 * x^3 + 27 = (a * x + b) * (c * x^2 + d * x + e)) →
  a + b + c + d + e = 92 := by
sorry

end sum_of_coefficients_l3311_331118


namespace total_remaining_apples_l3311_331175

def tree_A : ℕ := 200
def tree_B : ℕ := 250
def tree_C : ℕ := 300

def picked_A : ℕ := tree_A / 5
def picked_B : ℕ := 2 * picked_A
def picked_C : ℕ := picked_A + 20

def remaining_A : ℕ := tree_A - picked_A
def remaining_B : ℕ := tree_B - picked_B
def remaining_C : ℕ := tree_C - picked_C

theorem total_remaining_apples :
  remaining_A + remaining_B + remaining_C = 570 := by
  sorry

end total_remaining_apples_l3311_331175


namespace unique_solution_l3311_331179

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: 2001 is the only natural number n that satisfies n + S(n) = 2004 -/
theorem unique_solution : ∀ n : ℕ, n + S n = 2004 ↔ n = 2001 := by sorry

end unique_solution_l3311_331179


namespace largest_fraction_l3311_331125

theorem largest_fraction :
  let a := (2 : ℚ) / 5
  let b := (4 : ℚ) / 9
  let c := (7 : ℚ) / 15
  let d := (11 : ℚ) / 18
  let e := (16 : ℚ) / 35
  d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end largest_fraction_l3311_331125


namespace triangle_cosine_l3311_331147

theorem triangle_cosine (X Y Z : ℝ) (h1 : X + Y + Z = Real.pi) 
  (h2 : X = Real.pi / 2) (h3 : Y = Real.pi / 4) (h4 : Real.tan Z = 1 / 2) : 
  Real.cos Z = Real.sqrt 5 / 5 := by
  sorry

end triangle_cosine_l3311_331147


namespace fifth_term_of_geometric_sequence_l3311_331162

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_third_term : a 3 = 12 / 5)
  (h_seventh_term : a 7 = 48) :
  a 5 = 12 / 5 :=
sorry

end fifth_term_of_geometric_sequence_l3311_331162


namespace max_rooks_on_chessboard_l3311_331109

/-- Represents a chessboard --/
def Chessboard := Fin 8 → Fin 8 → Bool

/-- Checks if a rook at position (x, y) attacks an odd number of rooks on the board --/
def attacks_odd (board : Chessboard) (x y : Fin 8) : Bool :=
  sorry

/-- Returns the number of rooks on the board --/
def count_rooks (board : Chessboard) : Nat :=
  sorry

/-- Checks if a board configuration is valid according to the rules --/
def is_valid_configuration (board : Chessboard) : Prop :=
  sorry

theorem max_rooks_on_chessboard :
  ∃ (board : Chessboard),
    is_valid_configuration board ∧
    count_rooks board = 63 ∧
    ∀ (other_board : Chessboard),
      is_valid_configuration other_board →
      count_rooks other_board ≤ 63 :=
by sorry

end max_rooks_on_chessboard_l3311_331109


namespace molecular_weight_N2O5_is_108_l3311_331120

/-- The molecular weight of N2O5 in grams per mole. -/
def molecular_weight_N2O5 : ℝ := 108

/-- The number of moles used in the given condition. -/
def given_moles : ℝ := 10

/-- The total weight of the given number of moles in grams. -/
def given_total_weight : ℝ := 1080

/-- Theorem stating that the molecular weight of N2O5 is 108 grams/mole,
    given that 10 moles of N2O5 weigh 1080 grams. -/
theorem molecular_weight_N2O5_is_108 :
  molecular_weight_N2O5 = given_total_weight / given_moles :=
by sorry

end molecular_weight_N2O5_is_108_l3311_331120


namespace smallest_square_arrangement_l3311_331170

theorem smallest_square_arrangement : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → ¬ ∃ k : ℕ+, m * (1^2 + 2^2 + 3^2) = k^2) ∧
  (∃ k : ℕ+, n * (1^2 + 2^2 + 3^2) = k^2) :=
by sorry

end smallest_square_arrangement_l3311_331170


namespace sum_of_square_and_pentagon_angles_l3311_331166

theorem sum_of_square_and_pentagon_angles : 
  let square_angle := 180 * (4 - 2) / 4
  let pentagon_angle := 180 * (5 - 2) / 5
  square_angle + pentagon_angle = 198 := by
sorry

end sum_of_square_and_pentagon_angles_l3311_331166


namespace original_number_problem_l3311_331128

theorem original_number_problem (x : ℝ) : ((x + 5 - 2) / 4 = 7) → x = 25 := by
  sorry

end original_number_problem_l3311_331128


namespace class_trip_cost_l3311_331131

/-- Calculates the total cost of a class trip to a science museum --/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (student_ticket_price : ℚ) 
  (teacher_ticket_price : ℚ) (discount_rate : ℚ) (min_group_size : ℕ) 
  (bus_fee : ℚ) (meal_price : ℚ) : ℚ :=
  let total_people := num_students + num_teachers
  let ticket_cost := num_students * student_ticket_price + num_teachers * teacher_ticket_price
  let discounted_ticket_cost := 
    if total_people ≥ min_group_size 
    then ticket_cost * (1 - discount_rate) 
    else ticket_cost
  let meal_cost := meal_price * total_people
  discounted_ticket_cost + bus_fee + meal_cost

/-- Theorem stating the total cost for the class trip --/
theorem class_trip_cost : 
  total_cost 30 4 8 12 0.2 25 150 10 = 720.4 := by
  sorry

end class_trip_cost_l3311_331131


namespace scientific_notation_of_44300000_l3311_331129

theorem scientific_notation_of_44300000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 44300000 = a * (10 : ℝ) ^ n ∧ a = 4.43 ∧ n = 7 := by
  sorry

end scientific_notation_of_44300000_l3311_331129


namespace astronomers_use_analogical_reasoning_l3311_331115

/-- Represents a celestial body in the solar system -/
structure CelestialBody where
  name : String
  hasLife : Bool

/-- Represents a type of reasoning -/
inductive ReasoningType
  | Inductive
  | Analogical
  | Deductive
  | ProofByContradiction

/-- Determines if two celestial bodies are similar -/
def areSimilar (a b : CelestialBody) : Bool := sorry

/-- Represents the astronomers' reasoning process -/
def astronomersReasoning (earth mars : CelestialBody) : ReasoningType :=
  if areSimilar earth mars ∧ earth.hasLife then
    ReasoningType.Analogical
  else
    sorry

/-- Theorem stating that the astronomers' reasoning is analogical -/
theorem astronomers_use_analogical_reasoning (earth mars : CelestialBody) 
  (h1 : areSimilar earth mars = true)
  (h2 : earth.hasLife = true) :
  astronomersReasoning earth mars = ReasoningType.Analogical := by
  sorry

end astronomers_use_analogical_reasoning_l3311_331115


namespace sock_pair_probability_l3311_331106

def total_socks : ℕ := 40
def white_socks : ℕ := 10
def red_socks : ℕ := 12
def black_socks : ℕ := 18
def drawn_socks : ℕ := 3

theorem sock_pair_probability :
  let total_ways := Nat.choose total_socks drawn_socks
  let all_different := white_socks * red_socks * black_socks
  let at_least_one_pair := total_ways - all_different
  (at_least_one_pair : ℚ) / total_ways = 193 / 247 := by
  sorry

end sock_pair_probability_l3311_331106


namespace visited_neither_country_l3311_331195

theorem visited_neither_country (total : ℕ) (visited_iceland : ℕ) (visited_norway : ℕ) (visited_both : ℕ) :
  total = 90 →
  visited_iceland = 55 →
  visited_norway = 33 →
  visited_both = 51 →
  total - (visited_iceland + visited_norway - visited_both) = 53 := by
sorry

end visited_neither_country_l3311_331195


namespace not_adjacent_2010_2011_l3311_331145

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Checks if two natural numbers are consecutive -/
def are_consecutive (a b : ℕ) : Prop := b = a + 1

/-- Checks if a natural number is within a sequence of 100 consecutive numbers starting from start -/
def in_sequence (n start : ℕ) : Prop := start ≤ n ∧ n < start + 100

theorem not_adjacent_2010_2011 (start : ℕ) : 
  ¬(in_sequence 2010 start ∧ in_sequence 2011 start ∧
    (∀ (x y : ℕ), in_sequence x start → in_sequence y start →
      (digit_sum x < digit_sum y ∨ (digit_sum x = digit_sum y ∧ x < y)) →
      x < y) →
    are_consecutive 2010 2011) :=
sorry

end not_adjacent_2010_2011_l3311_331145


namespace pond_and_field_dimensions_l3311_331183

/-- Given a square field with a circular pond inside, this theorem proves
    the diameter of the pond and the side length of the field. -/
theorem pond_and_field_dimensions :
  ∀ (pond_diameter field_side : ℝ),
    pond_diameter > 0 →
    field_side > pond_diameter →
    (field_side^2 - (pond_diameter/2)^2 * 3) = 13.75 * 240 →
    field_side - pond_diameter = 40 →
    pond_diameter = 20 ∧ field_side = 60 := by
  sorry

end pond_and_field_dimensions_l3311_331183


namespace part_one_part_two_l3311_331177

-- Define the logarithmic function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem for part 1
theorem part_one (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 8 = 3) :
  a = 2 := by sorry

-- Theorem for part 2
theorem part_two (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → {x : ℝ | f a x ≤ f a (2 - 3*x)} = {x : ℝ | 0 < x ∧ x ≤ 1/2}) ∧
  (0 < a ∧ a < 1 → {x : ℝ | f a x ≤ f a (2 - 3*x)} = {x : ℝ | 1/2 ≤ x ∧ x < 2/3}) := by sorry

end part_one_part_two_l3311_331177


namespace intersection_complement_theorem_l3311_331121

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_complement_theorem :
  N ∩ (Set.univ \ M) = {x | 1 < x ∧ x ≤ 2} := by sorry

end intersection_complement_theorem_l3311_331121


namespace expression_evaluation_l3311_331138

theorem expression_evaluation : (50 - (2210 - 251)) + (2210 - (251 - 50)) = 100 := by
  sorry

end expression_evaluation_l3311_331138


namespace books_before_sale_l3311_331194

theorem books_before_sale (books_bought : ℕ) (total_books : ℕ) 
  (h1 : books_bought = 56) 
  (h2 : total_books = 91) : 
  total_books - books_bought = 35 := by
  sorry

end books_before_sale_l3311_331194


namespace solve_for_k_l3311_331168

theorem solve_for_k (x y k : ℝ) (hx : x = 2) (hy : y = 1) (heq : k * x - y = 3) : k = 2 := by
  sorry

end solve_for_k_l3311_331168


namespace monkey_climb_l3311_331103

/-- Monkey's climb on a greased pole -/
theorem monkey_climb (pole_height : ℝ) (ascent : ℝ) (total_minutes : ℕ) (slip : ℝ) : 
  pole_height = 10 →
  ascent = 2 →
  total_minutes = 17 →
  (total_minutes / 2 : ℝ) * ascent - ((total_minutes - 1) / 2 : ℝ) * slip = pole_height →
  slip = 6/7 := by
  sorry

end monkey_climb_l3311_331103


namespace range_of_a_l3311_331160

def f (x : ℝ) : ℝ := -x^5 - 3*x^3 - 5*x + 3

theorem range_of_a (a : ℝ) (h : f a + f (a - 2) > 6) : a < 1 := by
  sorry

end range_of_a_l3311_331160


namespace function_characterization_l3311_331171

-- Define the property that f must satisfy
def SatisfiesProperty (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (m + f n) - f m = n

-- Theorem statement
theorem function_characterization :
  ∀ f : ℤ → ℤ, SatisfiesProperty f →
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end function_characterization_l3311_331171


namespace last_four_digits_of_5_to_2017_l3311_331182

theorem last_four_digits_of_5_to_2017 :
  ∃ n : ℕ, 5^2017 ≡ 3125 [ZMOD 10000] :=
by
  -- We define the cycle of last four digits
  let cycle := [3125, 5625, 8125, 0625]
  
  -- We state that 5^5, 5^6, and 5^7 match the first three elements of the cycle
  have h1 : 5^5 ≡ cycle[0] [ZMOD 10000] := by sorry
  have h2 : 5^6 ≡ cycle[1] [ZMOD 10000] := by sorry
  have h3 : 5^7 ≡ cycle[2] [ZMOD 10000] := by sorry
  
  -- We state that the cycle repeats every 4 terms
  have h_cycle : ∀ k : ℕ, 5^(k+4) ≡ 5^k [ZMOD 10000] := by sorry
  
  -- The proof goes here
  sorry


end last_four_digits_of_5_to_2017_l3311_331182


namespace expression_simplification_l3311_331100

theorem expression_simplification (a b c : ℝ) (ha : a = 12) (hb : b = 14) (hc : c = 18) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end expression_simplification_l3311_331100


namespace num_positive_divisors_180_l3311_331108

/-- The number of positive divisors of a natural number -/
def numPositiveDivisors (n : ℕ) : ℕ := sorry

/-- The prime factorization of 180 -/
def primeFactorization180 : List (ℕ × ℕ) := [(2, 2), (3, 2), (5, 1)]

/-- Theorem: The number of positive divisors of 180 is 18 -/
theorem num_positive_divisors_180 : numPositiveDivisors 180 = 18 := by sorry

end num_positive_divisors_180_l3311_331108


namespace triangle_ABC_properties_l3311_331151

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Median AM in triangle ABC --/
def median_AM (t : Triangle) : ℝ := sorry

theorem triangle_ABC_properties (t : Triangle) 
  (h1 : t.a^2 - (t.b - t.c)^2 = (2 - Real.sqrt 3) * t.b * t.c)
  (h2 : Real.sin t.A * Real.sin t.B = (Real.cos (t.C / 2))^2)
  (h3 : median_AM t = Real.sqrt 7) :
  t.A = π / 6 ∧ t.B = π / 6 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry


end triangle_ABC_properties_l3311_331151


namespace max_value_implies_a_l3311_331144

/-- Given a function f(x) = 2x^3 - 3x^2 + a, prove that if its maximum value is 6, then a = 6 -/
theorem max_value_implies_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = 2 * x^3 - 3 * x^2 + a)
  (h2 : ∃ M, M = 6 ∧ ∀ x, f x ≤ M) : 
  a = 6 := by sorry

end max_value_implies_a_l3311_331144


namespace chairs_left_theorem_l3311_331122

/-- The number of chairs left to move given the total number of chairs and the number of chairs moved by each person. -/
def chairs_left_to_move (total : ℕ) (moved_by_carey : ℕ) (moved_by_pat : ℕ) : ℕ :=
  total - (moved_by_carey + moved_by_pat)

/-- Theorem stating that given 74 total chairs, with 28 moved by Carey and 29 moved by Pat, there are 17 chairs left to move. -/
theorem chairs_left_theorem : chairs_left_to_move 74 28 29 = 17 := by
  sorry

end chairs_left_theorem_l3311_331122


namespace equation_result_l3311_331161

theorem equation_result : (88320 : ℤ) + 1315 + 9211 - 1569 = 97277 := by
  sorry

end equation_result_l3311_331161


namespace complement_of_B_l3311_331152

-- Define the set B
def B : Set ℝ := {x | x^2 ≤ 4}

-- State the theorem
theorem complement_of_B : 
  (Set.univ : Set ℝ) \ B = {x | x < -2 ∨ x > 2} := by sorry

end complement_of_B_l3311_331152


namespace number_calculation_l3311_331190

theorem number_calculation (N : ℝ) : (0.15 * 0.30 * 0.50 * N = 108) → N = 4800 := by
  sorry

end number_calculation_l3311_331190


namespace geometric_sequence_common_ratio_l3311_331188

/-- Given three non-zero real numbers x, y, and z forming a geometric sequence
    x(y-z), y(z-x), and z(y-x), prove that the common ratio q satisfies q^2 - q - 1 = 0 -/
theorem geometric_sequence_common_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hseq : ∃ q : ℝ, q ≠ 0 ∧ y * (z - x) = q * (x * (y - z)) ∧ z * (y - x) = q * (y * (z - x))) :
  ∃ q : ℝ, q^2 - q - 1 = 0 ∧ y * (z - x) = q * (x * (y - z)) ∧ z * (y - x) = q * (y * (z - x)) :=
sorry

end geometric_sequence_common_ratio_l3311_331188


namespace fraction_identity_condition_l3311_331101

theorem fraction_identity_condition (a b c d : ℝ) :
  (∀ x : ℝ, (a * x + c) / (b * x + d) = (a + c * x) / (b + d * x)) →
  a / b = c / d :=
by sorry

end fraction_identity_condition_l3311_331101


namespace inequality_proof_l3311_331146

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b * (a + b) + a * c * (a + c) + b * c * (b + c)) / (a * b * c) ≥ 6 := by
  sorry

end inequality_proof_l3311_331146


namespace toaster_customers_l3311_331193

/-- Represents the inverse proportionality between customers and cost -/
def inverse_prop (k : ℝ) (p c : ℝ) : Prop := p * c = k

/-- Applies a discount to a given price -/
def apply_discount (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

theorem toaster_customers : 
  ∀ (k : ℝ),
  inverse_prop k 12 600 →
  (∃ (p : ℝ), 
    inverse_prop k p (apply_discount (2 * 400) 0.1) ∧ 
    p = 10) := by
sorry

end toaster_customers_l3311_331193


namespace inequality_multiplication_l3311_331192

theorem inequality_multiplication (x y : ℝ) (h : x < y) : 2 * x < 2 * y := by
  sorry

end inequality_multiplication_l3311_331192


namespace square_root_two_minus_one_squared_plus_two_times_plus_three_l3311_331156

theorem square_root_two_minus_one_squared_plus_two_times_plus_three (x : ℝ) :
  x = Real.sqrt 2 - 1 → x^2 + 2*x + 3 = 4 := by
  sorry

end square_root_two_minus_one_squared_plus_two_times_plus_three_l3311_331156


namespace remainder_of_division_l3311_331111

theorem remainder_of_division (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
    (h1 : dividend = 1235678)
    (h2 : divisor = 127)
    (h3 : remainder < divisor)
    (h4 : dividend = quotient * divisor + remainder) :
  remainder = 69 := by
  sorry

end remainder_of_division_l3311_331111
