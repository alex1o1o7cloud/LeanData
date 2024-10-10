import Mathlib

namespace quadratic_solution_property_l908_90896

theorem quadratic_solution_property : ∀ a b : ℝ, 
  a^2 + 8*a - 209 = 0 → 
  b^2 + 8*b - 209 = 0 → 
  a ≠ b →
  (a * b) / (a + b) = 209 / 8 := by
sorry

end quadratic_solution_property_l908_90896


namespace nine_div_repeating_third_eq_twentyseven_l908_90814

/-- The repeating decimal 0.3333... --/
def repeating_third : ℚ := 1 / 3

/-- Theorem stating that 9 divided by 0.3333... equals 27 --/
theorem nine_div_repeating_third_eq_twentyseven :
  9 / repeating_third = 27 := by sorry

end nine_div_repeating_third_eq_twentyseven_l908_90814


namespace abhay_sameer_speed_comparison_l908_90821

theorem abhay_sameer_speed_comparison 
  (distance : ℝ) 
  (abhay_speed : ℝ) 
  (time_difference : ℝ) :
  distance = 42 →
  abhay_speed = 7 →
  time_difference = 2 →
  distance / abhay_speed = distance / (distance / (distance / abhay_speed - time_difference)) + time_difference →
  distance / (2 * abhay_speed) = (distance / (distance / (distance / abhay_speed - time_difference))) - 1 :=
by sorry

end abhay_sameer_speed_comparison_l908_90821


namespace x_squared_congruence_l908_90844

theorem x_squared_congruence (x : ℤ) : 
  (5 * x ≡ 10 [ZMOD 25]) → (4 * x ≡ 20 [ZMOD 25]) → (x^2 ≡ 0 [ZMOD 25]) := by
sorry

end x_squared_congruence_l908_90844


namespace unique_solution_power_equation_l908_90870

theorem unique_solution_power_equation :
  ∃! (x y z t : ℕ+), 2^y.val + 2^z.val * 5^t.val - 5^x.val = 1 ∧
    x = 2 ∧ y = 4 ∧ z = 1 ∧ t = 1 := by
  sorry

end unique_solution_power_equation_l908_90870


namespace annika_hike_distance_l908_90840

/-- Represents Annika's hiking scenario -/
structure HikingScenario where
  flat_speed : ℝ  -- minutes per kilometer on flat terrain
  uphill_speed : ℝ  -- minutes per kilometer uphill
  downhill_speed : ℝ  -- minutes per kilometer downhill
  initial_distance : ℝ  -- kilometers hiked initially
  total_time : ℝ  -- total time available to return

/-- Calculates the total distance hiked east given a hiking scenario -/
def total_distance_east (scenario : HikingScenario) : ℝ :=
  sorry

/-- Theorem stating the total distance hiked east in the given scenario -/
theorem annika_hike_distance (scenario : HikingScenario) 
  (h1 : scenario.flat_speed = 10)
  (h2 : scenario.uphill_speed = 15)
  (h3 : scenario.downhill_speed = 7)
  (h4 : scenario.initial_distance = 2.5)
  (h5 : scenario.total_time = 35) :
  total_distance_east scenario = 3.0833 :=
sorry

end annika_hike_distance_l908_90840


namespace solution_set_inequality_l908_90876

open Real

theorem solution_set_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : ∀ x, f x > f' x + 1)
  (h3 : ∀ x, f x - 2024 = -(f (-x) - 2024)) :
  {x : ℝ | f x - 2023 * exp x < 1} = {x : ℝ | x > 0} := by
  sorry

end solution_set_inequality_l908_90876


namespace sum_of_digits_of_1996_digit_multiple_of_9_l908_90893

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a 1996-digit integer -/
def is1996Digit (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_1996_digit_multiple_of_9 (n : ℕ) 
  (h1 : is1996Digit n) 
  (h2 : n % 9 = 0) : 
  let p := sumOfDigits n
  let q := sumOfDigits p
  let r := sumOfDigits q
  r = 9 := by sorry

end sum_of_digits_of_1996_digit_multiple_of_9_l908_90893


namespace complex_circle_l908_90812

-- Define a complex number
def z : ℂ := sorry

-- Define the condition |z - (-1 + i)| = 4
def condition (z : ℂ) : Prop := Complex.abs (z - (-1 + Complex.I)) = 4

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 16

-- Theorem statement
theorem complex_circle (z : ℂ) (h : condition z) :
  circle_equation z.re z.im := by sorry

end complex_circle_l908_90812


namespace scientific_notation_86560_l908_90864

theorem scientific_notation_86560 : 
  86560 = 8.656 * (10 ^ 4) := by sorry

end scientific_notation_86560_l908_90864


namespace tangent_at_2_minus_6_tangent_through_origin_l908_90850

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x - 16

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem for the tangent line at (2, -6)
theorem tangent_at_2_minus_6 :
  let x₀ : ℝ := 2
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 13 * x - 32 :=
sorry

-- Theorem for the tangent line passing through the origin
theorem tangent_through_origin :
  ∃ x₀ y₀ : ℝ,
    f x₀ = y₀ ∧
    f' x₀ * (-x₀) + y₀ = 0 ∧
    (∀ x y : ℝ, y = f' x₀ * x ↔ y = 13 * x) ∧
    x₀ = -2 ∧
    y₀ = -26 :=
sorry

end tangent_at_2_minus_6_tangent_through_origin_l908_90850


namespace contrapositive_equivalence_l908_90880

theorem contrapositive_equivalence (x : ℝ) :
  (¬(x = 1) → ¬(x^2 = 1)) ↔ (¬(x = 1) → (x ≠ 1 ∧ x ≠ -1)) := by sorry

end contrapositive_equivalence_l908_90880


namespace complex_product_magnitude_l908_90827

theorem complex_product_magnitude : 
  Complex.abs ((5 - 3*Complex.I) * (7 + 24*Complex.I)) = 25 * Real.sqrt 34 := by
  sorry

end complex_product_magnitude_l908_90827


namespace f_satisfies_conditions_l908_90867

/-- The function f satisfying the given conditions -/
noncomputable def f (x : ℝ) : ℝ := -5 * (4^x - 5^x)

/-- Theorem stating that f satisfies the required conditions -/
theorem f_satisfies_conditions :
  (f 1 = 5) ∧
  (∀ x y : ℝ, f (x + y) = 4^y * f x + 5^x * f y) :=
by sorry

end f_satisfies_conditions_l908_90867


namespace triangle_vector_representation_l908_90834

/-- Given a triangle ABC and a point P on line AB, prove that CP can be represented
    in terms of CA and CB under certain conditions. -/
theorem triangle_vector_representation (A B C P : EuclideanSpace ℝ (Fin 3))
    (a b : EuclideanSpace ℝ (Fin 3)) : 
    (C - A = a) →  -- CA = a
    (C - B = b) →  -- CB = b
    (∃ t : ℝ, P = (1 - t) • A + t • B) →  -- P is on line AB
    (A - P = 2 • (P - B)) →  -- AP = 2PB
    (C - P = (1/3) • a + (2/3) • b) := by
  sorry

end triangle_vector_representation_l908_90834


namespace basketball_probability_l908_90846

theorem basketball_probability (p_free_throw p_high_school p_pro : ℚ) 
  (h1 : p_free_throw = 4/5)
  (h2 : p_high_school = 1/2)
  (h3 : p_pro = 1/3) :
  1 - (1 - p_free_throw) * (1 - p_high_school) * (1 - p_pro) = 14/15 := by
  sorry

end basketball_probability_l908_90846


namespace systematic_sampling_interval_count_l908_90830

theorem systematic_sampling_interval_count
  (total_papers : Nat)
  (selected_papers : Nat)
  (interval_start : Nat)
  (interval_end : Nat)
  (h1 : total_papers = 1000)
  (h2 : selected_papers = 50)
  (h3 : interval_start = 850)
  (h4 : interval_end = 949)
  (h5 : interval_start ≤ interval_end)
  (h6 : interval_end ≤ total_papers) :
  let sample_interval := total_papers / selected_papers
  let interval_size := interval_end - interval_start + 1
  interval_size / sample_interval = 5 :=
by sorry

end systematic_sampling_interval_count_l908_90830


namespace smallest_repeating_block_of_nine_elevenths_l908_90883

/-- The number of digits in the smallest repeating block of the decimal expansion of 9/11 -/
def smallest_repeating_block_length : ℕ :=
  2

/-- The fraction we're considering -/
def fraction : ℚ :=
  9 / 11

theorem smallest_repeating_block_of_nine_elevenths :
  smallest_repeating_block_length = 2 ∧
  ∃ (a b : ℕ) (k : ℕ+), fraction = (a * 10^smallest_repeating_block_length + b) / (10^smallest_repeating_block_length - 1) / k :=
by sorry

end smallest_repeating_block_of_nine_elevenths_l908_90883


namespace factorization_of_2m_squared_minus_2_l908_90898

theorem factorization_of_2m_squared_minus_2 (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end factorization_of_2m_squared_minus_2_l908_90898


namespace digit_sum_inequality_l908_90875

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Condition for the existence of c_k -/
def HasValidCk (k : ℕ) : Prop :=
  ∃ (c_k : ℝ), c_k > 0 ∧ ∀ (n : ℕ), n > 0 → S (k * n) ≥ c_k * S n

/-- k has no prime divisors other than 2 or 5 -/
def HasOnly2And5Factors (k : ℕ) : Prop :=
  ∀ (p : ℕ), p.Prime → p ∣ k → p = 2 ∨ p = 5

/-- Main theorem -/
theorem digit_sum_inequality (k : ℕ) (h : k > 1) :
  HasValidCk k ↔ HasOnly2And5Factors k := by sorry

end digit_sum_inequality_l908_90875


namespace function_characterization_l908_90843

def DivisibilityCondition (f : ℕ+ → ℕ+) : Prop :=
  ∀ (a b : ℕ+), a + b > 2019 → (a + f b) ∣ (a^2 + b * f a)

theorem function_characterization (f : ℕ+ → ℕ+) 
  (h : DivisibilityCondition f) : 
  ∃ (r : ℕ+), ∀ (x : ℕ+), f x = r * x := by
  sorry

end function_characterization_l908_90843


namespace garys_gold_cost_per_gram_l908_90804

/-- Proves that Gary's gold costs $15 per gram given the conditions of the problem -/
theorem garys_gold_cost_per_gram (gary_grams : ℝ) (anna_grams : ℝ) (anna_cost_per_gram : ℝ) (total_cost : ℝ)
  (h1 : gary_grams = 30)
  (h2 : anna_grams = 50)
  (h3 : anna_cost_per_gram = 20)
  (h4 : total_cost = 1450)
  (h5 : gary_grams * x + anna_grams * anna_cost_per_gram = total_cost) :
  x = 15 := by
  sorry

#check garys_gold_cost_per_gram

end garys_gold_cost_per_gram_l908_90804


namespace sports_league_games_l908_90815

/-- The number of games in a complete season for a sports league -/
def total_games (n : ℕ) (d : ℕ) (t : ℕ) (s : ℕ) (c : ℕ) : ℕ :=
  (n * (d - 1) * s + n * t * c) / 2

/-- Theorem: The total number of games in the given sports league is 296 -/
theorem sports_league_games :
  total_games 8 8 8 3 2 = 296 := by
  sorry

end sports_league_games_l908_90815


namespace symmetric_line_correct_l908_90829

/-- Given a line with equation ax + by + c = 0, returns the equation of the line
    symmetric to it with respect to y = x as a triple (a', b', c') representing
    a'x + b'y + c' = 0 -/
def symmetric_line (a b c : ℝ) : ℝ × ℝ × ℝ := (b, a, c)

theorem symmetric_line_correct :
  let original_line := (1, -3, 5)  -- Represents x - 3y + 5 = 0
  let symm_line := symmetric_line 1 (-3) 5
  symm_line = (3, -1, -5)  -- Represents 3x - y - 5 = 0
  := by sorry

end symmetric_line_correct_l908_90829


namespace binomial_eight_choose_two_l908_90800

theorem binomial_eight_choose_two : (8 : ℕ).choose 2 = 28 := by sorry

end binomial_eight_choose_two_l908_90800


namespace roots_always_real_l908_90884

/-- Given real numbers a, b, and c, the discriminant of the quadratic equation
    resulting from 1/(x+a) + 1/(x+b) + 1/(x+c) = 3/x is non-negative. -/
theorem roots_always_real (a b c : ℝ) : 
  2 * (a^2 * (b - c)^2 + b^2 * (c - a)^2 + c^2 * (a - b)^2) ≥ 0 := by
  sorry

#check roots_always_real

end roots_always_real_l908_90884


namespace complex_number_problem_l908_90838

theorem complex_number_problem (a : ℝ) :
  (((a^2 - 1) : ℂ) + (a + 1) * I).im ≠ 0 ∧ ((a^2 - 1) : ℂ).re = 0 →
  (a + I^2016) / (1 + I) = 1 - I := by
sorry

end complex_number_problem_l908_90838


namespace fraction_equality_problem_l908_90858

theorem fraction_equality_problem (y : ℝ) : 
  (4 + y) / (6 + y) = (2 + y) / (3 + y) ↔ y = 0 :=
by sorry

end fraction_equality_problem_l908_90858


namespace largest_difference_l908_90872

def A : ℕ := 3 * 2003^2002
def B : ℕ := 2003^2002
def C : ℕ := 2002 * 2003^2001
def D : ℕ := 3 * 2003^2001
def E : ℕ := 2003^2001
def F : ℕ := 2003^2000

theorem largest_difference (A B C D E F : ℕ) 
  (hA : A = 3 * 2003^2002)
  (hB : B = 2003^2002)
  (hC : C = 2002 * 2003^2001)
  (hD : D = 3 * 2003^2001)
  (hE : E = 2003^2001)
  (hF : F = 2003^2000) :
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) :=
by sorry

end largest_difference_l908_90872


namespace chandler_saves_26_weeks_l908_90842

/-- The number of weeks it takes Chandler to save for a mountain bike. -/
def weeks_to_save : ℕ :=
  let bike_cost : ℕ := 650
  let birthday_money : ℕ := 80 + 35 + 15
  let weekly_earnings : ℕ := 20
  (bike_cost - birthday_money) / weekly_earnings

/-- Theorem stating that it takes 26 weeks for Chandler to save for the mountain bike. -/
theorem chandler_saves_26_weeks : weeks_to_save = 26 := by
  sorry

end chandler_saves_26_weeks_l908_90842


namespace bakery_pie_division_l908_90802

theorem bakery_pie_division (total_pie : ℚ) (num_friends : ℕ) : 
  total_pie = 5/8 → num_friends = 4 → total_pie / num_friends = 5/32 := by
  sorry

end bakery_pie_division_l908_90802


namespace sine_of_supplementary_angles_l908_90803

theorem sine_of_supplementary_angles (α β : Real) :
  α + β = Real.pi → Real.sin α = Real.sin β := by
  sorry

end sine_of_supplementary_angles_l908_90803


namespace octahedron_flattenable_l908_90836

/-- Represents a cube -/
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)
  faces : Finset (Fin 6)

/-- Represents an octahedron -/
structure Octahedron where
  vertices : Finset (Fin 6)
  edges : Finset (Fin 12)
  faces : Finset (Fin 8)

/-- Defines the relationship between a cube and its corresponding octahedron -/
def correspondingOctahedron (c : Cube) : Octahedron :=
  sorry

/-- Defines what it means for a set of faces to be connected -/
def isConnected {α : Type*} (s : Finset α) : Prop :=
  sorry

/-- Defines what it means for a set of faces to be flattenable -/
def isFlattenable {α : Type*} (s : Finset α) : Prop :=
  sorry

/-- Defines the operation of cutting edges on a polyhedron -/
def cutEdges {α : Type*} (edges : Finset α) (toCut : Finset α) : Finset α :=
  sorry

theorem octahedron_flattenable (c : Cube) (cubeCuts : Finset (Fin 12)) :
  (cubeCuts.card = 7) →
  (isConnected (cutEdges c.edges cubeCuts)) →
  (isFlattenable (cutEdges c.edges cubeCuts)) →
  let o := correspondingOctahedron c
  let octaCuts := c.edges \ cubeCuts
  (isConnected (cutEdges o.edges octaCuts)) ∧
  (isFlattenable (cutEdges o.edges octaCuts)) := by
  sorry

end octahedron_flattenable_l908_90836


namespace equilateral_triangle_on_concentric_circles_l908_90879

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a circle with center and radius
structure Circle where
  center : Point2D
  radius : ℝ

-- Define an equilateral triangle
structure EquilateralTriangle where
  a : Point2D
  b : Point2D
  c : Point2D

-- Function to check if a point lies on a circle
def pointOnCircle (p : Point2D) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

-- Theorem statement
theorem equilateral_triangle_on_concentric_circles 
  (center : Point2D) (r₁ r₂ r₃ : ℝ) 
  (h₁ : 0 < r₁) (h₂ : r₁ < r₂) (h₃ : r₂ < r₃) :
  ∃ (t : EquilateralTriangle),
    pointOnCircle t.a (Circle.mk center r₂) ∧
    pointOnCircle t.b (Circle.mk center r₁) ∧
    pointOnCircle t.c (Circle.mk center r₃) :=
sorry

end equilateral_triangle_on_concentric_circles_l908_90879


namespace intercollegiate_competition_l908_90825

theorem intercollegiate_competition (day1 day2 day3 day1_and_2 day2_and_3 only_day1 : ℕ)
  (h1 : day1 = 175)
  (h2 : day2 = 210)
  (h3 : day3 = 150)
  (h4 : day1_and_2 = 80)
  (h5 : day2_and_3 = 70)
  (h6 : only_day1 = 45)
  : ∃ all_days : ℕ,
    day1 = only_day1 + day1_and_2 + all_days ∧
    day2 = day1_and_2 + day2_and_3 + all_days ∧
    day3 = day2_and_3 + all_days ∧
    all_days = 50 := by
  sorry

end intercollegiate_competition_l908_90825


namespace ancient_chinese_fruit_problem_l908_90841

/-- Represents the ancient Chinese fruit problem -/
theorem ancient_chinese_fruit_problem 
  (x y : ℚ) -- x: number of bitter fruits, y: number of sweet fruits
  (h1 : x + y = 1000) -- total number of fruits
  (h2 : 7 * (4 / 7 : ℚ) = 4) -- cost of 7 bitter fruits
  (h3 : 9 * (11 / 9 : ℚ) = 11) -- cost of 9 sweet fruits
  (h4 : (4 / 7 : ℚ) * x + (11 / 9 : ℚ) * y = 999) -- total cost
  : (x + y = 1000 ∧ (4 / 7 : ℚ) * x + (11 / 9 : ℚ) * y = 999) :=
by sorry

end ancient_chinese_fruit_problem_l908_90841


namespace max_stamps_proof_l908_90811

/-- The price of a stamp in cents -/
def stamp_price : ℕ := 50

/-- The total budget in cents -/
def total_budget : ℕ := 5000

/-- The number of stamps required for discount eligibility -/
def discount_threshold : ℕ := 80

/-- The discount amount per stamp in cents -/
def discount_amount : ℕ := 5

/-- The maximum number of stamps that can be purchased with the given conditions -/
def max_stamps : ℕ := 111

theorem max_stamps_proof :
  ∀ n : ℕ,
  n ≤ max_stamps ∧
  (n > discount_threshold → n * (stamp_price - discount_amount) ≤ total_budget) ∧
  (n ≤ discount_threshold → n * stamp_price ≤ total_budget) ∧
  (max_stamps > discount_threshold → max_stamps * (stamp_price - discount_amount) ≤ total_budget) ∧
  (max_stamps + 1 > discount_threshold → (max_stamps + 1) * (stamp_price - discount_amount) > total_budget) := by
  sorry

end max_stamps_proof_l908_90811


namespace poster_difference_l908_90817

/-- The number of posters Mario made -/
def mario_posters : ℕ := 18

/-- The total number of posters made by Mario and Samantha -/
def total_posters : ℕ := 51

/-- The number of posters Samantha made -/
def samantha_posters : ℕ := total_posters - mario_posters

/-- Samantha made more posters than Mario -/
axiom samantha_made_more : samantha_posters > mario_posters

theorem poster_difference : samantha_posters - mario_posters = 15 := by
  sorry

end poster_difference_l908_90817


namespace exactly_four_separators_l908_90869

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of five points in a plane -/
def FivePointSet := Fin 5 → Point

/-- A circle in a plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if three points are collinear -/
def are_collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if four points are concyclic -/
def are_concyclic (p q r s : Point) : Prop := sorry

/-- Predicate to check if a point is inside a circle -/
def is_inside (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a point is outside a circle -/
def is_outside (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a point is on a circle -/
def is_on_circle (p : Point) (c : Circle) : Prop := sorry

/-- Predicate to check if a circle is a separator for a set of five points -/
def is_separator (c : Circle) (s : FivePointSet) : Prop :=
  ∃ (i j k l m : Fin 5),
    i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ m ≠ i ∧ m ≠ j ∧ m ≠ k ∧ m ≠ l ∧
    is_on_circle (s i) c ∧ is_on_circle (s j) c ∧ is_on_circle (s k) c ∧
    is_inside (s l) c ∧ is_outside (s m) c

/-- The main theorem -/
theorem exactly_four_separators (s : FivePointSet) :
  (∀ (i j k : Fin 5), i ≠ j → j ≠ k → k ≠ i → ¬are_collinear (s i) (s j) (s k)) →
  (∀ (i j k l : Fin 5), i ≠ j → j ≠ k → k ≠ l → l ≠ i → ¬are_concyclic (s i) (s j) (s k) (s l)) →
  ∃! (separators : Finset Circle), (∀ c ∈ separators, is_separator c s) ∧ separators.card = 4 :=
sorry

end exactly_four_separators_l908_90869


namespace complement_of_union_l908_90806

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

theorem complement_of_union (U M N : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5, 6})
  (hM : M = {2, 3, 5})
  (hN : N = {4, 5}) :
  (M ∪ N)ᶜ = {1, 6} := by
  sorry

end complement_of_union_l908_90806


namespace tangent_lines_at_P_l908_90808

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Define the point P
def P : ℝ × ℝ := (2, -6)

-- Define the two potential tangent lines
def line1 (x y : ℝ) : Prop := 3*x + y = 0
def line2 (x y : ℝ) : Prop := 24*x - y - 54 = 0

-- Theorem statement
theorem tangent_lines_at_P :
  (∃ t : ℝ, f t = P.2 ∧ f' t * (P.1 - t) = P.2 - f t ∧ (line1 P.1 P.2 ∨ line2 P.1 P.2)) ∧
  (∀ x y : ℝ, (line1 x y ∨ line2 x y) → ∃ t : ℝ, f t = y ∧ f' t * (x - t) = y - f t) :=
sorry

end tangent_lines_at_P_l908_90808


namespace product_of_polynomials_l908_90857

theorem product_of_polynomials (d p q : ℝ) : 
  (4 * d^3 + 2 * d^2 - 5 * d + p) * (6 * d^2 + q * d - 3) = 
  24 * d^5 + q * d^4 - 33 * d^3 - 15 * d^2 + q * d - 15 → 
  p + q = 12.5 := by
sorry

end product_of_polynomials_l908_90857


namespace sqrt_of_four_l908_90835

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y * y = x}

-- Theorem statement
theorem sqrt_of_four : sqrt 4 = {2, -2} := by
  sorry

end sqrt_of_four_l908_90835


namespace parallel_line_equation_perpendicular_lines_equation_l908_90897

-- Define the slope of line l₁
def slope_l1 : ℚ := -3 / 4

-- Define a point that l₂ passes through
def point_l2 : ℚ × ℚ := (-1, 3)

-- Define the area of the triangle formed by l₂ and the coordinate axes
def triangle_area : ℚ := 4

-- Theorem for the parallel line
theorem parallel_line_equation :
  ∃ (c : ℚ), 3 * point_l2.1 + 4 * point_l2.2 + c = 0 ∧
  ∀ (x y : ℚ), 3 * x + 4 * y + c = 0 ↔ 3 * x + 4 * y - 9 = 0 :=
sorry

-- Theorem for the perpendicular lines
theorem perpendicular_lines_equation :
  ∃ (n : ℚ), (n^2 = 96) ∧
  (∀ (x y : ℚ), 4 * x - 3 * y + n = 0 ↔ 4 * x - 3 * y + 4 * Real.sqrt 6 = 0 ∨
                                        4 * x - 3 * y - 4 * Real.sqrt 6 = 0) ∧
  (1/2 * |n/4| * |n/3| = triangle_area) :=
sorry

end parallel_line_equation_perpendicular_lines_equation_l908_90897


namespace solution_count_l908_90820

/-- The number of distinct solutions to the system of equations:
    x = x^2 + y^2
    y = 3x^2y - y^3 -/
theorem solution_count : 
  (Set.ncard {p : ℝ × ℝ | let (x, y) := p; x = x^2 + y^2 ∧ y = 3*x^2*y - y^3} : ℕ) = 2 := by
  sorry

end solution_count_l908_90820


namespace hexagon_rounding_exists_l908_90881

/-- Represents a hexagon with numbers on its vertices and sums on its sides. -/
structure Hexagon where
  -- Vertex numbers
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ
  a₄ : ℝ
  a₅ : ℝ
  a₆ : ℝ
  -- Side sums
  s₁ : ℝ
  s₂ : ℝ
  s₃ : ℝ
  s₄ : ℝ
  s₅ : ℝ
  s₆ : ℝ
  -- Ensure side sums match vertex sums
  h₁ : s₁ = a₁ + a₂
  h₂ : s₂ = a₂ + a₃
  h₃ : s₃ = a₃ + a₄
  h₄ : s₄ = a₄ + a₅
  h₅ : s₅ = a₅ + a₆
  h₆ : s₆ = a₆ + a₁

/-- Represents a rounding strategy for the hexagon. -/
structure RoundedHexagon where
  -- Rounded vertex numbers
  r₁ : ℤ
  r₂ : ℤ
  r₃ : ℤ
  r₄ : ℤ
  r₅ : ℤ
  r₆ : ℤ
  -- Rounded side sums
  t₁ : ℤ
  t₂ : ℤ
  t₃ : ℤ
  t₄ : ℤ
  t₅ : ℤ
  t₆ : ℤ

/-- Theorem: For any hexagon, there exists a rounding strategy that maintains the sum property. -/
theorem hexagon_rounding_exists (h : Hexagon) : 
  ∃ (r : RoundedHexagon), 
    (r.t₁ = r.r₁ + r.r₂) ∧
    (r.t₂ = r.r₂ + r.r₃) ∧
    (r.t₃ = r.r₃ + r.r₄) ∧
    (r.t₄ = r.r₄ + r.r₅) ∧
    (r.t₅ = r.r₅ + r.r₆) ∧
    (r.t₆ = r.r₆ + r.r₁) :=
  sorry

end hexagon_rounding_exists_l908_90881


namespace sum_of_ages_proof_l908_90868

/-- Proves that the sum of ages of a mother and daughter is 70 years,
    given the daughter's age and the age difference. -/
theorem sum_of_ages_proof (daughter_age mother_daughter_diff : ℕ) : 
  daughter_age = 19 →
  mother_daughter_diff = 32 →
  daughter_age + (daughter_age + mother_daughter_diff) = 70 := by
sorry

end sum_of_ages_proof_l908_90868


namespace sampling_survey_correct_l908_90852

/-- Represents a statement about quality testing methods -/
inductive QualityTestingMethod
| SamplingSurvey
| Other

/-- Represents the correctness of a statement -/
inductive Correctness
| Correct
| Incorrect

/-- The correct method for testing the quality of a batch of light bulbs -/
def lightBulbQualityTestingMethod : QualityTestingMethod := QualityTestingMethod.SamplingSurvey

/-- Theorem stating that sampling survey is the correct method for testing light bulb quality -/
theorem sampling_survey_correct :
  Correctness.Correct = match lightBulbQualityTestingMethod with
    | QualityTestingMethod.SamplingSurvey => Correctness.Correct
    | QualityTestingMethod.Other => Correctness.Incorrect :=
by sorry

end sampling_survey_correct_l908_90852


namespace solve_distance_problem_l908_90805

def distance_problem (initial_speed : ℝ) (initial_time : ℝ) (speed_increase : ℝ) (additional_time : ℝ) : Prop :=
  let initial_distance := initial_speed * initial_time
  let new_speed := initial_speed * (1 + speed_increase)
  let additional_distance := new_speed * additional_time
  let total_distance := initial_distance + additional_distance
  total_distance = 13

theorem solve_distance_problem :
  distance_problem 2 2 0.5 3 := by
  sorry

end solve_distance_problem_l908_90805


namespace two_a_plus_b_values_l908_90828

theorem two_a_plus_b_values (a b : ℝ) 
  (h1 : |a - 1| = 4)
  (h2 : |-b| = |-7|)
  (h3 : |a + b| ≠ a + b) :
  2 * a + b = 3 ∨ 2 * a + b = -13 := by
sorry

end two_a_plus_b_values_l908_90828


namespace largest_number_from_hcf_lcm_l908_90839

theorem largest_number_from_hcf_lcm (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  Nat.gcd a b = 210 →
  Nat.gcd (Nat.gcd a b) c = 210 →
  Nat.lcm (Nat.lcm a b) c = 902910 →
  max a (max b c) = 4830 :=
sorry

end largest_number_from_hcf_lcm_l908_90839


namespace quadratic_function_unique_form_l908_90871

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_unique_form 
  (f : ℝ → ℝ) 
  (h_vertex : f (-1) = 4 ∧ ∀ x, f x ≤ f (-1)) 
  (h_point : f 2 = -5) :
  ∃ a b c : ℝ, f = quadratic_function a b c ∧ a = -1 ∧ b = -2 ∧ c = 3 :=
sorry

end quadratic_function_unique_form_l908_90871


namespace even_fraction_integers_l908_90878

theorem even_fraction_integers (a : ℤ) : 
  (∃ k : ℤ, a / (1011 - a) = 2 * k) ↔ 
  a ∈ ({1010, 1012, 1008, 1014, 674, 1348, 0, 2022} : Set ℤ) := by
sorry

end even_fraction_integers_l908_90878


namespace jordan_no_quiz_probability_l908_90860

theorem jordan_no_quiz_probability (p_quiz : ℚ) (h : p_quiz = 5/9) :
  1 - p_quiz = 4/9 := by
sorry

end jordan_no_quiz_probability_l908_90860


namespace percentage_sum_l908_90819

theorem percentage_sum : (0.15 * 25) + (0.12 * 45) = 9.15 := by
  sorry

end percentage_sum_l908_90819


namespace parallel_lines_condition_l908_90823

/-- Two lines ax+2y+1=0 and 3x+(a-1)y+1=0 are parallel if and only if a = -2 -/
theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + 1 = 0 ∧ 3*x + (a-1)*y + 1 = 0) ↔ a = -2 :=
by sorry

end parallel_lines_condition_l908_90823


namespace raisin_difference_is_twenty_l908_90859

/-- The number of raisin cookies Helen baked yesterday -/
def yesterday_raisin : ℕ := 300

/-- The number of raisin cookies Helen baked today -/
def today_raisin : ℕ := 280

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_chocolate : ℕ := 519

/-- The number of chocolate chip cookies Helen baked today -/
def today_chocolate : ℕ := 359

/-- The difference in raisin cookies baked between yesterday and today -/
def raisin_difference : ℕ := yesterday_raisin - today_raisin

theorem raisin_difference_is_twenty : raisin_difference = 20 := by
  sorry

end raisin_difference_is_twenty_l908_90859


namespace ron_book_picks_l908_90855

/-- Represents a book club with its properties --/
structure BookClub where
  members : ℕ
  weekly_meetings : ℕ
  holiday_breaks : ℕ
  guest_picks : ℕ
  leap_year_extra_meeting : ℕ

/-- Calculates the number of times a member gets to pick a book --/
def picks_per_member (club : BookClub) (is_leap_year : Bool) : ℕ :=
  let total_meetings := club.weekly_meetings - club.holiday_breaks + (if is_leap_year then club.leap_year_extra_meeting else 0)
  let member_picks := total_meetings - club.guest_picks - (if is_leap_year then 1 else 0)
  member_picks / club.members

/-- Theorem stating that Ron gets to pick 3 books in both leap and non-leap years --/
theorem ron_book_picks (club : BookClub) 
    (h1 : club.members = 13)
    (h2 : club.weekly_meetings = 52)
    (h3 : club.holiday_breaks = 5)
    (h4 : club.guest_picks = 6)
    (h5 : club.leap_year_extra_meeting = 1) : 
    picks_per_member club false = 3 ∧ picks_per_member club true = 3 := by
  sorry

end ron_book_picks_l908_90855


namespace range_of_m_l908_90809

theorem range_of_m (p q : ℝ → Prop) (m : ℝ) : 
  (∀ x, p x ↔ -2 ≤ 1 - (x-1)/3 ∧ 1 - (x-1)/3 ≤ 2) →
  (∀ x, q x ↔ x^2 - 2*x + (1-m^2) ≤ 0) →
  (m > 0) →
  (∀ x, ¬(p x) → ¬(q x)) →
  (∃ x, ¬(p x) ∧ q x) →
  m ≥ 9 ∧ ∀ k ≥ 9, ∃ x, ¬(p x) ∧ q x :=
by sorry

end range_of_m_l908_90809


namespace building_shadow_length_l908_90816

/-- Given a flagstaff and a building with their respective heights and shadow lengths,
    prove that the length of the shadow cast by the building is as calculated. -/
theorem building_shadow_length 
  (flagstaff_height : ℝ) 
  (flagstaff_shadow : ℝ)
  (building_height : ℝ) :
  flagstaff_height = 17.5 →
  flagstaff_shadow = 40.25 →
  building_height = 12.5 →
  ∃ (building_shadow : ℝ),
    building_shadow = 28.75 ∧
    flagstaff_height / flagstaff_shadow = building_height / building_shadow :=
by sorry

end building_shadow_length_l908_90816


namespace line_intersects_ellipse_chord_length_implies_line_equation_l908_90818

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1

-- Define the line equation
def line (x y m : ℝ) : Prop := y = x + m

-- Theorem for part 1
theorem line_intersects_ellipse (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
sorry

-- Theorem for part 2
theorem chord_length_implies_line_equation (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ m ∧ line x₂ y₂ m ∧
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = (2 * Real.sqrt 10 / 5)^2) →
  m = 0 :=
sorry

end line_intersects_ellipse_chord_length_implies_line_equation_l908_90818


namespace range_of_x_when_m_is_2_range_of_m_when_q_necessary_not_sufficient_l908_90894

-- Define propositions p and q
def p (x m : ℝ) : Prop := x^2 - 5*m*x + 6*m^2 < 0

def q (x : ℝ) : Prop := (x - 5) / (x - 1) < 0

-- Theorem 1
theorem range_of_x_when_m_is_2 (x : ℝ) :
  (p x 2 ∨ q x) → 1 < x ∧ x < 6 := by sorry

-- Theorem 2
theorem range_of_m_when_q_necessary_not_sufficient (m : ℝ) :
  (m > 0 ∧ ∀ x, p x m → q x) ∧ (∃ x, q x ∧ ¬p x m) →
  1/2 ≤ m ∧ m ≤ 5/3 := by sorry

end range_of_x_when_m_is_2_range_of_m_when_q_necessary_not_sufficient_l908_90894


namespace ceiling_negative_example_l908_90885

theorem ceiling_negative_example : ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end ceiling_negative_example_l908_90885


namespace rest_area_location_l908_90861

/-- Represents a highway with exits and a rest area -/
structure Highway where
  fifth_exit : ℝ
  seventh_exit : ℝ
  rest_area : ℝ

/-- The rest area is located halfway between the fifth and seventh exits -/
def is_halfway (h : Highway) : Prop :=
  h.rest_area = (h.fifth_exit + h.seventh_exit) / 2

/-- Theorem: Given the conditions, prove that the rest area is at milepost 65 -/
theorem rest_area_location (h : Highway) 
    (h_fifth : h.fifth_exit = 35)
    (h_seventh : h.seventh_exit = 95)
    (h_halfway : is_halfway h) : 
    h.rest_area = 65 := by
  sorry

#check rest_area_location

end rest_area_location_l908_90861


namespace min_value_and_nonexistence_l908_90807

theorem min_value_and_nonexistence (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1 / a + 1 / b = Real.sqrt (a * b)) :
  (∀ x y, x > 0 → y > 0 → 1 / x + 1 / y = Real.sqrt (x * y) → x^3 + y^3 ≥ 4 * Real.sqrt 2) ∧ 
  (¬∃ x y, x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = Real.sqrt (x * y) ∧ 2 * x + 3 * y = 6) := by
  sorry

end min_value_and_nonexistence_l908_90807


namespace gizi_does_not_catch_up_l908_90845

/-- Represents the work progress of Kató and Gizi -/
structure WorkProgress where
  kato_lines : ℕ
  gizi_lines : ℕ

/-- Represents the copying rates and page capacities -/
structure CopyingParameters where
  kato_lines_per_page : ℕ
  gizi_lines_per_page : ℕ
  kato_rate : ℕ
  gizi_rate : ℕ

def initial_state : WorkProgress :=
  { kato_lines := 80,  -- 4 pages * 20 lines per page
    gizi_lines := 0 }

def copying_params : CopyingParameters :=
  { kato_lines_per_page := 20,
    gizi_lines_per_page := 30,
    kato_rate := 3,
    gizi_rate := 4 }

def setup_time_progress (wp : WorkProgress) : WorkProgress :=
  { kato_lines := wp.kato_lines + 3,  -- 2.5 rounded up to 3
    gizi_lines := wp.gizi_lines }

def update_progress (wp : WorkProgress) (cp : CopyingParameters) : WorkProgress :=
  { kato_lines := wp.kato_lines + cp.kato_rate,
    gizi_lines := wp.gizi_lines + cp.gizi_rate }

def gizi_catches_up (wp : WorkProgress) : Prop :=
  wp.gizi_lines * 4 ≥ wp.kato_lines * 3

theorem gizi_does_not_catch_up :
  ¬∃ n : ℕ, gizi_catches_up (n.iterate (update_progress · copying_params) (setup_time_progress initial_state)) ∧
            (n.iterate (update_progress · copying_params) (setup_time_progress initial_state)).gizi_lines ≤ 150 :=
sorry

end gizi_does_not_catch_up_l908_90845


namespace chooseAndAssignTheorem_l908_90813

-- Define the set of members
inductive Member : Type
| Alice : Member
| Bob : Member
| Carol : Member
| Dave : Member

-- Define the set of officer roles
inductive Role : Type
| President : Role
| Secretary : Role
| Treasurer : Role

-- Define a function to calculate the number of ways to choose and assign roles
def waysToChooseAndAssign : ℕ :=
  -- Number of ways to choose 3 out of 4 members
  (Nat.choose 4 3) *
  -- Number of ways to assign 3 roles to 3 chosen members
  (Nat.factorial 3)

-- Theorem statement
theorem chooseAndAssignTheorem : waysToChooseAndAssign = 24 := by
  sorry


end chooseAndAssignTheorem_l908_90813


namespace cat_grooming_time_l908_90887

/-- Calculates the total grooming time for a cat -/
def total_grooming_time (
  claws_per_foot : ℕ
  ) (
  feet : ℕ
  ) (
  clip_time : ℕ
  ) (
  ear_clean_time : ℕ
  ) (
  shampoo_time : ℕ
  ) : ℕ :=
  let total_claws := claws_per_foot * feet
  let total_clip_time := total_claws * clip_time
  let total_ear_clean_time := 2 * ear_clean_time
  let total_shampoo_time := shampoo_time * 60
  total_clip_time + total_ear_clean_time + total_shampoo_time

theorem cat_grooming_time :
  total_grooming_time 4 4 10 90 5 = 640 := by
  sorry

end cat_grooming_time_l908_90887


namespace single_plane_division_two_planes_division_l908_90877

-- Define a type for space
structure Space :=
  (points : Set Point)

-- Define a type for plane
structure Plane :=
  (equation : Point → Prop)

-- Define a function to count the number of parts a set of planes divides space into
def countParts (space : Space) (planes : Set Plane) : ℕ :=
  sorry

-- Theorem for a single plane
theorem single_plane_division (space : Space) (plane : Plane) :
  countParts space {plane} = 2 :=
sorry

-- Theorem for two planes
theorem two_planes_division (space : Space) (plane1 plane2 : Plane) :
  countParts space {plane1, plane2} = 3 ∨ countParts space {plane1, plane2} = 4 :=
sorry

end single_plane_division_two_planes_division_l908_90877


namespace log_division_simplification_l908_90847

theorem log_division_simplification :
  Real.log 27 / Real.log (1 / 27) = -1 := by
  sorry

end log_division_simplification_l908_90847


namespace sum_first_four_terms_l908_90837

def arithmetic_sequence (a : ℤ) (d : ℤ) : ℕ → ℤ
  | 0 => a
  | n + 1 => arithmetic_sequence a d n + d

theorem sum_first_four_terms
  (a d : ℤ)
  (h5 : arithmetic_sequence a d 4 = 10)
  (h6 : arithmetic_sequence a d 5 = 14)
  (h7 : arithmetic_sequence a d 6 = 18) :
  (arithmetic_sequence a d 0) +
  (arithmetic_sequence a d 1) +
  (arithmetic_sequence a d 2) +
  (arithmetic_sequence a d 3) = 0 :=
by sorry

end sum_first_four_terms_l908_90837


namespace odd_function_property_l908_90866

-- Define an odd function f on ℝ
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h1 : f 1 = 2) 
  (h2 : f 2 = 3) : 
  f (f (-1)) = -3 := by
  sorry


end odd_function_property_l908_90866


namespace bee_colony_loss_rate_l908_90832

/-- Proves that given a colony of bees with an initial population of 80,000 individuals,
    if after 50 days the population reduces to one-fourth of its initial size,
    then the daily loss rate is 1,200 bees per day. -/
theorem bee_colony_loss_rate (initial_population : ℕ) (days : ℕ) (final_population : ℕ) :
  initial_population = 80000 →
  days = 50 →
  final_population = initial_population / 4 →
  (initial_population - final_population) / days = 1200 := by
  sorry

end bee_colony_loss_rate_l908_90832


namespace triangle_inequality_l908_90891

theorem triangle_inequality (a b c r s : ℝ) :
  a > 0 → b > 0 → c > 0 → r > 0 →
  s = (a + b + c) / 2 →
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  1 / (s - a)^2 + 1 / (s - b)^2 + 1 / (s - c)^2 ≥ 1 / r^2 := by
sorry


end triangle_inequality_l908_90891


namespace root_product_theorem_l908_90831

theorem root_product_theorem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a^2 + 1/b)^2 - p*(a^2 + 1/b) + r = 0) →
  ((b^2 + 1/a)^2 - p*(b^2 + 1/a) + r = 0) →
  r = 46/3 := by
sorry

end root_product_theorem_l908_90831


namespace solve_for_x_l908_90890

theorem solve_for_x (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 14) : x = 12 := by
  sorry

end solve_for_x_l908_90890


namespace inverse_of_A_squared_l908_90822

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A⁻¹ = !![3, 4; -2, -2]) : 
  (A^2)⁻¹ = !![1, 4; -2, -4] := by
sorry

end inverse_of_A_squared_l908_90822


namespace triangle_tangent_product_l908_90826

theorem triangle_tangent_product (A B C : Real) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  A + B + C = π →
  (Real.cos A)^2 + (Real.cos B)^2 + (Real.cos C)^2 = (Real.sin B)^2 →
  (Real.tan A) * (Real.tan C) = 3 := by
  sorry

end triangle_tangent_product_l908_90826


namespace markup_constant_l908_90886

theorem markup_constant (C S : ℝ) (k : ℝ) (hk : k > 0) (hC : C > 0) (hS : S > 0) : 
  (S = C + k * S) → (k * S = 0.25 * C) → k = 1/5 := by
sorry

end markup_constant_l908_90886


namespace quadratic_equation_from_roots_l908_90801

theorem quadratic_equation_from_roots (x₁ x₂ : ℝ) (hx₁ : x₁ = 3) (hx₂ : x₂ = -4) :
  ∃ a b c : ℝ, a ≠ 0 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ (x - x₁) * (x - x₂) = 0) :=
sorry

end quadratic_equation_from_roots_l908_90801


namespace stephanie_remaining_payment_l908_90865

/-- Represents the bills and payments in Stephanie's household budget --/
structure BudgetInfo where
  electricity_bill : ℝ
  gas_bill : ℝ
  water_bill : ℝ
  internet_bill : ℝ
  gas_initial_payment_fraction : ℝ
  gas_additional_payment : ℝ
  water_payment_fraction : ℝ
  internet_payment_count : ℕ
  internet_payment_amount : ℝ

/-- Calculates the remaining amount to pay given the budget information --/
def remaining_payment (budget : BudgetInfo) : ℝ :=
  let total_bills := budget.electricity_bill + budget.gas_bill + budget.water_bill + budget.internet_bill
  let total_paid := budget.electricity_bill +
                    (budget.gas_bill * budget.gas_initial_payment_fraction + budget.gas_additional_payment) +
                    (budget.water_bill * budget.water_payment_fraction) +
                    (budget.internet_payment_count : ℝ) * budget.internet_payment_amount
  total_bills - total_paid

/-- Theorem stating that the remaining payment for Stephanie's bills is $30 --/
theorem stephanie_remaining_payment :
  let budget : BudgetInfo := {
    electricity_bill := 60,
    gas_bill := 40,
    water_bill := 40,
    internet_bill := 25,
    gas_initial_payment_fraction := 0.75,
    gas_additional_payment := 5,
    water_payment_fraction := 0.5,
    internet_payment_count := 4,
    internet_payment_amount := 5
  }
  remaining_payment budget = 30 := by sorry

end stephanie_remaining_payment_l908_90865


namespace exchange_problem_l908_90851

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- Represents the exchange problem and proves the sum of digits -/
theorem exchange_problem (d : ℕ) : 
  (11 * d : ℚ) / 8 - 70 = d → sumOfDigits d = 16 := by
  sorry

#eval sumOfDigits 187  -- Expected output: 16

end exchange_problem_l908_90851


namespace equation_solution_sum_l908_90873

theorem equation_solution_sum (a : ℝ) (h : a ≥ 1) :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt (a - Real.sqrt (a + x)) = x ∧
  (∀ y : ℝ, y ≥ 0 ∧ Real.sqrt (a - Real.sqrt (a + y)) = y → y = x) ∧
  x = (Real.sqrt (4 * a - 3) - 1) / 2 :=
by sorry

end equation_solution_sum_l908_90873


namespace auditorium_seats_cost_l908_90810

theorem auditorium_seats_cost 
  (rows : ℕ) 
  (seats_per_row : ℕ) 
  (cost_per_seat : ℕ) 
  (discount_rate : ℚ) 
  (seats_per_discount_group : ℕ) : 
  rows = 5 → 
  seats_per_row = 8 → 
  cost_per_seat = 30 → 
  discount_rate = 1/10 → 
  seats_per_discount_group = 10 → 
  (rows * seats_per_row * cost_per_seat : ℚ) - 
    ((rows * seats_per_row / seats_per_discount_group : ℚ) * 
     (seats_per_discount_group * cost_per_seat * discount_rate)) = 1080 := by
  sorry

end auditorium_seats_cost_l908_90810


namespace time_until_800_l908_90874

def minutes_since_730 : ℕ := 16

def current_time : ℕ := 7 * 60 + 30 + minutes_since_730

def target_time : ℕ := 8 * 60

theorem time_until_800 : target_time - current_time = 14 := by
  sorry

end time_until_800_l908_90874


namespace train_probability_is_half_l908_90848

-- Define the time interval (in minutes)
def timeInterval : ℝ := 60

-- Define the waiting time of the train (in minutes)
def waitingTime : ℝ := 30

-- Define a function to calculate the probability
noncomputable def trainProbability : ℝ :=
  let triangleArea := (1 / 2) * waitingTime * waitingTime
  let trapezoidArea := (1 / 2) * (waitingTime + timeInterval) * (timeInterval - waitingTime)
  (triangleArea + trapezoidArea) / (timeInterval * timeInterval)

-- Theorem statement
theorem train_probability_is_half :
  trainProbability = 1 / 2 :=
by sorry

end train_probability_is_half_l908_90848


namespace minimum_balls_to_draw_thirty_eight_sufficient_l908_90854

/-- Represents the number of balls of each color in the bag -/
structure BagContents :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (other : ℕ)

/-- Represents the configuration of drawn balls -/
structure DrawnBalls :=
  (red : ℕ)
  (blue : ℕ)
  (yellow : ℕ)
  (other : ℕ)

/-- Check if a given configuration of drawn balls satisfies the condition -/
def satisfiesCondition (drawn : DrawnBalls) : Prop :=
  drawn.red ≥ 10 ∨ drawn.blue ≥ 10 ∨ drawn.yellow ≥ 10

/-- Check if it's possible to draw a given configuration from the bag -/
def canDraw (bag : BagContents) (drawn : DrawnBalls) : Prop :=
  drawn.red ≤ bag.red ∧
  drawn.blue ≤ bag.blue ∧
  drawn.yellow ≤ bag.yellow ∧
  drawn.other ≤ bag.other ∧
  drawn.red + drawn.blue + drawn.yellow + drawn.other ≤ bag.red + bag.blue + bag.yellow + bag.other

theorem minimum_balls_to_draw (bag : BagContents)
  (h1 : bag.red = 20)
  (h2 : bag.blue = 20)
  (h3 : bag.yellow = 20)
  (h4 : bag.other = 10) :
  ∀ n : ℕ, n < 38 →
    ∃ drawn : DrawnBalls, canDraw bag drawn ∧ ¬satisfiesCondition drawn ∧ drawn.red + drawn.blue + drawn.yellow + drawn.other = n :=
by sorry

theorem thirty_eight_sufficient (bag : BagContents)
  (h1 : bag.red = 20)
  (h2 : bag.blue = 20)
  (h3 : bag.yellow = 20)
  (h4 : bag.other = 10) :
  ∀ drawn : DrawnBalls, canDraw bag drawn → drawn.red + drawn.blue + drawn.yellow + drawn.other = 38 →
    satisfiesCondition drawn :=
by sorry

end minimum_balls_to_draw_thirty_eight_sufficient_l908_90854


namespace exam_logic_l908_90824

structure Student where
  name : String
  score : ℝ
  grade : String

def exam_rule (s : Student) : Prop :=
  s.score ≥ 0.8 → s.grade = "A"

theorem exam_logic (s : Student) (h : exam_rule s) :
  (s.grade ≠ "A" → s.score < 0.8) ∧
  (s.score ≥ 0.8 → s.grade = "A") := by
  sorry

end exam_logic_l908_90824


namespace square_sum_equals_21_l908_90882

theorem square_sum_equals_21 (x y : ℝ) (h1 : (x + y)^2 = 9) (h2 : x * y = -6) :
  x^2 + y^2 = 21 := by
  sorry

end square_sum_equals_21_l908_90882


namespace toilet_paper_weeks_l908_90849

/-- The number of bathrooms in the bed and breakfast -/
def num_bathrooms : ℕ := 6

/-- The number of rolls Stella stocks per bathroom per day -/
def rolls_per_bathroom_per_day : ℕ := 1

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of rolls in a pack (1 dozen) -/
def rolls_per_pack : ℕ := 12

/-- The number of packs Stella buys -/
def packs_bought : ℕ := 14

/-- The number of weeks Stella bought toilet paper for -/
def weeks_bought : ℚ :=
  (packs_bought * rolls_per_pack) / (num_bathrooms * rolls_per_bathroom_per_day * days_per_week)

theorem toilet_paper_weeks : weeks_bought = 4 := by sorry

end toilet_paper_weeks_l908_90849


namespace min_value_expression_min_value_equality_l908_90892

theorem min_value_expression (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (hx₁ : x₁ ≥ 0) (hx₂ : x₂ ≥ 0) (hx₃ : x₃ ≥ 0) 
  (hy₁ : y₁ ≥ 0) (hy₂ : y₂ ≥ 0) (hy₃ : y₃ ≥ 0) : 
  Real.sqrt ((2018 - y₁ - y₂ - y₃)^2 + x₃^2) + 
  Real.sqrt (y₃^2 + x₂^2) + 
  Real.sqrt (y₂^2 + x₁^2) + 
  Real.sqrt (y₁^2 + (x₁ + x₂ + x₃)^2) ≥ 2018 :=
by sorry

theorem min_value_equality (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) : 
  (x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0 ∧ y₁ = 0 ∧ y₂ = 0 ∧ y₃ = 0) → 
  Real.sqrt ((2018 - y₁ - y₂ - y₃)^2 + x₃^2) + 
  Real.sqrt (y₃^2 + x₂^2) + 
  Real.sqrt (y₂^2 + x₁^2) + 
  Real.sqrt (y₁^2 + (x₁ + x₂ + x₃)^2) = 2018 :=
by sorry

end min_value_expression_min_value_equality_l908_90892


namespace max_min_difference_l908_90863

theorem max_min_difference (a b c d : ℕ+) 
  (h1 : a + b = 20)
  (h2 : a + c = 24)
  (h3 : a + d = 22) : 
  (Nat.max (a + b + c + d) (a + b + c + d) : ℤ) - 
  (Nat.min (a + b + c + d) (a + b + c + d) : ℤ) = 36 :=
by sorry

end max_min_difference_l908_90863


namespace quadratic_equation_solution_l908_90888

theorem quadratic_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 3 * x^2 + 8 * x - 16 = 0 :=
by
  use 4/3
  sorry

end quadratic_equation_solution_l908_90888


namespace inequality_solution_range_l908_90895

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x^2 + a*x + 4 < 0) → (a < -4 ∨ a > 4) := by
  sorry

end inequality_solution_range_l908_90895


namespace total_profit_is_100_l908_90856

/-- Calculates the total profit given investments, time periods, and A's share --/
def calculate_total_profit (a_investment : ℕ) (a_months : ℕ) (b_investment : ℕ) (b_months : ℕ) (a_share : ℕ) : ℕ :=
  let a_weight := a_investment * a_months
  let b_weight := b_investment * b_months
  let total_weight := a_weight + b_weight
  let part_value := a_share * total_weight / a_weight
  part_value

theorem total_profit_is_100 :
  calculate_total_profit 150 12 200 6 60 = 100 := by
  sorry

end total_profit_is_100_l908_90856


namespace sum_of_three_consecutive_odd_divisible_by_three_l908_90889

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

theorem sum_of_three_consecutive_odd_divisible_by_three : 
  ∀ (a b c : ℕ), 
    (is_odd a ∧ is_odd b ∧ is_odd c) → 
    (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0) →
    (∃ k, b = a + 6*k + 6 ∧ c = b + 6) →
    (c = 27) →
    (a + b + c = 63) := by
  sorry

end sum_of_three_consecutive_odd_divisible_by_three_l908_90889


namespace value_of_M_l908_90853

theorem value_of_M : ∃ M : ℝ, (0.3 * M = 0.6 * 500) ∧ (M = 1000) := by
  sorry

end value_of_M_l908_90853


namespace perpendicular_lines_a_value_l908_90833

/-- Two lines y = ax - 2 and y = (a+2)x + 1 are perpendicular -/
def are_perpendicular (a : ℝ) : Prop :=
  a * (a + 2) + 1 = 0

/-- Theorem: If the lines y = ax - 2 and y = (a+2)x + 1 are perpendicular, then a = -1 -/
theorem perpendicular_lines_a_value :
  ∀ a : ℝ, are_perpendicular a → a = -1 := by
  sorry

end perpendicular_lines_a_value_l908_90833


namespace base6_addition_l908_90862

/-- Converts a base 6 number represented as a list of digits to its decimal (base 10) equivalent -/
def base6ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

/-- Converts a decimal (base 10) number to its base 6 representation as a list of digits -/
def decimalToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

/-- The main theorem stating that 3454₆ + 12345₆ = 142042₆ in base 6 -/
theorem base6_addition :
  decimalToBase6 (base6ToDecimal [3, 4, 5, 4] + base6ToDecimal [1, 2, 3, 4, 5]) =
  [1, 4, 2, 0, 4, 2] := by
  sorry

end base6_addition_l908_90862


namespace triangle_altitude_length_l908_90899

/-- Given a rectangle with sides a and b, and a triangle with its base as the diagonal of the rectangle
    and area twice that of the rectangle, the length of the altitude of the triangle to its base
    (the diagonal) is (4ab)/√(a² + b²). -/
theorem triangle_altitude_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let rectangle_area := a * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let triangle_area := 2 * rectangle_area
  let altitude := (2 * triangle_area) / diagonal
  altitude = (4 * a * b) / Real.sqrt (a^2 + b^2) := by
  sorry

end triangle_altitude_length_l908_90899
