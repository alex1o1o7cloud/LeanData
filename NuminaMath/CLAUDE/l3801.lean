import Mathlib

namespace smallest_perfect_cube_l3801_380158

theorem smallest_perfect_cube (Z K : ℤ) : 
  (2000 < Z) → (Z < 3000) → (K > 1) → (Z = K * K^2) → 
  (∃ n : ℤ, Z = n^3) → 
  (∀ K' : ℤ, K' < K → ¬(2000 < K'^3 ∧ K'^3 < 3000)) →
  K = 13 := by
sorry

end smallest_perfect_cube_l3801_380158


namespace min_distance_four_points_l3801_380105

/-- Given four points in a metric space with specified distances between consecutive points,
    the theorem states that the minimum possible distance between the first and last points is 3. -/
theorem min_distance_four_points (X : Type*) [MetricSpace X] (P Q R S : X) :
  dist P Q = 12 →
  dist Q R = 7 →
  dist R S = 2 →
  ∃ (configuration : X → X), dist (configuration P) (configuration S) = 3 ∧
  (∀ (P' Q' R' S' : X),
    dist P' Q' = 12 →
    dist Q' R' = 7 →
    dist R' S' = 2 →
    dist P' S' ≥ 3) :=
by sorry

end min_distance_four_points_l3801_380105


namespace plane_division_l3801_380182

/-- The maximum number of parts a plane can be divided into by n lines -/
def f (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- Theorem: The maximum number of parts a plane can be divided into by n lines is (n^2 + n + 2) / 2 -/
theorem plane_division (n : ℕ) : f n = (n^2 + n + 2) / 2 := by
  sorry

end plane_division_l3801_380182


namespace geometric_sequence_fourth_term_l3801_380130

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_product : a 1 * a 7 = 3/4) :
  a 4 = Real.sqrt 3 / 2 := by
sorry

end geometric_sequence_fourth_term_l3801_380130


namespace shekars_social_studies_score_l3801_380162

theorem shekars_social_studies_score 
  (math_score science_score english_score biology_score : ℕ)
  (average_score : ℚ)
  (total_subjects : ℕ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 67)
  (h4 : biology_score = 85)
  (h5 : average_score = 75)
  (h6 : total_subjects = 5) :
  ∃ (social_studies_score : ℕ),
    social_studies_score = 82 ∧
    (math_score + science_score + english_score + biology_score + social_studies_score : ℚ) / total_subjects = average_score :=
by
  sorry

end shekars_social_studies_score_l3801_380162


namespace difference_of_squares_701_697_l3801_380168

theorem difference_of_squares_701_697 : 701^2 - 697^2 = 5592 := by
  sorry

end difference_of_squares_701_697_l3801_380168


namespace other_root_of_quadratic_l3801_380172

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 + m*x - 5 = 0 ∧ x = -1) → 
  (∃ y : ℝ, y^2 + m*y - 5 = 0 ∧ y = 5) :=
sorry

end other_root_of_quadratic_l3801_380172


namespace subtracted_number_l3801_380194

theorem subtracted_number (m n x : ℕ) : 
  m > 0 → n > 0 → m = 15 * n - x → m % 5 = 4 → x = 1 := by
  sorry

end subtracted_number_l3801_380194


namespace money_value_difference_l3801_380129

theorem money_value_difference (exchange_rate : ℝ) (marco_dollars : ℝ) (juliette_euros : ℝ) :
  exchange_rate = 1.5 →
  marco_dollars = 600 →
  juliette_euros = 350 →
  let juliette_dollars := juliette_euros * exchange_rate
  (marco_dollars - juliette_dollars) / marco_dollars * 100 = 12.5 := by
  sorry

end money_value_difference_l3801_380129


namespace physics_marks_l3801_380140

theorem physics_marks (total_average : ℝ) (phys_math_avg : ℝ) (phys_chem_avg : ℝ)
  (h1 : total_average = 65)
  (h2 : phys_math_avg = 90)
  (h3 : phys_chem_avg = 70) :
  ∃ (physics chemistry mathematics : ℝ),
    physics + chemistry + mathematics = 3 * total_average ∧
    physics + mathematics = 2 * phys_math_avg ∧
    physics + chemistry = 2 * phys_chem_avg ∧
    physics = 125 := by
  sorry

end physics_marks_l3801_380140


namespace min_value_of_a_l3801_380142

theorem min_value_of_a (a b : ℤ) : 
  a < 26 → 
  b > 14 → 
  b < 31 → 
  (a : ℚ) / (b : ℚ) ≥ 4/3 → 
  a ≥ 20 :=
sorry

end min_value_of_a_l3801_380142


namespace shaded_area_calculation_l3801_380115

-- Define the radius of the larger circle
def R : ℝ := 10

-- Define the radius of the smaller circles
def r : ℝ := 5

-- Theorem statement
theorem shaded_area_calculation :
  let larger_circle_area := π * R^2
  let smaller_circle_area := π * r^2
  let shaded_area := larger_circle_area - 2 * smaller_circle_area
  shaded_area = 50 * π := by
  sorry

end shaded_area_calculation_l3801_380115


namespace triangle_tangent_identity_l3801_380126

theorem triangle_tangent_identity (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) :
  Real.tan (A/2) * Real.tan (B/2) + Real.tan (B/2) * Real.tan (C/2) + Real.tan (C/2) * Real.tan (A/2) = 1 := by
  sorry

end triangle_tangent_identity_l3801_380126


namespace perfect_square_solution_l3801_380119

theorem perfect_square_solution : 
  ∃! (n : ℤ), ∃ (m : ℤ), n^2 + 20*n + 11 = m^2 :=
by
  -- The unique solution is n = 35
  use 35
  sorry

end perfect_square_solution_l3801_380119


namespace trigonometric_identity_l3801_380155

theorem trigonometric_identity : 
  (Real.sin (110 * π / 180) * Real.sin (20 * π / 180)) / 
  (Real.cos (155 * π / 180)^2 - Real.sin (155 * π / 180)^2) = 1/2 := by
sorry

end trigonometric_identity_l3801_380155


namespace function_min_value_l3801_380141

/-- Given constants a and b, and a function f with specific properties, 
    prove that its minimum value on (0, +∞) is -4 -/
theorem function_min_value 
  (a b : ℝ) 
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x^3 + b * Real.log (x + Real.sqrt (1 + x^2)) + 3)
  (h_max : ∀ x < 0, f x ≤ 10)
  (h_exists_max : ∃ x < 0, f x = 10) :
  ∃ y > 0, ∀ x > 0, f x ≥ f y ∧ f y = -4 := by
sorry

end function_min_value_l3801_380141


namespace gcd_98_63_l3801_380133

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l3801_380133


namespace total_musicians_l3801_380147

/-- Represents a musical group with a specific number of male and female musicians. -/
structure MusicGroup where
  males : Nat
  females : Nat

/-- The total number of musicians in a group is the sum of males and females. -/
def MusicGroup.total (g : MusicGroup) : Nat :=
  g.males + g.females

/-- The orchestra has 11 males and 12 females. -/
def orchestra : MusicGroup :=
  { males := 11, females := 12 }

/-- The band has twice the number of musicians as the orchestra. -/
def band : MusicGroup :=
  { males := 2 * orchestra.males, females := 2 * orchestra.females }

/-- The choir has 12 males and 17 females. -/
def choir : MusicGroup :=
  { males := 12, females := 17 }

/-- Theorem: The total number of musicians in the orchestra, band, and choir is 98. -/
theorem total_musicians :
  orchestra.total + band.total + choir.total = 98 := by
  sorry

end total_musicians_l3801_380147


namespace player_a_wins_iff_perfect_square_l3801_380163

/-- The divisor erasing game on a positive integer N -/
def DivisorGame (N : ℕ+) :=
  ∀ (d : ℕ+), d ∣ N → (∃ (m : ℕ+), m ∣ N ∧ (d ∣ m ∨ m ∣ d))

/-- Player A's winning condition -/
def PlayerAWins (N : ℕ+) :=
  ∀ (strategy : ℕ+ → ℕ+),
    (∀ (d : ℕ+), d ∣ N → strategy d ∣ N ∧ (d ∣ strategy d ∨ strategy d ∣ d)) →
    ∃ (move : ℕ+ → ℕ+), 
      (∀ (d : ℕ+), d ∣ N → move d ∣ N ∧ (d ∣ move d ∨ move d ∣ d)) ∧
      (∀ (d : ℕ+), d ∣ N → move (strategy (move d)) ≠ d)

/-- The main theorem: Player A wins if and only if N is a perfect square -/
theorem player_a_wins_iff_perfect_square (N : ℕ+) :
  PlayerAWins N ↔ ∃ (n : ℕ+), N = n * n :=
sorry

end player_a_wins_iff_perfect_square_l3801_380163


namespace sin_cos_power_inequality_l3801_380184

theorem sin_cos_power_inequality (x : Real) (h : 0 < x ∧ x < Real.pi / 4) :
  (Real.sin x) ^ (Real.sin x) < (Real.cos x) ^ (Real.cos x) := by
  sorry

end sin_cos_power_inequality_l3801_380184


namespace framed_painting_ratio_l3801_380149

/-- Represents the dimensions and frame properties of a painting --/
structure FramedPainting where
  width : ℝ
  height : ℝ
  side_frame_width : ℝ

/-- Calculates the dimensions of the framed painting --/
def framedDimensions (p : FramedPainting) : ℝ × ℝ :=
  (p.width + 2 * p.side_frame_width, p.height + 6 * p.side_frame_width)

/-- Calculates the area of the framed painting --/
def framedArea (p : FramedPainting) : ℝ :=
  let (w, h) := framedDimensions p
  w * h

/-- Calculates the area of the original painting --/
def paintingArea (p : FramedPainting) : ℝ :=
  p.width * p.height

/-- Theorem statement for the framed painting problem --/
theorem framed_painting_ratio (p : FramedPainting)
  (h1 : p.width = 20)
  (h2 : p.height = 30)
  (h3 : framedArea p = 2 * paintingArea p) :
  let (w, h) := framedDimensions p
  w / h = 1 / 2 := by
  sorry

#check framed_painting_ratio

end framed_painting_ratio_l3801_380149


namespace first_group_number_from_sixteenth_l3801_380164

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  population_size : ℕ
  sample_size : ℕ
  group_size : ℕ
  first_group_number : ℕ
  sixteenth_group_number : ℕ

/-- Theorem stating the relationship between the 1st and 16th group numbers in the given systematic sampling -/
theorem first_group_number_from_sixteenth
  (s : SystematicSampling)
  (h1 : s.population_size = 160)
  (h2 : s.sample_size = 20)
  (h3 : s.group_size = 8)
  (h4 : s.sixteenth_group_number = 126) :
  s.first_group_number = 6 := by
  sorry

end first_group_number_from_sixteenth_l3801_380164


namespace problem_solution_l3801_380150

theorem problem_solution (x : ℝ) (h : x^2 - 3*x - 1 = 0) : -3*x^2 + 9*x + 4 = 1 := by
  sorry

end problem_solution_l3801_380150


namespace tan_sum_of_quadratic_roots_l3801_380193

theorem tan_sum_of_quadratic_roots (α β : Real) (h : ∀ x, x^2 + 6*x + 7 = 0 ↔ x = Real.tan α ∨ x = Real.tan β) :
  Real.tan (α + β) = 1 := by
  sorry

end tan_sum_of_quadratic_roots_l3801_380193


namespace divisibility_implication_l3801_380127

theorem divisibility_implication (n m : ℤ) : 
  (31 ∣ (6 * n + 11 * m)) → (31 ∣ (n + 7 * m)) := by
sorry

end divisibility_implication_l3801_380127


namespace product_of_symmetric_complex_l3801_380103

/-- Two complex numbers are symmetric about the angle bisector of the first and third quadrants if their real and imaginary parts are interchanged. -/
def symmetric_about_bisector (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

theorem product_of_symmetric_complex : ∀ z₁ z₂ : ℂ,
  symmetric_about_bisector z₁ z₂ → z₁ = 1 + 2*I → z₁ * z₂ = 5*I :=
by sorry

end product_of_symmetric_complex_l3801_380103


namespace complex_equation_sum_l3801_380114

theorem complex_equation_sum (a b : ℝ) : 
  (Complex.mk a b = (2 * Complex.I) / (1 + Complex.I)) → a + b = 2 := by
  sorry

end complex_equation_sum_l3801_380114


namespace inequality_equivalence_l3801_380161

theorem inequality_equivalence (x : ℝ) :
  (x - 3) / ((x - 4)^2 + 8) ≥ 0 ↔ x ≥ 3 := by
  sorry

end inequality_equivalence_l3801_380161


namespace angle_1120_in_first_quadrant_l3801_380109

/-- An angle is in the first quadrant if its equivalent angle in [0, 360) is between 0 and 90 degrees. -/
def in_first_quadrant (angle : ℝ) : Prop :=
  0 ≤ (angle % 360) ∧ (angle % 360) < 90

/-- Theorem stating that 1120 degrees is in the first quadrant -/
theorem angle_1120_in_first_quadrant : in_first_quadrant 1120 := by
  sorry

end angle_1120_in_first_quadrant_l3801_380109


namespace order_of_expressions_l3801_380165

theorem order_of_expressions :
  let a := (1/3 : ℝ) ^ Real.pi
  let b := (1/3 : ℝ) ^ (1/2)
  let c := Real.pi ^ (1/2)
  c > b ∧ b > a := by sorry

end order_of_expressions_l3801_380165


namespace trapezoid_area_difference_l3801_380199

/-- A trapezoid with specific properties -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  right_angles : ℕ
  
/-- The area difference between largest and smallest regions -/
def area_difference (t : Trapezoid) : ℝ := sorry

theorem trapezoid_area_difference :
  ∀ t : Trapezoid,
    t.side1 = 4 ∧ 
    t.side2 = 4 ∧ 
    t.side3 = 5 ∧ 
    t.side4 = Real.sqrt 17 ∧
    t.right_angles = 2 →
    240 * (area_difference t) = 240 :=
by sorry

end trapezoid_area_difference_l3801_380199


namespace simplify_expression_l3801_380169

theorem simplify_expression (x y : ℚ) (hx : x = 3) (hy : y = 2) :
  12 * x^3 * y^4 / (9 * x^2 * y^3) = 8 := by
  sorry

end simplify_expression_l3801_380169


namespace greater_number_problem_l3801_380192

theorem greater_number_problem (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 16) : max x y = 33 := by
  sorry

end greater_number_problem_l3801_380192


namespace escalator_walking_speed_l3801_380136

/-- Proves that a person walks at 5 ft/sec on an escalator given specific conditions -/
theorem escalator_walking_speed 
  (escalator_speed : ℝ) 
  (escalator_length : ℝ) 
  (time_taken : ℝ) 
  (h1 : escalator_speed = 15) 
  (h2 : escalator_length = 200) 
  (h3 : time_taken = 10) : 
  ∃ (walking_speed : ℝ), 
    walking_speed = 5 ∧ 
    escalator_length = (walking_speed + escalator_speed) * time_taken :=
by
  sorry

end escalator_walking_speed_l3801_380136


namespace large_circle_radius_large_circle_radius_value_l3801_380120

/-- The radius of a circle that internally touches two circles of radius 2 and both internally
    and externally touches a third circle of radius 2 (where all three smaller circles are
    externally tangent to each other) is equal to 4 + 2√3. -/
theorem large_circle_radius : ℝ → ℝ → Prop :=
  fun (small_radius large_radius : ℝ) =>
    small_radius = 2 ∧
    (∃ (centers : Fin 3 → ℝ × ℝ) (large_center : ℝ × ℝ),
      (∀ i j, i ≠ j → dist (centers i) (centers j) = 2 * small_radius) ∧
      (∀ i, dist (centers i) large_center ≤ large_radius + small_radius) ∧
      (∃ k, dist (centers k) large_center = large_radius - small_radius) ∧
      (∃ l, dist (centers l) large_center = large_radius + small_radius)) →
    large_radius = 4 + 2 * Real.sqrt 3

theorem large_circle_radius_value : large_circle_radius 2 (4 + 2 * Real.sqrt 3) := by
  sorry

end large_circle_radius_large_circle_radius_value_l3801_380120


namespace field_resizing_problem_l3801_380198

theorem field_resizing_problem : ∃ m : ℝ, 
  m > 0 ∧ (3 * m + 14) * (m + 1) = 240 ∧ abs (m - 6.3) < 0.01 := by
  sorry

end field_resizing_problem_l3801_380198


namespace smallest_marble_collection_l3801_380190

theorem smallest_marble_collection : ∀ n : ℕ,
  (n % 4 = 0) →  -- one fourth are red
  (n % 5 = 0) →  -- one fifth are blue
  (n ≥ 8 + 5) →  -- at least 8 white and 5 green
  (∃ r b w g : ℕ, 
    r + b + w + g = n ∧
    r = n / 4 ∧
    b = n / 5 ∧
    w = 8 ∧
    g = 5) →
  n ≥ 220 :=
by
  sorry

end smallest_marble_collection_l3801_380190


namespace triangle_area_bounds_l3801_380187

/-- Given a triangle with sides a, b, c, this theorem proves bounds on its area S. -/
theorem triangle_area_bounds (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := S / p
  3 * Real.sqrt 3 * r^2 ≤ S ∧ S ≤ p^2 / (3 * Real.sqrt 3) ∧
  S ≤ (a^2 + b^2 + c^2) / (4 * Real.sqrt 3) := by
  sorry

end triangle_area_bounds_l3801_380187


namespace video_rental_percentage_l3801_380174

theorem video_rental_percentage (a : ℕ) : 
  let action := a
  let drama := 5 * a
  let comedy := 10 * a
  let total := action + drama + comedy
  (comedy : ℚ) / total * 100 = 62.5 := by
sorry

end video_rental_percentage_l3801_380174


namespace lucky_iff_power_of_two_l3801_380101

/-- Represents the three colors of cubes -/
inductive Color
  | White
  | Blue
  | Red

/-- Represents an arrangement of N cubes in a circle -/
def Arrangement (N : ℕ) := Fin N → Color

/-- Determines if an arrangement is good (final cube color doesn't depend on starting position) -/
def is_good (N : ℕ) (arr : Arrangement N) : Prop := sorry

/-- Determines if N is lucky (all arrangements of N cubes are good) -/
def is_lucky (N : ℕ) : Prop :=
  ∀ arr : Arrangement N, is_good N arr

/-- Main theorem: N is lucky if and only if it's a power of 2 -/
theorem lucky_iff_power_of_two (N : ℕ) :
  is_lucky N ↔ ∃ k : ℕ, N = 2^k :=
sorry

end lucky_iff_power_of_two_l3801_380101


namespace employees_with_advanced_degrees_l3801_380153

/-- Proves that the number of employees with advanced degrees is 78 given the conditions in the problem -/
theorem employees_with_advanced_degrees :
  ∀ (total_employees : ℕ) 
    (female_employees : ℕ) 
    (male_college_only : ℕ) 
    (female_advanced : ℕ),
  total_employees = 148 →
  female_employees = 92 →
  male_college_only = 31 →
  female_advanced = 53 →
  ∃ (male_advanced : ℕ),
    male_advanced + female_advanced + male_college_only + (female_employees - female_advanced) = total_employees ∧
    male_advanced + female_advanced = 78 :=
by sorry

end employees_with_advanced_degrees_l3801_380153


namespace units_digit_47_power_47_l3801_380178

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the theorem
theorem units_digit_47_power_47 : unitsDigit (47^47) = 3 := by
  sorry

end units_digit_47_power_47_l3801_380178


namespace father_ate_eight_brownies_l3801_380197

/-- The number of brownies Father ate -/
def fatherAte (initialBrownies : ℕ) (mooneyAte : ℕ) (additionalBrownies : ℕ) (finalBrownies : ℕ) : ℕ :=
  initialBrownies + additionalBrownies - mooneyAte - finalBrownies

/-- Proves that Father ate 8 brownies given the problem conditions -/
theorem father_ate_eight_brownies :
  fatherAte (2 * 12) 4 (2 * 12) 36 = 8 := by
  sorry

#eval fatherAte (2 * 12) 4 (2 * 12) 36

end father_ate_eight_brownies_l3801_380197


namespace sin_sum_arcsin_arctan_l3801_380125

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan 3) = 13 * Real.sqrt 10 / 50 := by sorry

end sin_sum_arcsin_arctan_l3801_380125


namespace carl_driving_hours_l3801_380146

theorem carl_driving_hours :
  let daily_hours : ℕ := 2
  let additional_weekly_hours : ℕ := 6
  let days_in_two_weeks : ℕ := 14
  let weeks : ℕ := 2
  (daily_hours * days_in_two_weeks) + (additional_weekly_hours * weeks) = 40 :=
by sorry

end carl_driving_hours_l3801_380146


namespace jenny_lasagna_profit_l3801_380106

/-- Calculates the profit for Jenny's lasagna business -/
def lasagna_profit (cost_per_pan : ℝ) (num_pans : ℕ) (price_per_pan : ℝ) : ℝ :=
  num_pans * price_per_pan - num_pans * cost_per_pan

/-- Proves that Jenny's profit is $300.00 given the specified conditions -/
theorem jenny_lasagna_profit :
  lasagna_profit 10 20 25 = 300 :=
by sorry

end jenny_lasagna_profit_l3801_380106


namespace simplified_expression_terms_l3801_380160

-- Define the exponent
def n : ℕ := 2008

-- Define the function to count terms
def countTerms (n : ℕ) : ℕ :=
  (n / 2 + 1) * (n + 1)

-- Theorem statement
theorem simplified_expression_terms :
  countTerms n = 2018045 :=
sorry

end simplified_expression_terms_l3801_380160


namespace cost_per_side_of_square_park_l3801_380159

/-- Represents the cost of fencing a square park -/
def CostOfFencing : Type :=
  { total : ℕ // total > 0 }

/-- Calculates the cost of fencing each side of a square park -/
def costPerSide (c : CostOfFencing) : ℕ :=
  c.val / 4

/-- Theorem: The cost of fencing each side of a square park is 43 dollars,
    given that the total cost of fencing is 172 dollars -/
theorem cost_per_side_of_square_park :
  ∀ (c : CostOfFencing), c.val = 172 → costPerSide c = 43 := by
  sorry

end cost_per_side_of_square_park_l3801_380159


namespace number_square_equation_l3801_380128

theorem number_square_equation : ∃ x : ℝ, x^2 + 145 = (x - 19)^2 ∧ x = 108/19 := by
  sorry

end number_square_equation_l3801_380128


namespace contact_lenses_sold_l3801_380143

/-- Represents the number of pairs of hard contact lenses sold -/
def hard_lenses : ℕ := sorry

/-- Represents the number of pairs of soft contact lenses sold -/
def soft_lenses : ℕ := sorry

/-- The price of a pair of hard contact lenses in cents -/
def hard_price : ℕ := 8500

/-- The price of a pair of soft contact lenses in cents -/
def soft_price : ℕ := 15000

/-- The total sales in cents -/
def total_sales : ℕ := 145500

theorem contact_lenses_sold :
  (soft_lenses = hard_lenses + 5) →
  (hard_price * hard_lenses + soft_price * soft_lenses = total_sales) →
  (hard_lenses + soft_lenses = 11) := by
  sorry

end contact_lenses_sold_l3801_380143


namespace log_expression_equals_three_l3801_380179

theorem log_expression_equals_three :
  (Real.log 243 / Real.log 3) / (Real.log 27 / Real.log 3) -
  (Real.log 729 / Real.log 3) / (Real.log 9 / Real.log 3) = 3 := by
  sorry

end log_expression_equals_three_l3801_380179


namespace range_of_b_l3801_380195

theorem range_of_b (a b c : ℝ) (sum_cond : a + b + c = 9) (prod_cond : a * b + b * c + c * a = 24) :
  1 ≤ b ∧ b ≤ 5 :=
sorry

end range_of_b_l3801_380195


namespace exactly_two_approve_probability_l3801_380100

def approval_rate : ℝ := 0.8
def num_voters : ℕ := 4
def num_approving : ℕ := 2

def probability_exactly_two_approve : ℝ := 
  (Nat.choose num_voters num_approving) * (approval_rate ^ num_approving) * ((1 - approval_rate) ^ (num_voters - num_approving))

theorem exactly_two_approve_probability :
  probability_exactly_two_approve = 0.1536 :=
sorry

end exactly_two_approve_probability_l3801_380100


namespace candy_distribution_l3801_380183

theorem candy_distribution (total : ℕ) (portions : ℕ) (increment : ℕ) (smallest : ℕ) : 
  total = 40 →
  portions = 4 →
  increment = 2 →
  (smallest + (smallest + increment) + (smallest + 2 * increment) + (smallest + 3 * increment) = total) →
  smallest = 7 := by
  sorry

end candy_distribution_l3801_380183


namespace min_sum_squares_l3801_380117

theorem min_sum_squares (x y z : ℝ) (h : x + 2*y + z = 1) : 
  ∃ (m : ℝ), (∀ a b c : ℝ, a + 2*b + c = 1 → a^2 + b^2 + c^2 ≥ m) ∧ 
  (∃ p q r : ℝ, p + 2*q + r = 1 ∧ p^2 + q^2 + r^2 = m) ∧ 
  m = 1/6 :=
sorry

end min_sum_squares_l3801_380117


namespace equal_pairs_comparison_l3801_380145

theorem equal_pairs_comparison : 
  (-3^5 = (-3)^5) ∧ 
  (-2^2 ≠ (-2)^2) ∧ 
  (-4 * 2^3 ≠ -4^2 * 3) ∧ 
  (-(-3)^2 ≠ -(-2)^3) :=
by sorry

end equal_pairs_comparison_l3801_380145


namespace clothing_tax_rate_l3801_380135

-- Define the total amount spent excluding taxes
variable (T : ℝ)

-- Define the tax rate on clothing as a percentage
variable (x : ℝ)

-- Define the spending percentages
def clothing_percent : ℝ := 0.45
def food_percent : ℝ := 0.45
def other_percent : ℝ := 0.10

-- Define the tax rates
def other_tax_rate : ℝ := 0.10
def total_tax_rate : ℝ := 0.0325

-- Theorem statement
theorem clothing_tax_rate :
  clothing_percent * T * (x / 100) + other_percent * T * other_tax_rate = total_tax_rate * T →
  x = 5 := by
sorry

end clothing_tax_rate_l3801_380135


namespace family_ages_l3801_380181

theorem family_ages (oleg_age : ℕ) (father_age : ℕ) (grandfather_age : ℕ) :
  father_age = oleg_age + 32 →
  grandfather_age = father_age + 32 →
  (oleg_age - 3) + (father_age - 3) + (grandfather_age - 3) < 100 →
  oleg_age > 0 →
  oleg_age = 4 ∧ father_age = 36 ∧ grandfather_age = 68 := by
  sorry

#check family_ages

end family_ages_l3801_380181


namespace wendy_sold_nine_pastries_l3801_380112

/-- The number of pastries Wendy sold at the bake sale -/
def pastries_sold (cupcakes cookies leftover : ℕ) : ℕ :=
  cupcakes + cookies - leftover

/-- Proof that Wendy sold 9 pastries at the bake sale -/
theorem wendy_sold_nine_pastries :
  pastries_sold 4 29 24 = 9 := by
  sorry

end wendy_sold_nine_pastries_l3801_380112


namespace prob_at_least_four_matching_dice_l3801_380111

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define the probability of getting at least four matching dice
def prob_at_least_four_matching : ℚ := 13 / 648

-- Theorem statement
theorem prob_at_least_four_matching_dice (n : ℕ) (s : ℕ) 
  (h1 : n = num_dice) (h2 : s = num_sides) : 
  prob_at_least_four_matching = 13 / 648 := by
  sorry

end prob_at_least_four_matching_dice_l3801_380111


namespace max_garden_area_l3801_380110

/-- Represents a rectangular garden with given constraints -/
structure Garden where
  width : ℝ
  length : ℝ
  fence_length : ℝ
  fence_constraint : length + 2 * width = fence_length
  size_constraint : length ≥ 2 * width

/-- The area of a garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- The maximum area of a garden given the constraints -/
theorem max_garden_area :
  ∃ (g : Garden), g.fence_length = 480 ∧ 
    (∀ (h : Garden), h.fence_length = 480 → g.area ≥ h.area) ∧
    g.area = 28800 := by
  sorry

end max_garden_area_l3801_380110


namespace toy_cost_price_l3801_380188

/-- Given a man sold 18 toys for Rs. 25200 and gained the cost price of 3 toys,
    prove that the cost price of a single toy is Rs. 1200. -/
theorem toy_cost_price (total_selling_price : ℕ) (num_toys_sold : ℕ) (num_toys_gain : ℕ) 
    (h1 : total_selling_price = 25200)
    (h2 : num_toys_sold = 18)
    (h3 : num_toys_gain = 3) :
  ∃ (cost_price : ℕ), cost_price = 1200 ∧ 
    total_selling_price = num_toys_sold * (cost_price + (num_toys_gain * cost_price) / num_toys_sold) :=
by sorry

end toy_cost_price_l3801_380188


namespace parabola_translation_equivalence_l3801_380113

/-- Represents a parabola in the form y = a(x - h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The original parabola y = 2x^2 -/
def original_parabola : Parabola := { a := 2, h := 0, k := 0 }

/-- The transformed parabola y = 2(x - 4)^2 - 1 -/
def transformed_parabola : Parabola := { a := 2, h := 4, k := -1 }

/-- Translation of a parabola -/
def translate (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a, h := p.h + dx, k := p.k + dy }

theorem parabola_translation_equivalence :
  translate original_parabola 4 (-1) = transformed_parabola := by
  sorry

end parabola_translation_equivalence_l3801_380113


namespace third_angle_is_40_l3801_380118

/-- A geometric configuration with an isosceles triangle connected to a right-angled triangle -/
structure GeometricConfig where
  -- Angles of the isosceles triangle
  α : ℝ
  β : ℝ
  γ : ℝ
  -- Angles of the right-angled triangle
  δ : ℝ
  ε : ℝ
  ζ : ℝ

/-- Properties of the geometric configuration -/
def is_valid_config (c : GeometricConfig) : Prop :=
  c.α = 65 ∧ c.β = 65 ∧  -- Two angles of isosceles triangle are 65°
  c.α + c.β + c.γ = 180 ∧  -- Sum of angles in isosceles triangle is 180°
  c.δ = 90 ∧  -- One angle of right-angled triangle is 90°
  c.γ = c.ε ∧  -- Vertically opposite angles are equal
  c.δ + c.ε + c.ζ = 180  -- Sum of angles in right-angled triangle is 180°

/-- Theorem stating that the third angle of the right-angled triangle is 40° -/
theorem third_angle_is_40 (c : GeometricConfig) (h : is_valid_config c) : c.ζ = 40 :=
sorry

end third_angle_is_40_l3801_380118


namespace sum_of_special_series_l3801_380189

def arithmeticSequenceSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_special_series :
  let a₁ := 1
  let d := 2
  let secondToLast := 99
  let last := 100
  let n := (secondToLast - a₁) / d + 1
  arithmeticSequenceSum a₁ d n + last = 2600 := by
  sorry

end sum_of_special_series_l3801_380189


namespace crayons_lost_or_given_away_l3801_380139

/-- Proves that the number of crayons lost or given away is equal to the sum of crayons given away and crayons lost -/
theorem crayons_lost_or_given_away 
  (initial : ℕ) 
  (given_away : ℕ) 
  (lost : ℕ) 
  (left : ℕ) 
  (h1 : initial = given_away + lost + left) : 
  given_away + lost = initial - left :=
by sorry

end crayons_lost_or_given_away_l3801_380139


namespace symmetric_point_x_axis_l3801_380132

/-- The coordinates of a point symmetric to P(2,3) with respect to the x-axis are (2,-3) -/
theorem symmetric_point_x_axis : 
  let P : ℝ × ℝ := (2, 3)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point P = (2, -3) := by sorry

end symmetric_point_x_axis_l3801_380132


namespace balloons_distribution_l3801_380173

theorem balloons_distribution (total_balloons : ℕ) (num_friends : ℕ) 
  (h1 : total_balloons = 236) (h2 : num_friends = 10) :
  total_balloons % num_friends = 6 := by
  sorry

end balloons_distribution_l3801_380173


namespace ellipse_constants_l3801_380154

/-- An ellipse with foci at (1, 1) and (1, 5) passing through (12, -4) -/
structure Ellipse where
  foci1 : ℝ × ℝ := (1, 1)
  foci2 : ℝ × ℝ := (1, 5)
  point : ℝ × ℝ := (12, -4)

/-- The standard form of an ellipse equation -/
def standard_form (a b h k : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- The theorem stating the constants of the ellipse -/
theorem ellipse_constants (e : Ellipse) :
  ∃ (a b h k : ℝ),
    a > 0 ∧ b > 0 ∧
    a = 13 ∧ b = Real.sqrt 153 ∧ h = 1 ∧ k = 3 ∧
    standard_form a b h k e.point.1 e.point.2 :=
by sorry

end ellipse_constants_l3801_380154


namespace quadratic_roots_sum_product_l3801_380175

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 2 * x^2 - m * x + n = 0 ∧ 2 * y^2 - m * y + n = 0 ∧ x + y = 6 ∧ x * y = 10) →
  m + n = 32 := by
sorry

end quadratic_roots_sum_product_l3801_380175


namespace otimes_h_otimes_h_l3801_380124

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 - x*y + y^2

/-- Theorem stating the result of h ⊗ (h ⊗ h) -/
theorem otimes_h_otimes_h (h : ℝ) : otimes h (otimes h h) = h^6 - h^4 + h^3 := by
  sorry

end otimes_h_otimes_h_l3801_380124


namespace unique_solution_for_system_l3801_380171

theorem unique_solution_for_system :
  ∃! (x y z : ℕ+), 
    (x.val : ℤ)^2 + y.val - z.val = 100 ∧ 
    (x.val : ℤ) + y.val^2 - z.val = 124 := by
  sorry

end unique_solution_for_system_l3801_380171


namespace complement_of_28_39_l3801_380107

/-- Represents an angle in degrees and minutes -/
structure Angle where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the complement of an angle -/
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  { degrees := totalMinutes / 60,
    minutes := totalMinutes % 60,
    valid := by sorry }

theorem complement_of_28_39 :
  let a : Angle := { degrees := 28, minutes := 39, valid := by sorry }
  complement a = { degrees := 61, minutes := 21, valid := by sorry } := by
  sorry

end complement_of_28_39_l3801_380107


namespace students_who_got_on_correct_l3801_380186

/-- The number of students who got on the bus at the first stop -/
def students_who_got_on (initial_students final_students : ℝ) : ℝ :=
  final_students - initial_students

theorem students_who_got_on_correct (initial_students final_students : ℝ) 
  (h1 : initial_students = 10.0) (h2 : final_students = 13) :
  students_who_got_on initial_students final_students = 3 := by
  sorry

end students_who_got_on_correct_l3801_380186


namespace vanya_cookies_l3801_380176

theorem vanya_cookies (total : ℚ) (vanya_before : ℚ) (shared : ℚ) :
  total > 0 ∧ vanya_before ≥ 0 ∧ shared ≥ 0 ∧
  total = vanya_before + shared ∧
  vanya_before + shared / 2 = 5 * (shared / 2) →
  vanya_before / total = 2 / 3 :=
by sorry

end vanya_cookies_l3801_380176


namespace candy_redistribution_l3801_380177

/-- Represents the distribution of candies in boxes -/
def CandyDistribution := List Nat

/-- An operation on the candy distribution -/
def redistribute (dist : CandyDistribution) (i j : Nat) : CandyDistribution :=
  sorry

/-- Checks if a distribution is valid (total candies = n^2) -/
def isValidDistribution (n : Nat) (dist : CandyDistribution) : Prop :=
  sorry

/-- Checks if a distribution is the goal distribution (n candies in each box) -/
def isGoalDistribution (n : Nat) (dist : CandyDistribution) : Prop :=
  sorry

/-- Checks if a number is a power of 2 -/
def isPowerOfTwo (n : Nat) : Prop :=
  sorry

theorem candy_redistribution (n : Nat) :
  (n > 2) →
  (∀ (init : CandyDistribution), isValidDistribution n init →
    ∃ (final : CandyDistribution), isGoalDistribution n final ∧
      ∃ (ops : List (Nat × Nat)), final = ops.foldl (fun d (i, j) => redistribute d i j) init) ↔
  isPowerOfTwo n :=
sorry

end candy_redistribution_l3801_380177


namespace cricket_average_score_l3801_380157

theorem cricket_average_score (total_matches : ℕ) (matches1 matches2 : ℕ) 
  (avg1 avg2 : ℚ) (h1 : total_matches = matches1 + matches2) 
  (h2 : matches1 = 2) (h3 : matches2 = 3) (h4 : avg1 = 30) (h5 : avg2 = 40) : 
  (matches1 * avg1 + matches2 * avg2) / total_matches = 36 := by
  sorry

end cricket_average_score_l3801_380157


namespace john_candies_proof_l3801_380170

/-- The number of candies John has -/
def john_candies : ℕ := 35

/-- The number of candies Mark has -/
def mark_candies : ℕ := 30

/-- The number of candies Peter has -/
def peter_candies : ℕ := 25

/-- The number of candies each person has after sharing equally -/
def shared_candies : ℕ := 30

/-- The number of people sharing the candies -/
def num_people : ℕ := 3

theorem john_candies_proof :
  john_candies = shared_candies * num_people - mark_candies - peter_candies :=
by sorry

end john_candies_proof_l3801_380170


namespace rectangle_length_proof_l3801_380102

theorem rectangle_length_proof (l w : ℝ) : l = w + 3 ∧ l * w = 4 → l = 4 := by
  sorry

end rectangle_length_proof_l3801_380102


namespace arithmetic_sequence_general_term_l3801_380167

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  h1 : a 6 = 12
  h2 : S 3 = 12
  h3 : ∀ n : ℕ, S n = (n / 2) * (a 1 + a n)  -- Sum formula for arithmetic sequence
  h4 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Common difference property

/-- The general term of the arithmetic sequence is 2n -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n :=
sorry

end arithmetic_sequence_general_term_l3801_380167


namespace intersection_M_N_l3801_380180

def M : Set ℝ := {-1, 0, 1}

def N : Set ℝ := {x : ℝ | (x + 2) * (x - 1) < 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by sorry

end intersection_M_N_l3801_380180


namespace salt_mixture_proof_l3801_380185

theorem salt_mixture_proof :
  let initial_amount : ℝ := 150
  let initial_concentration : ℝ := 0.35
  let added_amount : ℝ := 120
  let added_concentration : ℝ := 0.80
  let final_concentration : ℝ := 0.55
  
  (initial_amount * initial_concentration + added_amount * added_concentration) / (initial_amount + added_amount) = final_concentration :=
by
  sorry

end salt_mixture_proof_l3801_380185


namespace negation_of_union_l3801_380148

theorem negation_of_union (A B : Set α) (x : α) :
  ¬(x ∈ A ∪ B) ↔ x ∉ A ∧ x ∉ B := by
  sorry

end negation_of_union_l3801_380148


namespace sum_of_three_numbers_l3801_380122

theorem sum_of_three_numbers : 300 + 2020 + 10001 = 12321 := by
  sorry

end sum_of_three_numbers_l3801_380122


namespace trajectory_of_Q_l3801_380152

/-- The trajectory of point Q given point P on a circle -/
theorem trajectory_of_Q (m n : ℝ) : 
  m^2 + n^2 = 2 →   -- P is on the circle x^2 + y^2 = 2
  ∃ x y : ℝ,
    x = m + n ∧     -- x-coordinate of Q
    y = 2 * m * n ∧ -- y-coordinate of Q
    y = x^2 - 2 ∧   -- trajectory equation
    -2 ≤ x ∧ x ≤ 2  -- domain constraint
  := by sorry

end trajectory_of_Q_l3801_380152


namespace print_shop_charge_l3801_380196

/-- The charge per color copy at print shop X -/
def charge_x : ℝ := 1.20

/-- The charge per color copy at print shop Y -/
def charge_y : ℝ := 1.70

/-- The number of copies -/
def num_copies : ℕ := 40

/-- The additional charge at print shop Y compared to print shop X -/
def additional_charge : ℝ := 20

theorem print_shop_charge : 
  charge_x * num_copies + additional_charge = charge_y * num_copies :=
by sorry

end print_shop_charge_l3801_380196


namespace square_sum_given_diff_and_product_l3801_380108

theorem square_sum_given_diff_and_product (x y : ℝ) 
  (h1 : x - y = 20) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 418 := by
sorry

end square_sum_given_diff_and_product_l3801_380108


namespace gcd_of_specific_squares_l3801_380131

theorem gcd_of_specific_squares : Nat.gcd (123^2 + 235^2 + 347^2) (122^2 + 234^2 + 348^2) = 1 := by
  sorry

end gcd_of_specific_squares_l3801_380131


namespace prob_more_heads_than_tails_is_correct_l3801_380138

/-- The probability of getting more heads than tails when flipping 10 coins -/
def prob_more_heads_than_tails : ℚ := 193 / 512

/-- The number of coins being flipped -/
def num_coins : ℕ := 10

/-- The total number of possible outcomes when flipping 10 coins -/
def total_outcomes : ℕ := 2^num_coins

/-- The probability of getting exactly 5 heads (and 5 tails) when flipping 10 coins -/
def prob_equal_heads_tails : ℚ := 63 / 256

theorem prob_more_heads_than_tails_is_correct :
  prob_more_heads_than_tails = (1 - prob_equal_heads_tails) / 2 :=
sorry

end prob_more_heads_than_tails_is_correct_l3801_380138


namespace min_value_theorem_l3801_380137

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 5) :
  (9 / a) + (16 / b) + (25 / c) ≥ 144 / 5 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ a' + b' + c' = 5 ∧
    (9 / a') + (16 / b') + (25 / c') = 144 / 5 :=
by sorry

end min_value_theorem_l3801_380137


namespace expression_evaluation_l3801_380104

theorem expression_evaluation : 
  (2023^3 - 3 * 2023^2 * 2024 + 5 * 2023 * 2024^2 - 2024^3 + 5) / (2023 * 2024) = 4048 := by
  sorry

end expression_evaluation_l3801_380104


namespace fibonacci_eighth_term_l3801_380121

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_eighth_term : fibonacci 7 = 21 := by
  sorry

end fibonacci_eighth_term_l3801_380121


namespace coin_coverage_theorem_l3801_380116

/-- Represents the arrangement of 7 identical coins on an infinite plane -/
structure CoinArrangement where
  radius : ℝ
  num_coins : Nat
  touches_six : Bool

/-- Calculates the percentage of the plane covered by the coins -/
def coverage_percentage (arrangement : CoinArrangement) : ℝ :=
  sorry

/-- Theorem stating that the coverage percentage is 50π/√3 % -/
theorem coin_coverage_theorem (arrangement : CoinArrangement) 
  (h1 : arrangement.num_coins = 7)
  (h2 : arrangement.touches_six = true) : 
  coverage_percentage arrangement = (50 * Real.pi) / Real.sqrt 3 := by
  sorry

end coin_coverage_theorem_l3801_380116


namespace arccos_one_equals_zero_l3801_380151

theorem arccos_one_equals_zero :
  Real.arccos 1 = 0 := by
  sorry

#check arccos_one_equals_zero

end arccos_one_equals_zero_l3801_380151


namespace fly_distance_from_ceiling_l3801_380123

/-- The distance of a fly from the ceiling in a room -/
theorem fly_distance_from_ceiling :
  ∀ (z : ℝ),
  (2 : ℝ)^2 + 5^2 + z^2 = 7^2 →
  z = 2 * Real.sqrt 5 := by
sorry

end fly_distance_from_ceiling_l3801_380123


namespace apps_deleted_minus_added_l3801_380166

theorem apps_deleted_minus_added (initial_apps added_apps final_apps : ℕ) : 
  initial_apps + added_apps - final_apps - added_apps = 3 :=
by
  sorry

#check apps_deleted_minus_added 32 125 29

end apps_deleted_minus_added_l3801_380166


namespace rectangles_not_always_similar_l3801_380191

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

-- Define similarity for rectangles
def are_similar (r1 r2 : Rectangle) : Prop :=
  r1.length / r1.width = r2.length / r2.width

-- Theorem statement
theorem rectangles_not_always_similar :
  ∃ (r1 r2 : Rectangle), ¬(are_similar r1 r2) :=
sorry

end rectangles_not_always_similar_l3801_380191


namespace cube_volume_ratio_l3801_380156

-- Define the edge lengths
def edge_length_cube1 : ℚ := 4
def edge_length_cube2 : ℚ := 24  -- 2 feet = 24 inches

-- Define the volume ratio
def volume_ratio : ℚ := (edge_length_cube1 / edge_length_cube2) ^ 3

-- Theorem statement
theorem cube_volume_ratio :
  volume_ratio = 1 / 216 :=
by sorry

end cube_volume_ratio_l3801_380156


namespace dans_egg_purchase_l3801_380134

/-- The number of eggs in a dozen -/
def eggs_per_dozen : ℕ := 12

/-- The total number of eggs Dan bought -/
def total_eggs : ℕ := 108

/-- The number of dozens of eggs Dan bought -/
def dozens_bought : ℕ := total_eggs / eggs_per_dozen

theorem dans_egg_purchase : dozens_bought = 9 := by
  sorry

end dans_egg_purchase_l3801_380134


namespace distinct_arrangements_eq_factorial_l3801_380144

/-- The number of ways to arrange n distinct objects in n positions -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of boxes -/
def num_boxes : ℕ := 5

/-- The number of digits to place -/
def num_digits : ℕ := 4

/-- Theorem: The number of ways to arrange 4 distinct digits and 1 blank in 5 boxes
    is equal to 5! -/
theorem distinct_arrangements_eq_factorial :
  factorial num_boxes = 120 := by sorry

end distinct_arrangements_eq_factorial_l3801_380144
