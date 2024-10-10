import Mathlib

namespace least_k_inequality_l2659_265959

theorem least_k_inequality (a b c : ℝ) : 
  ∃ (k : ℝ), k = 8 ∧ (∀ (x : ℝ), x ≥ k → 
    (2*a/(a-b))^2 + (2*b/(b-c))^2 + (2*c/(c-a))^2 + x ≥ 
    4*((2*a/(a-b)) + (2*b/(b-c)) + (2*c/(c-a)))) ∧
  (∀ (y : ℝ), y < k → 
    ∃ (a' b' c' : ℝ), (2*a'/(a'-b'))^2 + (2*b'/(b'-c'))^2 + (2*c'/(c'-a'))^2 + y < 
    4*((2*a'/(a'-b')) + (2*b'/(b'-c')) + (2*c'/(c'-a')))) :=
sorry

end least_k_inequality_l2659_265959


namespace termite_ridden_not_collapsing_l2659_265976

theorem termite_ridden_not_collapsing (total_homes : ℕ) (termite_ridden : ℕ) (collapsing : ℕ) :
  termite_ridden = total_homes / 3 →
  collapsing = termite_ridden / 4 →
  (termite_ridden - collapsing) = total_homes / 4 :=
by sorry

end termite_ridden_not_collapsing_l2659_265976


namespace speaker_cost_correct_l2659_265913

/-- The amount Keith spent on speakers -/
def speaker_cost : ℚ := 136.01

/-- The amount Keith spent on a CD player -/
def cd_player_cost : ℚ := 139.38

/-- The amount Keith spent on new tires -/
def tire_cost : ℚ := 112.46

/-- The total amount Keith spent -/
def total_cost : ℚ := 387.85

/-- Theorem stating that the speaker cost is correct given the other expenses -/
theorem speaker_cost_correct : 
  speaker_cost = total_cost - (cd_player_cost + tire_cost) := by
  sorry

end speaker_cost_correct_l2659_265913


namespace binomial_coefficient_relation_l2659_265925

theorem binomial_coefficient_relation (n : ℕ) : 
  (n ≥ 2) →
  (Nat.choose n 2 * 3^(n-2) = 5 * 3^n) →
  n = 10 := by
sorry

end binomial_coefficient_relation_l2659_265925


namespace platform_length_specific_platform_length_l2659_265908

/-- The length of a platform given train speed, crossing time, and train length -/
theorem platform_length 
  (train_speed_kmph : ℝ) 
  (crossing_time_s : ℝ) 
  (train_length_m : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time_s
  total_distance - train_length_m

/-- Proof of the specific platform length problem -/
theorem specific_platform_length : 
  platform_length 72 26 260.0416 = 259.9584 := by
  sorry

end platform_length_specific_platform_length_l2659_265908


namespace sum_of_squares_l2659_265947

theorem sum_of_squares (a b c : ℝ) 
  (sum_condition : a + b + c = 5)
  (product_sum_condition : a * b + b * c + a * c = 5) :
  a^2 + b^2 + c^2 = 15 := by
sorry

end sum_of_squares_l2659_265947


namespace quadratic_equation_positive_roots_l2659_265972

theorem quadratic_equation_positive_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
    x₁^2 - 2*x₁ + m + 1 = 0 ∧ x₂^2 - 2*x₂ + m + 1 = 0) ↔ 
  (-1 < m ∧ m ≤ 0) :=
sorry

end quadratic_equation_positive_roots_l2659_265972


namespace product_decreasing_implies_inequality_l2659_265944

theorem product_decreasing_implies_inequality
  (f g : ℝ → ℝ)
  (hf : Differentiable ℝ f)
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, (deriv f x) * g x + f x * (deriv g x) < 0)
  (a b x : ℝ)
  (h_x : a < x ∧ x < b) :
  f x * g x > f b * g b :=
sorry

end product_decreasing_implies_inequality_l2659_265944


namespace hexagon_trapezoid_height_l2659_265956

/-- Given a 9 × 16 rectangle cut into two congruent hexagons that can form a larger rectangle
    with width 12, prove that the height of the internal trapezoid in one hexagon is 12. -/
theorem hexagon_trapezoid_height (original_width : ℝ) (original_height : ℝ)
  (resultant_width : ℝ) (y : ℝ) :
  original_width = 16 ∧ original_height = 9 ∧ resultant_width = 12 →
  y = 12 :=
by sorry

end hexagon_trapezoid_height_l2659_265956


namespace ellipse_focal_length_l2659_265949

/-- The focal length of an ellipse with equation x^2 + 2y^2 = 2 is 2 -/
theorem ellipse_focal_length : 
  let ellipse_eq : ℝ → ℝ → Prop := λ x y => x^2 + 2*y^2 = 2
  ∃ a b c : ℝ, 
    (∀ x y, ellipse_eq x y ↔ (x^2 / a^2) + (y^2 / b^2) = 1) ∧ 
    c^2 = a^2 - b^2 ∧
    2 * c = 2 := by
  sorry

end ellipse_focal_length_l2659_265949


namespace stamps_needed_l2659_265993

/-- The weight of one piece of paper in ounces -/
def paper_weight : ℚ := 1/5

/-- The number of pieces of paper used -/
def num_papers : ℕ := 8

/-- The weight of the envelope in ounces -/
def envelope_weight : ℚ := 2/5

/-- The number of stamps needed per ounce -/
def stamps_per_ounce : ℕ := 1

/-- The theorem stating the number of stamps needed for Jessica's letter -/
theorem stamps_needed : 
  ⌈(num_papers * paper_weight + envelope_weight) * stamps_per_ounce⌉ = 2 := by
  sorry

end stamps_needed_l2659_265993


namespace trebled_result_l2659_265915

theorem trebled_result (x : ℕ) (h : x = 7) : 3 * ((2 * x) + 9) = 69 := by
  sorry

end trebled_result_l2659_265915


namespace part_one_part_two_l2659_265952

-- Define combinatorial and permutation functions
def C (n k : ℕ) : ℕ := Nat.choose n k
def A (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Part 1: Prove that C₁₀⁴ - C₇³A₃³ = 0
theorem part_one : C 10 4 - C 7 3 * A 3 3 = 0 := by sorry

-- Part 2: Prove that the solution to 3A₈ˣ = 4A₉ˣ⁻¹ is x = 6
theorem part_two : ∃ (x : ℕ), 3 * A 8 x = 4 * A 9 (x - 1) ∧ x = 6 := by sorry

end part_one_part_two_l2659_265952


namespace jazmin_dolls_count_l2659_265914

/-- The number of dolls Geraldine has -/
def geraldine_dolls : ℕ := 2186

/-- The total number of dolls Jazmin and Geraldine have together -/
def total_dolls : ℕ := 3395

/-- The number of dolls Jazmin has -/
def jazmin_dolls : ℕ := total_dolls - geraldine_dolls

theorem jazmin_dolls_count : jazmin_dolls = 1209 := by
  sorry

end jazmin_dolls_count_l2659_265914


namespace min_value_expression_l2659_265918

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 27 * b^3 + 64 * c^3 + 27 / (8 * a * b * c) ≥ 18 * Real.sqrt 3 := by
  sorry

end min_value_expression_l2659_265918


namespace weight_of_new_person_l2659_265974

theorem weight_of_new_person
  (n : ℕ) -- number of persons
  (old_weight : ℝ) -- weight of the person being replaced
  (avg_increase : ℝ) -- increase in average weight
  (h1 : n = 15) -- there are 15 persons
  (h2 : old_weight = 80) -- the replaced person weighs 80 kg
  (h3 : avg_increase = 3.2) -- average weight increases by 3.2 kg
  : ∃ (new_weight : ℝ), new_weight = n * avg_increase + old_weight :=
by
  sorry

end weight_of_new_person_l2659_265974


namespace min_value_of_fraction_l2659_265978

theorem min_value_of_fraction (a b : ℝ) (h1 : a > b) (h2 : a * b = 1) :
  (∀ x y : ℝ, x > y ∧ x * y = 1 → (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2) ∧
  ∃ x y : ℝ, x > y ∧ x * y = 1 ∧ (x^2 + y^2) / (x - y) = 2 * Real.sqrt 2 :=
by sorry

end min_value_of_fraction_l2659_265978


namespace bake_sale_solution_l2659_265934

def bake_sale_problem (brownie_price : ℚ) (brownie_count : ℕ) 
                      (lemon_square_price : ℚ) (lemon_square_count : ℕ)
                      (total_goal : ℚ) (cookie_count : ℕ) : Prop :=
  let current_total : ℚ := brownie_price * brownie_count + lemon_square_price * lemon_square_count
  let remaining_goal : ℚ := total_goal - current_total
  let cookie_price : ℚ := remaining_goal / cookie_count
  cookie_price = 4

theorem bake_sale_solution :
  bake_sale_problem 3 4 2 5 50 7 := by
  sorry

end bake_sale_solution_l2659_265934


namespace fraction_simplification_l2659_265904

theorem fraction_simplification (a x : ℝ) (h : a^2 + x^2 ≠ 0) :
  (Real.sqrt (a^2 + x^2) + (x^2 - a^2) / Real.sqrt (a^2 + x^2)) / (a^2 + x^2) =
  2 * x^2 / (a^2 + x^2)^(3/2) := by
  sorry

end fraction_simplification_l2659_265904


namespace blue_face_ratio_one_third_l2659_265994

/-- Represents a cube with side length n -/
structure Cube (n : ℕ) where
  side_length : n > 0

/-- The number of blue faces after cutting the cube into unit cubes -/
def blue_faces (c : Cube n) : ℕ := 6 * n^2

/-- The total number of faces of all unit cubes -/
def total_faces (c : Cube n) : ℕ := 6 * n^3

/-- The theorem stating that the ratio of blue faces to total faces is 1/3 iff n = 3 -/
theorem blue_face_ratio_one_third (n : ℕ) (c : Cube n) :
  (blue_faces c : ℚ) / (total_faces c : ℚ) = 1/3 ↔ n = 3 := by sorry

end blue_face_ratio_one_third_l2659_265994


namespace smallest_three_star_three_star_divisibility_l2659_265970

/-- A three-star number is a positive three-digit integer that is the product of three distinct prime numbers. -/
def is_three_star (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ p q r : ℕ, 
    Prime p ∧ Prime q ∧ Prime r ∧ 
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    n = p * q * r

/-- The smallest three-star number is 102. -/
theorem smallest_three_star : 
  is_three_star 102 ∧ ∀ n, is_three_star n → 102 ≤ n :=
sorry

/-- Every three-star number is divisible by 2, 3, or 5. -/
theorem three_star_divisibility (n : ℕ) :
  is_three_star n → (2 ∣ n) ∨ (3 ∣ n) ∨ (5 ∣ n) :=
sorry

end smallest_three_star_three_star_divisibility_l2659_265970


namespace line_parametric_to_standard_l2659_265905

/-- Given a line with parametric equations x = -2 - 2t and y = 3 + √2 t,
    prove that its standard form is x + √2 y + 2 - 3√2 = 0 -/
theorem line_parametric_to_standard :
  ∀ (t x y : ℝ),
  (x = -2 - 2*t ∧ y = 3 + Real.sqrt 2 * t) →
  x + Real.sqrt 2 * y + 2 - 3 * Real.sqrt 2 = 0 :=
by sorry

end line_parametric_to_standard_l2659_265905


namespace min_sum_squares_l2659_265968

theorem min_sum_squares (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  ∃ (m : ℝ), m = 1/3 ∧ a^2 + b^2 + c^2 ≥ m ∧ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧ x^2 + y^2 + z^2 = m :=
by
  sorry

end min_sum_squares_l2659_265968


namespace sequence_problem_l2659_265950

def second_difference (a : ℕ → ℤ) : ℕ → ℤ := λ n => a (n + 2) - 2 * a (n + 1) + a n

theorem sequence_problem (a : ℕ → ℤ) 
  (h1 : ∀ n, second_difference a n = 16)
  (h2 : a 63 = 10)
  (h3 : a 89 = 10) :
  a 51 = 3658 := by
sorry

end sequence_problem_l2659_265950


namespace sequence_a_property_sequence_a_positive_sequence_a_first_two_terms_sequence_a_bounds_sequence_a_decreasing_l2659_265997

def sequence_a (n : ℕ+) : ℝ := sorry

theorem sequence_a_property (n : ℕ+) : 
  sequence_a n + n * sequence_a n - 1 = 0 := sorry

theorem sequence_a_positive (n : ℕ+) : 
  sequence_a n > 0 := sorry

theorem sequence_a_first_two_terms : 
  sequence_a 1 = 1/2 ∧ sequence_a 2 = 1/4 := sorry

theorem sequence_a_bounds (n : ℕ+) : 
  0 < sequence_a n ∧ sequence_a n < 1 := sorry

theorem sequence_a_decreasing (n : ℕ+) : 
  sequence_a n > sequence_a (n + 1) := sorry

end sequence_a_property_sequence_a_positive_sequence_a_first_two_terms_sequence_a_bounds_sequence_a_decreasing_l2659_265997


namespace sum_of_fractions_equals_one_l2659_265981

-- Define the variables
variable (a b c p q r : ℝ)

-- Define the conditions
axiom eq1 : 17 * p + b * q + c * r = 0
axiom eq2 : a * p + 29 * q + c * r = 0
axiom eq3 : a * p + b * q + 56 * r = 0
axiom a_ne_17 : a ≠ 17
axiom p_ne_0 : p ≠ 0

-- State the theorem
theorem sum_of_fractions_equals_one :
  a / (a - 17) + b / (b - 29) + c / (c - 56) = 1 := by sorry

end sum_of_fractions_equals_one_l2659_265981


namespace xy_sum_l2659_265916

theorem xy_sum (x y : ℕ) : 
  0 < x ∧ x < 20 ∧ 0 < y ∧ y < 20 ∧ x + y + x * y = 95 → x + y = 18 ∨ x + y = 20 := by
  sorry

end xy_sum_l2659_265916


namespace first_valid_year_is_1979_l2659_265998

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 1970 ∧ sum_of_digits year = 15

theorem first_valid_year_is_1979 :
  (∀ y : ℕ, y < 1979 → ¬(is_valid_year y)) ∧ is_valid_year 1979 := by
  sorry

end first_valid_year_is_1979_l2659_265998


namespace stratified_sample_male_count_l2659_265900

/-- Represents a stratified sample from a population of students -/
structure StratifiedSample where
  total_students : ℕ
  male_students : ℕ
  female_students : ℕ
  sample_female : ℕ
  sample_male : ℕ

/-- Theorem stating that in a given stratified sample, the number of male students in the sample is 18 -/
theorem stratified_sample_male_count 
  (sample : StratifiedSample) 
  (h1 : sample.total_students = 680)
  (h2 : sample.male_students = 360)
  (h3 : sample.female_students = 320)
  (h4 : sample.sample_female = 16)
  (h5 : sample.female_students * sample.sample_male = sample.male_students * sample.sample_female) :
  sample.sample_male = 18 := by
  sorry


end stratified_sample_male_count_l2659_265900


namespace complex_multiplication_l2659_265919

theorem complex_multiplication (i : ℂ) : i * i = -1 → (3 + i) * (1 - 2*i) = 5 - 5*i := by
  sorry

end complex_multiplication_l2659_265919


namespace fruit_seller_gain_percent_l2659_265987

theorem fruit_seller_gain_percent (cost_price selling_price : ℝ) 
  (h1 : cost_price > 0)
  (h2 : selling_price > 0)
  (h3 : 150 * selling_price - 150 * cost_price = 30 * selling_price) :
  (((150 * selling_price - 150 * cost_price) / (150 * cost_price)) * 100 = 25) :=
by sorry

end fruit_seller_gain_percent_l2659_265987


namespace gravel_path_cost_l2659_265983

/-- Calculate the cost of gravelling a path around a rectangular plot -/
theorem gravel_path_cost 
  (plot_length : ℝ) 
  (plot_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm_paise : ℝ) : 
  plot_length = 150 ∧ 
  plot_width = 95 ∧ 
  path_width = 4.5 ∧ 
  cost_per_sqm_paise = 90 → 
  (((plot_length + 2 * path_width) * (plot_width + 2 * path_width) - 
    plot_length * plot_width) * 
   (cost_per_sqm_paise / 100)) = 2057.40 :=
by sorry

end gravel_path_cost_l2659_265983


namespace rectangle_perimeter_l2659_265907

/-- Given a square with side length 2y that is divided into a center square
    with side length y and four congruent rectangles, prove that the perimeter
    of one of these rectangles is 3y. -/
theorem rectangle_perimeter (y : ℝ) (y_pos : 0 < y) :
  let large_square_side := 2 * y
  let center_square_side := y
  let rectangle_width := (large_square_side - center_square_side) / 2
  let rectangle_length := center_square_side
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_width)
  rectangle_perimeter = 3 * y :=
sorry

end rectangle_perimeter_l2659_265907


namespace angle_sum_in_triangle_l2659_265946

theorem angle_sum_in_triangle (A B C : ℝ) : 
  -- Triangle ABC exists
  -- Sum of angles A and B is 80°
  (A + B = 80) →
  -- Sum of all angles in a triangle is 180°
  (A + B + C = 180) →
  -- Angle C measures 100°
  C = 100 := by
sorry

end angle_sum_in_triangle_l2659_265946


namespace log_equation_solution_l2659_265953

theorem log_equation_solution :
  ∃ t : ℝ, t > 0 ∧ 4 * (Real.log t / Real.log 3) = Real.log (4 * t) / Real.log 3 → t = (4 : ℝ) ^ (1/3) := by
  sorry

end log_equation_solution_l2659_265953


namespace line_x_intercept_l2659_265992

/-- The x-intercept of a line passing through (2, -2) and (6, 10) is 8/3 -/
theorem line_x_intercept :
  let p1 : ℝ × ℝ := (2, -2)
  let p2 : ℝ × ℝ := (6, 10)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let x_intercept : ℝ := -b / m
  x_intercept = 8/3 := by
sorry


end line_x_intercept_l2659_265992


namespace cos_alpha_plus_5pi_over_4_l2659_265929

theorem cos_alpha_plus_5pi_over_4 (α : ℝ) (h : Real.sin (α - π/4) = 1/3) :
  Real.cos (α + 5*π/4) = 1/3 := by
  sorry

end cos_alpha_plus_5pi_over_4_l2659_265929


namespace neon_signs_blink_together_l2659_265948

theorem neon_signs_blink_together : Nat.lcm (Nat.lcm 7 11) 13 = 1001 := by
  sorry

end neon_signs_blink_together_l2659_265948


namespace inverse_function_equality_l2659_265937

/-- Given a function f(x) = 2 / (ax + b) where a and b are nonzero constants,
    prove that if f^(-1)(x) = 2, then b = 1 - 2a. -/
theorem inverse_function_equality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f := fun x : ℝ => 2 / (a * x + b)
  (∃ x, f x = 2) → b = 1 - 2 * a :=
by sorry

end inverse_function_equality_l2659_265937


namespace birth_rate_calculation_l2659_265989

/-- Represents the number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- Represents the number of population changes per day -/
def changes_per_day : ℕ := seconds_per_day / 2

/-- Represents the death rate in people per two seconds -/
def death_rate : ℕ := 2

/-- Represents the daily net population increase -/
def daily_net_increase : ℕ := 345600

/-- Represents the average birth rate in people per two seconds -/
def birth_rate : ℕ := 10

theorem birth_rate_calculation :
  (birth_rate - death_rate) * changes_per_day = daily_net_increase :=
by sorry

end birth_rate_calculation_l2659_265989


namespace katie_cookies_l2659_265982

def pastry_sale (cupcakes sold leftover : ℕ) : Prop :=
  ∃ (cookies total : ℕ),
    total = sold + leftover ∧
    total = cupcakes + cookies ∧
    cupcakes = 7 ∧
    sold = 4 ∧
    leftover = 8 ∧
    cookies = 5

theorem katie_cookies : pastry_sale 7 4 8 := by
  sorry

end katie_cookies_l2659_265982


namespace martha_has_19_butterflies_l2659_265932

/-- The number of butterflies in Martha's collection -/
structure ButterflyCollection where
  blue : ℕ
  yellow : ℕ
  black : ℕ

/-- Martha's butterfly collection satisfies the given conditions -/
def marthasCollection : ButterflyCollection where
  blue := 6
  yellow := 3
  black := 10

/-- The total number of butterflies in a collection -/
def totalButterflies (c : ButterflyCollection) : ℕ :=
  c.blue + c.yellow + c.black

/-- Theorem stating that Martha's collection has 19 butterflies in total -/
theorem martha_has_19_butterflies :
  totalButterflies marthasCollection = 19 ∧
  marthasCollection.blue = 2 * marthasCollection.yellow :=
by
  sorry


end martha_has_19_butterflies_l2659_265932


namespace carlos_singles_percentage_l2659_265941

/-- Represents the number of hits for each type of hit in baseball --/
structure HitCounts where
  total : ℕ
  homeRuns : ℕ
  triples : ℕ
  doubles : ℕ

/-- Calculates the percentage of singles given the hit counts --/
def percentageSingles (hits : HitCounts) : ℚ :=
  let singles := hits.total - (hits.homeRuns + hits.triples + hits.doubles)
  (singles : ℚ) / hits.total * 100

/-- Carlos's hit counts for the baseball season --/
def carlosHits : HitCounts :=
  { total := 50
  , homeRuns := 3
  , triples := 2
  , doubles := 8 }

/-- Theorem stating that the percentage of Carlos's hits that were singles is 74% --/
theorem carlos_singles_percentage :
  percentageSingles carlosHits = 74 := by
  sorry


end carlos_singles_percentage_l2659_265941


namespace arithmetic_sequence_sum_divisibility_l2659_265999

theorem arithmetic_sequence_sum_divisibility :
  ∀ (a d : ℕ+), 
  ∃ (k : ℕ), (15 : ℕ) * (a + 7 * d) = k := by
  sorry

end arithmetic_sequence_sum_divisibility_l2659_265999


namespace trigonometric_identity_l2659_265995

theorem trigonometric_identity (x : ℝ) :
  Real.cos (4 * x) * Real.cos (π + 2 * x) - Real.sin (2 * x) * Real.cos (π / 2 - 4 * x) = 
  Real.sqrt 2 / 2 * Real.sin (4 * x) := by
  sorry

end trigonometric_identity_l2659_265995


namespace sum_of_valid_a_values_l2659_265920

theorem sum_of_valid_a_values : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, 
    (∃ x : ℝ, x + 1 > (x - 1) / 3 ∧ x + a < 3) ∧ 
    (∃ y : ℤ, y > 0 ∧ y ≠ 2 ∧ (y - a) / (y - 2) + 1 = 1 / (y - 2))) ∧
  (∀ a : ℤ, 
    ((∃ x : ℝ, x + 1 > (x - 1) / 3 ∧ x + a < 3) ∧ 
     (∃ y : ℤ, y > 0 ∧ y ≠ 2 ∧ (y - a) / (y - 2) + 1 = 1 / (y - 2))) → 
    a ∈ S) ∧
  S.sum id = 2 := by
  sorry

end sum_of_valid_a_values_l2659_265920


namespace area_rectangle_circumscribing_right_triangle_l2659_265979

/-- The area of a rectangle circumscribing a right triangle with legs of length 5 and 6 is 30. -/
theorem area_rectangle_circumscribing_right_triangle : 
  ∀ (A B C D E : ℝ × ℝ),
    -- Right triangle ABC
    (B.1 - A.1) * (C.2 - B.2) = (C.1 - B.1) * (B.2 - A.2) →
    -- AB = 5
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 →
    -- BC = 6
    Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 6 →
    -- Rectangle ADEC circumscribes triangle ABC
    A.1 = D.1 ∧ A.2 = D.2 ∧
    C.1 = E.1 ∧ C.2 = E.2 ∧
    D.2 = E.2 ∧ A.1 = C.1 →
    -- Area of rectangle ADEC is 30
    (E.1 - D.1) * (E.2 - D.2) = 30 := by
  sorry


end area_rectangle_circumscribing_right_triangle_l2659_265979


namespace poll_percentage_equal_l2659_265963

theorem poll_percentage_equal (total : ℕ) (women_favor_percent : ℚ) (women_opposed : ℕ) : 
  total = 120 → women_favor_percent = 35/100 → women_opposed = 39 →
  ∃ (women men : ℕ), 
    women + men = total ∧ 
    women_opposed = (1 - women_favor_percent) * women ∧
    women = men ∧ 
    women / total = 1/2 ∧ 
    men / total = 1/2 := by
  sorry

#check poll_percentage_equal

end poll_percentage_equal_l2659_265963


namespace seven_ways_to_make_eight_cents_l2659_265964

/-- Represents the number of ways to make a certain amount with given coins -/
def WaysToMakeAmount (oneCent twoCent fiveCent target : ℕ) : ℕ := sorry

/-- Theorem stating that there are exactly 7 ways to make 8 cents with the given coins -/
theorem seven_ways_to_make_eight_cents :
  WaysToMakeAmount 8 4 1 8 = 7 := by sorry

end seven_ways_to_make_eight_cents_l2659_265964


namespace sequence_minimum_term_minimum_term_value_minimum_term_occurs_at_24_l2659_265911

theorem sequence_minimum_term (n : ℕ) (h1 : 7 ≤ n) (h2 : n ≤ 95) :
  (Real.sqrt (n / 6) + Real.sqrt (96 / n : ℝ)) ≥ 4 :=
by sorry

theorem minimum_term_value (n : ℕ) (h1 : 7 ≤ n) (h2 : n ≤ 95) :
  (Real.sqrt (24 / 6) + Real.sqrt (96 / 24 : ℝ)) = 4 :=
by sorry

theorem minimum_term_occurs_at_24 :
  ∃ (n : ℕ), 7 ≤ n ∧ n ≤ 95 ∧
  (Real.sqrt (n / 6) + Real.sqrt (96 / n : ℝ)) = 4 ∧
  n = 24 :=
by sorry

end sequence_minimum_term_minimum_term_value_minimum_term_occurs_at_24_l2659_265911


namespace sum_of_ages_l2659_265969

/-- Given information about the ages of Nacho, Divya, and Samantha, prove that the sum of their current ages is 80 years. -/
theorem sum_of_ages (nacho divya samantha : ℕ) : 
  (divya = 5) →
  (nacho + 5 = 3 * (divya + 5)) →
  (samantha = 2 * nacho) →
  (nacho + divya + samantha = 80) :=
by sorry

end sum_of_ages_l2659_265969


namespace fourth_side_length_l2659_265917

/-- A quadrilateral inscribed in a circle with three known side lengths -/
structure InscribedQuadrilateral where
  -- Three known side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- The fourth side length
  d : ℝ
  -- Condition that the quadrilateral is inscribed in a circle
  inscribed : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
  -- Condition that the areas of triangles ABC and ACD are equal
  equal_areas : a * b = c * d

/-- Theorem stating the possible values of the fourth side length -/
theorem fourth_side_length (q : InscribedQuadrilateral) 
  (h1 : q.a = 5 ∨ q.b = 5 ∨ q.c = 5)
  (h2 : q.a = 8 ∨ q.b = 8 ∨ q.c = 8)
  (h3 : q.a = 10 ∨ q.b = 10 ∨ q.c = 10) :
  q.d = 4 ∨ q.d = 6.25 ∨ q.d = 16 := by
  sorry

end fourth_side_length_l2659_265917


namespace part_one_part_two_l2659_265936

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 1|

-- Part 1
theorem part_one : 
  (∀ x : ℝ, f 2 x < 5 ↔ x ∈ Set.Ioo (-2) 3) :=
sorry

-- Part 2
theorem part_two :
  (∀ a : ℝ, (∀ x : ℝ, f a x ≥ 4 - |a - 1|) ↔ 
    a ∈ Set.Iic (-2) ∪ Set.Ici 2) :=
sorry

end part_one_part_two_l2659_265936


namespace optimal_decomposition_2008_l2659_265943

theorem optimal_decomposition_2008 (decomp : List Nat) :
  (decomp.sum = 2008) →
  (decomp.prod ≤ (List.replicate 668 3 ++ List.replicate 2 2).prod) :=
by sorry

end optimal_decomposition_2008_l2659_265943


namespace completing_square_result_l2659_265909

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 2*x - 2 = 0

-- Define the completed square form
def completed_square (x n : ℝ) : Prop := (x - 1)^2 = n

-- Theorem statement
theorem completing_square_result : 
  ∃ n : ℝ, (∀ x : ℝ, quadratic_equation x ↔ completed_square x n) ∧ n = 3 :=
sorry

end completing_square_result_l2659_265909


namespace turtle_count_relationship_lonely_island_turtle_count_l2659_265965

/-- The number of turtles on Happy Island -/
def happy_turtles : ℕ := 60

/-- The number of turtles on Lonely Island -/
def lonely_turtles : ℕ := 25

/-- Theorem stating the relationship between turtles on Happy and Lonely Islands -/
theorem turtle_count_relationship : happy_turtles = 2 * lonely_turtles + 10 := by
  sorry

/-- Theorem proving the number of turtles on Lonely Island -/
theorem lonely_island_turtle_count : lonely_turtles = 25 := by
  sorry

end turtle_count_relationship_lonely_island_turtle_count_l2659_265965


namespace complex_arithmetic_proof_l2659_265931

theorem complex_arithmetic_proof : ((2 : ℂ) + 5*I + (3 : ℂ) - 6*I) * ((1 : ℂ) + 2*I) = (7 : ℂ) + 9*I := by
  sorry

end complex_arithmetic_proof_l2659_265931


namespace negation_of_square_non_negative_l2659_265958

theorem negation_of_square_non_negative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by sorry

end negation_of_square_non_negative_l2659_265958


namespace figure_b_cannot_be_assembled_l2659_265954

-- Define the basic rhombus
structure Rhombus :=
  (color1 : String)
  (color2 : String)

-- Define the operation of rotation
def rotate (r : Rhombus) : Rhombus := r

-- Define the larger figures
inductive LargerFigure
  | A
  | B
  | C
  | D

-- Define a function to check if a larger figure can be assembled
def can_assemble (figure : LargerFigure) (r : Rhombus) : Prop :=
  match figure with
  | LargerFigure.A => True
  | LargerFigure.B => False
  | LargerFigure.C => True
  | LargerFigure.D => True

-- Theorem statement
theorem figure_b_cannot_be_assembled (r : Rhombus) :
  ¬(can_assemble LargerFigure.B r) ∧
  (can_assemble LargerFigure.A r) ∧
  (can_assemble LargerFigure.C r) ∧
  (can_assemble LargerFigure.D r) :=
sorry

end figure_b_cannot_be_assembled_l2659_265954


namespace expression_evaluation_l2659_265985

theorem expression_evaluation (a b : ℤ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 := by
  sorry

end expression_evaluation_l2659_265985


namespace quadratic_rational_solutions_l2659_265996

theorem quadratic_rational_solutions (d : ℕ+) : 
  (∃ x : ℚ, 7 * x^2 + 13 * x + d.val = 0) ↔ d = 6 := by
  sorry

end quadratic_rational_solutions_l2659_265996


namespace routes_on_3x2_grid_l2659_265901

/-- The number of routes on a grid from (0,0) to (m,n) moving only right or down -/
def numRoutes (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

/-- The width of the grid -/
def gridWidth : ℕ := 3

/-- The height of the grid -/
def gridHeight : ℕ := 2

theorem routes_on_3x2_grid : 
  numRoutes gridWidth gridHeight = 10 := by
  sorry

end routes_on_3x2_grid_l2659_265901


namespace longest_side_range_l2659_265942

/-- An obtuse triangle with sides a, b, and c, where c is the longest side -/
structure ObtuseTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c
  c_longest : c ≥ max a b
  obtuse : c^2 > a^2 + b^2

/-- The theorem stating the range of the longest side in a specific obtuse triangle -/
theorem longest_side_range (t : ObtuseTriangle) (ha : t.a = 1) (hb : t.b = 2) :
  Real.sqrt 5 < t.c ∧ t.c < 3 := by
  sorry

end longest_side_range_l2659_265942


namespace rectangular_field_diagonal_l2659_265922

/-- Given a rectangular field with width 4 m and area 12 m², 
    prove that its diagonal is 5 m. -/
theorem rectangular_field_diagonal : 
  ∀ (w l d : ℝ), 
    w = 4 → 
    w * l = 12 → 
    d ^ 2 = w ^ 2 + l ^ 2 → 
    d = 5 := by
  sorry

end rectangular_field_diagonal_l2659_265922


namespace no_distributive_laws_hold_l2659_265906

-- Define the * operation
def star (a b : ℝ) : ℝ := 3 * (a + b)

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  (∀ x y z : ℝ, star x (y + z) = star x y + star x z) ∧
  (∀ x y z : ℝ, x + star y z = star (x + y) (x + z)) ∧
  (∀ x y z : ℝ, star x (star y z) = star (star x y) (star x z)) →
  False :=
by sorry

end no_distributive_laws_hold_l2659_265906


namespace power_sum_value_l2659_265927

theorem power_sum_value (a m n : ℝ) (h1 : a^m = 5) (h2 : a^n = 3) :
  a^(m + n) = 15 := by
  sorry

end power_sum_value_l2659_265927


namespace inequality_and_equality_condition_l2659_265910

theorem inequality_and_equality_condition (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_one : a + b + c + d = 1) :
  (b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
  (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) ≤ 1/9 ∧
  ((b * c * d / (1 - a)^2) + (a * c * d / (1 - b)^2) + 
   (a * b * d / (1 - c)^2) + (a * b * c / (1 - d)^2) = 1/9 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end inequality_and_equality_condition_l2659_265910


namespace sum_of_even_integers_ranges_l2659_265945

def S1 : ℕ := (100 / 2) * (2 + 200)

def S2 : ℕ := (150 / 2) * (102 + 400)

theorem sum_of_even_integers_ranges (R : ℕ) : R = S1 + S2 → R = 47750 := by
  sorry

end sum_of_even_integers_ranges_l2659_265945


namespace usual_time_to_catch_bus_l2659_265977

/-- Proves that given a person walking at 3/5 of their usual speed and missing the bus by 5 minutes, their usual time to catch the bus is 7.5 minutes. -/
theorem usual_time_to_catch_bus : ∀ (usual_speed : ℝ) (usual_time : ℝ),
  usual_time > 0 →
  usual_speed > 0 →
  (3/5 * usual_speed) * (usual_time + 5) = usual_speed * usual_time →
  usual_time = 7.5 := by
sorry

end usual_time_to_catch_bus_l2659_265977


namespace range_of_a_l2659_265980

theorem range_of_a (x : ℝ) (h : x > 1) :
  ∃ (S : Set ℝ), S = {a : ℝ | a ≤ x + 1 / (x - 1)} ∧ 
  ∀ (ε : ℝ), ε > 0 → ∃ (a : ℝ), a ∈ S ∧ a > 3 - ε :=
by sorry

end range_of_a_l2659_265980


namespace quadratic_roots_property_l2659_265991

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p ^ 2 - 5 * p - 14 = 0) →
  (3 * q ^ 2 - 5 * q - 14 = 0) →
  p ≠ q →
  (3 * p ^ 2 - 3 * q ^ 2) / (p - q) = 5 := by
sorry

end quadratic_roots_property_l2659_265991


namespace equalSideToWidthRatio_l2659_265903

/-- Represents a rectangle with given width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Represents an isosceles triangle with two equal sides and a base -/
structure IsoscelesTriangle where
  equalSide : ℝ
  base : ℝ

/-- Calculates the perimeter of a rectangle -/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.length)

/-- Calculates the perimeter of an isosceles triangle -/
def IsoscelesTriangle.perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.equalSide + t.base

/-- Theorem: The ratio of the equal side of an isosceles triangle to the width of a rectangle
    is 5/2, given that both shapes have a perimeter of 60 and the rectangle's length is twice its width -/
theorem equalSideToWidthRatio :
  ∀ (r : Rectangle) (t : IsoscelesTriangle),
    r.perimeter = 60 →
    t.perimeter = 60 →
    r.length = 2 * r.width →
    t.equalSide / r.width = 5 / 2 := by
  sorry

end equalSideToWidthRatio_l2659_265903


namespace composite_and_infinite_x_l2659_265984

theorem composite_and_infinite_x (a : ℕ) :
  (∃ x : ℕ, ∃ y z : ℕ, y > 1 ∧ z > 1 ∧ a * x + 1 = y * z) ∧
  (∀ n : ℕ, ∃ x : ℕ, x > n ∧ ∃ y z : ℕ, y > 1 ∧ z > 1 ∧ a * x + 1 = y * z) :=
by sorry

end composite_and_infinite_x_l2659_265984


namespace particle_speed_l2659_265975

/-- Given a particle with position (3t + 4, 5t - 9) at time t, 
    its speed after a time interval of 2 units is √136. -/
theorem particle_speed (t : ℝ) : 
  let pos (t : ℝ) := (3 * t + 4, 5 * t - 9)
  let Δt := 2
  let Δx := (pos (t + Δt)).1 - (pos t).1
  let Δy := (pos (t + Δt)).2 - (pos t).2
  Real.sqrt (Δx ^ 2 + Δy ^ 2) = Real.sqrt 136 := by
  sorry

end particle_speed_l2659_265975


namespace eight_lines_theorem_l2659_265921

/-- Represents a collection of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  no_parallel : Bool
  no_concurrent : Bool

/-- Calculates the number of regions formed by a given line configuration -/
def num_regions (config : LineConfiguration) : ℕ :=
  sorry

/-- Theorem: Eight non-parallel, non-concurrent lines divide a plane into 37 regions -/
theorem eight_lines_theorem (config : LineConfiguration) :
  config.num_lines = 8 ∧ config.no_parallel ∧ config.no_concurrent →
  num_regions config = 37 :=
by sorry

end eight_lines_theorem_l2659_265921


namespace rectangle_length_l2659_265961

/-- Given a rectangle with a length to width ratio of 6:5 and a width of 20 inches,
    prove that its length is 24 inches. -/
theorem rectangle_length (width : ℝ) (length : ℝ) : 
  width = 20 → length / width = 6 / 5 → length = 24 := by
  sorry

end rectangle_length_l2659_265961


namespace bubble_pass_probability_specific_l2659_265939

/-- The probability that in a sequence of n distinct terms,
    the kth term ends up in the mth position after one bubble pass. -/
def bubble_pass_probability (n k m : ℕ) : ℚ :=
  if k ≤ m ∧ m < n then 1 / ((m - k + 2) * (m - k + 1))
  else 0

theorem bubble_pass_probability_specific :
  bubble_pass_probability 50 25 40 = 1 / 272 := by
  sorry

end bubble_pass_probability_specific_l2659_265939


namespace matrix_multiplication_proof_l2659_265938

theorem matrix_multiplication_proof :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, -2; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![2, -6; -1, 3]
  A * B = !![8, -24; 3, -9] := by sorry

end matrix_multiplication_proof_l2659_265938


namespace chord_length_is_2_sqrt_2_l2659_265955

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The chord length of a circle intercepted by a line -/
def chordLength (c : Circle) (l : Line) : ℝ := sorry

/-- The given circle x^2 + y^2 - 4y = 0 -/
def givenCircle : Circle :=
  { center := (0, 2),
    radius := 2 }

/-- The line passing through the origin with slope 1 -/
def givenLine : Line :=
  { slope := 1,
    yIntercept := 0 }

/-- Theorem: The chord length of the given circle intercepted by the given line is 2√2 -/
theorem chord_length_is_2_sqrt_2 :
  chordLength givenCircle givenLine = 2 * Real.sqrt 2 := by sorry

end chord_length_is_2_sqrt_2_l2659_265955


namespace exactly_one_solves_l2659_265924

theorem exactly_one_solves (p_A p_B p_C : ℝ) 
  (h_A : p_A = 1/2)
  (h_B : p_B = 1/3)
  (h_C : p_C = 1/4)
  (h_independent : True) -- Representing the independence condition
  : p_A * (1 - p_B) * (1 - p_C) + 
    (1 - p_A) * p_B * (1 - p_C) + 
    (1 - p_A) * (1 - p_B) * p_C = 11/24 := by
  sorry

end exactly_one_solves_l2659_265924


namespace billy_final_lap_is_150_seconds_l2659_265926

/-- Represents the swimming competition between Billy and Margaret -/
structure SwimmingCompetition where
  billy_first_5_laps : ℕ  -- time in seconds
  billy_next_3_laps : ℕ  -- time in seconds
  billy_9th_lap : ℕ      -- time in seconds
  margaret_total_time : ℕ -- time in seconds
  billy_win_margin : ℕ   -- time in seconds

/-- Calculates Billy's final lap time given the competition details -/
def billy_final_lap_time (comp : SwimmingCompetition) : ℕ :=
  comp.margaret_total_time - comp.billy_win_margin - 
  (comp.billy_first_5_laps + comp.billy_next_3_laps + comp.billy_9th_lap)

/-- Theorem stating that Billy's final lap time is 150 seconds -/
theorem billy_final_lap_is_150_seconds (comp : SwimmingCompetition) 
  (h1 : comp.billy_first_5_laps = 120)
  (h2 : comp.billy_next_3_laps = 240)
  (h3 : comp.billy_9th_lap = 60)
  (h4 : comp.margaret_total_time = 600)
  (h5 : comp.billy_win_margin = 30) :
  billy_final_lap_time comp = 150 := by
  sorry

end billy_final_lap_is_150_seconds_l2659_265926


namespace river_objects_l2659_265966

/-- The number of objects Bill tossed into the river -/
def bill_objects (ted_sticks ted_rocks bill_sticks bill_rocks : ℕ) : ℕ :=
  bill_sticks + bill_rocks

/-- The problem statement -/
theorem river_objects 
  (ted_sticks ted_rocks : ℕ) 
  (h1 : ted_sticks = 10)
  (h2 : ted_rocks = 10)
  (h3 : ∃ bill_sticks : ℕ, bill_sticks = ted_sticks + 6)
  (h4 : ∃ bill_rocks : ℕ, ted_rocks = 2 * bill_rocks) :
  ∃ bill_sticks bill_rocks : ℕ, 
    bill_objects ted_sticks ted_rocks bill_sticks bill_rocks = 21 :=
by
  sorry

end river_objects_l2659_265966


namespace decimal_93_to_binary_l2659_265960

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryRepresentation := List Nat

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : Nat) : BinaryRepresentation :=
  sorry

/-- Checks if a given BinaryRepresentation is valid (contains only 0s and 1s) -/
def isValidBinary (b : BinaryRepresentation) : Prop :=
  sorry

/-- Converts a binary representation back to decimal -/
def binaryToDecimal (b : BinaryRepresentation) : Nat :=
  sorry

theorem decimal_93_to_binary :
  let binary : BinaryRepresentation := [1, 0, 1, 1, 1, 0, 1]
  isValidBinary binary ∧
  binaryToDecimal binary = 93 ∧
  decimalToBinary 93 = binary :=
by sorry

end decimal_93_to_binary_l2659_265960


namespace m_range_proof_l2659_265971

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a < b ∧ (m + 1 = a) ∧ (3 - m = b)

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 2*m + 3 ≠ 0

-- Define the range of m
def m_range (m : ℝ) : Prop := 1 ≤ m ∧ m < 3

-- Theorem statement
theorem m_range_proof (m : ℝ) : (¬(p m ∧ q m) ∧ (p m ∨ q m)) → m_range m :=
sorry

end m_range_proof_l2659_265971


namespace value_calculation_l2659_265928

theorem value_calculation (number : ℝ) (value : ℝ) : 
  number = 8 → 
  value = 0.75 * number + 2 → 
  value = 8 := by
sorry

end value_calculation_l2659_265928


namespace equal_intercept_line_equation_l2659_265962

/-- A line passing through (4,1) with equal intercepts on both axes -/
structure EqualInterceptLine where
  /-- The slope of the line -/
  m : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (4,1) -/
  point_condition : m * 4 + b = 1
  /-- The line has equal intercepts on both axes -/
  equal_intercepts : b ≠ 0 → -b = b / m

/-- The equation of an EqualInterceptLine is either x - 4y = 0 or x + y - 5 = 0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (l.m = 1/4 ∧ l.b = 0) ∨ (l.m = -1 ∧ l.b = 5) := by
  sorry

end equal_intercept_line_equation_l2659_265962


namespace zeros_in_nine_nines_squared_l2659_265988

/-- The number of zeros in the square of a number consisting of n repeated 9s -/
def zeros_in_square (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- The number 999,999,999 -/
def nine_nines : ℕ := 999999999

theorem zeros_in_nine_nines_squared :
  zeros_in_square 9 = 8 :=
sorry

#check zeros_in_nine_nines_squared

end zeros_in_nine_nines_squared_l2659_265988


namespace women_at_gathering_l2659_265940

/-- Represents a social gathering with men and women dancing -/
structure SocialGathering where
  men : ℕ
  women : ℕ
  man_partners : ℕ
  woman_partners : ℕ

/-- Calculates the total number of dance pairs -/
def total_pairs (g : SocialGathering) : ℕ := g.men * g.man_partners

/-- Theorem: In a social gathering where 15 men attended, each man danced with 4 women,
    and each woman danced with 3 men, the number of women who attended is 20. -/
theorem women_at_gathering (g : SocialGathering) 
  (h1 : g.men = 15)
  (h2 : g.man_partners = 4)
  (h3 : g.woman_partners = 3)
  (h4 : total_pairs g = g.women * g.woman_partners) :
  g.women = 20 := by
  sorry

end women_at_gathering_l2659_265940


namespace line_through_intersection_and_parallel_l2659_265986

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 5 = 0
def line2 (x y : ℝ) : Prop := x + y + 2 = 0
def line3 (x y : ℝ) : Prop := 3 * x + y - 1 = 0
def result_line (x y : ℝ) : Prop := 3 * x + y = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define parallel lines
def parallel_lines (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

theorem line_through_intersection_and_parallel :
  (∃ (x y : ℝ), intersection_point x y ∧ result_line x y) ∧
  parallel_lines line3 result_line :=
sorry

end line_through_intersection_and_parallel_l2659_265986


namespace max_sum_2023_factors_l2659_265933

theorem max_sum_2023_factors :
  ∃ (A B C : ℕ+), 
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A * B * C = 2023 ∧
    ∀ (X Y Z : ℕ+), 
      X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z →
      X * Y * Z = 2023 →
      X + Y + Z ≤ A + B + C ∧
      A + B + C = 297 := by
sorry

end max_sum_2023_factors_l2659_265933


namespace total_buildable_area_l2659_265951

def num_sections : ℕ := 7
def section_area : ℝ := 9473
def open_space_percent : ℝ := 0.15

theorem total_buildable_area :
  (num_sections : ℝ) * section_area * (1 - open_space_percent) = 56364.35 := by
  sorry

end total_buildable_area_l2659_265951


namespace divisibility_property_l2659_265935

theorem divisibility_property (a b d : ℕ+) 
  (h1 : (a + b : ℕ) % d = 0)
  (h2 : (a * b : ℕ) % (d * d) = 0) :
  (a : ℕ) % d = 0 ∧ (b : ℕ) % d = 0 := by
  sorry

end divisibility_property_l2659_265935


namespace sine_cosine_inequality_condition_l2659_265967

theorem sine_cosine_inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ c > Real.sqrt (a^2 + b^2) := by
  sorry

end sine_cosine_inequality_condition_l2659_265967


namespace ratio_equality_l2659_265912

theorem ratio_equality (x : ℚ) : (x / (2 / 6)) = ((3 / 4) / (1 / 2)) → x = 1 / 2 := by
  sorry

end ratio_equality_l2659_265912


namespace jessie_min_score_l2659_265902

/-- Represents the test scores and conditions for Jessie's problem -/
structure TestScores where
  max_score : ℕ
  first_three : Fin 3 → ℕ
  total_tests : ℕ
  target_average : ℕ

/-- The minimum score needed on one of the remaining tests -/
def min_score (ts : TestScores) : ℕ :=
  let total_needed := ts.target_average * ts.total_tests
  let current_total := (ts.first_three 0) + (ts.first_three 1) + (ts.first_three 2)
  let remaining_total := total_needed - current_total
  remaining_total - 2 * ts.max_score

/-- Theorem stating the minimum score Jessie needs to achieve -/
theorem jessie_min_score :
  let ts : TestScores := {
    max_score := 120,
    first_three := ![88, 105, 96],
    total_tests := 6,
    target_average := 90
  }
  min_score ts = 11 := by sorry

end jessie_min_score_l2659_265902


namespace olivia_trip_length_l2659_265973

theorem olivia_trip_length :
  ∀ (total_length : ℚ),
    (1 / 4 : ℚ) * total_length + 30 + (1 / 6 : ℚ) * total_length = total_length →
    total_length = 360 / 7 := by
  sorry

end olivia_trip_length_l2659_265973


namespace lewis_items_count_l2659_265990

theorem lewis_items_count (tanya samantha lewis james : ℕ) : 
  tanya = 4 →
  samantha = 4 * tanya →
  lewis = samantha - (samantha / 3) →
  james = 2 * lewis →
  lewis = 11 := by
sorry

end lewis_items_count_l2659_265990


namespace total_shirts_is_ten_l2659_265957

/-- Represents the total number of shirts sold by the retailer -/
def total_shirts : ℕ := 10

/-- Represents the number of initially sold shirts -/
def initial_shirts : ℕ := 3

/-- Represents the prices of the initially sold shirts -/
def initial_prices : List ℝ := [20, 22, 25]

/-- Represents the desired overall average price -/
def desired_average : ℝ := 20

/-- Represents the minimum average price of the remaining shirts -/
def min_remaining_average : ℝ := 19

/-- Theorem stating that the total number of shirts is 10 given the conditions -/
theorem total_shirts_is_ten :
  total_shirts = initial_shirts + (total_shirts - initial_shirts) ∧
  (List.sum initial_prices + min_remaining_average * (total_shirts - initial_shirts)) / total_shirts > desired_average :=
by sorry

end total_shirts_is_ten_l2659_265957


namespace intersection_point_k_value_l2659_265923

theorem intersection_point_k_value :
  ∀ (k : ℝ),
  (∃ (y : ℝ), -3 * (-6) + 2 * y = k ∧ 0.75 * (-6) + y = 16) →
  k = 59 := by
sorry

end intersection_point_k_value_l2659_265923


namespace jen_bird_count_l2659_265930

/-- The number of ducks Jen has -/
def num_ducks : ℕ := 150

/-- The number of chickens Jen has -/
def num_chickens : ℕ := (num_ducks - 10) / 4

/-- The total number of birds Jen has -/
def total_birds : ℕ := num_ducks + num_chickens

theorem jen_bird_count : total_birds = 185 := by
  sorry

end jen_bird_count_l2659_265930
