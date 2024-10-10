import Mathlib

namespace perfect_square_trinomial_m_l772_77213

/-- If x^2 - 10x + m is a perfect square trinomial, then m = 25 -/
theorem perfect_square_trinomial_m (m : ℝ) : 
  (∃ a b : ℝ, ∀ x, x^2 - 10*x + m = (a*x + b)^2) → m = 25 := by
sorry

end perfect_square_trinomial_m_l772_77213


namespace concentric_circles_chords_l772_77219

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two consecutive chords is 60°, then the number of chords needed to
    complete a full rotation is 3. -/
theorem concentric_circles_chords (angle : ℝ) (n : ℕ) : 
  angle = 60 → n * angle = 360 → n = 3 := by sorry

end concentric_circles_chords_l772_77219


namespace simple_interest_time_calculation_l772_77288

/-- Simple interest calculation -/
theorem simple_interest_time_calculation
  (principal : ℝ)
  (simple_interest : ℝ)
  (rate : ℝ)
  (h1 : principal = 400)
  (h2 : simple_interest = 140)
  (h3 : rate = 17.5) :
  (simple_interest * 100) / (principal * rate) = 2 :=
by sorry

end simple_interest_time_calculation_l772_77288


namespace wrong_mark_calculation_l772_77227

theorem wrong_mark_calculation (n : ℕ) (initial_avg correct_avg : ℚ) (correct_mark : ℕ) :
  n = 30 ∧ 
  initial_avg = 60 ∧ 
  correct_avg = 57.5 ∧ 
  correct_mark = 15 →
  ∃ wrong_mark : ℕ,
    (n * initial_avg - wrong_mark + correct_mark) / n = correct_avg ∧
    wrong_mark = 90 := by
  sorry

end wrong_mark_calculation_l772_77227


namespace fraction_irreducible_l772_77276

theorem fraction_irreducible (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) : 
  ¬∃ (f g : ℝ → ℝ → ℝ → ℝ), (∀ x y z, f x y z ≠ 0 ∧ g x y z ≠ 0) ∧ 
    (∀ x y z, (x^2 + y^2 - z^2 + x*y) / (x^2 + z^2 - y^2 + y*z) = f x y z / g x y z) ∧
    (f a b c / g a b c ≠ (a^2 + b^2 - c^2 + a*b) / (a^2 + c^2 - b^2 + b*c)) :=
sorry

end fraction_irreducible_l772_77276


namespace alina_twist_result_l772_77278

/-- Alina's twisting method for periodic decimal fractions -/
def twist (n : ℚ) : ℚ :=
  sorry

/-- The period length of the decimal representation of 503/2022 -/
def period_length : ℕ := 336

theorem alina_twist_result :
  twist (503 / 2022) = 9248267898383371824480369515011881956675900099900099900099 / (10^period_length - 1) :=
sorry

end alina_twist_result_l772_77278


namespace remainder_s_mod_6_l772_77262

theorem remainder_s_mod_6 (s t : ℕ) (hs : s > t) (h_mod : (s - t) % 6 = 5) : s % 6 = 5 := by
  sorry

end remainder_s_mod_6_l772_77262


namespace jake_biking_speed_l772_77236

/-- Represents the distance to the water park -/
def distance_to_waterpark : ℝ := 22

/-- Represents Jake's dad's driving time in hours -/
def dad_driving_time : ℝ := 0.5

/-- Represents Jake's dad's first half speed in miles per hour -/
def dad_speed1 : ℝ := 28

/-- Represents Jake's dad's second half speed in miles per hour -/
def dad_speed2 : ℝ := 60

/-- Represents Jake's biking time in hours -/
def jake_biking_time : ℝ := 2

/-- Theorem stating that Jake's biking speed is 11 miles per hour -/
theorem jake_biking_speed : 
  distance_to_waterpark / jake_biking_time = 11 := by
  sorry

/-- Lemma showing that the distance is correctly calculated -/
lemma distance_calculation : 
  distance_to_waterpark = 
    dad_speed1 * (dad_driving_time / 2) + 
    dad_speed2 * (dad_driving_time / 2) := by
  sorry

end jake_biking_speed_l772_77236


namespace total_games_in_season_l772_77208

/-- The number of hockey games per month -/
def games_per_month : ℕ := 13

/-- The number of months in the hockey season -/
def months_in_season : ℕ := 14

/-- The total number of hockey games in the season -/
def total_games : ℕ := games_per_month * months_in_season

/-- Theorem stating that the total number of hockey games in the season is 182 -/
theorem total_games_in_season : total_games = 182 := by
  sorry

end total_games_in_season_l772_77208


namespace quadratic_inequality_solution_l772_77270

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 3*x < 10 ↔ -5 < x ∧ x < 2 := by
  sorry

end quadratic_inequality_solution_l772_77270


namespace complex_purely_imaginary_l772_77260

theorem complex_purely_imaginary (a : ℝ) :
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
  sorry

end complex_purely_imaginary_l772_77260


namespace inequality_solution_l772_77268

theorem inequality_solution (x : ℝ) : 2 - 1 / (3 * x + 4) < 5 ↔ x > -4/3 := by sorry

end inequality_solution_l772_77268


namespace cos_2x_value_l772_77257

theorem cos_2x_value (x : Real) (h : 2 * Real.sin x + Real.cos (π / 2 - x) = 1) :
  Real.cos (2 * x) = 7 / 9 := by
sorry

end cos_2x_value_l772_77257


namespace smallest_factor_for_square_l772_77295

theorem smallest_factor_for_square (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 10 → ¬∃ k : ℕ, 4410 * m = k * k) ∧ 
  (∃ k : ℕ, 4410 * 10 = k * k) := by
sorry

end smallest_factor_for_square_l772_77295


namespace horner_method_correctness_horner_method_equivalence_l772_77209

/-- Horner's Method evaluation for a specific polynomial -/
def horner_eval (x : ℝ) : ℝ := 
  (((((4 * x - 3) * x + 4) * x - 2) * x - 2) * x + 3)

/-- Count of multiplication operations in Horner's Method for this polynomial -/
def horner_mult_count : ℕ := 5

/-- Count of addition operations in Horner's Method for this polynomial -/
def horner_add_count : ℕ := 5

/-- Theorem stating the correctness of Horner's Method for the given polynomial -/
theorem horner_method_correctness : 
  horner_eval 3 = 816 ∧ 
  horner_mult_count = 5 ∧ 
  horner_add_count = 5 := by sorry

/-- Theorem stating that Horner's Method gives the same result as direct polynomial evaluation -/
theorem horner_method_equivalence (x : ℝ) : 
  horner_eval x = 4 * x^5 - 3 * x^4 + 4 * x^3 - 2 * x^2 - 2 * x + 3 := by sorry

end horner_method_correctness_horner_method_equivalence_l772_77209


namespace num_grade_assignments_l772_77261

/-- The number of students in the class -/
def num_students : ℕ := 10

/-- The number of possible grades (A, B, C) -/
def num_grades : ℕ := 3

/-- Theorem: The number of ways to assign grades to all students -/
theorem num_grade_assignments : (num_grades ^ num_students : ℕ) = 59049 := by
  sorry

end num_grade_assignments_l772_77261


namespace sticks_for_800_hexagons_l772_77218

/-- The number of sticks required to form a row of n hexagons -/
def sticksForHexagons (n : ℕ) : ℕ :=
  if n = 0 then 0 else 6 + 5 * (n - 1)

/-- Theorem: The number of sticks required for 800 hexagons is 4001 -/
theorem sticks_for_800_hexagons : sticksForHexagons 800 = 4001 := by
  sorry

#eval sticksForHexagons 800  -- To verify the result

end sticks_for_800_hexagons_l772_77218


namespace symmetry_xoy_plane_l772_77274

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the xOy plane -/
def symmetricXOY (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetry_xoy_plane :
  let A : Point3D := { x := 1, y := 2, z := 3 }
  let B : Point3D := symmetricXOY A
  B = { x := 1, y := 2, z := -3 } := by
  sorry

end symmetry_xoy_plane_l772_77274


namespace expression_value_l772_77211

theorem expression_value (x y z : ℝ) (hx : x = 1) (hy : y = 1) (hz : z = 3) :
  x^2 * y * z - x * y * z^2 = -6 := by
  sorry

end expression_value_l772_77211


namespace probability_of_dime_l772_77220

/-- Represents the types of coins in the jar -/
inductive Coin
| Dime
| Nickel
| Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
| Coin.Dime => 10
| Coin.Nickel => 5
| Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def total_value : Coin → ℕ
| Coin.Dime => 500
| Coin.Nickel => 300
| Coin.Penny => 200

/-- The number of coins of each type in the jar -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := coin_count Coin.Dime + coin_count Coin.Nickel + coin_count Coin.Penny

/-- The probability of randomly selecting a dime from the jar -/
theorem probability_of_dime : 
  (coin_count Coin.Dime : ℚ) / total_coins = 5 / 31 := by
  sorry


end probability_of_dime_l772_77220


namespace pythons_for_fifteen_alligators_l772_77279

/-- The number of Burmese pythons required to eat a given number of alligators in a specified time period. -/
def pythons_required (alligators : ℕ) (weeks : ℕ) : ℕ :=
  (alligators + weeks - 1) / weeks

/-- The theorem stating that 5 Burmese pythons are required to eat 15 alligators in 3 weeks. -/
theorem pythons_for_fifteen_alligators : pythons_required 15 3 = 5 := by
  sorry

#eval pythons_required 15 3

end pythons_for_fifteen_alligators_l772_77279


namespace square_sum_from_means_l772_77214

theorem square_sum_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 20) 
  (h_geometric : Real.sqrt (a * b) = Real.sqrt 104) : 
  a^2 + b^2 = 1392 := by sorry

end square_sum_from_means_l772_77214


namespace union_of_sets_l772_77296

def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {a + 2, 5}

theorem union_of_sets (a : ℕ) (h : A ∩ B a = {3}) : A ∪ B a = {1, 3, 5} := by
  sorry

end union_of_sets_l772_77296


namespace tangent_line_logarithm_l772_77238

theorem tangent_line_logarithm (x y : ℝ) :
  let f : ℝ → ℝ := λ t => Real.log t
  let tangent_line : ℝ → ℝ → Prop := λ a b => x - y - 1 = 0
  let perpendicular_line : ℝ → ℝ → Prop := λ a b => b = -a
  ∃ x₀ y₀ : ℝ,
    (y₀ = f x₀) ∧
    (∀ t : ℝ, (deriv f) x₀ * (deriv (λ s => -(s : ℝ))) t = -1) ∧
    tangent_line x₀ y₀ :=
by
  sorry

end tangent_line_logarithm_l772_77238


namespace geometric_arithmetic_progression_l772_77265

theorem geometric_arithmetic_progression (a b c : ℤ) : 
  (∃ (q : ℚ), b = a * q ∧ c = b * q) →  -- Geometric progression condition
  (2 * (b + 8) = a + c) →               -- Arithmetic progression condition
  ((b + 8)^2 = a * (c + 64)) →          -- Second geometric progression condition
  (a = 4 ∧ b = 12 ∧ c = 36) :=           -- Conclusion
by sorry

end geometric_arithmetic_progression_l772_77265


namespace largest_two_digit_prime_factor_of_binomial_200_100_l772_77269

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem largest_two_digit_prime_factor_of_binomial_200_100 :
  ∃ (p : ℕ), Prime p ∧ p < 100 ∧ p ∣ binomial 200 100 ∧
  ∀ (q : ℕ), Prime q → q < 100 → q ∣ binomial 200 100 → q ≤ p :=
by sorry

end largest_two_digit_prime_factor_of_binomial_200_100_l772_77269


namespace triangle_properties_l772_77202

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem combining all parts of the problem --/
theorem triangle_properties (t : Triangle) (p : ℝ) :
  (Real.sqrt 3 * Real.sin t.B - Real.cos t.B) * (Real.sqrt 3 * Real.sin t.C - Real.cos t.C) = 4 * Real.cos t.B * Real.cos t.C →
  t.A = π / 3 ∧
  (t.a = 2 → 0 < (1/2 * t.b * t.c * Real.sin t.A) ∧ (1/2 * t.b * t.c * Real.sin t.A) ≤ Real.sqrt 3) ∧
  (Real.sin t.B = p * Real.sin t.C → 1/2 < p ∧ p < 2) :=
by sorry


end triangle_properties_l772_77202


namespace point_on_line_with_equal_distances_quadrant_l772_77223

/-- A point with coordinates (x, y) -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line y = 2x + 3 -/
def lineEquation (p : Point) : Prop :=
  p.y = 2 * p.x + 3

/-- Equal distance to both coordinate axes -/
def equalDistanceToAxes (p : Point) : Prop :=
  abs p.x = abs p.y

/-- Second quadrant -/
def inSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Third quadrant -/
def inThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Theorem: A point on the line y = 2x + 3 with equal distances to both axes is in the second or third quadrant -/
theorem point_on_line_with_equal_distances_quadrant (p : Point) 
  (h1 : lineEquation p) (h2 : equalDistanceToAxes p) : 
  inSecondQuadrant p ∨ inThirdQuadrant p := by
  sorry

end point_on_line_with_equal_distances_quadrant_l772_77223


namespace intersection_A_B_l772_77290

open Set

def A : Set ℝ := {x | -4 < x ∧ x < 2}
def B : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_A_B : A ∩ B = Ioo 0 2 := by sorry

end intersection_A_B_l772_77290


namespace cube_spheres_surface_area_ratio_l772_77263

/-- The ratio of the surface area of a cube's inscribed sphere to its circumscribed sphere -/
theorem cube_spheres_surface_area_ratio (a : ℝ) (h : a > 0) : 
  (4 * Real.pi * (a / 2)^2) / (4 * Real.pi * (a * Real.sqrt 3 / 2)^2) = 1 / 3 := by
  sorry


end cube_spheres_surface_area_ratio_l772_77263


namespace student_activity_arrangements_l772_77233

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of arrangements for distributing students between two activities -/
def total_arrangements (n : ℕ) : ℕ :=
  choose n 4 + choose n 3 + choose n 2

theorem student_activity_arrangements :
  total_arrangements 6 = 50 := by sorry

end student_activity_arrangements_l772_77233


namespace solve_equation_l772_77297

theorem solve_equation : ∃ x : ℝ, (3 * x) / 4 = 15 ∧ x = 20 := by
  sorry

end solve_equation_l772_77297


namespace min_reciprocal_sum_l772_77264

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ 1 / 3 ∧ ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ x + y = 12 ∧ 1 / x + 1 / y = 1 / 3 := by
  sorry

end min_reciprocal_sum_l772_77264


namespace count_monomials_l772_77231

-- Define what a monomial is
def is_monomial (expr : String) : Bool :=
  match expr with
  | "0" => true
  | "2x-1" => false
  | "a" => true
  | "1/x" => false
  | "-2/3" => true
  | "(x-y)/2" => false
  | "2x/5" => true
  | _ => false

-- Define the set of expressions
def expressions : List String :=
  ["0", "2x-1", "a", "1/x", "-2/3", "(x-y)/2", "2x/5"]

-- Theorem statement
theorem count_monomials :
  (expressions.filter is_monomial).length = 4 := by sorry

end count_monomials_l772_77231


namespace probability_of_red_ball_l772_77254

/-- Given a bag with red and yellow balls, calculate the probability of drawing a red ball -/
theorem probability_of_red_ball (num_red : ℕ) (num_yellow : ℕ) :
  num_red = 6 → num_yellow = 3 →
  (num_red : ℚ) / (num_red + num_yellow : ℚ) = 2/3 :=
by
  sorry

end probability_of_red_ball_l772_77254


namespace students_studying_both_subjects_l772_77259

theorem students_studying_both_subjects (total : ℕ) 
  (physics_min physics_max chemistry_min chemistry_max : ℕ) : 
  total = 2500 →
  physics_min = 1750 →
  physics_max = 1875 →
  chemistry_min = 875 →
  chemistry_max = 1125 →
  ∃ (m M : ℕ),
    m = physics_min + chemistry_min - total ∧
    M = physics_max + chemistry_max - total ∧
    M - m = 375 :=
by sorry

end students_studying_both_subjects_l772_77259


namespace square_area_is_16_l772_77283

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 + 4*x + 3

-- Define the horizontal line
def horizontal_line : ℝ := 3

-- Theorem statement
theorem square_area_is_16 : ∃ (x₁ x₂ : ℝ),
  x₁ ≠ x₂ ∧
  parabola x₁ = horizontal_line ∧
  parabola x₂ = horizontal_line ∧
  (x₂ - x₁)^2 = 16 :=
sorry

end square_area_is_16_l772_77283


namespace arithmetic_sequence_sum_l772_77229

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 2 + a 4 + a 9 + a 11 = 32 →
  a 6 + a 7 = 16 := by
  sorry

end arithmetic_sequence_sum_l772_77229


namespace hex_B1C_equals_2844_l772_77256

/-- Converts a hexadecimal digit to its decimal value -/
def hexToDecimal (c : Char) : Nat :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | _ => c.toString.toNat!

/-- Converts a hexadecimal string to its decimal value -/
def hexStringToDecimal (s : String) : Nat :=
  s.foldl (fun acc c => 16 * acc + hexToDecimal c) 0

/-- The hexadecimal number B1C is equal to 2844 in decimal -/
theorem hex_B1C_equals_2844 : hexStringToDecimal "B1C" = 2844 := by
  sorry

end hex_B1C_equals_2844_l772_77256


namespace product_95_105_l772_77251

theorem product_95_105 : 95 * 105 = 9975 := by
  sorry

end product_95_105_l772_77251


namespace parabola_line_intersection_slope_product_l772_77294

/-- Given a parabola y^2 = 2px (p > 0) and a line y = x - p intersecting the parabola at points A and B,
    the product of the slopes of lines OA and OB is -2, where O is the coordinate origin. -/
theorem parabola_line_intersection_slope_product (p : ℝ) (h : p > 0) : 
  ∃ (A B : ℝ × ℝ),
    (A.2^2 = 2*p*A.1) ∧ 
    (B.2^2 = 2*p*B.1) ∧
    (A.2 = A.1 - p) ∧ 
    (B.2 = B.1 - p) ∧
    ((A.2 / A.1) * (B.2 / B.1) = -2) :=
sorry

end parabola_line_intersection_slope_product_l772_77294


namespace evaluate_expression_l772_77205

theorem evaluate_expression : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end evaluate_expression_l772_77205


namespace class_size_l772_77240

theorem class_size (dog_video_percentage : ℚ) (dog_movie_percentage : ℚ) (dog_preference_count : ℕ) :
  dog_video_percentage = 1/2 →
  dog_movie_percentage = 1/10 →
  dog_preference_count = 18 →
  (dog_video_percentage + dog_movie_percentage) * ↑dog_preference_count / (dog_video_percentage + dog_movie_percentage) = 30 :=
by sorry

end class_size_l772_77240


namespace quadratic_points_relationship_l772_77287

/-- A quadratic function f(x) = -2x^2 - 8x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := -2 * x^2 - 8 * x + m

theorem quadratic_points_relationship (m : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h₁ : f m (-1) = y₁)
  (h₂ : f m (-2) = y₂)
  (h₃ : f m (-4) = y₃) :
  y₃ < y₁ ∧ y₁ < y₂ := by
sorry

end quadratic_points_relationship_l772_77287


namespace family_heights_l772_77225

/-- Given the heights of a family, prove the calculated heights of specific members -/
theorem family_heights (cary bill jan tim sara : ℝ) : 
  cary = 72 →
  bill = 0.8 * cary →
  jan = bill + 5 →
  tim = (bill + jan) / 2 - 4 →
  sara = 1.2 * ((cary + bill + jan + tim) / 4) →
  (bill = 57.6 ∧ jan = 62.6 ∧ tim = 56.1 ∧ sara = 74.49) := by
  sorry

end family_heights_l772_77225


namespace solve_pet_sitting_problem_l772_77258

def pet_sitting_problem (hourly_rate : ℝ) (hours_this_week : ℝ) (total_earnings : ℝ) : Prop :=
  let earnings_this_week := hourly_rate * hours_this_week
  let earnings_last_week := total_earnings - earnings_this_week
  let hours_last_week := earnings_last_week / hourly_rate
  hourly_rate = 5 ∧ hours_this_week = 30 ∧ total_earnings = 250 → hours_last_week = 20

theorem solve_pet_sitting_problem :
  pet_sitting_problem 5 30 250 := by
  sorry

end solve_pet_sitting_problem_l772_77258


namespace total_tickets_sold_l772_77241

/-- Represents the number of tickets sold for a theater performance --/
structure TheaterTickets where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total revenue from ticket sales --/
def totalRevenue (tickets : TheaterTickets) : ℕ :=
  12 * tickets.orchestra + 8 * tickets.balcony

/-- Theorem stating the total number of tickets sold given the conditions --/
theorem total_tickets_sold : 
  ∃ (tickets : TheaterTickets), 
    totalRevenue tickets = 3320 ∧ 
    tickets.balcony = tickets.orchestra + 140 ∧
    tickets.orchestra + tickets.balcony = 360 := by
  sorry

#check total_tickets_sold

end total_tickets_sold_l772_77241


namespace female_attendees_on_time_l772_77292

theorem female_attendees_on_time (total_attendees : ℝ) :
  let male_fraction : ℝ := 3/5
  let male_on_time_fraction : ℝ := 7/8
  let not_on_time_fraction : ℝ := 0.155
  let female_fraction : ℝ := 1 - male_fraction
  let on_time_fraction : ℝ := 1 - not_on_time_fraction
  let male_on_time : ℝ := male_fraction * male_on_time_fraction * total_attendees
  let total_on_time : ℝ := on_time_fraction * total_attendees
  let female_on_time : ℝ := total_on_time - male_on_time
  let female_attendees : ℝ := female_fraction * total_attendees
  female_on_time / female_attendees = 4/5 := by sorry

end female_attendees_on_time_l772_77292


namespace digit_equation_solutions_l772_77203

theorem digit_equation_solutions (n : ℕ) (hn : n ≥ 2) :
  let a (x : ℕ) := x * (10^n - 1) / 9
  let b (y : ℕ) := y * (10^n - 1) / 9
  let c (z : ℕ) := z * (10^(2*n) - 1) / 9
  ∀ x y z : ℕ, (a x)^2 + b y = c z →
    ((x = 3 ∧ y = 2 ∧ z = 1) ∨
     (x = 6 ∧ y = 8 ∧ z = 4) ∨
     (x = 8 ∧ y = 3 ∧ z = 7)) :=
by sorry

end digit_equation_solutions_l772_77203


namespace books_sold_to_store_l772_77250

def book_problem (initial_books : ℕ) (book_club_months : ℕ) (bookstore_books : ℕ) 
  (yard_sale_books : ℕ) (daughter_books : ℕ) (mother_books : ℕ) (donated_books : ℕ) 
  (final_books : ℕ) : ℕ :=
  let total_acquired := initial_books + book_club_months + bookstore_books + 
                        yard_sale_books + daughter_books + mother_books
  let before_selling := total_acquired - donated_books
  before_selling - final_books

theorem books_sold_to_store : 
  book_problem 72 12 5 2 1 4 12 81 = 3 := by
  sorry

end books_sold_to_store_l772_77250


namespace largest_increase_2006_2007_l772_77201

def students : Fin 5 → ℕ
  | 0 => 80  -- 2003
  | 1 => 88  -- 2004
  | 2 => 94  -- 2005
  | 3 => 106 -- 2006
  | 4 => 130 -- 2007

def percentageIncrease (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def largestIncreaseYears : Fin 4 := 3

theorem largest_increase_2006_2007 :
  ∀ i : Fin 4, percentageIncrease (students i) (students (i + 1)) ≤ 
    percentageIncrease (students largestIncreaseYears) (students (largestIncreaseYears + 1)) :=
by sorry

end largest_increase_2006_2007_l772_77201


namespace cube_sum_reciprocal_l772_77200

theorem cube_sum_reciprocal (r : ℝ) (h : (r + 1/r)^2 = 3) : r^3 + 1/r^3 = 0 := by
  sorry

end cube_sum_reciprocal_l772_77200


namespace fiftieth_term_of_sequence_l772_77215

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1 : ℤ) * d

theorem fiftieth_term_of_sequence (a₁ d : ℤ) (h₁ : a₁ = 48) (h₂ : d = -2) :
  arithmeticSequenceTerm a₁ d 50 = -50 := by
  sorry

end fiftieth_term_of_sequence_l772_77215


namespace arithmetic_square_root_of_16_l772_77242

theorem arithmetic_square_root_of_16 : Real.sqrt 16 = 4 := by
  sorry

end arithmetic_square_root_of_16_l772_77242


namespace relationship_abc_l772_77266

theorem relationship_abc : 
  let a : ℝ := Real.rpow 0.8 0.7
  let b : ℝ := Real.rpow 0.8 0.9
  let c : ℝ := Real.rpow 1.1 0.6
  c > a ∧ a > b := by sorry

end relationship_abc_l772_77266


namespace min_longest_palindrome_length_l772_77277

/-- A string consisting only of characters 'A' and 'B' -/
def ABString : Type := List Char

/-- Check if a string is a palindrome -/
def isPalindrome (s : ABString) : Prop :=
  s = s.reverse

/-- The length of the longest palindromic substring in an ABString -/
def longestPalindromeLength (s : ABString) : ℕ :=
  sorry

theorem min_longest_palindrome_length :
  (∀ s : ABString, s.length = 2021 → longestPalindromeLength s ≥ 4) ∧
  (∃ s : ABString, s.length = 2021 ∧ longestPalindromeLength s = 4) :=
sorry

end min_longest_palindrome_length_l772_77277


namespace point_symmetry_l772_77239

/-- Given three points A, B, and P in a 2D Cartesian coordinate system,
    prove that if B has coordinates (1, 2), P is symmetric to A with respect to the x-axis,
    and P is symmetric to B with respect to the y-axis, then A has coordinates (-1, -2). -/
theorem point_symmetry (A B P : ℝ × ℝ) : 
  B = (1, 2) → 
  P.1 = A.1 ∧ P.2 = -A.2 →  -- P is symmetric to A with respect to x-axis
  P.1 = -B.1 ∧ P.2 = B.2 →  -- P is symmetric to B with respect to y-axis
  A = (-1, -2) := by
sorry

end point_symmetry_l772_77239


namespace difference_x_y_l772_77252

theorem difference_x_y (x y : ℤ) 
  (sum_eq : x + y = 20)
  (diff_eq : x - y = 10)
  (x_val : x = 15) :
  x - y = 10 := by
  sorry

end difference_x_y_l772_77252


namespace water_left_l772_77217

theorem water_left (initial_water : ℚ) (used_water : ℚ) (water_left : ℚ) : 
  initial_water = 3 ∧ used_water = 11/4 → water_left = initial_water - used_water → water_left = 1/4 :=
by sorry

end water_left_l772_77217


namespace sum_of_digits_l772_77206

def is_valid_arrangement (digits : Finset ℕ) (vertical horizontal : Finset ℕ) : Prop :=
  digits.card = 7 ∧ 
  digits ⊆ Finset.range 9 ∧ 
  vertical.card = 4 ∧ 
  horizontal.card = 4 ∧ 
  (vertical ∩ horizontal).card = 1 ∧
  vertical ⊆ digits ∧ 
  horizontal ⊆ digits

theorem sum_of_digits 
  (digits : Finset ℕ) 
  (vertical horizontal : Finset ℕ) 
  (h_valid : is_valid_arrangement digits vertical horizontal)
  (h_vertical_sum : vertical.sum id = 26)
  (h_horizontal_sum : horizontal.sum id = 20) :
  digits.sum id = 32 :=
sorry

end sum_of_digits_l772_77206


namespace jerrys_breakfast_l772_77248

theorem jerrys_breakfast (pancake_calories : ℕ) (bacon_calories : ℕ) (cereal_calories : ℕ) 
  (total_calories : ℕ) (bacon_strips : ℕ) :
  pancake_calories = 120 →
  bacon_calories = 100 →
  cereal_calories = 200 →
  total_calories = 1120 →
  bacon_strips = 2 →
  ∃ (num_pancakes : ℕ), 
    num_pancakes * pancake_calories + 
    bacon_strips * bacon_calories + 
    cereal_calories = total_calories ∧
    num_pancakes = 6 := by
  sorry

end jerrys_breakfast_l772_77248


namespace no_real_solution_log_equation_l772_77204

theorem no_real_solution_log_equation :
  ¬∃ x : ℝ, (Real.log (x + 6) + Real.log (x - 2) = Real.log (x^2 - 3*x - 18)) ∧
             (x + 6 > 0) ∧ (x - 2 > 0) ∧ (x^2 - 3*x - 18 > 0) := by
  sorry

end no_real_solution_log_equation_l772_77204


namespace unique_quadrilateral_perimeter_unique_perimeter_value_l772_77244

/-- Represents a quadrilateral with integer side lengths -/
structure Quadrilateral where
  AB : ℕ+
  BC : ℕ+
  CD : ℕ+
  AD : ℕ+

/-- The perimeter of a quadrilateral -/
def perimeter (q : Quadrilateral) : ℕ :=
  q.AB.val + q.BC.val + q.CD.val + q.AD.val

/-- Theorem stating that there is a unique quadrilateral satisfying the given conditions -/
theorem unique_quadrilateral_perimeter :
  ∃! (q : Quadrilateral),
    q.AB = 3 ∧
    q.BC = q.AD - 1 ∧
    q.BC = q.CD - 1 ∧
    (q.AB ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    (q.CD ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    perimeter q = 17 :=
  sorry

/-- Corollary: The perimeter of the unique quadrilateral is 17 -/
theorem unique_perimeter_value (p : ℕ) :
  (∃ (q : Quadrilateral),
    q.AB = 3 ∧
    q.BC = q.AD - 1 ∧
    q.BC = q.CD - 1 ∧
    (q.AB ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    (q.CD ^ 2 + q.BC ^ 2 : ℕ) = q.AD ^ 2 ∧
    perimeter q = p) →
  p = 17 :=
  sorry

end unique_quadrilateral_perimeter_unique_perimeter_value_l772_77244


namespace volleyball_team_selection_16_6_2_1_l772_77212

def volleyball_team_selection (n : ℕ) (k : ℕ) (t : ℕ) (c : ℕ) : ℕ :=
  Nat.choose (n - t - c) (k - c) + t * Nat.choose (n - t - c) (k - c - 1)

theorem volleyball_team_selection_16_6_2_1 :
  volleyball_team_selection 16 6 2 1 = 2717 := by
  sorry

end volleyball_team_selection_16_6_2_1_l772_77212


namespace sum_of_min_max_x_l772_77298

theorem sum_of_min_max_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 8) :
  ∃ m M : ℝ, (∀ x y z : ℝ, x + y + z = 5 → x^2 + y^2 + z^2 = 8 → m ≤ x ∧ x ≤ M) ∧
            (∃ x y z : ℝ, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8 ∧ x = m) ∧
            (∃ x y z : ℝ, x + y + z = 5 ∧ x^2 + y^2 + z^2 = 8 ∧ x = M) ∧
            m + M = 4 :=
by sorry

end sum_of_min_max_x_l772_77298


namespace cone_height_from_circular_sector_l772_77245

/-- The height of a cone formed by rolling one of four congruent sectors cut from a circular sheet of paper. -/
theorem cone_height_from_circular_sector (r : ℝ) (h : r = 10) :
  let sector_angle : ℝ := 2 * Real.pi / 4
  let base_radius : ℝ := r * sector_angle / (2 * Real.pi)
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  height = (5 * Real.sqrt 15) / 2 := by
  sorry

end cone_height_from_circular_sector_l772_77245


namespace driving_equation_correct_l772_77249

/-- Represents a driving scenario where the actual speed is faster than planned. -/
structure DrivingScenario where
  distance : ℝ
  planned_speed : ℝ
  actual_speed : ℝ
  time_saved : ℝ

/-- The equation correctly represents the driving scenario. -/
theorem driving_equation_correct (scenario : DrivingScenario) 
  (h1 : scenario.distance = 240)
  (h2 : scenario.actual_speed = 1.5 * scenario.planned_speed)
  (h3 : scenario.time_saved = 1)
  (h4 : scenario.planned_speed > 0) :
  scenario.distance / scenario.planned_speed - scenario.distance / scenario.actual_speed = scenario.time_saved := by
  sorry

#check driving_equation_correct

end driving_equation_correct_l772_77249


namespace quadratic_coefficients_l772_77216

def is_vertex (f : ℝ → ℝ) (x₀ y₀ : ℝ) : Prop :=
  ∀ x, f x ≥ f x₀ ∧ f x₀ = y₀

def has_vertical_symmetry_axis (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f (x₀ + x) = f (x₀ - x)

theorem quadratic_coefficients 
  (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = a * x^2 + b * x + c) →
  is_vertex f (-2) 5 →
  has_vertical_symmetry_axis f (-2) →
  f 0 = 9 →
  a = 1 ∧ b = 4 ∧ c = 9 := by
  sorry

end quadratic_coefficients_l772_77216


namespace boys_neither_happy_nor_sad_l772_77284

theorem boys_neither_happy_nor_sad
  (total_children : ℕ)
  (happy_children : ℕ)
  (sad_children : ℕ)
  (neither_children : ℕ)
  (total_boys : ℕ)
  (total_girls : ℕ)
  (happy_boys : ℕ)
  (sad_girls : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neither_children = 20)
  (h5 : total_boys = 22)
  (h6 : total_girls = 38)
  (h7 : happy_boys = 6)
  (h8 : sad_girls = 4)
  (h9 : total_children = happy_children + sad_children + neither_children)
  (h10 : total_children = total_boys + total_girls) :
  total_boys - (happy_boys + (sad_children - sad_girls)) = 10 :=
by sorry

end boys_neither_happy_nor_sad_l772_77284


namespace product_of_radicals_l772_77286

theorem product_of_radicals (p : ℝ) (hp : p > 0) :
  Real.sqrt (42 * p) * Real.sqrt (14 * p) * Real.sqrt (7 * p) = 14 * p * Real.sqrt (21 * p) := by
  sorry

end product_of_radicals_l772_77286


namespace team_formation_theorem_l772_77253

/-- The number of ways to form a team with at least one female student -/
def team_formation_count (male_count : ℕ) (female_count : ℕ) (team_size : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of ways to form the team under given conditions -/
theorem team_formation_theorem :
  team_formation_count 5 3 4 = 780 :=
sorry

end team_formation_theorem_l772_77253


namespace quadratic_equation_solution_l772_77271

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = (2 + Real.sqrt 2) / 2 ∧ 2 * x₁^2 = 4 * x₁ - 1) ∧
  (x₂ = (2 - Real.sqrt 2) / 2 ∧ 2 * x₂^2 = 4 * x₂ - 1) := by
sorry

end quadratic_equation_solution_l772_77271


namespace min_value_of_expression_l772_77226

theorem min_value_of_expression (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 ≥ 2018 ∧
  ∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018 :=
by sorry

end min_value_of_expression_l772_77226


namespace race_heartbeats_l772_77207

/-- Calculates the total number of heartbeats during a race given the race distance, cycling pace, and heart rate. -/
def total_heartbeats (race_distance : ℕ) (cycling_pace : ℕ) (heart_rate : ℕ) : ℕ :=
  race_distance * cycling_pace * heart_rate

/-- Theorem stating that for a 100-mile race, with a cycling pace of 4 minutes per mile and a heart rate of 120 beats per minute, the total number of heartbeats is 48000. -/
theorem race_heartbeats :
  total_heartbeats 100 4 120 = 48000 := by
  sorry

end race_heartbeats_l772_77207


namespace complex_simplification_l772_77237

/-- Given that i is the imaginary unit, prove that 
    (4*I)/((1-I)^2 + 2) + I^2018 = -2 + I -/
theorem complex_simplification :
  (4 * I) / ((1 - I)^2 + 2) + I^2018 = -2 + I :=
by sorry

end complex_simplification_l772_77237


namespace circle_configurations_exist_l772_77235

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the theorem
theorem circle_configurations_exist :
  ∃ (A B : ℝ × ℝ) (a b : ℝ),
    a > b ∧
    ∃ (AB : ℝ),
      AB > 0 ∧
      (a - b < AB) ∧
      (∃ AB', AB' > 0 ∧ a + b = AB') ∧
      (∃ AB'', AB'' > 0 ∧ a + b < AB'') ∧
      (∃ AB''', AB''' > 0 ∧ a - b = AB''') :=
by sorry

end circle_configurations_exist_l772_77235


namespace m_range_l772_77272

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (2*m - 2)*x + 3

-- Define the proposition p
def p (m : ℝ) : Prop := ∀ x < 0, ∀ y < x, f m x > f m y

-- Define the proposition q
def q (m : ℝ) : Prop := ∀ x, x^2 - 4*x + 1 - m > 0

-- State the theorem
theorem m_range :
  (∀ m : ℝ, p m → m ≤ 1) →
  (∀ m : ℝ, q m → m < -3) →
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  ∃ a b : ℝ, a = -3 ∧ b = 1 ∧ ∀ m : ℝ, a ≤ m ∧ m ≤ b :=
sorry

end m_range_l772_77272


namespace part_to_whole_ratio_l772_77247

theorem part_to_whole_ratio (N A : ℕ) (h1 : N = 48) (h2 : A = 15) : 
  ∃ P : ℕ, P + A = 27 → P * 4 = N := by
  sorry

end part_to_whole_ratio_l772_77247


namespace largest_c_for_negative_two_in_range_l772_77255

/-- The function f(x) defined as x^2 + 3x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + c

/-- Theorem stating that the largest value of c such that -2 is in the range of f(x) = x^2 + 3x + c is 1/4 -/
theorem largest_c_for_negative_two_in_range :
  (∃ (c : ℝ), ∀ (d : ℝ), (∃ (x : ℝ), f d x = -2) → d ≤ c) ∧
  (∃ (x : ℝ), f (1/4) x = -2) :=
sorry

end largest_c_for_negative_two_in_range_l772_77255


namespace expansion_sum_l772_77293

-- Define the sum of coefficients of the expansion
def P (n : ℕ) : ℕ := 4^n

-- Define the sum of all binomial coefficients
def S (n : ℕ) : ℕ := 2^n

-- Theorem statement
theorem expansion_sum (n : ℕ) : P n + S n = 272 → n = 4 := by
  sorry

end expansion_sum_l772_77293


namespace lcm_18_30_l772_77232

theorem lcm_18_30 : Nat.lcm 18 30 = 90 := by
  sorry

end lcm_18_30_l772_77232


namespace find_A_l772_77246

theorem find_A (A : ℕ) (h : A % 9 = 6 ∧ A / 9 = 2) : A = 24 := by
  sorry

end find_A_l772_77246


namespace complex_number_quadrant_l772_77291

theorem complex_number_quadrant (z : ℂ) (h : z = 1 + Complex.I) :
  2 / z + z^2 = 1 + Complex.I := by sorry

end complex_number_quadrant_l772_77291


namespace area_of_inscribed_hexagon_l772_77210

/-- The area of a regular hexagon inscribed in a circle with radius 3 units is 13.5√3 square units. -/
theorem area_of_inscribed_hexagon : 
  let r : ℝ := 3  -- radius of the circle
  let hexagon_area : ℝ := 6 * (r^2 * Real.sqrt 3 / 4)  -- area of hexagon as 6 times the area of an equilateral triangle
  hexagon_area = 13.5 * Real.sqrt 3 := by
  sorry

end area_of_inscribed_hexagon_l772_77210


namespace smallest_valid_number_l772_77230

def is_odd (n : ℕ) : Bool := n % 2 = 1

def is_even (n : ℕ) : Bool := n % 2 = 0

def digit_count (n : ℕ) : ℕ := (String.length (toString n))

def sum_of_digits (n : ℕ) : ℕ :=
  (toString n).toList.map (fun c => c.toNat - '0'.toNat) |>.sum

def is_valid_number (n : ℕ) : Bool :=
  digit_count n = 4 ∧
  n % 9 = 0 ∧
  (is_odd (n / 1000 % 10) + is_odd (n / 100 % 10) + is_odd (n / 10 % 10) + is_odd (n % 10) = 3) ∧
  (is_even (n / 1000 % 10) + is_even (n / 100 % 10) + is_even (n / 10 % 10) + is_even (n % 10) = 1)

theorem smallest_valid_number : 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 1215 → ¬ is_valid_number m) ∧ is_valid_number 1215 := by sorry

end smallest_valid_number_l772_77230


namespace arithmetic_sequence_length_l772_77221

theorem arithmetic_sequence_length (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ = 2 →
  aₙ = 3006 →
  d = 4 →
  aₙ = a₁ + (n - 1) * d →
  n = 752 := by
  sorry

end arithmetic_sequence_length_l772_77221


namespace hyperbola_focal_length_l772_77280

theorem hyperbola_focal_length :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 / 10 - y^2 / 2 = 1
  ∃ (f : ℝ), f = 4 * Real.sqrt 3 ∧ 
    ∀ (x y : ℝ), h x y → 
      f = 2 * Real.sqrt ((Real.sqrt 10)^2 + (Real.sqrt 2)^2) :=
by sorry

end hyperbola_focal_length_l772_77280


namespace bookstore_editions_l772_77267

-- Define the universe of books in the bookstore
variable (Book : Type)

-- Define a predicate for new editions
variable (is_new_edition : Book → Prop)

-- Theorem statement
theorem bookstore_editions (h : ¬∀ (b : Book), is_new_edition b) :
  (∃ (b : Book), ¬is_new_edition b) ∧ (¬∀ (b : Book), is_new_edition b) := by
  sorry

end bookstore_editions_l772_77267


namespace certain_value_multiplication_l772_77281

theorem certain_value_multiplication (x : ℝ) : x * (1/7)^2 = 7^3 → x = 16807 := by
  sorry

end certain_value_multiplication_l772_77281


namespace no_decreasing_h_for_increasing_f_l772_77289

-- Define the function f in terms of h
def f (h : ℝ → ℝ) (x : ℝ) : ℝ := (x^2 - x + 1) * h x

-- State the theorem
theorem no_decreasing_h_for_increasing_f :
  ¬ ∃ h : ℝ → ℝ,
    (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x ≤ y → h y ≤ h x) ∧
    (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x ≤ y → f h x ≤ f h y) :=
by sorry

end no_decreasing_h_for_increasing_f_l772_77289


namespace change_in_quadratic_expression_l772_77285

theorem change_in_quadratic_expression (x b : ℝ) (h : b > 0) :
  let f := fun x => 2 * x^2 + 5
  let change_plus := f (x + b) - f x
  let change_minus := f (x - b) - f x
  change_plus = 4 * x * b + 2 * b^2 ∧ change_minus = -4 * x * b + 2 * b^2 :=
by sorry

end change_in_quadratic_expression_l772_77285


namespace trisomy21_caused_by_sperm_l772_77243

/-- Represents a genotype for the STR marker on chromosome 21 -/
inductive Genotype
  | Negative
  | Positive
  | DoublePositive

/-- Represents a person with their genotype -/
structure Person where
  genotype : Genotype

/-- Represents a family with a child, father, and mother -/
structure Family where
  child : Person
  father : Person
  mother : Person

/-- Defines Trisomy 21 syndrome -/
def hasTrisomy21 (p : Person) : Prop := p.genotype = Genotype.DoublePositive

/-- Defines the condition of sperm having 2 chromosome 21s -/
def spermHasTwoChromosome21 (f : Family) : Prop :=
  f.father.genotype = Genotype.Positive ∧
  f.mother.genotype = Genotype.Negative ∧
  f.child.genotype = Genotype.DoublePositive

/-- Theorem stating that given the family's genotypes, the child's Trisomy 21 is caused by sperm with 2 chromosome 21s -/
theorem trisomy21_caused_by_sperm (f : Family)
  (h_child : f.child.genotype = Genotype.DoublePositive)
  (h_father : f.father.genotype = Genotype.Positive)
  (h_mother : f.mother.genotype = Genotype.Negative) :
  hasTrisomy21 f.child ∧ spermHasTwoChromosome21 f := by
  sorry


end trisomy21_caused_by_sperm_l772_77243


namespace parabola_through_point_l772_77224

-- Define a parabola type
structure Parabola where
  -- A parabola is defined by its equation
  equation : ℝ → ℝ → Prop

-- Define the condition that a parabola passes through a point
def passes_through (p : Parabola) (x y : ℝ) : Prop :=
  p.equation x y

-- Define the two possible standard forms of a parabola
def vertical_parabola (a : ℝ) : Parabola :=
  ⟨λ x y => y^2 = 4*a*x⟩

def horizontal_parabola (b : ℝ) : Parabola :=
  ⟨λ x y => x^2 = 4*b*y⟩

-- The theorem to be proved
theorem parabola_through_point :
  ∃ (p : Parabola), passes_through p (-2) 4 ∧
    ((∃ a : ℝ, p = vertical_parabola a ∧ a = -2) ∨
     (∃ b : ℝ, p = horizontal_parabola b ∧ b = 1/4)) :=
sorry

end parabola_through_point_l772_77224


namespace greatest_integer_radius_of_semicircle_l772_77299

theorem greatest_integer_radius_of_semicircle (A : ℝ) (h : A < 45 * Real.pi) :
  ∃ (r : ℕ), r = 9 ∧ (∀ (n : ℕ), (↑n : ℝ)^2 * Real.pi / 2 ≤ A → n ≤ 9) :=
sorry

end greatest_integer_radius_of_semicircle_l772_77299


namespace accounting_equation_l772_77275

def p : ℂ := 7
def z : ℂ := 7 + 175 * Complex.I

theorem accounting_equation (h : 3 * p - z = 15000) : 
  p = 5002 + (175 / 3) * Complex.I := by
  sorry

end accounting_equation_l772_77275


namespace evaluate_expression_l772_77273

theorem evaluate_expression : (24 ^ 40) / (72 ^ 20) = 2 ^ 60 := by
  sorry

end evaluate_expression_l772_77273


namespace winter_olympics_theorem_l772_77234

/-- Represents the scoring system for the Winter Olympics knowledge competition. -/
structure ScoringSystem where
  num_questions : ℕ
  correct_points : ℕ
  incorrect_points : ℤ

/-- Calculates the total score given the number of correct and incorrect answers. -/
def calculate_score (system : ScoringSystem) (correct : ℕ) (incorrect : ℕ) : ℤ :=
  (correct : ℤ) * system.correct_points - incorrect * system.incorrect_points

/-- Calculates the minimum number of students required for at least 3 to have the same score. -/
def min_students_for_same_score (system : ScoringSystem) : ℕ :=
  (system.num_questions * system.correct_points + 1) * 2 + 1

/-- The Winter Olympics knowledge competition theorem. -/
theorem winter_olympics_theorem (system : ScoringSystem)
  (h_num_questions : system.num_questions = 10)
  (h_correct_points : system.correct_points = 5)
  (h_incorrect_points : system.incorrect_points = 1)
  (h_xiao_ming_correct : ℕ)
  (h_xiao_ming_incorrect : ℕ)
  (h_xiao_ming_total : h_xiao_ming_correct + h_xiao_ming_incorrect = system.num_questions)
  (h_xiao_ming_correct_8 : h_xiao_ming_correct = 8)
  (h_xiao_ming_incorrect_2 : h_xiao_ming_incorrect = 2) :
  (calculate_score system h_xiao_ming_correct h_xiao_ming_incorrect = 38) ∧
  (min_students_for_same_score system = 23) := by
  sorry

end winter_olympics_theorem_l772_77234


namespace abs_difference_of_product_and_sum_l772_77222

theorem abs_difference_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 6) 
  (h2 : p + q = 7) : 
  |p - q| = Real.sqrt 37 := by
  sorry

end abs_difference_of_product_and_sum_l772_77222


namespace sum_of_angles_equals_540_l772_77228

-- Define the angles as real numbers
variable (a b c d e f g : ℝ)

-- Define the straight lines (we don't need to explicitly define them, 
-- but we'll use their properties in the theorem statement)

-- State the theorem
theorem sum_of_angles_equals_540 :
  a + b + c + d + e + f + g = 540 := by
  sorry


end sum_of_angles_equals_540_l772_77228


namespace stream_top_width_l772_77282

/-- 
Theorem: Given a trapezoidal cross-section of a stream with specified dimensions,
prove that the width at the top of the stream is 10 meters.
-/
theorem stream_top_width 
  (area : ℝ) 
  (depth : ℝ) 
  (bottom_width : ℝ) 
  (h_area : area = 640) 
  (h_depth : depth = 80) 
  (h_bottom : bottom_width = 6) :
  let top_width := (2 * area / depth) - bottom_width
  top_width = 10 := by
sorry

end stream_top_width_l772_77282
