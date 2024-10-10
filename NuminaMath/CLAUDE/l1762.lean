import Mathlib

namespace selection_theorem_l1762_176231

/-- The number of students in the group -/
def total_students : ℕ := 7

/-- The number of representatives to be selected -/
def representatives : ℕ := 3

/-- The number of special students (A and B) -/
def special_students : ℕ := 2

/-- The number of ways to select 3 representatives from 7 students,
    with the condition that only one of students A and B is selected -/
def selection_ways : ℕ := Nat.choose special_students 1 * Nat.choose (total_students - special_students) (representatives - 1)

theorem selection_theorem : selection_ways = 20 := by sorry

end selection_theorem_l1762_176231


namespace alpha_value_l1762_176216

theorem alpha_value (α β : ℂ) 
  (h1 : (α + β).im = 0 ∧ (α + β).re > 0)
  (h2 : (Complex.I * (α - 3 * β)).im = 0 ∧ (Complex.I * (α - 3 * β)).re > 0)
  (h3 : β = 4 + 3 * Complex.I) : 
  α = 6 - 3 * Complex.I :=
sorry

end alpha_value_l1762_176216


namespace min_value_of_sum_of_ratios_l1762_176250

theorem min_value_of_sum_of_ratios (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_eq : (a + c) * (b + d) = a * c + b * d) : 
  a / b + b / c + c / d + d / a ≥ 8 ∧ 
  ∃ (a' b' c' d' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ d' > 0 ∧
    (a' + c') * (b' + d') = a' * c' + b' * d' ∧
    a' / b' + b' / c' + c' / d' + d' / a' = 8 :=
by sorry

end min_value_of_sum_of_ratios_l1762_176250


namespace log_equation_solution_l1762_176218

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * Real.log x / Real.log 3 = Real.log (6 * x) / Real.log 3 → x = (6 : ℝ) ^ (1/3) := by
  sorry

end log_equation_solution_l1762_176218


namespace arithmetic_sequence_k_value_l1762_176295

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_k_value
  (a : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h_arith : arithmetic_sequence a d)
  (h_d_nonzero : d ≠ 0)
  (h_a1 : a 1 = 9 * d)
  (h_geom_mean : a k ^ 2 = a 1 * a (2 * k)) :
  k = 4 := by
sorry

end arithmetic_sequence_k_value_l1762_176295


namespace triangle_formation_l1762_176288

/-- Given two sticks of lengths 3 and 5, determine if a third stick of length l can form a triangle with them. -/
def can_form_triangle (l : ℝ) : Prop :=
  l > 0 ∧ l + 3 > 5 ∧ l + 5 > 3 ∧ 3 + 5 > l

theorem triangle_formation :
  can_form_triangle 5 ∧
  ¬can_form_triangle 2 ∧
  ¬can_form_triangle 8 ∧
  ¬can_form_triangle 11 :=
sorry

end triangle_formation_l1762_176288


namespace actual_distance_travelled_l1762_176233

/-- The actual distance travelled by a person, given two walking speeds and an additional distance condition. -/
theorem actual_distance_travelled (speed1 speed2 additional_distance : ℝ) 
  (h1 : speed1 = 10)
  (h2 : speed2 = 14)
  (h3 : additional_distance = 20)
  (h4 : (actual_distance / speed1) = ((actual_distance + additional_distance) / speed2)) :
  actual_distance = 50 := by
  sorry

end actual_distance_travelled_l1762_176233


namespace belinda_passed_twenty_percent_l1762_176247

-- Define the total number of flyers
def total_flyers : ℕ := 200

-- Define the number of flyers passed out by each person
def ryan_flyers : ℕ := 42
def alyssa_flyers : ℕ := 67
def scott_flyers : ℕ := 51

-- Define Belinda's flyers as the remaining flyers
def belinda_flyers : ℕ := total_flyers - (ryan_flyers + alyssa_flyers + scott_flyers)

-- Define the percentage of flyers Belinda passed out
def belinda_percentage : ℚ := (belinda_flyers : ℚ) / (total_flyers : ℚ) * 100

-- Theorem stating that Belinda passed out 20% of the flyers
theorem belinda_passed_twenty_percent : belinda_percentage = 20 := by sorry

end belinda_passed_twenty_percent_l1762_176247


namespace system_solution_unique_l1762_176222

theorem system_solution_unique :
  ∃! (x y : ℝ), (4 * x - 3 * y = 11) ∧ (2 * x + y = 13) :=
by
  -- The proof goes here
  sorry

end system_solution_unique_l1762_176222


namespace chess_game_outcome_l1762_176292

theorem chess_game_outcome (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.3)
  (h2 : prob_A_not_lose = 0.7) :
  let prob_draw := prob_A_not_lose - prob_A_win
  prob_draw > prob_A_win ∧ prob_draw > (1 - prob_A_not_lose) :=
by sorry

end chess_game_outcome_l1762_176292


namespace line_equation_l1762_176239

-- Define a line by its slope and y-intercept
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point is on a line
def point_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.y_intercept

-- Function to translate a line
def translate_line (l : Line) (dx : ℝ) (dy : ℝ) : Line :=
  { slope := l.slope,
    y_intercept := l.y_intercept - l.slope * dx + dy }

-- Theorem statement
theorem line_equation (l : Line) :
  point_on_line { x := 1, y := 1 } l ∧
  translate_line (translate_line l 2 0) 0 (-1) = l →
  l.slope = 1/2 ∧ l.y_intercept = 1/2 :=
sorry

end line_equation_l1762_176239


namespace f_properties_l1762_176230

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_periodic : ∀ x : ℝ, f (x + 6) = f x + f 3

-- Theorem to prove
theorem f_properties :
  (f 3 = 0) ∧
  (f (-3) = 0) ∧
  (∀ x : ℝ, f (6 + x) = f (6 - x)) :=
by sorry

end f_properties_l1762_176230


namespace sqrt_of_negative_six_squared_l1762_176278

theorem sqrt_of_negative_six_squared (x : ℝ) : Real.sqrt ((-6)^2) = 6 := by
  sorry

end sqrt_of_negative_six_squared_l1762_176278


namespace count_equal_pairs_l1762_176236

/-- Definition of the sequence a_n -/
def a (n : ℕ) : ℤ := n^2 - 22*n + 10

/-- The number of pairs of distinct positive integers (m,n) satisfying a_m = a_n -/
def num_pairs : ℕ := 10

/-- Theorem stating that there are exactly 10 pairs of distinct positive integers (m,n) 
    satisfying a_m = a_n -/
theorem count_equal_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    pairs.card = num_pairs ∧ 
    (∀ (m n : ℕ), (m, n) ∈ pairs ↔ 
      m ≠ n ∧ m > 0 ∧ n > 0 ∧ a m = a n) :=
sorry

end count_equal_pairs_l1762_176236


namespace ceiling_floor_product_l1762_176246

theorem ceiling_floor_product (y : ℝ) : 
  y < 0 → ⌈y⌉ * ⌊y⌋ = 72 → -9 < y ∧ y < -8 := by
  sorry

end ceiling_floor_product_l1762_176246


namespace age_difference_l1762_176215

theorem age_difference (A B : ℕ) : B = 37 → A + 10 = 2 * (B - 10) → A - B = 7 := by
  sorry

end age_difference_l1762_176215


namespace solution_set_l1762_176265

theorem solution_set (x y z : ℝ) : 
  x = (4 * z^2) / (1 + 4 * z^2) ∧
  y = (4 * x^2) / (1 + 4 * x^2) ∧
  z = (4 * y^2) / (1 + 4 * y^2) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by sorry

end solution_set_l1762_176265


namespace ap_has_ten_terms_l1762_176294

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  n : ℕ                -- number of terms
  a : ℝ                -- first term
  d : ℝ                -- common difference
  n_even : Even n
  sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 28
  sum_even : (n / 2) * (2 * a + n * d) = 38
  last_first_diff : a + (n - 1) * d - a = 16

/-- Theorem stating that an arithmetic progression with the given properties has 10 terms -/
theorem ap_has_ten_terms (ap : ArithmeticProgression) : ap.n = 10 := by
  sorry

end ap_has_ten_terms_l1762_176294


namespace quadratic_equation_coefficients_l1762_176241

/-- The coefficients of the quadratic equation in general form -/
def quadratic_coefficients (a b c : ℝ) : ℝ × ℝ := (a, b)

/-- The original quadratic equation -/
def original_equation (x : ℝ) : Prop := 3 * x^2 + 1 = 6 * x

/-- The general form of the quadratic equation -/
def general_form (x : ℝ) : Prop := 3 * x^2 - 6 * x + 1 = 0

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), (∀ x, original_equation x ↔ general_form x) ∧
  quadratic_coefficients a b c = (3, -6) := by sorry

end quadratic_equation_coefficients_l1762_176241


namespace square_roots_problem_l1762_176200

theorem square_roots_problem (m : ℝ) (n : ℝ) (h1 : n > 0) (h2 : 2*m - 1 = (n ^ (1/2 : ℝ))) (h3 : 2 - m = (n ^ (1/2 : ℝ))) : n = 9 := by
  sorry

end square_roots_problem_l1762_176200


namespace necessary_but_not_sufficient_condition_l1762_176289

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (0 < a ∧ a < b → (1 / a > 1 / b)) ∧
  ¬(∀ a b : ℝ, (1 / a > 1 / b) → (0 < a ∧ a < b)) :=
by sorry

end necessary_but_not_sufficient_condition_l1762_176289


namespace chord_length_l1762_176226

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - 4 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C₃ (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 14/5 = 0

-- Define the common chord of C₁ and C₂
def common_chord (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- Theorem statement
theorem chord_length :
  ∃ (a b : ℝ), 
    (∀ x y, C₁ x y ∧ C₂ x y → common_chord x y) ∧
    (∃ x₁ y₁ x₂ y₂, 
      C₃ x₁ y₁ ∧ C₃ x₂ y₂ ∧ 
      common_chord x₁ y₁ ∧ common_chord x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = 4^2) :=
by sorry

end chord_length_l1762_176226


namespace cube_tower_surface_area_l1762_176279

/-- Represents a cube with its side length -/
structure Cube where
  side : ℕ

/-- Calculates the volume of a cube -/
def Cube.volume (c : Cube) : ℕ := c.side ^ 3

/-- Calculates the surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℕ := 6 * c.side ^ 2

/-- Represents the tower of cubes -/
def CubeTower : List Cube := [
  { side := 8 },
  { side := 7 },
  { side := 6 },
  { side := 5 },
  { side := 4 },
  { side := 3 },
  { side := 2 },
  { side := 1 }
]

/-- Calculates the visible surface area of a cube in the tower -/
def visibleSurfaceArea (c : Cube) (isBottom : Bool) : ℕ :=
  if isBottom then
    5 * c.side ^ 2  -- 5 visible faces for bottom cube
  else if c.side = 1 then
    5 * c.side ^ 2  -- 5 visible faces for top cube
  else
    4 * c.side ^ 2  -- 4 visible faces for other cubes (3 full + 2 partial = 4)

/-- Calculates the total visible surface area of the cube tower -/
def totalVisibleSurfaceArea (tower : List Cube) : ℕ :=
  let rec aux (cubes : List Cube) (acc : ℕ) (isFirst : Bool) : ℕ :=
    match cubes with
    | [] => acc
    | c :: rest => aux rest (acc + visibleSurfaceArea c isFirst) false
  aux tower 0 true

/-- The main theorem stating that the total visible surface area of the cube tower is 945 -/
theorem cube_tower_surface_area :
  totalVisibleSurfaceArea CubeTower = 945 := by
  sorry  -- Proof omitted

end cube_tower_surface_area_l1762_176279


namespace tree_growth_rate_l1762_176211

/-- Proves that a tree with given initial and final heights over a specific time period has a certain growth rate per week. -/
theorem tree_growth_rate 
  (initial_height : ℝ) 
  (final_height : ℝ) 
  (months : ℕ) 
  (weeks_per_month : ℕ) 
  (h1 : initial_height = 10)
  (h2 : final_height = 42)
  (h3 : months = 4)
  (h4 : weeks_per_month = 4) :
  (final_height - initial_height) / (months * weeks_per_month : ℝ) = 2 := by
  sorry

#check tree_growth_rate

end tree_growth_rate_l1762_176211


namespace geq_one_necessary_not_sufficient_for_gt_one_l1762_176277

theorem geq_one_necessary_not_sufficient_for_gt_one :
  (∀ x : ℝ, x > 1 → x ≥ 1) ∧
  (∃ x : ℝ, x ≥ 1 ∧ ¬(x > 1)) := by
  sorry

end geq_one_necessary_not_sufficient_for_gt_one_l1762_176277


namespace sqrt_x_minus_one_real_l1762_176273

theorem sqrt_x_minus_one_real (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end sqrt_x_minus_one_real_l1762_176273


namespace difference_calculation_l1762_176296

theorem difference_calculation (total : ℝ) (h : total = 8000) : 
  (1 / 10 : ℝ) * total - (1 / 20 : ℝ) * (1 / 100 : ℝ) * total = 796 := by
  sorry

end difference_calculation_l1762_176296


namespace money_distribution_l1762_176201

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (AC_sum : A + C = 200)
  (C_amount : C = 10) :
  B + C = 310 := by sorry

end money_distribution_l1762_176201


namespace range_of_p_l1762_176298

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - 10*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 2*x - 10

-- Define set A
def A : Set ℝ := {x | f' x ≤ 0}

-- Define set B
def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

-- Theorem statement
theorem range_of_p (p : ℝ) : A ∪ B p = A → p ≤ 3 := by
  sorry

end range_of_p_l1762_176298


namespace ad_eq_bc_necessary_not_sufficient_l1762_176259

/-- A sequence of four non-zero real numbers forms a geometric sequence -/
def IsGeometricSequence (a b c d : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

/-- The condition ad=bc for four non-zero real numbers -/
def AdEqualsBc (a b c d : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ a * d = b * c

theorem ad_eq_bc_necessary_not_sufficient :
  (∀ a b c d : ℝ, IsGeometricSequence a b c d → AdEqualsBc a b c d) ∧
  (∃ a b c d : ℝ, AdEqualsBc a b c d ∧ ¬IsGeometricSequence a b c d) :=
sorry

end ad_eq_bc_necessary_not_sufficient_l1762_176259


namespace thermostat_changes_l1762_176232

theorem thermostat_changes (initial_temp : ℝ) : 
  initial_temp = 40 →
  let doubled := initial_temp * 2
  let after_dad := doubled - 30
  let after_mom := after_dad * 0.7
  let final_temp := after_mom + 24
  final_temp = 59 := by sorry

end thermostat_changes_l1762_176232


namespace polynomial_equality_l1762_176269

theorem polynomial_equality (a k n : ℤ) : 
  (∀ x : ℝ, (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n) → 
  a - n + k = 3 :=
by sorry

end polynomial_equality_l1762_176269


namespace job_completion_proof_l1762_176297

/-- The number of days it takes the initial group of machines to finish the job -/
def initial_days : ℕ := 36

/-- The number of additional machines added -/
def additional_machines : ℕ := 5

/-- The number of days it takes after adding more machines -/
def reduced_days : ℕ := 27

/-- The number of machines initially working on the job -/
def initial_machines : ℕ := 20

theorem job_completion_proof :
  (initial_machines : ℚ) / initial_days = (initial_machines + additional_machines) / reduced_days :=
by sorry

end job_completion_proof_l1762_176297


namespace system_solution_l1762_176272

/-- The system of equations has only two solutions -/
theorem system_solution :
  ∀ x y z : ℝ,
  x + y + z = 13 →
  x^2 + y^2 + z^2 = 61 →
  x*y + x*z = 2*y*z →
  ((x = 4 ∧ y = 3 ∧ z = 6) ∨ (x = 4 ∧ y = 6 ∧ z = 3)) :=
by sorry

end system_solution_l1762_176272


namespace complex_number_quadrant_l1762_176209

theorem complex_number_quadrant (z : ℂ) (h : z * (2 - Complex.I) = 2 + Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end complex_number_quadrant_l1762_176209


namespace sum_of_roots_cubic_l1762_176293

/-- Given a cubic function f and two points (a, f(a)) and (b, f(b)), prove that a + b = -2 --/
theorem sum_of_roots_cubic (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = x^3 + 3*x^2 + 6*x) →
  f a = 1 →
  f b = -9 →
  a + b = -2 :=
by sorry

end sum_of_roots_cubic_l1762_176293


namespace no_solution_exists_l1762_176248

theorem no_solution_exists : ¬∃ (a b c d : ℝ), 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧ 
  (a/b + b/c + c/d + d/a = 6) ∧ 
  (b/a + c/b + d/c + a/d = 32) := by
  sorry

end no_solution_exists_l1762_176248


namespace least_11_heavy_three_digit_is_11_heavy_106_least_11_heavy_three_digit_is_106_l1762_176290

theorem least_11_heavy_three_digit : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 106 → n % 11 ≤ 6 :=
by sorry

theorem is_11_heavy_106 : 106 % 11 > 6 :=
by sorry

theorem least_11_heavy_three_digit_is_106 : 
  ∀ n : ℕ, 100 ≤ n ∧ n % 11 > 6 → n ≥ 106 :=
by sorry

end least_11_heavy_three_digit_is_11_heavy_106_least_11_heavy_three_digit_is_106_l1762_176290


namespace problem_solution_l1762_176251

theorem problem_solution : 
  (Real.sqrt 12 - 3 * Real.sqrt (1/3) + Real.sqrt 8 = Real.sqrt 3 + 2 * Real.sqrt 2) ∧ 
  ((Real.sqrt 5 - 1)^2 + Real.sqrt 5 * (Real.sqrt 5 + 2) = 11) := by
  sorry

end problem_solution_l1762_176251


namespace unique_square_sum_l1762_176299

theorem unique_square_sum : ∃! x : ℕ+, 
  (∃ m : ℕ+, (x : ℕ) + 100 = m^2) ∧ 
  (∃ n : ℕ+, (x : ℕ) + 168 = n^2) :=
by
  -- Proof goes here
  sorry

end unique_square_sum_l1762_176299


namespace keenan_essay_words_l1762_176261

/-- Represents Keenan's essay writing scenario -/
structure EssayWriting where
  initial_rate : ℕ  -- Words per hour for first two hours
  later_rate : ℕ    -- Words per hour after first two hours
  total_time : ℕ    -- Total time available in hours

/-- Calculates the total number of words Keenan can write -/
def total_words (e : EssayWriting) : ℕ :=
  (e.initial_rate * 2) + (e.later_rate * (e.total_time - 2))

/-- Theorem stating that Keenan can write 1200 words given the conditions -/
theorem keenan_essay_words :
  ∃ (e : EssayWriting), e.initial_rate = 400 ∧ e.later_rate = 200 ∧ e.total_time = 4 ∧ total_words e = 1200 := by
  sorry

end keenan_essay_words_l1762_176261


namespace extreme_points_property_l1762_176260

theorem extreme_points_property (a : ℝ) (f : ℝ → ℝ) (x₁ x₂ : ℝ) :
  0 < a → a < 1/2 →
  (∀ x, f x = x * (Real.log x - a * x)) →
  x₁ < x₂ →
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₁ - ε) (x₁ + ε), f x ≤ f x₁) →
  (∃ ε > 0, ∀ x ∈ Set.Ioo (x₂ - ε) (x₂ + ε), f x ≤ f x₂) →
  f x₁ < 0 ∧ f x₂ > -1/2 := by
sorry

end extreme_points_property_l1762_176260


namespace probability_not_hearing_favorite_song_l1762_176213

-- Define the number of songs
def num_songs : ℕ := 12

-- Define the length of the shortest song in seconds
def shortest_song : ℕ := 45

-- Define the common difference between song lengths
def song_length_diff : ℕ := 45

-- Define the length of the favorite song in seconds
def favorite_song_length : ℕ := 375

-- Define the total listening time in seconds
def total_listening_time : ℕ := 420

-- Function to calculate the length of the nth song
def song_length (n : ℕ) : ℕ := shortest_song + (n - 1) * song_length_diff

-- Theorem stating the probability of not hearing the entire favorite song
theorem probability_not_hearing_favorite_song :
  let total_orderings := num_songs.factorial
  let favorable_orderings := 3 * (num_songs - 1).factorial
  (total_orderings - favorable_orderings) / total_orderings = 3 / 4 :=
sorry

end probability_not_hearing_favorite_song_l1762_176213


namespace unique_solution_to_equation_l1762_176282

theorem unique_solution_to_equation : ∃! t : ℝ, 4 * (4 : ℝ)^t + Real.sqrt (16 * 16^t) + 2^t = 34 := by
  sorry

end unique_solution_to_equation_l1762_176282


namespace symmetry_implies_coordinates_l1762_176204

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₂ = -x₁ ∧ y₂ = -y₁

/-- Given points A(a,-2) and B(3,b) are symmetric with respect to the origin, prove a = -3 and b = 2 -/
theorem symmetry_implies_coordinates (a b : ℝ) 
  (h : symmetric_wrt_origin a (-2) 3 b) : a = -3 ∧ b = 2 := by
  sorry

end symmetry_implies_coordinates_l1762_176204


namespace student_weight_l1762_176274

theorem student_weight (student_weight sister_weight : ℝ) 
  (h1 : student_weight - 5 = 2 * sister_weight)
  (h2 : student_weight + sister_weight = 116) :
  student_weight = 79 := by
sorry

end student_weight_l1762_176274


namespace potassium_count_in_compound_l1762_176281

/-- Represents the number of atoms of an element in a compound -/
structure AtomCount where
  k : ℕ  -- number of Potassium atoms
  cr : ℕ -- number of Chromium atoms
  o : ℕ  -- number of Oxygen atoms

/-- Calculates the molecular weight of a compound given its atom counts and atomic weights -/
def molecularWeight (count : AtomCount) (k_weight cr_weight o_weight : ℝ) : ℝ :=
  count.k * k_weight + count.cr * cr_weight + count.o * o_weight

/-- Theorem stating that a compound with 2 Chromium atoms, 7 Oxygen atoms, 
    and a total molecular weight of 296 g/mol must contain 2 Potassium atoms -/
theorem potassium_count_in_compound :
  ∀ (count : AtomCount),
    count.cr = 2 →
    count.o = 7 →
    molecularWeight count 39.1 52.0 16.0 = 296 →
    count.k = 2 := by
  sorry

end potassium_count_in_compound_l1762_176281


namespace quadratic_transformation_l1762_176207

-- Define the coefficients of the quadratic equation
variable (a b c : ℝ)

-- Define the condition that ax^2 + bx + c can be expressed as 3(x - 5)^2 + 7
def quadratic_condition (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 3 * (x - 5)^2 + 7

-- Define the expanded form of 4ax^2 + 4bx + 4c
def expanded_quadratic (x : ℝ) : ℝ :=
  4 * a * x^2 + 4 * b * x + 4 * c

-- Theorem statement
theorem quadratic_transformation (h : ∀ x, quadratic_condition a b c x) :
  ∃ (n k : ℝ), ∀ x, expanded_quadratic a b c x = n * (x - 5)^2 + k :=
sorry

end quadratic_transformation_l1762_176207


namespace farm_feet_count_l1762_176268

/-- Given a farm with hens and cows, prove the total number of feet -/
theorem farm_feet_count (total_heads : ℕ) (hen_count : ℕ) : 
  total_heads = 50 → hen_count = 28 → 
  (hen_count * 2 + (total_heads - hen_count) * 4 = 144) :=
by
  sorry

#check farm_feet_count

end farm_feet_count_l1762_176268


namespace parallelogram_side_sum_l1762_176254

/-- Given a parallelogram with consecutive side lengths 10, 5y+3, 12, and 4x-1, prove that x + y = 91/20 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (4 * x - 1 = 10) →   -- First pair of opposite sides
  (5 * y + 3 = 12) →   -- Second pair of opposite sides
  x + y = 91/20 := by sorry

end parallelogram_side_sum_l1762_176254


namespace three_geometric_sequences_l1762_176264

/-- An arithmetic sequence starting with 1 -/
structure ArithmeticSequence :=
  (d : ℝ)
  (a₁ : ℝ := 1 + d)
  (a₂ : ℝ := 1 + 2*d)
  (a₃ : ℝ := 1 + 3*d)
  (positive : 0 < a₁ ∧ 0 < a₂ ∧ 0 < a₃)

/-- A function that counts the number of geometric sequences that can be formed from 1 and the terms of an arithmetic sequence -/
def countGeometricSequences (seq : ArithmeticSequence) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly 3 geometric sequences -/
theorem three_geometric_sequences (seq : ArithmeticSequence) : 
  countGeometricSequences seq = 3 :=
sorry

end three_geometric_sequences_l1762_176264


namespace polynomial_division_remainder_l1762_176229

theorem polynomial_division_remainder : ∃ q r : Polynomial ℤ,
  (3 * X^4 + 14 * X^3 - 35 * X^2 - 80 * X + 56) = 
  (X^2 + 8 * X - 6) * q + r ∧ 
  r.degree < 2 ∧ 
  r = 364 * X - 322 :=
sorry

end polynomial_division_remainder_l1762_176229


namespace max_attendance_l1762_176270

/-- Represents the number of students that can attend an event --/
structure EventAttendance where
  boys : ℕ
  girls : ℕ

/-- Represents the capacities of the three auditoriums --/
structure AuditoriumCapacities where
  A : ℕ
  B : ℕ
  C : ℕ

/-- Checks if the attendance satisfies the given conditions --/
def satisfiesConditions (attendance : EventAttendance) (capacities : AuditoriumCapacities) : Prop :=
  -- The ratio of boys to girls is 7:11
  11 * attendance.boys = 7 * attendance.girls
  -- There are 72 more girls than boys
  ∧ attendance.girls = attendance.boys + 72
  -- The total attendance doesn't exceed any individual auditorium's capacity
  ∧ attendance.boys + attendance.girls ≤ capacities.A
  ∧ attendance.boys + attendance.girls ≤ capacities.B
  ∧ attendance.boys + attendance.girls ≤ capacities.C

/-- The main theorem stating the maximum number of students that can attend --/
theorem max_attendance (capacities : AuditoriumCapacities)
    (hA : capacities.A = 180)
    (hB : capacities.B = 220)
    (hC : capacities.C = 150) :
    ∃ (attendance : EventAttendance),
      satisfiesConditions attendance capacities
      ∧ ∀ (other : EventAttendance),
          satisfiesConditions other capacities →
          attendance.boys + attendance.girls ≥ other.boys + other.girls
      ∧ attendance.boys + attendance.girls = 324 :=
  sorry


end max_attendance_l1762_176270


namespace suitcase_electronics_weight_l1762_176238

/-- Given a suitcase with books, clothes, and electronics, prove the weight of electronics -/
theorem suitcase_electronics_weight 
  (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive
  (h2 : 4 * x > 7) -- Ensure we can remove 7 pounds of clothing
  (h3 : 5 * x / (4 * x - 7) = 5 / 2) -- Ratio doubles after removing 7 pounds
  : 2 * x = 7 := by
  sorry

end suitcase_electronics_weight_l1762_176238


namespace number_equation_proof_l1762_176214

theorem number_equation_proof : ∃ n : ℝ, n + 11.95 - 596.95 = 3054 ∧ n = 3639 := by
  sorry

end number_equation_proof_l1762_176214


namespace no_real_roots_quadratic_l1762_176219

theorem no_real_roots_quadratic (c : ℝ) :
  (∀ x : ℝ, x^2 + x - c ≠ 0) → c < -1/4 := by
  sorry

end no_real_roots_quadratic_l1762_176219


namespace gcd_lcm_product_l1762_176243

theorem gcd_lcm_product (a b : ℕ) : Nat.gcd a b * Nat.lcm a b = a * b := by
  sorry

end gcd_lcm_product_l1762_176243


namespace decreasing_g_implies_a_bound_f_nonpositive_implies_a_bound_l1762_176221

noncomputable section

variables (a b : ℝ)

def f (x : ℝ) : ℝ := Real.log x + a * x - 1 / x + b

def g (x : ℝ) : ℝ := f a b x + 2 / x

theorem decreasing_g_implies_a_bound :
  (∀ x > 0, ∀ y > 0, x < y → g a b y < g a b x) →
  a ≤ -1/4 := by sorry

theorem f_nonpositive_implies_a_bound :
  (∀ x > 0, f a b x ≤ 0) →
  a ≤ 1 - b := by sorry

end decreasing_g_implies_a_bound_f_nonpositive_implies_a_bound_l1762_176221


namespace inequality_system_solution_l1762_176263

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, (x + 2*a > 4 ∧ 2*x - b < 5) ↔ (0 < x ∧ x < 2)) →
  (a + b)^2023 = 1 := by
sorry

end inequality_system_solution_l1762_176263


namespace divisibility_property_l1762_176256

theorem divisibility_property (a b c d u : ℤ) 
  (h1 : u ∣ a * c) 
  (h2 : u ∣ b * c + a * d) 
  (h3 : u ∣ b * d) : 
  (u ∣ b * c) ∧ (u ∣ a * d) := by
  sorry

end divisibility_property_l1762_176256


namespace product_factor_adjustment_l1762_176266

theorem product_factor_adjustment (a b c : ℝ) (h1 : a * b = c) (h2 : a / 100 * (b * 100) = c) : 
  b * 100 = b * 100 := by sorry

end product_factor_adjustment_l1762_176266


namespace fraction_simplification_l1762_176287

theorem fraction_simplification (y : ℝ) (h : y = 3) : 
  (y^6 + 8*y^3 + 16) / (y^3 + 4) = 31 := by
  sorry

end fraction_simplification_l1762_176287


namespace equation_rewrite_l1762_176224

theorem equation_rewrite (x y : ℝ) : 5 * x + 3 * y = 1 ↔ y = (1 - 5 * x) / 3 := by
  sorry

end equation_rewrite_l1762_176224


namespace x_times_one_minus_f_equals_one_l1762_176255

/-- Given x = (3 + 2√2)^1000, n = ⌊x⌋, and f = x - n, prove that x(1 - f) = 1 -/
theorem x_times_one_minus_f_equals_one :
  let x : ℝ := (3 + 2 * Real.sqrt 2) ^ 1000
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by
sorry

end x_times_one_minus_f_equals_one_l1762_176255


namespace candy_distribution_l1762_176271

/-- The number of distinct pieces of candy --/
def n : ℕ := 8

/-- The number of bags --/
def k : ℕ := 3

/-- The number of ways to distribute n distinct objects into k groups,
    where each group must have at least one object --/
def distribute_distinct (n k : ℕ) : ℕ :=
  (n - k + 1).choose (k - 1) * n.factorial

theorem candy_distribution :
  distribute_distinct n k = 846720 := by
  sorry

end candy_distribution_l1762_176271


namespace equation_solution_l1762_176223

theorem equation_solution : ∃ x : ℝ, (x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) ∧ 
                            (2 * x - 5 ≠ 0) ∧ (5 - 2 * x ≠ 0) ∧ (x = 0) := by
  sorry

end equation_solution_l1762_176223


namespace comparison_theorem_l1762_176275

theorem comparison_theorem :
  (5.6 - 7/8 > 4.6) ∧ (638/81 < 271/29) := by
  sorry

end comparison_theorem_l1762_176275


namespace average_equation_solution_l1762_176253

theorem average_equation_solution (y : ℚ) : 
  (1 / 3 : ℚ) * ((y + 10) + (5 * y + 4) + (3 * y + 12)) = 6 * y - 8 → y = 50 / 9 := by
  sorry

end average_equation_solution_l1762_176253


namespace compound_interest_rate_l1762_176283

theorem compound_interest_rate (P : ℝ) (r : ℝ) 
  (h1 : P * (1 + r / 100)^2 = 2420)
  (h2 : P * (1 + r / 100)^3 = 2662) :
  r = 10 := by
  sorry

end compound_interest_rate_l1762_176283


namespace f_monotone_decreasing_on_zero_one_l1762_176249

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - Real.log x

theorem f_monotone_decreasing_on_zero_one :
  ∀ x ∈ Set.Ioo (0 : ℝ) 1, StrictMonoOn f (Set.Ioo (0 : ℝ) 1) :=
sorry

end f_monotone_decreasing_on_zero_one_l1762_176249


namespace f_three_fourths_equals_three_l1762_176257

-- Define g(x)
def g (x : ℝ) : ℝ := 1 - x^2

-- Define f(g(x))
noncomputable def f (y : ℝ) : ℝ :=
  if y ≠ 1 then (1 - (1 - y)) / (1 - y) else 0

-- Theorem statement
theorem f_three_fourths_equals_three : f (3/4) = 3 := by
  sorry

end f_three_fourths_equals_three_l1762_176257


namespace arithmetic_mean_problem_l1762_176276

theorem arithmetic_mean_problem (m n : ℝ) 
  (h1 : (m + 2*n) / 2 = 4)
  (h2 : (2*m + n) / 2 = 5) :
  (m + n) / 2 = 3.5 := by
  sorry

end arithmetic_mean_problem_l1762_176276


namespace desmond_toy_purchase_l1762_176237

/-- The number of toys Mr. Desmond bought for his elder son -/
def elder_son_toys : ℕ := 60

/-- The number of toys Mr. Desmond bought for his younger son -/
def younger_son_toys : ℕ := 3 * elder_son_toys

/-- The total number of toys Mr. Desmond bought -/
def total_toys : ℕ := elder_son_toys + younger_son_toys

theorem desmond_toy_purchase :
  total_toys = 240 := by sorry

end desmond_toy_purchase_l1762_176237


namespace sin_cos_15_product_eq_neg_sqrt3_div_2_l1762_176203

theorem sin_cos_15_product_eq_neg_sqrt3_div_2 :
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) *
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end sin_cos_15_product_eq_neg_sqrt3_div_2_l1762_176203


namespace second_player_winning_strategy_l1762_176206

/-- Represents a node in the hexagonal grid --/
structure Node :=
  (x : ℤ)
  (y : ℤ)
  (z : ℤ)
  (sum_zero : x + y + z = 0)

/-- Represents the hexagonal grid game --/
structure HexagonGame :=
  (n : ℕ)
  (current_player : ℕ)
  (token : Node)
  (visited : Set Node)

/-- Defines a valid move in the game --/
def valid_move (game : HexagonGame) (new_pos : Node) : Prop :=
  (abs (new_pos.x - game.token.x) + abs (new_pos.y - game.token.y) + abs (new_pos.z - game.token.z) = 2) ∧
  (new_pos ∉ game.visited)

/-- Defines the winning condition for the second player --/
def second_player_wins (n : ℕ) : Prop :=
  ∀ (game : HexagonGame),
    game.n = n →
    (game.current_player = 1 → ∃ (move : Node), valid_move game move) →
    (game.current_player = 2 → ∀ (move : Node), valid_move game move → 
      ∃ (counter_move : Node), valid_move (HexagonGame.mk n 1 move (game.visited.insert game.token)) counter_move)

/-- The main theorem: The second player has a winning strategy for all n --/
theorem second_player_winning_strategy :
  ∀ n : ℕ, second_player_wins n :=
sorry

end second_player_winning_strategy_l1762_176206


namespace mod_seven_difference_l1762_176262

theorem mod_seven_difference (n : ℕ) : (47^824 - 25^824) % 7 = 0 := by
  sorry

end mod_seven_difference_l1762_176262


namespace complex_coordinate_to_z_l1762_176202

theorem complex_coordinate_to_z (z : ℂ) :
  (z / Complex.I).re = 3 ∧ (z / Complex.I).im = -1 → z = 1 + 3 * Complex.I :=
by sorry

end complex_coordinate_to_z_l1762_176202


namespace no_infinite_sequence_exists_l1762_176217

theorem no_infinite_sequence_exists : 
  ¬ ∃ (a : ℕ → ℕ), ∀ (n : ℕ), 
    a (n + 2) = a (n + 1) + Real.sqrt (a (n + 1) + a n : ℝ) := by
  sorry

end no_infinite_sequence_exists_l1762_176217


namespace roads_per_neighborhood_is_four_l1762_176286

/-- The number of roads passing through each neighborhood in a town with the following properties:
  * The town has 10 neighborhoods.
  * Each road has 250 street lights on each opposite side.
  * The total number of street lights in the town is 20000.
-/
def roads_per_neighborhood : ℕ := 4

/-- Theorem stating that the number of roads passing through each neighborhood is 4 -/
theorem roads_per_neighborhood_is_four :
  let neighborhoods : ℕ := 10
  let lights_per_side : ℕ := 250
  let total_lights : ℕ := 20000
  roads_per_neighborhood * neighborhoods * (2 * lights_per_side) = total_lights :=
by sorry

end roads_per_neighborhood_is_four_l1762_176286


namespace light_bulb_conditional_probability_l1762_176284

theorem light_bulb_conditional_probability 
  (p_3000 : ℝ) 
  (p_4500 : ℝ) 
  (h1 : p_3000 = 0.8) 
  (h2 : p_4500 = 0.2) 
  (h3 : p_3000 ≠ 0) : 
  p_4500 / p_3000 = 0.25 := by
sorry

end light_bulb_conditional_probability_l1762_176284


namespace evaluate_expression_l1762_176235

theorem evaluate_expression : 3^5 * 6^5 * 3^6 * 6^6 = 18^11 := by
  sorry

end evaluate_expression_l1762_176235


namespace sufficient_but_not_necessary_l1762_176210

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧ 
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x < -1)) := by
  sorry

end sufficient_but_not_necessary_l1762_176210


namespace certain_number_problem_l1762_176240

theorem certain_number_problem :
  ∃ x : ℝ, x ≥ 0 ∧ 5 * (Real.sqrt x + 3) = 19 ∧ x = 0.64 := by
  sorry

end certain_number_problem_l1762_176240


namespace nanoseconds_to_scientific_notation_l1762_176267

/-- Conversion factor from nanoseconds to seconds -/
def nanosecond_to_second : ℝ := 1e-9

/-- The number of nanoseconds we want to convert -/
def nanoseconds : ℝ := 20

/-- The expected result in scientific notation (in seconds) -/
def expected_result : ℝ := 2e-8

theorem nanoseconds_to_scientific_notation :
  nanoseconds * nanosecond_to_second = expected_result := by
  sorry

end nanoseconds_to_scientific_notation_l1762_176267


namespace triangle_area_proof_l1762_176220

/-- Represents a triangle with parallel lines -/
structure TriangleWithParallelLines where
  /-- The area of the largest part -/
  largest_part_area : ℝ
  /-- The number of parallel lines -/
  num_parallel_lines : ℕ
  /-- The number of equal segments on the other two sides -/
  num_segments : ℕ
  /-- The number of parts the triangle is divided into -/
  num_parts : ℕ

/-- Theorem: If a triangle with 9 parallel lines dividing the sides into 10 equal segments
    has its largest part with an area of 38, then the total area of the triangle is 200 -/
theorem triangle_area_proof (t : TriangleWithParallelLines)
    (h1 : t.largest_part_area = 38)
    (h2 : t.num_parallel_lines = 9)
    (h3 : t.num_segments = 10)
    (h4 : t.num_parts = 10) :
    ∃ (total_area : ℝ), total_area = 200 := by
  sorry

end triangle_area_proof_l1762_176220


namespace find_other_number_l1762_176291

theorem find_other_number (a b : ℕ+) 
  (h1 : Nat.lcm a b = 4620)
  (h2 : Nat.gcd a b = 21)
  (h3 : a = 210) :
  b = 462 := by
  sorry

end find_other_number_l1762_176291


namespace postcard_area_l1762_176228

/-- Represents a rectangular postcard -/
structure Postcard where
  vertical_length : ℝ
  horizontal_length : ℝ

/-- Calculates the area of a postcard -/
def area (p : Postcard) : ℝ := p.vertical_length * p.horizontal_length

/-- Calculates the perimeter of two attached postcards -/
def attached_perimeter (p : Postcard) : ℝ := 2 * p.vertical_length + 4 * p.horizontal_length

theorem postcard_area (p : Postcard) 
  (h1 : p.vertical_length = 15)
  (h2 : attached_perimeter p = 70) : 
  area p = 150 := by
  sorry

#check postcard_area

end postcard_area_l1762_176228


namespace coefficient_of_x_l1762_176242

theorem coefficient_of_x (some_number : ℝ) : 
  (2 * (1/2)^2 + some_number * (1/2) - 5 = 0) → some_number = 9 := by
  sorry

end coefficient_of_x_l1762_176242


namespace five_letter_words_count_l1762_176280

/-- The number of vowels in the alphabet -/
def num_vowels : ℕ := 5

/-- The number of consonants in the alphabet -/
def num_consonants : ℕ := 21

/-- The total number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of five-letter words starting with a vowel and ending with a consonant -/
def num_words : ℕ := num_vowels * num_letters * num_letters * num_letters * num_consonants

theorem five_letter_words_count : num_words = 1844760 := by
  sorry

end five_letter_words_count_l1762_176280


namespace base_eight_subtraction_l1762_176258

/-- Represents a number in base 8 --/
def BaseEight : Type := Nat

/-- Converts a base 8 number to its decimal representation --/
def to_decimal (n : BaseEight) : Nat := sorry

/-- Converts a decimal number to its base 8 representation --/
def from_decimal (n : Nat) : BaseEight := sorry

/-- Subtracts two base 8 numbers --/
def base_eight_sub (a b : BaseEight) : BaseEight := sorry

theorem base_eight_subtraction :
  base_eight_sub (from_decimal 4765) (from_decimal 2314) = from_decimal 2447 := by sorry

end base_eight_subtraction_l1762_176258


namespace quadratic_real_roots_l1762_176245

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + (2*m - 3)*x + (m^2 - 3) = 0) ↔ m ≤ 7/4 := by sorry

end quadratic_real_roots_l1762_176245


namespace penny_socks_l1762_176285

def sock_problem (initial_amount : ℕ) (sock_cost : ℕ) (hat_cost : ℕ) (remaining_amount : ℕ) : Prop :=
  ∃ (num_socks : ℕ), 
    initial_amount = sock_cost * num_socks + hat_cost + remaining_amount

theorem penny_socks : sock_problem 20 2 7 5 → ∃ (num_socks : ℕ), num_socks = 4 := by
  sorry

end penny_socks_l1762_176285


namespace hyperbola_focus_parameter_l1762_176208

/-- Given a hyperbola with equation y²/m - x²/9 = 1 and a focus at (0, 5),
    prove that m = 16. -/
theorem hyperbola_focus_parameter (m : ℝ) : 
  (∀ x y : ℝ, y^2/m - x^2/9 = 1 → (x = 0 ∧ y = 5) → m = 16) :=
by sorry

end hyperbola_focus_parameter_l1762_176208


namespace y_plus_z_squared_positive_l1762_176225

theorem y_plus_z_squared_positive 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 2 < z ∧ z < 3) : 
  0 < y + z^2 := by
  sorry

end y_plus_z_squared_positive_l1762_176225


namespace mod_equivalence_unique_solution_l1762_176234

theorem mod_equivalence_unique_solution :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 11 ∧ n ≡ -5033 [ZMOD 12] := by sorry

end mod_equivalence_unique_solution_l1762_176234


namespace min_max_values_l1762_176244

theorem min_max_values (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) 
  (h : x^2 + 4*y^2 + 4*x*y + 4*x^2*y^2 = 32) : 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a^2 + 4*b^2 + 4*a*b + 4*a^2*b^2 = 32 → x + 2*y ≤ a + 2*b) ∧ 
  (∀ a b : ℝ, a ≥ 0 → b ≥ 0 → a^2 + 4*b^2 + 4*a*b + 4*a^2*b^2 = 32 → 
    Real.sqrt 7 * (x + 2*y) + 2*x*y ≥ Real.sqrt 7 * (a + 2*b) + 2*a*b) ∧
  x + 2*y = 4 ∧ 
  Real.sqrt 7 * (x + 2*y) + 2*x*y = 4 * Real.sqrt 7 + 4 := by
  sorry

end min_max_values_l1762_176244


namespace selling_price_increase_for_3360_profit_max_profit_at_10_yuan_increase_l1762_176252

/-- Represents the profit function for T-shirt sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 3000

/-- Represents the constraint for a specific profit -/
def profit_constraint (x : ℝ) : Prop := profit_function x = 3360

/-- Theorem: The selling price increase that results in a profit of 3360 yuan is 2 yuan -/
theorem selling_price_increase_for_3360_profit :
  ∃ x : ℝ, profit_constraint x ∧ x = 2 := by sorry

/-- Theorem: The maximum profit occurs when the selling price is increased by 10 yuan, resulting in a profit of 4000 yuan -/
theorem max_profit_at_10_yuan_increase :
  ∃ x : ℝ, x = 10 ∧ profit_function x = 4000 ∧ 
  ∀ y : ℝ, profit_function y ≤ profit_function x := by sorry

end selling_price_increase_for_3360_profit_max_profit_at_10_yuan_increase_l1762_176252


namespace caseys_corn_rows_l1762_176205

/-- Represents the problem of calculating the number of corn plant rows Casey can water --/
theorem caseys_corn_rows :
  let pump_rate : ℚ := 3  -- gallons per minute
  let pump_time : ℕ := 25  -- minutes
  let plants_per_row : ℕ := 15
  let water_per_plant : ℚ := 1/2  -- gallons
  let num_pigs : ℕ := 10
  let water_per_pig : ℚ := 4  -- gallons
  let num_ducks : ℕ := 20
  let water_per_duck : ℚ := 1/4  -- gallons
  
  let total_water : ℚ := pump_rate * pump_time
  let water_for_animals : ℚ := num_pigs * water_per_pig + num_ducks * water_per_duck
  let water_for_plants : ℚ := total_water - water_for_animals
  let num_plants : ℚ := water_for_plants / water_per_plant
  let num_rows : ℚ := num_plants / plants_per_row
  
  num_rows = 4 := by sorry

end caseys_corn_rows_l1762_176205


namespace function_value_at_six_l1762_176212

/-- Given a function f such that f(4x+2) = x^2 - x + 1 for all real x, prove that f(6) = 1/2 -/
theorem function_value_at_six (f : ℝ → ℝ) (h : ∀ x : ℝ, f (4 * x + 2) = x^2 - x + 1) : f 6 = 1/2 := by
  sorry

end function_value_at_six_l1762_176212


namespace negation_of_forall_leq_zero_l1762_176227

theorem negation_of_forall_leq_zero :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) := by sorry

end negation_of_forall_leq_zero_l1762_176227
