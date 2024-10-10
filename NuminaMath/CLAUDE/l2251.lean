import Mathlib

namespace celine_erasers_l2251_225175

/-- Proves that Celine collected 10 erasers given the conditions of the problem -/
theorem celine_erasers (gabriel : ℕ) (celine : ℕ) (julian : ℕ) : 
  celine = 2 * gabriel → 
  julian = 2 * celine → 
  gabriel + celine + julian = 35 → 
  celine = 10 := by
sorry

end celine_erasers_l2251_225175


namespace sum_of_last_two_digits_11_pow_2002_l2251_225115

theorem sum_of_last_two_digits_11_pow_2002 : ∃ n : ℕ, 
  11^2002 = n * 100 + 21 ∧ n ≥ 0 :=
sorry

end sum_of_last_two_digits_11_pow_2002_l2251_225115


namespace projectile_height_l2251_225108

theorem projectile_height (t : ℝ) : 
  (∃ t₀ : ℝ, t₀ > 0 ∧ -4.9 * t₀^2 + 30 * t₀ = 35 ∧ 
   ∀ t' : ℝ, t' > 0 ∧ -4.9 * t'^2 + 30 * t' = 35 → t₀ ≤ t') → 
  t = 10/7 := by
sorry

end projectile_height_l2251_225108


namespace ellipse_eccentricity_l2251_225142

/-- Given an ellipse with equation x²/a² + y²/4 = 1 and one focus at (2,0),
    prove that its eccentricity is √2/2 -/
theorem ellipse_eccentricity (a : ℝ) (h : a > 0) :
  let c := 2  -- distance from center to focus
  let b := 2  -- √4, as y²/4 = 1 in the equation
  let e := c / a  -- definition of eccentricity
  (∀ x y, x^2 / a^2 + y^2 / 4 = 1 → (x - c)^2 + y^2 = a^2) →  -- ellipse definition
  e = Real.sqrt 2 / 2 :=
by sorry

end ellipse_eccentricity_l2251_225142


namespace red_toy_percentage_l2251_225138

/-- Represents a toy production lot -/
structure ToyLot where
  total : ℕ
  red : ℕ
  green : ℕ
  small : ℕ
  large : ℕ
  redSmall : ℕ
  redLarge : ℕ
  greenLarge : ℕ

/-- The conditions of the toy production lot -/
def validToyLot (lot : ToyLot) : Prop :=
  lot.total > 0 ∧
  lot.red + lot.green = lot.total ∧
  lot.small + lot.large = lot.total ∧
  lot.small = lot.large ∧
  lot.redSmall = (lot.total * 10) / 100 ∧
  lot.greenLarge = 40 ∧
  lot.redLarge = 60

/-- The theorem stating the percentage of red toys -/
theorem red_toy_percentage (lot : ToyLot) :
  validToyLot lot → (lot.red : ℚ) / lot.total = 2/5 := by
  sorry

end red_toy_percentage_l2251_225138


namespace inequality_solution_exists_implies_a_leq_4_l2251_225159

theorem inequality_solution_exists_implies_a_leq_4 :
  (∃ x : ℝ, |x - 2| - |x + 2| ≥ a) → a ≤ 4 :=
by sorry

end inequality_solution_exists_implies_a_leq_4_l2251_225159


namespace existence_of_distinct_pairs_l2251_225106

theorem existence_of_distinct_pairs 
  (S T : Type) [Finite S] [Finite T] 
  (U : Set (S × T)) 
  (h1 : ∀ s : S, ∃ t : T, (s, t) ∉ U) 
  (h2 : ∀ t : T, ∃ s : S, (s, t) ∈ U) :
  ∃ (s₁ s₂ : S) (t₁ t₂ : T), 
    s₁ ≠ s₂ ∧ t₁ ≠ t₂ ∧ 
    (s₁, t₁) ∈ U ∧ (s₂, t₂) ∈ U ∧ 
    (s₁, t₂) ∉ U ∧ (s₂, t₁) ∉ U :=
by sorry

end existence_of_distinct_pairs_l2251_225106


namespace football_shaped_area_l2251_225167

/-- The area of two quarter-circle sectors minus two right triangles in a square with side length 4 -/
theorem football_shaped_area (π : ℝ) (h_π : π = Real.pi) : 
  let side_length : ℝ := 4
  let diagonal : ℝ := side_length * Real.sqrt 2
  let quarter_circle_area : ℝ := (π * diagonal^2) / 4
  let triangle_area : ℝ := side_length^2 / 2
  2 * (quarter_circle_area - triangle_area) = 16 * π - 16 := by
sorry

end football_shaped_area_l2251_225167


namespace congruent_rectangle_perimeter_l2251_225156

/-- Given a rectangle of dimensions a × b units divided into a smaller rectangle of dimensions p × q units
    and four congruent rectangles, the perimeter of one of the four congruent rectangles is 2(a + b - p - q) units. -/
theorem congruent_rectangle_perimeter
  (a b p q : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : p > 0)
  (h4 : q > 0)
  (h5 : p < a)
  (h6 : q < b) :
  ∃ (l1 l2 : ℝ), l1 = b - q ∧ l2 = a - p ∧ 2 * (l1 + l2) = 2 * (a + b - p - q) :=
sorry

end congruent_rectangle_perimeter_l2251_225156


namespace weight_replacement_l2251_225163

theorem weight_replacement (n : ℕ) (new_weight avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : new_weight = 81)
  (h3 : avg_increase = 2) :
  let total_increase := n * avg_increase
  let replaced_weight := new_weight - total_increase
  replaced_weight = 65 := by sorry

end weight_replacement_l2251_225163


namespace even_function_implies_k_equals_one_l2251_225104

/-- A function f is even if f(-x) = f(x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = kx^2 + (k-1)x + 3 --/
def f (k : ℝ) (x : ℝ) : ℝ :=
  k * x^2 + (k - 1) * x + 3

/-- If f(x) = kx^2 + (k-1)x + 3 is an even function, then k = 1 --/
theorem even_function_implies_k_equals_one :
  ∀ k : ℝ, IsEven (f k) → k = 1 := by
  sorry

end even_function_implies_k_equals_one_l2251_225104


namespace max_correct_answers_is_30_l2251_225139

/-- Represents the scoring system and results of a math contest. -/
structure ContestResult where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correct answers possible given a contest result. -/
def max_correct_answers (result : ContestResult) : ℕ :=
  sorry

/-- Theorem stating that for the given contest parameters, the maximum number of correct answers is 30. -/
theorem max_correct_answers_is_30 :
  let result : ContestResult := {
    total_questions := 50,
    correct_points := 5,
    incorrect_points := -2,
    total_score := 115
  }
  max_correct_answers result = 30 := by sorry

end max_correct_answers_is_30_l2251_225139


namespace calculate_expression_l2251_225116

theorem calculate_expression : -5 + 2 * (-3) + (-12) / (-2) = -5 := by
  sorry

end calculate_expression_l2251_225116


namespace area_of_specific_l_shaped_figure_l2251_225126

/-- Represents an L-shaped figure with given dimensions -/
structure LShapedFigure where
  bottom_length : ℕ
  bottom_width : ℕ
  central_length : ℕ
  central_width : ℕ
  top_length : ℕ
  top_width : ℕ

/-- Calculates the area of an L-shaped figure -/
def area_of_l_shaped_figure (f : LShapedFigure) : ℕ :=
  f.bottom_length * f.bottom_width +
  f.central_length * f.central_width +
  f.top_length * f.top_width

/-- Theorem stating that the area of the given L-shaped figure is 81 square units -/
theorem area_of_specific_l_shaped_figure :
  let f : LShapedFigure := {
    bottom_length := 10,
    bottom_width := 6,
    central_length := 4,
    central_width := 4,
    top_length := 5,
    top_width := 1
  }
  area_of_l_shaped_figure f = 81 := by
  sorry

end area_of_specific_l_shaped_figure_l2251_225126


namespace arrangement_theorem_l2251_225166

/-- The number of ways to arrange 5 people in 5 seats with exactly 2 matching --/
def arrangement_count : ℕ := 20

/-- The number of ways to choose 2 items from 5 --/
def choose_two_from_five : ℕ := 10

/-- The number of ways to arrange the remaining 3 people --/
def arrange_remaining : ℕ := 2

theorem arrangement_theorem : 
  arrangement_count = choose_two_from_five * arrange_remaining := by
  sorry


end arrangement_theorem_l2251_225166


namespace parabola_c_value_l2251_225145

/-- A parabola with equation x = ay^2 + by + c, vertex (4, 1), and passing through point (-1, 3) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_x : ℝ := 4
  vertex_y : ℝ := 1
  point_x : ℝ := -1
  point_y : ℝ := 3
  eq_vertex : 4 = a * 1^2 + b * 1 + c
  eq_point : -1 = a * 3^2 + b * 3 + c

/-- The value of c for the given parabola is 11/4 -/
theorem parabola_c_value (p : Parabola) : p.c = 11/4 := by
  sorry

end parabola_c_value_l2251_225145


namespace cookies_theorem_l2251_225107

def cookies_problem (total_cookies : ℕ) : Prop :=
  let father_cookies := (total_cookies : ℚ) * (1 / 10)
  let mother_cookies := father_cookies / 2
  let brother_cookies := mother_cookies + 2
  let sister_cookies := brother_cookies * (3 / 2)
  let aunt_cookies := father_cookies * 2
  let cousin_cookies := aunt_cookies * (4 / 5)
  let grandmother_cookies := cousin_cookies / 3
  let eaten_cookies := father_cookies + mother_cookies + brother_cookies + 
                       sister_cookies + aunt_cookies + cousin_cookies + 
                       grandmother_cookies
  let monica_cookies := total_cookies - eaten_cookies.floor
  monica_cookies = 120

theorem cookies_theorem : cookies_problem 400 := by
  sorry

end cookies_theorem_l2251_225107


namespace fraction_difference_l2251_225128

theorem fraction_difference (A B C : ℚ) (k m : ℕ) : 
  A = 3 * k / (2 * m) →
  B = 2 * k / (3 * m) →
  C = k / (4 * m) →
  A + B + C = 29 / 60 →
  A - B - C = 7 / 60 := by
sorry

end fraction_difference_l2251_225128


namespace square_ratio_problem_l2251_225194

theorem square_ratio_problem (area_ratio : ℚ) (a b c : ℕ) :
  area_ratio = 48 / 125 →
  (a : ℚ) * Real.sqrt b / c = Real.sqrt (area_ratio) →
  a = 4 ∧ b = 15 ∧ c = 25 ∧ a + b + c = 44 := by
  sorry

end square_ratio_problem_l2251_225194


namespace jelly_bean_probability_l2251_225193

theorem jelly_bean_probability (p_red p_green p_orange_yellow : ℝ) :
  p_red = 0.25 →
  p_green = 0.35 →
  p_red + p_green + p_orange_yellow = 1 →
  p_orange_yellow = 0.4 :=
by sorry

end jelly_bean_probability_l2251_225193


namespace min_value_trig_expression_l2251_225112

theorem min_value_trig_expression (x : ℝ) : 
  (Real.sin x)^4 + (Real.cos x)^4 + 2 ≥ (2/3) * ((Real.sin x)^2 + (Real.cos x)^2 + 2) := by
  sorry

end min_value_trig_expression_l2251_225112


namespace base_nine_solution_l2251_225147

/-- Convert a list of digits in base b to its decimal representation -/
def toDecimal (digits : List ℕ) (b : ℕ) : ℕ :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Check if the equation is valid in base b -/
def isValidEquation (b : ℕ) : Prop :=
  toDecimal [5, 7, 4, 2] b + toDecimal [6, 9, 3, 1] b = toDecimal [1, 2, 7, 7, 3] b

theorem base_nine_solution :
  ∃ (b : ℕ), b > 1 ∧ isValidEquation b ∧ ∀ (x : ℕ), x > 1 ∧ x ≠ b → ¬isValidEquation x :=
by sorry

end base_nine_solution_l2251_225147


namespace total_weight_of_balls_l2251_225150

theorem total_weight_of_balls (blue_weight brown_weight : ℝ) :
  blue_weight = 6 → brown_weight = 3.12 →
  blue_weight + brown_weight = 9.12 := by
  sorry

end total_weight_of_balls_l2251_225150


namespace profit_calculation_l2251_225121

/-- Profit calculation for a company --/
theorem profit_calculation (total_profit second_half_profit first_half_profit : ℚ) :
  total_profit = 3635000 →
  first_half_profit = second_half_profit + 2750000 →
  total_profit = first_half_profit + second_half_profit →
  second_half_profit = 442500 := by
sorry

end profit_calculation_l2251_225121


namespace range_of_a_l2251_225199

open Real

noncomputable def f (a x : ℝ) : ℝ := x - (a + 1) * log x

noncomputable def g (a x : ℝ) : ℝ := a / x - 3

noncomputable def h (a x : ℝ) : ℝ := f a x - g a x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f a x ≥ g a x) →
  a ∈ Set.Iic (exp 1 * (exp 1 + 2) / (exp 1 + 1)) :=
by sorry

end range_of_a_l2251_225199


namespace press_conference_seating_l2251_225143

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def seating_arrangements (team_sizes : List ℕ) : ℕ :=
  (factorial team_sizes.length) * (team_sizes.map factorial).prod

theorem press_conference_seating :
  seating_arrangements [3, 3, 2, 2] = 3456 := by
  sorry

end press_conference_seating_l2251_225143


namespace equation_one_l2251_225165

theorem equation_one (x : ℚ) : 3 * (x + 8) - 5 = 6 * (2 * x - 1) → x = 25 / 9 := by
  sorry

end equation_one_l2251_225165


namespace denis_numbers_sum_l2251_225133

theorem denis_numbers_sum : 
  ∀ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d → 
    a * d = 32 → 
    b * c = 14 → 
    a + b + c + d = 42 := by
  sorry

end denis_numbers_sum_l2251_225133


namespace value_of_x_l2251_225160

theorem value_of_x (x y z : ℚ) 
  (h1 : x = y / 2) 
  (h2 : y = z / 3) 
  (h3 : z = 100) : 
  x = 50 / 3 := by
  sorry

end value_of_x_l2251_225160


namespace factory_sampling_is_systematic_l2251_225196

/-- Represents a sampling method --/
inductive SamplingMethod
  | Stratified
  | SimpleRandom
  | Systematic
  | Other

/-- Represents a factory's production and inspection process --/
structure ProductionProcess where
  inspectionInterval : ℕ  -- Interval between inspections in minutes
  samplePosition : ℕ      -- Fixed position on the conveyor belt for sampling

/-- Determines the sampling method based on the production process --/
def determineSamplingMethod (process : ProductionProcess) : SamplingMethod :=
  if process.inspectionInterval > 0 ∧ process.samplePosition > 0 then
    SamplingMethod.Systematic
  else
    SamplingMethod.Other

/-- Theorem stating that the described process is systematic sampling --/
theorem factory_sampling_is_systematic (process : ProductionProcess) 
  (h1 : process.inspectionInterval = 10)
  (h2 : process.samplePosition > 0) :
  determineSamplingMethod process = SamplingMethod.Systematic := by
  sorry


end factory_sampling_is_systematic_l2251_225196


namespace parabola_c_value_l2251_225110

/-- A parabola with equation y = 2x^2 + bx + c passes through the points (1, 4) and (5, 4). -/
theorem parabola_c_value (b c : ℝ) : 
  (4 = 2 * 1^2 + b * 1 + c) → 
  (4 = 2 * 5^2 + b * 5 + c) → 
  c = 14 := by
  sorry

end parabola_c_value_l2251_225110


namespace kids_left_playing_l2251_225179

theorem kids_left_playing (initial_kids : ℝ) (kids_gone_home : ℝ) 
  (h1 : initial_kids = 22.0) 
  (h2 : kids_gone_home = 14.0) : 
  initial_kids - kids_gone_home = 8.0 := by
sorry

end kids_left_playing_l2251_225179


namespace polynomial_real_root_l2251_225151

theorem polynomial_real_root 
  (P : ℝ → ℝ) 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_nonzero : a₁ * a₂ * a₃ ≠ 0)
  (h_poly : ∀ x : ℝ, P (a₁ * x + b₁) + P (a₂ * x + b₂) = P (a₃ * x + b₃)) :
  ∃ r : ℝ, P r = 0 :=
by sorry

end polynomial_real_root_l2251_225151


namespace cubic_roots_arithmetic_progression_l2251_225137

-- Define the polynomial
def cubic_polynomial (a b : ℝ) (x : ℂ) : ℂ := x^3 - 9*x^2 + b*x + a

-- Define the arithmetic progression property for complex roots
def arithmetic_progression (r₁ r₂ r₃ : ℂ) : Prop :=
  ∃ (d : ℝ), r₂ - r₁ = d ∧ r₃ - r₂ = d

-- State the theorem
theorem cubic_roots_arithmetic_progression (a b : ℝ) :
  (∃ (r₁ r₂ r₃ : ℂ), 
    (∀ x : ℂ, cubic_polynomial a b x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) ∧
    arithmetic_progression r₁ r₂ r₃ ∧
    (∃ i : ℝ, r₂ = i * Complex.I)) →
  (a = 27 + 3 * (Real.sqrt ((a - 27) / 3))^2 ∧ b = -27) :=
by sorry

end cubic_roots_arithmetic_progression_l2251_225137


namespace additional_people_for_lawn_mowing_l2251_225186

/-- The number of additional people needed to mow a lawn in a shorter time -/
theorem additional_people_for_lawn_mowing 
  (initial_people : ℕ) 
  (initial_time : ℕ) 
  (new_time : ℕ) 
  (h1 : initial_people > 0)
  (h2 : initial_time > 0)
  (h3 : new_time > 0)
  (h4 : new_time < initial_time) :
  let total_work := initial_people * initial_time
  let new_people := total_work / new_time
  new_people - initial_people = 10 :=
by sorry

end additional_people_for_lawn_mowing_l2251_225186


namespace mary_clothing_expense_l2251_225123

-- Define the costs of the shirt and jacket
def shirt_cost : Real := 13.04
def jacket_cost : Real := 12.27

-- Define the total cost
def total_cost : Real := shirt_cost + jacket_cost

-- Theorem statement
theorem mary_clothing_expense : total_cost = 25.31 := by
  sorry

end mary_clothing_expense_l2251_225123


namespace initial_games_count_l2251_225130

theorem initial_games_count (sold : ℕ) (added : ℕ) (final : ℕ) : 
  sold = 68 → added = 47 → final = 74 → 
  ∃ initial : ℕ, initial - sold + added = final ∧ initial = 95 := by
sorry

end initial_games_count_l2251_225130


namespace solution_set_1_correct_solution_set_2_correct_l2251_225105

-- Define the solution set for the first inequality
def solution_set_1 : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the solution set for the second inequality based on the value of a
def solution_set_2 (a : ℝ) : Set ℝ :=
  if a = -2 then Set.univ
  else if a > -2 then {x | x ≤ -2 ∨ x ≥ a}
  else {x | x ≤ a ∨ x ≥ -2}

-- Theorem for the first inequality
theorem solution_set_1_correct :
  ∀ x : ℝ, x ∈ solution_set_1 ↔ (2 * x) / (x + 1) < 1 :=
sorry

-- Theorem for the second inequality
theorem solution_set_2_correct :
  ∀ a x : ℝ, x ∈ solution_set_2 a ↔ x^2 + (2 - a) * x - 2 * a ≥ 0 :=
sorry

end solution_set_1_correct_solution_set_2_correct_l2251_225105


namespace square_side_length_l2251_225172

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents an octagon -/
structure Octagon :=
  (A B C D E F G H : Point)

/-- Represents a square -/
structure Square :=
  (W X Y Z : Point)

/-- The octagon ABCDEFGH -/
def octagon : Octagon := sorry

/-- The inscribed square WXYZ -/
def square : Square := sorry

/-- W is on BC -/
axiom W_on_BC : square.W.x ≥ octagon.B.x ∧ square.W.x ≤ octagon.C.x ∧ 
                square.W.y = octagon.B.y ∧ square.W.y = octagon.C.y

/-- X is on DE -/
axiom X_on_DE : square.X.x ≥ octagon.D.x ∧ square.X.x ≤ octagon.E.x ∧ 
                square.X.y = octagon.D.y ∧ square.X.y = octagon.E.y

/-- Y is on FG -/
axiom Y_on_FG : square.Y.x ≥ octagon.F.x ∧ square.Y.x ≤ octagon.G.x ∧ 
                square.Y.y = octagon.F.y ∧ square.Y.y = octagon.G.y

/-- Z is on HA -/
axiom Z_on_HA : square.Z.x ≥ octagon.H.x ∧ square.Z.x ≤ octagon.A.x ∧ 
                square.Z.y = octagon.H.y ∧ square.Z.y = octagon.A.y

/-- AB = 50 -/
axiom AB_length : Real.sqrt ((octagon.A.x - octagon.B.x)^2 + (octagon.A.y - octagon.B.y)^2) = 50

/-- GH = 50(√3 - 1) -/
axiom GH_length : Real.sqrt ((octagon.G.x - octagon.H.x)^2 + (octagon.G.y - octagon.H.y)^2) = 50 * (Real.sqrt 3 - 1)

/-- The side length of square WXYZ is 50 -/
theorem square_side_length : 
  Real.sqrt ((square.W.x - square.Z.x)^2 + (square.W.y - square.Z.y)^2) = 50 := by
  sorry

end square_side_length_l2251_225172


namespace samson_sandwich_count_l2251_225124

/-- The number of sandwiches Samson ate at lunch on Monday -/
def lunch_sandwiches : ℕ := sorry

/-- The number of sandwiches Samson ate at dinner on Monday -/
def dinner_sandwiches : ℕ := 2 * lunch_sandwiches

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_breakfast : ℕ := 1

/-- The total number of sandwiches Samson ate on Monday -/
def monday_total : ℕ := lunch_sandwiches + dinner_sandwiches

/-- The total number of sandwiches Samson ate on Tuesday -/
def tuesday_total : ℕ := tuesday_breakfast

theorem samson_sandwich_count : lunch_sandwiches = 3 := by
  have h1 : monday_total = tuesday_total + 8 := by sorry
  sorry

end samson_sandwich_count_l2251_225124


namespace hyperbola_min_value_l2251_225192

/-- The minimum value of (b² + 1) / a for a hyperbola with eccentricity 2 -/
theorem hyperbola_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2  -- eccentricity
  let c := e * a  -- focal distance
  (c^2 = a^2 + b^2) →  -- hyperbola property
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (b^2 + 1) / a ≥ 2 * Real.sqrt 3) ∧
  (∃ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ∧ (b^2 + 1) / a = 2 * Real.sqrt 3) :=
by sorry

end hyperbola_min_value_l2251_225192


namespace complement_of_P_relative_to_U_l2251_225141

def U : Set ℤ := {-1, 0, 1, 2, 3}
def P : Set ℤ := {-1, 2, 3}

theorem complement_of_P_relative_to_U :
  {x ∈ U | x ∉ P} = {0, 1} := by sorry

end complement_of_P_relative_to_U_l2251_225141


namespace equation_solutions_l2251_225195

def equation (x : ℝ) : Prop :=
  x ≥ 1 ∧ Real.sqrt (x + 5 - 6 * Real.sqrt (x - 1)) + Real.sqrt (x + 10 - 8 * Real.sqrt (x - 1)) = 3

theorem equation_solutions :
  {x : ℝ | equation x} = {5, 26} :=
sorry

end equation_solutions_l2251_225195


namespace nabla_calculation_l2251_225185

-- Define the ∇ operation
def nabla (a b : ℕ) : ℕ :=
  (b * (2 * a + b - 1)) / 2

-- State the theorem
theorem nabla_calculation : nabla 2 (nabla 0 (nabla 1 7)) = 71859 := by
  sorry

end nabla_calculation_l2251_225185


namespace line_relationships_exhaustive_l2251_225113

-- Define the possible spatial relationships between lines
inductive LineRelationship
  | Parallel
  | Intersecting
  | Skew

-- Define a line in 3D space
structure Line3D where
  -- We don't need to specify the exact representation of a line here
  -- as it's not relevant for the statement of the theorem

-- Define the relationship between two lines
def relationshipBetweenLines (l1 l2 : Line3D) : LineRelationship :=
  sorry -- The actual implementation is not needed for the statement

-- Theorem statement
theorem line_relationships_exhaustive (l1 l2 : Line3D) :
  ∃ (r : LineRelationship), relationshipBetweenLines l1 l2 = r :=
sorry

end line_relationships_exhaustive_l2251_225113


namespace system_solution_unique_l2251_225127

theorem system_solution_unique :
  ∃! (x y : ℚ), 3 * x - 2 * y = 5 ∧ x + 4 * y = 4 :=
by
  sorry

end system_solution_unique_l2251_225127


namespace sequences_properties_l2251_225136

/-- Arithmetic sequence with first term 3 and common difference 2 -/
def a (n : ℕ) : ℕ := 3 + 2 * (n - 1)

/-- Geometric sequence with first term 1 and common ratio 2 -/
def b (n : ℕ) : ℕ := 2^(n - 1)

/-- Product sequence of a and b -/
def c (n : ℕ) : ℕ := a n * b n

/-- Sum of first n terms of arithmetic sequence a -/
def S (n : ℕ) : ℕ := n * (a 1 + a n) / 2

/-- Sum of first n terms of product sequence c -/
def T (n : ℕ) : ℕ := (2 * n - 1) * 2^n + 1

theorem sequences_properties :
  (a 1 = 3) ∧
  (b 1 = 1) ∧
  (b 2 + S 2 = 10) ∧
  (a 5 - 2 * b 2 = a 3) ∧
  (∀ n : ℕ, a n = 2 * n + 1) ∧
  (∀ n : ℕ, b n = 2^(n - 1)) ∧
  (∀ n : ℕ, T n = (2 * n - 1) * 2^n + 1) := by
  sorry

#check sequences_properties

end sequences_properties_l2251_225136


namespace x_power_plus_inverse_l2251_225102

theorem x_power_plus_inverse (θ : ℝ) (x : ℂ) (n : ℤ) 
  (h1 : 0 < θ) (h2 : θ < π) (h3 : x + 1/x = 2 * Real.cos θ) :
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end x_power_plus_inverse_l2251_225102


namespace oil_transfer_height_l2251_225158

/-- Given a cone with base radius 9 cm and height 27 cm, when its volume is transferred to a cylinder with base radius 18 cm, the height of the liquid in the cylinder is 2.25 cm. -/
theorem oil_transfer_height :
  let cone_radius : ℝ := 9
  let cone_height : ℝ := 27
  let cylinder_radius : ℝ := 18
  let cone_volume : ℝ := (1/3) * Real.pi * cone_radius^2 * cone_height
  let cylinder_height : ℝ := cone_volume / (Real.pi * cylinder_radius^2)
  cylinder_height = 2.25
  := by sorry

end oil_transfer_height_l2251_225158


namespace book_collection_average_l2251_225152

def arithmeticSequenceSum (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

def arithmeticSequenceAverage (a d n : ℕ) : ℚ :=
  (arithmeticSequenceSum a d n : ℚ) / n

theorem book_collection_average :
  arithmeticSequenceAverage 12 12 7 = 48 := by
  sorry

end book_collection_average_l2251_225152


namespace smallest_number_l2251_225182

theorem smallest_number (S : Finset ℕ) (h : S = {5, 8, 1, 2}) : 
  ∃ m ∈ S, ∀ n ∈ S, m ≤ n ∧ m = 1 :=
by sorry

end smallest_number_l2251_225182


namespace smallest_b_value_l2251_225125

theorem smallest_b_value (a b : ℕ+) 
  (h1 : a.val - b.val = 10)
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ x : ℕ+, 2 ≤ x.val → x.val < b.val → False :=
by sorry

end smallest_b_value_l2251_225125


namespace round_trip_speed_calculation_l2251_225155

/-- Represents a round trip journey between two cities -/
structure RoundTrip where
  initial_speed : ℝ
  initial_time : ℝ
  return_time : ℝ
  average_speed : ℝ
  distance : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem round_trip_speed_calculation (trip : RoundTrip) 
  (h1 : trip.return_time = 2 * trip.initial_time)
  (h2 : trip.average_speed = 34)
  (h3 : trip.distance > 0)
  (h4 : trip.initial_speed > 0) :
  trip.initial_speed = 51 := by
  sorry

end round_trip_speed_calculation_l2251_225155


namespace random_event_identification_l2251_225187

theorem random_event_identification :
  -- Event ①
  (∀ x : ℝ, x^2 + 1 ≠ 0) ∧
  -- Event ②
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x > 1/x ∧ y ≤ 1/y) ∧
  -- Event ③
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → 1/x > 1/y) ∧
  -- Event ④
  (∀ a b : ℝ, a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end random_event_identification_l2251_225187


namespace triangle_side_length_l2251_225161

theorem triangle_side_length (X Y Z : Real) (XY : Real) :
  -- XYZ is a triangle
  X + Y + Z = Real.pi →
  -- cos(2X - Y) + sin(X + Y) = 2
  Real.cos (2 * X - Y) + Real.sin (X + Y) = 2 →
  -- XY = 6
  XY = 6 →
  -- Then YZ = 3√3
  ∃ (YZ : Real), YZ = 3 * Real.sqrt 3 := by
    sorry

end triangle_side_length_l2251_225161


namespace car_travel_time_l2251_225103

theorem car_travel_time (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) 
  (h1 : distance = 270)
  (h2 : new_speed = 30)
  (h3 : time_ratio = 3/2)
  : ∃ (initial_time : ℝ), 
    initial_time = 6 ∧ 
    distance = new_speed * (initial_time * time_ratio) :=
by sorry

end car_travel_time_l2251_225103


namespace difference_between_half_and_sixth_l2251_225129

theorem difference_between_half_and_sixth (x y : ℚ) : x = 1/2 → y = 1/6 → x - y = 1/3 := by
  sorry

end difference_between_half_and_sixth_l2251_225129


namespace shrink_ray_reduction_l2251_225177

/-- The shrink ray problem -/
theorem shrink_ray_reduction (initial_cups : ℕ) (initial_coffee_per_cup : ℝ) (final_total_coffee : ℝ) :
  initial_cups = 5 →
  initial_coffee_per_cup = 8 →
  final_total_coffee = 20 →
  (1 - final_total_coffee / (initial_cups * initial_coffee_per_cup)) * 100 = 50 := by
  sorry

#check shrink_ray_reduction

end shrink_ray_reduction_l2251_225177


namespace lavinia_son_katie_daughter_age_ratio_l2251_225119

/-- Proves that the ratio of Lavinia's son's age to Katie's daughter's age is 2:1 given the specified conditions. -/
theorem lavinia_son_katie_daughter_age_ratio :
  ∀ (katie_daughter_age : ℕ) 
    (lavinia_daughter_age : ℕ) 
    (lavinia_son_age : ℕ),
  katie_daughter_age = 12 →
  lavinia_daughter_age = katie_daughter_age - 10 →
  lavinia_son_age = lavinia_daughter_age + 22 →
  ∃ (k : ℕ), k * katie_daughter_age = lavinia_son_age →
  lavinia_son_age / katie_daughter_age = 2 :=
by sorry

end lavinia_son_katie_daughter_age_ratio_l2251_225119


namespace characterization_of_special_numbers_l2251_225170

/-- Sum of digits of a natural number in base 10 -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The set of numbers that satisfy n = S(n)^2 - S(n) + 1 -/
def specialNumbers : Set ℕ :=
  {n : ℕ | n = (sumOfDigits n)^2 - sumOfDigits n + 1}

/-- Theorem stating that the set of special numbers is exactly {1, 13, 43, 91, 157} -/
theorem characterization_of_special_numbers :
  specialNumbers = {1, 13, 43, 91, 157} := by sorry

end characterization_of_special_numbers_l2251_225170


namespace point_in_second_quadrant_l2251_225148

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- For any real number m, the point (-1, m^2 + 1) is in the second quadrant -/
theorem point_in_second_quadrant (m : ℝ) : in_second_quadrant (-1) (m^2 + 1) := by
  sorry

end point_in_second_quadrant_l2251_225148


namespace running_time_proof_l2251_225168

/-- Proves that the time taken for Joe and Pete to be 16 km apart is 80 minutes -/
theorem running_time_proof (joe_speed : ℝ) (pete_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  joe_speed = 0.133333333333 →
  pete_speed = joe_speed / 2 →
  distance = 16 →
  time * (joe_speed + pete_speed) = distance →
  time = 80 := by
sorry

end running_time_proof_l2251_225168


namespace arithmetic_sequence_ratio_l2251_225188

def arithmeticSequenceSum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_ratio :
  let numerator := arithmeticSequenceSum 4 4 52
  let denominator := arithmeticSequenceSum 6 6 78
  numerator / denominator = 2 / 3 := by sorry

end arithmetic_sequence_ratio_l2251_225188


namespace triangle_not_acute_l2251_225131

theorem triangle_not_acute (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 30) (h3 : B = 50) :
  ¬ (A < 90 ∧ B < 90 ∧ C < 90) :=
sorry

end triangle_not_acute_l2251_225131


namespace statement_2_statement_3_l2251_225171

-- Define the types for lines and planes
variable {Line Plane : Type}

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (planeParallel : Plane → Plane → Prop)
variable (planePerpendicular : Plane → Plane → Prop)
variable (lineParallelPlane : Line → Plane → Prop)
variable (linePerpendicularPlane : Line → Plane → Prop)

-- Statement 2
theorem statement_2 (α β : Plane) (m : Line) :
  planePerpendicular α β → lineParallelPlane m α → linePerpendicularPlane m β := by
  sorry

-- Statement 3
theorem statement_3 (α β : Plane) (m : Line) :
  linePerpendicularPlane m β → planeParallel β α → planePerpendicular α β := by
  sorry

end statement_2_statement_3_l2251_225171


namespace rocky_miles_total_l2251_225114

/-- Calculates the total miles run by Rocky in the first three days of training -/
def rocky_miles : ℕ :=
  let day1 : ℕ := 4
  let day2 : ℕ := 2 * day1
  let day3 : ℕ := 3 * day2
  day1 + day2 + day3

theorem rocky_miles_total : rocky_miles = 36 := by
  sorry

end rocky_miles_total_l2251_225114


namespace consecutive_even_numbers_sum_l2251_225101

theorem consecutive_even_numbers_sum (a b c : ℤ) : 
  (∃ k : ℤ, b = 2 * k) →  -- b is even
  (a = b - 2) →           -- a is the previous even number
  (c = b + 2) →           -- c is the next even number
  (a + b = 18) →          -- sum of first and second
  (a + c = 22) →          -- sum of first and third
  (b + c = 28) →          -- sum of second and third
  b = 11 :=               -- middle number is 11
by sorry

end consecutive_even_numbers_sum_l2251_225101


namespace tables_left_l2251_225176

theorem tables_left (original_tables : ℝ) (customers_per_table : ℝ) (current_customers : ℕ) :
  original_tables = 44.0 →
  customers_per_table = 8.0 →
  current_customers = 256 →
  original_tables - (current_customers : ℝ) / customers_per_table = 12.0 := by
  sorry

end tables_left_l2251_225176


namespace fathers_age_is_38_l2251_225183

/-- The age of the son 5 years ago -/
def sons_age_5_years_ago : ℕ := 14

/-- The current age of the son -/
def sons_current_age : ℕ := sons_age_5_years_ago + 5

/-- The age of the father when the son was born -/
def fathers_age_at_sons_birth : ℕ := sons_current_age

/-- The current age of the father -/
def fathers_current_age : ℕ := fathers_age_at_sons_birth + sons_current_age

theorem fathers_age_is_38 : fathers_current_age = 38 := by
  sorry

end fathers_age_is_38_l2251_225183


namespace total_results_l2251_225189

theorem total_results (total_sum : ℕ) (total_count : ℕ) 
  (first_six_sum : ℕ) (last_six_sum : ℕ) (sixth_result : ℕ) :
  total_sum / total_count = 60 →
  first_six_sum = 6 * 58 →
  last_six_sum = 6 * 63 →
  sixth_result = 66 →
  total_sum = first_six_sum + last_six_sum - sixth_result →
  total_count = 11 := by
sorry

end total_results_l2251_225189


namespace penny_frog_count_l2251_225191

/-- The number of tree frogs Penny counted -/
def tree_frogs : ℕ := 55

/-- The number of poison frogs Penny counted -/
def poison_frogs : ℕ := 10

/-- The number of wood frogs Penny counted -/
def wood_frogs : ℕ := 13

/-- The total number of frogs Penny counted -/
def total_frogs : ℕ := tree_frogs + poison_frogs + wood_frogs

theorem penny_frog_count : total_frogs = 78 := by
  sorry

end penny_frog_count_l2251_225191


namespace symmetric_point_origin_symmetric_point_negative_two_five_l2251_225157

def symmetric_point (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem symmetric_point_origin (x y : ℝ) : 
  symmetric_point x y = (-x, -y) := by sorry

theorem symmetric_point_negative_two_five : 
  symmetric_point (-2) 5 = (2, -5) := by sorry

end symmetric_point_origin_symmetric_point_negative_two_five_l2251_225157


namespace complex_power_multiply_l2251_225154

theorem complex_power_multiply (i : ℂ) : i^2 = -1 → i^13 * (1 + i) = -1 + i := by
  sorry

end complex_power_multiply_l2251_225154


namespace min_avg_of_two_l2251_225122

theorem min_avg_of_two (a b c d : ℕ+) : 
  (a + b + c + d : ℝ) / 4 = 50 → 
  max c d ≤ 130 →
  (a + b : ℝ) / 2 ≥ 35 :=
by sorry

end min_avg_of_two_l2251_225122


namespace star_properties_l2251_225144

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - 2

theorem star_properties :
  (∀ x y : ℝ, star x y = star y x) ∧
  (∃ x y z : ℝ, star x (y + z) ≠ star x y + star x z) ∧
  (∃ x : ℝ, star (x - 2) (x + 2) ≠ star x x - 2) ∧
  (¬ ∃ e : ℝ, ∀ x : ℝ, star x e = x ∧ star e x = x) ∧
  (∃ x y z : ℝ, star (star x y) z ≠ star x (star y z)) := by
  sorry

end star_properties_l2251_225144


namespace pastry_difference_l2251_225190

/-- Represents the number of pastries each person has -/
structure Pastries where
  frank : ℕ
  calvin : ℕ
  phoebe : ℕ
  grace : ℕ

/-- The conditions of the pastry problem -/
def PastryProblem (p : Pastries) : Prop :=
  p.calvin = p.frank + 8 ∧
  p.phoebe = p.frank + 8 ∧
  p.grace = 30 ∧
  p.frank + p.calvin + p.phoebe + p.grace = 97

theorem pastry_difference (p : Pastries) (h : PastryProblem p) :
  p.grace - p.calvin = 5 ∧ p.grace - p.phoebe = 5 :=
sorry

end pastry_difference_l2251_225190


namespace summer_work_hours_adjustment_l2251_225120

theorem summer_work_hours_adjustment 
  (initial_weeks : ℕ) 
  (initial_hours_per_week : ℝ) 
  (unavailable_weeks : ℕ) 
  (adjusted_hours_per_week : ℝ) :
  initial_weeks > unavailable_weeks →
  initial_weeks * initial_hours_per_week = 
    (initial_weeks - unavailable_weeks) * adjusted_hours_per_week →
  adjusted_hours_per_week = initial_hours_per_week * (initial_weeks / (initial_weeks - unavailable_weeks)) :=
by
  sorry

#eval (31.25 : Float)

end summer_work_hours_adjustment_l2251_225120


namespace inequalities_not_necessarily_true_l2251_225169

theorem inequalities_not_necessarily_true
  (x y a b : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hxa : x < a)
  (hyb : y ≠ b) :
  ∃ (x' y' a' b' : ℝ),
    x' ≠ 0 ∧ y' ≠ 0 ∧ a' ≠ 0 ∧ b' ≠ 0 ∧
    x' < a' ∧ y' ≠ b' ∧
    ¬(x' + y' < a' + b') ∧
    ¬(x' - y' < a' - b') ∧
    ¬(x' * y' < a' * b') ∧
    ¬(x' / y' < a' / b') :=
by sorry

end inequalities_not_necessarily_true_l2251_225169


namespace max_points_at_distance_l2251_225135

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in 2D space
def Point : Type := ℝ × ℝ

-- Function to check if a point is outside a circle
def isOutside (p : Point) (c : Circle) : Prop :=
  let (px, py) := p
  let (cx, cy) := c.center
  (px - cx)^2 + (py - cy)^2 > c.radius^2

-- Function to count points on circle at fixed distance from a point
def countPointsAtDistance (c : Circle) (p : Point) (d : ℝ) : ℕ :=
  sorry

-- Theorem statement
theorem max_points_at_distance (c : Circle) (p : Point) (d : ℝ) 
  (h : isOutside p c) : 
  countPointsAtDistance c p d ≤ 2 :=
sorry

end max_points_at_distance_l2251_225135


namespace inequality_chain_l2251_225174

theorem inequality_chain (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) :
  a > -b ∧ -b > b ∧ b > -a := by sorry

end inequality_chain_l2251_225174


namespace tangent_point_x_coordinate_l2251_225198

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_point_x_coordinate (x : ℝ) (h : x > 0) : 
  (deriv f x = 2) → x = Real.exp 1 := by
  sorry

end tangent_point_x_coordinate_l2251_225198


namespace rectangle_perimeter_l2251_225100

/-- For a rectangle with sum of length and width equal to 28 meters, the perimeter is 56 meters. -/
theorem rectangle_perimeter (l w : ℝ) (h : l + w = 28) : 2 * (l + w) = 56 := by
  sorry

end rectangle_perimeter_l2251_225100


namespace domino_set_size_l2251_225181

theorem domino_set_size (num_players : ℕ) (dominoes_per_player : ℕ) 
  (h1 : num_players = 4) 
  (h2 : dominoes_per_player = 7) : 
  num_players * dominoes_per_player = 28 := by
  sorry

end domino_set_size_l2251_225181


namespace train_speed_calculation_l2251_225109

-- Define the length of the train in meters
def train_length : ℝ := 83.33333333333334

-- Define the time taken to cross the pole in seconds
def crossing_time : ℝ := 5

-- Define the speed of the train in km/hr
def train_speed : ℝ := 60

-- Theorem to prove
theorem train_speed_calculation :
  train_speed = (train_length / 1000) / (crossing_time / 3600) :=
by sorry

end train_speed_calculation_l2251_225109


namespace glenda_skating_speed_l2251_225134

/-- Prove Glenda's skating speed given the conditions of the problem -/
theorem glenda_skating_speed 
  (ann_speed : ℝ) 
  (time : ℝ) 
  (total_distance : ℝ) 
  (h1 : ann_speed = 6)
  (h2 : time = 3)
  (h3 : total_distance = 42) :
  ∃ (glenda_speed : ℝ), 
    glenda_speed = 8 ∧ 
    ann_speed * time + glenda_speed * time = total_distance :=
by sorry

end glenda_skating_speed_l2251_225134


namespace fraction_equation_solution_l2251_225118

theorem fraction_equation_solution : 
  ∃ x : ℚ, (3/4 * 60 - x * 60 + 63 = 12) ∧ (x = 8/5) := by
  sorry

end fraction_equation_solution_l2251_225118


namespace rod_cutting_l2251_225146

/-- Given a rod of length 42.5 meters that can be cut into 50 equal pieces,
    prove that the length of each piece is 0.85 meters. -/
theorem rod_cutting (rod_length : Real) (num_pieces : Nat) (piece_length : Real) 
    (h1 : rod_length = 42.5)
    (h2 : num_pieces = 50)
    (h3 : piece_length * num_pieces = rod_length) : 
  piece_length = 0.85 := by
  sorry

end rod_cutting_l2251_225146


namespace rogers_final_balance_theorem_l2251_225153

/-- Calculates Roger's final balance in US dollars after all transactions -/
def rogers_final_balance (initial_balance : ℝ) (video_game_percentage : ℝ) 
  (euros_spent : ℝ) (euro_to_dollar : ℝ) (canadian_dollars_received : ℝ) 
  (canadian_to_dollar : ℝ) : ℝ :=
  let remaining_after_game := initial_balance * (1 - video_game_percentage)
  let remaining_after_euros := remaining_after_game - euros_spent * euro_to_dollar
  remaining_after_euros + canadian_dollars_received * canadian_to_dollar

/-- Theorem stating Roger's final balance after all transactions -/
theorem rogers_final_balance_theorem : 
  rogers_final_balance 45 0.35 20 1.2 46 0.8 = 42.05 := by
  sorry

end rogers_final_balance_theorem_l2251_225153


namespace p_and_s_not_third_l2251_225180

-- Define the set of runners
inductive Runner : Type
  | P | Q | R | S | T | U

-- Define the finishing order relation
def finishes_before (x y : Runner) : Prop := sorry

-- Define the race conditions
axiom p_beats_q : finishes_before Runner.P Runner.Q
axiom p_beats_r : finishes_before Runner.P Runner.R
axiom q_beats_s : finishes_before Runner.Q Runner.S
axiom t_between_p_and_q : finishes_before Runner.P Runner.T ∧ finishes_before Runner.T Runner.Q
axiom u_after_r_before_t : finishes_before Runner.R Runner.U ∧ finishes_before Runner.U Runner.T

-- Define what it means to finish third
def finishes_third (x : Runner) : Prop :=
  ∃ (a b : Runner), (a ≠ x ∧ b ≠ x ∧ a ≠ b) ∧
    finishes_before a x ∧ finishes_before b x ∧
    ∀ y : Runner, y ≠ x → y ≠ a → y ≠ b → finishes_before x y

-- Theorem to prove
theorem p_and_s_not_third :
  ¬(finishes_third Runner.P) ∧ ¬(finishes_third Runner.S) :=
sorry

end p_and_s_not_third_l2251_225180


namespace normal_distribution_probability_l2251_225197

/-- A normally distributed random variable -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  mean_pos : 0 < mean
  std_dev_pos : 0 < std_dev

/-- The probability that a normal random variable falls within an interval -/
noncomputable def prob_in_interval (X : NormalDistribution) (lower upper : ℝ) : ℝ := sorry

/-- Theorem: If P(0 < X < a) = 0.3 for X ~ N(a, d²), then P(0 < X < 2a) = 0.6 -/
theorem normal_distribution_probability 
  (X : NormalDistribution) 
  (h : prob_in_interval X 0 X.mean = 0.3) : 
  prob_in_interval X 0 (2 * X.mean) = 0.6 := by sorry

end normal_distribution_probability_l2251_225197


namespace cube_expansion_value_l2251_225140

theorem cube_expansion_value (y : ℝ) (h : y = 50) : 
  y^3 + 3*y^2*(2*y) + 3*y*(2*y)^2 + (2*y)^3 = 3375000 := by
sorry

end cube_expansion_value_l2251_225140


namespace continued_fraction_theorem_l2251_225173

-- Define the continued fraction for part 1
def continued_fraction_1 : ℚ :=
  1 + 1 / (2 + 1 / (3 + 1 / 4))

-- Define the continued fraction for part 2
def continued_fraction_2 (a b c : ℕ) : ℚ :=
  a + 1 / (b + 1 / c)

-- Define the equation for part 3
def continued_fraction_equation (y : ℝ) : Prop :=
  y = 8 + 1 / y

theorem continued_fraction_theorem :
  (continued_fraction_1 = 43 / 30) ∧
  (355 / 113 = continued_fraction_2 3 7 16) ∧
  (∃ y : ℝ, continued_fraction_equation y ∧ y = 4 + Real.sqrt 17) :=
by sorry

end continued_fraction_theorem_l2251_225173


namespace reciprocal_opposite_square_minus_product_l2251_225149

theorem reciprocal_opposite_square_minus_product (a b c d : ℝ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) : 
  (c + d)^2 - a * b = -1 := by
sorry

end reciprocal_opposite_square_minus_product_l2251_225149


namespace polynomial_Q_l2251_225132

/-- Given a polynomial Q(x) = Q(0) + Q(1)x + Q(3)x³ where Q(-1) = 2,
    prove that Q(x) = -2x + (2/9)x³ - 2/9 -/
theorem polynomial_Q (Q : ℝ → ℝ) : 
  (∀ x, Q x = Q 0 + Q 1 * x + Q 3 * x^3) → 
  Q (-1) = 2 → 
  ∀ x, Q x = -2 * x + (2/9) * x^3 - 2/9 := by
sorry

end polynomial_Q_l2251_225132


namespace legoland_kangaroos_l2251_225184

theorem legoland_kangaroos (koalas kangaroos : ℕ) : 
  kangaroos = 5 * koalas →
  koalas + kangaroos = 216 →
  kangaroos = 180 := by
sorry

end legoland_kangaroos_l2251_225184


namespace color_film_fraction_l2251_225111

theorem color_film_fraction (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw := 40 * x
  let total_color := 4 * y
  let selected_bw := (y / x) * (40 * x) / 100
  let selected_color := total_color
  (selected_color) / (selected_bw + selected_color) = 10 / 11 := by
sorry

end color_film_fraction_l2251_225111


namespace triangle_problem_l2251_225162

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < π) (h3 : 0 < B ∧ B < π) (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π) (h6 : b * (Real.cos C + Real.sin C) = a)
  (h7 : a * Real.sin B = b * Real.sin A) (h8 : b * Real.sin C = c * Real.sin B)
  (h9 : a * Real.sin C = c * Real.sin A) (h10 : a * (1/4) = a * Real.sin A * Real.sin C) :
  B = π/4 ∧ Real.cos A = -Real.sqrt 5 / 5 := by
  sorry

end triangle_problem_l2251_225162


namespace polynomial_expansion_problem_l2251_225164

theorem polynomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → 
  10 * p^9 * q = 45 * p^8 * q^2 →
  p + 2*q = 1 →
  p = 9/13 := by sorry

end polynomial_expansion_problem_l2251_225164


namespace barbara_butcher_cost_l2251_225117

/-- The cost of Barbara's purchase at the butcher's --/
def butcher_cost (steak_weight : ℝ) (steak_price : ℝ) (chicken_weight : ℝ) (chicken_price : ℝ) : ℝ :=
  steak_weight * steak_price + chicken_weight * chicken_price

/-- Theorem: Barbara's total cost at the butcher's is $79.50 --/
theorem barbara_butcher_cost :
  butcher_cost 4.5 15 1.5 8 = 79.5 := by
  sorry

end barbara_butcher_cost_l2251_225117


namespace weight_of_one_bag_is_five_l2251_225178

-- Define the given values
def total_harvest : ℕ := 405
def juice_amount : ℕ := 90
def restaurant_amount : ℕ := 60
def total_revenue : ℕ := 408
def price_per_bag : ℕ := 8

-- Define the weight of one bag as a function of the given values
def weight_of_one_bag : ℚ :=
  (total_harvest - juice_amount - restaurant_amount) / (total_revenue / price_per_bag)

-- Theorem to prove
theorem weight_of_one_bag_is_five :
  weight_of_one_bag = 5 := by
  sorry

end weight_of_one_bag_is_five_l2251_225178
