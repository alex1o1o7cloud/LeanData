import Mathlib

namespace janet_sculpture_weight_l52_5225

/-- Calculates the weight of the second sculpture given Janet's work details --/
theorem janet_sculpture_weight
  (exterminator_rate : ℝ)
  (sculpture_rate : ℝ)
  (exterminator_hours : ℝ)
  (first_sculpture_weight : ℝ)
  (total_income : ℝ)
  (h1 : exterminator_rate = 70)
  (h2 : sculpture_rate = 20)
  (h3 : exterminator_hours = 20)
  (h4 : first_sculpture_weight = 5)
  (h5 : total_income = 1640)
  : ∃ (second_sculpture_weight : ℝ),
    second_sculpture_weight = 7 ∧
    total_income = exterminator_rate * exterminator_hours +
                   sculpture_rate * (first_sculpture_weight + second_sculpture_weight) :=
by
  sorry

end janet_sculpture_weight_l52_5225


namespace problem_solution_l52_5239

theorem problem_solution (x : ℝ) : 
  x + Real.sqrt (x^2 - 1) + 1 / (x - Real.sqrt (x^2 - 1)) = 15 →
  x^3 + Real.sqrt (x^6 - 1) + 1 / (x^3 + Real.sqrt (x^6 - 1)) = 3970049 / 36000 :=
by sorry

end problem_solution_l52_5239


namespace quadratic_solution_set_implies_linear_and_inverse_quadratic_l52_5221

/-- Given a quadratic function f(x) = ax² + bx + c, where a, b, and c are real numbers and a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) := λ x : ℝ => a * x^2 + b * x + c

theorem quadratic_solution_set_implies_linear_and_inverse_quadratic
  (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x, QuadraticFunction a b c x > 0 ↔ x < -2 ∨ x > 3) →
  (∀ x, b * x - c > 0 ↔ x < 6) ∧
  (∀ x, c * x^2 - b * x + a ≥ 0 ↔ -1/3 ≤ x ∧ x ≤ 1/2) :=
by sorry

end quadratic_solution_set_implies_linear_and_inverse_quadratic_l52_5221


namespace positive_rational_cube_sum_representation_l52_5242

theorem positive_rational_cube_sum_representation (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end positive_rational_cube_sum_representation_l52_5242


namespace quadratic_reciprocal_roots_l52_5277

theorem quadratic_reciprocal_roots (a b c : ℤ) (x₁ x₂ : ℚ) : 
  a ≠ 0 →
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ * x₂ = 1) →
  (x₁ + x₂ = 4) →
  (∃ n m : ℤ, x₁ = n ∧ x₂ = m) →
  (c = a ∧ b = -4*a) :=
sorry

end quadratic_reciprocal_roots_l52_5277


namespace y1_value_l52_5228

theorem y1_value (y1 y2 y3 : ℝ) 
  (h1 : 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1)
  (h2 : (1 - y1)^2 + 2*(y1 - y2)^2 + 2*(y2 - y3)^2 + y3^2 = 1/2) :
  y1 = (2*Real.sqrt 2 - 1) / (2*Real.sqrt 2) := by
sorry

end y1_value_l52_5228


namespace smallest_distance_between_complex_points_l52_5256

open Complex

theorem smallest_distance_between_complex_points (z w : ℂ) 
  (hz : abs (z + 2 + 4*I) = 2)
  (hw : abs (w - 6 - 7*I) = 4) :
  ∃ (d : ℝ), d = Real.sqrt 185 - 6 ∧ ∀ (z' w' : ℂ), 
    abs (z' + 2 + 4*I) = 2 → abs (w' - 6 - 7*I) = 4 → 
    abs (z' - w') ≥ d ∧ ∃ (z'' w'' : ℂ), abs (z'' - w'') = d :=
by sorry

end smallest_distance_between_complex_points_l52_5256


namespace symmetric_point_l52_5287

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the line x + y = 0 -/
def symmetryLine (p : Point) : Prop :=
  p.x + p.y = 0

/-- Defines the property of two points being symmetric with respect to the line x + y = 0 -/
def isSymmetric (p1 p2 : Point) : Prop :=
  symmetryLine ⟨(p1.x + p2.x) / 2, (p1.y + p2.y) / 2⟩

/-- Theorem: The point symmetric to P(2, 5) with respect to the line x + y = 0 has coordinates (-5, -2) -/
theorem symmetric_point : 
  let p1 : Point := ⟨2, 5⟩
  let p2 : Point := ⟨-5, -2⟩
  isSymmetric p1 p2 := by sorry

end symmetric_point_l52_5287


namespace clinton_school_earnings_l52_5254

/-- Represents the total compensation for all students -/
def total_compensation : ℝ := 1456

/-- Represents the number of students from Arlington school -/
def arlington_students : ℕ := 8

/-- Represents the number of days Arlington students worked -/
def arlington_days : ℕ := 4

/-- Represents the number of students from Bradford school -/
def bradford_students : ℕ := 6

/-- Represents the number of days Bradford students worked -/
def bradford_days : ℕ := 7

/-- Represents the number of students from Clinton school -/
def clinton_students : ℕ := 7

/-- Represents the number of days Clinton students worked -/
def clinton_days : ℕ := 8

/-- Theorem stating that the total earnings for Clinton school students is 627.20 dollars -/
theorem clinton_school_earnings :
  let total_student_days := arlington_students * arlington_days + bradford_students * bradford_days + clinton_students * clinton_days
  let daily_wage := total_compensation / total_student_days
  clinton_students * clinton_days * daily_wage = 627.2 := by
  sorry

end clinton_school_earnings_l52_5254


namespace floor_with_133_black_tiles_has_4489_total_tiles_l52_5288

/-- Represents a square floor covered with tiles -/
structure TiledFloor where
  side : ℕ
  black_tiles : ℕ

/-- The number of black tiles on the diagonals of a square floor -/
def diagonal_tiles (floor : TiledFloor) : ℕ :=
  2 * floor.side - 1

/-- The total number of tiles on a square floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.side ^ 2

/-- Theorem stating that a square floor with 133 black tiles on its diagonals has 4489 total tiles -/
theorem floor_with_133_black_tiles_has_4489_total_tiles (floor : TiledFloor) 
    (h : floor.black_tiles = 133) : total_tiles floor = 4489 := by
  sorry


end floor_with_133_black_tiles_has_4489_total_tiles_l52_5288


namespace fresh_grapes_weight_calculation_l52_5245

/-- The weight of dried grapes in kilograms -/
def dried_grapes_weight : ℝ := 66.67

/-- The fraction of water in fresh grapes by weight -/
def fresh_water_fraction : ℝ := 0.75

/-- The fraction of water in dried grapes by weight -/
def dried_water_fraction : ℝ := 0.25

/-- The weight of fresh grapes in kilograms -/
def fresh_grapes_weight : ℝ := 200.01

theorem fresh_grapes_weight_calculation :
  fresh_grapes_weight = dried_grapes_weight * (1 - dried_water_fraction) / (1 - fresh_water_fraction) :=
by sorry

end fresh_grapes_weight_calculation_l52_5245


namespace problem_solution_l52_5271

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 4 * x

theorem problem_solution :
  -- Part 1
  (∀ x : ℝ, f 2 x ≥ 2 * x + 1 ↔ x ∈ Set.Ici (-1)) ∧
  -- Part 2
  (∀ a : ℝ, a > 0 →
    (∀ x : ℝ, x ∈ Set.Ioi (-2) → f a (2 * x) > 7 * x + a^2 - 3) →
    a ∈ Set.Ioo 0 2) :=
by sorry

end problem_solution_l52_5271


namespace quadratic_form_sum_l52_5253

theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 4 * x^2 - 8 * x - 3 = a * (x - h)^2 + k) →
  a + h + k = -2 := by sorry

end quadratic_form_sum_l52_5253


namespace geometric_sequence_first_term_l52_5236

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_first_term
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_eq : a 3 * a 9 = 2 * (a 5)^2)
  (h_a2 : a 2 = 1) :
  a 1 = Real.sqrt 2 / 2 :=
sorry

end geometric_sequence_first_term_l52_5236


namespace train_speed_theorem_l52_5268

/-- The speed of the first train in km/h -/
def first_train_speed : ℝ := 70

/-- The speed of the second train in km/h -/
def second_train_speed : ℝ := 80

/-- The time difference between the starts of the two trains in hours -/
def time_difference : ℝ := 1

/-- The total travel time of the first train in hours -/
def first_train_travel_time : ℝ := 8

/-- The total travel time of the second train in hours -/
def second_train_travel_time : ℝ := 7

theorem train_speed_theorem : 
  first_train_speed * first_train_travel_time = 
  second_train_speed * second_train_travel_time :=
by sorry

end train_speed_theorem_l52_5268


namespace solution_for_a_l52_5205

theorem solution_for_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (eq1 : a + 2 / b = 17) (eq2 : b + 2 / a = 1 / 3) :
  a = 6 ∨ a = 17 := by
  sorry

end solution_for_a_l52_5205


namespace carnation_bouquet_problem_l52_5248

/-- Given 3 bouquets of carnations with known quantities in the first and third bouquets,
    and a known average, prove the quantity in the second bouquet. -/
theorem carnation_bouquet_problem (b1 b3 avg : ℕ) (h1 : b1 = 9) (h3 : b3 = 13) (havg : avg = 12) :
  ∃ b2 : ℕ, (b1 + b2 + b3) / 3 = avg ∧ b2 = 14 :=
by sorry

end carnation_bouquet_problem_l52_5248


namespace motorcycle_trip_time_difference_specific_motorcycle_problem_l52_5241

/-- Given a motorcycle traveling at a constant speed, prove the time difference between two trips -/
theorem motorcycle_trip_time_difference (speed : ℝ) (distance1 : ℝ) (distance2 : ℝ) : 
  speed > 0 → 
  distance1 > 0 →
  distance2 > 0 →
  distance1 > distance2 →
  (distance1 / speed - distance2 / speed) * 60 = (distance1 - distance2) / speed * 60 := by
  sorry

/-- Specific instance of the theorem for the given problem -/
theorem specific_motorcycle_problem : 
  (400 / 40 - 360 / 40) * 60 = 60 := by
  sorry

end motorcycle_trip_time_difference_specific_motorcycle_problem_l52_5241


namespace base_eight_representation_l52_5263

theorem base_eight_representation (a b c d e f : ℕ) 
  (h1 : 208208 = 8^5 * a + 8^4 * b + 8^3 * c + 8^2 * d + 8 * e + f)
  (h2 : a ≤ 7 ∧ b ≤ 7 ∧ c ≤ 7 ∧ d ≤ 7 ∧ e ≤ 7 ∧ f ≤ 7) :
  a * b * c + d * e * f = 72 := by
  sorry

end base_eight_representation_l52_5263


namespace two_digit_numbers_product_5681_sum_154_l52_5210

theorem two_digit_numbers_product_5681_sum_154 : 
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 5681 ∧ a + b = 154 := by
  sorry

end two_digit_numbers_product_5681_sum_154_l52_5210


namespace projection_onto_xoy_plane_l52_5233

/-- Given a space orthogonal coordinate system Oxyz, prove that the projection
    of point P(1, 2, 3) onto the xOy plane has coordinates (1, 2, 0). -/
theorem projection_onto_xoy_plane :
  let P : ℝ × ℝ × ℝ := (1, 2, 3)
  let xoy_plane : Set (ℝ × ℝ × ℝ) := {v | v.2.2 = 0}
  let projection (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, p.2.1, 0)
  projection P ∈ xoy_plane ∧ projection P = (1, 2, 0) := by
  sorry

end projection_onto_xoy_plane_l52_5233


namespace four_propositions_l52_5215

theorem four_propositions :
  (∀ a b : ℝ, |a + b| - 2 * |a| ≤ |a - b|) ∧
  (∀ a b : ℝ, |a - b| < 1 → |a| < |b| + 1) ∧
  (∀ x y : ℝ, |x| < 2 ∧ |y| > 3 → |x / y| < 2/3) ∧
  (∀ A B : ℝ, A > 0 ∧ B > 0 → Real.log ((|A| + |B|) / 2) ≥ (Real.log |A| + Real.log |B|) / 2) :=
by sorry

end four_propositions_l52_5215


namespace smallest_n_for_less_than_one_percent_probability_l52_5280

def double_factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | n+2 => (n+2) * double_factorial n

def townspeople_win_probability (n : ℕ) : ℚ :=
  (n.factorial : ℚ) / (double_factorial (2*n+1) : ℚ)

theorem smallest_n_for_less_than_one_percent_probability :
  ∀ k : ℕ, k < 6 → townspeople_win_probability k ≥ 1/100 ∧
  townspeople_win_probability 6 < 1/100 :=
by sorry

end smallest_n_for_less_than_one_percent_probability_l52_5280


namespace trigonometric_simplification_l52_5293

theorem trigonometric_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by
sorry

end trigonometric_simplification_l52_5293


namespace double_age_in_four_years_l52_5201

/-- The number of years until Fouad's age is double Ahmed's age -/
def years_until_double_age (ahmed_age : ℕ) (fouad_age : ℕ) : ℕ :=
  fouad_age - ahmed_age

theorem double_age_in_four_years (ahmed_age : ℕ) (fouad_age : ℕ) 
  (h1 : ahmed_age = 11) (h2 : fouad_age = 26) : 
  years_until_double_age ahmed_age fouad_age = 4 := by
  sorry

#check double_age_in_four_years

end double_age_in_four_years_l52_5201


namespace max_sum_exp_l52_5269

theorem max_sum_exp (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 ≤ 4) :
  ∃ (M : ℝ), M = 4 * Real.exp 1 ∧ ∀ (x y z w : ℝ), x^2 + y^2 + z^2 + w^2 ≤ 4 →
    Real.exp x + Real.exp y + Real.exp z + Real.exp w ≤ M :=
by sorry

end max_sum_exp_l52_5269


namespace second_investment_rate_l52_5295

/-- Calculates the interest rate of the second investment given the total investment,
    the amount and rate of the first investment, and the total amount after interest. -/
theorem second_investment_rate (total_investment : ℝ) (first_investment : ℝ) 
  (first_rate : ℝ) (total_after_interest : ℝ) :
  total_investment = 1000 →
  first_investment = 200 →
  first_rate = 0.03 →
  total_after_interest = 1046 →
  (total_after_interest - total_investment - (first_investment * first_rate)) / 
  (total_investment - first_investment) = 0.05 := by
  sorry

end second_investment_rate_l52_5295


namespace original_number_proof_l52_5262

theorem original_number_proof :
  ∃ x : ℝ, x * 1.1 = 550 ∧ x = 500 := by
  sorry

end original_number_proof_l52_5262


namespace joan_football_games_l52_5244

/-- The number of football games Joan attended this year -/
def games_this_year : ℕ := 4

/-- The number of football games Joan attended last year -/
def games_last_year : ℕ := 9

/-- The total number of football games Joan attended -/
def total_games : ℕ := games_this_year + games_last_year

theorem joan_football_games :
  total_games = 13 := by sorry

end joan_football_games_l52_5244


namespace percentage_difference_l52_5264

theorem percentage_difference (A B y : ℝ) : 
  A > 0 → B > A → B = A * (1 + y / 100) → y = 100 * (B - A) / A := by
  sorry

end percentage_difference_l52_5264


namespace yellow_pairs_count_l52_5234

theorem yellow_pairs_count (blue_count : ℕ) (yellow_count : ℕ) (total_count : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_count = 63 →
  yellow_count = 69 →
  total_count = blue_count + yellow_count →
  total_pairs = 66 →
  blue_blue_pairs = 27 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 30 ∧ 
    yellow_yellow_pairs = (yellow_count - (total_pairs - blue_blue_pairs - (yellow_count - blue_count) / 2)) / 2 :=
by sorry

end yellow_pairs_count_l52_5234


namespace smallest_solution_abs_equation_l52_5258

theorem smallest_solution_abs_equation :
  ∃ x : ℝ, x * |x| = 3 * x - 2 ∧
  ∀ y : ℝ, y * |y| = 3 * y - 2 → x ≤ y :=
by
  sorry

end smallest_solution_abs_equation_l52_5258


namespace product_of_fractions_l52_5229

theorem product_of_fractions : 
  (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) = 1/5 := by
  sorry

end product_of_fractions_l52_5229


namespace solve_exponential_equation_l52_5259

theorem solve_exponential_equation (y : ℝ) : 5^(2*y) = Real.sqrt 125 → y = 3/4 := by
  sorry

end solve_exponential_equation_l52_5259


namespace peters_correct_percentage_l52_5238

theorem peters_correct_percentage (y : ℕ) :
  let total_questions : ℕ := 7 * y
  let missed_questions : ℕ := 2 * y
  let correct_questions : ℕ := total_questions - missed_questions
  (correct_questions : ℚ) / (total_questions : ℚ) * 100 = 500 / 7 :=
by sorry

end peters_correct_percentage_l52_5238


namespace total_tissues_l52_5255

def group1 : ℕ := 15
def group2 : ℕ := 20
def group3 : ℕ := 18
def group4 : ℕ := 22
def group5 : ℕ := 25
def tissues_per_box : ℕ := 70

theorem total_tissues : 
  (group1 + group2 + group3 + group4 + group5) * tissues_per_box = 7000 := by
  sorry

end total_tissues_l52_5255


namespace parabola_focus_x_coordinate_l52_5226

/-- Given a parabola y^2 = 2px (p > 0) and a point A(1, 2) on this parabola,
    if the distance from point A to point B(x, 0) is equal to its distance to the line x = -1,
    then x = 1. -/
theorem parabola_focus_x_coordinate (p : ℝ) (x : ℝ) :
  p > 0 →
  (2 : ℝ)^2 = 2 * p * 1 →
  (∀ y : ℝ, y^2 = 2 * p * x ↔ (y = 2 ∧ x = 1)) →
  (x - 1)^2 + 2^2 = (1 - (-1))^2 →
  x = 1 := by
sorry

end parabola_focus_x_coordinate_l52_5226


namespace function_inequality_l52_5294

theorem function_inequality (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 2) = -f x)
  (h2 : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → x₁ < x₂ → x₂ ≤ 2 → f x₁ < f x₂)
  (h3 : ∀ x : ℝ, f (x + 2) = f (-x + 2)) :
  f 4.5 < f 7 ∧ f 7 < f 6.5 :=
sorry

end function_inequality_l52_5294


namespace sum_of_i_powers_l52_5214

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^12 + i^17 + i^22 + i^27 + i^32 + i^37 = 2 := by
  sorry

end sum_of_i_powers_l52_5214


namespace same_solution_implies_k_equals_two_l52_5217

theorem same_solution_implies_k_equals_two (k : ℝ) :
  (∃ x : ℝ, (2 * x - 1) / 3 = 5 ∧ k * x - 1 = 15) →
  k = 2 :=
by sorry

end same_solution_implies_k_equals_two_l52_5217


namespace zeros_of_odd_and_even_functions_l52_5297

-- Define odd and even functions
def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def EvenFunction (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Define the number of zeros for a function
def NumberOfZeros (f : ℝ → ℝ) : ℕ := sorry

-- Theorem statement
theorem zeros_of_odd_and_even_functions 
  (f g : ℝ → ℝ) 
  (hf : OddFunction f) 
  (hg : EvenFunction g) :
  (∃ k : ℕ, NumberOfZeros f = 2 * k + 1) ∧ 
  (∃ m : ℕ, NumberOfZeros g = m) :=
sorry

end zeros_of_odd_and_even_functions_l52_5297


namespace square_tens_seven_units_six_l52_5278

theorem square_tens_seven_units_six (n : ℤ) : 
  (n^2 % 100 ≥ 70) ∧ (n^2 % 100 < 80) → n^2 % 10 = 6 := by
  sorry

end square_tens_seven_units_six_l52_5278


namespace total_ways_to_choose_courses_l52_5276

-- Define the number of courses of each type
def num_courses_A : ℕ := 4
def num_courses_B : ℕ := 2

-- Define the total number of courses to be chosen
def total_courses_to_choose : ℕ := 4

-- Define the function to calculate the number of ways to choose courses
def num_ways_to_choose : ℕ := 
  (num_courses_B.choose 1 * num_courses_A.choose 3) +
  (num_courses_B.choose 2 * num_courses_A.choose 2)

-- Theorem statement
theorem total_ways_to_choose_courses : num_ways_to_choose = 14 := by
  sorry


end total_ways_to_choose_courses_l52_5276


namespace wednesday_production_l52_5286

/-- The number of clay pots Nancy created on each day of the week --/
structure ClayPotProduction where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- The conditions of Nancy's clay pot production --/
def nancysProduction : ClayPotProduction where
  monday := 12
  tuesday := 12 * 2
  wednesday := 50 - (12 + 12 * 2)

/-- Theorem stating that Nancy created 14 clay pots on Wednesday --/
theorem wednesday_production : nancysProduction.wednesday = 14 := by
  sorry

end wednesday_production_l52_5286


namespace compound_weight_proof_l52_5212

/-- Atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- Atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- Number of Aluminum atoms in the compound -/
def num_Al : ℕ := 1

/-- Number of Bromine atoms in the compound -/
def num_Br : ℕ := 3

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 267

/-- Theorem stating that the molecular weight of the compound is approximately 267 g/mol -/
theorem compound_weight_proof :
  ∃ ε > 0, abs (molecular_weight - (num_Al * atomic_weight_Al + num_Br * atomic_weight_Br)) < ε :=
sorry

end compound_weight_proof_l52_5212


namespace girl_scout_cookies_l52_5213

theorem girl_scout_cookies (total_goal : ℕ) (boxes_left : ℕ) (first_customer : ℕ) : 
  total_goal = 150 →
  boxes_left = 75 →
  first_customer = 5 →
  let second_customer := 4 * first_customer
  let third_customer := second_customer / 2
  let fourth_customer := 3 * third_customer
  let sold_to_first_four := first_customer + second_customer + third_customer + fourth_customer
  total_goal - boxes_left - sold_to_first_four = 10 := by
sorry


end girl_scout_cookies_l52_5213


namespace count_distinct_sums_of_special_fractions_l52_5290

def IsSpecialFraction (a b : ℕ+) : Prop := a + b = 17

def SumOfSpecialFractions (n : ℕ) : Prop :=
  ∃ (a₁ b₁ a₂ b₂ : ℕ+),
    IsSpecialFraction a₁ b₁ ∧
    IsSpecialFraction a₂ b₂ ∧
    n = (a₁ : ℚ) / b₁ + (a₂ : ℚ) / b₂

theorem count_distinct_sums_of_special_fractions :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, SumOfSpecialFractions n) ∧
    (∀ n, SumOfSpecialFractions n → n ∈ s) ∧
    s.card = 2 :=
sorry

end count_distinct_sums_of_special_fractions_l52_5290


namespace min_cost_floppy_cd_l52_5237

/-- The minimum cost of 3 floppy disks and 9 CDs given price constraints -/
theorem min_cost_floppy_cd (x y : ℝ) 
  (h1 : 4 * x + 5 * y ≥ 20) 
  (h2 : 6 * x + 3 * y ≤ 24) : 
  ∃ (m : ℝ), m = 3 * x + 9 * y ∧ m ≥ 22 ∧ ∀ (n : ℝ), n = 3 * x + 9 * y → n ≥ m :=
sorry

end min_cost_floppy_cd_l52_5237


namespace final_student_count_l52_5261

def initial_students : ℕ := 31
def students_left : ℕ := 5
def new_students : ℕ := 11

theorem final_student_count : 
  initial_students - students_left + new_students = 37 := by
  sorry

end final_student_count_l52_5261


namespace square_field_area_proof_l52_5209

def square_field_area (wire_cost_per_meter : ℚ) (total_cost : ℚ) (gate_width : ℚ) (num_gates : ℕ) : ℚ :=
  let side_length := ((total_cost / wire_cost_per_meter + 2 * gate_width * num_gates) / 4 : ℚ)
  side_length * side_length

theorem square_field_area_proof (wire_cost_per_meter : ℚ) (total_cost : ℚ) (gate_width : ℚ) (num_gates : ℕ) :
  wire_cost_per_meter = 3/2 ∧ total_cost = 999 ∧ gate_width = 1 ∧ num_gates = 2 →
  square_field_area wire_cost_per_meter total_cost gate_width num_gates = 27889 := by
  sorry

#eval square_field_area (3/2) 999 1 2

end square_field_area_proof_l52_5209


namespace min_max_f_l52_5298

noncomputable def f (x : ℝ) : ℝ := Real.cos x + (x + 1) * Real.sin x + 1

theorem min_max_f :
  ∃ (min max : ℝ),
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≥ min) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = min) ∧
    (∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 (2 * Real.pi), f x = max) ∧
    min = -3 * Real.pi / 2 ∧
    max = Real.pi / 2 + 2 :=
by sorry

end min_max_f_l52_5298


namespace inequality_solution_l52_5211

theorem inequality_solution (x : ℕ+) : 
  (x.val - 3) / 3 < 7 - (5 / 3) * x.val ↔ x.val = 1 ∨ x.val = 2 ∨ x.val = 3 := by
  sorry

end inequality_solution_l52_5211


namespace complex_exp_conversion_l52_5274

theorem complex_exp_conversion : Complex.exp (13 * π * Complex.I / 4) * (Real.sqrt 2) = 1 + Complex.I := by
  sorry

end complex_exp_conversion_l52_5274


namespace female_managers_count_l52_5240

/-- Represents a company with employees and managers -/
structure Company where
  total_employees : ℕ
  female_employees : ℕ
  total_managers : ℕ
  male_managers : ℕ

/-- Conditions for the company -/
def company_conditions (c : Company) : Prop :=
  c.female_employees = 500 ∧
  c.total_managers = (2 * c.total_employees) / 5 ∧
  c.male_managers = (2 * (c.total_employees - c.female_employees)) / 5

/-- The theorem to be proved -/
theorem female_managers_count (c : Company) 
  (h : company_conditions c) : 
  c.total_managers - c.male_managers = 200 := by
  sorry

end female_managers_count_l52_5240


namespace probability_A_selected_l52_5204

/-- The number of people in the group -/
def group_size : ℕ := 4

/-- The number of representatives to be selected -/
def representatives : ℕ := 2

/-- The probability of person A being selected as a representative -/
def prob_A_selected : ℚ := 1/2

/-- Theorem stating the probability of person A being selected as a representative -/
theorem probability_A_selected :
  prob_A_selected = (representatives : ℚ) / group_size :=
by sorry

end probability_A_selected_l52_5204


namespace product_remainder_l52_5289

theorem product_remainder (N : ℕ) : 
  (1274 * 1275 * N * 1285) % 12 = 6 → N % 12 = 1 := by
  sorry

end product_remainder_l52_5289


namespace min_value_expression_l52_5270

theorem min_value_expression (x : ℝ) (h : 0 ≤ x ∧ x < 4) :
  ∃ (min : ℝ), min = Real.sqrt 5 ∧
  ∀ y, 0 ≤ y ∧ y < 4 → (y^2 + 2*y + 6) / (2*y + 2) ≥ min :=
sorry

end min_value_expression_l52_5270


namespace nancy_age_proof_l52_5216

/-- Nancy's age in years -/
def nancy_age : ℕ := 5

/-- Nancy's grandmother's age in years -/
def grandmother_age : ℕ := 10 * nancy_age

/-- Age difference between Nancy's grandmother and Nancy at Nancy's birth -/
def age_difference : ℕ := 45

theorem nancy_age_proof :
  nancy_age = 5 ∧
  grandmother_age = 10 * nancy_age ∧
  grandmother_age - nancy_age = age_difference :=
by sorry

end nancy_age_proof_l52_5216


namespace find_number_l52_5222

theorem find_number (x : ℚ) : ((x / 9) - 13) / 7 - 8 = 13 → x = 1440 := by
  sorry

end find_number_l52_5222


namespace iceland_visitors_iceland_visitor_count_l52_5246

theorem iceland_visitors (total : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 50)
  (h2 : norway = 23)
  (h3 : both = 21)
  (h4 : neither = 23) :
  total = (total - norway - neither + both) + norway - both + neither :=
by sorry

theorem iceland_visitor_count : 
  ∃ (iceland : ℕ), iceland = 50 - 23 - 23 + 21 ∧ iceland = 25 :=
by sorry

end iceland_visitors_iceland_visitor_count_l52_5246


namespace erasers_per_friend_l52_5281

/-- Given 9306 erasers shared among 99 friends, prove that each friend receives 94 erasers. -/
theorem erasers_per_friend :
  let total_erasers : ℕ := 9306
  let num_friends : ℕ := 99
  let erasers_per_friend : ℕ := total_erasers / num_friends
  erasers_per_friend = 94 := by sorry

end erasers_per_friend_l52_5281


namespace correct_remaining_insects_l52_5284

/-- Calculates the number of remaining insects in the playground -/
def remaining_insects (spiders ants ladybugs flown_away : ℕ) : ℕ :=
  spiders + ants + ladybugs - flown_away

/-- Theorem stating that the number of remaining insects is correct -/
theorem correct_remaining_insects :
  remaining_insects 3 12 8 2 = 21 := by
  sorry

end correct_remaining_insects_l52_5284


namespace ellipse_k_range_l52_5206

-- Define the equation
def ellipse_equation (x y k : ℝ) : Prop := x^2 + k*y^2 = 2

-- Define what it means for the equation to represent an ellipse with foci on the x-axis
def is_ellipse_with_foci_on_x_axis (k : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  ∀ (x y : ℝ), ellipse_equation x y k ↔ (x^2 / (a^2) + y^2 / (b^2) = 1)

-- Theorem statement
theorem ellipse_k_range :
  ∀ k : ℝ, is_ellipse_with_foci_on_x_axis k ↔ k > 1 := by
  sorry

end ellipse_k_range_l52_5206


namespace steven_name_day_l52_5247

def wordsOnDay (n : ℕ) : ℕ :=
  2 * (n / 2) + 4 * ((n - 1) / 2)

theorem steven_name_day (n : ℕ) : wordsOnDay n = 44 ↔ n = 16 := by
  sorry

end steven_name_day_l52_5247


namespace cyclic_sum_inequality_l52_5203

theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) ≤ 3 * (a * b + b * c + c * a) / (2 * (a + b + c)) := by
  sorry

end cyclic_sum_inequality_l52_5203


namespace find_a_l52_5250

def A : Set ℝ := {x | 1 < x ∧ x < 7}
def B (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 5}

theorem find_a : ∃ a : ℝ, A ∩ B a = {x | 3 < x ∧ x < 7} → a = 2 := by
  sorry

end find_a_l52_5250


namespace expression_evaluation_l52_5267

theorem expression_evaluation : 2 - (-3) - 4 + (-5) - 6 + 7 = -3 := by
  sorry

end expression_evaluation_l52_5267


namespace combined_average_age_l52_5231

theorem combined_average_age (room_a_count : ℕ) (room_b_count : ℕ) 
  (room_a_avg : ℚ) (room_b_avg : ℚ) :
  room_a_count = 8 →
  room_b_count = 6 →
  room_a_avg = 35 →
  room_b_avg = 30 →
  let total_count := room_a_count + room_b_count
  let total_age := room_a_count * room_a_avg + room_b_count * room_b_avg
  (total_age / total_count : ℚ) = 32.86 := by
  sorry

#eval (8 * 35 + 6 * 30) / (8 + 6)

end combined_average_age_l52_5231


namespace even_numbers_set_builder_notation_l52_5224

-- Define the set of even numbers
def EvenNumbers : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}

-- State the theorem
theorem even_numbers_set_builder_notation : 
  EvenNumbers = {x : ℤ | ∃ n : ℤ, x = 2 * n} := by sorry

end even_numbers_set_builder_notation_l52_5224


namespace complex_power_magnitude_l52_5260

theorem complex_power_magnitude : Complex.abs ((2 : ℂ) + Complex.I) ^ 8 = 625 := by
  sorry

end complex_power_magnitude_l52_5260


namespace sqrt_3_times_sqrt_12_l52_5292

theorem sqrt_3_times_sqrt_12 : Real.sqrt 3 * Real.sqrt 12 = 6 := by sorry

end sqrt_3_times_sqrt_12_l52_5292


namespace namjoons_position_proof_l52_5273

def namjoons_position (seokjins_position : ℕ) (people_between : ℕ) : ℕ :=
  seokjins_position + people_between

theorem namjoons_position_proof (seokjins_position : ℕ) (people_between : ℕ) :
  namjoons_position seokjins_position people_between = seokjins_position + people_between :=
by
  sorry

end namjoons_position_proof_l52_5273


namespace smallest_number_in_arithmetic_sequence_l52_5252

theorem smallest_number_in_arithmetic_sequence (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 29 →
  b = 30 →
  c = b + 5 →
  a < b ∧ b < c →
  a = 22 := by sorry

end smallest_number_in_arithmetic_sequence_l52_5252


namespace sqrt_sum_square_product_l52_5279

theorem sqrt_sum_square_product (x : ℝ) :
  Real.sqrt (9 + x) + Real.sqrt (25 - x) = 10 →
  (9 + x) * (25 - x) = 1089 := by
  sorry

end sqrt_sum_square_product_l52_5279


namespace quadratic_function_theorem_l52_5266

-- Define the quadratic function f(x)
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

-- State the theorem
theorem quadratic_function_theorem (a b : ℝ) :
  (∀ x, (f (f a b x + 2*x) a b) / (f a b x) = x^2 + 2023*x + 2040) →
  a = 2021 ∧ b = 1 := by
  sorry

end quadratic_function_theorem_l52_5266


namespace mystery_book_price_l52_5219

theorem mystery_book_price (biography_price : ℝ) (total_discount : ℝ) 
  (biography_quantity : ℕ) (mystery_quantity : ℕ) (total_discount_rate : ℝ) 
  (mystery_discount_rate : ℝ) :
  biography_price = 20 →
  total_discount = 19 →
  biography_quantity = 5 →
  mystery_quantity = 3 →
  total_discount_rate = 0.43 →
  mystery_discount_rate = 0.375 →
  ∃ (mystery_price : ℝ),
    mystery_price * mystery_quantity * mystery_discount_rate + 
    biography_price * biography_quantity * (total_discount_rate - mystery_discount_rate) = 
    total_discount ∧
    mystery_price = 12 :=
by sorry

end mystery_book_price_l52_5219


namespace mall_sales_growth_rate_l52_5272

theorem mall_sales_growth_rate :
  let initial_sales := 1000000  -- January sales in yuan
  let feb_decrease := 0.1       -- 10% decrease in February
  let april_sales := 1296000    -- April sales in yuan
  let growth_rate := 0.2        -- 20% growth rate to be proven
  (initial_sales * (1 - feb_decrease) * (1 + growth_rate)^2 = april_sales) := by
  sorry

end mall_sales_growth_rate_l52_5272


namespace polygon_sides_l52_5251

theorem polygon_sides (sum_interior_angles sum_exterior_angles : ℕ) : 
  sum_interior_angles - sum_exterior_angles = 720 →
  sum_exterior_angles = 360 →
  (∃ n : ℕ, sum_interior_angles = (n - 2) * 180 ∧ n = 8) :=
by sorry

end polygon_sides_l52_5251


namespace probability_of_27_l52_5200

/-- Represents a die with numbered and blank faces -/
structure Die :=
  (total_faces : ℕ)
  (numbered_faces : ℕ)
  (min_number : ℕ)
  (max_number : ℕ)

/-- Calculates the number of ways to get a sum with two dice -/
def waysToGetSum (d1 d2 : Die) (sum : ℕ) : ℕ :=
  sorry

/-- Calculates the total number of possible outcomes when rolling two dice -/
def totalOutcomes (d1 d2 : Die) : ℕ :=
  d1.total_faces * d2.total_faces

/-- Theorem: Probability of rolling 27 with given dice is 3/100 -/
theorem probability_of_27 :
  let die1 : Die := ⟨20, 18, 1, 18⟩
  let die2 : Die := ⟨20, 17, 3, 20⟩
  (waysToGetSum die1 die2 27 : ℚ) / (totalOutcomes die1 die2 : ℚ) = 3 / 100 := by
  sorry

end probability_of_27_l52_5200


namespace maria_earnings_l52_5257

/-- Calculates the total earnings of a flower saleswoman over three days --/
def flower_sales_earnings (tulip_price rose_price : ℚ) 
  (day1_tulips day1_roses : ℕ) 
  (day2_multiplier : ℚ) 
  (day3_tulip_percentage : ℚ) 
  (day3_roses : ℕ) : ℚ :=
  let day1_earnings := tulip_price * day1_tulips + rose_price * day1_roses
  let day2_earnings := day2_multiplier * day1_earnings
  let day3_tulips := day3_tulip_percentage * (day2_multiplier * day1_tulips)
  let day3_earnings := tulip_price * day3_tulips + rose_price * day3_roses
  day1_earnings + day2_earnings + day3_earnings

/-- Theorem stating that Maria's total earnings over three days is $420 --/
theorem maria_earnings : 
  flower_sales_earnings 2 3 30 20 2 (1/10) 16 = 420 := by
  sorry

end maria_earnings_l52_5257


namespace arithmetic_sequence_problem_l52_5235

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The problem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : IsArithmeticSequence a) 
  (h_eq : 4 * a 3 + a 11 - 3 * a 5 = 10) : 
  a 4 = 5 := by
  sorry

end arithmetic_sequence_problem_l52_5235


namespace recurrence_sequence_uniqueness_l52_5283

/-- A sequence of natural numbers satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

/-- A bounded sequence of natural numbers -/
def BoundedSequence (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n, a n ≤ M

theorem recurrence_sequence_uniqueness (a : ℕ → ℕ) 
  (h_recurrence : RecurrenceSequence a) (h_bounded : BoundedSequence a) :
  ∀ n, a n = 2 := by
  sorry

end recurrence_sequence_uniqueness_l52_5283


namespace middleton_marching_band_max_members_l52_5207

theorem middleton_marching_band_max_members :
  ∀ n : ℕ,
  (30 * n % 21 = 9) →
  (30 * n < 1500) →
  (∀ m : ℕ, (30 * m % 21 = 9) → (30 * m < 1500) → (30 * m ≤ 30 * n)) →
  30 * n = 1470 :=
by sorry

end middleton_marching_band_max_members_l52_5207


namespace toms_ribbon_length_l52_5275

theorem toms_ribbon_length 
  (num_gifts : ℕ) 
  (ribbon_per_gift : ℝ) 
  (remaining_ribbon : ℝ) 
  (h1 : num_gifts = 8)
  (h2 : ribbon_per_gift = 1.5)
  (h3 : remaining_ribbon = 3) :
  (num_gifts : ℝ) * ribbon_per_gift + remaining_ribbon = 15 := by
  sorry

end toms_ribbon_length_l52_5275


namespace sixth_term_of_geometric_sequence_l52_5265

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2 : a 2 = 2)
  (h_a4 : a 4 = 8) :
  a 6 = 32 := by
sorry

end sixth_term_of_geometric_sequence_l52_5265


namespace range_of_a_l52_5227

def f (x a : ℝ) : ℝ := |x - a| + x + 5

theorem range_of_a (a : ℝ) : (∀ x, f x a ≥ 8) ↔ |a + 5| ≥ 3 := by
  sorry

end range_of_a_l52_5227


namespace raisin_mixture_problem_l52_5208

theorem raisin_mixture_problem (raisin_cost nut_cost : ℝ) (raisin_weight : ℝ) :
  nut_cost = 3 * raisin_cost →
  raisin_weight * raisin_cost = 0.29411764705882354 * (raisin_weight * raisin_cost + 4 * nut_cost) →
  raisin_weight = 5 := by
sorry

end raisin_mixture_problem_l52_5208


namespace identity_function_satisfies_conditions_l52_5232

def is_identity_function (f : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, f m = m

theorem identity_function_satisfies_conditions (f : ℕ → ℕ) 
  (h1 : ∀ m : ℕ, f m = 1 ↔ m = 1)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n / f (Nat.gcd m n))
  (h3 : ∀ m : ℕ, (f^[2012]) m = m) :
  is_identity_function f :=
sorry

end identity_function_satisfies_conditions_l52_5232


namespace halfway_fraction_l52_5299

theorem halfway_fraction (a b c : ℚ) : 
  a = 1/4 → b = 1/2 → c = (a + b) / 2 → c = 3/8 := by sorry

end halfway_fraction_l52_5299


namespace product_ab_equals_one_l52_5220

-- Define the variables a and b
variable (a b : ℝ)

-- State the theorem
theorem product_ab_equals_one (h1 : a - b = 4) (h2 : a^2 + b^2 = 18) : a * b = 1 := by
  sorry

end product_ab_equals_one_l52_5220


namespace sum_of_double_root_k_values_l52_5282

/-- The quadratic equation we're working with -/
def quadratic (k x : ℝ) : ℝ := x^2 + 2*k*x + 7*k - 10

/-- The discriminant of our quadratic equation -/
def discriminant (k : ℝ) : ℝ := (2*k)^2 - 4*(7*k - 10)

/-- A value of k for which the quadratic equation has exactly one solution -/
def is_double_root (k : ℝ) : Prop := discriminant k = 0

/-- The theorem stating that the sum of the values of k for which
    the quadratic equation has exactly one solution is 7 -/
theorem sum_of_double_root_k_values :
  ∃ k₁ k₂ : ℝ, k₁ ≠ k₂ ∧ is_double_root k₁ ∧ is_double_root k₂ ∧ k₁ + k₂ = 7 :=
sorry

end sum_of_double_root_k_values_l52_5282


namespace triangle_special_sequence_equilateral_l52_5218

/-- A triangle with angles forming an arithmetic sequence and reciprocals of side lengths forming an arithmetic sequence is equilateral. -/
theorem triangle_special_sequence_equilateral (A B C : ℝ) (a b c : ℝ) :
  -- Angles form an arithmetic sequence
  ∃ (d : ℝ), (B = A + d ∧ C = B + d) →
  -- Reciprocals of side lengths form an arithmetic sequence
  ∃ (k : ℝ), (1/b = 1/a + k ∧ 1/c = 1/b + k) →
  -- Angles sum to 180°
  A + B + C = 180 →
  -- Side lengths are positive
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Triangle inequality
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Conclusion: The triangle is equilateral
  A = 60 ∧ B = 60 ∧ C = 60 := by
sorry

end triangle_special_sequence_equilateral_l52_5218


namespace promotions_recipients_l52_5285

def stadium_capacity : ℕ := 4500
def cap_interval : ℕ := 90
def shirt_interval : ℕ := 45
def sunglasses_interval : ℕ := 60

theorem promotions_recipients : 
  (∀ n : ℕ, n ≤ stadium_capacity → 
    (n % cap_interval = 0 ∧ n % shirt_interval = 0 ∧ n % sunglasses_interval = 0) ↔ 
    n % 180 = 0) →
  (stadium_capacity / 180 = 25) :=
by sorry

end promotions_recipients_l52_5285


namespace lorenzo_stamps_l52_5202

def stamps_needed (current : ℕ) (row_size : ℕ) : ℕ :=
  (row_size - (current % row_size)) % row_size

theorem lorenzo_stamps : stamps_needed 37 8 = 3 := by
  sorry

end lorenzo_stamps_l52_5202


namespace decimal_expansion_of_one_forty_ninth_l52_5291

/-- The repeating sequence in the decimal expansion of 1/49 -/
def repeating_sequence : List Nat :=
  [0, 2, 0, 4, 0, 8, 1, 6, 3, 2, 6, 5, 3, 0, 6, 1, 2, 2, 4, 4, 8, 9, 7, 9,
   5, 9, 1, 8, 3, 6, 7, 3, 4, 6, 9, 3, 8, 7, 7, 5, 5, 1]

/-- The length of the repeating sequence -/
def sequence_length : Nat := 42

/-- Theorem stating that the decimal expansion of 1/49 has the given repeating sequence -/
theorem decimal_expansion_of_one_forty_ninth :
  ∃ (n : Nat), (1 : ℚ) / 49 = (n : ℚ) / (10^sequence_length - 1) ∧
  repeating_sequence = (n.digits 10).reverse.take sequence_length :=
sorry

end decimal_expansion_of_one_forty_ninth_l52_5291


namespace statement_equivalence_l52_5296

theorem statement_equivalence (x y : ℝ) :
  ((x - 1) * (y + 2) ≠ 0 → x ≠ 1 ∧ y ≠ -2) ↔
  (x = 1 ∨ y = -2 → (x - 1) * (y + 2) = 0) := by
  sorry

end statement_equivalence_l52_5296


namespace julia_tag_total_l52_5230

/-- The number of kids Julia played tag with on Monday -/
def monday_kids : ℕ := 7

/-- The number of kids Julia played tag with on Tuesday -/
def tuesday_kids : ℕ := 13

/-- The total number of kids Julia played tag with -/
def total_kids : ℕ := monday_kids + tuesday_kids

theorem julia_tag_total : total_kids = 20 := by
  sorry

end julia_tag_total_l52_5230


namespace intersection_condition_l52_5249

/-- The function f(x) = (m-3)x^2 - 4x + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 3) * x^2 - 4*x + 2

/-- The graph of f intersects the x-axis at only one point -/
def intersects_at_one_point (m : ℝ) : Prop :=
  ∃! x, f m x = 0

/-- Theorem: The graph of f(x) = (m-3)x^2 - 4x + 2 intersects the x-axis at only one point
    if and only if m = 3 or m = 5 -/
theorem intersection_condition (m : ℝ) :
  intersects_at_one_point m ↔ m = 3 ∨ m = 5 := by
  sorry

end intersection_condition_l52_5249


namespace angle_difference_range_l52_5223

theorem angle_difference_range (α β : Real) (h1 : -π/2 < α) (h2 : α < β) (h3 : β < π/2) :
  ∃ (x : Real), -π < x ∧ x < 0 ∧ x = α - β :=
sorry

end angle_difference_range_l52_5223


namespace questions_left_blank_l52_5243

/-- Given a math test with a total number of questions and the number of questions answered,
    prove that the number of questions left blank is the difference between the total and answered questions. -/
theorem questions_left_blank (total : ℕ) (answered : ℕ) (h : answered ≤ total) :
  total - answered = total - answered :=
by sorry

end questions_left_blank_l52_5243
