import Mathlib

namespace percent_decrease_l791_79180

theorem percent_decrease (original_price sale_price : ℝ) (h1 : original_price = 100) (h2 : sale_price = 50) :
  (original_price - sale_price) / original_price * 100 = 50 := by
  sorry

end percent_decrease_l791_79180


namespace no_obtuse_equilateral_triangle_l791_79143

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of triangles
def Triangle.isEquilateral (t : Triangle) : Prop :=
  t.angle1 = t.angle2 ∧ t.angle2 = t.angle3

def Triangle.isObtuse (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: An obtuse equilateral triangle cannot exist
theorem no_obtuse_equilateral_triangle :
  ¬ ∃ (t : Triangle), t.isEquilateral ∧ t.isObtuse ∧ t.angle1 + t.angle2 + t.angle3 = 180 :=
by sorry

end no_obtuse_equilateral_triangle_l791_79143


namespace area_of_triangle_DBC_l791_79178

/-- Given points A, B, C, D, and E in a coordinate plane, prove that the area of triangle DBC is 20 -/
theorem area_of_triangle_DBC (A B C D E : ℝ × ℝ) : 
  A = (0, 8) → 
  B = (0, 0) → 
  C = (10, 0) → 
  D = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  E = (B.1 + (C.1 - B.1) / 3, B.2 + (C.2 - B.2) / 3) → 
  (1 / 2) * (C.1 - B.1) * D.2 = 20 := by
  sorry

end area_of_triangle_DBC_l791_79178


namespace division_by_reciprocal_five_divided_by_one_fifth_l791_79146

theorem division_by_reciprocal (a : ℝ) (b : ℝ) (hb : b ≠ 0) :
  a / (1 / b) = a * b := by sorry

theorem five_divided_by_one_fifth :
  5 / (1 / 5) = 25 := by sorry

end division_by_reciprocal_five_divided_by_one_fifth_l791_79146


namespace max_sum_n_l791_79139

/-- An arithmetic sequence with first term 11 and common difference -2 -/
def arithmeticSequence (n : ℕ) : ℤ :=
  11 - 2 * (n - 1)

/-- The sum of the first n terms of the arithmetic sequence -/
def sumOfTerms (n : ℕ) : ℤ :=
  n * (arithmeticSequence 1 + arithmeticSequence n) / 2

/-- The value of n that maximizes the sum of the first n terms -/
theorem max_sum_n : ∃ (n : ℕ), n = 6 ∧ 
  ∀ (m : ℕ), sumOfTerms m ≤ sumOfTerms n :=
sorry

end max_sum_n_l791_79139


namespace limit_equals_third_derivative_at_one_l791_79187

-- Define a real-valued function f that is differentiable on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- State the theorem
theorem limit_equals_third_derivative_at_one :
  (∀ ε > 0, ∃ δ > 0, ∀ x ∈ Set.Ioo (1 - δ) (1 + δ) \ {1},
    |((f (1 + (x - 1)) - f 1) / (3 * (x - 1))) - (1/3 * deriv f 1)| < ε) :=
sorry

end limit_equals_third_derivative_at_one_l791_79187


namespace pipes_remaining_proof_l791_79149

/-- The number of pipes in a triangular pyramid with n layers -/
def triangular_pyramid (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of pipes available -/
def total_pipes : ℕ := 200

/-- The maximum number of complete layers in the pyramid -/
def max_layers : ℕ := 19

/-- The number of pipes left over -/
def pipes_left_over : ℕ := total_pipes - triangular_pyramid max_layers

theorem pipes_remaining_proof :
  pipes_left_over = 10 :=
sorry

end pipes_remaining_proof_l791_79149


namespace tangent_to_ln_curve_l791_79199

theorem tangent_to_ln_curve (k : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ k * x = Real.log x ∧ k = (Real.log x) / x) →
  (k * 0 = Real.log 0) →
  k = 1 / Real.exp 1 := by
sorry

end tangent_to_ln_curve_l791_79199


namespace binomial_15_choose_4_l791_79179

theorem binomial_15_choose_4 : Nat.choose 15 4 = 1365 := by
  sorry

end binomial_15_choose_4_l791_79179


namespace f_sum_positive_l791_79112

/-- The function f(x) = x + x³ -/
def f (x : ℝ) : ℝ := x + x^3

/-- Theorem: For x₁, x₂ ∈ ℝ with x₁ + x₂ > 0, f(x₁) + f(x₂) > 0 -/
theorem f_sum_positive (x₁ x₂ : ℝ) (h : x₁ + x₂ > 0) : f x₁ + f x₂ > 0 := by
  sorry

end f_sum_positive_l791_79112


namespace ratio_of_x_intercepts_l791_79125

/-- Two lines with the same non-zero y-intercept, one with slope 8 and x-intercept (s, 0),
    the other with slope 4 and x-intercept (t, 0), have s/t = 1/2 -/
theorem ratio_of_x_intercepts (b s t : ℝ) (hb : b ≠ 0) : 
  (0 = 8 * s + b) → (0 = 4 * t + b) → s / t = 1 / 2 := by
  sorry

end ratio_of_x_intercepts_l791_79125


namespace aria_cookie_expense_is_2356_l791_79173

/-- The amount Aria spent on cookies in March -/
def aria_cookie_expense : ℕ :=
  let cookies_per_day : ℕ := 4
  let cost_per_cookie : ℕ := 19
  let days_in_march : ℕ := 31
  cookies_per_day * cost_per_cookie * days_in_march

/-- Theorem stating that Aria spent 2356 dollars on cookies in March -/
theorem aria_cookie_expense_is_2356 : aria_cookie_expense = 2356 := by
  sorry

end aria_cookie_expense_is_2356_l791_79173


namespace consecutive_majors_probability_l791_79181

/-- Represents the number of people around the table -/
def total_people : ℕ := 12

/-- Represents the number of math majors -/
def math_majors : ℕ := 5

/-- Represents the number of physics majors -/
def physics_majors : ℕ := 4

/-- Represents the number of biology majors -/
def biology_majors : ℕ := 3

/-- Represents the probability of the desired seating arrangement -/
def seating_probability : ℚ := 1 / 5775

theorem consecutive_majors_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := 
    Nat.factorial (math_majors - 1) * Nat.factorial physics_majors * Nat.factorial biology_majors
  (favorable_arrangements : ℚ) / total_arrangements = seating_probability := by
  sorry

end consecutive_majors_probability_l791_79181


namespace total_books_on_cart_l791_79157

/-- The number of books Nancy shelved from the cart -/
structure BookCart where
  top_history : ℕ
  top_romance : ℕ
  top_poetry : ℕ
  bottom_western : ℕ
  bottom_biography : ℕ
  bottom_scifi : ℕ
  bottom_culinary : ℕ

/-- The theorem stating the total number of books on the cart -/
theorem total_books_on_cart (cart : BookCart) : 
  cart.top_history = 12 →
  cart.top_romance = 8 →
  cart.top_poetry = 4 →
  cart.bottom_western = 5 →
  cart.bottom_biography = 6 →
  cart.bottom_scifi = 3 →
  cart.bottom_culinary = 2 →
  ∃ (total : ℕ), total = 88 ∧ 
    total = cart.top_history + cart.top_romance + cart.top_poetry + 
            (cart.bottom_western + cart.bottom_biography + cart.bottom_scifi + cart.bottom_culinary) * 4 :=
by sorry


end total_books_on_cart_l791_79157


namespace dan_found_no_money_l791_79147

/-- The amount of money Dan spent on a snake toy -/
def snake_toy_cost : ℚ := 11.76

/-- The amount of money Dan spent on a cage -/
def cage_cost : ℚ := 14.54

/-- The total cost of Dan's purchases -/
def total_cost : ℚ := 26.3

/-- The amount of money Dan found on the ground -/
def money_found : ℚ := total_cost - (snake_toy_cost + cage_cost)

theorem dan_found_no_money : money_found = 0 := by sorry

end dan_found_no_money_l791_79147


namespace system_solution_l791_79189

theorem system_solution : ∃ (x y : ℚ), 
  (4 * x + 3 * y = 1) ∧ 
  (6 * x - 9 * y = -8) ∧ 
  (x = -5/18) ∧ 
  (y = 19/27) := by
  sorry

end system_solution_l791_79189


namespace biased_coin_probability_l791_79126

theorem biased_coin_probability (p : ℝ) (n : ℕ) (h_p : p = 3/4) (h_n : n = 4) :
  1 - p^n = 175/256 := by
  sorry

end biased_coin_probability_l791_79126


namespace f_of_f_2_l791_79118

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.log x / Real.log 2
  else if x ≥ 1 then 1 / x^2
  else 0  -- This case is added to make the function total

theorem f_of_f_2 : f (f 2) = -2 := by
  sorry

end f_of_f_2_l791_79118


namespace fish_weight_l791_79138

/-- Given a barrel of fish with the following properties:
  1. The total weight of the barrel and fish is 54 kg.
  2. After removing half of the fish, the total weight is 29 kg.
  This theorem proves that the initial weight of the fish alone is 50 kg. -/
theorem fish_weight (barrel_weight : ℝ) (fish_weight : ℝ) 
  (h1 : barrel_weight + fish_weight = 54)
  (h2 : barrel_weight + fish_weight / 2 = 29) :
  fish_weight = 50 := by
  sorry

end fish_weight_l791_79138


namespace smallest_divisible_by_15_and_48_l791_79158

theorem smallest_divisible_by_15_and_48 : ∀ n : ℕ, n > 0 ∧ 15 ∣ n ∧ 48 ∣ n → n ≥ 240 := by
  sorry

end smallest_divisible_by_15_and_48_l791_79158


namespace perfect_square_sum_l791_79190

theorem perfect_square_sum (a b : ℕ) : 
  ∃ (n : ℕ), 3^a + 4^b = n^2 ↔ a = 2 ∧ b = 2 := by
  sorry

end perfect_square_sum_l791_79190


namespace popcorn_probability_l791_79114

theorem popcorn_probability : 
  let white_ratio : ℚ := 3/4
  let yellow_ratio : ℚ := 1/4
  let white_pop_prob : ℚ := 2/3
  let yellow_pop_prob : ℚ := 3/4
  let fizz_prob : ℚ := 1/4
  
  let white_pop_fizz : ℚ := white_ratio * white_pop_prob * fizz_prob
  let yellow_pop_fizz : ℚ := yellow_ratio * yellow_pop_prob * fizz_prob
  let total_pop_fizz : ℚ := white_pop_fizz + yellow_pop_fizz
  
  white_pop_fizz / total_pop_fizz = 8/11 := by
  sorry

end popcorn_probability_l791_79114


namespace lee_cookies_l791_79192

/-- Given that Lee can make 18 cookies with 2 cups of flour, 
    this function calculates how many cookies he can make with any number of cups of flour. -/
def cookies_from_flour (cups : ℚ) : ℚ :=
  (18 / 2) * cups

/-- Theorem stating that Lee can make 45 cookies with 5 cups of flour. -/
theorem lee_cookies : cookies_from_flour 5 = 45 := by
  sorry

end lee_cookies_l791_79192


namespace cost_decrease_l791_79176

theorem cost_decrease (original_cost : ℝ) (decrease_percentage : ℝ) (new_cost : ℝ) : 
  original_cost = 200 →
  decrease_percentage = 50 →
  new_cost = original_cost * (1 - decrease_percentage / 100) →
  new_cost = 100 := by
sorry

end cost_decrease_l791_79176


namespace five_balls_four_boxes_count_l791_79122

/-- The number of ways to distribute n indistinguishable balls among k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls among 4 distinguishable boxes -/
def five_balls_four_boxes : ℕ := distribute_balls 5 4

theorem five_balls_four_boxes_count : five_balls_four_boxes = 56 := by sorry

end five_balls_four_boxes_count_l791_79122


namespace ed_hotel_stay_l791_79133

/-- The number of hours Ed stayed in the hotel last night -/
def hours_stayed : ℕ := 6

/-- The cost per hour for staying at night -/
def night_cost_per_hour : ℚ := 3/2

/-- The cost per hour for staying in the morning -/
def morning_cost_per_hour : ℚ := 2

/-- Ed's initial money -/
def initial_money : ℕ := 80

/-- The number of hours Ed stayed in the morning -/
def morning_hours : ℕ := 4

/-- The amount of money Ed had left after paying for his stay -/
def money_left : ℕ := 63

theorem ed_hotel_stay :
  hours_stayed * night_cost_per_hour + 
  morning_hours * morning_cost_per_hour = 
  initial_money - money_left :=
by sorry

end ed_hotel_stay_l791_79133


namespace ellipse_equation_and_slope_l791_79161

/-- Represents an ellipse with given properties -/
structure Ellipse where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  eccentricity : ℝ
  passes_through : ℝ × ℝ

/-- Theorem about the equation of the ellipse and the slope of line l -/
theorem ellipse_equation_and_slope (e : Ellipse) 
  (h1 : e.center = (0, 0))
  (h2 : e.foci_on_x_axis = true)
  (h3 : e.eccentricity = Real.sqrt 3 / 2)
  (h4 : e.passes_through = (Real.sqrt 2, Real.sqrt 2 / 2)) :
  (∃ (x y : ℝ), x^2 / 4 + y^2 = 1) ∧ 
  (∃ (k : ℝ), k = 1/2 ∨ k = -1/2) := by sorry

end ellipse_equation_and_slope_l791_79161


namespace chebyshev_polynomial_3_and_root_sum_l791_79183

-- Define Chebyshev polynomials
def is_chebyshev_polynomial (P : ℝ → ℝ) (n : ℕ) : Prop :=
  ∀ x, P (Real.cos x) = Real.cos (n * x)

-- Define P₃
def P₃ (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem chebyshev_polynomial_3_and_root_sum :
  ∃ (a b c d : ℝ),
    (is_chebyshev_polynomial (P₃ a b c d) 3) ∧
    (a = 4 ∧ b = 0 ∧ c = -3 ∧ d = 0) ∧
    (∃ (x₁ x₂ x₃ : ℝ),
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
      x₁ ∈ Set.Ioo (-1 : ℝ) 1 ∧
      x₂ ∈ Set.Ioo (-1 : ℝ) 1 ∧
      x₃ ∈ Set.Ioo (-1 : ℝ) 1 ∧
      (4 * x₁^3 - 3 * x₁ = 1/2) ∧
      (4 * x₂^3 - 3 * x₂ = 1/2) ∧
      (4 * x₃^3 - 3 * x₃ = 1/2) ∧
      x₁ + x₂ + x₃ = 0) :=
by
  sorry


end chebyshev_polynomial_3_and_root_sum_l791_79183


namespace correct_calculation_l791_79121

theorem correct_calculation (x : ℝ) (h : x - 21 = 52) : 40 * x = 2920 := by
  sorry

end correct_calculation_l791_79121


namespace range_of_f_l791_79134

def f (x : ℕ) : ℤ := 2 * x - 3

def domain : Set ℕ := {x : ℕ | 1 ≤ x ∧ x ≤ 5}

theorem range_of_f : {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 1, 3, 5, 7} := by sorry

end range_of_f_l791_79134


namespace max_bent_strips_achievable_14_bent_strips_max_bent_strips_is_14_l791_79124

/-- Represents a 3x3x3 cube --/
structure Cube :=
  (side_length : ℕ := 3)

/-- Represents a 3x1 strip --/
structure Strip :=
  (length : ℕ := 3)
  (width : ℕ := 1)

/-- Represents the configuration of strips on the cube --/
structure CubeConfiguration :=
  (cube : Cube)
  (total_strips : ℕ := 18)
  (bent_strips : ℕ)
  (flat_strips : ℕ)

/-- The theorem stating the maximum number of bent strips --/
theorem max_bent_strips (config : CubeConfiguration) : config.bent_strips ≤ 14 :=
by sorry

/-- The theorem stating that 14 bent strips is achievable --/
theorem achievable_14_bent_strips : ∃ (config : CubeConfiguration), config.bent_strips = 14 ∧ config.flat_strips = 4 :=
by sorry

/-- The main theorem combining the above results --/
theorem max_bent_strips_is_14 : 
  (∀ (config : CubeConfiguration), config.bent_strips ≤ 14) ∧
  (∃ (config : CubeConfiguration), config.bent_strips = 14) :=
by sorry

end max_bent_strips_achievable_14_bent_strips_max_bent_strips_is_14_l791_79124


namespace complex_expression_simplification_l791_79185

theorem complex_expression_simplification :
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) - Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt (2/3)) = -Real.sqrt 2 := by
  sorry

end complex_expression_simplification_l791_79185


namespace divisors_of_720_l791_79113

theorem divisors_of_720 : Finset.card (Nat.divisors 720) = 30 := by
  sorry

end divisors_of_720_l791_79113


namespace total_crayons_l791_79110

def number_of_boxes : ℕ := 7
def crayons_per_box : ℕ := 5

theorem total_crayons : number_of_boxes * crayons_per_box = 35 := by
  sorry

end total_crayons_l791_79110


namespace prob_all_heads_or_five_plus_tails_is_one_eighth_l791_79136

/-- The number of coins being flipped -/
def num_coins : ℕ := 6

/-- The probability of getting heads on a single fair coin flip -/
def p_heads : ℚ := 1/2

/-- The probability of getting tails on a single fair coin flip -/
def p_tails : ℚ := 1/2

/-- The probability of getting all heads or at least five tails when flipping six fair coins -/
def prob_all_heads_or_five_plus_tails : ℚ := 1/8

/-- Theorem stating that the probability of getting all heads or at least five tails 
    when flipping six fair coins is 1/8 -/
theorem prob_all_heads_or_five_plus_tails_is_one_eighth :
  prob_all_heads_or_five_plus_tails = 1/8 := by
  sorry

end prob_all_heads_or_five_plus_tails_is_one_eighth_l791_79136


namespace problem_statement_l791_79104

theorem problem_statement (x : ℚ) (h : 5 * x - 8 = 15 * x - 2) : 5 * (x - 3) = -18 := by
  sorry

end problem_statement_l791_79104


namespace total_marbles_l791_79188

theorem total_marbles (blue red orange : ℕ) : 
  blue = red + orange → -- Half of the marbles are blue
  red = 6 →             -- There are 6 red marbles
  orange = 6 →          -- There are 6 orange marbles
  blue + red + orange = 24 := by
sorry

end total_marbles_l791_79188


namespace square_of_odd_is_sum_of_consecutive_integers_l791_79172

theorem square_of_odd_is_sum_of_consecutive_integers :
  ∀ n : ℕ, n > 1 → Odd n → ∃ j : ℕ, n^2 = j + (j + 1) := by sorry

end square_of_odd_is_sum_of_consecutive_integers_l791_79172


namespace jake_eighth_week_hours_l791_79164

def hours_worked : List ℕ := [14, 9, 12, 15, 11, 13, 10]
def total_weeks : ℕ := 8
def required_average : ℕ := 12

theorem jake_eighth_week_hours :
  ∃ (x : ℕ), 
    (List.sum hours_worked + x) / total_weeks = required_average ∧
    x = 12 := by
  sorry

end jake_eighth_week_hours_l791_79164


namespace box_face_ratio_l791_79175

/-- Given a rectangular box with length l, width w, and height h -/
structure Box where
  l : ℝ
  w : ℝ
  h : ℝ

/-- Properties of the box -/
def BoxProperties (box : Box) : Prop :=
  box.l > 0 ∧ box.w > 0 ∧ box.h > 0 ∧
  box.l * box.w * box.h = 5184 ∧
  box.l * box.h = 288 ∧
  box.w * box.h = (1/2) * box.l * box.w

theorem box_face_ratio (box : Box) (hp : BoxProperties box) :
  (box.l * box.w) / (box.l * box.h) = 3 / 2 := by
  sorry

end box_face_ratio_l791_79175


namespace circle_equation_and_intersection_range_l791_79163

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (h : ℝ), (x - h)^2 + (y - (3*h - 5))^2 = 1 ∧ (3 - h)^2 + (3 - (3*h - 5))^2 = 1 ∧ (2 - h)^2 + (4 - (3*h - 5))^2 = 1

-- Define the circle with diameter PQ
def circle_PQ (m x y : ℝ) : Prop := x^2 + y^2 = m^2

theorem circle_equation_and_intersection_range :
  ∃ (h : ℝ), 
    (∀ x y : ℝ, circle_C x y ↔ (x - 3)^2 + (y - 4)^2 = 1) ∧
    (∀ m : ℝ, m > 0 → 
      (∃ x y : ℝ, circle_C x y ∧ circle_PQ m x y) ↔ 
      (4 ≤ m ∧ m ≤ 6)) :=
sorry

end circle_equation_and_intersection_range_l791_79163


namespace students_answering_both_correctly_l791_79168

theorem students_answering_both_correctly 
  (total_students : ℕ) 
  (answered_q1 : ℕ) 
  (answered_q2 : ℕ) 
  (not_taken : ℕ) 
  (h1 : total_students = 30) 
  (h2 : answered_q1 = 25) 
  (h3 : answered_q2 = 22) 
  (h4 : not_taken = 5) :
  answered_q1 + answered_q2 - (total_students - not_taken) = 22 := by
  sorry

end students_answering_both_correctly_l791_79168


namespace profit_increase_after_cost_decrease_l791_79177

theorem profit_increase_after_cost_decrease (x y : ℝ) (a : ℝ) 
  (h1 : y - x = x * (a / 100))  -- Initial profit percentage
  (h2 : y - 0.9 * x = 0.9 * x * ((a + 20) / 100))  -- New profit percentage
  : a = 80 := by
sorry

end profit_increase_after_cost_decrease_l791_79177


namespace mark_apple_count_l791_79195

/-- The number of apples Mark has chosen -/
def num_apples (total fruit_count banana_count orange_count : ℕ) : ℕ :=
  total - (banana_count + orange_count)

/-- Theorem stating that Mark has chosen 3 apples -/
theorem mark_apple_count :
  num_apples 12 4 5 = 3 := by
  sorry

end mark_apple_count_l791_79195


namespace pollywogs_disappear_in_44_days_l791_79162

/-- The number of days it takes for all pollywogs to disappear from Elmer's pond -/
def days_until_empty (initial_pollywogs : ℕ) (maturation_rate : ℕ) (melvin_catch_rate : ℕ) (melvin_catch_days : ℕ) : ℕ :=
  let first_phase := melvin_catch_days
  let pollywogs_after_first_phase := initial_pollywogs - (maturation_rate + melvin_catch_rate) * first_phase
  let second_phase := pollywogs_after_first_phase / maturation_rate
  first_phase + second_phase

/-- Theorem stating that it takes 44 days for all pollywogs to disappear from Elmer's pond -/
theorem pollywogs_disappear_in_44_days :
  days_until_empty 2400 50 10 20 = 44 := by
  sorry

end pollywogs_disappear_in_44_days_l791_79162


namespace mary_savings_problem_l791_79182

theorem mary_savings_problem (S : ℝ) (x : ℝ) (h1 : S > 0) (h2 : 0 ≤ x ∧ x ≤ 1) 
  (h3 : 12 * S * x = 7 * S * (1 - x)) : 
  1 - x = 12 / 19 := by sorry

end mary_savings_problem_l791_79182


namespace mailbox_distance_l791_79116

/-- Represents Jeffrey's walking pattern and the total steps taken -/
structure WalkingPattern where
  forward_steps : ℕ
  backward_steps : ℕ
  total_steps : ℕ

/-- Calculates the effective distance covered given a walking pattern -/
def effectiveDistance (pattern : WalkingPattern) : ℕ :=
  let cycle := pattern.forward_steps + pattern.backward_steps
  let effective_steps_per_cycle := pattern.forward_steps - pattern.backward_steps
  (pattern.total_steps / cycle) * effective_steps_per_cycle

/-- Theorem: Given Jeffrey's walking pattern and total steps, the distance to the mailbox is 110 steps -/
theorem mailbox_distance (pattern : WalkingPattern) 
  (h1 : pattern.forward_steps = 3)
  (h2 : pattern.backward_steps = 2)
  (h3 : pattern.total_steps = 330) :
  effectiveDistance pattern = 110 := by
  sorry

end mailbox_distance_l791_79116


namespace max_k_value_l791_79170

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x + x * Real.log x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := a + Real.log x + 1

-- State the theorem
theorem max_k_value (a : ℝ) :
  (f' a (Real.exp (-1)) = 1) →
  (∃ k : ℤ, ∀ x > 1, f a x - k * x + k > 0) →
  (∀ k : ℤ, k > 3 → ∃ x > 1, f a x - k * x + k ≤ 0) :=
sorry

end

end max_k_value_l791_79170


namespace no_valid_chess_sequence_l791_79102

/-- Represents a sequence of moves on a 6x6 chessboard -/
def ChessSequence := Fin 36 → Fin 36

/-- Checks if the difference between consecutive terms alternates between 1 and 2 -/
def validMoves (seq : ChessSequence) : Prop :=
  ∀ i : Fin 35, (i.val % 2 = 0 → |seq (i + 1) - seq i| = 1) ∧
                (i.val % 2 = 1 → |seq (i + 1) - seq i| = 2)

/-- Checks if all elements in the sequence are distinct -/
def allDistinct (seq : ChessSequence) : Prop :=
  ∀ i j : Fin 36, i ≠ j → seq i ≠ seq j

/-- The main theorem: no valid chess sequence exists -/
theorem no_valid_chess_sequence :
  ¬∃ (seq : ChessSequence), validMoves seq ∧ allDistinct seq :=
sorry

end no_valid_chess_sequence_l791_79102


namespace parabola_coefficient_l791_79154

/-- 
Given a parabola y = ax^2 + bx + c with vertex (h, h) and y-intercept (0, -2h), 
where h ≠ 0, the value of b is 6.
-/
theorem parabola_coefficient (a b c h : ℝ) : 
  h ≠ 0 → 
  (∀ x y, y = a * x^2 + b * x + c ↔ y - h = a * (x - h)^2) → 
  c = -2 * h → 
  b = 6 := by sorry

end parabola_coefficient_l791_79154


namespace expression_equals_sum_l791_79152

theorem expression_equals_sum (a b c : ℚ) (ha : a = 7) (hb : b = 11) (hc : c = 13) :
  let numerator := a^3 * (1/b - 1/c) + b^3 * (1/c - 1/a) + c^3 * (1/a - 1/b)
  let denominator := a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)
  numerator / denominator = a + b + c :=
by sorry

end expression_equals_sum_l791_79152


namespace kannon_bananas_l791_79115

/-- Proves that Kannon had 1 banana last night given the conditions of the problem -/
theorem kannon_bananas : 
  ∀ (bananas_last_night : ℕ),
    (3 + bananas_last_night + 4) +  -- fruits last night
    ((3 + 4) + 10 * bananas_last_night + 2 * (3 + 4)) = 39 → -- fruits today
    bananas_last_night = 1 := by
  sorry

end kannon_bananas_l791_79115


namespace range_of_a_l791_79197

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x - 3 ≤ 0) ↔ -3 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l791_79197


namespace total_regular_games_count_l791_79145

def num_teams : ℕ := 15
def top_teams : ℕ := 5
def mid_teams : ℕ := 5
def bottom_teams : ℕ := 5

def top_vs_top_games : ℕ := 12
def top_vs_others_games : ℕ := 8
def mid_vs_mid_games : ℕ := 10
def mid_vs_top_games : ℕ := 6
def bottom_vs_bottom_games : ℕ := 8

def combinations (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem total_regular_games_count : 
  (combinations top_teams 2 * top_vs_top_games + 
   top_teams * (num_teams - top_teams) * top_vs_others_games +
   combinations mid_teams 2 * mid_vs_mid_games + 
   mid_teams * top_teams * mid_vs_top_games +
   combinations bottom_teams 2 * bottom_vs_bottom_games) = 850 := by
  sorry

end total_regular_games_count_l791_79145


namespace f_monotone_decreasing_f_min_on_interval_f_max_on_interval_l791_79111

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x - 2

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Theorem for monotonically decreasing intervals
theorem f_monotone_decreasing :
  (∀ x < -1, (f' x) < 0) ∧ (∀ x > 3, (f' x) < 0) :=
sorry

-- Theorem for minimum value on [-2, 2]
theorem f_min_on_interval :
  ∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f x ≤ f y ∧ f x = -7 :=
sorry

-- Theorem for maximum value on [-2, 2]
theorem f_max_on_interval :
  ∃ x ∈ Set.Icc (-2) 2, ∀ y ∈ Set.Icc (-2) 2, f y ≤ f x ∧ f x = 20 :=
sorry

end f_monotone_decreasing_f_min_on_interval_f_max_on_interval_l791_79111


namespace sufficient_not_necessary_condition_l791_79106

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 5 ∧ 
  ¬(a ≥ 5 → ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) :=
by sorry

end sufficient_not_necessary_condition_l791_79106


namespace vector_difference_magnitude_l791_79140

-- Define the vectors
def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ x ↦ (x, 4)

-- Define the parallel condition
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

-- State the theorem
theorem vector_difference_magnitude :
  ∃ x : ℝ, parallel a (b x) ∧ 
    Real.sqrt ((a.1 - (b x).1)^2 + (a.2 - (b x).2)^2) = 3 * Real.sqrt 5 :=
by sorry

end vector_difference_magnitude_l791_79140


namespace bobs_muffin_cost_l791_79132

/-- The cost of a single muffin for Bob -/
def muffin_cost (muffins_per_day : ℕ) (days_per_week : ℕ) (selling_price : ℚ) (weekly_profit : ℚ) : ℚ :=
  let total_muffins : ℕ := muffins_per_day * days_per_week
  let total_revenue : ℚ := (total_muffins : ℚ) * selling_price
  let total_cost : ℚ := total_revenue - weekly_profit
  total_cost / (total_muffins : ℚ)

/-- Theorem stating that Bob's muffin cost is $0.75 -/
theorem bobs_muffin_cost :
  muffin_cost 12 7 (3/2) 63 = 3/4 := by
  sorry

end bobs_muffin_cost_l791_79132


namespace central_angle_is_45_degrees_l791_79105

/-- Represents a circular dartboard divided into sectors -/
structure Dartboard where
  smallSectors : Nat
  largeSectors : Nat
  smallSectorProbability : ℝ

/-- Calculate the central angle of a smaller sector in degrees -/
def centralAngleSmallSector (d : Dartboard) : ℝ :=
  360 * d.smallSectorProbability

/-- Theorem: The central angle of a smaller sector is 45° for the given dartboard -/
theorem central_angle_is_45_degrees (d : Dartboard) 
  (h1 : d.smallSectors = 3)
  (h2 : d.largeSectors = 1)
  (h3 : d.smallSectorProbability = 1/8) :
  centralAngleSmallSector d = 45 := by
  sorry

end central_angle_is_45_degrees_l791_79105


namespace binary_to_octal_conversion_l791_79108

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents the given binary number 1011001₍₂₎ -/
def binary_number : List Bool := [true, false, false, true, true, false, true]

/-- The octal number we want to prove equality with -/
def octal_number : ℕ := 131

theorem binary_to_octal_conversion :
  binary_to_decimal binary_number = octal_number := by
  sorry

end binary_to_octal_conversion_l791_79108


namespace circle_c_equation_l791_79184

/-- A circle C with center in the first quadrant, satisfying specific conditions -/
structure CircleC where
  a : ℝ
  b : ℝ
  r : ℝ
  center_in_first_quadrant : a > 0 ∧ b > 0
  y_axis_chord : 2 * (r^2 - a^2).sqrt = 2
  x_axis_chord : 2 * (r^2 - b^2).sqrt = 4
  arc_length_ratio : (3 : ℝ) / 4 * 2 * Real.pi * r = 3 * ((1 : ℝ) / 4 * 2 * Real.pi * r)

/-- The equation of circle C is (x-√7)² + (y-2)² = 8 -/
theorem circle_c_equation (c : CircleC) : 
  c.a = Real.sqrt 7 ∧ c.b = 2 ∧ c.r = 2 * Real.sqrt 2 :=
sorry

end circle_c_equation_l791_79184


namespace walking_rate_ratio_l791_79119

theorem walking_rate_ratio (usual_time new_time : ℝ) 
  (h1 : usual_time = 16)
  (h2 : new_time = usual_time - 4) :
  new_time / usual_time = 3 / 4 :=
by sorry

end walking_rate_ratio_l791_79119


namespace line_slope_equals_y_coord_l791_79131

/-- Given a line passing through points (-1, -4) and (4, y), 
    if the slope of the line is equal to y, then y = 1. -/
theorem line_slope_equals_y_coord (y : ℝ) : 
  (y - (-4)) / (4 - (-1)) = y → y = 1 := by
  sorry

end line_slope_equals_y_coord_l791_79131


namespace firewood_sacks_filled_l791_79165

/-- Calculates the number of sacks filled with firewood -/
def sacks_filled (wood_per_sack : ℕ) (total_wood : ℕ) : ℕ :=
  total_wood / wood_per_sack

/-- Theorem stating that the number of sacks filled is 4 -/
theorem firewood_sacks_filled :
  let wood_per_sack : ℕ := 20
  let total_wood : ℕ := 80
  sacks_filled wood_per_sack total_wood = 4 := by
  sorry

end firewood_sacks_filled_l791_79165


namespace floor_sum_example_l791_79117

theorem floor_sum_example : ⌊(24.8 : ℝ)⌋ + ⌊(-24.8 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_example_l791_79117


namespace area_between_concentric_circles_l791_79156

theorem area_between_concentric_circles :
  ∀ (r : ℝ),
  r > 0 →
  3 * r - r = 3 →
  π * (3 * r)^2 - π * r^2 = 18 * π := by
sorry

end area_between_concentric_circles_l791_79156


namespace football_players_count_l791_79171

theorem football_players_count (total : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 40)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 11) :
  ∃ football : ℕ, football = 26 ∧ 
    football + tennis - both = total - neither :=
by sorry

end football_players_count_l791_79171


namespace sphere_volume_increase_l791_79159

theorem sphere_volume_increase (r : ℝ) (h : r > 0) :
  let V (radius : ℝ) := (4 / 3) * Real.pi * radius ^ 3
  V (2 * r) = 8 * V r :=
by sorry

end sphere_volume_increase_l791_79159


namespace unbounded_fraction_value_l791_79174

theorem unbounded_fraction_value (M : ℝ) :
  ∃ (x y : ℝ), -3 ≤ x ∧ x ≤ 1 ∧ x ≠ 0 ∧ 1 ≤ y ∧ y ≤ 3 ∧ (x + y + 1) / x > M :=
by sorry

end unbounded_fraction_value_l791_79174


namespace average_book_width_l791_79109

/-- The average width of 7 books with given widths is 4.5 cm -/
theorem average_book_width : 
  let book_widths : List ℝ := [5, 3/4, 1.5, 3, 12, 2, 7.5]
  (book_widths.sum / book_widths.length : ℝ) = 4.5 := by
  sorry

end average_book_width_l791_79109


namespace sin_arccos_eight_seventeenths_l791_79196

theorem sin_arccos_eight_seventeenths : 
  Real.sin (Real.arccos (8 / 17)) = 15 / 17 := by
  sorry

end sin_arccos_eight_seventeenths_l791_79196


namespace quadratic_root_implies_q_l791_79123

theorem quadratic_root_implies_q (p q : ℝ) : 
  (∃ (x : ℂ), 3 * x^2 + p * x + q = 0 ∧ x = 3 + 4*I) → q = 75 := by
  sorry

end quadratic_root_implies_q_l791_79123


namespace distinct_triangles_in_grid_l791_79142

/-- The number of points in each row or column of the grid -/
def grid_size : ℕ := 3

/-- The total number of points in the grid -/
def total_points : ℕ := grid_size * grid_size

/-- The number of collinear cases (rows + columns + diagonals) -/
def collinear_cases : ℕ := 2 * grid_size + 2

/-- Calculates the number of combinations of k items from n items -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of distinct triangles in a 3x3 grid -/
theorem distinct_triangles_in_grid :
  combinations total_points 3 - collinear_cases = 76 := by
  sorry

end distinct_triangles_in_grid_l791_79142


namespace min_value_expression_l791_79127

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hbc : b + c = 1) :
  (8 * a * c^2 + a) / (b * c) + 32 / (a + 1) ≥ 24 := by
  sorry

end min_value_expression_l791_79127


namespace bandi_has_winning_strategy_l791_79137

/-- Represents a player in the game -/
inductive Player
| Andi
| Bandi

/-- Represents a digit in the binary number -/
inductive Digit
| Zero
| One

/-- Represents a strategy for a player -/
def Strategy := List Digit → Digit

/-- The game state -/
structure GameState :=
(sequence : List Digit)
(turn : Player)
(moves_left : Nat)

/-- The result of the game -/
inductive GameResult
| AndiWin
| BandiWin

/-- Converts a list of digits to a natural number -/
def binary_to_nat (digits : List Digit) : Nat :=
  sorry

/-- Checks if a number is the sum of two squares -/
def is_sum_of_squares (n : Nat) : Prop :=
  sorry

/-- Plays the game given strategies for both players -/
def play_game (andi_strategy : Strategy) (bandi_strategy : Strategy) : GameResult :=
  sorry

/-- Theorem stating that Bandi has a winning strategy -/
theorem bandi_has_winning_strategy :
  ∃ (bandi_strategy : Strategy),
    ∀ (andi_strategy : Strategy),
      play_game andi_strategy bandi_strategy = GameResult.BandiWin :=
sorry

end bandi_has_winning_strategy_l791_79137


namespace reporter_wrong_l791_79169

/-- Represents a round-robin chess tournament --/
structure ChessTournament where
  num_players : ℕ
  wins : Fin num_players → ℕ
  draws : Fin num_players → ℕ
  losses : Fin num_players → ℕ

/-- The total number of games in a round-robin tournament --/
def total_games (t : ChessTournament) : ℕ :=
  t.num_players * (t.num_players - 1) / 2

/-- The total points scored in the tournament --/
def total_points (t : ChessTournament) : ℕ :=
  2 * total_games t

/-- Theorem stating that it's impossible for each player to have won as many games as they drew --/
theorem reporter_wrong (t : ChessTournament) (h1 : t.num_players = 20) 
    (h2 : ∀ i, t.wins i = t.draws i) : False := by
  sorry


end reporter_wrong_l791_79169


namespace exist_three_points_in_small_circle_l791_79129

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a set of points within a unit square -/
def PointsInUnitSquare (points : Set Point) : Prop :=
  ∀ p ∈ points, 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1

/-- Checks if three points can be enclosed by a circle with radius 1/7 -/
def CanBeEnclosedByCircle (p1 p2 p3 : Point) : Prop :=
  ∃ (center : Point), (center.x - p1.x)^2 + (center.y - p1.y)^2 ≤ (1/7)^2 ∧
                      (center.x - p2.x)^2 + (center.y - p2.y)^2 ≤ (1/7)^2 ∧
                      (center.x - p3.x)^2 + (center.y - p3.y)^2 ≤ (1/7)^2

/-- Main theorem: In any set of 51 points within a unit square, 
    there exist three points that can be enclosed by a circle with radius 1/7 -/
theorem exist_three_points_in_small_circle 
  (points : Set Point) 
  (h1 : PointsInUnitSquare points) 
  (h2 : Fintype points) 
  (h3 : Fintype.card points = 51) :
  ∃ p1 p2 p3 : Point, p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
               p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
               CanBeEnclosedByCircle p1 p2 p3 :=
by sorry

end exist_three_points_in_small_circle_l791_79129


namespace sum_remainder_six_l791_79107

theorem sum_remainder_six (m : ℤ) : (9 - m + (m + 5)) % 8 = 6 := by
  sorry

end sum_remainder_six_l791_79107


namespace sum_of_integers_l791_79167

theorem sum_of_integers (a b c d : ℤ) 
  (eq1 : 2 * (a - b + c) = 10)
  (eq2 : 2 * (b - c + d) = 12)
  (eq3 : 2 * (c - d + a) = 6)
  (eq4 : 2 * (d - a + b) = 4) :
  a + b + c + d = 8 := by
  sorry

end sum_of_integers_l791_79167


namespace clarence_oranges_l791_79150

/-- The total number of oranges Clarence has -/
def total_oranges (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Clarence has 8 oranges in total -/
theorem clarence_oranges :
  total_oranges 5 3 = 8 := by
  sorry

end clarence_oranges_l791_79150


namespace Q_equals_set_l791_79155

def P : Set ℕ := {1, 2}

def Q : Set ℕ := {z | ∃ x y, x ∈ P ∧ y ∈ P ∧ z = x + y}

theorem Q_equals_set : Q = {2, 3, 4} := by sorry

end Q_equals_set_l791_79155


namespace mole_fractions_C4H8O2_l791_79186

/-- Represents a chemical compound with counts of carbon, hydrogen, and oxygen atoms -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the total number of atoms in a compound -/
def totalAtoms (c : Compound) : ℕ := c.carbon + c.hydrogen + c.oxygen

/-- Calculates the mole fraction of an element in a compound -/
def moleFraction (elementCount : ℕ) (c : Compound) : ℚ :=
  elementCount / (totalAtoms c)

/-- The compound C4H8O2 -/
def C4H8O2 : Compound := ⟨4, 8, 2⟩

theorem mole_fractions_C4H8O2 :
  moleFraction C4H8O2.carbon C4H8O2 = 2/7 ∧
  moleFraction C4H8O2.hydrogen C4H8O2 = 4/7 ∧
  moleFraction C4H8O2.oxygen C4H8O2 = 1/7 := by
  sorry


end mole_fractions_C4H8O2_l791_79186


namespace fourth_number_11th_row_l791_79141

/-- The last number in row i of the pattern -/
def lastNumber (i : ℕ) : ℕ := 5 * i

/-- The fourth number in row i of the pattern -/
def fourthNumber (i : ℕ) : ℕ := lastNumber i - 1

/-- Theorem: The fourth number in the 11th row is 54 -/
theorem fourth_number_11th_row :
  fourthNumber 11 = 54 := by sorry

end fourth_number_11th_row_l791_79141


namespace quadratic_function_properties_l791_79100

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem quadratic_function_properties :
  (∃ x, f x = 1 ∧ ∀ y, f y ≥ f x) ∧
  f 0 = 3 ∧ f 2 = 3 ∧
  (∀ a : ℝ, (0 < a ∧ a < 1/2) ↔ 
    ¬(∀ x y : ℝ, 2*a ≤ x ∧ x < y ∧ y ≤ a+1 → f x < f y ∨ f x > f y)) ∧
  (∀ m : ℝ, m < 1 ↔ ∀ x : ℝ, -3 ≤ x ∧ x ≤ 0 → f x > 2*x + 2*m + 1) :=
by sorry

end quadratic_function_properties_l791_79100


namespace total_shells_is_195_l791_79128

/-- The total number of conch shells owned by David, Mia, Ava, and Alice -/
def total_shells (david_shells : ℕ) : ℕ :=
  let mia_shells := 4 * david_shells
  let ava_shells := mia_shells + 20
  let alice_shells := ava_shells / 2
  david_shells + mia_shells + ava_shells + alice_shells

/-- Theorem stating that the total number of shells is 195 when David has 15 shells -/
theorem total_shells_is_195 : total_shells 15 = 195 := by
  sorry

end total_shells_is_195_l791_79128


namespace a_value_is_two_l791_79130

/-- Represents the chemical reaction 3A + B ⇌ aC + 2D -/
structure Reaction where
  a : ℕ

/-- Represents the reaction conditions -/
structure ReactionConditions where
  initial_A : ℝ
  initial_B : ℝ
  volume : ℝ
  time : ℝ
  final_C : ℝ
  rate_D : ℝ

/-- Determines the value of 'a' in the reaction equation -/
def determine_a (reaction : Reaction) (conditions : ReactionConditions) : ℕ :=
  sorry

/-- Theorem stating that the value of 'a' is 2 given the specified conditions -/
theorem a_value_is_two :
  ∀ (reaction : Reaction) (conditions : ReactionConditions),
    conditions.initial_A = 0.6 ∧
    conditions.initial_B = 0.5 ∧
    conditions.volume = 0.4 ∧
    conditions.time = 5 ∧
    conditions.final_C = 0.2 ∧
    conditions.rate_D = 0.1 →
    determine_a reaction conditions = 2 :=
  sorry

end a_value_is_two_l791_79130


namespace quadratic_inequality_range_l791_79191

theorem quadratic_inequality_range (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*m*x + 1 ≥ 0) ↔ -1 ≤ m ∧ m ≤ 1 := by
  sorry

end quadratic_inequality_range_l791_79191


namespace alphabet_composition_l791_79193

/-- Represents an alphabet with letters containing dots and/or straight lines -/
structure Alphabet where
  total : ℕ
  only_line : ℕ
  only_dot : ℕ
  both : ℕ
  all_accounted : total = only_line + only_dot + both

/-- Theorem: In an alphabet of 40 letters, if 24 contain only a straight line
    and 5 contain only a dot, then 11 must contain both -/
theorem alphabet_composition (a : Alphabet)
  (h1 : a.total = 40)
  (h2 : a.only_line = 24)
  (h3 : a.only_dot = 5) :
  a.both = 11 := by
  sorry

end alphabet_composition_l791_79193


namespace knockout_tournament_matches_l791_79198

/-- The number of matches in a knockout tournament -/
def num_matches (n : ℕ) : ℕ := n - 1

/-- A knockout tournament with 64 players -/
def tournament_size : ℕ := 64

theorem knockout_tournament_matches :
  num_matches tournament_size = 63 := by
  sorry

end knockout_tournament_matches_l791_79198


namespace angle_of_inclination_30_degrees_l791_79120

theorem angle_of_inclination_30_degrees (x y : ℝ) :
  2 * x - 2 * Real.sqrt 3 * y + 1 = 0 →
  Real.arctan (Real.sqrt 3 / 3) = 30 * π / 180 :=
by sorry

end angle_of_inclination_30_degrees_l791_79120


namespace arrangements_count_l791_79151

/-- The number of arrangements of 6 people with specific conditions -/
def num_arrangements : ℕ :=
  let total_people : ℕ := 6
  let num_teachers : ℕ := 1
  let num_male_students : ℕ := 2
  let num_female_students : ℕ := 3
  let male_students_arrangements : ℕ := 2  -- A_{2}^{2}
  let female_adjacent_pair_selections : ℕ := 3  -- C_{3}^{2}
  let remaining_people_arrangements : ℕ := 12  -- A_{3}^{3}
  male_students_arrangements * female_adjacent_pair_selections * remaining_people_arrangements

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count : num_arrangements = 24 := by
  sorry

end arrangements_count_l791_79151


namespace kaleb_second_half_score_l791_79160

/-- 
Given that Kaleb scored 43 points in the first half of a trivia game and 66 points in total,
this theorem proves that he scored 23 points in the second half.
-/
theorem kaleb_second_half_score 
  (first_half_score : ℕ) 
  (total_score : ℕ) 
  (h1 : first_half_score = 43)
  (h2 : total_score = 66) :
  total_score - first_half_score = 23 := by
  sorry

end kaleb_second_half_score_l791_79160


namespace min_disks_theorem_l791_79101

/-- The number of labels -/
def n : ℕ := 60

/-- The minimum number of disks with the same label we want to guarantee -/
def k : ℕ := 12

/-- The sum of arithmetic sequence from 1 to m -/
def sum_to (m : ℕ) : ℕ := m * (m + 1) / 2

/-- The total number of disks -/
def total_disks : ℕ := sum_to n

/-- The function to calculate the minimum number of disks to draw -/
def min_disks_to_draw : ℕ := sum_to (k - 1) + (n - (k - 1)) * (k - 1) + 1

/-- The theorem stating the minimum number of disks to draw -/
theorem min_disks_theorem : min_disks_to_draw = 606 := by sorry

end min_disks_theorem_l791_79101


namespace complex_fraction_squared_l791_79194

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_squared : (2 * i / (1 + i)) ^ 2 = 2 * i :=
by sorry

end complex_fraction_squared_l791_79194


namespace f_of_3_equals_13_l791_79144

theorem f_of_3_equals_13 (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = 2 * x + 5) : f 3 = 13 := by
  sorry

end f_of_3_equals_13_l791_79144


namespace inequality_one_inequality_two_l791_79148

-- Statement 1
theorem inequality_one (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 2) - Real.sqrt (a + 6) > Real.sqrt a - Real.sqrt (a + 4) := by
  sorry

-- Statement 2
theorem inequality_two (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  Real.sqrt (a + 1/2) + Real.sqrt (b + 1/2) ≤ 2 := by
  sorry

end inequality_one_inequality_two_l791_79148


namespace age_when_billy_was_born_l791_79153

/-- Proves the age when Billy was born given the current ages -/
theorem age_when_billy_was_born
  (my_current_age billy_current_age : ℕ)
  (h1 : my_current_age = 4 * billy_current_age)
  (h2 : billy_current_age = 4)
  : my_current_age - billy_current_age = my_current_age - billy_current_age :=
by sorry

end age_when_billy_was_born_l791_79153


namespace telescope_purchase_sum_l791_79103

/-- The sum of Joan and Karl's telescope purchases -/
def sum_of_purchases (joan_price karl_price : ℕ) : ℕ :=
  joan_price + karl_price

/-- Theorem stating the sum of Joan and Karl's telescope purchases -/
theorem telescope_purchase_sum :
  ∀ (joan_price karl_price : ℕ),
    joan_price = 158 →
    2 * joan_price = karl_price + 74 →
    sum_of_purchases joan_price karl_price = 400 := by
  sorry

end telescope_purchase_sum_l791_79103


namespace only_C_nonlinear_l791_79166

-- Define the structure for a system of two equations
structure SystemOfEquations where
  eq1 : ℝ → ℝ → ℝ
  eq2 : ℝ → ℝ → ℝ

-- Define the systems A, B, C, and D
def systemA : SystemOfEquations := ⟨λ x y => x - 2, λ x y => y - 3⟩
def systemB : SystemOfEquations := ⟨λ x y => x + y - 1, λ x y => x - y - 2⟩
def systemC : SystemOfEquations := ⟨λ x y => x + y - 5, λ x y => x * y - 1⟩
def systemD : SystemOfEquations := ⟨λ x y => y - x, λ x y => x - 2*y - 1⟩

-- Define what it means for an equation to be linear
def isLinear (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x y : ℝ, f x y = a * x + b * y + c

-- Define what it means for a system to be linear
def isLinearSystem (s : SystemOfEquations) : Prop :=
  isLinear s.eq1 ∧ isLinear s.eq2

-- Theorem statement
theorem only_C_nonlinear :
  isLinearSystem systemA ∧
  isLinearSystem systemB ∧
  ¬isLinearSystem systemC ∧
  isLinearSystem systemD :=
sorry

end only_C_nonlinear_l791_79166


namespace opposite_of_negative_three_l791_79135

theorem opposite_of_negative_three : 
  ∃ x : ℤ, x + (-3) = 0 ∧ x = 3 := by
  sorry

end opposite_of_negative_three_l791_79135
