import Mathlib

namespace voldemort_calorie_limit_l2872_287258

/-- Voldemort's daily calorie intake limit -/
def daily_calorie_limit : ℕ := by sorry

/-- Calories from breakfast -/
def breakfast_calories : ℕ := 560

/-- Calories from lunch -/
def lunch_calories : ℕ := 780

/-- Calories from dinner -/
def dinner_calories : ℕ := 110 + 310 + 215

/-- Remaining calories Voldemort can still take -/
def remaining_calories : ℕ := 525

/-- Theorem stating Voldemort's daily calorie intake limit -/
theorem voldemort_calorie_limit :
  daily_calorie_limit = breakfast_calories + lunch_calories + dinner_calories + remaining_calories := by
  sorry

end voldemort_calorie_limit_l2872_287258


namespace f_2009_equals_3_l2872_287260

/-- Given a function f and constants a, b, α, β, prove that f(2009) = 3 -/
theorem f_2009_equals_3 
  (f : ℝ → ℝ) 
  (a b α β : ℝ) 
  (h1 : ∀ x, f x = a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4)
  (h2 : f 2000 = 5) :
  f 2009 = 3 := by
  sorry

end f_2009_equals_3_l2872_287260


namespace ricardo_coin_difference_l2872_287230

/-- The number of coins Ricardo has -/
def total_coins : ℕ := 1980

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of pennies Ricardo has -/
def num_pennies : ℕ → ℕ := λ p => p

/-- The number of dimes Ricardo has -/
def num_dimes : ℕ → ℕ := λ p => total_coins - p

/-- The total value of Ricardo's coins in cents -/
def total_value : ℕ → ℕ := λ p => penny_value * num_pennies p + dime_value * num_dimes p

/-- The maximum possible value of Ricardo's coins in cents -/
def max_value : ℕ := total_value 1

/-- The minimum possible value of Ricardo's coins in cents -/
def min_value : ℕ := total_value (total_coins - 1)

theorem ricardo_coin_difference :
  max_value - min_value = 17802 :=
sorry

end ricardo_coin_difference_l2872_287230


namespace skittles_eaten_l2872_287216

/-- Proves that the number of Skittles eaten is the difference between initial and final amounts --/
theorem skittles_eaten (initial_skittles final_skittles : ℝ) (oranges_bought : ℝ) :
  initial_skittles = 7 →
  final_skittles = 2 →
  oranges_bought = 18 →
  initial_skittles - final_skittles = 5 := by
  sorry

end skittles_eaten_l2872_287216


namespace christmas_tree_lights_l2872_287273

theorem christmas_tree_lights (total : ℕ) (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 95 → red = 26 → yellow = 37 → blue = total - red - yellow → blue = 32 := by
  sorry

end christmas_tree_lights_l2872_287273


namespace teacher_class_choices_l2872_287283

theorem teacher_class_choices (n_teachers : ℕ) (n_classes : ℕ) : 
  n_teachers = 5 → n_classes = 4 → (n_classes : ℕ) ^ n_teachers = 4^5 := by
  sorry

end teacher_class_choices_l2872_287283


namespace marys_number_proof_l2872_287217

theorem marys_number_proof : ∃! x : ℕ, 
  10 ≤ x ∧ x < 100 ∧ 
  (∃ a b : ℕ, 
    3 * x + 11 = 10 * a + b ∧
    10 * b + a ≥ 71 ∧ 
    10 * b + a ≤ 75) ∧
  x = 12 := by
sorry

end marys_number_proof_l2872_287217


namespace unique_positive_zero_implies_a_range_l2872_287228

/-- The function f(x) = ax^3 - 3x^2 + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 6 * x

theorem unique_positive_zero_implies_a_range (a : ℝ) :
  (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) → a < -2 :=
by sorry

end unique_positive_zero_implies_a_range_l2872_287228


namespace joe_needs_twelve_more_cars_l2872_287227

/-- Given that Joe has 50 toy cars initially and will have 62 cars after getting more,
    prove that he needs to get 12 more toy cars. -/
theorem joe_needs_twelve_more_cars 
  (initial_cars : ℕ) 
  (final_cars : ℕ) 
  (h1 : initial_cars = 50) 
  (h2 : final_cars = 62) : 
  final_cars - initial_cars = 12 := by
  sorry

end joe_needs_twelve_more_cars_l2872_287227


namespace probability_two_nondefective_pens_l2872_287218

/-- Given a box of 8 pens with 3 defective pens, the probability of selecting 2 non-defective pens without replacement is 5/14. -/
theorem probability_two_nondefective_pens (total_pens : Nat) (defective_pens : Nat) 
  (h1 : total_pens = 8) (h2 : defective_pens = 3) :
  let nondefective_pens := total_pens - defective_pens
  let prob_first := nondefective_pens / total_pens
  let prob_second := (nondefective_pens - 1) / (total_pens - 1)
  prob_first * prob_second = 5 / 14 := by
  sorry

end probability_two_nondefective_pens_l2872_287218


namespace home_run_multiple_l2872_287289

/-- The multiple of Dave Winfield's home runs that Hank Aaron's home runs are compared to -/
theorem home_run_multiple (aaron_hr winfield_hr : ℕ) (difference : ℕ) : 
  aaron_hr = 755 →
  winfield_hr = 465 →
  aaron_hr + difference = 2 * winfield_hr →
  2 = (aaron_hr + difference) / winfield_hr :=
by
  sorry

end home_run_multiple_l2872_287289


namespace ratio_problem_l2872_287204

theorem ratio_problem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (h5 : a / b = 1 / 4)
  (h6 : c / d = 5 / 13)
  (h7 : a / d = 0.1388888888888889) :
  b / c = 13 / 9 := by
  sorry

end ratio_problem_l2872_287204


namespace profit_maximizing_price_l2872_287278

/-- Represents the selling price of the product -/
def selling_price : ℝ → ℝ := id

/-- Represents the purchase cost of the product -/
def purchase_cost : ℝ := 40

/-- Represents the number of units sold at a given price -/
def units_sold (x : ℝ) : ℝ := 500 - 20 * (x - 50)

/-- Represents the profit at a given selling price -/
def profit (x : ℝ) : ℝ := (x - purchase_cost) * (units_sold x)

/-- Theorem stating that the profit-maximizing selling price is 57.5 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), ∀ (y : ℝ), profit x ≥ profit y ∧ x = 57.5 := by sorry

end profit_maximizing_price_l2872_287278


namespace floor_of_expression_equals_eight_l2872_287256

theorem floor_of_expression_equals_eight :
  ⌊(2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023)⌋ = 8 := by
  sorry

end floor_of_expression_equals_eight_l2872_287256


namespace sculpture_surface_area_l2872_287237

/-- Represents a cube sculpture with three layers --/
structure CubeSculpture where
  topLayer : ℕ
  middleLayer : ℕ
  bottomLayer : ℕ
  totalCubes : ℕ
  (total_eq : totalCubes = topLayer + middleLayer + bottomLayer)

/-- Calculates the exposed surface area of the sculpture --/
def exposedSurfaceArea (s : CubeSculpture) : ℕ :=
  5 * s.topLayer +  -- Top cube: 5 sides exposed
  (5 + 4 * 4) +     -- Middle layer: 1 cube with 5 sides, 4 cubes with 4 sides
  s.bottomLayer     -- Bottom layer: only top faces exposed

/-- The main theorem to prove --/
theorem sculpture_surface_area :
  ∃ (s : CubeSculpture),
    s.topLayer = 1 ∧
    s.middleLayer = 5 ∧
    s.bottomLayer = 11 ∧
    s.totalCubes = 17 ∧
    exposedSurfaceArea s = 37 :=
  sorry

end sculpture_surface_area_l2872_287237


namespace polynomial_factorization_l2872_287233

theorem polynomial_factorization (x : ℝ) : 2 * x^2 - 2 = 2 * (x + 1) * (x - 1) := by
  sorry

end polynomial_factorization_l2872_287233


namespace boys_to_girls_ratio_l2872_287294

theorem boys_to_girls_ratio (total_students girls : ℕ) (h1 : total_students = 780) (h2 : girls = 300) :
  (total_students - girls) / girls = 8 / 5 := by
  sorry

end boys_to_girls_ratio_l2872_287294


namespace ceiling_neg_sqrt_64_over_9_l2872_287261

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end ceiling_neg_sqrt_64_over_9_l2872_287261


namespace rational_function_property_l2872_287265

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ  -- Linear function
  q : ℝ → ℝ  -- Quadratic function
  linear_p : ∃ a b : ℝ, ∀ x, p x = a * x + b
  quadratic_q : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  asymptote_minus_four : q (-4) = 0
  asymptote_one : q 1 = 0
  passes_through_point : p 1 / q 1 = 2

theorem rational_function_property (f : RationalFunction) : f.p 0 / f.q 0 = 0 := by
  sorry

end rational_function_property_l2872_287265


namespace line_passes_through_circle_center_l2872_287201

/-- The center of a circle given by the equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop :=
  3 * x + y + a = 0

/-- The circle equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- Theorem: If the line 3x + y + a = 0 passes through the center of the circle
    x^2 + y^2 + 2x - 4y = 0, then a = 1 -/
theorem line_passes_through_circle_center (a : ℝ) :
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end line_passes_through_circle_center_l2872_287201


namespace gcd_60_75_l2872_287291

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l2872_287291


namespace concurrent_lines_theorem_l2872_287205

/-- A line that intersects opposite sides of a square -/
structure DividingLine where
  divides_square : Bool
  area_ratio : Rat
  intersects_opposite_sides : Bool

/-- A configuration of lines dividing a square -/
structure SquareDivision where
  lines : Finset DividingLine
  square : Set (ℝ × ℝ)

/-- The number of concurrent lines in a square division -/
def num_concurrent (sd : SquareDivision) : ℕ := sorry

theorem concurrent_lines_theorem (sd : SquareDivision) 
  (h1 : sd.lines.card = 2005)
  (h2 : ∀ l ∈ sd.lines, l.divides_square)
  (h3 : ∀ l ∈ sd.lines, l.area_ratio = 2 / 3)
  (h4 : ∀ l ∈ sd.lines, l.intersects_opposite_sides) :
  num_concurrent sd ≥ 502 := by sorry

end concurrent_lines_theorem_l2872_287205


namespace sue_initial_savings_proof_l2872_287247

/-- The cost of the perfume in dollars -/
def perfume_cost : ℝ := 50

/-- Christian's initial savings in dollars -/
def christian_initial_savings : ℝ := 5

/-- Number of yards Christian mowed -/
def yards_mowed : ℕ := 4

/-- Cost per yard mowed in dollars -/
def cost_per_yard : ℝ := 5

/-- Number of dogs Sue walked -/
def dogs_walked : ℕ := 6

/-- Cost per dog walked in dollars -/
def cost_per_dog : ℝ := 2

/-- Additional amount needed in dollars -/
def additional_needed : ℝ := 6

/-- Sue's initial savings in dollars -/
def sue_initial_savings : ℝ := 7

theorem sue_initial_savings_proof :
  sue_initial_savings = 
    perfume_cost - 
    (christian_initial_savings + 
     (yards_mowed : ℝ) * cost_per_yard + 
     (dogs_walked : ℝ) * cost_per_dog) := by
  sorry

end sue_initial_savings_proof_l2872_287247


namespace tangent_line_to_cubic_l2872_287226

/-- The tangent line to a cubic curve at a specific point -/
theorem tangent_line_to_cubic (a k b : ℝ) :
  (∃ (x y : ℝ), x = 2 ∧ y = 3 ∧ 
    y = x^3 + a*x + 1 ∧ 
    y = k*x + b ∧ 
    (3 * x^2 + a) = k) →
  b = -15 := by
  sorry

end tangent_line_to_cubic_l2872_287226


namespace event_organization_ways_l2872_287253

def number_of_friends : ℕ := 5
def number_of_organizers : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem event_organization_ways :
  choose number_of_friends number_of_organizers = 10 := by
  sorry

end event_organization_ways_l2872_287253


namespace arithmetic_sequence_sum_l2872_287249

theorem arithmetic_sequence_sum : 
  let a₁ : ℤ := 1
  let aₙ : ℤ := 1996
  let n : ℕ := 96
  let s := n * (a₁ + aₙ) / 2
  s = 95856 := by
sorry

end arithmetic_sequence_sum_l2872_287249


namespace chris_win_probability_l2872_287279

theorem chris_win_probability :
  let chris_head_prob : ℝ := 1/4
  let drew_head_prob : ℝ := 1/3
  let both_tail_prob : ℝ := (1 - chris_head_prob) * (1 - drew_head_prob)
  chris_head_prob / (1 - both_tail_prob) = 1/2 :=
by sorry

end chris_win_probability_l2872_287279


namespace smaller_number_proof_l2872_287280

theorem smaller_number_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = 3 / 4) (h4 : a + b = 21) (h5 : max a b = 12) : 
  min a b = 9 :=
by
  sorry

end smaller_number_proof_l2872_287280


namespace max_value_is_b_l2872_287244

theorem max_value_is_b (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b = max (1/2) (max b (max (2*a*b) (a^2 + b^2))) := by
  sorry

end max_value_is_b_l2872_287244


namespace next_next_perfect_square_l2872_287284

theorem next_next_perfect_square (x : ℕ) (k : ℕ) (h : x = k^2) :
  (k + 2)^2 = x + 4 * Int.sqrt x + 4 :=
sorry

end next_next_perfect_square_l2872_287284


namespace chemistry_physics_score_difference_l2872_287232

theorem chemistry_physics_score_difference
  (math_score physics_score chemistry_score : ℕ)
  (total_math_physics : math_score + physics_score = 60)
  (avg_math_chemistry : (math_score + chemistry_score) / 2 = 40)
  (chemistry_higher : chemistry_score > physics_score) :
  chemistry_score - physics_score = 20 :=
by sorry

end chemistry_physics_score_difference_l2872_287232


namespace election_votes_l2872_287210

theorem election_votes (candidate_percentage : ℝ) (vote_difference : ℕ) (total_votes : ℕ) : 
  candidate_percentage = 35 / 100 → 
  vote_difference = 2100 →
  total_votes = (vote_difference : ℝ) / (1 - 2 * candidate_percentage) →
  total_votes = 7000 := by
sorry

end election_votes_l2872_287210


namespace hall_area_l2872_287272

/-- The area of a rectangular hall with specific proportions -/
theorem hall_area : 
  ∀ (length width : ℝ),
  width = (1/2) * length →
  length - width = 20 →
  length * width = 800 := by
sorry

end hall_area_l2872_287272


namespace unique_solution_abc_l2872_287266

theorem unique_solution_abc (a b c : ℝ) 
  (ha : a > 2) (hb : b > 2) (hc : c > 2)
  (heq : (a+1)^2 / (b+c-1) + (b+2)^2 / (c+a-3) + (c+3)^2 / (a+b-5) = 32) :
  a = 8 ∧ b = 6 ∧ c = 5 := by
sorry

end unique_solution_abc_l2872_287266


namespace roots_equation_result_l2872_287220

theorem roots_equation_result (γ δ : ℝ) : 
  γ^2 - 3*γ + 1 = 0 → δ^2 - 3*δ + 1 = 0 → 8*γ^3 + 15*δ^2 = 179 := by
  sorry

end roots_equation_result_l2872_287220


namespace perpendicular_line_equation_l2872_287229

/-- Given a line L1 with equation 3x + 4y + 5 = 0 and a point P (0, -3),
    prove that the line L2 passing through P and perpendicular to L1
    has the equation 4y - 3x - 3 = 0 -/
theorem perpendicular_line_equation 
  (L1 : Set (ℝ × ℝ)) 
  (P : ℝ × ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ L1 ↔ 3 * x + 4 * y + 5 = 0) →
  P = (0, -3) →
  ∃ (L2 : Set (ℝ × ℝ)),
    (∀ (x y : ℝ), (x, y) ∈ L2 ↔ 4 * y - 3 * x - 3 = 0) ∧
    P ∈ L2 ∧
    (∀ (v w : ℝ × ℝ), v ∈ L1 → w ∈ L1 → v ≠ w →
      ∀ (p q : ℝ × ℝ), p ∈ L2 → q ∈ L2 → p ≠ q →
        ((v.1 - w.1) * (p.1 - q.1) + (v.2 - w.2) * (p.2 - q.2) = 0)) :=
by sorry

end perpendicular_line_equation_l2872_287229


namespace min_abs_beta_plus_delta_l2872_287286

open Complex

theorem min_abs_beta_plus_delta :
  ∀ β δ : ℂ,
  let g : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^2 + β*z + δ
  (g 1).im = 0 →
  (g (-I)).im = 0 →
  ∃ (β' δ' : ℂ),
    let g' : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^2 + β'*z + δ'
    (g' 1).im = 0 ∧
    (g' (-I)).im = 0 ∧
    Complex.abs β' + Complex.abs δ' = 4 ∧
    ∀ (β'' δ'' : ℂ),
      let g'' : ℂ → ℂ := λ z ↦ (3 + 2*I)*z^2 + β''*z + δ''
      (g'' 1).im = 0 →
      (g'' (-I)).im = 0 →
      Complex.abs β'' + Complex.abs δ'' ≥ 4 :=
by sorry

end min_abs_beta_plus_delta_l2872_287286


namespace pi_half_irrational_l2872_287251

theorem pi_half_irrational : Irrational (Real.pi / 2) := by
  sorry

end pi_half_irrational_l2872_287251


namespace max_value_of_expression_l2872_287290

theorem max_value_of_expression (x : ℝ) (h : x > 1) :
  x + 1 / (x - 1) ≤ 3 :=
sorry

end max_value_of_expression_l2872_287290


namespace latest_90_degrees_time_l2872_287299

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

-- Define the theorem
theorem latest_90_degrees_time :
  ∃ t : ℝ, t ≤ 17 ∧ temperature t = 90 ∧
  ∀ s : ℝ, s > 17 → temperature s ≠ 90 :=
by sorry

end latest_90_degrees_time_l2872_287299


namespace tangent_line_at_one_range_of_m_l2872_287234

-- Define the function f(x)
noncomputable def f (x m : ℝ) : ℝ := (x - 1) * Real.log x - m * (x + 1)

-- Part 1: Tangent line equation
theorem tangent_line_at_one (m : ℝ) (h : m = 1) :
  ∃ (a b c : ℝ), a * x + b * y + c = 0 ∧
  (∀ x y : ℝ, y = f x m → (a * x + b * y + c = 0 ↔ x = 1)) :=
sorry

-- Part 2: Range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, x > 0 → f x m ≥ 0) ↔ m ≤ 0 :=
sorry

end tangent_line_at_one_range_of_m_l2872_287234


namespace track_length_is_480_l2872_287257

/-- Represents the circular track and the runners' movements --/
structure TrackSystem where
  trackLength : ℝ
  janetSpeed : ℝ
  leahSpeed : ℝ

/-- Conditions of the problem --/
def ProblemConditions (s : TrackSystem) : Prop :=
  s.janetSpeed > 0 ∧ 
  s.leahSpeed > 0 ∧ 
  120 / s.janetSpeed = (s.trackLength / 2 - 120) / s.leahSpeed ∧
  (s.trackLength / 2 - 40) / s.janetSpeed = 200 / s.leahSpeed

/-- The main theorem to prove --/
theorem track_length_is_480 (s : TrackSystem) : 
  ProblemConditions s → s.trackLength = 480 :=
sorry

end track_length_is_480_l2872_287257


namespace tan_alpha_2_implies_expression_5_l2872_287242

theorem tan_alpha_2_implies_expression_5 (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end tan_alpha_2_implies_expression_5_l2872_287242


namespace residue_of_negative_1235_mod_29_l2872_287277

theorem residue_of_negative_1235_mod_29 : 
  ∃ k : ℤ, -1235 = 29 * k + 12 ∧ 0 ≤ 12 ∧ 12 < 29 := by
  sorry

end residue_of_negative_1235_mod_29_l2872_287277


namespace reflection_across_x_axis_l2872_287288

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (-3, -4)

/-- The expected reflected point -/
def expected_reflection : ℝ × ℝ := (-3, 4)

theorem reflection_across_x_axis :
  reflect_x original_point = expected_reflection := by
  sorry

end reflection_across_x_axis_l2872_287288


namespace age_difference_l2872_287235

/-- Given Sierra's current age and Diaz's age 20 years from now, 
    prove that the difference between (10 times Diaz's current age minus 40) 
    and (10 times Sierra's current age) is 20. -/
theorem age_difference (sierra_age : ℕ) (diaz_future_age : ℕ) : 
  sierra_age = 30 → 
  diaz_future_age = 56 → 
  (10 * (diaz_future_age - 20) - 40) - (10 * sierra_age) = 20 :=
by sorry

end age_difference_l2872_287235


namespace square_roots_problem_l2872_287241

theorem square_roots_problem (x : ℝ) :
  (∃ (a : ℝ), a > 0 ∧ (x + 1)^2 = a ∧ (x - 5)^2 = a) → x = 2 := by
  sorry

end square_roots_problem_l2872_287241


namespace points_per_enemy_l2872_287269

/-- 
Given a video game level with the following conditions:
- There are 11 enemies in total
- Defeating all but 3 enemies results in 72 points
This theorem proves that the number of points earned for defeating one enemy is 9.
-/
theorem points_per_enemy (total_enemies : ℕ) (remaining_enemies : ℕ) (total_points : ℕ) :
  total_enemies = 11 →
  remaining_enemies = 3 →
  total_points = 72 →
  (total_points / (total_enemies - remaining_enemies) : ℚ) = 9 := by
  sorry

#check points_per_enemy

end points_per_enemy_l2872_287269


namespace teacher_assignment_ways_l2872_287231

/-- The number of ways to assign teachers to classes -/
def assignmentWays (n m : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose (n - 2) 2 * Nat.choose (n - 4) 2 * Nat.choose (n - 6) 2

/-- Theorem stating the number of ways to assign 4 teachers to 8 classes -/
theorem teacher_assignment_ways :
  assignmentWays 8 4 = 2520 :=
by sorry

#eval assignmentWays 8 4

end teacher_assignment_ways_l2872_287231


namespace B_power_150_is_identity_l2872_287238

def B : Matrix (Fin 3) (Fin 3) ℝ := !![0, 1, 0; 0, 0, 1; 1, 0, 0]

theorem B_power_150_is_identity :
  B^150 = (1 : Matrix (Fin 3) (Fin 3) ℝ) := by
  sorry

end B_power_150_is_identity_l2872_287238


namespace no_winning_strategy_l2872_287222

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (red : ℕ)
  (black : ℕ)
  (total : ℕ)
  (h_total : total = red + black)
  (h_standard : total = 52 ∧ red = 26 ∧ black = 26)

/-- Represents a strategy for playing the card game -/
def Strategy := Deck → Bool

/-- The probability of winning given a deck state and a strategy -/
noncomputable def win_probability (d : Deck) (s : Strategy) : ℝ :=
  d.red / d.total

/-- Theorem stating that no strategy can have a winning probability greater than 0.5 -/
theorem no_winning_strategy (s : Strategy) :
  ∀ d : Deck, win_probability d s ≤ 0.5 :=
sorry

end no_winning_strategy_l2872_287222


namespace shopkeeper_loss_per_metre_l2872_287223

/-- Calculates the loss per metre of cloth sold by a shopkeeper -/
theorem shopkeeper_loss_per_metre 
  (total_metres : ℕ) 
  (total_selling_price : ℕ) 
  (cost_price_per_metre : ℕ) 
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 36000)
  (h3 : cost_price_per_metre = 70) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 10 := by
  sorry

end shopkeeper_loss_per_metre_l2872_287223


namespace roots_of_equation_l2872_287209

theorem roots_of_equation (x : ℝ) : 
  (x - 3)^2 = 4 ↔ x = 5 ∨ x = 1 := by sorry

end roots_of_equation_l2872_287209


namespace proof_uses_synthetic_method_l2872_287215

-- Define the proof process as a string
def proofProcess : String := "cos 4θ - sin 4θ = (cos 2θ + sin 2θ) ⋅ (cos 2θ - sin 2θ) = cos 2θ - sin 2θ = cos 2θ"

-- Define the possible proof methods
inductive ProofMethod
| Analytical
| Synthetic
| Combined
| Indirect

-- Define a function to determine the proof method
def determineProofMethod (process : String) : ProofMethod := sorry

-- Theorem stating that the proof process uses the Synthetic Method
theorem proof_uses_synthetic_method : 
  determineProofMethod proofProcess = ProofMethod.Synthetic := sorry

end proof_uses_synthetic_method_l2872_287215


namespace shane_semester_distance_l2872_287270

/-- Calculates the total distance traveled for round trips during a semester -/
def total_semester_distance (daily_one_way_distance : ℕ) (semester_days : ℕ) : ℕ :=
  2 * daily_one_way_distance * semester_days

/-- Proves that the total distance traveled during the semester is 1600 miles -/
theorem shane_semester_distance :
  total_semester_distance 10 80 = 1600 := by
  sorry

end shane_semester_distance_l2872_287270


namespace diaries_calculation_l2872_287262

theorem diaries_calculation (initial_diaries : ℕ) : initial_diaries = 8 →
  (initial_diaries + 2 * initial_diaries) * 3 / 4 = 18 := by
  sorry

end diaries_calculation_l2872_287262


namespace hyperbola_vertex_distance_l2872_287255

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  4 * x^2 - 16 * x - 16 * y^2 + 32 * y + 144 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem hyperbola_vertex_distance :
  vertex_distance hyperbola_eq = 12 :=
sorry

end hyperbola_vertex_distance_l2872_287255


namespace college_cost_calculation_l2872_287285

/-- The total cost of Sabina's first year of college -/
def total_cost : ℝ := 30000

/-- Sabina's savings -/
def savings : ℝ := 10000

/-- The percentage of the remainder covered by the grant -/
def grant_percentage : ℝ := 0.40

/-- The amount of the loan Sabina needs -/
def loan_amount : ℝ := 12000

/-- Theorem stating that the total cost is correct given the conditions -/
theorem college_cost_calculation :
  total_cost = savings + grant_percentage * (total_cost - savings) + loan_amount := by
  sorry

end college_cost_calculation_l2872_287285


namespace percentage_error_calculation_l2872_287239

theorem percentage_error_calculation : 
  let incorrect_factor : ℚ := 3 / 5
  let correct_factor : ℚ := 5 / 3
  let ratio := incorrect_factor / correct_factor
  let error_percentage := (1 - ratio) * 100
  error_percentage = 64 := by
sorry

end percentage_error_calculation_l2872_287239


namespace h_over_g_equals_64_l2872_287295

theorem h_over_g_equals_64 (G H : ℤ) :
  (∀ x : ℝ, x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5 →
    G / (x + 3) + H / (x^2 - 5*x) = (x^2 - 3*x + 8) / (x^3 + x^2 - 15*x)) →
  (H : ℚ) / (G : ℚ) = 64 := by
sorry

end h_over_g_equals_64_l2872_287295


namespace train_length_l2872_287206

/-- Calculates the length of a train given its speed, time to pass a bridge, and the bridge length -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) : 
  speed = 45 → time = 36.8 → bridge_length = 140 → 
  (speed * 1000 / 3600) * time - bridge_length = 320 := by
  sorry

#check train_length

end train_length_l2872_287206


namespace total_weight_proof_l2872_287276

theorem total_weight_proof (jim_weight steve_weight stan_weight : ℕ) 
  (h1 : stan_weight = steve_weight + 5)
  (h2 : steve_weight = jim_weight - 8)
  (h3 : jim_weight = 110) : 
  jim_weight + steve_weight + stan_weight = 319 :=
by
  sorry

end total_weight_proof_l2872_287276


namespace ellipse_equation_l2872_287267

/-- An ellipse with the given properties has the standard equation x²/4 + y² = 1 -/
theorem ellipse_equation (a b c : ℝ) (h1 : a + b = c * Real.sqrt 3) 
  (h2 : c = Real.sqrt 3) : 
  ∃ (x y : ℝ), x^2 / 4 + y^2 = 1 :=
sorry

end ellipse_equation_l2872_287267


namespace exam_score_unique_solution_l2872_287203

theorem exam_score_unique_solution (n : ℕ) : 
  (∃ t : ℚ, t > 0 ∧ 
    15 * t + (1/3 : ℚ) * ((n : ℚ) - 20) * t = (1/2 : ℚ) * (n : ℚ) * t) → 
  n = 50 :=
by sorry

end exam_score_unique_solution_l2872_287203


namespace comparison_theorems_l2872_287281

theorem comparison_theorems :
  (∀ a : ℝ, a < 0 → a / (a - 1) > 0) ∧
  (∀ x : ℝ, x < -1 → 2 / (x^2 - 1) > (x - 1) / (x^2 - 2*x + 1)) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → 2*x*y / (x + y) < (x + y) / 2) :=
by sorry

end comparison_theorems_l2872_287281


namespace ages_sum_five_years_ago_l2872_287212

/-- Proves that the sum of Angela's and Beth's ages 5 years ago was 39 years -/
theorem ages_sum_five_years_ago : 
  ∀ (angela_age beth_age : ℕ),
  angela_age = 4 * beth_age →
  angela_age + 5 = 44 →
  (angela_age - 5) + (beth_age - 5) = 39 :=
by
  sorry

end ages_sum_five_years_ago_l2872_287212


namespace modulo_residue_sum_l2872_287211

theorem modulo_residue_sum : (255 + 7 * 51 + 9 * 187 + 5 * 34) % 17 = 0 := by
  sorry

end modulo_residue_sum_l2872_287211


namespace difference_of_squares_65_55_l2872_287296

theorem difference_of_squares_65_55 : 65^2 - 55^2 = 1200 := by
  sorry

end difference_of_squares_65_55_l2872_287296


namespace median_of_special_list_l2872_287259

def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def isMedian (m : ℕ) : Prop :=
  let N := sumOfSquares 100
  let leftCount := sumOfSquares (m - 1)
  let rightCount := sumOfSquares m
  N / 2 > leftCount ∧ N / 2 ≤ rightCount

theorem median_of_special_list : isMedian 72 := by
  sorry

end median_of_special_list_l2872_287259


namespace digital_display_overlap_l2872_287202

/-- Represents a digital number display in a rectangle -/
structure DigitalDisplay where
  width : Nat
  height : Nat
  numbers : List Nat

/-- Represents the overlap of two digital displays -/
def overlap (d1 d2 : DigitalDisplay) : Nat :=
  sorry

/-- The main theorem about overlapping digital displays -/
theorem digital_display_overlap :
  ∀ (d : DigitalDisplay),
    d.width = 8 ∧ 
    d.height = 5 ∧ 
    d.numbers = [1, 2, 1, 9] ∧
    (overlap d (DigitalDisplay.mk 8 5 [6, 1, 2, 1])) = 30 := by
  sorry

end digital_display_overlap_l2872_287202


namespace money_left_after_purchase_l2872_287274

/-- The amount of money left after buying a gift and cake for their mother -/
def money_left (gift_cost cake_cost erika_savings : ℚ) : ℚ :=
  let rick_savings := gift_cost / 2
  let total_savings := erika_savings + rick_savings
  let total_cost := gift_cost + cake_cost
  total_savings - total_cost

/-- Theorem stating the amount of money left after buying the gift and cake -/
theorem money_left_after_purchase : 
  money_left 250 25 155 = 5 := by sorry

end money_left_after_purchase_l2872_287274


namespace divisibility_by_seventeen_l2872_287268

theorem divisibility_by_seventeen (x : ℤ) (y z w : ℕ) 
  (hy : Odd y) (hz : Odd z) (hw : Odd w) : 
  ∃ k : ℤ, x^(y^(z^w)) - x^(y^z) = 17 * k := by
sorry

end divisibility_by_seventeen_l2872_287268


namespace rationalize_denominator_cube_root_rationalize_35_cube_root_l2872_287275

theorem rationalize_denominator_cube_root (x : ℝ) (hx : x > 0) :
  (x / x^(1/3)) = x^(2/3) :=
by sorry

theorem rationalize_35_cube_root :
  (35 : ℝ) / (35 : ℝ)^(1/3) = (1225 : ℝ)^(1/3) :=
by sorry

end rationalize_denominator_cube_root_rationalize_35_cube_root_l2872_287275


namespace matthews_friends_l2872_287236

theorem matthews_friends (total_crackers : ℕ) (crackers_per_friend : ℕ) 
  (h1 : total_crackers = 22)
  (h2 : crackers_per_friend = 2) :
  total_crackers / crackers_per_friend = 11 := by
  sorry

end matthews_friends_l2872_287236


namespace quadratic_inequality_solution_l2872_287254

/-- Given a quadratic inequality ax^2 + (ab+1)x + b > 0 with solution set {x | 1 < x < 3},
    prove that a + b = -4 or a + b = -4/3 -/
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + (a*b + 1)*x + b > 0 ↔ 1 < x ∧ x < 3) →
  (a + b = -4 ∨ a + b = -4/3) :=
sorry

end quadratic_inequality_solution_l2872_287254


namespace square_triangle_equal_perimeter_l2872_287221

theorem square_triangle_equal_perimeter (x : ℝ) : 
  (4 * (x + 2) = 3 * (2 * x)) → x = 4 := by
  sorry

end square_triangle_equal_perimeter_l2872_287221


namespace books_ratio_l2872_287245

theorem books_ratio (harry_books : ℕ) (total_books : ℕ) : 
  harry_books = 50 →
  total_books = 175 →
  ∃ (flora_books : ℕ),
    flora_books = 2 * harry_books ∧
    harry_books + flora_books + (harry_books / 2) = total_books :=
by sorry

end books_ratio_l2872_287245


namespace smallest_value_quadratic_l2872_287287

theorem smallest_value_quadratic :
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end smallest_value_quadratic_l2872_287287


namespace f_properties_l2872_287213

-- Define the function f
def f (x : ℝ) : ℝ := |x + 7| + |x - 1|

-- Theorem statement
theorem f_properties :
  (∀ x : ℝ, f x ≥ 8) ∧
  (∀ x : ℝ, |x - 3| - 2*x ≤ 4 ↔ x ≥ -1/3) :=
by sorry

end f_properties_l2872_287213


namespace cyclist_speed_proof_l2872_287282

/-- The speed of cyclist C in miles per hour -/
def speed_C : ℝ := 28.5

/-- The speed of cyclist D in miles per hour -/
def speed_D : ℝ := speed_C + 6

/-- The distance between City X and City Y in miles -/
def distance_XY : ℝ := 100

/-- The distance C travels before turning back in miles -/
def distance_C_before_turn : ℝ := 80

/-- The distance from City Y where C and D meet after turning back in miles -/
def meeting_distance : ℝ := 15

theorem cyclist_speed_proof :
  speed_C = 28.5 ∧
  speed_D = speed_C + 6 ∧
  (distance_C_before_turn + meeting_distance) / speed_C = 
  (distance_XY + meeting_distance) / speed_D :=
sorry

end cyclist_speed_proof_l2872_287282


namespace coin_trick_theorem_l2872_287271

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a sequence of coins -/
def CoinSequence (n : ℕ) := Fin n → CoinState

/-- Represents the strategy for the assistant and magician -/
structure Strategy (n : ℕ) where
  encode : CoinSequence n → Fin n → Fin n
  decode : CoinSequence n → Fin n

/-- Defines when a strategy is valid -/
def is_valid_strategy (n : ℕ) (s : Strategy n) : Prop :=
  ∀ (seq : CoinSequence n) (chosen : Fin n),
    ∃ (flipped : Fin n),
      s.decode (Function.update seq flipped (CoinState.Tails)) = chosen

/-- The main theorem: the trick is possible iff n is a power of 2 -/
theorem coin_trick_theorem (n : ℕ) :
  (∃ (s : Strategy n), is_valid_strategy n s) ↔ ∃ (k : ℕ), n = 2^k :=
sorry

end coin_trick_theorem_l2872_287271


namespace no_integer_solutions_l2872_287252

theorem no_integer_solutions : ¬∃ (x y : ℤ), x = x^2 + y^2 + 1 ∧ y = 3*x*y := by sorry

end no_integer_solutions_l2872_287252


namespace no_integer_solutions_l2872_287208

theorem no_integer_solutions : 
  ¬∃ (x y z : ℤ), 
    (x^2 - 4*x*y + 3*y^2 - z^2 = 41) ∧ 
    (-x^2 + 4*y*z + 2*z^2 = 52) ∧ 
    (x^2 + x*y + 5*z^2 = 110) :=
by sorry

end no_integer_solutions_l2872_287208


namespace headcount_averages_l2872_287214

def spring_headcounts : List ℕ := [10900, 10500, 10700, 11300]
def fall_headcounts : List ℕ := [11700, 11500, 11600, 11300]

theorem headcount_averages :
  (spring_headcounts.sum / spring_headcounts.length : ℚ) = 10850 ∧
  (fall_headcounts.sum / fall_headcounts.length : ℚ) = 11525 := by
  sorry

end headcount_averages_l2872_287214


namespace geometric_series_sum_l2872_287207

theorem geometric_series_sum : ∀ (a r : ℝ), 
  a = 9 → r = -2/3 → abs r < 1 → 
  (∑' n, a * r^n) = 5.4 := by sorry

end geometric_series_sum_l2872_287207


namespace sum_of_possible_N_values_l2872_287248

-- Define the set of expressions
def S (x y : ℝ) : Set ℝ := {(x + y)^2, (x - y)^2, x * y, x / y}

-- Define the given set of values
def T (N : ℝ) : Set ℝ := {4, 12.8, 28.8, N}

-- Theorem statement
theorem sum_of_possible_N_values (x y N : ℝ) (hy : y ≠ 0) 
  (h_equal : S x y = T N) : 
  ∃ (N₁ N₂ N₃ : ℝ), 
    (S x y = T N₁) ∧ 
    (S x y = T N₂) ∧ 
    (S x y = T N₃) ∧ 
    N₁ + N₂ + N₃ = 85.2 :=
sorry

end sum_of_possible_N_values_l2872_287248


namespace circumscribed_equal_triangulation_only_square_l2872_287292

/-- A convex polygon with n sides -/
structure ConvexPolygon (n : ℕ) where
  sides : n > 3

/-- A polygon is circumscribed if all its sides are tangent to a common circle -/
def IsCircumscribed (P : ConvexPolygon n) : Prop :=
  sorry

/-- A polygon can be dissected into equal triangles by non-intersecting diagonals -/
def HasEqualTriangulation (P : ConvexPolygon n) : Prop :=
  sorry

/-- The main theorem -/
theorem circumscribed_equal_triangulation_only_square
  (n : ℕ) (P : ConvexPolygon n)
  (h_circ : IsCircumscribed P)
  (h_triang : HasEqualTriangulation P) :
  n = 4 :=
sorry

end circumscribed_equal_triangulation_only_square_l2872_287292


namespace largest_number_with_distinct_digits_summing_to_19_l2872_287243

/-- Checks if all digits in a number are different -/
def hasDistinctDigits (n : ℕ) : Bool := sorry

/-- Calculates the sum of digits of a number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The target sum of digits -/
def targetSum : ℕ := 19

/-- The proposed largest number -/
def largestNumber : ℕ := 943210

theorem largest_number_with_distinct_digits_summing_to_19 :
  (∀ m : ℕ, m > largestNumber → 
    ¬(hasDistinctDigits m ∧ digitSum m = targetSum)) ∧
  hasDistinctDigits largestNumber ∧
  digitSum largestNumber = targetSum :=
sorry

end largest_number_with_distinct_digits_summing_to_19_l2872_287243


namespace max_value_complex_expression_l2872_287219

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 3) :
  Complex.abs ((z - 1)^3 * (z + 1)) ≤ 12 * Real.sqrt 3 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 3 ∧ Complex.abs ((w - 1)^3 * (w + 1)) = 12 * Real.sqrt 3 :=
sorry

end max_value_complex_expression_l2872_287219


namespace tea_consumption_average_l2872_287246

/-- Represents the inverse proportionality between research hours and tea quantity -/
def inverse_prop (k : ℝ) (r t : ℝ) : Prop := r * t = k

theorem tea_consumption_average (k : ℝ) :
  k = 8 * 3 →
  let t1 := k / 5
  let t2 := k / 10
  let t3 := k / 7
  (t1 + t2 + t3) / 3 = 124 / 35 := by
  sorry

end tea_consumption_average_l2872_287246


namespace arithmetic_sequence_fifth_term_l2872_287224

/-- Given an arithmetic sequence {a_n} where a₂ + a₄ = 8 and a₁ = 2, prove that a₅ = 6 -/
theorem arithmetic_sequence_fifth_term (a : ℕ → ℝ) 
  (h_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_sum : a 2 + a 4 = 8)
  (h_first : a 1 = 2) : 
  a 5 = 6 := by
  sorry

end arithmetic_sequence_fifth_term_l2872_287224


namespace springdale_rainfall_l2872_287225

theorem springdale_rainfall (first_week : ℝ) (second_week : ℝ) : 
  second_week = 1.5 * first_week →
  second_week = 24 →
  first_week + second_week = 40 := by
sorry

end springdale_rainfall_l2872_287225


namespace joes_test_count_l2872_287250

theorem joes_test_count (n : ℕ) (initial_average final_average lowest_score : ℚ) 
  (h1 : initial_average = 50)
  (h2 : final_average = 55)
  (h3 : lowest_score = 35)
  (h4 : n * initial_average = (n - 1) * final_average + lowest_score)
  (h5 : n > 1) : n = 4 := by
  sorry

end joes_test_count_l2872_287250


namespace fraction_simplification_l2872_287293

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 11 + 2 / 9) = 4257 / 2345 := by
  sorry

end fraction_simplification_l2872_287293


namespace iceland_visitors_l2872_287200

theorem iceland_visitors (total : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 90)
  (h2 : norway = 33)
  (h3 : both = 51)
  (h4 : neither = 53) :
  total - neither - norway + both = 55 := by
  sorry

end iceland_visitors_l2872_287200


namespace cruz_marbles_l2872_287297

/-- Proof that Cruz has 8 marbles given the conditions of the problem -/
theorem cruz_marbles :
  ∀ (atticus jensen cruz : ℕ),
  3 * (atticus + jensen + cruz) = 60 →
  atticus = jensen / 2 →
  atticus = 4 →
  cruz = 8 := by
sorry

end cruz_marbles_l2872_287297


namespace zero_in_A_l2872_287240

def A : Set ℝ := {x : ℝ | x * (x - 2) = 0}

theorem zero_in_A : (0 : ℝ) ∈ A := by sorry

end zero_in_A_l2872_287240


namespace mn_positive_necessary_not_sufficient_l2872_287264

/-- A curve represented by the equation mx^2 + ny^2 = 1 is an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- The main theorem stating that mn > 0 is a necessary but not sufficient condition for the curve to be an ellipse -/
theorem mn_positive_necessary_not_sufficient (m n : ℝ) :
  (is_ellipse m n → m * n > 0) ∧
  ∃ m n, m * n > 0 ∧ ¬(is_ellipse m n) :=
sorry

end mn_positive_necessary_not_sufficient_l2872_287264


namespace no_m_exists_for_equal_sets_l2872_287298

theorem no_m_exists_for_equal_sets : ¬∃ m : ℝ, 
  {x : ℝ | x^2 - 8*x - 20 ≤ 0} = {x : ℝ | 1 - m ≤ x ∧ x ≤ 1 + m} := by
  sorry

end no_m_exists_for_equal_sets_l2872_287298


namespace medicine_supply_duration_l2872_287263

theorem medicine_supply_duration (pills_per_supply : ℕ) (pill_fraction : ℚ) (days_between_doses : ℕ) (days_per_month : ℕ) : 
  pills_per_supply = 90 →
  pill_fraction = 1/3 →
  days_between_doses = 3 →
  days_per_month = 30 →
  (pills_per_supply * (days_between_doses / pill_fraction) / days_per_month : ℚ) = 27 := by
  sorry

end medicine_supply_duration_l2872_287263
