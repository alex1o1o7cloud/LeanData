import Mathlib

namespace age_difference_l3033_303333

/-- Proves that given a man and his son, where the son's present age is 24 and in two years
    the man's age will be twice his son's age, the difference between their present ages is 26 years. -/
theorem age_difference (man_age son_age : ℕ) : 
  son_age = 24 →
  man_age + 2 = 2 * (son_age + 2) →
  man_age - son_age = 26 := by
  sorry

end age_difference_l3033_303333


namespace total_paths_is_nine_l3033_303371

/-- A graph representing the paths between points A, B, C, and D -/
structure PathGraph where
  paths_AB : ℕ
  paths_BD : ℕ
  paths_DC : ℕ
  direct_AC : ℕ

/-- The total number of paths from A to C in the given graph -/
def total_paths (g : PathGraph) : ℕ :=
  g.paths_AB * g.paths_BD * g.paths_DC + g.direct_AC

/-- Theorem stating that the total number of paths from A to C is 9 -/
theorem total_paths_is_nine (g : PathGraph) 
  (h1 : g.paths_AB = 2)
  (h2 : g.paths_BD = 2)
  (h3 : g.paths_DC = 2)
  (h4 : g.direct_AC = 1) :
  total_paths g = 9 := by
  sorry

end total_paths_is_nine_l3033_303371


namespace sqrt_meaningful_range_l3033_303329

theorem sqrt_meaningful_range (x : ℝ) :
  (∃ y : ℝ, y ^ 2 = 3 - x) ↔ x ≤ 3 := by
  sorry

end sqrt_meaningful_range_l3033_303329


namespace probability_of_eight_in_three_elevenths_l3033_303376

/-- The decimal representation of a rational number -/
def decimal_representation (q : ℚ) : List ℕ := sorry

/-- The probability of a digit occurring in a list of digits -/
def digit_probability (d : ℕ) (l : List ℕ) : ℚ := sorry

theorem probability_of_eight_in_three_elevenths :
  digit_probability 8 (decimal_representation (3/11)) = 0 := by sorry

end probability_of_eight_in_three_elevenths_l3033_303376


namespace third_smallest_four_digit_pascal_l3033_303303

/-- Pascal's triangle function -/
def pascal (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Predicate for a number being in Pascal's triangle -/
def inPascalTriangle (x : ℕ) : Prop := ∃ n k, pascal n k = x

/-- The set of four-digit numbers in Pascal's triangle -/
def fourDigitPascalNumbers : Set ℕ := {x | 1000 ≤ x ∧ x ≤ 9999 ∧ inPascalTriangle x}

/-- The third smallest element in a set of natural numbers -/
noncomputable def thirdSmallest (S : Set ℕ) : ℕ := sorry

theorem third_smallest_four_digit_pascal :
  thirdSmallest fourDigitPascalNumbers = 1002 := by sorry

end third_smallest_four_digit_pascal_l3033_303303


namespace sum_of_reciprocal_cubes_l3033_303349

theorem sum_of_reciprocal_cubes (a b c : ℝ) 
  (sum_condition : a + b + c = 6)
  (prod_sum_condition : a * b + b * c + c * a = 5)
  (prod_condition : a * b * c = 1) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 128 := by
sorry

end sum_of_reciprocal_cubes_l3033_303349


namespace probability_point_in_circle_l3033_303347

/-- The probability of a randomly selected point in a square with side length 6 
    being within 2 units of the center is π/9 -/
theorem probability_point_in_circle (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 6 → 
  circle_radius = 2 → 
  (π * circle_radius^2) / (square_side^2) = π / 9 := by
  sorry

end probability_point_in_circle_l3033_303347


namespace exists_n_power_half_eq_ten_l3033_303304

theorem exists_n_power_half_eq_ten :
  ∃ n : ℝ, n > 0 ∧ n ^ (n / 2) = 10 := by
sorry

end exists_n_power_half_eq_ten_l3033_303304


namespace cone_volume_from_sphere_properties_l3033_303334

/-- Given a sphere and a cone with specific properties, prove that the volume of the cone is 12288π cm³ -/
theorem cone_volume_from_sphere_properties (r : ℝ) (h : ℝ) (S_sphere : ℝ) (S_cone : ℝ) (V_cone : ℝ) :
  r = 24 →
  h = 2 * r →
  S_sphere = 4 * π * r^2 →
  S_cone = S_sphere →
  V_cone = (1/3) * π * (S_cone / (2 * π * h))^2 * h →
  V_cone = 12288 * π := by
sorry

end cone_volume_from_sphere_properties_l3033_303334


namespace value_of_t_l3033_303382

-- Define variables
variable (p j t : ℝ)

-- Define the conditions
def condition1 : Prop := j = 0.75 * p
def condition2 : Prop := j = 0.80 * t
def condition3 : Prop := t = p * (1 - t / 100)

-- Theorem statement
theorem value_of_t (h1 : condition1 p j) (h2 : condition2 j t) (h3 : condition3 p t) : t = 6.25 := by
  sorry

end value_of_t_l3033_303382


namespace negation_equivalence_l3033_303348

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end negation_equivalence_l3033_303348


namespace machine_production_difference_l3033_303307

/-- Proves that Machine B makes 20 more products than Machine A under given conditions -/
theorem machine_production_difference :
  ∀ (rate_A rate_B total_B : ℕ) (time : ℚ),
    rate_A = 8 →
    rate_B = 10 →
    total_B = 100 →
    time = total_B / rate_B →
    total_B - (rate_A * time) = 20 :=
by
  sorry

end machine_production_difference_l3033_303307


namespace xiao_ming_final_score_l3033_303312

/-- Calculate the final score given individual scores and weights -/
def final_score (content_score language_score demeanor_score : ℝ)
  (content_weight language_weight demeanor_weight : ℝ) : ℝ :=
  content_score * content_weight +
  language_score * language_weight +
  demeanor_score * demeanor_weight

/-- Theorem stating that Xiao Ming's final score is 86.2 -/
theorem xiao_ming_final_score :
  final_score 85 90 82 0.6 0.3 0.1 = 86.2 := by
  sorry

#eval final_score 85 90 82 0.6 0.3 0.1

end xiao_ming_final_score_l3033_303312


namespace remainder_seven_times_quotient_l3033_303335

theorem remainder_seven_times_quotient :
  {n : ℕ+ | ∃ q : ℕ, n = 23 * q + 7 * q ∧ 7 * q < 23} = {30, 60, 90} := by
  sorry

end remainder_seven_times_quotient_l3033_303335


namespace max_intersections_8_6_l3033_303331

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersections for 8 x-axis points and 6 y-axis points -/
theorem max_intersections_8_6 :
  max_intersections 8 6 = 420 := by
  sorry

end max_intersections_8_6_l3033_303331


namespace wendy_facebook_pictures_l3033_303314

def total_pictures (one_album_pictures : ℕ) (num_other_albums : ℕ) (pictures_per_other_album : ℕ) : ℕ :=
  one_album_pictures + num_other_albums * pictures_per_other_album

theorem wendy_facebook_pictures :
  total_pictures 27 9 2 = 45 := by
  sorry

end wendy_facebook_pictures_l3033_303314


namespace cistern_empty_in_eight_minutes_l3033_303391

/-- Given a pipe that can empty 2/3 of a cistern in 10 minutes,
    this function calculates the part of the cistern that will be empty in a given number of minutes. -/
def cisternEmptyPart (emptyRate : Rat) (totalTime : Nat) (elapsedTime : Nat) : Rat :=
  (emptyRate / totalTime) * elapsedTime

/-- Theorem stating that given a pipe that can empty 2/3 of a cistern in 10 minutes,
    the part of the cistern that will be empty in 8 minutes is 8/15. -/
theorem cistern_empty_in_eight_minutes :
  cisternEmptyPart (2/3) 10 8 = 8/15 := by
  sorry

end cistern_empty_in_eight_minutes_l3033_303391


namespace unique_rectangle_with_half_perimeter_quarter_area_l3033_303351

theorem unique_rectangle_with_half_perimeter_quarter_area 
  (a b : ℝ) (hab : a < b) : 
  ∃! (x y : ℝ), x < b ∧ y < b ∧ 
  2 * (x + y) = a + b ∧ 
  x * y = (a * b) / 4 := by
sorry

end unique_rectangle_with_half_perimeter_quarter_area_l3033_303351


namespace remainder_problem_l3033_303326

theorem remainder_problem (a b : ℕ) (h1 : a - b = 1311) (h2 : a / b = 11) (h3 : a = 1430) :
  a % b = 121 := by
  sorry

end remainder_problem_l3033_303326


namespace amusement_park_average_cost_l3033_303339

/-- Represents the cost and trips data for a child's season pass -/
structure ChildData where
  pass_cost : ℕ
  trips : ℕ

/-- Calculates the average cost per trip given a list of ChildData -/
def average_cost_per_trip (children : List ChildData) : ℚ :=
  let total_cost := children.map (λ c => c.pass_cost) |>.sum
  let total_trips := children.map (λ c => c.trips) |>.sum
  (total_cost : ℚ) / total_trips

/-- The main theorem stating the average cost per trip for the given scenario -/
theorem amusement_park_average_cost :
  let children : List ChildData := [
    { pass_cost := 100, trips := 35 },
    { pass_cost := 90, trips := 25 },
    { pass_cost := 80, trips := 20 },
    { pass_cost := 70, trips := 15 }
  ]
  abs (average_cost_per_trip children - 3.58) < 0.01 := by
  sorry


end amusement_park_average_cost_l3033_303339


namespace problem_1_l3033_303332

theorem problem_1 (x : ℝ) : 4 * (x + 1)^2 = 49 → x = 5/2 ∨ x = -9/2 := by
  sorry

end problem_1_l3033_303332


namespace circle_y_axis_intersection_sum_l3033_303369

theorem circle_y_axis_intersection_sum :
  ∀ (y₁ y₂ : ℝ),
  (0 + 3)^2 + (y₁ - 5)^2 = 8^2 →
  (0 + 3)^2 + (y₂ - 5)^2 = 8^2 →
  y₁ ≠ y₂ →
  y₁ + y₂ = 10 := by
sorry

end circle_y_axis_intersection_sum_l3033_303369


namespace arithmetic_mean_neg6_to_8_l3033_303390

def arithmetic_mean (a b : Int) : ℚ :=
  let n := b - a + 1
  let sum := (n * (a + b)) / 2
  sum / n

theorem arithmetic_mean_neg6_to_8 :
  arithmetic_mean (-6) 8 = 1 := by
  sorry

end arithmetic_mean_neg6_to_8_l3033_303390


namespace total_scholarship_amount_l3033_303386

-- Define the scholarship amounts
def wendy_scholarship : ℕ := 20000
def kelly_scholarship : ℕ := 2 * wendy_scholarship
def nina_scholarship : ℕ := kelly_scholarship - 8000

-- Theorem statement
theorem total_scholarship_amount :
  wendy_scholarship + kelly_scholarship + nina_scholarship = 92000 := by
  sorry

end total_scholarship_amount_l3033_303386


namespace two_by_two_table_sum_l3033_303370

theorem two_by_two_table_sum (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a + b = c + d →
  a * c = b * d →
  a + b + c + d = 0 := by
sorry

end two_by_two_table_sum_l3033_303370


namespace complex_subtraction_l3033_303398

theorem complex_subtraction (a b : ℂ) (h1 : a = 5 - 3*I) (h2 : b = 4 + I) :
  a - 3*b = -7 - 6*I := by
  sorry

end complex_subtraction_l3033_303398


namespace symmetry_and_monotonicity_l3033_303321

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def symmetric_about_one (f : ℝ → ℝ) : Prop := ∀ x, f (2 - x) = f x

def increasing_on_zero_one (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 → (x₁ - x₂) * (f x₁ - f x₂) > 0

theorem symmetry_and_monotonicity
  (f : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_sym : symmetric_about_one f)
  (h_inc : increasing_on_zero_one f) :
  (∀ x, f (4 - x) + f x = 0) ∧
  (∀ x, 2 < x ∧ x < 3 → ∀ y, 2 < y ∧ y < 3 ∧ x < y → f y < f x) :=
sorry

end symmetry_and_monotonicity_l3033_303321


namespace minimum_questionnaires_to_mail_l3033_303368

theorem minimum_questionnaires_to_mail (response_rate : ℝ) (required_responses : ℕ) :
  response_rate = 0.62 →
  required_responses = 300 →
  ∃ n : ℕ, n ≥ (required_responses : ℝ) / response_rate ∧
    ∀ m : ℕ, m < n → (m : ℝ) * response_rate < required_responses := by
  sorry

#check minimum_questionnaires_to_mail

end minimum_questionnaires_to_mail_l3033_303368


namespace max_area_rectangle_l3033_303344

/-- A rectangle with a given perimeter -/
structure Rectangle where
  length : ℝ
  width : ℝ
  perimeter_constraint : length + width = 20

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: The rectangle with maximum area among all rectangles with perimeter 40 is a square with sides 10 -/
theorem max_area_rectangle :
  ∀ r : Rectangle, area r ≤ area { length := 10, width := 10, perimeter_constraint := by norm_num } :=
sorry

end max_area_rectangle_l3033_303344


namespace focal_length_of_hyperbola_l3033_303363

-- Define the hyperbola C
def hyperbola (m : ℝ) (x y : ℝ) : Prop := x^2 / m - y^2 = 1

-- Define the asymptote
def asymptote (m : ℝ) (x y : ℝ) : Prop := Real.sqrt 3 * x + m * y = 0

-- Theorem statement
theorem focal_length_of_hyperbola (m : ℝ) (h1 : m > 0) 
  (h2 : ∀ x y, hyperbola m x y ↔ asymptote m x y) : 
  ∃ a b c : ℝ, a^2 = m ∧ b^2 = 1 ∧ c^2 = a^2 + b^2 ∧ 2 * c = 4 := by
  sorry

end focal_length_of_hyperbola_l3033_303363


namespace max_value_of_g_l3033_303387

-- Define the function g(x)
def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

-- State the theorem
theorem max_value_of_g :
  ∃ (c : ℝ), c ∈ Set.Icc 0 2 ∧ 
  (∀ x, x ∈ Set.Icc 0 2 → g x ≤ g c) ∧
  g c = 25 / 8 := by
  sorry

end max_value_of_g_l3033_303387


namespace binomial_sum_l3033_303378

theorem binomial_sum (n : ℕ) (h : n > 0) : 
  Nat.choose n 1 + Nat.choose n (n - 2) = (n^2 + n) / 2 := by
  sorry

end binomial_sum_l3033_303378


namespace even_function_implies_a_zero_l3033_303308

def f (a : ℝ) (x : ℝ) : ℝ := 2 - |x + a|

theorem even_function_implies_a_zero :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 := by
  sorry

end even_function_implies_a_zero_l3033_303308


namespace bananas_per_box_l3033_303364

theorem bananas_per_box (total_bananas : ℕ) (num_boxes : ℕ) 
  (h1 : total_bananas = 40) (h2 : num_boxes = 8) : 
  total_bananas / num_boxes = 5 := by
  sorry

end bananas_per_box_l3033_303364


namespace ages_of_peter_and_grace_l3033_303396

/-- Represents a person's age -/
structure Age where
  value : ℕ

/-- Represents the ages of Peter, Jacob, and Grace -/
structure AgeGroup where
  peter : Age
  jacob : Age
  grace : Age

/-- Check if the given ages satisfy the problem conditions -/
def satisfies_conditions (ages : AgeGroup) : Prop :=
  (ages.peter.value - 10 = (ages.jacob.value - 10) / 3) ∧
  (ages.jacob.value = ages.peter.value + 12) ∧
  (ages.grace.value = (ages.peter.value + ages.jacob.value) / 2)

theorem ages_of_peter_and_grace (ages : AgeGroup) 
  (h : satisfies_conditions ages) : 
  ages.peter.value = 16 ∧ ages.grace.value = 22 := by
  sorry

#check ages_of_peter_and_grace

end ages_of_peter_and_grace_l3033_303396


namespace no_solution_for_x_equals_one_l3033_303325

theorem no_solution_for_x_equals_one (a : ℝ) (h : a ≠ 0) :
  ¬∃ x : ℝ, x = 1 ∧ a^2 * x^2 + (a + 1) * x + 1 = 0 := by
sorry

end no_solution_for_x_equals_one_l3033_303325


namespace hiker_distance_l3033_303330

/-- Hiker's walking problem -/
theorem hiker_distance 
  (x y : ℝ) 
  (h1 : x * y = 18) 
  (D2 : ℝ := (y - 1) * (x + 1))
  (D3 : ℝ := 5 * 3)
  (D_total : ℝ := 18 + D2 + D3)
  (T_total : ℝ := y + (y - 1) + 3)
  (Z : ℝ)
  (h2 : Z = D_total / T_total) :
  D_total = x * y + y - x + 32 := by
sorry

end hiker_distance_l3033_303330


namespace bottle_caps_remaining_l3033_303381

theorem bottle_caps_remaining (initial : Nat) (removed : Nat) (h1 : initial = 16) (h2 : removed = 6) :
  initial - removed = 10 := by
  sorry

end bottle_caps_remaining_l3033_303381


namespace first_box_contacts_l3033_303357

/-- Given two boxes of contacts, prove that the first box contains 75 contacts. -/
theorem first_box_contacts (price1 : ℚ) (quantity2 : ℕ) (price2 : ℚ) 
  (chosen_price : ℚ) (chosen_quantity : ℕ) :
  price1 = 25 →
  quantity2 = 99 →
  price2 = 33 →
  chosen_price = 1 →
  chosen_quantity = 3 →
  ∃ quantity1 : ℕ, quantity1 = 75 ∧ 
    price1 / quantity1 = min (price1 / quantity1) (price2 / quantity2) ∧
    price1 / quantity1 = chosen_price / chosen_quantity :=
by sorry


end first_box_contacts_l3033_303357


namespace sara_book_cost_l3033_303319

/-- The cost of Sara's first book -/
def first_book_cost : ℝ := sorry

/-- The cost of Sara's second book -/
def second_book_cost : ℝ := 6.5

/-- The amount Sara paid -/
def amount_paid : ℝ := 20

/-- The change Sara received -/
def change_received : ℝ := 8

theorem sara_book_cost : first_book_cost = 5.5 := by sorry

end sara_book_cost_l3033_303319


namespace smallest_block_volume_l3033_303353

theorem smallest_block_volume (a b c : ℕ) : 
  (a - 1) * (b - 1) * (c - 1) = 252 → 
  a * b * c ≥ 392 ∧ 
  ∃ (a' b' c' : ℕ), (a' - 1) * (b' - 1) * (c' - 1) = 252 ∧ a' * b' * c' = 392 := by
  sorry

end smallest_block_volume_l3033_303353


namespace students_liking_both_sports_l3033_303366

theorem students_liking_both_sports (basketball : ℕ) (cricket : ℕ) (total : ℕ) : 
  basketball = 12 → cricket = 8 → total = 17 → 
  basketball + cricket - total = 3 := by
sorry

end students_liking_both_sports_l3033_303366


namespace unique_solution_equation_l3033_303302

theorem unique_solution_equation (x : ℝ) :
  x ≥ 0 →
  (2021 * x = 2022 * (x ^ (2021 / 2022)) - 1) ↔
  x = 1 :=
by sorry

end unique_solution_equation_l3033_303302


namespace tangent_to_parabola_l3033_303380

-- Define the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the given line
def given_line (x y : ℝ) : Prop := 4 * x - y + 3 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x - y - 2 = 0

theorem tangent_to_parabola :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the parabola
    y₀ = parabola x₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The tangent line is parallel to the given line
    (∀ (x y : ℝ), tangent_line x y ↔ ∃ (k : ℝ), y = 4 * x + k) ∧
    -- The tangent line touches the parabola at exactly one point
    (∀ (x y : ℝ), x ≠ x₀ → y = parabola x → ¬ tangent_line x y) :=
sorry

end tangent_to_parabola_l3033_303380


namespace coconut_oil_needed_l3033_303379

/-- Calculates the amount of coconut oil needed for baking brownies --/
theorem coconut_oil_needed
  (butter_per_cup : ℝ)
  (coconut_oil_per_cup : ℝ)
  (butter_available : ℝ)
  (total_baking_mix : ℝ)
  (h1 : butter_per_cup = 2)
  (h2 : coconut_oil_per_cup = 2)
  (h3 : butter_available = 4)
  (h4 : total_baking_mix = 6) :
  (total_baking_mix - butter_available / butter_per_cup) * coconut_oil_per_cup = 8 :=
by sorry

end coconut_oil_needed_l3033_303379


namespace cosine_sine_identity_l3033_303317

theorem cosine_sine_identity : Real.cos (75 * π / 180) * Real.cos (15 * π / 180) - Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 0 := by
  sorry

end cosine_sine_identity_l3033_303317


namespace friendship_theorem_l3033_303397

/-- A simple graph with 17 vertices where each vertex has degree 4 -/
structure FriendshipGraph where
  vertices : Finset (Fin 17)
  edges : Finset (Fin 17 × Fin 17)
  edge_symmetric : ∀ a b, (a, b) ∈ edges → (b, a) ∈ edges
  no_self_loops : ∀ a, (a, a) ∉ edges
  degree_four : ∀ v, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 4

/-- Two vertices are acquainted if there's an edge between them -/
def acquainted (G : FriendshipGraph) (a b : Fin 17) : Prop :=
  (a, b) ∈ G.edges

/-- Two vertices share a common neighbor if there exists a vertex connected to both -/
def share_neighbor (G : FriendshipGraph) (a b : Fin 17) : Prop :=
  ∃ c, acquainted G a c ∧ acquainted G b c

/-- Main theorem: There exist two vertices that are not acquainted and do not share a neighbor -/
theorem friendship_theorem (G : FriendshipGraph) : 
  ∃ a b, a ≠ b ∧ ¬(acquainted G a b) ∧ ¬(share_neighbor G a b) := by
  sorry

end friendship_theorem_l3033_303397


namespace remainder_problem_l3033_303343

theorem remainder_problem (L S R : ℕ) (h1 : L - S = 1365) (h2 : S = 270) (h3 : L = 6 * S + R) : R = 15 := by
  sorry

end remainder_problem_l3033_303343


namespace min_decimal_digits_l3033_303310

def fraction : ℚ := 987654321 / (2^30 * 5^2)

theorem min_decimal_digits (f : ℚ) (h : f = fraction) : 
  (∃ (n : ℕ), n ≥ 30 ∧ ∃ (m : ℤ), f * 10^n = m) ∧ 
  (∀ (k : ℕ), k < 30 → ¬∃ (m : ℤ), f * 10^k = m) := by
  sorry

end min_decimal_digits_l3033_303310


namespace pencil_cost_with_discount_cost_of_3000_pencils_l3033_303355

/-- The cost of pencils with a bulk discount -/
theorem pencil_cost_with_discount (base_quantity : ℕ) (base_cost : ℚ) 
  (order_quantity : ℕ) (discount_threshold : ℕ) (discount_rate : ℚ) : ℚ :=
  let base_price_per_pencil := base_cost / base_quantity
  let discounted_price_per_pencil := base_price_per_pencil * (1 - discount_rate)
  let total_cost := if order_quantity > discount_threshold
                    then order_quantity * discounted_price_per_pencil
                    else order_quantity * base_price_per_pencil
  total_cost

/-- The cost of 3000 pencils with the given conditions -/
theorem cost_of_3000_pencils : 
  pencil_cost_with_discount 150 40 3000 1000 (5/100) = 760 := by
  sorry

end pencil_cost_with_discount_cost_of_3000_pencils_l3033_303355


namespace spelling_contest_result_l3033_303320

/-- In a spelling contest, given the following conditions:
  * There were 52 total questions
  * Drew got 20 questions correct
  * Drew got 6 questions wrong
  * Carla got twice as many questions wrong as Drew
Prove that Carla got 40 questions correct. -/
theorem spelling_contest_result (total_questions : Nat) (drew_correct : Nat) (drew_wrong : Nat) (carla_wrong_multiplier : Nat) :
  total_questions = 52 →
  drew_correct = 20 →
  drew_wrong = 6 →
  carla_wrong_multiplier = 2 →
  total_questions - (carla_wrong_multiplier * drew_wrong) = 40 := by
  sorry

#check spelling_contest_result

end spelling_contest_result_l3033_303320


namespace third_term_of_sequence_l3033_303393

/-- Given a sequence {a_n} with S_n as the sum of the first n terms, and S_n = n^2 + n, prove a_3 = 6 -/
theorem third_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ n, S n = n^2 + n) : a 3 = 6 := by
  sorry

end third_term_of_sequence_l3033_303393


namespace modulus_of_complex_number_l3033_303345

theorem modulus_of_complex_number (i : ℂ) (h : i * i = -1) :
  Complex.abs (2 + 1 / i) = Real.sqrt 5 := by
  sorry

end modulus_of_complex_number_l3033_303345


namespace range_of_H_l3033_303346

def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

theorem range_of_H :
  Set.range H = {-4, 4} := by sorry

end range_of_H_l3033_303346


namespace range_of_g_l3033_303385

/-- A function g defined on the interval [-1, 1] with g(x) = cx + d, where c < 0 and d > 0 -/
def g (c d : ℝ) (hc : c < 0) (hd : d > 0) : ℝ → ℝ :=
  fun x => c * x + d

/-- The range of g is [c + d, -c + d] -/
theorem range_of_g (c d : ℝ) (hc : c < 0) (hd : d > 0) :
  Set.range (g c d hc hd) = Set.Icc (c + d) (-c + d) := by
  sorry

end range_of_g_l3033_303385


namespace quadratic_root_to_coefficient_l3033_303342

theorem quadratic_root_to_coefficient (m : ℚ) : 
  (∀ x : ℂ, 6 * x^2 + 5 * x + m = 0 ↔ x = (-5 + Complex.I * Real.sqrt 231) / 12 ∨ x = (-5 - Complex.I * Real.sqrt 231) / 12) →
  m = 32 / 3 :=
by sorry

end quadratic_root_to_coefficient_l3033_303342


namespace problem_statement_l3033_303358

theorem problem_statement (a b m : ℝ) : 
  2^a = m ∧ 3^b = m ∧ a * b ≠ 0 ∧ 
  ∃ (k : ℝ), a + k = a * b ∧ a * b + k = b → 
  m = Real.sqrt 6 := by
sorry

end problem_statement_l3033_303358


namespace not_necessarily_equal_proportion_l3033_303338

theorem not_necessarily_equal_proportion (a b c d : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a * d = b * c) : 
  ¬(∀ a b c d, (a + 1) / b = (c + 1) / d) :=
by
  sorry

end not_necessarily_equal_proportion_l3033_303338


namespace f_difference_l3033_303360

/-- The function f(x) = 3x^2 + 5x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

/-- Theorem stating that f(x+h) - f(x) = h(6x + 3h + 5) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h + 5) := by
  sorry

end f_difference_l3033_303360


namespace triangle_b_range_l3033_303395

open Real Set

-- Define the triangle and its properties
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions for the triangle
def TriangleConditions (t : Triangle) : Prop :=
  t.a = Real.sqrt 3 ∧ t.A = π / 3

-- Define the condition for exactly one solution
def ExactlyOneSolution (t : Triangle) : Prop :=
  (t.a = t.b * Real.sin t.A) ∨ (t.a ≥ t.b ∧ t.a > t.b * Real.sin t.A)

-- Define the range of values for b
def BRange : Set ℝ := Ioc 0 (Real.sqrt 3) ∪ {2}

-- State the theorem
theorem triangle_b_range (t : Triangle) :
  TriangleConditions t → ExactlyOneSolution t → t.b ∈ BRange :=
by sorry

end triangle_b_range_l3033_303395


namespace liquid_X_percentage_l3033_303305

/-- The percentage of liquid X in solution A -/
def percentage_X_in_A : ℝ := 1.67

/-- The weight of solution A in grams -/
def weight_A : ℝ := 600

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in solution B -/
def percentage_X_in_B : ℝ := 1.8

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 1.74

theorem liquid_X_percentage :
  (percentage_X_in_A * weight_A + percentage_X_in_B * weight_B) / (weight_A + weight_B) = percentage_X_in_mixture := by
  sorry

end liquid_X_percentage_l3033_303305


namespace ratio_of_tenth_terms_l3033_303309

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a 1 + (n - 1) * seq.d) / 2

theorem ratio_of_tenth_terms 
  (a b : ArithmeticSequence)
  (h : ∀ n, sumFirstN a n / sumFirstN b n = (3 * n - 1) / (2 * n + 3)) :
  a.a 10 / b.a 10 = 57 / 41 := by
  sorry

end ratio_of_tenth_terms_l3033_303309


namespace smallest_product_of_factors_l3033_303324

def is_factor (n m : ℕ) : Prop := m % n = 0

theorem smallest_product_of_factors : 
  ∃ (a b : ℕ), 
    a ≠ b ∧ 
    a > 0 ∧ 
    b > 0 ∧ 
    is_factor a 60 ∧ 
    is_factor b 60 ∧ 
    ¬(is_factor (a * b) 60) ∧
    a * b = 8 ∧
    (∀ (c d : ℕ), 
      c ≠ d → 
      c > 0 → 
      d > 0 → 
      is_factor c 60 → 
      is_factor d 60 → 
      ¬(is_factor (c * d) 60) → 
      c * d ≥ 8) :=
by sorry

end smallest_product_of_factors_l3033_303324


namespace annual_growth_rate_for_doubling_l3033_303323

theorem annual_growth_rate_for_doubling (x : ℝ) (y : ℝ) (h : x > 0) :
  x * (1 + y)^2 = 2*x → y = Real.sqrt 2 - 1 := by
  sorry

end annual_growth_rate_for_doubling_l3033_303323


namespace lesser_fraction_l3033_303375

theorem lesser_fraction (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : 
  min x y = 1/4 := by
sorry

end lesser_fraction_l3033_303375


namespace z_minimum_l3033_303399

/-- The function z(x, y) defined in the problem -/
def z (x y : ℝ) : ℝ := x^2 + 2*x*y + 2*y^2 + 2*x + 4*y + 3

/-- Theorem stating the minimum value of z and where it occurs -/
theorem z_minimum :
  (∀ x y : ℝ, z x y ≥ 1) ∧ (z 0 (-1) = 1) :=
by sorry

end z_minimum_l3033_303399


namespace circle_center_proof_l3033_303374

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation x² - 10x + y² - 8y = 16, 
    prove that its center is (5, 4) -/
theorem circle_center_proof (eq : CircleEquation) 
  (h1 : eq.a = 1)
  (h2 : eq.b = -10)
  (h3 : eq.c = 1)
  (h4 : eq.d = -8)
  (h5 : eq.e = -16) :
  CircleCenter.mk 5 4 = CircleCenter.mk (-eq.b / (2 * eq.a)) (-eq.d / (2 * eq.c)) :=
by sorry

end circle_center_proof_l3033_303374


namespace largest_divisible_by_eight_l3033_303318

theorem largest_divisible_by_eight (A B C : ℕ) : 
  A = 8 * B + C → 
  B = C → 
  C < 8 → 
  (∃ k : ℕ, A = 8 * k) → 
  A ≤ 63 :=
by sorry

end largest_divisible_by_eight_l3033_303318


namespace sequence_problem_l3033_303352

theorem sequence_problem (n : ℕ+) (b : ℕ → ℝ)
  (h0 : b 0 = 40)
  (h1 : b 1 = 70)
  (hn : b n = 0)
  (hk : ∀ k : ℕ, 1 ≤ k → k < n → b (k + 1) = b (k - 1) - 2 / b k) :
  n = 1401 := by sorry

end sequence_problem_l3033_303352


namespace salary_restoration_l3033_303383

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) : 
  let reduced_salary := original_salary * (1 - 0.2)
  let restoration_factor := reduced_salary * (1 + 0.25)
  restoration_factor = original_salary := by
sorry

end salary_restoration_l3033_303383


namespace print_gift_wrap_price_l3033_303367

/-- The price of print gift wrap per roll -/
def print_price : ℝ := 6

/-- The price of solid color gift wrap per roll -/
def solid_price : ℝ := 4

/-- The total number of rolls sold -/
def total_rolls : ℕ := 480

/-- The total amount of money collected -/
def total_money : ℝ := 2340

/-- The number of print rolls sold -/
def print_rolls : ℕ := 210

theorem print_gift_wrap_price :
  print_price * print_rolls + solid_price * (total_rolls - print_rolls) = total_money :=
sorry

end print_gift_wrap_price_l3033_303367


namespace perpendicular_line_through_point_l3033_303311

-- Define the given line
def given_line (x y : ℝ) : Prop := x + 2 * y - 3 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x - y + 5 = 0

-- Theorem statement
theorem perpendicular_line_through_point :
  (∀ x y, given_line x y → perpendicular_line x y → x * 2 + y * 1 = -1) ∧
  (perpendicular_line point_P.1 point_P.2) ∧
  (∀ x y, given_line x y → ∀ a b, perpendicular_line a b → 
    (y - point_P.2) * (x - point_P.1) = -(b - point_P.2) * (a - point_P.1)) :=
by sorry

end perpendicular_line_through_point_l3033_303311


namespace pentagon_largest_angle_l3033_303327

theorem pentagon_largest_angle (P Q R S T : ℝ) : 
  P = 70 ∧ 
  Q = 100 ∧ 
  R = S ∧ 
  T = 2 * R + 20 ∧ 
  P + Q + R + S + T = 540 → 
  max P (max Q (max R (max S T))) = 195 := by
  sorry

end pentagon_largest_angle_l3033_303327


namespace cricket_innings_problem_l3033_303341

theorem cricket_innings_problem (current_average : ℚ) (next_innings_runs : ℚ) (average_increase : ℚ) :
  current_average = 32 →
  next_innings_runs = 158 →
  average_increase = 6 →
  ∃ n : ℕ,
    n * current_average + next_innings_runs = (n + 1) * (current_average + average_increase) ∧
    n = 20 := by
  sorry

end cricket_innings_problem_l3033_303341


namespace increasing_condition_m_range_l3033_303388

-- Define the linear function
def y (m x : ℝ) : ℝ := (m - 2) * x + 6

-- Part 1: y increases as x increases iff m > 2
theorem increasing_condition (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → y m x₁ < y m x₂) ↔ m > 2 :=
sorry

-- Part 2: Range of m when -2 ≤ x ≤ 4 and y ≤ 10
theorem m_range (m : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 4 → y m x ≤ 10) ↔ (2 < m ∧ m ≤ 3) ∨ (0 ≤ m ∧ m < 2) :=
sorry

end increasing_condition_m_range_l3033_303388


namespace max_dominoes_20x19_grid_l3033_303373

/-- Represents a rectangular grid --/
structure Grid where
  rows : ℕ
  cols : ℕ

/-- Represents a domino --/
structure Domino where
  length : ℕ
  width : ℕ

/-- The maximum number of dominoes that can be placed on a grid --/
def max_dominoes (g : Grid) (d : Domino) : ℕ :=
  (g.rows * g.cols) / (d.length * d.width)

/-- The theorem stating the maximum number of 3×1 dominoes on a 20×19 grid --/
theorem max_dominoes_20x19_grid :
  let grid : Grid := ⟨20, 19⟩
  let domino : Domino := ⟨3, 1⟩
  max_dominoes grid domino = 126 := by
  sorry

#eval max_dominoes ⟨20, 19⟩ ⟨3, 1⟩

end max_dominoes_20x19_grid_l3033_303373


namespace book_price_increase_l3033_303392

theorem book_price_increase (original_price : ℝ) (new_price : ℝ) (increase_percentage : ℝ) : 
  new_price = 330 ∧ 
  increase_percentage = 10 ∧ 
  new_price = original_price * (1 + increase_percentage / 100) →
  original_price = 300 := by
sorry

end book_price_increase_l3033_303392


namespace pebble_throwing_difference_l3033_303365

/-- The number of pebbles Candy throws -/
def candy_pebbles : ℕ := 4

/-- The number of pebbles Lance throws -/
def lance_pebbles : ℕ := 3 * candy_pebbles

/-- The difference between Lance's pebbles and Candy's pebbles -/
def pebble_difference : ℕ := lance_pebbles - candy_pebbles

theorem pebble_throwing_difference :
  pebble_difference = 8 := by sorry

end pebble_throwing_difference_l3033_303365


namespace jake_car_soap_cost_l3033_303300

/-- Represents the cost of car soap for Jake's car washing schedule -/
def car_soap_cost (washes_per_bottle : ℕ) (bottle_cost : ℚ) (total_washes : ℕ) : ℚ :=
  (total_washes / washes_per_bottle : ℚ) * bottle_cost

/-- Theorem: Jake spends $20.00 on car soap for washing his car once a week for 20 weeks -/
theorem jake_car_soap_cost :
  car_soap_cost 4 4 20 = 20 := by
  sorry

end jake_car_soap_cost_l3033_303300


namespace incorrect_proposition_l3033_303313

theorem incorrect_proposition :
  ¬(∀ (p q : Prop), (p ∧ q = False) → (p = False ∧ q = False)) := by
  sorry

end incorrect_proposition_l3033_303313


namespace sum_a_2b_is_zero_l3033_303372

theorem sum_a_2b_is_zero (a b : ℝ) (h : (a^2 + 4*a + 6)*(2*b^2 - 4*b + 7) ≤ 10) : 
  a + 2*b = 0 := by
sorry

end sum_a_2b_is_zero_l3033_303372


namespace f_of_two_equals_zero_l3033_303301

-- Define the function f
def f : ℝ → ℝ := fun x => (x - 1)^2 - 1

-- State the theorem
theorem f_of_two_equals_zero : f 2 = 0 := by
  sorry

end f_of_two_equals_zero_l3033_303301


namespace first_grade_muffins_l3033_303394

/-- The number of muffins baked by Mrs. Brier's class -/
def muffins_brier : ℕ := 18

/-- The number of muffins baked by Mrs. MacAdams's class -/
def muffins_macadams : ℕ := 20

/-- The number of muffins baked by Mrs. Flannery's class -/
def muffins_flannery : ℕ := 17

/-- The total number of muffins baked by first grade -/
def total_muffins : ℕ := muffins_brier + muffins_macadams + muffins_flannery

theorem first_grade_muffins : total_muffins = 55 := by
  sorry

end first_grade_muffins_l3033_303394


namespace tangent_line_perpendicular_l3033_303336

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.log (x + 1)

theorem tangent_line_perpendicular (n : ℝ) : 
  (∃ m : ℝ, (∀ x : ℝ, f x = m * x + f 0) ∧ 
   (m * (1 / n) = -1)) → n = -2 :=
by sorry

end tangent_line_perpendicular_l3033_303336


namespace quadratic_equation_condition_l3033_303328

/-- The equation (m-2)x^2 - 3x = 0 is quadratic in x if and only if m ≠ 2 -/
theorem quadratic_equation_condition (m : ℝ) :
  (∀ x, ∃ a b c : ℝ, a ≠ 0 ∧ (m - 2) * x^2 - 3 * x = a * x^2 + b * x + c) ↔ m ≠ 2 :=
by sorry

end quadratic_equation_condition_l3033_303328


namespace rectangle_perimeter_l3033_303359

theorem rectangle_perimeter (L W : ℝ) (h1 : L * W = (L + 6) * (W - 2)) (h2 : L * W = (L - 12) * (W + 6)) : 
  2 * (L + W) = 132 := by
sorry

end rectangle_perimeter_l3033_303359


namespace johns_numbers_l3033_303377

/-- Given a natural number, returns the number with its digits reversed -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is between 96 and 98 inclusive -/
def isBetween96And98 (n : ℕ) : Prop :=
  96 ≤ n ∧ n ≤ 98

/-- Represents the operation John performed on his number -/
def johnOperation (x : ℕ) : ℕ :=
  reverseDigits (4 * x + 17)

/-- A two-digit number satisfies John's conditions -/
def satisfiesConditions (x : ℕ) : Prop :=
  10 ≤ x ∧ x ≤ 99 ∧ isBetween96And98 (johnOperation x)

theorem johns_numbers :
  ∃ x y : ℕ, x ≠ y ∧ satisfiesConditions x ∧ satisfiesConditions y ∧
  (∀ z : ℕ, satisfiesConditions z → z = x ∨ z = y) ∧
  x = 13 ∧ y = 18 := by sorry

end johns_numbers_l3033_303377


namespace expression_evaluation_l3033_303322

theorem expression_evaluation : 
  let f (x : ℚ) := (2*x - 2) / (x + 2)
  let g (x : ℚ) := (2 * f x - 2) / (f x + 2)
  g 2 = -2/5 := by sorry

end expression_evaluation_l3033_303322


namespace school_meeting_attendance_l3033_303306

/-- The number of parents at a school meeting -/
def num_parents : ℕ := 23

/-- The number of teachers at a school meeting -/
def num_teachers : ℕ := 8

/-- The total number of people at the school meeting -/
def total_people : ℕ := 31

/-- The number of parents who asked questions to the Latin teacher -/
def latin_teacher_parents : ℕ := 16

theorem school_meeting_attendance :
  (num_parents + num_teachers = total_people) ∧
  (num_parents = latin_teacher_parents + num_teachers - 1) ∧
  (∀ i : ℕ, i < num_teachers → latin_teacher_parents + i ≤ num_parents) ∧
  (latin_teacher_parents + num_teachers - 1 = num_parents) := by
  sorry

end school_meeting_attendance_l3033_303306


namespace three_digit_base15_double_l3033_303354

/-- A function that converts a number from base 10 to base 15 --/
def toBase15 (n : ℕ) : ℕ :=
  (n / 100) * 15^2 + ((n / 10) % 10) * 15 + (n % 10)

/-- The set of three-digit numbers that satisfy the condition --/
def validNumbers : Finset ℕ := {150, 145, 290}

/-- The property that a number, when converted to base 15, is twice its original value --/
def satisfiesCondition (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ toBase15 n = 2 * n

theorem three_digit_base15_double :
  ∀ n : ℕ, satisfiesCondition n ↔ n ∈ validNumbers :=
sorry


end three_digit_base15_double_l3033_303354


namespace product_of_powers_l3033_303350

theorem product_of_powers (y : ℝ) (hy : y ≠ 0) :
  (18 * y^3) * (4 * y^2) * (1 / (2*y)^3) = 9 * y^2 := by
sorry

end product_of_powers_l3033_303350


namespace max_gcd_13n_plus_4_8n_plus_3_l3033_303384

theorem max_gcd_13n_plus_4_8n_plus_3 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) = 7) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 4) (8 * n + 3) ≤ 7) := by
  sorry

end max_gcd_13n_plus_4_8n_plus_3_l3033_303384


namespace perpendicular_vectors_l3033_303356

/-- Two vectors in R² -/
def Vector2 := Fin 2 → ℝ

/-- Dot product of two vectors in R² -/
def dot_product (v w : Vector2) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

/-- First direction vector -/
def v1 : Vector2 := ![- 6, 2]

/-- Second direction vector -/
def v2 (b : ℝ) : Vector2 := ![b, 3]

/-- Theorem: The value of b that makes the vectors perpendicular is 1 -/
theorem perpendicular_vectors : 
  ∃ b : ℝ, dot_product v1 (v2 b) = 0 ∧ b = 1 := by
sorry


end perpendicular_vectors_l3033_303356


namespace circle_and_tangent_line_l3033_303389

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the tangent line l
def tangent_line_l (x y : ℝ) : Prop := x = 2 ∨ 4*x - 3*y + 1 = 0

-- Theorem statement
theorem circle_and_tangent_line :
  -- Circle E passes through (0,0) and (1,1)
  circle_E 0 0 ∧ circle_E 1 1 ∧
  -- One of the three conditions is satisfied
  (circle_E 2 0 ∨ 
   (∀ m : ℝ, ∃ x y : ℝ, circle_E x y ∧ m*x - y - m = 0) ∨
   (∃ x : ℝ, circle_E x 0 ∧ x = 0)) →
  -- The tangent line passes through (2,3)
  (∃ x y : ℝ, circle_E x y ∧ tangent_line_l x y ∧
   ((x - 2)^2 + (y - 3)^2).sqrt = 1) :=
sorry

end circle_and_tangent_line_l3033_303389


namespace complex_number_problem_l3033_303340

theorem complex_number_problem (z : ℂ) (m n : ℝ) :
  (Complex.abs z = 2 * Real.sqrt 10) →
  (Complex.im ((3 - Complex.I) * z) = 0) →
  (Complex.re z < 0) →
  (2 * z^2 + m * z - n = 0) →
  (∃ (a b : ℝ), z = Complex.mk a b ∧ ((a = 2 ∧ b = -6) ∨ (a = -2 ∧ b = 6))) ∧
  (m + n = -72) := by
  sorry

end complex_number_problem_l3033_303340


namespace range_of_a_l3033_303361

def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}

def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2 * x + 3}

def C (a : ℝ) : Set ℝ := {t | ∃ x ∈ A a, t = x^2}

theorem range_of_a (a : ℝ) (h1 : a ≥ -2) (h2 : C a ⊆ B a) : 
  a ∈ Set.Icc (1/2 : ℝ) 3 := by
  sorry

end range_of_a_l3033_303361


namespace student_score_l3033_303362

-- Define the number of questions
def num_questions : ℕ := 5

-- Define the points per question
def points_per_question : ℕ := 20

-- Define the number of correct answers
def num_correct_answers : ℕ := 4

-- Theorem statement
theorem student_score (total_score : ℕ) :
  total_score = num_correct_answers * points_per_question :=
by sorry

end student_score_l3033_303362


namespace tangent_line_at_one_minimum_value_of_f_l3033_303316

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m * (x - 1) + f 1 → (x - y - 1 = 0) :=
sorry

-- Theorem for the minimum value of f(x)
theorem minimum_value_of_f :
  ∃ x, f x = -1 / Real.exp 1 ∧ ∀ y, f y ≥ -1 / Real.exp 1 :=
sorry

end tangent_line_at_one_minimum_value_of_f_l3033_303316


namespace union_of_sets_l3033_303337

def A (m : ℝ) : Set ℝ := {2, 2^m}
def B (m n : ℝ) : Set ℝ := {m, n}

theorem union_of_sets (m n : ℝ) (h : A m ∩ B m n = {1/4}) : 
  A m ∪ B m n = {2, -2, 1/4} := by
sorry

end union_of_sets_l3033_303337


namespace largest_fraction_l3033_303315

theorem largest_fraction : 
  let a := (1 / 17 - 1 / 19) / 20
  let b := (1 / 15 - 1 / 21) / 60
  let c := (1 / 13 - 1 / 23) / 100
  let d := (1 / 11 - 1 / 25) / 140
  d > a ∧ d > b ∧ d > c := by sorry

end largest_fraction_l3033_303315
