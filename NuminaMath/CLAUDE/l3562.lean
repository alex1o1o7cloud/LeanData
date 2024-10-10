import Mathlib

namespace sock_ratio_l3562_356235

/-- The ratio of black socks to blue socks in an order satisfying certain conditions -/
theorem sock_ratio :
  ∀ (b : ℕ) (x : ℝ),
  x > 0 →
  (18 * x + b * x) * 1.6 = 3 * b * x + 6 * x →
  (6 : ℝ) / b = 3 / 8 := by
sorry

end sock_ratio_l3562_356235


namespace sum_max_value_sum_max_x_product_max_value_product_max_x_l3562_356204

/-- Represents a point on an ellipse --/
structure EllipsePoint where
  x : ℝ
  y : ℝ
  a : ℝ
  b : ℝ
  h_ellipse : x^2 / a^2 + y^2 / b^2 = 1
  h_positive : a > 0 ∧ b > 0

/-- The sum of x and y coordinates has a maximum value --/
theorem sum_max_value (p : EllipsePoint) :
  ∃ m : ℝ, m = Real.sqrt (p.a^2 + p.b^2) ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b → q.x + q.y ≤ m :=
sorry

/-- The sum of x and y coordinates reaches its maximum when x has a specific value --/
theorem sum_max_x (p : EllipsePoint) :
  ∃ x_max : ℝ, x_max = p.a^2 / Real.sqrt (p.a^2 + p.b^2) ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b ∧ q.x + q.y = Real.sqrt (p.a^2 + p.b^2) →
      q.x = x_max :=
sorry

/-- The product of x and y coordinates has a maximum value --/
theorem product_max_value (p : EllipsePoint) :
  ∃ m : ℝ, m = p.a * p.b / 2 ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b → q.x * q.y ≤ m :=
sorry

/-- The product of x and y coordinates reaches its maximum when x has a specific value --/
theorem product_max_x (p : EllipsePoint) :
  ∃ x_max : ℝ, x_max = p.a * Real.sqrt 2 / 2 ∧
    ∀ q : EllipsePoint, q.a = p.a ∧ q.b = p.b ∧ q.x * q.y = p.a * p.b / 2 →
      q.x = x_max :=
sorry

end sum_max_value_sum_max_x_product_max_value_product_max_x_l3562_356204


namespace horner_method_example_l3562_356278

def f (x : ℝ) : ℝ := 9 + 15*x - 8*x^2 - 20*x^3 + 6*x^4 + 3*x^5

theorem horner_method_example : f 4 = 3269 := by
  sorry

end horner_method_example_l3562_356278


namespace expected_expenditure_2017_l3562_356239

/-- Represents the average income over five years -/
def average_income : ℝ := 10

/-- Represents the average expenditure over five years -/
def average_expenditure : ℝ := 8

/-- The slope of the regression line -/
def b_hat : ℝ := 0.76

/-- The y-intercept of the regression line -/
def a_hat : ℝ := average_expenditure - b_hat * average_income

/-- The regression function -/
def regression_function (x : ℝ) : ℝ := b_hat * x + a_hat

/-- The income in 10,000 yuan for which we want to predict the expenditure -/
def income_2017 : ℝ := 15

theorem expected_expenditure_2017 : 
  regression_function income_2017 = 11.8 := by sorry

end expected_expenditure_2017_l3562_356239


namespace yujin_wire_length_l3562_356223

/-- The length of Yujin's wire given Junhoe's wire length and the ratio --/
theorem yujin_wire_length (junhoe_length : ℝ) (ratio : ℝ) (h1 : junhoe_length = 134.5) (h2 : ratio = 1.06) :
  junhoe_length * ratio = 142.57 := by
  sorry

end yujin_wire_length_l3562_356223


namespace students_in_line_l3562_356248

theorem students_in_line (front : ℕ) (behind : ℕ) (taehyung : ℕ) 
  (h1 : front = 9) 
  (h2 : behind = 16) 
  (h3 : taehyung = 1) : 
  front + behind + taehyung = 26 := by
  sorry

end students_in_line_l3562_356248


namespace chicken_nuggets_order_l3562_356226

/-- The number of chicken nuggets ordered by Alyssa, Keely, and Kendall -/
theorem chicken_nuggets_order (alyssa keely kendall : ℕ) 
  (h1 : alyssa = 20)
  (h2 : keely = 2 * alyssa)
  (h3 : kendall = 2 * alyssa) :
  alyssa + keely + kendall = 100 := by
  sorry

end chicken_nuggets_order_l3562_356226


namespace min_value_expression_l3562_356241

theorem min_value_expression (x : ℝ) (h : x > -1) :
  x + 4 / (x + 1) ≥ 3 ∧ (x + 4 / (x + 1) = 3 ↔ x = 1) := by
  sorry

end min_value_expression_l3562_356241


namespace sine_function_properties_l3562_356291

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sine_function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < π/2) 
  (h_max : f ω φ (π/4) = 1) 
  (h_min : f ω φ (7*π/12) = -1) 
  (h_period : ∃ T > 0, ∀ x, f ω φ (x + T) = f ω φ x) :
  ω = 3 ∧ 
  φ = -π/4 ∧ 
  ∀ k : ℤ, ∀ x ∈ Set.Icc (2*k*π/3 + π/4) (2*k*π/3 + 7*π/12), 
    ∀ y ∈ Set.Icc (2*k*π/3 + π/4) (2*k*π/3 + 7*π/12), 
      x ≤ y → f ω φ x ≥ f ω φ y :=
by sorry

end sine_function_properties_l3562_356291


namespace sum_convergence_implies_k_value_l3562_356279

/-- Given a real number k > 1 such that the sum of (7n-3)/k^n from n=1 to infinity equals 20/3,
    prove that k = 1.9125 -/
theorem sum_convergence_implies_k_value (k : ℝ) 
  (h1 : k > 1)
  (h2 : ∑' n, (7 * n - 3) / k^n = 20/3) : 
  k = 1.9125 := by
  sorry

end sum_convergence_implies_k_value_l3562_356279


namespace sqrt_225_equals_15_l3562_356231

theorem sqrt_225_equals_15 : Real.sqrt 225 = 15 := by
  sorry

end sqrt_225_equals_15_l3562_356231


namespace probability_two_red_shoes_l3562_356234

def total_shoes : ℕ := 10
def red_shoes : ℕ := 4
def green_shoes : ℕ := 6
def drawn_shoes : ℕ := 2

theorem probability_two_red_shoes :
  (Nat.choose red_shoes drawn_shoes : ℚ) / (Nat.choose total_shoes drawn_shoes) = 2 / 15 := by
  sorry

end probability_two_red_shoes_l3562_356234


namespace min_value_sum_reciprocals_l3562_356265

theorem min_value_sum_reciprocals (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_two : a + b + c + d = 2) :
  (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 
   1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 9 ∧
  ∃ (a' b' c' d' : ℝ), 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧
    a' + b' + c' + d' = 2 ∧
    (1 / (a' + b') + 1 / (a' + c') + 1 / (a' + d') + 
     1 / (b' + c') + 1 / (b' + d') + 1 / (c' + d')) = 9 :=
by sorry

end min_value_sum_reciprocals_l3562_356265


namespace jan_ian_distance_difference_l3562_356213

/-- Represents the driving scenario of Ian, Han, and Jan -/
structure DrivingScenario where
  t : ℝ  -- Ian's driving time
  s : ℝ  -- Ian's driving speed
  ian_distance : ℝ := t * s
  han_distance : ℝ := (t + 2) * (s + 10)
  jan_distance : ℝ := (t + 3) * (s + 15)

/-- The theorem stating the difference between Jan's and Ian's distances -/
theorem jan_ian_distance_difference (scenario : DrivingScenario) 
  (h : scenario.han_distance = scenario.ian_distance + 100) : 
  scenario.jan_distance - scenario.ian_distance = 165 := by
  sorry

#check jan_ian_distance_difference

end jan_ian_distance_difference_l3562_356213


namespace expand_expression_l3562_356202

theorem expand_expression (x y : ℝ) : -12 * (3 * x - 4 + 2 * y) = -36 * x + 48 - 24 * y := by
  sorry

end expand_expression_l3562_356202


namespace chess_club_mixed_groups_l3562_356251

/-- Represents the chess club scenario -/
structure ChessClub where
  total_children : ℕ
  num_groups : ℕ
  children_per_group : ℕ
  boy_vs_boy_games : ℕ
  girl_vs_girl_games : ℕ

/-- The number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ :=
  (club.total_children - club.boy_vs_boy_games - club.girl_vs_girl_games) / 2

/-- The theorem stating the number of mixed groups in the given scenario -/
theorem chess_club_mixed_groups :
  let club := ChessClub.mk 90 30 3 30 14
  mixed_groups club = 23 := by sorry

end chess_club_mixed_groups_l3562_356251


namespace distance_between_centers_l3562_356217

/-- The distance between the centers of inscribed and circumscribed circles in a right triangle -/
theorem distance_between_centers (a b : ℝ) (ha : a = 6) (hb : b = 8) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a + b + c) / 2
  let r := (a * b) / (2 * s)
  let x_i := r
  let y_i := r
  let x_o := c / 2
  let y_o := 0
  Real.sqrt ((x_o - x_i)^2 + (y_o - y_i)^2) = Real.sqrt 13 :=
by sorry

end distance_between_centers_l3562_356217


namespace largest_rank_3_less_than_quarter_proof_l3562_356245

def rank (q : ℚ) : ℕ :=
  sorry

def largest_rank_3_less_than_quarter : ℚ :=
  sorry

theorem largest_rank_3_less_than_quarter_proof :
  rank largest_rank_3_less_than_quarter = 3 ∧
  largest_rank_3_less_than_quarter < 1/4 ∧
  largest_rank_3_less_than_quarter = 1/5 + 1/21 + 1/421 ∧
  ∀ q : ℚ, rank q = 3 → q < 1/4 → q ≤ largest_rank_3_less_than_quarter :=
by sorry

end largest_rank_3_less_than_quarter_proof_l3562_356245


namespace angle_relation_l3562_356258

-- Define a circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on a circle
def PointOnCircle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

-- Define an angle between three points
def Angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- State the angle bisector theorem
axiom angle_bisector_theorem (c : Circle) (A X Y C S : ℝ × ℝ) 
  (hA : PointOnCircle c A) (hX : PointOnCircle c X) (hY : PointOnCircle c Y) 
  (hC : PointOnCircle c C) (hS : PointOnCircle c S) :
  Angle A X C - Angle A Y C = Angle A S C

-- State the theorem to be proved
theorem angle_relation (c : Circle) (B X Y D S : ℝ × ℝ) 
  (hB : PointOnCircle c B) (hX : PointOnCircle c X) (hY : PointOnCircle c Y) 
  (hD : PointOnCircle c D) (hS : PointOnCircle c S) :
  Angle B X D - Angle B Y D = Angle B S D := by
  sorry

end angle_relation_l3562_356258


namespace john_pays_21_l3562_356224

/-- The amount John pays for candy bars -/
def john_payment (total_bars : ℕ) (dave_bars : ℕ) (price_per_bar : ℚ) : ℚ :=
  (total_bars - dave_bars : ℚ) * price_per_bar

/-- Theorem: John pays $21 for candy bars -/
theorem john_pays_21 :
  john_payment 20 6 (3/2) = 21 := by
  sorry

end john_pays_21_l3562_356224


namespace simplified_fraction_ratio_l3562_356244

theorem simplified_fraction_ratio (m : ℝ) : 
  let expression := (6 * m + 12) / 3
  ∃ (c d : ℤ), expression = c * m + d ∧ (c : ℚ) / d = 1 / 2 := by
  sorry

end simplified_fraction_ratio_l3562_356244


namespace function_domain_constraint_l3562_356211

theorem function_domain_constraint (f : ℝ → ℝ) (h : ∀ x, x ∈ (Set.Icc 0 1) → f x ≠ 0) :
  ∀ a : ℝ, (∀ x, x ∈ (Set.Icc 0 1) → (f (x - a) + f (x + a)) ≠ 0) ↔ a ∈ (Set.Icc (-1/2) (1/2)) :=
by sorry

end function_domain_constraint_l3562_356211


namespace greg_harvest_l3562_356296

theorem greg_harvest (sharon_harvest : Real) (greg_additional : Real) : 
  sharon_harvest = 0.1 →
  greg_additional = 0.3 →
  sharon_harvest + greg_additional = 0.4 := by
  sorry

end greg_harvest_l3562_356296


namespace oranges_count_l3562_356294

theorem oranges_count (joan_initial : ℕ) (tom_initial : ℕ) (sara_sold : ℕ) (christine_gave : ℕ)
  (h1 : joan_initial = 75)
  (h2 : tom_initial = 42)
  (h3 : sara_sold = 40)
  (h4 : christine_gave = 15) :
  joan_initial + tom_initial - sara_sold + christine_gave = 92 :=
by sorry

end oranges_count_l3562_356294


namespace total_fish_count_l3562_356271

/-- The number of fish tanks James has -/
def num_tanks : ℕ := 3

/-- The number of fish in the first tank -/
def fish_in_first_tank : ℕ := 20

/-- The number of fish in each of the other tanks -/
def fish_in_other_tanks : ℕ := 2 * fish_in_first_tank

/-- The total number of fish in all tanks -/
def total_fish : ℕ := fish_in_first_tank + 2 * fish_in_other_tanks

theorem total_fish_count : total_fish = 100 := by
  sorry

end total_fish_count_l3562_356271


namespace friends_contribution_proof_l3562_356203

def check_amount : ℝ := 200
def tip_percentage : ℝ := 0.20
def marks_contribution : ℝ := 30

theorem friends_contribution_proof :
  ∃ (friend_contribution : ℝ),
    tip_percentage * check_amount = friend_contribution + marks_contribution ∧
    friend_contribution = 10 := by
  sorry

end friends_contribution_proof_l3562_356203


namespace garden_area_is_400_l3562_356298

/-- A rectangular garden with specific walking distances -/
structure Garden where
  length : ℝ
  width : ℝ
  length_total : ℝ
  perimeter_total : ℝ
  length_walks : ℕ
  perimeter_walks : ℕ

/-- The garden satisfies the given conditions -/
def garden_satisfies_conditions (g : Garden) : Prop :=
  g.length_total = 2000 ∧
  g.perimeter_total = 2000 ∧
  g.length_walks = 50 ∧
  g.perimeter_walks = 20 ∧
  g.length_total = g.length * g.length_walks ∧
  g.perimeter_total = g.perimeter_walks * (2 * g.length + 2 * g.width)

/-- The theorem stating that a garden satisfying the conditions has an area of 400 square meters -/
theorem garden_area_is_400 (g : Garden) (h : garden_satisfies_conditions g) : 
  g.length * g.width = 400 := by
  sorry

#check garden_area_is_400

end garden_area_is_400_l3562_356298


namespace magic_trick_strategy_exists_l3562_356242

/-- Represents a card in the set of 29 cards -/
def Card := Fin 29

/-- Represents a pair of cards -/
def CardPair := (Card × Card)

/-- The strategy function for the assistant -/
def AssistantStrategy := (CardPair → CardPair)

/-- The deduction function for the magician -/
def MagicianDeduction := (CardPair → CardPair)

/-- Theorem stating the existence of a successful strategy -/
theorem magic_trick_strategy_exists :
  ∃ (strategy : AssistantStrategy) (deduction : MagicianDeduction),
    ∀ (audience_choice : CardPair),
      deduction (strategy audience_choice) = audience_choice :=
by sorry

end magic_trick_strategy_exists_l3562_356242


namespace percentage_difference_l3562_356280

theorem percentage_difference : (60 / 100 * 50) - (40 / 100 * 30) = 18 := by
  sorry

end percentage_difference_l3562_356280


namespace number_exceeding_80_percent_l3562_356263

theorem number_exceeding_80_percent : ∃ x : ℝ, x = 0.8 * x + 120 ∧ x = 600 := by
  sorry

end number_exceeding_80_percent_l3562_356263


namespace complex_roots_problem_l3562_356212

theorem complex_roots_problem (p q r : ℂ) : 
  p + q + r = 1 → p * q * r = 1 → p * q + p * r + q * r = 0 →
  (∃ (σ : Equiv.Perm (Fin 3)), 
    σ.1 0 = p ∧ σ.1 1 = q ∧ σ.1 2 = r ∧
    (∀ x, x^3 - x^2 - 1 = 0 ↔ (x = 2 ∨ x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2))) :=
by sorry

end complex_roots_problem_l3562_356212


namespace z_in_fourth_quadrant_l3562_356277

def z : ℂ := (1 + Complex.I) * (1 - 2 * Complex.I)

theorem z_in_fourth_quadrant : 
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
by sorry

end z_in_fourth_quadrant_l3562_356277


namespace evaluate_expression_l3562_356297

theorem evaluate_expression (x : ℝ) (h : x = -3) :
  Real.sqrt ((x - 1)^2) + Real.sqrt (x^2 + 4*x + 4) = 5 := by
  sorry

end evaluate_expression_l3562_356297


namespace sin_B_in_triangle_ABC_l3562_356240

theorem sin_B_in_triangle_ABC (a b : ℝ) (sin_A : ℝ) :
  a = 15 →
  b = 10 →
  sin_A = (Real.sqrt 3) / 2 →
  (b * sin_A) / a = (Real.sqrt 3) / 3 :=
sorry

end sin_B_in_triangle_ABC_l3562_356240


namespace sum_of_21st_set_l3562_356225

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The first element of the nth set -/
def first_element (n : ℕ) : ℕ := 1 + sum_first_n (n - 1)

/-- The last element of the nth set -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- The theorem to prove -/
theorem sum_of_21st_set : S 21 = 4641 := by sorry

end sum_of_21st_set_l3562_356225


namespace strictly_decreasing_implies_inequality_odd_function_property_l3562_356201

-- Define a function f from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Statement 1
theorem strictly_decreasing_implies_inequality (h : ∀ x y, x < y → f x > f y) : f (-4) > f 4 := by
  sorry

-- Statement 2
theorem odd_function_property (h : ∀ x, f (-x) = -f x) : f (-4) + f 4 = 0 := by
  sorry

end strictly_decreasing_implies_inequality_odd_function_property_l3562_356201


namespace new_person_weight_l3562_356288

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * average_increase

/-- Theorem stating that the weight of the new person is 80 kg -/
theorem new_person_weight :
  weight_of_new_person 6 2.5 65 = 80 := by
  sorry

end new_person_weight_l3562_356288


namespace z_in_second_quadrant_l3562_356257

def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem z_in_second_quadrant (z : ℂ) (h : z * Complex.I = -1 - Complex.I) :
  is_in_second_quadrant z := by
  sorry

end z_in_second_quadrant_l3562_356257


namespace sum_of_roots_quadratic_l3562_356247

theorem sum_of_roots_quadratic (x : ℝ) (h : x^2 - 3*x = 12) : 
  ∃ y : ℝ, y^2 - 3*y = 12 ∧ x + y = 3 := by
sorry

end sum_of_roots_quadratic_l3562_356247


namespace winnings_proof_l3562_356249

theorem winnings_proof (total : ℝ) 
  (h1 : total > 0)
  (h2 : total / 4 + total / 7 + 17 = total) : 
  total = 28 := by
sorry

end winnings_proof_l3562_356249


namespace smallest_block_volume_l3562_356216

theorem smallest_block_volume (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 120 → 
  l * m * n ≥ 216 :=
by sorry

end smallest_block_volume_l3562_356216


namespace power_function_through_point_l3562_356227

/-- Given a power function f(x) = x^n that passes through (2, √2), prove f(9) = 3 -/
theorem power_function_through_point (f : ℝ → ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 2 = Real.sqrt 2 →
  f 9 = 3 := by
  sorry

end power_function_through_point_l3562_356227


namespace rational_roots_count_l3562_356287

/-- A polynomial with integer coefficients of the form 9x^4 + a₃x³ + a₂x² + a₁x + 15 = 0 -/
def IntPolynomial (a₃ a₂ a₁ : ℤ) (x : ℚ) : ℚ :=
  9 * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + 15

/-- The set of possible rational roots of the polynomial -/
def PossibleRoots : Finset ℚ :=
  {1, -1, 3, -3, 5, -5, 15, -15, 1/3, -1/3, 5/3, -5/3, 1/9, -1/9, 5/9, -5/9}

theorem rational_roots_count (a₃ a₂ a₁ : ℤ) :
  (PossibleRoots.filter (fun x => IntPolynomial a₃ a₂ a₁ x = 0)).card ≤ 16 :=
sorry

end rational_roots_count_l3562_356287


namespace quadratic_root_k_l3562_356283

theorem quadratic_root_k (k : ℝ) : (1 : ℝ)^2 + k * 1 - 3 = 0 → k = 2 := by
  sorry

end quadratic_root_k_l3562_356283


namespace flower_bed_path_area_l3562_356273

/-- The area of a circular ring around a flower bed -/
theorem flower_bed_path_area (circumference : Real) (path_width : Real) : 
  circumference = 314 → path_width = 2 →
  let inner_radius := circumference / (2 * Real.pi)
  let outer_radius := inner_radius + path_width
  abs (Real.pi * (outer_radius^2 - inner_radius^2) - 640.56) < 0.01 := by
sorry

end flower_bed_path_area_l3562_356273


namespace sum_of_fourth_powers_l3562_356208

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_condition : a + b + c = 2)
  (sum_squares : a^2 + b^2 + c^2 = 5)
  (sum_cubes : a^3 + b^3 + c^3 = 10) :
  a^4 + b^4 + c^4 = 68/3 := by
sorry

end sum_of_fourth_powers_l3562_356208


namespace circumcenter_rational_l3562_356205

-- Define a triangle with rational coordinates
structure RationalTriangle where
  a : ℚ × ℚ
  b : ℚ × ℚ
  c : ℚ × ℚ

-- Define the center of the circumscribed circle
def circumcenter (t : RationalTriangle) : ℚ × ℚ :=
  sorry

-- Theorem statement
theorem circumcenter_rational (t : RationalTriangle) :
  ∃ (x y : ℚ), circumcenter t = (x, y) :=
sorry

end circumcenter_rational_l3562_356205


namespace sin_120_degrees_l3562_356289

theorem sin_120_degrees : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_120_degrees_l3562_356289


namespace complex_fraction_simplification_l3562_356230

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * I
  let z₂ : ℂ := 4 - 7 * I
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 := by sorry

end complex_fraction_simplification_l3562_356230


namespace green_blue_difference_l3562_356276

/-- Represents the color of a disk -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  blue : ℕ
  yellow : ℕ
  green : ℕ
  color_sum : blue + yellow + green = total
  ratio : ∃ (k : ℕ), blue = 3 * k ∧ yellow = 7 * k ∧ green = 8 * k

/-- The main theorem to prove -/
theorem green_blue_difference (bag : DiskBag) 
  (h_total : bag.total = 108) :
  bag.green - bag.blue = 30 := by
  sorry

#check green_blue_difference

end green_blue_difference_l3562_356276


namespace store_discount_problem_l3562_356262

/-- Represents the store discount problem --/
theorem store_discount_problem (shirt_price : ℝ) (shirt_count : ℕ)
                                (pants_price : ℝ) (pants_count : ℕ)
                                (suit_price : ℝ)
                                (sweater_price : ℝ) (sweater_count : ℕ)
                                (coupon_discount : ℝ)
                                (final_price : ℝ) :
  shirt_price = 15 →
  shirt_count = 4 →
  pants_price = 40 →
  pants_count = 2 →
  suit_price = 150 →
  sweater_price = 30 →
  sweater_count = 2 →
  coupon_discount = 0.1 →
  final_price = 252 →
  ∃ (store_discount : ℝ),
    store_discount = 0.2 ∧
    final_price = (shirt_price * shirt_count +
                   pants_price * pants_count +
                   suit_price +
                   sweater_price * sweater_count) *
                  (1 - store_discount) *
                  (1 - coupon_discount) := by
  sorry

end store_discount_problem_l3562_356262


namespace teachers_count_l3562_356252

/-- Given a school with girls, boys, and teachers, calculates the number of teachers. -/
def calculate_teachers (girls boys total : ℕ) : ℕ :=
  total - (girls + boys)

/-- Proves that there are 772 teachers in a school with 315 girls, 309 boys, and 1396 people in total. -/
theorem teachers_count : calculate_teachers 315 309 1396 = 772 := by
  sorry

end teachers_count_l3562_356252


namespace distance_to_midpoint_zero_l3562_356250

theorem distance_to_midpoint_zero (x₁ y₁ x₂ y₂ : ℝ) 
  (h1 : x₁ = 10 ∧ y₁ = 20) 
  (h2 : x₂ = -10 ∧ y₂ = -20) : 
  Real.sqrt (((x₁ + x₂) / 2)^2 + ((y₁ + y₂) / 2)^2) = 0 := by
  sorry

end distance_to_midpoint_zero_l3562_356250


namespace unique_prime_sum_10003_l3562_356266

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_of_two_primes (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = n

theorem unique_prime_sum_10003 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10003 :=
sorry

end unique_prime_sum_10003_l3562_356266


namespace marias_school_students_l3562_356282

theorem marias_school_students (m d : ℕ) : 
  m = 4 * d → 
  m - d = 1800 → 
  m = 2400 := by
sorry

end marias_school_students_l3562_356282


namespace simple_interest_problem_l3562_356236

theorem simple_interest_problem (principal rate time : ℝ) : 
  principal = 2100 →
  principal * (rate + 1) * time / 100 = principal * rate * time / 100 + 63 →
  time = 3 := by
sorry

end simple_interest_problem_l3562_356236


namespace bug_return_probability_l3562_356269

/-- Probability of returning to the starting vertex after n steps -/
def Q (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1/4 : ℚ) + (1/2 : ℚ) * Q (n-1)

/-- Regular tetrahedron with bug movement rules -/
theorem bug_return_probability :
  Q 6 = 354/729 := by sorry

end bug_return_probability_l3562_356269


namespace complex_expression_equality_combinatorial_equality_l3562_356256

-- Part I
theorem complex_expression_equality : 
  (((Complex.abs (1 - Complex.I)) / Real.sqrt 2) ^ 16 + 
   ((1 + 2 * Complex.I) ^ 2) / (1 - Complex.I)) = 
  (-5 / 2 : ℂ) + (1 / 2 : ℂ) * Complex.I := by sorry

-- Part II
theorem combinatorial_equality (m : ℕ) : 
  (1 / Nat.choose 5 m : ℚ) - (1 / Nat.choose 6 m : ℚ) = 
  (7 : ℚ) / (10 * Nat.choose 7 m) → 
  Nat.choose 8 m = 28 := by sorry

end complex_expression_equality_combinatorial_equality_l3562_356256


namespace ap_num_terms_l3562_356206

/-- Represents an arithmetic progression with an even number of terms. -/
structure ArithmeticProgression where
  n : ℕ                   -- Number of terms
  a : ℚ                   -- First term
  d : ℚ                   -- Common difference
  n_even : Even n         -- n is even
  last_minus_first : a + (n - 1) * d - a = 16  -- Last term exceeds first by 16

/-- The sum of odd-numbered terms in the arithmetic progression. -/
def sum_odd_terms (ap : ArithmeticProgression) : ℚ :=
  (ap.n / 2 : ℚ) * (2 * ap.a + (ap.n - 2) * ap.d)

/-- The sum of even-numbered terms in the arithmetic progression. -/
def sum_even_terms (ap : ArithmeticProgression) : ℚ :=
  (ap.n / 2 : ℚ) * (2 * ap.a + 2 * ap.d + (ap.n - 2) * ap.d)

/-- Theorem stating the conditions and conclusion about the number of terms. -/
theorem ap_num_terms (ap : ArithmeticProgression) 
  (h_odd : sum_odd_terms ap = 81)
  (h_even : sum_even_terms ap = 75) : 
  ap.n = 8 := by sorry

end ap_num_terms_l3562_356206


namespace calculate_fifth_subject_score_l3562_356299

/-- Given a student's scores in 4 subjects and the average of all 5 subjects,
    calculate the score in the 5th subject. -/
theorem calculate_fifth_subject_score
  (math_score science_score english_score biology_score : ℕ)
  (average_score : ℚ)
  (h1 : math_score = 76)
  (h2 : science_score = 65)
  (h3 : english_score = 62)
  (h4 : biology_score = 85)
  (h5 : average_score = 74)
  : ∃ (social_studies_score : ℕ),
    (math_score + science_score + english_score + biology_score + social_studies_score : ℚ) / 5 = average_score ∧
    social_studies_score = 82 :=
by
  sorry

end calculate_fifth_subject_score_l3562_356299


namespace circular_permutations_count_l3562_356295

/-- The number of elements of type 'a' -/
def num_a : ℕ := 2

/-- The number of elements of type 'b' -/
def num_b : ℕ := 2

/-- The number of elements of type 'c' -/
def num_c : ℕ := 4

/-- The total number of elements -/
def total_elements : ℕ := num_a + num_b + num_c

/-- First-class circular permutations -/
def first_class_permutations : ℕ := 52

/-- Second-class circular permutations -/
def second_class_permutations : ℕ := 33

theorem circular_permutations_count :
  (first_class_permutations = 52) ∧ (second_class_permutations = 33) := by
  sorry

end circular_permutations_count_l3562_356295


namespace first_digit_change_largest_l3562_356214

def original_number : ℚ := 0.12345678

def change_digit (n : ℚ) (position : ℕ) : ℚ :=
  n + (9 - (n * 10^position % 10)) / 10^position

theorem first_digit_change_largest :
  ∀ position : ℕ, position > 0 → 
    change_digit original_number 1 ≥ change_digit original_number position :=
by sorry

end first_digit_change_largest_l3562_356214


namespace milk_remaining_l3562_356237

theorem milk_remaining (initial : ℚ) (given_away : ℚ) (remaining : ℚ) : 
  initial = 5 → given_away = 18/7 → remaining = initial - given_away → remaining = 17/7 := by
  sorry

end milk_remaining_l3562_356237


namespace largest_corner_sum_l3562_356290

-- Define the face values of the cube
def face_values : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the property that opposite faces sum to 9
def opposite_sum_9 (faces : List ℕ) : Prop :=
  ∀ x ∈ faces, (9 - x) ∈ faces

-- Define a function to check if three numbers can be on adjacent faces
def can_be_adjacent (a b c : ℕ) : Prop :=
  a + b ≠ 9 ∧ b + c ≠ 9 ∧ a + c ≠ 9

-- Theorem statement
theorem largest_corner_sum :
  ∀ (cube : List ℕ),
  cube = face_values →
  opposite_sum_9 cube →
  (∃ (a b c : ℕ),
    a ∈ cube ∧ b ∈ cube ∧ c ∈ cube ∧
    can_be_adjacent a b c ∧
    (∀ (x y z : ℕ),
      x ∈ cube → y ∈ cube → z ∈ cube →
      can_be_adjacent x y z →
      x + y + z ≤ a + b + c)) →
  (∃ (a b c : ℕ),
    a ∈ cube ∧ b ∈ cube ∧ c ∈ cube ∧
    can_be_adjacent a b c ∧
    a + b + c = 18) :=
by
  sorry

end largest_corner_sum_l3562_356290


namespace loss_percent_calculation_l3562_356229

def cost_price : ℝ := 600
def selling_price : ℝ := 550

theorem loss_percent_calculation :
  let loss := cost_price - selling_price
  let loss_percent := (loss / cost_price) * 100
  ∃ ε > 0, abs (loss_percent - 8.33) < ε :=
by sorry

end loss_percent_calculation_l3562_356229


namespace expected_interval_is_three_l3562_356207

/-- Represents the train system with given conditions --/
structure TrainSystem where
  northern_route_time : ℝ
  southern_route_time : ℝ
  arrival_time_difference : ℝ
  commute_time_difference : ℝ

/-- The expected interval between trains in one direction --/
def expected_interval (ts : TrainSystem) : ℝ := 3

/-- Theorem stating that the expected interval is 3 minutes given the conditions --/
theorem expected_interval_is_three (ts : TrainSystem) 
  (h1 : ts.northern_route_time = 17)
  (h2 : ts.southern_route_time = 11)
  (h3 : ts.arrival_time_difference = 1.25)
  (h4 : ts.commute_time_difference = 1) :
  expected_interval ts = 3 := by
  sorry

#check expected_interval_is_three

end expected_interval_is_three_l3562_356207


namespace f_expression_for_x_less_than_2_l3562_356200

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_expression_for_x_less_than_2
  (f : ℝ → ℝ)
  (h1 : is_even_function (λ x ↦ f (x + 2)))
  (h2 : ∀ x ≥ 2, f x = 3^x - 1) :
  ∀ x < 2, f x = 3^(4 - x) - 1 := by
  sorry

end f_expression_for_x_less_than_2_l3562_356200


namespace largest_divisor_of_product_l3562_356281

theorem largest_divisor_of_product (n : ℕ) (h : Even n) (h' : n > 0) :
  (∃ (k : ℕ), (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) = 15 * k) ∧
  (∀ (d : ℕ), d > 15 → ∃ (m : ℕ), Even m ∧ m > 0 ∧
    ¬(∃ (k : ℕ), (m + 3) * (m + 5) * (m + 7) * (m + 9) * (m + 11) = d * k)) :=
by sorry

end largest_divisor_of_product_l3562_356281


namespace square_plus_minus_one_divisible_by_five_l3562_356221

theorem square_plus_minus_one_divisible_by_five (n : ℤ) (h : ¬ 5 ∣ n) : 
  5 ∣ (n^2 + 1) ∨ 5 ∣ (n^2 - 1) := by
  sorry

end square_plus_minus_one_divisible_by_five_l3562_356221


namespace sum_of_max_and_min_g_l3562_356286

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 5| + |x - 3| - |3*x - 15|

-- Define the domain
def domain : Set ℝ := { x | 3 ≤ x ∧ x ≤ 10 }

-- Theorem statement
theorem sum_of_max_and_min_g :
  ∃ (max_g min_g : ℝ),
    (∀ x ∈ domain, g x ≤ max_g) ∧
    (∃ x ∈ domain, g x = max_g) ∧
    (∀ x ∈ domain, min_g ≤ g x) ∧
    (∃ x ∈ domain, g x = min_g) ∧
    (max_g + min_g = -2) :=
by sorry

end sum_of_max_and_min_g_l3562_356286


namespace max_value_of_f_l3562_356253

/-- The quadratic function f(x) = -5x^2 + 25x - 7 -/
def f (x : ℝ) : ℝ := -5 * x^2 + 25 * x - 7

/-- The maximum value of f(x) is 53/4 -/
theorem max_value_of_f :
  ∃ (M : ℝ), M = 53/4 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l3562_356253


namespace range_of_a_for_max_and_min_l3562_356270

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*(a + 2)*x + 1

/-- The theorem stating the range of a for which f has both a maximum and a minimum -/
theorem range_of_a_for_max_and_min (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) → (a > 2 ∨ a < -1) :=
sorry

end range_of_a_for_max_and_min_l3562_356270


namespace log_sum_equation_l3562_356243

theorem log_sum_equation (y : ℝ) (h : y > 0) :
  Real.log y / Real.log 3 + Real.log y / Real.log 9 = 5 →
  y = 3 ^ (10 / 3) := by
sorry

end log_sum_equation_l3562_356243


namespace smallest_with_12_divisors_l3562_356275

/-- The number of positive integer divisors of n -/
def numDivisors (n : ℕ) : ℕ := sorry

/-- n has exactly 12 positive integer divisors -/
def has12Divisors (n : ℕ) : Prop :=
  numDivisors n = 12

/-- 72 is the smallest positive integer with exactly 12 positive integer divisors -/
theorem smallest_with_12_divisors :
  has12Divisors 72 ∧ ∀ m : ℕ, 0 < m → m < 72 → ¬has12Divisors m := by sorry

end smallest_with_12_divisors_l3562_356275


namespace obtuse_triangle_side_range_l3562_356215

theorem obtuse_triangle_side_range (x : ℝ) : 
  (x > 0 ∧ x + 1 > 0 ∧ x + 2 > 0) →  -- Positive side lengths
  (x + (x + 1) > (x + 2) ∧ (x + 2) + x > (x + 1) ∧ (x + 2) + (x + 1) > x) →  -- Triangle inequality
  ((x + 2)^2 > x^2 + (x + 1)^2) →  -- Obtuse triangle condition
  (1 < x ∧ x < 3) :=
by sorry

end obtuse_triangle_side_range_l3562_356215


namespace remainder_problem_l3562_356264

theorem remainder_problem (h1 : Nat.Prime 73) (h2 : ¬(73 ∣ 57)) :
  (57^35 + 47) % 73 = 55 := by
  sorry

end remainder_problem_l3562_356264


namespace payment_difference_equation_l3562_356254

/-- Represents the payment structure for two artists painting murals. -/
structure MuralPayment where
  diego : ℝ  -- Diego's payment
  celina : ℝ  -- Celina's payment
  total : ℝ   -- Total payment
  h1 : celina > 4 * diego  -- Celina's payment is more than 4 times Diego's
  h2 : celina + diego = total  -- Sum of payments equals total

/-- The difference between Celina's payment and 4 times Diego's payment. -/
def payment_difference (p : MuralPayment) : ℝ := p.celina - 4 * p.diego

/-- Theorem stating the relationship between the payment difference and Diego's payment. -/
theorem payment_difference_equation (p : MuralPayment) (h3 : p.total = 50000) :
  payment_difference p = 50000 - 5 * p.diego := by
  sorry


end payment_difference_equation_l3562_356254


namespace bob_raised_beds_l3562_356268

/-- Represents the dimensions of a raised bed -/
structure BedDimensions where
  height : ℕ
  width : ℕ
  length : ℕ

/-- Calculates the number of planks needed for one raised bed -/
def planksPerBed (dims : BedDimensions) (plankWidth : ℕ) : ℕ :=
  2 * dims.height * (dims.length / plankWidth) + 1

/-- Calculates the number of raised beds that can be constructed -/
def numberOfBeds (dims : BedDimensions) (plankWidth : ℕ) (totalPlanks : ℕ) : ℕ :=
  totalPlanks / planksPerBed dims plankWidth

/-- Theorem: Bob can construct 10 raised beds -/
theorem bob_raised_beds :
  let dims : BedDimensions := { height := 2, width := 2, length := 8 }
  let plankWidth := 1
  let totalPlanks := 50
  numberOfBeds dims plankWidth totalPlanks = 10 := by
  sorry

end bob_raised_beds_l3562_356268


namespace roots_transformation_l3562_356259

theorem roots_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 4*r₁^2 + 9 = 0) ∧ 
  (r₂^3 - 4*r₂^2 + 9 = 0) ∧ 
  (r₃^3 - 4*r₃^2 + 9 = 0) →
  ((3*r₁)^3 - 12*(3*r₁)^2 + 243 = 0) ∧
  ((3*r₂)^3 - 12*(3*r₂)^2 + 243 = 0) ∧
  ((3*r₃)^3 - 12*(3*r₃)^2 + 243 = 0) := by
sorry

end roots_transformation_l3562_356259


namespace min_rectangles_correct_l3562_356220

/-- The minimum number of rectangles needed to cover a board -/
def min_rectangles (n : ℕ) : ℕ := 2 * n

/-- A rectangle with integer side lengths and area equal to a power of 2 -/
structure PowerRect where
  width : ℕ
  height : ℕ
  is_power_of_two : ∃ k : ℕ, width * height = 2^k

/-- A covering of the board with rectangles -/
structure BoardCovering (n : ℕ) where
  rectangles : List PowerRect
  covers_board : (List.sum (rectangles.map (λ r => r.width * r.height))) = (2^n - 1) * (2^n + 1)

theorem min_rectangles_correct (n : ℕ) :
  ∀ (cover : BoardCovering n), cover.rectangles.length ≥ min_rectangles n ∧
  ∃ (optimal_cover : BoardCovering n), optimal_cover.rectangles.length = min_rectangles n :=
sorry

end min_rectangles_correct_l3562_356220


namespace green_peaches_count_l3562_356255

theorem green_peaches_count (red : ℕ) (yellow : ℕ) (total : ℕ) (green : ℕ) : 
  red = 7 → yellow = 15 → total = 30 → green = total - (red + yellow) → green = 8 := by
sorry

end green_peaches_count_l3562_356255


namespace rectangle_perimeter_l3562_356284

theorem rectangle_perimeter (square_perimeter : ℝ) (h1 : square_perimeter = 16) :
  let square_side := square_perimeter / 4
  let rectangle_side := 2 * square_side
  rectangle_side * 4 = 32 := by
sorry

end rectangle_perimeter_l3562_356284


namespace repeating_decimal_eq_l3562_356228

/-- The repeating decimal 0.565656... expressed as a rational number -/
def repeating_decimal : ℚ := 56 / 99

/-- The theorem stating that the repeating decimal 0.565656... equals 56/99 -/
theorem repeating_decimal_eq : repeating_decimal = 56 / 99 := by
  sorry

end repeating_decimal_eq_l3562_356228


namespace light_wattage_increase_l3562_356246

theorem light_wattage_increase (original_wattage new_wattage : ℝ) 
  (h1 : original_wattage = 80)
  (h2 : new_wattage = 100) :
  (new_wattage - original_wattage) / original_wattage * 100 = 25 := by
  sorry

end light_wattage_increase_l3562_356246


namespace nearest_integer_to_3_plus_sqrt2_power_5_l3562_356210

theorem nearest_integer_to_3_plus_sqrt2_power_5 :
  ∃ n : ℤ, n = 1926 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^5 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^5 - (m : ℝ)| :=
sorry

end nearest_integer_to_3_plus_sqrt2_power_5_l3562_356210


namespace line_tangent_to_circle_l3562_356238

/-- A line is tangent to a circle if and only if the discriminant of the resulting quadratic equation is zero -/
theorem line_tangent_to_circle (a b : ℤ) : 
  (∃ x y : ℝ, y - 1 = (4 - a*x - b) / b ∧ 
              b^2*(x-1)^2 + (a*x+b-4)^2 - b^2 = 0 ∧ 
              (a*b - 4*a - b^2)^2 = (a^2 + b^2)*(b - 4)^2) ↔ 
  ((a = 12 ∧ b = 5) ∨ (a = -4 ∧ b = 3) ∨ (a = 8 ∧ b = 6) ∨ 
   (a = 0 ∧ b = 2) ∨ (a = 6 ∧ b = 8) ∨ (a = 2 ∧ b = 0) ∨ 
   (a = 5 ∧ b = 12) ∨ (a = 3 ∧ b = -4)) :=
sorry

end line_tangent_to_circle_l3562_356238


namespace max_value_nonnegative_inequality_condition_l3562_356272

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + a

theorem max_value_nonnegative (a : ℝ) :
  ∀ x₀ : ℝ, (∀ x : ℝ, f a x ≤ f a x₀) → f a x₀ ≥ 0 := by sorry

theorem inequality_condition (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x + Real.exp (x - 1) ≥ 1) ↔ a ≤ 2 := by sorry

end max_value_nonnegative_inequality_condition_l3562_356272


namespace no_arithmetic_progression_with_product_l3562_356260

theorem no_arithmetic_progression_with_product : ¬∃ (a b : ℝ), 
  (b - a = a - 5) ∧ (a * b - b = b - a) := by
  sorry

end no_arithmetic_progression_with_product_l3562_356260


namespace min_distance_ellipse_to_line_l3562_356232

noncomputable section

-- Define the ellipse C₁
def C₁ (x y : ℝ) : Prop := y^2 / 3 + x^2 = 1

-- Define the line C₂
def C₂ (x y : ℝ) : Prop := x - y = 4

-- Define the distance function between a point (x, y) and the line C₂
def distance_to_C₂ (x y : ℝ) : ℝ := |x - y - 4| / Real.sqrt 2

-- State the theorem
theorem min_distance_ellipse_to_line :
  ∃ (α : ℝ), 
    let x := Real.sin α
    let y := Real.sqrt 3 * Real.cos α
    C₁ x y ∧ 
    (∀ β : ℝ, distance_to_C₂ (Real.sin β) (Real.sqrt 3 * Real.cos β) ≥ Real.sqrt 2) ∧
    distance_to_C₂ x y = Real.sqrt 2 ∧
    x = 1/2 ∧ y = -3/2 :=
sorry

end min_distance_ellipse_to_line_l3562_356232


namespace circle_and_locus_equations_l3562_356222

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (a : ℝ), (x - a)^2 + (y - (2 - a))^2 = (a - 4)^2 + (2 - a)^2 ∧
              (x - a)^2 + (y - (2 - a))^2 = (a - 2)^2 + (2 - a - 2)^2

-- Define the locus of midpoint M
def locus_M (x y : ℝ) : Prop :=
  ∃ (x₁ y₁ : ℝ), circle_C x₁ y₁ ∧ x = (x₁ + 5) / 2 ∧ y = y₁ / 2

theorem circle_and_locus_equations :
  (∀ x y, circle_C x y ↔ (x - 2)^2 + y^2 = 4) ∧
  (∀ x y, locus_M x y ↔ x^2 - 7*x + y^2 + 45/4 = 0) :=
sorry

end circle_and_locus_equations_l3562_356222


namespace simplify_fraction_l3562_356267

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 48 + Real.sqrt 147) = 5 * Real.sqrt 3 / 72 := by
sorry

end simplify_fraction_l3562_356267


namespace problem_statement_l3562_356274

theorem problem_statement (a b : ℚ) 
  (h1 : 3 * a + 4 * b = 0) 
  (h2 : a = 2 * b - 3) : 
  5 * b = 9 / 2 := by
sorry

end problem_statement_l3562_356274


namespace min_shoeing_time_for_scenario_l3562_356218

/-- The minimum time needed for blacksmiths to shoe horses -/
def min_shoeing_time (blacksmiths horses : ℕ) (time_per_shoe : ℕ) : ℕ :=
  let total_shoes := horses * 4
  let total_time := total_shoes * time_per_shoe
  (total_time + blacksmiths - 1) / blacksmiths

/-- Theorem stating the minimum time needed for the given scenario -/
theorem min_shoeing_time_for_scenario :
  min_shoeing_time 48 60 5 = 25 := by
sorry

#eval min_shoeing_time 48 60 5

end min_shoeing_time_for_scenario_l3562_356218


namespace inequality_one_l3562_356233

theorem inequality_one (x y : ℝ) : (x + 1) * (x - 2*y + 1) + y^2 ≥ 0 := by
  sorry

end inequality_one_l3562_356233


namespace unpainted_area_specific_case_l3562_356292

/-- Represents the configuration of two crossed boards -/
structure CrossedBoards where
  width1 : ℝ
  width2 : ℝ
  angle : ℝ

/-- Calculates the area of the unpainted region on the first board -/
def unpainted_area (boards : CrossedBoards) : ℝ :=
  boards.width1 * boards.width2

/-- Theorem stating the area of the unpainted region for specific board widths and angle -/
theorem unpainted_area_specific_case :
  let boards : CrossedBoards := ⟨5, 7, 45⟩
  unpainted_area boards = 35 := by sorry

end unpainted_area_specific_case_l3562_356292


namespace square_area_l3562_356293

theorem square_area (x : ℝ) : 
  (5 * x - 10 = 3 * (x + 4)) → 
  (5 * x - 10)^2 = 2025 := by
  sorry

end square_area_l3562_356293


namespace ticket_price_is_28_l3562_356285

/-- The price of a single ticket given the total money, number of tickets, and remaining money -/
def ticket_price (total_money : ℕ) (num_tickets : ℕ) (remaining_money : ℕ) : ℕ :=
  (total_money - remaining_money) / num_tickets

/-- Theorem stating that the ticket price is $28 given the problem conditions -/
theorem ticket_price_is_28 :
  ticket_price 251 6 83 = 28 := by
  sorry

end ticket_price_is_28_l3562_356285


namespace arithmetic_mean_problem_l3562_356219

theorem arithmetic_mean_problem (x : ℝ) : 
  (10 + 20 + 60) / 3 = (10 + 40 + x) / 3 + 5 → x = 25 := by
  sorry

end arithmetic_mean_problem_l3562_356219


namespace sphere_cylinder_volume_difference_l3562_356261

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) (h_sphere : r_sphere = 7) (h_cylinder : r_cylinder = 4) :
  let v_sphere := (4 / 3) * Real.pi * r_sphere ^ 3
  let h_cylinder := 2 * r_sphere
  let v_cylinder := Real.pi * r_cylinder ^ 2 * h_cylinder
  v_sphere - v_cylinder = (700 / 3) * Real.pi :=
by sorry

end sphere_cylinder_volume_difference_l3562_356261


namespace percent_profit_calculation_l3562_356209

theorem percent_profit_calculation (C S : ℝ) (h : 55 * C = 50 * S) : 
  (S - C) / C * 100 = 10 := by
  sorry

end percent_profit_calculation_l3562_356209
