import Mathlib

namespace NUMINAMATH_CALUDE_complex_magnitude_l3916_391643

theorem complex_magnitude (z : ℂ) (h : z * Complex.I = 2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3916_391643


namespace NUMINAMATH_CALUDE_min_sum_abs_min_sum_abs_achieved_l3916_391661

theorem min_sum_abs (x : ℝ) : 
  |x + 3| + |x + 4| + |x + 6| + |x + 8| ≥ 12 :=
by sorry

theorem min_sum_abs_achieved : 
  ∃ x : ℝ, |x + 3| + |x + 4| + |x + 6| + |x + 8| = 12 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_abs_min_sum_abs_achieved_l3916_391661


namespace NUMINAMATH_CALUDE_cristinas_pace_l3916_391614

/-- Prove Cristina's pace in a race with given conditions -/
theorem cristinas_pace (race_distance : ℝ) (head_start : ℝ) (nickys_pace : ℝ) (catch_up_time : ℝ)
  (h1 : race_distance = 500)
  (h2 : head_start = 12)
  (h3 : nickys_pace = 3)
  (h4 : catch_up_time = 30) :
  let cristinas_distance := nickys_pace * (head_start + catch_up_time)
  cristinas_distance / catch_up_time = 5.4 := by
  sorry

#check cristinas_pace

end NUMINAMATH_CALUDE_cristinas_pace_l3916_391614


namespace NUMINAMATH_CALUDE_activity_ranking_l3916_391668

def fishing_popularity : ℚ := 13/36
def hiking_popularity : ℚ := 8/27
def painting_popularity : ℚ := 7/18

theorem activity_ranking :
  painting_popularity > fishing_popularity ∧
  fishing_popularity > hiking_popularity := by
  sorry

end NUMINAMATH_CALUDE_activity_ranking_l3916_391668


namespace NUMINAMATH_CALUDE_percentage_loss_is_twenty_percent_l3916_391621

/-- Calculates the percentage loss given the selling conditions --/
def calculate_percentage_loss (initial_articles : ℕ) (initial_price : ℚ) (initial_gain_percent : ℚ) 
  (final_articles : ℚ) (final_price : ℚ) : ℚ :=
  let initial_cost := initial_price / (1 + initial_gain_percent / 100)
  let cost_per_article := initial_cost / initial_articles
  let final_cost := cost_per_article * final_articles
  let loss := final_cost - final_price
  (loss / final_cost) * 100

/-- The percentage loss is 20% given the specified conditions --/
theorem percentage_loss_is_twenty_percent :
  calculate_percentage_loss 20 60 20 20 40 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_loss_is_twenty_percent_l3916_391621


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3916_391629

theorem sum_of_two_numbers (smaller larger : ℕ) : 
  smaller = 31 → larger = 3 * smaller → smaller + larger = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3916_391629


namespace NUMINAMATH_CALUDE_divisibility_inequality_l3916_391619

theorem divisibility_inequality (a b c d e f : ℕ) 
  (h_f_lt_a : f < a)
  (h_div_c : ∃ k : ℕ, a * b * d + 1 = k * c)
  (h_div_b : ∃ l : ℕ, a * c * e + 1 = l * b)
  (h_div_a : ∃ m : ℕ, b * c * f + 1 = m * a)
  (h_ineq : (d : ℚ) / c < 1 - (e : ℚ) / b) :
  (d : ℚ) / c < 1 - (f : ℚ) / a :=
by sorry

end NUMINAMATH_CALUDE_divisibility_inequality_l3916_391619


namespace NUMINAMATH_CALUDE_mod_seven_difference_powers_l3916_391609

theorem mod_seven_difference_powers : (81^1814 - 25^1814) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_mod_seven_difference_powers_l3916_391609


namespace NUMINAMATH_CALUDE_chess_piece_probability_l3916_391638

/-- The probability of drawing a red piece first and a green piece second from a bag of chess pieces -/
theorem chess_piece_probability (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 32) 
  (h2 : red = 16) 
  (h3 : green = 16) 
  (h4 : red + green = total) : 
  (red / total) * (green / (total - 1)) = 8 / 31 := by
  sorry

end NUMINAMATH_CALUDE_chess_piece_probability_l3916_391638


namespace NUMINAMATH_CALUDE_dinosaur_book_cost_l3916_391651

def dictionary_cost : ℕ := 5
def cookbook_cost : ℕ := 5
def saved_amount : ℕ := 19
def additional_needed : ℕ := 2

theorem dinosaur_book_cost :
  dictionary_cost + cookbook_cost + (saved_amount + additional_needed - (dictionary_cost + cookbook_cost)) = 11 := by
  sorry

end NUMINAMATH_CALUDE_dinosaur_book_cost_l3916_391651


namespace NUMINAMATH_CALUDE_forest_trees_l3916_391627

/-- Calculates the total number of trees in a forest given the conditions --/
theorem forest_trees (street_side : ℝ) (forest_area_multiplier : ℝ) (trees_per_sqm : ℝ) : 
  street_side = 100 →
  forest_area_multiplier = 3 →
  trees_per_sqm = 4 →
  (forest_area_multiplier * street_side^2 * trees_per_sqm : ℝ) = 120000 := by
  sorry

#check forest_trees

end NUMINAMATH_CALUDE_forest_trees_l3916_391627


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l3916_391691

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- State the condition that f(x) = f(4-x) for all x
axiom f_symmetry (x : ℝ) : f x = f (4 - x)

-- Define what it means for a line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

-- Theorem statement
theorem axis_of_symmetry :
  is_axis_of_symmetry 2 :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l3916_391691


namespace NUMINAMATH_CALUDE_min_sum_inequality_l3916_391637

theorem min_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (3 * b) + b / (6 * c) + c / (9 * a) ≥ 3 / Real.rpow 162 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_inequality_l3916_391637


namespace NUMINAMATH_CALUDE_max_ab_value_l3916_391653

/-- Two circles C₁ and C₂ -/
structure Circles where
  a : ℝ
  b : ℝ

/-- C₁: (x-a)² + (y+2)² = 4 -/
def C₁ (c : Circles) (x y : ℝ) : Prop :=
  (x - c.a)^2 + (y + 2)^2 = 4

/-- C₂: (x+b)² + (y+2)² = 1 -/
def C₂ (c : Circles) (x y : ℝ) : Prop :=
  (x + c.b)^2 + (y + 2)^2 = 1

/-- The circles are externally tangent -/
def externally_tangent (c : Circles) : Prop :=
  c.a + c.b = 3

/-- The maximum value of ab is 9/4 -/
theorem max_ab_value (c : Circles) (h : externally_tangent c) :
  c.a * c.b ≤ 9/4 ∧ ∃ (c' : Circles), externally_tangent c' ∧ c'.a * c'.b = 9/4 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l3916_391653


namespace NUMINAMATH_CALUDE_negative_reciprocal_equality_l3916_391636

theorem negative_reciprocal_equality (a b : ℝ) : 
  (-1 / a = 8) → (-1 / (-b) = 8) → a = b := by sorry

end NUMINAMATH_CALUDE_negative_reciprocal_equality_l3916_391636


namespace NUMINAMATH_CALUDE_oil_measurement_l3916_391673

theorem oil_measurement (initial_oil : ℚ) (added_oil : ℚ) (total_oil : ℚ) :
  initial_oil = 17/100 →
  added_oil = 67/100 →
  total_oil = initial_oil + added_oil →
  total_oil = 84/100 := by
sorry

end NUMINAMATH_CALUDE_oil_measurement_l3916_391673


namespace NUMINAMATH_CALUDE_equation_equivalence_l3916_391689

theorem equation_equivalence (x y : ℝ) : 
  (2 * x + y = 1) ↔ (y = 1 - 2 * x) := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3916_391689


namespace NUMINAMATH_CALUDE_sum_gcf_lcm_8_12_l3916_391684

theorem sum_gcf_lcm_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_gcf_lcm_8_12_l3916_391684


namespace NUMINAMATH_CALUDE_amc_12_score_problem_l3916_391602

theorem amc_12_score_problem (total_problems : Nat) (attempted_problems : Nat) 
  (correct_points : Nat) (incorrect_points : Nat) (unanswered_points : Nat) 
  (unanswered_count : Nat) (min_score : Nat) :
  total_problems = 30 →
  attempted_problems = 26 →
  correct_points = 7 →
  incorrect_points = 0 →
  unanswered_points = 1 →
  unanswered_count = 4 →
  min_score = 150 →
  ∃ (correct_count : Nat), 
    correct_count * correct_points + 
    (attempted_problems - correct_count) * incorrect_points + 
    unanswered_count * unanswered_points ≥ min_score ∧
    correct_count = 21 ∧
    ∀ (x : Nat), x < 21 → 
      x * correct_points + 
      (attempted_problems - x) * incorrect_points + 
      unanswered_count * unanswered_points < min_score :=
by sorry

end NUMINAMATH_CALUDE_amc_12_score_problem_l3916_391602


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3916_391665

theorem quadratic_equations_solutions :
  (∃ x : ℝ, 2 * x^2 - 2 * Real.sqrt 2 * x + 1 = 0 ∧ x = Real.sqrt 2 / 2) ∧
  (∃ x₁ x₂ : ℝ, x₁ * (2 * x₁ - 5) = 4 * x₁ - 10 ∧
                x₂ * (2 * x₂ - 5) = 4 * x₂ - 10 ∧
                x₁ = 5 / 2 ∧ x₂ = 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3916_391665


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3916_391652

theorem complex_magnitude_problem (x y : ℝ) (h : (5 : ℂ) - x * I = y + 1 - 3 * I) : 
  Complex.abs (x - y * I) = 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3916_391652


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l3916_391640

/-- The lateral surface area of a cone with base radius 1 and height √3 is 2π -/
theorem cone_lateral_surface_area : 
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (h^2 + r^2)
  let lateral_area : ℝ := π * r * l
  lateral_area = 2 * π := by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l3916_391640


namespace NUMINAMATH_CALUDE_yardley_snowfall_l3916_391631

/-- The total snowfall in Yardley throughout the day -/
def total_snowfall (early_morning late_morning afternoon evening : Real) : Real :=
  early_morning + late_morning + afternoon + evening

/-- Theorem: The total snowfall in Yardley is 1.22 inches -/
theorem yardley_snowfall :
  total_snowfall 0.12 0.24 0.5 0.36 = 1.22 := by
  sorry

end NUMINAMATH_CALUDE_yardley_snowfall_l3916_391631


namespace NUMINAMATH_CALUDE_solution_set_inequality_l3916_391642

theorem solution_set_inequality (a : ℝ) (h : (4 : ℝ)^a = 2^(a + 2)) :
  {x : ℝ | a^(2*x + 1) > a^(x - 1)} = {x : ℝ | x > -2} := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l3916_391642


namespace NUMINAMATH_CALUDE_staircase_steps_l3916_391620

theorem staircase_steps (x : ℤ) 
  (h1 : x % 2 = 1)
  (h2 : x % 3 = 2)
  (h3 : x % 4 = 3)
  (h4 : x % 5 = 4)
  (h5 : x % 6 = 5)
  (h6 : x % 7 = 0) :
  ∃ k : ℤ, x = 119 + 420 * k := by
sorry

end NUMINAMATH_CALUDE_staircase_steps_l3916_391620


namespace NUMINAMATH_CALUDE_boy_running_speed_l3916_391644

theorem boy_running_speed (side_length : ℝ) (time : ℝ) (speed : ℝ) : 
  side_length = 30 →
  time = 36 →
  speed = (4 * side_length / 1000) / (time / 3600) →
  speed = 12 := by
sorry

end NUMINAMATH_CALUDE_boy_running_speed_l3916_391644


namespace NUMINAMATH_CALUDE_age_ratio_problem_l3916_391664

/-- Represents the ratio of two numbers as a pair of integers -/
structure Ratio where
  num : Int
  den : Int
  pos : 0 < den

def Ratio.of (a b : Int) (h : 0 < b) : Ratio :=
  ⟨a, b, h⟩

theorem age_ratio_problem (rahul_future_age deepak_current_age : ℕ) 
  (h1 : rahul_future_age = 50)
  (h2 : deepak_current_age = 33) :
  ∃ (r : Ratio), r = Ratio.of 4 3 (by norm_num) ∧ 
    (rahul_future_age - 6 : ℚ) / deepak_current_age = r.num / r.den := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l3916_391664


namespace NUMINAMATH_CALUDE_sum_last_two_digits_modified_fibonacci_factorial_series_l3916_391687

def modifiedFibonacciFactorialSeries : List Nat := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 144]

def lastTwoDigits (n : Nat) : Nat :=
  n % 100

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem sum_last_two_digits_modified_fibonacci_factorial_series :
  (modifiedFibonacciFactorialSeries.map (λ x => lastTwoDigits (factorial x))).sum % 100 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_last_two_digits_modified_fibonacci_factorial_series_l3916_391687


namespace NUMINAMATH_CALUDE_quartic_equation_roots_l3916_391690

theorem quartic_equation_roots : 
  let f (x : ℝ) := 4*x^4 - 28*x^3 + 53*x^2 - 28*x + 4
  ∀ x : ℝ, f x = 0 ↔ x = 4 ∨ x = 2 ∨ x = (1/4 : ℝ) ∨ x = (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_quartic_equation_roots_l3916_391690


namespace NUMINAMATH_CALUDE_original_purchase_price_l3916_391695

/-- Represents the original purchase price of the pants -/
def purchase_price : ℝ := sorry

/-- Represents the original selling price of the pants -/
def selling_price : ℝ := sorry

/-- The markup is 25% of the selling price -/
axiom markup_condition : selling_price = purchase_price + 0.25 * selling_price

/-- The new selling price after 20% decrease -/
def new_selling_price : ℝ := 0.8 * selling_price

/-- The gross profit is $5.40 -/
axiom gross_profit_condition : new_selling_price - purchase_price = 5.40

/-- Theorem stating that the original purchase price is $81 -/
theorem original_purchase_price : purchase_price = 81 := by sorry

end NUMINAMATH_CALUDE_original_purchase_price_l3916_391695


namespace NUMINAMATH_CALUDE_train_platform_length_equality_l3916_391660

def train_speed : ℝ := 144  -- km/hr
def crossing_time : ℝ := 1  -- minute
def train_length : ℝ := 1200  -- meters

theorem train_platform_length_equality :
  let platform_length := train_speed * 1000 / 60 * crossing_time - train_length
  platform_length = train_length :=
by sorry

end NUMINAMATH_CALUDE_train_platform_length_equality_l3916_391660


namespace NUMINAMATH_CALUDE_ellipse_intersecting_line_fixed_point_l3916_391676

/-- An ellipse with center at origin and axes along coordinate axes -/
structure Ellipse where
  a : ℝ
  b : ℝ
  ha : a > 0
  hb : b > 0
  hne : a ≠ b

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in slope-intercept form -/
structure Line where
  k : ℝ
  t : ℝ

def Ellipse.standardEq (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

def Line.eq (l : Line) (p : Point) : Prop :=
  p.y = l.k * p.x + l.t

def tangentAt (e : Ellipse) (l : Line) (p : Point) : Prop :=
  Ellipse.standardEq e p ∧ Line.eq l p

def intersects (e : Ellipse) (l : Line) (a b : Point) : Prop :=
  Ellipse.standardEq e a ∧ Ellipse.standardEq e b ∧ Line.eq l a ∧ Line.eq l b

def circleDiameterPassesThrough (a b c : Point) : Prop :=
  (a.x - b.x) * (c.x - b.x) + (a.y - b.y) * (c.y - b.y) = 0

theorem ellipse_intersecting_line_fixed_point 
  (e : Ellipse) (l : Line) (p a b : Point) :
  e.a^2 = 3 →
  e.b^2 = 4 →
  p.x = 3/2 →
  p.y = 1 →
  tangentAt e { k := 2, t := 4 } p →
  (∃ a b, intersects e l a b ∧ 
    a ≠ b ∧ 
    a.x ≠ e.a ∧ a.x ≠ -e.a ∧ 
    b.x ≠ e.a ∧ b.x ≠ -e.a ∧
    circleDiameterPassesThrough a b { x := 0, y := 2 }) →
  l.eq { x := 0, y := 2/7 } :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersecting_line_fixed_point_l3916_391676


namespace NUMINAMATH_CALUDE_min_value_problem_l3916_391659

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  2/(a-1) + 1/(b-2) ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l3916_391659


namespace NUMINAMATH_CALUDE_season_games_l3916_391605

/-- The number of hockey games in a season -/
def total_games (games_per_month : ℕ) (season_length : ℕ) : ℕ :=
  games_per_month * season_length

/-- Proof that there are 450 hockey games in the season -/
theorem season_games : total_games 25 18 = 450 := by
  sorry

end NUMINAMATH_CALUDE_season_games_l3916_391605


namespace NUMINAMATH_CALUDE_inequality_solution_set_max_m_value_l3916_391685

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 2|
def g (m : ℝ) (x : ℝ) : ℝ := -|x + 3| + m

-- Theorem for the solution set of the inequality
theorem inequality_solution_set (a : ℝ) :
  (∀ x, f x + a - 1 > 0 ↔ 
    (a = 1 ∧ x ≠ 2) ∨
    (a > 1) ∨
    (a < 1 ∧ (x < a + 1 ∨ x > 3 - a))) :=
sorry

-- Theorem for the maximum value of m
theorem max_m_value :
  ∀ m, (∀ x, f x > g m x) ↔ m < 5 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_max_m_value_l3916_391685


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3916_391616

theorem quadratic_function_property (a b c : ℝ) :
  let f := fun x => a * x^2 + b * x + c
  (f 1 = f 3 ∧ f 1 > f 4) → (a < 0 ∧ 4 * a + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3916_391616


namespace NUMINAMATH_CALUDE_fifth_month_sale_is_6500_l3916_391646

/-- Calculates the sale in the fifth month given the sales for other months and the average -/
def fifth_month_sale (m1 m2 m3 m4 m6 avg : ℕ) : ℕ :=
  6 * avg - (m1 + m2 + m3 + m4 + m6)

/-- Theorem stating that the sale in the fifth month is 6500 -/
theorem fifth_month_sale_is_6500 :
  fifth_month_sale 6400 7000 6800 7200 5100 6500 = 6500 := by
  sorry

end NUMINAMATH_CALUDE_fifth_month_sale_is_6500_l3916_391646


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3916_391649

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- Formula for calculating the number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem stating that the number of diagonals in a regular decagon is 35 -/
theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3916_391649


namespace NUMINAMATH_CALUDE_emily_beads_count_l3916_391625

theorem emily_beads_count (beads_per_necklace : ℕ) (necklaces_made : ℕ) 
  (h1 : beads_per_necklace = 5) 
  (h2 : necklaces_made = 4) : 
  beads_per_necklace * necklaces_made = 20 := by
  sorry

end NUMINAMATH_CALUDE_emily_beads_count_l3916_391625


namespace NUMINAMATH_CALUDE_a_in_M_sufficient_not_necessary_for_a_in_N_l3916_391615

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {-1, 0, 1, 2}

theorem a_in_M_sufficient_not_necessary_for_a_in_N :
  (∀ a, a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by sorry

end NUMINAMATH_CALUDE_a_in_M_sufficient_not_necessary_for_a_in_N_l3916_391615


namespace NUMINAMATH_CALUDE_paper_length_proof_l3916_391697

/-- Given a rectangular sheet of paper with specific dimensions and margins,
    prove that the length of the sheet is 10 inches. -/
theorem paper_length_proof (paper_width : Real) (margin : Real) (picture_area : Real) :
  paper_width = 8.5 →
  margin = 1.5 →
  picture_area = 38.5 →
  ∃ (paper_length : Real),
    paper_length = 10 ∧
    picture_area = (paper_length - 2 * margin) * (paper_width - 2 * margin) :=
by sorry

end NUMINAMATH_CALUDE_paper_length_proof_l3916_391697


namespace NUMINAMATH_CALUDE_octal_subtraction_3456_1234_l3916_391607

/-- Represents a number in base 8 --/
def OctalNumber := List Nat

/-- Converts an octal number to its decimal representation --/
def octal_to_decimal (n : OctalNumber) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

/-- Subtracts two octal numbers --/
def octal_subtract (a b : OctalNumber) : OctalNumber :=
  sorry -- Actual implementation would go here

theorem octal_subtraction_3456_1234 :
  let a : OctalNumber := [6, 5, 4, 3]  -- 3456 in base 8
  let b : OctalNumber := [4, 3, 2, 1]  -- 1234 in base 8
  let result : OctalNumber := [2, 2, 2, 2]  -- 2222 in base 8
  octal_subtract a b = result := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_3456_1234_l3916_391607


namespace NUMINAMATH_CALUDE_unique_solution_l3916_391667

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x^2 * y - x * y^2 - 5*x + 5*y + 3 = 0 ∧
  x^3 * y - x * y^3 - 5*x^2 + 5*y^2 + 15 = 0

/-- The solution to the system of equations is unique and equal to (4, 1) -/
theorem unique_solution : ∃! p : ℝ × ℝ, system p.1 p.2 ∧ p = (4, 1) := by sorry

end NUMINAMATH_CALUDE_unique_solution_l3916_391667


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3916_391677

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 - 2*x = 0 ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3916_391677


namespace NUMINAMATH_CALUDE_parallel_condition_l3916_391647

/-- Two lines in the form of ax + by + c = 0 and dx + ey + f = 0 are parallel if and only if ae = bd -/
def are_parallel (a b c d e f : ℝ) : Prop := a * e = b * d

/-- The first line: ax + 2y - 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0

/-- The second line: x + (a + 1)y + 4 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

theorem parallel_condition (a : ℝ) :
  (a = -2 → are_parallel a 2 (-1) 1 (a + 1) 4) ∧
  (∃ b : ℝ, b ≠ -2 ∧ are_parallel b 2 (-1) 1 (b + 1) 4) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l3916_391647


namespace NUMINAMATH_CALUDE_ceiling_sqrt_900_l3916_391674

theorem ceiling_sqrt_900 : ⌈Real.sqrt 900⌉ = 30 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_900_l3916_391674


namespace NUMINAMATH_CALUDE_octal_53_to_decimal_l3916_391682

/-- Converts an octal digit to its decimal value -/
def octal_to_decimal (d : ℕ) : ℕ := d

/-- Converts a two-digit octal number to its decimal equivalent -/
def octal_2digit_to_decimal (d1 d0 : ℕ) : ℕ :=
  octal_to_decimal d1 * 8 + octal_to_decimal d0

/-- The decimal representation of the octal number 53 is 43 -/
theorem octal_53_to_decimal :
  octal_2digit_to_decimal 5 3 = 43 := by sorry

end NUMINAMATH_CALUDE_octal_53_to_decimal_l3916_391682


namespace NUMINAMATH_CALUDE_divisible_by_27_l3916_391656

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A natural number is three times the sum of its digits -/
def is_three_times_sum_of_digits (n : ℕ) : Prop :=
  n = 3 * sum_of_digits n

theorem divisible_by_27 (n : ℕ) (h : is_three_times_sum_of_digits n) : 
  27 ∣ n := by sorry

end NUMINAMATH_CALUDE_divisible_by_27_l3916_391656


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l3916_391641

-- Define a triangle type
structure Triangle where
  base : ℝ
  median1 : ℝ
  median2 : ℝ

-- Define the area function
def triangleArea (t : Triangle) : ℝ :=
  sorry  -- The actual calculation would go here

-- Theorem statement
theorem triangle_area_theorem (t : Triangle) 
  (h1 : t.base = 20)
  (h2 : t.median1 = 18)
  (h3 : t.median2 = 24) : 
  triangleArea t = 288 := by
  sorry


end NUMINAMATH_CALUDE_triangle_area_theorem_l3916_391641


namespace NUMINAMATH_CALUDE_factorization_equality_l3916_391617

theorem factorization_equality (z : ℝ) :
  70 * z^20 + 154 * z^40 + 224 * z^60 = 14 * z^20 * (5 + 11 * z^20 + 16 * z^40) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3916_391617


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3916_391630

theorem contrapositive_equivalence (x : ℝ) :
  (¬(-1 < x ∧ x < 0) ∨ x^2 < 1) ↔ (x^2 ≥ 1 → (x ≥ 0 ∨ x ≤ -1)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3916_391630


namespace NUMINAMATH_CALUDE_sum_m_twice_n_l3916_391654

/-- The sum of m and twice n is equal to m + 2n -/
theorem sum_m_twice_n (m n : ℤ) : m + 2*n = m + 2*n := by sorry

end NUMINAMATH_CALUDE_sum_m_twice_n_l3916_391654


namespace NUMINAMATH_CALUDE_function_satisfying_conditions_l3916_391600

theorem function_satisfying_conditions (f : ℝ → ℝ) : 
  (∀ x : ℝ, x ≠ 0 → f x ≠ 0) →
  f 1 = 1 →
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x + y ≠ 0 → f (1 / (x + y)) = f (1 / x) + f (1 / y)) →
  (∀ x y : ℝ, x ≠ 0 → y ≠ 0 → x + y ≠ 0 → (x + y) * f (x + y) = x * y * f x * f y) →
  (∀ x : ℝ, x ≠ 0 → f x = 1 / x) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_conditions_l3916_391600


namespace NUMINAMATH_CALUDE_sophie_wallet_problem_l3916_391678

theorem sophie_wallet_problem :
  ∃ (x y z : ℕ), 
    x + y + z = 60 ∧
    x + 2*y + 5*z = 175 ∧
    x = 5 := by
  sorry

end NUMINAMATH_CALUDE_sophie_wallet_problem_l3916_391678


namespace NUMINAMATH_CALUDE_worksheets_to_memorize_l3916_391688

/-- Calculate the number of worksheets that can be memorized given study conditions --/
theorem worksheets_to_memorize (
  chapters : ℕ)
  (hours_per_chapter : ℝ)
  (hours_per_worksheet : ℝ)
  (max_hours_per_day : ℝ)
  (break_duration : ℝ)
  (breaks_per_day : ℕ)
  (snack_breaks : ℕ)
  (snack_break_duration : ℝ)
  (lunch_duration : ℝ)
  (study_days : ℕ)
  (h1 : chapters = 2)
  (h2 : hours_per_chapter = 3)
  (h3 : hours_per_worksheet = 1.5)
  (h4 : max_hours_per_day = 4)
  (h5 : break_duration = 1/6)  -- 10 minutes in hours
  (h6 : breaks_per_day = 4)
  (h7 : snack_breaks = 3)
  (h8 : snack_break_duration = 1/6)  -- 10 minutes in hours
  (h9 : lunch_duration = 0.5)  -- 30 minutes in hours
  (h10 : study_days = 4) :
  ⌊(study_days * (max_hours_per_day - (breaks_per_day * break_duration + snack_breaks * snack_break_duration + lunch_duration)) - chapters * hours_per_chapter) / hours_per_worksheet⌋ = 2 :=
by sorry

end NUMINAMATH_CALUDE_worksheets_to_memorize_l3916_391688


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l3916_391655

theorem polynomial_identity_sum (a b c d e f : ℤ) :
  (∀ x : ℤ, (3 * x + 1)^5 = a * x^5 + b * x^4 + c * x^3 + d * x^2 + e * x + f) →
  a - b + c - d + e - f = 32 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l3916_391655


namespace NUMINAMATH_CALUDE_unique_solution_for_odd_prime_l3916_391613

theorem unique_solution_for_odd_prime (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 + p*x = y^2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_odd_prime_l3916_391613


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3916_391635

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  man : ℝ  -- Speed of the man in still water
  stream : ℝ  -- Speed of the stream

/-- Calculates the effective speed when swimming downstream. -/
def downstream_speed (s : SwimmerSpeed) : ℝ := s.man + s.stream

/-- Calculates the effective speed when swimming upstream. -/
def upstream_speed (s : SwimmerSpeed) : ℝ := s.man - s.stream

/-- Theorem stating that given the conditions of the problem, the man's speed in still water is 12 km/h. -/
theorem swimmer_speed_in_still_water :
  ∀ (s : SwimmerSpeed),
    54 = downstream_speed s * 3 →
    18 = upstream_speed s * 3 →
    s.man = 12 := by
  sorry


end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3916_391635


namespace NUMINAMATH_CALUDE_mary_sugar_added_l3916_391681

/-- Given a recipe that requires a certain amount of sugar and the amount of sugar still needed,
    calculate the amount of sugar already added. -/
def sugar_already_added (recipe_required : ℕ) (sugar_needed : ℕ) : ℕ :=
  recipe_required - sugar_needed

/-- Theorem stating that Mary has already added 10 cups of sugar. -/
theorem mary_sugar_added :
  sugar_already_added 11 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_sugar_added_l3916_391681


namespace NUMINAMATH_CALUDE_breath_holding_contest_results_l3916_391626

/-- Represents the breath-holding times of swimmers in a contest. -/
structure SwimmersBreathHoldingTimes where
  kelly : ℕ
  brittany : ℕ
  buffy : ℕ
  carmen : ℕ
  denise : ℕ

/-- Calculates the total and average breath-holding times for the swimmers. -/
def calculateBreathHoldingTimes (times : SwimmersBreathHoldingTimes) : ℕ × ℚ :=
  let total := times.kelly + times.brittany + times.buffy + times.carmen + times.denise
  let average := (total : ℚ) / 5
  (total, average)

/-- Theorem stating the correct total and average breath-holding times for the given conditions. -/
theorem breath_holding_contest_results : 
  let times : SwimmersBreathHoldingTimes := {
    kelly := 180,
    brittany := 180 - 20,
    buffy := 180 - 20 - 40,
    carmen := 180 + 15,
    denise := 180 + 15 - 35
  }
  let (total, average) := calculateBreathHoldingTimes times
  total = 815 ∧ average = 163 := by sorry

end NUMINAMATH_CALUDE_breath_holding_contest_results_l3916_391626


namespace NUMINAMATH_CALUDE_container_capacity_l3916_391679

theorem container_capacity : 
  ∀ (capacity : ℝ), 
  (1/4 : ℝ) * capacity + 120 = (2/3 : ℝ) * capacity → 
  capacity = 288 := by
  sorry

end NUMINAMATH_CALUDE_container_capacity_l3916_391679


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l3916_391623

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 2310 → 
  Nat.gcd a b = 26 → 
  a = 210 → 
  b = 286 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l3916_391623


namespace NUMINAMATH_CALUDE_square_diagonal_triangle_dimensions_l3916_391692

theorem square_diagonal_triangle_dimensions :
  ∀ (square_side : ℝ) (triangle_leg1 triangle_leg2 triangle_hypotenuse : ℝ),
    square_side = 10 →
    triangle_leg1 = square_side →
    triangle_leg2 = square_side →
    triangle_hypotenuse^2 = triangle_leg1^2 + triangle_leg2^2 →
    (triangle_leg1 = 10 ∧ triangle_leg2 = 10 ∧ triangle_hypotenuse = 10 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_triangle_dimensions_l3916_391692


namespace NUMINAMATH_CALUDE_factorization_example_l3916_391669

/-- Represents factorization from left to right -/
def is_factorization_left_to_right (f : ℝ → ℝ) (g : ℝ → ℝ) (h : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x * h x ∧ (∃ c, g x = c * x ∨ g x = x)

/-- The given equation represents factorization from left to right -/
theorem factorization_example :
  is_factorization_left_to_right
    (λ a : ℝ => 2 * a^2 + a)
    (λ a : ℝ => a)
    (λ a : ℝ => 2 * a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_example_l3916_391669


namespace NUMINAMATH_CALUDE_system_solution_l3916_391694

theorem system_solution (x y : ℝ) (eq1 : 3 * x + y = 21) (eq2 : x + 3 * y = 1) : 2 * x + 2 * y = 11 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3916_391694


namespace NUMINAMATH_CALUDE_conference_handshakes_l3916_391683

/-- Represents a conference with two groups of people -/
structure Conference :=
  (total_people : ℕ)
  (group1_size : ℕ)
  (group2_size : ℕ)
  (h_total : total_people = group1_size + group2_size)

/-- Calculates the number of handshakes in the conference -/
def handshakes (conf : Conference) : ℕ :=
  let group2_external := conf.group2_size * (conf.total_people - 1)
  let group2_internal := (conf.group2_size * (conf.group2_size - 1)) / 2
  group2_external + group2_internal

/-- Theorem stating the number of handshakes in the specific conference scenario -/
theorem conference_handshakes :
  ∃ (conf : Conference),
    conf.total_people = 50 ∧
    conf.group1_size = 30 ∧
    conf.group2_size = 20 ∧
    handshakes conf = 1170 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l3916_391683


namespace NUMINAMATH_CALUDE_constant_term_equals_96_l3916_391666

/-- The constant term in the expansion of (2x + a/x)^4 -/
def constantTerm (a : ℝ) : ℝ := a^2 * 2^2 * 6

theorem constant_term_equals_96 (a : ℝ) (h : a > 0) : 
  constantTerm a = 96 → a = 2 := by sorry

end NUMINAMATH_CALUDE_constant_term_equals_96_l3916_391666


namespace NUMINAMATH_CALUDE_distinct_points_difference_l3916_391672

-- Define the equation of the graph
def graph_equation (x y : ℝ) : Prop := y^2 + x^4 = 3 * x^2 * y + 2

-- Define the constant e
noncomputable def e : ℝ := Real.exp 1

-- Theorem statement
theorem distinct_points_difference (a b : ℝ) 
  (ha : graph_equation (Real.sqrt e) a)
  (hb : graph_equation (Real.sqrt e) b)
  (hab : a ≠ b) : 
  |a - b| = Real.sqrt (5 * e^2 + 8) := by sorry

end NUMINAMATH_CALUDE_distinct_points_difference_l3916_391672


namespace NUMINAMATH_CALUDE_least_bananas_total_l3916_391624

/-- Represents the number of bananas each monkey initially takes -/
structure BananaDistribution where
  b1 : ℕ  -- bananas taken by first monkey
  b2 : ℕ  -- bananas taken by second monkey
  b3 : ℕ  -- bananas taken by third monkey

/-- Calculates the final number of bananas for each monkey -/
def finalBananas (d : BananaDistribution) : (ℕ × ℕ × ℕ) :=
  ( d.b1 / 2 + d.b2 / 6 + d.b3 / 8,
    d.b1 / 4 + d.b2 * 2 / 3 + d.b3 / 8,
    d.b1 / 4 + d.b2 / 6 + d.b3 * 3 / 4 )

/-- Checks if the distribution results in whole numbers of bananas -/
def isWholeDistribution (d : BananaDistribution) : Prop :=
  (d.b1 % 4 = 0) ∧ (d.b2 % 6 = 0) ∧ (d.b3 % 8 = 0)

/-- Checks if the final distribution satisfies the 4:3:2 ratio -/
def satisfiesRatio (d : BananaDistribution) : Prop :=
  let (f1, f2, f3) := finalBananas d
  (3 * f1 = 4 * f2) ∧ (3 * f2 = 4 * f3)

theorem least_bananas_total :
  ∃ (d : BananaDistribution),
    isWholeDistribution d ∧
    satisfiesRatio d ∧
    (∀ (d' : BananaDistribution),
      isWholeDistribution d' ∧ satisfiesRatio d' →
      d.b1 + d.b2 + d.b3 ≤ d'.b1 + d'.b2 + d'.b3) ∧
    d.b1 + d.b2 + d.b3 = 216 := by
  sorry


end NUMINAMATH_CALUDE_least_bananas_total_l3916_391624


namespace NUMINAMATH_CALUDE_school_age_problem_l3916_391601

theorem school_age_problem (num_students : ℕ) (num_teachers : ℕ) (avg_age_students : ℝ) 
  (avg_age_with_teachers : ℝ) (avg_age_with_principal : ℝ) :
  num_students = 30 →
  num_teachers = 3 →
  avg_age_students = 14 →
  avg_age_with_teachers = 16 →
  avg_age_with_principal = 17 →
  ∃ (total_age_teachers : ℝ) (age_principal : ℝ),
    total_age_teachers = 108 ∧ age_principal = 50 := by
  sorry

end NUMINAMATH_CALUDE_school_age_problem_l3916_391601


namespace NUMINAMATH_CALUDE_max_popsicles_for_zoe_l3916_391686

/-- Represents the pricing options for popsicles -/
structure PopsicleOptions where
  single_price : ℕ
  four_pack_price : ℕ
  seven_pack_price : ℕ

/-- Calculates the maximum number of popsicles that can be bought with a given budget -/
def max_popsicles (options : PopsicleOptions) (budget : ℕ) : ℕ :=
  sorry

/-- The store's pricing options -/
def store_options : PopsicleOptions :=
  { single_price := 2
  , four_pack_price := 3
  , seven_pack_price := 5 }

/-- Zoe's budget -/
def zoe_budget : ℕ := 11

/-- Theorem: The maximum number of popsicles Zoe can buy with $11 is 14 -/
theorem max_popsicles_for_zoe :
  max_popsicles store_options zoe_budget = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_popsicles_for_zoe_l3916_391686


namespace NUMINAMATH_CALUDE_melissa_shoe_repair_time_l3916_391633

/-- The total time Melissa spends repairing shoes -/
theorem melissa_shoe_repair_time (buckle_time heel_time strap_time sole_time : ℕ) 
  (num_pairs : ℕ) : 
  buckle_time = 5 → 
  heel_time = 10 → 
  strap_time = 7 → 
  sole_time = 12 → 
  num_pairs = 8 → 
  (buckle_time + heel_time + strap_time + sole_time) * 2 * num_pairs = 544 :=
by sorry

end NUMINAMATH_CALUDE_melissa_shoe_repair_time_l3916_391633


namespace NUMINAMATH_CALUDE_five_student_committees_from_eight_l3916_391693

theorem five_student_committees_from_eight (n : ℕ) (k : ℕ) : n = 8 → k = 5 → Nat.choose n k = 56 := by
  sorry

end NUMINAMATH_CALUDE_five_student_committees_from_eight_l3916_391693


namespace NUMINAMATH_CALUDE_square_not_prime_plus_square_l3916_391618

theorem square_not_prime_plus_square (n : ℕ) (h1 : n ≥ 5) (h2 : n % 3 = 2) :
  ¬ ∃ (p k : ℕ), Prime p ∧ n^2 = p + k^2 := by
sorry

end NUMINAMATH_CALUDE_square_not_prime_plus_square_l3916_391618


namespace NUMINAMATH_CALUDE_rhombus_tangent_distance_l3916_391604

/-- A rhombus with an inscribed circle -/
structure RhombusWithCircle where
  /-- Side length of the rhombus -/
  side : ℝ
  /-- Radius of the inscribed circle -/
  radius : ℝ
  /-- Condition that the first diagonal is less than the second diagonal -/
  diag_condition : ℝ → ℝ → Prop

/-- The distance between tangent points on adjacent sides of the rhombus -/
def tangent_distance (r : RhombusWithCircle) : ℝ := sorry

/-- Theorem stating the distance between tangent points on adjacent sides -/
theorem rhombus_tangent_distance
  (r : RhombusWithCircle)
  (h1 : r.side = 5)
  (h2 : r.radius = 2.4)
  (h3 : r.diag_condition (2 * r.radius) (2 * r.side * (1 - r.radius / r.side))) :
  tangent_distance r = 3.84 := by sorry

end NUMINAMATH_CALUDE_rhombus_tangent_distance_l3916_391604


namespace NUMINAMATH_CALUDE_remainder_theorem_l3916_391696

theorem remainder_theorem (x y q r : ℕ) (h1 : x = q * y + r) (h2 : r < y) :
  (x - 3 * q * y) % y = r := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3916_391696


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l3916_391639

theorem halfway_between_fractions :
  let a : ℚ := 1/8
  let b : ℚ := 1/3
  (a + b) / 2 = 11/48 := by
sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l3916_391639


namespace NUMINAMATH_CALUDE_potato_cooking_time_l3916_391670

theorem potato_cooking_time (total_potatoes : ℕ) (cooked_potatoes : ℕ) (remaining_time : ℕ) :
  total_potatoes = 15 →
  cooked_potatoes = 8 →
  remaining_time = 63 →
  (remaining_time / (total_potatoes - cooked_potatoes) : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_potato_cooking_time_l3916_391670


namespace NUMINAMATH_CALUDE_largest_multiple_of_18_with_6_and_9_l3916_391698

/-- A function that checks if a natural number consists only of digits 6 and 9 -/
def only_six_and_nine (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 6 ∨ d = 9

/-- The largest number consisting of only 6 and 9 digits that is divisible by 18 -/
def m : ℕ := 969696

theorem largest_multiple_of_18_with_6_and_9 :
  (∀ k : ℕ, k > m → ¬(only_six_and_nine k ∧ 18 ∣ k)) ∧
  only_six_and_nine m ∧
  18 ∣ m ∧
  m / 18 = 53872 := by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_18_with_6_and_9_l3916_391698


namespace NUMINAMATH_CALUDE_puzzle_solution_l3916_391650

theorem puzzle_solution (D E F : ℕ) 
  (h1 : D + E + F = 16)
  (h2 : F + D + 1 = 16)
  (h3 : E - 1 = D)
  (h4 : D ≠ E ∧ D ≠ F ∧ E ≠ F)
  (h5 : D < 10 ∧ E < 10 ∧ F < 10) : E = 1 := by
  sorry

#check puzzle_solution

end NUMINAMATH_CALUDE_puzzle_solution_l3916_391650


namespace NUMINAMATH_CALUDE_point_B_coordinates_l3916_391662

-- Define the points and lines
def A : ℝ × ℝ := (0, -1)
def line1 (x y : ℝ) : Prop := x - y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- Define the perpendicularity condition
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- State the theorem
theorem point_B_coordinates :
  ∀ B : ℝ × ℝ,
  line1 B.1 B.2 →
  perpendicular ((B.2 - A.2) / (B.1 - A.1)) (-1/2) →
  B = (2, 3) := by
sorry


end NUMINAMATH_CALUDE_point_B_coordinates_l3916_391662


namespace NUMINAMATH_CALUDE_green_ball_probability_l3916_391632

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The probability of selecting a green ball given three containers -/
def totalGreenProbability (c1 c2 c3 : Container) : ℚ :=
  (1 / 3) * (greenProbability c1 + greenProbability c2 + greenProbability c3)

theorem green_ball_probability :
  let c1 := Container.mk 8 4
  let c2 := Container.mk 3 5
  let c3 := Container.mk 4 6
  totalGreenProbability c1 c2 c3 = 187 / 360 := by
  sorry

end NUMINAMATH_CALUDE_green_ball_probability_l3916_391632


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l3916_391671

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 210 →
  B = 330 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l3916_391671


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l3916_391610

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 3) (Fin 3) ℝ),
  (N.mulVec (![1, 0, 0] : Fin 3 → ℝ) = ![(-1), 4, 0]) ∧
  (N.mulVec (![0, 1, 0] : Fin 3 → ℝ) = ![2, (-3), 5]) ∧
  (N.mulVec (![0, 0, 1] : Fin 3 → ℝ) = ![5, 2, (-1)]) ∧
  (N.mulVec (![1, 1, 1] : Fin 3 → ℝ) = ![6, 3, 4]) :=
by
  sorry

#check matrix_N_satisfies_conditions

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l3916_391610


namespace NUMINAMATH_CALUDE_expression_value_l3916_391675

theorem expression_value : 
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3916_391675


namespace NUMINAMATH_CALUDE_valerie_stamps_l3916_391658

/-- Calculates the total number of stamps needed for mailing various items. -/
def total_stamps (thank_you_cards : ℕ) (bills : ℕ) (extra_rebates : ℕ) : ℕ :=
  let rebates := bills + extra_rebates
  let job_applications := 2 * rebates
  let regular_stamps := thank_you_cards + bills - 1 + rebates + job_applications
  regular_stamps + 1  -- Add 1 for the extra stamp on the electric bill

theorem valerie_stamps :
  total_stamps 3 2 3 = 21 :=
by sorry

end NUMINAMATH_CALUDE_valerie_stamps_l3916_391658


namespace NUMINAMATH_CALUDE_simplest_form_sum_l3916_391663

theorem simplest_form_sum (a b : ℕ) (h : a = 63 ∧ b = 117) :
  let (n, d) := (a / gcd a b, b / gcd a b)
  n + d = 20 := by
sorry

end NUMINAMATH_CALUDE_simplest_form_sum_l3916_391663


namespace NUMINAMATH_CALUDE_cost_price_calculation_article_cost_price_l3916_391612

theorem cost_price_calculation (selling_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : ℝ :=
  let discounted_price := selling_price * (1 - discount_rate)
  let cost_price := discounted_price / (1 + profit_rate)
  cost_price

theorem article_cost_price : 
  cost_price_calculation 15000 0.1 0.08 = 12500 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_article_cost_price_l3916_391612


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l3916_391603

theorem logarithm_sum_simplification :
  1 / (Real.log 3 / Real.log 12 + Real.log 4 / Real.log 12 + 1) +
  1 / (Real.log 2 / Real.log 8 + Real.log 4 / Real.log 8 + 1) +
  1 / (1 + (Real.log 5 / Real.log 15 + Real.log 3 / Real.log 15)) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l3916_391603


namespace NUMINAMATH_CALUDE_line_intercept_ratio_l3916_391657

theorem line_intercept_ratio (b : ℝ) (s t : ℝ) 
  (h_b : b ≠ 0)
  (h_s : 0 = 10 * s + b)
  (h_t : 0 = 6 * t + b) :
  s / t = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_line_intercept_ratio_l3916_391657


namespace NUMINAMATH_CALUDE_virginia_eggs_remaining_l3916_391606

/-- Given Virginia starts with 96 eggs and Amy takes 3 eggs away, 
    prove that Virginia ends up with 93 eggs. -/
theorem virginia_eggs_remaining : 
  let initial_eggs : ℕ := 96
  let eggs_taken : ℕ := 3
  initial_eggs - eggs_taken = 93 := by sorry

end NUMINAMATH_CALUDE_virginia_eggs_remaining_l3916_391606


namespace NUMINAMATH_CALUDE_fraction_ratio_equality_l3916_391648

theorem fraction_ratio_equality : ∃ x : ℚ, (5 / 34) / (7 / 48) = x / (1 / 13) ∧ x = 5 / 64 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ratio_equality_l3916_391648


namespace NUMINAMATH_CALUDE_waiter_customer_count_l3916_391680

/-- Represents the scenario of a waiter serving customers and receiving tips -/
structure WaiterScenario where
  total_customers : ℕ
  non_tipping_customers : ℕ
  tip_amount : ℕ
  total_tips : ℕ

/-- Theorem stating that given the conditions, the waiter had 7 customers in total -/
theorem waiter_customer_count (scenario : WaiterScenario) 
  (h1 : scenario.non_tipping_customers = 5)
  (h2 : scenario.tip_amount = 3)
  (h3 : scenario.total_tips = 6) :
  scenario.total_customers = 7 := by
  sorry


end NUMINAMATH_CALUDE_waiter_customer_count_l3916_391680


namespace NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l3916_391634

theorem x_fourth_plus_inverse_x_fourth (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_plus_inverse_x_fourth_l3916_391634


namespace NUMINAMATH_CALUDE_expression_equality_l3916_391699

theorem expression_equality : (2 + Real.sqrt 6) * (2 - Real.sqrt 6) - (Real.sqrt 3 + 1)^2 = -6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3916_391699


namespace NUMINAMATH_CALUDE_parabola_focus_specific_parabola_focus_l3916_391622

/-- The focus of a parabola with equation y^2 = ax has coordinates (a/4, 0) -/
theorem parabola_focus (a : ℝ) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = a * x}
  let focus := (a / 4, 0)
  focus ∈ parabola ∧ 
  ∀ (p : ℝ × ℝ), p ∈ parabola → dist p focus = dist p (0, -a/4) :=
sorry

/-- The focus of the parabola y^2 = 8x has coordinates (2, 0) -/
theorem specific_parabola_focus :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 8 * x}
  let focus := (2, 0)
  focus ∈ parabola ∧ 
  ∀ (p : ℝ × ℝ), p ∈ parabola → dist p focus = dist p (0, -2) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_specific_parabola_focus_l3916_391622


namespace NUMINAMATH_CALUDE_simplify_expression_l3916_391628

theorem simplify_expression (y : ℝ) : 4*y + 9*y^2 + 6 - (3 - 4*y - 9*y^2) = 18*y^2 + 8*y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3916_391628


namespace NUMINAMATH_CALUDE_new_device_improvement_l3916_391611

/-- Represents the sample mean and variance of a device's measurements -/
structure DeviceStats where
  mean : ℝ
  variance : ℝ

/-- Determines if there's a significant improvement between two devices -/
def significantImprovement (old new : DeviceStats) : Prop :=
  new.mean - old.mean ≥ 2 * Real.sqrt ((old.variance + new.variance) / 10)

theorem new_device_improvement (old new : DeviceStats) 
  (h_old : old.mean = 10.3 ∧ old.variance = 0.04)
  (h_new : new.mean = 10 ∧ new.variance = 0.036) :
  significantImprovement old new :=
sorry

end NUMINAMATH_CALUDE_new_device_improvement_l3916_391611


namespace NUMINAMATH_CALUDE_expression_value_l3916_391608

theorem expression_value : 
  (3^2016 + 3^2014 + 3^2012) / (3^2016 - 3^2014 + 3^2012) = 91/73 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3916_391608


namespace NUMINAMATH_CALUDE_max_sum_on_circle_l3916_391645

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 100) : 
  x + y ≤ 14 ∧ ∃ (a b : ℤ), a^2 + b^2 = 100 ∧ a + b = 14 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_on_circle_l3916_391645
