import Mathlib

namespace sakshi_work_time_l1894_189473

/-- Proves that given Tanya is 25% more efficient than Sakshi and takes 8 days to complete a piece of work, Sakshi will take 10 days to complete the same work. -/
theorem sakshi_work_time (sakshi_time tanya_time : ℝ) 
  (h1 : tanya_time = 8)
  (h2 : sakshi_time * 1 = tanya_time * 1.25) : 
  sakshi_time = 10 := by
sorry

end sakshi_work_time_l1894_189473


namespace a_range_l1894_189469

-- Define propositions p and q as functions of a
def p (a : ℝ) : Prop := 0 < a ∧ a < 6

def q (a : ℝ) : Prop := a ≥ 5 ∨ a ≤ 1

-- Define the range of a
def range_a (a : ℝ) : Prop := a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5)

theorem a_range :
  (∀ a : ℝ, (p a ∨ q a)) ∧ (∀ a : ℝ, ¬(p a ∧ q a)) →
  ∀ a : ℝ, range_a a ↔ (p a ∨ q a) :=
by sorry

end a_range_l1894_189469


namespace triangle_side_length_l1894_189459

theorem triangle_side_length (x y z : ℝ) (Y Z : ℝ) :
  y = 7 →
  z = 3 →
  Real.cos (Y - Z) = 11 / 15 →
  x^2 = y^2 + z^2 - 2 * y * z * Real.cos Y →
  x = Real.sqrt 38.4 :=
by sorry

end triangle_side_length_l1894_189459


namespace hannah_mugs_problem_l1894_189426

/-- The number of mugs Hannah has of a color other than blue, red, or yellow -/
def other_color_mugs (total : ℕ) (blue red yellow : ℕ) : ℕ :=
  total - (blue + red + yellow)

theorem hannah_mugs_problem :
  ∀ (total blue red yellow : ℕ),
  total = 40 →
  blue = 3 * red →
  yellow = 12 →
  red = yellow / 2 →
  other_color_mugs total blue red yellow = 4 := by
sorry

end hannah_mugs_problem_l1894_189426


namespace even_function_implies_m_equals_neg_one_l1894_189448

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (x+m)(x+1) -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (x + m) * (x + 1)

/-- If f(x) = (x+m)(x+1) is an even function, then m = -1 -/
theorem even_function_implies_m_equals_neg_one :
  ∀ m : ℝ, IsEven (f m) → m = -1 := by
  sorry

end even_function_implies_m_equals_neg_one_l1894_189448


namespace work_distance_calculation_l1894_189487

/-- The distance to Tim's work in miles -/
def work_distance : ℝ := 20

/-- The number of workdays Tim rides his bike -/
def workdays : ℕ := 5

/-- The distance of Tim's weekend bike ride in miles -/
def weekend_ride : ℝ := 200

/-- Tim's biking speed in miles per hour -/
def biking_speed : ℝ := 25

/-- The total time Tim spends biking in a week in hours -/
def total_biking_time : ℝ := 16

theorem work_distance_calculation : 
  2 * workdays * work_distance + weekend_ride = biking_speed * total_biking_time := by
  sorry

end work_distance_calculation_l1894_189487


namespace mean_temperature_l1894_189460

def temperatures : List ℝ := [-6.5, -3, -2, 4, 2.5]

theorem mean_temperature : (temperatures.sum / temperatures.length) = -1 := by
  sorry

end mean_temperature_l1894_189460


namespace power_multiplication_l1894_189409

theorem power_multiplication (m : ℝ) : m^2 * m^3 = m^5 := by
  sorry

end power_multiplication_l1894_189409


namespace min_grid_size_l1894_189407

theorem min_grid_size (k : ℝ) (h : k > 0.9999) :
  (∃ n : ℕ, n ≥ 51 ∧ 4 * n * (n - 1) * k * (1 - k) = k) ∧
  (∀ m : ℕ, m < 51 → 4 * m * (m - 1) * k * (1 - k) ≠ k) := by
  sorry

end min_grid_size_l1894_189407


namespace remainder_invariance_l1894_189437

theorem remainder_invariance (S A : ℤ) (K : ℤ) : 
  S % A = (S + A * K) % A := by sorry

end remainder_invariance_l1894_189437


namespace inequality_proof_l1894_189430

theorem inequality_proof (a b c : ℝ) (ha : a = (Real.log 2) / 2) 
  (hb : b = (Real.log Real.pi) / Real.pi) (hc : c = (Real.log 5) / 5) : 
  b > a ∧ a > c := by
  sorry

end inequality_proof_l1894_189430


namespace range_of_m_l1894_189489

theorem range_of_m (m : ℝ) : 
  (¬ ∃ x ∈ Set.Icc (1/2 : ℝ) 2, x^2 - 2*x - m ≤ 0) → m < -1 := by
  sorry

end range_of_m_l1894_189489


namespace count_four_digit_numbers_l1894_189498

def four_digit_numbers_with_1_and_2 : ℕ :=
  let one_one := 4  -- 1 occurrence of 1, 3 occurrences of 2
  let two_ones := 6 -- 2 occurrences of 1, 2 occurrences of 2
  let three_ones := 4 -- 3 occurrences of 1, 1 occurrence of 2
  one_one + two_ones + three_ones

theorem count_four_digit_numbers : four_digit_numbers_with_1_and_2 = 14 := by
  sorry

end count_four_digit_numbers_l1894_189498


namespace odd_times_abs_even_is_odd_l1894_189406

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_times_abs_even_is_odd
  (f g : ℝ → ℝ)
  (h_odd : isOdd f)
  (h_even : isEven g) :
  isOdd (fun x ↦ f x * |g x|) := by
  sorry

end odd_times_abs_even_is_odd_l1894_189406


namespace circle_radius_l1894_189436

/-- Given a circle with diameter 26 centimeters, prove that its radius is 13 centimeters. -/
theorem circle_radius (diameter : ℝ) (h : diameter = 26) : diameter / 2 = 13 := by
  sorry

end circle_radius_l1894_189436


namespace complex_multiplication_l1894_189432

/-- Given that i² = -1, prove that (4-5i)(-5+5i) = 5 + 45i --/
theorem complex_multiplication (i : ℂ) (hi : i^2 = -1) :
  (4 - 5*i) * (-5 + 5*i) = 5 + 45*i := by
  sorry

end complex_multiplication_l1894_189432


namespace power_of_product_cube_l1894_189422

theorem power_of_product_cube (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end power_of_product_cube_l1894_189422


namespace triangle_inequality_sum_l1894_189425

theorem triangle_inequality_sum (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 := by
  sorry

end triangle_inequality_sum_l1894_189425


namespace line_plane_intersection_l1894_189479

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation for lines and planes
variable (subset : Line → Plane → Prop)

-- Define the intersection relation for lines and planes
variable (intersect : Line → Line → Prop)
variable (intersect_planes : Plane → Plane → Prop)

-- Theorem statement
theorem line_plane_intersection 
  (a b : Line) (α β : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β) 
  (h3 : subset a α) 
  (h4 : subset b β) 
  (h5 : intersect a b) : 
  intersect_planes α β :=
sorry

end line_plane_intersection_l1894_189479


namespace smallest_n_divisible_by_31_l1894_189483

theorem smallest_n_divisible_by_31 :
  ∃ (n : ℕ), n > 0 ∧ (31 ∣ (5^n + n)) ∧ ∀ (m : ℕ), m > 0 ∧ (31 ∣ (5^m + m)) → n ≤ m :=
by
  use 30
  sorry

end smallest_n_divisible_by_31_l1894_189483


namespace three_special_lines_l1894_189429

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if a line has intercepts on both axes with equal absolute values -/
def Line.hasEqualIntercepts (l : Line) : Prop :=
  ∃ t : ℝ, t ≠ 0 ∧ (l.a * t + l.c = 0 ∧ l.b * t + l.c = 0)

/-- The set of lines passing through (1, 2) with equal intercepts -/
def specialLines : Set Line :=
  {l : Line | l.contains 1 2 ∧ l.hasEqualIntercepts}

theorem three_special_lines :
  ∃ (l₁ l₂ l₃ : Line),
    l₁ ∈ specialLines ∧
    l₂ ∈ specialLines ∧
    l₃ ∈ specialLines ∧
    l₁ ≠ l₂ ∧ l₁ ≠ l₃ ∧ l₂ ≠ l₃ ∧
    ∀ l : Line, l ∈ specialLines → l = l₁ ∨ l = l₂ ∨ l = l₃ :=
  sorry

end three_special_lines_l1894_189429


namespace m_value_after_subtraction_l1894_189471

theorem m_value_after_subtraction (M : ℝ) : 
  (25 / 100 : ℝ) * M = (55 / 100 : ℝ) * 2500 → 
  M - (10 / 100 : ℝ) * M = 4950 := by
  sorry

end m_value_after_subtraction_l1894_189471


namespace words_per_page_l1894_189474

theorem words_per_page (total_pages : Nat) (max_words_per_page : Nat) (total_words_mod : Nat) :
  total_pages = 154 →
  max_words_per_page = 120 →
  total_words_mod = 221 →
  ∃ words_per_page : Nat,
    words_per_page ≤ max_words_per_page ∧
    (total_pages * words_per_page) % 217 = total_words_mod ∧
    words_per_page = 106 := by
  sorry

end words_per_page_l1894_189474


namespace greatest_x_given_lcm_l1894_189455

theorem greatest_x_given_lcm (x : ℕ+) : 
  (Nat.lcm x (Nat.lcm 12 18) = 180) → x ≤ 180 ∧ ∃ y : ℕ+, y > 180 → Nat.lcm y (Nat.lcm 12 18) > 180 :=
by sorry

end greatest_x_given_lcm_l1894_189455


namespace taehyung_age_l1894_189458

theorem taehyung_age (taehyung_age uncle_age : ℕ) 
  (h1 : uncle_age = taehyung_age + 17)
  (h2 : (taehyung_age + 4) + (uncle_age + 4) = 43) : 
  taehyung_age = 9 := by
  sorry

end taehyung_age_l1894_189458


namespace min_trig_expression_min_trig_expression_equality_l1894_189470

theorem min_trig_expression (θ : Real) (h : 0 < θ ∧ θ < Real.pi / 2) :
  3 * Real.cos θ + 2 / Real.sin θ + 2 * Real.sqrt 2 * Real.tan θ ≥ 7 * Real.sqrt 2 / 2 :=
by sorry

theorem min_trig_expression_equality :
  3 * Real.cos (Real.pi / 4) + 2 / Real.sin (Real.pi / 4) + 2 * Real.sqrt 2 * Real.tan (Real.pi / 4) = 7 * Real.sqrt 2 / 2 :=
by sorry

end min_trig_expression_min_trig_expression_equality_l1894_189470


namespace dans_tshirt_production_rate_l1894_189401

/-- The time it takes Dan to make one t-shirt in the first hour -/
def time_per_shirt_first_hour (total_shirts : ℕ) (second_hour_rate : ℕ) (minutes_per_hour : ℕ) : ℕ :=
  let second_hour_shirts := minutes_per_hour / second_hour_rate
  let first_hour_shirts := total_shirts - second_hour_shirts
  minutes_per_hour / first_hour_shirts

/-- Theorem stating that it takes Dan 12 minutes to make one t-shirt in the first hour -/
theorem dans_tshirt_production_rate :
  time_per_shirt_first_hour 15 6 60 = 12 :=
by
  sorry

#eval time_per_shirt_first_hour 15 6 60

end dans_tshirt_production_rate_l1894_189401


namespace chord_intersection_sum_of_squares_l1894_189484

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit
def O : Point := Unit.unit -- Center of the circle
def A : Point := Unit.unit
def B : Point := Unit.unit
def C : Point := Unit.unit
def D : Point := Unit.unit
def E : Point := Unit.unit

-- Define the radius of the circle
def radius : ℝ := 10

-- Define the necessary functions and properties
def isOnCircle (p : Point) : Prop := sorry
def isChord (p q : Point) : Prop := sorry
def isDiameter (p q : Point) : Prop := sorry
def intersectsAt (l1 l2 : Point × Point) (p : Point) : Prop := sorry
def distance (p q : Point) : ℝ := sorry
def angle (p q r : Point) : ℝ := sorry

-- State the theorem
theorem chord_intersection_sum_of_squares 
  (h1 : isOnCircle A ∧ isOnCircle B ∧ isOnCircle C ∧ isOnCircle D)
  (h2 : isDiameter A B)
  (h3 : isChord C D)
  (h4 : intersectsAt (A, B) (C, D) E)
  (h5 : distance B E = 6)
  (h6 : angle A E C = 60) :
  (distance C E)^2 + (distance D E)^2 = 300 := by sorry

end chord_intersection_sum_of_squares_l1894_189484


namespace prob_two_math_books_l1894_189428

def total_books : ℕ := 5
def math_books : ℕ := 3

def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem prob_two_math_books : 
  (choose math_books 2 : ℚ) / (choose total_books 2) = 3 / 10 := by
  sorry

end prob_two_math_books_l1894_189428


namespace proposition_p_or_q_exclusive_l1894_189495

theorem proposition_p_or_q_exclusive (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - x + a^2 - 6*a = 0 ∧ y^2 - y + a^2 - 6*a = 0) ∨
  (∃ x : ℝ, x^2 + (a - 3)*x + 1 = 0) ∧
  ¬((∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - x + a^2 - 6*a = 0 ∧ y^2 - y + a^2 - 6*a = 0) ∧
    (∃ x : ℝ, x^2 + (a - 3)*x + 1 = 0)) ↔
  a ≥ 6 ∨ a ≤ 0 ∨ (1 < a ∧ a < 5) :=
by sorry

end proposition_p_or_q_exclusive_l1894_189495


namespace mistaken_division_correction_l1894_189494

theorem mistaken_division_correction (n : ℕ) : 
  (n / 7 = 12 ∧ n % 7 = 4) → n / 3 = 29 := by
  sorry

end mistaken_division_correction_l1894_189494


namespace OPRQ_shape_l1894_189405

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Checks if three points are collinear -/
def collinear (p q r : Point) : Prop :=
  (q.x - p.x) * (r.y - p.y) = (r.x - p.x) * (q.y - p.y)

/-- Represents the figure OPRQ -/
structure OPRQ :=
  (O : Point)
  (P : Point)
  (Q : Point)
  (R : Point)
  (h_distinct : P ≠ Q)
  (h_R : R.x = P.x + Q.x ∧ R.y = P.y + Q.y)
  (h_O : O.x = 0 ∧ O.y = 0)

/-- The figure OPRQ is a parallelogram -/
def is_parallelogram (f : OPRQ) : Prop :=
  f.O.x + f.R.x = f.P.x + f.Q.x ∧ f.O.y + f.R.y = f.P.y + f.Q.y

/-- The figure OPRQ is a straight line -/
def is_straight_line (f : OPRQ) : Prop :=
  collinear f.O f.P f.Q ∧ collinear f.O f.P f.R

theorem OPRQ_shape (f : OPRQ) :
  is_parallelogram f ∨ is_straight_line f :=
sorry

end OPRQ_shape_l1894_189405


namespace cistern_filling_time_l1894_189421

/-- The time it takes to fill a cistern without a leak, given that:
    1. With a leak, it takes T + 2 hours to fill
    2. When full, it takes 24 hours to empty due to the leak -/
theorem cistern_filling_time (T : ℝ) : 
  (∀ (t : ℝ), t > 0 → (1 / T - 1 / (T + 2) = 1 / 24)) → 
  T = 6 := by
sorry

end cistern_filling_time_l1894_189421


namespace log_inequality_solution_l1894_189493

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log (1/2)

-- Define the solution set
def solution_set : Set ℝ := {x | log_half (2*x - 1) < log_half (-x + 5)}

-- Theorem statement
theorem log_inequality_solution :
  solution_set = Set.Ioo 2 5 :=
by sorry

end log_inequality_solution_l1894_189493


namespace sum_of_four_cubes_equals_three_l1894_189420

theorem sum_of_four_cubes_equals_three (k : ℤ) :
  (1 + 6 * k^3)^3 + (1 - 6 * k^3)^3 + (-6 * k^2)^3 + 1^3 = 3 := by
  sorry

end sum_of_four_cubes_equals_three_l1894_189420


namespace coffee_price_percentage_increase_l1894_189457

def highest_price : ℝ := 45
def lowest_price : ℝ := 30

theorem coffee_price_percentage_increase :
  (highest_price - lowest_price) / lowest_price * 100 = 50 := by
  sorry

end coffee_price_percentage_increase_l1894_189457


namespace quadratic_equation_solution_l1894_189499

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 6 ∧ x₂ = -2) ∧ 
  (x₁^2 - 4*x₁ = 12) ∧ 
  (x₂^2 - 4*x₂ = 12) ∧
  (∀ x : ℝ, x^2 - 4*x = 12 → x = x₁ ∨ x = x₂) := by
sorry

end quadratic_equation_solution_l1894_189499


namespace min_value_of_sum_l1894_189477

theorem min_value_of_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 6) : 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 6 → 
    9/a + 4/b + 1/c ≤ 9/x + 4/y + 1/z) ∧ 
  (9/a + 4/b + 1/c = 6) := by
sorry

end min_value_of_sum_l1894_189477


namespace slope_base_extension_l1894_189452

/-- Extension of slope base to change inclination angle -/
theorem slope_base_extension (slope_length : ℝ) (initial_angle final_angle : ℝ) 
  (h_slope : slope_length = 1)
  (h_initial : initial_angle = 20 * π / 180)
  (h_final : final_angle = 10 * π / 180) :
  let extension := slope_length
  extension = 1 := by sorry

end slope_base_extension_l1894_189452


namespace point_not_on_ln_graph_l1894_189434

theorem point_not_on_ln_graph (a b : ℝ) (h : b = Real.log a) :
  ¬(1 + b = Real.log (a + Real.exp 1)) := by
  sorry

end point_not_on_ln_graph_l1894_189434


namespace cos_25_minus_alpha_equals_one_third_l1894_189468

theorem cos_25_minus_alpha_equals_one_third 
  (h : Real.sin (65 * π / 180 + α) = 1 / 3) : 
  Real.cos (25 * π / 180 - α) = 1 / 3 := by
  sorry

end cos_25_minus_alpha_equals_one_third_l1894_189468


namespace tetrahedron_surface_area_l1894_189464

-- Define a tetrahedron with edge length 2
def Tetrahedron := {edge_length : ℝ // edge_length = 2}

-- Define the surface area of a tetrahedron
noncomputable def surfaceArea (t : Tetrahedron) : ℝ :=
  4 * Real.sqrt 3

-- Theorem statement
theorem tetrahedron_surface_area (t : Tetrahedron) :
  surfaceArea t = 4 * Real.sqrt 3 := by
  sorry

end tetrahedron_surface_area_l1894_189464


namespace average_marks_of_passed_candidates_l1894_189440

theorem average_marks_of_passed_candidates 
  (total_candidates : ℕ) 
  (overall_average : ℚ) 
  (failed_average : ℚ) 
  (passed_count : ℕ) 
  (h1 : total_candidates = 120)
  (h2 : overall_average = 35)
  (h3 : failed_average = 15)
  (h4 : passed_count = 100) :
  (total_candidates * overall_average - (total_candidates - passed_count) * failed_average) / passed_count = 39 := by
sorry

end average_marks_of_passed_candidates_l1894_189440


namespace total_money_l1894_189478

theorem total_money (A B C : ℕ) 
  (h1 : A + C = 200)
  (h2 : B + C = 350)
  (h3 : C = 50) :
  A + B + C = 500 := by
  sorry

end total_money_l1894_189478


namespace bobby_average_increase_l1894_189439

/-- Represents Bobby's deadlift capabilities and progress --/
structure DeadliftProgress where
  initial_weight : ℕ  -- Initial deadlift weight at age 13
  final_weight : ℕ    -- Final deadlift weight at age 18
  initial_age : ℕ     -- Age when initial weight was lifted
  final_age : ℕ       -- Age when final weight was lifted

/-- Calculates the average yearly increase in deadlift weight --/
def average_yearly_increase (progress : DeadliftProgress) : ℚ :=
  (progress.final_weight - progress.initial_weight : ℚ) / (progress.final_age - progress.initial_age)

/-- Bobby's actual deadlift progress --/
def bobby_progress : DeadliftProgress := {
  initial_weight := 300,
  final_weight := 850,
  initial_age := 13,
  final_age := 18
}

/-- Theorem stating that Bobby's average yearly increase in deadlift weight is 110 pounds --/
theorem bobby_average_increase : 
  average_yearly_increase bobby_progress = 110 := by
  sorry

end bobby_average_increase_l1894_189439


namespace function_composition_l1894_189492

/-- Given a function f where f(3x) = 3 / (3 + x) for all x > 0, prove that 3f(x) = 27 / (9 + x) -/
theorem function_composition (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 3 / (3 + x)) :
  ∀ x > 0, 3 * f x = 27 / (9 + x) := by
  sorry

end function_composition_l1894_189492


namespace katies_friends_games_l1894_189462

/-- Given that Katie has 57 new games and 39 old games, and she has 62 more games than her friends,
    prove that her friends have 34 new games. -/
theorem katies_friends_games (katie_new : ℕ) (katie_old : ℕ) (difference : ℕ) 
    (h1 : katie_new = 57)
    (h2 : katie_old = 39)
    (h3 : difference = 62) :
  katie_new + katie_old - difference = 34 := by
  sorry

end katies_friends_games_l1894_189462


namespace complex_equation_sum_l1894_189417

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (1 + Complex.I) * (2 - Complex.I) → a + b = 4 := by
  sorry

end complex_equation_sum_l1894_189417


namespace apple_grape_worth_l1894_189451

theorem apple_grape_worth (apple_value grape_value : ℚ) :
  (3/4 * 16) * apple_value = 10 * grape_value →
  (1/3 * 9) * apple_value = (5/2) * grape_value := by
  sorry

end apple_grape_worth_l1894_189451


namespace set_intersection_equality_l1894_189403

def S : Set Int := {s | ∃ n : Int, s = 2*n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4*n + 1}

theorem set_intersection_equality : S ∩ T = T := by
  sorry

end set_intersection_equality_l1894_189403


namespace debby_bottles_remaining_l1894_189485

/-- Calculates the number of water bottles remaining after a period of consumption. -/
def bottles_remaining (initial : ℕ) (daily_consumption : ℕ) (days : ℕ) : ℕ :=
  initial - daily_consumption * days

/-- Proves that Debby has 99 bottles left after her consumption period. -/
theorem debby_bottles_remaining :
  bottles_remaining 264 15 11 = 99 := by
  sorry

end debby_bottles_remaining_l1894_189485


namespace regular_polygon_properties_l1894_189476

/-- A regular polygon with exterior angles measuring 18 degrees -/
structure RegularPolygon where
  -- The number of sides
  sides : ℕ
  -- The measure of each exterior angle in degrees
  exterior_angle : ℝ
  -- The measure of each interior angle in degrees
  interior_angle : ℝ
  -- Condition: The polygon is regular and each exterior angle measures 18 degrees
  h_exterior : exterior_angle = 18
  -- Relationship between number of sides and exterior angle
  h_sides : sides = (360 : ℝ) / exterior_angle
  -- Relationship between interior and exterior angles
  h_interior : interior_angle = 180 - exterior_angle

/-- Theorem about the properties of the specific regular polygon -/
theorem regular_polygon_properties (p : RegularPolygon) : 
  p.sides = 20 ∧ p.interior_angle = 162 := by
  sorry


end regular_polygon_properties_l1894_189476


namespace composite_3p_squared_plus_15_l1894_189497

theorem composite_3p_squared_plus_15 (p : ℕ) (h : Nat.Prime p) :
  ¬ Nat.Prime (3 * p^2 + 15) := by
  sorry

end composite_3p_squared_plus_15_l1894_189497


namespace watermelon_slices_l1894_189446

theorem watermelon_slices (danny_watermelons : ℕ) (sister_watermelon : ℕ) 
  (sister_slices : ℕ) (total_slices : ℕ) (danny_slices : ℕ) : 
  danny_watermelons = 3 → 
  sister_watermelon = 1 → 
  sister_slices = 15 → 
  total_slices = 45 → 
  danny_watermelons * danny_slices + sister_watermelon * sister_slices = total_slices → 
  danny_slices = 10 := by
sorry

end watermelon_slices_l1894_189446


namespace f_properties_l1894_189431

noncomputable def f (x : ℝ) := Real.sin x + Real.cos x

theorem f_properties :
  ∃ (T : ℝ) (max_val : ℝ) (k : ℤ → ℝ),
    (∀ x, f (x + T) = f x) ∧ 
    (∀ y, y > 0 → (∀ x, f (x + y) = f x) → y ≥ T) ∧
    (T = 2 * Real.pi) ∧
    (∀ x, f x ≤ max_val) ∧
    (max_val = 2) ∧
    (∀ n, f (k n) = max_val) ∧
    (∀ n, k n = 2 * n * Real.pi + Real.pi / 4) := by
  sorry

end f_properties_l1894_189431


namespace black_triangles_2008_l1894_189416

/-- Given a sequence of triangles in the pattern ▲▲△△▲△, 
    this function returns the number of black triangles in n triangles -/
def black_triangles (n : ℕ) : ℕ :=
  (n - n % 6) / 2 + min 2 (n % 6)

/-- Theorem: In a sequence of 2008 triangles following the pattern ▲▲△△▲△,
    there are 1004 black triangles -/
theorem black_triangles_2008 : black_triangles 2008 = 1004 := by
  sorry

end black_triangles_2008_l1894_189416


namespace third_day_temp_is_two_l1894_189412

/-- The temperature on the third day of a sequence of 8 days, given other temperatures and the mean -/
def third_day_temperature (t1 t2 t4 t5 t6 t7 t8 mean : ℚ) : ℚ :=
  let sum := t1 + t2 + t4 + t5 + t6 + t7 + t8
  8 * mean - sum

theorem third_day_temp_is_two :
  let t1 := -6
  let t2 := -3
  let t4 := -6
  let t5 := 2
  let t6 := 4
  let t7 := 3
  let t8 := 0
  let mean := -0.5
  third_day_temperature t1 t2 t4 t5 t6 t7 t8 mean = 2 := by
  sorry

#eval third_day_temperature (-6) (-3) (-6) 2 4 3 0 (-0.5)

end third_day_temp_is_two_l1894_189412


namespace hulk_jump_exceeds_500_l1894_189454

def hulk_jump (n : ℕ) : ℝ :=
  2 * (3 : ℝ) ^ (n - 1)

theorem hulk_jump_exceeds_500 :
  (∀ k < 7, hulk_jump k ≤ 500) ∧ hulk_jump 7 > 500 := by
  sorry

end hulk_jump_exceeds_500_l1894_189454


namespace population_percentage_l1894_189442

theorem population_percentage (W M : ℝ) (h : M = 1.1111111111111111 * W) : 
  W = 0.9 * M := by
sorry

end population_percentage_l1894_189442


namespace ellipse_line_intersection_l1894_189490

-- Define the ellipse
def E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the lines
def l (k x y : ℝ) : Prop := y = k * x + 1
def l₁ (k₁ x y : ℝ) : Prop := y = k₁ * x + 1

-- Define the symmetry line
def sym_line (x y : ℝ) : Prop := y = x + 1

-- Define the theorem
theorem ellipse_line_intersection
  (k k₁ : ℝ) 
  (hk : k > 0) 
  (hk_neq : k ≠ 1) 
  (h_sym : ∀ x y, l k x y ↔ l₁ k₁ (y - 1) (x - 1)) :
  ∃ P : ℝ × ℝ, 
    k * k₁ = 1 ∧ 
    ∀ M N : ℝ × ℝ, 
      (E M.1 M.2 ∧ l k M.1 M.2) → 
      (E N.1 N.2 ∧ l₁ k₁ N.1 N.2) → 
      ∃ t : ℝ, P = (1 - t) • M + t • N :=
by sorry

end ellipse_line_intersection_l1894_189490


namespace marble_count_l1894_189465

theorem marble_count : ∀ (r b : ℕ),
  (r - 2) * 10 = r + b - 2 →
  r * 6 = r + b - 3 →
  (r - 2) * 8 = r + b - 4 →
  r + b = 42 := by
sorry

end marble_count_l1894_189465


namespace distance_to_reflection_over_x_axis_distance_D_to_D_l1894_189400

/-- The distance between a point and its reflection over the x-axis --/
theorem distance_to_reflection_over_x_axis (x y : ℝ) : 
  let D : ℝ × ℝ := (x, y)
  let D' : ℝ × ℝ := (x, -y)
  Real.sqrt ((D'.1 - D.1)^2 + (D'.2 - D.2)^2) = 2 * abs y := by
  sorry

/-- The specific case for point D(2, 4) --/
theorem distance_D_to_D'_is_8 : 
  let D : ℝ × ℝ := (2, 4)
  let D' : ℝ × ℝ := (2, -4)
  Real.sqrt ((D'.1 - D.1)^2 + (D'.2 - D.2)^2) = 8 := by
  sorry

end distance_to_reflection_over_x_axis_distance_D_to_D_l1894_189400


namespace m_values_l1894_189414

def A (m : ℝ) : Set ℝ := {0, m, m^2 - 3*m + 2}

theorem m_values (m : ℝ) (h : 2 ∈ A m) : m = 0 ∨ m = 2 ∨ m = 3 := by
  sorry

end m_values_l1894_189414


namespace snack_eaters_final_count_l1894_189456

/-- Calculates the final number of snack eaters after a series of events --/
def finalSnackEaters (initialGathering : ℕ) (initialSnackEaters : ℕ)
  (firstNewGroup : ℕ) (secondNewGroup : ℕ) (thirdLeaving : ℕ) : ℕ :=
  let afterFirst := initialSnackEaters + firstNewGroup
  let afterHalfLeft := afterFirst / 2
  let afterSecondNew := afterHalfLeft + secondNewGroup
  let afterThirdLeft := afterSecondNew - thirdLeaving
  afterThirdLeft / 2

/-- Theorem stating that given the initial conditions and sequence of events,
    the final number of snack eaters is 20 --/
theorem snack_eaters_final_count :
  finalSnackEaters 200 100 20 10 30 = 20 := by
  sorry

#eval finalSnackEaters 200 100 20 10 30

end snack_eaters_final_count_l1894_189456


namespace chocolate_bars_left_l1894_189444

theorem chocolate_bars_left (initial_bars : ℕ) (thomas_friends : ℕ) (piper_return : ℕ) (paul_extra : ℕ) : 
  initial_bars = 500 →
  thomas_friends = 7 →
  piper_return = 7 →
  paul_extra = 5 →
  ∃ (thomas_take piper_take paul_take : ℕ),
    thomas_take = (initial_bars / 3 / thomas_friends) * thomas_friends + 2 ∧
    piper_take = initial_bars / 4 - piper_return ∧
    paul_take = piper_take + paul_extra ∧
    initial_bars - (thomas_take + piper_take + paul_take) = 96 := by
  sorry

end chocolate_bars_left_l1894_189444


namespace speed_ratio_is_five_sixths_l1894_189445

/-- Two objects A and B moving uniformly along perpendicular paths -/
structure MotionProblem where
  /-- Speed of object A in yards per minute -/
  vA : ℝ
  /-- Speed of object B in yards per minute -/
  vB : ℝ
  /-- Initial distance of B from point O in yards -/
  initialDistB : ℝ
  /-- Time when A and B are first equidistant from O in minutes -/
  t1 : ℝ
  /-- Time when A and B are again equidistant from O in minutes -/
  t2 : ℝ

/-- The theorem stating the ratio of speeds given the problem conditions -/
theorem speed_ratio_is_five_sixths (p : MotionProblem)
  (h1 : p.initialDistB = 500)
  (h2 : p.t1 = 2)
  (h3 : p.t2 = 10)
  (h4 : p.t1 * p.vA = abs (p.initialDistB - p.t1 * p.vB))
  (h5 : p.t2 * p.vA = abs (p.initialDistB - p.t2 * p.vB)) :
  p.vA / p.vB = 5 / 6 := by
  sorry


end speed_ratio_is_five_sixths_l1894_189445


namespace geometric_sum_is_60_l1894_189402

/-- The sum of a geometric sequence with 4 terms, first term 4, and common ratio 2 -/
def geometric_sum : ℕ := 
  let a := 4  -- first term
  let r := 2  -- common ratio
  let n := 4  -- number of terms
  a * (r^n - 1) / (r - 1)

/-- Theorem stating that the geometric sum is equal to 60 -/
theorem geometric_sum_is_60 : geometric_sum = 60 := by
  sorry

end geometric_sum_is_60_l1894_189402


namespace x_values_l1894_189441

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- State the theorem
theorem x_values (x : ℝ) (h : A x ∩ B x = B x) : x = 0 ∨ x = 2 ∨ x = -2 := by
  sorry

end x_values_l1894_189441


namespace purely_imaginary_z_and_z_plus_one_squared_l1894_189418

theorem purely_imaginary_z_and_z_plus_one_squared (z : ℂ) :
  (z.re = 0) → ((z + 1)^2).re = 0 → (z = Complex.I ∨ z = -Complex.I) := by
  sorry

end purely_imaginary_z_and_z_plus_one_squared_l1894_189418


namespace perimeter_equality_y_approximation_l1894_189447

-- Define the side length of the square
def y : ℝ := sorry

-- Define the radius of the circle
def r : ℝ := 4

-- Theorem stating the equality of square perimeter and circle circumference
theorem perimeter_equality : 4 * y = 2 * Real.pi * r := sorry

-- Theorem stating the approximate value of y
theorem y_approximation : ∃ (ε : ℝ), ε < 0.005 ∧ |y - 6.28| < ε := sorry

end perimeter_equality_y_approximation_l1894_189447


namespace gym_cost_is_twelve_l1894_189482

/-- Calculates the monthly cost of a gym membership given the total cost for 3 years and the down payment. -/
def monthly_gym_cost (total_cost : ℚ) (down_payment : ℚ) : ℚ :=
  (total_cost - down_payment) / (3 * 12)

/-- Theorem stating that the monthly cost of the gym is $12 under given conditions. -/
theorem gym_cost_is_twelve :
  let total_cost : ℚ := 482
  let down_payment : ℚ := 50
  monthly_gym_cost total_cost down_payment = 12 := by
  sorry

end gym_cost_is_twelve_l1894_189482


namespace square_area_9cm_l1894_189408

/-- The area of a square with side length 9 cm is 81 cm² -/
theorem square_area_9cm (square : Real → Real) (h : ∀ x, square x = x * x) :
  square 9 = 81 :=
by sorry

end square_area_9cm_l1894_189408


namespace min_value_product_l1894_189453

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x * y * z = 1 → (a + 1) * (b + 1) * (c + 1) ≤ (x + 1) * (y + 1) * (z + 1)) ∧
  (a + 1) * (b + 1) * (c + 1) = 8 :=
sorry

end min_value_product_l1894_189453


namespace angles_on_squared_paper_l1894_189410

/-- Three angles marked on squared paper sum to 90 degrees -/
theorem angles_on_squared_paper (α β γ : ℝ) : α + β + γ = 90 := by
  sorry

end angles_on_squared_paper_l1894_189410


namespace revenue_percent_change_l1894_189415

/-- Calculates the percent change in revenue given initial conditions and tax changes -/
theorem revenue_percent_change 
  (initial_consumption : ℝ)
  (initial_tax_rate : ℝ)
  (tax_decrease_percent : ℝ)
  (consumption_increase_percent : ℝ)
  (additional_tax_decrease_percent : ℝ)
  (h1 : initial_consumption = 150)
  (h2 : tax_decrease_percent = 0.2)
  (h3 : consumption_increase_percent = 0.2)
  (h4 : additional_tax_decrease_percent = 0.02)
  (h5 : initial_consumption * (1 + consumption_increase_percent) < 200) :
  let new_consumption := initial_consumption * (1 + consumption_increase_percent)
  let new_tax_rate := initial_tax_rate * (1 - tax_decrease_percent - additional_tax_decrease_percent)
  let initial_revenue := initial_consumption * initial_tax_rate
  let new_revenue := new_consumption * new_tax_rate
  let percent_change := (new_revenue - initial_revenue) / initial_revenue * 100
  percent_change = -6.4 := by
sorry

end revenue_percent_change_l1894_189415


namespace intersection_of_A_and_complement_of_B_l1894_189496

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x > 2}

theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | 1 < x ∧ x ≤ 2} := by sorry

end intersection_of_A_and_complement_of_B_l1894_189496


namespace class_configuration_exists_l1894_189413

theorem class_configuration_exists (n : ℕ) (hn : n = 30) :
  ∃ (b g : ℕ),
    b + g = n ∧
    b = g ∧
    (∀ i j : ℕ, i < b → j < b → i ≠ j → ∃ k : ℕ, k < g ∧ (∃ f : ℕ → ℕ → Prop, f i k ≠ f j k)) ∧
    (∀ i j : ℕ, i < g → j < g → i ≠ j → ∃ k : ℕ, k < b ∧ (∃ f : ℕ → ℕ → Prop, f k i ≠ f k j)) :=
by
  sorry

end class_configuration_exists_l1894_189413


namespace two_digit_reverse_sum_l1894_189423

theorem two_digit_reverse_sum (x y n : ℕ) : 
  (10 ≤ x ∧ x < 100) →  -- x is a two-digit integer
  (10 ≤ y ∧ y < 100) →  -- y is a two-digit integer
  (∃ a b : ℕ, x = 10 * a + b ∧ y = 10 * b + a ∧ a ≠ 0 ∧ b ≠ 0) →  -- y is reverse of x
  (∃ a b : ℕ, x = 10 * a + b ∧ a + b = 8) →  -- sum of digits of x is 8
  x^2 - y^2 = n^2 →  -- x^2 - y^2 = n^2
  x + y + n = 144 := by
sorry

end two_digit_reverse_sum_l1894_189423


namespace set_equality_l1894_189433

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 3)}
def B : Set ℝ := {x | x ≤ -1}

-- Define the set we want to prove is equal to the complement of A ∪ B
def C : Set ℝ := {x | -1 < x ∧ x ≤ 3}

-- State the theorem
theorem set_equality : C = (Set.univ : Set ℝ) \ (A ∪ B) := by
  sorry

end set_equality_l1894_189433


namespace not_all_parallel_lines_in_plane_l1894_189424

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (contained_in_plane : Line → Plane → Prop)

-- State the theorem
theorem not_all_parallel_lines_in_plane 
  (b : Line) (a : Line) (α : Plane)
  (h1 : parallel_line_plane b α)
  (h2 : contained_in_plane a α) :
  ¬ (∀ (l : Line), parallel_line_plane l α → ∀ (m : Line), contained_in_plane m α → parallel_lines l m) :=
sorry

end not_all_parallel_lines_in_plane_l1894_189424


namespace estate_distribution_l1894_189463

/-- Mrs. K's estate distribution problem -/
theorem estate_distribution (E : ℝ) 
  (daughters_share : ℝ) 
  (husband_share : ℝ) 
  (gardener_share : ℝ) : 
  (daughters_share = 0.4 * E) →
  (husband_share = 3 * daughters_share) →
  (gardener_share = 1000) →
  (E = daughters_share + husband_share + gardener_share) →
  (E = 2500) := by
sorry

end estate_distribution_l1894_189463


namespace rabbit_speed_l1894_189443

/-- Proves that a rabbit catching up to a cat in 1 hour, given the cat's speed and head start, has a speed of 25 mph. -/
theorem rabbit_speed (cat_speed : ℝ) (head_start : ℝ) (catch_up_time : ℝ) :
  cat_speed = 20 →
  head_start = 0.25 →
  catch_up_time = 1 →
  let rabbit_speed := (cat_speed * (catch_up_time + head_start)) / catch_up_time
  rabbit_speed = 25 := by
  sorry

end rabbit_speed_l1894_189443


namespace parabola_coefficient_l1894_189419

/-- Given a parabola y = ax^2 + bx + c with vertex at (q, q) and y-intercept at (0, -2q),
    where q ≠ 0, the value of b is 6/q. -/
theorem parabola_coefficient (a b c q : ℝ) : q ≠ 0 →
  (∀ x y, y = a * x^2 + b * x + c ↔ 
    ((x - q)^2 = 0 → y = q) ∧ 
    (x = 0 → y = -2 * q)) →
  b = 6 / q := by sorry

end parabola_coefficient_l1894_189419


namespace calories_left_for_dinner_l1894_189475

def daily_allowance : ℕ := 2200
def breakfast_calories : ℕ := 353
def lunch_calories : ℕ := 885
def snack_calories : ℕ := 130

theorem calories_left_for_dinner :
  daily_allowance - (breakfast_calories + lunch_calories + snack_calories) = 832 := by
  sorry

end calories_left_for_dinner_l1894_189475


namespace probability_is_correct_l1894_189427

/-- The set of numbers from which we're selecting -/
def number_set : Set Nat := {n | 60 ≤ n ∧ n ≤ 1000}

/-- Predicate for a number being two-digit and divisible by 3 -/
def is_two_digit_div_by_three (n : Nat) : Prop := 60 ≤ n ∧ n ≤ 99 ∧ n % 3 = 0

/-- The count of numbers in the set -/
def total_count : Nat := 941

/-- The count of two-digit numbers divisible by 3 in the set -/
def favorable_count : Nat := 14

/-- The probability of selecting a two-digit number divisible by 3 from the set -/
def probability : Rat := favorable_count / total_count

theorem probability_is_correct : probability = 14 / 941 := by
  sorry

end probability_is_correct_l1894_189427


namespace max_value_x_minus_2y_l1894_189461

theorem max_value_x_minus_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) :
  x - 2*y ≤ 10 :=
by sorry

end max_value_x_minus_2y_l1894_189461


namespace pure_imaginary_quotient_condition_l1894_189449

theorem pure_imaginary_quotient_condition (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 3 - 4 * Complex.I
  (∃ (b : ℝ), z₁ / z₂ = b * Complex.I) → a = 4/3 := by
sorry

end pure_imaginary_quotient_condition_l1894_189449


namespace sqrt_expression_equals_sqrt_three_l1894_189466

theorem sqrt_expression_equals_sqrt_three :
  Real.sqrt 48 - Real.sqrt 2 * Real.sqrt 6 - Real.sqrt 15 / Real.sqrt 5 = Real.sqrt 3 := by
  sorry

end sqrt_expression_equals_sqrt_three_l1894_189466


namespace parallel_vectors_x_value_l1894_189467

def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b : ℝ × ℝ := (-3, 4)

def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (vector_a x) vector_b → x = -4/3 := by
  sorry

end parallel_vectors_x_value_l1894_189467


namespace soccer_penalty_kicks_l1894_189435

theorem soccer_penalty_kicks (total_players : ℕ) (goalies : ℕ) (h1 : total_players = 24) (h2 : goalies = 4) (h3 : goalies < total_players) :
  (total_players - 1) * goalies = 92 :=
by sorry

end soccer_penalty_kicks_l1894_189435


namespace tangent_three_implications_l1894_189480

theorem tangent_three_implications (α : Real) (h : Real.tan α = 3) :
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5/7 ∧
  1 - 4 * Real.sin α * Real.cos α + 2 * (Real.cos α)^2 = 0 := by
  sorry

end tangent_three_implications_l1894_189480


namespace sqrt_inequality_triangle_inequality_l1894_189404

-- Problem 1
theorem sqrt_inequality (a : ℝ) (ha : a > 0) :
  Real.sqrt (a + 5) - Real.sqrt (a + 3) > Real.sqrt (a + 6) - Real.sqrt (a + 4) := by
  sorry

-- Problem 2
theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  (a + b) / (1 + a + b) > c / (1 + c) := by
  sorry

end sqrt_inequality_triangle_inequality_l1894_189404


namespace molecular_weight_3_moles_Fe2SO43_l1894_189411

/-- The atomic weight of Iron in g/mol -/
def atomic_weight_Fe : ℝ := 55.845

/-- The atomic weight of Sulfur in g/mol -/
def atomic_weight_S : ℝ := 32.065

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The molecular weight of one mole of Iron(III) sulfate in g/mol -/
def molecular_weight_Fe2SO43 : ℝ :=
  2 * atomic_weight_Fe + 3 * (atomic_weight_S + 4 * atomic_weight_O)

/-- The number of moles of Iron(III) sulfate -/
def moles_Fe2SO43 : ℝ := 3

theorem molecular_weight_3_moles_Fe2SO43 :
  moles_Fe2SO43 * molecular_weight_Fe2SO43 = 1199.619 := by
  sorry

end molecular_weight_3_moles_Fe2SO43_l1894_189411


namespace prob_not_all_even_l1894_189491

/-- The number of sides on a fair die -/
def sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of even outcomes on a single die -/
def even_outcomes : ℕ := 3

/-- The probability that not all dice show an even number when rolling five fair 6-sided dice -/
theorem prob_not_all_even : 
  1 - (even_outcomes : ℚ) ^ num_dice / sides ^ num_dice = 7533 / 7776 := by
sorry

end prob_not_all_even_l1894_189491


namespace ranch_minimum_animals_l1894_189438

theorem ranch_minimum_animals : ∀ (ponies horses : ℕ),
  ponies > 0 →
  horses = ponies + 4 →
  (5 * ponies) % 6 = 0 →
  (10 * ponies) % 18 = 0 →
  ponies + horses ≥ 40 ∧
  ∀ (p h : ℕ), p > 0 → h = p + 4 → (5 * p) % 6 = 0 → (10 * p) % 18 = 0 → p + h ≥ ponies + horses :=
by sorry

end ranch_minimum_animals_l1894_189438


namespace base_for_125_with_4_digits_l1894_189488

theorem base_for_125_with_4_digits : ∃! b : ℕ, b > 1 ∧ b^3 ≤ 125 ∧ 125 < b^4 := by
  sorry

end base_for_125_with_4_digits_l1894_189488


namespace circles_intersect_l1894_189472

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 9

-- Define the distance between centers
def distance_between_centers : ℝ := 2

-- Define the radii of the circles
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  distance_between_centers > abs (radius1 - radius2) ∧
  distance_between_centers < radius1 + radius2 :=
sorry

end circles_intersect_l1894_189472


namespace quadratic_function_problem_l1894_189481

theorem quadratic_function_problem (a b : ℝ) : 
  (1^2 + a*1 + b = 2) → 
  ((-2)^2 + a*(-2) + b = -1) → 
  ((-3)^2 + a*(-3) + b = 2) :=
by
  sorry

end quadratic_function_problem_l1894_189481


namespace tangent_line_at_one_l1894_189450

noncomputable def f (x : ℝ) := Real.exp x * (x^2 - 2*x - 1)

theorem tangent_line_at_one (x y : ℝ) :
  let p := (1, f 1)
  let m := (Real.exp 1) * ((1:ℝ)^2 - 3)
  (y - f 1 = m * (x - 1)) ↔ (2 * Real.exp 1 * x + y = 0) :=
by sorry

end tangent_line_at_one_l1894_189450


namespace pebble_color_difference_l1894_189486

theorem pebble_color_difference (total_pebbles red_pebbles blue_pebbles : ℕ) 
  (h1 : total_pebbles = 40)
  (h2 : red_pebbles = 9)
  (h3 : blue_pebbles = 13)
  (h4 : (total_pebbles - red_pebbles - blue_pebbles) % 3 = 0) :
  blue_pebbles - (total_pebbles - red_pebbles - blue_pebbles) / 3 = 7 := by
  sorry

end pebble_color_difference_l1894_189486
