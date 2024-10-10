import Mathlib

namespace geometry_propositions_l1498_149850

-- Define the types for planes and lines
def Plane : Type := Unit
def Line : Type := Unit

-- Define the operations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def intersect (p1 p2 : Plane) : Line := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry

-- The theorem
theorem geometry_propositions (α β : Plane) (m n : Line) :
  (∀ α β m, perpendicular m α → perpendicular m β → parallel_planes α β) ∧
  ¬(∀ α β m n, parallel_line_plane m α → intersect α β = n → parallel_lines m n) ∧
  (∀ α m n, parallel_lines m n → perpendicular m α → perpendicular n α) ∧
  (∀ α β m n, perpendicular m α → parallel_lines m n → contained_in n β → perpendicular α β) :=
by sorry

end geometry_propositions_l1498_149850


namespace total_wine_age_l1498_149810

-- Define the ages of the wines
def carlo_rosi_age : ℕ := 40
def franzia_age : ℕ := 3 * carlo_rosi_age
def twin_valley_age : ℕ := carlo_rosi_age / 4

-- Theorem statement
theorem total_wine_age :
  franzia_age + carlo_rosi_age + twin_valley_age = 170 :=
by sorry

end total_wine_age_l1498_149810


namespace coins_in_second_stack_l1498_149893

theorem coins_in_second_stack (total_coins : ℕ) (first_stack : ℕ) (h1 : total_coins = 12) (h2 : first_stack = 4) :
  total_coins - first_stack = 8 := by
  sorry

end coins_in_second_stack_l1498_149893


namespace bridge_length_calculation_l1498_149867

/-- Calculates the length of a bridge given train parameters and crossing time -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 36 →
  crossing_time = 24.198064154867613 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 131.98064154867613 :=
by sorry

end bridge_length_calculation_l1498_149867


namespace product_mod_23_l1498_149832

theorem product_mod_23 : (2011 * 2012 * 2013 * 2014 * 2015) % 23 = 5 := by
  sorry

end product_mod_23_l1498_149832


namespace point_on_bisector_l1498_149894

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line y = x -/
def line_y_eq_x (p : Point) : Prop := p.y = p.x

theorem point_on_bisector (a b : ℝ) : 
  let A : Point := ⟨a, b⟩
  let B : Point := ⟨b, a⟩
  A = B → line_y_eq_x A := by
  sorry

end point_on_bisector_l1498_149894


namespace place_balls_count_l1498_149840

/-- The number of ways to place six numbered balls into six numbered boxes --/
def place_balls : ℕ :=
  let n : ℕ := 6  -- number of balls and boxes
  let k : ℕ := 2  -- number of balls placed in boxes with the same number
  let choose_two : ℕ := n.choose k
  let derangement_four : ℕ := 8  -- number of valid derangements for remaining 4 balls
  choose_two * derangement_four

/-- Theorem stating that the number of ways to place the balls is 120 --/
theorem place_balls_count : place_balls = 120 := by
  sorry

end place_balls_count_l1498_149840


namespace one_minus_repeating_thirds_l1498_149847

/-- The decimal 0.333... (repeating 3) -/
def repeating_thirds : ℚ :=
  1 / 3

theorem one_minus_repeating_thirds :
  1 - repeating_thirds = 2 / 3 := by
  sorry

end one_minus_repeating_thirds_l1498_149847


namespace ron_ticket_sales_l1498_149841

/-- Proves that Ron sold 12 student tickets given the problem conditions -/
theorem ron_ticket_sales
  (student_price : ℝ)
  (adult_price : ℝ)
  (total_tickets : ℕ)
  (total_income : ℝ)
  (h1 : student_price = 2)
  (h2 : adult_price = 4.5)
  (h3 : total_tickets = 20)
  (h4 : total_income = 60)
  : ∃ (student_tickets : ℕ) (adult_tickets : ℕ),
    student_tickets + adult_tickets = total_tickets ∧
    student_price * student_tickets + adult_price * adult_tickets = total_income ∧
    student_tickets = 12 :=
by sorry

end ron_ticket_sales_l1498_149841


namespace closest_integer_to_cube_root_200_l1498_149882

theorem closest_integer_to_cube_root_200 : 
  ∀ n : ℤ, |n^3 - 200| ≥ |6^3 - 200| :=
by
  sorry

end closest_integer_to_cube_root_200_l1498_149882


namespace equation_two_distinct_roots_l1498_149887

-- Define the equation
def equation (a x : ℝ) : Prop :=
  x + |x| = 2 * Real.sqrt (3 + 2*a*x - 4*a)

-- Define the set of valid 'a' values
def valid_a_set : Set ℝ :=
  {a | (0 < a ∧ a < 3/4) ∨ (a > 3)}

-- Theorem statement
theorem equation_two_distinct_roots (a : ℝ) :
  (∃ x y, x ≠ y ∧ equation a x ∧ equation a y) ↔ a ∈ valid_a_set :=
sorry

end equation_two_distinct_roots_l1498_149887


namespace quadratic_minimum_value_l1498_149876

/-- The minimum value of a quadratic function y = (x - a)(x - b) -/
theorem quadratic_minimum_value (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a^2 ≠ b^2) :
  ∃ x₀, ∀ x, (x - a) * (x - b) ≥ (x₀ - a) * (x₀ - b) ∧ 
  (x₀ - a) * (x₀ - b) = -(|a - b| / 2)^2 := by
  sorry

end quadratic_minimum_value_l1498_149876


namespace seventh_power_equation_l1498_149821

theorem seventh_power_equation (x : ℝ) (hx : x ≠ 0) :
  (7 * x)^5 = (14 * x)^4 ↔ x = 16/7 := by sorry

end seventh_power_equation_l1498_149821


namespace complex_fraction_equals_two_l1498_149813

theorem complex_fraction_equals_two (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 - c*d + d^2 = 0) : 
  (c^6 + d^6) / (c - d)^6 = 2 := by
  sorry

end complex_fraction_equals_two_l1498_149813


namespace max_distance_from_origin_l1498_149804

theorem max_distance_from_origin (x y : ℝ) :
  x^2 + y^2 - 4*x - 4*y + 6 = 0 →
  ∃ (max_val : ℝ), max_val = 3 * Real.sqrt 2 ∧
    ∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 →
      Real.sqrt (x'^2 + y'^2) ≤ max_val :=
by sorry

end max_distance_from_origin_l1498_149804


namespace common_factor_of_polynomial_l1498_149891

/-- The common factor of the polynomial 2m^2n + 6mn - 4m^3n is 2mn -/
theorem common_factor_of_polynomial (m n : ℤ) : 
  ∃ (k : ℤ), 2 * m^2 * n + 6 * m * n - 4 * m^3 * n = 2 * m * n * k :=
by sorry

end common_factor_of_polynomial_l1498_149891


namespace three_over_x_plus_one_is_fraction_l1498_149846

/-- A fraction is an expression where the denominator includes a variable. -/
def is_fraction (n d : ℝ → ℝ) : Prop :=
  ∃ x, d x ≠ d 0

/-- The expression 3/(x+1) is a fraction. -/
theorem three_over_x_plus_one_is_fraction :
  is_fraction (λ _ ↦ 3) (λ x ↦ x + 1) := by
sorry

end three_over_x_plus_one_is_fraction_l1498_149846


namespace only_one_always_true_l1498_149836

theorem only_one_always_true (a b c : ℝ) : 
  (∃! p : Prop, p = true) ∧ 
  (((a > b → a * c > b * c) = p) ∨
   ((a > b → a^2 * c^2 > b^2 * c^2) = p) ∨
   ((a^2 * c^2 > b^2 * c^2 → a > b) = p)) :=
by sorry

end only_one_always_true_l1498_149836


namespace cube_volume_from_surface_area_l1498_149888

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) : 
  surface_area = 150 → volume = (surface_area / 6) ^ (3/2) → volume = 125 := by
  sorry

end cube_volume_from_surface_area_l1498_149888


namespace matchstick_ratio_is_half_l1498_149844

/-- The ratio of matchsticks used to matchsticks originally had -/
def matchstick_ratio (houses : ℕ) (sticks_per_house : ℕ) (original_sticks : ℕ) : ℚ :=
  (houses * sticks_per_house : ℚ) / original_sticks

/-- Proof that the ratio of matchsticks used to matchsticks originally had is 1/2 -/
theorem matchstick_ratio_is_half :
  matchstick_ratio 30 10 600 = 1/2 := by
  sorry

end matchstick_ratio_is_half_l1498_149844


namespace initial_number_proof_l1498_149851

theorem initial_number_proof (x : ℝ) : ((x / 13) / 29) * (1/4) / 2 = 0.125 → x = 754 := by
  sorry

end initial_number_proof_l1498_149851


namespace odd_function_a_value_l1498_149824

-- Define the function f
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x =>
  if x > 0 then 1 + a^x else -1 - a^(-x)

-- State the theorem
theorem odd_function_a_value :
  ∀ a : ℝ,
  a > 0 →
  a ≠ 1 →
  (∀ x : ℝ, f a (-x) = -(f a x)) →
  f a (-1) = -3/2 →
  a = 1/2 :=
by
  sorry

end odd_function_a_value_l1498_149824


namespace parabola_line_intersection_l1498_149803

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  m : ℝ
  b : ℝ

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Theorem statement -/
theorem parabola_line_intersection (C : Parabola) (l : Line) (M N : ℝ × ℝ) :
  l.m = -Real.sqrt 3 ∧ l.b = Real.sqrt 3 →  -- Line equation: y = -√3(x-1)
  (C.p / 2, 0) ∈ {(x, y) | y = l.m * x + l.b} →  -- Line passes through focus
  M ∈ {(x, y) | y^2 = 2 * C.p * x} ∧ N ∈ {(x, y) | y^2 = 2 * C.p * x} →  -- M and N on parabola
  M ∈ {(x, y) | y = l.m * x + l.b} ∧ N ∈ {(x, y) | y = l.m * x + l.b} →  -- M and N on line
  ∃ (circ : Circle), circ.center = ((M.1 + N.1) / 2, (M.2 + N.2) / 2) ∧
                     circ.radius = Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 2 →
  C.p = 2 ∧  -- First conclusion
  circ.center.1 - circ.radius = -C.p / 2  -- Second conclusion: circle tangent to directrix
  := by sorry

end parabola_line_intersection_l1498_149803


namespace f_continuous_iff_l1498_149809

noncomputable def f (b c x : ℝ) : ℝ :=
  if x ≤ 4 then 2 * x^2 + 5
  else if x ≤ 6 then b * x + 3
  else c * x^2 - 2 * x + 9

theorem f_continuous_iff (b c : ℝ) :
  Continuous (f b c) ↔ b = 8.5 ∧ c = 19/12 := by sorry

end f_continuous_iff_l1498_149809


namespace total_pencils_count_l1498_149826

/-- The number of pencils each child has -/
def pencils_per_child : ℕ := 2

/-- The number of children -/
def number_of_children : ℕ := 8

/-- The total number of pencils -/
def total_pencils : ℕ := pencils_per_child * number_of_children

theorem total_pencils_count : total_pencils = 16 := by
  sorry

end total_pencils_count_l1498_149826


namespace triangle_inequality_l1498_149860

theorem triangle_inequality (A B C : ℝ) (h_triangle : A + B + C = π) : 
  (Real.sqrt (Real.sin A * Real.sin B) / Real.sin (C / 2)) + 
  (Real.sqrt (Real.sin B * Real.sin C) / Real.sin (A / 2)) + 
  (Real.sqrt (Real.sin C * Real.sin A) / Real.sin (B / 2)) ≥ 3 * Real.sqrt 3 := by
  sorry

end triangle_inequality_l1498_149860


namespace five_balls_three_boxes_count_l1498_149854

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 3 distinguishable boxes -/
def five_balls_three_boxes : ℕ := distribute_balls 5 3

theorem five_balls_three_boxes_count : five_balls_three_boxes = 21 := by sorry

end five_balls_three_boxes_count_l1498_149854


namespace perpendicular_vectors_difference_magnitude_l1498_149806

/-- Given two vectors a and b in ℝ², where a is perpendicular to b,
    prove that the magnitude of their difference is √10. -/
theorem perpendicular_vectors_difference_magnitude 
  (a b : ℝ × ℝ) 
  (h1 : a.1 = x ∧ a.2 = 1)
  (h2 : b = (1, -2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) : 
  ‖(a.1 - b.1, a.2 - b.2)‖ = Real.sqrt 10 :=
by sorry

end perpendicular_vectors_difference_magnitude_l1498_149806


namespace expression_evaluation_l1498_149871

theorem expression_evaluation :
  let a : ℝ := Real.sqrt 2 - 3
  (2*a + Real.sqrt 3) * (2*a - Real.sqrt 3) - 3*a*(a - 2) + 3 = -7 := by
  sorry

end expression_evaluation_l1498_149871


namespace percentage_students_like_blue_l1498_149848

/-- Proves that 30% of students like blue given the problem conditions --/
theorem percentage_students_like_blue :
  ∀ (total_students : ℕ) (blue_yellow_count : ℕ) (red_ratio : ℚ),
    total_students = 200 →
    blue_yellow_count = 144 →
    red_ratio = 2/5 →
    ∃ (blue_ratio : ℚ),
      blue_ratio = 3/10 ∧
      blue_ratio * total_students + 
      (1 - blue_ratio) * (1 - red_ratio) * total_students = blue_yellow_count :=
by sorry

end percentage_students_like_blue_l1498_149848


namespace quadratic_root_problem_l1498_149863

theorem quadratic_root_problem (m : ℝ) : 
  (3^2 - m * 3 + 3 = 0) → 
  (∃ (x : ℝ), x ≠ 3 ∧ x^2 - m * x + 3 = 0 ∧ x = 1) :=
by sorry

end quadratic_root_problem_l1498_149863


namespace platform_length_l1498_149869

/-- Calculates the length of a platform given the speed of a train, time to cross the platform, and length of the train. -/
theorem platform_length
  (train_speed_kmh : ℝ)
  (crossing_time_s : ℝ)
  (train_length_m : ℝ)
  (h1 : train_speed_kmh = 72)
  (h2 : crossing_time_s = 26)
  (h3 : train_length_m = 270) :
  let train_speed_ms : ℝ := train_speed_kmh * (5 / 18)
  let total_distance : ℝ := train_speed_ms * crossing_time_s
  let platform_length : ℝ := total_distance - train_length_m
  platform_length = 250 := by
  sorry

end platform_length_l1498_149869


namespace probability_two_in_same_group_l1498_149842

/-- The probability of two specific individuals being in the same group when dividing 4 individuals into two equal groups -/
def probability_same_group : ℚ := 1 / 3

/-- The number of ways to divide 4 individuals into two equal groups -/
def total_ways : ℕ := 3

/-- The number of ways to have two specific individuals in the same group when dividing 4 individuals into two equal groups -/
def favorable_ways : ℕ := 1

theorem probability_two_in_same_group :
  probability_same_group = favorable_ways / total_ways := by
  sorry

#eval probability_same_group

end probability_two_in_same_group_l1498_149842


namespace patty_score_proof_l1498_149845

def june_score : ℝ := 97
def josh_score : ℝ := 100
def henry_score : ℝ := 94
def average_score : ℝ := 94

theorem patty_score_proof (patty_score : ℝ) : 
  (june_score + josh_score + henry_score + patty_score) / 4 = average_score →
  patty_score = 85 := by
sorry

end patty_score_proof_l1498_149845


namespace max_stamps_for_50_dollars_max_stamps_is_maximum_l1498_149873

/-- The maximum number of stamps that can be purchased with a given budget and stamp price. -/
def maxStamps (budget : ℕ) (stampPrice : ℕ) : ℕ :=
  (budget / stampPrice : ℕ)

/-- Theorem stating the maximum number of stamps that can be purchased with $50 when each stamp costs 45 cents. -/
theorem max_stamps_for_50_dollars : maxStamps 5000 45 = 111 := by
  sorry

/-- Proof that the calculated maximum is indeed the largest possible number of stamps. -/
theorem max_stamps_is_maximum (budget : ℕ) (stampPrice : ℕ) :
  ∀ n : ℕ, n * stampPrice ≤ budget → n ≤ maxStamps budget stampPrice := by
  sorry

end max_stamps_for_50_dollars_max_stamps_is_maximum_l1498_149873


namespace smallest_factor_l1498_149852

theorem smallest_factor (w : ℕ) (other : ℕ) : 
  w = 144 →
  (∃ k : ℕ, w * other = k * 2^5) →
  (∃ k : ℕ, w * other = k * 3^3) →
  (∃ k : ℕ, w * other = k * 12^2) →
  (∀ x : ℕ, x < other → 
    (∃ k : ℕ, w * x = k * 2^5) ∧ 
    (∃ k : ℕ, w * x = k * 3^3) ∧ 
    (∃ k : ℕ, w * x = k * 12^2) → false) →
  other = 6 := by
sorry

end smallest_factor_l1498_149852


namespace ap_has_twelve_terms_l1498_149878

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  n : ℕ
  a : ℝ
  d : ℝ
  odd_sum : ℝ
  even_sum : ℝ
  last_term : ℝ
  third_term : ℝ

/-- The conditions of the arithmetic progression -/
def APConditions (ap : ArithmeticProgression) : Prop :=
  Even ap.n ∧
  ap.odd_sum = 36 ∧
  ap.even_sum = 42 ∧
  ap.last_term = ap.a + 12 ∧
  ap.third_term = 6 ∧
  ap.third_term = ap.a + 2 * ap.d ∧
  ap.odd_sum = (ap.n / 2 : ℝ) * (ap.a + (ap.a + (ap.n - 2) * ap.d)) ∧
  ap.even_sum = (ap.n / 2 : ℝ) * ((ap.a + ap.d) + (ap.a + (ap.n - 1) * ap.d))

/-- The theorem to be proved -/
theorem ap_has_twelve_terms (ap : ArithmeticProgression) :
  APConditions ap → ap.n = 12 := by
  sorry

end ap_has_twelve_terms_l1498_149878


namespace polynomial_equality_l1498_149830

theorem polynomial_equality (x : ℝ) : 
  let g : ℝ → ℝ := λ x => -2*x^5 + 4*x^4 - 12*x^3 + 2*x^2 + 4*x + 4
  2*x^5 + 3*x^3 - 4*x + 1 + g x = 4*x^4 - 9*x^3 + 2*x^2 + 5 := by
  sorry

end polynomial_equality_l1498_149830


namespace trivia_team_groups_l1498_149857

/-- Given a total number of students, number of students not picked, and number of students per group,
    calculate the number of groups formed. -/
def calculate_groups (total : ℕ) (not_picked : ℕ) (per_group : ℕ) : ℕ :=
  (total - not_picked) / per_group

/-- Theorem stating that with 65 total students, 17 not picked, and 6 per group, 8 groups are formed. -/
theorem trivia_team_groups : calculate_groups 65 17 6 = 8 := by
  sorry

end trivia_team_groups_l1498_149857


namespace bookstore_ratio_l1498_149884

theorem bookstore_ratio : 
  ∀ (sarah_paperback sarah_hardback brother_total : ℕ),
    sarah_paperback = 6 →
    sarah_hardback = 4 →
    brother_total = 10 →
    ∃ (brother_paperback brother_hardback : ℕ),
      brother_paperback = sarah_paperback / 3 →
      brother_hardback + brother_paperback = brother_total →
      (brother_hardback : ℚ) / sarah_hardback = 2 / 1 :=
by sorry

end bookstore_ratio_l1498_149884


namespace max_value_of_f_l1498_149883

-- Define the function
def f (x : ℝ) : ℝ := 5 * x - 4 * x^2 + 6

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), (∀ (x : ℝ), f x ≤ max) ∧ (max = 121 / 16) := by
  sorry

end max_value_of_f_l1498_149883


namespace sqrt_four_fourth_powers_l1498_149817

theorem sqrt_four_fourth_powers : Real.sqrt (4^4 + 4^4 + 4^4 + 4^4) = 32 := by
  sorry

end sqrt_four_fourth_powers_l1498_149817


namespace even_count_in_pascal_triangle_l1498_149880

/-- Pascal's Triangle coefficient -/
def binomial (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Check if a natural number is even -/
def isEven (n : ℕ) : Bool :=
  n % 2 = 0

/-- Count even numbers in a single row of Pascal's Triangle -/
def countEvenInRow (row : ℕ) : ℕ :=
  (List.range (row + 1)).filter (fun k => isEven (binomial row k)) |>.length

/-- Count even numbers in the first n rows of Pascal's Triangle -/
def countEvenInTriangle (n : ℕ) : ℕ :=
  (List.range n).map countEvenInRow |>.sum

/-- Theorem: There are 64 even integers in the first 15 rows of Pascal's Triangle -/
theorem even_count_in_pascal_triangle : countEvenInTriangle 15 = 64 := by
  sorry

end even_count_in_pascal_triangle_l1498_149880


namespace last_digit_to_appear_is_zero_l1498_149833

def modifiedFibonacci : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | n + 2 => modifiedFibonacci (n + 1) + modifiedFibonacci n

def unitsDigit (n : ℕ) : ℕ := n % 10

def allDigitsAppeared (n : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → ∃ k : ℕ, k ≤ n ∧ unitsDigit (modifiedFibonacci k) = d

theorem last_digit_to_appear_is_zero :
  ∃ N : ℕ, allDigitsAppeared N ∧
    ¬(allDigitsAppeared (N - 1)) ∧
    unitsDigit (modifiedFibonacci N) = 0 :=
  sorry

end last_digit_to_appear_is_zero_l1498_149833


namespace first_quarter_2016_has_91_days_l1498_149828

/-- The number of days in the first quarter of 2016 -/
def first_quarter_days_2016 : ℕ :=
  let year := 2016
  let is_leap_year := year % 4 = 0
  let february_days := if is_leap_year then 29 else 28
  let january_days := 31
  let march_days := 31
  january_days + february_days + march_days

/-- Theorem stating that the first quarter of 2016 has 91 days -/
theorem first_quarter_2016_has_91_days :
  first_quarter_days_2016 = 91 := by
  sorry

end first_quarter_2016_has_91_days_l1498_149828


namespace fraction_equality_l1498_149807

theorem fraction_equality : (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
                            (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end fraction_equality_l1498_149807


namespace power_multiplication_l1498_149861

theorem power_multiplication (t : ℝ) : t^5 * t^2 = t^7 := by
  sorry

end power_multiplication_l1498_149861


namespace john_lost_socks_l1498_149886

/-- The number of individual socks lost given initial pairs and maximum remaining pairs -/
def socks_lost (initial_pairs : ℕ) (max_remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * max_remaining_pairs

theorem john_lost_socks (initial_pairs : ℕ) (max_remaining_pairs : ℕ) 
  (h1 : initial_pairs = 10) (h2 : max_remaining_pairs = 7) : 
  socks_lost initial_pairs max_remaining_pairs = 6 := by
  sorry

#eval socks_lost 10 7

end john_lost_socks_l1498_149886


namespace approximate_number_properties_l1498_149889

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : Float
  exponent : Int

/-- Determines if a number is accurate to a specific place value -/
def is_accurate_to (n : ScientificNotation) (place : Int) : Prop :=
  sorry

/-- Counts the number of significant figures in a number -/
def count_significant_figures (n : ScientificNotation) : Nat :=
  sorry

/-- The hundreds place value -/
def hundreds : Int :=
  2

theorem approximate_number_properties (n : ScientificNotation) 
  (h1 : n.coefficient = 8.8)
  (h2 : n.exponent = 3) :
  is_accurate_to n hundreds ∧ count_significant_figures n = 2 := by
  sorry

end approximate_number_properties_l1498_149889


namespace max_table_sum_l1498_149892

def numbers : List ℕ := [2, 3, 5, 7, 11, 13]

def is_valid_arrangement (top : List ℕ) (left : List ℕ) : Prop :=
  top.length = 3 ∧ left.length = 3 ∧ (top ++ left).toFinset = numbers.toFinset

def table_sum (top : List ℕ) (left : List ℕ) : ℕ :=
  (top.sum * left.sum)

theorem max_table_sum :
  ∀ (top left : List ℕ), is_valid_arrangement top left →
    table_sum top left ≤ 420 :=
  sorry

end max_table_sum_l1498_149892


namespace erica_ride_duration_l1498_149870

/-- The duration in minutes that Dave can ride the merry-go-round -/
def dave_duration : ℝ := 10

/-- The factor by which Chuck can ride longer than Dave -/
def chuck_factor : ℝ := 5

/-- The percentage longer that Erica can ride compared to Chuck -/
def erica_percentage : ℝ := 0.30

/-- The duration in minutes that Chuck can ride the merry-go-round -/
def chuck_duration : ℝ := dave_duration * chuck_factor

/-- The duration in minutes that Erica can ride the merry-go-round -/
def erica_duration : ℝ := chuck_duration * (1 + erica_percentage)

/-- Theorem stating that Erica can ride for 65 minutes -/
theorem erica_ride_duration : erica_duration = 65 := by sorry

end erica_ride_duration_l1498_149870


namespace systematic_sample_smallest_number_l1498_149818

/-- Systematic sampling function -/
def systematicSample (n : ℕ) (k : ℕ) (i : ℕ) : ℕ := i * k

/-- Proposition: In a systematic sample of size 5 from 80 products, if 42 is in the sample, 
    then the smallest number in the sample is 10 -/
theorem systematic_sample_smallest_number :
  ∀ (i : ℕ), i < 5 →
  systematicSample 80 5 i = 42 →
  (∀ (j : ℕ), j < 5 → systematicSample 80 5 j ≥ 10) ∧
  (∃ (j : ℕ), j < 5 ∧ systematicSample 80 5 j = 10) :=
by sorry

end systematic_sample_smallest_number_l1498_149818


namespace jason_pokemon_cards_l1498_149875

/-- Given that Jason initially has 3 Pokemon cards and Benny buys 2 of them,
    prove that Jason will have 1 Pokemon card left. -/
theorem jason_pokemon_cards (initial_cards : ℕ) (cards_bought : ℕ) 
  (h1 : initial_cards = 3)
  (h2 : cards_bought = 2) :
  initial_cards - cards_bought = 1 := by
  sorry

end jason_pokemon_cards_l1498_149875


namespace inequality_theorem_l1498_149858

theorem inequality_theorem (x y : ℝ) : 
  x^2 + y^2 + 1 ≥ 2*(x*y - x + y) ∧ 
  (x^2 + y^2 + 1 = 2*(x*y - x + y) ↔ x = y - 1) := by
  sorry

end inequality_theorem_l1498_149858


namespace star_equation_roots_l1498_149834

-- Define the operation ※
def star (a b : ℝ) : ℝ := a^2 + a*b

-- Theorem statement
theorem star_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ star x 3 = -m) → m = 2 :=
by
  sorry

end star_equation_roots_l1498_149834


namespace unique_p_type_prime_l1498_149825

/-- A prime number q is a P-type prime if q + 1 is a perfect square. -/
def is_p_type_prime (q : ℕ) : Prop :=
  Nat.Prime q ∧ ∃ m : ℕ, q + 1 = m^2

/-- There exists exactly one P-type prime number. -/
theorem unique_p_type_prime : ∃! q : ℕ, is_p_type_prime q :=
sorry

end unique_p_type_prime_l1498_149825


namespace store_customers_l1498_149897

/-- Proves that the number of customers is 1000 given the specified conditions --/
theorem store_customers (return_rate : ℝ) (book_price : ℝ) (final_sales : ℝ) :
  return_rate = 0.37 →
  book_price = 15 →
  final_sales = 9450 →
  (1 - return_rate) * book_price * (final_sales / ((1 - return_rate) * book_price)) = 1000 := by
sorry

#eval (1 - 0.37) * 15 * (9450 / ((1 - 0.37) * 15)) -- Should output 1000.0

end store_customers_l1498_149897


namespace candy_game_bounds_l1498_149816

/-- Represents the colors of candies -/
inductive Color
  | Yellow
  | Red
  | Green
  | Blue

/-- Represents a collection of candies -/
structure CandyCollection :=
  (yellow : ℕ)
  (red : ℕ)
  (green : ℕ)
  (blue : ℕ)

/-- Represents the game state -/
structure GameState :=
  (remaining : CandyCollection)
  (jia : CandyCollection)
  (yi : CandyCollection)

def total_candies (c : CandyCollection) : ℕ :=
  c.yellow + c.red + c.green + c.blue

/-- Yi's turn: takes two candies (or the last one if only one is left) -/
def yi_turn (state : GameState) : GameState :=
  sorry

/-- Jia's turn: takes one candy of each color from the remaining candies -/
def jia_turn (state : GameState) : GameState :=
  sorry

/-- Plays the game until all candies are taken -/
def play_game (initial_state : GameState) : GameState :=
  sorry

theorem candy_game_bounds :
  ∀ (initial : CandyCollection),
    total_candies initial = 22 →
    initial.yellow ≥ initial.red ∧
    initial.yellow ≥ initial.green ∧
    initial.yellow ≥ initial.blue →
    let final_state := play_game { remaining := initial, jia := ⟨0,0,0,0⟩, yi := ⟨0,0,0,0⟩ }
    total_candies final_state.jia = total_candies final_state.yi →
    8 ≤ initial.yellow ∧ initial.yellow ≤ 16 :=
  sorry

end candy_game_bounds_l1498_149816


namespace change_for_fifty_cents_l1498_149835

/-- Represents the number of ways to make change for a given amount in cents -/
def makeChange (amount : ℕ) (maxQuarters : ℕ) : ℕ := sorry

/-- The value of a quarter in cents -/
def quarterValue : ℕ := 25

/-- The value of a nickel in cents -/
def nickelValue : ℕ := 5

/-- The value of a penny in cents -/
def pennyValue : ℕ := 1

theorem change_for_fifty_cents :
  makeChange 50 2 = 18 := by sorry

end change_for_fifty_cents_l1498_149835


namespace kitchen_cleaning_time_l1498_149874

theorem kitchen_cleaning_time (alice_time bob_fraction : ℚ) (h1 : alice_time = 40) (h2 : bob_fraction = 3/8) :
  bob_fraction * alice_time = 15 := by
  sorry

end kitchen_cleaning_time_l1498_149874


namespace sum_of_products_zero_l1498_149877

theorem sum_of_products_zero 
  (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (eq1 : x^2 + x*y + y^2 = 27)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 43) :
  x*y + y*z + x*z = 0 := by
sorry

end sum_of_products_zero_l1498_149877


namespace function_properties_l1498_149866

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 * Real.sin x + a

theorem function_properties (t : ℝ) :
  (∃ a : ℝ, f a π = 1 ∧ f a t = 2) →
  (∃ a : ℝ, a = 1 ∧ f a (-t) = 0) := by
  sorry

end function_properties_l1498_149866


namespace smallest_even_triangle_perimeter_l1498_149898

/-- A triangle with consecutive even number side lengths. -/
structure EvenTriangle where
  n : ℕ
  side1 : ℕ := 2 * n
  side2 : ℕ := 2 * n + 2
  side3 : ℕ := 2 * n + 4

/-- The triangle inequality holds for an EvenTriangle. -/
def satisfiesTriangleInequality (t : EvenTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The perimeter of an EvenTriangle. -/
def perimeter (t : EvenTriangle) : ℕ :=
  t.side1 + t.side2 + t.side3

/-- Theorem: The smallest possible perimeter of a triangle with consecutive even number
    side lengths that satisfies the triangle inequality is 18. -/
theorem smallest_even_triangle_perimeter :
  ∃ (t : EvenTriangle), satisfiesTriangleInequality t ∧
    perimeter t = 18 ∧
    ∀ (t' : EvenTriangle), satisfiesTriangleInequality t' → perimeter t' ≥ 18 :=
sorry

end smallest_even_triangle_perimeter_l1498_149898


namespace worker_wage_problem_l1498_149820

theorem worker_wage_problem (ordinary_rate : ℝ) (overtime_rate : ℝ) (total_hours : ℕ) 
  (overtime_hours : ℕ) (total_earnings : ℝ) :
  overtime_rate = 0.90 →
  total_hours = 50 →
  overtime_hours = 8 →
  total_earnings = 32.40 →
  ordinary_rate * (total_hours - overtime_hours : ℝ) + overtime_rate * overtime_hours = total_earnings →
  ordinary_rate = 0.60 := by
sorry

end worker_wage_problem_l1498_149820


namespace imaginary_part_of_z_l1498_149827

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 - Complex.I) = 3 + Complex.I) : 
  z.im = -2 := by sorry

end imaginary_part_of_z_l1498_149827


namespace sheila_hourly_wage_l1498_149805

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  hours_tue_thu : ℕ
  weekly_earnings : ℕ

/-- Calculates the total hours worked in a week --/
def total_hours (schedule : WorkSchedule) : ℕ :=
  3 * schedule.hours_mon_wed_fri + 2 * schedule.hours_tue_thu

/-- Calculates the hourly wage --/
def hourly_wage (schedule : WorkSchedule) : ℚ :=
  schedule.weekly_earnings / (total_hours schedule)

/-- Sheila's work schedule --/
def sheila_schedule : WorkSchedule :=
  { hours_mon_wed_fri := 8
  , hours_tue_thu := 6
  , weekly_earnings := 252 }

/-- Theorem stating that Sheila's hourly wage is $7 --/
theorem sheila_hourly_wage : hourly_wage sheila_schedule = 7 := by
  sorry


end sheila_hourly_wage_l1498_149805


namespace problem_solution_l1498_149837

theorem problem_solution (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_abc : a * b * c = 1)
  (h_a_c : a + 1 / c = 7)
  (h_b_a : b + 1 / a = 35) :
  c + 1 / b = 11 / 61 := by
sorry

end problem_solution_l1498_149837


namespace polynomial_coefficient_sum_l1498_149823

theorem polynomial_coefficient_sum (b₁ b₂ b₃ c₁ c₂ : ℝ) :
  (∀ x : ℝ, x^7 - x^6 + x^5 - x^4 + x^3 - x^2 + x - 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + 1)) →
  b₁*c₁ + b₂*c₂ + b₃ = -1 := by
sorry

end polynomial_coefficient_sum_l1498_149823


namespace equation_describes_line_l1498_149859

theorem equation_describes_line :
  ∀ (x y : ℝ), (x - y)^2 = 2*(x^2 + y^2) ↔ y = -x := by sorry

end equation_describes_line_l1498_149859


namespace effective_distance_is_seven_l1498_149855

/-- Calculates the effective distance walked given a constant walking rate, wind resistance reduction, and walking duration. -/
def effective_distance_walked (rate : ℝ) (wind_resistance : ℝ) (duration : ℝ) : ℝ :=
  (rate - wind_resistance) * duration

/-- Proves that given the specified conditions, the effective distance walked is 7 miles. -/
theorem effective_distance_is_seven :
  let rate : ℝ := 4
  let wind_resistance : ℝ := 0.5
  let duration : ℝ := 2
  effective_distance_walked rate wind_resistance duration = 7 := by
sorry

end effective_distance_is_seven_l1498_149855


namespace obtuse_triangle_proof_l1498_149822

theorem obtuse_triangle_proof (α : Real) (h1 : 0 < α ∧ α < π) 
  (h2 : Real.sin α + Real.cos α = 2/3) : π/2 < α := by
  sorry

end obtuse_triangle_proof_l1498_149822


namespace swimmers_pass_count_l1498_149849

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the swimming problem setup --/
structure SwimmingProblem where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  totalTime : ℝ

/-- Calculates the number of times the swimmers pass each other --/
def countPasses (problem : SwimmingProblem) : ℕ :=
  sorry

/-- The main theorem to prove --/
theorem swimmers_pass_count (problem : SwimmingProblem) 
  (h1 : problem.poolLength = 120)
  (h2 : problem.swimmer1.speed = 4)
  (h3 : problem.swimmer2.speed = 3)
  (h4 : problem.swimmer1.startPosition = 0)
  (h5 : problem.swimmer2.startPosition = 120)
  (h6 : problem.totalTime = 15 * 60) : 
  countPasses problem = 29 := by
  sorry

end swimmers_pass_count_l1498_149849


namespace guppies_theorem_l1498_149815

def guppies_problem (haylee_guppies : ℕ) (jose_ratio : ℚ) (charliz_ratio : ℚ) (nicolai_ratio : ℕ) : Prop :=
  let jose_guppies := (haylee_guppies : ℚ) * jose_ratio
  let charliz_guppies := jose_guppies * charliz_ratio
  let nicolai_guppies := (charliz_guppies * nicolai_ratio : ℚ)
  (haylee_guppies : ℚ) + jose_guppies + charliz_guppies + nicolai_guppies = 84

theorem guppies_theorem :
  guppies_problem 36 (1/2) (1/3) 4 :=
sorry

end guppies_theorem_l1498_149815


namespace quadratic_roots_range_l1498_149802

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 1 ∧ x₂ > 1 ∧ x₁^2 - 4*x₁ + a = 0 ∧ x₂^2 - 4*x₂ + a = 0) 
  ↔ 
  (3 < a ∧ a ≤ 4) :=
sorry

end quadratic_roots_range_l1498_149802


namespace centroid_dot_product_l1498_149819

/-- Triangle ABC with centroid G -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ

/-- Vector from point P to point Q -/
def vector (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Squared distance between two points -/
def distance_squared (P Q : ℝ × ℝ) : ℝ := (Q.1 - P.1)^2 + (Q.2 - P.2)^2

theorem centroid_dot_product (t : Triangle) : 
  (distance_squared t.A t.B = 1) →
  (distance_squared t.B t.C = 2) →
  (distance_squared t.A t.C = 3) →
  (t.G = ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)) →
  (dot_product (vector t.A t.G) (vector t.A t.C) = 4/3) := by
  sorry

end centroid_dot_product_l1498_149819


namespace range_of_g_l1498_149872

noncomputable def g (x : ℝ) : ℝ := Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_of_g :
  Set.range g = {-π/3, π/3} := by sorry

end range_of_g_l1498_149872


namespace probability_is_one_eighth_l1498_149843

/-- A standard die with 8 sides -/
def StandardDie : Finset ℕ := Finset.range 8

/-- The set of all possible outcomes when rolling the die twice -/
def AllOutcomes : Finset (ℕ × ℕ) := StandardDie.product StandardDie

/-- The set of favorable outcomes (pairs that differ by 3) -/
def FavorableOutcomes : Finset (ℕ × ℕ) :=
  AllOutcomes.filter (fun p => (p.1 + 3 = p.2) ∨ (p.2 + 3 = p.1))

/-- The probability of rolling two integers that differ by 3 -/
def probability : ℚ := (FavorableOutcomes.card : ℚ) / (AllOutcomes.card : ℚ)

theorem probability_is_one_eighth :
  probability = 1 / 8 := by sorry

end probability_is_one_eighth_l1498_149843


namespace min_value_of_function_l1498_149800

theorem min_value_of_function (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ y : ℝ, y > 0 ∧ y < 1 → (4 / x + 1 / (1 - x)) ≤ (4 / y + 1 / (1 - y))) ∧
  (∃ z : ℝ, z > 0 ∧ z < 1 ∧ 4 / z + 1 / (1 - z) = 9) :=
sorry

end min_value_of_function_l1498_149800


namespace remaining_clothing_problem_l1498_149812

/-- The number of remaining pieces of clothing to fold -/
def remaining_clothing (total_shirts : ℕ) (total_shorts : ℕ) (folded_shirts : ℕ) (folded_shorts : ℕ) : ℕ :=
  (total_shirts - folded_shirts) + (total_shorts - folded_shorts)

/-- Theorem stating that given 20 shirts and 8 pairs of shorts, if 12 shirts and 5 shorts are folded,
    the remaining number of pieces of clothing to fold is 11. -/
theorem remaining_clothing_problem :
  remaining_clothing 20 8 12 5 = 11 := by
  sorry

end remaining_clothing_problem_l1498_149812


namespace function_inequality_solution_l1498_149839

theorem function_inequality_solution (f g h : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, (x - y) * (f x) + h x - x * y + y^2 ≤ h y)
  (h2 : ∀ x y : ℝ, h y ≤ (x - y) * (g x) + h x - x * y + y^2) :
  ∃ a b : ℝ, 
    (∀ x : ℝ, f x = -x + a) ∧ 
    (∀ x : ℝ, g x = -x + a) ∧ 
    (∀ x : ℝ, h x = x^2 - a*x + b) := by
  sorry

end function_inequality_solution_l1498_149839


namespace bottle_cap_collection_l1498_149864

/-- Given that 7 bottle caps weigh one ounce and a collection of bottle caps weighs 18 pounds,
    prove that the number of bottle caps in the collection is 2016. -/
theorem bottle_cap_collection (caps_per_ounce : ℕ) (collection_weight_pounds : ℕ) 
  (h1 : caps_per_ounce = 7)
  (h2 : collection_weight_pounds = 18) :
  caps_per_ounce * (collection_weight_pounds * 16) = 2016 := by
  sorry

#check bottle_cap_collection

end bottle_cap_collection_l1498_149864


namespace digit_150_of_one_thirteenth_l1498_149885

/-- The repeating cycle of the decimal representation of 1/13 -/
def cycle : List Nat := [0, 7, 6, 9, 2, 3]

/-- The length of the repeating cycle -/
def cycle_length : Nat := 6

/-- The position we're interested in -/
def target_position : Nat := 150

/-- Theorem: The 150th digit after the decimal point in the decimal 
    representation of 1/13 is 3 -/
theorem digit_150_of_one_thirteenth (h : cycle = [0, 7, 6, 9, 2, 3]) :
  cycle[target_position % cycle_length] = 3 := by
  sorry

end digit_150_of_one_thirteenth_l1498_149885


namespace mikes_score_l1498_149899

theorem mikes_score (max_score : ℕ) (pass_percentage : ℚ) (shortfall : ℕ) (actual_score : ℕ) : 
  max_score = 750 → 
  pass_percentage = 30 / 100 → 
  shortfall = 13 → 
  actual_score = max_score * pass_percentage - shortfall →
  actual_score = 212 :=
by sorry

end mikes_score_l1498_149899


namespace initial_deposit_proof_l1498_149829

def bank_account (initial_deposit : ℝ) : ℝ := 
  ((initial_deposit * 1.1 + 10) * 1.1 + 10)

theorem initial_deposit_proof (initial_deposit : ℝ) : 
  bank_account initial_deposit = 142 → initial_deposit = 100 := by
sorry

end initial_deposit_proof_l1498_149829


namespace octal_subtraction_l1498_149838

/-- Converts a base-8 number represented as a list of digits to a natural number -/
def fromOctal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base-8 representation as a list of digits -/
def toOctal (n : Nat) : List Nat :=
  if n < 8 then [n]
  else (n % 8) :: toOctal (n / 8)

/-- The main theorem stating that 5273₈ - 3614₈ = 1457₈ -/
theorem octal_subtraction :
  fromOctal [3, 7, 2, 5] - fromOctal [4, 1, 6, 3] = fromOctal [7, 5, 4, 1] := by
  sorry

#eval toOctal (fromOctal [3, 7, 2, 5] - fromOctal [4, 1, 6, 3])

end octal_subtraction_l1498_149838


namespace simplify_expression_l1498_149865

theorem simplify_expression (a b c : ℝ) : a - (b - c) = a - b + c := by
  sorry

end simplify_expression_l1498_149865


namespace not_always_reducible_box_dimension_l1498_149881

structure RectangularParallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ
  positive_dimensions : 0 < length ∧ 0 < width ∧ 0 < height

def fits_in (p q : RectangularParallelepiped) : Prop :=
  p.length ≤ q.length ∧ p.width ≤ q.width ∧ p.height ≤ q.height

def is_defective (original defective : RectangularParallelepiped) : Prop :=
  (defective.length < original.length ∧ defective.width = original.width ∧ defective.height = original.height) ∨
  (defective.length = original.length ∧ defective.width < original.width ∧ defective.height = original.height) ∨
  (defective.length = original.length ∧ defective.width = original.width ∧ defective.height < original.height)

theorem not_always_reducible_box_dimension 
  (box : RectangularParallelepiped) 
  (parallelepipeds : List RectangularParallelepiped) 
  (original_parallelepipeds : List RectangularParallelepiped) 
  (h1 : ∀ p ∈ parallelepipeds, fits_in p box)
  (h2 : parallelepipeds.length = original_parallelepipeds.length)
  (h3 : ∀ (i : Fin parallelepipeds.length), is_defective (original_parallelepipeds[i]) (parallelepipeds[i])) :
  ¬ (∀ (reduced_box : RectangularParallelepiped), 
    (reduced_box.length < box.length ∨ reduced_box.width < box.width ∨ reduced_box.height < box.height) → 
    (∀ p ∈ parallelepipeds, fits_in p reduced_box)) :=
by sorry

end not_always_reducible_box_dimension_l1498_149881


namespace temperature_decrease_fraction_l1498_149879

theorem temperature_decrease_fraction (current_temp : ℝ) (decrease : ℝ) 
  (h1 : current_temp = 84)
  (h2 : decrease = 21) :
  (current_temp - decrease) / current_temp = 3/4 := by
sorry

end temperature_decrease_fraction_l1498_149879


namespace simplify_expression_l1498_149853

theorem simplify_expression (x : ℝ) : (3*x - 4)*(x + 9) - (x + 6)*(3*x + 2) = 3*x - 48 := by
  sorry

end simplify_expression_l1498_149853


namespace radish_carrot_ratio_l1498_149801

theorem radish_carrot_ratio :
  let cucumbers : ℕ := 15
  let radishes : ℕ := 3 * cucumbers
  let carrots : ℕ := 9
  (radishes : ℚ) / carrots = 5 := by
  sorry

end radish_carrot_ratio_l1498_149801


namespace min_sides_for_80_intersections_l1498_149890

/-- The number of intersection points between two n-sided polygons -/
def intersection_points (n : ℕ) : ℕ := 80

/-- Proposition: The minimum value of n for which two n-sided polygons can have exactly 80 intersection points is 10 -/
theorem min_sides_for_80_intersections :
  ∀ n : ℕ, intersection_points n = 80 → n ≥ 10 ∧ 
  ∃ (m : ℕ), m = 10 ∧ intersection_points m = 80 :=
sorry

end min_sides_for_80_intersections_l1498_149890


namespace sum_of_coordinates_of_B_l1498_149868

/-- Given two points A and B in a 2D plane, where A is at the origin and B is on the line y = 5,
    with the slope of the line AB being 3/4, prove that the sum of the x- and y-coordinates of B is 35/3. -/
theorem sum_of_coordinates_of_B (A B : ℝ × ℝ) : 
  A = (0, 0) →
  B.2 = 5 →
  (B.2 - A.2) / (B.1 - A.1) = 3 / 4 →
  B.1 + B.2 = 35 / 3 := by
  sorry

end sum_of_coordinates_of_B_l1498_149868


namespace unique_solution_l1498_149896

-- Define the circles
variable (A B C D E F : ℕ)

-- Define the conditions
def valid_arrangement (A B C D E F : ℕ) : Prop :=
  -- All numbers are between 1 and 6
  (A ∈ Finset.range 6) ∧ (B ∈ Finset.range 6) ∧ (C ∈ Finset.range 6) ∧
  (D ∈ Finset.range 6) ∧ (E ∈ Finset.range 6) ∧ (F ∈ Finset.range 6) ∧
  -- All numbers are distinct
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F ∧
  -- Sums on each line are equal
  A + C + D = A + B ∧
  A + C + D = B + D + F ∧
  A + C + D = E + F ∧
  A + C + D = E + B + C

-- Theorem statement
theorem unique_solution :
  ∀ A B C D E F : ℕ, valid_arrangement A B C D E F → A = 6 ∧ B = 3 :=
sorry


end unique_solution_l1498_149896


namespace donna_card_shop_days_l1498_149862

/-- Represents Donna's work schedule and earnings --/
structure DonnaWork where
  dog_walking_hours : ℕ
  dog_walking_rate : ℚ
  card_shop_hours : ℕ
  card_shop_rate : ℚ
  babysitting_hours : ℕ
  babysitting_rate : ℚ
  total_earnings : ℚ
  total_days : ℕ

/-- Calculates the number of days Donna worked at the card shop --/
def card_shop_days (work : DonnaWork) : ℚ :=
  let dog_walking_earnings := ↑work.dog_walking_hours * work.dog_walking_rate * ↑work.total_days
  let babysitting_earnings := ↑work.babysitting_hours * work.babysitting_rate
  let card_shop_earnings := work.total_earnings - dog_walking_earnings - babysitting_earnings
  card_shop_earnings / (↑work.card_shop_hours * work.card_shop_rate)

/-- Theorem stating that Donna worked 5 days at the card shop --/
theorem donna_card_shop_days :
  ∀ (work : DonnaWork),
  work.dog_walking_hours = 2 ∧
  work.dog_walking_rate = 10 ∧
  work.card_shop_hours = 2 ∧
  work.card_shop_rate = 25/2 ∧
  work.babysitting_hours = 4 ∧
  work.babysitting_rate = 10 ∧
  work.total_earnings = 305 ∧
  work.total_days = 7 →
  card_shop_days work = 5 := by
  sorry


end donna_card_shop_days_l1498_149862


namespace greatest_y_value_l1498_149808

theorem greatest_y_value (y : ℕ) (h1 : y > 0) (h2 : ∃ k : ℕ, y = 4 * k) (h3 : y^3 < 8000) :
  y ≤ 16 ∧ ∃ (y' : ℕ), y' = 16 ∧ ∃ (k : ℕ), y' = 4 * k ∧ y'^3 < 8000 :=
sorry

end greatest_y_value_l1498_149808


namespace australians_in_group_l1498_149831

theorem australians_in_group (total : Nat) (chinese : Nat) (americans : Nat) 
  (h1 : total = 49)
  (h2 : chinese = 22)
  (h3 : americans = 16) :
  total - (chinese + americans) = 11 := by
  sorry

end australians_in_group_l1498_149831


namespace expression_evaluation_l1498_149895

theorem expression_evaluation : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 2023) / (2023 * 2024) = -4044 := by
  sorry

end expression_evaluation_l1498_149895


namespace system_stable_l1498_149811

-- Define the system of differential equations
def system (x y : ℝ → ℝ) : Prop :=
  ∀ t, (deriv x t = -y t) ∧ (deriv y t = x t)

-- Define Lyapunov stability for the zero solution
def lyapunov_stable (x y : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x₀ y₀ : ℝ,
    x₀^2 + y₀^2 < δ^2 →
    (∀ t ≥ 0, x t^2 + y t^2 < ε^2) ∧
    (x 0 = x₀) ∧ (y 0 = y₀) ∧ system x y

-- Theorem statement
theorem system_stable :
  ∃ x y : ℝ → ℝ, lyapunov_stable x y ∧ system x y ∧ x 0 = 0 ∧ y 0 = 0 :=
sorry

end system_stable_l1498_149811


namespace fraction_deviation_from_sqrt_l1498_149856

theorem fraction_deviation_from_sqrt (x : ℝ) (h : 1 ≤ x ∧ x ≤ 9) : 
  |Real.sqrt x - (6 * x + 6) / (x + 11)| < 0.05 := by
  sorry

end fraction_deviation_from_sqrt_l1498_149856


namespace stairs_climbed_total_l1498_149814

theorem stairs_climbed_total (jonny_stairs : ℕ) (julia_stairs : ℕ) : 
  jonny_stairs = 1269 →
  julia_stairs = jonny_stairs / 3 - 7 →
  jonny_stairs + julia_stairs = 1685 :=
by sorry

end stairs_climbed_total_l1498_149814
