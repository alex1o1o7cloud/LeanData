import Mathlib

namespace minor_premise_identification_l1886_188647

-- Define the basic propositions
def ship_departs_on_time : Prop := sorry
def ship_arrives_on_time : Prop := sorry

-- Define the syllogism structure
structure Syllogism :=
  (major_premise : Prop)
  (minor_premise : Prop)
  (conclusion : Prop)

-- Define our specific syllogism
def our_syllogism : Syllogism :=
  { major_premise := ship_departs_on_time → ship_arrives_on_time,
    minor_premise := ship_arrives_on_time,
    conclusion := ship_departs_on_time }

-- Theorem to prove
theorem minor_premise_identification :
  our_syllogism.minor_premise = ship_arrives_on_time :=
by sorry

end minor_premise_identification_l1886_188647


namespace circular_garden_fence_area_ratio_l1886_188653

theorem circular_garden_fence_area_ratio (r : ℝ) (h : r = 12) : 
  (2 * Real.pi * r) / (Real.pi * r^2) = 1/6 := by
  sorry

end circular_garden_fence_area_ratio_l1886_188653


namespace absolute_value_equality_l1886_188645

theorem absolute_value_equality (x : ℝ) (y : ℝ) :
  y > 0 →
  |3 * x - 2 * Real.log y| = 3 * x + 2 * Real.log y →
  x = 0 ∧ y = 1 := by
  sorry

end absolute_value_equality_l1886_188645


namespace line_equation_l1886_188672

/-- A line passing through (2,3) with opposite-sign intercepts -/
structure LineWithOppositeIntercepts where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2,3)
  passes_through : 3 = m * 2 + b
  -- The line has opposite-sign intercepts
  opposite_intercepts : (b ≠ 0 ∧ (-b/m) * b < 0) ∨ (b = 0 ∧ m ≠ 0)

/-- The equation of the line is either 3x - 2y = 0 or x - y + 1 = 0 -/
theorem line_equation (l : LineWithOppositeIntercepts) :
  (l.m = 3/2 ∧ l.b = 0) ∨ (l.m = 1 ∧ l.b = -1) := by
  sorry

end line_equation_l1886_188672


namespace prob_two_red_before_three_green_is_two_sevenths_l1886_188691

/-- Represents the outcome of drawing chips from a hat -/
inductive DrawOutcome
| TwoRed
| ThreeGreen

/-- The probability of drawing 2 red chips before 3 green chips -/
def prob_two_red_before_three_green : ℚ :=
  2 / 7

/-- The number of red chips in the hat initially -/
def initial_red_chips : ℕ := 4

/-- The number of green chips in the hat initially -/
def initial_green_chips : ℕ := 3

/-- The total number of chips in the hat initially -/
def total_chips : ℕ := initial_red_chips + initial_green_chips

/-- Theorem stating that the probability of drawing 2 red chips before 3 green chips is 2/7 -/
theorem prob_two_red_before_three_green_is_two_sevenths :
  prob_two_red_before_three_green = 2 / 7 := by
  sorry

end prob_two_red_before_three_green_is_two_sevenths_l1886_188691


namespace new_triangle_is_acute_l1886_188667

-- Define a right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : c^2 = a^2 + b^2
  positive : 0 < a ∧ 0 < b ∧ 0 < c

-- Define the new triangle after increasing each side by x
def NewTriangle (t : RightTriangle) (x : ℝ) : Prop :=
  ∀ (h : 0 < x),
    let new_a := t.a + x
    let new_b := t.b + x
    let new_c := t.c + x
    (new_a^2 + new_b^2 - new_c^2) / (2 * new_a * new_b) > 0

-- Theorem statement
theorem new_triangle_is_acute (t : RightTriangle) :
  ∀ x, NewTriangle t x :=
sorry

end new_triangle_is_acute_l1886_188667


namespace arrangement_count_is_48_l1886_188620

/-- The number of ways to arrange 5 different items into two rows -/
def arrangement_count : ℕ := 48

/-- The total number of items -/
def total_items : ℕ := 5

/-- The minimum number of items in each row -/
def min_items_per_row : ℕ := 2

/-- The number of items that must be in the front row -/
def fixed_front_items : ℕ := 2

/-- Theorem stating that the number of arrangements is 48 -/
theorem arrangement_count_is_48 :
  arrangement_count = 48 ∧
  total_items = 5 ∧
  min_items_per_row = 2 ∧
  fixed_front_items = 2 :=
sorry

end arrangement_count_is_48_l1886_188620


namespace quadratic_point_relationship_l1886_188654

/-- A quadratic function of the form y = -(x-1)² + k -/
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := -(x - 1)^2 + k

theorem quadratic_point_relationship (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  quadratic_function k (-1) = y₁ →
  quadratic_function k 2 = y₂ →
  quadratic_function k 4 = y₃ →
  y₃ < y₁ ∧ y₁ < y₂ := by
  sorry

end quadratic_point_relationship_l1886_188654


namespace unique_intersection_point_l1886_188603

theorem unique_intersection_point (m : ℤ) : 
  (∃ (x : ℕ+), -3 * x + 2 = m * (x^2 - x + 1)) ↔ m = -1 :=
by sorry

end unique_intersection_point_l1886_188603


namespace power_of_power_l1886_188634

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l1886_188634


namespace school_classes_l1886_188613

theorem school_classes (daily_usage_per_class : ℕ) (weekly_usage_total : ℕ) (school_days_per_week : ℕ) :
  daily_usage_per_class = 200 →
  weekly_usage_total = 9000 →
  school_days_per_week = 5 →
  weekly_usage_total / school_days_per_week / daily_usage_per_class = 9 := by
sorry

end school_classes_l1886_188613


namespace tournament_outcomes_l1886_188610

/-- Represents a knockout tournament with 6 players -/
structure Tournament :=
  (num_players : Nat)
  (num_games : Nat)

/-- The number of possible outcomes for each game -/
def outcomes_per_game : Nat := 2

/-- Theorem stating that the number of possible prize orders is 32 -/
theorem tournament_outcomes (t : Tournament) (h1 : t.num_players = 6) (h2 : t.num_games = 5) : 
  outcomes_per_game ^ t.num_games = 32 := by
  sorry

#eval outcomes_per_game ^ 5

end tournament_outcomes_l1886_188610


namespace sin_beta_value_l1886_188630

-- Define acute angles
def is_acute (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- State the theorem
theorem sin_beta_value (α β : Real) 
  (h_acute_α : is_acute α) (h_acute_β : is_acute β)
  (h_sin_α : Real.sin α = (4/7) * Real.sqrt 3)
  (h_cos_sum : Real.cos (α + β) = -11/14) :
  Real.sin β = Real.sqrt 3 / 2 := by
  sorry

end sin_beta_value_l1886_188630


namespace intersection_A_B_l1886_188642

-- Define set A
def A : Set ℝ := {x | x^2 - 1 ≥ 0}

-- Define set B
def B : Set ℝ := {x | 1 ≤ x ∧ x < 3}

-- Theorem statement
theorem intersection_A_B : 
  ∀ x : ℝ, x ∈ A ∩ B ↔ 1 ≤ x ∧ x < 3 := by sorry

end intersection_A_B_l1886_188642


namespace unique_room_dimensions_l1886_188646

/-- A room with integer dimensions where the unpainted border area is four times the painted area --/
structure PaintedRoom where
  a : ℕ
  b : ℕ
  h1 : 0 < a
  h2 : 0 < b
  h3 : b > a
  h4 : 4 * ((a - 4) * (b - 4)) = a * b - (a - 4) * (b - 4)

/-- The only valid dimensions for the room are 6 by 30 feet --/
theorem unique_room_dimensions : 
  ∀ (room : PaintedRoom), room.a = 6 ∧ room.b = 30 :=
by sorry

end unique_room_dimensions_l1886_188646


namespace least_integer_abs_inequality_l1886_188695

theorem least_integer_abs_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), |3*y + 5| ≤ 20 → x ≤ y) ∧ |3*x + 5| ≤ 20 :=
by
  -- The proof goes here
  sorry

end least_integer_abs_inequality_l1886_188695


namespace det_A_eq_one_l1886_188659

/-- The matrix A_n as defined in the problem -/
def A (n : ℕ+) : Matrix (Fin n) (Fin n) ℚ :=
  λ i j => (i.val + j.val - 2).choose (j.val - 1)

/-- The theorem stating that the determinant of A_n is 1 for all positive integers n -/
theorem det_A_eq_one (n : ℕ+) : Matrix.det (A n) = 1 := by sorry

end det_A_eq_one_l1886_188659


namespace max_three_layer_structures_l1886_188601

theorem max_three_layer_structures :
  ∃ (a b c : ℕ),
    1 ≤ a ∧ a ≤ b - 2 ∧ b - 2 ≤ c - 4 ∧
    a^2 + b^2 + c^2 ≤ 1988 ∧
    ∀ (x y z : ℕ),
      1 ≤ x ∧ x ≤ y - 2 ∧ y - 2 ≤ z - 4 ∧
      x^2 + y^2 + z^2 ≤ 1988 →
      (b - a - 1)^2 * (c - b - 1)^2 ≥ (y - x - 1)^2 * (z - y - 1)^2 ∧
    (b - a - 1)^2 * (c - b - 1)^2 = 345 :=
by sorry

end max_three_layer_structures_l1886_188601


namespace stratified_sampling_young_teachers_l1886_188635

theorem stratified_sampling_young_teachers 
  (total_teachers : ℕ) 
  (young_teachers : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_teachers = 200)
  (h2 : young_teachers = 100)
  (h3 : sample_size = 40) :
  (young_teachers : ℚ) / (total_teachers : ℚ) * (sample_size : ℚ) = 20 := by
  sorry

end stratified_sampling_young_teachers_l1886_188635


namespace coefficient_x10_expansion_l1886_188623

/-- The coefficient of x^10 in the expansion of ((1+x+x^2)(1-x)^10) is 36 -/
theorem coefficient_x10_expansion : 
  let f : ℕ → ℤ := fun n => 
    (Finset.range (n + 1)).sum (fun k => 
      (-1)^k * (Nat.choose n k) * (Finset.range 3).sum (fun i => Nat.choose 2 i * k^(2-i)))
  f 10 = 36 := by
  sorry

end coefficient_x10_expansion_l1886_188623


namespace valid_arrangement_exists_l1886_188632

/-- Represents an arrangement of numbers satisfying the given conditions -/
def ValidArrangement (n : ℕ) := List ℕ

/-- Checks if the arrangement is valid for a given n -/
def isValidArrangement (n : ℕ) (arr : ValidArrangement n) : Prop :=
  (arr.length = 2*n + 1) ∧
  (arr.count 0 = 1) ∧
  (∀ m : ℕ, m ≥ 1 → m ≤ n → arr.count m = 2) ∧
  (∀ m : ℕ, m ≥ 1 → m ≤ n → 
    ∃ i j : ℕ, i < j ∧ 
    (arr.get! i = m) ∧ 
    (arr.get! j = m) ∧ 
    (j - i - 1 = m))

/-- Theorem stating that a valid arrangement exists for any natural number n -/
theorem valid_arrangement_exists (n : ℕ) : ∃ arr : ValidArrangement n, isValidArrangement n arr :=
sorry

end valid_arrangement_exists_l1886_188632


namespace min_total_time_for_three_students_l1886_188663

/-- Represents a student with their bucket filling time -/
structure Student where
  name : String
  fillTime : Real

/-- Calculates the minimum total time for students to fill their buckets -/
def minTotalTime (students : List Student) : Real :=
  sorry

/-- Theorem stating the minimum total time for the given scenario -/
theorem min_total_time_for_three_students :
  let students := [
    { name := "A", fillTime := 1.5 },
    { name := "B", fillTime := 0.5 },
    { name := "C", fillTime := 1.0 }
  ]
  minTotalTime students = 5 := by sorry

end min_total_time_for_three_students_l1886_188663


namespace product_remainder_l1886_188698

theorem product_remainder (a b c : ℕ) (ha : a = 2456) (hb : b = 8743) (hc : c = 92431) :
  (a * b * c) % 10 = 8 := by
  sorry

end product_remainder_l1886_188698


namespace line_equation_l1886_188639

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y + 21 = 0

-- Define point A
def point_A : ℝ × ℝ := (-6, 7)

-- Define the property of being tangent to the circle
def is_tangent_to_circle (a b c : ℝ) : Prop :=
  let center := (4, -3)
  let radius := 2
  abs (a * center.1 + b * center.2 + c) / Real.sqrt (a^2 + b^2) = radius

-- Theorem statement
theorem line_equation :
  ∃ (a b c : ℝ), 
    (a * point_A.1 + b * point_A.2 + c = 0) ∧
    is_tangent_to_circle a b c ∧
    ((a = 3 ∧ b = 4 ∧ c = -10) ∨ (a = 4 ∧ b = 3 ∧ c = 3)) := by
  sorry

end line_equation_l1886_188639


namespace sqrt_40_simplification_l1886_188633

theorem sqrt_40_simplification : Real.sqrt 40 = 2 * Real.sqrt 10 := by
  sorry

end sqrt_40_simplification_l1886_188633


namespace trigonometric_expression_equals_one_third_l1886_188658

theorem trigonometric_expression_equals_one_third :
  let θ : Real := 30 * π / 180  -- 30 degrees in radians
  (Real.tan θ)^2 - (Real.cos θ)^2 = 1/3 * ((Real.tan θ)^2 * (Real.cos θ)^2) := by
  sorry

end trigonometric_expression_equals_one_third_l1886_188658


namespace odd_tau_tau_count_l1886_188631

/-- The number of positive integer divisors of n -/
def τ (n : ℕ) : ℕ := sorry

/-- Predicate to check if a natural number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- The count of integers n between 1 and 50 (inclusive) such that τ(τ(n)) is odd -/
def countOddTauTau : ℕ := sorry

theorem odd_tau_tau_count : countOddTauTau = 17 := by sorry

end odd_tau_tau_count_l1886_188631


namespace right_triangle_tan_A_l1886_188649

theorem right_triangle_tan_A (A B C : Real) (sinB : Real) :
  -- ABC is a right triangle with angle C = 90°
  A + B + C = Real.pi →
  C = Real.pi / 2 →
  -- sin B = 3/5
  sinB = 3 / 5 →
  -- tan A = 4/3
  Real.tan A = 4 / 3 := by
  sorry

end right_triangle_tan_A_l1886_188649


namespace intersection_A_B_when_a_4_A_subset_B_condition_l1886_188609

-- Define the sets A and B
def A : Set ℝ := {x | (1 - x) / (x - 7) > 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - 2*x - a^2 - 2*a < 0}

-- Theorem 1: Intersection of A and B when a = 4
theorem intersection_A_B_when_a_4 : A ∩ B 4 = {x | 1 < x ∧ x < 6} := by sorry

-- Theorem 2: Condition for A to be a subset of B
theorem A_subset_B_condition (a : ℝ) : A ⊆ B a ↔ a ≤ -7 ∨ a ≥ 5 := by sorry

end intersection_A_B_when_a_4_A_subset_B_condition_l1886_188609


namespace second_class_average_mark_l1886_188622

theorem second_class_average_mark (students1 students2 : ℕ) (avg1 avg_total : ℚ) :
  students1 = 30 →
  students2 = 50 →
  avg1 = 40 →
  avg_total = 58.75 →
  (students1 : ℚ) * avg1 + (students2 : ℚ) * ((students1 + students2 : ℚ) * avg_total - (students1 : ℚ) * avg1) / (students2 : ℚ) =
    (students1 + students2 : ℚ) * avg_total →
  ((students1 + students2 : ℚ) * avg_total - (students1 : ℚ) * avg1) / (students2 : ℚ) = 70 :=
by sorry

end second_class_average_mark_l1886_188622


namespace arctan_equation_solution_l1886_188605

theorem arctan_equation_solution :
  ∃ x : ℝ, 2 * Real.arctan (1/5) + 2 * Real.arctan (1/10) + Real.arctan (1/x) = π/2 ∧ x = 120/119 := by
  sorry

end arctan_equation_solution_l1886_188605


namespace product_of_invertible_labels_l1886_188680

def is_invertible (f : ℕ → Bool) := f 2 = false ∧ f 3 = true ∧ f 4 = true ∧ f 5 = true

theorem product_of_invertible_labels (f : ℕ → Bool) (h : is_invertible f) :
  (List.filter (λ i => f i) [2, 3, 4, 5]).prod = 60 :=
by sorry

end product_of_invertible_labels_l1886_188680


namespace lemonade_stand_operational_cost_l1886_188607

/-- Yulia's lemonade stand finances -/
def lemonade_stand_finances (net_profit babysitting_revenue lemonade_revenue : ℕ) : Prop :=
  ∃ (operational_cost : ℕ),
    net_profit + operational_cost = babysitting_revenue + lemonade_revenue ∧
    operational_cost = 34

/-- Theorem: Given Yulia's financial information, prove that her lemonade stand's operational cost is $34 -/
theorem lemonade_stand_operational_cost :
  lemonade_stand_finances 44 31 47 :=
by
  sorry

end lemonade_stand_operational_cost_l1886_188607


namespace john_notebooks_l1886_188682

/-- Calculates the maximum number of notebooks that can be purchased with a given amount of money, considering a bulk discount. -/
def max_notebooks (total_cents : ℕ) (notebook_price : ℕ) (discount : ℕ) (bulk_size : ℕ) : ℕ :=
  let discounted_price := notebook_price - discount
  let bulk_set_price := discounted_price * bulk_size
  let bulk_sets := total_cents / bulk_set_price
  let remaining_cents := total_cents % bulk_set_price
  let additional_notebooks := remaining_cents / notebook_price
  bulk_sets * bulk_size + additional_notebooks

/-- Proves that given 2545 cents, with notebooks costing 235 cents each and a 15 cent discount
    per notebook when bought in sets of 5, the maximum number of notebooks that can be purchased is 11. -/
theorem john_notebooks : max_notebooks 2545 235 15 5 = 11 := by
  sorry

end john_notebooks_l1886_188682


namespace not_sufficient_nor_necessary_l1886_188606

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ 
  (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by sorry

end not_sufficient_nor_necessary_l1886_188606


namespace complement_B_intersect_A_m_value_for_intersection_l1886_188621

-- Define set A
def A : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}

-- Define set B with parameter m
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1
theorem complement_B_intersect_A :
  (Set.compl (B 3) ∩ A) = {x | 3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2
theorem m_value_for_intersection :
  ∃ m : ℝ, (A ∩ B m) = {x | -1 < x ∧ x < 4} → m = 8 := by sorry

end complement_B_intersect_A_m_value_for_intersection_l1886_188621


namespace cattle_area_calculation_l1886_188671

def farm_length : ℝ := 3.6

theorem cattle_area_calculation (width : ℝ) (total_area : ℝ) (cattle_area : ℝ)
  (h1 : width = 2.5 * farm_length)
  (h2 : total_area = farm_length * width)
  (h3 : cattle_area = total_area / 2) :
  cattle_area = 16.2 := by
  sorry

end cattle_area_calculation_l1886_188671


namespace power_64_five_sixths_l1886_188661

theorem power_64_five_sixths : (64 : ℝ) ^ (5/6) = 32 := by sorry

end power_64_five_sixths_l1886_188661


namespace smallest_n_congruence_two_satisfies_congruence_two_is_smallest_smallest_positive_integer_congruence_l1886_188664

theorem smallest_n_congruence (n : ℕ) : n > 0 ∧ 527 * n ≡ 1083 * n [ZMOD 30] → n ≥ 2 :=
by sorry

theorem two_satisfies_congruence : 527 * 2 ≡ 1083 * 2 [ZMOD 30] :=
by sorry

theorem two_is_smallest : ∀ m : ℕ, m > 0 ∧ 527 * m ≡ 1083 * m [ZMOD 30] → m ≥ 2 :=
by sorry

theorem smallest_positive_integer_congruence : 
  ∃! n : ℕ, n > 0 ∧ 527 * n ≡ 1083 * n [ZMOD 30] ∧ ∀ m : ℕ, (m > 0 ∧ 527 * m ≡ 1083 * m [ZMOD 30] → m ≥ n) :=
by sorry

end smallest_n_congruence_two_satisfies_congruence_two_is_smallest_smallest_positive_integer_congruence_l1886_188664


namespace hyperbola_equation_l1886_188668

/-- The equation of a hyperbola passing through a specific point with its asymptote tangent to a circle -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (16 / a^2 - 4 / b^2 = 1) →  -- Hyperbola passes through (4, 2)
  (|2 * Real.sqrt 2 * b| / Real.sqrt (b^2 + a^2) = Real.sqrt (8/3)) →  -- Asymptote tangent to circle
  (∀ x y : ℝ, x^2 / 8 - y^2 / 4 = 1 ↔ x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end hyperbola_equation_l1886_188668


namespace race_distance_l1886_188602

/-- 
Given a race with two contestants A and B, where:
- The ratio of speeds of A and B is 3:4
- A has a start of 140 meters
- A wins by 20 meters

Prove that the total distance of the race is 480 meters.
-/
theorem race_distance (speed_A speed_B : ℝ) (total_distance : ℝ) : 
  speed_A / speed_B = 3 / 4 →
  total_distance - (total_distance - 140 + 20) = speed_A / speed_B * total_distance →
  total_distance = 480 := by
  sorry

end race_distance_l1886_188602


namespace sqrt3_times_sqrt10_minus_sqrt3_bounds_l1886_188652

theorem sqrt3_times_sqrt10_minus_sqrt3_bounds :
  2 < Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3) ∧
  Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3) < 3 :=
by sorry

end sqrt3_times_sqrt10_minus_sqrt3_bounds_l1886_188652


namespace bill_score_l1886_188627

theorem bill_score (john sue bill : ℕ) 
  (score_diff : bill = john + 20)
  (bill_half_sue : bill * 2 = sue)
  (total_score : john + bill + sue = 160) :
  bill = 45 := by
sorry

end bill_score_l1886_188627


namespace book_arrangement_count_l1886_188673

/-- The number of ways to arrange books on a shelf --/
def arrange_books (math_books : ℕ) (english_books : ℕ) : ℕ :=
  Nat.factorial 3 * Nat.factorial math_books * Nat.factorial english_books

/-- Theorem stating the number of ways to arrange 4 math books, 7 English books, and 1 journal --/
theorem book_arrangement_count :
  arrange_books 4 7 = 725760 :=
by sorry

end book_arrangement_count_l1886_188673


namespace repeating_decimal_to_fraction_l1886_188626

theorem repeating_decimal_to_fraction :
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (n : ℚ) / d = 7 + (789 : ℚ) / 10000 / (1 - 1 / 10000) :=
by
  -- The fraction 365/85 satisfies this property
  use 365, 85
  sorry

end repeating_decimal_to_fraction_l1886_188626


namespace negative_324_same_terminal_side_as_36_l1886_188644

/-- Two angles have the same terminal side if their difference is a multiple of 360 degrees -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, β - α = k * 360

/-- The main theorem: -324° has the same terminal side as 36° -/
theorem negative_324_same_terminal_side_as_36 :
  same_terminal_side 36 (-324) := by
  sorry

end negative_324_same_terminal_side_as_36_l1886_188644


namespace vehicle_speeds_and_distance_l1886_188684

theorem vehicle_speeds_and_distance (total_distance : ℝ) 
  (speed_ratio : ℝ) (time_delay : ℝ) :
  total_distance = 90 →
  speed_ratio = 1.5 →
  time_delay = 1/3 →
  ∃ (speed_slow speed_fast distance_traveled : ℝ),
    speed_slow = 90 ∧
    speed_fast = 135 ∧
    distance_traveled = 30 ∧
    speed_fast = speed_ratio * speed_slow ∧
    total_distance / speed_slow - total_distance / speed_fast = time_delay ∧
    distance_traveled = speed_slow * time_delay :=
by sorry

end vehicle_speeds_and_distance_l1886_188684


namespace oliver_gave_janet_ten_pounds_l1886_188662

/-- The amount of candy Oliver gave to Janet -/
def candy_given_to_janet (initial_candy : ℕ) (remaining_candy : ℕ) : ℕ :=
  initial_candy - remaining_candy

/-- Proof that Oliver gave Janet 10 pounds of candy -/
theorem oliver_gave_janet_ten_pounds :
  candy_given_to_janet 78 68 = 10 := by
  sorry

end oliver_gave_janet_ten_pounds_l1886_188662


namespace unique_positive_solution_l1886_188676

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (1/2 * (3*x^2 - 1) = (x^2 - 50*x - 10)*(x^2 + 25*x + 5)) ∧ x = 25 + Real.sqrt 159 := by
  sorry

end unique_positive_solution_l1886_188676


namespace inequality_solution_l1886_188624

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x - 45) / (x + 7) < 0 ↔ (x > -7 ∧ x < -5) ∨ (x > -5 ∧ x < 9) :=
by sorry

end inequality_solution_l1886_188624


namespace candy_box_price_increase_l1886_188685

theorem candy_box_price_increase : 
  ∀ (original_candy_price original_soda_price : ℝ),
  original_candy_price + original_soda_price = 20 →
  original_soda_price * 1.5 = 6 →
  original_candy_price * 1.25 = 20 →
  (20 - original_candy_price) / original_candy_price = 0.25 :=
by
  sorry

end candy_box_price_increase_l1886_188685


namespace f_is_even_g_is_odd_h_is_neither_l1886_188669

-- Define the functions
def f (x : ℝ) : ℝ := 1 + x^2 + x^4
def g (x : ℝ) : ℝ := x + x^3 + x^5
def h (x : ℝ) : ℝ := 1 + x + x^2 + x^3 + x^4

-- Define properties of even and odd functions
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Theorem statements
theorem f_is_even : is_even f := by sorry

theorem g_is_odd : is_odd g := by sorry

theorem h_is_neither : ¬(is_even h) ∧ ¬(is_odd h) := by sorry

end f_is_even_g_is_odd_h_is_neither_l1886_188669


namespace negative_x_over_two_abs_x_positive_l1886_188618

theorem negative_x_over_two_abs_x_positive (x : ℝ) (h : x < 0) :
  -x / (2 * |x|) > 0 := by
  sorry

end negative_x_over_two_abs_x_positive_l1886_188618


namespace cube_multiplication_division_equality_l1886_188679

theorem cube_multiplication_division_equality : (12 ^ 3 * 6 ^ 3) / 432 = 864 := by
  sorry

end cube_multiplication_division_equality_l1886_188679


namespace red_balls_count_l1886_188614

/-- The number of times 18 balls are taken out after the initial 60 balls -/
def x : ℕ := sorry

/-- The total number of balls in the bag -/
def total_balls : ℕ := 60 + 18 * x

/-- The total number of red balls in the bag -/
def red_balls : ℕ := 56 + 14 * x

/-- The proportion of red balls to total balls is 4/5 -/
axiom proportion_axiom : (red_balls : ℚ) / total_balls = 4 / 5

theorem red_balls_count : red_balls = 336 := by sorry

end red_balls_count_l1886_188614


namespace smallest_m_congruence_l1886_188693

theorem smallest_m_congruence : ∃ m : ℕ+, 
  (∀ k : ℕ+, k < m → ¬(790 * k.val ≡ 1430 * k.val [ZMOD 30])) ∧ 
  (790 * m.val ≡ 1430 * m.val [ZMOD 30]) :=
by sorry

end smallest_m_congruence_l1886_188693


namespace subset_implies_m_equals_one_l1886_188687

theorem subset_implies_m_equals_one (m : ℝ) :
  let A : Set ℝ := {-1, 2, 2*m - 1}
  let B : Set ℝ := {2, m^2}
  B ⊆ A → m = 1 := by
sorry

end subset_implies_m_equals_one_l1886_188687


namespace video_cassette_cost_l1886_188616

theorem video_cassette_cost (audio_cost video_cost : ℕ) : 
  (7 * audio_cost + 3 * video_cost = 1110) →
  (5 * audio_cost + 4 * video_cost = 1350) →
  video_cost = 300 := by
sorry

end video_cassette_cost_l1886_188616


namespace track_circumference_l1886_188697

/-- Represents the circular track and the runners' movement --/
structure TrackSystem where
  circumference : ℝ
  speed_a : ℝ
  speed_b : ℝ

/-- The conditions of the problem --/
def satisfies_conditions (s : TrackSystem) : Prop :=
  s.speed_a > 0 ∧ s.speed_b > 0 ∧
  s.speed_a ≠ s.speed_b ∧
  (s.circumference / 2) / s.speed_b = 150 / s.speed_a ∧
  (s.circumference - 90) / s.speed_a = (s.circumference / 2 + 90) / s.speed_b

/-- The theorem to be proved --/
theorem track_circumference (s : TrackSystem) :
  satisfies_conditions s → s.circumference = 540 := by
  sorry

end track_circumference_l1886_188697


namespace canadian_scientist_ratio_l1886_188651

/-- Proves that the ratio of Canadian scientists to total scientists is 1:5 -/
theorem canadian_scientist_ratio (total : ℕ) (usa : ℕ) : 
  total = 70 → 
  usa = 21 → 
  (total - (total / 2) - usa) / total = 1 / 5 := by
sorry

end canadian_scientist_ratio_l1886_188651


namespace exists_z_satisfying_equation_l1886_188640

-- Define the function f
def f (x : ℝ) : ℝ := 2 * (3 * x)^3 + 3 * x + 5

-- State the theorem
theorem exists_z_satisfying_equation :
  ∃ z : ℝ, f (3 * z) = 3 ∧ z = -2 / 729 := by
  sorry

end exists_z_satisfying_equation_l1886_188640


namespace train_length_l1886_188625

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 40 → time = 27 → ∃ length : ℝ, abs (length - 299.97) < 0.01 := by
  sorry

#check train_length

end train_length_l1886_188625


namespace triangle_area_triangle_area_proof_l1886_188675

/-- The area of a triangle with sides 13, 14, and 15 is 84 -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    let a : ℝ := 13
    let b : ℝ := 14
    let c : ℝ := 15
    let s : ℝ := (a + b + c) / 2
    area = Real.sqrt (s * (s - a) * (s - b) * (s - c)) ∧ area = 84

/-- Proof of the triangle area theorem -/
theorem triangle_area_proof : ∃ area : ℝ, triangle_area area := by
  sorry

end triangle_area_triangle_area_proof_l1886_188675


namespace tiffany_albums_l1886_188617

theorem tiffany_albums (phone_pics camera_pics pics_per_album : ℕ) 
  (h1 : phone_pics = 7)
  (h2 : camera_pics = 13)
  (h3 : pics_per_album = 4) :
  (phone_pics + camera_pics) / pics_per_album = 5 := by
  sorry

end tiffany_albums_l1886_188617


namespace find_M_l1886_188629

theorem find_M : ∃ M : ℕ, (992 + 994 + 996 + 998 + 1000 = 5000 - M) ∧ (M = 20) := by
  sorry

end find_M_l1886_188629


namespace equation1_solution_equation2_no_solution_l1886_188637

-- Define the equations
def equation1 (x : ℝ) : Prop := (3 / (x^2 - 9)) + (x / (x - 3)) = 1
def equation2 (x : ℝ) : Prop := 2 - (1 / (2 - x)) = (3 - x) / (x - 2)

-- Theorem for equation 1
theorem equation1_solution : 
  ∃! x : ℝ, equation1 x ∧ x ≠ 3 ∧ x ≠ -3 := by sorry

-- Theorem for equation 2
theorem equation2_no_solution : 
  ∀ x : ℝ, ¬(equation2 x ∧ x ≠ 2) := by sorry

end equation1_solution_equation2_no_solution_l1886_188637


namespace add_zero_eq_self_l1886_188650

theorem add_zero_eq_self (x : ℝ) : x + 0 = x := by
  sorry

end add_zero_eq_self_l1886_188650


namespace union_of_sets_l1886_188692

-- Define the sets A and B
def A (a : ℕ) : Set ℕ := {3, 2^a}
def B (a b : ℕ) : Set ℕ := {a, b}

-- Theorem statement
theorem union_of_sets (a b : ℕ) :
  (A a ∩ B a b = {2}) → (A a ∪ B a b = {1, 2, 3}) := by
  sorry

end union_of_sets_l1886_188692


namespace candy_jar_problem_l1886_188655

/-- Represents the number of candies in a jar -/
structure JarContents where
  red : ℕ
  yellow : ℕ

/-- The problem statement -/
theorem candy_jar_problem :
  ∀ (jar1 jar2 : JarContents),
    -- Both jars have the same total number of candies
    (jar1.red + jar1.yellow = jar2.red + jar2.yellow) →
    -- Jar 1 has a red to yellow ratio of 7:3
    (7 * jar1.yellow = 3 * jar1.red) →
    -- Jar 2 has a red to yellow ratio of 5:4
    (5 * jar2.yellow = 4 * jar2.red) →
    -- The total number of yellow candies is 108
    (jar1.yellow + jar2.yellow = 108) →
    -- The difference in red candies between Jar 1 and Jar 2 is 21
    (jar1.red - jar2.red = 21) :=
by sorry

end candy_jar_problem_l1886_188655


namespace problem_1_l1886_188641

theorem problem_1 : (1.5 - 0.6) * (3 - 1.8) = 1.08 := by
  sorry

end problem_1_l1886_188641


namespace quadratic_root_implies_m_l1886_188660

theorem quadratic_root_implies_m (x m : ℝ) : 
  x = -1 → x^2 + m*x = 3 → m = -2 := by
  sorry

end quadratic_root_implies_m_l1886_188660


namespace picture_tube_consignment_l1886_188608

theorem picture_tube_consignment (defective : ℕ) (prob : ℚ) (total : ℕ) : 
  defective = 5 →
  prob = 5263157894736842 / 100000000000000000 →
  (defective : ℚ) / total * (defective - 1 : ℚ) / (total - 1) = prob →
  total = 20 := by
  sorry

end picture_tube_consignment_l1886_188608


namespace equation_solutions_l1886_188612

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x^2 + 2*x)^(1/3) + (3*x^2 + 6*x - 4)^(1/3) = (x^2 + 2*x - 4)^(1/3)

-- Theorem statement
theorem equation_solutions :
  {x : ℝ | equation x} = {-2, 0} :=
sorry

end equation_solutions_l1886_188612


namespace round_75_36_bar_l1886_188670

/-- Represents a number with a repeating decimal part -/
structure RepeatingDecimal where
  wholePart : ℕ
  nonRepeatingPart : ℕ
  repeatingPart : ℕ

/-- Rounds a RepeatingDecimal to the nearest hundredth -/
def roundToHundredth (x : RepeatingDecimal) : ℚ :=
  sorry

/-- The number 75.363636... -/
def number : RepeatingDecimal :=
  { wholePart := 75,
    nonRepeatingPart := 36,
    repeatingPart := 36 }

theorem round_75_36_bar : roundToHundredth number = 75.37 := by
  sorry

end round_75_36_bar_l1886_188670


namespace hyperbola_C_equation_l1886_188674

/-- A hyperbola passing through a point and sharing asymptotes with another hyperbola -/
def hyperbola_C (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), (a > 0 ∧ b > 0) ∧ 
  (x^2 / a^2 - y^2 / b^2 = 1) ∧
  (3^2 / a^2 - 2 / b^2 = 1) ∧
  (a^2 / b^2 = 3)

/-- Theorem stating the standard equation of hyperbola C -/
theorem hyperbola_C_equation :
  ∀ x y : ℝ, hyperbola_C x y → (x^2 / 3 - y^2 = 1) :=
by sorry

end hyperbola_C_equation_l1886_188674


namespace marble_244_is_white_l1886_188619

/-- Represents the color of a marble -/
inductive MarbleColor
  | White
  | Gray
  | Black

/-- Returns the color of the nth marble in the sequence -/
def marbleColor (n : ℕ) : MarbleColor :=
  let cyclePosition := n % 12
  if cyclePosition ≤ 4 then MarbleColor.White
  else if cyclePosition ≤ 9 then MarbleColor.Gray
  else MarbleColor.Black

/-- Theorem: The 244th marble in the sequence is white -/
theorem marble_244_is_white : marbleColor 244 = MarbleColor.White := by
  sorry


end marble_244_is_white_l1886_188619


namespace decagon_diagonals_l1886_188648

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end decagon_diagonals_l1886_188648


namespace jamie_oyster_collection_l1886_188666

/-- The proportion of oysters that have pearls -/
def pearl_ratio : ℚ := 1/4

/-- The number of dives Jamie makes -/
def num_dives : ℕ := 14

/-- The total number of pearls Jamie collects -/
def total_pearls : ℕ := 56

/-- The number of oysters Jamie can collect during each dive -/
def oysters_per_dive : ℕ := 16

theorem jamie_oyster_collection :
  oysters_per_dive = (total_pearls / num_dives) / pearl_ratio := by
  sorry

end jamie_oyster_collection_l1886_188666


namespace power_station_output_scientific_notation_l1886_188696

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem power_station_output_scientific_notation :
  toScientificNotation 448000 = ScientificNotation.mk 4.48 5 (by sorry) :=
sorry

end power_station_output_scientific_notation_l1886_188696


namespace angle_properties_l1886_188656

-- Define the angle θ
variable (θ : Real)

-- Define the condition that the terminal side of θ passes through (4, -3)
def terminal_side_condition : Prop := ∃ (k : Real), k > 0 ∧ k * Real.cos θ = 4 ∧ k * Real.sin θ = -3

-- Theorem statement
theorem angle_properties (h : terminal_side_condition θ) : 
  Real.tan θ = -3/4 ∧ 
  (Real.sin (θ + Real.pi/2) + Real.cos θ) / (Real.sin θ - Real.cos (θ - Real.pi)) = 8 := by
  sorry

end angle_properties_l1886_188656


namespace stock_price_increase_l1886_188638

theorem stock_price_increase (opening_price closing_price : ℝ) 
  (percent_increase : ℝ) : 
  opening_price = 6 → 
  percent_increase = 33.33 → 
  closing_price = opening_price * (1 + percent_increase / 100) → 
  closing_price = 8 := by
sorry

end stock_price_increase_l1886_188638


namespace grace_and_henry_weight_l1886_188604

/-- Given the weights of pairs of people, prove that Grace and Henry weigh 250 pounds together. -/
theorem grace_and_henry_weight
  (e f g h : ℝ)  -- Weights of Ella, Finn, Grace, and Henry
  (h1 : e + f = 280)  -- Ella and Finn weigh 280 pounds together
  (h2 : f + g = 230)  -- Finn and Grace weigh 230 pounds together
  (h3 : e + h = 300)  -- Ella and Henry weigh 300 pounds together
  : g + h = 250 := by
  sorry

end grace_and_henry_weight_l1886_188604


namespace randy_initial_money_l1886_188688

/-- Calculates the initial amount of money Randy had in his piggy bank. -/
def initial_money (cost_per_trip : ℕ) (trips_per_month : ℕ) (months : ℕ) (money_left : ℕ) : ℕ :=
  cost_per_trip * trips_per_month * months + money_left

/-- Proves that Randy started with $200 given the problem conditions. -/
theorem randy_initial_money :
  initial_money 2 4 12 104 = 200 := by
  sorry

end randy_initial_money_l1886_188688


namespace soccer_team_composition_l1886_188657

theorem soccer_team_composition (total_players goalies defenders : ℕ) 
  (h1 : total_players = 40)
  (h2 : goalies = 3)
  (h3 : defenders = 10)
  : total_players - (goalies + defenders + 2 * defenders) = 7 := by
  sorry

end soccer_team_composition_l1886_188657


namespace investor_purchase_price_l1886_188611

/-- The dividend rate paid by the company -/
def dividend_rate : ℚ := 185 / 1000

/-- The face value of each share -/
def face_value : ℚ := 50

/-- The return on investment received by the investor -/
def roi : ℚ := 1 / 4

/-- The purchase price per share -/
def purchase_price : ℚ := 37

theorem investor_purchase_price : 
  dividend_rate * face_value / purchase_price = roi := by sorry

end investor_purchase_price_l1886_188611


namespace n_squared_plus_n_plus_one_properties_l1886_188665

theorem n_squared_plus_n_plus_one_properties (n : ℕ) :
  (Odd (n^2 + n + 1)) ∧ (¬ ∃ m : ℕ, n^2 + n + 1 = m^2) := by
  sorry

end n_squared_plus_n_plus_one_properties_l1886_188665


namespace product_of_cosines_l1886_188628

theorem product_of_cosines (π : Real) : 
  (1 + Real.cos (π / 9)) * (1 + Real.cos (2 * π / 9)) * 
  (1 + Real.cos (4 * π / 9)) * (1 + Real.cos (5 * π / 9)) = 
  (1 / 2) * (Real.sin (π / 9))^4 := by
  sorry

end product_of_cosines_l1886_188628


namespace xy_greater_than_xz_l1886_188636

theorem xy_greater_than_xz (x y z : ℝ) 
  (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z := by
  sorry

end xy_greater_than_xz_l1886_188636


namespace opposite_of_negative_2023_l1886_188689

theorem opposite_of_negative_2023 : -((-2023 : ℤ)) = 2023 := by
  sorry

end opposite_of_negative_2023_l1886_188689


namespace isosceles_triangle_rectangle_equal_area_l1886_188615

/-- Given an isosceles triangle with base 2s and height h, and a rectangle with side length s,
    if their areas are equal, then the height of the triangle equals the side length of the rectangle. -/
theorem isosceles_triangle_rectangle_equal_area
  (s h : ℝ) -- s: side length of rectangle, h: height of triangle
  (h_positive : s > 0) -- Ensure s is positive
  (area_equal : s * h = s^2) -- Areas are equal
  : h = s := by
  sorry

end isosceles_triangle_rectangle_equal_area_l1886_188615


namespace cheryl_material_usage_l1886_188600

theorem cheryl_material_usage
  (material1 : ℚ) (material2 : ℚ) (leftover : ℚ)
  (h1 : material1 = 4 / 9)
  (h2 : material2 = 2 / 3)
  (h3 : leftover = 8 / 18) :
  material1 + material2 - leftover = 2 / 3 :=
by sorry

end cheryl_material_usage_l1886_188600


namespace round_table_seats_l1886_188694

/-- Represents a round table with equally spaced seats numbered clockwise -/
structure RoundTable where
  num_seats : ℕ

/-- Represents a seat at the round table -/
structure Seat where
  number : ℕ

/-- Two seats are opposite if they are half the table size apart -/
def are_opposite (t : RoundTable) (s1 s2 : Seat) : Prop :=
  (s2.number - s1.number) % t.num_seats = t.num_seats / 2

theorem round_table_seats (t : RoundTable) (s1 s2 : Seat) :
  s1.number = 10 →
  s2.number = 29 →
  are_opposite t s1 s2 →
  t.num_seats = 38 := by
  sorry

end round_table_seats_l1886_188694


namespace rectangle_area_at_stage_4_l1886_188690

/-- Represents the stage number of the rectangle formation process -/
def Stage : ℕ := 4

/-- The side length of each square added at each stage -/
def SquareSideLength : ℝ := 5

/-- The area of the rectangle at a given stage -/
def RectangleArea (stage : ℕ) : ℝ :=
  (stage : ℝ) * SquareSideLength * SquareSideLength

/-- Theorem stating that the area of the rectangle at Stage 4 is 100 square inches -/
theorem rectangle_area_at_stage_4 : RectangleArea Stage = 100 := by
  sorry

end rectangle_area_at_stage_4_l1886_188690


namespace distance_between_blue_lights_l1886_188686

/-- Represents the sequence of lights -/
inductive Light
| Blue
| Yellow

/-- The pattern of lights -/
def light_pattern : List Light := [Light.Blue, Light.Blue, Light.Yellow, Light.Yellow, Light.Yellow]

/-- The distance between each light in inches -/
def light_distance : ℕ := 8

/-- Calculates the position of the nth blue light -/
def blue_light_position (n : ℕ) : ℕ :=
  sorry

/-- Calculates the distance between two positions in feet -/
def distance_in_feet (pos1 pos2 : ℕ) : ℚ :=
  sorry

theorem distance_between_blue_lights :
  distance_in_feet (blue_light_position 4) (blue_light_position 26) = 100/3 :=
sorry

end distance_between_blue_lights_l1886_188686


namespace negation_of_nonnegative_squares_l1886_188678

theorem negation_of_nonnegative_squares :
  (¬ (∀ x : ℝ, x^2 ≥ 0)) ↔ (∃ x : ℝ, x^2 < 0) := by sorry

end negation_of_nonnegative_squares_l1886_188678


namespace percentage_fraction_difference_l1886_188699

theorem percentage_fraction_difference : 
  (65 / 100 * 40) - (4 / 5 * 25) = 6 := by sorry

end percentage_fraction_difference_l1886_188699


namespace simplify_trig_expression_l1886_188683

theorem simplify_trig_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = 2 * Real.cos θ := by
  sorry

end simplify_trig_expression_l1886_188683


namespace correct_mark_is_63_l1886_188643

/-- Proves that the correct mark is 63 given the conditions of the problem -/
theorem correct_mark_is_63 (n : ℕ) (wrong_mark : ℕ) (avg_increase : ℚ) : 
  n = 40 → 
  wrong_mark = 83 → 
  avg_increase = 1/2 → 
  (wrong_mark - (n * avg_increase : ℚ).floor : ℤ) = 63 := by
  sorry

end correct_mark_is_63_l1886_188643


namespace cubic_inequality_l1886_188677

theorem cubic_inequality (a b : ℝ) 
  (h1 : a^3 - b^3 = 2) 
  (h2 : a^5 - b^5 ≥ 4) : 
  a > b ∧ a^2 + b^2 ≥ 2 := by
sorry

end cubic_inequality_l1886_188677


namespace contrapositive_quadratic_roots_l1886_188681

theorem contrapositive_quadratic_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, a * x^2 - b * x + c = 0 → x > 0) → a * c > 0
  ↔
  a * c ≤ 0 → ∃ x : ℝ, a * x^2 - b * x + c = 0 ∧ x ≤ 0 :=
by sorry

end contrapositive_quadratic_roots_l1886_188681
