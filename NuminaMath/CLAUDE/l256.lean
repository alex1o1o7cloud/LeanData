import Mathlib

namespace complex_number_solution_l256_25623

theorem complex_number_solution (z : ℂ) (h : z + Complex.abs z = 2 + 8 * I) : z = -15 + 8 * I := by
  sorry

end complex_number_solution_l256_25623


namespace max_value_implies_A_l256_25615

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sin x, 1)

noncomputable def n (A x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * A * Real.cos x, A / 2 * Real.cos (2 * x))

noncomputable def f (A x : ℝ) : ℝ := (m x).1 * (n A x).1 + (m x).2 * (n A x).2

theorem max_value_implies_A (A : ℝ) (h1 : A > 0) (h2 : ∀ x, f A x ≤ 6) (h3 : ∃ x, f A x = 6) : A = 6 := by
  sorry

end max_value_implies_A_l256_25615


namespace least_integer_y_l256_25676

theorem least_integer_y : ∃ y : ℤ, (∀ z : ℤ, |3*z - 4| ≤ 25 → y ≤ z) ∧ |3*y - 4| ≤ 25 :=
by sorry

end least_integer_y_l256_25676


namespace line_circle_intersection_l256_25634

/-- The line x - y + 1 = 0 intersects the circle (x - a)² + y² = 2 
    if and only if a is in the closed interval [-3, 1] -/
theorem line_circle_intersection (a : ℝ) : 
  (∃ x y : ℝ, x - y + 1 = 0 ∧ (x - a)^2 + y^2 = 2) ↔ a ∈ Set.Icc (-3) 1 := by
sorry

end line_circle_intersection_l256_25634


namespace prob_at_least_one_female_l256_25641

/-- The probability of selecting at least one female student when choosing two students from a group of three male and two female students is 7/10. -/
theorem prob_at_least_one_female (total : ℕ) (male : ℕ) (female : ℕ) (select : ℕ) :
  total = male + female →
  total = 5 →
  male = 3 →
  female = 2 →
  select = 2 →
  (1 : ℚ) - (Nat.choose male select : ℚ) / (Nat.choose total select : ℚ) = 7 / 10 := by
  sorry

end prob_at_least_one_female_l256_25641


namespace number_ratio_l256_25616

theorem number_ratio (first second third : ℚ) : 
  first + second + third = 220 →
  second = 60 →
  third = (1 / 3) * first →
  first / second = 2 := by
sorry

end number_ratio_l256_25616


namespace least_n_for_product_exceeding_million_l256_25653

theorem least_n_for_product_exceeding_million (n : ℕ) : 
  (∀ k < 23, (2 : ℝ) ^ ((k * (k + 1)) / 26) ≤ 1000000) ∧
  (2 : ℝ) ^ ((23 * 24) / 26) > 1000000 := by
  sorry

end least_n_for_product_exceeding_million_l256_25653


namespace average_pastry_sales_l256_25650

def pastry_sales : List Nat := [2, 3, 4, 5, 6, 7, 8]

theorem average_pastry_sales : 
  (List.sum pastry_sales) / pastry_sales.length = 5 := by
  sorry

end average_pastry_sales_l256_25650


namespace right_triangle_case1_right_triangle_case2_right_triangle_case3_l256_25692

-- Define a right triangle
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  right_angle : angleC = 90
  angle_sum : angleA + angleB + angleC = 180

-- Case 1
theorem right_triangle_case1 (t : RightTriangle) (h1 : t.angleB = 60) (h2 : t.a = 4) :
  t.b = 4 * Real.sqrt 3 ∧ t.c = 8 := by
  sorry

-- Case 2
theorem right_triangle_case2 (t : RightTriangle) (h1 : t.a = Real.sqrt 3 - 1) (h2 : t.b = 3 - Real.sqrt 3) :
  t.angleB = 60 ∧ t.angleA = 30 ∧ t.c = 2 * Real.sqrt 3 - 2 := by
  sorry

-- Case 3
theorem right_triangle_case3 (t : RightTriangle) (h1 : t.angleA = 60) (h2 : t.c = 2 + Real.sqrt 3) :
  t.angleB = 30 ∧ t.a = Real.sqrt 3 + 3/2 ∧ t.b = (2 + Real.sqrt 3)/2 := by
  sorry

end right_triangle_case1_right_triangle_case2_right_triangle_case3_l256_25692


namespace largest_number_l256_25671

theorem largest_number (π : ℝ) (h : 3 < π ∧ π < 4) : 
  π = max π (max 3 (max (1 - π) (-π^2))) := by
sorry

end largest_number_l256_25671


namespace winning_candidate_vote_percentage_l256_25631

/-- The percentage of votes received by the winning candidate in an election with three candidates -/
theorem winning_candidate_vote_percentage 
  (votes : Fin 3 → ℕ)
  (h1 : votes 0 = 3000)
  (h2 : votes 1 = 5000)
  (h3 : votes 2 = 15000) :
  (votes 2 : ℚ) / (votes 0 + votes 1 + votes 2) * 100 = 15000 / 23000 * 100 := by
  sorry

#eval (15000 : ℚ) / 23000 * 100 -- To display the approximate result

end winning_candidate_vote_percentage_l256_25631


namespace triangle_rectangle_area_coefficient_l256_25610

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define the rectangle PQRS
structure Rectangle :=
  (ω : ℝ)
  (α β : ℝ)

-- Define the area function for the rectangle
def rectangleArea (rect : Rectangle) : ℝ :=
  rect.α * rect.ω - rect.β * rect.ω^2

-- State the theorem
theorem triangle_rectangle_area_coefficient
  (triangle : Triangle)
  (rect : Rectangle)
  (h1 : triangle.a = 13)
  (h2 : triangle.b = 26)
  (h3 : triangle.c = 15)
  (h4 : rectangleArea rect = 0 → rect.ω = 26)
  (h5 : rectangleArea rect = (triangle.a * triangle.b) / 4 → rect.ω = 13) :
  rect.β = 105 / 338 := by
sorry

end triangle_rectangle_area_coefficient_l256_25610


namespace power_of_three_plus_five_mod_seven_l256_25683

theorem power_of_three_plus_five_mod_seven : (3^90 + 5) % 7 = 6 := by
  sorry

end power_of_three_plus_five_mod_seven_l256_25683


namespace total_pictures_sum_l256_25656

/-- Represents the number of pictures Zoe has taken -/
structure PictureCount where
  initial : ℕ
  dolphinShow : ℕ
  total : ℕ

/-- Theorem: The total number of pictures is the sum of initial and dolphin show pictures -/
theorem total_pictures_sum (z : PictureCount) 
  (h1 : z.initial = 28)
  (h2 : z.dolphinShow = 16)
  (h3 : z.total = 44) :
  z.total = z.initial + z.dolphinShow := by
  sorry

#check total_pictures_sum

end total_pictures_sum_l256_25656


namespace count_equal_S_is_11_l256_25607

/-- S(n) is the smallest positive integer divisible by each of the positive integers 1, 2, 3, ..., n -/
def S (n : ℕ) : ℕ := sorry

/-- The count of positive integers n with 1 ≤ n ≤ 100 that have S(n) = S(n+4) -/
def count_equal_S : ℕ := sorry

theorem count_equal_S_is_11 : count_equal_S = 11 := by sorry

end count_equal_S_is_11_l256_25607


namespace lottery_blank_probability_l256_25649

theorem lottery_blank_probability :
  let num_prizes : ℕ := 10
  let num_blanks : ℕ := 25
  let total_outcomes : ℕ := num_prizes + num_blanks
  (num_blanks : ℚ) / (total_outcomes : ℚ) = 5 / 7 :=
by sorry

end lottery_blank_probability_l256_25649


namespace jennifer_future_age_l256_25604

def jennifer_age_in_10_years : ℕ := 30

def jordana_current_age : ℕ := 80

theorem jennifer_future_age :
  jennifer_age_in_10_years = 30 :=
by
  have h1 : jordana_current_age + 10 = 3 * jennifer_age_in_10_years :=
    sorry
  sorry

#check jennifer_future_age

end jennifer_future_age_l256_25604


namespace equation_solution_l256_25628

theorem equation_solution :
  ∃! x : ℝ, (3 : ℝ)^x * (9 : ℝ)^x = (27 : ℝ)^(x - 4) :=
by
  use -6
  sorry

end equation_solution_l256_25628


namespace fourth_selected_is_34_l256_25673

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  total_students : Nat
  num_groups : Nat
  first_selected : Nat
  second_selected : Nat

/-- Calculates the number of the selected student for a given group -/
def selected_student (s : SystematicSampling) (group : Nat) : Nat :=
  s.first_selected + (s.total_students / s.num_groups) * group

/-- Theorem stating that the fourth selected student will be number 34 -/
theorem fourth_selected_is_34 (s : SystematicSampling) 
  (h1 : s.total_students = 50)
  (h2 : s.num_groups = 5)
  (h3 : s.first_selected = 4)
  (h4 : s.second_selected = 14) :
  selected_student s 3 = 34 := by
  sorry

end fourth_selected_is_34_l256_25673


namespace equation_represents_parabola_l256_25655

/-- The equation |y - 3| = √((x+4)² + (y-1)²) represents a parabola -/
theorem equation_represents_parabola :
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ x y : ℝ, |y - 3| = Real.sqrt ((x + 4)^2 + (y - 1)^2) →
    y = a * x^2 + b * x + c) :=
by sorry

end equation_represents_parabola_l256_25655


namespace john_started_five_days_ago_l256_25627

/-- Represents the number of days John has worked -/
def days_worked : ℕ := sorry

/-- Represents the daily wage John earns -/
def daily_wage : ℚ := sorry

/-- The total amount John has earned so far -/
def current_earnings : ℚ := 250

/-- The number of additional days John needs to work -/
def additional_days : ℕ := 10

theorem john_started_five_days_ago :
  days_worked = 5 ∧
  daily_wage * days_worked = current_earnings ∧
  daily_wage * (days_worked + additional_days) = 2 * current_earnings :=
sorry

end john_started_five_days_ago_l256_25627


namespace nine_sided_polygon_diagonals_l256_25697

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  sides : ℕ
  is_irregular : Bool
  is_convex : Bool
  right_angles : ℕ

/-- The number of diagonals in a polygon -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem stating that a nine-sided polygon has 27 diagonals -/
theorem nine_sided_polygon_diagonals (P : Polygon 9) 
  (h1 : P.is_irregular = true) 
  (h2 : P.is_convex = true) 
  (h3 : P.right_angles = 2) : 
  num_diagonals 9 = 27 := by
  sorry

end nine_sided_polygon_diagonals_l256_25697


namespace distance_to_focus_is_13_l256_25602

/-- Parabola with equation y^2 = 16x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ
  axis_of_symmetry : ℝ → ℝ

/-- Point on the parabola -/
structure PointOnParabola (p : Parabola) where
  point : ℝ × ℝ
  on_parabola : p.equation point.1 point.2

theorem distance_to_focus_is_13 (p : Parabola) (P : PointOnParabola p) 
  (h_equation : p.equation = fun x y => y^2 = 16*x)
  (h_distance : abs P.point.2 = 12) :
  dist P.point p.focus = 13 := by
  sorry

end distance_to_focus_is_13_l256_25602


namespace sqrt_product_equals_140_l256_25608

theorem sqrt_product_equals_140 :
  Real.sqrt (13 + Real.sqrt (28 + Real.sqrt 281)) *
  Real.sqrt (13 - Real.sqrt (28 + Real.sqrt 281)) *
  Real.sqrt (141 + Real.sqrt 281) = 140 := by
  sorry

end sqrt_product_equals_140_l256_25608


namespace monomial_satisfies_conditions_l256_25611

-- Define a structure for monomials
structure Monomial (α : Type) [CommRing α] where
  coeff : α
  vars : List (Nat × Nat)

-- Define the monomial -2mn^2
def target_monomial : Monomial ℤ := ⟨-2, [(1, 1), (2, 2)]⟩

-- Define functions to check the conditions
def has_variables (m : Monomial ℤ) (vars : List Nat) : Prop :=
  ∀ v ∈ vars, ∃ p ∈ m.vars, v = p.1

def coefficient (m : Monomial ℤ) : ℤ := m.coeff

def degree (m : Monomial ℤ) : Nat :=
  m.vars.foldr (fun p acc => acc + p.2) 0

-- Theorem statement
theorem monomial_satisfies_conditions :
  has_variables target_monomial [1, 2] ∧
  coefficient target_monomial = -2 ∧
  degree target_monomial = 3 := by
  sorry

end monomial_satisfies_conditions_l256_25611


namespace jaguar_snake_consumption_l256_25643

theorem jaguar_snake_consumption 
  (beetles_per_bird : ℕ) 
  (birds_per_snake : ℕ) 
  (total_jaguars : ℕ) 
  (total_beetles_eaten : ℕ) 
  (h1 : beetles_per_bird = 12)
  (h2 : birds_per_snake = 3)
  (h3 : total_jaguars = 6)
  (h4 : total_beetles_eaten = 1080) :
  total_beetles_eaten / total_jaguars / beetles_per_bird / birds_per_snake = 5 := by
  sorry

end jaguar_snake_consumption_l256_25643


namespace car_expenses_sum_l256_25647

theorem car_expenses_sum : 
  let speakers_cost : ℚ := 118.54
  let tires_cost : ℚ := 106.33
  let tints_cost : ℚ := 85.27
  let maintenance_cost : ℚ := 199.75
  let cover_cost : ℚ := 15.63
  speakers_cost + tires_cost + tints_cost + maintenance_cost + cover_cost = 525.52 := by
  sorry

end car_expenses_sum_l256_25647


namespace fb_is_80_l256_25668

/-- A right-angled triangle ABC with a point F on BC -/
structure TriangleABCF where
  /-- The length of side AB -/
  ab : ℝ
  /-- The length of side AC -/
  ac : ℝ
  /-- The length of side BC -/
  bc : ℝ
  /-- The length of BF -/
  bf : ℝ
  /-- The length of CF -/
  cf : ℝ
  /-- AB is 120 meters -/
  hab : ab = 120
  /-- AC is 160 meters -/
  hac : ac = 160
  /-- ABC is a right-angled triangle -/
  hright : ab^2 + ac^2 = bc^2
  /-- F is on BC -/
  hf_on_bc : bf + cf = bc
  /-- Jack and Jill jog the same distance -/
  heq_dist : ac + cf = ab + bf

/-- The main theorem: FB is 80 meters -/
theorem fb_is_80 (t : TriangleABCF) : t.bf = 80 := by
  sorry

end fb_is_80_l256_25668


namespace jennas_driving_speed_l256_25690

/-- Proves that Jenna's driving speed is 50 miles per hour given the road trip conditions -/
theorem jennas_driving_speed 
  (total_distance : ℝ) 
  (jenna_distance : ℝ) 
  (friend_distance : ℝ)
  (total_time : ℝ) 
  (break_time : ℝ) 
  (friend_speed : ℝ) 
  (h1 : total_distance = jenna_distance + friend_distance)
  (h2 : total_distance = 300)
  (h3 : jenna_distance = 200)
  (h4 : friend_distance = 100)
  (h5 : total_time = 10)
  (h6 : break_time = 1)
  (h7 : friend_speed = 20) : 
  jenna_distance / (total_time - break_time - friend_distance / friend_speed) = 50 := by
  sorry

#check jennas_driving_speed

end jennas_driving_speed_l256_25690


namespace no_valid_n_l256_25636

theorem no_valid_n : ¬∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℕ), n^2 - 21*n + 110 = x^2) ∧ 
  (15 % n = 0) := by
  sorry

end no_valid_n_l256_25636


namespace joanne_part_time_hours_l256_25687

/-- Calculates the number of hours Joanne works at her part-time job each day -/
def part_time_hours_per_day (main_job_hourly_rate : ℚ) (main_job_hours_per_day : ℚ) 
  (part_time_hourly_rate : ℚ) (days_per_week : ℚ) (total_weekly_earnings : ℚ) : ℚ :=
  let main_job_daily_earnings := main_job_hourly_rate * main_job_hours_per_day
  let main_job_weekly_earnings := main_job_daily_earnings * days_per_week
  let part_time_weekly_earnings := total_weekly_earnings - main_job_weekly_earnings
  let part_time_weekly_hours := part_time_weekly_earnings / part_time_hourly_rate
  part_time_weekly_hours / days_per_week

theorem joanne_part_time_hours : 
  part_time_hours_per_day 16 8 (27/2) 5 775 = 2 := by
  sorry

end joanne_part_time_hours_l256_25687


namespace count_valid_numbers_l256_25672

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧  -- nine-digit number
  (∃ (digits : List ℕ), 
    digits.length = 9 ∧
    digits.count 3 = 8 ∧
    digits.count 0 = 1 ∧
    digits.foldl (λ acc d => acc * 10 + d) 0 = n)

def leaves_remainder_one (n : ℕ) : Prop :=
  n % 4 = 1

theorem count_valid_numbers : 
  (∃ (S : Finset ℕ), 
    (∀ n ∈ S, is_valid_number n ∧ leaves_remainder_one n) ∧
    S.card = 7 ∧
    (∀ n, is_valid_number n ∧ leaves_remainder_one n → n ∈ S)) :=
sorry

end count_valid_numbers_l256_25672


namespace quadrilateral_side_sum_l256_25688

/-- Represents a quadrilateral with side lengths a, b, c, d --/
structure Quadrilateral :=
  (a b c d : ℝ)

/-- Predicate to check if angles are in arithmetic progression --/
def angles_in_arithmetic_progression (q : Quadrilateral) : Prop :=
  sorry

/-- Predicate to check if the largest side is opposite the largest angle --/
def largest_side_opposite_largest_angle (q : Quadrilateral) : Prop :=
  sorry

/-- The main theorem --/
theorem quadrilateral_side_sum (q : Quadrilateral) 
  (h1 : angles_in_arithmetic_progression q)
  (h2 : largest_side_opposite_largest_angle q)
  (h3 : q.a = 7)
  (h4 : q.b = 8)
  (h5 : ∃ (a b c : ℕ), q.c = a + Real.sqrt b + Real.sqrt c ∧ a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (a b c : ℕ), q.c = a + Real.sqrt b + Real.sqrt c ∧ a + b + c = 113 :=
sorry

end quadrilateral_side_sum_l256_25688


namespace absolute_value_inequality_l256_25640

-- Define the function f
def f (x m : ℝ) : ℝ := |2 * x - m|

-- State the theorem
theorem absolute_value_inequality (m : ℝ) :
  (∀ x : ℝ, f x m ≤ 6 ↔ -2 ≤ x ∧ x ≤ 4) →
  (m = 2 ∧
   ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 2 →
     (∀ x : ℝ, f x m + f ((1/2) * x + 3) m ≤ 8/a + 2/b ↔ -3 ≤ x ∧ x ≤ 7/3)) :=
by sorry

end absolute_value_inequality_l256_25640


namespace vector_operation_result_l256_25670

/-- Prove that the result of 3 * (-3, 2, 6) + (4, -5, 2) is (-5, 1, 20) -/
theorem vector_operation_result :
  (3 : ℝ) • ((-3 : ℝ), (2 : ℝ), (6 : ℝ)) + ((4 : ℝ), (-5 : ℝ), (2 : ℝ)) = ((-5 : ℝ), (1 : ℝ), (20 : ℝ)) := by
  sorry

end vector_operation_result_l256_25670


namespace abc_product_l256_25669

theorem abc_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) :
  a * b * c = 720 := by
sorry

end abc_product_l256_25669


namespace grocery_shop_sales_l256_25651

theorem grocery_shop_sales (sales1 sales3 sales4 sales5 sales6 : ℕ) 
  (h1 : sales1 = 6335)
  (h3 : sales3 = 7230)
  (h4 : sales4 = 6562)
  (h5 : sales5 = 6855)
  (h6 : sales6 = 5091)
  (h_avg : (sales1 + sales3 + sales4 + sales5 + sales6 + 6927) / 6 = 6500) :
  ∃ sales2 : ℕ, sales2 = 6927 := by
  sorry

end grocery_shop_sales_l256_25651


namespace total_prank_combinations_l256_25693

/-- The number of different combinations of people Tim could involve in the prank --/
def prank_combinations (day1 day2 day3 day4 day5 : ℕ) : ℕ :=
  day1 * day2 * day3 * day4 * day5

/-- Theorem stating the total number of different combinations for Tim's prank --/
theorem total_prank_combinations :
  prank_combinations 1 2 5 4 1 = 40 := by
  sorry

end total_prank_combinations_l256_25693


namespace last_four_average_l256_25613

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 4 = 63.75 := by
sorry

end last_four_average_l256_25613


namespace shooting_probability_l256_25680

theorem shooting_probability (p : ℝ) : 
  (p ≥ 0 ∧ p ≤ 1) →  -- Ensure p is a valid probability
  (1 - (1 - 1/2) * (1 - 2/3) * (1 - p) = 7/8) → 
  p = 1/4 := by
sorry

end shooting_probability_l256_25680


namespace stool_height_is_80_l256_25699

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height : ℝ :=
  let ceiling_height : ℝ := 300
  let light_bulb_below_ceiling : ℝ := 15
  let alice_height : ℝ := 150
  let alice_reach : ℝ := 50
  let decoration_below_bulb : ℝ := 5
  let light_bulb_height : ℝ := ceiling_height - light_bulb_below_ceiling
  let effective_reach_height : ℝ := light_bulb_height - decoration_below_bulb
  effective_reach_height - (alice_height + alice_reach)

theorem stool_height_is_80 :
  stool_height = 80 := by
  sorry

end stool_height_is_80_l256_25699


namespace complement_of_A_in_U_l256_25632

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

theorem complement_of_A_in_U :
  {x ∈ U | x ∉ A} = {2, 4} := by sorry

end complement_of_A_in_U_l256_25632


namespace peter_statement_consistency_l256_25625

/-- Represents the day of the week -/
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

/-- Represents whether a person is telling the truth or lying -/
inductive TruthState
| Truthful | Lying

/-- Represents a statement that can be made -/
inductive Statement
| A | B | C | D | E

/-- Function to determine if a day follows another -/
def follows (d1 d2 : Day) : Prop := sorry

/-- Function to determine if a number is divisible by another -/
def is_divisible_by (n m : Nat) : Prop := sorry

/-- Peter's truth-telling state on a given day -/
def peter_truth_state (d : Day) : TruthState := sorry

/-- The content of each statement -/
def statement_content (s : Statement) (today : Day) : Prop :=
  match s with
  | Statement.A => peter_truth_state (sorry : Day) = TruthState.Lying ∧ 
                   peter_truth_state (sorry : Day) = TruthState.Lying
  | Statement.B => peter_truth_state today = TruthState.Truthful ∧ 
                   peter_truth_state (sorry : Day) = TruthState.Truthful
  | Statement.C => is_divisible_by 2024 11
  | Statement.D => (sorry : Day) = Day.Wednesday
  | Statement.E => follows (sorry : Day) Day.Saturday

/-- The main theorem -/
theorem peter_statement_consistency 
  (today : Day) 
  (statements : Finset Statement) 
  (h1 : statements.card = 4) 
  (h2 : Statement.C ∉ statements) :
  ∀ s ∈ statements, 
    (peter_truth_state today = TruthState.Truthful → statement_content s today) ∧
    (peter_truth_state today = TruthState.Lying → ¬statement_content s today) := by
  sorry


end peter_statement_consistency_l256_25625


namespace inequalities_theorem_l256_25630

theorem inequalities_theorem (a b c d : ℝ) 
  (h1 : a > 0) (h2 : 0 > b) (h3 : b > -a) (h4 : c < d) (h5 : d < 0) : 
  (a * d ≤ b * c) ∧ 
  (a / d + b / c < 0) ∧ 
  (a - c > b - d) ∧ 
  (a * (d - c) > b * (d - c)) := by
  sorry

end inequalities_theorem_l256_25630


namespace triangle_problem_l256_25684

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) →
  (a * Real.sin C) / (1 - Real.cos A) = Real.sqrt 3 * c →
  b + c = 10 →
  (1 / 2) * b * c * Real.sin A = 4 * Real.sqrt 3 →
  (A = π / 3) ∧ (a = 2 * Real.sqrt 13) := by
sorry


end triangle_problem_l256_25684


namespace quadratic_form_l256_25621

theorem quadratic_form (x : ℝ) : ∃ (b c : ℝ), x^2 - 16*x + 64 = (x + b)^2 + c ∧ b + c = -8 := by
  sorry

end quadratic_form_l256_25621


namespace tan_315_degrees_l256_25659

theorem tan_315_degrees : Real.tan (315 * π / 180) = -1 := by
  sorry

end tan_315_degrees_l256_25659


namespace two_Z_six_l256_25624

/-- Definition of the operation Z -/
def Z (a b : ℤ) : ℤ := b + 10 * a - a ^ 2

/-- Theorem stating that 2Z6 = 22 -/
theorem two_Z_six : Z 2 6 = 22 := by
  sorry

end two_Z_six_l256_25624


namespace chessboard_tiling_l256_25664

/-- A type representing a chessboard -/
structure Chessboard :=
  (size : ℕ)

/-- A type representing a tiling piece -/
structure TilingPiece :=
  (coverage : ℕ)

/-- Function to check if a chessboard can be tiled with given pieces -/
def can_tile (board : Chessboard) (piece : TilingPiece) : Prop :=
  (board.size * board.size) % piece.coverage = 0

theorem chessboard_tiling :
  (∃ (piece : TilingPiece), piece.coverage = 4 ∧ can_tile ⟨8⟩ piece) ∧
  (∀ (piece : TilingPiece), piece.coverage = 4 → ¬can_tile ⟨10⟩ piece) :=
sorry

end chessboard_tiling_l256_25664


namespace min_value_quadratic_l256_25686

theorem min_value_quadratic (x y : ℝ) : 2*x^2 + 3*y^2 - 8*x + 12*y + 40 ≥ 20 := by
  sorry

end min_value_quadratic_l256_25686


namespace problem_1_problem_2_l256_25618

-- Problem 1
theorem problem_1 (x y : ℝ) :
  3 * x^2 * (-3 * x * y)^2 - x^2 * (x^2 * y^2 - 2 * x) = 26 * x^4 * y^2 + 2 * x^3 :=
by sorry

-- Problem 2
theorem problem_2 (a b c : ℝ) :
  -2 * (-a^2 * b * c)^2 * (1/2) * a * (b * c)^3 - (-a * b * c)^3 * (-a * b * c)^2 = 0 :=
by sorry

end problem_1_problem_2_l256_25618


namespace mauras_seashells_l256_25677

/-- Represents the number of seashells Maura found during her summer vacation. -/
def total_seashells : ℕ := 75

/-- Represents the number of seashells Maura kept after giving some to her sister. -/
def kept_seashells : ℕ := 57

/-- Represents the number of seashells Maura gave to her sister. -/
def given_seashells : ℕ := 18

/-- Represents the number of days Maura's family stayed at the beach house. -/
def beach_days : ℕ := 21

/-- Proves that the total number of seashells Maura found is equal to the sum of
    the seashells she kept and the seashells she gave away. -/
theorem mauras_seashells : total_seashells = kept_seashells + given_seashells := by
  sorry

end mauras_seashells_l256_25677


namespace joe_hvac_cost_per_vent_l256_25638

/-- The cost per vent of an HVAC system -/
def cost_per_vent (total_cost : ℕ) (num_zones : ℕ) (vents_per_zone : ℕ) : ℚ :=
  total_cost / (num_zones * vents_per_zone)

/-- Theorem: The cost per vent of Joe's HVAC system is $2,000 -/
theorem joe_hvac_cost_per_vent :
  cost_per_vent 20000 2 5 = 2000 := by
  sorry

end joe_hvac_cost_per_vent_l256_25638


namespace acme_savings_threshold_l256_25642

/-- Acme T-Shirt Plus Company's pricing structure -/
def acme_cost (x : ℕ) : ℚ := 75 + 8 * x

/-- Gamma T-shirt Company's pricing structure -/
def gamma_cost (x : ℕ) : ℚ := 12 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme_savings : ℕ := 19

theorem acme_savings_threshold :
  (∀ x : ℕ, x ≥ min_shirts_for_acme_savings → acme_cost x < gamma_cost x) ∧
  (∀ x : ℕ, x < min_shirts_for_acme_savings → acme_cost x ≥ gamma_cost x) :=
sorry

end acme_savings_threshold_l256_25642


namespace geometric_sequence_problem_l256_25691

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Property of geometric sequences: if m + n = p + q, then a_m * a_n = a_p * a_q -/
axiom geometric_property {a : ℕ → ℝ} (h : GeometricSequence a) :
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_problem (a : ℕ → ℝ) (h : GeometricSequence a) 
    (h_sum : a 4 + a 8 = -3) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 := by
  sorry

end geometric_sequence_problem_l256_25691


namespace unique_root_when_b_zero_c_positive_odd_function_when_c_zero_symmetric_about_zero_one_iff_c_one_l256_25666

-- Define the function f
def f (x b c : ℝ) : ℝ := |x| * x + b * x + c

-- Theorem 1: When b=0 and c>0, f(x) = 0 has only one root
theorem unique_root_when_b_zero_c_positive (c : ℝ) (hc : c > 0) :
  ∃! x : ℝ, f x 0 c = 0 :=
sorry

-- Theorem 2: When c=0, y=f(x) is an odd function
theorem odd_function_when_c_zero (b : ℝ) :
  ∀ x : ℝ, f (-x) b 0 = -f x b 0 :=
sorry

-- Theorem 3: The graph of y=f(x) is symmetric about (0,1) iff c=1
theorem symmetric_about_zero_one_iff_c_one (b : ℝ) :
  (∀ x : ℝ, f x b 1 = 2 - f (-x) b 1) ↔ c = 1 :=
sorry

end unique_root_when_b_zero_c_positive_odd_function_when_c_zero_symmetric_about_zero_one_iff_c_one_l256_25666


namespace three_digit_square_sum_l256_25665

theorem three_digit_square_sum (N : ℕ) : 
  (100 ≤ N ∧ N ≤ 999) →
  (∃ (a b c : ℕ), 
    0 ≤ a ∧ a ≤ 9 ∧ a ≠ 0 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    N = 100 * a + 10 * b + c ∧
    N = 11 * (a^2 + b^2 + c^2)) →
  (N = 550 ∨ N = 803) :=
by sorry

end three_digit_square_sum_l256_25665


namespace inscribed_shape_perimeter_lower_bound_l256_25617

/-- A shape inscribed in a circle -/
structure InscribedShape where
  -- The radius of the circumscribed circle
  radius : ℝ
  -- The perimeter of the shape
  perimeter : ℝ
  -- Predicate indicating if the center of the circle is inside or on the boundary of the shape
  center_inside : Prop

/-- Theorem: The perimeter of a shape inscribed in a circle is at least 4 times the radius
    if the center of the circle is inside or on the boundary of the shape -/
theorem inscribed_shape_perimeter_lower_bound
  (shape : InscribedShape)
  (h : shape.center_inside) :
  shape.perimeter ≥ 4 * shape.radius :=
sorry

end inscribed_shape_perimeter_lower_bound_l256_25617


namespace inequality_not_always_true_l256_25606

theorem inequality_not_always_true (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  ¬ (∀ a b c : ℝ, a > b → b > c → a * c > b * c) := by
  sorry

end inequality_not_always_true_l256_25606


namespace z_in_fourth_quadrant_l256_25637

-- Define the complex number z
def z : ℂ := (3 + Complex.I) * (1 - Complex.I)

-- Theorem stating that z is in the fourth quadrant
theorem z_in_fourth_quadrant :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
by
  sorry

end z_in_fourth_quadrant_l256_25637


namespace purple_chip_count_l256_25609

theorem purple_chip_count (blue green purple red : ℕ) (x : ℕ) :
  blue > 0 → green > 0 → purple > 0 → red > 0 →
  5 < x → x < 11 →
  1^blue * 5^green * x^purple * 11^red = 140800 →
  purple = 1 ∧ x = 7 := by
  sorry

end purple_chip_count_l256_25609


namespace base_k_conversion_l256_25695

theorem base_k_conversion (k : ℕ) (h : k > 0) : 
  (1 * k^2 + 3 * k + 2 = 30) → k = 4 := by
  sorry

end base_k_conversion_l256_25695


namespace first_butcher_packages_correct_l256_25600

/-- The number of packages delivered by the first butcher -/
def first_butcher_packages : ℕ := 10

/-- The weight of each package in pounds -/
def package_weight : ℕ := 4

/-- The number of packages delivered by the second butcher -/
def second_butcher_packages : ℕ := 7

/-- The number of packages delivered by the third butcher -/
def third_butcher_packages : ℕ := 8

/-- The total weight of all delivered packages in pounds -/
def total_weight : ℕ := 100

/-- Theorem stating that the number of packages delivered by the first butcher is correct -/
theorem first_butcher_packages_correct :
  package_weight * first_butcher_packages +
  package_weight * second_butcher_packages +
  package_weight * third_butcher_packages = total_weight :=
by sorry

end first_butcher_packages_correct_l256_25600


namespace quiz_score_impossibility_l256_25633

theorem quiz_score_impossibility :
  ∀ (c u i : ℕ),
    c + u + i = 25 →
    4 * c + 2 * u - i ≠ 79 :=
by
  sorry

end quiz_score_impossibility_l256_25633


namespace finite_gcd_lcm_process_terminates_l256_25619

theorem finite_gcd_lcm_process_terminates 
  (n : ℕ) 
  (a : Fin n → ℕ+) : 
  ∃ (k : ℕ), ∀ (j k : Fin n), j < k → (a j).val ∣ (a k).val :=
sorry

end finite_gcd_lcm_process_terminates_l256_25619


namespace chocolate_bar_count_l256_25605

/-- Given a very large box containing small boxes of chocolate bars, 
    calculate the total number of chocolate bars. -/
theorem chocolate_bar_count 
  (num_small_boxes : ℕ) 
  (bars_per_small_box : ℕ) 
  (h1 : num_small_boxes = 150) 
  (h2 : bars_per_small_box = 37) : 
  num_small_boxes * bars_per_small_box = 5550 := by
  sorry

#check chocolate_bar_count

end chocolate_bar_count_l256_25605


namespace emma_account_balance_l256_25689

def remaining_balance (initial_balance : ℕ) (daily_spending : ℕ) (days : ℕ) (bill_denomination : ℕ) : ℕ :=
  let balance_after_spending := initial_balance - daily_spending * days
  let withdrawal_amount := (balance_after_spending / bill_denomination) * bill_denomination
  balance_after_spending - withdrawal_amount

theorem emma_account_balance :
  remaining_balance 100 8 7 5 = 4 := by
  sorry

end emma_account_balance_l256_25689


namespace contrapositive_equivalence_l256_25675

theorem contrapositive_equivalence (x y : ℝ) :
  (¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ↔ (x^2 + y^2 = 0 → x = 0 ∧ y = 0) :=
by sorry

end contrapositive_equivalence_l256_25675


namespace integral_ln_sin_x_l256_25648

theorem integral_ln_sin_x (x : ℝ) : 
  ∫ x in (0)..(π/2), Real.log (Real.sin x) = -(π/2) * Real.log 2 := by sorry

end integral_ln_sin_x_l256_25648


namespace circle_graph_percentage_l256_25639

theorem circle_graph_percentage (sector_degrees : ℝ) (total_degrees : ℝ) 
  (h1 : sector_degrees = 18)
  (h2 : total_degrees = 360) :
  (sector_degrees / total_degrees) * 100 = 5 := by
sorry

end circle_graph_percentage_l256_25639


namespace f_satisfies_equation_l256_25652

/-- A function that satisfies f(xy) = f(x) + f(y) + 1 for all x and y -/
def f (x : ℝ) : ℝ := -1

/-- Theorem stating that f satisfies the given functional equation -/
theorem f_satisfies_equation (x y : ℝ) : f (x * y) = f x + f y + 1 := by
  sorry

end f_satisfies_equation_l256_25652


namespace ratio_of_percentages_l256_25660

theorem ratio_of_percentages (P Q M N : ℝ) 
  (hM : M = 0.4 * Q) 
  (hQ : Q = 0.25 * P) 
  (hN : N = 0.6 * P) : 
  M / N = 1 / 6 := by
  sorry

end ratio_of_percentages_l256_25660


namespace difference_of_squares_factorization_l256_25696

theorem difference_of_squares_factorization (x : ℝ) : 9 - 4 * x^2 = (3 - 2*x) * (3 + 2*x) := by
  sorry

end difference_of_squares_factorization_l256_25696


namespace product_H₁_H₂_is_square_l256_25658

/-- For any positive integer n, H₁ is the set of odd numbers from 1 to 2n-1 -/
def H₁ (n : ℕ+) : Finset ℕ :=
  Finset.range n |>.image (fun i => 2*i + 1)

/-- For any positive integers n and k, H₂ is the set obtained by adding k to each element of H₁ -/
def H₂ (n : ℕ+) (k : ℕ+) : Finset ℕ :=
  H₁ n |>.image (fun x => x + k)

/-- The product of all elements in the union of H₁ and H₂ -/
def product_H₁_H₂ (n : ℕ+) (k : ℕ+) : ℕ :=
  (H₁ n ∪ H₂ n k).prod id

/-- For any positive integer n, when k = 2n + 1, the product of all elements in H₁ ∪ H₂ is a perfect square -/
theorem product_H₁_H₂_is_square (n : ℕ+) :
  ∃ m : ℕ, product_H₁_H₂ n (2*n + 1) = m^2 := by
  sorry

end product_H₁_H₂_is_square_l256_25658


namespace inequality_solution_range_l256_25679

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → (a < 1 ∨ a > 3) :=
by sorry

end inequality_solution_range_l256_25679


namespace square_cut_divisible_by_four_l256_25667

/-- A rectangle on a grid --/
structure GridRectangle where
  length : ℕ
  width : ℕ

/-- A square on a grid --/
structure GridSquare where
  side : ℕ

/-- Function to cut a square into rectangles along grid lines --/
def cutSquareIntoRectangles (square : GridSquare) : List GridRectangle :=
  sorry

/-- Function to calculate the perimeter of a rectangle --/
def rectanglePerimeter (rect : GridRectangle) : ℕ :=
  2 * (rect.length + rect.width)

theorem square_cut_divisible_by_four (square : GridSquare) 
    (h : square.side = 2009) :
    ∃ (rect : GridRectangle), rect ∈ cutSquareIntoRectangles square ∧ 
    (rectanglePerimeter rect) % 4 = 0 := by
  sorry

end square_cut_divisible_by_four_l256_25667


namespace intersection_of_M_and_N_l256_25603

def M : Set ℕ := {0, 1, 2, 3}
def N : Set ℕ := {2, 3}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end intersection_of_M_and_N_l256_25603


namespace log_ratio_squared_l256_25644

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (hx_neq : x ≠ 1) (hy_neq : y ≠ 1)
  (h_log : Real.log x / Real.log 2 = Real.log 16 / Real.log y)
  (h_prod : x * y = 64) : 
  ((Real.log x - Real.log y) / Real.log 2)^2 = 20 := by
sorry

end log_ratio_squared_l256_25644


namespace max_sum_given_sum_of_squares_and_product_l256_25635

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 98 → xy = 36 → x + y ≤ Real.sqrt 170 :=
by sorry

end max_sum_given_sum_of_squares_and_product_l256_25635


namespace prob_queen_first_three_cards_l256_25698

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : Type := Unit

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Queens in a standard deck -/
def num_queens : ℕ := 4

/-- The probability of drawing at least one Queen in the first three cards -/
def prob_at_least_one_queen (d : StandardDeck) : ℚ :=
  1 - (deck_size - num_queens) * (deck_size - num_queens - 1) * (deck_size - num_queens - 2) /
      (deck_size * (deck_size - 1) * (deck_size - 2))

theorem prob_queen_first_three_cards :
  ∀ d : StandardDeck, prob_at_least_one_queen d = 2174 / 10000 :=
by sorry

end prob_queen_first_three_cards_l256_25698


namespace quadratic_factorization_l256_25685

theorem quadratic_factorization (a b c : ℤ) :
  (∀ x, x^2 + 11*x + 28 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 21 := by
  sorry

end quadratic_factorization_l256_25685


namespace bus_journey_distance_l256_25622

/-- Represents the bus journey with an obstruction --/
structure BusJourney where
  initialSpeed : ℝ
  totalDistance : ℝ
  obstructionTime : ℝ
  delayTime : ℝ
  speedReductionFactor : ℝ
  lateArrivalTime : ℝ
  alternativeObstructionDistance : ℝ
  alternativeLateArrivalTime : ℝ

/-- Theorem stating that given the conditions, the total distance of the journey is 570 miles --/
theorem bus_journey_distance (j : BusJourney) 
  (h1 : j.obstructionTime = 2)
  (h2 : j.delayTime = 2/3)
  (h3 : j.speedReductionFactor = 5/6)
  (h4 : j.lateArrivalTime = 2.75)
  (h5 : j.alternativeObstructionDistance = 50)
  (h6 : j.alternativeLateArrivalTime = 2 + 1/3)
  : j.totalDistance = 570 := by
  sorry

end bus_journey_distance_l256_25622


namespace sin_alpha_for_point_l256_25674

/-- If the terminal side of angle α passes through point (-2, 4), then sin α = (2√5) / 5 -/
theorem sin_alpha_for_point (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * (Real.cos α) = -2 ∧ r * (Real.sin α) = 4) →
  Real.sin α = (2 * Real.sqrt 5) / 5 := by
sorry

end sin_alpha_for_point_l256_25674


namespace paytons_score_l256_25654

theorem paytons_score (total_students : ℕ) (students_without_payton : ℕ) 
  (avg_without_payton : ℝ) (avg_with_payton : ℝ) :
  total_students = 15 →
  students_without_payton = 14 →
  avg_without_payton = 80 →
  avg_with_payton = 81 →
  (students_without_payton * avg_without_payton + 
    (total_students - students_without_payton) * 
    ((total_students * avg_with_payton - students_without_payton * avg_without_payton) / 
    (total_students - students_without_payton))) / total_students = avg_with_payton →
  (total_students * avg_with_payton - students_without_payton * avg_without_payton) / 
  (total_students - students_without_payton) = 95 := by
sorry

end paytons_score_l256_25654


namespace elena_garden_petals_l256_25682

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals each lily has -/
def petals_per_lily : ℕ := 6

/-- The number of petals each tulip has -/
def petals_per_tulip : ℕ := 3

/-- The total number of flower petals in Elena's garden -/
def total_petals : ℕ := num_lilies * petals_per_lily + num_tulips * petals_per_tulip

theorem elena_garden_petals : total_petals = 63 := by
  sorry

end elena_garden_petals_l256_25682


namespace parabola_distance_theorem_l256_25657

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance_theorem (A : ℝ × ℝ) :
  parabola A.1 A.2 →  -- A lies on the parabola
  (A.1 - focus.1)^2 + (A.2 - focus.2)^2 = (B.1 - focus.1)^2 + (B.2 - focus.2)^2 →  -- |AF| = |BF|
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8 :=  -- |AB|^2 = 8, which implies |AB| = 2√2
by
  sorry


end parabola_distance_theorem_l256_25657


namespace tennis_balls_problem_l256_25620

theorem tennis_balls_problem (brian frodo lily : ℕ) : 
  brian = 2 * frodo ∧ 
  frodo = lily + 8 ∧ 
  brian = 22 → 
  lily = 3 := by sorry

end tennis_balls_problem_l256_25620


namespace remainder_theorem_l256_25662

theorem remainder_theorem : (43^43 + 43) % 44 = 42 := by
  sorry

end remainder_theorem_l256_25662


namespace unique_solution_l256_25663

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the equation TETA + BETA = GAMMA -/
def EquationSatisfied (T E B G M A : Digit) : Prop :=
  1000 * T.val + 100 * E.val + 10 * T.val + A.val +
  1000 * B.val + 100 * E.val + 10 * T.val + A.val =
  10000 * G.val + 1000 * A.val + 100 * M.val + 10 * M.val + A.val

/-- All digits are different except for repeated letters -/
def DigitsDifferent (T E B G M A : Digit) : Prop :=
  T ≠ E ∧ T ≠ B ∧ T ≠ G ∧ T ≠ M ∧ T ≠ A ∧
  E ≠ B ∧ E ≠ G ∧ E ≠ M ∧ E ≠ A ∧
  B ≠ G ∧ B ≠ M ∧ B ≠ A ∧
  G ≠ M ∧ G ≠ A ∧
  M ≠ A

theorem unique_solution :
  ∃! (T E B G M A : Digit),
    EquationSatisfied T E B G M A ∧
    DigitsDifferent T E B G M A ∧
    T.val = 4 ∧ E.val = 9 ∧ B.val = 5 ∧ G.val = 1 ∧ M.val = 8 ∧ A.val = 0 :=
by sorry

end unique_solution_l256_25663


namespace custom_mult_zero_l256_25601

/-- Custom multiplication operation for real numbers -/
def custom_mult (a b : ℝ) : ℝ := (a - b)^2

/-- Theorem stating that (x-y)^2 * (y-x)^2 = 0 under the custom multiplication -/
theorem custom_mult_zero (x y : ℝ) : custom_mult ((x - y)^2) ((y - x)^2) = 0 := by
  sorry

end custom_mult_zero_l256_25601


namespace sum_of_numbers_l256_25645

theorem sum_of_numbers : 0.45 + 0.003 + (1/4 : ℚ) = 0.703 := by
  sorry

end sum_of_numbers_l256_25645


namespace sum_45_52_base4_l256_25694

/-- Converts a natural number from base 10 to base 4 -/
def toBase4 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 4 to a natural number in base 10 -/
def fromBase4 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_45_52_base4 : 
  toBase4 (45 + 52) = [1, 2, 0, 1] :=
sorry

end sum_45_52_base4_l256_25694


namespace angle_C_value_triangle_area_l256_25614

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_condition (t : Triangle) : Prop :=
  t.a^2 - t.a * t.b - 2 * t.b^2 = 0

-- Theorem 1
theorem angle_C_value (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.B = π / 6) : 
  t.C = π / 3 := by sorry

-- Theorem 2
theorem triangle_area (t : Triangle) 
  (h1 : satisfies_condition t) 
  (h2 : t.C = 2 * π / 3) 
  (h3 : t.c = 14) : 
  (1 / 2) * t.a * t.b * Real.sin t.C = 14 * Real.sqrt 3 := by sorry

end angle_C_value_triangle_area_l256_25614


namespace mark_payment_l256_25626

def hours : ℕ := 3
def hourly_rate : ℚ := 15
def tip_percentage : ℚ := 20 / 100

def total_paid : ℚ :=
  let base_cost := hours * hourly_rate
  let tip := base_cost * tip_percentage
  base_cost + tip

theorem mark_payment : total_paid = 54 := by
  sorry

end mark_payment_l256_25626


namespace estate_distribution_theorem_l256_25661

/-- Represents the estate distribution problem --/
structure EstateDistribution where
  total : ℚ
  daughter_share : ℚ
  son_share : ℚ
  wife_share : ℚ
  nephew_share : ℚ
  gardener_share : ℚ

/-- Theorem stating the conditions and the result to be proved --/
theorem estate_distribution_theorem (e : EstateDistribution) : 
  e.daughter_share + e.son_share = (2 : ℚ) / (3 : ℚ) * e.total ∧ 
  e.daughter_share = (5 : ℚ) / (9 : ℚ) * ((2 : ℚ) / (3 : ℚ) * e.total) ∧
  e.son_share = (4 : ℚ) / (9 : ℚ) * ((2 : ℚ) / (3 : ℚ) * e.total) ∧
  e.wife_share = 3 * e.son_share ∧
  e.nephew_share = 1000 ∧
  e.gardener_share = 600 ∧
  e.total = e.daughter_share + e.son_share + e.wife_share + e.nephew_share + e.gardener_share
  →
  e.total = 2880 := by
  sorry


end estate_distribution_theorem_l256_25661


namespace cube_order_l256_25612

theorem cube_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end cube_order_l256_25612


namespace rectangle_width_range_l256_25678

/-- Given a wire of length 20 cm shaped into a rectangle with length at least 6 cm,
    prove that the width x satisfies 0 < x ≤ 20/3 -/
theorem rectangle_width_range :
  ∀ x : ℝ,
  (∃ l : ℝ, l ≥ 6 ∧ 2 * (x + l) = 20) →
  (0 < x ∧ x ≤ 20 / 3) :=
by sorry

end rectangle_width_range_l256_25678


namespace arithmetic_sequence_terms_l256_25681

/-- An arithmetic sequence with first term 10, last term 140, and common difference 5 has 27 terms. -/
theorem arithmetic_sequence_terms : ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) = a n + 5) →  -- arithmetic sequence with common difference 5
  a 1 = 10 →                    -- first term is 10
  (∃ m, a m = 140) →            -- last term is 140
  (∃ m, a m = 140 ∧ ∀ k, k > m → a k > 140) →  -- 140 is the last term not exceeding 140
  (∃ m, m = 27 ∧ a m = 140) :=  -- the sequence has exactly 27 terms
by sorry

end arithmetic_sequence_terms_l256_25681


namespace proportion_problem_l256_25629

theorem proportion_problem (x y : ℝ) : 
  (0.60 : ℝ) / x = y / 2 → 
  x = 0.19999999999999998 → 
  y = 6 := by sorry

end proportion_problem_l256_25629


namespace meat_for_hamburgers_l256_25646

/-- Given that 5 pounds of meat can make 12 hamburgers, 
    prove that 15 pounds of meat are needed to make 36 hamburgers. -/
theorem meat_for_hamburgers : 
  ∀ (meat_per_batch : ℝ) (hamburgers_per_batch : ℝ) (total_hamburgers : ℝ),
    meat_per_batch = 5 →
    hamburgers_per_batch = 12 →
    total_hamburgers = 36 →
    (meat_per_batch / hamburgers_per_batch) * total_hamburgers = 15 := by
  sorry

end meat_for_hamburgers_l256_25646
