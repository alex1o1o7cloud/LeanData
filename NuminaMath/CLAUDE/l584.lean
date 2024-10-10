import Mathlib

namespace acute_and_less_than_90_subset_l584_58487

-- Define the sets
def A : Set ℝ := {x | ∃ k : ℤ, k * 360 < x ∧ x < k * 360 + 90}
def B : Set ℝ := {x | 0 < x ∧ x < 90}
def C : Set ℝ := {x | x < 90}

-- Theorem statement
theorem acute_and_less_than_90_subset :
  B ∪ C ⊆ C := by sorry

end acute_and_less_than_90_subset_l584_58487


namespace never_return_to_initial_l584_58476

def transform (q : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := q
  (a * b, b * c, c * d, d * a)

def iterate_transform (q : ℝ × ℝ × ℝ × ℝ) (n : ℕ) : ℝ × ℝ × ℝ × ℝ :=
  match n with
  | 0 => q
  | n + 1 => transform (iterate_transform q n)

theorem never_return_to_initial (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hna : a ≠ 1) (hnb : b ≠ 1) (hnc : c ≠ 1) (hnd : d ≠ 1) :
  ∀ n : ℕ, iterate_transform (a, b, c, d) n ≠ (a, b, c, d) :=
sorry

end never_return_to_initial_l584_58476


namespace unknown_number_proof_l584_58417

theorem unknown_number_proof (n : ℕ) 
  (h1 : Nat.lcm 24 n = 168) 
  (h2 : Nat.gcd 24 n = 4) : 
  n = 28 := by
sorry

end unknown_number_proof_l584_58417


namespace digit_101_of_7_over_26_l584_58433

theorem digit_101_of_7_over_26 : ∃ (d : ℕ), d = 2 ∧ 
  (∃ (a : ℕ → ℕ), 
    (∀ n, a n < 10) ∧ 
    (∀ n, (7 * 10^(n+1)) / 26 % 10 = a n) ∧ 
    a 100 = d) := by
  sorry

end digit_101_of_7_over_26_l584_58433


namespace grey_cats_count_l584_58431

/-- The number of grey cats in a house after a series of events -/
def grey_cats_after_events : ℕ :=
  let initial_total : ℕ := 16
  let initial_white : ℕ := 2
  let initial_black : ℕ := (25 * initial_total) / 100
  let black_after_leaving : ℕ := initial_black / 2
  let white_after_arrival : ℕ := initial_white + 2
  let initial_grey : ℕ := initial_total - initial_white - initial_black
  initial_grey + 1

/-- Theorem stating the number of grey cats after the events -/
theorem grey_cats_count : grey_cats_after_events = 11 := by
  sorry

end grey_cats_count_l584_58431


namespace cubic_root_sum_l584_58405

theorem cubic_root_sum (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 7*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  k + m = 22 := by
sorry

end cubic_root_sum_l584_58405


namespace multiply_72_28_l584_58497

theorem multiply_72_28 : 72 * 28 = 4896 := by
  sorry

end multiply_72_28_l584_58497


namespace unhappy_no_skills_no_skills_purple_l584_58441

/-- Represents the properties of a snake --/
structure Snake where
  purple : Bool
  happy : Bool
  can_add : Bool
  can_subtract : Bool

/-- Tom's collection of snakes --/
def toms_snakes : Finset Snake := sorry

/-- The number of snakes in Tom's collection --/
axiom total_snakes : toms_snakes.card = 17

/-- The number of purple snakes --/
axiom purple_snakes : (toms_snakes.filter (fun s => s.purple)).card = 5

/-- All purple snakes are unhappy --/
axiom purple_unhappy : ∀ s ∈ toms_snakes, s.purple → ¬s.happy

/-- The number of happy snakes --/
axiom happy_snakes : (toms_snakes.filter (fun s => s.happy)).card = 7

/-- All happy snakes can add and subtract --/
axiom happy_skills : ∀ s ∈ toms_snakes, s.happy → s.can_add ∧ s.can_subtract

/-- No purple snakes can add or subtract --/
axiom purple_no_skills : ∀ s ∈ toms_snakes, s.purple → ¬s.can_add ∧ ¬s.can_subtract

theorem unhappy_no_skills :
  ∀ s ∈ toms_snakes, ¬s.happy → ¬s.can_add ∨ ¬s.can_subtract :=
sorry

theorem no_skills_purple :
  ∀ s ∈ toms_snakes, ¬s.can_add ∧ ¬s.can_subtract → s.purple :=
sorry

end unhappy_no_skills_no_skills_purple_l584_58441


namespace seventh_root_of_unity_product_l584_58403

theorem seventh_root_of_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^5 - 1) * (r^6 - 1) = 2 := by
  sorry

end seventh_root_of_unity_product_l584_58403


namespace simplified_fraction_value_l584_58425

theorem simplified_fraction_value (k : ℝ) : 
  ∃ (a b : ℤ), (10 * k + 15) / 5 = a * k + b → a / b = 2 / 3 := by
  sorry

end simplified_fraction_value_l584_58425


namespace runners_meet_time_l584_58450

/-- The length of the circular track in meters -/
def track_length : ℝ := 400

/-- The speeds of the three runners in meters per second -/
def runner_speeds : Fin 3 → ℝ
  | 0 => 5
  | 1 => 5.5
  | 2 => 6

/-- The time in seconds for the runners to meet again at the starting point -/
def meeting_time : ℝ := 800

theorem runners_meet_time :
  ∀ (i : Fin 3), ∃ (n : ℕ), (runner_speeds i * meeting_time) = n * track_length :=
sorry

end runners_meet_time_l584_58450


namespace shaded_area_of_concentric_circles_l584_58474

theorem shaded_area_of_concentric_circles (r1 r2 r3 : ℝ) (shaded unshaded : ℝ) : 
  r1 = 4 → r2 = 5 → r3 = 6 →
  shaded + unshaded = π * (r1^2 + r2^2 + r3^2) →
  shaded = (3/7) * unshaded →
  shaded = (1617 * π) / 70 := by
  sorry

end shaded_area_of_concentric_circles_l584_58474


namespace diophantine_equation_solutions_l584_58452

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 12 * x^2 + 7 * y^2 = 4620 ↔
    ((x = 7 ∨ x = -7) ∧ (y = 24 ∨ y = -24)) ∨
    ((x = 14 ∨ x = -14) ∧ (y = 18 ∨ y = -18)) := by
  sorry

end diophantine_equation_solutions_l584_58452


namespace simplify_and_evaluate_l584_58436

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2) :
  (1 - 1 / a) / ((a^2 - 1) / a) = Real.sqrt 2 - 1 := by
  sorry

end simplify_and_evaluate_l584_58436


namespace vector_dot_product_roots_l584_58447

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.sqrt 3 * (Real.cos x)^2)
noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, -2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem vector_dot_product_roots (x₁ x₂ : ℝ) :
  0 < x₁ ∧ x₁ < x₂ ∧ x₂ < π ∧
  dot_product (m x₁) (n x₁) = 1/2 - Real.sqrt 3 ∧
  dot_product (m x₂) (n x₂) = 1/2 - Real.sqrt 3 →
  Real.sin (x₁ - x₂) = -Real.sqrt 15 / 4 := by
sorry

end vector_dot_product_roots_l584_58447


namespace parallelogram_d_coordinates_l584_58451

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a vector in 2D space
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define a parallelogram
structure Parallelogram where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

def vector_between_points (p1 p2 : Point2D) : Vector2D :=
  { x := p2.x - p1.x, y := p2.y - p1.y }

theorem parallelogram_d_coordinates :
  ∀ (ABCD : Parallelogram),
    ABCD.A = { x := 1, y := 2 } →
    ABCD.B = { x := -2, y := 0 } →
    vector_between_points ABCD.A ABCD.C = { x := 2, y := -3 } →
    ABCD.D = { x := 6, y := 1 } :=
by
  sorry


end parallelogram_d_coordinates_l584_58451


namespace vertex_in_fourth_quadrant_l584_58458

-- Define the line y = x + m
def line (x m : ℝ) : ℝ := x + m

-- Define the parabola y = (x + m)^2 - 1
def parabola (x m : ℝ) : ℝ := (x + m)^2 - 1

-- Define what it means for a line to pass through the first, third, and fourth quadrants
def passes_through_134 (m : ℝ) : Prop :=
  ∃ (x1 x3 x4 : ℝ), 
    (x1 > 0 ∧ line x1 m > 0) ∧
    (x3 < 0 ∧ line x3 m < 0) ∧
    (x4 > 0 ∧ line x4 m < 0)

-- Define the fourth quadrant
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem vertex_in_fourth_quadrant (m : ℝ) :
  passes_through_134 m → in_fourth_quadrant (-m) (-1) :=
sorry

end vertex_in_fourth_quadrant_l584_58458


namespace range_of_a_l584_58495

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Define set B (parameterized by a)
def B (a : ℝ) : Set ℝ := {x : ℝ | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

-- Theorem statement
theorem range_of_a (a : ℝ) : (∃ x, x ∈ A ∩ B a) → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end range_of_a_l584_58495


namespace cube_skeleton_theorem_l584_58473

/-- The number of small cubes forming the skeleton of an n x n x n cube -/
def skeleton_cubes (n : ℕ) : ℕ := 12 * n - 16

/-- The number of small cubes to be removed to obtain the skeleton of an n x n x n cube -/
def removed_cubes (n : ℕ) : ℕ := n^3 - skeleton_cubes n

theorem cube_skeleton_theorem (n : ℕ) (h : n > 2) :
  skeleton_cubes n = 12 * n - 16 ∧
  removed_cubes n = n^3 - (12 * n - 16) := by
  sorry

#eval skeleton_cubes 6  -- Expected: 56
#eval removed_cubes 7   -- Expected: 275

end cube_skeleton_theorem_l584_58473


namespace parallel_line_k_value_l584_58493

/-- A line through (1, -8) and (k, 15) is parallel to 6x + 9y = -12 iff k = -33.5 -/
theorem parallel_line_k_value : ∀ k : ℝ,
  (∃ m b : ℝ, (∀ x y : ℝ, y = m*x + b ↔ (x = 1 ∧ y = -8) ∨ (x = k ∧ y = 15)) ∧
               m = -2/3) ↔
  k = -33.5 := by sorry

end parallel_line_k_value_l584_58493


namespace position_relationships_l584_58489

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the lines and planes
variable (a b : Line)
variable (α β γ : Plane)

-- State the theorem
theorem position_relationships :
  (∀ (a b : Line) (α : Plane), 
    parallel a b → subset b α → parallel_plane a α) = False ∧ 
  (∀ (a b : Line) (α : Plane), 
    parallel a b → parallel_plane a α → parallel_plane b α) = False ∧
  (∀ (a b : Line) (α β γ : Plane),
    intersect α β a → subset b γ → parallel_plane b β → subset a γ → parallel a b) = True :=
sorry

end position_relationships_l584_58489


namespace remainder_theorem_l584_58404

theorem remainder_theorem (x : ℝ) : ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ),
  (∀ x, x^100 = (x^2 - 3*x + 2) * Q x + R x) ∧
  (∃ a b, R = fun x ↦ a * x + b) ∧
  R = fun x ↦ 2^100 * (x - 1) - (x - 2) := by
  sorry

end remainder_theorem_l584_58404


namespace scientific_notation_of_316000000_l584_58422

/-- Definition of scientific notation -/
def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ ∃ (x : ℝ), x = a * (10 : ℝ) ^ n

/-- The number to be represented in scientific notation -/
def number : ℝ := 316000000

/-- Theorem stating that 316000000 in scientific notation is 3.16 × 10^8 -/
theorem scientific_notation_of_316000000 :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n ∧ number = a * (10 : ℝ) ^ n ∧ a = 3.16 ∧ n = 8 :=
sorry

end scientific_notation_of_316000000_l584_58422


namespace domain_of_f_half_x_l584_58481

-- Define the original function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(lg x)
def domain_f_lg_x : Set ℝ := { x | 0.1 ≤ x ∧ x ≤ 100 }

-- State the theorem
theorem domain_of_f_half_x (h : ∀ x ∈ domain_f_lg_x, f (Real.log x / Real.log 10) = f (Real.log x / Real.log 10)) :
  { x : ℝ | f (x / 2) = f (x / 2) } = { x : ℝ | -2 ≤ x ∧ x ≤ 4 } :=
sorry

end domain_of_f_half_x_l584_58481


namespace complex_division_l584_58440

theorem complex_division (z₁ z₂ : ℂ) : z₁ = 1 + I ∧ z₂ = 2 * I → z₂ / z₁ = 1 + I := by
  sorry

end complex_division_l584_58440


namespace trajectory_and_intersection_l584_58446

-- Define the trajectory C
def C (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y + Real.sqrt 3)^2) + Real.sqrt (x^2 + (y - Real.sqrt 3)^2) = 4

-- Define the intersection line
def intersectionLine (x y : ℝ) : Prop := y = (1/2) * x

theorem trajectory_and_intersection :
  -- The equation of trajectory C
  (∀ x y, C x y ↔ x^2 + y^2/4 = 1) ∧
  -- The length of chord AB
  (∃ x₁ y₁ x₂ y₂, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ 
    intersectionLine x₁ y₁ ∧ intersectionLine x₂ y₂ ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4) :=
by sorry

end trajectory_and_intersection_l584_58446


namespace problem_solution_l584_58406

theorem problem_solution (x y m a b : ℝ) : 
  (∃ k : ℤ, (x - 1 = k^2 * 4)) →
  ((4 * x + y)^(1/3) = 3) →
  (m^2 = y - x) →
  (5 + m = a + b) →
  (∃ n : ℤ, a = n) →
  (0 < b) →
  (b < 1) →
  (m = Real.sqrt 2 ∧ a - (Real.sqrt 2 - b)^2 = 5) := by
sorry

end problem_solution_l584_58406


namespace total_marbles_count_l584_58402

/-- The number of marbles Connie has -/
def connie_marbles : ℕ := 39

/-- The number of marbles Juan has -/
def juan_marbles : ℕ := connie_marbles + 25

/-- The number of marbles Maria has -/
def maria_marbles : ℕ := 2 * juan_marbles

/-- The total number of marbles for all three people -/
def total_marbles : ℕ := connie_marbles + juan_marbles + maria_marbles

theorem total_marbles_count : total_marbles = 231 := by
  sorry

end total_marbles_count_l584_58402


namespace three_good_pairs_l584_58460

-- Define a structure for a line in slope-intercept form
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the lines
def L1 : Line := { slope := 2, intercept := 3 }
def L2 : Line := { slope := 2, intercept := 3 }
def L3 : Line := { slope := 4, intercept := -2 }
def L4 : Line := { slope := -4, intercept := 3 }
def L5 : Line := { slope := -4, intercept := 3 }

-- Define what it means for two lines to be parallel
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Line) : Prop := l1.slope * l2.slope = -1

-- Define what it means for two lines to be "good"
def good (l1 l2 : Line) : Prop := parallel l1 l2 ∨ perpendicular l1 l2

-- The main theorem
theorem three_good_pairs :
  ∃ (pairs : List (Line × Line)),
    pairs.length = 3 ∧
    (∀ p ∈ pairs, good p.1 p.2) ∧
    (∀ l1 l2 : Line, l1 ≠ l2 → good l1 l2 → (l1, l2) ∈ pairs ∨ (l2, l1) ∈ pairs) :=
by
  sorry

end three_good_pairs_l584_58460


namespace simplify_expression_l584_58457

theorem simplify_expression : 
  (Real.sqrt 8 + Real.sqrt 12) - (2 * Real.sqrt 3 - Real.sqrt 2) = 3 * Real.sqrt 2 := by
  sorry

end simplify_expression_l584_58457


namespace train_length_l584_58428

/-- The length of a train given its speed and time to pass an observer -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 144 → time_s = 10 → speed_kmh * (1000 / 3600) * time_s = 400 := by
  sorry

#check train_length

end train_length_l584_58428


namespace prob_rain_holiday_l584_58494

/-- The probability of rain on Friday without a storm -/
def prob_rain_friday : ℝ := 0.3

/-- The probability of rain on Monday without a storm -/
def prob_rain_monday : ℝ := 0.6

/-- The increase in probability of rain if a storm develops -/
def storm_increase : ℝ := 0.2

/-- The probability of a storm developing -/
def prob_storm : ℝ := 0.5

/-- Assumption that all probabilities are independent -/
axiom probabilities_independent : True

/-- The probability of rain on at least one day during the holiday -/
def prob_rain_at_least_one_day : ℝ := 
  1 - (prob_storm * (1 - (prob_rain_friday + storm_increase)) * (1 - (prob_rain_monday + storm_increase)) + 
       (1 - prob_storm) * (1 - prob_rain_friday) * (1 - prob_rain_monday))

theorem prob_rain_holiday : prob_rain_at_least_one_day = 0.81 := by
  sorry

end prob_rain_holiday_l584_58494


namespace arithmetic_sequence_2015th_term_l584_58471

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_2015th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 2) 
  (h_a5 : a 5 = 6) : 
  a 2015 = 2016 := by
sorry

end arithmetic_sequence_2015th_term_l584_58471


namespace right_triangle_condition_l584_58442

theorem right_triangle_condition (a d : ℝ) (ha : a > 0) (hd : d > 1) :
  (a * d^2)^2 = a^2 + (a * d)^2 ↔ d = Real.sqrt ((1 + Real.sqrt 5) / 2) :=
sorry

end right_triangle_condition_l584_58442


namespace book_cost_price_l584_58416

/-- The cost price of a book sold for $200 with a 20% profit is $166.67 -/
theorem book_cost_price (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 200 ∧ profit_percentage = 20 → 
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 166.67 := by
sorry

end book_cost_price_l584_58416


namespace gcf_lcm_sum_9_15_45_l584_58468

theorem gcf_lcm_sum_9_15_45 : ∃ (C D : ℕ),
  (C = Nat.gcd 9 (Nat.gcd 15 45)) ∧
  (D = Nat.lcm 9 (Nat.lcm 15 45)) ∧
  (C + D = 60) := by
sorry

end gcf_lcm_sum_9_15_45_l584_58468


namespace three_digit_number_property_l584_58488

theorem three_digit_number_property (A : ℕ) : 
  100 ≤ A → A < 1000 → 
  let B := 1001 * A
  (((B / 7) / 11) / 13) = A := by
  sorry

end three_digit_number_property_l584_58488


namespace two_pairs_satisfy_equation_l584_58491

theorem two_pairs_satisfy_equation :
  ∃ (x₁ y₁ x₂ y₂ : ℕ),
    (x₁ = 2 ∧ y₁ = 2 ∧ 2 * x₁^3 = y₁^4) ∧
    (x₂ = 32 ∧ y₂ = 16 ∧ 2 * x₂^3 = y₂^4) :=
by sorry

end two_pairs_satisfy_equation_l584_58491


namespace x_equation_implies_a_plus_b_l584_58424

theorem x_equation_implies_a_plus_b (x : ℝ) (a b : ℕ+) :
  x^2 + 5*x + 5/x + 1/x^2 = 34 →
  x = a + Real.sqrt b →
  (a : ℝ) + b = 5 := by sorry

end x_equation_implies_a_plus_b_l584_58424


namespace smallest_number_l584_58482

/-- Converts a number from base k to decimal --/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

/-- The binary number 111111₍₂₎ --/
def binary_num : List Nat := [1, 1, 1, 1, 1, 1]

/-- The base-6 number 150₍₆₎ --/
def base6_num : List Nat := [0, 5, 1]

/-- The base-4 number 1000₍₄₎ --/
def base4_num : List Nat := [0, 0, 0, 1]

/-- The octal number 101₍₈₎ --/
def octal_num : List Nat := [1, 0, 1]

theorem smallest_number :
  to_decimal binary_num 2 < to_decimal base6_num 6 ∧
  to_decimal binary_num 2 < to_decimal base4_num 4 ∧
  to_decimal binary_num 2 < to_decimal octal_num 8 :=
by sorry

end smallest_number_l584_58482


namespace rate_of_discount_l584_58465

/-- Calculate the rate of discount given the marked price and selling price -/
theorem rate_of_discount (marked_price selling_price : ℝ) :
  marked_price = 200 →
  selling_price = 120 →
  (marked_price - selling_price) / marked_price * 100 = 40 := by
sorry

end rate_of_discount_l584_58465


namespace number_with_specific_remainders_l584_58418

theorem number_with_specific_remainders (n : ℕ) :
  ∃ (x : ℕ+), 
    x > 1 ∧ 
    n % x = 2 ∧ 
    (2 * n) % x = 4 → 
    x = 6 := by
  sorry

end number_with_specific_remainders_l584_58418


namespace working_hours_growth_equation_l584_58469

-- Define the initial and final average working hours
def initial_hours : ℝ := 40
def final_hours : ℝ := 48.4

-- Define the growth rate variable
variable (x : ℝ)

-- State the theorem
theorem working_hours_growth_equation :
  initial_hours * (1 + x)^2 = final_hours := by
  sorry

end working_hours_growth_equation_l584_58469


namespace f_properties_l584_58414

noncomputable def f (x : ℝ) : ℝ := 1/2 * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 1

theorem f_properties :
  let period : ℝ := Real.pi
  let max_value : ℝ := 7/4
  let min_value : ℝ := (5 + Real.sqrt 3) / 4
  let interval : Set ℝ := Set.Icc (Real.pi / 12) (Real.pi / 4)
  (∀ x : ℝ, f (x + period) = f x) ∧
  (∀ t : ℝ, t > 0 → (∀ x : ℝ, f (x + t) = f x) → t ≥ period) ∧
  (∃ x ∈ interval, f x = max_value ∧ ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, f x = min_value ∧ ∀ y ∈ interval, f y ≥ f x) ∧
  (f (Real.pi / 6) = max_value) ∧
  (f (Real.pi / 12) = min_value) ∧
  (f (Real.pi / 4) = min_value) :=
by sorry

end f_properties_l584_58414


namespace median_salary_is_manager_salary_l584_58437

/-- Represents a job position with its title, number of employees, and salary -/
structure Position where
  title : String
  count : Nat
  salary : Nat

/-- Calculates the median salary given a list of positions -/
def medianSalary (positions : List Position) : Nat :=
  sorry

/-- The list of positions in the company -/
def companyPositions : List Position :=
  [{ title := "CEO", count := 1, salary := 140000 },
   { title := "Senior Manager", count := 4, salary := 95000 },
   { title := "Manager", count := 13, salary := 78000 },
   { title := "Assistant Manager", count := 7, salary := 55000 },
   { title := "Clerk", count := 38, salary := 25000 }]

/-- The total number of employees in the company -/
def totalEmployees : Nat :=
  companyPositions.foldl (fun acc pos => acc + pos.count) 0

theorem median_salary_is_manager_salary :
  medianSalary companyPositions = 78000 ∧ totalEmployees = 63 := by
  sorry

end median_salary_is_manager_salary_l584_58437


namespace walnut_problem_l584_58449

/-- Calculates the final number of walnuts in the main burrow after the actions of three squirrels. -/
def final_walnut_count (initial : ℕ) (boy_gather boy_drop boy_hide : ℕ)
  (girl_bring girl_eat girl_give girl_lose girl_knock : ℕ)
  (third_gather third_drop third_hide third_return third_give : ℕ) : ℕ :=
  initial + boy_gather - boy_drop - boy_hide +
  girl_bring - girl_eat - girl_give - girl_lose - girl_knock +
  third_return

/-- The final number of walnuts in the main burrow is 44. -/
theorem walnut_problem :
  final_walnut_count 30 20 4 8 15 5 4 3 2 10 1 3 6 1 = 44 := by
  sorry

end walnut_problem_l584_58449


namespace stratified_sampling_theorem_l584_58498

/-- Represents the number of cities in a group -/
def num_cities : ℕ := 8

/-- Represents the probability of a city being selected -/
def selection_probability : ℚ := 1/4

/-- Represents the number of cities drawn from the group -/
def cities_drawn : ℚ := num_cities * selection_probability

theorem stratified_sampling_theorem :
  cities_drawn = 2 := by sorry

end stratified_sampling_theorem_l584_58498


namespace pop_spent_15_l584_58438

def cereal_spending (pop crackle snap : ℝ) : Prop :=
  pop + crackle + snap = 150 ∧
  snap = 2 * crackle ∧
  crackle = 3 * pop

theorem pop_spent_15 :
  ∃ (pop crackle snap : ℝ), cereal_spending pop crackle snap ∧ pop = 15 := by
  sorry

end pop_spent_15_l584_58438


namespace expression_evaluation_l584_58456

theorem expression_evaluation (a b : ℝ) (h1 : a = 2) (h2 : b = -1) :
  5 * (a^2 + b) - 2 * (b + 2 * a^2) + 2 * b = -1 := by
  sorry

end expression_evaluation_l584_58456


namespace perfect_square_count_l584_58464

theorem perfect_square_count : 
  ∃! (count : ℕ), ∃ (S : Finset ℕ), 
    (Finset.card S = count) ∧ 
    (∀ n, n ∈ S ↔ ∃ x : ℤ, (4:ℤ)^n - 15 = x^2) :=
by sorry

end perfect_square_count_l584_58464


namespace square_sum_of_roots_l584_58409

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

-- State the theorem
theorem square_sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : (a + b)^2 = 4 := by
  sorry

end square_sum_of_roots_l584_58409


namespace max_M_value_l584_58401

theorem max_M_value (x y z u : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hu : u > 0)
  (eq1 : x - 2*y = z - 2*u) (eq2 : 2*y*z = u*x) (h_zy : z ≥ y) :
  ∃ M : ℝ, M > 0 ∧ M ≤ z/y ∧ ∀ N : ℝ, (N > 0 ∧ N ≤ z/y → N ≤ M) ∧ M = 6 + 4*Real.sqrt 2 :=
sorry

end max_M_value_l584_58401


namespace number_with_quotient_and_remainder_l584_58459

theorem number_with_quotient_and_remainder (x : ℕ) : 
  (x / 7 = 4) ∧ (x % 7 = 6) → x = 34 := by
  sorry

end number_with_quotient_and_remainder_l584_58459


namespace children_per_seat_l584_58410

theorem children_per_seat (total_children : ℕ) (total_seats : ℕ) 
  (h1 : total_children = 58) (h2 : total_seats = 29) : 
  total_children / total_seats = 2 := by
sorry

end children_per_seat_l584_58410


namespace factorization_of_x_squared_plus_x_l584_58475

theorem factorization_of_x_squared_plus_x (x : ℝ) : x^2 + x = x * (x + 1) := by
  sorry

end factorization_of_x_squared_plus_x_l584_58475


namespace money_left_proof_l584_58477

def salary : ℚ := 150000.00000000003

def food_fraction : ℚ := 1 / 5
def rent_fraction : ℚ := 1 / 10
def clothes_fraction : ℚ := 3 / 5

def money_left : ℚ := salary - (salary * food_fraction + salary * rent_fraction + salary * clothes_fraction)

theorem money_left_proof : money_left = 15000 := by
  sorry

end money_left_proof_l584_58477


namespace fraction_equality_implies_numerator_equality_l584_58445

theorem fraction_equality_implies_numerator_equality 
  (a b c : ℝ) (h : c ≠ 0) : a / c = b / c → a = b := by
  sorry

end fraction_equality_implies_numerator_equality_l584_58445


namespace binary_to_decimal_110011_l584_58490

/-- Converts a list of binary digits to a decimal number -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number we want to convert -/
def binaryNumber : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary number 110011₂ is equal to the decimal number 51 -/
theorem binary_to_decimal_110011 : binaryToDecimal binaryNumber = 51 := by
  sorry

end binary_to_decimal_110011_l584_58490


namespace leesburg_population_l584_58496

theorem leesburg_population (salem_factor : ℕ) (moved_out : ℕ) (women_ratio : ℚ) (women_count : ℕ) :
  salem_factor = 15 →
  moved_out = 130000 →
  women_ratio = 1/2 →
  women_count = 377050 →
  ∃ (leesburg_pop : ℕ), leesburg_pop = 58940 ∧ 
    salem_factor * leesburg_pop = 2 * women_count + moved_out :=
sorry

end leesburg_population_l584_58496


namespace current_price_calculation_l584_58430

/-- The current unit price after price adjustments -/
def current_price (x : ℝ) : ℝ := (1 - 0.25) * (x + 10)

/-- Theorem stating that the current price calculation is correct -/
theorem current_price_calculation (x : ℝ) : 
  current_price x = (1 - 0.25) * (x + 10) := by
  sorry

end current_price_calculation_l584_58430


namespace second_pipe_rate_l584_58480

def well_capacity : ℝ := 1200
def first_pipe_rate : ℝ := 48
def filling_time : ℝ := 5

theorem second_pipe_rate : 
  ∃ (rate : ℝ), 
    rate * filling_time + first_pipe_rate * filling_time = well_capacity ∧ 
    rate = 192 :=
by sorry

end second_pipe_rate_l584_58480


namespace unique_solution_for_diophantine_equation_l584_58413

theorem unique_solution_for_diophantine_equation :
  ∃! (a b : ℕ), 
    Nat.Prime a ∧ 
    b > 0 ∧ 
    9 * (2 * a + b)^2 = 509 * (4 * a + 511 * b) ∧
    a = 251 ∧ 
    b = 7 := by sorry

end unique_solution_for_diophantine_equation_l584_58413


namespace factor_expression_l584_58426

theorem factor_expression (x : ℝ) : 63 * x + 42 = 21 * (3 * x + 2) := by
  sorry

end factor_expression_l584_58426


namespace complex_number_line_l584_58400

theorem complex_number_line (z : ℂ) (h : 2 * (1 + Complex.I) * z = 1 - Complex.I) :
  z.im = -1/2 * z.re := by
  sorry

end complex_number_line_l584_58400


namespace total_marbles_l584_58472

/-- Given a collection of red, blue, and green marbles, where:
  1. There are 25% more red marbles than blue marbles
  2. There are 60% more green marbles than red marbles
  3. The number of red marbles is r
Prove that the total number of marbles in the collection is 3.4r -/
theorem total_marbles (r : ℝ) (b : ℝ) (g : ℝ) 
  (h1 : r = 1.25 * b) 
  (h2 : g = 1.6 * r) : 
  r + b + g = 3.4 * r := by
  sorry


end total_marbles_l584_58472


namespace some_negative_numbers_satisfy_inequality_l584_58415

theorem some_negative_numbers_satisfy_inequality :
  (∃ x : ℝ, x < 0 ∧ (1 + x) * (1 - 9 * x) > 0) ↔
  (∃ x₀ : ℝ, x₀ < 0 ∧ (1 + x₀) * (1 - 9 * x₀) > 0) :=
by sorry

end some_negative_numbers_satisfy_inequality_l584_58415


namespace divisibility_conditions_l584_58454

theorem divisibility_conditions (n : ℕ) (hn : n ≥ 1) :
  (n ∣ 2^n - 1 ↔ n = 1) ∧
  (n % 2 = 1 ∧ n ∣ 3^n + 1 ↔ n = 1) := by
  sorry

end divisibility_conditions_l584_58454


namespace inscribed_squares_segment_product_l584_58448

theorem inscribed_squares_segment_product :
  ∀ (a b : ℝ),
    (∃ (inner_area outer_area : ℝ),
      inner_area = 16 ∧
      outer_area = 18 ∧
      (∃ (inner_side outer_side : ℝ),
        inner_side^2 = inner_area ∧
        outer_side^2 = outer_area ∧
        a + b = outer_side ∧
        (a^2 + b^2) = inner_side^2)) →
    a * b = -7 := by
  sorry

end inscribed_squares_segment_product_l584_58448


namespace complex_fraction_simplification_l584_58470

theorem complex_fraction_simplification :
  let z₁ : ℂ := -2 + 5*I
  let z₂ : ℂ := 6 - 3*I
  z₁ / z₂ = -9/15 + 8/15*I :=
by sorry

end complex_fraction_simplification_l584_58470


namespace sqrt_eight_equals_two_sqrt_two_l584_58421

theorem sqrt_eight_equals_two_sqrt_two : Real.sqrt 8 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_eight_equals_two_sqrt_two_l584_58421


namespace garrett_roses_count_l584_58420

/-- The number of red roses Mrs. Santiago has -/
def santiago_roses : ℕ := 58

/-- The difference in the number of roses between Mrs. Santiago and Mrs. Garrett -/
def difference : ℕ := 34

/-- The number of red roses Mrs. Garrett has -/
def garrett_roses : ℕ := santiago_roses - difference

theorem garrett_roses_count : garrett_roses = 24 := by
  sorry

end garrett_roses_count_l584_58420


namespace infinite_geometric_series_first_term_l584_58443

/-- For an infinite geometric series with common ratio 1/4 and sum 80, the first term is 60. -/
theorem infinite_geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (h1 : r = (1 : ℝ) / 4)
  (h2 : S = 80)
  (h3 : S = a / (1 - r)) :
  a = 60 := by
  sorry

end infinite_geometric_series_first_term_l584_58443


namespace not_kth_power_consecutive_product_l584_58408

theorem not_kth_power_consecutive_product (m k : ℕ) (hk : k > 1) :
  ¬ ∃ (a : ℤ), m * (m + 1) = a^k := by
  sorry

end not_kth_power_consecutive_product_l584_58408


namespace florist_roses_l584_58434

/-- The number of roses a florist has after selling some and picking more. -/
def roses_after_selling_and_picking (initial : ℕ) (sold : ℕ) (picked : ℕ) : ℕ :=
  initial - sold + picked

/-- Theorem: Given the specific numbers from the problem, 
    the florist ends up with 40 roses. -/
theorem florist_roses : roses_after_selling_and_picking 37 16 19 = 40 := by
  sorry

end florist_roses_l584_58434


namespace sharon_trip_distance_l584_58455

def normal_time : ℝ := 200
def reduced_speed_time : ℝ := 310
def speed_reduction : ℝ := 30

def trip_distance : ℝ := 220

theorem sharon_trip_distance :
  let normal_speed := trip_distance / normal_time
  let reduced_speed := normal_speed - speed_reduction / 60
  (trip_distance / 3) / normal_speed + (2 * trip_distance / 3) / reduced_speed = reduced_speed_time :=
by sorry

end sharon_trip_distance_l584_58455


namespace unique_number_with_18_factors_l584_58463

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

theorem unique_number_with_18_factors (x : ℕ) : 
  num_factors x = 18 ∧ 
  18 ∣ x ∧ 
  24 ∣ x → 
  x = 288 := by sorry

end unique_number_with_18_factors_l584_58463


namespace inequality_solution_set_l584_58467

theorem inequality_solution_set (x : ℝ) : (x + 1) / (x - 1) ≤ 0 ↔ x ∈ Set.Icc (-1) 1 \ {1} := by
  sorry

end inequality_solution_set_l584_58467


namespace afternoon_email_count_l584_58432

/-- Represents the number of emails Jack received at different times of the day -/
structure EmailCount where
  morning : ℕ
  afternoon : ℕ
  evening : ℕ

/-- The theorem stating that Jack received 7 emails in the afternoon -/
theorem afternoon_email_count (e : EmailCount) 
  (h1 : e.morning = 10)
  (h2 : e.evening = 17)
  (h3 : e.morning = e.afternoon + 3) :
  e.afternoon = 7 := by
  sorry

end afternoon_email_count_l584_58432


namespace equation_solution_l584_58485

theorem equation_solution : ∃ x : ℝ, (x - 1) / (2 * x + 1) = 1 ∧ x = -2 := by
  sorry

end equation_solution_l584_58485


namespace job_completion_time_l584_58492

/-- Given two people A and B who can complete a job individually in 9 and 18 days respectively,
    this theorem proves that they can complete the job together in 6 days. -/
theorem job_completion_time (a_time b_time combined_time : ℚ) 
  (ha : a_time = 9)
  (hb : b_time = 18)
  (hc : combined_time = 6)
  (h_combined : (1 / a_time + 1 / b_time)⁻¹ = combined_time) : 
  combined_time = 6 := by sorry

end job_completion_time_l584_58492


namespace max_digit_sum_18_l584_58411

/-- Represents a digit (1 to 9) -/
def Digit := {d : ℕ // 1 ≤ d ∧ d ≤ 9}

/-- Calculates the value of a number with n identical digits -/
def digitSum (d : Digit) (n : ℕ) : ℕ := d.val * ((10^n - 1) / 9)

/-- The main theorem -/
theorem max_digit_sum_18 :
  ∃ (a b c : Digit) (n₁ n₂ : ℕ+),
    n₁ ≠ n₂ ∧
    digitSum c (2 * n₁) - digitSum b n₁ = (digitSum a n₁)^2 ∧
    digitSum c (2 * n₂) - digitSum b n₂ = (digitSum a n₂)^2 ∧
    ∀ (a' b' c' : Digit),
      (∃ (m₁ m₂ : ℕ+), m₁ ≠ m₂ ∧
        digitSum c' (2 * m₁) - digitSum b' m₁ = (digitSum a' m₁)^2 ∧
        digitSum c' (2 * m₂) - digitSum b' m₂ = (digitSum a' m₂)^2) →
      a'.val + b'.val + c'.val ≤ a.val + b.val + c.val ∧
      a.val + b.val + c.val = 18 :=
by sorry

end max_digit_sum_18_l584_58411


namespace largest_sum_and_simplification_l584_58461

theorem largest_sum_and_simplification : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/2, 1/3 + 1/9, 1/3 + 1/6]
  (∀ s ∈ sums, s ≤ (1/3 + 1/2)) ∧ (1/3 + 1/2 = 5/6) := by
  sorry

end largest_sum_and_simplification_l584_58461


namespace factorial_fraction_simplification_l584_58478

theorem factorial_fraction_simplification (N : ℕ) :
  (Nat.factorial (N + 1) * (N - 2)) / Nat.factorial (N + 2) = 
  (Nat.factorial N * (N - 2)) / (N + 2) := by
sorry

end factorial_fraction_simplification_l584_58478


namespace school_A_percentage_l584_58427

theorem school_A_percentage (total : ℕ) (science_percent : ℚ) (non_science : ℕ) :
  total = 300 →
  science_percent = 30 / 100 →
  non_science = 42 →
  ∃ (school_A_percent : ℚ),
    school_A_percent = 20 / 100 ∧
    non_science = (1 - science_percent) * (school_A_percent * total) :=
by sorry

end school_A_percentage_l584_58427


namespace det_of_specific_matrix_l584_58419

theorem det_of_specific_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![6, -2; -3, 5]
  Matrix.det A = 24 := by
sorry

end det_of_specific_matrix_l584_58419


namespace alices_number_l584_58499

theorem alices_number (n : ℕ) : 
  180 ∣ n → 75 ∣ n → 900 ≤ n → n < 3000 → n = 900 ∨ n = 1800 ∨ n = 2700 := by
  sorry

end alices_number_l584_58499


namespace repeating_decimal_value_l584_58435

/-- The repeating decimal 0.0000253253325333... -/
def x : ℚ := 253 / 990000

/-- The result of (10^7 - 10^5) * x -/
def result : ℚ := (10^7 - 10^5) * x

/-- Theorem stating that the result is equal to 253/990 -/
theorem repeating_decimal_value : result = 253 / 990 := by
  sorry

end repeating_decimal_value_l584_58435


namespace f_properties_l584_58462

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - 8^x)

theorem f_properties :
  (∀ x, f x ≠ 0 → x ≤ 2/3) ∧
  (∀ y, (∃ x, f x = y) → 0 ≤ y ∧ y < 2) ∧
  (∀ x, f x ≤ 1 → Real.log 3 / Real.log 8 ≤ x ∧ x ≤ 2/3) :=
sorry

end f_properties_l584_58462


namespace tan_sin_function_property_l584_58484

/-- Given a function f(x) = tan x + sin x + 1, prove that if f(b) = 2, then f(-b) = 0 -/
theorem tan_sin_function_property (b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.tan x + Real.sin x + 1
  f b = 2 → f (-b) = 0 := by
sorry

end tan_sin_function_property_l584_58484


namespace solution_set_f_min_value_2a_plus_b_min_value_2a_plus_b_is_9_8_l584_58423

-- Define the function f(x)
def f (x : ℝ) : ℝ := x + 1 + |3 - x|

-- Theorem for the solution set of f(x) ≤ 6
theorem solution_set_f (x : ℝ) (h : x ≥ -1) :
  f x ≤ 6 ↔ -1 ≤ x ∧ x ≤ 4 := by sorry

-- Define n as the minimum value of f(x)
def n : ℝ := 4

-- Theorem for the minimum value of 2a + b
theorem min_value_2a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * n * a * b = a + 2 * b) :
  2 * a + b ≥ 9/8 := by sorry

-- Theorem stating that 9/8 is indeed the minimum value
theorem min_value_2a_plus_b_is_9_8 :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * n * a * b = a + 2 * b ∧ 2 * a + b = 9/8 := by sorry

end solution_set_f_min_value_2a_plus_b_min_value_2a_plus_b_is_9_8_l584_58423


namespace equation_solution_l584_58412

theorem equation_solution : ∃! x : ℝ, 3 * x - 4 = -2 * x + 11 ∧ x = 3 := by sorry

end equation_solution_l584_58412


namespace hash_composition_20_l584_58486

-- Define the # operation
def hash (N : ℝ) : ℝ := (0.5 * N)^2 + 1

-- State the theorem
theorem hash_composition_20 : hash (hash (hash 20)) = 1627102.64 := by
  sorry

end hash_composition_20_l584_58486


namespace paint_calculation_l584_58483

theorem paint_calculation (num_bedrooms : ℕ) (num_other_rooms : ℕ) 
  (total_cans : ℕ) (white_can_size : ℚ) :
  num_bedrooms = 3 →
  num_other_rooms = 2 * num_bedrooms →
  total_cans = 10 →
  white_can_size = 3 →
  (total_cans - num_bedrooms) * white_can_size / num_other_rooms = 3.5 := by
  sorry

end paint_calculation_l584_58483


namespace different_set_l584_58453

def set_A : Set ℝ := {x | x = 1}
def set_B : Set ℝ := {x | x^2 = 1}
def set_C : Set ℝ := {1}
def set_D : Set ℝ := {y | (y - 1)^2 = 0}

theorem different_set :
  (set_A = set_C) ∧ (set_A = set_D) ∧ (set_C = set_D) ∧ (set_B ≠ set_A) ∧ (set_B ≠ set_C) ∧ (set_B ≠ set_D) :=
sorry

end different_set_l584_58453


namespace angles_between_plane_and_legs_l584_58466

/-- Given a right triangle with an acute angle α and a plane through the smallest median
    forming an angle β with the triangle's plane, this theorem states the angles between
    the plane and the legs of the triangle. -/
theorem angles_between_plane_and_legs (α β : Real) 
  (h_acute : 0 < α ∧ α < Real.pi / 2)
  (h_right_triangle : True)  -- Placeholder for the right triangle condition
  (h_smallest_median : True) -- Placeholder for the smallest median condition
  (h_plane_angle : True)     -- Placeholder for the plane angle condition
  : ∃ (γ θ : Real),
    γ = Real.arcsin (Real.sin β * Real.cos α) ∧
    θ = Real.arcsin (Real.sin β * Real.sin α) :=
by sorry

end angles_between_plane_and_legs_l584_58466


namespace square_semicircle_perimeter_l584_58444

theorem square_semicircle_perimeter : 
  let square_side : ℝ := 2 / Real.pi
  let semicircle_diameter : ℝ := square_side
  let full_circle_circumference : ℝ := Real.pi * semicircle_diameter
  let region_perimeter : ℝ := 2 * full_circle_circumference
  region_perimeter = 4 := by
  sorry

end square_semicircle_perimeter_l584_58444


namespace nested_radical_floor_l584_58479

theorem nested_radical_floor (y : ℝ) (B : ℤ) : 
  y > 0 → y^2 = 10 + y → B = ⌊10 + y⌋ → B = 13 := by sorry

end nested_radical_floor_l584_58479


namespace library_fine_fifth_day_l584_58429

def fine_calculation (initial_fine : Float) (increase : Float) (days : Nat) : Float :=
  let rec calc_fine (current_fine : Float) (day : Nat) : Float :=
    if day = 0 then
      current_fine
    else
      let increased := current_fine + increase
      let doubled := current_fine * 2
      calc_fine (min increased doubled) (day - 1)
  calc_fine initial_fine days

theorem library_fine_fifth_day :
  fine_calculation 0.07 0.30 4 = 0.86 := by
  sorry

end library_fine_fifth_day_l584_58429


namespace arrangement_theorem_l584_58407

def number_of_arrangements (n m : ℕ) : ℕ := Nat.choose n m * Nat.factorial m

theorem arrangement_theorem : number_of_arrangements 6 4 = 360 := by
  sorry

end arrangement_theorem_l584_58407


namespace binomial_coefficient_modulo_power_of_two_l584_58439

theorem binomial_coefficient_modulo_power_of_two 
  (n : ℕ) (r : ℕ) (h_r_odd : Odd r) :
  ∃ i : ℕ, i < 2^n ∧ Nat.choose (2^n + i) i ≡ r [MOD 2^(n+1)] := by
  sorry

end binomial_coefficient_modulo_power_of_two_l584_58439
