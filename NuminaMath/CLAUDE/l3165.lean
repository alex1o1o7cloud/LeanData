import Mathlib

namespace simplify_expression_l3165_316521

theorem simplify_expression (x : ℝ) :
  3*x + 4*x^2 + 2 - (5 - 3*x - 5*x^2 + x^3) = -x^3 + 9*x^2 + 6*x - 3 := by
  sorry

end simplify_expression_l3165_316521


namespace johns_age_difference_l3165_316577

theorem johns_age_difference (brother_age : ℕ) (john_age : ℕ) : 
  brother_age = 8 → 
  john_age + brother_age = 10 → 
  6 * brother_age - john_age = 46 := by
sorry

end johns_age_difference_l3165_316577


namespace train_overtake_time_l3165_316581

/-- The time it takes for a train to overtake a motorbike -/
theorem train_overtake_time (train_speed motorbike_speed train_length : ℝ) 
  (h1 : train_speed = 100)
  (h2 : motorbike_speed = 64)
  (h3 : train_length = 850.068) : 
  (train_length / ((train_speed - motorbike_speed) * (1000 / 3600))) = 85.0068 := by
  sorry

end train_overtake_time_l3165_316581


namespace count_rational_coefficient_terms_l3165_316584

/-- The number of terms with rational coefficients in the expansion of (x⁴√2 + y⁵√3)^1200 -/
def rational_coefficient_terms : ℕ :=
  let n : ℕ := 1200
  let f (k : ℕ) : Bool := k % 4 = 0 ∧ (n - k) % 5 = 0
  (List.range (n + 1)).filter f |>.length

theorem count_rational_coefficient_terms : 
  rational_coefficient_terms = 61 := by sorry

end count_rational_coefficient_terms_l3165_316584


namespace min_overlap_percentage_l3165_316588

theorem min_overlap_percentage (math_pref science_pref : ℝ) 
  (h1 : math_pref = 0.90)
  (h2 : science_pref = 0.85) :
  let overlap := math_pref + science_pref - 1
  overlap ≥ 0.75 ∧ 
  ∀ x, x ≥ 0 ∧ x < overlap → 
    ∃ total_pref, total_pref ≤ 1 ∧ 
      total_pref = math_pref + science_pref - x :=
by sorry

end min_overlap_percentage_l3165_316588


namespace geometric_sequence_property_l3165_316530

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_product : a 1 * a 5 = 16) : 
  a 3 = 4 ∨ a 3 = -4 := by
sorry

end geometric_sequence_property_l3165_316530


namespace arc_minutes_to_degrees_l3165_316563

theorem arc_minutes_to_degrees :
  ∀ (arc_minutes : ℝ) (degrees : ℝ),
  (arc_minutes = 1200) →
  (degrees = 20) →
  (arc_minutes * (1 / 60) = degrees) :=
by
  sorry

end arc_minutes_to_degrees_l3165_316563


namespace fourth_square_dots_l3165_316576

/-- The number of dots in the nth square of the series -/
def dots_in_square (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else dots_in_square (n - 1) + 4 * n

theorem fourth_square_dots :
  dots_in_square 4 = 37 := by
  sorry

end fourth_square_dots_l3165_316576


namespace problem_solution_l3165_316553

/-- The number of problems completed given the rate and time -/
def problems_completed (p t : ℕ) : ℕ := p * t

/-- The condition that my friend's completion matches mine -/
def friend_completion_matches (p t : ℕ) : Prop :=
  p * t = (2 * p - 6) * (t - 3)

theorem problem_solution (p t : ℕ) 
  (h1 : p > 15) 
  (h2 : t > 3)
  (h3 : friend_completion_matches p t) :
  problems_completed p t = 216 := by
  sorry

end problem_solution_l3165_316553


namespace cross_section_area_l3165_316525

/-- Regular hexagonal pyramid with square lateral sides -/
structure HexagonalPyramid where
  /-- Side length of the base -/
  a : ℝ
  /-- Assumption that a is positive -/
  a_pos : 0 < a

/-- Cross-section of the hexagonal pyramid -/
def cross_section (pyramid : HexagonalPyramid) : Set (Fin 3 → ℝ) :=
  sorry

/-- Area of a set in ℝ² -/
noncomputable def area (s : Set (Fin 3 → ℝ)) : ℝ :=
  sorry

/-- Theorem: The area of the cross-section is 3a² -/
theorem cross_section_area (pyramid : HexagonalPyramid) :
    area (cross_section pyramid) = 3 * pyramid.a ^ 2 := by
  sorry

end cross_section_area_l3165_316525


namespace binomial_133_133_l3165_316561

theorem binomial_133_133 : Nat.choose 133 133 = 1 := by
  sorry

end binomial_133_133_l3165_316561


namespace f_max_min_on_interval_l3165_316591

def f (x : ℝ) := 4 * x - x^3

theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (0 : ℝ) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (0 : ℝ) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (0 : ℝ) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (0 : ℝ) 2, f x = min) ∧
    max = 16 * Real.sqrt 3 / 9 ∧
    min = 0 :=
by sorry

end f_max_min_on_interval_l3165_316591


namespace cylinder_cone_volume_relation_l3165_316502

/-- The volume of a cylinder with the same base and height as a cone is 3 times the volume of the cone -/
theorem cylinder_cone_volume_relation (Vcone : ℝ) (Vcylinder : ℝ) :
  Vcone > 0 → Vcylinder = 3 * Vcone := by
  sorry

end cylinder_cone_volume_relation_l3165_316502


namespace candy_mixture_cost_l3165_316510

/-- Proves that the desired cost per pound of a candy mixture is $6.00 -/
theorem candy_mixture_cost
  (weight_expensive : ℝ)
  (price_expensive : ℝ)
  (weight_cheap : ℝ)
  (price_cheap : ℝ)
  (h1 : weight_expensive = 25)
  (h2 : price_expensive = 8)
  (h3 : weight_cheap = 50)
  (h4 : price_cheap = 5) :
  (weight_expensive * price_expensive + weight_cheap * price_cheap) /
  (weight_expensive + weight_cheap) = 6 := by
  sorry

end candy_mixture_cost_l3165_316510


namespace a_range_l3165_316565

/-- The inequality holds for all positive real x -/
def inequality_holds (a : ℝ) : Prop :=
  ∀ x > 0, a * Real.log (a * x) ≤ Real.exp x

/-- The theorem stating the range of a given the inequality -/
theorem a_range (a : ℝ) (h : inequality_holds a) : 0 < a ∧ a ≤ Real.exp 1 :=
sorry

end a_range_l3165_316565


namespace line_passes_through_P_x_coordinate_range_l3165_316517

-- Define the line l
def line_l (θ : ℝ) (x y : ℝ) : Prop :=
  (Real.cos θ)^2 * x + Real.cos (2*θ) * y - 1 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the point P
def point_P : ℝ × ℝ := (2, -1)

-- Theorem 1: Line l always passes through point P
theorem line_passes_through_P :
  ∀ θ : ℝ, line_l θ (point_P.1) (point_P.2) :=
sorry

-- Define the range for x-coordinate of M
def x_range (x : ℝ) : Prop :=
  (2 - Real.sqrt 5) / 2 ≤ x ∧ x ≤ 4/5

-- Theorem 2: The x-coordinate of M is in the specified range
theorem x_coordinate_range :
  ∀ θ x y xm : ℝ,
  line_l θ x y →
  circle_C x y →
  -- Additional conditions for point M would be defined here
  x_range xm :=
sorry

end line_passes_through_P_x_coordinate_range_l3165_316517


namespace x_value_theorem_l3165_316585

theorem x_value_theorem (x y z a b c : ℝ) 
  (h1 : x * y / (x + y) = a)
  (h2 : x * z / (x + z) = b)
  (h3 : y * z / (y + z) = c)
  (h4 : a ≠ 0)
  (h5 : b ≠ 0)
  (h6 : c ≠ 0)
  (h7 : x + y + z = a * b * c) :
  x = 2 * a * b * c / (a * b + b * c + a * c) := by
  sorry

end x_value_theorem_l3165_316585


namespace fraction_zero_implies_x_equals_one_l3165_316560

theorem fraction_zero_implies_x_equals_one (x : ℝ) : 
  (x - 1) / (2 - x) = 0 → x = 1 := by
  sorry

end fraction_zero_implies_x_equals_one_l3165_316560


namespace work_completion_proof_l3165_316541

/-- Represents the time taken to complete the work -/
def total_days : ℕ := 11

/-- Represents the rate at which person a completes the work -/
def rate_a : ℚ := 1 / 24

/-- Represents the rate at which person b completes the work -/
def rate_b : ℚ := 1 / 30

/-- Represents the rate at which person c completes the work -/
def rate_c : ℚ := 1 / 40

/-- Represents the days c left before completion of work -/
def days_c_left : ℕ := 4

theorem work_completion_proof :
  ∃ (x : ℕ), x = days_c_left ∧
  (rate_a + rate_b + rate_c) * (total_days - x : ℚ) + (rate_a + rate_b) * x = 1 :=
by sorry

end work_completion_proof_l3165_316541


namespace no_constant_term_in_expansion_l3165_316567

theorem no_constant_term_in_expansion :
  ∀ k : ℕ, k ≤ 12 →
    (12 - k : ℚ) / 2 - 2 * k ≠ 0 :=
by
  sorry

end no_constant_term_in_expansion_l3165_316567


namespace fruit_vendor_problem_l3165_316513

-- Define the parameters
def total_boxes : ℕ := 60
def strawberry_price : ℕ := 60
def apple_price : ℕ := 40
def total_spent : ℕ := 3100
def profit_strawberry_A : ℕ := 15
def profit_apple_A : ℕ := 20
def profit_strawberry_B : ℕ := 12
def profit_apple_B : ℕ := 16
def profit_A : ℕ := 600

-- Define the theorem
theorem fruit_vendor_problem :
  ∃ (strawberry_boxes apple_boxes : ℕ),
    strawberry_boxes + apple_boxes = total_boxes ∧
    strawberry_boxes * strawberry_price + apple_boxes * apple_price = total_spent ∧
    strawberry_boxes = 35 ∧
    apple_boxes = 25 ∧
    (∃ (a b : ℕ),
      a + b ≤ total_boxes ∧
      a * profit_strawberry_A + b * profit_apple_A = profit_A ∧
      (strawberry_boxes - a) * profit_strawberry_B + (apple_boxes - b) * profit_apple_B = 340 ∧
      (a + b = 52 ∨ a + b = 53)) :=
sorry

end fruit_vendor_problem_l3165_316513


namespace steering_wheel_translational_on_straight_road_l3165_316575

/-- A road is considered straight if it has no curves or turns. -/
def is_straight_road (road : Type) : Prop := sorry

/-- A motion is translational if it involves no rotation. -/
def is_translational_motion (motion : Type) : Prop := sorry

/-- The steering wheel motion when driving on a given road. -/
def steering_wheel_motion (road : Type) : Type := sorry

/-- Theorem: The steering wheel motion is translational when driving on a straight road. -/
theorem steering_wheel_translational_on_straight_road (road : Type) :
  is_straight_road road → is_translational_motion (steering_wheel_motion road) := by sorry

end steering_wheel_translational_on_straight_road_l3165_316575


namespace min_value_x_plus_y_l3165_316551

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (1 / x) + (9 / y) = 1) : 
  x + y ≥ 16 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (1 / x) + (9 / y) = 1 ∧ x + y = 16 :=
sorry

end min_value_x_plus_y_l3165_316551


namespace sarah_meal_options_l3165_316554

/-- The number of distinct meals Sarah can order -/
def total_meals (main_courses sides drinks desserts : ℕ) : ℕ :=
  main_courses * sides * drinks * desserts

/-- Theorem stating that Sarah can order 48 distinct meals -/
theorem sarah_meal_options : total_meals 4 3 2 2 = 48 := by
  sorry

end sarah_meal_options_l3165_316554


namespace trig_identity_l3165_316579

theorem trig_identity (α : ℝ) : 
  4.3 * Real.sin (4 * α) - Real.sin (5 * α) - Real.sin (6 * α) + Real.sin (7 * α) = 
  -4 * Real.sin (α / 2) * Real.sin α * Real.sin (11 * α / 2) := by
  sorry

end trig_identity_l3165_316579


namespace pencil_weight_l3165_316540

theorem pencil_weight (total_weight : ℝ) (case_weight : ℝ) (num_pencils : ℕ) 
  (h1 : total_weight = 11.14)
  (h2 : case_weight = 0.5)
  (h3 : num_pencils = 14) :
  (total_weight - case_weight) / num_pencils = 0.76 := by
sorry

end pencil_weight_l3165_316540


namespace inequality_properties_l3165_316536

theorem inequality_properties (a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < c) :
  (a * c < b * c) ∧ (a + b < b + c) ∧ (c / a > c / b) := by
  sorry

end inequality_properties_l3165_316536


namespace ceiling_sum_sqrt_l3165_316524

theorem ceiling_sum_sqrt : ⌈Real.sqrt 3⌉ + ⌈Real.sqrt 33⌉ + ⌈Real.sqrt 333⌉ = 27 := by
  sorry

end ceiling_sum_sqrt_l3165_316524


namespace sum_of_max_and_min_is_eight_l3165_316543

-- Define the function f(x) = x + 2
def f (x : ℝ) : ℝ := x + 2

-- State the theorem
theorem sum_of_max_and_min_is_eight :
  let a : ℝ := 0
  let b : ℝ := 4
  (∀ x ∈ Set.Icc a b, f x ≤ f b) ∧ 
  (∀ x ∈ Set.Icc a b, f a ≤ f x) →
  f a + f b = 8 := by
  sorry

end sum_of_max_and_min_is_eight_l3165_316543


namespace fold_five_cut_once_l3165_316512

/-- The number of segments created by folding a rope n times and then cutting it once -/
def rope_segments (n : ℕ) : ℕ :=
  2^n + 1

/-- Theorem: Folding a rope 5 times and cutting it once results in 33 segments -/
theorem fold_five_cut_once : rope_segments 5 = 33 := by
  sorry

end fold_five_cut_once_l3165_316512


namespace work_completion_time_l3165_316568

/-- 
Given:
- Person A can complete a work in 30 days
- Person A and B together complete 0.38888888888888884 part of the work in 7 days

Prove:
Person B can complete the work alone in 45 days
-/
theorem work_completion_time (a b : ℝ) (h1 : a = 30) 
  (h2 : 7 * (1 / a + 1 / b) = 0.38888888888888884) : b = 45 := by
  sorry

end work_completion_time_l3165_316568


namespace sum_of_squares_fourth_degree_equation_l3165_316557

-- Part 1
theorem sum_of_squares (x y : ℝ) :
  (x^2 + y^2 - 4) * (x^2 + y^2 + 2) = 7 → x^2 + y^2 = 5 :=
by sorry

-- Part 2
theorem fourth_degree_equation (x : ℝ) :
  x^4 - 6*x^2 + 8 = 0 → x = Real.sqrt 2 ∨ x = -Real.sqrt 2 ∨ x = 2 ∨ x = -2 :=
by sorry

end sum_of_squares_fourth_degree_equation_l3165_316557


namespace initial_amount_proof_l3165_316598

/-- Prove that given an initial amount P, after applying 5% interest for the first year
    and 6% interest for the second year, if the final amount is 5565, then P must be 5000. -/
theorem initial_amount_proof (P : ℝ) : 
  P * (1 + 0.05) * (1 + 0.06) = 5565 → P = 5000 := by
  sorry

end initial_amount_proof_l3165_316598


namespace right_angled_triangle_l3165_316546

theorem right_angled_triangle (A B C : ℝ) (h : Real.sin A + Real.sin B = Real.sin C * (Real.cos A + Real.cos B)) :
  Real.cos C = 0 :=
sorry

end right_angled_triangle_l3165_316546


namespace mike_initial_cards_l3165_316592

/-- The number of baseball cards Mike has initially -/
def initial_cards : ℕ := sorry

/-- The number of baseball cards Sam gave to Mike -/
def cards_from_sam : ℕ := 13

/-- The total number of baseball cards Mike has after receiving cards from Sam -/
def total_cards : ℕ := 100

/-- Theorem stating that Mike initially had 87 baseball cards -/
theorem mike_initial_cards : initial_cards = 87 := by
  sorry

end mike_initial_cards_l3165_316592


namespace geometric_sequence_l3165_316562

theorem geometric_sequence (a : ℕ → ℝ) :
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) →
  a 1 + a 3 = 10 →
  a 2 + a 4 = 5 →
  a 8 = 1 / 16 :=
sorry

end geometric_sequence_l3165_316562


namespace sum_abcd_equals_negative_eleven_l3165_316586

theorem sum_abcd_equals_negative_eleven (a b c d : ℚ) 
  (h : 2*a + 3 = 2*b + 4 ∧ 2*a + 3 = 2*c + 5 ∧ 2*a + 3 = 2*d + 6 ∧ 2*a + 3 = a + b + c + d + 10) : 
  a + b + c + d = -11 := by
sorry

end sum_abcd_equals_negative_eleven_l3165_316586


namespace max_value_sqrt_sum_l3165_316593

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y : ℝ, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end max_value_sqrt_sum_l3165_316593


namespace julian_borrows_eight_l3165_316566

/-- The amount Julian borrows -/
def additional_borrowed (current_debt new_debt : ℕ) : ℕ :=
  new_debt - current_debt

/-- Proof that Julian borrows 8 dollars -/
theorem julian_borrows_eight :
  let current_debt := 20
  let new_debt := 28
  additional_borrowed current_debt new_debt = 8 := by
  sorry

end julian_borrows_eight_l3165_316566


namespace sugar_box_surface_area_l3165_316580

theorem sugar_box_surface_area :
  ∀ (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 →
    a * b * c = 280 →
    2 * (a * b + b * c + c * a) = 262 :=
by
  sorry

end sugar_box_surface_area_l3165_316580


namespace center_numbers_l3165_316595

/-- Represents a 4x4 grid of integers -/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Checks if two positions in the grid share an edge -/
def sharesEdge (p1 p2 : Fin 4 × Fin 4) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

/-- Checks if a grid satisfies the conditions of the problem -/
def validGrid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 17) ∧
  (∀ n : ℕ, n ∈ Finset.range 16 → ∃ p1 p2, g p1.1 p1.2 = n ∧ g p2.1 p2.2 = n + 1 ∧ sharesEdge p1 p2) ∧
  (g 0 0 + g 0 3 + g 3 0 + g 3 3 = 34)

/-- The center 2x2 grid -/
def centerGrid (g : Grid) : Finset ℕ :=
  {g 1 1, g 1 2, g 2 1, g 2 2}

theorem center_numbers (g : Grid) (h : validGrid g) :
  centerGrid g = {9, 10, 11, 12} :=
sorry

end center_numbers_l3165_316595


namespace spring_length_increase_l3165_316583

def spring_length (x : ℝ) : ℝ :=
  20 + 0.5 * x

theorem spring_length_increase (x₁ x₂ : ℝ) 
  (h1 : 0 ≤ x₁ ∧ x₁ < 5) 
  (h2 : x₂ = x₁ + 1) : 
  spring_length x₂ - spring_length x₁ = 0.5 := by
  sorry

end spring_length_increase_l3165_316583


namespace pages_read_later_l3165_316542

/-- Given that Jake initially read some pages of a book and then read more later,
    prove that the number of pages he read later is the difference between
    the total pages read and the initial pages read. -/
theorem pages_read_later (initial_pages total_pages pages_read_later : ℕ) :
  initial_pages + pages_read_later = total_pages →
  pages_read_later = total_pages - initial_pages := by
  sorry

#check pages_read_later

end pages_read_later_l3165_316542


namespace f_intersects_x_axis_l3165_316504

-- Define the function f(x) = x + 1
def f (x : ℝ) : ℝ := x + 1

-- Theorem stating that f intersects the x-axis
theorem f_intersects_x_axis : ∃ x : ℝ, f x = 0 := by
  sorry

end f_intersects_x_axis_l3165_316504


namespace min_value_of_M_l3165_316516

theorem min_value_of_M (a b : ℕ+) : 
  ∃ (m : ℕ), m = 3 * a.val ^ 2 - a.val * b.val ^ 2 - 2 * b.val - 4 ∧ 
  m ≥ 2 ∧ 
  ∀ (k : ℕ), k = 3 * a.val ^ 2 - a.val * b.val ^ 2 - 2 * b.val - 4 → k ≥ 2 :=
by sorry

end min_value_of_M_l3165_316516


namespace money_difference_l3165_316527

/-- Given that Bob has $60, Phil has 1/3 of Bob's amount, and Jenna has twice Phil's amount,
    prove that the difference between Bob's and Jenna's amounts is $20. -/
theorem money_difference (bob_amount : ℕ) (phil_amount : ℕ) (jenna_amount : ℕ)
    (h1 : bob_amount = 60)
    (h2 : phil_amount = bob_amount / 3)
    (h3 : jenna_amount = 2 * phil_amount) :
    bob_amount - jenna_amount = 20 := by
  sorry

end money_difference_l3165_316527


namespace price_per_working_game_l3165_316548

def total_games : ℕ := 10
def non_working_games : ℕ := 2
def total_earnings : ℕ := 32

theorem price_per_working_game :
  (total_earnings : ℚ) / (total_games - non_working_games) = 4 := by
  sorry

end price_per_working_game_l3165_316548


namespace dance_club_average_age_l3165_316537

theorem dance_club_average_age 
  (num_females : Nat) 
  (num_males : Nat) 
  (avg_age_females : ℝ) 
  (avg_age_males : ℝ) 
  (h1 : num_females = 12)
  (h2 : num_males = 18)
  (h3 : avg_age_females = 25)
  (h4 : avg_age_males = 40) :
  let total_people := num_females + num_males
  let total_age := num_females * avg_age_females + num_males * avg_age_males
  total_age / total_people = 34 := by
  sorry

end dance_club_average_age_l3165_316537


namespace old_pump_fills_in_600_seconds_l3165_316582

/-- Represents the time (in seconds) taken by the old pump to fill the trough alone. -/
def old_pump_time : ℝ := 600

/-- Represents the time (in seconds) taken by the new pump to fill the trough alone. -/
def new_pump_time : ℝ := 200

/-- Represents the time (in seconds) taken by both pumps working together to fill the trough. -/
def combined_time : ℝ := 150

/-- 
Proves that the old pump takes 600 seconds to fill the trough alone, given the times for the new pump
and both pumps working together.
-/
theorem old_pump_fills_in_600_seconds :
  (1 / old_pump_time) + (1 / new_pump_time) = (1 / combined_time) :=
by sorry

end old_pump_fills_in_600_seconds_l3165_316582


namespace number_division_remainder_l3165_316509

theorem number_division_remainder (N : ℕ) : 
  (N / 5 = 5 ∧ N % 5 = 0) → N % 11 = 3 := by
sorry

end number_division_remainder_l3165_316509


namespace solve_equation_l3165_316501

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 4
def g (x : ℝ) : ℝ := x^2 - 2

-- State the theorem
theorem solve_equation (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 18) : a = Real.sqrt (Real.sqrt 14 + 2) := by
  sorry

end solve_equation_l3165_316501


namespace handshakes_five_people_l3165_316508

/-- The number of handshakes between n people, where each person shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

/-- There are 5 people in the room. -/
def num_people : ℕ := 5

theorem handshakes_five_people : handshakes num_people = 10 := by
  sorry

end handshakes_five_people_l3165_316508


namespace alloy_combination_theorem_l3165_316545

/-- Represents the composition of an alloy --/
structure AlloyComposition where
  copper : ℝ
  zinc : ℝ

/-- The first alloy composition --/
def firstAlloy : AlloyComposition :=
  { copper := 2, zinc := 1 }

/-- The second alloy composition --/
def secondAlloy : AlloyComposition :=
  { copper := 1, zinc := 5 }

/-- Combines two alloys in a given ratio --/
def combineAlloys (a1 a2 : AlloyComposition) (r1 r2 : ℝ) : AlloyComposition :=
  { copper := r1 * a1.copper + r2 * a2.copper
  , zinc := r1 * a1.zinc + r2 * a2.zinc }

/-- The theorem to be proved --/
theorem alloy_combination_theorem :
  let combinedAlloy := combineAlloys firstAlloy secondAlloy 1 2
  combinedAlloy.zinc = 2 * combinedAlloy.copper := by
  sorry

end alloy_combination_theorem_l3165_316545


namespace star_power_equality_l3165_316507

/-- The k-th smallest positive integer not in X -/
def f (X : Finset ℕ) (k : ℕ) : ℕ := sorry

/-- The operation * for finite sets of positive integers -/
def starOp (X Y : Finset ℕ) : Finset ℕ :=
  X ∪ (Y.image (f X))

/-- Repeated application of starOp -/
def starPower (X : Finset ℕ) : ℕ → Finset ℕ
  | 0 => X
  | n + 1 => starOp X (starPower X n)

theorem star_power_equality {A B : Finset ℕ} (ha : A.card > 0) (hb : B.card > 0)
    (h : starOp A B = starOp B A) :
    starPower A B.card = starPower B A.card := by
  sorry

end star_power_equality_l3165_316507


namespace race_speed_ratio_l3165_316526

/-- Given two racers a and b, where a's speed is some multiple of b's speed,
    and a gives b a 0.2 part of the race length as a head start resulting in a dead heat,
    prove that the ratio of a's speed to b's speed is 5:4 -/
theorem race_speed_ratio (L : ℝ) (v_a v_b : ℝ) (h1 : v_a > 0) (h2 : v_b > 0) 
    (h3 : ∃ k : ℝ, v_a = k * v_b) 
    (h4 : L / v_a = (0.8 * L) / v_b) : 
  v_a / v_b = 5 / 4 := by
sorry

end race_speed_ratio_l3165_316526


namespace power_sum_equals_two_l3165_316596

theorem power_sum_equals_two : (-1)^2 + (1/3)^0 = 2 := by
  sorry

end power_sum_equals_two_l3165_316596


namespace min_value_a_l3165_316528

theorem min_value_a (p : Prop) (h : ¬∀ x > 0, a < x + 1/x) : 
  ∃ a : ℝ, (∀ b : ℝ, (∃ x > 0, b ≥ x + 1/x) → a ≤ b) ∧ (∃ x > 0, a ≥ x + 1/x) ∧ a = 2 :=
sorry

end min_value_a_l3165_316528


namespace rhombus_fourth_vertex_area_l3165_316531

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its four vertices -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- A rhombus defined by its four vertices -/
structure Rhombus where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- Check if a point is on a line segment defined by two other points -/
def isOnSegment (p : Point) (a : Point) (b : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = a.x + t * (b.x - a.x) ∧ p.y = a.y + t * (b.y - a.y)

/-- The set of all possible locations for the fourth vertex of the rhombus -/
def fourthVertexSet (sq : Square) : Set Point :=
  { p : Point | ∃ r : Rhombus,
    isOnSegment r.P sq.A sq.B ∧
    isOnSegment r.Q sq.B sq.C ∧
    isOnSegment r.R sq.A sq.D ∧
    r.S = p }

/-- The area of a set of points in 2D space -/
noncomputable def area (s : Set Point) : ℝ := sorry

/-- The theorem to be proved -/
theorem rhombus_fourth_vertex_area (sq : Square) :
  sq.A = Point.mk 0 0 →
  sq.B = Point.mk 1 0 →
  sq.C = Point.mk 1 1 →
  sq.D = Point.mk 0 1 →
  area (fourthVertexSet sq) = 7/3 :=
by
  sorry

end rhombus_fourth_vertex_area_l3165_316531


namespace equation_solutions_l3165_316520

theorem equation_solutions :
  (∃ (x : ℝ), (1/2) * (2*x - 5)^2 - 2 = 0 ↔ x = 7/2 ∨ x = 3/2) ∧
  (∃ (x : ℝ), x^2 - 4*x - 4 = 0 ↔ x = 2 + 2*Real.sqrt 2 ∨ x = 2 - 2*Real.sqrt 2) :=
by sorry

end equation_solutions_l3165_316520


namespace decimal_2_09_to_percentage_l3165_316549

/-- Converts a decimal number to a percentage -/
def decimal_to_percentage (x : ℝ) : ℝ := 100 * x

theorem decimal_2_09_to_percentage :
  decimal_to_percentage 2.09 = 209 := by sorry

end decimal_2_09_to_percentage_l3165_316549


namespace carnival_wait_time_l3165_316556

/-- Proves that the wait time for the roller coaster is 30 minutes given the carnival conditions --/
theorem carnival_wait_time (total_time : ℕ) (tilt_a_whirl_wait : ℕ) (giant_slide_wait : ℕ)
  (roller_coaster_rides : ℕ) (tilt_a_whirl_rides : ℕ) (giant_slide_rides : ℕ) :
  total_time = 4 * 60 ∧
  tilt_a_whirl_wait = 60 ∧
  giant_slide_wait = 15 ∧
  roller_coaster_rides = 4 ∧
  tilt_a_whirl_rides = 1 ∧
  giant_slide_rides = 4 →
  ∃ (roller_coaster_wait : ℕ),
    roller_coaster_wait = 30 ∧
    total_time = roller_coaster_rides * roller_coaster_wait +
                 tilt_a_whirl_rides * tilt_a_whirl_wait +
                 giant_slide_rides * giant_slide_wait :=
by
  sorry

end carnival_wait_time_l3165_316556


namespace geometric_sequence_common_ratio_l3165_316587

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_sum : ∀ n : ℕ, S n = (a 0) * (1 - (a 1 / a 0)^n) / (1 - (a 1 / a 0))) 
  (h_a2 : a 2 = 1/4) 
  (h_S3 : S 3 = 7/8) :
  (a 1 / a 0 = 2) ∨ (a 1 / a 0 = 1/2) :=
sorry

end geometric_sequence_common_ratio_l3165_316587


namespace relay_team_orders_l3165_316514

/-- The number of permutations of n elements -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of different orders for a relay team of 6 runners,
    where one specific runner is fixed to run the last lap -/
def relay_orders : ℕ := factorial 5

theorem relay_team_orders :
  relay_orders = 120 := by sorry

end relay_team_orders_l3165_316514


namespace inequality_proof_l3165_316599

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (3 * x^2 - x) / (1 + x^2) + (3 * y^2 - y) / (1 + y^2) + (3 * z^2 - z) / (1 + z^2) ≥ 0 := by
  sorry

end inequality_proof_l3165_316599


namespace monic_cubic_polynomial_theorem_l3165_316539

-- Define a monic cubic polynomial with real coefficients
def monicCubicPolynomial (a b c : ℝ) : ℝ → ℂ :=
  fun x => (x : ℂ)^3 + a * (x : ℂ)^2 + b * (x : ℂ) + c

-- State the theorem
theorem monic_cubic_polynomial_theorem (a b c : ℝ) :
  let q := monicCubicPolynomial a b c
  (q (3 - 2*I) = 0 ∧ q 0 = -108) →
  a = -(186/13) ∧ b = 1836/13 ∧ c = -108 :=
by sorry

end monic_cubic_polynomial_theorem_l3165_316539


namespace tangent_line_implies_b_equals_3_l3165_316552

/-- A line y = kx + 1 is tangent to the curve y = x^3 + ax + b at the point (1, 3) -/
def is_tangent (k a b : ℝ) : Prop :=
  -- The point (1, 3) lies on both the line and the curve
  3 = k * 1 + 1 ∧ 3 = 1^3 + a * 1 + b ∧
  -- The slopes of the line and the curve are equal at (1, 3)
  k = 3 * 1^2 + a

theorem tangent_line_implies_b_equals_3 (k a b : ℝ) :
  is_tangent k a b → b = 3 := by
  sorry

#check tangent_line_implies_b_equals_3

end tangent_line_implies_b_equals_3_l3165_316552


namespace square_midpoint_dot_product_l3165_316570

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let CD := (D.1 - C.1, D.2 - C.2)
  let DA := (A.1 - D.1, A.2 - D.2)
  (AB.1 * AB.1 + AB.2 * AB.2 = 4) ∧
  (BC.1 * BC.1 + BC.2 * BC.2 = 4) ∧
  (CD.1 * CD.1 + CD.2 * CD.2 = 4) ∧
  (DA.1 * DA.1 + DA.2 * DA.2 = 4) ∧
  (AB.1 * BC.1 + AB.2 * BC.2 = 0) ∧
  (BC.1 * CD.1 + BC.2 * CD.2 = 0) ∧
  (CD.1 * DA.1 + CD.2 * DA.2 = 0) ∧
  (DA.1 * AB.1 + DA.2 * AB.2 = 0)

-- Define the midpoint E of CD
def Midpoint (C D E : ℝ × ℝ) : Prop :=
  E.1 = (C.1 + D.1) / 2 ∧ E.2 = (C.2 + D.2) / 2

-- Define the dot product of two vectors
def DotProduct (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Theorem statement
theorem square_midpoint_dot_product 
  (A B C D E : ℝ × ℝ) 
  (h1 : Square A B C D) 
  (h2 : Midpoint C D E) : 
  DotProduct (E.1 - A.1, E.2 - A.2) (D.1 - B.1, D.2 - B.2) = 2 := by
  sorry

end square_midpoint_dot_product_l3165_316570


namespace no_integer_solution_l3165_316532

theorem no_integer_solution :
  ¬ ∃ (a b c : ℤ), a^2 + b^2 - 8*c = 6 := by
  sorry

end no_integer_solution_l3165_316532


namespace solve_equation_l3165_316550

theorem solve_equation (x : ℝ) (h : x + 1 = 2) : x = 1 := by
  sorry

end solve_equation_l3165_316550


namespace eyes_saw_airplane_l3165_316571

/-- Given 200 students and 3/4 of them looking up, prove that 300 eyes saw the airplane. -/
theorem eyes_saw_airplane (total_students : ℕ) (fraction_looked_up : ℚ) (h1 : total_students = 200) (h2 : fraction_looked_up = 3/4) :
  (fraction_looked_up * total_students : ℚ).num * 2 = 300 := by
  sorry

end eyes_saw_airplane_l3165_316571


namespace swim_time_calculation_l3165_316574

/-- 
Given a person's swimming speed in still water, the speed of the water current,
and the time taken to swim with the current for a certain distance,
calculate the time taken to swim back against the current for the same distance.
-/
theorem swim_time_calculation (still_speed water_speed with_current_time : ℝ) 
  (still_speed_pos : still_speed > 0)
  (water_speed_pos : water_speed > 0)
  (with_current_time_pos : with_current_time > 0)
  (h_still_speed : still_speed = 16)
  (h_water_speed : water_speed = 8)
  (h_with_current_time : with_current_time = 1.5) :
  let against_current_speed := still_speed - water_speed
  let with_current_speed := still_speed + water_speed
  let distance := with_current_speed * with_current_time
  let against_current_time := distance / against_current_speed
  against_current_time = 4.5 := by
sorry

end swim_time_calculation_l3165_316574


namespace largest_gcd_sum_1008_l3165_316515

theorem largest_gcd_sum_1008 :
  ∃ (a b : ℕ+), a + b = 1008 ∧ 
  ∀ (c d : ℕ+), c + d = 1008 → Nat.gcd a b ≥ Nat.gcd c d ∧
  Nat.gcd a b = 504 :=
sorry

end largest_gcd_sum_1008_l3165_316515


namespace peanut_mixture_relation_l3165_316529

/-- Represents the peanut mixture problem -/
def PeanutMixture (S T : ℝ) : Prop :=
  let virginiaWeight : ℝ := 10
  let virginiaCost : ℝ := 3.50
  let spanishCost : ℝ := 3.00
  let texanCost : ℝ := 4.00
  let mixtureCost : ℝ := 3.60
  let totalWeight : ℝ := virginiaWeight + S + T
  let totalCost : ℝ := virginiaWeight * virginiaCost + S * spanishCost + T * texanCost
  (totalCost / totalWeight = mixtureCost) ∧ (0.40 * T - 0.60 * S = 1)

/-- Theorem stating the relationship between Spanish and Texan peanut weights -/
theorem peanut_mixture_relation (S T : ℝ) :
  PeanutMixture S T → (0.40 * T - 0.60 * S = 1) :=
by
  sorry

end peanut_mixture_relation_l3165_316529


namespace extreme_point_of_g_l3165_316533

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x - 1

noncomputable def g (x : ℝ) : ℝ := x * (f 1 x) + (1/2) * x^2 + 2 * x

def has_unique_extreme_point (h : ℝ → ℝ) (m : ℤ) : Prop :=
  ∃! (x : ℝ), m < x ∧ x < m + 1 ∧ 
  (∀ y ∈ Set.Ioo m (m + 1), h y ≤ h x) ∨ 
  (∀ y ∈ Set.Ioo m (m + 1), h y ≥ h x)

theorem extreme_point_of_g :
  ∃ m : ℤ, has_unique_extreme_point g m → m = 0 ∨ m = 3 :=
sorry

end extreme_point_of_g_l3165_316533


namespace pancake_stacks_sold_l3165_316511

/-- The number of stacks of pancakes sold at a fundraiser -/
def pancake_stacks : ℕ := sorry

/-- The cost of one stack of pancakes in dollars -/
def pancake_cost : ℚ := 4

/-- The cost of one slice of bacon in dollars -/
def bacon_cost : ℚ := 2

/-- The number of bacon slices sold -/
def bacon_slices : ℕ := 90

/-- The total revenue from the fundraiser in dollars -/
def total_revenue : ℚ := 420

/-- Theorem stating that the number of pancake stacks sold is 60 -/
theorem pancake_stacks_sold : pancake_stacks = 60 := by sorry

end pancake_stacks_sold_l3165_316511


namespace transform_G_to_cup_l3165_316518

-- Define the set of shapes (including letters and symbols)
def Shape : Type := String

-- Define the transformations
def T₁ (s : Shape) : Shape := sorry
def T₂ (s : Shape) : Shape := sorry

-- Define the composition of transformations
def T (s : Shape) : Shape := T₂ (T₁ s)

-- State the theorem
theorem transform_G_to_cup (h1 : T₁ "R" = "y") (h2 : T₂ "y" = "B")
                           (h3 : T₁ "L" = "⌝") (h4 : T₂ "⌝" = "Γ") :
  T "G" = "∪" := by sorry

end transform_G_to_cup_l3165_316518


namespace power_nap_duration_l3165_316589

/-- Converts hours to minutes -/
def hours_to_minutes (hours : ℚ) : ℚ := hours * 60

/-- Represents one fourth of an hour -/
def quarter_hour : ℚ := 1 / 4

theorem power_nap_duration :
  hours_to_minutes quarter_hour = 15 := by sorry

end power_nap_duration_l3165_316589


namespace max_sum_of_factors_l3165_316594

theorem max_sum_of_factors (clubsuit heartsuit : ℕ) : 
  clubsuit * heartsuit = 48 → 
  Even clubsuit → 
  ∃ (a b : ℕ), a * b = 48 ∧ Even a ∧ a + b ≤ clubsuit + heartsuit ∧ a + b = 26 :=
by sorry

end max_sum_of_factors_l3165_316594


namespace square_area_equals_triangle_perimeter_l3165_316558

/-- Given a right-angled triangle with sides 6 cm and 8 cm, 
    a square with the same perimeter as this triangle has an area of 36 cm². -/
theorem square_area_equals_triangle_perimeter : 
  ∃ (triangle_hypotenuse : ℝ) (square_side : ℝ),
    triangle_hypotenuse^2 = 6^2 + 8^2 ∧ 
    6 + 8 + triangle_hypotenuse = 4 * square_side ∧
    square_side^2 = 36 := by
  sorry

end square_area_equals_triangle_perimeter_l3165_316558


namespace problem_distribution_l3165_316547

def distribute_problems (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial k

theorem problem_distribution :
  distribute_problems 9 7 = 181440 := by
  sorry

end problem_distribution_l3165_316547


namespace four_numbers_product_2002_sum_less_40_l3165_316534

theorem four_numbers_product_2002_sum_less_40 (a b c d : ℕ+) :
  a * b * c * d = 2002 ∧ a + b + c + d < 40 →
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
  (a = 2 ∧ b = 7 ∧ c = 13 ∧ d = 11) ∨ (a = 1 ∧ b = 14 ∧ c = 13 ∧ d = 11) ∨
  (a = 2 ∧ b = 11 ∧ c = 7 ∧ d = 13) ∨ (a = 1 ∧ b = 11 ∧ c = 14 ∧ d = 13) ∨
  (a = 2 ∧ b = 11 ∧ c = 13 ∧ d = 7) ∨ (a = 1 ∧ b = 11 ∧ c = 13 ∧ d = 14) ∨
  (a = 2 ∧ b = 13 ∧ c = 7 ∧ d = 11) ∨ (a = 1 ∧ b = 13 ∧ c = 14 ∧ d = 11) ∨
  (a = 2 ∧ b = 13 ∧ c = 11 ∧ d = 7) ∨ (a = 1 ∧ b = 13 ∧ c = 11 ∧ d = 14) :=
by sorry

end four_numbers_product_2002_sum_less_40_l3165_316534


namespace problem_statement_l3165_316572

theorem problem_statement :
  (∀ (a b m : ℝ), (a * m^2 < b * m^2 → a < b) ∧ ¬(a < b → a * m^2 < b * m^2)) ∧
  (¬(∀ x : ℝ, x^3 - x^2 - 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 - 1 > 0)) ∧
  (∀ (p q : Prop), ¬p → ¬q → ¬(p ∧ q)) ∧
  ¬(∀ x : ℝ, (x ≠ 1 ∨ x ≠ -1 → x^2 ≠ 1) ↔ (x^2 = 1 → x = 1 ∨ x = -1)) :=
by sorry

end problem_statement_l3165_316572


namespace special_function_properties_l3165_316503

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x - f y)

/-- The theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  f 2 = 0 ∧ ∃! v : ℝ, f 2 = v :=
sorry

end special_function_properties_l3165_316503


namespace border_collie_catch_up_time_l3165_316506

/-- The time it takes for a border collie to catch up to a thrown ball -/
theorem border_collie_catch_up_time
  (ball_speed : ℝ)
  (ball_flight_time : ℝ)
  (dog_speed : ℝ)
  (h1 : ball_speed = 20)
  (h2 : ball_flight_time = 8)
  (h3 : dog_speed = 5) :
  (ball_speed * ball_flight_time) / dog_speed = 32 := by
  sorry

end border_collie_catch_up_time_l3165_316506


namespace dereks_car_dog_ratio_l3165_316522

/-- Represents Derek's possessions at different ages --/
structure DereksPossessions where
  dogs_at_6 : ℕ
  cars_at_6 : ℕ
  dogs_at_16 : ℕ
  cars_at_16 : ℕ

/-- Theorem stating the ratio of cars to dogs when Derek is 16 --/
theorem dereks_car_dog_ratio (d : DereksPossessions) 
  (h1 : d.dogs_at_6 = 90)
  (h2 : d.dogs_at_6 = 3 * d.cars_at_6)
  (h3 : d.dogs_at_16 = 120)
  (h4 : d.cars_at_16 = d.cars_at_6 + 210)
  : d.cars_at_16 / d.dogs_at_16 = 2 := by
  sorry

#check dereks_car_dog_ratio

end dereks_car_dog_ratio_l3165_316522


namespace fraction_equality_l3165_316544

theorem fraction_equality (a b c x : ℝ) (hx : x = a / b) (hc : c ≠ 0) (hb : b ≠ 0) (ha : a ≠ c * b) :
  (a + c * b) / (a - c * b) = (x + c) / (x - c) := by
  sorry

end fraction_equality_l3165_316544


namespace quadratic_roots_exist_l3165_316505

theorem quadratic_roots_exist (a b c : ℝ) (h : a * c < 0) : 
  ∃ x : ℝ, a * x^2 + b * x + c = 0 := by
  sorry

end quadratic_roots_exist_l3165_316505


namespace kevin_initial_phones_l3165_316500

/-- The number of phones Kevin had at the beginning of the day -/
def initial_phones : ℕ := 33

/-- The number of phones Kevin repaired by afternoon -/
def repaired_phones : ℕ := 3

/-- The number of phones dropped off by a client -/
def new_phones : ℕ := 6

/-- The number of phones each person (Kevin and coworker) will repair -/
def phones_per_person : ℕ := 9

theorem kevin_initial_phones :
  initial_phones = 33 ∧
  repaired_phones = 3 ∧
  new_phones = 6 ∧
  phones_per_person = 9 →
  initial_phones + new_phones - repaired_phones = 2 * phones_per_person :=
by sorry

end kevin_initial_phones_l3165_316500


namespace justin_run_time_l3165_316555

/-- Represents Justin's running speed and route information -/
structure RunningInfo where
  flat_speed : ℚ  -- blocks per minute on flat ground
  uphill_speed : ℚ  -- blocks per minute uphill
  total_distance : ℕ  -- total blocks to home
  uphill_distance : ℕ  -- blocks that are uphill

/-- Calculates the total time Justin needs to run home -/
def time_to_run_home (info : RunningInfo) : ℚ :=
  let flat_distance := info.total_distance - info.uphill_distance
  let flat_time := flat_distance / info.flat_speed
  let uphill_time := info.uphill_distance / info.uphill_speed
  flat_time + uphill_time

/-- Theorem stating that Justin will take 13 minutes to run home -/
theorem justin_run_time :
  let info : RunningInfo := {
    flat_speed := 1,  -- 2 blocks / 2 minutes
    uphill_speed := 2/3,  -- 2 blocks / 3 minutes
    total_distance := 10,
    uphill_distance := 6
  }
  time_to_run_home info = 13 := by
  sorry


end justin_run_time_l3165_316555


namespace newsletter_cost_l3165_316535

def newsletter_cost_exists : Prop :=
  ∃ x : ℝ, 
    (14 * x < 16) ∧ 
    (19 * x > 21) ∧ 
    (∀ y : ℝ, (14 * y < 16) ∧ (19 * y > 21) → |x - 1.11| ≤ |y - 1.11|)

theorem newsletter_cost : newsletter_cost_exists := by sorry

end newsletter_cost_l3165_316535


namespace initial_deadlift_weight_l3165_316519

def initial_squat : ℝ := 700
def initial_bench : ℝ := 400
def squat_loss_percentage : ℝ := 30
def deadlift_loss : ℝ := 200
def new_total : ℝ := 1490

theorem initial_deadlift_weight :
  ∃ (initial_deadlift : ℝ),
    initial_deadlift - deadlift_loss +
    initial_bench +
    initial_squat * (1 - squat_loss_percentage / 100) = new_total ∧
    initial_deadlift = 800 := by
  sorry

end initial_deadlift_weight_l3165_316519


namespace triangle_angle_not_all_greater_than_60_l3165_316564

theorem triangle_angle_not_all_greater_than_60 :
  ∀ (a b c : ℝ), 
  (a + b + c = 180) →  -- Sum of angles in a triangle is 180°
  (a > 0) → (b > 0) → (c > 0) →  -- All angles are positive
  ¬(a > 60 ∧ b > 60 ∧ c > 60) :=
by sorry

end triangle_angle_not_all_greater_than_60_l3165_316564


namespace min_value_when_a_is_one_f_greater_than_x_iff_a_negative_l3165_316597

noncomputable section

variable (x : ℝ) (a : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := x^2 - Real.log x - a*x

theorem min_value_when_a_is_one :
  ∃ (min : ℝ), min = 0 ∧ ∀ x > 0, f x 1 ≥ min := by sorry

theorem f_greater_than_x_iff_a_negative :
  (∀ x > 0, f x a > x) ↔ a < 0 := by sorry

end min_value_when_a_is_one_f_greater_than_x_iff_a_negative_l3165_316597


namespace sphere_surface_area_l3165_316538

theorem sphere_surface_area (d : ℝ) (h : d = 4) : 
  4 * Real.pi * (d / 2)^2 = 16 * Real.pi := by
  sorry

end sphere_surface_area_l3165_316538


namespace feeding_and_trapping_sets_l3165_316569

/-- A set is a feeding set for a sequence if every open subinterval of the set contains infinitely many terms of the sequence. -/
def IsFeeder (s : Set ℝ) (seq : ℕ → ℝ) : Prop :=
  ∀ a b, a < b → a ∈ s → b ∈ s → Set.Infinite {n : ℕ | seq n ∈ Set.Ioo a b}

/-- A set is a trapping set for a sequence if no infinite subset of the sequence remains outside the set. -/
def IsTrap (s : Set ℝ) (seq : ℕ → ℝ) : Prop :=
  Set.Finite {n : ℕ | seq n ∉ s}

theorem feeding_and_trapping_sets :
  (∃ seq : ℕ → ℝ, IsFeeder (Set.Icc 0 1) seq ∧ IsFeeder (Set.Icc 2 3) seq) ∧
  (¬ ∃ seq : ℕ → ℝ, IsTrap (Set.Icc 0 1) seq ∧ IsTrap (Set.Icc 2 3) seq) :=
sorry

end feeding_and_trapping_sets_l3165_316569


namespace simplify_and_evaluate_expression_l3165_316578

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = -2) :
  (1 + 1 / (a - 1)) / (2 * a / (a^2 - 1)) = -1/2 := by
  sorry

end simplify_and_evaluate_expression_l3165_316578


namespace angle_measure_of_special_triangle_l3165_316523

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (C : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (angle_range : 0 < C ∧ C < π)
  (side_relation : a^2 + b^2 + a*b = c^2)

-- Theorem statement
theorem angle_measure_of_special_triangle (t : Triangle) : t.C = 2*π/3 := by
  sorry

end angle_measure_of_special_triangle_l3165_316523


namespace chess_tournament_games_l3165_316573

/-- Calculate the number of games in a chess tournament --/
def tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- The number of players in the tournament --/
def num_players : ℕ := 12

/-- The number of times each pair of players compete --/
def games_per_pair : ℕ := 2

theorem chess_tournament_games :
  tournament_games num_players * games_per_pair = 264 := by
  sorry

end chess_tournament_games_l3165_316573


namespace gcd_8_factorial_6_factorial_squared_l3165_316559

theorem gcd_8_factorial_6_factorial_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_8_factorial_6_factorial_squared_l3165_316559


namespace complex_purely_imaginary_l3165_316590

theorem complex_purely_imaginary (x : ℝ) :
  let z : ℂ := Complex.mk (x^2 + x - 2) (x^2 + 3*x + 2)
  (z.re = 0 ∧ z.im ≠ 0) → x = 1 := by
sorry

end complex_purely_imaginary_l3165_316590
