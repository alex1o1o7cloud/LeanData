import Mathlib

namespace inequality_proof_l136_13687

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a / (1 + b + c) + b / (1 + c + a) + c / (1 + a + b) ≥ 
       a * b / (1 + a + b) + b * c / (1 + b + c) + c * a / (1 + c + a)) :
  (a^2 + b^2 + c^2) / (a * b + b * c + c * a) + a + b + c + 2 ≥ 
  2 * (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a)) := by
  sorry

end inequality_proof_l136_13687


namespace connie_watch_purchase_l136_13609

/-- The additional amount Connie needs to buy a watch -/
def additional_amount (savings : ℕ) (watch_cost : ℕ) : ℕ :=
  watch_cost - savings

/-- Theorem: Given Connie's savings and the watch cost, prove the additional amount needed -/
theorem connie_watch_purchase (connie_savings : ℕ) (watch_price : ℕ) 
  (h1 : connie_savings = 39)
  (h2 : watch_price = 55) :
  additional_amount connie_savings watch_price = 16 := by
  sorry

end connie_watch_purchase_l136_13609


namespace basketball_shot_expectation_l136_13618

theorem basketball_shot_expectation 
  (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (h_sum : a + b + c = 1) 
  (h_expect : 3 * a + 2 * b = 2) : 
  (∀ x y, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 2 → (2 / x + 1 / (3 * y)) ≥ 16 / 3) ∧ 
  (∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 2 ∧ 2 / x + 1 / (3 * y) = 16 / 3) :=
by sorry

end basketball_shot_expectation_l136_13618


namespace square_exterior_points_l136_13638

/-- Given a square ABCD with side length 15 and exterior points G and H, prove that GH^2 = 1126 - 1030√2 -/
theorem square_exterior_points (A B C D G H K : ℝ × ℝ) : 
  let side_length : ℝ := 15
  let bg : ℝ := 7
  let dh : ℝ := 7
  let ag : ℝ := 13
  let ch : ℝ := 13
  let dk : ℝ := 8
  -- Square ABCD conditions
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = side_length^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = side_length^2 ∧
  (D.1 - C.1)^2 + (D.2 - C.2)^2 = side_length^2 ∧
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = side_length^2 ∧
  -- Exterior points conditions
  (G.1 - B.1)^2 + (G.2 - B.2)^2 = bg^2 ∧
  (H.1 - D.1)^2 + (H.2 - D.2)^2 = dh^2 ∧
  (G.1 - A.1)^2 + (G.2 - A.2)^2 = ag^2 ∧
  (H.1 - C.1)^2 + (H.2 - C.2)^2 = ch^2 ∧
  -- K on extension of BD
  (K.1 - D.1)^2 + (K.2 - D.2)^2 = dk^2 ∧
  (K.1 - B.1) / (D.1 - B.1) = (K.2 - B.2) / (D.2 - B.2) ∧
  (K.1 - D.1) / (B.1 - D.1) > 1 →
  (G.1 - H.1)^2 + (G.2 - H.2)^2 = 1126 - 1030 * Real.sqrt 2 :=
by sorry

end square_exterior_points_l136_13638


namespace boat_current_rate_l136_13670

/-- Proves that given a boat with a speed of 15 km/hr in still water, 
    traveling 3.6 km downstream in 12 minutes, the rate of the current is 3 km/hr. -/
theorem boat_current_rate 
  (boat_speed : ℝ) 
  (distance_downstream : ℝ) 
  (time_minutes : ℝ) 
  (h1 : boat_speed = 15) 
  (h2 : distance_downstream = 3.6) 
  (h3 : time_minutes = 12) : 
  ∃ (current_rate : ℝ), current_rate = 3 ∧ 
    distance_downstream = (boat_speed + current_rate) * (time_minutes / 60) := by
  sorry

end boat_current_rate_l136_13670


namespace tuesday_to_monday_ratio_l136_13675

/-- Represents the number of crates of eggs sold on each day of the week --/
structure EggSales where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- Theorem stating the ratio of Tuesday's sales to Monday's sales --/
theorem tuesday_to_monday_ratio (sales : EggSales) : 
  sales.monday = 5 ∧ 
  sales.wednesday = sales.tuesday - 2 ∧ 
  sales.thursday = sales.tuesday / 2 ∧
  sales.monday + sales.tuesday + sales.wednesday + sales.thursday = 28 →
  sales.tuesday = 2 * sales.monday := by
  sorry

#check tuesday_to_monday_ratio

end tuesday_to_monday_ratio_l136_13675


namespace remainder_mod_48_l136_13653

theorem remainder_mod_48 (x : ℤ) 
  (h1 : (2 + x) % (2^3) = 2^3 % (2^3))
  (h2 : (4 + x) % (4^3) = 4^2 % (4^3))
  (h3 : (6 + x) % (6^3) = 6^2 % (6^3)) :
  x % 48 = 6 := by
sorry

end remainder_mod_48_l136_13653


namespace boxes_with_neither_l136_13648

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (erasers : ℕ) (both : ℕ)
  (h_total : total = 15)
  (h_markers : markers = 10)
  (h_erasers : erasers = 5)
  (h_both : both = 4) :
  total - (markers + erasers - both) = 4 := by
  sorry

end boxes_with_neither_l136_13648


namespace alternate_interior_angles_relationship_l136_13604

-- Define a structure for a line
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a structure for an angle
structure Angle :=
  (measure : ℝ)

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to create alternate interior angles
def alternate_interior_angles (l1 l2 l3 : Line) : (Angle × Angle) :=
  sorry

-- Theorem statement
theorem alternate_interior_angles_relationship 
  (l1 l2 l3 : Line) : 
  ¬ (∀ (a1 a2 : Angle), 
    (a1, a2) = alternate_interior_angles l1 l2 l3 → 
    a1.measure = a2.measure ∨ 
    a1.measure ≠ a2.measure) :=
sorry

end alternate_interior_angles_relationship_l136_13604


namespace cube_sum_over_product_equals_thirteen_l136_13694

theorem cube_sum_over_product_equals_thirteen
  (a b c : ℂ)
  (nonzero_a : a ≠ 0)
  (nonzero_b : b ≠ 0)
  (nonzero_c : c ≠ 0)
  (sum_equals_ten : a + b + c = 10)
  (squared_diff_sum : (a - b)^2 + (a - c)^2 + (b - c)^2 = 2*a*b*c) :
  (a^3 + b^3 + c^3) / (a*b*c) = 13 := by
sorry

end cube_sum_over_product_equals_thirteen_l136_13694


namespace cone_surface_area_l136_13637

/-- The surface area of a cone with base radius 1 and slant height 3 is 4π. -/
theorem cone_surface_area : 
  let r : ℝ := 1  -- base radius
  let l : ℝ := 3  -- slant height
  let S : ℝ := π * r * (r + l)  -- surface area formula
  S = 4 * π :=
by sorry

end cone_surface_area_l136_13637


namespace solution_set_of_even_decreasing_function_l136_13623

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 1) * (a * x + b)

-- State the theorem
theorem solution_set_of_even_decreasing_function (a b : ℝ) :
  (∀ x, f a b x = f a b (-x)) →  -- f is even
  (∀ x y, 0 < x ∧ x < y → f a b y < f a b x) →  -- f is decreasing on (0, +∞)
  {x : ℝ | f a b (2 - x) < 0} = {x : ℝ | x < 1 ∨ x > 3} :=
by sorry


end solution_set_of_even_decreasing_function_l136_13623


namespace tims_children_treats_l136_13692

/-- The total number of treats Tim's children get while trick-or-treating -/
def total_treats (num_children : ℕ) (hours_out : ℕ) (houses_per_hour : ℕ) (treats_per_kid : ℕ) : ℕ :=
  num_children * hours_out * houses_per_hour * treats_per_kid

/-- Theorem stating that Tim's children get 180 treats in total -/
theorem tims_children_treats : 
  total_treats 3 4 5 3 = 180 := by
  sorry

end tims_children_treats_l136_13692


namespace candy_bar_cost_l136_13662

/-- The cost of a candy bar given the conditions -/
theorem candy_bar_cost : 
  ∀ (candy_cost chocolate_cost : ℝ),
  candy_cost + chocolate_cost = 3 →
  candy_cost = chocolate_cost + 3 →
  candy_cost = 3 :=
by
  sorry

end candy_bar_cost_l136_13662


namespace remaining_apple_pies_l136_13601

/-- Proves the number of apple pies remaining with Cooper --/
theorem remaining_apple_pies (pies_per_day : ℕ) (days : ℕ) (pies_eaten : ℕ) : 
  pies_per_day = 7 → days = 12 → pies_eaten = 50 → 
  pies_per_day * days - pies_eaten = 34 := by
  sorry

#check remaining_apple_pies

end remaining_apple_pies_l136_13601


namespace trig_expression_equals_three_l136_13659

theorem trig_expression_equals_three :
  let sin_60 : ℝ := Real.sqrt 3 / 2
  let tan_45 : ℝ := 1
  let tan_60 : ℝ := Real.sqrt 3
  ∀ (sin_25 cos_25 : ℝ), 
    sin_25^2 + cos_25^2 = 1 →
    sin_25^2 + 2 * sin_60 + tan_45 - tan_60 + cos_25^2 = 3 := by
  sorry

end trig_expression_equals_three_l136_13659


namespace bananas_per_truck_l136_13645

-- Define the given quantities
def total_apples : ℝ := 132.6
def apples_per_truck : ℝ := 13.26
def total_bananas : ℝ := 6.4

-- Define the theorem
theorem bananas_per_truck :
  (total_bananas / (total_apples / apples_per_truck)) = 0.64 := by
  sorry

end bananas_per_truck_l136_13645


namespace intersection_of_M_and_N_l136_13614

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x : ℝ | -5 < x ∧ x < 5}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -3 < x ∧ x < 5} := by
  sorry

end intersection_of_M_and_N_l136_13614


namespace sally_cards_bought_l136_13691

def cards_bought (initial : ℕ) (received : ℕ) (final : ℕ) : ℕ :=
  final - (initial + received)

theorem sally_cards_bought :
  cards_bought 27 41 88 = 20 := by
  sorry

end sally_cards_bought_l136_13691


namespace y_squared_plus_7y_plus_12_bounds_l136_13696

theorem y_squared_plus_7y_plus_12_bounds (y : ℝ) (h : y^2 - 7*y + 12 < 0) : 
  42 < y^2 + 7*y + 12 ∧ y^2 + 7*y + 12 < 56 := by
sorry

end y_squared_plus_7y_plus_12_bounds_l136_13696


namespace weight_of_second_new_student_l136_13602

theorem weight_of_second_new_student
  (initial_students : Nat)
  (initial_avg_weight : ℝ)
  (new_students : Nat)
  (new_avg_weight : ℝ)
  (weight_of_first_new_student : ℝ)
  (h1 : initial_students = 29)
  (h2 : initial_avg_weight = 28)
  (h3 : new_students = initial_students + 2)
  (h4 : new_avg_weight = 27.5)
  (h5 : weight_of_first_new_student = 25)
  : ∃ (weight_of_second_new_student : ℝ),
    weight_of_second_new_student = 20.5 ∧
    (initial_students : ℝ) * initial_avg_weight + weight_of_first_new_student + weight_of_second_new_student =
    (new_students : ℝ) * new_avg_weight :=
by sorry

end weight_of_second_new_student_l136_13602


namespace circle_and_line_theorem_l136_13634

-- Define the given points and circles
def M : ℝ × ℝ := (2, -2)
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 3
def circle_2 (x y : ℝ) : Prop := x^2 + y^2 + 3*x = 0

-- Define the resulting circle and line
def result_circle (x y : ℝ) : Prop := 3*x^2 + 3*y^2 - 5*x - 14 = 0
def line_AB (x y : ℝ) : Prop := 2*x - 2*y = 3

theorem circle_and_line_theorem :
  -- (1) The result_circle passes through M and intersects with circle_O and circle_2
  (result_circle M.1 M.2) ∧
  (∃ x y : ℝ, result_circle x y ∧ circle_O x y) ∧
  (∃ x y : ℝ, result_circle x y ∧ circle_2 x y) ∧
  -- (2) line_AB is tangent to circle_O at two points
  (∃ A B : ℝ × ℝ, 
    A ≠ B ∧
    circle_O A.1 A.2 ∧ circle_O B.1 B.2 ∧
    line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧
    (∀ x y : ℝ, line_AB x y → circle_O x y → (x, y) = A ∨ (x, y) = B)) :=
by sorry

end circle_and_line_theorem_l136_13634


namespace four_composition_odd_l136_13603

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem four_composition_odd (f : ℝ → ℝ) (h : IsOdd f) : IsOdd (fun x ↦ f (f (f (f x)))) := by
  sorry

end four_composition_odd_l136_13603


namespace scientific_notation_of_2590000_l136_13690

theorem scientific_notation_of_2590000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 2590000 = a * (10 : ℝ) ^ n ∧ a = 2.59 ∧ n = 6 := by
  sorry

end scientific_notation_of_2590000_l136_13690


namespace bead_division_problem_l136_13606

/-- The number of equal parts into which the beads were divided -/
def n : ℕ := sorry

/-- The total number of beads -/
def total_beads : ℕ := 23 + 16

/-- The number of beads in each part after division but before removal -/
def beads_per_part : ℚ := total_beads / n

/-- The number of beads in each part after removal but before doubling -/
def beads_after_removal : ℚ := beads_per_part - 10

/-- The final number of beads in each part after doubling -/
def final_beads : ℕ := 6

theorem bead_division_problem :
  2 * beads_after_removal = final_beads ∧ n = 3 := by sorry

end bead_division_problem_l136_13606


namespace min_value_product_min_value_product_achieved_l136_13666

theorem min_value_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 8) :
  (2 * a + b) * (a + 3 * c) * (b * c + 2) ≥ 128 :=
by sorry

theorem min_value_product_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 8 ∧
    (2 * a + b) * (a + 3 * c) * (b * c + 2) < 128 + ε :=
by sorry

end min_value_product_min_value_product_achieved_l136_13666


namespace chess_club_committees_l136_13612

/-- The number of teams in the chess club -/
def num_teams : ℕ := 5

/-- The number of members in each team -/
def team_size : ℕ := 6

/-- The number of members selected from the host team -/
def host_selection : ℕ := 4

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 3

/-- The total number of members in the organizing committee -/
def committee_size : ℕ := 16

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem chess_club_committees :
  (num_teams * choose team_size host_selection * (choose team_size non_host_selection) ^ (num_teams - 1)) = 12000000 := by
  sorry

end chess_club_committees_l136_13612


namespace triangle_side_length_l136_13677

/-- Given a triangle ABC where angle A is 6 degrees, angle C is 75 degrees, 
    and side BC has length √3, prove that the length of side AC 
    is equal to (√3 * sin 6°) / sin 45° -/
theorem triangle_side_length (A B C : ℝ) (AC BC : ℝ) : 
  A = 6 * π / 180 →  -- Convert 6° to radians
  C = 75 * π / 180 →  -- Convert 75° to radians
  BC = Real.sqrt 3 →
  AC = (Real.sqrt 3 * Real.sin (6 * π / 180)) / Real.sin (45 * π / 180) :=
by sorry

end triangle_side_length_l136_13677


namespace total_seeds_equals_685_l136_13608

/- Morning plantings -/
def mike_morning_tomato : ℕ := 50
def mike_morning_pepper : ℕ := 30
def ted_morning_tomato : ℕ := 2 * mike_morning_tomato
def ted_morning_pepper : ℕ := mike_morning_pepper / 2
def sarah_morning_tomato : ℕ := mike_morning_tomato + 30
def sarah_morning_pepper : ℕ := mike_morning_pepper + 30

/- Afternoon plantings -/
def mike_afternoon_tomato : ℕ := 60
def mike_afternoon_pepper : ℕ := 40
def ted_afternoon_tomato : ℕ := mike_afternoon_tomato - 20
def ted_afternoon_pepper : ℕ := mike_afternoon_pepper
def sarah_afternoon_tomato : ℕ := sarah_morning_tomato + 20
def sarah_afternoon_pepper : ℕ := sarah_morning_pepper + 10

/- Total seeds planted -/
def total_seeds : ℕ := 
  mike_morning_tomato + mike_morning_pepper + 
  ted_morning_tomato + ted_morning_pepper + 
  sarah_morning_tomato + sarah_morning_pepper + 
  mike_afternoon_tomato + mike_afternoon_pepper + 
  ted_afternoon_tomato + ted_afternoon_pepper + 
  sarah_afternoon_tomato + sarah_afternoon_pepper

theorem total_seeds_equals_685 : total_seeds = 685 := by
  sorry

end total_seeds_equals_685_l136_13608


namespace fib_pisano_period_l136_13697

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Pisano period for modulus 10 -/
def pisano_period : ℕ := 60

theorem fib_pisano_period :
  (∀ n : ℕ, n > 0 → fib n % 10 = fib (n + pisano_period) % 10) ∧
  (∀ t : ℕ, t > 0 → t < pisano_period →
    ∃ n : ℕ, n > 0 ∧ fib n % 10 ≠ fib (n + t) % 10) := by
  sorry

end fib_pisano_period_l136_13697


namespace apples_remaining_after_three_days_l136_13644

/-- Represents the number of apples picked from a tree on a given day -/
structure Picking where
  treeA : ℕ
  treeB : ℕ
  treeC : ℕ

/-- Calculates the total number of apples remaining after three days of picking -/
def applesRemaining (initialA initialB initialC : ℕ) (day1 day2 day3 : Picking) : ℕ :=
  (initialA - day1.treeA - day2.treeA - day3.treeA) +
  (initialB - day1.treeB - day2.treeB - day3.treeB) +
  (initialC - day1.treeC - day2.treeC - day3.treeC)

theorem apples_remaining_after_three_days :
  let initialA := 200
  let initialB := 250
  let initialC := 300
  let day1 := Picking.mk 40 25 0
  let day2 := Picking.mk 0 80 38
  let day3 := Picking.mk 60 0 40
  applesRemaining initialA initialB initialC day1 day2 day3 = 467 := by
  sorry


end apples_remaining_after_three_days_l136_13644


namespace least_integer_with_2023_divisors_l136_13640

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Check if n is divisible by d -/
def is_divisible_by (n d : ℕ) : Prop := sorry

theorem least_integer_with_2023_divisors :
  ∃ (m k : ℕ),
    (num_divisors (m * 6^k) = 2023) ∧
    (¬ is_divisible_by m 6) ∧
    (∀ n : ℕ, num_divisors n = 2023 → n ≥ m * 6^k) ∧
    m = 9216 ∧
    k = 6 :=
sorry

end least_integer_with_2023_divisors_l136_13640


namespace complete_square_sum_l136_13635

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 + 6*x - 9 = 0 ↔ (x + b)^2 = c) → 
  b + c = 21 := by
sorry

end complete_square_sum_l136_13635


namespace smallest_number_with_remainder_one_l136_13683

theorem smallest_number_with_remainder_one : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 7 = 1 ∧ 
  n % 11 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 7 = 1 ∧ m % 11 = 1 → m ≥ n) ∧ 
  n = 78 := by
sorry

end smallest_number_with_remainder_one_l136_13683


namespace tomato_planting_theorem_l136_13613

def tomato_planting (total_seedlings : ℕ) (remi_day1 : ℕ) : Prop :=
  let remi_day2 := 2 * remi_day1
  let father_day3 := 3 * remi_day2
  let father_day4 := 4 * remi_day2
  let sister_day5 := remi_day1
  let sister_day6 := 5 * remi_day1
  let remi_total := remi_day1 + remi_day2
  let sister_total := sister_day5 + sister_day6
  let father_total := total_seedlings - remi_total - sister_total
  (remi_total = 600) ∧
  (sister_total = 1200) ∧
  (father_total = 6400) ∧
  (remi_total + sister_total + father_total = total_seedlings)

theorem tomato_planting_theorem :
  tomato_planting 8200 200 :=
by
  sorry

end tomato_planting_theorem_l136_13613


namespace complex_power_difference_zero_l136_13665

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference_zero :
  (1 + i)^24 - (1 - i)^24 = 0 :=
by
  sorry

end complex_power_difference_zero_l136_13665


namespace endpoint_coordinate_product_l136_13668

/-- Given a line segment with midpoint (3, -5) and one endpoint at (7, -1),
    the product of the coordinates of the other endpoint is 9. -/
theorem endpoint_coordinate_product : 
  ∀ (x y : ℝ), 
  (3 = (x + 7) / 2) →  -- midpoint x-coordinate
  (-5 = (y + (-1)) / 2) →  -- midpoint y-coordinate
  x * y = 9 := by
sorry

end endpoint_coordinate_product_l136_13668


namespace bob_water_percentage_l136_13628

def corn_water_usage : ℝ := 20
def cotton_water_usage : ℝ := 80
def bean_water_usage : ℝ := 2 * corn_water_usage

def bob_corn_acres : ℝ := 3
def bob_cotton_acres : ℝ := 9
def bob_bean_acres : ℝ := 12

def brenda_corn_acres : ℝ := 6
def brenda_cotton_acres : ℝ := 7
def brenda_bean_acres : ℝ := 14

def bernie_corn_acres : ℝ := 2
def bernie_cotton_acres : ℝ := 12

def bob_water_usage : ℝ := 
  bob_corn_acres * corn_water_usage + 
  bob_cotton_acres * cotton_water_usage + 
  bob_bean_acres * bean_water_usage

def total_water_usage : ℝ := 
  (bob_corn_acres + brenda_corn_acres + bernie_corn_acres) * corn_water_usage +
  (bob_cotton_acres + brenda_cotton_acres + bernie_cotton_acres) * cotton_water_usage +
  (bob_bean_acres + brenda_bean_acres) * bean_water_usage

theorem bob_water_percentage : 
  bob_water_usage / total_water_usage * 100 = 36 := by sorry

end bob_water_percentage_l136_13628


namespace light_distance_500_years_l136_13607

theorem light_distance_500_years :
  let distance_one_year : ℝ := 5870000000000
  let years : ℕ := 500
  (distance_one_year * years : ℝ) = 2.935 * (10 ^ 15) :=
by sorry

end light_distance_500_years_l136_13607


namespace three_topping_pizzas_l136_13615

theorem three_topping_pizzas (n : ℕ) (k : ℕ) : n = 7 → k = 3 → Nat.choose n k = 35 := by
  sorry

end three_topping_pizzas_l136_13615


namespace alaya_fruit_salads_l136_13639

def fruit_salad_problem (alaya : ℕ) (angel : ℕ) : Prop :=
  angel = 2 * alaya ∧ alaya + angel = 600

theorem alaya_fruit_salads :
  ∃ (alaya : ℕ), fruit_salad_problem alaya (2 * alaya) ∧ alaya = 200 :=
by
  sorry

end alaya_fruit_salads_l136_13639


namespace inequality_equivalence_l136_13695

theorem inequality_equivalence (x y : ℝ) : 
  Real.sqrt (x^2 - 2*x*y) > Real.sqrt (1 - y^2) ↔ 
  ((x - y > 1 ∧ -1 < y ∧ y < 1) ∨ (x - y < -1 ∧ -1 < y ∧ y < 1)) :=
by sorry

end inequality_equivalence_l136_13695


namespace astronaut_selection_probability_l136_13647

theorem astronaut_selection_probability : 
  let total_astronauts : ℕ := 4
  let male_astronauts : ℕ := 2
  let female_astronauts : ℕ := 2
  let selected_astronauts : ℕ := 2

  -- Probability of selecting one male and one female
  (Nat.choose male_astronauts 1 * Nat.choose female_astronauts 1 : ℚ) / 
  (Nat.choose total_astronauts selected_astronauts) = 2 / 3 :=
by sorry

end astronaut_selection_probability_l136_13647


namespace increasing_function_inequality_l136_13626

theorem increasing_function_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_inequality : f (a^2 - a) > f (2*a^2 - 4*a)) : 
  0 < a ∧ a < 3 := by
sorry

end increasing_function_inequality_l136_13626


namespace ellipse_condition_ellipse_condition_converse_l136_13641

/-- Represents a point in a 2D rectangular coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the equation m(x^2 + y^2 + 2y + 1) = (x - 2y + 3)^2 -/
def ellipseEquation (m : ℝ) (p : Point) : Prop :=
  m * (p.x^2 + p.y^2 + 2*p.y + 1) = (p.x - 2*p.y + 3)^2

/-- Defines what it means for the equation to represent an ellipse -/
def isEllipse (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
    ∀ (p : Point), ellipseEquation m p ↔ 
      (p.x^2 / a^2) + (p.y^2 / b^2) = 1

/-- The main theorem: if the equation represents an ellipse, then m > 5 -/
theorem ellipse_condition (m : ℝ) :
  isEllipse m → m > 5 := by
  sorry

/-- The converse: if m > 5, then the equation represents an ellipse -/
theorem ellipse_condition_converse (m : ℝ) :
  m > 5 → isEllipse m := by
  sorry

end ellipse_condition_ellipse_condition_converse_l136_13641


namespace age_multiple_problem_l136_13643

theorem age_multiple_problem (P_age Q_age : ℝ) (M : ℝ) : 
  P_age + Q_age = 100 →
  Q_age = 37.5 →
  37.5 = M * (Q_age - (P_age - Q_age)) →
  M = 3 := by
sorry

end age_multiple_problem_l136_13643


namespace multiply_by_9999_l136_13651

theorem multiply_by_9999 : ∃! x : ℤ, x * 9999 = 806006795 :=
  by sorry

end multiply_by_9999_l136_13651


namespace special_function_1988_l136_13642

/-- A function from positive integers to positive integers satisfying the given property -/
def special_function (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, f (f m + f n) = m + n

/-- The main theorem stating that any function satisfying the special property maps 1988 to 1988 -/
theorem special_function_1988 (f : ℕ+ → ℕ+) (h : special_function f) : f 1988 = 1988 := by
  sorry

end special_function_1988_l136_13642


namespace function_property_l136_13622

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ p q, f (p + q) = f p * f q) 
  (h2 : f 1 = 3) : 
  (f 1^2 + f 2) / f 1 + 
  (f 2^2 + f 4) / f 3 + 
  (f 3^2 + f 6) / f 5 + 
  (f 4^2 + f 8) / f 7 + 
  (f 5^2 + f 10) / f 9 = 30 := by
  sorry

end function_property_l136_13622


namespace stratified_sampling_boys_l136_13654

theorem stratified_sampling_boys (total_boys : ℕ) (total_girls : ℕ) (sample_size : ℕ) :
  total_boys = 48 →
  total_girls = 36 →
  sample_size = 21 →
  (total_boys * sample_size) / (total_boys + total_girls) = 12 :=
by sorry

end stratified_sampling_boys_l136_13654


namespace pure_imaginary_implies_m_eq_neg_four_l136_13693

def z (m : ℝ) : ℂ := Complex.mk (m^2 + 2*m - 8) (m - 2)

theorem pure_imaginary_implies_m_eq_neg_four :
  ∀ m : ℝ, (z m).re = 0 ∧ (z m).im ≠ 0 → m = -4 := by
  sorry

end pure_imaginary_implies_m_eq_neg_four_l136_13693


namespace half_page_ad_cost_l136_13671

/-- Calculates the cost of a half-page advertisement in Math Magazine -/
theorem half_page_ad_cost : 
  let page_length : ℝ := 9
  let page_width : ℝ := 12
  let cost_per_square_inch : ℝ := 8
  let full_page_area : ℝ := page_length * page_width
  let half_page_area : ℝ := full_page_area / 2
  let total_cost : ℝ := half_page_area * cost_per_square_inch
  total_cost = 432 :=
by sorry

end half_page_ad_cost_l136_13671


namespace ball_count_theorem_l136_13663

theorem ball_count_theorem (total : ℕ) (red_freq black_freq : ℚ) :
  total = 120 →
  red_freq = 15 / 100 →
  black_freq = 45 / 100 →
  ∃ (red black white : ℕ),
    red = (total : ℚ) * red_freq ∧
    black = (total : ℚ) * black_freq ∧
    white = total - red - black ∧
    white = 48 := by
  sorry

end ball_count_theorem_l136_13663


namespace cube_sum_divisibility_l136_13619

theorem cube_sum_divisibility (x y z : ℤ) :
  7 ∣ (x^3 + y^3 + z^3) → 7 ∣ (x * y * z) := by
  sorry

end cube_sum_divisibility_l136_13619


namespace triangle_theorem_l136_13633

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Theorem about a specific triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : (t.a + t.b - t.c) * (t.a + t.b + t.c) = t.a * t.b)
  (h2 : t.c = 2 * t.a * Real.cos t.B)
  (h3 : t.b = 2) : 
  t.C = 2 * Real.pi / 3 ∧ 
  (1/2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end triangle_theorem_l136_13633


namespace rectangle_combinations_l136_13658

-- Define the number of horizontal and vertical lines
def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4

-- Define the number of lines needed to form a rectangle
def lines_for_rectangle : ℕ := 2

-- Theorem statement
theorem rectangle_combinations :
  (Nat.choose horizontal_lines lines_for_rectangle) *
  (Nat.choose vertical_lines lines_for_rectangle) = 60 := by
  sorry

end rectangle_combinations_l136_13658


namespace octopus_leg_counts_l136_13656

/-- Represents the possible number of legs an octopus can have -/
inductive LegCount
  | six
  | seven
  | eight

/-- Represents an octopus with a name and a number of legs -/
structure Octopus :=
  (name : String)
  (legs : LegCount)

/-- Determines if an octopus is telling the truth based on its leg count -/
def isTruthful (o : Octopus) : Bool :=
  match o.legs with
  | LegCount.seven => false
  | _ => true

/-- Converts LegCount to a natural number -/
def legCountToNat (lc : LegCount) : Nat :=
  match lc with
  | LegCount.six => 6
  | LegCount.seven => 7
  | LegCount.eight => 8

/-- The main theorem about the octopuses' leg counts -/
theorem octopus_leg_counts (blue green red yellow : Octopus)
  (h1 : blue.name = "Blue" ∧ green.name = "Green" ∧ red.name = "Red" ∧ yellow.name = "Yellow")
  (h2 : (isTruthful blue) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 25))
  (h3 : (isTruthful green) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 26))
  (h4 : (isTruthful red) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 27))
  (h5 : (isTruthful yellow) = (legCountToNat blue.legs + legCountToNat green.legs + legCountToNat red.legs + legCountToNat yellow.legs = 28)) :
  blue.legs = LegCount.seven ∧ green.legs = LegCount.seven ∧ red.legs = LegCount.six ∧ yellow.legs = LegCount.seven :=
sorry

end octopus_leg_counts_l136_13656


namespace trapezoid_area_l136_13605

theorem trapezoid_area (outer_area inner_area : ℝ) (h1 : outer_area = 36) (h2 : inner_area = 4) :
  (outer_area - inner_area) / 3 = 32 / 3 := by
  sorry

end trapezoid_area_l136_13605


namespace math_competition_probabilities_l136_13616

theorem math_competition_probabilities :
  let total_students : ℕ := 6
  let boys : ℕ := 3
  let girls : ℕ := 3
  let selected : ℕ := 2

  let prob_exactly_one_boy : ℚ := 3/5
  let prob_at_least_one_boy : ℚ := 4/5
  let prob_at_most_one_boy : ℚ := 4/5

  (total_students = boys + girls) →
  (prob_exactly_one_boy = 0.6) ∧
  (prob_at_least_one_boy = 0.8) ∧
  (prob_at_most_one_boy = 0.8) :=
by
  sorry

end math_competition_probabilities_l136_13616


namespace product_restoration_l136_13698

theorem product_restoration (P : ℕ) : 
  P = (List.range 11).foldl (· * ·) 1 →
  ∃ (a b c : ℕ), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    P = 399 * 100000 + a * 10000 + 68 * 100 + b * 10 + c →
  P = 39916800 := by sorry

end product_restoration_l136_13698


namespace function_composition_identity_l136_13610

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x + b
  else if x < 3 then 2 * x - 1
  else 10 - 4 * x

theorem function_composition_identity (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 1/2 := by
  sorry

end function_composition_identity_l136_13610


namespace rectangular_hyperbola_foci_distance_l136_13672

/-- The distance between foci of a rectangular hyperbola xy = 4 -/
theorem rectangular_hyperbola_foci_distance : 
  ∀ (x y : ℝ), x * y = 4 → ∃ (f₁ f₂ : ℝ × ℝ), 
    (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 16 :=
by sorry

end rectangular_hyperbola_foci_distance_l136_13672


namespace distribute_seven_balls_four_boxes_l136_13600

/-- Represents the number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute (balls : ℕ) (boxes : ℕ) (min_per_box : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 3 ways to distribute 7 balls into 4 boxes -/
theorem distribute_seven_balls_four_boxes :
  distribute 7 4 1 = 3 := by sorry

end distribute_seven_balls_four_boxes_l136_13600


namespace sqrt_sum_comparison_cubic_vs_quadratic_l136_13611

-- Part 1
theorem sqrt_sum_comparison : Real.sqrt 7 + Real.sqrt 10 > Real.sqrt 3 + Real.sqrt 14 := by
  sorry

-- Part 2
theorem cubic_vs_quadratic (x : ℝ) (h : x > 1) : x^3 > x^2 - x + 1 := by
  sorry

end sqrt_sum_comparison_cubic_vs_quadratic_l136_13611


namespace inequality_proof_l136_13676

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) := by
  sorry

end inequality_proof_l136_13676


namespace alan_cd_cost_l136_13667

/-- The total cost of CDs Alan buys -/
def total_cost (price_avn : ℝ) (num_dark : ℕ) (num_90s : ℕ) : ℝ :=
  let price_dark := 2 * price_avn
  let cost_dark := num_dark * price_dark
  let cost_avn := price_avn
  let cost_others := cost_dark + cost_avn
  let cost_90s := 0.4 * cost_others
  cost_dark + cost_avn + cost_90s

/-- Theorem stating the total cost of Alan's CD purchase -/
theorem alan_cd_cost :
  total_cost 12 2 5 = 84 := by
  sorry

end alan_cd_cost_l136_13667


namespace fried_chicken_dinner_pieces_l136_13680

/-- Represents the number of pieces of chicken in a family-size Fried Chicken Dinner -/
def fried_chicken_pieces : ℕ := sorry

/-- The number of pieces of chicken in a Chicken Pasta order -/
def chicken_pasta_pieces : ℕ := 2

/-- The number of pieces of chicken in a Barbecue Chicken order -/
def barbecue_chicken_pieces : ℕ := 3

/-- The number of Fried Chicken Dinner orders -/
def fried_chicken_orders : ℕ := 2

/-- The number of Chicken Pasta orders -/
def chicken_pasta_orders : ℕ := 6

/-- The number of Barbecue Chicken orders -/
def barbecue_chicken_orders : ℕ := 3

/-- The total number of pieces of chicken needed for all orders -/
def total_chicken_pieces : ℕ := 37

theorem fried_chicken_dinner_pieces :
  fried_chicken_pieces * fried_chicken_orders +
  chicken_pasta_pieces * chicken_pasta_orders +
  barbecue_chicken_pieces * barbecue_chicken_orders =
  total_chicken_pieces ∧
  fried_chicken_pieces = 8 := by sorry

end fried_chicken_dinner_pieces_l136_13680


namespace intersection_A_B_l136_13621

-- Define sets A and B
def A : Set ℝ := {x | x > 3}
def B : Set ℝ := {x | (x - 1) / (x - 4) < 0}

-- State the theorem
theorem intersection_A_B :
  ∀ x : ℝ, x ∈ A ∩ B ↔ 3 < x ∧ x < 4 := by sorry

end intersection_A_B_l136_13621


namespace min_cos_B_angle_A_values_l136_13631

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given conditions for the triangle -/
def TriangleConditions (t : Triangle) : Prop :=
  t.a + t.c = 3 * Real.sqrt 3 ∧ t.b = 3

/-- The minimum value of cos B -/
theorem min_cos_B (t : Triangle) (h : TriangleConditions t) :
    (∀ t' : Triangle, TriangleConditions t' → Real.cos t'.B ≥ Real.cos t.B) →
    Real.cos t.B = 1/3 := by sorry

/-- The possible values of angle A when BA · BC = 3 -/
theorem angle_A_values (t : Triangle) (h : TriangleConditions t) :
    t.a * t.c * Real.cos t.B = 3 →
    t.A = Real.pi / 2 ∨ t.A = Real.pi / 6 := by sorry

end min_cos_B_angle_A_values_l136_13631


namespace total_investment_amount_l136_13688

/-- Prove that the total investment is $8000 given the specified conditions --/
theorem total_investment_amount (total_income : ℝ) (rate1 rate2 : ℝ) (investment1 : ℝ) :
  total_income = 575 →
  rate1 = 0.085 →
  rate2 = 0.064 →
  investment1 = 3000 →
  ∃ (investment2 : ℝ),
    total_income = investment1 * rate1 + investment2 * rate2 ∧
    investment1 + investment2 = 8000 := by
  sorry

end total_investment_amount_l136_13688


namespace ellipse_fixed_point_intersection_l136_13655

/-- Defines an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Defines a line -/
structure Line where
  k : ℝ
  m : ℝ

/-- Theorem statement -/
theorem ellipse_fixed_point_intersection 
  (E : Ellipse) 
  (h_point : E.a^2 + (3/2)^2 / E.b^2 = 1) 
  (h_ecc : (E.a^2 - E.b^2) / E.a^2 = 1/4) 
  (l : Line) 
  (h_intersect : ∃ (M N : ℝ × ℝ), M ≠ N ∧ 
    M.1^2 / E.a^2 + M.2^2 / E.b^2 = 1 ∧
    N.1^2 / E.a^2 + N.2^2 / E.b^2 = 1 ∧
    M.2 = l.k * M.1 + l.m ∧
    N.2 = l.k * N.1 + l.m)
  (h_perp : ∀ (M N : ℝ × ℝ), 
    M.1^2 / E.a^2 + M.2^2 / E.b^2 = 1 →
    N.1^2 / E.a^2 + N.2^2 / E.b^2 = 1 →
    M.2 = l.k * M.1 + l.m →
    N.2 = l.k * N.1 + l.m →
    (M.1 - E.a) * (N.1 - E.a) + M.2 * N.2 = 0) :
  l.k * (2/7) + l.m = 0 :=
sorry

end ellipse_fixed_point_intersection_l136_13655


namespace largest_n_for_quadratic_equation_l136_13652

theorem largest_n_for_quadratic_equation : 
  (∃ (n : ℕ), ∀ (m : ℕ), 
    (∃ (x y z : ℤ), n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) ∧
    (m > n → ¬∃ (x y z : ℤ), m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12)) ∧
  (∃ (x y z : ℤ), 10^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 4*x + 4*y + 4*z - 12) :=
by sorry

end largest_n_for_quadratic_equation_l136_13652


namespace classrooms_needed_l136_13632

def total_students : ℕ := 1675
def students_per_classroom : ℕ := 37

theorem classrooms_needed : 
  ∃ (n : ℕ), n * students_per_classroom ≥ total_students ∧ 
  ∀ (m : ℕ), m * students_per_classroom ≥ total_students → m ≥ n :=
by sorry

end classrooms_needed_l136_13632


namespace tree_planting_growth_rate_l136_13650

/-- Represents the annual average growth rate of tree planting -/
def annual_growth_rate : ℝ → Prop :=
  λ x => 400 * (1 + x)^2 = 625

/-- Theorem stating the relationship between the number of trees planted
    in the first and third years, and the annual average growth rate -/
theorem tree_planting_growth_rate :
  ∃ x : ℝ, annual_growth_rate x :=
sorry

end tree_planting_growth_rate_l136_13650


namespace circle_properties_l136_13664

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Theorem statement
theorem circle_properties :
  -- The center is on the y-axis
  ∃ b : ℝ, circle_equation 0 b ∧
  -- The radius is 1
  (∀ x y : ℝ, circle_equation x y → (x^2 + (y - 2)^2 = 1)) ∧
  -- The circle passes through (1, 2)
  circle_equation 1 2 :=
sorry

end circle_properties_l136_13664


namespace simplify_and_rationalize_l136_13689

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (1 + 1 / (Real.sqrt 5 + 2)) = (Real.sqrt 5 - 1) / 4 := by
  sorry

end simplify_and_rationalize_l136_13689


namespace intersection_range_l136_13674

def set_A (a : ℝ) : Set ℝ := {x | |x - a| < 1}
def set_B : Set ℝ := {x | 1 < x ∧ x < 5}

theorem intersection_range (a : ℝ) : (set_A a ∩ set_B).Nonempty → 0 < a ∧ a < 6 := by
  sorry

end intersection_range_l136_13674


namespace quadratic_equation_roots_l136_13620

theorem quadratic_equation_roots (m : ℝ) : m > 0 →
  (∃ (x : ℝ), x^2 + x - m = 0) ∧
  (∃ (x : ℝ), x^2 + x - m = 0) → m > 0 ∨
  m ≤ 0 → ¬(∃ (x : ℝ), x^2 + x - m = 0) ∨
  ¬(∃ (x : ℝ), x^2 + x - m = 0) → m ≤ 0 :=
by sorry

end quadratic_equation_roots_l136_13620


namespace intersection_in_first_quadrant_l136_13686

/-- The range of k for which the intersection of two lines lies in the first quadrant -/
theorem intersection_in_first_quadrant (k : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x - 1 ∧ x + y - 1 = 0) ↔ k > 1 :=
by sorry

end intersection_in_first_quadrant_l136_13686


namespace problem_1_problem_2_l136_13627

-- Problem 1
theorem problem_1 (a b : ℝ) (h : a ≠ b) : 
  (a^2 / (a - b)) - (b^2 / (a - b)) = a + b :=
sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) (h3 : x ≠ 1) : 
  ((x^2 - 1) / (x^2 + 2*x + 1)) / ((x^2 - x) / (x + 1)) = 1 / x :=
sorry

end problem_1_problem_2_l136_13627


namespace f_properties_l136_13684

-- Define the function f(x) = sin(1/x)
noncomputable def f (x : ℝ) : ℝ := Real.sin (1 / x)

-- State the theorem
theorem f_properties :
  -- The range of f(x) is [-1, 1]
  (∀ y ∈ Set.range f, -1 ≤ y ∧ y ≤ 1) ∧
  (∀ y : ℝ, -1 ≤ y ∧ y ≤ 1 → ∃ x ≠ 0, f x = y) ∧
  -- f(x) is monotonically decreasing on [2/π, +∞)
  (∀ x₁ x₂ : ℝ, x₁ ≥ 2/Real.pi ∧ x₂ ≥ 2/Real.pi ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  -- For any m ∈ [-1, 1], f(x) = m has infinitely many solutions in (0, 1)
  (∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → ∃ (S : Set ℝ), S.Infinite ∧ S ⊆ Set.Ioo 0 1 ∧ ∀ x ∈ S, f x = m) :=
by sorry

end f_properties_l136_13684


namespace fence_perimeter_l136_13681

/-- The outer perimeter of a square fence given the number of posts, post width, and gap between posts. -/
def outerPerimeter (numPosts : ℕ) (postWidth : ℚ) (gapWidth : ℕ) : ℚ :=
  let postsPerSide : ℕ := numPosts / 4
  let numGaps : ℕ := postsPerSide - 1
  let gapLength : ℚ := numGaps * gapWidth
  let postLength : ℚ := postsPerSide * postWidth
  let sideLength : ℚ := gapLength + postLength
  4 * sideLength

/-- Theorem stating that the outer perimeter of the fence is 274 feet. -/
theorem fence_perimeter :
  outerPerimeter 36 (1/2) 8 = 274 := by sorry

end fence_perimeter_l136_13681


namespace union_equals_A_l136_13673

def A : Set ℝ := {x | x^2 + x - 2 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem union_equals_A (m : ℝ) : A ∪ B m = A → m = 0 ∨ m = -1 ∨ m = 1/2 := by
  sorry

end union_equals_A_l136_13673


namespace regular_hexagon_most_symmetry_l136_13678

/-- Number of lines of symmetry for a given shape -/
def linesOfSymmetry (shape : String) : ℕ :=
  match shape with
  | "regular pentagon" => 5
  | "parallelogram" => 0
  | "oval" => 2
  | "right triangle" => 0
  | "regular hexagon" => 6
  | _ => 0

/-- The set of shapes we're considering -/
def shapes : List String := ["regular pentagon", "parallelogram", "oval", "right triangle", "regular hexagon"]

theorem regular_hexagon_most_symmetry :
  ∀ s ∈ shapes, linesOfSymmetry "regular hexagon" ≥ linesOfSymmetry s :=
by sorry

end regular_hexagon_most_symmetry_l136_13678


namespace cuttable_triangle_type_l136_13649

/-- A triangle that can be cut into two parts formable into a rectangle -/
structure CuttableTriangle where
  /-- The triangle can be cut into two parts -/
  can_be_cut : Bool
  /-- The parts can be rearranged into a rectangle -/
  forms_rectangle : Bool

/-- Types of triangles -/
inductive TriangleType
  | Right
  | Isosceles
  | Other

/-- Theorem stating that a cuttable triangle must be right or isosceles -/
theorem cuttable_triangle_type (t : CuttableTriangle) :
  t.can_be_cut ∧ t.forms_rectangle →
  (∃ tt : TriangleType, tt = TriangleType.Right ∨ tt = TriangleType.Isosceles) :=
by
  sorry

end cuttable_triangle_type_l136_13649


namespace largest_constant_inequality_l136_13630

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ↔ C ≤ Real.sqrt (4/3) := by
  sorry

end largest_constant_inequality_l136_13630


namespace trigonometric_calculation_l136_13646

theorem trigonometric_calculation :
  let a := |1 - Real.tan (60 * π / 180)| - (-1/2)⁻¹ + Real.sin (45 * π / 180) + Real.sqrt (1/2)
  let b := -1^2022 + Real.sqrt 12 - (π - 3)^0 - Real.cos (30 * π / 180)
  (a = Real.sqrt 3 + Real.sqrt 2 + 1) ∧ (b = -2 + (3/2) * Real.sqrt 3) := by sorry

end trigonometric_calculation_l136_13646


namespace hyperbola_equivalence_l136_13679

-- Define the equation
def hyperbola_eq (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + y^2) - Real.sqrt ((x + 3)^2 + y^2) = 4

-- Define the standard form of the hyperbola
def hyperbola_standard_form (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1 ∧ x ≤ -2

-- Theorem stating the equivalence
theorem hyperbola_equivalence :
  ∀ x y : ℝ, hyperbola_eq x y ↔ hyperbola_standard_form x y :=
sorry

end hyperbola_equivalence_l136_13679


namespace fraction_of_repeating_decimals_l136_13657

def repeating_decimal_142857 : ℚ := 142857 / 999999
def repeating_decimal_857143 : ℚ := 857143 / 999999

theorem fraction_of_repeating_decimals : 
  (repeating_decimal_142857) / (2 + repeating_decimal_857143) = 1 / 20 := by
  sorry

end fraction_of_repeating_decimals_l136_13657


namespace fifth_friend_payment_l136_13625

/-- Represents the payment made by each friend -/
structure Payment where
  first : ℝ
  second : ℝ
  third : ℝ
  fourth : ℝ
  fifth : ℝ

/-- Conditions for the gift payment problem -/
def GiftPaymentConditions (p : Payment) : Prop :=
  p.first + p.second + p.third + p.fourth + p.fifth = 120 ∧
  p.first = (1/3) * (p.second + p.third + p.fourth + p.fifth) ∧
  p.second = (1/4) * (p.first + p.third + p.fourth + p.fifth) ∧
  p.third = (1/5) * (p.first + p.second + p.fourth + p.fifth)

/-- Theorem stating that under the given conditions, the fifth friend paid $40 -/
theorem fifth_friend_payment (p : Payment) : 
  GiftPaymentConditions p → p.fifth = 40 := by
  sorry

end fifth_friend_payment_l136_13625


namespace opposite_of_negative_two_thirds_l136_13682

theorem opposite_of_negative_two_thirds : 
  -(-(2/3 : ℚ)) = 2/3 := by sorry

end opposite_of_negative_two_thirds_l136_13682


namespace cleaning_staff_lcm_l136_13629

theorem cleaning_staff_lcm : Nat.lcm (Nat.lcm (Nat.lcm 5 6) 8) 10 = 120 := by
  sorry

end cleaning_staff_lcm_l136_13629


namespace prob_one_student_two_books_is_eight_ninths_l136_13617

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of books --/
def num_books : ℕ := 4

/-- The probability of exactly one student receiving two different books --/
def prob_one_student_two_books : ℚ := 8/9

/-- Theorem stating that the probability of exactly one student receiving two different books
    when four distinct books are randomly gifted to three students is equal to 8/9 --/
theorem prob_one_student_two_books_is_eight_ninths :
  prob_one_student_two_books = 8/9 := by
  sorry


end prob_one_student_two_books_is_eight_ninths_l136_13617


namespace unique_positive_solution_l136_13636

theorem unique_positive_solution : ∃! (x : ℝ), x > 0 ∧ (x - 5) / 10 = 5 / (x - 10) := by
  sorry

end unique_positive_solution_l136_13636


namespace four_drivers_sufficient_l136_13624

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  inv_minutes : minutes < 60

/-- Represents a driver -/
inductive Driver
| A | B | C | D

/-- Represents a trip -/
structure Trip where
  driver : Driver
  departure : Time
  arrival : Time

def one_way_duration : Time := ⟨2, 40, sorry⟩
def round_trip_duration : Time := ⟨5, 20, sorry⟩
def min_rest_duration : Time := ⟨1, 0, sorry⟩

def driver_A_return : Time := ⟨12, 40, sorry⟩
def driver_D_departure : Time := ⟨13, 5, sorry⟩
def driver_B_return : Time := ⟨16, 0, sorry⟩
def driver_A_fifth_departure : Time := ⟨16, 10, sorry⟩
def driver_B_sixth_departure : Time := ⟨17, 30, sorry⟩
def alexey_return : Time := ⟨21, 30, sorry⟩

def is_valid_schedule (trips : List Trip) : Prop :=
  sorry

theorem four_drivers_sufficient :
  ∃ (trips : List Trip),
    trips.length ≥ 6 ∧
    is_valid_schedule trips ∧
    (∃ (last_trip : Trip),
      last_trip ∈ trips ∧
      last_trip.driver = Driver.A ∧
      last_trip.departure = driver_A_fifth_departure ∧
      last_trip.arrival = alexey_return) ∧
    (∀ (trip : Trip), trip ∈ trips → trip.driver ∈ [Driver.A, Driver.B, Driver.C, Driver.D]) :=
  sorry

end four_drivers_sufficient_l136_13624


namespace cube_sum_theorem_l136_13660

theorem cube_sum_theorem (p q r : ℝ) 
  (sum_eq : p + q + r = 3)
  (sum_prod_eq : p * q + q * r + r * p = 11)
  (prod_eq : p * q * r = -6) :
  p^3 + q^3 + r^3 = -90 := by
  sorry

end cube_sum_theorem_l136_13660


namespace diagonals_30_gon_skipping_2_l136_13661

/-- A convex polygon with n sides --/
structure ConvexPolygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of diagonals in a convex n-gon that skip exactly k adjacent vertices at each end --/
def diagonals_skipping (n k : ℕ) : ℕ :=
  (n * (n - 2*k - 1)) / 2

/-- Theorem: In a 30-sided convex polygon, there are 375 diagonals that skip exactly 2 adjacent vertices at each end --/
theorem diagonals_30_gon_skipping_2 :
  diagonals_skipping 30 2 = 375 := by
  sorry

end diagonals_30_gon_skipping_2_l136_13661


namespace f_bijective_iff_power_of_two_l136_13699

/-- The set of all possible lamp configurations for n lamps -/
def Ψ (n : ℕ) := Fin (2^n)

/-- The cool procedure function -/
def f (n : ℕ) : Ψ n → Ψ n := sorry

/-- Theorem stating that f is bijective if and only if n is a power of 2 -/
theorem f_bijective_iff_power_of_two (n : ℕ) :
  Function.Bijective (f n) ↔ ∃ k : ℕ, n = 2^k := by sorry

end f_bijective_iff_power_of_two_l136_13699


namespace problem_solution_l136_13669

theorem problem_solution (x : ℝ) (h : x + 1/x = 7) : (x - 3)^2 + 49/(x - 3)^2 = 24 := by
  sorry

end problem_solution_l136_13669


namespace train_passing_bridge_time_l136_13685

/-- Calculates the time for a train to pass a bridge -/
theorem train_passing_bridge_time (train_length : Real) (bridge_length : Real) (train_speed_kmh : Real) :
  let total_distance : Real := train_length + bridge_length
  let train_speed_ms : Real := train_speed_kmh * 1000 / 3600
  let time : Real := total_distance / train_speed_ms
  train_length = 200 ∧ bridge_length = 180 ∧ train_speed_kmh = 65 →
  ∃ ε > 0, |time - 21.04| < ε :=
by
  sorry

end train_passing_bridge_time_l136_13685
