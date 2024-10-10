import Mathlib

namespace reduced_price_is_80_l2356_235692

/-- Represents the price reduction percentage -/
def price_reduction : ℚ := 1/2

/-- Represents the additional amount of oil obtained after price reduction -/
def additional_oil : ℕ := 5

/-- Represents the fixed amount of money spent -/
def fixed_cost : ℕ := 800

/-- Theorem stating that given the conditions, the reduced price per kg is 80 -/
theorem reduced_price_is_80 :
  ∀ (original_price : ℚ) (original_amount : ℚ),
  original_amount * original_price = fixed_cost →
  (original_amount + additional_oil) * (original_price * (1 - price_reduction)) = fixed_cost →
  original_price * (1 - price_reduction) = 80 :=
by sorry

end reduced_price_is_80_l2356_235692


namespace tournament_committee_count_l2356_235646

/-- The number of teams in the league -/
def num_teams : ℕ := 4

/-- The number of members in each team -/
def team_size : ℕ := 7

/-- The number of members selected from the host team -/
def host_selection : ℕ := 3

/-- The number of members selected from each non-host team -/
def non_host_selection : ℕ := 2

/-- The total number of members in the tournament committee -/
def committee_size : ℕ := 9

/-- Theorem stating the total number of possible tournament committees -/
theorem tournament_committee_count :
  (num_teams : ℕ) * (Nat.choose team_size host_selection) * 
  (Nat.choose team_size non_host_selection)^(num_teams - 1) = 1296540 := by
  sorry

end tournament_committee_count_l2356_235646


namespace horsemen_speeds_exist_l2356_235606

/-- Represents a set of speeds for horsemen on a circular track -/
def SpeedSet (n : ℕ) := Fin n → ℝ

/-- Predicate that checks if all speeds in a set are distinct and positive -/
def distinct_positive (s : SpeedSet n) : Prop :=
  ∀ i j, i ≠ j → s i ≠ s j ∧ s i > 0 ∧ s j > 0

/-- Predicate that checks if all overtakings occur at a single point -/
def single_overtaking_point (s : SpeedSet n) : Prop :=
  ∀ i j, i ≠ j → ∃ k : ℤ, (s i) / (s i - s j) = k

/-- Theorem stating that for any number of horsemen (≥ 3), 
    there exists a set of speeds satisfying the required conditions -/
theorem horsemen_speeds_exist (n : ℕ) (h : n ≥ 3) :
  ∃ (s : SpeedSet n), distinct_positive s ∧ single_overtaking_point s :=
sorry

end horsemen_speeds_exist_l2356_235606


namespace exists_non_isosceles_equidistant_inscribed_center_l2356_235610

/-- A triangle with side lengths a, b, and c. -/
structure Triangle :=
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

/-- The center of the inscribed circle of a triangle. -/
def InscribedCenter (t : Triangle) : ℝ × ℝ := sorry

/-- The midpoint of a line segment. -/
def Midpoint (a b : ℝ × ℝ) : ℝ × ℝ := sorry

/-- The distance between two points. -/
def Distance (a b : ℝ × ℝ) : ℝ := sorry

/-- Predicate to check if a triangle is isosceles. -/
def IsIsosceles (t : Triangle) : Prop := 
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Theorem: There exists a non-isosceles triangle where the center of its inscribed circle
    is equidistant from the midpoints of two sides. -/
theorem exists_non_isosceles_equidistant_inscribed_center :
  ∃ (t : Triangle), 
    ¬IsIsosceles t ∧
    ∃ (s₁ s₂ : ℝ × ℝ), 
      Distance (InscribedCenter t) (Midpoint s₁ s₂) = 
      Distance (InscribedCenter t) (Midpoint s₂ (s₁.1 + t.a - s₁.1, s₁.2 + t.b - s₁.2)) :=
sorry

end exists_non_isosceles_equidistant_inscribed_center_l2356_235610


namespace parallel_line_equation_perpendicular_line_coefficient_l2356_235640

-- Define the lines
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def l₃ (x y : ℝ) : Prop := x - 2 * y - 1 = 0
def l₅ (a x y : ℝ) : Prop := a * x - 2 * y + 1 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-2, 2)

-- Theorem 1: The equation of line l₄
theorem parallel_line_equation : 
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ (P.1 = x ∧ P.2 = y ∨ l₃ x y)) ∧ m = 1/2 ∧ b = 3 := by
  sorry

-- Theorem 2: The value of a for perpendicular lines
theorem perpendicular_line_coefficient :
  ∃! a : ℝ, ∀ x y : ℝ, (l₅ a x y ∧ l₂ x y) → a = 1 := by
  sorry

end parallel_line_equation_perpendicular_line_coefficient_l2356_235640


namespace probability_all_white_drawn_l2356_235630

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 6

def probability_all_white : ℚ := 4 / 715

theorem probability_all_white_drawn (total : ℕ) (white : ℕ) (black : ℕ) (drawn : ℕ) :
  total = white + black →
  white ≥ drawn →
  probability_all_white = (Nat.choose white drawn : ℚ) / (Nat.choose total drawn : ℚ) :=
sorry

end probability_all_white_drawn_l2356_235630


namespace incorrect_parentheses_removal_l2356_235676

theorem incorrect_parentheses_removal (a b c : ℝ) : c - 2*(a + b) ≠ c - 2*a + 2*b := by
  sorry

end incorrect_parentheses_removal_l2356_235676


namespace strongest_signal_l2356_235668

def signal_strength (x : ℤ) : ℝ := |x|

def is_stronger (x y : ℤ) : Prop := signal_strength x < signal_strength y

theorem strongest_signal :
  let signals : List ℤ := [-50, -60, -70, -80]
  ∀ s ∈ signals, s ≠ -50 → is_stronger (-50) s :=
by sorry

end strongest_signal_l2356_235668


namespace root_equation_value_l2356_235678

theorem root_equation_value (a : ℝ) : 
  a^2 + 3*a - 1 = 0 → 2*a^2 + 6*a + 2021 = 2023 := by
  sorry

end root_equation_value_l2356_235678


namespace circle_M_properties_l2356_235651

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 1 = 0

-- Define the line L
def line_L (x y : ℝ) : Prop := x + 3*y - 2 = 0

-- Theorem statement
theorem circle_M_properties :
  -- The radius of M is √5
  (∃ (h k r : ℝ), r = Real.sqrt 5 ∧ ∀ (x y : ℝ), circle_M x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧
  -- M is symmetric with respect to the line L
  (∃ (h k : ℝ), circle_M h k ∧ line_L h k ∧
    ∀ (x y : ℝ), circle_M x y → 
      ∃ (x' y' : ℝ), circle_M x' y' ∧ line_L ((x + x')/2) ((y + y')/2)) :=
by sorry

end circle_M_properties_l2356_235651


namespace equivalent_functions_l2356_235628

theorem equivalent_functions (x : ℝ) : x^2 = (x^6)^(1/3) := by
  sorry

end equivalent_functions_l2356_235628


namespace sum_of_P_and_R_is_eight_l2356_235611

theorem sum_of_P_and_R_is_eight :
  ∀ (P Q R S : ℕ),
    P ∈ ({1, 2, 3, 5} : Set ℕ) →
    Q ∈ ({1, 2, 3, 5} : Set ℕ) →
    R ∈ ({1, 2, 3, 5} : Set ℕ) →
    S ∈ ({1, 2, 3, 5} : Set ℕ) →
    P ≠ Q → P ≠ R → P ≠ S → Q ≠ R → Q ≠ S → R ≠ S →
    (P : ℚ) / Q - (R : ℚ) / S = 2 →
    P + R = 8 := by
  sorry

end sum_of_P_and_R_is_eight_l2356_235611


namespace largest_four_digit_divisible_by_five_l2356_235684

theorem largest_four_digit_divisible_by_five : 
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 5 = 0 → n ≤ 9995 :=
by sorry

end largest_four_digit_divisible_by_five_l2356_235684


namespace function_attains_minimum_l2356_235645

def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def IsAdditive (f : ℝ → ℝ) : Prop := ∀ x y, f (x + y) = f x + f y

theorem function_attains_minimum (f : ℝ → ℝ) (a b : ℝ) 
  (h_odd : IsOdd f) 
  (h_additive : IsAdditive f)
  (h_neg : ∀ x > 0, f x < 0)
  (h_ab : a < b) :
  ∀ x ∈ Set.Icc a b, f b ≤ f x :=
sorry

end function_attains_minimum_l2356_235645


namespace intersection_sum_l2356_235632

theorem intersection_sum (c d : ℚ) : 
  (3 = (1/3) * (-1) + c) → 
  (-1 = (1/3) * 3 + d) → 
  c + d = 4/3 := by
sorry

end intersection_sum_l2356_235632


namespace ellipse_major_axis_length_l2356_235616

/-- Given a right circular cylinder of radius 2 intersected by a plane forming an ellipse,
    if the major axis is 20% longer than the minor axis, then the length of the major axis is 4.8. -/
theorem ellipse_major_axis_length (cylinder_radius : ℝ) (minor_axis : ℝ) (major_axis : ℝ) : 
  cylinder_radius = 2 →
  minor_axis = 2 * cylinder_radius →
  major_axis = minor_axis * 1.2 →
  major_axis = 4.8 := by
sorry

end ellipse_major_axis_length_l2356_235616


namespace value_equivalence_l2356_235636

theorem value_equivalence : 3000 * (3000^3000 + 3000^2999) = 3001 * 3000^3000 := by
  sorry

end value_equivalence_l2356_235636


namespace pencil_count_l2356_235681

/-- Proves that given the ratio of pens to pencils is 5:6 and there are 9 more pencils than pens, the number of pencils is 54. -/
theorem pencil_count (pens pencils : ℕ) : 
  (pens : ℚ) / (pencils : ℚ) = 5 / 6 →
  pencils = pens + 9 →
  pencils = 54 := by
sorry

end pencil_count_l2356_235681


namespace arrange_objects_count_l2356_235685

/-- The number of ways to arrange 7 indistinguishable objects of one type
    and 3 indistinguishable objects of another type in a row of 10 positions -/
def arrangeObjects : ℕ := Nat.choose 10 3

/-- Theorem stating that the number of arrangements is equal to binomial coefficient (10 choose 3) -/
theorem arrange_objects_count : arrangeObjects = 120 := by
  sorry

end arrange_objects_count_l2356_235685


namespace root_product_expression_l2356_235604

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α - 1 = 0) → 
  (β^2 + p*β - 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = p^2 - q^2 := by
sorry

end root_product_expression_l2356_235604


namespace f_one_equals_four_l2356_235618

/-- A function f(x) that is always non-negative for real x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 3*a - 9

/-- The theorem stating that f(1) = 4 given the conditions -/
theorem f_one_equals_four (a : ℝ) (h : ∀ x : ℝ, f a x ≥ 0) : f a 1 = 4 := by
  sorry

end f_one_equals_four_l2356_235618


namespace instantaneous_velocity_at_4_l2356_235625

/-- The position function of the object -/
def s (t : ℝ) : ℝ := 3 * t^2 + t + 4

/-- The velocity function of the object (derivative of s) -/
def v (t : ℝ) : ℝ := 6 * t + 1

theorem instantaneous_velocity_at_4 : v 4 = 25 := by
  sorry

end instantaneous_velocity_at_4_l2356_235625


namespace system_solvability_l2356_235683

-- Define the system of equations
def system (x y a b : ℝ) : Prop :=
  x * Real.cos a + y * Real.sin a + 4 ≤ 0 ∧
  x^2 + y^2 + 10*x + 2*y - b^2 - 8*b + 10 = 0

-- Define the set of valid b values
def valid_b_set (b : ℝ) : Prop :=
  b ≤ -8 - Real.sqrt 26 ∨ b ≥ Real.sqrt 26

-- Theorem statement
theorem system_solvability (b : ℝ) :
  (∀ a, ∃ x y, system x y a b) ↔ valid_b_set b :=
sorry

end system_solvability_l2356_235683


namespace increasing_f_implies_a_range_l2356_235638

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 + (a^3 - a) * x + 1

-- State the theorem
theorem increasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → x ≤ -1 → f a x < f a y) →
  -Real.sqrt 3 ≤ a ∧ a < 0 :=
by sorry

end increasing_f_implies_a_range_l2356_235638


namespace function_properties_l2356_235674

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2*a - x) = f x

theorem function_properties (f : ℝ → ℝ) :
  (∀ x, f (x + 2) = f (x - 2)) →
  (∀ x, f (4 - x) = f x) →
  (is_periodic f 4 ∧ is_symmetric_about f 2) :=
by sorry

end function_properties_l2356_235674


namespace smallest_x_absolute_value_equation_l2356_235626

theorem smallest_x_absolute_value_equation :
  ∃ x : ℝ, (∀ y : ℝ, y * |y| = 3 * y + 2 → x ≤ y) ∧ x * |x| = 3 * x + 2 :=
by sorry

end smallest_x_absolute_value_equation_l2356_235626


namespace root_sum_squares_l2356_235637

theorem root_sum_squares (p q r : ℝ) : 
  (p^3 - 15*p^2 + 22*p - 8 = 0) → 
  (q^3 - 15*q^2 + 22*q - 8 = 0) → 
  (r^3 - 15*r^2 + 22*r - 8 = 0) → 
  (p+q)^2 + (q+r)^2 + (r+p)^2 = 406 := by
sorry

end root_sum_squares_l2356_235637


namespace exists_wonderful_with_many_primes_l2356_235620

/-- A number is wonderful if it's divisible by the sum of its prime factors -/
def IsWonderful (n : ℕ) : Prop :=
  n > 1 ∧ ∃ (factors : List ℕ), (factors.all Nat.Prime) ∧ 
  (factors.prod = n) ∧ (n % factors.sum = 0)

/-- There exists a wonderful number with at least 10^2002 distinct prime factors -/
theorem exists_wonderful_with_many_primes : 
  ∃ (n : ℕ), IsWonderful n ∧ (∃ (factors : List ℕ), 
    (factors.all Nat.Prime) ∧ (factors.prod = n) ∧ 
    (factors.length ≥ 10^2002) ∧ (factors.Nodup)) := by
  sorry

end exists_wonderful_with_many_primes_l2356_235620


namespace gcd_459_357_l2356_235687

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l2356_235687


namespace sin_135_degrees_l2356_235693

theorem sin_135_degrees : Real.sin (135 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_135_degrees_l2356_235693


namespace empire_state_building_total_height_l2356_235667

/-- The height of the Empire State Building -/
def empire_state_building_height (top_floor_height antenna_height : ℕ) : ℕ :=
  top_floor_height + antenna_height

/-- Theorem: The Empire State Building is 1454 feet tall -/
theorem empire_state_building_total_height :
  empire_state_building_height 1250 204 = 1454 := by
  sorry

end empire_state_building_total_height_l2356_235667


namespace not_perfect_square_l2356_235663

theorem not_perfect_square (a b : ℕ) (h1 : a > b) (h2 : Nat.gcd (a - b) (a * b + 1) = 1) (h3 : Nat.gcd (a + b) (a * b - 1) = 1) :
  ¬ ∃ k : ℕ, (a - b)^2 + (a * b + 1)^2 = k^2 := by
  sorry

end not_perfect_square_l2356_235663


namespace battery_problem_l2356_235631

theorem battery_problem :
  ∀ (x y z : ℚ),
  (x > 0) → (y > 0) → (z > 0) →
  (4*x + 18*y + 16*z = 4*x + 15*y + 24*z) →
  (4*x + 18*y + 16*z = 6*x + 12*y + 20*z) →
  (∃ (W : ℚ), W * z = 4*x + 18*y + 16*z ∧ W = 48) :=
by
  sorry

end battery_problem_l2356_235631


namespace santino_papaya_trees_l2356_235691

/-- The number of papaya trees Santino has -/
def num_papaya_trees : ℕ := sorry

/-- The number of mango trees Santino has -/
def num_mango_trees : ℕ := 3

/-- The number of papayas produced by each papaya tree -/
def papayas_per_tree : ℕ := 10

/-- The number of mangos produced by each mango tree -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits Santino has -/
def total_fruits : ℕ := 80

/-- Theorem stating that Santino has 2 papaya trees -/
theorem santino_papaya_trees :
  num_papaya_trees * papayas_per_tree + num_mango_trees * mangos_per_tree = total_fruits ∧
  num_papaya_trees = 2 := by sorry

end santino_papaya_trees_l2356_235691


namespace pirate_treasure_distribution_l2356_235601

theorem pirate_treasure_distribution (x : ℕ) : x > 0 → (x * (x + 1)) / 2 = 5 * x → x + 5 * x = 54 := by
  sorry

end pirate_treasure_distribution_l2356_235601


namespace art_dealer_earnings_l2356_235664

/-- Calculates the total money made from selling etchings -/
def total_money_made (total_etchings : ℕ) (first_group_count : ℕ) (first_group_price : ℕ) (second_group_price : ℕ) : ℕ :=
  let second_group_count := total_etchings - first_group_count
  (first_group_count * first_group_price) + (second_group_count * second_group_price)

/-- Proves that the art dealer made $630 from selling the etchings -/
theorem art_dealer_earnings : total_money_made 16 9 35 45 = 630 := by
  sorry

end art_dealer_earnings_l2356_235664


namespace geometric_sequence_problem_l2356_235634

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the problem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 10)^2 - 11*(a 10) + 16 = 0 →
  (a 30)^2 - 11*(a 30) + 16 = 0 →
  a 20 = 4 :=
by
  sorry

end geometric_sequence_problem_l2356_235634


namespace tv_price_change_l2356_235666

theorem tv_price_change (initial_price : ℝ) (h : initial_price > 0) : 
  let price_after_decrease : ℝ := 0.8 * initial_price
  let final_price : ℝ := 1.24 * initial_price
  ∃ x : ℝ, price_after_decrease * (1 + x / 100) = final_price ∧ x = 55 := by
sorry

end tv_price_change_l2356_235666


namespace spicy_hot_noodles_count_l2356_235696

theorem spicy_hot_noodles_count (total_plates lobster_rolls seafood_noodles : ℕ) 
  (h1 : total_plates = 55)
  (h2 : lobster_rolls = 25)
  (h3 : seafood_noodles = 16) :
  total_plates - (lobster_rolls + seafood_noodles) = 14 := by
  sorry

end spicy_hot_noodles_count_l2356_235696


namespace parallel_lines_sum_l2356_235669

/-- Two parallel lines with a given distance between them -/
structure ParallelLines where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  distance : ℝ
  parallel : a / b = c / d
  dist_formula : distance = |c - a| / Real.sqrt (c^2 + d^2)

/-- The theorem to be proved -/
theorem parallel_lines_sum (lines : ParallelLines) 
  (h1 : lines.a = 3 ∧ lines.b = 4)
  (h2 : lines.c = 6)
  (h3 : lines.distance = 3) :
  (lines.d + lines.c = -12) ∨ (lines.d + lines.c = 48) := by
  sorry

end parallel_lines_sum_l2356_235669


namespace screen_area_difference_l2356_235680

/-- The area difference between two square screens given their diagonal lengths -/
theorem screen_area_difference (d1 d2 : ℝ) (h1 : d1 = 20) (h2 : d2 = 18) :
  d1^2 - d2^2 = 76 := by
  sorry

#check screen_area_difference

end screen_area_difference_l2356_235680


namespace total_vertices_eq_21_l2356_235633

/-- The number of vertices in a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of vertices in a hexagon -/
def hexagon_vertices : ℕ := 6

/-- The number of triangles -/
def num_triangles : ℕ := 1

/-- The number of hexagons -/
def num_hexagons : ℕ := 3

/-- The total number of vertices in all shapes -/
def total_vertices : ℕ := num_triangles * triangle_vertices + num_hexagons * hexagon_vertices

theorem total_vertices_eq_21 : total_vertices = 21 := by
  sorry

end total_vertices_eq_21_l2356_235633


namespace sum_of_powers_mod_17_l2356_235635

theorem sum_of_powers_mod_17 : ∃ (a b c d : ℕ), 
  (3 * a) % 17 = 1 ∧ 
  (3 * b) % 17 = 3 ∧ 
  (3 * c) % 17 = 9 ∧ 
  (3 * d) % 17 = 10 ∧ 
  (a + b + c + d) % 17 = 5 := by
  sorry

end sum_of_powers_mod_17_l2356_235635


namespace overtaking_car_speed_l2356_235623

/-- Proves that given a red car traveling at 30 mph with a 20-mile head start,
    if another car overtakes it in 1 hour, the speed of the other car must be 50 mph. -/
theorem overtaking_car_speed
  (red_car_speed : ℝ)
  (red_car_lead : ℝ)
  (overtake_time : ℝ)
  (h1 : red_car_speed = 30)
  (h2 : red_car_lead = 20)
  (h3 : overtake_time = 1) :
  let other_car_speed := (red_car_speed * overtake_time + red_car_lead) / overtake_time
  other_car_speed = 50 := by
sorry

end overtaking_car_speed_l2356_235623


namespace intersection_A_B_union_B_C_implies_a_range_l2356_235622

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem for part (2)
theorem union_B_C_implies_a_range (a : ℝ) : B ∪ C a = C a → a ≤ 3 := by sorry

end intersection_A_B_union_B_C_implies_a_range_l2356_235622


namespace parentheses_equivalence_l2356_235695

theorem parentheses_equivalence (a b c : ℝ) : a - b + c = a - (b - c) := by
  sorry

end parentheses_equivalence_l2356_235695


namespace right_triangle_sin_R_l2356_235643

theorem right_triangle_sin_R (P Q R : ℝ) (h_right_triangle : P + Q + R = π) 
  (h_sin_P : Real.sin P = 3/5) (h_sin_Q : Real.sin Q = 1) : Real.sin R = 4/5 := by
  sorry

end right_triangle_sin_R_l2356_235643


namespace complement_of_A_in_U_l2356_235698

-- Define the universal set U
def U : Set Int := {x | -3 ≤ x ∧ x ≤ 3}

-- Define set A
def A : Set Int := {0, 1, 2, 3}

-- State the theorem
theorem complement_of_A_in_U : 
  (U \ A) = {-3, -2, -1} := by sorry

end complement_of_A_in_U_l2356_235698


namespace sample_is_sixteen_l2356_235672

/-- Represents a stratified sampling scenario in a factory -/
structure StratifiedSampling where
  totalSample : ℕ
  totalProducts : ℕ
  workshopProducts : ℕ
  h_positive : 0 < totalSample ∧ 0 < totalProducts ∧ 0 < workshopProducts
  h_valid : workshopProducts ≤ totalProducts

/-- Calculates the number of items sampled from a specific workshop -/
def sampleFromWorkshop (s : StratifiedSampling) : ℕ :=
  (s.totalSample * s.workshopProducts) / s.totalProducts

/-- Theorem stating that for the given scenario, the sample from the workshop is 16 -/
theorem sample_is_sixteen (s : StratifiedSampling) 
  (h_total_sample : s.totalSample = 128)
  (h_total_products : s.totalProducts = 2048)
  (h_workshop_products : s.workshopProducts = 256) : 
  sampleFromWorkshop s = 16 := by
  sorry

#eval sampleFromWorkshop { 
  totalSample := 128, 
  totalProducts := 2048, 
  workshopProducts := 256, 
  h_positive := by norm_num, 
  h_valid := by norm_num 
}

end sample_is_sixteen_l2356_235672


namespace marble_draw_theorem_l2356_235613

/-- Represents the number of marbles of each color in the bucket -/
structure MarbleCounts where
  red : Nat
  green : Nat
  blue : Nat
  yellow : Nat
  orange : Nat
  purple : Nat

/-- The actual counts of marbles in the bucket -/
def initialCounts : MarbleCounts :=
  { red := 35, green := 25, blue := 24, yellow := 18, orange := 15, purple := 12 }

/-- The minimum number of marbles to guarantee at least 20 of a single color -/
def minMarblesToDraw : Nat := 103

theorem marble_draw_theorem (counts : MarbleCounts := initialCounts) :
  (∀ n : Nat, n < minMarblesToDraw →
    ∃ c : MarbleCounts, c.red < 20 ∧ c.green < 20 ∧ c.blue < 20 ∧
      c.yellow < 20 ∧ c.orange < 20 ∧ c.purple < 20 ∧
      c.red + c.green + c.blue + c.yellow + c.orange + c.purple = n) ∧
  (∀ c : MarbleCounts,
    c.red + c.green + c.blue + c.yellow + c.orange + c.purple = minMarblesToDraw →
    c.red ≥ 20 ∨ c.green ≥ 20 ∨ c.blue ≥ 20 ∨ c.yellow ≥ 20 ∨ c.orange ≥ 20 ∨ c.purple ≥ 20) :=
by sorry

end marble_draw_theorem_l2356_235613


namespace fifteen_non_congruent_triangles_l2356_235689

-- Define the points
variable (A B C M N P : ℝ × ℝ)

-- Define the isosceles triangle
def is_isosceles_triangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist A C

-- Define M as midpoint of AB
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define N on AC with 1:2 ratio
def divides_in_ratio_one_two (N A C : ℝ × ℝ) : Prop :=
  dist A N = (1/3) * dist A C

-- Define P on BC with 1:3 ratio
def divides_in_ratio_one_three (P B C : ℝ × ℝ) : Prop :=
  dist B P = (1/4) * dist B C

-- Define a function to count non-congruent triangles
def count_non_congruent_triangles (A B C M N P : ℝ × ℝ) : ℕ := sorry

-- State the theorem
theorem fifteen_non_congruent_triangles
  (h1 : is_isosceles_triangle A B C)
  (h2 : is_midpoint M A B)
  (h3 : divides_in_ratio_one_two N A C)
  (h4 : divides_in_ratio_one_three P B C) :
  count_non_congruent_triangles A B C M N P = 15 := by sorry

end fifteen_non_congruent_triangles_l2356_235689


namespace equation_solution_l2356_235697

theorem equation_solution (x : ℝ) : 14 * x + 5 - 21 * x^2 = -2 → 6 * x^2 - 4 * x + 5 = 7 := by
  sorry

end equation_solution_l2356_235697


namespace inequality_solution_range_l2356_235629

theorem inequality_solution_range (a : ℝ) : 
  ((3 - a) * (3 + 2*a - 1)^2 * (3 - 3*a) ≤ 0) →
  (a = -1 ∨ (1 ≤ a ∧ a ≤ 3)) :=
by sorry

end inequality_solution_range_l2356_235629


namespace rectangular_box_volume_l2356_235688

/-- The volume of a rectangular box with face areas 36, 18, and 12 square inches -/
theorem rectangular_box_volume (l w h : ℝ) 
  (face1 : l * w = 36)
  (face2 : w * h = 18)
  (face3 : l * h = 12) :
  l * w * h = 36 * Real.sqrt 6 := by
sorry

end rectangular_box_volume_l2356_235688


namespace beach_problem_l2356_235644

/-- The number of people in the third row at the beach -/
def third_row_count (total_rows : Nat) (initial_first_row : Nat) (left_first_row : Nat) 
  (initial_second_row : Nat) (left_second_row : Nat) (total_left : Nat) : Nat :=
  total_left - ((initial_first_row - left_first_row) + (initial_second_row - left_second_row))

/-- Theorem: The number of people in the third row is 18 -/
theorem beach_problem : 
  third_row_count 3 24 3 20 5 54 = 18 := by
  sorry

end beach_problem_l2356_235644


namespace problem_solution_l2356_235639

theorem problem_solution (x z : ℝ) (hx : x ≠ 0) (h1 : x/3 = z^2 + 1) (h2 : x/5 = 5*z + 2) :
  x = (685 + 25 * Real.sqrt 541) / 6 := by
sorry

end problem_solution_l2356_235639


namespace parabola_directrix_l2356_235602

/-- The directrix of a parabola y^2 = 16x is x = -4 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 16*x → (∃ (a : ℝ), a = 4 ∧ x = -a) :=
by sorry

end parabola_directrix_l2356_235602


namespace coursework_materials_expense_l2356_235686

def budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

theorem coursework_materials_expense : 
  budget * (1 - (food_percentage + accommodation_percentage + entertainment_percentage)) = 300 := by
  sorry

end coursework_materials_expense_l2356_235686


namespace g_of_three_l2356_235682

/-- Given a function g such that g(x-1) = 2x + 6 for all x, prove that g(3) = 14 -/
theorem g_of_three (g : ℝ → ℝ) (h : ∀ x, g (x - 1) = 2 * x + 6) : g 3 = 14 := by
  sorry

end g_of_three_l2356_235682


namespace curve_is_hyperbola_l2356_235654

/-- Given parametric equations representing a curve, prove that it forms a hyperbola -/
theorem curve_is_hyperbola (θ : ℝ) (h_θ : ∀ n : ℤ, θ ≠ n * π / 2) :
  ∃ (x y : ℝ → ℝ),
    (∀ t, x t = ((Real.exp t + Real.exp (-t)) / 2) * Real.cos θ) ∧
    (∀ t, y t = ((Real.exp t - Real.exp (-t)) / 2) * Real.sin θ) →
    ∀ t, (x t)^2 / (Real.cos θ)^2 - (y t)^2 / (Real.sin θ)^2 = 1 :=
sorry

end curve_is_hyperbola_l2356_235654


namespace smallest_positive_integer_with_remainders_l2356_235677

theorem smallest_positive_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 4 = 3) ∧ 
  (x % 6 = 5) ∧ 
  (∀ (y : ℕ), y > 0 → y % 4 = 3 → y % 6 = 5 → x ≤ y) ∧
  (x = 11) := by
  sorry

end smallest_positive_integer_with_remainders_l2356_235677


namespace complex_real_part_l2356_235615

theorem complex_real_part (z : ℂ) (h : (z^2 + z).im = 0) : z.re = -1/2 := by
  sorry

end complex_real_part_l2356_235615


namespace axis_of_symmetry_l2356_235679

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define the transformed function
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2*x + 1)

-- Theorem statement
theorem axis_of_symmetry (f : ℝ → ℝ) (h : even_function f) :
  ∀ x : ℝ, g f ((-1) + x) = g f ((-1) - x) :=
sorry

end axis_of_symmetry_l2356_235679


namespace production_performance_l2356_235650

/-- Represents the production schedule and actual performance of a team of workers. -/
structure ProductionSchedule where
  total_parts : ℕ
  days_ahead : ℕ
  extra_parts_per_day : ℕ

/-- Calculates the intended time frame and daily overachievement percentage. -/
def calculate_performance (schedule : ProductionSchedule) : ℕ × ℚ :=
  sorry

/-- Theorem stating that for the given production schedule, 
    the intended time frame was 40 days and the daily overachievement was 25%. -/
theorem production_performance :
  let schedule := ProductionSchedule.mk 8000 8 50
  calculate_performance schedule = (40, 25/100) := by
  sorry

end production_performance_l2356_235650


namespace parallel_iff_equal_slope_l2356_235619

/-- Two lines in the plane -/
structure Line where
  k : ℝ
  c : ℝ
  eq : ∀ x y : ℝ, k * x + y + c = 0

/-- Parallel lines have the same slope -/
def parallel (l₁ l₂ : Line) : Prop := l₁.k = l₂.k

theorem parallel_iff_equal_slope (l₁ l₂ : Line) : 
  parallel l₁ l₂ ↔ l₁.k = l₂.k :=
sorry

end parallel_iff_equal_slope_l2356_235619


namespace cube_preserves_order_l2356_235657

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_preserves_order_l2356_235657


namespace ratio_equality_l2356_235627

theorem ratio_equality (a b : ℝ) (h1 : 3 * a = 4 * b) (h2 : a * b ≠ 0) :
  (a / 4) / (b / 3) = 1 := by
  sorry

end ratio_equality_l2356_235627


namespace megacorp_fine_l2356_235612

/-- Represents the days of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Calculate the profit percentage for a given day -/
def profitPercentage (d : Day) : ℝ :=
  match d with
  | Day.Monday => 0.1
  | Day.Tuesday => 0.2
  | Day.Wednesday => 0.15
  | Day.Thursday => 0.25
  | Day.Friday => 0.3
  | Day.Saturday => 0
  | Day.Sunday => 0

/-- Daily earnings from mining -/
def miningEarnings : ℝ := 3000000

/-- Daily earnings from oil refining -/
def oilRefiningEarnings : ℝ := 5000000

/-- Monthly expenses -/
def monthlyExpenses : ℝ := 30000000

/-- Tax rate on profits -/
def taxRate : ℝ := 0.35

/-- Fine rate on annual profits -/
def fineRate : ℝ := 0.01

/-- Number of days in a month (assumed average) -/
def daysInMonth : ℕ := 30

/-- Number of months in a year -/
def monthsInYear : ℕ := 12

/-- Calculate MegaCorp's fine -/
def calculateFine : ℝ :=
  let dailyEarnings := miningEarnings + oilRefiningEarnings
  let weeklyProfits := (List.sum (List.map (fun d => dailyEarnings * profitPercentage d) [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday]))
  let monthlyRevenue := dailyEarnings * daysInMonth
  let monthlyProfits := monthlyRevenue - monthlyExpenses - (taxRate * (weeklyProfits * 4))
  let annualProfits := monthlyProfits * monthsInYear
  fineRate * annualProfits

theorem megacorp_fine : calculateFine = 23856000 := by sorry


end megacorp_fine_l2356_235612


namespace necessary_not_sufficient_condition_l2356_235621

theorem necessary_not_sufficient_condition (a b c : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a / b = b / c ∧ b ^ 2 = a * c) ∧ 
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b ^ 2 = a * c ∧ a / b ≠ b / c) :=
by sorry

end necessary_not_sufficient_condition_l2356_235621


namespace molecular_weight_of_Y_l2356_235605

/-- Represents a chemical compound with its molecular weight -/
structure Compound where
  molecularWeight : ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactant1 : Compound
  reactant2 : Compound
  product : Compound
  reactant2Coefficient : ℕ

/-- The law of conservation of mass in a chemical reaction -/
def conservationOfMass (r : Reaction) : Prop :=
  r.reactant1.molecularWeight + r.reactant2Coefficient * r.reactant2.molecularWeight = r.product.molecularWeight

/-- Theorem: The molecular weight of Y in the given reaction -/
theorem molecular_weight_of_Y : 
  let X : Compound := ⟨136⟩
  let C6H8O7 : Compound := ⟨192⟩
  let Y : Compound := ⟨1096⟩
  let reaction : Reaction := ⟨X, C6H8O7, Y, 5⟩
  conservationOfMass reaction := by
  sorry

end molecular_weight_of_Y_l2356_235605


namespace triangle_side_length_l2356_235609

/-- Given a triangle ABC where ∠B = 45°, AB = 100, and AC = 100√2, prove that BC = 100√(5 + √2(√6 - √2)). -/
theorem triangle_side_length (A B C : ℝ×ℝ) : 
  let angleB := Real.arccos ((B.1 - A.1) / Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2))
  let sideAB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let sideAC := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  angleB = π/4 ∧ sideAB = 100 ∧ sideAC = 100 * Real.sqrt 2 →
  Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 100 * Real.sqrt (5 + Real.sqrt 2 * (Real.sqrt 6 - Real.sqrt 2)) :=
by sorry

end triangle_side_length_l2356_235609


namespace average_writing_rate_l2356_235624

/-- Given a writer who completed 50,000 words in 100 hours, 
    prove that the average writing rate is 500 words per hour. -/
theorem average_writing_rate 
  (total_words : ℕ) 
  (total_hours : ℕ) 
  (h1 : total_words = 50000) 
  (h2 : total_hours = 100) : 
  (total_words : ℚ) / total_hours = 500 := by
  sorry

end average_writing_rate_l2356_235624


namespace honey_shop_problem_l2356_235655

/-- The honey shop problem -/
theorem honey_shop_problem (bulk_price tax min_spend penny_paid : ℚ)
  (h1 : bulk_price = 5)
  (h2 : tax = 1)
  (h3 : min_spend = 40)
  (h4 : penny_paid = 240) :
  (penny_paid / (bulk_price + tax) - min_spend / bulk_price) = 32 := by
  sorry

end honey_shop_problem_l2356_235655


namespace other_factor_of_prime_multiple_l2356_235614

theorem other_factor_of_prime_multiple (p n : ℕ) : 
  Nat.Prime p → 
  (∃ k, n = k * p) → 
  (∀ d : ℕ, d ∣ n ↔ d = 1 ∨ d = n) → 
  ∃ k : ℕ, n = k * p ∧ k = 1 :=
by sorry

end other_factor_of_prime_multiple_l2356_235614


namespace dodecahedron_edge_probability_l2356_235675

/-- A regular dodecahedron with 20 vertices -/
structure Dodecahedron :=
  (vertices : Finset Nat)
  (h_card : vertices.card = 20)

/-- The probability of two randomly chosen vertices being endpoints of an edge -/
def edge_probability (d : Dodecahedron) : ℚ :=
  3 / 19

theorem dodecahedron_edge_probability (d : Dodecahedron) :
  edge_probability d = 3 / 19 := by
  sorry

end dodecahedron_edge_probability_l2356_235675


namespace function_inequality_equivalence_l2356_235647

theorem function_inequality_equivalence 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = 3 * (x + 2)^2 - 1) 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) : 
  (∀ x, |x + 2| < b → |f x - 7| < a) ↔ b^2 = (8 + a) / 3 :=
by sorry

end function_inequality_equivalence_l2356_235647


namespace sum_mod_nine_l2356_235652

theorem sum_mod_nine : (7155 + 7156 + 7157 + 7158 + 7159) % 9 = 1 := by
  sorry

end sum_mod_nine_l2356_235652


namespace decline_type_composition_l2356_235641

/-- Represents the age composition of a population --/
inductive AgeComposition
  | Growth
  | Stable
  | Decline

/-- Represents the relative distribution of age groups in a population --/
structure PopulationDistribution where
  young : ℕ
  adult : ℕ
  elderly : ℕ

/-- Determines the age composition based on the population distribution --/
def determineAgeComposition (pop : PopulationDistribution) : AgeComposition :=
  sorry

/-- Theorem stating that a population with fewer young individuals and more adults and elderly individuals has a decline type age composition --/
theorem decline_type_composition (pop : PopulationDistribution) 
  (h1 : pop.young < pop.adult)
  (h2 : pop.young < pop.elderly) :
  determineAgeComposition pop = AgeComposition.Decline :=
sorry

end decline_type_composition_l2356_235641


namespace computer_price_reduction_l2356_235607

/-- The average percentage decrease per price reduction for a computer model -/
theorem computer_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 5000)
  (h2 : final_price = 2560)
  (h3 : ∃ x : ℝ, original_price * (1 - x/100)^3 = final_price) :
  ∃ x : ℝ, x = 20 ∧ original_price * (1 - x/100)^3 = final_price := by
sorry


end computer_price_reduction_l2356_235607


namespace square_of_98_l2356_235653

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by sorry

end square_of_98_l2356_235653


namespace cristine_lemons_l2356_235600

theorem cristine_lemons : ∀ (initial_lemons : ℕ),
  (3 / 4 : ℚ) * initial_lemons = 9 →
  initial_lemons = 12 := by
  sorry

end cristine_lemons_l2356_235600


namespace quadratic_roots_sum_l2356_235661

theorem quadratic_roots_sum (a b : ℝ) (ha : a^2 - 8*a + 5 = 0) (hb : b^2 - 8*b + 5 = 0) (hab : a ≠ b) :
  (b - 1) / (a - 1) + (a - 1) / (b - 1) = -20 := by
sorry

end quadratic_roots_sum_l2356_235661


namespace complementary_event_is_both_red_l2356_235659

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure TwoBallDraw where
  first : Color
  second : Color

/-- The set of all possible outcomes when drawing two balls -/
def allOutcomes : Set TwoBallDraw :=
  {⟨Color.Red, Color.Red⟩, ⟨Color.Red, Color.White⟩, 
   ⟨Color.White, Color.Red⟩, ⟨Color.White, Color.White⟩}

/-- Event A: At least one white ball -/
def eventA : Set TwoBallDraw :=
  {draw ∈ allOutcomes | draw.first = Color.White ∨ draw.second = Color.White}

/-- The complementary event of A -/
def complementA : Set TwoBallDraw :=
  allOutcomes \ eventA

theorem complementary_event_is_both_red :
  complementA = {⟨Color.Red, Color.Red⟩} :=
by sorry

end complementary_event_is_both_red_l2356_235659


namespace expression_evaluation_l2356_235673

theorem expression_evaluation :
  let f (x : ℝ) := -7*x + 2*(x^2 - 1) - (2*x^2 - x + 3)
  f 1 = -11 := by
sorry

end expression_evaluation_l2356_235673


namespace donut_area_l2356_235690

/-- The area of a donut shape formed by two concentric circles -/
theorem donut_area (r₁ r₂ : ℝ) (h₁ : r₁ = 7) (h₂ : r₂ = 10) :
  (r₂^2 - r₁^2) * π = 51 * π := by
  sorry

#check donut_area

end donut_area_l2356_235690


namespace tim_cabinet_price_l2356_235699

/-- The amount Tim paid for a cabinet with a discount -/
def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

/-- Proof that Tim paid $1020 for the cabinet -/
theorem tim_cabinet_price :
  let original_price : ℝ := 1200
  let discount_rate : ℝ := 0.15
  discounted_price original_price discount_rate = 1020 := by
sorry

end tim_cabinet_price_l2356_235699


namespace quadratic_integer_solution_l2356_235660

theorem quadratic_integer_solution (a : ℕ+) :
  (∃ x : ℤ, a * x^2 + 2*(2*a - 1)*x + 4*a - 7 = 0) ↔ (a = 1 ∨ a = 5) := by
  sorry

end quadratic_integer_solution_l2356_235660


namespace right_triangle_division_area_ratio_l2356_235608

/-- Given a right triangle divided into a rectangle and two smaller right triangles,
    where one smaller triangle has area n times the rectangle's area,
    and the rectangle's length is twice its width,
    prove that the ratio of the other small triangle's area to the rectangle's area is 1/(4n). -/
theorem right_triangle_division_area_ratio (n : ℝ) (h : n > 0) :
  ∃ (x : ℝ) (t s : ℝ),
    x > 0 ∧ 
    2 * x^2 > 0 ∧  -- Area of rectangle
    (1/2) * t * x = n * (2 * x^2) ∧  -- Area of one small triangle
    s / x = x / (2 * n * x) ∧  -- Similar triangles ratio
    ((1/2) * (2*x) * s) / (2 * x^2) = 1 / (4*n) :=
by sorry

end right_triangle_division_area_ratio_l2356_235608


namespace right_triangle_hypotenuse_l2356_235642

theorem right_triangle_hypotenuse (a b h : ℝ) : 
  a = 15 → b = 21 → h^2 = a^2 + b^2 → h = Real.sqrt 666 := by sorry

end right_triangle_hypotenuse_l2356_235642


namespace min_cosine_sqrt3_sine_l2356_235617

theorem min_cosine_sqrt3_sine (A : Real) :
  let f := λ A : Real => Real.cos (A / 2) + Real.sqrt 3 * Real.sin (A / 2)
  ∃ (min : Real), f min ≤ f A ∧ min = 840 * Real.pi / 180 :=
sorry

end min_cosine_sqrt3_sine_l2356_235617


namespace lock_min_moves_l2356_235603

/-- Represents a combination lock with n discs, each having d digits -/
structure CombinationLock (n : ℕ) (d : ℕ) where
  discs : Fin n → Fin d

/-- Represents a move on the lock -/
def move (lock : CombinationLock n d) (disc : Fin n) (direction : Bool) : CombinationLock n d :=
  sorry

/-- Checks if a combination is valid (for part b) -/
def is_valid_combination (lock : CombinationLock n d) : Bool :=
  sorry

/-- The number of moves required to ensure finding the correct combination -/
def min_moves (n : ℕ) (d : ℕ) (initial : CombinationLock n d) (valid : CombinationLock n d → Bool) : ℕ :=
  sorry

theorem lock_min_moves :
  let n : ℕ := 6
  let d : ℕ := 10
  let initial : CombinationLock n d := sorry
  let valid_a : CombinationLock n d → Bool := λ _ => true
  let valid_b : CombinationLock n d → Bool := is_valid_combination
  (∀ (i : Fin n), initial.discs i = 0) →
  min_moves n d initial valid_a = 999998 ∧
  min_moves n d initial valid_b = 999998 :=
sorry

end lock_min_moves_l2356_235603


namespace calculate_biology_marks_l2356_235658

theorem calculate_biology_marks (english math physics chemistry : ℕ) (average : ℚ) :
  english = 96 →
  math = 95 →
  physics = 82 →
  chemistry = 87 →
  average = 90.4 →
  (english + math + physics + chemistry + (5 * average - (english + math + physics + chemistry))) / 5 = average :=
by sorry

end calculate_biology_marks_l2356_235658


namespace gear_system_teeth_count_l2356_235665

theorem gear_system_teeth_count (teeth1 teeth2 rotations3 : ℕ) 
  (h1 : teeth1 = 32)
  (h2 : teeth2 = 24)
  (h3 : rotations3 = 8)
  (h4 : ∃ total_teeth : ℕ, 
    total_teeth % 8 = 0 ∧ 
    total_teeth > teeth1 * 4 ∧ 
    total_teeth < teeth2 * 6 ∧
    total_teeth % teeth1 = 0 ∧
    total_teeth % teeth2 = 0 ∧
    total_teeth % rotations3 = 0) :
  total_teeth / rotations3 = 17 := by
  sorry

end gear_system_teeth_count_l2356_235665


namespace expression_evaluation_l2356_235649

theorem expression_evaluation (a b c : ℝ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  (c^2 + a^2 + b)^2 - (c^2 + a^2 - b)^2 = 80 := by
  sorry

end expression_evaluation_l2356_235649


namespace gcd_14568_78452_l2356_235648

theorem gcd_14568_78452 : Int.gcd 14568 78452 = 4 := by
  sorry

end gcd_14568_78452_l2356_235648


namespace max_value_of_expression_l2356_235671

theorem max_value_of_expression (a b c : ℝ) 
  (nonneg_a : 0 ≤ a) (nonneg_b : 0 ≤ b) (nonneg_c : 0 ≤ c)
  (sum_squares : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt (3/2) ∧ 
  ∃ a₀ b₀ c₀ : ℝ, 0 ≤ a₀ ∧ 0 ≤ b₀ ∧ 0 ≤ c₀ ∧ 
    a₀^2 + b₀^2 + c₀^2 = 1 ∧
    2 * a₀ * b₀ + 2 * b₀ * c₀ * Real.sqrt 2 = Real.sqrt (3/2) :=
by sorry

end max_value_of_expression_l2356_235671


namespace m_range_l2356_235662

/-- The function f(x) = x³ - ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x + 2

/-- The function g(x) = f(x) + mx -/
def g (a m : ℝ) (x : ℝ) : ℝ := f a x + m*x

theorem m_range (a m : ℝ) : 
  (∃ x₀, ∀ x, f a x ≤ f a x₀ ∧ f a x₀ = 4) →
  (∃ x₁ ∈ Set.Ioo (-3) (a - 1), ∀ x ∈ Set.Ioo (-3) (a - 1), g a m x₁ ≤ g a m x ∧ g a m x₁ ≤ m - 1) →
  -9 < m ∧ m ≤ -15/4 := by sorry

end m_range_l2356_235662


namespace school_teachers_l2356_235670

/-- Calculates the number of teachers in a school given specific conditions -/
theorem school_teachers (students : ℕ) (classes_per_student : ℕ) (classes_per_teacher : ℕ) (students_per_class : ℕ) :
  students = 2400 →
  classes_per_student = 5 →
  classes_per_teacher = 4 →
  students_per_class = 30 →
  (students * classes_per_student) / (classes_per_teacher * students_per_class) = 100 := by
  sorry

end school_teachers_l2356_235670


namespace abs_sum_inequality_l2356_235656

theorem abs_sum_inequality (k : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 := by sorry

end abs_sum_inequality_l2356_235656


namespace minutes_before_noon_l2356_235694

theorem minutes_before_noon (x : ℕ) : x = 65 :=
  -- Define the conditions
  let minutes_between_9am_and_12pm := 180
  let minutes_ago := 20
  -- The equation: 180 - (x - 20) = 3 * (x - 20)
  have h : minutes_between_9am_and_12pm - (x - minutes_ago) = 3 * (x - minutes_ago) := by sorry
  -- Prove that x = 65
  sorry

#check minutes_before_noon

end minutes_before_noon_l2356_235694
