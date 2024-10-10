import Mathlib

namespace smallest_cube_volume_for_pyramid_l4006_400601

/-- Represents the dimensions of a rectangular pyramid -/
structure PyramidDimensions where
  height : ℝ
  baseLength : ℝ
  baseWidth : ℝ

/-- Calculates the volume of a cube given its side length -/
def cubeVolume (side : ℝ) : ℝ := side^3

/-- Theorem: The volume of the smallest cube-shaped box that can house a given rectangular pyramid uprightly -/
theorem smallest_cube_volume_for_pyramid (p : PyramidDimensions) 
  (h_height : p.height = 15)
  (h_length : p.baseLength = 8)
  (h_width : p.baseWidth = 12) :
  cubeVolume (max p.height (max p.baseLength p.baseWidth)) = 3375 := by
  sorry

#check smallest_cube_volume_for_pyramid

end smallest_cube_volume_for_pyramid_l4006_400601


namespace square_difference_l4006_400699

theorem square_difference (x y : ℚ) 
  (h1 : (x + y)^2 = 49/144) 
  (h2 : (x - y)^2 = 1/144) : 
  x^2 - y^2 = 7/144 := by
sorry

end square_difference_l4006_400699


namespace sqrt_three_squared_l4006_400669

theorem sqrt_three_squared : (Real.sqrt 3)^2 = 3 := by
  sorry

end sqrt_three_squared_l4006_400669


namespace eight_divided_by_repeating_decimal_l4006_400686

/-- The repeating decimal 0.4444... as a rational number -/
def repeating_decimal : ℚ := 4 / 9

/-- The result of 8 divided by the repeating decimal 0.4444... -/
theorem eight_divided_by_repeating_decimal : 8 / repeating_decimal = 18 := by sorry

end eight_divided_by_repeating_decimal_l4006_400686


namespace opposite_of_cube_root_eight_l4006_400647

theorem opposite_of_cube_root_eight (x : ℝ) : x^3 = 8 → -x = -2 := by sorry

end opposite_of_cube_root_eight_l4006_400647


namespace isosceles_triangle_angle_b_l4006_400663

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (sum_of_angles : A + B + C = 180)

-- Define an isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.A = t.B ∨ t.B = t.C ∨ t.A = t.C

-- Theorem statement
theorem isosceles_triangle_angle_b (t : Triangle) 
  (ext_angle_A : ℝ) 
  (h_ext_angle : ext_angle_A = 110) 
  (h_ext_prop : t.B + t.C = ext_angle_A) :
  IsIsosceles t → t.B = 70 ∨ t.B = 55 ∨ t.B = 40 := by
  sorry

end isosceles_triangle_angle_b_l4006_400663


namespace factor_x4_minus_81_l4006_400698

theorem factor_x4_minus_81 : 
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  sorry

end factor_x4_minus_81_l4006_400698


namespace arithmetic_sequence_eighth_term_l4006_400644

/-- An arithmetic sequence is a sequence where the difference between 
    each consecutive term is constant. -/
structure ArithmeticSequence where
  first : ℝ
  diff : ℝ

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + seq.diff * (n - 1)

theorem arithmetic_sequence_eighth_term 
  (seq : ArithmeticSequence) 
  (h4 : seq.nthTerm 4 = 23) 
  (h6 : seq.nthTerm 6 = 47) : 
  seq.nthTerm 8 = 71 := by
sorry


end arithmetic_sequence_eighth_term_l4006_400644


namespace sum_fourth_fifth_sixth_l4006_400606

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 2
  sum_second_third : a 2 + a 3 = 13

/-- The sum of the 4th, 5th, and 6th terms equals 42 -/
theorem sum_fourth_fifth_sixth (seq : ArithmeticSequence) :
  seq.a 4 + seq.a 5 + seq.a 6 = 42 := by
  sorry

end sum_fourth_fifth_sixth_l4006_400606


namespace midpoint_coordinates_l4006_400626

/-- Given two points M and N in a 2D plane, this theorem proves that the midpoint P of the line segment MN has specific coordinates. -/
theorem midpoint_coordinates (M N : ℝ × ℝ) (hM : M = (3, -2)) (hN : N = (-1, 0)) :
  let P := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  P = (1, -1) := by
  sorry

end midpoint_coordinates_l4006_400626


namespace pet_ownership_l4006_400642

theorem pet_ownership (total_students : ℕ) (dog_owners : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 45)
  (h2 : dog_owners = 25)
  (h3 : cat_owners = 34)
  (h4 : ∀ s, s ∈ Finset.range total_students → 
    (s ∈ Finset.range dog_owners ∨ s ∈ Finset.range cat_owners)) :
  Finset.card (Finset.range dog_owners ∩ Finset.range cat_owners) = 14 := by
  sorry

end pet_ownership_l4006_400642


namespace inscribed_square_area_l4006_400657

-- Define the parabola function
def parabola (x : ℝ) : ℝ := (x - 6)^2 - 9

-- Define the square
structure InscribedSquare where
  center : ℝ  -- x-coordinate of the square's center
  sideLength : ℝ

-- Predicate to check if a square is inscribed in the region
def isInscribed (square : InscribedSquare) : Prop :=
  let halfSide := square.sideLength / 2
  let leftX := square.center - halfSide
  let rightX := square.center + halfSide
  leftX ≥ 0 ∧ 
  rightX ≥ 0 ∧ 
  parabola rightX = -square.sideLength

-- Theorem statement
theorem inscribed_square_area :
  ∃ (square : InscribedSquare), 
    isInscribed square ∧ 
    square.sideLength^2 = 40 - 4 * Real.sqrt 10 := by
  sorry

end inscribed_square_area_l4006_400657


namespace abby_emma_weight_l4006_400687

/-- The combined weight of two people given their individual weights -/
def combined_weight (w1 w2 : ℝ) : ℝ := w1 + w2

/-- Proves that Abby and Emma weigh 310 pounds together given the weights of pairs -/
theorem abby_emma_weight
  (a b c d e : ℝ)  -- Individual weights of Abby, Bart, Cindy, Damon, and Emma
  (h1 : combined_weight a b = 270)  -- Abby and Bart
  (h2 : combined_weight b c = 255)  -- Bart and Cindy
  (h3 : combined_weight c d = 280)  -- Cindy and Damon
  (h4 : combined_weight d e = 295)  -- Damon and Emma
  : combined_weight a e = 310 := by
  sorry

#check abby_emma_weight

end abby_emma_weight_l4006_400687


namespace cylinder_radius_proof_l4006_400676

/-- The radius of a cylinder with given conditions -/
def cylinder_radius : ℝ := 12

theorem cylinder_radius_proof (h : ℝ) (r : ℝ) :
  h = 4 →
  (π * (r + 4)^2 * h = π * r^2 * (h + 4)) →
  r = cylinder_radius := by
  sorry

end cylinder_radius_proof_l4006_400676


namespace rhombus_diagonal_l4006_400636

/-- Given a rhombus with one diagonal of length 60 meters and an area of 1950 square meters,
    prove that the length of the other diagonal is 65 meters. -/
theorem rhombus_diagonal (d₁ : ℝ) (d₂ : ℝ) (area : ℝ) 
    (h₁ : d₁ = 60)
    (h₂ : area = 1950)
    (h₃ : area = (d₁ * d₂) / 2) : 
  d₂ = 65 := by
  sorry

end rhombus_diagonal_l4006_400636


namespace exam_pass_percentage_l4006_400634

/-- Given an examination where 260 students failed out of 400 total students,
    prove that 35% of students passed the examination. -/
theorem exam_pass_percentage :
  let total_students : ℕ := 400
  let failed_students : ℕ := 260
  let passed_students : ℕ := total_students - failed_students
  let pass_percentage : ℚ := (passed_students : ℚ) / (total_students : ℚ) * 100
  pass_percentage = 35 := by
  sorry

end exam_pass_percentage_l4006_400634


namespace paul_sold_94_books_l4006_400610

/-- Calculates the number of books Paul sold given his initial, purchased, and final book counts. -/
def books_sold (initial : ℕ) (purchased : ℕ) (final : ℕ) : ℕ :=
  initial + purchased - final

theorem paul_sold_94_books : books_sold 2 150 58 = 94 := by
  sorry

end paul_sold_94_books_l4006_400610


namespace total_fruit_cost_l4006_400613

def grapes_cost : ℚ := 12.08
def cherries_cost : ℚ := 9.85

theorem total_fruit_cost : grapes_cost + cherries_cost = 21.93 := by
  sorry

end total_fruit_cost_l4006_400613


namespace score_for_175_enemies_l4006_400630

def points_per_enemy : ℕ := 10

def bonus_percentage (enemies_killed : ℕ) : ℚ :=
  if enemies_killed ≥ 200 then 1
  else if enemies_killed ≥ 150 then 3/4
  else if enemies_killed ≥ 100 then 1/2
  else 0

def calculate_score (enemies_killed : ℕ) : ℕ :=
  let base_score := enemies_killed * points_per_enemy
  let bonus := (base_score : ℚ) * bonus_percentage enemies_killed
  ⌊(base_score : ℚ) + bonus⌋₊

theorem score_for_175_enemies :
  calculate_score 175 = 3063 := by sorry

end score_for_175_enemies_l4006_400630


namespace smallest_perfect_square_divisible_by_2_3_5_l4006_400651

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∀ n : ℕ, n > 0 → n.sqrt ^ 2 = n → n % 2 = 0 → n % 3 = 0 → n % 5 = 0 → n ≥ 900 :=
by
  sorry

end smallest_perfect_square_divisible_by_2_3_5_l4006_400651


namespace cell_growth_after_12_days_l4006_400628

/-- The number of cells after a given number of periods, where each cell triples every period. -/
def cell_count (initial_cells : ℕ) (periods : ℕ) : ℕ :=
  initial_cells * 3^periods

/-- The problem statement -/
theorem cell_growth_after_12_days :
  let initial_cells := 5
  let days := 12
  let period := 3
  let periods := days / period
  cell_count initial_cells periods = 135 := by
  sorry

end cell_growth_after_12_days_l4006_400628


namespace systematic_sampling_example_l4006_400678

def isSystematicSample (sample : List Nat) (totalItems : Nat) (sampleSize : Nat) : Prop :=
  sample.length = sampleSize ∧
  ∀ i, i ∈ sample → i ≤ totalItems ∧
  ∀ i j, i < j → i ∈ sample → j ∈ sample → ∃ k, j - i = k * ((totalItems - 1) / (sampleSize - 1))

theorem systematic_sampling_example :
  isSystematicSample [3, 13, 23, 33, 43] 50 5 := by
  sorry

end systematic_sampling_example_l4006_400678


namespace lottery_investment_ratio_l4006_400655

def lottery_winnings : ℕ := 12006
def savings_amount : ℕ := 1000
def fun_money : ℕ := 2802

theorem lottery_investment_ratio :
  let after_tax := lottery_winnings / 2
  let after_loans := after_tax - (after_tax / 3)
  let after_savings := after_loans - savings_amount
  let stock_investment := after_savings - fun_money
  (stock_investment : ℚ) / savings_amount = 1 / 5 := by sorry

end lottery_investment_ratio_l4006_400655


namespace number_division_l4006_400631

theorem number_division (x : ℚ) : x / 3 = 27 → x / 9 = 9 := by
  sorry

end number_division_l4006_400631


namespace correct_seating_arrangements_seating_satisfies_spacing_l4006_400679

/-- The number of ways to seat people on chairs with spacing requirements -/
def seating_arrangements (people chairs : ℕ) : ℕ :=
  if people = 3 ∧ chairs = 8 then 36 else 0

/-- Theorem stating the correct number of seating arrangements -/
theorem correct_seating_arrangements :
  seating_arrangements 3 8 = 36 := by
  sorry

/-- Theorem proving the seating arrangement satisfies the spacing requirement -/
theorem seating_satisfies_spacing (arrangement : Fin 8 → Option (Fin 3)) :
  seating_arrangements 3 8 = 36 →
  (∀ i j : Fin 3, i ≠ j →
    ∀ s t : Fin 8, arrangement s = some i ∧ arrangement t = some j →
      (s : ℕ) + 1 < (t : ℕ) ∨ (t : ℕ) + 1 < (s : ℕ)) :=
by
  sorry

end correct_seating_arrangements_seating_satisfies_spacing_l4006_400679


namespace whitney_fish_books_l4006_400617

/-- The number of books about fish Whitney bought -/
def fish_books : ℕ := 7

/-- The number of books about whales Whitney bought -/
def whale_books : ℕ := 9

/-- The number of magazines Whitney bought -/
def magazines : ℕ := 3

/-- The cost of each book in dollars -/
def book_cost : ℕ := 11

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 1

/-- The total amount Whitney spent in dollars -/
def total_spent : ℕ := 179

theorem whitney_fish_books :
  whale_books * book_cost + fish_books * book_cost + magazines * magazine_cost = total_spent :=
sorry

end whitney_fish_books_l4006_400617


namespace triangle_determines_plane_l4006_400662

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Three points are non-collinear if they do not lie on the same line -/
def NonCollinear (p1 p2 p3 : Point3D) : Prop :=
  ¬∃ (t : ℝ), p3.x = p1.x + t * (p2.x - p1.x) ∧
               p3.y = p1.y + t * (p2.y - p1.y) ∧
               p3.z = p1.z + t * (p2.z - p1.z)

/-- A plane contains a point if the point satisfies the plane equation -/
def PlaneContainsPoint (plane : Plane) (point : Point3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Theorem: Three non-collinear points uniquely determine a plane -/
theorem triangle_determines_plane (p1 p2 p3 : Point3D) 
  (h : NonCollinear p1 p2 p3) : 
  ∃! (plane : Plane), PlaneContainsPoint plane p1 ∧ 
                      PlaneContainsPoint plane p2 ∧ 
                      PlaneContainsPoint plane p3 :=
sorry

end triangle_determines_plane_l4006_400662


namespace clock_angle_at_8_30_l4006_400600

/-- The angle between the hour and minute hands at 8:30 on a standard 12-hour clock -/
def clock_angle : ℝ :=
  let numbers_on_clock : ℕ := 12
  let angle_between_numbers : ℝ := 30
  let hour_hand_position : ℝ := 8.5  -- Between 8 and 9
  let minute_hand_position : ℝ := 6
  angle_between_numbers * (minute_hand_position - hour_hand_position)

theorem clock_angle_at_8_30 : clock_angle = 75 := by
  sorry

end clock_angle_at_8_30_l4006_400600


namespace cubic_derivative_odd_implies_nonzero_l4006_400673

/-- A cubic function with a constant term of 2 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- A function is odd if f(-x) = -f(x) for all x -/
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem cubic_derivative_odd_implies_nonzero (a b c : ℝ) :
  is_odd (f' a b c) → a^2 + c^2 ≠ 0 := by sorry

end cubic_derivative_odd_implies_nonzero_l4006_400673


namespace right_triangular_pyramid_surface_area_l4006_400661

/-- A regular triangular pyramid with right-angled lateral faces -/
structure RightTriangularPyramid where
  base_edge : ℝ
  is_regular : Bool
  lateral_faces_right_angled : Bool

/-- The total surface area of a right triangular pyramid -/
def total_surface_area (p : RightTriangularPyramid) : ℝ :=
  sorry

/-- Theorem: The total surface area of a regular triangular pyramid with 
    right-angled lateral faces and base edge length 2 is 3 + √3 -/
theorem right_triangular_pyramid_surface_area :
  ∀ (p : RightTriangularPyramid), 
    p.base_edge = 2 → 
    p.is_regular = true → 
    p.lateral_faces_right_angled = true → 
    total_surface_area p = 3 + Real.sqrt 3 :=
by
  sorry

end right_triangular_pyramid_surface_area_l4006_400661


namespace syllogism_flaw_l4006_400643

theorem syllogism_flaw : ¬(∀ a : ℝ, a^2 > 0) := by sorry

end syllogism_flaw_l4006_400643


namespace chocolate_distribution_l4006_400605

/-- Represents the number of bags of chocolates initially bought by Robie -/
def initial_bags : ℕ := 3

/-- Represents the number of pieces of chocolate in each bag -/
def pieces_per_bag : ℕ := 30

/-- Represents the number of bags given to siblings -/
def bags_to_siblings : ℕ := 2

/-- Represents the number of Robie's siblings -/
def num_siblings : ℕ := 4

/-- Represents the percentage of chocolates received by the oldest sibling -/
def oldest_sibling_share : ℚ := 40 / 100

/-- Represents the percentage of chocolates received by the second oldest sibling -/
def second_oldest_sibling_share : ℚ := 30 / 100

/-- Represents the percentage of chocolates shared by the last two siblings -/
def youngest_siblings_share : ℚ := 30 / 100

/-- Represents the number of additional bags bought by Robie -/
def additional_bags : ℕ := 3

/-- Represents the discount percentage on the third additional bag -/
def discount_percentage : ℚ := 50 / 100

/-- Represents the cost of each non-discounted bag in dollars -/
def cost_per_bag : ℕ := 12

/-- Theorem stating the total amount spent, Robie's remaining chocolates, and siblings' remaining chocolates -/
theorem chocolate_distribution :
  let total_spent := initial_bags * cost_per_bag + 
                     (additional_bags - 1) * cost_per_bag + 
                     (1 - discount_percentage) * cost_per_bag
  let robie_remaining := (initial_bags - bags_to_siblings) * pieces_per_bag + 
                         additional_bags * pieces_per_bag
  let sibling_remaining := 0
  (total_spent = 66 ∧ robie_remaining = 90 ∧ sibling_remaining = 0) := by sorry

end chocolate_distribution_l4006_400605


namespace waiter_problem_l4006_400619

/-- The number of customers who left a waiter's table. -/
def customers_left (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

theorem waiter_problem :
  let initial : ℕ := 14
  let remaining : ℕ := 9
  customers_left initial remaining = 5 := by
sorry

end waiter_problem_l4006_400619


namespace largest_angle_in_ratio_triangle_l4006_400664

theorem largest_angle_in_ratio_triangle : 
  ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
    a + b + c = 180 →        -- Sum of angles in a triangle is 180°
    ∃ (x : ℝ), 
      a = 3*x ∧ b = 4*x ∧ c = 5*x →  -- Angles are in ratio 3:4:5
      max a (max b c) = 75 :=  -- The largest angle is 75°
by sorry

end largest_angle_in_ratio_triangle_l4006_400664


namespace equal_numbers_sum_l4006_400683

theorem equal_numbers_sum (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 →
  a = 12 →
  b = 25 →
  c = 18 →
  d = e →
  d + e = 45 := by
sorry

end equal_numbers_sum_l4006_400683


namespace seating_arrangement_one_between_AB_seating_arrangement_no_adjacent_empty_l4006_400621

/-- Number of students -/
def num_students : ℕ := 4

/-- Number of seats in the row -/
def num_seats : ℕ := 6

/-- Number of seating arrangements with exactly one person between A and B and no empty seats between them -/
def arrangements_with_one_between_AB : ℕ := 48

/-- Number of seating arrangements where all empty seats are not adjacent -/
def arrangements_no_adjacent_empty : ℕ := 240

/-- Theorem for the first question -/
theorem seating_arrangement_one_between_AB :
  (num_students = 4) → (num_seats = 6) →
  (arrangements_with_one_between_AB = 48) := by sorry

/-- Theorem for the second question -/
theorem seating_arrangement_no_adjacent_empty :
  (num_students = 4) → (num_seats = 6) →
  (arrangements_no_adjacent_empty = 240) := by sorry

end seating_arrangement_one_between_AB_seating_arrangement_no_adjacent_empty_l4006_400621


namespace disaster_relief_team_selection_part1_disaster_relief_team_selection_part2_l4006_400660

-- Define the number of internal medicine doctors and surgeons
def num_internal_med : ℕ := 12
def num_surgeons : ℕ := 8

-- Define the number of doctors needed for the team
def team_size : ℕ := 5

-- Theorem for part (1)
theorem disaster_relief_team_selection_part1 :
  (Nat.choose (num_internal_med + num_surgeons - 2) (team_size - 1)) = 3060 :=
sorry

-- Theorem for part (2)
theorem disaster_relief_team_selection_part2 :
  (Nat.choose (num_internal_med + num_surgeons) team_size) -
  (Nat.choose num_surgeons team_size) -
  (Nat.choose num_internal_med team_size) = 14656 :=
sorry

end disaster_relief_team_selection_part1_disaster_relief_team_selection_part2_l4006_400660


namespace number_problem_l4006_400627

theorem number_problem (n : ℝ) : (0.6 * (3/5) * n = 36) → n = 100 := by
  sorry

end number_problem_l4006_400627


namespace part_one_part_two_l4006_400620

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

-- Part 1
theorem part_one :
  A ∪ B 1 = Set.Icc (-1) 3 ∧
  (Set.univ \ B 1) = {x | x < 0 ∨ x > 3} :=
sorry

-- Part 2
theorem part_two (a : ℝ) :
  (Set.univ \ A) ∩ B a = ∅ ↔ (0 ≤ a ∧ a ≤ 1) ∨ a < -2 :=
sorry

end part_one_part_two_l4006_400620


namespace company_picnic_attendance_l4006_400632

theorem company_picnic_attendance
  (total_employees : ℕ)
  (men_percentage : ℚ)
  (women_percentage : ℚ)
  (men_attendance_rate : ℚ)
  (women_attendance_rate : ℚ)
  (h1 : men_percentage = 1/2)
  (h2 : women_percentage = 1 - men_percentage)
  (h3 : men_attendance_rate = 1/5)
  (h4 : women_attendance_rate = 2/5) :
  (men_percentage * men_attendance_rate + women_percentage * women_attendance_rate) = 3/10 := by
  sorry

end company_picnic_attendance_l4006_400632


namespace equation_solution_l4006_400616

theorem equation_solution : ∃ x : ℝ, (x + 1 ≠ 0 ∧ x^2 - 1 ≠ 0) ∧ 
  (2 * x / (x + 1) - 2 = 3 / (x^2 - 1)) ∧ x = -1/2 := by
  sorry

end equation_solution_l4006_400616


namespace hulk_seventh_jump_exceeds_1500_l4006_400612

def hulk_jump (n : ℕ) : ℝ :=
  3 * (3 ^ (n - 1))

theorem hulk_seventh_jump_exceeds_1500 :
  (∀ k < 7, hulk_jump k ≤ 1500) ∧ hulk_jump 7 > 1500 :=
sorry

end hulk_seventh_jump_exceeds_1500_l4006_400612


namespace fruit_crate_total_l4006_400607

theorem fruit_crate_total (strawberry_count : ℕ) (kiwi_fraction : ℚ) 
  (h1 : kiwi_fraction = 1/3)
  (h2 : strawberry_count = 52) :
  ∃ (total : ℕ), total = 78 ∧ 
    strawberry_count = (1 - kiwi_fraction) * total ∧
    kiwi_fraction * total + strawberry_count = total := by
  sorry

end fruit_crate_total_l4006_400607


namespace cubic_roots_reciprocal_squares_sum_l4006_400671

theorem cubic_roots_reciprocal_squares_sum : 
  ∀ a b c : ℝ, 
  (a^3 - 8*a^2 + 15*a - 7 = 0) → 
  (b^3 - 8*b^2 + 15*b - 7 = 0) → 
  (c^3 - 8*c^2 + 15*c - 7 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1/a^2 + 1/b^2 + 1/c^2 = 113/49) :=
by sorry

end cubic_roots_reciprocal_squares_sum_l4006_400671


namespace sum_of_star_tip_angles_l4006_400682

/-- The angle measurement of one tip of an 8-pointed star formed by connecting
    eight evenly spaced points on a circle -/
def star_tip_angle : ℝ := 67.5

/-- The number of tips in an 8-pointed star -/
def num_tips : ℕ := 8

/-- Theorem: The sum of the angle measurements of the eight tips of an 8-pointed star,
    formed by connecting eight evenly spaced points on a circle, is equal to 540° -/
theorem sum_of_star_tip_angles :
  (num_tips : ℝ) * star_tip_angle = 540 := by sorry

end sum_of_star_tip_angles_l4006_400682


namespace cost_per_set_is_20_verify_profit_equation_l4006_400609

/-- Represents the manufacturing and sales scenario of horseshoe sets -/
structure HorseshoeManufacturing where
  initialOutlay : ℝ
  sellingPrice : ℝ
  setsSold : ℕ
  profit : ℝ
  costPerSet : ℝ

/-- The cost per set is $20 given the specified conditions -/
theorem cost_per_set_is_20 (h : HorseshoeManufacturing) 
    (h_initialOutlay : h.initialOutlay = 10000)
    (h_sellingPrice : h.sellingPrice = 50)
    (h_setsSold : h.setsSold = 500)
    (h_profit : h.profit = 5000) :
    h.costPerSet = 20 := by
  sorry

/-- Verifies that the calculated cost per set satisfies the profit equation -/
theorem verify_profit_equation (h : HorseshoeManufacturing) 
    (h_initialOutlay : h.initialOutlay = 10000)
    (h_sellingPrice : h.sellingPrice = 50)
    (h_setsSold : h.setsSold = 500)
    (h_profit : h.profit = 5000)
    (h_costPerSet : h.costPerSet = 20) :
    h.profit = h.sellingPrice * h.setsSold - (h.initialOutlay + h.costPerSet * h.setsSold) := by
  sorry

end cost_per_set_is_20_verify_profit_equation_l4006_400609


namespace x_squared_plus_3xy_plus_y_squared_l4006_400665

theorem x_squared_plus_3xy_plus_y_squared (x y : ℝ) 
  (h1 : x * y = -3) 
  (h2 : x + y = -4) : 
  x^2 + 3*x*y + y^2 = 13 := by
sorry

end x_squared_plus_3xy_plus_y_squared_l4006_400665


namespace park_route_length_l4006_400629

/-- A bike route in a park -/
structure BikeRoute where
  horizontal_segments : List Float
  vertical_segments : List Float

/-- The total length of a bike route -/
def total_length (route : BikeRoute) : Float :=
  2 * (route.horizontal_segments.sum + route.vertical_segments.sum)

/-- The specific bike route described in the problem -/
def park_route : BikeRoute :=
  { horizontal_segments := [4, 7, 2],
    vertical_segments := [6, 7] }

theorem park_route_length :
  total_length park_route = 52 := by
  sorry

#eval total_length park_route

end park_route_length_l4006_400629


namespace prime_square_minus_one_divisible_by_twelve_l4006_400659

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) 
  (h_prime : Nat.Prime p) (h_ge_five : p ≥ 5) : 
  12 ∣ p^2 - 1 := by
  sorry

end prime_square_minus_one_divisible_by_twelve_l4006_400659


namespace original_to_circle_l4006_400623

/-- The original curve in polar coordinates -/
def original_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (3 * (Real.cos θ)^2 + 4 * (Real.sin θ)^2)

/-- The transformation applied to the curve -/
def transformation (x y x'' y'' : ℝ) : Prop :=
  x'' = (1/2) * x ∧ y'' = (Real.sqrt 3 / 3) * y

/-- The resulting curve after transformation -/
def resulting_curve (x'' y'' : ℝ) : Prop :=
  x''^2 + y''^2 = 1

/-- Theorem stating that the original curve transforms into a circle -/
theorem original_to_circle :
  ∀ (ρ θ x y x'' y'' : ℝ),
    original_curve ρ θ →
    x = ρ * Real.cos θ →
    y = ρ * Real.sin θ →
    transformation x y x'' y'' →
    resulting_curve x'' y'' :=
sorry

end original_to_circle_l4006_400623


namespace BaCl2_mass_produced_l4006_400618

-- Define the molar masses
def molar_mass_BaCl2 : ℝ := 208.23

-- Define the initial amounts
def initial_BaCl2_moles : ℝ := 8
def initial_NaOH_moles : ℝ := 12

-- Define the stoichiometric ratios
def ratio_NaOH_to_BaCl2 : ℝ := 2
def ratio_BaOH2_to_BaCl2 : ℝ := 1

-- Define the theorem
theorem BaCl2_mass_produced : 
  let BaCl2_produced := min initial_BaCl2_moles (initial_NaOH_moles / ratio_NaOH_to_BaCl2)
  BaCl2_produced * molar_mass_BaCl2 = 1665.84 :=
by sorry

end BaCl2_mass_produced_l4006_400618


namespace least_subtraction_for_divisibility_l4006_400672

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 9 ∧ (427751 - k) % 10 = 0 ∧ 
  ∀ (m : ℕ), m < k → (427751 - m) % 10 ≠ 0 :=
sorry

end least_subtraction_for_divisibility_l4006_400672


namespace jelly_cost_l4006_400668

/-- Proof of the cost of jelly given bread, peanut butter, and leftover money -/
theorem jelly_cost (bread_price : ℚ) (bread_quantity : ℕ) (peanut_butter_price : ℚ) 
  (total_money : ℚ) (leftover_money : ℚ) :
  bread_price = 2.25 →
  bread_quantity = 3 →
  peanut_butter_price = 2 →
  total_money = 14 →
  leftover_money = 5.25 →
  leftover_money = total_money - (bread_price * bread_quantity + peanut_butter_price) :=
by
  sorry

#check jelly_cost

end jelly_cost_l4006_400668


namespace danny_larry_score_difference_l4006_400625

theorem danny_larry_score_difference :
  ∀ (keith larry danny : ℕ),
    keith = 3 →
    larry = 3 * keith →
    danny > larry →
    keith + larry + danny = 26 →
    danny - larry = 5 := by
  sorry

end danny_larry_score_difference_l4006_400625


namespace triangle_side_product_l4006_400694

theorem triangle_side_product (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = pi) →
  ((a + b)^2 - c^2 = 4) →
  (C = pi / 3) →
  (a * b = 4 / 3) :=
by sorry

end triangle_side_product_l4006_400694


namespace function_satisfies_conditions_l4006_400684

theorem function_satisfies_conditions (m n : ℕ) : 
  let f : ℕ → ℕ → ℕ := λ m n => m * n
  (m ≥ 1 ∧ n ≥ 1 → 2 * (f m n) = 2 + f (m + 1) (n - 1) + f (m - 1) (n + 1)) ∧
  f m 0 = 0 ∧ f 0 n = 0 := by
  sorry

end function_satisfies_conditions_l4006_400684


namespace nathan_bananas_l4006_400692

/-- The number of bananas Nathan has, given the specified bunches -/
def total_bananas (bunches_of_eight : Nat) (bananas_per_bunch_eight : Nat)
                  (bunches_of_seven : Nat) (bananas_per_bunch_seven : Nat) : Nat :=
  bunches_of_eight * bananas_per_bunch_eight + bunches_of_seven * bananas_per_bunch_seven

/-- Proof that Nathan has 83 bananas given the specified bunches -/
theorem nathan_bananas :
  total_bananas 6 8 5 7 = 83 := by
  sorry

end nathan_bananas_l4006_400692


namespace inequality_solution_l4006_400690

theorem inequality_solution (x : ℝ) : 2*x + 6 > 5*x - 3 → x < 3 := by
  sorry

end inequality_solution_l4006_400690


namespace beadshop_profit_l4006_400611

theorem beadshop_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ)
  (h_total : total_profit = 1200)
  (h_monday : monday_fraction = 1/3)
  (h_tuesday : tuesday_fraction = 1/4) :
  total_profit - (monday_fraction * total_profit + tuesday_fraction * total_profit) = 500 := by
  sorry

end beadshop_profit_l4006_400611


namespace train_crossing_lamppost_l4006_400603

/-- Calculates the time for a train to cross a lamp post given its length, bridge length, and time to cross the bridge -/
theorem train_crossing_lamppost 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (bridge_crossing_time : ℝ) 
  (h1 : train_length = 400)
  (h2 : bridge_length = 800)
  (h3 : bridge_crossing_time = 45)
  : (train_length * bridge_crossing_time) / bridge_length = 22.5 := by
  sorry

end train_crossing_lamppost_l4006_400603


namespace subtracted_number_l4006_400633

theorem subtracted_number (x : ℚ) : x = 40 → ∃ y : ℚ, ((x / 4) * 5 + 10) - y = 48 ∧ y = 12 := by
  sorry

end subtracted_number_l4006_400633


namespace fourth_side_distance_l4006_400639

/-- A square with a point inside it -/
structure SquareWithPoint where
  side_length : ℝ
  d1 : ℝ
  d2 : ℝ
  d3 : ℝ
  d4 : ℝ
  h_positive : 0 < side_length
  h_inside : d1 + d2 + d3 + d4 = side_length
  h_d1 : 0 < d1
  h_d2 : 0 < d2
  h_d3 : 0 < d3
  h_d4 : 0 < d4

/-- The theorem stating the possible distances to the fourth side -/
theorem fourth_side_distance (s : SquareWithPoint) 
  (h1 : s.d1 = 4)
  (h2 : s.d2 = 7)
  (h3 : s.d3 = 13) :
  s.d4 = 10 ∨ s.d4 = 16 := by
  sorry

end fourth_side_distance_l4006_400639


namespace increasing_perfect_powers_sum_l4006_400681

def s (n : ℕ+) : ℕ := sorry

theorem increasing_perfect_powers_sum (x : ℝ) :
  ∃ N : ℕ, ∀ n > N, (Finset.range n).sup (fun i => s ⟨i + 1, Nat.succ_pos i⟩) / n > x := by
  sorry

end increasing_perfect_powers_sum_l4006_400681


namespace no_triangle_tangent_to_both_curves_l4006_400650

/-- C₁ is the unit circle -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- C₂ is an ellipse with semi-major axis a and semi-minor axis b -/
def C₂ (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- A triangle is externally tangent to C₁ if all its vertices lie outside or on C₁ 
    and each side is tangent to C₁ -/
def externally_tangent_C₁ (A B C : ℝ × ℝ) : Prop := sorry

/-- A triangle is internally tangent to C₂ if all its vertices lie inside or on C₂ 
    and each side is tangent to C₂ -/
def internally_tangent_C₂ (a b : ℝ) (A B C : ℝ × ℝ) : Prop := sorry

theorem no_triangle_tangent_to_both_curves (a b : ℝ) :
  a > b ∧ b > 0 ∧ C₂ a b 1 1 →
  ¬ ∃ (A B C : ℝ × ℝ), externally_tangent_C₁ A B C ∧ internally_tangent_C₂ a b A B C :=
by sorry

end no_triangle_tangent_to_both_curves_l4006_400650


namespace solve_trailer_problem_l4006_400608

/-- Represents the trailer home problem --/
def trailer_problem (initial_count : ℕ) (initial_avg_age : ℕ) (current_avg_age : ℕ) (time_elapsed : ℕ) : Prop :=
  ∃ (new_count : ℕ),
    (initial_count * (initial_avg_age + time_elapsed) + new_count * time_elapsed) / (initial_count + new_count) = current_avg_age

/-- The theorem statement for the trailer home problem --/
theorem solve_trailer_problem :
  trailer_problem 30 15 10 3 → ∃ (new_count : ℕ), new_count = 34 :=
by
  sorry


end solve_trailer_problem_l4006_400608


namespace unique_solution_to_equation_l4006_400670

theorem unique_solution_to_equation :
  ∃! x : ℝ, x ≠ -1 ∧ x ≠ -3 ∧
  (x^3 + 3*x^2 - x) / (x^2 + 4*x + 3) + x = -7 ∧
  x = -5 := by
  sorry

end unique_solution_to_equation_l4006_400670


namespace johns_salary_increase_l4006_400667

theorem johns_salary_increase (original_salary new_salary : ℝ) 
  (h1 : new_salary = 110)
  (h2 : new_salary = original_salary * (1 + 0.8333333333333334)) : 
  original_salary = 60 := by
sorry

end johns_salary_increase_l4006_400667


namespace goose_eggs_laid_l4006_400624

theorem goose_eggs_laid (
  hatch_rate : ℚ)
  (first_month_survival : ℚ)
  (first_six_months_death : ℚ)
  (first_year_death : ℚ)
  (survived_first_year : ℕ)
  (h1 : hatch_rate = 3 / 7)
  (h2 : first_month_survival = 5 / 9)
  (h3 : first_six_months_death = 11 / 16)
  (h4 : first_year_death = 7 / 12)
  (h5 : survived_first_year = 84) :
  ∃ (eggs : ℕ), eggs ≥ 678 ∧
    (eggs : ℚ) * hatch_rate * first_month_survival * (1 - first_six_months_death) * (1 - first_year_death) = survived_first_year :=
by sorry

end goose_eggs_laid_l4006_400624


namespace simplify_trig_expression_l4006_400637

theorem simplify_trig_expression (a : Real) (h : 0 < a ∧ a < π / 2) :
  Real.sqrt (1 + Real.sin a) + Real.sqrt (1 - Real.sin a) - Real.sqrt (2 + 2 * Real.cos a) = 0 := by
  sorry

end simplify_trig_expression_l4006_400637


namespace sally_next_birthday_l4006_400677

theorem sally_next_birthday (adam mary sally danielle : ℝ) 
  (h1 : adam = 1.3 * mary)
  (h2 : mary = 0.75 * sally)
  (h3 : sally = 0.8 * danielle)
  (h4 : adam + mary + sally + danielle = 60) :
  ⌈sally⌉ = 16 := by
  sorry

end sally_next_birthday_l4006_400677


namespace sin_negative_120_degrees_l4006_400653

theorem sin_negative_120_degrees : Real.sin (-(120 * π / 180)) = -Real.sqrt 3 / 2 := by
  sorry

end sin_negative_120_degrees_l4006_400653


namespace converse_proposition_l4006_400604

theorem converse_proposition :
  let P : Prop := x ≥ 2 ∧ y ≥ 3
  let Q : Prop := x + y ≥ 5
  let original : Prop := P → Q
  let converse : Prop := Q → P
  converse = (x + y ≥ 5 → x ≥ 2 ∧ y ≥ 3) := by
sorry

end converse_proposition_l4006_400604


namespace complex_absolute_value_l4006_400602

theorem complex_absolute_value (z : ℂ) : z = 10 + 3*I → Complex.abs (z^2 + 8*z + 85) = 4 * Real.sqrt 3922 := by
  sorry

end complex_absolute_value_l4006_400602


namespace hypotenuse_square_l4006_400675

/-- Given complex numbers a, b, and c that are zeros of a polynomial P(z) = z^3 + pz + q,
    and satisfy |a|^2 + |b|^2 + |c|^2 = 360, if the points corresponding to a, b, and c
    form a right triangle with the right angle at a, then the square of the length of
    the hypotenuse is 432. -/
theorem hypotenuse_square (a b c : ℂ) (p q : ℂ) :
  (a^3 + p*a + q = 0) →
  (b^3 + p*b + q = 0) →
  (c^3 + p*c + q = 0) →
  (Complex.abs a)^2 + (Complex.abs b)^2 + (Complex.abs c)^2 = 360 →
  (b - a) • (c - a) = 0 →  -- Right angle at a
  (Complex.abs (b - c))^2 = 432 := by
  sorry

end hypotenuse_square_l4006_400675


namespace tangent_circle_radius_l4006_400680

theorem tangent_circle_radius (R : ℝ) (chord_length : ℝ) (ratio : ℝ) :
  R = 5 →
  chord_length = 8 →
  ratio = 1/3 →
  ∃ (r₁ r₂ : ℝ), (r₁ = 8/9 ∧ r₂ = 32/9) ∧
    (∀ (r : ℝ), (r = r₁ ∨ r = r₂) ↔
      (∃ (C : ℝ × ℝ),
        C.1^2 + C.2^2 = R^2 ∧
        C.1^2 + (C.2 - chord_length * ratio)^2 = r^2 ∧
        (R - r)^2 = (r + C.2)^2 + C.1^2)) :=
by sorry


end tangent_circle_radius_l4006_400680


namespace power_product_rule_l4006_400666

theorem power_product_rule (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end power_product_rule_l4006_400666


namespace aspirin_percentage_of_max_dosage_l4006_400652

-- Define the medication schedule and dosages
def aspirin_dosage : ℕ := 325
def aspirin_frequency : ℕ := 12
def aspirin_max_dosage : ℕ := 4000
def hours_per_day : ℕ := 24

-- Define the function to calculate total daily dosage
def total_daily_dosage (dosage frequency : ℕ) : ℕ :=
  dosage * (hours_per_day / frequency)

-- Define the function to calculate percentage of max dosage
def percentage_of_max_dosage (daily_dosage max_dosage : ℕ) : ℚ :=
  (daily_dosage : ℚ) / (max_dosage : ℚ) * 100

-- Theorem statement
theorem aspirin_percentage_of_max_dosage :
  percentage_of_max_dosage 
    (total_daily_dosage aspirin_dosage aspirin_frequency) 
    aspirin_max_dosage = 16.25 := by
  sorry

end aspirin_percentage_of_max_dosage_l4006_400652


namespace roots_on_circle_l4006_400614

theorem roots_on_circle (a : ℝ) : 
  (∃ (z₁ z₂ z₃ z₄ : ℂ), z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    (z₁^2 - 2*z₁ + 5)*(z₁^2 + 2*a*z₁ + 1) = 0 ∧
    (z₂^2 - 2*z₂ + 5)*(z₂^2 + 2*a*z₂ + 1) = 0 ∧
    (z₃^2 - 2*z₃ + 5)*(z₃^2 + 2*a*z₃ + 1) = 0 ∧
    (z₄^2 - 2*z₄ + 5)*(z₄^2 + 2*a*z₄ + 1) = 0 ∧
    ∃ (c : ℂ) (r : ℝ), r > 0 ∧ 
      Complex.abs (z₁ - c) = r ∧
      Complex.abs (z₂ - c) = r ∧
      Complex.abs (z₃ - c) = r ∧
      Complex.abs (z₄ - c) = r) ↔
  (a > -1 ∧ a < 1) ∨ a = -3 :=
by sorry

end roots_on_circle_l4006_400614


namespace min_value_x2_plus_2y2_l4006_400688

theorem min_value_x2_plus_2y2 (x y : ℝ) (h : x^2 - x*y + y^2 = 1) :
  ∃ (m : ℝ), m = (6 - 2*Real.sqrt 3) / 3 ∧ ∀ (a b : ℝ), a^2 - a*b + b^2 = 1 → x^2 + 2*y^2 ≥ m :=
sorry

end min_value_x2_plus_2y2_l4006_400688


namespace expression_simplification_and_evaluation_l4006_400658

theorem expression_simplification_and_evaluation :
  let f (a : ℝ) := a / (a - 1) + (a + 1) / (a^2 - 1)
  let g (a : ℝ) := (a + 1) / (a - 1)
  ∀ a : ℝ, a^2 - 1 ≠ 0 →
    f a = g a ∧
    (a = 0 → g a = -1) :=
by sorry

end expression_simplification_and_evaluation_l4006_400658


namespace fraction_calculation_l4006_400635

theorem fraction_calculation : 
  (1/4 + 1/5) / (3/7 - 1/8) = 42/25 := by sorry

end fraction_calculation_l4006_400635


namespace x_eq_y_sufficient_not_necessary_for_abs_x_eq_abs_y_l4006_400654

theorem x_eq_y_sufficient_not_necessary_for_abs_x_eq_abs_y :
  (∀ x y : ℝ, x = y → |x| = |y|) ∧
  (∃ x y : ℝ, |x| = |y| ∧ x ≠ y) := by
  sorry

end x_eq_y_sufficient_not_necessary_for_abs_x_eq_abs_y_l4006_400654


namespace height_sum_l4006_400674

/-- Given the heights of John, Lena, and Rebeca, prove that the sum of Lena's and Rebeca's heights is 295 cm. -/
theorem height_sum (john_height lena_height rebeca_height : ℕ) 
  (h1 : john_height = 152)
  (h2 : john_height = lena_height + 15)
  (h3 : rebeca_height = john_height + 6) :
  lena_height + rebeca_height = 295 := by
  sorry

end height_sum_l4006_400674


namespace raduzhny_population_l4006_400646

/-- The number of villages in Sunny Valley -/
def num_villages : ℕ := 10

/-- The population of Znoynoe village -/
def znoynoe_population : ℕ := 1000

/-- The amount by which Znoynoe's population exceeds the average -/
def excess_population : ℕ := 90

/-- The total population of all villages in Sunny Valley -/
def total_population : ℕ := znoynoe_population + (num_villages - 1) * (znoynoe_population - excess_population)

/-- The average population of villages in Sunny Valley -/
def average_population : ℕ := total_population / num_villages

theorem raduzhny_population : 
  ∃ (raduzhny_pop : ℕ), 
    raduzhny_pop = average_population ∧ 
    raduzhny_pop = 900 :=
sorry

end raduzhny_population_l4006_400646


namespace inequality_proof_l4006_400645

theorem inequality_proof (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_condition : a + b + c ≤ 3) : 
  (3 > (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1)) ∧ 
   (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1)) ≥ 3 / 2) ∧
  ((a + 1) / (a * (a + 2)) + (b + 1) / (b * (b + 2)) + (c + 1) / (c * (c + 2)) ≥ 2) := by
  sorry

end inequality_proof_l4006_400645


namespace arithmetic_sequence_fourth_term_l4006_400638

/-- 
Given an arithmetic sequence {a_n} with S_n as the sum of its first n terms,
prove that if S_6 = 24 and S_9 = 63, then a_4 = 5.
-/
theorem arithmetic_sequence_fourth_term 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) 
  (h_S6 : S 6 = 24) 
  (h_S9 : S 9 = 63) : 
  a 4 = 5 := by sorry

end arithmetic_sequence_fourth_term_l4006_400638


namespace courtyard_paving_l4006_400693

/-- The number of bricks required to pave a rectangular courtyard -/
def bricks_required (courtyard_length courtyard_width brick_length brick_width : ℚ) : ℚ :=
  (courtyard_length * courtyard_width) / (brick_length * brick_width)

/-- Theorem stating the number of bricks required for the specific courtyard and brick sizes -/
theorem courtyard_paving :
  bricks_required 25 15 0.2 0.1 = 18750 := by
  sorry

end courtyard_paving_l4006_400693


namespace probability_even_and_prime_on_two_dice_l4006_400622

/-- A die is a finite set of natural numbers from 1 to 6 -/
def Die : Finset ℕ := Finset.range 6

/-- Even numbers on a die -/
def EvenNumbers : Finset ℕ := {2, 4, 6}

/-- Prime numbers on a die -/
def PrimeNumbers : Finset ℕ := {2, 3, 5}

/-- The probability of an event occurring in a finite sample space -/
def probability (event : Finset ℕ) (sampleSpace : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (sampleSpace.card : ℚ)

theorem probability_even_and_prime_on_two_dice : 
  (probability EvenNumbers Die) * (probability PrimeNumbers Die) = 1 / 4 := by
  sorry

end probability_even_and_prime_on_two_dice_l4006_400622


namespace slower_train_speed_l4006_400615

/-- Proves that the speed of the slower train is 36 km/hr given the problem conditions -/
theorem slower_train_speed (faster_speed : ℝ) (passing_time : ℝ) (train_length : ℝ)
  (h1 : faster_speed = 46)
  (h2 : passing_time = 54)
  (h3 : train_length = 75) :
  ∃ (slower_speed : ℝ),
    slower_speed = 36 ∧
    (faster_speed - slower_speed) * passing_time * (1000 / 3600) = 2 * train_length :=
by
  sorry

#check slower_train_speed

end slower_train_speed_l4006_400615


namespace theater_ticket_area_l4006_400696

/-- The area of a rectangular theater ticket -/
theorem theater_ticket_area (perimeter width : ℝ) (h1 : perimeter = 28) (h2 : width = 6) :
  let length := (perimeter - 2 * width) / 2
  width * length = 48 := by
  sorry

end theater_ticket_area_l4006_400696


namespace smallest_k_value_l4006_400695

theorem smallest_k_value (m n k : ℤ) (h : 221 * m + 247 * n + 323 * k = 2001) :
  (k > 100 ∧ ∀ k' > 100, 221 * m + 247 * n + 323 * k' = 2001 → k ≤ k') → k = 111 := by
  sorry

end smallest_k_value_l4006_400695


namespace coefficient_x_squared_in_product_l4006_400649

def p (x : ℝ) : ℝ := -3 * x^3 - 4 * x^2 - 8 * x + 2
def q (x : ℝ) : ℝ := -2 * x^2 - 7 * x + 3

theorem coefficient_x_squared_in_product :
  ∃ (a b c d e : ℝ), p x * q x = a * x^4 + b * x^3 + 40 * x^2 + d * x + e :=
sorry

end coefficient_x_squared_in_product_l4006_400649


namespace cube_split_theorem_l4006_400640

/-- Given a natural number m > 1, returns the first odd number in the split of m³ -/
def firstSplitNumber (m : ℕ) : ℕ := m * (m - 1) + 1

/-- Given a natural number m > 1, returns the list of odd numbers in the split of m³ -/
def splitNumbers (m : ℕ) : List ℕ :=
  List.range m |>.map (λ i => firstSplitNumber m + 2 * i)

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) (h2 : 333 ∈ splitNumbers m) : m = 18 := by
  sorry

end cube_split_theorem_l4006_400640


namespace expression_values_l4006_400697

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ (k : ℤ), k ∈ ({5, 2, 1, -2, -3} : Set ℤ) ∧
  (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d| = k) := by
  sorry

end expression_values_l4006_400697


namespace max_value_constraint_l4006_400641

theorem max_value_constraint (x y : ℝ) (h : 9*x^2 + y^2 + x*y = 1) :
  ∃ (M : ℝ), M = (2 * Real.sqrt 21) / 7 ∧ 3*x + y ≤ M :=
sorry

end max_value_constraint_l4006_400641


namespace simplify_expression_l4006_400689

theorem simplify_expression (s t : ℝ) : 105 * s - 37 * s + 18 * t = 68 * s + 18 * t := by
  sorry

end simplify_expression_l4006_400689


namespace shelf_filling_l4006_400648

theorem shelf_filling (A H C S M N E : ℕ) (x y z : ℝ) (l : ℝ) 
  (hA : A > 0) (hH : H > 0) (hC : C > 0) (hS : S > 0) (hM : M > 0) (hN : N > 0) (hE : E > 0)
  (hDistinct : A ≠ H ∧ A ≠ C ∧ A ≠ S ∧ A ≠ M ∧ A ≠ N ∧ A ≠ E ∧
               H ≠ C ∧ H ≠ S ∧ H ≠ M ∧ H ≠ N ∧ H ≠ E ∧
               C ≠ S ∧ C ≠ M ∧ C ≠ N ∧ C ≠ E ∧
               S ≠ M ∧ S ≠ N ∧ S ≠ E ∧
               M ≠ N ∧ M ≠ E ∧
               N ≠ E)
  (hThickness : 0 < x ∧ x < y ∧ x < z)
  (hFill1 : A * x + H * y + C * z = l)
  (hFill2 : S * x + M * y + N * z = l)
  (hFill3 : E * x = l) :
  E = (A * M + C * N - S * H - N * H) / (M + N - H) :=
sorry

end shelf_filling_l4006_400648


namespace daughters_and_granddaughters_without_children_l4006_400691

/-- Represents the family structure of Marilyn and her descendants -/
structure FamilyStructure where
  daughters : ℕ
  granddaughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of daughters each daughter with children has -/
def daughters_per_mother : ℕ := 5

/-- Axioms representing the given conditions -/
axiom marilyn : FamilyStructure
axiom marilyn_daughters : marilyn.daughters = 10
axiom marilyn_total : marilyn.total_descendants = 40
axiom marilyn_granddaughters : marilyn.granddaughters = marilyn.total_descendants - marilyn.daughters
axiom marilyn_daughters_with_children : 
  marilyn.daughters_with_children * daughters_per_mother = marilyn.granddaughters

/-- The main theorem to prove -/
theorem daughters_and_granddaughters_without_children : 
  marilyn.granddaughters + (marilyn.daughters - marilyn.daughters_with_children) = 34 := by
  sorry

end daughters_and_granddaughters_without_children_l4006_400691


namespace linlins_speed_l4006_400685

/-- Proves that Linlin's speed is 400 meters per minute given the problem conditions --/
theorem linlins_speed (total_distance : ℕ) (time_taken : ℕ) (qingqing_speed : ℕ) :
  total_distance = 3290 →
  time_taken = 7 →
  qingqing_speed = 70 →
  (total_distance / time_taken - qingqing_speed : ℕ) = 400 :=
by sorry

end linlins_speed_l4006_400685


namespace smaller_number_problem_l4006_400656

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) :
  min x y = 3 := by sorry

end smaller_number_problem_l4006_400656
