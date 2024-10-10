import Mathlib

namespace inequality_solution_equivalence_l1156_115652

-- Define the types for our variables
variable (x : ℝ)
variable (a b : ℝ)

-- Define the original inequality and its solution set
def original_inequality (x a b : ℝ) : Prop := a * (x + b) * (x + 5 / a) > 0
def original_solution_set (x : ℝ) : Prop := x < -1 ∨ x > 3

-- Define the new inequality we want to solve
def new_inequality (x a b : ℝ) : Prop := x^2 + b*x - 2*a < 0

-- Define the solution set we want to prove
def target_solution_set (x : ℝ) : Prop := x > -2 ∧ x < 5

-- State the theorem
theorem inequality_solution_equivalence :
  (∀ x, original_inequality x a b ↔ original_solution_set x) →
  (∀ x, new_inequality x a b ↔ target_solution_set x) :=
sorry

end inequality_solution_equivalence_l1156_115652


namespace calculate_expression_largest_integer_solution_three_is_largest_integer_solution_l1156_115630

-- Part 1
theorem calculate_expression : 4 * Real.sin (π / 3) - |-1| + (Real.sqrt 3 - 1)^0 + Real.sqrt 48 = 6 * Real.sqrt 3 := by
  sorry

-- Part 2
theorem largest_integer_solution (x : ℝ) :
  (1/2 * (x - 1) ≤ 1 ∧ 1 - x < 2) → x ≤ 3 := by
  sorry

theorem three_is_largest_integer_solution :
  ∃ (x : ℤ), x = 3 ∧ (1/2 * (x - 1) ≤ 1 ∧ 1 - x < 2) ∧
  ∀ (y : ℤ), y > 3 → ¬(1/2 * (y - 1) ≤ 1 ∧ 1 - y < 2) := by
  sorry

end calculate_expression_largest_integer_solution_three_is_largest_integer_solution_l1156_115630


namespace sqrt_2x_minus_4_meaningful_l1156_115619

theorem sqrt_2x_minus_4_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 :=
by sorry

end sqrt_2x_minus_4_meaningful_l1156_115619


namespace road_network_impossibility_l1156_115643

/-- Represents an intersection in the road network -/
structure Intersection where
  branches : ℕ
  (branch_count : branches ≥ 2)

/-- Represents the road network -/
structure RoadNetwork where
  A : Intersection
  B : Intersection
  C : Intersection
  k_A : ℕ
  k_B : ℕ
  k_C : ℕ
  (k_A_def : A.branches = k_A)
  (k_B_def : B.branches = k_B)
  (k_C_def : C.branches = k_C)

/-- Total number of toll stations in the network -/
def total_toll_stations (rn : RoadNetwork) : ℕ :=
  4 + 4 * (rn.k_A + rn.k_B + rn.k_C)

/-- Theorem stating the impossibility of the road network design -/
theorem road_network_impossibility (rn : RoadNetwork) :
  ¬ ∃ (distances : Finset ℕ), 
    distances.card = (total_toll_stations rn).choose 2 ∧ 
    (∀ i ∈ distances, i ≤ distances.card) ∧
    (∀ i ≤ distances.card, i ∈ distances) :=
sorry

end road_network_impossibility_l1156_115643


namespace sum_interior_angles_octagon_l1156_115600

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- An octagon is a polygon with 8 sides -/
def octagon_sides : ℕ := 8

/-- Theorem: The sum of the interior angles of an octagon is 1080 degrees -/
theorem sum_interior_angles_octagon :
  sum_interior_angles octagon_sides = 1080 := by sorry

end sum_interior_angles_octagon_l1156_115600


namespace square_sum_constant_l1156_115648

theorem square_sum_constant (x : ℝ) : (2*x + 3)^2 + 2*(2*x + 3)*(5 - 2*x) + (5 - 2*x)^2 = 64 := by
  sorry

end square_sum_constant_l1156_115648


namespace mod_37_5_l1156_115617

theorem mod_37_5 : 37 % 5 = 2 := by
  sorry

end mod_37_5_l1156_115617


namespace at_least_two_inequalities_false_l1156_115660

theorem at_least_two_inequalities_false (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  ¬(((x + y > 0) ∧ (y + z > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0)) ∨
    ((x + y > 0) ∧ (y + z > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (z + 2*x < 0)) ∨
    ((x + y > 0) ∧ (y + z > 0) ∧ (z + x > 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0)) ∨
    ((x + y > 0) ∧ (y + z > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0)) ∨
    ((x + y > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0)) ∨
    ((y + z > 0) ∧ (z + x > 0) ∧ (x + 2*y < 0) ∧ (y + 2*z < 0) ∧ (z + 2*x < 0))) :=
by
  sorry


end at_least_two_inequalities_false_l1156_115660


namespace quadratic_inequality_solution_l1156_115604

/-- Given a quadratic inequality ax² + bx + 2 > 0 with solution set {x | -1/2 < x < 1/3},
    prove that a + b = -14 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) →
  a + b = -14 :=
sorry

end quadratic_inequality_solution_l1156_115604


namespace matrix_square_result_l1156_115636

theorem matrix_square_result (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : M.mulVec ![1, 0] = ![1, 0])
  (h2 : M.mulVec ![1, 1] = ![2, 2]) :
  (M ^ 2).mulVec ![1, -1] = ![-2, -4] := by
  sorry

end matrix_square_result_l1156_115636


namespace square_difference_theorem_l1156_115634

theorem square_difference_theorem (a b A : ℝ) : 
  (5*a + 3*b)^2 = (5*a - 3*b)^2 + A → A = 60*a*b := by
sorry

end square_difference_theorem_l1156_115634


namespace line_and_points_l1156_115692

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = -2

-- Define the points
def point_A : ℝ × ℝ := (2, -2)
def point_B : ℝ × ℝ := (3, 2)

-- Theorem statement
theorem line_and_points :
  (∀ x y : ℝ, line_equation x y → y = -2) ∧  -- Line equation is y = -2
  (∀ x : ℝ, line_equation x (-2))            -- Line is parallel to x-axis
  ∧ line_equation point_A.1 point_A.2        -- Point A lies on the line
  ∧ ¬line_equation point_B.1 point_B.2 :=    -- Point B does not lie on the line
by sorry

end line_and_points_l1156_115692


namespace marco_dad_strawberries_l1156_115677

/-- The weight of additional strawberries found by Marco's dad -/
def additional_strawberries (initial_total final_marco final_dad : ℕ) : ℕ :=
  (final_marco + final_dad) - initial_total

theorem marco_dad_strawberries :
  additional_strawberries 22 36 16 = 30 := by
  sorry

end marco_dad_strawberries_l1156_115677


namespace purely_imaginary_complex_number_l1156_115623

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ (z : ℂ), z = Complex.mk (a * (a - 1)) (a) ∧ z.re = 0) → a = 1 := by
  sorry

end purely_imaginary_complex_number_l1156_115623


namespace opposite_numbers_l1156_115605

theorem opposite_numbers : -(-(3 : ℤ)) = -(-3) := by sorry

end opposite_numbers_l1156_115605


namespace perimeter_pedal_relation_not_implies_equilateral_l1156_115661

/-- A triangle with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The pedal triangle of a given triangle -/
def pedalTriangle (t : Triangle) : Triangle := sorry

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℝ := sorry

/-- Predicate to check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Theorem stating that the original statement is false -/
theorem perimeter_pedal_relation_not_implies_equilateral :
  ∃ t : Triangle, perimeter t = 2 * perimeter (pedalTriangle t) ∧ ¬isEquilateral t := by
  sorry

end perimeter_pedal_relation_not_implies_equilateral_l1156_115661


namespace watermelon_seeds_count_l1156_115699

/-- The total number of watermelon seeds Yeon, Gwi, and Bom have together -/
def total_seeds (bom gwi yeon : ℕ) : ℕ := bom + gwi + yeon

/-- Theorem stating the total number of watermelon seeds -/
theorem watermelon_seeds_count :
  ∀ (bom gwi yeon : ℕ),
  bom = 300 →
  gwi = bom + 40 →
  yeon = 3 * gwi →
  total_seeds bom gwi yeon = 1660 :=
by
  sorry

end watermelon_seeds_count_l1156_115699


namespace cartoon_time_l1156_115673

theorem cartoon_time (cartoon_ratio : ℚ) (chore_ratio : ℚ) (chore_time : ℚ) : 
  cartoon_ratio / chore_ratio = 5 / 4 →
  chore_time = 96 →
  (cartoon_ratio * chore_time) / chore_ratio / 60 = 2 := by
sorry

end cartoon_time_l1156_115673


namespace basketball_tryouts_l1156_115691

theorem basketball_tryouts (girls boys called_back : ℕ) 
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : called_back = 26) :
  girls + boys - called_back = 17 := by
sorry

end basketball_tryouts_l1156_115691


namespace grid_paths_6x5_l1156_115667

/-- The number of paths from (0,0) to (m,n) on a grid, moving only right and up -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The dimensions of our grid -/
def gridWidth : ℕ := 6
def gridHeight : ℕ := 5

theorem grid_paths_6x5 : 
  gridPaths gridWidth gridHeight = 462 := by sorry

end grid_paths_6x5_l1156_115667


namespace fourth_separation_at_136pm_l1156_115697

-- Define the distance between cities
def distance_between_cities : ℝ := 300

-- Define the start time
def start_time : ℝ := 6

-- Define the time of first 50 km separation
def first_separation_time : ℝ := 8

-- Define the distance of separation
def separation_distance : ℝ := 50

-- Define the function to calculate the fourth separation time
def fourth_separation_time : ℝ := start_time + 7.6

-- Theorem statement
theorem fourth_separation_at_136pm 
  (h1 : distance_between_cities = 300)
  (h2 : start_time = 6)
  (h3 : first_separation_time = 8)
  (h4 : separation_distance = 50) :
  fourth_separation_time = 13.6 := by sorry

end fourth_separation_at_136pm_l1156_115697


namespace loop_iterations_count_l1156_115620

theorem loop_iterations_count (i : ℕ) : 
  i = 1 → (∀ j, j ≥ 1 ∧ j < 21 → i + j = j + 1) → i + 20 = 21 :=
by sorry

end loop_iterations_count_l1156_115620


namespace pig_count_l1156_115674

theorem pig_count (initial_pigs additional_pigs : ℝ) 
  (h1 : initial_pigs = 2465.25)
  (h2 : additional_pigs = 5683.75) : 
  initial_pigs + additional_pigs = 8149 :=
by sorry

end pig_count_l1156_115674


namespace alan_collected_48_shells_l1156_115641

/-- Given the number of shells collected by Laurie, calculate the number of shells collected by Alan. -/
def alan_shells (laurie_shells : ℕ) : ℕ :=
  let ben_shells := laurie_shells / 3
  4 * ben_shells

/-- Theorem stating that if Laurie collected 36 shells, Alan collected 48 shells. -/
theorem alan_collected_48_shells :
  alan_shells 36 = 48 := by
  sorry

end alan_collected_48_shells_l1156_115641


namespace quadratic_equation_q_value_l1156_115659

theorem quadratic_equation_q_value 
  (p q : ℝ) 
  (h : ∃ x : ℂ, 3 * x^2 + p * x + q = 0 ∧ x = 4 + 3*I) : 
  q = 75 := by
sorry

end quadratic_equation_q_value_l1156_115659


namespace common_factor_extraction_l1156_115611

def polynomial (a b c : ℤ) : ℤ := 8 * a^3 * b^2 + 12 * a^3 * b * c - 4 * a^2 * b

theorem common_factor_extraction (a b c : ℤ) :
  ∃ (k : ℤ), polynomial a b c = (4 * a^2 * b) * k ∧ 
  (∀ (d : ℤ), (∃ (l : ℤ), polynomial a b c = d * l) → d ≤ 4 * a^2 * b) :=
sorry

end common_factor_extraction_l1156_115611


namespace expression_simplification_and_evaluation_l1156_115629

theorem expression_simplification_and_evaluation :
  let x : ℝ := 2 * Real.sqrt 5 - 1
  (1 / (x^2 + 2*x + 1)) * (1 + 3 / (x - 1)) / ((x + 2) / (x^2 - 1)) = Real.sqrt 5 / 10 :=
by sorry

end expression_simplification_and_evaluation_l1156_115629


namespace amusement_park_earnings_l1156_115601

/-- Calculates the total earnings of an amusement park for a week --/
theorem amusement_park_earnings 
  (ticket_price : ℕ)
  (weekday_visitors : ℕ)
  (saturday_visitors : ℕ)
  (sunday_visitors : ℕ) :
  ticket_price = 3 →
  weekday_visitors = 100 →
  saturday_visitors = 200 →
  sunday_visitors = 300 →
  (5 * weekday_visitors + saturday_visitors + sunday_visitors) * ticket_price = 3000 := by
  sorry

#check amusement_park_earnings

end amusement_park_earnings_l1156_115601


namespace shirts_made_today_proof_l1156_115672

/-- Calculates the number of shirts made today given the production rate,
    yesterday's working time, and the total number of shirts made. -/
def shirts_made_today (rate : ℕ) (yesterday_time : ℕ) (total_shirts : ℕ) : ℕ :=
  total_shirts - (rate * yesterday_time)

/-- Proves that the number of shirts made today is 84 given the specified conditions. -/
theorem shirts_made_today_proof :
  shirts_made_today 6 12 156 = 84 := by
  sorry

end shirts_made_today_proof_l1156_115672


namespace unique_digit_solution_l1156_115664

theorem unique_digit_solution :
  ∃! (square boxplus boxtimes boxminus : ℕ),
    square < 10 ∧ boxplus < 10 ∧ boxtimes < 10 ∧ boxminus < 10 ∧
    square = 423 / 47 ∧
    1448 = 282 * boxminus + square * boxtimes ∧
    423 * boxplus = 282 * 3 ∧
    square = 9 ∧ boxplus = 2 ∧ boxtimes = 8 ∧ boxminus = 5 :=
by sorry

end unique_digit_solution_l1156_115664


namespace triangle_inequality_theorem_l1156_115686

/-- Given a triangle ABC with side lengths a, b, c, and a point M inside it,
    Ra, Rb, Rc are distances from M to sides BC, CA, AB respectively,
    da, db, dc are perpendicular distances from vertices A, B, C to the line through M parallel to the opposite sides. -/
def triangle_inequality (a b c Ra Rb Rc da db dc : ℝ) : Prop :=
  a * Ra + b * Rb + c * Rc ≥ 2 * (a * da + b * db + c * dc)

/-- M is the orthocenter of triangle ABC -/
def is_orthocenter (M : Point) (A B C : Point) : Prop := sorry

theorem triangle_inequality_theorem 
  (A B C M : Point) (a b c Ra Rb Rc da db dc : ℝ) :
  triangle_inequality a b c Ra Rb Rc da db dc ∧ 
  (triangle_inequality a b c Ra Rb Rc da db dc = (a * Ra + b * Rb + c * Rc = 2 * (a * da + b * db + c * dc)) ↔ 
   is_orthocenter M A B C) := by sorry

end triangle_inequality_theorem_l1156_115686


namespace function_value_at_negative_one_l1156_115633

/-- Given a function f(x) = ax³ + b sin(x) + 1 where f(1) = 5, prove that f(-1) = -3 -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * Real.sin x + 1) 
  (h2 : f 1 = 5) : 
  f (-1) = -3 :=
sorry

end function_value_at_negative_one_l1156_115633


namespace fraction_problem_l1156_115663

theorem fraction_problem (f : ℝ) : 
  (0.5 * 100 = f * 100 - 10) → f = 0.6 := by
  sorry

end fraction_problem_l1156_115663


namespace sandwich_bread_count_l1156_115609

/-- The number of pieces of bread needed for one double meat sandwich -/
def double_meat_bread : ℕ := 3

/-- The number of regular sandwiches -/
def regular_sandwiches : ℕ := 14

/-- The number of double meat sandwiches -/
def double_meat_sandwiches : ℕ := 12

/-- The number of pieces of bread needed for one regular sandwich -/
def regular_bread : ℕ := 2

/-- The total number of pieces of bread used -/
def total_bread : ℕ := 64

theorem sandwich_bread_count : 
  regular_sandwiches * regular_bread + double_meat_sandwiches * double_meat_bread = total_bread := by
  sorry

end sandwich_bread_count_l1156_115609


namespace product_of_roots_l1156_115644

theorem product_of_roots (x z : ℝ) (h1 : x - z = 6) (h2 : x^3 - z^3 = 108) : x * z = 6 := by
  sorry

end product_of_roots_l1156_115644


namespace total_marbles_l1156_115675

theorem total_marbles (jar_a jar_b jar_c : ℕ) : 
  jar_a = 28 →
  jar_b = jar_a + 12 →
  jar_c = 2 * jar_b →
  jar_a + jar_b + jar_c = 148 := by
  sorry

end total_marbles_l1156_115675


namespace geometric_sequence_fifth_term_l1156_115678

theorem geometric_sequence_fifth_term 
  (t : ℕ → ℝ) 
  (h_geometric : ∃ (a r : ℝ), ∀ n, t n = a * r^(n-1))
  (h_t1 : t 1 = 3)
  (h_t2 : t 2 = 6) :
  t 5 = 48 :=
sorry

end geometric_sequence_fifth_term_l1156_115678


namespace x_in_M_l1156_115625

def M : Set ℝ := {x | x ≤ 7}

theorem x_in_M : 4 ∈ M := by
  sorry

end x_in_M_l1156_115625


namespace paramon_solomon_meeting_time_l1156_115670

/- Define the total distance between A and B -/
variable (S : ℝ) (S_pos : S > 0)

/- Define the speeds of Paramon, Solomon, and Agafon -/
variable (x y z : ℝ) (x_pos : x > 0) (y_pos : y > 0) (z_pos : z > 0)

/- Define the time when Paramon and Solomon meet -/
def meeting_time : ℝ := 1

/- Theorem stating that Paramon and Solomon meet at 13:00 (1 hour after 12:00) -/
theorem paramon_solomon_meeting_time :
  (S / (2 * x) = 1) ∧                   /- Paramon travels half the distance in 1 hour -/
  (2 * z = S / 2 + 2 * x) ∧             /- Agafon catches up with Paramon at 14:00 -/
  (4 / 3 * (y + z) = S) ∧               /- Agafon meets Solomon at 13:20 -/
  (S / 2 + x * meeting_time = y * meeting_time) /- Paramon and Solomon meet -/
  → meeting_time = 1 := by sorry

end paramon_solomon_meeting_time_l1156_115670


namespace circumscribed_sphere_radius_is_four_l1156_115690

/-- Represents a triangular pyramid with specific dimensions -/
structure TriangularPyramid where
  base_side_length : ℝ
  perpendicular_edge_length : ℝ

/-- Calculates the radius of the circumscribed sphere around a triangular pyramid -/
def circumscribed_sphere_radius (pyramid : TriangularPyramid) : ℝ :=
  sorry

/-- Theorem stating that the radius of the circumscribed sphere is 4 for the given pyramid -/
theorem circumscribed_sphere_radius_is_four :
  let pyramid : TriangularPyramid := { base_side_length := 6, perpendicular_edge_length := 4 }
  circumscribed_sphere_radius pyramid = 4 := by
  sorry

end circumscribed_sphere_radius_is_four_l1156_115690


namespace bookmark_difference_l1156_115656

/-- The price of a bookmark in cents -/
def bookmark_price : ℕ := sorry

/-- The number of fifth graders who bought bookmarks -/
def fifth_graders : ℕ := sorry

/-- The number of fourth graders who bought bookmarks -/
def fourth_graders : ℕ := 20

theorem bookmark_difference : 
  bookmark_price > 0 ∧ 
  bookmark_price * fifth_graders = 225 ∧ 
  bookmark_price * fourth_graders = 260 →
  fourth_graders - fifth_graders = 7 := by sorry

end bookmark_difference_l1156_115656


namespace james_age_l1156_115614

theorem james_age (dan_age james_age : ℕ) : 
  (dan_age : ℚ) / james_age = 6 / 5 →
  dan_age + 4 = 28 →
  james_age = 20 := by
sorry

end james_age_l1156_115614


namespace prob_two_queens_or_at_least_one_jack_is_9_221_l1156_115695

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_jacks : ℕ)
  (num_queens : ℕ)

/-- Calculates the probability of the given event -/
def probability_two_queens_or_at_least_one_jack (d : Deck) : ℚ :=
  sorry

/-- Theorem stating the probability of drawing either two queens or at least one jack -/
theorem prob_two_queens_or_at_least_one_jack_is_9_221 :
  let standard_deck : Deck := ⟨52, 1, 3⟩
  probability_two_queens_or_at_least_one_jack standard_deck = 9/221 := by
  sorry

end prob_two_queens_or_at_least_one_jack_is_9_221_l1156_115695


namespace ratio_sum_theorem_l1156_115649

theorem ratio_sum_theorem (a b c : ℕ+) 
  (h1 : (a : ℚ) / b = 3 / 4)
  (h2 : (b : ℚ) / c = 5 / 6)
  (h3 : a + b + c = 1680) :
  a = 426 := by
  sorry

end ratio_sum_theorem_l1156_115649


namespace triangle_area_from_perimeter_and_inradius_l1156_115696

/-- The area of a triangle given its perimeter and inradius -/
theorem triangle_area_from_perimeter_and_inradius
  (perimeter : ℝ) (inradius : ℝ) (angle_smallest_sides : ℝ)
  (h_perimeter : perimeter = 36)
  (h_inradius : inradius = 2.5)
  (h_angle : angle_smallest_sides = 75) :
  inradius * (perimeter / 2) = 45 := by
  sorry

end triangle_area_from_perimeter_and_inradius_l1156_115696


namespace find_B_over_A_l1156_115657

-- Define the equation
def equation (A B x : ℝ) : Prop :=
  A / (x + 6) + B / (x^2 - 5*x) = (x^3 - 3*x^2 + 12) / (x^3 + x^2 - 30*x)

-- Theorem statement
theorem find_B_over_A (A B : ℤ) :
  (∀ x : ℝ, x ≠ -6 ∧ x ≠ 0 ∧ x ≠ 5 → equation A B x) →
  (B : ℝ) / (A : ℝ) = 2.2 := by
  sorry

end find_B_over_A_l1156_115657


namespace polynomial_xy_coefficient_l1156_115666

theorem polynomial_xy_coefficient (k : ℝ) : 
  (∀ x y : ℝ, x^2 - 3*k*x*y - 3*y^2 + 6*x*y - 8 = x^2 + (-3*k + 6)*x*y - 3*y^2 - 8) →
  (-3*k + 6 = 0) →
  k = 2 :=
by
  sorry

end polynomial_xy_coefficient_l1156_115666


namespace fern_purchase_cost_l1156_115613

/-- The total cost of purchasing high heels and ballet slippers -/
def total_cost (high_heel_price : ℝ) (ballet_slipper_ratio : ℝ) (ballet_slipper_count : ℕ) : ℝ :=
  high_heel_price + (ballet_slipper_ratio * high_heel_price * ballet_slipper_count)

/-- Theorem stating the total cost of Fern's purchase -/
theorem fern_purchase_cost :
  total_cost 60 (2/3) 5 = 260 := by
  sorry

end fern_purchase_cost_l1156_115613


namespace dandelion_puffs_to_dog_l1156_115676

/-- The number of dandelion puffs Caleb gave to his dog -/
def puffs_to_dog (total : ℕ) (to_mom : ℕ) (to_sister : ℕ) (to_grandmother : ℕ) 
                 (num_friends : ℕ) (to_each_friend : ℕ) : ℕ :=
  total - (to_mom + to_sister + to_grandmother + num_friends * to_each_friend)

/-- Theorem stating the number of dandelion puffs Caleb gave to his dog -/
theorem dandelion_puffs_to_dog : 
  puffs_to_dog 40 3 3 5 3 9 = 2 := by
  sorry

end dandelion_puffs_to_dog_l1156_115676


namespace division_of_decimals_l1156_115612

theorem division_of_decimals : (0.2 : ℚ) / (0.005 : ℚ) = 40 := by
  sorry

end division_of_decimals_l1156_115612


namespace yellow_square_area_l1156_115680

-- Define the cube's edge length
def cube_edge : ℝ := 15

-- Define the total amount of purple paint
def total_purple_paint : ℝ := 900

-- Define the number of faces on a cube
def num_faces : ℕ := 6

-- Theorem statement
theorem yellow_square_area :
  let total_surface_area := num_faces * (cube_edge ^ 2)
  let purple_area_per_face := total_purple_paint / num_faces
  let yellow_area_per_face := cube_edge ^ 2 - purple_area_per_face
  yellow_area_per_face = 75 := by sorry

end yellow_square_area_l1156_115680


namespace expression_equals_75_l1156_115651

-- Define the expression
def expression : ℚ := 150 / (10 / 5)

-- State the theorem
theorem expression_equals_75 : expression = 75 := by
  sorry

end expression_equals_75_l1156_115651


namespace max_distance_circle_centers_l1156_115683

/-- The maximum distance between the centers of two circles with 8-inch diameters
    placed within a 16-inch by 20-inch rectangle is 4√13 inches. -/
theorem max_distance_circle_centers (rect_width rect_height circle_diameter : ℝ)
  (hw : rect_width = 20)
  (hh : rect_height = 16)
  (hd : circle_diameter = 8)
  (h_nonneg : rect_width > 0 ∧ rect_height > 0 ∧ circle_diameter > 0) :
  Real.sqrt ((rect_width - circle_diameter) ^ 2 + (rect_height - circle_diameter) ^ 2) = 4 * Real.sqrt 13 :=
by sorry

end max_distance_circle_centers_l1156_115683


namespace arithmetic_simplification_l1156_115618

theorem arithmetic_simplification : 2537 + 240 * 3 / 60 - 347 = 2202 := by
  sorry

end arithmetic_simplification_l1156_115618


namespace at_least_one_positive_l1156_115679

theorem at_least_one_positive (x y z : ℝ) : 
  (x^2 - 2*y + Real.pi/2 > 0) ∨ 
  (y^2 - 2*z + Real.pi/3 > 0) ∨ 
  (z^2 - 2*x + Real.pi/6 > 0) := by
  sorry

end at_least_one_positive_l1156_115679


namespace parabola_point_to_directrix_distance_l1156_115624

/-- The distance from a point on a parabola to its directrix -/
theorem parabola_point_to_directrix_distance 
  (p : ℝ) -- Parameter of the parabola
  (A : ℝ × ℝ) -- Point A
  (h1 : A.1 = 1) -- x-coordinate of A is 1
  (h2 : A.2 = Real.sqrt 5) -- y-coordinate of A is √5
  (h3 : A.2^2 = 2 * p * A.1) -- A lies on the parabola y² = 2px
  : |A.1 - (-p/2)| = 9/4 := by
sorry


end parabola_point_to_directrix_distance_l1156_115624


namespace line_slope_angle_l1156_115669

theorem line_slope_angle (a : ℝ) : 
  (∃ (x y : ℝ), a * x - y - 1 = 0) → -- Line equation
  (Real.tan (π / 3) = a) →           -- Slope angle condition
  a = Real.sqrt 3 :=                 -- Conclusion
by sorry

end line_slope_angle_l1156_115669


namespace train_speed_calculation_l1156_115632

/-- Calculates the speed of a train given the parameters of a passing goods train -/
theorem train_speed_calculation (goods_train_speed : ℝ) (goods_train_length : ℝ) (passing_time : ℝ) : 
  goods_train_speed = 108 →
  goods_train_length = 340 →
  passing_time = 8 →
  ∃ (man_train_speed : ℝ), man_train_speed = 45 :=
by sorry

end train_speed_calculation_l1156_115632


namespace lamp_cost_ratio_l1156_115684

/-- The ratio of the cost of the most expensive lamp to the cheapest lamp -/
theorem lamp_cost_ratio 
  (cheapest_lamp : ℕ) 
  (frank_money : ℕ) 
  (remaining_money : ℕ) 
  (h1 : cheapest_lamp = 20)
  (h2 : frank_money = 90)
  (h3 : remaining_money = 30) :
  (frank_money - remaining_money) / cheapest_lamp = 3 := by
sorry

end lamp_cost_ratio_l1156_115684


namespace min_value_reciprocal_sum_l1156_115635

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 ∧
  ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ a₀ + 2 * b₀ = 1 ∧ 1 / a₀ + 1 / b₀ = 3 + 2 * Real.sqrt 2 :=
by sorry

end min_value_reciprocal_sum_l1156_115635


namespace polygon_sides_from_exterior_angle_l1156_115631

theorem polygon_sides_from_exterior_angle (n : ℕ) (exterior_angle : ℝ) : 
  (n > 2) → 
  (exterior_angle > 0) → 
  (exterior_angle < 180) → 
  (n * exterior_angle = 360) → 
  (exterior_angle = 30) → 
  n = 12 := by
sorry

end polygon_sides_from_exterior_angle_l1156_115631


namespace investment_average_interest_rate_l1156_115621

/-- Proves that given a total investment split into two parts with different interest rates
    and equal annual returns, the average interest rate is as calculated. -/
theorem investment_average_interest_rate 
  (total_investment : ℝ)
  (rate1 rate2 : ℝ)
  (h_total : total_investment = 4500)
  (h_rates : rate1 = 0.04 ∧ rate2 = 0.06)
  (h_equal_returns : ∃ (x : ℝ), 
    x > 0 ∧ x < total_investment ∧
    rate1 * (total_investment - x) = rate2 * x) :
  (rate1 * (total_investment - x) + rate2 * x) / total_investment = 0.048 := by
  sorry

#check investment_average_interest_rate

end investment_average_interest_rate_l1156_115621


namespace complex_equation_solution_l1156_115610

theorem complex_equation_solution : ∃ (z : ℂ), z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) ∧ z = (1/2 : ℂ) - Complex.I * ((Real.sqrt 3)/2) := by
  sorry

end complex_equation_solution_l1156_115610


namespace count_valid_numbers_l1156_115627

/-- A function that generates all valid numbers from digits 1, 2, and 3 without repetition -/
def validNumbers : List ℕ :=
  [1, 2, 3, 12, 13, 21, 23, 31, 32, 123, 132, 213, 231, 312, 321]

/-- The count of natural numbers composed of digits 1, 2, and 3 without repetition -/
theorem count_valid_numbers : validNumbers.length = 15 := by
  sorry

end count_valid_numbers_l1156_115627


namespace cats_in_academy_l1156_115694

/-- The number of cats that can jump -/
def jump : ℕ := 40

/-- The number of cats that can climb -/
def climb : ℕ := 25

/-- The number of cats that can hunt -/
def hunt : ℕ := 30

/-- The number of cats that can jump and climb -/
def jump_and_climb : ℕ := 10

/-- The number of cats that can climb and hunt -/
def climb_and_hunt : ℕ := 15

/-- The number of cats that can jump and hunt -/
def jump_and_hunt : ℕ := 12

/-- The number of cats that can do all three skills -/
def all_skills : ℕ := 5

/-- The number of cats that cannot perform any skills -/
def no_skills : ℕ := 6

/-- The total number of cats in the academy -/
def total_cats : ℕ := 69

theorem cats_in_academy :
  total_cats = jump + climb + hunt - jump_and_climb - climb_and_hunt - jump_and_hunt + all_skills + no_skills := by
  sorry

end cats_in_academy_l1156_115694


namespace bessonov_tax_refund_l1156_115647

def income_tax : ℝ := 156000
def education_expense : ℝ := 130000
def medical_expense : ℝ := 10000
def tax_rate : ℝ := 0.13

def total_deductible_expenses : ℝ := education_expense + medical_expense

def max_refund : ℝ := tax_rate * total_deductible_expenses

theorem bessonov_tax_refund :
  min max_refund income_tax = 18200 :=
sorry

end bessonov_tax_refund_l1156_115647


namespace cycle_original_price_l1156_115606

/-- Proves that the original price of a cycle is 1600 when sold at a 10% loss for 1440 --/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1440)
  (h2 : loss_percentage = 10) : 
  selling_price / (1 - loss_percentage / 100) = 1600 := by
sorry

end cycle_original_price_l1156_115606


namespace count_integer_solutions_l1156_115646

theorem count_integer_solutions : 
  ∃! (S : Finset (ℕ × ℕ)), 
    (∀ (m n : ℕ), (m, n) ∈ S ↔ m > 0 ∧ n > 0 ∧ 5 / m + 3 / n = 1) ∧ 
    Finset.card S = 4 :=
by sorry

end count_integer_solutions_l1156_115646


namespace infinite_solutions_iff_b_eq_neg_six_l1156_115603

theorem infinite_solutions_iff_b_eq_neg_six :
  ∀ b : ℝ, (∀ x : ℝ, 5 * (3 * x - b) = 3 * (5 * x + 10)) ↔ b = -6 := by
sorry

end infinite_solutions_iff_b_eq_neg_six_l1156_115603


namespace nickel_chocolates_l1156_115616

/-- Given that Robert ate 7 chocolates and 4 more than Nickel, prove that Nickel ate 3 chocolates. -/
theorem nickel_chocolates (robert : ℕ) (nickel : ℕ) 
  (h1 : robert = 7)
  (h2 : robert = nickel + 4) : 
  nickel = 3 := by
  sorry

end nickel_chocolates_l1156_115616


namespace ab_difference_l1156_115693

theorem ab_difference (A B C : ℕ) : 
  A > 0 → B > 0 → C > 0 →
  (1212017 * 100 * A + 1212017 * 10 * B + 1212017 * C) % 45 = 0 →
  ∃ (max_AB min_AB : ℕ),
    (∀ A' B' C' : ℕ, 
      A' > 0 → B' > 0 → C' > 0 →
      (1212017 * 100 * A' + 1212017 * 10 * B' + 1212017 * C') % 45 = 0 →
      A' * 10 + B' ≤ max_AB) ∧
    (∀ A' B' C' : ℕ, 
      A' > 0 → B' > 0 → C' > 0 →
      (1212017 * 100 * A' + 1212017 * 10 * B' + 1212017 * C') % 45 = 0 →
      A' * 10 + B' ≥ min_AB) ∧
    max_AB - min_AB = 85 :=
by sorry

end ab_difference_l1156_115693


namespace machine_value_is_35000_l1156_115639

/-- Represents the denomination of a bill in dollars -/
inductive BillType
  | five
  | ten
  | twenty

/-- Returns the value of a bill in dollars -/
def billValue : BillType → Nat
  | BillType.five => 5
  | BillType.ten => 10
  | BillType.twenty => 20

/-- Represents a bundle of bills -/
structure Bundle where
  billType : BillType
  count : Nat

/-- Represents a cash machine -/
structure CashMachine where
  bundles : List Bundle

/-- The number of bills in each bundle -/
def billsPerBundle : Nat := 100

/-- The number of bundles for each bill type -/
def bundlesPerType : Nat := 10

/-- Calculates the total value of a bundle -/
def bundleValue (b : Bundle) : Nat :=
  billValue b.billType * b.count

/-- Calculates the total value of all bundles in the machine -/
def machineValue (m : CashMachine) : Nat :=
  m.bundles.map bundleValue |>.sum

/-- The cash machine configuration -/
def filledMachine : CashMachine :=
  { bundles := [
    { billType := BillType.five, count := billsPerBundle },
    { billType := BillType.ten, count := billsPerBundle },
    { billType := BillType.twenty, count := billsPerBundle }
  ] }

/-- Theorem: The total amount of money required to fill the machine is $35,000 -/
theorem machine_value_is_35000 : 
  machineValue filledMachine = 35000 := by sorry

end machine_value_is_35000_l1156_115639


namespace arithmetic_sequence_sum_10_l1156_115665

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  S : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_def : ∀ n : ℕ+, S n = (n : ℚ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_10 (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = 16) (h2 : seq.S 20 = 20) : seq.S 10 = 110 := by
  sorry

end arithmetic_sequence_sum_10_l1156_115665


namespace at_least_one_correct_l1156_115687

theorem at_least_one_correct (pA pB : ℝ) (hA : pA = 0.8) (hB : pB = 0.9) :
  1 - (1 - pA) * (1 - pB) = 0.98 := by
  sorry

end at_least_one_correct_l1156_115687


namespace fraction_division_specific_fraction_division_l1156_115615

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem specific_fraction_division :
  (3 : ℚ) / 7 / ((2 : ℚ) / 5) = 15 / 14 := by sorry

end fraction_division_specific_fraction_division_l1156_115615


namespace plane_relations_theorem_l1156_115658

-- Define a type for planes
def Plane : Type := Unit

-- Define the relations between planes
def perpendicular (p q : Plane) : Prop := sorry
def parallel (p q : Plane) : Prop := sorry

-- Define a predicate for three non-collinear points on a plane being equidistant from another plane
def three_points_equidistant (p q : Plane) : Prop := sorry

-- The theorem to be proven
theorem plane_relations_theorem (α β γ : Plane) : 
  ¬((perpendicular α β ∧ perpendicular β γ → parallel α γ) ∨ 
    (three_points_equidistant α β → parallel α β)) := by sorry

end plane_relations_theorem_l1156_115658


namespace quadratic_factorization_sum_l1156_115637

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 9*x + 20 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 11*x - 60 = (x + b)*(x - c)) →
  a + b + c = 23 := by
sorry

end quadratic_factorization_sum_l1156_115637


namespace smallest_integer_with_remainders_l1156_115608

theorem smallest_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 4 = 3) ∧ 
  (n % 3 = 2) ∧ 
  (∀ m : ℕ, m > 0 → m % 4 = 3 → m % 3 = 2 → m ≥ n) ∧
  n = 11 := by
sorry

end smallest_integer_with_remainders_l1156_115608


namespace regular_octagon_perimeter_l1156_115642

/-- A regular octagon is a polygon with 8 equal sides -/
def RegularOctagon : Type := Unit

/-- The side length of the regular octagon -/
def side_length : ℝ := 3

/-- The number of sides in an octagon -/
def num_sides : ℕ := 8

/-- The perimeter of a regular octagon is the product of its number of sides and side length -/
def perimeter (o : RegularOctagon) : ℝ := num_sides * side_length

theorem regular_octagon_perimeter : 
  ∀ (o : RegularOctagon), perimeter o = 24 := by sorry

end regular_octagon_perimeter_l1156_115642


namespace a_not_in_A_l1156_115681

def A : Set ℝ := {x | x ≤ 4}

theorem a_not_in_A : 3 * Real.sqrt 3 ∉ A := by sorry

end a_not_in_A_l1156_115681


namespace mr_callen_wooden_toys_solution_is_eight_l1156_115622

/-- Proves that the number of wooden toys bought is 8, given the conditions of Mr. Callen's purchase and sale. -/
theorem mr_callen_wooden_toys : ℕ :=
  let num_paintings : ℕ := 10
  let painting_cost : ℚ := 40
  let toy_cost : ℚ := 20
  let painting_discount : ℚ := 0.1
  let toy_discount : ℚ := 0.15
  let total_loss : ℚ := 64

  let painting_revenue := num_paintings * (painting_cost * (1 - painting_discount))
  let toy_revenue (num_toys : ℕ) := num_toys * (toy_cost * (1 - toy_discount))
  let total_cost (num_toys : ℕ) := num_paintings * painting_cost + num_toys * toy_cost
  let total_revenue (num_toys : ℕ) := painting_revenue + toy_revenue num_toys

  have h : ∃ (num_toys : ℕ), total_cost num_toys - total_revenue num_toys = total_loss :=
    sorry

  Classical.choose h

/-- The solution to the problem is 8 wooden toys. -/
theorem solution_is_eight : mr_callen_wooden_toys = 8 := by
  sorry

end mr_callen_wooden_toys_solution_is_eight_l1156_115622


namespace composite_5n_plus_3_l1156_115655

theorem composite_5n_plus_3 (n : ℕ) (h1 : ∃ x : ℕ, 2 * n + 1 = x^2) (h2 : ∃ y : ℕ, 3 * n + 1 = y^2) : 
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 5 * n + 3 = a * b :=
by sorry

end composite_5n_plus_3_l1156_115655


namespace polynomial_value_l1156_115626

def f (x : ℝ) : ℝ := x^5 + 3*x^4 - 5*x^3 + 7*x^2 - 9*x + 11

theorem polynomial_value : f 4 = 1559 := by
  sorry

end polynomial_value_l1156_115626


namespace simplify_expression_l1156_115650

theorem simplify_expression (a b : ℝ) (h1 : 2*b - a < 3) (h2 : 2*a - b < 5) :
  -|2*b - a - 7| - |b - 2*a + 8| + |a + b - 9| = -6 := by
  sorry

end simplify_expression_l1156_115650


namespace cube_root_of_a_plus_b_l1156_115645

theorem cube_root_of_a_plus_b (a b : ℝ) : 
  a > 0 → (2*b - 1)^2 = a → (b + 4)^2 = a → (a + b)^(1/3) = 2 := by
  sorry

end cube_root_of_a_plus_b_l1156_115645


namespace parabola_through_origin_l1156_115689

/-- A parabola is defined by the equation y = ax^2 + bx + c, where a, b, and c are real numbers. -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point on a 2D plane is represented by its x and y coordinates. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The origin is the point (0, 0) on a 2D plane. -/
def origin : Point := ⟨0, 0⟩

/-- A point lies on a parabola if its coordinates satisfy the parabola's equation. -/
def lies_on (p : Point) (para : Parabola) : Prop :=
  p.y = para.a * p.x^2 + para.b * p.x + para.c

/-- Theorem: A parabola passes through the origin if and only if its c coefficient is zero. -/
theorem parabola_through_origin (para : Parabola) :
  lies_on origin para ↔ para.c = 0 := by
  sorry

end parabola_through_origin_l1156_115689


namespace system_solution_l1156_115671

theorem system_solution (x y : ℝ) : 
  (x^2 + x*y + y^2 = 23) ∧ (x^4 + x^2*y^2 + y^4 = 253) →
  ((x = Real.sqrt 29 ∧ y = Real.sqrt 5) ∨ 
   (x = Real.sqrt 29 ∧ y = -Real.sqrt 5) ∨
   (x = -Real.sqrt 29 ∧ y = Real.sqrt 5) ∨
   (x = -Real.sqrt 29 ∧ y = -Real.sqrt 5)) :=
by sorry

end system_solution_l1156_115671


namespace child_ticket_price_l1156_115628

/-- Given the following information about a movie theater's ticket sales:
  - Total tickets sold is 900
  - Total revenue is $5,100
  - Adult ticket price is $7
  - Number of adult tickets sold is 500
  Prove that the price of a child's ticket is $4. -/
theorem child_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (adult_price : ℕ) 
  (adult_tickets : ℕ) 
  (h1 : total_tickets = 900) 
  (h2 : total_revenue = 5100) 
  (h3 : adult_price = 7) 
  (h4 : adult_tickets = 500) : 
  (total_revenue - adult_price * adult_tickets) / (total_tickets - adult_tickets) = 4 := by
sorry


end child_ticket_price_l1156_115628


namespace exactly_one_even_negation_l1156_115685

/-- Represents the property of a natural number being even -/
def IsEven (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

/-- Represents the property of a natural number being odd -/
def IsOdd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

/-- States that exactly one of three natural numbers is even -/
def ExactlyOneEven (a b c : ℕ) : Prop :=
  (IsEven a ∧ IsOdd b ∧ IsOdd c) ∨
  (IsOdd a ∧ IsEven b ∧ IsOdd c) ∨
  (IsOdd a ∧ IsOdd b ∧ IsEven c)

/-- States that at least two of three natural numbers are even or all are odd -/
def AtLeastTwoEvenOrAllOdd (a b c : ℕ) : Prop :=
  (IsEven a ∧ IsEven b) ∨
  (IsEven a ∧ IsEven c) ∨
  (IsEven b ∧ IsEven c) ∨
  (IsOdd a ∧ IsOdd b ∧ IsOdd c)

theorem exactly_one_even_negation (a b c : ℕ) :
  ¬(ExactlyOneEven a b c) ↔ AtLeastTwoEvenOrAllOdd a b c :=
sorry

end exactly_one_even_negation_l1156_115685


namespace cupcakes_problem_l1156_115607

/-- Calculates the number of cupcakes per package given the initial number of cupcakes,
    the number of cupcakes eaten, and the number of packages. -/
def cupcakes_per_package (initial : ℕ) (eaten : ℕ) (packages : ℕ) : ℕ :=
  (initial - eaten) / packages

/-- Proves that given 50 initial cupcakes, 5 cupcakes eaten, and 9 equal packages,
    the number of cupcakes in each package is 5. -/
theorem cupcakes_problem :
  cupcakes_per_package 50 5 9 = 5 := by
  sorry

#eval cupcakes_per_package 50 5 9

end cupcakes_problem_l1156_115607


namespace toy_selling_price_l1156_115668

/-- Calculates the total selling price of toys given the number of toys sold,
    the number of toys whose cost price was gained, and the cost price per toy. -/
def totalSellingPrice (numToysSold : ℕ) (numToysGained : ℕ) (costPrice : ℕ) : ℕ :=
  numToysSold * costPrice + numToysGained * costPrice

/-- Theorem stating that for the given conditions, the total selling price is 27300. -/
theorem toy_selling_price :
  totalSellingPrice 18 3 1300 = 27300 := by
  sorry

end toy_selling_price_l1156_115668


namespace concert_songs_theorem_l1156_115638

/-- Represents the number of songs sung by each girl -/
structure SongCounts where
  mary : ℕ
  alina : ℕ
  tina : ℕ
  hanna : ℕ

/-- The conditions of the problem -/
def satisfies_conditions (sc : SongCounts) : Prop :=
  sc.hanna = 4 ∧
  sc.mary = 7 ∧
  sc.alina > sc.hanna ∧
  sc.alina < sc.mary ∧
  sc.tina > sc.hanna ∧
  sc.tina < sc.mary

/-- The total number of songs sung by the trios -/
def total_songs (sc : SongCounts) : ℕ :=
  (sc.mary + sc.alina + sc.tina + sc.hanna) / 3

/-- The main theorem to be proved -/
theorem concert_songs_theorem (sc : SongCounts) :
  satisfies_conditions sc → total_songs sc = 7 := by
  sorry

end concert_songs_theorem_l1156_115638


namespace sin_A_value_area_ABC_l1156_115662

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- Conditions
  c = Real.sqrt 2 ∧
  a = 1 ∧
  Real.cos C = 3/4

-- Theorem for sin A
theorem sin_A_value (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : Real.sin A = Real.sqrt 14 / 8 := by
  sorry

-- Theorem for the area of triangle ABC
theorem area_ABC (A B C : ℝ) (a b c : ℝ) 
  (h : triangle_ABC A B C a b c) : (1/2) * a * b * Real.sin C = Real.sqrt 7 / 4 := by
  sorry

end sin_A_value_area_ABC_l1156_115662


namespace simplify_and_multiply_l1156_115654

theorem simplify_and_multiply (b : ℝ) : (2 * (3 * b) * (4 * b^2) * (5 * b^3)) * 6 = 720 * b^6 := by
  sorry

end simplify_and_multiply_l1156_115654


namespace valid_star_arrangement_exists_l1156_115640

/-- Represents a domino piece with two sides -/
structure Domino :=
  (side1 : Nat)
  (side2 : Nat)

/-- Represents a ray in the star arrangement -/
structure Ray :=
  (pieces : List Domino)
  (length : Nat)
  (sum : Nat)

/-- Represents the center of the star -/
structure Center :=
  (tiles : List Nat)

/-- Represents the entire star arrangement -/
structure StarArrangement :=
  (rays : List Ray)
  (center : Center)

/-- Checks if a domino arrangement is valid according to domino rules -/
def isValidDominoArrangement (arrangement : List Domino) : Prop :=
  sorry

/-- Checks if a ray is valid (correct length and sum) -/
def isValidRay (ray : Ray) : Prop :=
  ray.length ∈ [3, 4] ∧ ray.sum = 21 ∧ isValidDominoArrangement ray.pieces

/-- Checks if the center is valid -/
def isValidCenter (center : Center) : Prop :=
  center.tiles.length = 8 ∧
  (∀ n, n ∈ [1, 2, 3, 4, 5, 6] → n ∈ center.tiles) ∧
  (center.tiles.filter (· = 0)).length = 2

/-- Checks if the entire star arrangement is valid -/
def isValidStarArrangement (star : StarArrangement) : Prop :=
  star.rays.length = 8 ∧
  (∀ ray ∈ star.rays, isValidRay ray) ∧
  isValidCenter star.center

/-- The main theorem stating that a valid star arrangement exists -/
theorem valid_star_arrangement_exists : ∃ (star : StarArrangement), isValidStarArrangement star :=
  sorry

end valid_star_arrangement_exists_l1156_115640


namespace motorcycle_wheels_l1156_115682

theorem motorcycle_wheels (total_wheels : ℕ) (num_cars : ℕ) (num_motorcycles : ℕ) 
  (wheels_per_car : ℕ) (h1 : total_wheels = 117) (h2 : num_cars = 19) 
  (h3 : num_motorcycles = 11) (h4 : wheels_per_car = 5) :
  (total_wheels - num_cars * wheels_per_car) / num_motorcycles = 2 :=
by sorry

end motorcycle_wheels_l1156_115682


namespace articles_bought_l1156_115698

/-- The number of articles bought at the cost price -/
def X : ℝ := sorry

/-- The cost price of each article -/
def C : ℝ := sorry

/-- The selling price of each article -/
def S : ℝ := sorry

/-- The gain percent -/
def gain_percent : ℝ := 8.695652173913043

theorem articles_bought (h1 : X * C = 46 * S) 
                        (h2 : gain_percent = ((S - C) / C) * 100) : 
  X = 50 := by sorry

end articles_bought_l1156_115698


namespace range_of_e_l1156_115688

theorem range_of_e (a b c d e : ℝ) 
  (sum_condition : a + b + c + d + e = 8)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  0 ≤ e ∧ e ≤ 16/5 := by
sorry

end range_of_e_l1156_115688


namespace consecutive_odd_sum_2005_2006_l1156_115653

theorem consecutive_odd_sum_2005_2006 :
  (∃ (n k : ℕ), n ≥ 2 ∧ 2005 = n * (2 * k + n)) ∧
  (¬ ∃ (n k : ℕ), n ≥ 2 ∧ 2006 = n * (2 * k + n)) := by
  sorry

end consecutive_odd_sum_2005_2006_l1156_115653


namespace translation_sum_l1156_115602

/-- A translation that moves a point 5 units right and 3 units up -/
def translation (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + 5, p.2 + 3)

/-- Apply a translation n times to a point -/
def apply_translation (p : ℝ × ℝ) (n : ℕ) : ℝ × ℝ :=
  Nat.recOn n p (fun _ q => translation q)

theorem translation_sum (initial : ℝ × ℝ) :
  let final := apply_translation initial 6
  final.1 + final.2 = 47 :=
sorry

end translation_sum_l1156_115602
