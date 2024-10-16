import Mathlib

namespace NUMINAMATH_CALUDE_pyramid_section_is_trapezoid_l87_8735

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  d : ℝ

/-- Represents a pyramid -/
structure Pyramid where
  apex : Point3D
  base : List Point3D

/-- Represents a parallelogram -/
structure Parallelogram where
  vertices : List Point3D

/-- Represents a trapezoid -/
structure Trapezoid where
  vertices : List Point3D

def is_parallelogram (p : Parallelogram) : Prop := sorry

def is_point_on_edge (p : Point3D) (e1 e2 : Point3D) : Prop := sorry

def intersection_is_trapezoid (plane : Plane) (pyr : Pyramid) : Prop := sorry

theorem pyramid_section_is_trapezoid 
  (S A B C D M : Point3D) 
  (base : Parallelogram) 
  (pyr : Pyramid) 
  (plane : Plane) :
  is_parallelogram base →
  pyr.apex = S →
  pyr.base = base.vertices →
  is_point_on_edge M S C →
  plane.normal = sorry → -- Define the normal vector of plane ABM
  plane.d = sorry → -- Define the d value for plane ABM
  intersection_is_trapezoid plane pyr := by
  sorry

#check pyramid_section_is_trapezoid

end NUMINAMATH_CALUDE_pyramid_section_is_trapezoid_l87_8735


namespace NUMINAMATH_CALUDE_max_edges_cube_plane_intersection_l87_8739

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the internal structure of a cube for this problem

/-- A plane is a flat, two-dimensional surface that extends infinitely far -/
structure Plane where
  -- We don't need to define the internal structure of a plane for this problem

/-- A polygon is a plane figure with straight sides -/
structure Polygon where
  edges : ℕ

/-- The result of intersecting a cube with a plane is a polygon -/
def intersect (c : Cube) (p : Plane) : Polygon :=
  sorry

/-- Theorem: The maximum number of edges in a polygon formed by the intersection of a cube and a plane is 6 -/
theorem max_edges_cube_plane_intersection (c : Cube) (p : Plane) :
  (intersect c p).edges ≤ 6 ∧ ∃ (c' : Cube) (p' : Plane), (intersect c' p').edges = 6 :=
sorry

end NUMINAMATH_CALUDE_max_edges_cube_plane_intersection_l87_8739


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l87_8765

/-- The volume of a cylinder formed by rotating a rectangle about its vertical line of symmetry -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (h_length : length = 20) (h_width : width = 10) :
  let radius := width / 2
  let height := length
  let volume := π * radius^2 * height
  volume = 500 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l87_8765


namespace NUMINAMATH_CALUDE_average_mile_time_l87_8790

theorem average_mile_time (mile1 mile2 mile3 mile4 : ℕ) 
  (h1 : mile1 = 6)
  (h2 : mile2 = 5)
  (h3 : mile3 = 5)
  (h4 : mile4 = 4) :
  (mile1 + mile2 + mile3 + mile4) / 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_average_mile_time_l87_8790


namespace NUMINAMATH_CALUDE_original_paint_intensity_l87_8778

/-- 
Given a paint mixture where 20% of the original paint is replaced with a 25% solution,
resulting in a mixture with 45% intensity, prove that the original paint intensity was 50%.
-/
theorem original_paint_intensity 
  (original_intensity : ℝ) 
  (replaced_fraction : ℝ) 
  (replacement_solution_intensity : ℝ) 
  (final_intensity : ℝ) : 
  replaced_fraction = 0.2 →
  replacement_solution_intensity = 25 →
  final_intensity = 45 →
  (1 - replaced_fraction) * original_intensity + 
    replaced_fraction * replacement_solution_intensity = final_intensity →
  original_intensity = 50 := by
sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l87_8778


namespace NUMINAMATH_CALUDE_guppies_count_l87_8762

-- Define the number of guppies each person has
def haylee_guppies : ℕ := 3 * 12 -- 3 dozen
def jose_guppies : ℕ := haylee_guppies / 2
def charliz_guppies : ℕ := jose_guppies / 3
def nicolai_guppies : ℕ := charliz_guppies * 4

-- Define the total number of guppies
def total_guppies : ℕ := haylee_guppies + jose_guppies + charliz_guppies + nicolai_guppies

-- Theorem to prove
theorem guppies_count : total_guppies = 84 := by
  sorry

end NUMINAMATH_CALUDE_guppies_count_l87_8762


namespace NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l87_8752

-- Define a regular octagon
structure RegularOctagon :=
  (vertices : Fin 8 → ℝ × ℝ)
  (is_regular : ∀ i j : Fin 8, dist (vertices i) (vertices ((i + 1) % 8)) = dist (vertices j) (vertices ((j + 1) % 8)))

-- Define the extension of sides CD and FG
def extend_sides (octagon : RegularOctagon) : ℝ × ℝ :=
  sorry

-- Define the angle at point Q
def angle_at_Q (octagon : RegularOctagon) : ℝ :=
  sorry

-- Theorem statement
theorem regular_octagon_extended_sides_angle (octagon : RegularOctagon) :
  angle_at_Q octagon = 180 :=
sorry

end NUMINAMATH_CALUDE_regular_octagon_extended_sides_angle_l87_8752


namespace NUMINAMATH_CALUDE_function_equality_l87_8716

theorem function_equality : 
  (∀ x : ℝ, |x| = Real.sqrt (x^2)) ∧ 
  (∀ x : ℝ, x^2 = (fun t => t^2) x) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l87_8716


namespace NUMINAMATH_CALUDE_power_equation_non_negative_l87_8713

theorem power_equation_non_negative (a b c d : ℤ) 
  (h : (2 : ℝ)^a + (2 : ℝ)^b = (5 : ℝ)^c + (5 : ℝ)^d) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d := by
  sorry

end NUMINAMATH_CALUDE_power_equation_non_negative_l87_8713


namespace NUMINAMATH_CALUDE_exists_A_all_A_digit_numbers_A_minus_1_expressible_l87_8757

/-- Represents the concatenation operation -/
def concatenate (a b : ℕ) : ℕ := sorry

/-- Checks if a number is m-expressible -/
def is_m_expressible (n m : ℕ) : Prop := sorry

/-- The main theorem to be proved -/
theorem exists_A_all_A_digit_numbers_A_minus_1_expressible :
  ∃ A : ℕ, ∀ n : ℕ, (10^(A-1) ≤ n ∧ n < 10^A) → is_m_expressible n (A-1) := by
  sorry

end NUMINAMATH_CALUDE_exists_A_all_A_digit_numbers_A_minus_1_expressible_l87_8757


namespace NUMINAMATH_CALUDE_different_arrangements_count_l87_8760

def num_red_balls : ℕ := 6
def num_green_balls : ℕ := 3
def num_selected_balls : ℕ := 4

def num_arrangements (r g s : ℕ) : ℕ :=
  (Nat.choose s s) +
  (Nat.choose s 1) * 2 +
  (Nat.choose s 2)

theorem different_arrangements_count :
  num_arrangements num_red_balls num_green_balls num_selected_balls = 15 := by
  sorry

end NUMINAMATH_CALUDE_different_arrangements_count_l87_8760


namespace NUMINAMATH_CALUDE_sum_of_fractions_l87_8732

theorem sum_of_fractions : (3 / 10 : ℚ) + (29 / 5 : ℚ) = 61 / 10 := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l87_8732


namespace NUMINAMATH_CALUDE_members_playing_both_l87_8796

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  totalMembers : ℕ
  badmintonPlayers : ℕ
  tennisPlayers : ℕ
  neitherPlayers : ℕ

/-- Calculates the number of members playing both badminton and tennis -/
def playBoth (club : SportsClub) : ℕ :=
  club.badmintonPlayers + club.tennisPlayers - (club.totalMembers - club.neitherPlayers)

/-- Theorem stating the number of members playing both sports in the given scenario -/
theorem members_playing_both (club : SportsClub)
    (h1 : club.totalMembers = 50)
    (h2 : club.badmintonPlayers = 25)
    (h3 : club.tennisPlayers = 32)
    (h4 : club.neitherPlayers = 5) :
    playBoth club = 12 := by
  sorry


end NUMINAMATH_CALUDE_members_playing_both_l87_8796


namespace NUMINAMATH_CALUDE_hillary_saturday_reading_l87_8750

/-- Calculates the number of minutes read on Saturday given the total assignment time and time read on Friday and Sunday. -/
def minutes_read_saturday (total_assignment : ℕ) (friday_reading : ℕ) (sunday_reading : ℕ) : ℕ :=
  total_assignment - (friday_reading + sunday_reading)

/-- Theorem stating that given the specific conditions of Hillary's reading assignment, she read for 28 minutes on Saturday. -/
theorem hillary_saturday_reading :
  minutes_read_saturday 60 16 16 = 28 := by
  sorry

end NUMINAMATH_CALUDE_hillary_saturday_reading_l87_8750


namespace NUMINAMATH_CALUDE_total_animal_eyes_pond_animal_eyes_l87_8782

theorem total_animal_eyes (num_frogs num_crocodiles : ℕ) 
  (eyes_per_frog eyes_per_crocodile : ℕ) : ℕ :=
  num_frogs * eyes_per_frog + num_crocodiles * eyes_per_crocodile

theorem pond_animal_eyes : total_animal_eyes 20 6 2 2 = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_animal_eyes_pond_animal_eyes_l87_8782


namespace NUMINAMATH_CALUDE_ab_value_l87_8729

theorem ab_value (a b : ℝ) (h : (a + 2)^2 + |b - 4| = 0) : a^b = 16 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l87_8729


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l87_8717

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l87_8717


namespace NUMINAMATH_CALUDE_football_score_proof_l87_8738

theorem football_score_proof :
  let hawks_touchdowns : ℕ := 4
  let hawks_successful_extra_points : ℕ := 2
  let hawks_failed_extra_points : ℕ := 2
  let hawks_field_goals : ℕ := 2
  let eagles_touchdowns : ℕ := 3
  let eagles_successful_extra_points : ℕ := 3
  let eagles_field_goals : ℕ := 3
  let touchdown_points : ℕ := 6
  let extra_point_points : ℕ := 1
  let field_goal_points : ℕ := 3
  
  let hawks_score : ℕ := hawks_touchdowns * touchdown_points + 
                         hawks_successful_extra_points * extra_point_points + 
                         hawks_field_goals * field_goal_points
  
  let eagles_score : ℕ := eagles_touchdowns * touchdown_points + 
                          eagles_successful_extra_points * extra_point_points + 
                          eagles_field_goals * field_goal_points
  
  let total_score : ℕ := hawks_score + eagles_score
  
  total_score = 62 := by
    sorry

end NUMINAMATH_CALUDE_football_score_proof_l87_8738


namespace NUMINAMATH_CALUDE_halfway_fraction_l87_8749

theorem halfway_fraction : (3 / 4 + 5 / 6) / 2 = 19 / 24 := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_l87_8749


namespace NUMINAMATH_CALUDE_seventh_observation_value_l87_8723

/-- Given 6 initial observations with an average of 15, prove that adding a 7th observation
    that decreases the overall average by 1 results in the 7th observation having a value of 8. -/
theorem seventh_observation_value (n : ℕ) (initial_average new_average : ℚ) :
  n = 6 →
  initial_average = 15 →
  new_average = initial_average - 1 →
  ∃ x : ℚ, x = 8 ∧ (n : ℚ) * initial_average + x = (n + 1 : ℚ) * new_average :=
by sorry

end NUMINAMATH_CALUDE_seventh_observation_value_l87_8723


namespace NUMINAMATH_CALUDE_ladder_problem_l87_8768

theorem ladder_problem (ladder_length height_on_wall : ℝ) 
  (h1 : ladder_length = 13)
  (h2 : height_on_wall = 12) :
  ∃ (distance_from_wall : ℝ), 
    distance_from_wall ^ 2 + height_on_wall ^ 2 = ladder_length ^ 2 ∧ 
    distance_from_wall = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_problem_l87_8768


namespace NUMINAMATH_CALUDE_tangent_line_equation_l87_8774

-- Define the function f(x) = x³ + x
def f (x : ℝ) : ℝ := x^3 + x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3 * x^2 + 1

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ m * x - y + b = 0) ∧
    (m = f' 1) ∧
    (f 1 = m * 1 + b) ∧
    (m * 1 - f 1 + b = 0) ∧
    (m = 4 ∧ b = -2) := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_equation_l87_8774


namespace NUMINAMATH_CALUDE_doubly_underlined_count_l87_8780

def count_doubly_underlined (n : ℕ) : ℕ :=
  let multiples_of_6_not_4 := (n / 6 + 1) / 2
  let multiples_of_4_not_3 := 2 * (n / 4 + 1) / 3
  multiples_of_6_not_4 + multiples_of_4_not_3

theorem doubly_underlined_count :
  count_doubly_underlined 2016 = 504 := by
  sorry

end NUMINAMATH_CALUDE_doubly_underlined_count_l87_8780


namespace NUMINAMATH_CALUDE_number_with_75_halves_l87_8758

theorem number_with_75_halves (n : ℚ) : (∃ k : ℕ, n = k * (1/2) ∧ k = 75) → n = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_number_with_75_halves_l87_8758


namespace NUMINAMATH_CALUDE_correct_operation_l87_8744

theorem correct_operation (a b : ℝ) : -a^2*b + 2*a^2*b = a^2*b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l87_8744


namespace NUMINAMATH_CALUDE_expansion_terms_count_l87_8721

/-- The number of terms in the expansion of a product of two sums -/
def num_terms_in_expansion (m n : ℕ) : ℕ := m * n

/-- Theorem: The expansion of (a+b+c+d)(e+f+g+h+i) has 20 terms -/
theorem expansion_terms_count : num_terms_in_expansion 4 5 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l87_8721


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l87_8798

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 1 →
  (∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) →
  a 5 = 9 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l87_8798


namespace NUMINAMATH_CALUDE_triangle_area_l87_8788

/-- The area of a triangle with vertices A(0,0), B(1424233,2848467), and C(1424234,2848469) is 1/2 -/
theorem triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1424233, 2848467)
  let C : ℝ × ℝ := (1424234, 2848469)
  let triangle_area := abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2
  triangle_area = 1/2 := by
sorry

#eval (1424233 * 2848469 - 1424234 * 2848467) / 2

end NUMINAMATH_CALUDE_triangle_area_l87_8788


namespace NUMINAMATH_CALUDE_two_digit_sum_divisible_by_11_l87_8705

theorem two_digit_sum_divisible_by_11 (A B : ℕ) (h1 : A < 10) (h2 : B < 10) :
  ∃ k : ℤ, (10 * A + B : ℤ) + (10 * B + A : ℤ) = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_divisible_by_11_l87_8705


namespace NUMINAMATH_CALUDE_real_part_of_2_minus_i_l87_8747

theorem real_part_of_2_minus_i : Complex.re (2 - Complex.I) = 2 := by sorry

end NUMINAMATH_CALUDE_real_part_of_2_minus_i_l87_8747


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l87_8719

theorem adult_ticket_cost (num_adults num_children : ℕ) (total_bill child_ticket_cost : ℚ) :
  num_adults = 10 →
  num_children = 11 →
  total_bill = 124 →
  child_ticket_cost = 4 →
  (total_bill - num_children * child_ticket_cost) / num_adults = 8 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l87_8719


namespace NUMINAMATH_CALUDE_clock_cost_price_l87_8702

theorem clock_cost_price (total_clocks : ℕ) (clocks_sold_10_percent : ℕ) (clocks_sold_20_percent : ℕ)
  (profit_10_percent : ℝ) (profit_20_percent : ℝ) (uniform_profit : ℝ) (price_difference : ℝ) :
  total_clocks = 90 →
  clocks_sold_10_percent = 40 →
  clocks_sold_20_percent = 50 →
  profit_10_percent = 0.1 →
  profit_20_percent = 0.2 →
  uniform_profit = 0.15 →
  price_difference = 40 →
  ∃ (cost_price : ℝ),
    cost_price * (clocks_sold_10_percent * (1 + profit_10_percent) + 
      clocks_sold_20_percent * (1 + profit_20_percent)) - 
    cost_price * total_clocks * (1 + uniform_profit) = price_difference ∧
    cost_price = 80 :=
by sorry


end NUMINAMATH_CALUDE_clock_cost_price_l87_8702


namespace NUMINAMATH_CALUDE_solve_system_l87_8704

theorem solve_system (x y : ℝ) (eq1 : x + y = 15) (eq2 : x - y = 5) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l87_8704


namespace NUMINAMATH_CALUDE_jake_current_weight_l87_8728

/-- Jake's current weight in pounds -/
def jake_weight : ℕ := 219

/-- Jake's sister's current weight in pounds -/
def sister_weight : ℕ := 318 - jake_weight

theorem jake_current_weight : 
  (jake_weight + sister_weight = 318) ∧ 
  (jake_weight - 12 = 2 * (sister_weight + 4)) → 
  jake_weight = 219 := by sorry

end NUMINAMATH_CALUDE_jake_current_weight_l87_8728


namespace NUMINAMATH_CALUDE_octahedron_tetrahedron_volume_ratio_l87_8777

/-- The volume of a regular tetrahedron with edge length a -/
noncomputable def tetrahedron_volume (a : ℝ) : ℝ := sorry

/-- The volume of a regular octahedron with edge length a -/
noncomputable def octahedron_volume (a : ℝ) : ℝ := sorry

/-- Theorem stating that the volume of a regular octahedron is 4 times 
    the volume of a regular tetrahedron with the same edge length -/
theorem octahedron_tetrahedron_volume_ratio (a : ℝ) (h : a > 0) : 
  octahedron_volume a = 4 * tetrahedron_volume a := by sorry

end NUMINAMATH_CALUDE_octahedron_tetrahedron_volume_ratio_l87_8777


namespace NUMINAMATH_CALUDE_money_ratio_l87_8764

/-- Proves that given the total money between three people is $68, one person (Doug) has $32, 
    and another person (Josh) has 3/4 as much as Doug, the ratio of Josh's money to the 
    third person's (Brad's) money is 2:1. -/
theorem money_ratio (total : ℚ) (doug : ℚ) (josh : ℚ) (brad : ℚ) 
  (h1 : total = 68)
  (h2 : doug = 32)
  (h3 : josh = (3/4) * doug)
  (h4 : total = josh + doug + brad) :
  josh / brad = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_l87_8764


namespace NUMINAMATH_CALUDE_sunday_school_average_class_size_l87_8784

/-- Represents the number of students in each age group -/
structure AgeGroups where
  three_year_olds : Nat
  four_year_olds : Nat
  five_year_olds : Nat
  six_year_olds : Nat
  seven_year_olds : Nat
  eight_year_olds : Nat

/-- Calculates the average class size given the age groups -/
def averageClassSize (groups : AgeGroups) : Rat :=
  let class1 := groups.three_year_olds + groups.four_year_olds
  let class2 := groups.five_year_olds + groups.six_year_olds
  let class3 := groups.seven_year_olds + groups.eight_year_olds
  let totalStudents := class1 + class2 + class3
  (totalStudents : Rat) / 3

/-- The specific age groups given in the problem -/
def sundaySchoolGroups : AgeGroups := {
  three_year_olds := 13,
  four_year_olds := 20,
  five_year_olds := 15,
  six_year_olds := 22,
  seven_year_olds := 18,
  eight_year_olds := 25
}

theorem sunday_school_average_class_size :
  averageClassSize sundaySchoolGroups = 113 / 3 := by
  sorry

#eval averageClassSize sundaySchoolGroups

end NUMINAMATH_CALUDE_sunday_school_average_class_size_l87_8784


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l87_8766

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l87_8766


namespace NUMINAMATH_CALUDE_quadratic_solution_l87_8743

/-- Given nonzero real numbers c and d such that 2x^2 + cx + d = 0 has solutions 2c and 2d,
    prove that c = 1/2 and d = -5/8 -/
theorem quadratic_solution (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0)
  (h : ∀ x, 2 * x^2 + c * x + d = 0 ↔ x = 2 * c ∨ x = 2 * d) :
  c = 1/2 ∧ d = -5/8 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_l87_8743


namespace NUMINAMATH_CALUDE_arrangement_count_eq_960_l87_8708

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a row,
    where the elderly people must be adjacent but not at the ends. -/
def arrangement_count : ℕ :=
  let n_volunteers : ℕ := 5
  let n_elderly : ℕ := 2
  let n_total : ℕ := n_volunteers + n_elderly
  let n_ends : ℕ := 2
  let n_remaining_volunteers : ℕ := n_volunteers - n_ends
  let elderly_group : ℕ := 1  -- Treat adjacent elderly as one group

  (n_volunteers.choose n_ends) *    -- Ways to choose volunteers for the ends
  ((n_remaining_volunteers + elderly_group).factorial) *  -- Ways to arrange middle positions
  (n_elderly.factorial)             -- Ways to arrange within elderly group

theorem arrangement_count_eq_960 : arrangement_count = 960 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_eq_960_l87_8708


namespace NUMINAMATH_CALUDE_abhay_speed_l87_8706

theorem abhay_speed (distance : ℝ) (a s : ℝ) : 
  distance = 18 →
  distance / a = distance / s + 2 →
  distance / (2 * a) = distance / s - 1 →
  a = 81 / 10 := by
  sorry

end NUMINAMATH_CALUDE_abhay_speed_l87_8706


namespace NUMINAMATH_CALUDE_prime_representation_l87_8737

theorem prime_representation (p : ℕ) (hp : Prime p) (hp2 : p > 2) :
  (p % 8 = 1 ↔ ∃ x y : ℤ, p = x^2 + 16*y^2) ∧
  (p % 8 = 5 ↔ ∃ x y : ℤ, p = 4*x^2 + 4*x*y + 5*y^2) :=
by sorry

end NUMINAMATH_CALUDE_prime_representation_l87_8737


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l87_8711

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - x + 1 > 0) ↔ a > 1/4 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l87_8711


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l87_8797

theorem arithmetic_square_root_of_16 : ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l87_8797


namespace NUMINAMATH_CALUDE_choose_three_cooks_from_twelve_l87_8763

theorem choose_three_cooks_from_twelve (n : Nat) (k : Nat) : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_cooks_from_twelve_l87_8763


namespace NUMINAMATH_CALUDE_simplify_expression_l87_8700

theorem simplify_expression : 10 * (15 / 8) * (-40 / 45) = -50 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l87_8700


namespace NUMINAMATH_CALUDE_lisa_investment_interest_l87_8771

/-- Calculates the interest earned on an investment with annual compounding -/
def interest_earned (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * ((1 + rate) ^ years - 1)

/-- The interest earned on Lisa's investment -/
theorem lisa_investment_interest :
  let principal : ℝ := 2000
  let rate : ℝ := 0.02
  let years : ℕ := 10
  ∃ ε > 0, |interest_earned principal rate years - 438| < ε :=
by sorry

end NUMINAMATH_CALUDE_lisa_investment_interest_l87_8771


namespace NUMINAMATH_CALUDE_square_sum_theorem_l87_8769

theorem square_sum_theorem (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 3) :
  a^2 + b^2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l87_8769


namespace NUMINAMATH_CALUDE_max_value_of_f_l87_8741

noncomputable def f (x : ℝ) : ℝ := x^2 - 8*x + 6*Real.log x + 1

theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ f c ∧ f c = -6 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l87_8741


namespace NUMINAMATH_CALUDE_cycle_selling_price_l87_8772

theorem cycle_selling_price (cost_price : ℝ) (loss_percentage : ℝ) : 
  cost_price = 1200 → loss_percentage = 15 → 
  cost_price * (1 - loss_percentage / 100) = 1020 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l87_8772


namespace NUMINAMATH_CALUDE_candy_distribution_l87_8733

/-- The number of candy pieces in the bag -/
def total_candy : ℕ := 120

/-- Predicate to check if a number is a valid count of students -/
def is_valid_student_count (n : ℕ) : Prop :=
  n > 0 ∧ (total_candy - 1) % n = 0

/-- The theorem stating the possible number of students -/
theorem candy_distribution :
  ∃ (n : ℕ), is_valid_student_count n ∧ (n = 7 ∨ n = 17) :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_l87_8733


namespace NUMINAMATH_CALUDE_black_tshirt_cost_black_tshirt_cost_is_30_l87_8724

/-- The cost of black t-shirts given the sale conditions -/
theorem black_tshirt_cost (total_tshirts : ℕ) (sale_duration : ℕ) 
  (white_tshirt_cost : ℕ) (revenue_per_minute : ℕ) : ℕ :=
  let total_revenue := sale_duration * revenue_per_minute
  let num_black_tshirts := total_tshirts / 2
  let num_white_tshirts := total_tshirts / 2
  let white_tshirt_revenue := num_white_tshirts * white_tshirt_cost
  let black_tshirt_revenue := total_revenue - white_tshirt_revenue
  black_tshirt_revenue / num_black_tshirts

/-- The cost of black t-shirts is $30 given the specific sale conditions -/
theorem black_tshirt_cost_is_30 : 
  black_tshirt_cost 200 25 25 220 = 30 := by
  sorry

end NUMINAMATH_CALUDE_black_tshirt_cost_black_tshirt_cost_is_30_l87_8724


namespace NUMINAMATH_CALUDE_appropriate_grouping_l87_8794

theorem appropriate_grouping : 
  (43 + 27) + ((-78) + (-52)) = 43 + (-78) + 27 + (-52) := by
  sorry

end NUMINAMATH_CALUDE_appropriate_grouping_l87_8794


namespace NUMINAMATH_CALUDE_product_equals_zero_l87_8701

theorem product_equals_zero (a : ℤ) (h : a = 3) : 
  (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_zero_l87_8701


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l87_8748

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 1 / b) ≥ 4 + 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l87_8748


namespace NUMINAMATH_CALUDE_stratified_sample_probability_l87_8779

/-- Represents the number of classes selected from each grade -/
structure GradeSelection where
  grade1 : Nat
  grade2 : Nat
  grade3 : Nat

/-- The probability of selecting two classes from the same grade in a stratified sample -/
def probability_same_grade (selection : GradeSelection) : Rat :=
  let total_combinations := (selection.grade1 + selection.grade2 + selection.grade3).choose 2
  let same_grade_combinations := selection.grade1.choose 2
  same_grade_combinations / total_combinations

theorem stratified_sample_probability 
  (selection : GradeSelection)
  (h_ratio : selection.grade1 = 3 ∧ selection.grade2 = 2 ∧ selection.grade3 = 1) :
  probability_same_grade selection = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_probability_l87_8779


namespace NUMINAMATH_CALUDE_ef_fraction_of_gh_l87_8792

/-- Given a line segment GH with points E and F on it, prove that EF is 5/36 of GH 
    when GE is 3 times EH and GF is 8 times FH. -/
theorem ef_fraction_of_gh (G E F H : ℝ) : 
  G < E → E < F → F < H →  -- E and F lie on GH
  G - E = 3 * (H - E) →    -- GE is 3 times EH
  G - F = 8 * (H - F) →    -- GF is 8 times FH
  F - E = 5/36 * (H - G) := by
  sorry

end NUMINAMATH_CALUDE_ef_fraction_of_gh_l87_8792


namespace NUMINAMATH_CALUDE_h_value_at_4_l87_8776

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 2*x - 5

-- Define the properties of h
def is_valid_h (h : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ),
    (∀ x, f x = 0 ↔ x = a ∨ x = b ∨ x = c) ∧
    (∀ x, h x = 0 ↔ x = a^2 ∨ x = b^2 ∨ x = c^2) ∧
    (h 1 = 2)

-- Theorem statement
theorem h_value_at_4 (h : ℝ → ℝ) (hvalid : is_valid_h h) : h 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_h_value_at_4_l87_8776


namespace NUMINAMATH_CALUDE_exists_non_increasing_log_l87_8754

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem exists_non_increasing_log :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ¬(∀ (x y : ℝ), x < y → log a x < log a y) :=
sorry

end NUMINAMATH_CALUDE_exists_non_increasing_log_l87_8754


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l87_8726

/-- A function that returns true if a number has unique digits -/
def hasUniqueDigits (n : Nat) : Bool :=
  sorry

/-- A function that returns the sum of digits of a number -/
def sumOfDigits (n : Nat) : Nat :=
  sorry

/-- A function that checks if two numbers use all digits from 1 to 9 exactly once between them -/
def useAllDigitsOnce (x y : Nat) : Bool :=
  sorry

theorem smallest_sum_of_digits_of_sum (x y : Nat) : 
  x ≥ 100 ∧ x < 1000 ∧ 
  y ≥ 100 ∧ y < 1000 ∧ 
  hasUniqueDigits x ∧ 
  hasUniqueDigits y ∧ 
  useAllDigitsOnce x y ∧
  x + y < 1000 →
  ∃ (T : Nat), T = x + y ∧ sumOfDigits T ≥ 21 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l87_8726


namespace NUMINAMATH_CALUDE_nested_radical_sixteen_l87_8710

theorem nested_radical_sixteen (x : ℝ) : x = Real.sqrt (16 + x) → x = (1 + Real.sqrt 65) / 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_radical_sixteen_l87_8710


namespace NUMINAMATH_CALUDE_concyclic_projections_l87_8745

/-- Four points are concyclic if they lie on the same circle. -/
def Concyclic (A B C D : Point) : Prop := sorry

/-- The orthogonal projection of a point onto a line. -/
def OrthogonalProjection (P Q R : Point) : Point := sorry

/-- The main theorem: if A, B, C, D are concyclic, and A', C' are orthogonal projections of A, C 
    onto BD, and B', D' are orthogonal projections of B, D onto AC, then A', B', C', D' are concyclic. -/
theorem concyclic_projections 
  (A B C D : Point) 
  (h_concyclic : Concyclic A B C D) 
  (A' : Point) (h_A' : A' = OrthogonalProjection A B D)
  (C' : Point) (h_C' : C' = OrthogonalProjection C B D)
  (B' : Point) (h_B' : B' = OrthogonalProjection B A C)
  (D' : Point) (h_D' : D' = OrthogonalProjection D A C) :
  Concyclic A' B' C' D' :=
sorry

end NUMINAMATH_CALUDE_concyclic_projections_l87_8745


namespace NUMINAMATH_CALUDE_first_week_pushups_l87_8761

theorem first_week_pushups (initial_pushups : ℕ) (daily_increase : ℕ) (workout_days : ℕ) : 
  initial_pushups = 10 →
  daily_increase = 5 →
  workout_days = 3 →
  (initial_pushups + (initial_pushups + daily_increase) + (initial_pushups + 2 * daily_increase)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_first_week_pushups_l87_8761


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_two_l87_8714

theorem least_subtraction_for_divisibility_by_two (n : ℕ) (h : n = 9671) :
  ∃ (k : ℕ), k = 1 ∧ 
  (∀ (m : ℕ), m < k → ¬(∃ (q : ℕ), n - m = 2 * q)) ∧
  (∃ (q : ℕ), n - k = 2 * q) :=
sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_by_two_l87_8714


namespace NUMINAMATH_CALUDE_map_to_actual_ratio_l87_8759

-- Define the actual distance in kilometers
def actual_distance_km : ℝ := 6

-- Define the map distance in centimeters
def map_distance_cm : ℝ := 20

-- Define the conversion factor from kilometers to centimeters
def km_to_cm : ℝ := 100000

-- Theorem statement
theorem map_to_actual_ratio :
  (map_distance_cm / (actual_distance_km * km_to_cm)) = (1 / 30000) := by
  sorry

end NUMINAMATH_CALUDE_map_to_actual_ratio_l87_8759


namespace NUMINAMATH_CALUDE_factorization_equality_l87_8781

theorem factorization_equality (a b x y : ℝ) :
  8 * a * x - b * y + 4 * a * y - 2 * b * x = (4 * a - b) * (2 * x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l87_8781


namespace NUMINAMATH_CALUDE_minutes_conversion_l87_8722

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- Converts minutes to seconds -/
def minutes_to_seconds (minutes : ℚ) : ℚ :=
  minutes * seconds_per_minute

/-- Converts minutes to hours -/
def minutes_to_hours (minutes : ℚ) : ℚ :=
  minutes / minutes_per_hour

theorem minutes_conversion (minutes : ℚ) :
  minutes = 25/2 →
  minutes_to_seconds minutes = 750 ∧ minutes_to_hours minutes = 5/24 := by
  sorry

end NUMINAMATH_CALUDE_minutes_conversion_l87_8722


namespace NUMINAMATH_CALUDE_negation_is_false_l87_8736

theorem negation_is_false : 
  ¬(∀ x y : ℝ, (x > 2 ∧ y > 3) → x + y > 5) = False :=
sorry

end NUMINAMATH_CALUDE_negation_is_false_l87_8736


namespace NUMINAMATH_CALUDE_toy_problem_solution_l87_8775

/-- Represents the problem of Mia and her mom putting toys in a box -/
def ToyProblem (totalToys : ℕ) (putIn : ℕ) (takeOut : ℕ) (cycleTime : ℚ) : Prop :=
  let netIncrease := putIn - takeOut
  let cycles := (totalToys - 1) / netIncrease + 1
  cycles * cycleTime / 60 = 12.5

/-- The theorem statement for the toy problem -/
theorem toy_problem_solution :
  ToyProblem 50 5 3 (30 / 60) :=
sorry

end NUMINAMATH_CALUDE_toy_problem_solution_l87_8775


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l87_8799

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (5 * n) % 26 = 1463 % 26 ∧ ∀ (m : ℕ), m > 0 → (5 * m) % 26 = 1463 % 26 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l87_8799


namespace NUMINAMATH_CALUDE_team_ages_mode_l87_8725

def team_ages : List Nat := [17, 17, 18, 18, 16, 18, 17, 15, 18, 18, 17, 16, 18, 17, 18, 14]

def mode (l : List Nat) : Nat :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem team_ages_mode :
  mode team_ages = 18 := by
  sorry

end NUMINAMATH_CALUDE_team_ages_mode_l87_8725


namespace NUMINAMATH_CALUDE_units_digit_factorial_sum_7_l87_8742

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def units_digit (n : ℕ) : ℕ := n % 10

def factorial_sum (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_factorial_sum_7 : 
  units_digit (factorial_sum 7) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_factorial_sum_7_l87_8742


namespace NUMINAMATH_CALUDE_rotate_point_around_OA_l87_8734

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Rotate a point around a ray by a given angle -/
def rotateAroundRay (p : Point3D) (origin : Point3D) (axis : Point3D) (angle : ℝ) : Point3D :=
  sorry

/-- The theorem to prove -/
theorem rotate_point_around_OA : 
  let A : Point3D := ⟨1, 1, 1⟩
  let P : Point3D := ⟨1, 1, 0⟩
  let O : Point3D := ⟨0, 0, 0⟩
  let angle : ℝ := π / 3  -- 60 degrees in radians
  let rotated_P : Point3D := rotateAroundRay P O A angle
  rotated_P = ⟨1/3, 4/3, 1/3⟩ := by sorry

end NUMINAMATH_CALUDE_rotate_point_around_OA_l87_8734


namespace NUMINAMATH_CALUDE_sports_club_intersection_l87_8785

/-- Given a sports club with the following properties:
  - There are 30 total members
  - 16 members play badminton
  - 19 members play tennis
  - 2 members play neither badminton nor tennis
  Prove that 7 members play both badminton and tennis -/
theorem sports_club_intersection (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) :
  total = 30 →
  badminton = 16 →
  tennis = 19 →
  neither = 2 →
  badminton + tennis - (total - neither) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_intersection_l87_8785


namespace NUMINAMATH_CALUDE_rachels_reading_homework_l87_8793

/-- Given that Rachel had 9 pages of math homework and 7 more pages of math homework than reading homework, prove that she had 2 pages of reading homework. -/
theorem rachels_reading_homework (math_homework : ℕ) (reading_homework : ℕ) 
  (h1 : math_homework = 9)
  (h2 : math_homework = reading_homework + 7) :
  reading_homework = 2 := by
  sorry

end NUMINAMATH_CALUDE_rachels_reading_homework_l87_8793


namespace NUMINAMATH_CALUDE_prob_red_two_cans_l87_8740

/-- Represents a can containing red and white balls -/
structure Can where
  red : ℕ
  white : ℕ

/-- The probability of drawing a red ball from a can -/
def probRed (c : Can) : ℚ :=
  c.red / (c.red + c.white)

/-- The probability of drawing a white ball from a can -/
def probWhite (c : Can) : ℚ :=
  c.white / (c.red + c.white)

/-- The probability of drawing a red ball from can B after transferring a ball from can A -/
def probRedAfterTransfer (a b : Can) : ℚ :=
  probRed a * probRed (Can.mk (b.red + 1) b.white) +
  probWhite a * probRed (Can.mk b.red (b.white + 1))

theorem prob_red_two_cans (a b : Can) (ha : a.red = 2 ∧ a.white = 3) (hb : b.red = 4 ∧ b.white = 1) :
  probRedAfterTransfer a b = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_two_cans_l87_8740


namespace NUMINAMATH_CALUDE_chipmunks_went_away_count_l87_8712

/-- Represents the chipmunk population in a forest --/
structure ChipmunkForest where
  originalFamilies : ℕ
  remainingFamilies : ℕ
  avgMembersRemaining : ℕ
  avgMembersLeft : ℕ

/-- Calculates the number of chipmunks that went away --/
def chipmunksWentAway (forest : ChipmunkForest) : ℕ :=
  (forest.originalFamilies - forest.remainingFamilies) * forest.avgMembersLeft

/-- Theorem stating the number of chipmunks that went away --/
theorem chipmunks_went_away_count (forest : ChipmunkForest) 
  (h1 : forest.originalFamilies = 86)
  (h2 : forest.remainingFamilies = 21)
  (h3 : forest.avgMembersRemaining = 15)
  (h4 : forest.avgMembersLeft = 18) :
  chipmunksWentAway forest = 1170 := by
  sorry

#eval chipmunksWentAway { originalFamilies := 86, remainingFamilies := 21, avgMembersRemaining := 15, avgMembersLeft := 18 }

end NUMINAMATH_CALUDE_chipmunks_went_away_count_l87_8712


namespace NUMINAMATH_CALUDE_flower_beds_fraction_l87_8773

/-- Represents a rectangular yard with two congruent isosceles right triangular flower beds -/
structure YardWithFlowerBeds where
  /-- Length of the shorter parallel side of the trapezoid -/
  short_side : ℝ
  /-- Length of the longer parallel side of the trapezoid -/
  long_side : ℝ
  /-- Assumption that the short side is 20 meters -/
  short_side_eq : short_side = 20
  /-- Assumption that the long side is 30 meters -/
  long_side_eq : long_side = 30

/-- The fraction of the yard occupied by the flower beds is 1/6 -/
theorem flower_beds_fraction (yard : YardWithFlowerBeds) : 
  (yard.long_side - yard.short_side)^2 / (4 * yard.long_side * (yard.long_side - yard.short_side)) = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_flower_beds_fraction_l87_8773


namespace NUMINAMATH_CALUDE_annie_diorama_building_time_l87_8751

/-- The time Annie spent building her diorama -/
def building_time (planning_time : ℕ) : ℕ := 3 * planning_time - 5

/-- The total time Annie spent on her diorama project -/
def total_time (planning_time : ℕ) : ℕ := building_time planning_time + planning_time

theorem annie_diorama_building_time :
  ∃ (planning_time : ℕ), total_time planning_time = 67 ∧ building_time planning_time = 49 := by
sorry

end NUMINAMATH_CALUDE_annie_diorama_building_time_l87_8751


namespace NUMINAMATH_CALUDE_functional_equation_solution_l87_8727

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x * f y) : 
  (∀ x : ℚ, f x = 0) ∨ (∀ x : ℚ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l87_8727


namespace NUMINAMATH_CALUDE_equal_population_after_17_years_l87_8730

/-- The number of years needed for two villages' populations to become equal -/
def years_until_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  (x_initial - y_initial) / (x_decrease + y_increase)

/-- Theorem stating that the populations of Village X and Village Y will be equal after 17 years -/
theorem equal_population_after_17_years :
  years_until_equal_population 76000 1200 42000 800 = 17 := by
  sorry

end NUMINAMATH_CALUDE_equal_population_after_17_years_l87_8730


namespace NUMINAMATH_CALUDE_survival_rate_definition_correct_l87_8770

/-- Survival rate is defined as the percentage of living seedlings out of the total seedlings -/
def survival_rate (living_seedlings total_seedlings : ℕ) : ℚ :=
  (living_seedlings : ℚ) / (total_seedlings : ℚ) * 100

/-- The given definition of survival rate is correct -/
theorem survival_rate_definition_correct :
  ∀ (living_seedlings total_seedlings : ℕ),
  survival_rate living_seedlings total_seedlings =
  (living_seedlings : ℚ) / (total_seedlings : ℚ) * 100 :=
by
  sorry

end NUMINAMATH_CALUDE_survival_rate_definition_correct_l87_8770


namespace NUMINAMATH_CALUDE_paint_area_is_127_l87_8718

/-- Calculates the area to be painted on a wall with two windows. -/
def areaToPaint (wallHeight wallLength window1Height window1Width window2Height window2Width : ℝ) : ℝ :=
  wallHeight * wallLength - (window1Height * window1Width + window2Height * window2Width)

/-- Proves that the area to be painted is 127 square feet given the specified dimensions. -/
theorem paint_area_is_127 :
  areaToPaint 10 15 3 5 2 4 = 127 := by
  sorry

#eval areaToPaint 10 15 3 5 2 4

end NUMINAMATH_CALUDE_paint_area_is_127_l87_8718


namespace NUMINAMATH_CALUDE_joes_fast_food_cost_purchase_cost_l87_8720

/-- The cost of purchasing sandwiches and sodas at Joe's Fast Food -/
theorem joes_fast_food_cost : ℕ → ℕ → ℕ
  | sandwich_count, soda_count => 
    4 * sandwich_count + 3 * soda_count

/-- Proof that purchasing 7 sandwiches and 9 sodas costs $55 -/
theorem purchase_cost : joes_fast_food_cost 7 9 = 55 := by
  sorry

end NUMINAMATH_CALUDE_joes_fast_food_cost_purchase_cost_l87_8720


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l87_8703

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l87_8703


namespace NUMINAMATH_CALUDE_f_at_7_equals_3_l87_8767

-- Define the function f
def f (p q : ℝ) (x : ℝ) : ℝ := p * x^3 + q * x + 5

-- State the theorem
theorem f_at_7_equals_3 (p q b : ℝ) :
  (f p q (-7) = Real.sqrt 2 * b + 1) →
  f p q 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_at_7_equals_3_l87_8767


namespace NUMINAMATH_CALUDE_sin_identity_l87_8787

theorem sin_identity (α : Real) (h : Real.sin (α - π/4) = 1/2) :
  Real.sin (5*π/4 - α) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_identity_l87_8787


namespace NUMINAMATH_CALUDE_discriminant_not_necessary_nor_sufficient_l87_8707

/-- The function f(x) = ax^2 + bx + c --/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that the graph of f is always above the x-axis --/
def always_above (a b c : ℝ) : Prop :=
  ∀ x, f a b c x > 0

/-- The discriminant condition --/
def discriminant_condition (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

theorem discriminant_not_necessary_nor_sufficient :
  ¬(∀ a b c : ℝ, discriminant_condition a b c ↔ always_above a b c) :=
sorry

end NUMINAMATH_CALUDE_discriminant_not_necessary_nor_sufficient_l87_8707


namespace NUMINAMATH_CALUDE_proportion_solutions_l87_8715

theorem proportion_solutions :
  (∃ x : ℚ, 0.75 / (1/2) = 12 / x ∧ x = 8) ∧
  (∃ x : ℚ, 0.7 / x = 14 / 5 ∧ x = 0.25) ∧
  (∃ x : ℚ, (2/15) / (1/6) = x / (2/3) ∧ x = 8/15) ∧
  (∃ x : ℚ, 4 / 4.5 = x / 27 ∧ x = 24) := by
  sorry

end NUMINAMATH_CALUDE_proportion_solutions_l87_8715


namespace NUMINAMATH_CALUDE_carries_box_capacity_l87_8756

/-- Represents a rectangular box with height, width, and length -/
structure Box where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box -/
def Box.volume (b : Box) : ℝ := b.height * b.width * b.length

/-- Represents the number of jellybeans a box can hold -/
def jellybeanCapacity (b : Box) (density : ℝ) : ℝ := b.volume * density

/-- Theorem: Carrie's box capacity given Bert's box capacity -/
theorem carries_box_capacity
  (bert_box : Box)
  (bert_capacity : ℝ)
  (density : ℝ)
  (h1 : jellybeanCapacity bert_box density = bert_capacity)
  (h2 : bert_capacity = 150)
  (carrie_box : Box)
  (h3 : carrie_box.height = 3 * bert_box.height)
  (h4 : carrie_box.width = 2 * bert_box.width)
  (h5 : carrie_box.length = 4 * bert_box.length) :
  jellybeanCapacity carrie_box density = 3600 := by
  sorry

end NUMINAMATH_CALUDE_carries_box_capacity_l87_8756


namespace NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l87_8786

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 3| = |x + 2| ∧ |x + 2| = |x - 5| ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_absolute_value_equation_l87_8786


namespace NUMINAMATH_CALUDE_sheep_transaction_gain_l87_8755

/-- Calculates the percent gain on a sheep transaction given specific conditions. -/
theorem sheep_transaction_gain : ∀ (x : ℝ),
  x > 0 →  -- x represents the cost per sheep
  let total_cost : ℝ := 850 * x
  let first_sale_revenue : ℝ := total_cost
  let first_sale_price_per_sheep : ℝ := first_sale_revenue / 800
  let second_sale_price_per_sheep : ℝ := first_sale_price_per_sheep * 1.1
  let second_sale_revenue : ℝ := second_sale_price_per_sheep * 50
  let total_revenue : ℝ := first_sale_revenue + second_sale_revenue
  let profit : ℝ := total_revenue - total_cost
  let percent_gain : ℝ := (profit / total_cost) * 100
  percent_gain = 6.875 := by
  sorry


end NUMINAMATH_CALUDE_sheep_transaction_gain_l87_8755


namespace NUMINAMATH_CALUDE_odd_function_log_property_l87_8746

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_log_property (f : ℝ → ℝ) 
  (h_odd : odd_function f) 
  (h_pos : ∀ x > 0, f x = Real.log (x + 1)) : 
  ∀ x < 0, f x = -Real.log (1 - x) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_log_property_l87_8746


namespace NUMINAMATH_CALUDE_nursery_school_age_distribution_l87_8789

theorem nursery_school_age_distribution (total : ℕ) (four_and_older : ℕ) (not_between_three_and_four : ℕ) :
  total = 50 →
  four_and_older = total / 10 →
  not_between_three_and_four = 25 →
  four_and_older + (total - four_and_older - (total - not_between_three_and_four)) = not_between_three_and_four →
  total - four_and_older - (total - not_between_three_and_four) = 20 := by
sorry

end NUMINAMATH_CALUDE_nursery_school_age_distribution_l87_8789


namespace NUMINAMATH_CALUDE_fourth_month_sales_l87_8753

def sales_problem (sales1 sales2 sales3 sales5 sales6 : ℕ) (average : ℕ) : Prop :=
  let total := average * 6
  let known_sales := sales1 + sales2 + sales3 + sales5 + sales6
  total - known_sales = 7230

theorem fourth_month_sales :
  sales_problem 6735 6927 6855 6562 4691 6500 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_month_sales_l87_8753


namespace NUMINAMATH_CALUDE_inequality_proof_l87_8795

theorem inequality_proof (x y : ℝ) (m n : ℤ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) :
  (1 - x^n.toNat)^m.toNat + (1 - y^m.toNat)^n.toNat ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l87_8795


namespace NUMINAMATH_CALUDE_main_theorem_l87_8791

/-- Definition of H function -/
def is_H_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ + f x₂) / 2 > f ((x₁ + x₂) / 2)

/-- Definition of even function -/
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

/-- The main theorem -/
theorem main_theorem (c : ℝ) (f : ℝ → ℝ) (h : f = fun x ↦ x^2 + c*x) 
  (h_even : is_even f) : c = 0 ∧ is_H_function f := by
  sorry

end NUMINAMATH_CALUDE_main_theorem_l87_8791


namespace NUMINAMATH_CALUDE_jason_age_2004_l87_8709

/-- Jason's age at the end of 1997 -/
def jason_age_1997 : ℝ := 35.5

/-- Jason's grandmother's age at the end of 1997 -/
def grandmother_age_1997 : ℝ := 3 * jason_age_1997

/-- The sum of Jason's and his grandmother's birth years -/
def birth_years_sum : ℕ := 3852

/-- The year we're considering for Jason's age -/
def target_year : ℕ := 2004

/-- The reference year for ages -/
def reference_year : ℕ := 1997

theorem jason_age_2004 :
  jason_age_1997 + (target_year - reference_year) = 42.5 ∧
  jason_age_1997 = grandmother_age_1997 / 3 ∧
  (reference_year - jason_age_1997) + (reference_year - grandmother_age_1997) = birth_years_sum :=
by sorry

end NUMINAMATH_CALUDE_jason_age_2004_l87_8709


namespace NUMINAMATH_CALUDE_mila_coin_collection_value_l87_8731

/-- The total value of Mila's coin collection -/
def total_value (gold_coins silver_coins : ℕ) (gold_value silver_value : ℚ) : ℚ :=
  gold_coins * gold_value + silver_coins * silver_value

/-- Theorem stating the total value of Mila's coin collection -/
theorem mila_coin_collection_value :
  let gold_coins : ℕ := 20
  let silver_coins : ℕ := 15
  let gold_value : ℚ := 10 / 4
  let silver_value : ℚ := 15 / 5
  total_value gold_coins silver_coins gold_value silver_value = 95
  := by sorry

end NUMINAMATH_CALUDE_mila_coin_collection_value_l87_8731


namespace NUMINAMATH_CALUDE_prob_at_edge_is_two_thirds_l87_8783

/-- The number of students in the row -/
def num_students : ℕ := 3

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := num_students.factorial

/-- The number of arrangements where a specific student is at the edge -/
def edge_arrangements : ℕ := 2 * (num_students - 1).factorial

/-- The probability of a specific student standing at the edge -/
def prob_at_edge : ℚ := edge_arrangements / total_arrangements

theorem prob_at_edge_is_two_thirds : 
  prob_at_edge = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_at_edge_is_two_thirds_l87_8783
