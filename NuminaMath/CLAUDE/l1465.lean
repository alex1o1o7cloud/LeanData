import Mathlib

namespace NUMINAMATH_CALUDE_oranges_per_tree_l1465_146531

/-- Represents the number of oranges picked by Betty -/
def betty_oranges : ℕ := 15

/-- Represents the number of oranges picked by Bill -/
def bill_oranges : ℕ := 12

/-- Represents the number of oranges picked by Frank -/
def frank_oranges : ℕ := 3 * (betty_oranges + bill_oranges)

/-- Represents the number of seeds Frank planted -/
def seeds_planted : ℕ := 2 * frank_oranges

/-- Represents the total number of oranges Philip can pick -/
def philip_total_oranges : ℕ := 810

/-- Theorem stating that the number of oranges per tree for Philip to pick is 5 -/
theorem oranges_per_tree :
  philip_total_oranges / seeds_planted = 5 := by sorry

end NUMINAMATH_CALUDE_oranges_per_tree_l1465_146531


namespace NUMINAMATH_CALUDE_problem_solution_l1465_146514

def second_order_det (a b c d : ℝ) : ℝ := a * d - b * c

theorem problem_solution :
  (second_order_det 3 (-2) 4 (-3) = -1) ∧
  (∀ x : ℝ, second_order_det (2*x-3) (x+2) 2 4 = 6*x - 16) ∧
  (second_order_det 5 6 2 4 = 8) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1465_146514


namespace NUMINAMATH_CALUDE_pyramid_volume_l1465_146574

/-- The volume of a square-based pyramid with given dimensions -/
theorem pyramid_volume (base_side : ℝ) (apex_distance : ℝ) (volume : ℝ) : 
  base_side = 24 →
  apex_distance = Real.sqrt 364 →
  volume = (1 / 3) * base_side^2 * Real.sqrt ((apex_distance^2) - (1/2 * base_side * Real.sqrt 2)^2) →
  volume = 384 * Real.sqrt 19 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_volume_l1465_146574


namespace NUMINAMATH_CALUDE_childrens_ticket_cost_l1465_146578

/-- Prove that the cost of a children's ticket is $4.50 -/
theorem childrens_ticket_cost
  (adult_ticket_cost : ℝ)
  (total_tickets : ℕ)
  (total_revenue : ℝ)
  (childrens_tickets : ℕ)
  (h1 : adult_ticket_cost = 6)
  (h2 : total_tickets = 400)
  (h3 : total_revenue = 2100)
  (h4 : childrens_tickets = 200) :
  ∃ (childrens_ticket_cost : ℝ),
    childrens_ticket_cost * childrens_tickets +
    adult_ticket_cost * (total_tickets - childrens_tickets) = total_revenue ∧
    childrens_ticket_cost = 4.5 :=
by
  sorry


end NUMINAMATH_CALUDE_childrens_ticket_cost_l1465_146578


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1465_146557

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 10 = 0 ∧ x^4 - 3*x^3 - 4*x^2 + 12*x + 9 = 219 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1465_146557


namespace NUMINAMATH_CALUDE_sum_of_variables_l1465_146596

theorem sum_of_variables (a b c : ℝ) 
  (eq1 : b + c = 12 - 3*a)
  (eq2 : a + c = -14 - 3*b)
  (eq3 : a + b = 7 - 3*c) :
  2*a + 2*b + 2*c = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l1465_146596


namespace NUMINAMATH_CALUDE_average_age_across_rooms_l1465_146518

theorem average_age_across_rooms (room_a_people room_b_people room_c_people : ℕ)
                                 (room_a_avg room_b_avg room_c_avg : ℚ)
                                 (h1 : room_a_people = 8)
                                 (h2 : room_b_people = 5)
                                 (h3 : room_c_people = 7)
                                 (h4 : room_a_avg = 35)
                                 (h5 : room_b_avg = 30)
                                 (h6 : room_c_avg = 25) :
  (room_a_people * room_a_avg + room_b_people * room_b_avg + room_c_people * room_c_avg) /
  (room_a_people + room_b_people + room_c_people : ℚ) = 30.25 := by
  sorry

end NUMINAMATH_CALUDE_average_age_across_rooms_l1465_146518


namespace NUMINAMATH_CALUDE_nth_equation_solutions_l1465_146595

theorem nth_equation_solutions (n : ℕ+) :
  let eq := fun x : ℝ => x + (n^2 + n) / x + (2*n + 1)
  eq (-n : ℝ) = 0 ∧ eq (-(n + 1) : ℝ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_nth_equation_solutions_l1465_146595


namespace NUMINAMATH_CALUDE_alphaBetaArrangementsCount_l1465_146526

/-- The number of distinct arrangements of 9 letters, where one letter appears 4 times
    and six other letters appear once each. -/
def alphaBetaArrangements : ℕ :=
  Nat.factorial 9 / (Nat.factorial 4 * (Nat.factorial 1)^6)

/-- Theorem stating that the number of distinct arrangements of letters in "alpha beta"
    under the given conditions is 15120. -/
theorem alphaBetaArrangementsCount : alphaBetaArrangements = 15120 := by
  sorry

end NUMINAMATH_CALUDE_alphaBetaArrangementsCount_l1465_146526


namespace NUMINAMATH_CALUDE_square_area_ratio_l1465_146521

theorem square_area_ratio (s : ℝ) (h : s > 0) : 
  (2 * s)^2 / s^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1465_146521


namespace NUMINAMATH_CALUDE_count_integer_root_cases_correct_l1465_146575

/-- The number of real values 'a' for which x^2 + ax + 12a = 0 has only integer roots -/
def count_integer_root_cases : ℕ := 8

/-- A function that returns true if the quadratic equation x^2 + ax + 12a = 0 has only integer roots -/
def has_only_integer_roots (a : ℝ) : Prop :=
  ∃ p q : ℤ, ∀ x : ℝ, x^2 + a*x + 12*a = 0 ↔ x = p ∨ x = q

/-- The theorem stating that there are exactly 8 real numbers 'a' for which
    the quadratic equation x^2 + ax + 12a = 0 has only integer roots -/
theorem count_integer_root_cases_correct :
  (∃ S : Finset ℝ, Finset.card S = count_integer_root_cases ∧
    (∀ a : ℝ, a ∈ S ↔ has_only_integer_roots a)) := by
  sorry


end NUMINAMATH_CALUDE_count_integer_root_cases_correct_l1465_146575


namespace NUMINAMATH_CALUDE_same_color_probability_value_l1465_146551

/-- The number of green balls in the bag -/
def green_balls : ℕ := 8

/-- The number of red balls in the bag -/
def red_balls : ℕ := 5

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := green_balls + red_balls + blue_balls

/-- The probability of drawing two balls of the same color with replacement -/
def same_color_probability : ℚ :=
  (green_balls / total_balls) ^ 2 +
  (red_balls / total_balls) ^ 2 +
  (blue_balls / total_balls) ^ 2

theorem same_color_probability_value :
  same_color_probability = 49 / 128 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_value_l1465_146551


namespace NUMINAMATH_CALUDE_divisibility_by_six_divisibility_by_120_divisibility_by_48_divisibility_by_1152_not_always_divisible_by_720_l1465_146512

-- Part (a)
theorem divisibility_by_six (a b c : ℤ) (h : 6 ∣ (a + b + c)) : 6 ∣ (a^5 + b^3 + c) := by sorry

-- Part (b)
theorem divisibility_by_120 (n : ℤ) : 120 ∣ (n^5 - 5*n^3 + 4*n) := by sorry

-- Part (c)
theorem divisibility_by_48 (n : ℤ) (h : Odd n) : 48 ∣ (n^3 + 3*n^2 - n - 3) := by sorry

-- Part (d)
theorem divisibility_by_1152 (n : ℤ) (h : Odd n) : 1152 ∣ (n^8 - n^6 - n^4 + n^2) := by sorry

-- Part (e)
theorem not_always_divisible_by_720 : ∃ n : ℤ, ¬(720 ∣ (n*(n^2 - 1)*(n^2 - 4))) := by sorry

end NUMINAMATH_CALUDE_divisibility_by_six_divisibility_by_120_divisibility_by_48_divisibility_by_1152_not_always_divisible_by_720_l1465_146512


namespace NUMINAMATH_CALUDE_square_root_three_expansion_square_root_three_specific_case_simplify_square_root_expression_l1465_146571

-- Part 1
theorem square_root_three_expansion (a b m n : ℕ+) :
  a + b * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 →
  a = m^2 + 3*n^2 ∧ b = 2*m*n :=
sorry

-- Part 2
theorem square_root_three_specific_case (a m n : ℕ+) :
  a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 →
  a = 13 ∨ a = 7 :=
sorry

-- Part 3
theorem simplify_square_root_expression :
  Real.sqrt (6 + 2 * Real.sqrt 5) = 1 + Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_square_root_three_specific_case_simplify_square_root_expression_l1465_146571


namespace NUMINAMATH_CALUDE_special_polyhedron_body_diagonals_l1465_146564

/-- A convex polyhedron with specific face composition -/
structure SpecialPolyhedron where
  /-- The polyhedron is convex -/
  is_convex : Bool
  /-- Number of square faces -/
  num_squares : Nat
  /-- Number of regular hexagon faces -/
  num_hexagons : Nat
  /-- Number of regular octagon faces -/
  num_octagons : Nat
  /-- At each vertex, a square, a hexagon, and an octagon meet -/
  vertex_composition : Bool
  /-- The surface is composed of exactly 12 squares, 8 hexagons, and 6 octagons -/
  face_composition : num_squares = 12 ∧ num_hexagons = 8 ∧ num_octagons = 6

/-- The number of body diagonals in the special polyhedron -/
def num_body_diagonals (p : SpecialPolyhedron) : Nat :=
  sorry

/-- Theorem: The number of body diagonals in the special polyhedron is 840 -/
theorem special_polyhedron_body_diagonals (p : SpecialPolyhedron) : 
  num_body_diagonals p = 840 := by
  sorry

end NUMINAMATH_CALUDE_special_polyhedron_body_diagonals_l1465_146564


namespace NUMINAMATH_CALUDE_squares_after_six_steps_l1465_146504

/-- The number of squares after n steps, given an initial configuration of 5 squares 
    and each step adds 3 squares -/
def num_squares (n : ℕ) : ℕ := 5 + 3 * n

/-- Theorem stating that after 6 steps, there are 23 squares -/
theorem squares_after_six_steps : num_squares 6 = 23 := by
  sorry

end NUMINAMATH_CALUDE_squares_after_six_steps_l1465_146504


namespace NUMINAMATH_CALUDE_point_not_in_quadrants_III_IV_l1465_146513

theorem point_not_in_quadrants_III_IV (m : ℝ) : 
  let A : ℝ × ℝ := (m, m^2 + 1)
  ¬(A.1 ≤ 0 ∧ A.2 ≤ 0) ∧ ¬(A.1 ≥ 0 ∧ A.2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_quadrants_III_IV_l1465_146513


namespace NUMINAMATH_CALUDE_total_letters_received_l1465_146589

theorem total_letters_received (brother_letters : ℕ) (greta_extra : ℕ) (mother_multiplier : ℕ) : 
  brother_letters = 40 → 
  greta_extra = 10 → 
  mother_multiplier = 2 → 
  (brother_letters + (brother_letters + greta_extra) + 
   mother_multiplier * (brother_letters + (brother_letters + greta_extra))) = 270 :=
by
  sorry

end NUMINAMATH_CALUDE_total_letters_received_l1465_146589


namespace NUMINAMATH_CALUDE_garden_ratio_l1465_146559

/-- Represents a rectangular garden with given perimeter and length -/
structure RectangularGarden where
  perimeter : ℝ
  length : ℝ
  width : ℝ
  perimeter_eq : perimeter = 2 * (length + width)
  length_positive : length > 0
  width_positive : width > 0

/-- The ratio of length to width for a rectangular garden with perimeter 150 and length 50 is 2:1 -/
theorem garden_ratio (garden : RectangularGarden) 
  (h_perimeter : garden.perimeter = 150) 
  (h_length : garden.length = 50) : 
  garden.length / garden.width = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l1465_146559


namespace NUMINAMATH_CALUDE_captain_selection_criterion_l1465_146561

-- Define the universe of players
variable (Player : Type)

-- Define predicates
variable (attends_all_sessions : Player → Prop)
variable (always_on_time : Player → Prop)
variable (considered_for_captain : Player → Prop)

-- Theorem statement
theorem captain_selection_criterion
  (h : ∀ p : Player, (attends_all_sessions p ∧ always_on_time p) → considered_for_captain p) :
  ∀ p : Player, ¬(considered_for_captain p) → (¬(attends_all_sessions p) ∨ ¬(always_on_time p)) :=
by sorry

end NUMINAMATH_CALUDE_captain_selection_criterion_l1465_146561


namespace NUMINAMATH_CALUDE_cosine_derivative_at_pi_sixth_l1465_146520

theorem cosine_derivative_at_pi_sixth :
  let f : ℝ → ℝ := λ x ↦ Real.cos x
  (deriv f) (π / 6) = - (1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cosine_derivative_at_pi_sixth_l1465_146520


namespace NUMINAMATH_CALUDE_team_ratio_is_correct_l1465_146576

/-- Represents a co-ed softball team -/
structure Team where
  total_players : ℕ
  men : ℕ
  women : ℕ
  h_total : men + women = total_players
  h_ratio : ∀ (group : ℕ), group * 3 ≤ total_players → group * 2 = women - men

/-- The specific team in the problem -/
def problem_team : Team where
  total_players := 25
  men := 8
  women := 17
  h_total := by sorry
  h_ratio := by sorry

theorem team_ratio_is_correct (team : Team) (h : team = problem_team) :
  team.men = 8 ∧ team.women = 17 := by sorry

end NUMINAMATH_CALUDE_team_ratio_is_correct_l1465_146576


namespace NUMINAMATH_CALUDE_f_monotone_decreasing_l1465_146540

noncomputable def f (x : ℝ) := x * Real.log x

theorem f_monotone_decreasing :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.exp (-1)),
    StrictMonoOn f (Set.Ioo 0 (Real.exp (-1))) :=
by
  sorry

end NUMINAMATH_CALUDE_f_monotone_decreasing_l1465_146540


namespace NUMINAMATH_CALUDE_min_value_x2_plus_y2_l1465_146567

theorem min_value_x2_plus_y2 (x y : ℝ) (h : 5 * x^2 * y^2 + y^4 = 1) :
  x^2 + y^2 ≥ 4/5 ∧ ∃ x y : ℝ, 5 * x^2 * y^2 + y^4 = 1 ∧ x^2 + y^2 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_y2_l1465_146567


namespace NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l1465_146581

theorem shaded_area_of_concentric_circles (R r : ℝ) : 
  R > 0 → r > 0 → r = R / 2 → R^2 * π = 81 * π → 
  (π * R^2) / 2 + (π * r^2) / 2 = 50.625 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_concentric_circles_l1465_146581


namespace NUMINAMATH_CALUDE_room_tiling_theorem_l1465_146522

/-- Calculates the number of tiles needed to cover a rectangular room with a border of larger tiles -/
def tilesNeeded (roomLength roomWidth borderTileSize innerTileSize : ℕ) : ℕ :=
  let borderTiles := 2 * (roomLength / borderTileSize + roomWidth / borderTileSize) - 4
  let innerLength := roomLength - 2 * borderTileSize
  let innerWidth := roomWidth - 2 * borderTileSize
  let innerTiles := (innerLength / innerTileSize) * (innerWidth / innerTileSize)
  borderTiles + innerTiles

/-- The theorem stating that 310 tiles are needed for the given room specifications -/
theorem room_tiling_theorem :
  tilesNeeded 24 18 2 1 = 310 := by
  sorry

end NUMINAMATH_CALUDE_room_tiling_theorem_l1465_146522


namespace NUMINAMATH_CALUDE_fifty_cent_items_count_l1465_146545

def total_items : ℕ := 35
def total_price : ℕ := 4000  -- in cents

def is_valid_purchase (x y z : ℕ) : Prop :=
  x + y + z = total_items ∧
  50 * x + 300 * y + 400 * z = total_price

theorem fifty_cent_items_count : ∃ (x y z : ℕ), is_valid_purchase x y z ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_fifty_cent_items_count_l1465_146545


namespace NUMINAMATH_CALUDE_prime_square_mod_30_l1465_146577

theorem prime_square_mod_30 (p : ℕ) (hp : Prime p) (h2 : p ≠ 2) (h3 : p ≠ 3) (h5 : p ≠ 5) :
  p ^ 2 % 30 = 1 ∨ p ^ 2 % 30 = 19 := by
sorry

end NUMINAMATH_CALUDE_prime_square_mod_30_l1465_146577


namespace NUMINAMATH_CALUDE_union_of_sets_l1465_146517

theorem union_of_sets : 
  let A : Set ℕ := {0, 1, 3}
  let B : Set ℕ := {1, 2, 4}
  A ∪ B = {0, 1, 2, 3, 4} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1465_146517


namespace NUMINAMATH_CALUDE_playground_count_l1465_146556

theorem playground_count (a b c d e x : ℕ) (h1 : a = 6) (h2 : b = 12) (h3 : c = 1) (h4 : d = 12) (h5 : e = 7)
  (h_mean : (a + b + c + d + e + x) / 6 = 7) : x = 4 := by
  sorry

end NUMINAMATH_CALUDE_playground_count_l1465_146556


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_two_thirds_l1465_146515

theorem negative_sixty_four_to_two_thirds : (-64 : ℝ) ^ (2/3) = 16 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_two_thirds_l1465_146515


namespace NUMINAMATH_CALUDE_cone_sphere_volume_equality_implies_lateral_area_l1465_146570

/-- Given a cone with base radius 1 and a sphere with radius 1, if their volumes are equal,
    then the lateral surface area of the cone is √17π. -/
theorem cone_sphere_volume_equality_implies_lateral_area (π : ℝ) (h : ℝ) :
  (1/3 : ℝ) * π * 1^2 * h = (4/3 : ℝ) * π * 1^3 →
  π * 1 * (1^2 + h^2).sqrt = π * Real.sqrt 17 := by sorry

end NUMINAMATH_CALUDE_cone_sphere_volume_equality_implies_lateral_area_l1465_146570


namespace NUMINAMATH_CALUDE_wall_length_theorem_l1465_146593

/-- Calculates the length of a wall built by a different number of workers in a different time, 
    given the original wall length and worker-days. -/
def calculate_wall_length (original_workers : ℕ) (original_days : ℕ) (original_length : ℕ) 
                          (new_workers : ℕ) (new_days : ℕ) : ℚ :=
  (original_workers * original_days * original_length : ℚ) / (new_workers * new_days)

theorem wall_length_theorem (original_workers : ℕ) (original_days : ℕ) (original_length : ℕ) 
                            (new_workers : ℕ) (new_days : ℕ) :
  original_workers = 18 →
  original_days = 42 →
  original_length = 140 →
  new_workers = 30 →
  new_days = 18 →
  calculate_wall_length original_workers original_days original_length new_workers new_days = 196 := by
  sorry

#eval calculate_wall_length 18 42 140 30 18

end NUMINAMATH_CALUDE_wall_length_theorem_l1465_146593


namespace NUMINAMATH_CALUDE_smallest_sum_m_p_l1465_146532

/-- The function f(x) = arcsin(log_m(px)) has a domain that is a closed interval of length 1/1007 -/
def domain_length (m p : ℕ) : ℚ := (m^2 - 1 : ℚ) / (m * p)

/-- The theorem statement -/
theorem smallest_sum_m_p :
  ∀ m p : ℕ,
  m > 1 ∧ 
  p > 0 ∧ 
  domain_length m p = 1 / 1007 →
  m + p ≥ 2031 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_m_p_l1465_146532


namespace NUMINAMATH_CALUDE_election_margin_of_victory_l1465_146569

theorem election_margin_of_victory 
  (total_votes : ℕ) 
  (winning_percentage : ℚ) 
  (winning_votes : ℕ) : 
  winning_percentage = 29/50 → 
  winning_votes = 1044 → 
  (winning_votes : ℚ) / winning_percentage = total_votes → 
  winning_votes - (total_votes - winning_votes) = 288 :=
by sorry

end NUMINAMATH_CALUDE_election_margin_of_victory_l1465_146569


namespace NUMINAMATH_CALUDE_sequence_problem_l1465_146558

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℝ := sorry

-- Define the sum of first n terms of a_n
def S (n : ℕ) : ℝ := sorry

-- Define the geometric sequence b_n
def b (n : ℕ) : ℝ := sorry

-- Define the sequence c_n
def c (n : ℕ) : ℝ := sorry

-- Define the sum of first n terms of c_n
def T (n : ℕ) : ℝ := sorry

theorem sequence_problem :
  (a 2 = 6) ∧
  (b 1 = 1) ∧
  (∀ n, b n > 0) ∧
  (b 2 + S 4 = 33) ∧
  (b 3 = S 2) ∧
  (∀ n, c n = 4 * b n - a 5) →
  ((∀ n, a n = 3 * n) ∧
   (∀ n, b n = 3^(n-1)) ∧
   (∀ n, n ≥ 4 → T n > S 6) ∧
   (∀ n, n < 4 → T n ≤ S 6)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_problem_l1465_146558


namespace NUMINAMATH_CALUDE_marbles_shared_proof_l1465_146530

/-- The number of marbles Jack starts with -/
def initial_marbles : ℕ := 62

/-- The number of marbles Jack ends with -/
def final_marbles : ℕ := 29

/-- The number of marbles Jack shared with Rebecca -/
def shared_marbles : ℕ := initial_marbles - final_marbles

theorem marbles_shared_proof : shared_marbles = 33 := by
  sorry

end NUMINAMATH_CALUDE_marbles_shared_proof_l1465_146530


namespace NUMINAMATH_CALUDE_find_m_value_l1465_146582

/-- Given functions f and g, prove that m = 4 when 3f(4) = g(4) -/
theorem find_m_value (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 - 3*x + m
  let g : ℝ → ℝ := λ x => x^2 - 3*x + 5*m
  3 * f 4 = g 4 → m = 4 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l1465_146582


namespace NUMINAMATH_CALUDE_find_N_l1465_146534

theorem find_N : ∃ N : ℚ, (5 + 6 + 7 + 8) / 4 = (2014 + 2015 + 2016 + 2017) / N ∧ N = 1240 := by
  sorry

end NUMINAMATH_CALUDE_find_N_l1465_146534


namespace NUMINAMATH_CALUDE_sum_of_squares_positive_l1465_146592

theorem sum_of_squares_positive (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ≠ x₂ → 
  x₁^2 + 2*x₁ - k = 0 → 
  x₂^2 + 2*x₂ - k = 0 → 
  x₁^2 + x₂^2 > 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_positive_l1465_146592


namespace NUMINAMATH_CALUDE_trig_function_equality_l1465_146523

/-- Given two functions f and g defined on real numbers, prove that g(x) equals f(π/4 + x) for all real x. -/
theorem trig_function_equality (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = Real.sin (2 * x + π / 3))
  (hg : ∀ x, g x = Real.cos (2 * x + π / 3)) :
  ∀ x, g x = f (π / 4 + x) := by
  sorry

end NUMINAMATH_CALUDE_trig_function_equality_l1465_146523


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1465_146539

theorem quadratic_rewrite (x : ℝ) :
  ∃ (b c : ℝ), x^2 + 1400*x + 1400 = (x + b)^2 + c ∧ c / b = -698 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1465_146539


namespace NUMINAMATH_CALUDE_complex_power_modulus_l1465_146500

theorem complex_power_modulus : Complex.abs ((2 + 2 * Complex.I) ^ 6 + 3) = 515 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_modulus_l1465_146500


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l1465_146552

theorem complex_number_quadrant (i : ℂ) (z : ℂ) :
  i * i = -1 →
  z = 1 - i →
  (0 < z.re ∧ z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l1465_146552


namespace NUMINAMATH_CALUDE_max_unglazed_windows_l1465_146560

/-- Represents a window or a pane of glass with a specific size. -/
structure Pane :=
  (size : ℕ)

/-- Represents the state of glazing process. -/
structure GlazingState :=
  (windows : List Pane)
  (glasses : List Pane)

/-- Simulates the glazier's process of matching glasses to windows. -/
def glazierProcess (state : GlazingState) : ℕ :=
  sorry

/-- Theorem stating the maximum number of unglazed windows. -/
theorem max_unglazed_windows :
  ∀ (initial_state : GlazingState),
    initial_state.windows.length = 15 ∧
    initial_state.glasses.length = 15 ∧
    (∀ w ∈ initial_state.windows, ∃ g ∈ initial_state.glasses, w.size = g.size) →
    glazierProcess initial_state ≤ 7 :=
  sorry

end NUMINAMATH_CALUDE_max_unglazed_windows_l1465_146560


namespace NUMINAMATH_CALUDE_find_larger_number_l1465_146598

theorem find_larger_number (L S : ℕ) (h1 : L - S = 1515) (h2 : L = 16 * S + 15) : L = 1615 := by
  sorry

end NUMINAMATH_CALUDE_find_larger_number_l1465_146598


namespace NUMINAMATH_CALUDE_valid_subsets_l1465_146536

def P : Set Nat := {1, 2, 3, 4}

def satisfies_conditions (A : Set Nat) : Prop :=
  A ⊆ P ∧
  (∀ x, x ∈ A → 2*x ∉ A) ∧
  (∀ x, x ∈ (P \ A) → 2*x ∉ (P \ A))

theorem valid_subsets :
  {A : Set Nat | satisfies_conditions A} =
  {{2}, {1,4}, {2,3}, {1,3,4}} := by sorry

end NUMINAMATH_CALUDE_valid_subsets_l1465_146536


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1465_146562

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 12

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x > 0} = solution_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1465_146562


namespace NUMINAMATH_CALUDE_apple_basket_solution_l1465_146591

def basket_problem (x : ℕ) : Prop :=
  let first_sale := x / 4 + 6
  let remaining_after_first := x - first_sale
  let second_sale := remaining_after_first / 3 + 4
  let remaining_after_second := remaining_after_first - second_sale
  let third_sale := remaining_after_second / 2 + 3
  let final_remaining := remaining_after_second - third_sale
  final_remaining = 4

theorem apple_basket_solution :
  ∃ x : ℕ, basket_problem x ∧ x = 28 :=
sorry

end NUMINAMATH_CALUDE_apple_basket_solution_l1465_146591


namespace NUMINAMATH_CALUDE_jack_gerald_notebook_difference_l1465_146549

theorem jack_gerald_notebook_difference :
  ∀ (jack_initial gerald : ℕ),
    jack_initial > gerald →
    gerald = 8 →
    jack_initial - 5 - 6 = 10 →
    jack_initial - gerald = 13 := by
  sorry

end NUMINAMATH_CALUDE_jack_gerald_notebook_difference_l1465_146549


namespace NUMINAMATH_CALUDE_second_candidate_votes_l1465_146588

theorem second_candidate_votes
  (total_votes : ℕ)
  (first_candidate_percentage : ℚ)
  (h1 : total_votes = 1200)
  (h2 : first_candidate_percentage = 80 / 100) :
  (1 - first_candidate_percentage) * total_votes = 240 :=
by sorry

end NUMINAMATH_CALUDE_second_candidate_votes_l1465_146588


namespace NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l1465_146573

def f (a x : ℝ) : ℝ := |x - a| - 2

theorem solution_set_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, |f a x| < 1 ↔ x ∈ Set.union (Set.Ioo (-2) 0) (Set.Ioo 2 4)) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_equals_one_l1465_146573


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1465_146585

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 5 * x + b

-- Define the solution set type
def SolutionSet := Set ℝ

-- State the theorem
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h1 : a = -5) 
  (h2 : b = 30) 
  (h3 : {x : ℝ | f a b x > 0} = {x : ℝ | -3 < x ∧ x < 2}) :
  {x : ℝ | f b (-a) x > 0} = {x : ℝ | x < -1/3 ∨ x > 1/2} :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1465_146585


namespace NUMINAMATH_CALUDE_leap_stride_difference_l1465_146544

/-- The number of strides Elmer takes between consecutive poles -/
def elmer_strides : ℕ := 56

/-- The number of leaps Oscar takes between consecutive poles -/
def oscar_leaps : ℕ := 16

/-- The total number of poles -/
def total_poles : ℕ := 51

/-- The distance in feet between the first and last pole -/
def total_distance : ℕ := 8000

/-- Elmer's stride length in feet -/
def elmer_stride_length : ℚ := total_distance / (elmer_strides * (total_poles - 1))

/-- Oscar's leap length in feet -/
def oscar_leap_length : ℚ := total_distance / (oscar_leaps * (total_poles - 1))

theorem leap_stride_difference : 
  ⌊oscar_leap_length - elmer_stride_length⌋ = 7 := by sorry

end NUMINAMATH_CALUDE_leap_stride_difference_l1465_146544


namespace NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_2_mod_17_l1465_146568

theorem smallest_five_digit_negative_congruent_to_2_mod_17 : 
  ∃ (n : ℤ), 
    n = -10011 ∧ 
    n ≡ 2 [ZMOD 17] ∧ 
    n < 0 ∧ 
    -99999 ≤ n ∧ 
    ∀ (m : ℤ), (m ≡ 2 [ZMOD 17] ∧ m < 0 ∧ -99999 ≤ m) → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_smallest_five_digit_negative_congruent_to_2_mod_17_l1465_146568


namespace NUMINAMATH_CALUDE_coefficient_of_z_in_equation3_l1465_146537

-- Define the system of equations
def equation1 (x y z : ℝ) : Prop := 6*x - 5*y + 3*z = 22
def equation2 (x y z : ℝ) : Prop := 4*x + 8*y - 11*z = 7
def equation3 (x y z : ℝ) : Prop := 5*x - 6*y + z = 12/2

-- Define the sum condition
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coefficient_of_z_in_equation3 (x y z : ℝ) 
  (eq1 : equation1 x y z) 
  (eq2 : equation2 x y z) 
  (eq3 : equation3 x y z) 
  (sum : sum_condition x y z) : 
  ∃ (a b c : ℝ), equation3 x y z ∧ c = 1 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_z_in_equation3_l1465_146537


namespace NUMINAMATH_CALUDE_blue_balls_to_balance_l1465_146509

/-- The number of blue balls that balance one green ball -/
def blue_per_green : ℚ := 2

/-- The number of blue balls that balance one yellow ball -/
def blue_per_yellow : ℚ := 8/3

/-- The number of blue balls that balance one white ball -/
def blue_per_white : ℚ := 5/3

/-- The number of green balls on the scale -/
def green_count : ℕ := 3

/-- The number of yellow balls on the scale -/
def yellow_count : ℕ := 3

/-- The number of white balls on the scale -/
def white_count : ℕ := 3

theorem blue_balls_to_balance : 
  green_count * blue_per_green + 
  yellow_count * blue_per_yellow + 
  white_count * blue_per_white = 19 := by sorry

end NUMINAMATH_CALUDE_blue_balls_to_balance_l1465_146509


namespace NUMINAMATH_CALUDE_complex_point_in_first_quadrant_l1465_146505

theorem complex_point_in_first_quadrant : 
  let z : ℂ := (1 - 2*I)^3 / I
  (z.re > 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_point_in_first_quadrant_l1465_146505


namespace NUMINAMATH_CALUDE_isabellas_haircut_l1465_146528

/-- Given an initial hair length and an amount cut off, 
    calculate the resulting hair length after a haircut. -/
def hair_length_after_cut (initial_length cut_length : ℕ) : ℕ :=
  initial_length - cut_length

/-- Theorem: Isabella's hair length after the haircut is 9 inches. -/
theorem isabellas_haircut : hair_length_after_cut 18 9 = 9 := by
  sorry

end NUMINAMATH_CALUDE_isabellas_haircut_l1465_146528


namespace NUMINAMATH_CALUDE_circle_C_equation_line_l_equation_l1465_146555

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 ≤ 2}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 4}

-- Define the point that line l passes through
def point_on_l : ℝ × ℝ := (2, 3)

-- Define the length of the chord intercepted by circle C on line l
def chord_length : ℝ := 2

-- Theorem stating the standard equation of circle C
theorem circle_C_equation :
  ∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - 1)^2 + (p.2 - 1)^2 = 2 :=
sorry

-- Theorem stating the equation of line l
theorem line_l_equation :
  ∀ p : ℝ × ℝ, (p ∈ circle_C ∧ (∃ q : ℝ × ℝ, q ∈ circle_C ∧ q ≠ p ∧
    ((p.1 - q.1)^2 + (p.2 - q.2)^2 = chord_length^2) ∧
    (p.1 - point_on_l.1) * (q.2 - point_on_l.2) = (q.1 - point_on_l.1) * (p.2 - point_on_l.2)))
  → (3 * p.1 - 4 * p.2 + 6 = 0 ∨ p.1 = 2) :=
sorry

end NUMINAMATH_CALUDE_circle_C_equation_line_l_equation_l1465_146555


namespace NUMINAMATH_CALUDE_inequality_proof_l1465_146548

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 / b) + (c^2 / d) ≥ ((a + c)^2) / (b + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1465_146548


namespace NUMINAMATH_CALUDE_calculation_proof_l1465_146572

theorem calculation_proof : (-3)^3 + 5^2 - (-2)^2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1465_146572


namespace NUMINAMATH_CALUDE_sqrt_eight_plus_sqrt_two_equals_three_sqrt_two_l1465_146524

theorem sqrt_eight_plus_sqrt_two_equals_three_sqrt_two : 
  Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_plus_sqrt_two_equals_three_sqrt_two_l1465_146524


namespace NUMINAMATH_CALUDE_matrix_power_result_l1465_146597

theorem matrix_power_result (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B.mulVec ![3, -1] = ![6, -2]) : 
  (B ^ 3).mulVec ![3, -1] = ![24, -8] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_result_l1465_146597


namespace NUMINAMATH_CALUDE_staircase_steps_l1465_146503

/-- The number of toothpicks in a staircase with n steps -/
def toothpicks (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 3

/-- The proposition that 8 steps result in 490 toothpicks -/
theorem staircase_steps : toothpicks 8 = 490 := by
  sorry

#check staircase_steps

end NUMINAMATH_CALUDE_staircase_steps_l1465_146503


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1465_146506

theorem solve_exponential_equation :
  ∃ x : ℤ, (2^x : ℝ) - (2^(x-2) : ℝ) = 3 * (2^10 : ℝ) ∧ x = 12 :=
by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1465_146506


namespace NUMINAMATH_CALUDE_positive_sum_l1465_146543

theorem positive_sum (x y z : ℝ) 
  (hx : -1 < x ∧ x < 1) 
  (hy : -1 < y ∧ y < 0) 
  (hz : 1 < z ∧ z < 2) : 
  y + z > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_l1465_146543


namespace NUMINAMATH_CALUDE_fourth_derivative_y_l1465_146553

noncomputable def y (x : ℝ) : ℝ := (3 * x - 7) * (3 : ℝ)^(-x)

theorem fourth_derivative_y (x : ℝ) :
  (deriv^[4] y) x = (7 * Real.log 3 - 12 - 3 * Real.log 3 * x) * (Real.log 3)^3 * (3 : ℝ)^(-x) :=
by sorry

end NUMINAMATH_CALUDE_fourth_derivative_y_l1465_146553


namespace NUMINAMATH_CALUDE_existence_of_prime_and_integer_l1465_146538

theorem existence_of_prime_and_integer (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ (q : ℕ) (n : ℕ+), q.Prime ∧ q < p ∧ p ∣ n.val^2 - q := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_and_integer_l1465_146538


namespace NUMINAMATH_CALUDE_complex_sum_magnitude_l1465_146535

theorem complex_sum_magnitude (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1)
  (h₂ : Complex.abs z₂ = 1)
  (h₃ : Complex.abs (z₁ - z₂) = 1) :
  Complex.abs (z₁ + z₂) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_magnitude_l1465_146535


namespace NUMINAMATH_CALUDE_circle_centers_distance_l1465_146554

theorem circle_centers_distance (r R : ℝ) (h : r > 0 ∧ R > 0) :
  let d := Real.sqrt (R^2 + r^2 + (10/3) * R * r)
  ∃ (ext_tangent int_tangent : ℝ),
    ext_tangent > 0 ∧ int_tangent > 0 ∧
    ext_tangent = 2 * int_tangent ∧
    d^2 = (R + r)^2 - int_tangent^2 ∧
    d^2 = (R - r)^2 + ext_tangent^2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_centers_distance_l1465_146554


namespace NUMINAMATH_CALUDE_angle_three_times_complement_l1465_146507

theorem angle_three_times_complement (x : ℝ) : 
  (x = 3 * (180 - x)) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_three_times_complement_l1465_146507


namespace NUMINAMATH_CALUDE_divisors_ending_in_2_mod_2010_l1465_146550

-- Define the number 2010
def n : ℕ := 2010

-- Define the function that counts divisors ending in 2
def count_divisors_ending_in_2 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisors_ending_in_2_mod_2010 : 
  count_divisors_ending_in_2 (n^n) % n = 503 := by sorry

end NUMINAMATH_CALUDE_divisors_ending_in_2_mod_2010_l1465_146550


namespace NUMINAMATH_CALUDE_farm_problem_l1465_146599

/-- Represents the number of animals on a farm -/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- Represents the conditions of the farm problem -/
def farm_conditions (f : FarmAnimals) : Prop :=
  f.pigs = 2 * f.cows ∧
  f.cows > f.goats ∧
  f.cows + f.pigs + f.goats = 56 ∧
  f.goats = 11

/-- Theorem stating that under the given conditions, the farmer has 4 more cows than goats -/
theorem farm_problem (f : FarmAnimals) (h : farm_conditions f) : f.cows - f.goats = 4 := by
  sorry

end NUMINAMATH_CALUDE_farm_problem_l1465_146599


namespace NUMINAMATH_CALUDE_chord_length_for_60_degree_line_and_circle_l1465_146584

/-- The length of the chord formed by the intersection of a line passing through the origin
    with a slope angle of 60° and the circle x² + y² - 4y = 0 is equal to 2√3. -/
theorem chord_length_for_60_degree_line_and_circle : 
  let line := {(x, y) : ℝ × ℝ | y = Real.sqrt 3 * x}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*y = 0}
  let chord := line ∩ circle
  ∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_for_60_degree_line_and_circle_l1465_146584


namespace NUMINAMATH_CALUDE_circle_center_l1465_146566

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y = 0

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem stating that (3, -4) is the center of the circle defined by CircleEquation -/
theorem circle_center : 
  ∃ (c : CircleCenter), c.x = 3 ∧ c.y = -4 ∧ 
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - c.x)^2 + (y - c.y)^2 = 25 :=
sorry

end NUMINAMATH_CALUDE_circle_center_l1465_146566


namespace NUMINAMATH_CALUDE_tammy_haircuts_needed_l1465_146508

/-- Represents the haircut system for Tammy -/
structure HaircutSystem where
  total_haircuts : ℕ
  free_haircuts : ℕ
  haircuts_until_next_free : ℕ

/-- Calculates the number of haircuts needed for the next free one -/
def haircuts_needed (system : HaircutSystem) : ℕ :=
  system.haircuts_until_next_free

/-- Theorem stating that Tammy needs 5 more haircuts for her next free one -/
theorem tammy_haircuts_needed (system : HaircutSystem) 
  (h1 : system.total_haircuts = 79)
  (h2 : system.free_haircuts = 5)
  (h3 : system.haircuts_until_next_free = 5) :
  haircuts_needed system = 5 := by
  sorry

#eval haircuts_needed { total_haircuts := 79, free_haircuts := 5, haircuts_until_next_free := 5 }

end NUMINAMATH_CALUDE_tammy_haircuts_needed_l1465_146508


namespace NUMINAMATH_CALUDE_sum_of_coefficients_eq_value_at_one_l1465_146579

/-- The polynomial for which we want to find the sum of coefficients -/
def p (x : ℝ) : ℝ := 3*(x^8 - x^5 + 2*x^3 - 6) - 5*(x^4 + 3*x^2) + 2*(x^6 - 5)

/-- The sum of coefficients of a polynomial is equal to its value at x = 1 -/
theorem sum_of_coefficients_eq_value_at_one :
  p 1 = -40 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_eq_value_at_one_l1465_146579


namespace NUMINAMATH_CALUDE_mixed_fraction_product_l1465_146541

theorem mixed_fraction_product (X Y : ℕ) : 
  (X > 0 ∧ Y > 0) →
  (5 : ℝ) < 5 + 1 / X ∧ 5 + 1 / X ≤ (11 : ℝ) / 2 →
  (86 : ℝ) / 11 < Y ∧ Y < 9 →
  (5 + 1 / X) * (Y + 1 / 2) = 43 →
  X = 17 ∧ Y = 8 :=
by sorry

end NUMINAMATH_CALUDE_mixed_fraction_product_l1465_146541


namespace NUMINAMATH_CALUDE_middle_digit_is_zero_l1465_146542

/-- Represents a three-digit number in base 8 -/
structure Base8Number where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds < 8 ∧ tens < 8 ∧ ones < 8

/-- Converts a Base8Number to its decimal (base 10) representation -/
def toDecimal (n : Base8Number) : Nat :=
  64 * n.hundreds + 8 * n.tens + n.ones

/-- Represents a three-digit number in base 10 -/
structure Base10Number where
  hundreds : Nat
  tens : Nat
  ones : Nat
  valid : hundreds < 10 ∧ tens < 10 ∧ ones < 10

/-- Checks if a Base8Number has its digits reversed in base 10 representation -/
def hasReversedDigits (n : Base8Number) : Prop :=
  ∃ (m : Base10Number), 
    toDecimal n = 100 * m.hundreds + 10 * m.tens + m.ones ∧
    n.hundreds = m.ones ∧
    n.tens = m.tens ∧
    n.ones = m.hundreds

theorem middle_digit_is_zero (n : Base8Number) 
  (h : hasReversedDigits n) : n.tens = 0 := by
  sorry

end NUMINAMATH_CALUDE_middle_digit_is_zero_l1465_146542


namespace NUMINAMATH_CALUDE_central_angle_invariant_under_doubling_l1465_146546

theorem central_angle_invariant_under_doubling 
  (r : ℝ) (l : ℝ) (h_r : r > 0) (h_l : l > 0) :
  l / r = (2 * l) / (2 * r) :=
by sorry

end NUMINAMATH_CALUDE_central_angle_invariant_under_doubling_l1465_146546


namespace NUMINAMATH_CALUDE_cd_purchase_cost_l1465_146587

/-- Calculates the total cost of purchasing CDs -/
def total_cost (life_journey_price : ℕ) (day_life_price : ℕ) (rescind_price : ℕ) (quantity : ℕ) : ℕ :=
  quantity * (life_journey_price + day_life_price + rescind_price)

/-- Theorem: The total cost of buying 3 CDs each of The Life Journey ($100), 
    A Day a Life ($50), and When You Rescind ($85) is $705 -/
theorem cd_purchase_cost : total_cost 100 50 85 3 = 705 := by
  sorry

end NUMINAMATH_CALUDE_cd_purchase_cost_l1465_146587


namespace NUMINAMATH_CALUDE_problem_solving_probability_l1465_146502

theorem problem_solving_probability 
  (prob_A prob_B : ℚ) 
  (h_A : prob_A = 2/3) 
  (h_B : prob_B = 3/4) : 
  1 - (1 - prob_A) * (1 - prob_B) = 11/12 := by
sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l1465_146502


namespace NUMINAMATH_CALUDE_terminal_side_of_half_angle_l1465_146580

theorem terminal_side_of_half_angle (θ : Real) 
  (h1 : |Real.cos θ| = Real.cos θ) 
  (h2 : |Real.tan θ| = -Real.tan θ) : 
  (∃ (k : Int), 
    (k * Real.pi + Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ k * Real.pi + Real.pi) ∨
    (k * Real.pi + 3 * Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ k * Real.pi + 2 * Real.pi) ∨
    (∃ (n : Int), θ / 2 = n * Real.pi)) :=
sorry

end NUMINAMATH_CALUDE_terminal_side_of_half_angle_l1465_146580


namespace NUMINAMATH_CALUDE_hawks_total_points_l1465_146565

theorem hawks_total_points (touchdowns : ℕ) (points_per_touchdown : ℕ) 
  (h1 : touchdowns = 3) 
  (h2 : points_per_touchdown = 7) : 
  touchdowns * points_per_touchdown = 21 := by
  sorry

end NUMINAMATH_CALUDE_hawks_total_points_l1465_146565


namespace NUMINAMATH_CALUDE_wall_painting_fraction_l1465_146527

theorem wall_painting_fraction :
  let total_wall : ℚ := 1
  let matilda_half : ℚ := 1/2
  let ellie_half : ℚ := 1/2
  let matilda_painted : ℚ := matilda_half * (1/2)
  let ellie_painted : ℚ := ellie_half * (1/3)
  matilda_painted + ellie_painted = 5/12 := by
  sorry

end NUMINAMATH_CALUDE_wall_painting_fraction_l1465_146527


namespace NUMINAMATH_CALUDE_square_area_perimeter_relationship_l1465_146519

/-- The relationship between the area and perimeter of a square is quadratic -/
theorem square_area_perimeter_relationship (x y : ℝ) (h_pos : x > 0) :
  ∃ k : ℝ, y = k * x^2 ↔ 
  (∃ a : ℝ, a > 0 ∧ x = 4 * a ∧ y = a^2) :=
by sorry

end NUMINAMATH_CALUDE_square_area_perimeter_relationship_l1465_146519


namespace NUMINAMATH_CALUDE_susan_babysitting_earnings_l1465_146594

def susan_earnings (initial : ℝ) : Prop :=
  let after_clothes := initial / 2
  let after_books := after_clothes / 2
  after_books = 150

theorem susan_babysitting_earnings :
  ∃ (initial : ℝ), susan_earnings initial ∧ initial = 600 :=
sorry

end NUMINAMATH_CALUDE_susan_babysitting_earnings_l1465_146594


namespace NUMINAMATH_CALUDE_exists_non_increasing_function_exists_larger_increasing_interval_inverse_function_not_decreasing_monotonic_function_extrema_at_endpoints_l1465_146510

-- Statement 1
theorem exists_non_increasing_function : 
  ∃ f : ℝ → ℝ, f (-1) < f 3 ∧ ¬(∀ x y : ℝ, x < y → f x < f y) := by sorry

-- Statement 2
theorem exists_larger_increasing_interval : 
  ∃ f : ℝ → ℝ, (∀ x y : ℝ, x ∈ Set.Ici 1 → y ∈ Set.Ici 1 → x < y → f x < f y) ∧ 
  ∃ a b : ℝ, a < 1 ∧ (∀ x y : ℝ, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x < f y) := by sorry

-- Statement 3
theorem inverse_function_not_decreasing :
  ¬(∀ x y : ℝ, x ∈ (Set.Iio 0 ∪ Set.Ioi 0) → y ∈ (Set.Iio 0 ∪ Set.Ioi 0) → x < y → (1 : ℝ) / x > (1 : ℝ) / y) := by sorry

-- Statement 4
theorem monotonic_function_extrema_at_endpoints {a b : ℝ} (f : ℝ → ℝ) (h : Monotone f) :
  ∀ x ∈ Set.Icc a b, f a ≤ f x ∧ f x ≤ f b := by sorry

end NUMINAMATH_CALUDE_exists_non_increasing_function_exists_larger_increasing_interval_inverse_function_not_decreasing_monotonic_function_extrema_at_endpoints_l1465_146510


namespace NUMINAMATH_CALUDE_alice_wins_iff_zero_l1465_146586

/-- Alice's winning condition in the quadratic equation game -/
theorem alice_wins_iff_zero (a b c : ℝ) : 
  (∀ d : ℝ, ¬(∃ x y : ℝ, x ≠ y ∧ 
    ((a + d) * x^2 + (b + d) * x + (c + d) = 0) ∧ 
    ((a + d) * y^2 + (b + d) * y + (c + d) = 0)))
  ↔ 
  (a = 0 ∧ b = 0 ∧ c = 0) :=
sorry

end NUMINAMATH_CALUDE_alice_wins_iff_zero_l1465_146586


namespace NUMINAMATH_CALUDE_right_triangle_one_one_sqrt_two_l1465_146533

theorem right_triangle_one_one_sqrt_two :
  let a : ℝ := 1
  let b : ℝ := 1
  let c : ℝ := Real.sqrt 2
  a^2 + b^2 = c^2 := by sorry

end NUMINAMATH_CALUDE_right_triangle_one_one_sqrt_two_l1465_146533


namespace NUMINAMATH_CALUDE_meetings_percentage_of_work_day_l1465_146529

/-- Represents the duration of a work day in minutes -/
def work_day_duration : ℕ := 10 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_duration : ℕ := 80

/-- Represents the duration of the break between meetings in minutes -/
def break_duration : ℕ := 15

/-- Calculates the total time spent in meetings and on break -/
def total_meeting_time : ℕ :=
  first_meeting_duration + (3 * first_meeting_duration) + break_duration

/-- Theorem stating that the percentage of work day spent in meetings and on break is 56% -/
theorem meetings_percentage_of_work_day :
  (total_meeting_time : ℚ) / work_day_duration * 100 = 56 := by
  sorry

end NUMINAMATH_CALUDE_meetings_percentage_of_work_day_l1465_146529


namespace NUMINAMATH_CALUDE_qin_jiushao_area_formula_l1465_146511

theorem qin_jiushao_area_formula (a b c : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = Real.sqrt 3) (h3 : c = 2) :
  let S := Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))
  S = Real.sqrt 23 / 4 := by
sorry

end NUMINAMATH_CALUDE_qin_jiushao_area_formula_l1465_146511


namespace NUMINAMATH_CALUDE_lesser_number_problem_l1465_146516

theorem lesser_number_problem (x y : ℝ) (h_sum : x + y = 70) (h_product : x * y = 1050) : 
  min x y = 30 := by
sorry

end NUMINAMATH_CALUDE_lesser_number_problem_l1465_146516


namespace NUMINAMATH_CALUDE_division_problem_l1465_146525

theorem division_problem (x : ℝ) (h1 : 10 * x = 50) : 20 / x = 4 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1465_146525


namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l1465_146590

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y + 4 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y - 2 = 0

-- Theorem for parallel lines
theorem parallel_lines (m : ℝ) :
  (∀ x y : ℝ, l₁ m x y ↔ l₂ m x y) → m = 1 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines (m : ℝ) :
  (∀ x y : ℝ, l₁ m x y → l₂ m x y → x * x + y * y = 0) → m = -2/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l1465_146590


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l1465_146501

/-- The trajectory of the midpoint of a line segment connecting a point on a unit circle and a fixed point -/
theorem midpoint_trajectory (x y x₀ y₀ : ℝ) : 
  (x₀^2 + y₀^2 = 1) →  -- P is on the unit circle
  (x = (x₀ + 3) / 2) →  -- x-coordinate of midpoint M
  (y = y₀ / 2) →  -- y-coordinate of midpoint M
  ((2*x - 3)^2 + 4*y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l1465_146501


namespace NUMINAMATH_CALUDE_factorial_ratio_evaluation_l1465_146583

theorem factorial_ratio_evaluation : (Nat.factorial 10 * Nat.factorial 4 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 5 / 21 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_evaluation_l1465_146583


namespace NUMINAMATH_CALUDE_monkey_banana_problem_l1465_146563

/-- The number of monkeys in the initial scenario -/
def initial_monkeys : ℕ := 8

/-- The time taken to eat bananas in minutes -/
def eating_time : ℕ := 8

/-- The number of bananas eaten in the initial scenario -/
def initial_bananas : ℕ := 8

/-- The number of monkeys in the second scenario -/
def second_monkeys : ℕ := 3

/-- The number of bananas eaten in the second scenario -/
def second_bananas : ℕ := 3

theorem monkey_banana_problem :
  (initial_monkeys * eating_time = initial_bananas * eating_time) ∧
  (second_monkeys * eating_time = second_bananas * eating_time) →
  initial_monkeys = initial_bananas :=
by sorry

end NUMINAMATH_CALUDE_monkey_banana_problem_l1465_146563


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l1465_146547

theorem trigonometric_expression_equality : 
  (Real.tan (150 * π / 180)) * (Real.cos (-210 * π / 180)) * (Real.sin (-420 * π / 180)) / 
  ((Real.sin (1050 * π / 180)) * (Real.cos (-600 * π / 180))) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l1465_146547
