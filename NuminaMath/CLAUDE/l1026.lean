import Mathlib

namespace NUMINAMATH_CALUDE_expression_equality_l1026_102645

/-- Proof that the given expression K is equal to 80xyz(x^2 + y^2 + z^2) -/
theorem expression_equality (x y z : ℝ) :
  (x + y + z)^5 - (-x + y + z)^5 - (x - y + z)^5 - (x + y - z)^5 = 80 * x * y * z * (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1026_102645


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l1026_102673

/-- The maximum number of intersection points in the first quadrant
    given 15 points on the x-axis and 10 points on the y-axis -/
def max_intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersection points -/
theorem intersection_points_theorem :
  max_intersection_points 15 10 = 4725 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l1026_102673


namespace NUMINAMATH_CALUDE_max_z_value_l1026_102619

theorem max_z_value (x y z : ℝ) (sum_eq : x + y + z = 9) (prod_eq : x*y + y*z + z*x = 24) :
  z ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_max_z_value_l1026_102619


namespace NUMINAMATH_CALUDE_square_last_digits_l1026_102644

theorem square_last_digits :
  (∃ n : ℕ, n^2 ≡ 444 [ZMOD 1000]) ∧
  (∀ k : ℤ, (1000*k + 38)^2 ≡ 444 [ZMOD 1000]) ∧
  (¬ ∃ n : ℤ, n^2 ≡ 4444 [ZMOD 10000]) := by
  sorry

end NUMINAMATH_CALUDE_square_last_digits_l1026_102644


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l1026_102655

def is_transformation (f g : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x : ℝ, f x = g (a * x + b) + c

theorem f_sum_symmetric (f : ℝ → ℝ) (hf : Differentiable ℝ f)
  (h1 : ∀ x : ℝ, f x + f (1 - x) = 10)
  (h2 : ∀ x : ℝ, f (1 + x) = 3 + f x)
  (h3 : ∃ g : ℝ → ℝ, is_transformation f g) :
  ∀ x : ℝ, f x + f (-x) = 7 :=
sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l1026_102655


namespace NUMINAMATH_CALUDE_base3_sum_theorem_l1026_102681

/-- Converts a base 3 number represented as a list of digits to its decimal equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 3 * acc + d) 0

/-- Converts a decimal number to its base 3 representation as a list of digits -/
def decimalToBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 3) ((m % 3) :: acc)
    aux n []

/-- The main theorem stating the sum of the given base 3 numbers -/
theorem base3_sum_theorem :
  let a := base3ToDecimal [2, 1, 2, 1]
  let b := base3ToDecimal [1, 2, 1, 2]
  let c := base3ToDecimal [2, 1, 2]
  let d := base3ToDecimal [2]
  decimalToBase3 (a + b + c + d) = [2, 2, 0, 1] := by sorry

end NUMINAMATH_CALUDE_base3_sum_theorem_l1026_102681


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1026_102689

theorem restaurant_bill_calculation (num_adults num_teenagers num_children : ℕ)
  (adult_meal_cost teenager_meal_cost child_meal_cost : ℚ)
  (soda_cost dessert_cost appetizer_cost : ℚ)
  (num_desserts num_appetizers : ℕ)
  (h1 : num_adults = 6)
  (h2 : num_teenagers = 3)
  (h3 : num_children = 1)
  (h4 : adult_meal_cost = 9)
  (h5 : teenager_meal_cost = 7)
  (h6 : child_meal_cost = 5)
  (h7 : soda_cost = 2.5)
  (h8 : dessert_cost = 4)
  (h9 : appetizer_cost = 6)
  (h10 : num_desserts = 3)
  (h11 : num_appetizers = 2) :
  (num_adults * adult_meal_cost +
   num_teenagers * teenager_meal_cost +
   num_children * child_meal_cost +
   (num_adults + num_teenagers + num_children) * soda_cost +
   num_desserts * dessert_cost +
   num_appetizers * appetizer_cost) = 129 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1026_102689


namespace NUMINAMATH_CALUDE_derivative_periodicity_l1026_102649

theorem derivative_periodicity (f : ℝ → ℝ) (T : ℝ) (h_diff : Differentiable ℝ f) (h_periodic : ∀ x, f (x + T) = f x) (h_pos : T > 0) :
  ∀ x, deriv f (x + T) = deriv f x :=
by sorry

end NUMINAMATH_CALUDE_derivative_periodicity_l1026_102649


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l1026_102615

theorem simplify_complex_expression (a : ℝ) (h : a > 0) :
  Real.sqrt ((2 * a) / ((1 + a) * (1 + a) ^ (1/3))) *
  ((4 + 8 / a + 4 / a^2) / Real.sqrt 2) ^ (1/3) =
  (2 * a^(5/6)) / a := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l1026_102615


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1026_102696

theorem complex_number_in_second_quadrant :
  let z : ℂ := (2 * Complex.I) / (2 - Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l1026_102696


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1026_102638

theorem max_value_of_expression (x : ℝ) : 
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ 15 ∧ 
  ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) = 15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1026_102638


namespace NUMINAMATH_CALUDE_two_distinct_roots_range_l1026_102676

theorem two_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - (m+2)*x - m + 1 = 0 ∧ y^2 - (m+2)*y - m + 1 = 0) ↔
  m < -8 ∨ m > 0 := by
sorry

end NUMINAMATH_CALUDE_two_distinct_roots_range_l1026_102676


namespace NUMINAMATH_CALUDE_richard_needs_three_touchdowns_per_game_l1026_102666

/-- Represents a football player's touchdown record --/
structure TouchdownRecord where
  player : String
  touchdowns : ℕ
  games : ℕ

/-- Calculates the number of touchdowns needed to beat a record --/
def touchdownsNeededToBeat (record : TouchdownRecord) : ℕ :=
  record.touchdowns + 1

/-- Theorem: Richard needs to average 3 touchdowns per game in the final two games to beat Archie's record --/
theorem richard_needs_three_touchdowns_per_game
  (archie : TouchdownRecord)
  (richard_current_touchdowns : ℕ)
  (richard_current_games : ℕ)
  (total_games : ℕ)
  (h1 : archie.player = "Archie")
  (h2 : archie.touchdowns = 89)
  (h3 : archie.games = 16)
  (h4 : richard_current_touchdowns = 6 * richard_current_games)
  (h5 : richard_current_games = 14)
  (h6 : total_games = 16) :
  (touchdownsNeededToBeat archie - richard_current_touchdowns) / (total_games - richard_current_games) = 3 :=
sorry

end NUMINAMATH_CALUDE_richard_needs_three_touchdowns_per_game_l1026_102666


namespace NUMINAMATH_CALUDE_sufficiency_not_necessity_l1026_102637

theorem sufficiency_not_necessity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b = 2 → a * b ≤ 1) ∧
  ∃ (c d : ℝ), 0 < c ∧ 0 < d ∧ c * d ≤ 1 ∧ c + d ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_sufficiency_not_necessity_l1026_102637


namespace NUMINAMATH_CALUDE_enclosed_area_is_four_l1026_102685

-- Define the functions for the curve and the line
def f (x : ℝ) := 3 * x^2
def g (x : ℝ) := 3

-- Define the intersection points
def x₁ : ℝ := -1
def x₂ : ℝ := 1

-- State the theorem
theorem enclosed_area_is_four :
  (∫ (x : ℝ) in x₁..x₂, g x - f x) = 4 := by
  sorry

end NUMINAMATH_CALUDE_enclosed_area_is_four_l1026_102685


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l1026_102668

def total_players : ℕ := 15
def lineup_size : ℕ := 6
def pre_selected_players : ℕ := 2

theorem starting_lineup_combinations :
  Nat.choose (total_players - pre_selected_players) (lineup_size - pre_selected_players) = 715 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l1026_102668


namespace NUMINAMATH_CALUDE_circle_equation_l1026_102675

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Function to check if a point is on a circle -/
def isOnCircle (p : Point2D) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Function to check if two points are symmetric with respect to y = x -/
def isSymmetricYEqX (p1 p2 : Point2D) : Prop :=
  p1.x = p2.y ∧ p1.y = p2.x

/-- Theorem: Given a circle C with radius 1 and center symmetric to (1, 0) 
    with respect to the line y = x, its standard equation is x^2 + (y - 1)^2 = 1 -/
theorem circle_equation (C : Circle) 
    (h1 : C.radius = 1)
    (h2 : isSymmetricYEqX C.center ⟨1, 0⟩) : 
    ∀ (p : Point2D), isOnCircle p C ↔ p.x^2 + (p.y - 1)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l1026_102675


namespace NUMINAMATH_CALUDE_stones_can_be_combined_l1026_102683

/-- Definition of similar sizes -/
def similar_sizes (x y : ℕ) : Prop := x ≤ y ∧ y ≤ 2 * x

/-- A step in the combining process -/
inductive combine_step (stones : List ℕ) : List ℕ → Prop
  | combine (x y : ℕ) (rest : List ℕ) :
      x ∈ stones →
      y ∈ stones →
      similar_sizes x y →
      combine_step stones ((x + y) :: (stones.filter (λ z ↦ z ≠ x ∧ z ≠ y)))

/-- The transitive closure of combine_step -/
def can_combine := Relation.ReflTransGen combine_step

/-- The main theorem -/
theorem stones_can_be_combined (initial_stones : List ℕ) :
  ∃ (final_pile : ℕ), can_combine initial_stones [final_pile] :=
sorry

end NUMINAMATH_CALUDE_stones_can_be_combined_l1026_102683


namespace NUMINAMATH_CALUDE_specific_solid_volume_l1026_102623

/-- A solid with a square base and specific edge lengths -/
structure Solid where
  s : ℝ
  base_side_length : s > 0
  upper_edge_length : ℝ
  upper_edge_parallel : upper_edge_length = 3 * s
  other_edges_length : ℝ
  other_edges_equal_s : other_edges_length = s

/-- The volume of the solid -/
noncomputable def volume (solid : Solid) : ℝ := sorry

/-- Theorem stating the volume of the specific solid -/
theorem specific_solid_volume :
  ∀ (solid : Solid),
    solid.s = 4 * Real.sqrt 2 →
    volume solid = 144 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_specific_solid_volume_l1026_102623


namespace NUMINAMATH_CALUDE_triangle_areas_l1026_102620

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the point O (intersection of altitudes)
def O : ℝ × ℝ := sorry

-- Define the points P, Q, R on the sides of the triangle
def P : ℝ × ℝ := sorry
def Q : ℝ × ℝ := sorry
def R : ℝ × ℝ := sorry

-- Define the given conditions
axiom parallel_RP_AC : sorry
axiom AC_length : sorry
axiom sin_ABC : sorry

-- Define the areas of triangles ABC and ROC
noncomputable def area_ABC (t : Triangle) : ℝ := sorry
noncomputable def area_ROC (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_areas (t : Triangle) :
  (area_ABC t = 16/3 ∧ area_ROC t = 21/25) ∨
  (area_ABC t = 3 ∧ area_ROC t = 112/75) :=
sorry

end NUMINAMATH_CALUDE_triangle_areas_l1026_102620


namespace NUMINAMATH_CALUDE_star_difference_l1026_102672

def star (x y : ℝ) : ℝ := 2*x*y - 3*x + y

theorem star_difference : (star 6 4) - (star 4 6) = -8 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l1026_102672


namespace NUMINAMATH_CALUDE_find_r_value_l1026_102651

theorem find_r_value (x y k : ℝ) (h : y^2 + 4*y + 4 + Real.sqrt (x + y + k) = 0) :
  let r := |x * y|
  r = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_r_value_l1026_102651


namespace NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l1026_102624

/-- Two lines in a plane -/
structure TwoLines where
  a : ℝ
  l₁ : ℝ → ℝ → Prop := λ x y => (a - 1) * x + 2 * y + 1 = 0
  l₂ : ℝ → ℝ → Prop := λ x y => x + a * y + 3 = 0

/-- Parallel lines theorem -/
theorem parallel_lines (lines : TwoLines) :
  (∀ x y, lines.l₁ x y ↔ ∃ k, lines.l₂ (x + k) (y + k)) →
  lines.a = 2 ∨ lines.a = -1 := by sorry

/-- Perpendicular lines theorem -/
theorem perpendicular_lines (lines : TwoLines) :
  (∀ x₁ y₁ x₂ y₂, lines.l₁ x₁ y₁ → lines.l₂ x₂ y₂ → 
    ((x₂ - x₁) * (lines.a - 1) + (y₂ - y₁) * 2 = 0) ∧
    ((x₂ - x₁) * 1 + (y₂ - y₁) * lines.a = 0)) →
  (lines.a - 1) + 2 * lines.a = 0 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_perpendicular_lines_l1026_102624


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l1026_102648

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (s : ℝ) (h : s = 8) :
  let cube_volume := s^3
  let tetrahedron_volume := cube_volume / 3
  tetrahedron_volume = 512 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l1026_102648


namespace NUMINAMATH_CALUDE_all_squares_similar_l1026_102662

-- Define a square
structure Square where
  side : ℝ
  side_positive : side > 0

-- Define similarity for squares
def similar (s1 s2 : Square) : Prop :=
  ∃ k : ℝ, k > 0 ∧ s1.side = k * s2.side

-- Theorem: Any two squares are similar
theorem all_squares_similar (s1 s2 : Square) : similar s1 s2 := by
  sorry


end NUMINAMATH_CALUDE_all_squares_similar_l1026_102662


namespace NUMINAMATH_CALUDE_max_ab_for_tangent_circle_l1026_102630

/-- Given a line l: x + 2y = 0 tangent to a circle C: (x-a)² + (y-b)² = 5,
    where the center (a,b) of C is above l, the maximum value of ab is 25/8 -/
theorem max_ab_for_tangent_circle (a b : ℝ) : 
  (∀ x y : ℝ, (x + 2*y = 0) → ((x - a)^2 + (y - b)^2 = 5)) →  -- tangency condition
  (a + 2*b > 0) →  -- center above the line
  (∀ a' b' : ℝ, (∀ x y : ℝ, (x + 2*y = 0) → ((x - a')^2 + (y - b')^2 = 5)) → 
                (a' + 2*b' > 0) → 
                a * b ≤ a' * b') →
  a * b = 25/8 := by sorry

end NUMINAMATH_CALUDE_max_ab_for_tangent_circle_l1026_102630


namespace NUMINAMATH_CALUDE_num_arrangements_eq_360_l1026_102643

/-- The number of volunteers --/
def num_volunteers : ℕ := 6

/-- The number of people to be selected --/
def num_selected : ℕ := 4

/-- The number of distinct tasks --/
def num_tasks : ℕ := 4

/-- Theorem stating the number of arrangements --/
theorem num_arrangements_eq_360 : 
  (num_volunteers.factorial) / ((num_volunteers - num_selected).factorial) = 360 :=
sorry

end NUMINAMATH_CALUDE_num_arrangements_eq_360_l1026_102643


namespace NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_one_l1026_102658

theorem negation_of_forall_x_squared_gt_one :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_x_squared_gt_one_l1026_102658


namespace NUMINAMATH_CALUDE_bubble_gum_count_l1026_102659

theorem bubble_gum_count (total_cost : ℕ) (cost_per_piece : ℕ) (h1 : total_cost = 2448) (h2 : cost_per_piece = 18) :
  total_cost / cost_per_piece = 136 := by
  sorry

end NUMINAMATH_CALUDE_bubble_gum_count_l1026_102659


namespace NUMINAMATH_CALUDE_largest_n_for_exponential_inequality_l1026_102687

theorem largest_n_for_exponential_inequality :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (x : ℝ), Real.exp (n * x) + Real.exp (-n * x) ≥ n) ∧
  (∀ (m : ℕ), m > n → ∃ (y : ℝ), Real.exp (m * y) + Real.exp (-m * y) < m) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_exponential_inequality_l1026_102687


namespace NUMINAMATH_CALUDE_six_planes_max_parts_l1026_102678

/-- The maximum number of parts that n planes can divide space into -/
def max_parts (n : ℕ) : ℕ := (n^3 + 5*n + 6) / 6

/-- Theorem: 6 planes can divide space into at most 42 parts -/
theorem six_planes_max_parts : max_parts 6 = 42 := by
  sorry

end NUMINAMATH_CALUDE_six_planes_max_parts_l1026_102678


namespace NUMINAMATH_CALUDE_other_diagonal_length_l1026_102691

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  d1 : ℝ  -- Length of the first diagonal
  d2 : ℝ  -- Length of the second diagonal
  area : ℝ -- Area of the rhombus

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.d1 * r.d2) / 2

theorem other_diagonal_length :
  ∀ r : Rhombus, r.d1 = 160 ∧ r.area = 5600 → r.d2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l1026_102691


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1026_102684

/-- Given a hyperbola and a circle with specific properties, prove the eccentricity of the hyperbola -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (b / a) * x}
  let chord_length := Real.sqrt 3
  (∃ (p q : ℝ × ℝ), p ∈ asymptote ∧ q ∈ asymptote ∧ p ∈ circle ∧ q ∈ circle ∧
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = chord_length) →
  let c := Real.sqrt (a^2 + b^2)
  c / a = 2 / 3 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1026_102684


namespace NUMINAMATH_CALUDE_election_winner_votes_l1026_102627

theorem election_winner_votes (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : (62 : ℚ) / 100 * total_votes - (38 : ℚ) / 100 * total_votes = 288) :
  (62 : ℚ) / 100 * total_votes = 744 := by
sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1026_102627


namespace NUMINAMATH_CALUDE_solution_difference_l1026_102694

theorem solution_difference (x y : ℝ) : 
  (Int.floor x + (y - Int.floor y) = 3.7) →
  ((x - Int.floor x) + Int.floor y = 4.2) →
  |x - y| = 1.5 := by sorry

end NUMINAMATH_CALUDE_solution_difference_l1026_102694


namespace NUMINAMATH_CALUDE_quadratic_divisibility_l1026_102652

theorem quadratic_divisibility (p : ℕ) (a b c : ℕ) (h_prime : Nat.Prime p) 
  (h_a : 0 < a ∧ a ≤ p) (h_b : 0 < b ∧ b ≤ p) (h_c : 0 < c ∧ c ≤ p)
  (h_div : ∀ (x : ℕ), x > 0 → (p ∣ (a * x^2 + b * x + c))) :
  a + b + c = 3 * p := by
sorry

end NUMINAMATH_CALUDE_quadratic_divisibility_l1026_102652


namespace NUMINAMATH_CALUDE_min_absolute_T_l1026_102664

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) - a n = d

def T (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (a n) + (a (n+1)) + (a (n+2)) + (a (n+3)) + (a (n+4)) + (a (n+5))

theorem min_absolute_T (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 5 = 15 →
  a 10 = -10 →
  (∃ n : ℕ, ∀ m : ℕ, |T a n| ≤ |T a m|) →
  (∃ n : ℕ, n = 5 ∨ n = 6 ∧ ∀ m : ℕ, |T a n| ≤ |T a m|) :=
by sorry

end NUMINAMATH_CALUDE_min_absolute_T_l1026_102664


namespace NUMINAMATH_CALUDE_min_value_trig_fraction_l1026_102690

theorem min_value_trig_fraction (x : ℝ) : 
  (Real.sin x)^8 + (Real.cos x)^8 + 3 ≥ (2/3) * ((Real.sin x)^6 + (Real.cos x)^6 + 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_fraction_l1026_102690


namespace NUMINAMATH_CALUDE_joes_lift_l1026_102608

theorem joes_lift (first_lift second_lift : ℕ) 
  (h1 : first_lift + second_lift = 1800)
  (h2 : 2 * first_lift = second_lift + 300) : 
  first_lift = 700 := by
sorry

end NUMINAMATH_CALUDE_joes_lift_l1026_102608


namespace NUMINAMATH_CALUDE_product_of_two_numbers_l1026_102612

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 340) : 
  x * y = 97.9450625 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_l1026_102612


namespace NUMINAMATH_CALUDE_cost_of_one_milk_carton_l1026_102633

/-- The cost of 1 one-litre carton of milk, given that 4 cartons cost $4.88 -/
theorem cost_of_one_milk_carton :
  let total_cost : ℚ := 488/100  -- $4.88 represented as a rational number
  let num_cartons : ℕ := 4
  let cost_per_carton : ℚ := total_cost / num_cartons
  cost_per_carton = 122/100  -- $1.22 represented as a rational number
:= by sorry

end NUMINAMATH_CALUDE_cost_of_one_milk_carton_l1026_102633


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1026_102661

theorem quadratic_one_solution (m : ℝ) : 
  (∃! x, 3 * x^2 + m * x + 9 = 0) ↔ m = 6 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1026_102661


namespace NUMINAMATH_CALUDE_focus_of_ellipse_l1026_102693

/-- The equation of the ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 / 4 + y^2 / 5 = 1

/-- Definition of a focus of an ellipse -/
def is_focus (a b c : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = c^2 ∧ a^2 = b^2 + c^2 ∧ a > b ∧ b > 0

/-- Theorem: (0, 1) is a focus of the given ellipse -/
theorem focus_of_ellipse :
  ∃ (a b c : ℝ), a^2 = 5 ∧ b^2 = 4 ∧ 
  (∀ (x y : ℝ), ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  is_focus a b c 0 1 :=
sorry

end NUMINAMATH_CALUDE_focus_of_ellipse_l1026_102693


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l1026_102628

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define point K
def K : ℝ × ℝ := (-1, 0)

-- Define the property of a line passing through K
def line_through_K (m : ℝ) (x y : ℝ) : Prop := x = m*y - 1

-- Define the intersection points A and B
def intersection_points (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧
  line_through_K m x₁ y₁ ∧ line_through_K m x₂ y₂ ∧
  y₁ ≠ y₂

-- Define point D as symmetric to A with respect to x-axis
def point_D (x₁ y₁ : ℝ) : ℝ × ℝ := (x₁, -y₁)

-- Define point F
def F : ℝ × ℝ := (1, 0)

-- The main theorem
theorem fixed_point_theorem (m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  m ≠ 0 →
  intersection_points m x₁ y₁ x₂ y₂ →
  ∃ (t : ℝ), t ∈ Set.Icc 0 1 ∧ 
    F.1 = (1 - t) * x₂ + t * (point_D x₁ y₁).1 ∧
    F.2 = (1 - t) * y₂ + t * (point_D x₁ y₁).2 :=
sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l1026_102628


namespace NUMINAMATH_CALUDE_solution_set_of_equation_l1026_102605

theorem solution_set_of_equation (x : ℝ) : 
  (Real.sin (2 * x) - π * Real.sin x) * Real.sqrt (11 * x^2 - x^4 - 10) = 0 ↔ 
  x ∈ ({-Real.sqrt 10, -π, -1, 1, π, Real.sqrt 10} : Set ℝ) := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_equation_l1026_102605


namespace NUMINAMATH_CALUDE_spade_evaluation_l1026_102632

-- Define the ♠ operation
def spade (a b : ℚ) : ℚ := (3 * a + b) / (a + b)

-- Theorem statement
theorem spade_evaluation :
  spade (spade 5 (spade 3 6)) 1 = 17 / 7 := by
  sorry

end NUMINAMATH_CALUDE_spade_evaluation_l1026_102632


namespace NUMINAMATH_CALUDE_base_7_multiplication_l1026_102635

/-- Converts a number from base 7 to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

/-- Converts a number from base 10 to base 7 --/
def to_base_7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 7) ((m % 7) :: acc)
  aux n []

/-- Theorem statement --/
theorem base_7_multiplication :
  to_base_7 (to_base_10 [4, 2, 3] * to_base_10 [3]) = [5, 0, 3, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_7_multiplication_l1026_102635


namespace NUMINAMATH_CALUDE_min_distance_squared_to_point_l1026_102609

/-- The minimum distance squared from a point on the line x - y - 1 = 0 to the point (2, 2) is 1/2 -/
theorem min_distance_squared_to_point : 
  ∀ x y : ℝ, x - y - 1 = 0 → ∃ m : ℝ, m = (1 : ℝ) / 2 ∧ ∀ a b : ℝ, a - b - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≤ (a - 2)^2 + (b - 2)^2 := by
  sorry


end NUMINAMATH_CALUDE_min_distance_squared_to_point_l1026_102609


namespace NUMINAMATH_CALUDE_farm_animals_l1026_102606

theorem farm_animals (goats chickens ducks pigs : ℕ) : 
  goats = 66 →
  chickens = 2 * goats →
  ducks = (goats + chickens) / 2 →
  pigs = ducks / 3 →
  goats - pigs = 33 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l1026_102606


namespace NUMINAMATH_CALUDE_one_point_45_deg_equals_1_deg_27_min_l1026_102650

/-- Conversion of degrees to minutes -/
def deg_to_min (d : ℝ) : ℝ := d * 60

/-- Theorem stating that 1.45° is equal to 1°27′ -/
theorem one_point_45_deg_equals_1_deg_27_min :
  ∃ (deg min : ℕ), deg = 1 ∧ min = 27 ∧ 1.45 = deg + (min : ℝ) / 60 :=
by
  sorry

end NUMINAMATH_CALUDE_one_point_45_deg_equals_1_deg_27_min_l1026_102650


namespace NUMINAMATH_CALUDE_union_equals_reals_l1026_102617

-- Define sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Theorem statement
theorem union_equals_reals : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_equals_reals_l1026_102617


namespace NUMINAMATH_CALUDE_function_value_symmetry_l1026_102667

/-- Given a function f(x) = ax^7 + bx - 2 where f(2008) = 10, prove that f(-2008) = -12 -/
theorem function_value_symmetry (a b : ℝ) :
  let f := λ x : ℝ => a * x^7 + b * x - 2
  f 2008 = 10 → f (-2008) = -12 := by
sorry

end NUMINAMATH_CALUDE_function_value_symmetry_l1026_102667


namespace NUMINAMATH_CALUDE_crayon_distribution_sum_l1026_102639

def arithmeticSequenceSum (n : ℕ) (a₁ : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem crayon_distribution_sum :
  arithmeticSequenceSum 18 12 2 = 522 := by
  sorry

end NUMINAMATH_CALUDE_crayon_distribution_sum_l1026_102639


namespace NUMINAMATH_CALUDE_circle_area_increase_l1026_102636

theorem circle_area_increase (r : ℝ) (hr : r > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 5.25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l1026_102636


namespace NUMINAMATH_CALUDE_intersection_area_of_circles_l1026_102647

/-- The area of intersection of two circles with radius 4 and centers 4/α apart -/
theorem intersection_area_of_circles (α : ℝ) (h : α = 1/2) : 
  let r : ℝ := 4
  let d : ℝ := 4/α
  let β : ℝ := (2*r)^2 - 2*(π*r^2/2)
  β = 64 - 16*π := by sorry

end NUMINAMATH_CALUDE_intersection_area_of_circles_l1026_102647


namespace NUMINAMATH_CALUDE_total_fish_is_23_l1026_102614

/-- The total number of fish caught by Brendan and his dad -/
def total_fish (morning_catch : ℕ) (thrown_back : ℕ) (afternoon_catch : ℕ) (dad_catch : ℕ) : ℕ :=
  (morning_catch - thrown_back + afternoon_catch) + dad_catch

/-- Theorem stating that the total number of fish caught is 23 -/
theorem total_fish_is_23 :
  total_fish 8 3 5 13 = 23 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_is_23_l1026_102614


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l1026_102603

theorem factorization_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_one_l1026_102603


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_l1026_102654

/-- Given a hyperbola (x²/m² - y² = 1) with m > 0, if one of its asymptotes
    is the line x + √3y = 0, then m = √3 -/
theorem hyperbola_asymptote_implies_m (m : ℝ) :
  m > 0 →
  (∃ x y : ℝ, x^2 / m^2 - y^2 = 1) →
  (∃ x y : ℝ, x + Real.sqrt 3 * y = 0) →
  m = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_implies_m_l1026_102654


namespace NUMINAMATH_CALUDE_point_slope_problem_l1026_102621

/-- If m > 0 and the points (m, 4) and (2, m) lie on a line with slope m², then m = 2. -/
theorem point_slope_problem (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 4) / (2 - m) = m^2) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_slope_problem_l1026_102621


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l1026_102679

theorem angle_sum_at_point (y : ℝ) : 
  (170 : ℝ) + y + y = 360 → y = 95 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l1026_102679


namespace NUMINAMATH_CALUDE_minimum_red_chips_l1026_102610

theorem minimum_red_chips 
  (w b r : ℕ) 
  (blue_white : b ≥ w / 4)
  (blue_red : b ≤ r / 6)
  (white_blue_total : w + b ≥ 75) :
  r ≥ 90 ∧ ∀ r', (∃ w' b', 
    b' ≥ w' / 4 ∧ 
    b' ≤ r' / 6 ∧ 
    w' + b' ≥ 75 ∧ 
    r' < 90) → False :=
sorry

end NUMINAMATH_CALUDE_minimum_red_chips_l1026_102610


namespace NUMINAMATH_CALUDE_mixture_volume_proof_l1026_102670

/-- The initial volume of the mixture -/
def initial_volume : ℝ := 150

/-- The percentage of water in the initial mixture -/
def initial_water_percentage : ℝ := 0.15

/-- The volume of water added to the mixture -/
def added_water : ℝ := 20

/-- The percentage of water in the new mixture after adding water -/
def new_water_percentage : ℝ := 0.25

theorem mixture_volume_proof :
  initial_volume = 150 ∧
  initial_water_percentage * initial_volume + added_water = new_water_percentage * (initial_volume + added_water) :=
by sorry

end NUMINAMATH_CALUDE_mixture_volume_proof_l1026_102670


namespace NUMINAMATH_CALUDE_voldemort_lunch_calories_l1026_102604

/-- Calculates the calories consumed for lunch given the daily calorie limit,
    calories from dinner items, breakfast, and remaining calories. -/
def calories_for_lunch (daily_limit : ℕ) (cake : ℕ) (chips : ℕ) (coke : ℕ)
                       (breakfast : ℕ) (remaining : ℕ) : ℕ :=
  daily_limit - (cake + chips + coke + breakfast + remaining)

/-- Proves that Voldemort consumed 780 calories for lunch. -/
theorem voldemort_lunch_calories :
  calories_for_lunch 2500 110 310 215 560 525 = 780 := by
  sorry

end NUMINAMATH_CALUDE_voldemort_lunch_calories_l1026_102604


namespace NUMINAMATH_CALUDE_expression_equality_l1026_102692

theorem expression_equality : 10 * 0.2 * 5 * 0.1 + 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1026_102692


namespace NUMINAMATH_CALUDE_bakery_flour_usage_l1026_102663

theorem bakery_flour_usage :
  0.2 + 0.1 + 0.15 + 0.05 + 0.1 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_usage_l1026_102663


namespace NUMINAMATH_CALUDE_bottles_per_child_per_day_is_three_l1026_102602

/-- Represents a children's camp with water consumption information -/
structure ChildrenCamp where
  group1 : Nat
  group2 : Nat
  group3 : Nat
  initialCases : Nat
  bottlesPerCase : Nat
  campDuration : Nat
  additionalBottles : Nat

/-- Calculates the number of bottles each child consumes per day -/
def bottlesPerChildPerDay (camp : ChildrenCamp) : Rat :=
  let group4 := (camp.group1 + camp.group2 + camp.group3) / 2
  let totalChildren := camp.group1 + camp.group2 + camp.group3 + group4
  let initialBottles := camp.initialCases * camp.bottlesPerCase
  let totalBottles := initialBottles + camp.additionalBottles
  (totalBottles : Rat) / (totalChildren * camp.campDuration)

/-- Theorem stating that for the given camp configuration, each child consumes 3 bottles per day -/
theorem bottles_per_child_per_day_is_three :
  let camp := ChildrenCamp.mk 14 16 12 13 24 3 255
  bottlesPerChildPerDay camp = 3 := by sorry

end NUMINAMATH_CALUDE_bottles_per_child_per_day_is_three_l1026_102602


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l1026_102698

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- angles are supplementary
  a / b = 5 / 3 →  -- ratio of angles is 5:3
  b = 67.5 :=  -- smaller angle is 67.5°
by
  sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l1026_102698


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1026_102695

/-- The eccentricity of a hyperbola with given equation and asymptote -/
theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∀ x y : ℝ, Real.sqrt 5 * x - 2 * y = 0 → y = (Real.sqrt 5 / 2) * x) →
  b / a = Real.sqrt 5 / 2 →
  Real.sqrt ((a^2 + b^2) / a^2) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1026_102695


namespace NUMINAMATH_CALUDE_total_hike_length_l1026_102607

/-- Represents Ella's hike over three days -/
structure HikeData where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ

/-- Conditions of Ella's hike -/
def isValidHike (h : HikeData) : Prop :=
  h.day1 + h.day2 = 18 ∧
  (h.day1 + h.day3) / 2 = 12 ∧
  h.day2 + h.day3 = 24 ∧
  h.day2 + h.day3 = 20

/-- Theorem stating the total length of the trail -/
theorem total_hike_length (h : HikeData) (hValid : isValidHike h) :
  h.day1 + h.day2 + h.day3 = 31 := by
  sorry

end NUMINAMATH_CALUDE_total_hike_length_l1026_102607


namespace NUMINAMATH_CALUDE_tan_2alpha_values_l1026_102611

theorem tan_2alpha_values (α : ℝ) (h : 2 * Real.sin (2 * α) = 1 + Real.cos (2 * α)) :
  Real.tan (2 * α) = 4/3 ∨ Real.tan (2 * α) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_2alpha_values_l1026_102611


namespace NUMINAMATH_CALUDE_counterexample_exists_l1026_102677

theorem counterexample_exists : ∃ a : ℝ, a > -2 ∧ ¬(a^2 > 4) :=
  ⟨0, by
    constructor
    · -- Prove 0 > -2
      sorry
    · -- Prove ¬(0^2 > 4)
      sorry⟩

#check counterexample_exists

end NUMINAMATH_CALUDE_counterexample_exists_l1026_102677


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l1026_102669

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_s

/-- Proof that a train with speed 30 km/h crossing a pole in 9 seconds has a length of approximately 75 meters -/
theorem train_length_proof (ε : ℝ) (h_ε : ε > 0) :
  ∃ (l : ℝ), abs (l - train_length 30 9) < ε ∧ l = 75 := by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l1026_102669


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l1026_102653

/-- 
Given a line ax + by + 1 = 0 where the distance from the origin to this line is 1/2,
prove that the circles (x - a)² + y² = 1 and x² + (y - b)² = 1 are externally tangent.
-/
theorem circles_externally_tangent (a b : ℝ) 
  (h : (a^2 + b^2)⁻¹ = 1/4) : 
  let d := Real.sqrt (a^2 + b^2)
  d = 2 := by sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l1026_102653


namespace NUMINAMATH_CALUDE_three_white_marbles_possible_l1026_102660

/-- Represents the possible operations on the urn --/
inductive Operation
  | op1 : Operation  -- Remove 4 black, add 2 black
  | op2 : Operation  -- Remove 3 black and 1 white, add 1 black
  | op3 : Operation  -- Remove 2 black and 2 white, add 2 white and 1 black
  | op4 : Operation  -- Remove 1 black and 3 white, add 3 white
  | op5 : Operation  -- Remove 4 white, add 2 black and 1 white

/-- Represents the state of the urn --/
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

/-- Applies an operation to the urn state --/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.op1 => ⟨state.white, state.black - 2⟩
  | Operation.op2 => ⟨state.white - 1, state.black - 2⟩
  | Operation.op3 => ⟨state.white, state.black - 1⟩
  | Operation.op4 => ⟨state.white, state.black - 1⟩
  | Operation.op5 => ⟨state.white - 3, state.black + 2⟩

/-- Theorem: It's possible to reach a state with 3 white marbles --/
theorem three_white_marbles_possible :
  ∃ (ops : List Operation), 
    let finalState := ops.foldl applyOperation ⟨150, 150⟩
    finalState.white = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_white_marbles_possible_l1026_102660


namespace NUMINAMATH_CALUDE_dress_designs_count_l1026_102697

/-- The number of fabric colors available -/
def num_colors : ℕ := 4

/-- The number of patterns available -/
def num_patterns : ℕ := 5

/-- Each dress design requires exactly one color and one pattern -/
def one_color_one_pattern : Prop := True

/-- The number of different dress designs possible -/
def num_designs : ℕ := num_colors * num_patterns

/-- Theorem stating that the number of different dress designs is 20 -/
theorem dress_designs_count : num_designs = 20 := by
  sorry

end NUMINAMATH_CALUDE_dress_designs_count_l1026_102697


namespace NUMINAMATH_CALUDE_girls_only_wind_count_l1026_102657

/-- Represents the number of students in different categories of the school bands -/
structure BandParticipation where
  wind_boys : ℕ
  wind_girls : ℕ
  string_boys : ℕ
  string_girls : ℕ
  total_students : ℕ
  boys_in_both : ℕ

/-- Calculates the number of girls participating only in the wind band -/
def girls_only_wind (bp : BandParticipation) : ℕ :=
  bp.wind_girls - (bp.total_students - (bp.wind_boys + bp.wind_girls + bp.string_boys + bp.string_girls - bp.boys_in_both) - bp.boys_in_both)

/-- The main theorem stating that given the specific band participation numbers, 
    the number of girls participating only in the wind band is 10 -/
theorem girls_only_wind_count : 
  let bp : BandParticipation := {
    wind_boys := 100,
    wind_girls := 80,
    string_boys := 80,
    string_girls := 100,
    total_students := 230,
    boys_in_both := 60
  }
  girls_only_wind bp = 10 := by sorry

end NUMINAMATH_CALUDE_girls_only_wind_count_l1026_102657


namespace NUMINAMATH_CALUDE_largest_number_bound_l1026_102680

theorem largest_number_bound (a b c : ℝ) (sum_zero : a + b + c = 0) (product_eight : a * b * c = 8) :
  max a (max b c) ≥ 2 * Real.rpow 4 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_largest_number_bound_l1026_102680


namespace NUMINAMATH_CALUDE_max_value_of_g_l1026_102626

/-- Given f(x) = sin x + a cos x with a symmetry axis at x = 5π/3,
    prove that the maximum value of g(x) = a sin x + cos x is 2√3/3 -/
theorem max_value_of_g (a : ℝ) (f g : ℝ → ℝ) (h₁ : ∀ x, f x = Real.sin x + a * Real.cos x)
    (h₂ : ∀ x, f x = f (10 * Real.pi / 3 - x))
    (h₃ : ∀ x, g x = a * Real.sin x + Real.cos x) :
    (∀ x, g x ≤ 2 * Real.sqrt 3 / 3) ∧ ∃ x, g x = 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_g_l1026_102626


namespace NUMINAMATH_CALUDE_exact_fare_payment_l1026_102641

/-- The bus fare in kopecks -/
def busFare : ℕ := 5

/-- The smallest coin denomination in kopecks -/
def smallestCoin : ℕ := 10

/-- The number of passengers is always a multiple of 4 -/
def numPassengers (k : ℕ) : ℕ := 4 * k

/-- The minimum number of coins required for exact fare payment -/
def minCoins (k : ℕ) : ℕ := 5 * k

theorem exact_fare_payment (k : ℕ) (h : k > 0) :
  ∀ (n : ℕ), n < minCoins k → ¬∃ (coins : List ℕ),
    (∀ c ∈ coins, c ≥ smallestCoin) ∧
    coins.length = n ∧
    coins.sum = busFare * numPassengers k :=
  sorry

#check exact_fare_payment

end NUMINAMATH_CALUDE_exact_fare_payment_l1026_102641


namespace NUMINAMATH_CALUDE_factory_produces_4000_candies_l1026_102674

/-- Represents a candy factory with its production rate and work schedule. -/
structure CandyFactory where
  production_rate : ℕ  -- candies per hour
  work_hours_per_day : ℕ
  work_days : ℕ

/-- Calculates the total number of candies produced by a factory. -/
def total_candies_produced (factory : CandyFactory) : ℕ :=
  factory.production_rate * factory.work_hours_per_day * factory.work_days

/-- Theorem stating that a factory with the given parameters produces 4000 candies. -/
theorem factory_produces_4000_candies 
  (factory : CandyFactory) 
  (h1 : factory.production_rate = 50)
  (h2 : factory.work_hours_per_day = 10)
  (h3 : factory.work_days = 8) : 
  total_candies_produced factory = 4000 := by
  sorry

#eval total_candies_produced { production_rate := 50, work_hours_per_day := 10, work_days := 8 }

end NUMINAMATH_CALUDE_factory_produces_4000_candies_l1026_102674


namespace NUMINAMATH_CALUDE_sibling_ages_sum_l1026_102646

theorem sibling_ages_sum (a b c : ℕ+) : 
  a < b → b < c → a * b * c = 72 → a + b + c = 13 := by sorry

end NUMINAMATH_CALUDE_sibling_ages_sum_l1026_102646


namespace NUMINAMATH_CALUDE_waiter_customers_l1026_102613

/-- Calculates the final number of customers for a waiter given the initial number,
    the number who left, and the number of new customers. -/
def final_customers (initial : ℕ) (left : ℕ) (new : ℕ) : ℕ :=
  initial - left + new

/-- Theorem stating that for the given scenario, the final number of customers is 28. -/
theorem waiter_customers : final_customers 33 31 26 = 28 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l1026_102613


namespace NUMINAMATH_CALUDE_married_couples_with_more_than_three_children_l1026_102625

theorem married_couples_with_more_than_three_children 
  (total_couples : ℝ) 
  (couples_more_than_one_child : ℝ) 
  (couples_two_or_three_children : ℝ) 
  (h1 : couples_more_than_one_child = (3 / 5) * total_couples)
  (h2 : couples_two_or_three_children = 0.2 * total_couples) :
  (couples_more_than_one_child - couples_two_or_three_children) / total_couples = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_married_couples_with_more_than_three_children_l1026_102625


namespace NUMINAMATH_CALUDE_bucket_capacity_reduction_l1026_102634

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 200 →
  capacity_ratio = 4 / 5 →
  (original_buckets : ℚ) / capacity_ratio = 250 := by
sorry

end NUMINAMATH_CALUDE_bucket_capacity_reduction_l1026_102634


namespace NUMINAMATH_CALUDE_no_linear_term_implies_k_equals_four_l1026_102656

theorem no_linear_term_implies_k_equals_four (k : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + k) * (x - 4) = a * x^2 + b) → k = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_implies_k_equals_four_l1026_102656


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1026_102616

theorem regular_polygon_sides (interior_angle : ℝ) (n : ℕ) : 
  interior_angle = 120 → (n : ℝ) * (180 - interior_angle) = 360 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1026_102616


namespace NUMINAMATH_CALUDE_base_number_proof_l1026_102618

theorem base_number_proof (n : ℕ) (x : ℝ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^28) 
  (h2 : n = 27) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l1026_102618


namespace NUMINAMATH_CALUDE_square_of_95_l1026_102686

theorem square_of_95 : 95^2 = 9025 := by
  sorry

end NUMINAMATH_CALUDE_square_of_95_l1026_102686


namespace NUMINAMATH_CALUDE_slope_range_of_intersecting_line_l1026_102688

/-- Given points A, B, and P, and a line l passing through P and intersecting line segment AB,
    prove that the range of the slope of line l is [0, π/4] ∪ [3π/4, π). -/
theorem slope_range_of_intersecting_line (A B P : ℝ × ℝ) (l : Set (ℝ × ℝ)) 
    (hA : A = (1, -2))
    (hB : B = (2, 1))
    (hP : P = (0, -1))
    (hl : P ∈ l)
    (hintersect : ∃ Q ∈ l, Q ∈ Set.Icc A B) :
  ∃ s : Set ℝ, s = Set.Icc 0 (π/4) ∪ Set.Ico (3*π/4) π ∧
    ∀ θ : ℝ, (∃ Q ∈ l, Q ≠ P ∧ Real.tan θ = (Q.2 - P.2) / (Q.1 - P.1)) → θ ∈ s :=
sorry

end NUMINAMATH_CALUDE_slope_range_of_intersecting_line_l1026_102688


namespace NUMINAMATH_CALUDE_product_of_decimals_l1026_102622

theorem product_of_decimals : (0.5 : ℝ) * 0.3 = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l1026_102622


namespace NUMINAMATH_CALUDE_apartment_fraction_sum_l1026_102629

theorem apartment_fraction_sum : 
  let one_bedroom : ℝ := 0.12
  let two_bedroom : ℝ := 0.26
  let three_bedroom : ℝ := 0.38
  let four_bedroom : ℝ := 0.24
  one_bedroom + two_bedroom + three_bedroom = 0.76 :=
by sorry

end NUMINAMATH_CALUDE_apartment_fraction_sum_l1026_102629


namespace NUMINAMATH_CALUDE_max_full_pikes_l1026_102642

/-- The maximum number of full pikes given initial conditions -/
theorem max_full_pikes (initial_pikes : ℕ) (full_requirement : ℕ) 
  (h1 : initial_pikes = 30)
  (h2 : full_requirement = 3) : 
  ∃ (max_full : ℕ), max_full = 9 ∧ 
  (∀ (n : ℕ), n ≤ initial_pikes - 1 → n * full_requirement ≤ initial_pikes - 1 ↔ n ≤ max_full) :=
sorry

end NUMINAMATH_CALUDE_max_full_pikes_l1026_102642


namespace NUMINAMATH_CALUDE_model_c_sample_size_l1026_102601

/-- Calculates the number of units to be sampled from a specific model in stratified sampling. -/
def stratified_sample_size (total_units : ℕ) (sample_size : ℕ) (model_units : ℕ) : ℕ :=
  (model_units * sample_size) / total_units

/-- Theorem stating that the stratified sample size for Model C is 10 units. -/
theorem model_c_sample_size :
  let total_units : ℕ := 1400 + 5600 + 2000
  let sample_size : ℕ := 45
  let model_c_units : ℕ := 2000
  stratified_sample_size total_units sample_size model_c_units = 10 := by
  sorry

end NUMINAMATH_CALUDE_model_c_sample_size_l1026_102601


namespace NUMINAMATH_CALUDE_equation_is_parabola_l1026_102671

/-- Represents a conic section --/
inductive ConicSection
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines the type of conic section for the given equation --/
def determine_conic_section (equation : ℝ → ℝ → Prop) : ConicSection := sorry

/-- The equation |x-3| = √((y+4)² + x²) --/
def equation (x y : ℝ) : Prop :=
  |x - 3| = Real.sqrt ((y + 4)^2 + x^2)

theorem equation_is_parabola :
  determine_conic_section equation = ConicSection.Parabola := by sorry

end NUMINAMATH_CALUDE_equation_is_parabola_l1026_102671


namespace NUMINAMATH_CALUDE_add_minutes_theorem_l1026_102600

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Adds minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

/-- The starting DateTime -/
def startTime : DateTime :=
  { year := 2020, month := 2, day := 1, hour := 18, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3457

/-- The expected end DateTime -/
def endTime : DateTime :=
  { year := 2020, month := 2, day := 4, hour := 3, minute := 37 }

/-- Theorem stating that adding minutesToAdd to startTime results in endTime -/
theorem add_minutes_theorem : addMinutes startTime minutesToAdd = endTime :=
  sorry

end NUMINAMATH_CALUDE_add_minutes_theorem_l1026_102600


namespace NUMINAMATH_CALUDE_abc_greater_than_28_l1026_102665

-- Define the polynomials P and Q
def P (a b c x : ℝ) : ℝ := a * x^3 + (b - a) * x^2 - (c + b) * x + c
def Q (a b c x : ℝ) : ℝ := x^4 + (b - 1) * x^3 + (a - b) * x^2 - (c + a) * x + c

-- State the theorem
theorem abc_greater_than_28 
  (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (hb_pos : b > 0)
  (hP_roots : ∃ x₀ x₁ x₂ : ℝ, x₀ ≠ x₁ ∧ x₀ ≠ x₂ ∧ x₁ ≠ x₂ ∧ 
    P a b c x₀ = 0 ∧ P a b c x₁ = 0 ∧ P a b c x₂ = 0)
  (hQ_roots : ∃ x₀ x₁ x₂ : ℝ, 
    Q a b c x₀ = 0 ∧ Q a b c x₁ = 0 ∧ Q a b c x₂ = 0) :
  a * b * c > 28 :=
sorry

end NUMINAMATH_CALUDE_abc_greater_than_28_l1026_102665


namespace NUMINAMATH_CALUDE_triangle_side_length_l1026_102640

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 9 →
  b = 2 * Real.sqrt 3 →
  C = 150 * π / 180 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1026_102640


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1026_102682

/-- The set of complex numbers z satisfying the equation (1/5)^|z-3| = (1/5)^(|z+3|-1) 
    forms a hyperbola with foci on the x-axis, a real semi-axis length of 1/2, 
    and specifically represents the right branch. -/
theorem hyperbola_equation (z : ℂ) : 
  (1/5 : ℝ) ^ Complex.abs (z - 3) = (1/5 : ℝ) ^ (Complex.abs (z + 3) - 1) →
  ∃ (a : ℝ), a = 1/2 ∧ 
    Complex.abs (z + 3) - Complex.abs (z - 3) = 2 * a ∧
    z.re > 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1026_102682


namespace NUMINAMATH_CALUDE_bead_arrangement_probability_l1026_102631

/-- Represents the number of beads of each color -/
structure BeadCounts where
  red : Nat
  white : Nat
  blue : Nat
  green : Nat

/-- The probability of arranging beads with no adjacent same colors -/
def probability_no_adjacent_same_color (counts : BeadCounts) : Rat :=
  sorry

/-- The main theorem stating the probability for the given bead counts -/
theorem bead_arrangement_probability : 
  probability_no_adjacent_same_color ⟨4, 3, 2, 1⟩ = 1 / 252 := by
  sorry

end NUMINAMATH_CALUDE_bead_arrangement_probability_l1026_102631


namespace NUMINAMATH_CALUDE_walnut_logs_per_tree_l1026_102699

theorem walnut_logs_per_tree (pine_trees maple_trees walnut_trees : ℕ)
  (logs_per_pine logs_per_maple total_logs : ℕ) :
  pine_trees = 8 →
  maple_trees = 3 →
  walnut_trees = 4 →
  logs_per_pine = 80 →
  logs_per_maple = 60 →
  total_logs = 1220 →
  ∃ logs_per_walnut : ℕ,
    logs_per_walnut = 100 ∧
    total_logs = pine_trees * logs_per_pine + maple_trees * logs_per_maple + walnut_trees * logs_per_walnut :=
by sorry

end NUMINAMATH_CALUDE_walnut_logs_per_tree_l1026_102699
