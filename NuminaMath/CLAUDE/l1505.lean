import Mathlib

namespace NUMINAMATH_CALUDE_box_cubes_required_l1505_150589

theorem box_cubes_required (length width height cube_volume : ℕ) 
  (h1 : length = 12)
  (h2 : width = 16)
  (h3 : height = 6)
  (h4 : cube_volume = 3) : 
  (length * width * height) / cube_volume = 384 := by
  sorry

end NUMINAMATH_CALUDE_box_cubes_required_l1505_150589


namespace NUMINAMATH_CALUDE_triangle_count_is_53_l1505_150579

/-- Represents a rectangle divided into smaller sections -/
structure DividedRectangle where
  width : ℕ
  height : ℕ
  horizontal_divisions : ℕ
  vertical_divisions : ℕ

/-- Counts the number of triangles in a divided rectangle -/
def count_triangles (r : DividedRectangle) : ℕ :=
  let smallest_triangles := 24
  let isosceles_triangles := 6
  let rectangular_half_area_triangles := 12
  let larger_right_triangles := 8
  let large_isosceles_triangles := 3
  smallest_triangles + isosceles_triangles + rectangular_half_area_triangles + 
  larger_right_triangles + large_isosceles_triangles

/-- The total number of triangles in the given divided rectangle is 53 -/
theorem triangle_count_is_53 (r : DividedRectangle) : 
  count_triangles r = 53 := by sorry

end NUMINAMATH_CALUDE_triangle_count_is_53_l1505_150579


namespace NUMINAMATH_CALUDE_equation_solution_l1505_150502

theorem equation_solution :
  ∃ (x : ℝ), x > 0 ∧ 6 * Real.sqrt (4 + x) + 6 * Real.sqrt (4 - x) = 8 * Real.sqrt 5 ∧ x = Real.sqrt (1280 / 81) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1505_150502


namespace NUMINAMATH_CALUDE_range_of_a_l1505_150550

-- Define the conditions
def p (x : ℝ) : Prop := (4*x - 3)^2 ≤ 1
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

-- Define the set A (solution set for p)
def A : Set ℝ := {x | p x}

-- Define the set B (solution set for q)
def B (a : ℝ) : Set ℝ := {x | q x a}

-- State the theorem
theorem range_of_a :
  (∀ a : ℝ, (A ⊆ B a) ∧ (A ≠ B a)) →
  (∃ a_min a_max : ℝ, a_min = 0 ∧ a_max = 1/2 ∧ ∀ a : ℝ, a_min ≤ a ∧ a ≤ a_max) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1505_150550


namespace NUMINAMATH_CALUDE_parabola_equation_l1505_150500

/-- A parabola with the same shape and orientation as y = -2x^2 + 2 and vertex (4, -2) -/
structure Parabola where
  shape_coeff : ℝ
  vertex : ℝ × ℝ
  shape_matches : shape_coeff = -2
  vertex_coords : vertex = (4, -2)

/-- The analytical expression of the parabola -/
def parabola_expression (p : Parabola) (x : ℝ) : ℝ :=
  p.shape_coeff * (x - p.vertex.1)^2 + p.vertex.2

theorem parabola_equation (p : Parabola) :
  ∀ x, parabola_expression p x = -2 * (x - 4)^2 - 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1505_150500


namespace NUMINAMATH_CALUDE_triangle_angle_tangent_difference_l1505_150513

theorem triangle_angle_tangent_difference (A B : Real) (cosA tanB : Real) 
  (h1 : cosA = -Real.sqrt 2 / 2)
  (h2 : tanB = 1 / 3) :
  Real.tan (A - B) = -2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_tangent_difference_l1505_150513


namespace NUMINAMATH_CALUDE_points_per_enemy_l1505_150590

theorem points_per_enemy (total_enemies : ℕ) (enemies_not_destroyed : ℕ) (total_points : ℕ) : 
  total_enemies = 8 →
  enemies_not_destroyed = 6 →
  total_points = 10 →
  (total_points : ℚ) / (total_enemies - enemies_not_destroyed : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_points_per_enemy_l1505_150590


namespace NUMINAMATH_CALUDE_complex_number_properties_l1505_150544

theorem complex_number_properties (z₁ z₂ : ℂ) 
  (hz₁ : z₁ = 1 + 2*I) (hz₂ : z₂ = 3 - 4*I) : 
  (Complex.im (z₁ * z₂) = 2) ∧ 
  (Complex.re (z₁ * z₂) > 0 ∧ Complex.im (z₁ * z₂) > 0) ∧
  (Complex.re z₁ > 0 ∧ Complex.im z₁ > 0) := by
  sorry


end NUMINAMATH_CALUDE_complex_number_properties_l1505_150544


namespace NUMINAMATH_CALUDE_exam_students_count_l1505_150503

theorem exam_students_count : 
  ∀ (total : ℕ) (first_div second_div just_passed : ℝ),
    first_div = 0.25 * total →
    second_div = 0.54 * total →
    just_passed = total - first_div - second_div →
    just_passed = 63 →
    total = 300 := by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l1505_150503


namespace NUMINAMATH_CALUDE_sum_of_roots_l1505_150563

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 3*a^2 - 4*a + 12 = 0)
  (hb : 3*b^3 + 9*b^2 - 11*b - 3 = 0) : 
  a + b = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1505_150563


namespace NUMINAMATH_CALUDE_first_terrific_tuesday_after_school_start_l1505_150581

/-- Represents a date in October --/
structure OctoberDate :=
  (day : Nat)
  (h : day ≥ 1 ∧ day ≤ 31)

/-- Represents a day of the week --/
inductive DayOfWeek
  | Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

/-- Function to determine the day of the week for a given October date --/
def dayOfWeek (date : OctoberDate) : DayOfWeek :=
  sorry

/-- Predicate to check if a date is a Tuesday --/
def isTuesday (date : OctoberDate) : Prop :=
  dayOfWeek date = DayOfWeek.Tuesday

/-- Function to count the number of Tuesdays before a given date in October --/
def tuesdaysBeforeDate (date : OctoberDate) : Nat :=
  sorry

/-- Predicate to check if a date is a Terrific Tuesday --/
def isTerrificTuesday (date : OctoberDate) : Prop :=
  isTuesday date ∧ tuesdaysBeforeDate date = 4

/-- The school start date --/
def schoolStartDate : OctoberDate :=
  ⟨5, sorry⟩

/-- Theorem: The first Terrific Tuesday after school starts is October 31 --/
theorem first_terrific_tuesday_after_school_start :
  ∃ (date : OctoberDate),
    date.day = 31 ∧
    isTerrificTuesday date ∧
    (∀ (earlier_date : OctoberDate),
      earlier_date.day > schoolStartDate.day ∧
      earlier_date.day < date.day →
      ¬isTerrificTuesday earlier_date) :=
  sorry

end NUMINAMATH_CALUDE_first_terrific_tuesday_after_school_start_l1505_150581


namespace NUMINAMATH_CALUDE_total_painting_cost_l1505_150549

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

def count_digits (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

def house_cost (address : ℕ) : ℚ :=
  (1.5 : ℚ) * (count_digits address)

def side_cost (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℚ :=
  List.sum (List.map house_cost (List.map (arithmetic_sequence a₁ d) (List.range n)))

theorem total_painting_cost :
  side_cost 5 6 25 + side_cost 2 6 25 = 171 := by
  sorry

end NUMINAMATH_CALUDE_total_painting_cost_l1505_150549


namespace NUMINAMATH_CALUDE_total_meows_eq_286_l1505_150530

/-- The number of meows for eight cats over 12 minutes -/
def total_meows : ℕ :=
  let cat1_meows := 3 * 12
  let cat2_meows := (3 * 2) * 12
  let cat3_meows := ((3 * 2) / 3) * 12
  let cat4_meows := 4 * 12
  let cat5_meows := (60 / 45) * 12
  let cat6_meows := (5 / 2) * 12
  let cat7_meows := ((3 * 2) / 2) * 12
  let cat8_meows := (6 / 3) * 12
  cat1_meows + cat2_meows + cat3_meows + cat4_meows + 
  cat5_meows + cat6_meows + cat7_meows + cat8_meows

theorem total_meows_eq_286 : total_meows = 286 := by
  sorry

#eval total_meows

end NUMINAMATH_CALUDE_total_meows_eq_286_l1505_150530


namespace NUMINAMATH_CALUDE_min_value_reciprocal_squares_l1505_150582

theorem min_value_reciprocal_squares (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_constraint : a + b + c = 3) : 
  1/a^2 + 1/b^2 + 1/c^2 ≥ 3 ∧ 
  (1/a^2 + 1/b^2 + 1/c^2 = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

#check min_value_reciprocal_squares

end NUMINAMATH_CALUDE_min_value_reciprocal_squares_l1505_150582


namespace NUMINAMATH_CALUDE_hexagon_area_theorem_l1505_150574

/-- Regular hexagon with vertices A and C -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a regular hexagon given its vertices A and C -/
def hexagon_area (h : RegularHexagon) : ℝ :=
  sorry

theorem hexagon_area_theorem (h : RegularHexagon) 
  (h_A : h.A = (0, 0)) 
  (h_C : h.C = (8, 2)) : 
  hexagon_area h = 34 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_theorem_l1505_150574


namespace NUMINAMATH_CALUDE_inequality_transformation_l1505_150598

theorem inequality_transformation (a : ℝ) : 
  (∀ x, (1 - a) * x > 2 ↔ x < 2 / (1 - a)) → a > 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_transformation_l1505_150598


namespace NUMINAMATH_CALUDE_comparison_problems_l1505_150547

theorem comparison_problems :
  (-0.1 < -0.01) ∧
  (-(-1) = abs (-1)) ∧
  (-abs (-7/8) < -(5/6)) := by sorry

end NUMINAMATH_CALUDE_comparison_problems_l1505_150547


namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l1505_150521

theorem quadratic_square_of_binomial (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 24*x + c = (a*x + b)^2) → c = 16 := by
sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l1505_150521


namespace NUMINAMATH_CALUDE_total_cost_theorem_l1505_150517

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def children_per_brother : ℕ := 2

def total_relatives : ℕ := num_parents + num_brothers + num_brothers + (num_brothers * children_per_brother)

theorem total_cost_theorem : cost_per_package * total_relatives = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_theorem_l1505_150517


namespace NUMINAMATH_CALUDE_no_perfect_square_with_conditions_l1505_150534

def is_nine_digit (n : ℕ) : Prop := 10^8 ≤ n ∧ n < 10^9

def contains_all_nonzero_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 → ∃ k : ℕ, n / 10^k % 10 = d

def last_digit_is_five (n : ℕ) : Prop := n % 10 = 5

theorem no_perfect_square_with_conditions :
  ¬ ∃ n : ℕ, is_nine_digit n ∧ contains_all_nonzero_digits n ∧ last_digit_is_five n ∧ ∃ m : ℕ, n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_no_perfect_square_with_conditions_l1505_150534


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1505_150526

theorem min_value_trigonometric_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 48 :=
by sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l1505_150526


namespace NUMINAMATH_CALUDE_conic_is_ellipse_l1505_150511

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 + 4*y^2 - 6*x + 8*y + 9 = 0

/-- Definition of an ellipse in standard form -/
def is_ellipse (h k a b : ℝ) (x y : ℝ) : Prop :=
  ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

/-- Theorem stating that the given equation represents an ellipse -/
theorem conic_is_ellipse :
  ∃ h k a b : ℝ, ∀ x y : ℝ, conic_equation x y ↔ is_ellipse h k a b x y :=
sorry

end NUMINAMATH_CALUDE_conic_is_ellipse_l1505_150511


namespace NUMINAMATH_CALUDE_percentage_problem_l1505_150595

theorem percentage_problem (x : ℝ) (h : 0.4 * x = 160) : 0.3 * x = 120 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1505_150595


namespace NUMINAMATH_CALUDE_central_angle_doubles_when_radius_halves_l1505_150593

theorem central_angle_doubles_when_radius_halves (r l α β : ℝ) (h1 : r > 0) (h2 : l > 0) (h3 : α > 0) :
  α = l / r →
  β = l / (r / 2) →
  β = 2 * α := by
sorry

end NUMINAMATH_CALUDE_central_angle_doubles_when_radius_halves_l1505_150593


namespace NUMINAMATH_CALUDE_area_of_right_isosceles_triangle_l1505_150512

/-- A right-angled isosceles triangle with the sum of the areas of squares on its sides equal to 72 -/
structure RightIsoscelesTriangle where
  /-- The length of each of the two equal sides -/
  side : ℝ
  /-- The sum of the areas of squares on the sides is 72 -/
  sum_of_squares : side^2 + side^2 + (2 * side^2) = 72

/-- The area of a right-angled isosceles triangle with the given property is 9 -/
theorem area_of_right_isosceles_triangle (t : RightIsoscelesTriangle) : 
  (1/2 : ℝ) * t.side * t.side = 9 := by
  sorry

end NUMINAMATH_CALUDE_area_of_right_isosceles_triangle_l1505_150512


namespace NUMINAMATH_CALUDE_octagon_edge_length_l1505_150505

/-- The length of one edge of a regular octagon made from the same thread as a regular pentagon with one edge of 16 cm -/
theorem octagon_edge_length (pentagon_edge : ℝ) (thread_length : ℝ) : 
  pentagon_edge = 16 → thread_length = 5 * pentagon_edge → thread_length / 8 = 10 := by
  sorry

#check octagon_edge_length

end NUMINAMATH_CALUDE_octagon_edge_length_l1505_150505


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l1505_150567

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of vertices in a regular nonagon -/
def num_vertices : ℕ := 9

/-- The number of diagonals in a regular nonagon -/
def num_diagonals (n : RegularNonagon) : ℕ := (num_vertices * (num_vertices - 3)) / 2

/-- The number of pairs of intersecting diagonals in a regular nonagon -/
def num_intersecting_diagonals (n : RegularNonagon) : ℕ := Nat.choose num_vertices 4

/-- The total number of pairs of diagonals in a regular nonagon -/
def total_diagonal_pairs (n : RegularNonagon) : ℕ := Nat.choose (num_diagonals n) 2

/-- The probability that two randomly chosen diagonals intersect inside the nonagon -/
def intersection_probability (n : RegularNonagon) : ℚ :=
  (num_intersecting_diagonals n : ℚ) / (total_diagonal_pairs n : ℚ)

theorem nonagon_diagonal_intersection_probability (n : RegularNonagon) :
  intersection_probability n = 14 / 39 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l1505_150567


namespace NUMINAMATH_CALUDE_fraction_equality_l1505_150573

theorem fraction_equality : (12 : ℚ) / (8 * 75) = 1 / 50 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1505_150573


namespace NUMINAMATH_CALUDE_cos_3x_minus_pi_3_eq_sin_3x_plus_pi_18_l1505_150576

theorem cos_3x_minus_pi_3_eq_sin_3x_plus_pi_18 (x : ℝ) : 
  Real.cos (3 * x - π / 3) = Real.sin (3 * (x + π / 18)) := by
  sorry

end NUMINAMATH_CALUDE_cos_3x_minus_pi_3_eq_sin_3x_plus_pi_18_l1505_150576


namespace NUMINAMATH_CALUDE_marbles_given_l1505_150575

theorem marbles_given (drew_initial : ℕ) (marcus_initial : ℕ) (marbles_given : ℕ) : 
  drew_initial = marcus_initial + 24 →
  drew_initial - marbles_given = 25 →
  marcus_initial + marbles_given = 25 →
  marbles_given = 12 := by
sorry

end NUMINAMATH_CALUDE_marbles_given_l1505_150575


namespace NUMINAMATH_CALUDE_num_divisors_36_eq_9_l1505_150564

/-- The number of positive divisors of 36 -/
def num_divisors_36 : ℕ :=
  (Finset.filter (· ∣ 36) (Finset.range 37)).card

/-- Theorem stating that the number of positive divisors of 36 is 9 -/
theorem num_divisors_36_eq_9 : num_divisors_36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_36_eq_9_l1505_150564


namespace NUMINAMATH_CALUDE_min_games_for_prediction_l1505_150585

/-- Represents the chess tournament setup -/
structure ChessTournament where
  white_rook : ℕ  -- number of students from "White Rook" school
  black_elephant : ℕ  -- number of students from "Black Elephant" school
  total_games : ℕ  -- total number of games to be played

/-- Checks if the tournament setup is valid -/
def is_valid_tournament (t : ChessTournament) : Prop :=
  t.white_rook * t.black_elephant = t.total_games

/-- Represents the state of the tournament after some games -/
structure TournamentState where
  tournament : ChessTournament
  games_played : ℕ

/-- Checks if Sasha can predict a participant in the next game -/
def can_predict_participant (state : TournamentState) : Prop :=
  state.games_played ≥ state.tournament.total_games - state.tournament.black_elephant

/-- The main theorem to be proved -/
theorem min_games_for_prediction (t : ChessTournament) 
    (h_valid : is_valid_tournament t) 
    (h_white : t.white_rook = 15) 
    (h_black : t.black_elephant = 20) : 
    ∀ n : ℕ, can_predict_participant ⟨t, n⟩ ↔ n ≥ 280 := by
  sorry

end NUMINAMATH_CALUDE_min_games_for_prediction_l1505_150585


namespace NUMINAMATH_CALUDE_cube_root_simplification_l1505_150572

theorem cube_root_simplification :
  ∀ (a b : ℚ), a = 17 + 1/9 → (a^(1/3) : ℝ) = (b^(1/3) : ℝ) / (9^(1/3) : ℝ) → b = 154 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l1505_150572


namespace NUMINAMATH_CALUDE_souvenir_relationship_l1505_150531

/-- Represents the number of souvenirs of each type -/
structure SouvenirCount where
  x : ℕ  -- 20 cents souvenirs
  y : ℕ  -- 25 cents souvenirs
  z : ℕ  -- 35 cents souvenirs

/-- Conditions of the souvenir distribution problem -/
def SouvenirProblem (s : SouvenirCount) : Prop :=
  s.x + s.y + s.z = 2000 ∧
  20 * s.x + 25 * s.y + 35 * s.z = 52000

/-- Theorem stating the relationship between 25 cents and 35 cents souvenirs -/
theorem souvenir_relationship (s : SouvenirCount) 
  (h : SouvenirProblem s) : 5 * s.y + 15 * s.z = 12000 := by
  sorry

end NUMINAMATH_CALUDE_souvenir_relationship_l1505_150531


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1505_150596

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a / b = 3 ∧ 
   (∀ x : ℝ, x^2 - 4*x + m = 0 ↔ (x = a ∨ x = b))) → 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1505_150596


namespace NUMINAMATH_CALUDE_max_school_leaders_l1505_150528

/-- Represents the number of years in a period -/
def period : ℕ := 10

/-- Represents the length of a principal's term in years -/
def principal_term : ℕ := 3

/-- Represents the length of an assistant principal's term in years -/
def assistant_principal_term : ℕ := 2

/-- Calculates the maximum number of individuals serving in a role given the period and term length -/
def max_individuals (period : ℕ) (term : ℕ) : ℕ :=
  (period + term - 1) / term

/-- Theorem stating the maximum number of principals and assistant principals over the given period -/
theorem max_school_leaders :
  max_individuals period principal_term + max_individuals period assistant_principal_term = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_school_leaders_l1505_150528


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l1505_150587

-- Define the # operation
def hash (a b : ℝ) : ℝ := a + 2 * b

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  (∃ x y z : ℝ, hash x (y + z) ≠ hash x y + hash x z) ∧
  (∃ x y z : ℝ, x + hash y z ≠ hash (x + y) (x + z)) ∧
  (∃ x y z : ℝ, hash x (hash y z) ≠ hash (hash x y) (hash x z)) :=
by
  sorry


end NUMINAMATH_CALUDE_no_distributive_laws_hold_l1505_150587


namespace NUMINAMATH_CALUDE_train_length_l1505_150522

/-- Calculates the length of a train given the time it takes to cross a bridge and pass a lamp post. -/
theorem train_length (bridge_length : ℝ) (bridge_time : ℝ) (lamp_time : ℝ) :
  bridge_length = 150 →
  bridge_time = 7.5 →
  lamp_time = 2.5 →
  ∃ (train_length : ℝ), train_length = 75 ∧ 
    (train_length / lamp_time = (train_length + bridge_length) / bridge_time) :=
by sorry


end NUMINAMATH_CALUDE_train_length_l1505_150522


namespace NUMINAMATH_CALUDE_pages_per_chapter_book_chapters_calculation_l1505_150561

theorem pages_per_chapter 
  (total_chapters : Nat) 
  (days_to_finish : Nat) 
  (chapters_per_day : Nat) : Nat :=
  let total_chapters_read := days_to_finish * chapters_per_day
  total_chapters_read / total_chapters

theorem book_chapters_calculation 
  (total_chapters : Nat) 
  (days_to_finish : Nat) 
  (chapters_per_day : Nat) :
  total_chapters = 2 →
  days_to_finish = 664 →
  chapters_per_day = 332 →
  pages_per_chapter total_chapters days_to_finish chapters_per_day = 110224 := by
  sorry

end NUMINAMATH_CALUDE_pages_per_chapter_book_chapters_calculation_l1505_150561


namespace NUMINAMATH_CALUDE_sum_of_squares_impossible_l1505_150562

theorem sum_of_squares_impossible (n : ℤ) :
  (n % 4 = 3 → ¬∃ (a b : ℤ), n = a^2 + b^2) ∧
  (n % 8 = 7 → ¬∃ (a b c : ℤ), n = a^2 + b^2 + c^2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_impossible_l1505_150562


namespace NUMINAMATH_CALUDE_bathroom_break_duration_l1505_150509

theorem bathroom_break_duration
  (total_distance : ℝ)
  (driving_speed : ℝ)
  (lunch_break : ℝ)
  (num_bathroom_breaks : ℕ)
  (total_trip_time : ℝ)
  (h1 : total_distance = 480)
  (h2 : driving_speed = 60)
  (h3 : lunch_break = 0.5)
  (h4 : num_bathroom_breaks = 2)
  (h5 : total_trip_time = 9) :
  (total_trip_time - total_distance / driving_speed - lunch_break) / num_bathroom_breaks = 0.25 := by
  sorry

#check bathroom_break_duration

end NUMINAMATH_CALUDE_bathroom_break_duration_l1505_150509


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l1505_150543

theorem greatest_integer_less_than_negative_fraction :
  ⌊-22/5⌋ = -5 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_fraction_l1505_150543


namespace NUMINAMATH_CALUDE_sawyer_cleaning_time_l1505_150577

/-- Represents the time in hours it takes to clean the house -/
structure CleaningTime where
  sawyer : ℝ  -- Time for Sawyer to clean the entire house alone
  nick : ℝ    -- Time for Nick to clean the entire house alone

/-- The cleaning scenario with given conditions -/
def cleaning_scenario (t : CleaningTime) : Prop :=
  t.sawyer = (2/3) * t.nick ∧  -- Sawyer's time is 2/3 of Nick's time
  (1 / t.sawyer + 1 / t.nick) = 1 / 3.6  -- Combined work rate

/-- Theorem stating Sawyer's cleaning time -/
theorem sawyer_cleaning_time (t : CleaningTime) 
  (h : cleaning_scenario t) : t.sawyer = 6 := by
  sorry

end NUMINAMATH_CALUDE_sawyer_cleaning_time_l1505_150577


namespace NUMINAMATH_CALUDE_y_divisibility_l1505_150566

def y : ℕ := 96 + 144 + 200 + 320 + 480 + 512 + 4096

theorem y_divisibility :
  (∃ k : ℕ, y = 4 * k) ∧
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  ¬(∃ k : ℕ, y = 32 * k) := by
  sorry

end NUMINAMATH_CALUDE_y_divisibility_l1505_150566


namespace NUMINAMATH_CALUDE_inequality_not_always_true_l1505_150504

theorem inequality_not_always_true (a b c : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : a * c < 0) :
  ∃ a b c, a < b ∧ b < c ∧ a * c < 0 ∧ c^2 / a ≥ b^2 / a :=
sorry

end NUMINAMATH_CALUDE_inequality_not_always_true_l1505_150504


namespace NUMINAMATH_CALUDE_petes_total_miles_l1505_150520

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_steps : ℕ
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total miles walked given a pedometer and steps per mile --/
def total_miles_walked (p : Pedometer) (steps_per_mile : ℕ) : ℚ :=
  ((p.resets * (p.max_steps + 1) + p.final_reading) : ℚ) / steps_per_mile

/-- Theorem stating that Pete walked 2512.5 miles given the problem conditions --/
theorem petes_total_miles :
  let p : Pedometer := ⟨99999, 50, 25000⟩
  let steps_per_mile : ℕ := 2000
  total_miles_walked p steps_per_mile = 2512.5 := by
  sorry


end NUMINAMATH_CALUDE_petes_total_miles_l1505_150520


namespace NUMINAMATH_CALUDE_product_and_difference_theorem_l1505_150541

theorem product_and_difference_theorem (a b : ℝ) 
  (h1 : a * b = 2 * (a + b) + 10) 
  (h2 : b - a = 5) : 
  b = 9 := by sorry

end NUMINAMATH_CALUDE_product_and_difference_theorem_l1505_150541


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1505_150558

theorem geometric_series_sum (a : ℕ+) (n : ℕ+) (h : (a : ℝ) / (1 - 1 / (n : ℝ)) = 3) :
  (a : ℝ) + (a : ℝ) / (n : ℝ) = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1505_150558


namespace NUMINAMATH_CALUDE_count_numbers_equals_fifteen_l1505_150580

/-- The set of digits available on the cards -/
def digits : Finset Nat := {1, 2, 3}

/-- The function to count valid numbers formed from the digits -/
def count_numbers (digits : Finset Nat) : Nat :=
  (digits.card) +  -- one-digit numbers
  (digits.card.choose 2 * 2) +  -- two-digit numbers
  (digits.card.factorial)  -- three-digit numbers

/-- Theorem stating that the total number of different natural numbers
    that can be formed using the digits 1, 2, and 3 is equal to 15 -/
theorem count_numbers_equals_fifteen :
  count_numbers digits = 15 := by
  sorry


end NUMINAMATH_CALUDE_count_numbers_equals_fifteen_l1505_150580


namespace NUMINAMATH_CALUDE_four_times_three_plus_two_l1505_150592

theorem four_times_three_plus_two : (4 * 3) + 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_four_times_three_plus_two_l1505_150592


namespace NUMINAMATH_CALUDE_f_min_at_negative_seven_l1505_150557

/-- The quadratic function f(x) = x^2 + 14x - 12 -/
def f (x : ℝ) : ℝ := x^2 + 14*x - 12

/-- The point where f attains its minimum -/
def min_point : ℝ := -7

theorem f_min_at_negative_seven :
  ∀ x : ℝ, f x ≥ f min_point := by
sorry

end NUMINAMATH_CALUDE_f_min_at_negative_seven_l1505_150557


namespace NUMINAMATH_CALUDE_stock_worth_calculation_l1505_150584

/-- Calculates the total worth of a stock given specific selling conditions and overall loss --/
theorem stock_worth_calculation (stock_value : ℝ) : 
  (0.2 * stock_value * 1.2 + 0.8 * stock_value * 0.9) - stock_value = -500 → 
  stock_value = 12500 := by
  sorry

#check stock_worth_calculation

end NUMINAMATH_CALUDE_stock_worth_calculation_l1505_150584


namespace NUMINAMATH_CALUDE_sum_of_solutions_equation_l1505_150591

theorem sum_of_solutions_equation (x₁ x₂ : ℚ) : 
  (4 * x₁ + 7 = 0 ∨ 5 * x₁ - 8 = 0) ∧
  (4 * x₂ + 7 = 0 ∨ 5 * x₂ - 8 = 0) ∧
  x₁ ≠ x₂ →
  x₁ + x₂ = -3/20 := by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equation_l1505_150591


namespace NUMINAMATH_CALUDE_intersection_point_l1505_150514

/-- The line equation -/
def line_equation (x y z : ℝ) : Prop :=
  (x - 1) / 2 = (y - 1) / (-1) ∧ (y - 1) / (-1) = (z + 2) / 3

/-- The plane equation -/
def plane_equation (x y z : ℝ) : Prop :=
  4 * x + 2 * y - z - 11 = 0

/-- The theorem stating that (3, 0, 1) is the unique intersection point -/
theorem intersection_point :
  ∃! (x y z : ℝ), line_equation x y z ∧ plane_equation x y z ∧ x = 3 ∧ y = 0 ∧ z = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1505_150514


namespace NUMINAMATH_CALUDE_alpha_value_l1505_150556

theorem alpha_value (α : Real) 
  (h1 : -π/2 < α ∧ α < π/2) 
  (h2 : Real.sin α + Real.cos α = Real.sqrt 2 / 2) : 
  α = -π/12 := by
sorry

end NUMINAMATH_CALUDE_alpha_value_l1505_150556


namespace NUMINAMATH_CALUDE_lakeside_club_overlap_l1505_150518

/-- The number of students in both the theater and robotics clubs at Lakeside High School -/
def students_in_both_clubs (total_students theater_members robotics_members either_or_both : ℕ) : ℕ :=
  theater_members + robotics_members - either_or_both

/-- Theorem: Given the conditions from Lakeside High School, 
    the number of students in both the theater and robotics clubs is 25 -/
theorem lakeside_club_overlap : 
  let total_students : ℕ := 250
  let theater_members : ℕ := 85
  let robotics_members : ℕ := 120
  let either_or_both : ℕ := 180
  students_in_both_clubs total_students theater_members robotics_members either_or_both = 25 := by
  sorry

end NUMINAMATH_CALUDE_lakeside_club_overlap_l1505_150518


namespace NUMINAMATH_CALUDE_shaded_area_of_carpet_l1505_150532

/-- Theorem: Total shaded area of a square carpet with specific shaded squares -/
theorem shaded_area_of_carpet (S T : ℝ) : 
  S = 12 / 4 →              -- S is 1/4 of the carpet side length
  T = S / 4 →               -- T is 1/4 of S
  S^2 + 4 * T^2 = 11.25 :=  -- Total shaded area
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_carpet_l1505_150532


namespace NUMINAMATH_CALUDE_nabla_calculation_l1505_150539

def nabla (a b : ℕ) : ℕ := 3 + (Nat.factorial b) ^ a

theorem nabla_calculation : nabla (nabla 2 3) 4 = 3 + 24 ^ 39 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l1505_150539


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1505_150569

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*Complex.I)*z = 3 - 4*Complex.I) : 
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1505_150569


namespace NUMINAMATH_CALUDE_lcm_18_24_l1505_150546

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l1505_150546


namespace NUMINAMATH_CALUDE_gcd_1554_2405_l1505_150519

theorem gcd_1554_2405 : Nat.gcd 1554 2405 = 37 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1554_2405_l1505_150519


namespace NUMINAMATH_CALUDE_sophomore_freshman_difference_l1505_150508

/-- Represents the number of students in each grade -/
structure GradeDistribution where
  freshman : ℕ
  sophomore : ℕ
  junior : ℕ

/-- Represents the sample size for each grade -/
structure SampleDistribution where
  freshman : ℕ
  sophomore : ℕ
  junior : ℕ

/-- Calculates the stratified sample distribution based on the grade distribution and total sample size -/
def stratifiedSample (grades : GradeDistribution) (totalSample : ℕ) : SampleDistribution :=
  let total := grades.freshman + grades.sophomore + grades.junior
  let freshmanSample := (grades.freshman * totalSample) / total
  let sophomoreSample := (grades.sophomore * totalSample) / total
  let juniorSample := totalSample - freshmanSample - sophomoreSample
  { freshman := freshmanSample
  , sophomore := sophomoreSample
  , junior := juniorSample }

/-- The main theorem to be proved -/
theorem sophomore_freshman_difference
  (grades : GradeDistribution)
  (h1 : grades.freshman = 1000)
  (h2 : grades.sophomore = 1050)
  (h3 : grades.junior = 1200)
  (totalSample : ℕ)
  (h4 : totalSample = 65) :
  let sample := stratifiedSample grades totalSample
  sample.sophomore = sample.freshman + 1 := by
  sorry

end NUMINAMATH_CALUDE_sophomore_freshman_difference_l1505_150508


namespace NUMINAMATH_CALUDE_beatrice_prob_five_given_win_l1505_150586

-- Define the number of players and die sides
def num_players : ℕ := 5
def num_sides : ℕ := 8

-- Define the probability of rolling a specific number
def prob_roll (n : ℕ) : ℚ := 1 / num_sides

-- Define the probability of winning for any player
def prob_win : ℚ := 1 / num_players

-- Define the probability of other players rolling less than 5
def prob_others_less_than_5 : ℚ := (4 / 8) ^ (num_players - 1)

-- Define the probability of winning with a 5 (including tie-breaks)
def prob_win_with_5 : ℚ := prob_others_less_than_5 + 369 / 2048

-- State the theorem
theorem beatrice_prob_five_given_win :
  (prob_roll 5 * prob_win_with_5) / prob_win = 115 / 1024 := by
sorry

end NUMINAMATH_CALUDE_beatrice_prob_five_given_win_l1505_150586


namespace NUMINAMATH_CALUDE_target_hit_probability_l1505_150597

/-- The probability of hitting a target in one shot. -/
def p : ℝ := 0.6

/-- The number of shots taken. -/
def n : ℕ := 3

/-- The probability of hitting the target at least twice in n shots. -/
def prob_at_least_two_hits : ℝ := 
  (n.choose 2) * p^2 * (1 - p) + (n.choose 3) * p^3

theorem target_hit_probability : prob_at_least_two_hits = 81/125 := by
  sorry

end NUMINAMATH_CALUDE_target_hit_probability_l1505_150597


namespace NUMINAMATH_CALUDE_intersection_integer_coordinates_l1505_150527

theorem intersection_integer_coordinates (n : ℕ+) 
  (h : ∃ (x y : ℤ), 17 * x + 7 * y = 833 ∧ y = n * x - 3) : n = 15 := by
  sorry

end NUMINAMATH_CALUDE_intersection_integer_coordinates_l1505_150527


namespace NUMINAMATH_CALUDE_system_solution_existence_and_uniqueness_l1505_150578

theorem system_solution_existence_and_uniqueness :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ)),
    (solutions.card = 7) ∧
    (∀ (x y z : ℝ), (x, y, z) ∈ solutions ↔
      (2 * x + x^2 * y = y ∧
       2 * y + y^2 * z = z ∧
       2 * z + z^2 * x = x) ∧
      (∃ k : Fin 7, x = Real.tan (k * π / 7) ∧
                    y = Real.tan (2 * k * π / 7) ∧
                    z = Real.tan (4 * k * π / 7))) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_existence_and_uniqueness_l1505_150578


namespace NUMINAMATH_CALUDE_selection_theorem_1_selection_theorem_2_selection_theorem_3_l1505_150565

/-- The number of female students -/
def num_female : ℕ := 5

/-- The number of male students -/
def num_male : ℕ := 4

/-- The number of students to be selected -/
def num_selected : ℕ := 4

/-- The number of ways to select exactly 2 male and 2 female students -/
def selection_method_1 : ℕ := 1440

/-- The number of ways to select at least 1 male and 1 female student -/
def selection_method_2 : ℕ := 2880

/-- The number of ways to select at least 1 male and 1 female student, 
    but male student A and female student B cannot be selected together -/
def selection_method_3 : ℕ := 2376

/-- Theorem for the first selection method -/
theorem selection_theorem_1 : 
  (Nat.choose num_male 2 * Nat.choose num_female 2) * (Nat.factorial num_selected) = selection_method_1 := by
  sorry

/-- Theorem for the second selection method -/
theorem selection_theorem_2 : 
  ((Nat.choose num_male 1 * Nat.choose num_female 3) + 
   (Nat.choose num_male 2 * Nat.choose num_female 2) + 
   (Nat.choose num_male 3 * Nat.choose num_female 1)) * 
  (Nat.factorial num_selected) = selection_method_2 := by
  sorry

/-- Theorem for the third selection method -/
theorem selection_theorem_3 : 
  selection_method_2 - 
  ((Nat.choose (num_male - 1) 2 + Nat.choose (num_female - 1) 1 * Nat.choose (num_male - 1) 1 + 
    Nat.choose (num_female - 1) 2) * Nat.factorial num_selected) = selection_method_3 := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_1_selection_theorem_2_selection_theorem_3_l1505_150565


namespace NUMINAMATH_CALUDE_initial_average_weight_l1505_150554

/-- Proves that the initially calculated average weight was 58.4 kg given the conditions of the problem. -/
theorem initial_average_weight (class_size : ℕ) (misread_weight : ℝ) (correct_weight : ℝ) (correct_average : ℝ) :
  class_size = 20 →
  misread_weight = 56 →
  correct_weight = 65 →
  correct_average = 58.85 →
  ∃ (initial_average : ℝ),
    initial_average * class_size + (correct_weight - misread_weight) = correct_average * class_size ∧
    initial_average = 58.4 := by
  sorry

end NUMINAMATH_CALUDE_initial_average_weight_l1505_150554


namespace NUMINAMATH_CALUDE_probability_two_white_and_one_white_one_red_l1505_150568

/-- Represents the color of a ball -/
inductive Color
  | White
  | Red

/-- Represents a bag of balls -/
structure Bag :=
  (total : Nat)
  (white : Nat)
  (red : Nat)
  (h_total : total = white + red)

/-- Calculates the probability of drawing two balls of a specific color combination -/
def probability_draw_two (bag : Bag) (first second : Color) : Rat :=
  sorry

theorem probability_two_white_and_one_white_one_red 
  (bag : Bag)
  (h_total : bag.total = 6)
  (h_white : bag.white = 4)
  (h_red : bag.red = 2) :
  (probability_draw_two bag Color.White Color.White = 2/5) ∧
  (probability_draw_two bag Color.White Color.Red = 8/15) :=
sorry

end NUMINAMATH_CALUDE_probability_two_white_and_one_white_one_red_l1505_150568


namespace NUMINAMATH_CALUDE_solution_when_a_is_3_root_of_multiplicity_l1505_150545

-- Define the equation
def equation (a x : ℝ) : Prop :=
  (a * x + 1) / (x - 1) - 2 / (1 - x) = 1

-- Part 1: Prove that when a = 3, the solution is x = -2
theorem solution_when_a_is_3 :
  ∃ x : ℝ, x ≠ 1 ∧ equation 3 x ∧ x = -2 :=
sorry

-- Part 2: Prove that the equation has a root of multiplicity when a = -3
theorem root_of_multiplicity :
  ∃ x : ℝ, x = 1 ∧ equation (-3) x :=
sorry

end NUMINAMATH_CALUDE_solution_when_a_is_3_root_of_multiplicity_l1505_150545


namespace NUMINAMATH_CALUDE_wire_cutting_l1505_150553

/-- Given a wire cut into two pieces, where the shorter piece is 2/5th of the longer piece
    and is 17.14285714285714 cm long, prove that the total length of the wire before cutting is 60 cm. -/
theorem wire_cutting (shorter_piece : ℝ) (longer_piece : ℝ) :
  shorter_piece = 17.14285714285714 →
  shorter_piece = (2 / 5) * longer_piece →
  shorter_piece + longer_piece = 60 := by
  sorry

end NUMINAMATH_CALUDE_wire_cutting_l1505_150553


namespace NUMINAMATH_CALUDE_nail_boxes_theorem_l1505_150533

theorem nail_boxes_theorem : ∃ (a b c d : ℕ), 24 * a + 23 * b + 17 * c + 16 * d = 100 := by
  sorry

end NUMINAMATH_CALUDE_nail_boxes_theorem_l1505_150533


namespace NUMINAMATH_CALUDE_water_volume_is_16_l1505_150552

/-- Represents a cubical water tank -/
structure CubicalTank where
  side_length : ℝ
  water_level : ℝ
  capacity_ratio : ℝ

/-- Calculates the volume of water in a cubical tank -/
def water_volume (tank : CubicalTank) : ℝ :=
  tank.water_level * tank.side_length * tank.side_length

/-- Theorem: The volume of water in the specified cubical tank is 16 cubic feet -/
theorem water_volume_is_16 (tank : CubicalTank) 
  (h1 : tank.water_level = 1)
  (h2 : tank.capacity_ratio = 0.25)
  (h3 : tank.water_level = tank.capacity_ratio * tank.side_length) :
  water_volume tank = 16 := by
  sorry

end NUMINAMATH_CALUDE_water_volume_is_16_l1505_150552


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1505_150501

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 2, 4}

theorem complement_of_M_in_U : 
  (U \ M) = {3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1505_150501


namespace NUMINAMATH_CALUDE_contrapositive_proof_l1505_150515

theorem contrapositive_proof (a : ℝ) : 
  a < 1 → ∀ x : ℝ, x^2 + (2*a+1)*x + a^2 + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_proof_l1505_150515


namespace NUMINAMATH_CALUDE_acute_angle_range_l1505_150524

def a (x : ℝ) : ℝ × ℝ := (1, x)
def b (x : ℝ) : ℝ × ℝ := (2*x + 3, -x)

def acute_angle_obtainable (x : ℝ) : Prop :=
  let dot_product := (a x).1 * (b x).1 + (a x).2 * (b x).2
  dot_product > 0 ∧ dot_product < (Real.sqrt ((a x).1^2 + (a x).2^2) * Real.sqrt ((b x).1^2 + (b x).2^2))

theorem acute_angle_range :
  ∀ x : ℝ, acute_angle_obtainable x ↔ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 3) :=
by sorry

end NUMINAMATH_CALUDE_acute_angle_range_l1505_150524


namespace NUMINAMATH_CALUDE_max_concentration_time_l1505_150535

def drug_concentration_peak_time : ℝ := 0.65
def time_uncertainty : ℝ := 0.15

theorem max_concentration_time :
  drug_concentration_peak_time + time_uncertainty = 0.8 := by sorry

end NUMINAMATH_CALUDE_max_concentration_time_l1505_150535


namespace NUMINAMATH_CALUDE_rhombus_adjacent_sides_equal_but_not_all_parallelograms_l1505_150538

-- Define a parallelogram
class Parallelogram :=
  (sides : Fin 4 → ℝ)
  (opposite_sides_equal : sides 0 = sides 2 ∧ sides 1 = sides 3)

-- Define a rhombus as a special case of parallelogram
class Rhombus extends Parallelogram :=
  (all_sides_equal : ∀ i j : Fin 4, sides i = sides j)

-- Theorem statement
theorem rhombus_adjacent_sides_equal_but_not_all_parallelograms 
  (r : Rhombus) (p : Parallelogram) : 
  (∀ i : Fin 4, r.sides i = r.sides ((i + 1) % 4)) ∧ 
  ¬(∀ (p : Parallelogram), ∀ i : Fin 4, p.sides i = p.sides ((i + 1) % 4)) :=
sorry

end NUMINAMATH_CALUDE_rhombus_adjacent_sides_equal_but_not_all_parallelograms_l1505_150538


namespace NUMINAMATH_CALUDE_expression_equality_l1505_150537

theorem expression_equality : 2 * Real.sqrt 3 * (3/2)^(1/3) * 12^(1/6) = 6 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l1505_150537


namespace NUMINAMATH_CALUDE_sixth_episode_length_is_115_l1505_150523

/-- The length of the sixth episode in a series of six episodes -/
def sixth_episode_length (ep1 ep2 ep3 ep4 ep5 total : ℕ) : ℕ :=
  total - (ep1 + ep2 + ep3 + ep4 + ep5)

/-- Theorem stating the length of the sixth episode -/
theorem sixth_episode_length_is_115 :
  sixth_episode_length 58 62 65 71 79 450 = 115 := by
  sorry

end NUMINAMATH_CALUDE_sixth_episode_length_is_115_l1505_150523


namespace NUMINAMATH_CALUDE_ac_operating_time_l1505_150571

/-- Calculates the average operating time for air conditioners given the total number,
    maximum simultaneous operation, and total time period. -/
def avgOperatingTime (totalAC : ℕ) (maxSimultaneous : ℕ) (totalHours : ℕ) : ℚ :=
  (maxSimultaneous * totalHours : ℚ) / totalAC

/-- Theorem stating that for 6 air conditioners with a maximum of 5 operating simultaneously
    over 24 hours, the average operating time is 20 hours. -/
theorem ac_operating_time :
  avgOperatingTime 6 5 24 = 20 := by
  sorry

#eval avgOperatingTime 6 5 24

end NUMINAMATH_CALUDE_ac_operating_time_l1505_150571


namespace NUMINAMATH_CALUDE_sum_of_integers_l1505_150594

theorem sum_of_integers (x y : ℕ+) (h1 : x^2 + y^2 = 289) (h2 : x * y = 120) : x + y = 23 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1505_150594


namespace NUMINAMATH_CALUDE_unique_score_determination_l1505_150548

/-- Scoring system function -/
def score (c w : ℕ) : ℕ := 30 + 4 * c - w

/-- Proposition: There exists a unique combination of c and w such that the score is 92,
    and this is the only score above 90 that allows for a unique determination of c and w -/
theorem unique_score_determination :
  ∃! (c w : ℕ), score c w = 92 ∧
  (∀ (c' w' : ℕ), score c' w' > 90 ∧ score c' w' ≠ 92 → 
    ∃ (c'' w'' : ℕ), c'' ≠ c' ∧ w'' ≠ w' ∧ score c'' w'' = score c' w') :=
sorry

end NUMINAMATH_CALUDE_unique_score_determination_l1505_150548


namespace NUMINAMATH_CALUDE_complex_number_theorem_l1505_150529

theorem complex_number_theorem (m : ℝ) : 
  let z : ℂ := m + (m^2 - 1) * Complex.I
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l1505_150529


namespace NUMINAMATH_CALUDE_sequence_condition_l1505_150536

/-- A sequence is monotonically increasing if each term is greater than the previous one. -/
def MonotonicallyIncreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

/-- The general term of the sequence a_n = n^2 + bn -/
def a (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem sequence_condition (b : ℝ) :
  MonotonicallyIncreasing (a · b) → b > -3 := by
  sorry

end NUMINAMATH_CALUDE_sequence_condition_l1505_150536


namespace NUMINAMATH_CALUDE_same_terminal_side_l1505_150560

theorem same_terminal_side : ∀ θ : Real,
  θ ≥ 0 ∧ θ < 2 * Real.pi →
  (θ = 2 * Real.pi / 3) ↔ ∃ k : Int, θ = -4 * Real.pi / 3 + 2 * Real.pi * k := by
  sorry

end NUMINAMATH_CALUDE_same_terminal_side_l1505_150560


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2985_l1505_150588

theorem smallest_prime_factor_of_2985 : ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2985 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2985 → p ≤ q :=
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2985_l1505_150588


namespace NUMINAMATH_CALUDE_wall_volume_calculation_l1505_150525

/-- Proves that the volume of a wall is 345 cubic meters given specific brick dimensions and quantity --/
theorem wall_volume_calculation (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) 
  (brick_count : ℕ) (h1 : brick_length = 20) (h2 : brick_width = 10) (h3 : brick_height = 7.5) 
  (h4 : brick_count = 23000) : 
  (brick_length * brick_width * brick_height * brick_count) / 1000000 = 345 := by
  sorry

end NUMINAMATH_CALUDE_wall_volume_calculation_l1505_150525


namespace NUMINAMATH_CALUDE_function_properties_l1505_150506

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (log x) / x - k / x

theorem function_properties (k : ℝ) :
  (∀ x ≥ 1, x^2 * f k x + 1 / (x + 1) ≥ 0) →
  (∀ x ≥ 1, k ≥ 1/2 * x^2 + (exp 2 - 2) * x - exp x - 7) →
  (∀ x > 0, deriv (f k) x = (1 - log x + k) / x^2) →
  (deriv (f k) 1 = 10) →
  (∃ x_max > 0, ∀ x > 0, f k x ≤ f k x_max ∧ f k x_max = 1 / (exp 10)) ∧
  (exp 2 - 9 ≤ k ∧ k ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_function_properties_l1505_150506


namespace NUMINAMATH_CALUDE_lucy_fish_count_l1505_150570

/-- The number of fish Lucy needs to buy -/
def fish_to_buy : ℕ := 68

/-- The total number of fish Lucy wants to have -/
def total_fish : ℕ := 280

/-- The number of fish Lucy currently has -/
def current_fish : ℕ := total_fish - fish_to_buy

theorem lucy_fish_count : current_fish = 212 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l1505_150570


namespace NUMINAMATH_CALUDE_proportion_solution_l1505_150516

theorem proportion_solution (x : ℝ) : (0.25 / x = 2 / 6) → x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l1505_150516


namespace NUMINAMATH_CALUDE_max_value_implies_m_l1505_150583

-- Define the variables
variable (x y m : ℝ)

-- Define the function z
def z (x y : ℝ) : ℝ := x - 3 * y

-- State the theorem
theorem max_value_implies_m (h1 : y ≥ x) (h2 : x + 3 * y ≤ 4) (h3 : x ≥ m)
  (h4 : ∀ x' y', y' ≥ x' → x' + 3 * y' ≤ 4 → x' ≥ m → z x' y' ≤ 8) 
  (h5 : ∃ x' y', y' ≥ x' ∧ x' + 3 * y' ≤ 4 ∧ x' ≥ m ∧ z x' y' = 8) : m = -4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l1505_150583


namespace NUMINAMATH_CALUDE_specific_pyramid_volume_l1505_150507

/-- Represents a pyramid with a square base and specified face areas -/
structure Pyramid where
  base_area : ℝ
  face_area1 : ℝ
  face_area2 : ℝ

/-- Calculates the volume of a pyramid given its properties -/
noncomputable def pyramid_volume (p : Pyramid) : ℝ :=
  let base_side := Real.sqrt p.base_area
  let height1 := 2 * p.face_area1 / base_side
  let height2 := 2 * p.face_area2 / base_side
  let a := (height1^2 - height2^2 + base_side^2) / (2 * base_side)
  let h := Real.sqrt (height1^2 - (base_side - a)^2)
  (1/3) * p.base_area * h

/-- The theorem stating the volume of the specific pyramid -/
theorem specific_pyramid_volume :
  let p := Pyramid.mk 256 128 112
  ∃ ε > 0, |pyramid_volume p - 1230.83| < ε :=
sorry

end NUMINAMATH_CALUDE_specific_pyramid_volume_l1505_150507


namespace NUMINAMATH_CALUDE_area_of_triangle_l1505_150599

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

-- Define the foci
def left_focus : ℝ × ℝ := (-5, 0)
def right_focus : ℝ × ℝ := (5, 0)

-- Define a point on the hyperbola
variable (P : ℝ × ℝ)

-- State that P is on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Define the right angle condition
def right_angle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0

-- State that F₁PF₂ forms a right angle
axiom right_angle_condition : right_angle left_focus P right_focus

-- The theorem to prove
theorem area_of_triangle : 
  ∃ (S : ℝ), S = 16 ∧ S = (1/2) * ‖P - left_focus‖ * ‖P - right_focus‖ :=
sorry

end NUMINAMATH_CALUDE_area_of_triangle_l1505_150599


namespace NUMINAMATH_CALUDE_not_all_perfect_squares_l1505_150559

theorem not_all_perfect_squares (a b c : ℕ+) : 
  ¬(∃ (x y z : ℕ), (a^2 + b + c : ℕ) = x^2 ∧ (b^2 + c + a : ℕ) = y^2 ∧ (c^2 + a + b : ℕ) = z^2) :=
sorry

end NUMINAMATH_CALUDE_not_all_perfect_squares_l1505_150559


namespace NUMINAMATH_CALUDE_max_value_fraction_l1505_150510

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2*x + y)) + (y / (x + 2*y)) ≤ 2/3 ∧ 
  ((x / (2*x + y)) + (y / (x + 2*y)) = 2/3 ↔ 2*x + y = x + 2*y) :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l1505_150510


namespace NUMINAMATH_CALUDE_min_value_theorem_l1505_150555

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  2 * x^2 + 24 * x + 128 / x^3 ≥ 168 ∧
  (2 * x^2 + 24 * x + 128 / x^3 = 168 ↔ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1505_150555


namespace NUMINAMATH_CALUDE_local_maximum_at_e_l1505_150542

open Real

-- Define the function f(x) = ln(x) / x
noncomputable def f (x : ℝ) : ℝ := (log x) / x

-- State the theorem
theorem local_maximum_at_e :
  ∃ δ > 0, ∀ x ∈ Set.Ioo (Real.exp 1 - δ) (Real.exp 1 + δ),
    x ≠ Real.exp 1 → f x < f (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_local_maximum_at_e_l1505_150542


namespace NUMINAMATH_CALUDE_tims_sleep_hours_l1505_150551

/-- Proves that Tim slept 6 hours each day for the first 2 days given the conditions -/
theorem tims_sleep_hours (x : ℝ) : 
  (2 * x + 2 * 10 = 32) → x = 6 := by sorry

end NUMINAMATH_CALUDE_tims_sleep_hours_l1505_150551


namespace NUMINAMATH_CALUDE_range_of_m_value_of_m_l1505_150540

-- Define the quadratic equation
def quadratic (m : ℝ) (x : ℝ) : ℝ := x^2 + (2*m - 1)*x + m^2

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := (2*m - 1)^2 - 4*1*m^2

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∃ α β : ℝ, α ≠ β ∧ quadratic m α = 0 ∧ quadratic m β = 0) → m < 1/4 :=
sorry

-- Theorem for the value of m when α² + β² = 1
theorem value_of_m (m : ℝ) (α β : ℝ) :
  (α ≠ β ∧ quadratic m α = 0 ∧ quadratic m β = 0 ∧ α^2 + β^2 = 1) → m = 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_value_of_m_l1505_150540
