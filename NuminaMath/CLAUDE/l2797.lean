import Mathlib

namespace NUMINAMATH_CALUDE_max_six_yuan_items_proof_l2797_279791

/-- The maximum number of 6-yuan items that can be bought given the conditions -/
def max_six_yuan_items : ℕ := 7

theorem max_six_yuan_items_proof :
  ∀ (x y z : ℕ),
    6 * x + 4 * y + 2 * z = 60 →
    x + y + z = 16 →
    x ≤ max_six_yuan_items :=
by
  sorry

#check max_six_yuan_items_proof

end NUMINAMATH_CALUDE_max_six_yuan_items_proof_l2797_279791


namespace NUMINAMATH_CALUDE_gcf_of_lcms_l2797_279770

theorem gcf_of_lcms : Nat.gcd (Nat.lcm 9 21) (Nat.lcm 10 22) = 1 := by sorry

end NUMINAMATH_CALUDE_gcf_of_lcms_l2797_279770


namespace NUMINAMATH_CALUDE_vector_calculation_l2797_279767

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem vector_calculation (a b : V) :
  (1 / 3 : ℝ) • (a - 2 • b) + b = (1 / 3 : ℝ) • a + (1 / 3 : ℝ) • b :=
by sorry

end NUMINAMATH_CALUDE_vector_calculation_l2797_279767


namespace NUMINAMATH_CALUDE_button_probability_l2797_279707

/-- Represents a jar containing buttons -/
structure Jar :=
  (red : ℕ)
  (blue : ℕ)

/-- Calculate the probability of choosing a blue button from a jar -/
def blueProbability (jar : Jar) : ℚ :=
  jar.blue / (jar.red + jar.blue)

theorem button_probability (jarA jarB : Jar) : 
  jarA.red = 6 ∧ 
  jarA.blue = 10 ∧ 
  jarB.red = 3 ∧ 
  jarB.blue = 5 ∧ 
  (jarA.red + jarA.blue : ℚ) = 2/3 * (6 + 10) →
  blueProbability jarA * blueProbability jarB = 25/64 := by
  sorry

#check button_probability

end NUMINAMATH_CALUDE_button_probability_l2797_279707


namespace NUMINAMATH_CALUDE_regular_hexagon_dimensions_l2797_279742

/-- Regular hexagon with given area and side lengths -/
structure RegularHexagon where
  area : ℝ
  x : ℝ
  y : ℝ
  area_eq : area = 54 * Real.sqrt 3
  side_length : x > 0
  diagonal_length : y > 0

/-- Theorem: For a regular hexagon with area 54√3 cm², if AB = x cm and AC = y√3 cm, then x = 6 and y = 6 -/
theorem regular_hexagon_dimensions (h : RegularHexagon) : h.x = 6 ∧ h.y = 6 := by
  sorry

end NUMINAMATH_CALUDE_regular_hexagon_dimensions_l2797_279742


namespace NUMINAMATH_CALUDE_evaluate_expression_l2797_279725

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 5
  (2 * x - a + 4) = a + 14 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2797_279725


namespace NUMINAMATH_CALUDE_attendance_scientific_notation_l2797_279772

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  mantissa : ℝ
  exponent : ℤ
  mantissa_range : 1 ≤ mantissa ∧ mantissa < 10

/-- Convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem attendance_scientific_notation :
  toScientificNotation 204000 = ScientificNotation.mk 2.04 5 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_attendance_scientific_notation_l2797_279772


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l2797_279797

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : Real.sin α < 0) (h2 : Real.tan α < 0) : 
  ∃ (x y : Real), x > 0 ∧ y < 0 ∧ Real.cos α = x ∧ Real.sin α = y :=
sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l2797_279797


namespace NUMINAMATH_CALUDE_gcd_54000_36000_l2797_279794

theorem gcd_54000_36000 : Nat.gcd 54000 36000 = 18000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_54000_36000_l2797_279794


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2797_279701

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l2797_279701


namespace NUMINAMATH_CALUDE_solve_bowtie_equation_l2797_279774

-- Define the operation ⊗
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem solve_bowtie_equation (y : ℝ) : bowtie 5 y = 15 → y = 90 := by
  sorry

end NUMINAMATH_CALUDE_solve_bowtie_equation_l2797_279774


namespace NUMINAMATH_CALUDE_jellybean_problem_l2797_279793

/-- The number of jellybeans initially in the jar -/
def initial_jellybeans : ℕ := 90

/-- The number of jellybeans Samantha took -/
def samantha_took : ℕ := 24

/-- The number of jellybeans Shelby ate -/
def shelby_ate : ℕ := 12

/-- The final number of jellybeans in the jar -/
def final_jellybeans : ℕ := 72

theorem jellybean_problem :
  initial_jellybeans - samantha_took - shelby_ate +
  ((samantha_took + shelby_ate) / 2) = final_jellybeans :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l2797_279793


namespace NUMINAMATH_CALUDE_fraction_inequality_l2797_279726

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < 0) :
  a / (a - c) > b / (b - c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l2797_279726


namespace NUMINAMATH_CALUDE_simplify_expressions_l2797_279766

variable (x y : ℝ)

theorem simplify_expressions :
  (3 * x^2 - 2*x*y + y^2 - 3*x^2 + 3*x*y = x*y + y^2) ∧
  ((7*x^2 - 3*x*y) - 6*(x^2 - 1/3*x*y) = x^2 - x*y) := by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2797_279766


namespace NUMINAMATH_CALUDE_cylinder_height_l2797_279762

/-- The height of a cylindrical tin given its diameter and volume -/
theorem cylinder_height (diameter : ℝ) (volume : ℝ) (h_diameter : diameter = 14) (h_volume : volume = 245) :
  (volume / (π * (diameter / 2)^2)) = 245 / (49 * π) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_l2797_279762


namespace NUMINAMATH_CALUDE_kitchen_planks_l2797_279765

/-- Represents the number of wooden planks used in Andrew's house flooring project. -/
structure FlooringProject where
  bedroom : ℕ
  livingRoom : ℕ
  guestBedroom : ℕ
  hallway : ℕ
  kitchen : ℕ
  leftover : ℕ
  replacedBedroom : ℕ
  replacedGuestBedroom : ℕ

/-- Theorem stating the number of planks used for the kitchen in Andrew's flooring project. -/
theorem kitchen_planks (project : FlooringProject) 
    (h1 : project.bedroom = 8)
    (h2 : project.livingRoom = 20)
    (h3 : project.guestBedroom = project.bedroom - 2)
    (h4 : project.hallway = 4 * 2)
    (h5 : project.leftover = 6)
    (h6 : project.replacedBedroom = 3)
    (h7 : project.replacedGuestBedroom = 3)
    : project.kitchen = 6 := by
  sorry


end NUMINAMATH_CALUDE_kitchen_planks_l2797_279765


namespace NUMINAMATH_CALUDE_order_of_fractions_l2797_279790

theorem order_of_fractions (x y : ℝ) (hx : x < 0) (hy : 0 < y ∧ y < 1) :
  (1 / x) < (y / x) ∧ (y / x) < (y^2 / x) := by
  sorry

end NUMINAMATH_CALUDE_order_of_fractions_l2797_279790


namespace NUMINAMATH_CALUDE_fencing_cost_is_105_rupees_l2797_279787

/-- Represents a rectangular field -/
structure RectangularField where
  length : ℝ
  width : ℝ
  area : ℝ
  ratio : ℝ × ℝ

/-- Calculates the cost of fencing a rectangular field -/
def fencingCost (field : RectangularField) (costPerMeter : ℝ) : ℝ :=
  2 * (field.length + field.width) * costPerMeter

/-- Theorem: The cost of fencing a specific rectangular field is 105 rupees -/
theorem fencing_cost_is_105_rupees : 
  ∀ (field : RectangularField),
    field.ratio = (3, 4) →
    field.area = 10800 →
    fencingCost field 0.25 = 105 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_is_105_rupees_l2797_279787


namespace NUMINAMATH_CALUDE_davids_math_marks_l2797_279746

theorem davids_math_marks (english : ℕ) (physics : ℕ) (chemistry : ℕ) (biology : ℕ) (average : ℕ) (total_subjects : ℕ) :
  english = 96 →
  physics = 82 →
  chemistry = 97 →
  biology = 95 →
  average = 93 →
  total_subjects = 5 →
  (english + physics + chemistry + biology + (average * total_subjects - (english + physics + chemistry + biology))) / total_subjects = average :=
by sorry

end NUMINAMATH_CALUDE_davids_math_marks_l2797_279746


namespace NUMINAMATH_CALUDE_square_of_binomial_with_sqrt_l2797_279732

theorem square_of_binomial_with_sqrt : 36^2 + 2 * 36 * Real.sqrt 49 + (Real.sqrt 49)^2 = 1849 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_with_sqrt_l2797_279732


namespace NUMINAMATH_CALUDE_survey_total_students_l2797_279775

theorem survey_total_students : 
  let mac_preference : ℕ := 60
  let both_preference : ℕ := mac_preference / 3
  let no_preference : ℕ := 90
  let windows_preference : ℕ := 40
  mac_preference + both_preference + no_preference + windows_preference = 210 := by
sorry

end NUMINAMATH_CALUDE_survey_total_students_l2797_279775


namespace NUMINAMATH_CALUDE_systematic_sampling_60_5_l2797_279736

/-- Systematic sampling function that returns a list of sample numbers -/
def systematicSample (totalPopulation : ℕ) (sampleSize : ℕ) : List ℕ :=
  let interval := totalPopulation / sampleSize
  List.range sampleSize |>.map (fun i => i * interval + interval)

/-- Theorem: The systematic sampling of 5 students from a class of 60 yields [6, 18, 30, 42, 54] -/
theorem systematic_sampling_60_5 :
  systematicSample 60 5 = [6, 18, 30, 42, 54] := by
  sorry

#eval systematicSample 60 5

end NUMINAMATH_CALUDE_systematic_sampling_60_5_l2797_279736


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l2797_279704

/-- A quadratic function with vertex (-2, 5) passing through (0, 3) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_coefficients :
  ∃ (a b c : ℝ),
    (∀ x, f a b c x = a * x^2 + b * x + c) ∧
    (f a b c (-2) = 5) ∧
    (∀ x, f a b c (x) = f a b c (-x - 4)) ∧
    (f a b c 0 = 3) ∧
    (a = -1/2 ∧ b = -2 ∧ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l2797_279704


namespace NUMINAMATH_CALUDE_outfit_combinations_l2797_279702

theorem outfit_combinations (n : ℕ) (h : n = 7) : n^3 - n = 336 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l2797_279702


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l2797_279731

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | ∃ y, y = Real.log (1 - x^2)}
def B : Set ℝ := {y : ℝ | ∃ x, y = (4 : ℝ)^(x - 2)}

-- State the theorem
theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = Set.Ioc (-1) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l2797_279731


namespace NUMINAMATH_CALUDE_bike_riders_count_l2797_279700

theorem bike_riders_count (total : ℕ) (hikers : ℕ) (bikers : ℕ) :
  total = hikers + bikers →
  hikers = bikers + 178 →
  total = 676 →
  bikers = 249 := by
sorry

end NUMINAMATH_CALUDE_bike_riders_count_l2797_279700


namespace NUMINAMATH_CALUDE_snow_probability_l2797_279785

theorem snow_probability (p : ℝ) (h : p = 2/3) :
  1 - (1 - p)^3 = 26/27 := by sorry

end NUMINAMATH_CALUDE_snow_probability_l2797_279785


namespace NUMINAMATH_CALUDE_square_58_sexagesimal_l2797_279706

/-- Represents a number in sexagesimal form a•b, where the value is a*60 + b -/
structure Sexagesimal where
  a : ℕ
  b : ℕ
  h : b < 60

/-- Converts a natural number to its sexagesimal representation -/
def to_sexagesimal (n : ℕ) : Sexagesimal :=
  ⟨n / 60, n % 60, sorry⟩

/-- The statement to be proved -/
theorem square_58_sexagesimal : 
  to_sexagesimal (58^2) = Sexagesimal.mk 56 4 sorry := by sorry

end NUMINAMATH_CALUDE_square_58_sexagesimal_l2797_279706


namespace NUMINAMATH_CALUDE_diary_ratio_proof_l2797_279799

theorem diary_ratio_proof (initial_diaries : ℕ) (final_diaries : ℕ) (bought_diaries : ℕ) :
  initial_diaries = 8 →
  final_diaries = 18 →
  final_diaries = (initial_diaries + bought_diaries) * 3 / 4 →
  bought_diaries / initial_diaries = 2 := by
  sorry

#check diary_ratio_proof

end NUMINAMATH_CALUDE_diary_ratio_proof_l2797_279799


namespace NUMINAMATH_CALUDE_tokyo_tech_1956_entrance_exam_l2797_279796

theorem tokyo_tech_1956_entrance_exam
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha1 : a < 1) (hb1 : b < 1) (hc1 : c < 1) :
  a + b + c - a * b * c < 2 :=
sorry

end NUMINAMATH_CALUDE_tokyo_tech_1956_entrance_exam_l2797_279796


namespace NUMINAMATH_CALUDE_triangle_inequality_l2797_279744

variable (A B C : ℝ) -- Angles of the triangle
variable (da db dc : ℝ) -- Distances from P to sides
variable (Ra Rb Rc : ℝ) -- Distances from P to vertices

-- Assume all variables are non-negative
variable (h1 : 0 ≤ A) (h2 : 0 ≤ B) (h3 : 0 ≤ C)
variable (h4 : 0 ≤ da) (h5 : 0 ≤ db) (h6 : 0 ≤ dc)
variable (h7 : 0 ≤ Ra) (h8 : 0 ≤ Rb) (h9 : 0 ≤ Rc)

-- Assume A, B, C form a valid triangle
variable (h10 : A + B + C = Real.pi)

theorem triangle_inequality (A B C da db dc Ra Rb Rc : ℝ)
  (h1 : 0 ≤ A) (h2 : 0 ≤ B) (h3 : 0 ≤ C)
  (h4 : 0 ≤ da) (h5 : 0 ≤ db) (h6 : 0 ≤ dc)
  (h7 : 0 ≤ Ra) (h8 : 0 ≤ Rb) (h9 : 0 ≤ Rc)
  (h10 : A + B + C = Real.pi) :
  3 * (da^2 + db^2 + dc^2) ≥ (Ra * Real.sin A)^2 + (Rb * Real.sin B)^2 + (Rc * Real.sin C)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2797_279744


namespace NUMINAMATH_CALUDE_almost_order_lineup_correct_almost_order_lineup_10_l2797_279723

/-- Represents the number of ways to line up n people in almost-order -/
def almost_order_lineup (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | 2 => 2
  | n + 3 => almost_order_lineup (n + 1) + almost_order_lineup (n + 2)

/-- The height difference between consecutive people -/
def height_diff : ℕ := 5

/-- The maximum allowed height difference for almost-order -/
def max_height_diff : ℕ := 8

/-- The height of the shortest person -/
def min_height : ℕ := 140

theorem almost_order_lineup_correct (n : ℕ) :
  (∀ i j, i < j → j ≤ n → min_height + i * height_diff ≤ min_height + j * height_diff + max_height_diff) →
  almost_order_lineup n = if n ≤ 2 then n else almost_order_lineup (n - 1) + almost_order_lineup (n - 2) :=
sorry

theorem almost_order_lineup_10 : almost_order_lineup 10 = 89 :=
sorry

end NUMINAMATH_CALUDE_almost_order_lineup_correct_almost_order_lineup_10_l2797_279723


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l2797_279718

theorem square_area_from_diagonal (d : ℝ) (h : d = 12) : 
  (d^2 / 2 : ℝ) = 72 := by
  sorry

#check square_area_from_diagonal

end NUMINAMATH_CALUDE_square_area_from_diagonal_l2797_279718


namespace NUMINAMATH_CALUDE_camel_cannot_end_adjacent_l2797_279720

/-- Represents a hexagonal board with side length m -/
structure HexBoard where
  m : ℕ

/-- The total number of fields on a hexagonal board -/
def HexBoard.total_fields (board : HexBoard) : ℕ :=
  3 * board.m^2 - 3 * board.m + 1

/-- The number of moves a camel makes on the board -/
def HexBoard.camel_moves (board : HexBoard) : ℕ :=
  board.total_fields - 1

/-- Theorem stating that a camel cannot end on an adjacent field to its starting position -/
theorem camel_cannot_end_adjacent (board : HexBoard) :
  ∃ (start finish : ℕ), start ≠ finish ∧ 
  finish ≠ (start + 1) ∧ finish ≠ (start - 1) ∧
  finish = (start + board.camel_moves) % board.total_fields :=
sorry

end NUMINAMATH_CALUDE_camel_cannot_end_adjacent_l2797_279720


namespace NUMINAMATH_CALUDE_number_ratio_l2797_279708

theorem number_ratio (x : ℚ) (h : 3 * (2 * x + 15) = 75) : x / (2 * x) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l2797_279708


namespace NUMINAMATH_CALUDE_complex_number_coordinates_i_times_one_minus_i_l2797_279709

theorem complex_number_coordinates : Complex → Complex → Prop :=
  fun z w => z = w

theorem i_times_one_minus_i (i : Complex) (h : i * i = -1) :
  complex_number_coordinates (i * (1 - i)) (1 + i) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_i_times_one_minus_i_l2797_279709


namespace NUMINAMATH_CALUDE_second_player_can_always_win_l2797_279733

/-- Represents a square on the game board -/
inductive Square
| Empty : Square
| S : Square
| O : Square

/-- Represents the game board -/
def Board := Vector Square 2000

/-- Represents a player in the game -/
inductive Player
| First : Player
| Second : Player

/-- Checks if the game is over (SOS pattern found) -/
def is_game_over (board : Board) : Prop := sorry

/-- Represents a valid move in the game -/
structure Move where
  position : Fin 2000
  symbol : Square

/-- Applies a move to the board -/
def apply_move (board : Board) (move : Move) : Board := sorry

/-- Represents the game state -/
structure GameState where
  board : Board
  current_player : Player

/-- Represents a strategy for a player -/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for a player -/
def is_winning_strategy (player : Player) (strategy : Strategy) : Prop := sorry

/-- The main theorem to prove -/
theorem second_player_can_always_win :
  ∃ (strategy : Strategy), is_winning_strategy Player.Second strategy := sorry

end NUMINAMATH_CALUDE_second_player_can_always_win_l2797_279733


namespace NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l2797_279781

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem cube_minus_cylinder_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ)
  (h1 : cube_side = 5)
  (h2 : cylinder_radius = 1)
  (h3 : cylinder_height = 5) :
  cube_side ^ 3 - π * cylinder_radius ^ 2 * cylinder_height = 125 - 5 * π := by
  sorry

#check cube_minus_cylinder_volume

end NUMINAMATH_CALUDE_cube_minus_cylinder_volume_l2797_279781


namespace NUMINAMATH_CALUDE_plane_line_parallel_l2797_279783

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)  -- Line is subset of plane
variable (parallel : Line → Line → Prop) -- Lines are parallel
variable (parallel_plane : Line → Plane → Prop) -- Line is parallel to plane
variable (intersect : Plane → Plane → Line → Prop) -- Planes intersect in a line

-- State the theorem
theorem plane_line_parallel 
  (α β : Plane) (m n : Line) 
  (h1 : intersect α β m) 
  (h2 : parallel n m) 
  (h3 : ¬ subset n α) 
  (h4 : ¬ subset n β) : 
  parallel_plane n α ∧ parallel_plane n β :=
sorry

end NUMINAMATH_CALUDE_plane_line_parallel_l2797_279783


namespace NUMINAMATH_CALUDE_unique_matching_number_l2797_279717

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_range : hundreds ≥ 1 ∧ hundreds ≤ 9
  t_range : tens ≥ 0 ∧ tens ≤ 9
  u_range : units ≥ 0 ∧ units ≤ 9

/-- Checks if two ThreeDigitNumbers match in exactly one digit place -/
def matchesOneDigit (a b : ThreeDigitNumber) : Prop :=
  (a.hundreds = b.hundreds ∧ a.tens ≠ b.tens ∧ a.units ≠ b.units) ∨
  (a.hundreds ≠ b.hundreds ∧ a.tens = b.tens ∧ a.units ≠ b.units) ∨
  (a.hundreds ≠ b.hundreds ∧ a.tens ≠ b.tens ∧ a.units = b.units)

/-- The theorem to be proved -/
theorem unique_matching_number : ∃! n : ThreeDigitNumber,
  matchesOneDigit n ⟨1, 0, 9, by sorry, by sorry, by sorry⟩ ∧
  matchesOneDigit n ⟨7, 0, 4, by sorry, by sorry, by sorry⟩ ∧
  matchesOneDigit n ⟨1, 2, 4, by sorry, by sorry, by sorry⟩ ∧
  n = ⟨7, 2, 9, by sorry, by sorry, by sorry⟩ :=
sorry

end NUMINAMATH_CALUDE_unique_matching_number_l2797_279717


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2797_279768

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = 2) : x^3 + y^3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2797_279768


namespace NUMINAMATH_CALUDE_optimal_bus_rental_plan_l2797_279715

/-- Represents a bus rental plan -/
structure BusRentalPlan where
  modelA : ℕ
  modelB : ℕ

/-- Calculates the total capacity of a bus rental plan -/
def totalCapacity (plan : BusRentalPlan) : ℕ :=
  40 * plan.modelA + 55 * plan.modelB

/-- Calculates the total cost of a bus rental plan -/
def totalCost (plan : BusRentalPlan) : ℕ :=
  600 * plan.modelA + 700 * plan.modelB

/-- Checks if a bus rental plan is valid -/
def isValidPlan (plan : BusRentalPlan) : Prop :=
  plan.modelA + plan.modelB = 10 ∧ 
  plan.modelA ≥ 1 ∧ 
  plan.modelB ≥ 1 ∧
  totalCapacity plan ≥ 502

/-- Theorem stating the properties of the optimal bus rental plan -/
theorem optimal_bus_rental_plan :
  ∃ (optimalPlan : BusRentalPlan),
    isValidPlan optimalPlan ∧
    optimalPlan.modelA = 3 ∧
    optimalPlan.modelB = 7 ∧
    totalCost optimalPlan = 6700 ∧
    (∀ (plan : BusRentalPlan), isValidPlan plan → totalCost plan ≥ totalCost optimalPlan) ∧
    (∀ (plan : BusRentalPlan), isValidPlan plan → plan.modelA ≤ 3) :=
  sorry


end NUMINAMATH_CALUDE_optimal_bus_rental_plan_l2797_279715


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2797_279779

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) ∧
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2797_279779


namespace NUMINAMATH_CALUDE_cost_780_candies_l2797_279760

/-- The cost of buying a given number of chocolate candies -/
def chocolateCost (candies : ℕ) : ℚ :=
  let boxSize := 30
  let boxCost := 8
  let discountThreshold := 500
  let discountRate := 0.1
  let boxes := (candies + boxSize - 1) / boxSize  -- Ceiling division
  let totalCost := boxes * boxCost
  if candies > discountThreshold then
    totalCost * (1 - discountRate)
  else
    totalCost

/-- Theorem: The cost of buying 780 chocolate candies is $187.2 -/
theorem cost_780_candies :
  chocolateCost 780 = 187.2 := by
  sorry

end NUMINAMATH_CALUDE_cost_780_candies_l2797_279760


namespace NUMINAMATH_CALUDE_sauce_per_burger_is_quarter_cup_l2797_279752

/-- The amount of barbecue sauce per burger -/
def sauce_per_burger (total_sauce : ℚ) (sauce_per_sandwich : ℚ) (num_sandwiches : ℕ) (num_burgers : ℕ) : ℚ :=
  (total_sauce - sauce_per_sandwich * num_sandwiches) / num_burgers

/-- Theorem stating that the amount of sauce per burger is 1/4 cup -/
theorem sauce_per_burger_is_quarter_cup :
  sauce_per_burger 5 (1/6) 18 8 = 1/4 := by sorry

end NUMINAMATH_CALUDE_sauce_per_burger_is_quarter_cup_l2797_279752


namespace NUMINAMATH_CALUDE_intercepted_arc_is_60_degrees_l2797_279755

/-- An equilateral triangle with a circle rolling along its side -/
structure RollingCircleTriangle where
  -- The side length of the equilateral triangle
  side : ℝ
  -- The radius of the rolling circle
  radius : ℝ
  -- The radius equals the height of the triangle
  height_eq_radius : radius = (side * Real.sqrt 3) / 2

/-- The angular measure of the arc intercepted on the circle by the sides of the triangle -/
def intercepted_arc_measure (t : RollingCircleTriangle) : ℝ := 
  -- Definition to be proved
  60

/-- Theorem: The angular measure of the arc intercepted on the circle 
    by the sides of the triangle is always 60° -/
theorem intercepted_arc_is_60_degrees (t : RollingCircleTriangle) : 
  intercepted_arc_measure t = 60 := by
  sorry

end NUMINAMATH_CALUDE_intercepted_arc_is_60_degrees_l2797_279755


namespace NUMINAMATH_CALUDE_stock_exchange_problem_l2797_279782

theorem stock_exchange_problem (h l : ℕ) : 
  h = l + l / 5 →  -- 20% more stocks closed higher
  h = 1080 →      -- 1080 stocks closed higher
  h + l = 1980    -- Total number of stocks
  := by sorry

end NUMINAMATH_CALUDE_stock_exchange_problem_l2797_279782


namespace NUMINAMATH_CALUDE_scientific_notation_of_nanometer_l2797_279776

def nanometer : ℝ := 0.000000001

theorem scientific_notation_of_nanometer :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ nanometer = a * (10 : ℝ) ^ n ∧ a = 1 ∧ n = -9 :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_nanometer_l2797_279776


namespace NUMINAMATH_CALUDE_unique_pairs_theorem_l2797_279788

theorem unique_pairs_theorem (x y : ℕ) : 
  x ≥ 2 → y ≥ 2 → 
  (3 * x) % y = 1 → 
  (3 * y) % x = 1 → 
  (x * y) % 3 = 1 → 
  ((x = 2 ∧ y = 5) ∨ (x = 5 ∧ y = 2)) := by
  sorry

#check unique_pairs_theorem

end NUMINAMATH_CALUDE_unique_pairs_theorem_l2797_279788


namespace NUMINAMATH_CALUDE_atomic_weight_sodium_l2797_279786

/-- The atomic weight of chlorine in atomic mass units (amu) -/
def atomic_weight_chlorine : ℝ := 35.45

/-- The atomic weight of oxygen in atomic mass units (amu) -/
def atomic_weight_oxygen : ℝ := 16.00

/-- The molecular weight of the compound in atomic mass units (amu) -/
def molecular_weight_compound : ℝ := 74.00

/-- Theorem stating that the atomic weight of sodium is 22.55 amu -/
theorem atomic_weight_sodium :
  molecular_weight_compound = atomic_weight_chlorine + atomic_weight_oxygen + 22.55 := by
  sorry

end NUMINAMATH_CALUDE_atomic_weight_sodium_l2797_279786


namespace NUMINAMATH_CALUDE_count_numbers_with_4_or_6_eq_1105_l2797_279789

-- Define the range of numbers we're considering
def range_end : Nat := 2401

-- Define a function to check if a number in base 8 contains 4 or 6
def contains_4_or_6 (n : Nat) : Bool :=
  sorry

-- Define the count of numbers containing 4 or 6
def count_numbers_with_4_or_6 : Nat :=
  (List.range range_end).filter contains_4_or_6 |>.length

-- Theorem to prove
theorem count_numbers_with_4_or_6_eq_1105 :
  count_numbers_with_4_or_6 = 1105 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_4_or_6_eq_1105_l2797_279789


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l2797_279703

theorem smallest_integer_with_given_remainders : ∃ n : ℕ,
  n > 0 ∧
  n % 3 = 2 ∧
  n % 5 = 4 ∧
  n % 7 = 6 ∧
  n % 11 = 10 ∧
  ∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 5 = 4 ∧ m % 7 = 6 ∧ m % 11 = 10 → n ≤ m :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l2797_279703


namespace NUMINAMATH_CALUDE_not_prime_257_pow_1092_plus_1092_l2797_279741

theorem not_prime_257_pow_1092_plus_1092 : ¬ Nat.Prime (257^1092 + 1092) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_257_pow_1092_plus_1092_l2797_279741


namespace NUMINAMATH_CALUDE_sum_m_n_in_interval_l2797_279780

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Define the theorem
theorem sum_m_n_in_interval (m n : ℝ) :
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (-5) 4) →
  (∀ y ∈ Set.Icc (-5) 4, ∃ x ∈ Set.Icc m n, f x = y) →
  m + n ∈ Set.Icc 1 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_m_n_in_interval_l2797_279780


namespace NUMINAMATH_CALUDE_A_older_than_B_by_two_l2797_279721

-- Define the ages of A, B, and C
def B : ℕ := 14
def C : ℕ := B / 2
def A : ℕ := 37 - B - C

-- Theorem statement
theorem A_older_than_B_by_two : A = B + 2 := by
  sorry

end NUMINAMATH_CALUDE_A_older_than_B_by_two_l2797_279721


namespace NUMINAMATH_CALUDE_octagon_pebble_arrangements_l2797_279719

/-- The number of symmetries (rotations and reflections) of a regular octagon -/
def octagon_symmetries : ℕ := 16

/-- The number of vertices in a regular octagon -/
def octagon_vertices : ℕ := 8

/-- The number of distinct arrangements of pebbles on a regular octagon -/
def distinct_arrangements : ℕ := Nat.factorial octagon_vertices / octagon_symmetries

theorem octagon_pebble_arrangements :
  distinct_arrangements = 2520 := by sorry

end NUMINAMATH_CALUDE_octagon_pebble_arrangements_l2797_279719


namespace NUMINAMATH_CALUDE_trapezoid_area_between_triangles_l2797_279722

/-- Given two equilateral triangles, one inside the other, this theorem calculates
    the area of one of the three congruent trapezoids formed between them. -/
theorem trapezoid_area_between_triangles
  (outer_area : ℝ)
  (inner_area : ℝ)
  (h_outer : outer_area = 36)
  (h_inner : inner_area = 4)
  (h_positive : 0 < inner_area ∧ inner_area < outer_area) :
  (outer_area - inner_area) / 3 = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_between_triangles_l2797_279722


namespace NUMINAMATH_CALUDE_total_balls_is_seven_l2797_279759

/-- The number of balls in the first box -/
def box1_balls : ℕ := 3

/-- The number of balls in the second box -/
def box2_balls : ℕ := 4

/-- The total number of balls in both boxes -/
def total_balls : ℕ := box1_balls + box2_balls

/-- Theorem stating that the total number of balls is 7 -/
theorem total_balls_is_seven : total_balls = 7 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_is_seven_l2797_279759


namespace NUMINAMATH_CALUDE_square_difference_l2797_279735

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 12) : 
  (x - y)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2797_279735


namespace NUMINAMATH_CALUDE_f_min_value_l2797_279713

/-- The function f to be minimized -/
def f (x y z : ℝ) : ℝ :=
  x^2 + 2*y^2 + 3*z^2 + 2*x*y + 4*y*z + 2*z*x - 6*x - 10*y - 12*z

/-- Theorem stating that -14 is the minimum value of f -/
theorem f_min_value :
  ∀ x y z : ℝ, f x y z ≥ -14 :=
by sorry

end NUMINAMATH_CALUDE_f_min_value_l2797_279713


namespace NUMINAMATH_CALUDE_favorite_fruit_strawberries_l2797_279795

theorem favorite_fruit_strawberries (total : ℕ) (oranges pears apples bananas grapes : ℕ)
  (h_total : total = 900)
  (h_oranges : oranges = 130)
  (h_pears : pears = 210)
  (h_apples : apples = 275)
  (h_bananas : bananas = 93)
  (h_grapes : grapes = 119) :
  total - (oranges + pears + apples + bananas + grapes) = 73 := by
  sorry

end NUMINAMATH_CALUDE_favorite_fruit_strawberries_l2797_279795


namespace NUMINAMATH_CALUDE_two_different_buttons_l2797_279728

/-- Represents the size of a button -/
inductive Size
| Big
| Small

/-- Represents the color of a button -/
inductive Color
| White
| Black

/-- Represents a button with a size and color -/
structure Button :=
  (size : Size)
  (color : Color)

/-- A set of buttons satisfying the given conditions -/
structure ButtonSet :=
  (buttons : Set Button)
  (has_big : ∃ b ∈ buttons, b.size = Size.Big)
  (has_small : ∃ b ∈ buttons, b.size = Size.Small)
  (has_white : ∃ b ∈ buttons, b.color = Color.White)
  (has_black : ∃ b ∈ buttons, b.color = Color.Black)

/-- Theorem stating that there exist two buttons with different size and color -/
theorem two_different_buttons (bs : ButtonSet) :
  ∃ (b1 b2 : Button), b1 ∈ bs.buttons ∧ b2 ∈ bs.buttons ∧
  b1.size ≠ b2.size ∧ b1.color ≠ b2.color :=
sorry

end NUMINAMATH_CALUDE_two_different_buttons_l2797_279728


namespace NUMINAMATH_CALUDE_tony_weightlifting_ratio_l2797_279711

/-- Given Tony's weightlifting capabilities, prove the ratio of squat to military press weight -/
theorem tony_weightlifting_ratio :
  let curl_weight : ℝ := 90
  let military_press_weight : ℝ := 2 * curl_weight
  let squat_weight : ℝ := 900
  squat_weight / military_press_weight = 5 := by sorry

end NUMINAMATH_CALUDE_tony_weightlifting_ratio_l2797_279711


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2797_279745

def z : ℂ := 2 + Complex.I

theorem imaginary_part_of_z : z.im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2797_279745


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l2797_279757

/-- A random variable following a normal distribution -/
structure NormalRandomVariable where
  μ : ℝ
  σ : ℝ
  hσ : σ > 0

/-- The cumulative distribution function (CDF) of a normal random variable -/
noncomputable def normalCDF (X : NormalRandomVariable) (x : ℝ) : ℝ := sorry

/-- The probability that a random variable falls within an interval -/
noncomputable def probInterval (X : NormalRandomVariable) (a b : ℝ) : ℝ := 
  normalCDF X b - normalCDF X a

theorem normal_distribution_probability 
  (X : NormalRandomVariable) 
  (h1 : X.μ = 3) 
  (h2 : normalCDF X 4 = 0.84) : 
  probInterval X 2 4 = 0.68 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l2797_279757


namespace NUMINAMATH_CALUDE_fourth_day_earning_l2797_279730

/-- Represents the daily earnings of a mechanic for a week -/
def MechanicEarnings : Type := Fin 7 → ℝ

/-- The average earning for the first 4 days is 18 -/
def avg_first_four (e : MechanicEarnings) : Prop :=
  (e 0 + e 1 + e 2 + e 3) / 4 = 18

/-- The average earning for the last 4 days is 22 -/
def avg_last_four (e : MechanicEarnings) : Prop :=
  (e 3 + e 4 + e 5 + e 6) / 4 = 22

/-- The average earning for the whole week is 21 -/
def avg_whole_week (e : MechanicEarnings) : Prop :=
  (e 0 + e 1 + e 2 + e 3 + e 4 + e 5 + e 6) / 7 = 21

/-- The theorem stating that given the conditions, the earning on the fourth day is 13 -/
theorem fourth_day_earning (e : MechanicEarnings) 
  (h1 : avg_first_four e) 
  (h2 : avg_last_four e) 
  (h3 : avg_whole_week e) : 
  e 3 = 13 := by sorry

end NUMINAMATH_CALUDE_fourth_day_earning_l2797_279730


namespace NUMINAMATH_CALUDE_equation_solutions_l2797_279738

/-- The set of solutions to the equation (3x+6)/(x^2+5x-14) = (3-x)/(x-2) -/
def solutions : Set ℝ := {x | x = 3 ∨ x = -5}

/-- The original equation -/
def equation (x : ℝ) : Prop :=
  x ≠ 2 ∧ x ≠ -7 ∧ (3*x + 6) / (x^2 + 5*x - 14) = (3 - x) / (x - 2)

theorem equation_solutions :
  ∀ x : ℝ, equation x ↔ x ∈ solutions := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2797_279738


namespace NUMINAMATH_CALUDE_inscribed_circle_ratio_l2797_279749

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a circle -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Checks if a triangle is isosceles with base AB -/
def IsIsoscelesAB (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2

/-- Checks if a circle is inscribed in a triangle -/
def IsInscribed (c : Circle) (t : Triangle) : Prop := sorry

/-- Checks if a point is on a line segment -/
def IsOnSegment (p : Point) (a : Point) (b : Point) : Prop := sorry

/-- Checks if a point is on a circle -/
def IsOnCircle (p : Point) (c : Circle) : Prop := sorry

/-- Calculates the distance between two points -/
def Distance (a : Point) (b : Point) : ℝ := sorry

/-- The main theorem -/
theorem inscribed_circle_ratio 
  (t : Triangle) 
  (c : Circle) 
  (M N : Point) 
  (k : ℝ) :
  IsIsoscelesAB t →
  IsInscribed c t →
  IsOnSegment M t.B t.C →
  IsOnCircle M c →
  IsOnSegment N t.A M →
  IsOnCircle N c →
  Distance t.A t.B / Distance t.B t.C = k →
  Distance M N / Distance t.A N = 2 * (2 - k) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_ratio_l2797_279749


namespace NUMINAMATH_CALUDE_sqrt_meaningfulness_l2797_279754

theorem sqrt_meaningfulness (x : ℝ) : x = 5 → (2*x - 4 ≥ 0) ∧ 
  (x = -1 → ¬(2*x - 4 ≥ 0)) ∧ 
  (x = 0 → ¬(2*x - 4 ≥ 0)) ∧ 
  (x = 1 → ¬(2*x - 4 ≥ 0)) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningfulness_l2797_279754


namespace NUMINAMATH_CALUDE_coffee_stock_problem_l2797_279764

/-- Represents the coffee stock problem --/
theorem coffee_stock_problem 
  (initial_stock : ℝ) 
  (initial_decaf_percent : ℝ) 
  (second_batch_decaf_percent : ℝ) 
  (final_decaf_percent : ℝ) 
  (second_batch : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.20)
  (h3 : second_batch_decaf_percent = 0.60)
  (h4 : final_decaf_percent = 0.28000000000000004)
  (h5 : (initial_stock * initial_decaf_percent + second_batch * second_batch_decaf_percent) / 
        (initial_stock + second_batch) = final_decaf_percent) : 
  second_batch = 100 := by
  sorry

end NUMINAMATH_CALUDE_coffee_stock_problem_l2797_279764


namespace NUMINAMATH_CALUDE_milk_cisterns_l2797_279739

theorem milk_cisterns (x y z : ℝ) (h1 : x + y + z = 780) 
  (h2 : (3/4) * x = (4/5) * y) (h3 : (3/4) * x = (4/7) * z) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  x = 240 ∧ y = 225 ∧ z = 315 := by
  sorry

end NUMINAMATH_CALUDE_milk_cisterns_l2797_279739


namespace NUMINAMATH_CALUDE_probability_same_color_is_19_39_l2797_279777

def num_green_balls : ℕ := 5
def num_white_balls : ℕ := 8

def total_balls : ℕ := num_green_balls + num_white_balls

def probability_same_color : ℚ :=
  (Nat.choose num_green_balls 2 + Nat.choose num_white_balls 2) / Nat.choose total_balls 2

theorem probability_same_color_is_19_39 :
  probability_same_color = 19 / 39 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_19_39_l2797_279777


namespace NUMINAMATH_CALUDE_power_function_property_l2797_279747

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 2 = Real.sqrt 2) : 
  f 8 = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_power_function_property_l2797_279747


namespace NUMINAMATH_CALUDE_sum_reciprocals_and_powers_l2797_279734

theorem sum_reciprocals_and_powers (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 1 / b ≥ 4) ∧ (1 / a^2016 + 1 / b^2016 ≥ 2^2017) := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_and_powers_l2797_279734


namespace NUMINAMATH_CALUDE_intersection_line_and_chord_length_l2797_279769

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y + 1 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 11 = 0

-- Define the line equation
def lineAB (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0

-- Theorem statement
theorem intersection_line_and_chord_length :
  ∃ (A B : ℝ × ℝ),
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧
    (∀ (x y : ℝ), C₁ x y ∧ C₂ x y → lineAB x y) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 24/5 :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_and_chord_length_l2797_279769


namespace NUMINAMATH_CALUDE_most_tickets_have_four_hits_l2797_279751

/-- Number of matches in a lottery ticket -/
def num_matches : ℕ := 13

/-- Number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 3

/-- Number of tickets with k correct predictions -/
def tickets_with_k_hits (k : ℕ) : ℕ :=
  (num_matches.choose k) * (outcomes_per_match - 1)^(num_matches - k)

/-- The number of correct predictions that maximizes the number of tickets -/
def max_hits : ℕ := 4

theorem most_tickets_have_four_hits :
  ∀ k : ℕ, k ≤ num_matches → k ≠ max_hits →
    tickets_with_k_hits k ≤ tickets_with_k_hits max_hits :=
by sorry

end NUMINAMATH_CALUDE_most_tickets_have_four_hits_l2797_279751


namespace NUMINAMATH_CALUDE_grape_juice_solution_l2797_279798

/-- Represents the problem of adding grape juice to a mixture --/
def GrapeJuiceProblem (initial_volume : ℝ) (initial_concentration : ℝ) (final_concentration : ℝ) (added_juice : ℝ) : Prop :=
  let final_volume := initial_volume + added_juice
  let initial_juice := initial_volume * initial_concentration
  let final_juice := final_volume * final_concentration
  final_juice = initial_juice + added_juice

/-- Theorem stating the solution to the grape juice problem --/
theorem grape_juice_solution :
  GrapeJuiceProblem 30 0.1 0.325 10 := by
  sorry

end NUMINAMATH_CALUDE_grape_juice_solution_l2797_279798


namespace NUMINAMATH_CALUDE_function_identification_l2797_279716

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

-- State the theorem
theorem function_identification (a b c : ℝ) :
  f a b c 0 = 1 ∧ 
  (∃ k m : ℝ, k = 4 * a * 1^3 + 2 * b * 1 ∧ 
              m = a * 1^4 + b * 1^2 + c ∧ 
              k = 1 ∧ 
              m = -1) →
  ∀ x, f a b c x = 5/2 * x^4 - 9/2 * x^2 + 1 :=
by sorry

end NUMINAMATH_CALUDE_function_identification_l2797_279716


namespace NUMINAMATH_CALUDE_swimming_difference_l2797_279771

theorem swimming_difference (camden_total : ℕ) (susannah_total : ℕ) (weeks : ℕ) : 
  camden_total = 16 → susannah_total = 24 → weeks = 4 →
  (susannah_total / weeks) - (camden_total / weeks) = 2 := by
  sorry

end NUMINAMATH_CALUDE_swimming_difference_l2797_279771


namespace NUMINAMATH_CALUDE_town_square_length_l2797_279712

/-- The length of the town square in miles -/
def square_length : ℝ := 5.25

/-- The number of times runners go around the square -/
def laps : ℕ := 7

/-- The time (in minutes) it took the winner to finish the race this year -/
def winner_time : ℝ := 42

/-- The time (in minutes) it took last year's winner to finish the race -/
def last_year_time : ℝ := 47.25

/-- The time difference (in minutes) for running one mile between this year and last year -/
def speed_improvement : ℝ := 1

theorem town_square_length :
  square_length = (last_year_time - winner_time) / speed_improvement :=
by sorry

end NUMINAMATH_CALUDE_town_square_length_l2797_279712


namespace NUMINAMATH_CALUDE_coin_value_theorem_l2797_279710

theorem coin_value_theorem (n d : ℕ) : 
  n + d = 25 →
  (10 * n + 5 * d) - (5 * n + 10 * d) = 100 →
  5 * n + 10 * d = 140 := by
  sorry

end NUMINAMATH_CALUDE_coin_value_theorem_l2797_279710


namespace NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l2797_279784

theorem negative_sixty_four_to_four_thirds : (-64 : ℝ) ^ (4/3) = 256 := by
  sorry

end NUMINAMATH_CALUDE_negative_sixty_four_to_four_thirds_l2797_279784


namespace NUMINAMATH_CALUDE_fib_2n_square_sum_l2797_279724

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: For a Fibonacci sequence, f_{2n} = f_{n-1}^2 + f_n^2 for all natural numbers n -/
theorem fib_2n_square_sum (n : ℕ) : fib (2 * n) = (fib (n - 1))^2 + (fib n)^2 := by
  sorry

end NUMINAMATH_CALUDE_fib_2n_square_sum_l2797_279724


namespace NUMINAMATH_CALUDE_and_or_relationship_l2797_279773

theorem and_or_relationship (p q : Prop) :
  (∀ (p q : Prop), p ∧ q → p ∨ q) ∧
  (∃ (p q : Prop), p ∨ q ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_and_or_relationship_l2797_279773


namespace NUMINAMATH_CALUDE_exists_equidistant_point_l2797_279740

/-- A line in a plane --/
structure Line where
  -- Add necessary fields for a line

/-- A point in a plane --/
structure Point where
  -- Add necessary fields for a point

/-- Three lines in a plane --/
def three_lines : Fin 3 → Line := sorry

/-- Condition that at most two lines are parallel --/
def at_most_two_parallel (lines : Fin 3 → Line) : Prop := sorry

/-- A point is equidistant from three lines --/
def equidistant_from_lines (p : Point) (lines : Fin 3 → Line) : Prop := sorry

/-- Main theorem: There always exists a point equidistant from three lines
    given that at most two of them are parallel --/
theorem exists_equidistant_point (lines : Fin 3 → Line) 
  (h : at_most_two_parallel lines) : 
  ∃ (p : Point), equidistant_from_lines p lines := by
  sorry

end NUMINAMATH_CALUDE_exists_equidistant_point_l2797_279740


namespace NUMINAMATH_CALUDE_quadratic_roots_and_triangle_perimeter_l2797_279737

/-- The quadratic equation in terms of x and k -/
def quadratic (x k : ℝ) : Prop :=
  x^2 - (3*k + 1)*x + 2*k^2 + 2*k = 0

/-- An isosceles triangle with side lengths a, b, and c -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  isIsosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

/-- The theorem to be proved -/
theorem quadratic_roots_and_triangle_perimeter :
  (∀ k : ℝ, ∃ x : ℝ, quadratic x k) ∧
  (∃ t : IsoscelesTriangle, 
    t.a = 6 ∧
    quadratic t.b (t.b/2) ∧
    quadratic t.c ((t.c - 1)/2) ∧
    (t.a + t.b + t.c = 16 ∨ t.a + t.b + t.c = 22)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_and_triangle_perimeter_l2797_279737


namespace NUMINAMATH_CALUDE_S_3_5_equals_42_l2797_279714

-- Define the operation S
def S (a b : ℕ) : ℕ := 4 * a + 6 * b

-- Theorem to prove
theorem S_3_5_equals_42 : S 3 5 = 42 := by
  sorry

end NUMINAMATH_CALUDE_S_3_5_equals_42_l2797_279714


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2797_279743

theorem roots_of_quadratic_equation (a b : ℝ) : 
  (a^2 + a - 5 = 0) → 
  (b^2 + b - 5 = 0) → 
  (a + b = -1) → 
  (a * b = -5) → 
  (2 * a^2 + a + b^2 = 16) := by
sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2797_279743


namespace NUMINAMATH_CALUDE_work_done_by_force_l2797_279778

theorem work_done_by_force (F : ℝ → ℝ) (x₁ x₂ : ℝ) :
  (∀ x, F x = 1 + Real.exp x) →
  x₁ = 0 →
  x₂ = 1 →
  ∫ x in x₁..x₂, F x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_work_done_by_force_l2797_279778


namespace NUMINAMATH_CALUDE_limit_fraction_to_one_third_l2797_279758

theorem limit_fraction_to_one_third :
  ∀ ε > 0, ∃ N : ℝ, ∀ n : ℝ, n > N → |((n + 20) / (3 * n + 1)) - (1 / 3)| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_limit_fraction_to_one_third_l2797_279758


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l2797_279753

theorem unique_solution_inequality (x : ℝ) : 
  x > 0 → x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x^3) ≥ 18 → x = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l2797_279753


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2797_279748

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other -/
def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- Given that point A(2, a) is symmetric to point B(b, -3) with respect to the x-axis,
    prove that a + b = 5 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_x_axis (2, a) (b, -3)) : a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2797_279748


namespace NUMINAMATH_CALUDE_smallest_sum_is_84_l2797_279761

/-- Represents a rectangular prism made of dice -/
structure DicePrism where
  length : Nat
  width : Nat
  height : Nat
  total_dice : Nat
  dice_opposite_sum : Nat

/-- Calculates the smallest possible sum of visible values on the prism faces -/
def smallest_visible_sum (prism : DicePrism) : Nat :=
  sorry

/-- Theorem stating the smallest possible sum for the given prism configuration -/
theorem smallest_sum_is_84 (prism : DicePrism) 
  (h1 : prism.length = 4)
  (h2 : prism.width = 3)
  (h3 : prism.height = 2)
  (h4 : prism.total_dice = 24)
  (h5 : prism.dice_opposite_sum = 7) :
  smallest_visible_sum prism = 84 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_is_84_l2797_279761


namespace NUMINAMATH_CALUDE_max_consecutive_new_numbers_l2797_279750

def is_new (n : Nat) : Prop :=
  n > 5 ∧ ∃ m : Nat, (∀ k < n, m % k = 0) ∧ m % n ≠ 0

theorem max_consecutive_new_numbers :
  ∃ a : Nat, a > 5 ∧
    is_new a ∧ is_new (a + 1) ∧ is_new (a + 2) ∧
    ¬(is_new (a - 1) ∧ is_new a ∧ is_new (a + 1) ∧ is_new (a + 2)) ∧
    ¬(is_new a ∧ is_new (a + 1) ∧ is_new (a + 2) ∧ is_new (a + 3)) :=
  sorry

end NUMINAMATH_CALUDE_max_consecutive_new_numbers_l2797_279750


namespace NUMINAMATH_CALUDE_equation_solution_l2797_279763

theorem equation_solution : 
  ∃! x : ℚ, (x - 20) / 3 = (4 - 3 * x) / 4 ∧ x = 92 / 13 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2797_279763


namespace NUMINAMATH_CALUDE_max_M_inequality_l2797_279705

theorem max_M_inequality (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  (∃ (M : ℝ), ∀ (a b c : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → 
    a^3 + b^3 + c^3 - 3*a*b*c ≥ M*(a-b)*(b-c)*(c-a)) ↔ 
  (M ≤ Real.sqrt (9 + 6 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_max_M_inequality_l2797_279705


namespace NUMINAMATH_CALUDE_donut_selection_problem_l2797_279727

theorem donut_selection_problem :
  let n : ℕ := 5  -- number of donuts to select
  let k : ℕ := 4  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 56 := by
sorry

end NUMINAMATH_CALUDE_donut_selection_problem_l2797_279727


namespace NUMINAMATH_CALUDE_lake_fish_population_l2797_279729

/-- Represents the fish population in a lake --/
structure FishPopulation where
  initial_tagged : ℕ
  second_catch : ℕ
  tagged_in_second_catch : ℕ
  new_migrants : ℕ

/-- Calculates the approximate total number of fish in the lake --/
def approximate_total_fish (fp : FishPopulation) : ℕ :=
  (fp.initial_tagged * fp.second_catch) / fp.tagged_in_second_catch

/-- The main theorem stating the approximate number of fish in the lake --/
theorem lake_fish_population (fp : FishPopulation) 
  (h1 : fp.initial_tagged = 500)
  (h2 : fp.second_catch = 300)
  (h3 : fp.tagged_in_second_catch = 6)
  (h4 : fp.new_migrants = 250) :
  approximate_total_fish fp = 25000 := by
  sorry

#eval approximate_total_fish { initial_tagged := 500, second_catch := 300, tagged_in_second_catch := 6, new_migrants := 250 }

end NUMINAMATH_CALUDE_lake_fish_population_l2797_279729


namespace NUMINAMATH_CALUDE_bookshelf_average_l2797_279756

theorem bookshelf_average (initial_books : ℕ) (new_books : ℕ) (shelves : ℕ) (leftover : ℕ) 
  (h1 : initial_books = 56)
  (h2 : new_books = 26)
  (h3 : shelves = 4)
  (h4 : leftover = 2) :
  (initial_books + new_books - leftover) / shelves = 20 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_average_l2797_279756


namespace NUMINAMATH_CALUDE_origin_outside_circle_l2797_279792

theorem origin_outside_circle (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2*y + a - 2 = 0 → (x^2 + y^2 > 0)) ↔ (2 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_origin_outside_circle_l2797_279792
