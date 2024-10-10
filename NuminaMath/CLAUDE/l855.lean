import Mathlib

namespace melanie_has_41_balloons_l855_85510

/-- The number of blue balloons Joan has -/
def joan_balloons : ℕ := 40

/-- The total number of blue balloons -/
def total_balloons : ℕ := 81

/-- The number of blue balloons Melanie has -/
def melanie_balloons : ℕ := total_balloons - joan_balloons

theorem melanie_has_41_balloons : melanie_balloons = 41 := by
  sorry

end melanie_has_41_balloons_l855_85510


namespace bird_stork_difference_l855_85531

theorem bird_stork_difference (initial_birds storks additional_birds : ℕ) :
  initial_birds = 3 →
  storks = 4 →
  additional_birds = 2 →
  (initial_birds + additional_birds) - storks = 1 := by
  sorry

end bird_stork_difference_l855_85531


namespace point_on_line_l855_85561

/-- Given a line in 3D space defined by the vector equation (x,y,z) = (5,0,3) + t(0,3,0),
    this theorem proves that the point on the line when t = 1/2 has coordinates (5,3/2,3). -/
theorem point_on_line (x y z t : ℝ) : 
  (x, y, z) = (5, 0, 3) + t • (0, 3, 0) → 
  t = 1/2 → 
  (x, y, z) = (5, 3/2, 3) := by
sorry

end point_on_line_l855_85561


namespace phil_charlie_difference_l855_85525

/-- Represents the number of games won by each player -/
structure GamesWon where
  perry : ℕ
  dana : ℕ
  charlie : ℕ
  phil : ℕ

/-- Conditions for the golf game results -/
def golf_conditions (g : GamesWon) : Prop :=
  g.perry = g.dana + 5 ∧
  g.charlie = g.dana - 2 ∧
  g.phil > g.charlie ∧
  g.phil = 12 ∧
  g.perry = g.phil + 4

/-- Theorem stating the difference between Phil's and Charlie's games -/
theorem phil_charlie_difference (g : GamesWon) (h : golf_conditions g) : 
  g.phil - g.charlie = 3 := by
  sorry

end phil_charlie_difference_l855_85525


namespace nonconvex_quadrilateral_theorem_l855_85518

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Checks if a quadrilateral is nonconvex -/
def is_nonconvex (q : Quadrilateral) : Prop := sorry

/-- Calculates the angle at a vertex of a quadrilateral -/
def angle_at_vertex (q : Quadrilateral) (v : Point2D) : ℝ := sorry

/-- Finds the intersection point of two lines -/
def line_intersection (p1 p2 q1 q2 : Point2D) : Point2D := sorry

/-- Checks if a point lies on a line segment -/
def point_on_segment (p : Point2D) (a b : Point2D) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point2D) : ℝ := sorry

theorem nonconvex_quadrilateral_theorem (ABCD : Quadrilateral) 
  (hnonconvex : is_nonconvex ABCD)
  (hC_angle : angle_at_vertex ABCD ABCD.C > 180)
  (F : Point2D) (hF : F = line_intersection ABCD.D ABCD.C ABCD.A ABCD.B)
  (E : Point2D) (hE : E = line_intersection ABCD.B ABCD.C ABCD.A ABCD.D)
  (K L J I : Point2D)
  (hK : point_on_segment K ABCD.A ABCD.B)
  (hL : point_on_segment L ABCD.A ABCD.D)
  (hJ : point_on_segment J ABCD.B ABCD.C)
  (hI : point_on_segment I ABCD.C ABCD.D)
  (hDI_CF : distance ABCD.D I = distance ABCD.C F)
  (hBJ_CE : distance ABCD.B J = distance ABCD.C E) :
  distance K J = distance I L := by sorry

end nonconvex_quadrilateral_theorem_l855_85518


namespace negative_integer_problem_l855_85550

theorem negative_integer_problem (n : ℤ) : n < 0 → n * (-3) + 2 = 65 → n = -21 := by
  sorry

end negative_integer_problem_l855_85550


namespace forester_tree_planting_l855_85592

/-- A forester's tree planting problem --/
theorem forester_tree_planting (initial_trees : ℕ) (total_goal : ℕ) : 
  initial_trees = 30 →
  total_goal = 300 →
  let monday_planted := 2 * initial_trees
  let tuesday_planted := monday_planted / 3
  let wednesday_planted := 2 * tuesday_planted
  let total_planted := monday_planted + tuesday_planted + wednesday_planted
  total_planted = 120 ∧ initial_trees + total_planted = total_goal := by
  sorry

end forester_tree_planting_l855_85592


namespace candy_necklaces_per_pack_l855_85577

theorem candy_necklaces_per_pack (total_packs : ℕ) (opened_packs : ℕ) (leftover_necklaces : ℕ) 
  (h1 : total_packs = 9)
  (h2 : opened_packs = 4)
  (h3 : leftover_necklaces = 40) :
  leftover_necklaces / (total_packs - opened_packs) = 8 := by
  sorry

end candy_necklaces_per_pack_l855_85577


namespace bug_position_after_2021_jumps_l855_85504

/-- Represents the seven points on the circle -/
inductive Point
| One | Two | Three | Four | Five | Six | Seven

/-- Determines if a point is prime -/
def isPrime : Point → Bool
  | Point.Two => true
  | Point.Three => true
  | Point.Five => true
  | Point.Seven => true
  | _ => false

/-- Calculates the next point based on the current point -/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.One => Point.Four
  | Point.Two => Point.Four
  | Point.Three => Point.Five
  | Point.Four => Point.Seven
  | Point.Five => Point.Seven
  | Point.Six => Point.Two
  | Point.Seven => Point.Two

/-- Calculates the bug's position after n jumps -/
def bugPosition (start : Point) (n : Nat) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (bugPosition start n)

/-- The main theorem to prove -/
theorem bug_position_after_2021_jumps :
  bugPosition Point.Seven 2021 = Point.Two :=
sorry

end bug_position_after_2021_jumps_l855_85504


namespace smallest_square_partition_l855_85541

/-- Represents a square partition of a larger square -/
structure SquarePartition where
  side_length : ℕ
  partitions : List ℕ
  partition_count : partitions.length = 15
  all_integer : ∀ n ∈ partitions, n > 0
  sum_areas : (partitions.map (λ x => x * x)).sum = side_length * side_length
  unit_squares : (partitions.filter (λ x => x = 1)).length ≥ 12

/-- The smallest square that satisfies the partition conditions has side length 5 -/
theorem smallest_square_partition :
  ∀ sp : SquarePartition, sp.side_length ≥ 5 ∧
  ∃ sp' : SquarePartition, sp'.side_length = 5 :=
by sorry

end smallest_square_partition_l855_85541


namespace fraction_equality_l855_85520

theorem fraction_equality (a b c d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) 
  (h1 : (a / b)^2 = (c / d)^2) (h2 : a * c < 0) : 
  a / b = -(c / d) := by sorry

end fraction_equality_l855_85520


namespace first_digit_of_y_in_base_9_l855_85529

def base_3_num : List Nat := [1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1]

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

def y : Nat := to_base_10 base_3_num 3

def first_digit_base_9 (n : Nat) : Nat :=
  if n = 0 then 0 else
  let log_9 := Nat.log n 9
  (n / (9 ^ log_9)) % 9

theorem first_digit_of_y_in_base_9 :
  first_digit_base_9 y = 4 := by sorry

end first_digit_of_y_in_base_9_l855_85529


namespace fraction_equality_proof_l855_85559

theorem fraction_equality_proof (x : ℝ) : 
  x ≠ 4 ∧ x ≠ 8 → ((x - 3) / (x - 4) = (x - 5) / (x - 8) ↔ x = 2) :=
by sorry

end fraction_equality_proof_l855_85559


namespace gmat_scores_l855_85519

theorem gmat_scores (x y z : ℝ) (h1 : x - y = 1/3) (h2 : z = (x + y) / 2) :
  y = x - 1/3 ∧ z = x - 1/6 := by
  sorry

end gmat_scores_l855_85519


namespace walking_path_area_l855_85569

/-- The area of a circular walking path -/
theorem walking_path_area (outer_radius inner_radius : ℝ) 
  (h_outer : outer_radius = 26)
  (h_inner : inner_radius = 16) : 
  π * (outer_radius^2 - inner_radius^2) = 420 * π := by
  sorry

#check walking_path_area

end walking_path_area_l855_85569


namespace roots_of_cubic_equations_l855_85502

theorem roots_of_cubic_equations (p q r s : ℂ) (m : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0)
  (h1 : p * m^3 + q * m^2 + r * m + s = 0)
  (h2 : q * m^3 + r * m^2 + s * m + p = 0) :
  m = 1 ∨ m = -1 ∨ m = Complex.I ∨ m = -Complex.I := by
sorry

end roots_of_cubic_equations_l855_85502


namespace unique_intersection_implies_a_value_l855_85566

/-- Given a line y = 2a and a function y = |x-a| - 1 in the Cartesian coordinate system,
    if they have only one intersection point, then a = -1/2 --/
theorem unique_intersection_implies_a_value (a : ℝ) :
  (∃! p : ℝ × ℝ, p.2 = 2*a ∧ p.2 = |p.1 - a| - 1) → a = -1/2 := by
  sorry

end unique_intersection_implies_a_value_l855_85566


namespace division_problem_l855_85527

theorem division_problem (a b q : ℕ) (h1 : a - b = 1365) (h2 : a = 1620) (h3 : a = b * q + 15) : q = 6 := by
  sorry

end division_problem_l855_85527


namespace white_stamp_price_is_20_cents_l855_85554

/-- The price of a white stamp that satisfies the given conditions -/
def white_stamp_price : ℚ :=
  let red_stamps : ℕ := 30
  let white_stamps : ℕ := 80
  let red_stamp_price : ℚ := 1/2
  let sales_difference : ℚ := 1
  (sales_difference + red_stamps * red_stamp_price) / white_stamps

/-- Theorem stating that the white stamp price is 20 cents -/
theorem white_stamp_price_is_20_cents :
  white_stamp_price = 1/5 := by sorry

end white_stamp_price_is_20_cents_l855_85554


namespace valid_assignment_l855_85599

-- Define the squares
inductive Square
| A | B | C | D | E | F | G

-- Define the arrow directions
def nextSquare : Square → Square
| Square.B => Square.E
| Square.E => Square.C
| Square.C => Square.D
| Square.D => Square.A
| Square.A => Square.G
| Square.G => Square.F
| Square.F => Square.A  -- This should point to the square with 9, which is not in our Square type

-- Define the assignment of numbers to squares
def assignment : Square → Fin 8
| Square.A => 6
| Square.B => 2
| Square.C => 4
| Square.D => 5
| Square.E => 3
| Square.F => 8
| Square.G => 7

-- Theorem statement
theorem valid_assignment : 
  (∀ s : Square, assignment (nextSquare s) = assignment s + 1) ∧
  (∀ i : Fin 8, ∃ s : Square, assignment s = i) :=
by sorry

end valid_assignment_l855_85599


namespace sum_of_six_numbers_l855_85562

theorem sum_of_six_numbers : (36 : ℕ) + 17 + 32 + 54 + 28 + 3 = 170 := by
  sorry

end sum_of_six_numbers_l855_85562


namespace center_locus_is_single_point_l855_85539

/-- Two fixed points in a plane -/
structure FixedPoints (α : Type*) [NormedAddCommGroup α] where
  P : α
  Q : α

/-- A circle passing through two fixed points with constant radius -/
structure Circle (α : Type*) [NormedAddCommGroup α] where
  center : α
  radius : ℝ
  fixedPoints : FixedPoints α

/-- The locus of centers of circles passing through two fixed points -/
def CenterLocus (α : Type*) [NormedAddCommGroup α] (a : ℝ) : Set α :=
  {C : α | ∃ (circ : Circle α), circ.center = C ∧ circ.radius = a}

/-- The theorem stating that the locus of centers is a single point -/
theorem center_locus_is_single_point
  (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α]
  (a : ℝ) (points : FixedPoints α)
  (h : ‖points.P - points.Q‖ = 2 * a) :
  ∃! C, C ∈ CenterLocus α a :=
sorry

end center_locus_is_single_point_l855_85539


namespace arithmetic_sequence_sum_l855_85558

/-- An arithmetic sequence with first term a₁ and common difference d -/
structure ArithmeticSequence where
  a₁ : ℤ
  d : ℤ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a₁ + (n * (n - 1) / 2) * seq.d

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  seq.a₁ = -2014 →
  (sum_n seq 2012 / 2012 : ℚ) - (sum_n seq 10 / 10 : ℚ) = 2002 →
  sum_n seq 2016 = 2016 := by
  sorry

end arithmetic_sequence_sum_l855_85558


namespace quadratic_roots_exist_sum_minus_product_equals_two_l855_85511

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 3*x + 1 = 0

-- Define the roots
theorem quadratic_roots_exist : ∃ (x₁ x₂ : ℝ), quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ x₁ ≠ x₂ :=
sorry

-- Theorem to prove
theorem sum_minus_product_equals_two :
  ∃ (x₁ x₂ : ℝ), quadratic_equation x₁ ∧ quadratic_equation x₂ ∧ x₁ ≠ x₂ ∧ x₁ + x₂ - x₁*x₂ = 2 :=
sorry

end quadratic_roots_exist_sum_minus_product_equals_two_l855_85511


namespace abc_value_for_specific_factorization_l855_85538

theorem abc_value_for_specific_factorization (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = (x - 1) * (x - 2)) → a * b * c = -6 := by
  sorry

end abc_value_for_specific_factorization_l855_85538


namespace matrix_sum_equality_l855_85598

def A : Matrix (Fin 3) (Fin 3) ℤ := !![4, 1, -3; 0, -2, 5; 7, 0, 1]
def B : Matrix (Fin 3) (Fin 3) ℤ := !![-6, 9, 2; 3, -4, -8; 0, 5, -3]
def C : Matrix (Fin 3) (Fin 3) ℤ := !![-2, 10, -1; 3, -6, -3; 7, 5, -2]

theorem matrix_sum_equality : A + B = C := by sorry

end matrix_sum_equality_l855_85598


namespace morning_and_evening_emails_sum_l855_85575

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 3

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 8

/-- Theorem: The sum of emails Jack received in the morning and evening is 11 -/
theorem morning_and_evening_emails_sum :
  morning_emails + evening_emails = 11 := by sorry

end morning_and_evening_emails_sum_l855_85575


namespace cubic_function_value_l855_85555

/-- Given a cubic function f(x) = ax³ + bx + 3 where f(-3) = 10, prove that f(3) = 27a + 3b + 3 -/
theorem cubic_function_value (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x + 3)
  (h2 : f (-3) = 10) :
  f 3 = 27 * a + 3 * b + 3 := by
  sorry

end cubic_function_value_l855_85555


namespace correct_equation_l855_85580

theorem correct_equation : (-3)^2 * |-(1/3)| = 3 := by
  sorry

end correct_equation_l855_85580


namespace pokemon_cards_bought_l855_85524

theorem pokemon_cards_bought (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 676)
  (h2 : final_cards = 900) :
  final_cards - initial_cards = 224 := by
  sorry

end pokemon_cards_bought_l855_85524


namespace race_time_proof_l855_85567

-- Define the race parameters
def race_distance : ℝ := 120
def distance_difference : ℝ := 72
def time_difference : ℝ := 10

-- Define the theorem
theorem race_time_proof :
  ∀ (v_a v_b t_a : ℝ),
  v_a > 0 → v_b > 0 → t_a > 0 →
  v_a = race_distance / t_a →
  v_b = (race_distance - distance_difference) / t_a →
  v_b = distance_difference / (t_a + time_difference) →
  t_a = 20 := by
sorry


end race_time_proof_l855_85567


namespace complex_magnitude_problem_l855_85545

theorem complex_magnitude_problem : 
  let z : ℂ := (1 + 3*I) / (3 - I) - 3*I
  Complex.abs z = 2 := by sorry

end complex_magnitude_problem_l855_85545


namespace exists_decreasing_arithmetic_with_non_decreasing_sums_l855_85587

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sequence of partial sums of a given sequence -/
def partial_sums (a : ℕ → ℝ) : ℕ → ℝ
  | 0 => 0
  | n + 1 => partial_sums a n + a (n + 1)

/-- A sequence is decreasing -/
def is_decreasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ a n

theorem exists_decreasing_arithmetic_with_non_decreasing_sums :
  ∃ a : ℕ → ℝ,
    arithmetic_sequence a ∧
    is_decreasing a ∧
    (∀ n : ℕ, a n = -2 * n + 7) ∧
    ¬(is_decreasing (partial_sums a)) := by
  sorry

end exists_decreasing_arithmetic_with_non_decreasing_sums_l855_85587


namespace rhombus_diagonal_sum_max_l855_85572

theorem rhombus_diagonal_sum_max (s x y : ℝ) : 
  s = 5 → 
  x^2 + y^2 = 4 * s^2 →
  x ≥ 6 →
  y ≤ 6 →
  x + y ≤ 14 :=
by sorry

end rhombus_diagonal_sum_max_l855_85572


namespace angle_sum_equation_l855_85508

theorem angle_sum_equation (α β : Real) (h : (1 + Real.sqrt 3 * Real.tan α) * (1 + Real.sqrt 3 * Real.tan β) = 4) :
  α + β = π / 3 := by
sorry

end angle_sum_equation_l855_85508


namespace bears_per_shelf_l855_85523

theorem bears_per_shelf (initial_stock : ℕ) (new_shipment : ℕ) (num_shelves : ℕ) 
  (h1 : initial_stock = 5)
  (h2 : new_shipment = 7)
  (h3 : num_shelves = 2)
  : (initial_stock + new_shipment) / num_shelves = 6 := by
  sorry

end bears_per_shelf_l855_85523


namespace parrot_seed_consumption_l855_85588

/-- Calculates the weekly seed consumption of a parrot given the total birdseed supply,
    number of weeks, and the cockatiel's weekly consumption. --/
theorem parrot_seed_consumption
  (total_boxes : ℕ)
  (seeds_per_box : ℕ)
  (weeks : ℕ)
  (cockatiel_weekly : ℕ)
  (h1 : total_boxes = 8)
  (h2 : seeds_per_box = 225)
  (h3 : weeks = 12)
  (h4 : cockatiel_weekly = 50) :
  (total_boxes * seeds_per_box - weeks * cockatiel_weekly) / weeks = 100 := by
  sorry

#check parrot_seed_consumption

end parrot_seed_consumption_l855_85588


namespace domino_tiling_theorem_l855_85507

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a domino placement on a board -/
def Tiling (b : Board) := Set (ℕ × ℕ × Bool)

/-- Checks if a tiling is valid for a given board -/
def is_valid_tiling (b : Board) (t : Tiling b) : Prop := sorry

/-- Checks if a line bisects at least one domino in the tiling -/
def line_bisects_domino (b : Board) (t : Tiling b) (line : ℕ × Bool) : Prop := sorry

/-- Counts the number of internal lines in a board -/
def internal_lines_count (b : Board) : ℕ := 
  b.rows + b.cols - 2

/-- Main theorem statement -/
theorem domino_tiling_theorem :
  (¬ ∃ (t : Tiling ⟨6, 6⟩), 
    is_valid_tiling ⟨6, 6⟩ t ∧ 
    ∀ (line : ℕ × Bool), line_bisects_domino ⟨6, 6⟩ t line) ∧
  (∃ (t : Tiling ⟨5, 6⟩), 
    is_valid_tiling ⟨5, 6⟩ t ∧ 
    ∀ (line : ℕ × Bool), line_bisects_domino ⟨5, 6⟩ t line) :=
sorry

end domino_tiling_theorem_l855_85507


namespace ab_leq_one_l855_85509

theorem ab_leq_one (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b = 2) : a * b ≤ 1 := by
  sorry

end ab_leq_one_l855_85509


namespace seventh_group_draw_l855_85516

/-- Represents the systematic sampling method for a population -/
structure SystematicSampling where
  population_size : Nat
  num_groups : Nat
  sample_size : Nat
  m : Nat

/-- Calculates the number drawn from a specific group -/
def number_drawn (ss : SystematicSampling) (group : Nat) : Nat :=
  let group_size := ss.population_size / ss.num_groups
  let start := (group - 1) * group_size
  let units_digit := (ss.m + group) % 10
  start + units_digit

/-- Theorem stating that the number drawn from the 7th group is 63 -/
theorem seventh_group_draw (ss : SystematicSampling) 
  (h1 : ss.population_size = 100)
  (h2 : ss.num_groups = 10)
  (h3 : ss.sample_size = 10)
  (h4 : ss.m = 6) :
  number_drawn ss 7 = 63 := by
  sorry

end seventh_group_draw_l855_85516


namespace cats_after_sale_l855_85513

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_after_sale 
  (siamese : ℕ) -- Initial number of Siamese cats
  (house : ℕ) -- Initial number of house cats
  (sold : ℕ) -- Number of cats sold during the sale
  (h1 : siamese = 12)
  (h2 : house = 20)
  (h3 : sold = 20) :
  siamese + house - sold = 12 := by
  sorry

end cats_after_sale_l855_85513


namespace computer_music_time_l855_85564

def total_time : ℕ := 120
def piano_time : ℕ := 30
def reading_time : ℕ := 38
def exerciser_time : ℕ := 27

theorem computer_music_time : 
  total_time - (piano_time + reading_time + exerciser_time) = 25 := by
sorry

end computer_music_time_l855_85564


namespace area_between_concentric_circles_l855_85549

theorem area_between_concentric_circles 
  (r_outer : ℝ) 
  (r_inner : ℝ) 
  (chord_length : ℝ) 
  (h_r_outer : r_outer = 60) 
  (h_r_inner : r_inner = 36) 
  (h_chord : chord_length = 96) 
  (h_tangent : chord_length / 2 = Real.sqrt (r_outer^2 - r_inner^2)) : 
  π * (r_outer^2 - r_inner^2) = 2304 * π := by
sorry

end area_between_concentric_circles_l855_85549


namespace x_minus_y_value_l855_85536

theorem x_minus_y_value (x y : ℝ) (h1 : x + y = 20) (h2 : x^2 - y^2 = 36) : x - y = 9/5 := by
  sorry

end x_minus_y_value_l855_85536


namespace ourSystem_is_valid_l855_85581

-- Define a structure for a linear equation
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  -- ax + by = c

-- Define a system of two linear equations
structure SystemOfTwoLinearEquations where
  eq1 : LinearEquation
  eq2 : LinearEquation

-- Define the specific system we want to prove is valid
def ourSystem : SystemOfTwoLinearEquations := {
  eq1 := { a := 1, b := 1, c := 5 },  -- x + y = 5
  eq2 := { a := 0, b := 1, c := 2 }   -- y = 2
}

-- Theorem stating that our system is a valid system of two linear equations
theorem ourSystem_is_valid : 
  (ourSystem.eq1.a ≠ 0 ∨ ourSystem.eq1.b ≠ 0) ∧ 
  (ourSystem.eq2.a ≠ 0 ∨ ourSystem.eq2.b ≠ 0) :=
by sorry

end ourSystem_is_valid_l855_85581


namespace monthly_income_proof_l855_85521

/-- Given the average monthly incomes of three people, prove the income of one person. -/
theorem monthly_income_proof (P Q R : ℝ) 
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (Q + R) / 2 = 5250)
  (h3 : (P + R) / 2 = 6200) :
  P = 3000 := by
  sorry

end monthly_income_proof_l855_85521


namespace astrophysics_budget_decrease_l855_85546

def current_year_allocations : List (String × Rat) :=
  [("Microphotonics", 14/100),
   ("Home Electronics", 24/100),
   ("Food Additives", 15/100),
   ("Genetically Modified Microorganisms", 19/100),
   ("Industrial Lubricants", 8/100)]

def previous_year_allocations : List (String × Rat) :=
  [("Microphotonics", 12/100),
   ("Home Electronics", 22/100),
   ("Food Additives", 13/100),
   ("Genetically Modified Microorganisms", 18/100),
   ("Industrial Lubricants", 7/100)]

def calculate_astrophysics_allocation (allocations : List (String × Rat)) : Rat :=
  1 - (allocations.map (fun x => x.2)).sum

def calculate_percentage_change (old_value : Rat) (new_value : Rat) : Rat :=
  (new_value - old_value) / old_value * 100

theorem astrophysics_budget_decrease :
  let current_astrophysics := calculate_astrophysics_allocation current_year_allocations
  let previous_astrophysics := calculate_astrophysics_allocation previous_year_allocations
  let percentage_change := calculate_percentage_change previous_astrophysics current_astrophysics
  percentage_change = -2857/100 := by sorry

end astrophysics_budget_decrease_l855_85546


namespace last_problem_number_l855_85596

theorem last_problem_number (start : ℕ) (solved : ℕ) (last : ℕ) : 
  start = 78 → solved = 48 → last = start + solved - 1 → last = 125 := by
  sorry

end last_problem_number_l855_85596


namespace rectangle_length_l855_85590

theorem rectangle_length (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 6075 →
  length = 135 := by
sorry

end rectangle_length_l855_85590


namespace new_students_average_age_l855_85594

theorem new_students_average_age
  (original_strength : ℕ)
  (original_average_age : ℝ)
  (new_students : ℕ)
  (new_average_age : ℝ) :
  original_strength = 10 →
  original_average_age = 40 →
  new_students = 10 →
  new_average_age = 36 →
  let total_original_age := original_strength * original_average_age
  let total_new_age := (original_strength + new_students) * new_average_age
  let new_students_total_age := total_new_age - total_original_age
  new_students_total_age / new_students = 32 := by
sorry

end new_students_average_age_l855_85594


namespace one_valid_placement_l855_85530

/-- Represents the number of pegs of each color -/
structure PegCounts where
  purple : Nat
  yellow : Nat
  red : Nat
  green : Nat
  blue : Nat

/-- Represents a hexagonal peg board -/
structure HexBoard where
  rows : Nat
  columns : Nat

/-- Counts the number of valid peg placements -/
def countValidPlacements (board : HexBoard) (pegs : PegCounts) : Nat :=
  sorry

/-- Theorem stating that there is exactly one valid placement -/
theorem one_valid_placement (board : HexBoard) (pegs : PegCounts) : 
  board.rows = 6 ∧ board.columns = 6 ∧ 
  pegs.purple = 6 ∧ pegs.yellow = 5 ∧ pegs.red = 4 ∧ pegs.green = 3 ∧ pegs.blue = 2 →
  countValidPlacements board pegs = 1 := by
  sorry

end one_valid_placement_l855_85530


namespace x_values_difference_l855_85551

theorem x_values_difference (x : ℝ) : 
  (x + 3)^2 / (3*x + 29) = 2 → ∃ (x₁ x₂ : ℝ), x₁ - x₂ = 14 ∧ 
    ((x₁ + 3)^2 / (3*x₁ + 29) = 2) ∧ ((x₂ + 3)^2 / (3*x₂ + 29) = 2) := by
  sorry

end x_values_difference_l855_85551


namespace quadratic_root_shift_l855_85568

theorem quadratic_root_shift (a b c t : ℤ) (ha : a ≠ 0) :
  (a * t^2 + b * t + c = 0) →
  ∃ (p q r : ℤ), p ≠ 0 ∧ p * (t + 2)^2 + q * (t + 2) + r = 0 :=
by sorry

end quadratic_root_shift_l855_85568


namespace davids_trip_money_l855_85591

theorem davids_trip_money (initial_amount spent_amount remaining_amount : ℕ) :
  remaining_amount = 500 →
  spent_amount = remaining_amount + 800 →
  initial_amount = spent_amount + remaining_amount →
  initial_amount = 1800 :=
by sorry

end davids_trip_money_l855_85591


namespace pump_fill_time_l855_85552

/-- The time it takes to fill the tank with the leak present -/
def fill_time_with_leak : ℝ := 3

/-- The time it takes for the leak to drain the full tank -/
def leak_drain_time : ℝ := 5.999999999999999

/-- The time it takes for the pump to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 2

theorem pump_fill_time :
  (1 / fill_time_without_leak) - (1 / leak_drain_time) = (1 / fill_time_with_leak) :=
sorry

end pump_fill_time_l855_85552


namespace sum_of_coefficients_l855_85532

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  (∀ x y : ℝ, (x - 2*y)^5 = a*(x + 2*y)^5 + a₁*(x + 2*y)^4*y + a₂*(x + 2*y)^3*y^2 + 
                            a₃*(x + 2*y)^2*y^3 + a₄*(x + 2*y)*y^4 + a₅*y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -243 := by
sorry

end sum_of_coefficients_l855_85532


namespace square_land_side_length_l855_85557

theorem square_land_side_length (area : ℝ) (h : area = Real.sqrt 900) :
  ∃ (side : ℝ), side * side = area ∧ side = 30 := by
  sorry

end square_land_side_length_l855_85557


namespace final_s_is_negative_one_l855_85597

/-- Represents the state of the algorithm at each iteration -/
structure AlgorithmState where
  s : Int
  iterations : Nat

/-- The algorithm's step function -/
def step (state : AlgorithmState) : AlgorithmState :=
  if state.iterations % 2 = 0 then
    { s := state.s + 1, iterations := state.iterations + 1 }
  else
    { s := state.s - 1, iterations := state.iterations + 1 }

/-- The initial state of the algorithm -/
def initialState : AlgorithmState := { s := 0, iterations := 0 }

/-- Applies the step function n times -/
def applyNTimes (n : Nat) (state : AlgorithmState) : AlgorithmState :=
  match n with
  | 0 => state
  | n + 1 => step (applyNTimes n state)

/-- The final state after 5 iterations -/
def finalState : AlgorithmState := applyNTimes 5 initialState

/-- The theorem stating that the final value of s is -1 -/
theorem final_s_is_negative_one : finalState.s = -1 := by
  sorry


end final_s_is_negative_one_l855_85597


namespace series_sum_l855_85553

/-- The sum of the infinite series ∑(n=1 to ∞) (5n-1)/(3^n) is equal to 13/6 -/
theorem series_sum : (∑' n : ℕ, (5 * n - 1 : ℝ) / (3 : ℝ) ^ n) = 13 / 6 := by
  sorry

end series_sum_l855_85553


namespace particular_number_problem_l855_85522

theorem particular_number_problem (x : ℤ) : 
  (x - 29 + 64 = 76) → x = 41 := by
sorry

end particular_number_problem_l855_85522


namespace least_valid_number_l855_85576

def is_valid (n : ℕ) : Prop :=
  n > 1 ∧
  n % 3 = 2 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 6 = 2 ∧
  n % 7 = 2 ∧
  n % 8 = 2 ∧
  n % 9 = 2 ∧
  n % 11 = 2

theorem least_valid_number : 
  is_valid 27722 ∧ ∀ m : ℕ, m < 27722 → ¬is_valid m :=
by sorry

end least_valid_number_l855_85576


namespace expression_evaluation_l855_85505

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  x^2 + y^2 - z^2 - 2*x*y = 24 := by
sorry

end expression_evaluation_l855_85505


namespace career_preference_proof_l855_85537

/-- Represents the ratio of boys to girls in a class -/
def boyGirlRatio : ℚ := 2/3

/-- Represents the fraction of the circle graph allocated to a specific career -/
def careerFraction : ℚ := 192/360

/-- Represents the fraction of girls who prefer the specific career -/
def girlPreferenceFraction : ℚ := 2/3

/-- Represents the fraction of boys who prefer the specific career -/
def boyPreferenceFraction : ℚ := 1/3

theorem career_preference_proof :
  let totalStudents := boyGirlRatio + 1
  let boyFraction := boyGirlRatio / totalStudents
  let girlFraction := 1 / totalStudents
  careerFraction = boyFraction * boyPreferenceFraction + girlFraction * girlPreferenceFraction :=
by sorry

end career_preference_proof_l855_85537


namespace midpoint_complex_coordinates_l855_85500

theorem midpoint_complex_coordinates (A B C : ℂ) :
  A = 6 + 5*I ∧ B = -2 + 3*I ∧ C = (A + B) / 2 →
  C = 2 + 4*I :=
by sorry

end midpoint_complex_coordinates_l855_85500


namespace satellite_forecast_probability_l855_85556

theorem satellite_forecast_probability (p_a p_b : ℝ) (h_a : p_a = 0.8) (h_b : p_b = 0.75) :
  1 - (1 - p_a) * (1 - p_b) = 0.95 := by
  sorry

end satellite_forecast_probability_l855_85556


namespace overlapping_squares_areas_l855_85542

/-- Represents the side lengths of three overlapping squares -/
structure SquareSides where
  largest : ℝ
  middle : ℝ
  smallest : ℝ

/-- Represents the areas of three overlapping squares -/
structure SquareAreas where
  largest : ℝ
  middle : ℝ
  smallest : ℝ

/-- Calculates the areas of three overlapping squares given their side lengths -/
def calculateAreas (sides : SquareSides) : SquareAreas :=
  { largest := sides.largest ^ 2,
    middle := sides.middle ^ 2,
    smallest := sides.smallest ^ 2 }

/-- Theorem stating the areas of three overlapping squares given specific conditions -/
theorem overlapping_squares_areas :
  ∀ (sides : SquareSides),
    sides.largest = sides.middle + 1 →
    sides.largest = sides.smallest + 2 →
    (sides.largest - 1) * (sides.middle - 1) = 100 →
    (sides.middle - 1) * (sides.smallest - 1) = 64 →
    calculateAreas sides = { largest := 361, middle := 324, smallest := 289 } := by
  sorry


end overlapping_squares_areas_l855_85542


namespace complex_division_simplification_l855_85585

theorem complex_division_simplification :
  (1 - 2 * Complex.I) / (1 + Complex.I) = -1/2 - 3/2 * Complex.I := by
  sorry

end complex_division_simplification_l855_85585


namespace min_operations_to_300_l855_85579

def Calculator (n : ℕ) : Set ℕ :=
  { m | ∃ (ops : List (ℕ → ℕ)), 
    (∀ op ∈ ops, op = (· + 1) ∨ op = (· * 2)) ∧
    ops.foldl (λ acc f => f acc) 1 = m ∧
    ops.length = n }

theorem min_operations_to_300 :
  (∀ n < 11, 300 ∉ Calculator n) ∧ 300 ∈ Calculator 11 :=
sorry

end min_operations_to_300_l855_85579


namespace least_four_digit_11_heavy_l855_85517

def is_11_heavy (n : ℕ) : Prop := n % 11 > 7

theorem least_four_digit_11_heavy : 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 1000 → ¬(is_11_heavy m)) ∧ is_11_heavy 1000 :=
sorry

end least_four_digit_11_heavy_l855_85517


namespace robs_reading_l855_85526

/-- Given Rob's planned reading time, actual reading time as a fraction of planned time,
    and his reading speed, calculate the number of pages he read. -/
theorem robs_reading (planned_hours : ℝ) (actual_fraction : ℝ) (pages_per_minute : ℝ) : 
  planned_hours = 3 →
  actual_fraction = 3/4 →
  pages_per_minute = 1/15 →
  (planned_hours * actual_fraction * 60) * pages_per_minute = 9 := by
  sorry

end robs_reading_l855_85526


namespace sin_graph_shift_l855_85534

theorem sin_graph_shift (x : ℝ) :
  3 * Real.sin (2 * (x - π / 10)) = 3 * Real.sin (2 * x - π / 5) := by sorry

end sin_graph_shift_l855_85534


namespace parabola_symmetry_transform_l855_85582

/-- Given a parabola with equation y = -2(x+1)^2 + 3, prove that its transformation
    by symmetry about the line y = 1 results in the equation y = 2(x+1)^2 - 1. -/
theorem parabola_symmetry_transform (x y : ℝ) :
  (y = -2 * (x + 1)^2 + 3) →
  (∃ (y' : ℝ), y' = 2 * (x + 1)^2 - 1 ∧ 
    (∀ (p q : ℝ × ℝ), (p.2 = -2 * (p.1 + 1)^2 + 3 ∧ q.2 = y') → 
      (p.1 = q.1 ∧ p.2 + q.2 = 2))) :=
by sorry

end parabola_symmetry_transform_l855_85582


namespace fraction_division_problem_solution_l855_85570

theorem fraction_division (a b c d : ℚ) (hb : b ≠ 0) (hd : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem problem_solution : (5 / 3) / (8 / 15) = 25 / 8 :=
by
  -- Apply the fraction division theorem
  have h1 : (5 / 3) / (8 / 15) = (5 * 15) / (3 * 8) := by sorry
  
  -- Simplify the numerator and denominator
  have h2 : (5 * 15) / (3 * 8) = 75 / 24 := by sorry
  
  -- Further simplify the fraction
  have h3 : 75 / 24 = 25 / 8 := by sorry
  
  -- Combine the steps
  sorry

end fraction_division_problem_solution_l855_85570


namespace tan_alpha_plus_pi_over_four_l855_85543

theorem tan_alpha_plus_pi_over_four (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 := by
  sorry

end tan_alpha_plus_pi_over_four_l855_85543


namespace factory_task_excess_l855_85565

theorem factory_task_excess (first_half : Rat) (second_half : Rat)
  (h1 : first_half = 2 / 3)
  (h2 : second_half = 3 / 5) :
  first_half + second_half - 1 = 4 / 15 := by
  sorry

end factory_task_excess_l855_85565


namespace division_problem_l855_85589

theorem division_problem : (72 : ℚ) / ((6 : ℚ) / 3) = 36 := by
  sorry

end division_problem_l855_85589


namespace min_values_l855_85512

def min_value_exponential (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 2*y = 1 → 2^x + 4^y ≥ 2*Real.sqrt 2

def min_value_reciprocal (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 2*y = 1 → 1/x + 2/y ≥ 9

def min_value_squared (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x + 2*y = 1 → x^2 + 4*y^2 ≥ 1/2

theorem min_values (x y : ℝ) :
  min_value_exponential x y ∧
  min_value_reciprocal x y ∧
  min_value_squared x y :=
sorry

end min_values_l855_85512


namespace cubic_roots_sum_cubes_l855_85506

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (x^3 - 5*x^2 + 13*x - 7 = (x - a) * (x - b) * (x - c)) → 
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 490 := by
  sorry

end cubic_roots_sum_cubes_l855_85506


namespace zoo_trip_theorem_l855_85560

/-- Calculates the remaining money for lunch and snacks after a zoo trip for two people -/
def zoo_trip_remaining_money (ticket_price : ℚ) (bus_fare_one_way : ℚ) (total_money : ℚ) : ℚ :=
  let num_people : ℚ := 2
  let total_ticket_cost := ticket_price * num_people
  let total_bus_fare := bus_fare_one_way * num_people * 2
  let total_trip_cost := total_ticket_cost + total_bus_fare
  total_money - total_trip_cost

theorem zoo_trip_theorem :
  zoo_trip_remaining_money 5 1.5 40 = 24 := by
  sorry

end zoo_trip_theorem_l855_85560


namespace probability_two_girls_l855_85586

theorem probability_two_girls (total_members : ℕ) (girl_members : ℕ) : 
  total_members = 15 → girl_members = 6 → 
  (Nat.choose girl_members 2 : ℚ) / (Nat.choose total_members 2 : ℚ) = 1 / 7 := by
  sorry

end probability_two_girls_l855_85586


namespace motion_equation_l855_85501

/-- The acceleration function -/
def a (t : ℝ) : ℝ := 6 * t - 2

/-- The velocity function -/
def v (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

/-- The position function -/
def s (t : ℝ) : ℝ := t^3 - t^2 + t

theorem motion_equation (t : ℝ) :
  (∀ t, deriv v t = a t) ∧
  (∀ t, deriv s t = v t) ∧
  v 0 = 1 ∧
  s 0 = 0 →
  s t = t^3 - t^2 + t :=
by
  sorry

end motion_equation_l855_85501


namespace good_quadruple_inequality_l855_85548

/-- A good quadruple is a set of positive integers (p, a, b, c) satisfying certain conditions. -/
structure GoodQuadruple where
  p : Nat
  a : Nat
  b : Nat
  c : Nat
  p_prime : Nat.Prime p
  p_odd : Odd p
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c
  div_ab : p ∣ (a * b + 1)
  div_bc : p ∣ (b * c + 1)
  div_ca : p ∣ (c * a + 1)

/-- The main theorem about good quadruples. -/
theorem good_quadruple_inequality (q : GoodQuadruple) :
  q.p + 2 ≤ (q.a + q.b + q.c) / 3 ∧
  (q.p + 2 = (q.a + q.b + q.c) / 3 ↔ q.a = 2 ∧ q.b = 2 + q.p ∧ q.c = 2 + 2 * q.p) :=
by sorry

end good_quadruple_inequality_l855_85548


namespace missing_figure_proof_l855_85535

theorem missing_figure_proof (x : ℝ) : (1.2 / 100) * x = 0.6 → x = 50 := by
  sorry

end missing_figure_proof_l855_85535


namespace jordans_weight_loss_l855_85571

/-- Calculates Jordan's final weight after 13 weeks of an exercise program --/
theorem jordans_weight_loss (initial_weight : ℕ) : 
  initial_weight = 250 →
  (initial_weight 
    - (3 * 4)  -- Weeks 1-4
    - 5        -- Week 5
    - (2 * 4)  -- Weeks 6-9
    + 3        -- Week 10
    - (4 * 3)) -- Weeks 11-13
  = 216 := by
  sorry

#check jordans_weight_loss

end jordans_weight_loss_l855_85571


namespace vectors_not_collinear_l855_85503

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the proposition
def proposition (a b : V) : Prop :=
  ∀ k₁ k₂ : ℝ, k₁ • a + k₂ • b = 0 → k₁ ≠ 0 ∧ k₂ ≠ 0

-- State the theorem
theorem vectors_not_collinear (a b : V) :
  proposition a b → ¬(∃ (t : ℝ), a = t • b ∨ b = t • a) :=
by sorry

end vectors_not_collinear_l855_85503


namespace sin_6phi_value_l855_85595

theorem sin_6phi_value (φ : ℝ) (h : Complex.exp (Complex.I * φ) = (3 + Complex.I * Real.sqrt 8) / 5) :
  Real.sin (6 * φ) = -396 * Real.sqrt 2 / 15625 := by
  sorry

end sin_6phi_value_l855_85595


namespace inequality_proof_l855_85544

theorem inequality_proof (p q : ℝ) (m n : ℕ) 
  (h_pos_p : p > 0) (h_pos_q : q > 0) (h_sum : p + q = 1) (h_pos_m : m > 0) (h_pos_n : n > 0) : 
  (1 - p^m)^n + (1 - q^n)^m ≥ 1 := by
  sorry

end inequality_proof_l855_85544


namespace distinct_collections_eq_33_l855_85515

/-- Represents the number of each letter in 'MATHEMATICS' -/
def letter_counts : Fin 26 → Nat :=
  fun i => match i with
  | 0  => 2  -- 'A'
  | 4  => 1  -- 'E'
  | 8  => 1  -- 'I'
  | 12 => 2  -- 'M'
  | 19 => 2  -- 'T'
  | 2  => 1  -- 'C'
  | 7  => 1  -- 'H'
  | 18 => 1  -- 'S'
  | _  => 0

/-- The total number of letters -/
def total_letters : Nat := 11

/-- The number of vowels that fall off -/
def vowels_off : Nat := 3

/-- The number of consonants that fall off -/
def consonants_off : Nat := 2

/-- Function to check if a letter is a vowel -/
def is_vowel (i : Fin 26) : Bool :=
  i = 0 ∨ i = 4 ∨ i = 8 ∨ i = 14 ∨ i = 20

/-- Function to calculate the number of distinct collections -/
noncomputable def distinct_collections : Nat :=
  sorry

/-- Theorem stating that the number of distinct collections is 33 -/
theorem distinct_collections_eq_33 : distinct_collections = 33 :=
  sorry

end distinct_collections_eq_33_l855_85515


namespace boat_speed_in_still_water_l855_85573

/-- The speed of a boat in still water, given downstream travel information and stream speed. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 6)
  (h2 : downstream_distance = 72)
  (h3 : downstream_time = 3.6) :
  downstream_distance / downstream_time - stream_speed = 14 := by
  sorry

end boat_speed_in_still_water_l855_85573


namespace largest_number_l855_85578

theorem largest_number (a b c d e : ℝ) : 
  a = 12345 + 1/5678 →
  b = 12345 - 1/5678 →
  c = 12345 * 1/5678 →
  d = 12345 / (1/5678) →
  e = 12345.5678 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end largest_number_l855_85578


namespace expansion_coefficient_equality_l855_85533

theorem expansion_coefficient_equality (n : ℕ+) : 
  (8 * (Nat.choose n 3)) = (8 * 2 * (Nat.choose n 1)) ↔ n = 5 := by
  sorry

end expansion_coefficient_equality_l855_85533


namespace kates_retirement_fund_l855_85593

/-- The initial value of Kate's retirement fund, given the current value and the decrease amount. -/
def initial_value (current_value decrease : ℕ) : ℕ := current_value + decrease

/-- Theorem stating that Kate's initial retirement fund value was $1472. -/
theorem kates_retirement_fund : initial_value 1460 12 = 1472 := by
  sorry

end kates_retirement_fund_l855_85593


namespace infinite_sum_of_square_and_prime_infinite_not_sum_of_square_and_prime_l855_85583

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define a function to check if a perfect square can be expressed as the sum of a perfect square and a prime
def is_sum_of_square_and_prime (n : ℕ) : Prop :=
  is_perfect_square n ∧ ∃ a b : ℕ, is_perfect_square a ∧ is_prime b ∧ n = a + b

-- Statement 1: The set of perfect squares that can be expressed as the sum of a perfect square and a prime number is infinite
theorem infinite_sum_of_square_and_prime :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_sum_of_square_and_prime n :=
sorry

-- Statement 2: The set of perfect squares that cannot be expressed as the sum of a perfect square and a prime number is infinite
theorem infinite_not_sum_of_square_and_prime :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧ is_perfect_square n ∧ ¬is_sum_of_square_and_prime n :=
sorry

end infinite_sum_of_square_and_prime_infinite_not_sum_of_square_and_prime_l855_85583


namespace ratio_of_300_to_2_l855_85540

theorem ratio_of_300_to_2 : 
  let certain_number := 300
  300 / 2 = 150 := by sorry

end ratio_of_300_to_2_l855_85540


namespace quadratic_inequality_all_reals_l855_85514

theorem quadratic_inequality_all_reals
  (a b c : ℝ) :
  (∀ x, (a / 3) * x^2 + 2 * b * x - c < 0) ↔ (a > 0 ∧ 4 * b^2 - (4 / 3) * a * c < 0) :=
by sorry

end quadratic_inequality_all_reals_l855_85514


namespace cost_system_correct_l855_85528

/-- Represents the cost of seedlings in yuan -/
def CostSystem (x y : ℝ) : Prop :=
  (4 * x + 3 * y = 180) ∧ (x - y = 10)

/-- The cost system correctly represents the seedling pricing scenario -/
theorem cost_system_correct (x y : ℝ) :
  (4 * x + 3 * y = 180) →
  (y = x - 10) →
  CostSystem x y :=
by sorry

end cost_system_correct_l855_85528


namespace tangent_line_slope_at_zero_l855_85584

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 1)

theorem tangent_line_slope_at_zero :
  let f' := deriv f
  f' 0 = 1 := by sorry

end tangent_line_slope_at_zero_l855_85584


namespace sufficient_condition_for_inequality_l855_85563

theorem sufficient_condition_for_inequality (a x : ℝ) : 
  (-2 < x ∧ x < -1) → (a > 2 → (a + x) * (1 + x) < 0) := by sorry

end sufficient_condition_for_inequality_l855_85563


namespace vector_sum_equality_l855_85574

theorem vector_sum_equality : 
  4 • ![-3, 6] + 3 • ![-2, 5] = ![-18, 39] := by
  sorry

end vector_sum_equality_l855_85574


namespace rectangular_garden_length_l855_85547

theorem rectangular_garden_length 
  (perimeter : ℝ) 
  (breadth : ℝ) 
  (h1 : perimeter = 1200) 
  (h2 : breadth = 240) : 
  2 * (breadth + (perimeter / 2 - breadth)) = perimeter := by
  sorry

end rectangular_garden_length_l855_85547
