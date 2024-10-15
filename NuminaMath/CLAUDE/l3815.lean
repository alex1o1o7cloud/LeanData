import Mathlib

namespace NUMINAMATH_CALUDE_shmacks_in_shneids_l3815_381553

-- Define the conversion rates
def shmacks_per_shick : ℚ := 5 / 2
def shicks_per_shure : ℚ := 3 / 5
def shures_per_shneid : ℚ := 2 / 9

-- Define the problem
def shneids_to_convert : ℚ := 6

-- Theorem to prove
theorem shmacks_in_shneids : 
  shneids_to_convert * shures_per_shneid * shicks_per_shure * shmacks_per_shick = 2 := by
  sorry

end NUMINAMATH_CALUDE_shmacks_in_shneids_l3815_381553


namespace NUMINAMATH_CALUDE_tree_planting_theorem_l3815_381569

/-- Represents the tree planting activity -/
structure TreePlanting where
  totalVolunteers : ℕ
  poplars : ℕ
  seaBuckthorns : ℕ
  poplarTime : ℚ
  seaBuckthornsTime1 : ℚ
  seaBuckthornsTime2 : ℚ
  transferredVolunteers : ℕ

/-- Calculates the optimal allocation and durations for the tree planting activity -/
def optimalAllocation (tp : TreePlanting) : 
  (ℕ × ℕ) × ℚ × ℚ :=
  sorry

/-- The theorem stating the correctness of the optimal allocation and durations -/
theorem tree_planting_theorem (tp : TreePlanting) 
  (h1 : tp.totalVolunteers = 52)
  (h2 : tp.poplars = 150)
  (h3 : tp.seaBuckthorns = 200)
  (h4 : tp.poplarTime = 2/5)
  (h5 : tp.seaBuckthornsTime1 = 1/2)
  (h6 : tp.seaBuckthornsTime2 = 2/3)
  (h7 : tp.transferredVolunteers = 6) :
  let (allocation, initialDuration, finalDuration) := optimalAllocation tp
  allocation = (20, 32) ∧ 
  initialDuration = 25/8 ∧
  finalDuration = 27/7 :=
sorry

end NUMINAMATH_CALUDE_tree_planting_theorem_l3815_381569


namespace NUMINAMATH_CALUDE_f_comp_three_roots_l3815_381556

/-- A quadratic function f(x) = x^2 + 6x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 6*x + c

/-- The composition of f with itself -/
def f_comp (c : ℝ) (x : ℝ) : ℝ := f c (f c x)

/-- Counts the number of distinct real roots of a function -/
noncomputable def count_distinct_roots (g : ℝ → ℝ) : ℕ := sorry

theorem f_comp_three_roots :
  ∃! c : ℝ, count_distinct_roots (f_comp c) = 3 ∧ c = (11 - Real.sqrt 13) / 2 := by sorry

end NUMINAMATH_CALUDE_f_comp_three_roots_l3815_381556


namespace NUMINAMATH_CALUDE_asterisk_replacement_l3815_381571

theorem asterisk_replacement : (60 / 20) * (60 / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_asterisk_replacement_l3815_381571


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l3815_381526

theorem sugar_solution_percentage (original_percentage : ℝ) (final_percentage : ℝ) : 
  original_percentage = 10 →
  final_percentage = 18 →
  ∃ (second_percentage : ℝ),
    second_percentage = 42 ∧
    (3/4 * original_percentage + 1/4 * second_percentage) / 100 = final_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l3815_381526


namespace NUMINAMATH_CALUDE_stratified_sampling_juniors_l3815_381523

theorem stratified_sampling_juniors 
  (total_students : ℕ) 
  (juniors : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1200)
  (h2 : juniors = 500)
  (h3 : sample_size = 120) :
  (juniors : ℚ) / total_students * sample_size = 50 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_juniors_l3815_381523


namespace NUMINAMATH_CALUDE_travel_ways_count_l3815_381522

/-- The number of available train trips -/
def train_trips : ℕ := 4

/-- The number of available ferry trips -/
def ferry_trips : ℕ := 3

/-- The total number of ways to travel from A to B -/
def total_ways : ℕ := train_trips + ferry_trips

theorem travel_ways_count : total_ways = 7 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_count_l3815_381522


namespace NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l3815_381586

theorem geese_percentage_among_non_swans :
  ∀ (total : ℝ) (geese swan heron duck : ℝ),
    geese / total = 0.35 →
    swan / total = 0.20 →
    heron / total = 0.15 →
    duck / total = 0.30 →
    total > 0 →
    geese / (total - swan) = 0.4375 :=
by
  sorry

end NUMINAMATH_CALUDE_geese_percentage_among_non_swans_l3815_381586


namespace NUMINAMATH_CALUDE_nh_not_equal_nk_l3815_381578

/-- A structure representing a point in a plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A structure representing a line in a plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Given two points, return the line passing through them -/
def line_through_points (p1 p2 : Point) : Line :=
  sorry

/-- Given a point and a line, return the perpendicular line passing through the point -/
def perpendicular_line (p : Point) (l : Line) : Line :=
  sorry

/-- Given two lines, return the angle between them in radians -/
def angle_between_lines (l1 l2 : Line) : ℝ :=
  sorry

/-- Given two points, return the distance between them -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Given three points A, B, C, return a point that is 1/3 of the way from A to B -/
def one_third_point (a b : Point) : Point :=
  sorry

theorem nh_not_equal_nk (h k y z : Point) :
  let hk : Line := line_through_points h k
  let yz : Line := line_through_points y z
  let n : Point := one_third_point y z
  let yh : Line := perpendicular_line y hk
  let zk : Line := line_through_points z k
  angle_between_lines hk zk = π / 4 →
  distance n h ≠ distance n k :=
sorry

end NUMINAMATH_CALUDE_nh_not_equal_nk_l3815_381578


namespace NUMINAMATH_CALUDE_cost_of_25_pencils_20_notebooks_l3815_381562

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- The pricing conditions for pencils and notebooks -/
axiom pricing_condition_1 : 9 * pencil_cost + 10 * notebook_cost = 5.45
axiom pricing_condition_2 : 7 * pencil_cost + 6 * notebook_cost = 3.67
axiom pricing_condition_3 : 20 * pencil_cost + 15 * notebook_cost = 10.00

/-- The theorem stating the cost of 25 pencils and 20 notebooks -/
theorem cost_of_25_pencils_20_notebooks : 
  25 * pencil_cost + 20 * notebook_cost = 12.89 := by sorry

end NUMINAMATH_CALUDE_cost_of_25_pencils_20_notebooks_l3815_381562


namespace NUMINAMATH_CALUDE_lucas_sandwich_problem_l3815_381524

/-- Luca's sandwich shop problem --/
theorem lucas_sandwich_problem (sandwich_price : ℝ) (discount_rate : ℝ) 
  (avocado_price : ℝ) (salad_price : ℝ) (total_bill : ℝ) 
  (h1 : sandwich_price = 8)
  (h2 : discount_rate = 1/4)
  (h3 : avocado_price = 1)
  (h4 : salad_price = 3)
  (h5 : total_bill = 12) :
  total_bill - (sandwich_price * (1 - discount_rate) + avocado_price + salad_price) = 2 := by
  sorry

#check lucas_sandwich_problem

end NUMINAMATH_CALUDE_lucas_sandwich_problem_l3815_381524


namespace NUMINAMATH_CALUDE_solution_approximation_l3815_381557

theorem solution_approximation : ∃ x : ℝ, 
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * x * 0.5)) = 2800.0000000000005 ∧ 
  abs (x - 0.225) < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_solution_approximation_l3815_381557


namespace NUMINAMATH_CALUDE_line_relationship_l3815_381573

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (intersecting : Line → Line → Prop)

-- State the theorem
theorem line_relationship (a b c : Line) 
  (h1 : skew a b) 
  (h2 : parallel c a) : 
  skew c b ∨ intersecting c b :=
sorry

end NUMINAMATH_CALUDE_line_relationship_l3815_381573


namespace NUMINAMATH_CALUDE_min_value_inequality_l3815_381589

theorem min_value_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d) :
  (a + b + c + d) * (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3815_381589


namespace NUMINAMATH_CALUDE_money_ratio_l3815_381567

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

end NUMINAMATH_CALUDE_money_ratio_l3815_381567


namespace NUMINAMATH_CALUDE_sam_total_pennies_l3815_381593

def initial_pennies : ℕ := 98
def found_pennies : ℕ := 93

theorem sam_total_pennies :
  initial_pennies + found_pennies = 191 := by
  sorry

end NUMINAMATH_CALUDE_sam_total_pennies_l3815_381593


namespace NUMINAMATH_CALUDE_triangle_number_puzzle_l3815_381510

theorem triangle_number_puzzle :
  ∀ (A B C D E F : ℕ),
    A ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    B ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    C ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    D ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    E ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    F ∈ ({1, 2, 3, 4, 5, 6} : Set ℕ) →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F →
    D + E + B = 14 →
    A + C + F = 6 →
    A = 1 ∧ B = 3 ∧ C = 2 ∧ D = 5 ∧ E = 6 ∧ F = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_number_puzzle_l3815_381510


namespace NUMINAMATH_CALUDE_nail_painting_problem_l3815_381575

theorem nail_painting_problem (total_nails purple_nails blue_nails : ℕ) 
  (h1 : total_nails = 20)
  (h2 : purple_nails = 6)
  (h3 : blue_nails = 8)
  (h4 : (blue_nails : ℚ) / total_nails - (striped_nails : ℚ) / total_nails = 1/10) :
  striped_nails = 6 :=
by
  sorry

#check nail_painting_problem

end NUMINAMATH_CALUDE_nail_painting_problem_l3815_381575


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3815_381563

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + a = 0 ∧ x₂^2 - 2*x₂ + a = 0) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l3815_381563


namespace NUMINAMATH_CALUDE_solution_comparison_l3815_381579

theorem solution_comparison (p p' q q' : ℝ) (hp : p ≠ 0) (hp' : p' ≠ 0) :
  (-q' / p > -q / p') ↔ (q' / p < q / p') :=
by sorry

end NUMINAMATH_CALUDE_solution_comparison_l3815_381579


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l3815_381506

theorem arithmetic_progression_sum (n : ℕ) : 
  (n ≥ 3 ∧ n ≤ 14) ↔ 
  (n : ℝ) / 2 * (2 * 25 + (n - 1) * (-3)) ≥ 66 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l3815_381506


namespace NUMINAMATH_CALUDE_stone_game_termination_and_uniqueness_l3815_381598

/-- Represents the state of stones on the infinite strip --/
def StoneConfiguration := Int → ℕ

/-- Represents a move on the strip --/
inductive Move
  | typeA (n : Int) : Move
  | typeB (n : Int) : Move

/-- Applies a move to a configuration --/
def applyMove (config : StoneConfiguration) (move : Move) : StoneConfiguration :=
  match move with
  | Move.typeA n => fun i =>
      if i = n - 1 || i = n then config i - 1
      else if i = n + 1 then config i + 1
      else config i
  | Move.typeB n => fun i =>
      if i = n then config i - 2
      else if i = n + 1 || i = n - 2 then config i + 1
      else config i

/-- Checks if a move is valid for a given configuration --/
def isValidMove (config : StoneConfiguration) (move : Move) : Prop :=
  match move with
  | Move.typeA n => config (n - 1) > 0 ∧ config n > 0
  | Move.typeB n => config n ≥ 2

/-- Checks if any move is possible for a given configuration --/
def canMove (config : StoneConfiguration) : Prop :=
  ∃ (move : Move), isValidMove config move

/-- The theorem to be proved --/
theorem stone_game_termination_and_uniqueness 
  (initial : StoneConfiguration) : 
  ∃! (final : StoneConfiguration), 
    (∃ (moves : List Move), (moves.foldl applyMove initial = final)) ∧ 
    ¬(canMove final) := by
  sorry

end NUMINAMATH_CALUDE_stone_game_termination_and_uniqueness_l3815_381598


namespace NUMINAMATH_CALUDE_subset_transitive_and_complement_subset_l3815_381541

variable {α : Type*}
variable (U : Set α)

theorem subset_transitive_and_complement_subset : 
  (∀ A B C : Set α, A ⊆ B → B ⊆ C → A ⊆ C) ∧ 
  (∀ A B : Set α, A ⊆ B → (U \ B) ⊆ (U \ A)) :=
sorry

end NUMINAMATH_CALUDE_subset_transitive_and_complement_subset_l3815_381541


namespace NUMINAMATH_CALUDE_lucas_siblings_product_l3815_381594

/-- A family with Lauren and Lucas as members -/
structure Family where
  lauren_sisters : ℕ
  lauren_brothers : ℕ
  lucas : Member

/-- A member of the family -/
inductive Member
  | Lauren
  | Lucas
  | OtherSister
  | OtherBrother

/-- The number of sisters Lucas has in the family -/
def lucas_sisters (f : Family) : ℕ :=
  f.lauren_sisters + 1

/-- The number of brothers Lucas has in the family -/
def lucas_brothers (f : Family) : ℕ :=
  f.lauren_brothers - 1

theorem lucas_siblings_product (f : Family) 
  (h1 : f.lauren_sisters = 4)
  (h2 : f.lauren_brothers = 7)
  (h3 : f.lucas = Member.Lucas) :
  lucas_sisters f * lucas_brothers f = 35 := by
  sorry

end NUMINAMATH_CALUDE_lucas_siblings_product_l3815_381594


namespace NUMINAMATH_CALUDE_preimage_of_three_l3815_381588

def A : Set ℝ := Set.univ
def B : Set ℝ := Set.univ

def f : ℝ → ℝ := fun x ↦ 2 * x - 1

theorem preimage_of_three (h : f 2 = 3) : 
  ∃ x ∈ A, f x = 3 ∧ x = 2 :=
sorry

end NUMINAMATH_CALUDE_preimage_of_three_l3815_381588


namespace NUMINAMATH_CALUDE_relationship_correctness_l3815_381532

theorem relationship_correctness :
  (∃ a b c : ℝ, (a > b ↔ a * c^2 > b * c^2) → False) ∧
  (∃ a b : ℝ, (a > b → 1 / a < 1 / b) → False) ∧
  (∃ a b c d : ℝ, (a > b ∧ b > 0 ∧ c > d → a / d > b / c) → False) ∧
  (∀ a b c : ℝ, a > b ∧ b > 0 → a^c < b^c) :=
by sorry


end NUMINAMATH_CALUDE_relationship_correctness_l3815_381532


namespace NUMINAMATH_CALUDE_seven_fifth_sum_minus_two_fifth_l3815_381591

theorem seven_fifth_sum_minus_two_fifth (n : ℕ) : 
  (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) + (7^5 : ℕ) - (2^5 : ℕ) = 6 * (7^5 : ℕ) - 32 := by
sorry

end NUMINAMATH_CALUDE_seven_fifth_sum_minus_two_fifth_l3815_381591


namespace NUMINAMATH_CALUDE_y_value_theorem_l3815_381512

theorem y_value_theorem (y : ℝ) :
  (y / 5) / 3 = 5 / (y / 3) → y = 15 ∨ y = -15 := by
  sorry

end NUMINAMATH_CALUDE_y_value_theorem_l3815_381512


namespace NUMINAMATH_CALUDE_right_triangle_acute_angles_l3815_381564

theorem right_triangle_acute_angles (θ : ℝ) : 
  θ = 27 → 
  90 + θ + (90 - θ) = 180 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_acute_angles_l3815_381564


namespace NUMINAMATH_CALUDE_stream_speed_l3815_381500

/-- Proves that the speed of a stream is 5 km/hr given the conditions of the boat problem -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 22 →
  distance = 135 →
  time = 5 →
  distance = (boat_speed + stream_speed) * time →
  stream_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l3815_381500


namespace NUMINAMATH_CALUDE_chemical_solution_concentration_l3815_381596

theorem chemical_solution_concentration 
  (initial_concentration : ℝ)
  (replacement_concentration : ℝ)
  (replaced_portion : ℝ)
  (h1 : initial_concentration = 0.85)
  (h2 : replacement_concentration = 0.3)
  (h3 : replaced_portion = 0.8181818181818182)
  (h4 : replaced_portion ≥ 0 ∧ replaced_portion ≤ 1) :
  let remaining_portion := 1 - replaced_portion
  let final_concentration := 
    (remaining_portion * initial_concentration + replaced_portion * replacement_concentration)
  final_concentration = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_chemical_solution_concentration_l3815_381596


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l3815_381590

/-- A line passing through a point and perpendicular to another line --/
structure PerpendicularLine where
  -- The point that the line passes through
  point : ℝ × ℝ
  -- The line that our line is perpendicular to, represented by its coefficients (a, b, c) in ax + by + c = 0
  perp_line : ℝ × ℝ × ℝ

/-- The equation of a line, represented by its coefficients (a, b, c) in ax + by + c = 0 --/
def LineEquation := ℝ × ℝ × ℝ

/-- Check if a point lies on a line given by its equation --/
def point_on_line (p : ℝ × ℝ) (l : LineEquation) : Prop :=
  let (x, y) := p
  let (a, b, c) := l
  a * x + b * y + c = 0

/-- Check if two lines are perpendicular --/
def perpendicular (l1 l2 : LineEquation) : Prop :=
  let (a1, b1, _) := l1
  let (a2, b2, _) := l2
  a1 * a2 + b1 * b2 = 0

/-- The main theorem --/
theorem perpendicular_line_equation (l : PerpendicularLine) :
  let given_line : LineEquation := (1, -2, -3)
  let result_line : LineEquation := (2, 1, -1)
  l.point = (-1, 3) ∧ perpendicular given_line (result_line) →
  point_on_line l.point result_line ∧ perpendicular given_line result_line :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l3815_381590


namespace NUMINAMATH_CALUDE_correct_operation_l3815_381552

theorem correct_operation (a b : ℝ) : -a^2*b + 2*a^2*b = a^2*b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3815_381552


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l3815_381536

/-- Represents a square quilt block -/
structure QuiltBlock where
  total_squares : Nat
  divided_squares : Nat
  shaded_triangles : Nat

/-- The fraction of a square covered by a shaded triangle -/
def triangle_coverage : Rat := 1/2

/-- Calculates the fraction of the quilt block that is shaded -/
def shaded_fraction (quilt : QuiltBlock) : Rat :=
  (quilt.shaded_triangles : Rat) * triangle_coverage / (quilt.total_squares : Rat)

/-- Theorem stating that the shaded fraction of the quilt block is 1/8 -/
theorem quilt_shaded_fraction :
  ∀ (quilt : QuiltBlock),
    quilt.total_squares = 16 ∧
    quilt.divided_squares = 4 ∧
    quilt.shaded_triangles = 4 →
    shaded_fraction quilt = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l3815_381536


namespace NUMINAMATH_CALUDE_couscous_shipment_l3815_381580

theorem couscous_shipment (first_shipment second_shipment num_dishes couscous_per_dish : ℕ)
  (h1 : first_shipment = 7)
  (h2 : second_shipment = 13)
  (h3 : num_dishes = 13)
  (h4 : couscous_per_dish = 5) :
  let total_used := num_dishes * couscous_per_dish
  let first_two_shipments := first_shipment + second_shipment
  total_used - first_two_shipments = 45 := by
    sorry

end NUMINAMATH_CALUDE_couscous_shipment_l3815_381580


namespace NUMINAMATH_CALUDE_min_students_in_math_club_l3815_381502

/-- Represents a math club with boys and girls -/
structure MathClub where
  boys : ℕ
  girls : ℕ

/-- The condition that more than 60% of students are boys -/
def moreThan60PercentBoys (club : MathClub) : Prop :=
  (club.boys : ℚ) / (club.boys + club.girls : ℚ) > 60 / 100

/-- The theorem stating the minimum number of students in the club -/
theorem min_students_in_math_club :
  ∀ (club : MathClub),
  moreThan60PercentBoys club →
  club.girls = 5 →
  club.boys + club.girls ≥ 13 :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_math_club_l3815_381502


namespace NUMINAMATH_CALUDE_simplify_fraction_l3815_381597

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  3 / (x - 1) + (x - 3) / (1 - x^2) = (2*x + 6) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3815_381597


namespace NUMINAMATH_CALUDE_sandwich_combinations_l3815_381504

theorem sandwich_combinations (meat : ℕ) (cheese : ℕ) (condiment : ℕ) :
  meat = 12 →
  cheese = 8 →
  condiment = 5 →
  (meat) * (cheese.choose 2) * (condiment) = 1680 :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l3815_381504


namespace NUMINAMATH_CALUDE_root_difference_square_range_l3815_381519

/-- Given a quadratic equation with two distinct real roots, 
    prove that the square of the difference of the roots has a specific range -/
theorem root_difference_square_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 - 2*a*x₁ + 2*a^2 - 3*a + 2 = 0 ∧
   x₂^2 - 2*a*x₂ + 2*a^2 - 3*a + 2 = 0) →
  ∃ y : ℝ, 0 < y ∧ y ≤ 1 ∧
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ →
    x₁^2 - 2*a*x₁ + 2*a^2 - 3*a + 2 = 0 →
    x₂^2 - 2*a*x₂ + 2*a^2 - 3*a + 2 = 0 →
    (x₁ - x₂)^2 = y :=
by sorry

end NUMINAMATH_CALUDE_root_difference_square_range_l3815_381519


namespace NUMINAMATH_CALUDE_not_iff_eq_mul_eq_irrational_iff_irrational_plus_five_not_gt_implies_sq_gt_lt_three_implies_lt_five_l3815_381501

-- Statement 1
theorem not_iff_eq_mul_eq (a b c : ℝ) : ¬(a = b ↔ a * c = b * c) :=
sorry

-- Statement 2
theorem irrational_iff_irrational_plus_five (a : ℝ) : Irrational (a + 5) ↔ Irrational a :=
sorry

-- Statement 3
theorem not_gt_implies_sq_gt (a b : ℝ) : ¬(a > b → a^2 > b^2) :=
sorry

-- Statement 4
theorem lt_three_implies_lt_five (a : ℝ) : a < 3 → a < 5 :=
sorry

end NUMINAMATH_CALUDE_not_iff_eq_mul_eq_irrational_iff_irrational_plus_five_not_gt_implies_sq_gt_lt_three_implies_lt_five_l3815_381501


namespace NUMINAMATH_CALUDE_sports_equipment_problem_l3815_381539

/-- Represents the purchase and selling prices of sports equipment -/
structure SportsPrices where
  tabletennis_purchase : ℝ
  badminton_purchase : ℝ
  tabletennis_sell : ℝ
  badminton_sell : ℝ

/-- Represents the number of sets and profit -/
structure SalesData where
  tabletennis_sets : ℝ
  profit : ℝ

/-- Theorem stating the conditions and results of the sports equipment problem -/
theorem sports_equipment_problem 
  (prices : SportsPrices)
  (sales : SalesData) :
  -- Conditions
  2 * prices.tabletennis_purchase + prices.badminton_purchase = 110 ∧
  4 * prices.tabletennis_purchase + 3 * prices.badminton_purchase = 260 ∧
  prices.tabletennis_sell = 50 ∧
  prices.badminton_sell = 60 ∧
  sales.tabletennis_sets ≤ 150 ∧
  sales.tabletennis_sets ≥ (300 - sales.tabletennis_sets) / 2 →
  -- Results
  prices.tabletennis_purchase = 35 ∧
  prices.badminton_purchase = 40 ∧
  sales.profit = -5 * sales.tabletennis_sets + 6000 ∧
  100 ≤ sales.tabletennis_sets ∧ sales.tabletennis_sets ≤ 150 ∧
  (∀ a : ℝ, 0 < a ∧ a < 10 →
    (a < 5 → sales.tabletennis_sets = 100) ∧
    (a > 5 → sales.tabletennis_sets = 150) ∧
    (a = 5 → sales.profit = 6000)) :=
by sorry

end NUMINAMATH_CALUDE_sports_equipment_problem_l3815_381539


namespace NUMINAMATH_CALUDE_bottle_production_l3815_381503

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 5 such machines will produce 900 bottles in 4 minutes. -/
theorem bottle_production 
  (machines : ℕ) 
  (bottles_per_minute : ℕ) 
  (h1 : machines = 6) 
  (h2 : bottles_per_minute = 270) : 
  (5 : ℕ) * (4 : ℕ) * (bottles_per_minute / machines) = 900 := by
  sorry


end NUMINAMATH_CALUDE_bottle_production_l3815_381503


namespace NUMINAMATH_CALUDE_ferris_wheel_broken_seats_l3815_381545

/-- The number of broken seats on a Ferris wheel -/
def broken_seats (total_seats : ℕ) (capacity_per_seat : ℕ) (current_capacity : ℕ) : ℕ :=
  total_seats - (current_capacity / capacity_per_seat)

/-- Theorem stating the number of broken seats on the Ferris wheel -/
theorem ferris_wheel_broken_seats :
  let total_seats : ℕ := 18
  let capacity_per_seat : ℕ := 15
  let current_capacity : ℕ := 120
  broken_seats total_seats capacity_per_seat current_capacity = 10 := by
  sorry

#eval broken_seats 18 15 120

end NUMINAMATH_CALUDE_ferris_wheel_broken_seats_l3815_381545


namespace NUMINAMATH_CALUDE_periodic_odd_function_at_one_l3815_381550

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_at_one (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_odd : is_odd f) : 
  f 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_periodic_odd_function_at_one_l3815_381550


namespace NUMINAMATH_CALUDE_total_copies_is_7050_l3815_381527

/-- The total number of copies made by four copy machines in 30 minutes -/
def total_copies : ℕ :=
  let machine1 := 35 * 30
  let machine2 := 65 * 30
  let machine3 := 50 * 15 + 80 * 15
  let machine4 := 90 * 10 + 60 * 20
  machine1 + machine2 + machine3 + machine4

/-- Theorem stating that the total number of copies made by the four machines in 30 minutes is 7050 -/
theorem total_copies_is_7050 : total_copies = 7050 := by
  sorry

end NUMINAMATH_CALUDE_total_copies_is_7050_l3815_381527


namespace NUMINAMATH_CALUDE_constant_value_proof_l3815_381576

theorem constant_value_proof (f : ℝ → ℝ) (c : ℝ) 
  (h1 : ∀ x, f x + 3 * f (c - x) = x) 
  (h2 : f 2 = 2) : 
  c = 8 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_proof_l3815_381576


namespace NUMINAMATH_CALUDE_equal_integers_from_equation_l3815_381582

/-- The least prime divisor of a positive integer greater than 1 -/
def least_prime_divisor (m : ℕ) : ℕ :=
  Nat.minFac m

theorem equal_integers_from_equation (a b : ℕ) 
  (ha : a > 1) (hb : b > 1)
  (h : a^2 + b = least_prime_divisor a + (least_prime_divisor b)^2) :
  a = b :=
sorry

end NUMINAMATH_CALUDE_equal_integers_from_equation_l3815_381582


namespace NUMINAMATH_CALUDE_quadratic_other_root_l3815_381542

theorem quadratic_other_root 
  (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : a * 2^2 = b) : 
  a * (-2)^2 = b := by
sorry

end NUMINAMATH_CALUDE_quadratic_other_root_l3815_381542


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3815_381584

theorem rectangle_circle_area_ratio 
  (l w r : ℝ) 
  (h1 : l = 2 * w) 
  (h2 : 2 * l + 2 * w = 2 * Real.pi * r) : 
  (l * w) / (Real.pi * r^2) = 2 * Real.pi / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3815_381584


namespace NUMINAMATH_CALUDE_radical_product_simplification_l3815_381540

theorem radical_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (21 * q) :=
by sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l3815_381540


namespace NUMINAMATH_CALUDE_sliced_meat_variety_pack_l3815_381577

theorem sliced_meat_variety_pack :
  let base_cost : ℚ := 40
  let rush_delivery_rate : ℚ := 0.3
  let cost_per_type : ℚ := 13
  let total_cost : ℚ := base_cost * (1 + rush_delivery_rate)
  (total_cost / cost_per_type : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_sliced_meat_variety_pack_l3815_381577


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3815_381528

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3815_381528


namespace NUMINAMATH_CALUDE_cannot_make_105_with_5_coins_l3815_381572

def coin_denominations : List ℕ := [1, 5, 10, 25, 50]

def is_valid_sum (sum : ℕ) (n : ℕ) : Prop :=
  ∃ (coins : List ℕ), 
    coins.all (λ c => c ∈ coin_denominations) ∧ 
    coins.length = n ∧
    coins.sum = sum

theorem cannot_make_105_with_5_coins : 
  ¬ (is_valid_sum 105 5) :=
sorry

end NUMINAMATH_CALUDE_cannot_make_105_with_5_coins_l3815_381572


namespace NUMINAMATH_CALUDE_smallest_w_l3815_381533

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w : 
  let w := 2571912
  ∀ x : ℕ, x > 0 →
    (is_factor (2^5) (3692 * x) ∧
     is_factor (3^4) (3692 * x) ∧
     is_factor (7^3) (3692 * x) ∧
     is_factor (17^2) (3692 * x)) →
    x ≥ w :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l3815_381533


namespace NUMINAMATH_CALUDE_cucumber_water_percentage_l3815_381558

/-- Given the initial and final conditions of cucumbers after water evaporation,
    prove that the initial water percentage was 99%. -/
theorem cucumber_water_percentage
  (initial_weight : ℝ)
  (final_water_percentage : ℝ)
  (final_weight : ℝ)
  (h_initial_weight : initial_weight = 100)
  (h_final_water_percentage : final_water_percentage = 96)
  (h_final_weight : final_weight = 25) :
  (initial_weight - (1 - final_water_percentage / 100) * final_weight) / initial_weight * 100 = 99 :=
by sorry

end NUMINAMATH_CALUDE_cucumber_water_percentage_l3815_381558


namespace NUMINAMATH_CALUDE_repeating_decimal_eq_fraction_l3815_381505

/-- The repeating decimal 0.4̅5̅6̅ as a rational number -/
def repeating_decimal : ℚ := 0.4 + (56 : ℚ) / 990

/-- The fraction 226/495 -/
def fraction : ℚ := 226 / 495

/-- Theorem stating that the repeating decimal 0.4̅5̅6̅ is equal to the fraction 226/495 -/
theorem repeating_decimal_eq_fraction : repeating_decimal = fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_eq_fraction_l3815_381505


namespace NUMINAMATH_CALUDE_complex_magnitude_l3815_381544

theorem complex_magnitude (z : ℂ) : z = (2 + Complex.I) / Complex.I + Complex.I → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l3815_381544


namespace NUMINAMATH_CALUDE_man_and_son_work_time_l3815_381570

/-- Given a task that takes a man 5 days and his son 20 days to complete individually,
    prove that they can complete the task together in 4 days. -/
theorem man_and_son_work_time (task : ℝ) (man_rate son_rate combined_rate : ℝ) : 
  task > 0 ∧ 
  man_rate = task / 5 ∧ 
  son_rate = task / 20 ∧ 
  combined_rate = man_rate + son_rate →
  task / combined_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_man_and_son_work_time_l3815_381570


namespace NUMINAMATH_CALUDE_art_supply_sales_percentage_l3815_381551

theorem art_supply_sales_percentage (total_percentage brush_percentage paint_percentage : ℝ) :
  total_percentage = 100 ∧
  brush_percentage = 45 ∧
  paint_percentage = 22 →
  total_percentage - brush_percentage - paint_percentage = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_art_supply_sales_percentage_l3815_381551


namespace NUMINAMATH_CALUDE_pure_imaginary_modulus_l3815_381565

theorem pure_imaginary_modulus (b : ℝ) : 
  (Complex.I : ℂ).re * ((2 + b * Complex.I) * (2 - Complex.I)).re = 0 ∧ 
  (Complex.I : ℂ).im * ((2 + b * Complex.I) * (2 - Complex.I)).im ≠ 0 → 
  Complex.abs (1 + b * Complex.I) = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_modulus_l3815_381565


namespace NUMINAMATH_CALUDE_visitor_decrease_l3815_381568

theorem visitor_decrease (P V : ℝ) (h1 : P > 0) (h2 : V > 0) : 
  let R := P * V
  let P' := 1.5 * P
  let R' := 1.2 * R
  ∃ V', R' = P' * V' ∧ V' = 0.8 * V :=
by sorry

end NUMINAMATH_CALUDE_visitor_decrease_l3815_381568


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_B_l3815_381525

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = |x| - 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem A_intersect_B_eq_B : A ∩ B = B := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_B_l3815_381525


namespace NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_of_a_for_two_distinct_fixed_points_l3815_381509

def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + b - 1

def is_fixed_point (f : ℝ → ℝ) (x : ℝ) : Prop := f x = x

theorem fixed_points_for_specific_values :
  is_fixed_point (f 1 (-2)) (-1) ∧ is_fixed_point (f 1 (-2)) 3 :=
sorry

theorem range_of_a_for_two_distinct_fixed_points :
  (∀ b : ℝ, ∃ x y : ℝ, x ≠ y ∧ is_fixed_point (f a b) x ∧ is_fixed_point (f a b) y) →
  (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_fixed_points_for_specific_values_range_of_a_for_two_distinct_fixed_points_l3815_381509


namespace NUMINAMATH_CALUDE_coefficient_x5y3_in_binomial_expansion_l3815_381521

theorem coefficient_x5y3_in_binomial_expansion :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (1 : ℕ)^(8 - k) * (1 : ℕ)^k) = 256 ∧
  Nat.choose 8 3 = 56 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x5y3_in_binomial_expansion_l3815_381521


namespace NUMINAMATH_CALUDE_fraction_equality_implies_values_l3815_381535

theorem fraction_equality_implies_values (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 4 →
    (B * x - 17) / (x^2 - 7*x + 12) = A / (x - 3) + 4 / (x - 4)) →
  A = 5/4 ∧ B = 21/4 ∧ A + B = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_values_l3815_381535


namespace NUMINAMATH_CALUDE_special_polynomial_max_value_l3815_381534

/-- A polynomial with integer coefficients satisfying certain conditions -/
def SpecialPolynomial (p : ℤ → ℤ) : Prop :=
  (∀ m n : ℤ, (p m - p n) ∣ (m^2 - n^2)) ∧ 
  p 0 = 1 ∧ 
  p 1 = 2

/-- The maximum value of p(100) for a SpecialPolynomial p is 10001 -/
theorem special_polynomial_max_value : 
  ∀ p : ℤ → ℤ, SpecialPolynomial p → p 100 ≤ 10001 :=
by sorry

end NUMINAMATH_CALUDE_special_polynomial_max_value_l3815_381534


namespace NUMINAMATH_CALUDE_parabola_vertex_sum_max_l3815_381513

theorem parabola_vertex_sum_max (a T : ℤ) (h_T : T ≠ 0) : 
  let parabola (x y : ℝ) := ∃ b c : ℝ, y = a * x^2 + b * x + c
  let passes_through (x y : ℝ) := parabola x y
  let M := let x_v := 3 * T / 2
            let y_v := -3 * a * T^2 / 4
            x_v + y_v
  (passes_through 0 0) ∧ 
  (passes_through (3 * T) 0) ∧
  (passes_through (3 * T + 1) 35) →
  ∀ m : ℝ, M ≤ m → m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_sum_max_l3815_381513


namespace NUMINAMATH_CALUDE_mango_rate_calculation_l3815_381547

/-- The rate per kg of mangoes given the purchase details --/
def mango_rate (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (total_payment : ℕ) : ℕ :=
  (total_payment - grape_quantity * grape_rate) / mango_quantity

theorem mango_rate_calculation :
  mango_rate 9 70 9 1125 = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_calculation_l3815_381547


namespace NUMINAMATH_CALUDE_prob_red_two_cans_l3815_381560

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

end NUMINAMATH_CALUDE_prob_red_two_cans_l3815_381560


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l3815_381574

theorem inverse_proportion_k_value (k : ℝ) (y x : ℝ → ℝ) :
  (∀ x, y x = k / x) →  -- y is an inverse proportion function of x
  k < 0 →  -- k is negative
  (∀ x, 1 ≤ x → x ≤ 3 → y x ≤ y 1 ∧ y x ≥ y 3) →  -- y is decreasing on [1, 3]
  y 1 - y 3 = 4 →  -- difference between max and min values is 4
  k = -6 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l3815_381574


namespace NUMINAMATH_CALUDE_total_rulers_problem_solution_l3815_381543

/-- Given an initial number of rulers and a number of rulers added, 
    the total number of rulers is equal to the sum of the initial number and the added number. -/
theorem total_rulers (initial_rulers added_rulers : ℕ) :
  initial_rulers + added_rulers = initial_rulers + added_rulers := by sorry

/-- The specific case mentioned in the problem -/
theorem problem_solution :
  let initial_rulers : ℕ := 11
  let added_rulers : ℕ := 14
  initial_rulers + added_rulers = 25 := by sorry

end NUMINAMATH_CALUDE_total_rulers_problem_solution_l3815_381543


namespace NUMINAMATH_CALUDE_min_value_is_11_l3815_381549

-- Define the variables and constraints
def is_feasible (x y : ℝ) : Prop :=
  x * y - 3 ≥ 0 ∧ x - y ≥ 1 ∧ y ≥ 5

-- Define the objective function
def objective_function (x y : ℝ) : ℝ :=
  3 * x + 4 * y

-- Theorem statement
theorem min_value_is_11 :
  ∀ x y : ℝ, is_feasible x y →
  objective_function x y ≥ 11 ∧
  ∃ x₀ y₀ : ℝ, is_feasible x₀ y₀ ∧ objective_function x₀ y₀ = 11 :=
sorry

end NUMINAMATH_CALUDE_min_value_is_11_l3815_381549


namespace NUMINAMATH_CALUDE_town_population_l3815_381515

/-- 
Given a town with initial population P:
- 100 new people move in
- 400 of the original population move out
- The population is halved every year for 4 years
- After 4 years, the population is 60 people

Prove that the initial population P was 1260 people.
-/
theorem town_population (P : ℕ) : (P + 100 - 400) / (2^4 : ℕ) = 60 → P = 1260 := by
  sorry

end NUMINAMATH_CALUDE_town_population_l3815_381515


namespace NUMINAMATH_CALUDE_expression_evaluation_l3815_381538

theorem expression_evaluation (a b : ℤ) (h1 : a = -1) (h2 : b = 2) :
  (2*a + b) - 2*(3*a - 2*b) = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3815_381538


namespace NUMINAMATH_CALUDE_opposite_of_two_thirds_l3815_381595

theorem opposite_of_two_thirds :
  -(2 / 3 : ℚ) = -2 / 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_two_thirds_l3815_381595


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3815_381520

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l3815_381520


namespace NUMINAMATH_CALUDE_option2_more_cost_effective_l3815_381546

/-- The cost of a pair of badminton rackets in dollars -/
def racket_cost : ℕ := 100

/-- The cost of a box of shuttlecocks in dollars -/
def shuttlecock_cost : ℕ := 20

/-- The number of pairs of badminton rackets the school wants to buy -/
def racket_pairs : ℕ := 10

/-- The number of boxes of shuttlecocks the school wants to buy -/
def shuttlecock_boxes : ℕ := 60

/-- The cost of Option 1 in dollars -/
def option1_cost (x : ℕ) : ℕ := 20 * x + 800

/-- The cost of Option 2 in dollars -/
def option2_cost (x : ℕ) : ℕ := 18 * x + 900

/-- Theorem stating that Option 2 is more cost-effective when x = 60 -/
theorem option2_more_cost_effective :
  shuttlecock_boxes > 10 →
  option1_cost shuttlecock_boxes > option2_cost shuttlecock_boxes :=
by sorry

end NUMINAMATH_CALUDE_option2_more_cost_effective_l3815_381546


namespace NUMINAMATH_CALUDE_square_sum_theorem_l3815_381583

theorem square_sum_theorem (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 3) :
  a^2 + b^2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_theorem_l3815_381583


namespace NUMINAMATH_CALUDE_cage_cost_proof_l3815_381537

def snake_toy_cost : ℝ := 11.76
def dollar_found : ℝ := 1
def total_cost : ℝ := 26.3

theorem cage_cost_proof :
  total_cost - (snake_toy_cost + dollar_found) = 13.54 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_proof_l3815_381537


namespace NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l3815_381516

theorem divisibility_of_sum_of_powers (k : ℕ) : 
  Odd k → (∃ (n : ℕ), n = 9 * 7 * 4 * k ∧ 2018 ∣ (1 + 2^n + 3^n + 4^n)) :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_sum_of_powers_l3815_381516


namespace NUMINAMATH_CALUDE_option_2_cheaper_for_42_options_equal_at_45_unique_equality_at_45_l3815_381581

-- Define the ticket price
def ticket_price : ℕ := 30

-- Define the discount rates
def discount_rate_1 : ℚ := 0.2
def discount_rate_2 : ℚ := 0.1

-- Define the number of free tickets in Option 2
def free_tickets : ℕ := 5

-- Function to calculate cost for Option 1
def cost_option_1 (students : ℕ) : ℚ :=
  (students : ℚ) * ticket_price * (1 - discount_rate_1)

-- Function to calculate cost for Option 2
def cost_option_2 (students : ℕ) : ℚ :=
  ((students - free_tickets) : ℚ) * ticket_price * (1 - discount_rate_2)

-- Theorem 1: For 42 students, Option 2 is cheaper
theorem option_2_cheaper_for_42 : cost_option_2 42 < cost_option_1 42 := by sorry

-- Theorem 2: Both options are equal when there are 45 students
theorem options_equal_at_45 : cost_option_1 45 = cost_option_2 45 := by sorry

-- Theorem 3: 45 is the only number of students (> 40) where both options are equal
theorem unique_equality_at_45 :
  ∀ n : ℕ, n > 40 → cost_option_1 n = cost_option_2 n → n = 45 := by sorry

end NUMINAMATH_CALUDE_option_2_cheaper_for_42_options_equal_at_45_unique_equality_at_45_l3815_381581


namespace NUMINAMATH_CALUDE_cube_through_cube_l3815_381592

theorem cube_through_cube (a : ℝ) (h : a > 0) : ∃ (s : ℝ), s > a ∧ s = (2 * a * Real.sqrt 2) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_through_cube_l3815_381592


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l3815_381585

/-- 
Given a quadratic expression of the form 6x^2 + nx + 72, 
this theorem states that the largest value of n for which 
the expression can be factored as the product of two linear 
factors with integer coefficients is 433.
-/
theorem largest_n_for_factorization : 
  (∀ n : ℤ, ∃ a b c d : ℤ, 
    (6 * x^2 + n * x + 72 = (a * x + b) * (c * x + d)) → n ≤ 433) ∧ 
  (∃ a b c d : ℤ, 6 * x^2 + 433 * x + 72 = (a * x + b) * (c * x + d)) := by
sorry


end NUMINAMATH_CALUDE_largest_n_for_factorization_l3815_381585


namespace NUMINAMATH_CALUDE_system_solutions_l3815_381530

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x + y = (5 * x * y) / (1 + x * y) ∧
  y + z = (6 * y * z) / (1 + y * z) ∧
  z + x = (7 * z * x) / (1 + z * x)

-- Define the set of solutions
def solutions : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0),
   ((3 + Real.sqrt 5) / 2, 1, 2 + Real.sqrt 3),
   ((3 + Real.sqrt 5) / 2, 1, 2 - Real.sqrt 3),
   ((3 - Real.sqrt 5) / 2, 1, 2 + Real.sqrt 3),
   ((3 - Real.sqrt 5) / 2, 1, 2 - Real.sqrt 3)}

-- Theorem statement
theorem system_solutions :
  ∀ x y z, system x y z ↔ (x, y, z) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l3815_381530


namespace NUMINAMATH_CALUDE_choose_three_cooks_from_twelve_l3815_381566

theorem choose_three_cooks_from_twelve (n : Nat) (k : Nat) : Nat.choose 12 3 = 220 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_cooks_from_twelve_l3815_381566


namespace NUMINAMATH_CALUDE_remainder_of_product_divided_by_11_l3815_381511

theorem remainder_of_product_divided_by_11 : (108 * 110) % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_product_divided_by_11_l3815_381511


namespace NUMINAMATH_CALUDE_triangle_areas_l3815_381507

theorem triangle_areas (BD DC : ℝ) (area_ABD : ℝ) :
  BD / DC = 2 / 5 →
  area_ABD = 28 →
  ∃ (area_ADC area_ABC : ℝ),
    area_ADC = 70 ∧
    area_ABC = 98 :=
by sorry

end NUMINAMATH_CALUDE_triangle_areas_l3815_381507


namespace NUMINAMATH_CALUDE_urn_problem_l3815_381518

theorem urn_problem (w : ℕ) : 
  (10 : ℝ) / (10 + w) * 9 / (9 + w) = 0.4285714285714286 → w = 5 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l3815_381518


namespace NUMINAMATH_CALUDE_additional_decorations_to_buy_l3815_381555

def halloween_decorations (skulls broomsticks spiderwebs pumpkins cauldron left_to_put_up total : ℕ) : Prop :=
  skulls = 12 ∧
  broomsticks = 4 ∧
  spiderwebs = 12 ∧
  pumpkins = 2 * spiderwebs ∧
  cauldron = 1 ∧
  left_to_put_up = 10 ∧
  total = 83

theorem additional_decorations_to_buy 
  (skulls broomsticks spiderwebs pumpkins cauldron left_to_put_up total : ℕ)
  (h : halloween_decorations skulls broomsticks spiderwebs pumpkins cauldron left_to_put_up total) :
  total - (skulls + broomsticks + spiderwebs + pumpkins + cauldron) - left_to_put_up = 20 :=
sorry

end NUMINAMATH_CALUDE_additional_decorations_to_buy_l3815_381555


namespace NUMINAMATH_CALUDE_triangle_height_l3815_381559

theorem triangle_height (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  AB = 3 ∧ BC = Real.sqrt 13 ∧ AC = 4 →
  ∃ D : ℝ × ℝ, 
    (D.1 - A.1) * (C.1 - A.1) + (D.2 - A.2) * (C.2 - A.2) = 0 ∧
    Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 3/2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_l3815_381559


namespace NUMINAMATH_CALUDE_product_97_103_l3815_381587

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end NUMINAMATH_CALUDE_product_97_103_l3815_381587


namespace NUMINAMATH_CALUDE_book_pages_count_l3815_381529

theorem book_pages_count : 
  let total_chapters : ℕ := 31
  let first_ten_pages : ℕ := 61
  let middle_ten_pages : ℕ := 59
  let last_eleven_pages : List ℕ := [58, 65, 62, 63, 64, 57, 66, 60, 59, 67]
  
  (10 * first_ten_pages) + 
  (10 * middle_ten_pages) + 
  (last_eleven_pages.sum) = 1821 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l3815_381529


namespace NUMINAMATH_CALUDE_daily_savings_l3815_381599

def original_coffees : ℕ := 4
def original_price : ℚ := 2
def price_increase_percentage : ℚ := 50
def new_coffees_ratio : ℚ := 1/2

def original_spending : ℚ := original_coffees * original_price

def new_price : ℚ := original_price * (1 + price_increase_percentage / 100)
def new_coffees : ℚ := original_coffees * new_coffees_ratio
def new_spending : ℚ := new_coffees * new_price

theorem daily_savings : original_spending - new_spending = 2 := by sorry

end NUMINAMATH_CALUDE_daily_savings_l3815_381599


namespace NUMINAMATH_CALUDE_max_value_of_f_l3815_381561

noncomputable def f (x : ℝ) : ℝ := x^2 - 8*x + 6*Real.log x + 1

theorem max_value_of_f :
  ∃ (c : ℝ), c > 0 ∧ ∀ (x : ℝ), x > 0 → f x ≤ f c ∧ f c = -6 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l3815_381561


namespace NUMINAMATH_CALUDE_max_dimes_grace_l3815_381514

/-- The value of a dime in cents -/
def dime_value : ℚ := 10

/-- The value of a penny in cents -/
def penny_value : ℚ := 1

/-- The total amount Grace has in cents -/
def total_amount : ℚ := 480

theorem max_dimes_grace : 
  ∀ d : ℕ, d * (dime_value + penny_value) ≤ total_amount → d ≤ 43 :=
by sorry

end NUMINAMATH_CALUDE_max_dimes_grace_l3815_381514


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3815_381548

/-- A triangle with sides a, b, and c is isosceles if at least two sides are equal -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

/-- The triangle inequality theorem -/
def SatisfiesTriangleInequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ a + c > b

/-- The perimeter of a triangle -/
def Perimeter (a b c : ℝ) : ℝ :=
  a + b + c

/-- The unique isosceles triangle with sides 10 and 22 has perimeter 54 -/
theorem isosceles_triangle_perimeter :
  ∃! (a b c : ℝ),
    a = 10 ∧ (b = 22 ∨ c = 22) ∧
    IsIsosceles a b c ∧
    SatisfiesTriangleInequality a b c ∧
    Perimeter a b c = 54 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3815_381548


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3815_381554

theorem inequality_solution_range (b : ℝ) : 
  (∀ x : ℤ, |3 * (x : ℝ) - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) →
  (5 < b ∧ b < 7) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3815_381554


namespace NUMINAMATH_CALUDE_correct_equation_transformation_l3815_381517

theorem correct_equation_transformation (x : ℝ) : x - 1 = 4 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_transformation_l3815_381517


namespace NUMINAMATH_CALUDE_mask_production_in_july_l3815_381508

def initial_production : ℕ := 3000
def months_passed : ℕ := 4

theorem mask_production_in_july :
  initial_production * (2 ^ months_passed) = 48000 :=
by
  sorry

end NUMINAMATH_CALUDE_mask_production_in_july_l3815_381508


namespace NUMINAMATH_CALUDE_dumplings_remaining_l3815_381531

theorem dumplings_remaining (cooked : ℕ) (eaten : ℕ) (h1 : cooked = 14) (h2 : eaten = 7) :
  cooked - eaten = 7 := by
  sorry

end NUMINAMATH_CALUDE_dumplings_remaining_l3815_381531
