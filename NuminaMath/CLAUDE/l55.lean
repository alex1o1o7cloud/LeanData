import Mathlib

namespace x_minus_y_equals_twenty_l55_5558

theorem x_minus_y_equals_twenty (x y : ℝ) 
  (h1 : x * (y + 2) = 100) 
  (h2 : y * (x + 2) = 60) : 
  x - y = 20 := by
sorry

end x_minus_y_equals_twenty_l55_5558


namespace red_balls_per_box_l55_5520

theorem red_balls_per_box (total_balls : ℕ) (num_boxes : ℕ) (balls_per_box : ℕ) :
  total_balls = 10 →
  num_boxes = 5 →
  total_balls = num_boxes * balls_per_box →
  balls_per_box = 2 := by
sorry

end red_balls_per_box_l55_5520


namespace triangle_rotation_theorem_l55_5514

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  O : Point
  P : Point
  Q : Point

/-- Rotates a point -90° around the origin -/
def rotate90Clockwise (p : Point) : Point :=
  { x := p.y, y := -p.x }

theorem triangle_rotation_theorem (t : Triangle) :
  t.O = { x := 0, y := 0 } →
  t.Q = { x := 3, y := 0 } →
  t.P.x > 0 →
  t.P.y > 0 →
  (t.P.y - t.O.y) / (t.P.x - t.O.x) = 1 →
  (t.Q.x - t.O.x) * (t.P.x - t.Q.x) + (t.Q.y - t.O.y) * (t.P.y - t.Q.y) = 0 →
  rotate90Clockwise t.P = { x := 3, y := -3 } := by
  sorry

#check triangle_rotation_theorem

end triangle_rotation_theorem_l55_5514


namespace brittany_age_theorem_l55_5560

/-- Brittany's age when she returns from vacation -/
def brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ) : ℕ :=
  rebecca_age + age_difference + vacation_duration

/-- Theorem stating Brittany's age after vacation -/
theorem brittany_age_theorem (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ)
  (h1 : rebecca_age = 25)
  (h2 : age_difference = 3)
  (h3 : vacation_duration = 4) :
  brittany_age_after_vacation rebecca_age age_difference vacation_duration = 32 := by
  sorry

#check brittany_age_theorem

end brittany_age_theorem_l55_5560


namespace task_assignment_count_l55_5564

def number_of_ways (n m : ℕ) : ℕ :=
  Nat.choose n m

theorem task_assignment_count : 
  (number_of_ways 10 4) * (number_of_ways 4 2) * (number_of_ways 2 1) = 2520 := by
  sorry

end task_assignment_count_l55_5564


namespace jason_total_spent_l55_5571

/-- The amount Jason spent on the flute -/
def flute_cost : ℚ := 142.46

/-- The amount Jason spent on the music tool -/
def music_tool_cost : ℚ := 8.89

/-- The amount Jason spent on the song book -/
def song_book_cost : ℚ := 7

/-- The total amount Jason spent at the music store -/
def total_spent : ℚ := flute_cost + music_tool_cost + song_book_cost

/-- Theorem stating that the total amount Jason spent is $158.35 -/
theorem jason_total_spent : total_spent = 158.35 := by sorry

end jason_total_spent_l55_5571


namespace distance_between_points_on_lines_l55_5569

/-- Given two lines and points A and B on these lines, prove that the distance between A and B is 5 -/
theorem distance_between_points_on_lines (a : ℝ) :
  let line1 := λ (x y : ℝ) => 3 * a * x - y - 2 = 0
  let line2 := λ (x y : ℝ) => (2 * a - 1) * x + 3 * a * y - 3 = 0
  let A : ℝ × ℝ := (0, -2)
  let B : ℝ × ℝ := (-3, 2)
  line1 A.1 A.2 ∧ line2 B.1 B.2 →
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 5 := by
sorry


end distance_between_points_on_lines_l55_5569


namespace sequence_inequality_l55_5502

theorem sequence_inequality (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, a (n + 1) ≥ a n ^ 2 + 1/5) : 
  ∀ n : ℕ, n ≥ 5 → Real.sqrt (a (n + 5)) ≥ a (n - 5) ^ 2 := by
  sorry

end sequence_inequality_l55_5502


namespace contrapositive_equivalence_l55_5528

def original_proposition (x : ℝ) : Prop := x = 1 → x^2 - 3*x + 2 = 0

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) ↔ 
  (∀ x : ℝ, original_proposition x) :=
by sorry

end contrapositive_equivalence_l55_5528


namespace simplify_expression_l55_5542

theorem simplify_expression (x : ℝ) : 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x^2 + 4*x + 4) = |x - 2| + |x + 2| := by
  sorry

end simplify_expression_l55_5542


namespace gambler_outcome_l55_5562

def gamble (initial_amount : ℚ) (bet_sequence : List Bool) : ℚ :=
  bet_sequence.foldl
    (fun amount win =>
      if win then amount + amount / 2
      else amount - amount / 2)
    initial_amount

theorem gambler_outcome :
  let initial_amount : ℚ := 100
  let bet_sequence : List Bool := [true, false, true, false]
  let final_amount := gamble initial_amount bet_sequence
  final_amount = 56.25 ∧ initial_amount - final_amount = 43.75 := by
  sorry

end gambler_outcome_l55_5562


namespace inequality_solution_l55_5572

theorem inequality_solution (x : ℝ) : 
  (5 ≤ (x - 1) / (3 * x - 7) ∧ (x - 1) / (3 * x - 7) < 10) ↔ 
  (69 / 29 < x ∧ x ≤ 17 / 7) := by sorry

end inequality_solution_l55_5572


namespace solution_to_system_of_equations_l55_5585

theorem solution_to_system_of_equations :
  ∃ (x y : ℚ), 3 * x - 18 * y = 2 ∧ 4 * y - x = 6 ∧ x = -58/3 ∧ y = -10/3 := by
  sorry

end solution_to_system_of_equations_l55_5585


namespace train_distance_45_minutes_l55_5570

/-- Represents the distance traveled by a train in miles -/
def train_distance (time : ℕ) : ℕ :=
  (time / 2 : ℕ)

/-- Proves that a train traveling 1 mile every 2 minutes will cover 22 miles in 45 minutes -/
theorem train_distance_45_minutes : train_distance 45 = 22 := by
  sorry

end train_distance_45_minutes_l55_5570


namespace arithmetic_mean_reciprocals_first_five_primes_l55_5552

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem arithmetic_mean_reciprocals_first_five_primes :
  let reciprocals := first_five_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 2927 / 11550 := by
  sorry

end arithmetic_mean_reciprocals_first_five_primes_l55_5552


namespace odd_swaps_change_perm_l55_5591

/-- Represents a permutation of three elements -/
inductive Perm3
  | abc
  | acb
  | bac
  | bca
  | cab
  | cba

/-- Represents whether a permutation is "correct" or "incorrect" -/
def isCorrect (p : Perm3) : Bool :=
  match p with
  | Perm3.abc => true
  | Perm3.bca => true
  | Perm3.cab => true
  | _ => false

/-- Represents a single adjacent swap -/
def swap (p : Perm3) : Perm3 :=
  match p with
  | Perm3.abc => Perm3.acb
  | Perm3.acb => Perm3.abc
  | Perm3.bac => Perm3.bca
  | Perm3.bca => Perm3.bac
  | Perm3.cab => Perm3.cba
  | Perm3.cba => Perm3.cab

/-- Theorem: After an odd number of swaps, the permutation cannot be the same as the initial one -/
theorem odd_swaps_change_perm (n : Nat) (h : Odd n) (p : Perm3) :
  (n.iterate swap p) ≠ p :=
  sorry

#check odd_swaps_change_perm

end odd_swaps_change_perm_l55_5591


namespace bert_kangaroo_count_l55_5589

/-- The number of kangaroos Kameron has -/
def kameron_kangaroos : ℕ := 100

/-- The number of kangaroos Bert buys per day -/
def bert_daily_increase : ℕ := 2

/-- The number of days until Bert has the same number of kangaroos as Kameron -/
def days_until_equal : ℕ := 40

/-- The number of kangaroos Bert currently has -/
def bert_current_kangaroos : ℕ := 20

theorem bert_kangaroo_count :
  bert_current_kangaroos + bert_daily_increase * days_until_equal = kameron_kangaroos :=
sorry

end bert_kangaroo_count_l55_5589


namespace mans_age_twice_sons_l55_5544

/-- Proves that it takes 2 years for a man's age to be twice his son's age -/
theorem mans_age_twice_sons (son_age : ℕ) (age_difference : ℕ) : 
  son_age = 33 →
  age_difference = 35 →
  ∃ (years : ℕ), years = 2 ∧ 
    (son_age + age_difference + years) = 2 * (son_age + years) := by
  sorry

end mans_age_twice_sons_l55_5544


namespace garrison_provisions_l55_5580

theorem garrison_provisions (initial_men : ℕ) (initial_days : ℕ) (reinforcement : ℕ) (days_before_reinforcement : ℕ) : 
  initial_men = 2000 →
  initial_days = 54 →
  reinforcement = 1600 →
  days_before_reinforcement = 18 →
  let total_provisions := initial_men * initial_days
  let used_provisions := initial_men * days_before_reinforcement
  let remaining_provisions := total_provisions - used_provisions
  let total_men_after_reinforcement := initial_men + reinforcement
  (remaining_provisions / total_men_after_reinforcement : ℚ) = 20 := by
sorry

end garrison_provisions_l55_5580


namespace coin_jar_problem_l55_5546

theorem coin_jar_problem (y : ℕ) :
  (5 * y + 10 * y + 25 * y = 1440) → y = 36 :=
by sorry

end coin_jar_problem_l55_5546


namespace vector_problem_l55_5573

/-- Given vectors in R² -/
def a : Fin 2 → ℝ := ![4, 2]
def b : Fin 2 → ℝ := ![-1, 2]
def c (m : ℝ) : Fin 2 → ℝ := ![2, m]

/-- Dot product of two vectors in R² -/
def dot (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

/-- Parallel vectors in R² -/
def parallel (u v : Fin 2 → ℝ) : Prop := ∃ k : ℝ, u = fun i => k * (v i)

theorem vector_problem (m : ℝ) :
  (dot a (c m) < m^2 → (m > 4 ∨ m < -2)) ∧
  (parallel (fun i => a i + c m i) b → m = -14) := by
  sorry

end vector_problem_l55_5573


namespace chord_count_l55_5590

/-- The number of points on the circumference of a circle -/
def n : ℕ := 7

/-- The number of points needed to form a chord -/
def r : ℕ := 2

/-- The number of different chords that can be drawn -/
def num_chords : ℕ := n.choose r

theorem chord_count : num_chords = 21 := by
  sorry

end chord_count_l55_5590


namespace soccer_goal_ratio_l55_5593

/-- Prove the ratio of goals scored by The Spiders to The Kickers in the first period -/
theorem soccer_goal_ratio :
  let kickers_first := 2
  let kickers_second := 2 * kickers_first
  let spiders_second := 2 * kickers_second
  let total_goals := 15
  let spiders_first := total_goals - (kickers_first + kickers_second + spiders_second)
  (spiders_first : ℚ) / kickers_first = 1 / 2 := by
  sorry

end soccer_goal_ratio_l55_5593


namespace purple_shells_count_l55_5553

/-- Represents the number of shells of each color --/
structure ShellCounts where
  total : Nat
  pink : Nat
  yellow : Nat
  blue : Nat
  orange : Nat

/-- Theorem stating that the number of purple shells is 13 --/
theorem purple_shells_count (s : ShellCounts) 
  (h1 : s.total = 65)
  (h2 : s.pink = 8)
  (h3 : s.yellow = 18)
  (h4 : s.blue = 12)
  (h5 : s.orange = 14) :
  s.total - (s.pink + s.yellow + s.blue + s.orange) = 13 := by
  sorry

#check purple_shells_count

end purple_shells_count_l55_5553


namespace ellipse_focal_distance_l55_5519

-- Define the three given points
def p1 : ℝ × ℝ := (1, 5)
def p2 : ℝ × ℝ := (4, -3)
def p3 : ℝ × ℝ := (9, 5)

-- Define the ellipse type
structure Ellipse where
  endpoints : List (ℝ × ℝ)
  h_endpoints : endpoints.length = 3

-- Define the function to calculate the distance between foci
def focalDistance (e : Ellipse) : ℝ := sorry

-- Theorem statement
theorem ellipse_focal_distance :
  ∀ e : Ellipse, e.endpoints = [p1, p2, p3] → focalDistance e = 14 := by
  sorry

end ellipse_focal_distance_l55_5519


namespace f_max_value_when_a_eq_one_unique_root_f_eq_g_l55_5588

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x
def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (2*a + 1) * x + (a + 1) * Real.log x

-- Theorem for the maximum value of f when a = 1
theorem f_max_value_when_a_eq_one :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 1 y ≤ f 1 x ∧ f 1 x = -1 := by sorry

-- Theorem for the unique root of f(x) = g(x) when a ≥ 1
theorem unique_root_f_eq_g (a : ℝ) (h : a ≥ 1) :
  ∃! (x : ℝ), x > 0 ∧ f a x = g a x := by sorry

end

end f_max_value_when_a_eq_one_unique_root_f_eq_g_l55_5588


namespace planning_committee_subcommittees_l55_5532

theorem planning_committee_subcommittees (total_members : ℕ) (professor_count : ℕ) (subcommittee_size : ℕ) : 
  total_members = 12 →
  professor_count = 5 →
  subcommittee_size = 5 →
  (Nat.choose total_members subcommittee_size) - (Nat.choose (total_members - professor_count) subcommittee_size) = 771 :=
by sorry

end planning_committee_subcommittees_l55_5532


namespace ellipse_k_range_l55_5596

/-- An ellipse with foci on the y-axis is represented by the equation (x^2)/(15-k) + (y^2)/(k-9) = 1,
    where k is a real number. This theorem states that the range of k is (12, 15). -/
theorem ellipse_k_range (k : ℝ) :
  (∀ x y : ℝ, x^2 / (15 - k) + y^2 / (k - 9) = 1) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) →
  k ∈ Set.Ioo 12 15 :=
sorry

end ellipse_k_range_l55_5596


namespace train_speeds_equal_l55_5594

-- Define the speeds and times
def speed_A : ℝ := 110
def time_A_after_meeting : ℝ := 9
def time_B_after_meeting : ℝ := 4

-- Define the theorem
theorem train_speeds_equal :
  ∀ (speed_B : ℝ) (time_before_meeting : ℝ),
    speed_B > 0 →
    time_before_meeting > 0 →
    speed_A * time_before_meeting + speed_A * time_A_after_meeting =
    speed_B * time_before_meeting + speed_B * time_B_after_meeting →
    speed_A * time_before_meeting = speed_B * time_before_meeting →
    speed_B = speed_A :=
by
  sorry

#check train_speeds_equal

end train_speeds_equal_l55_5594


namespace volume_from_vessel_b_l55_5530

def vessel_a_concentration : ℝ := 0.45
def vessel_b_concentration : ℝ := 0.30
def vessel_c_concentration : ℝ := 0.10
def vessel_a_volume : ℝ := 4
def vessel_c_volume : ℝ := 6
def resultant_concentration : ℝ := 0.26

theorem volume_from_vessel_b (x : ℝ) : 
  vessel_a_concentration * vessel_a_volume + 
  vessel_b_concentration * x + 
  vessel_c_concentration * vessel_c_volume = 
  resultant_concentration * (vessel_a_volume + x + vessel_c_volume) → 
  x = 5 := by
sorry

end volume_from_vessel_b_l55_5530


namespace fourth_competitor_jump_distance_l55_5543

theorem fourth_competitor_jump_distance (first_jump second_jump third_jump fourth_jump : ℕ) :
  first_jump = 22 ∧
  second_jump = first_jump + 1 ∧
  third_jump = second_jump - 2 ∧
  fourth_jump = third_jump + 3 →
  fourth_jump = 24 := by
  sorry

end fourth_competitor_jump_distance_l55_5543


namespace fair_haired_men_nonmanagerial_percentage_l55_5523

/-- Represents the hair color distribution in the company -/
structure HairColorDistribution where
  fair : ℝ
  dark : ℝ
  red : ℝ
  ratio_fair_dark_red : fair / dark = 4 / 9 ∧ fair / red = 4 / 7

/-- Represents the gender distribution in the company -/
structure GenderDistribution where
  women : ℝ
  men : ℝ
  ratio_women_men : women / men = 3 / 5

/-- Represents the position distribution in the company -/
structure PositionDistribution where
  managerial : ℝ
  nonmanagerial : ℝ
  ratio_managerial_nonmanagerial : managerial / nonmanagerial = 1 / 4

/-- Represents the distribution of fair-haired employees -/
structure FairHairedDistribution where
  women_percentage : ℝ
  women_percentage_is_40 : women_percentage = 0.4
  women_managerial_percentage : ℝ
  women_managerial_percentage_is_60 : women_managerial_percentage = 0.6
  men_nonmanagerial_percentage : ℝ
  men_nonmanagerial_percentage_is_70 : men_nonmanagerial_percentage = 0.7

/-- Theorem: The percentage of fair-haired men in non-managerial positions is 42% -/
theorem fair_haired_men_nonmanagerial_percentage
  (hair : HairColorDistribution)
  (gender : GenderDistribution)
  (position : PositionDistribution)
  (fair_haired : FairHairedDistribution) :
  (1 - fair_haired.women_percentage) * fair_haired.men_nonmanagerial_percentage = 0.42 := by
  sorry

end fair_haired_men_nonmanagerial_percentage_l55_5523


namespace find_particular_number_l55_5598

theorem find_particular_number (x : ℤ) : x - 29 + 64 = 76 → x = 41 := by
  sorry

end find_particular_number_l55_5598


namespace divisibility_equivalence_l55_5533

theorem divisibility_equivalence (m n : ℕ+) : 
  (19 ∣ (11 * m.val + 2 * n.val)) ↔ (19 ∣ (18 * m.val + 5 * n.val)) := by
  sorry

end divisibility_equivalence_l55_5533


namespace geometric_sequence_property_l55_5581

theorem geometric_sequence_property (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_product : a 1 * a 7 = 36) : 
  a 4 = 6 := by
sorry

end geometric_sequence_property_l55_5581


namespace genuine_items_count_l55_5554

theorem genuine_items_count (total_purses total_handbags : ℕ) 
  (h1 : total_purses = 26)
  (h2 : total_handbags = 24)
  (h3 : total_purses / 2 + total_handbags / 4 = (total_purses + total_handbags) - 31) :
  31 = total_purses + total_handbags - (total_purses / 2 + total_handbags / 4) :=
by sorry

end genuine_items_count_l55_5554


namespace right_triangle_existence_and_uniqueness_l55_5559

theorem right_triangle_existence_and_uniqueness 
  (c d : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c > d) :
  ∃! (a b : ℝ), 
    a > 0 ∧ b > 0 ∧ 
    a^2 + b^2 = c^2 ∧ 
    a - b = d := by
  sorry

end right_triangle_existence_and_uniqueness_l55_5559


namespace box_surface_area_l55_5582

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle --/
def rectangleArea (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the cardboard sheet with cut corners --/
structure CutCardboard where
  sheet : Rectangle
  smallCutSize : ℕ
  largeCutSize : ℕ

/-- Calculates the surface area of the interior of the box formed from the cut cardboard --/
def interiorSurfaceArea (c : CutCardboard) : ℕ :=
  rectangleArea c.sheet -
  (2 * rectangleArea ⟨c.smallCutSize, c.smallCutSize⟩) -
  (2 * rectangleArea ⟨c.largeCutSize, c.largeCutSize⟩)

theorem box_surface_area :
  let cardboard : CutCardboard := {
    sheet := { length := 35, width := 25 },
    smallCutSize := 3,
    largeCutSize := 4
  }
  interiorSurfaceArea cardboard = 825 := by
  sorry

end box_surface_area_l55_5582


namespace tractors_moved_l55_5505

/-- Represents the farming field scenario -/
structure FarmingField where
  initialTractors : ℕ
  initialDays : ℕ
  initialHectaresPerDay : ℕ
  remainingTractors : ℕ
  remainingDays : ℕ
  remainingHectaresPerDay : ℕ

/-- The theorem stating the number of tractors moved -/
theorem tractors_moved (field : FarmingField)
  (h1 : field.initialTractors = 6)
  (h2 : field.initialDays = 4)
  (h3 : field.initialHectaresPerDay = 120)
  (h4 : field.remainingTractors = 4)
  (h5 : field.remainingDays = 5)
  (h6 : field.remainingHectaresPerDay = 144)
  (h7 : field.initialTractors * field.initialDays * field.initialHectaresPerDay =
        field.remainingTractors * field.remainingDays * field.remainingHectaresPerDay) :
  field.initialTractors - field.remainingTractors = 2 := by
  sorry

#check tractors_moved

end tractors_moved_l55_5505


namespace domain_of_f_l55_5515

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x + 3)) / (x^2 + 4*x + 3)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -3 < x ∧ x ≠ -1} :=
sorry

end domain_of_f_l55_5515


namespace system_solution_ratio_l55_5547

/-- Given a system of linear equations with a parameter m, prove that when the system has a nontrivial solution, the ratio of xz/y^2 is 20. -/
theorem system_solution_ratio (m : ℚ) (x y z : ℚ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 →
  x + m*y + 4*z = 0 →
  4*x + m*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  (∃ m, x + m*y + 4*z = 0 ∧ 4*x + m*y - 3*z = 0 ∧ 3*x + 5*y - 4*z = 0) →
  x*z / (y^2) = 20 := by
sorry

end system_solution_ratio_l55_5547


namespace joy_meets_grandma_l55_5575

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days until Joy sees her grandma -/
def days_until_meeting : ℕ := 2

/-- The time zone difference between Joy and her grandma in hours -/
def time_zone_difference : ℤ := 3

/-- The total number of hours until Joy sees her grandma -/
def total_hours : ℕ := hours_per_day * days_until_meeting

theorem joy_meets_grandma : total_hours = 48 := by sorry

end joy_meets_grandma_l55_5575


namespace problem_1_l55_5512

theorem problem_1 : 6 - (-12) / (-3) = 2 := by
  sorry

end problem_1_l55_5512


namespace power_function_through_point_l55_5548

/-- If f(x) = x^n is a power function and f(2) = √2, then f(4) = 2 -/
theorem power_function_through_point (n : ℝ) (f : ℝ → ℝ) : 
  (∀ x > 0, f x = x ^ n) →    -- f is a power function
  f 2 = Real.sqrt 2 →         -- f passes through (2, √2)
  f 4 = 2 := by               -- then f(4) = 2
sorry

end power_function_through_point_l55_5548


namespace sum_of_u_and_v_l55_5508

theorem sum_of_u_and_v (u v : ℚ) 
  (eq1 : 3 * u - 4 * v = 17) 
  (eq2 : 5 * u - 2 * v = 1) : 
  u + v = -8 := by
  sorry

end sum_of_u_and_v_l55_5508


namespace range_of_a_l55_5545

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x + 1 = 0) ∧ 
  (∀ x : ℝ, x^2 - 2*x + a > 0) →
  a ≥ 2 := by sorry

end range_of_a_l55_5545


namespace cube_opposite_face_l55_5550

-- Define the faces of the cube
inductive Face : Type
| A | B | C | D | E | F

-- Define the adjacency relation between faces
def adjacent : Face → Face → Prop := sorry

-- Define the opposite relation between faces
def opposite : Face → Face → Prop := sorry

-- Define the shares_vertex relation between faces
def shares_vertex : Face → Face → Prop := sorry

-- Define the shares_edge relation between faces
def shares_edge : Face → Face → Prop := sorry

theorem cube_opposite_face :
  -- Condition 2: Face B is adjacent to Face A
  adjacent Face.B Face.A →
  -- Condition 3: Face C and Face D are adjacent to each other
  adjacent Face.C Face.D →
  -- Condition 3: Face C shares a vertex with Face A
  shares_vertex Face.C Face.A →
  -- Condition 3: Face D shares a vertex with Face A
  shares_vertex Face.D Face.A →
  -- Condition 4: Face E and Face F share an edge with each other
  shares_edge Face.E Face.F →
  -- Condition 4: Face E does not share an edge with Face A
  ¬ shares_edge Face.E Face.A →
  -- Condition 4: Face F does not share an edge with Face A
  ¬ shares_edge Face.F Face.A →
  -- Conclusion: Face F is opposite to Face A
  opposite Face.F Face.A := by sorry

end cube_opposite_face_l55_5550


namespace ramesh_discount_percentage_l55_5510

/-- The discount percentage Ramesh received on the refrigerator --/
def discount_percentage (purchase_price transport_cost installation_cost no_discount_sale_price : ℚ) : ℚ :=
  let labelled_price := no_discount_sale_price / 1.1
  let discount := labelled_price - purchase_price
  (discount / labelled_price) * 100

/-- Theorem stating the discount percentage Ramesh received --/
theorem ramesh_discount_percentage :
  let purchase_price : ℚ := 14500
  let transport_cost : ℚ := 125
  let installation_cost : ℚ := 250
  let no_discount_sale_price : ℚ := 20350
  abs (discount_percentage purchase_price transport_cost installation_cost no_discount_sale_price - 21.62) < 0.01 := by
  sorry


end ramesh_discount_percentage_l55_5510


namespace microphotonics_budget_percentage_l55_5518

theorem microphotonics_budget_percentage 
  (total_degrees : ℝ)
  (home_electronics : ℝ)
  (food_additives : ℝ)
  (genetically_modified_microorganisms : ℝ)
  (industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : home_electronics = 24)
  (h3 : food_additives = 15)
  (h4 : genetically_modified_microorganisms = 29)
  (h5 : industrial_lubricants = 8)
  (h6 : basic_astrophysics_degrees = 43.2) : 
  (100 - (home_electronics + food_additives + genetically_modified_microorganisms + industrial_lubricants + (basic_astrophysics_degrees / total_degrees * 100))) = 12 := by
  sorry

end microphotonics_budget_percentage_l55_5518


namespace tilted_cube_segment_length_l55_5568

/-- Represents a tilted cube container with liquid -/
structure TiltedCube where
  edge_length : ℝ
  initial_fill_ratio : ℝ
  kb_length : ℝ
  lc_length : ℝ

/-- The length of segment LC in the tilted cube -/
def segment_lc_length (cube : TiltedCube) : ℝ := cube.lc_length

theorem tilted_cube_segment_length 
  (cube : TiltedCube)
  (h1 : cube.edge_length = 12)
  (h2 : cube.initial_fill_ratio = 5/8)
  (h3 : cube.lc_length = 2 * cube.kb_length)
  (h4 : cube.edge_length * (cube.initial_fill_ratio * cube.edge_length) = 
        (cube.lc_length + cube.kb_length) * cube.edge_length / 2) :
  segment_lc_length cube = 10 := by
sorry

end tilted_cube_segment_length_l55_5568


namespace constant_r_is_circle_l55_5506

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- A circle centered at the origin -/
def Circle (radius : ℝ) := {p : PolarPoint | p.r = radius}

/-- The set of points satisfying r = 5 in polar coordinates -/
def ConstantR : Set PolarPoint := {p : PolarPoint | p.r = 5}

/-- Theorem stating that ConstantR is a circle with radius 5 -/
theorem constant_r_is_circle : ConstantR = Circle 5 := by sorry

end constant_r_is_circle_l55_5506


namespace zero_not_in_range_of_g_l55_5583

-- Define the function g(x)
noncomputable def g : ℝ → ℤ
| x => if x > -1 then Int.ceil (1 / (x + 1))
       else if x < -1 then Int.floor (1 / (x + 1))
       else 0  -- arbitrary value for x = -1, as g is not defined there

-- Theorem statement
theorem zero_not_in_range_of_g : ∀ x : ℝ, g x ≠ 0 := by
  sorry

end zero_not_in_range_of_g_l55_5583


namespace no_function_satisfies_condition_l55_5536

theorem no_function_satisfies_condition :
  ∀ (f : ℝ → ℝ), ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ f (x + y^2) < f x + y :=
by sorry

end no_function_satisfies_condition_l55_5536


namespace triple_tilde_47_l55_5561

-- Define the tilde operation
def tilde (N : ℝ) : ℝ := 0.4 * N + 2

-- State the theorem
theorem triple_tilde_47 : tilde (tilde (tilde 47)) = 6.128 := by sorry

end triple_tilde_47_l55_5561


namespace inequality_proof_l55_5576

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3*b^2 + 5*c^2 ≤ 1 := by
sorry

end inequality_proof_l55_5576


namespace ratio_problem_l55_5504

theorem ratio_problem (second_part : ℝ) (percent : ℝ) (first_part : ℝ) : 
  second_part = 10 →
  percent = 20 →
  first_part / second_part = percent / 100 →
  first_part = 2 := by
sorry

end ratio_problem_l55_5504


namespace tank_dimension_proof_l55_5537

/-- Proves that for a rectangular tank with given dimensions and insulation cost,
    the third dimension is 3 feet. -/
theorem tank_dimension_proof (x : ℝ) : 
  let length : ℝ := 4
  let width : ℝ := 5
  let cost_per_sqft : ℝ := 20
  let total_cost : ℝ := 1880
  let surface_area : ℝ := 2 * (length * width + length * x + width * x)
  surface_area * cost_per_sqft = total_cost → x = 3 :=
by sorry

end tank_dimension_proof_l55_5537


namespace olivia_calculation_l55_5500

def round_to_nearest_ten (n : Int) : Int :=
  10 * ((n + 5) / 10)

theorem olivia_calculation : round_to_nearest_ten ((57 + 68) - 15) = 110 := by
  sorry

end olivia_calculation_l55_5500


namespace piano_lesson_discount_percentage_l55_5566

/-- Calculates the discount percentage on piano lessons given the piano cost, number of lessons,
    cost per lesson, and total cost after discount. -/
theorem piano_lesson_discount_percentage
  (piano_cost : ℝ)
  (num_lessons : ℕ)
  (cost_per_lesson : ℝ)
  (total_cost_after_discount : ℝ)
  (h1 : piano_cost = 500)
  (h2 : num_lessons = 20)
  (h3 : cost_per_lesson = 40)
  (h4 : total_cost_after_discount = 1100) :
  (1 - (total_cost_after_discount - piano_cost) / (num_lessons * cost_per_lesson)) * 100 = 25 :=
by sorry


end piano_lesson_discount_percentage_l55_5566


namespace inner_rectangle_area_l55_5525

theorem inner_rectangle_area (a b : ℕ) : 
  a > 2 → 
  b > 2 → 
  (3 * a + 4) * (b + 3) = 65 → 
  a * b = 3 :=
by sorry

end inner_rectangle_area_l55_5525


namespace local_extremum_and_minimum_l55_5555

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the derivative of f(x)
def f_prime (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem local_extremum_and_minimum (a b : ℝ) :
  (f a b 1 = 10) ∧ 
  (f_prime a b 1 = 0) →
  (a = 4 ∧ b = -11) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 3, f a b x ≥ 10) :=
by sorry

end local_extremum_and_minimum_l55_5555


namespace night_temperature_l55_5567

/-- Given the temperature changes throughout a day, prove the night temperature. -/
theorem night_temperature (morning_temp : ℝ) (noon_rise : ℝ) (night_drop : ℝ) :
  morning_temp = 22 →
  noon_rise = 6 →
  night_drop = 10 →
  morning_temp + noon_rise - night_drop = 18 := by
  sorry

end night_temperature_l55_5567


namespace sum_of_min_max_x_l55_5540

theorem sum_of_min_max_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  ∃ (m M : ℝ), (∀ x', ∃ y' z', x' + y' + z' = 5 ∧ x'^2 + y'^2 + z'^2 = 11 → m ≤ x' ∧ x' ≤ M) ∧
                m + M = 8/3 :=
sorry

end sum_of_min_max_x_l55_5540


namespace smallest_addition_for_divisibility_l55_5586

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ 751 * 503 - 1) ∧ 
  ((956734 + x) % (751 * 503) = 0) ∧
  ∀ y : ℕ, y < x → ((956734 + y) % (751 * 503) ≠ 0) :=
by sorry

end smallest_addition_for_divisibility_l55_5586


namespace perpendicular_tangents_exist_and_unique_l55_5578

/-- The line on which we search for the point. -/
def line (x y : ℝ) : Prop := 3 * x + 4 * y = 2

/-- The parabola to which we find tangents. -/
def parabola (x y : ℝ) : Prop := y = x^2

/-- A point is on a tangent line to the parabola. -/
def is_on_tangent (x y x₀ y₀ : ℝ) : Prop :=
  y = y₀ + 2 * x₀ * (x - x₀)

/-- Two lines are perpendicular. -/
def are_perpendicular (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1

/-- The theorem stating the existence and uniqueness of the point and its tangents. -/
theorem perpendicular_tangents_exist_and_unique :
  ∃! x₀ y₀ k₁ k₂,
    line x₀ y₀ ∧
    parabola x₀ y₀ ∧
    are_perpendicular (2 * x₀) (2 * x₀) ∧
    (∀ x y, is_on_tangent x y x₀ y₀ → (y = -1/4 + k₁ * (x - 1) ∨ y = -1/4 + k₂ * (x - 1))) ∧
    k₁ = 2 + Real.sqrt 5 ∧
    k₂ = 2 - Real.sqrt 5 :=
  sorry

end perpendicular_tangents_exist_and_unique_l55_5578


namespace no_solution_cosine_sine_equation_l55_5509

theorem no_solution_cosine_sine_equation :
  ∀ x : ℝ, Real.cos (Real.cos (Real.cos (Real.cos x))) > Real.sin (Real.sin (Real.sin (Real.sin x))) :=
by sorry

end no_solution_cosine_sine_equation_l55_5509


namespace kiwifruit_problem_l55_5538

-- Define the structure for weight difference and box count
structure WeightDifference :=
  (difference : ℝ)
  (count : ℕ)

-- Define the problem parameters
def standard_weight : ℝ := 25
def total_boxes : ℕ := 20
def selling_price_per_kg : ℝ := 10.6

-- Define the weight differences
def weight_differences : List WeightDifference := [
  ⟨-3, 1⟩, ⟨-2, 4⟩, ⟨-1.5, 2⟩, ⟨0, 3⟩, ⟨1, 2⟩, ⟨2.5, 8⟩
]

-- Calculate the total overweight
def total_overweight : ℝ :=
  weight_differences.foldr (λ wd acc => acc + wd.difference * wd.count) 0

-- Calculate the total weight
def total_weight : ℝ :=
  standard_weight * total_boxes + total_overweight

-- Calculate the total selling price
def total_selling_price : ℝ :=
  total_weight * selling_price_per_kg

-- Theorem to prove
theorem kiwifruit_problem :
  total_overweight = 8 ∧ total_selling_price = 5384.8 := by
  sorry


end kiwifruit_problem_l55_5538


namespace range_of_x_minus_2y_range_of_2a_plus_3b_l55_5535

-- Problem 1
theorem range_of_x_minus_2y (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 1) :
  ∃ (z : ℝ), -3 ≤ z ∧ z ≤ 2 ∧ ∃ (x' y' : ℝ), 
    (-1 ≤ x' ∧ x' ≤ 2) ∧ (0 ≤ y' ∧ y' ≤ 1) ∧ z = x' - 2 * y' :=
sorry

-- Problem 2
theorem range_of_2a_plus_3b (a b : ℝ) (hab1 : -1 < a + b ∧ a + b < 3) (hab2 : 2 < a - b ∧ a - b < 4) :
  ∃ (z : ℝ), -9/2 < z ∧ z < 13/2 ∧ ∃ (a' b' : ℝ), 
    (-1 < a' + b' ∧ a' + b' < 3) ∧ (2 < a' - b' ∧ a' - b' < 4) ∧ z = 2 * a' + 3 * b' :=
sorry

end range_of_x_minus_2y_range_of_2a_plus_3b_l55_5535


namespace cubic_odd_and_increasing_l55_5511

-- Define the function
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem cubic_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) :=
by sorry

end cubic_odd_and_increasing_l55_5511


namespace water_flow_speed_l55_5563

/-- The speed of the water flow given the ship's travel times and distances -/
theorem water_flow_speed (x y : ℝ) : 
  (135 / (x + y) + 70 / (x - y) = 12.5) →
  (75 / (x + y) + 110 / (x - y) = 12.5) →
  y = 3.2 := by
sorry

end water_flow_speed_l55_5563


namespace village_assistant_selection_l55_5531

theorem village_assistant_selection (n : ℕ) (k : ℕ) : 
  n = 10 → k = 3 → (Nat.choose 9 3) - (Nat.choose 7 3) = 49 := by
  sorry

end village_assistant_selection_l55_5531


namespace periodic_function_value_l55_5592

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem periodic_function_value (f : ℝ → ℝ) (h1 : is_periodic f 1.5) (h2 : f 1 = 20) :
  f 13 = 20 := by
  sorry

end periodic_function_value_l55_5592


namespace range_of_a_l55_5557

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x + 1 > 0) ∧ 
  (∀ x : ℝ, a*x^2 + x - 1 ≤ 0) → 
  -2 < a ∧ a ≤ -1/4 := by
sorry

end range_of_a_l55_5557


namespace jackie_eligible_for_free_shipping_l55_5551

def shampoo_price : ℝ := 12.50
def conditioner_price : ℝ := 15.00
def face_cream_price : ℝ := 20.00
def discount_rate : ℝ := 0.10
def free_shipping_threshold : ℝ := 75.00

def total_cost : ℝ := 2 * shampoo_price + 3 * conditioner_price + face_cream_price

def discounted_cost : ℝ := total_cost * (1 - discount_rate)

theorem jackie_eligible_for_free_shipping :
  discounted_cost ≥ free_shipping_threshold := by
  sorry

end jackie_eligible_for_free_shipping_l55_5551


namespace gear_rotation_problem_l55_5516

/-- The number of revolutions per minute of gear q -/
def q_rpm : ℝ := 40

/-- The time elapsed in minutes -/
def time : ℝ := 1.5

/-- The difference in revolutions between gears q and p after 90 seconds -/
def rev_diff : ℝ := 45

/-- The number of revolutions per minute of gear p -/
def p_rpm : ℝ := 10

theorem gear_rotation_problem :
  p_rpm * time + rev_diff = q_rpm * time := by sorry

end gear_rotation_problem_l55_5516


namespace sports_club_members_l55_5584

/-- Represents a sports club with members playing badminton and tennis -/
structure SportsClub where
  badminton : ℕ
  tennis : ℕ
  both : ℕ
  neither : ℕ

/-- Calculates the total number of members in the sports club -/
def total_members (club : SportsClub) : ℕ :=
  club.badminton + club.tennis - club.both + club.neither

/-- Theorem stating that the total number of members in the given sports club is 30 -/
theorem sports_club_members :
  ∃ (club : SportsClub),
    club.badminton = 17 ∧
    club.tennis = 19 ∧
    club.both = 9 ∧
    club.neither = 3 ∧
    total_members club = 30 :=
by
  sorry

end sports_club_members_l55_5584


namespace smallest_divisible_by_18_and_64_l55_5597

theorem smallest_divisible_by_18_and_64 : ∀ n : ℕ, n > 0 → n % 18 = 0 → n % 64 = 0 → n ≥ 576 := by
  sorry

end smallest_divisible_by_18_and_64_l55_5597


namespace intersection_of_A_and_complement_of_B_l55_5517

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x < 0}

-- Define set B
def B : Set ℝ := {x | x - 1 ≥ 0}

-- Theorem statement
theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end intersection_of_A_and_complement_of_B_l55_5517


namespace sin_cos_relation_l55_5507

theorem sin_cos_relation (α : Real) (h : Real.sin (π / 3 + α) = 1 / 3) :
  Real.cos (α - 7 * π / 6) = -1 / 3 := by
  sorry

end sin_cos_relation_l55_5507


namespace banana_permutations_eq_60_l55_5549

/-- The number of distinct permutations of the word "BANANA" -/
def banana_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of distinct permutations of "BANANA" is 60 -/
theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end banana_permutations_eq_60_l55_5549


namespace ceiling_negative_sqrt_fraction_l55_5541

theorem ceiling_negative_sqrt_fraction : ⌈-Real.sqrt (81 / 9)⌉ = -3 := by sorry

end ceiling_negative_sqrt_fraction_l55_5541


namespace solution_implies_a_equals_one_l55_5513

def f (x a : ℝ) : ℝ := |x - a| - 2

theorem solution_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, |f x a| < 1 ↔ (x ∈ Set.Ioo (-2) 0 ∨ x ∈ Set.Ioo 2 4)) →
  a = 1 := by
sorry

end solution_implies_a_equals_one_l55_5513


namespace geometric_sequence_sum_l55_5524

def geometric_sequence (n : ℕ) : ℝ := (-3) ^ (n - 1)

theorem geometric_sequence_sum :
  let a := geometric_sequence
  (a 1) + |a 2| + (a 3) + |a 4| + (a 5) = 121 := by
  sorry

end geometric_sequence_sum_l55_5524


namespace power_of_power_l55_5565

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end power_of_power_l55_5565


namespace octagon_placement_l55_5577

/-- A set of numbers from 1 to 12 -/
def CardSet : Set ℕ := {n | 1 ≤ n ∧ n ≤ 12}

/-- A function representing the placement of numbers on the octagon vertices -/
def Placement := Fin 8 → ℕ

/-- Predicate to check if a placement is valid according to the given conditions -/
def ValidPlacement (p : Placement) : Prop :=
  (∀ i, p i ∈ CardSet) ∧
  (∀ i, (p i + p ((i + 4) % 8)) % 3 = 0)

/-- The set of numbers not placed on the octagon -/
def NotPlaced (p : Placement) : Set ℕ := CardSet \ (Set.range p)

/-- Main theorem -/
theorem octagon_placement :
  ∀ p : Placement, ValidPlacement p → NotPlaced p = {3, 6, 9, 12} := by sorry

end octagon_placement_l55_5577


namespace parallel_line_k_value_l55_5529

/-- Two points in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of parallel to x-axis for a line segment -/
def parallelToXAxis (p1 p2 : Point2D) : Prop :=
  p1.y = p2.y

theorem parallel_line_k_value (A B : Point2D) (k : ℝ) 
    (hA : A = ⟨2, 3⟩) 
    (hB : B = ⟨4, k⟩) 
    (hParallel : parallelToXAxis A B) : 
  k = 3 := by
  sorry

end parallel_line_k_value_l55_5529


namespace minimum_explorers_l55_5539

theorem minimum_explorers (large_capacity small_capacity : ℕ) 
  (h1 : large_capacity = 24)
  (h2 : small_capacity = 9)
  (explorers : ℕ) :
  (∃ k : ℕ, explorers = k * large_capacity - 4) ∧
  (∃ m : ℕ, explorers = m * small_capacity - 4) →
  explorers ≥ 68 :=
by sorry

end minimum_explorers_l55_5539


namespace utensil_pack_composition_l55_5501

/-- Represents a pack of utensils -/
structure UtensilPack where
  knives : ℕ
  forks : ℕ
  spoons : ℕ
  total : knives + forks + spoons = 30

/-- Theorem about the composition of utensil packs -/
theorem utensil_pack_composition 
  (pack : UtensilPack) 
  (h : 5 * pack.spoons = 50) : 
  pack.spoons = 10 ∧ pack.knives + pack.forks = 20 := by
  sorry


end utensil_pack_composition_l55_5501


namespace at_least_one_prime_between_nfact_minus_n_and_nfact_l55_5579

theorem at_least_one_prime_between_nfact_minus_n_and_nfact (n : ℕ) (h : n > 2) :
  ∃ p : ℕ, Prime p ∧ n! - n < p ∧ p < n! :=
sorry

end at_least_one_prime_between_nfact_minus_n_and_nfact_l55_5579


namespace four_solutions_three_solutions_l55_5526

/-- The equation x^2 - 4|x| + k = 0 with integer k and x -/
def equation (k : ℤ) (x : ℤ) : Prop := x^2 - 4 * x.natAbs + k = 0

/-- The set of integer solutions to the equation -/
def solution_set (k : ℤ) : Set ℤ := {x : ℤ | equation k x}

theorem four_solutions :
  solution_set 3 = {1, -1, 3, -3} :=
sorry

theorem three_solutions :
  solution_set 0 = {0, 4, -4} :=
sorry

end four_solutions_three_solutions_l55_5526


namespace fraction_division_and_addition_l55_5595

theorem fraction_division_and_addition : 
  (5 / 6 : ℚ) / (9 / 10 : ℚ) + 1 / 15 = 402 / 405 := by
  sorry

end fraction_division_and_addition_l55_5595


namespace specific_right_triangle_perimeter_l55_5527

/-- A right triangle with integer side lengths, one of which is 11. -/
structure RightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  is_right : a^2 + b^2 = c^2
  has_eleven : a = 11 ∨ b = 11

/-- The perimeter of a right triangle. -/
def perimeter (t : RightTriangle) : ℕ := t.a + t.b + t.c

/-- Theorem stating that the perimeter of the specific right triangle is 132. -/
theorem specific_right_triangle_perimeter :
  ∃ t : RightTriangle, perimeter t = 132 :=
sorry

end specific_right_triangle_perimeter_l55_5527


namespace book_borrowing_growth_l55_5503

/-- The number of books borrowed in 2015 -/
def books_2015 : ℕ := 7500

/-- The number of books borrowed in 2017 -/
def books_2017 : ℕ := 10800

/-- The average annual growth rate from 2015 to 2017 -/
def growth_rate : ℝ := 0.2

/-- The expected number of books borrowed in 2018 -/
def books_2018 : ℕ := 12960

/-- Theorem stating the relationship between the given values and the calculated growth rate and expected books for 2018 -/
theorem book_borrowing_growth :
  (books_2017 : ℝ) = books_2015 * (1 + growth_rate)^2 ∧
  books_2018 = Int.floor (books_2017 * (1 + growth_rate)) :=
sorry

end book_borrowing_growth_l55_5503


namespace perfect_square_condition_l55_5599

theorem perfect_square_condition (p : Nat) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  let n : Nat := ((p - 1) / 2) ^ 2
  ∃ (k : Nat), n * p + n^2 = k^2 := by
  sorry

end perfect_square_condition_l55_5599


namespace speeding_ticket_percentage_l55_5534

/-- Proves that 20% of motorists who exceed the speed limit do not receive tickets -/
theorem speeding_ticket_percentage (M : ℝ) (h1 : M > 0) : 
  let exceed_limit := 0.125 * M
  let receive_ticket := 0.1 * M
  (exceed_limit - receive_ticket) / exceed_limit = 0.2 := by
sorry

end speeding_ticket_percentage_l55_5534


namespace curve_and_point_properties_l55_5587

-- Define the curve C
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = (p.1 + 1)^2}

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the property of the curve
def curve_property (p : ℝ × ℝ) : Prop :=
  p ∈ C → (p.1 - 1)^2 + p.2^2 = (p.1 + 1)^2

-- Define the equation of the curve
def curve_equation (p : ℝ × ℝ) : Prop :=
  p ∈ C → p.2^2 = 4 * p.1

-- Define the properties for point M
def M_properties (m : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  m ∈ C ∧ a ∈ C ∧ b ∈ C ∧
  ∃ (k : ℝ), k ≠ 0 ∧
    (a.2 - m.2) / (a.1 - m.1) = k ∧
    (b.2 - m.2) / (b.1 - m.1) = -k

-- Define the properties for points D and E
def DE_properties (d e : ℝ × ℝ) (a b : ℝ × ℝ) : Prop :=
  d ∈ C ∧ e ∈ C ∧
  (d.2 - e.2) / (d.1 - e.1) = -(b.1 - a.1) / (b.2 - a.2) ∧
  (d.1 - F.1) * (b.1 - a.1) + (d.2 - F.2) * (b.2 - a.2) = 0 ∧
  (e.1 - d.1)^2 + (e.2 - d.2)^2 = 64

-- State the theorem
theorem curve_and_point_properties :
  (∀ p, curve_property p) →
  (∀ p, curve_equation p) →
  ∀ m a b d e,
    M_properties m a b →
    DE_properties d e a b →
    m = (1, 2) ∨ m = (1, -2) := by sorry

end curve_and_point_properties_l55_5587


namespace handle_break_even_point_l55_5574

/-- Represents the break-even point calculation for a company producing handles --/
theorem handle_break_even_point
  (fixed_cost : ℝ)
  (variable_cost : ℝ)
  (selling_price : ℝ)
  (break_even_quantity : ℝ)
  (h1 : fixed_cost = 7640)
  (h2 : variable_cost = 0.60)
  (h3 : selling_price = 4.60)
  (h4 : break_even_quantity = 1910) :
  fixed_cost + variable_cost * break_even_quantity = selling_price * break_even_quantity :=
by
  sorry

#check handle_break_even_point

end handle_break_even_point_l55_5574


namespace min_value_product_l55_5556

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z * (x + y + z) = 1) : 
  (x + y) * (y + z) ≥ 2 := by
  sorry

end min_value_product_l55_5556


namespace power_equality_l55_5522

theorem power_equality (m n : ℤ) (P Q : ℝ) (h1 : P = 2^m) (h2 : Q = 3^n) :
  P^(2*n) * Q^m = 12^(m*n) := by
  sorry

end power_equality_l55_5522


namespace complex_equation_solution_l55_5521

theorem complex_equation_solution (z : ℂ) (i : ℂ) (h : i * i = -1) :
  i / z = 1 + i → z = (1 + i) / 2 := by
sorry

end complex_equation_solution_l55_5521
