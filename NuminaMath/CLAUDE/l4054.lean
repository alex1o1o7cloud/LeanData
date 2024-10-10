import Mathlib

namespace theresas_work_hours_l4054_405408

theorem theresas_work_hours : 
  let weekly_hours : List ℕ := [10, 13, 9, 14, 8, 0]
  let total_weeks : ℕ := 7
  let required_average : ℕ := 12
  let final_week_hours : ℕ := 30
  (List.sum weekly_hours + final_week_hours) / total_weeks = required_average :=
by
  sorry

end theresas_work_hours_l4054_405408


namespace half_angle_quadrant_l4054_405473

-- Define the fourth quadrant
def fourth_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi - Real.pi / 2 < α ∧ α < 2 * k * Real.pi

-- Define the second quadrant
def second_quadrant (α : Real) : Prop :=
  ∃ n : ℤ, (2 * n + 1) * Real.pi - Real.pi / 2 < α ∧ α < (2 * n + 1) * Real.pi

-- Define the fourth quadrant
def fourth_quadrant' (α : Real) : Prop :=
  ∃ n : ℤ, 2 * n * Real.pi - Real.pi / 2 < α ∧ α < 2 * n * Real.pi

-- Theorem statement
theorem half_angle_quadrant (α : Real) :
  fourth_quadrant α → (second_quadrant (α/2) ∨ fourth_quadrant' (α/2)) :=
by sorry

end half_angle_quadrant_l4054_405473


namespace can_capacity_l4054_405488

/-- The capacity of a can given specific milk-water ratios --/
theorem can_capacity (initial_milk : ℝ) (initial_water : ℝ) (added_milk : ℝ) : 
  initial_water = 5 * initial_milk →
  added_milk = 2 →
  (initial_milk + added_milk) / initial_water = 2.00001 / 5.00001 →
  initial_milk + initial_water + added_milk = 14 := by
  sorry

end can_capacity_l4054_405488


namespace expression_value_at_two_l4054_405452

theorem expression_value_at_two :
  let x : ℚ := 2
  (x^2 - x - 6) / (x - 3) = 4 := by sorry

end expression_value_at_two_l4054_405452


namespace sentence_B_is_error_free_l4054_405412

/-- Represents a sentence in the problem --/
inductive Sentence
| A : Sentence
| B : Sentence
| C : Sentence
| D : Sentence

/-- Checks if a sentence is free from linguistic errors --/
def is_error_free (s : Sentence) : Prop :=
  match s with
  | Sentence.A => False
  | Sentence.B => True
  | Sentence.C => False
  | Sentence.D => False

/-- The main theorem stating that Sentence B is free from linguistic errors --/
theorem sentence_B_is_error_free : is_error_free Sentence.B := by
  sorry

end sentence_B_is_error_free_l4054_405412


namespace intersection_length_and_product_l4054_405417

noncomputable section

-- Define the line L
def line (α : Real) (t : Real) : Real × Real :=
  (2 + t * Real.cos α, Real.sqrt 3 + t * Real.sin α)

-- Define the curve C
def curve (θ : Real) : Real × Real :=
  (2 * Real.cos θ, Real.sin θ)

-- Define point P
def P : Real × Real := (2, Real.sqrt 3)

-- Define the origin O
def O : Real × Real := (0, 0)

-- Define the distance between two points
def distance (p1 p2 : Real × Real) : Real :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_length_and_product (α : Real) :
  (α = Real.pi/3 → ∃ A B : Real × Real,
    A ≠ B ∧
    (∃ t : Real, line α t = A) ∧
    (∃ θ : Real, curve θ = A) ∧
    (∃ t : Real, line α t = B) ∧
    (∃ θ : Real, curve θ = B) ∧
    distance A B = 8 * Real.sqrt 10 / 13) ∧
  (Real.tan α = Real.sqrt 5 / 4 →
    ∃ A B : Real × Real,
    A ≠ B ∧
    (∃ t : Real, line α t = A) ∧
    (∃ θ : Real, curve θ = A) ∧
    (∃ t : Real, line α t = B) ∧
    (∃ θ : Real, curve θ = B) ∧
    distance P A * distance P B = distance O P ^ 2) :=
sorry

end

end intersection_length_and_product_l4054_405417


namespace total_tickets_bought_l4054_405401

theorem total_tickets_bought (adult_price children_price total_spent adult_count : ℚ)
  (h1 : adult_price = 5.5)
  (h2 : children_price = 3.5)
  (h3 : total_spent = 83.5)
  (h4 : adult_count = 5) :
  ∃ (children_count : ℚ), adult_count + children_count = 21 ∧
    adult_price * adult_count + children_price * children_count = total_spent :=
by sorry

end total_tickets_bought_l4054_405401


namespace special_sequence_theorem_l4054_405492

/-- A sequence satisfying certain properties -/
def SpecialSequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  c > 1 ∧
  a 1 = 1 ∧
  a 2 = 2 ∧
  (∀ m n, a (m * n) = a m * a n) ∧
  (∀ m n, a (m + n) ≤ c * (a m + a n))

/-- The main theorem: if a sequence satisfies the SpecialSequence properties,
    then a_n = n for all natural numbers n -/
theorem special_sequence_theorem (a : ℕ → ℝ) (c : ℝ) 
    (h : SpecialSequence a c) : ∀ n : ℕ, a n = n := by
  sorry

end special_sequence_theorem_l4054_405492


namespace sqrt_three_difference_of_squares_l4054_405493

theorem sqrt_three_difference_of_squares : (Real.sqrt 3 - 1) * (Real.sqrt 3 + 1) = 2 := by
  sorry

end sqrt_three_difference_of_squares_l4054_405493


namespace stripe_area_on_cylinder_l4054_405487

/-- The area of a stripe wrapped around a cylindrical object -/
theorem stripe_area_on_cylinder (diameter : ℝ) (stripe_width : ℝ) (revolutions : ℕ) :
  diameter = 30 →
  stripe_width = 4 →
  revolutions = 3 →
  stripe_width * revolutions * (π * diameter) = 360 * π := by
  sorry

end stripe_area_on_cylinder_l4054_405487


namespace rhombus_diagonals_l4054_405426

/-- Given a rhombus with perimeter 100 cm and sum of diagonals 62 cm, 
    prove that its diagonals are 48 cm and 14 cm. -/
theorem rhombus_diagonals (s : ℝ) (d₁ d₂ : ℝ) 
  (h_perimeter : 4 * s = 100)
  (h_diag_sum : d₁ + d₂ = 62)
  (h_pythag : s^2 = (d₁/2)^2 + (d₂/2)^2) :
  (d₁ = 48 ∧ d₂ = 14) ∨ (d₁ = 14 ∧ d₂ = 48) := by
  sorry

end rhombus_diagonals_l4054_405426


namespace hit_frequency_l4054_405471

theorem hit_frequency (total_shots : ℕ) (hits : ℕ) (h1 : total_shots = 20) (h2 : hits = 15) :
  (hits : ℚ) / total_shots = 3 / 4 := by
  sorry

end hit_frequency_l4054_405471


namespace trig_identity_l4054_405415

theorem trig_identity (θ φ : ℝ) 
  (h : (Real.sin θ)^6 / (Real.sin φ)^3 + (Real.cos θ)^6 / (Real.cos φ)^3 = 1) :
  (Real.sin φ)^6 / (Real.sin θ)^3 + (Real.cos φ)^6 / (Real.cos θ)^3 = 1 :=
by sorry

end trig_identity_l4054_405415


namespace ice_machine_cubes_l4054_405485

/-- The number of ice chests -/
def num_chests : ℕ := 7

/-- The number of ice cubes per chest -/
def cubes_per_chest : ℕ := 42

/-- The total number of ice cubes in the ice machine -/
def total_cubes : ℕ := num_chests * cubes_per_chest

/-- Theorem stating that the total number of ice cubes is 294 -/
theorem ice_machine_cubes : total_cubes = 294 := by
  sorry

end ice_machine_cubes_l4054_405485


namespace train_length_l4054_405404

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 3 → ∃ (length : ℝ), abs (length - 50.01) < 0.01 := by
  sorry

#check train_length

end train_length_l4054_405404


namespace display_window_configurations_l4054_405480

/-- The number of permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- The number of configurations for a single window with n books -/
def window_configurations (n : ℕ) : ℕ := factorial n

/-- The total number of configurations for two windows -/
def total_configurations (left_window : ℕ) (right_window : ℕ) : ℕ :=
  window_configurations left_window * window_configurations right_window

theorem display_window_configurations :
  total_configurations 3 3 = 36 :=
by sorry

end display_window_configurations_l4054_405480


namespace steps_on_sunday_l4054_405419

def target_average : ℕ := 9000
def days_in_week : ℕ := 7
def known_days : ℕ := 4
def friday_saturday_average : ℕ := 9050

def steps_known_days : List ℕ := [9100, 8300, 9200, 8900]

theorem steps_on_sunday (
  target_total : target_average * days_in_week = 63000)
  (known_total : steps_known_days.sum = 35500)
  (friday_saturday_total : friday_saturday_average * 2 = 18100)
  : 63000 - 35500 - 18100 = 9400 := by
  sorry

end steps_on_sunday_l4054_405419


namespace parallel_vectors_k_value_l4054_405405

/-- Two vectors are parallel if and only if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value :
  ∀ k : ℝ,
  let a : ℝ × ℝ := (2 * k + 2, 4)
  let b : ℝ × ℝ := (k + 1, 8)
  are_parallel a b → k = -1 := by
sorry

end parallel_vectors_k_value_l4054_405405


namespace second_catch_up_race_result_l4054_405442

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℝ

/-- Represents the state of the race -/
structure RaceState where
  runner1 : Runner
  runner2 : Runner
  laps_completed : ℝ

/-- The race setup with initial conditions -/
def initial_race : RaceState :=
  { runner1 := { speed := 3 },
    runner2 := { speed := 1 },
    laps_completed := 0.5 }

/-- The race state after the second runner doubles their speed -/
def race_after_speed_up (r : RaceState) : RaceState :=
  { runner1 := r.runner1,
    runner2 := { speed := 2 * r.runner2.speed },
    laps_completed := r.laps_completed }

/-- Theorem stating that the first runner will catch up again at 2.5 laps -/
theorem second_catch_up (r : RaceState) :
  let r' := race_after_speed_up r
  r'.runner1.speed > r'.runner2.speed →
  r.runner1.speed = 3 * r.runner2.speed →
  r.laps_completed = 0.5 →
  ∃ t : ℝ, t > 0 ∧ r'.runner1.speed * t = (2.5 - r.laps_completed + 1) * r'.runner2.speed * t :=
by
  sorry

/-- Main theorem combining all conditions and results -/
theorem race_result :
  let r := initial_race
  let r' := race_after_speed_up r
  r'.runner1.speed > r'.runner2.speed ∧
  r.runner1.speed = 3 * r.runner2.speed ∧
  r.laps_completed = 0.5 ∧
  (∃ t : ℝ, t > 0 ∧ r'.runner1.speed * t = (2.5 - r.laps_completed + 1) * r'.runner2.speed * t) :=
by
  sorry

end second_catch_up_race_result_l4054_405442


namespace sixteenth_occurrence_shift_l4054_405460

/-- Represents the number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- Calculates the sum of the first n even numbers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- Calculates the shift for the nth occurrence of a letter -/
def shift (n : ℕ) : ℕ := sum_even n % alphabet_size

/-- Theorem: The 16th occurrence of a letter is shifted by 16 places -/
theorem sixteenth_occurrence_shift :
  shift 16 = 16 := by sorry

end sixteenth_occurrence_shift_l4054_405460


namespace rotate90_neg_6_minus_3i_l4054_405475

def rotate90 (z : ℂ) : ℂ := z * Complex.I

theorem rotate90_neg_6_minus_3i :
  rotate90 (-6 - 3 * Complex.I) = (3 : ℂ) - 6 * Complex.I :=
by sorry

end rotate90_neg_6_minus_3i_l4054_405475


namespace inequality_solution_set_l4054_405453

theorem inequality_solution_set (x : ℝ) : 
  (x^2 - 4) * (x - 6)^2 ≤ 0 ↔ -2 ≤ x ∧ x ≤ 2 ∨ x = 6 :=
by sorry

end inequality_solution_set_l4054_405453


namespace ceiling_floor_sum_l4054_405437

theorem ceiling_floor_sum : ⌈(7 : ℚ) / 3⌉ + ⌊-(7 : ℚ) / 3⌋ = 0 := by sorry

end ceiling_floor_sum_l4054_405437


namespace intersecting_squares_area_difference_l4054_405434

/-- Given four intersecting squares with side lengths 12, 9, 7, and 3 (from left to right),
    the sum of the areas of the black regions minus the sum of the areas of the gray regions equals 103. -/
theorem intersecting_squares_area_difference : 
  let a := 12 -- side length of the largest square
  let b := 9  -- side length of the second largest square
  let c := 7  -- side length of the third largest square
  let d := 3  -- side length of the smallest square
  (a^2 + c^2) - (b^2 + d^2) = 103 := by sorry

end intersecting_squares_area_difference_l4054_405434


namespace complex_on_negative_y_axis_l4054_405451

theorem complex_on_negative_y_axis (a : ℝ) : 
  (∃ y : ℝ, y < 0 ∧ (a + Complex.I)^2 = Complex.I * y) → a = -1 := by
  sorry

end complex_on_negative_y_axis_l4054_405451


namespace smallest_N_for_P_less_than_half_l4054_405495

/-- The probability that at least 2/3 of the green balls are on the same side of either of the red balls -/
def P (N : ℕ) : ℚ :=
  sorry

/-- N is a multiple of 6 -/
def is_multiple_of_six (N : ℕ) : Prop :=
  ∃ k : ℕ, N = 6 * k

theorem smallest_N_for_P_less_than_half :
  (is_multiple_of_six 18) ∧
  (P 18 < 1/2) ∧
  (∀ N : ℕ, is_multiple_of_six N → N < 18 → P N ≥ 1/2) :=
sorry

end smallest_N_for_P_less_than_half_l4054_405495


namespace samantha_calculation_l4054_405413

theorem samantha_calculation : 
  let incorrect_input := 125 * 320
  let correct_product := 0.125 * 3.2
  let final_result := correct_product + 2.5
  incorrect_input = 40000 ∧ final_result = 6.5 := by
  sorry

end samantha_calculation_l4054_405413


namespace probability_at_least_one_woman_l4054_405427

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ) :
  total_people = men + women →
  men = 10 →
  women = 5 →
  selected = 4 →
  1 - (Nat.choose men selected / Nat.choose total_people selected) = 11 / 13 := by
  sorry

end probability_at_least_one_woman_l4054_405427


namespace square_area_error_l4054_405472

theorem square_area_error (S : ℝ) (S' : ℝ) (A : ℝ) (A' : ℝ) : 
  S > 0 →
  S' = S * 1.04 →
  A = S^2 →
  A' = S'^2 →
  (A' - A) / A * 100 = 8.16 := by
sorry

end square_area_error_l4054_405472


namespace monthly_average_production_l4054_405431

/-- Calculates the daily average production for a month given the production data for two periods. -/
theorem monthly_average_production 
  (days_first_period : ℕ) 
  (days_second_period : ℕ) 
  (avg_first_period : ℚ) 
  (avg_second_period : ℚ) : 
  days_first_period = 25 →
  days_second_period = 5 →
  avg_first_period = 70 →
  avg_second_period = 58 →
  (days_first_period * avg_first_period + days_second_period * avg_second_period) / (days_first_period + days_second_period) = 68 := by
  sorry

#check monthly_average_production

end monthly_average_production_l4054_405431


namespace product_negative_implies_one_less_than_one_l4054_405466

theorem product_negative_implies_one_less_than_one (a b c : ℝ) :
  (a - 1) * (b - 1) * (c - 1) < 0 →
  (a < 1) ∨ (b < 1) ∨ (c < 1) :=
by
  sorry

end product_negative_implies_one_less_than_one_l4054_405466


namespace negation_of_p_l4054_405409

theorem negation_of_p : ∀ x : ℝ, -2 < x ∧ x < 2 → |x - 1| + |x + 2| < 6 := by sorry

end negation_of_p_l4054_405409


namespace sphere_surface_area_ratio_l4054_405454

theorem sphere_surface_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (4 / 3 * Real.pi * r₁^3) / (4 / 3 * Real.pi * r₂^3) = 8 / 27 →
  (4 * Real.pi * r₁^2) / (4 * Real.pi * r₂^2) = 4 / 9 :=
by sorry

end sphere_surface_area_ratio_l4054_405454


namespace solve_for_n_l4054_405438

/-- Given an equation x + 1315 + n - 1569 = 11901 where x = 88320,
    prove that the value of n is -75165. -/
theorem solve_for_n (x n : ℤ) (h1 : x + 1315 + n - 1569 = 11901) (h2 : x = 88320) :
  n = -75165 := by
  sorry

end solve_for_n_l4054_405438


namespace roots_shifted_l4054_405444

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

-- Define the roots of the original polynomial
def roots_exist (a b c : ℝ) : Prop := 
  original_poly a = 0 ∧ original_poly b = 0 ∧ original_poly c = 0

-- Define the new polynomial
def new_poly (x : ℝ) : ℝ := x^3 + 7*x^2 + 14*x + 10

-- Theorem statement
theorem roots_shifted (a b c : ℝ) : 
  roots_exist a b c → 
  (new_poly (a - 3) = 0 ∧ new_poly (b - 3) = 0 ∧ new_poly (c - 3) = 0) :=
by sorry

end roots_shifted_l4054_405444


namespace biology_enrollment_percentage_l4054_405402

theorem biology_enrollment_percentage (total_students : ℕ) (not_enrolled : ℕ) : 
  total_students = 880 → not_enrolled = 462 → 
  (((total_students - not_enrolled : ℚ) / total_students) * 100 : ℚ) = 47.5 := by
  sorry

end biology_enrollment_percentage_l4054_405402


namespace insert_books_combinations_l4054_405421

theorem insert_books_combinations (n m : ℕ) : 
  n = 5 → m = 3 → (n + 1) * (n + 2) * (n + 3) = 336 := by
  sorry

end insert_books_combinations_l4054_405421


namespace unique_root_quadratic_l4054_405436

theorem unique_root_quadratic (p : ℝ) : 
  (∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ 
   (∀ x : ℝ, x^2 - 5*p*x + 2*p^3 = 0 ↔ (x = a ∨ x = b)) ∧
   (∃! x : ℝ, x^2 - a*x + b = 0)) →
  p = 3 := by
sorry


end unique_root_quadratic_l4054_405436


namespace sam_and_david_licks_l4054_405446

/-- The number of licks it takes for Dan to reach the center of a lollipop -/
def dan_licks : ℕ := 58

/-- The number of licks it takes for Michael to reach the center of a lollipop -/
def michael_licks : ℕ := 63

/-- The number of licks it takes for Lance to reach the center of a lollipop -/
def lance_licks : ℕ := 39

/-- The total number of people -/
def total_people : ℕ := 5

/-- The average number of licks for all people -/
def average_licks : ℕ := 60

/-- The theorem stating that Sam and David together take 140 licks to reach the center of a lollipop -/
theorem sam_and_david_licks : 
  total_people * average_licks - (dan_licks + michael_licks + lance_licks) = 140 := by
  sorry

end sam_and_david_licks_l4054_405446


namespace cosine_of_angle_between_tangents_l4054_405464

/-- The circle equation: x^2 - 2x + y^2 - 2y + 1 = 0 -/
def circle_equation (x y : ℝ) : Prop := x^2 - 2*x + y^2 - 2*y + 1 = 0

/-- The external point P -/
def P : ℝ × ℝ := (3, 2)

/-- The cosine of the angle between two tangent lines -/
noncomputable def cos_angle_between_tangents : ℝ := 3/5

theorem cosine_of_angle_between_tangents :
  let (px, py) := P
  ∀ x y : ℝ, circle_equation x y →
  cos_angle_between_tangents = 3/5 := by sorry

end cosine_of_angle_between_tangents_l4054_405464


namespace sandy_jessica_marble_ratio_l4054_405469

/-- Proves that Sandy has 4 times more red marbles than Jessica -/
theorem sandy_jessica_marble_ratio :
  let jessica_marbles : ℕ := 3 * 12 -- 3 dozen
  let sandy_marbles : ℕ := 144
  (sandy_marbles : ℚ) / jessica_marbles = 4 := by
  sorry

end sandy_jessica_marble_ratio_l4054_405469


namespace parallel_equidistant_lines_theorem_l4054_405474

/-- Represents a line segment with a length -/
structure LineSegment where
  length : ℝ

/-- Represents three parallel, equidistant line segments -/
structure ParallelEquidistantLines where
  line1 : LineSegment
  line2 : LineSegment
  line3 : LineSegment

/-- Given three parallel, equidistant lines where the first line is 120 cm and the second is 80 cm,
    the length of the third line is 160/3 cm -/
theorem parallel_equidistant_lines_theorem (lines : ParallelEquidistantLines) 
    (h1 : lines.line1.length = 120)
    (h2 : lines.line2.length = 80) :
    lines.line3.length = 160 / 3 := by
  sorry

end parallel_equidistant_lines_theorem_l4054_405474


namespace max_friendly_groups_19_20_l4054_405458

/-- A friendly group in a tournament --/
structure FriendlyGroup (α : Type*) :=
  (a b c : α)
  (a_beats_b : a ≠ b)
  (b_beats_c : b ≠ c)
  (c_beats_a : c ≠ a)

/-- Round-robin tournament results --/
def RoundRobinTournament (α : Type*) := α → α → Prop

/-- Maximum number of friendly groups in a tournament --/
def MaxFriendlyGroups (n : ℕ) : ℕ :=
  if n % 2 = 1 then
    n * (n - 1) * (n + 1) / 24
  else
    n * (n - 2) * (n + 2) / 24

/-- Theorem about maximum friendly groups in tournaments with 19 and 20 teams --/
theorem max_friendly_groups_19_20 :
  (MaxFriendlyGroups 19 = 285) ∧ (MaxFriendlyGroups 20 = 330) :=
by sorry

end max_friendly_groups_19_20_l4054_405458


namespace both_knaves_lied_yesterday_on_friday_l4054_405445

-- Define the days of the week
inductive Day : Type
  | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define the knaves
inductive Knave : Type
  | Hearts | Diamonds

-- Define the truth-telling function for each knave
def tells_truth (k : Knave) (d : Day) : Prop :=
  match k with
  | Knave.Hearts => d = Day.Monday ∨ d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Thursday
  | Knave.Diamonds => d = Day.Friday ∨ d = Day.Saturday ∨ d = Day.Sunday ∨ d = Day.Monday

-- Define the function to check if a knave lied yesterday
def lied_yesterday (k : Knave) (d : Day) : Prop :=
  ¬(tells_truth k (match d with
    | Day.Monday => Day.Sunday
    | Day.Tuesday => Day.Monday
    | Day.Wednesday => Day.Tuesday
    | Day.Thursday => Day.Wednesday
    | Day.Friday => Day.Thursday
    | Day.Saturday => Day.Friday
    | Day.Sunday => Day.Saturday))

-- Theorem: The only day when both knaves can truthfully say "Yesterday I told lies" is Friday
theorem both_knaves_lied_yesterday_on_friday :
  ∀ d : Day, (tells_truth Knave.Hearts d ∧ lied_yesterday Knave.Hearts d ∧
              tells_truth Knave.Diamonds d ∧ lied_yesterday Knave.Diamonds d) 
             ↔ d = Day.Friday :=
sorry

end both_knaves_lied_yesterday_on_friday_l4054_405445


namespace triangle_abc_acute_angled_l4054_405461

theorem triangle_abc_acute_angled (A B C : ℝ) 
  (h1 : A + B + C = 180) 
  (h2 : A = B) 
  (h3 : A = 2 * C) : 
  A < 90 ∧ B < 90 ∧ C < 90 := by
sorry


end triangle_abc_acute_angled_l4054_405461


namespace quadratic_equation_identification_l4054_405435

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation from option A -/
def eq_A (x : ℝ) : ℝ := 3 * x + 1

/-- The equation from option B -/
def eq_B (x : ℝ) : ℝ := x^2 - (2 * x - 3 * x^2)

/-- The equation from option C -/
def eq_C (x y : ℝ) : ℝ := x^2 - y + 5

/-- The equation from option D -/
def eq_D (x y : ℝ) : ℝ := x - x * y - 1 - x^2

theorem quadratic_equation_identification :
  ¬ is_quadratic_equation eq_A ∧
  is_quadratic_equation eq_B ∧
  ¬ (∃ f : ℝ → ℝ, ∀ x y, eq_C x y = f x) ∧
  ¬ (∃ f : ℝ → ℝ, ∀ x y, eq_D x y = f x) :=
sorry

end quadratic_equation_identification_l4054_405435


namespace car_trip_duration_l4054_405432

/-- Proves that a car trip with given conditions has a total duration of 6 hours -/
theorem car_trip_duration (initial_speed initial_time additional_speed average_speed : ℝ) :
  initial_speed = 75 →
  initial_time = 4 →
  additional_speed = 60 →
  average_speed = 70 →
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) /
      total_time = average_speed ∧
    total_time = 6 :=
by sorry

end car_trip_duration_l4054_405432


namespace cost_of_item_d_l4054_405410

/-- Represents the prices and taxes for items in a shopping scenario -/
structure ShoppingScenario where
  total_spent : ℝ
  total_abc : ℝ
  total_tax : ℝ
  tax_rate_a : ℝ
  tax_rate_b : ℝ
  tax_rate_c : ℝ
  discount_a : ℝ
  discount_b : ℝ

/-- Theorem stating that the cost of item D is 25 given the shopping scenario -/
theorem cost_of_item_d (s : ShoppingScenario)
  (h1 : s.total_spent = 250)
  (h2 : s.total_abc = 225)
  (h3 : s.total_tax = 30)
  (h4 : s.tax_rate_a = 0.05)
  (h5 : s.tax_rate_b = 0.12)
  (h6 : s.tax_rate_c = 0.18)
  (h7 : s.discount_a = 0.1)
  (h8 : s.discount_b = 0.05) :
  s.total_spent - s.total_abc = 25 := by
  sorry

#check cost_of_item_d

end cost_of_item_d_l4054_405410


namespace train_platform_crossing_time_l4054_405497

/-- Given a train of length 1200 m that crosses a tree in 80 seconds,
    prove that the time it takes to pass a platform of length 1000 m is 146.67 seconds. -/
theorem train_platform_crossing_time 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1200)
  (h2 : tree_crossing_time = 80)
  (h3 : platform_length = 1000) :
  (train_length + platform_length) / (train_length / tree_crossing_time) = 146.67 := by
sorry

end train_platform_crossing_time_l4054_405497


namespace problem_statement_l4054_405425

/-- Given two real numbers a and b with average 110, and b and c with average 170,
    if a - c = 120, then c = -120 -/
theorem problem_statement (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110)
  (h2 : (b + c) / 2 = 170)
  (h3 : a - c = 120) :
  c = -120 := by
  sorry

end problem_statement_l4054_405425


namespace range_of_a_l4054_405448

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ∈ Set.Ici (-8) :=
sorry

end range_of_a_l4054_405448


namespace problem_N4_l4054_405430

theorem problem_N4 (a b : ℕ+) 
  (h : ∀ n : ℕ+, n > 2020^2020 → 
    ∃ m : ℕ+, Nat.Coprime m.val n.val ∧ (a^n.val + b^n.val ∣ a^m.val + b^m.val)) :
  a = b := by
  sorry

end problem_N4_l4054_405430


namespace pets_remaining_l4054_405489

theorem pets_remaining (initial_puppies initial_kittens sold_puppies sold_kittens : ℕ) 
  (h1 : initial_puppies = 7)
  (h2 : initial_kittens = 6)
  (h3 : sold_puppies = 2)
  (h4 : sold_kittens = 3) :
  initial_puppies + initial_kittens - (sold_puppies + sold_kittens) = 8 := by
sorry

end pets_remaining_l4054_405489


namespace vector_on_line_l4054_405482

/-- Given two complex numbers z₁ and z₂, representing points A and B in the complex plane,
    we define z as the vector from A to B, and prove that when z lies on the line y = 1/2 x,
    we can determine the value of parameter a. -/
theorem vector_on_line (a : ℝ) :
  let z₁ : ℂ := 2 * a + 6 * Complex.I
  let z₂ : ℂ := -1 + Complex.I
  let z : ℂ := z₂ - z₁
  z.im = (1/2 : ℝ) * z.re →
  z = -1 - 2 * a - 5 * Complex.I ∧ a = (9/2 : ℝ) := by
  sorry

end vector_on_line_l4054_405482


namespace percentage_increase_proof_l4054_405449

def original_earnings : ℝ := 30
def new_earnings : ℝ := 40

theorem percentage_increase_proof :
  (new_earnings - original_earnings) / original_earnings * 100 =
  (40 - 30) / 30 * 100 :=
by sorry

end percentage_increase_proof_l4054_405449


namespace fuwa_selection_theorem_l4054_405462

/-- The number of types of "Chinese Fuwa" mascots -/
def num_types : ℕ := 5

/-- The total number of Fuwa mascots -/
def total_fuwa : ℕ := 10

/-- The number of Fuwa to be selected -/
def select_num : ℕ := 5

/-- The number of ways to select Fuwa mascots -/
def ways_to_select : ℕ := 160

/-- Theorem stating the number of ways to select Fuwa mascots -/
theorem fuwa_selection_theorem :
  (num_types = 5) →
  (total_fuwa = 10) →
  (select_num = 5) →
  (ways_to_select = 
    2 * (Nat.choose num_types 1) * (2^(num_types - 1))) :=
by sorry

end fuwa_selection_theorem_l4054_405462


namespace max_y_over_x_l4054_405443

theorem max_y_over_x (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≥ 0) (h3 : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (k : ℝ), ∀ (x' y' : ℝ), x' ≠ 0 → y' ≥ 0 → x'^2 + y'^2 - 4*x' + 1 = 0 → y'/x' ≤ k ∧ k = Real.sqrt 3 :=
sorry

end max_y_over_x_l4054_405443


namespace max_value_of_s_l4054_405484

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20) :
  s ≤ (5 * (1 + Real.sqrt 21)) / 2 :=
sorry

end max_value_of_s_l4054_405484


namespace triangle_sine_product_l4054_405456

theorem triangle_sine_product (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b + c = 3 →
  A = π / 3 →
  0 < B →
  B < π →
  0 < C →
  C < π →
  A + B + C = π →
  a = 2 * Real.sin (B / 2) * Real.sin (C / 2) →
  b = 2 * Real.sin (A / 2) * Real.sin (C / 2) →
  c = 2 * Real.sin (A / 2) * Real.sin (B / 2) →
  Real.sin B * Real.sin C = 1 / 2 := by
  sorry

end triangle_sine_product_l4054_405456


namespace sets_intersection_empty_l4054_405470

-- Define set A
def A : Set ℝ := {x | x^2 + 5*x + 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 2*x + 15)}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem sets_intersection_empty (a : ℝ) : (A ∪ B) ∩ C a = ∅ ↔ a ≥ 5 ∨ a ≤ -4 := by
  sorry

end sets_intersection_empty_l4054_405470


namespace integer_root_of_polynomial_l4054_405455

theorem integer_root_of_polynomial (b c : ℚ) : 
  (∃ (x : ℝ), x^3 + b*x + c = 0 ∧ x = 3 - Real.sqrt 5) → 
  (-6 : ℝ)^3 + b*(-6) + c = 0 := by
sorry

end integer_root_of_polynomial_l4054_405455


namespace seventh_observation_value_l4054_405441

theorem seventh_observation_value 
  (n : Nat) 
  (initial_avg : ℝ) 
  (new_avg : ℝ) 
  (h1 : n = 6) 
  (h2 : initial_avg = 11) 
  (h3 : new_avg = initial_avg - 1) : 
  (n : ℝ) * initial_avg - ((n + 1) : ℝ) * new_avg = -4 := by
  sorry

end seventh_observation_value_l4054_405441


namespace net_marble_change_l4054_405416

def marble_transactions (initial : Int) (lost : Int) (found : Int) (traded_out : Int) (traded_in : Int) (gave_away : Int) (received : Int) : Int :=
  initial - lost + found - traded_out + traded_in - gave_away + received

theorem net_marble_change : 
  marble_transactions 20 16 8 5 9 3 4 = -3 := by
  sorry

end net_marble_change_l4054_405416


namespace circle_area_ratio_l4054_405420

-- Define the circles C and D
variables (C D : ℝ → Prop)

-- Define the radii of circles C and D
variables (r_C r_D : ℝ)

-- Define the common arc length
variable (L : ℝ)

-- State the theorem
theorem circle_area_ratio 
  (h1 : L = (60 / 360) * (2 * Real.pi * r_C)) 
  (h2 : L = (45 / 360) * (2 * Real.pi * r_D)) 
  (h3 : 2 * Real.pi * r_D = 2 * (2 * Real.pi * r_C)) :
  (Real.pi * r_D^2) / (Real.pi * r_C^2) = 4 := by
  sorry

end circle_area_ratio_l4054_405420


namespace used_computer_cost_l4054_405498

/-- Proves the cost of each used computer given the conditions of the problem -/
theorem used_computer_cost
  (new_computer_cost : ℕ)
  (new_computer_lifespan : ℕ)
  (used_computer_lifespan : ℕ)
  (savings : ℕ)
  (h1 : new_computer_cost = 600)
  (h2 : new_computer_lifespan = 6)
  (h3 : used_computer_lifespan = 3)
  (h4 : 2 * used_computer_lifespan = new_computer_lifespan)
  (h5 : savings = 200)
  (h6 : ∃ (used_computer_cost : ℕ),
        new_computer_cost = 2 * used_computer_cost + savings) :
  ∃ (used_computer_cost : ℕ), used_computer_cost = 200 := by
  sorry

end used_computer_cost_l4054_405498


namespace product_remainder_l4054_405414

theorem product_remainder (a b c : ℕ) (h : a = 1125 ∧ b = 1127 ∧ c = 1129) : 
  (a * b * c) % 12 = 3 := by
sorry

end product_remainder_l4054_405414


namespace age_ratio_l4054_405499

theorem age_ratio (sum_ages : ℕ) (your_age : ℕ) : 
  sum_ages = 40 → your_age = 10 → (sum_ages - your_age) / your_age = 3 := by
  sorry

end age_ratio_l4054_405499


namespace bankers_discount_l4054_405465

/-- Banker's discount calculation --/
theorem bankers_discount (bankers_gain : ℚ) (interest_rate : ℚ) (time : ℕ) : 
  bankers_gain = 270 → interest_rate = 12 / 100 → time = 3 → 
  let present_value := (bankers_gain * 100) / (interest_rate * time)
  let face_value := present_value + bankers_gain
  let bankers_discount := (face_value * interest_rate * time)
  bankers_discount = 36720 / 100 := by
  sorry

end bankers_discount_l4054_405465


namespace lowest_possible_score_l4054_405486

def exam_max_score : ℕ := 120
def num_exams : ℕ := 5
def goal_average : ℕ := 100
def current_scores : List ℕ := [90, 108, 102]

theorem lowest_possible_score :
  let total_needed : ℕ := goal_average * num_exams
  let current_total : ℕ := current_scores.sum
  let remaining_total : ℕ := total_needed - current_total
  let max_score_one_exam : ℕ := min exam_max_score remaining_total
  ∃ (lowest : ℕ), 
    lowest = remaining_total - max_score_one_exam ∧
    lowest = 80 :=
sorry

end lowest_possible_score_l4054_405486


namespace exchange_rate_change_l4054_405422

theorem exchange_rate_change 
  (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : 0 < y ∧ y < 1) 
  (hz : 0 < z ∧ z < 1) 
  (h_eq : (1 + x) * (1 + y) * (1 + z) = (1 - x) * (1 - y) * (1 - z)) : 
  (1 - x) * (1 - y) * (1 - z) < 1 :=
by sorry

end exchange_rate_change_l4054_405422


namespace depression_comparison_l4054_405450

-- Define the prevalence of depression for women and men
def depression_prevalence_women : ℝ := 2
def depression_prevalence_men : ℝ := 1

-- Define the correct comparative phrase
def correct_phrase : String := "twice as...as"

-- Theorem to prove
theorem depression_comparison (w m : ℝ) (phrase : String) :
  w = 2 * m → phrase = correct_phrase → 
  (w = depression_prevalence_women ∧ m = depression_prevalence_men) :=
by sorry

end depression_comparison_l4054_405450


namespace expand_and_simplify_l4054_405429

theorem expand_and_simplify (x : ℝ) : (x + 6) * (x - 11) = x^2 - 5*x - 66 := by
  sorry

end expand_and_simplify_l4054_405429


namespace mini_cupcakes_count_l4054_405494

theorem mini_cupcakes_count (students : ℕ) (donut_holes : ℕ) (desserts_per_student : ℕ) :
  students = 13 →
  donut_holes = 12 →
  desserts_per_student = 2 →
  ∃ (mini_cupcakes : ℕ), 
    mini_cupcakes + donut_holes = students * desserts_per_student ∧
    mini_cupcakes = 14 :=
by
  sorry

end mini_cupcakes_count_l4054_405494


namespace power_product_equals_6300_l4054_405496

theorem power_product_equals_6300 : 2^2 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end power_product_equals_6300_l4054_405496


namespace complex_sum_real_part_l4054_405479

theorem complex_sum_real_part (z₁ z₂ z₃ : ℂ) (r : ℝ) 
  (h₁ : Complex.abs z₁ = 1) 
  (h₂ : Complex.abs z₂ = 1) 
  (h₃ : Complex.abs z₃ = 1) 
  (h₄ : Complex.abs (z₁ + z₂ + z₃) = r) : 
  (z₁ / z₂ + z₂ / z₃ + z₃ / z₁).re = (r^2 - 3) / 2 := by
  sorry

#check complex_sum_real_part

end complex_sum_real_part_l4054_405479


namespace sturgeon_books_problem_l4054_405433

theorem sturgeon_books_problem (total_volumes : ℕ) (paperback_cost hardcover_cost total_cost : ℕ) 
  (h : total_volumes = 10)
  (hp : paperback_cost = 15)
  (hh : hardcover_cost = 25)
  (ht : total_cost = 220) :
  ∃ (hardcover_count : ℕ), 
    hardcover_count * hardcover_cost + (total_volumes - hardcover_count) * paperback_cost = total_cost ∧
    hardcover_count = 7 := by
  sorry

end sturgeon_books_problem_l4054_405433


namespace video_archive_space_theorem_l4054_405423

/-- Represents the number of days in the video archive -/
def days : ℕ := 15

/-- Represents the total disk space used by the archive in megabytes -/
def total_space : ℕ := 30000

/-- Calculates the total number of minutes in a given number of days -/
def total_minutes (d : ℕ) : ℕ := d * 24 * 60

/-- Calculates the average disk space per minute of video -/
def avg_space_per_minute : ℚ :=
  total_space / total_minutes days

theorem video_archive_space_theorem :
  abs (avg_space_per_minute - 1.388) < 0.001 :=
sorry

end video_archive_space_theorem_l4054_405423


namespace expression_value_l4054_405403

-- Define opposite numbers
def opposite (m n : ℝ) : Prop := m + n = 0

-- Define reciprocal numbers
def reciprocal (p q : ℝ) : Prop := p * q = 1

-- Theorem statement
theorem expression_value 
  (m n p q : ℝ) 
  (h1 : opposite m n) 
  (h2 : m ≠ n) 
  (h3 : reciprocal p q) : 
  (m + n) / m + 2 * p * q - m / n = 3 := by
sorry

end expression_value_l4054_405403


namespace thomas_score_l4054_405428

def class_size : ℕ := 20
def initial_average : ℚ := 78
def final_average : ℚ := 79

theorem thomas_score :
  ∃ (score : ℚ),
    (class_size - 1) * initial_average + score = class_size * final_average ∧
    score = 98 := by
  sorry

end thomas_score_l4054_405428


namespace h_equals_neg_f_of_six_minus_x_l4054_405459

-- Define a function that reflects a graph across the y-axis
def reflectY (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (-x)

-- Define a function that reflects a graph across the x-axis
def reflectX (f : ℝ → ℝ) : ℝ → ℝ := λ x => -f x

-- Define a function that shifts a graph to the right by a given amount
def shiftRight (f : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ := λ x => f (x - shift)

-- Define the composition of these transformations
def h (f : ℝ → ℝ) : ℝ → ℝ := shiftRight (reflectX (reflectY f)) 6

-- State the theorem
theorem h_equals_neg_f_of_six_minus_x (f : ℝ → ℝ) : 
  ∀ x : ℝ, h f x = -f (6 - x) := by sorry

end h_equals_neg_f_of_six_minus_x_l4054_405459


namespace tom_roses_count_l4054_405440

/-- The number of roses in a dozen -/
def roses_per_dozen : ℕ := 12

/-- The number of dozens Tom sends per day -/
def dozens_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of roses Tom sent in a week -/
def total_roses : ℕ := days_in_week * dozens_per_day * roses_per_dozen

theorem tom_roses_count : total_roses = 168 := by
  sorry

end tom_roses_count_l4054_405440


namespace last_digit_product_divisible_by_six_l4054_405418

/-- The last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- The remaining digits of a natural number -/
def remainingDigits (n : ℕ) : ℕ := n / 10

/-- Theorem: For all n > 3, the product of the last digit of 2^n and the remaining digits is divisible by 6 -/
theorem last_digit_product_divisible_by_six (n : ℕ) (h : n > 3) :
  ∃ k : ℕ, (lastDigit (2^n) * remainingDigits (2^n)) = 6 * k := by
  sorry

end last_digit_product_divisible_by_six_l4054_405418


namespace salary_calculation_l4054_405478

theorem salary_calculation (food_fraction rent_fraction clothes_fraction remainder : ℚ) 
  (h1 : food_fraction = 1/5)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : remainder = 18000) :
  let total_spent_fraction := food_fraction + rent_fraction + clothes_fraction
  let remaining_fraction := 1 - total_spent_fraction
  let salary := remainder / remaining_fraction
  salary = 180000 := by sorry

end salary_calculation_l4054_405478


namespace quadratic_equivalent_forms_l4054_405467

theorem quadratic_equivalent_forms : ∀ x : ℝ, x^2 - 2*x - 1 = (x - 1)^2 - 2 := by
  sorry

end quadratic_equivalent_forms_l4054_405467


namespace hexagon_diagonal_intersection_probability_l4054_405406

/-- A regular hexagon -/
structure RegularHexagon where
  -- Add any necessary properties here

/-- A diagonal of a regular hexagon -/
structure Diagonal (h : RegularHexagon) where
  -- Add any necessary properties here

/-- Predicate to check if two diagonals intersect inside the hexagon -/
def intersect_inside (h : RegularHexagon) (d1 d2 : Diagonal h) : Prop :=
  sorry

/-- The set of all diagonals in a regular hexagon -/
def all_diagonals (h : RegularHexagon) : Set (Diagonal h) :=
  sorry

/-- The probability that two randomly chosen diagonals intersect inside the hexagon -/
def intersection_probability (h : RegularHexagon) : ℚ :=
  sorry

theorem hexagon_diagonal_intersection_probability (h : RegularHexagon) :
  intersection_probability h = 5 / 12 := by
  sorry

end hexagon_diagonal_intersection_probability_l4054_405406


namespace intersection_when_a_is_two_subset_condition_l4054_405424

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Theorem 1: When a = 2, A ∩ B = (4, 5)
theorem intersection_when_a_is_two :
  A 2 ∩ B 2 = Set.Ioo 4 5 := by sorry

-- Theorem 2: B ⊆ A if and only if a ∈ [1, 3] ∪ {-1}
theorem subset_condition (a : ℝ) :
  B a ⊆ A a ↔ a ∈ Set.Icc 1 3 ∪ {-1} := by sorry

end intersection_when_a_is_two_subset_condition_l4054_405424


namespace remainder_theorem_l4054_405463

theorem remainder_theorem (P E M S F N T : ℕ) 
  (h1 : P = E * M + S) 
  (h2 : M = N * F + T) : 
  P % (E * F + 1) = E * T + S - N :=
sorry

end remainder_theorem_l4054_405463


namespace hiker_distance_l4054_405400

/-- Calculates the final straight-line distance of a hiker from their starting point
    given their movements in cardinal directions. -/
theorem hiker_distance (north south west east : ℝ) :
  north = 20 ∧ south = 8 ∧ west = 15 ∧ east = 10 →
  Real.sqrt ((north - south)^2 + (west - east)^2) = 13 := by
  sorry

end hiker_distance_l4054_405400


namespace cubic_polynomial_relation_l4054_405477

/-- Given a cubic polynomial f(x) = x^3 + 3x^2 + x + 1, and another cubic polynomial h
    such that h(0) = 1 and the roots of h are the cubes of the roots of f,
    prove that h(-8) = -115. -/
theorem cubic_polynomial_relation (f h : ℝ → ℝ) : 
  (∀ x, f x = x^3 + 3*x^2 + x + 1) →
  (∃ a b c : ℝ, ∀ x, h x = (x - a^3) * (x - b^3) * (x - c^3)) →
  h 0 = 1 →
  (∀ x, f x = 0 ↔ h (x^3) = 0) →
  h (-8) = -115 := by
  sorry

end cubic_polynomial_relation_l4054_405477


namespace derivative_exponential_plus_sine_l4054_405491

theorem derivative_exponential_plus_sine (x : ℝ) :
  let y := fun x => Real.exp x + Real.sin x
  HasDerivAt y (Real.exp x + Real.cos x) x :=
by sorry

end derivative_exponential_plus_sine_l4054_405491


namespace simplify_expression_l4054_405411

theorem simplify_expression (q : ℝ) : 
  ((6 * q - 2) - 3 * q * 5) * 2 + (5 - 2 / 4) * (8 * q - 12) = 18 * q - 58 := by
  sorry

end simplify_expression_l4054_405411


namespace club_probability_theorem_l4054_405476

theorem club_probability_theorem (total_members : ℕ) (boys : ℕ) (girls : ℕ) :
  total_members = 15 →
  boys = 8 →
  girls = 7 →
  total_members = boys + girls →
  (Nat.choose total_members 2 - Nat.choose girls 2 : ℚ) / Nat.choose total_members 2 = 4 / 5 := by
  sorry

end club_probability_theorem_l4054_405476


namespace marathon_speed_fraction_l4054_405481

theorem marathon_speed_fraction (t₃ t₆ : ℝ) (h₁ : t₃ > 0) (h₂ : t₆ > 0) : 
  (3 * t₃ + 6 * t₆) / (t₃ + t₆) = 5 → t₃ / (t₃ + t₆) = 1/3 := by
  sorry

end marathon_speed_fraction_l4054_405481


namespace square_on_hypotenuse_l4054_405468

theorem square_on_hypotenuse (a b : ℝ) (ha : a = 9) (hb : b = 12) :
  let c := Real.sqrt (a^2 + b^2)
  let s := (a * b * c) / (a^2 + b^2)
  s = 45 / 8 := by sorry

end square_on_hypotenuse_l4054_405468


namespace other_cat_weight_l4054_405439

/-- Represents the weights of animals in a household -/
structure HouseholdWeights where
  cat1 : ℝ
  cat2 : ℝ
  dog : ℝ

/-- Theorem stating the weight of the second cat given the conditions -/
theorem other_cat_weight (h : HouseholdWeights) 
    (h1 : h.cat1 = 10)
    (h2 : h.dog = 34)
    (h3 : h.dog = 2 * (h.cat1 + h.cat2)) : 
  h.cat2 = 7 := by
  sorry

end other_cat_weight_l4054_405439


namespace carpet_area_needed_l4054_405407

-- Define the room dimensions in feet
def room_length : ℝ := 18
def room_width : ℝ := 12

-- Define the conversion factor from feet to yards
def feet_per_yard : ℝ := 3

-- Define the area already covered in square yards
def area_covered : ℝ := 4

-- Theorem to prove
theorem carpet_area_needed : 
  let length_yards := room_length / feet_per_yard
  let width_yards := room_width / feet_per_yard
  let total_area := length_yards * width_yards
  total_area - area_covered = 20 := by sorry

end carpet_area_needed_l4054_405407


namespace sport_formulation_corn_syrup_amount_l4054_405457

/-- Represents the ratio of ingredients in a drink formulation -/
structure DrinkRatio :=
  (flavoring : ℚ)
  (corn_syrup : ℚ)
  (water : ℚ)

/-- The standard formulation of the drink -/
def standard_ratio : DrinkRatio :=
  { flavoring := 1, corn_syrup := 12, water := 30 }

/-- The sport formulation of the drink -/
def sport_ratio : DrinkRatio :=
  { flavoring := standard_ratio.flavoring,
    corn_syrup := standard_ratio.corn_syrup / 3,
    water := standard_ratio.water * 2 }

/-- Amount of water in the large bottle of sport formulation -/
def water_amount : ℚ := 45

theorem sport_formulation_corn_syrup_amount :
  (water_amount * sport_ratio.corn_syrup) / sport_ratio.water = water_amount :=
sorry

end sport_formulation_corn_syrup_amount_l4054_405457


namespace rectangle_D_max_sum_l4054_405483

-- Define the rectangle structure
structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the rectangles
def rectangleA : Rectangle := ⟨9, 3, 5, 7⟩
def rectangleB : Rectangle := ⟨8, 2, 4, 6⟩
def rectangleC : Rectangle := ⟨7, 1, 3, 5⟩
def rectangleD : Rectangle := ⟨10, 0, 6, 8⟩
def rectangleE : Rectangle := ⟨6, 4, 2, 0⟩

-- Define the list of all rectangles
def rectangles : List Rectangle := [rectangleA, rectangleB, rectangleC, rectangleD, rectangleE]

-- Function to check if a value is unique in a list
def isUnique (n : ℕ) (l : List ℕ) : Bool :=
  (l.filter (· = n)).length = 1

-- Theorem: Rectangle D has the maximum sum of w + z where z is unique
theorem rectangle_D_max_sum : 
  ∀ r ∈ rectangles, 
    isUnique r.z (rectangles.map Rectangle.z) → 
      r.w + r.z ≤ rectangleD.w + rectangleD.z :=
sorry

end rectangle_D_max_sum_l4054_405483


namespace min_passengers_for_no_loss_l4054_405447

/-- Represents the monthly expenditure of the bus in yuan -/
def monthly_expenditure : ℕ := 6000

/-- Represents the fare per person in yuan -/
def fare_per_person : ℕ := 2

/-- Represents the relationship between the number of passengers (x) and the difference between income and expenditure (y) -/
def income_expenditure_difference (x : ℕ) : ℤ :=
  (fare_per_person * x : ℤ) - monthly_expenditure

/-- Represents the condition for the bus to operate without a loss -/
def no_loss (x : ℕ) : Prop :=
  income_expenditure_difference x ≥ 0

/-- States that the minimum number of passengers required for the bus to operate without a loss is 3000 -/
theorem min_passengers_for_no_loss :
  ∀ x : ℕ, no_loss x ↔ x ≥ 3000 :=
by sorry

end min_passengers_for_no_loss_l4054_405447


namespace certification_cost_coverage_percentage_l4054_405490

/-- Calculates the percentage of certification cost covered by insurance for a seeing-eye dog. -/
theorem certification_cost_coverage_percentage
  (adoption_fee : ℕ)
  (training_cost_per_week : ℕ)
  (training_weeks : ℕ)
  (certification_cost : ℕ)
  (total_out_of_pocket : ℕ)
  (h1 : adoption_fee = 150)
  (h2 : training_cost_per_week = 250)
  (h3 : training_weeks = 12)
  (h4 : certification_cost = 3000)
  (h5 : total_out_of_pocket = 3450) :
  (100 * (certification_cost - (total_out_of_pocket - adoption_fee - training_cost_per_week * training_weeks))) / certification_cost = 90 :=
by sorry

end certification_cost_coverage_percentage_l4054_405490
