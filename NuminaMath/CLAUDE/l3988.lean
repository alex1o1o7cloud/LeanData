import Mathlib

namespace NUMINAMATH_CALUDE_alternating_fraction_value_l3988_398892

theorem alternating_fraction_value :
  ∃ y : ℝ, y = 3 + 5 / (2 + 5 / y) ∧ y = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_alternating_fraction_value_l3988_398892


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3988_398844

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x) + (9 / y) ≥ 16 ∧
  ((1 / x) + (9 / y) = 16 ↔ x = 1/4 ∧ y = 3/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l3988_398844


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3988_398854

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 32 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 3 + a 10 + a 11 = 32

/-- Theorem: If a is an arithmetic sequence satisfying the sum condition,
    then the sum of the 6th and 7th terms is 16 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : a 6 + a 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3988_398854


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3988_398842

theorem fraction_decomposition (x : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  (2 / (x^2 - 1) = 1 / (x - 1) - 1 / (x + 1)) ∧
  (2 * x / (x^2 - 1) = 1 / (x - 1) + 1 / (x + 1)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3988_398842


namespace NUMINAMATH_CALUDE_problem_solution_l3988_398834

theorem problem_solution : 
  (2008^2 - 2007 * 2009 = 1) ∧ 
  ((-0.125)^2011 * 8^2010 = -0.125) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3988_398834


namespace NUMINAMATH_CALUDE_prob_A_or_B_l3988_398821

/- Given probabilities -/
def P_A : ℝ := 0.4
def P_B : ℝ := 0.65
def P_A_and_B : ℝ := 0.25

/- Theorem to prove -/
theorem prob_A_or_B : P_A + P_B - P_A_and_B = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_or_B_l3988_398821


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l3988_398819

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 2|

-- State the theorem
theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, f a x ≤ |x - 4|) → a ∈ Set.Icc (-3) 0 := by
  sorry


end NUMINAMATH_CALUDE_range_of_a_for_inequality_l3988_398819


namespace NUMINAMATH_CALUDE_matrix_power_zero_l3988_398881

open Matrix Complex

theorem matrix_power_zero (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℂ) 
  (h1 : A * B = B * A)
  (h2 : B.det ≠ 0)
  (h3 : ∀ z : ℂ, Complex.abs z = 1 → Complex.abs ((A + z • B).det) = 1) :
  A ^ n = 0 := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_zero_l3988_398881


namespace NUMINAMATH_CALUDE_negative_a_cubed_times_negative_a_fourth_l3988_398809

theorem negative_a_cubed_times_negative_a_fourth (a : ℝ) : -a^3 * (-a)^4 = -a^7 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_cubed_times_negative_a_fourth_l3988_398809


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3988_398852

theorem negation_of_existence (P : ℝ → Prop) : 
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x : ℝ, x^2 - x + 3 = 0) ↔ (∀ x : ℝ, x^2 - x + 3 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l3988_398852


namespace NUMINAMATH_CALUDE_two_activities_count_l3988_398827

/-- Represents a school club with three activities -/
structure Club where
  total_members : ℕ
  cannot_paint : ℕ
  cannot_sculpt : ℕ
  cannot_draw : ℕ

/-- Calculates the number of members involved in exactly two activities -/
def members_in_two_activities (c : Club) : ℕ :=
  let can_paint := c.total_members - c.cannot_paint
  let can_sculpt := c.total_members - c.cannot_sculpt
  let can_draw := c.total_members - c.cannot_draw
  can_paint + can_sculpt + can_draw - c.total_members

/-- Theorem stating the number of members involved in exactly two activities -/
theorem two_activities_count (c : Club) 
  (h1 : c.total_members = 150)
  (h2 : c.cannot_paint = 55)
  (h3 : c.cannot_sculpt = 90)
  (h4 : c.cannot_draw = 40) :
  members_in_two_activities c = 115 := by
  sorry

#eval members_in_two_activities ⟨150, 55, 90, 40⟩

end NUMINAMATH_CALUDE_two_activities_count_l3988_398827


namespace NUMINAMATH_CALUDE_total_pencils_l3988_398888

theorem total_pencils (jessica_pencils sandy_pencils jason_pencils : ℕ) :
  jessica_pencils = 8 → sandy_pencils = 8 → jason_pencils = 8 →
  jessica_pencils + sandy_pencils + jason_pencils = 24 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3988_398888


namespace NUMINAMATH_CALUDE_octal_to_decimal_l3988_398837

theorem octal_to_decimal (r : ℕ) : 175 = 120 + r → r = 5 := by
  sorry

end NUMINAMATH_CALUDE_octal_to_decimal_l3988_398837


namespace NUMINAMATH_CALUDE_max_side_length_triangle_l3988_398836

theorem max_side_length_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧                -- Three different side lengths
  a + b + c = 24 ∧               -- Perimeter is 24
  a ≥ 4 ∧                        -- Shortest side is at least 4
  c < a + b →                    -- Triangle inequality
  c ≤ 11 :=                      -- Maximum side length is 11
by sorry

end NUMINAMATH_CALUDE_max_side_length_triangle_l3988_398836


namespace NUMINAMATH_CALUDE_length_PR_l3988_398804

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def O : ℝ × ℝ := (0, 0)  -- Center of the circle
def radius : ℝ := 10

-- Define points P, Q, and R
variable (P Q R : ℝ × ℝ)

-- State the conditions
variable (h1 : P ∈ Circle O radius)
variable (h2 : Q ∈ Circle O radius)
variable (h3 : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 12^2)
variable (h4 : R ∈ Circle O radius)
variable (h5 : R.1 = (P.1 + Q.1) / 2 ∧ R.2 = (P.2 + Q.2) / 2)

-- State the theorem
theorem length_PR : (P.1 - R.1)^2 + (P.2 - R.2)^2 = 40 := by sorry

end NUMINAMATH_CALUDE_length_PR_l3988_398804


namespace NUMINAMATH_CALUDE_table_runner_coverage_l3988_398830

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) (coverage_percentage : ℝ) (three_layer_area : ℝ) :
  total_runner_area = 212 →
  table_area = 175 →
  coverage_percentage = 0.80 →
  three_layer_area = 24 →
  ∃ (two_layer_area : ℝ),
    two_layer_area = 48 ∧
    two_layer_area + three_layer_area + (coverage_percentage * table_area - two_layer_area - three_layer_area) = coverage_percentage * table_area ∧
    two_layer_area + three_layer_area = total_runner_area - (coverage_percentage * table_area - two_layer_area - three_layer_area) :=
by sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l3988_398830


namespace NUMINAMATH_CALUDE_three_integers_sum_and_ratio_l3988_398898

theorem three_integers_sum_and_ratio : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b + c = 90 ∧
  2 * b = 3 * a ∧
  2 * c = 5 * a ∧
  a = 18 ∧ b = 27 ∧ c = 45 := by
sorry

end NUMINAMATH_CALUDE_three_integers_sum_and_ratio_l3988_398898


namespace NUMINAMATH_CALUDE_simplify_expression_l3988_398818

theorem simplify_expression (a : ℝ) (h : a < -3) :
  Real.sqrt ((2 * a - 1)^2) + Real.sqrt ((a + 3)^2) = -3 * a - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3988_398818


namespace NUMINAMATH_CALUDE_cristina_croissants_l3988_398824

/-- The number of croissants Cristina baked -/
def total_croissants (num_guests : ℕ) (croissants_per_guest : ℕ) : ℕ :=
  num_guests * croissants_per_guest

/-- Proof that Cristina baked 14 croissants -/
theorem cristina_croissants :
  total_croissants 7 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_cristina_croissants_l3988_398824


namespace NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l3988_398808

theorem trigonometric_expression_evaluation :
  (Real.sqrt 3 * Real.tan (12 * π / 180) - 3) /
  (Real.sin (12 * π / 180) * (4 * Real.cos (12 * π / 180) ^ 2 - 2)) = -4 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_expression_evaluation_l3988_398808


namespace NUMINAMATH_CALUDE_workshop_average_salary_l3988_398887

theorem workshop_average_salary 
  (num_technicians : ℕ)
  (num_total_workers : ℕ)
  (avg_salary_technicians : ℚ)
  (avg_salary_others : ℚ)
  (h1 : num_technicians = 7)
  (h2 : num_total_workers = 22)
  (h3 : avg_salary_technicians = 1000)
  (h4 : avg_salary_others = 780) :
  (num_technicians * avg_salary_technicians + (num_total_workers - num_technicians) * avg_salary_others) / num_total_workers = 850 := by
sorry

end NUMINAMATH_CALUDE_workshop_average_salary_l3988_398887


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3988_398805

/-- Given an ellipse defined by the equation 4x² + y² = 16, 
    its major axis has length 8. -/
theorem ellipse_major_axis_length :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  (∀ x y : ℝ, 4 * x^2 + y^2 = 16 ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
  2 * max a b = 8 :=
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3988_398805


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3988_398845

theorem quadratic_equation_solution (p q : ℝ) 
  (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 9 / 2) : 
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3988_398845


namespace NUMINAMATH_CALUDE_period_of_sincos_sum_l3988_398865

/-- The period of y = 3sin(2x) + 4cos(2x) is π -/
theorem period_of_sincos_sum (x : ℝ) : 
  let y : ℝ → ℝ := λ x => 3 * Real.sin (2 * x) + 4 * Real.cos (2 * x)
  ∃ (p : ℝ), p > 0 ∧ ∀ (t : ℝ), y (t + p) = y t ∧ ∀ (q : ℝ), 0 < q ∧ q < p → ∃ (s : ℝ), y (s + q) ≠ y s :=
by sorry

end NUMINAMATH_CALUDE_period_of_sincos_sum_l3988_398865


namespace NUMINAMATH_CALUDE_set_operations_and_intersection_l3988_398885

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def C (a : ℝ) : Set ℝ := {x | a < x}

theorem set_operations_and_intersection :
  (A ∪ B = {x | 1 < x ∧ x ≤ 8}) ∧
  ((Aᶜ) ∩ B = {x | 1 < x ∧ x < 2}) ∧
  (∀ a : ℝ, (A ∩ C a).Nonempty ↔ a < 8) := by sorry

end NUMINAMATH_CALUDE_set_operations_and_intersection_l3988_398885


namespace NUMINAMATH_CALUDE_average_listening_time_is_55_minutes_l3988_398894

/-- Represents the distribution of audience members and their listening times --/
structure AudienceDistribution where
  total_audience : ℕ
  lecture_duration : ℕ
  full_listeners_percent : ℚ
  sleepers_percent : ℚ
  quarter_listeners_percent : ℚ
  half_listeners_percent : ℚ
  three_quarter_listeners_percent : ℚ

/-- Calculates the average listening time for the given audience distribution --/
def average_listening_time (dist : AudienceDistribution) : ℚ :=
  sorry

/-- The theorem to be proved --/
theorem average_listening_time_is_55_minutes 
  (dist : AudienceDistribution)
  (h1 : dist.total_audience = 200)
  (h2 : dist.lecture_duration = 90)
  (h3 : dist.full_listeners_percent = 30 / 100)
  (h4 : dist.sleepers_percent = 15 / 100)
  (h5 : dist.quarter_listeners_percent = (1 - dist.full_listeners_percent - dist.sleepers_percent) / 4)
  (h6 : dist.half_listeners_percent = (1 - dist.full_listeners_percent - dist.sleepers_percent) / 4)
  (h7 : dist.three_quarter_listeners_percent = 1 - dist.full_listeners_percent - dist.sleepers_percent - dist.quarter_listeners_percent - dist.half_listeners_percent)
  : average_listening_time dist = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_listening_time_is_55_minutes_l3988_398894


namespace NUMINAMATH_CALUDE_alice_prob_three_turns_correct_l3988_398825

/-- Represents the player who has the ball -/
inductive Player : Type
| Alice : Player
| Bob : Player

/-- The probability of keeping the ball for each player -/
def keep_prob (p : Player) : ℚ :=
  match p with
  | Player.Alice => 2/3
  | Player.Bob => 3/4

/-- The probability of tossing the ball for each player -/
def toss_prob (p : Player) : ℚ :=
  match p with
  | Player.Alice => 1/3
  | Player.Bob => 1/4

/-- The probability of Alice having the ball after three turns -/
def alice_prob_after_three_turns : ℚ := 203/432

theorem alice_prob_three_turns_correct :
  alice_prob_after_three_turns =
    keep_prob Player.Alice * keep_prob Player.Alice * keep_prob Player.Alice +
    toss_prob Player.Alice * toss_prob Player.Bob * keep_prob Player.Alice +
    keep_prob Player.Alice * toss_prob Player.Alice * toss_prob Player.Bob +
    toss_prob Player.Alice * keep_prob Player.Bob * toss_prob Player.Bob :=
by sorry

end NUMINAMATH_CALUDE_alice_prob_three_turns_correct_l3988_398825


namespace NUMINAMATH_CALUDE_prime_arithmetic_sequence_difference_l3988_398843

theorem prime_arithmetic_sequence_difference (p₁ p₂ p₃ d : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧
  p₁ > 3 ∧ p₂ > 3 ∧ p₃ > 3 ∧
  p₂ = p₁ + d ∧ p₃ = p₂ + d →
  6 ∣ d := by
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_sequence_difference_l3988_398843


namespace NUMINAMATH_CALUDE_zahra_kimmie_ratio_l3988_398848

def kimmie_earnings : ℚ := 450
def total_savings : ℚ := 375

theorem zahra_kimmie_ratio (zahra_earnings : ℚ) 
  (h1 : zahra_earnings < kimmie_earnings)
  (h2 : total_savings = (1/2) * kimmie_earnings + (1/2) * zahra_earnings) :
  zahra_earnings / kimmie_earnings = 2/3 := by
sorry

end NUMINAMATH_CALUDE_zahra_kimmie_ratio_l3988_398848


namespace NUMINAMATH_CALUDE_car_fuel_consumption_l3988_398890

/-- Represents the distance a car can travel with a given amount of fuel -/
def distance_traveled (fuel_fraction : ℚ) (distance : ℚ) : ℚ := distance / fuel_fraction

/-- Represents the remaining distance a car can travel -/
def remaining_distance (total_distance : ℚ) (traveled_distance : ℚ) : ℚ :=
  total_distance - traveled_distance

theorem car_fuel_consumption 
  (initial_distance : ℚ) 
  (initial_fuel_fraction : ℚ) 
  (h1 : initial_distance = 165) 
  (h2 : initial_fuel_fraction = 3/8) : 
  remaining_distance (distance_traveled 1 initial_fuel_fraction) initial_distance = 275 := by
  sorry

#eval remaining_distance (distance_traveled 1 (3/8)) 165

end NUMINAMATH_CALUDE_car_fuel_consumption_l3988_398890


namespace NUMINAMATH_CALUDE_number_of_keepers_l3988_398806

/-- Represents the number of feet for each animal type --/
def animalFeet : Nat → Nat
| 0 => 2  -- hen
| 1 => 4  -- goat
| 2 => 4  -- camel
| 3 => 8  -- spider
| 4 => 8  -- octopus
| _ => 0

/-- Represents the count of each animal type --/
def animalCount : Nat → Nat
| 0 => 50  -- hens
| 1 => 45  -- goats
| 2 => 8   -- camels
| 3 => 12  -- spiders
| 4 => 6   -- octopuses
| _ => 0

/-- Calculates the total number of animal feet --/
def totalAnimalFeet : Nat :=
  List.range 5
    |> List.map (fun i => animalFeet i * animalCount i)
    |> List.sum

/-- Calculates the total number of animal heads --/
def totalAnimalHeads : Nat :=
  List.range 5
    |> List.map animalCount
    |> List.sum

/-- Theorem stating the number of keepers in the caravan --/
theorem number_of_keepers :
  ∃ k : Nat,
    k = 39 ∧
    totalAnimalFeet + (2 * k - 2) = totalAnimalHeads + k + 372 :=
by
  sorry


end NUMINAMATH_CALUDE_number_of_keepers_l3988_398806


namespace NUMINAMATH_CALUDE_no_zero_roots_l3988_398868

theorem no_zero_roots : 
  (∀ x : ℝ, 5 * x^2 - 3 = 50 → x ≠ 0) ∧
  (∀ x : ℝ, (3*x - 1)^2 = (x - 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, x^2 - 9 ≥ 0 → 2*x - 2 ≥ 0 → x^2 - 9 = 2*x - 2 → x ≠ 0) := by
  sorry


end NUMINAMATH_CALUDE_no_zero_roots_l3988_398868


namespace NUMINAMATH_CALUDE_isosceles_triangle_determination_l3988_398841

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is isosceles with AB = AC -/
def isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2

/-- The incenter of a triangle -/
def incenter (t : Triangle) : Point :=
  sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  sorry

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point :=
  sorry

theorem isosceles_triangle_determination
  (I M H : Point) :
  ∃! (t : Triangle), isIsosceles t ∧
    incenter t = I ∧
    centroid t = M ∧
    orthocenter t = H :=
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_determination_l3988_398841


namespace NUMINAMATH_CALUDE_interest_equivalence_l3988_398873

/-- Simple interest calculation function -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The problem statement -/
theorem interest_equivalence (P : ℝ) : 
  simple_interest 100 0.05 8 = simple_interest P 0.10 2 → P = 200 := by
  sorry

end NUMINAMATH_CALUDE_interest_equivalence_l3988_398873


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3988_398851

/-- A regular polygon with side length 7 units and exterior angle 90 degrees has a perimeter of 28 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 7 ∧ 
  exterior_angle = 90 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 28 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3988_398851


namespace NUMINAMATH_CALUDE_jenny_essay_copies_l3988_398801

/-- Represents the problem of determining how many copies Jenny wants to print -/
theorem jenny_essay_copies : 
  let cost_per_page : ℚ := 1 / 10
  let essay_pages : ℕ := 25
  let num_pens : ℕ := 7
  let cost_per_pen : ℚ := 3 / 2
  let payment : ℕ := 2 * 20
  let change : ℕ := 12
  
  let total_spent : ℚ := payment - change
  let pen_cost : ℚ := num_pens * cost_per_pen
  let printing_cost : ℚ := total_spent - pen_cost
  let cost_per_copy : ℚ := cost_per_page * essay_pages
  let num_copies : ℚ := printing_cost / cost_per_copy
  
  num_copies = 7 := by sorry

end NUMINAMATH_CALUDE_jenny_essay_copies_l3988_398801


namespace NUMINAMATH_CALUDE_flour_needed_for_one_batch_l3988_398810

/-- The number of cups of flour needed for one batch of cookies -/
def flour_per_batch : ℝ := 4

/-- The number of cups of sugar needed for one batch of cookies -/
def sugar_per_batch : ℝ := 1.5

/-- The total number of cups of flour and sugar needed for 8 batches -/
def total_for_eight_batches : ℝ := 44

theorem flour_needed_for_one_batch :
  flour_per_batch = 4 :=
by
  have h1 : sugar_per_batch = 1.5 := rfl
  have h2 : total_for_eight_batches = 44 := rfl
  have h3 : 8 * flour_per_batch + 8 * sugar_per_batch = total_for_eight_batches := by sorry
  sorry

end NUMINAMATH_CALUDE_flour_needed_for_one_batch_l3988_398810


namespace NUMINAMATH_CALUDE_range_of_f_l3988_398875

def f (x : ℤ) : ℤ := x^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l3988_398875


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3988_398877

theorem rhombus_longer_diagonal (side_length : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side_length = 40 →
  shorter_diagonal = 30 →
  longer_diagonal = 10 * Real.sqrt 55 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l3988_398877


namespace NUMINAMATH_CALUDE_adams_shelves_l3988_398863

theorem adams_shelves (action_figures_per_shelf : ℕ) (num_shelves : ℕ) (total_capacity : ℕ) :
  action_figures_per_shelf = 11 →
  num_shelves = 4 →
  total_capacity = action_figures_per_shelf * num_shelves →
  total_capacity = 44 :=
by sorry

end NUMINAMATH_CALUDE_adams_shelves_l3988_398863


namespace NUMINAMATH_CALUDE_ninth_term_value_l3988_398880

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  S : ℕ → ℚ  -- The sum function
  sum_def : ∀ n, S n = (n * (a 1 + a n)) / 2
  arith_def : ∀ n, a (n + 1) - a n = a 2 - a 1

/-- The main theorem -/
theorem ninth_term_value (seq : ArithmeticSequence) 
    (h6 : seq.S 6 = 3) 
    (h11 : seq.S 11 = 18) : 
  seq.a 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_value_l3988_398880


namespace NUMINAMATH_CALUDE_intersection_A_B_l3988_398846

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| > 2}
def B : Set ℝ := {x | x * (x - 5) < 0}

-- State the theorem
theorem intersection_A_B :
  A ∩ B = {x : ℝ | 3 < x ∧ x < 5} :=
by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3988_398846


namespace NUMINAMATH_CALUDE_existence_of_irrational_sum_l3988_398831

theorem existence_of_irrational_sum (n : ℕ) (a : Fin n → ℝ) :
  ∃ (x : ℝ), ∀ (i : Fin n), Irrational (x + a i) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_irrational_sum_l3988_398831


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3988_398870

theorem fraction_equivalence (k : ℝ) (h : k ≠ -5) :
  (k + 3) / (k + 5) = 3 / 5 ↔ k = 0 := by
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3988_398870


namespace NUMINAMATH_CALUDE_flock_size_lcm_equals_min_ducks_l3988_398840

/-- Represents the flock size of ducks -/
def duck_flock_size : ℕ := 18

/-- Represents the flock size of seagulls -/
def seagull_flock_size : ℕ := 10

/-- Represents the smallest number of ducks observed -/
def min_ducks_observed : ℕ := 90

/-- Theorem stating that the least common multiple of the flock sizes
    is equal to the smallest number of ducks observed -/
theorem flock_size_lcm_equals_min_ducks :
  Nat.lcm duck_flock_size seagull_flock_size = min_ducks_observed := by
  sorry

end NUMINAMATH_CALUDE_flock_size_lcm_equals_min_ducks_l3988_398840


namespace NUMINAMATH_CALUDE_darryl_break_even_point_l3988_398884

/-- Calculates the break-even point for Darryl's machine sales -/
theorem darryl_break_even_point 
  (parts_cost : ℕ) 
  (patent_cost : ℕ) 
  (selling_price : ℕ) 
  (h1 : parts_cost = 3600)
  (h2 : patent_cost = 4500)
  (h3 : selling_price = 180) :
  (parts_cost + patent_cost) / selling_price = 45 :=
by sorry

end NUMINAMATH_CALUDE_darryl_break_even_point_l3988_398884


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3988_398861

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x > 0}
def B : Set ℝ := {x : ℝ | x < 4}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 < x ∧ x < 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3988_398861


namespace NUMINAMATH_CALUDE_elmo_has_24_books_l3988_398871

def elmos_books (elmo_multiplier laura_multiplier stu_books : ℕ) : ℕ :=
  elmo_multiplier * (laura_multiplier * stu_books)

theorem elmo_has_24_books :
  elmos_books 3 2 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_elmo_has_24_books_l3988_398871


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l3988_398869

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (∀ n, a n > 0) →
  a 1 * a 99 = 16 →
  a 1 + a 99 = 10 →
  a 20 * a 50 * a 80 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l3988_398869


namespace NUMINAMATH_CALUDE_cubic_function_properties_l3988_398896

/-- A cubic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * x + 4

/-- The function reaches an extreme value at x = 2 -/
def extreme_at_2 (a b : ℝ) : Prop := f a b 2 = -4/3

/-- The derivative of f is zero at x = 2 -/
def derivative_zero_at_2 (a b : ℝ) : Prop := 3 * a * 2^2 - b = 0

theorem cubic_function_properties (a b : ℝ) 
  (h1 : extreme_at_2 a b) 
  (h2 : derivative_zero_at_2 a b) :
  (∀ x, f a b x = (1/3) * x^3 - 4 * x + 4) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a b x ≥ f a b y) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a b x ≤ f a b y) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f a b x = 28/3) ∧
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, f a b x = -4/3) :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l3988_398896


namespace NUMINAMATH_CALUDE_investment_calculation_l3988_398811

/-- Given two investors p and q, where p invested 52000 and the profit is divided in the ratio 4:5,
    prove that q invested 65000. -/
theorem investment_calculation (p q : ℕ) : 
  p = 52000 → 
  (4 : ℚ) / 5 = p / q →
  q = 65000 := by
sorry

end NUMINAMATH_CALUDE_investment_calculation_l3988_398811


namespace NUMINAMATH_CALUDE_derivative_of_f_l3988_398838

noncomputable def f (x : ℝ) : ℝ := 3 * Real.log x + x^2

theorem derivative_of_f :
  deriv f = λ x => 3 / x + 2 * x :=
by sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3988_398838


namespace NUMINAMATH_CALUDE_quadratic_root_k_value_l3988_398882

theorem quadratic_root_k_value (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 + 3 * x - k = 0) ∧ (2 * 4^2 + 3 * 4 - k = 0) → k = 44 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_k_value_l3988_398882


namespace NUMINAMATH_CALUDE_mixture_proportion_l3988_398878

/-- Represents a solution with a given percentage of chemical a -/
structure Solution :=
  (percent_a : ℝ)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (sol_x : Solution)
  (sol_y : Solution)
  (percent_x : ℝ)
  (percent_mixture_a : ℝ)

/-- The theorem stating the proportion of solution x in the mixture -/
theorem mixture_proportion
  (mix : Mixture)
  (hx : mix.sol_x.percent_a = 0.3)
  (hy : mix.sol_y.percent_a = 0.4)
  (hm : mix.percent_mixture_a = 0.32)
  : mix.percent_x = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_mixture_proportion_l3988_398878


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3988_398802

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x ≥ 1 → x^2 - 1 < 0) ↔ (∃ x : ℝ, x ≥ 1 ∧ x^2 - 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3988_398802


namespace NUMINAMATH_CALUDE_battle_station_staffing_l3988_398835

theorem battle_station_staffing (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (n.factorial / (n - k).factorial) = 30240 := by
  sorry

end NUMINAMATH_CALUDE_battle_station_staffing_l3988_398835


namespace NUMINAMATH_CALUDE_striped_shirt_ratio_l3988_398833

theorem striped_shirt_ratio (total : ℕ) (checkered shorts striped : ℕ) : 
  total = 81 →
  total = checkered + striped →
  shorts = checkered + 19 →
  striped = shorts + 8 →
  striped * 3 = total * 2 := by
sorry

end NUMINAMATH_CALUDE_striped_shirt_ratio_l3988_398833


namespace NUMINAMATH_CALUDE_xiaohong_fruit_money_l3988_398867

/-- The price difference between 500g of apples and 500g of pears in yuan -/
def price_difference : ℚ := 55 / 100

/-- The amount saved when buying 5 kg of apples in yuan -/
def apple_savings : ℚ := 4

/-- The amount saved when buying 6 kg of pears in yuan -/
def pear_savings : ℚ := 3

/-- The price of 1 kg of pears in yuan -/
def pear_price : ℚ := 45 / 10

theorem xiaohong_fruit_money : 
  ∃ (total : ℚ), 
    total = 6 * pear_price - pear_savings ∧ 
    total = 5 * (pear_price + 2 * price_difference) - apple_savings ∧
    total = 24 := by
  sorry

end NUMINAMATH_CALUDE_xiaohong_fruit_money_l3988_398867


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3988_398812

theorem sum_of_squares_of_roots (a b c d : ℝ) : 
  (3 * a^4 - 6 * a^3 + 11 * a^2 + 15 * a - 7 = 0) →
  (3 * b^4 - 6 * b^3 + 11 * b^2 + 15 * b - 7 = 0) →
  (3 * c^4 - 6 * c^3 + 11 * c^2 + 15 * c - 7 = 0) →
  (3 * d^4 - 6 * d^3 + 11 * d^2 + 15 * d - 7 = 0) →
  a^2 + b^2 + c^2 + d^2 = -10/3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3988_398812


namespace NUMINAMATH_CALUDE_equation_solution_l3988_398879

theorem equation_solution (a b : ℤ) : 
  (((a + 2 : ℚ) / (b + 1) + (a + 1 : ℚ) / (b + 2) = 1 + 6 / (a + b + 1)) ∧ 
   (b + 1 ≠ 0) ∧ (b + 2 ≠ 0) ∧ (a + b + 1 ≠ 0)) ↔ 
  ((∃ t : ℤ, t ≠ 0 ∧ t ≠ -1 ∧ a = -3 - t ∧ b = t) ∨ (a = 1 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3988_398879


namespace NUMINAMATH_CALUDE_identity_condition_l3988_398864

/-- 
Proves that the equation (3x-a)(2x+5)-x = 6x^2+2(5x-b) is an identity 
for all x if and only if a = 2 and b = 5.
-/
theorem identity_condition (a b : ℝ) : 
  (∀ x : ℝ, (3*x - a)*(2*x + 5) - x = 6*x^2 + 2*(5*x - b)) ↔ (a = 2 ∧ b = 5) := by
  sorry

end NUMINAMATH_CALUDE_identity_condition_l3988_398864


namespace NUMINAMATH_CALUDE_sin_cos_sum_11_19_l3988_398839

theorem sin_cos_sum_11_19 : 
  Real.sin (11 * π / 180) * Real.cos (19 * π / 180) + 
  Real.cos (11 * π / 180) * Real.sin (19 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_sum_11_19_l3988_398839


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l3988_398853

def polynomial (x : ℝ) := 3 * (x^4 + 2*x^3 + 5*x^2 + 2)

theorem sum_of_squares_of_coefficients : 
  (3^2 : ℝ) + (6^2 : ℝ) + (15^2 : ℝ) + (6^2 : ℝ) = 306 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l3988_398853


namespace NUMINAMATH_CALUDE_chromatid_non_separation_can_result_in_XXY_l3988_398828

/- Define the basic types and structures -/
inductive Chromosome
| X
| Y

structure Sperm :=
(chromosomes : List Chromosome)

structure Egg :=
(chromosomes : List Chromosome)

structure Offspring :=
(chromosomes : List Chromosome)

/- Define the process of sperm formation with non-separation of chromatids -/
def spermFormationWithNonSeparation : List Sperm :=
[{chromosomes := [Chromosome.X, Chromosome.X]}, {chromosomes := [Chromosome.Y, Chromosome.Y]}]

/- Define a normal egg -/
def normalEgg : Egg :=
{chromosomes := [Chromosome.X]}

/- Define the fertilization process -/
def fertilize (sperm : Sperm) (egg : Egg) : Offspring :=
{chromosomes := sperm.chromosomes ++ egg.chromosomes}

/- The theorem to be proved -/
theorem chromatid_non_separation_can_result_in_XXY :
  ∃ (sperm : Sperm) (egg : Egg),
    sperm ∈ spermFormationWithNonSeparation ∧
    egg = normalEgg ∧
    (fertilize sperm egg).chromosomes = [Chromosome.X, Chromosome.X, Chromosome.Y] :=
sorry

end NUMINAMATH_CALUDE_chromatid_non_separation_can_result_in_XXY_l3988_398828


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_hundred_is_one_fourth_more_than_fiftysix_l3988_398874

theorem thirty_percent_less_than_hundred_is_one_fourth_more_than_fiftysix : ∃ x : ℝ, 
  (100 - 0.3 * 100 = x + 0.25 * x) ∧ x = 56 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_hundred_is_one_fourth_more_than_fiftysix_l3988_398874


namespace NUMINAMATH_CALUDE_square_sum_and_reciprocal_l3988_398800

theorem square_sum_and_reciprocal (x : ℝ) (h : x + (1/x) = 2) : x^2 + (1/x^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_and_reciprocal_l3988_398800


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bounds_l3988_398883

theorem arithmetic_geometric_mean_difference_bounds (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - Real.sqrt (a * b) ∧
  (a + b) / 2 - Real.sqrt (a * b) < (a - b)^2 / (8 * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_difference_bounds_l3988_398883


namespace NUMINAMATH_CALUDE_sum_of_twos_and_threes_3024_l3988_398856

/-- The number of ways to write a positive integer as an unordered sum of 2s and 3s -/
def sumOfTwosAndThrees (n : ℕ) : ℕ := 
  (n / 3 : ℕ) - (n % 3) / 3 + 1

/-- Theorem stating that there are 337 ways to write 3024 as an unordered sum of 2s and 3s -/
theorem sum_of_twos_and_threes_3024 : sumOfTwosAndThrees 3024 = 337 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_twos_and_threes_3024_l3988_398856


namespace NUMINAMATH_CALUDE_cookies_per_batch_is_three_l3988_398897

/-- Given the total number of chocolate chips, number of batches, and chips per cookie,
    calculate the number of cookies in a batch. -/
def cookiesPerBatch (totalChips : ℕ) (numBatches : ℕ) (chipsPerCookie : ℕ) : ℕ :=
  (totalChips / numBatches) / chipsPerCookie

/-- Prove that the number of cookies in a batch is 3 given the problem conditions. -/
theorem cookies_per_batch_is_three :
  cookiesPerBatch 81 3 9 = 3 := by
  sorry

#eval cookiesPerBatch 81 3 9

end NUMINAMATH_CALUDE_cookies_per_batch_is_three_l3988_398897


namespace NUMINAMATH_CALUDE_find_constant_k_l3988_398815

theorem find_constant_k (c : ℝ) (k : ℝ) :
  c = 2 →
  (∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - c)*(x - 4)) →
  k = -16 := by
  sorry

end NUMINAMATH_CALUDE_find_constant_k_l3988_398815


namespace NUMINAMATH_CALUDE_binomial_p_value_l3988_398820

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV) : ℝ := X.n * X.p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_p_value (X : BinomialRV) 
  (h2 : expected_value X = 30)
  (h3 : variance X = 20) : 
  X.p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_binomial_p_value_l3988_398820


namespace NUMINAMATH_CALUDE_compound_composition_l3988_398862

/-- Represents the number of atoms of each element in a compound -/
structure Compound where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (carbonWeight hydrogenWeight oxygenWeight : ℝ) : ℝ :=
  c.carbon * carbonWeight + c.hydrogen * hydrogenWeight + c.oxygen * oxygenWeight

theorem compound_composition :
  ∃ (c : Compound),
    c.hydrogen = 8 ∧
    c.oxygen = 7 ∧
    molecularWeight c 12.01 1.01 16.00 = 192 ∧
    c.carbon = 6 := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l3988_398862


namespace NUMINAMATH_CALUDE_unpainted_cubes_6x6x6_l3988_398872

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_cubes : Nat
  painted_per_face : Nat
  strip_width : Nat
  strip_length : Nat

/-- Calculate the number of unpainted cubes in a painted cube -/
def unpainted_cubes (c : PaintedCube) : Nat :=
  sorry

/-- Theorem stating the number of unpainted cubes in the specific problem -/
theorem unpainted_cubes_6x6x6 :
  let c : PaintedCube := {
    size := 6,
    total_cubes := 216,
    painted_per_face := 10,
    strip_width := 2,
    strip_length := 5
  }
  unpainted_cubes c = 186 := by
  sorry

end NUMINAMATH_CALUDE_unpainted_cubes_6x6x6_l3988_398872


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3988_398886

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the third and fourth terms is 1/2 -/
def third_fourth_sum (a : ℕ → ℚ) : Prop :=
  a 3 + a 4 = 1/2

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a → third_fourth_sum a → a 1 + a 6 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3988_398886


namespace NUMINAMATH_CALUDE_f_sum_positive_l3988_398866

def f (x : ℝ) : ℝ := x^2015

theorem f_sum_positive (a b : ℝ) (h : a + b > 0) : f a + f b > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_positive_l3988_398866


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3988_398857

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 6)
  (h_S : S = 42)
  (h_sum : S = a / (1 - r))
  (h_convergence : abs r < 1) :
  a = 35 :=
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3988_398857


namespace NUMINAMATH_CALUDE_foil_covered_prism_width_l3988_398876

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The inner core of the prism not touching tin foil -/
def inner : PrismDimensions :=
  { length := 2^(5/3),
    width := 2^(8/3),
    height := 2^(5/3) }

/-- The outer prism covered in tin foil -/
def outer : PrismDimensions :=
  { length := inner.length + 2,
    width := inner.width + 2,
    height := inner.height + 2 }

theorem foil_covered_prism_width :
  (inner.length * inner.width * inner.height = 128) →
  (inner.width = 2 * inner.length) →
  (inner.width = 2 * inner.height) →
  (outer.width = 10) := by
  sorry

end NUMINAMATH_CALUDE_foil_covered_prism_width_l3988_398876


namespace NUMINAMATH_CALUDE_evaluate_expression_l3988_398822

theorem evaluate_expression : -(18 / 3 * 8 - 70 + 5 * 7) = -13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3988_398822


namespace NUMINAMATH_CALUDE_dagger_five_eighths_three_fourths_l3988_398899

-- Define the operation †
def dagger (m n p q : ℚ) : ℚ := 2 * m * p * (q / n)

-- Theorem statement
theorem dagger_five_eighths_three_fourths :
  dagger (5/8) (8/8) (3/4) (4/4) = 15 := by
  sorry

end NUMINAMATH_CALUDE_dagger_five_eighths_three_fourths_l3988_398899


namespace NUMINAMATH_CALUDE_product_selection_theorem_l3988_398803

def total_products : ℕ := 10
def defective_products : ℕ := 3
def good_products : ℕ := 7
def products_drawn : ℕ := 5

theorem product_selection_theorem :
  (∃ (no_defective : ℕ) (exactly_two_defective : ℕ) (at_least_one_defective : ℕ),
    -- No defective products
    no_defective = Nat.choose good_products products_drawn ∧
    -- Exactly 2 defective products
    exactly_two_defective = Nat.choose defective_products 2 * Nat.choose good_products 3 ∧
    -- At least 1 defective product
    at_least_one_defective = Nat.choose total_products products_drawn - Nat.choose good_products products_drawn) :=
by
  sorry

end NUMINAMATH_CALUDE_product_selection_theorem_l3988_398803


namespace NUMINAMATH_CALUDE_train_speed_problem_l3988_398850

/-- Proves the speed of the second train given the conditions of the problem -/
theorem train_speed_problem (train_length : ℝ) (train1_speed : ℝ) (passing_time : ℝ) :
  train_length = 210 →
  train1_speed = 90 →
  passing_time = 8.64 →
  ∃ train2_speed : ℝ,
    train2_speed = 85 ∧
    (train_length * 2) / passing_time * 3.6 = train1_speed + train2_speed :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l3988_398850


namespace NUMINAMATH_CALUDE_modular_home_cost_l3988_398889

/-- Calculates the cost of a modular home given specific module costs and sizes. -/
theorem modular_home_cost 
  (kitchen_size : ℕ) (kitchen_cost : ℕ)
  (bathroom_size : ℕ) (bathroom_cost : ℕ)
  (other_cost_per_sqft : ℕ)
  (total_size : ℕ) (num_bathrooms : ℕ) : 
  kitchen_size = 400 →
  kitchen_cost = 20000 →
  bathroom_size = 150 →
  bathroom_cost = 12000 →
  other_cost_per_sqft = 100 →
  total_size = 2000 →
  num_bathrooms = 2 →
  (kitchen_cost + num_bathrooms * bathroom_cost + 
   (total_size - kitchen_size - num_bathrooms * bathroom_size) * other_cost_per_sqft) = 174000 :=
by sorry

end NUMINAMATH_CALUDE_modular_home_cost_l3988_398889


namespace NUMINAMATH_CALUDE_work_completion_time_l3988_398893

/-- Given that two workers A and B can complete a work together in a certain number of days,
    and worker A can complete the work alone in a certain number of days,
    this function calculates the number of days worker B would take to complete the work alone. -/
def days_for_B (days_together days_A_alone : ℚ) : ℚ :=
  1 / (1 / days_together - 1 / days_A_alone)

/-- Theorem stating that if A and B can complete a work in 12 days, and A alone can complete
    the work in 20 days, then B alone will complete the work in 30 days. -/
theorem work_completion_time :
  days_for_B 12 20 = 30 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3988_398893


namespace NUMINAMATH_CALUDE_simplified_rational_expression_l3988_398858

theorem simplified_rational_expression (x : ℝ) 
  (h1 : x^2 - 5*x + 6 ≠ 0) 
  (h2 : x^2 - 7*x + 12 ≠ 0) 
  (h3 : x^2 - 5*x + 4 ≠ 0) : 
  (x^2 - 3*x + 2) / (x^2 - 5*x + 6) / ((x^2 - 5*x + 4) / (x^2 - 7*x + 12)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplified_rational_expression_l3988_398858


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l3988_398860

theorem subtraction_preserves_inequality (a b : ℝ) (h : a > b) : a - 3 > b - 3 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l3988_398860


namespace NUMINAMATH_CALUDE_cafeteria_pie_problem_l3988_398813

/-- Given a cafeteria with initial apples, some handed out, and a number of pies made,
    calculate the number of apples used per pie. -/
def apples_per_pie (initial_apples : ℕ) (handed_out : ℕ) (pies_made : ℕ) : ℕ :=
  (initial_apples - handed_out) / pies_made

/-- Theorem: In the specific case of 50 initial apples, 5 handed out, and 9 pies made,
    the number of apples per pie is 5. -/
theorem cafeteria_pie_problem :
  apples_per_pie 50 5 9 = 5 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pie_problem_l3988_398813


namespace NUMINAMATH_CALUDE_workshop_production_balance_l3988_398817

theorem workshop_production_balance :
  let total_workers : ℕ := 85
  let type_a_rate : ℕ := 16
  let type_b_rate : ℕ := 10
  let set_a_parts : ℕ := 2
  let set_b_parts : ℕ := 3
  let workers_a : ℕ := 25
  let workers_b : ℕ := 60
  (total_workers = workers_a + workers_b) ∧
  ((type_a_rate * workers_a) / set_a_parts = (type_b_rate * workers_b) / set_b_parts) := by
  sorry

end NUMINAMATH_CALUDE_workshop_production_balance_l3988_398817


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3988_398849

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  1/a + 4/b ≥ 3 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 3 ∧ 1/a₀ + 4/b₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3988_398849


namespace NUMINAMATH_CALUDE_robertson_seymour_theorem_l3988_398855

-- Define a graph type
structure Graph (V : Type) where
  edge : V → V → Prop

-- Define a complete graph
def CompleteGraph (n : ℕ) : Graph (Fin n) where
  edge i j := i ≠ j

-- Define the concept of a minor
def IsMinor {V W : Type} (G : Graph V) (H : Graph W) : Prop := sorry

-- Define tree decomposition
structure TreeDecomposition (V : Type) where
  T : Type
  bags : T → Set V
  -- Other properties of tree decomposition

-- Define k-almost embeddable
def KAlmostEmbeddable (k : ℕ) (G : Graph V) (S : Type) : Prop := sorry

-- Define the concept of a surface where K^n cannot be embedded
def SurfaceWithoutKn (n : ℕ) (S : Type) : Prop := sorry

-- The main theorem
theorem robertson_seymour_theorem {V : Type} (n : ℕ) (hn : n ≥ 5) :
  ∃ k : ℕ, ∀ (G : Graph V),
    ¬IsMinor G (CompleteGraph n) →
    ∃ (td : TreeDecomposition V) (S : Type),
      SurfaceWithoutKn n S ∧
      KAlmostEmbeddable k G S :=
sorry

end NUMINAMATH_CALUDE_robertson_seymour_theorem_l3988_398855


namespace NUMINAMATH_CALUDE_prob_red_then_black_custom_deck_l3988_398814

/-- A deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- The probability of drawing a red card first and then a black card from a shuffled deck -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) * (d.black_cards : ℚ) / ((d.total_cards : ℚ) * (d.total_cards - 1 : ℚ))

/-- The theorem stating the probability for the given deck -/
theorem prob_red_then_black_custom_deck :
  let d : Deck := ⟨60, 30, 30⟩
  prob_red_then_black d = 15 / 59 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_black_custom_deck_l3988_398814


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l3988_398895

theorem chocolate_box_problem (total : ℕ) (caramels : ℕ) (nougats : ℕ) (truffles : ℕ) (peanut_clusters : ℕ) :
  total = 50 →
  caramels = 3 →
  nougats = 2 * caramels →
  truffles = caramels + (truffles - caramels) →
  peanut_clusters = (64 * total) / 100 →
  total = caramels + nougats + truffles + peanut_clusters →
  truffles - caramels = 6 := by
sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l3988_398895


namespace NUMINAMATH_CALUDE_probability_of_successful_meeting_l3988_398832

/-- Friend's train arrival time in minutes after 1:00 -/
def FriendArrivalTime : Type := {t : ℝ // 0 ≤ t ∧ t ≤ 60}

/-- Alex's arrival time in minutes after 1:00 -/
def AlexArrivalTime : Type := {t : ℝ // 0 ≤ t ∧ t ≤ 120}

/-- The waiting time of the friend's train in minutes -/
def WaitingTime : ℝ := 10

/-- The event that Alex arrives while the friend's train is still at the station -/
def SuccessfulMeeting (f : FriendArrivalTime) (a : AlexArrivalTime) : Prop :=
  f.val ≤ a.val ∧ a.val ≤ f.val + WaitingTime

/-- The probability measure for the problem -/
noncomputable def P : Set (FriendArrivalTime × AlexArrivalTime) → ℝ := sorry

/-- The theorem stating the probability of a successful meeting -/
theorem probability_of_successful_meeting :
  P {p : FriendArrivalTime × AlexArrivalTime | SuccessfulMeeting p.1 p.2} = 1/4 := by sorry

end NUMINAMATH_CALUDE_probability_of_successful_meeting_l3988_398832


namespace NUMINAMATH_CALUDE_march_greatest_drop_l3988_398823

/-- Represents the months of the year --/
inductive Month
| january | february | march | april | may | june | july

/-- The price change for each month --/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.january  => -0.75
  | Month.february => 1.50
  | Month.march    => -3.00
  | Month.april    => 2.50
  | Month.may      => -1.00
  | Month.june     => 0.50
  | Month.july     => -2.50

/-- The set of months considered in the problem --/
def considered_months : List Month :=
  [Month.january, Month.february, Month.march, Month.april, Month.may, Month.june, Month.july]

/-- Predicate to check if a month has a price drop --/
def has_price_drop (m : Month) : Prop :=
  price_change m < 0

/-- The theorem stating that March had the greatest monthly drop in price --/
theorem march_greatest_drop :
  ∀ m ∈ considered_months, has_price_drop m →
    price_change Month.march ≤ price_change m :=
  sorry

end NUMINAMATH_CALUDE_march_greatest_drop_l3988_398823


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3988_398807

theorem arithmetic_geometric_sequence : 
  ∃ (a b c d e : ℚ), 
    -- The five numbers
    a = 4 ∧ b = 8 ∧ c = 12 ∧ d = 16 ∧ e = 64/3 ∧
    -- First four form an arithmetic progression
    (b - a = c - b) ∧ (c - b = d - c) ∧
    -- Sum of first four is 40
    (a + b + c + d = 40) ∧
    -- Last three form a geometric progression
    (c^2 = b * d) ∧
    -- Product of outer terms of geometric progression is 32 times the second number
    (c * e = 32 * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l3988_398807


namespace NUMINAMATH_CALUDE_sum_and_simplest_form_l3988_398816

theorem sum_and_simplest_form :
  ∃ (n d : ℕ), n > 0 ∧ d > 0 ∧ (2 : ℚ) / 3 + (7 : ℚ) / 8 = (n : ℚ) / d ∧ 
  ∀ (n' d' : ℕ), n' > 0 → d' > 0 → (n' : ℚ) / d' = (n : ℚ) / d → n' ≥ n ∧ d' ≥ d :=
by
  sorry

end NUMINAMATH_CALUDE_sum_and_simplest_form_l3988_398816


namespace NUMINAMATH_CALUDE_asparagus_per_plate_l3988_398829

theorem asparagus_per_plate
  (bridgette_guests : ℕ)
  (alex_guests : ℕ)
  (extra_plates : ℕ)
  (total_asparagus : ℕ)
  (h1 : bridgette_guests = 84)
  (h2 : alex_guests = 2 * bridgette_guests / 3)
  (h3 : extra_plates = 10)
  (h4 : total_asparagus = 1200) :
  total_asparagus / (bridgette_guests + alex_guests + extra_plates) = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_asparagus_per_plate_l3988_398829


namespace NUMINAMATH_CALUDE_subset_implies_m_range_l3988_398826

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_implies_m_range (m : ℝ) : B m ⊆ A → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_range_l3988_398826


namespace NUMINAMATH_CALUDE_hexagon_area_division_l3988_398891

/-- A hexagon constructed from unit squares -/
structure Hexagon :=
  (area : ℝ)
  (line_PQ : ℝ → ℝ)
  (area_below : ℝ)
  (area_above : ℝ)
  (XQ : ℝ)
  (QY : ℝ)

/-- The theorem statement -/
theorem hexagon_area_division (h : Hexagon) :
  h.area = 8 ∧
  h.area_below = h.area_above ∧
  h.area_below = 1 + (1/2 * 4 * (3/2)) ∧
  h.XQ + h.QY = 4 →
  h.XQ / h.QY = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_hexagon_area_division_l3988_398891


namespace NUMINAMATH_CALUDE_min_cost_for_equal_distribution_l3988_398847

def tangerines_needed (initial : ℕ) (people : ℕ) : ℕ :=
  (people - initial % people) % people

def cost_of_additional_tangerines (initial : ℕ) (people : ℕ) (price : ℕ) : ℕ :=
  tangerines_needed initial people * price

theorem min_cost_for_equal_distribution (initial : ℕ) (people : ℕ) (price : ℕ) 
  (h1 : initial = 98) (h2 : people = 12) (h3 : price = 450) :
  cost_of_additional_tangerines initial people price = 4500 := by
  sorry

end NUMINAMATH_CALUDE_min_cost_for_equal_distribution_l3988_398847


namespace NUMINAMATH_CALUDE_power_multiplication_l3988_398859

theorem power_multiplication (m : ℝ) : m^3 * m^2 = m^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3988_398859
