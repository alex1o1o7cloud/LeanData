import Mathlib

namespace frog_count_frog_count_correct_l2898_289854

theorem frog_count (num_crocodiles : ℕ) (total_eyes : ℕ) (frog_eyes : ℕ) (crocodile_eyes : ℕ) : ℕ :=
  let num_frogs := (total_eyes - num_crocodiles * crocodile_eyes) / frog_eyes
  num_frogs

theorem frog_count_correct :
  frog_count 6 52 2 2 = 20 := by sorry

end frog_count_frog_count_correct_l2898_289854


namespace second_meeting_time_is_four_minutes_l2898_289812

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the swimming pool scenario --/
structure PoolScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  firstMeetingTime : ℝ
  firstMeetingPosition : ℝ

/-- Calculates the time of the second meeting --/
def secondMeetingTime (scenario : PoolScenario) : ℝ :=
  sorry

/-- Theorem stating that the second meeting occurs 4 minutes after starting --/
theorem second_meeting_time_is_four_minutes (scenario : PoolScenario) 
    (h1 : scenario.poolLength = 50)
    (h2 : scenario.swimmer1.startPosition = 0)
    (h3 : scenario.swimmer2.startPosition = 50)
    (h4 : scenario.firstMeetingTime = 2)
    (h5 : scenario.firstMeetingPosition = 20) :
    secondMeetingTime scenario = 4 := by
  sorry

end second_meeting_time_is_four_minutes_l2898_289812


namespace max_regions_correct_l2898_289869

/-- The maximum number of regions into which n circles can divide the plane -/
def max_regions (n : ℕ) : ℕ := n^2 - n + 2

/-- Theorem stating that max_regions gives the maximum number of regions -/
theorem max_regions_correct (n : ℕ) :
  max_regions n = n^2 - n + 2 :=
by sorry

end max_regions_correct_l2898_289869


namespace combination_sum_equals_462_l2898_289823

theorem combination_sum_equals_462 : 
  (Nat.choose 4 4) + (Nat.choose 5 4) + (Nat.choose 6 4) + (Nat.choose 7 4) + 
  (Nat.choose 8 4) + (Nat.choose 9 4) + (Nat.choose 10 4) = 462 := by
  sorry

end combination_sum_equals_462_l2898_289823


namespace ivanov_family_problem_l2898_289811

/-- The Ivanov family problem -/
theorem ivanov_family_problem (father mother daughter : ℕ) : 
  father + mother + daughter = 74 →  -- Current sum of ages
  father + mother + daughter - 30 = 47 →  -- Sum of ages 10 years ago
  mother - 26 = daughter →  -- Mother's age at daughter's birth
  mother = 33 := by
  sorry

end ivanov_family_problem_l2898_289811


namespace problem_statement_l2898_289894

theorem problem_statement :
  (∀ x : ℝ, x > 0 → x > Real.sin x) ∧
  (¬(∀ x : ℝ, x > 0 → x - Real.log x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x - Real.log x ≤ 0)) ∧
  (∀ p q : Prop, (p ∨ q → p ∧ q) → False) ∧
  (∀ p q : Prop, p ∧ q → p ∨ q) ∧
  (∀ a b : ℝ, (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0)) := by
  sorry

end problem_statement_l2898_289894


namespace milk_calculation_l2898_289880

theorem milk_calculation (initial : ℚ) (given : ℚ) (received : ℚ) :
  initial = 5 →
  given = 18 / 4 →
  received = 7 / 4 →
  initial - given + received = 9 / 4 := by
sorry

end milk_calculation_l2898_289880


namespace min_value_f_min_value_sum_squares_l2898_289889

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the minimum value of f(x)
theorem min_value_f : 
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 5 :=
sorry

-- Theorem for the minimum value of a^2 + 2b^2 + 3c^2
theorem min_value_sum_squares :
  ∃ m : ℝ, m = 15/2 ∧
  (∀ a b c : ℝ, a + 2*b + c = 5 → a^2 + 2*b^2 + 3*c^2 ≥ m) ∧
  (∃ a b c : ℝ, a + 2*b + c = 5 ∧ a^2 + 2*b^2 + 3*c^2 = m) :=
sorry

end min_value_f_min_value_sum_squares_l2898_289889


namespace sum_of_xyz_l2898_289864

/-- An arithmetic sequence with six terms where the first term is 4 and the last term is 31 -/
def arithmetic_sequence (x y z : ℝ) : Prop :=
  let d := (31 - 4) / 5
  (y - x = d) ∧ (16 - y = d) ∧ (z - 16 = d)

/-- The theorem stating that the sum of x, y, and z in the given arithmetic sequence is 45.6 -/
theorem sum_of_xyz (x y z : ℝ) (h : arithmetic_sequence x y z) : x + y + z = 45.6 := by
  sorry

end sum_of_xyz_l2898_289864


namespace delaney_missed_bus_l2898_289818

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem delaney_missed_bus (busLeaveTime : Time) (travelTime : Nat) (leftHomeTime : Time) :
  busLeaveTime = { hours := 8, minutes := 0 } →
  travelTime = 30 →
  leftHomeTime = { hours := 7, minutes := 50 } →
  timeDifference (addMinutes leftHomeTime travelTime) busLeaveTime = 20 := by
  sorry

end delaney_missed_bus_l2898_289818


namespace parabola_directrix_l2898_289833

/-- Given a parabola with equation y² = 2x, its directrix has the equation x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 2*x → (∃ (p : ℝ), p > 0 ∧ y^2 = 4*p*x ∧ x = -1/2) := by
  sorry

end parabola_directrix_l2898_289833


namespace other_stamp_price_l2898_289809

-- Define the total number of stamps
def total_stamps : ℕ := 75

-- Define the total amount received in cents
def total_amount : ℕ := 480

-- Define the price of the known stamp type
def known_stamp_price : ℕ := 8

-- Define the number of stamps sold of one kind
def stamps_of_one_kind : ℕ := 40

-- Define the function to calculate the price of the unknown stamp type
def unknown_stamp_price (x : ℕ) : Prop :=
  (stamps_of_one_kind * known_stamp_price + (total_stamps - stamps_of_one_kind) * x = total_amount) ∧
  (x > 0) ∧ (x < known_stamp_price)

-- Theorem stating that the price of the unknown stamp type is 5 cents
theorem other_stamp_price : unknown_stamp_price 5 := by
  sorry

end other_stamp_price_l2898_289809


namespace sequence_a_increasing_l2898_289845

def a (n : ℕ) : ℚ := (n - 1 : ℚ) / (n + 1 : ℚ)

theorem sequence_a_increasing : ∀ n ≥ 2, a n < a (n + 1) := by
  sorry

end sequence_a_increasing_l2898_289845


namespace additional_cans_needed_l2898_289847

def martha_cans : ℕ := 90
def diego_cans : ℕ := martha_cans / 2 + 10
def leah_cans : ℕ := martha_cans / 3 - 5

def martha_aluminum : ℕ := (martha_cans * 70) / 100
def diego_aluminum : ℕ := (diego_cans * 50) / 100
def leah_aluminum : ℕ := (leah_cans * 80) / 100

def total_needed : ℕ := 200

theorem additional_cans_needed :
  total_needed - (martha_aluminum + diego_aluminum + leah_aluminum) = 90 :=
by sorry

end additional_cans_needed_l2898_289847


namespace simultaneous_inequality_condition_l2898_289871

theorem simultaneous_inequality_condition (a : ℝ) : 
  (∃ x₀ : ℝ, x₀^2 - a*x₀ + a + 3 < 0 ∧ a*x₀ - 2*a < 0) ↔ a > 7 := by
  sorry

end simultaneous_inequality_condition_l2898_289871


namespace system_solution_l2898_289868

theorem system_solution : ∃! (x y : ℝ), x + y = 5 ∧ x - y = 3 ∧ x = 4 ∧ y = 1 := by
  sorry

end system_solution_l2898_289868


namespace cells_after_one_week_l2898_289872

/-- The number of cells after n days, given that each cell divides into three new cells every day -/
def num_cells (n : ℕ) : ℕ := 3^n

/-- Theorem: After 7 days, there will be 2187 cells -/
theorem cells_after_one_week : num_cells 7 = 2187 := by
  sorry

end cells_after_one_week_l2898_289872


namespace three_fifths_of_negative_twelve_sevenths_l2898_289848

theorem three_fifths_of_negative_twelve_sevenths :
  (3 : ℚ) / 5 * (-12 : ℚ) / 7 = -36 / 35 := by sorry

end three_fifths_of_negative_twelve_sevenths_l2898_289848


namespace bridge_length_l2898_289816

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 265 :=
by sorry

end bridge_length_l2898_289816


namespace f_1000_is_even_l2898_289842

/-- A function that satisfies the given functional equation -/
def SatisfiesEquation (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (f^[f n] n) = n^2 / (f (f n))

/-- Theorem stating that f(1000) is even for any function satisfying the equation -/
theorem f_1000_is_even (f : ℕ → ℕ) (h : SatisfiesEquation f) : 
  ∃ k : ℕ, f 1000 = 2 * k :=
sorry

end f_1000_is_even_l2898_289842


namespace subset_implies_a_equals_one_l2898_289891

def A : Set ℝ := {x : ℝ | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

theorem subset_implies_a_equals_one (a : ℝ) (h1 : B a ⊆ A) (h2 : a > 0) : a = 1 := by
  sorry

end subset_implies_a_equals_one_l2898_289891


namespace parabola_properties_l2898_289804

/-- Parabola with equation y = ax(x-6) + 1 where a ≠ 0 -/
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x * (x - 6) + 1

theorem parabola_properties (a : ℝ) (h : a ≠ 0) :
  /- Point (0,1) lies on the parabola -/
  (parabola a 0 = 1) ∧
  /- If the distance from the vertex to the x-axis is 5, then a = 2/3 or a = -4/9 -/
  (∃ (x : ℝ), (∀ y : ℝ, parabola a y ≥ parabola a x) →
    |parabola a x| = 5 → (a = 2/3 ∨ a = -4/9)) ∧
  /- If the length of the segment formed by the intersection of the parabola with the x-axis
     is less than or equal to 4, then 1/9 < a ≤ 1/5 -/
  ((∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ parabola a x₁ = 0 ∧ parabola a x₂ = 0 ∧ x₂ - x₁ ≤ 4) →
    1/9 < a ∧ a ≤ 1/5) :=
by sorry

end parabola_properties_l2898_289804


namespace sanity_determination_question_exists_l2898_289843

/-- Represents the sanity state of a guest -/
inductive Sanity
| Sane
| Insane

/-- Represents possible answers to a question -/
inductive Answer
| Yes
| Ball

/-- A function representing how a guest answers a question based on their sanity -/
def guest_answer (s : Sanity) : Answer :=
  match s with
  | Sanity.Sane => Answer.Ball
  | Sanity.Insane => Answer.Yes

/-- The theorem stating that there exists a question that can determine a guest's sanity -/
theorem sanity_determination_question_exists :
  ∃ (question : Sanity → Answer),
    (∀ s : Sanity, question s = guest_answer s) ∧
    (∀ s₁ s₂ : Sanity, question s₁ = question s₂ → s₁ = s₂) :=
by sorry

end sanity_determination_question_exists_l2898_289843


namespace coloring_book_shelves_l2898_289824

theorem coloring_book_shelves (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 120 → books_sold = 39 → books_per_shelf = 9 →
  (initial_stock - books_sold) / books_per_shelf = 9 := by
sorry

end coloring_book_shelves_l2898_289824


namespace algae_coverage_day_18_and_19_l2898_289805

/-- Represents the coverage of algae on the pond on a given day -/
def algaeCoverage (day : ℕ) : ℚ :=
  (1 : ℚ) / 3^(20 - day)

/-- The problem statement -/
theorem algae_coverage_day_18_and_19 :
  algaeCoverage 18 < (1 : ℚ) / 4 ∧ (1 : ℚ) / 4 < algaeCoverage 19 := by
  sorry

#eval algaeCoverage 18  -- Expected: 1/9
#eval algaeCoverage 19  -- Expected: 1/3

end algae_coverage_day_18_and_19_l2898_289805


namespace remaining_area_after_cutouts_cutout_area_is_nine_dark_gray_rectangle_area_light_gray_rectangle_area_l2898_289851

/-- The area of a square with side length 6, minus specific triangular cutouts, equals 27 -/
theorem remaining_area_after_cutouts (square_side : ℝ) (cutout_area : ℝ) : 
  square_side = 6 → 
  cutout_area = 9 → 
  square_side^2 - cutout_area = 27 := by
  sorry

/-- The area of triangular cutouts in a 6x6 square equals 9 -/
theorem cutout_area_is_nine (dark_gray_rect_area light_gray_rect_area : ℝ) :
  dark_gray_rect_area = 3 →
  light_gray_rect_area = 6 →
  dark_gray_rect_area + light_gray_rect_area = 9 := by
  sorry

/-- The area of a rectangle formed by dark gray triangles is 3 -/
theorem dark_gray_rectangle_area (length width : ℝ) :
  length = 1 →
  width = 3 →
  length * width = 3 := by
  sorry

/-- The area of a rectangle formed by light gray triangles is 6 -/
theorem light_gray_rectangle_area (length width : ℝ) :
  length = 2 →
  width = 3 →
  length * width = 6 := by
  sorry

end remaining_area_after_cutouts_cutout_area_is_nine_dark_gray_rectangle_area_light_gray_rectangle_area_l2898_289851


namespace box_two_three_neg_one_l2898_289832

-- Define the box operation for integers a, b, and c
def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) - (b ^ c : ℚ) + (c ^ a : ℚ)

-- Theorem statement
theorem box_two_three_neg_one : box 2 3 (-1) = 26 / 3 := by
  sorry

end box_two_three_neg_one_l2898_289832


namespace linear_function_composition_l2898_289807

/-- A linear function is a function of the form f(x) = kx + b for some constants k and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ x, f x = k * x + b

/-- The main theorem: if f is a linear function satisfying f(f(x)) = 4x + 6,
    then f(x) = 2x + 2 or f(x) = -2x - 6. -/
theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 6) →
  (∀ x, f x = 2 * x + 2) ∨ (∀ x, f x = -2 * x - 6) := by
  sorry

end linear_function_composition_l2898_289807


namespace min_distance_exp_curve_to_line_l2898_289820

/-- The minimum distance from any point on the curve y = e^x to the line y = x - 1 is √2 -/
theorem min_distance_exp_curve_to_line : 
  ∀ (x₀ y₀ : ℝ), y₀ = Real.exp x₀ → 
  (∃ (d : ℝ), d = |y₀ - (x₀ - 1)| / Real.sqrt 2 ∧ 
   ∀ (x y : ℝ), y = Real.exp x → 
   d ≤ |y - (x - 1)| / Real.sqrt 2) → 
  ∃ (d : ℝ), d = Real.sqrt 2 ∧
  (∀ (x y : ℝ), y = Real.exp x → 
   d ≤ |y - (x - 1)| / Real.sqrt 2) :=
by sorry


end min_distance_exp_curve_to_line_l2898_289820


namespace find_g_of_x_l2898_289859

/-- Given that 4x^4 + 2x^2 - 7x + 3 + g(x) = 5x^3 - 8x^2 + 4x - 1,
    prove that g(x) = -4x^4 + 5x^3 - 10x^2 + 11x - 4 -/
theorem find_g_of_x (g : ℝ → ℝ) :
  (∀ x : ℝ, 4 * x^4 + 2 * x^2 - 7 * x + 3 + g x = 5 * x^3 - 8 * x^2 + 4 * x - 1) →
  (∀ x : ℝ, g x = -4 * x^4 + 5 * x^3 - 10 * x^2 + 11 * x - 4) :=
by sorry

end find_g_of_x_l2898_289859


namespace quadratic_root_relation_l2898_289886

theorem quadratic_root_relation (b c : ℚ) : 
  (∃ r s : ℚ, 5 * r^2 - 8 * r + 2 = 0 ∧ 5 * s^2 - 8 * s + 2 = 0 ∧
   (r - 3)^2 + b * (r - 3) + c = 0 ∧ (s - 3)^2 + b * (s - 3) + c = 0) →
  c = 23/5 := by
sorry

end quadratic_root_relation_l2898_289886


namespace value_of_a_l2898_289849

theorem value_of_a (M : Set ℝ) (a : ℝ) : 
  M = {0, 1, a + 1} → -1 ∈ M → a = -2 := by
  sorry

end value_of_a_l2898_289849


namespace midpoint_octahedron_volume_ratio_l2898_289895

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  -- We don't need to specify the vertices or edge length here

/-- An octahedron formed by the midpoints of a tetrahedron's edges -/
def midpoint_octahedron (t : RegularTetrahedron) : Set (Fin 4 → ℝ) :=
  sorry

/-- The volume of a regular tetrahedron -/
def volume_tetrahedron (t : RegularTetrahedron) : ℝ :=
  sorry

/-- The volume of the octahedron formed by midpoints -/
def volume_midpoint_octahedron (t : RegularTetrahedron) : ℝ :=
  sorry

/-- Theorem: The ratio of the volume of the midpoint octahedron to the volume of the regular tetrahedron is 3/16 -/
theorem midpoint_octahedron_volume_ratio (t : RegularTetrahedron) :
  volume_midpoint_octahedron t / volume_tetrahedron t = 3 / 16 :=
sorry

end midpoint_octahedron_volume_ratio_l2898_289895


namespace sum_of_squares_l2898_289813

theorem sum_of_squares (a b c d e f : ℤ) :
  (∀ x : ℝ, 1728 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 23456 := by
  sorry

end sum_of_squares_l2898_289813


namespace empty_bottle_weight_l2898_289874

/-- Given a full bottle of sesame oil weighing 3.4 kg and the same bottle weighing 2.98 kg
    after using 1/5 of the oil, the weight of the empty bottle is 1.3 kg. -/
theorem empty_bottle_weight (full_weight : ℝ) (partial_weight : ℝ) (empty_weight : ℝ) : 
  full_weight = 3.4 →
  partial_weight = 2.98 →
  full_weight = empty_weight + (5/4) * (partial_weight - empty_weight) →
  empty_weight = 1.3 := by
  sorry

#check empty_bottle_weight

end empty_bottle_weight_l2898_289874


namespace total_seashells_l2898_289870

def seashells_day1 : ℕ := 5
def seashells_day2 : ℕ := 7
def seashells_day3 (x y : ℕ) : ℕ := 2 * (x + y)

theorem total_seashells : 
  seashells_day1 + seashells_day2 + seashells_day3 seashells_day1 seashells_day2 = 36 := by
  sorry

end total_seashells_l2898_289870


namespace inequality_proof_l2898_289808

theorem inequality_proof (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 4) : 
  |a*c + b*d| ≤ 2 := by
  sorry

end inequality_proof_l2898_289808


namespace perpendicular_vectors_l2898_289877

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 4)

theorem perpendicular_vectors (x : ℝ) :
  (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2 = 0) → x = -8 := by
  sorry

end perpendicular_vectors_l2898_289877


namespace roots_sum_of_squares_l2898_289803

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 8*a + 8 = 0) → (b^2 - 8*b + 8 = 0) → (a^2 + b^2 = 48) := by
  sorry

end roots_sum_of_squares_l2898_289803


namespace sqrt_D_irrational_l2898_289853

theorem sqrt_D_irrational (x : ℝ) : 
  ∀ (y : ℝ), y ^ 2 ≠ 3 * (2 * x) ^ 2 + 3 * (2 * x + 1) ^ 2 + (4 * x + 1) ^ 2 :=
sorry

end sqrt_D_irrational_l2898_289853


namespace polar_to_cartesian_coordinates_l2898_289817

theorem polar_to_cartesian_coordinates :
  let r : ℝ := 2
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 1 ∧ y = Real.sqrt 3) :=
by sorry

end polar_to_cartesian_coordinates_l2898_289817


namespace wall_painting_theorem_l2898_289860

theorem wall_painting_theorem (heidi_time tim_time total_time : ℚ) 
  (h1 : heidi_time = 45)
  (h2 : tim_time = 30)
  (h3 : total_time = 9) :
  let heidi_rate : ℚ := 1 / heidi_time
  let tim_rate : ℚ := 1 / tim_time
  let combined_rate : ℚ := heidi_rate + tim_rate
  (combined_rate * total_time) = 1/2 := by sorry

end wall_painting_theorem_l2898_289860


namespace age_problem_l2898_289858

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 := by sorry

end age_problem_l2898_289858


namespace hyperbola_eccentricity_l2898_289830

/-- Given a hyperbola with semi-major axis a and semi-minor axis b, 
    a point P on its right branch, F as its right focus, 
    and M on the line x = -a²/c, where c is the focal distance,
    prove that if OP = OF + OM and OP ⋅ FM = 0, then the eccentricity is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (P F M O : ℝ × ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (P.1^2 / a^2) - (P.2^2 / b^2) = 1 →  -- P is on the right branch of the hyperbola
  P.1 > 0 →  -- P is on the right branch
  F = (c, 0) →  -- F is the right focus
  M.1 = -a^2 / c →  -- M is on the line x = -a²/c
  P.1 - O.1 = F.1 - O.1 + M.1 - O.1 →  -- OP = OF + OM (x-component)
  P.2 - O.2 = F.2 - O.2 + M.2 - O.2 →  -- OP = OF + OM (y-component)
  (P.1 - F.1) * (M.1 - F.1) + (P.2 - F.2) * (M.2 - F.2) = 0 →  -- OP ⋅ FM = 0
  c / a = 2 :=  -- eccentricity is 2
by sorry

end hyperbola_eccentricity_l2898_289830


namespace thirty_two_distributions_l2898_289881

/-- Represents a knockout tournament with 6 players. -/
structure Tournament :=
  (players : Fin 6 → ℕ)

/-- The number of possible outcomes for each match. -/
def match_outcomes : ℕ := 2

/-- The number of rounds in the tournament. -/
def num_rounds : ℕ := 5

/-- Calculates the total number of possible prize distribution orders. -/
def prize_distributions (t : Tournament) : ℕ :=
  match_outcomes ^ num_rounds

/-- Theorem stating that there are 32 possible prize distribution orders. -/
theorem thirty_two_distributions (t : Tournament) :
  prize_distributions t = 32 := by
  sorry

end thirty_two_distributions_l2898_289881


namespace alex_age_l2898_289890

theorem alex_age (inez_age : ℕ) (zack_age : ℕ) (jose_age : ℕ) (alex_age : ℕ)
  (h1 : inez_age = 18)
  (h2 : zack_age = inez_age + 5)
  (h3 : jose_age = zack_age - 6)
  (h4 : alex_age = jose_age - 2) :
  alex_age = 15 := by
  sorry

end alex_age_l2898_289890


namespace greatest_x_with_lcm_l2898_289839

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (y : ℕ), y = Nat.lcm x (Nat.lcm 8 12) ∧ y = 120) → 
  x ≤ 120 ∧ ∃ (z : ℕ), z = 120 ∧ Nat.lcm z (Nat.lcm 8 12) = 120 :=
by sorry

end greatest_x_with_lcm_l2898_289839


namespace trees_planted_correct_l2898_289893

/-- The number of maple trees planted in a park --/
def trees_planted (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that the number of trees planted is the difference between final and initial counts --/
theorem trees_planted_correct (initial final : ℕ) (h : final ≥ initial) :
  trees_planted initial final = final - initial :=
by sorry

end trees_planted_correct_l2898_289893


namespace complex_fraction_simplification_l2898_289896

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem complex_fraction_simplification :
  (1 + i) / (1 - i) = i :=
by sorry

end complex_fraction_simplification_l2898_289896


namespace unique_a_value_l2898_289884

theorem unique_a_value (a : ℝ) : 
  (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    x₁ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ 
    x₂ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ 
    x₃ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧
    Real.sin x₁ + Real.sqrt 3 * Real.cos x₁ = a ∧
    Real.sin x₂ + Real.sqrt 3 * Real.cos x₂ = a ∧
    Real.sin x₃ + Real.sqrt 3 * Real.cos x₃ = a) ↔ 
  a = Real.sqrt 3 :=
sorry

end unique_a_value_l2898_289884


namespace stratified_sampling_third_year_count_l2898_289827

theorem stratified_sampling_third_year_count 
  (total_students : ℕ) 
  (third_year_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 900) 
  (h2 : third_year_students = 400) 
  (h3 : sample_size = 45) :
  (third_year_students * sample_size) / total_students = 20 := by
  sorry

end stratified_sampling_third_year_count_l2898_289827


namespace arithmetic_seq_properties_l2898_289834

/-- An arithmetic sequence with a_1 = 1 and a_3 - a_2 = 1 -/
def arithmetic_seq (n : ℕ) : ℕ := n

/-- Sum of the first n terms of the arithmetic sequence -/
def arithmetic_seq_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem arithmetic_seq_properties :
  let a := arithmetic_seq
  let S := arithmetic_seq_sum
  (∀ n : ℕ, n ≥ 1 → a n = n) ∧
  (∀ n : ℕ, n ≥ 1 → S n = n * (n + 1) / 2) :=
by
  sorry

#check arithmetic_seq_properties

end arithmetic_seq_properties_l2898_289834


namespace same_color_probability_l2898_289802

def total_balls : ℕ := 5 + 8 + 4 + 3

def green_balls : ℕ := 5
def white_balls : ℕ := 8
def blue_balls : ℕ := 4
def red_balls : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem same_color_probability :
  (choose green_balls 4 + choose white_balls 4 + choose blue_balls 4) / choose total_balls 4 = 76 / 4845 := by
  sorry

end same_color_probability_l2898_289802


namespace set_A_representation_l2898_289883

def A : Set (ℤ × ℤ) := {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_representation : A = {(-1, 0), (0, -1), (1, 0)} := by
  sorry

end set_A_representation_l2898_289883


namespace sin_symmetry_l2898_289800

theorem sin_symmetry (t : ℝ) : 
  Real.sin ((π / 6 + t) + π / 3) = Real.sin ((π / 6 - t) + π / 3) := by
  sorry

end sin_symmetry_l2898_289800


namespace propositions_true_l2898_289825

-- Definition of reciprocals
def are_reciprocals (x y : ℝ) : Prop := x * y = 1

-- Definition of real roots for a quadratic equation
def has_real_roots (b : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*b*x + b^2 + b = 0

theorem propositions_true : 
  (∀ x y : ℝ, are_reciprocals x y → x * y = 1) ∧
  (∀ b : ℝ, ¬(has_real_roots b) → b > -1) ∧
  (∃ α β : ℝ, Real.sin (α + β) = Real.sin α + Real.sin β) :=
by sorry

end propositions_true_l2898_289825


namespace commission_breakpoint_l2898_289882

/-- Proves that for a sale of $800, if the commission is 20% of the first $X plus 25% of the remainder, 
    and the total commission is 21.875% of the sale, then X = $500. -/
theorem commission_breakpoint (X : ℝ) : 
  let total_sale := 800
  let commission_rate_1 := 0.20
  let commission_rate_2 := 0.25
  let total_commission_rate := 0.21875
  commission_rate_1 * X + commission_rate_2 * (total_sale - X) = total_commission_rate * total_sale →
  X = 500 :=
by sorry

end commission_breakpoint_l2898_289882


namespace largest_product_of_three_l2898_289810

def S : Set ℤ := {-3, -2, 4, 5}

theorem largest_product_of_three (a b c : ℤ) :
  a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
  ∀ x y z : ℤ, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  a * b * c ≤ 30 ∧ (∃ p q r : ℤ, p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p * q * r = 30) :=
by
  sorry

end largest_product_of_three_l2898_289810


namespace distance_equals_radius_l2898_289850

/-- A circle resting on the x-axis and tangent to the line x=3 -/
structure TangentCircle where
  /-- The x-coordinate of the circle's center -/
  h : ℝ
  /-- The radius of the circle -/
  r : ℝ
  /-- The circle rests on the x-axis and is tangent to x=3 -/
  tangent_condition : r = |3 - h|

/-- The distance from the center to the point of tangency equals the radius -/
theorem distance_equals_radius (c : TangentCircle) :
  |3 - c.h| = c.r := by sorry

end distance_equals_radius_l2898_289850


namespace group_size_proof_l2898_289898

def group_size (adult_meal_cost : ℕ) (total_cost : ℕ) (num_kids : ℕ) : ℕ :=
  (total_cost / adult_meal_cost) + num_kids

theorem group_size_proof :
  group_size 2 14 2 = 9 := by
  sorry

end group_size_proof_l2898_289898


namespace equation_solution_l2898_289856

theorem equation_solution : 
  let y : ℚ := 6/7
  ∀ (y : ℚ), y ≠ -2 ∧ y ≠ -1 →
  (7*y) / ((y+2)*(y+1)) - 4 / ((y+2)*(y+1)) = 2 / ((y+2)*(y+1)) →
  y = 6/7 := by
sorry

end equation_solution_l2898_289856


namespace only_one_milk_chocolate_affordable_l2898_289878

-- Define the prices of chocolates
def dark_chocolate_price : ℚ := 5
def milk_chocolate_price : ℚ := 9/2
def white_chocolate_price : ℚ := 6

-- Define the sales tax rate
def sales_tax_rate : ℚ := 7/100

-- Define Leonardo's budget
def leonardo_budget : ℚ := 459/100

-- Function to calculate price with tax
def price_with_tax (price : ℚ) : ℚ := price * (1 + sales_tax_rate)

-- Theorem statement
theorem only_one_milk_chocolate_affordable :
  (price_with_tax dark_chocolate_price > leonardo_budget) ∧
  (price_with_tax white_chocolate_price > leonardo_budget) ∧
  (price_with_tax milk_chocolate_price ≤ leonardo_budget) ∧
  (2 * price_with_tax milk_chocolate_price > leonardo_budget) :=
by sorry

end only_one_milk_chocolate_affordable_l2898_289878


namespace minor_premise_is_proposition1_l2898_289841

-- Define the propositions
def proposition1 : Prop := 0 < (1/2 : ℝ) ∧ (1/2 : ℝ) < 1
def proposition2 (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^y < a^x
def proposition3 : Prop := ∀ a : ℝ, 0 < a ∧ a < 1 → (∀ x y : ℝ, x < y → a^y < a^x)

-- Define the syllogism structure
structure Syllogism :=
  (major_premise : Prop)
  (minor_premise : Prop)
  (conclusion : Prop)

-- Theorem statement
theorem minor_premise_is_proposition1 :
  ∃ s : Syllogism, s.major_premise = proposition3 ∧
                   s.minor_premise = proposition1 ∧
                   s.conclusion = proposition2 (1/2) :=
sorry

end minor_premise_is_proposition1_l2898_289841


namespace nested_fraction_evaluation_l2898_289801

theorem nested_fraction_evaluation : 
  2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end nested_fraction_evaluation_l2898_289801


namespace median_and_perpendicular_bisector_equations_l2898_289838

/-- Given three points in a plane, prove the equations of the median and perpendicular bisector of a side -/
theorem median_and_perpendicular_bisector_equations 
  (A B C : ℝ × ℝ) 
  (hA : A = (1, 2)) 
  (hB : B = (-1, 4)) 
  (hC : C = (5, 2)) : 
  (∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) ↔ (y - 3 = -1 * (x - 0))) ∧ 
  (∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) ↔ (y = x + 3)) := by
  sorry


end median_and_perpendicular_bisector_equations_l2898_289838


namespace rectangular_garden_diagonal_ratio_l2898_289866

theorem rectangular_garden_diagonal_ratio (b : ℝ) (h : b > 0) :
  let a := 3 * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let perimeter := 2 * (a + b)
  diagonal / perimeter = Real.sqrt 10 / 8 ∧ perimeter - diagonal = b :=
by
  sorry

end rectangular_garden_diagonal_ratio_l2898_289866


namespace lune_area_l2898_289855

/-- The area of a lune formed by two semicircles -/
theorem lune_area (r₁ r₂ : ℝ) (h : r₁ = 2 * r₂) : 
  let lune_area := π * r₂^2 / 2 + r₁ * r₂ - π * r₁^2 / 4
  lune_area = 1 - π / 2 := by
  sorry

end lune_area_l2898_289855


namespace point_not_in_transformed_plane_l2898_289873

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane3D) (k : ℝ) : Plane3D :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point satisfies the equation of a plane -/
def pointSatisfiesPlane (point : Point3D) (plane : Plane3D) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem to be proved -/
theorem point_not_in_transformed_plane :
  let A : Point3D := { x := -1, y := 1, z := -2 }
  let a : Plane3D := { a := 4, b := -1, c := 3, d := -6 }
  let k : ℝ := -5/3
  let transformedPlane := transformPlane a k
  ¬ pointSatisfiesPlane A transformedPlane :=
by
  sorry


end point_not_in_transformed_plane_l2898_289873


namespace unique_solution_implies_a_equals_three_l2898_289892

theorem unique_solution_implies_a_equals_three (a : ℝ) :
  (∃! x : ℝ, x^2 + a * |x| + a^2 - 9 = 0) → a = 3 :=
by sorry

end unique_solution_implies_a_equals_three_l2898_289892


namespace not_zero_necessary_not_sufficient_for_positive_l2898_289888

theorem not_zero_necessary_not_sufficient_for_positive (x : ℝ) :
  (∃ x, x ≠ 0 ∧ x ≤ 0) ∧ (∀ x, x > 0 → x ≠ 0) :=
by sorry

end not_zero_necessary_not_sufficient_for_positive_l2898_289888


namespace difference_of_squares_l2898_289815

theorem difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by sorry

end difference_of_squares_l2898_289815


namespace plant_purchase_cost_l2898_289876

/-- Calculates the actual amount spent on plants given the original cost and discount. -/
def actualCost (originalCost discount : ℚ) : ℚ :=
  originalCost - discount

/-- Theorem stating that given the specific original cost and discount, the actual amount spent is $68.00. -/
theorem plant_purchase_cost :
  let originalCost : ℚ := 467
  let discount : ℚ := 399
  actualCost originalCost discount = 68 := by
sorry

end plant_purchase_cost_l2898_289876


namespace range_of_m_l2898_289840

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, m^2 * x^2 + 2*m*x - 4 < 2*x^2 + 4*x) → 
  -2 < m ∧ m ≤ 2 :=
by sorry

end range_of_m_l2898_289840


namespace find_x1_l2898_289857

theorem find_x1 (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 3/4) :
  x1 = 3 * Real.sqrt 3 / 8 := by
  sorry

end find_x1_l2898_289857


namespace rahul_savings_l2898_289814

/-- Rahul's savings problem -/
theorem rahul_savings (nsc ppf : ℕ) : 
  (3 * nsc = 2 * ppf) →  -- One-third of NSC equals one-half of PPF
  (nsc + ppf = 180000) → -- Total savings
  (ppf = 72000) :=       -- PPF savings to prove
by sorry

end rahul_savings_l2898_289814


namespace roberts_chocolates_l2898_289867

theorem roberts_chocolates (nickel_chocolates : ℕ) (difference : ℕ) : 
  nickel_chocolates = 3 → difference = 9 → nickel_chocolates + difference = 12 := by
  sorry

end roberts_chocolates_l2898_289867


namespace total_nails_and_claws_is_524_l2898_289887

/-- The total number of nails and claws Cassie needs to cut -/
def total_nails_and_claws : ℕ :=
  -- Dogs
  4 * 4 * 4 +
  -- Parrots
  (7 * 2 * 3 + 1 * 2 * 4 + 1 * 2 * 2) +
  -- Cats
  (1 * 2 * 5 + 1 * 2 * 4 + 1) +
  -- Rabbits
  (5 * 4 * 9 + 3 * 9 + 2) +
  -- Lizards
  (4 * 4 * 5 + 1 * 4 * 4) +
  -- Tortoises
  (2 * 4 * 4 + 3 * 4 + 5 + 3 * 4 + 3)

/-- Theorem stating that the total number of nails and claws is 524 -/
theorem total_nails_and_claws_is_524 : total_nails_and_claws = 524 := by
  sorry

end total_nails_and_claws_is_524_l2898_289887


namespace line_circle_relationship_l2898_289828

/-- The line equation -/
def line_equation (k x y : ℝ) : Prop :=
  (3*k + 2) * x - k * y - 2 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 2 = 0

/-- The theorem stating the positional relationship between the line and the circle -/
theorem line_circle_relationship :
  ∀ k : ℝ, ∃ x y : ℝ, 
    (line_equation k x y ∧ circle_equation x y) ∨ 
    (∃ x₀ y₀ : ℝ, line_equation k x₀ y₀ ∧ circle_equation x₀ y₀ ∧ 
      ∀ x y : ℝ, line_equation k x y ∧ circle_equation x y → (x, y) = (x₀, y₀)) :=
by sorry

end line_circle_relationship_l2898_289828


namespace total_pay_calculation_l2898_289862

/-- Calculates the total pay for a worker given regular and overtime hours --/
def total_pay (regular_rate : ℝ) (regular_hours : ℝ) (overtime_hours : ℝ) : ℝ :=
  let overtime_rate := 2 * regular_rate
  regular_rate * regular_hours + overtime_rate * overtime_hours

/-- Theorem stating that under the given conditions, the total pay is $192 --/
theorem total_pay_calculation :
  let regular_rate := 3
  let regular_hours := 40
  let overtime_hours := 12
  total_pay regular_rate regular_hours overtime_hours = 192 := by
  sorry

end total_pay_calculation_l2898_289862


namespace same_terminal_side_angles_l2898_289863

theorem same_terminal_side_angles (π : ℝ) : 
  {β : ℝ | ∃ k : ℤ, β = π / 3 + 2 * k * π ∧ -2 * π ≤ β ∧ β < 4 * π} = 
  {-5 * π / 3, π / 3, 7 * π / 3} := by sorry

end same_terminal_side_angles_l2898_289863


namespace sequence_general_term_1_l2898_289846

theorem sequence_general_term_1 (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h : ∀ n, S n = 2 * n^2 - 3 * n + 2) :
  (a 1 = 1 ∧ ∀ n ≥ 2, a n = 4 * n - 5) ↔ 
  (∀ n, n ≥ 1 → a n = S n - S (n-1)) :=
sorry


end sequence_general_term_1_l2898_289846


namespace arithmetic_sequence_formula_l2898_289897

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_formula :
  ∀ n : ℕ, n ≥ 1 → arithmetic_sequence (-3) 4 n = 4*n - 7 :=
by sorry

end arithmetic_sequence_formula_l2898_289897


namespace student_sister_weight_l2898_289852

/-- The combined weight of a student and his sister, given specific conditions --/
theorem student_sister_weight (student_weight sister_weight : ℝ) : 
  student_weight = 79 →
  student_weight - 5 = 2 * sister_weight →
  student_weight + sister_weight = 116 := by
sorry

end student_sister_weight_l2898_289852


namespace power_multiplication_calculate_3000_power_l2898_289831

theorem power_multiplication (a : ℕ) (m n : ℕ) :
  a * (a ^ n) = a ^ (n + 1) :=
by sorry

theorem calculate_3000_power :
  3000 * (3000 ^ 1999) = 3000 ^ 2000 :=
by sorry

end power_multiplication_calculate_3000_power_l2898_289831


namespace bridge_crossing_time_l2898_289829

/-- Proves that a man walking at 10 km/hr takes 15 minutes to cross a 2500-meter bridge -/
theorem bridge_crossing_time :
  let walking_speed : ℝ := 10  -- km/hr
  let bridge_length : ℝ := 2.5  -- km (2500 meters)
  let crossing_time : ℝ := bridge_length / walking_speed * 60  -- minutes
  crossing_time = 15 := by sorry

end bridge_crossing_time_l2898_289829


namespace right_triangle_sine_cosine_l2898_289865

theorem right_triangle_sine_cosine (P Q R : Real) (h1 : 3 * Real.sin P = 4 * Real.cos P) :
  Real.sin P = 4/5 := by
  sorry

end right_triangle_sine_cosine_l2898_289865


namespace sum_of_coefficients_l2898_289821

/-- A quadratic function with vertex (-3, 4) passing through (1, 2) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_coefficients (a b c : ℝ) :
  (∀ x, f a b c x = a * x^2 + b * x + c) →
  f a b c (-3) = 4 →
  (∀ h : ℝ, f a b c (-3 + h) = f a b c (-3 - h)) →
  f a b c 1 = 2 →
  a + b + c = 3 := by
  sorry

end sum_of_coefficients_l2898_289821


namespace number_of_children_l2898_289899

/-- Given that each child has 8 crayons and there are 56 crayons in total,
    prove that the number of children is 7. -/
theorem number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) 
  (h1 : crayons_per_child = 8) (h2 : total_crayons = 56) :
  total_crayons / crayons_per_child = 7 := by
  sorry

end number_of_children_l2898_289899


namespace product_of_17_terms_geometric_sequence_l2898_289879

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the product of the first n terms of a sequence
def product_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc i => acc * a (i + 1)) 1

-- Theorem statement
theorem product_of_17_terms_geometric_sequence 
  (a : ℕ → ℝ) (h_geo : geometric_sequence a) (h_a9 : a 9 = -2) :
  product_of_first_n_terms a 17 = -2^17 := by
  sorry

end product_of_17_terms_geometric_sequence_l2898_289879


namespace balloon_sum_equals_total_l2898_289837

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := 6

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- Theorem stating that the sum of individual balloon counts equals the total -/
theorem balloon_sum_equals_total :
  fred_balloons + sam_balloons + mary_balloons = total_balloons :=
by sorry

end balloon_sum_equals_total_l2898_289837


namespace g_of_5_l2898_289819

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 35*x^2 - 40*x + 24

theorem g_of_5 : g 5 = 74 := by
  sorry

end g_of_5_l2898_289819


namespace changhee_semester_average_l2898_289861

/-- Calculates the average score for a semester given midterm and final exam scores and subject counts. -/
def semesterAverage (midtermAvg : ℚ) (midtermSubjects : ℕ) (finalAvg : ℚ) (finalSubjects : ℕ) : ℚ :=
  (midtermAvg * midtermSubjects + finalAvg * finalSubjects) / (midtermSubjects + finalSubjects)

/-- Proves that Changhee's semester average is 83.5 given the exam scores and subject counts. -/
theorem changhee_semester_average :
  semesterAverage 83.1 10 84 8 = 83.5 := by
  sorry

#eval semesterAverage 83.1 10 84 8

end changhee_semester_average_l2898_289861


namespace f_at_seven_l2898_289822

def f (x : ℝ) : ℝ := 7*x^5 + 12*x^4 - 5*x^3 - 6*x^2 + 3*x - 5

theorem f_at_seven : f 7 = 144468 := by
  sorry

end f_at_seven_l2898_289822


namespace shipping_cost_proof_l2898_289844

/-- Calculates the total shipping cost for fish -/
def total_shipping_cost (total_weight : ℕ) (crate_weight : ℕ) (cost_per_crate : ℚ) (surcharge_per_crate : ℚ) (flat_fee : ℚ) : ℚ :=
  let num_crates : ℕ := total_weight / crate_weight
  let crate_total_cost : ℚ := (cost_per_crate + surcharge_per_crate) * num_crates
  crate_total_cost + flat_fee

/-- Proves that the total shipping cost for the given conditions is $46.00 -/
theorem shipping_cost_proof :
  total_shipping_cost 540 30 (3/2) (1/2) 10 = 46 := by
  sorry

end shipping_cost_proof_l2898_289844


namespace envelope_touches_all_C_a_l2898_289836

/-- The curve C_a is defined by the equation (y - a^2)^2 = x^2(a^2 - x^2) for a > 0 -/
def C_a (a : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ (y - a^2)^2 = x^2 * (a^2 - x^2)

/-- The envelope curve -/
def envelope_curve (x y : ℝ) : Prop :=
  y = (3 * x^2) / 4

/-- Theorem stating that the envelope curve touches all C_a curves -/
theorem envelope_touches_all_C_a :
  ∀ (a x y : ℝ), C_a a x y → ∃ (x₀ y₀ : ℝ), 
    envelope_curve x₀ y₀ ∧ 
    C_a a x₀ y₀ ∧
    (∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
      ∀ (x' y' : ℝ), 
        ((x' - x₀)^2 + (y' - y₀)^2 < δ^2) →
        (envelope_curve x' y' → ¬C_a a x' y') ∧
        (C_a a x' y' → ¬envelope_curve x' y')) :=
by sorry

end envelope_touches_all_C_a_l2898_289836


namespace chandler_skateboard_savings_l2898_289835

/-- Calculates the minimum number of full weeks required to save for a skateboard -/
def min_weeks_to_save (skateboard_cost : ℕ) (gift_money : ℕ) (weekly_earnings : ℕ) : ℕ :=
  ((skateboard_cost - gift_money + weekly_earnings - 1) / weekly_earnings : ℕ)

theorem chandler_skateboard_savings :
  min_weeks_to_save 550 130 18 = 24 := by
  sorry

end chandler_skateboard_savings_l2898_289835


namespace prob_no_eight_correct_l2898_289826

/-- The probability of selecting a number from 1 to 10000 that doesn't contain the digit 8 -/
def prob_no_eight : ℚ :=
  (9^4 : ℚ) / 10000

/-- Theorem stating that the probability of selecting a number from 1 to 10000
    that doesn't contain the digit 8 is equal to (9^4) / 10000 -/
theorem prob_no_eight_correct :
  prob_no_eight = (9^4 : ℚ) / 10000 := by sorry

end prob_no_eight_correct_l2898_289826


namespace sufficient_not_necessary_l2898_289885

theorem sufficient_not_necessary :
  (∀ x : ℝ, x < -1 → (x < -1 ∨ x > 1)) ∧
  ¬(∀ x : ℝ, (x < -1 ∨ x > 1) → x < -1) :=
by sorry

end sufficient_not_necessary_l2898_289885


namespace cubic_equation_solution_l2898_289806

theorem cubic_equation_solution 
  (a b c d : ℝ) 
  (h1 : a * d = b * c) 
  (h2 : a * d ≠ 0) 
  (h3 : b * d < 0) :
  let x1 := -b / a
  let x2 := Real.sqrt (-d / b)
  let x3 := -Real.sqrt (-d / b)
  (a * x1^3 + b * x1^2 + c * x1 + d = 0) ∧
  (a * x2^3 + b * x2^2 + c * x2 + d = 0) ∧
  (a * x3^3 + b * x3^2 + c * x3 + d = 0) :=
by sorry

end cubic_equation_solution_l2898_289806


namespace triangular_pizza_area_l2898_289875

theorem triangular_pizza_area :
  ∀ (base height hypotenuse : ℝ),
  base = 9 →
  hypotenuse = 15 →
  base ^ 2 + height ^ 2 = hypotenuse ^ 2 →
  (base * height) / 2 = 54 :=
by
  sorry

end triangular_pizza_area_l2898_289875
