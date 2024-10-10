import Mathlib

namespace final_student_count_l2915_291567

/-- Calculates the number of students on a bus after three stops -/
def studentsOnBus (initial : ℝ) (stop1On stop1Off stop2On stop2Off stop3On stop3Off : ℝ) : ℝ :=
  initial + (stop1On - stop1Off) + (stop2On - stop2Off) + (stop3On - stop3Off)

/-- Theorem stating the final number of students on the bus -/
theorem final_student_count :
  studentsOnBus 21 7.5 2 1.2 5.3 11 4.8 = 28.6 := by
  sorry

end final_student_count_l2915_291567


namespace possible_zero_point_l2915_291588

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem possible_zero_point (hf : Continuous f) 
  (h2007 : f 2007 < 0) (h2008 : f 2008 < 0) (h2009 : f 2009 > 0) :
  ∃ x ∈ Set.Ioo 2007 2008, f x = 0 ∨ ∃ y ∈ Set.Ioo 2008 2009, f y = 0 :=
by sorry


end possible_zero_point_l2915_291588


namespace ali_sold_ten_books_tuesday_l2915_291589

/-- The number of books Ali sold on Tuesday -/
def books_sold_tuesday (initial_stock : ℕ) (sold_monday : ℕ) (sold_wednesday : ℕ)
  (sold_thursday : ℕ) (sold_friday : ℕ) (not_sold : ℕ) : ℕ :=
  initial_stock - not_sold - (sold_monday + sold_wednesday + sold_thursday + sold_friday)

/-- Theorem stating that Ali sold 10 books on Tuesday -/
theorem ali_sold_ten_books_tuesday :
  books_sold_tuesday 800 60 20 44 66 600 = 10 := by
  sorry

end ali_sold_ten_books_tuesday_l2915_291589


namespace decagon_diagonal_intersection_probability_l2915_291530

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
def RegularDecagon : Type := Unit

/-- The number of diagonals in a regular decagon, excluding the sides -/
def num_diagonals : ℕ := 35

/-- The number of ways to choose 2 diagonals from the total number of diagonals -/
def num_diagonal_pairs : ℕ := 595

/-- The number of sets of intersecting diagonals in a regular decagon -/
def num_intersecting_diagonals : ℕ := 210

/-- The probability that two randomly chosen diagonals in a regular decagon intersect inside the decagon -/
theorem decagon_diagonal_intersection_probability (d : RegularDecagon) : 
  (num_intersecting_diagonals : ℚ) / num_diagonal_pairs = 42 / 119 := by
  sorry

end decagon_diagonal_intersection_probability_l2915_291530


namespace positive_integer_solutions_of_equation_l2915_291507

theorem positive_integer_solutions_of_equation : 
  ∀ x y : ℕ+, 
    (x : ℚ) - (y : ℚ) = (x : ℚ) / (y : ℚ) + (x : ℚ)^2 / (y : ℚ)^2 + (x : ℚ)^3 / (y : ℚ)^3 
    ↔ (x = 28 ∧ y = 14) ∨ (x = 112 ∧ y = 28) :=
by sorry

end positive_integer_solutions_of_equation_l2915_291507


namespace system_demonstrates_transformational_thinking_l2915_291590

/-- A system of two linear equations in two variables -/
structure LinearSystem :=
  (eq1 : ℝ → ℝ → ℝ)
  (eq2 : ℝ → ℝ → ℝ)

/-- The process of substituting one equation into another -/
def substitute (sys : LinearSystem) : ℝ → ℝ :=
  λ y => sys.eq1 (sys.eq2 y y) y

/-- Transformational thinking in the context of solving linear systems -/
def transformational_thinking (sys : LinearSystem) : Prop :=
  ∃ (simplified_eq : ℝ → ℝ), substitute sys = simplified_eq

/-- The given system of linear equations -/
def given_system : LinearSystem :=
  { eq1 := λ x y => 2*x + y
  , eq2 := λ x y => x - 2*y }

/-- Theorem stating that the given system demonstrates transformational thinking -/
theorem system_demonstrates_transformational_thinking :
  transformational_thinking given_system :=
sorry


end system_demonstrates_transformational_thinking_l2915_291590


namespace suresh_job_completion_time_l2915_291582

/-- The time it takes Suresh to complete the job alone -/
def suresh_time : ℝ := 15

/-- The time it takes Ashutosh to complete the job alone -/
def ashutosh_time : ℝ := 20

/-- The time Suresh works on the job -/
def suresh_work_time : ℝ := 9

/-- The time Ashutosh works to complete the remaining job -/
def ashutosh_completion_time : ℝ := 8

theorem suresh_job_completion_time : 
  (suresh_work_time / suresh_time) + (ashutosh_completion_time / ashutosh_time) = 1 ∧ 
  suresh_time = 15 := by
  sorry


end suresh_job_completion_time_l2915_291582


namespace geometric_sequence_common_ratio_l2915_291551

/-- An increasing geometric sequence with specific conditions has a common ratio of 2 -/
theorem geometric_sequence_common_ratio : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) > a n) →  -- increasing sequence
  (∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) →  -- geometric sequence
  (a 1 + a 5 = 17) →  -- first condition
  (a 2 * a 4 = 16) →  -- second condition
  ∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = q * a n :=
by sorry

end geometric_sequence_common_ratio_l2915_291551


namespace triangle_median_theorem_l2915_291516

/-- Triangle XYZ with given side lengths and median --/
structure Triangle where
  XY : ℝ
  XZ : ℝ
  XM : ℝ
  YZ : ℝ

/-- The theorem stating the relationship between sides and median in the given triangle --/
theorem triangle_median_theorem (t : Triangle) (h1 : t.XY = 6) (h2 : t.XZ = 9) (h3 : t.XM = 4) :
  t.YZ = Real.sqrt 170 := by
  sorry

#check triangle_median_theorem

end triangle_median_theorem_l2915_291516


namespace nested_radical_eighteen_l2915_291554

theorem nested_radical_eighteen (x : ℝ) : x = Real.sqrt (18 + x) → x = (1 + Real.sqrt 73) / 2 := by
  sorry

end nested_radical_eighteen_l2915_291554


namespace smallest_positive_difference_l2915_291597

/-- Vovochka's addition method for three-digit numbers -/
def vovochka_sum (a b c d e f : ℕ) : ℕ :=
  (a + d) * 1000 + (b + e) * 100 + (c + f)

/-- Correct addition method for three-digit numbers -/
def correct_sum (a b c d e f : ℕ) : ℕ :=
  (a + d) * 100 + (b + e) * 10 + (c + f)

/-- The difference between Vovochka's sum and the correct sum -/
def sum_difference (a b c d e f : ℕ) : ℤ :=
  (vovochka_sum a b c d e f : ℤ) - (correct_sum a b c d e f : ℤ)

theorem smallest_positive_difference :
  ∃ (a b c d e f : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
    sum_difference a b c d e f > 0 ∧
    ∀ (x y z u v w : ℕ),
      x < 10 → y < 10 → z < 10 → u < 10 → v < 10 → w < 10 →
      sum_difference x y z u v w > 0 →
      sum_difference a b c d e f ≤ sum_difference x y z u v w ∧
    sum_difference a b c d e f = 1800 :=
sorry

end smallest_positive_difference_l2915_291597


namespace scientific_notation_of_170000_l2915_291550

/-- Definition of scientific notation -/
def is_scientific_notation (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ a ∧ a < 10 ∧ ∃ (x : ℝ), x = a * (10 : ℝ) ^ n

/-- The problem statement -/
theorem scientific_notation_of_170000 :
  ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n ∧ 170000 = a * (10 : ℝ) ^ n :=
by sorry

end scientific_notation_of_170000_l2915_291550


namespace joshuas_skittles_l2915_291598

/-- Given that Joshua gave 40.0 Skittles to each of his 5.0 friends,
    prove that the total number of Skittles his friends have is 200.0. -/
theorem joshuas_skittles (skittles_per_friend : ℝ) (num_friends : ℝ) 
    (h1 : skittles_per_friend = 40.0)
    (h2 : num_friends = 5.0) : 
  skittles_per_friend * num_friends = 200.0 := by
  sorry

end joshuas_skittles_l2915_291598


namespace x_squared_minus_y_squared_l2915_291561

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 2/15) : x^2 - y^2 = 16/225 := by
  sorry

end x_squared_minus_y_squared_l2915_291561


namespace product_of_good_sequences_is_good_l2915_291547

-- Define a sequence as a function from ℕ to ℝ
def Sequence := ℕ → ℝ

-- Define the first derivative of a sequence
def FirstDerivative (a : Sequence) : Sequence :=
  λ n => a (n + 1) - a n

-- Define the k-th derivative of a sequence
def KthDerivative (a : Sequence) : ℕ → Sequence
  | 0 => a
  | k + 1 => FirstDerivative (KthDerivative a k)

-- Define a good sequence
def IsGoodSequence (a : Sequence) : Prop :=
  ∀ k n, KthDerivative a k n > 0

-- Theorem statement
theorem product_of_good_sequences_is_good
  (a b : Sequence)
  (ha : IsGoodSequence a)
  (hb : IsGoodSequence b) :
  IsGoodSequence (λ n => a n * b n) :=
by sorry

end product_of_good_sequences_is_good_l2915_291547


namespace prob_same_heads_is_five_thirty_seconds_l2915_291549

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 2

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The probability of getting heads on a single penny toss -/
def prob_heads : ℚ := 1/2

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def prob_same_heads : ℚ := 5/32

/-- Theorem stating that the probability of Ephraim getting the same number of heads as Keiko is 5/32 -/
theorem prob_same_heads_is_five_thirty_seconds :
  prob_same_heads = 5/32 := by sorry

end prob_same_heads_is_five_thirty_seconds_l2915_291549


namespace a_14_equals_41_l2915_291563

/-- An arithmetic sequence with a_2 = 5 and a_6 = 17 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  a 2 = 5 ∧ a 6 = 17 ∧ ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In the given arithmetic sequence, a_14 = 41 -/
theorem a_14_equals_41 (a : ℕ → ℝ) (h : arithmetic_sequence a) : a 14 = 41 := by
  sorry

end a_14_equals_41_l2915_291563


namespace find_other_number_l2915_291522

theorem find_other_number (A B : ℕ) (hA : A = 24) (hHCF : Nat.gcd A B = 14) (hLCM : Nat.lcm A B = 312) : B = 182 := by
  sorry

end find_other_number_l2915_291522


namespace spaceship_age_conversion_l2915_291501

/-- Converts a three-digit number in base 9 to base 10 --/
def base9_to_base10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 9^2 + tens * 9^1 + ones * 9^0

/-- The age of the alien spaceship --/
def spaceship_age : Nat := 362

theorem spaceship_age_conversion :
  base9_to_base10 3 6 2 = 299 :=
by sorry

end spaceship_age_conversion_l2915_291501


namespace frequency_not_necessarily_equal_probability_l2915_291555

/-- Represents the outcome of a single trial in the random simulation -/
inductive Outcome
  | selected
  | notSelected

/-- Represents the result of multiple trials in the random simulation -/
structure SimulationResult :=
  (trials : ℕ)
  (selections : ℕ)

/-- The theoretical probability of selecting a specific student from 6 students -/
def theoretical_probability : ℚ := 1 / 6

/-- The frequency of selecting a specific student in a simulation -/
def frequency (result : SimulationResult) : ℚ :=
  result.selections / result.trials

/-- Statement: The frequency in a random simulation is not necessarily equal to the theoretical probability -/
theorem frequency_not_necessarily_equal_probability :
  ∃ (result : SimulationResult), frequency result ≠ theoretical_probability :=
sorry

end frequency_not_necessarily_equal_probability_l2915_291555


namespace power_equation_solution_l2915_291521

theorem power_equation_solution (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26 → n = 25 := by
  sorry

end power_equation_solution_l2915_291521


namespace rectangle_ratio_l2915_291544

/-- Represents the configuration of squares and a rectangle forming a larger square -/
structure SquareConfiguration where
  small_square_side : ℝ
  large_square_side : ℝ
  rectangle_length : ℝ
  rectangle_width : ℝ

/-- The theorem stating the ratio of the rectangle's length to its width -/
theorem rectangle_ratio (config : SquareConfiguration) 
  (h1 : config.large_square_side = 4 * config.small_square_side)
  (h2 : config.rectangle_length = config.large_square_side)
  (h3 : config.rectangle_width = config.large_square_side - 3 * config.small_square_side) :
  config.rectangle_length / config.rectangle_width = 4 := by
  sorry

end rectangle_ratio_l2915_291544


namespace cube_sum_reciprocal_l2915_291540

theorem cube_sum_reciprocal (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end cube_sum_reciprocal_l2915_291540


namespace gcd_cube_plus_square_and_linear_l2915_291539

theorem gcd_cube_plus_square_and_linear (n m : ℤ) (hn : n > 2^3) : 
  Int.gcd (n^3 + m^2) (n + 2) = 1 := by sorry

end gcd_cube_plus_square_and_linear_l2915_291539


namespace billy_sam_money_multiple_l2915_291574

/-- Given that Sam has $75 and Billy has $25 less than a multiple of Sam's money,
    and together they have $200, prove that the multiple is 2. -/
theorem billy_sam_money_multiple : 
  ∀ (sam_money : ℕ) (total_money : ℕ) (multiple : ℚ),
    sam_money = 75 →
    total_money = 200 →
    total_money = sam_money + (multiple * sam_money - 25) →
    multiple = 2 := by
  sorry

end billy_sam_money_multiple_l2915_291574


namespace problem_solution_l2915_291556

theorem problem_solution : ∃ x : ℝ, 400 * x = 28000 * 100^1 ∧ x = 7000 := by
  sorry

end problem_solution_l2915_291556


namespace simplify_86_with_95_base_l2915_291599

/-- Simplifies a score based on a given base score. -/
def simplify_score (score : Int) (base : Int) : Int :=
  score - base

/-- The base score considered as excellent. -/
def excellent_score : Int := 95

/-- Theorem: Simplifying a score of 86 with 95 as the base results in -9. -/
theorem simplify_86_with_95_base :
  simplify_score 86 excellent_score = -9 := by
  sorry

end simplify_86_with_95_base_l2915_291599


namespace flour_with_weevils_l2915_291558

theorem flour_with_weevils 
  (p_good_milk : ℝ) 
  (p_good_egg : ℝ) 
  (p_all_good : ℝ) 
  (h1 : p_good_milk = 0.8) 
  (h2 : p_good_egg = 0.4) 
  (h3 : p_all_good = 0.24) : 
  ∃ (p_good_flour : ℝ), 
    p_good_milk * p_good_egg * p_good_flour = p_all_good ∧ 
    1 - p_good_flour = 0.25 :=
by sorry

end flour_with_weevils_l2915_291558


namespace geometric_series_sum_127_128_l2915_291535

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_127_128 : 
  geometric_series_sum (1/2) (1/2) 7 = 127/128 := by
  sorry

end geometric_series_sum_127_128_l2915_291535


namespace sin_plus_cos_value_l2915_291596

theorem sin_plus_cos_value (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π) 
  (h3 : Real.tan (θ + π/4) = 1/7) : 
  Real.sin θ + Real.cos θ = -1/5 := by
sorry

end sin_plus_cos_value_l2915_291596


namespace reservoir_capacity_proof_l2915_291528

theorem reservoir_capacity_proof (current_level : ℝ) (normal_level : ℝ) (total_capacity : ℝ)
  (h1 : current_level = 30)
  (h2 : current_level = 2 * normal_level)
  (h3 : current_level = 0.75 * total_capacity) :
  total_capacity - normal_level = 25 := by
  sorry

end reservoir_capacity_proof_l2915_291528


namespace problem_solution_l2915_291533

-- Define the conditions
def is_square_root_of_same_number (x y : ℝ) : Prop := ∃ z : ℝ, z > 0 ∧ x^2 = z ∧ y^2 = z

-- Main theorem
theorem problem_solution :
  ∀ (a b c : ℝ),
  (is_square_root_of_same_number (a + 3) (2*a - 15)) →
  (b^(1/3 : ℝ) = -2) →
  (c ≥ 0 ∧ c^(1/2 : ℝ) = c) →
  ((c = 0 → a + b - 2*c = -4) ∧ (c = 1 → a + b - 2*c = -6)) :=
by sorry

end problem_solution_l2915_291533


namespace root_and_c_value_l2915_291581

theorem root_and_c_value (x : ℝ) (c : ℝ) : 
  (2 + Real.sqrt 3)^2 - 4*(2 + Real.sqrt 3) + c = 0 →
  (∃ y : ℝ, y ≠ 2 + Real.sqrt 3 ∧ y^2 - 4*y + c = 0 ∧ y = 2 - Real.sqrt 3) ∧
  c = 1 := by
  sorry

end root_and_c_value_l2915_291581


namespace second_project_breadth_l2915_291564

/-- Represents a digging project with depth, length, and breadth dimensions -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of a digging project -/
def volume (project : DiggingProject) : ℝ :=
  project.depth * project.length * project.breadth

/-- Theorem: Given two digging projects with equal volumes, 
    the breadth of the second project is 50 meters -/
theorem second_project_breadth
  (project1 : DiggingProject)
  (project2 : DiggingProject)
  (h1 : project1.depth = 100)
  (h2 : project1.length = 25)
  (h3 : project1.breadth = 30)
  (h4 : project2.depth = 75)
  (h5 : project2.length = 20)
  (h_equal_volume : volume project1 = volume project2) :
  project2.breadth = 50 := by
  sorry

#check second_project_breadth

end second_project_breadth_l2915_291564


namespace shekar_science_marks_l2915_291525

/-- Represents a student's marks in different subjects -/
structure StudentMarks where
  mathematics : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ
  science : ℕ
  average : ℕ
  total_subjects : ℕ

/-- Theorem stating that given Shekar's marks in other subjects and his average, 
    his science marks must be 65 -/
theorem shekar_science_marks (marks : StudentMarks) 
  (h1 : marks.mathematics = 76)
  (h2 : marks.social_studies = 82)
  (h3 : marks.english = 47)
  (h4 : marks.biology = 85)
  (h5 : marks.average = 71)
  (h6 : marks.total_subjects = 5)
  : marks.science = 65 := by
  sorry

#check shekar_science_marks

end shekar_science_marks_l2915_291525


namespace perpendicular_vectors_lambda_l2915_291585

/-- Given two vectors a and b in ℝ², where a is perpendicular to (a - b), prove that the second component of b equals 4. -/
theorem perpendicular_vectors_lambda (a b : ℝ × ℝ) : 
  a = (-1, 3) → 
  b.1 = 2 → 
  a • (a - b) = 0 → 
  b.2 = 4 := by sorry

end perpendicular_vectors_lambda_l2915_291585


namespace parrots_per_cage_l2915_291505

/-- Given a pet store with bird cages, prove the number of parrots in each cage. -/
theorem parrots_per_cage 
  (num_cages : ℕ) 
  (parakeets_per_cage : ℕ) 
  (total_birds : ℕ) 
  (h1 : num_cages = 9)
  (h2 : parakeets_per_cage = 6)
  (h3 : total_birds = 72) :
  (total_birds - num_cages * parakeets_per_cage) / num_cages = 2 := by
sorry

end parrots_per_cage_l2915_291505


namespace hyperbola_ellipse_relationship_range_of_m_l2915_291514

def P (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/(m-1) + y^2/(m-4) = 1 ∧ (m-1)*(m-4) < 0

def Q (m : ℝ) : Prop := ∃ (x y : ℝ), x^2/(m-2) + y^2/(4-m) = 1 ∧ m-2 > 0 ∧ 4-m > 0 ∧ m-2 ≠ 4-m

theorem hyperbola_ellipse_relationship (m : ℝ) :
  (P m → Q m) ∧ ¬(Q m → P m) :=
sorry

theorem range_of_m (m : ℝ) :
  (¬(P m ∧ Q m) ∧ (P m ∨ Q m)) → ((1 < m ∧ m ≤ 2) ∨ m = 3) :=
sorry

end hyperbola_ellipse_relationship_range_of_m_l2915_291514


namespace axiom_1_l2915_291545

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the membership relations
variable (pointOnLine : Point → Line → Prop)
variable (pointInPlane : Point → Plane → Prop)
variable (lineInPlane : Line → Plane → Prop)

-- State the theorem
theorem axiom_1 (A B : Point) (l : Line) (α : Plane) :
  pointOnLine A l → pointOnLine B l → pointInPlane A α → pointInPlane B α →
  lineInPlane l α := by
  sorry

end axiom_1_l2915_291545


namespace simplify_and_rationalize_l2915_291529

theorem simplify_and_rationalize (a b c d e f g h i : ℝ) 
  (ha : a = 3) (hb : b = 7) (hc : c = 5) (hd : d = 8) (he : e = 6) (hf : f = 9) :
  (Real.sqrt a / Real.sqrt b) * (Real.sqrt c / Real.sqrt d) * (Real.sqrt e / Real.sqrt f) = 
  Real.sqrt 35 / 14 := by
  sorry

end simplify_and_rationalize_l2915_291529


namespace money_distribution_l2915_291527

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 700)
  (ac_sum : A + C = 300)
  (bc_sum : B + C = 600) :
  C = 200 := by
  sorry

end money_distribution_l2915_291527


namespace triangle_area_l2915_291575

/-- The area of a triangle with base 12 and height 9 is 54 -/
theorem triangle_area : ∀ (base height : ℝ), 
  base = 12 → height = 9 → (1/2 : ℝ) * base * height = 54 := by
  sorry

#check triangle_area

end triangle_area_l2915_291575


namespace initial_travel_time_l2915_291512

theorem initial_travel_time (distance : ℝ) (new_speed : ℝ) :
  distance = 540 ∧ new_speed = 60 →
  ∃ initial_time : ℝ,
    distance = new_speed * (3/4 * initial_time) ∧
    initial_time = 12 := by
  sorry

end initial_travel_time_l2915_291512


namespace total_voters_l2915_291587

/-- The number of voters in each district --/
structure VoterCount where
  district1 : ℕ
  district2 : ℕ
  district3 : ℕ
  district4 : ℕ
  district5 : ℕ
  district6 : ℕ
  district7 : ℕ

/-- The conditions for voter counts in each district --/
def validVoterCount (v : VoterCount) : Prop :=
  v.district1 = 322 ∧
  v.district2 = v.district1 / 2 - 19 ∧
  v.district3 = 2 * v.district1 ∧
  v.district4 = v.district2 + 45 ∧
  v.district5 = 3 * v.district3 - 150 ∧
  v.district6 = (v.district1 + v.district4) + (v.district1 + v.district4) / 5 ∧
  v.district7 = v.district2 + (v.district5 - v.district2) / 2

/-- The theorem stating that the sum of voters in all districts is 4650 --/
theorem total_voters (v : VoterCount) (h : validVoterCount v) :
  v.district1 + v.district2 + v.district3 + v.district4 + v.district5 + v.district6 + v.district7 = 4650 := by
  sorry

end total_voters_l2915_291587


namespace hyperbola_asymptotes_l2915_291506

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 36 = 1

-- Define the asymptote equations
def asymptote_equations (x y : ℝ) : Prop :=
  y = 3*x ∨ y = -3*x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equations x y :=
sorry

end hyperbola_asymptotes_l2915_291506


namespace point_coordinates_l2915_291570

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The second quadrant of the 2D plane -/
def SecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def DistanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def DistanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Theorem: If a point P is in the second quadrant, its distance to the x-axis is 3,
    and its distance to the y-axis is 10, then its coordinates are (-10, 3) -/
theorem point_coordinates (P : Point)
  (h1 : SecondQuadrant P)
  (h2 : DistanceToXAxis P = 3)
  (h3 : DistanceToYAxis P = 10) :
  P.x = -10 ∧ P.y = 3 := by
  sorry

end point_coordinates_l2915_291570


namespace tissue_pallet_ratio_l2915_291560

theorem tissue_pallet_ratio (total_pallets : ℕ) 
  (paper_towel_pallets : ℕ) (paper_plate_pallets : ℕ) (paper_cup_pallets : ℕ) :
  total_pallets = 20 →
  paper_towel_pallets = total_pallets / 2 →
  paper_plate_pallets = total_pallets / 5 →
  paper_cup_pallets = 1 →
  let tissue_pallets := total_pallets - (paper_towel_pallets + paper_plate_pallets + paper_cup_pallets)
  (tissue_pallets : ℚ) / (total_pallets : ℚ) = 1 / 4 := by
sorry

end tissue_pallet_ratio_l2915_291560


namespace tangent_line_curve_intersection_l2915_291536

/-- Given a line y = kx + 1 tangent to the curve y = x^3 + ax + b at the point (1, 3), 
    prove that b = -3 -/
theorem tangent_line_curve_intersection (k a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 
    (k * x + 1 = x^3 + a * x + b → x = 1) ∧ 
    (k * 1 + 1 = 1^3 + a * 1 + b) ∧
    (k = 3 * 1^2 + a)) → 
  (∃ b : ℝ, k * 1 + 1 = 1^3 + a * 1 + b ∧ b = -3) :=
sorry

end tangent_line_curve_intersection_l2915_291536


namespace committee_formation_count_l2915_291511

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of Republicans in the city council -/
def num_republicans : ℕ := 10

/-- The number of Democrats in the city council -/
def num_democrats : ℕ := 7

/-- The number of Republicans needed in the committee -/
def committee_republicans : ℕ := 4

/-- The number of Democrats needed in the committee -/
def committee_democrats : ℕ := 3

/-- The total number of ways to form the committee -/
def total_committee_formations : ℕ := 
  binomial num_republicans committee_republicans * binomial num_democrats committee_democrats

theorem committee_formation_count : total_committee_formations = 7350 := by
  sorry

end committee_formation_count_l2915_291511


namespace sin_660_deg_l2915_291520

theorem sin_660_deg : Real.sin (660 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_660_deg_l2915_291520


namespace tank_capacity_l2915_291595

theorem tank_capacity (initial_fullness final_fullness : ℚ) (added_water : ℕ) : 
  initial_fullness = 1/4 →
  final_fullness = 3/4 →
  added_water = 208 →
  (final_fullness - initial_fullness) * (added_water / (final_fullness - initial_fullness)) = 416 :=
by sorry

end tank_capacity_l2915_291595


namespace absolute_value_integral_l2915_291568

theorem absolute_value_integral : ∫ x in (0)..(4), |x - 2| = 4 := by
  sorry

end absolute_value_integral_l2915_291568


namespace alice_bob_meeting_l2915_291519

/-- The number of points on the circle -/
def n : ℕ := 18

/-- Alice's movement (clockwise) -/
def a : ℕ := 7

/-- Bob's movement (counterclockwise) -/
def b : ℕ := 13

/-- The number of turns it takes for Alice and Bob to meet again -/
def meetingTurns : ℕ := 9

/-- Theorem stating that Alice and Bob meet after 'meetingTurns' turns -/
theorem alice_bob_meeting :
  (meetingTurns * (a + n - b)) % n = 0 := by sorry

end alice_bob_meeting_l2915_291519


namespace polygon_equidistant_point_l2915_291593

-- Define a convex polygon
def ConvexPolygon (V : Type*) := V → ℝ × ℝ

-- Define a point inside the polygon
def InsidePoint (P : ConvexPolygon V) (O : ℝ × ℝ) : Prop := sorry

-- Define the property of forming isosceles triangles
def FormsIsoscelesTriangles (P : ConvexPolygon V) (O : ℝ × ℝ) : Prop :=
  ∀ (v1 v2 : V), v1 ≠ v2 → ‖P v1 - O‖ = ‖P v2 - O‖

-- Define the property of being equidistant from all vertices
def EquidistantFromVertices (P : ConvexPolygon V) (O : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), ∀ (v : V), ‖P v - O‖ = r

-- State the theorem
theorem polygon_equidistant_point {V : Type*} (P : ConvexPolygon V) (O : ℝ × ℝ) :
  InsidePoint P O → FormsIsoscelesTriangles P O → EquidistantFromVertices P O :=
sorry

end polygon_equidistant_point_l2915_291593


namespace triangle_properties_l2915_291592

/-- Triangle ABC with vertices A(0, 3), B(-2, -1), and C(4, 3) -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The given triangle -/
def givenTriangle : Triangle where
  A := (0, 3)
  B := (-2, -1)
  C := (4, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The altitude from side AB -/
def altitudeAB (t : Triangle) : Line :=
  sorry

/-- The point symmetric to C with respect to line AB -/
def symmetricPointC (t : Triangle) : Point :=
  sorry

theorem triangle_properties (t : Triangle) (h : t = givenTriangle) :
  (altitudeAB t = Line.mk 1 2 (-10)) ∧
  (symmetricPointC t = Point.mk (-12/5) (31/5)) := by
  sorry

end triangle_properties_l2915_291592


namespace exponential_function_determination_l2915_291583

theorem exponential_function_determination (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a^x)
  (h2 : a > 0)
  (h3 : a ≠ 1)
  (h4 : f 2 = 4) :
  ∀ x, f x = 2^x :=
by sorry

end exponential_function_determination_l2915_291583


namespace quadratic_inequality_solution_set_l2915_291524

theorem quadratic_inequality_solution_set :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2} = {x : ℝ | -x^2 + 3*x - 2 ≥ 0} := by sorry

end quadratic_inequality_solution_set_l2915_291524


namespace semicircle_circumference_from_rectangle_l2915_291526

/-- The circumference of a semicircle given rectangle dimensions --/
theorem semicircle_circumference_from_rectangle (l b : ℝ) (h1 : l = 24) (h2 : b = 16) :
  let rectangle_perimeter := 2 * (l + b)
  let square_side := rectangle_perimeter / 4
  let semicircle_circumference := π * square_side / 2 + square_side
  ‖semicircle_circumference - 51.40‖ < 0.01 := by
  sorry

end semicircle_circumference_from_rectangle_l2915_291526


namespace vann_teeth_cleaning_l2915_291584

/-- The number of teeth a dog has -/
def dog_teeth : ℕ := 42

/-- The number of teeth a cat has -/
def cat_teeth : ℕ := 30

/-- The number of teeth a pig has -/
def pig_teeth : ℕ := 44

/-- The number of teeth a horse has -/
def horse_teeth : ℕ := 40

/-- The number of teeth a rabbit has -/
def rabbit_teeth : ℕ := 28

/-- The number of dogs Vann will clean -/
def num_dogs : ℕ := 7

/-- The number of cats Vann will clean -/
def num_cats : ℕ := 12

/-- The number of pigs Vann will clean -/
def num_pigs : ℕ := 9

/-- The number of horses Vann will clean -/
def num_horses : ℕ := 4

/-- The number of rabbits Vann will clean -/
def num_rabbits : ℕ := 15

/-- The total number of teeth Vann will clean -/
def total_teeth : ℕ := 
  num_dogs * dog_teeth + 
  num_cats * cat_teeth + 
  num_pigs * pig_teeth + 
  num_horses * horse_teeth + 
  num_rabbits * rabbit_teeth

theorem vann_teeth_cleaning : total_teeth = 1630 := by
  sorry

end vann_teeth_cleaning_l2915_291584


namespace M_subset_N_l2915_291542

-- Define the set M
def M : Set ℝ := {x | ∃ k : ℤ, x = (k / 2 : ℝ) * 180 + 45}

-- Define the set N
def N : Set ℝ := {x | ∃ k : ℤ, x = (k / 4 : ℝ) * 180 + 45}

-- Theorem stating that M is a subset of N
theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l2915_291542


namespace mutually_exclusive_events_l2915_291578

-- Define the sample space
def Ω : Type := Unit

-- Define the events
def both_miss : Set Ω := sorry
def hit_at_least_once : Set Ω := sorry

-- Define the theorem
theorem mutually_exclusive_events : 
  both_miss ∩ hit_at_least_once = ∅ := by sorry

end mutually_exclusive_events_l2915_291578


namespace two_numbers_difference_l2915_291537

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (squares_diff_eq : x^2 - y^2 = 20) : 
  |x - y| = 2 := by
sorry

end two_numbers_difference_l2915_291537


namespace greyson_fuel_expense_l2915_291531

/-- Calculates the total fuel expense for a week given the number of refills and cost per refill -/
def total_fuel_expense (num_refills : ℕ) (cost_per_refill : ℕ) : ℕ :=
  num_refills * cost_per_refill

/-- Proves that Greyson's total fuel expense for the week is $40 -/
theorem greyson_fuel_expense :
  total_fuel_expense 4 10 = 40 := by
  sorry

end greyson_fuel_expense_l2915_291531


namespace overall_percentage_correct_l2915_291503

theorem overall_percentage_correct (score1 score2 score3 : ℚ)
  (problems1 problems2 problems3 : ℕ) : 
  score1 = 75 / 100 ∧ 
  score2 = 85 / 100 ∧ 
  score3 = 60 / 100 ∧
  problems1 = 20 ∧
  problems2 = 50 ∧
  problems3 = 15 →
  (score1 * problems1 + score2 * problems2 + score3 * problems3) / 
  (problems1 + problems2 + problems3) = 79 / 100 := by
sorry

#eval (15 + 43 + 9) / (20 + 50 + 15)  -- Should evaluate to approximately 0.7882

end overall_percentage_correct_l2915_291503


namespace bird_watching_average_l2915_291523

theorem bird_watching_average :
  let marcus_birds : ℕ := 7
  let humphrey_birds : ℕ := 11
  let darrel_birds : ℕ := 9
  let total_birds : ℕ := marcus_birds + humphrey_birds + darrel_birds
  let num_people : ℕ := 3
  (total_birds : ℚ) / num_people = 9 := by sorry

end bird_watching_average_l2915_291523


namespace mean_calculation_l2915_291569

theorem mean_calculation (x : ℝ) : 
  (28 + x + 50 + 78 + 104) / 5 = 62 → 
  (48 + 62 + 98 + 124 + x) / 5 = 76.4 := by
sorry

end mean_calculation_l2915_291569


namespace quadratic_form_ratio_l2915_291562

theorem quadratic_form_ratio (x : ℝ) : 
  ∃ (d e : ℝ), x^2 + 2023*x + 2023 = (x + d)^2 + e ∧ e/d = -1009.75 := by
sorry

end quadratic_form_ratio_l2915_291562


namespace sum_becomes_27_l2915_291552

def numbers : List ℝ := [1.05, 1.15, 1.25, 1.4, 1.5, 1.6, 1.75, 1.85, 1.95]

def sum_with_error (nums : List ℝ) (error_index : Nat) : ℝ :=
  let error_value := nums[error_index]! * 10
  (nums.sum - nums[error_index]!) + error_value

theorem sum_becomes_27 :
  ∃ (i : Nat), i < numbers.length ∧ sum_with_error numbers i = 27 := by
  sorry

end sum_becomes_27_l2915_291552


namespace polynomial_remainder_l2915_291517

theorem polynomial_remainder (x : ℝ) : 
  (x^5 + 2*x^2 + 1) % (x - 2) = 41 := by
  sorry

end polynomial_remainder_l2915_291517


namespace smallest_three_digit_multiple_l2915_291510

theorem smallest_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (∃ k : ℕ, n = 3 * k + 1) ∧
  (∃ k : ℕ, n = 4 * k + 1) ∧
  (∃ k : ℕ, n = 5 * k + 1) ∧
  (∃ k : ℕ, n = 7 * k + 1) ∧
  (∀ m : ℕ, m < n → 
    (m < 100 ∨ m ≥ 1000 ∨
    (∀ k : ℕ, m ≠ 3 * k + 1) ∨
    (∀ k : ℕ, m ≠ 4 * k + 1) ∨
    (∀ k : ℕ, m ≠ 5 * k + 1) ∨
    (∀ k : ℕ, m ≠ 7 * k + 1))) ∧
  n = 421 :=
by
  sorry

end smallest_three_digit_multiple_l2915_291510


namespace simplify_fraction_l2915_291557

theorem simplify_fraction : (90 : ℚ) / 150 = 3 / 5 := by
  sorry

end simplify_fraction_l2915_291557


namespace shower_usage_solution_l2915_291518

/-- The water usage for Roman and Remy's showers -/
def shower_usage (R : ℝ) : Prop :=
  let remy_usage := 3 * R + 1
  R + remy_usage = 33 ∧ remy_usage = 25

/-- Theorem stating that there exists a value for Roman's usage satisfying the conditions -/
theorem shower_usage_solution : ∃ R : ℝ, shower_usage R := by
  sorry

end shower_usage_solution_l2915_291518


namespace geometric_sequence_sum_l2915_291580

/-- A geometric sequence where the sum of every two adjacent terms forms a geometric sequence --/
def GeometricSequenceWithAdjacentSums (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 2) + a (n + 3) = r * (a n + a (n + 1))

/-- The theorem stating the sum of specific terms in the geometric sequence --/
theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequenceWithAdjacentSums a)
  (h_sum1 : a 1 + a 2 = 1/2)
  (h_sum2 : a 3 + a 4 = 1) :
  a 7 + a 8 + a 9 + a 10 = 12 := by
  sorry

end geometric_sequence_sum_l2915_291580


namespace triangle_trigonometric_identity_l2915_291534

theorem triangle_trigonometric_identity (A B C : Real) : 
  C = Real.pi / 3 →
  Real.tan (A / 2) + Real.tan (B / 2) = 1 →
  A + B + C = Real.pi →
  Real.sin (A / 2) * Real.sin (B / 2) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end triangle_trigonometric_identity_l2915_291534


namespace min_value_expression_l2915_291591

theorem min_value_expression (x : ℝ) (hx : x > 0) :
  4 * Real.sqrt x + 4 / x + 1 / (x^2) ≥ 9 ∧
  ∃ y : ℝ, y > 0 ∧ 4 * Real.sqrt y + 4 / y + 1 / (y^2) = 9 :=
sorry

end min_value_expression_l2915_291591


namespace sally_mcqueen_cost_l2915_291508

/-- The cost of Sally McQueen given the costs of Lightning McQueen and Mater -/
theorem sally_mcqueen_cost 
  (lightning_cost : ℝ) 
  (mater_cost_percentage : ℝ) 
  (sally_cost_multiplier : ℝ) 
  (h1 : lightning_cost = 140000)
  (h2 : mater_cost_percentage = 0.1)
  (h3 : sally_cost_multiplier = 3) : 
  sally_cost_multiplier * (mater_cost_percentage * lightning_cost) = 42000 := by
  sorry

end sally_mcqueen_cost_l2915_291508


namespace train_final_speed_l2915_291576

/-- Given a train with the following properties:
  * Length: 360 meters
  * Initial velocity: 0 m/s (starts from rest)
  * Acceleration: 1 m/s²
  * Time to cross a man on the platform: 20 seconds
Prove that the final speed of the train is 20 m/s. -/
theorem train_final_speed
  (length : ℝ)
  (initial_velocity : ℝ)
  (acceleration : ℝ)
  (time : ℝ)
  (h1 : length = 360)
  (h2 : initial_velocity = 0)
  (h3 : acceleration = 1)
  (h4 : time = 20)
  : initial_velocity + acceleration * time = 20 := by
  sorry

end train_final_speed_l2915_291576


namespace specific_ellipse_area_l2915_291548

/-- An ellipse with given properties --/
structure Ellipse where
  major_axis_endpoint1 : ℝ × ℝ
  major_axis_endpoint2 : ℝ × ℝ
  point_on_ellipse : ℝ × ℝ

/-- The area of an ellipse --/
def ellipse_area (e : Ellipse) : ℝ := sorry

/-- Theorem stating the area of the specific ellipse --/
theorem specific_ellipse_area :
  let e : Ellipse := {
    major_axis_endpoint1 := (-8, 3),
    major_axis_endpoint2 := (12, 3),
    point_on_ellipse := (10, 6)
  }
  ellipse_area e = 50 * Real.pi := by sorry

end specific_ellipse_area_l2915_291548


namespace truck_travel_distance_l2915_291577

/-- Given a truck that travels 300 miles on 10 gallons of diesel,
    prove that it will travel 450 miles on 15 gallons of diesel. -/
theorem truck_travel_distance (initial_distance : ℝ) (initial_fuel : ℝ) (new_fuel : ℝ)
    (h1 : initial_distance = 300)
    (h2 : initial_fuel = 10)
    (h3 : new_fuel = 15) :
    (initial_distance / initial_fuel) * new_fuel = 450 :=
by sorry

end truck_travel_distance_l2915_291577


namespace profit_percent_calculation_l2915_291579

theorem profit_percent_calculation (selling_price cost_price : ℝ) 
  (h : cost_price = 0.82 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100 / 82 - 1) * 100 := by
  sorry

end profit_percent_calculation_l2915_291579


namespace distance_is_150_l2915_291513

/-- The distance between point A and point B in kilometers. -/
def distance : ℝ := 150

/-- The original speed of the car in kilometers per hour. -/
def original_speed : ℝ := sorry

/-- The original travel time in hours. -/
def original_time : ℝ := sorry

/-- Condition 1: If the car's speed is increased by 20%, the car can arrive 25 minutes earlier. -/
axiom condition1 : distance / (original_speed * 1.2) = original_time - 25 / 60

/-- Condition 2: If the car travels 100 kilometers at the original speed and then increases its speed by 25%, the car can arrive 10 minutes earlier. -/
axiom condition2 : 100 / original_speed + (distance - 100) / (original_speed * 1.25) = original_time - 10 / 60

/-- The theorem stating that the distance between point A and point B is 150 kilometers. -/
theorem distance_is_150 : distance = 150 := by sorry

end distance_is_150_l2915_291513


namespace tan_value_from_sin_cos_equation_l2915_291502

theorem tan_value_from_sin_cos_equation (x : ℝ) 
  (h1 : 0 < x ∧ x < Real.pi / 2) 
  (h2 : Real.sin x ^ 4 / 42 + Real.cos x ^ 4 / 75 = 1 / 117) : 
  Real.tan x = Real.sqrt 14 / 5 := by
sorry

end tan_value_from_sin_cos_equation_l2915_291502


namespace glass_pane_impact_l2915_291504

/-- Represents a point inside a rectangle --/
structure ImpactPoint (width height : ℝ) where
  x : ℝ
  y : ℝ
  x_bound : 0 < x ∧ x < width
  y_bound : 0 < y ∧ y < height

/-- The glass pane problem --/
theorem glass_pane_impact
  (width : ℝ)
  (height : ℝ)
  (p : ImpactPoint width height)
  (h_width : width = 8)
  (h_height : height = 6)
  (h_right_area : p.x * height = 3 * (width - p.x) * height)
  (h_bottom_area : p.y * width = 2 * (height - p.y) * p.x) :
  p.x = 2 ∧ (width - p.x) = 6 ∧ p.y = 3 ∧ (height - p.y) = 3 := by
  sorry

end glass_pane_impact_l2915_291504


namespace complex_equation_solution_l2915_291594

theorem complex_equation_solution (z : ℂ) : z * (2 - Complex.I) = 3 + Complex.I → z = 1 + Complex.I := by
  sorry

end complex_equation_solution_l2915_291594


namespace remainder_problem_l2915_291538

theorem remainder_problem : Int.mod (179 + 231 - 359) 37 = 14 := by
  sorry

end remainder_problem_l2915_291538


namespace plywood_cut_perimeter_difference_l2915_291553

/-- Represents the dimensions of a rectangle --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the plywood and its cutting --/
structure Plywood where
  original : Rectangle
  pieces : Fin 8 → Rectangle

/-- The original plywood dimensions --/
def original_plywood : Rectangle := { length := 16, width := 4 }

theorem plywood_cut_perimeter_difference :
  ∃ (max_cut min_cut : Plywood),
    (∀ i : Fin 8, perimeter (max_cut.pieces i) ≥ perimeter (min_cut.pieces i)) ∧
    (∀ cut : Plywood, ∀ i : Fin 8, 
      perimeter (cut.pieces i) ≤ perimeter (max_cut.pieces i) ∧
      perimeter (cut.pieces i) ≥ perimeter (min_cut.pieces i)) ∧
    (∀ i j : Fin 8, max_cut.pieces i = max_cut.pieces j) ∧
    (∀ i j : Fin 8, min_cut.pieces i = min_cut.pieces j) ∧
    (max_cut.original = original_plywood) ∧
    (min_cut.original = original_plywood) ∧
    (perimeter (max_cut.pieces 0) - perimeter (min_cut.pieces 0) = 21) :=
by sorry

end plywood_cut_perimeter_difference_l2915_291553


namespace mika_gave_six_stickers_l2915_291572

/-- Represents the number of stickers Mika had, bought, received, used, and gave away --/
structure StickerCount where
  initial : Nat
  bought : Nat
  birthday : Nat
  usedForCard : Nat
  leftOver : Nat

/-- Calculates the number of stickers Mika gave to her sister --/
def stickersGivenToSister (s : StickerCount) : Nat :=
  s.initial + s.bought + s.birthday - (s.usedForCard + s.leftOver)

/-- Theorem stating that Mika gave 6 stickers to her sister --/
theorem mika_gave_six_stickers (s : StickerCount) 
  (h1 : s.initial = 20)
  (h2 : s.bought = 26)
  (h3 : s.birthday = 20)
  (h4 : s.usedForCard = 58)
  (h5 : s.leftOver = 2) : 
  stickersGivenToSister s = 6 := by
  sorry

end mika_gave_six_stickers_l2915_291572


namespace greatest_b_for_quadratic_range_l2915_291571

theorem greatest_b_for_quadratic_range : ∃ (b : ℤ), 
  (∀ x : ℝ, x^2 + b*x + 20 ≠ -4) ∧ 
  (∀ c : ℤ, c > b → ∃ x : ℝ, x^2 + c*x + 20 = -4) ∧
  b = 9 := by
  sorry

end greatest_b_for_quadratic_range_l2915_291571


namespace brenda_age_is_three_l2915_291586

/-- Represents the ages of family members -/
structure FamilyAges where
  addison : ℕ
  brenda : ℕ
  janet : ℕ

/-- The conditions given in the problem -/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.addison = 4 * ages.brenda ∧
  ages.janet = ages.brenda + 9 ∧
  ages.addison = ages.janet

/-- Theorem stating that if the family ages are valid, Brenda's age is 3 -/
theorem brenda_age_is_three (ages : FamilyAges) 
  (h : validFamilyAges ages) : ages.brenda = 3 := by
  sorry

#check brenda_age_is_three

end brenda_age_is_three_l2915_291586


namespace value_of_b_l2915_291532

theorem value_of_b (a b : ℝ) (h1 : a = 5) (h2 : a^2 + a*b = 60) : b = 7 := by
  sorry

end value_of_b_l2915_291532


namespace mean_median_difference_l2915_291543

/-- Represents the absence data for a class of students -/
structure AbsenceData where
  students : ℕ
  absences : List (ℕ × ℕ)  -- (days missed, number of students)

/-- Calculates the median number of days missed -/
def median (data : AbsenceData) : ℚ := sorry

/-- Calculates the mean number of days missed -/
def mean (data : AbsenceData) : ℚ := sorry

/-- The main theorem stating the difference between mean and median -/
theorem mean_median_difference (data : AbsenceData) : 
  data.students = 20 ∧ 
  data.absences = [(0, 4), (1, 3), (2, 7), (3, 2), (4, 2), (5, 1), (6, 1)] →
  mean data - median data = 1 / 10 := by sorry

end mean_median_difference_l2915_291543


namespace seashells_after_month_l2915_291573

/-- Calculates the number of seashells after a given number of weeks -/
def seashells_after_weeks (initial : ℕ) (weekly_increase : ℕ) (weeks : ℕ) : ℕ :=
  initial + weekly_increase * weeks

/-- Theorem stating that starting with 50 seashells and adding 20 per week for 4 weeks results in 130 seashells -/
theorem seashells_after_month (initial : ℕ) (weekly_increase : ℕ) (weeks : ℕ) 
    (h1 : initial = 50) 
    (h2 : weekly_increase = 20) 
    (h3 : weeks = 4) : 
  seashells_after_weeks initial weekly_increase weeks = 130 := by
  sorry

#eval seashells_after_weeks 50 20 4

end seashells_after_month_l2915_291573


namespace parallel_segments_length_l2915_291546

/-- Represents a line segment with a length -/
structure Segment where
  length : ℝ

/-- Represents three parallel line segments -/
structure ParallelSegments where
  ef : Segment
  gh : Segment
  ij : Segment

/-- Theorem: Given three parallel line segments EF, GH, and IJ,
    where IJ = 120 cm and EF = 180 cm, the length of GH is 72 cm -/
theorem parallel_segments_length 
  (segments : ParallelSegments) 
  (h1 : segments.ij.length = 120) 
  (h2 : segments.ef.length = 180) : 
  segments.gh.length = 72 := by
  sorry

end parallel_segments_length_l2915_291546


namespace mail_difference_l2915_291541

theorem mail_difference (monday tuesday wednesday thursday : ℕ) : 
  monday = 65 →
  tuesday = monday + 10 →
  wednesday < tuesday →
  thursday = wednesday + 15 →
  monday + tuesday + wednesday + thursday = 295 →
  tuesday - wednesday = 5 := by
sorry

end mail_difference_l2915_291541


namespace complementary_angle_of_60_13_25_l2915_291515

/-- Represents an angle in degrees, minutes, and seconds -/
structure DMS where
  degrees : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the complementary angle of a given angle in DMS format -/
def complementaryAngle (angle : DMS) : DMS :=
  sorry

/-- Theorem stating that the complementary angle of 60°13'25" is 29°46'35" -/
theorem complementary_angle_of_60_13_25 :
  let givenAngle : DMS := ⟨60, 13, 25⟩
  complementaryAngle givenAngle = ⟨29, 46, 35⟩ := by
  sorry

end complementary_angle_of_60_13_25_l2915_291515


namespace circle_center_coordinates_l2915_291565

/-- Given a circle tangent to lines 3x + 4y = 24 and 3x + 4y = 0,
    with its center on the line x - 2y = 0,
    prove that the center (x, y) satisfies the given equations. -/
theorem circle_center_coordinates (x y : ℚ) : 
  (∃ (r : ℚ), r > 0 ∧ 
    (∀ (x' y' : ℚ), (x' - x)^2 + (y' - y)^2 = r^2 → 
      (3*x' + 4*y' = 24 ∨ 3*x' + 4*y' = 0))) →
  x - 2*y = 0 →
  3*x + 4*y = 12 := by
sorry

end circle_center_coordinates_l2915_291565


namespace goat_cost_is_400_l2915_291509

/-- The cost of a single goat in dollars -/
def goat_cost : ℝ := sorry

/-- The number of goats purchased -/
def num_goats : ℕ := 3

/-- The number of llamas purchased -/
def num_llamas : ℕ := 6

/-- The cost of a single llama in terms of goat cost -/
def llama_cost : ℝ := 1.5 * goat_cost

/-- The total amount spent on all animals -/
def total_spent : ℝ := 4800

theorem goat_cost_is_400 : goat_cost = 400 :=
  sorry

end goat_cost_is_400_l2915_291509


namespace no_four_distinct_real_roots_l2915_291566

theorem no_four_distinct_real_roots (a b : ℝ) : 
  ¬ ∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (∀ x : ℝ, x^4 - 4*x^3 + 6*x^2 + a*x + b = 0 ↔ (x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄)) :=
sorry

end no_four_distinct_real_roots_l2915_291566


namespace product_remainder_by_ten_l2915_291500

theorem product_remainder_by_ten (a b c : ℕ) (ha : a = 1824) (hb : b = 5435) (hc : c = 80525) : 
  (a * b * c) % 10 = 0 := by
sorry

end product_remainder_by_ten_l2915_291500


namespace arithmetic_progression_with_prime_terms_l2915_291559

-- Define an arithmetic progression
def ArithmeticProgression (a k : ℕ) : ℕ → ℕ := fun n => a + k * n

-- Define the property of having infinitely many prime terms at prime indices
def HasInfinitelyManyPrimeTermsAtPrimeIndices (seq : ℕ → ℕ) : Prop :=
  ∃ N : ℕ, ∀ p : ℕ, Prime p → p > N → Prime (seq p)

-- State the theorem
theorem arithmetic_progression_with_prime_terms (seq : ℕ → ℕ) :
  (∃ a k : ℕ, seq = ArithmeticProgression a k) →
  HasInfinitelyManyPrimeTermsAtPrimeIndices seq →
  (∃ P : ℕ, Prime P ∧ seq = fun _ => P) ∨ seq = id :=
sorry

end arithmetic_progression_with_prime_terms_l2915_291559
