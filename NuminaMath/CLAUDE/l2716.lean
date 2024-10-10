import Mathlib

namespace test_marks_theorem_l2716_271627

/-- Represents a test section with a number of questions and a success rate -/
structure TestSection where
  questions : ℕ
  successRate : ℚ
  
/-- Calculates the total marks for a given test -/
def calculateTotalMarks (sections : List TestSection) : ℚ :=
  let correctAnswers := sections.map (fun s => (s.questions : ℚ) * s.successRate)
  let totalCorrect := correctAnswers.sum
  let totalQuestions := (sections.map (fun s => s.questions)).sum
  let incorrectAnswers := totalQuestions - totalCorrect.floor
  totalCorrect.floor - 0.25 * incorrectAnswers

/-- The theorem states that given the specific test conditions, the total marks obtained is 115 -/
theorem test_marks_theorem :
  let sections := [
    { questions := 50, successRate := 85/100 },
    { questions := 60, successRate := 70/100 },
    { questions := 40, successRate := 95/100 }
  ]
  calculateTotalMarks sections = 115 := by
  sorry

end test_marks_theorem_l2716_271627


namespace exam_students_count_l2716_271652

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
    N > 0 →
    T / N = 80 →
    (T - 100) / (N - 5) = 90 →
    N = 35 :=
by
  sorry

end exam_students_count_l2716_271652


namespace living_room_curtain_length_l2716_271655

/-- Given the dimensions of a bolt of fabric, bedroom curtain, and living room curtain width,
    as well as the remaining fabric area, prove the length of the living room curtain. -/
theorem living_room_curtain_length
  (bolt_width : ℝ)
  (bolt_length : ℝ)
  (bedroom_width : ℝ)
  (bedroom_length : ℝ)
  (living_room_width : ℝ)
  (remaining_area : ℝ)
  (h1 : bolt_width = 16)
  (h2 : bolt_length = 12)
  (h3 : bedroom_width = 2)
  (h4 : bedroom_length = 4)
  (h5 : living_room_width = 4)
  (h6 : remaining_area = 160)
  (h7 : bolt_width * bolt_length - (bedroom_width * bedroom_length + living_room_width * living_room_length) = remaining_area) :
  living_room_length = 6 :=
by sorry

#check living_room_curtain_length

end living_room_curtain_length_l2716_271655


namespace typists_letters_problem_l2716_271660

theorem typists_letters_problem (typists_initial : ℕ) (letters_initial : ℕ) (time_initial : ℕ) 
  (typists_final : ℕ) (time_final : ℕ) :
  typists_initial = 20 →
  letters_initial = 44 →
  time_initial = 20 →
  typists_final = 30 →
  time_final = 60 →
  (typists_final : ℚ) * (letters_initial : ℚ) * (time_final : ℚ) / 
    ((typists_initial : ℚ) * (time_initial : ℚ)) = 198 := by
  sorry

end typists_letters_problem_l2716_271660


namespace min_students_for_three_discussing_same_l2716_271624

/-- Represents a discussion between two students about a problem -/
structure Discussion where
  student1 : ℕ
  student2 : ℕ
  problem : Fin 3

/-- Represents a valid discussion configuration for n students -/
def ValidConfiguration (n : ℕ) (discussions : List Discussion) : Prop :=
  ∀ i j : Fin n, i ≠ j →
    ∃! d : Discussion, d ∈ discussions ∧
      ((d.student1 = i.val ∧ d.student2 = j.val) ∨
       (d.student1 = j.val ∧ d.student2 = i.val))

/-- Checks if there are at least 3 students discussing the same problem -/
def HasThreeDiscussingSame (n : ℕ) (discussions : List Discussion) : Prop :=
  ∃ p : Fin 3, ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (∃ d1 d2 d3 : Discussion,
      d1 ∈ discussions ∧ d2 ∈ discussions ∧ d3 ∈ discussions ∧
      d1.problem = p ∧ d2.problem = p ∧ d3.problem = p ∧
      ((d1.student1 = i.val ∧ d1.student2 = j.val) ∨ (d1.student1 = j.val ∧ d1.student2 = i.val)) ∧
      ((d2.student1 = j.val ∧ d2.student2 = k.val) ∨ (d2.student1 = k.val ∧ d2.student2 = j.val)) ∧
      ((d3.student1 = i.val ∧ d3.student2 = k.val) ∨ (d3.student1 = k.val ∧ d3.student2 = i.val)))

theorem min_students_for_three_discussing_same :
  (∃ n : ℕ, ∀ discussions : List Discussion,
    ValidConfiguration n discussions → HasThreeDiscussingSame n discussions) ∧
  (∀ m : ℕ, m < 17 →
    ∃ discussions : List Discussion,
      ValidConfiguration m discussions ∧ ¬HasThreeDiscussingSame m discussions) :=
by sorry

end min_students_for_three_discussing_same_l2716_271624


namespace remainder_divisibility_l2716_271648

theorem remainder_divisibility (N : ℤ) : 
  N % 2 = 1 → N % 35 = 1 → N % 5 = 1 := by
sorry

end remainder_divisibility_l2716_271648


namespace green_faction_liars_exceed_truthful_l2716_271666

/-- Represents the three factions in the parliament --/
inductive Faction
  | Blue
  | Red
  | Green

/-- Represents whether a deputy tells the truth or lies --/
inductive Honesty
  | Truthful
  | Liar

/-- Represents the parliament with its properties --/
structure Parliament where
  total_deputies : ℕ
  blue_affirmative : ℕ
  red_affirmative : ℕ
  green_affirmative : ℕ
  deputies : Faction → Honesty → ℕ

/-- The theorem to be proved --/
theorem green_faction_liars_exceed_truthful (p : Parliament)
  (h1 : p.total_deputies = 2016)
  (h2 : p.blue_affirmative = 1208)
  (h3 : p.red_affirmative = 908)
  (h4 : p.green_affirmative = 608)
  (h5 : p.total_deputies = p.deputies Faction.Blue Honesty.Truthful + p.deputies Faction.Blue Honesty.Liar +
                           p.deputies Faction.Red Honesty.Truthful + p.deputies Faction.Red Honesty.Liar +
                           p.deputies Faction.Green Honesty.Truthful + p.deputies Faction.Green Honesty.Liar)
  (h6 : p.blue_affirmative = p.deputies Faction.Blue Honesty.Truthful + p.deputies Faction.Red Honesty.Liar + p.deputies Faction.Green Honesty.Liar)
  (h7 : p.red_affirmative = p.deputies Faction.Red Honesty.Truthful + p.deputies Faction.Blue Honesty.Liar + p.deputies Faction.Green Honesty.Liar)
  (h8 : p.green_affirmative = p.deputies Faction.Green Honesty.Truthful + p.deputies Faction.Blue Honesty.Liar + p.deputies Faction.Red Honesty.Liar) :
  p.deputies Faction.Green Honesty.Liar = p.deputies Faction.Green Honesty.Truthful + 100 := by
  sorry


end green_faction_liars_exceed_truthful_l2716_271666


namespace sams_book_count_l2716_271687

/-- The number of books Sam bought at the school's book fair -/
def total_books (adventure_books mystery_books crime_books : ℝ) : ℝ :=
  adventure_books + mystery_books + crime_books

/-- Theorem stating the total number of books Sam bought -/
theorem sams_book_count :
  total_books 13 17 15 = 45 := by
  sorry

end sams_book_count_l2716_271687


namespace number_problem_l2716_271615

theorem number_problem (n p q : ℝ) 
  (h1 : n / p = 6)
  (h2 : n / q = 15)
  (h3 : p - q = 0.3) :
  n = 3 := by
sorry

end number_problem_l2716_271615


namespace fraction_sum_equals_one_fraction_division_simplification_l2716_271665

-- Problem 1
theorem fraction_sum_equals_one (m n : ℝ) (h : m ≠ n) :
  m / (m - n) + n / (n - m) = 1 := by sorry

-- Problem 2
theorem fraction_division_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 / (x^2 - 1)) / (1 / (x + 1)) = 2 / (x - 1) := by sorry

end fraction_sum_equals_one_fraction_division_simplification_l2716_271665


namespace min_sum_with_real_roots_l2716_271684

theorem min_sum_with_real_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : ∃ x : ℝ, x^2 + a*x + 3*b = 0)
  (h2 : ∃ x : ℝ, x^2 + 2*b*x + a = 0) :
  a + b ≥ Real.rpow 1728 (1/3) ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧
    (∃ x : ℝ, x^2 + a₀*x + 3*b₀ = 0) ∧
    (∃ x : ℝ, x^2 + 2*b₀*x + a₀ = 0) ∧
    a₀ + b₀ = Real.rpow 1728 (1/3) := by
  sorry

end min_sum_with_real_roots_l2716_271684


namespace continuous_stripe_probability_l2716_271639

/-- Represents a cube with stripes on its faces -/
structure StripedCube where
  faces : Fin 6 → Bool
  -- True represents one stripe orientation, False represents the other

/-- The probability of a continuous stripe encircling the cube -/
def probability_continuous_stripe : ℚ :=
  3 / 16

/-- Theorem stating the probability of a continuous stripe encircling the cube -/
theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 16 := by
  sorry

end continuous_stripe_probability_l2716_271639


namespace no_perfect_square_n_n_plus_one_l2716_271644

theorem no_perfect_square_n_n_plus_one : ¬∃ (n : ℕ), n > 0 ∧ ∃ (k : ℕ), n * (n + 1) = k^2 := by
  sorry

end no_perfect_square_n_n_plus_one_l2716_271644


namespace person_c_payment_l2716_271692

def personA : ℕ := 560
def personB : ℕ := 350
def personC : ℕ := 180
def totalDuty : ℕ := 100

def totalMoney : ℕ := personA + personB + personC

def proportionalPayment (money : ℕ) : ℚ :=
  (totalDuty : ℚ) * (money : ℚ) / (totalMoney : ℚ)

theorem person_c_payment :
  round (proportionalPayment personC) = 17 := by
  sorry

end person_c_payment_l2716_271692


namespace exponential_function_passes_through_point_l2716_271673

theorem exponential_function_passes_through_point
  (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 3
  f 1 = 4 := by
sorry

end exponential_function_passes_through_point_l2716_271673


namespace sales_not_notebooks_or_markers_l2716_271629

/-- The percentage of sales that are not notebooks or markers -/
def other_sales_percentage (notebook_percentage marker_percentage : ℝ) : ℝ :=
  100 - (notebook_percentage + marker_percentage)

/-- Theorem stating that the percentage of sales not consisting of notebooks or markers is 33% -/
theorem sales_not_notebooks_or_markers :
  other_sales_percentage 42 25 = 33 := by
  sorry

end sales_not_notebooks_or_markers_l2716_271629


namespace escalator_speed_l2716_271695

theorem escalator_speed (escalator_length : ℝ) (person_speed : ℝ) (time_taken : ℝ) :
  escalator_length = 180 →
  person_speed = 3 →
  time_taken = 10 →
  ∃ (escalator_speed : ℝ),
    escalator_speed = 15 ∧
    (person_speed + escalator_speed) * time_taken = escalator_length :=
by sorry

end escalator_speed_l2716_271695


namespace spice_combinations_l2716_271698

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem spice_combinations : choose 7 3 = 35 := by
  sorry

end spice_combinations_l2716_271698


namespace sphere_surface_area_circumscribing_cube_l2716_271633

theorem sphere_surface_area_circumscribing_cube (edge_length : ℝ) (sphere_radius : ℝ) :
  edge_length = 2 →
  sphere_radius = edge_length * Real.sqrt 3 / 2 →
  4 * Real.pi * sphere_radius^2 = 12 * Real.pi := by
  sorry

end sphere_surface_area_circumscribing_cube_l2716_271633


namespace absent_students_percentage_l2716_271686

theorem absent_students_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (boys_absent_fraction : ℚ) (girls_absent_fraction : ℚ) :
  total_students = 120 →
  boys = 70 →
  girls = 50 →
  boys_absent_fraction = 1 / 7 →
  girls_absent_fraction = 1 / 5 →
  (↑(boys_absent_fraction * boys + girls_absent_fraction * girls) : ℚ) / total_students = 1 / 6 := by
  sorry

end absent_students_percentage_l2716_271686


namespace fraction_multiplication_division_l2716_271600

theorem fraction_multiplication_division (a b c d e f g h : ℚ) 
  (h1 : c = 663 / 245)
  (h2 : f = 328 / 15) :
  a / b * c / d / f = g / h :=
by
  sorry

#check fraction_multiplication_division (145 : ℚ) (273 : ℚ) (663 / 245 : ℚ) (1 : ℚ) (1 : ℚ) (328 / 15 : ℚ) (7395 : ℚ) (112504 : ℚ)

end fraction_multiplication_division_l2716_271600


namespace count_ordered_pairs_l2716_271645

/-- The number of ordered pairs of positive integers (x,y) satisfying xy = 1944 -/
def num_ordered_pairs : ℕ := 24

/-- The prime factorization of 1944 -/
def prime_factorization_1944 : List (ℕ × ℕ) := [(2, 3), (3, 5)]

/-- Theorem stating that the number of ordered pairs (x,y) of positive integers
    satisfying xy = 1944 is equal to 24, given the prime factorization of 1944 -/
theorem count_ordered_pairs :
  (∀ (x y : ℕ), x * y = 1944 → x > 0 ∧ y > 0) →
  prime_factorization_1944 = [(2, 3), (3, 5)] →
  num_ordered_pairs = 24 := by
  sorry

#check count_ordered_pairs

end count_ordered_pairs_l2716_271645


namespace second_division_percentage_l2716_271601

theorem second_division_percentage (total_students : ℕ) 
  (first_division_percentage : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  first_division_percentage = 30 / 100 →
  just_passed = 48 →
  (total_students : ℚ) * first_division_percentage + 
    (just_passed : ℚ) + 
    (total_students : ℚ) * (54 / 100) = total_students :=
by
  sorry

end second_division_percentage_l2716_271601


namespace president_vice_president_election_committee_members_election_l2716_271631

-- Define the number of candidates
def num_candidates : ℕ := 4

-- Define the number of positions for the first question (president and vice president)
def num_positions_1 : ℕ := 2

-- Define the number of positions for the second question (committee members)
def num_positions_2 : ℕ := 3

-- Theorem for the first question
theorem president_vice_president_election :
  (num_candidates.choose num_positions_1) * num_positions_1.factorial = 12 := by
  sorry

-- Theorem for the second question
theorem committee_members_election :
  num_candidates.choose num_positions_2 = 4 := by
  sorry

end president_vice_president_election_committee_members_election_l2716_271631


namespace inequality_solution_l2716_271603

theorem inequality_solution (x : ℝ) : 
  (2*x - 1)/(x^2 + 2) > 5/x + 21/10 ↔ -5 < x ∧ x < 0 :=
by sorry

end inequality_solution_l2716_271603


namespace oasis_water_consumption_l2716_271604

theorem oasis_water_consumption (traveler_ounces camel_multiplier ounces_per_gallon : ℕ) 
  (h1 : traveler_ounces = 32)
  (h2 : camel_multiplier = 7)
  (h3 : ounces_per_gallon = 128) :
  (traveler_ounces + camel_multiplier * traveler_ounces) / ounces_per_gallon = 2 := by
  sorry

#check oasis_water_consumption

end oasis_water_consumption_l2716_271604


namespace nine_integer_chords_l2716_271676

/-- Represents a circle with a given radius and a point at a given distance from its center -/
structure CircleWithPoint where
  radius : ℝ
  pointDistance : ℝ

/-- Counts the number of different integer-length chords containing the given point -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem nine_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 20) 
  (h2 : c.pointDistance = 12) : 
  countIntegerChords c = 9 := by
    sorry

end nine_integer_chords_l2716_271676


namespace coin_sum_theorem_l2716_271609

/-- Represents the possible coin values in cents -/
inductive Coin : Type
  | Penny : Coin
  | Nickel : Coin
  | Dime : Coin
  | Quarter : Coin
  | HalfDollar : Coin

/-- Returns the value of a coin in cents -/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Checks if a given amount can be achieved using exactly six coins -/
def canAchieveWithSixCoins (amount : Nat) : Prop :=
  ∃ (c1 c2 c3 c4 c5 c6 : Coin), 
    coinValue c1 + coinValue c2 + coinValue c3 + coinValue c4 + coinValue c5 + coinValue c6 = amount

theorem coin_sum_theorem : 
  ¬ canAchieveWithSixCoins 62 ∧ 
  canAchieveWithSixCoins 80 ∧ 
  canAchieveWithSixCoins 90 ∧ 
  canAchieveWithSixCoins 96 := by
  sorry

end coin_sum_theorem_l2716_271609


namespace square_polynomial_l2716_271626

theorem square_polynomial (k : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + k*x + 16 = (a*x + b)^2) → (k = 8 ∨ k = -8) := by
  sorry

end square_polynomial_l2716_271626


namespace max_trucks_orchard_l2716_271625

def apples : ℕ := 170
def tangerines : ℕ := 268
def mangoes : ℕ := 120

def apples_leftover : ℕ := 8
def tangerines_short : ℕ := 2
def mangoes_leftover : ℕ := 12

theorem max_trucks_orchard : 
  let apples_distributed := apples - apples_leftover
  let tangerines_distributed := tangerines + tangerines_short
  let mangoes_distributed := mangoes - mangoes_leftover
  ∃ (n : ℕ), n > 0 ∧ 
    apples_distributed % n = 0 ∧ 
    tangerines_distributed % n = 0 ∧ 
    mangoes_distributed % n = 0 ∧
    ∀ (m : ℕ), m > n → 
      (apples_distributed % m = 0 ∧ 
       tangerines_distributed % m = 0 ∧ 
       mangoes_distributed % m = 0) → False :=
by sorry

end max_trucks_orchard_l2716_271625


namespace tommys_pencils_l2716_271619

/-- Represents the contents of Tommy's pencil case -/
structure PencilCase where
  total_items : ℕ
  num_pencils : ℕ
  num_pens : ℕ
  num_erasers : ℕ

/-- Theorem stating the number of pencils in Tommy's pencil case -/
theorem tommys_pencils (pc : PencilCase) 
  (h1 : pc.total_items = 13)
  (h2 : pc.num_pens = 2 * pc.num_pencils)
  (h3 : pc.num_erasers = 1)
  (h4 : pc.total_items = pc.num_pencils + pc.num_pens + pc.num_erasers) :
  pc.num_pencils = 4 := by
  sorry

end tommys_pencils_l2716_271619


namespace max_sum_under_constraints_l2716_271637

theorem max_sum_under_constraints :
  ∃ (M : ℝ), M = 32/17 ∧
  (∀ x y : ℝ, 5*x + 3*y ≤ 9 → 3*x + 5*y ≤ 11 → x + y ≤ M) ∧
  (∃ x y : ℝ, 5*x + 3*y ≤ 9 ∧ 3*x + 5*y ≤ 11 ∧ x + y = M) :=
by sorry

end max_sum_under_constraints_l2716_271637


namespace intersection_of_A_and_B_l2716_271630

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end intersection_of_A_and_B_l2716_271630


namespace no_real_roots_condition_l2716_271658

theorem no_real_roots_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 :=
by sorry

end no_real_roots_condition_l2716_271658


namespace function_inequality_l2716_271643

open Real

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, x * deriv f x > -f x) (a b : ℝ) (hab : a > b) : 
  a * f a > b * f b := by
  sorry

end function_inequality_l2716_271643


namespace shifted_sine_function_phi_l2716_271677

theorem shifted_sine_function_phi (θ φ : ℝ) : 
  -π/2 < θ → θ < π/2 → φ > 0 →
  (∃ f g : ℝ → ℝ, 
    (∀ x, f x = 3 * Real.sin (2 * x + θ)) ∧
    (∀ x, g x = 3 * Real.sin (2 * (x - φ) + θ)) ∧
    f 0 = 3 * Real.sqrt 2 / 2 ∧
    g 0 = 3 * Real.sqrt 2 / 2) →
  φ ≠ 5 * π / 4 :=
by sorry

end shifted_sine_function_phi_l2716_271677


namespace f_8_equals_952_l2716_271657

def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 27*x^2 - 24*x - 72

theorem f_8_equals_952 : f 8 = 952 := by
  sorry

end f_8_equals_952_l2716_271657


namespace hexagon_area_difference_l2716_271667

/-- The area between a regular hexagon with side length 8 and a smaller hexagon
    formed by joining the midpoints of its sides is 72√3. -/
theorem hexagon_area_difference : 
  let s : ℝ := 8
  let area_large := (3 * Real.sqrt 3 / 2) * s^2
  let area_small := (3 * Real.sqrt 3 / 2) * (s/2)^2
  area_large - area_small = 72 * Real.sqrt 3 := by
  sorry

end hexagon_area_difference_l2716_271667


namespace nina_unanswered_questions_l2716_271670

/-- Represents the scoring details for a math test -/
structure ScoringSystem where
  initialPoints : ℕ
  correctPoints : ℕ
  wrongPoints : ℤ
  unansweredPoints : ℕ

/-- Represents the test results -/
structure TestResult where
  totalQuestions : ℕ
  score : ℕ

theorem nina_unanswered_questions
  (oldSystem : ScoringSystem)
  (newSystem : ScoringSystem)
  (oldResult : TestResult)
  (newResult : TestResult)
  (h1 : oldSystem = {
    initialPoints := 40,
    correctPoints := 5,
    wrongPoints := -2,
    unansweredPoints := 0
  })
  (h2 : newSystem = {
    initialPoints := 0,
    correctPoints := 6,
    wrongPoints := 0,
    unansweredPoints := 3
  })
  (h3 : oldResult = {totalQuestions := 35, score := 95})
  (h4 : newResult = {totalQuestions := 35, score := 120})
  (h5 : oldResult.totalQuestions = newResult.totalQuestions) :
  ∃ (correct wrong unanswered : ℕ),
    correct + wrong + unanswered = oldResult.totalQuestions ∧
    oldSystem.initialPoints + oldSystem.correctPoints * correct + oldSystem.wrongPoints * wrong = oldResult.score ∧
    newSystem.correctPoints * correct + newSystem.unansweredPoints * unanswered = newResult.score ∧
    unanswered = 10 :=
by sorry

end nina_unanswered_questions_l2716_271670


namespace initial_players_count_video_game_players_l2716_271697

theorem initial_players_count (players_quit : ℕ) (lives_per_player : ℕ) (total_lives : ℕ) : ℕ :=
  let remaining_players := total_lives / lives_per_player
  remaining_players + players_quit

theorem video_game_players : initial_players_count 7 8 24 = 10 := by
  sorry

end initial_players_count_video_game_players_l2716_271697


namespace cos_sum_when_sin_product_one_l2716_271622

theorem cos_sum_when_sin_product_one (α β : Real) 
  (h : Real.sin α * Real.sin β = 1) : 
  Real.cos (α + β) = -1 := by
  sorry

end cos_sum_when_sin_product_one_l2716_271622


namespace constant_term_in_expansion_l2716_271669

theorem constant_term_in_expansion (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (x + a / x) * (2 * x - 1 / x)^5 = 2) →
  ∃ coeffs : List ℝ, 
    (∀ x : ℝ, x ≠ 0 → (x + a / x) * (2 * x - 1 / x)^5 = coeffs.sum) ∧
    coeffs.sum = 2 ∧
    (∃ const_term : ℝ, const_term = 40 ∧ 
      ∀ x : ℝ, x ≠ 0 → (x + a / x) * (2 * x - 1 / x)^5 = 
        const_term + x * (coeffs.sum - const_term - a / x) + 
        1 / x * (coeffs.sum - const_term - a * x)) :=
by
  sorry

end constant_term_in_expansion_l2716_271669


namespace equation_a_correct_equation_b_correct_equation_c_correct_equation_d_incorrect_l2716_271612

/-- Represents the principal amount in yuan -/
def principal : ℝ := sorry

/-- The annual interest rate -/
def interest_rate : ℝ := 0.03

/-- The total amount withdrawn after one year -/
def total_amount : ℝ := 20600

/-- Theorem stating that equation A is correct -/
theorem equation_a_correct : principal + interest_rate * principal = total_amount := by sorry

/-- Theorem stating that equation B is correct -/
theorem equation_b_correct : interest_rate * principal = total_amount - principal := by sorry

/-- Theorem stating that equation C is correct -/
theorem equation_c_correct : principal - total_amount = -(interest_rate * principal) := by sorry

/-- Theorem stating that equation D is incorrect -/
theorem equation_d_incorrect : principal + interest_rate ≠ total_amount := by sorry

end equation_a_correct_equation_b_correct_equation_c_correct_equation_d_incorrect_l2716_271612


namespace divisors_of_60_and_90_l2716_271610

theorem divisors_of_60_and_90 : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, n > 0 ∧ 60 % n = 0 ∧ 90 % n = 0) ∧ 
  (∀ n : ℕ, n > 0 → 60 % n = 0 → 90 % n = 0 → n ∈ S) ∧
  Finset.card S = 8 := by
sorry

end divisors_of_60_and_90_l2716_271610


namespace tangent_half_angle_identity_l2716_271654

theorem tangent_half_angle_identity (α : Real) (h : Real.tan (α / 2) = 2) :
  (1 + Real.cos α) / Real.sin α = 1 / 2 := by
  sorry

end tangent_half_angle_identity_l2716_271654


namespace cubic_coefficient_B_l2716_271634

/-- A cubic function with roots at -2 and 2, and value -1 at x = 0 -/
def g (A B C D : ℝ) (x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

/-- Theorem stating that under given conditions, B = 1 -/
theorem cubic_coefficient_B (A B C D : ℝ) :
  g A B C D (-2) = 0 →
  g A B C D 0 = -1 →
  g A B C D 2 = 0 →
  B = 1 := by
    sorry

end cubic_coefficient_B_l2716_271634


namespace probability_one_and_three_faces_l2716_271618

/-- Represents a cube with side length 5, assembled from unit cubes -/
def LargeCube := Fin 5 → Fin 5 → Fin 5 → Bool

/-- The number of unit cubes in the large cube -/
def totalUnitCubes : ℕ := 125

/-- The number of unit cubes with exactly one painted face -/
def oneRedFaceCubes : ℕ := 26

/-- The number of unit cubes with exactly three painted faces -/
def threeRedFaceCubes : ℕ := 4

/-- The probability of selecting one cube with one red face and one with three red faces -/
def probabilityOneAndThree : ℚ := 52 / 3875

theorem probability_one_and_three_faces (cube : LargeCube) :
  probabilityOneAndThree = (oneRedFaceCubes * threeRedFaceCubes : ℚ) / (totalUnitCubes.choose 2) :=
sorry

end probability_one_and_three_faces_l2716_271618


namespace polynomial_expansion_equality_l2716_271649

theorem polynomial_expansion_equality (x : ℝ) :
  (3*x^2 + 4*x + 8)*(x + 2) - (x + 2)*(x^2 + 5*x - 72) + (4*x - 15)*(x + 2)*(x + 6) =
  6*x^3 + 20*x^2 + 6*x - 20 := by
  sorry

end polynomial_expansion_equality_l2716_271649


namespace system_solution_equivalence_l2716_271683

-- Define the system of linear inequalities
def system (x : ℝ) : Prop := (x - 2 > 1) ∧ (x < 4)

-- Define the solution set
def solution_set : Set ℝ := {x | 3 < x ∧ x < 4}

-- Theorem statement
theorem system_solution_equivalence :
  {x : ℝ | system x} = solution_set :=
sorry

end system_solution_equivalence_l2716_271683


namespace apples_per_basket_l2716_271659

theorem apples_per_basket (total_apples : ℕ) (num_baskets : ℕ) 
  (h1 : total_apples = 495) 
  (h2 : num_baskets = 19) 
  (h3 : total_apples % num_baskets = 0) : 
  total_apples / num_baskets = 26 := by
sorry

end apples_per_basket_l2716_271659


namespace marys_height_marys_final_height_l2716_271623

theorem marys_height (initial_height : ℝ) (sallys_new_height : ℝ) : ℝ :=
  let sallys_growth_factor : ℝ := 1.2
  let sallys_growth : ℝ := sallys_new_height - initial_height
  let marys_growth : ℝ := sallys_growth / 2
  initial_height + marys_growth

theorem marys_final_height : 
  ∀ (initial_height : ℝ),
    initial_height > 0 →
    marys_height initial_height 180 = 165 :=
by
  sorry

end marys_height_marys_final_height_l2716_271623


namespace quadratic_roots_ratio_l2716_271621

/-- 
Given a quadratic equation x^2 + 8x + k = 0 with nonzero roots in the ratio 3:1,
prove that k = 12
-/
theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 8*x + k = 0 ∧ y^2 + 8*y + k = 0) → k = 12 := by
  sorry

end quadratic_roots_ratio_l2716_271621


namespace correct_quadratic_equation_l2716_271605

def quadratic_equation (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

def is_root (f : ℝ → ℝ) (r : ℝ) : Prop := f r = 0

theorem correct_quadratic_equation :
  ∃ (a b c : ℝ),
    (∃ (b₁ c₁ : ℝ), is_root (quadratic_equation a b₁ c₁) 5 ∧ is_root (quadratic_equation a b₁ c₁) 3) ∧
    (∃ (b₂ : ℝ), is_root (quadratic_equation a b₂ c) (-6) ∧ is_root (quadratic_equation a b₂ c) (-4)) ∧
    quadratic_equation a b c = quadratic_equation 1 (-8) 24 :=
by sorry

end correct_quadratic_equation_l2716_271605


namespace position_of_2000_l2716_271688

/-- Represents the column number (1 to 5) in the table -/
inductive Column
| one
| two
| three
| four
| five

/-- Represents a position in the table -/
structure Position where
  row : Nat
  column : Column

/-- Function to determine the position of a given even number in the table -/
def positionOfEvenNumber (n : Nat) : Position :=
  sorry

/-- The arrangement of positive even numbers follows the pattern described in the problem -/
axiom arrangement_pattern : ∀ n : Nat, n % 2 = 0 → n > 0 → 
  (positionOfEvenNumber n).column = Column.one ↔ n % 8 = 0

/-- Theorem stating that 2000 is in Row 250, Column 1 -/
theorem position_of_2000 : positionOfEvenNumber 2000 = { row := 250, column := Column.one } :=
  sorry

end position_of_2000_l2716_271688


namespace intersection_area_theorem_l2716_271641

/-- A rectangle in the 2D plane -/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- A circle in the 2D plane -/
structure Circle where
  center_x : ℝ
  center_y : ℝ
  radius : ℝ

/-- The area of intersection between a rectangle and a circle -/
def intersectionArea (rect : Rectangle) (circ : Circle) : ℝ := sorry

/-- The theorem stating the area of intersection between the specific rectangle and circle -/
theorem intersection_area_theorem :
  let rect : Rectangle := { x1 := 3, y1 := -3, x2 := 14, y2 := 10 }
  let circ : Circle := { center_x := 3, center_y := -3, radius := 4 }
  intersectionArea rect circ = 4 * Real.pi := by sorry

end intersection_area_theorem_l2716_271641


namespace sum_of_cubes_zero_l2716_271650

theorem sum_of_cubes_zero (a b : ℝ) (h1 : a + b = 0) (h2 : a * b = -4) : 
  a^3 + b^3 = 0 := by
sorry

end sum_of_cubes_zero_l2716_271650


namespace paving_cost_l2716_271613

/-- The cost of paving a rectangular floor given its dimensions and the rate per square metre. -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 8) (h2 : width = 4.75) (h3 : rate = 900) :
  length * width * rate = 34200 := by
  sorry

end paving_cost_l2716_271613


namespace average_weight_problem_l2716_271685

theorem average_weight_problem (A B C : ℝ) : 
  (A + B) / 2 = 40 →
  (B + C) / 2 = 43 →
  B = 31 →
  (A + B + C) / 3 = 45 := by
sorry

end average_weight_problem_l2716_271685


namespace kombucha_half_fill_time_l2716_271672

/-- Represents the area of kombucha in the jar as a fraction of the full jar -/
def kombucha_area (days : ℕ) : ℚ :=
  1 / 2^(19 - days)

theorem kombucha_half_fill_time : 
  (∀ d : ℕ, d < 19 → kombucha_area (d + 1) = 2 * kombucha_area d) →
  kombucha_area 19 = 1 →
  kombucha_area 18 = 1/2 :=
by
  sorry

end kombucha_half_fill_time_l2716_271672


namespace last_digit_is_zero_l2716_271640

def number (last_digit : Nat) : Nat :=
  626840 + last_digit

theorem last_digit_is_zero :
  ∀ d : Nat, d < 10 →
  (number d % 8 = 0 ∧ number d % 5 = 0) →
  d = 0 := by
sorry

end last_digit_is_zero_l2716_271640


namespace linear_program_unbounded_l2716_271699

def objective_function (x₁ x₂ x₃ x₄ : ℝ) : ℝ := x₁ - x₂ + 2*x₃ - x₄

def constraint1 (x₁ x₂ : ℝ) : Prop := x₁ + x₂ = 1
def constraint2 (x₂ x₃ x₄ : ℝ) : Prop := x₂ + x₃ - x₄ = 1
def non_negative (x : ℝ) : Prop := x ≥ 0

theorem linear_program_unbounded :
  ∀ M : ℝ, ∃ x₁ x₂ x₃ x₄ : ℝ,
    constraint1 x₁ x₂ ∧
    constraint2 x₂ x₃ x₄ ∧
    non_negative x₁ ∧
    non_negative x₂ ∧
    non_negative x₃ ∧
    non_negative x₄ ∧
    objective_function x₁ x₂ x₃ x₄ > M :=
by
  sorry


end linear_program_unbounded_l2716_271699


namespace right_triangle_inradius_l2716_271681

/-- The inradius of a right triangle with side lengths 5, 12, and 13 is 2. -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 →
  r = (a * b) / (2 * (a + b + c)) →
  r = 2 := by sorry

end right_triangle_inradius_l2716_271681


namespace ellipse_theorem_l2716_271636

/-- Definition of the ellipse C -/
def ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the major axis length -/
def majorAxisLength (a : ℝ) : Prop :=
  2 * a = 2 * Real.sqrt 2

/-- Definition of the range for point N's x-coordinate -/
def NxRange (x : ℝ) : Prop :=
  -1/4 < x ∧ x < 0

/-- Main theorem -/
theorem ellipse_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : majorAxisLength a) :
  (∀ x y : ℝ, ellipse x y a b ↔ x^2 / 2 + y^2 = 1) ∧
  (∀ N A B : ℝ × ℝ,
    NxRange N.1 →
    (∃ k : ℝ, 
      ellipse A.1 A.2 (Real.sqrt 2) 1 ∧
      ellipse B.1 B.2 (Real.sqrt 2) 1 ∧
      A.2 = k * (A.1 + 1) ∧
      B.2 = k * (B.1 + 1)) →
    3 * Real.sqrt 2 / 2 < Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) < 2 * Real.sqrt 2) :=
sorry

end ellipse_theorem_l2716_271636


namespace min_value_theorem_l2716_271674

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (a + 1) + 4 / (b + 1) ≥ 9 / 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / (a₀ + 1) + 4 / (b₀ + 1) = 9 / 4 :=
by sorry

end min_value_theorem_l2716_271674


namespace percentage_of_number_seventy_six_point_five_percent_of_1287_l2716_271620

theorem percentage_of_number (x : ℝ) (y : ℝ) (z : ℝ) (h : z = x * (y / 100)) :
  z = x * (y / 100) := by
  sorry

theorem seventy_six_point_five_percent_of_1287 :
  (76.5 / 100) * 1287 = 984.495 := by
  sorry

end percentage_of_number_seventy_six_point_five_percent_of_1287_l2716_271620


namespace coefficient_x_squared_in_expansion_coefficient_x_squared_in_expansion_proof_l2716_271662

/-- The coefficient of x^2 in the expansion of (x + 2/x^2)^5 is 10 -/
theorem coefficient_x_squared_in_expansion : ℕ :=
  let expansion := (fun x => (x + 2 / x^2)^5)
  let coefficient_x_squared := 10
  coefficient_x_squared

/-- Proof of the theorem -/
theorem coefficient_x_squared_in_expansion_proof :
  coefficient_x_squared_in_expansion = 10 := by
  sorry

end coefficient_x_squared_in_expansion_coefficient_x_squared_in_expansion_proof_l2716_271662


namespace smallest_square_coverage_l2716_271664

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Represents a square with integer side length -/
structure Square where
  side : ℕ

/-- The number of rectangles needed to cover a square -/
def rectangles_needed (r : Rectangle) (s : Square) : ℕ :=
  (s.side * s.side) / (r.length * r.width)

/-- Checks if a square can be exactly covered by rectangles -/
def is_exactly_coverable (r : Rectangle) (s : Square) : Prop :=
  (s.side * s.side) % (r.length * r.width) = 0

theorem smallest_square_coverage (r : Rectangle) (s : Square) : 
  r.length = 3 ∧ r.width = 2 ∧ 
  s.side = 12 ∧
  is_exactly_coverable r s ∧
  rectangles_needed r s = 24 ∧
  (∀ s' : Square, s'.side < s.side → ¬(is_exactly_coverable r s')) := by
  sorry

end smallest_square_coverage_l2716_271664


namespace polynomial_evaluation_l2716_271696

theorem polynomial_evaluation : 
  let a : ℤ := 2999
  let b : ℤ := 3000
  b^3 - a*b^2 - a^2*b + a^3 = b + a := by sorry

end polynomial_evaluation_l2716_271696


namespace cube_root_fourteen_problem_l2716_271675

theorem cube_root_fourteen_problem (x y z : ℝ) 
  (eq1 : (x + y) / (1 + z) = (1 - z + z^2) / (x^2 - x*y + y^2))
  (eq2 : (x - y) / (3 - z) = (9 + 3*z + z^2) / (x^2 + x*y + y^2)) :
  x = (14 : ℝ)^(1/3) := by
  sorry

end cube_root_fourteen_problem_l2716_271675


namespace complex_expression_evaluation_l2716_271668

theorem complex_expression_evaluation :
  (1 : ℝ) * (0.25 ^ (1/2 : ℝ)) - 
  (-2 * ((3/7 : ℝ) ^ (0 : ℝ))) ^ 2 * 
  ((-2 : ℝ) ^ 3) ^ (4/3 : ℝ) + 
  ((2 : ℝ) ^ (1/2 : ℝ) - 1) ^ (-1 : ℝ) - 
  (2 : ℝ) ^ (1/2 : ℝ) = -125/2 := by
  sorry

end complex_expression_evaluation_l2716_271668


namespace range_of_quadratic_expression_l2716_271608

theorem range_of_quadratic_expression (x : ℝ) :
  ((x - 1) * (x - 2) < 2) →
  ∃ y, y = (x + 1) * (x - 3) ∧ -4 ≤ y ∧ y < 0 :=
by sorry

end range_of_quadratic_expression_l2716_271608


namespace sqrt_product_simplification_l2716_271611

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by sorry

end sqrt_product_simplification_l2716_271611


namespace two_std_dev_below_mean_l2716_271691

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ

/-- Calculates the value that is a given number of standard deviations away from the mean --/
def value_at_std_devs (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 15 and standard deviation 1.5,
    the value 2 standard deviations below the mean is 12 --/
theorem two_std_dev_below_mean :
  let d : NormalDistribution := { mean := 15, std_dev := 1.5 }
  value_at_std_devs d 2 = 12 := by
  sorry

end two_std_dev_below_mean_l2716_271691


namespace magnitude_2a_equals_6_l2716_271656

def a : Fin 3 → ℝ := ![-1, 2, 2]

theorem magnitude_2a_equals_6 : ‖(2 : ℝ) • a‖ = 6 := by sorry

end magnitude_2a_equals_6_l2716_271656


namespace expression_undefined_l2716_271635

theorem expression_undefined (θ : ℝ) (h1 : θ > 0) (h2 : θ + 90 = 180) : 
  ¬∃x : ℝ, x = (Real.sin θ + Real.sin (2*θ) + Real.sin (3*θ) + Real.sin (4*θ)) / 
            (Real.cos (θ/2) * Real.cos θ * Real.cos (2*θ)) := by
  sorry

end expression_undefined_l2716_271635


namespace max_d_value_l2716_271638

theorem max_d_value : 
  let f : ℝ → ℝ := λ d => (5 + Real.sqrt 244) / 3 - d
  ∃ d : ℝ, (4 * Real.sqrt 3) ^ 2 + (d + 5) ^ 2 = (2 * d) ^ 2 ∧ 
    (∀ x : ℝ, (4 * Real.sqrt 3) ^ 2 + (x + 5) ^ 2 = (2 * x) ^ 2 → f x ≥ 0) :=
by sorry

end max_d_value_l2716_271638


namespace tan_75_deg_l2716_271647

/-- Tangent of angle addition formula -/
axiom tan_add (a b : ℝ) : Real.tan (a + b) = (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)

/-- Proof that tan 75° = 2 + √3 -/
theorem tan_75_deg : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  have h1 : 75 * π / 180 = 60 * π / 180 + 15 * π / 180 := by sorry
  have h2 : Real.tan (60 * π / 180) = Real.sqrt 3 := by sorry
  have h3 : Real.tan (15 * π / 180) = 2 - Real.sqrt 3 := by sorry
  sorry


end tan_75_deg_l2716_271647


namespace quadratic_minimum_l2716_271628

theorem quadratic_minimum (a b : ℝ) (x₀ : ℝ) (h : a > 0) :
  (a * x₀ = b) ↔ ∀ x : ℝ, (1/2) * a * x^2 - b * x ≥ (1/2) * a * x₀^2 - b * x₀ :=
by sorry

end quadratic_minimum_l2716_271628


namespace squash_league_max_salary_l2716_271614

/-- Represents the maximum salary a player can earn in a professional squash league --/
def max_salary (team_size : ℕ) (min_salary : ℕ) (total_payroll : ℕ) : ℕ :=
  total_payroll - (team_size - 1) * min_salary

/-- Theorem stating the maximum salary in the given conditions --/
theorem squash_league_max_salary :
  max_salary 22 16000 880000 = 544000 := by
  sorry

end squash_league_max_salary_l2716_271614


namespace cricket_bat_theorem_l2716_271653

def cricket_bat_problem (a_cost_price b_selling_price c_purchase_price : ℝ) 
  (a_profit_percentage : ℝ) : Prop :=
  let a_selling_price := a_cost_price * (1 + a_profit_percentage)
  let b_profit := c_purchase_price - a_selling_price
  let b_profit_percentage := b_profit / a_selling_price * 100
  a_cost_price = 156 ∧ 
  a_profit_percentage = 0.20 ∧ 
  c_purchase_price = 234 → 
  b_profit_percentage = 25

theorem cricket_bat_theorem : 
  ∃ (a_cost_price b_selling_price c_purchase_price : ℝ),
    cricket_bat_problem a_cost_price b_selling_price c_purchase_price 0.20 :=
by
  sorry

end cricket_bat_theorem_l2716_271653


namespace five_trip_ticket_cost_l2716_271663

/-- Represents the cost of tickets in gold coins -/
structure TicketCost where
  one : ℕ
  five : ℕ
  twenty : ℕ

/-- Conditions for the ticket costs -/
def valid_ticket_cost (t : TicketCost) : Prop :=
  5 * t.one > t.five ∧ 
  4 * t.five > t.twenty ∧
  t.twenty + 3 * t.five = 33 ∧
  20 + 3 * 5 = 35

theorem five_trip_ticket_cost (t : TicketCost) (h : valid_ticket_cost t) : t.five = 5 :=
sorry

end five_trip_ticket_cost_l2716_271663


namespace max_a_value_l2716_271679

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- State the theorem
theorem max_a_value (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 2, f a x ≤ 6) →
  a ≤ -1 ∧ ∃ x ∈ Set.Ioo 0 2, f (-1) x = 6 :=
by sorry

end max_a_value_l2716_271679


namespace red_cells_count_l2716_271616

/-- Represents the dimensions of the grid -/
structure GridDim where
  rows : Nat
  cols : Nat

/-- Represents the painter's movement -/
structure Movement where
  left : Nat
  down : Nat

/-- Calculates the number of distinct cells visited before returning to the start -/
def distinctCellsVisited (dim : GridDim) (move : Movement) : Nat :=
  Nat.lcm dim.rows dim.cols

/-- The main theorem stating the number of red cells on the grid -/
theorem red_cells_count (dim : GridDim) (move : Movement) 
  (h1 : dim.rows = 2000) 
  (h2 : dim.cols = 70) 
  (h3 : move.left = 1) 
  (h4 : move.down = 1) : 
  distinctCellsVisited dim move = 14000 := by
  sorry

#eval distinctCellsVisited ⟨2000, 70⟩ ⟨1, 1⟩

end red_cells_count_l2716_271616


namespace same_color_probability_l2716_271682

/-- The probability of drawing two balls of the same color from a bag containing green and white balls. -/
theorem same_color_probability (green white : ℕ) (h : green = 9 ∧ white = 8) :
  let total := green + white
  let p_green := green * (green - 1) / (total * (total - 1))
  let p_white := white * (white - 1) / (total * (total - 1))
  p_green + p_white = 8 / 17 := by
  sorry

#check same_color_probability

end same_color_probability_l2716_271682


namespace parabola_hyperbola_equations_l2716_271678

/-- Represents a parabola with vertex at the origin -/
structure Parabola where
  c : ℝ
  eq : ℝ × ℝ → Prop := fun (x, y) ↦ y^2 = 4 * c * x

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  eq : ℝ × ℝ → Prop := fun (x, y) ↦ x^2 / a^2 - y^2 / b^2 = 1

/-- The main theorem -/
theorem parabola_hyperbola_equations 
  (p : Parabola) 
  (h : Hyperbola) 
  (h_a_pos : h.a > 0)
  (h_b_pos : h.b > 0)
  (directrix_passes_focus : ∃ (f : ℝ × ℝ), h.eq f ∧ p.c = 2 * h.a)
  (intersection_point : p.eq (3/2, Real.sqrt 6) ∧ h.eq (3/2, Real.sqrt 6)) :
  p.eq = fun (x, y) ↦ y^2 = 4 * x ∧ 
  h.eq = fun (x, y) ↦ 4 * x^2 - 4 * y^2 / 3 = 1 := by
  sorry


end parabola_hyperbola_equations_l2716_271678


namespace equation_proof_l2716_271646

theorem equation_proof : (8 - 2) + 5 * (3 - 2) = 11 := by
  sorry

end equation_proof_l2716_271646


namespace weight_of_new_person_new_person_weight_l2716_271690

theorem weight_of_new_person (initial_count : ℕ) (average_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  let new_weight := replaced_weight + initial_count * average_increase
  new_weight

/-- Given a group of 8 people where one person weighing 65 kg is replaced by a new person, 
    and the average weight of the group increases by 3 kg, 
    the weight of the new person is 89 kg. -/
theorem new_person_weight : weight_of_new_person 8 3 65 = 89 := by
  sorry

end weight_of_new_person_new_person_weight_l2716_271690


namespace reinforcement_arrival_day_l2716_271632

/-- Calculates the number of days passed before reinforcement arrived -/
def days_before_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
  (reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  ((initial_garrison * initial_duration) - 
   ((initial_garrison + reinforcement) * remaining_duration)) / initial_garrison

/-- Theorem stating the number of days passed before reinforcement arrived -/
theorem reinforcement_arrival_day : 
  days_before_reinforcement 2000 54 1600 20 = 18 := by
  sorry

end reinforcement_arrival_day_l2716_271632


namespace percentage_calculation_l2716_271651

theorem percentage_calculation (whole : ℝ) (part : ℝ) (h1 : whole = 200) (h2 : part = 50) :
  (part / whole) * 100 = 25 := by
  sorry

end percentage_calculation_l2716_271651


namespace stationary_tank_radius_l2716_271694

theorem stationary_tank_radius 
  (h : Real) 
  (r : Real) 
  (h_truck : Real) 
  (r_truck : Real) 
  (h_drop : Real) :
  h = 25 → 
  r_truck = 4 → 
  h_truck = 10 → 
  h_drop = 0.016 → 
  π * r^2 * h_drop = π * r_truck^2 * h_truck → 
  r = 100 := by
sorry

end stationary_tank_radius_l2716_271694


namespace colors_in_box_l2716_271661

/-- The number of color boxes -/
def num_boxes : ℕ := 3

/-- The total number of pencils -/
def total_pencils : ℕ := 21

/-- The number of colors in each box -/
def colors_per_box : ℕ := total_pencils / num_boxes

/-- Theorem stating that the number of colors in each box is 7 -/
theorem colors_in_box : colors_per_box = 7 := by
  sorry

end colors_in_box_l2716_271661


namespace cubic_equation_solutions_l2716_271606

theorem cubic_equation_solutions :
  ∀ (x y z n : ℕ), x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 →
  ((x = 1 ∧ y = 1 ∧ z = 1 ∧ n = 3) ∨
   (x = 1 ∧ y = 2 ∧ z = 3 ∧ n = 1) ∨
   (x = 1 ∧ y = 3 ∧ z = 2 ∧ n = 1) ∨
   (x = 2 ∧ y = 1 ∧ z = 3 ∧ n = 1) ∨
   (x = 2 ∧ y = 3 ∧ z = 1 ∧ n = 1) ∨
   (x = 3 ∧ y = 1 ∧ z = 2 ∧ n = 1) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ n = 1)) :=
by sorry

end cubic_equation_solutions_l2716_271606


namespace problem_solution_l2716_271602

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x * log x + a * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := |exp x - a| + a^2 / 2

theorem problem_solution (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 (exp 1), Monotone (f a)) →
  (∃ M m : ℝ, (∀ x ∈ Set.Icc 0 (log 3), m ≤ g a x ∧ g a x ≤ M) ∧ M - m = 3/2) →
  a = 5/2 := by sorry

end problem_solution_l2716_271602


namespace intersection_point_l2716_271680

-- Define the rectangle ABCD
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (0, 4)
def C : ℝ × ℝ := (6, 4)
def D : ℝ × ℝ := (6, 0)

-- Define the lines from A and B
def lineA1 (x : ℝ) : ℝ := x -- y = x (45° line from A)
def lineA2 (x : ℝ) : ℝ := -x -- y = -x (135° line from A)
def lineB1 (x : ℝ) : ℝ := 4 - x -- y = 4 - x (-45° line from B)
def lineB2 (x : ℝ) : ℝ := 4 + x -- y = 4 + x (-135° line from B)

-- Theorem statement
theorem intersection_point : 
  ∃! p : ℝ × ℝ, 
    (lineA1 p.1 = p.2 ∧ lineB1 p.1 = p.2) ∧ 
    (lineA2 p.1 = p.2 ∧ lineB2 p.1 = p.2) ∧
    p = (2, 2) := by
  sorry

end intersection_point_l2716_271680


namespace vector_operations_and_parallelism_l2716_271617

/-- Given three vectors in R², prove the results of vector operations and parallelism condition. -/
theorem vector_operations_and_parallelism 
  (a b c : ℝ × ℝ) 
  (ha : a = (3, 2)) 
  (hb : b = (-1, 2)) 
  (hc : c = (4, 1)) : 
  (3 • a + b - 2 • c = (0, 6)) ∧ 
  (∃ k : ℝ, k = -16/13 ∧ ∃ t : ℝ, t • (a + k • c) = 2 • b - a) := by
sorry


end vector_operations_and_parallelism_l2716_271617


namespace cube_sum_inequality_l2716_271671

theorem cube_sum_inequality (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1)
  (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 ≤ 4 ∧ 
  (a + b + c + a^2 + b^2 + c^2 = 4 ↔ 
    ((a = 1 ∧ b = 1 ∧ c = -1) ∨
     (a = 1 ∧ b = -1 ∧ c = 1) ∨
     (a = -1 ∧ b = 1 ∧ c = 1))) :=
by sorry

end cube_sum_inequality_l2716_271671


namespace fraction_sum_inequality_l2716_271693

theorem fraction_sum_inequality (a b : ℝ) (h : a * b < 0) :
  a / b + b / a ≤ -2 := by sorry

end fraction_sum_inequality_l2716_271693


namespace drink_cost_is_2_50_l2716_271642

/-- The cost of a meal and drink with tip, given the following conditions:
  * The meal costs $10
  * The tip is 20% of the total cost (meal + drink)
  * The total amount paid is $15 -/
def total_cost (drink_cost : ℝ) : ℝ :=
  10 + drink_cost + 0.2 * (10 + drink_cost)

/-- Proves that the cost of the drink is $2.50 given the conditions -/
theorem drink_cost_is_2_50 :
  ∃ (drink_cost : ℝ), total_cost drink_cost = 15 ∧ drink_cost = 2.5 := by
sorry

end drink_cost_is_2_50_l2716_271642


namespace tenth_grade_enrollment_l2716_271607

/-- Represents the number of students enrolled only in science class -/
def students_only_science (total_students science_students art_students : ℕ) : ℕ :=
  science_students - (science_students + art_students - total_students)

/-- Theorem stating that given the conditions, 65 students are enrolled only in science class -/
theorem tenth_grade_enrollment (total_students science_students art_students : ℕ) 
  (h1 : total_students = 140)
  (h2 : science_students = 100)
  (h3 : art_students = 75) :
  students_only_science total_students science_students art_students = 65 := by
  sorry

#eval students_only_science 140 100 75

end tenth_grade_enrollment_l2716_271607


namespace cylinder_radius_problem_l2716_271689

theorem cylinder_radius_problem (rounds1 rounds2 : ℕ) (radius2 : ℝ) (radius1 : ℝ) :
  rounds1 = 70 →
  rounds2 = 49 →
  radius2 = 20 →
  rounds1 * (2 * Real.pi * radius1) = rounds2 * (2 * Real.pi * radius2) →
  radius1 = 14 := by
sorry

end cylinder_radius_problem_l2716_271689
