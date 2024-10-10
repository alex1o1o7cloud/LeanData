import Mathlib

namespace hexagon_area_is_19444_l287_28733

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (positive_a : a > 0)
  (positive_b : b > 0)
  (positive_c : c > 0)
  (triangle_inequality_ab : a + b > c)
  (triangle_inequality_bc : b + c > a)
  (triangle_inequality_ca : c + a > b)

-- Define the specific triangle with sides 13, 14, and 15
def specific_triangle : Triangle :=
  { a := 13
  , b := 14
  , c := 15
  , positive_a := by norm_num
  , positive_b := by norm_num
  , positive_c := by norm_num
  , triangle_inequality_ab := by norm_num
  , triangle_inequality_bc := by norm_num
  , triangle_inequality_ca := by norm_num }

-- Define the area of the hexagon A₅A₆B₅B₆C₅C₆
def hexagon_area (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem hexagon_area_is_19444 :
  hexagon_area specific_triangle = 19444 := by sorry

end hexagon_area_is_19444_l287_28733


namespace root_quadruples_l287_28713

theorem root_quadruples : ∀ a b c d : ℝ,
  (a ≠ b ∧ 
   2 * a^2 - 3 * c * a + 8 * d = 0 ∧
   2 * b^2 - 3 * c * b + 8 * d = 0 ∧
   c ≠ d ∧
   2 * c^2 - 3 * a * c + 8 * b = 0 ∧
   2 * d^2 - 3 * a * d + 8 * b = 0) →
  ((a = 4 ∧ b = 8 ∧ c = 4 ∧ d = 8) ∨
   (a = -2 ∧ b = -22 ∧ c = -8 ∧ d = 11) ∨
   (a = -8 ∧ b = 2 ∧ c = -2 ∧ d = -4)) :=
by sorry

end root_quadruples_l287_28713


namespace impossibleToAchieveTwoHundreds_l287_28792

/-- Represents the score changes that can be applied to exam scores. -/
inductive ScoreChange
  | AddOneToAll
  | DecreaseOneIncreaseTwo

/-- Represents the scores for three exams. -/
structure ExamScores where
  russian : ℕ
  physics : ℕ
  mathematics : ℕ

/-- Applies a score change to the exam scores. -/
def applyScoreChange (scores : ExamScores) (change : ScoreChange) : ExamScores :=
  match change with
  | ScoreChange.AddOneToAll =>
      { russian := scores.russian + 1,
        physics := scores.physics + 1,
        mathematics := scores.mathematics + 1 }
  | ScoreChange.DecreaseOneIncreaseTwo =>
      { russian := scores.russian - 3,
        physics := scores.physics + 1,
        mathematics := scores.mathematics + 1 }

/-- Checks if at least two scores are equal to 100. -/
def atLeastTwoEqual100 (scores : ExamScores) : Prop :=
  (scores.russian = 100 ∧ scores.physics = 100) ∨
  (scores.russian = 100 ∧ scores.mathematics = 100) ∨
  (scores.physics = 100 ∧ scores.mathematics = 100)

/-- Theorem stating the impossibility of achieving at least two scores of 100. -/
theorem impossibleToAchieveTwoHundreds (initialScores : ExamScores)
  (hRussian : initialScores.russian = initialScores.physics - 5)
  (hPhysics : initialScores.physics = initialScores.mathematics - 9)
  (hMaxScore : ∀ scores : ExamScores, scores.russian ≤ 100 ∧ scores.physics ≤ 100 ∧ scores.mathematics ≤ 100) :
  ¬∃ (changes : List ScoreChange), atLeastTwoEqual100 (changes.foldl applyScoreChange initialScores) :=
sorry

end impossibleToAchieveTwoHundreds_l287_28792


namespace world_cup_teams_l287_28706

theorem world_cup_teams (total_gifts : ℕ) (gifts_per_team : ℕ) : 
  total_gifts = 14 → 
  gifts_per_team = 2 → 
  total_gifts / gifts_per_team = 7 :=
by
  sorry

end world_cup_teams_l287_28706


namespace sum_of_roots_l287_28791

theorem sum_of_roots (α β : ℝ) : 
  α^3 - 3*α^2 + 5*α - 4 = 0 → 
  β^3 - 3*β^2 + 5*β - 2 = 0 → 
  α + β = 2 := by
sorry

end sum_of_roots_l287_28791


namespace quadratic_coefficient_l287_28795

theorem quadratic_coefficient (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ = (1 + Real.sqrt 3) / 2 ∧ 
                 x₂ = (1 - Real.sqrt 3) / 2 ∧ 
                 a * x₁^2 - x₁ - 1/2 = 0 ∧ 
                 a * x₂^2 - x₂ - 1/2 = 0) → 
  a = 1 := by
sorry

end quadratic_coefficient_l287_28795


namespace negative_inequality_l287_28796

theorem negative_inequality (a b : ℝ) (h : a < b) : -2 * a > -2 * b := by
  sorry

end negative_inequality_l287_28796


namespace min_value_theorem_l287_28799

theorem min_value_theorem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1/2) :
  (4/a + 1/b) ≥ 18 := by sorry

end min_value_theorem_l287_28799


namespace only_B_forms_grid_l287_28719

/-- Represents a shape that can be used in the puzzle game -/
inductive Shape
  | A
  | B
  | C

/-- Represents a 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a shape can form a complete 4x4 grid without gaps or overlaps -/
def canFormGrid (s : Shape) : Prop :=
  ∃ (g : Grid), ∀ (i j : Fin 4), g i j = true

/-- Theorem stating that only shape B can form a complete 4x4 grid -/
theorem only_B_forms_grid :
  (canFormGrid Shape.B) ∧ 
  (¬ canFormGrid Shape.A) ∧ 
  (¬ canFormGrid Shape.C) :=
sorry

end only_B_forms_grid_l287_28719


namespace arithmetic_sequence_sum_l287_28726

theorem arithmetic_sequence_sum : 
  ∀ (a₁ : ℤ) (aₙ : ℤ) (d : ℤ) (n : ℕ),
    a₁ = -25 →
    aₙ = 19 →
    d = 4 →
    aₙ = a₁ + (n - 1) * d →
    (n : ℤ) * (a₁ + aₙ) / 2 = -36 :=
by
  sorry

end arithmetic_sequence_sum_l287_28726


namespace five_nine_difference_l287_28739

/-- Count of a specific digit in page numbers from 1 to n -/
def digitCount (digit : Nat) (n : Nat) : Nat :=
  sorry

/-- The difference between the count of 5's and 9's in page numbers from 1 to 600 -/
theorem five_nine_difference : digitCount 5 600 - digitCount 9 600 = 100 := by
  sorry

end five_nine_difference_l287_28739


namespace ratio_of_roots_quadratic_l287_28781

theorem ratio_of_roots_quadratic (p : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    x₁^2 + p*x₁ - 16 = 0 ∧ 
    x₂^2 + p*x₂ - 16 = 0 ∧ 
    x₁/x₂ = -4) → 
  p = 6 ∨ p = -6 := by
sorry

end ratio_of_roots_quadratic_l287_28781


namespace circle_definition_l287_28741

/-- Definition of a circle in a plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- Theorem: The set of all points in a plane at a fixed distance from a given point forms a circle -/
theorem circle_definition (center : ℝ × ℝ) (radius : ℝ) :
  {p : ℝ × ℝ | Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) = radius} = Circle center radius :=
by sorry

end circle_definition_l287_28741


namespace book_arrangements_eq_103680_l287_28753

/-- The number of ways to arrange 11 books (3 Arabic, 2 German, 4 Spanish, and 2 French) on a shelf,
    keeping the Arabic books together and the Spanish books together. -/
def book_arrangements : ℕ :=
  let total_books : ℕ := 11
  let arabic_books : ℕ := 3
  let german_books : ℕ := 2
  let spanish_books : ℕ := 4
  let french_books : ℕ := 2
  let grouped_units : ℕ := 1 + 1 + german_books + french_books  -- Arabic and Spanish groups + individual German and French books
  (Nat.factorial grouped_units) * (Nat.factorial arabic_books) * (Nat.factorial spanish_books)

theorem book_arrangements_eq_103680 : book_arrangements = 103680 := by
  sorry

end book_arrangements_eq_103680_l287_28753


namespace determinant_zero_l287_28707

theorem determinant_zero (α β : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![0, Real.cos α, -Real.sin α],
                                        ![-Real.cos α, 0, Real.cos β],
                                        ![Real.sin α, -Real.cos β, 0]]
  Matrix.det M = 0 := by
sorry

end determinant_zero_l287_28707


namespace good_2013_implies_good_20_l287_28735

/-- A sequence of positive integers is non-decreasing -/
def IsNonDecreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n ≤ a (n + 1)

/-- A number is good if it can be expressed as i/a_i for some index i -/
def IsGood (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∃ i : ℕ, n = i / a i

theorem good_2013_implies_good_20 (a : ℕ → ℕ) 
  (h_nondec : IsNonDecreasingSeq a) 
  (h_2013 : IsGood 2013 a) : 
  IsGood 20 a :=
sorry

end good_2013_implies_good_20_l287_28735


namespace isosceles_triangle_l287_28759

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem isosceles_triangle (t : Triangle) 
  (h : 2 * Real.cos t.A * Real.cos t.B = 1 - Real.cos t.C) : 
  t.A = t.B := by
  sorry

end isosceles_triangle_l287_28759


namespace milk_calculation_l287_28710

/-- The initial amount of milk in quarts -/
def initial_milk : ℝ := 1000

/-- The percentage of butterfat in the initial milk -/
def initial_butterfat_percent : ℝ := 4

/-- The percentage of butterfat in the final milk -/
def final_butterfat_percent : ℝ := 3

/-- The amount of cream separated in quarts -/
def separated_cream : ℝ := 50

/-- The percentage of butterfat in the separated cream -/
def cream_butterfat_percent : ℝ := 23

theorem milk_calculation :
  initial_milk = 1000 ∧
  initial_butterfat_percent / 100 * initial_milk =
    final_butterfat_percent / 100 * (initial_milk - separated_cream) +
    cream_butterfat_percent / 100 * separated_cream :=
by sorry

end milk_calculation_l287_28710


namespace magical_stack_with_151_fixed_l287_28765

/-- A stack of cards is magical if at least one card from each pile retains its original position after restacking -/
def is_magical (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≤ n ∧ b > n ∧ b ≤ 2*n ∧
  (a = 2*a - 1 ∨ b = 2*(b - n))

theorem magical_stack_with_151_fixed (n : ℕ) 
  (h_magical : is_magical n) 
  (h_151_fixed : 151 ≤ n ∧ 151 = 2*151 - 1) : 
  n = 226 ∧ 2*n = 452 := by sorry

end magical_stack_with_151_fixed_l287_28765


namespace quadratic_equation_solution_l287_28732

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 2 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end quadratic_equation_solution_l287_28732


namespace min_value_theorem_l287_28768

theorem min_value_theorem (C : ℝ) (x : ℝ) (h1 : C > 0) (h2 : x^3 - 1/x^3 = C) :
  C^2 + 9 ≥ 6 * C ∧ ∃ (C₀ : ℝ) (x₀ : ℝ), C₀ > 0 ∧ x₀^3 - 1/x₀^3 = C₀ ∧ C₀^2 + 9 = 6 * C₀ :=
by sorry

end min_value_theorem_l287_28768


namespace cube_volume_from_face_area_l287_28754

theorem cube_volume_from_face_area (face_area : ℝ) (volume : ℝ) :
  face_area = 16 →
  volume = face_area ^ (3/2) →
  volume = 64 := by
  sorry

end cube_volume_from_face_area_l287_28754


namespace quadratic_root_property_l287_28788

theorem quadratic_root_property (a : ℝ) (h : a^2 - 2*a - 3 = 0) : a^2 - 2*a + 1 = 4 := by
  sorry

end quadratic_root_property_l287_28788


namespace dodecagon_diagonals_l287_28785

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon is a 12-sided polygon -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l287_28785


namespace fourth_root_simplification_l287_28734

theorem fourth_root_simplification :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ 
  (2^8 * 3^5)^(1/4 : ℝ) = a * (b : ℝ)^(1/4 : ℝ) ∧
  a + b = 15 :=
by sorry

end fourth_root_simplification_l287_28734


namespace fifth_term_sequence_l287_28715

theorem fifth_term_sequence (n : ℕ) : 
  let a : ℕ → ℕ := λ k => k * (k + 1) / 2
  a 5 = 15 := by
sorry

end fifth_term_sequence_l287_28715


namespace comparison_inequalities_l287_28747

theorem comparison_inequalities :
  (Real.sqrt 37 > 6) ∧ ((Real.sqrt 5 - 1) / 2 > 1 / 2) := by
  sorry

end comparison_inequalities_l287_28747


namespace students_in_neither_course_l287_28708

theorem students_in_neither_course (total : ℕ) (coding : ℕ) (robotics : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : coding = 90)
  (h3 : robotics = 70)
  (h4 : both = 25) :
  total - (coding + robotics - both) = 15 := by
  sorry

end students_in_neither_course_l287_28708


namespace dog_food_consumption_l287_28721

/-- The amount of dog food one dog eats per day, in scoops -/
def dog_food_per_dog : ℝ := 0.12

/-- The number of dogs Ella owns -/
def number_of_dogs : ℕ := 2

/-- The total amount of dog food consumed by all dogs in a day, in scoops -/
def total_food_consumed : ℝ := dog_food_per_dog * number_of_dogs

theorem dog_food_consumption :
  total_food_consumed = 0.24 := by
  sorry

end dog_food_consumption_l287_28721


namespace expression_evaluation_l287_28763

theorem expression_evaluation (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 13) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end expression_evaluation_l287_28763


namespace six_meetings_in_middle_l287_28712

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℕ  -- Speed in meters per minute

/-- Calculates the number of meetings in the middle for two runners -/
def numberOfMeetings (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- Theorem stating that two runners with given speeds meet 6 times in the middle -/
theorem six_meetings_in_middle :
  let runner1 : Runner := ⟨240⟩
  let runner2 : Runner := ⟨180⟩
  numberOfMeetings runner1 runner2 = 6 :=
by sorry

end six_meetings_in_middle_l287_28712


namespace candy_division_l287_28750

theorem candy_division (mark peter susan john lucy : ℝ) 
  (h1 : mark = 90)
  (h2 : peter = 120.5)
  (h3 : susan = 74.75)
  (h4 : john = 150)
  (h5 : lucy = 85.25)
  (total_people : ℕ)
  (h6 : total_people = 10) :
  (mark + peter + susan + john + lucy) / total_people = 52.05 := by
  sorry

end candy_division_l287_28750


namespace percent_subtraction_problem_l287_28770

theorem percent_subtraction_problem : ∃ x : ℝ, 0.12 * 160 - 0.38 * x = 11.2 := by
  sorry

end percent_subtraction_problem_l287_28770


namespace acute_angle_in_first_quadrant_l287_28705

-- Define what an acute angle is
def is_acute_angle (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Define what it means for an angle to be in the first quadrant
def in_first_quadrant (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

-- Theorem stating that an acute angle is in the first quadrant
theorem acute_angle_in_first_quadrant (θ : Real) : 
  is_acute_angle θ → in_first_quadrant θ := by
  sorry


end acute_angle_in_first_quadrant_l287_28705


namespace quadratic_factorization_l287_28783

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end quadratic_factorization_l287_28783


namespace photo_arrangements_l287_28794

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def adjacent_arrangements (total_people : ℕ) (adjacent_pair : ℕ) : ℕ :=
  factorial (total_people - adjacent_pair + 1) * factorial adjacent_pair

theorem photo_arrangements :
  adjacent_arrangements 6 2 = 240 := by
  sorry

end photo_arrangements_l287_28794


namespace line_not_in_second_quadrant_iff_l287_28729

/-- A line that does not pass through the second quadrant -/
def LineNotInSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (a - 2) * y = (3 * a - 1) * x - 1 → (x ≤ 0 → y ≤ 0)

/-- The main theorem: characterization of a for which the line doesn't pass through the second quadrant -/
theorem line_not_in_second_quadrant_iff (a : ℝ) :
  LineNotInSecondQuadrant a ↔ a ≥ 2 := by
  sorry

end line_not_in_second_quadrant_iff_l287_28729


namespace red_balls_drawn_is_random_variable_l287_28752

/-- A bag containing black and red balls -/
structure Bag where
  black : ℕ
  red : ℕ

/-- The result of drawing balls from the bag -/
structure DrawResult where
  total : ℕ
  red : ℕ

/-- A random variable is a function that assigns a real number to each outcome of a random experiment -/
def RandomVariable (α : Type) := α → ℝ

/-- The bag containing 2 black balls and 6 red balls -/
def bag : Bag := { black := 2, red := 6 }

/-- The number of balls drawn -/
def numDrawn : ℕ := 2

/-- The function that counts the number of red balls drawn -/
def countRedBalls : DrawResult → ℕ := fun r => r.red

/-- Statement: The number of red balls drawn is a random variable -/
theorem red_balls_drawn_is_random_variable :
  ∃ (rv : RandomVariable DrawResult), ∀ (result : DrawResult),
    result.total = numDrawn ∧ result.red ≤ bag.red →
      rv result = (countRedBalls result : ℝ) :=
sorry

end red_balls_drawn_is_random_variable_l287_28752


namespace midnight_temperature_l287_28769

/-- Given an initial temperature, a temperature rise, and a temperature drop,
    calculate the final temperature. -/
def final_temperature (initial : Int) (rise : Int) (drop : Int) : Int :=
  initial + rise - drop

/-- Theorem stating that given the specific temperature changes in the problem,
    the final temperature is 2°C. -/
theorem midnight_temperature :
  final_temperature (-2) 12 8 = 2 := by
  sorry

end midnight_temperature_l287_28769


namespace factorial_sum_solution_l287_28775

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_sum_solution :
  ∀ a b c d e f : ℕ,
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 →
    a > b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e ∧ e ≥ f →
    factorial a = factorial b + factorial c + factorial d + factorial e + factorial f →
    ((a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1 ∧ e = 1 ∧ f = 1) ∨
     (a = 5 ∧ b = 4 ∧ c = 4 ∧ d = 4 ∧ e = 4 ∧ f = 4)) :=
by
  sorry

#check factorial_sum_solution

end factorial_sum_solution_l287_28775


namespace square_root_of_average_squares_ge_arithmetic_mean_l287_28722

theorem square_root_of_average_squares_ge_arithmetic_mean
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end square_root_of_average_squares_ge_arithmetic_mean_l287_28722


namespace hyperbola_focal_length_l287_28782

theorem hyperbola_focal_length (b : ℝ) : 
  (b > 0) → 
  (∃ (x y : ℝ), x^2 - y^2/b^2 = 1) → 
  (∃ (c : ℝ), c = 2) → 
  b = Real.sqrt 3 := by
sorry

end hyperbola_focal_length_l287_28782


namespace playground_count_l287_28751

theorem playground_count (numbers : List Nat) : 
  numbers.length = 6 ∧ 
  numbers.take 5 = [6, 12, 1, 12, 7] ∧ 
  (numbers.sum / numbers.length : ℚ) = 7 →
  numbers.getLast! = 4 := by
sorry

end playground_count_l287_28751


namespace least_positive_t_for_geometric_progression_l287_28718

open Real

theorem least_positive_t_for_geometric_progression :
  ∃ (t : ℝ), t > 0 ∧
  (∀ (α : ℝ), 0 < α → α < π / 3 →
    ∃ (r : ℝ), r > 0 ∧
    (arcsin (sin (3 * α)) * r = arcsin (sin (6 * α))) ∧
    (arcsin (sin (6 * α)) * r = arccos (cos (10 * α))) ∧
    (arccos (cos (10 * α)) * r = arcsin (sin (t * α)))) ∧
  (∀ (t' : ℝ), t' > 0 →
    (∀ (α : ℝ), 0 < α → α < π / 3 →
      ∃ (r : ℝ), r > 0 ∧
      (arcsin (sin (3 * α)) * r = arcsin (sin (6 * α))) ∧
      (arcsin (sin (6 * α)) * r = arccos (cos (10 * α))) ∧
      (arccos (cos (10 * α)) * r = arcsin (sin (t' * α)))) →
    t ≤ t') ∧
  t = 10 :=
by sorry

end least_positive_t_for_geometric_progression_l287_28718


namespace human_habitable_fraction_l287_28723

theorem human_habitable_fraction (water_fraction : ℚ) (inhabitable_fraction : ℚ) (agriculture_fraction : ℚ)
  (h1 : water_fraction = 3/5)
  (h2 : inhabitable_fraction = 2/3)
  (h3 : agriculture_fraction = 1/2) :
  (1 - water_fraction) * inhabitable_fraction * (1 - agriculture_fraction) = 2/15 := by
  sorry

end human_habitable_fraction_l287_28723


namespace food_distribution_l287_28762

/-- Given a total amount of food and a number of full boxes, calculates the amount of food per box. -/
def food_per_box (total_food : ℕ) (num_boxes : ℕ) : ℚ :=
  (total_food : ℚ) / (num_boxes : ℚ)

/-- Proves that given 777 kilograms of food and 388 full boxes, each box contains 2 kilograms of food. -/
theorem food_distribution (total_food : ℕ) (num_boxes : ℕ) 
  (h1 : total_food = 777) (h2 : num_boxes = 388) : 
  food_per_box total_food num_boxes = 2 := by
  sorry

end food_distribution_l287_28762


namespace exists_fibonacci_divisible_by_1000_l287_28738

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem exists_fibonacci_divisible_by_1000 :
  ∃ n : ℕ, n ≤ 1000001 ∧ fibonacci n % 1000 = 0 := by
  sorry

end exists_fibonacci_divisible_by_1000_l287_28738


namespace quadratic_equal_roots_l287_28771

theorem quadratic_equal_roots (k : ℝ) (A : ℝ) : 
  (∃ x : ℝ, A * x^2 + 6 * k * x + 2 = 0 ∧ 
   ∀ y : ℝ, A * y^2 + 6 * k * y + 2 = 0 → y = x) ∧ 
  k = 0.4444444444444444 → 
  A = 9 * k^2 / 2 :=
by sorry

end quadratic_equal_roots_l287_28771


namespace h_range_l287_28727

-- Define the function h(x)
def h (x : ℝ) : ℝ := 2 * (x - 3)

-- Define the domain of h(x)
def dom_h : Set ℝ := {x : ℝ | x ≠ -7}

-- Define the range of h(x)
def range_h : Set ℝ := {y : ℝ | y ≠ -20}

-- Theorem statement
theorem h_range : 
  {y : ℝ | ∃ x ∈ dom_h, h x = y} = range_h :=
sorry

end h_range_l287_28727


namespace solution_value_l287_28764

theorem solution_value (a : ℝ) : (1 + 1) * a = 2 * (2 * 1 - a) → a = 1 := by
  sorry

end solution_value_l287_28764


namespace train_speed_l287_28776

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 150)
  (h3 : crossing_time = 32) : 
  (train_length + bridge_length) / crossing_time = 12.5 := by
  sorry

#check train_speed

end train_speed_l287_28776


namespace sum_of_roots_equals_negative_two_l287_28702

theorem sum_of_roots_equals_negative_two
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (h1 : c^2 + a*c + b = 0)
  (h2 : d^2 + a*d + b = 0)
  (h3 : a^2 + c*a + d = 0)
  (h4 : b^2 + c*b + d = 0) :
  a + b + c + d = -2 := by
sorry

end sum_of_roots_equals_negative_two_l287_28702


namespace max_value_of_f_l287_28760

-- Define the function f
def f (x : ℝ) : ℝ := -4 * x^3 + 3 * x + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 3 ∧ ∀ x ∈ Set.Icc 0 1, f x ≤ M :=
by sorry

end max_value_of_f_l287_28760


namespace simplify_sqrt_x_squared_y_second_quadrant_l287_28730

theorem simplify_sqrt_x_squared_y_second_quadrant (x y : ℝ) (h1 : x < 0) (h2 : y > 0) :
  Real.sqrt (x^2 * y) = -x * Real.sqrt y := by
  sorry

end simplify_sqrt_x_squared_y_second_quadrant_l287_28730


namespace hyperbola_eccentricity_l287_28784

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 5/4 ∧ 
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ (x y : ℝ), (y = (4/3) * x ∨ y = -(4/3) * x) ↔ (y = (a/b) * x ∨ y = -(a/b) * x)) →
  (∀ (x y : ℝ), y^2/a^2 - x^2/b^2 = 1 → x = 0 → ∃ (c : ℝ), y^2 = a^2 + c^2) →
  e = (a^2 + b^2).sqrt / a := by
sorry

end hyperbola_eccentricity_l287_28784


namespace find_number_l287_28703

theorem find_number : ∃ x : ℝ, 4 * x - 23 = 33 ∧ x = 14 := by
  sorry

end find_number_l287_28703


namespace gas_usage_multiple_l287_28701

theorem gas_usage_multiple (felicity_usage adhira_usage : ℕ) 
  (h1 : felicity_usage = 23)
  (h2 : adhira_usage = 7)
  (h3 : ∃ m : ℕ, felicity_usage = m * adhira_usage - 5) :
  ∃ m : ℕ, m = 4 ∧ felicity_usage = m * adhira_usage - 5 :=
by sorry

end gas_usage_multiple_l287_28701


namespace expression_evaluation_l287_28786

theorem expression_evaluation :
  let a : ℚ := -1/6
  2 * (a + 1) * (a - 1) - a * (2 * a - 3) = -5/2 := by
  sorry

end expression_evaluation_l287_28786


namespace equation_solution_l287_28761

theorem equation_solution : ∃ x : ℝ, 2 * x - 4 = 0 ∧ x = 2 := by
  sorry

end equation_solution_l287_28761


namespace inequality_solution_set_l287_28745

theorem inequality_solution_set (m n : ℝ) : 
  (∀ x, mx - n > 0 ↔ x < 1/3) → 
  (∀ x, (m + n) * x < n - m ↔ x > -1/2) :=
sorry

end inequality_solution_set_l287_28745


namespace parabola_perpendicular_line_passes_through_point_l287_28740

/-- The parabola y = x^2 -/
def parabola (p : ℝ × ℝ) : Prop := p.2 = p.1^2

/-- Two points are different -/
def different (p q : ℝ × ℝ) : Prop := p ≠ q

/-- A point is not the origin -/
def not_origin (p : ℝ × ℝ) : Prop := p ≠ (0, 0)

/-- Two vectors are perpendicular -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- A point lies on a line defined by two other points -/
def on_line (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r = (1 - t) • p + t • q

theorem parabola_perpendicular_line_passes_through_point
  (A B : ℝ × ℝ)
  (h_parabola_A : parabola A)
  (h_parabola_B : parabola B)
  (h_different : different A B)
  (h_not_origin_A : not_origin A)
  (h_not_origin_B : not_origin B)
  (h_perpendicular : perpendicular A B) :
  on_line A B (0, 1) :=
sorry

end parabola_perpendicular_line_passes_through_point_l287_28740


namespace min_n_for_120n_divisibility_l287_28742

theorem min_n_for_120n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (4 ∣ 120 * m) ∧ (8 ∣ 120 * m) ∧ (12 ∣ 120 * m) → n ≤ m) ∧
  (4 ∣ 120 * n) ∧ (8 ∣ 120 * n) ∧ (12 ∣ 120 * n) :=
by
  -- Proof goes here
  sorry

#check min_n_for_120n_divisibility

end min_n_for_120n_divisibility_l287_28742


namespace contest_ranking_l287_28716

theorem contest_ranking (A B C D : ℝ) 
  (non_negative : A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ D ≥ 0)
  (sum_equality : B + D = A + C)
  (interchange_inequality : A + B > C + D)
  (dick_exceeds : D > B + C) :
  A > D ∧ D > B ∧ B > C := by
sorry

end contest_ranking_l287_28716


namespace line_AB_equation_l287_28704

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 - 4*y^2 = 4

/-- Point P -/
def P : ℝ × ℝ := (8, 1)

/-- A point lies on the line AB -/
def on_line_AB (x y : ℝ) : Prop := ∃ (t : ℝ), x = 8 + t ∧ y = 1 + 2*t

/-- A and B are intersection points of line AB and the hyperbola -/
def A_B_intersection (A B : ℝ × ℝ) : Prop :=
  on_line_AB A.1 A.2 ∧ on_line_AB B.1 B.2 ∧
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2

/-- P is the midpoint of AB -/
def P_is_midpoint (A B : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + B.1) / 2 ∧ P.2 = (A.2 + B.2) / 2

/-- The main theorem -/
theorem line_AB_equation :
  ∃ (A B : ℝ × ℝ), A_B_intersection A B ∧ P_is_midpoint A B →
  ∀ (x y : ℝ), on_line_AB x y ↔ 2*x - y - 15 = 0 :=
sorry

end line_AB_equation_l287_28704


namespace endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l287_28758

/-- Given a line segment with one endpoint at (6, 1) and midpoint at (5, 7),
    the sum of the coordinates of the other endpoint is 17. -/
theorem endpoint_coordinate_sum : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → Prop :=
  fun endpoint1 midpoint endpoint2 =>
    endpoint1 = (6, 1) ∧
    midpoint = (5, 7) ∧
    midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
    endpoint2.1 + endpoint2.2 = 17

/-- Proof of the theorem -/
theorem endpoint_coordinate_sum_proof : ∃ (endpoint2 : ℝ × ℝ),
  endpoint_coordinate_sum (6, 1) (5, 7) endpoint2 := by
  sorry

end endpoint_coordinate_sum_endpoint_coordinate_sum_proof_l287_28758


namespace cosine_function_minimum_l287_28728

theorem cosine_function_minimum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (∀ x, a * Real.cos (b * x + c) ≥ a * Real.cos c) ∧ 
  (∀ ε > 0, ∃ x, a * Real.cos (b * x + (c - ε)) > a * Real.cos (c - ε)) →
  c = π :=
sorry

end cosine_function_minimum_l287_28728


namespace pieces_per_box_l287_28777

/-- Given information about Adam's chocolate candy boxes -/
structure ChocolateBoxes where
  totalBought : ℕ
  givenAway : ℕ
  piecesLeft : ℕ

/-- Theorem stating the number of pieces in each box -/
theorem pieces_per_box (boxes : ChocolateBoxes)
  (h1 : boxes.totalBought = 13)
  (h2 : boxes.givenAway = 7)
  (h3 : boxes.piecesLeft = 36) :
  boxes.piecesLeft / (boxes.totalBought - boxes.givenAway) = 6 := by
  sorry


end pieces_per_box_l287_28777


namespace horner_method_v3_l287_28757

def horner_polynomial (x : ℤ) : ℤ := 10 + 25*x - 8*x^2 + x^4 + 6*x^5 + 2*x^6

def horner_v3 (x : ℤ) : ℤ :=
  let v0 := 2
  let v1 := v0 * x + 6
  let v2 := v1 * x + 1
  v2 * x + 0

theorem horner_method_v3 :
  horner_v3 (-4) = -36 ∧
  horner_polynomial (-4) = ((((horner_v3 (-4) * (-4) - 8) * (-4) + 25) * (-4)) + 10) :=
by sorry

end horner_method_v3_l287_28757


namespace intersection_of_A_and_B_l287_28790

def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

def B : Set ℤ := {b | ∃ n : ℤ, b = n^2 - 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0, 3} := by sorry

end intersection_of_A_and_B_l287_28790


namespace perpendicular_plane_line_condition_l287_28780

-- Define the types for planes and lines
variable (Point : Type) (Vector : Type)
variable (Plane : Type) (Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define what it means for a line to be in a plane
variable (line_in_plane : Line → Plane → Prop)

theorem perpendicular_plane_line_condition 
  (α β : Plane) (m : Line) 
  (h_diff : α ≠ β) 
  (h_m_in_α : line_in_plane m α) :
  (∀ m, line_in_plane m α → perp_line_plane m β → perp_planes α β) ∧ 
  (∃ m, line_in_plane m α ∧ perp_planes α β ∧ ¬perp_line_plane m β) :=
sorry

end perpendicular_plane_line_condition_l287_28780


namespace expression_value_l287_28797

theorem expression_value (y : ℝ) (some_variable : ℝ) 
  (h1 : some_variable / (2 * y) = 3 / 2)
  (h2 : (7 * some_variable + 5 * y) / (some_variable - 2 * y) = 26) :
  some_variable = 3 * y :=
by sorry

end expression_value_l287_28797


namespace meeting_gender_ratio_l287_28736

theorem meeting_gender_ratio (total_population : ℕ) (females_attending : ℕ) : 
  total_population = 300 →
  females_attending = 50 →
  (total_population / 2 - females_attending) / females_attending = 2 := by
  sorry

end meeting_gender_ratio_l287_28736


namespace find_d_l287_28720

theorem find_d (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + 4 = d + Real.sqrt (a + b + c - d + 3)) : 
  d = 75/16 := by
sorry

end find_d_l287_28720


namespace book_writing_time_l287_28749

/-- Calculates the number of weeks required to write a book -/
def weeks_to_write_book (pages_per_hour : ℕ) (hours_per_day : ℕ) (total_pages : ℕ) : ℕ :=
  (total_pages / (pages_per_hour * hours_per_day) + 6) / 7

/-- Theorem: It takes 7 weeks to write a 735-page book at 5 pages per hour, 3 hours per day -/
theorem book_writing_time :
  weeks_to_write_book 5 3 735 = 7 := by
  sorry

#eval weeks_to_write_book 5 3 735

end book_writing_time_l287_28749


namespace correct_calculation_l287_28789

theorem correct_calculation (x : ℚ) (h : 15 / x = 5) : 21 / x = 7 := by
  sorry

end correct_calculation_l287_28789


namespace calculate_expression_l287_28766

theorem calculate_expression : 
  3⁻¹ + (27 : ℝ) ^ (1/3) - (5 - Real.sqrt 5)^0 + |Real.sqrt 3 - 1/3| = 2 + Real.sqrt 3 := by
  sorry

end calculate_expression_l287_28766


namespace cyclic_quadrilateral_diameter_l287_28767

/-- A cyclic quadrilateral is a quadrilateral that can be inscribed in a circle -/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The diameter of the circumscribed circle of a cyclic quadrilateral -/
def circumscribedCircleDiameter (q : CyclicQuadrilateral) : ℝ := sorry

/-- Theorem: The diameter of the circumscribed circle of a cyclic quadrilateral 
    with side lengths 25, 39, 52, and 60 is 65 -/
theorem cyclic_quadrilateral_diameter :
  ∀ (q : CyclicQuadrilateral), 
    q.a = 25 ∧ q.b = 39 ∧ q.c = 52 ∧ q.d = 60 →
    circumscribedCircleDiameter q = 65 := by sorry

end cyclic_quadrilateral_diameter_l287_28767


namespace range_of_m_for_inequality_l287_28746

theorem range_of_m_for_inequality (m : ℝ) : 
  (∃ x : ℝ, Real.sqrt ((x + m) ^ 2) + Real.sqrt ((x - 1) ^ 2) ≤ 3) ↔ 
  -4 ≤ m ∧ m ≤ 2 := by
sorry

end range_of_m_for_inequality_l287_28746


namespace jane_final_crayons_l287_28787

/-- The number of crayons Jane ends up with after the hippopotamus incident and finding additional crayons -/
def final_crayon_count (x y : ℕ) : ℕ :=
  y - x + 15

/-- Theorem stating that given the conditions, Jane ends up with 95 crayons -/
theorem jane_final_crayons :
  let x : ℕ := 7  -- number of crayons eaten by the hippopotamus
  let y : ℕ := 87 -- number of crayons Jane had initially
  final_crayon_count x y = 95 := by
  sorry

end jane_final_crayons_l287_28787


namespace original_number_proof_l287_28755

theorem original_number_proof : 
  ∃! x : ℕ, 
    (∃ k : ℕ, x + 5 = 23 * k) ∧ 
    (∀ y : ℕ, y < 5 → ∀ m : ℕ, x + y ≠ 23 * m) :=
by
  sorry

end original_number_proof_l287_28755


namespace max_value_of_f_on_I_l287_28714

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

-- Define the closed interval [0, 3]
def I : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Statement of the theorem
theorem max_value_of_f_on_I :
  ∃ (c : ℝ), c ∈ I ∧ ∀ (x : ℝ), x ∈ I → f x ≤ f c ∧ f c = 2 :=
sorry

end max_value_of_f_on_I_l287_28714


namespace integral_convergence_l287_28798

/-- The floor function, returning the greatest integer less than or equal to a real number -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The integrand of our improper integral -/
noncomputable def f (x : ℝ) : ℝ :=
  (-1 : ℝ) ^ (floor (1 / x)) / x

/-- Statement of the convergence properties of our improper integral -/
theorem integral_convergence :
  ¬ (∃ (I : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (a : ℝ), 0 < a ∧ a < δ → |∫ x in a..1, |f x| - I| < ε) ∧
  (∃ (I : ℝ), ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (a : ℝ), 0 < a ∧ a < δ → |∫ x in a..1, f x - I| < ε) :=
by sorry

end integral_convergence_l287_28798


namespace expansion_coefficient_l287_28773

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

def coefficient_x2y2 (a b : ℕ) : ℕ := 
  binomial_coefficient a 2 * binomial_coefficient b 2

theorem expansion_coefficient : 
  coefficient_x2y2 3 4 = 18 := by sorry

end expansion_coefficient_l287_28773


namespace annual_growth_rate_l287_28700

theorem annual_growth_rate (initial_value final_value : ℝ) (h1 : initial_value = 70400) 
  (h2 : final_value = 89100) : ∃ r : ℝ, initial_value * (1 + r)^2 = final_value ∧ r = 0.125 := by
  sorry

end annual_growth_rate_l287_28700


namespace max_value_x_plus_y_l287_28793

theorem max_value_x_plus_y (x y : ℝ) (h : x - Real.sqrt (x + 1) = Real.sqrt (y + 3) - y) :
  ∃ (M : ℝ), M = 4 ∧ x + y ≤ M ∧ ∀ (N : ℝ), (x + y ≤ N) → (M ≤ N) := by
  sorry

end max_value_x_plus_y_l287_28793


namespace inscribed_circle_area_ratio_l287_28772

/-- 
Given a right triangle with hypotenuse h, legs a and b, and an inscribed circle with radius r,
prove that the ratio of the area of the inscribed circle to the area of the triangle is πr / (h + r).
-/
theorem inscribed_circle_area_ratio (h a b r : ℝ) (h_positive : h > 0) (r_positive : r > 0) 
  (right_triangle : a^2 + b^2 = h^2) (inscribed_circle : r = (a + b - h) / 2) : 
  (π * r^2) / ((1/2) * a * b) = π * r / (h + r) := by
  sorry

end inscribed_circle_area_ratio_l287_28772


namespace birth_year_problem_l287_28725

theorem birth_year_problem (x : ℕ) : 
  (1850 ≤ x^2 + x) ∧ (x^2 + x < 1900) → -- Born in second half of 19th century
  (x^2 + 2*x - x = x^2 + x) →           -- x years old in year x^2 + 2x
  x^2 + x = 1892                        -- Year of birth is 1892
:= by sorry

end birth_year_problem_l287_28725


namespace x_squared_is_quadratic_l287_28709

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing x^2 = 0 -/
def f (x : ℝ) : ℝ := x^2

/-- Theorem stating that x^2 = 0 is a quadratic equation in one variable -/
theorem x_squared_is_quadratic : is_quadratic_equation f :=
sorry

end x_squared_is_quadratic_l287_28709


namespace fourth_power_sum_geq_four_times_product_l287_28774

theorem fourth_power_sum_geq_four_times_product (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  a^4 + b^4 + c^4 + d^4 ≥ 4 * a * b * c * d := by
  sorry

end fourth_power_sum_geq_four_times_product_l287_28774


namespace equation_represents_hyperbola_l287_28724

/-- Represents a conic section -/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section given by the equation ax² + bxy + cy² + dx + ey + f = 0 -/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x² - 4y² + 6x - 8 = 0 represents a hyperbola -/
theorem equation_represents_hyperbola :
  determineConicSection 1 0 (-4) 6 0 (-8) = ConicSection.Hyperbola :=
sorry

end equation_represents_hyperbola_l287_28724


namespace triangle_inequality_l287_28731

theorem triangle_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end triangle_inequality_l287_28731


namespace min_value_expression_l287_28737

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_xyz : x * y * z = 1 / 2) :
  x^3 + 4*x*y + 16*y^3 + 8*y*z + 3*z^3 ≥ 18 ∧
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ * y₀ * z₀ = 1 / 2 ∧
    x₀^3 + 4*x₀*y₀ + 16*y₀^3 + 8*y₀*z₀ + 3*z₀^3 = 18 :=
by sorry

end min_value_expression_l287_28737


namespace monomial_sum_condition_l287_28748

/-- If the sum of two monomials 2a^5*b^(2m+4) and a^(2n-3)*b^8 is still a monomial,
    then m = 2 and n = 4 -/
theorem monomial_sum_condition (a b : ℝ) (m n : ℕ) :
  (∃ (k : ℝ) (p q : ℕ), 2 * a^5 * b^(2*m+4) + a^(2*n-3) * b^8 = k * a^p * b^q) →
  m = 2 ∧ n = 4 := by
sorry


end monomial_sum_condition_l287_28748


namespace kolya_cannot_descend_l287_28717

/-- Represents the possible jump sizes Kolya can make. -/
inductive JumpSize
  | six
  | seven
  | eight

/-- Represents a sequence of jumps Kolya makes. -/
def JumpSequence := List JumpSize

/-- The total number of steps on the ladder. -/
def totalSteps : Nat := 100

/-- Converts a JumpSize to its corresponding natural number. -/
def jumpSizeToNat (j : JumpSize) : Nat :=
  match j with
  | JumpSize.six => 6
  | JumpSize.seven => 7
  | JumpSize.eight => 8

/-- Calculates the position after a sequence of jumps. -/
def finalPosition (jumps : JumpSequence) : Int :=
  totalSteps - (jumps.map jumpSizeToNat).sum

/-- Checks if a sequence of jumps results in unique positions. -/
def hasUniquePositions (jumps : JumpSequence) : Prop :=
  let positions := List.scanl (fun pos jump => pos - jumpSizeToNat jump) totalSteps jumps
  positions.Nodup

/-- Theorem stating that Kolya cannot descend the ladder under the given conditions. -/
theorem kolya_cannot_descend :
  ¬∃ (jumps : JumpSequence), finalPosition jumps = 0 ∧ hasUniquePositions jumps :=
sorry


end kolya_cannot_descend_l287_28717


namespace isosceles_triangle_perimeter_l287_28744

/-- An isosceles triangle with side lengths m-2, 2m+1, and 8 has a perimeter of 17.5 -/
theorem isosceles_triangle_perimeter : ∀ m : ℝ,
  let a := m - 2
  let b := 2 * m + 1
  let c := 8
  (a = c ∨ b = c) → -- isosceles condition
  (a + b > c ∧ b + c > a ∧ c + a > b) → -- triangle inequality
  a + b + c = 17.5 := by
  sorry

end isosceles_triangle_perimeter_l287_28744


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l287_28743

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x + y = -b / a) :=
by sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x ↦ 2 * x^2 + 2006 * x - 2007
  (∃ x y, f x = 0 ∧ f y = 0 ∧ x + y = -1003) :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l287_28743


namespace oil_tank_depth_l287_28711

/-- Represents a right frustum oil tank -/
structure RightFrustumTank where
  volume : ℝ  -- Volume in liters
  top_edge : ℝ  -- Length of top edge in cm
  bottom_edge : ℝ  -- Length of bottom edge in cm

/-- Calculates the depth of a right frustum oil tank -/
def calculate_depth (tank : RightFrustumTank) : ℝ :=
  sorry

/-- Theorem stating that the depth of the given oil tank is 75 cm -/
theorem oil_tank_depth (tank : RightFrustumTank) 
  (h1 : tank.volume = 190)
  (h2 : tank.top_edge = 60)
  (h3 : tank.bottom_edge = 40) :
  calculate_depth tank = 75 :=
sorry

end oil_tank_depth_l287_28711


namespace max_value_of_f_l287_28756

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 / x else -x^2 + 2

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l287_28756


namespace eric_egg_collection_l287_28779

/-- Represents the types of birds on Eric's farm -/
inductive BirdType
  | Chicken
  | Duck
  | Goose

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def num_birds (b : BirdType) : Nat :=
  match b with
  | BirdType.Chicken => 6
  | BirdType.Duck => 4
  | BirdType.Goose => 2

def normal_laying_rate (b : BirdType) : Nat :=
  match b with
  | BirdType.Chicken => 3
  | BirdType.Duck => 2
  | BirdType.Goose => 1

def is_sunday (d : Day) : Bool :=
  match d with
  | Day.Sunday => true
  | _ => false

def laying_rate (b : BirdType) (d : Day) : Nat :=
  if is_sunday d then
    max (normal_laying_rate b - 1) 0
  else
    normal_laying_rate b

def daily_eggs (d : Day) : Nat :=
  (num_birds BirdType.Chicken * laying_rate BirdType.Chicken d) +
  (num_birds BirdType.Duck * laying_rate BirdType.Duck d) +
  (num_birds BirdType.Goose * laying_rate BirdType.Goose d)

def weekly_eggs : Nat :=
  daily_eggs Day.Monday +
  daily_eggs Day.Tuesday +
  daily_eggs Day.Wednesday +
  daily_eggs Day.Thursday +
  daily_eggs Day.Friday +
  daily_eggs Day.Saturday +
  daily_eggs Day.Sunday

theorem eric_egg_collection : weekly_eggs = 184 := by
  sorry

end eric_egg_collection_l287_28779


namespace equal_sums_exist_l287_28778

/-- Represents a cell in the table -/
inductive Cell
  | Neg : Cell  -- Represents -1
  | Zero : Cell -- Represents 0
  | Pos : Cell  -- Represents 1

/-- Represents a (2n+1) × (2n+1) table -/
def Table (n : ℕ) := Fin (2*n+1) → Fin (2*n+1) → Cell

/-- Calculates the sum of a row or column -/
def sum_line (t : Table n) (is_row : Bool) (i : Fin (2*n+1)) : ℤ :=
  sorry

/-- The main theorem -/
theorem equal_sums_exist (n : ℕ) (t : Table n) :
  ∃ (i j : Fin (2*n+1)) (b₁ b₂ : Bool), 
    (i ≠ j ∨ b₁ ≠ b₂) ∧ sum_line t b₁ i = sum_line t b₂ j :=
sorry

end equal_sums_exist_l287_28778
