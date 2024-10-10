import Mathlib

namespace nested_series_sum_l2796_279666

def nested_series : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * (1 + nested_series n)

theorem nested_series_sum : nested_series 7 = 510 := by
  sorry

end nested_series_sum_l2796_279666


namespace quadratic_roots_complex_and_distinct_l2796_279654

-- Define the coefficients of the quadratic equation
def a : ℂ := 1
def b : ℂ := 2 + 2*Complex.I
def c : ℂ := 5

-- Define the discriminant
def discriminant : ℂ := b^2 - 4*a*c

-- Theorem statement
theorem quadratic_roots_complex_and_distinct :
  ¬(discriminant = 0) ∧ ¬(discriminant.im = 0) := by
  sorry

end quadratic_roots_complex_and_distinct_l2796_279654


namespace min_values_for_constrained_x_y_l2796_279619

theorem min_values_for_constrained_x_y :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 →
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a + b = 1 → 2 / x + 1 / y ≤ 2 / a + 1 / b) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → 2 * a + b = 1 → 4 * x^2 + y^2 ≤ 4 * a^2 + b^2) ∧
  (2 / x + 1 / y = 9) ∧
  (4 * x^2 + y^2 = 1/2) := by
sorry

end min_values_for_constrained_x_y_l2796_279619


namespace petya_can_prevent_natural_sum_l2796_279675

/-- Represents a player's turn in the game -/
structure Turn where
  player : Bool  -- true for Petya, false for Vasya
  fractions : List Nat  -- List of denominators of fractions written

/-- The state of the game board -/
structure GameState where
  turns : List Turn
  sum : Rat

/-- Vasya's strategy to choose fractions -/
def vasyaStrategy (state : GameState) : List Nat := sorry

/-- Petya's strategy to choose a fraction -/
def petyaStrategy (state : GameState) : Nat := sorry

/-- Checks if the sum of fractions is a natural number -/
def isNaturalSum (sum : Rat) : Bool := sorry

/-- Simulates the game for a given number of rounds -/
def playGame (rounds : Nat) : GameState := sorry

/-- Theorem stating that Petya can prevent Vasya from achieving a natural number sum -/
theorem petya_can_prevent_natural_sum :
  ∀ (rounds : Nat), ¬(isNaturalSum (playGame rounds).sum) := by sorry

end petya_can_prevent_natural_sum_l2796_279675


namespace sum_of_edge_lengths_specific_prism_l2796_279668

/-- Regular hexagonal prism with given base side length and height -/
structure RegularHexagonalPrism where
  base_side_length : ℝ
  height : ℝ

/-- Calculate the sum of the lengths of all edges of a regular hexagonal prism -/
def sum_of_edge_lengths (prism : RegularHexagonalPrism) : ℝ :=
  12 * prism.base_side_length + 6 * prism.height

/-- Theorem: The sum of edge lengths for a regular hexagonal prism with base side 6 cm and height 11 cm is 138 cm -/
theorem sum_of_edge_lengths_specific_prism :
  let prism : RegularHexagonalPrism := ⟨6, 11⟩
  sum_of_edge_lengths prism = 138 := by
  sorry

end sum_of_edge_lengths_specific_prism_l2796_279668


namespace divisibility_property_l2796_279677

theorem divisibility_property (p n q : ℕ) : 
  Prime p → 
  n > 0 → 
  q > 0 → 
  q ∣ ((n + 1)^p - n^p) → 
  p ∣ (q - 1) := by
sorry

end divisibility_property_l2796_279677


namespace fair_draw_l2796_279601

/-- Represents the number of players in the game -/
def num_players : ℕ := 10

/-- Represents the number of red balls in the hat -/
def red_balls : ℕ := 1

/-- Represents the number of white balls in the hat -/
def white_balls (h : ℕ) : ℕ := 10 * h - 1

/-- The probability of the host drawing a red ball -/
def host_probability (k n : ℕ) : ℚ := k / (k + n)

/-- The probability of the next player drawing a red ball -/
def next_player_probability (k n : ℕ) : ℚ := (n / (k + n)) * (k / (k + n - 1))

/-- Theorem stating the condition for a fair draw -/
theorem fair_draw (h : ℕ) :
  host_probability red_balls (white_balls h) = next_player_probability red_balls (white_balls h) :=
sorry

end fair_draw_l2796_279601


namespace sequence_sum_property_l2796_279690

theorem sequence_sum_property (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n : ℕ+, S n = n^2 - a n) →
  (∃ k : ℕ+, 1 < S k ∧ S k < 9) →
  (∃ k : ℕ+, k = 2 ∧ 1 < S k ∧ S k < 9) :=
by sorry

end sequence_sum_property_l2796_279690


namespace number_plus_sqrt_equals_24_l2796_279628

theorem number_plus_sqrt_equals_24 : ∃ x : ℝ, x + Real.sqrt (-4 + 6 * 4 * 3) = 24 := by
  sorry

end number_plus_sqrt_equals_24_l2796_279628


namespace original_number_proof_l2796_279665

theorem original_number_proof (n : ℕ) (k : ℕ) : 
  (∃ m : ℕ, n + k = 5 * m) ∧ 
  (n + k = 2500) ∧ 
  (∀ j : ℕ, j < k → ¬∃ m : ℕ, n + j = 5 * m) →
  n = 2500 :=
by sorry

end original_number_proof_l2796_279665


namespace unique_solution_sin_equation_l2796_279609

theorem unique_solution_sin_equation :
  ∃! x : ℝ, x = Real.sin x + 1993 := by sorry

end unique_solution_sin_equation_l2796_279609


namespace S_finite_iff_power_of_two_l2796_279627

def S (k : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 % 2 = 1 ∧ 
       Nat.gcd t.2.1 t.2.2 = 1 ∧ 
       t.2.1 + t.2.2 = k ∧ 
       t.1 ∣ (t.2.1 ^ t.1 + t.2.2 ^ t.1)}

theorem S_finite_iff_power_of_two (k : ℕ) (h : k > 1) :
  Set.Finite (S k) ↔ ∃ α : ℕ, k = 2^α ∧ α > 0 :=
sorry

end S_finite_iff_power_of_two_l2796_279627


namespace unique_solution_for_exponential_equation_l2796_279644

theorem unique_solution_for_exponential_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (7^x = 3^y + 4) → (x = 1 ∧ y = 1) :=
by sorry

end unique_solution_for_exponential_equation_l2796_279644


namespace expected_score_is_80_l2796_279646

/-- A math test with multiple-choice questions -/
structure MathTest where
  num_questions : ℕ
  points_per_correct : ℕ
  prob_correct : ℝ

/-- Expected score for a math test -/
def expected_score (test : MathTest) : ℝ :=
  test.num_questions * test.points_per_correct * test.prob_correct

/-- Theorem: The expected score for the given test is 80 points -/
theorem expected_score_is_80 (test : MathTest) 
    (h1 : test.num_questions = 25)
    (h2 : test.points_per_correct = 4)
    (h3 : test.prob_correct = 0.8) : 
  expected_score test = 80 := by
  sorry

end expected_score_is_80_l2796_279646


namespace point_in_second_quadrant_l2796_279648

theorem point_in_second_quadrant (m : ℝ) : 
  let p : ℝ × ℝ := (-1, m^2 + 1)
  p.1 < 0 ∧ p.2 > 0 :=
by sorry

end point_in_second_quadrant_l2796_279648


namespace sum_of_squares_l2796_279610

theorem sum_of_squares (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (eq1 : x₁ + 4*x₂ + 9*x₃ + 16*x₄ + 25*x₅ + 36*x₆ = 1)
  (eq2 : 4*x₁ + 9*x₂ + 16*x₃ + 25*x₄ + 36*x₅ + 49*x₆ = 14)
  (eq3 : 9*x₁ + 16*x₂ + 25*x₃ + 36*x₄ + 49*x₅ + 64*x₆ = 135) :
  16*x₁ + 25*x₂ + 36*x₃ + 49*x₄ + 64*x₅ + 81*x₆ = 832 := by
  sorry

end sum_of_squares_l2796_279610


namespace green_bean_to_corn_ratio_l2796_279651

/-- Represents the number of servings produced by each type of plant. -/
structure PlantServings where
  carrot : ℕ
  corn : ℕ
  greenBean : ℕ

/-- Represents the number of plants in each plot. -/
def plantsPerPlot : ℕ := 9

/-- Represents the total number of servings produced. -/
def totalServings : ℕ := 306

/-- The theorem stating the ratio of green bean to corn servings. -/
theorem green_bean_to_corn_ratio (s : PlantServings) :
  s.carrot = 4 →
  s.corn = 5 * s.carrot →
  s.greenBean * plantsPerPlot + s.carrot * plantsPerPlot + s.corn * plantsPerPlot = totalServings →
  s.greenBean * 2 = s.corn := by
  sorry

#check green_bean_to_corn_ratio

end green_bean_to_corn_ratio_l2796_279651


namespace smallest_bdf_value_l2796_279603

theorem smallest_bdf_value (a b c d e f : ℕ) : 
  (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) →
  (((a + 1) / b * c / d * e / f) - (a / b * c / d * e / f) = 3) →
  ((a / b * (c + 1) / d * e / f) - (a / b * c / d * e / f) = 4) →
  ((a / b * c / d * (e + 1) / f) - (a / b * c / d * e / f) = 5) →
  60 ≤ b * d * f ∧ ∃ (b' d' f' : ℕ), b' * d' * f' = 60 := by
  sorry

end smallest_bdf_value_l2796_279603


namespace nathaniel_ticket_distribution_l2796_279656

/-- The number of tickets Nathaniel gives to each of his best friends -/
def tickets_per_friend (initial_tickets : ℕ) (remaining_tickets : ℕ) (num_friends : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) / num_friends

/-- Proof that Nathaniel gave 2 tickets to each of his best friends -/
theorem nathaniel_ticket_distribution :
  tickets_per_friend 11 3 4 = 2 := by
  sorry

end nathaniel_ticket_distribution_l2796_279656


namespace correct_contribution_l2796_279630

/-- The cost of the project in billions of dollars -/
def project_cost : ℝ := 25

/-- The number of participants in millions -/
def num_participants : ℝ := 300

/-- The contribution required from each participant -/
def individual_contribution : ℝ := 83

/-- Theorem stating that the individual contribution is correct given the project cost and number of participants -/
theorem correct_contribution : 
  (project_cost * 1000) / num_participants = individual_contribution := by
  sorry

end correct_contribution_l2796_279630


namespace eighth_power_sum_exists_l2796_279674

theorem eighth_power_sum_exists (ζ₁ ζ₂ ζ₃ : ℂ) 
  (sum_condition : ζ₁ + ζ₂ + ζ₃ = 2)
  (square_sum_condition : ζ₁^2 + ζ₂^2 + ζ₃^2 = 6)
  (cube_sum_condition : ζ₁^3 + ζ₂^3 + ζ₃^3 = 18) :
  ∃ s₈ : ℂ, ζ₁^8 + ζ₂^8 + ζ₃^8 = s₈ := by
  sorry

end eighth_power_sum_exists_l2796_279674


namespace triangle_existence_from_bisector_and_segments_l2796_279682

/-- Given an angle bisector and the segments it divides a side into,
    prove the existence of a triangle satisfying these conditions. -/
theorem triangle_existence_from_bisector_and_segments
  (l_c a' b' : ℝ) (h_positive : l_c > 0 ∧ a' > 0 ∧ b' > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    l_c ^ 2 = a * b - a' * b' ∧
    a' / b' = a / b :=
sorry

end triangle_existence_from_bisector_and_segments_l2796_279682


namespace planes_parallel_if_perpendicular_to_parallel_lines_l2796_279620

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem statement
theorem planes_parallel_if_perpendicular_to_parallel_lines
  (m n : Line) (α β : Plane) :
  parallel_lines m n →
  perpendicular_line_plane m α →
  perpendicular_line_plane n β →
  parallel_planes α β :=
sorry

end planes_parallel_if_perpendicular_to_parallel_lines_l2796_279620


namespace cube_inequality_l2796_279686

theorem cube_inequality (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_inequality_l2796_279686


namespace equation_solution_l2796_279653

theorem equation_solution : 
  {x : ℝ | x + 60 / (x - 5) = -12} = {0, -7} := by sorry

end equation_solution_l2796_279653


namespace min_perimeter_triangle_l2796_279611

-- Define the plane
variable (Plane : Type)

-- Define points
variable (O A B P P1 P2 A' B' : Plane)

-- Define the angle
variable (angle : Plane → Plane → Plane → Prop)

-- Define the property of being inside an angle
variable (insideAngle : Plane → Plane → Plane → Plane → Prop)

-- Define the property of being on a line
variable (onLine : Plane → Plane → Plane → Prop)

-- Define symmetry with respect to a line
variable (symmetricToLine : Plane → Plane → Plane → Plane → Prop)

-- Define intersection of two lines
variable (intersect : Plane → Plane → Plane → Plane → Plane → Prop)

-- Define perimeter of a triangle
variable (perimeter : Plane → Plane → Plane → ℝ)

-- Define the theorem
theorem min_perimeter_triangle
  (h1 : angle O A B)
  (h2 : insideAngle O A B P)
  (h3 : onLine O A A)
  (h4 : onLine O B B)
  (h5 : symmetricToLine P P1 O A)
  (h6 : symmetricToLine P P2 O B)
  (h7 : intersect P1 P2 O A A')
  (h8 : intersect P1 P2 O B B') :
  ∀ X Y, onLine O A X → onLine O B Y →
    perimeter P X Y ≥ perimeter P A' B' :=
  sorry

end min_perimeter_triangle_l2796_279611


namespace probability_five_or_king_l2796_279617

/-- A standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (unique_combinations : Bool)

/-- The probability of drawing a specific set of cards from a deck -/
def probability (d : Deck) (favorable_outcomes : Nat) : ℚ :=
  favorable_outcomes / d.cards

/-- Theorem: The probability of drawing a 5 or King from a standard deck is 2/13 -/
theorem probability_five_or_king (d : Deck) 
  (h1 : d.cards = 52)
  (h2 : d.ranks = 13)
  (h3 : d.suits = 4)
  (h4 : d.unique_combinations = true) :
  probability d 8 = 2 / 13 := by
  sorry

end probability_five_or_king_l2796_279617


namespace power_equality_l2796_279699

theorem power_equality : (243 : ℕ)^4 = 3^12 * 3^8 := by sorry

end power_equality_l2796_279699


namespace original_equals_scientific_l2796_279685

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 284000000

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation :=
  { significand := 2.84
    exponent := 8
    is_valid := by sorry }

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.significand * (10 : ℝ) ^ scientific_form.exponent := by
  sorry

end original_equals_scientific_l2796_279685


namespace unique_grid_with_star_one_l2796_279634

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → Fin 5

/-- Checks if a given row in the grid contains 1, 2, 3, 4, 5 without repetition -/
def valid_row (g : Grid) (row : Fin 5) : Prop :=
  ∀ n : Fin 5, ∃! col : Fin 5, g row col = n

/-- Checks if a given column in the grid contains 1, 2, 3, 4, 5 without repetition -/
def valid_column (g : Grid) (col : Fin 5) : Prop :=
  ∀ n : Fin 5, ∃! row : Fin 5, g row col = n

/-- Checks if a given 3x3 box in the grid contains 1, 2, 3, 4, 5 without repetition -/
def valid_box (g : Grid) (box_row box_col : Fin 2) : Prop :=
  ∀ n : Fin 5, ∃! (row col : Fin 3), g (3 * box_row + row) (3 * box_col + col) = n

/-- Checks if the entire grid is valid according to the problem constraints -/
def valid_grid (g : Grid) : Prop :=
  (∀ row : Fin 5, valid_row g row) ∧
  (∀ col : Fin 5, valid_column g col) ∧
  (∀ box_row box_col : Fin 2, valid_box g box_row box_col)

/-- The position of the cell marked with a star -/
def star_position : Fin 5 × Fin 5 := ⟨2, 4⟩

/-- The main theorem: There exists a unique valid grid where the star cell contains 1 -/
theorem unique_grid_with_star_one :
  ∃! g : Grid, valid_grid g ∧ g star_position.1 star_position.2 = 1 := by sorry

end unique_grid_with_star_one_l2796_279634


namespace smallest_valid_coloring_l2796_279604

/-- A coloring function that assigns a color (represented by a natural number) to each integer in the range [2, 31] -/
def Coloring := Fin 30 → Nat

/-- Predicate to check if a coloring is valid according to the given conditions -/
def IsValidColoring (c : Coloring) : Prop :=
  ∀ m n, 2 ≤ m ∧ m ≤ 31 ∧ 2 ≤ n ∧ n ≤ 31 →
    m ≠ n → m % n = 0 → c (m - 2) ≠ c (n - 2)

/-- The existence of a valid coloring using k colors -/
def ExistsValidColoring (k : Nat) : Prop :=
  ∃ c : Coloring, IsValidColoring c ∧ ∀ i, c i < k

/-- The main theorem: The smallest number of colors needed is 4 -/
theorem smallest_valid_coloring : (∃ k, ExistsValidColoring k ∧ ∀ j, j < k → ¬ExistsValidColoring j) ∧
                                   ExistsValidColoring 4 :=
sorry

end smallest_valid_coloring_l2796_279604


namespace maya_car_arrangement_l2796_279615

theorem maya_car_arrangement (current_cars : ℕ) (cars_per_row : ℕ) (additional_cars : ℕ) : 
  current_cars = 29 →
  cars_per_row = 7 →
  (current_cars + additional_cars) % cars_per_row = 0 →
  ∀ n : ℕ, n < additional_cars → (current_cars + n) % cars_per_row ≠ 0 →
  additional_cars = 6 := by
sorry

end maya_car_arrangement_l2796_279615


namespace positive_sum_from_positive_difference_l2796_279672

theorem positive_sum_from_positive_difference (a b : ℝ) : a - |b| > 0 → b + a > 0 := by
  sorry

end positive_sum_from_positive_difference_l2796_279672


namespace arithmetic_mean_problem_l2796_279655

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 20 + 3*x + 18 + (3*x + 6)) / 5 = 32 → x = 106 / 7 := by
  sorry

end arithmetic_mean_problem_l2796_279655


namespace claire_pets_l2796_279621

theorem claire_pets (total_pets : ℕ) (total_males : ℕ) 
  (h_total : total_pets = 92)
  (h_males : total_males = 25) :
  ∃ (gerbils hamsters : ℕ),
    gerbils + hamsters = total_pets ∧
    (gerbils / 4 : ℚ) + (hamsters / 3 : ℚ) = total_males ∧
    gerbils = 68 := by
  sorry


end claire_pets_l2796_279621


namespace school_sampling_is_systematic_l2796_279684

/-- Represents a student with a unique student number -/
structure Student where
  number : ℕ

/-- Represents the sampling method used -/
inductive SamplingMethod
  | Stratified
  | Lottery
  | Random
  | Systematic

/-- Function to check if a student number ends in 4 -/
def endsInFour (n : ℕ) : Bool :=
  n % 10 = 4

/-- The sampling method used in the school -/
def schoolSamplingMethod (students : List Student) : SamplingMethod :=
  SamplingMethod.Systematic

/-- Theorem stating that the school's sampling method is systematic sampling -/
theorem school_sampling_is_systematic (students : List Student) :
  schoolSamplingMethod students = SamplingMethod.Systematic :=
by sorry

end school_sampling_is_systematic_l2796_279684


namespace cost_per_bushel_approx_12_l2796_279641

-- Define the given constants
def apple_price : ℚ := 0.40
def apples_per_bushel : ℕ := 48
def profit : ℚ := 15
def apples_sold : ℕ := 100

-- Define the function to calculate the cost per bushel
def cost_per_bushel : ℚ :=
  let revenue := apple_price * apples_sold
  let cost := revenue - profit
  let bushels_sold := apples_sold / apples_per_bushel
  cost / bushels_sold

-- Theorem statement
theorem cost_per_bushel_approx_12 : 
  ∃ ε > 0, |cost_per_bushel - 12| < ε :=
sorry

end cost_per_bushel_approx_12_l2796_279641


namespace george_boxes_count_l2796_279614

/-- The number of blocks each box can hold -/
def blocks_per_box : ℕ := 6

/-- The total number of blocks George has -/
def total_blocks : ℕ := 12

/-- The number of boxes George has -/
def number_of_boxes : ℕ := total_blocks / blocks_per_box

theorem george_boxes_count : number_of_boxes = 2 := by
  sorry

end george_boxes_count_l2796_279614


namespace initial_deposit_is_one_l2796_279618

def initial_amount : ℕ := 100
def weeks : ℕ := 52
def final_total : ℕ := 1478

def arithmetic_sum (a₁ : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ))

theorem initial_deposit_is_one :
  ∃ (x : ℚ), 
    arithmetic_sum x weeks + initial_amount = final_total ∧ 
    x = 1 := by sorry

end initial_deposit_is_one_l2796_279618


namespace equation_solution_l2796_279613

theorem equation_solution : 
  ∀ x : ℝ, 
    (((x + 1)^2 + 1) / (x + 1) + ((x + 4)^2 + 4) / (x + 4) = 
     ((x + 2)^2 + 2) / (x + 2) + ((x + 3)^2 + 3) / (x + 3)) ↔ 
    (x = 0 ∨ x = -5/2) :=
by sorry

end equation_solution_l2796_279613


namespace smallest_solution_of_equation_l2796_279631

theorem smallest_solution_of_equation :
  let f : ℝ → ℝ := λ x => (3*x)/(x-3) + (3*x^2 - 27*x)/x
  ∃ x : ℝ, f x = 14 ∧ x = (-41 - Real.sqrt 4633) / 12 ∧
  ∀ y : ℝ, f y = 14 → y ≥ (-41 - Real.sqrt 4633) / 12 :=
by sorry

end smallest_solution_of_equation_l2796_279631


namespace paving_cost_calculation_l2796_279692

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation (length width rate : ℝ) 
  (h1 : length = 5)
  (h2 : width = 4.75)
  (h3 : rate = 900) :
  paving_cost length width rate = 21375 := by
  sorry

end paving_cost_calculation_l2796_279692


namespace max_x_minus_y_l2796_279616

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), w = a - b ∧ a^2 + b^2 - 4*a - 2*b - 4 = 0) →
  w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_l2796_279616


namespace equation_solutions_count_l2796_279608

theorem equation_solutions_count :
  ∃! (S : Set ℝ), (∀ x ∈ S, (x^2 - 7)^2 = 25) ∧ S.Finite ∧ S.ncard = 4 := by
  sorry

end equation_solutions_count_l2796_279608


namespace complex_subtraction_example_l2796_279660

theorem complex_subtraction_example : (6 - 2*I) - (3*I + 1) = 5 - 5*I := by
  sorry

end complex_subtraction_example_l2796_279660


namespace smallest_multiplier_for_perfect_square_l2796_279650

theorem smallest_multiplier_for_perfect_square : 
  ∃ (n : ℕ), n > 0 ∧ ∃ (m : ℕ), 1008 * n = m^2 ∧ ∀ (k : ℕ), k > 0 → k < n → ¬∃ (l : ℕ), 1008 * k = l^2 :=
by sorry

end smallest_multiplier_for_perfect_square_l2796_279650


namespace matrix_N_properties_l2796_279695

def N : Matrix (Fin 2) (Fin 2) ℚ := !![5, -3; -1/2, 2]

theorem matrix_N_properties :
  let v1 : Matrix (Fin 2) (Fin 1) ℚ := !![2; -1]
  let v2 : Matrix (Fin 2) (Fin 1) ℚ := !![0; 3]
  let r1 : Matrix (Fin 2) (Fin 1) ℚ := !![5; -3]
  let r2 : Matrix (Fin 2) (Fin 1) ℚ := !![-9; 6]
  (N * v1 = r1) ∧
  (N * v2 = r2) ∧
  (N 0 0 - N 1 1 = 3) :=
by
  sorry

end matrix_N_properties_l2796_279695


namespace chord_equation_l2796_279698

theorem chord_equation (x y : ℝ) :
  (x^2 + y^2 - 2*x = 0) →  -- Circle equation
  (∃ (t : ℝ), x = 1/2 + t ∧ y = 1/2 + t) →  -- Midpoint condition
  (x - y = 0) :=  -- Line equation
by sorry

end chord_equation_l2796_279698


namespace negative_two_is_square_root_of_four_l2796_279622

-- Define square root
def is_square_root (x y : ℝ) : Prop := y^2 = x

-- Theorem statement
theorem negative_two_is_square_root_of_four :
  is_square_root 4 (-2) :=
sorry

end negative_two_is_square_root_of_four_l2796_279622


namespace cyclist_speed_proof_l2796_279681

/-- The distance between Town X and Town Y in miles -/
def distance : ℝ := 90

/-- The speed difference between cyclists D and C in miles per hour -/
def speed_difference : ℝ := 5

/-- The distance from Town Y where cyclists C and D meet on D's return trip in miles -/
def meeting_point : ℝ := 15

/-- The speed of Cyclist C in miles per hour -/
def speed_C : ℝ := 12.5

theorem cyclist_speed_proof :
  ∃ (speed_D : ℝ),
    speed_D = speed_C + speed_difference ∧
    distance / speed_C = (distance + meeting_point) / speed_D :=
by sorry

end cyclist_speed_proof_l2796_279681


namespace sum_of_fraction_parts_l2796_279605

def repeating_decimal : ℚ := 2.5252525

theorem sum_of_fraction_parts : ∃ (n d : ℕ), 
  repeating_decimal = n / d ∧ 
  Nat.gcd n d = 1 ∧ 
  n + d = 349 := by sorry

end sum_of_fraction_parts_l2796_279605


namespace marble_jar_count_l2796_279639

theorem marble_jar_count : ∃ (total : ℕ), 
  (total / 2 : ℕ) + (total / 4 : ℕ) + 27 + 14 = total ∧ total = 164 := by
  sorry

end marble_jar_count_l2796_279639


namespace range_of_m_plus_n_l2796_279673

/-- The function f(x) -/
noncomputable def f (m n x : ℝ) : ℝ := m * Real.exp x + x^2 + n * x

/-- The set of roots of f(x) -/
def roots (m n : ℝ) : Set ℝ := {x | f m n x = 0}

/-- The set of roots of f(f(x)) -/
def double_roots (m n : ℝ) : Set ℝ := {x | f m n (f m n x) = 0}

/-- Main theorem: Given f(x) = me^x + x^2 + nx, where the roots of f and f(f) are the same and non-empty,
    the range of m+n is [0, 4) -/
theorem range_of_m_plus_n (m n : ℝ) 
    (h1 : roots m n = double_roots m n) 
    (h2 : roots m n ≠ ∅) : 
    0 ≤ m + n ∧ m + n < 4 := by sorry

end range_of_m_plus_n_l2796_279673


namespace circle_radius_implies_a_value_l2796_279687

theorem circle_radius_implies_a_value (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*a*x + 4*a*y = 0) → 
  (∃ (c_x c_y : ℝ), ∀ (x y : ℝ), (x - c_x)^2 + (y - c_y)^2 = 5) →
  a = 1 ∨ a = -1 := by
sorry

end circle_radius_implies_a_value_l2796_279687


namespace cube_division_l2796_279697

theorem cube_division (n : ℕ) (small_edge : ℝ) : 
  12 / n = small_edge ∧ 
  n * 6 * small_edge^2 = 8 * 6 * 12^2 → 
  n^3 = 512 ∧ small_edge = 1.5 := by
  sorry

end cube_division_l2796_279697


namespace stratified_sampling_theorem_l2796_279691

theorem stratified_sampling_theorem :
  let total_employees : ℕ := 150
  let senior_titles : ℕ := 15
  let intermediate_titles : ℕ := 45
  let junior_titles : ℕ := 90
  let sample_size : ℕ := 30
  
  senior_titles + intermediate_titles + junior_titles = total_employees →
  
  let senior_sample := sample_size * senior_titles / total_employees
  let intermediate_sample := sample_size * intermediate_titles / total_employees
  let junior_sample := sample_size * junior_titles / total_employees
  
  (senior_sample = 3 ∧ intermediate_sample = 9 ∧ junior_sample = 18) :=
by
  sorry

end stratified_sampling_theorem_l2796_279691


namespace hydrochloric_acid_moles_l2796_279659

/-- Represents the chemical reaction between Sodium bicarbonate and Hydrochloric acid -/
structure ChemicalReaction where
  sodium_bicarbonate : ℝ  -- moles of Sodium bicarbonate
  hydrochloric_acid : ℝ   -- moles of Hydrochloric acid
  sodium_chloride : ℝ     -- moles of Sodium chloride produced

/-- Theorem stating that when 1 mole of Sodium bicarbonate reacts to produce 1 mole of Sodium chloride,
    the amount of Hydrochloric acid used is also 1 mole -/
theorem hydrochloric_acid_moles (reaction : ChemicalReaction)
  (h1 : reaction.sodium_bicarbonate = 1)
  (h2 : reaction.sodium_chloride = 1) :
  reaction.hydrochloric_acid = 1 := by
  sorry


end hydrochloric_acid_moles_l2796_279659


namespace jason_balloons_l2796_279689

/-- Given an initial number of violet balloons and a number of lost violet balloons,
    calculate the remaining number of violet balloons. -/
def remaining_balloons (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that Jason's remaining violet balloons is 4,
    given he started with 7 and lost 3. -/
theorem jason_balloons : remaining_balloons 7 3 = 4 := by
  sorry

end jason_balloons_l2796_279689


namespace range_of_a_l2796_279638

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 2) → a ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
by sorry

end range_of_a_l2796_279638


namespace shaded_to_unshaded_ratio_l2796_279669

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  s : Point
  t : Point
  u : Point
  v : Point

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculate the area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop := sorry

/-- Check if two line segments are parallel -/
def isParallel (a : Point) (b : Point) (c : Point) (d : Point) : Prop := sorry

/-- Check if two line segments are equal in length -/
def segmentEqual (a : Point) (b : Point) (c : Point) (d : Point) : Prop := sorry

theorem shaded_to_unshaded_ratio 
  (s : Square) 
  (q p r o : Point) 
  (t1 t2 t3 : Triangle) :
  isMidpoint q s.s s.t →
  isMidpoint p s.u s.v →
  segmentEqual p r q r →
  isParallel s.v q p r →
  t1 = Triangle.mk q o r →
  t2 = Triangle.mk p o r →
  t3 = Triangle.mk q p s.v →
  (triangleArea t1 + triangleArea t2 + triangleArea t3) / 
  (squareArea s - (triangleArea t1 + triangleArea t2 + triangleArea t3)) = 3 / 5 := by
  sorry

end shaded_to_unshaded_ratio_l2796_279669


namespace race_time_proof_l2796_279649

/-- Represents the time taken by the first five runners to finish the race -/
def first_five_time : ℝ → ℝ := λ t => 5 * t

/-- Represents the time taken by the last three runners to finish the race -/
def last_three_time : ℝ → ℝ := λ t => 3 * (t + 2)

/-- Represents the total time taken by all runners to finish the race -/
def total_time : ℝ → ℝ := λ t => first_five_time t + last_three_time t

theorem race_time_proof :
  ∃ t : ℝ, total_time t = 70 ∧ first_five_time t = 40 :=
sorry

end race_time_proof_l2796_279649


namespace freshmen_sophomores_without_pets_l2796_279606

theorem freshmen_sophomores_without_pets (total_students : ℕ) 
  (freshmen_sophomore_ratio : ℚ) (pet_owner_ratio : ℚ) : ℕ :=
  by
  sorry

#check freshmen_sophomores_without_pets 400 (1/2) (1/5) = 160

end freshmen_sophomores_without_pets_l2796_279606


namespace rent_distribution_l2796_279629

/-- Represents an individual renting the pasture -/
structure Renter where
  name : String
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a renter -/
def calculateShare (r : Renter) (totalRent : ℚ) (totalOxMonths : ℕ) : ℚ :=
  (r.oxen * r.months : ℚ) * totalRent / totalOxMonths

/-- The main theorem stating the properties of rent distribution -/
theorem rent_distribution
  (renters : List Renter)
  (totalRent : ℚ)
  (h_positive_rent : totalRent > 0)
  (h_renters : renters = [
    ⟨"A", 10, 7⟩,
    ⟨"B", 12, 5⟩,
    ⟨"C", 15, 3⟩,
    ⟨"D", 8, 6⟩,
    ⟨"E", 20, 2⟩
  ])
  (h_total_rent : totalRent = 385) :
  let totalOxMonths := (renters.map (fun r => r.oxen * r.months)).sum
  let shares := renters.map (fun r => calculateShare r totalRent totalOxMonths)
  (∀ (r : Renter), r ∈ renters → 
    calculateShare r totalRent totalOxMonths = 
    (r.oxen * r.months : ℚ) * totalRent / totalOxMonths) ∧
  shares.sum = totalRent :=
sorry

end rent_distribution_l2796_279629


namespace intersection_of_A_and_B_l2796_279623

-- Define the sets A and B
def A : Set ℝ := {x | x * (x - 2) < 0}
def B : Set ℝ := {x | x - 1 > 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Ioo 1 2 := by sorry

end intersection_of_A_and_B_l2796_279623


namespace product_remainder_mod_five_l2796_279671

theorem product_remainder_mod_five : ∃ k : ℕ, 114 * 232 * 454^2 * 678 = 5 * k + 4 := by
  sorry

end product_remainder_mod_five_l2796_279671


namespace right_triangle_area_l2796_279664

/-- A right triangle ABC in the xy-plane with specific properties -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle_at_C : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  hypotenuse_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 900
  median_A_on_y_eq_x : A.1 = A.2
  median_B_on_y_eq_x_plus_1 : B.2 = B.1 + 1

/-- The area of the right triangle ABC is 448 -/
theorem right_triangle_area (t : RightTriangle) : 
  (1/2) * abs ((t.A.1 * (t.B.2 - t.C.2) + t.B.1 * (t.C.2 - t.A.2) + t.C.1 * (t.A.2 - t.B.2))) = 448 := by
  sorry

end right_triangle_area_l2796_279664


namespace no_solution_exists_l2796_279680

theorem no_solution_exists : ¬ ∃ (a b c t x₁ x₂ x₃ : ℝ),
  (a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁) ∧
  (a * x₁^2 + b * t * x₁ + c = 0) ∧
  (a * x₂^2 + b * t * x₂ + c = 0) ∧
  (b * x₂^2 + c * x₂ + a = 0) ∧
  (b * x₃^2 + c * x₃ + a = 0) ∧
  (c * x₃^2 + a * t * x₃ + b = 0) ∧
  (c * x₁^2 + a * t * x₁ + b = 0) :=
sorry

#check no_solution_exists

end no_solution_exists_l2796_279680


namespace parallel_properties_l2796_279683

-- Define a type for lines
variable {Line : Type}

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Define a relation for two lines being parallel to the same line
def parallel_to_same (l1 l2 : Line) : Prop :=
  ∃ l3 : Line, parallel l1 l3 ∧ parallel l2 l3

theorem parallel_properties :
  (∀ l1 l2 : Line, parallel_to_same parallel l1 l2 → parallel l1 l2) ∧
  (∀ l1 l2 : Line, parallel l1 l2 → parallel_to_same parallel l1 l2) ∧
  (∀ l1 l2 : Line, ¬parallel_to_same parallel l1 l2 → ¬parallel l1 l2) ∧
  (∀ l1 l2 : Line, ¬parallel l1 l2 → ¬parallel_to_same parallel l1 l2) :=
by sorry

end parallel_properties_l2796_279683


namespace no_call_days_l2796_279636

theorem no_call_days (total_days : ℕ) (call_period1 call_period2 call_period3 : ℕ) : 
  total_days = 365 ∧ call_period1 = 2 ∧ call_period2 = 5 ∧ call_period3 = 7 →
  total_days - (
    (total_days / call_period1 + total_days / call_period2 + total_days / call_period3) -
    (total_days / (Nat.lcm call_period1 call_period2) + 
     total_days / (Nat.lcm call_period1 call_period3) + 
     total_days / (Nat.lcm call_period2 call_period3)) +
    total_days / (Nat.lcm call_period1 (Nat.lcm call_period2 call_period3))
  ) = 125 := by
  sorry

end no_call_days_l2796_279636


namespace problem_statement_l2796_279640

theorem problem_statement (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (heq : a + b = 1/a + 1/b) : 
  (a + b ≥ 2) ∧ ¬(a^2 + a < 2 ∧ b^2 + b < 2) := by
  sorry

end problem_statement_l2796_279640


namespace parabola_vertex_coordinates_l2796_279633

/-- The vertex of the parabola y = x^2 - 4x + a lies on the line y = -4x - 1 -/
def vertex_on_line (a : ℝ) : Prop :=
  ∃ x y : ℝ, y = x^2 - 4*x + a ∧ y = -4*x - 1

/-- The coordinates of the vertex of the parabola y = x^2 - 4x + a -/
def vertex_coordinates (a : ℝ) : ℝ × ℝ := (2, -9)

theorem parabola_vertex_coordinates (a : ℝ) :
  vertex_on_line a → vertex_coordinates a = (2, -9) := by
  sorry


end parabola_vertex_coordinates_l2796_279633


namespace inequality_preservation_l2796_279661

theorem inequality_preservation (a b c : ℝ) (h : a > b) : a + c^2 > b + c^2 := by
  sorry

end inequality_preservation_l2796_279661


namespace f_symmetry_l2796_279612

-- Define a convex polygon as a list of vectors
def ConvexPolygon := List (ℝ × ℝ)

-- Define the projection function
def projection (v : ℝ × ℝ) (line : ℝ × ℝ) : ℝ := sorry

-- Define the function f
def f (P Q : ConvexPolygon) : ℝ :=
  List.sum (List.map (λ p => 
    (norm p) * (List.sum (List.map (λ q => abs (projection q p)) Q))
  ) P)

-- State the theorem
theorem f_symmetry (P Q : ConvexPolygon) : f P Q = f Q P := by sorry

end f_symmetry_l2796_279612


namespace total_cost_calculation_l2796_279676

/-- Calculates the total cost of beef and vegetables -/
theorem total_cost_calculation (beef_weight : ℝ) (veg_weight : ℝ) (veg_price : ℝ) :
  beef_weight = 4 →
  veg_weight = 6 →
  veg_price = 2 →
  beef_weight * (3 * veg_price) + veg_weight * veg_price = 36 := by
  sorry

#check total_cost_calculation

end total_cost_calculation_l2796_279676


namespace slope_of_line_with_60_degree_inclination_l2796_279602

theorem slope_of_line_with_60_degree_inclination :
  let angle_of_inclination : ℝ := 60 * π / 180
  let slope : ℝ := Real.tan angle_of_inclination
  slope = Real.sqrt 3 := by
  sorry

end slope_of_line_with_60_degree_inclination_l2796_279602


namespace triangle_area_in_rectangle_config_l2796_279626

/-- The area of a triangle in a specific geometric configuration -/
theorem triangle_area_in_rectangle_config : 
  ∀ (base height : ℝ),
  base = 16 ∧ 
  height = 12 * (18 / 39) →
  (1 / 2 : ℝ) * base * height = 1536 / 13 := by
  sorry

end triangle_area_in_rectangle_config_l2796_279626


namespace equilateral_triangle_l2796_279663

theorem equilateral_triangle (a b c : ℝ) 
  (h1 : a + b - c = 2) 
  (h2 : 2 * a * b - c^2 = 4) : 
  a = 2 ∧ b = 2 ∧ c = 2 := by
  sorry

end equilateral_triangle_l2796_279663


namespace four_intersection_points_iff_c_gt_one_l2796_279667

-- Define the ellipse equation
def ellipse (x y c : ℝ) : Prop :=
  x^2 + y^2/4 = c^2

-- Define the parabola equation
def parabola (x y c : ℝ) : Prop :=
  y = x^2 - 2*c

-- Define the intersection points
def intersection_points (c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ellipse p.1 p.2 c ∧ parabola p.1 p.2 c}

-- Theorem statement
theorem four_intersection_points_iff_c_gt_one (c : ℝ) :
  (∃ (p₁ p₂ p₃ p₄ : ℝ × ℝ), p₁ ∈ intersection_points c ∧
                            p₂ ∈ intersection_points c ∧
                            p₃ ∈ intersection_points c ∧
                            p₄ ∈ intersection_points c ∧
                            p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧
                            p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧
                            p₃ ≠ p₄) ↔
  c > 1 := by
  sorry

end four_intersection_points_iff_c_gt_one_l2796_279667


namespace root_difference_implies_k_l2796_279647

theorem root_difference_implies_k (k : ℝ) : 
  (∀ r s : ℝ, r^2 + k*r + 6 = 0 ∧ s^2 + k*s + 6 = 0 → 
    ∃ r' s' : ℝ, r'^2 - k*r' + 6 = 0 ∧ s'^2 - k*s' + 6 = 0 ∧ 
    r' = r + 5 ∧ s' = s + 5) → 
  k = 5 := by
sorry

end root_difference_implies_k_l2796_279647


namespace mother_escape_time_max_mother_time_l2796_279657

/-- Represents a family member with their tunnel traversal time -/
structure FamilyMember where
  name : String
  time : Nat

/-- Represents the cave escape scenario -/
structure CaveEscape where
  father : FamilyMember
  mother : FamilyMember
  son : FamilyMember
  daughter : FamilyMember
  timeLimit : Nat

/-- The main theorem to prove -/
theorem mother_escape_time (scenario : CaveEscape) :
  scenario.father.time = 1 ∧
  scenario.son.time = 4 ∧
  scenario.daughter.time = 5 ∧
  scenario.timeLimit = 12 →
  scenario.mother.time = 2 := by
  sorry

/-- Helper function to calculate the minimum time for two people to cross -/
def crossTime (a b : FamilyMember) : Nat :=
  max a.time b.time

/-- Helper function to check if a given escape plan is valid -/
def isValidEscapePlan (scenario : CaveEscape) (motherTime : Nat) : Prop :=
  let totalTime := crossTime scenario.father scenario.daughter +
                   scenario.father.time +
                   crossTime scenario.father scenario.son +
                   motherTime
  totalTime ≤ scenario.timeLimit

/-- Theorem stating that 2 minutes is the maximum possible time for the mother -/
theorem max_mother_time (scenario : CaveEscape) :
  scenario.father.time = 1 ∧
  scenario.son.time = 4 ∧
  scenario.daughter.time = 5 ∧
  scenario.timeLimit = 12 →
  ∀ t : Nat, t > 2 → ¬(isValidEscapePlan scenario t) := by
  sorry

end mother_escape_time_max_mother_time_l2796_279657


namespace math_olympiad_scores_l2796_279658

theorem math_olympiad_scores (n : ℕ) (scores : Fin n → ℕ) : 
  n = 20 →
  (∀ i j : Fin n, i ≠ j → scores i ≠ scores j) →
  (∀ i j k : Fin n, i ≠ j → j ≠ k → i ≠ k → scores i < scores j + scores k) →
  ∀ i : Fin n, scores i > 18 := by
  sorry

end math_olympiad_scores_l2796_279658


namespace sqrt_necessary_not_sufficient_l2796_279632

-- Define the necessary condition
def necessary_condition (x y : ℝ) : Prop :=
  (∀ x y, (Real.log x > Real.log y) → (Real.sqrt x > Real.sqrt y))

-- Define the sufficient condition
def sufficient_condition (x y : ℝ) : Prop :=
  (∀ x y, (Real.sqrt x > Real.sqrt y) → (Real.log x > Real.log y))

-- Theorem stating that the condition is necessary but not sufficient
theorem sqrt_necessary_not_sufficient :
  (∃ x y, necessary_condition x y) ∧ (¬∃ x y, sufficient_condition x y) := by
  sorry

end sqrt_necessary_not_sufficient_l2796_279632


namespace sum_reciprocal_and_sum_squares_l2796_279688

theorem sum_reciprocal_and_sum_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (1 / a + 1 / b ≥ 4) ∧ (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → a^2 + b^2 ≤ x^2 + y^2) := by
  sorry

#check sum_reciprocal_and_sum_squares

end sum_reciprocal_and_sum_squares_l2796_279688


namespace zeros_in_Q_l2796_279694

/-- R_k is an integer whose base-ten representation is a sequence of k ones -/
def R (k : ℕ+) : ℕ := (10^k.val - 1) / 9

/-- Q is the quotient of R_28 divided by R_8 -/
def Q : ℕ := R 28 / R 8

/-- count_zeros counts the number of zeros in the base-ten representation of a natural number -/
def count_zeros (n : ℕ) : ℕ := sorry

theorem zeros_in_Q : count_zeros Q = 21 := by sorry

end zeros_in_Q_l2796_279694


namespace tourist_group_size_l2796_279643

theorem tourist_group_size (k : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) → 
  (∃ n : ℕ, n = 23 ∧ 2 * k ≡ 1 [MOD n] ∧ 3 * k ≡ 13 [MOD n]) :=
by sorry

end tourist_group_size_l2796_279643


namespace solve_exponential_equation_l2796_279637

theorem solve_exponential_equation :
  ∃ y : ℕ, (8 : ℕ)^4 = 2^y ∧ y = 12 :=
by
  sorry

end solve_exponential_equation_l2796_279637


namespace lloyd_earnings_correct_l2796_279679

/-- Calculates Lloyd's earnings for the given work days -/
def lloyd_earnings (regular_rate : ℚ) (normal_hours : ℚ) (overtime_rate : ℚ) (saturday_rate : ℚ) 
  (monday_hours : ℚ) (tuesday_hours : ℚ) (saturday_hours : ℚ) : ℚ :=
  let monday_earnings := 
    min normal_hours monday_hours * regular_rate + 
    max 0 (monday_hours - normal_hours) * regular_rate * overtime_rate
  let tuesday_earnings := 
    min normal_hours tuesday_hours * regular_rate + 
    max 0 (tuesday_hours - normal_hours) * regular_rate * overtime_rate
  let saturday_earnings := saturday_hours * regular_rate * saturday_rate
  monday_earnings + tuesday_earnings + saturday_earnings

theorem lloyd_earnings_correct : 
  lloyd_earnings 5 8 (3/2) 2 (21/2) 9 6 = 665/4 := by
  sorry

end lloyd_earnings_correct_l2796_279679


namespace f_decreasing_implies_a_range_l2796_279642

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then -x^2 + 2*a*x - 2*a else a*x + 1

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) < 0) →
  a ∈ Set.Icc (-2 : ℝ) 0 ∧ a ≠ 0 :=
sorry

end f_decreasing_implies_a_range_l2796_279642


namespace smallest_perimeter_isosceles_triangle_l2796_279625

/-- Triangle with positive integer side lengths --/
structure IsoscelesTriangle where
  side : ℕ+
  base : ℕ+

/-- Point representing the intersection of angle bisectors --/
structure AngleBisectorIntersection where
  distance : ℕ+

/-- Theorem stating the smallest possible perimeter of the triangle --/
theorem smallest_perimeter_isosceles_triangle 
  (t : IsoscelesTriangle) 
  (j : AngleBisectorIntersection) 
  (h : j.distance = 10) : 
  2 * (t.side + t.base) ≥ 198 := by
  sorry

#check smallest_perimeter_isosceles_triangle

end smallest_perimeter_isosceles_triangle_l2796_279625


namespace fraction_simplification_l2796_279645

theorem fraction_simplification (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  (a^3 - a^2*b) / (a^2*b) - (a^2*b - b^3) / (a*b - b^2) - (a*b) / (a^2 - b^2) = -3*a / (a^2 - b^2) := by
  sorry

end fraction_simplification_l2796_279645


namespace alice_has_ball_after_two_turns_l2796_279670

/-- Represents the probability of Alice tossing the ball to Bob -/
def alice_toss_prob : ℚ := 5/8

/-- Represents the probability of Alice keeping the ball -/
def alice_keep_prob : ℚ := 3/8

/-- Represents the probability of Bob tossing the ball to Alice -/
def bob_toss_prob : ℚ := 1/4

/-- Represents the probability of Bob keeping the ball -/
def bob_keep_prob : ℚ := 3/4

/-- The theorem stating the probability of Alice having the ball after two turns -/
theorem alice_has_ball_after_two_turns : 
  alice_toss_prob * bob_toss_prob + alice_keep_prob * alice_keep_prob = 19/64 := by
  sorry

#check alice_has_ball_after_two_turns

end alice_has_ball_after_two_turns_l2796_279670


namespace inequality_system_solution_l2796_279662

theorem inequality_system_solution :
  ∀ x : ℝ, (x + 2 > -1 ∧ x - 5 < 3 * (x - 1)) ↔ x > -1 :=
by sorry

end inequality_system_solution_l2796_279662


namespace factorization_identity_l2796_279607

theorem factorization_identity (x y a : ℝ) : x * (a - y) - y * (y - a) = (x + y) * (a - y) := by
  sorry

end factorization_identity_l2796_279607


namespace salary_B_is_5000_l2796_279693

/-- Calculates the salary of person B given the salaries of other people and the average salary -/
def calculate_salary_B (salary_A salary_C salary_D salary_E average_salary : ℕ) : ℕ :=
  5 * average_salary - (salary_A + salary_C + salary_D + salary_E)

/-- Proves that B's salary is 5000 given the conditions in the problem -/
theorem salary_B_is_5000 :
  let salary_A : ℕ := 8000
  let salary_C : ℕ := 11000
  let salary_D : ℕ := 7000
  let salary_E : ℕ := 9000
  let average_salary : ℕ := 8000
  calculate_salary_B salary_A salary_C salary_D salary_E average_salary = 5000 := by
  sorry

#eval calculate_salary_B 8000 11000 7000 9000 8000

end salary_B_is_5000_l2796_279693


namespace hamburger_price_is_5_l2796_279600

-- Define the variables
def num_hamburgers : ℕ := 2
def num_cola : ℕ := 3
def cola_price : ℚ := 2
def discount : ℚ := 4
def total_paid : ℚ := 12

-- Define the theorem
theorem hamburger_price_is_5 :
  ∃ (hamburger_price : ℚ),
    hamburger_price * num_hamburgers + cola_price * num_cola - discount = total_paid ∧
    hamburger_price = 5 :=
by sorry

end hamburger_price_is_5_l2796_279600


namespace sum_remainder_mod_17_l2796_279635

theorem sum_remainder_mod_17 : (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) % 17 = 11 := by
  sorry

end sum_remainder_mod_17_l2796_279635


namespace sandy_book_purchase_l2796_279652

/-- The number of books Sandy bought from the second shop -/
def books_from_second_shop : ℕ := 55

/-- The number of books Sandy bought from the first shop -/
def books_from_first_shop : ℕ := 65

/-- The amount Sandy spent at the first shop in cents -/
def cost_first_shop : ℕ := 148000

/-- The amount Sandy spent at the second shop in cents -/
def cost_second_shop : ℕ := 92000

/-- The average price per book in cents -/
def average_price : ℕ := 2000

theorem sandy_book_purchase :
  books_from_second_shop = 55 :=
sorry

end sandy_book_purchase_l2796_279652


namespace triangle_perimeter_impossibility_l2796_279678

theorem triangle_perimeter_impossibility (a b x : ℝ) : 
  a = 20 → b = 15 → x > 0 → a + b + x ≠ 72 := by sorry

end triangle_perimeter_impossibility_l2796_279678


namespace decreasing_cubic_implies_nonpositive_a_l2796_279696

/-- A function f: ℝ → ℝ is decreasing if for all x y, x < y implies f x > f y -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

/-- The cubic function f(x) = ax³ - x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x + 1

theorem decreasing_cubic_implies_nonpositive_a :
  ∀ a : ℝ, DecreasingFunction (f a) → a ≤ 0 := by
  sorry

end decreasing_cubic_implies_nonpositive_a_l2796_279696


namespace tile_difference_l2796_279624

/-- The side length of the nth square in the sequence -/
def side_length (n : ℕ) : ℕ := 1 + 2 * (n - 1)

/-- The number of tiles in the nth square -/
def tiles (n : ℕ) : ℕ := (side_length n) ^ 2

/-- The difference in tiles between the 11th and 10th squares -/
theorem tile_difference : tiles 11 - tiles 10 = 80 := by
  sorry

end tile_difference_l2796_279624
