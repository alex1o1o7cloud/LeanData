import Mathlib

namespace cone_lateral_surface_area_l325_32584

/-- Given a cone with base radius 6 and volume 30π, its lateral surface area is 39π -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1 / 3) * π * r^2 * h = 30 * π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39 * π := by
sorry

end cone_lateral_surface_area_l325_32584


namespace ellipse_eccentricity_l325_32579

/-- An ellipse with given properties -/
structure Ellipse where
  -- Point P on the ellipse
  p : ℝ × ℝ
  -- Focus F₁
  f1 : ℝ × ℝ
  -- Focus F₂
  f2 : ℝ × ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating the eccentricity of the specific ellipse -/
theorem ellipse_eccentricity :
  let e : Ellipse := {
    p := (2, 3)
    f1 := (-2, 0)
    f2 := (2, 0)
  }
  eccentricity e = 1/2 := by
  sorry

end ellipse_eccentricity_l325_32579


namespace speed_above_limit_l325_32587

def distance : ℝ := 150
def time : ℝ := 2
def speed_limit : ℝ := 60

theorem speed_above_limit : (distance / time) - speed_limit = 15 := by
  sorry

end speed_above_limit_l325_32587


namespace women_in_third_group_l325_32545

/-- Represents the work rate of a single person -/
structure WorkRate where
  rate : ℝ
  positive : rate > 0

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  women : ℕ

/-- Calculates the total work rate of a group -/
def totalWorkRate (m w : WorkRate) (group : WorkGroup) : ℝ :=
  group.men • m.rate + group.women • w.rate

theorem women_in_third_group 
  (m w : WorkRate)
  (group1 group2 group3 : WorkGroup) :
  totalWorkRate m w group1 = totalWorkRate m w group2 →
  group1.men = 3 →
  group1.women = 8 →
  group2.men = 6 →
  group2.women = 2 →
  group3.men = 4 →
  totalWorkRate m w group3 = 0.9285714285714286 * totalWorkRate m w group1 →
  group3.women = 5 := by
  sorry


end women_in_third_group_l325_32545


namespace consecutive_numbers_sum_l325_32535

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) = 60) → 
  ((n + 2) + (n + 3) + (n + 4) = 66) := by
  sorry

end consecutive_numbers_sum_l325_32535


namespace expression_simplification_l325_32565

theorem expression_simplification (x y : ℝ) : 1 - (2 - (3 - (4 - (5 - x)))) - y = 3 - (x + y) := by
  sorry

end expression_simplification_l325_32565


namespace suit_price_theorem_l325_32599

theorem suit_price_theorem (original_price : ℝ) : 
  (original_price * 1.25 * 0.75 = 150) → original_price = 160 := by
  sorry

end suit_price_theorem_l325_32599


namespace pencil_count_l325_32593

/-- The total number of pencils after adding more to an initial amount -/
def total_pencils (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 41 initial pencils and 30 added pencils, the total is 71 -/
theorem pencil_count : total_pencils 41 30 = 71 := by
  sorry

end pencil_count_l325_32593


namespace trig_identity_l325_32542

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10/3 := by
  sorry

end trig_identity_l325_32542


namespace shelf_fill_relation_l325_32594

/-- Represents the number of books needed to fill a shelf. -/
structure ShelfFill :=
  (A H S M F : ℕ)
  (distinct : A ≠ H ∧ A ≠ S ∧ A ≠ M ∧ A ≠ F ∧
              H ≠ S ∧ H ≠ M ∧ H ≠ F ∧
              S ≠ M ∧ S ≠ F ∧
              M ≠ F)
  (positive : A > 0 ∧ H > 0 ∧ S > 0 ∧ M > 0 ∧ F > 0)
  (history_thicker : H < A ∧ M < S)

/-- Theorem stating the relation between the number of books needed to fill the shelf. -/
theorem shelf_fill_relation (sf : ShelfFill) : sf.F = (sf.A * sf.F - sf.S * sf.H) / (sf.M - sf.H) :=
  sorry

end shelf_fill_relation_l325_32594


namespace complex_inequality_l325_32595

theorem complex_inequality : ∀ (i : ℂ), i^2 = -1 → Complex.abs (2 - i) > 2 * (i^4).re :=
fun i h =>
  sorry

end complex_inequality_l325_32595


namespace erasers_per_box_l325_32598

/-- Given that Jacqueline has 4 boxes of erasers and a total of 40 erasers,
    prove that there are 10 erasers in each box. -/
theorem erasers_per_box (total_erasers : ℕ) (num_boxes : ℕ) (h1 : total_erasers = 40) (h2 : num_boxes = 4) :
  total_erasers / num_boxes = 10 := by
  sorry

#check erasers_per_box

end erasers_per_box_l325_32598


namespace geese_count_l325_32550

/-- The number of ducks in the marsh -/
def ducks : ℕ := 37

/-- The total number of birds in the marsh -/
def total_birds : ℕ := 95

/-- The number of geese in the marsh -/
def geese : ℕ := total_birds - ducks

theorem geese_count : geese = 58 := by sorry

end geese_count_l325_32550


namespace triangle_inequality_l325_32501

/-- Given a triangle ABC with side lengths a, b, c, circumradius R, inradius r, and semiperimeter p,
    prove that (a / (p - a)) + (b / (p - b)) + (c / (p - c)) ≥ 3R / r,
    with equality if and only if the triangle is equilateral. -/
theorem triangle_inequality (a b c R r p : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
    (hR : R > 0) (hr : r > 0) (hp : p > 0) (h_semi : p = (a + b + c) / 2)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
    (a / (p - a)) + (b / (p - b)) + (c / (p - c)) ≥ 3 * R / r ∧
    ((a / (p - a)) + (b / (p - b)) + (c / (p - c)) = 3 * R / r ↔ a = b ∧ b = c) := by
  sorry

end triangle_inequality_l325_32501


namespace exam_questions_attempted_student_exam_result_l325_32510

theorem exam_questions_attempted (correct_score : ℕ) (wrong_penalty : ℕ) 
  (total_score : ℤ) (correct_answers : ℕ) : ℕ :=
  let wrong_answers := total_score - correct_score * correct_answers
  correct_answers + wrong_answers.toNat

-- Statement of the problem
theorem student_exam_result : 
  exam_questions_attempted 4 1 130 38 = 60 := by
  sorry

end exam_questions_attempted_student_exam_result_l325_32510


namespace water_added_to_alcohol_solution_l325_32515

/-- Proves that adding 5 liters of water to a 15-liter solution with 26% alcohol 
    results in a new solution with 19.5% alcohol -/
theorem water_added_to_alcohol_solution :
  let initial_volume : ℝ := 15
  let initial_alcohol_percentage : ℝ := 0.26
  let water_added : ℝ := 5
  let final_alcohol_percentage : ℝ := 0.195
  let initial_alcohol_volume := initial_volume * initial_alcohol_percentage
  let final_volume := initial_volume + water_added
  initial_alcohol_volume / final_volume = final_alcohol_percentage := by
  sorry


end water_added_to_alcohol_solution_l325_32515


namespace circle_intersection_l325_32538

/-- The equation of the circle C -/
def C (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 4*y + m = 0

/-- The equation of the line l -/
def l (x y : ℝ) : Prop := x + 2*y - 4 = 0

/-- Theorem stating when C represents a circle and the value of m when C intersects l -/
theorem circle_intersection :
  (∃ (m : ℝ), ∀ (x y : ℝ), C x y m → m < 5) ∧
  (∃ (m : ℝ), ∀ (x y : ℝ), C x y m → l x y → 
    ∃ (M N : ℝ × ℝ), C M.1 M.2 m ∧ C N.1 N.2 m ∧ l M.1 M.2 ∧ l N.1 N.2 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 = (4 / Real.sqrt 5)^2 → m = 4) :=
sorry

end circle_intersection_l325_32538


namespace coffee_ratio_is_two_to_one_l325_32586

/-- Represents the amount of coffee used for different strengths -/
structure CoffeeAmount where
  weak : ℕ
  strong : ℕ

/-- Calculates the ratio of strong to weak coffee -/
def coffeeRatio (amount : CoffeeAmount) : ℚ :=
  amount.strong / amount.weak

/-- Theorem stating the ratio of strong to weak coffee is 2:1 -/
theorem coffee_ratio_is_two_to_one :
  ∃ (amount : CoffeeAmount),
    amount.weak + amount.strong = 36 ∧
    amount.weak = 12 ∧
    coffeeRatio amount = 2 := by
  sorry

end coffee_ratio_is_two_to_one_l325_32586


namespace percentage_decrease_l325_32569

theorem percentage_decrease (x y z : ℝ) : 
  x = 1.3 * y ∧ x = 0.65 * z → y = 0.5 * z :=
by sorry

end percentage_decrease_l325_32569


namespace distance_to_line_l325_32507

/-- The distance from a point in polar coordinates to a line in polar form -/
def distance_polar_to_line (m : ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  |m - 2|

/-- The theorem stating the distance from the point (m, π/3) to the line ρcos(θ - π/3) = 2 -/
theorem distance_to_line (m : ℝ) (h : m > 0) :
  distance_polar_to_line m (fun ρ θ ↦ ρ * Real.cos (θ - Real.pi / 3) = 2) = |m - 2| := by
  sorry

end distance_to_line_l325_32507


namespace hamster_lifespan_difference_l325_32558

/-- Represents the lifespans of a hamster, bat, and frog. -/
structure AnimalLifespans where
  hamster : ℕ
  bat : ℕ
  frog : ℕ

/-- The conditions of the problem. -/
def problemConditions (a : AnimalLifespans) : Prop :=
  a.bat = 10 ∧
  a.frog = 4 * a.hamster ∧
  a.hamster + a.bat + a.frog = 30

/-- The theorem to be proved. -/
theorem hamster_lifespan_difference (a : AnimalLifespans) 
  (h : problemConditions a) : a.bat - a.hamster = 6 := by
  sorry

#check hamster_lifespan_difference

end hamster_lifespan_difference_l325_32558


namespace range_of_a_max_value_sum_of_roots_l325_32534

-- Define the function f
def f (x : ℝ) : ℝ := |x - 4| - |x + 2|

-- Part 1
theorem range_of_a (a : ℝ) :
  (∀ x, f x - a^2 + 5*a ≥ 0) → 2 ≤ a ∧ a ≤ 3 :=
sorry

-- Part 2
theorem max_value_sum_of_roots (M a b c : ℝ) :
  (∀ x, f x ≤ M) →
  a > 0 → b > 0 → c > 0 →
  a + b + c = M →
  (∃ (max_val : ℝ), ∀ a' b' c' : ℝ,
    a' > 0 → b' > 0 → c' > 0 →
    a' + b' + c' = M →
    Real.sqrt (a' + 1) + Real.sqrt (b' + 2) + Real.sqrt (c' + 3) ≤ max_val ∧
    max_val = 6) :=
sorry

end range_of_a_max_value_sum_of_roots_l325_32534


namespace smallest_winning_k_l325_32557

/-- Represents a square on the game board --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Represents the game state --/
structure GameState where
  board : Square → Option Char
  mike_moves : Nat
  harry_moves : Nat

/-- Checks if a sequence forms a winning pattern --/
def is_winning_sequence (s : List Char) : Bool :=
  s = ['H', 'M', 'M'] || s = ['M', 'M', 'H']

/-- Checks if there's a winning sequence on the board --/
def has_winning_sequence (state : GameState) : Bool :=
  sorry

/-- Represents a strategy for Mike --/
def MikeStrategy := Nat → List Square

/-- Represents a strategy for Harry --/
def HarryStrategy := GameState → List Square

/-- Simulates a game with given strategies --/
def play_game (k : Nat) (mike_strat : MikeStrategy) (harry_strat : HarryStrategy) : Bool :=
  sorry

/-- Defines what it means for Mike to have a winning strategy --/
def mike_has_winning_strategy (k : Nat) : Prop :=
  ∃ (mike_strat : MikeStrategy), ∀ (harry_strat : HarryStrategy), 
    play_game k mike_strat harry_strat = true

/-- The main theorem stating that 16 is the smallest k for which Mike has a winning strategy --/
theorem smallest_winning_k : 
  (mike_has_winning_strategy 16) ∧ 
  (∀ k < 16, ¬(mike_has_winning_strategy k)) :=
sorry

end smallest_winning_k_l325_32557


namespace sin_sum_to_product_l325_32543

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x :=
by sorry

end sin_sum_to_product_l325_32543


namespace parallel_transitivity_l325_32560

-- Define a type for lines
variable (Line : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem parallel_transitivity (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 := by
  sorry

end parallel_transitivity_l325_32560


namespace arithmetic_sequence_ratio_l325_32562

theorem arithmetic_sequence_ratio (x y d₁ d₂ : ℝ) (h₁ : d₁ ≠ 0) (h₂ : d₂ ≠ 0) : 
  (x + 4 * d₁ = y) → (x + 5 * d₂ = y) → d₁ / d₂ = 5 / 4 := by
  sorry

end arithmetic_sequence_ratio_l325_32562


namespace rectangular_block_height_l325_32585

/-- The height of a rectangular block with given volume and base area -/
theorem rectangular_block_height (volume : ℝ) (base_area : ℝ) (height : ℝ) : 
  volume = 120 → base_area = 24 → volume = base_area * height → height = 5 := by
  sorry

end rectangular_block_height_l325_32585


namespace monday_grading_percentage_l325_32536

/-- The percentage of exams graded on Monday -/
def monday_percentage : ℝ := 40

/-- The total number of exams -/
def total_exams : ℕ := 120

/-- The percentage of remaining exams graded on Tuesday -/
def tuesday_percentage : ℝ := 75

/-- The number of exams left to grade after Tuesday -/
def exams_left : ℕ := 12

theorem monday_grading_percentage :
  monday_percentage = 40 ∧
  (total_exams : ℝ) - (monday_percentage / 100) * total_exams -
    (tuesday_percentage / 100) * ((100 - monday_percentage) / 100 * total_exams) = exams_left :=
by sorry

end monday_grading_percentage_l325_32536


namespace nested_expression_sum_l325_32568

theorem nested_expression_sum : 
  4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4*(1 + 4)))))))) = 1398100 := by
  sorry

end nested_expression_sum_l325_32568


namespace evaluate_expression_l325_32583

theorem evaluate_expression : (4 + 6 + 7) / 3 - 2 / 3 = 5 := by
  sorry

end evaluate_expression_l325_32583


namespace geometric_sequence_product_l325_32525

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 4 = 5 →
  a 8 = 6 →
  a 2 * a 10 = 30 := by
sorry

end geometric_sequence_product_l325_32525


namespace two_person_subcommittees_of_six_l325_32554

/-- The number of two-person sub-committees from a six-person committee -/
def two_person_subcommittees (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Theorem: The number of two-person sub-committees from a six-person committee is 15 -/
theorem two_person_subcommittees_of_six :
  two_person_subcommittees 6 = 15 := by
  sorry

end two_person_subcommittees_of_six_l325_32554


namespace henry_games_count_l325_32574

theorem henry_games_count :
  ∀ (h n l : ℕ),
    h = 3 * n →                 -- Henry had 3 times as many games as Neil initially
    h = 2 * l →                 -- Henry had 2 times as many games as Linda initially
    n = 7 →                     -- Neil had 7 games initially
    l = 7 →                     -- Linda had 7 games initially
    h - 10 = 4 * (n + 6) →      -- After giving games, Henry has 4 times more games than Neil
    h = 62                      -- Henry originally had 62 games
  := by sorry

end henry_games_count_l325_32574


namespace problem_solution_l325_32591

noncomputable def problem (a b c k x y z : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ k ≠ 0 ∧
  x * y / (x + y) = a ∧
  x * z / (x + z) = b ∧
  y * z / (y + z) = c ∧
  x * y * z / (x + y + z) = k

theorem problem_solution (a b c k x y z : ℝ) (h : problem a b c k x y z) :
  x = 2 * k * a * b / (a * b + b * c - a * c) := by
  sorry

end problem_solution_l325_32591


namespace unique_student_count_l325_32504

theorem unique_student_count :
  ∃! n : ℕ, n < 400 ∧ n % 17 = 15 ∧ n % 19 = 10 ∧ n = 219 :=
by sorry

end unique_student_count_l325_32504


namespace intersection_and_min_area_l325_32544

noncomputable section

-- Define the hyperbola C₁
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (2 * a^2) = 1

-- Define the parabola C₂
def C₂ (a : ℝ) (x y : ℝ) : Prop := y^2 = -4 * Real.sqrt 3 * a * x

-- Define the focus F₁
def F₁ (a : ℝ) : ℝ × ℝ := (-Real.sqrt 3 * a, 0)

-- Define a chord through F₁
def chord_through_F₁ (a k : ℝ) (x y : ℝ) : Prop := y = k * (x + Real.sqrt 3 * a)

-- Define the area of triangle AOB
def area_AOB (a k : ℝ) : ℝ := 6 * a^2 * Real.sqrt (1 + 1/k^2)

theorem intersection_and_min_area (a : ℝ) (h : a > 0) :
  (∃! (p q : ℝ × ℝ), p ≠ q ∧ C₁ a p.1 p.2 ∧ C₂ a p.1 p.2 ∧ C₁ a q.1 q.2 ∧ C₂ a q.1 q.2) ∧
  (∀ k : ℝ, area_AOB a k ≥ 6 * a^2) ∧
  (∃ k : ℝ, chord_through_F₁ a k (-Real.sqrt 3 * a) 0 ∧ area_AOB a k = 6 * a^2) :=
sorry

end intersection_and_min_area_l325_32544


namespace problem_solution_l325_32575

theorem problem_solution : ∃ x : ℝ, 4 * x - 4 = 2 * 4 + 20 ∧ x = 8 := by
  sorry

end problem_solution_l325_32575


namespace boat_rental_cost_l325_32540

theorem boat_rental_cost (students : ℕ) (boat_capacity : ℕ) (rental_fee : ℕ) 
  (h1 : students = 42)
  (h2 : boat_capacity = 6)
  (h3 : rental_fee = 125) :
  (((students + boat_capacity - 1) / boat_capacity) * rental_fee) = 875 :=
by
  sorry

#check boat_rental_cost

end boat_rental_cost_l325_32540


namespace paintings_distribution_l325_32506

theorem paintings_distribution (total_paintings : ℕ) (num_rooms : ℕ) (paintings_per_room : ℕ) :
  total_paintings = 32 →
  num_rooms = 4 →
  paintings_per_room = total_paintings / num_rooms →
  paintings_per_room = 8 := by
  sorry

end paintings_distribution_l325_32506


namespace pie_shop_revenue_l325_32556

/-- The revenue calculation for a pie shop --/
theorem pie_shop_revenue : 
  (price_per_slice : ℕ) → 
  (slices_per_pie : ℕ) → 
  (number_of_pies : ℕ) → 
  price_per_slice = 5 →
  slices_per_pie = 4 →
  number_of_pies = 9 →
  price_per_slice * slices_per_pie * number_of_pies = 180 := by
  sorry

end pie_shop_revenue_l325_32556


namespace family_income_theorem_l325_32582

theorem family_income_theorem (initial_members : ℕ) (new_average : ℝ) (deceased_income : ℝ) :
  initial_members = 4 →
  new_average = 650 →
  deceased_income = 990 →
  (initial_members - 1) * new_average + deceased_income = initial_members * 735 :=
by sorry

end family_income_theorem_l325_32582


namespace room_width_calculation_l325_32564

theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 8 →
  cost_per_sqm = 900 →
  total_cost = 34200 →
  width = total_cost / cost_per_sqm / length →
  width = 4.75 :=
by
  sorry

end room_width_calculation_l325_32564


namespace board_numbers_product_l325_32527

theorem board_numbers_product (a b c d e : ℤ) : 
  ({a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e} : Finset ℤ) = 
    {6, 9, 10, 13, 13, 14, 17, 17, 20, 21} →
  a * b * c * d * e = 4320 := by
sorry

end board_numbers_product_l325_32527


namespace unused_streetlights_l325_32563

theorem unused_streetlights (total_streetlights : ℕ) (num_squares : ℕ) (lights_per_square : ℕ) :
  total_streetlights = 200 →
  num_squares = 15 →
  lights_per_square = 12 →
  total_streetlights - (num_squares * lights_per_square) = 20 := by
  sorry

#check unused_streetlights

end unused_streetlights_l325_32563


namespace pablo_puzzle_completion_time_l325_32572

/-- The number of days it takes Pablo to complete all his puzzles -/
def days_to_complete_puzzles (pieces_per_hour : ℕ) (max_hours_per_day : ℕ) 
  (num_puzzles_300 : ℕ) (num_puzzles_500 : ℕ) : ℕ :=
  let total_pieces := num_puzzles_300 * 300 + num_puzzles_500 * 500
  let pieces_per_day := pieces_per_hour * max_hours_per_day
  (total_pieces + pieces_per_day - 1) / pieces_per_day

/-- Theorem stating that it takes Pablo 7 days to complete all his puzzles -/
theorem pablo_puzzle_completion_time :
  days_to_complete_puzzles 100 7 8 5 = 7 := by
  sorry

#eval days_to_complete_puzzles 100 7 8 5

end pablo_puzzle_completion_time_l325_32572


namespace hyperbola_relation_l325_32552

/-- Two hyperbolas M and N with the given properties -/
structure HyperbolaPair where
  /-- Eccentricity of hyperbola M -/
  e₁ : ℝ
  /-- Eccentricity of hyperbola N -/
  e₂ : ℝ
  /-- Half the length of the transverse axis of hyperbola N -/
  a : ℝ
  /-- Half the length of the conjugate axis of both hyperbolas -/
  b : ℝ
  /-- M and N are centered at the origin -/
  center_origin : True
  /-- Symmetric axes are coordinate axes -/
  symmetric_axes : True
  /-- Length of transverse axis of M is twice that of N -/
  transverse_axis_relation : True
  /-- Conjugate axes of M and N are equal -/
  conjugate_axis_equal : True
  /-- e₁ and e₂ are positive -/
  e₁_pos : e₁ > 0
  e₂_pos : e₂ > 0
  /-- a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- Definition of e₂ for hyperbola N -/
  e₂_def : e₂^2 = 1 + b^2 / a^2
  /-- Definition of e₁ for hyperbola M -/
  e₁_def : e₁^2 = 1 + b^2 / (4*a^2)

/-- The point (e₁, e₂) satisfies the equation of the hyperbola 4x²-y²=3 -/
theorem hyperbola_relation (h : HyperbolaPair) : 4 * h.e₁^2 - h.e₂^2 = 3 := by
  sorry

end hyperbola_relation_l325_32552


namespace cube_sum_reciprocal_l325_32521

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = 5) : x^3 + 1/x^3 = 110 := by
  sorry

end cube_sum_reciprocal_l325_32521


namespace smallest_divisor_sum_of_squares_l325_32502

theorem smallest_divisor_sum_of_squares (n : ℕ) : n ≥ 2 →
  (∃ a b : ℕ, 
    a > 1 ∧ 
    a ∣ n ∧ 
    (∀ d : ℕ, d > 1 → d ∣ n → d ≥ a) ∧
    b ∣ n ∧
    n = a^2 + b^2) →
  n = 8 ∨ n = 20 := by
sorry

end smallest_divisor_sum_of_squares_l325_32502


namespace min_sum_of_squares_l325_32549

theorem min_sum_of_squares (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (m : ℝ), (∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → a^2 + b^2 ≥ m) ∧ 
  (∃ (c d : ℝ), (c + 5)^2 + (d - 12)^2 = 14^2 ∧ c^2 + d^2 = m) ∧
  m = 1 :=
by sorry

end min_sum_of_squares_l325_32549


namespace area_relationship_l325_32522

/-- Represents a right triangle with a circumscribed circle -/
structure RightTriangleWithCircumcircle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right_triangle : side1^2 + side2^2 = hypotenuse^2
  side1_positive : side1 > 0
  side2_positive : side2 > 0

/-- The areas of the non-triangular regions in the circumcircle -/
structure CircumcircleAreas where
  A : ℝ
  B : ℝ
  C : ℝ
  C_largest : C ≥ A ∧ C ≥ B

/-- Theorem stating the relationship between areas A, B, and C -/
theorem area_relationship (triangle : RightTriangleWithCircumcircle)
    (areas : CircumcircleAreas) (h : triangle.side1 = 15 ∧ triangle.side2 = 36 ∧ triangle.hypotenuse = 39) :
    areas.A + areas.B + 270 = areas.C := by
  sorry

end area_relationship_l325_32522


namespace maria_bike_purchase_l325_32592

/-- The amount Maria needs to earn to buy a bike -/
def amount_to_earn (retail_price savings mother_contribution : ℕ) : ℕ :=
  retail_price - (savings + mother_contribution)

/-- Theorem: Maria needs to earn $230 to buy the bike -/
theorem maria_bike_purchase (retail_price savings mother_contribution : ℕ)
  (h1 : retail_price = 600)
  (h2 : savings = 120)
  (h3 : mother_contribution = 250) :
  amount_to_earn retail_price savings mother_contribution = 230 := by
  sorry

end maria_bike_purchase_l325_32592


namespace sixtieth_point_coordinates_l325_32516

/-- Represents a point with integer coordinates -/
structure Point where
  x : ℕ
  y : ℕ

/-- The sequence of points -/
def pointSequence : ℕ → Point := sorry

/-- The sum of x and y coordinates for the nth point -/
def coordinateSum (n : ℕ) : ℕ := (pointSequence n).x + (pointSequence n).y

/-- The row number for a given point in the sequence -/
def rowNumber (n : ℕ) : ℕ := sorry

/-- The property that the coordinate sum increases by 1 for every n points -/
axiom coordinate_sum_property (n : ℕ) :
  ∀ k, k > n → coordinateSum k = coordinateSum n + (rowNumber k - rowNumber n)

/-- The main theorem: The 60th point has coordinates (5,7) -/
theorem sixtieth_point_coordinates :
  pointSequence 60 = Point.mk 5 7 := by sorry

end sixtieth_point_coordinates_l325_32516


namespace negation_of_p_l325_32561

-- Define the proposition p
def p : Prop := ∃ m : ℝ, m > 0 ∧ ∃ x : ℝ, m * x^2 + x - 2*m = 0

-- State the theorem
theorem negation_of_p : ¬p ↔ ∀ m : ℝ, m > 0 → ∀ x : ℝ, m * x^2 + x - 2*m ≠ 0 := by sorry

end negation_of_p_l325_32561


namespace music_festival_group_formation_l325_32551

def total_friends : ℕ := 10
def musicians : ℕ := 4
def non_musicians : ℕ := 6
def group_size : ℕ := 4

theorem music_festival_group_formation :
  (Nat.choose total_friends group_size) - (Nat.choose non_musicians group_size) = 195 :=
sorry

end music_festival_group_formation_l325_32551


namespace g_behavior_at_infinity_l325_32555

-- Define the function g(x)
def g (x : ℝ) : ℝ := -3 * x^3 + 5 * x + 1

-- State the theorem
theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) := by
  sorry

end g_behavior_at_infinity_l325_32555


namespace intersection_complement_A_with_B_l325_32503

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {3, 4}

theorem intersection_complement_A_with_B :
  (U \ A) ∩ B = {4} := by sorry

end intersection_complement_A_with_B_l325_32503


namespace egg_sales_income_l325_32517

theorem egg_sales_income (num_hens : ℕ) (eggs_per_hen_per_week : ℕ) (price_per_dozen : ℕ) (num_weeks : ℕ) :
  num_hens = 10 →
  eggs_per_hen_per_week = 12 →
  price_per_dozen = 3 →
  num_weeks = 4 →
  (num_hens * eggs_per_hen_per_week * num_weeks / 12) * price_per_dozen = 120 := by
  sorry

#check egg_sales_income

end egg_sales_income_l325_32517


namespace more_students_than_rabbits_l325_32578

theorem more_students_than_rabbits : 
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 22
  let rabbits_per_classroom : ℕ := 2
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  total_students - total_rabbits = 100 := by
  sorry

end more_students_than_rabbits_l325_32578


namespace intersection_of_four_convex_sets_l325_32511

-- Define a type for points in a plane
variable {Point : Type}

-- Define a type for convex sets in a plane
variable {ConvexSet : Type}

-- Define a function to check if a point is in a convex set
variable (in_set : Point → ConvexSet → Prop)

-- Define a function to check if a set is convex
variable (is_convex : ConvexSet → Prop)

-- Define a function to represent the intersection of sets
variable (intersection : List ConvexSet → Set Point)

-- Theorem statement
theorem intersection_of_four_convex_sets
  (C1 C2 C3 C4 : ConvexSet)
  (convex1 : is_convex C1)
  (convex2 : is_convex C2)
  (convex3 : is_convex C3)
  (convex4 : is_convex C4)
  (intersect_three1 : (intersection [C1, C2, C3]).Nonempty)
  (intersect_three2 : (intersection [C1, C2, C4]).Nonempty)
  (intersect_three3 : (intersection [C1, C3, C4]).Nonempty)
  (intersect_three4 : (intersection [C2, C3, C4]).Nonempty) :
  (intersection [C1, C2, C3, C4]).Nonempty :=
sorry

end intersection_of_four_convex_sets_l325_32511


namespace total_earnings_after_seven_days_l325_32577

/- Define the prices of books -/
def fantasy_price : ℕ := 6
def literature_price : ℕ := fantasy_price / 2
def mystery_price : ℕ := 4

/- Define the daily sales quantities -/
def fantasy_sales : ℕ := 5
def literature_sales : ℕ := 8
def mystery_sales : ℕ := 3

/- Define the number of days -/
def days : ℕ := 7

/- Calculate daily earnings -/
def daily_earnings : ℕ := 
  fantasy_sales * fantasy_price + 
  literature_sales * literature_price + 
  mystery_sales * mystery_price

/- Theorem to prove -/
theorem total_earnings_after_seven_days : 
  daily_earnings * days = 462 := by sorry

end total_earnings_after_seven_days_l325_32577


namespace min_value_expression_l325_32514

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b + c) * (1 / (a + b) + 1 / (a + c) + 1 / (b + c)) ≥ 6 := by
sorry

end min_value_expression_l325_32514


namespace workshop_average_salary_l325_32520

theorem workshop_average_salary
  (total_workers : ℕ)
  (technicians : ℕ)
  (technician_salary : ℚ)
  (non_technician_salary : ℚ)
  (h1 : total_workers = 22)
  (h2 : technicians = 7)
  (h3 : technician_salary = 1000)
  (h4 : non_technician_salary = 780) :
  let non_technicians := total_workers - technicians
  let total_salary := technicians * technician_salary + non_technicians * non_technician_salary
  total_salary / total_workers = 850 := by
sorry

end workshop_average_salary_l325_32520


namespace compare_f_values_l325_32559

/-- Given 0 < a < 1, this function satisfies f(log_a x) = (a(x^2 - 1)) / (x(a^2 - 1)) for any x > 0 -/
noncomputable def f (a : ℝ) (t : ℝ) : ℝ := sorry

/-- Theorem: For 0 < a < 1, given function f and m > n > 0, we have f(1/n) > f(1/m) -/
theorem compare_f_values (a m n : ℝ) (ha : 0 < a) (ha' : a < 1) (hmn : m > n) (hn : n > 0) :
  f a (1/n) > f a (1/m) := by sorry

end compare_f_values_l325_32559


namespace diagonal_difference_bound_l325_32573

-- Define a cyclic quadrilateral
structure CyclicQuadrilateral :=
  (a b c d e f : ℝ)
  (cyclic : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0)
  (ptolemy : a * c + b * d = e * f)

-- State the theorem
theorem diagonal_difference_bound (q : CyclicQuadrilateral) :
  |q.e - q.f| ≤ |q.b - q.d| := by sorry

end diagonal_difference_bound_l325_32573


namespace rod_cutting_l325_32526

theorem rod_cutting (rod_length_m : ℕ) (piece_length_cm : ℕ) : 
  rod_length_m = 17 → piece_length_cm = 85 → (rod_length_m * 100) / piece_length_cm = 20 := by
  sorry

end rod_cutting_l325_32526


namespace product_of_fractions_and_powers_of_three_l325_32528

theorem product_of_fractions_and_powers_of_three (x : ℚ) : 
  x = (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 * (1/19683 : ℚ) * 59049 → 
  x = 243 := by
sorry

end product_of_fractions_and_powers_of_three_l325_32528


namespace carly_dog_grooming_l325_32590

theorem carly_dog_grooming (total_nails : ℕ) (three_legged_dogs : ℕ) :
  total_nails = 164 →
  three_legged_dogs = 3 →
  ∃ (total_dogs : ℕ),
    total_dogs * 4 * 4 - three_legged_dogs * 4 = total_nails ∧
    total_dogs = 11 :=
by sorry

end carly_dog_grooming_l325_32590


namespace composition_equals_26_l325_32541

-- Define the functions f and g
def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := 2 * x

-- Define the inverse functions
noncomputable def f_inv (x : ℝ) : ℝ := x - 3
noncomputable def g_inv (x : ℝ) : ℝ := x / 2

-- State the theorem
theorem composition_equals_26 : f (g_inv (f_inv (f_inv (g (f 23))))) = 26 := by sorry

end composition_equals_26_l325_32541


namespace expand_polynomial_l325_32553

/-- Proves the expansion of (12x^2 + 5x - 3) * (3x^3 + 2) -/
theorem expand_polynomial (x : ℝ) :
  (12 * x^2 + 5 * x - 3) * (3 * x^3 + 2) =
  36 * x^5 + 15 * x^4 - 9 * x^3 + 24 * x^2 + 10 * x - 6 := by
  sorry

end expand_polynomial_l325_32553


namespace rationalized_denominator_product_l325_32508

theorem rationalized_denominator_product (A B C : ℤ) : 
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = A + B * Real.sqrt C → A * B * C = 180 := by
  sorry

end rationalized_denominator_product_l325_32508


namespace quadratic_equation_necessary_not_sufficient_l325_32531

theorem quadratic_equation_necessary_not_sufficient :
  ∀ x : ℝ, 
    (x = 5 → x^2 - 4*x - 5 = 0) ∧ 
    ¬(x^2 - 4*x - 5 = 0 → x = 5) := by
  sorry

end quadratic_equation_necessary_not_sufficient_l325_32531


namespace james_recovery_time_l325_32519

/-- Calculates the total number of days before James can resume heavy lifting after an injury -/
def time_to_resume_heavy_lifting (
  initial_pain_duration : ℕ
  ) (healing_time_multiplier : ℕ
  ) (additional_caution_period : ℕ
  ) (light_exercises_duration : ℕ
  ) (potential_complication_duration : ℕ
  ) (moderate_intensity_duration : ℕ
  ) (transition_to_heavy_lifting : ℕ
  ) : ℕ :=
  let initial_healing_time := initial_pain_duration * healing_time_multiplier
  let total_initial_recovery := initial_healing_time + additional_caution_period
  let light_exercises_with_complication := light_exercises_duration + potential_complication_duration
  let total_before_transition := total_initial_recovery + light_exercises_with_complication + moderate_intensity_duration
  total_before_transition + transition_to_heavy_lifting

/-- Theorem stating that given the specific conditions, James will take 67 days to resume heavy lifting -/
theorem james_recovery_time : 
  time_to_resume_heavy_lifting 3 5 3 14 7 7 21 = 67 := by
  sorry

end james_recovery_time_l325_32519


namespace exist_consecutive_lucky_years_l325_32530

/-- Returns the first two digits of a four-digit number -/
def firstTwoDigits (n : ℕ) : ℕ := n / 100

/-- Returns the last two digits of a four-digit number -/
def lastTwoDigits (n : ℕ) : ℕ := n % 100

/-- Checks if a year is lucky -/
def isLuckyYear (year : ℕ) : Prop :=
  year % (firstTwoDigits year + lastTwoDigits year) = 0

/-- Theorem: There exist two consecutive lucky years -/
theorem exist_consecutive_lucky_years :
  ∃ (y : ℕ), 1000 ≤ y ∧ y < 9999 ∧ isLuckyYear y ∧ isLuckyYear (y + 1) := by
  sorry

end exist_consecutive_lucky_years_l325_32530


namespace missing_number_last_two_digits_l325_32597

def last_two_digits (n : ℕ) : ℕ := n % 100

def product_last_two_digits (nums : List ℕ) : ℕ :=
  last_two_digits (nums.foldl (λ acc x => last_two_digits (acc * last_two_digits x)) 1)

theorem missing_number_last_two_digits
  (h : product_last_two_digits [122, 123, 125, 129, x] = 50) :
  last_two_digits x = 1 :=
sorry

end missing_number_last_two_digits_l325_32597


namespace cubic_equation_root_l325_32567

theorem cubic_equation_root (a b : ℚ) : 
  ((-2 - 3 * Real.sqrt 3) ^ 3 + a * (-2 - 3 * Real.sqrt 3) ^ 2 + b * (-2 - 3 * Real.sqrt 3) + 49 = 0) → 
  a = -3/23 := by
sorry

end cubic_equation_root_l325_32567


namespace tiles_per_square_foot_l325_32539

def wall1_length : ℝ := 5
def wall1_width : ℝ := 8
def wall2_length : ℝ := 7
def wall2_width : ℝ := 8
def turquoise_cost : ℝ := 13
def purple_cost : ℝ := 11
def total_savings : ℝ := 768

theorem tiles_per_square_foot :
  let total_area := wall1_length * wall1_width + wall2_length * wall2_width
  let cost_difference := turquoise_cost - purple_cost
  let total_tiles := total_savings / cost_difference
  total_tiles / total_area = 4 := by sorry

end tiles_per_square_foot_l325_32539


namespace inequality_solution_set_l325_32580

-- Define the inequality
def inequality (x : ℝ) : Prop := (3 * x - 1) / (2 - x) ≥ 0

-- Define the solution set
def solution_set : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x < 2}

-- Theorem statement
theorem inequality_solution_set : 
  {x : ℝ | inequality x} = solution_set :=
sorry

end inequality_solution_set_l325_32580


namespace problem_statement_l325_32509

theorem problem_statement (a b c d : ℕ+) 
  (h1 : a ^ 5 = b ^ 4)
  (h2 : c ^ 3 = d ^ 2)
  (h3 : c - a = 19) :
  d - b = 757 := by
  sorry

end problem_statement_l325_32509


namespace one_fourths_in_three_eighths_l325_32547

theorem one_fourths_in_three_eighths (x : ℚ) : x = 3/8 → (x / (1/4)) = 3/2 := by
  sorry

end one_fourths_in_three_eighths_l325_32547


namespace jessica_red_marbles_l325_32576

theorem jessica_red_marbles (sandy_marbles : ℕ) (sandy_times_more : ℕ) :
  sandy_marbles = 144 →
  sandy_times_more = 4 →
  (sandy_marbles / sandy_times_more) / 12 = 3 :=
by
  sorry

end jessica_red_marbles_l325_32576


namespace quadratic_discriminant_l325_32512

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 - 9x + 4 -/
def a : ℝ := 5
def b : ℝ := -9
def c : ℝ := 4

theorem quadratic_discriminant :
  discriminant a b c = 1 := by sorry

end quadratic_discriminant_l325_32512


namespace line_passes_through_point_l325_32571

/-- Given a line y = 2x + b passing through the point (1, 2), prove that b = 0 -/
theorem line_passes_through_point (b : ℝ) : 
  (∀ x y : ℝ, y = 2 * x + b) → 2 = 2 * 1 + b → b = 0 := by
  sorry

end line_passes_through_point_l325_32571


namespace equation_describes_ellipse_l325_32513

-- Define the equation
def equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 12

-- Define the two fixed points
def point1 : ℝ × ℝ := (0, 2)
def point2 : ℝ × ℝ := (6, -4)

-- Theorem stating that the equation describes an ellipse
theorem equation_describes_ellipse :
  ∀ x y : ℝ, equation x y ↔ 
    (∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y ∧
      Real.sqrt ((p.1 - point1.1)^2 + (p.2 - point1.2)^2) +
      Real.sqrt ((p.1 - point2.1)^2 + (p.2 - point2.2)^2) = 12) :=
by sorry

end equation_describes_ellipse_l325_32513


namespace frog_population_difference_l325_32566

/-- Theorem: Given the conditions about frog populations in two lakes, prove that the percentage difference is 20%. -/
theorem frog_population_difference (lassie_frogs : ℕ) (total_frogs : ℕ) (P : ℝ) : 
  lassie_frogs = 45 →
  total_frogs = 81 →
  total_frogs = lassie_frogs + (lassie_frogs - P / 100 * lassie_frogs) →
  P = 20 := by
  sorry

end frog_population_difference_l325_32566


namespace matrix_product_R_S_l325_32529

def R : Matrix (Fin 3) (Fin 3) ℝ := !![0, -1, 0; 1, 0, 0; 0, 0, 1]

def S (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![d^2, d*c, d*b; d*c, c^2, c*b; d*b, c*b, b^2]

theorem matrix_product_R_S (b c d : ℝ) :
  R * S b c d = !![-(d*c), -(c^2), -(c*b); d^2, d*c, d*b; d*b, c*b, b^2] := by
  sorry

end matrix_product_R_S_l325_32529


namespace basketball_selection_probabilities_l325_32505

def shot_probability : ℚ := 2/3

def second_level_after_three_shots : ℚ := 8/27

def selected_probability : ℚ := 64/81

def selected_after_five_shots : ℚ := 16/81

theorem basketball_selection_probabilities :
  (2 * shot_probability * (1 - shot_probability) * shot_probability = second_level_after_three_shots) ∧
  (selected_after_five_shots / selected_probability = 1/4) := by
  sorry

end basketball_selection_probabilities_l325_32505


namespace count_special_numbers_is_324_l325_32524

/-- The count of 5-digit numbers beginning with 2 that have exactly three identical digits (which are not 2) -/
def count_special_numbers : ℕ :=
  4 * 9 * 9

/-- The theorem stating that the count of special numbers is 324 -/
theorem count_special_numbers_is_324 : count_special_numbers = 324 := by
  sorry

end count_special_numbers_is_324_l325_32524


namespace profit_loss_equality_l325_32570

/-- Given an article with cost price C, prove that if the profit at selling price S_p
    equals the loss when sold at $448, then the selling price for 30% profit is 1.30C. -/
theorem profit_loss_equality (C : ℝ) (S_p : ℝ) :
  S_p - C = C - 448 →
  ∃ (S_30 : ℝ), S_30 = 1.30 * C ∧ S_30 - C = 0.30 * C :=
by sorry

end profit_loss_equality_l325_32570


namespace fraction_decomposition_l325_32546

theorem fraction_decomposition (x : ℝ) (h1 : x ≠ 7) (h2 : x ≠ -6) :
  (3 * x + 5) / (x^2 - x - 42) = 2 / (x - 7) + 1 / (x + 6) := by
  sorry

end fraction_decomposition_l325_32546


namespace snow_depth_theorem_l325_32533

/-- Calculates the final snow depth after seven days given initial conditions and daily changes --/
def snow_depth_after_seven_days (initial_snow : Real) 
  (day2_snow : Real) (day2_compaction : Real)
  (daily_melt : Real) (day4_cleared : Real)
  (day5_multiplier : Real)
  (day6_melt : Real) (day6_accumulate : Real) : Real :=
  let day1 := initial_snow
  let day2 := day1 + day2_snow * (1 - day2_compaction)
  let day3 := day2 - daily_melt
  let day4 := day3 - daily_melt - day4_cleared
  let day5 := day4 - daily_melt + day5_multiplier * (day1 + day2_snow)
  let day6 := day5 - day6_melt + day6_accumulate
  day6

/-- The final snow depth after seven days is approximately 2.1667 feet --/
theorem snow_depth_theorem : 
  ∃ ε > 0, |snow_depth_after_seven_days 0.5 (8/12) 0.1 (1/12) (6/12) 1.5 (3/12) (4/12) - 2.1667| < ε :=
by sorry

end snow_depth_theorem_l325_32533


namespace find_a9_l325_32596

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a9 (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 3 = 5) (h3 : a 4 + a 8 = 22) : 
  ∃ x : ℝ, a 9 = x :=
sorry

end find_a9_l325_32596


namespace sum_in_base5_l325_32588

/-- Converts a number from base 4 to base 10 --/
def base4ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 5 --/
def base10ToBase5 (n : ℕ) : ℕ := sorry

/-- Represents a number in base 5 --/
structure Base5 (n : ℕ) where
  value : ℕ
  isBase5 : value < 5^n

theorem sum_in_base5 :
  let a := base4ToBase10 203
  let b := base4ToBase10 112
  let c := base4ToBase10 321
  let sum := a + b + c
  base10ToBase5 sum = 2222 :=
sorry

end sum_in_base5_l325_32588


namespace difference_of_squares_factorization_l325_32523

theorem difference_of_squares_factorization (x : ℝ) : 
  x^2 - 4 = (x + 2) * (x - 2) := by sorry

end difference_of_squares_factorization_l325_32523


namespace initial_tagged_fish_l325_32548

/-- The number of fish initially caught and tagged -/
def T : ℕ := sorry

/-- The total number of fish in the pond -/
def N : ℕ := 800

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 40

/-- The number of tagged fish in the second catch -/
def tagged_in_second : ℕ := 2

theorem initial_tagged_fish :
  (T : ℚ) / N = tagged_in_second / second_catch ∧ T = 40 := by sorry

end initial_tagged_fish_l325_32548


namespace exists_line_intersecting_four_circles_l325_32537

/-- Represents a circle with a given diameter -/
structure Circle where
  diameter : ℝ

/-- Represents a configuration of circles in a unit square -/
structure CircleConfiguration where
  circles : List Circle
  sum_of_circumferences_eq_10 : (circles.map (fun c => c.diameter * Real.pi)).sum = 10

/-- Main theorem: If the sum of circumferences of circles in a unit square is 10,
    then there exists a line intersecting at least 4 of these circles -/
theorem exists_line_intersecting_four_circles (config : CircleConfiguration) :
  ∃ (line : ℝ → ℝ → Prop), (∃ (intersected_circles : List Circle),
    intersected_circles.length ≥ 4 ∧
    ∀ c ∈ intersected_circles, c ∈ config.circles ∧
    ∃ (x y : ℝ), x ∈ Set.Icc 0 1 ∧ y ∈ Set.Icc 0 1 ∧ line x y) :=
sorry

end exists_line_intersecting_four_circles_l325_32537


namespace smallest_covering_circle_l325_32589

-- Define the plane region
def PlaneRegion (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + 2*y - 4 ≤ 0

-- Define the circle equation
def Circle (a b r : ℝ) (x y : ℝ) : Prop :=
  (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem smallest_covering_circle :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ x y, PlaneRegion x y → C x y) ∧
    (∀ D : ℝ → ℝ → Prop, (∀ x y, PlaneRegion x y → D x y) → 
      ∃ a b r, C = Circle a b r ∧ 
      ∀ a' b' r', D = Circle a' b' r' → r ≤ r') ∧
    C = Circle 2 1 (Real.sqrt 5) :=
sorry

end smallest_covering_circle_l325_32589


namespace election_total_votes_l325_32518

/-- Represents an election between two candidates -/
structure Election where
  total_votes : ℕ
  invalid_percent : ℚ
  b_votes : ℕ
  a_excess_percent : ℚ

/-- The election satisfies the given conditions -/
def valid_election (e : Election) : Prop :=
  e.invalid_percent = 1/5 ∧
  e.a_excess_percent = 3/20 ∧
  e.b_votes = 2184 ∧
  (e.total_votes : ℚ) * (1 - e.invalid_percent) = 
    (e.b_votes : ℚ) + (e.b_votes : ℚ) + e.total_votes * e.a_excess_percent

theorem election_total_votes (e : Election) (h : valid_election e) : 
  e.total_votes = 6720 := by
  sorry

#check election_total_votes

end election_total_votes_l325_32518


namespace butterfly_black_dots_l325_32532

/-- The number of black dots per butterfly -/
def blackDotsPerButterfly (totalButterflies : ℕ) (totalBlackDots : ℕ) : ℕ :=
  totalBlackDots / totalButterflies

/-- Theorem stating that each butterfly has 12 black dots -/
theorem butterfly_black_dots :
  blackDotsPerButterfly 397 4764 = 12 := by
  sorry

end butterfly_black_dots_l325_32532


namespace tom_gathering_plates_l325_32500

/-- The number of plates used during a multi-day stay with multiple meals per day -/
def plates_used (people : ℕ) (days : ℕ) (meals_per_day : ℕ) (courses_per_meal : ℕ) (plates_per_course : ℕ) : ℕ :=
  people * days * meals_per_day * courses_per_meal * plates_per_course

/-- Theorem: Given the conditions from Tom's gathering, the total number of plates used is 1728 -/
theorem tom_gathering_plates :
  plates_used 12 6 4 3 2 = 1728 := by
  sorry

end tom_gathering_plates_l325_32500


namespace complement_M_inter_N_l325_32581

def M : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}
def N : Set ℝ := {x : ℝ | |x - 2| ≤ 1}

theorem complement_M_inter_N : 
  (Set.compl M) ∩ N = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end complement_M_inter_N_l325_32581
