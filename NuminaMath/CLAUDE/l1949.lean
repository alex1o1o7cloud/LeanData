import Mathlib

namespace max_cube_sum_four_squares_l1949_194934

theorem max_cube_sum_four_squares {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = 4) :
  a^3 + b^3 + c^3 + d^3 ≤ 8 ∧ ∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ a₀^3 + b₀^3 + c₀^3 + d₀^3 = 8 :=
by sorry

end max_cube_sum_four_squares_l1949_194934


namespace min_value_of_2a_plus_1_l1949_194941

theorem min_value_of_2a_plus_1 (a : ℝ) (h : 9*a^2 + 7*a + 5 = 2) : 
  ∃ (min : ℝ), min = -1 ∧ ∀ (x : ℝ), 9*x^2 + 7*x + 5 = 2 → 2*x + 1 ≥ min := by
sorry

end min_value_of_2a_plus_1_l1949_194941


namespace rock_paper_scissors_winning_probability_l1949_194988

/-- Represents the possible outcomes of a single round of Rock, Paper, Scissors -/
inductive RockPaperScissorsOutcome
  | Win
  | Lose
  | Draw

/-- Represents a two-player game of Rock, Paper, Scissors -/
structure RockPaperScissors where
  player1 : String
  player2 : String

/-- The probability of winning for each player in Rock, Paper, Scissors -/
def winningProbability (game : RockPaperScissors) : ℚ :=
  1 / 3

/-- Theorem: The probability of winning for each player in Rock, Paper, Scissors is 1/3 -/
theorem rock_paper_scissors_winning_probability (game : RockPaperScissors) :
  winningProbability game = 1 / 3 := by
  sorry

end rock_paper_scissors_winning_probability_l1949_194988


namespace simplify_expression_l1949_194966

theorem simplify_expression (x : ℝ) : (2*x)^5 - (5*x)*(x^4) = 27*x^5 := by
  sorry

end simplify_expression_l1949_194966


namespace union_of_A_and_complement_of_B_l1949_194918

-- Define the set A
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define the set B
def B : Set ℝ := {x | x * (x - 3) < 0}

-- Define the result set
def result : Set ℝ := {x | x ≤ 2 ∨ x ≥ 3}

-- Theorem statement
theorem union_of_A_and_complement_of_B : A ∪ (Set.univ \ B) = result := by
  sorry

end union_of_A_and_complement_of_B_l1949_194918


namespace union_of_M_and_N_l1949_194901

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 3}
def N : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x > -1} := by sorry

end union_of_M_and_N_l1949_194901


namespace total_cinnamon_swirls_l1949_194952

/-- The number of people eating cinnamon swirls -/
def num_people : ℕ := 3

/-- The number of pieces Jane ate -/
def janes_pieces : ℕ := 4

/-- Theorem: The total number of cinnamon swirl pieces prepared is 12 -/
theorem total_cinnamon_swirls :
  ∀ (pieces_per_person : ℕ),
  (pieces_per_person = janes_pieces) →
  (num_people * pieces_per_person = 12) :=
by sorry

end total_cinnamon_swirls_l1949_194952


namespace little_john_money_distribution_l1949_194911

def problem (initial_amount : ℚ) (sweets_cost : ℚ) (num_friends : ℕ) (remaining_amount : ℚ) : Prop :=
  let total_spent : ℚ := initial_amount - remaining_amount
  let amount_given_away : ℚ := total_spent - sweets_cost
  let amount_per_friend : ℚ := amount_given_away / num_friends
  amount_per_friend = 1

theorem little_john_money_distribution :
  problem 7.1 1.05 2 4.05 := by
  sorry

end little_john_money_distribution_l1949_194911


namespace number_exceeding_percentage_l1949_194903

theorem number_exceeding_percentage (x : ℝ) : x = 60 ↔ x = 0.12 * x + 52.8 := by
  sorry

end number_exceeding_percentage_l1949_194903


namespace function_is_negation_l1949_194961

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g x + y) = g x + g (g y + g (-x)) - x

/-- The main theorem stating that g(x) = -x for all x -/
theorem function_is_negation (g : ℝ → ℝ) (h : FunctionalEquation g) :
  ∀ x : ℝ, g x = -x :=
sorry

end function_is_negation_l1949_194961


namespace minimum_value_sum_reciprocals_l1949_194963

theorem minimum_value_sum_reciprocals (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (sum_eq_three : a + b + c = 3) :
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a)) ≥ 1 :=
sorry

end minimum_value_sum_reciprocals_l1949_194963


namespace smallest_k_cube_sum_multiple_360_k_38_cube_sum_multiple_360_smallest_k_is_38_l1949_194953

theorem smallest_k_cube_sum_multiple_360 : 
  ∀ k : ℕ, k > 0 → (k * (k + 1) / 2)^2 % 360 = 0 → k ≥ 38 :=
by sorry

theorem k_38_cube_sum_multiple_360 : 
  (38 * (38 + 1) / 2)^2 % 360 = 0 :=
by sorry

theorem smallest_k_is_38 :
  ∀ k : ℕ, k > 0 → (k * (k + 1) / 2)^2 % 360 = 0 → k ≥ 38 ∧ 
  (38 * (38 + 1) / 2)^2 % 360 = 0 :=
by sorry

end smallest_k_cube_sum_multiple_360_k_38_cube_sum_multiple_360_smallest_k_is_38_l1949_194953


namespace john_lawyer_payment_l1949_194998

/-- Calculates John's payment for lawyer fees --/
def johnPayment (upfrontFee courtHours hourlyRate prepTimeFactor paperworkFee transportCost : ℕ) : ℕ :=
  let totalHours := courtHours * (1 + prepTimeFactor)
  let totalFee := upfrontFee + (totalHours * hourlyRate) + paperworkFee + transportCost
  totalFee / 2

theorem john_lawyer_payment :
  johnPayment 1000 50 100 2 500 300 = 8400 := by
  sorry

end john_lawyer_payment_l1949_194998


namespace total_ages_l1949_194991

/-- Given that Gabriel is 3 years younger than Frank and Frank is 10 years old,
    prove that the total of their ages is 17. -/
theorem total_ages (frank_age : ℕ) (gabriel_age : ℕ) : 
  frank_age = 10 → gabriel_age = frank_age - 3 → frank_age + gabriel_age = 17 := by
  sorry

end total_ages_l1949_194991


namespace salary_calculation_l1949_194942

theorem salary_calculation (food_fraction : Rat) (rent_fraction : Rat) (clothes_fraction : Rat) 
  (savings_fraction : Rat) (tax_fraction : Rat) (remaining_amount : ℝ) :
  food_fraction = 1/5 →
  rent_fraction = 1/10 →
  clothes_fraction = 3/5 →
  savings_fraction = 1/20 →
  tax_fraction = 1/8 →
  remaining_amount = 18000 →
  ∃ S : ℝ, (7/160 : ℝ) * S = remaining_amount :=
by
  sorry

end salary_calculation_l1949_194942


namespace max_value_of_z_l1949_194925

-- Define the objective function
def z (x y : ℝ) : ℝ := 3 * x + 2 * y

-- Define the feasible region
def feasible_region (x y : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ x + y ≤ 4

-- Theorem statement
theorem max_value_of_z :
  ∃ (x y : ℝ), feasible_region x y ∧
  ∀ (x' y' : ℝ), feasible_region x' y' → z x y ≥ z x' y' ∧
  z x y = 12 :=
sorry

end max_value_of_z_l1949_194925


namespace minimum_value_problem_l1949_194996

theorem minimum_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  y / x + 4 / y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 1 ∧ y₀ / x₀ + 4 / y₀ = 8 := by
  sorry

end minimum_value_problem_l1949_194996


namespace point_b_value_l1949_194909

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Represents the distance between two points on a number line -/
def distance (p q : Point) : ℝ := |p.value - q.value|

/-- Given points A and B on a number line, where A represents -2 and B is 5 units away from A,
    B must represent either -7 or 3. -/
theorem point_b_value (A B : Point) :
  A.value = -2 ∧ distance A B = 5 → B.value = -7 ∨ B.value = 3 := by
  sorry

end point_b_value_l1949_194909


namespace rectangle_strip_count_l1949_194992

theorem rectangle_strip_count 
  (outer_perimeter : ℕ) 
  (hole_perimeter : ℕ) 
  (horizontal_strips : ℕ) : 
  outer_perimeter = 50 → 
  hole_perimeter = 32 → 
  horizontal_strips = 20 → 
  ∃ (vertical_strips : ℕ), vertical_strips = 21 :=
by
  sorry

end rectangle_strip_count_l1949_194992


namespace fencing_requirement_l1949_194900

/-- Given a rectangular field with area 60 sq. feet and one side 20 feet,
    prove that the sum of the other three sides is 26 feet. -/
theorem fencing_requirement (length width : ℝ) : 
  length * width = 60 →
  length = 20 →
  length + 2 * width = 26 := by
  sorry

end fencing_requirement_l1949_194900


namespace museum_trip_total_l1949_194983

theorem museum_trip_total (first_bus second_bus third_bus fourth_bus : ℕ) : 
  first_bus = 12 →
  second_bus = 2 * first_bus →
  third_bus = second_bus - 6 →
  fourth_bus = first_bus + 9 →
  first_bus + second_bus + third_bus + fourth_bus = 75 := by
  sorry

end museum_trip_total_l1949_194983


namespace students_in_different_clubs_l1949_194981

/-- The number of clubs in the school -/
def num_clubs : ℕ := 3

/-- The probability of a student joining any specific club -/
def prob_join_club : ℚ := 1 / num_clubs

/-- The probability of two students joining different clubs -/
def prob_different_clubs : ℚ := 2 / 3

theorem students_in_different_clubs :
  prob_different_clubs = 1 - (num_clubs : ℚ) * prob_join_club * prob_join_club := by
  sorry

end students_in_different_clubs_l1949_194981


namespace hyperbola_equation_l1949_194975

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ  -- Half-length of the transverse axis
  b : ℝ  -- Half-length of the conjugate axis
  c : ℝ  -- Focal distance

/-- The standard equation of a hyperbola -/
def standardEquation (h : Hyperbola) (x y : ℝ) : Prop :=
  y^2 / h.a^2 - x^2 / h.b^2 = 1

/-- Theorem: Given a hyperbola with specific properties, its standard equation is y²/4 - x²/4 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
  (vertex_condition : h.a = 2)
  (axis_sum_condition : 2 * h.a + 2 * h.b = Real.sqrt 2 * 2 * h.c) :
  standardEquation h x y ↔ y^2 / 4 - x^2 / 4 = 1 := by
  sorry

end hyperbola_equation_l1949_194975


namespace vector_sum_norm_equality_implies_parallel_l1949_194980

/-- Given two non-zero vectors a and b, if |a + b| = |a| - |b|, then a and b are parallel -/
theorem vector_sum_norm_equality_implies_parallel
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (a b : V) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ‖a + b‖ = ‖a‖ - ‖b‖) :
  ∃ (k : ℝ), a = k • b :=
sorry

end vector_sum_norm_equality_implies_parallel_l1949_194980


namespace system_solution_l1949_194950

theorem system_solution (x y z : ℝ) : 
  (x^3 + y^3 = 3*y + 3*z + 4 ∧
   y^3 + z^3 = 3*z + 3*x + 4 ∧
   x^3 + z^3 = 3*x + 3*y + 4) ↔ 
  ((x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = -1 ∧ y = -1 ∧ z = -1)) :=
by sorry

end system_solution_l1949_194950


namespace max_a_value_l1949_194920

open Real

theorem max_a_value (e : ℝ) (h_e : e = exp 1) :
  let a_max := 1/2 + log 2/2 - e
  ∀ a : ℝ, (∀ x : ℝ, x ∈ Set.Icc (1/e) 2 → (a + e) * x - 1 - log x ≤ 0) →
  a ≤ a_max ∧
  ∃ x : ℝ, x ∈ Set.Icc (1/e) 2 ∧ (a_max + e) * x - 1 - log x = 0 :=
sorry

end max_a_value_l1949_194920


namespace equal_roots_condition_l1949_194958

/-- 
If the quadratic equation 2x^2 - ax + 2 = 0 has two equal real roots, 
then a = 4 or a = -4
-/
theorem equal_roots_condition (a : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - a * x + 2 = 0 ∧ 
   (∀ y : ℝ, 2 * y^2 - a * y + 2 = 0 → y = x)) → 
  (a = 4 ∨ a = -4) := by
sorry

end equal_roots_condition_l1949_194958


namespace fraction_inequality_solution_l1949_194913

theorem fraction_inequality_solution (x : ℝ) : 
  x ≠ 3 → (x * (x + 1) / (x - 3)^2 ≥ 9 ↔ 
    (2.13696 ≤ x ∧ x < 3) ∨ (3 < x ∧ x ≤ 4.73804)) :=
by sorry

end fraction_inequality_solution_l1949_194913


namespace three_nap_simultaneously_l1949_194955

-- Define the type for mathematicians
def Mathematician := Fin 5

-- Define the type for nap times
variable {T : Type*}

-- Define the nap function that assigns two nap times to each mathematician
variable (nap : Mathematician → Fin 2 → T)

-- Define the property that any two mathematicians share a nap time
variable (share_nap : ∀ m1 m2 : Mathematician, m1 ≠ m2 → ∃ t : T, (∃ i : Fin 2, nap m1 i = t) ∧ (∃ j : Fin 2, nap m2 j = t))

-- Theorem statement
theorem three_nap_simultaneously :
  ∃ t : T, ∃ m1 m2 m3 : Mathematician, m1 ≠ m2 ∧ m2 ≠ m3 ∧ m1 ≠ m3 ∧
  (∃ i j k : Fin 2, nap m1 i = t ∧ nap m2 j = t ∧ nap m3 k = t) :=
sorry

end three_nap_simultaneously_l1949_194955


namespace joe_running_speed_l1949_194959

/-- Proves that Joe's running speed is 16 km/h given the problem conditions --/
theorem joe_running_speed : 
  ∀ (joe_speed pete_speed : ℝ),
  joe_speed = 2 * pete_speed →  -- Joe runs twice as fast as Pete
  (joe_speed + pete_speed) * (40 / 60) = 16 →  -- Total distance after 40 minutes
  joe_speed = 16 := by
  sorry

end joe_running_speed_l1949_194959


namespace solution_set_part1_range_of_a_l1949_194974

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1: Solution set when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end solution_set_part1_range_of_a_l1949_194974


namespace initial_books_eq_sold_plus_unsold_l1949_194969

/-- The number of books Ali had initially --/
def initial_books : ℕ := sorry

/-- The number of books Ali sold on Monday --/
def monday_sales : ℕ := 60

/-- The number of books Ali sold on Tuesday --/
def tuesday_sales : ℕ := 10

/-- The number of books Ali sold on Wednesday --/
def wednesday_sales : ℕ := 20

/-- The number of books Ali sold on Thursday --/
def thursday_sales : ℕ := 44

/-- The number of books Ali sold on Friday --/
def friday_sales : ℕ := 66

/-- The number of books not sold --/
def unsold_books : ℕ := 600

/-- Theorem stating that the initial number of books is equal to the sum of books sold on each day plus the number of books not sold --/
theorem initial_books_eq_sold_plus_unsold :
  initial_books = monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales + unsold_books := by
  sorry

end initial_books_eq_sold_plus_unsold_l1949_194969


namespace geometric_sequence_problem_l1949_194908

/-- Given a geometric sequence {aₙ} with a₁ = 1/16 and a₃a₇ = 2a₅ - 1, prove that a₃ = 1/4. -/
theorem geometric_sequence_problem (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  a 1 = 1 / 16 →
  a 3 * a 7 = 2 * a 5 - 1 →
  a 3 = 1 / 4 := by
sorry

end geometric_sequence_problem_l1949_194908


namespace added_number_after_doubling_l1949_194957

theorem added_number_after_doubling (original : ℕ) (added : ℕ) : 
  original = 9 → 3 * (2 * original + added) = 72 → added = 6 := by
  sorry

end added_number_after_doubling_l1949_194957


namespace tuesday_toys_bought_l1949_194987

/-- The number of dog toys Daisy had on Monday -/
def monday_toys : ℕ := 5

/-- The number of dog toys Daisy had left on Tuesday after losing some -/
def tuesday_remaining : ℕ := 3

/-- The number of dog toys Daisy's owner bought on Wednesday -/
def wednesday_new : ℕ := 5

/-- The total number of dog toys Daisy would have if all lost toys were found -/
def total_if_found : ℕ := 13

/-- The number of dog toys Daisy's owner bought on Tuesday -/
def tuesday_new : ℕ := total_if_found - tuesday_remaining - wednesday_new

theorem tuesday_toys_bought :
  tuesday_new = 5 :=
by sorry

end tuesday_toys_bought_l1949_194987


namespace parallel_vectors_x_value_l1949_194945

/-- Given vectors a, b, and c in ℝ², prove that if a + b is parallel to c, then x = -5 -/
theorem parallel_vectors_x_value (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![2*x, x]
  let c : Fin 2 → ℝ := ![3, 1]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • c) →
  x = -5 := by
sorry


end parallel_vectors_x_value_l1949_194945


namespace parallel_vectors_magnitude_l1949_194944

/-- Given two parallel vectors a and b, prove that the magnitude of b is 2√10 -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, 3) → 
  b.1 = -2 → 
  (∃ k : ℝ, b = k • a) → 
  ‖b‖ = 2 * Real.sqrt 10 := by
  sorry


end parallel_vectors_magnitude_l1949_194944


namespace integral_proof_l1949_194931

theorem integral_proof (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) : 
  (deriv (fun x => Real.log (abs (x - 2)) - 3 / (2 * (x + 2)^2))) x = 
  (x^3 + 6*x^2 + 15*x + 2) / ((x - 2) * (x + 2)^3) :=
by sorry

end integral_proof_l1949_194931


namespace max_diagonal_sum_l1949_194999

/-- A rhombus with side length 5 -/
structure Rhombus where
  side_length : ℝ
  side_length_eq : side_length = 5

/-- The diagonals of the rhombus -/
structure RhombusDiagonals (r : Rhombus) where
  d1 : ℝ
  d2 : ℝ
  d1_le_6 : d1 ≤ 6
  d2_ge_6 : d2 ≥ 6

/-- The sum of the diagonals -/
def diagonal_sum (r : Rhombus) (d : RhombusDiagonals r) : ℝ := d.d1 + d.d2

/-- The theorem stating the maximum sum of diagonals -/
theorem max_diagonal_sum (r : Rhombus) :
  ∃ (d : RhombusDiagonals r), ∀ (d' : RhombusDiagonals r), diagonal_sum r d ≥ diagonal_sum r d' ∧ diagonal_sum r d = 14 :=
sorry

end max_diagonal_sum_l1949_194999


namespace specific_garden_area_l1949_194993

/-- Represents a circular garden with a path through it. -/
structure GardenWithPath where
  diameter : ℝ
  pathWidth : ℝ

/-- Calculates the remaining area of the garden not covered by the path. -/
def remainingArea (g : GardenWithPath) : ℝ :=
  sorry

/-- Theorem stating the remaining area for a specific garden configuration. -/
theorem specific_garden_area :
  let g : GardenWithPath := { diameter := 14, pathWidth := 4 }
  remainingArea g = 29 * Real.pi := by
  sorry

end specific_garden_area_l1949_194993


namespace max_integers_greater_than_20_l1949_194935

theorem max_integers_greater_than_20 (integers : List ℤ) : 
  integers.length = 8 → 
  integers.sum = -20 → 
  (integers.filter (λ x => x > 20)).length ≤ 7 ∧ 
  ∃ (valid_list : List ℤ), 
    valid_list.length = 8 ∧ 
    valid_list.sum = -20 ∧ 
    (valid_list.filter (λ x => x > 20)).length = 7 :=
by sorry

end max_integers_greater_than_20_l1949_194935


namespace nine_by_nine_corner_sum_l1949_194919

/-- Represents a square grid -/
structure Grid :=
  (size : ℕ)

/-- The value at a given position in the grid -/
def Grid.value (g : Grid) (row col : ℕ) : ℕ :=
  (row - 1) * g.size + col

/-- The sum of the corner values in the grid -/
def Grid.cornerSum (g : Grid) : ℕ :=
  g.value 1 1 + g.value 1 g.size + g.value g.size 1 + g.value g.size g.size

/-- Theorem: The sum of corner values in a 9x9 grid is 164 -/
theorem nine_by_nine_corner_sum :
  ∀ g : Grid, g.size = 9 → g.cornerSum = 164 :=
by
  sorry

end nine_by_nine_corner_sum_l1949_194919


namespace existence_of_g_l1949_194986

open Set
open Function
open ContinuousOn

theorem existence_of_g (a b : ℝ) (f : ℝ → ℝ) 
  (h_f_cont : ContinuousOn f (Icc a b))
  (h_f_deriv : DifferentiableOn ℝ f (Icc a b))
  (h_f_zero : ∀ x ∈ Icc a b, f x = 0 → deriv f x ≠ 0) :
  ∃ g : ℝ → ℝ, 
    ContinuousOn g (Icc a b) ∧ 
    DifferentiableOn ℝ g (Icc a b) ∧
    ∀ x ∈ Icc a b, f x * deriv g x > deriv f x * g x :=
sorry

end existence_of_g_l1949_194986


namespace remaining_bottles_l1949_194915

/-- Calculates the number of remaining bottles of juice after some are broken -/
theorem remaining_bottles (total_crates : ℕ) (bottles_per_crate : ℕ) (broken_crates : ℕ) :
  total_crates = 7 →
  bottles_per_crate = 6 →
  broken_crates = 3 →
  total_crates * bottles_per_crate - broken_crates * bottles_per_crate = 24 :=
by
  sorry

end remaining_bottles_l1949_194915


namespace power_multiplication_l1949_194912

theorem power_multiplication (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end power_multiplication_l1949_194912


namespace triangle_area_l1949_194917

def a : Fin 2 → ℝ := ![5, 1]
def b : Fin 2 → ℝ := ![2, 4]

theorem triangle_area : 
  (1/2 : ℝ) * |Matrix.det ![a, b]| = 9 := by sorry

end triangle_area_l1949_194917


namespace book_fraction_is_half_l1949_194954

-- Define the total amount Jennifer had
def total_money : ℚ := 120

-- Define the fraction spent on sandwich
def sandwich_fraction : ℚ := 1 / 5

-- Define the fraction spent on museum ticket
def museum_fraction : ℚ := 1 / 6

-- Define the amount left over
def left_over : ℚ := 16

-- Theorem to prove
theorem book_fraction_is_half :
  let sandwich_cost := total_money * sandwich_fraction
  let museum_cost := total_money * museum_fraction
  let total_spent := total_money - left_over
  let book_cost := total_spent - sandwich_cost - museum_cost
  book_cost / total_money = 1 / 2 := by
  sorry

end book_fraction_is_half_l1949_194954


namespace remainder_theorem_l1949_194924

theorem remainder_theorem (n : ℤ) : n % 9 = 5 → (4 * n - 6) % 9 = 5 := by
  sorry

end remainder_theorem_l1949_194924


namespace trigonometric_properties_l1949_194926

theorem trigonometric_properties :
  (∀ x : ℝ, -1 ≤ Real.sin x ∧ Real.sin x ≤ 1) ∧
  ¬(∃ x : ℝ, Real.sin x ^ 2 + Real.cos x ^ 2 > 1) :=
by sorry

end trigonometric_properties_l1949_194926


namespace meal_cost_is_45_l1949_194938

/-- The cost of a meal consisting of one pizza and three burgers -/
def meal_cost (burger_price : ℝ) : ℝ :=
  let pizza_price := 2 * burger_price
  pizza_price + 3 * burger_price

/-- Theorem: The cost of one pizza and three burgers is $45 -/
theorem meal_cost_is_45 :
  meal_cost 9 = 45 := by
  sorry

end meal_cost_is_45_l1949_194938


namespace highest_backing_is_5000_l1949_194978

/-- Represents the financial backing levels for a crowdfunding campaign -/
structure FinancialBacking where
  lowest_level : ℕ
  second_level : ℕ
  highest_level : ℕ
  backers_lowest : ℕ
  backers_second : ℕ
  backers_highest : ℕ
  total_raised : ℕ

/-- The financial backing levels satisfy the given conditions -/
def ValidFinancialBacking (fb : FinancialBacking) : Prop :=
  fb.second_level = 10 * fb.lowest_level ∧
  fb.highest_level = 10 * fb.second_level ∧
  fb.backers_lowest = 10 ∧
  fb.backers_second = 3 ∧
  fb.backers_highest = 2 ∧
  fb.total_raised = 12000 ∧
  fb.total_raised = fb.backers_lowest * fb.lowest_level + 
                    fb.backers_second * fb.second_level + 
                    fb.backers_highest * fb.highest_level

/-- Theorem: The highest level of financial backing is $5000 -/
theorem highest_backing_is_5000 (fb : FinancialBacking) 
  (h : ValidFinancialBacking fb) : fb.highest_level = 5000 := by
  sorry

end highest_backing_is_5000_l1949_194978


namespace right_triangle_area_l1949_194948

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 := by
sorry

end right_triangle_area_l1949_194948


namespace sqrt_sum_fractions_l1949_194929

theorem sqrt_sum_fractions : Real.sqrt (1/8 + 1/25) = Real.sqrt 33 / (10 * Real.sqrt 2) := by
  sorry

end sqrt_sum_fractions_l1949_194929


namespace athlete_speed_l1949_194921

/-- Given an athlete running 200 meters in 24 seconds, prove their speed is approximately 30 km/h -/
theorem athlete_speed (distance : Real) (time : Real) (h1 : distance = 200) (h2 : time = 24) :
  ∃ (speed : Real), abs (speed - 30) < 0.1 ∧ speed = (distance / 1000) / (time / 3600) := by
  sorry

end athlete_speed_l1949_194921


namespace radish_count_l1949_194997

theorem radish_count (total : ℕ) (difference : ℕ) (radishes : ℕ) : 
  total = 100 →
  difference = 24 →
  radishes = total - difference / 2 →
  radishes = 62 := by
sorry

end radish_count_l1949_194997


namespace parabolas_intersection_l1949_194956

def parabola1 (x : ℝ) : ℝ := 2 * x^2 + 5 * x + 1
def parabola2 (x : ℝ) : ℝ := -x^2 + 4 * x + 6

theorem parabolas_intersection :
  ∃ (y1 y2 : ℝ),
    (∀ x : ℝ, parabola1 x = parabola2 x ↔ x = (-1 + Real.sqrt 61) / 6 ∨ x = (-1 - Real.sqrt 61) / 6) ∧
    parabola1 ((-1 + Real.sqrt 61) / 6) = y1 ∧
    parabola1 ((-1 - Real.sqrt 61) / 6) = y2 :=
by sorry

end parabolas_intersection_l1949_194956


namespace slide_problem_l1949_194976

theorem slide_problem (initial_boys : ℕ) (total_boys : ℕ) (h1 : initial_boys = 22) (h2 : total_boys = 35) :
  total_boys - initial_boys = 13 := by
  sorry

end slide_problem_l1949_194976


namespace caden_coin_ratio_l1949_194967

/-- Represents the number of coins of each type -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCounts) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes + 25 * coins.quarters

/-- Represents Caden's coin collection -/
def cadenCoins : CoinCounts where
  pennies := 120
  nickels := 40
  dimes := 8
  quarters := 16

theorem caden_coin_ratio :
  cadenCoins.pennies = 120 ∧
  cadenCoins.pennies = 3 * cadenCoins.nickels ∧
  cadenCoins.quarters = 2 * cadenCoins.dimes ∧
  totalValue cadenCoins = 800 →
  cadenCoins.nickels = 5 * cadenCoins.dimes :=
by sorry

end caden_coin_ratio_l1949_194967


namespace decagon_ratio_l1949_194922

/-- Represents a decagon with specific properties -/
structure Decagon where
  total_area : ℝ
  squares_below : ℝ
  trapezoid_base1 : ℝ
  trapezoid_base2 : ℝ
  bisector : ℝ → ℝ → Prop

/-- Theorem stating the ratio of XQ to QY in the given decagon -/
theorem decagon_ratio (d : Decagon)
    (h_area : d.total_area = 12)
    (h_squares : d.squares_below = 2)
    (h_base1 : d.trapezoid_base1 = 3)
    (h_base2 : d.trapezoid_base2 = 6)
    (h_bisect : d.bisector (d.squares_below + (d.trapezoid_base1 + d.trapezoid_base2) / 2 * h) (d.total_area / 2))
    (x y : ℝ)
    (h_xy : x + y = 6)
    (h_bisect_xy : d.bisector x y) :
    x / y = 2 := by
  sorry

end decagon_ratio_l1949_194922


namespace one_fourth_divided_by_two_l1949_194923

theorem one_fourth_divided_by_two : (1 / 4 : ℚ) / 2 = 1 / 8 := by
  sorry

end one_fourth_divided_by_two_l1949_194923


namespace parallel_line_family_l1949_194947

/-- The line equation as a function of x, y, and a -/
def line_equation (x y a : ℝ) : ℝ := (a - 1) * x - y + 2 * a + 1

/-- Theorem stating that the lines form a parallel family -/
theorem parallel_line_family :
  ∀ a₁ a₂ : ℝ, ∃ k : ℝ, ∀ x y : ℝ,
    line_equation x y a₁ = 0 ↔ line_equation x y a₂ = k := by
  sorry

end parallel_line_family_l1949_194947


namespace octal_year_to_decimal_l1949_194937

/-- Converts an octal number represented as a list of digits to its decimal equivalent -/
def octal_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

/-- The octal representation of the year -/
def octal_year : List Nat := [7, 4, 2]

/-- Theorem stating that the octal year 742 is equal to 482 in decimal -/
theorem octal_year_to_decimal :
  octal_to_decimal octal_year = 482 := by sorry

end octal_year_to_decimal_l1949_194937


namespace percent_both_correct_l1949_194982

theorem percent_both_correct
  (percent_first : ℝ)
  (percent_second : ℝ)
  (percent_neither : ℝ)
  (h1 : percent_first = 75)
  (h2 : percent_second = 25)
  (h3 : percent_neither = 20)
  : ℝ :=
by
  -- Define the percentage of students who answered both questions correctly
  let percent_both : ℝ := percent_first + percent_second - (100 - percent_neither)
  
  -- Prove that percent_both equals 20
  have : percent_both = 20 := by sorry
  
  -- Return the result
  exact percent_both

end percent_both_correct_l1949_194982


namespace max_teams_advancing_l1949_194971

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  num_teams : Nat
  min_points_to_advance : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- Calculate the total number of games in the tournament -/
def total_games (t : FootballTournament) : Nat :=
  t.num_teams * (t.num_teams - 1) / 2

/-- Calculate the maximum total points that can be distributed in the tournament -/
def max_total_points (t : FootballTournament) : Nat :=
  (total_games t) * t.points_for_win

/-- Theorem stating the maximum number of teams that can advance -/
theorem max_teams_advancing (t : FootballTournament) 
  (h1 : t.num_teams = 6)
  (h2 : t.min_points_to_advance = 12)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1)
  (h5 : t.points_for_loss = 0) :
  ∃ (n : Nat), n ≤ 3 ∧ 
    n * t.min_points_to_advance ≤ max_total_points t ∧
    ∀ (m : Nat), m * t.min_points_to_advance ≤ max_total_points t → m ≤ n :=
by
  sorry

end max_teams_advancing_l1949_194971


namespace boxes_shipped_this_week_l1949_194932

/-- Represents the number of pomelos in a dozen -/
def dozen : ℕ := 12

/-- Represents the number of boxes shipped last week -/
def last_week_boxes : ℕ := 10

/-- Represents the total number of pomelos shipped last week -/
def last_week_pomelos : ℕ := 240

/-- Represents the number of dozens of pomelos shipped this week -/
def this_week_dozens : ℕ := 60

/-- Calculates the number of boxes shipped this week -/
def boxes_this_week : ℕ :=
  (this_week_dozens * dozen) / (last_week_pomelos / last_week_boxes)

theorem boxes_shipped_this_week :
  boxes_this_week = 30 := by sorry

end boxes_shipped_this_week_l1949_194932


namespace probability_two_diamonds_l1949_194964

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of suits in a standard deck
def num_suits : ℕ := 4

-- Define the number of ranks in a standard deck
def num_ranks : ℕ := 13

-- Define the number of cards of a single suit (Diamonds in this case)
def cards_per_suit : ℕ := total_cards / num_suits

-- Theorem statement
theorem probability_two_diamonds (total_cards num_suits num_ranks cards_per_suit : ℕ) 
  (h1 : total_cards = 52)
  (h2 : num_suits = 4)
  (h3 : num_ranks = 13)
  (h4 : cards_per_suit = total_cards / num_suits) :
  (cards_per_suit.choose 2 : ℚ) / (total_cards.choose 2) = 1 / 17 := by
  sorry

end probability_two_diamonds_l1949_194964


namespace days_without_calls_l1949_194960

-- Define the number of days in the year
def total_days : ℕ := 365

-- Define the periods of the calls
def period1 : ℕ := 3
def period2 : ℕ := 4
def period3 : ℕ := 5

-- Function to calculate the number of days with at least one call
def days_with_calls : ℕ :=
  (total_days / period1) +
  (total_days / period2) +
  (total_days / period3) -
  (total_days / (period1 * period2)) -
  (total_days / (period2 * period3)) -
  (total_days / (period1 * period3)) +
  (total_days / (period1 * period2 * period3))

-- Theorem to prove
theorem days_without_calls :
  total_days - days_with_calls = 146 :=
by sorry

end days_without_calls_l1949_194960


namespace tenth_row_white_squares_l1949_194995

/-- Represents the number of squares in the nth row of a stair-step figure -/
def totalSquares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in the nth row of a stair-step figure -/
def whiteSquares (n : ℕ) : ℕ := (totalSquares n) / 2

theorem tenth_row_white_squares :
  whiteSquares 10 = 9 := by sorry

end tenth_row_white_squares_l1949_194995


namespace power_seven_mod_twelve_l1949_194933

theorem power_seven_mod_twelve : 7^253 % 12 = 7 := by
  sorry

end power_seven_mod_twelve_l1949_194933


namespace power_of_product_l1949_194904

theorem power_of_product (a : ℝ) : (2 * a^2)^3 = 8 * a^6 := by
  sorry

end power_of_product_l1949_194904


namespace opposite_and_abs_of_sqrt3_minus2_l1949_194943

theorem opposite_and_abs_of_sqrt3_minus2 :
  (-(Real.sqrt 3 - 2) = 2 - Real.sqrt 3) ∧
  (|Real.sqrt 3 - 2| = 2 - Real.sqrt 3) := by
  sorry

end opposite_and_abs_of_sqrt3_minus2_l1949_194943


namespace complement_of_A_l1949_194973

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | (x - 1) * (x - 4) ≤ 0}

-- Theorem statement
theorem complement_of_A : 
  Set.compl A = {x : ℝ | x < 1 ∨ x > 4} := by sorry

end complement_of_A_l1949_194973


namespace max_placement_1002nd_round_max_placement_1001st_round_l1949_194914

/-- Represents the state of an election round -/
structure ElectionRound where
  candidateCount : Nat
  votes : List Nat

/-- Defines the election process -/
def runElection (initialRound : ElectionRound) : Nat → Option Nat :=
  sorry

/-- Theorem for the maximum initial placement allowing victory in the 1002nd round -/
theorem max_placement_1002nd_round 
  (initialCandidates : Nat) 
  (ostapInitialPlacement : Nat) :
  initialCandidates = 2002 →
  (∃ (initialVotes : List Nat), 
    initialVotes.length = initialCandidates ∧
    ostapInitialPlacement = 2001 ∧
    runElection ⟨initialCandidates, initialVotes⟩ 1002 = some ostapInitialPlacement) ∧
  (∀ k > 2001, ∀ (initialVotes : List Nat),
    initialVotes.length = initialCandidates →
    runElection ⟨initialCandidates, initialVotes⟩ 1002 ≠ some k) :=
  sorry

/-- Theorem for the maximum initial placement allowing victory in the 1001st round -/
theorem max_placement_1001st_round 
  (initialCandidates : Nat) 
  (ostapInitialPlacement : Nat) :
  initialCandidates = 2002 →
  (∃ (initialVotes : List Nat), 
    initialVotes.length = initialCandidates ∧
    ostapInitialPlacement = 1001 ∧
    runElection ⟨initialCandidates, initialVotes⟩ 1001 = some ostapInitialPlacement) ∧
  (∀ k > 1001, ∀ (initialVotes : List Nat),
    initialVotes.length = initialCandidates →
    runElection ⟨initialCandidates, initialVotes⟩ 1001 ≠ some k) :=
  sorry

end max_placement_1002nd_round_max_placement_1001st_round_l1949_194914


namespace motorboat_trip_time_l1949_194985

theorem motorboat_trip_time (v_b : ℝ) (d : ℝ) (h1 : v_b > 0) (h2 : d > 0) : 
  let v_c := v_b / 3
  let t_no_current := 2 * d / v_b
  let v_down := v_b + v_c
  let v_up := v_b - v_c
  let t_actual := d / v_down + d / v_up
  t_no_current = 44 / 60 → t_actual = 49.5 / 60 := by
sorry

end motorboat_trip_time_l1949_194985


namespace exam_class_size_l1949_194910

/-- Represents a class of students with their exam marks -/
structure ExamClass where
  total_students : ℕ
  total_marks : ℕ
  average_mark : ℚ
  excluded_students : ℕ
  excluded_average : ℚ
  remaining_average : ℚ

/-- Theorem stating the conditions and the result to be proven -/
theorem exam_class_size (c : ExamClass) 
  (h1 : c.average_mark = 80)
  (h2 : c.excluded_students = 5)
  (h3 : c.excluded_average = 40)
  (h4 : c.remaining_average = 90)
  (h5 : c.total_marks = c.total_students * c.average_mark)
  (h6 : c.total_marks - c.excluded_students * c.excluded_average = 
        (c.total_students - c.excluded_students) * c.remaining_average) :
  c.total_students = 25 := by
  sorry

end exam_class_size_l1949_194910


namespace circle_logarithm_l1949_194939

theorem circle_logarithm (a b : ℝ) (h_a : a > 0) (h_b : b > 0) : 
  (2 * Real.log a = Real.log (a^2)) →
  (4 * Real.log b = Real.log (b^4)) →
  (2 * π * Real.log (a^2) = Real.log (b^4)) →
  Real.log b / Real.log a = π :=
by sorry

end circle_logarithm_l1949_194939


namespace ellipse_focus_directrix_distance_l1949_194946

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 64 + y^2 / 28 = 1

/-- Distance from P to the left focus -/
def distance_to_left_focus (P : ℝ × ℝ) : ℝ := 4

/-- Distance from P to the right directrix -/
def distance_to_right_directrix (P : ℝ × ℝ) : ℝ := 16

/-- Theorem: If P is on the ellipse and 4 units from the left focus,
    then it is 16 units from the right directrix -/
theorem ellipse_focus_directrix_distance (P : ℝ × ℝ) :
  is_on_ellipse P.1 P.2 →
  distance_to_left_focus P = 4 →
  distance_to_right_directrix P = 16 :=
by
  sorry

end ellipse_focus_directrix_distance_l1949_194946


namespace negation_equivalence_l1949_194977

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 4*x + 5 ≤ 0) ↔ (∀ x : ℝ, x^2 + 4*x + 5 > 0) := by
  sorry

end negation_equivalence_l1949_194977


namespace sum_abc_l1949_194994

theorem sum_abc (a b c : ℤ) 
  (eq1 : 2 * a + 3 * b = 52)
  (eq2 : 3 * b + c = 41)
  (eq3 : b * c = 60) :
  a + b + c = 25 := by
sorry

end sum_abc_l1949_194994


namespace cinnamon_swirl_sharing_l1949_194970

theorem cinnamon_swirl_sharing (total_pieces : ℕ) (jane_pieces : ℕ) (h1 : total_pieces = 12) (h2 : jane_pieces = 4) :
  total_pieces / jane_pieces = 3 :=
by
  sorry

#check cinnamon_swirl_sharing

end cinnamon_swirl_sharing_l1949_194970


namespace max_two_digit_times_max_one_digit_is_three_digit_l1949_194962

theorem max_two_digit_times_max_one_digit_is_three_digit : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n = 99 * 9 := by
  sorry

end max_two_digit_times_max_one_digit_is_three_digit_l1949_194962


namespace expenditure_difference_l1949_194905

theorem expenditure_difference 
  (original_price : ℝ) 
  (required_amount : ℝ) 
  (price_increase_percentage : ℝ) 
  (purchased_amount_percentage : ℝ) :
  price_increase_percentage = 40 →
  purchased_amount_percentage = 62 →
  let new_price := original_price * (1 + price_increase_percentage / 100)
  let new_amount := required_amount * (purchased_amount_percentage / 100)
  let original_expenditure := original_price * required_amount
  let new_expenditure := new_price * new_amount
  let difference := new_expenditure - original_expenditure
  difference / original_expenditure = -0.132 :=
by sorry

end expenditure_difference_l1949_194905


namespace circle_equation_l1949_194940

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Check if a circle is tangent to a line -/
def Circle.tangentTo (c : Circle) (l : Line) : Prop :=
  let (cx, cy) := c.center
  |l.a * cx + l.b * cy + l.c| = c.radius * Real.sqrt (l.a^2 + l.b^2)

/-- The main theorem -/
theorem circle_equation (C : Circle) (l : Line) :
  C.contains (0, 0) →
  C.radius^2 * Real.pi = 2 * Real.pi →
  l.a = 1 ∧ l.b = -1 ∧ l.c = 2 →
  C.tangentTo l →
  (C.center = (1, 1) ∧ C.radius^2 = 2) ∨ (C.center = (-1, -1) ∧ C.radius^2 = 2) :=
by sorry

end circle_equation_l1949_194940


namespace max_garden_area_l1949_194984

/-- Represents a rectangular garden with given constraints -/
structure Garden where
  length : ℝ
  width : ℝ
  perimeter_eq : length * 2 + width * 2 = 400
  length_ge : length ≥ 100
  width_ge : width ≥ 50

/-- The area of a garden -/
def Garden.area (g : Garden) : ℝ := g.length * g.width

/-- Theorem stating the maximum area of a garden with given constraints -/
theorem max_garden_area :
  ∀ g : Garden, g.area ≤ 10000 :=
by
  sorry

end max_garden_area_l1949_194984


namespace product_of_reals_l1949_194928

theorem product_of_reals (x y : ℝ) (sum_eq : x + y = 10) (sum_cubes_eq : x^3 + y^3 = 370) : x * y = 21 := by
  sorry

end product_of_reals_l1949_194928


namespace max_player_salary_l1949_194906

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (total_salary_cap : ℕ) :
  num_players = 18 →
  min_salary = 20000 →
  total_salary_cap = 900000 →
  ∃ (max_salary : ℕ),
    max_salary = 560000 ∧
    max_salary + (num_players - 1) * min_salary ≤ total_salary_cap ∧
    ∀ (s : ℕ), s > max_salary →
      s + (num_players - 1) * min_salary > total_salary_cap :=
by sorry


end max_player_salary_l1949_194906


namespace circle_radii_formula_l1949_194972

/-- Given a triangle ABC with circumradius R and heights h_a, h_b, h_c,
    the radii t_a, t_b, t_c of circles tangent internally to the inscribed circle
    at vertices A, B, C and externally to each other satisfy the given formulas. -/
theorem circle_radii_formula (a b c R h_a h_b h_c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0) :
  ∃ (t_a t_b t_c : ℝ),
    (t_a > 0 ∧ t_b > 0 ∧ t_c > 0) ∧
    (t_a = (R * h_a) / (a + h_a)) ∧
    (t_b = (R * h_b) / (b + h_b)) ∧
    (t_c = (R * h_c) / (c + h_c)) :=
by sorry

end circle_radii_formula_l1949_194972


namespace football_team_linemen_l1949_194902

-- Define the constants from the problem
def cooler_capacity : ℕ := 126
def skill_players : ℕ := 10
def lineman_consumption : ℕ := 8
def skill_player_consumption : ℕ := 6
def skill_players_drinking : ℕ := 5

-- Define the number of linemen as a variable
def num_linemen : ℕ := sorry

-- Theorem statement
theorem football_team_linemen :
  num_linemen * lineman_consumption +
  skill_players_drinking * skill_player_consumption = cooler_capacity :=
by sorry

end football_team_linemen_l1949_194902


namespace power_of_two_sum_l1949_194949

theorem power_of_two_sum (y : ℕ) : 8^3 + 8^3 + 8^3 + 8^3 = 2^y → y = 11 := by
  sorry

end power_of_two_sum_l1949_194949


namespace number_of_history_books_l1949_194907

theorem number_of_history_books (total_books geography_books math_books : ℕ) 
  (h1 : total_books = 100)
  (h2 : geography_books = 25)
  (h3 : math_books = 43) :
  total_books - geography_books - math_books = 32 :=
by sorry

end number_of_history_books_l1949_194907


namespace subset_of_countable_is_finite_or_countable_l1949_194979

theorem subset_of_countable_is_finite_or_countable 
  (X : Set α) (hX : Countable X) (A : Set α) (hA : A ⊆ X) :
  (Finite A) ∨ (Countable A) :=
sorry

end subset_of_countable_is_finite_or_countable_l1949_194979


namespace point_on_line_p_value_l1949_194951

/-- Given that (m, n) and (m + p, n + 15) both lie on the line x = (y / 5) - (2 / 5),
    prove that p = 3. -/
theorem point_on_line_p_value (m n p : ℝ) : 
  (m = n / 5 - 2 / 5) → 
  (m + p = (n + 15) / 5 - 2 / 5) → 
  p = 3 := by
  sorry

end point_on_line_p_value_l1949_194951


namespace triangle_circle_area_difference_l1949_194916

/-- The difference between the area of an equilateral triangle with side length 6
    and the area of an inscribed circle with radius 3 -/
theorem triangle_circle_area_difference : ∃ (circle_area triangle_area : ℝ),
  circle_area = 9 * Real.pi ∧
  triangle_area = 9 * Real.sqrt 3 ∧
  triangle_area - circle_area = 9 * Real.sqrt 3 - 9 * Real.pi := by
  sorry

end triangle_circle_area_difference_l1949_194916


namespace second_number_proof_l1949_194990

theorem second_number_proof (x : ℝ) : 
  let set1 := [10, 60, 35]
  let set2 := [20, 60, x]
  (set2.sum / set2.length : ℝ) = (set1.sum / set1.length : ℝ) + 5 →
  x = 40 := by
sorry

end second_number_proof_l1949_194990


namespace bob_pennies_bob_pennies_proof_l1949_194968

theorem bob_pennies : ℕ → ℕ → Prop :=
  fun a b =>
    (b + 1 = 4 * (a - 1)) →
    (b - 1 = 3 * (a + 1)) →
    b = 31

-- The proof is omitted
theorem bob_pennies_proof : ∃ a b : ℕ, bob_pennies a b := by
  sorry

end bob_pennies_bob_pennies_proof_l1949_194968


namespace triangle_properties_l1949_194936

open Real

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  tan C = (sin A + sin B) / (cos A + cos B) →
  C = π / 3 ∧
  (∀ r : ℝ, r > 0 → 2 * r = 1 →
    3/4 < a^2 + b^2 ∧ a^2 + b^2 ≤ 3/2) :=
by sorry

end triangle_properties_l1949_194936


namespace water_added_to_mixture_water_added_is_ten_l1949_194927

/-- Given a mixture of alcohol and water, prove the amount of water added to change the ratio. -/
theorem water_added_to_mixture (initial_ratio : ℚ) (final_ratio : ℚ) (alcohol_quantity : ℚ) : ℚ :=
  let initial_water := (alcohol_quantity * 5) / 2
  let water_added := (7 * alcohol_quantity) / 2 - initial_water
  by
    -- Assumptions
    have h1 : initial_ratio = 2 / 5 := by sorry
    have h2 : final_ratio = 2 / 7 := by sorry
    have h3 : alcohol_quantity = 10 := by sorry

    -- Proof
    sorry

/-- The amount of water added to the mixture is 10 liters. -/
theorem water_added_is_ten : water_added_to_mixture (2/5) (2/7) 10 = 10 := by sorry

end water_added_to_mixture_water_added_is_ten_l1949_194927


namespace maxwells_speed_l1949_194930

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions -/
theorem maxwells_speed (total_distance : ℝ) (brads_speed : ℝ) (maxwell_time : ℝ) 
  (brad_delay : ℝ) (h1 : total_distance = 74) (h2 : brads_speed = 6) 
  (h3 : maxwell_time = 8) (h4 : brad_delay = 1) : 
  ∃ (maxwell_speed : ℝ), maxwell_speed = 4 := by
  sorry

end maxwells_speed_l1949_194930


namespace system_equations_proof_l1949_194989

theorem system_equations_proof (a x y : ℝ) : 
  x + y = -7 - a → 
  x - y = 1 + 3*a → 
  x ≤ 0 → 
  y < 0 → 
  (-2 < a ∧ a ≤ 3) ∧ 
  (abs (a - 3) + abs (a + 2) = 5) ∧ 
  (∀ (a : ℤ), -2 < a ∧ a ≤ 3 → (∀ x, 2*a*x + x > 2*a + 1 ↔ x < 1) ↔ a = -1) :=
by sorry

end system_equations_proof_l1949_194989


namespace bacteria_count_scientific_notation_l1949_194965

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Checks if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (n : ℝ) : Prop :=
  n = sn.coefficient * (10 : ℝ) ^ sn.exponent

/-- The number of bacteria in a fly's stomach -/
def bacteria_count : ℕ := 28000000

/-- The scientific notation representation of the bacteria count -/
def bacteria_scientific : ScientificNotation where
  coefficient := 2.8
  exponent := 7
  h1 := by sorry

/-- Theorem stating that the scientific notation correctly represents the bacteria count -/
theorem bacteria_count_scientific_notation :
    represents bacteria_scientific (bacteria_count : ℝ) := by sorry

end bacteria_count_scientific_notation_l1949_194965
