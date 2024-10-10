import Mathlib

namespace tangent_line_of_g_l1436_143686

/-- Given a function f with a tangent line y = 2x - 1 at (2, f(2)),
    the tangent line to g(x) = x^2 + f(x) at (2, g(2)) is 6x - y - 5 = 0 -/
theorem tangent_line_of_g (f : ℝ → ℝ) (h : HasDerivAt f 2 2) :
  let g := λ x => x^2 + f x
  ∃ L : ℝ → ℝ, HasDerivAt g 6 2 ∧ L x = 6*x - 5 := by
  sorry

end tangent_line_of_g_l1436_143686


namespace friends_marbles_theorem_l1436_143671

/-- Calculates the number of marbles Reggie's friend arrived with -/
def friends_initial_marbles (total_games : ℕ) (marbles_per_game : ℕ) (reggies_final_marbles : ℕ) (games_lost : ℕ) : ℕ :=
  let games_won := total_games - games_lost
  let marbles_gained := games_won * marbles_per_game
  let reggies_initial_marbles := reggies_final_marbles - marbles_gained
  reggies_initial_marbles + marbles_per_game

theorem friends_marbles_theorem (total_games : ℕ) (marbles_per_game : ℕ) (reggies_final_marbles : ℕ) (games_lost : ℕ)
  (h1 : total_games = 9)
  (h2 : marbles_per_game = 10)
  (h3 : reggies_final_marbles = 90)
  (h4 : games_lost = 1) :
  friends_initial_marbles total_games marbles_per_game reggies_final_marbles games_lost = 20 := by
  sorry

#eval friends_initial_marbles 9 10 90 1

end friends_marbles_theorem_l1436_143671


namespace major_premise_is_false_l1436_143641

theorem major_premise_is_false : ¬ ∀ (a : ℝ) (n : ℕ), (a^(1/n : ℝ))^n = a := by sorry

end major_premise_is_false_l1436_143641


namespace plans_equal_at_325_miles_unique_intersection_at_325_miles_l1436_143680

/-- Represents a car rental plan with an initial fee and a per-mile rate -/
structure RentalPlan where
  initialFee : ℝ
  perMileRate : ℝ

/-- The two rental plans available -/
def plan1 : RentalPlan := { initialFee := 65, perMileRate := 0.4 }
def plan2 : RentalPlan := { initialFee := 0, perMileRate := 0.6 }

/-- The cost of a rental plan for a given number of miles -/
def rentalCost (plan : RentalPlan) (miles : ℝ) : ℝ :=
  plan.initialFee + plan.perMileRate * miles

/-- The theorem stating that the two plans cost the same at 325 miles -/
theorem plans_equal_at_325_miles :
  rentalCost plan1 325 = rentalCost plan2 325 := by
  sorry

/-- The theorem stating that 325 is the unique point where the plans cost the same -/
theorem unique_intersection_at_325_miles :
  ∀ m : ℝ, rentalCost plan1 m = rentalCost plan2 m → m = 325 := by
  sorry

end plans_equal_at_325_miles_unique_intersection_at_325_miles_l1436_143680


namespace intersection_of_sets_l1436_143621

theorem intersection_of_sets : 
  let A : Set ℕ := {2, 3, 4}
  let B : Set ℕ := {1, 2, 3}
  A ∩ B = {2, 3} := by
sorry

end intersection_of_sets_l1436_143621


namespace triangle_properties_l1436_143683

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a * Real.cos t.B = (3 * t.c - t.b) * Real.cos t.A ∧
  t.a = 2 * Real.sqrt 2 ∧
  (1 / 2) * t.b * t.c * Real.sin t.A = Real.sqrt 2

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : triangle_conditions t) :
  Real.sin t.A = (2 * Real.sqrt 2) / 3 ∧ t.b + t.c = 4 := by
  sorry

end triangle_properties_l1436_143683


namespace ten_times_average_letters_l1436_143648

def elida_letters : ℕ := 5

def adrianna_letters : ℕ := 2 * elida_letters - 2

def average_letters : ℚ := (elida_letters + adrianna_letters) / 2

theorem ten_times_average_letters : 10 * average_letters = 65 := by
  sorry

end ten_times_average_letters_l1436_143648


namespace expression_evaluation_l1436_143697

theorem expression_evaluation : 
  let f (x : ℚ) := (x + 2) / (x - 2)
  let g (x : ℚ) := (f x + 2) / (f x - 2)
  g (1/3) = -31/37 := by sorry

end expression_evaluation_l1436_143697


namespace average_speed_calculation_l1436_143647

/-- Proves that the average speed for a 60-mile trip with specified conditions is 30 mph -/
theorem average_speed_calculation (total_distance : ℝ) (first_half_speed : ℝ) (speed_increase : ℝ) :
  total_distance = 60 →
  first_half_speed = 24 →
  speed_increase = 16 →
  let second_half_speed := first_half_speed + speed_increase
  let first_half_time := (total_distance / 2) / first_half_speed
  let second_half_time := (total_distance / 2) / second_half_speed
  let total_time := first_half_time + second_half_time
  total_distance / total_time = 30 := by
  sorry

#check average_speed_calculation

end average_speed_calculation_l1436_143647


namespace average_difference_l1436_143681

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 110) 
  (h2 : (b + c) / 2 = 170) : 
  a - c = -120 := by
sorry

end average_difference_l1436_143681


namespace trailing_zeros_25_factorial_l1436_143685

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- Theorem: The number of trailing zeros in 25! is 6 -/
theorem trailing_zeros_25_factorial :
  trailingZeros 25 = 6 := by
  sorry

end trailing_zeros_25_factorial_l1436_143685


namespace cricket_team_size_l1436_143600

theorem cricket_team_size :
  ∀ n : ℕ,
  n > 0 →
  let avg_age : ℚ := 26
  let wicket_keeper_age : ℚ := avg_age + 3
  let remaining_avg_age : ℚ := avg_age - 1
  (n : ℚ) * avg_age = wicket_keeper_age + avg_age + (n - 2 : ℚ) * remaining_avg_age →
  n = 5 := by
sorry

end cricket_team_size_l1436_143600


namespace estimated_students_above_average_l1436_143662

/-- Represents the time intervals for physical exercise --/
inductive TimeInterval
| LessThan30
| Between30And60
| Between60And90
| Between90And120

/-- Represents the data from the survey --/
structure SurveyData where
  sampleSize : Nat
  totalStudents : Nat
  mean : Nat
  studentsPerInterval : TimeInterval → Nat

/-- Theorem: Given the survey data, prove that the estimated number of students
    spending at least the average time on exercise is 130 --/
theorem estimated_students_above_average (data : SurveyData)
  (h1 : data.sampleSize = 20)
  (h2 : data.totalStudents = 200)
  (h3 : data.mean = 60)
  (h4 : data.studentsPerInterval TimeInterval.LessThan30 = 2)
  (h5 : data.studentsPerInterval TimeInterval.Between30And60 = 5)
  (h6 : data.studentsPerInterval TimeInterval.Between60And90 = 10)
  (h7 : data.studentsPerInterval TimeInterval.Between90And120 = 3) :
  (data.totalStudents * (data.studentsPerInterval TimeInterval.Between60And90 +
   data.studentsPerInterval TimeInterval.Between90And120) / data.sampleSize) = 130 := by
  sorry


end estimated_students_above_average_l1436_143662


namespace num_correct_statements_is_zero_l1436_143610

/-- Represents a programming statement --/
inductive Statement
  | Input (vars : List String)
  | Output (expr : String)
  | Assignment (lhs : String) (rhs : String)

/-- Checks if an input statement is correct --/
def isValidInput (s : Statement) : Bool :=
  match s with
  | Statement.Input vars => vars.length > 0 && vars.all (fun v => v.length > 0)
  | _ => false

/-- Checks if an output statement is correct --/
def isValidOutput (s : Statement) : Bool :=
  match s with
  | Statement.Output expr => expr.startsWith "PRINT"
  | _ => false

/-- Checks if an assignment statement is correct --/
def isValidAssignment (s : Statement) : Bool :=
  match s with
  | Statement.Assignment lhs rhs => lhs.length > 0 && !lhs.toList.head!.isDigit && !rhs.contains '='
  | _ => false

/-- Checks if a statement is correct --/
def isValidStatement (s : Statement) : Bool :=
  isValidInput s || isValidOutput s || isValidAssignment s

/-- The list of statements to check --/
def statements : List Statement :=
  [Statement.Input ["a;", "b;", "c"],
   Statement.Output "A=4",
   Statement.Assignment "3" "B",
   Statement.Assignment "A" "B=-2"]

/-- Theorem: The number of correct statements is 0 --/
theorem num_correct_statements_is_zero : 
  (statements.filter isValidStatement).length = 0 := by
  sorry


end num_correct_statements_is_zero_l1436_143610


namespace tank_egg_difference_l1436_143611

/-- The number of eggs Tank gathered in the first round -/
def tank_first : ℕ := 160

/-- The number of eggs Tank gathered in the second round -/
def tank_second : ℕ := 30

/-- The number of eggs Emma gathered in the first round -/
def emma_first : ℕ := tank_first - 10

/-- The number of eggs Emma gathered in the second round -/
def emma_second : ℕ := 60

/-- The total number of eggs collected by all 8 people -/
def total_eggs : ℕ := 400

theorem tank_egg_difference :
  tank_first - tank_second = 130 ∧
  emma_second = 2 * tank_second ∧
  tank_first > tank_second ∧
  tank_first = emma_first + 10 ∧
  tank_first + emma_first + tank_second + emma_second = total_eggs :=
by sorry

end tank_egg_difference_l1436_143611


namespace min_cards_is_smallest_l1436_143675

/-- The smallest number of cards needed to represent all integers from 1 to n! as sums of factorials -/
def min_cards (n : ℕ+) : ℕ :=
  n.val * (n.val + 1) / 2 + 1

/-- Theorem stating that min_cards gives the smallest possible number of cards needed -/
theorem min_cards_is_smallest (n : ℕ+) :
  ∀ (t : ℕ), t ≤ n.val.factorial →
  ∃ (S : Finset ℕ),
    (∀ m ∈ S, ∃ k : ℕ+, m = k.val.factorial) ∧
    (S.card ≤ min_cards n) ∧
    (t = S.sum id) :=
sorry

end min_cards_is_smallest_l1436_143675


namespace square_plot_area_l1436_143688

/-- Given a square plot with a fence around it, if the price per foot of the fence is 58 and
    the total cost is 2088, then the area of the square plot is 81 square feet. -/
theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) (side_length : ℝ) :
  price_per_foot = 58 →
  total_cost = 2088 →
  total_cost = 4 * side_length * price_per_foot →
  side_length ^ 2 = 81 :=
by sorry

end square_plot_area_l1436_143688


namespace points_on_line_procedure_l1436_143690

theorem points_on_line_procedure (x : ℕ) : ∃ x > 0, 9*x - 8 = 82 := by
  sorry

end points_on_line_procedure_l1436_143690


namespace divisibility_theorem_l1436_143605

theorem divisibility_theorem (a b c d e f : ℤ) 
  (h : (13 : ℤ) ∣ (a^12 + b^12 + c^12 + d^12 + e^12 + f^12)) : 
  (13^6 : ℤ) ∣ (a * b * c * d * e * f) := by
  sorry

end divisibility_theorem_l1436_143605


namespace arithmetic_sequence_sum_l1436_143650

/-- An arithmetic sequence -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 + a 2 + a 3 = 6) →
  (a 7 + a 8 + a 9 = 24) →
  (a 4 + a 5 + a 6 = 15) :=
by
  sorry

end arithmetic_sequence_sum_l1436_143650


namespace h_solutions_l1436_143637

noncomputable def h (x : ℝ) : ℝ :=
  if x < 2 then 4 * x + 10 else 3 * x - 12

theorem h_solutions :
  ∀ x : ℝ, h x = 6 ↔ x = -1 ∨ x = 6 :=
by sorry

end h_solutions_l1436_143637


namespace sine_angle_plus_pi_half_l1436_143626

theorem sine_angle_plus_pi_half (α : Real) : 
  (∃ r : Real, r > 0 ∧ -1 = r * Real.cos α ∧ Real.sqrt 3 = r * Real.sin α) →
  Real.sin (α + π/2) = -1/2 := by
sorry

end sine_angle_plus_pi_half_l1436_143626


namespace intersection_implies_m_value_l1436_143625

theorem intersection_implies_m_value (m : ℝ) : 
  let A : Set ℝ := {1, m-2}
  let B : Set ℝ := {2, 3}
  A ∩ B = {2} → m = 4 := by
sorry

end intersection_implies_m_value_l1436_143625


namespace hydrogen_atom_count_l1436_143695

/-- Represents the number of atoms of each element in the compound -/
structure AtomCount where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic weights of elements -/
structure AtomicWeights where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

/-- Calculates the molecular weight of a compound -/
def molecularWeight (count : AtomCount) (weights : AtomicWeights) : ℝ :=
  count.carbon * weights.carbon + count.hydrogen * weights.hydrogen + count.oxygen * weights.oxygen

/-- The main theorem stating the number of hydrogen atoms in the compound -/
theorem hydrogen_atom_count (weights : AtomicWeights) 
    (h_carbon : weights.carbon = 12)
    (h_hydrogen : weights.hydrogen = 1)
    (h_oxygen : weights.oxygen = 16) : 
  ∃ (count : AtomCount), 
    count.carbon = 3 ∧ 
    count.oxygen = 1 ∧ 
    molecularWeight count weights = 58 ∧ 
    count.hydrogen = 6 := by
  sorry

end hydrogen_atom_count_l1436_143695


namespace m_range_characterization_l1436_143664

def r (m : ℝ) (x : ℝ) : Prop := Real.sin x + Real.cos x > m

def s (m : ℝ) (x : ℝ) : Prop := x^2 + m*x + 1 > 0

theorem m_range_characterization (m : ℝ) :
  (∀ x : ℝ, (r m x ∧ ¬(s m x)) ∨ (¬(r m x) ∧ s m x)) ↔ 
  (m ≤ -2 ∨ (-Real.sqrt 2 ≤ m ∧ m < 2)) :=
sorry

end m_range_characterization_l1436_143664


namespace sufficient_not_necessary_condition_l1436_143672

theorem sufficient_not_necessary_condition (m : ℝ) : 
  (∀ x : ℝ, |x - 4| ≤ 6 → x ≤ 1 + m) ∧ 
  (∃ x : ℝ, x ≤ 1 + m ∧ |x - 4| > 6) ↔ 
  m ≥ 9 := by sorry

end sufficient_not_necessary_condition_l1436_143672


namespace system_solution_l1436_143659

theorem system_solution : 
  ∃ (x y : ℚ), (4 * x - 3 * y = -9) ∧ (5 * x + 4 * y = 14) ∧ (x = 6/31) ∧ (y = 101/31) := by
  sorry

end system_solution_l1436_143659


namespace exponential_equation_solutions_l1436_143638

theorem exponential_equation_solutions :
  ∀ x y : ℕ+, (3 : ℕ) ^ x.val = 2 ^ x.val * y.val + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) := by
  sorry

end exponential_equation_solutions_l1436_143638


namespace triangle_c_coordinates_l1436_143636

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the area function
def area (t : Triangle) : ℝ := sorry

-- Define the line equation
def onLine (p : ℝ × ℝ) : Prop :=
  3 * p.1 - p.2 + 3 = 0

-- Theorem statement
theorem triangle_c_coordinates :
  ∀ (t : Triangle),
    t.A = (3, 2) →
    t.B = (-1, 5) →
    onLine t.C →
    area t = 10 →
    (t.C = (-1, 0) ∨ t.C = (5/3, 8)) :=
by sorry

end triangle_c_coordinates_l1436_143636


namespace word_transformations_l1436_143629

-- Define the alphabet
inductive Letter : Type
| x : Letter
| y : Letter
| t : Letter

-- Define a word as a list of letters
def Word := List Letter

-- Define the transformation rules
inductive Transform : Word → Word → Prop
| xy_yyx : Transform (Letter.x::Letter.y::w) (Letter.y::Letter.y::Letter.x::w)
| xt_ttx : Transform (Letter.x::Letter.t::w) (Letter.t::Letter.t::Letter.x::w)
| yt_ty  : Transform (Letter.y::Letter.t::w) (Letter.t::Letter.y::w)
| refl   : ∀ w, Transform w w
| symm   : ∀ v w, Transform v w → Transform w v
| trans  : ∀ u v w, Transform u v → Transform v w → Transform u w

-- Define the theorem
theorem word_transformations :
  (¬ ∃ (w : Word), Transform [Letter.x, Letter.y] [Letter.x, Letter.t]) ∧
  (¬ ∃ (w : Word), Transform [Letter.x, Letter.y, Letter.t, Letter.x] [Letter.t, Letter.x, Letter.y, Letter.t]) ∧
  (∃ (w : Word), Transform [Letter.x, Letter.t, Letter.x, Letter.y, Letter.y] [Letter.t, Letter.t, Letter.x, Letter.y, Letter.y, Letter.y, Letter.y, Letter.x])
  := by sorry

end word_transformations_l1436_143629


namespace binomial_coefficient_inequality_l1436_143645

theorem binomial_coefficient_inequality (n k : ℕ) (h1 : n > k) (h2 : k > 0) : 
  (1 : ℝ) / (n + 1 : ℝ) * (n^n : ℝ) / ((k^k * (n-k)^(n-k)) : ℝ) < 
  (n.factorial : ℝ) / ((k.factorial * (n-k).factorial) : ℝ) ∧
  (n.factorial : ℝ) / ((k.factorial * (n-k).factorial) : ℝ) < 
  (n^n : ℝ) / ((k^k * (n-k)^(n-k)) : ℝ) :=
by sorry

end binomial_coefficient_inequality_l1436_143645


namespace additional_investment_rate_problem_l1436_143616

/-- Calculates the rate of additional investment needed to achieve a target total rate --/
def additional_investment_rate (initial_investment : ℚ) (initial_rate : ℚ) 
  (additional_investment : ℚ) (target_total_rate : ℚ) : ℚ :=
  let total_investment := initial_investment + additional_investment
  let initial_interest := initial_investment * initial_rate
  let total_desired_interest := total_investment * target_total_rate
  let additional_interest_needed := total_desired_interest - initial_interest
  additional_interest_needed / additional_investment

theorem additional_investment_rate_problem 
  (initial_investment : ℚ) 
  (initial_rate : ℚ) 
  (additional_investment : ℚ) 
  (target_total_rate : ℚ) :
  initial_investment = 8000 →
  initial_rate = 5 / 100 →
  additional_investment = 4000 →
  target_total_rate = 6 / 100 →
  additional_investment_rate initial_investment initial_rate additional_investment target_total_rate = 8 / 100 :=
by
  sorry

end additional_investment_rate_problem_l1436_143616


namespace tangent_line_to_quartic_curve_l1436_143656

/-- Given that y = 4x + b is a tangent line to y = x^4 - 1, prove that b = -4 -/
theorem tangent_line_to_quartic_curve (b : ℝ) : 
  (∃ x₀ : ℝ, (4 * x₀ + b = x₀^4 - 1) ∧ 
             (∀ x : ℝ, 4 * x + b ≥ x^4 - 1) ∧ 
             (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - x₀| ∧ |x - x₀| < δ → 4 * x + b > x^4 - 1)) → 
  b = -4 := by
  sorry

end tangent_line_to_quartic_curve_l1436_143656


namespace power_of_power_at_three_l1436_143634

theorem power_of_power_at_three :
  (3^3)^(3^3) = 27^27 := by sorry

end power_of_power_at_three_l1436_143634


namespace inequality_solution_set_l1436_143666

theorem inequality_solution_set : 
  {x : ℝ | x + 2 < 1} = {x : ℝ | x < -1} := by sorry

end inequality_solution_set_l1436_143666


namespace fraction_simplification_l1436_143679

theorem fraction_simplification (x : ℝ) (hx : x > 0) :
  (x^(3/4) - 25*x^(1/4)) / (x^(1/2) + 5*x^(1/4)) = x^(1/4) - 5 := by
  sorry

end fraction_simplification_l1436_143679


namespace parallel_line_slope_l1436_143644

/-- The slope of a line parallel to 3x - 6y = 12 is 1/2 -/
theorem parallel_line_slope (a b c : ℝ) (h : b ≠ 0) :
  (∃ k : ℝ, k ≠ 0 ∧ a = 3 * k ∧ b = -6 * k ∧ c = 12 * k) →
  (a / b : ℝ) = 1/2 := by
  sorry

end parallel_line_slope_l1436_143644


namespace chalk_per_box_l1436_143678

def total_chalk : ℕ := 3484
def full_boxes : ℕ := 194

theorem chalk_per_box : total_chalk / full_boxes = 18 := by
  sorry

end chalk_per_box_l1436_143678


namespace product_is_solution_quotient_is_solution_l1436_143653

/-- A type representing solutions of the equation x^2 - 5y^2 = 1 -/
structure Solution where
  x : ℝ
  y : ℝ
  property : x^2 - 5*y^2 = 1

/-- The product of two solutions is also a solution -/
theorem product_is_solution (s₁ s₂ : Solution) :
  ∃ (m n : ℝ), m^2 - 5*n^2 = 1 ∧ m + n * Real.sqrt 5 = (s₁.x + s₁.y * Real.sqrt 5) * (s₂.x + s₂.y * Real.sqrt 5) :=
by sorry

/-- The quotient of two solutions can be represented as p + q√5 and is also a solution -/
theorem quotient_is_solution (s₁ s₂ : Solution) (h : s₂.x^2 - 5*s₂.y^2 ≠ 0) :
  ∃ (p q : ℝ), p^2 - 5*q^2 = 1 ∧ p + q * Real.sqrt 5 = (s₁.x + s₁.y * Real.sqrt 5) / (s₂.x + s₂.y * Real.sqrt 5) :=
by sorry

end product_is_solution_quotient_is_solution_l1436_143653


namespace ceiling_floor_sum_zero_l1436_143608

theorem ceiling_floor_sum_zero : ⌈(7 : ℝ) / 3⌉ + ⌊-(7 : ℝ) / 3⌋ = 0 := by
  sorry

end ceiling_floor_sum_zero_l1436_143608


namespace rice_purchase_difference_l1436_143651

/-- Represents the price and quantity of rice from a supplier -/
structure RiceSupply where
  quantity : ℝ
  price : ℝ

/-- Calculates the total cost of rice supplies -/
def totalCost (supplies : List RiceSupply) : ℝ :=
  supplies.foldl (fun acc supply => acc + supply.quantity * supply.price) 0

/-- Represents the rice purchase scenario -/
structure RicePurchase where
  supplies : List RiceSupply
  keptRatio : ℝ
  conversionRate : ℝ

theorem rice_purchase_difference (purchase : RicePurchase) 
  (h1 : purchase.supplies = [
    ⟨15, 1.2⟩, ⟨10, 1.4⟩, ⟨12, 1.6⟩, ⟨8, 1.9⟩, ⟨5, 2.3⟩
  ])
  (h2 : purchase.keptRatio = 7/10)
  (h3 : purchase.conversionRate = 1.15) :
  let totalCostEuros := totalCost purchase.supplies
  let keptCostDollars := totalCostEuros * purchase.keptRatio * purchase.conversionRate
  let givenCostDollars := totalCostEuros * (1 - purchase.keptRatio) * purchase.conversionRate
  keptCostDollars - givenCostDollars = 35.88 := by
  sorry

end rice_purchase_difference_l1436_143651


namespace x_plus_y_value_l1436_143622

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| - 2*x + y = 1) 
  (h2 : x - |y| + y = 8) : 
  x + y = 17 ∨ x + y = 1 := by
sorry

end x_plus_y_value_l1436_143622


namespace max_value_of_f_l1436_143646

noncomputable section

variable (a : ℝ)
variable (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 * Real.exp x

theorem max_value_of_f (h : a ≠ 0) :
  (a > 0 → ∃ M, M = 4 * a * Real.exp (-2) ∧ ∀ x, f a x ≤ M) ∧
  (a < 0 → ∃ M, M = 0 ∧ ∀ x, f a x ≤ M) :=
sorry

end

end max_value_of_f_l1436_143646


namespace trajectory_of_point_l1436_143632

/-- The trajectory of a point M, given specific conditions -/
theorem trajectory_of_point (M : ℝ × ℝ) :
  (∀ (x y : ℝ), M = (x, y) →
    (x^2 + (y + 3)^2)^(1/2) = |y - 3|) →  -- M is equidistant from (0, -3) and y = 3
  (∃ (a b c : ℝ), ∀ (x y : ℝ), M = (x, y) → 
    a*x^2 + b*y + c = 0) →  -- Trajectory of M is a conic section (which includes parabolas)
  ∃ (x y : ℝ), M = (x, y) ∧ x^2 = -12*y :=
by sorry

end trajectory_of_point_l1436_143632


namespace min_score_given_average_l1436_143642

theorem min_score_given_average (x y : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 ∧ 
  y ≥ 0 ∧ y ≤ 100 ∧ 
  (69 + 53 + 69 + 71 + 78 + x + y) / 7 = 66 →
  x ≥ 22 :=
by sorry

end min_score_given_average_l1436_143642


namespace valid_outfits_count_l1436_143689

/-- The number of shirts available -/
def num_shirts : ℕ := 7

/-- The number of pairs of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 7

/-- The number of colors available for shirts and hats -/
def num_colors : ℕ := 7

/-- Calculate the number of valid outfits -/
def num_valid_outfits : ℕ := num_shirts * num_pants * num_hats - num_colors * num_pants

theorem valid_outfits_count :
  num_valid_outfits = 210 :=
by sorry

end valid_outfits_count_l1436_143689


namespace symmetric_point_line_equation_l1436_143661

/-- Given points A and M, if B is symmetric to A with respect to M, 
    and line l passes through the origin and point B, 
    then the equation of line l is 7x + 5y = 0 -/
theorem symmetric_point_line_equation 
  (A : ℝ × ℝ) (M : ℝ × ℝ) (B : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  A = (3, 1) →
  M = (4, -3) →
  B.1 = 2 * M.1 - A.1 →
  B.2 = 2 * M.2 - A.2 →
  (0, 0) ∈ l →
  B ∈ l →
  ∀ (x y : ℝ), (x, y) ∈ l ↔ 7 * x + 5 * y = 0 :=
by sorry

end symmetric_point_line_equation_l1436_143661


namespace loan_amount_calculation_l1436_143614

/-- Proves that given the initial amount, interest rate, and final amount, 
    the calculated loan amount is correct. -/
theorem loan_amount_calculation 
  (initial_amount : ℝ) 
  (interest_rate : ℝ) 
  (final_amount : ℝ) 
  (loan_amount : ℝ) : 
  initial_amount = 30 ∧ 
  interest_rate = 0.20 ∧ 
  final_amount = 33 ∧
  loan_amount = 2.50 → 
  initial_amount + loan_amount * (1 + interest_rate) = final_amount :=
by sorry

end loan_amount_calculation_l1436_143614


namespace pet_store_combinations_l1436_143639

def num_puppies : ℕ := 12
def num_kittens : ℕ := 8
def num_hamsters : ℕ := 10
def num_birds : ℕ := 5

theorem pet_store_combinations : 
  num_puppies * num_kittens * num_hamsters * num_birds * 4 * 3 * 2 * 1 = 115200 := by
  sorry

end pet_store_combinations_l1436_143639


namespace eulers_pedal_triangle_theorem_l1436_143624

/-- Euler's theorem on the area of pedal triangles -/
theorem eulers_pedal_triangle_theorem (S R d : ℝ) (hR : R > 0) : 
  ∃ (S' : ℝ), S' = (S / 4) * |1 - (d^2 / R^2)| := by
  sorry

end eulers_pedal_triangle_theorem_l1436_143624


namespace cupcake_cost_split_l1436_143628

theorem cupcake_cost_split (num_cupcakes : ℕ) (price_per_cupcake : ℚ) (num_people : ℕ) :
  num_cupcakes = 12 →
  price_per_cupcake = 3/2 →
  num_people = 2 →
  (num_cupcakes : ℚ) * price_per_cupcake / num_people = 9 := by
  sorry

end cupcake_cost_split_l1436_143628


namespace evaluate_expression_l1436_143657

theorem evaluate_expression : 
  45 * ((4 + 1/3) - (5 + 1/4)) / ((3 + 1/2) + (2 + 1/5)) = -(7 + 9/38) := by
  sorry

end evaluate_expression_l1436_143657


namespace smallest_integer_with_remainders_l1436_143613

theorem smallest_integer_with_remainders : ∃! N : ℕ+, 
  (N : ℤ) % 7 = 5 ∧ 
  (N : ℤ) % 8 = 6 ∧ 
  (N : ℤ) % 9 = 7 ∧ 
  ∀ M : ℕ+, 
    ((M : ℤ) % 7 = 5 ∧ (M : ℤ) % 8 = 6 ∧ (M : ℤ) % 9 = 7) → N ≤ M :=
by
  use 502
  sorry

end smallest_integer_with_remainders_l1436_143613


namespace problem_solution_l1436_143665

theorem problem_solution :
  -- Part 1
  (let a : ℤ := 2
   let b : ℤ := -1
   (3 * a^2 * b + (1/4) * a * b^2 - (3/4) * a * b^2 + a^2 * b) = -17) ∧
  -- Part 2
  (∀ x y : ℝ, ∃ a b : ℝ,
    (2*x^2 + a*x - y + 6) - (2*b*x^2 - 3*x + 5*y - 1) = 0 →
    5*a*b^2 - (a^2*b + 2*(a^2*b - 3*a*b^2)) = -60) := by
  sorry

end problem_solution_l1436_143665


namespace smallest_with_70_divisors_l1436_143643

/-- The number of natural divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A natural number has exactly 70 divisors -/
def has_70_divisors (n : ℕ) : Prop := num_divisors n = 70

/-- 25920 is the smallest natural number with exactly 70 divisors -/
theorem smallest_with_70_divisors : 
  has_70_divisors 25920 ∧ ∀ m < 25920, ¬has_70_divisors m :=
sorry

end smallest_with_70_divisors_l1436_143643


namespace largest_parallelogram_perimeter_l1436_143692

/-- Triangle with sides 13, 13, and 12 -/
structure Triangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- Parallelogram formed by four copies of a triangle -/
def Parallelogram (t : Triangle) :=
  { p : ℝ // ∃ (a b c d : ℝ), 
    a + b + c + d = p ∧
    a ≤ t.side1 ∧ b ≤ t.side1 ∧ c ≤ t.side2 ∧ d ≤ t.side3 }

/-- The theorem stating the largest possible perimeter of the parallelogram -/
theorem largest_parallelogram_perimeter :
  let t : Triangle := { side1 := 13, side2 := 13, side3 := 12 }
  ∀ p : Parallelogram t, p.val ≤ 76 :=
by sorry

end largest_parallelogram_perimeter_l1436_143692


namespace melanie_books_before_l1436_143654

/-- The number of books Melanie had before the yard sale -/
def books_before : ℕ := sorry

/-- The number of books Melanie bought at the yard sale -/
def books_bought : ℕ := 46

/-- The total number of books Melanie has after the yard sale -/
def books_after : ℕ := 87

/-- Theorem stating that Melanie had 41 books before the yard sale -/
theorem melanie_books_before : books_before = 41 := by sorry

end melanie_books_before_l1436_143654


namespace mary_flour_amount_l1436_143604

/-- The amount of flour Mary puts in the cake. -/
def total_flour (recipe_flour extra_flour : ℝ) : ℝ :=
  recipe_flour + extra_flour

/-- Theorem stating the total amount of flour Mary uses. -/
theorem mary_flour_amount :
  let recipe_flour : ℝ := 7.0
  let extra_flour : ℝ := 2.0
  total_flour recipe_flour extra_flour = 9.0 := by
  sorry

end mary_flour_amount_l1436_143604


namespace sum_of_squares_of_roots_l1436_143607

theorem sum_of_squares_of_roots (a b c : ℝ) : 
  (a^3 - 2*a^2 + 5*a + 7 = 0) → 
  (b^3 - 2*b^2 + 5*b + 7 = 0) → 
  (c^3 - 2*c^2 + 5*c + 7 = 0) → 
  a^2 + b^2 + c^2 = -6 :=
by
  sorry

end sum_of_squares_of_roots_l1436_143607


namespace inequality_problem_l1436_143655

theorem inequality_problem (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (h : a^4 + b^4 + c^4 ≤ 2*(a^2*b^2 + b^2*c^2 + c^2*a^2)) :
  (a ≤ b + c ∧ b ≤ a + c ∧ c ≤ a + b) ∧
  (a^2 + b^2 + c^2 ≤ 2*(a*b + b*c + c*a)) ∧
  ∃ (x y z : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧
    x^2 + y^2 + z^2 ≤ 2*(x*y + y*z + z*x) ∧
    ¬(x^4 + y^4 + z^4 ≤ 2*(x^2*y^2 + y^2*z^2 + z^2*x^2)) :=
by sorry

end inequality_problem_l1436_143655


namespace smallest_other_integer_l1436_143635

theorem smallest_other_integer (m n x : ℕ) : 
  m = 30 →
  x > 0 →
  Nat.gcd m n = x + 1 →
  Nat.lcm m n = x * (x + 1) →
  ∃ (n_min : ℕ), n_min = 6 ∧ ∀ (n' : ℕ), (
    Nat.gcd m n' = x + 1 ∧
    Nat.lcm m n' = x * (x + 1) →
    n' ≥ n_min
  ) := by sorry

end smallest_other_integer_l1436_143635


namespace expand_product_l1436_143602

theorem expand_product (x y : ℝ) : 4 * (x + 3) * (x + 2 + y) = 4 * x^2 + 4 * x * y + 20 * x + 12 * y + 24 := by
  sorry

end expand_product_l1436_143602


namespace total_cost_l1436_143630

/-- Represents the price of an enchilada in dollars -/
def enchilada_price : ℚ := sorry

/-- Represents the price of a taco in dollars -/
def taco_price : ℚ := sorry

/-- Represents the price of a drink in dollars -/
def drink_price : ℚ := sorry

/-- The first price condition: one enchilada, two tacos, and a drink cost $3.20 -/
axiom price_condition1 : enchilada_price + 2 * taco_price + drink_price = 32/10

/-- The second price condition: two enchiladas, three tacos, and a drink cost $4.90 -/
axiom price_condition2 : 2 * enchilada_price + 3 * taco_price + drink_price = 49/10

/-- Theorem stating that the cost of four enchiladas, five tacos, and two drinks is $8.30 -/
theorem total_cost : 4 * enchilada_price + 5 * taco_price + 2 * drink_price = 83/10 := by sorry

end total_cost_l1436_143630


namespace divisibility_problem_l1436_143612

theorem divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    ((a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end divisibility_problem_l1436_143612


namespace isosceles_triangle_perimeter_l1436_143691

theorem isosceles_triangle_perimeter : ∀ (a b : ℝ),
  a^2 - 9*a + 18 = 0 →
  b^2 - 9*b + 18 = 0 →
  a ≠ b →
  (∃ (leg base : ℝ), (leg = max a b ∧ base = min a b) ∧
    2*leg + base = 15) :=
by
  sorry

end isosceles_triangle_perimeter_l1436_143691


namespace max_primes_arithmetic_seq_diff_12_l1436_143687

theorem max_primes_arithmetic_seq_diff_12 :
  ∀ (seq : ℕ → ℕ) (n : ℕ),
    (∀ i < n, seq i.succ = seq i + 12) →
    (∀ i < n, Nat.Prime (seq i)) →
    n ≤ 5 :=
sorry

end max_primes_arithmetic_seq_diff_12_l1436_143687


namespace complex_symmetry_product_l1436_143615

theorem complex_symmetry_product (z₁ z₂ : ℂ) : 
  (z₁.re = -z₂.re ∧ z₁.im = z₂.im) →  -- symmetry about imaginary axis
  z₁ = 1 + 2*I →                     -- given value of z₁
  z₁ * z₂ = -5 :=                    -- product equals -5
by
  sorry

end complex_symmetry_product_l1436_143615


namespace prob_sum_greater_than_four_proof_l1436_143603

/-- The probability of rolling two dice and getting a sum greater than four -/
def prob_sum_greater_than_four : ℚ := 5/6

/-- The number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := 36

/-- The number of outcomes where the sum is less than or equal to four -/
def outcomes_sum_le_four : ℕ := 6

theorem prob_sum_greater_than_four_proof :
  prob_sum_greater_than_four = 1 - (outcomes_sum_le_four / total_outcomes) :=
by sorry

end prob_sum_greater_than_four_proof_l1436_143603


namespace f_3_range_l1436_143620

theorem f_3_range (a c : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a * x^2 - c)
  (h1 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
sorry

end f_3_range_l1436_143620


namespace multiplication_puzzle_l1436_143684

theorem multiplication_puzzle (c d : ℕ) : 
  c < 10 → d < 10 →
  (∃ n : ℕ, n < 1000 ∧ n % 100 = 8 ∧ 3 * c * 10 + c = n / d / 10) →
  c * 4 % 10 = 2 →
  (∃ x : ℕ, x < 10 ∧ 34 * d + 12 ≥ 10 * x * 10 + 60 ∧ 34 * d + 12 < 10 * x * 10 + 70) →
  c + d = 5 := by
sorry

end multiplication_puzzle_l1436_143684


namespace sum_of_smallest_and_largest_l1436_143652

-- Define the property of three consecutive even numbers
def ConsecutiveEvenNumbers (a b c : ℕ) : Prop :=
  ∃ n : ℕ, a = 2 * n ∧ b = 2 * n + 2 ∧ c = 2 * n + 4

theorem sum_of_smallest_and_largest (a b c : ℕ) :
  ConsecutiveEvenNumbers a b c → a + b + c = 1194 → a + c = 796 := by
  sorry

#check sum_of_smallest_and_largest

end sum_of_smallest_and_largest_l1436_143652


namespace complex_magnitude_plus_fraction_l1436_143649

theorem complex_magnitude_plus_fraction :
  Complex.abs (3/4 - 3*Complex.I) + 5/12 = (9*Real.sqrt 17 + 5) / 12 := by
  sorry

end complex_magnitude_plus_fraction_l1436_143649


namespace base_k_equality_l1436_143627

theorem base_k_equality (k : ℕ) : k^2 + 3*k + 2 = 30 → k = 4 := by
  sorry

end base_k_equality_l1436_143627


namespace largest_quantity_l1436_143606

theorem largest_quantity (a b c d e : ℝ) 
  (h : a - 1 = b + 2 ∧ a - 1 = c - 3 ∧ a - 1 = d + 4 ∧ a - 1 = e - 6) : 
  e = max a (max b (max c d)) :=
sorry

end largest_quantity_l1436_143606


namespace first_year_after_2020_with_digit_sum_5_l1436_143640

-- Define a function to calculate the sum of digits in a number
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- Define the property of being the first year after 2020 with digit sum 5
def isFirstYearAfter2020WithDigitSum5 (year : ℕ) : Prop :=
  year > 2020 ∧
  sumOfDigits year = 5 ∧
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 5

-- Theorem statement
theorem first_year_after_2020_with_digit_sum_5 :
  sumOfDigits 2020 = 4 →
  isFirstYearAfter2020WithDigitSum5 2021 :=
by
  sorry

end first_year_after_2020_with_digit_sum_5_l1436_143640


namespace cubic_roots_cubed_l1436_143619

/-- Given a cubic equation x³ + ax² + bx + c = 0 with roots α, β, and γ,
    the cubic equation whose roots are α³, β³, and γ³ is
    x³ + (-a³ + 3ab - 3c)x² + (-b³ + 3abc)x + c³ = 0 -/
theorem cubic_roots_cubed (a b c : ℝ) (α β γ : ℝ) : 
  (∀ x : ℝ, x^3 + a*x^2 + b*x + c = 0 ↔ x = α ∨ x = β ∨ x = γ) →
  (∀ x : ℝ, x^3 + (-a^3 + 3*a*b - 3*c)*x^2 + (-b^3 + 3*a*b*c)*x + c^3 = 0 
           ↔ x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
by sorry


end cubic_roots_cubed_l1436_143619


namespace danny_drive_to_work_l1436_143663

/-- Represents the distance Danny drives between different locations -/
structure DannyDrive where
  x : ℝ  -- Distance from Danny's house to the first friend's house
  first_to_second : ℝ := 0.5 * x
  second_to_third : ℝ := 2 * x
  third_to_fourth : ℝ  -- Will be calculated
  fourth_to_work : ℝ   -- To be proven

/-- Calculates the total distance driven up to the third friend's house -/
def total_to_third (d : DannyDrive) : ℝ :=
  d.x + d.first_to_second + d.second_to_third

/-- Theorem stating the distance Danny drives between the fourth friend's house and work -/
theorem danny_drive_to_work (d : DannyDrive) 
    (h1 : d.third_to_fourth = (1/3) * total_to_third d) 
    (h2 : d.fourth_to_work = 3 * (total_to_third d + d.third_to_fourth)) : 
  d.fourth_to_work = 14 * d.x := by
  sorry


end danny_drive_to_work_l1436_143663


namespace lcm_14_21_35_l1436_143660

theorem lcm_14_21_35 : Nat.lcm 14 (Nat.lcm 21 35) = 210 := by
  sorry

end lcm_14_21_35_l1436_143660


namespace sample_size_theorem_l1436_143696

theorem sample_size_theorem (N : ℕ) (sample_size : ℕ) (prob : ℚ) : 
  sample_size = 30 → prob = 1/4 → N * prob = sample_size → N = 120 := by
  sorry

end sample_size_theorem_l1436_143696


namespace no_solution_condition_l1436_143670

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ -4 → 1 / (x - 4) + m / (x + 4) ≠ (m + 3) / (x^2 - 16)) ↔ 
  (m = -1 ∨ m = 5 ∨ m = -1/3) :=
sorry

end no_solution_condition_l1436_143670


namespace max_parts_formula_initial_values_correct_l1436_143631

/-- The maximum number of parts that n ellipses can divide a plane into -/
def max_parts (n : ℕ+) : ℕ :=
  2 * n.val * n.val - 2 * n.val + 2

/-- Theorem stating the formula for the maximum number of parts -/
theorem max_parts_formula (n : ℕ+) : max_parts n = 2 * n.val * n.val - 2 * n.val + 2 := by
  sorry

/-- The first few values of the sequence are correct -/
theorem initial_values_correct :
  max_parts 1 = 2 ∧ max_parts 2 = 6 ∧ max_parts 3 = 14 ∧ max_parts 4 = 26 := by
  sorry

end max_parts_formula_initial_values_correct_l1436_143631


namespace conference_theorem_l1436_143618

/-- Represents the state of knowledge among scientists at a conference --/
structure ConferenceState where
  total_scientists : Nat
  initial_knowers : Nat
  final_knowers : Nat

/-- Calculates the probability of a specific number of scientists knowing the news after pairing --/
noncomputable def probability_of_final_knowers (state : ConferenceState) : ℚ :=
  sorry

/-- Calculates the expected number of scientists knowing the news after pairing --/
noncomputable def expected_final_knowers (state : ConferenceState) : ℚ :=
  sorry

theorem conference_theorem (state : ConferenceState) 
  (h1 : state.total_scientists = 18) 
  (h2 : state.initial_knowers = 10) : 
  (probability_of_final_knowers {total_scientists := 18, initial_knowers := 10, final_knowers := 13} = 0) ∧ 
  (probability_of_final_knowers {total_scientists := 18, initial_knowers := 10, final_knowers := 14} = 1120/2431) ∧
  (expected_final_knowers {total_scientists := 18, initial_knowers := 10, final_knowers := 0} = 14 + 12/17) :=
by sorry

end conference_theorem_l1436_143618


namespace value_of_a_l1436_143699

theorem value_of_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^3 / b = 1) (h2 : b^3 / c = 8) (h3 : c^3 / a = 27) :
  a = (24^(1/8 : ℝ))^(1/3 : ℝ) := by
sorry

end value_of_a_l1436_143699


namespace inequality_solution_set_l1436_143669

theorem inequality_solution_set (x : ℝ) :
  |x + 3| - |2*x - 1| < x/2 + 1 ↔ x < -2/5 ∨ x > 2 := by
  sorry

end inequality_solution_set_l1436_143669


namespace peter_twice_harriet_age_l1436_143673

/- Define the current ages and time span -/
def mother_age : ℕ := 60
def harriet_age : ℕ := 13
def years_passed : ℕ := 4

/- Define Peter's current age based on the given condition -/
def peter_age : ℕ := mother_age / 2

/- Define future ages -/
def peter_future_age : ℕ := peter_age + years_passed
def harriet_future_age : ℕ := harriet_age + years_passed

/- Theorem to prove -/
theorem peter_twice_harriet_age : 
  peter_future_age = 2 * harriet_future_age := by
  sorry


end peter_twice_harriet_age_l1436_143673


namespace multiply_three_six_and_quarter_l1436_143676

theorem multiply_three_six_and_quarter : 3.6 * 0.25 = 0.9 := by
  sorry

end multiply_three_six_and_quarter_l1436_143676


namespace arithmetic_geometric_sequence_property_l1436_143658

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b^2 = a * c

theorem arithmetic_geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a 2)
  (h2 : geometric_sequence (a 1) (a 3) (a 4)) :
  a 2 = -6 := by
sorry

end arithmetic_geometric_sequence_property_l1436_143658


namespace compare_expressions_l1436_143617

theorem compare_expressions (x : ℝ) (h : x > 1) : x^3 + 6*x > x^2 + 6 := by
  sorry

end compare_expressions_l1436_143617


namespace sets_equality_implies_coefficients_l1436_143677

def A : Set ℝ := {-1, 3}

def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem sets_equality_implies_coefficients (a b : ℝ) : 
  A = B a b → a = -2 ∧ b = -3 := by
  sorry

end sets_equality_implies_coefficients_l1436_143677


namespace fraction_addition_l1436_143667

theorem fraction_addition (y C D : ℚ) : 
  (6 * y - 15) / (3 * y^3 - 13 * y^2 + 4 * y + 12) = C / (y + 3) + D / (3 * y^2 - 10 * y + 4) →
  C = -3/17 ∧ D = 81/17 := by
sorry

end fraction_addition_l1436_143667


namespace twenty_solutions_implies_twenty_or_twentythree_l1436_143674

/-- Given a positive integer n, count_solutions n returns the number of solutions
    to the equation 3x + 3y + 2z = n in positive integers x, y, and z -/
def count_solutions (n : ℕ+) : ℕ :=
  sorry

theorem twenty_solutions_implies_twenty_or_twentythree (n : ℕ+) :
  count_solutions n = 20 → n = 20 ∨ n = 23 := by
  sorry

end twenty_solutions_implies_twenty_or_twentythree_l1436_143674


namespace complement_intersection_theorem_l1436_143693

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {0, 2, 4} := by sorry

end complement_intersection_theorem_l1436_143693


namespace lowest_degree_is_four_l1436_143698

/-- A polynomial with coefficients in ℤ -/
def IntPolynomial := Polynomial ℤ

/-- The set of coefficients of a polynomial -/
def coefficientSet (P : IntPolynomial) : Set ℤ :=
  {a : ℤ | ∃ (i : ℕ), a = P.coeff i}

/-- The property that a polynomial satisfies the given conditions -/
def satisfiesCondition (P : IntPolynomial) : Prop :=
  ∃ (b : ℤ), 
    (∃ (x y : ℤ), x ∈ coefficientSet P ∧ y ∈ coefficientSet P ∧ x < b ∧ b < y) ∧
    b ∉ coefficientSet P

/-- The theorem stating that the lowest degree of a polynomial satisfying the condition is 4 -/
theorem lowest_degree_is_four :
  ∃ (P : IntPolynomial), satisfiesCondition P ∧ P.degree = 4 ∧
  ∀ (Q : IntPolynomial), satisfiesCondition Q → Q.degree ≥ 4 :=
sorry

end lowest_degree_is_four_l1436_143698


namespace equation_solutions_l1436_143682

theorem equation_solutions : 
  let f (x : ℝ) := 
    10 / (Real.sqrt (x - 10) - 10) + 
    2 / (Real.sqrt (x - 10) - 5) + 
    14 / (Real.sqrt (x - 10) + 5) + 
    20 / (Real.sqrt (x - 10) + 10)
  ∀ x : ℝ, f x = 0 ↔ (x = 190 / 9 ∨ x = 5060 / 256) :=
by sorry

end equation_solutions_l1436_143682


namespace plane_not_perp_implies_no_perp_line_l1436_143623

-- Define planes and lines
variable (α β : Set (ℝ × ℝ × ℝ))
variable (l : Set (ℝ × ℝ × ℝ))

-- Define perpendicularity for planes and lines
def perpendicular_planes (p q : Set (ℝ × ℝ × ℝ)) : Prop := sorry
def perpendicular_line_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Define a line being in a plane
def line_in_plane (l : Set (ℝ × ℝ × ℝ)) (p : Set (ℝ × ℝ × ℝ)) : Prop := sorry

-- Theorem statement
theorem plane_not_perp_implies_no_perp_line :
  ¬(perpendicular_planes α β) →
  ¬∃ l, line_in_plane l α ∧ perpendicular_line_plane l β :=
sorry

end plane_not_perp_implies_no_perp_line_l1436_143623


namespace quadratic_inequality_equivalence_l1436_143633

theorem quadratic_inequality_equivalence (x : ℝ) : 
  x^2 - 50*x + 625 ≤ 25 ↔ 20 ≤ x ∧ x ≤ 30 := by
sorry

end quadratic_inequality_equivalence_l1436_143633


namespace complex_fraction_equality_l1436_143694

theorem complex_fraction_equality (c d : ℂ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (h : c^2 - c*d + d^2 = 0) : 
  (c^12 + d^12) / (c + d)^12 = 2 / 81 := by
  sorry

end complex_fraction_equality_l1436_143694


namespace geometric_sequence_ratio_l1436_143609

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (∀ n, a n > 0) →
  a 1 = 1 →
  a 5 = 16 →
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 :=
by sorry

end geometric_sequence_ratio_l1436_143609


namespace smallest_divisible_by_all_is_divisible_by_all_168_smallest_number_of_books_l1436_143601

def is_divisible_by_all (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0

theorem smallest_divisible_by_all :
  ∀ m : ℕ, m > 0 → is_divisible_by_all m → m ≥ 168 :=
by sorry

theorem is_divisible_by_all_168 : is_divisible_by_all 168 :=
by sorry

theorem smallest_number_of_books : 
  ∃! n : ℕ, n > 0 ∧ is_divisible_by_all n ∧ ∀ m : ℕ, m > 0 → is_divisible_by_all m → m ≥ n :=
by sorry

end smallest_divisible_by_all_is_divisible_by_all_168_smallest_number_of_books_l1436_143601


namespace arithmetic_sequence_sum_l1436_143668

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + 2 * a 6 + a 10 = 120) →
  (a 3 + a 9 = 60) :=
by
  sorry

end arithmetic_sequence_sum_l1436_143668
