import Mathlib

namespace NUMINAMATH_CALUDE_fifth_day_distance_l3641_364119

/-- Represents the daily walking distance sequence -/
def walkingSequence (d : ℕ) : ℕ → ℕ := fun n => 100 + (n - 1) * d

/-- The sum of the first n terms of the walking sequence -/
def walkingSum (d : ℕ) (n : ℕ) : ℕ := n * 100 + n * (n - 1) / 2 * d

theorem fifth_day_distance (d : ℕ) :
  walkingSum d 9 = 1260 → walkingSequence d 5 = 660 :=
by sorry

end NUMINAMATH_CALUDE_fifth_day_distance_l3641_364119


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l3641_364197

theorem complete_square_with_integer (y : ℝ) : ∃ (a b : ℤ), y^2 + 12*y + 50 = (y + ↑a)^2 + ↑b ∧ b = 14 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l3641_364197


namespace NUMINAMATH_CALUDE_economics_test_absentees_l3641_364118

theorem economics_test_absentees (total_students : ℕ) (q1_correct : ℕ) (q2_correct : ℕ) (both_correct : ℕ)
  (h1 : total_students = 40)
  (h2 : q1_correct = 30)
  (h3 : q2_correct = 29)
  (h4 : both_correct = 29)
  : total_students - (q1_correct + q2_correct - both_correct) = 10 := by
  sorry

#check economics_test_absentees

end NUMINAMATH_CALUDE_economics_test_absentees_l3641_364118


namespace NUMINAMATH_CALUDE_unsold_books_l3641_364162

theorem unsold_books (total : ℕ) (sold : ℕ) (price : ℕ) (revenue : ℕ) : 
  (2 : ℕ) * sold = 3 * total ∧ 
  price = 4 ∧ 
  revenue = 288 ∧ 
  sold * price = revenue → 
  total - sold = 36 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_l3641_364162


namespace NUMINAMATH_CALUDE_competition_probabilities_l3641_364160

/-- A student participates in a science and knowledge competition -/
structure Competition where
  /-- Probability of answering the first question correctly -/
  p1 : ℝ
  /-- Probability of answering the second question correctly -/
  p2 : ℝ
  /-- Probability of answering the third question correctly -/
  p3 : ℝ
  /-- All probabilities are between 0 and 1 -/
  h1 : 0 ≤ p1 ∧ p1 ≤ 1
  h2 : 0 ≤ p2 ∧ p2 ≤ 1
  h3 : 0 ≤ p3 ∧ p3 ≤ 1

/-- The probability of scoring 200 points in the competition -/
def prob_200_points (c : Competition) : ℝ :=
  c.p1 * c.p2 * (1 - c.p3) + (1 - c.p1) * (1 - c.p2) * c.p3

/-- The probability of scoring at least 300 points in the competition -/
def prob_at_least_300_points (c : Competition) : ℝ :=
  c.p1 * (1 - c.p2) * c.p3 + (1 - c.p1) * c.p2 * c.p3 + c.p1 * c.p2 * c.p3

/-- The main theorem about the probabilities in the competition -/
theorem competition_probabilities (c : Competition) 
    (h_p1 : c.p1 = 0.8) (h_p2 : c.p2 = 0.7) (h_p3 : c.p3 = 0.6) : 
    prob_200_points c = 0.26 ∧ prob_at_least_300_points c = 0.564 := by
  sorry


end NUMINAMATH_CALUDE_competition_probabilities_l3641_364160


namespace NUMINAMATH_CALUDE_min_value_expression_l3641_364189

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3641_364189


namespace NUMINAMATH_CALUDE_rectangle_area_is_143_l3641_364176

/-- Represents a square in the rectangle --/
structure Square where
  sideLength : ℝ
  area : ℝ
  area_eq : area = sideLength ^ 2

/-- Represents the rectangle ABCD --/
structure Rectangle where
  squares : Fin 6 → Square
  smallestSquare : squares 0 = { sideLength := 1, area := 1, area_eq := by simp }
  unequal : ∀ i j, i ≠ j → squares i ≠ squares j
  width : ℝ
  height : ℝ
  area : ℝ
  area_eq : area = width * height
  width_eq : width = (squares 1).sideLength + (squares 2).sideLength + (squares 0).sideLength
  height_eq : height = (squares 3).sideLength + (squares 0).sideLength + 1

theorem rectangle_area_is_143 (rect : Rectangle) : rect.area = 143 := by
  sorry

#check rectangle_area_is_143

end NUMINAMATH_CALUDE_rectangle_area_is_143_l3641_364176


namespace NUMINAMATH_CALUDE_least_possible_lcm_l3641_364191

theorem least_possible_lcm (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 30 ∧ (∀ (x y : ℕ), Nat.lcm x b = 20 → Nat.lcm b y = 24 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
by sorry

end NUMINAMATH_CALUDE_least_possible_lcm_l3641_364191


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3641_364153

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → 45 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3641_364153


namespace NUMINAMATH_CALUDE_food_supply_duration_l3641_364117

/-- Proves that the initial food supply was planned to last for 22 days given the problem conditions -/
theorem food_supply_duration (initial_men : ℕ) (additional_men : ℕ) (remaining_days : ℕ) : 
  initial_men = 760 → 
  additional_men = 2280 → 
  remaining_days = 5 → 
  (initial_men * (22 - 2) : ℕ) = (initial_men + additional_men) * remaining_days :=
by sorry

end NUMINAMATH_CALUDE_food_supply_duration_l3641_364117


namespace NUMINAMATH_CALUDE_initial_number_of_persons_l3641_364180

theorem initial_number_of_persons (average_increase : ℝ) (weight_difference : ℝ) : 
  average_increase = 1.5 → weight_difference = 12 → 
  (average_increase * initial_persons = weight_difference) → initial_persons = 8 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_of_persons_l3641_364180


namespace NUMINAMATH_CALUDE_total_balls_is_135_l3641_364193

/-- Represents a school with elementary and middle school classes -/
structure School where
  elementary : Nat
  middle : Nat

/-- Calculates the total number of soccer balls donated to all schools -/
def totalSoccerBalls (schools : List School) (ballsPerClass : Nat) : Nat :=
  schools.foldl (fun acc school => acc + (school.elementary + school.middle) * ballsPerClass) 0

/-- Theorem: The total number of soccer balls donated is 135 -/
theorem total_balls_is_135 (schoolA schoolB schoolC : School) (ballsPerClass : Nat) :
  schoolA.elementary = 4 →
  schoolA.middle = 5 →
  schoolB.elementary = 5 →
  schoolB.middle = 3 →
  schoolC.elementary = 6 →
  schoolC.middle = 4 →
  ballsPerClass = 5 →
  totalSoccerBalls [schoolA, schoolB, schoolC] ballsPerClass = 135 := by
  sorry


end NUMINAMATH_CALUDE_total_balls_is_135_l3641_364193


namespace NUMINAMATH_CALUDE_rodney_ian_money_difference_l3641_364198

def rodney_ian_difference (jessica_money : ℕ) (jessica_rodney_diff : ℕ) : ℕ :=
  let rodney_money := jessica_money - jessica_rodney_diff
  let ian_money := jessica_money / 2
  rodney_money - ian_money

theorem rodney_ian_money_difference :
  rodney_ian_difference 100 15 = 35 :=
by sorry

end NUMINAMATH_CALUDE_rodney_ian_money_difference_l3641_364198


namespace NUMINAMATH_CALUDE_min_folds_to_exceed_target_l3641_364161

def paper_thickness : ℝ := 0.1
def target_thickness : ℝ := 12

def thickness_after_folds (n : ℕ) : ℝ :=
  paper_thickness * (2 ^ n)

theorem min_folds_to_exceed_target : 
  ∀ n : ℕ, (thickness_after_folds n > target_thickness) ↔ (n ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_min_folds_to_exceed_target_l3641_364161


namespace NUMINAMATH_CALUDE_triangle_abc_is_obtuse_l3641_364127

theorem triangle_abc_is_obtuse (A B C : Real) (a b c : Real) :
  B = Real.pi / 6 →  -- 30 degrees in radians
  b = Real.sqrt 2 →
  c = 2 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = Real.pi →
  a / Real.sin A = b / Real.sin B →
  b / Real.sin B = c / Real.sin C →
  c / Real.sin C = a / Real.sin A →
  A > Real.pi / 2 ∨ B > Real.pi / 2 ∨ C > Real.pi / 2 := by
  sorry

#check triangle_abc_is_obtuse

end NUMINAMATH_CALUDE_triangle_abc_is_obtuse_l3641_364127


namespace NUMINAMATH_CALUDE_water_mixture_percentage_l3641_364170

theorem water_mixture_percentage (initial_volume : ℝ) (initial_water_percentage : ℝ) (added_water : ℝ) : 
  initial_volume = 125 →
  initial_water_percentage = 0.20 →
  added_water = 8.333333333333334 →
  let initial_water := initial_volume * initial_water_percentage
  let new_water := initial_water + added_water
  let new_volume := initial_volume + added_water
  new_water / new_volume = 0.25 := by
sorry

end NUMINAMATH_CALUDE_water_mixture_percentage_l3641_364170


namespace NUMINAMATH_CALUDE_geoffrey_remaining_money_l3641_364167

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial_amount : ℕ) (num_items : ℕ) (item_cost : ℕ) : ℕ :=
  initial_amount - num_items * item_cost

/-- Proves that Geoffrey has €20 left after his purchase -/
theorem geoffrey_remaining_money :
  remaining_money 125 3 35 = 20 := by
  sorry

end NUMINAMATH_CALUDE_geoffrey_remaining_money_l3641_364167


namespace NUMINAMATH_CALUDE_negation_even_prime_l3641_364155

theorem negation_even_prime :
  (¬ ∃ n : ℕ, Even n ∧ Prime n) ↔ (∀ n : ℕ, Even n → ¬ Prime n) :=
by sorry

end NUMINAMATH_CALUDE_negation_even_prime_l3641_364155


namespace NUMINAMATH_CALUDE_equation_solution_l3641_364190

theorem equation_solution (x y : ℝ) :
  2^(-Real.sin x^2) + 2^(-Real.cos x^2) = Real.sin y + Real.cos y →
  ∃ (k l : ℤ), x = π/4 + k*(π/2) ∧ y = π/4 + l*(2*π) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3641_364190


namespace NUMINAMATH_CALUDE_least_valid_number_l3641_364173

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  ∃ d₁ d₂ d₃ d₄ : ℕ,
    n = 1000 * d₁ + 100 * d₂ + 10 * d₃ + d₄ ∧
    (d₁ = 5 ∨ d₂ = 5 ∨ d₃ = 5 ∨ d₄ = 5) ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄ ∧
    d₁ ≠ 0 ∧ d₂ ≠ 0 ∧ d₃ ≠ 0 ∧ d₄ ≠ 0 ∧
    n % d₁ = 0 ∧ n % d₂ = 0 ∧ n % d₃ = 0 ∧ n % d₄ = 0 ∧
    n % 8 = 0

theorem least_valid_number :
  ∀ m : ℕ, is_valid_number m → m ≥ 5136 :=
by sorry

end NUMINAMATH_CALUDE_least_valid_number_l3641_364173


namespace NUMINAMATH_CALUDE_function_max_min_difference_l3641_364132

theorem function_max_min_difference (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  (∃ (max min : ℝ), (∀ x ∈ Set.Icc 1 2, f x ≤ max ∧ f x ≥ min) ∧ max - min = a / 2) →
  a = 3/2 ∨ a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_function_max_min_difference_l3641_364132


namespace NUMINAMATH_CALUDE_problem_statement_l3641_364136

theorem problem_statement (k : ℕ) : 
  (18^k : ℕ) ∣ 624938 → 2^k - k^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3641_364136


namespace NUMINAMATH_CALUDE_communication_scenarios_10_20_l3641_364126

/-- The number of different possible communication scenarios between two groups of radio operators. -/
def communication_scenarios (operators_a : ℕ) (operators_b : ℕ) : ℕ :=
  2^(operators_a * operators_b)

/-- Theorem stating the number of communication scenarios for 10 operators at A and 20 at B. -/
theorem communication_scenarios_10_20 :
  communication_scenarios 10 20 = 2^200 := by
  sorry

end NUMINAMATH_CALUDE_communication_scenarios_10_20_l3641_364126


namespace NUMINAMATH_CALUDE_impossible_all_even_impossible_all_divisible_by_three_l3641_364124

-- Define the cube structure
structure Cube :=
  (vertices : Fin 8 → ℕ)

-- Define the initial state of the cube
def initial_cube : Cube :=
  { vertices := λ i => if i = 0 then 1 else 0 }

-- Define the operation of adding 1 to both ends of an edge
def add_to_edge (c : Cube) (v1 v2 : Fin 8) : Cube :=
  { vertices := λ i => if i = v1 || i = v2 then c.vertices i + 1 else c.vertices i }

-- Define the property of all numbers being divisible by 2
def all_even (c : Cube) : Prop :=
  ∀ i, c.vertices i % 2 = 0

-- Define the property of all numbers being divisible by 3
def all_divisible_by_three (c : Cube) : Prop :=
  ∀ i, c.vertices i % 3 = 0

-- Theorem stating it's impossible to make all numbers even
theorem impossible_all_even :
  ¬ ∃ (operations : List (Fin 8 × Fin 8)), 
    all_even (operations.foldl (λ c (v1, v2) => add_to_edge c v1 v2) initial_cube) :=
sorry

-- Theorem stating it's impossible to make all numbers divisible by 3
theorem impossible_all_divisible_by_three :
  ¬ ∃ (operations : List (Fin 8 × Fin 8)), 
    all_divisible_by_three (operations.foldl (λ c (v1, v2) => add_to_edge c v1 v2) initial_cube) :=
sorry

end NUMINAMATH_CALUDE_impossible_all_even_impossible_all_divisible_by_three_l3641_364124


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l3641_364144

/-- The equation of the tangent line to y = x³ at (1, 1) is 3x - y - 2 = 0 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3) → -- Curve equation
  (∃ (m b : ℝ), ∀ (x' y' : ℝ), 
    (y' = m * (x' - 1) + 1) ∧ -- Point-slope form of tangent line
    (y' = x'^3 → x' = 1) → -- Tangent point (1, 1)
    (3 * x' - y' - 2 = 0)) -- Equation of the tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l3641_364144


namespace NUMINAMATH_CALUDE_equilateral_triangle_rotation_volume_l3641_364148

/-- The volume of a solid obtained by rotating an equilateral triangle -/
theorem equilateral_triangle_rotation_volume (a : ℝ) (ha : a > 0) :
  let h := a * Real.sqrt 3 / 2
  let V := 2 * π * (a / 2)^2 * h
  V = π * a^3 * Real.sqrt 3 / 4 := by
  sorry

#check equilateral_triangle_rotation_volume

end NUMINAMATH_CALUDE_equilateral_triangle_rotation_volume_l3641_364148


namespace NUMINAMATH_CALUDE_sunset_time_l3641_364156

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

def sunrise : Time := { hours := 6, minutes := 45 }
def daylight : Duration := { hours := 11, minutes := 12 }

theorem sunset_time :
  addTime sunrise daylight = { hours := 17, minutes := 57 } := by
  sorry

end NUMINAMATH_CALUDE_sunset_time_l3641_364156


namespace NUMINAMATH_CALUDE_arithmetic_and_another_seq_sum_l3641_364138

/-- An arithmetic sequence with the given properties -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  a 2 = 2 ∧ 
  ∃ q : ℝ, q * a 2 = a 3 + 1 ∧ q * (a 3 + 1) = a 4

/-- Another sequence with the given sum property -/
def another_seq (b : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 1 / S n = 1 / n - 1 / (n + 1)

/-- The main theorem -/
theorem arithmetic_and_another_seq_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_seq a → another_seq b S → a 8 + b 8 = 144 := by sorry

end NUMINAMATH_CALUDE_arithmetic_and_another_seq_sum_l3641_364138


namespace NUMINAMATH_CALUDE_find_m_l3641_364104

def U : Set Nat := {1, 2, 3}

def A (m : Nat) : Set Nat := {1, m}

def complement_A : Set Nat := {2}

theorem find_m :
  ∃ (m : Nat), m ∈ U ∧ A m ∪ complement_A = U ∧ A m ∩ complement_A = ∅ ∧ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l3641_364104


namespace NUMINAMATH_CALUDE_total_piggy_bank_value_l3641_364171

/-- Represents the capacity of a piggy bank for different coin types -/
structure PiggyBank where
  pennies : Nat
  dimes : Nat
  nickels : Nat
  quarters : Nat

/-- Calculates the total value in a piggy bank -/
def piggyBankValue (pb : PiggyBank) : Rat :=
  pb.pennies * 1 / 100 + pb.dimes * 10 / 100 + pb.nickels * 5 / 100 + pb.quarters * 25 / 100

/-- The first piggy bank -/
def piggyBank1 : PiggyBank := ⟨100, 50, 20, 10⟩

/-- The second piggy bank -/
def piggyBank2 : PiggyBank := ⟨150, 30, 40, 15⟩

/-- The third piggy bank -/
def piggyBank3 : PiggyBank := ⟨200, 60, 10, 20⟩

/-- Theorem stating that the total value in all three piggy banks is $33.25 -/
theorem total_piggy_bank_value :
  piggyBankValue piggyBank1 + piggyBankValue piggyBank2 + piggyBankValue piggyBank3 = 3325 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_piggy_bank_value_l3641_364171


namespace NUMINAMATH_CALUDE_paint_coverage_l3641_364178

/-- Proves that a gallon of paint covers 400 square feet given the problem conditions -/
theorem paint_coverage 
  (paint_cost : ℝ) 
  (wall_area : ℝ) 
  (coats : ℕ) 
  (contribution : ℝ) :
  paint_cost = 45 →
  wall_area = 1600 →
  coats = 2 →
  contribution = 180 →
  (2 * contribution) / paint_cost * wall_area * coats / ((2 * contribution) / paint_cost) = 400 :=
by
  sorry

#check paint_coverage

end NUMINAMATH_CALUDE_paint_coverage_l3641_364178


namespace NUMINAMATH_CALUDE_mixture_ratio_l3641_364188

/-- Given a mixture with initial volume and ratio, and additional water added, 
    calculate the new ratio of components -/
theorem mixture_ratio (initial_volume : ℚ) (milk_ratio water_ratio juice_ratio : ℕ) 
                      (added_water : ℚ) : 
  initial_volume = 60 ∧ 
  milk_ratio = 3 ∧ 
  water_ratio = 2 ∧ 
  juice_ratio = 1 ∧ 
  added_water = 24 →
  ∃ (new_milk new_water new_juice : ℚ),
    new_milk / 2 = 15 ∧
    new_water / 2 = 22 ∧
    new_juice / 2 = 5 ∧
    new_milk + new_water + new_juice = initial_volume + added_water :=
by sorry

end NUMINAMATH_CALUDE_mixture_ratio_l3641_364188


namespace NUMINAMATH_CALUDE_given_point_in_second_quadrant_l3641_364109

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The given point -/
def given_point : Point :=
  { x := -3, y := 2 }

/-- Theorem: The given point is in the second quadrant -/
theorem given_point_in_second_quadrant :
  is_in_second_quadrant given_point := by
  sorry

end NUMINAMATH_CALUDE_given_point_in_second_quadrant_l3641_364109


namespace NUMINAMATH_CALUDE_coeff_x4_product_l3641_364195

def p (x : ℝ) : ℝ := x^5 - 2*x^4 + 4*x^3 - 5*x^2 + 2*x - 1
def q (x : ℝ) : ℝ := 3*x^4 - x^3 + 2*x^2 + 6*x - 5

theorem coeff_x4_product (x : ℝ) : 
  ∃ (a b c d e : ℝ), p x * q x = a*x^5 + 19*x^4 + b*x^3 + c*x^2 + d*x + e := by
  sorry

end NUMINAMATH_CALUDE_coeff_x4_product_l3641_364195


namespace NUMINAMATH_CALUDE_mirror_solution_l3641_364164

/-- Represents the number of reflections seen in the house of mirrors --/
structure Reflections where
  sarah_tall : ℕ
  sarah_wide : ℕ
  ellie_tall : ℕ
  ellie_wide : ℕ
  tall_visits : ℕ
  wide_visits : ℕ
  total : ℕ

/-- The house of mirrors problem --/
def mirror_problem : Reflections where
  sarah_tall := 10
  sarah_wide := 5
  ellie_tall := 6  -- This is what we want to prove
  ellie_wide := 3
  tall_visits := 3
  wide_visits := 5
  total := 88

/-- Theorem stating that the given configuration solves the mirror problem --/
theorem mirror_solution :
  let r := mirror_problem
  r.sarah_tall * r.tall_visits + r.sarah_wide * r.wide_visits +
  r.ellie_tall * r.tall_visits + r.ellie_wide * r.wide_visits = r.total :=
by sorry

end NUMINAMATH_CALUDE_mirror_solution_l3641_364164


namespace NUMINAMATH_CALUDE_sharons_drive_distance_l3641_364146

theorem sharons_drive_distance (usual_time : ℝ) (actual_time : ℝ) 
  (h1 : usual_time = 180)
  (h2 : actual_time = 300)
  (h3 : ∃ (usual_speed : ℝ), 
    actual_time = 
      (1/3 * usual_time) + 
      (1/3 * usual_time * usual_speed / (usual_speed - 25)) + 
      (1/3 * usual_time * usual_speed / (usual_speed + 10)))
  : ∃ (distance : ℝ), distance = 135 := by
  sorry

end NUMINAMATH_CALUDE_sharons_drive_distance_l3641_364146


namespace NUMINAMATH_CALUDE_triangle_area_form_l3641_364131

/-- The radius of each circle -/
def r : ℝ := 44

/-- The side length of the equilateral triangle -/
noncomputable def s : ℝ := 2 * r * Real.sqrt 3

/-- The area of the equilateral triangle -/
noncomputable def area : ℝ := (s^2 * Real.sqrt 3) / 4

/-- Theorem stating the form of the area -/
theorem triangle_area_form :
  ∃ (a b : ℕ), area = Real.sqrt a + Real.sqrt b :=
sorry

end NUMINAMATH_CALUDE_triangle_area_form_l3641_364131


namespace NUMINAMATH_CALUDE_cats_sold_during_sale_l3641_364163

theorem cats_sold_during_sale 
  (initial_siamese : ℕ) 
  (initial_house : ℕ) 
  (cats_left : ℕ) 
  (h1 : initial_siamese = 13)
  (h2 : initial_house = 5)
  (h3 : cats_left = 8) :
  initial_siamese + initial_house - cats_left = 10 := by
sorry

end NUMINAMATH_CALUDE_cats_sold_during_sale_l3641_364163


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3641_364123

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 1 + a 2 + a 3 + a 4 = 30) →
  (a 2 + a 3 = 15) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3641_364123


namespace NUMINAMATH_CALUDE_cafe_prices_l3641_364110

/-- The cost of items at a roadside cafe -/
structure CafePrices where
  sandwich : ℕ
  coffee : ℕ
  donut : ℕ

/-- The problem statement -/
theorem cafe_prices (p : CafePrices) : 
  4 * p.sandwich + p.coffee + 10 * p.donut = 169 ∧ 
  3 * p.sandwich + p.coffee + 7 * p.donut = 126 →
  p.sandwich + p.coffee + p.donut = 40 := by
  sorry

end NUMINAMATH_CALUDE_cafe_prices_l3641_364110


namespace NUMINAMATH_CALUDE_new_students_average_age_l3641_364172

-- Define the given conditions
def original_average : ℕ := 40
def new_students : ℕ := 17
def average_decrease : ℕ := 4
def original_strength : ℕ := 17

-- Define the theorem
theorem new_students_average_age :
  let new_average := original_average - average_decrease
  let total_students := original_strength + new_students
  let original_total_age := original_strength * original_average
  let new_total_age := total_students * new_average
  (new_total_age - original_total_age) / new_students = 32 := by
  sorry

end NUMINAMATH_CALUDE_new_students_average_age_l3641_364172


namespace NUMINAMATH_CALUDE_motel_rent_theorem_l3641_364196

/-- Represents the total rent charged by a motel on a specific night -/
def TotalRent : ℕ → ℕ → ℕ 
  | r40, r60 => 40 * r40 + 60 * r60

/-- Represents the reduced total rent after changing 10 rooms from $60 to $40 -/
def ReducedRent : ℕ → ℕ → ℕ 
  | r40, r60 => 40 * (r40 + 10) + 60 * (r60 - 10)

theorem motel_rent_theorem (r40 r60 : ℕ) :
  (TotalRent r40 r60 - ReducedRent r40 r60 = 200) → 
  (ReducedRent r40 r60 = (9 * TotalRent r40 r60) / 10) → 
  TotalRent r40 r60 = 2000 := by
  sorry

#check motel_rent_theorem

end NUMINAMATH_CALUDE_motel_rent_theorem_l3641_364196


namespace NUMINAMATH_CALUDE_equal_intercepts_iff_area_two_iff_l3641_364145

/-- The line equation type -/
structure LineEquation where
  a : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y = -(a + 1) * x + (2 - a)

/-- Equal intercepts condition -/
def has_equal_intercepts (l : LineEquation) : Prop :=
  ∃ x y : ℝ, l.eq x 0 ∧ l.eq 0 y ∧ x = y

/-- Triangle area condition -/
def has_area_two (l : LineEquation) : Prop :=
  abs ((2 - l.a) * (2 - l.a)) / (2 * abs (l.a + 1)) = 2

/-- Theorem for equal intercepts -/
theorem equal_intercepts_iff (l : LineEquation) :
  has_equal_intercepts l ↔ l.a = 2 ∨ l.a = 0 :=
sorry

/-- Theorem for area of 2 -/
theorem area_two_iff (l : LineEquation) :
  has_area_two l ↔ l.a = 8 ∨ l.a = 0 :=
sorry

end NUMINAMATH_CALUDE_equal_intercepts_iff_area_two_iff_l3641_364145


namespace NUMINAMATH_CALUDE_drink_packing_l3641_364175

/-- The number of liters of Maaza -/
def maaza : ℕ := 215

/-- The number of liters of Pepsi -/
def pepsi : ℕ := 547

/-- The number of liters of Sprite -/
def sprite : ℕ := 991

/-- The least number of cans required to pack all drinks -/
def least_cans : ℕ := maaza + pepsi + sprite

theorem drink_packing :
  (∃ (can_size : ℕ), can_size > 0 ∧
    maaza % can_size = 0 ∧
    pepsi % can_size = 0 ∧
    sprite % can_size = 0) →
  least_cans = 1753 :=
sorry

end NUMINAMATH_CALUDE_drink_packing_l3641_364175


namespace NUMINAMATH_CALUDE_walnut_trees_planted_l3641_364135

/-- The number of walnut trees planted in the park -/
def trees_planted (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that 33 walnut trees were planted -/
theorem walnut_trees_planted :
  trees_planted 22 55 = 33 := by sorry

end NUMINAMATH_CALUDE_walnut_trees_planted_l3641_364135


namespace NUMINAMATH_CALUDE_dogwood_tree_count_l3641_364186

/-- The number of dogwood trees in the park after planting -/
def total_trees (current : ℕ) (today : ℕ) (tomorrow : ℕ) : ℕ :=
  current + today + tomorrow

/-- Theorem stating the total number of dogwood trees after planting -/
theorem dogwood_tree_count :
  total_trees 39 41 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_dogwood_tree_count_l3641_364186


namespace NUMINAMATH_CALUDE_range_of_a_l3641_364159

theorem range_of_a (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hsum : x + y + z = 1) (ha : ∀ a : ℝ, a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) :
  ∃ S : Set ℝ, S = Set.Ioo 0 (7 / 27) ∪ {7 / 27} ∧ 
  (∀ a : ℝ, a ∈ S ↔ ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 1 ∧
    a / (x * y * z) = 1 / x + 1 / y + 1 / z - 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3641_364159


namespace NUMINAMATH_CALUDE_fraction_equation_implies_sum_of_squares_l3641_364140

theorem fraction_equation_implies_sum_of_squares (A B : ℝ) : 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → (2*x - 3) / (x^2 - x) = A / (x - 1) + B / x) →
  A^2 + B^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_implies_sum_of_squares_l3641_364140


namespace NUMINAMATH_CALUDE_expression_simplification_l3641_364194

theorem expression_simplification (x : ℝ) : 7*x + 15 - 3*x + 2 = 4*x + 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3641_364194


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3641_364154

theorem unique_positive_solution : 
  ∃! x : ℝ, x > 0 ∧ (1/3) * (4*x^2 - 2) = (x^2 - 30*x - 5) * (x^2 + 15*x + 1) ∧ 
  x = 15 + Real.sqrt 8328 / 6 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3641_364154


namespace NUMINAMATH_CALUDE_cubic_system_solution_l3641_364151

theorem cubic_system_solution (x y z a : ℝ) 
  (eq1 : x + y + z = a)
  (eq2 : x^2 + y^2 + z^2 = a^2)
  (eq3 : x^3 + y^3 + z^3 = a^3) :
  (x = a ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = a ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = a) :=
sorry

end NUMINAMATH_CALUDE_cubic_system_solution_l3641_364151


namespace NUMINAMATH_CALUDE_juice_mixture_solution_l3641_364166

/-- Represents the juice mixture problem -/
def JuiceMixture (super_cost mixed_cost acai_cost : ℝ) (acai_amount : ℝ) : Prop :=
  ∃ (mixed_amount : ℝ),
    mixed_amount ≥ 0 ∧
    super_cost * (mixed_amount + acai_amount) =
      mixed_cost * mixed_amount + acai_cost * acai_amount

/-- The solution to the juice mixture problem is approximately 35 litres -/
theorem juice_mixture_solution :
  JuiceMixture 1399.45 262.85 3104.35 23.333333333333336 →
  ∃ (mixed_amount : ℝ),
    mixed_amount ≥ 0 ∧
    abs (mixed_amount - 35) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_juice_mixture_solution_l3641_364166


namespace NUMINAMATH_CALUDE_binomial_coefficient_third_term_x_minus_sqrt3_power10_l3641_364168

/-- The binomial coefficient of the third term in the expansion of (x - √3)^10 is 45 -/
theorem binomial_coefficient_third_term_x_minus_sqrt3_power10 :
  Nat.choose 10 2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_third_term_x_minus_sqrt3_power10_l3641_364168


namespace NUMINAMATH_CALUDE_orange_distribution_difference_l3641_364141

/-- Calculates the difference in oranges per student between initial and final distribution --/
def orange_difference (total_oranges : ℕ) (bad_oranges : ℕ) (num_students : ℕ) : ℕ :=
  (total_oranges / num_students) - ((total_oranges - bad_oranges) / num_students)

theorem orange_distribution_difference :
  orange_difference 108 36 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_difference_l3641_364141


namespace NUMINAMATH_CALUDE_ksyusha_wednesday_travel_time_l3641_364157

/-- The time taken for Ksyusha to travel from home to school on Wednesday -/
theorem ksyusha_wednesday_travel_time :
  -- Given conditions
  ∀ (S : ℝ) (v : ℝ),
  S > 0 → v > 0 →
  -- Tuesday's scenario
  (2 * S / v + S / (2 * v) = 30) →
  -- Wednesday's scenario
  (S / v + 2 * S / (2 * v) = 24) :=
by sorry

end NUMINAMATH_CALUDE_ksyusha_wednesday_travel_time_l3641_364157


namespace NUMINAMATH_CALUDE_trailing_zeroes_89_factorial_plus_97_factorial_l3641_364128

/-- The number of trailing zeroes in a natural number -/
def trailingZeroes (n : ℕ) : ℕ := sorry

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- Theorem: The number of trailing zeroes in 89! + 97! is 20 -/
theorem trailing_zeroes_89_factorial_plus_97_factorial :
  trailingZeroes (factorial 89 + factorial 97) = 20 := by sorry

end NUMINAMATH_CALUDE_trailing_zeroes_89_factorial_plus_97_factorial_l3641_364128


namespace NUMINAMATH_CALUDE_fraction_value_l3641_364133

theorem fraction_value : (12345 : ℕ) / (1 + 2 + 3 + 4 + 5) = 823 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l3641_364133


namespace NUMINAMATH_CALUDE_symmetric_trapezoid_theorem_l3641_364199

/-- A symmetric trapezoid inscribed in a circle -/
structure SymmetricTrapezoid (R : ℝ) where
  x : ℝ
  h_x_range : 0 ≤ x ∧ x ≤ 2*R

/-- The function y for a symmetric trapezoid -/
def y (R : ℝ) (t : SymmetricTrapezoid R) : ℝ :=
  (t.x - R)^2 + 3*R^2

theorem symmetric_trapezoid_theorem (R : ℝ) (h_R : R > 0) :
  ∀ (t : SymmetricTrapezoid R),
    y R t = (t.x - R)^2 + 3*R^2 ∧
    ∀ (a : ℝ), y R t = a^2 → 3*R^2 ≤ a^2 ∧ a^2 ≤ 4*R^2 := by
  sorry

#check symmetric_trapezoid_theorem

end NUMINAMATH_CALUDE_symmetric_trapezoid_theorem_l3641_364199


namespace NUMINAMATH_CALUDE_nine_digit_number_not_prime_l3641_364122

/-- A function that checks if a number is a three-digit prime -/
def isThreeDigitPrime (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ Nat.Prime n

/-- A function that forms a nine-digit number from three three-digit numbers -/
def concatenateThreeNumbers (a b c : ℕ) : ℕ :=
  a * 1000000 + b * 1000 + c

/-- The main theorem -/
theorem nine_digit_number_not_prime 
  (a b c : ℕ) 
  (h1 : isThreeDigitPrime a) 
  (h2 : isThreeDigitPrime b) 
  (h3 : isThreeDigitPrime c) 
  (h4 : ∃ (d : ℤ), b = a + d ∧ c = b + d) : 
  ¬ Nat.Prime (concatenateThreeNumbers a b c) := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_number_not_prime_l3641_364122


namespace NUMINAMATH_CALUDE_remaining_difference_l3641_364114

def recipe_flour : ℕ := 9
def recipe_sugar : ℕ := 6
def sugar_added : ℕ := 4

theorem remaining_difference : 
  recipe_flour - (recipe_sugar - sugar_added) = 7 := by
  sorry

end NUMINAMATH_CALUDE_remaining_difference_l3641_364114


namespace NUMINAMATH_CALUDE_right_triangle_three_colors_l3641_364111

-- Define the color type
inductive Color
| Red
| Green
| Blue

-- Define the point type
structure Point where
  x : Int
  y : Int

-- Define the coloring function
def coloring : Point → Color := sorry

-- Define the property that all three colors are present
def all_colors_present : Prop :=
  ∃ (p1 p2 p3 : Point), coloring p1 ≠ coloring p2 ∧ coloring p2 ≠ coloring p3 ∧ coloring p3 ≠ coloring p1

-- Define a right-angled triangle
def is_right_triangle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0 ∨
  (p1.x - p2.x) * (p3.x - p2.x) + (p1.y - p2.y) * (p3.y - p2.y) = 0 ∨
  (p1.x - p3.x) * (p2.x - p3.x) + (p1.y - p3.y) * (p2.y - p3.y) = 0

-- Theorem statement
theorem right_triangle_three_colors (h : all_colors_present) :
  ∃ (p1 p2 p3 : Point), is_right_triangle p1 p2 p3 ∧
    coloring p1 ≠ coloring p2 ∧ coloring p2 ≠ coloring p3 ∧ coloring p3 ≠ coloring p1 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_three_colors_l3641_364111


namespace NUMINAMATH_CALUDE_college_enrollment_l3641_364102

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_enrollment_percentage : ℚ := 30 / 100

/-- The number of students not enrolled in biology classes -/
def non_biology_students : ℕ := 616

/-- Theorem stating the relationship between the total number of students,
    the percentage enrolled in biology, and the number not enrolled in biology -/
theorem college_enrollment :
  total_students = non_biology_students / (1 - biology_enrollment_percentage) := by
  sorry

end NUMINAMATH_CALUDE_college_enrollment_l3641_364102


namespace NUMINAMATH_CALUDE_probability_of_second_defective_given_first_defective_l3641_364115

-- Define the total number of items
def total_items : ℕ := 20

-- Define the number of good items
def good_items : ℕ := 16

-- Define the number of defective items
def defective_items : ℕ := 4

-- Define the probability of drawing a defective item on the first draw
def prob_first_defective : ℚ := defective_items / total_items

-- Define the probability of drawing a defective item on the second draw given the first was defective
def prob_second_defective_given_first_defective : ℚ := (defective_items - 1) / (total_items - 1)

-- Theorem statement
theorem probability_of_second_defective_given_first_defective :
  prob_second_defective_given_first_defective = 3 / 19 :=
sorry

end NUMINAMATH_CALUDE_probability_of_second_defective_given_first_defective_l3641_364115


namespace NUMINAMATH_CALUDE_pyramid_surface_area_l3641_364103

/-- Represents a square-based pyramid made of unit cubes -/
structure CubePyramid where
  total_cubes : ℕ
  base_side_length : ℕ

/-- Calculates the total surface area of a cube pyramid -/
def total_surface_area (p : CubePyramid) : ℕ :=
  let base_area := p.base_side_length * p.base_side_length
  let top_area := base_area
  let vertical_area := 4 * (p.base_side_length * (p.base_side_length + 1) / 2)
  base_area + top_area + vertical_area

/-- Theorem stating that a pyramid of 30 cubes has a total surface area of 72 square units -/
theorem pyramid_surface_area :
  ∃ (p : CubePyramid), p.total_cubes = 30 ∧ total_surface_area p = 72 :=
by
  sorry


end NUMINAMATH_CALUDE_pyramid_surface_area_l3641_364103


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3641_364150

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 136) : 
  a * b = -6 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l3641_364150


namespace NUMINAMATH_CALUDE_circle_radius_l3641_364158

/-- Given a circle with equation x^2 + y^2 + 2ax + 9 = 0 and center coordinates (5, 0), its radius is 4 -/
theorem circle_radius (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*a*x + 9 = 0 ↔ (x - 5)^2 + y^2 = 16) := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_l3641_364158


namespace NUMINAMATH_CALUDE_inequality_proof_l3641_364143

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3641_364143


namespace NUMINAMATH_CALUDE_speech_competition_orders_l3641_364129

/-- The number of different possible orders in a speech competition --/
def num_orders (original : ℕ) (new : ℕ) : ℕ :=
  (original + 1) * (original + 2)

/-- Theorem: For 5 original participants and 2 new participants,
    the number of different possible orders is 42 --/
theorem speech_competition_orders :
  num_orders 5 2 = 42 := by
  sorry

end NUMINAMATH_CALUDE_speech_competition_orders_l3641_364129


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3641_364108

theorem no_integer_solutions : ¬∃ (m n : ℤ), m^3 + 3*m^2 + 2*m = 8*n^3 + 12*n^2 + 6*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3641_364108


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3641_364152

/-- Given a triangle with one angle of 60° and an adjacent angle of 45°, 
    the smallest angle in the triangle is 45°. -/
theorem smallest_angle_in_triangle (α β γ : ℝ) : 
  α + β + γ = 180 → α = 60 → β = 45 → min α (min β γ) = 45 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l3641_364152


namespace NUMINAMATH_CALUDE_probability_below_25_major_b_l3641_364106

/-- Represents the probability distribution of students in a graduating class -/
structure GraduatingClass where
  male_percentage : Real
  female_percentage : Real
  major_b_percentage : Real
  major_b_below_25_percentage : Real

/-- Theorem stating the probability of a randomly selected student being less than 25 years old and enrolled in Major B -/
theorem probability_below_25_major_b (gc : GraduatingClass) 
  (h1 : gc.male_percentage + gc.female_percentage = 1)
  (h2 : gc.major_b_percentage = 0.3)
  (h3 : gc.major_b_below_25_percentage = 0.6) :
  gc.major_b_percentage * gc.major_b_below_25_percentage = 0.18 := by
  sorry

#check probability_below_25_major_b

end NUMINAMATH_CALUDE_probability_below_25_major_b_l3641_364106


namespace NUMINAMATH_CALUDE_central_angle_measure_l3641_364116

-- Define the sector properties
def arc_length : ℝ := 4
def sector_area : ℝ := 2

-- Theorem statement
theorem central_angle_measure :
  ∀ (r θ : ℝ),
  r > 0 →
  sector_area = 1/2 * r * arc_length →
  arc_length = r * θ →
  θ = 4 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_measure_l3641_364116


namespace NUMINAMATH_CALUDE_betty_boxes_theorem_l3641_364125

/-- The number of boxes Betty uses in an average harvest -/
def num_boxes : ℕ := 20

/-- The capacity of each box in parsnips -/
def box_capacity : ℕ := 20

/-- The fraction of boxes that are full -/
def full_box_fraction : ℚ := 3/4

/-- The fraction of boxes that are half-full -/
def half_full_box_fraction : ℚ := 1/4

/-- The number of parsnips in a half-full box -/
def half_full_box_content : ℕ := box_capacity / 2

/-- The total number of parsnips in an average harvest -/
def total_parsnips : ℕ := 350

/-- Theorem stating that the number of boxes used is correct given the conditions -/
theorem betty_boxes_theorem : 
  (↑num_boxes * full_box_fraction * ↑box_capacity : ℚ) + 
  (↑num_boxes * half_full_box_fraction * ↑half_full_box_content : ℚ) = ↑total_parsnips :=
by sorry

end NUMINAMATH_CALUDE_betty_boxes_theorem_l3641_364125


namespace NUMINAMATH_CALUDE_no_solution_exists_l3641_364107

theorem no_solution_exists : ¬∃ (x y : ℕ+), 
  (x^(y:ℕ) + 3 = y^(x:ℕ)) ∧ (3 * x^(y:ℕ) = y^(x:ℕ) + 8) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3641_364107


namespace NUMINAMATH_CALUDE_spa_nail_polish_inconsistency_l3641_364113

theorem spa_nail_polish_inconsistency :
  ∀ (n : ℕ), n * 20 ≠ 25 :=
by
  sorry

#check spa_nail_polish_inconsistency

end NUMINAMATH_CALUDE_spa_nail_polish_inconsistency_l3641_364113


namespace NUMINAMATH_CALUDE_mary_bought_14_apples_l3641_364187

/-- The number of apples Mary bought initially -/
def apples : ℕ := sorry

/-- The number of oranges Mary bought initially -/
def oranges : ℕ := 9

/-- The number of blueberries Mary bought initially -/
def blueberries : ℕ := 6

/-- The number of fruits Mary has left after eating one of each -/
def fruits_left : ℕ := 26

/-- Theorem stating that Mary bought 14 apples initially -/
theorem mary_bought_14_apples : 
  apples = 14 ∧ 
  apples + oranges + blueberries - 3 = fruits_left :=
sorry

end NUMINAMATH_CALUDE_mary_bought_14_apples_l3641_364187


namespace NUMINAMATH_CALUDE_cubic_identity_l3641_364130

theorem cubic_identity (a b c : ℝ) :
  (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) - a * b * c := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l3641_364130


namespace NUMINAMATH_CALUDE_range_of_m_l3641_364147

-- Define the set A
def A (m : ℝ) : Set ℝ := {x | x^2 + (m+2)*x + 1 = 0}

-- State the theorem
theorem range_of_m (m : ℝ) : (A m ∩ {x : ℝ | x ≠ 0} ≠ ∅) → (-4 < m ∧ m < 0) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3641_364147


namespace NUMINAMATH_CALUDE_wire_length_for_max_area_circular_sector_l3641_364181

/-- The length of wire needed to create a circular sector with maximum area --/
theorem wire_length_for_max_area_circular_sector (r : ℝ) (h : r = 4) :
  2 * π * r + 2 * r = 8 * π + 8 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_for_max_area_circular_sector_l3641_364181


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3641_364192

theorem quadratic_equation_solution : 
  let x₁ : ℝ := 3 + Real.sqrt 7
  let x₂ : ℝ := 3 - Real.sqrt 7
  (x₁^2 - 6*x₁ + 2 = 0) ∧ (x₂^2 - 6*x₂ + 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3641_364192


namespace NUMINAMATH_CALUDE_largest_cube_surface_area_l3641_364121

/-- The surface area of the largest cube that can be cut from a cuboid -/
theorem largest_cube_surface_area (width length height : ℝ) 
  (hw : width = 12) (hl : length = 16) (hh : height = 14) : 
  let side_length := min width (min length height)
  6 * side_length^2 = 864 := by sorry

end NUMINAMATH_CALUDE_largest_cube_surface_area_l3641_364121


namespace NUMINAMATH_CALUDE_no_four_digit_numbers_divisible_by_5_ending_47_l3641_364100

theorem no_four_digit_numbers_divisible_by_5_ending_47 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 →
  n % 100 = 47 →
  ¬(n % 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_four_digit_numbers_divisible_by_5_ending_47_l3641_364100


namespace NUMINAMATH_CALUDE_smallest_n_congruence_fourteen_satisfies_congruence_fourteen_is_smallest_l3641_364174

theorem smallest_n_congruence (n : ℕ) : 
  (n > 0 ∧ 5 * n ≡ 850 [ZMOD 26]) → n ≥ 14 :=
by sorry

theorem fourteen_satisfies_congruence : 
  5 * 14 ≡ 850 [ZMOD 26] :=
by sorry

theorem fourteen_is_smallest : 
  ∀ m : ℕ, m > 0 ∧ m < 14 → ¬(5 * m ≡ 850 [ZMOD 26]) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_fourteen_satisfies_congruence_fourteen_is_smallest_l3641_364174


namespace NUMINAMATH_CALUDE_unique_square_numbers_l3641_364120

theorem unique_square_numbers : ∃! (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  ∃ (m n : ℕ), 
    (100 * a + b = m^2) ∧ 
    (201 * a + b = n^2) ∧ 
    1000 ≤ m^2 ∧ m^2 < 10000 ∧ 
    1000 ≤ n^2 ∧ n^2 < 10000 ∧
    a = 17 ∧ b = 64 := by
  sorry

end NUMINAMATH_CALUDE_unique_square_numbers_l3641_364120


namespace NUMINAMATH_CALUDE_periodic_trig_function_value_l3641_364134

theorem periodic_trig_function_value (m n α₁ α₂ : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hα₁ : α₁ ≠ 0) (hα₂ : α₂ ≠ 0) :
  let f : ℝ → ℝ := λ x => m * Real.sin (π * x + α₁) + n * Real.cos (π * x + α₂)
  f 2011 = 1 → f 2012 = -1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_trig_function_value_l3641_364134


namespace NUMINAMATH_CALUDE_exactly_one_correct_proposition_l3641_364165

theorem exactly_one_correct_proposition : 
  let prop1 := ∀ (a b : ℝ), a < b → a^2 < b^2
  let prop2 := ∀ (a : ℝ), (∀ (x : ℝ), |x+1| + |x-1| ≥ a) ↔ a ≤ 2
  let prop3 := (¬ ∃ (x : ℝ), x^2 - x > 0) ↔ (∀ (x : ℝ), x^2 - x < 0)
  (¬prop1 ∧ prop2 ∧ ¬prop3) := by sorry

end NUMINAMATH_CALUDE_exactly_one_correct_proposition_l3641_364165


namespace NUMINAMATH_CALUDE_equal_sequence_l3641_364149

theorem equal_sequence (x : Fin 2011 → ℝ) (x' : Fin 2011 → ℝ) 
  (h1 : ∀ i : Fin 2011, x i + x (i + 1) = 2 * x' i)
  (h2 : ∃ σ : Equiv.Perm (Fin 2011), ∀ i, x' i = x (σ i)) :
  ∀ i j : Fin 2011, x i = x j :=
by sorry

end NUMINAMATH_CALUDE_equal_sequence_l3641_364149


namespace NUMINAMATH_CALUDE_discount_difference_l3641_364169

def original_price : ℚ := 30
def flat_discount : ℚ := 5
def percent_discount : ℚ := 0.25

def price_flat_then_percent : ℚ := (original_price - flat_discount) * (1 - percent_discount)
def price_percent_then_flat : ℚ := (original_price * (1 - percent_discount)) - flat_discount

theorem discount_difference :
  (price_flat_then_percent - price_percent_then_flat) * 100 = 125 := by
  sorry

end NUMINAMATH_CALUDE_discount_difference_l3641_364169


namespace NUMINAMATH_CALUDE_camera_pics_count_l3641_364142

/-- Represents the number of pictures in Olivia's photo collection. -/
structure PhotoCollection where
  phone_pics : ℕ
  camera_pics : ℕ
  albums : ℕ
  pics_per_album : ℕ

/-- The properties of Olivia's photo collection as described in the problem. -/
def olivias_collection : PhotoCollection where
  phone_pics := 5
  camera_pics := 35  -- This is what we want to prove
  albums := 8
  pics_per_album := 5

/-- Theorem stating that the number of pictures from Olivia's camera is 35. -/
theorem camera_pics_count (p : PhotoCollection) 
  (h1 : p.phone_pics = 5)
  (h2 : p.albums = 8)
  (h3 : p.pics_per_album = 5)
  (h4 : p.phone_pics + p.camera_pics = p.albums * p.pics_per_album) :
  p.camera_pics = 35 := by
  sorry

#check camera_pics_count olivias_collection

end NUMINAMATH_CALUDE_camera_pics_count_l3641_364142


namespace NUMINAMATH_CALUDE_no_integer_root_l3641_364137

theorem no_integer_root (q : ℤ) : ¬ ∃ x : ℤ, x^2 + 7*x - 14*(q^2 + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_root_l3641_364137


namespace NUMINAMATH_CALUDE_equal_expressions_l3641_364105

theorem equal_expressions : 
  (¬ (3^2 / 4 = (3/4)^2)) ∧ 
  (-1^2013 = (-1)^2025) ∧ 
  (¬ (-3^2 = (-3)^2)) ∧ 
  (¬ (-(2^2) / 3 = (-2)^2 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_equal_expressions_l3641_364105


namespace NUMINAMATH_CALUDE_tan_half_theta_l3641_364179

theorem tan_half_theta (θ : Real) (h : 2 * Real.sin θ = 1 + Real.cos θ) :
  (1 + Real.cos θ ≠ 0 → Real.tan (θ / 2) = 1 / 2) ∧
  (1 + Real.cos θ = 0 → ¬∃ (x : Real), Real.tan (θ / 2) = x) :=
by sorry

end NUMINAMATH_CALUDE_tan_half_theta_l3641_364179


namespace NUMINAMATH_CALUDE_product_sum_difference_problem_l3641_364112

theorem product_sum_difference_problem (P Q R S : ℕ) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P > 0 ∧ Q > 0 ∧ R > 0 ∧ S > 0 →
  P * Q = 72 →
  R * S = 72 →
  P + Q = R - S →
  P = 4 ∨ Q = 4 ∨ R = 4 ∨ S = 4 :=
by sorry

#check product_sum_difference_problem

end NUMINAMATH_CALUDE_product_sum_difference_problem_l3641_364112


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l3641_364139

theorem shirt_cost_problem (total_shirts : ℕ) (known_shirt_count : ℕ) (known_shirt_cost : ℕ) (total_cost : ℕ) :
  total_shirts = 5 →
  known_shirt_count = 3 →
  known_shirt_cost = 15 →
  total_cost = 85 →
  (total_cost - known_shirt_count * known_shirt_cost) / (total_shirts - known_shirt_count) = 20 := by
sorry

end NUMINAMATH_CALUDE_shirt_cost_problem_l3641_364139


namespace NUMINAMATH_CALUDE_limit_cubic_fraction_l3641_364184

theorem limit_cubic_fraction :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |((x^3 - 1) / (x - 1)) - 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_cubic_fraction_l3641_364184


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3641_364185

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_a2 : a 2 = 3) 
  (h_a6 : a 6 = 48) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3641_364185


namespace NUMINAMATH_CALUDE_count_integers_with_conditions_l3641_364182

theorem count_integers_with_conditions : 
  ∃ (S : Finset ℤ), 
    (∀ n ∈ S, 150 < n ∧ n < 300 ∧ n % 7 = n % 9) ∧ 
    (∀ n, 150 < n → n < 300 → n % 7 = n % 9 → n ∈ S) ∧ 
    Finset.card S = 14 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_conditions_l3641_364182


namespace NUMINAMATH_CALUDE_quadratic_minimum_l3641_364101

theorem quadratic_minimum : 
  (∀ x : ℝ, x^2 + 6*x ≥ -9) ∧ (∃ x : ℝ, x^2 + 6*x = -9) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l3641_364101


namespace NUMINAMATH_CALUDE_divisibility_by_23_and_29_l3641_364183

theorem divisibility_by_23_and_29 (a b c : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) (hc : c ≤ 9) :
  ∃ (k m : ℕ), 200100 * a + 20010 * b + 2001 * c = 23 * k ∧ 200100 * a + 20010 * b + 2001 * c = 29 * m := by
  sorry

#check divisibility_by_23_and_29

end NUMINAMATH_CALUDE_divisibility_by_23_and_29_l3641_364183


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l3641_364177

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| - |x + 2|

-- Part 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ -x} = {x : ℝ | x ≤ -3 ∨ -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x ≤ a^2 + 1) → a ≤ -Real.sqrt 2 ∨ a ≥ Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_range_of_a_l3641_364177
