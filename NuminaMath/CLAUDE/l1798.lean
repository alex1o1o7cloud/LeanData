import Mathlib

namespace remainder_1234567891_div_98_l1798_179860

theorem remainder_1234567891_div_98 : 1234567891 % 98 = 23 := by
  sorry

end remainder_1234567891_div_98_l1798_179860


namespace connie_marbles_theorem_l1798_179838

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 593

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_marbles_theorem : initial_marbles = 776 := by sorry

end connie_marbles_theorem_l1798_179838


namespace james_waiting_time_l1798_179885

/-- The number of days it took for James' pain to subside -/
def pain_subsided_days : ℕ := 3

/-- The factor by which the full healing time is longer than the pain subsidence time -/
def healing_factor : ℕ := 5

/-- The number of days James waits after healing before working out -/
def wait_before_workout_days : ℕ := 3

/-- The total number of days until James can lift heavy again -/
def total_days_until_heavy_lifting : ℕ := 39

/-- The number of days in a week -/
def days_per_week : ℕ := 7

theorem james_waiting_time :
  (total_days_until_heavy_lifting - (pain_subsided_days * healing_factor + wait_before_workout_days)) / days_per_week = 3 := by
  sorry

end james_waiting_time_l1798_179885


namespace beverage_distribution_l1798_179839

/-- Represents the number of cans of beverage -/
def total_cans : ℚ := 5

/-- Represents the number of children -/
def num_children : ℚ := 8

/-- Represents each child's share of the total beverage -/
def share_of_total : ℚ := 1 / num_children

/-- Represents each child's share in terms of cans -/
def share_in_cans : ℚ := total_cans / num_children

theorem beverage_distribution :
  share_of_total = 1 / 8 ∧ share_in_cans = 5 / 8 := by sorry

end beverage_distribution_l1798_179839


namespace inscribed_circle_radius_l1798_179886

theorem inscribed_circle_radius (DE DF EF : ℝ) (h1 : DE = 7) (h2 : DF = 8) (h3 : EF = 9) :
  let s := (DE + DF + EF) / 2
  let A := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  A / s = Real.sqrt 5 := by sorry

end inscribed_circle_radius_l1798_179886


namespace third_term_is_five_l1798_179801

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℝ  -- second term
  d : ℝ  -- common difference

/-- The sum of the second and fourth terms is 10 -/
def sum_second_fourth (seq : ArithmeticSequence) : Prop :=
  seq.a + (seq.a + 2 * seq.d) = 10

/-- The third term of the sequence -/
def third_term (seq : ArithmeticSequence) : ℝ :=
  seq.a + seq.d

/-- Theorem: If the sum of the second and fourth terms of an arithmetic sequence is 10,
    then the third term is 5 -/
theorem third_term_is_five (seq : ArithmeticSequence) 
    (h : sum_second_fourth seq) : third_term seq = 5 := by
  sorry

end third_term_is_five_l1798_179801


namespace helmet_cost_l1798_179869

theorem helmet_cost (total_cost bicycle_cost helmet_cost : ℝ) : 
  total_cost = 240 →
  bicycle_cost = 5 * helmet_cost →
  total_cost = bicycle_cost + helmet_cost →
  helmet_cost = 40 := by
sorry

end helmet_cost_l1798_179869


namespace smallest_with_ten_divisors_l1798_179877

/-- A function that returns the number of positive integer divisors of a given natural number. -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 10 positive integer divisors. -/
def has_ten_divisors (n : ℕ) : Prop := num_divisors n = 10

/-- Theorem stating that 48 is the smallest positive integer with exactly 10 positive integer divisors. -/
theorem smallest_with_ten_divisors : 
  has_ten_divisors 48 ∧ ∀ m : ℕ, m < 48 → ¬(has_ten_divisors m) :=
sorry

end smallest_with_ten_divisors_l1798_179877


namespace odd_function_has_zero_point_l1798_179818

theorem odd_function_has_zero_point (f : ℝ → ℝ) (h : ∀ x, f (-x) = -f x) :
  ∃ x, f x = 0 := by sorry

end odd_function_has_zero_point_l1798_179818


namespace removed_number_for_mean_l1798_179857

theorem removed_number_for_mean (n : ℕ) (h : n ≥ 9) :
  ∃ x : ℕ, x ≤ n ∧ 
    (((n * (n + 1)) / 2 - x) / (n - 1) : ℚ) = 19/4 →
    x = 7 :=
  sorry

end removed_number_for_mean_l1798_179857


namespace exists_f_1984_eq_A_l1798_179810

-- Define the function property
def satisfies_property (f : ℤ → ℝ) : Prop :=
  ∀ x y : ℤ, f (x - y^2) = f x + (y^2 - 2*x) * f y

-- State the theorem
theorem exists_f_1984_eq_A (A : ℝ) :
  ∃ f : ℤ → ℝ, satisfies_property f ∧ f 1984 = A :=
sorry

end exists_f_1984_eq_A_l1798_179810


namespace urn_ball_removal_l1798_179804

theorem urn_ball_removal (total : ℕ) (red_percent : ℚ) (blue_removed : ℕ) (new_red_percent : ℚ) : 
  total = 150 →
  red_percent = 2/5 →
  blue_removed = 75 →
  new_red_percent = 4/5 →
  (red_percent * total : ℚ) / (total - blue_removed : ℚ) = new_red_percent :=
by sorry

end urn_ball_removal_l1798_179804


namespace student_ticket_cost_l1798_179890

/-- Proves that the cost of each student ticket is 2 dollars given the conditions of the ticket sales -/
theorem student_ticket_cost (total_tickets : ℕ) (total_revenue : ℕ) 
  (nonstudent_price : ℕ) (student_tickets : ℕ) :
  total_tickets = 821 →
  total_revenue = 1933 →
  nonstudent_price = 3 →
  student_tickets = 530 →
  ∃ (student_price : ℕ),
    student_price * student_tickets + 
    nonstudent_price * (total_tickets - student_tickets) = total_revenue ∧
    student_price = 2 :=
by
  sorry

end student_ticket_cost_l1798_179890


namespace equation_solution_range_l1798_179883

theorem equation_solution_range (a : ℝ) (m : ℝ) :
  a > 0 ∧ a ≠ 1 →
  (∃ x : ℝ, a^(2*x) + (1 + 1/m)*a^x + 1 = 0) ↔
  -1/3 ≤ m ∧ m < 0 :=
by sorry

end equation_solution_range_l1798_179883


namespace increasing_magnitude_l1798_179875

theorem increasing_magnitude (x : ℝ) (h : 0.85 < x ∧ x < 1.1) :
  x ≤ x + Real.sin x ∧ x + Real.sin x < x^(x^x) := by
  sorry

end increasing_magnitude_l1798_179875


namespace volume_Q_3_l1798_179827

/-- Recursive definition of polyhedron volumes -/
def Q : ℕ → ℚ
  | 0 => 8
  | (n + 1) => Q n + 4 * (1 / 27)^n

/-- The volume of Q₃ is 5972/729 -/
theorem volume_Q_3 : Q 3 = 5972 / 729 := by sorry

end volume_Q_3_l1798_179827


namespace mikes_purchase_cost_l1798_179813

/-- The total cost of a camera and lens purchase -/
def total_cost (old_camera_cost lens_price lens_discount : ℚ) : ℚ :=
  let new_camera_cost := old_camera_cost * (1 + 0.3)
  let discounted_lens_price := lens_price - lens_discount
  new_camera_cost + discounted_lens_price

/-- Theorem stating the total cost of Mike's camera and lens purchase -/
theorem mikes_purchase_cost :
  total_cost 4000 400 200 = 5400 := by
  sorry

end mikes_purchase_cost_l1798_179813


namespace imaginary_unit_cube_l1798_179843

theorem imaginary_unit_cube (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end imaginary_unit_cube_l1798_179843


namespace unique_solution_for_equation_l1798_179844

theorem unique_solution_for_equation (m n p : ℕ+) (h_prime : Nat.Prime p) :
  2^(m : ℕ) * p^2 + 1 = n^5 ↔ m = 1 ∧ n = 3 ∧ p = 11 := by
  sorry

end unique_solution_for_equation_l1798_179844


namespace number_equation_l1798_179814

theorem number_equation (x : ℝ) : 3 * x = (26 - x) + 10 ↔ x = 9 := by
  sorry

end number_equation_l1798_179814


namespace complex_on_real_axis_l1798_179816

theorem complex_on_real_axis (a : ℝ) : 
  let z : ℂ := (a - Complex.I) * (1 + Complex.I)
  (z.im = 0) → a = 1 := by
  sorry

end complex_on_real_axis_l1798_179816


namespace egg_count_problem_l1798_179806

/-- Calculates the final number of eggs for a family given initial count and various changes --/
def final_egg_count (initial : ℕ) (mother_used : ℕ) (father_used : ℕ) 
  (chicken1_laid : ℕ) (chicken2_laid : ℕ) (chicken3_laid : ℕ) (child_took : ℕ) : ℕ :=
  initial - mother_used - father_used + chicken1_laid + chicken2_laid + chicken3_laid - child_took

/-- Theorem stating that given the specific values in the problem, the final egg count is 19 --/
theorem egg_count_problem : 
  final_egg_count 20 5 3 4 3 2 2 = 19 := by sorry

end egg_count_problem_l1798_179806


namespace smallest_number_of_eggs_l1798_179809

theorem smallest_number_of_eggs (total_containers : ℕ) (deficient_containers : ℕ) 
  (container_capacity : ℕ) (min_total_eggs : ℕ) :
  total_containers > 10 ∧ 
  deficient_containers = 3 ∧ 
  container_capacity = 15 ∧ 
  min_total_eggs = 150 →
  (total_containers * container_capacity - deficient_containers = 
    (total_containers - deficient_containers) * container_capacity + 
    deficient_containers * (container_capacity - 1)) ∧
  (total_containers * container_capacity - deficient_containers > min_total_eggs) ∧
  ∀ n : ℕ, n < total_containers → 
    n * container_capacity - deficient_containers ≤ min_total_eggs :=
by sorry

end smallest_number_of_eggs_l1798_179809


namespace yellow_shirt_pairs_l1798_179864

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 75 →
  yellow_students = 105 →
  total_students = blue_students + yellow_students →
  total_pairs = 90 →
  blue_blue_pairs = 30 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 45 ∧ 
    yellow_yellow_pairs = (yellow_students - (total_students - 2 * blue_blue_pairs)) / 2 :=
by sorry

end yellow_shirt_pairs_l1798_179864


namespace angle_4_value_l1798_179884

theorem angle_4_value (angle1 angle2 angle3 angle4 : ℝ) : 
  angle1 + angle2 = 180 →
  angle3 = 2 * angle4 →
  angle1 = 50 →
  angle3 + angle4 = 130 →
  angle4 = 130 / 3 := by
sorry

end angle_4_value_l1798_179884


namespace graphing_calculator_count_l1798_179837

theorem graphing_calculator_count :
  ∀ (S G : ℕ),
    S + G = 45 →
    10 * S + 57 * G = 1625 →
    G = 25 :=
by
  sorry

end graphing_calculator_count_l1798_179837


namespace inequality_proof_l1798_179822

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) :
  x^2 + y^2 + x^2*y^2 ≤ 3 := by
  sorry

end inequality_proof_l1798_179822


namespace snow_removal_volume_l1798_179880

/-- The volume of snow to be removed from a rectangular driveway -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Proof that the volume of snow to be removed is 67.5 cubic feet -/
theorem snow_removal_volume :
  let length : ℝ := 30
  let width : ℝ := 3
  let depth : ℝ := 0.75
  snow_volume length width depth = 67.5 := by
sorry

end snow_removal_volume_l1798_179880


namespace wire_cutting_theorem_l1798_179821

def wire1 : ℕ := 1008
def wire2 : ℕ := 1260
def wire3 : ℕ := 882
def wire4 : ℕ := 1134

def segment_length : ℕ := 126
def total_segments : ℕ := 34

theorem wire_cutting_theorem :
  (∃ (n : ℕ), n > 0 ∧ 
    wire1 % n = 0 ∧ 
    wire2 % n = 0 ∧ 
    wire3 % n = 0 ∧ 
    wire4 % n = 0 ∧
    ∀ (m : ℕ), m > n → 
      (wire1 % m ≠ 0 ∨ 
       wire2 % m ≠ 0 ∨ 
       wire3 % m ≠ 0 ∨ 
       wire4 % m ≠ 0)) ∧
  segment_length = 126 ∧
  total_segments = 34 ∧
  wire1 / segment_length + 
  wire2 / segment_length + 
  wire3 / segment_length + 
  wire4 / segment_length = total_segments :=
by sorry

end wire_cutting_theorem_l1798_179821


namespace reciprocal_of_repeating_third_l1798_179854

/-- The repeating decimal 0.333... --/
def repeating_third : ℚ := 1/3

/-- The reciprocal of the common fraction form of 0.333... is 3 --/
theorem reciprocal_of_repeating_third : (repeating_third⁻¹ : ℚ) = 3 := by
  sorry

end reciprocal_of_repeating_third_l1798_179854


namespace car_rental_hours_per_day_l1798_179840

/-- Proves that given the rental conditions, the number of hours rented per day is 8 --/
theorem car_rental_hours_per_day 
  (hourly_rate : ℝ)
  (days_per_week : ℕ)
  (weekly_income : ℝ)
  (h : hourly_rate = 20)
  (d : days_per_week = 4)
  (w : weekly_income = 640) :
  (weekly_income / (hourly_rate * days_per_week : ℝ)) = 8 := by
  sorry

end car_rental_hours_per_day_l1798_179840


namespace initial_gasoline_percentage_l1798_179874

/-- Proves that the initial gasoline percentage is 95% given the problem conditions -/
theorem initial_gasoline_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (optimal_ethanol_percentage : ℝ)
  (added_ethanol : ℝ)
  (h1 : initial_volume = 36)
  (h2 : initial_ethanol_percentage = 0.05)
  (h3 : optimal_ethanol_percentage = 0.10)
  (h4 : added_ethanol = 2)
  (h5 : optimal_ethanol_percentage * (initial_volume + added_ethanol) =
        initial_ethanol_percentage * initial_volume + added_ethanol) :
  initial_volume * (1 - initial_ethanol_percentage) / initial_volume = 0.95 := by
  sorry

#check initial_gasoline_percentage

end initial_gasoline_percentage_l1798_179874


namespace backyard_sod_coverage_l1798_179895

/-- Represents the dimensions of a rectangular section -/
structure Section where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular section -/
def sectionArea (s : Section) : ℕ := s.length * s.width

/-- Represents the dimensions of a sod square -/
structure SodSquare where
  side : ℕ

/-- Calculates the area of a sod square -/
def sodSquareArea (s : SodSquare) : ℕ := s.side * s.side

/-- Calculates the number of sod squares needed to cover a given area -/
def sodSquaresNeeded (totalArea : ℕ) (sodSquare : SodSquare) : ℕ :=
  totalArea / sodSquareArea sodSquare

theorem backyard_sod_coverage (section1 : Section) (section2 : Section) (sodSquare : SodSquare) :
  section1.length = 30 →
  section1.width = 40 →
  section2.length = 60 →
  section2.width = 80 →
  sodSquare.side = 2 →
  sodSquaresNeeded (sectionArea section1 + sectionArea section2) sodSquare = 1500 := by
  sorry

end backyard_sod_coverage_l1798_179895


namespace f_maximum_l1798_179891

/-- The quadratic function f(x) = -3x^2 + 9x + 24 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 24

/-- The point where f attains its maximum -/
def x_max : ℝ := 1.5

theorem f_maximum :
  ∀ x : ℝ, f x ≤ f x_max := by sorry

end f_maximum_l1798_179891


namespace trigonometric_identity_l1798_179841

theorem trigonometric_identity :
  Real.sin (20 * π / 180) * Real.sin (80 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end trigonometric_identity_l1798_179841


namespace number_puzzle_l1798_179826

theorem number_puzzle : ∃ x : ℝ, ((x - 50) / 4) * 3 + 28 = 73 ∧ x = 110 := by
  sorry

end number_puzzle_l1798_179826


namespace max_value_under_constraint_l1798_179896

/-- The objective function to be maximized -/
def f (x y : ℝ) : ℝ := 8 * x^2 + 9 * x * y + 18 * y^2 + 2 * x + 3 * y

/-- The constraint function -/
def g (x y : ℝ) : ℝ := 4 * x^2 + 9 * y^2 - 8

/-- Theorem stating that the maximum value of f subject to the constraint g = 0 is 26 -/
theorem max_value_under_constraint : 
  ∃ (x y : ℝ), g x y = 0 ∧ f x y = 26 ∧ ∀ (x' y' : ℝ), g x' y' = 0 → f x' y' ≤ 26 := by
  sorry

end max_value_under_constraint_l1798_179896


namespace quadratic_coeff_unequal_l1798_179847

/-- Given a quadratic equation 3x^2 + 7x + 2k = 0 with zero discriminant,
    prove that the coefficients 3, 7, and k are unequal -/
theorem quadratic_coeff_unequal (k : ℝ) :
  (7^2 - 4*3*(2*k) = 0) →
  (3 ≠ 7 ∧ 3 ≠ k ∧ 7 ≠ k) :=
by sorry

end quadratic_coeff_unequal_l1798_179847


namespace hidden_dots_count_l1798_179802

/-- Represents a standard six-sided die -/
def standardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The sum of all numbers on a standard die -/
def standardDieSum : ℕ := Finset.sum standardDie id

/-- The number of dice in the stack -/
def numDice : ℕ := 4

/-- The visible numbers on the dice -/
def visibleNumbers : Finset ℕ := {1, 2, 2, 3, 3, 4, 5, 6}

/-- The sum of visible numbers -/
def visibleSum : ℕ := Finset.sum visibleNumbers id

/-- The total number of dots on all dice -/
def totalDots : ℕ := numDice * standardDieSum

theorem hidden_dots_count : totalDots - visibleSum = 58 := by sorry

end hidden_dots_count_l1798_179802


namespace function_F_theorem_l1798_179887

theorem function_F_theorem (F : ℝ → ℝ) 
  (h_diff : Differentiable ℝ F) 
  (h_init : F 0 = -1)
  (h_deriv : ∀ x, deriv F x = Real.sin (Real.sin (Real.sin (Real.sin x))) * 
    Real.cos (Real.sin (Real.sin x)) * Real.cos (Real.sin x) * Real.cos x) :
  ∀ x, F x = -Real.cos (Real.sin (Real.sin (Real.sin x))) := by
sorry

end function_F_theorem_l1798_179887


namespace arithmetic_calculation_l1798_179829

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end arithmetic_calculation_l1798_179829


namespace trapezoid_lines_parallel_or_concurrent_l1798_179859

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the Euclidean plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Trapezoid ABCD with diagonals intersecting at E -/
structure Trapezoid :=
  (A B C D E : Point)
  (AB_parallel_CD : Line)
  (AC_diagonal : Line)
  (BD_diagonal : Line)
  (E_on_AC_and_BD : Prop)

/-- P is the foot of altitude from A to BC -/
def altitude_foot_P (trap : Trapezoid) : Point :=
  sorry

/-- Q is the foot of altitude from B to AD -/
def altitude_foot_Q (trap : Trapezoid) : Point :=
  sorry

/-- F is the intersection of circumcircles of CEQ and DEP -/
def point_F (trap : Trapezoid) (P Q : Point) : Point :=
  sorry

/-- Line through two points -/
def line_through (P Q : Point) : Line :=
  sorry

/-- Check if three lines are parallel or concurrent -/
def parallel_or_concurrent (l₁ l₂ l₃ : Line) : Prop :=
  sorry

theorem trapezoid_lines_parallel_or_concurrent (trap : Trapezoid) :
  let P := altitude_foot_P trap
  let Q := altitude_foot_Q trap
  let F := point_F trap P Q
  let AP := line_through trap.A P
  let BQ := line_through trap.B Q
  let EF := line_through trap.E F
  parallel_or_concurrent AP BQ EF :=
sorry

end trapezoid_lines_parallel_or_concurrent_l1798_179859


namespace not_square_difference_l1798_179851

-- Define the square difference formula
def square_difference (a b : ℝ → ℝ) : ℝ → ℝ := λ x => (a x)^2 - (b x)^2

-- Define the expression we want to prove doesn't fit the square difference formula
def expression : ℝ → ℝ := λ x => (x + 1) * (1 + x)

-- Theorem statement
theorem not_square_difference :
  ¬ ∃ (a b : ℝ → ℝ), ∀ x, expression x = square_difference a b x :=
sorry

end not_square_difference_l1798_179851


namespace incorrect_value_calculation_l1798_179879

theorem incorrect_value_calculation (n : ℕ) (initial_mean correct_mean correct_value : ℝ) 
  (h1 : n = 30)
  (h2 : initial_mean = 150)
  (h3 : correct_mean = 151)
  (h4 : correct_value = 165) :
  let initial_sum := n * initial_mean
  let correct_sum := n * correct_mean
  let difference := correct_sum - initial_sum
  initial_sum + correct_value - difference = n * correct_mean := by sorry

end incorrect_value_calculation_l1798_179879


namespace janette_beef_jerky_dinner_l1798_179862

/-- Calculates the number of beef jerky pieces eaten for dinner each day during a camping trip. -/
def beef_jerky_for_dinner (days : ℕ) (total_pieces : ℕ) (breakfast_pieces : ℕ) (lunch_pieces : ℕ) (pieces_after_sharing : ℕ) : ℕ :=
  let pieces_before_sharing := 2 * pieces_after_sharing
  let pieces_eaten := total_pieces - pieces_before_sharing
  let pieces_for_breakfast_and_lunch := (breakfast_pieces + lunch_pieces) * days
  (pieces_eaten - pieces_for_breakfast_and_lunch) / days

/-- Theorem stating that Janette ate 2 pieces of beef jerky for dinner each day during her camping trip. -/
theorem janette_beef_jerky_dinner :
  beef_jerky_for_dinner 5 40 1 1 10 = 2 := by
  sorry

end janette_beef_jerky_dinner_l1798_179862


namespace intersection_A_B_union_A_B_l1798_179824

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | 2*x - 4 ≥ x - 2}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -1} := by sorry

end intersection_A_B_union_A_B_l1798_179824


namespace jane_work_days_jane_solo_days_l1798_179811

theorem jane_work_days (john_days : ℝ) (total_days : ℝ) (jane_stop_days : ℝ) : ℝ :=
  let john_rate := 1 / john_days
  let total_work := 1
  let jane_work_days := total_days - jane_stop_days
  let john_solo_work := john_rate * jane_stop_days
  let combined_work := total_work - john_solo_work
  combined_work / (john_rate + 1 / (total_days - jane_stop_days)) / jane_work_days

theorem jane_solo_days 
  (john_days : ℝ) 
  (total_days : ℝ) 
  (jane_stop_days : ℝ) 
  (h1 : john_days = 20)
  (h2 : total_days = 10)
  (h3 : jane_stop_days = 4)
  : jane_work_days john_days total_days jane_stop_days = 12 := by
  sorry

end jane_work_days_jane_solo_days_l1798_179811


namespace no_four_integers_with_odd_sum_and_product_l1798_179845

theorem no_four_integers_with_odd_sum_and_product : ¬∃ (a b c d : ℤ), 
  Odd (a + b + c + d) ∧ Odd (a * b * c * d) := by
  sorry

end no_four_integers_with_odd_sum_and_product_l1798_179845


namespace circle_distance_l1798_179856

theorem circle_distance (R r : ℝ) : 
  R^2 - 4*R + 2 = 0 → 
  r^2 - 4*r + 2 = 0 → 
  R ≠ r → 
  (∃ d : ℝ, d = abs (R - r) ∧ (d = 4 ∨ d = 2)) :=
by sorry

end circle_distance_l1798_179856


namespace percent_relation_l1798_179846

theorem percent_relation (a b c : ℝ) (h1 : c = 0.30 * a) (h2 : b = 1.20 * a) : 
  c = 0.25 * b := by
sorry

end percent_relation_l1798_179846


namespace foreign_language_teachers_l1798_179870

/-- The number of teachers who do not teach English, Japanese, or French -/
theorem foreign_language_teachers (total : ℕ) (english : ℕ) (japanese : ℕ) (french : ℕ)
  (eng_jap : ℕ) (eng_fre : ℕ) (jap_fre : ℕ) (all_three : ℕ) :
  total = 120 →
  english = 50 →
  japanese = 45 →
  french = 40 →
  eng_jap = 15 →
  eng_fre = 10 →
  jap_fre = 8 →
  all_three = 4 →
  total - (english + japanese + french - eng_jap - eng_fre - jap_fre + all_three) = 14 :=
by sorry

end foreign_language_teachers_l1798_179870


namespace fraction_relation_l1798_179889

theorem fraction_relation (a b c d : ℚ) 
  (h1 : a / b = 8)
  (h2 : c / b = 5)
  (h3 : c / d = 1 / 3) :
  d / a = 15 / 8 := by
sorry

end fraction_relation_l1798_179889


namespace debose_family_mean_age_l1798_179800

theorem debose_family_mean_age : 
  let ages : List ℕ := [8, 8, 16, 18]
  let num_children := ages.length
  let sum_ages := ages.sum
  (sum_ages : ℚ) / num_children = 25/2 := by sorry

end debose_family_mean_age_l1798_179800


namespace magic_square_solution_l1798_179881

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ
  magic_sum : ℕ
  row_sum : a + b + c = magic_sum ∧ d + e + f = magic_sum ∧ g + h + i = magic_sum
  col_sum : a + d + g = magic_sum ∧ b + e + h = magic_sum ∧ c + f + i = magic_sum
  diag_sum : a + e + i = magic_sum ∧ c + e + g = magic_sum

/-- Theorem: In a 3x3 magic square with top row entries x, 23, 102 and middle-left entry 5, x must equal 208 -/
theorem magic_square_solution (ms : MagicSquare) (h1 : ms.b = 23) (h2 : ms.c = 102) (h3 : ms.d = 5) : ms.a = 208 := by
  sorry


end magic_square_solution_l1798_179881


namespace integer_count_between_negatives_l1798_179861

theorem integer_count_between_negatives (a : ℚ) : 
  (a > 0) → 
  (∃ n : ℕ, n = (⌊a⌋ - ⌈-a⌉ - 1) ∧ n = 2007) → 
  (1003 < a ∧ a ≤ 1004) :=
by
  sorry

end integer_count_between_negatives_l1798_179861


namespace divide_negative_four_by_two_l1798_179873

theorem divide_negative_four_by_two : -4 / 2 = -2 := by
  sorry

end divide_negative_four_by_two_l1798_179873


namespace unique_four_digit_number_l1798_179899

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_less_than_10 : a < 10
  b_less_than_10 : b < 10
  c_less_than_10 : c < 10
  d_less_than_10 : d < 10

/-- The conditions given in the problem -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  n.a + n.b + n.c + n.d = 26 ∧
  (n.b * n.d) / 10 = n.a + n.c ∧
  ∃ k : Nat, n.b * n.d - n.c * n.c = 2 * k

/-- The theorem to prove -/
theorem unique_four_digit_number :
  ∃! n : FourDigitNumber, satisfiesConditions n ∧ 
    n.a = 1 ∧ n.b = 9 ∧ n.c = 7 ∧ n.d = 9 :=
by sorry

end unique_four_digit_number_l1798_179899


namespace roberta_initial_records_l1798_179834

/-- The number of records Roberta initially had -/
def initial_records : ℕ := sorry

/-- The number of records Roberta received as gifts -/
def gifted_records : ℕ := 12

/-- The number of records Roberta bought at a garage sale -/
def bought_records : ℕ := 30

/-- The number of days it takes Roberta to listen to one record -/
def days_per_record : ℕ := 2

/-- The total number of days it will take Roberta to listen to her entire collection -/
def total_listening_days : ℕ := 100

theorem roberta_initial_records :
  initial_records = 8 :=
by sorry

end roberta_initial_records_l1798_179834


namespace prob_diamond_or_club_half_l1798_179815

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (cards_per_suit : ℕ)
  (diamond_club_cards : ℕ)

/-- Probability of drawing a diamond or club from the top of a shuffled deck -/
def prob_diamond_or_club (d : Deck) : ℚ :=
  d.diamond_club_cards / d.total_cards

/-- Theorem stating the probability of drawing a diamond or club is 1/2 -/
theorem prob_diamond_or_club_half (d : Deck) 
  (h1 : d.total_cards = 52) 
  (h2 : d.cards_per_suit = 13) 
  (h3 : d.diamond_club_cards = 2 * d.cards_per_suit) : 
  prob_diamond_or_club d = 1/2 := by
  sorry

end prob_diamond_or_club_half_l1798_179815


namespace line_l_passes_through_A_and_B_l1798_179898

/-- The line l passes through points A(-1, 0) and B(1, 4) -/
def line_l (x y : ℝ) : Prop := y = 2 * x + 2

/-- Point A has coordinates (-1, 0) -/
def point_A : ℝ × ℝ := (-1, 0)

/-- Point B has coordinates (1, 4) -/
def point_B : ℝ × ℝ := (1, 4)

/-- The line l passes through points A and B -/
theorem line_l_passes_through_A_and_B : 
  line_l point_A.1 point_A.2 ∧ line_l point_B.1 point_B.2 := by sorry

end line_l_passes_through_A_and_B_l1798_179898


namespace teacher_zhang_age_in_five_years_l1798_179807

/-- Given Xiao Li's age and the relationship between Xiao Li's and Teacher Zhang's ages,
    prove Teacher Zhang's age after 5 years. -/
theorem teacher_zhang_age_in_five_years (a : ℕ) : 
  (3 * a - 2) + 5 = 3 * a + 3 :=
by sorry

end teacher_zhang_age_in_five_years_l1798_179807


namespace fermat_little_theorem_l1798_179853

theorem fermat_little_theorem (N p : ℕ) (hp : Prime p) (hN : ¬ p ∣ N) :
  p ∣ (N^(p - 1) - 1) := by
  sorry

end fermat_little_theorem_l1798_179853


namespace geometric_mean_of_square_sides_l1798_179855

theorem geometric_mean_of_square_sides (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 64) (h₂ : a₂ = 81) (h₃ : a₃ = 144) :
  (((a₁.sqrt * a₂.sqrt * a₃.sqrt) ^ (1/3 : ℝ)) : ℝ) = 6 * (4 ^ (1/3 : ℝ)) := by
  sorry

end geometric_mean_of_square_sides_l1798_179855


namespace cyclist_heartbeats_l1798_179897

/-- The number of heartbeats during a cycling race -/
def heartbeats_during_race (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Theorem: The cyclist's heart beats 57600 times during the race -/
theorem cyclist_heartbeats :
  heartbeats_during_race 120 4 120 = 57600 := by
  sorry

#eval heartbeats_during_race 120 4 120

end cyclist_heartbeats_l1798_179897


namespace stating_nth_smallest_d₀_is_correct_l1798_179894

/-- 
Given a non-negative integer d₀ and a positive integer v,
this function returns true if v² = 8d₀, false otherwise.
-/
def is_valid_pair (d₀ v : ℕ) : Prop :=
  v^2 = 8 * d₀

/-- 
This function returns the nth smallest non-negative integer d₀
such that there exists a positive integer v where v² = 8d₀.
-/
def nth_smallest_d₀ (n : ℕ) : ℕ :=
  4^(n-1)

/-- 
Theorem stating that nth_smallest_d₀ correctly computes
the nth smallest d₀ satisfying the required property.
-/
theorem nth_smallest_d₀_is_correct (n : ℕ) :
  n > 0 →
  (∃ v : ℕ, is_valid_pair (nth_smallest_d₀ n) v) ∧
  (∀ d : ℕ, d < nth_smallest_d₀ n →
    (∃ v : ℕ, is_valid_pair d v) →
    (∃ k < n, d = nth_smallest_d₀ k)) :=
by sorry

end stating_nth_smallest_d₀_is_correct_l1798_179894


namespace power_multiplication_l1798_179831

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end power_multiplication_l1798_179831


namespace expression_simplification_l1798_179825

theorem expression_simplification (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  ((a^2 + 4*a + 4) / (a^2 - 4) - (a + 3) / (a - 2)) / ((a + 2) / (a - 2)) = -1 / (a + 2) :=
by sorry

end expression_simplification_l1798_179825


namespace valid_k_values_l1798_179833

/-- A function f: ℤ → ℤ satisfies the given property for a positive integer k -/
def satisfies_property (f : ℤ → ℤ) (k : ℕ+) : Prop :=
  ∀ (a b c : ℤ), a + b + c = 0 →
    f a + f b + f c = (f (a - b) + f (b - c) + f (c - a)) / k

/-- A function f: ℤ → ℤ is nonlinear -/
def is_nonlinear (f : ℤ → ℤ) : Prop :=
  ∃ (a b x y : ℤ), f (a + x) + f (b + y) ≠ f a + f b + f x + f y

theorem valid_k_values :
  {k : ℕ+ | ∃ (f : ℤ → ℤ), satisfies_property f k ∧ is_nonlinear f} = {1, 3, 9} := by
  sorry

end valid_k_values_l1798_179833


namespace min_rice_purchase_exact_min_rice_purchase_l1798_179842

/-- The minimum amount of rice Maria could purchase, given the constraints on oats and rice. -/
theorem min_rice_purchase (o r : ℝ) 
  (h1 : o ≥ 4 + r / 3)  -- Condition 1: oats ≥ 4 + 1/3 * rice
  (h2 : o ≤ 3 * r)      -- Condition 2: oats ≤ 3 * rice
  : r ≥ 3/2 := by sorry

/-- The exact minimum amount of rice Maria could purchase is 1.5 kg. -/
theorem exact_min_rice_purchase : 
  ∃ (o r : ℝ), r = 3/2 ∧ o = 4.5 ∧ o ≥ 4 + r / 3 ∧ o ≤ 3 * r := by sorry

end min_rice_purchase_exact_min_rice_purchase_l1798_179842


namespace min_additional_squares_for_symmetry_l1798_179828

/-- Represents a position on the grid -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the grid -/
def Grid := List Position

/-- The initially shaded squares -/
def initial_shaded : Grid := 
  [⟨1, 2⟩, ⟨3, 1⟩, ⟨4, 4⟩, ⟨6, 1⟩]

/-- Function to check if a grid has both horizontal and vertical symmetry -/
def has_symmetry (g : Grid) : Bool := sorry

/-- Function to count the number of additional squares needed for symmetry -/
def additional_squares_needed (g : Grid) : Nat := sorry

/-- Theorem stating that 8 additional squares are needed for symmetry -/
theorem min_additional_squares_for_symmetry :
  additional_squares_needed initial_shaded = 8 := by sorry

end min_additional_squares_for_symmetry_l1798_179828


namespace toothpicks_stage_15_l1798_179817

/-- Calculates the number of toothpicks at a given stage -/
def toothpicks (stage : ℕ) : ℕ :=
  let initial := 3
  let baseIncrease := 2
  let extraIncreaseInterval := 3
  let extraIncrease := (stage - 1) / extraIncreaseInterval

  initial + (stage - 1) * baseIncrease + 
    ((stage - 1) / extraIncreaseInterval) * (stage - 1) * (stage - 2) / 2

theorem toothpicks_stage_15 : toothpicks 15 = 61 := by
  sorry

#eval toothpicks 15

end toothpicks_stage_15_l1798_179817


namespace total_cost_of_pen_and_pencil_l1798_179835

theorem total_cost_of_pen_and_pencil (pencil_cost : ℝ) (h1 : pencil_cost = 8) :
  let pen_cost := pencil_cost / 2
  pencil_cost + pen_cost = 12 := by
sorry

end total_cost_of_pen_and_pencil_l1798_179835


namespace product_inequality_l1798_179863

theorem product_inequality (n : ℕ) (x : ℕ → ℝ) 
  (h_n : n ≥ 3) 
  (h_x_pos : ∀ i ∈ Finset.range (n - 1), x (i + 2) > 0)
  (h_x_prod : (Finset.range (n - 1)).prod (λ i => x (i + 2)) = 1) :
  (Finset.range (n - 1)).prod (λ i => (1 + x (i + 2)) ^ (i + 2)) > n ^ n := by
  sorry

end product_inequality_l1798_179863


namespace equal_roots_quadratic_l1798_179868

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - k + 1 = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y - k + 1 = 0 → y = x) → 
  k = 0 := by
sorry

end equal_roots_quadratic_l1798_179868


namespace f_monotone_and_bounded_l1798_179852

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2 - a * Real.sin x - 1

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - x - a * Real.cos x

theorem f_monotone_and_bounded (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  (∀ M : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-π/3) (π/3) → |f_deriv a x| ≤ M) →
    ∀ x : ℝ, x ∈ Set.Icc (-π/3) (π/3) → |f a x| ≤ M) :=
by sorry

end f_monotone_and_bounded_l1798_179852


namespace lcm_of_153_180_560_l1798_179803

theorem lcm_of_153_180_560 : Nat.lcm 153 (Nat.lcm 180 560) = 85680 := by
  sorry

end lcm_of_153_180_560_l1798_179803


namespace z_in_second_quadrant_l1798_179820

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- The given complex number -/
def z : ℂ := (1 + 2 * i) * i

/-- A complex number is in the second quadrant if its real part is negative and its imaginary part is positive -/
def is_in_second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

theorem z_in_second_quadrant : is_in_second_quadrant z := by
  sorry

end z_in_second_quadrant_l1798_179820


namespace yarn_theorem_l1798_179865

def yarn_problem (B1 : ℝ) : Prop :=
  let B2 := 2 * B1
  let B3 := 3 * B1
  let B4 := 2 * B2
  let B5 := B3 + B4
  B3 = 27 ∧ B2 = 18

theorem yarn_theorem : ∃ B1 : ℝ, yarn_problem B1 := by
  sorry

end yarn_theorem_l1798_179865


namespace remainder_sum_powers_mod_5_l1798_179893

theorem remainder_sum_powers_mod_5 : (9^5 + 11^6 + 12^7) % 5 = 1 := by
  sorry

end remainder_sum_powers_mod_5_l1798_179893


namespace decimal_expansion_non_periodic_length_l1798_179823

/-- The length of the non-periodic part of the decimal expansion of 1/n -/
def nonPeriodicLength (n : ℕ) : ℕ :=
  max (Nat.factorization n 2) (Nat.factorization n 5)

/-- Theorem stating that for any natural number n > 1, the length of the non-periodic part
    of the decimal expansion of 1/n is equal to max[v₂(n), v₅(n)] -/
theorem decimal_expansion_non_periodic_length (n : ℕ) (h : n > 1) :
  nonPeriodicLength n = max (Nat.factorization n 2) (Nat.factorization n 5) := by
  sorry

#check decimal_expansion_non_periodic_length

end decimal_expansion_non_periodic_length_l1798_179823


namespace inequality_proof_l1798_179867

theorem inequality_proof (x : ℝ) (h1 : 3/2 ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end inequality_proof_l1798_179867


namespace age_problem_l1798_179805

theorem age_problem (a b : ℚ) : 
  (a = 2 * (b - (a - b))) →  -- Condition 1
  (a + (a - b) + b + (a - b) = 130) →  -- Condition 2
  (a = 57 + 7/9 ∧ b = 43 + 1/3) := by
sorry

end age_problem_l1798_179805


namespace lcm_problem_l1798_179812

theorem lcm_problem (m : ℕ+) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 24 := by
  sorry

end lcm_problem_l1798_179812


namespace factorial_sum_unit_digit_l1798_179882

theorem factorial_sum_unit_digit : (Nat.factorial 25 + Nat.factorial 17 - Nat.factorial 18) % 10 = 0 := by
  sorry

end factorial_sum_unit_digit_l1798_179882


namespace arithmetic_sequence_sum_l1798_179888

/-- An arithmetic sequence -/
def arithmeticSeq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The property that a₁ + a₇ + a₁₃ = 4 -/
def sumProperty (a : ℕ → ℚ) : Prop :=
  a 1 + a 7 + a 13 = 4

theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h1 : arithmeticSeq a) (h2 : sumProperty a) : 
  a 2 + a 12 = 8/3 := by
  sorry

end arithmetic_sequence_sum_l1798_179888


namespace solve_equation_one_solve_equation_two_l1798_179849

-- Equation 1
theorem solve_equation_one (x : ℝ) : 2 * x - 7 = 5 * x - 1 → x = -2 := by
  sorry

-- Equation 2
theorem solve_equation_two (x : ℝ) : (x - 2) / 2 - (x - 1) / 6 = 1 → x = 11 / 2 := by
  sorry

end solve_equation_one_solve_equation_two_l1798_179849


namespace smallest_four_digit_multiple_of_112_l1798_179858

theorem smallest_four_digit_multiple_of_112 : ∃ n : ℕ, 
  (n = 1008) ∧ 
  (n ≥ 1000) ∧ 
  (n < 10000) ∧ 
  (n % 112 = 0) ∧ 
  (∀ m : ℕ, m ≥ 1000 ∧ m < 10000 ∧ m % 112 = 0 → m ≥ n) :=
sorry

end smallest_four_digit_multiple_of_112_l1798_179858


namespace emery_shoe_alteration_cost_l1798_179836

theorem emery_shoe_alteration_cost :
  let num_pairs : ℕ := 17
  let cost_per_shoe : ℕ := 29
  let total_shoes : ℕ := num_pairs * 2
  let total_cost : ℕ := total_shoes * cost_per_shoe
  total_cost = 986 := by sorry

end emery_shoe_alteration_cost_l1798_179836


namespace sum_of_reciprocals_lower_bound_l1798_179866

theorem sum_of_reciprocals_lower_bound (a₁ a₂ a₃ : ℝ) 
  (pos₁ : a₁ > 0) (pos₂ : a₂ > 0) (pos₃ : a₃ > 0) 
  (sum_eq_one : a₁ + a₂ + a₃ = 1) : 
  1 / a₁ + 1 / a₂ + 1 / a₃ ≥ 9 := by
  sorry

end sum_of_reciprocals_lower_bound_l1798_179866


namespace euclidean_algorithm_fibonacci_bound_l1798_179871

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define the Euclidean algorithm
def euclidean_algorithm (m₀ m₁ : ℕ) : ℕ → Prop
  | 0 => m₁ = 0
  | k + 1 => ∃ q r, m₀ = q * m₁ + r ∧ r < m₁ ∧ euclidean_algorithm m₁ r k

-- Theorem statement
theorem euclidean_algorithm_fibonacci_bound {m₀ m₁ k : ℕ} 
  (h : euclidean_algorithm m₀ m₁ k) : 
  m₁ ≥ fib (k + 1) ∧ m₀ ≥ fib (k + 2) := by
  sorry

end euclidean_algorithm_fibonacci_bound_l1798_179871


namespace min_value_reciprocal_sum_l1798_179876

def data : List ℝ := [2, 4, 6, 8]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem min_value_reciprocal_sum 
  (m : ℝ) 
  (n : ℝ) 
  (hm : m = median data) 
  (hn : n = variance data) 
  (a : ℝ) 
  (b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (heq : m * a + n * b = 1) : 
  (1 / a + 1 / b) ≥ 20 :=
sorry

end min_value_reciprocal_sum_l1798_179876


namespace linear_function_property_l1798_179808

/-- A linear function is a function of the form f(x) = mx + b, where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (hLinear : LinearFunction g) 
  (hDiff : g 10 - g 0 = 20) : 
  g 20 - g 0 = 40 := by
  sorry

end linear_function_property_l1798_179808


namespace sum_difference_equals_three_l1798_179819

theorem sum_difference_equals_three : (2 + 4 + 6) - (1 + 3 + 5) = 3 := by
  sorry

end sum_difference_equals_three_l1798_179819


namespace amelia_weekly_goal_l1798_179892

/-- Amelia's weekly Jet Bar sales goal -/
def weekly_goal (monday_sales tuesday_sales remaining : ℕ) : ℕ :=
  monday_sales + tuesday_sales + remaining

/-- Theorem: Amelia's weekly Jet Bar sales goal is 90 -/
theorem amelia_weekly_goal :
  ∀ (monday_sales tuesday_sales remaining : ℕ),
  monday_sales = 45 →
  tuesday_sales = monday_sales - 16 →
  remaining = 16 →
  weekly_goal monday_sales tuesday_sales remaining = 90 :=
by
  sorry

end amelia_weekly_goal_l1798_179892


namespace min_a4_value_l1798_179872

theorem min_a4_value (a : Fin 10 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_a2 : a 2 = a 1 + a 5)
  (h_a3 : a 3 = a 2 + a 6)
  (h_a4 : a 4 = a 3 + a 7)
  (h_a6 : a 6 = a 5 + a 8)
  (h_a7 : a 7 = a 6 + a 9)
  (h_a9 : a 9 = a 8 + a 10) :
  ∀ b : Fin 10 → ℕ+, 
    (∀ i j, i ≠ j → b i ≠ b j) →
    (b 2 = b 1 + b 5) →
    (b 3 = b 2 + b 6) →
    (b 4 = b 3 + b 7) →
    (b 6 = b 5 + b 8) →
    (b 7 = b 6 + b 9) →
    (b 9 = b 8 + b 10) →
    a 4 ≤ b 4 :=
by sorry

#check min_a4_value

end min_a4_value_l1798_179872


namespace base_of_equation_l1798_179832

theorem base_of_equation (k x : ℝ) (h1 : (1/2)^23 * (1/81)^k = 1/x^23) (h2 : k = 11.5) : x = 18 := by
  sorry

end base_of_equation_l1798_179832


namespace priyanka_value_l1798_179830

/-- A system representing the values of individuals --/
structure ValueSystem where
  Neha : ℕ
  Sonali : ℕ
  Priyanka : ℕ
  Sadaf : ℕ
  Tanu : ℕ

/-- The theorem stating Priyanka's value in the given system --/
theorem priyanka_value (sys : ValueSystem) 
  (h1 : sys.Sonali = 15)
  (h2 : sys.Priyanka = 15)
  (h3 : sys.Sadaf = sys.Neha)
  (h4 : sys.Tanu = sys.Neha) :
  sys.Priyanka = 15 := by
    sorry

end priyanka_value_l1798_179830


namespace range_of_s_squared_minus_c_squared_l1798_179850

theorem range_of_s_squared_minus_c_squared (k : ℝ) (x y : ℝ) :
  k > 0 →
  x = k * y →
  let r := Real.sqrt (x^2 + y^2)
  let s := y / r
  let c := x / r
  (∀ z, s^2 - c^2 = z → -1 ≤ z ∧ z ≤ 1) ∧
  (∃ z, s^2 - c^2 = z ∧ z = -1) ∧
  (∃ z, s^2 - c^2 = z ∧ z = 1) :=
by sorry

end range_of_s_squared_minus_c_squared_l1798_179850


namespace crayons_per_friend_l1798_179848

def total_crayons : ℕ := 210
def num_friends : ℕ := 30

theorem crayons_per_friend :
  total_crayons / num_friends = 7 :=
by sorry

end crayons_per_friend_l1798_179848


namespace linear_function_quadrants_l1798_179878

/-- A linear function f(x) = mx + b passes through a quadrant if there exists a point (x, y) in that quadrant such that y = f(x) -/
def passes_through_quadrant (m b : ℝ) (q : Nat) : Prop :=
  match q with
  | 1 => ∃ x y, x > 0 ∧ y > 0 ∧ y = m * x + b
  | 2 => ∃ x y, x < 0 ∧ y > 0 ∧ y = m * x + b
  | 3 => ∃ x y, x < 0 ∧ y < 0 ∧ y = m * x + b
  | 4 => ∃ x y, x > 0 ∧ y < 0 ∧ y = m * x + b
  | _ => False

/-- The linear function y = -2x + 1 passes through Quadrants I, II, and IV -/
theorem linear_function_quadrants :
  passes_through_quadrant (-2) 1 1 ∧
  passes_through_quadrant (-2) 1 2 ∧
  passes_through_quadrant (-2) 1 4 :=
sorry

end linear_function_quadrants_l1798_179878
