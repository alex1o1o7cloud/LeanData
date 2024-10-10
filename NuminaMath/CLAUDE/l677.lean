import Mathlib

namespace find_k_l677_67705

theorem find_k (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x : ℝ, (x^2 - k)*(x + k) = x^3 + k*(x^2 - x - 7)) → k = 7 := by
  sorry

end find_k_l677_67705


namespace zain_coin_count_l677_67744

/-- Represents the number of coins Emerie has of each type -/
structure EmerieCoins where
  quarters : Nat
  dimes : Nat
  nickels : Nat

/-- Calculates the total number of coins Zain has given Emerie's coin counts -/
def zainTotalCoins (e : EmerieCoins) : Nat :=
  (e.quarters + 10) + (e.dimes + 10) + (e.nickels + 10)

theorem zain_coin_count (e : EmerieCoins) 
  (hq : e.quarters = 6) 
  (hd : e.dimes = 7) 
  (hn : e.nickels = 5) : 
  zainTotalCoins e = 48 := by
  sorry

end zain_coin_count_l677_67744


namespace special_polynomial_p_count_l677_67716

/-- Represents a polynomial of degree 4 with specific properties -/
structure SpecialPolynomial where
  m : ℤ
  n : ℤ
  p : ℤ
  zeros : Fin 4 → ℝ
  is_zero : ∀ i, (zeros i)^4 - 2004 * (zeros i)^3 + m * (zeros i)^2 + n * (zeros i) + p = 0
  distinct_zeros : ∀ i j, i ≠ j → zeros i ≠ zeros j
  positive_zeros : ∀ i, zeros i > 0
  integer_zero : ∃ i, ∃ k : ℤ, zeros i = k
  sum_property : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ zeros i = zeros j + zeros k
  product_property : ∃ i j k l, i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧ zeros i = zeros j * zeros k * zeros l

/-- The number of possible values for p in a SpecialPolynomial -/
def count_p_values : ℕ := 63000

/-- Theorem stating that there are exactly 63000 possible values for p -/
theorem special_polynomial_p_count :
  (∃ f : Set SpecialPolynomial → ℕ, f {sp | sp.p = p} = count_p_values) :=
sorry

end special_polynomial_p_count_l677_67716


namespace product_mod_600_l677_67717

theorem product_mod_600 : (1234 * 2047) % 600 = 198 := by
  sorry

end product_mod_600_l677_67717


namespace day_150_of_year_N_minus_1_is_friday_l677_67790

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek (yd : YearDay) : DayOfWeek := sorry

/-- Theorem stating the problem conditions and the result to be proved -/
theorem day_150_of_year_N_minus_1_is_friday 
  (N : Int) 
  (h1 : dayOfWeek ⟨N, 250⟩ = DayOfWeek.Monday) 
  (h2 : dayOfWeek ⟨N + 2, 300⟩ = DayOfWeek.Monday) :
  dayOfWeek ⟨N - 1, 150⟩ = DayOfWeek.Friday := by sorry

end day_150_of_year_N_minus_1_is_friday_l677_67790


namespace distance_calculation_l677_67784

/-- The distance between Maxwell's and Brad's homes -/
def distance_between_homes : ℝ := 34

/-- Maxwell's walking speed in km/h -/
def maxwell_speed : ℝ := 4

/-- Brad's running speed in km/h -/
def brad_speed : ℝ := 6

/-- Time Maxwell walks before meeting Brad, in hours -/
def maxwell_time : ℝ := 4

/-- Time Brad runs before meeting Maxwell, in hours -/
def brad_time : ℝ := 3

theorem distance_calculation :
  distance_between_homes = maxwell_speed * maxwell_time + brad_speed * brad_time :=
by sorry

end distance_calculation_l677_67784


namespace binary_equals_octal_l677_67712

-- Define the binary number
def binary_num : List Bool := [true, false, true, true, true, false]

-- Define the octal number
def octal_num : Nat := 56

-- Function to convert binary to decimal
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Function to convert decimal to octal
def decimal_to_octal (n : Nat) : Nat :=
  if n < 8 then n
  else 10 * (decimal_to_octal (n / 8)) + (n % 8)

-- Theorem stating that the binary number is equal to the octal number
theorem binary_equals_octal : 
  decimal_to_octal (binary_to_decimal binary_num) = octal_num := by
  sorry

end binary_equals_octal_l677_67712


namespace men_earnings_l677_67750

/-- The amount earned by a group of workers given their daily wage and work duration -/
def amount_earned (num_workers : ℕ) (days : ℕ) (daily_wage : ℕ) : ℕ :=
  num_workers * days * daily_wage

theorem men_earnings (woman_daily_wage : ℕ) :
  woman_daily_wage * 40 * 30 = 21600 →
  amount_earned 16 25 (2 * woman_daily_wage) = 14400 := by
  sorry

#check men_earnings

end men_earnings_l677_67750


namespace mingyoungs_animals_l677_67706

theorem mingyoungs_animals (chickens ducks rabbits : ℕ) : 
  chickens = 4 * ducks →
  ducks = rabbits + 17 →
  rabbits = 8 →
  chickens + ducks + rabbits = 133 := by
sorry

end mingyoungs_animals_l677_67706


namespace upper_bound_for_expression_l677_67781

theorem upper_bound_for_expression (n : ℤ) : 
  (∃ ub : ℤ, 
    (ub = 40) ∧ 
    (∀ m : ℤ, 1 < 4*m + 7 → 4*m + 7 < ub) ∧
    (∃! (l : List ℤ), l.length = 10 ∧ 
      (∀ k : ℤ, k ∈ l ↔ (1 < 4*k + 7 ∧ 4*k + 7 < ub)))) :=
by sorry

end upper_bound_for_expression_l677_67781


namespace infinite_solutions_sum_l677_67796

/-- If the equation ax - 4 = 14x + b has infinitely many solutions, then a + b = 10 -/
theorem infinite_solutions_sum (a b : ℝ) : 
  (∀ x, a * x - 4 = 14 * x + b) → a + b = 10 := by
  sorry

end infinite_solutions_sum_l677_67796


namespace clerk_salary_l677_67751

theorem clerk_salary (manager_salary : ℝ) (num_managers : ℕ) (num_clerks : ℕ) (total_salary : ℝ) :
  manager_salary = 5 →
  num_managers = 2 →
  num_clerks = 3 →
  total_salary = 16 →
  ∃ (clerk_salary : ℝ), clerk_salary = 2 ∧ total_salary = num_managers * manager_salary + num_clerks * clerk_salary :=
by
  sorry

end clerk_salary_l677_67751


namespace altitude_length_l677_67769

-- Define the right triangle DEF
def RightTriangleDEF (DE DF EF : ℝ) : Prop :=
  DE = 15 ∧ DF = 9 ∧ EF = 12 ∧ DE^2 = DF^2 + EF^2

-- Define the altitude from F to DE
def Altitude (DE DF EF h : ℝ) : Prop :=
  h * DE = 2 * (1/2 * DF * EF)

-- Theorem statement
theorem altitude_length (DE DF EF h : ℝ) 
  (hTriangle : RightTriangleDEF DE DF EF) 
  (hAltitude : Altitude DE DF EF h) : 
  h = 7.2 := by sorry

end altitude_length_l677_67769


namespace annie_initial_money_l677_67741

def hamburger_price : ℕ := 4
def cheeseburger_price : ℕ := 5
def fries_price : ℕ := 3
def milkshake_price : ℕ := 5
def smoothie_price : ℕ := 6

def hamburger_count : ℕ := 8
def cheeseburger_count : ℕ := 5
def fries_count : ℕ := 3
def milkshake_count : ℕ := 6
def smoothie_count : ℕ := 4

def discount : ℕ := 10
def money_left : ℕ := 45

def total_cost : ℕ := 
  hamburger_price * hamburger_count +
  cheeseburger_price * cheeseburger_count +
  fries_price * fries_count +
  milkshake_price * milkshake_count +
  smoothie_price * smoothie_count

def discounted_cost : ℕ := total_cost - discount

theorem annie_initial_money : 
  discounted_cost + money_left = 155 := by sorry

end annie_initial_money_l677_67741


namespace suzanne_reading_l677_67729

theorem suzanne_reading (total_pages : ℕ) (extra_pages : ℕ) (pages_left : ℕ) 
  (h1 : total_pages = 64)
  (h2 : extra_pages = 16)
  (h3 : pages_left = 18) :
  ∃ (monday_pages : ℕ), 
    monday_pages + (monday_pages + extra_pages) = total_pages - pages_left ∧ 
    monday_pages = 15 := by
  sorry

end suzanne_reading_l677_67729


namespace largest_angle_right_triangle_l677_67792

/-- A right triangle with acute angles in the ratio 8:1 has its largest angle measuring 90 degrees. -/
theorem largest_angle_right_triangle (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_acute_ratio : a / b = 8 ∨ b / a = 8) : max a (max b c) = 90 := by
  sorry

end largest_angle_right_triangle_l677_67792


namespace polynomial_transformation_l677_67794

-- Define the original polynomial
def original_poly (b : ℝ) (x : ℝ) : ℝ := x^4 - b*x - 3

-- Define the transformed polynomial
def transformed_poly (b : ℝ) (x : ℝ) : ℝ := 3*x^4 - b*x^3 - 1

theorem polynomial_transformation (b : ℝ) (a c d : ℝ) :
  (original_poly b a = 0 ∧ original_poly b b = 0 ∧ original_poly b c = 0 ∧ original_poly b d = 0) →
  (transformed_poly b ((a + b + c) / d^2) = 0 ∧
   transformed_poly b ((a + b + d) / c^2) = 0 ∧
   transformed_poly b ((a + c + d) / b^2) = 0 ∧
   transformed_poly b ((b + c + d) / a^2) = 0) :=
by sorry

end polynomial_transformation_l677_67794


namespace alexander_paintings_l677_67788

/-- The number of paintings at each new gallery --/
def paintings_per_new_gallery : ℕ := 2

theorem alexander_paintings :
  let first_gallery_paintings : ℕ := 9
  let new_galleries : ℕ := 5
  let pencils_per_painting : ℕ := 4
  let signature_pencils_per_gallery : ℕ := 2
  let total_pencils_used : ℕ := 88
  
  paintings_per_new_gallery = 
    ((total_pencils_used - 
      (signature_pencils_per_gallery * (new_galleries + 1)) - 
      (first_gallery_paintings * pencils_per_painting)) 
     / (new_galleries * pencils_per_painting)) :=
by
  sorry

end alexander_paintings_l677_67788


namespace money_left_calculation_l677_67713

-- Define the initial amount, spent amount, and amount given to each friend
def initial_amount : ℚ := 5.10
def spent_on_sweets : ℚ := 1.05
def given_to_friend : ℚ := 1.00
def number_of_friends : ℕ := 2

-- Theorem to prove
theorem money_left_calculation :
  initial_amount - (spent_on_sweets + number_of_friends * given_to_friend) = 2.05 := by
  sorry


end money_left_calculation_l677_67713


namespace no_adjacent_standing_probability_l677_67799

/-- Recursive function to calculate the number of valid arrangements -/
def validArrangements : ℕ → ℕ
| 0 => 1
| 1 => 2
| n + 2 => validArrangements (n + 1) + validArrangements n

/-- The number of people around the table -/
def numPeople : ℕ := 10

/-- The total number of possible outcomes when flipping n fair coins -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

/-- The probability of no two adjacent people standing for n people -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  (validArrangements n : ℚ) / (totalOutcomes n : ℚ)

theorem no_adjacent_standing_probability :
  noAdjacentStandingProb numPeople = 123 / 1024 := by
  sorry

end no_adjacent_standing_probability_l677_67799


namespace equation_solution_l677_67768

theorem equation_solution :
  ∃ x : ℚ, (3 / (2 * x - 2) + 1 / (1 - x) = 3) ∧ (x = 7 / 6) := by
  sorry

end equation_solution_l677_67768


namespace arithmetic_expression_equality_l677_67752

theorem arithmetic_expression_equality : 3 + 5 * 2^3 - 4 / 2 + 7 * 3 = 62 := by
  sorry

end arithmetic_expression_equality_l677_67752


namespace savings_calculation_l677_67715

def calculate_savings (initial_winnings : ℚ) (first_saving_ratio : ℚ) (profit_ratio : ℚ) (second_saving_ratio : ℚ) : ℚ :=
  let first_saving := initial_winnings * first_saving_ratio
  let second_bet := initial_winnings * (1 - first_saving_ratio)
  let second_bet_earnings := second_bet * (1 + profit_ratio)
  let second_saving := second_bet_earnings * second_saving_ratio
  first_saving + second_saving

theorem savings_calculation :
  calculate_savings 100 (1/2) (3/5) (1/2) = 90 := by
  sorry

end savings_calculation_l677_67715


namespace sqrt_difference_approximation_l677_67785

theorem sqrt_difference_approximation : 
  |Real.sqrt 75 - Real.sqrt 72 - 0.17| < 0.01 := by
  sorry

end sqrt_difference_approximation_l677_67785


namespace circle_line_tangent_l677_67721

/-- A circle C in the xy-plane -/
def Circle (a : ℝ) (x y : ℝ) : Prop :=
  x^2 - 2*a*x + y^2 = 0

/-- A line l in the xy-plane -/
def Line (x y : ℝ) : Prop :=
  x - Real.sqrt 3 * y + 3 = 0

/-- The circle and line are tangent if they intersect at exactly one point -/
def Tangent (a : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, Circle a p.1 p.2 ∧ Line p.1 p.2

theorem circle_line_tangent (a : ℝ) (h1 : a > 0) (h2 : Tangent a) : a = 3 := by
  sorry

end circle_line_tangent_l677_67721


namespace exists_point_X_l677_67777

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define the problem setup
def problem_setup (A B : ℝ × ℝ) (circle : Circle) (MN : Line) :=
  ∃ (X : ℝ × ℝ),
    -- X is on the circle
    (X.1 - circle.center.1)^2 + (X.2 - circle.center.2)^2 = circle.radius^2 ∧
    -- Define lines AX and BX
    let AX : Line := ⟨A, X⟩
    let BX : Line := ⟨B, X⟩
    -- C and D are intersections of AX and BX with the circle
    ∃ (C D : ℝ × ℝ),
      -- C and D are on the circle
      (C.1 - circle.center.1)^2 + (C.2 - circle.center.2)^2 = circle.radius^2 ∧
      (D.1 - circle.center.1)^2 + (D.2 - circle.center.2)^2 = circle.radius^2 ∧
      -- C is on AX, D is on BX
      (C.2 - A.2) * (X.1 - A.1) = (C.1 - A.1) * (X.2 - A.2) ∧
      (D.2 - B.2) * (X.1 - B.1) = (D.1 - B.1) * (X.2 - B.2) ∧
      -- CD is parallel to MN
      (C.2 - D.2) * (MN.point2.1 - MN.point1.1) = (C.1 - D.1) * (MN.point2.2 - MN.point1.2)

-- Theorem statement
theorem exists_point_X (A B : ℝ × ℝ) (circle : Circle) (MN : Line) :
  problem_setup A B circle MN :=
sorry

end exists_point_X_l677_67777


namespace green_blue_difference_l677_67761

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the count of disks for each color -/
structure DiskCounts where
  blue : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total number of disks -/
def totalDisks (counts : DiskCounts) : ℕ :=
  counts.blue + counts.yellow + counts.green

/-- Checks if the given counts match the specified ratio -/
def matchesRatio (counts : DiskCounts) (blueRatio yellowRatio greenRatio : ℕ) : Prop :=
  counts.blue * yellowRatio = counts.yellow * blueRatio ∧
  counts.blue * greenRatio = counts.green * blueRatio

theorem green_blue_difference (counts : DiskCounts) :
  totalDisks counts = 72 →
  matchesRatio counts 3 7 8 →
  counts.green - counts.blue = 20 := by
  sorry

end green_blue_difference_l677_67761


namespace multiple_in_selection_l677_67760

theorem multiple_in_selection (S : Finset ℕ) : 
  S ⊆ Finset.range 100 → S.card = 51 → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ∃ (k : ℕ), b = k * a :=
sorry

end multiple_in_selection_l677_67760


namespace nonzero_terms_count_l677_67782

def polynomial (x : ℝ) : ℝ :=
  (x - 3) * (3 * x^3 + 2 * x^2 - 4 * x + 1) + 4 * (x^4 + x^3 - 2 * x^2 + x) - 5 * (x^3 - 3 * x + 1)

theorem nonzero_terms_count :
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
    ∀ x, polynomial x = a * x^4 + b * x^3 + c * x^2 + d * x + e :=
by sorry

end nonzero_terms_count_l677_67782


namespace card_problem_solution_l677_67754

/-- Represents the types of cards -/
inductive CardType
  | WW  -- White-White
  | BB  -- Black-Black
  | BW  -- Black-White

/-- Represents the state of a set of cards -/
structure CardSet where
  total : Nat
  blackUp : Nat

/-- Represents the problem setup -/
structure CardProblem where
  initialState : CardSet
  afterFirst : CardSet
  afterSecond : CardSet
  afterThird : CardSet

/-- The main theorem to prove -/
theorem card_problem_solution (p : CardProblem) : 
  p.initialState.total = 12 ∧ 
  p.initialState.blackUp = 9 ∧
  p.afterFirst.blackUp = 4 ∧
  p.afterSecond.blackUp = 6 ∧
  p.afterThird.blackUp = 5 →
  ∃ (bw ww : Nat), bw = 9 ∧ ww = 3 ∧ bw + ww = p.initialState.total := by
  sorry


end card_problem_solution_l677_67754


namespace smallest_divisible_by_9_l677_67757

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def insert_digit (a b d : ℕ) : ℕ := a * 10 + d * 10 + b

theorem smallest_divisible_by_9 :
  ∀ d : ℕ, d ≥ 3 →
    is_divisible_by_9 (insert_digit 761 829 d) →
    insert_digit 761 829 3 ≤ insert_digit 761 829 d :=
by sorry

end smallest_divisible_by_9_l677_67757


namespace apple_pie_problem_l677_67767

def max_pies (total_apples unripe_apples apples_per_pie : ℕ) : ℕ :=
  (total_apples - unripe_apples) / apples_per_pie

theorem apple_pie_problem :
  max_pies 34 6 4 = 7 := by
  sorry

end apple_pie_problem_l677_67767


namespace function_inequality_l677_67783

/-- The function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The function g(x) = ax + 2, where a > 0 -/
def g (a : ℝ) (x : ℝ) : ℝ := a*x + 2

/-- The theorem statement -/
theorem function_inequality (a : ℝ) (h_a : a > 0) :
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, x₂ ≥ -2 ∧ f x₁ > g a x₂) →
  a > 3/2 :=
by sorry

end function_inequality_l677_67783


namespace sally_quarters_l677_67708

/-- Given an initial quantity of quarters and an additional amount received,
    calculate the total number of quarters. -/
def total_quarters (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given 760 initial quarters and 418 additional quarters,
    the total number of quarters is 1178. -/
theorem sally_quarters : total_quarters 760 418 = 1178 := by
  sorry

end sally_quarters_l677_67708


namespace modulus_z_is_sqrt_5_l677_67771

theorem modulus_z_is_sqrt_5 (z : ℂ) (h : (1 + Complex.I) * z = 3 + Complex.I) :
  Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_z_is_sqrt_5_l677_67771


namespace inequality_proof_l677_67775

theorem inequality_proof (a b c : ℝ) 
  (ha : a = (1/3)^(1/3)) 
  (hb : b = Real.log (1/2)) 
  (hc : c = Real.log (1/4) / Real.log (1/3)) : 
  b < a ∧ a < c := by
  sorry

end inequality_proof_l677_67775


namespace parallel_vectors_component_l677_67780

/-- Given two vectors a and b in ℝ², prove that if a is parallel to b,
    then the first component of a must be -1. -/
theorem parallel_vectors_component (a b : ℝ × ℝ) :
  a.1 = m ∧ a.2 = Real.sqrt 3 ∧ b.1 = Real.sqrt 3 ∧ b.2 = -3 ∧
  ∃ (k : ℝ), a = k • b →
  m = -1 := by
sorry

end parallel_vectors_component_l677_67780


namespace trig_expression_equals_negative_one_l677_67700

theorem trig_expression_equals_negative_one :
  (Real.sin (24 * π / 180) * Real.cos (6 * π / 180) - Real.sin (6 * π / 180) * Real.sin (66 * π / 180)) /
  (Real.sin (21 * π / 180) * Real.cos (39 * π / 180) - Real.sin (39 * π / 180) * Real.cos (21 * π / 180)) = -1 := by
  sorry

end trig_expression_equals_negative_one_l677_67700


namespace prob_one_black_one_red_is_three_fifths_l677_67797

/-- Represents the color of a ball -/
inductive BallColor
  | Red
  | Black

/-- Represents a ball with a color and number -/
structure Ball where
  color : BallColor
  number : Nat

/-- The bag of balls -/
def bag : Finset Ball := sorry

/-- The number of red balls in the bag -/
def num_red_balls : Nat := sorry

/-- The number of black balls in the bag -/
def num_black_balls : Nat := sorry

/-- The total number of balls in the bag -/
def total_balls : Nat := sorry

/-- The probability of drawing one black ball and one red ball in the first two draws -/
def prob_one_black_one_red : ℚ := sorry

/-- Theorem stating the probability of drawing one black ball and one red ball in the first two draws -/
theorem prob_one_black_one_red_is_three_fifths :
  prob_one_black_one_red = 3 / 5 := by sorry

end prob_one_black_one_red_is_three_fifths_l677_67797


namespace initial_overs_correct_l677_67714

/-- Represents the number of overs played initially in a cricket game. -/
def initial_overs : ℝ := 10

/-- The target score for the cricket game. -/
def target_score : ℝ := 282

/-- The initial run rate in runs per over. -/
def initial_run_rate : ℝ := 6.2

/-- The required run rate for the remaining overs in runs per over. -/
def required_run_rate : ℝ := 5.5

/-- The number of remaining overs. -/
def remaining_overs : ℝ := 40

/-- Theorem stating that the initial number of overs is correct given the conditions. -/
theorem initial_overs_correct : 
  target_score = initial_run_rate * initial_overs + required_run_rate * remaining_overs :=
by sorry

end initial_overs_correct_l677_67714


namespace intersection_with_complement_l677_67787

open Set

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def P : Set Nat := {1, 2, 3, 4}
def Q : Set Nat := {3, 4, 5}

theorem intersection_with_complement :
  P ∩ (U \ Q) = {1, 2} := by sorry

end intersection_with_complement_l677_67787


namespace probability_three_games_probability_best_of_five_l677_67709

-- Define the probability of A winning a single game
def p_A : ℚ := 2/3

-- Define the probability of B winning a single game
def p_B : ℚ := 1/3

-- Theorem for part (1)
theorem probability_three_games 
  (h1 : p_A + p_B = 1) 
  (h2 : p_A = 2/3) 
  (h3 : p_B = 1/3) :
  let p_A_wins_two := 3 * (p_A^2 * p_B)
  let p_B_wins_at_least_one := 1 - p_A^3
  (p_A_wins_two = 4/9) ∧ (p_B_wins_at_least_one = 19/27) := by
  sorry

-- Theorem for part (2)
theorem probability_best_of_five
  (h1 : p_A + p_B = 1)
  (h2 : p_A = 2/3)
  (h3 : p_B = 1/3) :
  let p_A_wins_three_straight := p_A^3
  let p_A_wins_in_four := 3 * (p_A^3 * p_B)
  let p_A_wins_in_five := 6 * (p_A^3 * p_B^2)
  p_A_wins_three_straight + p_A_wins_in_four + p_A_wins_in_five = 64/81 := by
  sorry

end probability_three_games_probability_best_of_five_l677_67709


namespace first_day_exceeding_150_l677_67765

def paperclips : ℕ → ℕ
  | 0 => 5  -- Monday (day 1)
  | n + 1 => 2 * paperclips n + 2

theorem first_day_exceeding_150 :
  ∃ n : ℕ, paperclips n > 150 ∧ ∀ m : ℕ, m < n → paperclips m ≤ 150 ∧ n = 5 :=
by sorry

end first_day_exceeding_150_l677_67765


namespace largest_power_dividing_factorial_l677_67730

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_power_dividing_factorial : 
  (∃ k : ℕ, k = 30 ∧ 
   (∀ m : ℕ, 2010^m ∣ factorial 2010 → m ≤ k) ∧
   2010^k ∣ factorial 2010) ∧
  2010 = 2 * 3 * 5 * 67 := by
sorry

end largest_power_dividing_factorial_l677_67730


namespace last_three_nonzero_digits_of_80_factorial_l677_67776

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- Returns the last three nonzero digits of a natural number -/
def lastThreeNonzeroDigits (n : ℕ) : ℕ :=
  n % 1000

theorem last_three_nonzero_digits_of_80_factorial :
  lastThreeNonzeroDigits (factorial 80) = 712 := by
  sorry

end last_three_nonzero_digits_of_80_factorial_l677_67776


namespace sin_6theta_l677_67747

theorem sin_6theta (θ : ℝ) : 
  Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 8) / 5 →
  Real.sin (6 * θ) = -630 * Real.sqrt 8 / 15625 := by
sorry

end sin_6theta_l677_67747


namespace quadratic_general_form_l677_67739

theorem quadratic_general_form :
  ∀ x : ℝ, (6 * x^2 = 5 * x - 4) ↔ (6 * x^2 - 5 * x + 4 = 0) :=
by sorry

end quadratic_general_form_l677_67739


namespace only_two_is_possible_l677_67727

/-- Represents a triangular grid with 9 cells -/
def TriangularGrid := Fin 9 → ℤ

/-- Represents a move on the triangular grid -/
inductive Move
| add (i j : Fin 9) : Move
| subtract (i j : Fin 9) : Move

/-- Applies a move to the grid -/
def applyMove (grid : TriangularGrid) (move : Move) : TriangularGrid :=
  match move with
  | Move.add i j => 
      fun k => if k = i ∨ k = j then grid k + 1 else grid k
  | Move.subtract i j => 
      fun k => if k = i ∨ k = j then grid k - 1 else grid k

/-- Checks if two cells are adjacent in the triangular grid -/
def isAdjacent (i j : Fin 9) : Prop := sorry

/-- Checks if a grid contains consecutive natural numbers from n to n+8 -/
def containsConsecutiveNumbers (grid : TriangularGrid) (n : ℕ) : Prop :=
  ∃ (perm : Fin 9 → Fin 9), ∀ i : Fin 9, grid (perm i) = n + i

/-- The main theorem stating that n = 2 is the only solution -/
theorem only_two_is_possible :
  ∀ (n : ℕ),
    (∃ (grid : TriangularGrid) (moves : List Move),
      (∀ i : Fin 9, grid i = 0) ∧
      (∀ move ∈ moves, ∃ i j, move = Move.add i j ∨ move = Move.subtract i j) ∧
      (∀ move ∈ moves, ∃ i j, isAdjacent i j) ∧
      (containsConsecutiveNumbers (moves.foldl applyMove grid) n)) ↔
    n = 2 := by
  sorry


end only_two_is_possible_l677_67727


namespace lottery_winnings_l677_67734

theorem lottery_winnings (total_given : ℝ) (num_students : ℕ) (fraction : ℝ) :
  total_given = 15525 →
  num_students = 100 →
  fraction = 1 / 1000 →
  ∃ winnings : ℝ, winnings = 155250 ∧ total_given = num_students * (fraction * winnings) :=
by sorry

end lottery_winnings_l677_67734


namespace product_equals_difference_of_powers_l677_67786

theorem product_equals_difference_of_powers : 
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) * (3^16 + 5^16) * 
  (3^32 + 5^32) * (3^64 + 5^64) * (3^128 + 5^128) = 3^256 - 5^256 := by
  sorry

end product_equals_difference_of_powers_l677_67786


namespace homework_time_calculation_l677_67724

theorem homework_time_calculation (total_time : ℝ) :
  (0.3 * total_time = 0.3 * total_time) ∧  -- Time spent on math
  (0.4 * total_time = 0.4 * total_time) ∧  -- Time spent on science
  (total_time - 0.3 * total_time - 0.4 * total_time = 45) →  -- Time spent on other subjects
  total_time = 150 := by
sorry

end homework_time_calculation_l677_67724


namespace max_consecutive_common_divisor_l677_67763

def a (n : ℕ) : ℤ :=
  if 7 ∣ n then n^6 - 2017 else (n^6 - 2017) / 7

theorem max_consecutive_common_divisor :
  (∃ k : ℕ, ∀ i : ℕ, ∃ d > 1, ∀ j : ℕ, j < k → d ∣ a (i + j)) ∧
  (¬∃ k > 2, ∀ i : ℕ, ∃ d > 1, ∀ j : ℕ, j < k → d ∣ a (i + j)) :=
sorry

end max_consecutive_common_divisor_l677_67763


namespace rosa_peach_apple_difference_l677_67735

-- Define the number of peaches and apples for Steven
def steven_peaches : ℕ := 17
def steven_apples : ℕ := 16

-- Define Jake's peaches and apples in terms of Steven's
def jake_peaches : ℕ := steven_peaches - 6
def jake_apples : ℕ := steven_apples + 8

-- Define Rosa's peaches and apples
def rosa_peaches : ℕ := 3 * jake_peaches
def rosa_apples : ℕ := steven_apples / 2

-- Theorem to prove
theorem rosa_peach_apple_difference : rosa_peaches - rosa_apples = 25 := by
  sorry

end rosa_peach_apple_difference_l677_67735


namespace quadratic_minimum_value_l677_67718

theorem quadratic_minimum_value :
  ∃ (min : ℝ), min = -3 ∧ ∀ x : ℝ, (x - 1)^2 - 3 ≥ min := by
  sorry

end quadratic_minimum_value_l677_67718


namespace oliver_candy_boxes_l677_67766

theorem oliver_candy_boxes : ∃ (initial : ℕ), initial + 6 = 14 := by
  sorry

end oliver_candy_boxes_l677_67766


namespace vector_operations_and_parallelism_l677_67791

/-- Given two 2D vectors a and b, prove properties about their linear combinations and parallelism. -/
theorem vector_operations_and_parallelism 
  (a b : ℝ × ℝ) 
  (ha : a = (2, 0)) 
  (hb : b = (1, 4)) : 
  (2 • a + 3 • b = (7, 12)) ∧ 
  (a - 2 • b = (0, -8)) ∧ 
  (∃ k : ℝ, k • a + b = (2*k + 1, 4) ∧ a + 2 • b = (4, 8) ∧ k = 1/2) := by
  sorry


end vector_operations_and_parallelism_l677_67791


namespace combinations_equal_thirty_l677_67745

/-- The number of color options available -/
def num_colors : ℕ := 5

/-- The number of painting method options available -/
def num_methods : ℕ := 3

/-- The number of finish type options available -/
def num_finishes : ℕ := 2

/-- The total number of combinations of color, painting method, and finish type -/
def total_combinations : ℕ := num_colors * num_methods * num_finishes

/-- Theorem stating that the total number of combinations is 30 -/
theorem combinations_equal_thirty : total_combinations = 30 := by
  sorry

end combinations_equal_thirty_l677_67745


namespace product_of_two_numbers_l677_67742

theorem product_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (sum_squares_eq : x^2 + y^2 = 120) : 
  x * y = -20 := by
  sorry

end product_of_two_numbers_l677_67742


namespace sign_of_a_equals_sign_of_r_l677_67772

-- Define the variables and their properties
variable (x y : ℝ → ℝ) -- x and y are real-valued functions
variable (r : ℝ) -- r is the correlation coefficient
variable (a b : ℝ) -- a and b are coefficients in the regression line equation

-- Define the linear relationship and regression line
def linear_relationship (x y : ℝ → ℝ) : Prop :=
  ∃ (m c : ℝ), ∀ t, y t = m * (x t) + c

-- Define the regression line equation
def regression_line (x y : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ t, y t = a * (x t) + b

-- Define the correlation coefficient
def correlation_coefficient (x y : ℝ → ℝ) (r : ℝ) : Prop :=
  ∃ (cov_xy std_x std_y : ℝ), r = cov_xy / (std_x * std_y) ∧ std_x > 0 ∧ std_y > 0

-- State the theorem
theorem sign_of_a_equals_sign_of_r
  (h_linear : linear_relationship x y)
  (h_regression : regression_line x y a b)
  (h_correlation : correlation_coefficient x y r) :
  (a > 0 ↔ r > 0) ∧ (a < 0 ↔ r < 0) :=
sorry

end sign_of_a_equals_sign_of_r_l677_67772


namespace percentage_of_S_grades_l677_67743

def grading_scale (score : ℕ) : String :=
  if 95 ≤ score ∧ score ≤ 100 then "S"
  else if 88 ≤ score ∧ score < 95 then "A"
  else if 80 ≤ score ∧ score < 88 then "B"
  else if 72 ≤ score ∧ score < 80 then "C"
  else if 65 ≤ score ∧ score < 72 then "D"
  else "F"

def scores : List ℕ := [95, 88, 70, 100, 75, 90, 80, 77, 67, 78, 85, 65, 72, 82, 96]

theorem percentage_of_S_grades :
  (scores.filter (λ score => grading_scale score = "S")).length / scores.length * 100 = 20 := by
  sorry

end percentage_of_S_grades_l677_67743


namespace sum_of_decimals_l677_67758

theorem sum_of_decimals :
  5.256 + 2.89 + 3.75 = 11.96 := by
  sorry

end sum_of_decimals_l677_67758


namespace limit_exponential_function_l677_67774

theorem limit_exponential_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 1| ∧ |x - 1| < δ → 
    |((2 * Real.exp (x - 1) - 1) ^ ((3 * x - 1) / (x - 1))) - Real.exp 4| < ε :=
by sorry

end limit_exponential_function_l677_67774


namespace cube_root_function_l677_67702

theorem cube_root_function (k : ℝ) : 
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, y = k * x^(1/3)) →
  (4 * Real.sqrt 3 = k * 64^(1/3)) →
  (2 * Real.sqrt 3 = k * 8^(1/3)) := by sorry

end cube_root_function_l677_67702


namespace max_m_and_a_value_l677_67789

/-- The function f(x) = |x+3| -/
def f (x : ℝ) : ℝ := |x + 3|

/-- The function g(x) = m - 2|x-11| -/
def g (m : ℝ) (x : ℝ) : ℝ := m - 2*|x - 11|

/-- The theorem stating the maximum value of m and the value of a -/
theorem max_m_and_a_value :
  (∀ x : ℝ, 2 * f x ≥ g m (x + 4)) →
  (∃ t : ℝ, t = 20 ∧ 
    (∀ m' : ℝ, (∀ x : ℝ, 2 * f x ≥ g m' (x + 4)) → m' ≤ t) ∧
    (∀ a : ℝ, a > 0 →
      (∃ x y z : ℝ, 2*x^2 + 3*y^2 + 6*z^2 = a ∧ 
        x + y + z = t/20 ∧
        (∀ x' y' z' : ℝ, 2*x'^2 + 3*y'^2 + 6*z'^2 = a → x' + y' + z' ≤ t/20)) →
      a = 1)) :=
sorry

end max_m_and_a_value_l677_67789


namespace unique_prime_pair_with_prime_root_l677_67728

theorem unique_prime_pair_with_prime_root :
  ∃! (m n : ℕ), Prime m ∧ Prime n ∧
  (∃ x : ℕ, Prime x ∧ x^2 - m*x - n = 0) :=
by
  -- The proof goes here
  sorry

end unique_prime_pair_with_prime_root_l677_67728


namespace quadruplet_solution_l677_67759

theorem quadruplet_solution (x y z w : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0)
  (h_eq : (x*y*z + 1)/(x + 1) = (y*z*w + 1)/(y + 1) ∧
          (y*z*w + 1)/(y + 1) = (z*w*x + 1)/(z + 1) ∧
          (z*w*x + 1)/(z + 1) = (w*x*y + 1)/(w + 1))
  (h_sum : x + y + z + w = 48) :
  x = 12 ∧ y = 12 ∧ z = 12 ∧ w = 12 := by
sorry

end quadruplet_solution_l677_67759


namespace solve_for_y_l677_67773

theorem solve_for_y (t : ℝ) (x y : ℝ) : 
  x = 3 - 2*t → y = 5*t + 3 → x = -7 → y = 28 := by
  sorry

end solve_for_y_l677_67773


namespace factorial_power_of_two_l677_67798

theorem factorial_power_of_two (k : ℕ) :
  ∀ n m : ℕ, (2^k).factorial = 2^n * m ↔
  ∃ t : ℕ, n = 2^k - 1 - t ∧ m = (2^k).factorial / 2^(2^k - 1 - t) := by
  sorry

end factorial_power_of_two_l677_67798


namespace integer_solutions_x_squared_plus_15_eq_y_squared_l677_67719

theorem integer_solutions_x_squared_plus_15_eq_y_squared :
  {(x, y) : ℤ × ℤ | x^2 + 15 = y^2} =
  {(7, 8), (-7, -8), (-7, 8), (7, -8), (1, 4), (-1, -4), (-1, 4), (1, -4)} := by
sorry

end integer_solutions_x_squared_plus_15_eq_y_squared_l677_67719


namespace van_distance_theorem_l677_67762

def distance_covered (initial_time : ℝ) (new_speed : ℝ) (time_ratio : ℝ) : ℝ :=
  new_speed * (initial_time * time_ratio)

theorem van_distance_theorem (initial_time : ℝ) (new_speed : ℝ) (time_ratio : ℝ) :
  initial_time = 5 →
  new_speed = 80 →
  time_ratio = 3/2 →
  distance_covered initial_time new_speed time_ratio = 600 := by
    sorry

end van_distance_theorem_l677_67762


namespace orthic_similarity_condition_l677_67795

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry

/-- The orthic triangle of a given triangle -/
def orthicTriangle (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry

/-- The sequence of orthic triangles starting from an initial triangle -/
def orthicSequence (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℕ → (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)
| 0 => t
| n + 1 => orthicTriangle (orthicSequence t n)

/-- Two triangles are similar -/
def areSimilar (t1 t2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

/-- The main theorem -/
theorem orthic_similarity_condition (n : ℕ) (p : RegularPolygon n) :
  (∃ (v1 v2 v3 : Fin n) (k : ℕ),
    areSimilar
      (p.vertices v1, p.vertices v2, p.vertices v3)
      (orthicSequence (p.vertices v1, p.vertices v2, p.vertices v3) k))
  ↔ Odd n :=
sorry

end orthic_similarity_condition_l677_67795


namespace matthews_cracker_distribution_l677_67753

theorem matthews_cracker_distribution (total_crackers : ℕ) (crackers_per_person : ℕ) (num_friends : ℕ) : 
  total_crackers = 36 → 
  crackers_per_person = 2 → 
  total_crackers = num_friends * crackers_per_person → 
  num_friends = 18 := by
sorry

end matthews_cracker_distribution_l677_67753


namespace rice_yield_and_conversion_l677_67755

-- Define the yield per acre of ordinary rice
def ordinary_yield : ℝ := 600

-- Define the yield per acre of hybrid rice
def hybrid_yield : ℝ := 2 * ordinary_yield

-- Define the acreage difference between fields
def acreage_difference : ℝ := 4

-- Define the total yield of field A
def field_A_yield : ℝ := 9600

-- Define the total yield of field B
def field_B_yield : ℝ := 7200

-- Define the minimum total yield after conversion
def min_total_yield : ℝ := 17700

-- Theorem statement
theorem rice_yield_and_conversion :
  -- Prove that the ordinary yield is 600 kg/acre
  ordinary_yield = 600 ∧
  -- Prove that the hybrid yield is 1200 kg/acre
  hybrid_yield = 1200 ∧
  -- Prove that at least 1.5 acres of field B should be converted
  ∃ (converted_acres : ℝ),
    converted_acres ≥ 1.5 ∧
    field_A_yield + 
    ordinary_yield * (field_B_yield / ordinary_yield - converted_acres) + 
    hybrid_yield * converted_acres ≥ min_total_yield :=
by sorry

end rice_yield_and_conversion_l677_67755


namespace strictly_increasing_function_property_l677_67701

/-- A function f: ℝ → ℝ that satisfies the given conditions -/
def StrictlyIncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

theorem strictly_increasing_function_property
  (f : ℝ → ℝ)
  (h : StrictlyIncreasingFunction f)
  (h1 : f 5 = -1)
  (h2 : f 7 = 0) :
  f (-3) < -1 := by
  sorry

end strictly_increasing_function_property_l677_67701


namespace abs_complex_value_l677_67704

theorem abs_complex_value : Complex.abs (-3 - (9/4)*Complex.I) = 15/4 := by
  sorry

end abs_complex_value_l677_67704


namespace point_not_on_transformed_plane_l677_67738

/-- The original plane equation -/
def plane_equation (x y z : ℝ) : Prop := x - 3*y + 5*z - 1 = 0

/-- The similarity transformation coefficient -/
def k : ℝ := -1

/-- The point A -/
def point_A : ℝ × ℝ × ℝ := (2, 0, -1)

/-- The transformed plane equation -/
def transformed_plane_equation (x y z : ℝ) : Prop := x - 3*y + 5*z + 1 = 0

/-- Theorem stating that point A does not belong to the transformed plane -/
theorem point_not_on_transformed_plane :
  ¬(transformed_plane_equation point_A.1 point_A.2.1 point_A.2.2) :=
sorry

end point_not_on_transformed_plane_l677_67738


namespace det_trig_matrix_zero_l677_67764

theorem det_trig_matrix_zero (α β : Real) : 
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![0, Real.cos α, Real.sin α],
    ![-Real.cos α, 0, Real.cos β],
    ![-Real.sin α, -Real.cos β, 0]
  ]
  Matrix.det M = 0 := by
sorry

end det_trig_matrix_zero_l677_67764


namespace student_distribution_ways_l677_67779

def num_universities : ℕ := 8
def num_students : ℕ := 3
def num_selected_universities : ℕ := 2

theorem student_distribution_ways :
  (num_students.choose 1) * (num_selected_universities.choose 2) * (num_universities.choose 2) = 168 := by
  sorry

end student_distribution_ways_l677_67779


namespace matthew_stocks_solution_l677_67740

def matthew_stocks (expensive_stock_price : ℕ) (cheap_stock_price : ℕ) (cheap_stock_shares : ℕ) (total_assets : ℕ) (expensive_stock_shares : ℕ) : Prop :=
  expensive_stock_price = 2 * cheap_stock_price ∧
  cheap_stock_shares = 26 ∧
  expensive_stock_price = 78 ∧
  total_assets = 2106 ∧
  expensive_stock_shares * expensive_stock_price + cheap_stock_shares * cheap_stock_price = total_assets

theorem matthew_stocks_solution :
  ∃ (expensive_stock_price cheap_stock_price cheap_stock_shares total_assets expensive_stock_shares : ℕ),
    matthew_stocks expensive_stock_price cheap_stock_price cheap_stock_shares total_assets expensive_stock_shares ∧
    expensive_stock_shares = 14 := by
  sorry

end matthew_stocks_solution_l677_67740


namespace pipe_length_proof_l677_67778

theorem pipe_length_proof (shorter_piece longer_piece total_length : ℕ) : 
  shorter_piece = 28 →
  longer_piece = shorter_piece + 12 →
  total_length = shorter_piece + longer_piece →
  total_length = 68 := by
  sorry

end pipe_length_proof_l677_67778


namespace partnership_profit_l677_67722

/-- Given the investment ratios and C's profit share, calculate the total profit -/
theorem partnership_profit (a b c : ℚ) (c_profit : ℚ) : 
  a = 6 ∧ b = 2 ∧ c = 9 ∧ c_profit = 6000.000000000001 →
  (a + b + c) * c_profit / c = 11333.333333333336 :=
by sorry

end partnership_profit_l677_67722


namespace subtraction_from_percentage_l677_67733

theorem subtraction_from_percentage (n : ℝ) : n = 85 → 0.4 * n - 11 = 23 := by
  sorry

end subtraction_from_percentage_l677_67733


namespace basketball_players_l677_67770

theorem basketball_players (C B_and_C B_or_C : ℕ) 
  (h1 : C = 8)
  (h2 : B_and_C = 5)
  (h3 : B_or_C = 10)
  : ∃ B : ℕ, B = 7 ∧ B_or_C = B + C - B_and_C :=
by
  sorry

end basketball_players_l677_67770


namespace unique_number_with_seven_coprimes_l677_67726

def connection (a b : ℕ) : ℚ :=
  (Nat.lcm a b : ℚ) / (a * b : ℚ)

def isCoprimeWithExactlyN (x n : ℕ) : Prop :=
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ y ∈ S, y < 20 ∧ connection x y = 1) ∧
    (∀ y < 20, y ∉ S → connection x y ≠ 1))

theorem unique_number_with_seven_coprimes :
  ∃! x, isCoprimeWithExactlyN x 7 :=
sorry

end unique_number_with_seven_coprimes_l677_67726


namespace one_third_minus_decimal_l677_67720

theorem one_third_minus_decimal : (1 : ℚ) / 3 - 333333 / 1000000 = 1 / (3 * 1000000) := by
  sorry

end one_third_minus_decimal_l677_67720


namespace smallest_divisible_by_one_to_ten_l677_67723

theorem smallest_divisible_by_one_to_ten : 
  ∃ n : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ 
    (∀ m : ℕ, m < n → ∃ j : ℕ, 1 ≤ j ∧ j ≤ 10 ∧ ¬(j ∣ m)) :=
by
  use 2520
  sorry

end smallest_divisible_by_one_to_ten_l677_67723


namespace unique_solution_l677_67748

/-- Returns the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- Returns the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- Returns the product of digits of a two-digit number -/
def digit_product (n : ℕ) : ℕ := tens_digit n * ones_digit n

/-- Returns the reversed number of a two-digit number -/
def reverse_number (n : ℕ) : ℕ := 10 * (ones_digit n) + tens_digit n

/-- Checks if a number satisfies the given conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n / digit_product n = 3) ∧ (n % digit_product n = 8) ∧
  (reverse_number n / digit_product n = 2) ∧ (reverse_number n % digit_product n = 5)

theorem unique_solution : ∃! n : ℕ, satisfies_conditions n ∧ n = 53 :=
  sorry


end unique_solution_l677_67748


namespace necessary_but_not_sufficient_condition_l677_67731

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 - 2*x + 1

-- Define the property of having a non-empty solution set
def has_solution (a : ℝ) : Prop := ∃ x, f a x < 0

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  (∀ a, has_solution a → a ≤ 1) ∧
  ¬(∀ a, a ≤ 1 → has_solution a) :=
sorry

end necessary_but_not_sufficient_condition_l677_67731


namespace A_equality_l677_67736

/-- The number of integer tuples (x₁, x₂, ..., xₖ) satisfying the given conditions -/
def A (n k r : ℕ+) : ℕ := sorry

/-- The theorem stating the equality of A for different arguments -/
theorem A_equality (s t : ℕ+) (hs : s ≥ 2) (ht : t ≥ 2) :
  A (s * t) s t = A (s * (t - 1)) s t ∧ A (s * t) s t = A ((s - 1) * t) s t :=
sorry

end A_equality_l677_67736


namespace correct_quadratic_equation_l677_67737

def quadratic_equation (b c : ℝ) := fun x : ℝ => x^2 + b*x + c

def roots (f : ℝ → ℝ) (r₁ r₂ : ℝ) : Prop :=
  f r₁ = 0 ∧ f r₂ = 0

theorem correct_quadratic_equation :
  ∃ (b₁ c₁ b₂ c₂ : ℝ),
    roots (quadratic_equation b₁ c₁) 5 3 ∧
    roots (quadratic_equation b₂ c₂) (-7) (-2) ∧
    b₁ = -8 ∧
    c₂ = 14 →
    quadratic_equation (-8) 14 = quadratic_equation b₁ c₂ :=
sorry

end correct_quadratic_equation_l677_67737


namespace jam_distribution_and_consumption_l677_67756

/-- Represents the amount of jam and consumption rate for each person -/
structure JamConsumption where
  amount : ℝ
  rate : ℝ

/-- Proves the correct distribution and consumption rates of jam for Ponchik and Syropchik -/
theorem jam_distribution_and_consumption 
  (total_jam : ℝ)
  (ponchik_hypothetical_days : ℝ)
  (syropchik_hypothetical_days : ℝ)
  (h_total : total_jam = 100)
  (h_ponchik : ponchik_hypothetical_days = 45)
  (h_syropchik : syropchik_hypothetical_days = 20)
  : ∃ (ponchik syropchik : JamConsumption),
    ponchik.amount + syropchik.amount = total_jam ∧
    ponchik.amount / ponchik.rate = syropchik.amount / syropchik.rate ∧
    syropchik.amount / ponchik_hypothetical_days = ponchik.rate ∧
    ponchik.amount / syropchik_hypothetical_days = syropchik.rate ∧
    ponchik.amount = 40 ∧
    syropchik.amount = 60 ∧
    ponchik.rate = 4/3 ∧
    syropchik.rate = 2 := by
  sorry


end jam_distribution_and_consumption_l677_67756


namespace min_reciprocal_sum_l677_67793

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 20) :
  (1 / x + 1 / y) ≥ 1 / 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 20 ∧ 1 / x₀ + 1 / y₀ = 1 / 5 := by
  sorry

end min_reciprocal_sum_l677_67793


namespace solution_difference_l677_67710

theorem solution_difference (p q : ℝ) : 
  (p - 5) * (p + 5) = 26 * p - 130 →
  (q - 5) * (q + 5) = 26 * q - 130 →
  p ≠ q →
  p > q →
  p - q = 16 := by
sorry

end solution_difference_l677_67710


namespace f_min_and_g_zeros_l677_67746

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 - 3*x + 2

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f x + (3 - m) * x

theorem f_min_and_g_zeros (h : ∀ x, x > 0 → f x ≥ 0) :
  (∃ x > 0, f x = 0) ∧
  (∀ m < 3, ¬∃ x₁ x₂, x₁ ≠ x₂ ∧ g m x₁ = 0 ∧ g m x₂ = 0) :=
by sorry

end f_min_and_g_zeros_l677_67746


namespace orange_juice_bottles_l677_67732

/-- The number of fluid ounces Christine must buy -/
def min_fl_oz : ℝ := 60

/-- The size of each bottle in milliliters -/
def bottle_size_ml : ℝ := 250

/-- The number of fluid ounces in 1 liter -/
def fl_oz_per_liter : ℝ := 33.8

/-- The smallest number of bottles Christine could buy -/
def min_bottles : ℕ := 8

theorem orange_juice_bottles :
  ∃ (n : ℕ), n = min_bottles ∧
  n * bottle_size_ml / 1000 * fl_oz_per_liter ≥ min_fl_oz ∧
  ∀ (m : ℕ), m * bottle_size_ml / 1000 * fl_oz_per_liter ≥ min_fl_oz → m ≥ n :=
by sorry

end orange_juice_bottles_l677_67732


namespace seventh_term_is_13_l677_67749

def fibonacci_like : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci_like n + fibonacci_like (n + 1)

theorem seventh_term_is_13 : fibonacci_like 6 = 13 := by
  sorry

end seventh_term_is_13_l677_67749


namespace marks_garden_flowers_l677_67711

/-- The number of flowers in Mark's garden -/
def total_flowers : ℕ := by sorry

/-- The number of yellow flowers -/
def yellow_flowers : ℕ := 10

/-- The number of purple flowers -/
def purple_flowers : ℕ := yellow_flowers + (yellow_flowers * 8 / 10)

/-- The number of green flowers -/
def green_flowers : ℕ := (yellow_flowers + purple_flowers) * 25 / 100

/-- The number of red flowers -/
def red_flowers : ℕ := (yellow_flowers + purple_flowers + green_flowers) * 35 / 100

theorem marks_garden_flowers :
  total_flowers = yellow_flowers + purple_flowers + green_flowers + red_flowers ∧
  total_flowers = 47 := by sorry

end marks_garden_flowers_l677_67711


namespace cosine_function_period_l677_67725

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, and d are positive constants,
    if the graph covers two periods in an interval of 2π, then b = 2. -/
theorem cosine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * x + c) + d) →
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.cos (b * (x + 2 * Real.pi) + c) + d) →
  b = 2 := by sorry

end cosine_function_period_l677_67725


namespace max_value_theorem_l677_67707

theorem max_value_theorem (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0)
  (sum_squares : a^2 + b^2 + c^2 = 1) :
  2 * a * b * Real.sqrt 2 + 2 * a * c ≤ 1 ∧ 
  ∃ a₀ b₀ c₀ : ℝ, a₀ ≥ 0 ∧ b₀ ≥ 0 ∧ c₀ ≥ 0 ∧ 
    a₀^2 + b₀^2 + c₀^2 = 1 ∧
    2 * a₀ * b₀ * Real.sqrt 2 + 2 * a₀ * c₀ = 1 :=
by sorry

end max_value_theorem_l677_67707


namespace erdos_mordell_two_points_l677_67703

/-- The Erdős–Mordell inequality for two points -/
theorem erdos_mordell_two_points
  (a b c : ℝ)
  (a₁ b₁ c₁ : ℝ)
  (a₂ b₂ c₂ : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha₁ : 0 ≤ a₁) (hb₁ : 0 ≤ b₁) (hc₁ : 0 ≤ c₁)
  (ha₂ : 0 ≤ a₂) (hb₂ : 0 ≤ b₂) (hc₂ : 0 ≤ c₂)
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  a * a₁ * a₂ + b * b₁ * b₂ + c * c₁ * c₂ ≥ a * b * c :=
by sorry

end erdos_mordell_two_points_l677_67703
