import Mathlib

namespace solution_condition_l2145_214541

theorem solution_condition (m : ℚ) : (∀ x, m * x = m ↔ x = 1) ↔ m ≠ 0 := by
  sorry

end solution_condition_l2145_214541


namespace simplify_fraction_l2145_214558

theorem simplify_fraction (x y z : ℚ) (hx : x = 5) (hz : z = 2) :
  (10 * x * y * z) / (15 * x^2 * z) = (2 * y) / 15 := by
  sorry

end simplify_fraction_l2145_214558


namespace quadratic_inequality_solution_l2145_214572

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + 3 * x + 2 > 0) ↔ (b < x ∧ x < 1)) → 
  (a = -5 ∧ b = -2/5) := by
  sorry

end quadratic_inequality_solution_l2145_214572


namespace prob_product_div_by_eight_l2145_214581

/-- The probability of rolling an odd number on a standard 6-sided die -/
def prob_odd : ℚ := 1/2

/-- The probability of rolling a 2 on a standard 6-sided die -/
def prob_two : ℚ := 1/6

/-- The probability of rolling a 4 on a standard 6-sided die -/
def prob_four : ℚ := 1/6

/-- The number of dice rolled -/
def num_dice : ℕ := 8

/-- Theorem: The probability that the product of 8 standard 6-sided dice rolls is divisible by 8 is 1651/1728 -/
theorem prob_product_div_by_eight : 
  (1 : ℚ) - (prob_odd ^ num_dice + 
    num_dice * prob_two * prob_odd ^ (num_dice - 1) + 
    (num_dice.choose 2) * prob_two^2 * prob_odd^(num_dice - 2) + 
    num_dice * prob_four * prob_odd^(num_dice - 1)) = 1651/1728 := by
  sorry

end prob_product_div_by_eight_l2145_214581


namespace star_equation_solution_l2145_214598

/-- Custom binary operation ⋆ -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

/-- Theorem stating that if 4 ⋆ x = 52, then x = 8 -/
theorem star_equation_solution (x : ℝ) (h : star 4 x = 52) : x = 8 := by
  sorry

end star_equation_solution_l2145_214598


namespace unique_n_value_l2145_214594

theorem unique_n_value : ∃! n : ℕ, 
  50 < n ∧ n < 120 ∧ 
  ∃ k : ℕ, n = 8 * k ∧
  n % 7 = 3 ∧
  n % 9 = 3 ∧
  n = 192 := by
  sorry

end unique_n_value_l2145_214594


namespace school_location_minimizes_distance_l2145_214507

/-- Represents a town with a number of students -/
structure Town where
  name : String
  students : ℕ

/-- Calculates the total distance traveled by students -/
def totalDistance (schoolLocation : Town) (townA : Town) (townB : Town) (distance : ℕ) : ℕ :=
  if schoolLocation.name = townA.name then
    townB.students * distance
  else if schoolLocation.name = townB.name then
    townA.students * distance
  else
    (townA.students + townB.students) * distance

/-- Theorem: Building a school in the town with more students minimizes total distance -/
theorem school_location_minimizes_distance (townA townB : Town) (distance : ℕ) :
  townA.students < townB.students →
  totalDistance townB townA townB distance ≤ totalDistance townA townA townB distance :=
by
  sorry

end school_location_minimizes_distance_l2145_214507


namespace heather_walk_distance_l2145_214551

theorem heather_walk_distance : 
  let car_to_entrance : Float := 0.645
  let to_carnival : Float := 1.235
  let to_animals : Float := 0.875
  let to_food : Float := 1.537
  let food_to_car : Float := 0.932
  car_to_entrance + to_carnival + to_animals + to_food + food_to_car = 5.224 := by
  sorry

end heather_walk_distance_l2145_214551


namespace y_intercept_of_line_l2145_214567

theorem y_intercept_of_line (x y : ℝ) :
  x + 2*y - 1 = 0 → x = 0 → y = 1/2 := by
  sorry

end y_intercept_of_line_l2145_214567


namespace ellipse_focal_length_l2145_214527

def ellipse_equation (x y a : ℝ) : Prop :=
  x^2 / (10 - a) + y^2 / (a - 2) = 1

theorem ellipse_focal_length (a : ℝ) : 
  (∃ x y : ℝ, ellipse_equation x y a) → 
  (∃ c : ℝ, c = 2) →
  (a = 4 ∨ a = 8) :=
by sorry

end ellipse_focal_length_l2145_214527


namespace segment_length_from_perpendicular_lines_and_midpoint_l2145_214547

/-- Given two perpendicular lines and a midpoint, prove the length of the segment. -/
theorem segment_length_from_perpendicular_lines_and_midpoint
  (A B : ℝ × ℝ) -- Points A and B
  (a : ℝ) -- Parameter in the equation of the second line
  (h1 : (2 * A.1 - A.2 = 0)) -- A is on the line 2x - y = 0
  (h2 : (B.1 + a * B.2 = 0)) -- B is on the line x + ay = 0
  (h3 : (2 : ℝ) * A.1 + (-1 : ℝ) * a = 0) -- Perpendicularity condition
  (h4 : (A.1 + B.1) / 2 = 0 ∧ (A.2 + B.2) / 2 = 10 / a) -- Midpoint condition
  : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 10 := by
  sorry

end segment_length_from_perpendicular_lines_and_midpoint_l2145_214547


namespace polynomial_simplification_l2145_214549

theorem polynomial_simplification (x : ℝ) : 
  (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1 = 32*x^5 := by
  sorry

end polynomial_simplification_l2145_214549


namespace quadratic_one_solution_sum_l2145_214591

theorem quadratic_one_solution_sum (b : ℝ) : 
  (∃! x : ℝ, 6 * x^2 + b * x + 12 * x + 18 = 0) →
  (∃ b₁ b₂ : ℝ, b = b₁ ∨ b = b₂) ∧ (b₁ + b₂ = -24) :=
by sorry

end quadratic_one_solution_sum_l2145_214591


namespace five_girls_five_boys_arrangements_l2145_214585

/-- The number of ways to arrange n girls and n boys around a circular table
    such that no two people of the same gender sit next to each other -/
def alternatingArrangements (n : ℕ) : ℕ :=
  2 * (n.factorial * n.factorial)

/-- Theorem: There are 28800 ways to arrange 5 girls and 5 boys around a circular table
    such that no two people of the same gender sit next to each other -/
theorem five_girls_five_boys_arrangements :
  alternatingArrangements 5 = 28800 := by
  sorry

end five_girls_five_boys_arrangements_l2145_214585


namespace min_values_xy_and_x_plus_y_l2145_214555

theorem min_values_xy_and_x_plus_y (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (h : 2*x + 8*y - x*y = 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → 2*a + 8*b - a*b = 0 → x*y ≤ a*b) ∧ 
  (∀ a b : ℝ, a > 0 → b > 0 → 2*a + 8*b - a*b = 0 → x + y ≤ a + b) ∧
  x*y = 64 ∧ x + y = 18 := by
sorry

end min_values_xy_and_x_plus_y_l2145_214555


namespace average_salary_all_employees_l2145_214525

theorem average_salary_all_employees
  (officer_avg_salary : ℕ)
  (non_officer_avg_salary : ℕ)
  (num_officers : ℕ)
  (num_non_officers : ℕ)
  (h1 : officer_avg_salary = 450)
  (h2 : non_officer_avg_salary = 110)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 495) :
  (officer_avg_salary * num_officers + non_officer_avg_salary * num_non_officers) / (num_officers + num_non_officers) = 120 :=
by sorry

end average_salary_all_employees_l2145_214525


namespace lucky_number_theorem_l2145_214521

/-- A "lucky number" is a three-digit positive integer that can be expressed as m(m+3) for some positive integer m. -/
def is_lucky_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m > 0 ∧ n = m * (m + 3)

/-- The largest "lucky number". -/
def largest_lucky_number : ℕ := 990

/-- The sum of all N where M and N are both "lucky numbers" and M - N = 350. -/
def sum_of_satisfying_N : ℕ := 614

theorem lucky_number_theorem :
  (∀ n : ℕ, is_lucky_number n → n ≤ largest_lucky_number) ∧
  (∀ M N : ℕ, is_lucky_number M → is_lucky_number N → M - N = 350 →
    N = 460 ∨ N = 154) ∧
  (sum_of_satisfying_N = 614) := by sorry

end lucky_number_theorem_l2145_214521


namespace two_solution_range_l2145_214517

/-- 
Given a system of equations:
  y = x^2
  y = x + m
The range of m for which the system has two distinct solutions is (-1/4, +∞).
-/
theorem two_solution_range (x y m : ℝ) : 
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ x₁^2 = x₁ + m ∧ x₂^2 = x₂ + m) ↔ m > -1/4 := by
  sorry

end two_solution_range_l2145_214517


namespace gray_area_division_l2145_214513

-- Define a rectangle
structure Rectangle where
  width : ℝ
  height : ℝ
  width_pos : width > 0
  height_pos : height > 0

-- Define a square within the rectangle
structure InternalSquare where
  side : ℝ
  x : ℝ  -- x-coordinate of the square's top-left corner
  y : ℝ  -- y-coordinate of the square's top-left corner
  side_pos : side > 0
  within_rectangle : (r : Rectangle) → x ≥ 0 ∧ y ≥ 0 ∧ x + side ≤ r.width ∧ y + side ≤ r.height

-- Define the theorem
theorem gray_area_division (r : Rectangle) (s : InternalSquare) :
  ∃ (line : ℝ → ℝ → Prop), 
    (∀ (x y : ℝ), (x ≥ 0 ∧ x ≤ r.width ∧ y ≥ 0 ∧ y ≤ r.height) →
      (¬(x ≥ s.x ∧ x ≤ s.x + s.side ∧ y ≥ s.y ∧ y ≤ s.y + s.side) →
        (line x y ∨ ¬line x y))) ∧
    (∃ (area1 area2 : ℝ), area1 = area2 ∧
      area1 + area2 = r.width * r.height - s.side * s.side) :=
by sorry

end gray_area_division_l2145_214513


namespace quadratic_inequality_solution_set_l2145_214508

/-- The solution set of the quadratic inequality (m^2-2m-3)x^2-(m-3)x-1<0 is ℝ if and only if -1/5 < m ≤ 3 -/
theorem quadratic_inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0) ↔ (-1/5 < m ∧ m ≤ 3) :=
sorry

end quadratic_inequality_solution_set_l2145_214508


namespace right_pyramid_height_l2145_214523

/-- A right pyramid with a square base -/
structure RightPyramid where
  /-- The perimeter of the square base in inches -/
  base_perimeter : ℝ
  /-- The distance from the apex to each vertex of the base in inches -/
  apex_to_vertex : ℝ

/-- The height of a right pyramid from its apex to the center of its base -/
def pyramid_height (p : RightPyramid) : ℝ :=
  sorry

theorem right_pyramid_height (p : RightPyramid) 
  (h1 : p.base_perimeter = 40)
  (h2 : p.apex_to_vertex = 12) :
  pyramid_height p = Real.sqrt 94 := by
  sorry

end right_pyramid_height_l2145_214523


namespace workshop_average_salary_l2145_214530

/-- Proves that the average salary of all workers in a workshop is 8000, given the specified conditions. -/
theorem workshop_average_salary 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_rest : ℕ) 
  (h1 : total_workers = 49)
  (h2 : num_technicians = 7)
  (h3 : avg_salary_technicians = 20000)
  (h4 : avg_salary_rest = 6000) :
  (num_technicians * avg_salary_technicians + (total_workers - num_technicians) * avg_salary_rest) / total_workers = 8000 := by
  sorry

#check workshop_average_salary

end workshop_average_salary_l2145_214530


namespace two_lucky_tickets_exist_l2145_214584

/-- A ticket number is a 6-digit integer -/
def TicketNumber := { n : ℕ // n ≥ 100000 ∧ n < 1000000 }

/-- Sum of the first three digits of a ticket number -/
def sumFirstThree (n : TicketNumber) : ℕ := 
  (n.val / 100000) + ((n.val / 10000) % 10) + ((n.val / 1000) % 10)

/-- Sum of the last three digits of a ticket number -/
def sumLastThree (n : TicketNumber) : ℕ := 
  ((n.val / 100) % 10) + ((n.val / 10) % 10) + (n.val % 10)

/-- A ticket is lucky if the sum of its first three digits equals the sum of its last three digits -/
def isLucky (n : TicketNumber) : Prop := sumFirstThree n = sumLastThree n

/-- There exist two lucky tickets among ten consecutive tickets -/
theorem two_lucky_tickets_exist : 
  ∃ (n : TicketNumber) (a b : ℕ), 0 ≤ a ∧ a < b ∧ b ≤ 9 ∧ 
    isLucky ⟨n.val + a, sorry⟩ ∧ isLucky ⟨n.val + b, sorry⟩ := by
  sorry

end two_lucky_tickets_exist_l2145_214584


namespace probability_two_fours_eight_dice_l2145_214578

theorem probability_two_fours_eight_dice : 
  let n : ℕ := 8  -- number of dice
  let k : ℕ := 2  -- number of successes (showing 4)
  let p : ℚ := 1 / 6  -- probability of rolling a 4 on a single die
  Nat.choose n k * p^k * (1 - p)^(n - k) = (28 * 15625 : ℚ) / 279936 := by
sorry

end probability_two_fours_eight_dice_l2145_214578


namespace janet_sculpture_weight_l2145_214573

/-- Given Janet's work details, prove the weight of the first sculpture -/
theorem janet_sculpture_weight
  (exterminator_rate : ℝ)
  (sculpture_rate : ℝ)
  (exterminator_hours : ℝ)
  (second_sculpture_weight : ℝ)
  (total_income : ℝ)
  (h1 : exterminator_rate = 70)
  (h2 : sculpture_rate = 20)
  (h3 : exterminator_hours = 20)
  (h4 : second_sculpture_weight = 7)
  (h5 : total_income = 1640)
  : ∃ (first_sculpture_weight : ℝ),
    first_sculpture_weight = 5 ∧
    total_income = exterminator_rate * exterminator_hours +
                   sculpture_rate * (first_sculpture_weight + second_sculpture_weight) :=
by sorry

end janet_sculpture_weight_l2145_214573


namespace multiple_count_l2145_214563

theorem multiple_count (n : ℕ) (h1 : n > 0) (h2 : n ≤ 400) : 
  (∃ (k : ℕ), k > 0 ∧ (∀ m : ℕ, m > 0 → m ≤ 400 → m % k = 0 → m ∈ Finset.range 401) ∧ 
  (Finset.filter (λ m => m % k = 0) (Finset.range 401)).card = 16) → 
  n = 25 :=
sorry

end multiple_count_l2145_214563


namespace sqrt_sum_equals_twelve_l2145_214509

theorem sqrt_sum_equals_twelve :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) + 2 = 12 := by
  sorry

end sqrt_sum_equals_twelve_l2145_214509


namespace unique_solution_for_equation_l2145_214552

theorem unique_solution_for_equation (N : ℕ+) :
  ∃! (m n : ℕ+), m + (1/2 : ℚ) * (m + n - 1) * (m + n - 2) = N := by
  sorry

end unique_solution_for_equation_l2145_214552


namespace roxanne_change_l2145_214575

/-- Calculates the change Roxanne should receive after her purchase. -/
def calculate_change : ℚ :=
  let lemonade_cost : ℚ := 2 * 2
  let sandwich_cost : ℚ := 2 * 2.5
  let watermelon_cost : ℚ := 1.25
  let chips_cost : ℚ := 1.75
  let cookie_cost : ℚ := 3 * 0.75
  let total_cost : ℚ := lemonade_cost + sandwich_cost + watermelon_cost + chips_cost + cookie_cost
  let payment : ℚ := 50
  payment - total_cost

/-- Theorem stating that Roxanne's change is $35.75. -/
theorem roxanne_change : calculate_change = 35.75 := by
  sorry

end roxanne_change_l2145_214575


namespace ellipse_parameter_sum_l2145_214515

-- Define the foci
def F₁ : ℝ × ℝ := (0, 2)
def F₂ : ℝ × ℝ := (8, 2)

-- Define the ellipse
def Ellipse : Set (ℝ × ℝ) :=
  {P | Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
       Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 12}

-- Define the ellipse equation parameters
noncomputable def h : ℝ := (F₁.1 + F₂.1) / 2
noncomputable def k : ℝ := (F₁.2 + F₂.2) / 2
noncomputable def a : ℝ := 6
noncomputable def b : ℝ := Real.sqrt (a^2 - ((F₂.1 - F₁.1) / 2)^2)

-- Theorem statement
theorem ellipse_parameter_sum :
  h + k + a + b = 12 + 2 * Real.sqrt 5 := by sorry

end ellipse_parameter_sum_l2145_214515


namespace divisible_numbers_in_range_l2145_214569

theorem divisible_numbers_in_range : ∃! n : ℕ, 
  1000 < n ∧ n < 2500 ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 8 ∣ n :=
by sorry

end divisible_numbers_in_range_l2145_214569


namespace not_prime_5n_plus_1_l2145_214533

theorem not_prime_5n_plus_1 (n : ℕ) (x y : ℕ) 
  (h1 : x^2 = 2*n + 1) (h2 : y^2 = 3*n + 1) : 
  ¬ Nat.Prime (5*n + 1) := by
sorry

end not_prime_5n_plus_1_l2145_214533


namespace club_average_age_l2145_214504

theorem club_average_age (num_females num_males num_children : ℕ)
                         (avg_age_females avg_age_males avg_age_children : ℚ) :
  num_females = 12 →
  num_males = 20 →
  num_children = 8 →
  avg_age_females = 28 →
  avg_age_males = 40 →
  avg_age_children = 10 →
  let total_sum := num_females * avg_age_females + num_males * avg_age_males + num_children * avg_age_children
  let total_people := num_females + num_males + num_children
  (total_sum / total_people : ℚ) = 30.4 := by
  sorry

end club_average_age_l2145_214504


namespace contrapositive_equivalence_l2145_214540

theorem contrapositive_equivalence (x : ℝ) :
  (x^2 < 1 → -1 ≤ x ∧ x < 1) ↔ (x < -1 ∨ x ≥ 1 → x^2 ≥ 1) := by
  sorry

end contrapositive_equivalence_l2145_214540


namespace part_one_part_two_l2145_214582

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part 1
theorem part_one : (Set.univ \ P 3) ∩ Q = {x | -2 ≤ x ∧ x < 4} := by sorry

-- Part 2
theorem part_two : {a : ℝ | P a ⊆ Q ∧ P a ≠ Q} = Set.Iic 2 := by sorry

end part_one_part_two_l2145_214582


namespace function_range_theorem_l2145_214586

open Real

theorem function_range_theorem (f : ℝ → ℝ) (a b m : ℝ) :
  (∀ x, x > 0 → f x = 2 - 1/x) →
  a < b →
  (∀ x, x ∈ Set.Ioo a b ↔ f x ∈ Set.Ioo (m*a) (m*b)) →
  m ∈ Set.Ioo 0 1 :=
sorry

end function_range_theorem_l2145_214586


namespace function_value_at_five_l2145_214531

open Real

theorem function_value_at_five
  (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x > -3 / x)
  (h2 : ∀ x > 0, f (f x + 3 / x) = 2) :
  f 5 = 7 / 5 := by
sorry

end function_value_at_five_l2145_214531


namespace difference_set_not_always_equal_l2145_214544

theorem difference_set_not_always_equal :
  ∃ (A B : Set α) (hA : A.Nonempty) (hB : B.Nonempty),
    (A \ B) ≠ (B \ A) :=
by sorry

end difference_set_not_always_equal_l2145_214544


namespace exists_xAxis_visitsAllLines_l2145_214596

/-- Represents a line in a plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Configuration of n lines in a plane -/
structure LineConfiguration where
  n : ℕ
  lines : Fin n → Line
  not_parallel : ∀ i j, i ≠ j → (lines i).slope ≠ (lines j).slope
  not_perpendicular : ∀ i j, i ≠ j → (lines i).slope * (lines j).slope ≠ -1
  not_concurrent : ∀ i j k, i ≠ j → j ≠ k → i ≠ k → 
    ¬∃ (x y : ℝ), (y = (lines i).slope * x + (lines i).intercept) ∧
                   (y = (lines j).slope * x + (lines j).intercept) ∧
                   (y = (lines k).slope * x + (lines k).intercept)

/-- A point visits all lines if it intersects with each line -/
def visitsAllLines (cfg : LineConfiguration) (xAxis : Line) : Prop :=
  ∀ i, ∃ x, xAxis.slope * x + xAxis.intercept = (cfg.lines i).slope * x + (cfg.lines i).intercept

/-- Main theorem: There exists a line that can be chosen as x-axis to visit all lines -/
theorem exists_xAxis_visitsAllLines (cfg : LineConfiguration) :
  ∃ xAxis, visitsAllLines cfg xAxis := by
  sorry

end exists_xAxis_visitsAllLines_l2145_214596


namespace area_triangle_AEF_l2145_214570

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Represents a triangle -/
structure Triangle :=
  (A B C : Point)

/-- Checks if a quadrilateral is a parallelogram -/
def isParallelogram (q : Quadrilateral) : Prop :=
  sorry

/-- Calculates the area of a quadrilateral -/
def areaQuadrilateral (q : Quadrilateral) : ℝ :=
  sorry

/-- Divides a line segment in a given ratio -/
def divideLineSegment (A B : Point) (ratio : ℚ) : Point :=
  sorry

/-- Calculates the area of a triangle -/
def areaTriangle (t : Triangle) : ℝ :=
  sorry

theorem area_triangle_AEF (ABCD : Quadrilateral) 
  (hParallelogram : isParallelogram ABCD)
  (hArea : areaQuadrilateral ABCD = 50)
  (E : Point) (hE : E = divideLineSegment ABCD.A ABCD.B (2/5))
  (F : Point) (hF : F = divideLineSegment ABCD.C ABCD.D (3/5))
  (G : Point) (hG : G = divideLineSegment ABCD.B ABCD.C (1/2)) :
  areaTriangle ⟨ABCD.A, E, F⟩ = 12 :=
sorry

end area_triangle_AEF_l2145_214570


namespace diamond_two_seven_l2145_214506

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 4 * x + 3 * y

-- Theorem statement
theorem diamond_two_seven : diamond 2 7 = 29 := by
  sorry

end diamond_two_seven_l2145_214506


namespace water_volume_in_solution_l2145_214559

/-- Calculates the volume of a component in a solution given the total volume and the component's proportion -/
def component_volume (total_volume : ℝ) (proportion : ℝ) : ℝ :=
  total_volume * proportion

theorem water_volume_in_solution (total_volume : ℝ) (water_proportion : ℝ) 
  (h1 : total_volume = 1.20)
  (h2 : water_proportion = 0.50) :
  component_volume total_volume water_proportion = 0.60 := by
  sorry

#eval component_volume 1.20 0.50

end water_volume_in_solution_l2145_214559


namespace cubes_equation_solution_l2145_214524

theorem cubes_equation_solution (x y z : ℤ) (h : x^3 + 2*y^3 = 4*z^3) : x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end cubes_equation_solution_l2145_214524


namespace brownie_pieces_count_l2145_214510

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- The pan of brownies -/
def pan : Rectangle := { length := 24, width := 20 }

/-- A single brownie piece -/
def piece : Rectangle := { length := 3, width := 4 }

/-- The number of brownie pieces that can be cut from the pan -/
def num_pieces : ℕ := (area pan) / (area piece)

theorem brownie_pieces_count : num_pieces = 40 := by
  sorry

end brownie_pieces_count_l2145_214510


namespace final_strawberry_count_l2145_214595

/-- The number of strawberry plants after n months of doubling, starting from an initial number. -/
def plants_after_months (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ months)

/-- The theorem stating the final number of strawberry plants -/
theorem final_strawberry_count :
  let initial_plants : ℕ := 3
  let months_passed : ℕ := 3
  let plants_given_away : ℕ := 4
  (plants_after_months initial_plants months_passed) - plants_given_away = 20 :=
by sorry

end final_strawberry_count_l2145_214595


namespace min_value_theorem_l2145_214568

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 3*x*y) :
  2*x + y ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 3*x₀*y₀ ∧ 2*x₀ + y₀ = 3 := by
  sorry

end min_value_theorem_l2145_214568


namespace area_of_triangle_fpg_l2145_214554

/-- Given a trapezoid EFGH with bases EF and GH, and point P at the intersection of diagonals,
    this theorem states that the area of triangle FPG is 28.125 square units. -/
theorem area_of_triangle_fpg (EF GH : ℝ) (area_EFGH : ℝ) :
  EF = 15 →
  GH = 25 →
  area_EFGH = 200 →
  ∃ (area_FPG : ℝ), area_FPG = 28.125 := by
  sorry

end area_of_triangle_fpg_l2145_214554


namespace base_conversion_problem_l2145_214579

theorem base_conversion_problem : ∃! (b : ℕ), b > 1 ∧ b ^ 3 ≤ 216 ∧ 216 < b ^ 4 :=
by
  -- The proof goes here
  sorry

end base_conversion_problem_l2145_214579


namespace modular_arithmetic_problem_l2145_214518

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), 
    (9 * a) % 35 = 1 ∧ 
    (7 * b) % 35 = 1 ∧ 
    (7 * a + 3 * b) % 35 = 8 := by
  sorry

end modular_arithmetic_problem_l2145_214518


namespace bruce_remaining_eggs_l2145_214534

def bruce_initial_eggs : ℕ := 75
def eggs_lost : ℕ := 70

theorem bruce_remaining_eggs :
  bruce_initial_eggs - eggs_lost = 5 :=
by sorry

end bruce_remaining_eggs_l2145_214534


namespace slope_condition_implies_m_zero_l2145_214528

theorem slope_condition_implies_m_zero (m : ℝ) : 
  (4 - m^2) / (m - (-2)) = 2 → m = 0 := by
  sorry

end slope_condition_implies_m_zero_l2145_214528


namespace max_value_of_f_l2145_214545

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 + 6*x + 13)

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc (-2 : ℝ) 2 ∧ 
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ≤ f c) ∧
  f c = 1/4 := by
sorry

end max_value_of_f_l2145_214545


namespace complex_modulus_problem_l2145_214526

theorem complex_modulus_problem (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
sorry

end complex_modulus_problem_l2145_214526


namespace festival_ferry_total_l2145_214587

/-- Represents the ferry schedule and passenger count --/
structure FerrySchedule where
  startTime : Nat  -- Start time in minutes after midnight
  endTime : Nat    -- End time in minutes after midnight
  interval : Nat   -- Interval between trips in minutes
  initialPassengers : Nat  -- Number of passengers on the first trip
  passengerDecrease : Nat  -- Decrease in passengers per trip

/-- Calculates the total number of people ferried --/
def totalPeopleFerried (schedule : FerrySchedule) : Nat :=
  let numTrips := (schedule.endTime - schedule.startTime) / schedule.interval + 1
  let lastTripPassengers := schedule.initialPassengers - (numTrips - 1) * schedule.passengerDecrease
  (numTrips * (schedule.initialPassengers + lastTripPassengers)) / 2

/-- The ferry schedule for the festival --/
def festivalFerry : FerrySchedule :=
  { startTime := 9 * 60  -- 9 AM in minutes
    endTime := 16 * 60   -- 4 PM in minutes
    interval := 30
    initialPassengers := 120
    passengerDecrease := 2 }

/-- Theorem stating the total number of people ferried to the festival --/
theorem festival_ferry_total : totalPeopleFerried festivalFerry = 1590 := by
  sorry


end festival_ferry_total_l2145_214587


namespace samara_detailing_cost_samara_detailing_cost_proof_l2145_214503

/-- Proves that Samara's spending on detailing equals $79 given the problem conditions -/
theorem samara_detailing_cost : ℕ → Prop :=
  fun (detailing_cost : ℕ) =>
    let alberto_total : ℕ := 2457
    let samara_oil : ℕ := 25
    let samara_tires : ℕ := 467
    let difference : ℕ := 1886
    alberto_total = samara_oil + samara_tires + detailing_cost + difference →
    detailing_cost = 79

/-- The proof of the theorem -/
theorem samara_detailing_cost_proof : samara_detailing_cost 79 := by
  sorry

end samara_detailing_cost_samara_detailing_cost_proof_l2145_214503


namespace largest_B_divisible_by_nine_l2145_214516

def number (B : Nat) : Nat := 5 * 100000 + B * 10000 + 4 * 1000 + 8 * 100 + 6 * 10 + 1

theorem largest_B_divisible_by_nine :
  ∀ B : Nat, B < 10 →
    (∃ m : Nat, number B = 9 * m) →
    B ≤ 9 ∧
    (∀ C : Nat, C < 10 → C > B → ¬∃ n : Nat, number C = 9 * n) :=
by sorry

end largest_B_divisible_by_nine_l2145_214516


namespace woodys_weekly_allowance_l2145_214501

/-- Woody's weekly allowance problem -/
theorem woodys_weekly_allowance 
  (console_cost : ℕ) 
  (initial_savings : ℕ) 
  (weeks_to_save : ℕ) 
  (h1 : console_cost = 282)
  (h2 : initial_savings = 42)
  (h3 : weeks_to_save = 10) :
  (console_cost - initial_savings) / weeks_to_save = 24 := by
  sorry

end woodys_weekly_allowance_l2145_214501


namespace opposite_of_negative_abs_two_fifths_l2145_214564

theorem opposite_of_negative_abs_two_fifths :
  -(- |2 / 5|) = 2 / 5 := by
  sorry

end opposite_of_negative_abs_two_fifths_l2145_214564


namespace berry_multiple_l2145_214532

/-- Given the number of berries for Skylar, Steve, and Stacy, and their relationships,
    prove that the multiple of Steve's berries that Stacy has 2 more than is 3. -/
theorem berry_multiple (skylar_berries : ℕ) (steve_berries : ℕ) (stacy_berries : ℕ) 
    (h1 : skylar_berries = 20)
    (h2 : steve_berries = skylar_berries / 2)
    (h3 : stacy_berries = 32)
    (h4 : ∃ m : ℕ, stacy_berries = m * steve_berries + 2) :
  ∃ m : ℕ, m = 3 ∧ stacy_berries = m * steve_berries + 2 :=
by sorry

end berry_multiple_l2145_214532


namespace cow_herd_distribution_l2145_214514

theorem cow_herd_distribution (total : ℕ) : 
  (total : ℚ) / 3 + (total : ℚ) / 6 + (total : ℚ) / 8 + 9 = total → total = 216 := by
  sorry

end cow_herd_distribution_l2145_214514


namespace toms_initial_money_l2145_214553

theorem toms_initial_money (current_money : ℕ) (weekend_earnings : ℕ) (initial_money : ℕ) :
  current_money = 86 →
  weekend_earnings = 12 →
  current_money = initial_money + weekend_earnings →
  initial_money = 74 :=
by sorry

end toms_initial_money_l2145_214553


namespace youngest_not_first_or_last_l2145_214535

def number_of_people : ℕ := 5

-- Define a function to calculate the number of permutations
def permutations (n : ℕ) : ℕ := Nat.factorial n

-- Define a function to calculate the number of valid arrangements
def valid_arrangements (n : ℕ) : ℕ :=
  permutations n - 2 * permutations (n - 1)

-- Theorem statement
theorem youngest_not_first_or_last :
  valid_arrangements number_of_people = 72 := by
  sorry

end youngest_not_first_or_last_l2145_214535


namespace sum_of_segments_is_224_l2145_214529

/-- Given seven points A, B, C, D, E, F, G on a line in that order, 
    this function calculates the sum of lengths of all segments with endpoints at these points. -/
def sumOfSegments (AG BF CE : ℝ) : ℝ :=
  6 * AG + 4 * BF + 2 * CE

/-- Theorem stating that for the given conditions, the sum of all segment lengths is 224 cm. -/
theorem sum_of_segments_is_224 (AG BF CE : ℝ) 
  (h1 : AG = 23) (h2 : BF = 17) (h3 : CE = 9) : 
  sumOfSegments AG BF CE = 224 := by
  sorry

#eval sumOfSegments 23 17 9

end sum_of_segments_is_224_l2145_214529


namespace tobias_driveways_l2145_214538

/-- The number of driveways Tobias shoveled -/
def num_driveways : ℕ :=
  let shoe_cost : ℕ := 95
  let months_saved : ℕ := 3
  let monthly_allowance : ℕ := 5
  let lawn_mowing_fee : ℕ := 15
  let driveway_shoveling_fee : ℕ := 7
  let change_after_purchase : ℕ := 15
  let lawns_mowed : ℕ := 4
  let total_money : ℕ := shoe_cost + change_after_purchase
  let money_from_allowance : ℕ := months_saved * monthly_allowance
  let money_from_mowing : ℕ := lawns_mowed * lawn_mowing_fee
  let money_from_shoveling : ℕ := total_money - money_from_allowance - money_from_mowing
  money_from_shoveling / driveway_shoveling_fee

theorem tobias_driveways : num_driveways = 2 := by
  sorry

end tobias_driveways_l2145_214538


namespace quadratic_constant_term_l2145_214550

/-- If a quadratic equation with real coefficients has 5 + 3i as a root, then its constant term is 34 -/
theorem quadratic_constant_term (b c : ℝ) : 
  (∃ x : ℂ, x^2 + b*x + c = 0 ∧ x = 5 + 3*Complex.I) →
  c = 34 := by sorry

end quadratic_constant_term_l2145_214550


namespace volleyball_tournament_max_wins_l2145_214577

theorem volleyball_tournament_max_wins (n : ℕ) : 
  let european_teams := n + 9
  let total_matches := n.choose 2 + european_teams.choose 2 + n * european_teams
  let european_wins := 9 * (total_matches - european_teams.choose 2)
  ∃ (k : ℕ), 
    k ≤ n * european_teams ∧ 
    european_teams.choose 2 + k = european_wins ∧
    (∀ m : ℕ, m ≤ n → m - 1 + min n european_teams ≤ 11) ∧
    (∃ m : ℕ, m ≤ n ∧ m - 1 + min n european_teams = 11) :=
by sorry

end volleyball_tournament_max_wins_l2145_214577


namespace remainder_2468135790_div_101_l2145_214557

theorem remainder_2468135790_div_101 : 
  2468135790 % 101 = 50 := by sorry

end remainder_2468135790_div_101_l2145_214557


namespace intersection_of_given_lines_l2145_214560

/-- The intersection point of two lines in 2D space -/
def intersection_point (line1_start : ℝ × ℝ) (line1_dir : ℝ × ℝ) (line2_start : ℝ × ℝ) (line2_dir : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem intersection_of_given_lines :
  let line1_start : ℝ × ℝ := (2, -3)
  let line1_dir : ℝ × ℝ := (3, 4)
  let line2_start : ℝ × ℝ := (-1, 4)
  let line2_dir : ℝ × ℝ := (5, -1)
  intersection_point line1_start line1_dir line2_start line2_dir = (124/5, 137/5) :=
by sorry

end intersection_of_given_lines_l2145_214560


namespace sum_of_reciprocals_l2145_214536

theorem sum_of_reciprocals (a b : ℝ) 
  (ha : a^2 + 2*a = 2) 
  (hb : b^2 + 2*b = 2) : 
  (1/a + 1/b = 1) ∨ 
  (1/a + 1/b = Real.sqrt 3 + 1) ∨ 
  (1/a + 1/b = -Real.sqrt 3 + 1) := by
  sorry

end sum_of_reciprocals_l2145_214536


namespace nested_sqrt_solution_l2145_214589

theorem nested_sqrt_solution :
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
sorry

end nested_sqrt_solution_l2145_214589


namespace square_floor_tiles_l2145_214505

theorem square_floor_tiles (n : ℕ) : 
  (2 * n - 1 = 37) → n^2 = 361 := by
  sorry

end square_floor_tiles_l2145_214505


namespace complex_modulus_equation_l2145_214565

theorem complex_modulus_equation (n : ℝ) : 
  Complex.abs (6 + n * Complex.I) = 6 * Real.sqrt 5 → n = 12 := by
  sorry

end complex_modulus_equation_l2145_214565


namespace cube_sum_and_reciprocal_l2145_214542

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = 10) : x^3 + 1/x^3 = 970 := by
  sorry

end cube_sum_and_reciprocal_l2145_214542


namespace percentage_runs_by_running_l2145_214537

def total_runs : ℕ := 120
def boundaries : ℕ := 5
def sixes : ℕ := 5
def runs_per_boundary : ℕ := 4
def runs_per_six : ℕ := 6

theorem percentage_runs_by_running (total_runs boundaries sixes runs_per_boundary runs_per_six : ℕ) 
  (h1 : total_runs = 120)
  (h2 : boundaries = 5)
  (h3 : sixes = 5)
  (h4 : runs_per_boundary = 4)
  (h5 : runs_per_six = 6) :
  (total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs * 100 = 
  (120 - (5 * 4 + 5 * 6)) / 120 * 100 := by sorry

end percentage_runs_by_running_l2145_214537


namespace quadratic_form_h_value_l2145_214556

theorem quadratic_form_h_value :
  ∃ (a k : ℝ), ∀ x, 3 * x^2 + 9 * x + 20 = a * (x - (-3/2))^2 + k :=
by sorry

end quadratic_form_h_value_l2145_214556


namespace c_range_l2145_214561

-- Define the functions
def f (c : ℝ) (x : ℝ) := x^2 - 2*c*x + 1

-- State the theorem
theorem c_range (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) :
  (((∀ x y : ℝ, x < y → c^x > c^y) ∨
    (∀ x y : ℝ, x > y → x > 1/2 → y > 1/2 → f c x > f c y)) ∧
   ¬((∀ x y : ℝ, x < y → c^x > c^y) ∧
     (∀ x y : ℝ, x > y → x > 1/2 → y > 1/2 → f c x > f c y))) →
  (1/2 < c ∧ c < 1) :=
by sorry

end c_range_l2145_214561


namespace joes_fast_food_cost_l2145_214588

/-- Calculates the total cost of a meal at Joe's Fast Food --/
def total_cost (sandwich_price : ℚ) (soda_price : ℚ) (fries_price : ℚ) 
                (sandwich_qty : ℕ) (soda_qty : ℕ) (fries_qty : ℕ) 
                (discount : ℚ) : ℚ :=
  sandwich_price * sandwich_qty + soda_price * soda_qty + fries_price * fries_qty - discount

/-- Theorem stating the total cost of the specified meal --/
theorem joes_fast_food_cost : 
  total_cost 4 (3/2) (5/2) 4 6 3 5 = 55/2 := by
  sorry

end joes_fast_food_cost_l2145_214588


namespace collinearity_condition_acute_angle_condition_l2145_214571

-- Define the vectors
def OA : ℝ × ℝ := (3, -4)
def OB : ℝ × ℝ := (6, -3)
def OC (m : ℝ) : ℝ × ℝ := (5 - m, -3 - m)

-- Define collinearity condition
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C.1 - A.1 = t * (B.1 - A.1) ∧ C.2 - A.2 = t * (B.2 - A.2)

-- Define acute angle condition
def acute_angle (A B C : ℝ × ℝ) : Prop :=
  let BA := (A.1 - B.1, A.2 - B.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  BA.1 * BC.1 + BA.2 * BC.2 > 0

-- Theorem 1: Collinearity condition
theorem collinearity_condition :
  ∀ m : ℝ, collinear OA OB (OC m) ↔ m = 1/2 := sorry

-- Theorem 2: Acute angle condition
theorem acute_angle_condition :
  ∀ m : ℝ, acute_angle OA OB (OC m) ↔ m ∈ Set.Ioo (-3/4 : ℝ) (1/2) ∪ Set.Ioi (1/2) := sorry

end collinearity_condition_acute_angle_condition_l2145_214571


namespace k_range_proof_l2145_214539

theorem k_range_proof (k : ℝ) : 
  (∀ x, x > k → 3 / (x + 1) < 1) ∧ 
  (∃ x, 3 / (x + 1) < 1 ∧ x ≤ k) ↔ 
  k ∈ Set.Ici 2 := by
sorry

end k_range_proof_l2145_214539


namespace square_divisible_into_2020_elegant_triangles_l2145_214511

/-- An elegant triangle is a right-angled triangle where one leg is 10 times longer than the other. -/
def ElegantTriangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (a = 10*b ∨ b = 10*a)

/-- A square can be divided into n identical elegant triangles. -/
def SquareDivisibleIntoElegantTriangles (n : ℕ) : Prop :=
  ∃ (s a b c : ℝ), s > 0 ∧ ElegantTriangle a b c ∧ 
    (n : ℝ) * (1/2 * a * b) = s^2

theorem square_divisible_into_2020_elegant_triangles :
  SquareDivisibleIntoElegantTriangles 2020 := by
  sorry


end square_divisible_into_2020_elegant_triangles_l2145_214511


namespace card_position_unique_card_position_valid_l2145_214519

/-- Represents a position in a 6x6 grid -/
structure Position where
  row : Fin 6
  col : Fin 6

/-- Represents the magician's trick setup -/
structure MagicTrick where
  initialColumn : Fin 6
  finalColumn : Fin 6

/-- Given the initial and final column numbers, determines the unique position of the card in the final layout -/
def findCardPosition (trick : MagicTrick) : Position :=
  { row := trick.initialColumn
  , col := trick.finalColumn }

/-- Theorem stating that the card position can be uniquely determined -/
theorem card_position_unique (trick : MagicTrick) :
  ∃! pos : Position, pos = findCardPosition trick :=
sorry

/-- Theorem stating that the determined position is valid within the 6x6 grid -/
theorem card_position_valid (trick : MagicTrick) :
  let pos := findCardPosition trick
  pos.row < 6 ∧ pos.col < 6 :=
sorry

end card_position_unique_card_position_valid_l2145_214519


namespace value_of_expression_l2145_214599

theorem value_of_expression (x : ℝ) (h : x = 5) : (3*x + 4)^2 = 361 := by
  sorry

end value_of_expression_l2145_214599


namespace divisibility_property_l2145_214502

def sequence_a : ℕ → ℕ
  | 0 => 3
  | n + 1 => (2 * (n + 2) * sequence_a n - (n + 1) - 2) / (n + 1)

theorem divisibility_property (p : ℕ) (hp : Nat.Prime p) (hp_odd : Odd p) :
  ∃ m : ℕ, p ∣ sequence_a m ∧ p ∣ sequence_a (m + 1) :=
by sorry

end divisibility_property_l2145_214502


namespace factorial_difference_l2145_214583

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end factorial_difference_l2145_214583


namespace total_questions_to_review_l2145_214548

-- Define the given conditions
def num_classes : ℕ := 5
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10

-- State the theorem
theorem total_questions_to_review :
  num_classes * students_per_class * questions_per_exam = 1750 := by
  sorry

end total_questions_to_review_l2145_214548


namespace rectangle_area_increase_l2145_214546

theorem rectangle_area_increase (l w : ℝ) (h1 : l > 0) (h2 : w > 0) : 
  let original_area := l * w
  let new_area := (2 * l) * (2 * w)
  (new_area - original_area) / original_area = 3 := by
sorry

end rectangle_area_increase_l2145_214546


namespace subtraction_of_fractions_l2145_214512

theorem subtraction_of_fractions : (5 : ℚ) / 6 - (1 : ℚ) / 12 = (3 : ℚ) / 4 := by
  sorry

end subtraction_of_fractions_l2145_214512


namespace segment_ratio_l2145_214597

/-- Given a line segment GH with points E and F on it, where GE is 3 times EH and GF is 5 times FH,
    prove that EF is 1/12 of GH. -/
theorem segment_ratio (G E F H : ℝ) (h1 : G ≤ E) (h2 : E ≤ F) (h3 : F ≤ H)
  (h4 : E - G = 3 * (H - E)) (h5 : F - G = 5 * (H - F)) :
  (F - E) / (H - G) = 1 / 12 := by sorry

end segment_ratio_l2145_214597


namespace chicken_egg_production_l2145_214522

theorem chicken_egg_production (num_chickens : ℕ) (total_eggs : ℕ) (num_days : ℕ) 
  (h1 : num_chickens = 4)
  (h2 : total_eggs = 36)
  (h3 : num_days = 3) :
  total_eggs / (num_chickens * num_days) = 3 := by
  sorry

end chicken_egg_production_l2145_214522


namespace triangles_from_circle_points_l2145_214574

def points_on_circle : ℕ := 10

theorem triangles_from_circle_points :
  Nat.choose points_on_circle 3 = 120 :=
by sorry

end triangles_from_circle_points_l2145_214574


namespace callie_summer_frogs_count_l2145_214566

def alster_frogs : ℚ := 2

def quinn_frogs (alster_frogs : ℚ) : ℚ := 2 * alster_frogs

def bret_frogs (quinn_frogs : ℚ) : ℚ := 3 * quinn_frogs

def callie_summer_frogs (bret_frogs : ℚ) : ℚ := (5/8) * bret_frogs

theorem callie_summer_frogs_count :
  callie_summer_frogs (bret_frogs (quinn_frogs alster_frogs)) = 7.5 := by
  sorry

end callie_summer_frogs_count_l2145_214566


namespace solutions_count_l2145_214593

/-- The number of different integer solutions (x, y) for |x|+|y|=n -/
def num_solutions (n : ℕ) : ℕ := 4 * n

theorem solutions_count :
  (num_solutions 1 = 4) ∧
  (num_solutions 2 = 8) ∧
  (num_solutions 3 = 12) →
  ∀ n : ℕ, num_solutions n = 4 * n :=
by sorry

end solutions_count_l2145_214593


namespace function_with_period_two_is_even_l2145_214543

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def smallest_positive_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  p > 0 ∧ is_periodic f p ∧ ∀ q, 0 < q ∧ q < p → ¬ is_periodic f q

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_with_period_two_is_even
  (f : ℝ → ℝ)
  (h_period : smallest_positive_period f 2)
  (h_symmetry : ∀ x, f (x + 2) = f (2 - x)) :
  is_even f :=
sorry

end function_with_period_two_is_even_l2145_214543


namespace complex_power_magnitude_l2145_214592

theorem complex_power_magnitude : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by sorry

end complex_power_magnitude_l2145_214592


namespace chess_team_shirt_numbers_l2145_214590

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 
  10 ≤ n ∧ n ≤ 99

theorem chess_team_shirt_numbers 
  (d e f : ℕ) 
  (h1 : isPrime d ∧ isPrime e ∧ isPrime f)
  (h2 : isTwoDigit d ∧ isTwoDigit e ∧ isTwoDigit f)
  (h3 : d ≠ e ∧ d ≠ f ∧ e ≠ f)
  (h4 : e + f = 36)
  (h5 : d + e = 30)
  (h6 : d + f = 32) :
  f = 19 := by
sorry

end chess_team_shirt_numbers_l2145_214590


namespace angle_A_value_triangle_area_l2145_214580

namespace TriangleABC

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition a * sin(B) = √3 * b * cos(A) -/
def condition1 (t : Triangle) : Prop :=
  t.a * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.A

/-- The conditions a = 3 and b = 2c -/
def condition2 (t : Triangle) : Prop :=
  t.a = 3 ∧ t.b = 2 * t.c

/-- The theorem stating that if condition1 holds, then A = π/3 -/
theorem angle_A_value (t : Triangle) (h : condition1 t) : t.A = Real.pi / 3 := by
  sorry

/-- The theorem stating that if condition1 and condition2 hold, then the area of the triangle is (3√3)/2 -/
theorem triangle_area (t : Triangle) (h1 : condition1 t) (h2 : condition2 t) : 
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

end TriangleABC

end angle_A_value_triangle_area_l2145_214580


namespace composition_result_l2145_214562

/-- Given two functions f and g, prove that f(g(f(3))) = 119 -/
theorem composition_result :
  let f (x : ℝ) := 2 * x + 5
  let g (x : ℝ) := 5 * x + 2
  f (g (f 3)) = 119 := by sorry

end composition_result_l2145_214562


namespace train_length_l2145_214500

/-- The length of a train given its speed and the time it takes to pass a bridge -/
theorem train_length (bridge_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) :
  bridge_length = 180 →
  train_speed_kmh = 65 →
  passing_time = 21.04615384615385 →
  ∃ train_length : ℝ, abs (train_length - 200) < 0.00001 :=
by
  sorry

end train_length_l2145_214500


namespace blue_crayons_count_l2145_214520

theorem blue_crayons_count (blue : ℕ) (red : ℕ) : 
  red = 4 * blue →  -- Condition 1: Red crayons are four times blue crayons
  blue > 0 →        -- Condition 2: There is at least one blue crayon
  blue + red = 15 → -- Condition 3: Total number of crayons is 15
  blue = 3 :=        -- Conclusion: Number of blue crayons is 3
by
  sorry

end blue_crayons_count_l2145_214520


namespace units_digit_of_expression_l2145_214576

-- Define a function to get the units digit of a number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the expression we're interested in
def expression : ℕ := 24^3 + 42^3

-- Theorem statement
theorem units_digit_of_expression :
  unitsDigit expression = 6 := by
  sorry


end units_digit_of_expression_l2145_214576
