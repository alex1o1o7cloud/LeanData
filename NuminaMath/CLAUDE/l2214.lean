import Mathlib

namespace missing_number_equation_l2214_221455

theorem missing_number_equation (x : ℤ) : 10111 - 10 * 2 * x = 10011 ↔ x = 5 := by
  sorry

end missing_number_equation_l2214_221455


namespace min_value_of_function_l2214_221437

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x^2 + 6*x + 36/x^2 ≥ 31 ∧
  (x^2 + 6*x + 36/x^2 = 31 ↔ x = 3) :=
by sorry

end min_value_of_function_l2214_221437


namespace angle_symmetry_l2214_221491

theorem angle_symmetry (α β : Real) :
  (∃ k : ℤ, α + β = 2 * k * Real.pi) →
  ∃ k : ℤ, α = 2 * k * Real.pi - β :=
by sorry

end angle_symmetry_l2214_221491


namespace violets_family_ticket_cost_l2214_221431

/-- Calculates the total cost of tickets for a family visit to the aquarium. -/
def total_ticket_cost (adult_price child_price : ℕ) (num_adults num_children : ℕ) : ℕ :=
  adult_price * num_adults + child_price * num_children

/-- Proves that the total cost for Violet's family to buy separate tickets is $155. -/
theorem violets_family_ticket_cost :
  total_ticket_cost 35 20 1 6 = 155 := by
  sorry

#eval total_ticket_cost 35 20 1 6

end violets_family_ticket_cost_l2214_221431


namespace A_intersect_B_eq_open_interval_l2214_221403

-- Define sets A and B
def A : Set ℝ := {x | Real.sqrt (x - 1) < Real.sqrt 2}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem A_intersect_B_eq_open_interval : A ∩ B = Set.Ioo 2 3 := by
  sorry

end A_intersect_B_eq_open_interval_l2214_221403


namespace max_salary_is_260000_l2214_221489

/-- Represents the maximum possible salary for a single player on a minor league soccer team -/
def max_player_salary (n : ℕ) (min_salary : ℕ) (total_cap : ℕ) : ℕ :=
  total_cap - (n - 1) * min_salary

/-- Theorem stating the maximum possible salary for a single player on the team -/
theorem max_salary_is_260000 :
  max_player_salary 18 20000 600000 = 260000 := by
  sorry

#eval max_player_salary 18 20000 600000

end max_salary_is_260000_l2214_221489


namespace range_of_a_l2214_221473

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a ≥ 0) → a ≥ -8 := by
  sorry

end range_of_a_l2214_221473


namespace quadratic_root_k_value_l2214_221466

theorem quadratic_root_k_value : ∀ k : ℝ, 
  ((-1 : ℝ)^2 + 3*(-1) + k = 0) → k = 2 := by
  sorry

end quadratic_root_k_value_l2214_221466


namespace class_size_problem_l2214_221463

/-- Given a class where:
    - The average mark of all students is 80
    - If 5 students with an average mark of 60 are excluded, the remaining students' average is 90
    Prove that the total number of students is 15 -/
theorem class_size_problem (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 60) 
  (h3 : remaining_average = 90) (h4 : excluded_count = 5) : 
  ∃ (N : ℕ), N = 15 ∧ 
  N * total_average = (N - excluded_count) * remaining_average + excluded_count * excluded_average :=
sorry

end class_size_problem_l2214_221463


namespace range_of_m_l2214_221493

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x ≥ 2}
def B (m : ℝ) : Set ℝ := {x : ℝ | x ≥ m}

-- State the theorem
theorem range_of_m (m : ℝ) (h : A ∪ B m = A) : m ≥ 2 := by
  sorry

end range_of_m_l2214_221493


namespace minimum_opponents_l2214_221441

/-- 
Given two integers h ≥ 1 and p ≥ 2, this theorem states that the minimum number of 
pairs of opponents in an hp-member parliament, such that in every partition into h 
houses of p members each, some house contains at least one pair of opponents, 
is equal to min((h-1)p + 1, (h+1 choose 2)).
-/
theorem minimum_opponents (h p : ℕ) (h_ge_one : h ≥ 1) (p_ge_two : p ≥ 2) :
  let parliament_size := h * p
  let min_opponents := min ((h - 1) * p + 1) (Nat.choose (h + 1) 2)
  ∀ (opponents : Finset (Finset (Fin parliament_size))),
    (∀ partition : Finset (Finset (Fin parliament_size)),
      (partition.card = h ∧ 
       ∀ house ∈ partition, house.card = p ∧
       partition.sup id = Finset.univ) →
      ∃ house ∈ partition, ∃ pair ∈ opponents, pair ⊆ house) →
    opponents.card ≥ min_opponents ∧
    ∃ opponents_min : Finset (Finset (Fin parliament_size)),
      opponents_min.card = min_opponents ∧
      (∀ partition : Finset (Finset (Fin parliament_size)),
        (partition.card = h ∧ 
         ∀ house ∈ partition, house.card = p ∧
         partition.sup id = Finset.univ) →
        ∃ house ∈ partition, ∃ pair ∈ opponents_min, pair ⊆ house) :=
by sorry

end minimum_opponents_l2214_221441


namespace kylie_jewelry_beads_l2214_221414

/-- The number of beaded necklaces Kylie makes on Monday -/
def monday_necklaces : ℕ := 10

/-- The number of beaded necklaces Kylie makes on Tuesday -/
def tuesday_necklaces : ℕ := 2

/-- The number of beaded bracelets Kylie makes on Wednesday -/
def wednesday_bracelets : ℕ := 5

/-- The number of beaded earrings Kylie makes on Wednesday -/
def wednesday_earrings : ℕ := 7

/-- The number of beads needed to make one beaded necklace -/
def beads_per_necklace : ℕ := 20

/-- The number of beads needed to make one beaded bracelet -/
def beads_per_bracelet : ℕ := 10

/-- The number of beads needed to make one beaded earring -/
def beads_per_earring : ℕ := 5

/-- The total number of beads Kylie uses to make her jewelry -/
def total_beads : ℕ := 
  (monday_necklaces + tuesday_necklaces) * beads_per_necklace +
  wednesday_bracelets * beads_per_bracelet +
  wednesday_earrings * beads_per_earring

theorem kylie_jewelry_beads : total_beads = 325 := by
  sorry

end kylie_jewelry_beads_l2214_221414


namespace students_per_van_l2214_221496

/-- Given five coaster vans transporting 60 boys and 80 girls, prove that each van carries 28 students. -/
theorem students_per_van (num_vans : ℕ) (num_boys : ℕ) (num_girls : ℕ) 
  (h1 : num_vans = 5)
  (h2 : num_boys = 60)
  (h3 : num_girls = 80) :
  (num_boys + num_girls) / num_vans = 28 := by
  sorry

end students_per_van_l2214_221496


namespace min_value_theorem_l2214_221494

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) :
  ∃ (min : ℝ), min = 4 ∧ ∀ z, z = 1/x + 1/(3*y) → z ≥ min :=
sorry

end min_value_theorem_l2214_221494


namespace dog_walking_problem_l2214_221422

/-- Greg's dog walking business problem -/
theorem dog_walking_problem (x : ℕ) : 
  (20 + x) +                 -- Cost for one dog
  (2 * 20 + 2 * 7 * 1) +     -- Cost for two dogs for 7 minutes
  (3 * 20 + 3 * 9 * 1) = 171 -- Cost for three dogs for 9 minutes
  → x = 10 := by
  sorry

end dog_walking_problem_l2214_221422


namespace inequality_preserved_by_exponential_l2214_221402

theorem inequality_preserved_by_exponential (a b : ℝ) (h : a > b) :
  ∀ x : ℝ, a * (2 : ℝ)^x > b * (2 : ℝ)^x :=
by
  sorry

end inequality_preserved_by_exponential_l2214_221402


namespace line_intercepts_l2214_221445

/-- Given a line with equation x/4 - y/3 = 1, prove that its x-intercept is 4 and y-intercept is -3 -/
theorem line_intercepts :
  let line := (fun (x y : ℝ) => x/4 - y/3 = 1)
  (∃ x : ℝ, line x 0 ∧ x = 4) ∧
  (∃ y : ℝ, line 0 y ∧ y = -3) := by
sorry

end line_intercepts_l2214_221445


namespace restaurant_meal_cost_l2214_221426

theorem restaurant_meal_cost 
  (total_people : ℕ) 
  (num_kids : ℕ) 
  (total_cost : ℕ) 
  (h1 : total_people = 11) 
  (h2 : num_kids = 2) 
  (h3 : total_cost = 72) :
  (total_cost : ℚ) / (total_people - num_kids : ℚ) = 8 := by
  sorry

end restaurant_meal_cost_l2214_221426


namespace parabola_focus_l2214_221435

-- Define the parabola
def parabola (x y : ℝ) : Prop := 2 * x^2 = -y

-- Define the focus of a parabola
def focus (f : ℝ × ℝ) (p : (ℝ → ℝ → Prop)) : Prop :=
  ∀ x y, p x y → (x - f.1)^2 = 4 * f.2 * (y - f.2)

-- Theorem statement
theorem parabola_focus :
  focus (0, -1/8) parabola :=
sorry

end parabola_focus_l2214_221435


namespace kim_status_update_time_l2214_221458

/-- Kim's morning routine -/
def morning_routine (coffee_time : ℕ) (payroll_time : ℕ) (num_employees : ℕ) (total_time : ℕ) (status_time : ℕ) : Prop :=
  coffee_time + num_employees * status_time + num_employees * payroll_time = total_time

/-- Theorem: Kim spends 2 minutes per employee getting a status update -/
theorem kim_status_update_time :
  ∃ (status_time : ℕ),
    morning_routine 5 3 9 50 status_time ∧
    status_time = 2 := by
  sorry

end kim_status_update_time_l2214_221458


namespace geometric_progression_ratio_l2214_221497

theorem geometric_progression_ratio (b₁ q : ℕ) (h_sum : b₁ * q^2 + b₁ * q^4 + b₁ * q^6 = 7371 * 2^2016) :
  q = 1 ∨ q = 2 ∨ q = 3 ∨ q = 4 := by
  sorry

end geometric_progression_ratio_l2214_221497


namespace thousandth_special_number_l2214_221478

/-- A function that returns true if n is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that returns true if n is a perfect cube --/
def isPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m * m = n

/-- The sequence of positive integers that are neither perfect squares nor perfect cubes --/
def specialSequence : ℕ → ℕ :=
  fun n => sorry

theorem thousandth_special_number :
  specialSequence 1000 = 1039 := by sorry

end thousandth_special_number_l2214_221478


namespace profit_percent_l2214_221432

/-- Given an article with cost price C and selling price P, 
    where selling at 2/3 of P results in a 14% loss,
    prove that selling at P results in a 29% profit -/
theorem profit_percent (C P : ℝ) (h : (2/3) * P = 0.86 * C) :
  (P - C) / C * 100 = 29 := by
  sorry

end profit_percent_l2214_221432


namespace eraser_ratio_l2214_221452

/-- The number of erasers each person has -/
structure EraserCounts where
  hanna : ℕ
  rachel : ℕ
  tanya : ℕ
  tanya_red : ℕ

/-- The conditions of the problem -/
def problem_conditions (counts : EraserCounts) : Prop :=
  counts.hanna = 2 * counts.rachel ∧
  counts.rachel = counts.tanya_red - 3 ∧
  counts.tanya = 20 ∧
  counts.tanya_red = counts.tanya / 2 ∧
  counts.hanna = 4

/-- The theorem to be proved -/
theorem eraser_ratio (counts : EraserCounts) 
  (h : problem_conditions counts) : 
  counts.rachel / counts.tanya_red = 1 / 5 := by
  sorry


end eraser_ratio_l2214_221452


namespace distance_from_displacements_l2214_221401

/-- The distance between two points given their net displacements -/
theorem distance_from_displacements (south west : ℝ) :
  south = 20 →
  west = 50 →
  Real.sqrt (south^2 + west^2) = 50 * Real.sqrt 2.9 := by
  sorry

end distance_from_displacements_l2214_221401


namespace division_problem_l2214_221413

theorem division_problem (dividend quotient remainder : ℕ) (divisor : ℕ) : 
  dividend = 55053 → 
  quotient = 120 → 
  remainder = 333 → 
  dividend = divisor * quotient + remainder → 
  divisor = 456 := by
sorry

end division_problem_l2214_221413


namespace expression_simplification_l2214_221407

theorem expression_simplification (a : ℝ) (h : a = 3 + Real.sqrt 3) :
  (1 - 1 / (a - 2)) / ((a^2 - 6*a + 9) / (a^2 - 2*a)) = Real.sqrt 3 + 1 := by
  sorry

end expression_simplification_l2214_221407


namespace geometric_series_ratio_l2214_221447

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a / (1 - r) = 64 * (a * r^4) / (1 - r)) → r = 1/2 := by
  sorry

end geometric_series_ratio_l2214_221447


namespace trapezoid_median_theorem_median_is_six_l2214_221467

/-- The length of the median of a trapezoid -/
def median_length : ℝ := 6

/-- The length of the longer base of the trapezoid -/
def longer_base : ℝ := 1.5 * median_length

/-- The length of the shorter base of the trapezoid -/
def shorter_base : ℝ := median_length - 3

/-- Theorem: The median of a trapezoid is the average of its bases -/
theorem trapezoid_median_theorem (median : ℝ) (longer_base shorter_base : ℝ) 
  (h1 : longer_base = 1.5 * median) 
  (h2 : shorter_base = median - 3) : 
  median = (longer_base + shorter_base) / 2 := by sorry

/-- Proof that the median length is 6 units -/
theorem median_is_six : 
  median_length = 6 ∧ 
  longer_base = 1.5 * median_length ∧ 
  shorter_base = median_length - 3 ∧
  median_length = (longer_base + shorter_base) / 2 := by sorry

end trapezoid_median_theorem_median_is_six_l2214_221467


namespace faye_bought_48_books_l2214_221439

/-- The number of coloring books Faye initially had -/
def initial_books : ℕ := 34

/-- The number of coloring books Faye gave away -/
def books_given_away : ℕ := 3

/-- The total number of coloring books Faye had after buying more -/
def final_total : ℕ := 79

/-- The number of coloring books Faye bought -/
def books_bought : ℕ := final_total - (initial_books - books_given_away)

theorem faye_bought_48_books : books_bought = 48 := by
  sorry

end faye_bought_48_books_l2214_221439


namespace arithmetic_geometric_sequence_l2214_221476

/-- 
Given three numbers forming an arithmetic sequence where the first number is 3,
and when the middle term is reduced by 6 it forms a geometric sequence,
prove that the third number (the unknown number) is either 3 or 27.
-/
theorem arithmetic_geometric_sequence (a b : ℝ) : 
  (2 * a = 3 + b) →  -- arithmetic sequence condition
  ((a - 6)^2 = 3 * b) →  -- geometric sequence condition after reduction
  (b = 3 ∨ b = 27) :=  -- conclusion: the unknown number is either 3 or 27
by sorry

end arithmetic_geometric_sequence_l2214_221476


namespace xyz_square_equality_implies_zero_l2214_221474

theorem xyz_square_equality_implies_zero (x y z : ℤ) : 
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

-- Note: The second part of the original problem doesn't have a definitive answer,
-- so we'll omit it from the Lean statement.

end xyz_square_equality_implies_zero_l2214_221474


namespace tan_function_property_l2214_221421

/-- 
Given positive constants a and b, if the function y = a * tan(b * x) 
has a period of π/2 and passes through the point (π/8, 1), then ab = 2.
-/
theorem tan_function_property (a b : ℝ) : 
  a > 0 → b > 0 → 
  (π / b = π / 2) → 
  (a * Real.tan (b * π / 8) = 1) → 
  a * b = 2 := by
sorry

end tan_function_property_l2214_221421


namespace javier_exercise_minutes_l2214_221436

/-- Proves that Javier exercised for 50 minutes each day given the conditions of the problem -/
theorem javier_exercise_minutes : ℕ → Prop :=
  fun x => 
    (∀ d : ℕ, d ≤ 7 → x > 0) →  -- Javier exercised some minutes every day for one week
    (3 * 90 + 7 * x = 620) →   -- Total exercise time for both Javier and Sanda
    x = 50                     -- Javier exercised for 50 minutes each day

/-- Proof of the theorem -/
lemma prove_javier_exercise_minutes : ∃ x : ℕ, javier_exercise_minutes x :=
  sorry

end javier_exercise_minutes_l2214_221436


namespace sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l2214_221427

-- Proposition 1
theorem sufficient_not_necessary :
  (∃ x : ℝ, x ≠ 1 ∧ x^2 - 3*x + 2 = 0) ∧
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) := by sorry

-- Proposition 2
theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔
  (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0) := by sorry

-- Proposition 3
theorem negation_equivalence :
  (¬∃ x : ℝ, x > 0 ∧ x^2 + x + 1 < 0) ↔
  (∀ x : ℝ, x > 0 → x^2 + x + 1 ≥ 0) := by sorry

end sufficient_not_necessary_contrapositive_equivalence_negation_equivalence_l2214_221427


namespace cos_shift_equals_sin_shift_l2214_221495

theorem cos_shift_equals_sin_shift (x : ℝ) : 
  Real.cos (2 * x - π / 4) = Real.sin (2 * (x + π / 8)) := by
  sorry

end cos_shift_equals_sin_shift_l2214_221495


namespace romanian_sequence_swaps_l2214_221406

/-- Represents a Romanian sequence -/
def RomanianSequence (n : ℕ) := { s : List Char // s.length = 3*n ∧ s.count 'I' = n ∧ s.count 'M' = n ∧ s.count 'O' = n }

/-- The minimum number of swaps required to transform one sequence into another -/
def minSwaps (s1 s2 : List Char) : ℕ := sorry

theorem romanian_sequence_swaps (n : ℕ) :
  ∀ (X : RomanianSequence n), ∃ (Y : RomanianSequence n), minSwaps X.val Y.val ≥ (3 * n^2) / 2 := by sorry

end romanian_sequence_swaps_l2214_221406


namespace min_unheard_lines_l2214_221460

/-- Represents the number of lines in a sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of sonnets Horatio read -/
def sonnets_read : ℕ := 7

/-- Represents the minimum number of additional sonnets Horatio prepared -/
def min_additional_sonnets : ℕ := 1

/-- Theorem stating the minimum number of unheard lines -/
theorem min_unheard_lines :
  min_additional_sonnets * lines_per_sonnet = 14 :=
by sorry

end min_unheard_lines_l2214_221460


namespace hyperbola_eccentricity_l2214_221400

/-- Proves that the eccentricity of a hyperbola with specific properties is 2√3/3 -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptote := fun (x : ℝ) ↦ b / a * x
  let F := (c, 0)
  let A := (c, b^2 / a)
  let B := (c, b * c / a)
  hyperbola c (b^2 / a) ∧ 
  A.1 = (F.1 + B.1) / 2 ∧ 
  A.2 = (F.2 + B.2) / 2 →
  c / a = 2 * Real.sqrt 3 / 3 := by
sorry

end hyperbola_eccentricity_l2214_221400


namespace sqrt_a_is_integer_l2214_221454

theorem sqrt_a_is_integer (a b : ℕ) (h1 : a > b) (h2 : b > 0)
  (h3 : ∃ k : ℤ, (Real.sqrt (Real.sqrt a + Real.sqrt b) + Real.sqrt (Real.sqrt a - Real.sqrt b)) = k) :
  ∃ n : ℕ, Real.sqrt a = n :=
sorry

end sqrt_a_is_integer_l2214_221454


namespace y_intercept_distance_of_intersecting_lines_l2214_221457

/-- A line in 2D space represented by its slope and a point it passes through -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Calculate the y-intercept of a line -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- The distance between two real numbers -/
def distance (a b : ℝ) : ℝ :=
  |a - b|

theorem y_intercept_distance_of_intersecting_lines :
  let l1 : Line := { slope := -2, point := (8, 20) }
  let l2 : Line := { slope := 4, point := (8, 20) }
  distance (y_intercept l1) (y_intercept l2) = 68 := by
  sorry

end y_intercept_distance_of_intersecting_lines_l2214_221457


namespace a_total_share_l2214_221461

def total_profit : ℚ := 9600
def a_investment : ℚ := 15000
def b_investment : ℚ := 25000
def management_fee_percentage : ℚ := 10 / 100

def total_investment : ℚ := a_investment + b_investment

def management_fee (profit : ℚ) : ℚ := management_fee_percentage * profit

def remaining_profit (profit : ℚ) : ℚ := profit - management_fee profit

def a_share_ratio : ℚ := a_investment / total_investment

theorem a_total_share :
  management_fee total_profit + (a_share_ratio * remaining_profit total_profit) = 4200 := by
  sorry

end a_total_share_l2214_221461


namespace isosceles_equilateral_conditions_l2214_221480

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- Represents the feet of perpendiculars from a point to the sides of a triangle -/
structure Perpendiculars where
  D : Point  -- foot on AB
  E : Point  -- foot on BC
  F : Point  -- foot on CA

/-- Calculates the feet of perpendiculars from a point to the sides of a triangle -/
def calculatePerpendiculars (p : Point) (t : Triangle) : Perpendiculars := sorry

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Represents an Apollonius circle of a triangle -/
structure ApolloniusCircle where
  center : Point
  radius : ℝ

/-- Calculates the Apollonius circles of a triangle -/
def calculateApolloniusCircles (t : Triangle) : List ApolloniusCircle := sorry

/-- Checks if a point lies on an Apollonius circle -/
def liesOnApolloniusCircle (p : Point) (c : ApolloniusCircle) : Prop := sorry

/-- Calculates the Fermat point of a triangle -/
def calculateFermatPoint (t : Triangle) : Point := sorry

/-- The main theorem -/
theorem isosceles_equilateral_conditions 
  (t : Triangle) 
  (h1 : isAcuteAngled t) 
  (p : Point) 
  (h2 : isInside p t) 
  (perps : Perpendiculars) 
  (h3 : perps = calculatePerpendiculars p t) :
  (isIsosceles (Triangle.mk perps.D perps.E perps.F) ↔ 
    ∃ c ∈ calculateApolloniusCircles t, liesOnApolloniusCircle p c) ∧
  (isEquilateral (Triangle.mk perps.D perps.E perps.F) ↔ 
    p = calculateFermatPoint t) := by sorry

end isosceles_equilateral_conditions_l2214_221480


namespace linear_function_through_origin_l2214_221484

/-- A linear function y = (m-1)x + m^2 - 1 passing through the origin has m = -1 -/
theorem linear_function_through_origin (m : ℝ) :
  (∀ x y : ℝ, y = (m - 1) * x + m^2 - 1) →
  (m - 1 ≠ 0) →
  (0 : ℝ) = (m - 1) * 0 + m^2 - 1 →
  m = -1 := by
  sorry

#check linear_function_through_origin

end linear_function_through_origin_l2214_221484


namespace sufficient_not_necessary_condition_l2214_221419

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, x > 2 → x^2 - 3*x + 2 > 0) ∧
  (∃ x : ℝ, x^2 - 3*x + 2 > 0 ∧ ¬(x > 2)) := by
  sorry

end sufficient_not_necessary_condition_l2214_221419


namespace isosceles_triangle_base_length_l2214_221442

/-- An isosceles triangle with perimeter 24 and a median that divides the perimeter in a 5:3 ratio -/
structure IsoscelesTriangle where
  /-- Length of each equal side -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The perimeter is 24 -/
  perimeter_eq : 2 * x + y = 24
  /-- The median divides the perimeter in a 5:3 ratio -/
  median_ratio : 3 * x / (x + y) = 5 / 3

/-- The base of the isosceles triangle is 4 -/
theorem isosceles_triangle_base_length (t : IsoscelesTriangle) : t.y = 4 := by
  sorry

end isosceles_triangle_base_length_l2214_221442


namespace book_store_inventory_l2214_221482

theorem book_store_inventory (initial_books : ℝ) (first_addition : ℝ) (second_addition : ℝ) :
  initial_books = 41.0 →
  first_addition = 33.0 →
  second_addition = 2.0 →
  initial_books + first_addition + second_addition = 76.0 := by
  sorry

end book_store_inventory_l2214_221482


namespace sandy_clothing_purchase_l2214_221471

/-- Represents the amount spent on clothes in a foreign currency and the exchange rate --/
structure ClothingPurchase where
  shorts : ℝ
  shirt : ℝ
  jacket : ℝ
  exchange_rate : ℝ

/-- Calculates the total amount spent in the home currency --/
def total_spent_home_currency (purchase : ClothingPurchase) : ℝ :=
  (purchase.shorts + purchase.shirt + purchase.jacket) * purchase.exchange_rate

/-- Theorem stating that the total amount spent in the home currency is 33.56 times the exchange rate --/
theorem sandy_clothing_purchase (purchase : ClothingPurchase)
  (h_shorts : purchase.shorts = 13.99)
  (h_shirt : purchase.shirt = 12.14)
  (h_jacket : purchase.jacket = 7.43) :
  total_spent_home_currency purchase = 33.56 * purchase.exchange_rate := by
  sorry

end sandy_clothing_purchase_l2214_221471


namespace minimum_value_and_range_l2214_221498

theorem minimum_value_and_range (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (|2*a + b| + |2*a - b|) / |a| ≥ 4) ∧
  (∀ x : ℝ, |2*a + b| + |2*a - b| ≥ |a| * (|2 + x| + |2 - x|) → -2 ≤ x ∧ x ≤ 2) :=
by sorry

end minimum_value_and_range_l2214_221498


namespace root_in_interval_l2214_221444

noncomputable def f (x : ℝ) := Real.exp x - x - 2

theorem root_in_interval :
  ∃! x : ℝ, x ∈ Set.Ioo 1 2 ∧ f x = 0 :=
sorry

end root_in_interval_l2214_221444


namespace sequence_problem_l2214_221410

theorem sequence_problem (a : ℕ → ℝ) 
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_relation : ∀ n, (a (n + 1))^2 + (a n)^2 = 2 * n * ((a (n + 1))^2 - (a n)^2)) :
  a 113 = 15 := by
  sorry

end sequence_problem_l2214_221410


namespace triangle_side_range_l2214_221443

theorem triangle_side_range (A B C : ℝ × ℝ) (x : ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  AB = 16 ∧ AC = 7 ∧ BC = x →
  9 < x ∧ x < 23 :=
by sorry

end triangle_side_range_l2214_221443


namespace min_smartphones_for_discount_l2214_221472

def smartphone_price : ℝ := 600
def discount_rate : ℝ := 0.05
def savings : ℝ := 90

theorem min_smartphones_for_discount :
  ∃ n : ℕ, n > 0 ∧ 
  n * smartphone_price * discount_rate = savings ∧
  ∀ m : ℕ, m > 0 → m * smartphone_price * discount_rate = savings → n ≤ m :=
by sorry

end min_smartphones_for_discount_l2214_221472


namespace trig_identity_l2214_221465

theorem trig_identity (x : ℝ) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin (5*π/6 - x) + (Real.sin (π/3 - x))^2 = 19/16 := by
  sorry

end trig_identity_l2214_221465


namespace polynomial_product_zero_l2214_221499

theorem polynomial_product_zero (a : ℚ) (h : a = 5/3) :
  (6*a^3 - 11*a^2 + 3*a - 2) * (3*a - 5) = 0 := by
  sorry

end polynomial_product_zero_l2214_221499


namespace no_real_with_negative_sum_of_abs_and_square_l2214_221428

theorem no_real_with_negative_sum_of_abs_and_square :
  ¬ (∃ x : ℝ, abs x + x^2 < 0) := by
sorry

end no_real_with_negative_sum_of_abs_and_square_l2214_221428


namespace problem_1_problem_2_problem_3_l2214_221451

-- Problem 1
theorem problem_1 : 24 - |(-2)| + (-16) - 8 = -2 := by sorry

-- Problem 2
theorem problem_2 : (-2) * (3/2) / (-3/4) * 4 = 4 := by sorry

-- Problem 3
theorem problem_3 : (-1)^2016 - (1 - 0.5) / 3 * (2 - (-3)^2) = 1/6 := by sorry

end problem_1_problem_2_problem_3_l2214_221451


namespace tank_difference_l2214_221418

theorem tank_difference (total : ℕ) (german allied sanchalian : ℕ) 
  (h1 : total = 115)
  (h2 : german = 2 * allied + 2)
  (h3 : allied = 3 * sanchalian + 1)
  (h4 : total = german + allied + sanchalian) :
  german - sanchalian = 59 := by
  sorry

end tank_difference_l2214_221418


namespace min_tests_for_16_people_l2214_221412

/-- Represents the number of people in the group -/
def total_people : ℕ := 16

/-- Represents the number of infected people -/
def infected_people : ℕ := 1

/-- The function that calculates the minimum number of tests required -/
def min_tests (n : ℕ) : ℕ := Nat.log2 n + 1

/-- Theorem stating that the minimum number of tests for 16 people is 4 -/
theorem min_tests_for_16_people :
  min_tests total_people = 4 :=
sorry

end min_tests_for_16_people_l2214_221412


namespace pentagonal_pyramid_edges_l2214_221448

-- Define a pentagonal pyramid
structure PentagonalPyramid where
  base : Pentagon
  triangular_faces : Fin 5 → Triangle
  common_vertex : Point

-- Define the number of edges in a pentagonal pyramid
def num_edges_pentagonal_pyramid (pp : PentagonalPyramid) : ℕ := 10

-- Theorem statement
theorem pentagonal_pyramid_edges (pp : PentagonalPyramid) :
  num_edges_pentagonal_pyramid pp = 10 := by
  sorry

end pentagonal_pyramid_edges_l2214_221448


namespace fourth_day_temperature_l2214_221449

theorem fourth_day_temperature
  (temp1 temp2 temp3 : ℤ)
  (avg_temp : ℚ)
  (h1 : temp1 = -36)
  (h2 : temp2 = 13)
  (h3 : temp3 = -15)
  (h4 : avg_temp = -12)
  (h5 : (temp1 + temp2 + temp3 + temp4 : ℚ) / 4 = avg_temp) :
  temp4 = -10 :=
sorry

end fourth_day_temperature_l2214_221449


namespace last_three_average_l2214_221411

theorem last_three_average (list : List ℝ) : 
  list.length = 6 →
  list.sum / 6 = 60 →
  (list.take 3).sum / 3 = 55 →
  (list.drop 3).sum / 3 = 65 := by
  sorry

end last_three_average_l2214_221411


namespace teacher_raise_percentage_l2214_221404

def former_salary : ℕ := 45000
def num_kids : ℕ := 9
def payment_per_kid : ℕ := 6000

def total_new_salary : ℕ := num_kids * payment_per_kid

def raise_amount : ℕ := total_new_salary - former_salary

def raise_percentage : ℚ := (raise_amount : ℚ) / (former_salary : ℚ) * 100

theorem teacher_raise_percentage :
  raise_percentage = 20 := by sorry

end teacher_raise_percentage_l2214_221404


namespace polygon_sequence_limit_l2214_221477

/-- Represents the sequence of polygons formed by cutting corners -/
def polygon_sequence (n : ℕ) : ℝ :=
  sorry

/-- The area of the triangle cut from each corner in the nth iteration -/
def cut_triangle_area (n : ℕ) : ℝ :=
  sorry

/-- The number of corners in the nth polygon -/
def num_corners (n : ℕ) : ℕ :=
  sorry

theorem polygon_sequence_limit :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |polygon_sequence n - 5/7| < ε :=
sorry

end polygon_sequence_limit_l2214_221477


namespace danica_plane_arrangement_l2214_221430

theorem danica_plane_arrangement : 
  (∃ n : ℕ, (17 + n) % 7 = 0 ∧ ∀ m : ℕ, m < n → (17 + m) % 7 ≠ 0) → 
  (∃ n : ℕ, (17 + n) % 7 = 0 ∧ ∀ m : ℕ, m < n → (17 + m) % 7 ≠ 0 ∧ n = 4) :=
by sorry

end danica_plane_arrangement_l2214_221430


namespace max_acute_angles_non_convex_polygon_l2214_221453

theorem max_acute_angles_non_convex_polygon (n : ℕ) (h : n ≥ 3) :
  let sum_interior_angles := (n - 2) * 180
  let max_acute_angles := (2 * n) / 3 + 1
  ∃ k : ℕ, k ≤ max_acute_angles ∧
    k * 90 + (n - k) * 360 < sum_interior_angles ∧
    ∀ m : ℕ, m > k → m * 90 + (n - m) * 360 ≥ sum_interior_angles :=
by sorry

end max_acute_angles_non_convex_polygon_l2214_221453


namespace arithmetic_log_implies_square_product_converse_not_always_true_l2214_221475

-- Define a predicate for arithmetic sequence of logarithms
def is_arithmetic_sequence (x y z : ℝ) : Prop :=
  ∃ (a d : ℝ), (Real.log x = a) ∧ (Real.log y = a + d) ∧ (Real.log z = a + 2*d)

-- Define the theorem
theorem arithmetic_log_implies_square_product (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  is_arithmetic_sequence x y z → y^2 = x*z :=
by sorry

-- Define a counterexample to show the converse is not necessarily true
theorem converse_not_always_true :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ y^2 = x*z ∧ ¬(is_arithmetic_sequence x y z) :=
by sorry

end arithmetic_log_implies_square_product_converse_not_always_true_l2214_221475


namespace train_length_l2214_221486

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 42 / 3600 → speed * time * 1000 = 700 := by sorry

end train_length_l2214_221486


namespace min_trains_for_800_passengers_l2214_221488

/-- Given a maximum capacity of passengers per train and a total number of passengers to transport,
    calculate the minimum number of trains required. -/
def min_trains (capacity : ℕ) (total_passengers : ℕ) : ℕ :=
  (total_passengers + capacity - 1) / capacity

theorem min_trains_for_800_passengers :
  min_trains 50 800 = 16 := by
  sorry

end min_trains_for_800_passengers_l2214_221488


namespace vector_equation_l2214_221446

/-- Given vectors a, b, c, and e in a vector space, 
    prove that 2a - 3b + c = 23e under certain conditions. -/
theorem vector_equation (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (e : V) (a b c : V) 
  (ha : a = 5 • e) 
  (hb : b = -3 • e) 
  (hc : c = 4 • e) : 
  2 • a - 3 • b + c = 23 • e := by sorry

end vector_equation_l2214_221446


namespace floor_equation_solution_l2214_221464

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3 * x⌋ + 1/3⌋ = ⌊x + 3⌋ ↔ 4/3 ≤ x ∧ x < 5/3 :=
sorry

end floor_equation_solution_l2214_221464


namespace line_equation_l2214_221459

/-- Circle C with center (3, 5) and radius sqrt(5) -/
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 5)^2 = 5

/-- Line l passing through the center of circle C -/
structure Line_l where
  slope : ℝ
  equation : ℝ → ℝ → Prop
  passes_through_center : equation 3 5

/-- Point on the circle -/
structure Point_on_circle where
  x : ℝ
  y : ℝ
  on_circle : circle_C x y

/-- Point on the y-axis -/
structure Point_on_y_axis where
  y : ℝ

/-- Midpoint condition -/
def is_midpoint (A B P : ℝ × ℝ) : Prop :=
  A.1 = (P.1 + B.1) / 2 ∧ A.2 = (P.2 + B.2) / 2

theorem line_equation (l : Line_l) 
  (A B : Point_on_circle) 
  (P : Point_on_y_axis)
  (h_A_on_l : l.equation A.x A.y)
  (h_B_on_l : l.equation B.x B.y)
  (h_P_on_l : l.equation 0 P.y)
  (h_midpoint : is_midpoint (A.x, A.y) (B.x, B.y) (0, P.y)) :
  (∃ k : ℝ, k = 2 ∨ k = -2) ∧ 
  (∀ x y, l.equation x y ↔ y - 5 = k * (x - 3)) :=
sorry

end line_equation_l2214_221459


namespace radical_expression_equality_l2214_221470

theorem radical_expression_equality : 
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) - (Real.sqrt 3 - Real.sqrt 2)^2 = 2 * Real.sqrt 6 - 2 := by
  sorry

end radical_expression_equality_l2214_221470


namespace ratio_equals_2021_l2214_221468

def numerator_sum : ℕ → ℚ
  | 0 => 0
  | n + 1 => numerator_sum n + (2021 - n) / (n + 1)

def denominator_sum : ℕ → ℚ
  | 0 => 0
  | n + 1 => denominator_sum n + 1 / (n + 3)

theorem ratio_equals_2021 : 
  (numerator_sum 2016) / (denominator_sum 2016) = 2021 := by
  sorry

end ratio_equals_2021_l2214_221468


namespace remainder_equivalence_l2214_221433

theorem remainder_equivalence (N : ℤ) (k : ℤ) : 
  N % 18 = 19 → N % 242 = (18 * k + 19) % 242 := by
  sorry

end remainder_equivalence_l2214_221433


namespace smallest_possible_students_l2214_221479

/-- Represents the number of students in each of the four equal-sized groups -/
def n : ℕ := 7

/-- The total number of students in the drama club -/
def total_students : ℕ := 4 * n + 2 * (n + 1)

/-- The drama club has six groups -/
axiom six_groups : ℕ

/-- Four groups have the same number of students -/
axiom four_equal_groups : ℕ

/-- Two groups have one more student than the other four -/
axiom two_larger_groups : ℕ

/-- The total number of groups is six -/
axiom total_groups : six_groups = 4 + 2

/-- The total number of students exceeds 40 -/
axiom exceeds_forty : total_students > 40

/-- 44 is the smallest number of students satisfying all conditions -/
theorem smallest_possible_students : total_students = 44 ∧ 
  ∀ m : ℕ, m < n → 4 * m + 2 * (m + 1) ≤ 40 := by sorry

end smallest_possible_students_l2214_221479


namespace largest_product_is_15_l2214_221438

def numbers : List ℤ := [2, -3, 4, -5]

theorem largest_product_is_15 : 
  (List.map (fun x => List.map (fun y => x * y) numbers) numbers).join.maximum? = some 15 := by
  sorry

end largest_product_is_15_l2214_221438


namespace pure_imaginary_complex_number_l2214_221409

theorem pure_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + ((m^2 - 3*m) : ℝ)*I) → m = 2 := by
  sorry

end pure_imaginary_complex_number_l2214_221409


namespace pair_and_triplet_count_two_pairs_count_l2214_221425

/- Define the structure of a deck of cards -/
def numSuits : Nat := 4
def numRanks : Nat := 13
def deckSize : Nat := numSuits * numRanks

/- Define the combination function -/
def choose (n k : Nat) : Nat :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/- Theorem for part 1 -/
theorem pair_and_triplet_count :
  choose numRanks 1 * choose numSuits 2 * choose (numRanks - 1) 1 * choose numSuits 3 = 3744 :=
by sorry

/- Theorem for part 2 -/
theorem two_pairs_count :
  choose numRanks 2 * (choose numSuits 2)^2 * choose (numRanks - 2) 1 * choose numSuits 1 = 123552 :=
by sorry

end pair_and_triplet_count_two_pairs_count_l2214_221425


namespace lamp_cost_l2214_221483

theorem lamp_cost (lamp_cost bulb_cost : ℝ) : 
  (bulb_cost = lamp_cost - 4) →
  (2 * lamp_cost + 6 * bulb_cost = 32) →
  lamp_cost = 7 := by
sorry

end lamp_cost_l2214_221483


namespace intersection_of_M_and_N_l2214_221423

def M : Set Nat := {1, 3, 5, 7}
def N : Set Nat := {5, 6, 7}

theorem intersection_of_M_and_N : M ∩ N = {5, 7} := by
  sorry

end intersection_of_M_and_N_l2214_221423


namespace speed_time_distance_return_trip_time_l2214_221429

/-- The distance to Yinping Mountain in kilometers -/
def distance : ℝ := 240

/-- The speed of the car in km/h -/
def speed (v : ℝ) : ℝ := v

/-- The time taken for the trip in hours -/
def time (t : ℝ) : ℝ := t

/-- The relationship between distance, speed, and time -/
theorem speed_time_distance (v t : ℝ) (h : t > 0) :
  speed v * time t = distance → v = distance / t :=
sorry

/-- The time taken for the return trip at 60 km/h -/
theorem return_trip_time :
  ∃ t : ℝ, t > 0 ∧ speed 60 * time t = distance ∧ t = 4 :=
sorry

end speed_time_distance_return_trip_time_l2214_221429


namespace garage_wheels_l2214_221487

/-- The number of wheels in a garage with bicycles and cars -/
def total_wheels (num_bicycles : ℕ) (num_cars : ℕ) : ℕ :=
  num_bicycles * 2 + num_cars * 4

/-- Theorem: The total number of wheels in the garage is 82 -/
theorem garage_wheels :
  total_wheels 9 16 = 82 := by
  sorry

end garage_wheels_l2214_221487


namespace circle_radius_from_square_perimeter_area_equality_l2214_221450

theorem circle_radius_from_square_perimeter_area_equality (r : ℝ) : 
  (4 * (r * Real.sqrt 2)) = (Real.pi * r^2) → r = (4 * Real.sqrt 2) / Real.pi :=
by sorry

end circle_radius_from_square_perimeter_area_equality_l2214_221450


namespace line_within_plane_is_subset_l2214_221490

-- Define a type for geometric objects
inductive GeometricObject
| Line : GeometricObject
| Plane : GeometricObject

-- Define a relation for "is within"
def isWithin (x y : GeometricObject) : Prop := sorry

-- Define the subset relation
def subset (x y : GeometricObject) : Prop := sorry

-- Theorem statement
theorem line_within_plane_is_subset (a α : GeometricObject) :
  a = GeometricObject.Line → α = GeometricObject.Plane → isWithin a α → subset a α := by
  sorry

end line_within_plane_is_subset_l2214_221490


namespace josh_book_cost_l2214_221417

/-- Represents the cost of items and quantities purchased by Josh --/
structure ShoppingTrip where
  numFilms : ℕ
  filmCost : ℕ
  numBooks : ℕ
  numCDs : ℕ
  cdCost : ℕ
  totalSpent : ℕ

/-- Calculates the cost of each book given the shopping trip details --/
def bookCost (trip : ShoppingTrip) : ℕ :=
  (trip.totalSpent - trip.numFilms * trip.filmCost - trip.numCDs * trip.cdCost) / trip.numBooks

/-- Theorem stating that the cost of each book in Josh's shopping trip is 4 --/
theorem josh_book_cost :
  let trip : ShoppingTrip := {
    numFilms := 9,
    filmCost := 5,
    numBooks := 4,
    numCDs := 6,
    cdCost := 3,
    totalSpent := 79
  }
  bookCost trip = 4 := by sorry

end josh_book_cost_l2214_221417


namespace necessary_but_not_sufficient_l2214_221415

def M : Set ℝ := {x | x > 2}
def P : Set ℝ := {x | x < 3}

theorem necessary_but_not_sufficient :
  (∀ x, x ∈ (M ∩ P) → x ∈ (M ∪ P)) ∧
  (∃ x, x ∈ (M ∪ P) ∧ x ∉ (M ∩ P)) := by
  sorry

end necessary_but_not_sufficient_l2214_221415


namespace major_axis_length_l2214_221405

def ellipse_equation (x y : ℝ) : Prop := y^2 / 25 + x^2 / 15 = 1

theorem major_axis_length :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ (x y : ℝ), ellipse_equation x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) ∧
  max a b = 5 :=
sorry

end major_axis_length_l2214_221405


namespace hash_2_5_3_equals_1_l2214_221492

-- Define the # operation
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Theorem statement
theorem hash_2_5_3_equals_1 : hash 2 5 3 = 1 := by sorry

end hash_2_5_3_equals_1_l2214_221492


namespace shaded_fraction_of_rectangle_l2214_221462

theorem shaded_fraction_of_rectangle (length width : ℕ) 
  (h_length : length = 15)
  (h_width : width = 20)
  (section_fraction : ℚ)
  (h_section : section_fraction = 1 / 5)
  (shaded_fraction : ℚ)
  (h_shaded : shaded_fraction = 1 / 4) :
  (shaded_fraction * section_fraction : ℚ) = 1 / 20 := by
sorry

end shaded_fraction_of_rectangle_l2214_221462


namespace main_theorem_l2214_221416

/-- The set S of ordered triples satisfying the given conditions -/
def S (n : ℕ) : Set (ℕ × ℕ × ℕ) :=
  {t | ∃ x y z, t = (x, y, z) ∧ 
       x ∈ Finset.range n ∧ y ∈ Finset.range n ∧ z ∈ Finset.range n ∧
       ((x < y ∧ y < z) ∨ (y < z ∧ z < x) ∨ (z < x ∧ x < y)) ∧
       ¬((x < y ∧ y < z) ∧ (y < z ∧ z < x)) ∧
       ¬((y < z ∧ z < x) ∧ (z < x ∧ x < y)) ∧
       ¬((z < x ∧ x < y) ∧ (x < y ∧ y < z))}

/-- The main theorem -/
theorem main_theorem (n : ℕ) (h : n ≥ 4) 
  (x y z w : ℕ) (hxyz : (x, y, z) ∈ S n) (hzwx : (z, w, x) ∈ S n) :
  (y, z, w) ∈ S n ∧ (x, y, w) ∈ S n := by
  sorry

end main_theorem_l2214_221416


namespace sum_of_products_negative_max_greater_or_equal_cube_root_four_l2214_221434

-- Define the conditions
def sum_zero (a b c : ℝ) : Prop := a + b + c = 0
def product_one (a b c : ℝ) : Prop := a * b * c = 1

-- Define the theorems to prove
theorem sum_of_products_negative (a b c : ℝ) (h1 : sum_zero a b c) (h2 : product_one a b c) :
  a * b + b * c + c * a < 0 :=
sorry

theorem max_greater_or_equal_cube_root_four (a b c : ℝ) (h1 : sum_zero a b c) (h2 : product_one a b c) :
  max a (max b c) ≥ (4 : ℝ) ^ (1/3) :=
sorry

end sum_of_products_negative_max_greater_or_equal_cube_root_four_l2214_221434


namespace inheritance_calculation_l2214_221469

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 38621

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.15

/-- The processing fee in dollars -/
def processing_fee : ℝ := 2500

/-- The total amount paid in taxes and fees in dollars -/
def total_paid : ℝ := 16500

theorem inheritance_calculation (x : ℝ) (h : x = inheritance) :
  federal_tax_rate * x + state_tax_rate * (1 - federal_tax_rate) * x + processing_fee = total_paid :=
by sorry

end inheritance_calculation_l2214_221469


namespace valid_triplet_configurations_l2214_221485

/-- A structure representing a configuration of triplet subsets satisfying the given conditions -/
structure TripletConfiguration (n : ℕ) :=
  (m : ℕ)
  (subsets : Fin m → Finset (Fin n))
  (cover_pairs : ∀ (i j : Fin n), i ≠ j → ∃ (k : Fin m), {i, j} ⊆ subsets k)
  (subset_size : ∀ (k : Fin m), (subsets k).card = 3)
  (intersect_one : ∀ (k₁ k₂ : Fin m), k₁ ≠ k₂ → (subsets k₁ ∩ subsets k₂).card = 1)

/-- The theorem stating that the only valid configurations are (1, 3) and (7, 7) -/
theorem valid_triplet_configurations :
  {n : ℕ | ∃ (c : TripletConfiguration n), True} = {3, 7} :=
sorry

end valid_triplet_configurations_l2214_221485


namespace positive_numbers_inequality_l2214_221424

theorem positive_numbers_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c ≤ (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ∧
  (a^2 + b^2)/(2*c) + (b^2 + c^2)/(2*a) + (c^2 + a^2)/(2*b) ≤ a^3/(b*c) + b^3/(c*a) + c^3/(a*b) :=
by sorry

end positive_numbers_inequality_l2214_221424


namespace log_158489_between_integers_l2214_221440

theorem log_158489_between_integers : ∃ p q : ℤ,
  (p : ℝ) < Real.log 158489 / Real.log 10 ∧
  Real.log 158489 / Real.log 10 < (q : ℝ) ∧
  q = p + 1 ∧
  p + q = 11 := by
  sorry

end log_158489_between_integers_l2214_221440


namespace expression_value_l2214_221456

theorem expression_value (a b : ℤ) (ha : a = 3) (hb : b = 2) : 3 * a + 4 * b - 5 = 12 := by
  sorry

end expression_value_l2214_221456


namespace slab_cost_l2214_221408

/-- The cost of a slab of beef given the conditions of kabob stick production -/
theorem slab_cost (cubes_per_stick : ℕ) (cubes_per_slab : ℕ) (sticks : ℕ) (total_cost : ℕ) : 
  cubes_per_stick = 4 →
  cubes_per_slab = 80 →
  sticks = 40 →
  total_cost = 50 →
  (total_cost : ℚ) / ((sticks * cubes_per_stick : ℕ) / cubes_per_slab : ℚ) = 25 := by
  sorry

end slab_cost_l2214_221408


namespace problem_solution_l2214_221420

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x^2 else a^x - 1

theorem problem_solution (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : Monotone (f a)) 
  (h4 : f a a = 5 * a - 2) : 
  a = 2 := by
sorry

end problem_solution_l2214_221420


namespace smallest_number_with_given_remainders_l2214_221481

theorem smallest_number_with_given_remainders : ∃! n : ℕ,
  (∀ m : ℕ, m < n → ¬(m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5)) ∧
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ n % 6 = 5 :=
by sorry

end smallest_number_with_given_remainders_l2214_221481
