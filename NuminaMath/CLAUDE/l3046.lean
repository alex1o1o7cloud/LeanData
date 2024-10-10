import Mathlib

namespace equation_has_four_solutions_l3046_304644

-- Define the equation
def equation (x : ℝ) : Prop := (2*x^2 - 10*x + 3)^2 = 4

-- State the theorem
theorem equation_has_four_solutions :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  equation a ∧ equation b ∧ equation c ∧ equation d ∧
  (∀ x : ℝ, equation x → (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
sorry

end equation_has_four_solutions_l3046_304644


namespace problem_statement_l3046_304639

theorem problem_statement (x y m n : ℤ) 
  (hxy : x > y) 
  (hmn : m > n) 
  (hsum_xy : x + y = 7) 
  (hprod_xy : x * y = 12) 
  (hsum_mn : m + n = 13) 
  (hsum_squares : m^2 + n^2 = 97) : 
  (x - y) - (m - n) = -4 := by
  sorry

end problem_statement_l3046_304639


namespace fixed_point_theorem_tangent_dot_product_range_l3046_304683

-- Define the curves C and M
def C (x y : ℝ) : Prop := y^2 = 4*x
def M (x y : ℝ) : Prop := (x-1)^2 + y^2 = 4 ∧ x ≥ 1

-- Define the line l
def L (m n : ℝ) (x y : ℝ) : Prop := x = m*y + n

-- Define points A and B on curve C and line l
def A_B_on_C_and_L (m n x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  C x₁ y₁ ∧ C x₂ y₂ ∧ L m n x₁ y₁ ∧ L m n x₂ y₂

-- Theorem 1
theorem fixed_point_theorem (m n x₁ y₁ x₂ y₂ : ℝ) :
  A_B_on_C_and_L m n x₁ y₁ x₂ y₂ →
  x₁*x₂ + y₁*y₂ = -4 →
  ∃ (m : ℝ), L m 2 2 0 :=
sorry

-- Theorem 2
theorem tangent_dot_product_range (m n x₁ y₁ x₂ y₂ : ℝ) :
  A_B_on_C_and_L m n x₁ y₁ x₂ y₂ →
  (∃ (x y : ℝ), M x y ∧ L m n x y) →
  (x₁-1)*(x₂-1) + y₁*y₂ ≤ -8 :=
sorry

end fixed_point_theorem_tangent_dot_product_range_l3046_304683


namespace constant_term_expansion_l3046_304685

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expansion function
def expansion_term (x : ℚ) (r k : ℕ) : ℚ :=
  (-1)^k * binomial r k * x^(r - 2*k)

-- Define the constant term of the expansion
def constant_term : ℚ :=
  1 - binomial 2 1 * binomial 4 2 + binomial 4 2 * binomial 0 0

-- Theorem statement
theorem constant_term_expansion :
  constant_term = -5 := by sorry

end constant_term_expansion_l3046_304685


namespace largest_prime_factor_of_2501_l3046_304604

theorem largest_prime_factor_of_2501 : ∃ p : ℕ, p.Prime ∧ p ∣ 2501 ∧ ∀ q : ℕ, q.Prime → q ∣ 2501 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_2501_l3046_304604


namespace min_sum_inverse_ratio_l3046_304662

theorem min_sum_inverse_ratio (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (3 * b) + b / (6 * c) + c / (9 * a)) ≥ 1 / (3 * Real.rpow 2 (1/3)) :=
sorry

end min_sum_inverse_ratio_l3046_304662


namespace stream_bottom_width_l3046_304602

/-- Represents the trapezoidal cross-section of a stream -/
structure StreamCrossSection where
  topWidth : ℝ
  bottomWidth : ℝ
  depth : ℝ
  area : ℝ

/-- The area of a trapezoid is equal to the average of its parallel sides multiplied by its height -/
def trapezoidAreaFormula (s : StreamCrossSection) : Prop :=
  s.area = (s.topWidth + s.bottomWidth) / 2 * s.depth

theorem stream_bottom_width
  (s : StreamCrossSection)
  (h1 : s.topWidth = 10)
  (h2 : s.depth = 80)
  (h3 : s.area = 640)
  (h4 : trapezoidAreaFormula s) :
  s.bottomWidth = 6 := by
  sorry

end stream_bottom_width_l3046_304602


namespace min_distance_circle_to_line_l3046_304636

/-- The minimum distance from a point on the circle x^2 + y^2 - 2x - 2y = 0 to the line x + y - 8 = 0 is 2√2. -/
theorem min_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | (p.1^2 + p.2^2 - 2*p.1 - 2*p.2) = 0}
  let line := {p : ℝ × ℝ | p.1 + p.2 - 8 = 0}
  ∃ (d : ℝ), d = 2 * Real.sqrt 2 ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      ∀ (q : ℝ × ℝ), q ∈ line →
        d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) :=
by sorry

end min_distance_circle_to_line_l3046_304636


namespace inequality_equivalence_l3046_304686

theorem inequality_equivalence (y : ℝ) : 
  (7 / 36 + |y - 13 / 72| < 11 / 24) ↔ (-1 / 12 < y ∧ y < 4 / 9) := by
  sorry

end inequality_equivalence_l3046_304686


namespace sum_of_odds_is_even_product_zero_implies_factor_zero_exists_even_prime_l3046_304691

-- Definition of odd number
def isOdd (n : Int) : Prop := ∃ k : Int, n = 2 * k + 1

-- Statement 1
theorem sum_of_odds_is_even (x y : Int) (hx : isOdd x) (hy : isOdd y) : 
  ∃ k : Int, x + y = 2 * k := by sorry

-- Statement 2
theorem product_zero_implies_factor_zero (x y : ℝ) (h : x * y = 0) : 
  x = 0 ∨ y = 0 := by sorry

-- Definition of prime number
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

-- Statement 3
theorem exists_even_prime : ∃ p : Nat, isPrime p ∧ ¬isOdd p := by sorry

end sum_of_odds_is_even_product_zero_implies_factor_zero_exists_even_prime_l3046_304691


namespace matthew_water_bottle_fills_l3046_304601

/-- Represents the number of times Matthew needs to fill his water bottle per week -/
def fill_times_per_week (glasses_per_day : ℕ) (ounces_per_glass : ℕ) (bottle_size : ℕ) : ℕ :=
  (7 * glasses_per_day * ounces_per_glass) / bottle_size

/-- Proves that Matthew will fill his water bottle 4 times per week -/
theorem matthew_water_bottle_fills :
  fill_times_per_week 4 5 35 = 4 := by
  sorry

end matthew_water_bottle_fills_l3046_304601


namespace line_through_points_l3046_304619

/-- A line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  point1 : (ℝ × ℝ)
  point2 : (ℝ × ℝ)
  eq1 : a * point1.1 + b = point1.2
  eq2 : a * point2.1 + b = point2.2

/-- Theorem: For a line y = ax + b passing through (3, 4) and (7, 16), a - b = 8 -/
theorem line_through_points (l : Line) 
  (h1 : l.point1 = (3, 4))
  (h2 : l.point2 = (7, 16)) : 
  l.a - l.b = 8 := by
  sorry

end line_through_points_l3046_304619


namespace inequality_solution_l3046_304684

theorem inequality_solution (x : ℝ) : (2 - x < 1) ↔ (x > 1) := by sorry

end inequality_solution_l3046_304684


namespace cars_meeting_time_l3046_304695

/-- Two cars meeting on a highway -/
theorem cars_meeting_time (highway_length : ℝ) (speed1 speed2 : ℝ) (meeting_time : ℝ) : 
  highway_length = 60 →
  speed1 = 13 →
  speed2 = 17 →
  meeting_time * (speed1 + speed2) = highway_length →
  meeting_time = 2 := by
sorry

end cars_meeting_time_l3046_304695


namespace green_caterpillar_length_l3046_304609

theorem green_caterpillar_length 
  (orange_length : ℝ) 
  (length_difference : ℝ) 
  (h1 : orange_length = 1.17)
  (h2 : length_difference = 1.83) : 
  orange_length + length_difference = 3.00 := by
  sorry

end green_caterpillar_length_l3046_304609


namespace parallel_resistors_l3046_304654

theorem parallel_resistors (x y R : ℝ) (hx : x = 4) (hy : y = 5) 
  (hR : 1 / R = 1 / x + 1 / y) : R = 20 / 9 := by
  sorry

end parallel_resistors_l3046_304654


namespace shaded_area_percentage_l3046_304621

theorem shaded_area_percentage (total_squares : ℕ) (shaded_squares : ℕ) : 
  total_squares = 16 → shaded_squares = 3 → 
  (shaded_squares : ℚ) / (total_squares : ℚ) * 100 = 18.75 := by
  sorry

end shaded_area_percentage_l3046_304621


namespace hyperbola_triangle_area_l3046_304690

-- Define the hyperbola
def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the foci of the hyperbola
def foci (F₁ F₂ : ℝ × ℝ) : Prop := ∃ c : ℝ, c > 0 ∧ F₁ = (c, 0) ∧ F₂ = (-c, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (P : ℝ × ℝ) : Prop := is_on_hyperbola P.1 P.2

-- Define the right angle condition
def right_angle (F₁ P F₂ : ℝ × ℝ) : Prop :=
  (P.1 - F₁.1)^2 + (P.2 - F₁.2)^2 + (P.1 - F₂.1)^2 + (P.2 - F₂.2)^2 =
  (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2

-- State the theorem
theorem hyperbola_triangle_area
  (F₁ F₂ P : ℝ × ℝ)
  (h_foci : foci F₁ F₂)
  (h_on_hyperbola : point_on_hyperbola P)
  (h_right_angle : right_angle F₁ P F₂) :
  (abs ((F₁.1 - P.1) * (F₂.2 - P.2) - (F₁.2 - P.2) * (F₂.1 - P.1))) / 2 = 4 :=
sorry

end hyperbola_triangle_area_l3046_304690


namespace base_seven_division_1452_14_l3046_304671

/-- Represents a number in base 7 --/
def BaseSevenNum := List Nat

/-- Converts a base 7 number to base 10 --/
def to_base_ten (n : BaseSevenNum) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (7 ^ i)) 0

/-- Converts a base 10 number to base 7 --/
def to_base_seven (n : Nat) : BaseSevenNum :=
  sorry

/-- Performs division in base 7 --/
def base_seven_div (a b : BaseSevenNum) : BaseSevenNum :=
  to_base_seven ((to_base_ten a) / (to_base_ten b))

theorem base_seven_division_1452_14 :
  base_seven_div [2, 5, 4, 1] [4, 1] = [3, 0, 1] :=
sorry

end base_seven_division_1452_14_l3046_304671


namespace trig_identity_l3046_304678

theorem trig_identity : 
  (2 * Real.sin (46 * π / 180) - Real.sqrt 3 * Real.cos (74 * π / 180)) / Real.cos (16 * π / 180) = 1 := by
  sorry

end trig_identity_l3046_304678


namespace final_sum_after_transformations_l3046_304613

theorem final_sum_after_transformations (S a b : ℝ) (h : a + b = S) :
  3 * ((a + 5) + (b + 5)) = 3 * S + 30 := by sorry

end final_sum_after_transformations_l3046_304613


namespace manufacturer_not_fraudulent_l3046_304643

/-- Represents the mass of a bread bag -/
structure BreadBag where
  labeledMass : ℝ
  tolerance : ℝ
  measuredMass : ℝ

/-- Determines if the manufacturer has engaged in fraudulent behavior -/
def isFraudulent (bag : BreadBag) : Prop :=
  bag.measuredMass < bag.labeledMass - bag.tolerance ∨ 
  bag.measuredMass > bag.labeledMass + bag.tolerance

theorem manufacturer_not_fraudulent (bag : BreadBag) 
  (h1 : bag.labeledMass = 200)
  (h2 : bag.tolerance = 3)
  (h3 : bag.measuredMass = 198) : 
  ¬(isFraudulent bag) := by
  sorry

#check manufacturer_not_fraudulent

end manufacturer_not_fraudulent_l3046_304643


namespace range_of_m_satisfying_conditions_l3046_304626

def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + (m + 6)*x + 1

theorem range_of_m_satisfying_conditions : 
  {m : ℝ | (∀ a ∈ Set.Icc 1 2, |m - 5| ≤ Real.sqrt (a^2 + 8)) ∧ 
           ¬(∃ (max min : ℝ), ∀ x, f m x ≤ max ∧ f m x ≥ min)} = 
  Set.Icc 2 6 := by sorry

end range_of_m_satisfying_conditions_l3046_304626


namespace complete_factorization_l3046_304624

theorem complete_factorization (x : ℝ) : 
  x^8 - 256 = (x^4 + 16) * (x^2 + 4) * (x + 2) * (x - 2) := by
  sorry

end complete_factorization_l3046_304624


namespace beetle_journey_l3046_304629

/-- Represents the beetle's movements in centimeters -/
def beetle_movements : List ℝ := [10, -9, 8, -6, 7.5, -6, 8, -7]

/-- Time taken per centimeter in seconds -/
def time_per_cm : ℝ := 2

/-- Calculates the final position of the beetle relative to the starting point -/
def final_position (movements : List ℝ) : ℝ :=
  movements.sum

/-- Calculates the total distance traveled by the beetle -/
def total_distance (movements : List ℝ) : ℝ :=
  movements.map abs |>.sum

/-- Calculates the total time taken for the journey -/
def total_time (movements : List ℝ) (time_per_cm : ℝ) : ℝ :=
  (total_distance movements) * time_per_cm

theorem beetle_journey :
  final_position beetle_movements = 5.5 ∧
  total_time beetle_movements time_per_cm = 123 := by
  sorry

#eval final_position beetle_movements
#eval total_time beetle_movements time_per_cm

end beetle_journey_l3046_304629


namespace lcm_problem_l3046_304620

theorem lcm_problem (a b c : ℕ) (h1 : Nat.lcm a b = 945) (h2 : Nat.lcm b c = 525) :
  Nat.lcm a c = 675 ∨ Nat.lcm a c = 4725 := by
  sorry

end lcm_problem_l3046_304620


namespace sequence_sum_l3046_304676

theorem sequence_sum (n : ℕ) (x : ℕ → ℚ) (h1 : x 1 = 1) 
  (h2 : ∀ k ∈ Finset.range (n - 1), x (k + 1) = x k + 1/2) : 
  Finset.sum (Finset.range n) (λ i => x (i + 1)) = (n^2 + 3*n) / 4 := by
  sorry

end sequence_sum_l3046_304676


namespace last_three_digits_of_6_to_150_l3046_304687

theorem last_three_digits_of_6_to_150 :
  6^150 % 1000 = 126 := by
  sorry

end last_three_digits_of_6_to_150_l3046_304687


namespace five_balls_four_boxes_l3046_304668

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  num_boxes ^ num_balls

/-- Theorem: There are 1024 ways to distribute 5 distinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 1024 := by
  sorry

end five_balls_four_boxes_l3046_304668


namespace loafer_cost_l3046_304610

/-- Calculate the cost of each pair of loafers given the sales and commission information -/
theorem loafer_cost (commission_rate : ℚ) (suit_price shirt_price : ℚ) 
  (suit_count shirt_count loafer_count : ℕ) (total_commission : ℚ) : 
  commission_rate = 15 / 100 →
  suit_count = 2 →
  shirt_count = 6 →
  loafer_count = 2 →
  suit_price = 700 →
  shirt_price = 50 →
  total_commission = 300 →
  (suit_count : ℚ) * suit_price * commission_rate + 
  (shirt_count : ℚ) * shirt_price * commission_rate + 
  (loafer_count : ℚ) * (total_commission / (loafer_count : ℚ)) = total_commission →
  total_commission / (loafer_count : ℚ) / commission_rate = 150 := by
sorry

end loafer_cost_l3046_304610


namespace table_height_is_36_l3046_304632

/-- Represents a cuboidal block of wood -/
structure Block where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Represents the arrangement in Figure 1 -/
def figure1 (b : Block) (table_height : ℝ) : ℝ :=
  b.length + table_height - b.depth

/-- Represents the arrangement in Figure 2 -/
def figure2 (b : Block) (table_height : ℝ) : ℝ :=
  2 * b.length + table_height

/-- Theorem stating the height of the table given the conditions -/
theorem table_height_is_36 (b : Block) (h : ℝ) :
  figure1 b h = 36 → figure2 b h = 46 → h = 36 := by
  sorry

#check table_height_is_36

end table_height_is_36_l3046_304632


namespace percentage_problem_l3046_304647

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1500 - 15 → x = 840 := by
  sorry

end percentage_problem_l3046_304647


namespace rotations_composition_l3046_304634

/-- A rotation in the plane. -/
structure Rotation where
  center : ℝ × ℝ
  angle : ℝ

/-- Represents the composition of two rotations. -/
def compose_rotations (r1 r2 : Rotation) : Rotation :=
  sorry

/-- The angles of a triangle formed by the centers of two rotations and their composition. -/
def triangle_angles (r1 r2 : Rotation) : ℝ × ℝ × ℝ :=
  sorry

theorem rotations_composition 
  (O₁ O₂ : ℝ × ℝ) (α β : ℝ) 
  (h1 : 0 ≤ α ∧ α < 2 * π) 
  (h2 : 0 ≤ β ∧ β < 2 * π) 
  (h3 : α + β ≠ 2 * π) :
  let r1 : Rotation := ⟨O₁, α⟩
  let r2 : Rotation := ⟨O₂, β⟩
  let r_composed := compose_rotations r1 r2
  let angles := triangle_angles r1 r2
  (r_composed.angle = α + β) ∧
  ((α + β < 2 * π → angles = (α/2, β/2, π - (α + β)/2)) ∧
   (α + β > 2 * π → angles = (π - α/2, π - β/2, (α + β)/2))) :=
by sorry

end rotations_composition_l3046_304634


namespace green_peaches_count_l3046_304605

theorem green_peaches_count (red_peaches : ℕ) (green_peaches : ℕ) : 
  red_peaches = 5 → green_peaches = red_peaches + 6 → green_peaches = 11 := by
  sorry

end green_peaches_count_l3046_304605


namespace sum_product_bounds_l3046_304623

theorem sum_product_bounds (a b c k : ℝ) (h : a + b + c = k) (h_nonzero : k ≠ 0) :
  -2/3 * k^2 ≤ a*b + a*c + b*c ∧ a*b + a*c + b*c ≤ k^2/2 := by
  sorry

end sum_product_bounds_l3046_304623


namespace card_combinations_l3046_304675

def deck_size : ℕ := 60
def hand_size : ℕ := 15

theorem card_combinations :
  Nat.choose deck_size hand_size = 660665664066 := by
  sorry

end card_combinations_l3046_304675


namespace budget_calculation_l3046_304652

theorem budget_calculation (initial_budget : ℕ) 
  (shirt_cost pants_cost coat_cost socks_cost belt_cost shoes_cost : ℕ) :
  initial_budget = 200 →
  shirt_cost = 30 →
  pants_cost = 46 →
  coat_cost = 38 →
  socks_cost = 11 →
  belt_cost = 18 →
  shoes_cost = 41 →
  initial_budget - (shirt_cost + pants_cost + coat_cost + socks_cost + belt_cost + shoes_cost) = 16 := by
  sorry

end budget_calculation_l3046_304652


namespace sally_weekday_pages_l3046_304699

/-- The number of pages Sally reads on weekdays -/
def weekday_pages : ℕ := sorry

/-- The number of pages Sally reads on weekends -/
def weekend_pages : ℕ := 20

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- The total number of pages in the book -/
def total_pages : ℕ := 180

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

theorem sally_weekday_pages :
  weekday_pages = 10 :=
by sorry

end sally_weekday_pages_l3046_304699


namespace arithmetic_geometric_ratio_l3046_304625

/-- Given an arithmetic sequence {a_n} with a₁ ≠ 0, if S₁, S₂, S₄ form a geometric sequence, 
    then a₂/a₁ = 1 or a₂/a₁ = 3 -/
theorem arithmetic_geometric_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (a 1 ≠ 0) →                          -- first term not zero
  (∀ n, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- sum formula
  (∃ r, S 2 = r * S 1 ∧ S 4 = r * S 2) →  -- geometric sequence condition
  a 2 / a 1 = 1 ∨ a 2 / a 1 = 3 :=
by sorry

end arithmetic_geometric_ratio_l3046_304625


namespace sauce_correction_l3046_304674

theorem sauce_correction (x : ℝ) : 
  (0.4 * x - 1 + 2.5 = 0.6 * x - 1.5) → x = 12.5 := by
  sorry

end sauce_correction_l3046_304674


namespace flower_beds_count_l3046_304633

/-- Given that there are 25 seeds in each flower bed and 750 seeds planted altogether,
    prove that the number of flower beds is 30. -/
theorem flower_beds_count (seeds_per_bed : ℕ) (total_seeds : ℕ) (num_beds : ℕ) 
    (h1 : seeds_per_bed = 25)
    (h2 : total_seeds = 750)
    (h3 : num_beds * seeds_per_bed = total_seeds) :
  num_beds = 30 := by
  sorry

end flower_beds_count_l3046_304633


namespace smallest_four_digit_divisible_by_37_l3046_304615

theorem smallest_four_digit_divisible_by_37 :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 1036 → ¬(37 ∣ n)) ∧ 
  1000 ≤ 1036 ∧ 
  1036 < 10000 ∧ 
  37 ∣ 1036 := by
  sorry

end smallest_four_digit_divisible_by_37_l3046_304615


namespace stamp_problem_l3046_304680

/-- Represents the number of ways to make a certain amount with given coin denominations -/
def numWays (amount : ℕ) (coins : List ℕ) : ℕ :=
  sorry

/-- The minimum number of coins needed to make the amount -/
def minCoins (amount : ℕ) (coins : List ℕ) : ℕ :=
  sorry

theorem stamp_problem :
  let stamps := [5, 7]
  minCoins 50 stamps = 8 :=
by sorry

end stamp_problem_l3046_304680


namespace unique_intersection_points_l3046_304648

/-- The first curve -/
def curve1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := -x^3 + 9 * x^2 - 4 * x + 2

/-- The intersection points of the two curves -/
def intersection_points : Set (ℝ × ℝ) := {(0, 2), (6, 86)}

/-- Theorem stating that the intersection_points are the only intersection points of curve1 and curve2 -/
theorem unique_intersection_points :
  ∀ x y : ℝ, curve1 x = curve2 x ∧ y = curve1 x ↔ (x, y) ∈ intersection_points :=
by sorry

end unique_intersection_points_l3046_304648


namespace min_value_expression_min_value_achieved_l3046_304677

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ((a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1)) / (a * b * c) ≥ 216 :=
by
  sorry

theorem min_value_achieved (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    ((x^2 + 4*x + 1) * (y^2 + 4*y + 1) * (z^2 + 4*z + 1)) / (x * y * z) = 216 :=
by
  sorry

end min_value_expression_min_value_achieved_l3046_304677


namespace largest_four_digit_multiple_of_three_l3046_304696

theorem largest_four_digit_multiple_of_three : ∃ n : ℕ, 
  n = 9999 ∧ 
  n % 3 = 0 ∧ 
  ∀ m : ℕ, m < 10000 ∧ m % 3 = 0 → m ≤ n :=
by sorry

end largest_four_digit_multiple_of_three_l3046_304696


namespace vector_operation_l3046_304688

/-- Given vectors a and b in R², prove that 2a - b equals the expected result. -/
theorem vector_operation (a b : Fin 2 → ℝ) (h1 : a = ![2, 1]) (h2 : b = ![-3, 4]) :
  (2 • a) - b = ![7, -2] := by sorry

end vector_operation_l3046_304688


namespace complex_square_l3046_304603

theorem complex_square (z : ℂ) (i : ℂ) : z = 2 - 3 * i → i^2 = -1 → z^2 = -5 - 12 * i := by
  sorry

end complex_square_l3046_304603


namespace jean_friday_calls_l3046_304646

/-- The number of calls Jean answered on each day of the week --/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day --/
def average_calls : ℕ := 40

/-- The number of working days in a week --/
def working_days : ℕ := 5

/-- Jean's call data for the week --/
def jean_calls : WeekCalls := {
  monday := 35,
  tuesday := 46,
  wednesday := 27,
  thursday := 61,
  friday := 31  -- This is what we want to prove
}

/-- Theorem stating that Jean answered 31 calls on Friday --/
theorem jean_friday_calls : 
  jean_calls.friday = 31 :=
by sorry

end jean_friday_calls_l3046_304646


namespace sequence_increasing_iff_k_greater_than_neg_three_l3046_304660

theorem sequence_increasing_iff_k_greater_than_neg_three (k : ℝ) :
  (∀ n : ℕ, (n^2 + k*n + 2) < ((n+1)^2 + k*(n+1) + 2)) ↔ k > -3 := by
  sorry

end sequence_increasing_iff_k_greater_than_neg_three_l3046_304660


namespace final_segment_speed_final_segment_speed_is_90_l3046_304653

/-- Calculates the average speed for the final segment of a journey given specific conditions. -/
theorem final_segment_speed (total_distance : ℝ) (total_time : ℝ) (first_hour_speed : ℝ) 
  (stop_time : ℝ) (second_segment_speed : ℝ) (second_segment_time : ℝ) : ℝ :=
  let net_driving_time := total_time - stop_time / 60
  let first_segment_distance := first_hour_speed * 1
  let second_segment_distance := second_segment_speed * second_segment_time
  let remaining_distance := total_distance - (first_segment_distance + second_segment_distance)
  let remaining_time := net_driving_time - (1 + second_segment_time)
  remaining_distance / remaining_time

/-- Proves that the average speed for the final segment is 90 mph under given conditions. -/
theorem final_segment_speed_is_90 : 
  final_segment_speed 150 3 45 30 50 0.75 = 90 := by
  sorry

end final_segment_speed_final_segment_speed_is_90_l3046_304653


namespace remainder_of_n_l3046_304618

theorem remainder_of_n (n : ℕ) 
  (h1 : n^2 % 7 = 3) 
  (h2 : n^3 % 7 = 6) : 
  n % 7 = 5 := by
  sorry

end remainder_of_n_l3046_304618


namespace train_bridge_problem_l3046_304673

/-- Given a train crossing a bridge, this theorem proves the length and speed of the train. -/
theorem train_bridge_problem (bridge_length : ℝ) (total_time : ℝ) (on_bridge_time : ℝ) 
  (h1 : bridge_length = 1000)
  (h2 : total_time = 60)
  (h3 : on_bridge_time = 40) :
  ∃ (train_length : ℝ) (train_speed : ℝ),
    train_length = 200 ∧ 
    train_speed = 20 ∧
    (bridge_length + train_length) / total_time = (bridge_length - train_length) / on_bridge_time :=
by
  sorry

#check train_bridge_problem

end train_bridge_problem_l3046_304673


namespace discount_calculation_l3046_304693

def list_price : ℝ := 70
def final_price : ℝ := 61.11
def first_discount : ℝ := 10

theorem discount_calculation (x : ℝ) :
  list_price * (1 - first_discount / 100) * (1 - x / 100) = final_price →
  x = 3 := by sorry

end discount_calculation_l3046_304693


namespace gcd_12740_220_minus_10_l3046_304663

theorem gcd_12740_220_minus_10 : Nat.gcd 12740 220 - 10 = 10 := by
  sorry

end gcd_12740_220_minus_10_l3046_304663


namespace five_fridays_september_implies_five_mondays_october_l3046_304689

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a month -/
structure Month where
  days : Nat
  first_day : DayOfWeek

/-- Given a day, returns the next day of the week -/
def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Counts the occurrences of a specific day in a month -/
def count_day_occurrences (m : Month) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: If September has five Fridays, October has five Mondays -/
theorem five_fridays_september_implies_five_mondays_october 
  (september : Month) 
  (october : Month) :
  september.days = 30 →
  october.days = 31 →
  count_day_occurrences september DayOfWeek.Friday = 5 →
  count_day_occurrences october DayOfWeek.Monday = 5 :=
  sorry

end five_fridays_september_implies_five_mondays_october_l3046_304689


namespace sugar_sold_is_two_kilograms_l3046_304630

/-- The number of sugar packets sold per week -/
def packets_per_week : ℕ := 20

/-- The amount of sugar in grams per packet -/
def grams_per_packet : ℕ := 100

/-- Conversion factor from grams to kilograms -/
def grams_per_kilogram : ℕ := 1000

/-- The amount of sugar sold per week in kilograms -/
def sugar_sold_per_week : ℚ :=
  (packets_per_week * grams_per_packet : ℚ) / grams_per_kilogram

theorem sugar_sold_is_two_kilograms :
  sugar_sold_per_week = 2 := by
  sorry

end sugar_sold_is_two_kilograms_l3046_304630


namespace no_real_roots_l3046_304670

theorem no_real_roots : ∀ x : ℝ, x^2 - 4*x + 8 ≠ 0 := by
  sorry

end no_real_roots_l3046_304670


namespace some_number_value_l3046_304631

theorem some_number_value (some_number : ℝ) : 
  (3.242 * 10) / some_number = 0.032420000000000004 → some_number = 1000 := by
sorry

end some_number_value_l3046_304631


namespace eagle_eye_camera_is_analogical_reasoning_l3046_304635

-- Define the different types of reasoning
inductive ReasoningType
  | Inductive
  | Analogical
  | Deductive

-- Define a structure for a reasoning process
structure ReasoningProcess where
  description : String
  type : ReasoningType

-- Define the four options
def optionA : ReasoningProcess :=
  { description := "People derive that the probability of getting heads when flipping a coin is 1/2 through numerous experiments",
    type := ReasoningType.Inductive }

def optionB : ReasoningProcess :=
  { description := "Scientists invent the eagle eye camera by studying the eyes of eagles",
    type := ReasoningType.Analogical }

def optionC : ReasoningProcess :=
  { description := "Determine the acidity or alkalinity of a solution by testing its pH value",
    type := ReasoningType.Deductive }

def optionD : ReasoningProcess :=
  { description := "Determine whether a function is periodic based on the definition of a periodic function in mathematics",
    type := ReasoningType.Deductive }

-- Theorem to prove
theorem eagle_eye_camera_is_analogical_reasoning :
  optionB.type = ReasoningType.Analogical :=
by sorry

end eagle_eye_camera_is_analogical_reasoning_l3046_304635


namespace interview_probability_l3046_304622

/-- The total number of students in at least one club -/
def total_students : ℕ := 30

/-- The number of students in the Robotics club -/
def robotics_students : ℕ := 22

/-- The number of students in the Drama club -/
def drama_students : ℕ := 19

/-- The probability of selecting two students who are not both from the same single club -/
theorem interview_probability : 
  (Nat.choose total_students 2 - (Nat.choose (robotics_students + drama_students - total_students) 2 + 
   Nat.choose (drama_students - (robotics_students + drama_students - total_students)) 2)) / 
  Nat.choose total_students 2 = 352 / 435 := by sorry

end interview_probability_l3046_304622


namespace student_A_most_stable_l3046_304612

/-- Represents a student with their score variance -/
structure Student where
  name : String
  variance : Real

/-- Theorem: Given the variances of four students' scores, prove that student A has the most stable performance -/
theorem student_A_most_stable
  (students : Finset Student)
  (hA : Student.mk "A" 3.8 ∈ students)
  (hB : Student.mk "B" 5.5 ∈ students)
  (hC : Student.mk "C" 10 ∈ students)
  (hD : Student.mk "D" 6 ∈ students)
  (h_count : students.card = 4)
  : ∀ s ∈ students, (Student.mk "A" 3.8).variance ≤ s.variance :=
by sorry


end student_A_most_stable_l3046_304612


namespace function_domain_range_implies_a_value_l3046_304616

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a^2 - 2*a - 3)*x^2 + (a - 3)*x + 1

-- State the theorem
theorem function_domain_range_implies_a_value :
  (∀ x, ∃ y, f a x = y) → -- Domain is ℝ
  (∀ y, ∃ x, f a x = y) → -- Range is ℝ
  a = -1 := by sorry

end function_domain_range_implies_a_value_l3046_304616


namespace jeremy_songs_theorem_l3046_304655

theorem jeremy_songs_theorem (x y : ℕ) : 
  x % 2 = 0 ∧ 
  9 = 2 * Int.sqrt x - 5 ∧ 
  y = (9 + x) / 2 → 
  9 + x + y = 110 := by
sorry

end jeremy_songs_theorem_l3046_304655


namespace negation_of_implication_is_true_l3046_304672

theorem negation_of_implication_is_true : 
  ¬(∀ a : ℝ, a ≤ 2 → a^2 < 4) := by sorry

end negation_of_implication_is_true_l3046_304672


namespace unit_digit_4137_754_l3046_304667

theorem unit_digit_4137_754 : (4137^754) % 10 = 9 := by
  sorry

end unit_digit_4137_754_l3046_304667


namespace parabola_intersection_l3046_304641

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 6 * x + 2
  let g (x : ℝ) := 9 * x^2 - 4 * x - 5
  (f (-7/3) = g (-7/3) ∧ f (-7/3) = 9) ∧
  (f (1/2) = g (1/2) ∧ f (1/2) = -1/4) :=
by sorry

end parabola_intersection_l3046_304641


namespace line_equation_through_points_l3046_304697

/-- A line passing through two points. -/
structure Line2D where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The equation of a line in the form ax + by + c = 0. -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: The equation of the line passing through (-1, 2) and (2, 5) is x - y + 3 = 0. -/
theorem line_equation_through_points :
  let l : Line2D := { point1 := (-1, 2), point2 := (2, 5) }
  let eq : LineEquation := { a := 1, b := -1, c := 3 }
  (∀ x y : ℝ, (x = l.point1.1 ∧ y = l.point1.2) ∨ (x = l.point2.1 ∧ y = l.point2.2) →
    eq.a * x + eq.b * y + eq.c = 0) ∧
  (∀ x y : ℝ, eq.a * x + eq.b * y + eq.c = 0 →
    ∃ t : ℝ, x = l.point1.1 + t * (l.point2.1 - l.point1.1) ∧
              y = l.point1.2 + t * (l.point2.2 - l.point1.2)) :=
by
  sorry


end line_equation_through_points_l3046_304697


namespace certain_number_equation_l3046_304607

theorem certain_number_equation (x : ℤ) : 34 + x - 53 = 28 ↔ x = 47 := by
  sorry

end certain_number_equation_l3046_304607


namespace polynomial_functional_equation_l3046_304640

theorem polynomial_functional_equation (a b c d : ℝ) :
  let f (x : ℝ) := a * x^3 + b * x^2 + c * x + d
  (∀ x, f x * f (-x) = f (x^3)) ↔ 
  ((a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1)) :=
by sorry

end polynomial_functional_equation_l3046_304640


namespace coffee_ounces_per_cup_l3046_304681

/-- Proves that the number of ounces of coffee per cup is 0.5 --/
theorem coffee_ounces_per_cup : 
  ∀ (people : ℕ) (cups_per_person_per_day : ℕ) (cost_per_ounce : ℚ) (weekly_expenditure : ℚ),
    people = 4 →
    cups_per_person_per_day = 2 →
    cost_per_ounce = 1.25 →
    weekly_expenditure = 35 →
    (weekly_expenditure / cost_per_ounce) / (people * cups_per_person_per_day * 7 : ℚ) = 0.5 := by
  sorry

end coffee_ounces_per_cup_l3046_304681


namespace larger_box_jellybean_count_l3046_304614

/-- The number of jellybeans in a box with dimensions thrice as large -/
def jellybeans_in_larger_box (original_capacity : ℕ) : ℕ :=
  original_capacity * 27

/-- Theorem: A box with dimensions thrice as large holds 4050 jellybeans -/
theorem larger_box_jellybean_count :
  jellybeans_in_larger_box 150 = 4050 := by
  sorry

end larger_box_jellybean_count_l3046_304614


namespace min_value_fraction_l3046_304649

theorem min_value_fraction (a : ℝ) (h1 : 0 < a) (h2 : a < 3) :
  1 / a + 9 / (3 - a) ≥ 16 / 3 := by
  sorry

end min_value_fraction_l3046_304649


namespace digit_interchange_effect_l3046_304658

theorem digit_interchange_effect (n : ℕ) (p q : ℕ) 
  (h1 : n = 9)
  (h2 : p < 10 ∧ q < 10)
  (h3 : p ≠ q)
  (original_sum : ℕ) 
  (new_sum : ℕ)
  (h4 : new_sum = original_sum - n)
  (h5 : new_sum = original_sum - (10*p + q - (10*q + p))) :
  p - q = 1 ∨ q - p = 1 :=
sorry

end digit_interchange_effect_l3046_304658


namespace smallest_of_three_powers_l3046_304627

theorem smallest_of_three_powers : 127^8 < 63^10 ∧ 63^10 < 33^12 := by
  sorry

end smallest_of_three_powers_l3046_304627


namespace smallest_circle_radius_l3046_304682

/-- Given a circle of radius r that touches two identical circles and a smaller circle,
    all externally tangent to each other, the radius of the smallest circle is r/6. -/
theorem smallest_circle_radius (r : ℝ) (hr : r > 0) : ∃ (r_small : ℝ), r_small = r / 6 :=
sorry

end smallest_circle_radius_l3046_304682


namespace spencer_walk_distance_l3046_304608

theorem spencer_walk_distance (total : ℝ) (house_to_library : ℝ) (post_office_to_home : ℝ)
  (h1 : total = 0.8)
  (h2 : house_to_library = 0.3)
  (h3 : post_office_to_home = 0.4) :
  total - house_to_library - post_office_to_home = 0.1 := by
  sorry

end spencer_walk_distance_l3046_304608


namespace chord_bisected_at_P_l3046_304656

/-- The equation of an ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 / 4 = 1

/-- A point is inside the ellipse if the left side of the equation is less than 1 -/
def inside_ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 / 4 < 1

/-- The fixed point P -/
def P : ℝ × ℝ := (1, 1)

/-- A chord is bisected at a point if that point is the midpoint of the chord -/
def is_bisected_at (A B M : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

/-- The equation of a line -/
def line_equation (x y : ℝ) : Prop := 2 * x + y - 3 = 0

theorem chord_bisected_at_P :
  inside_ellipse P.1 P.2 →
  ∀ A B : ℝ × ℝ,
    ellipse A.1 A.2 →
    ellipse B.1 B.2 →
    is_bisected_at A B P →
    ∀ x y : ℝ, (x, y) ∈ Set.Icc A B → line_equation x y :=
sorry

end chord_bisected_at_P_l3046_304656


namespace lineGraphMostSuitable_l3046_304664

/-- Represents different types of graphs --/
inductive GraphType
  | LineGraph
  | PieChart
  | BarGraph
  | Histogram

/-- Represents the properties of data to be visualized --/
structure DataProperties where
  timeDependent : Bool
  continuous : Bool
  showsTrends : Bool

/-- Determines if a graph type is suitable for given data properties --/
def isSuitable (g : GraphType) (d : DataProperties) : Prop :=
  match g with
  | GraphType.LineGraph => d.timeDependent ∧ d.continuous ∧ d.showsTrends
  | GraphType.PieChart => ¬d.timeDependent
  | GraphType.BarGraph => d.timeDependent
  | GraphType.Histogram => ¬d.timeDependent

/-- The properties of temperature data over a week --/
def temperatureDataProperties : DataProperties :=
  { timeDependent := true
    continuous := true
    showsTrends := true }

/-- Theorem stating that a line graph is the most suitable for temperature data --/
theorem lineGraphMostSuitable :
    ∀ g : GraphType, isSuitable g temperatureDataProperties → g = GraphType.LineGraph :=
  sorry


end lineGraphMostSuitable_l3046_304664


namespace hyperbola_proof_1_hyperbola_proof_2_l3046_304692

-- Part 1
def hyperbola_equation_1 (x y : ℝ) : Prop :=
  x^2 / 5 - y^2 = 1

theorem hyperbola_proof_1 (c : ℝ) (h1 : c = Real.sqrt 6) :
  hyperbola_equation_1 (-5) 2 ∧
  ∃ a b : ℝ, c^2 = a^2 + b^2 ∧ hyperbola_equation_1 x y ↔ x^2 / a^2 - y^2 / b^2 = 1 :=
sorry

-- Part 2
def hyperbola_equation_2 (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 9 = 1

theorem hyperbola_proof_2 :
  hyperbola_equation_2 3 (-4 * Real.sqrt 2) ∧
  hyperbola_equation_2 (9/4) 5 :=
sorry

end hyperbola_proof_1_hyperbola_proof_2_l3046_304692


namespace distance_calculation_l3046_304661

/-- The distance between A and B's homes from the city -/
def distance_difference : ℝ := 3

/-- The ratio of A's walking speed to B's walking speed -/
def walking_speed_ratio : ℝ := 1.5

/-- The ratio of B's truck speed to A's car speed -/
def vehicle_speed_ratio : ℝ := 1.5

/-- The ratio of A's car speed to A's walking speed -/
def car_to_walk_ratio : ℝ := 2

/-- B's distance from the city -/
def b_distance : ℝ := 13.5

/-- A's distance from the city -/
def a_distance : ℝ := 16.5

/-- The theorem stating that given the conditions, A lives 16.5 km from the city and B lives 13.5 km from the city -/
theorem distance_calculation :
  (a_distance - b_distance = distance_difference) ∧
  (a_distance = 16.5) ∧
  (b_distance = 13.5) := by
  sorry

#check distance_calculation

end distance_calculation_l3046_304661


namespace point_distance_on_x_axis_l3046_304617

theorem point_distance_on_x_axis (a : ℝ) : 
  let A : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (-3, 0)
  (‖A - B‖ = 5) → (a = -8 ∨ a = 2) :=
by sorry

end point_distance_on_x_axis_l3046_304617


namespace units_digit_square_equal_l3046_304659

theorem units_digit_square_equal (a b : ℕ) (h : (a % 10 + b % 10) = 10) : 
  (a^2 % 10) = (b^2 % 10) := by
sorry

end units_digit_square_equal_l3046_304659


namespace tank_circumference_l3046_304600

theorem tank_circumference (h_A h_B c_A : ℝ) (h_A_pos : h_A > 0) (h_B_pos : h_B > 0) (c_A_pos : c_A > 0) :
  h_A = 10 →
  h_B = 6 →
  c_A = 6 →
  (π * (c_A / (2 * π))^2 * h_A) = 0.6 * (π * (c_B / (2 * π))^2 * h_B) →
  c_B = 10 :=
by
  sorry

#check tank_circumference

end tank_circumference_l3046_304600


namespace isosceles_triangle_locus_l3046_304679

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a circle -/
def CircleEquation (center : Point) (radius : ℝ) (p : Point) : Prop :=
  (p.x - center.x)^2 + (p.y - center.y)^2 = radius^2

/-- The locus equation for point C -/
def LocusEquation (p : Point) : Prop :=
  p.x^2 + p.y^2 - 3*p.x + p.y = 2

theorem isosceles_triangle_locus :
  ∀ (C : Point),
    let A : Point := ⟨3, -2⟩
    let B : Point := ⟨0, 1⟩
    let M : Point := ⟨3/2, -1/2⟩  -- Midpoint of AB
    let r : ℝ := (3 * Real.sqrt 2) / 2  -- Radius of the circle
    C ≠ A ∧ C ≠ B →  -- Exclude points A and B
    (CircleEquation M r C ↔ LocusEquation C) :=
by sorry

end isosceles_triangle_locus_l3046_304679


namespace perfect_square_sums_l3046_304606

theorem perfect_square_sums : ∃ (x y : ℕ+), 
  (∃ (a : ℕ+), (x + y : ℕ) = a^2) ∧ 
  (∃ (b : ℕ+), (x^2 + y^2 : ℕ) = b^2) ∧ 
  (∃ (c : ℕ+), (x^3 + y^3 : ℕ) = c^2) := by
  sorry

end perfect_square_sums_l3046_304606


namespace peter_erasers_count_l3046_304611

def initial_erasers : ℕ := 8
def multiplier : ℕ := 3

theorem peter_erasers_count : 
  initial_erasers + multiplier * initial_erasers = 32 :=
by sorry

end peter_erasers_count_l3046_304611


namespace ned_chocolate_pieces_l3046_304637

theorem ned_chocolate_pieces : 
  ∀ (boxes_bought boxes_given pieces_per_box : ℝ),
    boxes_bought = 14.0 →
    boxes_given = 7.0 →
    pieces_per_box = 6.0 →
    (boxes_bought - boxes_given) * pieces_per_box = 42.0 := by
  sorry

end ned_chocolate_pieces_l3046_304637


namespace blue_ball_weight_l3046_304669

theorem blue_ball_weight (brown_weight total_weight : ℝ) 
  (h1 : brown_weight = 3.12)
  (h2 : total_weight = 9.12) :
  total_weight - brown_weight = 6 := by
  sorry

end blue_ball_weight_l3046_304669


namespace probability_after_removal_l3046_304642

theorem probability_after_removal (total : ℕ) (blue : ℕ) (removed : ℕ) 
  (h1 : total = 25)
  (h2 : blue = 9)
  (h3 : removed = 5)
  (h4 : removed < blue)
  (h5 : removed < total) :
  (blue - removed : ℚ) / (total - removed) = 1 / 5 := by
sorry

end probability_after_removal_l3046_304642


namespace graphing_to_scientific_ratio_l3046_304698

/-- Represents the cost of calculators and the transaction details -/
structure CalculatorPurchase where
  basic_cost : ℝ
  scientific_cost : ℝ
  graphing_cost : ℝ
  total_spent : ℝ

/-- The conditions of the calculator purchase problem -/
def calculator_problem : CalculatorPurchase :=
  { basic_cost := 8
  , scientific_cost := 16
  , graphing_cost := 72 - 8 - 16
  , total_spent := 100 - 28 }

/-- Theorem stating that the ratio of graphing to scientific calculator cost is 3:1 -/
theorem graphing_to_scientific_ratio :
  calculator_problem.graphing_cost / calculator_problem.scientific_cost = 3 := by
  sorry


end graphing_to_scientific_ratio_l3046_304698


namespace divisibility_problem_l3046_304645

theorem divisibility_problem (n : ℕ) (h1 : n > 0) (h2 : (n + 1) % 6 = 4) :
  n % 2 = 1 := by sorry

end divisibility_problem_l3046_304645


namespace train_speed_crossing_bridge_l3046_304638

/-- The speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 250) 
  (h2 : bridge_length = 300) 
  (h3 : crossing_time = 45) : 
  ∃ (speed : ℝ), abs (speed - (train_length + bridge_length) / crossing_time) < 0.01 :=
sorry

end train_speed_crossing_bridge_l3046_304638


namespace simplify_expression_l3046_304694

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  12 * x^5 * y / (6 * x * y) = 2 * x^4 := by
  sorry

end simplify_expression_l3046_304694


namespace cats_remaining_l3046_304666

def siamese_cats : ℕ := 38
def house_cats : ℕ := 25
def cats_sold : ℕ := 45

theorem cats_remaining : siamese_cats + house_cats - cats_sold = 18 := by
  sorry

end cats_remaining_l3046_304666


namespace min_packages_lcm_l3046_304657

/-- The load capacity of Sarah's trucks -/
def sarah_capacity : ℕ := 18

/-- The load capacity of Ryan's trucks -/
def ryan_capacity : ℕ := 11

/-- The load capacity of Emily's trucks -/
def emily_capacity : ℕ := 15

/-- The minimum number of packages each business must have shipped -/
def min_packages : ℕ := 990

theorem min_packages_lcm :
  Nat.lcm (Nat.lcm sarah_capacity ryan_capacity) emily_capacity = min_packages :=
sorry

end min_packages_lcm_l3046_304657


namespace total_jellybeans_needed_l3046_304651

/-- The number of jellybeans needed to fill a large glass -/
def large_glass_jellybeans : ℕ := 50

/-- The number of large glasses to be filled -/
def num_large_glasses : ℕ := 5

/-- The number of small glasses to be filled -/
def num_small_glasses : ℕ := 3

/-- The number of jellybeans needed to fill a small glass -/
def small_glass_jellybeans : ℕ := large_glass_jellybeans / 2

/-- Theorem: The total number of jellybeans needed to fill all glasses is 325 -/
theorem total_jellybeans_needed : 
  num_large_glasses * large_glass_jellybeans + num_small_glasses * small_glass_jellybeans = 325 := by
  sorry

end total_jellybeans_needed_l3046_304651


namespace children_creativity_center_contradiction_l3046_304665

theorem children_creativity_center_contradiction (N : ℕ) (d : Fin N → ℕ) : 
  N = 32 ∧ 
  (∀ i, d i = 6) ∧ 
  (∀ i j, i ≠ j → d i + d j = 13) → 
  False :=
sorry

end children_creativity_center_contradiction_l3046_304665


namespace pirates_walking_distance_l3046_304628

/-- The number of miles walked per day on the first two islands -/
def miles_per_day_first_two_islands (
  num_islands : ℕ)
  (days_per_island : ℚ)
  (miles_per_day_last_two : ℕ)
  (total_miles : ℕ) : ℚ :=
  (total_miles - 2 * (miles_per_day_last_two * days_per_island)) /
  (2 * days_per_island)

/-- Theorem stating that the miles walked per day on the first two islands is 20 -/
theorem pirates_walking_distance :
  miles_per_day_first_two_islands 4 (3/2) 25 135 = 20 := by
  sorry

end pirates_walking_distance_l3046_304628


namespace no_double_application_function_l3046_304650

theorem no_double_application_function : ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f (f x) = x + 1 := by
  sorry

end no_double_application_function_l3046_304650
