import Mathlib

namespace solutions_count_3x_2y_802_l4036_403631

theorem solutions_count_3x_2y_802 : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 2 * p.2 = 802 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 803) (Finset.range 402))).card = 133 := by
  sorry

end solutions_count_3x_2y_802_l4036_403631


namespace unsuitable_temp_l4036_403642

def storage_temp := -18
def temp_range := 2

def is_suitable_temp (temp : Int) : Prop :=
  (storage_temp - temp_range) ≤ temp ∧ temp ≤ (storage_temp + temp_range)

theorem unsuitable_temp :
  ¬(is_suitable_temp (-21)) :=
by
  sorry

end unsuitable_temp_l4036_403642


namespace science_fair_teams_l4036_403677

theorem science_fair_teams (total_students : Nat) (red_hats : Nat) (green_hats : Nat) 
  (total_teams : Nat) (red_red_teams : Nat) : 
  total_students = 144 →
  red_hats = 63 →
  green_hats = 81 →
  total_teams = 72 →
  red_red_teams = 28 →
  red_hats + green_hats = total_students →
  ∃ (green_green_teams : Nat), green_green_teams = 37 ∧ 
    green_green_teams + red_red_teams + (total_students - 2 * red_red_teams - 2 * green_green_teams) / 2 = total_teams :=
by
  sorry

end science_fair_teams_l4036_403677


namespace total_passengers_taking_l4036_403684

/-- Represents a train type with its characteristics -/
structure TrainType where
  interval : ℕ  -- Arrival interval in minutes
  leaving : ℕ   -- Number of passengers leaving
  taking : ℕ    -- Number of passengers taking

/-- Calculates the number of trains per hour given the arrival interval -/
def trainsPerHour (interval : ℕ) : ℕ := 60 / interval

/-- Calculates the total passengers for a given operation (leaving or taking) per hour -/
def totalPassengers (t : TrainType) (op : TrainType → ℕ) : ℕ :=
  (trainsPerHour t.interval) * (op t)

/-- Theorem: The total number of unique passengers taking trains at each station during an hour is 4360 -/
theorem total_passengers_taking (stationCount : ℕ) (type1 type2 type3 : TrainType) :
  stationCount = 4 →
  type1 = { interval := 10, leaving := 200, taking := 320 } →
  type2 = { interval := 15, leaving := 300, taking := 400 } →
  type3 = { interval := 20, leaving := 150, taking := 280 } →
  (totalPassengers type1 TrainType.taking +
   totalPassengers type2 TrainType.taking +
   totalPassengers type3 TrainType.taking) = 4360 :=
by sorry

end total_passengers_taking_l4036_403684


namespace exists_number_divisible_by_5_pow_1000_without_zero_l4036_403624

theorem exists_number_divisible_by_5_pow_1000_without_zero : ∃ n : ℕ, 
  (5^1000 ∣ n) ∧ 
  (∀ d : ℕ, d < 10 → (n.digits 10).all (λ digit => digit ≠ d) → d ≠ 0) := by
  sorry

end exists_number_divisible_by_5_pow_1000_without_zero_l4036_403624


namespace circle_center_l4036_403679

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + 8*x + y^2 - 4*y = 16

/-- The center of a circle -/
def CircleCenter (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 36

theorem circle_center :
  CircleCenter (-4) 2 :=
sorry

end circle_center_l4036_403679


namespace inscribed_pentagon_external_angles_sum_inscribed_pentagon_external_angles_sum_is_720_l4036_403608

/-- Represents a pentagon inscribed in a circle -/
structure InscribedPentagon where
  -- We don't need to define the specific properties of the pentagon,
  -- as the problem doesn't require detailed information about its structure

/-- 
Theorem: For a pentagon inscribed in a circle, the sum of the angles
inscribed in the five segments outside the pentagon but inside the circle
is equal to 720°.
-/
theorem inscribed_pentagon_external_angles_sum
  (p : InscribedPentagon) : Real :=
  720

/-- 
Main theorem: The sum of the angles inscribed in the five segments
outside an inscribed pentagon but inside the circle is 720°.
-/
theorem inscribed_pentagon_external_angles_sum_is_720
  (p : InscribedPentagon) :
  inscribed_pentagon_external_angles_sum p = 720 := by
  sorry

end inscribed_pentagon_external_angles_sum_inscribed_pentagon_external_angles_sum_is_720_l4036_403608


namespace intersection_locus_l4036_403641

theorem intersection_locus (m : ℝ) (x y : ℝ) : 
  (m * x - y + 1 = 0 ∧ x - m * y - 1 = 0) → 
  (x - y = 0 ∨ x - y + 1 = 0) := by
  sorry

end intersection_locus_l4036_403641


namespace initial_blue_balls_l4036_403664

theorem initial_blue_balls (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 15 → 
  removed = 3 → 
  prob = 1/3 → 
  (total - removed : ℚ) * prob = (total - removed - (total - removed - prob * (total - removed))) → 
  total - removed - (total - removed - prob * (total - removed)) + removed = 7 := by
  sorry

end initial_blue_balls_l4036_403664


namespace union_of_A_and_B_l4036_403634

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2 ≥ 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≤ 0}

-- State the theorem
theorem union_of_A_and_B :
  A ∪ B = {x | x ≤ -Real.sqrt 2 ∨ x ≥ 1} := by
  sorry

end union_of_A_and_B_l4036_403634


namespace hyperbola_and_slopes_l4036_403670

-- Define the hyperbola E
def E (b : ℝ) (x y : ℝ) : Prop := x^2 - y^2/b^2 = 1

-- Define point P
def P : ℝ × ℝ := (-2, -3)

-- Define point Q
def Q : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem hyperbola_and_slopes 
  (b : ℝ) 
  (h1 : b > 0) 
  (h2 : E b P.1 P.2) 
  (A B : ℝ × ℝ) 
  (h3 : A ≠ P ∧ B ≠ P ∧ A ≠ B) 
  (h4 : ∃ k : ℝ, A.2 = k * A.1 - 1 ∧ B.2 = k * B.1 - 1) 
  (h5 : E b A.1 A.2 ∧ E b B.1 B.2) :
  (b^2 = 3) ∧ 
  (((A.2 - P.2) / (A.1 - P.1)) + ((B.2 - P.2) / (B.1 - P.1)) = 3) :=
sorry

end hyperbola_and_slopes_l4036_403670


namespace triangle_perimeter_is_17_l4036_403601

-- Define the triangle ABC
structure Triangle :=
  (a b c : ℝ)

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (t.b - 5)^2 + |t.c - 7| = 0 ∧ |t.a - 3| = 2

-- Define the perimeter
def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Theorem statement
theorem triangle_perimeter_is_17 :
  ∀ t : Triangle, satisfies_conditions t → perimeter t = 17 :=
by
  sorry

end triangle_perimeter_is_17_l4036_403601


namespace coefficient_of_x_in_expansion_l4036_403691

-- Define the binomial expansion function
def binomialCoefficient (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x
def coefficientOfX (a b : ℤ) (n : ℕ) : ℤ :=
  binomialCoefficient n 2 * (b ^ 2)

-- Theorem statement
theorem coefficient_of_x_in_expansion :
  coefficientOfX 1 (-2) 5 = 40 := by sorry

end coefficient_of_x_in_expansion_l4036_403691


namespace square_sum_difference_l4036_403630

theorem square_sum_difference (n : ℕ) : n^2 + (n+1)^2 - (n+2)^2 = n*(n-2) - 3 := by
  sorry

end square_sum_difference_l4036_403630


namespace fraction_product_subtraction_l4036_403610

theorem fraction_product_subtraction : (1/2 : ℚ) * (1/3 : ℚ) * (1/6 : ℚ) * 72 - 2 = 0 := by
  sorry

end fraction_product_subtraction_l4036_403610


namespace meaningful_fraction_condition_l4036_403635

theorem meaningful_fraction_condition (x : ℝ) :
  (∃ y, y = (x - 1) / (x + 1)) ↔ x ≠ -1 :=
sorry

end meaningful_fraction_condition_l4036_403635


namespace max_non_fiction_books_l4036_403696

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_non_fiction_books :
  ∀ (fiction non_fiction : ℕ) (p : ℕ),
    fiction + non_fiction = 100 →
    fiction = non_fiction + p →
    is_prime p →
    non_fiction ≤ 49 :=
by sorry

end max_non_fiction_books_l4036_403696


namespace milk_needed_for_recipe_l4036_403614

-- Define the ratio of milk to flour
def milk_to_flour_ratio : ℚ := 75 / 250

-- Define the amount of flour Luca wants to use
def flour_amount : ℚ := 1250

-- Theorem: The amount of milk needed for 1250 mL of flour is 375 mL
theorem milk_needed_for_recipe : 
  milk_to_flour_ratio * flour_amount = 375 := by
  sorry


end milk_needed_for_recipe_l4036_403614


namespace illumination_theorem_l4036_403619

/-- Represents a direction: North, South, East, or West -/
inductive Direction
  | North
  | South
  | East
  | West

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a spotlight with a position and direction -/
structure Spotlight where
  position : Point
  direction : Direction

/-- Represents the configuration of 4 spotlights -/
def SpotlightConfiguration := Fin 4 → Spotlight

/-- Checks if a point is illuminated by a spotlight -/
def isIlluminated (p : Point) (s : Spotlight) : Prop :=
  match s.direction with
  | Direction.North => p.y ≥ s.position.y
  | Direction.South => p.y ≤ s.position.y
  | Direction.East => p.x ≥ s.position.x
  | Direction.West => p.x ≤ s.position.x

/-- The main theorem: there exists a configuration that illuminates the entire plane -/
theorem illumination_theorem (p1 p2 p3 p4 : Point) :
  ∃ (config : SpotlightConfiguration),
    ∀ (p : Point), ∃ (i : Fin 4), isIlluminated p (config i) := by
  sorry


end illumination_theorem_l4036_403619


namespace land_tax_calculation_l4036_403644

/-- Calculates the land tax for a given plot --/
def calculate_land_tax (area : ℝ) (cadastral_value_per_acre : ℝ) (tax_rate : ℝ) : ℝ :=
  area * cadastral_value_per_acre * tax_rate

/-- Proves that the land tax for the given conditions is 4500 rubles --/
theorem land_tax_calculation :
  let area : ℝ := 15
  let cadastral_value_per_acre : ℝ := 100000
  let tax_rate : ℝ := 0.003
  calculate_land_tax area cadastral_value_per_acre tax_rate = 4500 := by
  sorry

#eval calculate_land_tax 15 100000 0.003

end land_tax_calculation_l4036_403644


namespace base9_multiplication_l4036_403600

/-- Represents a number in base 9 --/
def Base9 : Type := ℕ

/-- Converts a base 9 number to a natural number --/
def to_nat (x : Base9) : ℕ := sorry

/-- Converts a natural number to a base 9 number --/
def from_nat (n : ℕ) : Base9 := sorry

/-- Multiplication operation for Base9 numbers --/
def mul_base9 (x y : Base9) : Base9 := sorry

theorem base9_multiplication :
  mul_base9 (from_nat 362) (from_nat 7) = from_nat 2875 :=
sorry

end base9_multiplication_l4036_403600


namespace square_sum_eighteen_l4036_403653

theorem square_sum_eighteen (x y : ℝ) 
  (h1 : y + 9 = (x - 3)^3)
  (h2 : x + 9 = (y - 3)^3)
  (h3 : x ≠ y) : 
  x^2 + y^2 = 18 := by
  sorry

end square_sum_eighteen_l4036_403653


namespace sqrt_product_plus_one_equals_3994001_l4036_403650

theorem sqrt_product_plus_one_equals_3994001 :
  Real.sqrt (1997 * 1998 * 1999 * 2000 + 1) = 3994001 := by
  sorry

end sqrt_product_plus_one_equals_3994001_l4036_403650


namespace leftover_apples_for_ivan_l4036_403609

/-- Given a number of initial apples and mini pies, calculate the number of leftover apples -/
def leftover_apples (initial_apples : ℕ) (mini_pies : ℕ) : ℕ :=
  initial_apples - (mini_pies / 2)

/-- Theorem: Given 48 initial apples and 24 mini pies, each requiring 1/2 an apple, 
    the number of leftover apples is 36 -/
theorem leftover_apples_for_ivan : leftover_apples 48 24 = 36 := by
  sorry

end leftover_apples_for_ivan_l4036_403609


namespace chess_tournament_participants_l4036_403697

theorem chess_tournament_participants (total_games : ℕ) 
  (h1 : total_games = 105) : ∃ n : ℕ, n > 0 ∧ n * (n - 1) / 2 = total_games := by
  sorry

end chess_tournament_participants_l4036_403697


namespace arithmetic_sequence_properties_l4036_403612

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  first_term : a 1 = 1
  third_term : a 3 = -3
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Theorem about the general formula and sum of the sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 3 - 2 * n) ∧
  (∃ k : ℕ, k * (seq.a 1 + seq.a k) / 2 = -35 ∧ k = 7) :=
sorry

end arithmetic_sequence_properties_l4036_403612


namespace remainder_theorem_l4036_403667

-- Define the polynomial q(x)
def q (D E F : ℝ) (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 5

-- State the theorem
theorem remainder_theorem (D E F : ℝ) :
  (∃ k : ℝ, q D E F x = (x - 2) * k + 15) →
  (∃ m : ℝ, q D E F x = (x + 2) * m + 15) :=
by sorry

end remainder_theorem_l4036_403667


namespace quadratic_equation_solutions_l4036_403655

/-- The number of integer solutions to the equation 2x^2 + 5xy + 3y^2 = 30 -/
def num_solutions : ℕ := 16

/-- The quadratic equation -/
def quadratic_equation (x y : ℤ) : Prop :=
  2 * x^2 + 5 * x * y + 3 * y^2 = 30

/-- Known solution to the equation -/
def known_solution : ℤ × ℤ := (9, -4)

theorem quadratic_equation_solutions :
  (quadratic_equation known_solution.1 known_solution.2) ∧
  (∃ (solutions : Finset (ℤ × ℤ)), 
    solutions.card = num_solutions ∧
    ∀ (sol : ℤ × ℤ), sol ∈ solutions ↔ quadratic_equation sol.1 sol.2) :=
by sorry

end quadratic_equation_solutions_l4036_403655


namespace altitude_length_is_one_l4036_403648

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point lies on the parabola y = x^2 -/
def onParabola (p : Point) : Prop :=
  p.y = p.x^2

/-- Checks if a line segment is parallel to the x-axis -/
def parallelToXAxis (p1 p2 : Point) : Prop :=
  p1.y = p2.y

/-- Checks if a triangle is right-angled -/
def isRightTriangle (t : Triangle) : Prop :=
  let AC := (t.C.x - t.A.x, t.C.y - t.A.y)
  let BC := (t.C.x - t.B.x, t.C.y - t.B.y)
  AC.1 * BC.1 + AC.2 * BC.2 = 0

/-- Calculates the length of the altitude from C to AB -/
def altitudeLength (t : Triangle) : ℝ :=
  t.A.y - t.C.y

/-- The main theorem -/
theorem altitude_length_is_one (t : Triangle) :
  isRightTriangle t →
  onParabola t.A ∧ onParabola t.B ∧ onParabola t.C →
  parallelToXAxis t.A t.B →
  altitudeLength t = 1 := by
  sorry

end altitude_length_is_one_l4036_403648


namespace bug_return_probability_l4036_403689

/-- Represents the probability of the bug being at its starting vertex after n moves -/
def P : ℕ → ℚ
| 0 => 1
| n + 1 => (2 : ℚ) / 3 * P n

/-- The probability of returning to the starting vertex on the 10th move -/
def probability_10th_move : ℚ := P 10

theorem bug_return_probability :
  probability_10th_move = 1024 / 59049 := by
  sorry

end bug_return_probability_l4036_403689


namespace triangle_angle_relations_l4036_403688

theorem triangle_angle_relations (a b c : ℝ) (α β γ : ℝ) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ 
  (0 < α) ∧ (0 < β) ∧ (0 < γ) ∧ 
  (α + β + γ = Real.pi) ∧
  (c^2 = a^2 + 2 * b^2 * Real.cos β) →
  ((γ = β / 2 + Real.pi / 2 ∧ α = Real.pi / 2 - 3 * β / 2 ∧ 0 < β ∧ β < Real.pi / 3) ∨
   (α = β / 2 ∧ γ = Real.pi - 3 * β / 2 ∧ 0 < β ∧ β < 2 * Real.pi / 3)) :=
by sorry

end triangle_angle_relations_l4036_403688


namespace dog_owners_count_l4036_403621

-- Define the sets of people owning each type of pet
def C : Finset ℕ := sorry
def D : Finset ℕ := sorry
def R : Finset ℕ := sorry

-- Define the theorem
theorem dog_owners_count :
  (C ∪ D ∪ R).card = 60 ∧
  C.card = 30 ∧
  R.card = 16 ∧
  ((C ∩ D) ∪ (C ∩ R) ∪ (D ∩ R)).card - (C ∩ D ∩ R).card = 12 ∧
  (C ∩ D ∩ R).card = 7 →
  D.card = 40 := by
sorry


end dog_owners_count_l4036_403621


namespace camp_children_count_l4036_403633

/-- The number of children currently in the camp -/
def current_children : ℕ := 25

/-- The percentage of boys currently in the camp -/
def boys_percentage : ℚ := 85/100

/-- The number of boys to be added -/
def boys_added : ℕ := 50

/-- The desired percentage of girls after adding boys -/
def desired_girls_percentage : ℚ := 5/100

theorem camp_children_count :
  (boys_percentage * current_children).num = 
    (desired_girls_percentage * (current_children + boys_added)).num * 
    ((1 - boys_percentage) * current_children).den := by sorry

end camp_children_count_l4036_403633


namespace trivia_team_groups_l4036_403603

theorem trivia_team_groups (total : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total = 17) 
  (h2 : not_picked = 5) 
  (h3 : num_groups = 3) :
  (total - not_picked) / num_groups = 4 :=
by sorry

end trivia_team_groups_l4036_403603


namespace tangent_line_constraint_l4036_403660

theorem tangent_line_constraint (a : ℝ) : 
  (∀ b : ℝ, ¬∃ x : ℝ, (x^3 - 3*a*x + x = b ∧ 3*x^2 - 3*a = -1)) → 
  a < 1/3 := by
sorry

end tangent_line_constraint_l4036_403660


namespace coach_number_divisibility_l4036_403693

/-- A function that checks if a number is of the form aabb, abba, or abab -/
def isValidFormat (n : ℕ) : Prop :=
  ∃ a b : ℕ, 
    (n = a * 1000 + a * 100 + b * 10 + b) ∨ 
    (n = a * 1000 + b * 100 + b * 10 + a) ∨ 
    (n = a * 1000 + b * 100 + a * 10 + b)

/-- The set of possible ages of the children -/
def childrenAges : Set ℕ := {3, 4, 5, 6, 7, 8, 9, 10, 11}

/-- The theorem to be proved -/
theorem coach_number_divisibility 
  (N : ℕ) 
  (h1 : isValidFormat N) 
  (h2 : ∀ (x : ℕ), x ∈ childrenAges → x ≠ 10 → N % x = 0) 
  (h3 : N % 10 ≠ 0) 
  (h4 : 1000 ≤ N ∧ N < 10000) : 
  ∃ (a b : ℕ), N = 7000 + 700 + 40 + 4 := by
  sorry

end coach_number_divisibility_l4036_403693


namespace quadratic_rewrite_sum_l4036_403695

theorem quadratic_rewrite_sum (x : ℝ) : ∃ (a b c : ℝ),
  (6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) ∧ (a + b + c = 171) := by
  sorry

end quadratic_rewrite_sum_l4036_403695


namespace gcf_and_multiples_of_90_and_135_l4036_403636

theorem gcf_and_multiples_of_90_and_135 :
  ∃ (gcf : ℕ), 
    (Nat.gcd 90 135 = gcf) ∧ 
    (gcf = 45) ∧
    (45 ∣ gcf) ∧ 
    (90 ∣ gcf) ∧ 
    (135 ∣ gcf) := by
  sorry

end gcf_and_multiples_of_90_and_135_l4036_403636


namespace smaller_number_problem_l4036_403654

theorem smaller_number_problem (a b : ℝ) (h1 : a + b = 60) (h2 : a - b = 10) : min a b = 25 := by
  sorry

end smaller_number_problem_l4036_403654


namespace powderman_distance_approximation_l4036_403687

/-- The speed of the powderman in yards per second -/
def powderman_speed : ℝ := 8

/-- The time in seconds when the powderman hears the blast -/
def time_of_hearing : ℝ := 30.68

/-- The distance the powderman runs in yards -/
def distance_run : ℝ := powderman_speed * time_of_hearing

theorem powderman_distance_approximation :
  ∃ ε > 0, abs (distance_run - 245) < ε := by sorry

end powderman_distance_approximation_l4036_403687


namespace total_ants_is_twenty_l4036_403627

/-- The number of ants found by Abe -/
def abe_ants : ℕ := 4

/-- The number of ants found by Beth -/
def beth_ants : ℕ := abe_ants + abe_ants / 2

/-- The number of ants found by CeCe -/
def cece_ants : ℕ := 2 * abe_ants

/-- The number of ants found by Duke -/
def duke_ants : ℕ := abe_ants / 2

/-- The total number of ants found by all four children -/
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

/-- Theorem stating that the total number of ants found is 20 -/
theorem total_ants_is_twenty : total_ants = 20 := by sorry

end total_ants_is_twenty_l4036_403627


namespace rectangle_area_l4036_403611

/-- Given a rectangle composed of 24 congruent squares arranged in a 6x4 format
    with a diagonal of 10 cm, the total area of the rectangle is 2400/13 square cm. -/
theorem rectangle_area (squares : ℕ) (rows cols : ℕ) (diagonal : ℝ) :
  squares = 24 →
  rows = 6 →
  cols = 4 →
  diagonal = 10 →
  (rows * cols : ℝ) * (diagonal^2 / (rows^2 + cols^2)) = 2400 / 13 := by
  sorry

end rectangle_area_l4036_403611


namespace min_sum_of_product_1176_l4036_403617

theorem min_sum_of_product_1176 (a b c : ℕ+) (h : a * b * c = 1176) :
  (∀ x y z : ℕ+, x * y * z = 1176 → a + b + c ≤ x + y + z) →
  a + b + c = 59 :=
sorry

end min_sum_of_product_1176_l4036_403617


namespace initial_chairs_count_l4036_403638

theorem initial_chairs_count (initial_chairs : ℕ) 
  (h1 : initial_chairs - (initial_chairs - 3) = 12) : initial_chairs = 15 := by
  sorry

end initial_chairs_count_l4036_403638


namespace bottle_production_l4036_403646

/-- Given that 6 identical machines produce 240 bottles per minute at a constant rate,
    prove that 10 such machines will produce 1600 bottles in 4 minutes. -/
theorem bottle_production
  (machines : ℕ)
  (bottles_per_minute : ℕ)
  (h1 : machines = 6)
  (h2 : bottles_per_minute = 240)
  (constant_rate : ℕ → ℕ → ℕ) -- Function to calculate production based on number of machines and time
  (h3 : constant_rate machines 1 = bottles_per_minute) -- Production rate for given machines in 1 minute
  : constant_rate 10 4 = 1600 := by
  sorry


end bottle_production_l4036_403646


namespace expansion_coefficients_l4036_403620

theorem expansion_coefficients (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) 
  (h : (2*(x-1)-1)^9 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                       a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9) : 
  a₂ = -144 ∧ a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = 2 := by
sorry

end expansion_coefficients_l4036_403620


namespace percentage_women_after_hiring_l4036_403604

/-- Percentage of women in a multinational company after new hires --/
theorem percentage_women_after_hiring (country_a_initial : ℕ) (country_b_initial : ℕ)
  (country_a_men_ratio : ℚ) (country_b_women_ratio : ℚ)
  (country_a_new_hires : ℕ) (country_b_new_hires : ℕ)
  (country_a_new_men_ratio : ℚ) (country_b_new_women_ratio : ℚ)
  (h1 : country_a_initial = 90)
  (h2 : country_b_initial = 150)
  (h3 : country_a_men_ratio = 2/3)
  (h4 : country_b_women_ratio = 3/5)
  (h5 : country_a_new_hires = 5)
  (h6 : country_b_new_hires = 8)
  (h7 : country_a_new_men_ratio = 3/5)
  (h8 : country_b_new_women_ratio = 1/2) :
  ∃ (percentage : ℚ), abs (percentage - 4980/10000) < 1/1000 ∧
  percentage = (country_a_initial * (1 - country_a_men_ratio) + country_b_initial * country_b_women_ratio +
    country_a_new_hires * (1 - country_a_new_men_ratio) + country_b_new_hires * country_b_new_women_ratio) /
    (country_a_initial + country_b_initial + country_a_new_hires + country_b_new_hires) * 100 :=
by
  sorry


end percentage_women_after_hiring_l4036_403604


namespace salt_mixture_theorem_l4036_403649

/-- Represents the salt mixture problem -/
def SaltMixture (cheap_price cheap_weight expensive_price expensive_weight profit_percentage : ℚ) : Prop :=
  let total_cost : ℚ := cheap_price * cheap_weight + expensive_price * expensive_weight
  let total_weight : ℚ := cheap_weight + expensive_weight
  let profit : ℚ := total_cost * (profit_percentage / 100)
  let selling_price : ℚ := total_cost + profit
  let selling_price_per_pound : ℚ := selling_price / total_weight
  selling_price_per_pound = 48 / 100

/-- The salt mixture theorem -/
theorem salt_mixture_theorem : SaltMixture (38/100) 40 (50/100) 8 20 := by
  sorry

end salt_mixture_theorem_l4036_403649


namespace differential_equation_solution_l4036_403699

/-- The general solution to the differential equation dr - r dφ = 0 -/
theorem differential_equation_solution (r φ : ℝ → ℝ) (C : ℝ) :
  (∀ t, (deriv r t) - r t * (deriv φ t) = 0) ↔
  ∃ C, C > 0 ∧ ∀ t, r t = C * Real.exp (φ t) :=
sorry

end differential_equation_solution_l4036_403699


namespace abs_equation_unique_solution_l4036_403629

theorem abs_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x - 3| :=
by
  sorry

end abs_equation_unique_solution_l4036_403629


namespace geometric_sequence_special_case_l4036_403665

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 6 * a 6 - 8 * a 6 + 4 = 0) →
  (a 10 * a 10 - 8 * a 10 + 4 = 0) →
  (a 8 = 2 ∨ a 8 = -2) :=
by sorry

end geometric_sequence_special_case_l4036_403665


namespace each_person_share_l4036_403661

/-- The cost to send a person to Mars in billions of dollars -/
def mars_cost : ℚ := 30

/-- The cost to establish a base on the Moon in billions of dollars -/
def moon_base_cost : ℚ := 10

/-- The number of people sharing the cost in millions -/
def number_of_people : ℚ := 200

/-- The total cost in billions of dollars -/
def total_cost : ℚ := mars_cost + moon_base_cost

/-- Theorem: Each person's share of the total cost is $200 -/
theorem each_person_share :
  (total_cost * 1000) / number_of_people = 200 := by sorry

end each_person_share_l4036_403661


namespace sum_9_is_27_l4036_403698

/-- An arithmetic sequence on a line through (5,3) -/
structure ArithmeticSequenceOnLine where
  a : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  on_line : ∀ n : ℕ+, ∃ k m : ℚ, a n = k * n + m ∧ 3 = k * 5 + m

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequenceOnLine) (n : ℕ+) : ℚ :=
  (n : ℚ) * seq.a n

/-- The sum of the first 9 terms of an arithmetic sequence on a line through (5,3) is 27 -/
theorem sum_9_is_27 (seq : ArithmeticSequenceOnLine) : sum_n seq 9 = 27 := by
  sorry

end sum_9_is_27_l4036_403698


namespace rectangle_max_area_l4036_403640

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) → 
  l * w = 100 :=
by sorry

end rectangle_max_area_l4036_403640


namespace intersection_problem_l4036_403616

/-- The problem statement as a theorem -/
theorem intersection_problem (m b k : ℝ) : 
  b ≠ 0 →
  7 = 2 * m + b →
  (∃ y₁ y₂ : ℝ, 
    y₁ = k^2 + 8*k + 7 ∧
    y₂ = m*k + b ∧
    |y₁ - y₂| = 4) →
  m = 6 ∧ b = -5 := by
sorry

end intersection_problem_l4036_403616


namespace periodic_binomial_remainder_l4036_403623

theorem periodic_binomial_remainder (K : ℕ+) : 
  (∃ (p : ℕ+), ∀ (n : ℕ), n ≥ p → 
    (∃ (T : ℕ+), ∀ (m : ℕ), m ≥ p → 
      (Nat.choose (2*(n+m)) (n+m)) % K = (Nat.choose (2*n) n) % K)) ↔ 
  (K = 1 ∨ K = 2) :=
sorry

end periodic_binomial_remainder_l4036_403623


namespace incorrect_propositions_are_one_and_three_l4036_403606

-- Define a proposition as a structure with an id and a correctness value
structure Proposition :=
  (id : Nat)
  (isCorrect : Bool)

-- Define our set of propositions
def propositions : List Proposition := [
  ⟨1, false⟩,  -- Three points determine a plane
  ⟨2, true⟩,   -- A rectangle is a plane figure
  ⟨3, false⟩,  -- Three lines intersecting in pairs determine a plane
  ⟨4, true⟩    -- Two intersecting planes divide the space into four regions
]

-- Define a function to get incorrect propositions
def getIncorrectPropositions (props : List Proposition) : List Nat :=
  (props.filter (λ p => !p.isCorrect)).map Proposition.id

-- Theorem statement
theorem incorrect_propositions_are_one_and_three :
  getIncorrectPropositions propositions = [1, 3] := by
  sorry

end incorrect_propositions_are_one_and_three_l4036_403606


namespace nested_cube_root_l4036_403663

theorem nested_cube_root (N : ℝ) (h : N > 1) :
  (N * (N * (N * N^(1/3))^(1/3))^(1/3))^(1/3) = N^(40/81) := by
  sorry

end nested_cube_root_l4036_403663


namespace anna_money_left_l4036_403669

def original_amount : ℚ := 32
def spent_fraction : ℚ := 1/4

theorem anna_money_left : 
  (1 - spent_fraction) * original_amount = 24 := by
  sorry

end anna_money_left_l4036_403669


namespace base_8_6_equivalence_l4036_403668

theorem base_8_6_equivalence :
  ∀ (n : ℕ), n > 0 →
  (∃ (C D : ℕ),
    C < 8 ∧ D < 8 ∧
    D < 6 ∧
    n = 8 * C + D ∧
    n = 6 * D + C) →
  n = 0 :=
by sorry

end base_8_6_equivalence_l4036_403668


namespace x_twelfth_power_l4036_403671

theorem x_twelfth_power (x : ℝ) (h : x + 1/x = 2) : x^12 = 1 := by
  sorry

end x_twelfth_power_l4036_403671


namespace sqrt_expression_equals_negative_seven_l4036_403615

theorem sqrt_expression_equals_negative_seven :
  (Real.sqrt 15)^2 / Real.sqrt 3 * (1 / Real.sqrt 3) - Real.sqrt 6 * Real.sqrt 24 = -7 := by
  sorry

end sqrt_expression_equals_negative_seven_l4036_403615


namespace sum_product_bounds_l4036_403639

theorem sum_product_bounds (x y z : ℝ) (h : x + y + z = 3) :
  -3/2 ≤ x*y + x*z + y*z ∧ x*y + x*z + y*z ≤ 3 :=
by sorry

end sum_product_bounds_l4036_403639


namespace fraction_simplification_l4036_403602

theorem fraction_simplification :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 3 + 5 * Real.sqrt 48) = (5 * Real.sqrt 3) / 84 := by
  sorry

end fraction_simplification_l4036_403602


namespace polynomial_nonnegative_iff_equal_roots_l4036_403678

theorem polynomial_nonnegative_iff_equal_roots (a b c : ℝ) :
  (∀ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) ≥ 0) ↔ 
  (a = b ∧ b = c) :=
by sorry

end polynomial_nonnegative_iff_equal_roots_l4036_403678


namespace white_animals_count_l4036_403645

theorem white_animals_count (total : ℕ) (black : ℕ) (white : ℕ) : 
  total = 13 → black = 6 → white = total - black → white = 7 := by sorry

end white_animals_count_l4036_403645


namespace min_value_expression_l4036_403692

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y > 2 * x) :
  ∃ (m : ℝ), m = 4 ∧ ∀ (z : ℝ), z = (y^2 - 2*x*y + x^2) / (x*y - 2*x^2) → z ≥ m :=
by sorry

end min_value_expression_l4036_403692


namespace solution_pairs_l4036_403613

theorem solution_pairs : ∃! (s : Set (ℝ × ℝ)), 
  s = {(1 + Real.sqrt 2, 1 - Real.sqrt 2), (1 - Real.sqrt 2, 1 + Real.sqrt 2)} ∧
  ∀ (x y : ℝ), (x, y) ∈ s ↔ 
    (x^2 + y^2 = (6 - x^2) + (6 - y^2)) ∧ 
    (x^2 - y^2 = (x - 2)^2 + (y - 2)^2) := by
  sorry

end solution_pairs_l4036_403613


namespace sum_of_costs_equals_power_l4036_403656

/-- An antipalindromic sequence of A's and B's -/
def AntipalindromicSequence : Type := List Bool

/-- The cost of a sequence is the product of positions of A's -/
def cost (seq : AntipalindromicSequence) : ℕ := sorry

/-- The set of all antipalindromic sequences of length 2020 -/
def allAntipalindromic2020 : Set AntipalindromicSequence := sorry

/-- The sum of costs of all antipalindromic sequences of length 2020 -/
def sumOfCosts : ℕ := sorry

/-- Main theorem: The sum of costs equals 2021^1010 -/
theorem sum_of_costs_equals_power :
  sumOfCosts = 2021^1010 := by sorry

end sum_of_costs_equals_power_l4036_403656


namespace translation_increases_y_l4036_403685

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a horizontal translation of a function -/
structure HorizontalTranslation where
  units : ℝ

/-- The original quadratic function y = -x^2 + 1 -/
def original_function : QuadraticFunction :=
  { a := -1, b := 0, c := 1 }

/-- The required translation -/
def translation : HorizontalTranslation :=
  { units := 2 }

/-- Theorem stating that the given translation makes y increase as x increases when x < 2 -/
theorem translation_increases_y (f : QuadraticFunction) (t : HorizontalTranslation) :
  f = original_function →
  t = translation →
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → x₂ < 2 →
  f.a * (x₁ - t.units)^2 + f.b * (x₁ - t.units) + f.c <
  f.a * (x₂ - t.units)^2 + f.b * (x₂ - t.units) + f.c :=
by sorry

end translation_increases_y_l4036_403685


namespace house_length_calculation_l4036_403626

/-- Given a house with width 10 feet and a porch measuring 6 feet by 4.5 feet,
    if 232 square feet of shingles are needed to roof both the house and the porch,
    then the length of the house is 20.5 feet. -/
theorem house_length_calculation (house_width porch_length porch_width total_shingle_area : ℝ) :
  house_width = 10 →
  porch_length = 6 →
  porch_width = 4.5 →
  total_shingle_area = 232 →
  ∃ house_length : ℝ,
    house_length * house_width + porch_length * porch_width = total_shingle_area ∧
    house_length = 20.5 :=
by sorry

end house_length_calculation_l4036_403626


namespace eighteen_wheeler_toll_l4036_403676

/-- Calculate the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  1.50 + 0.50 * (axles - 2)

/-- Calculate the number of axles for a truck given the total number of wheels -/
def axles_count (wheels : ℕ) : ℕ :=
  wheels / 2

theorem eighteen_wheeler_toll :
  let wheels : ℕ := 18
  let axles : ℕ := axles_count wheels
  toll axles = 5 := by sorry

end eighteen_wheeler_toll_l4036_403676


namespace percentage_problem_l4036_403680

theorem percentage_problem (P : ℝ) : P = 25 ↔ 0.15 * 40 = (P / 100) * 16 + 2 := by
  sorry

end percentage_problem_l4036_403680


namespace max_value_expression_l4036_403605

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 4 - Real.sqrt (x^4 + 16)) / x ≤ 2 * Real.sqrt 2 - 2 := by
  sorry

end max_value_expression_l4036_403605


namespace sum_18_probability_l4036_403637

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- The target sum we're aiming for -/
def target_sum : ℕ := 18

/-- The probability of rolling a sum of 18 with four standard 6-faced dice -/
def probability_sum_18 : ℚ := 5 / 216

/-- Theorem stating that the probability of rolling a sum of 18 with four standard 6-faced dice is 5/216 -/
theorem sum_18_probability : 
  probability_sum_18 = (num_favorable_outcomes : ℚ) / (num_faces ^ num_dice : ℚ) :=
by sorry

end sum_18_probability_l4036_403637


namespace power_of_three_equals_square_minus_sixteen_l4036_403690

theorem power_of_three_equals_square_minus_sixteen (a n : ℕ+) :
  (3 : ℕ) ^ (n : ℕ) = (a : ℕ) ^ 2 - 16 ↔ a = 5 ∧ n = 2 := by
  sorry

end power_of_three_equals_square_minus_sixteen_l4036_403690


namespace tangent_length_l4036_403681

/-- The circle C with equation x^2 + y^2 - 2x - 6y + 9 = 0 -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 6*y + 9 = 0

/-- The point P on the x-axis -/
def point_P : ℝ × ℝ := (1, 0)

/-- The length of the tangent from P to circle C is 2√2 -/
theorem tangent_length : 
  ∃ (t : ℝ × ℝ), 
    circle_C t.1 t.2 ∧ 
    ((t.1 - point_P.1)^2 + (t.2 - point_P.2)^2) = 8 :=
sorry

end tangent_length_l4036_403681


namespace ned_short_sleeve_shirts_l4036_403694

/-- The number of short sleeve shirts Ned had to wash -/
def short_sleeve_shirts : ℕ := sorry

/-- The number of long sleeve shirts Ned had to wash -/
def long_sleeve_shirts : ℕ := 21

/-- The number of shirts Ned washed before school started -/
def washed_shirts : ℕ := 29

/-- The number of shirts Ned did not wash -/
def unwashed_shirts : ℕ := 1

/-- The total number of shirts Ned had to wash -/
def total_shirts : ℕ := washed_shirts + unwashed_shirts

theorem ned_short_sleeve_shirts :
  short_sleeve_shirts = total_shirts - long_sleeve_shirts :=
by sorry

end ned_short_sleeve_shirts_l4036_403694


namespace original_price_from_loss_and_selling_price_l4036_403673

/-- Proves that if an item is sold at a 20% loss for 600 currency units, 
    then its original price was 750 currency units. -/
theorem original_price_from_loss_and_selling_price 
  (selling_price : ℝ) 
  (loss_percentage : ℝ) 
  (h1 : selling_price = 600) 
  (h2 : loss_percentage = 20) : 
  ∃ original_price : ℝ, 
    original_price = 750 ∧ 
    selling_price = original_price * (1 - loss_percentage / 100) := by
  sorry

end original_price_from_loss_and_selling_price_l4036_403673


namespace simplify_trig_expression_l4036_403657

theorem simplify_trig_expression (θ : Real) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = 2 * Real.cos θ := by
  sorry

end simplify_trig_expression_l4036_403657


namespace range_of_positive_f_l4036_403658

/-- A function is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem range_of_positive_f 
  (f : ℝ → ℝ) 
  (f' : ℝ → ℝ) 
  (hf_odd : OddFunction f)
  (hf_deriv : ∀ x, HasDerivAt f (f' x) x)
  (hf_neg_one : f (-1) = 0)
  (hf_pos : ∀ x > 0, x * f' x - f x > 0) :
  {x | f x > 0} = Set.Ioo (-1) 0 ∪ Set.Ioi 1 :=
sorry

end range_of_positive_f_l4036_403658


namespace solve_system_l4036_403643

theorem solve_system (y z : ℝ) 
  (h1 : y^2 - 6*y + 9 = 0) 
  (h2 : y + z = 11) : 
  y = 3 ∧ z = 8 := by
sorry

end solve_system_l4036_403643


namespace imaginary_part_of_complex_square_imaginary_part_of_one_minus_two_i_squared_l4036_403662

theorem imaginary_part_of_complex_square : ℂ → ℝ
  | ⟨re, im⟩ => im

theorem imaginary_part_of_one_minus_two_i_squared :
  imaginary_part_of_complex_square ((1 - 2 * Complex.I) ^ 2) = -4 := by sorry

end imaginary_part_of_complex_square_imaginary_part_of_one_minus_two_i_squared_l4036_403662


namespace line_segment_point_sum_l4036_403628

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := y = -5/6 * x + 10

/-- Point P is on the x-axis -/
def P : ℝ × ℝ := (12, 0)

/-- Point Q is on the y-axis -/
def Q : ℝ × ℝ := (0, 10)

/-- Point T is on the line segment PQ -/
def T : ℝ × ℝ → Prop
  | (r, s) => ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- Area of triangle POQ -/
def area_POQ : ℝ := 60

/-- Area of triangle TOP -/
def area_TOP : ℝ := 15

/-- Theorem: If the given conditions are met, then r + s = 11.5 -/
theorem line_segment_point_sum (r s : ℝ) : 
  line_eq r s → T (r, s) → area_POQ = 4 * area_TOP → r + s = 11.5 := by
  sorry

end line_segment_point_sum_l4036_403628


namespace min_value_expression_l4036_403618

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 2) :
  ∃ (m : ℝ), m = 25/2 ∧ ∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 2 → (1 + 4*x + 3*y) / (x*y) ≥ m :=
sorry

end min_value_expression_l4036_403618


namespace division_by_three_remainder_l4036_403682

theorem division_by_three_remainder (n : ℤ) : 
  (n % 3 ≠ 0) → (n % 3 = 1 ∨ n % 3 = 2) :=
by sorry

end division_by_three_remainder_l4036_403682


namespace number_divided_by_constant_l4036_403652

theorem number_divided_by_constant (x : ℝ) : x / 0.06 = 16.666666666666668 → x = 1 := by
  sorry

end number_divided_by_constant_l4036_403652


namespace number_problem_l4036_403659

theorem number_problem : ∃ x : ℝ, 3 * (2 * x + 8) = 84 := by
  sorry

end number_problem_l4036_403659


namespace call_center_team_b_fraction_l4036_403675

/-- Represents the fraction of calls processed by Team B given the relative
    call processing rates and team sizes of two teams in a call center. -/
theorem call_center_team_b_fraction :
  -- Each member of Team A processes 6/5 calls compared to Team B
  ∀ (call_rate_a call_rate_b : ℚ),
  call_rate_a = 6 / 5 * call_rate_b →
  -- Team A has 5/8 as many agents as Team B
  ∀ (team_size_a team_size_b : ℚ),
  team_size_a = 5 / 8 * team_size_b →
  -- The fraction of calls processed by Team B
  (team_size_b * call_rate_b) /
    (team_size_a * call_rate_a + team_size_b * call_rate_b) = 4 / 7 :=
by sorry

end call_center_team_b_fraction_l4036_403675


namespace parabola_chord_length_l4036_403651

/-- Given a parabola y² = 4x with a chord passing through its focus and endpoints A(x₁, y₁) and B(x₂, y₂),
    if x₁ + x₂ = 6, then the length of AB is 8. -/
theorem parabola_chord_length (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 4*x₁ → y₂^2 = 4*x₂ → x₁ + x₂ = 6 → 
  ∃ (AB : ℝ), AB = Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) ∧ AB = 8 := by
  sorry

end parabola_chord_length_l4036_403651


namespace quadratic_function_m_range_l4036_403672

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 5

theorem quadratic_function_m_range (a : ℝ) (m : ℝ) :
  (∀ t, f a t = f a (-4 - t)) →
  (∀ x ∈ Set.Icc m 0, f a x ≤ 5) →
  (∃ x ∈ Set.Icc m 0, f a x = 5) →
  (∀ x ∈ Set.Icc m 0, f a x ≥ 1) →
  (∃ x ∈ Set.Icc m 0, f a x = 1) →
  -4 ≤ m ∧ m ≤ -2 :=
by sorry

end quadratic_function_m_range_l4036_403672


namespace min_value_theorem_l4036_403647

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (4 / (x + 2)) + (1 / (y + 1)) ≥ 9/4 := by
  sorry

end min_value_theorem_l4036_403647


namespace paul_sandwich_consumption_l4036_403632

/-- Calculates the number of sandwiches eaten in one 3-day cycle -/
def sandwiches_per_cycle (initial : ℕ) : ℕ :=
  initial + 2 * initial + 4 * initial

/-- Calculates the total number of sandwiches eaten in a given number of days -/
def total_sandwiches (days : ℕ) (initial : ℕ) : ℕ :=
  (days / 3) * sandwiches_per_cycle initial + 
  if days % 3 = 1 then initial
  else if days % 3 = 2 then initial + 2 * initial
  else 0

theorem paul_sandwich_consumption :
  total_sandwiches 6 2 = 28 := by
  sorry

end paul_sandwich_consumption_l4036_403632


namespace path_cost_calculation_l4036_403683

/-- Calculates the total cost of building paths around a rectangular plot -/
def calculate_path_cost (plot_length : Real) (plot_width : Real) 
                        (gravel_path_width : Real) (concrete_path_width : Real)
                        (gravel_cost_per_sqm : Real) (concrete_cost_per_sqm : Real) : Real :=
  let gravel_path_area := 2 * plot_length * gravel_path_width
  let concrete_path_area := 2 * plot_width * concrete_path_width
  let gravel_cost := gravel_path_area * gravel_cost_per_sqm
  let concrete_cost := concrete_path_area * concrete_cost_per_sqm
  gravel_cost + concrete_cost

/-- Theorem stating that the total cost of building the paths is approximately Rs. 9.78 -/
theorem path_cost_calculation :
  let plot_length := 120
  let plot_width := 0.85
  let gravel_path_width := 0.05
  let concrete_path_width := 0.07
  let gravel_cost_per_sqm := 0.80
  let concrete_cost_per_sqm := 1.50
  abs (calculate_path_cost plot_length plot_width gravel_path_width concrete_path_width
                           gravel_cost_per_sqm concrete_cost_per_sqm - 9.78) < 0.01 := by
  sorry


end path_cost_calculation_l4036_403683


namespace prime_sum_47_l4036_403674

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define the property we want to prove
def no_prime_sum_47 : Prop :=
  ∀ p q : ℕ, is_prime p → is_prime q → p + q ≠ 47

-- State the theorem
theorem prime_sum_47 : no_prime_sum_47 :=
sorry

end prime_sum_47_l4036_403674


namespace no_rectangle_with_sum_76_l4036_403622

theorem no_rectangle_with_sum_76 : ¬∃ (w : ℕ), w > 0 ∧ 2 * w^2 + 6 * w = 76 := by
  sorry

end no_rectangle_with_sum_76_l4036_403622


namespace unique_integer_complex_sixth_power_l4036_403666

def complex_sixth_power_is_integer (n : ℤ) : Prop :=
  ∃ m : ℤ, (n + Complex.I) ^ 6 = m

theorem unique_integer_complex_sixth_power :
  ∃! n : ℤ, complex_sixth_power_is_integer n :=
sorry

end unique_integer_complex_sixth_power_l4036_403666


namespace second_smallest_dimension_is_twelve_l4036_403686

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical pillar -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate -/
def cylinderFitsInCrate (c : Cylinder) (d : CrateDimensions) : Prop :=
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.width) ∨
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.height) ∨
  (2 * c.radius ≤ d.width ∧ 2 * c.radius ≤ d.height)

/-- The theorem stating that the second smallest dimension of the crate is 12 feet -/
theorem second_smallest_dimension_is_twelve
  (d : CrateDimensions)
  (h1 : d.length = 6)
  (h2 : d.height = 12)
  (h3 : d.width > 0)
  (c : Cylinder)
  (h4 : c.radius = 6)
  (h5 : cylinderFitsInCrate c d) :
  d.width = 12 ∨ d.width = 12 :=
sorry

end second_smallest_dimension_is_twelve_l4036_403686


namespace expanded_polynomial_terms_count_l4036_403607

theorem expanded_polynomial_terms_count : 
  let factor1 := 4  -- number of terms in (a₁ + a₂ + a₃ + a₄)
  let factor2 := 2  -- number of terms in (b₁ + b₂)
  let factor3 := 3  -- number of terms in (c₁ + c₂ + c₃)
  factor1 * factor2 * factor3 = 24 := by
  sorry

end expanded_polynomial_terms_count_l4036_403607


namespace franks_breakfast_shopping_l4036_403625

/-- The cost of a bottle of milk in Frank's breakfast shopping -/
def milk_cost : ℝ := 2.5

/-- The cost of 10 buns -/
def buns_cost : ℝ := 1

/-- The number of bottles of milk Frank bought -/
def milk_bottles : ℕ := 1

/-- The cost of the carton of eggs -/
def eggs_cost : ℝ := 3 * milk_cost

/-- The total cost of Frank's breakfast shopping -/
def total_cost : ℝ := 11

theorem franks_breakfast_shopping :
  buns_cost + milk_bottles * milk_cost + eggs_cost = total_cost :=
by sorry

end franks_breakfast_shopping_l4036_403625
