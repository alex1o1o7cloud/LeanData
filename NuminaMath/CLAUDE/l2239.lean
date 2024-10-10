import Mathlib

namespace division_problem_additional_condition_l2239_223968

theorem division_problem (x : ℝ) : 2994 / x = 175 → x = 17.1 := by
  sorry

-- Additional theorem to include the unused condition
theorem additional_condition : 29.94 / 1.45 = 17.5 := by
  sorry

end division_problem_additional_condition_l2239_223968


namespace winning_configurations_l2239_223961

/-- Represents a wall configuration in the brick removal game -/
structure WallConfig :=
  (walls : List Nat)

/-- Calculates the nim-value of a single wall -/
def nimValue (wall : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum of a list of nim-values -/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a configuration is a winning position for the second player -/
def isWinningForSecondPlayer (config : WallConfig) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- The list of all possible starting configurations -/
def startingConfigs : List WallConfig :=
  [⟨[7, 3, 2]⟩, ⟨[7, 4, 1]⟩, ⟨[8, 3, 1]⟩, ⟨[7, 2, 2]⟩, ⟨[7, 3, 3]⟩]

/-- The main theorem to be proved -/
theorem winning_configurations :
  (∀ c ∈ startingConfigs, isWinningForSecondPlayer c ↔ (c = ⟨[7, 3, 2]⟩ ∨ c = ⟨[8, 3, 1]⟩)) :=
  sorry

end winning_configurations_l2239_223961


namespace evaluate_expression_l2239_223959

theorem evaluate_expression : 7^3 - 4 * 7^2 + 6 * 7 - 2 = 187 := by
  sorry

end evaluate_expression_l2239_223959


namespace rachel_class_selection_l2239_223930

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem rachel_class_selection :
  let total_classes : ℕ := 10
  let mandatory_classes : ℕ := 2
  let classes_to_choose : ℕ := 5
  let remaining_classes := total_classes - mandatory_classes
  let additional_classes := classes_to_choose - mandatory_classes
  choose remaining_classes additional_classes = 56 := by sorry

end rachel_class_selection_l2239_223930


namespace fourth_side_length_l2239_223962

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  /-- The radius of the circle -/
  radius : ℝ
  /-- The lengths of the four sides -/
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  /-- The quadrilateral is inscribed in the circle -/
  inscribed : True
  /-- The quadrilateral is not a rectangle -/
  not_rectangle : True

/-- Theorem: In a quadrilateral inscribed in a circle with radius 150√2,
    if three sides have length 150, then the fourth side has length 300√2 -/
theorem fourth_side_length (q : InscribedQuadrilateral)
    (h_radius : q.radius = 150 * Real.sqrt 2)
    (h_side1 : q.side1 = 150)
    (h_side2 : q.side2 = 150)
    (h_side3 : q.side3 = 150) :
    q.side4 = 300 * Real.sqrt 2 := by
  sorry

end fourth_side_length_l2239_223962


namespace second_class_size_l2239_223989

/-- Given two classes of students, where:
    - The first class has 24 students with an average mark of 40
    - The second class has an unknown number of students with an average mark of 60
    - The average mark of all students combined is 53.513513513513516
    This theorem proves that the number of students in the second class is 50. -/
theorem second_class_size (n : ℕ) :
  let first_class_size : ℕ := 24
  let first_class_avg : ℝ := 40
  let second_class_avg : ℝ := 60
  let total_avg : ℝ := 53.513513513513516
  let total_size : ℕ := first_class_size + n
  (first_class_size * first_class_avg + n * second_class_avg) / total_size = total_avg →
  n = 50 := by
sorry

end second_class_size_l2239_223989


namespace complex_modulus_problem_l2239_223921

theorem complex_modulus_problem (i : ℂ) (h : i^2 = -1) :
  Complex.abs (i / (1 + i^3)) = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_problem_l2239_223921


namespace minimum_savings_for_contribution_l2239_223928

def savings_september : ℕ := 50
def savings_october : ℕ := 37
def savings_november : ℕ := 11
def mom_contribution : ℕ := 25
def video_game_cost : ℕ := 87
def amount_left : ℕ := 36

def total_savings : ℕ := savings_september + savings_october + savings_november

theorem minimum_savings_for_contribution :
  total_savings = (amount_left + video_game_cost) - mom_contribution :=
by sorry

end minimum_savings_for_contribution_l2239_223928


namespace count_permutable_divisible_by_11_l2239_223957

/-- A function that counts the number of integers with k digits (including leading zeros)
    whose digits can be permuted to form a number divisible by 11 -/
def f (k : ℕ) : ℕ := sorry

/-- A predicate that checks if an integer's digits can be permuted to form a number divisible by 11 -/
def can_permute_to_divisible_by_11 (n : ℕ) : Prop := sorry

theorem count_permutable_divisible_by_11 (m : ℕ+) :
  f (2 * m) = 10 * f (2 * m - 1) :=
by sorry

end count_permutable_divisible_by_11_l2239_223957


namespace consumption_ranking_l2239_223953

-- Define the regions
inductive Region
| West
| NonWest
| Russia

-- Define the consumption function
def consumption : Region → ℝ
| Region.West => 21428
| Region.NonWest => 26848.55
| Region.Russia => 302790.13

-- Define the ranking function
def ranking (r : Region) : ℕ :=
  match r with
  | Region.West => 3
  | Region.NonWest => 2
  | Region.Russia => 1

-- Theorem statement
theorem consumption_ranking :
  ∀ r1 r2 : Region, ranking r1 < ranking r2 ↔ consumption r1 > consumption r2 :=
by sorry

end consumption_ranking_l2239_223953


namespace min_value_expression_l2239_223946

theorem min_value_expression (x : ℝ) (h : x > 1) :
  x + 9 / x - 2 ≥ 4 ∧ ∃ y > 1, y + 9 / y - 2 = 4 := by
  sorry

end min_value_expression_l2239_223946


namespace andy_max_demerits_l2239_223971

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def maxDemerits : ℕ := by sorry

/-- The number of demerits Andy gets per instance of showing up late -/
def demeritsPerLateInstance : ℕ := 2

/-- The number of times Andy showed up late -/
def lateInstances : ℕ := 6

/-- The number of demerits Andy got for making an inappropriate joke -/
def demeritsForJoke : ℕ := 15

/-- The number of additional demerits Andy can get before getting fired -/
def remainingDemerits : ℕ := 23

theorem andy_max_demerits :
  maxDemerits = demeritsPerLateInstance * lateInstances + demeritsForJoke + remainingDemerits := by
  sorry

end andy_max_demerits_l2239_223971


namespace gary_stickers_left_l2239_223943

/-- The number of stickers Gary had initially -/
def initial_stickers : ℕ := 99

/-- The number of stickers Gary gave to Lucy -/
def stickers_to_lucy : ℕ := 42

/-- The number of stickers Gary gave to Alex -/
def stickers_to_alex : ℕ := 26

/-- The number of stickers Gary had left after giving stickers to Lucy and Alex -/
def stickers_left : ℕ := initial_stickers - (stickers_to_lucy + stickers_to_alex)

theorem gary_stickers_left : stickers_left = 31 := by
  sorry

end gary_stickers_left_l2239_223943


namespace chord_length_perpendicular_bisector_l2239_223991

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 10) : 
  ∃ (c : ℝ), c = r * Real.sqrt 3 ∧ 
  c = 2 * Real.sqrt (r^2 - (r/2)^2) :=
by sorry

end chord_length_perpendicular_bisector_l2239_223991


namespace water_added_amount_l2239_223904

def initial_volume : ℝ := 340
def initial_water_percent : ℝ := 0.75
def initial_kola_percent : ℝ := 0.05
def initial_sugar_percent : ℝ := 1 - initial_water_percent - initial_kola_percent
def added_sugar : ℝ := 3.2
def added_kola : ℝ := 6.8
def final_sugar_percent : ℝ := 0.1966850828729282

theorem water_added_amount (added_water : ℝ) : 
  let initial_sugar := initial_volume * initial_sugar_percent
  let total_sugar := initial_sugar + added_sugar
  let final_volume := initial_volume + added_sugar + added_kola + added_water
  total_sugar / final_volume = final_sugar_percent →
  added_water = 12 := by sorry

end water_added_amount_l2239_223904


namespace direction_vector_b_l2239_223937

/-- Given a line passing through points (-4, 6) and (3, -3), 
    prove that the direction vector of the form (b, 1) has b = -7/9 -/
theorem direction_vector_b (p1 p2 : ℝ × ℝ) (b : ℝ) : 
  p1 = (-4, 6) → p2 = (3, -3) → 
  ∃ k : ℝ, k • (p2.1 - p1.1, p2.2 - p1.2) = (b, 1) → 
  b = -7/9 := by
sorry

end direction_vector_b_l2239_223937


namespace arithmetic_sequence_product_l2239_223992

/-- Given arithmetic sequences a and b satisfying certain conditions, prove a₁b₁ = 4 -/
theorem arithmetic_sequence_product (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- a is arithmetic
  (∀ n, b (n + 1) - b n = b (n + 2) - b (n + 1)) →  -- b is arithmetic
  a 2 * b 2 = 4 →
  a 3 * b 3 = 8 →
  a 4 * b 4 = 16 →
  a 1 * b 1 = 4 := by
sorry

end arithmetic_sequence_product_l2239_223992


namespace expression_simplification_l2239_223964

theorem expression_simplification (x y z : ℝ) 
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 4) :
  (x - 2) / (6 - z) * (y - 3) / (2 - x) * (z - 4) / (3 - y) = -1 := by
  sorry

end expression_simplification_l2239_223964


namespace geometric_sequence_third_term_l2239_223933

theorem geometric_sequence_third_term
  (a : ℕ → ℝ)
  (h_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1))
  (h_a1 : a 1 = 2)
  (h_a5 : a 5 = 8) :
  a 3 = 4 :=
sorry

end geometric_sequence_third_term_l2239_223933


namespace roof_dimension_difference_l2239_223911

/-- Represents the dimensions of a rectangular roof --/
structure RoofDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular roof --/
def area (r : RoofDimensions) : ℝ := r.width * r.length

/-- Theorem: For a rectangular roof with length 4 times its width and an area of 1024 square feet,
    the difference between the length and width is 48 feet. --/
theorem roof_dimension_difference (r : RoofDimensions) 
    (h1 : r.length = 4 * r.width) 
    (h2 : area r = 1024) : 
    r.length - r.width = 48 := by
  sorry


end roof_dimension_difference_l2239_223911


namespace winning_strategy_l2239_223947

/-- Represents the winner of the game -/
inductive Winner
  | FirstPlayer
  | SecondPlayer

/-- Determines the winner of the game based on board dimensions -/
def gameWinner (n k : ℕ) : Winner :=
  if (n + k) % 2 = 0 then Winner.SecondPlayer else Winner.FirstPlayer

/-- Theorem stating the winning condition for the game -/
theorem winning_strategy (n k : ℕ) (h1 : n > 0) (h2 : k > 1) :
  gameWinner n k = if (n + k) % 2 = 0 then Winner.SecondPlayer else Winner.FirstPlayer :=
by sorry

end winning_strategy_l2239_223947


namespace product_of_two_numbers_l2239_223999

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x^2 + y^2 = 289) 
  (h2 : x + y = 23) : 
  x * y = 120 := by
sorry

end product_of_two_numbers_l2239_223999


namespace factorization_xy_squared_minus_4x_l2239_223923

theorem factorization_xy_squared_minus_4x (x y : ℝ) : 
  x * y^2 - 4 * x = x * (y + 2) * (y - 2) := by sorry

end factorization_xy_squared_minus_4x_l2239_223923


namespace vector_relations_l2239_223984

-- Define the plane vector type
structure PlaneVector where
  x : ℝ
  y : ℝ

-- Define the "›" relation
def vecGreater (a b : PlaneVector) : Prop :=
  a.x > b.x ∨ (a.x = b.x ∧ a.y > b.y)

-- Define vector addition
def vecAdd (a b : PlaneVector) : PlaneVector :=
  ⟨a.x + b.x, a.y + b.y⟩

-- Define dot product
def vecDot (a b : PlaneVector) : ℝ :=
  a.x * b.x + a.y * b.y

-- Theorem statement
theorem vector_relations :
  let e₁ : PlaneVector := ⟨1, 0⟩
  let e₂ : PlaneVector := ⟨0, 1⟩
  let zero : PlaneVector := ⟨0, 0⟩
  
  -- Proposition 1
  (vecGreater e₁ e₂ ∧ vecGreater e₂ zero) ∧
  
  -- Proposition 2
  (∀ a₁ a₂ a₃ : PlaneVector, vecGreater a₁ a₂ → vecGreater a₂ a₃ → vecGreater a₁ a₃) ∧
  
  -- Proposition 3
  (∀ a₁ a₂ a : PlaneVector, vecGreater a₁ a₂ → vecGreater (vecAdd a₁ a) (vecAdd a₂ a)) ∧
  
  -- Proposition 4 (negation)
  ¬(∀ a a₁ a₂ : PlaneVector, vecGreater a zero → vecGreater a₁ a₂ → vecGreater ⟨vecDot a a₁, 0⟩ ⟨vecDot a a₂, 0⟩) :=
by
  sorry


end vector_relations_l2239_223984


namespace slant_height_neq_base_side_l2239_223998

/-- Represents a regular hexagonal pyramid --/
structure RegularHexagonalPyramid where
  r : ℝ  -- side length of each equilateral triangle in the base
  h : ℝ  -- height of the pyramid
  l : ℝ  -- slant height (lateral edge) of the pyramid
  r_pos : r > 0
  h_pos : h > 0
  l_pos : l > 0
  pythagorean : h^2 + r^2 = l^2

/-- Theorem: In a regular hexagonal pyramid, the slant height cannot be equal to the side length of the base hexagon --/
theorem slant_height_neq_base_side (p : RegularHexagonalPyramid) : p.l ≠ p.r := by
  sorry


end slant_height_neq_base_side_l2239_223998


namespace quarters_percentage_is_65_22_l2239_223969

/-- The number of dimes -/
def num_dimes : ℕ := 40

/-- The number of quarters -/
def num_quarters : ℕ := 30

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The total value of all coins in cents -/
def total_value : ℕ := num_dimes * dime_value + num_quarters * quarter_value

/-- The value of all quarters in cents -/
def quarters_value : ℕ := num_quarters * quarter_value

/-- The percentage of the total value that is in quarters -/
def quarters_percentage : ℚ := (quarters_value : ℚ) / (total_value : ℚ) * 100

theorem quarters_percentage_is_65_22 : 
  ∀ ε > 0, |quarters_percentage - 65.22| < ε :=
sorry

end quarters_percentage_is_65_22_l2239_223969


namespace order_of_abc_l2239_223976

theorem order_of_abc : ∀ (a b c : ℝ), 
  a = 2^(1/10) → 
  b = Real.log (1/2) → 
  c = (2/3)^Real.pi → 
  a > c ∧ c > b :=
by
  sorry

end order_of_abc_l2239_223976


namespace difference_sum_of_T_l2239_223935

def T : Finset ℕ := Finset.range 11

def difference_sum (s : Finset ℕ) : ℕ :=
  s.sum (fun i => s.sum (fun j => if i > j then (3^i - 3^j) else 0))

theorem difference_sum_of_T : difference_sum T = 793168 := by
  sorry

end difference_sum_of_T_l2239_223935


namespace triangle_perimeter_impossibility_l2239_223938

theorem triangle_perimeter_impossibility (a b x : ℝ) (h1 : a = 24) (h2 : b = 10) :
  (a + b + x = 73) → ¬(a + b > x ∧ a + x > b ∧ b + x > a) :=
by sorry

end triangle_perimeter_impossibility_l2239_223938


namespace hundred_million_scientific_notation_l2239_223952

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem hundred_million_scientific_notation :
  toScientificNotation 100000000 = ScientificNotation.mk 1 8 (by norm_num) :=
sorry

end hundred_million_scientific_notation_l2239_223952


namespace point_in_second_quadrant_l2239_223997

-- Define the point (x, y) in the Cartesian coordinate system
def point (a : ℝ) : ℝ × ℝ := (-2, a^2 + 1)

-- Define what it means for a point to be in the second quadrant
def in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

-- Theorem statement
theorem point_in_second_quadrant (a : ℝ) :
  in_second_quadrant (point a) := by
  sorry

end point_in_second_quadrant_l2239_223997


namespace a_ln_a_gt_b_ln_b_l2239_223902

theorem a_ln_a_gt_b_ln_b (a b : ℝ) (h1 : a > b) (h2 : b > 1) : a * Real.log a > b * Real.log b := by
  sorry

end a_ln_a_gt_b_ln_b_l2239_223902


namespace student_grade_problem_l2239_223960

theorem student_grade_problem (grade2 grade3 average : ℚ) 
  (h1 : grade2 = 80/100)
  (h2 : grade3 = 85/100)
  (h3 : average = 75/100)
  (h4 : (grade1 + grade2 + grade3) / 3 = average) :
  grade1 = 60/100 := by
  sorry

end student_grade_problem_l2239_223960


namespace complement_A_intersect_B_when_m_3_A_subset_B_iff_m_in_range_l2239_223970

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 18 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m - 8 ≤ x ∧ x ≤ m + 4}

-- Statement 1
theorem complement_A_intersect_B_when_m_3 : 
  (Set.univ \ A) ∩ B 3 = {x | -5 ≤ x ∧ x < -3 ∨ 6 < x ∧ x ≤ 7} := by sorry

-- Statement 2
theorem A_subset_B_iff_m_in_range : 
  ∀ m, A ∩ B m = A ↔ 2 ≤ m ∧ m ≤ 5 := by sorry

end complement_A_intersect_B_when_m_3_A_subset_B_iff_m_in_range_l2239_223970


namespace circle_radius_from_area_circumference_relation_l2239_223955

theorem circle_radius_from_area_circumference_relation : 
  ∀ r : ℝ, r > 0 → (3 * (2 * Real.pi * r) = Real.pi * r^2) → r = 6 := by
  sorry

end circle_radius_from_area_circumference_relation_l2239_223955


namespace solve_equation_l2239_223929

theorem solve_equation (x : ℚ) : (2 * x + 7) / 5 = 22 → x = 103 / 2 := by
  sorry

end solve_equation_l2239_223929


namespace share_difference_l2239_223916

/-- Given a distribution ratio and Vasim's share, calculate the difference between Ranjith's and Faruk's shares -/
theorem share_difference (faruk_ratio vasim_ratio ranjith_ratio vasim_share : ℕ) : 
  faruk_ratio = 3 → 
  vasim_ratio = 3 → 
  ranjith_ratio = 7 → 
  vasim_share = 1500 → 
  (ranjith_ratio * vasim_share / vasim_ratio) - (faruk_ratio * vasim_share / vasim_ratio) = 2000 := by
  sorry

end share_difference_l2239_223916


namespace polynomial_remainder_l2239_223919

def polynomial (x : ℝ) : ℝ := 4*x^8 - 3*x^7 + 2*x^6 - 8*x^4 + 5*x^3 - 9

def divisor (x : ℝ) : ℝ := 3*x - 6

theorem polynomial_remainder : 
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + 671 := by
  sorry

end polynomial_remainder_l2239_223919


namespace car_journey_cost_l2239_223924

/-- Calculates the total cost of a car journey given various expenses -/
theorem car_journey_cost
  (rental_cost : ℝ)
  (rental_discount_percent : ℝ)
  (gas_cost_per_gallon : ℝ)
  (gas_gallons : ℝ)
  (driving_cost_per_mile : ℝ)
  (miles_driven : ℝ)
  (toll_fees : ℝ)
  (parking_cost_per_day : ℝ)
  (parking_days : ℝ)
  (h1 : rental_cost = 150)
  (h2 : rental_discount_percent = 15)
  (h3 : gas_cost_per_gallon = 3.5)
  (h4 : gas_gallons = 8)
  (h5 : driving_cost_per_mile = 0.5)
  (h6 : miles_driven = 320)
  (h7 : toll_fees = 15)
  (h8 : parking_cost_per_day = 20)
  (h9 : parking_days = 3) :
  rental_cost * (1 - rental_discount_percent / 100) +
  gas_cost_per_gallon * gas_gallons +
  driving_cost_per_mile * miles_driven +
  toll_fees +
  parking_cost_per_day * parking_days = 390.5 := by
  sorry


end car_journey_cost_l2239_223924


namespace final_F_position_l2239_223900

-- Define the letter F as a type with base and stem directions
inductive LetterF
  | mk (base : ℝ × ℝ) (stem : ℝ × ℝ)

-- Define the initial position of F
def initial_F : LetterF := LetterF.mk (-1, 0) (0, -1)

-- Define the transformations
def rotate_180 (f : LetterF) : LetterF :=
  match f with
  | LetterF.mk (x, y) (a, b) => LetterF.mk (-x, -y) (-a, -b)

def reflect_y_axis (f : LetterF) : LetterF :=
  match f with
  | LetterF.mk (x, y) (a, b) => LetterF.mk (-x, y) (-a, b)

def rotate_90 (f : LetterF) : LetterF :=
  match f with
  | LetterF.mk (x, y) (a, b) => LetterF.mk (y, -x) (b, -a)

-- Define the final transformation as a composition of the three transformations
def final_transformation (f : LetterF) : LetterF :=
  rotate_90 (reflect_y_axis (rotate_180 f))

-- Theorem: The final position of F after transformations
theorem final_F_position :
  final_transformation initial_F = LetterF.mk (0, -1) (-1, 0) :=
by sorry

end final_F_position_l2239_223900


namespace ale_age_l2239_223925

/-- Represents a year-month combination -/
structure YearMonth where
  year : ℕ
  month : ℕ
  h_month_valid : month ≥ 1 ∧ month ≤ 12

/-- Calculates the age in years between two YearMonth dates -/
def ageInYears (birth death : YearMonth) : ℕ :=
  death.year - birth.year

theorem ale_age :
  let birth := YearMonth.mk 1859 1 (by simp)
  let death := YearMonth.mk 2014 8 (by simp)
  ageInYears birth death = 155 := by
  sorry

#check ale_age

end ale_age_l2239_223925


namespace speed_ratio_l2239_223977

/-- Represents the scenario of Xiaoqing and Xiaoqiang's journey --/
structure Journey where
  distance : ℝ
  walking_speed : ℝ
  motorcycle_speed : ℝ
  (walking_speed_pos : walking_speed > 0)
  (motorcycle_speed_pos : motorcycle_speed > 0)
  (distance_pos : distance > 0)

/-- The time taken for the entire journey is 2.5 times the direct trip --/
def journey_time_constraint (j : Journey) : Prop :=
  (j.distance / j.motorcycle_speed) * 2.5 = 
    (j.distance / j.motorcycle_speed) + 
    (j.distance / j.motorcycle_speed - j.distance / j.walking_speed)

/-- The theorem stating the ratio of speeds --/
theorem speed_ratio (j : Journey) 
  (h : journey_time_constraint j) : 
  j.motorcycle_speed / j.walking_speed = 3 := by
  sorry


end speed_ratio_l2239_223977


namespace copper_price_calculation_l2239_223993

/-- The price of copper per pound in cents -/
def copper_price : ℚ := 65

/-- The price of zinc per pound in cents -/
def zinc_price : ℚ := 30

/-- The weight of brass in pounds -/
def brass_weight : ℚ := 70

/-- The price of brass per pound in cents -/
def brass_price : ℚ := 45

/-- The weight of copper used in pounds -/
def copper_weight : ℚ := 30

/-- The weight of zinc used in pounds -/
def zinc_weight : ℚ := 40

theorem copper_price_calculation : 
  copper_price * copper_weight + zinc_price * zinc_weight = brass_price * brass_weight :=
sorry

end copper_price_calculation_l2239_223993


namespace length_AE_l2239_223942

-- Define the circle
def Circle := {c : ℝ × ℝ | c.1^2 + c.2^2 = 4}

-- Define points A, B, C, D, E
variable (A B C D E : ℝ × ℝ)

-- AB is a diameter of the circle
axiom diam : A ∈ Circle ∧ B ∈ Circle ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16

-- ABC is an equilateral triangle
axiom equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                    (B.1 - C.1)^2 + (B.2 - C.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2

-- D is the intersection of the circle and AC
axiom D_on_circle : D ∈ Circle
axiom D_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)

-- E is the intersection of the circle and BC
axiom E_on_circle : E ∈ Circle
axiom E_on_BC : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ E = (s * B.1 + (1 - s) * C.1, s * B.2 + (1 - s) * C.2)

-- Theorem: The length of AE is 2√3
theorem length_AE : (A.1 - E.1)^2 + (A.2 - E.2)^2 = 12 := by sorry

end length_AE_l2239_223942


namespace weight_identification_unbiased_weight_identification_biased_l2239_223987

/-- Represents a weight with a mass in grams -/
structure Weight where
  mass : ℕ

/-- Represents a balance scale -/
inductive BalanceResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing operation -/
def weighing (left right : List Weight) (bias : ℕ := 0) : BalanceResult :=
  sorry

/-- Represents the process of identifying weights -/
def identifyWeights (weights : List Weight) (numWeighings : ℕ) (bias : ℕ := 0) : Bool :=
  sorry

/-- The set of weights Tanya has -/
def tanyasWeights : List Weight :=
  [⟨1000⟩, ⟨1002⟩, ⟨1004⟩, ⟨1005⟩]

theorem weight_identification_unbiased :
  ¬ (identifyWeights tanyasWeights 4 0) :=
sorry

theorem weight_identification_biased :
  identifyWeights tanyasWeights 4 1 :=
sorry

end weight_identification_unbiased_weight_identification_biased_l2239_223987


namespace jake_tuesday_watching_time_l2239_223906

def monday_hours : ℝ := 12
def wednesday_hours : ℝ := 6
def friday_hours : ℝ := 19
def total_show_length : ℝ := 52

def tuesday_hours : ℝ := 4

theorem jake_tuesday_watching_time :
  let mon_to_wed := monday_hours + tuesday_hours + wednesday_hours
  let thursday_hours := mon_to_wed / 2
  monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours = total_show_length :=
by sorry

end jake_tuesday_watching_time_l2239_223906


namespace right_triangle_inequalities_l2239_223909

-- Define a structure for a right-angled triangle with height to hypotenuse
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  right_angle : a^2 + b^2 = c^2
  height_def : 2 * h * c = a * b

theorem right_triangle_inequalities (t : RightTriangle) :
  (t.a^2 + t.b^2 < t.c^2 + t.h^2) ∧ (t.a^4 + t.b^4 < t.c^4 + t.h^4) :=
by sorry

end right_triangle_inequalities_l2239_223909


namespace min_balls_for_target_color_l2239_223954

def red_balls : ℕ := 35
def green_balls : ℕ := 25
def yellow_balls : ℕ := 22
def blue_balls : ℕ := 15
def white_balls : ℕ := 12
def black_balls : ℕ := 11

def total_balls : ℕ := red_balls + green_balls + yellow_balls + blue_balls + white_balls + black_balls

def target_color_count : ℕ := 18

theorem min_balls_for_target_color :
  ∃ (n : ℕ), n = 89 ∧
  (∀ (m : ℕ), m < n → ∃ (r g y bl w bk : ℕ),
    r + g + y + bl + w + bk = m ∧
    r ≤ red_balls ∧ g ≤ green_balls ∧ y ≤ yellow_balls ∧
    bl ≤ blue_balls ∧ w ≤ white_balls ∧ bk ≤ black_balls ∧
    r < target_color_count ∧ g < target_color_count ∧ y < target_color_count ∧
    bl < target_color_count ∧ w < target_color_count ∧ bk < target_color_count) ∧
  (∀ (r g y bl w bk : ℕ),
    r + g + y + bl + w + bk = n →
    r ≤ red_balls → g ≤ green_balls → y ≤ yellow_balls →
    bl ≤ blue_balls → w ≤ white_balls → bk ≤ black_balls →
    r ≥ target_color_count ∨ g ≥ target_color_count ∨ y ≥ target_color_count ∨
    bl ≥ target_color_count ∨ w ≥ target_color_count ∨ bk ≥ target_color_count) :=
by sorry

#check min_balls_for_target_color

end min_balls_for_target_color_l2239_223954


namespace intersection_when_a_10_subset_condition_l2239_223963

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a + 1 ≤ x ∧ x ≤ 3*a - 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 22}

-- Theorem for part 1
theorem intersection_when_a_10 :
  A 10 ∩ B = {x | 21 ≤ x ∧ x ≤ 22} := by sorry

-- Theorem for part 2
theorem subset_condition :
  ∀ a : ℝ, A a ⊆ B ↔ a ≤ 9 := by sorry

end intersection_when_a_10_subset_condition_l2239_223963


namespace count_integers_satisfying_inequality_l2239_223922

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset Int), (∀ n : Int, (n - 3) * (n + 5) * (n - 1) < 0 ↔ n ∈ S) ∧ Finset.card S = 6 :=
by sorry

end count_integers_satisfying_inequality_l2239_223922


namespace optimal_chair_removal_l2239_223980

/-- Represents the number of chairs in a complete row -/
def chairs_per_row : ℕ := 11

/-- Represents the initial total number of chairs -/
def initial_chairs : ℕ := 110

/-- Represents the number of students attending the assembly -/
def students : ℕ := 70

/-- Represents the number of chairs to be removed -/
def chairs_to_remove : ℕ := 33

/-- Proves that removing 33 chairs results in the optimal arrangement -/
theorem optimal_chair_removal :
  let remaining_chairs := initial_chairs - chairs_to_remove
  (remaining_chairs % chairs_per_row = 0) ∧
  (remaining_chairs ≥ students) ∧
  (∀ n : ℕ, n < chairs_to_remove →
    ((initial_chairs - n) % chairs_per_row = 0) →
    (initial_chairs - n < students ∨ initial_chairs - n > remaining_chairs)) :=
by sorry

end optimal_chair_removal_l2239_223980


namespace inequality_equivalence_l2239_223978

theorem inequality_equivalence (x : ℝ) : (x - 2)^2 < 9 ↔ -1 < x ∧ x < 5 := by
  sorry

end inequality_equivalence_l2239_223978


namespace odd_even_digit_difference_l2239_223939

/-- The upper bound of the range of integers we're considering -/
def upper_bound : ℕ := 8 * 10^20

/-- Counts the number of integers up to n (inclusive) that contain only odd digits -/
def count_odd_digits (n : ℕ) : ℕ := sorry

/-- Counts the number of integers up to n (inclusive) that contain only even digits -/
def count_even_digits (n : ℕ) : ℕ := sorry

/-- The main theorem stating the difference between odd-digit-only and even-digit-only numbers -/
theorem odd_even_digit_difference :
  count_odd_digits upper_bound - count_even_digits upper_bound = (5^21 - 1) / 4 := by sorry

end odd_even_digit_difference_l2239_223939


namespace floor_equation_solutions_l2239_223903

theorem floor_equation_solutions : 
  (∃ (S : Finset ℤ), S.card = 30 ∧ 
    (∀ x ∈ S, 0 ≤ x ∧ x < 30 ∧ x = ⌊x/2⌋ + ⌊x/3⌋ + ⌊x/5⌋) ∧
    (∀ x : ℤ, 0 ≤ x ∧ x < 30 ∧ x = ⌊x/2⌋ + ⌊x/3⌋ + ⌊x/5⌋ → x ∈ S)) :=
by sorry


end floor_equation_solutions_l2239_223903


namespace basketball_lineup_selection_l2239_223996

theorem basketball_lineup_selection (n m k : ℕ) (hn : n = 12) (hm : m = 5) (hk : k = 1) :
  n * Nat.choose (n - k) (m - k) = 3960 := by
  sorry

end basketball_lineup_selection_l2239_223996


namespace school_departments_l2239_223910

/-- Given a school with departments where each department has 20 teachers and there are 140 teachers in total, prove that the number of departments is 7. -/
theorem school_departments (total_teachers : ℕ) (teachers_per_dept : ℕ) (h1 : total_teachers = 140) (h2 : teachers_per_dept = 20) :
  total_teachers / teachers_per_dept = 7 := by
  sorry

end school_departments_l2239_223910


namespace speaker_discount_savings_l2239_223994

/-- Calculates the savings from a discount given the initial price and discounted price. -/
def savings (initial_price discounted_price : ℝ) : ℝ :=
  initial_price - discounted_price

/-- Theorem stating that the savings from a discount on speakers priced at $475.00 and sold for $199.00 is equal to $276.00. -/
theorem speaker_discount_savings :
  savings 475 199 = 276 := by
  sorry

end speaker_discount_savings_l2239_223994


namespace height_difference_l2239_223940

theorem height_difference (parker daisy reese : ℕ) 
  (h1 : daisy = reese + 8)
  (h2 : reese = 60)
  (h3 : (parker + daisy + reese) / 3 = 64) :
  daisy - parker = 4 := by
sorry

end height_difference_l2239_223940


namespace smallest_number_with_conditions_l2239_223920

def is_divisible_by (n m : ℕ) : Prop := m ∣ n

def has_additional_prime_factors (n : ℕ) (count : ℕ) : Prop :=
  ∃ (factors : List ℕ), 
    factors.length = count ∧ 
    (∀ p ∈ factors, Nat.Prime p) ∧
    (∀ p ∈ factors, p ∣ n) ∧
    (∀ p ∈ factors, p ≠ 2 ∧ p ≠ 5)

theorem smallest_number_with_conditions : 
  (∀ n : ℕ, n < 840 → 
    ¬(is_divisible_by n 8 ∧ 
      is_divisible_by n 5 ∧ 
      has_additional_prime_factors n 2)) ∧
  (is_divisible_by 840 8 ∧ 
   is_divisible_by 840 5 ∧ 
   has_additional_prime_factors 840 2) := by
  sorry

end smallest_number_with_conditions_l2239_223920


namespace larry_dog_time_l2239_223950

/-- The number of minutes Larry spends on his dog each day -/
def time_spent_on_dog (walking_playing_time : ℕ) (feeding_time : ℕ) : ℕ :=
  walking_playing_time * 2 + feeding_time

theorem larry_dog_time :
  let walking_playing_time : ℕ := 30 -- half an hour in minutes
  let feeding_time : ℕ := 12 -- a fifth of an hour in minutes
  time_spent_on_dog walking_playing_time feeding_time = 72 := by
sorry

end larry_dog_time_l2239_223950


namespace min_value_of_complex_expression_l2239_223927

theorem min_value_of_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (min_u : ℝ), min_u = (3/2) * Real.sqrt 3 ∧
  ∀ (u : ℝ), u = Complex.abs (z^2 - z + 1) → u ≥ min_u :=
sorry

end min_value_of_complex_expression_l2239_223927


namespace age_sum_problem_l2239_223990

theorem age_sum_problem :
  ∀ (y k : ℕ+),
    y * (2 * y) * k = 72 →
    y + (2 * y) + k = 13 :=
by sorry

end age_sum_problem_l2239_223990


namespace polynomial_factor_l2239_223901

-- Define the polynomials
def P (c : ℝ) (x : ℝ) : ℝ := 3 * x^3 + c * x + 12
def Q (q : ℝ) (x : ℝ) : ℝ := x^2 + q * x + 2

-- Theorem statement
theorem polynomial_factor (c : ℝ) :
  (∃ q r : ℝ, ∀ x : ℝ, P c x = Q q x * (r * x + (12 / r))) → c = 8 := by
  sorry

end polynomial_factor_l2239_223901


namespace no_extreme_points_iff_l2239_223905

/-- A cubic function parameterized by a real number a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * a * x^2 + (a + 1) * x

/-- The derivative of f with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 4 * a * x + (a + 1)

/-- The discriminant of f_deriv -/
def discriminant (a : ℝ) : ℝ := 4 * a^2 - 12 * a

/-- Theorem stating that f has no extreme points iff 0 ≤ a ≤ 3 -/
theorem no_extreme_points_iff (a : ℝ) :
  (∀ x : ℝ, f_deriv a x ≠ 0) ↔ 0 ≤ a ∧ a ≤ 3 :=
sorry

end no_extreme_points_iff_l2239_223905


namespace division_remainder_problem_solution_l2239_223907

theorem division_remainder (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = divisor * quotient + remainder →
  remainder < divisor →
  dividend / divisor = quotient →
  dividend % divisor = remainder :=
by sorry

theorem problem_solution :
  14 / 3 = 4 →
  14 % 3 = 2 :=
by sorry

end division_remainder_problem_solution_l2239_223907


namespace negative_reciprocal_of_0125_l2239_223915

def negative_reciprocal (a b : ℝ) : Prop := a * b = -1

theorem negative_reciprocal_of_0125 :
  negative_reciprocal 0.125 (-8) := by
  sorry

end negative_reciprocal_of_0125_l2239_223915


namespace tom_build_time_l2239_223974

theorem tom_build_time (avery_time : ℝ) (joint_work_time : ℝ) (tom_finish_time : ℝ) :
  avery_time = 3 →
  joint_work_time = 1 →
  tom_finish_time = 39.99999999999999 / 60 →
  ∃ (tom_solo_time : ℝ),
    (1 / avery_time + 1 / tom_solo_time) * joint_work_time + 
    (1 / tom_solo_time) * tom_finish_time = 1 ∧
    tom_solo_time = 2.5 := by
  sorry

end tom_build_time_l2239_223974


namespace correct_num_pigs_l2239_223965

/-- The number of pigs Randy has -/
def num_pigs : ℕ := 2

/-- The amount of feed per pig per day in pounds -/
def feed_per_pig_per_day : ℕ := 10

/-- The total amount of feed for all pigs per week in pounds -/
def total_feed_per_week : ℕ := 140

/-- Theorem stating that the number of pigs is correct given the feeding conditions -/
theorem correct_num_pigs : 
  num_pigs * feed_per_pig_per_day * 7 = total_feed_per_week := by
  sorry


end correct_num_pigs_l2239_223965


namespace matrix_cube_sum_l2239_223995

/-- Given a 3x3 complex matrix N of the form [d e f; e f d; f d e] where N^2 = I and def = -1,
    the possible values of d^3 + e^3 + f^3 are 2 and 4. -/
theorem matrix_cube_sum (d e f : ℂ) : 
  let N : Matrix (Fin 3) (Fin 3) ℂ := !![d, e, f; e, f, d; f, d, e]
  (N ^ 2 = 1 ∧ d * e * f = -1) →
  (d^3 + e^3 + f^3 = 2 ∨ d^3 + e^3 + f^3 = 4) :=
by sorry

end matrix_cube_sum_l2239_223995


namespace toy_set_pricing_l2239_223936

/-- Represents the cost and sales data for Asian Games mascot plush toy sets -/
structure ToySetData where
  cost_price : ℝ
  batch1_quantity : ℕ
  batch1_price : ℝ
  batch2_quantity : ℕ
  batch2_price : ℝ
  total_profit : ℝ
  batch3_quantity : ℕ
  batch3_initial_price : ℝ
  day1_sales : ℕ
  day2_sales : ℕ
  sales_increase_per_reduction : ℝ
  reduction_step : ℝ
  day3_profit : ℝ

/-- Theorem stating the cost price and required price reduction for the toy sets -/
theorem toy_set_pricing (data : ToySetData) :
  data.cost_price = 60 ∧
  (∃ (price_reduction : ℝ), price_reduction = 10 ∧
    (data.batch1_quantity * (data.batch1_price - data.cost_price) +
     data.batch2_quantity * (data.batch2_price - data.cost_price) = data.total_profit) ∧
    (data.day1_sales * (data.batch3_initial_price - data.cost_price) +
     data.day2_sales * (data.batch3_initial_price - data.cost_price) +
     (data.day2_sales + price_reduction / data.reduction_step * data.sales_increase_per_reduction) *
       (data.batch3_initial_price - price_reduction - data.cost_price) = data.day3_profit)) :=
by sorry

end toy_set_pricing_l2239_223936


namespace all_nines_multiple_l2239_223912

theorem all_nines_multiple (p : Nat) (hp : Nat.Prime p) (hp2 : p ≠ 2) (hp5 : p ≠ 5) :
  ∃ k : Nat, k > 0 ∧ (((10^k - 1) / 9) % p = 0) := by
  sorry

end all_nines_multiple_l2239_223912


namespace abc_product_l2239_223926

theorem abc_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 24) (hac : a * c = 40) (hbc : b * c = 60) :
  a * b * c = 240 := by
  sorry

end abc_product_l2239_223926


namespace max_value_implies_a_l2239_223956

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * a * x + 1

theorem max_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧ 
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = -3 ∨ a = 3/8 := by
sorry

end max_value_implies_a_l2239_223956


namespace distance_sum_is_18_l2239_223944

/-- Given three points A, B, and D in a plane, prove that the sum of distances AD and BD is 18 -/
theorem distance_sum_is_18 (A B D : ℝ × ℝ) : 
  A = (16, 0) → B = (1, 1) → D = (4, 5) → 
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 18 := by
  sorry

end distance_sum_is_18_l2239_223944


namespace area_square_with_semicircles_l2239_223948

/-- The area of a shape formed by a square with semicircles on each side -/
theorem area_square_with_semicircles (π : ℝ) : 
  let square_side : ℝ := 2 * π
  let square_area : ℝ := square_side ^ 2
  let semicircle_radius : ℝ := square_side / 2
  let semicircle_area : ℝ := 1 / 2 * π * semicircle_radius ^ 2
  let total_semicircle_area : ℝ := 4 * semicircle_area
  let total_area : ℝ := square_area + total_semicircle_area
  total_area = 2 * π^2 * (π + 2) :=
by sorry

end area_square_with_semicircles_l2239_223948


namespace turban_price_turban_price_proof_l2239_223918

/-- The price of a turban given the following conditions:
  - The total salary for one year is Rs. 90 plus one turban
  - The servant leaves after 9 months
  - The servant receives Rs. 40 and the turban after 9 months
-/
theorem turban_price : ℝ → Prop :=
  fun price =>
    let total_salary : ℝ := 90 + price
    let months_worked : ℝ := 9
    let total_months : ℝ := 12
    let received_amount : ℝ := 40 + price
    (months_worked / total_months) * total_salary = received_amount →
    price = 110

/-- Proof of the turban price theorem -/
theorem turban_price_proof : ∃ price, turban_price price := by
  sorry

end turban_price_turban_price_proof_l2239_223918


namespace integral_sqrt_minus_x_squared_l2239_223958

theorem integral_sqrt_minus_x_squared :
  ∫ x in (0 : ℝ)..1, (Real.sqrt (1 - (x - 1)^2) - x^2) = π / 4 - 1 / 3 := by
  sorry

end integral_sqrt_minus_x_squared_l2239_223958


namespace line_circle_intersection_l2239_223985

noncomputable def m : ℝ → ℝ → ℝ → ℝ := sorry

theorem line_circle_intersection (m : ℝ) :
  (∃ A B : ℝ × ℝ,
    (A.1 - m * A.2 + 1 = 0 ∧ (A.1 - 1)^2 + A.2^2 = 4) ∧
    (B.1 - m * B.2 + 1 = 0 ∧ (B.1 - 1)^2 + B.2^2 = 4) ∧
    A ≠ B) →
  (let C : ℝ × ℝ := (1, 0);
   ∃ A B : ℝ × ℝ,
    (A.1 - m * A.2 + 1 = 0 ∧ (A.1 - 1)^2 + A.2^2 = 4) ∧
    (B.1 - m * B.2 + 1 = 0 ∧ (B.1 - 1)^2 + B.2^2 = 4) ∧
    A ≠ B ∧
    abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2)) / 2 = 8/5) →
  m = 2 ∨ m = -2 ∨ m = 1/2 ∨ m = -1/2 :=
sorry

end line_circle_intersection_l2239_223985


namespace updated_mean_after_decrement_l2239_223979

theorem updated_mean_after_decrement (n : ℕ) (original_mean : ℝ) (decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 6 →
  (n * original_mean - n * decrement) / n = 194 := by
  sorry

end updated_mean_after_decrement_l2239_223979


namespace equation1_solutions_equation2_solutions_l2239_223967

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := x^2 - 4*x + 1 = 0
def equation2 (x : ℝ) : Prop := 2*x^2 - 3*x + 1 = 0

-- Theorem for equation 1
theorem equation1_solutions :
  ∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 3 ∧ x₂ = 2 - Real.sqrt 3 ∧
  equation1 x₁ ∧ equation1 x₂ ∧
  ∀ x : ℝ, equation1 x → x = x₁ ∨ x = x₂ :=
sorry

-- Theorem for equation 2
theorem equation2_solutions :
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 1/2 ∧
  equation2 x₁ ∧ equation2 x₂ ∧
  ∀ x : ℝ, equation2 x → x = x₁ ∨ x = x₂ :=
sorry

end equation1_solutions_equation2_solutions_l2239_223967


namespace correct_costs_l2239_223931

/-- Represents the costs of a pen, pencil, and ink refill -/
structure ItemCosts where
  pen : ℚ
  pencil : ℚ
  ink_refill : ℚ

/-- Checks if the given costs satisfy the problem conditions -/
def satisfies_conditions (costs : ItemCosts) : Prop :=
  costs.pen + costs.pencil + costs.ink_refill = 2.4 ∧
  costs.pen = costs.ink_refill + 1.5 ∧
  costs.pencil = costs.ink_refill - 0.4

/-- Theorem stating the correct costs for the items -/
theorem correct_costs :
  ∃ (costs : ItemCosts),
    satisfies_conditions costs ∧
    costs.pen = 1.93 ∧
    costs.pencil = 0.03 ∧
    costs.ink_refill = 0.43 :=
by
  sorry

end correct_costs_l2239_223931


namespace salary_savings_percentage_l2239_223986

theorem salary_savings_percentage 
  (salary : ℝ) 
  (savings_after_increase : ℝ) 
  (expense_increase_percentage : ℝ) :
  salary = 5750 →
  savings_after_increase = 230 →
  expense_increase_percentage = 20 →
  ∃ (savings_percentage : ℝ),
    savings_percentage = 20 ∧
    savings_after_increase = salary - (1 + expense_increase_percentage / 100) * ((100 - savings_percentage) / 100 * salary) :=
by sorry

end salary_savings_percentage_l2239_223986


namespace chocolate_fraction_is_11_24_l2239_223966

/-- The fraction of students who chose chocolate ice cream -/
def chocolate_fraction (chocolate strawberry vanilla : ℕ) : ℚ :=
  chocolate / (chocolate + strawberry + vanilla)

/-- Theorem stating that the fraction of students who chose chocolate ice cream is 11/24 -/
theorem chocolate_fraction_is_11_24 :
  chocolate_fraction 11 5 8 = 11 / 24 := by
  sorry

end chocolate_fraction_is_11_24_l2239_223966


namespace cubic_equation_solution_l2239_223973

theorem cubic_equation_solution :
  ∀ x y z : ℤ, x^3 + y^3 + z^3 - 3*x*y*z = 2003 ↔
    ((x = 668 ∧ y = 668 ∧ z = 667) ∨
     (x = 668 ∧ y = 667 ∧ z = 668) ∨
     (x = 667 ∧ y = 668 ∧ z = 668)) :=
by sorry

end cubic_equation_solution_l2239_223973


namespace cyclic_inequality_l2239_223988

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z + x * y + y * z + z * x = 4) :
  Real.sqrt ((x * y + x + y) / z) + Real.sqrt ((y * z + y + z) / x) + Real.sqrt ((z * x + z + x) / y) ≥ 
  3 * Real.sqrt (3 * (x + 2) * (y + 2) * (z + 2) / ((2 * x + 1) * (2 * y + 1) * (2 * z + 1))) := by
  sorry

end cyclic_inequality_l2239_223988


namespace matrix_subtraction_l2239_223917

theorem matrix_subtraction : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 6, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, -8; 3, 7]
  let C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 5; 3, -2]
  A - B = C := by sorry

end matrix_subtraction_l2239_223917


namespace sum_of_five_consecutive_squares_not_perfect_square_l2239_223934

theorem sum_of_five_consecutive_squares_not_perfect_square (n : ℤ) : 
  ¬ ∃ m : ℤ, 5 * (n^2 + 2) = m^2 := by
  sorry

end sum_of_five_consecutive_squares_not_perfect_square_l2239_223934


namespace greatest_k_value_l2239_223914

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 117 :=
sorry

end greatest_k_value_l2239_223914


namespace cyclic_sum_inequality_l2239_223972

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 / ((2*x + y) * (2*x + z)) + y^2 / ((2*y + x) * (2*y + z)) + z^2 / ((2*z + x) * (2*z + y)) ≤ 1/3 := by
  sorry

end cyclic_sum_inequality_l2239_223972


namespace wire_poles_problem_l2239_223945

theorem wire_poles_problem (wire_length : ℝ) (distance_increase : ℝ) : 
  wire_length = 5000 →
  distance_increase = 1.25 →
  ∃ (n : ℕ), 
    n > 1 ∧
    wire_length / (n - 1 : ℝ) + distance_increase = wire_length / (n - 2 : ℝ) ∧
    n = 65 := by
  sorry

end wire_poles_problem_l2239_223945


namespace sum_of_valid_b_is_six_l2239_223932

/-- A function that checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m

/-- The sum of all positive integer values of b for which the quadratic equation 3x^2 + 7x + b = 0 has rational roots -/
def sum_of_valid_b : ℕ := sorry

/-- The main theorem stating that the sum of valid b values is 6 -/
theorem sum_of_valid_b_is_six : sum_of_valid_b = 6 := by
  sorry

end sum_of_valid_b_is_six_l2239_223932


namespace three_in_all_curriculums_l2239_223951

/-- Represents the number of people in each curriculum or combination of curriculums -/
structure CurriculumParticipants where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  cookingAndWeaving : ℕ
  allCurriculums : ℕ

/-- Theorem stating that given the conditions, 3 people participate in all curriculums -/
theorem three_in_all_curriculums (p : CurriculumParticipants) 
  (h1 : p.yoga = 25)
  (h2 : p.cooking = 15)
  (h3 : p.weaving = 8)
  (h4 : p.cookingOnly = 2)
  (h5 : p.cookingAndYoga = 7)
  (h6 : p.cookingAndWeaving = 3)
  (h7 : p.cooking = p.cookingOnly + p.cookingAndYoga + p.cookingAndWeaving + p.allCurriculums) :
  p.allCurriculums = 3 := by
  sorry


end three_in_all_curriculums_l2239_223951


namespace x_2007_equals_2_l2239_223941

def x : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (1 + x (n + 1)) / x n

theorem x_2007_equals_2 : x 2007 = 2 := by
  sorry

end x_2007_equals_2_l2239_223941


namespace impossible_transformation_number_54_impossible_l2239_223981

/-- Represents the allowed operations on the number -/
inductive Operation
  | Multiply2
  | Multiply3
  | Divide2
  | Divide3

/-- Applies a single operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Multiply2 => n * 2
  | Operation.Multiply3 => n * 3
  | Operation.Divide2 => n / 2
  | Operation.Divide3 => n / 3

/-- Applies a sequence of operations to a number -/
def applyOperations (n : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation n

/-- Returns the sum of exponents in the prime factorization of a number -/
def sumOfExponents (n : ℕ) : ℕ :=
  (Nat.factorization n).sum (fun _ e => e)

/-- Theorem stating that it's impossible to transform 12 into 54 with exactly 60 operations -/
theorem impossible_transformation :
  ∀ (ops : List Operation), ops.length = 60 → applyOperations 12 ops ≠ 54 := by
  sorry

/-- Corollary: The number 54 cannot appear on the screen after exactly one minute -/
theorem number_54_impossible : ∃ (ops : List Operation), ops.length = 60 ∧ applyOperations 12 ops = 54 → False := by
  sorry

end impossible_transformation_number_54_impossible_l2239_223981


namespace base_seven_digits_of_2401_l2239_223949

theorem base_seven_digits_of_2401 : ∃ n : ℕ, n > 0 ∧ 7^(n-1) ≤ 2401 ∧ 2401 < 7^n ∧ n = 5 := by
  sorry

end base_seven_digits_of_2401_l2239_223949


namespace election_majority_proof_l2239_223913

theorem election_majority_proof (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 440 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 176 := by
  sorry

end election_majority_proof_l2239_223913


namespace arithmetic_sequence_properties_l2239_223982

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum function
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Properties of a specific arithmetic sequence -/
def SpecificSequence (seq : ArithmeticSequence) : Prop :=
  seq.S 5 < seq.S 6 ∧ seq.S 6 = seq.S 7 ∧ seq.S 7 > seq.S 8

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
  (h : SpecificSequence seq) : 
  (∃ d, ∀ n, seq.a (n + 1) - seq.a n = d ∧ d < 0) ∧ 
  seq.S 9 < seq.S 5 ∧
  seq.a 7 = 0 ∧
  (∀ n, seq.S n ≤ seq.S 6) :=
sorry

end arithmetic_sequence_properties_l2239_223982


namespace gold_hoard_problem_l2239_223983

theorem gold_hoard_problem (total_per_brother : ℝ) (eldest_gold : ℝ) (eldest_silver_fraction : ℝ)
  (total_silver : ℝ) (h1 : total_per_brother = 100)
  (h2 : eldest_gold = 30)
  (h3 : eldest_silver_fraction = 1/5)
  (h4 : total_silver = 350) :
  eldest_gold + (total_silver - eldest_silver_fraction * total_silver) = 50 := by
  sorry


end gold_hoard_problem_l2239_223983


namespace inequality_equivalence_l2239_223908

theorem inequality_equivalence (x : ℝ) : 3 * x + 4 < 5 * x - 6 ↔ x > 5 := by
  sorry

end inequality_equivalence_l2239_223908


namespace absolute_value_equation_solution_l2239_223975

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x - 5| :=
by
  -- The unique solution is x = 4
  use 4
  sorry

end absolute_value_equation_solution_l2239_223975
