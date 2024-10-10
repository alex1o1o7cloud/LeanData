import Mathlib

namespace algorithm_can_contain_all_structures_l2137_213710

/-- Represents the types of logical structures in algorithms -/
inductive LogicalStructure
  | Sequential
  | Conditional
  | Loop

/-- Represents an algorithm -/
structure Algorithm where
  structures : List LogicalStructure

/-- Theorem stating that an algorithm can contain all three types of logical structures -/
theorem algorithm_can_contain_all_structures :
  ∃ (a : Algorithm), (LogicalStructure.Sequential ∈ a.structures) ∧
                     (LogicalStructure.Conditional ∈ a.structures) ∧
                     (LogicalStructure.Loop ∈ a.structures) :=
by sorry


end algorithm_can_contain_all_structures_l2137_213710


namespace ursula_shopping_cost_l2137_213779

/-- Represents the prices of items in Ursula's shopping trip -/
structure ShoppingPrices where
  butter : ℝ
  bread : ℝ
  cheese : ℝ
  tea : ℝ
  eggs : ℝ
  honey : ℝ

/-- Calculates the total cost of all items -/
def totalCost (prices : ShoppingPrices) : ℝ :=
  prices.butter + prices.bread + prices.cheese + prices.tea + prices.eggs + prices.honey

/-- Theorem stating the conditions and the result of Ursula's shopping trip -/
theorem ursula_shopping_cost (prices : ShoppingPrices) : 
  prices.bread = prices.butter / 2 →
  prices.butter = 0.8 * prices.cheese →
  prices.tea = 1.5 * (prices.bread + prices.butter + prices.cheese) →
  prices.tea = 10 →
  prices.eggs = prices.bread / 2 →
  prices.honey = prices.eggs + 3 →
  abs (totalCost prices - 20.87) < 0.01 := by
  sorry


end ursula_shopping_cost_l2137_213779


namespace diagonal_planes_increment_l2137_213705

/-- The number of diagonal planes in a prism with k edges -/
def f (k : ℕ) : ℕ := k * (k - 3) / 2

/-- Theorem: The number of diagonal planes in a prism with k+1 edges
    is equal to the number of diagonal planes in a prism with k edges plus k-1 -/
theorem diagonal_planes_increment (k : ℕ) :
  f (k + 1) = f k + k - 1 := by
  sorry

end diagonal_planes_increment_l2137_213705


namespace trigonometric_equation_solution_l2137_213764

theorem trigonometric_equation_solution :
  ∀ x : ℝ, (Real.sin (3 * x) * Real.cos (3 * x) + Real.cos (3 * x) * Real.sin (3 * x) = 3 / 8) ↔
  (∃ k : ℤ, x = (7.5 * π / 180) + k * (π / 2) ∨ x = (37.5 * π / 180) + k * (π / 2)) :=
by sorry

end trigonometric_equation_solution_l2137_213764


namespace simple_interest_problem_l2137_213716

theorem simple_interest_problem (P R : ℝ) (h : P * (R + 10) * 8 / 100 - P * R * 8 / 100 = 150) : P = 187.50 := by
  sorry

end simple_interest_problem_l2137_213716


namespace football_games_per_month_l2137_213750

theorem football_games_per_month 
  (total_games : ℕ) 
  (num_months : ℕ) 
  (h1 : total_games = 323) 
  (h2 : num_months = 17) 
  (h3 : total_games % num_months = 0) : 
  total_games / num_months = 19 := by
sorry

end football_games_per_month_l2137_213750


namespace fraction_simplification_l2137_213704

theorem fraction_simplification (x : ℝ) : (2*x - 3)/4 + (5 - 4*x)/3 = (-10*x + 11)/12 := by
  sorry

end fraction_simplification_l2137_213704


namespace rattlesnake_count_l2137_213772

theorem rattlesnake_count (total : ℕ) (pythons boa_constrictors rattlesnakes vipers : ℕ) :
  total = 350 ∧
  total = pythons + boa_constrictors + rattlesnakes + vipers ∧
  pythons = 2 * boa_constrictors ∧
  vipers = rattlesnakes / 2 ∧
  boa_constrictors = 60 ∧
  pythons + vipers = (40 * total) / 100 →
  rattlesnakes = 40 := by
sorry

end rattlesnake_count_l2137_213772


namespace quadratic_equation_result_l2137_213722

theorem quadratic_equation_result (x : ℝ) (h : 6 * x^2 + 9 = 4 * x + 16) : (12 * x - 4)^2 = 188 := by
  sorry

end quadratic_equation_result_l2137_213722


namespace smallest_solution_quadratic_l2137_213762

theorem smallest_solution_quadratic (x : ℝ) :
  (8 * x^2 - 38 * x + 35 = 0) → x ≥ 1.25 :=
by sorry

end smallest_solution_quadratic_l2137_213762


namespace geometric_sequence_product_l2137_213747

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) / a n = a 2 / a 1

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a → a 2 * a 4 * a 12 = 64 → a 6 = 4 := by
  sorry

end geometric_sequence_product_l2137_213747


namespace square_side_properties_l2137_213751

theorem square_side_properties (a : ℝ) (h : a > 0) (area_eq : a^2 = 10) :
  a = Real.sqrt 10 ∧ a^2 - 10 = 0 ∧ 3 < a ∧ a < 4 := by
  sorry

end square_side_properties_l2137_213751


namespace max_gcd_consecutive_terms_l2137_213769

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_terms : 
  ∃ (k : ℕ), ∀ (n : ℕ), Nat.gcd (b n) (b (n + 1)) ≤ k ∧ 
  ∃ (m : ℕ), Nat.gcd (b m) (b (m + 1)) = k :=
sorry

end max_gcd_consecutive_terms_l2137_213769


namespace train_speed_with_stoppages_l2137_213765

/-- Given a train that travels at 80 km/h without stoppages and stops for 15 minutes every hour,
    its average speed with stoppages is 60 km/h. -/
theorem train_speed_with_stoppages :
  let speed_without_stoppages : ℝ := 80
  let stop_time_per_hour : ℝ := 15/60
  let speed_with_stoppages : ℝ := speed_without_stoppages * (1 - stop_time_per_hour)
  speed_with_stoppages = 60 := by sorry

end train_speed_with_stoppages_l2137_213765


namespace pure_imaginary_complex_number_l2137_213746

theorem pure_imaginary_complex_number (θ : Real) : 
  θ ∈ Set.Icc 0 (2 * Real.pi) →
  (∃ (y : Real), (Complex.cos θ + Complex.I) * (2 * Complex.sin θ - Complex.I) = Complex.I * y) →
  θ = 3 * Real.pi / 4 ∨ θ = 7 * Real.pi / 4 :=
by sorry

end pure_imaginary_complex_number_l2137_213746


namespace eighth_term_value_l2137_213794

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  first_third_product : a 1 * a 3 = 4
  ninth_term : a 9 = 256

/-- The 8th term of the geometric sequence is either 128 or -128 -/
theorem eighth_term_value (seq : GeometricSequence) :
  seq.a 8 = 128 ∨ seq.a 8 = -128 := by
  sorry

end eighth_term_value_l2137_213794


namespace surface_area_of_rearranged_cube_l2137_213763

-- Define the cube and its properties
def cube_volume : ℝ := 8
def cube_side_length : ℝ := 2

-- Define the cuts
def first_cut_distance : ℝ := 1
def second_cut_distance : ℝ := 0.5

-- Define the heights of the pieces
def height_X : ℝ := first_cut_distance
def height_Y : ℝ := second_cut_distance
def height_Z : ℝ := cube_side_length - (first_cut_distance + second_cut_distance)

-- Define the total width of the rearranged pieces
def total_width : ℝ := height_X + height_Y + height_Z

-- Theorem statement
theorem surface_area_of_rearranged_cube :
  cube_volume = cube_side_length ^ 3 →
  (2 * cube_side_length * cube_side_length +    -- Top and bottom surfaces
   2 * total_width * cube_side_length +         -- Side surfaces
   2 * cube_side_length * cube_side_length) = 46 := by
sorry

end surface_area_of_rearranged_cube_l2137_213763


namespace sam_chewing_gums_l2137_213798

theorem sam_chewing_gums (total : ℕ) (mary : ℕ) (sue : ℕ) (h1 : total = 30) (h2 : mary = 5) (h3 : sue = 15) : 
  total - mary - sue = 10 := by
  sorry

end sam_chewing_gums_l2137_213798


namespace arithmetic_geometric_relation_l2137_213727

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ r : ℝ), r ≠ 0 ∧ ∀ n, b n = b₁ * r^(n - 1)

/-- The main theorem -/
theorem arithmetic_geometric_relation
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h_non_zero : ∀ n, a n ≠ 0)
  (h_relation : a 1 - (a 7)^2 + a 13 = 0)
  (h_equal : b 7 = a 7) :
  b 11 = 4 := by
  sorry

end arithmetic_geometric_relation_l2137_213727


namespace negation_equivalence_l2137_213792

theorem negation_equivalence :
  (¬ ∀ x : ℝ, |x - 2| + |x - 4| > 3) ↔ (∃ x : ℝ, |x - 2| + |x - 4| ≤ 3) := by
  sorry

end negation_equivalence_l2137_213792


namespace inequality_proof_l2137_213707

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := by
  sorry

end inequality_proof_l2137_213707


namespace skylar_donation_amount_l2137_213744

theorem skylar_donation_amount (start_age : ℕ) (current_age : ℕ) (total_donation : ℕ) : 
  start_age = 13 →
  current_age = 33 →
  total_donation = 105000 →
  (total_donation : ℚ) / ((current_age - start_age) : ℚ) = 5250 := by
  sorry

end skylar_donation_amount_l2137_213744


namespace remaining_pieces_count_l2137_213739

/-- Represents the number of pieces in a standard chess set -/
def standard_set : Nat := 32

/-- Represents the total number of missing pieces -/
def missing_pieces : Nat := 12

/-- Represents the number of missing kings -/
def missing_kings : Nat := 1

/-- Represents the number of missing queens -/
def missing_queens : Nat := 2

/-- Represents the number of missing knights -/
def missing_knights : Nat := 3

/-- Represents the number of missing pawns -/
def missing_pawns : Nat := 6

/-- Theorem stating that the number of remaining pieces is 20 -/
theorem remaining_pieces_count :
  standard_set - missing_pieces = 20 :=
by
  sorry

#check remaining_pieces_count

end remaining_pieces_count_l2137_213739


namespace sleep_ratio_theorem_l2137_213791

/-- Represents Billy's sleep pattern over four nights -/
structure SleepPattern where
  first_night : ℝ
  second_night : ℝ
  third_night : ℝ
  fourth_night : ℝ

/-- Theorem stating the ratio of the fourth night's sleep to the third night's sleep -/
theorem sleep_ratio_theorem (s : SleepPattern) 
  (h1 : s.first_night = 6)
  (h2 : s.second_night = s.first_night + 2)
  (h3 : s.third_night = s.second_night / 2)
  (h4 : s.fourth_night = s.third_night * (s.fourth_night / s.third_night))
  (h5 : s.first_night + s.second_night + s.third_night + s.fourth_night = 30) :
  s.fourth_night / s.third_night = 3 := by
  sorry

end sleep_ratio_theorem_l2137_213791


namespace line_mb_product_l2137_213725

/-- A line passing through two points (0, -2) and (2, 4) has mb = -6 --/
theorem line_mb_product (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  (-2 : ℝ) = b →                -- Line passes through (0, -2)
  4 = m * 2 + b →               -- Line passes through (2, 4)
  m * b = -6 := by sorry

end line_mb_product_l2137_213725


namespace trig_equality_l2137_213717

theorem trig_equality (α β γ : Real) 
  (h : (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = 
       (1 + Real.sin α) * (1 + Real.sin β) * (1 + Real.sin γ)) : 
  (1 - Real.sin α) * (1 - Real.sin β) * (1 - Real.sin γ) = 
  |Real.cos α * Real.cos β * Real.cos γ| := by
  sorry

end trig_equality_l2137_213717


namespace yellow_marbles_count_l2137_213773

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by
  sorry

end yellow_marbles_count_l2137_213773


namespace english_test_questions_l2137_213790

theorem english_test_questions (math_questions : ℕ) (math_correct_percentage : ℚ)
  (english_correct_percentage : ℚ) (total_correct : ℕ) :
  math_questions = 40 →
  math_correct_percentage = 75 / 100 →
  english_correct_percentage = 98 / 100 →
  total_correct = 79 →
  ∃ (english_questions : ℕ),
    english_questions = 50 ∧
    (math_questions : ℚ) * math_correct_percentage +
    (english_questions : ℚ) * english_correct_percentage = total_correct :=
by sorry

end english_test_questions_l2137_213790


namespace additional_spend_needed_l2137_213721

-- Define the minimum spend for free delivery
def min_spend : ℝ := 35

-- Define the prices and quantities of items
def chicken_price : ℝ := 6
def chicken_quantity : ℝ := 1.5
def lettuce_price : ℝ := 3
def cherry_tomatoes_price : ℝ := 2.5
def sweet_potato_price : ℝ := 0.75
def sweet_potato_quantity : ℕ := 4
def broccoli_price : ℝ := 2
def broccoli_quantity : ℕ := 2
def brussel_sprouts_price : ℝ := 2.5

-- Calculate the total cost of items in the cart
def total_cost : ℝ :=
  chicken_price * chicken_quantity +
  lettuce_price +
  cherry_tomatoes_price +
  sweet_potato_price * sweet_potato_quantity +
  broccoli_price * broccoli_quantity +
  brussel_sprouts_price

-- Theorem: The difference between min_spend and total_cost is 11
theorem additional_spend_needed : min_spend - total_cost = 11 := by
  sorry

end additional_spend_needed_l2137_213721


namespace tetrahedron_sphere_probability_l2137_213726

/-- Regular tetrahedron with inscribed and circumscribed spheres -/
structure RegularTetrahedron where
  r : ℝ  -- radius of inscribed sphere
  R : ℝ  -- radius of circumscribed sphere
  h : R = 3 * r  -- relationship between R and r

/-- External sphere tangent to a face of the tetrahedron and the circumscribed sphere -/
structure ExternalSphere (t : RegularTetrahedron) where
  radius : ℝ
  h : radius = 1.5 * t.r

/-- The probability theorem for the tetrahedron and spheres setup -/
theorem tetrahedron_sphere_probability (t : RegularTetrahedron) 
  (e : ExternalSphere t) (n : ℕ) (h_n : n = 4) :
  let v_external := n * (4 / 3 * Real.pi * e.radius ^ 3)
  let v_circumscribed := 4 / 3 * Real.pi * t.R ^ 3
  v_external ≤ v_circumscribed ∧ 
  v_external / v_circumscribed = 1 / 2 := by
  sorry

end tetrahedron_sphere_probability_l2137_213726


namespace orange_seller_gain_l2137_213776

/-- The percentage gain a man wants to achieve when selling oranges -/
def desired_gain (initial_rate : ℚ) (loss_percent : ℚ) (new_rate : ℚ) : ℚ :=
  let cost_price := 1 / (initial_rate * (1 - loss_percent / 100))
  let new_price := 1 / new_rate
  (new_price / cost_price - 1) * 100

/-- Theorem stating the desired gain for specific selling rates and loss percentage -/
theorem orange_seller_gain :
  desired_gain 18 8 (11420689655172414 / 1000000000000000) = 45 := by
  sorry

end orange_seller_gain_l2137_213776


namespace probability_specific_coin_sequence_l2137_213700

/-- The probability of getting a specific sequence of coin flips -/
def probability_specific_sequence (n : ℕ) (p : ℚ) : ℚ :=
  p^n

/-- The number of coin flips -/
def num_flips : ℕ := 10

/-- The probability of getting tails on a single flip -/
def prob_tails : ℚ := 1/2

/-- Theorem: The probability of getting the sequence TTT HHHH THT in 10 coin flips -/
theorem probability_specific_coin_sequence :
  probability_specific_sequence num_flips prob_tails = 1/1024 := by
  sorry

end probability_specific_coin_sequence_l2137_213700


namespace complement_of_equal_angles_is_proposition_l2137_213781

-- Define what a proposition is in this context
def is_proposition (statement : String) : Prop :=
  -- A statement is a proposition if it can be true or false
  ∃ (truth_value : Bool), (truth_value = true ∨ truth_value = false)

-- The statement we want to prove is a proposition
def complement_of_equal_angles_statement : String :=
  "The complement of equal angles are equal"

-- Theorem stating that the given statement is a proposition
theorem complement_of_equal_angles_is_proposition :
  is_proposition complement_of_equal_angles_statement :=
sorry

end complement_of_equal_angles_is_proposition_l2137_213781


namespace constant_term_proof_l2137_213706

/-- Given an equation (ax + w)(cx + d) = 6x^2 + x - 12, where a, w, c, and d are real numbers
    whose absolute values sum to 12, prove that the constant term in the expanded form is -12. -/
theorem constant_term_proof (a w c d : ℝ) 
    (eq : ∀ x, (a * x + w) * (c * x + d) = 6 * x^2 + x - 12)
    (sum_abs : |a| + |w| + |c| + |d| = 12) :
    w * d = -12 := by
  sorry

end constant_term_proof_l2137_213706


namespace fourth_circle_radius_l2137_213715

/-- Represents a configuration of seven circles tangent to each other and two lines -/
structure CircleConfiguration where
  radii : Fin 7 → ℝ
  is_geometric_sequence : ∃ r : ℝ, ∀ i : Fin 6, radii i.succ = radii i * r
  smallest_radius : radii 0 = 6
  largest_radius : radii 6 = 24

/-- The theorem stating that the radius of the fourth circle is 12 -/
theorem fourth_circle_radius (config : CircleConfiguration) : config.radii 3 = 12 := by
  sorry

end fourth_circle_radius_l2137_213715


namespace english_spanish_difference_l2137_213711

def hours_english : ℕ := 7
def hours_chinese : ℕ := 2
def hours_spanish : ℕ := 4

theorem english_spanish_difference : hours_english - hours_spanish = 3 := by
  sorry

end english_spanish_difference_l2137_213711


namespace newspaper_cost_difference_l2137_213702

/-- The amount Grant spends yearly on newspaper delivery -/
def grant_yearly_cost : ℝ := 200

/-- The amount Juanita spends on newspapers Monday through Saturday -/
def juanita_weekday_cost : ℝ := 0.5

/-- The amount Juanita spends on newspapers on Sunday -/
def juanita_sunday_cost : ℝ := 2

/-- The number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- The number of weekdays (Monday through Saturday) -/
def weekdays : ℕ := 6

theorem newspaper_cost_difference : 
  (weekdays * juanita_weekday_cost + juanita_sunday_cost) * weeks_per_year - grant_yearly_cost = 60 := by
  sorry

end newspaper_cost_difference_l2137_213702


namespace six_students_solved_only_B_l2137_213797

/-- Represents the number of students who solved each combination of problems -/
structure ProblemSolvers where
  a : ℕ  -- only A
  b : ℕ  -- only B
  c : ℕ  -- only C
  d : ℕ  -- A and B
  e : ℕ  -- A and C
  f : ℕ  -- B and C
  g : ℕ  -- A, B, and C

/-- The conditions of the math competition problem -/
def MathCompetitionConditions (s : ProblemSolvers) : Prop :=
  -- Total number of students is 25
  s.a + s.b + s.c + s.d + s.e + s.f + s.g = 25 ∧
  -- Among students who didn't solve A, number solving B is twice the number solving C
  s.b + s.f = 2 * (s.c + s.f) ∧
  -- Number of students solving only A is one more than number of students solving A among remaining students
  s.a = s.d + s.e + s.g + 1 ∧
  -- Among students solving only one problem, half didn't solve A
  s.a = s.b + s.c

/-- The theorem stating that 6 students solved only problem B -/
theorem six_students_solved_only_B (s : ProblemSolvers) 
  (h : MathCompetitionConditions s) : s.b = 6 := by
  sorry

end six_students_solved_only_B_l2137_213797


namespace inverse_implies_negation_l2137_213719

theorem inverse_implies_negation (p : Prop) : 
  (¬p → p) → ¬p :=
sorry

end inverse_implies_negation_l2137_213719


namespace inequality_proof_l2137_213770

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + 4*b^2)) + (b^3 / (b^2 + 4*c^2)) + (c^3 / (c^2 + 4*a^2)) ≥ (a + b + c) / 5 := by
  sorry

end inequality_proof_l2137_213770


namespace right_triangle_median_to_hypotenuse_l2137_213787

/-- Given a right triangle DEF with hypotenuse DE = 15, DF = 9, and EF = 12,
    the distance from F to the midpoint of DE is 7.5 -/
theorem right_triangle_median_to_hypotenuse (DE DF EF : ℝ) :
  DE = 15 →
  DF = 9 →
  EF = 12 →
  DE^2 = DF^2 + EF^2 →
  (DE / 2 : ℝ) = 7.5 := by
  sorry

end right_triangle_median_to_hypotenuse_l2137_213787


namespace kenneth_earnings_l2137_213789

theorem kenneth_earnings (spent_percentage : Real) (remaining_amount : Real) (total_earnings : Real) : 
  spent_percentage = 0.1 →
  remaining_amount = 405 →
  remaining_amount = (1 - spent_percentage) * total_earnings →
  total_earnings = 450 :=
by sorry

end kenneth_earnings_l2137_213789


namespace cherries_cost_correct_l2137_213775

/-- The amount Alyssa paid for cherries -/
def cherries_cost : ℚ := 985 / 100

/-- The amount Alyssa paid for grapes -/
def grapes_cost : ℚ := 1208 / 100

/-- The total amount Alyssa spent -/
def total_spent : ℚ := 2193 / 100

/-- Theorem stating that the amount Alyssa paid for cherries is correct -/
theorem cherries_cost_correct : cherries_cost = total_spent - grapes_cost := by
  sorry

end cherries_cost_correct_l2137_213775


namespace lowest_price_correct_l2137_213778

/-- Calculates the lowest price per unit to sell electronic components without making a loss. -/
def lowest_price_per_unit (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (num_units : ℕ) : ℚ :=
  (production_cost + shipping_cost + fixed_costs / num_units)

theorem lowest_price_correct (production_cost shipping_cost : ℚ) (fixed_costs : ℚ) (num_units : ℕ) :
  lowest_price_per_unit production_cost shipping_cost fixed_costs num_units =
  (production_cost * num_units + shipping_cost * num_units + fixed_costs) / num_units :=
by sorry

#eval lowest_price_per_unit 120 10 25000 100

end lowest_price_correct_l2137_213778


namespace intersection_M_N_l2137_213724

def M : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {0, 1, 2} := by sorry

end intersection_M_N_l2137_213724


namespace coffee_shop_multiple_l2137_213788

theorem coffee_shop_multiple (x : ℕ) : 
  (32 = x * 6 + 8) → x = 4 := by sorry

end coffee_shop_multiple_l2137_213788


namespace valid_arrangements_eq_48_l2137_213742

/-- The number of people in the lineup -/
def n : ℕ := 5

/-- A function that calculates the number of valid arrangements -/
def validArrangements (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of valid arrangements for 5 people is 48 -/
theorem valid_arrangements_eq_48 : validArrangements n = 48 := by sorry

end valid_arrangements_eq_48_l2137_213742


namespace vector_sum_magnitude_l2137_213730

-- Define the vectors
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![2, 2]

-- State the theorem
theorem vector_sum_magnitude :
  ‖(a + b)‖ = 5 := by sorry

end vector_sum_magnitude_l2137_213730


namespace lowest_dropped_score_l2137_213771

theorem lowest_dropped_score (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
  (a + b + c + d) / 4 = 45 →
  d ≤ a ∧ d ≤ b ∧ d ≤ c →
  (a + b + c) / 3 = 50 →
  d = 30 := by
sorry

end lowest_dropped_score_l2137_213771


namespace abc_system_property_l2137_213782

theorem abc_system_property (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (eq1 : a^2 + a = b^2)
  (eq2 : b^2 + b = c^2)
  (eq3 : c^2 + c = a^2) :
  (a - b) * (b - c) * (c - a) = 1 := by
sorry

end abc_system_property_l2137_213782


namespace sum_of_three_squares_l2137_213760

theorem sum_of_three_squares (n : ℕ+) : ¬ ∃ x y z : ℤ, x^2 + y^2 + z^2 = 8 * n + 7 := by
  sorry

end sum_of_three_squares_l2137_213760


namespace class_2_score_l2137_213749

/-- Calculates the comprehensive score for a class based on weighted scores -/
def comprehensive_score (study_score hygiene_score discipline_score activity_score : ℝ) : ℝ :=
  0.4 * study_score + 0.25 * hygiene_score + 0.25 * discipline_score + 0.1 * activity_score

/-- Theorem stating that the comprehensive score for the given class is 82.5 -/
theorem class_2_score : comprehensive_score 80 90 84 70 = 82.5 := by
  sorry

end class_2_score_l2137_213749


namespace delta_computation_l2137_213738

-- Define the custom operation
def delta (a b : ℕ) : ℕ := a^2 - b

-- State the theorem
theorem delta_computation :
  delta (5^(delta 6 2)) (4^(delta 7 3)) = 5^68 - 4^46 := by
  sorry

end delta_computation_l2137_213738


namespace find_b_l2137_213761

/-- Given two functions p and q, prove that b = 7 when p(q(5)) = 11 -/
theorem find_b (p q : ℝ → ℝ) (b : ℝ) 
  (hp : ∀ x, p x = 2 * x - 5)
  (hq : ∀ x, q x = 3 * x - b)
  (h_pq : p (q 5) = 11) : 
  b = 7 := by
sorry

end find_b_l2137_213761


namespace sum_sqrt_inequality_l2137_213729

theorem sum_sqrt_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 1) : 
  Real.sqrt (x*y/(z+x*y)) + Real.sqrt (y*z/(x+y*z)) + Real.sqrt (z*x/(y+z*x)) ≤ 3/2 := by
  sorry

end sum_sqrt_inequality_l2137_213729


namespace min_real_floor_power_inequality_l2137_213709

theorem min_real_floor_power_inequality :
  ∃ (x : ℝ), x = Real.rpow 3 (1/3) ∧
  (∀ (n : ℕ), ⌊x^n⌋ < ⌊x^(n+1)⌋) ∧
  (∀ (y : ℝ), y < x → ∃ (m : ℕ), ⌊y^m⌋ ≥ ⌊y^(m+1)⌋) :=
by sorry

end min_real_floor_power_inequality_l2137_213709


namespace ratio_x_to_y_l2137_213767

theorem ratio_x_to_y (x y : ℝ) (h : 0.8 * x = 0.2 * y) : x / y = 0.25 := by
  sorry

end ratio_x_to_y_l2137_213767


namespace isabel_math_homework_pages_l2137_213768

/-- Proves that Isabel had 2 pages of math homework given the problem conditions -/
theorem isabel_math_homework_pages :
  ∀ (total_pages math_pages reading_pages : ℕ) 
    (problems_per_page total_problems : ℕ),
  reading_pages = 4 →
  problems_per_page = 5 →
  total_problems = 30 →
  total_pages = math_pages + reading_pages →
  total_problems = total_pages * problems_per_page →
  math_pages = 2 := by
sorry


end isabel_math_homework_pages_l2137_213768


namespace smallest_w_l2137_213795

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) (hw : w > 0) 
  (h1 : is_factor (2^7) (936 * w))
  (h2 : is_factor (3^4) (936 * w))
  (h3 : is_factor (5^3) (936 * w))
  (h4 : is_factor (7^2) (936 * w))
  (h5 : is_factor (11^2) (936 * w)) :
  w ≥ 320166000 ∧ 
  (∀ v : ℕ, v > 0 → 
    is_factor (2^7) (936 * v) → 
    is_factor (3^4) (936 * v) → 
    is_factor (5^3) (936 * v) → 
    is_factor (7^2) (936 * v) → 
    is_factor (11^2) (936 * v) → 
    v ≥ w) :=
by sorry

end smallest_w_l2137_213795


namespace parabola_points_order_l2137_213733

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 + 2*x - 9

-- Define the points on the parabola
def A : ℝ × ℝ := (-2, f (-2))
def B : ℝ × ℝ := (1, f 1)
def C : ℝ × ℝ := (3, f 3)

-- Define y₁, y₂, y₃
def y₁ : ℝ := A.2
def y₂ : ℝ := B.2
def y₃ : ℝ := C.2

-- Theorem statement
theorem parabola_points_order : y₃ > y₂ ∧ y₂ > y₁ := by
  sorry

end parabola_points_order_l2137_213733


namespace dads_dimes_proof_l2137_213734

/-- The number of dimes Melanie's dad gave her -/
def dads_dimes : ℕ := 83 - (19 + 25)

/-- Melanie's initial number of dimes -/
def initial_dimes : ℕ := 19

/-- Number of dimes Melanie's mother gave her -/
def mothers_dimes : ℕ := 25

/-- Melanie's total number of dimes after receiving from both parents -/
def total_dimes : ℕ := 83

theorem dads_dimes_proof : 
  dads_dimes = total_dimes - (initial_dimes + mothers_dimes) := by
  sorry

end dads_dimes_proof_l2137_213734


namespace unique_real_root_of_cubic_l2137_213714

theorem unique_real_root_of_cubic (α : Real) (h : 0 ≤ α ∧ α ≤ Real.pi / 2) :
  ∃! x : Real, x^3 + x^2 * Real.cos α + x * Real.sin α + 1 = 0 := by
  sorry

end unique_real_root_of_cubic_l2137_213714


namespace infinitely_many_coprime_pairs_l2137_213737

theorem infinitely_many_coprime_pairs (m : ℤ) :
  ∃ f : ℕ → ℤ × ℤ, ∀ n : ℕ,
    let (x, y) := f n
    -- Condition 1: x and y are coprime
    Int.gcd x y = 1 ∧
    -- Condition 2: y divides x^2 + m
    (x^2 + m) % y = 0 ∧
    -- Condition 3: x divides y^2 + m
    (y^2 + m) % x = 0 ∧
    -- Ensure infinitely many distinct pairs
    (∀ k < n, f k ≠ f n) :=
sorry

end infinitely_many_coprime_pairs_l2137_213737


namespace photo_arrangement_count_l2137_213720

/-- The number of different arrangements for 5 students and 2 teachers in a row,
    with exactly 2 students between the teachers. -/
def photo_arrangements : ℕ := 960

/-- The number of students -/
def num_students : ℕ := 5

/-- The number of teachers -/
def num_teachers : ℕ := 2

/-- The number of students between the teachers -/
def students_between : ℕ := 2

theorem photo_arrangement_count :
  photo_arrangements = 960 ∧
  num_students = 5 ∧
  num_teachers = 2 ∧
  students_between = 2 := by sorry

end photo_arrangement_count_l2137_213720


namespace message_encoding_l2137_213777

-- Define the encoding functions
def oldEncode (s : String) : String := sorry

def newEncode (s : String) : String := sorry

-- Define the decoding function
def decode (s : String) : String := sorry

-- Theorem statement
theorem message_encoding :
  let originalMessage := "011011010011"
  let decodedMessage := decode originalMessage
  newEncode decodedMessage = "211221121" := by sorry

end message_encoding_l2137_213777


namespace starters_count_l2137_213752

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of triplets
def num_triplets : ℕ := 3

-- Define the number of twins
def num_twins : ℕ := 2

-- Define the number of starters to be chosen
def num_starters : ℕ := 6

-- Define the function to calculate the number of ways to choose starters
def choose_starters (total : ℕ) (triplets : ℕ) (twins : ℕ) (starters : ℕ) : ℕ :=
  -- No triplets and no twins
  Nat.choose (total - triplets - twins) starters +
  -- One triplet and no twins
  triplets * Nat.choose (total - triplets - twins) (starters - 1) +
  -- No triplets and one twin
  twins * Nat.choose (total - triplets - twins) (starters - 1) +
  -- One triplet and one twin
  triplets * twins * Nat.choose (total - triplets - twins) (starters - 2)

-- Theorem statement
theorem starters_count :
  choose_starters total_players num_triplets num_twins num_starters = 4752 :=
by sorry

end starters_count_l2137_213752


namespace trajectory_equation_l2137_213703

theorem trajectory_equation (x y : ℝ) :
  let A : ℝ × ℝ := (-2, 0)
  let B : ℝ × ℝ := (1, 0)
  let P : ℝ × ℝ := (x, y)
  let PA : ℝ := Real.sqrt ((x + 2)^2 + y^2)
  let PB : ℝ := Real.sqrt ((x - 1)^2 + y^2)
  PA = 2 * PB → x^2 + y^2 - 4*x = 0 := by
sorry

end trajectory_equation_l2137_213703


namespace surface_is_cone_l2137_213713

/-- A point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- The equation of the surface in spherical coordinates -/
def surface_equation (c : ℝ) (p : SphericalPoint) : Prop :=
  p.ρ = c * Real.sin p.φ

/-- Definition of a cone in spherical coordinates -/
def is_cone (S : Set SphericalPoint) : Prop :=
  ∃ c > 0, ∀ p ∈ S, surface_equation c p

theorem surface_is_cone (c : ℝ) (hc : c > 0) :
    is_cone {p : SphericalPoint | surface_equation c p} := by
  sorry

end surface_is_cone_l2137_213713


namespace price_reduction_sales_increase_l2137_213731

theorem price_reduction_sales_increase 
  (price_reduction : Real) 
  (revenue_increase : Real) 
  (sales_increase : Real) : 
  price_reduction = 0.35 → 
  revenue_increase = 0.17 → 
  (1 - price_reduction) * (1 + sales_increase) = 1 + revenue_increase → 
  sales_increase = 0.8 := by
sorry

end price_reduction_sales_increase_l2137_213731


namespace eight_dice_probability_l2137_213755

theorem eight_dice_probability : 
  let n : ℕ := 8  -- number of dice
  let k : ℕ := 4  -- number of dice showing even numbers
  let p : ℚ := 1/2  -- probability of a single die showing an even number
  Nat.choose n k * p^n = 35/128 := by
  sorry

end eight_dice_probability_l2137_213755


namespace complex_quadrant_problem_l2137_213799

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_quadrant_problem (a : ℝ) :
  is_purely_imaginary (Complex.mk (a^2 - 3*a - 4) (a - 4)) →
  a = -1 ∧ a < 0 ∧ -a > 0 :=
by sorry

end complex_quadrant_problem_l2137_213799


namespace height_derivative_at_one_l2137_213740

-- Define the height function
def h (t : ℝ) : ℝ := -4.9 * t^2 + 10 * t

-- State the theorem
theorem height_derivative_at_one :
  (deriv h) 1 = 0.2 := by sorry

end height_derivative_at_one_l2137_213740


namespace sphere_volume_for_maximized_tetrahedron_l2137_213728

theorem sphere_volume_for_maximized_tetrahedron (r : ℝ) (h : r = (3 * Real.sqrt 3) / 2) :
  (4 / 3) * Real.pi * r^3 = (27 * Real.sqrt 3 * Real.pi) / 2 :=
by sorry

end sphere_volume_for_maximized_tetrahedron_l2137_213728


namespace correct_factorization_l2137_213785

theorem correct_factorization (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := by
  sorry

#check correct_factorization

end correct_factorization_l2137_213785


namespace chess_tournament_games_l2137_213723

/-- The number of games in a chess tournament -/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  n * (n - 1) * games_per_pair / 2

/-- Proof that a chess tournament with 25 players, where each player plays 
    four times against every opponent, results in 1200 games total -/
theorem chess_tournament_games :
  num_games 25 4 = 1200 := by
  sorry

end chess_tournament_games_l2137_213723


namespace negation_of_exists_lt_one_squared_leq_one_l2137_213774

theorem negation_of_exists_lt_one_squared_leq_one :
  (¬ ∃ x : ℝ, x < 1 ∧ x^2 ≤ 1) ↔ (∀ x : ℝ, x < 1 → x^2 > 1) := by sorry

end negation_of_exists_lt_one_squared_leq_one_l2137_213774


namespace set_union_problem_l2137_213786

theorem set_union_problem (a b : ℝ) : 
  let A : Set ℝ := {-1, a}
  let B : Set ℝ := {3^a, b}
  A ∪ B = {-1, 0, 1} → a = 0 := by
sorry

end set_union_problem_l2137_213786


namespace delta_eight_four_l2137_213735

/-- The Δ operation for non-zero integers -/
def delta (a b : ℤ) : ℚ :=
  a - a / b

/-- Theorem stating that 8 Δ 4 = 6 -/
theorem delta_eight_four : delta 8 4 = 6 := by
  sorry

end delta_eight_four_l2137_213735


namespace smallest_resolvable_debt_l2137_213736

/-- The value of a cow in dollars -/
def cow_value : ℕ := 500

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 350

/-- The smallest positive debt that can be resolved using cows and sheep -/
def smallest_debt : ℕ := 50

/-- Theorem stating that the smallest_debt is the smallest positive value that can be expressed as a linear combination of cow_value and sheep_value with integer coefficients -/
theorem smallest_resolvable_debt : 
  smallest_debt = Nat.gcd cow_value sheep_value ∧
  ∃ (c s : ℤ), smallest_debt = c * cow_value + s * sheep_value ∧
  ∀ (d : ℕ), d > 0 → (∃ (x y : ℤ), d = x * cow_value + y * sheep_value) → d ≥ smallest_debt :=
sorry

end smallest_resolvable_debt_l2137_213736


namespace spring_deformation_l2137_213718

/-- A uniform spring with two attached weights -/
structure Spring :=
  (k : ℝ)  -- Spring constant
  (m₁ : ℝ) -- Mass of the top weight
  (m₂ : ℝ) -- Mass of the bottom weight

/-- The gravitational acceleration constant -/
def g : ℝ := 9.81

/-- Deformation when the spring is held vertically at its midpoint -/
def vertical_deformation (s : Spring) (x₁ x₂ : ℝ) : Prop :=
  2 * s.k * x₁ = s.m₁ * g ∧ x₁ = 0.08 ∧ x₂ = 0.15

/-- Deformation when the spring is laid horizontally -/
def horizontal_deformation (s : Spring) (x : ℝ) : Prop :=
  s.k * x = s.m₁ * g

/-- Theorem stating the relationship between vertical and horizontal deformations -/
theorem spring_deformation (s : Spring) (x₁ x₂ x : ℝ) :
  vertical_deformation s x₁ x₂ → horizontal_deformation s x → x = 0.16 := by
  sorry

end spring_deformation_l2137_213718


namespace geometric_sequence_sum_l2137_213796

/-- Geometric sequence with sum of first n terms S_n -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ S n = a 1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a S →
  S 4 = 1 →
  S 8 = 3 →
  a 17 + a 18 + a 19 + a 20 = 16 := by
sorry

end geometric_sequence_sum_l2137_213796


namespace mollys_age_condition_mollys_age_proof_l2137_213745

/-- Molly's present age -/
def mollys_present_age : ℕ := 12

/-- Condition: Molly's age in 18 years will be 5 times her age 6 years ago -/
theorem mollys_age_condition : 
  mollys_present_age + 18 = 5 * (mollys_present_age - 6) :=
by sorry

/-- Proof that Molly's present age is 12 years old -/
theorem mollys_age_proof : 
  mollys_present_age = 12 :=
by sorry

end mollys_age_condition_mollys_age_proof_l2137_213745


namespace simplify_square_root_l2137_213743

theorem simplify_square_root (x : ℝ) : 
  Real.sqrt (9 * x^6 + 3 * x^4) = Real.sqrt 3 * x^2 * Real.sqrt (3 * x^2 + 1) := by
  sorry

end simplify_square_root_l2137_213743


namespace no_coprime_natural_solution_l2137_213758

theorem no_coprime_natural_solution :
  ¬ ∃ (x y : ℕ), 
    (x ≠ 0) ∧ (y ≠ 0) ∧ 
    (Nat.gcd x y = 1) ∧ 
    (y^2 + y = x^3 - x) := by
  sorry

end no_coprime_natural_solution_l2137_213758


namespace sum_of_cubes_l2137_213783

theorem sum_of_cubes (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x + y + x^2*y + x*y^2 = 24) : 
  x^3 + y^3 = 68 := by
sorry

end sum_of_cubes_l2137_213783


namespace tessellation_theorem_l2137_213780

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  interiorAngle : ℝ

/-- Checks if two regular polygons can tessellate -/
def canTessellate (p1 p2 : RegularPolygon) : Prop :=
  ∃ (n m : ℕ), n * p1.interiorAngle + m * p2.interiorAngle = 360

theorem tessellation_theorem :
  let triangle : RegularPolygon := ⟨3, 60⟩
  let square : RegularPolygon := ⟨4, 90⟩
  let hexagon : RegularPolygon := ⟨6, 120⟩
  let octagon : RegularPolygon := ⟨8, 135⟩
  
  (canTessellate triangle square ∧
   canTessellate triangle hexagon ∧
   canTessellate octagon square) ∧
  ¬(canTessellate hexagon square) :=
by sorry

end tessellation_theorem_l2137_213780


namespace parallel_lines_condition_l2137_213756

/-- Given two lines l₁ and l₂ in the plane, prove that a=2 is a necessary and sufficient condition for l₁ to be parallel to l₂. -/
theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, 2*x - a*y + 1 = 0 ↔ (a-1)*x - y + a = 0) ↔ a = 2 := by
  sorry

end parallel_lines_condition_l2137_213756


namespace sam_winning_probability_l2137_213759

theorem sam_winning_probability :
  let hit_prob : ℚ := 2/5
  let miss_prob : ℚ := 3/5
  let p : ℚ := hit_prob + miss_prob * miss_prob * p
  p = 5/8 := by sorry

end sam_winning_probability_l2137_213759


namespace jacket_markup_percentage_l2137_213708

theorem jacket_markup_percentage 
  (purchase_price : ℝ)
  (markup_percentage : ℝ)
  (discount_percentage : ℝ)
  (gross_profit : ℝ)
  (h1 : purchase_price = 60)
  (h2 : discount_percentage = 0.20)
  (h3 : gross_profit = 4)
  (h4 : 0 ≤ markup_percentage ∧ markup_percentage < 1)
  (h5 : let selling_price := purchase_price / (1 - markup_percentage);
        gross_profit = selling_price * (1 - discount_percentage) - purchase_price) :
  markup_percentage = 0.25 := by
sorry

end jacket_markup_percentage_l2137_213708


namespace total_salaries_is_4000_l2137_213754

/-- The total amount of A and B's salaries is $4000 -/
theorem total_salaries_is_4000 
  (a_salary : ℝ) 
  (b_salary : ℝ) 
  (h1 : a_salary = 3000)
  (h2 : 0.05 * a_salary = 0.15 * b_salary) : 
  a_salary + b_salary = 4000 := by
  sorry

#check total_salaries_is_4000

end total_salaries_is_4000_l2137_213754


namespace ratio_problem_l2137_213766

theorem ratio_problem (x y a : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x / y = 5 / a) 
  (h4 : (x + 12) / (y + 12) = 3 / 4) : y - x = 9 := by
  sorry

end ratio_problem_l2137_213766


namespace circular_table_seating_l2137_213701

-- Define the number of people and seats
def total_people : ℕ := 9
def table_seats : ℕ := 7

-- Define the function to calculate the number of seating arrangements
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose n (n - k)) * (Nat.factorial (k - 1))

-- Theorem statement
theorem circular_table_seating :
  seating_arrangements total_people table_seats = 25920 :=
sorry

end circular_table_seating_l2137_213701


namespace original_rectangle_area_l2137_213753

theorem original_rectangle_area (original_area new_area : ℝ) : 
  (∀ (length width : ℝ), 
    length > 0 → width > 0 → 
    original_area = length * width → 
    new_area = (2 * length) * (2 * width)) →
  new_area = 32 →
  original_area = 8 :=
by
  sorry

end original_rectangle_area_l2137_213753


namespace equation_solutions_l2137_213732

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 2) = 2 * (x + 2) ↔ x = -2 ∨ x = 2) ∧
  (∀ x : ℝ, 3 * x^2 - x - 1 = 0 ↔ x = (1 + Real.sqrt 13) / 6 ∨ x = (1 - Real.sqrt 13) / 6) :=
by sorry

end equation_solutions_l2137_213732


namespace min_sum_of_products_l2137_213757

/-- A permutation of the numbers 1 to 12 -/
def Permutation12 := Fin 12 → Fin 12

/-- The sum of products for a given permutation -/
def sumOfProducts (p : Permutation12) : ℕ :=
  (p 0 + 1) * (p 1 + 1) * (p 2 + 1) +
  (p 3 + 1) * (p 4 + 1) * (p 5 + 1) +
  (p 6 + 1) * (p 7 + 1) * (p 8 + 1) +
  (p 9 + 1) * (p 10 + 1) * (p 11 + 1)

theorem min_sum_of_products :
  (∀ p : Permutation12, Function.Bijective p → sumOfProducts p ≥ 646) ∧
  (∃ p : Permutation12, Function.Bijective p ∧ sumOfProducts p = 646) :=
sorry

end min_sum_of_products_l2137_213757


namespace power_of_three_mod_five_l2137_213741

theorem power_of_three_mod_five : 3^2000 % 5 = 1 := by
  sorry

end power_of_three_mod_five_l2137_213741


namespace quadrangular_prism_has_12_edges_l2137_213793

/-- Number of edges in a prism with n sides -/
def prism_edges (n : ℕ) : ℕ := 3 * n

/-- Number of edges in a pyramid with n sides -/
def pyramid_edges (n : ℕ) : ℕ := 2 * n

theorem quadrangular_prism_has_12_edges :
  prism_edges 4 = 12 ∧
  pyramid_edges 4 ≠ 12 ∧
  pyramid_edges 5 ≠ 12 ∧
  prism_edges 5 ≠ 12 :=
by sorry

end quadrangular_prism_has_12_edges_l2137_213793


namespace eric_chicken_farm_eggs_l2137_213748

/-- Calculates the number of eggs collected given the number of chickens, eggs per chicken per day, and number of days. -/
def eggs_collected (num_chickens : ℕ) (eggs_per_chicken_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_chickens * eggs_per_chicken_per_day * num_days

/-- Proves that 4 chickens laying 3 eggs per day will produce 36 eggs in 3 days. -/
theorem eric_chicken_farm_eggs : eggs_collected 4 3 3 = 36 := by
  sorry

end eric_chicken_farm_eggs_l2137_213748


namespace problem_solution_l2137_213712

theorem problem_solution (x : ℝ) (h1 : x ≠ 0) : Real.sqrt ((5 * x) / 7) = x → x = 5 / 7 := by
  sorry

end problem_solution_l2137_213712


namespace find_number_l2137_213784

theorem find_number : ∃ x : ℝ, x - 2.95 - 2.95 = 9.28 ∧ x = 15.18 := by
  sorry

end find_number_l2137_213784
