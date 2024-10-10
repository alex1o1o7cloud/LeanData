import Mathlib

namespace hemisphere_surface_area_l26_2634

theorem hemisphere_surface_area (base_area : Real) (h : base_area = 225 * Real.pi) :
  let radius : Real := (base_area / Real.pi).sqrt
  let curved_surface_area : Real := 2 * Real.pi * radius^2
  let total_surface_area : Real := curved_surface_area + base_area
  total_surface_area = 675 * Real.pi := by
sorry

end hemisphere_surface_area_l26_2634


namespace prime_equation_solution_l26_2655

/-- Given that x and y are prime numbers, prove that x^y - y^x = xy^2 - 19 if and only if (x, y) = (2, 3) or (x, y) = (2, 7) -/
theorem prime_equation_solution (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) :
  x^y - y^x = x*y^2 - 19 ↔ (x = 2 ∧ y = 3) ∨ (x = 2 ∧ y = 7) := by
  sorry

end prime_equation_solution_l26_2655


namespace no_equal_digit_sum_decomposition_l26_2662

def digit_sum (n : ℕ) : ℕ := sorry

theorem no_equal_digit_sum_decomposition :
  ¬ ∃ (B C : ℕ), B + C = 999999999 ∧ digit_sum B = digit_sum C := by sorry

end no_equal_digit_sum_decomposition_l26_2662


namespace john_final_amount_l26_2649

def calculate_final_amount (initial_amount : ℚ) (game_cost : ℚ) (candy_cost : ℚ) 
  (soda_cost : ℚ) (magazine_cost : ℚ) (coupon_value : ℚ) (discount_rate : ℚ) 
  (allowance : ℚ) : ℚ :=
  let discounted_soda_cost := soda_cost * (1 - discount_rate)
  let magazine_paid := magazine_cost - coupon_value
  let total_expenses := game_cost + candy_cost + discounted_soda_cost + magazine_paid
  let remaining_after_expenses := initial_amount - total_expenses
  remaining_after_expenses + allowance

theorem john_final_amount :
  calculate_final_amount 5 2 1 1.5 3 0.5 0.1 26 = 24.15 := by
  sorry

end john_final_amount_l26_2649


namespace john_bench_press_sets_l26_2628

/-- The number of sets John does in his workout -/
def number_of_sets (weight_per_rep : ℕ) (reps_per_set : ℕ) (total_weight : ℕ) : ℕ :=
  total_weight / (weight_per_rep * reps_per_set)

/-- Theorem: John does 3 sets of bench presses -/
theorem john_bench_press_sets :
  number_of_sets 15 10 450 = 3 := by
  sorry

end john_bench_press_sets_l26_2628


namespace tangent_line_y_intercept_l26_2665

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane, represented by its slope and y-intercept --/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Check if a line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop :=
  let (x, y) := c.center
  (l.slope * x - y + l.yIntercept)^2 = c.radius^2 * (l.slope^2 + 1)

theorem tangent_line_y_intercept :
  ∃ (l : Line),
    isTangent l { center := (2, 0), radius := 2 } ∧
    isTangent l { center := (5, 0), radius := 1 } ∧
    l.yIntercept = 2 * Real.sqrt 2 := by
  sorry

end tangent_line_y_intercept_l26_2665


namespace laundry_cleaning_rate_l26_2623

/-- Given a total number of laundry pieces and available hours, 
    calculate the number of pieces to be cleaned per hour -/
def pieces_per_hour (total_pieces : ℕ) (available_hours : ℕ) : ℕ :=
  total_pieces / available_hours

/-- Theorem stating that cleaning 80 pieces of laundry in 4 hours 
    requires cleaning 20 pieces per hour -/
theorem laundry_cleaning_rate : pieces_per_hour 80 4 = 20 := by
  sorry

end laundry_cleaning_rate_l26_2623


namespace positive_number_relationship_l26_2679

theorem positive_number_relationship (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (h_a_eq : a^n = a + 1) 
  (h_b_eq : b^(2*n) = b + 3*a) : 
  a > b ∧ b > 1 := by
sorry

end positive_number_relationship_l26_2679


namespace function_bound_l26_2658

def function_property (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) - f x = 2 * x + 1

def bounded_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f x| ≤ 1

theorem function_bound (f : ℝ → ℝ) 
  (h1 : function_property f) 
  (h2 : bounded_on_unit_interval f) : 
  ∀ x : ℝ, |f x| ≤ 2 + x^2 := by
  sorry

end function_bound_l26_2658


namespace equivalent_representations_l26_2635

theorem equivalent_representations : ∀ (a b c d e : ℚ),
  (a = 15 ∧ b = 20 ∧ c = 6 ∧ d = 8 ∧ e = 75) →
  (a / b = c / d) ∧
  (a / b = 3 / 4) ∧
  (a / b = 0.75) ∧
  (a / b = e / 100) :=
by sorry

end equivalent_representations_l26_2635


namespace number_difference_l26_2666

theorem number_difference (x y : ℤ) : 
  x + y = 50 → 
  y = 19 → 
  x < 2 * y → 
  2 * y - x = 7 := by
sorry

end number_difference_l26_2666


namespace only_one_claim_impossible_l26_2680

-- Define the possible ring scores
def RingScores : List Nat := [1, 3, 5, 7, 9]

-- Define a structure for each person's claim
structure Claim where
  shots : Nat
  hits : Nat
  total_score : Nat

-- Define the claims
def claim_A : Claim := { shots := 5, hits := 5, total_score := 35 }
def claim_B : Claim := { shots := 6, hits := 6, total_score := 36 }
def claim_C : Claim := { shots := 3, hits := 3, total_score := 24 }
def claim_D : Claim := { shots := 4, hits := 3, total_score := 21 }

-- Function to check if a claim is possible
def is_claim_possible (c : Claim) : Prop :=
  ∃ (scores : List Nat),
    scores.length = c.hits ∧
    scores.all (· ∈ RingScores) ∧
    scores.sum = c.total_score

-- Theorem stating that only one claim is impossible
theorem only_one_claim_impossible :
  is_claim_possible claim_A ∧
  is_claim_possible claim_B ∧
  ¬is_claim_possible claim_C ∧
  is_claim_possible claim_D :=
sorry

end only_one_claim_impossible_l26_2680


namespace angle_A_measure_l26_2615

-- Define the measure of angles A and B
def measure_A : ℝ := sorry
def measure_B : ℝ := sorry

-- Define the conditions
axiom supplementary : measure_A + measure_B = 180
axiom relation : measure_A = 3 * measure_B

-- Theorem to prove
theorem angle_A_measure : measure_A = 135 := by
  sorry

end angle_A_measure_l26_2615


namespace csc_negative_330_degrees_l26_2644

-- Define the cosecant function
noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

-- State the theorem
theorem csc_negative_330_degrees : csc ((-330 : Real) * Real.pi / 180) = 2 := by
  sorry

end csc_negative_330_degrees_l26_2644


namespace complement_of_A_in_U_l26_2622

def U : Set ℕ := {x | x < 8}

def A : Set ℕ := {x | (x - 1) * (x - 3) * (x - 4) * (x - 7) = 0}

theorem complement_of_A_in_U : U \ A = {0, 2, 5, 6} := by
  sorry

end complement_of_A_in_U_l26_2622


namespace map_distance_theorem_l26_2670

/-- Represents the scale of a map as a ratio -/
def MapScale : ℚ := 1 / 250000

/-- Converts kilometers to centimeters -/
def kmToCm (km : ℚ) : ℚ := km * 100000

/-- Calculates the distance on a map given the actual distance and map scale -/
def mapDistance (actualDistance : ℚ) (scale : ℚ) : ℚ :=
  actualDistance * scale

theorem map_distance_theorem (actualDistanceKm : ℚ) 
  (h : actualDistanceKm = 5) : 
  mapDistance (kmToCm actualDistanceKm) MapScale = 2 := by
  sorry

#check map_distance_theorem

end map_distance_theorem_l26_2670


namespace increasing_function_condition_l26_2697

/-- The function f(x) = lg(x^2 - mx - m) is increasing on (1, +∞) iff m ≤ 1/2 -/
theorem increasing_function_condition (m : ℝ) :
  (∀ x > 1, StrictMono (fun x => Real.log (x^2 - m*x - m))) ↔ m ≤ 1/2 := by
  sorry

end increasing_function_condition_l26_2697


namespace farmer_cow_division_l26_2639

theorem farmer_cow_division (herd : ℕ) : 
  (herd / 3 : ℕ) + (herd / 6 : ℕ) + (herd / 8 : ℕ) + 9 = herd → herd = 24 := by
  sorry

end farmer_cow_division_l26_2639


namespace aria_cookie_spending_l26_2630

/-- The number of days in March -/
def days_in_march : ℕ := 31

/-- The number of cookies Aria purchased each day -/
def cookies_per_day : ℕ := 4

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 19

/-- The total amount Aria spent on cookies in March -/
def total_spent : ℕ := days_in_march * cookies_per_day * cost_per_cookie

theorem aria_cookie_spending :
  total_spent = 2356 := by sorry

end aria_cookie_spending_l26_2630


namespace completing_square_equivalence_l26_2613

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 - 4*x + 2 = 0) ↔ ((x - 2)^2 = 2) := by
  sorry

end completing_square_equivalence_l26_2613


namespace absolute_value_expression_l26_2689

theorem absolute_value_expression : |-2| * (|-25| - |5|) = 40 := by sorry

end absolute_value_expression_l26_2689


namespace floor_frac_equation_solutions_l26_2616

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define the fractional part function
noncomputable def frac (x : ℝ) : ℝ :=
  x - (floor x : ℝ)

-- State the theorem
theorem floor_frac_equation_solutions :
  ∀ x : ℝ, (floor x : ℝ) * frac x = 2019 * x ↔ x = 0 ∨ x = -1/2020 := by
  sorry

end floor_frac_equation_solutions_l26_2616


namespace susan_walk_distance_l26_2638

theorem susan_walk_distance (total_distance : ℝ) (erin_susan_diff : ℝ) (daniel_susan_ratio : ℝ) :
  total_distance = 32 ∧
  erin_susan_diff = 3 ∧
  daniel_susan_ratio = 2 →
  ∃ susan_distance : ℝ,
    susan_distance + (susan_distance - erin_susan_diff) + (daniel_susan_ratio * susan_distance) = total_distance ∧
    susan_distance = 8.75 := by
  sorry

end susan_walk_distance_l26_2638


namespace triangle_existence_l26_2681

theorem triangle_existence (y : ℕ+) : 
  (y + 1 + 6 > y^2 + 2*y + 3) ∧ 
  (y + 1 + (y^2 + 2*y + 3) > 6) ∧ 
  (6 + (y^2 + 2*y + 3) > y + 1) ↔ 
  y = 2 := by sorry

end triangle_existence_l26_2681


namespace game_ends_with_two_l26_2641

/-- Represents the state of the game board -/
structure GameBoard where
  ones : ℕ
  twos : ℕ

/-- Represents a move in the game -/
inductive Move
  | EraseOnes
  | EraseTwos
  | EraseOneTwo

/-- Applies a move to the game board -/
def applyMove (board : GameBoard) (move : Move) : GameBoard :=
  match move with
  | Move.EraseOnes => { ones := board.ones - 2, twos := board.twos + 1 }
  | Move.EraseTwos => { ones := board.ones, twos := board.twos - 1 }
  | Move.EraseOneTwo => { ones := board.ones - 1, twos := board.twos }

/-- The initial state of the game board -/
def initialBoard : GameBoard := { ones := 10, twos := 10 }

/-- Predicate to check if the game is over -/
def gameOver (board : GameBoard) : Prop :=
  board.ones + board.twos = 1

/-- Theorem stating that the game always ends with a two -/
theorem game_ends_with_two :
  ∀ (sequence : List Move),
    let finalBoard := sequence.foldl applyMove initialBoard
    gameOver finalBoard → finalBoard.twos = 1 :=
  sorry

end game_ends_with_two_l26_2641


namespace wrong_calculation_correction_l26_2625

theorem wrong_calculation_correction (x : ℝ) : 
  x / 5 + 16 = 58 → x / 15 + 74 = 88 := by
  sorry

end wrong_calculation_correction_l26_2625


namespace courtney_marbles_count_l26_2645

/-- The number of marbles in Courtney's first jar -/
def first_jar : ℕ := 80

/-- The number of marbles in Courtney's second jar -/
def second_jar : ℕ := 2 * first_jar

/-- The number of marbles in Courtney's third jar -/
def third_jar : ℕ := first_jar / 4

/-- The total number of marbles Courtney has -/
def total_marbles : ℕ := first_jar + second_jar + third_jar

theorem courtney_marbles_count : total_marbles = 260 := by
  sorry

end courtney_marbles_count_l26_2645


namespace problem_solution_l26_2688

theorem problem_solution (m n : ℕ) (x : ℝ) 
  (h1 : 2^m = 8)
  (h2 : 2^n = 32)
  (h3 : x = 2^m - 1) :
  (2^(2*m + n - 4) = 128) ∧ 
  (1 + 4^(m+1) = 4*x^2 + 8*x + 5) := by
  sorry

end problem_solution_l26_2688


namespace repeating_decimal_to_fraction_l26_2640

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 3 + 56 / 99) ∧ (x = 353 / 99) :=
by sorry

end repeating_decimal_to_fraction_l26_2640


namespace bike_shop_profit_l26_2646

/-- The profit calculation for Jim's bike shop -/
theorem bike_shop_profit (x : ℝ) 
  (h1 : x > 0) -- Charge for fixing bike tires is positive
  (h2 : 300 * x + 600 + 2000 - (300 * 5 + 100 + 4000) = 3000) -- Profit equation
  : x = 20 := by
  sorry

end bike_shop_profit_l26_2646


namespace tetrahedron_bisector_ratio_l26_2624

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents the area of a triangle -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Represents a point on an edge of the tetrahedron -/
def intersectionPoint (t : Tetrahedron) : Point3D := sorry

/-- Theorem: In a tetrahedron ABCD, where the bisector plane of the dihedral angle around edge CD
    intersects AB at point E, the ratio of AE to BE is equal to the ratio of the areas of
    triangles ACD and BCD -/
theorem tetrahedron_bisector_ratio (t : Tetrahedron) :
  let E := intersectionPoint t
  let AE := Real.sqrt ((t.A.x - E.x)^2 + (t.A.y - E.y)^2 + (t.A.z - E.z)^2)
  let BE := Real.sqrt ((t.B.x - E.x)^2 + (t.B.y - E.y)^2 + (t.B.z - E.z)^2)
  let t_ACD := triangleArea t.A t.C t.D
  let t_BCD := triangleArea t.B t.C t.D
  AE / BE = t_ACD / t_BCD := by sorry

end tetrahedron_bisector_ratio_l26_2624


namespace geometric_sequence_common_ratio_l26_2653

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_a1 : a 1 = -1) 
  (h_sum : a 2 + a 3 = -2) :
  ∃ q : ℝ, (q = -2 ∨ q = 1) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end geometric_sequence_common_ratio_l26_2653


namespace chipmunks_went_away_count_l26_2674

/-- Represents the chipmunk population in a forest --/
structure ChipmunkForest where
  originalFamilies : ℕ
  remainingFamilies : ℕ
  avgMembersRemaining : ℕ
  avgMembersLeft : ℕ

/-- Calculates the number of chipmunks that went away --/
def chipmunksWentAway (forest : ChipmunkForest) : ℕ :=
  (forest.originalFamilies - forest.remainingFamilies) * forest.avgMembersLeft

/-- Theorem stating the number of chipmunks that went away --/
theorem chipmunks_went_away_count (forest : ChipmunkForest) 
  (h1 : forest.originalFamilies = 86)
  (h2 : forest.remainingFamilies = 21)
  (h3 : forest.avgMembersRemaining = 15)
  (h4 : forest.avgMembersLeft = 18) :
  chipmunksWentAway forest = 1170 := by
  sorry

#eval chipmunksWentAway { originalFamilies := 86, remainingFamilies := 21, avgMembersRemaining := 15, avgMembersLeft := 18 }

end chipmunks_went_away_count_l26_2674


namespace no_prime_sum_53_l26_2661

theorem no_prime_sum_53 : ¬ ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p + q = 53 := by sorry

end no_prime_sum_53_l26_2661


namespace only_four_and_six_have_three_solutions_l26_2636

def X : Finset ℕ := {1, 2, 5, 7, 11, 13, 16, 17}

def hasThreedifferentsolutions (k : ℕ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℕ), 
    x₁ ∈ X ∧ y₁ ∈ X ∧ x₂ ∈ X ∧ y₂ ∈ X ∧ x₃ ∈ X ∧ y₃ ∈ X ∧
    x₁ - y₁ = k ∧ x₂ - y₂ = k ∧ x₃ - y₃ = k ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃)

theorem only_four_and_six_have_three_solutions :
  ∀ k : ℕ, k > 0 → (hasThreedifferentsolutions k ↔ k = 4 ∨ k = 6) := by sorry

end only_four_and_six_have_three_solutions_l26_2636


namespace circle_position_l26_2609

def circle_center : ℝ × ℝ := (1, 2)
def circle_radius : ℝ := 1

def distance_to_y_axis (center : ℝ × ℝ) : ℝ := |center.1|
def distance_to_x_axis (center : ℝ × ℝ) : ℝ := |center.2|

def is_tangent_to_y_axis (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  distance_to_y_axis center = radius

def is_disjoint_from_x_axis (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  distance_to_x_axis center > radius

theorem circle_position :
  is_tangent_to_y_axis circle_center circle_radius ∧
  is_disjoint_from_x_axis circle_center circle_radius :=
by sorry

end circle_position_l26_2609


namespace booknote_unique_letters_l26_2668

def word : String := "booknote"

def letter_set : Finset Char := word.toList.toFinset

theorem booknote_unique_letters : Finset.card letter_set = 6 := by
  sorry

end booknote_unique_letters_l26_2668


namespace dolphins_score_l26_2686

theorem dolphins_score (total_score winning_margin : ℕ) : 
  total_score = 48 → winning_margin = 20 → 
  ∃ (sharks_score dolphins_score : ℕ), 
    sharks_score + dolphins_score = total_score ∧ 
    sharks_score = dolphins_score + winning_margin ∧
    dolphins_score = 14 := by
  sorry

end dolphins_score_l26_2686


namespace triangle_perimeter_is_19_l26_2683

/-- Triangle PQR with given properties -/
structure Triangle where
  /-- Length of side PQ -/
  PQ : ℝ
  /-- Length of side QR -/
  QR : ℝ
  /-- Length of side PR -/
  PR : ℝ
  /-- Angle PQR equals angle PRQ -/
  angle_equality : PQ = PR

/-- The perimeter of a triangle is the sum of its side lengths -/
def perimeter (t : Triangle) : ℝ := t.PQ + t.QR + t.PR

/-- Theorem: The perimeter of the given triangle is 19 -/
theorem triangle_perimeter_is_19 (t : Triangle) 
  (h1 : t.QR = 5) 
  (h2 : t.PR = 7) : 
  perimeter t = 19 := by
  sorry

end triangle_perimeter_is_19_l26_2683


namespace friends_contribution_l26_2602

/-- Represents the expenses of a group of friends -/
structure Expenses where
  num_friends : Nat
  total_amount : Rat

/-- Calculates the amount each friend should contribute -/
def calculate_contribution (e : Expenses) : Rat :=
  e.total_amount / e.num_friends

/-- Theorem: For 5 friends with total expenses of $61, each should contribute $12.20 -/
theorem friends_contribution :
  let e : Expenses := { num_friends := 5, total_amount := 61 }
  calculate_contribution e = 61 / 5 := by sorry

end friends_contribution_l26_2602


namespace sector_max_area_l26_2650

/-- Given a sector with circumference 20cm, its area is maximized when the radius is 5cm -/
theorem sector_max_area (R : ℝ) : 
  let circumference := 20
  let arc_length := circumference - 2 * R
  let area := (1 / 2) * arc_length * R
  (∀ r : ℝ, area ≤ ((1 / 2) * (circumference - 2 * r) * r)) → R = 5 := by
sorry

end sector_max_area_l26_2650


namespace min_value_a_l26_2685

theorem min_value_a (a m n : ℕ) (h1 : a ≠ 0) (h2 : (2 : ℚ) / 10 * a = m ^ 2) (h3 : (5 : ℚ) / 10 * a = n ^ 3) :
  ∀ b : ℕ, b ≠ 0 ∧ (∃ p q : ℕ, (2 : ℚ) / 10 * b = p ^ 2 ∧ (5 : ℚ) / 10 * b = q ^ 3) → a ≤ b → a = 2000 :=
sorry

end min_value_a_l26_2685


namespace f_properties_l26_2643

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Theorem statement
theorem f_properties :
  -- 1. f(x) is increasing on (-∞, -1) and (1, +∞)
  (∀ x y, (x < y ∧ ((x < -1 ∧ y < -1) ∨ (x > 1 ∧ y > 1))) → f x < f y) ∧
  -- 2. f(x) is decreasing on (-1, 1)
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  -- 3. The maximum value of f(x) on [-3, 2] is 2
  (∀ x, -3 ≤ x ∧ x ≤ 2 → f x ≤ 2) ∧
  (∃ x, -3 ≤ x ∧ x ≤ 2 ∧ f x = 2) ∧
  -- 4. The minimum value of f(x) on [-3, 2] is -18
  (∀ x, -3 ≤ x ∧ x ≤ 2 → f x ≥ -18) ∧
  (∃ x, -3 ≤ x ∧ x ≤ 2 ∧ f x = -18) :=
by sorry

end f_properties_l26_2643


namespace possible_ad_values_l26_2612

/-- Represents a point on a line -/
structure Point :=
  (x : ℝ)

/-- The distance between two points -/
def distance (p q : Point) : ℝ := |p.x - q.x|

/-- Theorem: Possible values of AD given AB = 1, BC = 2, CD = 4 -/
theorem possible_ad_values (A B C D : Point) 
  (h1 : distance A B = 1)
  (h2 : distance B C = 2)
  (h3 : distance C D = 4) :
  (distance A D = 1) ∨ (distance A D = 3) ∨ (distance A D = 5) ∨ (distance A D = 7) :=
sorry

end possible_ad_values_l26_2612


namespace regular_polygon_sides_l26_2632

theorem regular_polygon_sides (exterior_angle : ℝ) (h : exterior_angle = 45) :
  (360 / exterior_angle : ℝ) = 8 := by sorry

end regular_polygon_sides_l26_2632


namespace quadratic_factorization_l26_2608

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end quadratic_factorization_l26_2608


namespace transformed_sine_sum_l26_2654

theorem transformed_sine_sum (ω A a φ : ℝ) (hω : ω > 0) (hA : A > 0) (ha : a > 0) (hφ : 0 < φ ∧ φ < π)
  (h : ∀ x, A * Real.sin (ω * x - φ) + a = 3 * Real.sin (2 * x - π / 6) + 1) :
  A + a + ω + φ = 16 / 3 + 11 * π / 12 := by
  sorry

end transformed_sine_sum_l26_2654


namespace monthly_income_calculation_l26_2671

theorem monthly_income_calculation (income : ℝ) : 
  (income / 2 - 20 = 100) → income = 240 := by
  sorry

end monthly_income_calculation_l26_2671


namespace fence_cost_square_plot_l26_2692

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (cost_per_foot : ℝ) (total_cost : ℝ) :
  area = 289 →
  cost_per_foot = 59 →
  total_cost = 4 * Real.sqrt area * cost_per_foot →
  total_cost = 4012 := by
  sorry

#check fence_cost_square_plot

end fence_cost_square_plot_l26_2692


namespace distribute_seven_among_three_l26_2611

/-- The number of ways to distribute n indistinguishable items among k distinct groups,
    with each group receiving at least one item. -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 15 ways to distribute 7 recommended places among 3 schools,
    with each school receiving at least one place. -/
theorem distribute_seven_among_three :
  distribute 7 3 = 15 := by
  sorry

end distribute_seven_among_three_l26_2611


namespace quadratic_one_solution_l26_2693

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 16 * x^2 + n * x + 4 = 0) ↔ (n = 16 ∨ n = -16) := by
  sorry

end quadratic_one_solution_l26_2693


namespace division_remainder_proof_l26_2667

theorem division_remainder_proof (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 729 → divisor = 38 → quotient = 19 → 
  dividend = divisor * quotient + remainder → remainder = 7 := by
sorry

end division_remainder_proof_l26_2667


namespace function_satisfying_condition_is_zero_function_l26_2660

theorem function_satisfying_condition_is_zero_function 
  (f : ℝ → ℝ) (h : ∀ x y : ℝ, f x + f y = f (f x * f y)) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end function_satisfying_condition_is_zero_function_l26_2660


namespace problem_solution_l26_2606

theorem problem_solution (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 125 := by
  sorry

end problem_solution_l26_2606


namespace b_share_calculation_l26_2607

theorem b_share_calculation (total : ℝ) : 
  let a := (2 : ℝ) / 15 * total
  let b := (3 : ℝ) / 15 * total
  let c := (4 : ℝ) / 15 * total
  let d := (6 : ℝ) / 15 * total
  d - c = 700 → b = 1050 := by
  sorry

end b_share_calculation_l26_2607


namespace tangent_condition_l26_2601

-- Define the curve and line
def curve (x y : ℝ) : Prop := x^2 + 3*y^2 = 12
def line (m x y : ℝ) : Prop := m*x + y = 16

-- Define the tangency condition
def is_tangent (m : ℝ) : Prop := ∃! p : ℝ × ℝ, curve p.1 p.2 ∧ line m p.1 p.2

-- State the theorem
theorem tangent_condition (m : ℝ) : is_tangent m → m^2 = 21 := by
  sorry

end tangent_condition_l26_2601


namespace highest_points_is_38_l26_2694

/-- The TRISQUARE game awards points for triangles and squares --/
structure TRISQUARE where
  small_triangles : ℕ
  large_triangles : ℕ
  small_squares : ℕ
  large_squares : ℕ
  triangle_points : ℕ
  square_points : ℕ

/-- Calculate the total points for a TRISQUARE game --/
def total_points (game : TRISQUARE) : ℕ :=
  (game.small_triangles + game.large_triangles) * game.triangle_points +
  (game.small_squares + game.large_squares) * game.square_points

/-- Theorem: The highest number of points achievable in the given TRISQUARE game is 38 --/
theorem highest_points_is_38 (game : TRISQUARE) 
  (h1 : game.small_triangles = 4)
  (h2 : game.large_triangles = 2)
  (h3 : game.small_squares = 4)
  (h4 : game.large_squares = 1)
  (h5 : game.triangle_points = 3)
  (h6 : game.square_points = 4) :
  total_points game = 38 := by
  sorry

#check highest_points_is_38

end highest_points_is_38_l26_2694


namespace divisibility_implication_l26_2678

theorem divisibility_implication (a b m n : ℕ) 
  (h1 : a > 1) 
  (h2 : Nat.gcd a b = 1) 
  (h3 : (a^n + b^n) ∣ (a^m + b^m)) : 
  n ∣ m := by sorry

end divisibility_implication_l26_2678


namespace max_clowns_proof_l26_2677

/-- The number of distinct colors available -/
def num_colors : ℕ := 12

/-- The minimum number of colors each clown must use -/
def min_colors_per_clown : ℕ := 5

/-- The maximum number of clowns that can use any particular color -/
def max_clowns_per_color : ℕ := 20

/-- The set of all possible color combinations for clowns -/
def color_combinations : Finset (Finset (Fin num_colors)) :=
  (Finset.powerset (Finset.univ : Finset (Fin num_colors))).filter (fun s => s.card ≥ min_colors_per_clown)

/-- The maximum number of clowns satisfying all conditions -/
def max_clowns : ℕ := num_colors * max_clowns_per_color

theorem max_clowns_proof :
  (∀ s : Finset (Fin num_colors), s ∈ color_combinations → s.card ≥ min_colors_per_clown) ∧
  (∀ c : Fin num_colors, (color_combinations.filter (fun s => c ∈ s)).card ≤ max_clowns_per_color) →
  color_combinations.card ≥ max_clowns ∧
  max_clowns = 240 := by
  sorry

end max_clowns_proof_l26_2677


namespace functional_equation_solution_l26_2682

theorem functional_equation_solution 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) : 
  ∃! f : ℝ → ℝ, 
    (∀ x, x > 0 → f x > 0) ∧ 
    (∀ x, x > 0 → f (f x) + a * f x = b * (a + b) * x) ∧
    (∀ x, x > 0 → f x = b * x) := by
  sorry

end functional_equation_solution_l26_2682


namespace z_in_fourth_quadrant_l26_2659

/-- Given a complex number z satisfying z(1+i) = 2, prove that z has a positive real part and a negative imaginary part. -/
theorem z_in_fourth_quadrant (z : ℂ) (h : z * (1 + Complex.I) = 2) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end z_in_fourth_quadrant_l26_2659


namespace box_length_proof_l26_2652

/-- Proves that a rectangular box with given dimensions and fill rate has a specific length -/
theorem box_length_proof (fill_rate : ℝ) (width depth time : ℝ) (h1 : fill_rate = 4)
    (h2 : width = 6) (h3 : depth = 2) (h4 : time = 21) :
  (fill_rate * time) / (width * depth) = 7 := by
  sorry

end box_length_proof_l26_2652


namespace certain_number_proof_l26_2617

theorem certain_number_proof (x : ℝ) : (60 / 100 * 500 = 50 / 100 * x) → x = 600 := by
  sorry

end certain_number_proof_l26_2617


namespace inscribed_circle_rectangle_area_l26_2610

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 - 20 * x - 8 * y + 72 = 0

/-- The circle is inscribed in a rectangle -/
def is_inscribed (circle : (ℝ → ℝ → Prop)) (rectangle : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), circle x y → (x, y) ∈ rectangle

/-- One pair of sides of the rectangle is parallel to the y-axis -/
def sides_parallel_to_y_axis (rectangle : Set (ℝ × ℝ)) : Prop :=
  ∃ (x₁ x₂ : ℝ), ∀ (y : ℝ), (x₁, y) ∈ rectangle ∨ (x₂, y) ∈ rectangle

/-- The area of the rectangle -/
def rectangle_area (rectangle : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The main theorem -/
theorem inscribed_circle_rectangle_area :
  ∀ (rectangle : Set (ℝ × ℝ)),
  is_inscribed circle_equation rectangle →
  sides_parallel_to_y_axis rectangle →
  rectangle_area rectangle = 28 :=
sorry

end inscribed_circle_rectangle_area_l26_2610


namespace sandy_correct_sums_l26_2647

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 45)
  (h3 : correct_marks = 3)
  (h4 : incorrect_marks = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums * correct_marks - (total_sums - correct_sums) * incorrect_marks = total_marks ∧
    correct_sums = 21 :=
by sorry

end sandy_correct_sums_l26_2647


namespace quadratic_roots_relation_l26_2629

theorem quadratic_roots_relation (r s : ℝ) (p q : ℝ) : 
  (3 * r^2 + 4 * r + 2 = 0) →
  (3 * s^2 + 4 * s + 2 = 0) →
  ((1/r^2) + (1/s^2) = -p) →
  ((1/r^2) * (1/s^2) = q) →
  p = -1 := by
sorry

end quadratic_roots_relation_l26_2629


namespace books_grabbed_l26_2600

/-- Calculates the number of books Henry grabbed from the "free to a good home" box -/
theorem books_grabbed (initial_books : ℕ) (donated_boxes : ℕ) (books_per_box : ℕ) 
  (room_books : ℕ) (coffee_table_books : ℕ) (kitchen_books : ℕ) (final_books : ℕ) : 
  initial_books = 99 →
  donated_boxes = 3 →
  books_per_box = 15 →
  room_books = 21 →
  coffee_table_books = 4 →
  kitchen_books = 18 →
  final_books = 23 →
  final_books - (initial_books - (donated_boxes * books_per_box + room_books + coffee_table_books + kitchen_books)) = 12 := by
  sorry

end books_grabbed_l26_2600


namespace upper_bound_of_prime_set_l26_2669

theorem upper_bound_of_prime_set (A : Set ℕ) : 
  (∀ x ∈ A, Nat.Prime x) →   -- A contains only prime numbers
  (∃ a ∈ A, a > 62) →        -- Lower bound is greater than 62
  (∀ a ∈ A, a > 62) →        -- All elements are greater than 62
  (∃ max min : ℕ, max ∈ A ∧ min ∈ A ∧ max - min = 16 ∧
    ∀ a ∈ A, min ≤ a ∧ a ≤ max) →  -- Range of A is 16
  (∃ x ∈ A, ∀ y ∈ A, y ≤ x) →  -- A has a maximum element
  (∃ x ∈ A, x = 83 ∧ ∀ y ∈ A, y ≤ x) :=  -- The upper bound (maximum) is 83
by sorry

end upper_bound_of_prime_set_l26_2669


namespace power_of_ten_negative_y_l26_2621

theorem power_of_ten_negative_y (y : ℝ) (h : (10 : ℝ) ^ (2 * y) = 25) : (10 : ℝ) ^ (-y) = 1/5 := by
  sorry

end power_of_ten_negative_y_l26_2621


namespace rectangle_perimeter_l26_2633

/-- The sum of the lengths of all sides of a rectangle with sides 9 cm and 11 cm is 40 cm. -/
theorem rectangle_perimeter (length width : ℝ) (h1 : length = 9) (h2 : width = 11) :
  2 * (length + width) = 40 := by
  sorry

#check rectangle_perimeter

end rectangle_perimeter_l26_2633


namespace additional_plates_count_l26_2603

/-- Represents the number of choices for each position in a license plate. -/
structure LicensePlateChoices where
  first : Nat
  second : Nat
  third : Nat
  fourth : Nat

/-- Calculates the total number of possible license plates. -/
def totalPlates (choices : LicensePlateChoices) : Nat :=
  choices.first * choices.second * choices.third * choices.fourth

/-- The original choices for each position in TriCity license plates. -/
def originalChoices : LicensePlateChoices :=
  { first := 3, second := 4, third := 2, fourth := 5 }

/-- The new choices after adding two new letters. -/
def newChoices : LicensePlateChoices :=
  { first := originalChoices.first + 1,
    second := originalChoices.second,
    third := originalChoices.third + 1,
    fourth := originalChoices.fourth }

/-- Theorem stating the number of additional license plates after the change. -/
theorem additional_plates_count :
  totalPlates newChoices - totalPlates originalChoices = 120 := by
  sorry

end additional_plates_count_l26_2603


namespace jelly_bean_ratio_l26_2664

/-- Given 1200 total jelly beans divided between two jars X and Y, where jar X has 800 jelly beans,
    prove that the ratio of jelly beans in jar X to jar Y is 2:1. -/
theorem jelly_bean_ratio :
  let total_beans : ℕ := 1200
  let jar_x : ℕ := 800
  let jar_y : ℕ := total_beans - jar_x
  (jar_x : ℚ) / jar_y = 2 := by sorry

end jelly_bean_ratio_l26_2664


namespace negation_existential_proposition_l26_2699

theorem negation_existential_proposition :
  (¬ ∃ x : ℝ, x > 0 ∧ Real.log x > x - 2) ↔ (∀ x : ℝ, x > 0 → Real.log x ≤ x - 2) := by
  sorry

end negation_existential_proposition_l26_2699


namespace woodburning_cost_l26_2672

def woodburning_problem (num_sold : ℕ) (price_per_item : ℚ) (profit : ℚ) : Prop :=
  let total_revenue := num_sold * price_per_item
  let cost_of_wood := total_revenue - profit
  cost_of_wood = 100

theorem woodburning_cost :
  woodburning_problem 20 15 200 := by
  sorry

end woodburning_cost_l26_2672


namespace ways_to_pay_100_l26_2637

/-- Represents the available coin denominations -/
def CoinDenominations : List Nat := [1, 2, 10, 20, 50]

/-- Calculates the number of ways to pay a given amount using the available coin denominations -/
def waysToPayAmount (amount : Nat) : Nat :=
  sorry -- Implementation details omitted

/-- Theorem stating that there are 784 ways to pay 100 using the given coin denominations -/
theorem ways_to_pay_100 : waysToPayAmount 100 = 784 := by
  sorry

end ways_to_pay_100_l26_2637


namespace rotation_implies_equilateral_l26_2631

/-- A triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Rotation of a point around another point by a given angle -/
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Check if a triangle is equilateral -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Theorem: If rotating a triangle 60° around point A moves B to C, then the triangle is equilateral -/
theorem rotation_implies_equilateral (t : Triangle) :
  rotate t.A (π / 3) t.B = t.C → is_equilateral t := by sorry

end rotation_implies_equilateral_l26_2631


namespace exchange_rate_problem_l26_2604

theorem exchange_rate_problem (d : ℕ) : 
  (3 / 2 : ℚ) * d - 72 = d → d = 144 := by sorry

end exchange_rate_problem_l26_2604


namespace decimal_sum_to_fraction_l26_2673

theorem decimal_sum_to_fraction :
  (0.1 : ℚ) + 0.02 + 0.003 + 0.0004 + 0.00005 + 0.000006 + 0.0000007 = 1234567 / 10000000 := by
  sorry

end decimal_sum_to_fraction_l26_2673


namespace student_arrangements_l26_2642

def num_male_students : ℕ := 4
def num_female_students : ℕ := 3
def total_students : ℕ := num_male_students + num_female_students

def arrangements_female_together : ℕ := sorry

def arrangements_no_adjacent_females : ℕ := sorry

def arrangements_ordered_females : ℕ := sorry

theorem student_arrangements :
  (arrangements_female_together = 720) ∧
  (arrangements_no_adjacent_females = 1440) ∧
  (arrangements_ordered_females = 840) := by sorry

end student_arrangements_l26_2642


namespace ratio_x_y_is_four_to_one_l26_2690

theorem ratio_x_y_is_four_to_one 
  (x y : ℝ) 
  (h1 : x > 0) 
  (h2 : y > 0) 
  (h3 : 2 * Real.log (x - 2*y) = Real.log x + Real.log y) : 
  x / y = 4 := by
sorry

end ratio_x_y_is_four_to_one_l26_2690


namespace supermarket_spending_l26_2648

/-- Represents the total amount spent at the supermarket -/
def total_spent : ℝ := 120

/-- Represents the amount spent on candy -/
def candy_spent : ℝ := 8

/-- Theorem stating the total amount spent at the supermarket -/
theorem supermarket_spending :
  (1/2 + 1/3 + 1/10) * total_spent + candy_spent = total_spent :=
by sorry

end supermarket_spending_l26_2648


namespace perpendicular_line_through_M_l26_2614

-- Define the line l: 2x - y - 4 = 0
def line_l (x y : ℝ) : Prop := 2 * x - y - 4 = 0

-- Define point M as the intersection of line l with the x-axis
def point_M : ℝ × ℝ := (2, 0)

-- Define the perpendicular line: x + 2y - 2 = 0
def perp_line (x y : ℝ) : Prop := x + 2 * y - 2 = 0

-- Theorem statement
theorem perpendicular_line_through_M :
  (perp_line (point_M.1) (point_M.2)) ∧
  (∀ x y : ℝ, line_l x y → perp_line x y → 
    (y - point_M.2) * (x - point_M.1) = -(2 * (x - point_M.1) * (y - point_M.2))) :=
by sorry

end perpendicular_line_through_M_l26_2614


namespace tshirts_sold_count_l26_2620

/-- The revenue generated from selling t-shirts -/
def tshirt_revenue : ℕ := 4300

/-- The revenue generated from each t-shirt -/
def revenue_per_tshirt : ℕ := 215

/-- The number of t-shirts sold -/
def num_tshirts : ℕ := tshirt_revenue / revenue_per_tshirt

theorem tshirts_sold_count : num_tshirts = 20 := by
  sorry

end tshirts_sold_count_l26_2620


namespace closest_integer_to_cube_root_150_l26_2651

theorem closest_integer_to_cube_root_150 :
  ∀ n : ℤ, |n^3 - 150| ≥ |5^3 - 150| := by sorry

end closest_integer_to_cube_root_150_l26_2651


namespace twenty_is_forty_percent_of_fifty_l26_2691

theorem twenty_is_forty_percent_of_fifty :
  ∀ x : ℝ, (20 : ℝ) / x = (40 : ℝ) / 100 → x = 50 :=
by
  sorry

end twenty_is_forty_percent_of_fifty_l26_2691


namespace sandra_savings_proof_l26_2605

-- Define the given conditions
def mother_contribution : ℝ := 4
def father_contribution : ℝ := 2 * mother_contribution
def candy_cost : ℝ := 0.5
def jelly_bean_cost : ℝ := 0.2
def candy_quantity : ℕ := 14
def jelly_bean_quantity : ℕ := 20
def money_left : ℝ := 11

-- Define Sandra's initial savings
def sandra_initial_savings : ℝ := 10

-- Theorem to prove
theorem sandra_savings_proof :
  sandra_initial_savings = 
    (candy_cost * candy_quantity + jelly_bean_cost * jelly_bean_quantity + money_left) - 
    (mother_contribution + father_contribution) := by
  sorry


end sandra_savings_proof_l26_2605


namespace least_subtraction_for_divisibility_by_two_l26_2676

theorem least_subtraction_for_divisibility_by_two (n : ℕ) (h : n = 9671) :
  ∃ (k : ℕ), k = 1 ∧ 
  (∀ (m : ℕ), m < k → ¬(∃ (q : ℕ), n - m = 2 * q)) ∧
  (∃ (q : ℕ), n - k = 2 * q) :=
sorry

end least_subtraction_for_divisibility_by_two_l26_2676


namespace f_negative_a_eq_zero_l26_2656

noncomputable def f (x : ℝ) : ℝ := x^3 * (Real.exp x + Real.exp (-x)) + 2

theorem f_negative_a_eq_zero (a : ℝ) (h : f a = 4) : f (-a) = 0 := by
  sorry

end f_negative_a_eq_zero_l26_2656


namespace triangle_max_area_l26_2627

theorem triangle_max_area (A B C : ℝ) (h1 : 0 < A) (h2 : 0 < B) (h3 : 0 < C) 
  (h4 : A + B + C = π) (h5 : Real.tan A * Real.tan B = 3/4) : 
  let a : ℝ := 4
  let b : ℝ := a * Real.sin B / Real.sin A
  let c : ℝ := a * Real.sin C / Real.sin A
  ∀ (S : ℝ), S = 1/2 * a * b * Real.sin C → S ≤ 2 * Real.sqrt 3 := by
sorry

end triangle_max_area_l26_2627


namespace range_of_sum_l26_2619

theorem range_of_sum (a b : ℝ) :
  (∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) ≥ -1) →
  -1 ≤ a + b ∧ a + b ≤ 2 := by
  sorry

end range_of_sum_l26_2619


namespace sara_team_wins_l26_2618

/-- Represents a basketball team's game statistics -/
structure TeamStats where
  total_games : ℕ
  lost_games : ℕ

/-- Calculates the number of games won by a team -/
def games_won (stats : TeamStats) : ℕ :=
  stats.total_games - stats.lost_games

/-- Theorem: For Sara's team, the number of games won is 12 -/
theorem sara_team_wins (sara_team : TeamStats) 
  (h1 : sara_team.total_games = 16) 
  (h2 : sara_team.lost_games = 4) : 
  games_won sara_team = 12 := by
  sorry

end sara_team_wins_l26_2618


namespace power_equation_non_negative_l26_2675

theorem power_equation_non_negative (a b c d : ℤ) 
  (h : (2 : ℝ)^a + (2 : ℝ)^b = (5 : ℝ)^c + (5 : ℝ)^d) : 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d := by
  sorry

end power_equation_non_negative_l26_2675


namespace relay_race_time_l26_2698

/-- The relay race problem -/
theorem relay_race_time (athlete1 athlete2 athlete3 athlete4 total : ℕ) : 
  athlete1 = 55 →
  athlete2 = athlete1 + 10 →
  athlete3 = athlete2 - 15 →
  athlete4 = athlete1 - 25 →
  total = athlete1 + athlete2 + athlete3 + athlete4 →
  total = 200 := by
  sorry

end relay_race_time_l26_2698


namespace square_fraction_count_l26_2695

theorem square_fraction_count : 
  ∃! (S : Finset ℤ), 
    (∀ n ∈ S, 0 ≤ n ∧ n ≤ 23 ∧ ∃ k : ℤ, (n : ℚ) / (24 - n) = k^2) ∧ 
    Finset.card S = 2 :=
by sorry

end square_fraction_count_l26_2695


namespace painted_equals_unpainted_l26_2657

/-- Represents a cube with edge length n, painted on two adjacent faces and sliced into unit cubes -/
structure PaintedCube where
  n : ℕ
  n_gt_two : n > 2

/-- The number of smaller cubes with exactly two faces painted -/
def two_faces_painted (c : PaintedCube) : ℕ := c.n - 2

/-- The number of smaller cubes completely without paint -/
def unpainted (c : PaintedCube) : ℕ := (c.n - 2)^3

/-- Theorem stating that the number of cubes with two faces painted equals the number of unpainted cubes if and only if n = 3 -/
theorem painted_equals_unpainted (c : PaintedCube) : 
  two_faces_painted c = unpainted c ↔ c.n = 3 := by
  sorry

end painted_equals_unpainted_l26_2657


namespace remaining_money_l26_2626

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def john_savings : Nat := base_to_decimal [5, 3, 2, 5] 9
def ticket_cost : Nat := base_to_decimal [0, 5, 2, 1] 8

theorem remaining_money :
  john_savings - ticket_cost = 3159 := by sorry

end remaining_money_l26_2626


namespace bees_count_second_day_l26_2687

theorem bees_count_second_day (first_day_count : ℕ) (second_day_multiplier : ℕ) :
  first_day_count = 144 →
  second_day_multiplier = 3 →
  first_day_count * second_day_multiplier = 432 :=
by
  sorry

end bees_count_second_day_l26_2687


namespace geometryville_schools_l26_2696

theorem geometryville_schools (n : ℕ) : 
  n > 0 → 
  let total_students := 4 * n
  let andreas_rank := (12 * n + 1) / 4
  andreas_rank > total_students / 2 →
  andreas_rank ≤ 3 * total_students / 4 →
  (∃ (teammate_rank : ℕ), 
    teammate_rank ≤ total_students / 2 ∧ 
    teammate_rank < andreas_rank) →
  (∃ (bottom_teammates : Fin 2 → ℕ), 
    ∀ i, bottom_teammates i > total_students / 2 ∧ 
         bottom_teammates i < andreas_rank) →
  n = 3 := by
sorry

end geometryville_schools_l26_2696


namespace share_division_l26_2663

/-- Given a total sum of 427 to be divided among three people A, B, and C,
    where 3 times A's share equals 4 times B's share equals 7 times C's share,
    C's share is 84. -/
theorem share_division (a b c : ℚ) : 
  a + b + c = 427 →
  3 * a = 4 * b →
  4 * b = 7 * c →
  c = 84 := by
  sorry

end share_division_l26_2663


namespace diophantine_equation_solutions_l26_2684

theorem diophantine_equation_solutions :
  ∀ x y z w : ℕ,
  2^x * 3^y - 5^z * 7^w = 1 ↔
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1) :=
by sorry

end diophantine_equation_solutions_l26_2684
