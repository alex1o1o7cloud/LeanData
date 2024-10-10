import Mathlib

namespace unique_prime_triple_l3969_396910

theorem unique_prime_triple : ∃! (p q r : ℕ), 
  Prime p ∧ Prime q ∧ Prime r ∧
  p > q ∧ q > r ∧
  Prime (p - q) ∧ Prime (p - r) ∧ Prime (q - r) ∧
  p = 7 ∧ q = 5 ∧ r = 2 := by
sorry

end unique_prime_triple_l3969_396910


namespace box_height_l3969_396935

theorem box_height (long_width short_width top_area total_area : ℝ) 
  (h_long : long_width = 8)
  (h_short : short_width = 5)
  (h_top : top_area = 40)
  (h_total : total_area = 236) : 
  ∃ height : ℝ, 
    2 * long_width * height + 2 * short_width * height + 2 * top_area = total_area ∧ 
    height = 6 := by
  sorry

end box_height_l3969_396935


namespace andrews_cookie_expenditure_l3969_396902

/-- The number of days in May -/
def days_in_may : ℕ := 31

/-- The number of cookies Andrew buys each day -/
def cookies_per_day : ℕ := 3

/-- The cost of each cookie in dollars -/
def cost_per_cookie : ℕ := 15

/-- The total amount Andrew spent on cookies in May -/
def total_spent : ℕ := days_in_may * cookies_per_day * cost_per_cookie

/-- Theorem stating that Andrew spent 1395 dollars on cookies in May -/
theorem andrews_cookie_expenditure : total_spent = 1395 := by
  sorry

end andrews_cookie_expenditure_l3969_396902


namespace arithmetic_sequence_10th_term_l3969_396954

/-- Given an arithmetic sequence where the 5th term is 25 and the 8th term is 43,
    prove that the 10th term is 55. -/
theorem arithmetic_sequence_10th_term
  (a : ℕ → ℕ)  -- The arithmetic sequence
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))  -- Arithmetic property
  (h_5th : a 5 = 25)  -- 5th term is 25
  (h_8th : a 8 = 43)  -- 8th term is 43
  : a 10 = 55 := by
  sorry

end arithmetic_sequence_10th_term_l3969_396954


namespace kids_to_adult_ticket_ratio_l3969_396992

def admission_price : ℝ := 30
def group_size : ℕ := 10
def num_children : ℕ := 4
def num_adults : ℕ := group_size - num_children
def discount_rate : ℝ := 0.2
def soda_price : ℝ := 5
def total_paid : ℝ := 197

def adult_ticket_price : ℝ := admission_price

theorem kids_to_adult_ticket_ratio :
  ∃ (kids_ticket_price : ℝ),
    kids_ticket_price > 0 ∧
    adult_ticket_price > 0 ∧
    (1 - discount_rate) * (num_adults * adult_ticket_price + num_children * kids_ticket_price) + soda_price = total_paid ∧
    kids_ticket_price / adult_ticket_price = 1 / 2 :=
by sorry

end kids_to_adult_ticket_ratio_l3969_396992


namespace prob_sum_greater_than_4_l3969_396967

/-- The number of faces on each die -/
def num_faces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def total_outcomes : ℕ := num_faces * num_faces

/-- The number of outcomes where the sum is 4 or less -/
def outcomes_sum_4_or_less : ℕ := 6

/-- The probability that the sum of two dice is greater than 4 -/
theorem prob_sum_greater_than_4 : 
  (total_outcomes - outcomes_sum_4_or_less : ℚ) / total_outcomes = 5 / 6 :=
sorry

end prob_sum_greater_than_4_l3969_396967


namespace chess_game_probability_l3969_396998

theorem chess_game_probability (p_not_lose p_draw : ℝ) 
  (h1 : p_not_lose = 0.8) 
  (h2 : p_draw = 0.5) : 
  p_not_lose - p_draw = 0.3 := by
  sorry

end chess_game_probability_l3969_396998


namespace plane_through_origin_l3969_396916

/-- A plane in 3D Cartesian coordinates represented by the equation Ax + By + Cz = 0 -/
structure Plane3D where
  A : ℝ
  B : ℝ
  C : ℝ
  not_all_zero : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0

/-- A point in 3D Cartesian coordinates -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The origin in 3D Cartesian coordinates -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- A point lies on a plane if it satisfies the plane's equation -/
def lies_on (p : Point3D) (plane : Plane3D) : Prop :=
  plane.A * p.x + plane.B * p.y + plane.C * p.z = 0

/-- A plane passes through the origin if the origin lies on the plane -/
def passes_through_origin (plane : Plane3D) : Prop :=
  lies_on origin plane

theorem plane_through_origin (plane : Plane3D) : 
  passes_through_origin plane :=
sorry

end plane_through_origin_l3969_396916


namespace middle_part_value_l3969_396999

theorem middle_part_value (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ x : ℚ, x > 0 ∧ a * x + b * x + c * x = total ∧ b * x = 40 :=
by sorry

end middle_part_value_l3969_396999


namespace square_graph_triangles_l3969_396978

/-- Represents a planar graph formed by connecting points in a square --/
structure SquareGraph where
  /-- The number of internal points marked in the square --/
  internalPoints : ℕ
  /-- The total number of vertices (internal points + 4 square vertices) --/
  totalVertices : ℕ
  /-- The number of edges in the graph --/
  edges : ℕ
  /-- The number of faces (regions) formed, including the external face --/
  faces : ℕ
  /-- Condition: The total vertices is the sum of internal points and square vertices --/
  vertexCount : totalVertices = internalPoints + 4
  /-- Condition: Euler's formula for planar graphs --/
  eulerFormula : totalVertices - edges + faces = 2
  /-- Condition: Relationship between edges and faces --/
  edgeFaceRelation : 2 * edges = 3 * (faces - 1) + 4

/-- Theorem: In a square with 20 internal points connected as described, 42 triangles are formed --/
theorem square_graph_triangles (g : SquareGraph) (h : g.internalPoints = 20) : g.faces - 1 = 42 := by
  sorry


end square_graph_triangles_l3969_396978


namespace gumball_packages_l3969_396988

theorem gumball_packages (package_size : ℕ) (total_consumed : ℕ) 
  (h1 : package_size = 5)
  (h2 : total_consumed = 20) :
  (total_consumed / package_size = 4) ∧ (total_consumed % package_size = 0) := by
  sorry

end gumball_packages_l3969_396988


namespace unique_prime_sum_difference_l3969_396958

theorem unique_prime_sum_difference : ∃! p : ℕ, 
  Nat.Prime p ∧ 
  (∃ a b : ℕ, Nat.Prime a ∧ Nat.Prime b ∧ p = a + b) ∧
  (∃ c d : ℕ, Nat.Prime c ∧ Nat.Prime d ∧ p = c - d) ∧
  p = 5 := by
  sorry

end unique_prime_sum_difference_l3969_396958


namespace factors_of_135_l3969_396907

theorem factors_of_135 : Nat.card (Nat.divisors 135) = 8 := by
  sorry

end factors_of_135_l3969_396907


namespace mark_distance_before_turning_l3969_396953

/-- Proves that Mark walked 7.5 miles before turning around -/
theorem mark_distance_before_turning (chris_speed : ℝ) (school_distance : ℝ) 
  (mark_extra_time : ℝ) (h1 : chris_speed = 3) (h2 : school_distance = 9) 
  (h3 : mark_extra_time = 2) : 
  let chris_time := school_distance / chris_speed
  let mark_time := chris_time + mark_extra_time
  let mark_total_distance := chris_speed * mark_time
  mark_total_distance / 2 = 7.5 := by
  sorry

end mark_distance_before_turning_l3969_396953


namespace max_value_theorem_l3969_396906

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 2) :
  a + b^3 + c^4 ≤ 2 ∧ ∃ (a' b' c' : ℝ), a' + b'^3 + c'^4 = 2 ∧ 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 2 :=
by sorry

end max_value_theorem_l3969_396906


namespace square_perimeters_product_l3969_396947

theorem square_perimeters_product (x y : ℝ) 
  (h1 : x ^ 2 + y ^ 2 = 85)
  (h2 : x ^ 2 - y ^ 2 = 45)
  : (4 * x) * (4 * y) = 32 * Real.sqrt 325 := by
  sorry

end square_perimeters_product_l3969_396947


namespace m_range_l3969_396915

/-- Given a real number a where 0 < a < 1, and m is a real number. -/
def a_condition (a : ℝ) : Prop := 0 < a ∧ a < 1

/-- The solution set of ax^2 - ax - 2a^2 > 1 is (-a, 2a) -/
def inequality_solution_set (a : ℝ) : Prop :=
  ∀ x, a * x^2 - a * x - 2 * a^2 > 1 ↔ -a < x ∧ x < 2*a

/-- The domain of f(x) = sqrt((1/a)^(x^2 + 2mx - m) - 1) is ℝ -/
def function_domain (a m : ℝ) : Prop :=
  ∀ x, (1/a)^(x^2 + 2*m*x - m) - 1 ≥ 0

/-- The main theorem stating that given the conditions, the range of m is [-1, 0] -/
theorem m_range (a m : ℝ) :
  a_condition a →
  inequality_solution_set a →
  function_domain a m →
  -1 ≤ m ∧ m ≤ 0 :=
by sorry

end m_range_l3969_396915


namespace shaded_to_white_area_ratio_l3969_396930

theorem shaded_to_white_area_ratio : 
  ∀ (quarter_shaded_triangles quarter_white_triangles : ℕ) 
    (total_quarters : ℕ) 
    (shaded_area white_area : ℝ),
  quarter_shaded_triangles = 5 →
  quarter_white_triangles = 3 →
  total_quarters = 4 →
  shaded_area = (quarter_shaded_triangles * total_quarters : ℝ) →
  white_area = (quarter_white_triangles * total_quarters : ℝ) →
  shaded_area / white_area = 5 / 3 := by
sorry

end shaded_to_white_area_ratio_l3969_396930


namespace system_solution_l3969_396971

theorem system_solution (x y : ℚ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : 
  (x + y) / 3 = 5 / 3 := by
  sorry

end system_solution_l3969_396971


namespace jill_clothing_expenditure_l3969_396934

theorem jill_clothing_expenditure 
  (total : ℝ) 
  (food_percent : ℝ) 
  (other_percent : ℝ) 
  (clothing_tax_rate : ℝ) 
  (other_tax_rate : ℝ) 
  (total_tax_rate : ℝ) 
  (h1 : food_percent = 0.2)
  (h2 : other_percent = 0.3)
  (h3 : clothing_tax_rate = 0.04)
  (h4 : other_tax_rate = 0.1)
  (h5 : total_tax_rate = 0.05)
  (h6 : clothing_tax_rate * (1 - food_percent - other_percent) * total + 
        other_tax_rate * other_percent * total = total_tax_rate * total) :
  1 - food_percent - other_percent = 0.5 := by
sorry

end jill_clothing_expenditure_l3969_396934


namespace AAA_not_congruence_l3969_396969

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)  -- side lengths
  (α β γ : ℝ)  -- angles in radians

-- Define triangle congruence
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

-- Define AAA condition
def AAA (t1 t2 : Triangle) : Prop :=
  t1.α = t2.α ∧ t1.β = t2.β ∧ t1.γ = t2.γ

-- Theorem: AAA does not imply congruence
theorem AAA_not_congruence :
  ∃ t1 t2 : Triangle, AAA t1 t2 ∧ ¬(congruent t1 t2) :=
sorry

end AAA_not_congruence_l3969_396969


namespace arithmetic_sequence_terms_l3969_396946

theorem arithmetic_sequence_terms (a₁ aₙ : ℤ) (n : ℕ) : 
  a₁ = -1 → aₙ = 89 → aₙ = a₁ + (n - 1) * ((aₙ - a₁) / (n - 1)) → n = 46 := by
  sorry

end arithmetic_sequence_terms_l3969_396946


namespace smallest_n_for_divisible_sum_l3969_396918

theorem smallest_n_for_divisible_sum (n : ℕ) : n ≥ 4 → (
  (∀ S : Finset ℤ, S.card = n → ∃ a b c d : ℤ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 20 ∣ (a + b - c - d))
  ↔ n ≥ 9
) := by sorry

end smallest_n_for_divisible_sum_l3969_396918


namespace circle_and_m_value_l3969_396903

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem conditions
def problem_conditions (C : Circle) (m : ℝ) : Prop :=
  -- Center C is on the line 2x-y-7=0
  2 * C.center.1 - C.center.2 - 7 = 0 ∧
  -- Circle intersects y-axis at (0, -4) and (0, -2)
  (0 - C.center.1)^2 + (-4 - C.center.2)^2 = C.radius^2 ∧
  (0 - C.center.1)^2 + (-2 - C.center.2)^2 = C.radius^2 ∧
  -- Line x+2y+m=0 intersects circle C
  ∃ (A B : ℝ × ℝ), 
    (A.1 + 2*A.2 + m = 0) ∧
    (B.1 + 2*B.2 + m = 0) ∧
    (A.1 - C.center.1)^2 + (A.2 - C.center.2)^2 = C.radius^2 ∧
    (B.1 - C.center.1)^2 + (B.2 - C.center.2)^2 = C.radius^2 ∧
  -- Parallelogram ACBD with CA and CB as adjacent sides, D on circle C
  ∃ (D : ℝ × ℝ),
    (D.1 - C.center.1)^2 + (D.2 - C.center.2)^2 = C.radius^2

-- Theorem statement
theorem circle_and_m_value (C : Circle) (m : ℝ) :
  problem_conditions C m →
  (C.center = (2, -3) ∧ C.radius^2 = 5) ∧ (m = 3/2 ∨ m = 13/2) :=
sorry

end circle_and_m_value_l3969_396903


namespace reachable_cells_after_ten_moves_l3969_396972

-- Define the board size
def boardSize : ℕ := 21

-- Define the number of moves
def numMoves : ℕ := 10

-- Define a function to calculate the number of reachable cells
def reachableCells (moves : ℕ) : ℕ :=
  if moves % 2 = 0 then
    1 + 2 * moves * (moves + 1)
  else
    (moves + 1) ^ 2

-- Theorem statement
theorem reachable_cells_after_ten_moves :
  reachableCells numMoves = 121 := by
  sorry

end reachable_cells_after_ten_moves_l3969_396972


namespace second_player_wins_l3969_396905

/-- Represents the possible moves in the game -/
inductive Move where
  | two : Move
  | four : Move
  | five : Move

/-- Defines the game state -/
structure GameState where
  chips : Nat
  player_turn : Bool  -- True for first player, False for second player

/-- Determines if a position is winning for the current player -/
def is_winning_position (state : GameState) : Bool :=
  match state.chips % 7 with
  | 0 | 1 | 3 => false
  | _ => true

/-- Theorem stating that the second player has a winning strategy when starting with 2016 chips -/
theorem second_player_wins :
  let initial_state : GameState := { chips := 2016, player_turn := true }
  ¬(is_winning_position initial_state) := by
  sorry

end second_player_wins_l3969_396905


namespace range_of_a_l3969_396989

-- Define the statements p and q
def p (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^x > a^y
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + x + a > 0

-- Define the set of valid a values
def valid_a : Set ℝ := {a | (0 < a ∧ a ≤ 1/4) ∨ (a ≥ 1)}

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ valid_a :=
sorry

end range_of_a_l3969_396989


namespace inscribed_circle_tangent_difference_l3969_396938

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadrilateralWithInscribedCircle where
  -- Side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Tangent points divide sides
  t_a : ℝ
  t_b : ℝ
  t_c : ℝ
  t_d : ℝ
  -- Conditions
  side_sum : a + b = t_a + t_b
  side_sum' : b + c = t_b + t_c
  side_sum'' : c + d = t_c + t_d
  side_sum''' : d + a = t_d + t_a

/-- The main theorem -/
theorem inscribed_circle_tangent_difference 
  (q : CyclicQuadrilateralWithInscribedCircle)
  (h1 : q.a = 70)
  (h2 : q.b = 90)
  (h3 : q.c = 130)
  (h4 : q.d = 110) :
  |q.t_c - (q.c - q.t_c)| = 13 := by sorry

end inscribed_circle_tangent_difference_l3969_396938


namespace not_divisible_by_8_main_result_l3969_396944

def last_three_digits (n : ℕ) : ℕ :=
  n % 1000

theorem not_divisible_by_8 (n : ℕ) (h : n = 456294604884) :
  ¬(8 ∣ n) ↔ ¬(8 ∣ last_three_digits n) :=
by
  sorry

theorem main_result : ¬(8 ∣ 456294604884) :=
by
  sorry

end not_divisible_by_8_main_result_l3969_396944


namespace complex_exponential_form_l3969_396963

/-- For the complex number z = 1 + i√3, when expressed in the form re^(iθ), θ = π/3 -/
theorem complex_exponential_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (π / 3)) :=
by sorry

end complex_exponential_form_l3969_396963


namespace concentric_circles_radii_inequality_l3969_396925

theorem concentric_circles_radii_inequality (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a < b) (h5 : b < c) : 
  b + a ≠ c + b := by
sorry

end concentric_circles_radii_inequality_l3969_396925


namespace mike_final_cards_l3969_396986

def mike_cards (initial : ℕ) (received : ℕ) (traded : ℕ) : ℕ :=
  initial + received - traded

theorem mike_final_cards :
  mike_cards 64 18 20 = 62 := by
  sorry

end mike_final_cards_l3969_396986


namespace arithmetic_sequence_length_l3969_396929

theorem arithmetic_sequence_length (a d last : ℕ) (h : last = a + (n - 1) * d) : 
  a = 2 → d = 5 → last = 2507 → n = 502 := by
  sorry

end arithmetic_sequence_length_l3969_396929


namespace evaluate_expression_l3969_396961

-- Define x in terms of b
def x (b : ℝ) : ℝ := b + 9

-- Theorem to prove
theorem evaluate_expression (b : ℝ) : x b - b + 5 = 14 := by
  sorry

end evaluate_expression_l3969_396961


namespace large_cube_probabilities_l3969_396970

/-- Represents a large cube composed of 27 smaller dice -/
structure LargeCube where
  dice : Fin 27 → Die

/-- Represents a single die -/
structure Die where
  faces : Fin 6 → Nat

/-- Represents the position of a die in the large cube -/
inductive Position
  | FaceCenter
  | Edge
  | Corner

/-- Returns the position of a die given its index in the large cube -/
def diePosition (i : Fin 27) : Position := sorry

/-- Returns the probability of a specific face showing based on the die's position -/
def faceProbability (p : Position) (face : Nat) : ℚ := sorry

/-- Calculates the probability of exactly 25 sixes showing on the surface -/
def probExactly25Sixes (c : LargeCube) : ℚ := sorry

/-- Calculates the probability of at least one 'one' showing on the surface -/
def probAtLeastOne1 (c : LargeCube) : ℚ := sorry

/-- Calculates the expected number of sixes showing on the surface -/
def expectedSixes (c : LargeCube) : ℚ := sorry

/-- Calculates the expected sum of the numbers showing on the surface -/
def expectedSum (c : LargeCube) : ℚ := sorry

/-- Calculates the expected number of distinct digits appearing on the surface -/
def expectedDistinctDigits (c : LargeCube) : ℚ := sorry

theorem large_cube_probabilities (c : LargeCube) :
  probExactly25Sixes c = 31 / (2^13 * 3^18) ∧
  probAtLeastOne1 c = 1 - (5^6 / (2^2 * 3^18)) ∧
  expectedSixes c = 9 ∧
  expectedSum c = 189 ∧
  expectedDistinctDigits c = 6 * (1 - (5^6 / (2^2 * 3^18))) := by
  sorry

end large_cube_probabilities_l3969_396970


namespace inequality_equivalence_l3969_396909

theorem inequality_equivalence (x : ℝ) : 
  (-1/3 : ℝ) ≤ (5-x)/2 ∧ (5-x)/2 < (1/3 : ℝ) ↔ (13/3 : ℝ) < x ∧ x ≤ (17/3 : ℝ) := by
  sorry

end inequality_equivalence_l3969_396909


namespace no_real_solutions_l3969_396974

theorem no_real_solutions (a b c : ℝ) : ¬ ∃ x y z : ℝ, 
  (a^2 + b^2 + c^2 + 3*(x^2 + y^2 + z^2) = 6) ∧ (a*x + b*y + c*z = 2) := by
  sorry

end no_real_solutions_l3969_396974


namespace roses_in_vase_l3969_396975

/-- The number of roses in a vase after adding more roses -/
def total_roses (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: Given 10 initial roses and 8 added roses, the total is 18 -/
theorem roses_in_vase : total_roses 10 8 = 18 := by
  sorry

end roses_in_vase_l3969_396975


namespace triangle_side_length_l3969_396911

theorem triangle_side_length (perimeter side2 side3 : ℝ) 
  (h_perimeter : perimeter = 160)
  (h_side2 : side2 = 50)
  (h_side3 : side3 = 70) :
  perimeter - side2 - side3 = 40 := by
  sorry

end triangle_side_length_l3969_396911


namespace opposite_numbers_proof_l3969_396956

theorem opposite_numbers_proof : 
  (-(5^2) = -((5^2))) ∧ ((5^2) = (-5)^2) → 
  (-(5^2) = -(((-5)^2))) ∧ (-(5^2) ≠ (-5)^2) := by
sorry

end opposite_numbers_proof_l3969_396956


namespace adjacent_supplementary_angles_l3969_396973

theorem adjacent_supplementary_angles (angle_AOB angle_BOC : ℝ) : 
  angle_AOB + angle_BOC = 180 →
  angle_AOB = angle_BOC + 18 →
  angle_AOB = 99 := by
sorry

end adjacent_supplementary_angles_l3969_396973


namespace medical_team_formation_plans_l3969_396985

theorem medical_team_formation_plans (male_doctors female_doctors : ℕ) 
  (h1 : male_doctors = 5)
  (h2 : female_doctors = 4) :
  (Nat.choose male_doctors 1 * Nat.choose female_doctors 2) +
  (Nat.choose male_doctors 2 * Nat.choose female_doctors 1) = 70 := by
  sorry

end medical_team_formation_plans_l3969_396985


namespace geometric_sum_proof_l3969_396913

theorem geometric_sum_proof : 
  let a₁ : ℚ := 3/4
  let r : ℚ := 3/4
  let n : ℕ := 10
  let S := a₁ * (1 - r^n) / (1 - r)
  S = 2968581/1048576 := by sorry

end geometric_sum_proof_l3969_396913


namespace yellow_raisins_amount_l3969_396908

theorem yellow_raisins_amount (yellow_raisins black_raisins total_raisins : ℝ) 
  (h1 : black_raisins = 0.4)
  (h2 : total_raisins = 0.7)
  (h3 : yellow_raisins + black_raisins = total_raisins) : 
  yellow_raisins = 0.3 := by
  sorry

end yellow_raisins_amount_l3969_396908


namespace sum_of_binary_numbers_l3969_396952

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 101₂ -/
def binary1 : List Bool := [true, false, true]

/-- The second binary number 110₂ -/
def binary2 : List Bool := [false, true, true]

theorem sum_of_binary_numbers :
  binary_to_decimal binary1 + binary_to_decimal binary2 = 11 := by
  sorry

end sum_of_binary_numbers_l3969_396952


namespace sum_of_last_two_digits_of_series_l3969_396957

def fibonacci_factorial_series := [2, 3, 5, 8, 13, 21]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem sum_of_last_two_digits_of_series : 
  (fibonacci_factorial_series.map (λ x => last_two_digits (factorial x))).sum = 8 := by
  sorry

end sum_of_last_two_digits_of_series_l3969_396957


namespace brothers_ages_l3969_396945

theorem brothers_ages (a b c : ℕ+) :
  a * b * c = 36 ∧ 
  a + b + c = 13 ∧ 
  (a ≤ b ∧ b ≤ c) ∧
  (b < c ∨ a < b) →
  a = 2 ∧ b = 2 ∧ c = 9 := by
  sorry

end brothers_ages_l3969_396945


namespace diminished_value_proof_l3969_396942

theorem diminished_value_proof : 
  let numbers := [12, 16, 18, 21, 28]
  let smallest_number := 1015
  let diminished_value := 7
  (∀ n ∈ numbers, (smallest_number - diminished_value) % n = 0) ∧
  (∀ m < smallest_number, ∃ n ∈ numbers, ∀ k : ℕ, m - k ≠ 0 ∨ (m - k) % n ≠ 0) :=
by sorry

end diminished_value_proof_l3969_396942


namespace conic_section_eccentricity_l3969_396996

/-- A conic section with foci F₁ and F₂ -/
structure ConicSection where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Point on a conic section -/
def PointOnConic (Γ : ConicSection) := ℝ × ℝ

/-- Distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Eccentricity of a conic section -/
def eccentricity (Γ : ConicSection) : ℝ := sorry

/-- Theorem: The eccentricity of a conic section with the given property is either 1/2 or 3/2 -/
theorem conic_section_eccentricity (Γ : ConicSection) 
  (P : PointOnConic Γ)
  (h : ∃ (k : ℝ), k > 0 ∧ 
       distance P Γ.F₁ = 4 * k ∧ 
       distance Γ.F₁ Γ.F₂ = 3 * k ∧ 
       distance P Γ.F₂ = 2 * k) :
  eccentricity Γ = 1/2 ∨ eccentricity Γ = 3/2 := by
  sorry

end conic_section_eccentricity_l3969_396996


namespace largest_value_l3969_396964

theorem largest_value (a b : ℝ) 
  (ha : 0 < a) (ha1 : a < 1) 
  (hb : 0 < b) (hb1 : b < 1) 
  (hab : a ≠ b) : 
  a + b ≥ 2 * Real.sqrt (a * b) ∧ a + b ≥ (a^2 + b^2) / (2 * a * b) := by
  sorry

end largest_value_l3969_396964


namespace prism_division_theorem_l3969_396919

/-- Represents a rectangular prism -/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c

/-- Represents the division of a rectangular prism by three planes -/
structure PrismDivision (T : RectangularPrism) where
  x : ℝ
  y : ℝ
  z : ℝ
  x_bounds : 0 < x ∧ x < T.a
  y_bounds : 0 < y ∧ y < T.b
  z_bounds : 0 < z ∧ z < T.c

/-- The theorem to be proved -/
theorem prism_division_theorem (T : RectangularPrism) (div : PrismDivision T) :
  let vol_black := div.x * div.y * div.z + 
                   div.x * (T.b - div.y) * (T.c - div.z) + 
                   (T.a - div.x) * div.y * (T.c - div.z) + 
                   (T.a - div.x) * (T.b - div.y) * div.z
  let vol_white := (T.a - div.x) * (T.b - div.y) * (T.c - div.z) + 
                   (T.a - div.x) * div.y * div.z + 
                   div.x * (T.b - div.y) * div.z + 
                   div.x * div.y * (T.c - div.z)
  vol_black = vol_white → 
  div.x = T.a / 2 ∨ div.y = T.b / 2 ∨ div.z = T.c / 2 := by
  sorry

end prism_division_theorem_l3969_396919


namespace angle_bisector_property_l3969_396923

theorem angle_bisector_property (x : ℝ) : 
  x > 0 ∧ x < 180 →
  x / 2 = (180 - x) / 3 →
  x = 72 := by
sorry

end angle_bisector_property_l3969_396923


namespace complement_A_intersect_B_A_subset_C_iff_l3969_396959

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Statement 1: Prove that (ℝ\A) ∩ B = {x | 2 < x < 3 or 7 ≤ x < 10}
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = {x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Statement 2: Prove that A ⊆ C(a) if and only if a ≥ 7
theorem A_subset_C_iff (a : ℝ) :
  A ⊆ C a ↔ a ≥ 7 := by sorry

end complement_A_intersect_B_A_subset_C_iff_l3969_396959


namespace man_wage_is_350_l3969_396960

/-- The daily wage of a man -/
def man_wage : ℝ := 350

/-- The daily wage of a woman -/
def woman_wage : ℝ := 200

/-- The total number of men -/
def num_men : ℕ := 24

/-- The total number of women -/
def num_women : ℕ := 16

/-- The total daily wages -/
def total_wages : ℝ := 11600

theorem man_wage_is_350 :
  (num_men * man_wage + num_women * woman_wage = total_wages) ∧
  ((num_men / 2) * man_wage + 37 * woman_wage = total_wages) →
  man_wage = 350 := by
  sorry

end man_wage_is_350_l3969_396960


namespace train_crossing_time_l3969_396991

/-- Proves that a train 40 meters long, traveling at 144 km/hr, will take 1 second to cross an electric pole. -/
theorem train_crossing_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 40 ∧ train_speed_kmh = 144 →
  (train_length / (train_speed_kmh * (5/18))) = 1 := by
  sorry

end train_crossing_time_l3969_396991


namespace area_of_triangle_ABC_l3969_396940

-- Define the centers of the circles
def A : ℝ × ℝ := (-2, 2)
def B : ℝ × ℝ := (3, 3)
def C : ℝ × ℝ := (10, 4)

-- Define the radii of the circles
def r_A : ℝ := 2
def r_B : ℝ := 3
def r_C : ℝ := 4

-- Define the distance between centers
def dist_AB : ℝ := r_A + r_B
def dist_BC : ℝ := r_B + r_C

-- Theorem statement
theorem area_of_triangle_ABC :
  let triangle_area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  triangle_area = 1 := by sorry

end area_of_triangle_ABC_l3969_396940


namespace building_windows_l3969_396955

/-- The number of windows already installed -/
def installed_windows : ℕ := 6

/-- The time it takes to install one window (in hours) -/
def hours_per_window : ℕ := 5

/-- The time it will take to install the remaining windows (in hours) -/
def remaining_hours : ℕ := 20

/-- The total number of windows needed for the building -/
def total_windows : ℕ := installed_windows + remaining_hours / hours_per_window

theorem building_windows : total_windows = 10 := by
  sorry

end building_windows_l3969_396955


namespace matrix_sum_proof_l3969_396981

def A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 5]
def B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; -3, -10]

theorem matrix_sum_proof :
  A + B = !![(-2 : ℤ), 5; -1, -5] := by sorry

end matrix_sum_proof_l3969_396981


namespace stereographic_projection_is_inversion_l3969_396928

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  point : Point3D

/-- Represents a sphere (Earth) -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- Stereographic projection from a pole onto a plane -/
def stereographicProjection (sphere : Sphere) (pole : Point3D) (plane : Plane) (point : Point3D) : Point3D :=
  sorry

/-- The mapping between corresponding points on two planes -/
def planeMapping (sphere : Sphere) (plane1 : Plane) (plane2 : Plane) (point : Point3D) : Point3D :=
  sorry

/-- Definition of inversion -/
def isInversion (f : Point3D → Point3D) (center : Point3D) (radius : ℝ) : Prop :=
  sorry

theorem stereographic_projection_is_inversion
  (sphere : Sphere)
  (northPole : Point3D)
  (southPole : Point3D)
  (plane1 : Plane)
  (plane2 : Plane)
  (h1 : plane1.point = northPole)
  (h2 : plane2.point = southPole)
  (h3 : northPole.z = sphere.radius)
  (h4 : southPole.z = -sphere.radius) :
  ∃ (center : Point3D) (radius : ℝ),
    isInversion (planeMapping sphere plane1 plane2) center radius :=
  sorry

end stereographic_projection_is_inversion_l3969_396928


namespace largest_mersenne_prime_under_1000_l3969_396980

def is_mersenne_prime (p : Nat) : Prop :=
  ∃ n : Nat, Prime n ∧ p = 2^n - 1 ∧ Prime p

theorem largest_mersenne_prime_under_1000 :
  (∀ q : Nat, is_mersenne_prime q ∧ q < 1000 → q ≤ 127) ∧
  is_mersenne_prime 127 ∧
  127 < 1000 :=
sorry

end largest_mersenne_prime_under_1000_l3969_396980


namespace probability_threshold_min_probability_value_l3969_396912

/-- The probability that Alex and Dylan are on the same team given their card picks -/
def probability_same_team (a : ℕ) : ℚ :=
  let total_outcomes := (50 : ℚ) * 49 / 2
  let favorable_outcomes := ((a - 1 : ℚ) * (a - 2) / 2) + ((43 - a : ℚ) * (42 - a) / 2)
  favorable_outcomes / total_outcomes

/-- The minimum value of a for which the probability is at least 1/2 -/
def min_a : ℕ := 8

theorem probability_threshold :
  probability_same_team min_a ≥ 1/2 ∧
  ∀ a < min_a, probability_same_team a < 1/2 :=
sorry

theorem min_probability_value :
  probability_same_team min_a = 88/175 :=
sorry

end probability_threshold_min_probability_value_l3969_396912


namespace sphere_volume_constant_l3969_396931

theorem sphere_volume_constant (cube_side : Real) (K : Real) : 
  cube_side = 3 →
  (4 / 3 * Real.pi * (((6 * cube_side^2) / (4 * Real.pi))^(3/2))) = K * Real.sqrt 6 / Real.sqrt Real.pi →
  K = 54 * Real.sqrt 2 := by
sorry

end sphere_volume_constant_l3969_396931


namespace complex_fraction_simplification_l3969_396979

theorem complex_fraction_simplification : (Complex.I + 2) / (1 - 2 * Complex.I) = Complex.I := by
  sorry

end complex_fraction_simplification_l3969_396979


namespace combinatorial_equality_l3969_396982

theorem combinatorial_equality (n : ℕ) : 
  (n.choose 2) * 2 = 42 → n.choose 3 = 35 := by
  sorry

end combinatorial_equality_l3969_396982


namespace circle_equation_k_range_l3969_396994

theorem circle_equation_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0 ∧ 
    ∃ h r : ℝ, ∀ x' y' : ℝ, (x' - h)^2 + (y' - r)^2 = (x^2 + y^2 + 2*k*x + 4*y + 3*k + 8)) 
  ↔ (k > 4 ∨ k < -1) := by
  sorry

end circle_equation_k_range_l3969_396994


namespace house_painting_time_l3969_396904

/-- Represents the time taken to paint a house given individual rates and a break -/
theorem house_painting_time
  (alice_rate : ℝ) (bob_rate : ℝ) (carlos_rate : ℝ) (break_time : ℝ) (total_time : ℝ)
  (h_alice : alice_rate = 1 / 4)
  (h_bob : bob_rate = 1 / 6)
  (h_carlos : carlos_rate = 1 / 12)
  (h_break : break_time = 2)
  (h_equation : (alice_rate + bob_rate + carlos_rate) * (total_time - break_time) = 1) :
  (1 / 4 + 1 / 6 + 1 / 12) * (total_time - 2) = 1 := by
sorry


end house_painting_time_l3969_396904


namespace processing_block_performs_assignment_calculation_l3969_396983

-- Define the types of program blocks
inductive ProgramBlock
  | Terminal
  | InputOutput
  | Processing
  | Decision

-- Define the functions that a block can perform
inductive BlockFunction
  | StartStop
  | InformationIO
  | AssignmentCalculation
  | ConditionCheck

-- Define a function that maps a block to its primary function
def blockPrimaryFunction : ProgramBlock → BlockFunction
  | ProgramBlock.Terminal => BlockFunction.StartStop
  | ProgramBlock.InputOutput => BlockFunction.InformationIO
  | ProgramBlock.Processing => BlockFunction.AssignmentCalculation
  | ProgramBlock.Decision => BlockFunction.ConditionCheck

-- Theorem statement
theorem processing_block_performs_assignment_calculation :
  ∀ (block : ProgramBlock),
    blockPrimaryFunction block = BlockFunction.AssignmentCalculation
    ↔ block = ProgramBlock.Processing :=
by sorry

end processing_block_performs_assignment_calculation_l3969_396983


namespace point_in_second_quadrant_l3969_396993

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant (m : ℝ) :
  second_quadrant (1 + m) 3 ↔ m < -1 := by
  sorry

end point_in_second_quadrant_l3969_396993


namespace subtraction_result_l3969_396901

theorem subtraction_result : 34.256 - 12.932 - 1.324 = 20 := by
  sorry

end subtraction_result_l3969_396901


namespace track_circumference_is_720_l3969_396976

/-- Represents the circular track and the movement of A and B -/
structure CircularTrack where
  circumference : ℝ
  speed_A : ℝ
  speed_B : ℝ

/-- The conditions of the problem -/
def problem_conditions (track : CircularTrack) : Prop :=
  ∃ (t₁ t₂ : ℝ),
    t₁ > 0 ∧ t₂ > t₁ ∧
    track.speed_B * t₁ = 150 ∧
    track.speed_A * t₁ = track.circumference / 2 - 150 ∧
    track.speed_A * t₂ = track.circumference - 90 ∧
    track.speed_B * t₂ = track.circumference / 2 + 90

/-- The theorem stating that the track circumference is 720 yards -/
theorem track_circumference_is_720 (track : CircularTrack) :
  problem_conditions track → track.circumference = 720 := by
  sorry


end track_circumference_is_720_l3969_396976


namespace negation_of_existence_squared_positive_l3969_396914

theorem negation_of_existence_squared_positive :
  (¬ ∃ x : ℝ, x^2 > 0) ↔ (∀ x : ℝ, x^2 ≤ 0) := by sorry

end negation_of_existence_squared_positive_l3969_396914


namespace sons_age_l3969_396941

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 29 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 27 := by
sorry

end sons_age_l3969_396941


namespace bobs_cycling_wins_l3969_396997

/-- The minimum number of additional weeks Bob must win first place to afford a puppy -/
theorem bobs_cycling_wins (puppy_cost : ℕ) (initial_wins : ℕ) (prize_money : ℕ) 
  (h1 : puppy_cost = 1000)
  (h2 : initial_wins = 2)
  (h3 : prize_money = 100) : 
  (puppy_cost - initial_wins * prize_money) / prize_money = 8 := by
  sorry

end bobs_cycling_wins_l3969_396997


namespace square_area_l3969_396900

/-- Square in a coordinate plane --/
structure Square where
  B : ℝ × ℝ
  C : ℝ × ℝ
  E : ℝ × ℝ
  BC_is_side : True  -- Represents that BC is a side of the square
  E_on_line : True   -- Represents that E is on a line intersecting another vertex

/-- The area of the square ABCD is 4 --/
theorem square_area (s : Square) (h1 : s.B = (0, 0)) (h2 : s.C = (2, 0)) (h3 : s.E = (2, 1)) : 
  (s.C.1 - s.B.1) ^ 2 = 4 := by
  sorry


end square_area_l3969_396900


namespace prism_configuration_impossible_l3969_396927

/-- A rectangular prism in 3D space -/
structure RectangularPrism where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  z_min : ℝ
  z_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max
  h_z : z_min < z_max

/-- Two prisms intersect if their projections overlap on all axes -/
def intersects (p q : RectangularPrism) : Prop :=
  (p.x_min < q.x_max ∧ q.x_min < p.x_max) ∧
  (p.y_min < q.y_max ∧ q.y_min < p.y_max) ∧
  (p.z_min < q.z_max ∧ q.z_min < p.z_max)

/-- A configuration of 12 prisms satisfying the problem conditions -/
structure PrismConfiguration where
  prisms : Fin 12 → RectangularPrism
  h_intersects : ∀ i j : Fin 12, i ≠ j → 
    (i.val + 1) % 12 ≠ j.val ∧ (i.val + 11) % 12 ≠ j.val → 
    intersects (prisms i) (prisms j)
  h_non_intersects : ∀ i : Fin 12, 
    ¬intersects (prisms i) (prisms ⟨(i.val + 1) % 12, sorry⟩) ∧
    ¬intersects (prisms i) (prisms ⟨(i.val + 11) % 12, sorry⟩)

/-- The main theorem stating the impossibility of such a configuration -/
theorem prism_configuration_impossible : ¬∃ (config : PrismConfiguration), True :=
  sorry

end prism_configuration_impossible_l3969_396927


namespace second_reduction_percentage_store_price_reduction_l3969_396951

theorem second_reduction_percentage 
  (initial_reduction : Real) 
  (final_price_percentage : Real) : Real :=
  let price_after_first_reduction := 1 - initial_reduction
  let second_reduction := (price_after_first_reduction - final_price_percentage) / price_after_first_reduction
  14 / 100

-- The main theorem
theorem store_price_reduction 
  (initial_reduction : Real)
  (final_price_percentage : Real)
  (h1 : initial_reduction = 10 / 100)
  (h2 : final_price_percentage = 77.4 / 100) :
  second_reduction_percentage initial_reduction final_price_percentage = 14 / 100 := by
  sorry

end second_reduction_percentage_store_price_reduction_l3969_396951


namespace largest_prime_to_check_for_primality_l3969_396995

theorem largest_prime_to_check_for_primality (n : ℕ) :
  2500 ≤ n → n ≤ 2600 →
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q : ℕ, Nat.Prime q → q^2 ≤ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q : ℕ, Nat.Prime q → q^2 ≤ n → q ≤ p ∧ p = 47) :=
by sorry

end largest_prime_to_check_for_primality_l3969_396995


namespace reciprocal_sum_range_l3969_396943

theorem reciprocal_sum_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 1 / y ≥ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1 / a + 1 / b = 4 :=
by sorry

end reciprocal_sum_range_l3969_396943


namespace intersecting_chords_length_l3969_396933

/-- Power of a Point theorem for intersecting chords --/
axiom power_of_point (AP BP CP DP : ℝ) : AP * BP = CP * DP

/-- Proof that DP = 8/3 given the conditions --/
theorem intersecting_chords_length (AP BP CP DP : ℝ) 
  (h1 : AP = 4) 
  (h2 : CP = 9) 
  (h3 : BP = 6) : 
  DP = 8/3 := by
  sorry


end intersecting_chords_length_l3969_396933


namespace boys_running_speed_l3969_396932

/-- Given a square field with side length 60 meters and a boy who runs around it in 96 seconds,
    prove that the boy's speed is 9 km/hr. -/
theorem boys_running_speed (side_length : ℝ) (time : ℝ) (speed : ℝ) : 
  side_length = 60 →
  time = 96 →
  speed = (4 * side_length) / time * 3.6 →
  speed = 9 := by sorry

end boys_running_speed_l3969_396932


namespace age_difference_l3969_396968

/-- Given that Sachin is 14 years old and the ratio of Sachin's age to Rahul's age is 7:9,
    prove that the difference between Rahul's age and Sachin's age is 4 years. -/
theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 14 → 
  sachin_age * 9 = rahul_age * 7 →
  rahul_age - sachin_age = 4 := by
sorry

end age_difference_l3969_396968


namespace periodic_trig_function_l3969_396936

/-- Given a function f(x) = a*sin(πx + α) + b*cos(πx - β), where α, β, a, and b are non-zero real numbers,
    if f(2016) = -1, then f(2017) = 1 -/
theorem periodic_trig_function (α β a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin (π * x + α) + b * Real.cos (π * x - β)
  f 2016 = -1 → f 2017 = 1 := by
  sorry

end periodic_trig_function_l3969_396936


namespace onion_root_tip_no_tetrads_l3969_396924

/-- Represents the type of cell division a plant tissue undergoes -/
inductive CellDivisionType
  | Mitosis
  | Meiosis

/-- Represents whether tetrads can be observed in a given tissue -/
def can_observe_tetrads (division_type : CellDivisionType) : Prop :=
  match division_type with
  | CellDivisionType.Meiosis => true
  | CellDivisionType.Mitosis => false

/-- The cell division type of onion root tips -/
def onion_root_tip_division : CellDivisionType := CellDivisionType.Mitosis

theorem onion_root_tip_no_tetrads :
  ¬(can_observe_tetrads onion_root_tip_division) :=
by sorry

end onion_root_tip_no_tetrads_l3969_396924


namespace petya_has_higher_chance_of_winning_l3969_396921

structure CandyGame where
  total_candies : ℕ
  prob_two_caramels : ℝ

def vasya_wins (game : CandyGame) : ℝ :=
  1 - game.prob_two_caramels

def petya_wins (game : CandyGame) : ℝ :=
  game.prob_two_caramels

theorem petya_has_higher_chance_of_winning (game : CandyGame)
  (h1 : game.total_candies = 25)
  (h2 : game.prob_two_caramels = 0.54)
  : petya_wins game > vasya_wins game := by
  sorry

end petya_has_higher_chance_of_winning_l3969_396921


namespace estimate_shaded_area_l3969_396949

/-- Estimates the area of a shaded region within a square using Monte Carlo method. -/
theorem estimate_shaded_area (side_length : ℝ) (total_points : ℕ) (shaded_points : ℕ) : 
  side_length = 6 →
  total_points = 800 →
  shaded_points = 200 →
  (shaded_points : ℝ) / (total_points : ℝ) * side_length^2 = 9 :=
by sorry

end estimate_shaded_area_l3969_396949


namespace intersection_nonempty_condition_l3969_396920

theorem intersection_nonempty_condition (k : ℝ) : 
  let A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
  let B : Set ℝ := {x | x - k ≥ 0}
  (A ∩ B).Nonempty → k ≤ 1 :=
by sorry

end intersection_nonempty_condition_l3969_396920


namespace vodka_alcohol_percentage_l3969_396987

/-- Calculates the percentage of pure alcohol in vodka -/
theorem vodka_alcohol_percentage
  (total_shots : ℕ)
  (ounces_per_shot : ℚ)
  (pure_alcohol_consumed : ℚ) :
  total_shots = 8 →
  ounces_per_shot = 3/2 →
  pure_alcohol_consumed = 3 →
  (pure_alcohol_consumed / (total_shots * ounces_per_shot)) * 100 = 25 := by
  sorry

end vodka_alcohol_percentage_l3969_396987


namespace function_identity_l3969_396966

theorem function_identity (f : ℕ → ℕ) :
  (∀ n : ℕ, f n + f (f n) + f (f (f n)) = 3 * n) →
  (∀ n : ℕ, f n = n) :=
by sorry

end function_identity_l3969_396966


namespace intercepts_sum_l3969_396937

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the specific parabola y = 3x^2 - 9x + 4 -/
def parabola : QuadraticFunction :=
  { a := 3, b := -9, c := 4 }

/-- The y-intercept of the parabola -/
def y_intercept : Point :=
  { x := 0, y := parabola.c }

/-- Theorem stating that the sum of the y-intercept's y-coordinate and the x-coordinates of the two x-intercepts equals 19/3 -/
theorem intercepts_sum (e f : ℝ) 
  (h1 : parabola.a * e^2 + parabola.b * e + parabola.c = 0)
  (h2 : parabola.a * f^2 + parabola.b * f + parabola.c = 0)
  (h3 : e ≠ f) : 
  y_intercept.y + e + f = 19/3 := by
  sorry

end intercepts_sum_l3969_396937


namespace average_thirteen_l3969_396950

theorem average_thirteen (x : ℝ) : 
  (6 + 16 + 8 + x) / 4 = 13 → x = 22 := by
sorry

end average_thirteen_l3969_396950


namespace oldest_sibling_age_difference_l3969_396984

theorem oldest_sibling_age_difference (siblings : Fin 4 → ℝ) 
  (avg_age : (siblings 0 + siblings 1 + siblings 2 + siblings 3) / 4 = 30)
  (youngest_age : siblings 3 = 25.75)
  : ∃ i : Fin 4, siblings i - siblings 3 ≥ 17 :=
by
  sorry

end oldest_sibling_age_difference_l3969_396984


namespace intersection_empty_iff_a_in_range_l3969_396948

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_empty_iff_a_in_range (a : ℝ) :
  A a ∩ B = ∅ ↔ a ∈ Set.Ioo 2 3 := by sorry

end intersection_empty_iff_a_in_range_l3969_396948


namespace function_property_proof_l3969_396939

theorem function_property_proof (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = f (4 - x))
  (h2 : ∀ x : ℝ, f (x + 1) = -f (x + 3))
  (h3 : ∃ a b : ℝ, ∀ x ∈ Set.Icc 0 4, f x = |x - a| + b) :
  ∃ a b : ℝ, (∀ x ∈ Set.Icc 0 4, f x = |x - a| + b) ∧ a + b = 1 := by
  sorry


end function_property_proof_l3969_396939


namespace expression_is_perfect_square_l3969_396917

/-- 
Given real numbers a and b, prove that the expression 
x^2 - 4bx + 4ab + p^2 - 2px is a perfect square when p = a - b
-/
theorem expression_is_perfect_square (a b x : ℝ) : 
  ∃ k : ℝ, x^2 - 4*b*x + 4*a*b + (a - b)^2 - 2*(a - b)*x = k^2 := by
  sorry

end expression_is_perfect_square_l3969_396917


namespace square_area_from_rectangles_l3969_396922

/-- The area of a square composed of four identical rectangles and a smaller square, 
    where the perimeter of each rectangle is 28. -/
theorem square_area_from_rectangles (l w : ℝ) : 
  (l + w ≥ 0) →  -- Ensure non-negative side length
  (2 * (l + w) = 28) →  -- Perimeter of rectangle
  (l + w) * (l + w) = 196 := by
  sorry

#check square_area_from_rectangles

end square_area_from_rectangles_l3969_396922


namespace quadratic_minimum_l3969_396965

theorem quadratic_minimum (k : ℝ) : 
  (∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → (1/2) * (x - 1)^2 + k ≥ 3) ∧
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 5 ∧ (1/2) * (x - 1)^2 + k = 3) →
  k = 1 := by
  sorry

end quadratic_minimum_l3969_396965


namespace license_plate_difference_l3969_396962

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of digits -/
def digit_size : ℕ := 10

/-- The number of letters in a California license plate -/
def california_letters : ℕ := 4

/-- The number of digits in a California license plate -/
def california_digits : ℕ := 3

/-- The number of letters in a Texas license plate -/
def texas_letters : ℕ := 3

/-- The number of digits in a Texas license plate -/
def texas_digits : ℕ := 4

/-- The number of possible California license plates -/
def california_plates : ℕ := alphabet_size ^ california_letters * digit_size ^ california_digits

/-- The number of possible Texas license plates -/
def texas_plates : ℕ := alphabet_size ^ texas_letters * digit_size ^ texas_digits

/-- The difference in the number of possible license plates between California and Texas -/
theorem license_plate_difference : california_plates - texas_plates = 281216000 := by
  sorry

end license_plate_difference_l3969_396962


namespace thomas_drawings_l3969_396990

theorem thomas_drawings (colored_pencil : ℕ) (blending_markers : ℕ) (charcoal : ℕ)
  (h1 : colored_pencil = 14)
  (h2 : blending_markers = 7)
  (h3 : charcoal = 4) :
  colored_pencil + blending_markers + charcoal = 25 :=
by sorry

end thomas_drawings_l3969_396990


namespace hyperbola_quadrilateral_area_ratio_max_l3969_396977

theorem hyperbola_quadrilateral_area_ratio_max
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (S₁ : ℝ) (hS₁ : S₁ = 2 * a * b)
  (S₂ : ℝ) (hS₂ : S₂ = 2 * (a^2 + b^2)) :
  (S₁ / S₂) ≤ (1 / 2) := by
sorry

end hyperbola_quadrilateral_area_ratio_max_l3969_396977


namespace least_subtraction_for_divisibility_least_subtraction_1234567890_div_17_l3969_396926

theorem least_subtraction_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by sorry

theorem least_subtraction_1234567890_div_17 :
  ∃ (k : Nat), k < 17 ∧ (1234567890 - k) % 17 = 0 ∧
  ∀ (m : Nat), m < k → (1234567890 - m) % 17 ≠ 0 ∧ k = 5 :=
by sorry

end least_subtraction_for_divisibility_least_subtraction_1234567890_div_17_l3969_396926
