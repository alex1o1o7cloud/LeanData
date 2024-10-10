import Mathlib

namespace multiplier_is_three_l694_69431

theorem multiplier_is_three :
  ∃ (x : ℤ), 
    (3 * x = (62 - x) + 26) ∧ 
    (x = 22) → 
    3 = 3 := by
  sorry

end multiplier_is_three_l694_69431


namespace candy_difference_l694_69495

theorem candy_difference (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : red = 40)
  (h2 : yellow < 3 * red)
  (h3 : blue = yellow / 2)
  (h4 : red + blue = 90) :
  3 * red - yellow = 20 := by
  sorry

end candy_difference_l694_69495


namespace A_equals_6x_squared_A_plus_2B_A_plus_2B_at_negative_one_l694_69430

-- Define polynomials A and B
def B (x : ℝ) : ℝ := 4 * x^2 - 5 * x - 7

axiom A_minus_2B (x : ℝ) : ∃ A : ℝ → ℝ, A x - 2 * (B x) = -2 * x^2 + 10 * x + 14

-- Theorem 1: A = 6x^2
theorem A_equals_6x_squared : ∃ A : ℝ → ℝ, ∀ x : ℝ, A x = 6 * x^2 := by sorry

-- Theorem 2: A + 2B = 14x^2 - 10x - 14
theorem A_plus_2B (x : ℝ) : ∃ A : ℝ → ℝ, A x + 2 * (B x) = 14 * x^2 - 10 * x - 14 := by sorry

-- Theorem 3: When x = -1, A + 2B = 10
theorem A_plus_2B_at_negative_one : ∃ A : ℝ → ℝ, A (-1) + 2 * (B (-1)) = 10 := by sorry

end A_equals_6x_squared_A_plus_2B_A_plus_2B_at_negative_one_l694_69430


namespace average_waiting_time_for_first_bite_l694_69488

theorem average_waiting_time_for_first_bite 
  (rod1_bites : ℝ) 
  (rod2_bites : ℝ) 
  (total_bites : ℝ) 
  (time_interval : ℝ) 
  (h1 : rod1_bites = 3)
  (h2 : rod2_bites = 2)
  (h3 : total_bites = rod1_bites + rod2_bites)
  (h4 : time_interval = 6) :
  (time_interval / total_bites) = 6/5 := by
  sorry

end average_waiting_time_for_first_bite_l694_69488


namespace valid_paths_count_l694_69437

/-- The number of paths on a grid from (0,0) to (m,n) -/
def gridPaths (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The number of paths from A to C on the 6x3 grid -/
def pathsAtoC : ℕ := gridPaths 4 1

/-- The number of paths from D to B on the 6x3 grid -/
def pathsDtoB : ℕ := gridPaths 2 2

/-- The number of paths from A to E on the 6x3 grid -/
def pathsAtoE : ℕ := gridPaths 2 2

/-- The number of paths from F to B on the 6x3 grid -/
def pathsFtoB : ℕ := gridPaths 4 0

/-- The total number of paths on the 6x3 grid -/
def totalPaths : ℕ := gridPaths 6 3

/-- The number of invalid paths through the first forbidden segment -/
def invalidPaths1 : ℕ := pathsAtoC * pathsDtoB

/-- The number of invalid paths through the second forbidden segment -/
def invalidPaths2 : ℕ := pathsAtoE * pathsFtoB

theorem valid_paths_count :
  totalPaths - (invalidPaths1 + invalidPaths2) = 48 := by sorry

end valid_paths_count_l694_69437


namespace cos_pi_minus_2alpha_l694_69407

theorem cos_pi_minus_2alpha (α : ℝ) (h : Real.sin (3 * Real.pi / 2 + α) = 1 / 3) :
  Real.cos (Real.pi - 2 * α) = - 7 / 9 := by
  sorry

end cos_pi_minus_2alpha_l694_69407


namespace circle_area_through_triangle_points_l694_69424

/-- Given a right triangle PQR with legs PQ = 6 and PR = 8, the area of the circle 
    passing through points Q, R, and the midpoint M of hypotenuse QR is 25π. -/
theorem circle_area_through_triangle_points (P Q R M : ℝ × ℝ) : 
  -- Triangle PQR is a right triangle
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 →
  -- PQ = 6
  (Q.1 - P.1)^2 + (Q.2 - P.2)^2 = 36 →
  -- PR = 8
  (R.1 - P.1)^2 + (R.2 - P.2)^2 = 64 →
  -- M is the midpoint of QR
  M = ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2) →
  -- The area of the circle passing through Q, R, and M
  π * ((Q.1 - M.1)^2 + (Q.2 - M.2)^2) = 25 * π :=
by sorry

end circle_area_through_triangle_points_l694_69424


namespace number_puzzle_l694_69454

theorem number_puzzle : ∃ x : ℤ, (x + 2) - 3 = 7 ∧ x = 8 := by
  sorry

end number_puzzle_l694_69454


namespace hyperbola_vertices_distance_l694_69491

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 + 16 * x - 9 * y^2 + 18 * y - 23 = 0

-- State the theorem
theorem hyperbola_vertices_distance :
  ∃ (a b c d : ℝ),
    (∀ x y, hyperbola_equation x y ↔ ((x - a)^2 / b^2 - (y - c)^2 / d^2 = 1)) ∧
    2 * Real.sqrt b^2 = Real.sqrt 30 :=
by sorry

end hyperbola_vertices_distance_l694_69491


namespace milk_cans_problem_l694_69426

theorem milk_cans_problem (x y : ℕ) : 
  x = 2 * y ∧ 
  x - 30 = 3 * (y - 20) → 
  x = 60 ∧ y = 30 := by
sorry

end milk_cans_problem_l694_69426


namespace largest_number_hcf_lcm_l694_69490

theorem largest_number_hcf_lcm (a b : ℕ+) : 
  (Nat.gcd a b = 52) → 
  (Nat.lcm a b = 52 * 11 * 12) → 
  (max a b = 624) := by
sorry

end largest_number_hcf_lcm_l694_69490


namespace truth_table_results_l694_69458

variable (p q : Prop)

theorem truth_table_results :
  (∀ p, ¬(p ∧ ¬p)) ∧
  (∀ p, p ∨ ¬p) ∧
  (∀ p q, ¬(p ∧ q) ↔ (¬p ∨ ¬q)) ∧
  (∀ p q, (p ∨ q) ∨ ¬p) :=
by sorry

end truth_table_results_l694_69458


namespace ellipse_a_range_l694_69455

-- Define the ellipse equation
def ellipse_equation (x y a : ℝ) : Prop :=
  x^2 / a^2 + y^2 / (a + 6) = 1

-- Define the condition that the ellipse has foci on the x-axis
def foci_on_x_axis (a : ℝ) : Prop :=
  a^2 > a + 6 ∧ a + 6 > 0

-- Theorem stating the range of a
theorem ellipse_a_range :
  ∀ a : ℝ, (∃ x y : ℝ, ellipse_equation x y a ∧ foci_on_x_axis a) →
  (a > 3 ∨ (-6 < a ∧ a < -2)) :=
sorry

end ellipse_a_range_l694_69455


namespace ellipse_t_range_l694_69421

def is_ellipse (t : ℝ) : Prop :=
  ∀ x y : ℝ, x^2 / (10 - t) + y^2 / (t - 4) = 1

theorem ellipse_t_range :
  {t : ℝ | is_ellipse t} = {t | t ∈ (Set.Ioo 4 7) ∪ (Set.Ioo 7 10)} :=
by sorry

end ellipse_t_range_l694_69421


namespace lcm_18_35_l694_69467

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end lcm_18_35_l694_69467


namespace first_orphanage_donation_l694_69483

/-- Given a total donation and donations to two orphanages, 
    calculate the donation to the first orphanage -/
def donation_to_first_orphanage (total : ℚ) (second : ℚ) (third : ℚ) : ℚ :=
  total - (second + third)

theorem first_orphanage_donation 
  (total : ℚ) (second : ℚ) (third : ℚ)
  (h_total : total = 650)
  (h_second : second = 225)
  (h_third : third = 250) :
  donation_to_first_orphanage total second third = 175 := by
  sorry

end first_orphanage_donation_l694_69483


namespace bottom_right_figure_impossible_l694_69406

/-- Represents a rhombus with a fixed white and gray pattern -/
structure Rhombus :=
  (pattern : ℕ → ℕ → Bool)

/-- Represents a rotation of a rhombus -/
def rotate (r : Rhombus) (angle : ℕ) : Rhombus :=
  sorry

/-- Represents a larger figure composed of rhombuses -/
structure LargeFigure :=
  (shape : List (Rhombus × ℕ × ℕ))

/-- The specific larger figure that cannot be assembled (bottom right) -/
def bottomRightFigure : LargeFigure :=
  sorry

/-- Predicate to check if a larger figure can be assembled using only rotations of the given rhombus -/
def canAssemble (r : Rhombus) (lf : LargeFigure) : Prop :=
  sorry

/-- Theorem stating that the bottom right figure cannot be assembled -/
theorem bottom_right_figure_impossible (r : Rhombus) :
  ¬ (canAssemble r bottomRightFigure) :=
sorry

end bottom_right_figure_impossible_l694_69406


namespace m_values_l694_69404

def A : Set ℝ := {1, 3}

def B (m : ℝ) : Set ℝ := {x | m * x - 3 = 0}

theorem m_values (m : ℝ) : A ∪ B m = A → m ∈ ({0, 1, 3} : Set ℝ) := by
  sorry

end m_values_l694_69404


namespace problem_statement_l694_69474

theorem problem_statement :
  (∃ (a b : ℝ), abs (a + b) < 1 ∧ abs a + abs b ≥ 1) ∧
  (∀ x : ℝ, (x ≤ -3 ∨ x ≥ 1) ↔ |x + 1| - 2 ≥ 0) :=
by sorry

end problem_statement_l694_69474


namespace cylinder_lateral_area_l694_69459

/-- Given a cylinder with a square cross-section of area 4, its lateral area is 4π. -/
theorem cylinder_lateral_area (r h : ℝ) : 
  r * r = 4 → 2 * π * r * h = 4 * π := by
  sorry

end cylinder_lateral_area_l694_69459


namespace train_crossing_time_l694_69456

/-- Theorem: Time taken for two trains to cross each other
    Given two trains moving in opposite directions with specified speeds and lengths,
    prove that the time taken for the slower train to cross the faster train is 24 seconds. -/
theorem train_crossing_time (speed1 speed2 length1 length2 : ℝ) 
    (h1 : speed1 = 315)
    (h2 : speed2 = 135)
    (h3 : length1 = 1.65)
    (h4 : length2 = 1.35) :
    (length1 + length2) / (speed1 + speed2) * 3600 = 24 := by
  sorry

#check train_crossing_time

end train_crossing_time_l694_69456


namespace triangle_side_and_area_l694_69496

theorem triangle_side_and_area 
  (A B C : Real) 
  (a b c : Real) 
  (h1 : a = 1) 
  (h2 : b = 2) 
  (h3 : C = 60 * π / 180) : 
  c = Real.sqrt 3 ∧ (1/2 * a * b * Real.sin C) = Real.sqrt 3 / 2 := by
  sorry

end triangle_side_and_area_l694_69496


namespace root_sum_reciprocals_l694_69457

theorem root_sum_reciprocals (p q r s : ℂ) : 
  (p^4 - 6*p^3 + 11*p^2 - 6*p + 3 = 0) →
  (q^4 - 6*q^3 + 11*q^2 - 6*q + 3 = 0) →
  (r^4 - 6*r^3 + 11*r^2 - 6*r + 3 = 0) →
  (s^4 - 6*s^3 + 11*s^2 - 6*s + 3 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3 := by
sorry

end root_sum_reciprocals_l694_69457


namespace ellipse_vertices_distance_l694_69411

/-- The distance between the vertices of the ellipse (x^2/144) + (y^2/36) = 1 is 24 -/
theorem ellipse_vertices_distance : 
  let ellipse := {p : ℝ × ℝ | (p.1^2 / 144) + (p.2^2 / 36) = 1}
  ∃ v1 v2 : ℝ × ℝ, v1 ∈ ellipse ∧ v2 ∈ ellipse ∧ 
    (∀ p ∈ ellipse, ‖p.1‖ ≤ ‖v1.1‖) ∧
    ‖v1 - v2‖ = 24 :=
by sorry

end ellipse_vertices_distance_l694_69411


namespace geometric_sequence_problem_l694_69449

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def arithmetic_sequence (a b c : ℝ) : Prop :=
  b - a = c - b

theorem geometric_sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a (n + 1) = 2 * a n) →
  arithmetic_sequence (a 2) (a 3 + 1) (a 4) →
  (∀ n : ℕ, b n = a n + n) →
  a 1 = 1 ∧
  (∀ n : ℕ, a n = 2^(n - 1)) ∧
  (b 1 + b 2 + b 3 + b 4 + b 5 = 46) :=
by sorry

end geometric_sequence_problem_l694_69449


namespace arithmetic_sequence_sum_ratio_l694_69476

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_ratio
  (seq : ArithmeticSequence)
  (h : seq.a 2 / seq.a 4 = 7 / 6) :
  S seq 7 / S seq 3 = 2 / 1 := by
  sorry


end arithmetic_sequence_sum_ratio_l694_69476


namespace sum_in_base7_l694_69489

/-- Converts a base 7 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 7 * acc + d) 0

/-- Converts a decimal number to its base 7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem sum_in_base7 :
  toBase7 (toDecimal [4, 2, 3, 1] + toDecimal [1, 3, 5, 2, 6]) = [6, 0, 0, 6, 0] := by
  sorry

end sum_in_base7_l694_69489


namespace conic_is_ellipse_l694_69444

-- Define the conic section equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y+2)^2) + Real.sqrt ((x-4)^2 + (y-3)^2) = 8

-- Define the focal points
def focal_point1 : ℝ × ℝ := (0, -2)
def focal_point2 : ℝ × ℝ := (4, 3)

-- Theorem stating that the equation describes an ellipse
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), conic_equation x y ↔
    (x - (focal_point1.1 + focal_point2.1) / 2)^2 / a^2 +
    (y - (focal_point1.2 + focal_point2.2) / 2)^2 / b^2 = 1 :=
sorry

end conic_is_ellipse_l694_69444


namespace event_ticket_revenue_l694_69485

theorem event_ticket_revenue (total_tickets : ℕ) (total_revenue : ℕ) 
  (h_total_tickets : total_tickets = 160)
  (h_total_revenue : total_revenue = 2400) :
  ∃ (full_price : ℕ) (half_price : ℕ) (price : ℕ),
    full_price + half_price = total_tickets ∧
    full_price * price + half_price * (price / 2) = total_revenue ∧
    full_price * price = 960 := by
  sorry

end event_ticket_revenue_l694_69485


namespace stating_min_nickels_needed_l694_69434

/-- Represents the cost of the book in cents -/
def book_cost : ℕ := 4750

/-- Represents the value of four $10 bills in cents -/
def ten_dollar_bills : ℕ := 4000

/-- Represents the value of five half-dollars in cents -/
def half_dollars : ℕ := 250

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- 
Theorem stating that the minimum number of nickels needed to reach 
or exceed the book cost, given the other money available, is 100.
-/
theorem min_nickels_needed : 
  ∀ n : ℕ, (n * nickel_value + ten_dollar_bills + half_dollars ≥ book_cost) → n ≥ 100 := by
  sorry

end stating_min_nickels_needed_l694_69434


namespace rearrange_3008_eq_6_l694_69481

/-- The number of different four-digit numbers that can be formed by rearranging the digits in 3008 -/
def rearrange_3008 : ℕ :=
  let digits : List ℕ := [3, 0, 0, 8]
  let total_permutations := Nat.factorial 4 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1)
  let valid_permutations := 
    (Nat.factorial 3 / Nat.factorial 2) +  -- starting with 3
    (Nat.factorial 3 / Nat.factorial 2)    -- starting with 8
  valid_permutations

theorem rearrange_3008_eq_6 : rearrange_3008 = 6 := by
  sorry

end rearrange_3008_eq_6_l694_69481


namespace lenny_pens_boxes_l694_69477

theorem lenny_pens_boxes : ∀ (total_pens : ℕ) (pens_per_box : ℕ),
  pens_per_box = 5 →
  (total_pens : ℚ) * (3 / 5 : ℚ) * (3 / 4 : ℚ) = 45 →
  total_pens / pens_per_box = 20 :=
by
  sorry

#check lenny_pens_boxes

end lenny_pens_boxes_l694_69477


namespace spherical_to_rectangular_conversion_l694_69408

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 10
  let θ : ℝ := 3 * π / 4
  let φ : ℝ := π / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x = -5 ∧ y = 5 ∧ z = 5 * Real.sqrt 2) := by
  sorry

end spherical_to_rectangular_conversion_l694_69408


namespace root_implies_m_value_always_real_roots_l694_69494

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - m*x + m - 1

-- Theorem 1: If x = 3 is a root, then m = 4
theorem root_implies_m_value (m : ℝ) : quadratic m 3 = 0 → m = 4 := by sorry

-- Theorem 2: The quadratic equation always has real roots
theorem always_real_roots (m : ℝ) : ∃ x : ℝ, quadratic m x = 0 := by sorry

end root_implies_m_value_always_real_roots_l694_69494


namespace charlotte_theorem_l694_69462

/-- Represents the state of boxes after each step of adding marbles --/
def BoxState (n : ℕ) := ℕ → ℕ

/-- Initial state of boxes --/
def initial_state (n : ℕ) : BoxState n :=
  λ i => if i ≤ n ∧ i > 0 then i else 0

/-- Add a marble to each box --/
def add_to_all (state : BoxState n) : BoxState n :=
  λ i => state i + 1

/-- Add a marble to boxes divisible by k --/
def add_to_divisible (k : ℕ) (state : BoxState n) : BoxState n :=
  λ i => if state i % k = 0 then state i + 1 else state i

/-- Perform Charlotte's procedure --/
def charlotte_procedure (n : ℕ) : BoxState n :=
  let initial := initial_state n
  let after_first_step := add_to_all initial
  (List.range n).foldl (λ state k => add_to_divisible (k + 2) state) after_first_step

/-- Check if all boxes have exactly n+1 marbles --/
def all_boxes_have_n_plus_one (n : ℕ) (state : BoxState n) : Prop :=
  ∀ i, i > 0 → i ≤ n → state i = n + 1

/-- The main theorem --/
theorem charlotte_theorem (n : ℕ) :
  all_boxes_have_n_plus_one n (charlotte_procedure n) ↔ Nat.Prime (n + 1) :=
sorry

end charlotte_theorem_l694_69462


namespace age_difference_l694_69429

theorem age_difference (sachin_age rahul_age : ℕ) : 
  sachin_age = 63 →
  sachin_age * 9 = rahul_age * 7 →
  rahul_age - sachin_age = 18 := by
sorry

end age_difference_l694_69429


namespace select_three_from_fifteen_l694_69427

theorem select_three_from_fifteen (n k : ℕ) : n = 15 ∧ k = 3 → Nat.choose n k = 455 := by
  sorry

end select_three_from_fifteen_l694_69427


namespace triangle_area_zero_l694_69433

def point_a : ℝ × ℝ × ℝ := (2, 3, 1)
def point_b : ℝ × ℝ × ℝ := (8, 6, 4)
def point_c : ℝ × ℝ × ℝ := (14, 9, 7)

def triangle_area (a b c : ℝ × ℝ × ℝ) : ℝ := sorry

theorem triangle_area_zero :
  triangle_area point_a point_b point_c = 0 := by sorry

end triangle_area_zero_l694_69433


namespace total_squares_on_grid_l694_69447

/-- Represents a point on the 5x5 grid -/
structure GridPoint where
  x : Fin 5
  y : Fin 5

/-- Represents the set of 20 nails on the grid -/
def NailSet : Set GridPoint :=
  sorry

/-- Determines if four points form a square -/
def isSquare (p1 p2 p3 p4 : GridPoint) : Prop :=
  sorry

/-- Counts the number of squares that can be formed using the nails -/
def countSquares (nails : Set GridPoint) : Nat :=
  sorry

theorem total_squares_on_grid :
  countSquares NailSet = 21 :=
sorry

end total_squares_on_grid_l694_69447


namespace theater_ticket_difference_l694_69451

theorem theater_ticket_difference :
  ∀ (orchestra_price balcony_price : ℕ) 
    (total_tickets total_cost : ℕ) 
    (orchestra_tickets balcony_tickets : ℕ),
  orchestra_price = 18 →
  balcony_price = 12 →
  total_tickets = 450 →
  total_cost = 6300 →
  orchestra_tickets + balcony_tickets = total_tickets →
  orchestra_price * orchestra_tickets + balcony_price * balcony_tickets = total_cost →
  balcony_tickets - orchestra_tickets = 150 := by
sorry

end theater_ticket_difference_l694_69451


namespace dog_reachable_area_theorem_l694_69472

/-- The area outside a regular hexagonal doghouse that a dog can reach when tethered to a vertex -/
def dogReachableArea (side_length : Real) (rope_length : Real) : Real :=
  -- Definition to be filled
  sorry

/-- Theorem stating the area the dog can reach outside the doghouse -/
theorem dog_reachable_area_theorem :
  dogReachableArea 1 4 = (82 * Real.pi) / 3 := by
  sorry

end dog_reachable_area_theorem_l694_69472


namespace bags_difference_l694_69446

def bags_on_monday : ℕ := 7
def bags_on_next_day : ℕ := 12

theorem bags_difference : bags_on_next_day - bags_on_monday = 5 := by
  sorry

end bags_difference_l694_69446


namespace no_preimage_iff_k_less_than_neg_two_l694_69443

/-- The function f: ℝ → ℝ defined by f(x) = x² - 2x - 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 1

/-- Theorem stating that there is no real solution to f(x) = k if and only if k < -2 -/
theorem no_preimage_iff_k_less_than_neg_two :
  ∀ k : ℝ, (¬∃ x : ℝ, f x = k) ↔ k < -2 := by sorry

end no_preimage_iff_k_less_than_neg_two_l694_69443


namespace hyperbola_equation_l694_69497

/-- Represents a hyperbola -/
structure Hyperbola where
  center : ℝ × ℝ
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  point : ℝ × ℝ

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : (ℝ → ℝ → Prop) :=
  fun x y => y^2 / 20 - x^2 / 16 = 1

/-- Theorem: Given a hyperbola with center at (0, 0), foci at (0, -6) and (0, 6),
    and passing through the point (2, -5), its standard equation is y^2/20 - x^2/16 = 1 -/
theorem hyperbola_equation (h : Hyperbola) 
    (h_center : h.center = (0, 0))
    (h_foci : h.foci = ((0, -6), (0, 6)))
    (h_point : h.point = (2, -5)) :
    standard_equation h = fun x y => y^2 / 20 - x^2 / 16 = 1 := by
  sorry

end hyperbola_equation_l694_69497


namespace divisor_difference_greater_than_sqrt_l694_69460

theorem divisor_difference_greater_than_sqrt (A B : ℕ) 
  (h1 : A > 1) 
  (h2 : B ∣ A^2 + 1) 
  (h3 : B > A) : 
  B - A > Real.sqrt A :=
sorry

end divisor_difference_greater_than_sqrt_l694_69460


namespace A_C_mutually_exclusive_not_complementary_l694_69419

-- Define the event space
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set Nat := {1, 3}
def B : Set Nat := {3, 5}
def C : Set Nat := {2, 4, 6}

-- Define mutually exclusive events
def mutually_exclusive (X Y : Set Nat) : Prop := X ∩ Y = ∅

-- Define complementary events
def complementary (X Y : Set Nat) : Prop := X ∪ Y = Ω ∧ X ∩ Y = ∅

-- Theorem to prove
theorem A_C_mutually_exclusive_not_complementary :
  mutually_exclusive A C ∧ ¬complementary A C :=
sorry

end A_C_mutually_exclusive_not_complementary_l694_69419


namespace sum_a_c_l694_69465

theorem sum_a_c (a b c d : ℝ) 
  (h1 : a * b + a * c + b * c + b * d + c * d + a * d = 40)
  (h2 : b^2 + d^2 = 29) : 
  a + c = 42 / 5 := by
sorry

end sum_a_c_l694_69465


namespace andreas_living_room_area_l694_69416

theorem andreas_living_room_area :
  ∀ (room_area carpet_area : ℝ),
    carpet_area = 6 * 12 →
    room_area * 0.2 = carpet_area →
    room_area = 360 :=
by
  sorry

end andreas_living_room_area_l694_69416


namespace clothing_percentage_proof_l694_69480

theorem clothing_percentage_proof (food_percent : ℝ) (other_percent : ℝ) 
  (clothing_tax_rate : ℝ) (food_tax_rate : ℝ) (other_tax_rate : ℝ) 
  (total_tax_rate : ℝ) :
  food_percent = 20 →
  other_percent = 30 →
  clothing_tax_rate = 4 →
  food_tax_rate = 0 →
  other_tax_rate = 8 →
  total_tax_rate = 4.4 →
  (100 - food_percent - other_percent) * clothing_tax_rate / 100 + 
    food_percent * food_tax_rate / 100 + 
    other_percent * other_tax_rate / 100 = total_tax_rate →
  100 - food_percent - other_percent = 50 := by
sorry

end clothing_percentage_proof_l694_69480


namespace area_triangle_ABC_l694_69402

/-- Regular octagon with side length 3 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Triangle ABC in the regular octagon -/
def triangle_ABC (octagon : RegularOctagon) : Set (ℝ × ℝ) :=
  sorry

/-- Area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem: Area of triangle ABC in a regular octagon with side length 3 -/
theorem area_triangle_ABC (octagon : RegularOctagon) :
  area (triangle_ABC octagon) = 9 * (2 + Real.sqrt 2) / 4 :=
sorry

end area_triangle_ABC_l694_69402


namespace painting_time_equation_l694_69418

/-- The time it takes Sarah to paint the room alone (in hours) -/
def sarah_time : ℝ := 4

/-- The time it takes Tom to paint the room alone (in hours) -/
def tom_time : ℝ := 6

/-- The duration of the break (in hours) -/
def break_time : ℝ := 2

/-- The total time it takes Sarah and Tom to paint the room together, including the break (in hours) -/
noncomputable def total_time : ℝ := sorry

/-- Theorem stating the equation that the total time satisfies -/
theorem painting_time_equation :
  (1 / sarah_time + 1 / tom_time) * (total_time - break_time) = 1 := by sorry

end painting_time_equation_l694_69418


namespace ceiling_equality_abs_diff_l694_69435

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- State the theorem
theorem ceiling_equality_abs_diff (x y : ℝ) :
  (∀ x y, ceiling x = ceiling y → |x - y| < 1) ∧
  (∃ x y, |x - y| < 1 ∧ ceiling x ≠ ceiling y) :=
by sorry

end ceiling_equality_abs_diff_l694_69435


namespace farmer_plots_allocation_l694_69486

theorem farmer_plots_allocation (x y : ℕ) (h : x ≠ y) : ∃ (a b : ℕ), a^2 + b^2 = 2 * (x^2 + y^2) := by
  sorry

end farmer_plots_allocation_l694_69486


namespace recipe_total_cups_l694_69413

/-- Represents the ratio of ingredients in the recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients given a ratio and the amount of flour -/
def totalCups (ratio : RecipeRatio) (flourCups : ℕ) : ℕ :=
  let partSize := flourCups / ratio.flour
  partSize * (ratio.butter + ratio.flour + ratio.sugar)

/-- Theorem stating that for the given ratio and flour amount, the total cups is 30 -/
theorem recipe_total_cups :
  let ratio : RecipeRatio := ⟨2, 5, 3⟩
  let flourCups : ℕ := 15
  totalCups ratio flourCups = 30 := by
  sorry

end recipe_total_cups_l694_69413


namespace intersection_x_sum_zero_l694_69469

theorem intersection_x_sum_zero (x₁ x₂ : ℝ) : 
  x₁^2 + 9^2 = 169 → x₂^2 + 9^2 = 169 → x₁ + x₂ = 0 := by
  sorry

end intersection_x_sum_zero_l694_69469


namespace arithmetic_sequence_sum_l694_69487

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a_1 + a_5 = 6,
    prove that the sum of the first five terms is 15. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 1 + a 5 = 6) : 
  a 1 + a 2 + a 3 + a 4 + a 5 = 15 := by
  sorry

end arithmetic_sequence_sum_l694_69487


namespace second_number_is_37_l694_69410

theorem second_number_is_37 (a b c d : ℕ) : 
  a + b + c + d = 260 →
  a = 2 * b →
  c = a / 3 →
  d = 2 * (b + c) →
  b = 37 := by
sorry

end second_number_is_37_l694_69410


namespace function_identity_l694_69482

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f (x + 1) + y - 1) = f x + y) : 
  ∀ x : ℝ, f x = x := by
sorry

end function_identity_l694_69482


namespace min_value_of_expression_l694_69428

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_parallel : (1 : ℝ) / 2 = (x - 2) / (-6 * y)) :
  (3 / x + 1 / y) ≥ 6 := by
  sorry

end min_value_of_expression_l694_69428


namespace job_application_ratio_l694_69478

theorem job_application_ratio (total_applications in_state_applications : ℕ) 
  (h1 : total_applications = 600)
  (h2 : in_state_applications = 200) :
  (total_applications - in_state_applications) / in_state_applications = 2 := by
sorry

end job_application_ratio_l694_69478


namespace kimberly_skittles_l694_69439

/-- The number of Skittles Kimberly initially had -/
def initial_skittles : ℕ := 5

/-- The number of Skittles Kimberly bought -/
def bought_skittles : ℕ := 7

/-- The total number of Skittles Kimberly has after buying more -/
def total_skittles : ℕ := 12

/-- Theorem stating that the initial number of Skittles plus the bought Skittles equals the total Skittles -/
theorem kimberly_skittles : initial_skittles + bought_skittles = total_skittles := by
  sorry

end kimberly_skittles_l694_69439


namespace arithmetic_mean_problem_l694_69473

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 12 + 23 + 17 + y) / 5 = 15 → y = 15 := by
sorry

end arithmetic_mean_problem_l694_69473


namespace natural_equation_example_natural_equation_condition_l694_69400

-- Definition of a natural equation
def is_natural_equation (a b c : ℤ) : Prop :=
  ∃ x₁ x₂ : ℤ, (a * x₁^2 + b * x₁ + c = 0) ∧ 
              (a * x₂^2 + b * x₂ + c = 0) ∧ 
              (abs (x₁ - x₂) = 1) ∧
              (a ≠ 0)

-- Theorem 1: x² + 3x + 2 = 0 is a natural equation
theorem natural_equation_example : is_natural_equation 1 3 2 := by
  sorry

-- Theorem 2: x² - (m+1)x + m = 0 is a natural equation iff m = 0 or m = 2
theorem natural_equation_condition (m : ℤ) : 
  is_natural_equation 1 (-(m+1)) m ↔ m = 0 ∨ m = 2 := by
  sorry

end natural_equation_example_natural_equation_condition_l694_69400


namespace sum_of_fourth_powers_squared_l694_69405

theorem sum_of_fourth_powers_squared (A B C : ℤ) (h : A + B + C = 0) :
  2 * (A^4 + B^4 + C^4) = (A^2 + B^2 + C^2)^2 := by
  sorry

end sum_of_fourth_powers_squared_l694_69405


namespace tan_138_less_than_tan_143_l694_69463

theorem tan_138_less_than_tan_143 :
  let angle1 : Real := 138 * π / 180
  let angle2 : Real := 143 * π / 180
  (π / 2 < angle1 ∧ angle1 < π) →
  (π / 2 < angle2 ∧ angle2 < π) →
  (∀ x y, π / 2 < x ∧ x < y ∧ y < π → Real.tan x > Real.tan y) →
  Real.tan angle1 < Real.tan angle2 :=
by
  sorry

end tan_138_less_than_tan_143_l694_69463


namespace geometric_sequence_properties_l694_69412

/-- A geometric sequence with the given first four terms -/
def geometric_sequence : Fin 5 → ℝ
  | 0 => 10
  | 1 => -15
  | 2 => 22.5
  | 3 => -33.75
  | 4 => 50.625

/-- The common ratio of the geometric sequence -/
def common_ratio : ℝ := -1.5

theorem geometric_sequence_properties :
  (∀ n : Fin 3, geometric_sequence (n + 1) = geometric_sequence n * common_ratio) ∧
  geometric_sequence 4 = geometric_sequence 3 * common_ratio :=
by sorry

end geometric_sequence_properties_l694_69412


namespace equation_solution_l694_69479

theorem equation_solution : 
  ∃! x : ℚ, (3 - 2*x) / (x + 2) + (3*x - 6) / (3 - 2*x) = 2 ∧ x = -3/5 := by
sorry

end equation_solution_l694_69479


namespace domain_of_f_l694_69436

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, f (2 * x - 3) ≠ 0 → -2 ≤ x ∧ x ≤ 2) →
  (∀ y, f y ≠ 0 → -7 ≤ y ∧ y ≤ 1) :=
sorry

end domain_of_f_l694_69436


namespace lucas_speed_equals_miguel_speed_l694_69415

/-- Given the relative speeds of Miguel, Sophie, and Lucas, prove that Lucas's speed equals Miguel's speed. -/
theorem lucas_speed_equals_miguel_speed (miguel_speed : ℝ) (sophie_speed : ℝ) (lucas_speed : ℝ)
  (h1 : miguel_speed = 6)
  (h2 : sophie_speed = 3/4 * miguel_speed)
  (h3 : lucas_speed = 4/3 * sophie_speed) :
  lucas_speed = miguel_speed :=
by sorry

end lucas_speed_equals_miguel_speed_l694_69415


namespace paper_I_maximum_mark_l694_69425

/-- The maximum mark for Paper I -/
def maximum_mark : ℝ := 150

/-- The passing percentage for Paper I -/
def passing_percentage : ℝ := 0.40

/-- The marks secured by the candidate -/
def secured_marks : ℝ := 40

/-- The marks by which the candidate failed -/
def failing_margin : ℝ := 20

/-- Theorem stating that the maximum mark for Paper I is 150 -/
theorem paper_I_maximum_mark :
  (passing_percentage * maximum_mark = secured_marks + failing_margin) ∧
  (maximum_mark = 150) := by
  sorry

end paper_I_maximum_mark_l694_69425


namespace magnitude_of_complex_number_l694_69445

theorem magnitude_of_complex_number (i : ℂ) : i^2 = -1 → Complex.abs ((1 + i) - 2 / i) = Real.sqrt 10 := by
  sorry

end magnitude_of_complex_number_l694_69445


namespace apple_distribution_l694_69498

theorem apple_distribution (martha_initial : ℕ) (jane_apples : ℕ) (martha_final : ℕ) (martha_remaining : ℕ) :
  martha_initial = 20 →
  jane_apples = 5 →
  martha_remaining = 4 →
  martha_final = martha_remaining + 4 →
  martha_initial - jane_apples - martha_final = jane_apples + 2 :=
by
  sorry

end apple_distribution_l694_69498


namespace exist_common_members_l694_69414

/-- A structure representing a parliament with committees -/
structure Parliament :=
  (members : Finset ℕ)
  (committees : Finset (Finset ℕ))
  (h_member_count : members.card = 1600)
  (h_committee_count : committees.card = 16000)
  (h_committee_size : ∀ c ∈ committees, c.card = 80)
  (h_committees_subset : ∀ c ∈ committees, c ⊆ members)

/-- Theorem stating that there exist at least two committees with at least 4 common members -/
theorem exist_common_members (p : Parliament) :
  ∃ c1 c2 : Finset ℕ, c1 ∈ p.committees ∧ c2 ∈ p.committees ∧ c1 ≠ c2 ∧ (c1 ∩ c2).card ≥ 4 :=
sorry

end exist_common_members_l694_69414


namespace like_terms_exponent_product_l694_69468

theorem like_terms_exponent_product (a b : ℤ) : 
  (6 = -2 * a) → (b = 2) → a * b = -6 := by
  sorry

end like_terms_exponent_product_l694_69468


namespace ratio_problem_l694_69448

theorem ratio_problem (a b x m : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  a / b = 4 / 5 ∧
  x = a * (1 + 0.25) ∧
  m = b * (1 - 0.80) →
  m / x = 1 / 5 := by
sorry

end ratio_problem_l694_69448


namespace jeffrey_steps_l694_69441

-- Define Jeffrey's walking pattern
def forward_steps : ℕ := 3
def backward_steps : ℕ := 2

-- Define the distance between house and mailbox
def distance : ℕ := 66

-- Define the function to calculate total steps
def total_steps (fwd : ℕ) (bwd : ℕ) (dist : ℕ) : ℕ :=
  dist * (fwd + bwd)

-- Theorem statement
theorem jeffrey_steps :
  total_steps forward_steps backward_steps distance = 330 := by
  sorry

end jeffrey_steps_l694_69441


namespace function_inequality_l694_69450

open Real

noncomputable def f (x : ℝ) : ℝ := x / cos x

theorem function_inequality (x₁ x₂ x₃ : ℝ) 
  (h₁ : |x₁| < π/2) (h₂ : |x₂| < π/2) (h₃ : |x₃| < π/2)
  (h₄ : f x₁ + f x₂ ≥ 0) (h₅ : f x₂ + f x₃ ≥ 0) (h₆ : f x₃ + f x₁ ≥ 0) :
  f (x₁ + x₂ + x₃) ≥ 0 := by
  sorry

end function_inequality_l694_69450


namespace k_mod_8_l694_69475

/-- An integer m covers 1998 if 1, 9, 9, 8 appear in this order as digits of m. -/
def covers_1998 (m : ℕ) : Prop := sorry

/-- k(n) is the number of positive integers that cover 1998 and have exactly n digits, all different from 0. -/
def k (n : ℕ) : ℕ := sorry

/-- The main theorem: k(n) is congruent to 1 modulo 8 for all n ≥ 5. -/
theorem k_mod_8 (n : ℕ) (h : n ≥ 5) : k n ≡ 1 [MOD 8] := by sorry

end k_mod_8_l694_69475


namespace range_of_a_l694_69464

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0) → 
  a ≥ 0 :=
by sorry

end range_of_a_l694_69464


namespace count_six_digit_integers_l694_69470

/-- The number of different positive six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, 8 -/
def sixDigitIntegersCount : ℕ := 60

/-- The multiset of digits used to form the integers -/
def digits : Multiset ℕ := {1, 1, 3, 3, 3, 8}

theorem count_six_digit_integers : 
  (Multiset.card digits = 6) → 
  (Multiset.count 1 digits = 2) → 
  (Multiset.count 3 digits = 3) → 
  (Multiset.count 8 digits = 1) → 
  sixDigitIntegersCount = 60 := by sorry

end count_six_digit_integers_l694_69470


namespace list_fraction_problem_l694_69417

theorem list_fraction_problem (l : List ℝ) (n : ℝ) (h1 : l.length = 21) 
  (h2 : n ∈ l) (h3 : n = 4 * ((l.sum - n) / 20)) : 
  n = (1 / 6) * l.sum :=
sorry

end list_fraction_problem_l694_69417


namespace greatest_divisor_with_remainders_l694_69461

theorem greatest_divisor_with_remainders : 
  let a := 690
  let b := 875
  let r₁ := 10
  let r₂ := 25
  Int.gcd (a - r₁) (b - r₂) = 170 :=
by sorry

end greatest_divisor_with_remainders_l694_69461


namespace full_time_more_than_three_years_l694_69453

/-- Represents the percentage of associates in each category -/
structure AssociatePercentages where
  secondYear : ℝ
  thirdYear : ℝ
  notFirstYear : ℝ
  partTime : ℝ
  partTimeMoreThanTwoYears : ℝ

/-- Theorem stating the percentage of full-time associates at the firm for more than three years -/
theorem full_time_more_than_three_years 
  (percentages : AssociatePercentages)
  (h1 : percentages.secondYear = 30)
  (h2 : percentages.thirdYear = 20)
  (h3 : percentages.notFirstYear = 60)
  (h4 : percentages.partTime = 10)
  (h5 : percentages.partTimeMoreThanTwoYears = percentages.partTime / 2)
  : ℝ := by
  sorry

#check full_time_more_than_three_years

end full_time_more_than_three_years_l694_69453


namespace stewart_farm_horse_food_l694_69471

/-- Proves that given the ratio of sheep to horses is 4:7, there are 32 sheep on the farm,
    and the farm needs a total of 12,880 ounces of horse food per day,
    each horse needs 230 ounces of horse food per day. -/
theorem stewart_farm_horse_food (sheep : ℕ) (horses : ℕ) (total_food : ℕ) :
  sheep = 32 →
  4 * horses = 7 * sheep →
  total_food = 12880 →
  total_food / horses = 230 := by
  sorry

end stewart_farm_horse_food_l694_69471


namespace symmetric_point_wrt_x_axis_l694_69420

/-- Given a point P(2, -5), its symmetric point P' with respect to the x-axis has coordinates (2, 5) -/
theorem symmetric_point_wrt_x_axis :
  let P : ℝ × ℝ := (2, -5)
  let P' : ℝ × ℝ := (2, 5)
  (∀ (x y : ℝ), (x, y) = P → (x, -y) = P') :=
by sorry

end symmetric_point_wrt_x_axis_l694_69420


namespace min_value_expression_l694_69409

theorem min_value_expression (x y : ℤ) (h : 4*x + 5*y = 7) :
  ∃ (m : ℤ), m = 1 ∧ ∀ (a b : ℤ), 4*a + 5*b = 7 → 5*|a| - 3*|b| ≥ m :=
sorry

end min_value_expression_l694_69409


namespace quadratic_inequality_condition_l694_69452

theorem quadratic_inequality_condition (m : ℝ) :
  (m > 1 → ∃ x : ℝ, x^2 + 2*m*x + 1 < 0) ∧
  (∃ m : ℝ, m ≤ 1 ∧ ∃ x : ℝ, x^2 + 2*m*x + 1 < 0) :=
by sorry

end quadratic_inequality_condition_l694_69452


namespace snowball_theorem_l694_69401

/-- Represents the action of throwing a snowball -/
def throws (n : Nat) (m : Nat) : Prop := sorry

/-- The number of children -/
def num_children : Nat := 43

theorem snowball_theorem :
  (∀ i : Nat, i > 0 ∧ i ≤ num_children → ∃! j : Nat, j > 0 ∧ j ≤ num_children ∧ throws i j) ∧
  (∀ j : Nat, j > 0 ∧ j ≤ num_children → ∃! i : Nat, i > 0 ∧ i ≤ num_children ∧ throws i j) ∧
  (∃ x : Nat, x > 0 ∧ x ≤ num_children ∧ throws 1 x ∧ throws x 2) ∧
  (∃ y : Nat, y > 0 ∧ y ≤ num_children ∧ throws 2 y ∧ throws y 3) ∧
  (∃ z : Nat, z > 0 ∧ z ≤ num_children ∧ throws num_children z ∧ throws z 1) →
  ∃ w : Nat, w = 24 ∧ throws w 3 := by sorry

end snowball_theorem_l694_69401


namespace crude_oil_mixture_l694_69492

/-- Given two sources of crude oil, prove that the second source contains 75% hydrocarbons -/
theorem crude_oil_mixture (
  source1_percent : ℝ)
  (source2_percent : ℝ)
  (final_volume : ℝ)
  (final_percent : ℝ)
  (source2_volume : ℝ) :
  source1_percent = 25 →
  final_volume = 50 →
  final_percent = 55 →
  source2_volume = 30 →
  source2_percent = 75 :=
by
  sorry

#check crude_oil_mixture

end crude_oil_mixture_l694_69492


namespace least_positive_angle_theta_l694_69466

/-- The least positive angle θ (in degrees) satisfying cos 10° = sin 15° + sin θ is 32.5° -/
theorem least_positive_angle_theta : 
  let θ : ℝ := 32.5
  ∀ φ : ℝ, 0 < φ ∧ φ < θ → Real.cos (10 * π / 180) ≠ Real.sin (15 * π / 180) + Real.sin (φ * π / 180) ∧
  Real.cos (10 * π / 180) = Real.sin (15 * π / 180) + Real.sin (θ * π / 180) := by
  sorry

end least_positive_angle_theta_l694_69466


namespace hyperbola_eccentricity_l694_69438

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0),
    if the distance from one focus to an asymptote is √5/3 * c,
    where c is the semi-focal length, then the eccentricity is 3/2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  (c * b / Real.sqrt (a^2 + b^2) = c * Real.sqrt 5 / 3) →
  c^2 = a^2 + b^2 →
  c / a = 3 / 2 := by
  sorry

end hyperbola_eccentricity_l694_69438


namespace prove_abc_equation_l694_69484

theorem prove_abc_equation (a b c : ℝ) 
  (h1 : a^4 * b^3 * c^5 = 18) 
  (h2 : a^3 * b^5 * c^4 = 8) : 
  a^5 * b * c^6 = 81/2 := by
  sorry

end prove_abc_equation_l694_69484


namespace min_distance_to_line_l694_69432

/-- The minimum value of (x-2)^2 + (y-2)^2 given that x - y - 1 = 0 -/
theorem min_distance_to_line : 
  (∃ (m : ℝ), ∀ (x y : ℝ), x - y - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≥ m) ∧ 
  (∃ (x y : ℝ), x - y - 1 = 0 ∧ (x - 2)^2 + (y - 2)^2 = 1/2) :=
sorry

end min_distance_to_line_l694_69432


namespace number_of_boys_l694_69493

theorem number_of_boys (total_amount : ℕ) (total_people : ℕ) (boy_amount : ℕ) (girl_amount : ℕ) 
  (h1 : total_amount = 460)
  (h2 : total_people = 41)
  (h3 : boy_amount = 12)
  (h4 : girl_amount = 8) :
  ∃ (boys : ℕ), boys = 33 ∧ 
  boys * boy_amount + (total_people - boys) * girl_amount = total_amount :=
by
  sorry

end number_of_boys_l694_69493


namespace polynomial_remainder_theorem_l694_69422

theorem polynomial_remainder_theorem (x : ℝ) : 
  (4 * x^3 - 10 * x^2 + 15 * x - 17) % (4 * x - 8) = 5 := by
  sorry

end polynomial_remainder_theorem_l694_69422


namespace weekly_distance_calculation_l694_69499

/-- Calculates the weekly running distance given the number of days, hours per day, and speed. -/
def weekly_running_distance (days_per_week : ℕ) (hours_per_day : ℝ) (speed_mph : ℝ) : ℝ :=
  days_per_week * hours_per_day * speed_mph

/-- Proves that running 5 days a week, 1.5 hours each day, at 8 mph results in 60 miles per week. -/
theorem weekly_distance_calculation :
  weekly_running_distance 5 1.5 8 = 60 := by
  sorry

end weekly_distance_calculation_l694_69499


namespace boat_round_trip_time_l694_69442

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water, 
    the stream's speed, and the distance to the destination. -/
theorem boat_round_trip_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (distance : ℝ) 
  (h1 : boat_speed = 8) 
  (h2 : stream_speed = 6) 
  (h3 : distance = 210) : 
  distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed) = 120 := by
  sorry

#check boat_round_trip_time

end boat_round_trip_time_l694_69442


namespace counterexample_exists_l694_69423

theorem counterexample_exists : ∃ n : ℕ, 
  (Even n) ∧ (¬ Prime n) ∧ (¬ Prime (n + 2)) := by
  sorry

end counterexample_exists_l694_69423


namespace root_sum_square_theorem_l694_69403

theorem root_sum_square_theorem (m n : ℝ) : 
  (m^2 + 2*m - 2025 = 0) → 
  (n^2 + 2*n - 2025 = 0) → 
  (m ≠ n) →
  (m^2 + 3*m + n = 2023) := by
sorry

end root_sum_square_theorem_l694_69403


namespace five_pages_thirty_lines_each_l694_69440

/-- Given a page capacity and number of pages, calculates the total lines of information. -/
def total_lines (lines_per_page : ℕ) (num_pages : ℕ) : ℕ :=
  lines_per_page * num_pages

/-- Theorem stating that 5 pages with 30 lines each result in 150 total lines. -/
theorem five_pages_thirty_lines_each :
  total_lines 30 5 = 150 := by
  sorry

end five_pages_thirty_lines_each_l694_69440
