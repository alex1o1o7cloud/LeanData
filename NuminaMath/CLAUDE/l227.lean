import Mathlib

namespace NUMINAMATH_CALUDE_johns_house_paintable_area_l227_22797

/-- Calculates the total paintable wall area in John's house -/
def total_paintable_area (num_bedrooms : ℕ) (length width height : ℝ) (non_paintable_area : ℝ) : ℝ :=
  num_bedrooms * (2 * (length * height + width * height) - non_paintable_area)

/-- Proves that the total paintable wall area in John's house is 1820 square feet -/
theorem johns_house_paintable_area :
  total_paintable_area 4 15 12 10 85 = 1820 := by
  sorry

#eval total_paintable_area 4 15 12 10 85

end NUMINAMATH_CALUDE_johns_house_paintable_area_l227_22797


namespace NUMINAMATH_CALUDE_combined_bus_ride_length_l227_22705

theorem combined_bus_ride_length 
  (vince_ride : ℝ) 
  (zachary_ride : ℝ) 
  (alexandra_ride : ℝ) 
  (h1 : vince_ride = 0.62) 
  (h2 : zachary_ride = 0.5) 
  (h3 : alexandra_ride = 0.72) : 
  vince_ride + zachary_ride + alexandra_ride = 1.84 := by
sorry

end NUMINAMATH_CALUDE_combined_bus_ride_length_l227_22705


namespace NUMINAMATH_CALUDE_f_is_even_l227_22792

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_is_even (g : ℝ → ℝ) (h : isEven g) :
  isEven (fun x ↦ |g (x^3)|) := by sorry

end NUMINAMATH_CALUDE_f_is_even_l227_22792


namespace NUMINAMATH_CALUDE_quadrilateral_side_length_l227_22710

/-- Given a quadrilateral ABCD with specific side lengths and angles, prove that AD = √7 -/
theorem quadrilateral_side_length (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let angle_ABC := Real.arccos ((AB^2 + BC^2 - (A.1 - C.1)^2 - (A.2 - C.2)^2) / (2 * AB * BC))
  let angle_BCD := Real.arccos ((BC^2 + CD^2 - (B.1 - D.1)^2 - (B.2 - D.2)^2) / (2 * BC * CD))
  AB = 1 ∧ BC = 2 ∧ CD = Real.sqrt 3 ∧ angle_ABC = 2 * Real.pi / 3 ∧ angle_BCD = Real.pi / 2 →
  AD = Real.sqrt 7 := by
sorry


end NUMINAMATH_CALUDE_quadrilateral_side_length_l227_22710


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l227_22784

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 258 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (eq1 : a + b = 17)
  (eq2 : a * b + c + d = 86)
  (eq3 : a * d + b * c = 180)
  (eq4 : c * d = 110) :
  a^2 + b^2 + c^2 + d^2 ≤ 258 ∧ ∃ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 = 258 ∧ 
    a + b = 17 ∧ a * b + c + d = 86 ∧ a * d + b * c = 180 ∧ c * d = 110 := by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_squares_l227_22784


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l227_22720

-- Problem 1
theorem simplify_expression_1 (x : ℝ) : (2*x + 1) * (2*x - 1) = 4*x^2 - 1 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x y : ℝ) : (x - 2*y)^2 - x*y = x^2 - 5*x*y + 4*y^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l227_22720


namespace NUMINAMATH_CALUDE_greeting_cards_group_size_l227_22712

theorem greeting_cards_group_size (n : ℕ) : 
  n * (n - 1) = 72 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_greeting_cards_group_size_l227_22712


namespace NUMINAMATH_CALUDE_intersection_condition_l227_22714

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = Real.sqrt (2 * p.1 - p.1^2)}
def N (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * (p.1 + 1)}

-- State the theorem
theorem intersection_condition (k : ℝ) :
  (∃ p, p ∈ M ∩ N k) ↔ 0 ≤ k ∧ k ≤ Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_condition_l227_22714


namespace NUMINAMATH_CALUDE_multiple_indecomposable_factorizations_l227_22796

/-- The set V_n for a given n -/
def V_n (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ+, m = 1 + k * n}

/-- A number is indecomposable in V_n if it cannot be expressed as the product of two members of V_n -/
def Indecomposable (m : ℕ) (n : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ a b : ℕ, a ∈ V_n n → b ∈ V_n n → a * b ≠ m

/-- There exists a number in V_n with multiple indecomposable factorizations -/
theorem multiple_indecomposable_factorizations (n : ℕ) (h : n > 2) :
  ∃ m : ℕ, m ∈ V_n n ∧
    ∃ (a b c d : ℕ),
      Indecomposable a n ∧ Indecomposable b n ∧ Indecomposable c n ∧ Indecomposable d n ∧
      a * b = m ∧ c * d = m ∧ (a ≠ c ∨ b ≠ d) :=
  sorry

end NUMINAMATH_CALUDE_multiple_indecomposable_factorizations_l227_22796


namespace NUMINAMATH_CALUDE_number_line_points_l227_22769

/-- Represents a point on a number line -/
structure Point where
  value : ℚ

/-- The distance between two points on a number line -/
def distance (p q : Point) : ℚ := q.value - p.value

theorem number_line_points (A B C : Point)
  (hA : A.value = 2)
  (hAB : distance A B = -7)
  (hBC : distance B C = 1 + 2/3) :
  B.value = -5 ∧ C.value = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_number_line_points_l227_22769


namespace NUMINAMATH_CALUDE_bhupathi_amount_l227_22732

theorem bhupathi_amount (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A + B = 1210) (h4 : (4/15) * A = (2/5) * B) : B = 484 :=
by
  sorry

end NUMINAMATH_CALUDE_bhupathi_amount_l227_22732


namespace NUMINAMATH_CALUDE_fangfang_floor_climb_l227_22733

def time_between_floors (start_floor end_floor : ℕ) (time : ℝ) : Prop :=
  time = (end_floor - start_floor) * 15

theorem fangfang_floor_climb : 
  time_between_floors 1 3 30 → time_between_floors 2 6 60 :=
by
  sorry

end NUMINAMATH_CALUDE_fangfang_floor_climb_l227_22733


namespace NUMINAMATH_CALUDE_smallest_x_for_cube_1680x_l227_22722

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m^3

theorem smallest_x_for_cube_1680x : 
  (∀ x : ℕ, x > 0 ∧ x < 44100 → ¬ is_perfect_cube (1680 * x)) ∧ 
  is_perfect_cube (1680 * 44100) := by
sorry

end NUMINAMATH_CALUDE_smallest_x_for_cube_1680x_l227_22722


namespace NUMINAMATH_CALUDE_soccer_handshakes_l227_22798

/-- Calculates the total number of handshakes in a soccer match -/
theorem soccer_handshakes (team_size : ℕ) (num_teams : ℕ) (num_referees : ℕ) : 
  team_size = 11 → num_teams = 2 → num_referees = 3 →
  (team_size * team_size * (num_teams - 1) / 2) + (team_size * num_teams * num_referees) = 187 := by
  sorry

end NUMINAMATH_CALUDE_soccer_handshakes_l227_22798


namespace NUMINAMATH_CALUDE_opponent_scissors_is_random_event_l227_22753

/-- Represents the possible choices in the game of rock, paper, scissors -/
inductive Choice
  | Rock
  | Paper
  | Scissors

/-- Represents a game of rock, paper, scissors -/
structure RockPaperScissors where
  opponentChoice : Choice

/-- Defines what it means for an event to be random in this context -/
def isRandomEvent (game : RockPaperScissors → Prop) : Prop :=
  ∀ (c : Choice), ∃ (g : RockPaperScissors), g.opponentChoice = c ∧ game g

/-- The main theorem: opponent choosing scissors is a random event -/
theorem opponent_scissors_is_random_event :
  isRandomEvent (λ g => g.opponentChoice = Choice.Scissors) :=
sorry

end NUMINAMATH_CALUDE_opponent_scissors_is_random_event_l227_22753


namespace NUMINAMATH_CALUDE_domain_of_f_l227_22709

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -3 ∧ x ≠ -2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_f_l227_22709


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l227_22795

theorem triangle_angle_measure (a b c : ℝ) : 
  a + b + c = 180 →  -- sum of angles in a triangle is 180°
  a = 40 →           -- one angle is 40°
  b = 2 * c →        -- one angle is twice the other
  c = 140 / 3 :=     -- prove that the third angle is 140/3°
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l227_22795


namespace NUMINAMATH_CALUDE_first_player_winning_strategy_exists_l227_22704

/-- Represents the state of the chocolate bar game -/
structure ChocolateGame where
  rows : Nat
  cols : Nat

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- The result of the game -/
structure GameResult where
  firstPlayerPieces : Nat
  secondPlayerPieces : Nat

/-- The strategy function type -/
def Strategy := ChocolateGame → Player → Option (Nat × Nat)

/-- Simulates the game given strategies for both players -/
def playGame (firstStrategy : Strategy) (secondStrategy : Strategy) : GameResult :=
  sorry

/-- The main theorem stating the existence of a winning strategy for the first player -/
theorem first_player_winning_strategy_exists :
  ∃ (strategy : Strategy),
    let result := playGame strategy (λ _ _ ↦ none)
    result.firstPlayerPieces ≥ result.secondPlayerPieces + 6 := by
  sorry

end NUMINAMATH_CALUDE_first_player_winning_strategy_exists_l227_22704


namespace NUMINAMATH_CALUDE_water_bottle_volume_l227_22744

theorem water_bottle_volume (total_cost : ℝ) (num_bottles : ℕ) (price_per_liter : ℝ) 
  (h1 : total_cost = 12)
  (h2 : num_bottles = 6)
  (h3 : price_per_liter = 1) :
  (total_cost / (num_bottles : ℝ)) / price_per_liter = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_bottle_volume_l227_22744


namespace NUMINAMATH_CALUDE_vectors_are_orthogonal_l227_22771

def vector1 : Fin 4 → ℝ := ![2, -4, 3, 1]
def vector2 : Fin 4 → ℝ := ![-3, 1, 4, -2]

theorem vectors_are_orthogonal :
  (Finset.sum Finset.univ (λ i => vector1 i * vector2 i)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_are_orthogonal_l227_22771


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l227_22740

/-- The greatest distance between centers of two circles in a rectangle --/
theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ)
  (rectangle_height : ℝ)
  (circle_diameter : ℝ)
  (h_width : rectangle_width = 18)
  (h_height : rectangle_height = 15)
  (h_diameter : circle_diameter = 7)
  (h_fit : circle_diameter ≤ min rectangle_width rectangle_height) :
  ∃ (d : ℝ), d = Real.sqrt 185 ∧
    ∀ (d' : ℝ), d' ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l227_22740


namespace NUMINAMATH_CALUDE_triangle_right_angled_l227_22728

theorem triangle_right_angled (α β γ : Real) : 
  0 < α ∧ 0 < β ∧ 0 < γ →  -- Angles are positive
  α + β + γ = Real.pi →    -- Sum of angles in a triangle
  (Real.sin α + Real.sin β) / (Real.cos α + Real.cos β) = Real.sin γ →
  γ = Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_right_angled_l227_22728


namespace NUMINAMATH_CALUDE_cylinder_height_ratio_l227_22793

/-- Given a cylinder whose radius is tripled and whose new volume is 18 times the original,
    prove that the ratio of the new height to the original height is 2:1. -/
theorem cylinder_height_ratio 
  (r : ℝ) -- original radius
  (h : ℝ) -- original height
  (h' : ℝ) -- new height
  (volume_ratio : ℝ) -- ratio of new volume to old volume
  (h_pos : 0 < h) -- ensure original height is positive
  (r_pos : 0 < r) -- ensure original radius is positive
  (volume_eq : π * (3 * r)^2 * h' = volume_ratio * (π * r^2 * h)) -- volume equation
  (volume_ratio_eq : volume_ratio = 18) -- new volume is 18 times the old one
  : h' / h = 2 := by
sorry

end NUMINAMATH_CALUDE_cylinder_height_ratio_l227_22793


namespace NUMINAMATH_CALUDE_distribute_toys_count_l227_22715

/-- The number of ways to distribute 4 toys out of 6 distinct toys to 4 distinct people -/
def distribute_toys : ℕ :=
  Nat.factorial 6 / Nat.factorial 2

/-- Theorem stating that distributing 4 toys out of 6 distinct toys to 4 distinct people results in 360 different arrangements -/
theorem distribute_toys_count : distribute_toys = 360 := by
  sorry

end NUMINAMATH_CALUDE_distribute_toys_count_l227_22715


namespace NUMINAMATH_CALUDE_inequality_solution_set_l227_22730

theorem inequality_solution_set (a : ℝ) : 
  (∃ x : ℝ, |x - 1| + |x + 2| < a) → a > 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l227_22730


namespace NUMINAMATH_CALUDE_sams_remaining_pennies_l227_22741

/-- Given an initial number of pennies and a number of spent pennies,
    calculate the remaining number of pennies. -/
def remaining_pennies (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem stating that Sam's remaining pennies are 5 given the initial and spent amounts. -/
theorem sams_remaining_pennies :
  remaining_pennies 98 93 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_pennies_l227_22741


namespace NUMINAMATH_CALUDE_not_domain_zero_to_three_l227_22773

/-- The function f(x) = x^2 - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The theorem stating that [0, 3] cannot be the domain of f(x) given its value range is [1, 2] -/
theorem not_domain_zero_to_three :
  (∀ y ∈ Set.Icc 1 2, ∃ x, f x = y) →
  ¬(∀ x ∈ Set.Icc 0 3, f x ∈ Set.Icc 1 2) :=
by sorry

end NUMINAMATH_CALUDE_not_domain_zero_to_three_l227_22773


namespace NUMINAMATH_CALUDE_intersection_condition_l227_22750

-- Define the parabola C: y^2 = x
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the line l: y = kx + 1
def line (k x y : ℝ) : Prop := y = k * x + 1

-- Define the condition for two distinct intersection points
def has_two_distinct_intersections (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    parabola x₁ y₁ ∧ line k x₁ y₁ ∧
    parabola x₂ y₂ ∧ line k x₂ y₂

-- Theorem statement
theorem intersection_condition :
  (∀ k : ℝ, has_two_distinct_intersections k → k ≠ 0) ∧
  (∃ k : ℝ, k ≠ 0 ∧ ¬has_two_distinct_intersections k) :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l227_22750


namespace NUMINAMATH_CALUDE_at_least_one_perpendicular_l227_22785

structure GeometricSpace where
  Plane : Type
  Line : Type
  Point : Type

variable {G : GeometricSpace}

-- Define the necessary relations
def perpendicular (α β : G.Plane) : Prop := sorry
def contains (α : G.Plane) (l : G.Line) : Prop := sorry
def perpendicular_lines (l₁ l₂ : G.Line) : Prop := sorry
def perpendicular_line_plane (l : G.Line) (α : G.Plane) : Prop := sorry

-- State the theorem
theorem at_least_one_perpendicular
  (α β : G.Plane) (n m : G.Line)
  (h1 : perpendicular α β)
  (h2 : contains α n)
  (h3 : contains β m)
  (h4 : perpendicular_lines m n) :
  perpendicular_line_plane n β ∨ perpendicular_line_plane m α :=
sorry

end NUMINAMATH_CALUDE_at_least_one_perpendicular_l227_22785


namespace NUMINAMATH_CALUDE_constant_function_proof_l227_22782

theorem constant_function_proof (f : ℚ → ℚ) 
  (h1 : ∀ x y : ℚ, f (x + y) = f x + f y + 2547)
  (h2 : f 2004 = 2547) : 
  f 2547 = 2547 := by sorry

end NUMINAMATH_CALUDE_constant_function_proof_l227_22782


namespace NUMINAMATH_CALUDE_factorization_problem1_l227_22779

theorem factorization_problem1 (a b : ℝ) :
  4 * a^2 + 12 * a * b + 9 * b^2 = (2*a + 3*b)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem1_l227_22779


namespace NUMINAMATH_CALUDE_power_of_negative_square_l227_22770

theorem power_of_negative_square (a : ℝ) : (-a^2)^3 = -a^6 := by sorry

end NUMINAMATH_CALUDE_power_of_negative_square_l227_22770


namespace NUMINAMATH_CALUDE_empty_box_weight_l227_22766

-- Define the number of balls
def num_balls : ℕ := 30

-- Define the weight of each ball in kg
def ball_weight : ℝ := 0.36

-- Define the total weight of the box with balls in kg
def total_weight : ℝ := 11.26

-- Theorem to prove
theorem empty_box_weight :
  total_weight - (num_balls : ℝ) * ball_weight = 0.46 := by
  sorry

end NUMINAMATH_CALUDE_empty_box_weight_l227_22766


namespace NUMINAMATH_CALUDE_fraction_value_at_three_l227_22726

theorem fraction_value_at_three : 
  let x : ℝ := 3
  (x^12 + 18*x^6 + 81) / (x^6 + 9) = 738 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_at_three_l227_22726


namespace NUMINAMATH_CALUDE_correct_substitution_l227_22721

theorem correct_substitution (x y : ℝ) :
  (5 * x + 3 * y = 22) ∧ (y = x - 2) →
  5 * x + 3 * (x - 2) = 22 :=
by sorry

end NUMINAMATH_CALUDE_correct_substitution_l227_22721


namespace NUMINAMATH_CALUDE_smallest_multiple_l227_22703

theorem smallest_multiple (x : ℕ) : x = 256 ↔ 
  (x > 0 ∧ 900 * x % 1024 = 0 ∧ ∀ y : ℕ, 0 < y ∧ y < x → 900 * y % 1024 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l227_22703


namespace NUMINAMATH_CALUDE_volleyball_scoring_l227_22759

/-- Volleyball scoring problem -/
theorem volleyball_scoring (L : ℕ) : 
  (∃ (N A : ℕ),
    N = L + 3 ∧ 
    A = 2 * (L + N) ∧ 
    L + N + A + 17 = 50) → 
  L = 6 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_scoring_l227_22759


namespace NUMINAMATH_CALUDE_gcf_of_1260_and_1440_l227_22777

theorem gcf_of_1260_and_1440 : Nat.gcd 1260 1440 = 180 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_1260_and_1440_l227_22777


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l227_22764

theorem least_k_for_inequality : ∃ k : ℤ, (∀ j : ℤ, 0.00010101 * (10 : ℝ)^j > 10 → k ≤ j) ∧ 0.00010101 * (10 : ℝ)^k > 10 ∧ k = 6 :=
by sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l227_22764


namespace NUMINAMATH_CALUDE_expression_evaluation_l227_22774

theorem expression_evaluation :
  let a : ℝ := 3 + Real.sqrt 5
  let b : ℝ := 3 - Real.sqrt 5
  ((a^2 - 2*a*b + b^2) / (a^2 - b^2)) * ((a*b) / (a - b)) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l227_22774


namespace NUMINAMATH_CALUDE_analytical_method_is_effect_to_cause_l227_22723

/-- Represents the possible descriptions of the analytical method -/
inductive AnalyticalMethodDescription
  | causeToEffect
  | effectToCause
  | mutualInference
  | converseProof

/-- Definition of the analytical method -/
structure AnalyticalMethod :=
  (description : AnalyticalMethodDescription)
  (isReasoningMethod : Bool)

/-- Theorem stating that the analytical method is correctly described as reasoning from effect to cause -/
theorem analytical_method_is_effect_to_cause :
  ∀ (am : AnalyticalMethod), 
    am.isReasoningMethod = true → 
    am.description = AnalyticalMethodDescription.effectToCause :=
by
  sorry

end NUMINAMATH_CALUDE_analytical_method_is_effect_to_cause_l227_22723


namespace NUMINAMATH_CALUDE_avg_people_per_hour_rounded_l227_22778

/-- The number of people moving to Texas in five days -/
def total_people : ℕ := 5000

/-- The number of days -/
def num_days : ℕ := 5

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Calculates the average number of people moving to Texas per hour -/
def avg_people_per_hour : ℚ :=
  total_people / (num_days * hours_per_day)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem avg_people_per_hour_rounded :
  round_to_nearest avg_people_per_hour = 42 := by
  sorry

end NUMINAMATH_CALUDE_avg_people_per_hour_rounded_l227_22778


namespace NUMINAMATH_CALUDE_treewidth_iff_bramble_order_l227_22768

/-- A graph represented by its vertex set and edge relation -/
structure Graph (V : Type) :=
  (edge : V → V → Prop)

/-- The treewidth of a graph -/
def treewidth {V : Type} (G : Graph V) : ℕ := sorry

/-- A bramble in a graph -/
def Bramble {V : Type} (G : Graph V) := Set (Set V)

/-- The order of a bramble -/
def brambleOrder {V : Type} (G : Graph V) (B : Bramble G) : ℕ := sorry

/-- Main theorem: A graph has treewidth ≥ k iff it contains a bramble of order > k -/
theorem treewidth_iff_bramble_order {V : Type} (G : Graph V) (k : ℕ) :
  treewidth G ≥ k ↔ ∃ (B : Bramble G), brambleOrder G B > k := by
  sorry

end NUMINAMATH_CALUDE_treewidth_iff_bramble_order_l227_22768


namespace NUMINAMATH_CALUDE_sum_inequality_l227_22735

theorem sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - 2*a) / (a^2 + b*c) + 
  (c + a - 2*b) / (b^2 + c*a) + 
  (a + b - 2*c) / (c^2 + a*b) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l227_22735


namespace NUMINAMATH_CALUDE_milburg_population_l227_22725

/-- The total population of Milburg -/
def total_population (adults children teenagers seniors : ℕ) : ℕ :=
  adults + children + teenagers + seniors

/-- Theorem: The total population of Milburg is 12,292 -/
theorem milburg_population : total_population 5256 2987 1709 2340 = 12292 := by
  sorry

end NUMINAMATH_CALUDE_milburg_population_l227_22725


namespace NUMINAMATH_CALUDE_factorization_example_l227_22711

theorem factorization_example : ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_example_l227_22711


namespace NUMINAMATH_CALUDE_sin_alpha_value_l227_22734

theorem sin_alpha_value (α : Real) (h : 3 * Real.sin (2 * α) = Real.cos α) : 
  Real.sin α = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l227_22734


namespace NUMINAMATH_CALUDE_simplify_expression_l227_22708

theorem simplify_expression (x y : ℝ) : 3*x + 2*y + 4*x + 5*y + 7 = 7*x + 7*y + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l227_22708


namespace NUMINAMATH_CALUDE_perfect_square_condition_l227_22716

theorem perfect_square_condition (k : ℕ) : 
  (∃ m : ℤ, k^2 - 101*k = m^2) ↔ (k = 101 ∨ k = 2601) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l227_22716


namespace NUMINAMATH_CALUDE_triangle_pqr_rotation_l227_22761

/-- Triangle PQR with given properties and rotation of PQ --/
theorem triangle_pqr_rotation (P Q R : ℝ × ℝ) (h1 : P = (0, 0)) (h2 : R = (8, 0))
  (h3 : Q.1 ≥ 0 ∧ Q.2 ≥ 0) -- Q in first quadrant
  (h4 : (Q.1 - R.1) * (Q.2 - R.2) = 0) -- ∠QRP = 90°
  (h5 : (Q.2 - P.2) = (Q.1 - P.1)) -- ∠QPR = 45°
  : (- Q.2, Q.1) = (-8, 8) := by
  sorry

end NUMINAMATH_CALUDE_triangle_pqr_rotation_l227_22761


namespace NUMINAMATH_CALUDE_triangle_third_side_range_l227_22757

theorem triangle_third_side_range (a b x : ℕ) : 
  a = 7 → b = 10 → (∃ (s : ℕ), s = x ∧ 4 ≤ s ∧ s ≤ 16) ↔ 
  (a + b > x ∧ x + a > b ∧ x + b > a) := by sorry

end NUMINAMATH_CALUDE_triangle_third_side_range_l227_22757


namespace NUMINAMATH_CALUDE_find_x_l227_22799

theorem find_x : ∃ x : ℚ, (3 * x + 5) / 7 = 17 ∧ x = 38 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l227_22799


namespace NUMINAMATH_CALUDE_exists_number_with_reversed_digits_and_middle_zero_l227_22700

/-- Represents a three-digit number in a given base -/
structure ThreeDigitNumber (base : ℕ) where
  d : ℕ
  e : ℕ
  f : ℕ
  d_lt_base : d < base
  e_lt_base : e < base
  f_lt_base : f < base

/-- Converts a ThreeDigitNumber to its numerical value -/
def to_nat {base : ℕ} (n : ThreeDigitNumber base) : ℕ :=
  n.d * base^2 + n.e * base + n.f

theorem exists_number_with_reversed_digits_and_middle_zero :
  ∃ (n : ThreeDigitNumber 6) (m : ThreeDigitNumber 8),
    to_nat n = to_nat m ∧
    n.d = m.f ∧
    n.e = 0 ∧
    n.e = m.e ∧
    n.f = m.d :=
sorry

end NUMINAMATH_CALUDE_exists_number_with_reversed_digits_and_middle_zero_l227_22700


namespace NUMINAMATH_CALUDE_yoongis_calculation_l227_22738

theorem yoongis_calculation (x : ℝ) : x / 9 = 30 → x - 37 = 233 := by
  sorry

end NUMINAMATH_CALUDE_yoongis_calculation_l227_22738


namespace NUMINAMATH_CALUDE_april_coffee_cost_l227_22717

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the cost of coffee for a given day -/
def coffeeCost (day: DayOfWeek) (isEarthDay: Bool) : ℚ :=
  match day with
  | DayOfWeek.Monday => 3.5
  | DayOfWeek.Friday => 3
  | _ => if isEarthDay then 3 else 4

/-- Calculates the total cost of coffee for April -/
def aprilCoffeeCost (startDay: DayOfWeek) : ℚ :=
  sorry

/-- Theorem stating that Jon's total spending on coffee in April is $112 -/
theorem april_coffee_cost :
  aprilCoffeeCost DayOfWeek.Thursday = 112 := by
  sorry

end NUMINAMATH_CALUDE_april_coffee_cost_l227_22717


namespace NUMINAMATH_CALUDE_men_in_room_l227_22781

/-- Represents the number of people in a room -/
structure RoomPopulation where
  men : ℕ
  women : ℕ

/-- Calculates the final number of men in the room -/
def finalMenCount (initial : RoomPopulation) : ℕ :=
  initial.men + 2

/-- Theorem: Given the initial conditions and final number of women,
    prove that there are 14 men in the room -/
theorem men_in_room (initial : RoomPopulation) 
    (h1 : initial.men = 4 * initial.women / 5)  -- Initial ratio 4:5
    (h2 : 2 * (initial.women - 3) = 24)         -- Final women count after changes
    : finalMenCount initial = 14 := by
  sorry


end NUMINAMATH_CALUDE_men_in_room_l227_22781


namespace NUMINAMATH_CALUDE_right_triangle_sets_l227_22754

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  ¬ is_right_triangle 1.1 1.5 1.9 ∧
  ¬ is_right_triangle 5 11 12 ∧
  is_right_triangle 1.2 1.6 2.0 ∧
  ¬ is_right_triangle 3 4 8 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l227_22754


namespace NUMINAMATH_CALUDE_bounded_sequence_periodic_l227_22763

/-- A bounded sequence of integers satisfying the given recurrence relation -/
def BoundedSequence (a : ℕ → ℤ) : Prop :=
  ∃ M : ℕ, ∀ n : ℕ, |a n| ≤ M ∧
  ∀ n ≥ 5, a n = (a (n-1) + a (n-2) + a (n-3) * a (n-4)) / (a (n-1) * a (n-2) + a (n-3) + a (n-4))

/-- Definition of a periodic sequence -/
def IsPeriodic (a : ℕ → ℤ) (l : ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ ∀ k ≥ l, a (k + T) = a k

/-- The main theorem -/
theorem bounded_sequence_periodic (a : ℕ → ℤ) (h : BoundedSequence a) :
  ∃ l : ℕ, IsPeriodic a l := by sorry

end NUMINAMATH_CALUDE_bounded_sequence_periodic_l227_22763


namespace NUMINAMATH_CALUDE_south_side_maximum_l227_22737

/-- Represents the number of paths for each side of the mountain -/
structure MountainPaths where
  east : Nat
  west : Nat
  south : Nat
  north : Nat

/-- Calculates the number of ways to ascend and descend for a given side -/
def waysForSide (paths : MountainPaths) (side : Nat) : Nat :=
  side * (paths.east + paths.west + paths.south + paths.north - side)

/-- Theorem stating that the south side provides the maximum number of ways -/
theorem south_side_maximum (paths : MountainPaths) 
    (h1 : paths.east = 2)
    (h2 : paths.west = 3)
    (h3 : paths.south = 4)
    (h4 : paths.north = 1) :
  ∀ side, waysForSide paths paths.south ≥ waysForSide paths side :=
by
  sorry

#eval waysForSide { east := 2, west := 3, south := 4, north := 1 } 4

end NUMINAMATH_CALUDE_south_side_maximum_l227_22737


namespace NUMINAMATH_CALUDE_power_of_three_mod_eleven_l227_22783

theorem power_of_three_mod_eleven : 3^1320 % 11 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eleven_l227_22783


namespace NUMINAMATH_CALUDE_only_possible_knight_count_l227_22736

/-- Represents a person on the island -/
inductive Person
| Knight
| Liar

/-- The total number of people on the island -/
def total_people : Nat := 2021

/-- A function that determines if a person's claim is true given their position and type -/
def claim_is_true (position : Nat) (person_type : Person) (num_knights : Nat) : Prop :=
  match person_type with
  | Person.Knight => total_people - position - (total_people - num_knights) > position - num_knights
  | Person.Liar => total_people - position - (total_people - num_knights) ≤ position - num_knights

/-- The main theorem stating that the only possible number of knights is 1010 -/
theorem only_possible_knight_count :
  ∃! num_knights : Nat,
    num_knights ≤ total_people ∧
    ∀ position : Nat, position < total_people →
      (claim_is_true position Person.Knight num_knights ∧ position < num_knights) ∨
      (claim_is_true position Person.Liar num_knights ∧ position ≥ num_knights) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_only_possible_knight_count_l227_22736


namespace NUMINAMATH_CALUDE_camp_total_is_250_l227_22755

/-- Represents the distribution of students in a boys camp --/
structure CampDistribution where
  total : ℕ
  schoolA : ℕ
  schoolB : ℕ
  schoolC : ℕ
  schoolAScience : ℕ
  schoolAMath : ℕ
  schoolALiterature : ℕ
  schoolBScience : ℕ
  schoolBMath : ℕ
  schoolBLiterature : ℕ
  schoolCScience : ℕ
  schoolCMath : ℕ
  schoolCLiterature : ℕ

/-- The camp distribution satisfies the given conditions --/
def isValidDistribution (d : CampDistribution) : Prop :=
  d.schoolA = d.total / 5 ∧
  d.schoolB = 3 * d.total / 10 ∧
  d.schoolC = d.total / 2 ∧
  d.schoolAScience = 3 * d.schoolA / 10 ∧
  d.schoolAMath = 2 * d.schoolA / 5 ∧
  d.schoolALiterature = 3 * d.schoolA / 10 ∧
  d.schoolBScience = d.schoolB / 4 ∧
  d.schoolBMath = 7 * d.schoolB / 20 ∧
  d.schoolBLiterature = 2 * d.schoolB / 5 ∧
  d.schoolCScience = 3 * d.schoolC / 20 ∧
  d.schoolCMath = d.schoolC / 2 ∧
  d.schoolCLiterature = 7 * d.schoolC / 20 ∧
  d.schoolA - d.schoolAScience = 35 ∧
  d.schoolBMath = 20

/-- Theorem: Given the conditions, the total number of boys in the camp is 250 --/
theorem camp_total_is_250 (d : CampDistribution) (h : isValidDistribution d) : d.total = 250 := by
  sorry


end NUMINAMATH_CALUDE_camp_total_is_250_l227_22755


namespace NUMINAMATH_CALUDE_remainder_of_B_l227_22780

theorem remainder_of_B (A : ℕ) : (9 * A + 13) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_B_l227_22780


namespace NUMINAMATH_CALUDE_probability_of_drawing_item_l227_22706

/-- Proves that the probability of drawing each item in a sample is 1/5 given the total number of components and sample size -/
theorem probability_of_drawing_item 
  (total_components : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_components = 100) 
  (h2 : sample_size = 20) : 
  (sample_size : ℚ) / (total_components : ℚ) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_drawing_item_l227_22706


namespace NUMINAMATH_CALUDE_no_triangle_pairs_l227_22746

/-- Given a set of n different elements, prove that if 4m ≤ n², 
    then there exists a set of m non-ordered pairs that do not form any triangles. -/
theorem no_triangle_pairs (n m : ℕ) (h : 4 * m ≤ n ^ 2) :
  ∃ (S : Finset (Fin n)) (P : Finset (Fin n × Fin n)),
    S.card = n ∧ 
    P.card = m ∧
    (∀ (p : Fin n × Fin n), p ∈ P → p.1 ≠ p.2) ∧
    (∀ (a b c : Fin n × Fin n), a ∈ P → b ∈ P → c ∈ P → 
      ¬(a.1 = b.1 ∧ b.2 = c.1 ∧ c.2 = a.2)) :=
by sorry

end NUMINAMATH_CALUDE_no_triangle_pairs_l227_22746


namespace NUMINAMATH_CALUDE_prism_volume_is_400_l227_22790

/-- The volume of a right rectangular prism with face areas 40, 50, and 80 square centimeters -/
def prism_volume : ℝ := 400

/-- The areas of the three faces of the prism -/
def face_area_1 : ℝ := 40
def face_area_2 : ℝ := 50
def face_area_3 : ℝ := 80

/-- Theorem: The volume of the prism is 400 cubic centimeters -/
theorem prism_volume_is_400 :
  ∃ (a b c : ℝ),
    a * b = face_area_1 ∧
    a * c = face_area_2 ∧
    b * c = face_area_3 ∧
    a * b * c = prism_volume :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_is_400_l227_22790


namespace NUMINAMATH_CALUDE_investment_difference_l227_22767

def initial_investment : ℝ := 500

def jackson_multiplier : ℝ := 4

def brandon_percentage : ℝ := 0.2

def jackson_final (initial : ℝ) (multiplier : ℝ) : ℝ := initial * multiplier

def brandon_final (initial : ℝ) (percentage : ℝ) : ℝ := initial * percentage

theorem investment_difference :
  jackson_final initial_investment jackson_multiplier - brandon_final initial_investment brandon_percentage = 1900 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_l227_22767


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l227_22707

theorem quadratic_solution_difference_squared (α β : ℝ) : 
  α ≠ β ∧ 
  α^2 - 3*α + 2 = 0 ∧ 
  β^2 - 3*β + 2 = 0 → 
  (α - β)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l227_22707


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l227_22776

theorem smaller_number_in_ratio (n m d u x y : ℝ) : 
  0 < n → n < m → x > 0 → y > 0 → x / y = n / m → x + y + u = d → 
  min x y = n * (d - u) / (n + m) := by
sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l227_22776


namespace NUMINAMATH_CALUDE_solve_for_a_l227_22788

theorem solve_for_a (x y a : ℚ) 
  (hx : x = 1)
  (hy : y = -2)
  (heq : 2 * x - a * y = 3) :
  a = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l227_22788


namespace NUMINAMATH_CALUDE_parabola_chords_fixed_point_and_isosceles_triangle_l227_22752

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point A
def point_A : ℝ × ℝ := (1, 2)

-- Define a chord on the parabola passing through A
def chord_through_A (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2 ∧ parabola point_A.1 point_A.2

-- Define perpendicularity of two chords
def perpendicular_chords (P Q : ℝ × ℝ) : Prop :=
  (P.1 - point_A.1) * (Q.1 - point_A.1) + (P.2 - point_A.2) * (Q.2 - point_A.2) = 0

-- Define the point T
def point_T : ℝ × ℝ := (5, -2)

-- Define a line passing through a point
def line_through_point (P Q : ℝ × ℝ) (T : ℝ × ℝ) : Prop :=
  (T.2 - P.2) * (Q.1 - P.1) = (Q.2 - P.2) * (T.1 - P.1)

-- Define an isosceles triangle
def isosceles_triangle (P Q : ℝ × ℝ) : Prop :=
  (P.1 - point_A.1)^2 + (P.2 - point_A.2)^2 = (Q.1 - point_A.1)^2 + (Q.2 - point_A.2)^2

-- Theorem statement
theorem parabola_chords_fixed_point_and_isosceles_triangle
  (P Q : ℝ × ℝ)
  (h1 : chord_through_A P)
  (h2 : chord_through_A Q)
  (h3 : perpendicular_chords P Q)
  (h4 : line_through_point P Q point_T) :
  (∀ R : ℝ × ℝ, chord_through_A R → perpendicular_chords P R → line_through_point P R point_T) ∧
  (∃! R : ℝ × ℝ, chord_through_A R ∧ perpendicular_chords P R ∧ isosceles_triangle P R) :=
sorry

end NUMINAMATH_CALUDE_parabola_chords_fixed_point_and_isosceles_triangle_l227_22752


namespace NUMINAMATH_CALUDE_pizza_order_cost_is_185_l227_22751

/-- Represents the cost calculation for a pizza order with special offers --/
def pizza_order_cost (
  large_pizza_price : ℚ)
  (medium_pizza_price : ℚ)
  (small_pizza_price : ℚ)
  (topping_price : ℚ)
  (drink_price : ℚ)
  (garlic_bread_price : ℚ)
  (triple_cheese_count : ℕ)
  (triple_cheese_toppings : ℕ)
  (meat_lovers_count : ℕ)
  (meat_lovers_toppings : ℕ)
  (veggie_delight_count : ℕ)
  (veggie_delight_toppings : ℕ)
  (drink_count : ℕ)
  (garlic_bread_count : ℕ) : ℚ :=
  let triple_cheese_cost := (triple_cheese_count / 2) * large_pizza_price + triple_cheese_count * triple_cheese_toppings * topping_price
  let meat_lovers_cost := ((meat_lovers_count + 2) / 3) * medium_pizza_price + meat_lovers_count * meat_lovers_toppings * topping_price
  let veggie_delight_cost := ((veggie_delight_count * 3) / 5) * small_pizza_price + veggie_delight_count * veggie_delight_toppings * topping_price
  let drink_and_bread_cost := drink_count * drink_price + max 0 (garlic_bread_count - drink_count) * garlic_bread_price
  triple_cheese_cost + meat_lovers_cost + veggie_delight_cost + drink_and_bread_cost

/-- Theorem stating that the given order costs $185 --/
theorem pizza_order_cost_is_185 :
  pizza_order_cost 10 8 5 (5/2) 2 4 6 2 4 3 10 1 8 5 = 185 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_cost_is_185_l227_22751


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l227_22713

theorem exponential_equation_solution (x y : ℝ) :
  (5 : ℝ) ^ (x + y + 4) = 625 ^ x → y = 3 * x - 4 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l227_22713


namespace NUMINAMATH_CALUDE_series_sum_equals_ln2_minus_half_l227_22727

open Real

/-- The sum of the series Σ(1/((2n-1) * 2n * (2n+1))) for n from 1 to infinity -/
noncomputable def seriesSum : ℝ := ∑' n, 1 / ((2*n - 1) * (2*n) * (2*n + 1))

/-- Theorem stating that the sum of the series equals ln 2 - 1/2 -/
theorem series_sum_equals_ln2_minus_half : seriesSum = log 2 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_ln2_minus_half_l227_22727


namespace NUMINAMATH_CALUDE_milk_production_increase_l227_22747

/-- Given the initial milk production rate and an increase in production rate,
    calculate the new amount of milk produced by double the cows in triple the time. -/
theorem milk_production_increase (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let initial_rate := y / (x * z)
  let increased_rate := initial_rate * 1.1
  let new_production := 2 * x * increased_rate * 3 * z
  new_production = 6.6 * y := by sorry

end NUMINAMATH_CALUDE_milk_production_increase_l227_22747


namespace NUMINAMATH_CALUDE_girls_to_boys_ratio_l227_22718

theorem girls_to_boys_ratio (girls boys : ℕ) : 
  girls = boys + 5 →
  girls + boys = 35 →
  girls * 3 = boys * 4 := by
sorry

end NUMINAMATH_CALUDE_girls_to_boys_ratio_l227_22718


namespace NUMINAMATH_CALUDE_tangent_line_correct_l227_22794

/-- The equation of a cubic curve -/
def f (x : ℝ) : ℝ := x^3 - x + 3

/-- The point of tangency -/
def point : ℝ × ℝ := (1, 3)

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := 2*x - y + 1 = 0

/-- Theorem stating that the tangent line equation is correct -/
theorem tangent_line_correct :
  let (a, b) := point
  tangent_line a b ∧
  ∀ x, tangent_line x (f x) → x = a := by sorry

end NUMINAMATH_CALUDE_tangent_line_correct_l227_22794


namespace NUMINAMATH_CALUDE_point_distance_from_origin_l227_22729

theorem point_distance_from_origin (x : ℚ) : 
  |x| = (5 : ℚ) / 2 → x = (5 : ℚ) / 2 ∨ x = -(5 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_point_distance_from_origin_l227_22729


namespace NUMINAMATH_CALUDE_smallest_first_term_arithmetic_progression_l227_22786

theorem smallest_first_term_arithmetic_progression 
  (S₃ S₆ : ℕ) (d₁ : ℚ) 
  (h₁ : d₁ ≥ 1/2) 
  (h₂ : S₃ = 3 * d₁ + 3 * (S₆ - 2 * S₃) / 3) 
  (h₃ : S₆ = 6 * d₁ + 15 * (S₆ - 2 * S₃) / 3) :
  d₁ ≥ 5/9 :=
sorry

end NUMINAMATH_CALUDE_smallest_first_term_arithmetic_progression_l227_22786


namespace NUMINAMATH_CALUDE_negation_of_positive_sum_l227_22743

theorem negation_of_positive_sum (x y : ℝ) :
  (¬(x > 0 ∧ y > 0 → x + y > 0)) ↔ (x ≤ 0 ∨ y ≤ 0 → x + y ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_positive_sum_l227_22743


namespace NUMINAMATH_CALUDE_bicycle_problem_l227_22756

/-- Prove that given the conditions of the bicycle problem, student B's speed is 12 km/h -/
theorem bicycle_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
  (h1 : distance = 12)
  (h2 : speed_ratio = 1.2)
  (h3 : time_difference = 1/6) :
  ∃ (speed_B : ℝ), 
    distance / speed_B - distance / (speed_ratio * speed_B) = time_difference ∧ 
    speed_B = 12 := by
sorry

end NUMINAMATH_CALUDE_bicycle_problem_l227_22756


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l227_22701

-- Define the hyperbola C
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_x_axis : Bool
  asymptotes_tangent_to_parabola : Bool

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x - 1

-- Define the eccentricity of a hyperbola
def eccentricity (h : Hyperbola) : ℝ := sorry

-- Theorem statement
theorem hyperbola_eccentricity (C : Hyperbola) :
  C.center = (0, 0) →
  C.foci_on_x_axis = true →
  C.asymptotes_tangent_to_parabola = true →
  eccentricity C = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l227_22701


namespace NUMINAMATH_CALUDE_claire_cleaning_hours_l227_22748

/-- Calculates the hours spent cleaning given Claire's daily schedule. -/
def hours_cleaning (total_day_hours sleep_hours cooking_hours crafting_hours : ℕ) : ℕ :=
  let working_hours := total_day_hours - sleep_hours
  let cleaning_hours := working_hours - cooking_hours - 2 * crafting_hours
  cleaning_hours

/-- Theorem stating that Claire spends 4 hours cleaning given her schedule. -/
theorem claire_cleaning_hours :
  hours_cleaning 24 8 2 5 = 4 := by
  sorry

#eval hours_cleaning 24 8 2 5

end NUMINAMATH_CALUDE_claire_cleaning_hours_l227_22748


namespace NUMINAMATH_CALUDE_karthik_weight_average_l227_22775

def karthik_weight_lower_bound : ℝ := 56
def karthik_weight_upper_bound : ℝ := 57

theorem karthik_weight_average :
  let min_weight := karthik_weight_lower_bound
  let max_weight := karthik_weight_upper_bound
  (min_weight + max_weight) / 2 = 56.5 := by sorry

end NUMINAMATH_CALUDE_karthik_weight_average_l227_22775


namespace NUMINAMATH_CALUDE_brenda_erasers_count_l227_22719

/-- The number of groups Brenda creates -/
def num_groups : ℕ := 3

/-- The number of erasers in each group -/
def erasers_per_group : ℕ := 90

/-- The total number of erasers Brenda has -/
def total_erasers : ℕ := num_groups * erasers_per_group

theorem brenda_erasers_count : total_erasers = 270 := by
  sorry

end NUMINAMATH_CALUDE_brenda_erasers_count_l227_22719


namespace NUMINAMATH_CALUDE_small_circle_radius_l227_22742

/-- Given a large circle of radius 10 meters containing seven congruent smaller circles
    arranged with six forming a hexagon around one central circle, prove that the radius
    of each smaller circle is 5 meters. -/
theorem small_circle_radius (R : ℝ) (r : ℝ) :
  R = 10 ∧  -- Radius of the large circle
  (2 * r + 2 * r = 2 * R) →  -- Diameter of large circle equals two radii plus one diameter of small circles
  r = 5 :=
by sorry

end NUMINAMATH_CALUDE_small_circle_radius_l227_22742


namespace NUMINAMATH_CALUDE_kevin_cards_l227_22760

/-- The number of cards Kevin ends up with given his initial cards and found cards -/
def total_cards (initial : ℕ) (found : ℕ) : ℕ :=
  initial + found

/-- Theorem stating that Kevin ends up with 54 cards -/
theorem kevin_cards : total_cards 7 47 = 54 := by
  sorry

end NUMINAMATH_CALUDE_kevin_cards_l227_22760


namespace NUMINAMATH_CALUDE_dress_price_calculation_l227_22758

-- Define the original price
def original_price : ℝ := 120

-- Define the discount rate
def discount_rate : ℝ := 0.30

-- Define the tax rate
def tax_rate : ℝ := 0.15

-- Define the total selling price
def total_selling_price : ℝ := original_price * (1 - discount_rate) * (1 + tax_rate)

-- Theorem to prove
theorem dress_price_calculation :
  total_selling_price = 96.6 := by sorry

end NUMINAMATH_CALUDE_dress_price_calculation_l227_22758


namespace NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l227_22772

theorem factor_81_minus_27x_cubed (x : ℝ) :
  81 - 27 * x^3 = 3 * (3 - x) * (81 + 27 * x + 9 * x^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l227_22772


namespace NUMINAMATH_CALUDE_min_fraction_sum_l227_22749

def is_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

def are_distinct (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem min_fraction_sum (A B C D : ℕ) 
  (h1 : is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D) 
  (h2 : are_distinct A B C D) :
  (A : ℚ) / B + (C : ℚ) / D ≥ 31 / 56 := by
  sorry

end NUMINAMATH_CALUDE_min_fraction_sum_l227_22749


namespace NUMINAMATH_CALUDE_geometric_figures_sequence_l227_22762

/-- The number of nonoverlapping unit squares in the nth figure -/
def f (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 4

theorem geometric_figures_sequence :
  f 0 = 4 ∧ f 1 = 10 ∧ f 2 = 20 ∧ f 3 = 34 → f 150 = 45604 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_figures_sequence_l227_22762


namespace NUMINAMATH_CALUDE_f_max_value_l227_22765

/-- A function f(x) with specific properties --/
def f (a b : ℝ) (x : ℝ) : ℝ := (4 - x^2) * (a * x^2 + b * x + 5)

/-- The theorem stating the maximum value of f(x) --/
theorem f_max_value (a b : ℝ) :
  (∀ x : ℝ, f a b x = f a b (-3 - x)) →  -- Symmetry condition
  (∃ M : ℝ, ∀ x : ℝ, f a b x ≤ M ∧ ∃ x₀ : ℝ, f a b x₀ = M) →  -- Maximum exists
  (∃ M : ℝ, M = 36 ∧ ∀ x : ℝ, f a b x ≤ M ∧ ∃ x₀ : ℝ, f a b x₀ = M) :=
by
  sorry


end NUMINAMATH_CALUDE_f_max_value_l227_22765


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l227_22789

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l227_22789


namespace NUMINAMATH_CALUDE_simplify_fraction_l227_22787

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -3) :
  (x + 4) / (x^2 + 3*x) - 1 / (3*x + x^2) = 1 / x := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l227_22787


namespace NUMINAMATH_CALUDE_net_calorie_intake_l227_22724

/-- Calculate net calorie intake after jogging -/
theorem net_calorie_intake
  (breakfast_calories : ℕ)
  (jogging_time : ℕ)
  (calorie_burn_rate : ℕ)
  (h1 : breakfast_calories = 900)
  (h2 : jogging_time = 30)
  (h3 : calorie_burn_rate = 10) :
  breakfast_calories - jogging_time * calorie_burn_rate = 600 :=
by sorry

end NUMINAMATH_CALUDE_net_calorie_intake_l227_22724


namespace NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l227_22702

/-- Given a geometric sequence with first term 2 and common ratio 5/3,
    the 10th term is equal to 3906250/19683. -/
theorem tenth_term_of_geometric_sequence :
  let a₁ : ℚ := 2
  let r : ℚ := 5/3
  let n : ℕ := 10
  let aₙ : ℕ → ℚ := λ k => a₁ * r^(k - 1)
  aₙ n = 3906250/19683 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_geometric_sequence_l227_22702


namespace NUMINAMATH_CALUDE_constant_term_expansion_l227_22791

/-- The constant term in the expansion of (x - 1/x)^6 -/
def constantTerm : ℤ := -20

/-- The binomial coefficient function -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_term_expansion :
  constantTerm = -binomial 6 3 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l227_22791


namespace NUMINAMATH_CALUDE_complex_number_solution_l227_22739

theorem complex_number_solution (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 1 - I) 
  (h₂ : z₁ * z₂ = 1 + I) : 
  z₂ = I := by sorry

end NUMINAMATH_CALUDE_complex_number_solution_l227_22739


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l227_22731

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (median : ℕ) : 
  n = 81 → sum = 9^5 → sum = n * median → median = 729 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l227_22731


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l227_22745

theorem trigonometric_equation_solution (x : ℝ) : 
  (∃ k : ℤ, x = 2 * π / 9 + 2 * π / 3 * k ∨ x = -2 * π / 9 + 2 * π / 3 * k) ↔ 
  Real.cos (3 * x - π / 6) - Real.sin (3 * x - π / 6) * Real.tan (π / 6) = 1 / (2 * Real.cos (7 * π / 6)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l227_22745
