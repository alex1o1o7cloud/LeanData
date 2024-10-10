import Mathlib

namespace tangent_alpha_equals_four_l104_10417

theorem tangent_alpha_equals_four (α : Real) 
  (h : 3 * Real.tan α - Real.sin α + 4 * Real.cos α = 12) : 
  Real.tan α = 4 := by
  sorry

end tangent_alpha_equals_four_l104_10417


namespace equation_solution_l104_10416

theorem equation_solution (x : ℝ) (h : x ≠ -2) :
  (x^2 + 2*x + 3) / (x + 2) = x + 3 ↔ x = -1 := by
  sorry

end equation_solution_l104_10416


namespace arithmetic_sequence_property_l104_10488

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₂ + a₈ = 15 - a₅, then a₅ = 5 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_eq : a 2 + a 8 = 15 - a 5) : 
  a 5 = 5 := by
sorry

end arithmetic_sequence_property_l104_10488


namespace perpendicular_lines_solution_l104_10412

theorem perpendicular_lines_solution (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 6 = 0 ∧ x + a*(a+1)*y + a^2 - 1 = 0 → 
   (a * 1 + 2 * (a*(a+1)) = 0)) → 
  (a = 0 ∨ a = -3/2) :=
sorry

end perpendicular_lines_solution_l104_10412


namespace quadratic_root_range_l104_10483

theorem quadratic_root_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 1 ∧ x₂ < 1 ∧ 
   7 * x₁^2 - (m + 13) * x₁ + m^2 - m - 2 = 0 ∧
   7 * x₂^2 - (m + 13) * x₂ + m^2 - m - 2 = 0) →
  -2 < m ∧ m < 4 :=
by sorry

end quadratic_root_range_l104_10483


namespace triangle_inequality_l104_10446

theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  2 * Real.sin A * Real.sin B < -Real.cos (2 * B + C) →
  a^2 + b^2 < c^2 := by
sorry

end triangle_inequality_l104_10446


namespace yq_length_l104_10405

-- Define the triangle PQR
structure Triangle (P Q R : ℝ × ℝ) : Prop where
  side_pq : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 21
  side_qr : Real.sqrt ((Q.1 - R.1)^2 + (Q.2 - R.2)^2) = 29
  side_pr : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 28

-- Define the inscribed triangle XYZ
structure InscribedTriangle (X Y Z : ℝ × ℝ) (P Q R : ℝ × ℝ) : Prop where
  x_on_qr : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ X = (t * Q.1 + (1 - t) * R.1, t * Q.2 + (1 - t) * R.2)
  y_on_rp : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Y = (t * R.1 + (1 - t) * P.1, t * R.2 + (1 - t) * P.2)
  z_on_pq : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Z = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2)

-- Define the arc equality conditions
def ArcEquality (P Q R X Y Z : ℝ × ℝ) : Prop :=
  ∃ (O₄ O₅ O₆ : ℝ × ℝ),
    (Real.sqrt ((P.1 - Y.1)^2 + (P.2 - Y.2)^2) = Real.sqrt ((X.1 - Q.1)^2 + (X.2 - Q.2)^2)) ∧
    (Real.sqrt ((Q.1 - Z.1)^2 + (Q.2 - Z.2)^2) = Real.sqrt ((Y.1 - R.1)^2 + (Y.2 - R.2)^2)) ∧
    (Real.sqrt ((P.1 - Z.1)^2 + (P.2 - Z.2)^2) = Real.sqrt ((Y.1 - Q.1)^2 + (Y.2 - Q.2)^2))

theorem yq_length 
  (P Q R X Y Z : ℝ × ℝ)
  (h₁ : Triangle P Q R)
  (h₂ : InscribedTriangle X Y Z P Q R)
  (h₃ : ArcEquality P Q R X Y Z) :
  Real.sqrt ((Y.1 - Q.1)^2 + (Y.2 - Q.2)^2) = 15 := by sorry

end yq_length_l104_10405


namespace specific_boy_girl_not_adjacent_girls_not_adjacent_l104_10423

-- Define the number of boys and girls
def num_boys : ℕ := 5
def num_girls : ℕ := 3

-- Define the total number of people
def total_people : ℕ := num_boys + num_girls

-- Define the total number of arrangements without restrictions
def total_arrangements : ℕ := (total_people - 1).factorial

-- Theorem for the first part of the problem
theorem specific_boy_girl_not_adjacent :
  (total_arrangements - 2 * (total_people - 2).factorial) = 3600 := by sorry

-- Theorem for the second part of the problem
theorem girls_not_adjacent :
  (num_boys - 1).factorial * (num_boys.choose num_girls) * num_girls.factorial = 1440 := by sorry

end specific_boy_girl_not_adjacent_girls_not_adjacent_l104_10423


namespace intersection_M_complement_N_l104_10465

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set N
def N : Set ℝ := {x | 2*x < 2}

-- Theorem statement
theorem intersection_M_complement_N :
  ∀ x : ℝ, x ∈ (M ∩ (Set.univ \ N)) ↔ 1 ≤ x ∧ x < 3 := by sorry

end intersection_M_complement_N_l104_10465


namespace distinct_colorings_l104_10492

-- Define the symmetry group of the circle
inductive CircleSymmetry
| id : CircleSymmetry
| rot120 : CircleSymmetry
| rot240 : CircleSymmetry
| refl1 : CircleSymmetry
| refl2 : CircleSymmetry
| refl3 : CircleSymmetry

-- Define the coloring function
def Coloring := Fin 3 → Fin 3

-- Define the action of symmetries on colorings
def act (g : CircleSymmetry) (c : Coloring) : Coloring :=
  sorry

-- Define the fixed points under a symmetry
def fixedPoints (g : CircleSymmetry) : Nat :=
  sorry

-- The main theorem
theorem distinct_colorings : 
  (List.sum (List.map fixedPoints [CircleSymmetry.id, CircleSymmetry.rot120, 
    CircleSymmetry.rot240, CircleSymmetry.refl1, CircleSymmetry.refl2, 
    CircleSymmetry.refl3])) / 6 = 10 := by
  sorry

end distinct_colorings_l104_10492


namespace division_in_third_quadrant_l104_10455

/-- Given two complex numbers z₁ and z₂, prove that z₁/z₂ is in the third quadrant -/
theorem division_in_third_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 1 - 2 * Complex.I) 
  (h₂ : z₂ = 2 + 3 * Complex.I) : 
  (z₁ / z₂).re < 0 ∧ (z₁ / z₂).im < 0 := by
  sorry

end division_in_third_quadrant_l104_10455


namespace game_terminates_l104_10400

/-- Represents the state of the game at each step -/
structure GameState where
  x : ℕ  -- First number on the blackboard
  y : ℕ  -- Second number on the blackboard
  r : ℕ  -- Lower bound of the possible range for the unknown number
  s : ℕ  -- Upper bound of the possible range for the unknown number

/-- The game terminates when the range becomes invalid (r > s) -/
def is_terminal (state : GameState) : Prop :=
  state.r > state.s

/-- The next state of the game after a question is asked -/
def next_state (state : GameState) : GameState :=
  { x := state.x
  , y := state.y
  , r := state.y - state.s
  , s := state.x - state.r }

/-- The main theorem: the game terminates in a finite number of steps -/
theorem game_terminates (a b : ℕ) (h : a > 0 ∧ b > 0) :
  ∃ n : ℕ, is_terminal (n.iterate next_state (GameState.mk (min (a + b) (a + b + 1)) (max (a + b) (a + b + 1)) 0 (a + b))) :=
sorry

end game_terminates_l104_10400


namespace total_rehabilitation_centers_l104_10464

/-- The number of rehabilitation centers visited by Lisa, Jude, Han, and Jane -/
def total_centers (lisa jude han jane : ℕ) : ℕ := lisa + jude + han + jane

/-- Theorem stating the total number of rehabilitation centers visited -/
theorem total_rehabilitation_centers :
  ∃ (lisa jude han jane : ℕ),
    lisa = 6 ∧
    jude = lisa / 2 ∧
    han = 2 * jude - 2 ∧
    jane = 2 * han + 6 ∧
    total_centers lisa jude han jane = 27 := by
  sorry

end total_rehabilitation_centers_l104_10464


namespace stop_after_fourth_draw_l104_10429

/-- The probability of stopping after the fourth draw in a box with 5 black and 4 white balls -/
theorem stop_after_fourth_draw (total_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) :
  total_balls = black_balls + white_balls →
  black_balls = 5 →
  white_balls = 4 →
  (black_balls / total_balls : ℚ)^3 * (white_balls / total_balls : ℚ) = (5/9 : ℚ)^3 * (4/9 : ℚ) :=
by sorry

end stop_after_fourth_draw_l104_10429


namespace function_analysis_l104_10447

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*a*x^2 - 5

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 6*a*x

-- Theorem statement
theorem function_analysis (a : ℝ) :
  (f' a 2 = 0) →  -- x = 2 is a critical point
  (a = 1) ∧       -- The value of a is 1
  (∀ x ∈ Set.Icc (-2 : ℝ) (4 : ℝ), f 1 x ≤ 15) ∧  -- Maximum value on [-2, 4] is 15
  (∀ x ∈ Set.Icc (-2 : ℝ) (4 : ℝ), f 1 x ≥ -21)   -- Minimum value on [-2, 4] is -21
:= by sorry

end function_analysis_l104_10447


namespace max_distance_circle_to_line_l104_10469

/-- The circle equation: x^2 + y^2 - 2x - 2y = 0 -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y = 0

/-- The line equation: x + y + 2 = 0 -/
def line_equation (x y : ℝ) : Prop :=
  x + y + 2 = 0

/-- The maximum distance from a point on the circle to the line is 3√2 -/
theorem max_distance_circle_to_line :
  ∃ (x y : ℝ), circle_equation x y ∧
  (∀ (a b : ℝ), circle_equation a b →
    Real.sqrt ((x - a)^2 + (y - b)^2) ≤ 3 * Real.sqrt 2) ∧
  (∃ (p q : ℝ), circle_equation p q ∧
    Real.sqrt ((x - p)^2 + (y - q)^2) = 3 * Real.sqrt 2) :=
sorry

end max_distance_circle_to_line_l104_10469


namespace time_difference_is_six_minutes_l104_10420

/-- The time difference between walking and biking to work -/
def time_difference (blocks : ℕ) (walk_time_per_block : ℚ) (bike_time_per_block : ℚ) : ℚ :=
  blocks * (walk_time_per_block - bike_time_per_block)

/-- Proof that the time difference is 6 minutes -/
theorem time_difference_is_six_minutes :
  time_difference 9 1 (20 / 60) = 6 := by
  sorry

#eval time_difference 9 1 (20 / 60)

end time_difference_is_six_minutes_l104_10420


namespace uncle_bob_parking_probability_l104_10471

def parking_spaces : ℕ := 18
def parked_cars : ℕ := 15
def rv_spaces : ℕ := 3

theorem uncle_bob_parking_probability :
  let total_arrangements := Nat.choose parking_spaces parked_cars
  let blocked_arrangements := Nat.choose (parking_spaces - rv_spaces + 1) (parked_cars - rv_spaces + 1)
  (total_arrangements - blocked_arrangements : ℚ) / total_arrangements = 16 / 51 := by
  sorry

end uncle_bob_parking_probability_l104_10471


namespace triangle_side_inequality_l104_10431

theorem triangle_side_inequality (a b c : ℝ) (h_area : 1 = (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b)))) (h_order : a ≤ b ∧ b ≤ c) : b ≥ Real.sqrt 2 := by
  sorry

end triangle_side_inequality_l104_10431


namespace inner_automorphism_is_automorphism_l104_10473

variable {G : Type*} [Group G]

def inner_automorphism (x : G) (y : G) : G := x⁻¹ * y * x

theorem inner_automorphism_is_automorphism (x : G) :
  Function.Bijective (inner_automorphism x) ∧
  ∀ y z : G, inner_automorphism x (y * z) = inner_automorphism x y * inner_automorphism x z :=
sorry

end inner_automorphism_is_automorphism_l104_10473


namespace sum_of_x_and_y_equals_two_l104_10432

theorem sum_of_x_and_y_equals_two (x y : ℝ) 
  (h : 2 * x^2 + 2 * y^2 = 20 * x - 12 * y + 68) : 
  x + y = 2 := by
sorry

end sum_of_x_and_y_equals_two_l104_10432


namespace digit_sum_problem_l104_10482

theorem digit_sum_problem (J K L : ℕ) : 
  J ≠ K ∧ J ≠ L ∧ K ≠ L →
  J < 10 ∧ K < 10 ∧ L < 10 →
  100 * J + 10 * K + L + 100 * J + 10 * L + L + 100 * J + 10 * K + L = 479 →
  J + K + L = 11 := by
  sorry

end digit_sum_problem_l104_10482


namespace three_sequence_comparison_l104_10497

theorem three_sequence_comparison 
  (a b c : ℕ → ℕ) : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q := by
sorry

end three_sequence_comparison_l104_10497


namespace population_doubling_time_l104_10494

/-- The annual birth rate per 1000 people -/
def birth_rate : ℝ := 39.4

/-- The annual death rate per 1000 people -/
def death_rate : ℝ := 19.4

/-- The number of years for the population to double -/
def doubling_time : ℝ := 35

/-- Theorem stating that given the birth and death rates, the population will double in 35 years -/
theorem population_doubling_time :
  let net_growth_rate := birth_rate - death_rate
  let percentage_growth_rate := net_growth_rate / 10  -- Converted to percentage
  70 / percentage_growth_rate = doubling_time := by sorry

end population_doubling_time_l104_10494


namespace order_of_abc_l104_10467

theorem order_of_abc (a b c : ℝ) : 
  a = 2/21 → b = Real.log 1.1 → c = 21/220 → a < b ∧ b < c := by sorry

end order_of_abc_l104_10467


namespace total_amc8_students_l104_10402

/-- Represents a math class at Euclid Middle School -/
structure MathClass where
  teacher : String
  totalStudents : Nat
  olympiadStudents : Nat

/-- Calculates the number of students in a class taking only AMC 8 -/
def studentsOnlyAMC8 (c : MathClass) : Nat :=
  c.totalStudents - c.olympiadStudents

/-- Theorem: The total number of students only taking AMC 8 is 26 -/
theorem total_amc8_students (germain newton young : MathClass)
  (h_germain : germain = { teacher := "Mrs. Germain", totalStudents := 13, olympiadStudents := 3 })
  (h_newton : newton = { teacher := "Mr. Newton", totalStudents := 10, olympiadStudents := 2 })
  (h_young : young = { teacher := "Mrs. Young", totalStudents := 12, olympiadStudents := 4 }) :
  studentsOnlyAMC8 germain + studentsOnlyAMC8 newton + studentsOnlyAMC8 young = 26 := by
  sorry

end total_amc8_students_l104_10402


namespace toms_friend_decks_l104_10438

/-- The problem of calculating how many decks Tom's friend bought -/
theorem toms_friend_decks :
  ∀ (cost_per_deck : ℕ) (toms_decks : ℕ) (total_spent : ℕ),
    cost_per_deck = 8 →
    toms_decks = 3 →
    total_spent = 64 →
    ∃ (friends_decks : ℕ),
      friends_decks * cost_per_deck + toms_decks * cost_per_deck = total_spent ∧
      friends_decks = 5 :=
by sorry

end toms_friend_decks_l104_10438


namespace sine_graph_shift_l104_10462

theorem sine_graph_shift (x : ℝ) :
  2 * Real.sin (2 * (x + π/8) - π/4) = 2 * Real.sin (2 * x) := by
  sorry

end sine_graph_shift_l104_10462


namespace family_probability_theorem_l104_10487

-- Define the family structure
structure Family :=
  (boys : ℕ)
  (girls : ℕ)

-- Define the list of families
def families : List Family := [
  ⟨0, 0⟩,  -- A
  ⟨1, 0⟩,  -- B
  ⟨0, 1⟩,  -- C
  ⟨1, 1⟩,  -- D
  ⟨1, 2⟩   -- E
]

-- Define the probability of selecting a girl from family E
def prob_girl_from_E : ℚ := 1/2

-- Define the probability distribution of X
def prob_dist_X : ℕ → ℚ
  | 0 => 1/10
  | 1 => 3/5
  | 2 => 3/10
  | _ => 0

-- Define the expected value of X
def expected_X : ℚ := 6/5

-- Theorem statement
theorem family_probability_theorem :
  (prob_girl_from_E = 1/2) ∧
  (prob_dist_X 0 = 1/10) ∧
  (prob_dist_X 1 = 3/5) ∧
  (prob_dist_X 2 = 3/10) ∧
  (expected_X = 6/5) := by
  sorry

end family_probability_theorem_l104_10487


namespace problem_solution_l104_10466

theorem problem_solution (x y : ℝ) 
  (h1 : 5^2 = x - 5)
  (h2 : (x + y)^(1/3) = 3) :
  x = 30 ∧ y = -3 ∧ Real.sqrt (x + 2*y) = 2 * Real.sqrt 6 ∨ Real.sqrt (x + 2*y) = -2 * Real.sqrt 6 :=
by sorry

end problem_solution_l104_10466


namespace function_properties_l104_10408

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -a^2 * x - 2 * a * x + 1

-- State the theorem
theorem function_properties (a : ℝ) (h_a : a > 1) :
  -- Part 1: Range of f(x)
  (∀ y : ℝ, (∃ x : ℝ, f a x = y) ↔ y < 1) ∧
  -- Part 2: Value of a when minimum on [-2, 1] is -7
  (∃ x : ℝ, x ∈ Set.Icc (-2) 1 ∧ f a x = -7 ∧ ∀ y ∈ Set.Icc (-2) 1, f a y ≥ f a x) → a = 2 :=
by sorry

end function_properties_l104_10408


namespace sum_equals_fraction_l104_10413

def sumFunction (n : ℕ) : ℚ :=
  (n^4 - 1) / (n^4 + 1)

def sumRange : List ℕ := [2, 3, 4, 5]

theorem sum_equals_fraction :
  (sumRange.map sumFunction).sum = 21182880 / 349744361 := by
  sorry

end sum_equals_fraction_l104_10413


namespace functions_satisfy_equation_l104_10476

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c
def g (x : ℝ) : ℝ := a * x^2 + b * x + c
def h (x : ℝ) : ℝ := a * x + b

theorem functions_satisfy_equation :
  ∀ (x y : ℝ), f a b c x - g a b c y = (x - y) * h a b (x + y) := by sorry

end functions_satisfy_equation_l104_10476


namespace book_cost_l104_10406

theorem book_cost (cost_of_three : ℝ) (h : cost_of_three = 45) :
  let cost_of_one := cost_of_three / 3
  8 * cost_of_one = 120 := by sorry

end book_cost_l104_10406


namespace a5_b5_ratio_l104_10495

def S (n : ℕ+) : ℝ := sorry
def T (n : ℕ+) : ℝ := sorry
def a : ℕ+ → ℝ := sorry
def b : ℕ+ → ℝ := sorry

axiom arithmetic_sum_property (n : ℕ+) : S n / T n = (n + 1) / (2 * n - 1)

theorem a5_b5_ratio : a 5 / b 5 = 10 / 17 := by
  sorry

end a5_b5_ratio_l104_10495


namespace sum_of_coefficients_l104_10493

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) :
  (∀ x : ℝ, (3*x - 1)^7 = a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₇ + a₆ + a₅ + a₄ + a₃ + a₂ + a₁ = 2186 := by
sorry

end sum_of_coefficients_l104_10493


namespace machine_present_value_l104_10442

/-- The present value of a machine given its future value and depreciation rate -/
theorem machine_present_value (future_value : ℝ) (depreciation_rate : ℝ) (years : ℕ) 
  (h1 : future_value = 810)
  (h2 : depreciation_rate = 0.1)
  (h3 : years = 2) :
  future_value = 1000 * (1 - depreciation_rate) ^ years := by
  sorry

end machine_present_value_l104_10442


namespace fixed_point_exponential_function_l104_10472

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 2
  f 2 = 3 := by sorry

end fixed_point_exponential_function_l104_10472


namespace car_speed_equality_l104_10426

/-- Prove that given the conditions of the car problem, the average speed of Car Y is equal to the average speed of Car X. -/
theorem car_speed_equality (speed_x : ℝ) (start_delay : ℝ) (distance_after_y_starts : ℝ)
  (h1 : speed_x = 35)
  (h2 : start_delay = 72 / 60)
  (h3 : distance_after_y_starts = 105) :
  ∃ (speed_y : ℝ), speed_y = speed_x := by
  sorry

end car_speed_equality_l104_10426


namespace smallest_positive_solution_l104_10419

theorem smallest_positive_solution :
  let f : ℝ → ℝ := λ x => Real.sqrt (3 * x) - 5 * x
  ∃ x : ℝ, x > 0 ∧ f x = 0 ∧ ∀ y : ℝ, y > 0 ∧ f y = 0 → x ≤ y :=
by
  sorry

end smallest_positive_solution_l104_10419


namespace bollards_contract_l104_10404

theorem bollards_contract (total : ℕ) (installed : ℕ) (remaining : ℕ) : 
  installed = (3 * total) / 4 →
  remaining = 2000 →
  remaining = total / 4 →
  total = 8000 := by
  sorry

end bollards_contract_l104_10404


namespace cost_per_minute_advertising_l104_10421

/-- The cost of one minute of advertising during a race, given the number of advertisements,
    duration of each advertisement, and total cost of transmission. -/
theorem cost_per_minute_advertising (num_ads : ℕ) (duration_per_ad : ℕ) (total_cost : ℕ) :
  num_ads = 5 →
  duration_per_ad = 3 →
  total_cost = 60000 →
  total_cost / (num_ads * duration_per_ad) = 4000 := by
  sorry

end cost_per_minute_advertising_l104_10421


namespace stating_sidorov_cash_sum_l104_10435

/-- The disposable cash of the Sidorov family as of June 1, 2018 -/
def sidorov_cash : ℝ := 724506.3

/-- The first part of the Sidorov family's cash -/
def first_part : ℝ := 496941.3

/-- The second part of the Sidorov family's cash -/
def second_part : ℝ := 227565

/-- 
Theorem stating that the disposable cash of the Sidorov family 
as of June 1, 2018, is the sum of two given parts
-/
theorem sidorov_cash_sum : 
  sidorov_cash = first_part + second_part := by
  sorry

end stating_sidorov_cash_sum_l104_10435


namespace candy_count_l104_10439

theorem candy_count (initial_bags : ℕ) (initial_cookies : ℕ) (remaining_bags : ℕ) 
  (h1 : initial_bags = 14)
  (h2 : initial_cookies = 28)
  (h3 : remaining_bags = 2)
  (h4 : initial_cookies % initial_bags = 0) :
  initial_cookies - (remaining_bags * (initial_cookies / initial_bags)) = 24 := by
  sorry

end candy_count_l104_10439


namespace edith_book_ratio_l104_10425

/-- Given that Edith has 80 novels on her schoolbook shelf and a total of 240 books (novels and writing books combined), 
    prove that the ratio of novels on the shelf to writing books in the suitcase is 1:2. -/
theorem edith_book_ratio :
  let novels_on_shelf : ℕ := 80
  let total_books : ℕ := 240
  let writing_books : ℕ := total_books - novels_on_shelf
  novels_on_shelf * 2 = writing_books := by
  sorry

end edith_book_ratio_l104_10425


namespace product_evaluation_l104_10441

theorem product_evaluation (n : ℕ) (h : n = 3) :
  (n - 2) * (n - 1) * n * (n + 1) * (n + 2) * (n + 3) = 720 := by
  sorry

end product_evaluation_l104_10441


namespace negation_of_proposition_l104_10410

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ (∀ x : ℝ, x ≥ 0 → x ≥ Real.sin x)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x < Real.sin x) :=
by sorry

end negation_of_proposition_l104_10410


namespace greatest_integer_inequality_l104_10470

theorem greatest_integer_inequality : ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 := by sorry

end greatest_integer_inequality_l104_10470


namespace diamond_symmetry_lines_l104_10437

-- Define the binary operation
def diamond (a b : ℝ) : ℝ := a^2 + a*b - b^2

-- Theorem statement
theorem diamond_symmetry_lines :
  ∀ x y : ℝ, diamond x y = diamond y x ↔ y = x ∨ y = -x :=
sorry

end diamond_symmetry_lines_l104_10437


namespace triangle_area_problem_l104_10486

/-- Line with slope m passing through point (x0, y0) -/
def Line (m : ℚ) (x0 y0 : ℚ) : ℚ → ℚ → Prop :=
  fun x y => y - y0 = m * (x - x0)

/-- Area of a triangle given coordinates of its vertices -/
def TriangleArea (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area_problem :
  let line1 := Line (3/4) 1 3
  let line2 := Line (-1/3) 1 3
  let line3 := fun x y => x + y = 8
  let x1 := 1
  let y1 := 3
  let x2 := 21/2
  let y2 := 11/2
  let x3 := 23/7
  let y3 := 32/7
  (∀ x y, line1 x y ↔ y = (3/4) * x + 9/4) ∧
  (∀ x y, line2 x y ↔ y = (-1/3) * x + 10/3) ∧
  line1 x1 y1 ∧
  line2 x1 y1 ∧
  line1 x3 y3 ∧
  line3 x3 y3 ∧
  line2 x2 y2 ∧
  line3 x2 y2 →
  TriangleArea x1 y1 x2 y2 x3 y3 = 475/28 := by
sorry

end triangle_area_problem_l104_10486


namespace k_travel_time_l104_10427

theorem k_travel_time (x : ℝ) 
  (h1 : x > 0) -- K's speed is positive
  (h2 : x - 0.5 > 0) -- M's speed is positive
  (h3 : 45 / (x - 0.5) - 45 / x = 3/4) -- K takes 45 minutes (3/4 hour) less than M
  : 45 / x = 9 := by
  sorry

end k_travel_time_l104_10427


namespace particular_number_plus_eight_l104_10454

theorem particular_number_plus_eight (n : ℝ) : n * 6 = 72 → n + 8 = 20 := by
  sorry

end particular_number_plus_eight_l104_10454


namespace unit_digit_of_15_100_pow_20_l104_10433

-- Define a function to get the unit digit of a natural number
def unitDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem unit_digit_of_15_100_pow_20 :
  unitDigit ((15^100)^20) = 5 := by sorry

end unit_digit_of_15_100_pow_20_l104_10433


namespace regular_hexagon_cosine_product_l104_10498

/-- A regular hexagon ABCDEF inscribed in a circle -/
structure RegularHexagon where
  /-- Side length of the hexagon -/
  side_length : ℝ
  /-- Length of diagonal AC -/
  diagonal_length : ℝ
  /-- Side length is positive -/
  side_pos : side_length > 0
  /-- Diagonal length is positive -/
  diagonal_pos : diagonal_length > 0
  /-- Relationship between side length and diagonal length in a regular hexagon -/
  hexagon_property : diagonal_length^2 = side_length^2 + side_length^2 - 2 * side_length * side_length * (-1/2)

/-- Theorem about the product of cosines in a regular hexagon -/
theorem regular_hexagon_cosine_product (h : RegularHexagon) (h_side : h.side_length = 5) (h_diag : h.diagonal_length = 2) :
  (1 - Real.cos (2 * Real.pi / 3)) * (1 - Real.cos (2 * Real.pi / 3)) = 2.25 := by
  sorry


end regular_hexagon_cosine_product_l104_10498


namespace sqrt_identity_l104_10407

theorem sqrt_identity (θ : Real) (h : θ = 40 * π / 180) :
  Real.sqrt (16 - 12 * Real.sin θ) = 4 + Real.sqrt 3 * (1 / Real.sin θ) := by
  sorry

end sqrt_identity_l104_10407


namespace exponent_increase_l104_10434

theorem exponent_increase (x : ℝ) (y : ℝ) (h : 3^x = y) : 3^(x+1) = 3*y := by
  sorry

end exponent_increase_l104_10434


namespace proposition_evaluations_l104_10430

theorem proposition_evaluations :
  (∀ x : ℝ, x^2 - x + 1 > 0) ∧
  (∀ x : ℝ, x > 2 → x^2 + x - 6 ≥ 0) ∧
  (∃ x : ℝ, x ≠ 2 ∧ x^2 - 5*x + 6 = 0) :=
by sorry

end proposition_evaluations_l104_10430


namespace parallel_lines_a_value_l104_10461

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The theorem to be proved -/
theorem parallel_lines_a_value :
  ∀ a : ℝ,
  let l1 : Line := ⟨a - 1, 2, 3⟩
  let l2 : Line := ⟨1, a, 3⟩
  parallel l1 l2 → a = -1 := by
sorry

end parallel_lines_a_value_l104_10461


namespace intersection_complement_theorem_l104_10445

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}
def N : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}

-- State the theorem
theorem intersection_complement_theorem :
  M ∩ (Set.univ \ N) = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end intersection_complement_theorem_l104_10445


namespace complex_quadrant_l104_10481

theorem complex_quadrant (z : ℂ) (h : (1 + Complex.I * Real.sqrt 3) * z = 2 - Complex.I * Real.sqrt 3) : 
  z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_quadrant_l104_10481


namespace sequence_difference_sum_l104_10440

theorem sequence_difference_sum : 
  (Finset.sum (Finset.range 100) (fun i => 3001 + i)) - 
  (Finset.sum (Finset.range 100) (fun i => 201 + i)) = 280000 := by
  sorry

end sequence_difference_sum_l104_10440


namespace obtuse_triangle_side_range_l104_10475

-- Define the triangle
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define an obtuse triangle
def ObtuseTriangle (a b c : ℝ) : Prop :=
  Triangle a b c ∧ (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2)

-- Theorem statement
theorem obtuse_triangle_side_range :
  ∀ c : ℝ, ObtuseTriangle 4 3 c → c ∈ Set.Ioo 1 (Real.sqrt 7) ∪ Set.Ioo 5 7 :=
by sorry

end obtuse_triangle_side_range_l104_10475


namespace exists_valid_arrangement_l104_10448

-- Define the grid type
def Grid := Matrix (Fin 5) (Fin 5) ℕ

-- Define the sum of a list of numbers
def list_sum (l : List ℕ) : ℕ := l.foldl (·+·) 0

-- Define the property that a grid contains numbers 1 to 12
def contains_one_to_twelve (g : Grid) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 12 → ∃ i j, g i j = n

-- Define the sum of central columns
def central_columns_sum (g : Grid) : Prop :=
  list_sum [g 0 2, g 1 2, g 2 2, g 3 2] = 26 ∧
  list_sum [g 0 3, g 1 3, g 2 3, g 3 3] = 26

-- Define the sum of central rows
def central_rows_sum (g : Grid) : Prop :=
  list_sum [g 2 0, g 2 1, g 2 2, g 2 3] = 26 ∧
  list_sum [g 3 0, g 3 1, g 3 2, g 3 3] = 26

-- Define the sum of roses pattern
def roses_sum (g : Grid) : Prop :=
  list_sum [g 0 2, g 1 2, g 2 2, g 2 3] = 26

-- Define the sum of shamrocks pattern
def shamrocks_sum (g : Grid) : Prop :=
  list_sum [g 2 0, g 3 1, g 4 2, g 1 2] = 26

-- Define the sum of thistle pattern
def thistle_sum (g : Grid) : Prop :=
  list_sum [g 2 2, g 3 2, g 3 3] = 26

-- The main theorem
theorem exists_valid_arrangement :
  ∃ g : Grid,
    contains_one_to_twelve g ∧
    central_columns_sum g ∧
    central_rows_sum g ∧
    roses_sum g ∧
    shamrocks_sum g ∧
    thistle_sum g := by
  sorry

end exists_valid_arrangement_l104_10448


namespace gcd_lcm_sum_l104_10422

theorem gcd_lcm_sum : Nat.gcd 42 70 + Nat.lcm 20 15 = 74 := by
  sorry

end gcd_lcm_sum_l104_10422


namespace parabola_focus_directrix_distance_l104_10444

/-- Given a parabola x^2 = 2py (p > 0), if a point on the parabola with ordinate 1 
    is at distance 3 from the focus, then the distance from the focus to the directrix is 4. -/
theorem parabola_focus_directrix_distance (p : ℝ) (h1 : p > 0) : 
  (∃ x : ℝ, x^2 = 2*p*1 ∧ 
   ((x - 0)^2 + (1 - p/2)^2)^(1/2) = 3) → 
  (0 - (-p/2)) = 4 := by
  sorry

end parabola_focus_directrix_distance_l104_10444


namespace power_sum_2001_l104_10485

theorem power_sum_2001 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) :
  x^2001 + y^2001 = 2^2001 ∨ x^2001 + y^2001 = -(2^2001) := by
  sorry

end power_sum_2001_l104_10485


namespace possible_values_of_a_l104_10459

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | x * a - 1 = 0}

theorem possible_values_of_a :
  ∀ a : ℝ, (A ∩ B a = B a) → (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by sorry

end possible_values_of_a_l104_10459


namespace no_real_roots_condition_l104_10477

theorem no_real_roots_condition (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 :=
by sorry

end no_real_roots_condition_l104_10477


namespace power_division_calculation_l104_10403

theorem power_division_calculation : ((6^6 / 6^5)^3 * 8^3) / 4^3 = 1728 := by
  sorry

end power_division_calculation_l104_10403


namespace inequality_range_l104_10460

theorem inequality_range (x y : ℝ) :
  y - x^2 < Real.sqrt (x^2) →
  ((x ≥ 0 → y < x + x^2) ∧ (x < 0 → y < -x + x^2)) :=
by sorry

end inequality_range_l104_10460


namespace inequality_system_unique_solution_l104_10490

/-- A system of inequalities with parameter a and variable x -/
structure InequalitySystem (a : ℝ) :=
  (x : ℤ)
  (ineq1 : x^3 + 3*x^2 - x - 3 > 0)
  (ineq2 : x^2 - 2*a*x - 1 ≤ 0)
  (a_pos : a > 0)

/-- The theorem stating the range of a for which the system has exactly one integer solution -/
theorem inequality_system_unique_solution :
  ∀ a : ℝ, (∃! s : InequalitySystem a, True) ↔ 3/4 ≤ a ∧ a < 4/3 :=
sorry

end inequality_system_unique_solution_l104_10490


namespace prob_red_after_transfer_l104_10409

/-- Represents the contents of a bag as a pair of natural numbers (white balls, red balls) -/
def BagContents := ℕ × ℕ

/-- The initial contents of bag A -/
def bagA : BagContents := (2, 1)

/-- The initial contents of bag B -/
def bagB : BagContents := (1, 2)

/-- Calculates the probability of drawing a red ball from a bag -/
def probRedBall (bag : BagContents) : ℚ :=
  (bag.2 : ℚ) / ((bag.1 + bag.2) : ℚ)

/-- Calculates the probability of transferring a red ball from bag A to bag B -/
def probTransferRed (bagA : BagContents) : ℚ :=
  (bagA.2 : ℚ) / ((bagA.1 + bagA.2) : ℚ)

/-- Theorem: The probability of drawing a red ball from bag B after transferring a random ball from bag A is 7/12 -/
theorem prob_red_after_transfer (bagA bagB : BagContents) :
  let probWhiteTransfer := 1 - probTransferRed bagA
  let probRedAfterWhite := probRedBall (bagB.1 + 1, bagB.2)
  let probRedAfterRed := probRedBall (bagB.1, bagB.2 + 1)
  probWhiteTransfer * probRedAfterWhite + probTransferRed bagA * probRedAfterRed = 7 / 12 :=
sorry

end prob_red_after_transfer_l104_10409


namespace savings_proof_l104_10415

/-- Calculates a person's savings given their income and income-to-expenditure ratio --/
def calculate_savings (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) : ℕ :=
  income - (income * expenditure_ratio) / income_ratio

/-- Proves that given the specified conditions, the savings are 4000 --/
theorem savings_proof (income : ℕ) (income_ratio : ℕ) (expenditure_ratio : ℕ) 
    (h1 : income = 20000)
    (h2 : income_ratio = 5)
    (h3 : expenditure_ratio = 4) :
  calculate_savings income income_ratio expenditure_ratio = 4000 := by
  sorry

#eval calculate_savings 20000 5 4

end savings_proof_l104_10415


namespace remainder_of_b_mod_13_l104_10463

/-- Given that b ≡ (2^(-1) + 3^(-1) + 5^(-1))^(-1) (mod 13), prove that b ≡ 6 (mod 13) -/
theorem remainder_of_b_mod_13 :
  (((2 : ZMod 13)⁻¹ + (3 : ZMod 13)⁻¹ + (5 : ZMod 13)⁻¹)⁻¹ : ZMod 13) = 6 := by
  sorry

end remainder_of_b_mod_13_l104_10463


namespace lecture_scheduling_l104_10484

theorem lecture_scheduling (n : ℕ) (h : n = 7) :
  let total_permutations := n.factorial
  let valid_orderings := total_permutations / 4
  valid_orderings = 1260 :=
by sorry

end lecture_scheduling_l104_10484


namespace f_x₁_gt_f_x₂_l104_10451

noncomputable section

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- f(x+1) is an even function
axiom f_even : ∀ x, f (x + 1) = f (-x + 1)

-- (x-1)f'(x) < 0
axiom f_decreasing : ∀ x, (x - 1) * f' x < 0

-- x₁ < x₂
variable (x₁ x₂ : ℝ)
axiom x₁_lt_x₂ : x₁ < x₂

-- x₁ + x₂ > 2
axiom sum_gt_two : x₁ + x₂ > 2

-- The theorem to prove
theorem f_x₁_gt_f_x₂ : f x₁ > f x₂ := by sorry

end f_x₁_gt_f_x₂_l104_10451


namespace rectangular_prism_volume_l104_10418

/-- The volume of a rectangular prism -/
def volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a rectangular prism with dimensions 0.6m, 0.3m, and 0.2m is 0.036 m³ -/
theorem rectangular_prism_volume : volume 0.6 0.3 0.2 = 0.036 := by
  sorry

end rectangular_prism_volume_l104_10418


namespace expression_evaluation_l104_10456

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 15) - 2 = -x^4 + 3*x^3 - 5*x^2 + 15*x - 2 := by
  sorry

end expression_evaluation_l104_10456


namespace sin_120_cos_1290_l104_10457

theorem sin_120_cos_1290 : Real.sin (-120 * π / 180) * Real.cos (1290 * π / 180) = 3 / 4 := by
  sorry

end sin_120_cos_1290_l104_10457


namespace hindi_speakers_count_l104_10496

/-- Represents the number of children who can speak a given language or combination of languages -/
structure LanguageCount where
  total : ℕ
  onlyEnglish : ℕ
  onlyHindi : ℕ
  onlySpanish : ℕ
  englishAndHindi : ℕ
  englishAndSpanish : ℕ
  hindiAndSpanish : ℕ
  allThree : ℕ

/-- Calculates the number of children who can speak Hindi -/
def hindiSpeakers (c : LanguageCount) : ℕ :=
  c.onlyHindi + c.englishAndHindi + c.hindiAndSpanish + c.allThree

/-- Theorem stating that the number of Hindi speakers is 45 given the conditions -/
theorem hindi_speakers_count (c : LanguageCount)
  (h_total : c.total = 90)
  (h_onlyEnglish : c.onlyEnglish = 90 * 25 / 100)
  (h_onlyHindi : c.onlyHindi = 90 * 15 / 100)
  (h_onlySpanish : c.onlySpanish = 90 * 10 / 100)
  (h_englishAndHindi : c.englishAndHindi = 90 * 20 / 100)
  (h_englishAndSpanish : c.englishAndSpanish = 90 * 15 / 100)
  (h_hindiAndSpanish : c.hindiAndSpanish = 90 * 10 / 100)
  (h_allThree : c.allThree = 90 * 5 / 100) :
  hindiSpeakers c = 45 := by
  sorry


end hindi_speakers_count_l104_10496


namespace solve_turtle_problem_l104_10453

def turtle_problem (owen_initial : ℕ) (johanna_difference : ℕ) : Prop :=
  let johanna_initial : ℕ := owen_initial - johanna_difference
  let owen_after_month : ℕ := owen_initial * 2
  let johanna_after_loss : ℕ := johanna_initial / 2
  let owen_final : ℕ := owen_after_month + johanna_after_loss
  owen_final = 50

theorem solve_turtle_problem :
  turtle_problem 21 5 := by sorry

end solve_turtle_problem_l104_10453


namespace twenty_paise_coins_l104_10479

theorem twenty_paise_coins (total_coins : ℕ) (total_value : ℚ) : 
  total_coins = 324 →
  total_value = 71 →
  ∃ (coins_20 : ℕ) (coins_25 : ℕ),
    coins_20 + coins_25 = total_coins ∧
    (20 * coins_20 + 25 * coins_25 : ℚ) / 100 = total_value ∧
    coins_20 = 200 := by
  sorry

end twenty_paise_coins_l104_10479


namespace theresa_chocolate_bars_double_kayla_l104_10478

/-- Represents the number of items Kayla bought -/
structure KaylasItems where
  chocolateBars : ℕ
  sodaCans : ℕ
  total : ℕ
  total_eq : chocolateBars + sodaCans = total

/-- Represents the number of items Theresa bought -/
structure TheresasItems where
  chocolateBars : ℕ
  sodaCans : ℕ

/-- The given conditions of the problem -/
class ProblemConditions where
  kayla : KaylasItems
  theresa : TheresasItems
  kayla_total_15 : kayla.total = 15
  theresa_double_kayla : theresa.chocolateBars = 2 * kayla.chocolateBars ∧
                         theresa.sodaCans = 2 * kayla.sodaCans

theorem theresa_chocolate_bars_double_kayla
  [conditions : ProblemConditions] :
  conditions.theresa.chocolateBars = 2 * conditions.kayla.chocolateBars :=
by sorry

end theresa_chocolate_bars_double_kayla_l104_10478


namespace impossible_all_coeffs_roots_l104_10468

/-- Given n > 1 monic quadratic polynomials and 2n distinct coefficients,
    prove that not all coefficients can be roots of the polynomials. -/
theorem impossible_all_coeffs_roots (n : ℕ) (a b : Fin n → ℝ) 
    (h_n : n > 1)
    (h_distinct : ∀ (i j : Fin n), i ≠ j → a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ b j)
    (h_poly : ∀ (i : Fin n), ∃ (x : ℝ), x^2 - a i * x + b i = 0) :
    ¬(∀ (i : Fin n), (∃ (j : Fin n), a i^2 - a j * a i + b j = 0) ∧
                     (∃ (k : Fin n), b i^2 - a k * b i + b k = 0)) :=
by sorry

end impossible_all_coeffs_roots_l104_10468


namespace disjunction_true_when_second_true_l104_10436

theorem disjunction_true_when_second_true (p q : Prop) (hp : ¬p) (hq : q) : p ∨ q := by
  sorry

end disjunction_true_when_second_true_l104_10436


namespace cannot_return_to_start_l104_10499

-- Define the type for points on the plane
def Point := ℝ × ℝ

-- Define the allowed moves
def move_up (p : Point) : Point := (p.1, p.2 + 2*p.1)
def move_down (p : Point) : Point := (p.1, p.2 - 2*p.1)
def move_right (p : Point) : Point := (p.1 + 2*p.2, p.2)
def move_left (p : Point) : Point := (p.1 - 2*p.2, p.2)

-- Define a sequence of moves
inductive Move
| up : Move
| down : Move
| right : Move
| left : Move

def apply_move (p : Point) (m : Move) : Point :=
  match m with
  | Move.up => move_up p
  | Move.down => move_down p
  | Move.right => move_right p
  | Move.left => move_left p

def apply_moves (p : Point) (ms : List Move) : Point :=
  ms.foldl apply_move p

-- The main theorem
theorem cannot_return_to_start : 
  ∀ (ms : List Move), apply_moves (1, Real.sqrt 2) ms ≠ (1, Real.sqrt 2) :=
sorry

end cannot_return_to_start_l104_10499


namespace square_rectangle_contradiction_l104_10480

-- Define the square and rectangle
structure Square where
  side : ℝ
  area : ℝ := side ^ 2

structure Rectangle where
  length : ℝ
  width : ℝ
  area : ℝ := length * width

-- Define the theorem
theorem square_rectangle_contradiction 
  (s : Square) 
  (r : Rectangle) 
  (h1 : r.area = 0.25 * s.area) 
  (h2 : s.area = 0.5 * r.area) : 
  False := by
  sorry

end square_rectangle_contradiction_l104_10480


namespace jar_water_problem_l104_10450

theorem jar_water_problem (small_capacity large_capacity water_amount : ℝ) 
  (h1 : water_amount = (1/6) * small_capacity)
  (h2 : water_amount = (1/3) * large_capacity)
  (h3 : small_capacity > 0)
  (h4 : large_capacity > 0) :
  water_amount / large_capacity = 1/3 := by
  sorry

end jar_water_problem_l104_10450


namespace no_acute_triangle_2016gon_l104_10491

/-- A regular polygon with n sides --/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ

/-- A function that determines if three points form an acute triangle --/
def isAcuteTriangle (a b c : ℝ × ℝ) : Prop :=
  sorry

/-- The minimum number of vertices to paint black in a 2016-gon to avoid acute triangles --/
def minBlackVertices : ℕ := 1008

theorem no_acute_triangle_2016gon (p : RegularPolygon 2016) :
  ∃ (blackVertices : Finset (Fin 2016)),
    blackVertices.card = minBlackVertices ∧
    ∀ (a b c : Fin 2016),
      a ∉ blackVertices → b ∉ blackVertices → c ∉ blackVertices →
      ¬isAcuteTriangle (p.vertices a) (p.vertices b) (p.vertices c) :=
sorry

end no_acute_triangle_2016gon_l104_10491


namespace matching_polygons_l104_10401

def is_matching (n m : ℕ) : Prop :=
  2 * ((n - 2) * 180 / n) = 3 * (360 / m)

theorem matching_polygons :
  ∀ n m : ℕ, n > 2 ∧ m > 2 →
    is_matching n m ↔ ((n = 3 ∧ m = 9) ∨ (n = 4 ∧ m = 6) ∨ (n = 5 ∧ m = 5) ∨ (n = 8 ∧ m = 4)) :=
by sorry

end matching_polygons_l104_10401


namespace orange_juice_glasses_l104_10452

theorem orange_juice_glasses (total_juice : ℕ) (juice_per_glass : ℕ) (h1 : total_juice = 153) (h2 : juice_per_glass = 30) :
  ∃ (num_glasses : ℕ), num_glasses * juice_per_glass ≥ total_juice ∧
  ∀ (m : ℕ), m * juice_per_glass ≥ total_juice → m ≥ num_glasses :=
by sorry

end orange_juice_glasses_l104_10452


namespace cosine_function_properties_l104_10411

/-- Given a cosine function f(x) = a * cos(b * x + c) with positive constants a, b, and c,
    if f(x) reaches its first maximum at x = -π/4 and has a maximum value of 3,
    then a = 3, b = 1, and c = π/4. -/
theorem cosine_function_properties (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) :
  let f : ℝ → ℝ := λ x ↦ a * Real.cos (b * x + c)
  (∀ x, f x ≤ 3) ∧ (f (-π/4) = 3) ∧ (∀ x < -π/4, f x < 3) →
  a = 3 ∧ b = 1 ∧ c = π/4 := by
  sorry

end cosine_function_properties_l104_10411


namespace quadratic_roots_and_sum_l104_10449

theorem quadratic_roots_and_sum : ∃ (m n p : ℕ), 
  (∀ x : ℝ, 2 * x * (5 * x - 11) = -5 ↔ x = (m + Real.sqrt n : ℝ) / p ∨ x = (m - Real.sqrt n : ℝ) / p) ∧ 
  Nat.gcd m (Nat.gcd n p) = 1 ∧
  m + n + p = 92 := by
  sorry

end quadratic_roots_and_sum_l104_10449


namespace g_2009_divisors_l104_10458

/-- g(n) returns the smallest positive integer k such that 1/k has exactly n+1 digits after the decimal point -/
def g (n : ℕ+) : ℕ+ := sorry

/-- The number of positive integer divisors of g(2009) -/
def num_divisors_g_2009 : ℕ := sorry

theorem g_2009_divisors : num_divisors_g_2009 = 2011 := by sorry

end g_2009_divisors_l104_10458


namespace fraction_simplification_l104_10474

theorem fraction_simplification : (1 : ℚ) / 462 + 23 / 42 = 127 / 231 := by sorry

end fraction_simplification_l104_10474


namespace translate_right_2_units_l104_10414

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Translation of a point to the right -/
def translateRight (p : Point2D) (units : ℝ) : Point2D :=
  { x := p.x + units, y := p.y }

theorem translate_right_2_units (A : Point2D) (h : A = ⟨-2, 3⟩) :
  translateRight A 2 = ⟨0, 3⟩ := by
  sorry

end translate_right_2_units_l104_10414


namespace fifth_term_of_arithmetic_sequence_l104_10489

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence. -/
def nthTerm (a : ℕ → ℤ) (n : ℕ) : ℤ := a n

theorem fifth_term_of_arithmetic_sequence
  (a : ℕ → ℤ) (h : ArithmeticSequence a)
  (h10 : nthTerm a 10 = 15)
  (h12 : nthTerm a 12 = 21) :
  nthTerm a 5 = 0 := by
  sorry

end fifth_term_of_arithmetic_sequence_l104_10489


namespace point_on_parallel_segment_l104_10424

/-- Given a point M and a line segment MN parallel to the x-axis, 
    prove that N has specific coordinates -/
theorem point_on_parallel_segment 
  (M : ℝ × ℝ) 
  (length_MN : ℝ) 
  (h_M : M = (2, -4)) 
  (h_length : length_MN = 5) : 
  ∃ (N : ℝ × ℝ), (N = (-3, -4) ∨ N = (7, -4)) ∧ 
                 (N.2 = M.2) ∧ 
                 ((N.1 - M.1)^2 + (N.2 - M.2)^2 = length_MN^2) := by
  sorry

end point_on_parallel_segment_l104_10424


namespace perpendicular_lines_a_value_l104_10428

/-- Given two lines in the form of linear equations,
    returns true if they are perpendicular. -/
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

/-- The slope of the first line 3y + 2x - 6 = 0 -/
def m1 : ℚ := -2/3

/-- The slope of the second line 4y + ax - 5 = 0 in terms of a -/
def m2 (a : ℚ) : ℚ := -a/4

/-- Theorem stating that if the two given lines are perpendicular, then a = -6 -/
theorem perpendicular_lines_a_value :
  are_perpendicular m1 (m2 a) → a = -6 := by
  sorry

end perpendicular_lines_a_value_l104_10428


namespace function_properties_l104_10443

-- Define the function f(x)
noncomputable def f (x a : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + 7 * Real.pi / 6) + a

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ),
    (∀ x, f x a ≤ 2) ∧  -- Maximum value is 2
    (∃ x, f x a = 2) ∧  -- Maximum value is attained
    (a = 1) ∧  -- Value of a
    (∀ x, f x a = f (x + Real.pi) a) ∧  -- Smallest positive period is π
    (∀ k : ℤ, ∀ x ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi),
      ∀ y ∈ Set.Icc (Real.pi / 6 + k * Real.pi) (5 * Real.pi / 12 + k * Real.pi),
      x ≤ y → f y a ≤ f x a)  -- Monotonically decreasing intervals
    :=
by sorry

end function_properties_l104_10443
