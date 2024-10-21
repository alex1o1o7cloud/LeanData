import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_model_l1307_130752

/-- The doubling time of the bacteria population in minutes -/
def doubling_time : ℝ := 3

/-- The final population of bacteria -/
def final_population : ℝ := 500000

/-- The time taken for the population to reach the final population in minutes -/
def time_taken : ℝ := 26.897352853986263

/-- The initial population of bacteria -/
def initial_population : ℝ := 1010

/-- Theorem stating that the given initial population grows to the final population
    in the specified time, given the doubling time -/
theorem bacteria_growth_model :
  abs (final_population - initial_population * (2 ^ (time_taken / doubling_time))) < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bacteria_growth_model_l1307_130752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_pentagon_circumradius_l1307_130775

/-- A cyclic pentagon in 2D Euclidean space -/
def CyclicPentagon (A B C D E : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

/-- The circumradius of a cyclic pentagon -/
noncomputable def circumradius (A B C D E : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

/-- A cyclic pentagon with specific side lengths has a circumradius of 13 -/
theorem cyclic_pentagon_circumradius (A B C D E : EuclideanSpace ℝ (Fin 2)) :
  CyclicPentagon A B C D E →
  dist A B = 5 →
  dist B C = 5 →
  dist C D = 12 →
  dist D E = 12 →
  dist A E = 14 →
  circumradius A B C D E = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclic_pentagon_circumradius_l1307_130775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_around_cylinder_l1307_130750

def cylinder_circumference : ℝ := 6
def cylinder_height : ℝ := 18
def num_loops : ℕ := 3

theorem string_length_around_cylinder :
  let string_length := (num_loops : ℝ) * (cylinder_height / num_loops) * Real.sqrt 2
  string_length = 18 * Real.sqrt 2 := by
  -- Proof steps would go here
  sorry

#check string_length_around_cylinder

end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_around_cylinder_l1307_130750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mod_two_eq_one_iff_power_of_two_l1307_130760

noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => λ _ => 1
  | 1 => λ _ => 1
  | (n + 2) => λ x => f (n + 1) x + x * f n x

theorem f_mod_two_eq_one_iff_power_of_two (n : ℕ) :
  (∀ x : ℝ, f n x % 2 = 1) ↔ ∃ α : ℕ, n = 2^α := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_mod_two_eq_one_iff_power_of_two_l1307_130760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiple_identity_l1307_130780

theorem matrix_scalar_multiple_identity (A : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ u : Fin 3 → ℝ, A.mulVec u = (3 : ℝ) • u) ↔
  A = ![![3, 0, 0], ![0, 3, 0], ![0, 0, 3]] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiple_identity_l1307_130780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_strips_cover_circle_l1307_130793

structure Strip where
  width : ℝ
  angle : ℝ

def totalWidth (strips : List Strip) : ℝ :=
  strips.map (·.width) |>.sum

theorem strips_cover_circle (strips : List Strip) (h : totalWidth strips = 100) :
  ∃ (translations : List ℝ), 
    let translatedStrips := List.zip strips translations
    ∀ (p : ℝ × ℝ), Real.sqrt (p.1^2 + p.2^2) ≤ 1 → 
      ∃ (s : Strip) (t : ℝ), (s, t) ∈ translatedStrips ∧ 
        ∃ (q : ℝ), |q| ≤ s.width / 2 ∧ 
          p = (t + q * Real.cos s.angle, q * Real.sin s.angle) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_strips_cover_circle_l1307_130793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l1307_130732

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := n - 1

noncomputable def sum_arithmetic_sequence (a : ℕ) : ℝ := (a * (a - 1 : ℝ)) / 2

theorem min_sum_arithmetic_sequence :
  ∀ a : ℕ,
  a > 0 →
  (a^2 : ℝ) - 4*a < 0 →
  (∀ b : ℕ, b > 0 → sum_arithmetic_sequence a ≤ sum_arithmetic_sequence b) →
  (a = 6 ∨ a = 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_arithmetic_sequence_l1307_130732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1307_130731

-- Define the statements p and q
def p (m : ℝ) : Prop := ∃ x > 0, x^2 - 2 * Real.exp 1 * Real.log x ≤ m

def q (m : ℝ) : Prop := 
  ∀ x ≥ 2, ∀ y ≥ x, (1/3: ℝ)^(2*y^2 - m*y + 2) ≤ (1/3 : ℝ)^(2*x^2 - m*x + 2)

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m > 8 ∨ m < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1307_130731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_post_it_area_l1307_130798

/-- Calculates the area of vertically connected post-it notes -/
theorem post_it_area (length width adhesive_length : ℝ) (num_notes : ℕ) :
  length = 9.4 ∧ width = 3.7 ∧ adhesive_length = 0.6 ∧ num_notes = 15 →
  (length + (length - adhesive_length) * (num_notes - 1 : ℝ)) * width = 490.62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_post_it_area_l1307_130798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ballot_box_game_min_boxes_l1307_130724

/-- The ballot box game -/
def BallotBoxGame (n m : ℕ) : Prop :=
  2 ≤ n ∧ 0 < m ∧
  ∃ (strategy : Type), 
    (∀ (b_strategy : Type), ∃ (game_state : Type),
      (∃ (player_move : game_state → strategy → game_state),
       ∃ (opponent_move : game_state → b_strategy → game_state),
       ∃ (win_condition : game_state → Prop),
       ∃ (final_state : game_state), 
         win_condition final_state))

/-- The minimum number of boxes for A to guarantee a win -/
def MinBoxes (n : ℕ) : ℕ :=
  2^(n-1) + 1

/-- Theorem: The minimum number of boxes for A to guarantee a win is 2^(n-1) + 1 -/
theorem ballot_box_game_min_boxes (n : ℕ) (h : 2 ≤ n) :
  (∀ m : ℕ, BallotBoxGame n m ↔ MinBoxes n ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ballot_box_game_min_boxes_l1307_130724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l1307_130723

-- Problem 1
theorem problem_1 : Real.sqrt (Real.sqrt 27) - (Real.sqrt 2 * Real.sqrt 6) / Real.sqrt 3 = 1 := by sorry

-- Problem 2
theorem problem_2 : 2 * Real.sqrt 32 - Real.sqrt 50 = 3 * Real.sqrt 2 := by sorry

-- Problem 3
theorem problem_3 : Real.sqrt 12 - Real.sqrt 8 + Real.sqrt (4/3) + 2 * Real.sqrt (1/2) = 8 * Real.sqrt 3 / 3 - Real.sqrt 2 := by sorry

-- Problem 4
theorem problem_4 : Real.sqrt 48 + Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 = 5 * Real.sqrt 3 + Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_problem_3_problem_4_l1307_130723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1307_130715

/-- The function f(x) = x^3 + 2x - 1 -/
def f (x : ℝ) : ℝ := x^3 + 2*x - 1

/-- Theorem stating that if f has a root in (k, k+1), then k = 0 -/
theorem root_in_interval : ∃ k : ℤ, (∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) → k = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_interval_l1307_130715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l1307_130733

def t : ℕ → ℚ
  | 0 => 1  -- Add a case for 0
  | 1 => 1
  | n + 2 => if Even (n + 2) then 1 + t ((n + 2) / 2) else 1 / t (n + 1)

theorem sequence_value (n : ℕ) (h : n > 0) : t n = 19 / 87 → n = 1905 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l1307_130733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_work_theorem_l1307_130797

/-- The work done to stretch a spring -/
noncomputable def work_done (force : ℝ) (displacement : ℝ) : ℝ :=
  0.5 * force * displacement

/-- The spring constant -/
noncomputable def spring_constant (force : ℝ) (displacement : ℝ) : ℝ :=
  force / displacement

theorem spring_work_theorem (initial_force : ℝ) (initial_displacement : ℝ) (final_displacement : ℝ) :
  initial_force > 0 →
  initial_displacement > 0 →
  final_displacement > 0 →
  initial_force * initial_displacement = 1 →
  work_done (spring_constant initial_force initial_displacement * final_displacement) final_displacement = 0.18 := by
  sorry

#check spring_work_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spring_work_theorem_l1307_130797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_with_perpendicular_chords_l1307_130755

-- Define the circle and its properties
structure Circle where
  center : ℝ × ℝ
  chordAB : Set (ℝ × ℝ)
  chordCD : Set (ℝ × ℝ)
  P : ℝ × ℝ
  perpendicular : Prop
  AP : ℝ
  BP : ℝ
  CD : ℝ

-- Define the theorem
theorem circle_area_with_perpendicular_chords
  (c : Circle)
  (h1 : c.perpendicular)
  (h2 : c.AP = 6)
  (h3 : c.BP = 12)
  (h4 : c.CD = 22) :
  ∃ (area : ℝ), area = 130 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_with_perpendicular_chords_l1307_130755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1307_130705

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x - Real.pi / 6) + 1

theorem f_properties :
  (∀ x, f x ≤ 3) ∧
  (∀ x, f (x + Real.pi/2) = f x) ∧
  (∀ x ∈ Set.Icc (Real.pi/3 : ℝ) (5*Real.pi/6), ∀ y ∈ Set.Icc (Real.pi/3 : ℝ) (5*Real.pi/6), x < y → f x > f y) ∧
  (∀ a ∈ Set.Ioo (0 : ℝ) (Real.pi/2), f (a/2) = 2 → a = Real.pi/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1307_130705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_not_divisible_by_seven_l1307_130702

def x : ℕ → ℕ
  | 0 => 2022  -- Add this case to handle Nat.zero
  | 1 => 2022
  | n + 1 => 7 * x n + 5

def not_divisible_by_seven (n m : ℕ) : Prop :=
  ∀ k : ℕ, k ≤ m → (x n - k + 1) % 7 ≠ 0

theorem max_m_not_divisible_by_seven :
  (∀ n : ℕ, n > 0 → not_divisible_by_seven n 404) ∧
  ∀ m : ℕ, m > 404 → ∃ n : ℕ, n > 0 ∧ ¬(not_divisible_by_seven n m) :=
by
  sorry  -- Use 'by sorry' to skip the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_not_divisible_by_seven_l1307_130702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_events_l1307_130748

-- Define the events
def event1 : Prop := ∀ x : ℝ, x^2 < 0
def event2 : Prop := ∃ t : Real, t = 180 -- Changed from Triangle to Real
def event3 : Prop := ∃ p : Nat, p > 0 -- Changed from Person to Nat
def event4 : Prop := ∃ p : Nat, p > 0 -- Changed from Person to Nat

-- Define a predicate for random events
def isRandom (e : Prop) : Prop := sorry

-- Theorem to prove
theorem random_events :
  isRandom event3 ∧ isRandom event4 := by
  sorry

#check random_events

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_events_l1307_130748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_polar_equation_l1307_130735

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := sorry

-- Define the point P as the intersection of circle C and the y-axis
def point_P : ℝ × ℝ := sorry

-- Define the polar coordinate system
def polar_coord_system : ℝ × ℝ → ℝ × ℝ := sorry

-- Define the tangent line to circle C passing through point P
def tangent_line : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem tangent_line_polar_equation :
  ∀ ρ θ : ℝ, (ρ, θ) ∈ polar_coord_system '' tangent_line ↔ ρ * Real.cos θ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_polar_equation_l1307_130735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_3_l1307_130783

def sequence_a : ℕ → ℤ
  | 0 => 3  -- We define a_0 as 3 to match a_1 in the problem
  | 1 => 6  -- This matches a_2 in the problem
  | (n + 2) => sequence_a (n + 1) - sequence_a n

theorem a_2016_equals_negative_3 : sequence_a 2015 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_negative_3_l1307_130783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_alpha_l1307_130791

noncomputable def f (x : ℝ) (α : ℝ) : ℝ :=
  if x > 0 then
    x^2 + Real.sin (x + Real.pi/3)
  else if x < 0 then
    -x^2 + Real.cos (x + α)
  else
    0

theorem odd_function_alpha (α : ℝ) :
  (α ≥ 0 ∧ α < 2*Real.pi) →
  (∀ x, f (-x) α = -f x α) →
  α = 5*Real.pi/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_alpha_l1307_130791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_from_three_l1307_130796

/-- A finite set with exactly three elements -/
def ThreeSet (α : Type) : Type := { s : Finset α // s.card = 3 }

/-- The probability of selecting a specific element from a set of three elements is 1/3 -/
theorem prob_select_from_three {α : Type} [DecidableEq α] (s : ThreeSet α) (x : α) (h : x ∈ s.val) :
  (1 : ℚ) / 3 = (Finset.filter (· = x) s.val).card / s.val.card :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_from_three_l1307_130796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l1307_130714

-- Define the ellipse C
def ellipse_C (a b : ℝ) (h : a > b ∧ b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the parabola
def parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the line passing through A and B
def line_AB (a b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | b * p.1 - a * p.2 - a * b = 0}

-- Define the moving line l
def line_l (k m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + m}

-- Define the theorem
theorem ellipse_and_line_theorem (a b : ℝ) (h : a > b ∧ b > 0) :
  -- Right focus of ellipse C coincides with focus of parabola
  ((a^2 - b^2).sqrt, 0) ∈ parabola →
  -- Distance from origin to line AB is 2√21/7
  (abs (a * b) / Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 21 / 7) →
  -- For any k and m such that line l intersects ellipse C at exactly one point
  ∀ k m : ℝ, (∃! p, p ∈ ellipse_C a b h ∧ p ∈ line_l k m) →
  -- The equation of ellipse C is x^2/4 + y^2/3 = 1
  (ellipse_C a b h = {p : ℝ × ℝ | (p.1^2 / 4) + (p.2^2 / 3) = 1}) ∧
  -- Point Q lies on the fixed line x = 4
  (∃ Q : ℝ × ℝ, Q.1 = 4 ∧ Q ∈ line_l k m ∧
    -- Q is on the perpendicular line to PF₁
    (Q.2 - 0) / (Q.1 - 1) = -(((k * Q.1 + m) - 0) / (Q.1 - 1))⁻¹) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l1307_130714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_cannot_reach_B_l1307_130792

/-- Represents a cyclist with a given speed -/
structure Cyclist where
  speed : ℝ

/-- Represents the problem setup -/
structure ProblemSetup where
  cyclist1 : Cyclist
  cyclist2 : Cyclist
  totalTime : ℝ
  distanceToB : ℝ

/-- Calculates the total distance traveled by both cyclists -/
noncomputable def totalDistanceTraveled (setup : ProblemSetup) : ℝ :=
  let ratio := setup.cyclist1.speed / setup.cyclist2.speed
  let time1 := setup.totalTime * ratio / (1 + ratio)
  let time2 := setup.totalTime - time1
  setup.cyclist1.speed * time1 + setup.cyclist2.speed * time2

/-- The main theorem stating that the cyclists cannot reach point B -/
theorem cyclists_cannot_reach_B (setup : ProblemSetup) 
    (h1 : setup.cyclist1.speed = 35)
    (h2 : setup.cyclist2.speed = 25)
    (h3 : setup.totalTime = 2)
    (h4 : setup.distanceToB = 30) : 
    totalDistanceTraveled setup < setup.distanceToB := by
  sorry

-- Example usage (commented out to avoid evaluation issues)
-- #eval ProblemSetup ⟨Cyclist.mk 35, Cyclist.mk 25, 2, 30⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cyclists_cannot_reach_B_l1307_130792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l1307_130758

theorem power_function_theorem (m : ℤ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x ^ ((m + 1) * (m - 2))) →
  (f 3 > f 5) →
  (∀ x : ℝ, f x = x ^ ((-2) : ℤ)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_theorem_l1307_130758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1307_130772

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x + 3) + x^2

-- State the theorem
theorem f_properties :
  -- f is increasing on [0, 1]
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y) ∧
  -- Maximum value of f on [0, 1] is ln 5 + 1
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x ≤ f 1) ∧
  (f 1 = Real.log 5 + 1) ∧
  -- Minimum value of f on [0, 1] is ln 3
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f 0 ≤ f x) ∧
  (f 0 = Real.log 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1307_130772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l1307_130774

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ) - 1

theorem phi_range (ω φ : ℝ) :
  (∀ x, f ω φ (x + 4 * π) = f ω φ x) →
  (∀ y, y > 0 → y < 4 * π → ¬(∀ x, f ω φ (x + y) = f ω φ x)) →
  (∃ x₁ x₂ x₃, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ ≤ 5 * π ∧
    f ω φ x₁ = 0 ∧ f ω φ x₂ = 0 ∧ f ω φ x₃ = 0) →
  (∀ x, 0 ≤ x ∧ x ≤ 5 * π → f ω φ x = 0 →
    x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (φ ∈ Set.Icc 0 (π / 6) ∪ Set.Icc (π / 3) (π / 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_range_l1307_130774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_theorem_l1307_130785

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Sum of squares of divisors -/
def sum_squares_divisors (N : ℕ) : ℕ :=
  (Finset.filter (· ∣ N) (Finset.range (N + 1))).sum (λ d => d * d)

theorem fibonacci_product_theorem (N : ℕ) (h : N > 0) 
    (h_sum : sum_squares_divisors N = N * (N + 3)) :
  ∃ i j : ℕ, N = fib i * fib j := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_theorem_l1307_130785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_in_fourth_quadrant_l1307_130727

theorem conjugate_in_fourth_quadrant (z : ℂ) (h : Complex.I * z = -1 + Complex.I) :
  0 < z.re ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conjugate_in_fourth_quadrant_l1307_130727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_side_significant_digits_l1307_130757

/-- The number of significant digits in a real number -/
def significantDigits (x : ℝ) : ℕ := sorry

/-- The area of the square in square inches -/
def squareArea : ℝ := 0.3600

/-- The precision of the area measurement in square inches -/
def areaPrecision : ℝ := 0.0001

/-- The side length of the square in inches -/
noncomputable def squareSide : ℝ := Real.sqrt squareArea

/-- Theorem stating that the side length has 4 significant digits -/
theorem side_significant_digits :
  significantDigits squareSide = 4 := by sorry

#eval squareArea
#eval areaPrecision

end NUMINAMATH_CALUDE_ERRORFEEDBACK_side_significant_digits_l1307_130757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_distinct_numbers_exist_l1307_130729

/-- A two-digit number represented as a pair of digits -/
def TwoDigitNumber := Fin 10 × Fin 10

/-- A set of 51 distinct two-digit numbers -/
def NumberSet := Finset TwoDigitNumber

/-- Predicate to check if two numbers share a digit in any position -/
def SharesDigit (a b : TwoDigitNumber) : Prop :=
  a.1 = b.1 ∨ a.2 = b.2

theorem six_distinct_numbers_exist (S : NumberSet) 
  (h : S.card = 51) :
  ∃ (T : Finset TwoDigitNumber), T ⊆ S ∧ T.card = 6 ∧
    ∀ (a b : TwoDigitNumber), a ∈ T → b ∈ T → a ≠ b → ¬(SharesDigit a b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_distinct_numbers_exist_l1307_130729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1307_130734

theorem range_of_a (a : ℝ) :
  ((a + 1)^(-(1/3 : ℝ)) < (3 - 2*a)^(-(1/3 : ℝ))) ↔ (2/3 < a ∧ a < 3/2) ∨ (a < -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1307_130734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_translation_phi_range_a_range_for_three_integer_solutions_l1307_130707

open Set

noncomputable section

variable (a x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a^2 * x^2

def φ (a : ℝ) (x : ℝ) : ℝ := a^2 * (x - 1)^2

theorem function_translation (h : a > 0) :
  φ a x = f a (x - 1) :=
by sorry

theorem phi_range (h : a > 0) :
  range (φ a) = Ici 0 :=
by sorry

theorem a_range_for_three_integer_solutions :
  {a : ℝ | a > 0 ∧ (∃ s : Finset ℤ, s.card = 3 ∧ ∀ x : ℤ, x ∈ s ↔ (x - 1)^2 > f a x)} = 
  Icc (4/3) (3/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_translation_phi_range_a_range_for_three_integer_solutions_l1307_130707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_subset_iff_property_P_l1307_130763

/-- The set Sn = {1, 2, 3, ..., 2n} where n ∈ ℕ+, n ≥ 4 -/
def Sn (n : ℕ) : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2*n}

/-- Definition of an "expected subset" -/
def is_expected_subset (n : ℕ) (A : Set ℕ) : Prop :=
  ∃ a b c, a ∈ Sn n ∧ b ∈ Sn n ∧ c ∈ Sn n ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a + b) ∈ A ∧ (b + c) ∈ A ∧ (c + a) ∈ A

/-- Definition of property P -/
def has_property_P (A : Set ℕ) : Prop :=
  ∃ x y z, x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ 
    x < y ∧ y < z ∧ x + y > z ∧ Even (x + y + z)

/-- Main theorem: A is an "expected subset" iff it has property P -/
theorem expected_subset_iff_property_P (n : ℕ) (hn : n ≥ 4) (A : Set ℕ) :
  is_expected_subset n A ↔ has_property_P A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_subset_iff_property_P_l1307_130763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_problem_l1307_130738

/-- A random variable X that takes on values 1, 2, 3, ..., n with equal probability -/
def X (n : ℕ) := Fin n

/-- The probability of X being less than 4 -/
noncomputable def prob_X_less_than_4 (n : ℕ) : ℝ := 3 / n

/-- Theorem: If X is a random variable taking values 1, 2, 3, ..., n with equal probability,
    and P(X < 4) = 0.3, then n = 10 -/
theorem random_variable_problem (n : ℕ) (h : prob_X_less_than_4 n = 0.3) : n = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_problem_l1307_130738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1307_130777

noncomputable def angle_pi_fifth : Real := Real.pi / 5

noncomputable def circle_radius : Real := 6

noncomputable def central_angle_degrees : Real := 15

noncomputable def sector_area (r : Real) (θ : Real) : Real := (1 / 2) * r^2 * θ

def positive_correlation_definition : String := 
  "Distribution of points in the lower-left to upper-right region of a scatter plot"

def number_of_correct_statements : Nat := 1

theorem problem_statement : number_of_correct_statements = 1 :=
  by
    have h1 : Set.Infinite {α : Real | ∃ k : Int, α = angle_pi_fifth + 2 * Real.pi * k} := by sorry
    have h2 : sector_area circle_radius (central_angle_degrees * Real.pi / 180) = (3 * Real.pi) / 2 := by sorry
    have h3 : positive_correlation_definition ≠ 
      "Distribution of points in the upper-left to lower-right region of a scatter plot" := by sorry
    have h4 : Real.cos (260 * Real.pi / 180) < 0 := by sorry

    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1307_130777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equivalence_l1307_130753

def PositiveIntegers : Set ℕ := {n : ℕ | n > 0}

def M : Set ℕ := {x ∈ PositiveIntegers | x ≤ 2}

theorem M_equivalence : M = {1, 2} ∧ {1, 2} ⊆ M := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equivalence_l1307_130753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_two_digit_number_satisfying_conditions_l1307_130765

theorem max_two_digit_number_satisfying_conditions : ℕ := by
  -- Define the set of numbers satisfying all conditions
  let satisfies_conditions (n : ℕ) : Prop :=
    10 ≤ n ∧ n < 100 ∧
    1000 ≤ n * 109 ∧ n * 109 < 10000 ∧
    n % 23 = 0 ∧
    1 ≤ n / 23 ∧ n / 23 < 10

  -- State that 69 satisfies all conditions
  have h1 : satisfies_conditions 69 := by sorry

  -- State that no number greater than 69 satisfies all conditions
  have h2 : ∀ m : ℕ, m > 69 → ¬satisfies_conditions m := by sorry

  -- Conclude that 69 is the maximum number satisfying all conditions
  exact 69

-- The proof is omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_two_digit_number_satisfying_conditions_l1307_130765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_exists_l1307_130749

/-- Represents a rectangle --/
structure Rectangle where
  width : ℝ
  height : ℝ
  width_pos : width > 0
  height_pos : height > 0

/-- Represents a division of a rectangle into smaller rectangles --/
structure RectangleDivision where
  original : Rectangle
  parts : ℕ
  divisions : List Rectangle

/-- Predicate to check if two rectangles are adjacent and form a larger rectangle --/
def are_adjacent_and_form_rectangle (r1 r2 : Rectangle) : Prop :=
  sorry

/-- Predicate to check if a division is valid according to the problem statement --/
def is_valid_division (d : RectangleDivision) : Prop :=
  d.parts ≥ 5 ∧
  d.divisions.length = d.parts ∧
  ∀ r1 r2, r1 ∈ d.divisions → r2 ∈ d.divisions → r1 ≠ r2 → ¬(are_adjacent_and_form_rectangle r1 r2)

/-- The main theorem to be proved --/
theorem rectangle_division_exists (r : Rectangle) (n : ℕ) (h : n ≥ 5) :
  ∃ (d : RectangleDivision), d.original = r ∧ d.parts = n ∧ is_valid_division d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_division_exists_l1307_130749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_twice_quadrilateral_l1307_130756

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Add convexity condition

-- Define the diagonals of the quadrilateral
def diagonals (q : ConvexQuadrilateral) : (ℝ × ℝ) × (ℝ × ℝ) :=
  (q.vertices 0, q.vertices 2)

-- Define the parallelogram formed by lines parallel to diagonals
noncomputable def formed_parallelogram (q : ConvexQuadrilateral) : ConvexQuadrilateral where
  vertices := sorry
  is_convex := sorry

-- Define the area function for ConvexQuadrilateral
noncomputable def area (q : ConvexQuadrilateral) : ℝ :=
  sorry -- Define area calculation for a quadrilateral

-- Theorem statement
theorem parallelogram_area_twice_quadrilateral 
  (q : ConvexQuadrilateral) : 
  area (formed_parallelogram q) = 2 * area q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_twice_quadrilateral_l1307_130756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fleas_on_board_l1307_130767

/-- Represents a position on the board -/
structure Position where
  x : Fin 10
  y : Fin 10

/-- Represents a direction of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a flea on the board -/
structure Flea where
  position : Position
  direction : Direction

/-- The state of the board at any given time -/
def BoardState := List Flea

/-- Function to update the position of a flea after one jump -/
def jumpFlea (f : Flea) : Flea :=
  sorry

/-- Function to update the entire board state after one minute -/
def updateBoard (state : BoardState) : BoardState :=
  sorry

/-- Predicate to check if two fleas occupy the same position -/
def noCollision (state : BoardState) : Prop :=
  sorry

/-- Main theorem: The maximum number of fleas on a 10x10 board is 40 -/
theorem max_fleas_on_board :
  ∃ (initialState : BoardState),
    (∀ t : ℕ, noCollision (Nat.iterate updateBoard t initialState)) ∧
    initialState.length = 40 ∧
    ∀ (s : BoardState),
      (∀ t : ℕ, noCollision (Nat.iterate updateBoard t s)) →
      s.length ≤ 40 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_fleas_on_board_l1307_130767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_equation_no_solutions_when_negative_one_solution_when_zero_or_geq_one_two_solutions_when_between_zero_and_one_l1307_130712

-- Define the function f(x) = |3^x - 1|
noncomputable def f (x : ℝ) : ℝ := |3^x - 1|

-- Theorem statement
theorem solutions_of_equation (k : ℝ) :
  (∀ x, f x ≠ k) ∨ 
  (∃! x, f x = k) ∨ 
  (∃ x y, x ≠ y ∧ f x = k ∧ f y = k ∧ ∀ z, f z = k → z = x ∨ z = y) :=
by
  sorry

-- No solutions when k < 0
theorem no_solutions_when_negative (k : ℝ) (h : k < 0) :
  ∀ x, f x ≠ k :=
by
  sorry

-- Exactly one solution when k = 0 or k ≥ 1
theorem one_solution_when_zero_or_geq_one (k : ℝ) (h : k = 0 ∨ k ≥ 1) :
  ∃! x, f x = k :=
by
  sorry

-- Exactly two solutions when 0 < k < 1
theorem two_solutions_when_between_zero_and_one (k : ℝ) (h : 0 < k ∧ k < 1) :
  ∃ x y, x ≠ y ∧ f x = k ∧ f y = k ∧ ∀ z, f z = k → z = x ∨ z = y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_equation_no_solutions_when_negative_one_solution_when_zero_or_geq_one_two_solutions_when_between_zero_and_one_l1307_130712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1307_130794

/-- A quadratic function satisfying specific conditions -/
def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧
  (∀ x, f x = a * x^2 + b * x + c) ∧
  (∀ x, f (2 * (-1) - x) = f x) ∧
  f 1 = 1 ∧
  (∃ x₀, ∀ x, f x ≥ f x₀) ∧
  (∃ x₀, f x₀ = 0)

/-- The specific quadratic function we're proving about -/
noncomputable def f (x : ℝ) : ℝ := (1/4) * x^2 + (1/2) * x + (1/4)

/-- The theorem stating our quadratic function satisfies all conditions
    and has the specified property for the largest m -/
theorem quadratic_function_theorem :
  quadratic_function f ∧
  (∃ m : ℝ, m > 1 ∧
    (∀ m' > m, ¬∃ t : ℝ, ∀ x ∈ Set.Icc 1 m', f (x + t) ≤ x) ∧
    (∃ t : ℝ, ∀ x ∈ Set.Icc 1 m, f (x + t) ≤ x) ∧
    m = 9) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1307_130794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l1307_130719

noncomputable def inscribed_circle_radius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s

theorem inscribed_circle_radius_isosceles_triangle (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  let s := (2 * a + b) / 2
  let area := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  area / s = 5 * Real.sqrt 39 / 13 →
    inscribed_circle_radius a a b = 5 * Real.sqrt 39 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l1307_130719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_isosceles_triangles_l1307_130725

/-- Given a quadrilateral ABCD with point M, prove the existence of point N with specific properties -/
theorem quadrilateral_isosceles_triangles
  (z_A z_B z_C z_D z_M : ℂ) :
  (Complex.abs (z_A - z_M) = Complex.abs (z_B - z_M)) →
  (Complex.abs (z_C - z_M) = Complex.abs (z_D - z_M)) →
  (Complex.arg ((z_B - z_M) / (z_A - z_M)) = 2 * Real.pi / 3) →
  (Complex.arg ((z_D - z_M) / (z_C - z_M)) = 2 * Real.pi / 3) →
  ∃ z_N : ℂ,
    (Complex.abs (z_B - z_N) = Complex.abs (z_C - z_N)) ∧
    (Complex.abs (z_D - z_N) = Complex.abs (z_A - z_N)) ∧
    (Complex.arg ((z_C - z_N) / (z_B - z_N)) = 2 * Real.pi / 3) ∧
    (Complex.arg ((z_A - z_N) / (z_D - z_N)) = 2 * Real.pi / 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_isosceles_triangles_l1307_130725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l1307_130726

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

theorem light_path_length 
  (prism : RectangularPrism)
  (E : Point3D)
  (Q : Point3D)
  (h1 : prism.length = 13)
  (h2 : prism.width = 15)
  (h3 : prism.height = 10)
  (h4 : E.x = 0 ∧ E.y = 15 ∧ E.z = 10)
  (h5 : Q.x = 9 ∧ Q.y = 6 ∧ Q.z = 10) :
  ∃ (nearest_vertex : Point3D),
    distance E Q + distance Q nearest_vertex = Real.sqrt 262 := by
  sorry

#check light_path_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_path_length_l1307_130726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1307_130709

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 2
  | (n + 1) => 2 - 1 / sequence_a n

theorem sequence_properties :
  (∀ n : ℕ, ∃ d : ℝ, 1 / (sequence_a (n + 1) - 1) - 1 / (sequence_a n - 1) = d) ∧
  (∀ n : ℕ, sequence_a n > 1) ∧
  (∀ ε > 0, ∃ n₀ : ℕ, ∀ n > n₀, |sequence_a (n + 1) - sequence_a n| < ε) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1307_130709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_income_after_death_l1307_130700

/-- Calculates the new average income after a family member's death -/
theorem new_average_income_after_death (initial_average : ℚ) (initial_members : ℕ) 
  (deceased_income : ℚ) (h1 : initial_average = 735) (h2 : initial_members = 4) 
  (h3 : deceased_income = 990) : 
  let total_income := initial_average * initial_members
  let new_total_income := total_income - deceased_income
  let remaining_members := initial_members - 1
  let new_average_income := new_total_income / remaining_members
  new_average_income = 650 := by
  sorry

#check new_average_income_after_death

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_average_income_after_death_l1307_130700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_unit_direction_vector_l1307_130721

/-- The unit direction vector of a line with slope 2 -/
noncomputable def unit_direction_vector : ℝ × ℝ :=
  (Real.sqrt 5 / 5, 2 * Real.sqrt 5 / 5)

/-- The slope of the line y = 2x + 2 -/
def line_slope : ℝ := 2

theorem line_unit_direction_vector :
  let v := unit_direction_vector
  (v.1^2 + v.2^2 = 1) ∧
  (v.2 / v.1 = line_slope ∨ v.2 / v.1 = -line_slope) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_unit_direction_vector_l1307_130721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1307_130769

def a : ℕ → ℚ
  | 0 => 3/2  -- Added case for n = 0
  | 1 => 3/2
  | n + 1 => (n * (a n - 1)) / (4*n*(n+1)*a n - 4*n^2 - 3*n + 1) + 1

theorem a_formula (n : ℕ) (hn : n > 0) : 
  a n = 1 / (2*n*(2*n-1)) + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l1307_130769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_alpha_l1307_130745

theorem cos_pi_half_plus_alpha (α : ℝ) 
  (h1 : Real.tan α = -3/4) 
  (h2 : α ∈ Set.Ioo (3*π/2) (2*π)) : 
  Real.cos (π/2 + α) = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_half_plus_alpha_l1307_130745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_temperature_l1307_130746

/-- Represents the temperatures for a week --/
structure WeekTemperatures where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ
  friday : ℚ
  saturday : ℚ

/-- The average temperature of three consecutive days --/
def average_temp (a b c : ℚ) : ℚ := (a + b + c) / 3

/-- Theorem stating that under given conditions, the temperature on Tuesday was 80 °C --/
theorem tuesday_temperature (w : WeekTemperatures) 
  (h1 : average_temp w.monday w.tuesday w.wednesday = 38)
  (h2 : average_temp w.tuesday w.wednesday w.thursday = 42)
  (h3 : average_temp w.wednesday w.thursday w.friday = 44)
  (h4 : average_temp w.thursday w.friday w.saturday = 46)
  (h5 : w.friday = 43)
  (h6 : w.saturday = w.monday + 1) : 
  w.tuesday = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tuesday_temperature_l1307_130746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_count_theorem_l1307_130730

/-- Represents the number of marbles of each color --/
structure MarbleCount where
  green : ℕ
  violet : ℕ
  blue : ℕ
deriving Repr

/-- Calculates the final number of marbles after taking away and returning --/
def finalMarbleCount (initial : MarbleCount) (taken : MarbleCount) (returned : MarbleCount) : MarbleCount :=
  { green := initial.green - taken.green + returned.green,
    violet := initial.violet - taken.violet + returned.violet,
    blue := initial.blue - taken.blue + returned.blue }

theorem marble_count_theorem (initial taken returned : MarbleCount) 
  (h_initial : initial = ⟨32, 38, 46⟩)
  (h_taken : taken = ⟨23, 15, 31⟩)
  (h_returned : returned = ⟨10, 8, 17⟩) :
  finalMarbleCount initial taken returned = ⟨19, 31, 32⟩ := by
  sorry

#eval finalMarbleCount ⟨32, 38, 46⟩ ⟨23, 15, 31⟩ ⟨10, 8, 17⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_count_theorem_l1307_130730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_constant_on_largest_interval_l1307_130782

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) + Real.sqrt (x + 24 - 10 * Real.sqrt (x - 1))

-- Define the interval
def I : Set ℝ := Set.Icc 1 26

-- Theorem statement
theorem f_constant_on_largest_interval :
  (∀ x ∈ I, f x = 5) ∧
  (∀ a b, a < 1 ∨ b > 26 → ¬(∀ x ∈ Set.Icc a b, f x = 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_constant_on_largest_interval_l1307_130782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1307_130722

theorem remainder_problem (x : ℕ) (hx : x > 0) (h : 100 % x = 10) : 1000 % x = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l1307_130722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_ends_l1307_130768

def IntegerSet : Finset ℕ := {1, 2, 4, 5, 6, 9, 10, 11, 13}

structure Arrangement where
  squares : List ℕ
  circles : List ℕ
  h_squares_length : squares.length = 5
  h_circles_length : circles.length = 4
  h_all_used : (squares.toFinset ∪ circles.toFinset) = IntegerSet
  h_circle_sum : ∀ i, i < 4 → circles[i]! = squares[i]! + squares[i+1]!

def is_valid_arrangement (arr : Arrangement) : Prop :=
  arr.squares.head?.isSome ∧ arr.squares.getLast?.isSome

theorem max_sum_of_ends (arr : Arrangement) (h_valid : is_valid_arrangement arr) :
  arr.squares[0]! + arr.squares[4]! ≤ 20 := by
  sorry

#eval IntegerSet

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_ends_l1307_130768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_eight_percent_l1307_130718

/-- Calculates the rate of interest given principal, time, and simple interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (simple_interest : ℝ) : ℝ :=
  (simple_interest * 100) / (principal * time)

/-- Theorem stating that given specific values, the interest rate is 8% -/
theorem interest_rate_is_eight_percent 
  (principal : ℝ) 
  (time : ℝ) 
  (simple_interest : ℝ) 
  (h1 : principal = 15041.875)
  (h2 : time = 5)
  (h3 : simple_interest = 6016.75) :
  calculate_interest_rate principal time simple_interest = 8 := by
  sorry

/-- Computes the interest rate for the given values -/
def compute_example : ℚ :=
  (6016.75 * 100) / (15041.875 * 5)

#eval compute_example

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_eight_percent_l1307_130718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l1307_130728

theorem lcm_problem (A B : ℕ) 
  (hcf : Nat.gcd A B = 11)
  (product : A * B = 2310)
  (multiple_of_seven : A % 7 = 0 ∨ B % 7 = 0) :
  Nat.lcm A B = 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_problem_l1307_130728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_comparison_l1307_130771

/-- Represents a square divided into segments -/
structure DividedSquare where
  total_area : ℝ
  shaded_area : ℝ

/-- Square I: divided by diagonals and midlines -/
noncomputable def square_I : DividedSquare :=
  { total_area := 1
    shaded_area := 1/4 }

/-- Square II: divided into 4 smaller squares by connecting midpoints -/
noncomputable def square_II : DividedSquare :=
  { total_area := 1
    shaded_area := 1/2 }

/-- Square III: divided into 16 smaller squares by two perpendicular sets of parallel lines connecting midpoints -/
noncomputable def square_III : DividedSquare :=
  { total_area := 1
    shaded_area := 1/4 }

theorem shaded_area_comparison :
  square_I.shaded_area = square_III.shaded_area ∧
  square_I.shaded_area ≠ square_II.shaded_area ∧
  square_II.shaded_area ≠ square_III.shaded_area :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_comparison_l1307_130771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1307_130744

theorem train_speed_problem (x : ℝ) (h : x > 0) :
  let v := (50 : ℝ) / 11
  (x / v) + (2 * x / 20) = (5 * x / 40) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l1307_130744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l1307_130704

-- Define the centers and radii of the circles
def center1 : ℝ × ℝ := (-3, 2)
def center2 : ℝ × ℝ := (3, -6)
def radius1 : ℝ := 2
def radius2 : ℝ := 8

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)

-- Theorem stating that the circles are externally tangent
theorem circles_externally_tangent :
  distance_between_centers = radius1 + radius2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_externally_tangent_l1307_130704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_sum_average_difference_l1307_130710

theorem consecutive_even_numbers_sum_average_difference :
  ∀ (a b c : ℕ),
  (Even a ∧ Even b ∧ Even c) →
  (b = a + 2 ∧ c = b + 2) →
  (c = 24) →
  (a + b + c) - ((a + b + c) / 3) = 44 :=
by
  intros a b c h1 h2 h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_even_numbers_sum_average_difference_l1307_130710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_number_proof_l1307_130708

theorem other_number_proof (a b : ℕ) (h1 : Nat.lcm a b = 2310) (h2 : Nat.gcd a b = 30) (h3 : a = 462) : b = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_number_proof_l1307_130708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_height_l1307_130720

/-- The height of a right pyramid with a square base -/
noncomputable def pyramid_height (base_perimeter : ℝ) (apex_to_vertex : ℝ) : ℝ :=
  let base_side := base_perimeter / 4
  let base_center_to_corner := base_side * Real.sqrt 2 / 2
  Real.sqrt (apex_to_vertex ^ 2 - base_center_to_corner ^ 2)

/-- Theorem: The height of the specific pyramid is 2√17 inches -/
theorem specific_pyramid_height :
  pyramid_height 32 10 = 2 * Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_height_l1307_130720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l1307_130713

noncomputable def f (x : ℝ) : ℝ := 7 - 8 * x

noncomputable def g (x : ℝ) : ℝ := (7 - x) / 8

theorem f_inverse_is_g : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l1307_130713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1307_130788

/-- Represents a parabola with specific properties -/
structure Parabola where
  -- The equation of the parabola in the form ax^2 + bxy + cy^2 + dx + ey + f = 0
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  -- c is positive
  c_pos : c > 0
  -- gcd of absolute values of coefficients is 1
  gcd_one : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1
  -- The parabola passes through the point (2, 7)
  passes_through : a * 2^2 + b * 2 * 7 + c * 7^2 + d * 2 + e * 7 + f = 0
  -- The y-coordinate of the focus is 5
  focus_y : ℤ
  focus_y_eq : focus_y = 5
  -- The axis of symmetry is parallel to the x-axis
  parallel_to_x : b = 0 ∧ a = 0
  -- The vertex lies on the y-axis
  vertex_on_y : ℤ
  vertex_on_y_eq : vertex_on_y = 5

/-- The theorem stating that the given equation represents the parabola with the specified properties -/
theorem parabola_equation : ∃ (p : Parabola), p.a = 0 ∧ p.b = 0 ∧ p.c = 1 ∧ p.d = -2 ∧ p.e = -10 ∧ p.f = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1307_130788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_river_journey_time_l1307_130764

/-- River journey problem --/
theorem river_journey_time (razumeyevo_distance : ℝ) (vkusnoteevo_downstream : ℝ) 
  (vkusnoteevo_from_river : ℝ) (river_width : ℝ) (current_speed : ℝ) 
  (swim_speed : ℝ) (walk_speed : ℝ) :
  razumeyevo_distance = 3 →
  vkusnoteevo_downstream = 3.25 →
  vkusnoteevo_from_river = 1 →
  river_width = 0.5 →
  current_speed = 1 →
  swim_speed = 2 →
  walk_speed = 4 →
  ∃ (total_time : ℝ), 
    (total_time ≥ 1.49 ∧ total_time ≤ 1.51) ∧ 
    total_time = razumeyevo_distance / walk_speed + 
                 river_width / Real.sqrt (swim_speed^2 - current_speed^2) +
                 vkusnoteevo_from_river / walk_speed :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_river_journey_time_l1307_130764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_2009_all_ones_l1307_130711

theorem multiple_of_2009_all_ones : ∃ n : ℕ, ∃ k : ℕ,
  k * 2009 = (10^n - 1) / 9 ∧ k > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_of_2009_all_ones_l1307_130711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1307_130770

/-- Represents the sum of the first n terms of an arithmetic sequence -/
noncomputable def S (n : ℕ) (a₁ d : ℝ) : ℝ := n / 2 * (2 * a₁ + (n - 1) * d)

/-- 
Given an arithmetic sequence with first term a₁ and common difference d,
if S₈ - S₃ = 10, then S₁₁ = 22
-/
theorem arithmetic_sequence_sum (a₁ d : ℝ) : 
  S 8 a₁ d - S 3 a₁ d = 10 → S 11 a₁ d = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l1307_130770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_and_intersection_sum_l1307_130799

-- Define the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

-- Define the curve C
noncomputable def curve_C (φ : ℝ) : Point where
  x := Real.sqrt 2 * Real.cos φ
  y := 2 * Real.sin φ

-- Define the line l in Cartesian form
def line_l (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y = Real.sqrt 3

-- Define point P
noncomputable def P : Point where
  x := 0
  y := Real.sqrt 3

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem point_on_line_and_intersection_sum :
  (line_l P.x P.y) ∧
  (∃ A B : Point, 
    (∃ φ₁ φ₂ : ℝ, A = curve_C φ₁ ∧ B = curve_C φ₂) ∧
    (line_l A.x A.y) ∧ (line_l B.x B.y) ∧
    (1 / distance P A + 1 / distance P B = Real.sqrt 14)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_and_intersection_sum_l1307_130799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1307_130742

/-- The length of a bridge that a train can cross -/
noncomputable def bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * crossing_time
  total_distance - train_length

/-- Theorem stating the length of the bridge -/
theorem bridge_length_calculation :
  bridge_length 160 45 30 = 215 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Simplify the expression
  simp
  -- Apply numerical approximation
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1307_130742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_lower_bound_l1307_130776

noncomputable def f (a x : ℝ) : ℝ := (x^2 + a*x + 7 + a) / (x + 1)

theorem function_lower_bound (a : ℝ) :
  (∀ x : ℕ+, f a (x : ℝ) ≥ 4) → a ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_lower_bound_l1307_130776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l1307_130751

/-- Given that f(x) = x + b/x is an increasing function on (1,e), prove that b ≤ 1 -/
theorem increasing_function_condition (b : ℝ) : 
  (∀ x ∈ Set.Ioo 1 (Real.exp 1), Monotone (fun x => x + b / x)) → b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l1307_130751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_good_numbers_l1307_130759

def is_good (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∣ n → (d + 1) ∣ (n + 1)

theorem characterization_of_good_numbers :
  ∀ n : ℕ, n ≠ 0 → (is_good n ↔ n = 1 ∨ (Nat.Prime n ∧ Odd n)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_good_numbers_l1307_130759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1307_130754

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x

-- State the theorem
theorem range_of_a (a : ℝ) (h : ∀ x, (deriv f x) ≥ a) : a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1307_130754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_properties_l1307_130789

/-- A random variable with a specific distribution -/
structure RandomVariable where
  m : ℝ
  n : ℝ
  sum_prob : m + 1/2 + n = 1
  exp_value : 0 * m + 1 * 1/2 + 2 * n = 1

/-- The expected value of a function of X -/
noncomputable def E (X : RandomVariable) (f : ℝ → ℝ) : ℝ :=
  f 0 * X.m + f 1 * 1/2 + f 2 * X.n

/-- The variance of X -/
noncomputable def D (X : RandomVariable) : ℝ :=
  E X (fun x ↦ x^2) - (E X id)^2

theorem random_variable_properties (X : RandomVariable) :
  X.n = 1/4 ∧ D X = 1/2 ∧ E X (fun x ↦ 2*x + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_random_variable_properties_l1307_130789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1307_130717

/-- Given a hyperbola with the specified properties, its focal distance is 2√5 -/
theorem hyperbola_focal_distance (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∀ x y, x^2/a^2 - y^2/b^2 = 1) →  -- Hyperbola equation
  c/a = Real.sqrt 5 / 2 →           -- Eccentricity
  a^2 + b^2 = c^2 →                 -- Relation between a, b, and c
  (∃ A B : ℝ × ℝ, 
    let O := (0, 0)
    let F := (c, 0)
    let d := b  -- Distance from F to asymptote
    (A.1 * B.2 = A.2 * B.1) ∧      -- A and B lie on a line through F
    ((F.1 - B.1) * B.1 + (F.2 - B.2) * B.2 = 0) ∧  -- FB perpendicular to OB
    (1/2 * A.1 * B.2 = 8/3)) →     -- Area of triangle OAB
  2*c = 2* Real.sqrt 5 :=           -- Focal distance
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distance_l1307_130717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_lines_l1307_130737

-- Define the ellipse C₁
def C₁ (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the focus point
def leftFocus : ℝ × ℝ := (-1, 0)

-- Define the minimum distance from a point on the ellipse to the focus
noncomputable def minDistance : ℝ := Real.sqrt 2 - 1

-- Define the point through which the tangent line passes
noncomputable def tangentPoint : ℝ × ℝ := (0, Real.sqrt 2)

-- Theorem statement
theorem ellipse_and_tangent_lines 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : ∀ x y, C₁ x y a b → (x + 1)^2 + y^2 ≥ minDistance^2) 
  (h4 : ∃ x y, C₁ x y a b ∧ (x + 1)^2 + y^2 = minDistance^2) :
  (∀ x y, C₁ x y (Real.sqrt 2) 1) ∧ 
  (∀ x y, (x - Real.sqrt 2 * y + 2 = 0 ∨ x + Real.sqrt 2 * y + 2 = 0) → 
    (∃ t, C₁ t y (Real.sqrt 2) 1 ∧ y = (Real.sqrt 2 * t + Real.sqrt 2))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_lines_l1307_130737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1307_130736

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 - 2*x - x^2)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1307_130736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_zero_l1307_130762

-- Define the function f as noncomputable
noncomputable def f (t : ℝ) : ℝ := ((t - 1) / 2) ^ 2

-- State the theorem
theorem f_of_three_equals_zero :
  (∀ x : ℝ, f (2 * x + 1) = x^2 - 2*x + 1) → f 3 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_zero_l1307_130762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_435_6_base_nine_l1307_130790

/-- Represents a number in base 9 --/
structure BaseNine where
  value : ℕ
  valid : value < 9^64 := by sorry

/-- Converts a base 9 number to its decimal (base 10) representation --/
def to_decimal (n : BaseNine) : ℕ := n.value

/-- Converts a decimal (base 10) number to its base 9 representation --/
def to_base_nine (n : ℕ) : BaseNine :=
  ⟨n % 9^64, by sorry⟩

instance : OfNat BaseNine n where
  ofNat := to_base_nine n

/-- Multiplies two base 9 numbers --/
def mul_base_nine (a b : BaseNine) : BaseNine :=
  to_base_nine (to_decimal a * to_decimal b)

theorem product_435_6_base_nine :
  mul_base_nine 435 6 = 2863 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_435_6_base_nine_l1307_130790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l1307_130761

/-- Apply a translation to a point in ℝ² -/
def translate (p : ℝ × ℝ) (dx dy : ℝ) : ℝ × ℝ :=
  (p.1 + dx, p.2 + dy)

theorem midpoint_after_translation (B G : ℝ × ℝ) :
  B = (1, 2) →
  G = (6, 2) →
  let B' := translate B (-4) (-3)
  let G' := translate G (-4) (-3)
  (B'.1 + G'.1) / 2 = -0.5 ∧ (B'.2 + G'.2) / 2 = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_after_translation_l1307_130761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_partition_with_continuous_function_l1307_130703

theorem no_partition_with_continuous_function : ¬∃ (A B : Set ℝ) (f : ℝ → ℝ),
  (A ∪ B = Set.Icc 0 1) ∧
  (A ∩ B = ∅) ∧
  (Continuous f) ∧
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc 0 1) ∧
  (∀ a ∈ A, f a ∈ B) ∧
  (∀ b ∈ B, f b ∈ A) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_partition_with_continuous_function_l1307_130703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_sculpture_volume_l1307_130740

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The total volume of three spheres with radii 4, 6, and 8 inches -/
noncomputable def total_volume : ℝ :=
  sphere_volume 4 + sphere_volume 6 + sphere_volume 8

theorem snow_sculpture_volume :
  total_volume = 1056 * Real.pi := by
  -- Unfold the definitions
  unfold total_volume sphere_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_sculpture_volume_l1307_130740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_days_2010_to_2015_l1307_130743

def is_leap_year (year : ℕ) : Bool :=
  year % 4 = 0 && (year % 100 ≠ 0 || year % 400 = 0)

def days_in_year (year : ℕ) : ℕ :=
  if is_leap_year year then 366 else 365

def total_days (start_year end_year : ℕ) : ℕ :=
  (List.range (end_year - start_year + 1)).map (fun i => days_in_year (start_year + i)) |>.sum

theorem total_days_2010_to_2015 :
  total_days 2010 2015 = 2191 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_days_2010_to_2015_l1307_130743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dressing_p_percentage_l1307_130706

/-- Represents a salad dressing with vinegar and oil percentages -/
structure Dressing where
  vinegar : ℚ
  oil : ℚ
  sum_to_100 : vinegar + oil = 100

/-- The percentage of dressing P in the new mixture -/
noncomputable def percentage_p (p q : Dressing) (new_vinegar_percentage : ℚ) : ℚ :=
  100 * (new_vinegar_percentage - q.vinegar) / (p.vinegar - q.vinegar)

/-- Theorem stating that the percentage of dressing P in the new mixture is 10% -/
theorem dressing_p_percentage
  (p : Dressing)
  (q : Dressing)
  (h_p : p.vinegar = 30)
  (h_q : q.vinegar = 10)
  (h_new : new_vinegar_percentage = 12) :
  percentage_p p q new_vinegar_percentage = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dressing_p_percentage_l1307_130706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_parallelism_conditions_plane_parallelism_conditions_reverse_l1307_130741

-- Define the type for planes
variable (Plane : Type)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (line_parallel_to_plane : Plane → Plane → Prop)

-- Define the property of two lines being non-parallel in a plane
variable (non_parallel_lines_in_plane : Plane → Prop)

-- Theorem statement
theorem plane_parallelism_conditions (α β : Plane) :
  (∃ γ : Plane, parallel γ α ∧ parallel γ β) →
  (non_parallel_lines_in_plane α ∧ line_parallel_to_plane α β) →
  parallel α β := by
  sorry

-- Reverse implication
theorem plane_parallelism_conditions_reverse (α β : Plane) :
  parallel α β →
  ((∃ γ : Plane, parallel γ α ∧ parallel γ β) ∧
   (non_parallel_lines_in_plane α ∧ line_parallel_to_plane α β)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_parallelism_conditions_plane_parallelism_conditions_reverse_l1307_130741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_k_range_l1307_130784

/-- Predicate to check if a line is tangent to a circle at a point -/
def IsTangentLine (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) (point : ℝ × ℝ) : Prop :=
  sorry

/-- The range of k values for which a line through (-1, 0) is tangent to the circle x^2 + y^2 + 2kx + 4y + 3k + 8 = 0 -/
theorem tangent_line_k_range :
  ∀ k : ℝ,
  (∃ m : ℝ, IsTangentLine (λ x y ↦ y = m * (x + 1)) 
    (λ x y ↦ x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0) (-1, 0))
  ↔ k ∈ Set.Ioo (-9) (-1) ∪ Set.Ioi 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_k_range_l1307_130784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_intersection_l1307_130701

-- Define the concepts of line, plane, and point
variable (Line Plane Point : Type)

-- Define the relation of a point being on a line
variable (on_line : Point → Line → Prop)

-- Define the relation of a point being outside a plane
variable (outside_plane : Point → Plane → Prop)

-- Define the concept of infinitely many points
variable (infinitely_many : (Point → Prop) → Prop)

-- Theorem statement
theorem line_plane_intersection 
  (l : Line) (p : Plane) :
  (∃ x : Point, on_line x l ∧ outside_plane x p) →
  infinitely_many (fun y ↦ on_line y l ∧ outside_plane y p) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_intersection_l1307_130701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_sum_l1307_130795

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the problem parameters
structure ProblemParams where
  intersection_point : ℝ × ℝ
  radii_product : ℝ
  n : ℝ
  d : ℕ
  e : ℕ
  f : ℕ

-- Define the conditions
def conditions (params : ProblemParams) (C₁ C₂ : Circle) : Prop :=
  -- Circles intersect at (10, 8)
  (10, 8) ∈ Set.inter {p | dist p C₁.center = C₁.radius} {p | dist p C₂.center = C₂.radius} ∧
  -- Product of radii is 50
  C₁.radius * C₂.radius = 50 ∧
  -- y-axis is tangent to both circles
  C₁.center.1 = C₁.radius ∧ C₂.center.1 = C₂.radius ∧
  -- Line y = nx is tangent to both circles
  (C₁.center.2 = params.n * C₁.center.1 + C₁.radius * Real.sqrt (1 + params.n^2)) ∧
  (C₂.center.2 = params.n * C₂.center.1 + C₂.radius * Real.sqrt (1 + params.n^2)) ∧
  -- n = d√e/f
  params.n = (params.d : ℝ) * Real.sqrt (params.e : ℝ) / (params.f : ℝ) ∧
  -- d, e, f are positive integers
  params.d > 0 ∧ params.e > 0 ∧ params.f > 0 ∧
  -- e is not divisible by the square of any prime
  ∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ params.e) ∧
  -- d and f are relatively prime
  Nat.Coprime params.d params.f

-- State the theorem
theorem circle_intersection_sum (params : ProblemParams) (C₁ C₂ : Circle) :
  conditions params C₁ C₂ → params.d + params.e + params.f = 24 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_sum_l1307_130795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_with_interior_point_l1307_130747

-- Define a triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define an interior point M
def InteriorPoint (t : Triangle) (M : ℝ × ℝ) : Prop :=
  ∃ (α β γ : ℝ), α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = 1 ∧
    M = (α * t.A.1 + β * t.B.1 + γ * t.C.1, α * t.A.2 + β * t.B.2 + γ * t.C.2)

-- Define distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem triangle_inequality_with_interior_point (t : Triangle) (M : ℝ × ℝ) 
  (h : InteriorPoint t M) :
  min (distance M t.A) (min (distance M t.B) (distance M t.C)) + 
  distance M t.A + distance M t.B + distance M t.C < 
  distance t.A t.B + distance t.B t.C + distance t.C t.A := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_with_interior_point_l1307_130747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_satisfies_conditions_l1307_130773

/-- The polynomial p(x) that satisfies the given conditions -/
noncomputable def p (x : ℝ) : ℝ := 4 * x^2 - 8 * x - 12

/-- The rational function formed by the given numerator and p(x) -/
noncomputable def f (x : ℝ) : ℝ := (x^3 - 3*x^2 - 4*x + 12) / p x

theorem p_satisfies_conditions :
  (∀ x, x ≠ 3 ∧ x ≠ -1 → f x ≠ 0) ∧  -- Vertical asymptotes at 3 and -1
  (¬ ∃ L, ∀ ε > 0, ∃ M, ∀ x, abs x > M → abs (f x - L) < ε) ∧  -- No horizontal asymptote
  p 4 = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_satisfies_conditions_l1307_130773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1307_130779

/-- Calculates the speed of a train given its length and time to pass a stationary point -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  length / time

/-- Theorem: A train with length 180 meters that passes a stationary point in 12 seconds has a speed of 15 meters per second -/
theorem train_speed_calculation :
  train_speed 180 12 = 15 := by
  -- Unfold the definition of train_speed
  unfold train_speed
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1307_130779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1307_130766

/-- A parallelogram with vertices at (1, 2), (7, 2), (5, 10), and (11, 10) has an area of 48 square units. -/
theorem parallelogram_area : 
  let v1 : ℝ × ℝ := (1, 2)
  let v2 : ℝ × ℝ := (7, 2)
  let v3 : ℝ × ℝ := (5, 10)
  let v4 : ℝ × ℝ := (11, 10)
  let base : ℝ := v2.1 - v1.1
  let height : ℝ := v3.2 - v1.2
  base * height = 48 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l1307_130766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cube_coloring_l1307_130778

/-- Represents the colors used to paint the cube -/
inductive Color
| Blue
| Red
| Green

/-- Represents a cube with colored sides -/
structure ColoredCube :=
  (top : Color)
  (front : Color)
  (right : Color)
  (bottom : Color)
  (back : Color)
  (left : Color)

/-- Checks if a colored cube is valid according to the problem conditions -/
def isValidCube (cube : ColoredCube) : Prop :=
  cube.top ≠ cube.front ∧
  cube.top ≠ cube.right ∧
  cube.front ≠ cube.right ∧
  cube.top = cube.bottom ∧
  cube.front = cube.back ∧
  cube.right = cube.left

/-- Represents all possible rotations of a cube -/
def CubeRotations : Type := ColoredCube → ColoredCube

/-- Checks if two cubes are equivalent under rotation -/
def areEquivalentCubes (c1 c2 : ColoredCube) : Prop :=
  ∃ (rotation : CubeRotations), rotation c1 = c2

theorem unique_cube_coloring :
  ∃! (cube : ColoredCube), isValidCube cube ∧
    ∀ (other : ColoredCube),
      isValidCube other →
      areEquivalentCubes cube other :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_cube_coloring_l1307_130778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_n_formula_l1307_130781

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := Real.log (1 + x)

-- Define the function g
def g (x : ℝ) : ℝ := x * (deriv f x)

-- Define the recursive function gₙ
def g_n : ℕ → ℝ → ℝ
| 0, x => x
| n + 1, x => g (g_n n x)

-- Theorem statement
theorem g_n_formula (n : ℕ) (x : ℝ) (h : x ≥ 0) :
  g_n (n + 1) x = x / (1 + (n + 1) * x) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_n_formula_l1307_130781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l1307_130716

def number_list : List ℕ := [33, 37, 39, 41, 43]

def is_prime (n : ℕ) : Bool :=
  n > 1 && (Nat.factors n).length == 1

def prime_list : List ℕ := number_list.filter is_prime

theorem arithmetic_mean_of_primes : 
  (prime_list.sum : ℚ) / prime_list.length = 121 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_primes_l1307_130716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_variation_l1307_130787

-- Define the types for points and distances
variable (Point : Type) (Distance : Type) [LinearOrder Distance]

-- Define the functions for distance calculation
variable (distance : Point → Point → Distance)

-- Define the points and radius
variable (X A B C : Point) (r : Distance)

-- Define the path segments
variable (circular_path : Set Point)
variable (triangular_path : Set Point)

-- Define the properties of the circular path
axiom circular_path_center : ∀ P ∈ circular_path, distance X P = r

-- Define the properties of the triangular path
axiom triangular_path_start : B ∈ triangular_path
axiom triangular_path_end : A ∈ triangular_path

-- Define a function to represent the ship's distance from X at any point in its journey
variable (ship_distance : ℝ → Distance)

-- State the theorem
theorem ship_distance_variation :
  ∃ t₁ t₂ : ℝ, t₁ < t₂ ∧
  (∀ t, t ≤ t₁ → ship_distance t = r) ∧
  (∀ t, t₁ < t ∧ t < t₂ → ship_distance t > r) ∧
  (ship_distance t₂ > r) ∧
  (∀ t, t > t₂ → ship_distance t < ship_distance t₂ ∧ ship_distance t > r) ∧
  (∃ t₃, t₃ > t₂ ∧ ship_distance t₃ = r) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_distance_variation_l1307_130787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_touching_spheres_l1307_130739

noncomputable def cone_vertex_angle (r : ℝ) (d : ℝ) : Set ℝ :=
  let φ := Real.arctan (r / d)
  let β := Real.arccos (3 / 5)
  { α | (α = Real.pi / 2) ∨ (α = 2 * Real.arctan (1 / 4)) }

theorem cone_touching_spheres (r d : ℝ) 
  (h_r : r = 4) 
  (h_d : d = 5) : 
  cone_vertex_angle r d = {Real.pi / 2, 2 * Real.arctan (1 / 4)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_touching_spheres_l1307_130739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_time_ratio_l1307_130786

/-- Given a project where three people (Pat, Kate, and Mark) charged time, prove that
    the ratio of Pat's time to Kate's time is 2:1 under specific conditions. -/
theorem project_time_ratio :
  ∀ (p k m : ℕ) (r : ℚ),
  p + k + m = 144 →  -- Total hours charged
  p = r * k →        -- Pat's time is r times Kate's time
  p = m / 3 →        -- Pat's time is 1/3 of Mark's time
  m = k + 80 →       -- Mark charged 80 hours more than Kate
  r = 2 := by        -- The ratio of Pat's time to Kate's time is 2:1
  intro p k m r h1 h2 h3 h4
  sorry

#check project_time_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_time_ratio_l1307_130786
