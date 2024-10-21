import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_proof_l425_42560

-- Define the constants
noncomputable def a : ℝ := 30.5
noncomputable def b : ℝ := Real.log 32 / Real.log 10
noncomputable def c : ℝ := Real.cos 2

-- Theorem statement
theorem ordering_proof : c < b ∧ b < a := by
  -- Split the conjunction into two parts
  constructor
  -- Prove c < b
  · sorry
  -- Prove b < a
  · sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_proof_l425_42560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_in_small_circle_l425_42581

theorem three_points_in_small_circle (points : Finset (ℝ × ℝ)) :
  (points.card = 51) →
  (∀ p ∈ points, p.1 ∈ (Set.Icc 0 1) ∧ p.2 ∈ (Set.Icc 0 1)) →
  ∃ (center : ℝ × ℝ) (r : ℝ), r ≤ 1/7 ∧ 
    ∃ (p1 p2 p3 : ℝ × ℝ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p2 ≠ p3 ∧
    Real.sqrt ((p1.1 - center.1)^2 + (p1.2 - center.2)^2) ≤ r ∧ 
    Real.sqrt ((p2.1 - center.1)^2 + (p2.2 - center.2)^2) ≤ r ∧ 
    Real.sqrt ((p3.1 - center.1)^2 + (p3.2 - center.2)^2) ≤ r :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_in_small_circle_l425_42581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l425_42504

/-- Definition of the line l -/
noncomputable def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y - 2 - k = 0

/-- The fixed point that the line passes through -/
def fixed_point : ℝ × ℝ := (1, -2)

/-- The line does not pass through the second quadrant -/
def not_in_second_quadrant (k : ℝ) : Prop :=
  ∀ x y, line_l k x y → (x ≤ 0 ∧ y ≥ 0 → False)

/-- The line intersects the positive x-axis and negative y-axis -/
def intersects_axes (k : ℝ) : Prop :=
  ∃ x y, x > 0 ∧ y < 0 ∧ line_l k x 0 ∧ line_l k 0 y

/-- The area of triangle AOB -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  let x := (2 + k) / k
  let y := -(2 + k)
  (1 / 2) * abs x * abs y

theorem line_l_properties :
  /- The line passes through the fixed point -/
  (∀ k, line_l k fixed_point.1 fixed_point.2) ∧
  /- If the line does not pass through the second quadrant, then k ∈ [0, +∞) -/
  (∀ k, not_in_second_quadrant k → k ≥ 0) ∧
  /- The minimum area of triangle AOB is 4, occurring when k = 2 -/
  (∀ k, intersects_axes k → triangle_area k ≥ 4) ∧
  (triangle_area 2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l425_42504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_theorem_l425_42596

/-- The time taken for two trains to completely pass each other -/
noncomputable def train_passing_time (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / (speed1 + speed2)

theorem train_passing_theorem :
  let length1 : ℝ := 125
  let length2 : ℝ := 115
  let speed1 : ℝ := 22
  let speed2 : ℝ := 18
  train_passing_time length1 length2 speed1 speed2 = 6 := by
  -- Unfold the definition of train_passing_time
  unfold train_passing_time
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

#eval (125 + 115) / (22 + 18) -- This will output 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_theorem_l425_42596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_critical_points_range_l425_42523

/-- The function f(x) = aᵉˣ - (1/2)x² -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - (1/2) * x^2

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x - x

theorem f_critical_points_range (a : ℝ) (x₁ x₂ : ℝ) :
  (∃ x₁ x₂, f' a x₁ = 0 ∧ f' a x₂ = 0 ∧ x₂ / x₁ ≥ 2) →
  0 < a ∧ a ≤ Real.log 2 / 2 := by
  sorry

#check f_critical_points_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_critical_points_range_l425_42523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_multiplication_equals_exponentiation_l425_42521

theorem repeated_multiplication_equals_exponentiation (a : ℝ) (n : ℕ) :
  (List.replicate n a).prod = a ^ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeated_multiplication_equals_exponentiation_l425_42521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_l425_42591

def P : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}
def M (m : ℝ) : Set ℝ := {x : ℝ | m*x - 1 = 0}

theorem subset_condition : ∀ m : ℝ, M m ⊆ P ↔ m ∈ ({1/2, 1/3, 0} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_l425_42591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l425_42570

theorem trigonometric_identity (α : Real) 
  (h : Real.cos α + Real.sin α = 2/3) : 
  (Real.sqrt 2 * Real.sin (2 * α - π/4) + 1) / (1 + Real.tan α) = -5/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l425_42570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcircle_diameter_l425_42530

theorem triangle_circumcircle_diameter
  (a b c : ℝ)
  (ha : a = 5)
  (hb : b = 8)
  (hc : c = 9)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  let cos_theta := (a^2 + b^2 - c^2) / (2 * a * b)
  let sin_theta := Real.sqrt (1 - cos_theta^2)
  let diameter := c / sin_theta
  diameter = 30 * Real.sqrt 11 / 11 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circumcircle_diameter_l425_42530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_range_l425_42567

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y = 4

-- Define a point on the circle
def point_on_circle_O (x y : ℝ) : Prop := circle_O x y

-- Define a point on the line
def point_on_line_l (x y : ℝ) : Prop := line_l x y

-- Define the angle between three points
noncomputable def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem x_coordinate_range (A B C : ℝ × ℝ) :
  point_on_line_l A.1 A.2 →
  point_on_circle_O B.1 B.2 →
  point_on_circle_O C.1 C.2 →
  angle A B C = 60 →
  0 ≤ A.1 ∧ A.1 ≤ 4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coordinate_range_l425_42567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_solution_l425_42518

theorem inverse_matrices_solution :
  ∀ (a b : ℚ),
  (let M₁ : Matrix (Fin 2) (Fin 2) ℚ := !![4, b; -7, 10];
   let M₂ : Matrix (Fin 2) (Fin 2) ℚ := !![10, a; 7, 4];
   M₁ * M₂ = 1) →
  (a = 39 / 7 ∧ b = -39 / 7) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_matrices_solution_l425_42518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l425_42571

-- Define the function f(x) = ln(x^2 - 2x - 8)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 8)

-- State the theorem
theorem f_monotone_increasing : 
  MonotoneOn f (Set.Ioi 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l425_42571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_l_intersection_range_m_l425_42549

-- Define the curve C
noncomputable def C (t : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

-- Define the polar equation of line l
def l_polar (ρ θ m : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 3) + m = 0

-- Define the Cartesian equation of line l
def l_cartesian (x y m : ℝ) : Prop := Real.sqrt 3 * x + y + 2 * m = 0

-- Theorem 1: Cartesian equation of l
theorem cartesian_equation_of_l (m : ℝ) : 
  ∀ x y ρ θ, (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) → 
  (l_polar ρ θ m ↔ l_cartesian x y m) := by
  sorry

-- Theorem 2: Range of m for intersection
theorem intersection_range_m : 
  ∀ m, (∃ t, l_cartesian (C t).1 (C t).2 m) ↔ 
  (-19/12 : ℝ) ≤ m ∧ m ≤ (5/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cartesian_equation_of_l_intersection_range_m_l425_42549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_passing_time_l425_42544

-- Define the given parameters
noncomputable def mans_train_speed : ℝ := 40  -- kmph
noncomputable def goods_train_speed : ℝ := 72  -- kmph
noncomputable def goods_train_length : ℝ := 280  -- meters

-- Define the function to calculate the time taken
noncomputable def time_to_pass (v1 v2 l : ℝ) : ℝ :=
  l / ((v1 + v2) * (1000 / 3600))

-- State the theorem
theorem goods_train_passing_time :
  ∃ (t : ℝ), abs (t - time_to_pass mans_train_speed goods_train_speed goods_train_length) < 0.5 ∧ 
  t = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goods_train_passing_time_l425_42544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_distribution_theorem_l425_42516

/-- Given a total number of marbles and three ratios, calculates the number of marbles for each person -/
def distributeMarbles (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) : ℕ × ℕ × ℕ :=
  let partValue := total / (ratio1 + ratio2 + ratio3)
  (partValue * ratio1, partValue * ratio2, partValue * ratio3)

/-- Represents the transfer of half of the marbles from one person to another -/
def transferHalf (fromPerson toPerson : ℕ) : ℕ × ℕ :=
  (fromPerson - fromPerson / 2, toPerson + fromPerson / 2)

theorem marble_distribution_theorem :
  let (brittany, alex, jamy) := distributeMarbles 600 3 5 7
  let (_, alexFinal) := transferHalf brittany alex
  alexFinal = 260 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_distribution_theorem_l425_42516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sequence_is_valid_correct_sequence_is_unique_l425_42552

-- Define the steps as an enumeration
inductive EmailStep
  | OpenMailbox
  | EnterRecipient
  | EnterSubject
  | EnterContent
  | ClickCompose
  | ClickSend
deriving DecidableEq, Repr

-- Define a type for a sequence of email steps
def EmailSequence := List EmailStep

-- Define the correct sequence
def correctSequence : EmailSequence :=
  [EmailStep.OpenMailbox, EmailStep.ClickCompose, EmailStep.EnterRecipient,
   EmailStep.EnterSubject, EmailStep.EnterContent, EmailStep.ClickSend]

-- Define a property that checks if a sequence is valid
def isValidSequence (seq : EmailSequence) : Prop :=
  seq.length = 6 ∧ 
  seq.toFinset = {EmailStep.OpenMailbox, EmailStep.EnterRecipient, 
                  EmailStep.EnterSubject, EmailStep.EnterContent, 
                  EmailStep.ClickCompose, EmailStep.ClickSend}

-- Theorem: The correct sequence is valid
theorem correct_sequence_is_valid :
  isValidSequence correctSequence := by
  sorry

-- Theorem: The correct sequence is the only valid sequence
theorem correct_sequence_is_unique :
  ∀ (seq : EmailSequence), isValidSequence seq → seq = correctSequence := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sequence_is_valid_correct_sequence_is_unique_l425_42552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_powers_1000_l425_42556

def count_special_powers (n : ℕ) : ℕ :=
  (Finset.filter (fun m => m^2 < n ∧ m^2 + 1 ≥ n) (Finset.range n)).card +
  (Finset.filter (fun m => m^3 < n ∧ m^3 + 1 ≥ n) (Finset.range n)).card +
  (Finset.filter (fun m => m^4 < n ∧ m^4 + 1 ≥ n) (Finset.range n)).card -
  (Finset.filter (fun m => m^6 < n ∧ m^6 + 1 ≥ n) (Finset.range n)).card -
  (Finset.filter (fun m => m^8 < n ∧ m^8 + 1 ≥ n) (Finset.range n)).card

theorem count_special_powers_1000 : count_special_powers 1000 = 41 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_powers_1000_l425_42556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_speed_calculation_l425_42550

theorem projectile_speed_calculation 
  (initial_distance : ℝ) 
  (speed1 : ℝ) 
  (time_minutes : ℝ) 
  (speed2 : ℝ) : 
  initial_distance = 1182 ∧ 
  speed1 = 460 ∧ 
  time_minutes = 72 ∧ 
  speed2 * (time_minutes / 60) + speed1 * (time_minutes / 60) = initial_distance → 
  speed2 = 525 := by
  intro h
  -- The proof steps would go here
  sorry

#check projectile_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projectile_speed_calculation_l425_42550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_normal_vectors_l425_42588

def normal_vector_α : ℝ × ℝ × ℝ := (2, -1, 1)
def normal_vector_β : ℝ × ℝ × ℝ := (4, -2, 2)

theorem parallel_planes_normal_vectors (α β : Set (ℝ × ℝ × ℝ)) :
  (∃ (k : ℝ), k ≠ 0 ∧ normal_vector_β = (k * normal_vector_α.1, k * normal_vector_α.2.1, k * normal_vector_α.2.2)) →
  (∀ (p q : ℝ × ℝ × ℝ), p ∈ α ∧ q ∈ α → p.1 - q.1 = p.2.1 - q.2.1 ∧ p.2.1 - q.2.1 = p.2.2 - q.2.2) →
  ∀ (p : ℝ × ℝ × ℝ), p ∈ β → 
    normal_vector_β.1 * p.1 + normal_vector_β.2.1 * p.2.1 + normal_vector_β.2.2 * p.2.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_planes_normal_vectors_l425_42588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_curve_l425_42508

-- Define the curve C'
def curve_C' (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the function to be minimized
noncomputable def f (x y : ℝ) : ℝ := x^2 - Real.sqrt 3 * x * y + 2 * y^2

-- Theorem statement
theorem min_value_on_curve (x y : ℝ) :
  curve_C' x y → f x y ≥ 1 ∧
  (f x y = 1 ↔ (x = 1 ∧ y = Real.sqrt 3 / 2) ∨ (x = -1 ∧ y = -Real.sqrt 3 / 2)) := by
  sorry

#check min_value_on_curve

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_on_curve_l425_42508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_extortion_l425_42525

/-- Represents the state of the line of participants -/
structure LineState where
  n : ℕ
  positions : Fin n → Fin n

/-- Represents a single move in the process -/
structure Move where
  i : ℕ
  forward : ℕ

/-- The maximum amount that can be extorted given n participants -/
def max_extorted (n : ℕ) : ℕ := 2^n - n - 1

/-- Checks if a move is valid according to the rules -/
def is_valid_move (state : LineState) (move : Move) : Prop :=
  move.i ≤ state.n ∧ move.forward ≤ move.i ∧
  ∃ (count : ℕ), count ≥ move.i ∧
    count = (Finset.filter (λ j ↦ state.positions j < state.positions ⟨move.i, by sorry⟩) Finset.univ).card

/-- The main theorem stating the maximum amount that can be extorted -/
theorem max_extortion (n : ℕ) :
  ∃ (initial_state : LineState) (moves : List Move),
    (∀ m ∈ moves, is_valid_move initial_state m) ∧
    (moves.length = max_extorted n) ∧
    (∀ (other_state : LineState) (other_moves : List Move),
      (∀ m ∈ other_moves, is_valid_move other_state m) →
      other_moves.length ≤ max_extorted n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_extortion_l425_42525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_city_death_rate_l425_42587

/-- Represents the population dynamics of a city --/
structure CityPopulation where
  birth_rate : ℚ  -- Birth rate per two seconds
  net_increase : ℕ  -- Net population increase per day
  death_rate : ℚ  -- Death rate per two seconds

/-- Calculates the death rate given birth rate and net increase --/
noncomputable def calculate_death_rate (city : CityPopulation) : ℚ :=
  let seconds_per_day : ℕ := 24 * 60 * 60
  let birth_rate_per_second : ℚ := city.birth_rate / 2
  let net_increase_per_second : ℚ := city.net_increase / seconds_per_day
  2 * (birth_rate_per_second - net_increase_per_second)

/-- Theorem: The death rate in the city is 2 people every two seconds --/
theorem city_death_rate :
  ∀ (city : CityPopulation),
    city.birth_rate = 7 ∧ 
    city.net_increase = 216000 →
    calculate_death_rate city = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_city_death_rate_l425_42587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l425_42592

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := x * (x + k) * (x + 2*k)

-- State the theorem
theorem find_k : ∃ k : ℝ, (deriv (f k) 0 = 8) ∧ (k = 2 ∨ k = -2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l425_42592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_max_value_of_g_exact_max_value_of_g_l425_42509

-- Define the functions f and g
def f (x : ℝ) := |x - 4|

noncomputable def g (x : ℝ) := 2 * Real.sqrt (|x - 2|) + Real.sqrt (|x - 6|)

-- Theorem for the range of x
theorem range_of_x (x : ℝ) (h : f x ≤ 2) : 2 ≤ x ∧ x ≤ 6 := by sorry

-- Theorem for the maximum value of g(x)
theorem max_value_of_g : 
  ∃ (M : ℝ), M = 2 * Real.sqrt 5 ∧ 
  ∀ x, 2 ≤ x ∧ x ≤ 6 → g x ≤ M := by sorry

-- Theorem for the exact maximum value
theorem exact_max_value_of_g : 
  ∃ (x : ℝ), 2 ≤ x ∧ x ≤ 6 ∧ g x = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_max_value_of_g_exact_max_value_of_g_l425_42509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l425_42540

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * x + 1 / (2 * x)

-- State the theorem
theorem f_monotone_decreasing :
  -- f is an odd function
  (∀ x, f (-x) = -f x) →
  -- f satisfies the given conditions
  f 1 = 5/2 →
  f 2 = 17/4 →
  -- f is monotonically decreasing in (0, 1/2)
  ∀ x y, 0 < x → x < y → y < 1/2 → f y < f x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l425_42540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_time_with_distractions_l425_42561

noncomputable def typing_rate (num_typists : ℕ) (hours_to_complete : ℝ) : ℝ :=
  (num_typists : ℝ) / hours_to_complete

noncomputable def combined_rate (fast_rate slow_rate additional_rate : ℝ) : ℝ :=
  fast_rate + slow_rate + additional_rate

noncomputable def distraction_loss (num_typists : ℕ) : ℝ :=
  (2 : ℝ) * (1 / 6)

theorem typing_time_with_distractions
  (fast_typists slow_typists additional_typists : ℕ)
  (fast_time slow_time additional_time : ℝ)
  (h_fast : typing_rate fast_typists fast_time = fast_typists / fast_time)
  (h_slow : typing_rate slow_typists slow_time = slow_typists / slow_time)
  (h_additional : typing_rate additional_typists additional_time = additional_typists / additional_time)
  (h_fast_typists : fast_typists = 2)
  (h_slow_typists : slow_typists = 3)
  (h_additional_typists : additional_typists = 2)
  (h_fast_time : fast_time = 2)
  (h_slow_time : slow_time = 3)
  (h_additional_time : additional_time = 4)
  (h_total_typists : fast_typists + slow_typists + additional_typists = 7) :
  let total_rate := combined_rate (typing_rate fast_typists fast_time) (typing_rate slow_typists slow_time) (typing_rate additional_typists additional_time)
  let effective_rate := total_rate - distraction_loss (fast_typists + slow_typists + additional_typists)
  effective_rate = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typing_time_with_distractions_l425_42561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l425_42531

noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, 2]
noncomputable def N : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, 0; 0, 1]

theorem curve_transformation (x' y' : ℝ) :
  let MN := M * N
  let original_curve (x : ℝ) := Real.sin x
  let transformed_point (x y : ℝ) := MN.mulVec ![x, y]
  let new_curve (x' : ℝ) := 2 * Real.sin (2 * x')
  (∃ x y, transformed_point x y = ![x', y'] ∧ y = original_curve x) →
  y' = new_curve x' := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l425_42531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_given_conditions_l425_42583

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The equation of a hyperbola centered at the origin -/
def hyperbola_equation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Checks if two vectors are perpendicular -/
def perpendicular (p q : Point) : Prop :=
  p.x * q.x + p.y * q.y = 0

/-- The slope of the line passing through the right focus -/
noncomputable def focus_line_slope : ℝ := Real.sqrt (3/5)

/-- Theorem: Given the conditions, the hyperbola has the equation x² - y²/3 = 1 -/
theorem hyperbola_equation_given_conditions 
  (h : Hyperbola) 
  (p q : Point) 
  (hp : hyperbola_equation h p)
  (hq : hyperbola_equation h q)
  (hfocus : p.x > 0 ∧ q.x > 0)  -- Points are on the right side
  (hslope : (q.y - p.y) / (q.x - p.x) = focus_line_slope)
  (hperp : perpendicular p q)
  (hdist : distance p q = 4) :
  h.a = 1 ∧ h.b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_given_conditions_l425_42583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_3_l425_42586

-- Define the line C₃
noncomputable def line_C3 (x : ℝ) : ℝ := Real.sqrt 3 * x

-- Define the circle C₂
def circle_C2 (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + (y + 2)^2 = 1

-- Define the intersection points A and B
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ y = line_C3 x ∧ circle_C2 x y}

-- Statement to prove
theorem intersection_distance_is_sqrt_3 :
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_is_sqrt_3_l425_42586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l425_42599

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x, f x = f (-x)
axiom f_decreasing_neg : ∀ x y, x < y ∧ y ≤ 0 → f x > f y

-- Define a, b, and c
noncomputable def a : ℝ := f (Real.log 3 / Real.log 2)
noncomputable def b : ℝ := f (Real.log 5 / Real.log 4)
noncomputable def c : ℝ := f (Real.sqrt 2)

-- State the theorem
theorem order_of_abc : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l425_42599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l425_42590

noncomputable section

/-- A rational function with a parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := (x^2 - x + c) / (x^2 + x - 20)

/-- The denominator of the function f -/
def denominator (x : ℝ) : ℝ := x^2 + x - 20

/-- The numerator of the function f -/
def numerator (c : ℝ) (x : ℝ) : ℝ := x^2 - x + c

/-- Theorem stating that f has exactly one vertical asymptote iff c = -12 or c = -30 -/
theorem one_vertical_asymptote (c : ℝ) :
  (∃! x, denominator x = 0 ∧ numerator c x ≠ 0) ↔ c = -12 ∨ c = -30 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_vertical_asymptote_l425_42590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_camera_rental_payment_l425_42576

/-- Camera rental payment theorem -/
theorem camera_rental_payment (camera_value rental_weeks : ℕ) 
  (weekly_rental_rate insurance_rate weekly_discount_rate : ℚ) 
  (discount_weeks : ℕ) 
  (mike_contribution_rate sarah_contribution_rate alex_contribution_rate : ℚ) : 
  camera_value = 5000 →
  rental_weeks = 4 →
  weekly_rental_rate = 1/10 →
  insurance_rate = 1/20 →
  weekly_discount_rate = 1/50 →
  discount_weeks = 2 →
  mike_contribution_rate = 1/5 →
  sarah_contribution_rate = 3/10 →
  alex_contribution_rate = 1/10 →
  (camera_value * weekly_rental_rate * rental_weeks +
   camera_value * insurance_rate -
   camera_value * weekly_discount_rate * discount_weeks) *
  (1 - (mike_contribution_rate + sarah_contribution_rate + alex_contribution_rate)) = 820 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_camera_rental_payment_l425_42576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_ln_x_over_x_l425_42517

open Real

theorem max_value_ln_x_over_x :
  ∃ (c : ℝ), c > 0 ∧ (∀ x > 0, (log x) / x ≤ (log c) / c) ∧ (log c) / c = 1 / (exp 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_ln_x_over_x_l425_42517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_E_perimeter_l425_42535

/-- The perimeter of a letter E formed by three 3x6 inch rectangles -/
theorem letter_E_perimeter :
  let width := 3
  let height := 6
  let num_rectangles := 3
  let vertical_contribution := 2 * height + width
  let horizontal_contribution := 2 * (width + height) - height
  let total_horizontal_contribution := 2 * horizontal_contribution
  vertical_contribution + total_horizontal_contribution = 39 := by
    -- Proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_letter_E_perimeter_l425_42535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l425_42527

noncomputable def N (x : ℝ) : ℝ := 3 * Real.sqrt x

def O (x : ℝ) : ℝ := x ^ 2

theorem nested_function_evaluation :
  N (O (N (O (N (O 2))))) = 54 := by
  -- Unfold definitions and simplify
  simp [N, O]
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_function_evaluation_l425_42527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l425_42501

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else a/x

-- State the theorem
theorem f_decreasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) ↔ a ∈ Set.Icc (1/6) (1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_in_range_l425_42501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_steps_to_catch_thief_l425_42510

-- Define the chessboard positions
inductive Position : Type
| A : Position
| B : Position
| C : Position
| D : Position
| F : Position

-- Define the game state
structure GameState where
  police : Position
  thief : Position
  turn : Nat

-- Define the valid moves
def validMove : Position → Position → Prop
| Position.A, Position.B => True
| Position.A, Position.C => True
| Position.B, Position.C => True
| Position.C, Position.D => True
| Position.D, Position.F => True
| _, _ => False

-- Define the initial state
def initialState : GameState :=
{ police := Position.A, thief := Position.B, turn := 0 }

-- Define a winning state for the police
def policeWins (state : GameState) : Prop :=
  state.police = state.thief

-- Define the theorem
theorem min_steps_to_catch_thief :
  ∃ (n : Nat), ∀ (strategy : Nat → Position → Position),
    ∃ (police_moves : Nat → Position),
      police_moves 0 = Position.A ∧
      (∀ k, k < n → validMove (police_moves k) (police_moves (k+1))) ∧
      policeWins { police := police_moves n, 
                   thief := strategy n (police_moves n), 
                   turn := n } ∧
      n = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_steps_to_catch_thief_l425_42510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_articles_sold_at_cost_price_l425_42547

/-- The number of articles sold at cost price -/
def X : ℕ := 40

/-- The cost price of one article -/
def C : ℝ := sorry

/-- The selling price of one article -/
def S (c : ℝ) : ℝ := 1.25 * c

/-- The profit percentage -/
def profit_percentage : ℝ := 0.25

theorem articles_sold_at_cost_price :
  (X * C = 32 * S C) ∧ (S C = C + profit_percentage * C) → X = 40 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_articles_sold_at_cost_price_l425_42547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_l425_42562

-- Define the pyramid
structure Pyramid :=
  (AB : ℝ)
  (AC : ℝ)
  (sinBAC : ℝ)
  (lateralAngle : ℝ)

-- Define the conditions
def pyramidConditions (p : Pyramid) : Prop :=
  p.AB = 5 ∧
  p.AC = 8 ∧
  p.sinBAC = 3/5 ∧
  p.lateralAngle ≤ Real.pi/3 ∧
  p.lateralAngle > 0

-- Define the volume function
noncomputable def pyramidVolume (p : Pyramid) : ℝ :=
  (1/6) * p.AB * p.AC * p.sinBAC * (10 * Real.sqrt 3)

-- Theorem statement
theorem max_pyramid_volume :
  ∀ p : Pyramid, pyramidConditions p →
  pyramidVolume p ≤ 10 * Real.sqrt 51 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_pyramid_volume_l425_42562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_six_terms_l425_42502

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

def sum_geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then n * a₁
  else a₁ * (1 - r^n) / (1 - r)

theorem sum_of_first_six_terms :
  sum_geometric_sequence 1 (1/2) 6 = 63 / 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_six_terms_l425_42502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_inequality_smallest_diameter_after_2013_folds_l425_42566

/-- A finite set of points in the plane -/
def X : Set (ℝ × ℝ) := sorry

/-- The largest possible area of a polygon with at most n vertices from X -/
noncomputable def f_X (n : ℕ) : ℝ := sorry

/-- Theorem: f_X(m) + f_X(n) ≥ f_X(m+1) + f_X(n-1) for m ≥ n > 2 -/
theorem polygon_area_inequality (m n : ℕ) (h1 : m ≥ n) (h2 : n > 2) :
  f_X m + f_X n ≥ f_X (m + 1) + f_X (n - 1) := by sorry

/-- The diameter of a polygon after i folds -/
noncomputable def diameter (i : ℕ) : ℝ := sorry

/-- Theorem: The smallest possible diameter of P_2013 is 0 -/
theorem smallest_diameter_after_2013_folds :
  ∀ ε > 0, ∃ N : ℕ, N ≤ 2013 ∧ diameter N < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_area_inequality_smallest_diameter_after_2013_folds_l425_42566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_puppies_total_is_twelve_l425_42551

-- Define variables in the theorem or lemma instead of using 'variable'
theorem total_puppies (P x : ℕ) : ∃ (given_away remaining : ℕ),
  given_away = 7 ∧ remaining = 5 ∧ P + x = given_away + remaining := by
  -- Define given_away and remaining inside the theorem
  let given_away := 7
  let remaining := 5
  
  -- Prove the existence of given_away and remaining
  use given_away, remaining
  
  -- Prove the three conditions
  constructor
  · rfl  -- given_away = 7
  constructor
  · rfl  -- remaining = 5
  · sorry -- P + x = given_away + remaining

-- Additional theorem to show that the total number of puppies is 12
theorem total_is_twelve (P x : ℕ) : ∃ (total : ℕ), 
  total = 12 ∧ P + x = total := by
  use 12
  constructor
  · rfl
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_puppies_total_is_twelve_l425_42551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_functions_equality_l425_42548

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x - 1

noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 else -1

-- Define the composite functions
noncomputable def f_g (x : ℝ) : ℝ :=
  if x ≥ 0 then 2 * x^2 - 1 else -3

noncomputable def g_f (x : ℝ) : ℝ :=
  if x ≥ 1/2 then (2 * x - 1)^2 else -1

-- State the theorem
theorem composite_functions_equality :
  ∀ x : ℝ, f (g x) = f_g x ∧ g (f x) = g_f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composite_functions_equality_l425_42548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l425_42512

-- Define the vector type
def Vec2 := ℝ × ℝ

-- Define the given vectors
def a : Vec2 := (1, 2)
def b (m : ℝ) : Vec2 := (-2, m)

-- Define parallelism condition
def parallel (v w : Vec2) : Prop := ∃ (k : ℝ), v = (k * w.1, k * w.2)

-- Define vector addition
def add (v w : Vec2) : Vec2 := (v.1 + w.1, v.2 + w.2)

-- Define scalar multiplication
def smul (k : ℝ) (v : Vec2) : Vec2 := (k * v.1, k * v.2)

-- Define vector magnitude
noncomputable def magnitude (v : Vec2) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- The theorem to prove
theorem vector_magnitude_proof :
  ∃ (m : ℝ), parallel a (b m) ∧ magnitude (add (smul 2 a) (smul 3 (b m))) = 4 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_proof_l425_42512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l425_42514

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 4*x + 2

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/3)^(f a x)

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x, f a x = f a (4 - x)) →  -- Axis of symmetry is x = 2
  (∃ k, ∀ x, f 1 x = f k x) ∧   -- f(x) = x^2 - 4x + 2
  Set.range (g 1) = Set.Ioo 0 9 -- Range of g(x) is (0, 9]
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l425_42514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_open_unit_interval_l425_42524

def A (a : ℝ) : Set ℝ := {-1, 0, a}

def B : Set ℝ := {x : ℝ | 1 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 2}

theorem a_in_open_unit_interval (a : ℝ) : 
  (A a ∩ B).Nonempty → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_in_open_unit_interval_l425_42524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l425_42563

theorem diophantine_equation_solutions :
  {(a, b, c) : ℕ+ × ℕ × ℕ+ | (2 : ℕ)^(a.val) * 3^b + 9 = c^2} =
  {(4, 0, 5), (3, 2, 9), (4, 3, 21), (3, 3, 15), (4, 5, 51)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diophantine_equation_solutions_l425_42563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_of_linear_function_l425_42532

theorem zero_point_of_linear_function :
  ∃ x : ℝ, 2 * x - 4 = 0 ∧ x = 2 := by
  use 2
  constructor
  · ring
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_of_linear_function_l425_42532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_circle_radius_sphere_radius_sphere_center_l425_42555

/-- A sphere in 3D space -/
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The sphere that intersects both xy-plane and yz-plane -/
def intersectingSphere : Sphere :=
  { center := (3, 5, -8),
    radius := 8 }

/-- The circle formed by the intersection of the sphere and xy-plane -/
def xyCircle : Circle :=
  { center := (3, 5),
    radius := 3 }

/-- The circle formed by the intersection of the sphere and yz-plane -/
noncomputable def yzCircle : Circle :=
  { center := (5, -8),
    radius := Real.sqrt 55 }

/-- Theorem stating that the radius of the yz-plane intersection circle is √55 -/
theorem yz_circle_radius :
  yzCircle.radius = Real.sqrt 55 := by
  -- The proof goes here
  sorry

/-- Theorem verifying the radius of the intersecting sphere -/
theorem sphere_radius :
  intersectingSphere.radius = 8 := by
  -- The proof goes here
  sorry

/-- Theorem verifying the center of the intersecting sphere -/
theorem sphere_center :
  intersectingSphere.center = (3, 5, -8) := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_yz_circle_radius_sphere_radius_sphere_center_l425_42555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l425_42522

noncomputable def y : ℕ → ℝ
  | 0 => 0  -- Adding a case for 0 to cover all natural numbers
  | 1 => 4^(1/4)
  | 2 => (4^(1/4))^(4^(1/4))
  | n + 3 => (y (n + 2))^(4^(1/4))

theorem smallest_integer_y : ∀ k : ℕ, k < 4 → ¬ (∃ m : ℤ, y k = m) ∧ ∃ m : ℤ, y 4 = m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_y_l425_42522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_one_in_four_l425_42503

theorem probability_of_one_in_four {α : Type*} [DecidableEq α] (S : Finset α) (a : α) 
  (h1 : S.card = 4) (h2 : a ∈ S) :
  (Finset.filter (· = a) S).card / S.card = (1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_one_in_four_l425_42503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compounding_frequency_is_two_l425_42553

/-- Compound interest calculation -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (frequency : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate / frequency) ^ (frequency * time)

/-- Proof of compounding frequency -/
theorem compounding_frequency_is_two :
  ∃ (frequency : ℝ),
    frequency = 2 ∧
    compound_interest 6000 0.1 frequency 1 = 6615 := by
  use 2
  constructor
  · rfl
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_compounding_frequency_is_two_l425_42553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l425_42541

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem floor_inequality_solution_set :
  {x : ℝ | (floor x)^2 - 5*(floor x) + 6 ≤ 0} = Set.Ici 2 ∩ Set.Iio 4 :=
by
  sorry -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_solution_set_l425_42541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_equality_l425_42595

theorem determinant_equality (a b c d : ℝ) : 
  Matrix.det !![a, b; c, d] = 4 → 
  Matrix.det !![a, 5*a + 3*b; c, 5*c + 3*d] = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_equality_l425_42595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_non_integers_l425_42558

theorem fraction_non_integers : 
  ∃ (a b c d : ℤ), 
    (a - b) / 2 ∉ (Set.range (Int.ofNat)) ∧ 
    (b - c) / 2 ∉ (Set.range (Int.ofNat)) ∧ 
    (c - d) / 2 ∉ (Set.range (Int.ofNat)) ∧ 
    (d - a) / 2 ∉ (Set.range (Int.ofNat)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_non_integers_l425_42558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_sum_k_l425_42594

/-- Function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (p₁ p₂ p₃ : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := p₁
  let (x₂, y₂) := p₂
  let (x₃, y₃) := p₃
  (1/2) * abs ((x₁ - x₃) * (y₂ - y₁) - (x₁ - x₂) * (y₃ - y₁))

/-- The sum of k values for which the area of the triangle with vertices 
    (2, 5), (8, 20), and (6, k) is minimum, given that k is an integer. -/
theorem min_area_triangle_sum_k : ∃ (k₁ k₂ : ℤ),
  k₁ ≠ k₂ ∧
  (∀ (k : ℤ), 
    area_triangle (2, 5) (8, 20) (6, k) ≥ area_triangle (2, 5) (8, 20) (6, k₁) ∧
    area_triangle (2, 5) (8, 20) (6, k) ≥ area_triangle (2, 5) (8, 20) (6, k₂)) ∧
  k₁ + k₂ = 30 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_sum_k_l425_42594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l425_42539

noncomputable def f (x : ℝ) := 12 / x + 4 * x

theorem minimum_value_of_f :
  ∀ x : ℝ, x > 0 → f x ≥ 8 * Real.sqrt 3 ∧ ∃ y : ℝ, y > 0 ∧ f y = 8 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_of_f_l425_42539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_product_digits_l425_42589

/-- A number composed of n repetitions of a digit d -/
def repeated_digit (n : ℕ) (d : ℕ) : ℕ :=
  d * (10^n - 1) / 9

/-- The product of two numbers, one composed of 666 threes and the other of 666 sixes -/
def special_product : ℕ :=
  (repeated_digit 666 3) * (repeated_digit 666 6)

/-- A predicate to check if a natural number consists only of digits 2, 1, and 7 -/
def consists_of_217 (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 2 ∨ d = 1 ∨ d = 7

/-- The main theorem stating that the special product consists only of digits 2, 1, and 7 -/
theorem special_product_digits :
  consists_of_217 special_product := by
  sorry

#eval special_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_product_digits_l425_42589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_problem_l425_42542

/-- Represents the number of valid towers that can be built with cubes of edge-lengths 1 to n -/
def num_towers : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | 2 => 1
  | n + 3 => 2 * num_towers (n + 2)

/-- The problem statement -/
theorem tower_problem :
  num_towers 7 = 32 := by
  rfl

#eval num_towers 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_problem_l425_42542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_max_length_l425_42534

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the point P on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := circle_C P.1 P.2

-- Define the point Q as the projection of P on the y-axis
def Q (P : ℝ × ℝ) : ℝ × ℝ := (0, P.2)

-- Define the vector OM in terms of QP
def vector_OM (P M : ℝ × ℝ) : Prop :=
  M.1 = Real.sqrt 3 * (P.1 - (Q P).1) ∧ M.2 = P.2

-- Define the trajectory E
def trajectory_E (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define a tangent line to circle C
def tangent_line (m n : ℝ) : Prop := n^2 = m^2 + 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) (m n : ℝ) : Prop :=
  trajectory_E A.1 A.2 ∧ trajectory_E B.1 B.2 ∧
  A.1 = m * A.2 + n ∧ B.1 = m * B.2 + n

theorem trajectory_and_max_length :
  ∀ (P M : ℝ × ℝ),
    point_on_circle P →
    vector_OM P M →
    (∀ (x y : ℝ), (x, y) = M → trajectory_E x y) ∧
    (∀ (A B : ℝ × ℝ) (m n : ℝ),
      tangent_line m n →
      intersection_points A B m n →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_max_length_l425_42534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_take_home_pay_increase_l425_42500

/-- Calculates the percentage increase in take-home pay after a salary raise and tax rate change -/
noncomputable def percentage_increase_take_home_pay (initial_salary : ℝ) (new_salary : ℝ) 
  (initial_tax_rate : ℝ) (new_tax_rate : ℝ) : ℝ :=
  let initial_take_home := initial_salary * (1 - initial_tax_rate)
  let new_take_home := new_salary * (1 - new_tax_rate)
  ((new_take_home - initial_take_home) / initial_take_home) * 100

/-- The percentage increase in John's take-home pay is approximately 12.55% -/
theorem john_take_home_pay_increase :
  |percentage_increase_take_home_pay 60 70 0.15 0.18 - 12.55| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_take_home_pay_increase_l425_42500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l425_42585

theorem expression_equality : 
  |(-2 * Real.sqrt 3)| - (1 - Real.pi)^(0 : ℝ) + 2 * Real.cos (π / 6) + (1/4)^(-1 : ℝ) = 3 * Real.sqrt 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l425_42585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_stick_l425_42533

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

def available_lengths : List ℝ := [3, 6, 11, 12]

theorem unique_valid_stick (a b : ℝ) (ha : a = 4) (hb : b = 7) :
  ∃! c, c ∈ available_lengths ∧ valid_triangle a b c :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_stick_l425_42533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_leq_two_l425_42584

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - 1/x - a * Real.log x

-- State the theorem
theorem f_nonnegative_iff_a_leq_two :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) ↔ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_iff_a_leq_two_l425_42584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_perpendicular_relations_l425_42577

/-- Two non-coincident lines are parallel if their direction vectors are scalar multiples of each other -/
def lines_parallel (a b : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.fst, k * b.snd.fst, k * b.snd.snd) ∨ b = (k * a.fst, k * a.snd.fst, k * a.snd.snd)

/-- Two planes are perpendicular if their normal vectors are perpendicular -/
def planes_perpendicular (u v : ℝ × ℝ × ℝ) : Prop :=
  u.fst * v.fst + u.snd.fst * v.snd.fst + u.snd.snd * v.snd.snd = 0

theorem parallel_and_perpendicular_relations :
  (lines_parallel (2, 3, -1) (-2, -3, 1)) ∧
  (planes_perpendicular (2, 2, -1) (-3, 4, 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_and_perpendicular_relations_l425_42577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotonicity_and_inequality_l425_42519

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) / (x^2 + 1)

-- Theorem statement
theorem odd_function_monotonicity_and_inequality :
  -- Part 1: f is an odd function on [-1, 1] implies a = 0
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f a x = -f a (-x)) →
  a = 0 ∧
  -- Part 2: f is monotonically increasing on [-1, 1]
  (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x ≤ y → f 0 x ≤ f 0 y) ∧
  -- Part 3: Solution set of the inequality
  {x : ℝ | f 0 (5*x - 1) < f 0 (6*x^2)} = {x : ℝ | 0 ≤ x ∧ x < 1/3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotonicity_and_inequality_l425_42519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_days_passed_l425_42580

/-- Proves that the fraction of days passed when 12 pills are left is 4/5 -/
theorem fraction_of_days_passed (total_days : ℕ) (daily_dose : ℕ) (pills_left : ℕ) : 
  total_days = 30 ∧ daily_dose = 2 ∧ pills_left = 12 → 
  (total_days - (pills_left / daily_dose)) / total_days = 4 / 5 := by
  intro h
  -- The proof steps would go here
  sorry

#check fraction_of_days_passed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_days_passed_l425_42580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l425_42593

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V) 
  (h1 : ‖a‖ = 1) 
  (h2 : ‖b‖ = 2) 
  (h3 : ‖a + b‖ = ‖a - b‖) : 
  ‖(2 : ℝ) • a + b‖ = 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l425_42593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_inequality_l425_42557

open Real BigOperators

variable {n : ℕ} -- number of terms
variable (x : Fin n → ℝ) -- x_i values
variable (m : ℝ) -- exponent
variable (a : ℝ) -- constant term
variable (s : ℝ) -- sum of x_i

theorem product_inequality (hx : ∀ i, x i > 0) (hm : m > 0) (ha : a ≥ 0)
  (hs : ∑ i, x i = s) (hsn : s ≤ n) :
  ∏ i, (x i ^ m + (x i ^ m)⁻¹ + a) ≥ ((s / n) ^ m + (n / s) ^ m + a) ^ n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_inequality_l425_42557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_and_sum_sum_of_components_l425_42559

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the expression before rationalization
noncomputable def original_expression : ℝ := 1 / (cubeRoot 7 - cubeRoot 5)

-- Define the expression after rationalization
noncomputable def rationalized_expression : ℝ := (cubeRoot 49 + cubeRoot 35 + cubeRoot 25) / 2

-- Statement to prove
theorem rationalize_denominator_and_sum :
  original_expression = rationalized_expression ∧
  49 + 35 + 25 + 2 = 111 := by
  sorry

-- Additional theorem to show the sum of A, B, C, and D
theorem sum_of_components :
  49 + 35 + 25 + 2 = 111 := by
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_denominator_and_sum_sum_of_components_l425_42559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_difference_l425_42579

-- Define a fair coin flip
def fair_coin_prob : ℚ := 1 / 2

-- Define the number of flips
def num_flips : ℕ := 4

-- Define the probability of exactly 3 heads in 4 flips
def prob_3_heads : ℚ := (Nat.choose num_flips 3) * (fair_coin_prob ^ 3) * (1 - fair_coin_prob)

-- Define the probability of 4 heads in 4 flips
def prob_4_heads : ℚ := fair_coin_prob ^ num_flips

-- Theorem statement
theorem coin_flip_probability_difference : 
  prob_3_heads - prob_4_heads = 7 / 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_flip_probability_difference_l425_42579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_in_sqrt_function_l425_42575

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x)

-- State the theorem
theorem range_of_x_in_sqrt_function :
  {x : ℝ | x ≤ 1} = {x : ℝ | ∃ y, f x = y} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_in_sqrt_function_l425_42575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_distance_l425_42545

/-- Given a rectangular plot and fence poles, calculate the distance between poles -/
theorem pole_distance (length width : ℝ) (num_poles : ℕ) 
  (h_length : length = 60)
  (h_width : width = 50)
  (h_poles : num_poles = 44) :
  (2 * (length + width)) / (num_poles - 1) = (2 * (length + width)) / (num_poles - 1) :=
by
  -- The proof is trivial as we're asserting equality to itself
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_distance_l425_42545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wine_division_l425_42582

theorem wine_division (m n : ℕ) :
  (∃ (steps : List (ℕ × ℕ × ℕ)), 
    steps.head? = some (m + n, 0, 0) ∧ 
    steps.getLast? = some ((m + n) / 2, (m + n) / 2, 0) ∧
    ∀ (i j k : ℕ), (i, j, k) ∈ steps → i + j + k = m + n ∧ j ≤ m ∧ k ≤ n) 
  ↔ 
  (Even (m + n) ∧ ((m + n) / 2 % Nat.gcd m n = 0)) :=
by sorry

#check wine_division

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wine_division_l425_42582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_is_80_minutes_l425_42565

/-- Represents the trip details --/
structure TripDetails where
  highway_distance : ℚ
  coastal_distance : ℚ
  coastal_time : ℚ
  highway_speed_multiplier : ℚ

/-- Calculates the total trip time given the trip details --/
def total_trip_time (trip : TripDetails) : ℚ :=
  let coastal_speed := trip.coastal_distance / trip.coastal_time
  let highway_speed := coastal_speed * trip.highway_speed_multiplier
  let highway_time := trip.highway_distance / highway_speed
  trip.coastal_time + highway_time

/-- Theorem stating that the total trip time is 80 minutes --/
theorem trip_time_is_80_minutes (trip : TripDetails)
  (h1 : trip.highway_distance = 50)
  (h2 : trip.coastal_distance = 10)
  (h3 : trip.coastal_time = 30)
  (h4 : trip.highway_speed_multiplier = 3) :
  total_trip_time trip = 80 := by
  sorry

def main : IO Unit := do
  let trip : TripDetails := {
    highway_distance := 50,
    coastal_distance := 10,
    coastal_time := 30,
    highway_speed_multiplier := 3
  }
  IO.println s!"Total trip time: {total_trip_time trip}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trip_time_is_80_minutes_l425_42565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_y_coordinates_is_10_l425_42554

-- Define the circle
def circle_center : ℝ × ℝ := (-4, 5)
def circle_radius : ℝ := 13

-- Define the function to calculate the sum of y-coordinates
noncomputable def sum_y_coordinates : ℝ :=
  let y₁ := circle_center.2 + Real.sqrt (circle_radius ^ 2 - circle_center.1 ^ 2)
  let y₂ := circle_center.2 - Real.sqrt (circle_radius ^ 2 - circle_center.1 ^ 2)
  y₁ + y₂

-- Theorem statement
theorem sum_y_coordinates_is_10 : sum_y_coordinates = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_y_coordinates_is_10_l425_42554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_equidistant_l425_42574

-- Define a line segment
structure LineSegment where
  start : ℝ × ℝ
  endpoint : ℝ × ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define the perpendicular bisector of a line segment
def PerpendicularBisector (segment : LineSegment) : Set Point := sorry

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ := sorry

-- State the theorem
theorem perpendicular_bisector_equidistant (segment : LineSegment) (p : Point) :
  p ∈ PerpendicularBisector segment →
  distance p segment.start = distance p segment.endpoint := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_bisector_equidistant_l425_42574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_sqrt_l425_42572

theorem max_sum_of_sqrt (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : 
  (Real.sqrt (2*x + 1) + Real.sqrt (2*y + 1) + Real.sqrt (2*z + 1)) ≤ 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_sqrt_l425_42572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_range_l425_42564

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4*x + 3*Real.sin x

-- State the theorem
theorem function_inequality_implies_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 1, f x = 4*x + 3*Real.sin x) →
  f (1 - a) + f (1 - a^2) < 0 →
  a ∈ Set.Ioo 1 (Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_implies_range_l425_42564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_typist_salary_calculation_l425_42568

/-- Calculates the final salary after a raise and a reduction -/
noncomputable def calculate_final_salary (original_salary : ℝ) (raise_percent : ℝ) (reduction_percent : ℝ) : ℝ :=
  let raised_salary := original_salary * (1 + raise_percent / 100)
  raised_salary * (1 - reduction_percent / 100)

/-- Theorem: The typist's final salary is 4180 after a 10% raise and 5% reduction -/
theorem typist_salary_calculation :
  let original_salary : ℝ := 4000.0000000000005
  let raise_percent : ℝ := 10
  let reduction_percent : ℝ := 5
  calculate_final_salary original_salary raise_percent reduction_percent = 4180 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_final_salary 4000.0000000000005 10 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_typist_salary_calculation_l425_42568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_conjugates_l425_42515

-- Define a complex number type
structure MyComplex where
  re : ℝ
  im : ℝ

-- Define conjugate for complex numbers
def conjugate (z : MyComplex) : MyComplex :=
  ⟨z.re, -z.im⟩

-- Define multiplication for complex numbers
def mul (z w : MyComplex) : MyComplex :=
  ⟨z.re * w.re - z.im * w.im, z.re * w.im + z.im * w.re⟩

-- State the theorem
theorem product_of_conjugates (x y : ℝ) :
  let z : MyComplex := ⟨x, y⟩
  let w : MyComplex := conjugate z
  (mul z w).im = 0 ∧ (mul z w).re = x^2 + y^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_conjugates_l425_42515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_on_escalators_l425_42597

/-- Represents the time taken to travel a distance on a moving platform -/
noncomputable def time_on_platform (platform_speed : ℝ) (person_speed : ℝ) (distance : ℝ) : ℝ :=
  distance / (platform_speed + person_speed)

/-- Theorem stating the total time taken to travel on both escalators -/
theorem total_time_on_escalators (
  first_escalator_speed : ℝ)
  (first_escalator_length : ℝ)
  (second_walkway_speed : ℝ)
  (second_walkway_length : ℝ)
  (person_speed : ℝ)
  (h1 : first_escalator_speed = 10)
  (h2 : first_escalator_length = 112)
  (h3 : second_walkway_speed = 6)
  (h4 : second_walkway_length = 80)
  (h5 : person_speed = 4)
  : time_on_platform first_escalator_speed person_speed first_escalator_length +
    time_on_platform second_walkway_speed person_speed second_walkway_length = 16 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_time_on_escalators_l425_42597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_squares_roots_l425_42538

theorem minimize_sum_of_squares_roots (a : ℝ) : 
  let f (x : ℝ) := x^2 - (3*a + 1)*x + (2*a^2 - 3*a - 2)
  let discriminant := (3*a + 1)^2 - 4*(2*a^2 - 3*a - 2)
  let sum_of_squares := 5*a^2 + 12*a + 5
  (∀ x, f x = 0 → x ∈ Set.range Real.sqrt) ∧ 
  (∀ b, (∀ x, f x = 0 → x ∈ Set.range Real.sqrt) → sum_of_squares ≤ 5*b^2 + 12*b + 5) →
  a = -9 + 6 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_sum_of_squares_roots_l425_42538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_selection_probability_l425_42506

/-- The probability of selecting 2 red and 2 green marbles from a bag -/
theorem marble_selection_probability
  (total : ℕ)
  (red : ℕ)
  (blue : ℕ)
  (green : ℕ)
  (h_total : total = red + blue + green)
  (h_red : red = 12)
  (h_blue : blue = 8)
  (h_green : green = 5) :
  (Nat.choose red 2 * Nat.choose green 2 : ℚ) / Nat.choose total 4 = 2 / 39 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marble_selection_probability_l425_42506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solutions_l425_42511

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the properties of g
def is_valid_g (g : ℝ → ℝ) : Prop :=
  (∀ x, f (g x) = 9*x^2 - 6*x + 1) ∧
  ∃ p : Polynomial ℝ, ∀ x, g x = p.eval x

-- Theorem statement
theorem g_solutions (g : ℝ → ℝ) (h : is_valid_g g) :
  (∀ x, g x = 3*x - 1) ∨ (∀ x, g x = -3*x + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solutions_l425_42511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_squares_l425_42543

theorem max_sum_of_squares (k : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - (k - 2) * x₁ + (k^2 + 3 * k + 5) = 0 →
  x₂^2 - (k - 2) * x₂ + (k^2 + 3 * k + 5) = 0 →
  x₁ ≠ x₂ →
  ∃ (M : ℝ), M = 18 ∧ x₁^2 + x₂^2 ≤ M ∧ ∃ (k₀ : ℝ), x₁^2 + x₂^2 = M :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_of_squares_l425_42543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l425_42569

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (2 - 3 * x) + (2 * x - 1) ^ 0

-- Define the domain
def domain (x : ℝ) : Prop := x < 2/3 ∧ x ≠ 1/2

-- Theorem statement
theorem f_domain : 
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ domain x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l425_42569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_value_l425_42537

-- Define the line
def line (x y : ℝ) : Prop := x + y = 1

-- Define the circle
def circle' (x y a : ℝ) : Prop := x^2 + y^2 = a

-- Define a point on the circle
def point_on_circle (x y a : ℝ) : Prop := circle' x y a

-- Define vector addition
def vector_sum (x1 y1 x2 y2 x0 y0 : ℝ) : Prop := x1 + x2 = x0 ∧ y1 + y2 = y0

theorem intersection_circle_value (a : ℝ) 
  (h_intersect : ∃ x1 y1 x2 y2, line x1 y1 ∧ line x2 y2 ∧ circle' x1 y1 a ∧ circle' x2 y2 a ∧ (x1 ≠ x2 ∨ y1 ≠ y2))
  (h_point_c : ∃ x0 y0, point_on_circle x0 y0 a)
  (h_vector_sum : ∀ x1 y1 x2 y2 x0 y0, 
    line x1 y1 → line x2 y2 → circle' x1 y1 a → circle' x2 y2 a → point_on_circle x0 y0 a → 
    vector_sum x1 y1 x2 y2 x0 y0) :
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_circle_value_l425_42537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_invertible_labels_l425_42513

-- Define the functions
noncomputable def f2 (x : ℝ) : ℝ := x^3 - 3*x
def f3 : Set (ℝ × ℝ) := {(1, 2), (2, 4), (3, 6)}
def f4_domain : Set ℝ := {-3*Real.pi/2, -Real.pi, -Real.pi/2, 0, Real.pi/2, Real.pi, 3*Real.pi/2}
noncomputable def f4 (x : ℝ) : ℝ := Real.sin x
noncomputable def f5 (x : ℝ) : ℝ := 1/x

-- Define invertibility for each function
def is_invertible_f2 : Prop := ∀ y : ℝ, ∃! x : ℝ, f2 x = y
def is_invertible_f3 : Prop := ∀ (a b c : ℝ), (1, a) ∈ f3 ∧ (2, b) ∈ f3 ∧ (3, c) ∈ f3 → a ≠ b ∧ b ≠ c ∧ a ≠ c
def is_invertible_f4 : Prop := ∀ x y : ℝ, x ∈ f4_domain ∧ y ∈ f4_domain ∧ x ≠ y → f4 x ≠ f4 y
def is_invertible_f5 : Prop := ∀ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x ≠ y → f5 x ≠ f5 y

-- Theorem statement
theorem product_of_invertible_labels :
  is_invertible_f2 ∧ is_invertible_f3 ∧ is_invertible_f4 ∧ is_invertible_f5 →
  2 * 3 * 4 * 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_invertible_labels_l425_42513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l425_42536

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 / x

-- State the theorem
theorem tangent_line_equation :
  let f' : ℝ → ℝ := λ x => 1 / x - 2 / (x ^ 2)
  let slope : ℝ := f' 1
  let point : ℝ × ℝ := (1, 2)
  let tangent_line (x y : ℝ) : Prop := x + y - 3 = 0
  (∀ x, x > 0 → (deriv f) x = f' x) →
  tangent_line (point.1) (point.2) ∧
  ∀ x y, tangent_line x y ↔ y - point.2 = slope * (x - point.1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l425_42536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l425_42573

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^8 + i^18 + i^(-32 : ℤ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_power_sum_l425_42573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_residual_example_l425_42507

/-- Calculates the residual for a linear regression model -/
def calculateResidual (slope : ℝ) (intercept : ℝ) (x : ℝ) (y : ℝ) : ℝ :=
  y - (slope * x + intercept)

/-- Theorem: The residual of the regression equation ŷ = 2.5x̂ + 0.31 at the point (4, 1.2) is -9.11 -/
theorem regression_residual_example : calculateResidual 2.5 0.31 4 1.2 = -9.11 := by
  -- Unfold the definition of calculateResidual
  unfold calculateResidual
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_residual_example_l425_42507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l425_42546

theorem sin_graph_shift (x : ℝ) :
  Real.sin (2*x - π/6) = Real.sin (2*(x - π/12)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l425_42546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_segments_line_exists_l425_42529

/-- Two concentric circles -/
structure ConcentricCircles where
  center : ℝ × ℝ
  radius1 : ℝ
  radius2 : ℝ
  h : radius1 < radius2

/-- A point on a circle -/
def PointOnCircle (c : ConcentricCircles) : Type :=
  { p : ℝ × ℝ // (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius1^2 }

/-- Reflection of a point through another point -/
def reflect (p : ℝ × ℝ) (through : ℝ × ℝ) : ℝ × ℝ :=
  (2 * through.1 - p.1, 2 * through.2 - p.2)

/-- Theorem: There exists a line intersecting two concentric circles with three equal segments -/
theorem equal_segments_line_exists (c : ConcentricCircles) :
  ∃ (x : PointOnCircle c) (y : ℝ × ℝ),
    (y.1 - c.center.1)^2 + (y.2 - c.center.2)^2 = c.radius2^2 ∧
    let midpoint := ((x.val.1 + y.1)/2, (x.val.2 + y.2)/2)
    (midpoint.1 - x.val.1)^2 + (midpoint.2 - x.val.2)^2 =
    (y.1 - midpoint.1)^2 + (y.2 - midpoint.2)^2 ∧
    (midpoint.1 - c.center.1)^2 + (midpoint.2 - c.center.2)^2 = c.radius1^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_segments_line_exists_l425_42529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l425_42528

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a parabola in the form ax^2 + bxy + cy^2 + dx + ey + f = 0 -/
structure Parabola where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ

/-- Checks if the coefficients of a parabola satisfy the required conditions -/
def validParabola (p : Parabola) : Prop :=
  p.a > 0 ∧ Int.gcd p.a.natAbs (Int.gcd p.b.natAbs (Int.gcd p.c.natAbs (Int.gcd p.d.natAbs (Int.gcd p.e.natAbs p.f.natAbs)))) = 1

/-- The main theorem statement -/
theorem parabola_equation (focus : Point) (directrix : Line) :
  ∃ (p : Parabola), 
    focus.x = 2 ∧ 
    focus.y = -1 ∧ 
    directrix.a = 1 ∧ 
    directrix.b = 2 ∧ 
    directrix.c = -5 ∧
    validParabola p ∧
    p.a = 4 ∧ 
    p.b = -4 ∧ 
    p.c = 5 ∧ 
    p.d = 0 ∧ 
    p.e = 10 ∧ 
    p.f = -20 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l425_42528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_foster_dog_food_lasts_six_days_l425_42598

/-- Represents the number of days food will last for foster dogs -/
def foster_dog_food_duration (mom_food_per_meal : ℚ) (mom_meals_per_day : ℕ) 
  (puppy_food_per_meal : ℚ) (puppy_meals_per_day : ℕ) (num_puppies : ℕ) 
  (total_food : ℚ) : ℕ :=
  let mom_food_per_day := mom_food_per_meal * mom_meals_per_day
  let puppy_food_per_day := puppy_food_per_meal * puppy_meals_per_day
  let total_food_per_day := mom_food_per_day + num_puppies * puppy_food_per_day
  (total_food / total_food_per_day).floor.toNat

/-- Theorem: Given the conditions, the food will last for 6 days -/
theorem foster_dog_food_lasts_six_days :
  foster_dog_food_duration (3/2) 3 (1/2) 2 5 57 = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_foster_dog_food_lasts_six_days_l425_42598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l425_42526

-- Define constants
noncomputable def train_length : ℝ := 150 -- meters
noncomputable def train_speed : ℝ := 83.99280057595394 -- kmph
noncomputable def man_speed : ℝ := 6 -- kmph

-- Define the function to calculate the time
noncomputable def time_to_pass (length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : ℝ :=
  let relative_speed_kmph := train_speed + man_speed
  let relative_speed_mps := relative_speed_kmph * 1000 / 3600
  length / relative_speed_mps

-- Theorem statement
theorem train_passing_time :
  abs (time_to_pass train_length train_speed man_speed - 6.00024) < 0.00001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l425_42526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_remainder_l425_42520

-- Define the remainder function as noncomputable
noncomputable def rem (x y : ℝ) : ℝ := x - y * ⌊x / y⌋

-- State the theorem
theorem remainder_of_remainder :
  rem (rem (1/3 : ℝ) (4/7 : ℝ)) (5/9 : ℝ) = 1/3 := by
  -- Proof steps will go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_remainder_l425_42520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_equilibrium_l425_42578

/-- Represents a weight placement strategy on a balance scale. -/
inductive PlacementStrategy
  | LighterSide
  | EitherSide

/-- Represents the state of a balance scale. -/
structure BalanceState where
  left : ℕ
  right : ℕ

/-- Defines the process of placing weights on a balance scale. -/
def place_weights (weights : List ℕ) (strategy : PlacementStrategy) : BalanceState :=
  sorry

/-- Theorem stating that the final state of the balance will be equilibrium. -/
theorem balance_equilibrium (n : ℕ) (weights : List ℕ) :
  weights.length = n + 1 →
  weights.sum = 2 * n →
  ∀ w ∈ weights, w > 0 →
  List.Sorted (· ≥ ·) weights →
  let final_state := place_weights weights PlacementStrategy.LighterSide
  final_state.left = final_state.right :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_balance_equilibrium_l425_42578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trip_cost_l425_42505

-- Define the cities and distances
def X : ℕ := 0
def Y : ℕ := 1
def Z : ℕ := 2

-- Define the distances between cities
noncomputable def distance (a b : ℕ) : ℝ :=
  if (a = X ∧ b = Y) ∨ (a = Y ∧ b = X) then 4500
  else if (a = X ∧ b = Z) ∨ (a = Z ∧ b = X) then 4000
  else Real.sqrt 4250000

-- Define the cost functions
def bus_cost (d : ℝ) : ℝ := 0.20 * d
def plane_cost (d : ℝ) : ℝ := 150 + 0.15 * d

-- Define the cheapest travel option between two cities
noncomputable def cheapest_travel (a b : ℕ) : ℝ :=
  min (bus_cost (distance a b)) (plane_cost (distance a b))

-- Theorem statement
theorem total_trip_cost :
  cheapest_travel X Y + cheapest_travel Y Z + cheapest_travel Z X = 1987.31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_trip_cost_l425_42505
