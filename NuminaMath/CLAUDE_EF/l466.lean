import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carrie_box_jellybeans_l466_46621

/-- Represents a box with dimensions length, width, and height -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box -/
def boxVolume (b : Box) : ℝ := b.length * b.width * b.height

/-- Represents Bert's cube-shaped box -/
noncomputable def bertBox : Box :=
  { length := (216 : ℝ) ^ (1/3),
    width := (216 : ℝ) ^ (1/3),
    height := (216 : ℝ) ^ (1/3) }

/-- Represents Carrie's box derived from Bert's box -/
noncomputable def carrieBox : Box :=
  { length := 2 * bertBox.length,
    width := 2 * bertBox.width,
    height := 3 * bertBox.height }

/-- The number of jellybeans a box can hold is proportional to its volume -/
noncomputable def jellybeansInBox (b : Box) : ℝ := (boxVolume b / boxVolume bertBox) * 216

theorem carrie_box_jellybeans :
  jellybeansInBox carrieBox = 2592 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carrie_box_jellybeans_l466_46621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_selecting_cocaptains_l466_46614

/-- Represents a math team -/
structure MathTeam where
  size : ℕ
  cocaptains : ℕ

/-- The set of math teams in the region -/
def teams : Finset MathTeam := sorry

/-- The probability of selecting a specific team -/
def teamProbability : ℚ := sorry

/-- The probability of selecting two co-captains from a given team -/
def cocaptainProbability (team : MathTeam) : ℚ := sorry

/-- The total probability of selecting two co-captains -/
def totalProbability : ℚ := sorry

theorem probability_of_selecting_cocaptains :
  teams.card = 4 ∧
  (∀ t ∈ teams, t.cocaptains = 3) ∧
  (∃ t₁ t₂ t₃ t₄, t₁ ∈ teams ∧ t₂ ∈ teams ∧ t₃ ∈ teams ∧ t₄ ∈ teams ∧
    t₁.size = 6 ∧ t₂.size = 8 ∧ t₃.size = 9 ∧ t₄.size = 11) ∧
  teamProbability = 1 / 4 →
  totalProbability = 1115 / 18480 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_selecting_cocaptains_l466_46614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_leftmost_vertex_l466_46688

/-- The area of a quadrilateral with vertices on y = e^x -/
noncomputable def quadrilateralArea (n : ℤ) : ℝ :=
  let v1 := (n, Real.exp n)
  let v2 := (n + 1, Real.exp (n + 1))
  let v3 := (n + 2, Real.exp (n + 2))
  let v4 := (n + 3, Real.exp (n + 3))
  1/2 * abs (
    Real.exp n * (n + 1) + Real.exp (n + 1) * (n + 2) + 
    Real.exp (n + 2) * (n + 3) + Real.exp (n + 3) * n - 
    (Real.exp (n + 1) * n + Real.exp (n + 2) * (n + 1) + 
     Real.exp (n + 3) * (n + 2) + Real.exp n * (n + 3))
  )

theorem quadrilateral_leftmost_vertex :
  ∃ n : ℤ, n ≥ 0 ∧ quadrilateralArea n = 1/2 ∧ 
  (∀ m : ℤ, m ≥ 0 ∧ quadrilateralArea m = 1/2 → m ≥ n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_leftmost_vertex_l466_46688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_theorem_l466_46653

/-- Represents the project parameters and calculates the minimum number of workers needed --/
def minimum_workers (total_days : ℕ) (days_worked : ℕ) (initial_workers : ℕ) (fraction_completed : ℚ) : ℕ :=
  let remaining_days := total_days - days_worked
  let remaining_work := 1 - fraction_completed
  let work_rate_per_person_per_day := fraction_completed / (initial_workers * days_worked)
  let required_workers := (remaining_work / (work_rate_per_person_per_day * remaining_days)).ceil.toNat
  required_workers

/-- Theorem stating that the minimum number of workers needed is 4 --/
theorem project_completion_theorem :
  minimum_workers 30 6 10 (1/4) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_completion_theorem_l466_46653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflect_and_translate_circle_center_l466_46636

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Translates a point vertically by a given amount -/
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

/-- Theorem: Reflecting (3, -4) across the y-axis and translating up by 5 results in (-3, 1) -/
theorem reflect_and_translate_circle_center : 
  let original : ℝ × ℝ := (3, -4)
  let reflected := reflect_y original
  let translated := translate_y reflected 5
  translated = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflect_and_translate_circle_center_l466_46636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nursing_home_theorem_l466_46671

/-- Represents the number of single rooms -/
def t : ℕ := 25  -- We provide a default value to avoid errors

/-- The total number of rooms -/
def total_rooms : ℕ := 100

/-- Constraint on the number of single rooms -/
def t_range : Prop := 10 ≤ t ∧ t ≤ 30

/-- Number of double rooms is twice the number of single rooms -/
def double_rooms : ℕ := 2 * t

/-- Number of triple rooms -/
def triple_rooms : ℕ := total_rooms - t - double_rooms

/-- Total number of nursing beds -/
def total_beds (t : ℕ) : ℕ := t + 2 * (2 * t) + 3 * (total_rooms - t - 2 * t)

theorem nursing_home_theorem :
  (total_beds t = 200 → t = 25) ∧
  (∀ t' : ℕ, 10 ≤ t' ∧ t' ≤ 30 → total_beds t' ≤ 260) ∧
  (∀ t' : ℕ, 10 ≤ t' ∧ t' ≤ 30 → total_beds t' ≥ 180) :=
by
  sorry

#eval total_beds 25  -- This will evaluate the function for t = 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nursing_home_theorem_l466_46671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_l466_46655

/-- A shop owner who cheats by 20% while buying and selling makes approximately 50% profit. -/
theorem shopkeeper_profit (cost_price : ℝ) (cost_price_positive : cost_price > 0) : 
  let buying_quantity := 1.2 * cost_price
  let selling_quantity := 0.8 * cost_price
  let actual_cost := (selling_quantity / buying_quantity) * cost_price
  let profit := cost_price - actual_cost
  let percentage_profit := (profit / actual_cost) * 100
  ∃ ε > 0, |percentage_profit - 50| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_l466_46655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_repeating_decimals_as_fraction_l466_46665

/-- Expresses the sum of two repeating decimals as a fraction in lowest terms -/
theorem sum_repeating_decimals_as_fraction :
  ∃ (a b : ℚ), (a = 2/9 ∧ b = 4/99) → a + b = 200 / 769 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_repeating_decimals_as_fraction_l466_46665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_rule_for_9_l466_46601

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem divisibility_rule_for_9 (n : ℕ) (hn : n = 27 ∨ n = 36 ∨ n = 45 ∨ n = 81) :
  (sum_of_digits n) % 9 = 0 → n % 9 = 0 := by
  sorry

#eval sum_of_digits 27
#eval sum_of_digits 36
#eval sum_of_digits 45
#eval sum_of_digits 81

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_rule_for_9_l466_46601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_increase_for_safe_overtake_l466_46651

/-- Represents a car with its speed and position -/
structure Car where
  speed : ℝ
  position : ℝ

/-- Calculates the time for one car to overtake another -/
noncomputable def overtakeTime (a b : Car) : ℝ :=
  (b.position - a.position) / (a.speed - b.speed)

/-- Calculates the time before two cars moving towards each other would meet -/
noncomputable def meetTime (a c : Car) : ℝ :=
  (c.position - a.position) / (a.speed + c.speed)

/-- Calculates the new distance between two cars after a given time -/
noncomputable def newDistance (a c : Car) (t : ℝ) : ℝ :=
  c.position - a.position - (a.speed + c.speed) * t

/-- Theorem: Minimum speed increase for safe overtake -/
theorem min_speed_increase_for_safe_overtake 
  (a b c : Car)
  (h1 : a.speed = 65)
  (h2 : b.speed = 50)
  (h3 : c.speed = 70)
  (h4 : b.position - a.position = 50)
  (h5 : c.position - a.position = 300)
  (h6 : ∀ Δv, Δv ≥ 0 → 
    let t_overtake := overtakeTime ⟨a.speed + Δv, a.position⟩ b
    let t_meet := meetTime a c
    t_overtake < t_meet ∧ 
    newDistance ⟨a.speed, a.position + (a.speed + Δv) * t_overtake⟩ 
                ⟨c.speed, c.position - c.speed * t_overtake⟩ 
                (t_meet - t_overtake) ≥ 100 →
    Δv ≥ 20) :
  ∃ Δv : ℝ, Δv = 20 ∧ 
    let t_overtake := overtakeTime ⟨a.speed + Δv, a.position⟩ b
    let t_meet := meetTime a c
    t_overtake < t_meet ∧ 
    newDistance ⟨a.speed, a.position + (a.speed + Δv) * t_overtake⟩ 
                ⟨c.speed, c.position - c.speed * t_overtake⟩ 
                (t_meet - t_overtake) ≥ 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_speed_increase_for_safe_overtake_l466_46651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l466_46610

theorem triangle_problem (A B C a b c : Real) :
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧  -- Angle constraints
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧  -- Side length constraints
  (A + B + C = π) ∧  -- Angle sum in a triangle
  (a * Real.cos B = 3) ∧  -- Given condition
  (b * Real.cos A = 1) ∧  -- Given condition
  (A - B = π / 6) →  -- Given condition
  (c = 4) ∧ (B = π / 6) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l466_46610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_of_line_l466_46620

/-- The direction vector of a parameterized line. -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- The line y = (2x + 1) / 3 -/
noncomputable def line (x : ℝ) : ℝ × ℝ :=
  (x, (2 * x + 1) / 3)

/-- The parameterized form of the line -/
noncomputable def parameterized_line (t : ℝ) : ℝ × ℝ :=
  ((-1, 0) : ℝ × ℝ) + t • (3 / Real.sqrt 13, 2 / Real.sqrt 13)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem direction_vector_of_line :
  direction_vector parameterized_line = (3 / Real.sqrt 13, 2 / Real.sqrt 13) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_of_line_l466_46620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_is_160_l466_46616

/-- Represents the components of a restaurant's meal sales and inventory --/
structure RestaurantData where
  beef_pounds : ℚ
  beef_price : ℚ
  pork_pounds : ℚ
  pork_price : ℚ
  chicken_pounds : ℚ
  chicken_price : ℚ
  deluxe_beef : ℚ
  deluxe_pork : ℚ
  deluxe_chicken : ℚ
  deluxe_price : ℚ
  standard_beef : ℚ
  standard_pork : ℚ
  standard_price : ℚ

/-- Calculates the total profit from meal sales given the restaurant data --/
def calculate_profit (data : RestaurantData) : ℚ :=
  let total_cost := data.beef_pounds * data.beef_price + 
                    data.pork_pounds * data.pork_price + 
                    data.chicken_pounds * data.chicken_price
  let num_meals := min (data.beef_pounds / (data.deluxe_beef + data.standard_beef))
                       (min (data.pork_pounds / (data.deluxe_pork + data.standard_pork))
                            (data.chicken_pounds / data.deluxe_chicken))
  let total_revenue := num_meals * (data.deluxe_price + data.standard_price)
  total_revenue - total_cost

/-- Theorem stating that the total profit is $160 given the specific conditions --/
theorem profit_is_160 : 
  let data : RestaurantData := {
    beef_pounds := 20, beef_price := 8,
    pork_pounds := 10, pork_price := 6,
    chicken_pounds := 5, chicken_price := 4,
    deluxe_beef := 2, deluxe_pork := 1, deluxe_chicken := 1, deluxe_price := 50,
    standard_beef := 3/2, standard_pork := 3/4, standard_price := 30
  }
  calculate_profit data = 160 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_is_160_l466_46616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l466_46657

-- Define the spinner with three parts
inductive Spinner
| A
| B
| C

-- Define the probability function
def prob : Spinner → ℚ
| Spinner.A => 1/3  -- Probability of landing on A
| Spinner.B => 5/12 -- Probability of landing on B
| Spinner.C => 1/4  -- Probability of landing on C (to be proved)

-- Theorem statement
theorem spinner_probability :
  (prob Spinner.A = 1/3) ∧ (prob Spinner.B = 5/12) →
  (prob Spinner.A + prob Spinner.B + prob Spinner.C = 1) →
  prob Spinner.C = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinner_probability_l466_46657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_quadrants_l466_46691

-- Define the function f(x) = a^x + b
noncomputable def f (a b x : ℝ) : ℝ := a^x + b

-- State the theorem
theorem graph_quadrants (a b : ℝ) (ha : a > 1) (hb : -1 < b ∧ b < 0) :
  ∃ (x₁ x₂ x₃ : ℝ),
    -- Point in quadrant I
    f a b x₁ > 0 ∧ x₁ > 0 ∧
    -- Point in quadrant II
    f a b x₂ > 0 ∧ x₂ < 0 ∧
    -- Point in quadrant III
    f a b x₃ < 0 ∧ x₃ < 0 ∧
    -- No point in quadrant IV
    ∀ x : ℝ, x > 0 → f a b x > 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_quadrants_l466_46691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_algorithm_statements_l466_46604

/-- Definition of an algorithm -/
structure Algorithm where
  steps : List String
  yieldsDefiniteResult : Bool
  endsInFiniteSteps : Bool

/-- A correct algorithm yields a definite result -/
axiom correct_algorithm_yields_result (a : Algorithm) :
  a.yieldsDefiniteResult = true

/-- A correct algorithm ends in a finite number of steps -/
axiom correct_algorithm_ends_in_finite_steps (a : Algorithm) :
  a.endsInFiniteSteps = true

/-- Predicate to represent that an algorithm solves a problem -/
def solves (a : Algorithm) (p : String) : Prop := sorry

/-- Multiple algorithms can solve the same problem -/
axiom multiple_algorithms_for_problem (p : String) :
  ∃ (a1 a2 : Algorithm), a1 ≠ a2 ∧ (solves a1 p) ∧ (solves a2 p)

/-- Main theorem: Correct statements about algorithms -/
theorem correct_algorithm_statements :
  (∀ (a : Algorithm), a.yieldsDefiniteResult = true) ∧
  (∃ (p : String) (a1 a2 : Algorithm), a1 ≠ a2 ∧ (solves a1 p) ∧ (solves a2 p)) ∧
  (∀ (a : Algorithm), a.endsInFiniteSteps = true) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_algorithm_statements_l466_46604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_line_segment_example_l466_46626

/-- Given two points on a plane and a ratio, this function calculates the coordinates
    of the point that divides the line segment between the two points in the given ratio. -/
noncomputable def divideLineSegment (x₁ y₁ x₂ y₂ m n : ℝ) : ℝ × ℝ :=
  ((m * x₂ + n * x₁) / (m + n), (m * y₂ + n * y₁) / (m + n))

/-- Theorem stating that the point (3, 2.75) divides the line segment
    joining (2.5, 3.5) and (4.5, 0.5) in the ratio 1:3. -/
theorem divide_line_segment_example :
  divideLineSegment 2.5 3.5 4.5 0.5 1 3 = (3, 2.75) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divide_line_segment_example_l466_46626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_transform_1234_to_2002_l466_46666

/-- Represents a four-digit number --/
structure FourDigitNumber where
  digits : Fin 4 → Nat
  is_valid : ∀ i, digits i < 10

/-- Allowed operations on four-digit numbers --/
inductive Operation
  | add_neighbors (i : Fin 3) : Operation
  | subtract_neighbors (i : Fin 3) : Operation

/-- Applies an operation to a four-digit number --/
def apply_operation (n : FourDigitNumber) (op : Operation) : Option FourDigitNumber :=
  sorry

/-- Calculates the invariant M = (d + b) - (a + c) for a four-digit number --/
def calculate_M (n : FourDigitNumber) : Int :=
  (n.digits 3 + n.digits 1) - (n.digits 0 + n.digits 2)

/-- Theorem stating that the transformation is impossible --/
theorem cannot_transform_1234_to_2002 :
  ∀ (ops : List Operation),
    let start := FourDigitNumber.mk (λ i => [1, 2, 3, 4].get ⟨i.val, by simp⟩) (by sorry)
    let end_goal := FourDigitNumber.mk (λ i => [2, 0, 0, 2].get ⟨i.val, by simp⟩) (by sorry)
    ¬ ∃ (result : FourDigitNumber),
      (ops.foldl (λ acc op => acc.bind (λ n => apply_operation n op)) (some start) = some result)
      ∧ (result = end_goal) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cannot_transform_1234_to_2002_l466_46666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l466_46679

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem problem_statement (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧ 
  (∃ m n : ℕ+, 
    (Real.sqrt (log a) : ℝ) = m ∧
    (Real.sqrt (log b) : ℝ) = n ∧
    (log (Real.sqrt a) : ℝ) = m^2 / 2 ∧
    (log (Real.sqrt b) : ℝ) = n^2 / 2 ∧
    (log (a * b) : ℝ) = m^2 + n^2) ∧
  Real.sqrt (log a) + Real.sqrt (log b) + log (Real.sqrt a) + log (Real.sqrt b) + log (a * b) = 150 →
  a + b = 2 * (10 : ℝ)^49 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l466_46679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reward_function_minimum_a_l466_46683

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (15 * x - a) / (x + 8)

theorem reward_function_minimum_a :
  ∃ (a : ℕ), a = 315 ∧
  (MonotoneOn (f a) (Set.Icc 50 500)) ∧
  (∀ x ∈ Set.Icc 50 500, f a x ≥ 7) ∧
  (∀ x ∈ Set.Icc 50 500, f a x ≤ 0.15 * x) ∧
  (∀ a' : ℕ, a' < 315 →
    ¬(MonotoneOn (f a') (Set.Icc 50 500) ∧
      (∀ x ∈ Set.Icc 50 500, f a' x ≥ 7 ∧ f a' x ≤ 0.15 * x))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reward_function_minimum_a_l466_46683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l466_46689

noncomputable def f (x : ℝ) := Real.log (x^2 - 2*x - 3) / Real.log (1/2)

theorem f_monotone_increasing :
  StrictMonoOn f (Set.Iio (-1)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l466_46689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_crosses_asymptote_l466_46624

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 * x^2 - 5 * x - 7) / (x^2 - 4 * x + 1)

-- Define the horizontal asymptote
def horizontal_asymptote : ℝ := 2

-- Theorem statement
theorem function_crosses_asymptote :
  f 3 = horizontal_asymptote := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_crosses_asymptote_l466_46624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_n_b_n_eq_25_l466_46611

/-- Definition of the function M -/
noncomputable def M (x : ℝ) : ℝ := x - x^3 / 3

/-- Definition of b_n using n iterations of M -/
noncomputable def b (n : ℕ+) : ℝ := (Nat.iterate (fun x => M x) n.val (25 / n.val))

/-- The limit of n * b_n as n approaches infinity is 25 -/
theorem limit_n_b_n_eq_25 : 
  Filter.Tendsto (fun n : ℕ+ => (n : ℝ) * b n) Filter.atTop (nhds 25) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_n_b_n_eq_25_l466_46611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_trip_theorem_l466_46634

def trips : List Int := [15, -3, 16, -11, 10, -12, 4, -15, 16, -18]
def fuel_consumption : Real := 0.6
def initial_fuel : Real := 72.2

theorem taxi_trip_theorem :
  let final_position := trips.sum
  let total_distance := (trips.map Int.natAbs).sum
  let fuel_used := fuel_consumption * (total_distance : Real)
  let remaining_fuel := initial_fuel - fuel_used
  final_position = 2 ∧ remaining_fuel ≥ 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_taxi_trip_theorem_l466_46634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_krish_remaining_money_l466_46678

/-- Calculates the remaining money after Krish's expenses --/
def remaining_money (initial_amount : ℝ) (sweets_cost : ℝ) (stickers_cost : ℝ) 
  (friend_gift : ℝ) (num_friends : ℕ) (charity_donation : ℝ) 
  (sent_abroad : ℝ) (exchange_rate_fee : ℝ) : ℝ :=
  initial_amount - (sweets_cost + stickers_cost + friend_gift * (num_friends : ℝ) + 
  charity_donation + sent_abroad * (1 + exchange_rate_fee))

/-- Theorem stating that Krish's remaining money is $16.40 --/
theorem krish_remaining_money : 
  remaining_money 200.50 35.25 10.75 25.20 4 15.30 20 0.1 = 16.40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_krish_remaining_money_l466_46678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_decreasing_intervals_l466_46612

-- Define the function f
noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

-- Define the theorem
theorem symmetry_and_decreasing_intervals 
  (φ : ℝ) 
  (h1 : -π < φ ∧ φ < 0) 
  (h2 : ∀ x : ℝ, f x φ = f ((π/4) - x) φ) : 
  (φ = -3*π/4) ∧ 
  (∀ k : ℤ, ∀ x ∈ Set.Icc (5*π/8 + k*π) (9*π/8 + k*π), 
    ∀ y ∈ Set.Icc (5*π/8 + k*π) (9*π/8 + k*π), 
    x ≤ y → f y φ ≤ f x φ) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_decreasing_intervals_l466_46612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coinciding_rest_days_total_coinciding_rest_days_l466_46618

/-- Al's schedule cycle length -/
def al_cycle : ℕ := 4

/-- Barb's schedule cycle length -/
def barb_cycle : ℕ := 10

/-- Total number of days -/
def total_days : ℕ := 1000

/-- Al's rest day in his cycle -/
def al_rest_day : ℕ := 4

/-- Barb's rest days in her cycle -/
def barb_rest_days : Finset ℕ := {8, 9, 10}

/-- The number of coinciding rest days in a complete cycle of both schedules -/
def coinciding_rest_days_per_cycle : ℕ := 2

theorem coinciding_rest_days (d : ℕ) : 
  d ≤ total_days → 
  (d % al_cycle = al_rest_day ∧ d % barb_cycle ∈ barb_rest_days) ↔ 
  d % (Nat.lcm al_cycle barb_cycle) ∈ ({8, 20} : Finset ℕ) :=
sorry

theorem total_coinciding_rest_days : 
  (total_days / (Nat.lcm al_cycle barb_cycle)) * coinciding_rest_days_per_cycle = 100 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coinciding_rest_days_total_coinciding_rest_days_l466_46618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_percentage_after_hiring_l466_46696

theorem women_percentage_after_hiring (initial_workers : ℕ) (initial_men_fraction : ℚ) 
  (new_hires : ℕ) (h1 : initial_workers = 90) (h2 : initial_men_fraction = 2/3) 
  (h3 : new_hires = 10) : 
  let initial_women := initial_workers - (initial_men_fraction * ↑initial_workers).floor
  let total_workers := initial_workers + new_hires
  let total_women := initial_women + new_hires
  (total_women : ℚ) / ↑total_workers * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_percentage_after_hiring_l466_46696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l466_46667

/-- The equations of asymptotes for a hyperbola with given eccentricity -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.sqrt (1 + b^2 / a^2) = Real.sqrt 5 / 2) →
  (∃ k : ℝ, k = 1/2 ∧ ∀ x y : ℝ, (y = k*x ∨ y = -k*x) → x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l466_46667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sum_l466_46692

theorem tan_value_from_sum (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = -Real.sqrt 10 / 5)
  (h2 : 0 < α ∧ α < Real.pi) : 
  Real.tan α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sum_l466_46692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_product_derivative_cos_over_x_derivative_ln_quadratic_l466_46654

-- Problem 1
theorem derivative_product (x : ℝ) :
  deriv (λ x => (x^3 + 1) * (x - 1)) x = 4*x^3 - 3*x^2 + x - 2 :=
by sorry

-- Problem 2
theorem derivative_cos_over_x (x : ℝ) (h : x ≠ 0) :
  deriv (λ x => Real.cos x / x) x = (-x * Real.sin x - Real.cos x) / x^2 :=
by sorry

-- Problem 3
theorem derivative_ln_quadratic (x : ℝ) :
  deriv (λ x => Real.log (2*x^2 + 1)) x = 4*x / (2*x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_product_derivative_cos_over_x_derivative_ln_quadratic_l466_46654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_lines_eccentricity_range_l466_46661

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- Represents a line passing through the left focus of a hyperbola -/
structure FocalLine (h : Hyperbola) where
  intersects_hyperbola : Prop
  chord_length_eq_4b : Prop

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The main theorem statement -/
theorem hyperbola_focal_lines_eccentricity_range (h : Hyperbola) :
  (∃ l1 l2 : FocalLine h, l1 ≠ l2 ∧ ∀ l : FocalLine h, l = l1 ∨ l = l2) →
  let e := eccentricity h
  (e ∈ Set.Ioo 1 (Real.sqrt 5 / 2) ∪ Set.Ioi (Real.sqrt 5)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_lines_eccentricity_range_l466_46661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integrality_l466_46638

def sequence_next (m : ℕ) (a : ℕ) : ℕ :=
  if a < 2^m then a^2 + 2^m else a / 2

def is_valid_sequence (m : ℕ) (a₁ : ℕ) : Prop :=
  ∀ n : ℕ, ∃ aₙ : ℕ,
    aₙ = (Nat.iterate (sequence_next m) n a₁) ∧
    (Nat.iterate (sequence_next m) (n + 1) a₁) = sequence_next m aₙ

theorem sequence_integrality (m : ℕ) (a₁ : ℕ) :
  (m > 0 ∧ is_valid_sequence m a₁) ↔ (m = 2 ∧ ∃ ℓ : ℕ, ℓ > 0 ∧ a₁ = 2^ℓ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integrality_l466_46638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increment_not_zero_l466_46623

/-- Definition of average rate of change -/
noncomputable def average_rate_of_change (f : ℝ → ℝ) (x : ℝ) (Δx : ℝ) : ℝ :=
  (f (x + Δx) - f x) / Δx

/-- Theorem: The increment Δx in the average rate of change must not be zero -/
theorem increment_not_zero (f : ℝ → ℝ) (x : ℝ) :
  ∀ Δx, average_rate_of_change f x Δx ∈ Set.univ → Δx ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increment_not_zero_l466_46623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_X_to_Z_is_5_hours_l466_46629

/-- Represents a town in the problem -/
inductive Town : Type
  | X : Town
  | Y : Town
  | Z : Town

/-- The speed of travel between two towns in miles per hour -/
def speed (a b : Town) : ℝ :=
  match a, b with
  | Town.X, Town.Z => 50
  | Town.Z, Town.Y => 60
  | _, _ => 0  -- For other combinations, we don't have information

/-- The time taken to travel from Z to Y in hours -/
def time_Z_to_Y : ℝ := 2.0833333333333335

/-- Theorem stating that the time taken to drive from X to Z is 5 hours -/
theorem time_X_to_Z_is_5_hours :
  let distance_Z_Y := speed Town.Z Town.Y * time_Z_to_Y
  let total_distance := 2 * distance_Z_Y  -- Y is midway
  total_distance / speed Town.X Town.Z = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_X_to_Z_is_5_hours_l466_46629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_candy_pieces_l466_46605

def candy_jar_count (red_pieces blue_pieces : ℕ) : ℕ :=
  red_pieces + blue_pieces

theorem total_candy_pieces : candy_jar_count 145 3264 = 3409 := by
  -- Unfold the definition of candy_jar_count
  unfold candy_jar_count
  -- Simplify the addition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_candy_pieces_l466_46605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mars_mission_cost_per_person_l466_46646

/-- Calculates the per-person cost in dollars for a Mars mission -/
theorem mars_mission_cost_per_person 
  (total_cost_euros : ℝ) 
  (num_people : ℝ) 
  (exchange_rate : ℝ) 
  (h1 : total_cost_euros = 30000000000) 
  (h2 : num_people = 500000000) 
  (h3 : exchange_rate = 1.1) : 
  (total_cost_euros * exchange_rate) / num_people = 66 := by
  sorry

#check mars_mission_cost_per_person

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mars_mission_cost_per_person_l466_46646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l466_46677

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sin x ^ 2 - 1/2

theorem f_properties :
  -- The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 → (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- The maximum value is √2/2
  (∀ (x : ℝ), f x ≤ Real.sqrt 2 / 2) ∧
  (∃ (x : ℝ), f x = Real.sqrt 2 / 2) ∧
  -- If α ∈ (0, π/2) and f(α) = √2/2, then α = 3π/8
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi / 2 → f α = Real.sqrt 2 / 2 → α = 3 * Real.pi / 8) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l466_46677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l466_46637

/-- Represents a parallelogram in a coordinate system -/
structure Parallelogram where
  x : ℝ
  y : ℝ
  z : ℝ
  h : z < x

/-- Represents a point in the coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle formed by joining endpoints of segments to point P -/
structure Triangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := (1 / 2) * t.base * t.height

/-- The main theorem stating the ratio of areas of triangles A and B -/
theorem area_ratio_theorem (para : Parallelogram) (n m : ℕ) (P : Point)
  (A B : Triangle) (h_P : P.x = (1/3) * (para.x + para.z) ∧ P.y = (1/3) * para.y)
  (h_A : A.base = para.y / n ∧ A.height = (para.x + para.z) / 3)
  (h_B : B.base = para.x / m ∧ B.height = para.y / 3) :
  triangleArea A / triangleArea B = m * (para.x + para.z) / (n * para.x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_theorem_l466_46637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_center_travel_is_2pi_wheel_center_travel_equals_circumference_l466_46607

/-- The horizontal distance traveled by the center of a wheel with radius 1 meter 
    when rolled through one complete revolution on a flat horizontal surface. -/
noncomputable def wheel_center_travel : ℝ := 2 * Real.pi

/-- Theorem stating that the horizontal distance traveled by the center of a wheel 
    with radius 1 meter, when rolled through one complete revolution on a flat 
    horizontal surface, is equal to 2π meters. -/
theorem wheel_center_travel_is_2pi : 
  wheel_center_travel = 2 * Real.pi := by
  -- Unfold the definition of wheel_center_travel
  unfold wheel_center_travel
  -- The equality is now trivial
  rfl

/-- Theorem stating that the horizontal distance traveled by the center of the wheel
    is equal to the circumference of the wheel. -/
theorem wheel_center_travel_equals_circumference (radius : ℝ) (h : radius > 0) :
  2 * Real.pi * radius = 2 * Real.pi * radius := by
  rfl

#check wheel_center_travel_is_2pi
#check wheel_center_travel_equals_circumference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_center_travel_is_2pi_wheel_center_travel_equals_circumference_l466_46607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l466_46648

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 3 * Real.cos (ω * x + Real.pi / 6) - Real.sin (ω * x - Real.pi / 3)

theorem max_value_of_f (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, f ω (x + Real.pi) = f ω x) :
  ∃ x₀ ∈ Set.Icc 0 (Real.pi / 2), ∀ x ∈ Set.Icc 0 (Real.pi / 2), f ω x ≤ f ω x₀ ∧ f ω x₀ = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l466_46648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_trig_l466_46682

theorem third_quadrant_trig (α : Real) : 
  α ∈ Set.Icc π (3*π/2) →  -- α is in the third quadrant
  Real.tan α = 3/4 →              -- tan α = 3/4
  Real.sin α + Real.cos α = -7/5 :=    -- prove that sin α + cos α = -7/5
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_quadrant_trig_l466_46682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_l466_46684

/-- Represents a point in the hexagonal lattice --/
inductive Point
| A
| Red1 | Red2
| Blue1 | Blue2 | Blue3 | Blue4
| Green1 | Green2 | Green3 | Green4
| Orange1 | Orange2
| B

/-- Represents a directed edge in the hexagonal lattice --/
def Edge := Point × Point

/-- The set of all valid edges in the lattice --/
def valid_edges : Set Edge := sorry

/-- A path in the lattice is a list of edges --/
def LatticeRoute := List Edge

/-- Checks if a path is valid according to the rules --/
def is_valid_path (p : LatticeRoute) : Prop := sorry

/-- Counts the number of valid paths from A to B --/
noncomputable def count_paths : ℕ := sorry

/-- The main theorem stating that there are 2400 valid paths from A to B --/
theorem path_count : count_paths = 2400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_count_l466_46684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_test_scores_l466_46687

def test_scores : List Int := [94, 92, 83, 81, 76, 65]

theorem correct_test_scores :
  let scores := test_scores
  (scores.length = 6) ∧
  (scores.sum / scores.length = 79) ∧
  ((scores.take 4).sum / 4 = 76) ∧
  (scores.take 2 = [81, 65]) ∧
  (∀ s ∈ scores, s < 95) ∧
  (scores.Nodup) ∧
  (scores = scores.reverse) := by
  sorry

#eval test_scores
#eval test_scores.length
#eval test_scores.sum / test_scores.length
#eval (test_scores.take 4).sum / 4
#eval test_scores.take 2
#eval test_scores.all (· < 95)
#eval test_scores.Nodup
#eval test_scores = test_scores.reverse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_test_scores_l466_46687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_reciprocals_l466_46627

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then (4 * x) / (1 + x^2) else -4 / x

theorem range_of_sum_reciprocals (t : ℝ) (x₁ x₂ x₃ : ℝ) 
  (h₁ : f x₁ = t) (h₂ : f x₂ = t) (h₃ : f x₃ = t)
  (h_distinct : x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃)
  (h_order : x₁ < x₂ ∧ x₂ < x₃) :
  ∃ y, y > 5/2 ∧ y = -1/x₁ + 1/x₂ + 1/x₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_reciprocals_l466_46627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arbitrarily_large_digit_sum_l466_46628

/-- Helper function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

/-- Given a natural number a ≥ 2 that is not divisible by 2 or 5, 
    prove that for any positive integer K, there exists a positive integer M 
    such that the sum of digits of a^m is greater than K for all m ≥ M. -/
theorem arbitrarily_large_digit_sum 
  (a : ℕ) 
  (h_a_ge_two : a ≥ 2) 
  (h_a_not_div_two : ¬ 2 ∣ a) 
  (h_a_not_div_five : ¬ 5 ∣ a) :
  ∀ K : ℕ+, ∃ M : ℕ+, ∀ m : ℕ, m ≥ M → (sum_of_digits (a^m)) > K :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arbitrarily_large_digit_sum_l466_46628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_in_congruent_right_triangles_l466_46652

-- Define the necessary structures and functions
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

def isCongruentTriangle (t1 t2 : Triangle) : Prop := sorry
def isRightTriangle (t : Triangle) : Prop := sorry
def angleMeasure (t : Triangle) (i : Fin 3) : ℝ := sorry
def hypotenuse (t : Triangle) : Set (ℝ × ℝ) := sorry
def intersect (s1 s2 : Set (ℝ × ℝ)) : Prop := sorry
def formsObtuseAngle (s1 s2 : Set (ℝ × ℝ)) (α : ℝ) : Prop := sorry

theorem obtuse_angle_in_congruent_right_triangles :
  ∀ (α : ℝ),
    (∃ (triangle1 triangle2 : Triangle),
      isCongruentTriangle triangle1 triangle2 ∧
      isRightTriangle triangle1 ∧
      isRightTriangle triangle2 ∧
      (∃ (a b : ℝ), angleMeasure triangle1 0 = a ∧ angleMeasure triangle1 1 = b ∧ a = 40 ∧ b = 50) ∧
      (∃ (c d : ℝ), angleMeasure triangle2 0 = c ∧ angleMeasure triangle2 1 = d ∧ c = 40 ∧ d = 50) ∧
      (let h1 := hypotenuse triangle1
       let h2 := hypotenuse triangle2
       intersect h1 h2 ∧
       formsObtuseAngle h1 h2 α)) →
    α = 170 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_angle_in_congruent_right_triangles_l466_46652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_79_l466_46699

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def scores : List ℕ := [73, 74, 79, 84]

theorem last_score_is_79 (h1 : ∀ n : ℕ, n ≤ 4 → n > 0 → (scores.take n).sum % n = 0)
                         (h2 : is_prime 73 ∧ is_prime 79)
                         (h3 : ¬ is_prime 74 ∧ ¬ is_prime 84) :
  scores.reverse.head! = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_score_is_79_l466_46699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l466_46615

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (3 * x^2 - 2 * a * x) / Real.log a

-- Define the property of f being decreasing on the interval [1/2, 1]
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, 1/2 ≤ x ∧ x < y ∧ y ≤ 1 → f a x > f a y

-- Theorem statement
theorem range_of_a (a : ℝ) :
  is_decreasing_on_interval a → 0 < a ∧ a < 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l466_46615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l466_46669

noncomputable def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem diamond_calculation : 
  diamond (diamond 7 24) (diamond (-24) 7) = 25 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_calculation_l466_46669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_some_is_all_not_l466_46642

-- Define the universe of triangles
variable (Triangle : Type)

-- Define the property of being an isosceles triangle
variable (isIsosceles : Triangle → Prop)

-- Define the original proposition
def someTrianglesAreIsosceles (Triangle : Type) (isIsosceles : Triangle → Prop) : Prop :=
  ∃ t : Triangle, isIsosceles t

-- Define the negation we want to prove
def allTrianglesAreNotIsosceles (Triangle : Type) (isIsosceles : Triangle → Prop) : Prop :=
  ∀ t : Triangle, ¬(isIsosceles t)

-- Theorem statement
theorem negation_of_some_is_all_not (Triangle : Type) (isIsosceles : Triangle → Prop) :
  ¬(someTrianglesAreIsosceles Triangle isIsosceles) ↔ allTrianglesAreNotIsosceles Triangle isIsosceles :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_some_is_all_not_l466_46642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l466_46675

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (x^2 + m*x + 1)

-- Define proposition p
def p (m : ℝ) : Prop := ∀ x : ℝ, (x^2 + m*x + 1) ≥ 0

-- Define proposition q
def q (m : ℝ) : Prop := m > 0 ∧ m < 2

-- Define the range of m
def m_range (m : ℝ) : Prop := (m ≥ -2 ∧ m ≤ 0) ∨ m = 2

-- Theorem statement
theorem range_of_m :
  (∀ m : ℝ, (p m ∧ q m) = False) →
  (∀ m : ℝ, (p m ∨ q m) = True) →
  (∀ m : ℝ, m_range m ↔ (p m ∨ q m)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l466_46675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_3_equals_31_over_4_l466_46600

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 5 / (3 - x)

-- Define the function g using f⁻¹
noncomputable def g (x : ℝ) : ℝ := 1 / (f⁻¹ x) + 7

-- Theorem statement
theorem g_of_3_equals_31_over_4 : g 3 = 31 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_of_3_equals_31_over_4_l466_46600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l466_46622

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 5

-- Theorem statement
theorem parabola_circle_intersection :
  ∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ),
    -- Parabola intersects x-axis at two points
    parabola x₁ = 0 ∧ parabola x₂ = 0 ∧ x₁ ≠ x₂ ∧
    -- Parabola intersects y-axis at one point
    parabola 0 = y₃ ∧
    -- These points lie on the circle
    circle_equation x₁ 0 ∧ circle_equation x₂ 0 ∧ circle_equation 0 y₃ := by
  sorry

#check parabola_circle_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_l466_46622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_sum_l466_46668

theorem logarithm_sum (a b : ℝ) : 
  a = Real.log 25 → b = Real.log 36 → (6 : ℝ)^(a/b) + (5 : ℝ)^(b/a) = 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_sum_l466_46668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_triangle_area_l466_46617

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with center at origin -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Check if a point is on the ellipse -/
def isOnEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- Check if a line passes through a point -/
def linePassesThrough (A B P : Point) : Prop :=
  (P.y - A.y) * (B.x - A.x) = (B.y - A.y) * (P.x - A.x)

/-- Check if a triangle is inscribed in an ellipse -/
def isInscribed (t : Triangle) (e : Ellipse) : Prop :=
  isOnEllipse t.A e ∧ isOnEllipse t.B e ∧ isOnEllipse t.C e

/-- Calculate the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)) / 2

/-- The main theorem -/
theorem max_inscribed_triangle_area 
  (e : Ellipse) 
  (P : Point) 
  (h1 : e.a = 3 ∧ e.b = 2) 
  (h2 : P.x = 1 ∧ P.y = 0) : 
  ∃ (t : Triangle), 
    isInscribed t e ∧ 
    linePassesThrough t.A t.B P ∧
    ∀ (t' : Triangle), 
      isInscribed t' e → 
      linePassesThrough t'.A t'.B P → 
      triangleArea t' ≤ triangleArea t ∧
      triangleArea t = 16 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_inscribed_triangle_area_l466_46617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stephanie_remaining_payment_l466_46603

noncomputable def remaining_amount (total : ℝ) (paid_percentage : ℝ) : ℝ :=
  total * (1 - paid_percentage / 100)

noncomputable def discounted_amount (amount : ℝ) (discount_percentage : ℝ) : ℝ :=
  amount * (1 - discount_percentage / 100)

theorem stephanie_remaining_payment (
  electricity_bill : ℝ)
  (gas_bill : ℝ)
  (water_bill : ℝ)
  (internet_bill : ℝ)
  (internet_payments : ℝ)
  (internet_discount : ℝ)
  (heating_bill : ℝ)
  (heating_subsidy : ℝ)
  (garbage_bill : ℝ)
  (phone_bill : ℝ)
  (cable_bill : ℝ)
  (h_electricity : electricity_bill = 180)
  (h_gas : gas_bill = 120)
  (h_water : water_bill = 200)
  (h_internet : internet_bill = 100)
  (h_internet_payments : internet_payments = 30)
  (h_internet_discount : internet_discount = 15)
  (h_heating : heating_bill = 150)
  (h_heating_subsidy : heating_subsidy = 30)
  (h_garbage : garbage_bill = 60)
  (h_phone : phone_bill = 90)
  (h_cable : cable_bill = 80) :
  remaining_amount electricity_bill 75 +
  remaining_amount gas_bill 60 +
  remaining_amount water_bill 25 +
  discounted_amount (internet_bill - internet_payments) internet_discount +
  remaining_amount (discounted_amount heating_bill heating_subsidy) 45 +
  remaining_amount garbage_bill 50 +
  remaining_amount phone_bill 20 +
  discounted_amount cable_bill 40 = 510.25 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stephanie_remaining_payment_l466_46603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verify_l466_46680

/-- The function F that we claim is a solution to the differential equation -/
noncomputable def F (x y : ℝ) : ℝ := Real.log (abs x) - y^2 / x

/-- The differential equation we're solving -/
def diff_eq (x y : ℝ) (dx dy : ℝ) : Prop :=
  (x + y^2) * dx - 2 * x * y * dy = 0

theorem solution_verify (x y : ℝ) (dx dy : ℝ) (h : x ≠ 0) :
  diff_eq x y dx dy ↔ 
  (deriv (fun x => F x y) x * dx + deriv (fun y => F x y) y * dy = 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_verify_l466_46680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_minimum_value_l466_46649

noncomputable section

variable (a : ℝ)

def f (x : ℝ) := a * x - Real.log x
def F (x : ℝ) := Real.exp x + a * x
def g (x : ℝ) := x * Real.exp (a * x - 1) - 2 * a * x + f a x

theorem monotonicity_and_minimum_value 
  (h1 : a < 0) :
  (∀ x ∈ Set.Ioo 0 (Real.log 3), 
    (∀ y ∈ Set.Ioo 0 (Real.log 3), x < y → f a x < f a y ↔ F a x < F a y) → 
    a ∈ Set.Iic (-3)) ∧
  (a ∈ Set.Iic (-1 / Real.exp 2) → 
    ∃ M, (∀ x, g a x ≥ M) ∧ 
    (∀ ε > 0, ∃ x, g a x < M + ε) → 
    M ≥ 0) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_and_minimum_value_l466_46649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_l466_46643

/-- Parabola function -/
def f (x : ℝ) : ℝ := x^2 - 2*x - 8

/-- X-coordinate of point P -/
noncomputable def x_p : ℝ := 1 + Real.sqrt 19

/-- X-coordinate of point R -/
def x_r : ℝ := 4

theorem parabola_distance :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  f x_p = 10 ∧ 
  f x_r = 0 ∧ 
  |x_r - x_p - 1.36| < ε := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_distance_l466_46643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_same_color_final_color_independent_glass_pieces_theorem_l466_46662

/-- Represents the three possible colors of glass pieces -/
inductive Color
  | Red
  | Yellow
  | Blue

/-- Represents the state of glass pieces -/
structure GlassState where
  total : Nat
  red : Nat
  yellow : Nat
  blue : Nat
  sum_constraint : red + yellow + blue = total

/-- Represents an operation on two glass pieces -/
def operation (s : GlassState) (c1 c2 : Color) : GlassState :=
  sorry

/-- Predicate to check if all pieces have the same color -/
def all_same_color (s : GlassState) : Prop :=
  (s.red = s.total) ∨ (s.yellow = s.total) ∨ (s.blue = s.total)

/-- Function to represent multiple iterations of operations -/
def iteration (n : Nat) (s : GlassState) : GlassState :=
  sorry

/-- Theorem stating that all pieces will eventually have the same color -/
theorem eventually_same_color (initial : GlassState) 
  (h : initial.total = 2002) : 
  ∃ (final : GlassState), all_same_color final ∧ 
  ∃ (n : Nat), (iteration n initial = final) :=
  sorry

/-- Theorem stating that the final color is independent of operation order -/
theorem final_color_independent (initial : GlassState) 
  (h : initial.total = 2002) :
  ∀ (final1 final2 : GlassState), 
  (all_same_color final1 ∧ ∃ (n : Nat), (iteration n initial = final1)) →
  (all_same_color final2 ∧ ∃ (m : Nat), (iteration m initial = final2)) →
  final1 = final2 :=
  sorry

/-- Main theorem combining both results -/
theorem glass_pieces_theorem (initial : GlassState) 
  (h : initial.total = 2002) :
  ∃ (final : GlassState), 
    all_same_color final ∧
    ∃ (n : Nat), (iteration n initial = final) ∧
    ∀ (other_final : GlassState),
      (all_same_color other_final ∧ ∃ (m : Nat), (iteration m initial = other_final)) →
      other_final = final :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_same_color_final_color_independent_glass_pieces_theorem_l466_46662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_150_degrees_l466_46673

/-- The angle of inclination of a line given its equation -/
noncomputable def angleOfInclination (a b c : ℝ) : ℝ :=
  Real.arctan (-a / b) * (180 / Real.pi)

/-- Theorem: The angle of inclination of the line x + √3 y - 3 = 0 is 150° -/
theorem line_inclination_150_degrees :
  angleOfInclination 1 (Real.sqrt 3) (-3) = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_150_degrees_l466_46673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_problem_l466_46650

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem vector_dot_product_problem (x : ℝ) 
  (h1 : x ∈ Set.Icc (-Real.pi/4) (Real.pi/4))
  (h2 : f x = 1) :
  x = 0 ∧ 
  (∀ y : ℝ, f (y + Real.pi) = f y) ∧
  (∀ k : ℤ, Set.Icc (k * Real.pi + Real.pi/6) (k * Real.pi + 2*Real.pi/3) ⊆ 
    {x | ∀ y ∈ Set.Icc x (k * Real.pi + 2*Real.pi/3), f y ≤ f x}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_dot_product_problem_l466_46650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_choice_statistics_l466_46693

/-- The probability of choosing a cultural landscape -/
noncomputable def p : ℝ := 2/3

/-- The probability of choosing a natural landscape -/
noncomputable def q : ℝ := 1/3

/-- The number of people in the sample -/
def n : ℕ := 5

/-- X is the number of people choosing a cultural landscape -/
noncomputable def X : ℕ → ℝ := sorry

/-- The probability of achieving a cumulative score of n points -/
noncomputable def P_n (n : ℕ) : ℝ := 3/5 - (4/15) * (-2/3)^(n-1)

/-- The expectation of X -/
noncomputable def expectation_X : ℝ := 10/3

/-- The variance of X -/
noncomputable def variance_X : ℝ := 10/9

theorem tourist_choice_statistics :
  expectation_X = 10/3 ∧
  variance_X = 10/9 ∧
  ∀ n : ℕ, P_n n = 3/5 - (4/15) * (-2/3)^(n-1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tourist_choice_statistics_l466_46693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cubes_in_box_l466_46633

theorem max_cubes_in_box :
  let box_length : ℚ := 8
  let box_width : ℚ := 9
  let box_height : ℚ := 12
  let cube_volume : ℚ := 27
  let box_volume : ℚ := box_length * box_width * box_height
  let max_cubes : ℕ := (box_volume / cube_volume).floor.toNat
  max_cubes = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cubes_in_box_l466_46633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_n_weekly_pay_is_250_l466_46697

/-- The weekly pay of employee N, given the total pay and the ratio of M's pay to N's pay -/
noncomputable def weekly_pay_n (total_pay : ℝ) (m_to_n_ratio : ℝ) : ℝ :=
  total_pay / (1 + m_to_n_ratio)

/-- Theorem stating that N's weekly pay is 250, given the problem conditions -/
theorem n_weekly_pay_is_250 :
  weekly_pay_n 550 1.2 = 250 := by
  -- Unfold the definition of weekly_pay_n
  unfold weekly_pay_n
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry

#eval (550 / 2.2 : Float)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_n_weekly_pay_is_250_l466_46697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_equation_l466_46686

/-- Given a triangle ABC with vertices A(4, -3), B(-6, 21), and C(10, 7),
    if the equation of the angle bisector of ∠B is of the form 3x + by + c = 0,
    then b + c equals a specific value. -/
theorem angle_bisector_equation (b c : ℝ) : 
  let A : ℝ × ℝ := (4, -3)
  let B : ℝ × ℝ := (-6, 21)
  let C : ℝ × ℝ := (10, 7)
  let bisector_eq (x y : ℝ) := 3 * x + b * y + c = 0
  (∀ x y, bisector_eq x y ↔ (y - B.2) / (x - B.1) = 
     ((C.2 - B.2) * (A.1 - B.1) - (C.1 - B.1) * (A.2 - B.2)) / 
     ((C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2))) →
  b + c = sorry -- The specific value would be filled here
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisector_equation_l466_46686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l466_46619

theorem coefficient_x_squared (x : ℝ) : 
  (Polynomial.coeff ((Polynomial.X + 1) ^ 5 * (Polynomial.X - 2)) 2) = -15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_squared_l466_46619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_filtrations_required_l466_46674

noncomputable def initial_impurity : ℝ := 10
noncomputable def filtration_reduction : ℝ := 1 / 3
noncomputable def target_impurity : ℝ := 0.5
noncomputable def lg2 : ℝ := 0.3010
noncomputable def lg3 : ℝ := 0.4771

theorem min_filtrations_required :
  ∃ n : ℕ, n = 8 ∧ 
  (∀ m : ℕ, m < n → initial_impurity * (1 - filtration_reduction) ^ m > target_impurity) ∧
  initial_impurity * (1 - filtration_reduction) ^ n ≤ target_impurity :=
by
  sorry

#check min_filtrations_required

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_filtrations_required_l466_46674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l466_46632

/-- Given two plane vectors a and b, where the angle between them is 60°,
    a = (2, 0), and |b| = 1, prove that |a + 2b| = 2√3 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (a.1 = 2 ∧ a.2 = 0) →  -- a = (2, 0)
  Real.sqrt (b.1^2 + b.2^2) = 1 →  -- |b| = 1
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 3 →  -- angle is 60°
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l466_46632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dee_last_standing_l466_46656

/-- Represents a student in the circle -/
inductive Student : Type
| Alice : Student
| Brad : Student
| Carl : Student
| Dee : Student
| Eve : Student
| Fay : Student
| Gus : Student
| Hal : Student

/-- Checks if a number should cause elimination -/
def isEliminationNumber (n : Nat) : Bool :=
  n % 5 = 0 || n % 7 = 0 || n.repr.any (fun c => c = '5' || c = '7')

/-- Simulates the elimination process -/
def eliminationProcess (students : List Student) : Student :=
  sorry

/-- The main theorem stating Dee is the last student remaining -/
theorem dee_last_standing :
  eliminationProcess [Student.Alice, Student.Brad, Student.Carl, Student.Dee,
                      Student.Eve, Student.Fay, Student.Gus, Student.Hal] = Student.Dee :=
by sorry

#eval isEliminationNumber 15  -- Should return true
#eval isEliminationNumber 17  -- Should return true
#eval isEliminationNumber 23  -- Should return false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dee_last_standing_l466_46656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l466_46635

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+1)
def domain_f_x_plus_1 : Set ℝ := Set.Ioo (-2) 0

-- State the theorem
theorem domain_f_2x_minus_1 (h : ∀ x, f (x + 1) ∈ domain_f_x_plus_1 ↔ x ∈ domain_f_x_plus_1) :
  ∀ x, f (2*x - 1) ∈ Set.Ioo 0 1 ↔ x ∈ Set.Ioo 0 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_f_2x_minus_1_l466_46635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_on_line_l466_46602

/-- Given a line passing through two points and a point on that line, 
    calculate the distance between that point and one of the given points. -/
theorem distance_on_line (x : ℝ) : 
  let p1 : ℝ × ℝ := (3, 6)
  let p2 : ℝ × ℝ := (-1, 8)
  let p : ℝ × ℝ := (x, 2)
  (p.2 - p1.2) / (p.1 - p1.1) = (p2.2 - p1.2) / (p2.1 - p1.1) →
  Real.sqrt ((p.1 - p1.1)^2 + (p.2 - p1.2)^2) = 2 * Real.sqrt 13 := by
  intro h
  sorry

#check distance_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_on_line_l466_46602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_B_l466_46606

theorem triangle_sin_B (A B C : ℝ) (a b c : ℝ) : 
  0 < B → B < π →
  0 < A → A < π →
  0 < C → C < π →
  A + B + C = π →
  a + c = 2 * b →
  A - C = π / 3 →
  Real.sin B = Real.sqrt 39 / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_B_l466_46606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_tiles_required_l466_46672

-- Define the room dimensions
def room_length : ℝ := 6.72
def room_width : ℝ := 4.32

-- Define the tile size
def tile_side : ℝ := 0.3

-- Calculate the room area
def room_area : ℝ := room_length * room_width

-- Calculate the tile area
def tile_area : ℝ := tile_side * tile_side

-- Calculate the number of tiles needed
noncomputable def tiles_needed : ℕ := Nat.ceil (room_area / tile_area)

-- Theorem statement
theorem minimum_tiles_required :
  tiles_needed = 323 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_tiles_required_l466_46672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composed_eq_four_has_three_solutions_l466_46613

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -x + 3 else 2*x - 5

-- Define the composite function f(f(x))
noncomputable def f_composed (x : ℝ) : ℝ := f (f x)

-- Theorem statement
theorem f_composed_eq_four_has_three_solutions :
  ∃ (a b c : ℝ), (∀ x : ℝ, f_composed x = 4 ↔ x = a ∨ x = b ∨ x = c) ∧ 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) := by
  sorry

-- Helper lemmas
lemma f_composed_eq_four_solutions (x : ℝ) :
  f_composed x = 4 ↔ x = 2 ∨ x = -3/2 ∨ x = 19/4 := by
  sorry

lemma solutions_distinct : 2 ≠ -3/2 ∧ -3/2 ≠ 19/4 ∧ 2 ≠ 19/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composed_eq_four_has_three_solutions_l466_46613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_objects_to_people_l466_46641

/-- The number of ways to distribute r identical objects to n people,
    where each person receives at least one object. -/
def number_of_ways_to_distribute (r n : ℕ) : ℕ :=
  sorry

theorem distribute_objects_to_people (r n : ℕ) (h : n ≤ r) :
  number_of_ways_to_distribute r n = Nat.choose (r - 1) (n - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_objects_to_people_l466_46641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_range_l466_46644

/-- The range of real numbers m for which the point (m, 2) can always be used to draw 
    two tangent lines to the circle (x+1)^2+(y-2)^2=4 -/
theorem tangent_point_range : 
  {m : ℝ | ∀ (x y : ℝ), (x + 1)^2 + (y - 2)^2 = 4 → 
    ∃ (t : ℝ), (m - (x + 1))^2 + (2 - (y - 2))^2 > 0 ∧ 
    (m - (x + 1)) * (x + 1) + (2 - (y - 2)) * (y - 2) = 0} = 
  Set.Ioi (-3 : ℝ) ∪ Set.Iio 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_point_range_l466_46644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_for_festival_l466_46660

/-- The amount of chicken needed for a given number of sandwiches -/
noncomputable def chicken_needed (sandwiches : ℕ) : ℚ :=
  (4 * sandwiches) / 10

/-- Conversion from kilograms to grams -/
def kg_to_grams (kg : ℚ) : ℚ :=
  kg * 1000

theorem chicken_for_festival : kg_to_grams (chicken_needed 35) = 14000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_for_festival_l466_46660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_crate_theorem_l466_46698

theorem apple_crate_theorem :
  ∀ (crate_contents : Fin 150 → ℕ),
    (∀ i, 100 ≤ crate_contents i ∧ crate_contents i ≤ 130) →
    ∃ n : ℕ, ∃ S : Finset (Fin 150),
      S.card ≥ 5 ∧ (∀ i j, i ∈ S → j ∈ S → crate_contents i = crate_contents j) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_crate_theorem_l466_46698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_problem_l466_46645

noncomputable def arithmetic_mean (s : Finset ℝ) : ℝ := (s.sum id) / s.card

theorem arithmetic_mean_problem (y : ℝ) : 
  let s : Finset ℝ := {8, 15, 24, 9, 12, y}
  arithmetic_mean s = 12 → y = 4 := by
  intro h
  -- The proof steps would go here
  sorry

#check arithmetic_mean_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_problem_l466_46645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_dot_product_l466_46647

theorem tangent_line_dot_product (x₁ : Real) :
  x₁ ∈ Set.Icc 0 π →
  let y₁ := Real.sin x₁
  let slope := 2 / π
  let xB := x₁ - (π / 2) * y₁
  (x₁ - xB) * (x₁ - xB) = (π^2 - 4) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_dot_product_l466_46647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_a_when_p_and_q_range_a_when_p_or_q_not_and_l466_46608

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - a*x + 4)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^a

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, (x^2 - a*x + 4) > 0
def q (a : ℝ) : Prop := StrictMono (g a)

-- Theorem 1
theorem range_a_when_p_and_q (a : ℝ) : p a ∧ q a → a ∈ Set.Ioo 0 4 := by
  sorry

-- Theorem 2
theorem range_a_when_p_or_q_not_and (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Ioi 4 ∪ Set.Ioc (-4) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_a_when_p_and_q_range_a_when_p_or_q_not_and_l466_46608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_rent_l466_46664

/-- Profit function for Dongfang Lodge -/
noncomputable def profit (x : ℝ) : ℝ := x * (100 - 10 * (x - 10) / 2)

/-- The rent that maximizes profit is either 14 or 16 yuan -/
theorem optimal_rent :
  ∃ (x : ℝ), (x = 14 ∨ x = 16) ∧
  (∀ y : ℝ, 10 ≤ y ∧ y < 30 → profit x ≥ profit y) := by
  sorry

#check optimal_rent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_rent_l466_46664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_on_line_implies_a_eq_neg_one_z_minus_one_abs_lower_bound_l466_46695

-- Define the complex number z as a function of real a
def z (a : ℝ) : ℂ := (1 + a * Complex.I) * (1 + Complex.I) + 2 + 4 * Complex.I

-- Part 1: Prove that if z lies on the line x-y=0, then a = -1
theorem z_on_line_implies_a_eq_neg_one :
  (∃ (a : ℝ), (z a).re = (z a).im) → 
  (∃ (a : ℝ), (z a).re = (z a).im ∧ a = -1) :=
sorry

-- Part 2: Prove that |z-1| ≥ 7√2/2 for all real a
theorem z_minus_one_abs_lower_bound (a : ℝ) :
  Complex.abs (z a - 1) ≥ 7 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_on_line_implies_a_eq_neg_one_z_minus_one_abs_lower_bound_l466_46695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radicals_combination_l466_46631

theorem quadratic_radicals_combination (m : ℝ) : 
  (∃ k : ℝ, k * Real.sqrt (2024 - 2023*m) = Real.sqrt (2023 - 2024*m)) → m = -1 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radicals_combination_l466_46631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_expression_l466_46676

theorem greatest_prime_factor_of_expression :
  (Nat.factors (5^8 + 10^7)).maximum? = some 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_expression_l466_46676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l466_46694

open Real

/-- The function f(x) as defined in the problem -/
noncomputable def f (ω : ℝ) (a : ℝ) (x : ℝ) : ℝ := 
  Real.sqrt 3 * (cos (ω * x))^2 + sin (ω * x) * cos (ω * x) + a

/-- The theorem statement -/
theorem function_properties (ω : ℝ) (a : ℝ) 
  (h_ω : ω > 0)
  (h_max : ∀ x ∈ Set.Ioo 0 (π/6), f ω a (π/6) ≥ f ω a x) :
  (∃ T > 0, ∀ x, f ω a (x + T) = f ω a x ∧ ∀ S ∈ Set.Ioo 0 T, ¬(∀ x, f ω a (x + S) = f ω a x)) ∧
  (∀ x ∈ Set.Icc (-π/3) (5*π/6), f ω a x ≥ Real.sqrt 3 → a = (Real.sqrt 3 + 1) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l466_46694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_pi_thirds_l466_46609

theorem cos_five_pi_thirds : Real.cos (5 * π / 3) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_five_pi_thirds_l466_46609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l466_46640

theorem triangle_properties (α : ℝ) 
  (h : Real.sin α + Real.cos α = 1/5) : 
  (Real.sin α - Real.cos α = -7/5) ∧ 
  (∃ (A B C : ℝ), 
    α ∈ Set.Ioo 0 π → 
    α ∈ Set.Ioo (π/2) π) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l466_46640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_with_discount_calculation_l466_46681

/-- Calculates the profit percentage with a discount, given the original profit percentage and discount rate. -/
noncomputable def profit_with_discount (original_profit_percent : ℝ) (discount_percent : ℝ) : ℝ :=
  let selling_price := 100 + original_profit_percent
  let discounted_price := selling_price * (1 - discount_percent / 100)
  (discounted_price - 100) -- profit amount
  
/-- Theorem stating that given a 5% discount and a 34% profit without discount, 
    the profit percentage with the discount is approximately 27.3%. -/
theorem profit_with_discount_calculation :
  let original_profit := 34
  let discount := 5
  abs (profit_with_discount original_profit discount - 27.3) < 0.01 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_with_discount_calculation_l466_46681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_2013th_term_sequence_general_formula_l466_46670

def Δ (a : ℕ+ → ℚ) : ℕ+ → ℚ := λ n => a (n + 1) - a n

def Δ_k (a : ℕ+ → ℚ) : ℕ → ℕ+ → ℚ
  | 0, n => a n
  | k + 1, n => Δ (Δ_k a k) n

theorem arithmetic_sequence_2013th_term 
  (a : ℕ+ → ℚ) 
  (h1 : ∀ n : ℕ+, Δ a n = 2) 
  (h2 : a 1 = 1) : 
  a 2013 = 4015 := by sorry

theorem sequence_general_formula 
  (a : ℕ+ → ℚ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n : ℕ+, Δ_k a 2 n - Δ a (n + 1) + a n = -(2 ^ (n : ℕ))) : 
  ∀ n : ℕ+, a n = n * 2^((n : ℕ) - 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_2013th_term_sequence_general_formula_l466_46670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_66kmph_l466_46690

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmph_to_ms (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

/-- Theorem stating that 66 kmph is approximately equal to 18.33 m/s -/
theorem speed_conversion_66kmph :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |kmph_to_ms 66 - 18.33| < ε := by
  -- We'll use 0.005 as our ε
  use 0.005
  constructor
  · -- Prove ε > 0
    norm_num
  constructor
  · -- Prove ε < 0.01
    norm_num
  · -- Prove |kmph_to_ms 66 - 18.33| < ε
    unfold kmph_to_ms
    norm_num
    -- The following line would complete the proof, but we'll use sorry for now
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_conversion_66kmph_l466_46690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_babysitter_rate_is_12_l466_46659

/-- Represents the babysitting scenario with given conditions -/
structure BabysittingScenario where
  current_rate : ℚ
  scream_charge : ℚ
  duration : ℚ
  scream_count : ℕ
  cost_difference : ℚ

/-- Calculates the hourly rate of the new babysitter -/
def new_babysitter_rate (scenario : BabysittingScenario) : ℚ :=
  (scenario.current_rate * scenario.duration - scenario.cost_difference - 
   scenario.scream_charge * scenario.scream_count) / scenario.duration

/-- Theorem stating that the new babysitter's rate is $12 per hour -/
theorem new_babysitter_rate_is_12 (scenario : BabysittingScenario) 
  (h1 : scenario.current_rate = 16)
  (h2 : scenario.scream_charge = 3)
  (h3 : scenario.duration = 6)
  (h4 : scenario.scream_count = 2)
  (h5 : scenario.cost_difference = 18) :
  new_babysitter_rate scenario = 12 := by
  sorry

def main : IO Unit := do
  let result := new_babysitter_rate {
    current_rate := 16,
    scream_charge := 3,
    duration := 6,
    scream_count := 2,
    cost_difference := 18
  }
  IO.println s!"The new babysitter's rate is: {result}"

#eval main

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_babysitter_rate_is_12_l466_46659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l466_46630

noncomputable def curve_C (t : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.cos (2 * t), 2 * Real.sin t)

def line_l_polar (ρ θ m : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi / 3) + m = 0

def line_l_cartesian (x y m : ℝ) : Prop :=
  Real.sqrt 3 * x + y + 2 * m = 0

theorem curve_line_intersection :
  ∀ m : ℝ,
  (∃ t : ℝ, line_l_cartesian (curve_C t).1 (curve_C t).2 m) ↔
  -19/12 ≤ m ∧ m ≤ 5/2 := by
  sorry

#check curve_line_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_line_intersection_l466_46630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_quadratic_l466_46625

/-- Given a quadratic function and a line, find the new quadratic function after transformation --/
theorem transformed_quadratic (f g h : ℝ → ℝ) (a : ℝ) :
  (f = λ x ↦ 3 * x^2 - 6 * x + 5) →
  (h = λ x ↦ -x - 2) →
  (∃ k, g = λ x ↦ -3 * (x - k)^2 + (f 1)) →
  (∃ x, g x = h x ∧ g x = -4) →
  (g = λ x ↦ -3 * x^2 + 6 * x - 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_quadratic_l466_46625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandths_place_of_seven_thirty_seconds_l466_46663

theorem thousandths_place_of_seven_thirty_seconds (n : ℕ) : 
  (7 : ℚ) / 32 = 0.21875 → n = 8 → 
  (Int.floor ((7 : ℚ) / 32 * 1000) % 10 : ℤ) = n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_thousandths_place_of_seven_thirty_seconds_l466_46663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l466_46685

/-- The speed of two trains crossing each other -/
theorem train_speed_problem (train_length : ℝ) (crossing_time : ℝ) : 
  train_length = 200 →
  crossing_time = 6 →
  (2 * train_length / crossing_time) / 2 * 3.6 = 120 := by
  intro h1 h2
  -- Calculation steps
  have relative_speed := 2 * train_length / crossing_time
  have train_speed := relative_speed / 2
  have train_speed_kmh := train_speed * 3.6
  -- Proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_problem_l466_46685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l466_46639

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + (1/2) * t, (Real.sqrt 3 / 2) * t)

noncomputable def curve_C (θ : ℝ) : ℝ := 8 * Real.cos θ / (Real.sin θ)^2

noncomputable def curve_C_rect (x : ℝ) : ℝ := Real.sqrt (8 * x)

theorem intersection_chord_length :
  let A := (6, 4 * Real.sqrt 3)
  let B := (2/3, -4 * Real.sqrt 3 / 3)
  (∃ t₁ t₂, line_l t₁ = A ∧ line_l t₂ = B) →
  (curve_C_rect 6 = 4 * Real.sqrt 3 ∧ curve_C_rect (2/3) = 4 * Real.sqrt 3 / 3) →
  Real.sqrt ((6 - 2/3)^2 + (4 * Real.sqrt 3 + 4 * Real.sqrt 3 / 3)^2) = 32/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_chord_length_l466_46639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_defined_g_l466_46658

-- Define g₁(x) as a geometric series
noncomputable def g₁ (x : ℝ) : ℝ := (1 / 3) * (1 / (1 - x))

-- Define g_n(x) recursively
noncomputable def g : ℕ → ℝ → ℝ
| 0, x => x
| 1, x => g₁ x
| (n+2), x => g₁ (g (n+1) x)

-- Define the property of g_n(x) being defined for some real x
def is_defined (n : ℕ) : Prop :=
  ∃ x : ℝ, ∀ k : ℕ, k ≤ n → -1 < g k x ∧ g k x < 1

-- State the theorem
theorem largest_defined_g :
  (∀ n : ℕ, n ≤ 5 → is_defined n) ∧
  ¬(is_defined 6) := by
  sorry

#check largest_defined_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_defined_g_l466_46658
