import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l832_83224

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 4 * y^2 + 24 * y + 35 = 0

/-- The radius of the circle -/
noncomputable def circle_radius : ℝ := Real.sqrt 5 / 2

/-- Theorem: The radius of the circle with the given equation is sqrt(5)/2 -/
theorem circle_radius_proof :
  ∃ (h k : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = circle_radius^2 := by
  sorry

#check circle_radius_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_proof_l832_83224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lines_theorem_l832_83251

/-- Represents the minimum number of lines in a program that computes f_n -/
def C (n : ℕ) : ℕ := sorry

/-- The theorem states that for n ≥ 4, the minimum number of lines to compute f_n
    is at least 2 more than the minimum number of lines to compute f_n-2 -/
theorem min_lines_theorem (n : ℕ) (h : n ≥ 4) : C n ≥ C (n - 2) + 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_lines_theorem_l832_83251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reach_shore_l832_83202

/-- Represents the probability of a bridge being intact --/
noncomputable def p : ℝ := 0.5

/-- Represents the probability of a bridge being destroyed --/
noncomputable def q : ℝ := 1 - p

/-- The sum of the infinite geometric series representing the probability of reaching the shore --/
noncomputable def probabilitySum : ℝ := q / (1 - p * q)

/-- Theorem stating that the probability of reaching the shore is 2/3 --/
theorem probability_reach_shore :
  probabilitySum = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_reach_shore_l832_83202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l832_83219

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi / 4 + x) ^ 2 - Real.sqrt 3 * Real.cos (2 * x)

theorem f_range :
  ∀ x : ℝ, Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2 →
  2 ≤ f x ∧ f x ≤ 3 ∧
  (∃ x₁ : ℝ, Real.pi / 4 ≤ x₁ ∧ x₁ ≤ Real.pi / 2 ∧ f x₁ = 2) ∧
  (∃ x₂ : ℝ, Real.pi / 4 ≤ x₂ ∧ x₂ ≤ Real.pi / 2 ∧ f x₂ = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l832_83219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_achieves_target_profit_l832_83208

/-- Represents the price reduction of greeting cards -/
noncomputable def price_reduction : ℝ := 0.1

/-- Initial number of cards sold per day -/
noncomputable def initial_sales : ℝ := 500

/-- Initial profit per card in yuan -/
noncomputable def initial_profit_per_card : ℝ := 0.3

/-- Increase in sales for every 0.05 yuan price reduction -/
noncomputable def sales_increase_per_0_05 : ℝ := 200

/-- Target average daily profit in yuan -/
noncomputable def target_daily_profit : ℝ := 180

/-- Calculates the number of additional cards sold based on price reduction -/
noncomputable def additional_sales (x : ℝ) : ℝ :=
  (x / 0.05) * sales_increase_per_0_05

/-- Calculates the total number of cards sold after price reduction -/
noncomputable def total_sales (x : ℝ) : ℝ :=
  initial_sales + additional_sales x

/-- Calculates the profit per card after price reduction -/
noncomputable def profit_per_card (x : ℝ) : ℝ :=
  initial_profit_per_card - x

/-- Calculates the total daily profit after price reduction -/
noncomputable def daily_profit (x : ℝ) : ℝ :=
  profit_per_card x * total_sales x

/-- Theorem stating that the price reduction of 0.1 yuan results in the target daily profit -/
theorem price_reduction_achieves_target_profit :
  daily_profit price_reduction = target_daily_profit :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_achieves_target_profit_l832_83208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sequence_exists_l832_83215

/-- A move on a sequence of real numbers -/
noncomputable def move (seq : List ℝ) (i j : Nat) : List ℝ :=
  if i < seq.length ∧ j < seq.length ∧ i ≠ j then
    let mean := (seq[i]! + seq[j]!) / 2
    seq.set i mean |>.set j mean
  else
    seq

/-- A sequence of moves -/
noncomputable def movesSequence (seq : List ℝ) (moves : List (Nat × Nat)) : List ℝ :=
  moves.foldl (fun s (i, j) => move s i j) seq

theorem constant_sequence_exists (initialSeq : List ℝ) :
  initialSeq.length = 2015 → initialSeq.Nodup →
  ∀ i j, i < 2015 → j < 2015 → i ≠ j →
  ∃ moves : List (Nat × Nat), 
    (movesSequence (move initialSeq i j) moves).all (· = (movesSequence (move initialSeq i j) moves).head!) := by
  sorry

#check constant_sequence_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sequence_exists_l832_83215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tickets_can_be_placed_in_50_boxes_l832_83298

/-- Represents a ticket number as a list of digits -/
def Ticket := List Nat

/-- Represents a box number as a pair of digits -/
def Box := Nat × Nat

/-- Function to check if a ticket can be placed in a box -/
def canPlace (t : Ticket) (b : Box) : Prop :=
  ∃ (i : Nat), i < t.length ∧ b.1 * 10 + b.2 = (t.take i ++ t.drop (i+1)).foldl (fun acc d => acc * 10 + d) 0

/-- The set of all valid tickets -/
def allTickets : Set Ticket :=
  {t | t.length = 6 ∧ 1 ≤ t.foldl (fun acc d => acc * 10 + d) 0 ∧ t.foldl (fun acc d => acc * 10 + d) 0 ≤ 999999}

/-- The set of all valid boxes -/
def allBoxes : Set Box :=
  {b | 0 ≤ b.1 ∧ b.1 ≤ 9 ∧ 0 ≤ b.2 ∧ b.2 ≤ 9}

/-- The theorem to be proved -/
theorem tickets_can_be_placed_in_50_boxes :
  ∃ (boxes : Finset Box),
    (↑boxes : Set Box) ⊆ allBoxes ∧
    boxes.card = 50 ∧
    ∀ t ∈ allTickets, ∃ b ∈ boxes, canPlace t b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tickets_can_be_placed_in_50_boxes_l832_83298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_island_puzzle_l832_83270

-- Define the possible types of residents
inductive ResidentType
  | Knight
  | Liar
  | Ordinary

-- Define a resident
structure Resident where
  name : String
  type : ResidentType

-- Define the statement function
def makes_statement (r : Resident) (s : Prop) : Prop :=
  match r.type with
  | ResidentType.Knight => s
  | ResidentType.Liar => ¬s
  | ResidentType.Ordinary => True

-- Define the theorem
theorem island_puzzle (A B : Resident) :
  (makes_statement A (B.type = ResidentType.Knight)) →
  (makes_statement B (A.type = ResidentType.Liar)) →
  ((A.type = ResidentType.Ordinary ∧ B.type ≠ ResidentType.Knight) ∨
   (B.type = ResidentType.Ordinary ∧ A.type ≠ ResidentType.Liar)) :=
by
  intros h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_island_puzzle_l832_83270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_lambda_l832_83233

-- Define the space
variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define the points
variable (A B C D O : V)

-- Define the conditions
variable (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
variable (h2 : ∃ (a b c : ℝ), a • A + b • B + c • C = D ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0))
variable (h3 : ∀ (x y z : V), (x = A ∧ y = B ∧ z = C) ∨ (x = A ∧ y = B ∧ z = D) ∨ 
                              (x = A ∧ y = C ∧ z = D) ∨ (x = B ∧ y = C ∧ z = D) → 
                              ¬ (∃ (t : ℝ), y - x = t • (z - x)))
variable (h4 : ¬ (∃ (a b c d : ℝ), a • A + b • B + c • C + d • D = O ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0)))
variable (h5 : ∃ (l : ℝ), D - O = 3 • (A - O) + 2 • (B - O) + l • (C - O))

-- Theorem statement
theorem vector_equation_lambda :
  ∃ (l : ℝ), D - O = 3 • (A - O) + 2 • (B - O) + l • (C - O) ∧ l = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_lambda_l832_83233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheelas_finances_calculation_l832_83281

/-- Represents Sheela's financial situation -/
structure SheelasFinances where
  savings_deposit : ℚ
  savings_percent : ℚ
  investment_percent : ℚ
  living_expenses_percent : ℚ
  income_tax_percent : ℚ
  health_insurance_percent : ℚ

/-- Calculates Sheela's monthly income and total net savings -/
def calculate_finances (f : SheelasFinances) : ℚ × ℚ :=
  let monthly_income := f.savings_deposit / f.savings_percent
  let investment_deposit := f.investment_percent * monthly_income
  let total_savings := f.savings_deposit + investment_deposit
  (monthly_income, total_savings)

/-- Theorem stating the correct calculation of Sheela's finances -/
theorem sheelas_finances_calculation :
  let f : SheelasFinances := {
    savings_deposit := 3800,
    savings_percent := 32/100,
    investment_percent := 15/100,
    living_expenses_percent := 28/100,
    income_tax_percent := 14/100,
    health_insurance_percent := 5/100
  }
  let (monthly_income, total_savings) := calculate_finances f
  monthly_income = 11875 ∧ total_savings = 5581.25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheelas_finances_calculation_l832_83281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estate_division_theorem_l832_83296

noncomputable def beneficiary_condition (x y : ℝ) : Prop :=
  |x - y| ≥ 0.2 * min x y

noncomputable def valid_beneficiaries (a b c d : ℝ) : Prop :=
  beneficiary_condition a b ∧
  beneficiary_condition a c ∧
  beneficiary_condition a d ∧
  beneficiary_condition b c ∧
  beneficiary_condition b d ∧
  beneficiary_condition c d

noncomputable def smallest_range (a b c d : ℝ) : ℝ :=
  max a (max b (max c d)) - min a (min b (min c d))

theorem estate_division_theorem :
  ∀ (b c d : ℝ),
  valid_beneficiaries 40000 b c d →
  (∀ (b' c' d' : ℝ),
    valid_beneficiaries 40000 b' c' d' →
    smallest_range 40000 b c d ≤ smallest_range 40000 b' c' d') →
  smallest_range 40000 b c d = 19520 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_estate_division_theorem_l832_83296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_szekeres_l832_83286

theorem erdos_szekeres (m n : ℕ) (s : Fin (m * n + 1) → ℝ) :
  (∃ (f : Fin (n + 1) → Fin (m * n + 1)), StrictMono f ∧ StrictMono (s ∘ f)) ∨
  (∃ (g : Fin (m + 1) → Fin (m * n + 1)), StrictMono g ∧ StrictMono (fun i => -(s (g i)))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_erdos_szekeres_l832_83286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l832_83276

theorem beta_value (α β : ℝ) 
  (h1 : Real.sin α = (4 / 7) * Real.sqrt 3)
  (h2 : Real.cos (α + β) = -(11 / 14))
  (h3 : 0 < α ∧ α < π/2)
  (h4 : 0 < β ∧ β < π/2) :
  β = π/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_beta_value_l832_83276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l832_83238

def T : Finset Nat := Finset.range 50

def valid_subset (S : Finset Nat) : Prop :=
  S ⊆ T ∧ ∀ x y, x ∈ S → y ∈ S → (x + y) % 7 ≠ 0

theorem max_subset_size :
  ∃ (S : Finset Nat), valid_subset S ∧ S.card = 23 ∧
    ∀ (S' : Finset Nat), valid_subset S' → S'.card ≤ 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l832_83238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_triangle_existence_l832_83271

/-- Represents a vertex of a cube -/
inductive Vertex : Type
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8

/-- Represents the movement of flies from initial to final positions -/
def fly_movement : Vertex → Vertex :=
  sorry

/-- Checks if three vertices form an equilateral triangle -/
def is_equilateral_triangle (a b c : Vertex) : Prop :=
  sorry

/-- The theorem states that there exist three vertices forming equilateral triangles
    before and after the movement -/
theorem fly_triangle_existence :
  ∃ (a b c : Vertex),
    is_equilateral_triangle a b c ∧
    is_equilateral_triangle (fly_movement a) (fly_movement b) (fly_movement c) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fly_triangle_existence_l832_83271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l832_83242

/-- The distance between two points in 3D space -/
noncomputable def distance3D (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

/-- Theorem: The distance between (0, 6, 2) and (8, 0, 6) is √116 -/
theorem distance_between_specific_points :
  distance3D (0, 6, 2) (8, 0, 6) = Real.sqrt 116 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_points_l832_83242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_map_scale_calculation_l832_83240

/-- Represents the scale of a map in inches per mile -/
noncomputable def map_scale (map_distance : ℚ) (actual_distance : ℚ) : ℚ :=
  map_distance / actual_distance

/-- Calculates the actual distance traveled given time and speed -/
def actual_distance (time : ℚ) (speed : ℚ) : ℚ :=
  time * speed

theorem map_scale_calculation (map_distance : ℚ) (travel_time : ℚ) (average_speed : ℚ)
    (h1 : map_distance = 5)
    (h2 : travel_time = 3/2)
    (h3 : average_speed = 60) :
    map_scale map_distance (actual_distance travel_time average_speed) = 1/18 := by
  sorry

#check map_scale_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_map_scale_calculation_l832_83240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_equality_l832_83299

theorem power_of_three_equality (x : ℝ) : (3 : ℝ)^x = 81 → (3 : ℝ)^(x+2) = 729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_three_equality_l832_83299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l832_83279

noncomputable def f (x : ℝ) : ℝ := 1/x - Real.sqrt x

theorem tangent_line_equation (x₀ y₀ : ℝ) (h₁ : x₀ = 4) (h₂ : y₀ = -7/4) (h₃ : y₀ = f x₀) :
  ∃ (a b c : ℝ), a*x₀ + b*y₀ + c = 0 ∧ 
  ∀ (x y : ℝ), y = f x → (y - y₀) = (deriv f x₀) * (x - x₀) → a*x + b*y + c = 0 := by
  sorry

#check tangent_line_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l832_83279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_solution_set_l832_83229

theorem sine_inequality_solution_set (x : ℝ) :
  x ∈ Set.Icc 0 (2 * Real.pi) →
  (Real.sin x < -Real.sqrt 3 / 2 ↔ x ∈ Set.Ioo (4 * Real.pi / 3) (5 * Real.pi / 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_inequality_solution_set_l832_83229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_climb_distance_l832_83241

/-- The distance a monkey hops each hour to climb a tree -/
noncomputable def monkey_hop (tree_height : ℝ) (total_hours : ℕ) (slip_distance : ℝ) : ℝ :=
  (tree_height + slip_distance * (total_hours - 1)) / total_hours

/-- Theorem: A monkey climbing a 17 ft tree in 15 hours, slipping 2 ft each hour except the last, hops 3 ft per hour -/
theorem monkey_climb_distance :
  monkey_hop 17 15 2 = 3 := by
  -- Unfold the definition of monkey_hop
  unfold monkey_hop
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monkey_climb_distance_l832_83241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_divisions_necessary_and_sufficient_l832_83205

/-- Represents a sequence of natural numbers written on a card -/
def CardSequence := List Nat

/-- The number of cards in the deck -/
def deckSize : Nat := 54

/-- A function that represents one division of the deck -/
noncomputable def divide : List CardSequence → List CardSequence :=
  sorry

/-- Predicate to check if all card sequences are unique -/
def allUnique (sequences : List CardSequence) : Prop :=
  ∀ s₁ s₂, s₁ ∈ sequences → s₂ ∈ sequences → s₁ ≠ s₂ → s₁.toFinset ≠ s₂.toFinset

/-- The main theorem stating that 3 divisions are necessary and sufficient -/
theorem three_divisions_necessary_and_sufficient :
  ∃ (initial : List CardSequence),
    initial.length = deckSize ∧
    (∀ seq, seq ∈ initial → seq.length = 1) ∧
    ¬(allUnique (divide initial)) ∧
    ¬(allUnique (divide (divide initial))) ∧
    (allUnique (divide (divide (divide initial)))) ∧
    (∀ (f : List CardSequence → List CardSequence),
      (allUnique (f (f (f initial)))) →
      (allUnique (divide (divide (divide initial))))) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_divisions_necessary_and_sufficient_l832_83205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_not_in_P_l832_83278

-- Define the set P
noncomputable def P : Set ℝ := {x | 0 ≤ x ∧ x ≤ Real.sqrt 2}

-- Define m
noncomputable def m : ℝ := Real.sqrt 3

-- Theorem statement
theorem m_not_in_P : m ∉ P := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_not_in_P_l832_83278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_n_l832_83246

noncomputable def f (x : ℝ) : ℝ := Real.exp x + (1/2) * x^2 - x

theorem range_of_n (n : ℝ) :
  (∃ m : ℝ, f m ≤ 2 * n^2 - n) → n ≤ -1/2 ∨ n ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_n_l832_83246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_transformation_possible_l832_83237

/-- A sequence of 10 elements, each being either +1 or -1 -/
def Sequence := Fin 10 → Int

/-- Predicate to check if a sequence contains only +1 and -1 -/
def valid_sequence (s : Sequence) : Prop :=
  ∀ i, s i = 1 ∨ s i = -1

/-- Function to flip the sign of an element -/
def flip_sign (x : Int) : Int :=
  -x

/-- Function to flip signs of 5 elements in a sequence -/
def flip_five (s : Sequence) (i₁ i₂ i₃ i₄ i₅ : Fin 10) : Sequence :=
  fun i => if i = i₁ ∨ i = i₂ ∨ i = i₃ ∨ i = i₄ ∨ i = i₅ then flip_sign (s i) else s i

/-- Predicate to check if two sequences differ in exactly one position -/
def differ_by_one (s₁ s₂ : Sequence) : Prop :=
  ∃ (i : Fin 10), (∀ j : Fin 10, j ≠ i → s₁ j = s₂ j) ∧ s₁ i ≠ s₂ i

/-- The main theorem to be proved -/
theorem sequence_transformation_possible (s : Sequence) (h : valid_sequence s) :
  ∃ (n : ℕ) (i₁ i₂ i₃ i₄ i₅ : ℕ → Fin 10),
    differ_by_one s (Nat.iterate (λ seq => flip_five seq (i₁ n) (i₂ n) (i₃ n) (i₄ n) (i₅ n)) n s) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_transformation_possible_l832_83237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l832_83210

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x + 4 * Real.cos x + 1

/-- The theorem statement -/
theorem problem_statement (p q r : ℝ) 
  (h : ∀ x : ℝ, p * f x + q * f (x + r) = 2018) : 
  p * Real.cos r + q = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l832_83210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l832_83264

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 8

-- Define the line of symmetry
def symmetry_line (a b x y : ℝ) : Prop := 2*a*x + b*y + 6 = 0

-- Define the point M
def point_M (a b : ℝ) : ℝ × ℝ := (a, b)

-- Define the tangent line length function
noncomputable def tangent_length (a b : ℝ) : ℝ := Real.sqrt ((a + 1)^2 + (b - 2)^2 - 8)

-- Theorem statement
theorem min_tangent_length :
  ∀ a b : ℝ,
  (∃ x y : ℝ, circle_C x y ∧ symmetry_line a b x y) →
  ∃ min_length : ℝ,
    (∀ a' b' : ℝ, tangent_length a' b' ≥ min_length) ∧
    (∃ a₀ b₀ : ℝ, tangent_length a₀ b₀ = min_length) ∧
    min_length = Real.sqrt 10 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_tangent_length_l832_83264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_power_function_l832_83220

/-- The domain of the function f(x) = x^(-3/4) is (0, +∞) -/
theorem domain_of_power_function (f : ℝ → ℝ) :
  (∀ x > 0, f x = x^(-(3/4 : ℝ))) →
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x > 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_power_function_l832_83220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_product_l832_83256

/-- Two lines intersecting at 30° where one slope is 4 times the other -/
structure IntersectingLines where
  m : ℝ  -- slope of the first line
  angle : ℝ  -- angle of intersection in radians
  h_angle : angle = π / 6  -- 30° in radians
  h_slope_relation : Real.tan angle = |3 * m / (1 + 4 * m^2)|

/-- The product of slopes of the two lines -/
def slope_product (l : IntersectingLines) : ℝ := 4 * l.m^2

/-- The maximum product of slopes for the given configuration -/
theorem max_slope_product :
  ∃ (l : IntersectingLines), ∀ (l' : IntersectingLines),
    slope_product l ≥ slope_product l' ∧
    slope_product l = (9 + 2 * Real.sqrt 33) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_product_l832_83256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l832_83221

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem angle_between_vectors (a b : V) 
  (ha : ‖a‖ = 1) (hb : ‖b‖ = 1) (hab : ‖a + (2 : ℝ) • b‖ = Real.sqrt 7) : 
  Real.arccos (inner a b) = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l832_83221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_unique_solution_l832_83258

noncomputable def f (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

theorem cubic_function_unique_solution :
  ∀ a b c d : ℝ,
  a ≠ 0 →
  (∃ x_max : ℝ, x_max = 1 ∧ IsLocalMax (f a b c d) x_max ∧ f a b c d x_max = 4) →
  (∃ x_min : ℝ, x_min = 3 ∧ IsLocalMin (f a b c d) x_min ∧ f a b c d x_min = 0) →
  f a b c d 0 = 0 →
  f a b c d = λ x ↦ x^3 - 6*x^2 + 9*x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_unique_solution_l832_83258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_group3_is_right_triangle_l832_83201

-- Define a function to check if three numbers can form a right-angled triangle
def isRightTriangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the groups of numbers
def group1 : ℕ × ℕ × ℕ := (1, 2, 3)
def group2 : ℕ × ℕ × ℕ := (2, 3, 4)
def group3 : ℕ × ℕ × ℕ := (3, 4, 5)
def group4 : ℕ × ℕ × ℕ := (4, 5, 6)

-- Theorem stating that only group3 forms a right-angled triangle
theorem only_group3_is_right_triangle :
  ¬(isRightTriangle group1.fst group1.snd.fst group1.snd.snd) ∧
  ¬(isRightTriangle group2.fst group2.snd.fst group2.snd.snd) ∧
  (isRightTriangle group3.fst group3.snd.fst group3.snd.snd) ∧
  ¬(isRightTriangle group4.fst group4.snd.fst group4.snd.snd) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_group3_is_right_triangle_l832_83201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_squared_is_three_l832_83275

/-- An ellipse with equation x^2 + 4y^2 = 4 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + 4 * p.2^2 = 4}

/-- An equilateral triangle with vertices on the ellipse -/
structure EquilateralTriangleOnEllipse where
  vertices : Fin 3 → ℝ × ℝ
  on_ellipse : ∀ i, vertices i ∈ Ellipse
  equilateral : ∀ i j, i ≠ j → dist (vertices i) (vertices j) = dist (vertices 0) (vertices 1)
  centroid_is_vertex : ∃ i, vertices i = (0, 0)

/-- Calculate the area of a triangle given its vertices -/
noncomputable def triangleArea (v : Fin 3 → ℝ × ℝ) : ℝ :=
  let a := dist (v 0) (v 1)
  let b := dist (v 1) (v 2)
  let c := dist (v 2) (v 0)
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The square of the area of the equilateral triangle on the ellipse is 3 -/
theorem area_squared_is_three (t : EquilateralTriangleOnEllipse) : 
  (triangleArea t.vertices)^2 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_squared_is_three_l832_83275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l832_83282

-- Proposition ①
def prop1 (p q : Prop) : Prop :=
  (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)

-- Proposition ②
def prop2 : Prop :=
  let p := ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0
  let not_p := ∀ x : ℝ, x^2 + 2*x + 2 > 0
  (¬p) ↔ not_p

-- Proposition ③
def prop3 (a : ℝ) : Prop :=
  (¬∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0) → (-1 ≤ a ∧ a ≤ 3)

-- Proposition ④
def prop4 : Prop :=
  let p := ∃ x : ℝ, Real.tan x = 1
  let q := ∀ x : ℝ, (x^2 - 3*x + 2 < 0) ↔ (1 < x ∧ x < 2)
  ¬(¬p ∨ ¬q)

theorem problem_solution :
  ¬(∀ p q : Prop, prop1 p q) ∧
  prop2 ∧
  (∀ a : ℝ, prop3 a) ∧
  prop4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l832_83282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_reciprocal_sum_l832_83294

theorem triangle_angle_ratio_reciprocal_sum (A B C : ℝ) (a b c : ℝ) :
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A + B + C = π ∧
  A = 4 * (π / 7) ∧ B = 2 * (π / 7) ∧ C = π / 7 →
  1/a + 1/b = 1/c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_ratio_reciprocal_sum_l832_83294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_implies_sin_sum_zero_l832_83263

theorem cos_product_implies_sin_sum_zero (α β : ℝ) :
  Real.cos α * Real.cos β = 1 → Real.sin (α + β) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_product_implies_sin_sum_zero_l832_83263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_70_equals_1_minus_2sin_10_squared_l832_83213

theorem sin_70_equals_1_minus_2sin_10_squared :
  Real.sin (70 * π / 180) = 1 - 2 * (Real.sin (10 * π / 180))^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_70_equals_1_minus_2sin_10_squared_l832_83213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_filling_time_l832_83217

/-- Represents a pipe that can fill a cistern -/
structure Pipe where
  fill_time : ℚ
  fill_rate : ℚ := 1 / fill_time

/-- Represents a cistern being filled by pipes -/
structure Cistern where
  capacity : ℚ := 1
  filled : ℚ

/-- The main theorem about filling the cistern -/
theorem cistern_filling_time 
  (p : Pipe) 
  (q : Pipe) 
  (initial_time : ℚ) 
  (c : Cistern) : 
  p.fill_time = 12 →
  q.fill_time = 15 →
  initial_time = 2 →
  c.filled = initial_time * (p.fill_rate + q.fill_rate) →
  (1 - c.filled) / q.fill_rate = 21/2 := by
  sorry

#check cistern_filling_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_filling_time_l832_83217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_functions_f₁_increasing_f₃_increasing_f₂_not_increasing_f₄_not_increasing_l832_83248

-- Definition of an increasing function
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

-- Given functions
noncomputable def f₁ (x : ℝ) : ℝ := 2 * x
noncomputable def f₂ (x : ℝ) : ℝ := -x + 1
noncomputable def f₃ (x : ℝ) : ℝ := x^2
noncomputable def f₄ (x : ℝ) : ℝ := -1 / x

-- Theorem statement
theorem increasing_functions :
  IsIncreasing f₁ ∧
  IsIncreasing (fun x => f₃ x) ∧
  ¬IsIncreasing f₂ ∧
  ¬IsIncreasing f₄ := by
  sorry

-- Separate proofs for each part of the conjunction
theorem f₁_increasing : IsIncreasing f₁ := by
  sorry

theorem f₃_increasing : IsIncreasing (fun x => f₃ x) := by
  sorry

theorem f₂_not_increasing : ¬IsIncreasing f₂ := by
  sorry

theorem f₄_not_increasing : ¬IsIncreasing f₄ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_functions_f₁_increasing_f₃_increasing_f₂_not_increasing_f₄_not_increasing_l832_83248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l832_83247

-- Define vectors a and b
noncomputable def a (x : Real) : Real × Real := (2 * (Real.cos x)^2, Real.sin x)
noncomputable def b (x : Real) : Real × Real := (1/2, Real.sqrt 3 * Real.cos x)

-- Define function f
noncomputable def f (x : Real) : Real := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Define triangle ABC
def triangle_ABC (A B : Real) (BC : Real) : Prop :=
  A + B = 7/12 * Real.pi ∧ f A = 1 ∧ BC = 2 * Real.sqrt 3

-- Theorem statement
theorem problem_solution :
  ∀ (A B : Real) (BC : Real),
  triangle_ABC A B BC →
  (∃ (T : Real), T = Real.pi ∧ ∀ (x : Real), f (x + T) = f x) ∧
  (∀ (k : Int), ∀ (x : Real), 
    Real.pi/6 + k * Real.pi ≤ x ∧ x ≤ 2*Real.pi/3 + k * Real.pi → 
    ∀ (y : Real), x < y → f y < f x) ∧
  (∃ (AC : Real), AC = 2 * Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l832_83247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extreme_value_l832_83259

-- Define the function f(x) = x - ln x
noncomputable def f (x : ℝ) : ℝ := x - Real.log x

-- State the theorem
theorem f_extreme_value :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = 1 := by
  -- We'll use x = 1 as the minimum point
  use 1
  
  constructor
  · -- Prove 1 > 0
    exact zero_lt_one

  · intro y hy
    constructor
    · sorry -- Proof that f y ≥ f 1 for all y > 0
    · -- Prove f 1 = 1
      unfold f
      simp [Real.log_one]

-- The proof is incomplete, but the structure is correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extreme_value_l832_83259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_eq_one_range_of_b_l832_83231

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1 - a) / 2 * x^2 - x

theorem min_value_when_a_eq_one :
  ∃ (min_val : ℝ), min_val = 1 - 1 / Real.exp 2 ∧ 
    ∀ x > 0, (x - Real.log x + 1) / x ≥ min_val :=
sorry

theorem range_of_b (b : ℝ) :
  (∀ x > 0, b * x + 1 ≥ f 1 x) ↔ b ≥ 1 / Real.exp 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_when_a_eq_one_range_of_b_l832_83231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_126_l832_83262

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 2  -- Add case for 0
  | 1 => 2
  | (n + 1) => 2 * a n

-- Define the sum S_n
def S (n : ℕ) : ℕ := (List.range n).map a |>.sum

-- Theorem statement
theorem sequence_sum_126 : ∃ n : ℕ, S n = 126 ∧ n = 6 := by
  -- Proof goes here
  sorry

#eval S 6  -- This will evaluate S 6 to check if it equals 126

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_126_l832_83262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_discount_percentage_l832_83206

/-- Proves that given an item with an original price of 9502.923976608186,
    after three successive discounts where the first is 20% and the second is 10%,
    if the final price is 6500, then the third discount is approximately 5.01%. -/
theorem third_discount_percentage (original_price final_price : ℝ)
  (first_discount second_discount third_discount : ℝ) :
  original_price = 9502.923976608186 →
  first_discount = 0.20 →
  second_discount = 0.10 →
  final_price = 6500 →
  final_price = original_price * (1 - first_discount) * (1 - second_discount) * (1 - third_discount) →
  abs (third_discount - 0.0501) < 0.0001 := by
  sorry

#eval (1 - 6500 / (9502.923976608186 * 0.8 * 0.9))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_discount_percentage_l832_83206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_whale_turn_back_time_l832_83236

/-- Represents the speed of a whale in km/h -/
def WhaleSpeed := ℝ

/-- Represents a time duration in hours -/
def TimeDuration := ℝ

/-- Represents a specific time of day -/
structure TimeOfDay where
  hour : ℕ
  minute : ℕ
  deriving Repr

noncomputable def TimeOfDay.toHours (t : TimeOfDay) : ℝ :=
  t.hour + t.minute / 60

noncomputable def hoursBetween (t1 t2 : TimeOfDay) : ℝ :=
  t2.toHours - t1.toHours

structure WhaleScenario where
  initialSpeed : WhaleSpeed
  fasterSpeed : WhaleSpeed
  separationTime : TimeOfDay
  meetingTime : TimeOfDay

/-- Calculates the time at which the faster whale turns back -/
noncomputable def calculateTurnBackTime (scenario : WhaleScenario) : TimeOfDay :=
  sorry

theorem whale_turn_back_time (scenario : WhaleScenario) :
  scenario.initialSpeed = (6 : ℝ) →
  scenario.fasterSpeed = (10 : ℝ) →
  scenario.separationTime = ⟨8, 15⟩ →
  scenario.meetingTime = ⟨10, 0⟩ →
  calculateTurnBackTime scenario = ⟨9, 51⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_whale_turn_back_time_l832_83236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_beta_l832_83204

theorem cosine_beta (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.cos α = 1/7 → 
  Real.sin (α + β) = 5 * Real.sqrt 3 / 14 → 
  Real.cos β = 1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_beta_l832_83204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_proof_l832_83283

/-- The height of a square-based pyramid with base edge length 10 units,
    given that its volume is equal to the volume of a cube with edge length 5 units. -/
noncomputable def pyramid_height : ℝ := 3.75

/-- Volume of a cube with edge length 5 units -/
noncomputable def cube_volume : ℝ := 5^3

/-- Volume of a square-based pyramid with base edge length 10 units and height h -/
noncomputable def pyramid_volume (h : ℝ) : ℝ := (1/3) * 10^2 * h

theorem pyramid_height_proof :
  pyramid_volume pyramid_height = cube_volume :=
by
  -- Unfold the definitions
  unfold pyramid_volume
  unfold pyramid_height
  unfold cube_volume
  
  -- Perform the calculation
  norm_num
  
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_proof_l832_83283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_has_minimum_l832_83257

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sumArithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + n * (n - 1 : ℝ) / 2 * d

theorem arithmetic_sequence_sum_has_minimum (a₁ d : ℝ) (h₁ : a₁ < 0) (h₂ : d > 0) :
  ∃ (m : ℝ), ∀ (n : ℕ), m ≤ sumArithmeticSequence a₁ d n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_has_minimum_l832_83257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l832_83223

def a : ℕ → ℚ
  | 0 => 0  -- Adding a case for 0
  | 1 => 3
  | n + 1 => (n * a n) / (3 * n)

def b (n : ℕ) : ℚ := 
  if n = 0 then 0 else (a n * (n + 1)) / (2 * n + 3)

def S (n : ℕ) : ℚ := (Finset.range n).sum (λ i => 1 / b (i + 1))

theorem sequence_inequality (n : ℕ) : 5/6 ≤ S n ∧ S n < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l832_83223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_locus_of_midpoints_l832_83297

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square ABCD with center at the origin -/
structure Square where
  side : ℝ

/-- Represents the fixed line l parallel to x-axis -/
structure Line where
  a : ℝ  -- y-intercept

/-- The geometric locus of midpoints of PQ -/
def geometricLocus (l : Line) : Set Point :=
  { p : Point | ∃ t : ℝ, p.x = t ∧ p.y = -t + l.a / 2 }

/-- P is the foot of perpendicular from D to l -/
noncomputable def footOfPerpendicular (s : Square) (l : Line) : Point :=
  { x := -s.side / 2, y := l.a }

/-- Q is the midpoint of AB -/
noncomputable def midpointAB (s : Square) : Point :=
  { x := s.side / 2, y := s.side / 2 }

/-- Midpoint of segment PQ -/
noncomputable def midpointPQ (s : Square) (l : Line) : Point :=
  { x := (footOfPerpendicular s l).x / 2 + (midpointAB s).x / 2,
    y := (footOfPerpendicular s l).y / 2 + (midpointAB s).y / 2 }

theorem geometric_locus_of_midpoints (s : Square) (l : Line) :
  ∀ θ : ℝ, midpointPQ s l ∈ geometricLocus l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_locus_of_midpoints_l832_83297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cards_for_jasmine_l832_83290

/-- The maximum number of whole cards that can be purchased given a budget and cost per card -/
def max_cards (budget : ℚ) (cost_per_card : ℚ) : ℕ :=
  (budget / cost_per_card).floor.toNat

/-- Proof that given $8.75 and $0.95 per card, the maximum number of cards is 9 -/
theorem max_cards_for_jasmine :
  max_cards (875 / 100) (95 / 100) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cards_for_jasmine_l832_83290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_B_l832_83277

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_sin_B (t : Triangle) 
  (h1 : t.a + t.c = 2 * t.b) 
  (h2 : t.A - t.C = π / 3) 
  (h3 : t.A + t.B + t.C = π) -- Triangle angle sum
  (h4 : t.a > 0 ∧ t.b > 0 ∧ t.c > 0) -- Positive side lengths
  (h5 : t.a / (Real.sin t.A) = t.b / (Real.sin t.B)) -- Law of sines
  (h6 : t.b / (Real.sin t.B) = t.c / (Real.sin t.C)) -- Law of sines
  : Real.sin t.B = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_B_l832_83277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l832_83295

/-- Given two parallel lines in the form ax + by + c = 0, 
    calculate the distance between them -/
noncomputable def distance_between_parallel_lines (a1 b1 c1 a2 b2 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / Real.sqrt (a1^2 + b1^2)

/-- Theorem: The distance between two specific parallel lines -/
theorem distance_between_specific_lines (n : ℝ) :
  (1 = 2/n) →  -- Condition for parallel lines
  distance_between_parallel_lines 1 1 (-1) 2 n 5 = (7 * Real.sqrt 2) / 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l832_83295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_operation_satisfies_equation_l832_83265

theorem unique_operation_satisfies_equation :
  ∃! op : ℤ → ℤ → ℤ, (op 8 4) + 5 - (3 * 1) = 6 ∧ op = (λ x y ↦ x - y) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_operation_satisfies_equation_l832_83265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l832_83222

-- Define the circle C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 + 4*p.1 = 0}

-- Define point P
def P : ℝ × ℝ := (-1, 0)

-- Define the line l (implicitly, as it passes through P)
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = -1 + t ∧ p.2 = t}

-- Define points A and B (implicitly, as intersections of l and C)
noncomputable def A : ℝ × ℝ := sorry
noncomputable def B : ℝ × ℝ := sorry

-- Define tangent lines l1 and l2 (implicitly)
def l1 : Set (ℝ × ℝ) := sorry
def l2 : Set (ℝ × ℝ) := sorry

-- Define point Q as the intersection of l1 and l2
noncomputable def Q : ℝ × ℝ := sorry

-- Define the center of the circle
def center : ℝ × ℝ := (-2, 0)

theorem circle_properties :
  -- 1. The shortest chord length between l and C is 2√3
  (∀ p : ℝ × ℝ, p ∈ l ∩ C → ‖p - P‖ ≥ 2 * Real.sqrt 3) ∧
  -- 2. Points Q, A, B, and center are concyclic
  (∃ (r : ℝ), ‖Q - center‖ = r ∧ ‖A - center‖ = r ∧ ‖B - center‖ = r) ∧
  -- 3. Q always lies on the line x = 2
  Q.1 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l832_83222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_exponential_equation_l832_83288

theorem no_solutions_exponential_equation :
  ¬∃ x : ℝ, x > -2 ∧ (2 : ℝ)^(4*x+2) * (8 : ℝ)^(2*x+4) = (16 : ℝ)^(3*x+5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solutions_exponential_equation_l832_83288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_intercept_and_f_on_curve_l832_83261

/-- Triangle ABC with vertices A(4,2), B(3,0), and C(4,0) -/
noncomputable def A : ℝ × ℝ := (4, 2)
noncomputable def B : ℝ × ℝ := (3, 0)
noncomputable def C : ℝ × ℝ := (4, 0)

/-- E is the point of symmetry -/
noncomputable def E : ℝ × ℝ := (2, 2)

/-- D is the midpoint of AC -/
noncomputable def D : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

/-- k is determined by the midpoint D -/
noncomputable def k : ℝ := D.1 * D.2

/-- F is symmetric to B with respect to E -/
noncomputable def F : ℝ × ℝ := (2 * E.1 - B.1, 2 * E.2 - B.2)

/-- The y-intercept of line AB is -6 and F lies on y = k/x -/
theorem ab_intercept_and_f_on_curve : 
  (A.2 - B.2) / (A.1 - B.1) * 0 + (A.2 - (A.2 - B.2) / (A.1 - B.1) * A.1) = -6 ∧
  F.2 = k / F.1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ab_intercept_and_f_on_curve_l832_83261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_decrease_minimum_value_range_of_k_l832_83211

noncomputable section

def f (k : ℝ) (x : ℝ) : ℝ := Real.log x + k / x

variable (k : ℝ)

theorem interval_of_decrease (h : (deriv (f k)) (Real.exp 1) = 0) :
  ∀ x ∈ Set.Ioo 0 (Real.exp 1), (deriv (f k)) x < 0 :=
sorry

theorem minimum_value :
  f (Real.exp 1) (Real.exp 1) = 2 :=
sorry

theorem range_of_k :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 → f k x₁ - f k x₂ < x₁ - x₂) →
  k ≥ 1/4 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_decrease_minimum_value_range_of_k_l832_83211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_axis_l832_83293

noncomputable def f (x φ : ℝ) : ℝ := 3 * Real.sin (2 * x + φ)

theorem symmetric_sine_axis (φ : ℝ) 
  (h_sym : ∀ x, f x φ = f (2 * (π / 3) - x) φ) 
  (h_phi : |φ| < π / 2) : 
  ∃ k : ℤ, ∀ x, f x φ = f (π / 6 - x) φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_sine_axis_l832_83293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_team_points_l832_83225

/-- Represents the points scored by a team in the tournament -/
def TeamPoints := Nat

/-- The number of teams in the tournament -/
def numTeams : Nat := 5

/-- The points awarded for a win -/
def winPoints : Nat := 3

/-- The points awarded for a draw -/
def drawPoints : Nat := 1

/-- The points awarded for a loss -/
def lossPoints : Nat := 0

/-- The total number of games played in the tournament -/
def totalGames : Nat := numTeams * (numTeams - 1) / 2

/-- The list of points scored by the first four teams -/
def knownTeamPoints : List Nat := [1, 2, 5, 7]

/-- Theorem: The fifth team must have scored 5 points -/
theorem fifth_team_points : 
  ∃ (fifthTeamPoints : Nat),
    fifthTeamPoints = 5 ∧
    (fifthTeamPoints + knownTeamPoints.sum = 2 * totalGames) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_team_points_l832_83225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_points_theorem_l832_83273

/-- Square with side length 1200 -/
def Square := {s : ℝ // s = 1200}

/-- Point on the side of the square -/
def PointOnSide := {x : ℝ // 0 ≤ x ∧ x ≤ 1200}

/-- Definition of angle in degrees -/
def Angle := {θ : ℝ // 0 ≤ θ ∧ θ < 360}

theorem square_points_theorem (ABCD : Square) (O : ℝ × ℝ) (E F : PointOnSide) 
  (h_center : O = (600, 600))
  (h_E_F : E.val < F.val)
  (h_angle : ∃ θ : Angle, θ.val = 60)
  (h_EF : F.val - E.val = 500)
  (h_BF : ∃ (p q r : ℕ), F.val = 1200 - (p + q * Real.sqrt r) ∧ 
    Nat.Coprime r (Nat.sqrt r)) :
  ∃ (p q r : ℕ), F.val = 1200 - (p + q * Real.sqrt r) ∧ p + q + r = 503 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_points_theorem_l832_83273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_187_5_l832_83291

/-- Regular hexagon ABCDEF is the base of right pyramid QABCDEF. 
    QAD is an equilateral triangle with side length 10. -/
noncomputable def pyramid_volume : ℝ :=
  let base_area : ℝ := 37.5 * Real.sqrt 3
  let height : ℝ := 5 * Real.sqrt 3
  (1 / 3) * base_area * height

/-- The volume of the pyramid QABCDEF is 187.5 -/
theorem pyramid_volume_is_187_5 : pyramid_volume = 187.5 := by
  unfold pyramid_volume
  simp [Real.sqrt_mul_self]
  norm_num
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_187_5_l832_83291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l832_83232

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 3 * x + 4 else 6 - 3 * x

-- Theorem to prove f(-1) = 1 and f(2) = 0
theorem f_values : f (-1) = 1 ∧ f 2 = 0 := by
  apply And.intro
  · -- Prove f(-1) = 1
    simp [f]
    norm_num
  · -- Prove f(2) = 0
    simp [f]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l832_83232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_survival_probability_l832_83284

/-- Represents a transplantation experiment with number of trees and survival rate -/
structure TransplantationData where
  num_trees : ℕ
  survival_rate : ℚ

/-- Function to find the maximum number of trees in a list of TransplantationData -/
def max_num_trees (data : List TransplantationData) : ℕ :=
  match (data.map (·.num_trees)).maximum? with
  | some n => n
  | none => 0

/-- Function to find the survival rate for the maximum number of trees -/
def survival_rate_for_max_trees (data : List TransplantationData) : ℚ :=
  match data.find? (·.num_trees == max_num_trees data) with
  | some d => d.survival_rate
  | none => 0

/-- Rounds a rational number to the nearest tenth -/
def round_to_tenth (q : ℚ) : ℚ :=
  (q * 10).floor / 10

/-- Main theorem: The estimated survival probability rounded to 0.1 is 0.9 -/
theorem estimated_survival_probability
  (data : List TransplantationData)
  (h_nonempty : data.length > 0)
  (h_data : data = [
    ⟨100, 87/100⟩,
    ⟨1000, 893/1000⟩,
    ⟨5000, 4485/5000⟩,
    ⟨8000, 7224/8000⟩,
    ⟨10000, 8983/10000⟩,
    ⟨15000, 13443/15000⟩,
    ⟨20000, 18044/20000⟩
  ]) :
  round_to_tenth (survival_rate_for_max_trees data) = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_estimated_survival_probability_l832_83284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_6_5953_l832_83227

/-- Rounds a real number to the nearest hundredth -/
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  (⌊x * 100 + 0.5⌋ : ℝ) / 100

/-- The original number to be rounded -/
def original_number : ℝ := 6.5953

/-- Theorem stating that rounding 6.5953 to the nearest hundredth equals 6.60 -/
theorem round_to_hundredth_6_5953 :
  round_to_hundredth original_number = 6.60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_hundredth_6_5953_l832_83227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_of_S_l832_83260

-- Define the set S
def S : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ‖‖|p.1| - 3‖ - 1‖ + ‖‖|p.2| - 3‖ - 1‖ = 2}

-- Define the total length of lines in S
noncomputable def total_length (S : Set (ℝ × ℝ)) : ℝ :=
  sorry -- Definition of total length calculation

-- Theorem statement
theorem total_length_of_S : total_length S = 128 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_of_S_l832_83260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l832_83234

noncomputable section

/-- The curve function -/
def f (x : ℝ) : ℝ := (1/2) * x^2 + x

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := x + 1

/-- The point of interest on the curve -/
def point : ℝ × ℝ := (2, 4)

/-- The slope of the tangent line at the point of interest -/
def m : ℝ := f' point.1

/-- The y-intercept of the tangent line -/
def y_intercept : ℝ := point.2 - m * point.1

/-- The x-intercept of the tangent line -/
def x_intercept : ℝ := -y_intercept / m

/-- The area of the triangle formed by the tangent line and the coordinate axes -/
def triangle_area : ℝ := (1/2) * x_intercept * point.2

/-- Theorem stating that the area of the triangle is 8/3 -/
theorem tangent_triangle_area : triangle_area = 8/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l832_83234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_replacement_in_march_l832_83228

/-- Represents months of the year -/
inductive Month
| january | february | march | april | may | june 
| july | august | september | october | november | december

/-- Calculates the next month -/
def nextMonth (m : Month) : Month :=
  match m with
  | .january => .february
  | .february => .march
  | .march => .april
  | .april => .may
  | .may => .june
  | .june => .july
  | .july => .august
  | .august => .september
  | .september => .october
  | .october => .november
  | .november => .december
  | .december => .january

/-- Calculates the month after a given number of months have passed -/
def monthAfter (start : Month) (months : Nat) : Month :=
  match months with
  | 0 => start
  | n + 1 => monthAfter (nextMonth start) n

/-- The theorem stating that the 15th replacement will be in March -/
theorem fifteenth_replacement_in_march :
  let replacementInterval := 7
  let firstReplacement := Month.january
  let totalReplacements := 15
  monthAfter firstReplacement ((totalReplacements - 1) * replacementInterval) = Month.march := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifteenth_replacement_in_march_l832_83228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l832_83252

theorem min_abs_difference (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : a * b - 3 * a + 4 * b = 211) :
  ∃ (a' b' : ℤ), a' > 0 ∧ b' > 0 ∧ a' * b' - 3 * a' + 4 * b' = 211 ∧
  ∀ (c d : ℤ), c > 0 → d > 0 → c * d - 3 * c + 4 * d = 211 →
  |a' - b'| ≤ |c - d| ∧
  |a' - b'| = 191 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l832_83252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l832_83274

/-- Prove that given two trains with specified lengths and initial separation,
    moving towards each other with one train at a known speed,
    the speed of the other train can be determined based on the time they take to meet. -/
theorem train_speed_calculation (train1_length train2_length initial_separation : ℝ)
                                (train2_speed meeting_time : ℝ) :
  train1_length = 100 →
  train2_length = 200 →
  initial_separation = 840 →
  train2_speed = 72 / 3.6 →
  meeting_time = 23.99808015358771 →
  ∃ (train1_speed : ℝ),
    (train1_speed + train2_speed) * meeting_time = initial_separation + train1_length + train2_length ∧
    abs (train1_speed - 27.5) < 0.1 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l832_83274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_interval_l832_83269

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 6))

theorem g_increasing_on_interval :
  StrictMonoOn g (Set.Icc (-5 * Real.pi / 12) (-Real.pi / 6)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_increasing_on_interval_l832_83269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l832_83209

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The vertices of the quadrilateral -/
def v1 : Point := ⟨1, 2⟩
def v2 : Point := ⟨4, 5⟩
def v3 : Point := ⟨5, 4⟩
def v4 : Point := ⟨4, 1⟩

/-- The perimeter of the quadrilateral -/
noncomputable def perimeter : ℝ :=
  distance v1 v2 + distance v2 v3 + distance v3 v4 + distance v4 v1

theorem quadrilateral_perimeter :
  ∃ (c d : ℤ), perimeter = c * Real.sqrt 2 + d * Real.sqrt 10 ∧ c + d = 6 := by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l832_83209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_square_eccentricity_l832_83216

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- An ellipse defined by its semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_ge_b : a ≥ b

/-- Check if four points form a square -/
def is_square (p1 p2 p3 p4 : Point) : Prop :=
  let d12 := (p1.x - p2.x)^2 + (p1.y - p2.y)^2
  let d23 := (p2.x - p3.x)^2 + (p2.y - p3.y)^2
  let d34 := (p3.x - p4.x)^2 + (p3.y - p4.y)^2
  let d41 := (p4.x - p1.x)^2 + (p4.y - p1.y)^2
  d12 = d23 ∧ d23 = d34 ∧ d34 = d41 ∧ d41 > 0

/-- Check if a point is on the ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b^2 / e.a^2))

/-- The main theorem -/
theorem ellipse_square_eccentricity (e : Ellipse) (p1 p2 p3 : Point) :
  (on_ellipse p1 e ∧ on_ellipse p2 e ∧ on_ellipse p3 e) →
  (is_square (Point.mk 0 0) p1 p2 p3) →
  eccentricity e = Real.sqrt 6 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_square_eccentricity_l832_83216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_region_area_l832_83245

/-- A square with side length 3 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 0) ∧ B = (3, 0) ∧ C = (3, 3) ∧ D = (0, 3))

/-- The angle between three points -/
noncomputable def angle (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The area of a region -/
noncomputable def area (R : Set (ℝ × ℝ)) : ℝ := sorry

/-- The set of points P satisfying the angle condition -/
def red_region (s : Square) : Set (ℝ × ℝ) :=
  {P | angle P s.A s.B ≥ Real.pi/3 ∧
       angle P s.B s.C ≥ Real.pi/3 ∧
       angle P s.C s.D ≥ Real.pi/3 ∧
       angle P s.D s.A ≥ Real.pi/3}

theorem red_region_area (s : Square) :
  area (red_region s) = 2 * Real.pi + 3 - 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_region_area_l832_83245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_l832_83249

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by x = k -/
structure VerticalLine where
  k : ℝ

/-- Distance between a point and a vertical line -/
def distToVerticalLine (p : Point) (l : VerticalLine) : ℝ :=
  |p.x - l.k|

/-- Distance between two points -/
noncomputable def distBetweenPoints (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The fixed point F -/
def F : Point := ⟨1, -1⟩

/-- The line l -/
def l : VerticalLine := ⟨1⟩

theorem trajectory_is_line :
  ∀ p : Point, distBetweenPoints p F = distToVerticalLine p l →
  ∃ m b : ℝ, p.y = m * p.x + b :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_l832_83249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l832_83239

noncomputable def complex_number : ℂ := (1 : ℂ) / (1 - Complex.I)^2 - Complex.I^4

theorem complex_number_in_second_quadrant :
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l832_83239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_cost_price_l832_83268

/-- Represents the selling price after applying a profit percentage to a cost price -/
noncomputable def apply_profit (cost_price : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

/-- Theorem: Given a bicycle sold twice with profits of 35% and 45% respectively,
    and a final selling price of 225, the original cost price is approximately 114.94 -/
theorem bicycle_cost_price (final_price : ℝ) 
    (h1 : final_price = 225) 
    (first_profit : ℝ) (h2 : first_profit = 35)
    (second_profit : ℝ) (h3 : second_profit = 45) :
    ∃ (original_cost : ℝ), 
      (apply_profit (apply_profit original_cost first_profit) second_profit) = final_price ∧ 
      abs (original_cost - 114.94) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_cost_price_l832_83268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l832_83244

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse with semi-major axis 5 and semi-minor axis 3 -/
def Ellipse (p : Point) : Prop :=
  p.x^2 / 25 + p.y^2 / 9 = 1

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Angle between three points -/
noncomputable def angle (A B C : Point) : ℝ :=
  Real.arccos ((distance A B)^2 + (distance B C)^2 - (distance A C)^2) / (2 * distance A B * distance B C)

/-- Theorem: Area of triangle F₁MF₂ is 3√3 -/
theorem ellipse_triangle_area
  (M F1 F2 : Point)
  (h_ellipse : Ellipse M)
  (h_foci : distance F1 F2 = 8)
  (h_angle : angle F1 M F2 = Real.pi / 3) : -- 60° in radians
  (1/2) * distance M F1 * distance M F2 * Real.sin (Real.pi / 3) = 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_triangle_area_l832_83244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_values_iff_a_in_range_l832_83266

/-- The function f(x) = x³ + ax² + (a+6)x + 1 has no extreme values if and only if -3 ≤ a ≤ 6 --/
theorem no_extreme_values_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ∃ ε > 0, ∀ y : ℝ, |y - x| < ε → 
    (x^3 + a*x^2 + (a+6)*x + 1 ≠ y^3 + a*y^2 + (a+6)*y + 1 ∨ x = y)) ↔ 
  (-3 ≤ a ∧ a ≤ 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_values_iff_a_in_range_l832_83266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l832_83250

/-- The focus of a parabola with equation x² = 8y is at the point (0, 2) -/
theorem parabola_focus (x y : ℝ) : 
  x^2 = 8*y → (0, 2) = (x, y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_l832_83250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l832_83292

theorem problem_solution (x z : ℕ) (y : ℤ) 
  (h1 : ∃ k : ℕ, x = 2 * k + 1)  -- x is odd
  (h2 : x ^ 3 % 11 = 4)  -- x^3 divided by 11 has remainder 4
  (h3 : x ^ 3 / 11 = y)  -- x^3 divided by 11 has quotient y
  (h4 : z ^ 2 % 6 = 1)  -- z^2 divided by 6 has remainder 1
  (h5 : z ^ 2 / 6 = 3 * y)  -- z^2 divided by 6 has quotient 3y
  (h6 : ∃ m : ℕ, z = 5 * m)  -- z is divisible by 5
  : 7 * y - x ^ 2 + z = 67 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l832_83292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthic_triangle_circumradius_l832_83285

-- Define a triangle with its angles and circumradius
structure Triangle where
  α : Real
  β : Real
  γ : Real
  R : Real
  angle_sum : α + β + γ = Real.pi

-- Define the orthic triangle
noncomputable def orthic_triangle (t : Triangle) : Triangle where
  α := Real.pi - 2 * t.α
  β := Real.pi - 2 * t.β
  γ := Real.pi - 2 * t.γ
  R := t.R / 2
  angle_sum := by
    sorry -- Proof of angle sum is omitted for brevity

-- Theorem statement
theorem orthic_triangle_circumradius (t : Triangle) :
  (orthic_triangle t).R = t.R / 2 := by
  -- Proof is trivial due to the definition of orthic_triangle
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthic_triangle_circumradius_l832_83285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l832_83243

noncomputable section

-- Define the function f(x) = e^(-x)
def f (x : ℝ) : ℝ := Real.exp (-x)

-- Define the point of tangency
def M : ℝ × ℝ := (1, Real.exp (-1))

-- Define the slope of the tangent line
def m : ℝ := -Real.exp (-1)

-- Define the y-intercept of the tangent line
def b : ℝ := 2 * Real.exp (-1)

-- Define the x-intercept of the tangent line
def x_intercept : ℝ := 2

-- Define the y-intercept of the tangent line
def y_intercept : ℝ := b

end noncomputable section

-- Theorem: The area of the triangle is 2/e
theorem tangent_triangle_area :
  (1/2 : ℝ) * x_intercept * y_intercept = 2 / Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l832_83243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_yield_is_ten_percent_l832_83235

/-- Represents the characteristics of a stock --/
structure Stock where
  dividend_rate : ℚ  -- Dividend rate as a rational number
  quote : ℚ         -- Current market price
  par_value : ℚ     -- Par value of the stock

/-- Calculates the yield percentage of a stock --/
def yield_percentage (s : Stock) : ℚ :=
  (s.dividend_rate * s.par_value / s.quote) * 100

/-- Theorem stating that the yield percentage of the given stock is 10% --/
theorem stock_yield_is_ten_percent 
  (s : Stock) 
  (h1 : s.dividend_rate = 8/100) 
  (h2 : s.quote = 80) 
  (h3 : s.par_value = 100) : 
  yield_percentage s = 10 := by
  sorry

#eval yield_percentage ⟨8/100, 80, 100⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_yield_is_ten_percent_l832_83235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repunit_polynomial_characterization_l832_83207

/-- A repunit is a positive integer whose digits in base 10 are all ones. -/
def IsRepunit (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

/-- The form of polynomials that map repunits to repunits. -/
def RepunitPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ m : ℕ, ∃ r : ℝ, r ≥ 1 - m ∧
    ∀ x, f x = (10^r * (9*x + 1)^m - 1) / 9

/-- Theorem stating the characterization of polynomials that map repunits to repunits. -/
theorem repunit_polynomial_characterization (f : ℝ → ℝ) :
  (∀ n : ℕ, IsRepunit n → IsRepunit (Int.toNat ⌊f n⌋)) ↔ RepunitPolynomial f :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repunit_polynomial_characterization_l832_83207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_J_3_neg15_10_l832_83230

/-- Definition of function J for nonzero real numbers -/
noncomputable def J (a b c : ℝ) : ℝ :=
  a / b + b / c + c / a

/-- Theorem stating that J(3, -15, 10) = 49/30 -/
theorem J_3_neg15_10 :
  J 3 (-15) 10 = 49 / 30 := by
  -- Unfold the definition of J
  unfold J
  -- Simplify the arithmetic expressions
  simp [div_eq_mul_inv]
  -- Perform the calculations
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_J_3_neg15_10_l832_83230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_set_correct_l832_83289

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  sqrt 3 * sin (2 * x - π / 6) + 2 * (sin (x - π / 12))^2

-- Theorem for the smallest positive period
theorem smallest_positive_period : 
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧ 
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ 
  T = π :=
sorry

-- Define the set of x for which f attains its maximum value
def max_value_set : Set ℝ := {x | ∃ (k : ℤ), x = k * π + 5 * π / 12}

-- Theorem for the maximum value set
theorem max_value_set_correct :
  ∀ (x : ℝ), x ∈ max_value_set ↔ 
    (∀ (y : ℝ), f y ≤ f x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_set_correct_l832_83289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l832_83218

noncomputable section

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2/a^2)

-- Define distance from origin to line
noncomputable def distanceToLine (b : ℝ) : ℝ := b / Real.sqrt 2

theorem ellipse_properties 
  (a b : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : eccentricity a b = Real.sqrt 6 / 3) 
  (h4 : distanceToLine b = Real.sqrt 2) :
  -- Part I
  ∃ (a' b' : ℝ), Ellipse a' b' = Ellipse (Real.sqrt 12) 2 ∧
  -- Part II.i
  (∀ (A B : ℝ × ℝ) (slope : ℝ),
    A ∈ Ellipse a b → 
    B ∈ Ellipse a b → 
    slope = 1 →
    (B.1 - A.1) / (B.2 - A.2) = slope →
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = Real.sqrt 3 →
    b = 1) ∧
  -- Part II.ii
  (∀ (M A B : ℝ × ℝ) (l m : ℝ),
    M ∈ Ellipse a b →
    A ∈ Ellipse a b →
    B ∈ Ellipse a b →
    M.1 = l * A.1 + m * B.1 →
    M.2 = l * A.2 + m * B.2 →
    l^2 + m^2 = 1) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l832_83218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_arcs_equal_chords_l832_83200

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- An arc on a circle. -/
structure Arc (c : Circle) where
  start_angle : ℝ
  end_angle : ℝ

/-- A chord of a circle. -/
structure Chord (c : Circle) where
  start_point : ℝ × ℝ
  end_point : ℝ × ℝ

/-- The length of an arc. -/
def arc_length (c : Circle) (a : Arc c) : ℝ := 
  c.radius * (a.end_angle - a.start_angle)

/-- The length of a chord. -/
noncomputable def chord_length (c : Circle) (ch : Chord c) : ℝ := 
  Real.sqrt ((ch.end_point.1 - ch.start_point.1)^2 + (ch.end_point.2 - ch.start_point.2)^2)

/-- The theorem stating that equal arcs correspond to equal chords. -/
theorem equal_arcs_equal_chords (c : Circle) (a1 a2 : Arc c) (ch1 ch2 : Chord c) :
  arc_length c a1 = arc_length c a2 → chord_length c ch1 = chord_length c ch2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_arcs_equal_chords_l832_83200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_over_pi_eq_648_sqrt_7_l832_83254

/-- The volume of a cone divided by π, where the cone is formed from a 270-degree sector of a circle with radius 24. -/
noncomputable def cone_volume_over_pi : ℝ :=
  let sector_angle : ℝ := 270
  let circle_radius : ℝ := 24
  let base_radius : ℝ := circle_radius * (sector_angle / 360)
  let height : ℝ := Real.sqrt (circle_radius^2 - base_radius^2)
  (1/3) * base_radius^2 * height

theorem cone_volume_over_pi_eq_648_sqrt_7 : 
  cone_volume_over_pi = 648 * Real.sqrt 7 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_over_pi_eq_648_sqrt_7_l832_83254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l832_83203

def a : ℕ → ℚ
  | 0 => 1
  | n + 1 => (2^(n+2) * a n) / (a n + 2^(n+1))

def b (n : ℕ) : ℚ := (2*n+1)*(n+2)*(a n)

def S (n : ℕ) : ℚ := (Finset.range n).sum (λ i => b i)

theorem sequence_properties :
  (∀ n : ℕ, (2^(n+2) / a (n+1)) - (2^(n+1) / a n) = 1) ∧
  (∀ n : ℕ, a n = 2^(n+1) / (n+2)) ∧
  (∀ n : ℕ, S n = (2*n-1) * 2^(n+2) + 6) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l832_83203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_driving_time_l832_83214

/-- Given a road trip with two drivers, this theorem calculates the driving time of the second driver. -/
theorem friend_driving_time 
  (total_distance : ℝ) 
  (christina_speed : ℝ) 
  (christina_time : ℝ) 
  (friend_speed : ℝ) 
  (h1 : total_distance = 210) 
  (h2 : christina_speed = 30) 
  (h3 : christina_time = 3) 
  (h4 : friend_speed = 40) : 
  (total_distance - christina_speed * christina_time) / friend_speed = 3 := by
  sorry

-- Remove the #eval line as it's causing issues with universe levels

end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_driving_time_l832_83214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_decomposition_l832_83272

theorem polynomial_decomposition (m : ℕ) (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (hm : m > 1)
  (hf₁ : a₁ ≠ 0 ∨ b₁ ≠ 0)
  (hf₂ : a₂ ≠ 0 ∨ b₂ ≠ 0)
  (hf₃ : a₃ ≠ 0 ∨ b₃ ≠ 0)
  (heq : ∀ x, (a₁ * x + b₁)^m + (a₂ * x + b₂)^m = (a₃ * x + b₃)^m) :
  ∃ (c₁ c₂ c₃ : ℝ) (q : Polynomial ℝ),
    (∀ x, a₁ * x + b₁ = c₁ * (Polynomial.eval x q)) ∧
    (∀ x, a₂ * x + b₂ = c₂ * (Polynomial.eval x q)) ∧
    (∀ x, a₃ * x + b₃ = c₃ * (Polynomial.eval x q)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_decomposition_l832_83272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_concyclic_l832_83267

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the configuration of circles
structure CircleConfiguration where
  outer : Circle
  inner : Fin 4 → Circle
  tangent : ∀ i : Fin 4, (inner i).center.1^2 + (inner i).center.2^2 = 
    ((outer.radius - (inner i).radius)^2 : ℝ)
  clockwise : ∀ i j : Fin 4, i < j → 
    (inner i).center.1 * (inner j).center.2 - (inner i).center.2 * (inner j).center.1 > 0

-- Define the external common tangent between two circles
noncomputable def externalCommonTangent (c1 c2 : Circle) : ℝ := 
  Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 - (c1.radius - c2.radius)^2)

-- Define the quadrilateral formed by external common tangents
noncomputable def tangentQuadrilateral (config : CircleConfiguration) : Fin 4 → ℝ × ℝ :=
  λ i ↦ (externalCommonTangent (config.inner i) (config.inner ((i + 1) % 4)),
         externalCommonTangent (config.inner i) (config.inner ((i + 2) % 4)))

-- State the theorem
theorem tangent_quadrilateral_concyclic (config : CircleConfiguration) :
  let vertices := tangentQuadrilateral config
  ∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i : Fin 4,
    (vertices i).1^2 + (vertices i).2^2 = radius^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_concyclic_l832_83267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l832_83226

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 2/x - Real.log x

-- State the theorem
theorem f_has_zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zero_in_interval_l832_83226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_savings_difference_l832_83280

/-- Represents the savings from a coupon --/
def Savings (price : ℝ) : ℕ → ℝ
| 0 => 0.20 * price -- Coupon A
| 1 => 40 -- Coupon B
| 2 => 0.30 * (price - 150) -- Coupon C
| _ => 0

/-- The theorem to be proved --/
theorem coupon_savings_difference :
  ∃ (x y : ℝ),
    x > 150 ∧ y > 150 ∧
    (∀ p : ℝ, p > 150 →
      (Savings p 0 ≥ Savings p 1 ∧ Savings p 0 ≥ Savings p 2) →
      x ≤ p ∧ p ≤ y) ∧
    y - x = 250 := by
  sorry

#check coupon_savings_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coupon_savings_difference_l832_83280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_plus_2b_equals_sqrt_7_l832_83287

open Real

noncomputable def a : ℝ × ℝ := (cos (5 * π / 180), sin (5 * π / 180))
noncomputable def b : ℝ × ℝ := (cos (65 * π / 180), sin (65 * π / 180))

theorem magnitude_a_plus_2b_equals_sqrt_7 :
  sqrt ((a.1 + 2 * b.1)^2 + (a.2 + 2 * b.2)^2) = sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_a_plus_2b_equals_sqrt_7_l832_83287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_tangent_locus_is_perpendicular_bisector_locus_is_entire_line_l832_83255

/-- Two circles in a plane -/
structure TwoCircles where
  O₁ : EuclideanSpace ℝ (Fin 2)  -- Center of first circle
  O₂ : EuclideanSpace ℝ (Fin 2)  -- Center of second circle
  R : ℝ  -- Radius of first circle
  r : ℝ  -- Radius of second circle

/-- The locus of points with equal tangents to two circles -/
def equalTangentLocus (c : TwoCircles) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {M | ‖M - c.O₁‖^2 - ‖M - c.O₂‖^2 = c.R^2 - c.r^2}

/-- The perpendicular bisector of the line segment between the centers -/
def perpendicularBisector (c : TwoCircles) : Set (EuclideanSpace ℝ (Fin 2)) :=
  {M | ‖M - c.O₁‖^2 = ‖M - c.O₂‖^2}

theorem equal_tangent_locus_is_perpendicular_bisector (c : TwoCircles) :
  equalTangentLocus c = perpendicularBisector c := by
  sorry

/-- The locus is the entire line when the circles don't intersect or overlap externally -/
theorem locus_is_entire_line (c : TwoCircles) 
  (h : ‖c.O₁ - c.O₂‖ ≥ c.R + c.r) :
  equalTangentLocus c = perpendicularBisector c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_tangent_locus_is_perpendicular_bisector_locus_is_entire_line_l832_83255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l832_83253

-- Define the function f(x) = 2/x
noncomputable def f (x : ℝ) : ℝ := 2 / x

-- Define the derivative of f
noncomputable def f_derivative (x : ℝ) : ℝ := -2 / (x^2)

-- Theorem: The equation of the tangent line to f(x) at (1, 2) is 2x + y - 4 = 0
theorem tangent_line_equation :
  ∀ x y : ℝ, f 1 = 2 → f_derivative 1 = -2 →
  (2 * x + y - 4 = 0 ↔ y - 2 = f_derivative 1 * (x - 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l832_83253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l832_83212

theorem binomial_expansion_coefficient (a : ℝ) : a > 0 →
  (∃ c : ℝ, c = 160 ∧ 
    c = (Finset.sum (Finset.range 6) (λ k ↦
      (Nat.choose 5 k) * (2^(5-k)) * ((-a)^k) * ↑(Nat.choose k 2)))) →
  a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_coefficient_l832_83212
