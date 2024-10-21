import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l736_73663

-- Define the constants
noncomputable def a : ℝ := 2^(3/10)
noncomputable def b : ℝ := (3/10)^2
noncomputable def c : ℝ := Real.log 2 / Real.log (3/10)

-- Theorem statement
theorem ascending_order : c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascending_order_l736_73663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteer_selection_probability_l736_73673

/-- The number of male students -/
def num_male : ℕ := 5

/-- The number of female students -/
def num_female : ℕ := 2

/-- The total number of students -/
def total_students : ℕ := num_male + num_female

/-- The number of volunteers to be selected -/
def num_volunteers : ℕ := 3

/-- The probability of selecting at least one male and one female student -/
def probability : ℚ := 5 / 7

theorem volunteer_selection_probability :
  (Nat.choose num_male 1 * Nat.choose num_female 2 + Nat.choose num_male 2 * Nat.choose num_female 1) / 
  (Nat.choose total_students num_volunteers) = probability := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volunteer_selection_probability_l736_73673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l736_73632

/-- Definition of a polynomial being factorable modulo a prime p -/
def IsFactorableModP (P : Polynomial ℤ) (p : ℕ) : Prop :=
  ∃ (f g : Polynomial ℤ), (∀ x : ℤ, P.eval x ≡ (f.eval x * g.eval x) [ZMOD p]) ∧ 
  f.degree > 0 ∧ g.degree > 0

/-- The main theorem to be proved -/
theorem polynomial_factorization :
  /- Part 1: x^4 - 2x^3 + 3x^2 - 2x - 5 is factorable modulo 5 -/
  IsFactorableModP (X^4 - 2*X^3 + 3*X^2 - 2*X - 5) 5 ∧
  /- Part 2: x^6 + 3 is irreducible over ℤ but factorable modulo every prime -/
  (Irreducible (X^6 + 3 : Polynomial ℤ) ∧ ∀ p : ℕ, Nat.Prime p → IsFactorableModP (X^6 + 3) p) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorization_l736_73632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_comparison_l736_73649

theorem sqrt_comparison : Real.sqrt 6 + Real.sqrt 7 > 2 * Real.sqrt 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_comparison_l736_73649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_determines_w_l736_73617

noncomputable def vector_to_project : ℝ → Fin 3 → ℝ := λ w i => 
  match i with
  | 0 => 2
  | 1 => 4
  | 2 => w

noncomputable def vector_to_project_onto : Fin 3 → ℝ
| 0 => 4
| 1 => -3
| 2 => 2

noncomputable def projection_scalar : ℝ := 10 / 29

theorem projection_determines_w : 
  ∃ w : ℝ, 
    (∀ i : Fin 3, vector_to_project w i • vector_to_project_onto i = projection_scalar * (vector_to_project_onto i • vector_to_project_onto i)) ∧
    w = 7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_determines_w_l736_73617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_side_of_triangle_l736_73683

theorem smallest_side_of_triangle : ∃ (t : ℕ),
  (78 + 51 > 10 * t) ∧
  (78 + 10 * t > 51) ∧
  (51 + 10 * t > 78) ∧
  (∀ (s : ℕ), s < t →
    (78 + 51 ≤ 10 * s) ∨
    (78 + 10 * s ≤ 51) ∨
    (51 + 10 * s ≤ 78)) ∧
  t = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_side_of_triangle_l736_73683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_kite_l736_73634

/-- Represents a parabola in the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The area of a kite given its diagonals -/
noncomputable def kiteArea (d1 d2 : ℝ) : ℝ := (1/2) * d1 * d2

theorem parabola_intersection_kite (c d : ℝ) :
  let p1 : Parabola := ⟨c, 3⟩
  let p2 : Parabola := ⟨-d, 5⟩
  -- Condition: The parabolas intersect the coordinate axes at exactly four points
  -- Condition: These points form the vertices of a kite
  -- Condition: The area of the kite is 8 square units
  (∃ (x1 x2 y1 y2 : ℝ),
    x1 ≠ 0 ∧ x2 ≠ 0 ∧ y1 ≠ 0 ∧ y2 ≠ 0 ∧
    c * x1^2 + 3 = 0 ∧ c * x2^2 + 3 = 0 ∧
    5 - d * x1^2 = 0 ∧ 5 - d * x2^2 = 0 ∧
    kiteArea (2 * |x2 - x1|) (|y2 - y1|) = 8) →
  c + d = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_kite_l736_73634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l736_73693

-- Define the function s(x) as noncomputable
noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

-- State the theorem about the range of s(x)
theorem s_range :
  ∀ y : ℝ, y ≠ 0 → ∃ x : ℝ, x ≠ 2 ∧ s x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_s_range_l736_73693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l736_73607

theorem log_equation_solution :
  ∃ x : ℝ, x > 0 ∧ Real.log 8 / Real.log x = Real.log 5 / Real.log 125 ∧ x = 512 := by
  -- We use Real.log 8 / Real.log x instead of Real.log x 8 because Lean uses this notation for logarithms
  -- Similarly, we use Real.log 5 / Real.log 125 instead of Real.log 125 5
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l736_73607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_count_l736_73653

theorem integer_root_count : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, x ≥ 0 ∧ ∃ n : ℤ, (256 - x^(1/3))^(1/2) = n) ∧ 
  Finset.card S = 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_count_l736_73653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_votes_in_election_l736_73690

/-- Represents the number of votes for each candidate in an election --/
structure ElectionResult where
  john : ℕ
  james : ℕ
  third : ℕ

/-- Checks if the given election result satisfies the conditions of the problem --/
def is_valid_election (result : ElectionResult) : Prop :=
  result.john + result.james + result.third = 1150 ∧
  result.third = result.john + 150 ∧
  result.james = Int.floor (0.7 * (1150 - result.john - result.third : ℚ))

theorem john_votes_in_election :
  ∃ (result : ElectionResult), is_valid_election result ∧ result.john = 500 := by
  sorry

#eval Int.floor (0.7 * (1150 - 500 - 650 : ℚ))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_votes_in_election_l736_73690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodicity_l736_73628

def sequence_u (a : ℕ) : ℕ → ℕ
  | 0 => a
  | n + 1 => a ^ (sequence_u a n)

theorem sequence_periodicity (a m : ℕ) (ha : a > 0) (hm : m > 0) :
  ∃ k N, ∀ n ≥ N, sequence_u a (n + k) ≡ sequence_u a n [MOD m] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_periodicity_l736_73628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_burritos_five_quesadillas_cost_l736_73666

/-- Represents the cost of food items at Ramon's food stand -/
structure FoodStandPrices where
  burrito : ℚ
  quesadilla : ℚ
  startingFee : ℚ

/-- Calculates the total cost for a given number of burritos and quesadillas -/
def totalCost (prices : FoodStandPrices) (numBurritos numQuesadillas : ℕ) : ℚ :=
  prices.burrito * numBurritos + prices.quesadilla * numQuesadillas + prices.startingFee

/-- Theorem stating the cost of 5 burritos and 5 quesadillas given the conditions -/
theorem five_burritos_five_quesadillas_cost
  (prices : FoodStandPrices)
  (h1 : totalCost prices 4 2 = 25/2)
  (h2 : totalCost prices 3 4 = 15)
  (h3 : prices.startingFee = 2) :
  totalCost prices 5 5 = 81/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_burritos_five_quesadillas_cost_l736_73666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l736_73619

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) / (x + 1)

-- State the theorem
theorem function_composition_identity (a b : ℝ) :
  (∀ x : ℝ, x ≠ -1 → f a b (f a b x) = x) ↔ ((a = 0 ∧ b = 0) ∨ (a = -1 ∧ b = -1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l736_73619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_no_complete_group_l736_73692

/-- A class of students with special grouping properties -/
structure SpecialClass where
  students : Finset Nat
  groups : Finset (Finset Nat)
  size_eq_46 : students.card = 46
  groups_size_3 : ∀ g, g ∈ groups → g.card = 3
  groups_intersection : ∀ g1 g2, g1 ∈ groups → g2 ∈ groups → g1 ≠ g2 → (g1 ∩ g2).card ≤ 1
  groups_subset : ∀ g, g ∈ groups → g ⊆ students

/-- A subset of students containing no complete group -/
def NoCompleteGroup (c : SpecialClass) (s : Finset Nat) : Prop :=
  s ⊆ c.students ∧ ∀ g, g ∈ c.groups → ¬(g ⊆ s)

/-- The theorem to be proved -/
theorem exists_large_no_complete_group (c : SpecialClass) :
  ∃ s : Finset Nat, s.card ≥ 10 ∧ NoCompleteGroup c s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_large_no_complete_group_l736_73692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_size_l736_73645

-- Define the number of subsets function
def n (S : Finset α) : ℕ := 2^(Finset.card S)

-- Theorem statement
theorem min_intersection_size 
  (A B C : Finset ℕ) 
  (h1 : n A + n B + n C = n (A ∪ B ∪ C)) 
  (h2 : Finset.card A = 100) 
  (h3 : Finset.card B = 100) :
  ∃ (x : ℕ), x = Finset.card (A ∩ B ∩ C) ∧ x ≥ 97 ∧ 
  ∀ (y : ℕ), y = Finset.card (A ∩ B ∩ C) → y ≥ x :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_size_l736_73645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_12_l736_73651

-- Define the cost function
def P (x : ℝ) : ℝ := 12 + 10 * x

-- Define the revenue function
noncomputable def Q (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 16 then -0.5 * x^2 + 22 * x
  else 224

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ := Q x - P x

-- State the theorem
theorem max_profit_at_12 :
  ∃ (max_profit : ℝ), max_profit = 60 ∧
  ∀ (x : ℝ), x ≥ 0 → f x ≤ max_profit ∧
  f 12 = max_profit := by
  sorry

#check max_profit_at_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_12_l736_73651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_216_l736_73644

/-- Represents a triangle with an arbitrary point and parallel lines --/
structure TriangleWithParallels where
  /-- The area of the first resulting triangle --/
  area1 : ℝ
  /-- The area of the second resulting triangle --/
  area2 : ℝ
  /-- The area of the third resulting triangle --/
  area3 : ℝ

/-- Calculates the area of the original triangle --/
noncomputable def calculateArea (t : TriangleWithParallels) : ℝ :=
  t.area1 + t.area2 + t.area3 + 2 * (Real.sqrt (t.area1 * t.area3) + Real.sqrt (t.area2 * t.area3) + Real.sqrt (t.area1 * t.area2))

/-- Theorem: The area of the original triangle is 216 cm² --/
theorem triangle_area_is_216 (t : TriangleWithParallels) 
  (h1 : t.area1 = 6) (h2 : t.area2 = 24) (h3 : t.area3 = 54) : 
  calculateArea t = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_216_l736_73644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_k_l736_73680

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are non-collinear if they are not scalar multiples of each other -/
def NonCollinear (a b : V) : Prop := ∀ (r : ℝ), a ≠ r • b

/-- Three points are collinear if the vector from the first to the third is a scalar multiple of the vector from the first to the second -/
def AreCollinear (A B D : V) : Prop := ∃ (r : ℝ), D - A = r • (B - A)

theorem value_of_k (a b : V) (k : ℝ) :
  NonCollinear a b →
  let AB := a + 2 • b
  let BC := 3 • a + k • b
  let CD := -a + b
  AreCollinear 0 AB (AB + BC + CD) →
  k = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_k_l736_73680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l736_73625

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angle_sum : A + B + C = Real.pi

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : (Real.cos t.A - 2 * Real.cos t.C) / Real.cos t.B = (2 * t.c - t.a) / t.b) :
  -- Part 1
  Real.sin t.C / Real.sin t.A = 2 ∧
  -- Part 2
  (Real.cos t.B = 1/4 ∧ t.b = 2) → 
    let S := (1/2) * t.a * t.c * Real.sin t.B
    S = Real.sqrt 15 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l736_73625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arithmetic_sequence_l736_73691

-- Define the interval (0°, 360°) in radians
def validAngle (b : ℝ) : Prop := 0 < b ∧ b < 2 * Real.pi

-- Define the arithmetic sequence condition
def isArithmeticSequence (b : ℝ) : Prop :=
  Real.sin b + Real.sin (3 * b) = 2 * Real.sin (2 * b)

-- Define the set of solutions in radians
def solutions : Set ℝ := {Real.pi/4, 3*Real.pi/4, 5*Real.pi/4, 7*Real.pi/4}

theorem sin_arithmetic_sequence :
  ∀ b : ℝ, validAngle b → (isArithmeticSequence b ↔ b ∈ solutions) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arithmetic_sequence_l736_73691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_connection_counts_l736_73601

/-- Represents a communication system with a fixed number of subscribers. -/
structure CommunicationSystem where
  subscriberCount : Nat
  connectionCount : Nat
  isValid : Prop := subscriberCount * connectionCount % 2 = 0

/-- Theorem stating the properties of valid connection counts in a system with 2001 subscribers. -/
theorem valid_connection_counts (sys : CommunicationSystem) 
  (h1 : sys.subscriberCount = 2001) 
  (h2 : ∃ (t : Nat), sys.connectionCount = 2 * t ∧ t ≤ 1000) :
  sys.isValid ∧ sys.connectionCount ≤ 2000 := by
  sorry

#check valid_connection_counts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_connection_counts_l736_73601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l736_73678

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (-2^x + b) / (2^x + a)

theorem odd_function_and_inequality (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  (a = 1 ∧ b = 1) ∧
  (∀ k : ℝ, (∀ t : ℝ, f 1 1 (t^2 - 2*t) + f 1 1 (2*t^2 - k) < 0) → k < -1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_and_inequality_l736_73678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_a_eq_two_l736_73669

/-- The function f(x) defined in terms of parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * (3^x) + 4 - a) / (4 * ((3^x) - 1))

/-- Theorem stating that f is an odd function if and only if a = 2 -/
theorem f_is_odd_iff_a_eq_two (a : ℝ) :
  (∀ x, f a x = -f a (-x)) ↔ a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_iff_a_eq_two_l736_73669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_bounce_count_l736_73675

noncomputable def initial_height : ℝ := 20
noncomputable def bounce_ratio : ℝ := 3/4
noncomputable def target_height : ℝ := 2

noncomputable def height_after_bounces (k : ℕ) : ℝ := initial_height * (bounce_ratio ^ k)

theorem smallest_bounce_count :
  ∃ k : ℕ, (∀ n < k, height_after_bounces n ≥ target_height) ∧
           (height_after_bounces k < target_height) ∧
           k = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_bounce_count_l736_73675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l736_73684

theorem sin_cos_identity (x : ℝ) : 
  Real.sin x ^ 9 * Real.cos x - Real.cos x ^ 9 * Real.sin x = Real.sin (4 * x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l736_73684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l736_73609

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.sin x) + Real.sqrt ((1/2) - Real.cos x)
noncomputable def g (x : ℝ) : ℝ := (Real.cos x)^2 - Real.sin x

-- Define the domain of f
def domain_f : Set ℝ := {x | ∃ k : ℤ, 2*k*Real.pi + Real.pi/3 ≤ x ∧ x ≤ 2*k*Real.pi + Real.pi}

-- Define the range of g
def range_g : Set ℝ := {y | (1 - Real.sqrt 2) / 2 ≤ y ∧ y < 5/4}

-- Theorem for the domain of f
theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f x = y} = domain_f := by sorry

-- Theorem for the range of g
theorem range_of_g : {y : ℝ | ∃ x : ℝ, -Real.pi/4 ≤ x ∧ x ≤ Real.pi/4 ∧ g x = y} = range_g := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_range_of_g_l736_73609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carousel_horses_count_l736_73657

theorem carousel_horses_count :
  let blue : ℕ := 5
  let purple : ℕ := 3 * blue
  let green : ℕ := 2 * purple
  let gold : ℕ := (2 * green) / 5
  let yellow : ℕ := gold / 2
  let silver : ℕ := yellow + 4
  let red : ℕ := 2 * silver - 7
  blue + purple + green + gold + yellow + silver + red = 91 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carousel_horses_count_l736_73657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_theorem_l736_73606

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 9 = 1

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define the line
def line (x y : ℝ) : Prop := y = 9*x - 26

-- Define the point P
def point_P : ℝ × ℝ := (3, 1)

theorem hyperbola_and_line_theorem :
  -- Given conditions
  (∃ (f : ℝ × ℝ), (∀ x y, ellipse x y → (x - f.1)^2 + (y - f.2)^2 = 25) ∧
                   (∀ x y, hyperbola x y → (x - f.1)^2 - (y - f.2)^2 = 4)) →
  (let e₁ := Real.sqrt (1 - 9/25)
   let e₂ := 14/5 - e₁
   e₂ = 2) →
  -- Prove the hyperbola equation
  (∀ x y, hyperbola x y ↔ x^2 / 4 - y^2 / 12 = 1) ∧
  -- Prove the line equation
  (∃ a b : ℝ × ℝ, 
    hyperbola a.1 a.2 ∧ hyperbola b.1 b.2 ∧
    point_P.1 = (a.1 + b.1) / 2 ∧ point_P.2 = (a.2 + b.2) / 2 ∧
    (∀ x y, line x y ↔ y = 9*x - 26)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_and_line_theorem_l736_73606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_assignment_l736_73688

/-- Represents the three types of inhabitants --/
inductive InhabitantType
  | TruthTeller
  | Liar
  | Trickster

/-- Represents the three inhabitants --/
inductive Inhabitant
  | K
  | M
  | R

/-- The function that assigns a type to each inhabitant --/
def assignment : Inhabitant → InhabitantType := sorry

/-- K's statement: "I am a trickster." --/
def k_statement (a : Inhabitant → InhabitantType) : Prop :=
  a Inhabitant.K = InhabitantType.Trickster

/-- M's statement: "That is true." (referring to K's statement) --/
def m_statement (a : Inhabitant → InhabitantType) : Prop :=
  k_statement a

/-- R's statement: "I am not a trickster." --/
def r_statement (a : Inhabitant → InhabitantType) : Prop :=
  a Inhabitant.R ≠ InhabitantType.Trickster

/-- A predicate that checks if a statement is true according to the inhabitant type --/
def is_true_statement (a : Inhabitant → InhabitantType) (i : Inhabitant) (s : Prop) : Prop :=
  match a i with
  | InhabitantType.TruthTeller => s
  | InhabitantType.Liar => ¬s
  | InhabitantType.Trickster => True

/-- The main theorem stating that the only valid assignment is K as Liar, M as Trickster, and R as TruthTeller --/
theorem unique_valid_assignment :
  ∀ (a : Inhabitant → InhabitantType),
    (∃! (x : Inhabitant), a x = InhabitantType.TruthTeller) ∧
    (∃! (x : Inhabitant), a x = InhabitantType.Liar) ∧
    (∃! (x : Inhabitant), a x = InhabitantType.Trickster) ∧
    (is_true_statement a Inhabitant.K (k_statement a)) ∧
    (is_true_statement a Inhabitant.M (m_statement a)) ∧
    (is_true_statement a Inhabitant.R (r_statement a)) →
    a Inhabitant.K = InhabitantType.Liar ∧
    a Inhabitant.M = InhabitantType.Trickster ∧
    a Inhabitant.R = InhabitantType.TruthTeller :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_assignment_l736_73688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_increase_l736_73615

/-- Calculates the percent increase between two prices -/
noncomputable def percentIncrease (initial : ℝ) (final : ℝ) : ℝ :=
  (final - initial) / initial * 100

/-- Theorem about share price increase -/
theorem share_price_increase (P : ℝ) (hP : P > 0) :
  let firstQuarterPrice := 1.20 * P
  let secondQuarterPrice := 1.60 * P
  percentIncrease firstQuarterPrice secondQuarterPrice = 100 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_share_price_increase_l736_73615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l736_73630

noncomputable def f (x : ℝ) : ℝ := (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) / (x^2 - 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -2 ∨ (-2 < x ∧ x < 2) ∨ 2 < x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l736_73630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_profit_proof_l736_73696

/-- Calculates the number of pencils needed to be sold to achieve a specific profit --/
def pencils_to_sell (num_purchased : ℕ) (cost_per_pencil : ℚ) (selling_price : ℚ) (desired_profit : ℚ) : ℕ :=
  let total_cost : ℚ := num_purchased * cost_per_pencil
  let revenue_needed : ℚ := total_cost + desired_profit
  (revenue_needed / selling_price).ceil.toNat

/-- Proves that selling 1200 pencils achieves the desired profit under given conditions --/
theorem pencil_profit_proof :
  pencils_to_sell 2000 (20/100) (50/100) 200 = 1200 := by
  sorry

#eval pencils_to_sell 2000 (20/100) (50/100) 200

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pencil_profit_proof_l736_73696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_bisector_l736_73641

-- Define the basic structures
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the points
variable (A K M₁ M₂ : ℝ × ℝ)

-- Define the circles
variable (C₁ C₂ : Circle)

-- Define necessary functions (without implementation)
def internal_common_tangents : Circle → Circle → Set (ℝ × ℝ) := sorry
def external_common_tangents : Circle → Circle → Set (ℝ × ℝ) := sorry
def is_orthogonal : (ℝ × ℝ) → Set (ℝ × ℝ) → Prop := sorry
def vector : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) := sorry
def boundary : Circle → Set (ℝ × ℝ) := sorry
def is_tangent : (ℝ × ℝ) → (ℝ × ℝ) → Circle → Prop := sorry
noncomputable def angle : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ := sorry
def bisects : (ℝ × ℝ) → ℝ → Prop := sorry

-- State the theorem
theorem tangent_bisector 
  (h_outside : C₁.center ≠ C₂.center ∧ (C₁.center.1 - C₂.center.1)^2 + (C₁.center.2 - C₂.center.2)^2 > (C₁.radius + C₂.radius)^2)
  (h_A : A ∈ internal_common_tangents C₁ C₂)
  (h_K : K ∈ external_common_tangents C₁ C₂ ∧ is_orthogonal (vector A K) (external_common_tangents C₁ C₂))
  (h_M₁ : M₁ ∈ boundary C₁ ∧ is_tangent K M₁ C₁)
  (h_M₂ : M₂ ∈ boundary C₂ ∧ is_tangent K M₂ C₂)
  : bisects (vector A K) (angle M₁ K M₂) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_bisector_l736_73641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_to_even_function_l736_73627

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + 2 * Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := f (x + 5 * Real.pi / 12)

theorem shift_to_even_function :
  (∀ x, g x = g (-x)) ∧
  (∀ h : ℝ, h < 5 * Real.pi / 12 → ∃ y, f (y + h) ≠ f (-y - h)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shift_to_even_function_l736_73627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_given_square_area_l736_73659

/-- Given a square and a rectangle, if the square's perimeter is 160 cm,
    the rectangle's width is 10 cm, and the square's area is five times
    the rectangle's area, then the rectangle's length is 32 cm. -/
theorem rectangle_length_given_square_area (square_perimeter : ℝ) (rect_width : ℝ) (rect_length : ℝ) :
  square_perimeter = 160 →
  rect_width = 10 →
  (square_perimeter / 4) ^ 2 = 5 * (rect_width * rect_length) →
  rect_length = 32 := by
  sorry

#check rectangle_length_given_square_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_length_given_square_area_l736_73659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l736_73637

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2 = 1

-- Define the line with slope 3
def line (x y m : ℝ) : Prop := y = 3*x + m

-- Define the intersection points A and B
def intersection_points (xA yA xB yB m : ℝ) : Prop :=
  hyperbola xA yA ∧ hyperbola xB yB ∧ line xA yA m ∧ line xB yB m

-- Define the midpoint M
def midpoint_of (xM yM xA yA xB yB : ℝ) : Prop :=
  xM = (xA + xB)/2 ∧ yM = (yA + yB)/2

-- Theorem statement
theorem midpoint_trajectory 
  (xA yA xB yB xM yM m : ℝ) 
  (h1 : intersection_points xA yA xB yB m)
  (h2 : midpoint_of xM yM xA yA xB yB) :
  xM - 12*yM = 0 ∧ |xM| > (12*Real.sqrt 35)/35 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_trajectory_l736_73637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l736_73668

-- Define the circle E
def circle_E (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

-- Define the points M, N, P, and C
noncomputable def point_M : ℝ × ℝ := (-1, 0)
noncomputable def point_N : ℝ × ℝ := (0, 1)
noncomputable def point_P : ℝ × ℝ := (1/2, -Real.sqrt 3/2)
noncomputable def point_C : ℝ × ℝ := (2, 2)

-- Define a line passing through point C
def line_through_C (m : ℝ) (x y : ℝ) : Prop :=
  y - point_C.2 = m * (x - point_C.1)

-- Define the intersection of a line with the circle E
def intersects_circle_E (m : ℝ) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    circle_E x₁ y₁ ∧ 
    circle_E x₂ y₂ ∧ 
    line_through_C m x₁ y₁ ∧ 
    line_through_C m x₂ y₂

-- Theorem statement
theorem line_equation : 
  circle_E point_M.1 point_M.2 ∧ 
  circle_E point_N.1 point_N.2 ∧ 
  circle_E point_P.1 point_P.2 →
  ∃ (m : ℝ), intersects_circle_E m ∧ 
    ∀ (x y : ℝ), line_through_C m x y ↔ x + y = 1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l736_73668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_pot_removal_l736_73654

/-- Represents a pot of coins -/
structure Pot where
  quantity : Nat
  is_silver : Bool

/-- The initial set of pots -/
def initial_pots : List Pot := [
  ⟨81, true⟩,
  ⟨71, true⟩,
  ⟨41, false⟩,
  ⟨37, false⟩,
  ⟨35, false⟩
]

/-- Calculates the total number of coins in a list of pots -/
def total_coins (pots : List Pot) : Nat :=
  pots.foldl (fun acc pot ↦ acc + pot.quantity) 0

/-- Calculates the number of silver coins in a list of pots -/
def silver_coins (pots : List Pot) : Nat :=
  pots.foldl (fun acc pot ↦ if pot.is_silver then acc + pot.quantity else acc) 0

/-- Calculates the number of gold coins in a list of pots -/
def gold_coins (pots : List Pot) : Nat :=
  pots.foldl (fun acc pot ↦ if not pot.is_silver then acc + pot.quantity else acc) 0

/-- The main theorem to be proved -/
theorem correct_pot_removal :
  let remaining_pots := initial_pots.filter (fun pot ↦ pot.quantity ≠ 37)
  silver_coins remaining_pots = 2 * gold_coins remaining_pots := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_pot_removal_l736_73654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roof_area_l736_73676

theorem roof_area (width length : ℚ) : 
  length = 4 * width →
  length - width = 28 →
  width * length = 348 + 4/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roof_area_l736_73676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l736_73650

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - 1 + Real.sqrt (6*x - x^2)

-- State the theorem
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 6 ∧ f x = y) ↔ -1 ≤ y ∧ y ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l736_73650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_to_original_volume_ratio_l736_73698

/-- A regular tetrahedron with an inner tetrahedron formed by connecting the centers of each face to the opposite vertex -/
structure InnerTetrahedron where
  /-- The side length of the original tetrahedron -/
  s : ℝ
  /-- Assumption that s is positive -/
  s_pos : s > 0

/-- The volume of the original tetrahedron -/
noncomputable def volume_original (t : InnerTetrahedron) : ℝ := t.s^3 * Real.sqrt 2 / 12

/-- The volume of the inner tetrahedron -/
noncomputable def volume_inner (t : InnerTetrahedron) : ℝ := 2 * Real.sqrt 2 * t.s^3 / 5832

/-- The theorem stating the ratio of the volumes -/
theorem inner_to_original_volume_ratio (t : InnerTetrahedron) :
  volume_inner t / volume_original t = 1 / 243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_to_original_volume_ratio_l736_73698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_a_zero_f_solutions_l736_73604

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2

theorem f_even_iff_a_zero (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 0 := by sorry

theorem f_solutions (a : ℝ) :
  f a (π / 4) = Real.sqrt 3 + 1 →
  {x : ℝ | x ∈ Set.Icc (-π) π ∧ f a x = 1 - Real.sqrt 2} =
  {-11 * π / 24, -5 * π / 24, 13 * π / 24, 19 * π / 24} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_iff_a_zero_f_solutions_l736_73604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_l736_73633

theorem cube_root_of_eight : (8 : ℝ) ^ (1/3) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_eight_l736_73633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_B_length_l736_73667

/-- Calculates the length of a train given the parameters of two trains passing each other. -/
noncomputable def calculate_train_length (length_A : ℝ) (speed_A : ℝ) (speed_B : ℝ) (time : ℝ) : ℝ :=
  let relative_speed := (speed_A + speed_B) * 1000 / 3600
  relative_speed * time - length_A

/-- Theorem stating the length of Train B given the specified conditions. -/
theorem train_B_length :
  let length_A : ℝ := 150
  let speed_A : ℝ := 120
  let speed_B : ℝ := 80
  let time : ℝ := 9
  ∃ ε > 0, |calculate_train_length length_A speed_A speed_B time - 349.95| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_B_length_l736_73667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l736_73638

def U : Set ℤ := {x | -2 < x ∧ x < 4 ∧ x ∈ Set.Icc 0 3}

def A : Set ℤ := {0, 2}

theorem complement_of_A : (Aᶜ : Set ℤ) = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l736_73638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l736_73622

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

-- Define the interval (-1,1)
def interval : Set ℝ := Set.Ioo (-1 : ℝ) 1

-- Theorem statement
theorem f_properties :
  -- f is odd
  (∀ x, x ∈ interval → f (-x) = -f x) ∧
  -- f is increasing on (-1,1)
  (∀ x y, x ∈ interval → y ∈ interval → x < y → f x < f y) ∧
  -- Solution set of f(2t-1) + f(t) < 0 is (0, 1/3)
  (Set.Ioo 0 (1/3 : ℝ) = {t : ℝ | f (2*t - 1) + f t < 0}) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l736_73622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_hop_probability_l736_73687

/-- Represents a position on the 4x4 grid -/
structure Position where
  x : Fin 4
  y : Fin 4

/-- Defines the grid -/
def Grid := Set Position

/-- Defines edge positions -/
def isEdge (p : Position) : Prop :=
  p.x = 0 ∨ p.x = 3 ∨ p.y = 0 ∨ p.y = 3

/-- Defines valid moves -/
def isValidMove (start finish : Position) : Prop :=
  (start.x = finish.x ∧ ((start.y + 1 : Fin 4) = finish.y ∨ start.y = (finish.y + 1 : Fin 4))) ∨
  (start.y = finish.y ∧ ((start.x + 1 : Fin 4) = finish.x ∨ start.x = (finish.x + 1 : Fin 4)))

/-- Defines the probability of reaching an edge within n hops -/
noncomputable def probReachEdge (start : Position) (n : Nat) : ℚ :=
  sorry

/-- The main theorem -/
theorem frog_hop_probability :
  probReachEdge ⟨1, 0⟩ 4 = 117/128 := by
  sorry

#eval "Frog hop probability theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frog_hop_probability_l736_73687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_relation_l736_73611

/-- Given two squares A and B, where A has a perimeter of 40 cm and B's area is
    three-quarters of A's area, prove that B's perimeter is 20√3 cm. -/
theorem square_perimeter_relation : 
  ∀ (side_a side_b : ℝ),
  side_a > 0 →
  side_b > 0 →
  4 * side_a = 40 →
  side_b^2 = 3/4 * side_a^2 →
  4 * side_b = 20 * Real.sqrt 3 :=
by
  intros side_a side_b h_pos_a h_pos_b h_perim_a h_area_b
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_relation_l736_73611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_radii_congruent_circle_definition_l736_73603

-- Define a circle in a plane
structure Circle (P : Type*) [MetricSpace P] where
  center : P
  radius : ℝ

-- Define congruence for circles
def CircleCongruent {P : Type*} [MetricSpace P] (c1 c2 : Circle P) : Prop :=
  c1.radius = c2.radius

-- Define a point being on a circle
def OnCircle {P : Type*} [MetricSpace P] (c : Circle P) (p : P) : Prop :=
  dist p c.center = c.radius

-- Theorem 1: Two circles with equal radii are congruent
theorem equal_radii_congruent {P : Type*} [MetricSpace P] (c1 c2 : Circle P) :
  c1.radius = c2.radius → CircleCongruent c1 c2 := by
  intro h
  exact h

-- Theorem 2: A circle is a set of points equidistant from a fixed point
theorem circle_definition {P : Type*} [MetricSpace P] (c : Circle P) (p : P) :
  OnCircle c p ↔ dist p c.center = c.radius := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_radii_congruent_circle_definition_l736_73603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_simplification_l736_73647

theorem trigonometric_expression_simplification (α : ℝ) :
  (4 * Real.sin (α - 5*Real.pi)^2 - Real.sin (2*α + Real.pi)^2) /
  (Real.cos (2*α - (3/2)*Real.pi)^2 - 4 + 4*Real.sin α^2) = -Real.tan α^4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_simplification_l736_73647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l736_73652

/-- Represents the set of ball numbers -/
def BallNumbers : Finset ℕ := {1, 2, 3, 4}

/-- The probability of drawing two balls with a sum no greater than 4 -/
def prob_sum_le_4 : ℚ :=
  (Finset.filter (fun (x, y) => x + y ≤ 4) (Finset.product BallNumbers BallNumbers)).card /
  (Finset.product BallNumbers BallNumbers).card

/-- The probability of drawing two balls (with replacement) such that the second number
    is less than the first number plus 2 -/
def prob_second_lt_first_plus_2 : ℚ :=
  (Finset.filter (fun (x, y) => y < x + 2) (Finset.product BallNumbers BallNumbers)).card /
  (Finset.product BallNumbers BallNumbers).card

theorem probability_theorem :
  prob_sum_le_4 = 1/3 ∧ prob_second_lt_first_plus_2 = 13/16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_theorem_l736_73652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_discriminants_l736_73664

/-- A monic quadratic polynomial with distinct roots -/
structure MonicQuadraticPolynomial where
  p : ℝ
  r : ℝ
  distinct_roots : p^2 - 4*r > 0

/-- The discriminant of a monic quadratic polynomial -/
def discriminant (f : MonicQuadraticPolynomial) : ℝ :=
  f.p^2 - 4*f.r

/-- Evaluate a monic quadratic polynomial at a given point -/
def eval (f : MonicQuadraticPolynomial) (x : ℝ) : ℝ :=
  x^2 + f.p*x + f.r

/-- The roots of a monic quadratic polynomial -/
noncomputable def roots (f : MonicQuadraticPolynomial) : ℝ × ℝ :=
  let sqrtDisc := Real.sqrt (f.p^2 - 4*f.r)
  ((-f.p + sqrtDisc) / 2, (-f.p - sqrtDisc) / 2)

theorem equal_discriminants (P Q : MonicQuadraticPolynomial) :
  let (a₁, a₂) := roots P
  let (b₁, b₂) := roots Q
  eval Q a₁ + eval Q a₂ = eval P b₁ + eval P b₂ →
  discriminant P = discriminant Q :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_discriminants_l736_73664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_with_two_rational_points_l736_73682

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in the Cartesian plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point is rational if both its coordinates are rational -/
def Point.isRational (p : Point) : Prop :=
  ∃ (qx qy : ℚ), p.x = qx ∧ p.y = qy

/-- A line contains a point if the point satisfies the line equation -/
def Line.contains (l : Line) (p : Point) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- A line contains at least two rational points -/
def Line.containsTwoRationalPoints (l : Line) : Prop :=
  ∃ p q : Point, p ≠ q ∧ p.isRational ∧ q.isRational ∧ l.contains p ∧ l.contains q

/-- The theorem statement -/
theorem unique_line_with_two_rational_points (a : ℝ) (h : ¬ ∃ (q : ℚ), a = q) :
  ∃! l : Line, l.contains ⟨a, 0⟩ ∧ l.containsTwoRationalPoints :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_with_two_rational_points_l736_73682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_modulus_m_l736_73699

/-- The quadratic equation with complex coefficients -/
def quadratic_equation (m : ℂ) (x : ℝ) : ℂ :=
  (4 + 3*Complex.I) * x^2 + m * x + (4 - 3*Complex.I)

/-- The condition that the equation has real roots -/
def has_real_roots (m : ℂ) : Prop :=
  ∃ x : ℝ, quadratic_equation m x = 0

/-- The theorem stating the minimum modulus of m -/
theorem min_modulus_m :
  ∃ min : ℝ, min = 8 ∧
  (∀ m : ℂ, has_real_roots m → Complex.abs m ≥ min) ∧
  (∃ m : ℂ, has_real_roots m ∧ Complex.abs m = min) := by
  sorry

#check min_modulus_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_modulus_m_l736_73699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tap_fill_time_l736_73643

/-- The time it takes for the first tap to fill the cistern -/
noncomputable def T : ℝ := sorry

/-- The rate at which the second tap empties the cistern -/
noncomputable def empty_rate : ℝ := 1 / 6

/-- The rate at which both taps together fill the cistern -/
noncomputable def combined_rate : ℝ := 1 / 6

/-- Theorem stating that T equals 3 hours -/
theorem first_tap_fill_time : T = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_tap_fill_time_l736_73643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_eq_neg_one_l736_73671

theorem sin_two_alpha_eq_neg_one (α : ℝ) (h : Real.sin α - Real.cos α = Real.sqrt 2) : 
  Real.sin (2 * α) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_two_alpha_eq_neg_one_l736_73671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_valid_partition_l736_73616

def T (m : ℕ) : Set ℕ := {n : ℕ | 4 ≤ n ∧ n ≤ m}

def hasTrioSumOfSquares (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a^2 + b^2 = c

def validPartition (m : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = T m → A ∩ B = ∅ →
    hasTrioSumOfSquares A ∨ hasTrioSumOfSquares B

theorem smallest_m_for_valid_partition :
  ∀ m : ℕ, m ≥ 4 →
    (∀ k : ℕ, 4 ≤ k ∧ k < 625 → ¬validPartition k) ∧
    validPartition 625 :=
by
  sorry

#check smallest_m_for_valid_partition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_for_valid_partition_l736_73616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l736_73662

/-- The area of a parallelogram with base 26 cm and height 16 cm is 416 square centimeters. -/
theorem parallelogram_area (base height : Real)
  (h1 : base = 26)
  (h2 : height = 16) :
  base * height = 416 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_l736_73662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_population_is_500_l736_73608

/-- Represents the population growth in a village over two years -/
structure VillagePopulation where
  initial : ℕ
  first_year_growth : ℕ
  second_year_growth : ℕ

/-- The conditions of the population growth problem -/
def population_conditions (v : VillagePopulation) : Prop :=
  v.first_year_growth = v.initial * 3 ∧
  v.second_year_growth = 300 ∧
  v.first_year_growth * v.first_year_growth = 30000

/-- The final population after two years of growth -/
def final_population (v : VillagePopulation) : ℕ :=
  v.initial + v.first_year_growth + v.second_year_growth

/-- Theorem stating that the final population is 500 given the conditions -/
theorem final_population_is_500 (v : VillagePopulation) :
  population_conditions v → final_population v = 500 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_population_is_500_l736_73608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_through_point_one_two_l736_73631

-- Define an inverse proportion function
noncomputable def inverse_proportion (k : ℝ) : ℝ → ℝ := fun x => k / x

-- State the theorem
theorem inverse_proportion_through_point_one_two :
  ∃ f : ℝ → ℝ, (∀ x ≠ 0, ∃ k : ℝ, f x = k / x) ∧ f 1 = 2 → 
  ∀ x ≠ 0, f x = 2 / x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_through_point_one_two_l736_73631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_when_a_is_one_range_of_a_for_inequality_l736_73613

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x + a

-- Part I
theorem monotonicity_of_f_when_a_is_one :
  (∀ x y : ℝ, x < y → x < 0 → y < 0 → f 1 y < f 1 x) ∧
  (∀ x y : ℝ, x < y → 0 < x → 0 < y → f 1 x < f 1 y) := by sorry

-- Part II
theorem range_of_a_for_inequality :
  ∀ a : ℝ, a > 0 →
  (∀ x : ℝ, x > 0 → f a x ≥ a * Real.log x) →
  a ≤ Real.exp 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_when_a_is_one_range_of_a_for_inequality_l736_73613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l736_73681

-- Define the curve C
noncomputable def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y = 0

-- Define the line passing through (1, 2)
noncomputable def line (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, 2 + (1 / 2) * t)

-- Define the distance function
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

-- Theorem statement
theorem intersection_distance_product : 
  ∃ t1 t2 : ℝ, 
    t1 ≠ t2 ∧
    curve_C ((line t1).1) ((line t1).2) ∧
    curve_C ((line t2).1) ((line t2).2) ∧
    (distance 1 2 ((line t1).1) ((line t1).2)) * 
    (distance 1 2 ((line t2).1) ((line t2).2)) = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_product_l736_73681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l736_73626

noncomputable def y (x : Real) : Real :=
  Real.tan (x + 3 * Real.pi / 4) - Real.tan (x + Real.pi / 3) + 
  Real.cos (x + Real.pi / 3) + Real.sin (x + Real.pi / 4)

theorem max_value_of_y :
  ∃ (max : Real), 
    (∀ x, -5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 4 → y x ≤ max) ∧
    (∃ x, -5 * Real.pi / 12 ≤ x ∧ x ≤ -Real.pi / 4 ∧ y x = max) ∧
    max = 3 / 2 - Real.sqrt 3 - Real.sqrt 2 / 2 :=
by
  sorry

#check max_value_of_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_y_l736_73626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_sector_l736_73621

theorem cone_height_from_sector (r : ℝ) (θ : ℝ) (h : ℝ) : 
  r = 5 →
  θ = 144 * π / 180 →
  r * θ = 2 * π * (r * Real.sin (θ / 2)) →
  h^2 + (r * Real.sin (θ / 2))^2 = r^2 →
  h = Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_sector_l736_73621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_l736_73636

theorem perfect_square_sum (m : ℕ) : ∃ k : ℕ, 
  (10^(2*m) - 1) / 9 + 4 * ((10^m - 1) / 9) + 1 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_sum_l736_73636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greg_distance_to_market_l736_73686

/-- Represents the travel details of Greg's journey --/
structure TravelDetails where
  total_distance : ℚ
  market_to_home_time : ℚ
  market_to_home_speed : ℚ

/-- Calculates the distance from workplace to farmer's market --/
def distance_to_market (t : TravelDetails) : ℚ :=
  t.total_distance - (t.market_to_home_time / 60) * t.market_to_home_speed

/-- Theorem stating that Greg's distance from workplace to farmer's market is 30 miles --/
theorem greg_distance_to_market :
  ∀ t : TravelDetails,
  t.total_distance = 40 ∧
  t.market_to_home_time = 30 ∧
  t.market_to_home_speed = 20 →
  distance_to_market t = 30 := by
  intro t h
  simp [distance_to_market]
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greg_distance_to_market_l736_73686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_four_l736_73639

/-- The infinite product of terms where the nth term is (2^n)^(1/2^n) equals 4 -/
theorem infinite_product_equals_four :
  (∏' n : ℕ, (2^n : ℝ)^(1/(2^n : ℝ))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_equals_four_l736_73639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_relation_find_line_parameters_l736_73602

open Real

noncomputable section

-- Define the curve E and points A, B, C
def curve_E (θ : ℝ) : ℝ := 4 * sin θ

def point_A (β : ℝ) : ℝ × ℝ :=
  (curve_E β * cos β, curve_E β * sin β)

def point_B (β : ℝ) : ℝ × ℝ :=
  (curve_E (β + π/6) * cos (β + π/6), curve_E (β + π/6) * sin (β + π/6))

def point_C (β : ℝ) : ℝ × ℝ :=
  (curve_E (β - π/6) * cos (β - π/6), curve_E (β - π/6) * sin (β - π/6))

-- Define the distance function
def distance (p : ℝ × ℝ) : ℝ :=
  sqrt (p.1^2 + p.2^2)

-- Theorem 1
theorem distance_relation (β : ℝ) :
  distance (point_B β) + distance (point_C β) = sqrt 3 * distance (point_A β) := by
  sorry

-- Theorem 2
theorem find_line_parameters :
  ∃ (y₀ α : ℝ), 
    let B := point_B (π/3)
    let C := point_C (π/3)
    B.2 = y₀ + B.1 * tan α ∧
    C.2 = y₀ + C.1 * tan α ∧
    y₀ = 4 ∧
    α = 2*π/3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_relation_find_line_parameters_l736_73602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l736_73642

def a : Fin 3 → ℝ := ![-2, -3, 1]
def b : Fin 3 → ℝ := ![2, 0, 4]
def c : Fin 3 → ℝ := ![4, 6, -2]

def parallel (v w : Fin 3 → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ i, v i = k * w i

def perpendicular (v w : Fin 3 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2) = 0

theorem vector_relations : parallel a c ∧ perpendicular a b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_relations_l736_73642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_eq_2_l736_73685

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then 3 * x + 1 else -x

theorem unique_solution_f_eq_2 : ∃! x, f x = 2 ∧ x = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_f_eq_2_l736_73685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l736_73660

/-- IsTriangle a b c means a, b, c satisfy the triangle inequality. -/
def IsTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a < b + c ∧ b < a + c ∧ c < a + b

/-- A structure representing a triangle with side lengths a, b, and c. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_triangle : IsTriangle a b c

theorem triangle_inequality (a b c : ℝ) (h : IsTriangle a b c) : a^2 - b^2 - c^2 - 2*b*c < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l736_73660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_result_l736_73629

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x / 6 + Real.pi / 3)

theorem dot_product_result (A B C : ℝ × ℝ) :
  (2 < A.1 ∧ A.1 < 10) →  -- Domain condition for A
  f A.1 = 0 →  -- A is on x-axis
  A.2 = 0 →  -- A is on x-axis
  ∃ (m b : ℝ), B.2 = m * B.1 + b ∧ C.2 = m * C.1 + b ∧ A.2 = m * A.1 + b →  -- B, C, and A are collinear
  f B.1 = B.2 →  -- B is on the graph of f
  f C.1 = C.2 →  -- C is on the graph of f
  (B.1 + C.1, B.2 + C.2) • A = 32 := by
  sorry

#check dot_product_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_result_l736_73629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_cost_prices_l736_73600

/-- Represents the selling price and profit/loss percentage of an article -/
structure Article where
  selling_price : ℚ
  profit_percentage : ℚ

/-- Calculates the cost price of an article given its selling price and profit/loss percentage -/
noncomputable def cost_price (article : Article) : ℚ :=
  article.selling_price / (1 + article.profit_percentage)

theorem shopkeeper_cost_prices 
  (article1 : Article)
  (article2 : Article)
  (article3 : Article)
  (h1 : article1.selling_price = 1110)
  (h2 : article1.profit_percentage = 1/5)
  (h3 : article2.selling_price = 1575)
  (h4 : article2.profit_percentage = -3/20)
  (h5 : article3.selling_price = 2040)
  (h6 : article3.profit_percentage = 0) :
  cost_price article1 = 925 ∧ 
  cost_price article2 = 1852.94 ∧ 
  cost_price article3 = 2040 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_cost_prices_l736_73600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cut_area_l736_73689

/-- The area of a flat surface created by cutting a right circular cylinder --/
theorem cylinder_cut_area (r h angle : ℝ) : 
  r = 8 → h = 10 → angle = 150 * π / 180 →
  (angle / (2 * π) * π * r^2) - (1 / 2 * r^2 * Real.sin angle) = 1456 / 9 * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_cut_area_l736_73689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_is_intersection_of_perpendicular_bisectors_l736_73623

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define a line
structure Line where
  p1 : Point
  p2 : Point

-- Define perpendicular bisector
noncomputable def perpendicular_bisector (l : Line) : Line :=
  sorry

-- Define intersection point
noncomputable def intersection_point (l1 l2 : Line) : Point :=
  sorry

-- Define circumcenter
noncomputable def circumcenter (t : Triangle) : Point :=
  sorry

-- Theorem statement
theorem circumcenter_is_intersection_of_perpendicular_bisectors (t : Triangle) :
  circumcenter t = intersection_point 
    (perpendicular_bisector ⟨t.A, t.B⟩)
    (perpendicular_bisector ⟨t.B, t.C⟩) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_is_intersection_of_perpendicular_bisectors_l736_73623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l736_73624

noncomputable def f (x : ℝ) : ℝ := (5*x - 20) / (4*x - 5)

theorem smallest_x_value :
  ∃ (x : ℝ), (f x)^2 + f x = 6 ∧
  ∀ (y : ℝ), (f y)^2 + f y = 6 → x ≤ y :=
by
  -- We know the smallest value is -10/3
  let x := -10/3
  
  -- Prove that f(x)^2 + f(x) = 6
  have h1 : (f x)^2 + f x = 6 := by
    -- This step would require actual computation
    sorry
  
  -- Prove that for all y, if f(y)^2 + f(y) = 6, then x ≤ y
  have h2 : ∀ (y : ℝ), (f y)^2 + f y = 6 → x ≤ y := by
    -- This step would require more detailed proof
    sorry
  
  -- Combine the two parts
  exact ⟨x, h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_value_l736_73624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alberto_bjorn_bike_difference_eq_22_specific_bike_difference_l736_73672

/-- The difference in miles biked between Alberto and Bjorn -/
def alberto_bjorn_bike_difference (alberto_hours : ℝ) (alberto_speed : ℝ) 
                                  (bjorn_hours : ℝ) (bjorn_speed : ℝ) : ℝ :=
  alberto_hours * alberto_speed - bjorn_hours * bjorn_speed

/-- Theorem stating the difference in miles biked between Alberto and Bjorn -/
theorem alberto_bjorn_bike_difference_eq_22 
  (alberto_hours : ℝ) (alberto_speed : ℝ) (bjorn_hours : ℝ) (bjorn_speed : ℝ) :
  alberto_bjorn_bike_difference alberto_hours alberto_speed bjorn_hours bjorn_speed = 22 → 
  alberto_hours * alberto_speed - bjorn_hours * bjorn_speed = 22 := by
  intro h
  exact h

/-- Proof of the specific problem instance -/
theorem specific_bike_difference : 
  alberto_bjorn_bike_difference 6 12 5 10 = 22 := by
  unfold alberto_bjorn_bike_difference
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alberto_bjorn_bike_difference_eq_22_specific_bike_difference_l736_73672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_seating_l736_73697

def number_of_unique_seating_arrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

theorem round_table_seating (n : ℕ) (h : n = 10) : 
  number_of_unique_seating_arrangements n = 362880 := by
  rw [number_of_unique_seating_arrangements]
  rw [h]
  simp
  rfl

#eval Nat.factorial 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_table_seating_l736_73697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_satisfying_condition_l736_73656

def D (n : ℕ+) : Set ℕ+ := {d | d ∣ n}

def F (n : ℕ+) (i : Fin 4) : Set ℕ+ := {a ∈ D n | a % 4 = i}

def f (n : ℕ+) (i : Fin 4) : ℕ := 
  Finset.card (Finset.filter (λ a => a % 4 = i.val) (Finset.filter (λ x => x ∣ n) (Finset.range n.val)))

def m : ℕ+ := ⟨2^34 * 3^6 * 7^2 * 11^2, by norm_num⟩

theorem smallest_m_satisfying_condition :
  (∀ k : ℕ+, k < m → f k 0 + f k 1 - f k 2 - f k 3 ≠ 2017) ∧
  f m 0 + f m 1 - f m 2 - f m 3 = 2017 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_satisfying_condition_l736_73656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_sum_l736_73612

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem about the sum of squared distances from a point on a hyperbola to its foci -/
theorem hyperbola_foci_distance_sum (h : Hyperbola) (f₁ f₂ p : Point) : 
  (h.a^2 - h.b^2 = 3) →  -- Equation of the hyperbola
  (distance f₁ p - distance f₂ p)^2 = 4 * h.a^2 →  -- Definition of hyperbola
  ∃ θ : ℝ, θ = 120 * π / 180 →  -- Angle in radians
  (distance f₁ p)^2 + (distance f₂ p)^2 - 2 * distance f₁ p * distance f₂ p * Real.cos θ = 4 * h.c^2 →  -- Cosine rule
  (distance f₁ p)^2 + (distance f₂ p)^2 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_sum_l736_73612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amina_reaches_one_amina_iterations_amina_iterations_count_l736_73655

def amina_sequence (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => (amina_sequence n k) / 2

theorem amina_reaches_one : ∃ k, amina_sequence 64 k = 1 := by
  sorry

theorem amina_iterations : ∃! k, amina_sequence 64 k = 1 ∧ ∀ j < k, amina_sequence 64 j > 1 := by
  sorry

def iterations_count : ℕ := 6

theorem amina_iterations_count : ∃ k, amina_sequence 64 k = 1 ∧ ∀ j < k, amina_sequence 64 j > 1 ∧ k = iterations_count := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amina_reaches_one_amina_iterations_amina_iterations_count_l736_73655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_purchase_optimization_l736_73635

/-- Represents the total cost function for TV purchases -/
noncomputable def total_cost (k : ℝ) (x : ℝ) : ℝ :=
  (3600 / x) * 400 + k * 2000 * x

theorem tv_purchase_optimization :
  ∃ (k : ℝ) (x : ℝ),
    k > 0 ∧
    x > 0 ∧
    total_cost k 400 = 43600 ∧
    (∀ y, y > 0 → total_cost k y ≥ 24000) ∧
    total_cost k x = 24000 ∧
    x = 120 := by
  sorry

#check tv_purchase_optimization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_purchase_optimization_l736_73635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_satisfies_conditions_total_questions_is_30_l736_73674

/-- Represents a school test with multiple sections -/
structure SchoolTest where
  num_sections : ℕ
  questions_per_section : ℕ
  correct_answers : ℕ

/-- Defines the conditions of the given test -/
def given_test : SchoolTest :=
  { num_sections := 5,
    questions_per_section := 6,  -- This is derived from the total questions (30) divided by sections (5)
    correct_answers := 20 }

/-- Theorem stating that the given test satisfies all conditions -/
theorem test_satisfies_conditions (test : SchoolTest := given_test) :
  test.num_sections = 5 ∧
  test.num_sections * test.questions_per_section = 30 ∧
  test.correct_answers = 20 ∧
  (60 : ℚ) / 100 < (test.correct_answers : ℚ) / ((test.num_sections * test.questions_per_section) : ℚ) ∧
  (test.correct_answers : ℚ) / ((test.num_sections * test.questions_per_section) : ℚ) < (70 : ℚ) / 100 :=
by
  -- Proof steps would go here
  sorry

/-- Main theorem proving that the total number of questions is 30 -/
theorem total_questions_is_30 (test : SchoolTest := given_test) : 
  test.num_sections * test.questions_per_section = 30 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_satisfies_conditions_total_questions_is_30_l736_73674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l736_73694

/-- The line equation x + y = 0 -/
def line (x y : ℝ) : Prop := x + y = 0

/-- The circle equation (x - a)² + (y - b)² = 2 -/
def circle_eq (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 2

/-- The line is tangent to the circle if they intersect at exactly one point -/
def is_tangent (a b : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, line p.1 p.2 ∧ circle_eq p.1 p.2 a b

/-- The condition "a = 1 and b = 1" is sufficient but not necessary for tangency -/
theorem tangent_condition :
  (is_tangent 1 1) ∧ (∃ a b : ℝ, a ≠ 1 ∨ b ≠ 1) ∧ (is_tangent a b) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_condition_l736_73694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l736_73610

-- Define the circles
def circle_C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def circle_C2 (x y : ℝ) : Prop := (x + 2)^2 + (y - 5)^2 = 36

-- Define the centers of the circles
def center_C1 : ℝ × ℝ := (1, 1)
def center_C2 : ℝ × ℝ := (-2, 5)

-- Define the radii of the circles
def radius_C1 : ℝ := 1
def radius_C2 : ℝ := 6

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := 
  Real.sqrt ((center_C2.1 - center_C1.1)^2 + (center_C2.2 - center_C1.2)^2)

-- Theorem: The circles are internally tangent
theorem circles_internally_tangent : 
  distance_between_centers = radius_C2 - radius_C1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_internally_tangent_l736_73610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equality_cosine_value_in_second_quadrant_tangent_fraction_equality_l736_73677

-- Part I
theorem logarithm_expression_equality : 
  2 * Real.log 5 / Real.log 10 + Real.log 4 / Real.log 10 + Real.log (Real.sqrt (Real.exp 1)) = 2.5 := by sorry

-- Part II
theorem cosine_value_in_second_quadrant (α : Real) (h1 : π/2 < α ∧ α < π) (h2 : Real.sin α = 1/3) : 
  Real.cos α = -2*Real.sqrt 2/3 := by sorry

-- Part III
theorem tangent_fraction_equality (α : Real) (h : Real.tan α = 2) : 
  (4 * Real.cos α + Real.sin α) / (3 * Real.cos α - 2 * Real.sin α) = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithm_expression_equality_cosine_value_in_second_quadrant_tangent_fraction_equality_l736_73677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_f_min_max_on_interval_l736_73695

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x * Real.cos x + Real.sqrt 3 * Real.cos (2 * x) + 2

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def monotone_intervals (k : ℤ) : Set ℝ :=
  Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12)

theorem f_monotone_intervals :
  ∀ k : ℤ, is_monotone_increasing f (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12) :=
sorry

theorem f_min_max_on_interval :
  let a := -Real.pi / 3
  let b := Real.pi / 3
  (∀ x ∈ Set.Icc a b, f x ≥ 2 - Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 4) ∧
  (∃ x ∈ Set.Icc a b, f x = 2 - Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc a b, f x = 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_intervals_f_min_max_on_interval_l736_73695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_ratio_is_two_to_one_l736_73605

/-- Represents the ratio of a man's total expenditure over two years to his first year expenditure -/
noncomputable def expenditure_ratio (first_year_income : ℝ) : ℝ :=
  let first_year_savings := 0.5 * first_year_income
  let first_year_expenditure := first_year_income - first_year_savings
  let second_year_income := 1.5 * first_year_income
  let second_year_savings := 2 * first_year_savings
  let second_year_expenditure := second_year_income - second_year_savings
  let total_expenditure := first_year_expenditure + second_year_expenditure
  total_expenditure / first_year_expenditure

/-- Theorem stating that the expenditure ratio is 2:1 -/
theorem expenditure_ratio_is_two_to_one (first_year_income : ℝ) (h : first_year_income > 0) :
  expenditure_ratio first_year_income = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expenditure_ratio_is_two_to_one_l736_73605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_in_range_l736_73614

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (4 - a/2)*x + 2

theorem monotonic_f_implies_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  4 ≤ a ∧ a < 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_f_implies_a_in_range_l736_73614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l736_73670

theorem sin_2x_value (x : ℝ) (h : Real.cos (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = -7 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l736_73670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_equations_l736_73658

/-- Triangle ABC with vertices A(1, -4), B(6, 6), and C(-2, 0) -/
structure Triangle where
  A : ℝ × ℝ := (1, -4)
  B : ℝ × ℝ := (6, 6)
  C : ℝ × ℝ := (-2, 0)

/-- The equation of a line in general form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about the equations of specific lines in the triangle -/
theorem triangle_line_equations (t : Triangle) :
  ∃ (l1 l2 : LineEquation),
    -- Line parallel to BC passing through midpoint of AB
    (l1.a = 6 ∧ l1.b = -8 ∧ l1.c = -13) ∧
    -- Median line on side BC
    (l2.a = 7 ∧ l2.b = -1 ∧ l2.c = -11) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_line_equations_l736_73658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_five_l736_73665

theorem divisors_multiple_of_five (n : ℕ) (h : n = 5400) :
  (Finset.filter (λ d ↦ d ∣ n ∧ 5 ∣ d) (Finset.range (n + 1))).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_multiple_of_five_l736_73665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_weekend_hours_l736_73661

/-- Represents the baker's bread production and schedule --/
structure BakerProduction where
  loaves_per_hour_per_oven : ℕ := 5
  num_ovens : ℕ := 4
  weekday_hours : ℕ := 5
  weekday_count : ℕ := 5
  total_weeks : ℕ := 3
  total_loaves : ℕ := 1740

/-- Calculates the number of hours the baker works on weekends --/
def weekend_hours (bp : BakerProduction) : ℚ :=
  let total_weekday_loaves := bp.loaves_per_hour_per_oven * bp.num_ovens * bp.weekday_hours * bp.weekday_count * bp.total_weeks
  let weekend_loaves := bp.total_loaves - total_weekday_loaves
  let loaves_per_hour := bp.loaves_per_hour_per_oven * bp.num_ovens
  (weekend_loaves : ℚ) / (loaves_per_hour : ℚ) / (2 * bp.total_weeks : ℚ)

/-- Theorem stating that the baker works 4 hours on weekends --/
theorem baker_weekend_hours :
  weekend_hours {
    loaves_per_hour_per_oven := 5,
    num_ovens := 4,
    weekday_hours := 5,
    weekday_count := 5,
    total_weeks := 3,
    total_loaves := 1740
  } = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_baker_weekend_hours_l736_73661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_iff_a_in_interval_l736_73640

/-- A cubic function parameterized by a real number a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + x - 5

/-- The derivative of f with respect to x -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

/-- Theorem stating that f has no extreme points iff a is in [-1, 1] -/
theorem no_extreme_points_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, f_derivative a x ≠ 0) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_extreme_points_iff_a_in_interval_l736_73640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_minus_3d_equals_12i_l736_73620

def c : ℂ := 6 - 3*Complex.I
def d : ℂ := 2 - 5*Complex.I

theorem c_minus_3d_equals_12i : c - 3*d = 12*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_minus_3d_equals_12i_l736_73620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l736_73618

theorem tan_value_from_sin_cos_sum (θ : Real) 
  (h1 : Real.sin θ + Real.cos θ = 1/5)
  (h2 : θ ∈ Set.Ioo 0 Real.pi) :
  Real.tan θ = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_from_sin_cos_sum_l736_73618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_is_seven_l736_73646

def is_valid_split (s : Finset ℕ) : Prop :=
  s ⊆ Finset.range 10 ∧ s.card > 0 ∧ s.card < 10

def P₁ (s : Finset ℕ) : ℕ := s.prod (λ x ↦ x + 1)

def P₂ (s : Finset ℕ) : ℕ := (Finset.range 10 \ s).prod (λ x ↦ x + 1)

theorem min_ratio_is_seven :
  ∃ (s : Finset ℕ), is_valid_split s ∧ P₁ s % P₂ s = 0 ∧
  ∀ (t : Finset ℕ), is_valid_split t → P₁ t % P₂ t = 0 →
  (P₁ s : ℚ) / P₂ s ≤ (P₁ t : ℚ) / P₂ t ∧
  (P₁ s : ℚ) / P₂ s = 7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ratio_is_seven_l736_73646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_county_count_l736_73679

/-- Represents the types of inhabitants in a county -/
inductive Inhabitant
  | Elf
  | Dwarf
  | Centaur

/-- Represents a county with its inhabitant type -/
structure County where
  inhabitant : Inhabitant

/-- The state of the island at any given time -/
structure IslandState where
  counties : List County

/-- Applies the splitting rule for a given year -/
def applySplitRule (state : IslandState) (year : Nat) : IslandState :=
  match year with
  | 1 => { counties := state.counties.bind (fun c => 
      match c.inhabitant with
      | Inhabitant.Elf => [c]
      | _ => [c, c, c]) }
  | 2 => { counties := state.counties.bind (fun c =>
      match c.inhabitant with
      | Inhabitant.Dwarf => [c]
      | _ => [c, c, c, c]) }
  | 3 => { counties := state.counties.bind (fun c =>
      match c.inhabitant with
      | Inhabitant.Centaur => [c]
      | _ => [c, c, c, c, c, c]) }
  | _ => state

/-- The initial state of the island -/
def initialState : IslandState :=
  { counties := [
    { inhabitant := Inhabitant.Elf },
    { inhabitant := Inhabitant.Dwarf },
    { inhabitant := Inhabitant.Centaur }
  ] }

/-- The theorem stating the final number of counties -/
theorem final_county_count :
  (applySplitRule (applySplitRule (applySplitRule initialState 1) 2) 3).counties.length = 54 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_county_count_l736_73679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_volume_maximum_l736_73648

/-- Traffic speed as a function of traffic density -/
noncomputable def v (x : ℝ) : ℝ :=
  if x ≤ 30 then 60
  else if x ≤ 210 then -1/3 * x + 70
  else 0

/-- Traffic volume as a function of traffic density -/
noncomputable def f (x : ℝ) : ℝ := x * v x

theorem traffic_volume_maximum :
  ∃ (x_max : ℝ), x_max ∈ Set.Icc 0 210 ∧
  (∀ x, x ∈ Set.Icc 0 210 → f x ≤ f x_max) ∧
  x_max = 105 ∧ f x_max = 3675 := by
  sorry

#check traffic_volume_maximum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_traffic_volume_maximum_l736_73648
