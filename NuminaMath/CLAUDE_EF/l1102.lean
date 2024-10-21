import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_implies_m_equals_31_l1102_110201

theorem equation_implies_m_equals_31 
  (a b n p r m : ℕ) 
  (ha : a > 0) (hb : b > 0) (hn : n > 0) (hp : p > 0) (hr : r > 0) (hm : m > 0)
  (h : ((a ^ m) * (b ^ n) / ((5 : ℕ) ^ m) * ((7 : ℕ) ^ n)) * (1 / ((4 : ℕ) ^ p)) = 1 / (2 * ((10 : ℕ) * r) ^ 31)) : 
  m = 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_implies_m_equals_31_l1102_110201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1102_110290

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The line x - 2y + 10 = 0 -/
def line (x y : ℝ) : Prop := x - 2*y + 10 = 0

/-- Distance from a point to the directrix of the parabola y^2 = 4x -/
def distance_to_directrix (x : ℝ) : ℝ := x + 1

/-- Distance from a point to the line x - 2y + 10 = 0 -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - 2*y + 10| / Real.sqrt 5

/-- The theorem stating the minimum value of d1 + d2 -/
theorem min_distance_sum :
  ∃ (x y : ℝ), parabola x y ∧
  (∀ (x' y' : ℝ), parabola x' y' →
    distance_to_directrix x + distance_to_line x y ≤
    distance_to_directrix x' + distance_to_line x' y') ∧
  distance_to_directrix x + distance_to_line x y = 11 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1102_110290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_relation_l1102_110213

noncomputable def first_sequence : List ℚ := List.range 20
noncomputable def second_sequence : List ℚ := List.map (λ x => 3 * x + 1) first_sequence

noncomputable def average (l : List ℚ) : ℚ := (l.sum) / (l.length : ℚ)

theorem average_relation (a : ℚ) (h : average first_sequence = a) :
  average second_sequence = 3 * a + 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_relation_l1102_110213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_distance_proof_l1102_110248

/-- The distance between home and school in kilometers. -/
noncomputable def distance : ℝ := 2

/-- The time it takes to travel at 4 km/hr in hours. -/
noncomputable def normal_time : ℝ := 23 / 60

theorem school_distance_proof :
  /- First day: speed 4 km/hr, 7 minutes late -/
  distance = 4 * (normal_time + 7 / 60) ∧
  /- Second day: speed 8 km/hr, 8 minutes early -/
  distance = 8 * (normal_time - 8 / 60) :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_distance_proof_l1102_110248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_third_l1102_110281

/-- The sum of the infinite series ∑(n=1 to ∞) (n³ + 2n² - n + 1) / ((n+3)!) is equal to 1/3 -/
theorem series_sum_equals_one_third :
  (∑' n : ℕ, (n^3 + 2*n^2 - n + 1) / (Nat.factorial (n+3))) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_equals_one_third_l1102_110281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l1102_110221

-- Define an acute-angled triangle ABC
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < π/2
  acute_B : 0 < B ∧ B < π/2
  acute_C : 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

-- Define the theorem
theorem acute_triangle_properties (t : AcuteTriangle) (a b : Real) 
  (h : 2 * a * Real.sin t.B = b) : 
  t.A = π/6 ∧ 
  Set.Icc (Real.sqrt 3) 2 ⊆ {x | ∃ (B C : Real), 
    x = Real.sqrt 3 * Real.sin B - Real.cos (C + π/6) ∧
    0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2 ∧ 
    B + C = 5*π/6} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l1102_110221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1102_110294

theorem vector_equation_solution (e₁ e₂ : ℝ × ℝ) (x y : ℝ) 
  (h_non_collinear : ¬ ∃ (k : ℝ), e₁ = (k * e₂.1, k * e₂.2))
  (h_eq : ((3*x - 4*y) * e₁.1, (3*x - 4*y) * e₁.2) + ((2*x - 3*y) * e₂.1, (2*x - 3*y) * e₂.2) = (6 * e₁.1, 6 * e₁.2) + (3 * e₂.1, 3 * e₂.2)) :
  x + y = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1102_110294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_cosine_product_l1102_110275

open Real

theorem periodic_cosine_product (n : ℤ) : 
  (∀ x : ℝ, Real.cos ((n - 1 : ℤ) * x) * Real.cos (15 * x / (2 * ↑n + 1)) = 
            Real.cos ((n - 1 : ℤ) * (x + π)) * Real.cos (15 * (x + π) / (2 * ↑n + 1))) ↔ 
  n ∈ ({0, -2, 2, -8} : Set ℤ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_cosine_product_l1102_110275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l1102_110259

theorem quadratic_equation_solution (h : ℝ) : 
  (∃ r s : ℝ, (r - h)^2 + 4*h = 5 + r ∧ 
               (s - h)^2 + 4*h = 5 + s ∧ 
               r^2 + s^2 = 20) → 
  |h| = Real.sqrt 22 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solution_l1102_110259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l1102_110256

theorem divisibility_property (n : ℕ+) : ∃ (a b : ℤ), (n : ℤ) ∣ (4 * a ^ 2 + 9 * b ^ 2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_property_l1102_110256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l1102_110282

/-- Calculates the profit percentage given cost price, selling price, and discount rate. -/
noncomputable def profit_percentage (cost_price selling_price discount_rate : ℝ) : ℝ :=
  let list_price := selling_price / (1 - discount_rate)
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem: Given the specified cost price, selling price, and discount rate,
    the profit percentage is approximately 31.57%. -/
theorem profit_percentage_approx (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |profit_percentage 51.50 67.76 0.05 - 31.57| < δ ∧ δ < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_approx_l1102_110282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centerpiece_lilies_l1102_110289

/-- Centerpiece flower arrangement problem --/
theorem centerpiece_lilies (centerpieces : ℕ) (roses_per_centerpiece : ℕ) 
  (max_total_flowers : ℕ) (flower_ratio : Fin 3 → ℕ) :
  centerpieces = 6 →
  roses_per_centerpiece = 8 →
  max_total_flowers = 120 →
  (∀ i : Fin 3, flower_ratio i = i.val + 1) →
  (max_total_flowers / (flower_ratio 0 + flower_ratio 1 + flower_ratio 2) : ℚ) * 
    (flower_ratio 2 : ℚ) / (centerpieces : ℚ) = 10 := by
  sorry

#check centerpiece_lilies

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centerpiece_lilies_l1102_110289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_undefined_inverse_l1102_110278

theorem smallest_undefined_inverse : ∃ (a : ℕ), a > 0 ∧ 
  (∀ (b : ℕ), b > 0 ∧ b < a → (∃ (x : ℤ), x * b % 70 = 1 ∨ x * b % 91 = 1)) ∧
  (∀ (y : ℤ), y * a % 70 ≠ 1 ∧ y * a % 91 ≠ 1) :=
by sorry

#check smallest_undefined_inverse

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_undefined_inverse_l1102_110278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_coprime_sets_l1102_110265

/-- A function that checks if two numbers are coprime -/
def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

/-- The theorem stating the maximum product of set sizes -/
theorem max_product_coprime_sets :
  ∀ (A B : Finset ℕ),
  (∀ a, a ∈ A → 2 ≤ a ∧ a ≤ 20) →
  (∀ b, b ∈ B → 2 ≤ b ∧ b ≤ 20) →
  (∀ (a b : ℕ), a ∈ A → b ∈ B → are_coprime a b) →
  A.card * B.card ≤ 65 :=
by
  sorry

#check max_product_coprime_sets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_product_coprime_sets_l1102_110265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1102_110251

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (F₁ F₂ A : ℝ × ℝ) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ t)) →  -- Hyperbola equation
  (F₁.1 < 0 ∧ F₂.1 > 0) →  -- F₁ is left focus, F₂ is right focus
  A ∈ Set.range (λ t : ℝ × ℝ ↦ t) →  -- A is on the hyperbola
  (A.1 - F₁.1) * (A.1 - F₂.1) + (A.2 - F₁.2) * (A.2 - F₂.2) = 0 →  -- AF₁ ⋅ AF₂ = 0
  ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) = 9 * ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) →  -- |AF₁| = 3|AF₂|
  c^2 / a^2 = 5/4  -- Eccentricity squared is 5/4
  :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1102_110251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_for_A_l1102_110292

-- Define the marksmen
inductive Marksman
| A
| B
| C

-- Define the hit probability function
def hit_probability : Marksman → ℝ
| Marksman.A => 0.3
| Marksman.B => 1
| Marksman.C => 0.5

-- Define the possible targets
inductive Target
| marksman : Marksman → Target
| air : Target

-- Define the survival probability function
noncomputable def survival_probability : Marksman → Target → ℝ := sorry

-- Theorem statement
theorem optimal_strategy_for_A :
  ∀ t : Target, survival_probability Marksman.A Target.air ≥ survival_probability Marksman.A t :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_strategy_for_A_l1102_110292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_1_13_l1102_110202

/-- The decimal representation of 1/13 -/
def decimal_rep_1_13 : ℕ → ℕ
| n => [0, 7, 6, 9, 2, 3].get! (n % 6)

/-- The 150th digit after the decimal point in the decimal representation of 1/13 -/
theorem digit_150_of_1_13 : decimal_rep_1_13 149 = 3 := by
  rfl

#eval decimal_rep_1_13 149

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_150_of_1_13_l1102_110202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_binary_multiple_of_225_l1102_110228

/-- A number is representable in binary if it can be written using only digits 0 and 1 -/
def IsRepresentableInBinary (n : ℕ) : Prop :=
  ∃ (digits : List ℕ), n = digits.foldl (fun acc d ↦ 10 * acc + d) 0 ∧ ∀ d ∈ digits, d = 0 ∨ d = 1

/-- The smallest positive multiple of 225 that can be written using only digits 0 and 1 -/
def SmallestBinaryMultipleOf225 : ℕ := 11111111100

theorem smallest_binary_multiple_of_225 :
  SmallestBinaryMultipleOf225 % 225 = 0 ∧
  IsRepresentableInBinary SmallestBinaryMultipleOf225 ∧
  ∀ n : ℕ, n < SmallestBinaryMultipleOf225 → ¬(n % 225 = 0 ∧ IsRepresentableInBinary n) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_binary_multiple_of_225_l1102_110228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_k_l1102_110223

def is_k_bad (k N : ℕ) : Prop :=
  ¬ ∃ x y : ℕ, N = 2020 * x + k * y

def special_k (k : ℕ) : Prop :=
  k > 1 ∧
  Nat.gcd k 2020 = 1 ∧
  ∀ m n : ℕ, m > 0 → n > 0 → m + n = 2019 * (k - 1) → m ≥ n →
    is_k_bad k m → is_k_bad k n

theorem sum_of_special_k : 
  ∃ S : Finset ℕ, (∀ k ∈ S, special_k k) ∧ S.sum id = 2360 := by
  sorry

#check sum_of_special_k

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_k_l1102_110223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l1102_110238

/-- Calculates the time (in seconds) for a train to pass a bridge -/
noncomputable def timeToCrossBridge (trainLength : ℝ) (bridgeLength : ℝ) (trainSpeed : ℝ) : ℝ :=
  (trainLength + bridgeLength) / (trainSpeed * 1000 / 3600)

theorem train_bridge_crossing_time :
  let trainLength : ℝ := 360
  let bridgeLength : ℝ := 140
  let trainSpeed : ℝ := 45
  timeToCrossBridge trainLength bridgeLength trainSpeed = 40 := by
  sorry

-- Use #eval with Nat instead of ℝ for computation
#eval (360 + 140) * 3600 / (45 * 1000) -- This will output 40

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_crossing_time_l1102_110238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_20_4_l1102_110271

/-- Represents a systematic sampling selection -/
def SystematicSampling (n : ℕ) (k : ℕ) (selection : Finset ℕ) : Prop :=
  selection.card = k ∧
  (∀ i j, i ∈ selection → j ∈ selection → i < j → 
    ∃ d : ℕ, d > 0 ∧ j - i = d) ∧
  ∀ x, x ∈ selection → x ≤ n

theorem systematic_sampling_20_4 :
  ∃ selection : Finset ℕ,
    SystematicSampling 20 4 selection ∧
    selection = {5, 10, 15, 20} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_systematic_sampling_20_4_l1102_110271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l1102_110254

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ) + Real.sqrt 3 * Real.cos (2 * x + φ)

noncomputable def g (x φ : ℝ) : ℝ := Real.cos (x + φ)

theorem min_value_of_g (φ : ℝ) (h1 : 0 < φ) (h2 : φ < Real.pi) :
  let f_translated (x : ℝ) := f (x + Real.pi / 4) φ
  let symmetric_point := (Real.pi / 2, 0)
  ∃ (x0 : ℝ), f_translated x0 = 0 ∧ f_translated (Real.pi - x0) = 0 →
    (∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 6), g x φ ≥ 1 / 2) ∧
    (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 6), g x φ = 1 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l1102_110254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1102_110261

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * (Real.cos x)^2 + 2 * Real.sin x * Real.cos x - Real.sqrt 3 * (Real.sin x)^2

theorem f_properties :
  (∀ x, f (x + π/12) = f (π/12 - x)) ∧
  (∀ x, f (π/3 + x) = -f (π/3 - x)) ∧
  (∃ x₁ x₂, |f x₁ - f x₂| ≥ 4) :=
by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1102_110261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_final_books_l1102_110293

def library_books (initial : ℕ) (return_percentages : List ℚ) (new_checkouts : List ℕ) : ℕ :=
  let process := λ (books : ℕ) (return_percent : ℚ) (new_books : ℕ) =>
    (books - Int.toNat ((books : ℚ) * return_percent).floor) + new_books
  List.foldl (λ acc (return_percent, new_books) => process acc return_percent new_books) initial (List.zip return_percentages new_checkouts)

theorem mary_final_books :
  library_books 15 [2/5, 1/4, 3/10, 1/2] [8, 6, 12, 10] = 23 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_final_books_l1102_110293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l1102_110241

noncomputable section

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The point coordinates -/
def point : ℝ × ℝ := (5, 0)

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  let (x₀, y₀) := p
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the point (5,0) to one of the asymptotic lines 
    of the hyperbola x^2/16 - y^2/9 = 1 is equal to 3 -/
theorem distance_to_asymptote : ∃ (a b c : ℝ), 
  (∀ x y, hyperbola x y → (a * x + b * y + c = 0 ∨ a * x - b * y + c = 0)) ∧
  distance_point_to_line point a b c = 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_asymptote_l1102_110241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_complete_subgraph_seven_l1102_110249

/-- A graph representing employee relationships -/
structure EmployeeGraph where
  /-- The number of vertices (employees) in the graph -/
  n : ℕ
  /-- The degree of each vertex (number of acquaintances per employee) -/
  d : ℕ
  /-- The adjacency relation of the graph -/
  adj : Fin n → Fin n → Bool
  /-- The graph is symmetric -/
  sym : ∀ i j, adj i j = adj j i
  /-- Each vertex has exactly d neighbors -/
  degree : ∀ i, (Finset.filter (fun j => adj i j) (Finset.univ : Finset (Fin n))).card = d

/-- A complete subgraph of size k in a graph -/
def CompleteSubgraph (G : EmployeeGraph) (k : ℕ) (S : Finset (Fin G.n)) : Prop :=
  S.card = k ∧ ∀ i j, i ∈ S → j ∈ S → i ≠ j → G.adj i j

/-- The main theorem: there exists a complete subgraph of size 7 -/
theorem exists_complete_subgraph_seven (G : EmployeeGraph) 
  (h_n : G.n = 2023) (h_d : G.d = 1686) : 
  ∃ S : Finset (Fin G.n), CompleteSubgraph G 7 S := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_complete_subgraph_seven_l1102_110249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1102_110200

theorem max_value_sqrt_sum (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
  (∃ (x y : ℝ), x = Real.sqrt (3*a + 1) ∧ y = Real.sqrt (3*b + 1) ∧ x + y ≤ Real.sqrt 10) ∧
  (∀ (x y : ℝ), x = Real.sqrt (3*a + 1) ∧ y = Real.sqrt (3*b + 1) → x + y ≤ Real.sqrt 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sqrt_sum_l1102_110200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l1102_110208

noncomputable section

/-- Given a function f(x) = ln x - (x-a)/x, where a > 0, prove that if the minimum value
    of f(x) in the interval [1,3] is 1/3, then a = e^(1/3). -/
theorem min_value_implies_a (a : ℝ) (h_a : a > 0) :
  let f := fun x => Real.log x - (x - a) / x
  (∀ x ∈ Set.Icc 1 3, f x ≥ 1/3) ∧ (∃ x ∈ Set.Icc 1 3, f x = 1/3) →
  a = Real.exp (1/3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_implies_a_l1102_110208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_computation_l1102_110297

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := (a - b) / (1 + a * b)

-- Theorem statement
theorem diamond_computation :
  diamond 1 (diamond 2 (diamond 3 (diamond 4 5))) = 87 / 59 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_computation_l1102_110297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_geq_q_l1102_110268

-- Define the real numbers a and x
variable (a x : ℝ)

-- Define p and q as functions of a and x
noncomputable def p (a : ℝ) : ℝ := a + 1 / (a - 2)
noncomputable def q (x : ℝ) : ℝ := (1 / 2) ^ (x^2 - 2)

-- State the theorem
theorem p_geq_q (h : a > 2) : p a ≥ q x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_geq_q_l1102_110268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1102_110229

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
def g (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem function_inequality (x : ℝ) (hx : x ≥ 1) :
  (f x ≤ (1/2) * g x) ∧
  (∀ m : ℝ, (∀ y : ℝ, y ≥ 1 → f y - m * g y ≤ 0) ↔ m ≥ 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1102_110229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angles_sum_for_n_plus_minus_two_l1102_110231

def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

theorem interior_angles_sum_for_n_plus_minus_two 
  (n : ℕ) 
  (h : sum_of_interior_angles n = 3600) : 
  sum_of_interior_angles (n + 2) = 3960 ∧ 
  sum_of_interior_angles (n - 2) = 3240 := by
  sorry

#check interior_angles_sum_for_n_plus_minus_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_angles_sum_for_n_plus_minus_two_l1102_110231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleToColorAllGray_l1102_110239

/-- Represents the state of the grid --/
def Grid := Fin 8 → Fin 8 → Bool

/-- The boundary length of the gray region --/
def boundaryLength (g : Grid) : ℕ := sorry

/-- Checks if a square can be colored gray --/
def canColorGray (g : Grid) (row col : Fin 8) : Bool := sorry

/-- Colors a square gray if possible --/
def colorGray (g : Grid) (row col : Fin 8) : Grid := sorry

/-- Checks if all squares are gray --/
def allGray (g : Grid) : Bool := sorry

/-- Initial all-white grid --/
def initialGrid : Grid := sorry

theorem impossibleToColorAllGray :
  ¬ ∃ (sequence : List (Fin 8 × Fin 8)), 
    let finalGrid := sequence.foldl (λ g (row, col) => colorGray g row col) initialGrid
    allGray finalGrid ∧ 
    ∀ i, i < sequence.length → 
      let partialGrid := (sequence.take i).foldl (λ g (row, col) => colorGray g row col) initialGrid
      canColorGray partialGrid (sequence.get ⟨i, by sorry⟩).1 (sequence.get ⟨i, by sorry⟩).2 :=
by sorry

#check impossibleToColorAllGray

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibleToColorAllGray_l1102_110239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1102_110273

-- Define the vectors
noncomputable def a : ℝ × ℝ := (Real.sqrt 3, 1)
def b : ℝ × ℝ := (0, 1)
noncomputable def c (t : ℝ) : ℝ × ℝ := (-Real.sqrt 3, t)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the perpendicularity condition
def is_perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Main theorem
theorem perpendicular_vectors (t : ℝ) :
  is_perpendicular (a.1 + 2 * b.1, a.2 + 2 * b.2) (c t) → t = 1 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l1102_110273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_size_at_most_six_l1102_110285

/-- The set of solutions to a system of polynomial equations in ℂ³ -/
def SolutionSet (a b c : ℝ) (p q r : ℂ) : Set (ℂ × ℂ × ℂ) :=
  {xyz : ℂ × ℂ × ℂ | let (x, y, z) := xyz
                     a * x + b * y + c * z = p ∧
                     a * x^2 + b * y^2 + c * z^2 = q ∧
                     a * x^3 + b * y^3 + c * z^3 = r}

/-- The theorem stating that the solution set has at most six elements -/
theorem solution_set_size_at_most_six (a b c : ℝ) (p q r : ℂ)
    (ha : a > 0) (hb : b > 0) (hc : c > 0) :
    ∃ (s : Finset (ℂ × ℂ × ℂ)), s.card ≤ 6 ∧ ∀ x ∈ SolutionSet a b c p q r, x ∈ s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_size_at_most_six_l1102_110285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l1102_110244

-- Define the function representing the curve
def f (x : ℝ) : ℝ := 3 - 3 * x^2

-- Define the area of the region
noncomputable def area : ℝ := 2 * ∫ x in (-1)..(0), f x

-- Theorem statement
theorem area_under_curve : area = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_under_curve_l1102_110244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_property_l1102_110236

/-- The ellipse with equation x²/9 + y²/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) + (p.2^2 / 4) = 1}

/-- The left focus of the ellipse -/
noncomputable def F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)

/-- The right focus of the ellipse -/
noncomputable def F₂ : ℝ × ℝ := (Real.sqrt 5, 0)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem ellipse_focal_property (P : ℝ × ℝ) (h_P : P ∈ Ellipse) (h_dist : distance P F₁ = 2) :
  distance P F₂ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_property_l1102_110236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_fraction_evaluation_l1102_110269

theorem ceiling_fraction_evaluation :
  (⌈(20 : ℚ) / 9 - ⌈(35 : ℚ) / 21⌉⌉) / (⌈(35 : ℚ) / 9 + ⌈9 * (20 : ℚ) / 35⌉⌉) = 1 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ceiling_fraction_evaluation_l1102_110269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_exponential_equality_l1102_110258

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) := by sorry

theorem negation_of_exponential_equality :
  (¬ ∃ x : ℝ, (2 : ℝ)^x = 1) ↔ (∀ x : ℝ, (2 : ℝ)^x ≠ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_exponential_equality_l1102_110258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_lia_l1102_110288

/-- The distance from the midpoint of two points to a third point -/
noncomputable def distanceFromMidpoint (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  Real.sqrt ((mx - x3)^2 + (my - y3)^2)

/-- Theorem stating that the distance from the midpoint of (10, -30) and (5, 22) to (6, 8) is √146.25 -/
theorem distance_to_lia : distanceFromMidpoint 10 (-30) 5 22 6 8 = Real.sqrt 146.25 := by
  -- Proof goes here
  sorry

#eval Float.sqrt 146.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_lia_l1102_110288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_satisfying_equation_l1102_110257

theorem count_pairs_satisfying_equation : 
  (Finset.filter (fun (pair : ℕ × ℕ) => 
    let m := pair.1 + 1
    let n := pair.2 + 1
    (1 : ℚ) / m + (1 : ℚ) / n = (1 : ℚ) / 2020)
  (Finset.product (Finset.range 4040) (Finset.range 4040))).card = 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_pairs_satisfying_equation_l1102_110257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_2015_l1102_110286

/-- The sum of k consecutive integers starting from n -/
def consecutiveSum (k n : ℕ) : ℕ := k * (2 * n + k - 1) / 2

/-- The number of ways to represent 2015 as a sum of consecutive integers -/
def representationCount : ℕ := 16

/-- Theorem stating that there are exactly 16 ways to represent 2015 as a sum of consecutive integers -/
theorem consecutive_sum_2015 :
  (∃! (count : ℕ), count = representationCount ∧
    count = Finset.card (Finset.filter (fun p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ consecutiveSum p.1 p.2 = 2015) (Finset.range 2016 ×ˢ Finset.range 2016))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_sum_2015_l1102_110286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_fibonacci_is_13_l1102_110225

def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem seventh_fibonacci_is_13 : fibonacci 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_fibonacci_is_13_l1102_110225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_integers_in_product_divisible_by_four_l1102_110240

theorem max_odd_integers_in_product_divisible_by_four :
  ∀ (a b c d e f : ℕ),
  a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 →
  (a * b * c * d * e * f) % 4 = 0 →
  (∃ (odd_count : ℕ),
    odd_count ≤ 5 ∧
    odd_count = ((a % 2) + (b % 2) + (c % 2) + (d % 2) + (e % 2) + (f % 2)) ∧
    ∀ (other_count : ℕ),
      other_count = ((a % 2) + (b % 2) + (c % 2) + (d % 2) + (e % 2) + (f % 2)) →
      other_count ≤ odd_count) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_odd_integers_in_product_divisible_by_four_l1102_110240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l1102_110242

theorem triangle_cosine_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  (Real.cos A / Real.cos B)^2 + (Real.cos B / Real.cos C)^2 + (Real.cos C / Real.cos A)^2 
  ≥ 4 * (Real.cos A^2 + Real.cos B^2 + Real.cos C^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l1102_110242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_powerSum5Seq_167th_term_l1102_110212

/-- Sequence of natural numbers made up of powers of 5 or sums of distinct powers of 5 -/
def powerSum5Seq : ℕ → ℕ := sorry

/-- The 167th term of the powerSum5Seq -/
def term167 : ℕ := powerSum5Seq 167

/-- Powers of 5 used in the 167th term -/
def powers167 : List ℕ := [7, 5, 2, 1, 0]

theorem powerSum5Seq_167th_term :
  term167 = (powers167.map (λ i => 5^i)).sum := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_powerSum5Seq_167th_term_l1102_110212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1102_110291

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 + 2 * y^2 = 2

/-- The line equation with slope 2 passing through a point on the x-axis -/
def line (x₀ x y : ℝ) : Prop := y = 2 * (x - x₀)

/-- The x-coordinate of the next point in the sequence -/
noncomputable def next_x (x : ℝ) : ℝ := (32 * x - 8) / 18

/-- The sequence of x-coordinates -/
noncomputable def x_seq : ℕ → ℝ
| 0 => -1/14
| n + 1 => next_x (x_seq n)

theorem ellipse_intersection_theorem :
  x_seq 0 = x_seq 4 ∧
  (∀ n, ellipse (x_seq n) 0) ∧
  (∀ n, ∃ y, ellipse (x_seq (n+1)) y ∧ line (x_seq n) (x_seq (n+1)) y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_intersection_theorem_l1102_110291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2017pi_div_6_l1102_110250

theorem sin_2017pi_div_6 : Real.sin (2017 * Real.pi / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2017pi_div_6_l1102_110250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_minus_four_and_four_satisfy_l1102_110276

/-- The set of integer solutions to the inequality x^2 + cx + 1 ≤ 0 -/
def SolutionSet (c : ℤ) : Set ℤ :=
  {x : ℤ | x^2 + c*x + 1 ≤ 0}

/-- The proposition that c satisfies the condition -/
def SatisfiesCondition (c : ℤ) : Prop :=
  ∃ (s : Finset ℤ), s.card = 3 ∧ ∀ x : ℤ, x ∈ SolutionSet c ↔ x ∈ s

/-- The theorem stating that only -4 and 4 satisfy the condition -/
theorem only_minus_four_and_four_satisfy :
  ∀ c : ℤ, SatisfiesCondition c ↔ c = -4 ∨ c = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_minus_four_and_four_satisfy_l1102_110276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l1102_110277

/-- The area of a triangle inscribed in a circle, where the vertices divide the circle into three arcs of lengths 5, 6, and 7 -/
noncomputable def triangleArea : ℝ := 202.2192 / Real.pi^2

/-- The circumference of the circle -/
def circleCircumference : ℝ := 5 + 6 + 7

/-- Theorem: The area of a triangle inscribed in a circle, where the vertices divide the circle into
    three arcs of lengths 5, 6, and 7, is equal to 202.2192 / π² -/
theorem inscribed_triangle_area :
  let r := circleCircumference / (2 * Real.pi)
  let angle1 := 5 * 360 / circleCircumference
  let angle2 := 6 * 360 / circleCircumference
  let angle3 := 7 * 360 / circleCircumference
  (1/2) * r^2 * (Real.sin (angle1 * Real.pi / 180) +
                 Real.sin (angle2 * Real.pi / 180) +
                 Real.sin (angle3 * Real.pi / 180)) = triangleArea :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_area_l1102_110277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_count_l1102_110252

/-- A sequence of 2018 integers on a circle. -/
def CircleSequence : Type := Fin 2018 → ℤ

/-- The property that each number is greater than the sum of the two preceding it clockwise. -/
def ValidSequence (seq : CircleSequence) : Prop :=
  ∀ i : Fin 2018, seq i > seq (i - 1) + seq (i - 2)

/-- The count of positive numbers in the sequence. -/
def PositiveCount (seq : CircleSequence) : ℕ :=
  Finset.card (Finset.filter (λ i => seq i > 0) Finset.univ)

/-- The maximum possible number of positive integers in a valid circle sequence is 1009. -/
theorem max_positive_count :
  ∃ (seq : CircleSequence), ValidSequence seq ∧
    ∀ (other : CircleSequence), ValidSequence other →
      PositiveCount other ≤ PositiveCount seq ∧
      PositiveCount seq = 1009 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_count_l1102_110252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_arrangement_l1102_110270

theorem choir_arrangement (n : Nat) (h : n = 90) :
  (Finset.filter (fun x => 6 ≤ x ∧ x ≤ 25 ∧ n % x = 0) (Finset.range 26)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choir_arrangement_l1102_110270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drift_time_is_48_hours_l1102_110218

/-- The time taken for a boat to travel downstream from Dock A to Dock B -/
def downstream_time : ℝ := 6

/-- The time taken for a boat to travel upstream from Dock B to Dock A -/
def upstream_time : ℝ := 8

/-- The speed of the boat in still water -/
noncomputable def boat_speed : ℝ := sorry

/-- The speed of the water flow -/
noncomputable def water_speed : ℝ := sorry

/-- The distance between Dock A and Dock B -/
noncomputable def distance : ℝ := sorry

/-- The time taken for a piece of plastic foam to drift downstream from Dock A to Dock B -/
noncomputable def drift_time : ℝ := distance / water_speed

theorem drift_time_is_48_hours :
  drift_time = 48 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drift_time_is_48_hours_l1102_110218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1102_110287

noncomputable def f (x : ℝ) := 2 * Real.cos (Real.pi / 3 + x / 2)

theorem f_properties :
  let T := 4 * Real.pi
  ∃ (k : ℤ),
    (∀ x : ℝ, f (x + T) = f x) ∧ 
    (∀ x : ℝ, x ∈ Set.Icc (4 * (k : ℝ) * Real.pi - 8 * Real.pi / 3) (4 * (k : ℝ) * Real.pi - 2 * Real.pi / 3) → 
      ∀ y : ℝ, y ∈ Set.Icc (4 * (k : ℝ) * Real.pi - 8 * Real.pi / 3) (4 * (k : ℝ) * Real.pi - 2 * Real.pi / 3) → 
        x ≤ y → f x ≤ f y) ∧
    (∀ x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi → f x ≤ 2) ∧
    (∀ x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi → -Real.sqrt 3 ≤ f x) ∧
    (∃ x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi ∧ f x = 2) ∧
    (∃ x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi ∧ f x = -Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1102_110287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1102_110230

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | Real.rpow 2 x > 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l1102_110230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_at_distance_2_l1102_110234

-- Define the original line
def original_line (x y : ℝ) : Prop := 3 * x - 4 * y - 1 = 0

-- Define the distance between a point and a line
noncomputable def distance_to_line (x y : ℝ) : ℝ := |3 * x - 4 * y - 1| / 5

-- Define the locus equations
def locus_equation1 (x y : ℝ) : Prop := 3 * x - 4 * y - 11 = 0
def locus_equation2 (x y : ℝ) : Prop := 3 * x - 4 * y + 9 = 0

-- Theorem statement
theorem locus_of_points_at_distance_2 :
  ∀ (x y : ℝ), distance_to_line x y = 2 ↔ (locus_equation1 x y ∨ locus_equation2 x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_points_at_distance_2_l1102_110234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_to_line_l1102_110210

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from point M(1,2) to the line 3x+4y-6=0 is 1 -/
theorem distance_M_to_line : distance_point_to_line 1 2 3 4 (-6) = 1 := by
  -- Expand the definition and simplify
  unfold distance_point_to_line
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M_to_line_l1102_110210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_a_value_l1102_110237

/-- A quadratic function with integer coefficients. -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ
  f : ℝ → ℝ := λ x => (a : ℝ) * x^2 + (b : ℝ) * x + (c : ℝ)

/-- The vertex of a quadratic function. -/
noncomputable def vertex (q : QuadraticFunction) : ℝ × ℝ := 
  (- (q.b : ℝ) / (2 * (q.a : ℝ)), q.f (- (q.b : ℝ) / (2 * (q.a : ℝ))))

theorem quadratic_a_value (q : QuadraticFunction) :
  vertex q = (2, 5) ∧ q.f 1 = 3 → q.a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_a_value_l1102_110237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_theorem_l1102_110227

theorem cubic_roots_theorem (a b c d : ℝ) :
  let p (x : ℝ) := x^2 - (a+d)*x + (a*d-b*c)
  let x₁ := (a + d + Real.sqrt ((a + d)^2 - 4*(a*d - b*c))) / 2
  let x₂ := (a + d - Real.sqrt ((a + d)^2 - 4*(a*d - b*c))) / 2
  (x₁^3)*(x₂^3) = (a*d-b*c)^3 ∧
  x₁^3 + x₂^3 = a^3 + d^3 + 3*a*b*c + 3*b*c*d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_roots_theorem_l1102_110227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_converges_to_L_L_is_fixed_point_L_is_unique_fixed_point_l1102_110247

/-- The sequence (x_n) defined by the given recurrence relation -/
noncomputable def x : ℕ → ℝ
  | 0 => 0  -- Added case for 0
  | 1 => 0
  | 2 => 2
  | (n + 3) => 2^(-x n) + 1/2

/-- The limit of the sequence (x_n) -/
def L : ℝ := 1

/-- Theorem stating that the sequence (x_n) converges to L -/
theorem x_converges_to_L : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |x n - L| < ε := by
  sorry

/-- Theorem stating that L is a fixed point of the recurrence relation -/
theorem L_is_fixed_point : L = 2^(-L) + 1/2 := by
  sorry

/-- Theorem stating that L is the unique fixed point -/
theorem L_is_unique_fixed_point : 
  ∀ y : ℝ, y = 2^(-y) + 1/2 → y = L := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_converges_to_L_L_is_fixed_point_L_is_unique_fixed_point_l1102_110247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_a_b_l1102_110255

/-- Given a function f(x) = -1/b * e^(ax) where a > 0 and b > 0,
    and its tangent line at x=0 is tangent to the circle x^2 + y^2 = 1,
    prove that the maximum value of a + b is √2. -/
theorem max_sum_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (f : ℝ → ℝ), f = λ x => -1/b * Real.exp (a*x)) ∧ 
  (∃ (l : ℝ → ℝ), l = λ x => (-a/b) * x - 1/b) ∧
  (∃ (t : ℝ), t^2 + ((-a/b) * t - 1/b)^2 = 1) →
  (a + b ≤ Real.sqrt 2) ∧ (∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_a_b_l1102_110255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doubled_dimensions_theorem_l1102_110233

/-- Represents the volume of a container in gallons -/
structure Container where
  volume : ℝ

/-- Given a container with initial volume and a scaling factor for its dimensions,
    calculates the new volume after scaling -/
def scaled_volume (initial : Container) (scale_factor : ℝ) : Container :=
  { volume := initial.volume * scale_factor ^ 3 }

theorem doubled_dimensions_theorem (initial : Container) :
  initial.volume = 4 → (scaled_volume initial 2).volume = 32 := by
  intro h
  simp [scaled_volume]
  rw [h]
  norm_num

#eval (scaled_volume { volume := 4 } 2).volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doubled_dimensions_theorem_l1102_110233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_hilt_pecan_pies_l1102_110246

/-- The number of pecan pies Mrs. Hilt baked -/
def pecan_pies (total_pies apple_pies : ℕ) : ℕ :=
  total_pies - apple_pies

theorem mrs_hilt_pecan_pies :
  ∀ (total_pies rows pies_per_row apple_pies : ℕ),
  total_pies = rows * pies_per_row →
  rows = 6 →
  pies_per_row = 5 →
  apple_pies = 14 →
  pecan_pies total_pies apple_pies = 16 := by
  sorry

#check mrs_hilt_pecan_pies

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mrs_hilt_pecan_pies_l1102_110246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_is_sin_l1102_110260

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => Real.cos
  | (n + 1) => λ x => deriv (f n) x

-- Theorem statement
theorem f_2015_is_sin : f 2015 = Real.sin := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2015_is_sin_l1102_110260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_2a_plus_b_l1102_110298

-- Define the vectors and their properties
variable (a b : ℝ × ℝ)

-- Define the angle between vectors a and b
noncomputable def angle_between : ℝ := 150 * Real.pi / 180

-- Define the magnitudes of vectors a and b
axiom mag_a : ‖a‖ = Real.sqrt 3
axiom mag_b : ‖b‖ = 2

-- Define the dot product of vectors a and b
noncomputable def dot_product : ℝ := ‖a‖ * ‖b‖ * Real.cos angle_between

-- Theorem to prove
theorem magnitude_of_2a_plus_b :
  ‖2 • a + b‖ = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_2a_plus_b_l1102_110298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_function_implies_m_negative_one_l1102_110263

/-- A power function of the form y = (m^2 - m - 1)x^m -/
noncomputable def powerFunction (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * (x^m)

/-- Definition of an odd function -/
def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_power_function_implies_m_negative_one :
  ∃ m : ℝ, isOddFunction (powerFunction m) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_power_function_implies_m_negative_one_l1102_110263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_is_quadratic_radical_l1102_110206

-- Define what a quadratic radical is
noncomputable def is_quadratic_radical (x : ℝ) : Prop := ∃ y : ℝ, y > 0 ∧ x = Real.sqrt y

-- Define the given options
noncomputable def option_A : ℝ := Real.sqrt 3
def option_B : ℝ := 1.732
noncomputable def option_C : ℂ := Complex.I * Real.sqrt 3
noncomputable def option_D : ℝ := (3 : ℝ) ^ (1/3)

-- The theorem to prove
theorem sqrt_3_is_quadratic_radical :
  is_quadratic_radical option_A ∧
  ¬is_quadratic_radical option_B ∧
  ¬is_quadratic_radical (option_C.re) ∧
  ¬is_quadratic_radical option_D :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_is_quadratic_radical_l1102_110206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_properties_l1102_110299

theorem vector_angle_properties (α β : ℝ) 
  (hα : 0 < α ∧ α < 2*Real.pi) 
  (hβ : Real.pi < β ∧ β < 0) 
  (h_dist : Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = Real.sqrt 2) 
  (h_sin_β : Real.sin β = -Real.sqrt 2/2) : 
  Real.cos (α - β) = 0 ∧ Real.sin α = -Real.sqrt 2/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_properties_l1102_110299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_solution_l1102_110215

open Set
open MeasureTheory
open Measure

theorem unique_function_solution (f : ℝ → ℝ) :
  ContinuousOn f (Icc 0 1) →
  DifferentiableOn ℝ f (Icc 0 1) →
  (∀ x ∈ Icc 0 1, f x > 0) →
  f 1 / f 0 = Real.exp 1 →
  (∫ (x : ℝ) in Set.Icc 0 1, (1 / (f x)^2) + (deriv f x)^2) ≤ 2 →
  ∃! c : ℝ, c > 0 ∧ ∀ x ∈ Icc 0 1, f x = Real.sqrt (2 * x + 2 * c) ∧ c = 1 / (Real.exp 2 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_solution_l1102_110215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_calculation_l1102_110284

/-- Calculate the present value given future value, interest rate, and time --/
noncomputable def presentValue (futureValue : ℝ) (interestRate : ℝ) (years : ℕ) : ℝ :=
  futureValue / (1 + interestRate) ^ years

/-- The problem statement --/
theorem investment_calculation (futureValue : ℝ) (interestRate : ℝ) (years : ℕ) :
  futureValue = 600000 →
  interestRate = 0.06 →
  years = 15 →
  ∃ (pv : ℝ), abs (presentValue futureValue interestRate years - pv) < 0.005 ∧ 
              pv ≥ 250310.77 ∧ pv < 250310.78 :=
by
  intro h1 h2 h3
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_calculation_l1102_110284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_of_triangle_l1102_110295

/-- The function f(x) = -x^2 + 1 -/
def f (x : ℝ) : ℝ := -x^2 + 1

/-- The derivative of f(x) -/
def f_deriv (x : ℝ) : ℝ := -2 * x

/-- The slope of the tangent line at point (x, f(x)) -/
def tangent_slope (x : ℝ) : ℝ := f_deriv x

/-- The y-intercept of the tangent line at point (x, f(x)) -/
noncomputable def y_intercept (x : ℝ) : ℝ := f x - tangent_slope x * x

/-- The x-intercept of the tangent line at point (x, f(x)) -/
noncomputable def x_intercept (x : ℝ) : ℝ := x - f x / tangent_slope x

/-- The area of the triangle formed by the tangent line and the coordinate axes -/
noncomputable def triangle_area (x : ℝ) : ℝ := (1/2) * x_intercept x * y_intercept x

/-- The minimum area of the triangle formed by any tangent line and the coordinate axes -/
noncomputable def min_triangle_area : ℝ := (4/9) * Real.sqrt 3

theorem min_area_of_triangle (x : ℝ) (hx : x ≠ 0) : 
  triangle_area x ≥ min_triangle_area := by
  sorry

#check min_area_of_triangle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_of_triangle_l1102_110295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_uncommon_cards_l1102_110214

def uncommon_cards_count 
  (num_packs : ℕ) 
  (cards_per_pack : ℕ) 
  (uncommon_fraction : ℚ) : ℕ :=
  (num_packs * (cards_per_pack * uncommon_fraction.num) / uncommon_fraction.den).natAbs

theorem john_uncommon_cards : 
  uncommon_cards_count 10 20 (1/4) = 50 := by
  -- Unfold the definition of uncommon_cards_count
  unfold uncommon_cards_count
  -- Simplify the arithmetic
  norm_num
  -- QED
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_uncommon_cards_l1102_110214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_two_correct_l1102_110243

/-- The exponent of 2 in the prime factorization of 1991^m - 1 -/
def exponent_of_two (m : ℕ) : ℕ :=
  if m % 2 = 0 then
    (Nat.log 2 m) + 3
  else 1

theorem exponent_of_two_correct (m : ℕ) (hm : m > 0) :
  ∃ (n : ℕ), (1991^m - 1) = 2^(exponent_of_two m) * (2 * n + 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponent_of_two_correct_l1102_110243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_solution_concentration_l1102_110209

/-- Represents a sugar solution with a given concentration -/
structure SugarSolution where
  concentration : ℝ

/-- Calculates the resulting concentration when mixing two solutions -/
def mix_solutions (s1 s2 : SugarSolution) (ratio : ℝ) : SugarSolution :=
  { concentration := (1 - ratio) * s1.concentration + ratio * s2.concentration }

theorem replacement_solution_concentration 
  (original : SugarSolution)
  (replacement : SugarSolution)
  (h_original : original.concentration = 0.1)
  (h_result : (mix_solutions original replacement 0.25).concentration = 0.14) :
  replacement.concentration = 0.26 := by
  sorry

-- Remove the #eval statement as it's causing issues
-- and is not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_replacement_solution_concentration_l1102_110209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_cube_equals_identity_l1102_110262

theorem cube_root_of_cube_equals_identity (x : ℝ) : Real.rpow x (1/3) * Real.rpow x (1/3) * Real.rpow x (1/3) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_cube_equals_identity_l1102_110262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1102_110205

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define a, b, and c as noncomputable
noncomputable def a : ℝ := f (sin (2 * π / 7))
noncomputable def b : ℝ := f (cos (5 * π / 7))
noncomputable def c : ℝ := f (tan (5 * π / 7))

-- Theorem statement
theorem order_of_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l1102_110205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_through_point_with_focus_hyperbola_with_shared_asymptotes_and_focus_distance_l1102_110235

-- Define the hyperbola type
structure Hyperbola where
  a : ℝ
  b : ℝ

-- Define the standard form of a hyperbola
def standard_form (h : Hyperbola) : ℝ × ℝ → Prop :=
  λ p ↦ (p.1^2 / h.a^2) - (p.2^2 / h.b^2) = 1

-- Define the distance from a point to a line (simplified for this context)
noncomputable def distance_to_line (p : ℝ × ℝ) (m : ℝ) : ℝ :=
  abs (p.2 - m * p.1) / Real.sqrt (1 + m^2)

-- Theorem 1
theorem hyperbola_through_point_with_focus 
  (h : Hyperbola) 
  (passes_through : standard_form h (6, -5))
  (focus_at : ∃ c, c^2 - h.a^2 = h.b^2 ∧ (-6, 0) = (-c, 0)) :
  h.a^2 = 16 ∧ h.b^2 = 20 :=
by sorry

-- Theorem 2
theorem hyperbola_with_shared_asymptotes_and_focus_distance
  (h : Hyperbola)
  (shared_asymptotes : ∀ (x y : ℝ), y = (Real.sqrt 2 / 2) * x ↔ y = (h.b / h.a) * x)
  (focus_distance : ∃ c, c^2 - h.a^2 = h.b^2 ∧ 
    distance_to_line (c, 0) (h.b / h.a) = 2) :
  h.a^2 = 8 ∧ h.b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_through_point_with_focus_hyperbola_with_shared_asymptotes_and_focus_distance_l1102_110235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1102_110283

/-- The inclination angle of a line with equation ax + by + c = 0 -/
noncomputable def inclinationAngle (a b : ℝ) : ℝ := Real.arctan (-a / b)

/-- The equation of the line: x + √3y + 2 = 0 -/
def lineEquation (x y : ℝ) : Prop := x + Real.sqrt 3 * y + 2 = 0

theorem line_inclination_angle :
  inclinationAngle 1 (Real.sqrt 3) = 150 * Real.pi / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l1102_110283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_decomposition_l1102_110216

-- Define f(x) as ln(e^x + 1)
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp x + 1)

-- Define F(x) as (f(x) + f(-x))/2
noncomputable def F (x : ℝ) : ℝ := (f x + f (-x)) / 2

-- Define G(x) as (f(x) - f(-x))/2
noncomputable def G (x : ℝ) : ℝ := (f x - f (-x)) / 2

-- Theorem: The odd function g(x) in the decomposition of f(x) is x/2
theorem odd_function_decomposition :
  ∀ x : ℝ, G x = x / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_decomposition_l1102_110216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1102_110220

/-- Represents an ellipse with equation 3x^2 + ky^2 = 1 and one focus at (0, 1) -/
structure Ellipse where
  k : ℝ
  eq : ∀ x y : ℝ, 3 * x^2 + k * y^2 = 1
  focus : ℝ × ℝ
  focus_prop : focus = (0, 1)

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt 2 / 2

/-- Theorem stating that the eccentricity of the given ellipse is √2/2 -/
theorem ellipse_eccentricity (e : Ellipse) : eccentricity e = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1102_110220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1102_110211

theorem problem_statement :
  (¬ ∃ a b : ℝ, a^2 - a*b + b^2 < 0) ∧
  (∀ A B C : ℝ, 0 < B ∧ B < A ∧ A < Real.pi → Real.sin A > Real.sin B) ∧
  (¬(∃ a b : ℝ, a^2 - a*b + b^2 < 0) ∧ 
   (∀ A B C : ℝ, 0 < B ∧ B < A ∧ A < Real.pi → Real.sin A > Real.sin B)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1102_110211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_divides_f_n_l1102_110224

/-- Definition of f(n) as the number of lists of different positive integers
    starting with 1 and ending with n, where each term except the last divides its successor. -/
def f : ℕ+ → ℕ := sorry

/-- Theorem stating that for any positive integer N, there exists a positive integer n
    such that f(n) is divisible by N. -/
theorem exists_n_divides_f_n (N : ℕ+) : ∃ n : ℕ+, (N : ℕ) ∣ f n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_n_divides_f_n_l1102_110224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identifiable_l1102_110219

/-- Represents the result of a weighing -/
inductive WeighResult
  | Left  : WeighResult  -- Left side is lighter
  | Right : WeighResult  -- Right side is lighter
  | Equal : WeighResult  -- Both sides are equal

/-- Represents a weighing strategy -/
def WeighingStrategy := List Nat → WeighResult → List Nat

/-- The number of coins -/
def n : Nat := 27

/-- The maximum number of allowed weighings -/
def max_weighings : Nat := 3

/-- Theorem: It's possible to find the counterfeit coin in at most three weighings -/
theorem counterfeit_coin_identifiable :
  ∃ (strategy : List WeighingStrategy),
    (strategy.length ≤ max_weighings) ∧
    (∀ (fake : Fin n),
      ∃ (result : List WeighResult),
        (result.length ≤ max_weighings) ∧
        (List.foldl (λ acc (s, r) => s acc r) (List.range n) (List.zip strategy result)).length = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_coin_identifiable_l1102_110219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_range_l1102_110272

/-- Given vectors a and b in the Cartesian coordinate plane, 
    where any vector c can be uniquely decomposed into c = λa + μb,
    prove that the range of values for m is {m | m ≠ 5}. -/
theorem vector_decomposition_range (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![m - 1, m + 3]
  (∀ c : Fin 2 → ℝ, ∃! p : ℝ × ℝ, c = p.1 • a + p.2 • b) ↔ m ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_decomposition_range_l1102_110272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1102_110204

-- Define propositions p and q
def p : Prop := ∀ α β : Real, (α = β ↔ Real.tan α = Real.tan β)
def q : Prop := ∀ (α : Type) (A : Set α), ∅ ⊆ A

-- State the theorem
theorem problem_statement :
  (¬p) ∧ q ∧ (p ∨ q) ∧ (¬p) := by
  constructor
  · sorry -- Proof that ¬p is true
  constructor
  · sorry -- Proof that q is true
  constructor
  · sorry -- Proof that p ∨ q is true
  · sorry -- Proof that ¬p is true (again)


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1102_110204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_length_is_100_l1102_110217

/-- Represents a rectangular lawn with two intersecting roads -/
structure LawnWithRoads where
  width : ℝ
  roadWidth : ℝ
  roadArea : ℝ

/-- Calculates the length of the lawn given its properties -/
noncomputable def calculateLawnLength (lawn : LawnWithRoads) : ℝ :=
  (lawn.roadArea - lawn.width * lawn.roadWidth + lawn.roadWidth * lawn.roadWidth) / lawn.roadWidth

/-- Theorem stating that the length of the lawn is 100 meters -/
theorem lawn_length_is_100 (lawn : LawnWithRoads) 
  (h1 : lawn.width = 60)
  (h2 : lawn.roadWidth = 10)
  (h3 : lawn.roadArea = 1500) : 
  calculateLawnLength lawn = 100 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculateLawnLength { width := 60, roadWidth := 10, roadArea := 1500 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lawn_length_is_100_l1102_110217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_is_360_l1102_110279

/-- Calculates the simple interest for a given principal, rate, and time --/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Represents the financial transaction described in the problem --/
structure Transaction where
  principalAmount : ℝ
  borrowingPeriod : ℝ
  borrowingRate : ℝ
  lendingPeriod : ℝ
  lendingRate : ℝ

/-- Calculates the gain per year for a given transaction --/
noncomputable def gainPerYear (t : Transaction) : ℝ :=
  let interestEarned := simpleInterest t.principalAmount t.lendingRate t.lendingPeriod
  let interestPaid := simpleInterest t.principalAmount t.borrowingRate t.borrowingPeriod
  (interestEarned - interestPaid) / t.borrowingPeriod

/-- The main theorem stating that the gain per year for the given transaction is 360 Rs --/
theorem transaction_gain_is_360 : 
  let t : Transaction := {
    principalAmount := 9000
    borrowingPeriod := 2
    borrowingRate := 4
    lendingPeriod := 2
    lendingRate := 6
  }
  gainPerYear t = 360 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_is_360_l1102_110279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_labeling_possible_iff_even_l1102_110222

/-- Represents a line in a plane -/
structure Line where
  id : ℕ

/-- Represents an intersection point of two lines -/
structure IntersectionPoint where
  line1 : Line
  line2 : Line
  label : ℕ

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  k : ℕ
  lines : Finset Line
  intersection_points : Finset IntersectionPoint

/-- Condition: No two lines are parallel -/
def no_parallel_lines (config : LineConfiguration) : Prop :=
  ∀ l1 l2, l1 ∈ config.lines → l2 ∈ config.lines → l1 ≠ l2 → 
    ∃ p, p ∈ config.intersection_points ∧ p.line1 = l1 ∧ p.line2 = l2

/-- Condition: No three lines are concurrent -/
def no_concurrent_lines (config : LineConfiguration) : Prop :=
  ∀ l1 l2 l3, l1 ∈ config.lines → l2 ∈ config.lines → l3 ∈ config.lines → 
    l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 →
    (∃ p, p ∈ config.intersection_points ∧ p.line1 = l1 ∧ p.line2 = l2) →
    (∃ q, q ∈ config.intersection_points ∧ q.line1 = l2 ∧ q.line2 = l3) →
    (∃ r, r ∈ config.intersection_points ∧ r.line1 = l1 ∧ r.line2 = l3)

/-- Condition: Each line contains all labels exactly once -/
def valid_labeling (config : LineConfiguration) : Prop :=
  ∀ l, l ∈ config.lines → ∀ i, i ∈ Finset.range (config.k - 1) →
    ∃! p, p ∈ config.intersection_points ∧ (p.line1 = l ∨ p.line2 = l) ∧ p.label = i + 1

/-- Main theorem: Labeling is possible if and only if k is even -/
theorem labeling_possible_iff_even (config : LineConfiguration) :
  config.lines.card = config.k →
  no_parallel_lines config →
  no_concurrent_lines config →
  (∃ labeling : LineConfiguration, labeling.k = config.k ∧
    labeling.lines = config.lines ∧
    valid_labeling labeling) ↔ Even config.k := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_labeling_possible_iff_even_l1102_110222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_EB_in_triangle_l1102_110280

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define D as the midpoint of BC
noncomputable def D (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := (1/2) • (B + C)

-- Define E as the midpoint of AD
noncomputable def E (A B C : EuclideanSpace ℝ (Fin 2)) : EuclideanSpace ℝ (Fin 2) := (1/2) • (A + D A B C)

-- State the theorem
theorem vector_EB_in_triangle (A B C : EuclideanSpace ℝ (Fin 2)) :
  E A B C - B = (3/4) • (B - A) - (1/4) • (C - A) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_EB_in_triangle_l1102_110280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_factor_of_polynomial_l1102_110232

theorem no_factor_of_polynomial (x : ℝ) : 
  let p := fun x => x^4 - x^2 + 4
  let f1 := fun x => x^2 + x + 2
  let f2 := fun x => x^2 - 1
  let f3 := fun x => x^2 + 1
  let f4 := fun x => x^2 + 4
  (∃ (q : ℝ → ℝ), p = fun x => (f1 x) * (q x)) ∨ 
  (∃ (q : ℝ → ℝ), p = fun x => (f2 x) * (q x)) ∨ 
  (∃ (q : ℝ → ℝ), p = fun x => (f3 x) * (q x)) ∨ 
  (∃ (q : ℝ → ℝ), p = fun x => (f4 x) * (q x)) = False := by
  sorry

#check no_factor_of_polynomial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_factor_of_polynomial_l1102_110232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_arithmetic_sequence_l1102_110274

/-- A function that returns the digits of a three-digit number -/
def digits (n : ℕ) : Fin 3 → ℕ :=
  λ i => (n / (10 ^ (2 - i.val))) % 10

/-- Checks if the digits of a number form an arithmetic sequence -/
def is_arithmetic_sequence (n : ℕ) : Prop :=
  ∃ d : ℤ, (digits n 1 : ℤ) - (digits n 0 : ℤ) = d ∧ (digits n 2 : ℤ) - (digits n 1 : ℤ) = d

/-- Checks if the digits of a number are distinct -/
def has_distinct_digits (n : ℕ) : Prop :=
  ∀ i j : Fin 3, i ≠ j → digits n i ≠ digits n j

theorem largest_three_digit_arithmetic_sequence :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 → has_distinct_digits n → is_arithmetic_sequence n →
  n ≤ 963 :=
by
  sorry

#check largest_three_digit_arithmetic_sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_three_digit_arithmetic_sequence_l1102_110274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreasing_l1102_110207

/-- An inverse proportion function -/
noncomputable def inverse_proportion (m : ℝ) (x : ℝ) : ℝ := m / x

/-- The derivative of the inverse proportion function -/
noncomputable def inverse_proportion_derivative (m : ℝ) (x : ℝ) : ℝ := -m / (x^2)

theorem inverse_proportion_decreasing (m : ℝ) :
  (∀ x > 0, (inverse_proportion_derivative m x) < 0) → m > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreasing_l1102_110207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_count_l1102_110296

/-- A convex hexagon with exactly two distinct side lengths -/
structure ConvexHexagon where
  sides : Fin 6 → ℝ
  convex : sorry
  two_distinct_lengths : ∃ a b, a ≠ b ∧ (∀ i, sides i = a ∨ sides i = b)

/-- The perimeter of a hexagon -/
def perimeter (h : ConvexHexagon) : ℝ :=
  (Finset.sum Finset.univ h.sides)

theorem hexagon_side_count (h : ConvexHexagon) 
  (side_7 : ∃ i, h.sides i = 7)
  (side_8 : ∃ i, h.sides i = 8)
  (perim_46 : perimeter h = 46) :
  (Finset.card (Finset.filter (λ i => h.sides i = 8) Finset.univ)) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_side_count_l1102_110296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_power_in_ap_l1102_110245

/-- An arithmetic progression with positive integer terms. -/
structure ArithmeticProgression where
  first : ℕ+  -- First term
  diff : ℕ+   -- Common difference
  seq : ℕ → ℕ+ 
  seq_def : ∀ n, seq n = first + n * diff

/-- Predicate to check if a number is in the arithmetic progression -/
def inAP (ap : ArithmeticProgression) (x : ℕ+) : Prop :=
  ∃ n, ap.seq n = x

theorem sixth_power_in_ap (ap : ArithmeticProgression) 
  (square_in_ap : ∃ x : ℕ+, inAP ap (x^2))
  (cube_in_ap : ∃ y : ℕ+, inAP ap (y^3)) :
  ∃ z : ℕ+, inAP ap (z^6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_power_in_ap_l1102_110245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_day_division_count_l1102_110267

theorem day_division_count : 
  (Finset.filter (fun pair : ℕ × ℕ => pair.1 > 0 ∧ pair.2 > 0 ∧ pair.1 * pair.2 = 86400) (Finset.product (Finset.range 86401) (Finset.range 86401))).card = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_day_division_count_l1102_110267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1102_110266

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6)

def interval : Set ℝ := Set.Icc (-Real.pi / 6) (Real.pi / 4)

theorem f_properties :
  (∀ k : ℤ, ∃ x : ℝ, x = Real.pi / 6 + k * Real.pi / 2 ∧ ∀ y : ℝ, f y = f (2 * x - y)) ∧
  (∃ x : ℝ, x ∈ interval ∧ f x = 3 ∧ ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x : ℝ, x ∈ interval ∧ f x = 0 ∧ ∀ y ∈ interval, f x ≤ f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1102_110266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_N_fifth_power_l1102_110226

theorem det_N_fifth_power {n : Type*} [Fintype n] [DecidableEq n] 
  (N : Matrix n n ℝ) (h : Matrix.det N = 3) :
  Matrix.det (N^5) = 243 := by
  have h1 : Matrix.det (N^5) = (Matrix.det N)^5 := by
    simp [Matrix.det_pow]
  rw [h1, h]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_N_fifth_power_l1102_110226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_white_cubes_l1102_110264

/-- Represents a 1x1x1 unit cube -/
structure UnitCube where
  has_gray_face : Bool

/-- Represents a 3x3x3 cube assembled from unit cubes -/
structure LargeCube where
  small_cubes : Vector UnitCube 27

/-- Counts the number of unit cubes with all white faces in a large cube -/
def count_all_white_cubes (cube : LargeCube) : Nat :=
  (cube.small_cubes.toList.filter (fun c => !c.has_gray_face)).length

/-- The maximum number of unit cubes with all white faces in a 3x3x3 cube
    where some faces are painted gray -/
theorem max_white_cubes (cube : LargeCube) : 
  count_all_white_cubes cube ≤ 15 := by
  sorry

#check max_white_cubes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_white_cubes_l1102_110264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l1102_110203

theorem cosine_identity (α β : ℝ) :
  (Real.cos α) ^ 2 + (Real.cos β) ^ 2 - 2 * (Real.cos α) * (Real.cos β) * (Real.cos (α - β)) = (Real.sin (α - β)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l1102_110203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_tangent_theorem_l1102_110253

-- Define the circle
variable (circle : Set (EuclideanSpace ℝ (Fin 2)))

-- Define the quadrilateral ABCD
variable (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define points M and N where tangents intersect
variable (M N : EuclideanSpace ℝ (Fin 2))

-- Define the intersection point O
variable (O : EuclideanSpace ℝ (Fin 2))

-- Define necessary predicates
def IsInscribed (circle : Set (EuclideanSpace ℝ (Fin 2))) (A B C D : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def IsTangentPoint (circle : Set (EuclideanSpace ℝ (Fin 2))) (P : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def IsOppositeVertices (P Q : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def TangentIntersection (P Q R : EuclideanSpace ℝ (Fin 2)) : Prop := sorry
def IntersectsOn (l1 l2 l3 : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry
def Line (P Q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Assumptions
variable (h1 : IsInscribed circle A B C D)
variable (h2 : IsTangentPoint circle M)
variable (h3 : IsTangentPoint circle N)
variable (h4 : IsOppositeVertices A C)
variable (h5 : IsOppositeVertices B D)
variable (h6 : TangentIntersection A C M)
variable (h7 : TangentIntersection B D N)

-- Theorem statement
theorem quadrilateral_tangent_theorem :
  (IntersectsOn (Line A B) (Line C D) (Line M N)) ∧
  (O ∈ Line M N) →
  (dist M O) / (dist O N) = (dist M A) / (dist N D) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_tangent_theorem_l1102_110253
