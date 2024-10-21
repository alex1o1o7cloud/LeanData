import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_logarithmic_equation_l791_79113

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem unique_solution_logarithmic_equation :
  ∃! x : ℝ, 0 < x ∧ x < 2.5 ∧
    log_base 3 (x + 2) * log_base 3 (2 * x + 1) * (3 - log_base 3 (2 * x^2 + 5 * x + 2)) = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_logarithmic_equation_l791_79113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_darts_score_second_round_l791_79196

theorem darts_score_second_round :
  ∀ (round1 round2 round3 : ℕ),
    round2 = 2 * round1 →
    round3 = (3 * round1 : ℕ) →
    round1 ≥ 8 * 3 →
    round3 ≤ 8 * 9 →
    round2 = 48 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_darts_score_second_round_l791_79196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_M_eq_one_l791_79193

open Real Matrix

-- Define the matrix as a function of α
noncomputable def M (α : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
λ i j => match i, j with
  | 0, 0 => cos α * cos (2 * α)
  | 0, 1 => cos α * sin (2 * α)
  | 0, 2 => -sin α
  | 1, 0 => -sin (2 * α)
  | 1, 1 => cos (2 * α)
  | 1, 2 => 0
  | 2, 0 => sin α * cos (2 * α)
  | 2, 1 => sin α * sin (2 * α)
  | 2, 2 => cos α

theorem det_M_eq_one (α : ℝ) : det (M α) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_M_eq_one_l791_79193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_k_values_l791_79129

/-- Given three vectors OA, OB, and OC in 2D space, and points A, B, and C being collinear,
    prove that the value of k satisfies the equation k = 11 or k = -2. -/
theorem collinear_vectors_k_values (k : ℝ) :
  let OA : Fin 2 → ℝ := ![k, 12]
  let OB : Fin 2 → ℝ := ![4, 5]
  let OC : Fin 2 → ℝ := ![10, k]
  (∃ (t : ℝ), OC - OB = t • (OB - OA)) →
  k = 11 ∨ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_k_values_l791_79129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_to_original_ratio_paper_folding_ratio_l791_79149

/-- A rectangular piece of paper with length twice its width -/
structure Paper where
  width : ℝ
  length : ℝ
  length_eq_twice_width : length = 2 * width

/-- The area of the paper before folding -/
noncomputable def original_area (p : Paper) : ℝ := p.width * p.length

/-- The area of the paper after folding -/
noncomputable def folded_area (p : Paper) : ℝ := (1 + Real.sqrt 2) * p.width * p.width / 2

/-- The theorem stating the ratio of the folded area to the original area -/
theorem folded_to_original_ratio (p : Paper) :
  folded_area p / original_area p = (1 + Real.sqrt 2) / 4 := by
  sorry

/-- The main theorem proving the ratio of areas -/
theorem paper_folding_ratio :
  ∃ (p : Paper), folded_area p / original_area p = (1 + Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_folded_to_original_ratio_paper_folding_ratio_l791_79149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alien_saturday_sequence_l791_79182

def alien_sequence : ℕ → String
  | 0 => "A"  -- Added case for 0
  | 1 => "A"
  | 2 => "AY"
  | 3 => "AYYA"
  | 4 => "AYYAYAAY"
  | n + 1 => alien_sequence n ++ String.map (fun c => if c = 'A' then 'Y' else 'A') (alien_sequence n)

theorem alien_saturday_sequence :
  alien_sequence 6 = "AYYAYAAYYAAYAYYAYAAYAYYAAAYAYAAY" := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alien_saturday_sequence_l791_79182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l791_79100

noncomputable section

-- Define the ellipse parameters
variable (a b c : ℝ)

-- Define points
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def F : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define conditions
variable (h1 : a > b)
variable (h2 : b > 0)
variable (h3 : a / b = 2)
variable (h4 : c = Real.sqrt 3)
variable (h5 : a^2 = b^2 + c^2)

-- Define line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m * y + 1

-- Define the circle with diameter MN
def circle_MN (M N : ℝ × ℝ) (x y : ℝ) : Prop :=
  (x - M.1) * (x - N.1) + (y - M.2) * (y - N.2) = 0

-- Theorem statement
theorem ellipse_and_line_equations :
  (∀ x y, ellipse a b x y ↔ x^2 / 4 + y^2 = 1) ∧
  (∃ M N : ℝ × ℝ,
    ellipse a b M.1 M.2 ∧ ellipse a b N.1 N.2 ∧
    (∃ m, line_l m M.1 M.2 ∧ line_l m N.1 N.2) ∧
    circle_MN M N B.1 B.2 ∧
    ((∀ x y, line_l (-1) x y ↔ x + y - 1 = 0) ∨
     (∀ x y, line_l (5/3) x y ↔ 3*x - 5*y - 3 = 0))) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_equations_l791_79100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_courier_earnings_l791_79198

/-- Calculates the after-tax earnings of a student working as a courier --/
theorem courier_earnings (daily_rate : ℕ) (days_per_week : ℕ) (weeks : ℕ) (tax_rate : ℚ) : 
  daily_rate = 1250 →
  days_per_week = 4 →
  weeks = 4 →
  tax_rate = 13/100 →
  ↑((daily_rate * days_per_week * weeks) - 
    (daily_rate * days_per_week * weeks * tax_rate)) = (17400 : ℚ) := by
  sorry

#eval (1250 * 4 * 4) - Int.floor ((1250 * 4 * 4 : ℚ) * 13 / 100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_courier_earnings_l791_79198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurrence_relation_for_Sn_l791_79122

/-- Given a quadratic equation ax^2 + bx + c = 0 with roots α and β,
    and S_n defined as α^n + β^n, prove the recurrence relation for S_n. -/
theorem recurrence_relation_for_Sn
  (a b c : ℝ) (α β : ℂ) (S : ℕ → ℂ) (n : ℕ) :
  a ≠ 0 →
  (∀ x : ℂ, a * x^2 + b * x + c = 0 ↔ x = α ∨ x = β) →
  (∀ m : ℕ, S m = α^m + β^m) →
  S (n + 2) = (-b * S (n + 1) - c * S n) / a :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurrence_relation_for_Sn_l791_79122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_circle_l791_79101

/-- The circle C defined by the equation x^2 + y^2 + 2y = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 + 2*p.2 = 0}

/-- Point A with coordinates (2, 1) -/
def A : ℝ × ℝ := (2, 1)

/-- The maximum distance from point A to any point on the circle C -/
noncomputable def max_distance : ℝ := 2 * Real.sqrt 2 + 1

/-- Theorem stating that the maximum distance from A to any point on C is 2√2 + 1 -/
theorem max_distance_to_circle :
  ∀ p ∈ Circle, dist A p ≤ max_distance ∧ ∃ q ∈ Circle, dist A q = max_distance :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_circle_l791_79101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_burning_theorem_l791_79151

/-- Represents the burning rate of a candle in fraction per hour -/
noncomputable def burning_rate (burn_time : ℝ) : ℝ := 1 / burn_time

/-- Represents the remaining fraction of a candle after burning for x hours -/
noncomputable def remaining_fraction (burn_time x : ℝ) : ℝ := 1 - (burning_rate burn_time) * x

theorem candle_burning_theorem (x : ℝ) :
  (remaining_fraction 4 x) / (remaining_fraction 3 x) = 2 ↔ x = 12 / 5 := by
  sorry

#check candle_burning_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candle_burning_theorem_l791_79151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l791_79185

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then 2^(1 - |x|) else -((x - 2)^2)

theorem f_property (m : ℝ) (hm : f m = 1/4) : f (1 - m) = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_property_l791_79185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subset_size_l791_79179

theorem largest_subset_size (n : ℕ) (hn : n = 1000) :
  ∃ (S : Finset ℕ),
    S ⊆ Finset.range (n + 1) ∧
    (∀ m ∈ S, ∀ k ∈ S, m ≠ k → m + k ∉ S) ∧
    S.card = n / 2 + 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_subset_size_l791_79179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_and_profit_l791_79174

/-- Represents the daily purchase quantity in kilograms -/
def daily_purchase : ℝ → ℝ := λ x ↦ x

/-- Represents the monthly profit in yuan -/
def monthly_profit : ℝ → ℝ := λ x ↦ -42 * x + 576

/-- The purchase price in yuan per kilogram -/
def purchase_price : ℝ := 4.2

/-- The selling price in yuan per kilogram -/
def selling_price : ℝ := 6

/-- The return price in yuan per kilogram -/
def return_price : ℝ := 1.2

/-- The number of days with high sales (10 kg) in a month -/
def high_sales_days : ℕ := 10

/-- The number of days with low sales (6 kg) in a month -/
def low_sales_days : ℕ := 20

/-- The total number of days in a month -/
def total_days : ℕ := 30

theorem optimal_purchase_and_profit :
  ∃ (x : ℝ),
    6 ≤ x ∧ x ≤ 10 ∧
    daily_purchase x = 6 ∧
    monthly_profit x = 324 ∧
    ∀ y, 6 ≤ y ∧ y ≤ 10 → monthly_profit y ≤ monthly_profit x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_purchase_and_profit_l791_79174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_increasing_l791_79127

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (1 + 2^x)

-- Theorem 1: f is an odd function
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by sorry

-- Theorem 2: f is strictly increasing on ℝ
theorem f_is_increasing : ∀ x y : ℝ, x < y → f x < f y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_f_is_increasing_l791_79127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelf_filling_relationship_l791_79109

/-- Represents the number of books that can fill a shelf --/
structure ShelfFilling where
  A : ℕ+  -- Number of algebra books
  P : ℕ+  -- Number of physics books
  H : ℕ+  -- Number of history books
  M : ℕ+  -- Number of mathematics books
  E : ℕ+  -- Number of algebra books in mixed filling
  C : ℕ+  -- Number of physics books in mixed filling
  hDistinct : A ≠ P ∧ A ≠ H ∧ A ≠ M ∧ A ≠ E ∧ A ≠ C ∧
              P ≠ H ∧ P ≠ M ∧ P ≠ E ∧ P ≠ C ∧
              H ≠ M ∧ H ≠ E ∧ H ≠ C ∧
              M ≠ E ∧ M ≠ C ∧
              E ≠ C

/-- Theorem stating the relationship between book quantities --/
theorem shelf_filling_relationship (s : ShelfFilling) :
  (s.C : ℚ) = (s.M : ℚ) * ((s.A : ℚ) - (s.E : ℚ)) / (2 * (s.A : ℚ) * (s.H : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelf_filling_relationship_l791_79109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chang_mixture_alcohol_amount_l791_79128

/-- The amount of pure alcohol in a mixture of two solutions -/
noncomputable def pure_alcohol_in_mixture (volume_a : ℝ) (volume_b : ℝ) (percent_a : ℝ) (percent_b : ℝ) : ℝ :=
  (volume_a * percent_a / 100) + (volume_b * percent_b / 100)

theorem chang_mixture_alcohol_amount :
  let volume_a : ℝ := 600
  let volume_b : ℝ := volume_a + 500
  let percent_a : ℝ := 16
  let percent_b : ℝ := 10
  pure_alcohol_in_mixture volume_a volume_b percent_a percent_b = 206 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chang_mixture_alcohol_amount_l791_79128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_vertex_l791_79153

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  vertex : Point
  width : ℝ
  height : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point could be a vertex of the rectangle -/
noncomputable def isValidVertex (r : Rectangle) (p : Point) : Prop :=
  distance r.vertex p ≤ Real.sqrt (r.width^2 + r.height^2)

theorem invalid_vertex (r : Rectangle) (p : Point) :
  r.vertex = Point.mk 1 2 →
  r.width = 3 →
  r.height = 4 →
  p = Point.mk 1 (-5) →
  ¬ isValidVertex r p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_invalid_vertex_l791_79153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l791_79165

noncomputable def data : List ℝ := [3, 5, 4, 7, 6]

noncomputable def mean (xs : List ℝ) : ℝ := xs.sum / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m)^2)).sum / xs.length

theorem variance_of_data : variance data = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l791_79165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l791_79148

/-- An arithmetic sequence with given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  a2_eq_8 : a 2 = 8
  a4_eq_4 : a 4 = 4

/-- Sum of first n terms of an arithmetic sequence -/
def S_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.a 9 = -6 ∧
  (∀ n, S_n seq n ≤ 30) ∧
  S_n seq 5 = 30 ∧
  S_n seq 6 = 30 ∧
  (∀ n, n ≠ 5 ∧ n ≠ 6 → S_n seq n < 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l791_79148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_g_l791_79112

-- Define the function g
noncomputable def g : ℝ → ℝ := sorry

-- Define the domain of g
def DomainG : Set ℝ := {x | ∃ y, g x = y}

-- State the property of g
axiom g_property (x : ℝ) : x ∈ DomainG → (1 / x^2) ∈ DomainG ∧ g x + g (1 / x^2) = x^2

-- Theorem: The largest set of real numbers in the domain of g is {-1, 1}
theorem largest_domain_g : DomainG = {-1, 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_domain_g_l791_79112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_problem_l791_79186

/-- A bowling problem -/
theorem bowling_problem 
  (initial_average : ℝ) 
  (last_match_wickets : ℕ) 
  (last_match_runs : ℕ) 
  (average_decrease : ℝ) : 
  initial_average = 12.4 →
  last_match_wickets = 6 →
  last_match_runs = 26 →
  average_decrease = 0.4 →
  ∃ (previous_wickets : ℕ), 
    previous_wickets = 115 ∧ 
    (initial_average * (previous_wickets : ℝ) + (last_match_runs : ℝ)) / 
      ((previous_wickets : ℝ) + (last_match_wickets : ℝ)) = 
    initial_average - average_decrease :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_problem_l791_79186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_equivalence_l791_79161

theorem odd_function_equivalence (f : ℝ → ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ↔ (∀ x : ℝ, f (x) = -f (-x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_equivalence_l791_79161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpeting_cost_theorem_l791_79145

/-- Calculates the total cost of carpeting a room -/
noncomputable def carpetingCost (roomLength : ℝ) (roomBreadth : ℝ) (carpetWidth : ℝ) (carpetCostPaisa : ℝ) : ℝ :=
  let roomArea : ℝ := roomLength * roomBreadth
  let carpetWidthMeters : ℝ := carpetWidth / 100
  let carpetLength : ℝ := roomArea / carpetWidthMeters
  let carpetCostRupees : ℝ := carpetCostPaisa / 100
  carpetLength * carpetCostRupees

/-- Theorem stating the total cost of carpeting the given room -/
theorem carpeting_cost_theorem :
  carpetingCost 15 6 75 30 = 36 := by
  -- Unfold the definition of carpetingCost
  unfold carpetingCost
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpeting_cost_theorem_l791_79145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l791_79187

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log (-x^2 + 2*x + 8)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Ioo (-2) 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l791_79187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_elevenths_rounded_l791_79131

/-- Rounds a real number to 3 decimal places -/
noncomputable def round_to_3dp (x : ℝ) : ℝ := 
  (⌊x * 1000 + 0.5⌋ : ℝ) / 1000

/-- The fraction 8/11 rounded to 3 decimal places equals 0.727 -/
theorem eight_elevenths_rounded : round_to_3dp (8/11) = 0.727 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_elevenths_rounded_l791_79131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_pi_thirds_minus_alpha_l791_79163

theorem cos_two_pi_thirds_minus_alpha (α : ℝ) :
  Real.sin (π / 6 - α) = 2 / 3 → Real.cos (2 * π / 3 - α) = -(2 / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_pi_thirds_minus_alpha_l791_79163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l791_79133

theorem equation_solution (x y z : Real) (n k m : Int) :
  (Real.cos x ≠ 0) →
  (Real.cos y ≠ 0) →
  ((Real.cos x)^2 + 1/(Real.cos x)^2)^3 + ((Real.cos y)^2 + 1/(Real.cos y)^2)^3 = 16 * Real.sin z ↔
  (x = π * n ∧ y = π * k ∧ z = π/2 + 2 * π * m) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l791_79133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerabek_theorem_l791_79108

-- Define the basic geometric structures
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)
structure Circle := (center : Point) (radius : ℝ)

-- Define the triangle and its components
structure Triangle :=
  (A B C : Point)
  (A' B' C' : Point)  -- Midpoints
  (L M N : Point)     -- Projections of orthocenter
  (H : Point)         -- Orthocenter
  (k : Circle)        -- Nine-point circle
  (D E F : Point)     -- Intersections of AA', BB', CC' with k
  (P Q R : Point)     -- Intersections of tangent lines with MN, LN, LM

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  ∃ (is_midpoint : Point → Point → Point → Prop)
     (is_orthocenter : Point → Point → Point → Point → Prop)
     (is_projection : Point → Point → Point → Point → Prop)
     (is_nine_point_circle : Circle → Point → Point → Point → Prop)
     (line_intersects_circle : Line → Circle → Point → Prop)
     (tangent_line_intersects : Circle × Point → Line → Point → Prop),
  is_midpoint t.A' t.B t.C ∧
  is_midpoint t.B' t.C t.A ∧
  is_midpoint t.C' t.A t.B ∧
  is_orthocenter t.H t.A t.B t.C ∧
  is_projection t.L t.H t.B t.C ∧
  is_projection t.M t.H t.C t.A ∧
  is_projection t.N t.H t.A t.B ∧
  is_nine_point_circle t.k t.A t.B t.C ∧
  line_intersects_circle (Line.mk 0 0 0) t.k t.D ∧
  line_intersects_circle (Line.mk 0 0 0) t.k t.E ∧
  line_intersects_circle (Line.mk 0 0 0) t.k t.F ∧
  tangent_line_intersects (t.k, t.D) (Line.mk 0 0 0) t.P ∧
  tangent_line_intersects (t.k, t.E) (Line.mk 0 0 0) t.Q ∧
  tangent_line_intersects (t.k, t.F) (Line.mk 0 0 0) t.R

-- Define collinearity
def collinear (P Q R : Point) : Prop :=
  ∃ (t : ℝ), Q.x - P.x = t * (R.x - P.x) ∧ Q.y - P.y = t * (R.y - P.y)

-- Theorem statement
theorem jerabek_theorem (t : Triangle) (h : is_valid_triangle t) : 
  collinear t.P t.Q t.R :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerabek_theorem_l791_79108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l791_79106

/-- Given two vectors a and b in R^2, if k*a + b is perpendicular to b, then k = -10/3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-2, 4))
    (h3 : (k • a + b) • b = 0) :
  k = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l791_79106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_power_function_l791_79114

-- Define the structure of a function
structure MyFunction where
  f : ℝ → ℝ

-- Define what it means for a function to be a power function
def isPowerFunction (f : MyFunction) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f.f x = x^a

-- Define the given functions
noncomputable def functionA : MyFunction := ⟨λ x => 2 * x^2⟩
noncomputable def functionB : MyFunction := ⟨λ x => x^3 + x⟩
noncomputable def functionC : MyFunction := ⟨λ x => 3^x⟩
noncomputable def functionD : MyFunction := ⟨λ x => x^(1/2)⟩

-- Theorem statement
theorem only_D_is_power_function :
  ¬(isPowerFunction functionA) ∧
  ¬(isPowerFunction functionB) ∧
  ¬(isPowerFunction functionC) ∧
  (isPowerFunction functionD) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_power_function_l791_79114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_pure_imaginary_l791_79142

-- Define the complex number z
def z (a : ℝ) : ℂ := a + Complex.I

-- Define the condition for a pure imaginary number
def isPureImaginary (c : ℂ) : Prop := c.re = 0 ∧ c.im ≠ 0

-- Theorem statement
theorem complex_product_pure_imaginary (a : ℝ) :
  isPureImaginary ((1 + Complex.I) * z a) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_product_pure_imaginary_l791_79142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dropped_to_initial_ratio_is_one_third_l791_79143

/-- Represents the supplies for Kelly's art project --/
structure Supplies where
  students : ℕ
  paper_per_student : ℕ
  glue_bottles : ℕ
  additional_paper : ℕ
  remaining_supplies : ℕ

/-- Calculates the ratio of dropped supplies to initial supplies --/
def dropped_to_initial_ratio (s : Supplies) : ℚ :=
  let initial_supplies := s.students * s.paper_per_student + s.glue_bottles
  let dropped_supplies := initial_supplies + s.additional_paper - s.remaining_supplies
  dropped_supplies / initial_supplies

/-- Theorem stating that the ratio of dropped supplies to initial supplies is 1:3 --/
theorem dropped_to_initial_ratio_is_one_third (s : Supplies) 
  (h1 : s.students = 8)
  (h2 : s.paper_per_student = 3)
  (h3 : s.glue_bottles = 6)
  (h4 : s.additional_paper = 5)
  (h5 : s.remaining_supplies = 20) :
  dropped_to_initial_ratio s = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dropped_to_initial_ratio_is_one_third_l791_79143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l791_79167

-- Define the universal set U
noncomputable def U : Set ℝ := {x | Real.exp x > 1}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (x - 1)

-- Define the domain A of f
def A : Set ℝ := {x | x > 1}

-- State the theorem
theorem complement_of_A_in_U :
  (Set.compl A ∩ U) = Set.Ioc 0 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l791_79167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_e_value_l791_79132

theorem unique_e_value (a b c d e : ℝ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →
  {a + b, a + c, b + c, a + d, b + d, c + d, a + e, b + e, c + e, d + e} = {32, 36, 37, 48, 51} ∪ {x : ℝ | 37 < x ∧ x < 48} →
  e = 27.5 := by
  sorry

#check unique_e_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_e_value_l791_79132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_row_col_ratio_l791_79194

/-- Represents a grid of integers with 50 rows and 60 columns -/
def Grid := Fin 50 → Fin 60 → ℤ

/-- Sum of elements in a row -/
def rowSum (g : Grid) (i : Fin 50) : ℤ :=
  (Finset.univ : Finset (Fin 60)).sum (λ j ↦ g i j)

/-- Sum of elements in a column -/
def colSum (g : Grid) (j : Fin 60) : ℤ :=
  (Finset.univ : Finset (Fin 50)).sum (λ i ↦ g i j)

/-- Average of row sums -/
noncomputable def avgRowSum (g : Grid) : ℚ :=
  ((Finset.univ : Finset (Fin 50)).sum (λ i ↦ rowSum g i)) / 50

/-- Average of column sums -/
noncomputable def avgColSum (g : Grid) : ℚ :=
  ((Finset.univ : Finset (Fin 60)).sum (λ j ↦ colSum g j)) / 60

theorem avg_row_col_ratio (g : Grid) : avgRowSum g / avgColSum g = 6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_avg_row_col_ratio_l791_79194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l791_79115

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x - y^2 - 6*y = 2

-- Define the center and radius
def center : ℝ × ℝ := (4, -3)
noncomputable def radius : ℝ := 3 * Real.sqrt 3

-- Theorem statement
theorem circle_properties :
  let (a, b) := center
  (∀ x y, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = radius^2) ∧
  a + b + radius = 1 + 3 * Real.sqrt 3 := by
  sorry

#check circle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l791_79115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_81_in_terms_of_m_l791_79136

theorem log_81_in_terms_of_m (m : ℝ) (h : Real.log 3 / Real.log 2 = m) : 
  Real.log 81 / Real.log 2 = 4 * m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_81_in_terms_of_m_l791_79136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_circle_radius_l791_79176

/-- Given a circle with area 39424 sq cm, prove that a square with perimeter
    equal to the circle's radius has an area of approximately 785.12 sq cm. -/
theorem square_area_from_circle_radius (circle_area : ℝ) (square_perimeter : ℝ) 
    (square_area : ℝ) : 
  circle_area = 39424 →
  square_perimeter = Real.sqrt (circle_area / Real.pi) →
  square_area = (square_perimeter / 4) ^ 2 →
  ∃ ε > 0, |square_area - 785.12| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_from_circle_radius_l791_79176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l791_79195

/-- Calculates the length of a train given its speed, time to cross a bridge, and the bridge length. -/
noncomputable def train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) : ℝ :=
  speed * (time / 3600) * 1000 - bridge_length

/-- Theorem stating that a train traveling at 45 km/h, crossing a 227.03-meter bridge in 30 seconds, has a length of 147.97 meters. -/
theorem train_length_calculation :
  let speed : ℝ := 45 -- km/h
  let time : ℝ := 30 -- seconds
  let bridge_length : ℝ := 227.03 -- meters
  abs (train_length speed time bridge_length - 147.97) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l791_79195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_book_lists_l791_79144

variable (num_students : ℕ) (num_algebra_books : ℕ) (num_geometry_books : ℕ)

/-- Represents the books chosen by a student -/
def StudentBooks := Finset (Fin num_algebra_books ⊕ Fin num_geometry_books)

/-- Function to get the books chosen by a student -/
noncomputable def student_books : Fin num_students → StudentBooks num_algebra_books num_geometry_books :=
  sorry

/-- The number of possible combinations of book choices -/
def num_combinations : ℕ := (Nat.choose num_algebra_books 3) * (Nat.choose num_geometry_books 3)

theorem identical_book_lists
  (h1 : num_students = 210)
  (h2 : num_algebra_books = 6)
  (h3 : num_geometry_books = 5) :
  ∃ (s1 s2 : Fin num_students), s1 ≠ s2 ∧ (student_books num_students num_algebra_books num_geometry_books s1 = student_books num_students num_algebra_books num_geometry_books s2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identical_book_lists_l791_79144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_quadrant_l791_79139

noncomputable def point_P (m n : ℝ) : ℝ × ℝ := (2*m - 3, (3*n - m) / 2)

def in_first_or_second_quadrant (p : ℝ × ℝ) : Prop :=
  (p.1 < 0 ∧ p.2 > 0) ∨ (p.1 > 0 ∧ p.2 > 0)

theorem point_P_quadrant (m n : ℝ) 
  (h1 : 2*(m-1)^2 - 7 = -5) 
  (h2 : n - 3 > 0) : 
  in_first_or_second_quadrant (point_P m n) := by
  sorry

#check point_P_quadrant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_P_quadrant_l791_79139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_HGF_l791_79156

-- Define the circle and points
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

structure Point :=
  (coords : ℝ × ℝ)

-- Define the problem setup
def problem_setup (c : Circle) (A B F G H : Point) : Prop :=
  -- AB is a diameter
  (A.coords.1 - c.center.1)^2 + (A.coords.2 - c.center.2)^2 = 4 * c.radius^2 ∧
  (B.coords.1 - c.center.1)^2 + (B.coords.2 - c.center.2)^2 = c.radius^2 ∧
  -- F is on the circle
  (F.coords.1 - c.center.1)^2 + (F.coords.2 - c.center.2)^2 = c.radius^2 ∧
  -- G is on the tangent at B
  ((G.coords.1 - B.coords.1) * (B.coords.1 - c.center.1) +
   (G.coords.2 - B.coords.2) * (B.coords.2 - c.center.2) = 0) ∧
  -- H is on AF and on the tangent at B
  (∃ t : ℝ, H.coords = (A.coords.1 + t * (F.coords.1 - A.coords.1),
                        A.coords.2 + t * (F.coords.2 - A.coords.2))) ∧
  ((H.coords.1 - B.coords.1) * (B.coords.1 - c.center.1) +
   (H.coords.2 - B.coords.2) * (B.coords.2 - c.center.2) = 0) ∧
  -- ∠BAF = 37°
  Real.cos (37 * Real.pi / 180) = 
    ((A.coords.1 - B.coords.1) * (F.coords.1 - B.coords.1) +
     (A.coords.2 - B.coords.2) * (F.coords.2 - B.coords.2)) /
    (((A.coords.1 - B.coords.1)^2 + (A.coords.2 - B.coords.2)^2)^(1/2) *
     ((F.coords.1 - B.coords.1)^2 + (F.coords.2 - B.coords.2)^2)^(1/2))

-- State the theorem
theorem angle_HGF (c : Circle) (A B F G H : Point) :
  problem_setup c A B F G H →
  Real.cos (53 * Real.pi / 180) = 
    ((H.coords.1 - G.coords.1) * (F.coords.1 - G.coords.1) +
     (H.coords.2 - G.coords.2) * (F.coords.2 - G.coords.2)) /
    (((H.coords.1 - G.coords.1)^2 + (H.coords.2 - G.coords.2)^2)^(1/2) *
     ((F.coords.1 - G.coords.1)^2 + (F.coords.2 - G.coords.2)^2)^(1/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_HGF_l791_79156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l791_79169

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := (x - a)^2 + |x - a| - a * (a - 1)

-- Define the function F
noncomputable def F (a x : ℝ) : ℝ := f a x + 4 / x

-- Theorem for part I
theorem part_one (a : ℝ) : f a 0 ≤ 1 → a ≤ 1/2 := by sorry

-- Theorem for part II
theorem part_two (a : ℝ) (h : a ≥ 2) :
  (a = 2 → ∃! x, x > 0 ∧ F a x = 0) ∧
  (a > 2 → ∃ x y, 0 < x ∧ x < y ∧ F a x = 0 ∧ F a y = 0 ∧
    ∀ z, 0 < z ∧ F a z = 0 → z = x ∨ z = y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l791_79169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l791_79190

/-- Circle M with center (1, 3) and radius 1 -/
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 3)^2 = 1

/-- Circle N with center (7, 5) and radius 2 -/
def circle_N (x y : ℝ) : Prop := (x - 7)^2 + (y - 5)^2 = 4

/-- Point P is on circle M -/
def P_on_M (P : ℝ × ℝ) : Prop := circle_M P.1 P.2

/-- Point Q is on circle N -/
def Q_on_N (Q : ℝ × ℝ) : Prop := circle_N Q.1 Q.2

/-- Point A is on the x-axis -/
def A_on_x_axis (A : ℝ × ℝ) : Prop := A.2 = 0

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The main theorem -/
theorem min_distance_sum (P Q A : ℝ × ℝ) 
  (hP : P_on_M P) (hQ : Q_on_N Q) (hA : A_on_x_axis A) :
  ∃ (P' Q' A' : ℝ × ℝ), P_on_M P' ∧ Q_on_N Q' ∧ A_on_x_axis A' ∧
    ∀ (P'' Q'' A'' : ℝ × ℝ), P_on_M P'' → Q_on_N Q'' → A_on_x_axis A'' →
      distance A' P' + distance A' Q' ≤ distance A'' P'' + distance A'' Q'' ∧
      distance A' P' + distance A' Q' = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l791_79190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_increase_l791_79181

/-- Theorem: If the volume of a cone increases by 100% when its height is increased
    (while keeping the radius constant), then the height must have increased by 100%. -/
theorem cone_height_increase (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  let v := (1/3) * Real.pi * r^2 * h
  let h' := { x : ℝ | (1/3) * Real.pi * r^2 * x = 2 * v }
  ∃ (x : ℝ), x ∈ h' ∧ x = 2 * h := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_increase_l791_79181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_plane_l791_79105

/-- The distance from a point to a plane passing through three points -/
theorem distance_point_to_plane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : 
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  -- Define the plane equation coefficients
  let A := (y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁)
  let B := (z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁)
  let C := (x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁)
  let D := -A * x₁ - B * y₁ - C * z₁
  -- Calculate the distance
  let distance := abs (A * x₀ + B * y₀ + C * z₀ + D) / Real.sqrt (A^2 + B^2 + C^2)
  -- Specific points given in the problem
  M₀ = (-5, 3, 7) →
  M₁ = (2, -1, 2) →
  M₂ = (1, 2, -1) →
  M₃ = (3, 2, 1) →
  distance = 2 * Real.sqrt 22 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_plane_l791_79105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l791_79155

/-- Represents a parabola with equation x^2 = -4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = -4*y

/-- The focus of a parabola with equation x^2 = -4y -/
def focus : ℝ × ℝ := (0, -1)

/-- The directrix of a parabola with equation x^2 = -4y -/
def directrix : ℝ → ℝ := λ _ => 1

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- 
For a parabola with equation x^2 = -4y, 
the distance from its focus to any point on its directrix is 2
-/
theorem parabola_focus_directrix_distance (p : Parabola) (x : ℝ) : 
  distance focus (x, directrix x) = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l791_79155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_15_l791_79154

def sequence_a : ℕ → ℕ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | (n + 1) => 2 * sequence_a n + 1

theorem a_4_equals_15 : sequence_a 4 = 15 := by
  -- Compute the values step by step
  have h1 : sequence_a 1 = 1 := rfl
  have h2 : sequence_a 2 = 3 := by rfl
  have h3 : sequence_a 3 = 7 := by rfl
  have h4 : sequence_a 4 = 15 := by rfl
  exact h4


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_4_equals_15_l791_79154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monge_projection_affinity_l791_79152

/-- Represents a point in 3D space -/
structure Point3D where
  x : Real
  y : Real
  z : Real

/-- Represents a plane in 3D space -/
structure Plane where
  normal : Point3D
  d : Real

/-- Represents a line in 3D space -/
structure Line where
  point : Point3D
  direction : Point3D

/-- Represents a polygon in 3D space -/
structure Polygon3D where
  vertices : List Point3D
  plane : Plane

/-- Represents Monge's projection system -/
structure MongeProjection where
  plane1 : Plane
  plane2 : Plane

/-- Represents an affinity transformation -/
structure Affinity where
  matrix : Matrix (Fin 3) (Fin 3) Real
  translation : Point3D

/-- Checks if a plane is perpendicular to another plane -/
def is_perpendicular (plane1 plane2 : Plane) : Prop :=
  sorry

/-- Checks if lines are parallel -/
def are_parallel (line1 line2 : Line) : Prop :=
  sorry

/-- Checks if points are collinear -/
def are_collinear (points : List Point3D) : Prop :=
  sorry

/-- Projects a point onto a plane -/
def project_point (point : Point3D) (plane : Plane) : Point3D :=
  sorry

/-- Main theorem -/
theorem monge_projection_affinity 
  (polygon : Polygon3D) 
  (projection : MongeProjection) :
  ¬(is_perpendicular polygon.plane projection.plane1) ∧
  ¬(is_perpendicular polygon.plane projection.plane2) →
  ∃ (affinity : Affinity),
    (∀ (p : Point3D), 
      p ∈ polygon.vertices →
      are_parallel 
        (Line.mk (project_point p projection.plane1) (project_point p projection.plane2))
        (Line.mk (Point3D.mk 0 0 0) affinity.translation)) ∧
    (∀ (l1 l2 : Line) (i : Point3D),
      (∃ (p1 p2 : Point3D), p1 ∈ polygon.vertices ∧ p2 ∈ polygon.vertices ∧ 
        l1 = Line.mk (project_point p1 projection.plane1) (project_point p2 projection.plane1) ∧
        l2 = Line.mk (project_point p1 projection.plane2) (project_point p2 projection.plane2)) →
      (∃ (i1 i2 : Point3D), 
        i1 ∈ l1.point :: l1.direction :: [] ∧
        i2 ∈ l2.point :: l2.direction :: [] ∧
        i = project_point i1 projection.plane1) →
      are_collinear [i, affinity.translation, Point3D.mk 0 0 0]) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monge_projection_affinity_l791_79152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_not_in_second_quadrant_l791_79126

/-- The vertex of the parabola y = 4x^2 - 4(a+1)x + a cannot be in the second quadrant for any real value of a. -/
theorem parabola_vertex_not_in_second_quadrant (a : ℝ) : 
  ¬∃ (x_v y_v : ℝ), x_v = (a + 1) / 2 ∧ y_v = -(a + 1)^2 + a ∧ x_v < 0 ∧ y_v > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_not_in_second_quadrant_l791_79126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_bound_l791_79177

/-- A sequence of natural numbers where every natural number appears at least once -/
def special_sequence (a : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, ∃ n : ℕ, a n = k

/-- The inequality condition for the sequence -/
def sequence_inequality (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n ≠ m → (1 : ℚ) / 1998 < (|Int.ofNat (a n) - Int.ofNat (a m)| : ℚ) / |Int.ofNat n - Int.ofNat m| ∧
                     (|Int.ofNat (a n) - Int.ofNat (a m)| : ℚ) / |Int.ofNat n - Int.ofNat m| < 1998

theorem special_sequence_bound (a : ℕ → ℕ) 
  (h1 : special_sequence a) 
  (h2 : sequence_inequality a) : 
  ∀ n : ℕ, |Int.ofNat (a n) - Int.ofNat n| < 2000000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_bound_l791_79177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_sum_geq_150_l791_79171

/-- The length of the nth equilateral triangle's base -/
noncomputable def a (n : ℕ) : ℝ := 2 * n / 3

/-- The sum of the lengths of the first n equilateral triangles' bases -/
noncomputable def S (n : ℕ) : ℝ := n * (n + 1) / 3

/-- The theorem stating that 21 is the least positive integer n such that S(n) ≥ 150 -/
theorem least_n_for_sum_geq_150 : ∀ k : ℕ, k > 0 → S k ≥ 150 → k ≥ 21 := by
  sorry

#check least_n_for_sum_geq_150

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_n_for_sum_geq_150_l791_79171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l791_79173

/-- An ellipse with specific properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_pos : 0 < c
  h_a_gt_b : b < a
  h_equation : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ ∃ t : ℝ, x = a * Real.cos t ∧ y = b * Real.sin t
  h_vector_equation : (3 * c, 3 * b) = (a, -b) + (-2 * c, 2 * b)

/-- The eccentricity of an ellipse is 1/5 given specific conditions -/
theorem ellipse_eccentricity (e : Ellipse) : e.c / e.a = 1/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l791_79173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_stride_difference_l791_79125

-- Define the number of strides and leaps between consecutive poles
def elmer_strides : ℕ := 54
def oscar_leaps : ℕ := 15

-- Define the number of poles
def num_poles : ℕ := 51

-- Define the total distance in feet
noncomputable def total_distance : ℝ := 5280

-- Define Elmer's stride length
noncomputable def elmer_stride_length : ℝ := total_distance / (elmer_strides * (num_poles - 1))

-- Define Oscar's leap length
noncomputable def oscar_leap_length : ℝ := total_distance / (oscar_leaps * (num_poles - 1))

-- Theorem to prove
theorem leap_stride_difference : 
  ‖oscar_leap_length - elmer_stride_length - 5‖ < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_stride_difference_l791_79125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_diameters_is_center_l791_79189

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a diameter
structure Diameter (c : Circle) where
  endpoint1 : Point
  endpoint2 : Point
  is_diameter : ((endpoint1.x - c.center.x)^2 + (endpoint1.y - c.center.y)^2 = c.radius^2) ∧
                ((endpoint2.x - c.center.x)^2 + (endpoint2.y - c.center.y)^2 = c.radius^2) ∧
                ((endpoint1.x - endpoint2.x)^2 + (endpoint1.y - endpoint2.y)^2 = (2 * c.radius)^2)

-- Define a function to check if a point is on a line segment
def is_on_segment (p : Point) (a : Point) (b : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p.x = a.x + t * (b.x - a.x) ∧ p.y = a.y + t * (b.y - a.y)

-- Theorem statement
theorem intersection_of_diameters_is_center (c : Circle) 
  (d1 d2 : Diameter c) (p : Point) 
  (h1 : is_on_segment p d1.endpoint1 d1.endpoint2) 
  (h2 : is_on_segment p d2.endpoint1 d2.endpoint2) :
  p = c.center :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_diameters_is_center_l791_79189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l791_79138

noncomputable section

open Real

def f (x : ℝ) : ℝ := (sqrt 3 / 2) * sin (2 * x) - (1 / 2) * (cos x ^ 2 - sin x ^ 2) - 1

def g (x : ℝ) : ℝ := f (x + π / 6)

theorem triangle_problem (A B C : ℝ) (a b c : ℝ) 
  (h1 : c = sqrt 7)
  (h2 : f C = 0)
  (h3 : sin B = 3 * sin A)
  (h4 : g B = 0)
  (h5 : A + B + C = π)
  (h6 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h7 : a * sin B = b * sin A)
  (h8 : b * sin C = c * sin B)
  (h9 : c * sin A = a * sin C) : 
  (a = 1 ∧ b = 3) ∧ 
  (∀ (m n : ℝ × ℝ), 
    m = (cos A, cos B) → 
    n = (1, sin A - cos A * tan B) → 
    0 < m.1 * n.1 + m.2 * n.2 ∧ m.1 * n.1 + m.2 * n.2 ≤ 1) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l791_79138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_formula_l791_79184

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

-- Define f_n recursively
noncomputable def f_n : ℕ → ℝ → ℝ
  | 0, x => x
  | 1, x => f x
  | (n + 2), x => f (f_n (n + 1) x)

-- State the theorem
theorem f_n_formula (n : ℕ) (x : ℝ) (h1 : n ≥ 2) (h2 : x > 0) :
  f_n n x = x / ((2^n - 1) * x + 2^n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_n_formula_l791_79184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_log_property_l791_79147

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_log_property
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_prod : a 3 * a 8 = 9) :
  Real.log (a 1) + Real.log (a 10) = 2 * Real.log 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_log_property_l791_79147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l791_79158

theorem sin_plus_cos_value (α β : Real) 
  (h1 : π/2 < β) (h2 : β < α) (h3 : α < 3*π/4)
  (h4 : Real.cos (α - β) = 12/13) (h5 : Real.sin (α + β) = -3/5) :
  Real.sin α + Real.cos α = 3 * Real.sqrt 65 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_value_l791_79158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_object_detection_l791_79199

/-- The trajectory of the mysterious object -/
def trajectory (x : ℝ) : ℝ := (((x^5 - 2013)^5 - 2013)^5 - 2013)^5

/-- The radar beam line -/
def radar_line (x : ℝ) : ℝ := x + 2013

/-- The square index for a given coordinate -/
noncomputable def square_index (z : ℝ) : ℤ := Int.floor z

theorem object_detection :
  ∃ (x : ℝ), 
    trajectory x = radar_line x ∧ 
    square_index x = 4 ∧ 
    square_index (radar_line x) = 2017 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_object_detection_l791_79199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l791_79141

/-- Represents a point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form my + b = x -/
structure Line where
  m : ℝ
  b : ℝ

/-- The parabola y² = 4x -/
def parabola (p : Point) : Prop := p.y^2 = 4 * p.x

/-- The line intersects the parabola at two distinct points -/
def intersects_parabola (l : Line) : Prop :=
  ∃ (A B : Point), A ≠ B ∧ parabola A ∧ parabola B ∧ l.m * A.y + l.b = A.x ∧ l.m * B.y + l.b = B.x

/-- The dot product of OA and OB is -4 -/
def dot_product_condition (A B : Point) : Prop :=
  A.x * B.x + A.y * B.y = -4

/-- The area of triangle OAB -/
noncomputable def triangle_area (A B : Point) : ℝ :=
  abs (A.x * B.y - A.y * B.x) / 2

/-- The theorem stating the minimum area of triangle OAB -/
theorem min_triangle_area (l : Line) :
  intersects_parabola l →
  (∃ (A B : Point), dot_product_condition A B) →
  ∃ (A B : Point), 
    ∀ (C D : Point), 
      dot_product_condition C D → 
      triangle_area A B ≤ triangle_area C D ∧ 
      triangle_area A B = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_triangle_area_l791_79141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l791_79117

/-- The number of days it takes for worker b to complete the work independently -/
noncomputable def b_days : ℝ := 12

/-- The number of days it takes for worker c to complete the work independently -/
noncomputable def c_days : ℝ := 15

/-- The rate at which worker a completes the work per day -/
noncomputable def a_rate : ℝ := 2 / b_days

/-- The rate at which worker b completes the work per day -/
noncomputable def b_rate : ℝ := 1 / b_days

/-- The rate at which worker c completes the work per day -/
noncomputable def c_rate : ℝ := 1 / c_days

/-- The combined rate at which workers a, b, and c complete the work per day -/
noncomputable def combined_rate : ℝ := a_rate + b_rate + c_rate

/-- The number of days it takes for workers a, b, and c to complete the work together -/
noncomputable def days_to_complete : ℝ := 1 / combined_rate

theorem work_completion_time :
  days_to_complete = 60 / 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l791_79117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_library_average_age_l791_79123

/-- Represents a room in the library --/
structure Room where
  people : ℕ
  average_age : ℚ

/-- Calculates the average age of people in two rooms combined --/
def combined_average_age (room_x room_y : Room) : ℚ :=
  (room_x.people * room_x.average_age + room_y.people * room_y.average_age) / (room_x.people + room_y.people)

/-- Theorem: The average age of all people in both rooms is 33 years --/
theorem library_average_age : 
  let room_x : Room := { people := 8, average_age := 35 }
  let room_y : Room := { people := 5, average_age := 30 }
  combined_average_age room_x room_y = 33 := by
  sorry

#eval combined_average_age { people := 8, average_age := 35 } { people := 5, average_age := 30 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_library_average_age_l791_79123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_time_approx_114_82_l791_79137

/-- The number of digits in the combination -/
def num_digits : ℕ := 5

/-- The number of possible values for each digit (0 to 8, inclusive) -/
def digit_options : ℕ := 9

/-- The time in seconds required for each trial -/
def seconds_per_trial : ℕ := 7

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Calculates the maximum time required to try all combinations -/
noncomputable def max_time_hours : ℝ :=
  (digit_options ^ num_digits * seconds_per_trial) / seconds_per_hour

/-- Theorem stating that the maximum time is approximately 114.82 hours -/
theorem max_time_approx_114_82 :
  abs (max_time_hours - 114.82) < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_time_approx_114_82_l791_79137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_of_unity_satisfy_equation_l791_79118

/-- A complex number z is a root of unity if |z| = 1 -/
def is_root_of_unity (z : ℂ) : Prop := Complex.abs z = 1

/-- A complex number z is a root of the equation 2z^2 + az + 1 = 0 for some integer a -/
def is_root_of_equation (z : ℂ) : Prop :=
  ∃ a : ℤ, 2 * z^2 + a * z + 1 = 0

/-- The theorem stating that there are exactly two roots of unity satisfying the equation -/
theorem two_roots_of_unity_satisfy_equation :
  ∃! (S : Finset ℂ), (∀ z ∈ S, is_root_of_unity z ∧ is_root_of_equation z ∧ 
    ∃ a : ℤ, -3 ≤ a ∧ a ≤ 3 ∧ 2 * z^2 + a * z + 1 = 0) ∧ S.card = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_roots_of_unity_satisfy_equation_l791_79118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_on_interval_l791_79170

noncomputable def f (x : ℝ) := -x + Real.sin x

theorem f_negative_on_interval :
  ∀ x ∈ Set.Ioo (0 : ℝ) (Real.pi / 2), f x < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_on_interval_l791_79170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_difference_l791_79140

/-- The eccentricity function for a hyperbola -/
noncomputable def eccentricity_function (a b : ℝ) (h : a > b ∧ b > 0) : ℝ → ℝ := 
  λ θ => sorry

/-- The hyperbola equation -/
def is_hyperbola (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Theorem stating the difference of eccentricities at specific angles -/
theorem eccentricity_difference
  (a b : ℝ) (h : a > b ∧ b > 0) :
  eccentricity_function a b h (2 * Real.pi / 3) - eccentricity_function a b h (Real.pi / 3) = 2 * Real.sqrt 3 / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_difference_l791_79140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_price_l791_79162

/-- Proves that the full price per ticket is $2.00 given the problem conditions --/
theorem concert_ticket_price (total_tickets : ℕ) (total_spent : ℚ) 
  (discounted_tickets : ℕ) (discounted_price : ℚ) : ℚ :=
by
  have h1 : total_tickets = 10 := by sorry
  have h2 : total_spent = 18.4 := by sorry
  have h3 : discounted_tickets = 4 := by sorry
  have h4 : discounted_price = 1.6 := by sorry

  -- Calculate the full price per ticket
  let full_price : ℚ := (total_spent - discounted_tickets * discounted_price) / 
    (total_tickets - discounted_tickets)

  -- Prove that full_price equals 2.00
  have h5 : full_price = 2 := by sorry

  exact full_price

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concert_ticket_price_l791_79162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_non_positive_implies_m_lower_bound_l791_79157

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 4 * Real.log x - m * x^2 + 1

theorem f_non_positive_implies_m_lower_bound (m : ℝ) :
  (∀ x ∈ Set.Icc 1 (Real.exp 1), f m x ≤ 0) → m ≥ 2 * Real.sqrt (Real.exp 1) / Real.exp 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_non_positive_implies_m_lower_bound_l791_79157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l791_79104

-- Define constants
noncomputable def a : ℝ := Real.pi ^ (3/10)
noncomputable def b : ℝ := Real.log 3 / Real.log Real.pi
noncomputable def c : ℝ := Real.log (Real.sin (2 * Real.pi / 3)) / Real.log 3

-- Theorem statement
theorem order_of_constants : a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_constants_l791_79104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_410_degrees_in_first_quadrant_l791_79159

noncomputable def angle_to_quadrant (angle : ℝ) : ℕ :=
  let normalized_angle := angle % 360
  if 0 ≤ normalized_angle ∧ normalized_angle < 90 then 1
  else if 90 ≤ normalized_angle ∧ normalized_angle < 180 then 2
  else if 180 ≤ normalized_angle ∧ normalized_angle < 270 then 3
  else 4

theorem terminal_side_410_degrees_in_first_quadrant :
  angle_to_quadrant 410 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_terminal_side_410_degrees_in_first_quadrant_l791_79159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l791_79192

open Real

/-- The function to be minimized -/
noncomputable def f (x y : ℝ) : ℝ := x^2 / (x + 2) + y^2 / (y + 1)

/-- The theorem stating the minimum value of the function -/
theorem min_value_theorem :
  ∃ (min : ℝ), min = 1/4 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 →
  f x y ≥ min := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l791_79192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l791_79135

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := x^3 + 1 / (1 + x)

-- State the theorem
theorem f_bounds : ∀ x : ℝ, x ≥ 0 ∧ x ≤ 1 →
  (f x ≥ 1 - x + x^2) ∧ (3/4 < f x ∧ f x ≤ 3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_bounds_l791_79135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_l791_79120

noncomputable section

def principal : ℚ := 5000
def time : ℚ := 2
def borrow_rate : ℚ := 4 / 100
def lend_rate : ℚ := 6 / 100

def simple_interest (p r t : ℚ) : ℚ := p * r * t

def interest_paid : ℚ := simple_interest principal borrow_rate time
def interest_earned : ℚ := simple_interest principal lend_rate time

def total_gain : ℚ := interest_earned - interest_paid
def gain_per_year : ℚ := total_gain / time

theorem transaction_gain : gain_per_year = 100 := by
  -- Unfold definitions
  unfold gain_per_year total_gain interest_earned interest_paid simple_interest
  -- Simplify the expression
  simp [principal, time, borrow_rate, lend_rate]
  -- Perform the calculation
  norm_num

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_l791_79120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l791_79103

theorem equation_solution (a n : ℕ) (ha : a > 0) (hn : n > 0) : 
  (7 * a * n - 3 * Nat.factorial n = 2020) → (a = 289 ∨ a = 68) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l791_79103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_of_five_l791_79124

noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 0 then -x^3 else x + 10

theorem g_composition_of_five : g (g (g (g (g 2)))) = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_of_five_l791_79124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l791_79130

def m : ℕ := 2^5 * 3^3 * 5^4 * 7^2

theorem number_of_factors_of_m : 
  (Finset.filter (λ x : ℕ => x ∣ m) (Finset.range (m + 1))).card = 360 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l791_79130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_l791_79197

/-- The number of revolutions the wheel makes in 1 km -/
def revolutions : ℝ := 424.6284501061571

/-- The distance traveled in meters -/
def distance : ℝ := 1000

/-- The diameter of the wheel in meters -/
noncomputable def wheel_diameter : ℝ := (distance / revolutions) / Real.pi

/-- Theorem stating that the wheel diameter is approximately 0.7495172 meters -/
theorem wheel_diameter_approx :
  |wheel_diameter - 0.7495172| < 0.0000001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_diameter_approx_l791_79197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_inequality_l791_79188

def is_even_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, x ∈ Set.Ioo a b → f x = f (-x)

def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Ioo a b → y ∈ Set.Ioo a b → x < y → f x < f y

structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  acute_A : 0 < A ∧ A < π / 2
  acute_B : 0 < B ∧ B < π / 2
  acute_C : 0 < C ∧ C < π / 2
  sum_angles : A + B + C = π

theorem acute_triangle_inequality (f : ℝ → ℝ) (triangle : AcuteTriangle) :
  is_even_on f (-1) 1 →
  monotone_increasing_on f (-1) 0 →
  f (Real.sin triangle.C) < f (Real.cos triangle.B) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_inequality_l791_79188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_possible_100th_terms_sum_kim_sum_correct_l791_79102

theorem arithmetic_progression_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) :
  a₁ + (n - 1) * d = a₁ + (n - 1) * d := by rfl

theorem possible_100th_terms_sum : 
  (7 + 99 * 1) + (7 + 99 * 2) + (7 + 99 * 4) = 714 := by
  ring

def kim_sum : ℕ := 714

theorem kim_sum_correct : kim_sum = 714 := rfl

#eval kim_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_possible_100th_terms_sum_kim_sum_correct_l791_79102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_l791_79119

noncomputable def f (x : ℝ) := |Real.log (x + 1)|

theorem min_value_sum (a b : ℝ) (h1 : a < b) (h2 : f a = f (-(b+1)/(b+2))) :
  (∃ (m : ℝ), ∀ (x : ℝ), f (8*a + 2*b + 11) ≤ f x) → a + b = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sum_l791_79119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_expression_l791_79175

theorem min_trig_expression :
  ∀ x : ℝ, (Real.sin x ^ 8 + Real.cos x ^ 8 + 1) / (Real.sin x ^ 6 + Real.cos x ^ 6 + 1) ≥ 0 ∧
  ∃ x : ℝ, (Real.sin x ^ 8 + Real.cos x ^ 8 + 1) / (Real.sin x ^ 6 + Real.cos x ^ 6 + 1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_expression_l791_79175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l791_79150

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C C' : ℝ) : ℝ :=
  |C - C'| / Real.sqrt (A^2 + B^2)

/-- First line equation: x + 2y + 3 = 0 -/
def line1 (x y : ℝ) : Prop := x + 2*y + 3 = 0

/-- Second line equation: x + 2y - 3 = 0 -/
def line2 (x y : ℝ) : Prop := x + 2*y - 3 = 0

theorem distance_between_lines :
  distance_between_parallel_lines 1 2 3 (-3) = 6 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_l791_79150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l791_79146

-- Define the revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 10.8 - x^2 / 30
  else if x > 10 then 10.8 / x - 1000 / (3 * x^2)
  else 0

-- Define the profit function
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 8.1 * x - x^3 / 30 - 10
  else if x > 10 then 98 - 1000 / (3 * x) - 2.7 * x
  else 0

-- Theorem statement
theorem max_profit_at_nine :
  ∃ (max_profit : ℝ), W 9 = max_profit ∧ ∀ x > 0, W x ≤ max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l791_79146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_is_scalene_triangle_l791_79180

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := 4 * x + 1
noncomputable def line2 (x : ℝ) : ℝ := -2 * x + 6
noncomputable def line3 : ℝ → ℝ := Function.const ℝ 1

-- Define the intersection points
noncomputable def point1 : ℝ × ℝ := (5/6, 13/3)
noncomputable def point2 : ℝ × ℝ := (0, 1)
noncomputable def point3 : ℝ × ℝ := (5/2, 1)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem polygon_is_scalene_triangle :
  let side1 := distance point1 point2
  let side2 := distance point1 point3
  let side3 := distance point2 point3
  (side1 ≠ side2 ∧ side2 ≠ side3 ∧ side1 ≠ side3) ∧
  (∀ x : ℝ, (x = point1.1 ∨ x = point2.1 ∨ x = point3.1) →
    (line1 x = line2 x ∨ line1 x = line3 x ∨ line2 x = line3 x)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_is_scalene_triangle_l791_79180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gregs_ppo_reward_is_108_l791_79172

/-- The maximum reward possible in the ProcGen environment -/
noncomputable def max_procgen_reward : ℝ := 240

/-- The maximum reward possible in the modified CoinRun environment -/
noncomputable def max_coinrun_reward : ℝ := max_procgen_reward / 2

/-- The percentage of the possible reward obtained by Greg's PPO algorithm -/
noncomputable def ppo_performance : ℝ := 0.9

/-- The reward obtained by Greg's PPO algorithm -/
noncomputable def gregs_ppo_reward : ℝ := max_coinrun_reward * ppo_performance

/-- Theorem stating that Greg's PPO algorithm's reward is equal to 108 -/
theorem gregs_ppo_reward_is_108 : gregs_ppo_reward = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gregs_ppo_reward_is_108_l791_79172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dartboard_area_ratio_l791_79121

/-- Represents a circular dartboard divided into 12 equal sectors with alternating colors -/
structure Dartboard where
  radius : ℝ
  sector_angle : ℝ
  red_sector_area : ℝ
  white_sector_area : ℝ

/-- The properties of the dartboard -/
def dartboard_properties (d : Dartboard) : Prop :=
  d.sector_angle = 30 ∧
  d.red_sector_area = (Real.pi * d.radius^2) / 12 ∧
  d.white_sector_area = (Real.pi * d.radius^2) / 12

/-- The theorem stating that the ratio of white sector area to red sector area is 1 -/
theorem dartboard_area_ratio (d : Dartboard) (h : dartboard_properties d) :
  d.white_sector_area / d.red_sector_area = 1 := by
  sorry

#check dartboard_area_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dartboard_area_ratio_l791_79121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l791_79134

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1/4) : Real.cos (3*θ) = -11/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_triple_angle_l791_79134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_solution_set_l791_79191

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 10| ≥ 8} = Set.Iic (-18) ∪ Set.Ici (-2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_inequality_solution_set_l791_79191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_progression_l791_79107

/-- Given a geometric progression with the first three terms 3^(1/2), 3^(1/3), and 3^(1/6),
    the fourth term is 1. -/
theorem fourth_term_of_geometric_progression (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3^(1/2))
    (h₂ : a₂ = 3^(1/3)) (h₃ : a₃ = 3^(1/6)) :
    ∃ a₄ : ℝ, a₄ = a₃ * (a₃ / a₂) ∧ a₄ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_term_of_geometric_progression_l791_79107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_true_propositions_l791_79166

open Real

-- Define the propositions
def proposition1 : Prop := ∀ x : ℝ, x > 1 → x > 2
def proposition2 : Prop := ∀ α : ℝ, Real.sin α ≠ 1/2 → α ≠ π/6
def proposition3 : Prop := ∀ x y : ℝ, (x ≠ 0 ∨ y ≠ 0) → x * y ≠ 0
def proposition4 : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

-- Theorem stating which propositions are true
theorem true_propositions :
  ¬proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ proposition4 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_true_propositions_l791_79166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l791_79164

theorem problem_solution (x y z : ℤ) 
  (h1 : x > 0) 
  (h2 : x = 7 * y + 3)
  (h3 : 2 * x = 6 * (3 * y) + 2)
  (h4 : 3 * x = 11 * z + 7) :
  x * (3 * y - z) - 11 * y = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l791_79164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_is_one_l791_79168

-- Define the set of invertible integers modulo 12 less than 12
def InvertibleMod12 : Set ℕ := {x | x < 12 ∧ ∃ y, (x * y) % 12 = 1}

-- Define the theorem
theorem remainder_is_one
  (a b c d e : ℕ)
  (ha : a ∈ InvertibleMod12)
  (hb : b ∈ InvertibleMod12)
  (hc : c ∈ InvertibleMod12)
  (hd : d ∈ InvertibleMod12)
  (he : e ∈ InvertibleMod12)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  (a * b * c * d * e + a * b * c * d + a * b * c * e + a * b * d * e + a * c * d * e + b * c * d * e) *
  (Nat.gcd a 12 * Nat.gcd b 12 * Nat.gcd c 12 * Nat.gcd d 12 * Nat.gcd e 12) % 12 = 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_is_one_l791_79168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l791_79116

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (bridge_length : ℝ) (train_speed_kmh : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem stating that a train with given parameters takes 30 seconds to cross a bridge -/
theorem train_crossing_bridge : 
  train_crossing_time 135 240 45 = 30 := by
  -- Unfold the definition of train_crossing_time
  unfold train_crossing_time
  -- Simplify the expression
  simp
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_l791_79116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_filling_theorem_l791_79160

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def pow (base : ℕ) (exp : ℕ) : ℕ :=
  match exp with
  | 0 => 1
  | exp + 1 => base * pow base exp

def grid_filling_ways (n : ℕ) : ℕ :=
  pow 2 ((n-1)*(n-1)) * pow (factorial n) 3

def number_of_valid_fillings (n : ℕ) : ℕ :=
  sorry

theorem grid_filling_theorem (n : ℕ) (h : n ≥ 3) :
  grid_filling_ways n = number_of_valid_fillings n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_filling_theorem_l791_79160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l791_79111

-- Define a triangle
structure Triangle where
  A : Real  -- angle A
  B : Real  -- angle B
  C : Real  -- angle C
  a : Real  -- side opposite to A
  b : Real  -- side opposite to B
  c : Real  -- side opposite to C
  angleSum : A + B + C = Real.pi
  sineLaw : a / Real.sin A = b / Real.sin B
  cosineLaw : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)

-- Part I
theorem part_one (t : Triangle) 
  (h : Real.sin (2*t.A + t.B) / Real.sin t.B = 2 + 2*Real.cos (t.A + t.B)) :
  t.b / t.a = 2 := by
  sorry

-- Part II
theorem part_two (t : Triangle) 
  (h1 : t.b / t.a = 2) (h2 : t.a = 1) (h3 : t.c = Real.sqrt 7) :
  1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l791_79111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_3_equals_one_third_l791_79183

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | n + 2 => 1 - 1 / (sequence_a (n + 1) + 1)

theorem a_3_equals_one_third : sequence_a 3 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_3_equals_one_third_l791_79183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_formula_l791_79110

/-- Right prism with given properties -/
structure RightPrism where
  a : ℝ  -- Length of side AB
  b : ℝ  -- Length of side BC
  α : ℝ  -- Angle between AB and BC
  β : ℝ  -- Acute angle between cross-section plane and base plane

/-- The area of the cross-section in the given prism -/
noncomputable def cross_section_area (p : RightPrism) : ℝ :=
  (p.a^2 * p.b * Real.sin p.α) / (2 * (p.a + p.b) * Real.cos p.β)

/-- Theorem: The area of the cross-section in the given prism is (a²b sin(α)) / (2(a+b) cos(β)) -/
theorem cross_section_area_formula (p : RightPrism) :
  cross_section_area p = (p.a^2 * p.b * Real.sin p.α) / (2 * (p.a + p.b) * Real.cos p.β) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cross_section_area_formula_l791_79110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_theorem_l791_79178

theorem complex_square_theorem (z : ℂ) :
  (Complex.arg z = 2 * Real.pi / 3) →
  (Complex.im z = Real.sqrt 3) →
  z^2 = -2 - 2 * Complex.I * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_square_theorem_l791_79178
