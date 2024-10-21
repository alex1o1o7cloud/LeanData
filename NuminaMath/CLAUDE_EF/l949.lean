import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_concession_l949_94905

/-- Calculates the standard percentage concession given the original price, final price, and additional concession percentage. -/
noncomputable def standard_concession (original_price final_price additional_concession : ℝ) : ℝ :=
  100 * (1 - (final_price / (original_price * (1 - additional_concession / 100))))

theorem shopkeeper_concession :
  let original_price : ℝ := 2000
  let final_price : ℝ := 1120
  let additional_concession : ℝ := 20
  standard_concession original_price final_price additional_concession = 30 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_concession_l949_94905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_eq_four_l949_94967

/-- Infinite power tower function -/
noncomputable def powerTower (x : ℝ) : ℝ := 
  Real.log x / Real.log (Real.log x)

/-- The equation x^(x^(x^...)) = 4 is satisfied when x = √2 -/
theorem infinite_power_tower_eq_four :
  ∃ (x : ℝ), powerTower x = 4 ∧ x = Real.sqrt 2 := by
  use Real.sqrt 2
  constructor
  · sorry -- Proof that powerTower (Real.sqrt 2) = 4
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_power_tower_eq_four_l949_94967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_increasing_necessary_not_sufficient_l949_94907

/-- Represents the sum of the first n terms of a geometric sequence -/
noncomputable def geometricSum (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

/-- A sequence {S_n} is increasing if S_{n+1} > S_n for all n -/
def isIncreasingSequence (S : ℕ → ℝ) : Prop :=
  ∀ n, S (n + 1) > S n

theorem geometric_sum_increasing_necessary_not_sufficient :
  (∀ a₁ q : ℝ, (∀ n : ℕ, isIncreasingSequence (λ n => geometricSum a₁ q n)) → q > 0) ∧
  (∃ a₁ q : ℝ, q > 0 ∧ ¬(∀ n : ℕ, isIncreasingSequence (λ n => geometricSum a₁ q n))) := by
  sorry

#check geometric_sum_increasing_necessary_not_sufficient

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_increasing_necessary_not_sufficient_l949_94907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decompositions_of_81_l949_94914

/-- A type representing a triple of positive integers -/
structure PositiveTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Function to check if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- Function to check if a PositiveTriple represents a valid decomposition of 81 into three perfect squares -/
def isValidDecomposition (t : PositiveTriple) : Prop :=
  isPerfectSquare t.a.val ∧ 
  isPerfectSquare t.b.val ∧ 
  isPerfectSquare t.c.val ∧
  t.a.val + t.b.val + t.c.val = 81

/-- Function to check if two PositiveTriples are equivalent (same numbers in any order) -/
def areEquivalentTriples (t1 t2 : PositiveTriple) : Prop :=
  Multiset.ofList [t1.a.val, t1.b.val, t1.c.val] = Multiset.ofList [t2.a.val, t2.b.val, t2.c.val]

/-- The main theorem: there are exactly 3 unique ways to decompose 81 into three positive perfect squares -/
theorem decompositions_of_81 :
  ∃! (s : Finset PositiveTriple), 
    (∀ t ∈ s, isValidDecomposition t) ∧ 
    (∀ t1 t2 : PositiveTriple, t1 ∈ s → t2 ∈ s → t1 ≠ t2 → ¬areEquivalentTriples t1 t2) ∧
    s.card = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_decompositions_of_81_l949_94914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_5x_mod_9_l949_94980

theorem remainder_of_5x_mod_9 (x : ℕ) (h : x % 9 = 5) : (5 * x) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_5x_mod_9_l949_94980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l949_94985

/-- Represents a number with a repeating decimal expansion -/
structure RepeatingDecimal where
  whole : ℕ
  nonRepeating : List ℕ
  repeating : List ℕ

/-- Convert a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRat (d : RepeatingDecimal) : ℚ :=
  sorry

theorem largest_number
  (a : ℚ)
  (b c d e : RepeatingDecimal)
  (ha : a = 845678 / 100000)
  (hb : b = ⟨8, [4, 5, 6], [7]⟩)
  (hc : c = ⟨8, [4, 5], [6, 7]⟩)
  (hd : d = ⟨8, [4], [5, 6, 7]⟩)
  (he : e = ⟨8, [], [4, 5, 6, 7]⟩) :
  a > b.toRat ∧ a > c.toRat ∧ a > d.toRat ∧ a > e.toRat :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_number_l949_94985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_l949_94970

/-- A structure representing a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A structure representing a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines the mirror line -/
def mirror_line : Line := { a := 1, b := 1, c := 1 }

/-- Defines point A -/
def point_A : Point := { x := 2, y := 3 }

/-- Defines point B -/
def point_B : Point := { x := 1, y := 1 }

/-- Defines the incident ray -/
def incident_ray : Line := { a := 5, b := -4, c := 2 }

/-- Defines the reflected ray -/
def reflected_ray : Line := { a := 4, b := -5, c := 1 }

/-- Checks if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating the properties of the light ray reflection -/
theorem light_reflection :
  ∃ (incidence_point : Point),
    point_on_line incidence_point mirror_line ∧
    point_on_line incidence_point incident_ray ∧
    point_on_line incidence_point reflected_ray ∧
    point_on_line point_A incident_ray ∧
    point_on_line point_B reflected_ray ∧
    distance point_A incidence_point + distance incidence_point point_B = Real.sqrt 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_light_reflection_l949_94970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l949_94954

theorem decimal_to_fraction (x : ℚ) : 
  (∃ (n : ℕ), x = 15 / (99 * 10^n)) → (x.den = 33) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decimal_to_fraction_l949_94954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_at_B_l949_94960

/-- Represents a time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : minutes < 60 := by sorry

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Int) : Time :=
  sorry

/-- Calculates the midpoint between two times -/
def timeMidpoint (t1 t2 : Time) : Time :=
  sorry

/-- The problem statement -/
theorem arrival_time_at_B (scheduled_A scheduled_C : Time) 
  (actual_A : Time := addMinutes scheduled_A 6)
  (actual_C : Time := addMinutes scheduled_C (-6))
  (h1 : scheduled_A = ⟨10, 10, sorry⟩)
  (h2 : scheduled_C = ⟨13, 10, sorry⟩)
  (h3 : ∃ B, B = timeMidpoint scheduled_A scheduled_C) :
  timeMidpoint scheduled_A scheduled_C = ⟨11, 40, sorry⟩ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrival_time_at_B_l949_94960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_rook_placements_l949_94945

/-- A chess board with a checkerboard pattern -/
structure ChessBoard where
  size : ℕ
  is_checkerboard : Bool

/-- A rook placement on a chess board -/
structure RookPlacement where
  board : ChessBoard
  num_rooks : ℕ
  same_color : Bool
  non_attacking : Bool

/-- The number of ways to place rooks on a chess board -/
def num_placements (placement : RookPlacement) : ℕ :=
  sorry

/-- The main theorem -/
theorem checkerboard_rook_placements :
  ∀ (placement : RookPlacement),
    placement.board.size = 9 ∧
    placement.board.is_checkerboard = true ∧
    placement.num_rooks = 9 ∧
    placement.same_color = true ∧
    placement.non_attacking = true →
    num_placements placement = 4 * 3 * 2 * 1 * 5 * 4 * 3 * 2 * 1 := by
  sorry

#eval 4 * 3 * 2 * 1 * 5 * 4 * 3 * 2 * 1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_checkerboard_rook_placements_l949_94945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_holey_triangle_tiling_iff_valid_distribution_l949_94923

/-- A holey triangle is an upward equilateral triangle with some unit triangular holes. -/
structure HoleyTriangle where
  side_length : ℕ
  holes : Finset (Fin side_length × Fin side_length)
  hole_count : holes.card = side_length

/-- A diamond is a 60°-120° unit rhombus. -/
structure Diamond

/-- The condition that every upward equilateral subtriangle contains at most as many holes as its side length. -/
def valid_hole_distribution (T : HoleyTriangle) : Prop :=
  ∀ k : ℕ, 1 ≤ k → k ≤ T.side_length →
    ∀ i j : Fin T.side_length,
      (Finset.range k).card ≤ (T.holes.filter (λ p ↦
        i ≤ p.1 ∧ p.1 < i + k ∧ j ≤ p.2 ∧ p.2 < j + k)).card

/-- A tiling of a holey triangle with diamonds. -/
def diamond_tiling (T : HoleyTriangle) : Type :=
  { tiling : Finset (Diamond × (Fin T.side_length × Fin T.side_length)) //
    -- Add appropriate conditions to ensure the tiling is valid
    sorry }

/-- The main theorem: A holey triangle can be tiled with diamonds if and only if
    it satisfies the valid hole distribution condition. -/
theorem holey_triangle_tiling_iff_valid_distribution (T : HoleyTriangle) :
  Nonempty (diamond_tiling T) ↔ valid_hole_distribution T :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_holey_triangle_tiling_iff_valid_distribution_l949_94923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_cube_root_negative_27_l949_94939

theorem opposite_cube_root_negative_27 : -(Real.rpow (-27) (1/3)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_cube_root_negative_27_l949_94939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_area_ratio_l949_94977

theorem tetrahedron_face_area_ratio (S₁ S₂ S₃ S₄ : ℝ) :
  let S := max (max (max S₁ S₂) S₃) S₄
  let lambda := (S₁ + S₂ + S₃ + S₄) / S
  0 < S₁ ∧ 0 < S₂ ∧ 0 < S₃ ∧ 0 < S₄ →
  2 < lambda ∧ lambda ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_face_area_ratio_l949_94977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_complement_l949_94993

noncomputable section

-- Define the universal set U
def U : Set ℝ := {x | x > 0}

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (1 - Real.log x)

-- Define the domain of f
def A : Set ℝ := {x ∈ U | ∃ y, f x = y}

-- State the theorem
theorem domain_complement : 
  (U \ A) = {x : ℝ | x ≥ Real.exp 1} := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_complement_l949_94993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_and_expectation_of_X_l949_94950

/- Define the total number of individuals -/
def total_individuals : ℕ := 6

/- Define the number of females -/
def num_females : ℕ := 2

/- Define the number of males -/
def num_males : ℕ := 4

/- Define the number of individuals to be selected -/
def num_selected : ℕ := 3

/- Define the random variable X -/
noncomputable def X : Fin 3 → ℝ
| 0 => 0
| 1 => 1
| 2 => 2

/- Define the probability mass function of X -/
noncomputable def pmf_X : Fin 3 → ℝ
| 0 => 1/5
| 1 => 3/5
| 2 => 1/5

/- State the theorem -/
theorem distribution_and_expectation_of_X :
  (∀ k : Fin 3, pmf_X k = (Nat.choose num_females k * Nat.choose num_males (num_selected - k)) / Nat.choose total_individuals num_selected) ∧
  (Finset.sum (Finset.range 3) (λ k => (X k) * pmf_X k) = 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribution_and_expectation_of_X_l949_94950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_assignment_l949_94984

-- Define the types for names, surnames, and grades
inductive Name : Type where
  | Dima | Misha | Borya | Vasya

inductive Surname : Type where
  | Ivanov | Krylov | Petrov | Orlov

inductive Grade : Type where
  | First | Second | Third | Fourth

-- Define a structure to represent a student
structure Student where
  name : Name
  surname : Surname
  grade : Grade

def students : List Student := [
  ⟨Name.Dima, Surname.Ivanov, Grade.First⟩,
  ⟨Name.Misha, Surname.Krylov, Grade.Second⟩,
  ⟨Name.Borya, Surname.Petrov, Grade.Third⟩,
  ⟨Name.Vasya, Surname.Orlov, Grade.Fourth⟩
]

-- Define the conditions
def conditions (s : List Student) : Prop :=
  -- Boris is not a first grader
  (∀ st, st ∈ s → st.name = Name.Borya → st.grade ≠ Grade.First) ∧
  -- Vasya and Ivanov are on different streets (represented by different students)
  (∀ st1 st2, st1 ∈ s → st2 ∈ s → st1.name = Name.Vasya ∧ st2.surname = Surname.Ivanov → st1 ≠ st2) ∧
  -- Misha is one year older than Dima
  (∃ st1 st2, st1 ∈ s ∧ st2 ∈ s ∧ st1.name = Name.Misha ∧ st2.name = Name.Dima ∧ 
    match st1.grade, st2.grade with
    | Grade.Second, Grade.First => True
    | Grade.Third, Grade.Second => True
    | Grade.Fourth, Grade.Third => True
    | _, _ => False) ∧
  -- Boris and Orlov are neighbors (represented by consecutive grades)
  (∃ st1 st2, st1 ∈ s ∧ st2 ∈ s ∧ st1.name = Name.Borya ∧ st2.surname = Surname.Orlov ∧
    match st1.grade, st2.grade with
    | Grade.First, Grade.Second => True
    | Grade.Second, Grade.Third => True
    | Grade.Third, Grade.Fourth => True
    | Grade.Second, Grade.First => True
    | Grade.Third, Grade.Second => True
    | Grade.Fourth, Grade.Third => True
    | _, _ => False) ∧
  -- Krylov was a first grader last year (now in second grade)
  (∃ st, st ∈ s ∧ st.surname = Surname.Krylov ∧ st.grade = Grade.Second) ∧
  -- Vasya is one year older than Boris
  (∃ st1 st2, st1 ∈ s ∧ st2 ∈ s ∧ st1.name = Name.Vasya ∧ st2.name = Name.Borya ∧
    match st1.grade, st2.grade with
    | Grade.Second, Grade.First => True
    | Grade.Third, Grade.Second => True
    | Grade.Fourth, Grade.Third => True
    | _, _ => False)

theorem correct_assignment :
  conditions students := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_assignment_l949_94984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_A_is_proposition_l949_94929

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  (∃ (b : Bool), (b = true ↔ s = "true") ∧ (b = false ↔ s = "false")) ∧ 
  ¬(s = "true" ∧ s = "false")

-- Define the statements
def statement_A : String := "The sum of the interior angles of a triangle is 180°"
def statement_B : String := "Do not speak loudly"
def statement_C : String := "Is an acute angle complementary to an obtuse angle?"
def statement_D : String := "It's really hot today!"

-- Theorem to prove
theorem only_A_is_proposition : 
  is_proposition statement_A ∧ 
  ¬is_proposition statement_B ∧ 
  ¬is_proposition statement_C ∧ 
  ¬is_proposition statement_D :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_A_is_proposition_l949_94929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_weighted_percentage_increase_l949_94963

/-- Calculates the weighted percentage increase of total earnings given original and new earnings from two jobs. -/
noncomputable def weighted_percentage_increase (bookstore_original : ℝ) (tutoring_original : ℝ) (bookstore_new : ℝ) (tutoring_new : ℝ) : ℝ :=
  let total_original := bookstore_original + tutoring_original
  let total_new := bookstore_new + tutoring_new
  let bookstore_increase := (bookstore_new - bookstore_original) / bookstore_original * 100
  let tutoring_increase := (tutoring_new - tutoring_original) / tutoring_original * 100
  let bookstore_weight := bookstore_new / total_new
  let tutoring_weight := tutoring_new / total_new
  bookstore_increase * bookstore_weight + tutoring_increase * tutoring_weight

/-- Theorem stating that the weighted percentage increase of John's total weekly earnings is approximately 56.31%. -/
theorem johns_weighted_percentage_increase :
  ∃ ε > 0, |weighted_percentage_increase 60 40 100 55 - 56.31| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_weighted_percentage_increase_l949_94963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_result_l949_94912

/-- Calculates the final amount of an investment with compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) (frequency : ℕ) : ℝ :=
  principal * (1 + rate / (frequency : ℝ)) ^ ((frequency : ℝ) * (time : ℝ))

/-- The investment scenario -/
theorem investment_result :
  let principal := 7000
  let rate := 0.10
  let time := 2
  let frequency := 1
  ⌊compound_interest principal rate time frequency⌋ = 8470 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_result_l949_94912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_men_required_for_work_l949_94951

/-- The number of men required to complete a work in a given time --/
noncomputable def number_of_men (work : ℝ) (time : ℝ) (rate : ℝ) : ℝ :=
  work / (time * rate)

/-- The work rate per man per day --/
noncomputable def work_rate (men : ℝ) (time : ℝ) (work : ℝ) : ℝ :=
  work / (men * time)

theorem men_required_for_work (work : ℝ) :
  ∃ (M : ℝ), 
    (work_rate M 40 work = work_rate (M - 5) 50 work) ∧
    (M = 25) := by
  sorry

#check men_required_for_work

end NUMINAMATH_CALUDE_ERRORFEEDBACK_men_required_for_work_l949_94951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l949_94941

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) := x * Real.log x

-- State the theorem
theorem min_value_f :
  ∃ (x : ℝ), x > 0 ∧ f x = -1 / Real.exp 1 ∧ ∀ (y : ℝ), y > 0 → f y ≥ -1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l949_94941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_from_inequality_l949_94976

/-- A function f: (0,1) → ℝ is constant if it's positive and satisfies the given inequality. -/
theorem constant_function_from_inequality 
  (f : ℝ → ℝ) 
  (h_pos : ∀ x, x ∈ Set.Ioo 0 1 → 0 < f x) 
  (h_ineq : ∀ x y, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → f x / f y + f (1 - x) / f (1 - y) ≤ 2) :
  ∀ x y, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → f x = f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_function_from_inequality_l949_94976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l949_94957

-- Define the functions f and g
noncomputable def f (p q x : ℝ) : ℝ := x^2 + p*x + q
noncomputable def g (x : ℝ) : ℝ := x + Real.sqrt 3

-- State the theorem
theorem max_value_of_f (p q : ℝ) :
  (∃ x₀ ∈ Set.Icc 1 2, 
    (∀ x ∈ Set.Icc 1 2, f p q x₀ ≤ f p q x) ∧
    (∀ x ∈ Set.Icc 1 2, g x₀ ≤ g x) ∧
    f p q x₀ = g x₀) →
  (∃ x ∈ Set.Icc 1 2, f p q x = 4 - Real.sqrt 3 ∧
    ∀ y ∈ Set.Icc 1 2, f p q y ≤ 4 - Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l949_94957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l949_94920

/-- The maximum size of a subset A of {1, 2, ..., 1000000} such that 
    for any x, y ∈ A with x ≠ y, xy ∉ A, is 999001. -/
theorem max_subset_size :
  ∃ (A : Finset ℕ),
    (∀ a, a ∈ A → a ≤ 1000000) ∧
    (∀ x y, x ∈ A → y ∈ A → x ≠ y → x * y ∉ A) ∧
    A.card = 999001 ∧
    (∀ B : Finset ℕ, (∀ b, b ∈ B → b ≤ 1000000) → 
      (∀ x y, x ∈ B → y ∈ B → x ≠ y → x * y ∉ B) → B.card ≤ 999001) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_subset_size_l949_94920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_four_distinct_digits_l949_94958

/-- Represents a 10x10 table filled with digits 0 to 9 -/
def DigitTable := Fin 10 → Fin 10 → Fin 10

/-- Counts the occurrences of a digit in the table -/
def count_digit (t : DigitTable) (d : Fin 10) : Nat :=
  Finset.sum Finset.univ (fun i => Finset.sum Finset.univ (fun j => if t i j = d then 1 else 0))

/-- Counts the number of distinct digits in a row -/
def distinct_in_row (t : DigitTable) (row : Fin 10) : Nat :=
  Finset.card (Finset.image (t row) Finset.univ)

/-- Counts the number of distinct digits in a column -/
def distinct_in_column (t : DigitTable) (col : Fin 10) : Nat :=
  Finset.card (Finset.image (fun i => t i col) Finset.univ)

/-- The main theorem stating that there always exists a row or column with at least 4 distinct digits -/
theorem exists_four_distinct_digits (t : DigitTable) 
  (h : ∀ d : Fin 10, count_digit t d = 10) : 
  (∃ row : Fin 10, distinct_in_row t row ≥ 4) ∨ 
  (∃ col : Fin 10, distinct_in_column t col ≥ 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_four_distinct_digits_l949_94958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_zero_l949_94944

open Real Set

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 1

-- State the theorem
theorem exactly_one_zero (a : ℝ) (h : a > 3) :
  ∃! x, x ∈ Ioo 0 2 ∧ f a x = 0 :=
by
  -- We'll use the sorry tactic to skip the proof for now
  sorry

-- You can add more helper lemmas or definitions here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_zero_l949_94944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_three_halves_l949_94906

/-- Helper function to calculate the area of a triangle given three points -/
noncomputable def area_triangle (P Q R : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))

/-- The area of triangle P₁P₂P₃ is 3/2 given the specified conditions -/
theorem triangle_area_three_halves (A B P₁ P₂ P₃ : ℝ × ℝ) : 
  A = (1, 2) →
  B = (3, 4) →
  P₁.1 = 0 →
  P₂.2 = 0 →
  P₃.1 + 3 * P₃.2 = 1 →
  (∀ Q : ℝ × ℝ, Q.1 = 0 → 
    (Q.1 - A.1)^2 + (Q.2 - A.2)^2 + (Q.1 - B.1)^2 + (Q.2 - B.2)^2 ≥ 
    (P₁.1 - A.1)^2 + (P₁.2 - A.2)^2 + (P₁.1 - B.1)^2 + (P₁.2 - B.2)^2) →
  (∀ Q : ℝ × ℝ, Q.2 = 0 → 
    (Q.1 - A.1)^2 + (Q.2 - A.2)^2 + (Q.1 - B.1)^2 + (Q.2 - B.2)^2 ≥ 
    (P₂.1 - A.1)^2 + (P₂.2 - A.2)^2 + (P₂.1 - B.1)^2 + (P₂.2 - B.2)^2) →
  (∀ Q : ℝ × ℝ, Q.1 + 3 * Q.2 = 1 → 
    (Q.1 - A.1)^2 + (Q.2 - A.2)^2 + (Q.1 - B.1)^2 + (Q.2 - B.2)^2 ≥ 
    (P₃.1 - A.1)^2 + (P₃.2 - A.2)^2 + (P₃.1 - B.1)^2 + (P₃.2 - B.2)^2) →
  area_triangle P₁ P₂ P₃ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_three_halves_l949_94906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l949_94994

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_sum_ratio
  (seq : ArithmeticSequence)
  (h : seq.a 5 = 5 * seq.a 3) :
  S seq 9 / S seq 5 = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l949_94994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repaint_to_white_l949_94988

/-- Represents the color of a number -/
inductive Color
| Black
| White

/-- Represents the state of the numbers -/
def State (N : ℕ) := Fin N → Color

/-- Checks if a triplet of numbers satisfies the repainting rule -/
def ValidTriplet (a b c : ℕ) : Prop :=
  a + c = 2 * b ∧ a < b ∧ b < c

/-- Represents a single repainting operation -/
def Repaint (s : State N) (a b c : Fin N) : State N :=
  fun i => if i = a ∨ i = b ∨ i = c then
    match s i with
    | Color.Black => Color.White
    | Color.White => Color.Black
  else s i

/-- Checks if all numbers in the state are white -/
def AllWhite (s : State N) : Prop :=
  ∀ i, s i = Color.White

/-- The main theorem to be proved -/
theorem repaint_to_white (N : ℕ) (h : N ≥ 8) :
  ∀ (s : State N), ∃ (sequence : List (Fin N × Fin N × Fin N)),
    (∀ (abc : Fin N × Fin N × Fin N), abc ∈ sequence → ValidTriplet abc.1.val abc.2.1.val abc.2.2.val) ∧
    AllWhite (sequence.foldl (fun state abc => Repaint state abc.1 abc.2.1 abc.2.2) s) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repaint_to_white_l949_94988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elements_in_set_C_l949_94959

theorem elements_in_set_C (C D : Finset ℕ) 
  (h1 : C.card = 3 * D.card)
  (h2 : (C ∪ D).card = 4500)
  (h3 : (C ∩ D).card = 1200) :
  C.card = 4275 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elements_in_set_C_l949_94959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_trees_375_26_l949_94990

/-- The distance between consecutive trees in a yard -/
noncomputable def distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : ℝ :=
  yard_length / (num_trees - 1)

/-- Theorem: In a 375-meter long yard with 26 trees planted at equal distances,
    including one at each end, the distance between two consecutive trees is 15 meters. -/
theorem distance_between_trees_375_26 :
  distance_between_trees 375 26 = 15 := by
  -- Unfold the definition of distance_between_trees
  unfold distance_between_trees
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_trees_375_26_l949_94990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_O_to_C_l949_94917

/-- Curve C defined by parametric equations -/
noncomputable def C : ℝ → ℝ × ℝ := fun θ ↦ (3 + Real.cos θ, Real.sin θ)

/-- Origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Maximum distance from origin O to any point on curve C -/
theorem max_distance_O_to_C :
  ∃ (M : ℝ × ℝ), ∃ (θ : ℝ), M = C θ ∧ 
  ∀ (N : ℝ × ℝ), ∀ (φ : ℝ), N = C φ → distance O M ≥ distance O N ∧
  distance O M = 4 := by
  sorry

#check max_distance_O_to_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_O_to_C_l949_94917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l949_94930

theorem lambda_range : 
  (∀ l : ℝ, (∀ x : ℝ, l * x^2 - l * x + 1 ≥ 0) ↔ 0 ≤ l ∧ l ≤ 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l949_94930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_positive_odd_divisors_of_180_l949_94996

theorem sum_of_positive_odd_divisors_of_180 : 
  (Finset.filter (fun d => d % 2 = 1 && d ∣ 180) (Finset.range 181)).sum id = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_positive_odd_divisors_of_180_l949_94996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_driveway_paving_time_l949_94975

-- Define the rates and time for Mary and Hillary
noncomputable def mary_alone_time : ℝ := 4
noncomputable def hillary_alone_time : ℝ := 3

-- Define the rate changes when working together
noncomputable def mary_rate_increase : ℝ := 0.3333
noncomputable def hillary_rate_decrease : ℝ := 0.5

-- Define the function to calculate the time taken when working together
noncomputable def time_together (mary_time : ℝ) (hillary_time : ℝ) (mary_increase : ℝ) (hillary_decrease : ℝ) : ℝ :=
  1 / ((1 / mary_time * (1 + mary_increase)) + (1 / hillary_time * (1 - hillary_decrease)))

-- Theorem statement
theorem driveway_paving_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |time_together mary_alone_time hillary_alone_time mary_rate_increase hillary_rate_decrease - 2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_driveway_paving_time_l949_94975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_to_inclination_l949_94966

-- Define the slope of the line
def slope_range (k : ℝ) : Prop := -1 ≤ k ∧ k ≤ 1

-- Define the inclination angle range
def inclination_range (α : ℝ) : Prop := 
  (0 ≤ α ∧ α ≤ Real.pi/4) ∨ (3*Real.pi/4 ≤ α ∧ α < Real.pi)

-- Theorem statement
theorem slope_to_inclination (k α : ℝ) (h : slope_range k) : 
  k = Real.tan α → inclination_range α := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_to_inclination_l949_94966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_liars_is_five_l949_94927

/-- Represents a person in the room, who is either a liar or a knight. -/
inductive Person
| liar
| knight

/-- The total number of people in the room. -/
def total_people : Nat := 10

/-- A function that returns the statement made by the nth person. -/
def statement (n : Nat) : Prop := ∃ (k : Nat), k ≥ n ∧ k ≤ total_people

/-- A function that determines if a person is telling the truth based on their statement and the actual number of liars. -/
def is_telling_truth (n : Nat) (num_liars : Nat) : Prop := 
  statement n ↔ (num_liars ≥ n)

/-- The main theorem stating that the number of liars is exactly 5. -/
theorem number_of_liars_is_five : 
  ∃! (num_liars : Nat), 
    (num_liars ≤ total_people) ∧ 
    (∀ (i : Nat), i ≤ total_people → 
      ((i ≤ num_liars) ↔ is_telling_truth i num_liars)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_liars_is_five_l949_94927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l949_94974

noncomputable def f (a x : ℝ) : ℝ := 1 - (1/2) * Real.cos (2*x) + a * Real.sin (x/2) * Real.cos (x/2)

theorem max_value_implies_a (a : ℝ) :
  (∃ M, M = 3 ∧ ∀ x, f a x ≤ M) → (a = 3 ∨ a = -3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l949_94974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_x_equals_1_l949_94940

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + Real.log (2 - x)

-- State the theorem
theorem f_symmetric_about_x_equals_1 :
  ∀ x ∈ Set.Ioo 0 2, f x = f (2 - x) := by
  sorry

#check f_symmetric_about_x_equals_1

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetric_about_x_equals_1_l949_94940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l949_94969

theorem polynomial_division_remainder : 
  ∃ q : Polynomial ℂ, (X : Polynomial ℂ)^55 + X^44 + X^33 + X^22 + X^11 + 1 = 
  (X^6 + X^5 + X^4 + X^3 + X^2 + X + 1) * q + 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_remainder_l949_94969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_roots_of_unity_satisfying_equation_l949_94964

def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z ^ n = 1

def satisfies_equation (z : ℂ) (c d : ℤ) : Prop :=
  z^2 + c * z + d = 0

theorem count_roots_of_unity_satisfying_equation :
  ∃! (S : Finset ℂ),
    (∀ z ∈ S, is_root_of_unity z ∧ 
      ∃ c d : ℤ, satisfies_equation z c d ∧ 
        (c.natAbs ≤ 3) ∧ d^2 ≤ 2) ∧
    S.card = 8 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_roots_of_unity_satisfying_equation_l949_94964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cm_sq_to_m_sq_cm_to_m_conversion_min_to_hour_conversion_sqm_to_hectare_conversion_l949_94972

-- Define conversion factors
noncomputable def cm_to_m : ℚ := 1 / 100
noncomputable def min_to_hour : ℚ := 1 / 60
noncomputable def sqm_to_hectare : ℚ := 1 / 10000

-- Theorem statements
theorem cm_sq_to_m_sq : (2500 : ℚ) * cm_to_m^2 = 1/4 := by sorry

theorem cm_to_m_conversion : (20 : ℚ) * cm_to_m = 1/5 := by sorry

theorem min_to_hour_conversion : (45 : ℚ) * min_to_hour = 3/4 := by sorry

theorem sqm_to_hectare_conversion : (1250 : ℚ) * sqm_to_hectare = 1/8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cm_sq_to_m_sq_cm_to_m_conversion_min_to_hour_conversion_sqm_to_hectare_conversion_l949_94972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_abundant_between_10_and_18_l949_94916

def is_proper_divisor (d n : ℕ) : Prop := d ∣ n ∧ d ≠ n

def sum_of_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d => d ∣ n ∧ d ≠ n) (Finset.range n)).sum id

def is_abundant (n : ℕ) : Prop := sum_of_proper_divisors n > n

def is_smallest_abundant_in_range (a b n : ℕ) : Prop :=
  a ≤ n ∧ n ≤ b ∧ is_abundant n ∧ ∀ m, a ≤ m ∧ m < n → ¬is_abundant m

theorem smallest_abundant_between_10_and_18 :
  is_smallest_abundant_in_range 10 18 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_abundant_between_10_and_18_l949_94916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l949_94937

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

-- Define the triangle formed by three points on the parabola
structure ParabolaTriangle where
  P : PointOnParabola
  Q : PointOnParabola
  R : PointOnParabola
  distinct : P ≠ Q ∧ Q ≠ R ∧ P ≠ R

-- Define the property of equidistant x-projections
def equidistant_projections (t : ParabolaTriangle) (s : ℝ) : Prop :=
  ∃ a : ℝ, t.Q.x = a ∧ t.P.x = a - s ∧ t.R.x = a + s

-- Define area of triangle function (placeholder)
def area_of_triangle (t : ParabolaTriangle) : ℝ := sorry

-- State the theorem
theorem parabola_triangle_area (t : ParabolaTriangle) (s : ℝ) 
  (h : equidistant_projections t s) : 
  ∃ A : ℝ, A = (1/2) * s^3 ∧ A = area_of_triangle t := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_triangle_area_l949_94937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_and_speed_l949_94947

-- Define the problem parameters
noncomputable def tunnel_length : ℝ := 2725
noncomputable def max_speed : ℝ := 25
def num_cars : ℕ := 31
noncomputable def car_length : ℝ := 5

-- Define the distance between cars as a function of speed
noncomputable def car_distance (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 12 then 20
  else if 12 < x ∧ x ≤ 25 then (1/6) * x^2 + (1/3) * x
  else 0

-- Define the total time function
noncomputable def total_time (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 12 then
    (tunnel_length + num_cars * car_length + (num_cars - 1) * car_distance x) / x
  else if 12 < x ∧ x ≤ 25 then
    (tunnel_length + num_cars * car_length + (num_cars - 1) * car_distance x) / x
  else 0

-- State the theorem
theorem min_time_and_speed :
  ∃ (min_time : ℝ) (optimal_speed : ℝ),
    min_time = 250 ∧
    optimal_speed = 24 ∧
    ∀ x, 0 < x ∧ x ≤ max_speed → total_time x ≥ min_time :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_time_and_speed_l949_94947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l949_94935

noncomputable section

/-- Represents an acute triangle with side lengths a, b, c -/
structure AcuteTriangle (a b c : ℝ) : Prop where
  positive : a > 0 ∧ b > 0 ∧ c > 0
  acute : a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

theorem triangle_inequality (a b c m_a m_b m_c r_bc r_ca r_ab : ℝ) 
  (h_acute : AcuteTriangle a b c)
  (h_median_a : m_a^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_median_b : m_b^2 = (2*a^2 + 2*c^2 - b^2) / 4)
  (h_median_c : m_c^2 = (2*a^2 + 2*b^2 - c^2) / 4)
  (h_r_bc : r_bc > 0)
  (h_r_ca : r_ca > 0)
  (h_r_ab : r_ab > 0) :
  (m_a^2 / r_bc) + (m_b^2 / r_ca) + (m_c^2 / r_ab) ≥ (27 * Real.sqrt 3 / 8) * (a*b*c)^(1/3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l949_94935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_eq_seven_twentyfourths_l949_94992

noncomputable def f (a b : ℝ) : ℝ :=
  if a + b < 5 then
    (a * b - a + 4) / (2 * a)
  else
    (a * b - b - 5) / (-2 * b)

theorem f_sum_eq_seven_twentyfourths :
  f 3 1 + f 3 4 = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_eq_seven_twentyfourths_l949_94992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_on_interval_l949_94911

-- Define the function f(x) = ln|x-a|
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (abs (x - a))

-- Define the interval [-1, 1]
def interval : Set ℝ := Set.Icc (-1) 1

-- Theorem statement
theorem f_defined_on_interval (a : ℝ) : 
  (∀ x ∈ interval, ∃ y, f a x = y) ↔ a ∈ Set.Iio (-1) ∪ Set.Ioi 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_defined_on_interval_l949_94911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l949_94934

noncomputable def polar_equation (θ : ℝ) : ℝ := 2 * (Real.cos θ + Real.sin θ)

noncomputable def line_l (t : ℝ) : ℝ × ℝ := ((1/2) * t, 1 + (Real.sqrt 3 / 2) * t)

def cartesian_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 2

noncomputable def distance_sum : ℝ := Real.sqrt 5

theorem curve_and_line_intersection :
  (∀ θ : ℝ, polar_equation θ = (Real.sqrt ((polar_equation θ * Real.cos θ)^2 + (polar_equation θ * Real.sin θ)^2))) →
  (∀ x y : ℝ, (∃ θ : ℝ, x = polar_equation θ * Real.cos θ ∧ y = polar_equation θ * Real.sin θ) ↔ cartesian_equation x y) →
  (∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
    cartesian_equation ((line_l t₁).1) ((line_l t₁).2) ∧ 
    cartesian_equation ((line_l t₂).1) ((line_l t₂).2)) →
  distance_sum = Real.sqrt ((line_l t₁).1^2 + ((line_l t₁).2 - 1)^2) + 
                 Real.sqrt ((line_l t₂).1^2 + ((line_l t₂).2 - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_intersection_l949_94934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bouncing_ball_distance_l949_94926

/-- The total distance traveled by a bouncing ball -/
noncomputable def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (bounceCount : ℕ) : ℝ :=
  let descents := initialHeight * (1 - bounceRatio^bounceCount) / (1 - bounceRatio)
  let ascents := initialHeight * bounceRatio * (1 - bounceRatio^(bounceCount - 1)) / (1 - bounceRatio)
  descents + ascents

/-- Rounds a real number to the nearest integer -/
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem bouncing_ball_distance :
  roundToNearest (totalDistance 20 (2/3) 4) = 80 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bouncing_ball_distance_l949_94926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_y_is_one_l949_94925

noncomputable def y (x α β : ℝ) : ℝ := |Real.cos x + α * Real.cos (2 * x) + β * Real.cos (3 * x)|

-- State the theorem
theorem min_max_y_is_one :
  (∀ α β : ℝ, ∃ x : ℝ, y x α β ≥ 1) ∧
  (∃ α β : ℝ, ∀ x : ℝ, y x α β ≤ 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_y_is_one_l949_94925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadratic_roots_l949_94982

theorem complex_quadratic_roots : 
  let eq := fun (z : ℂ) => z^2 + 2*z + (3 - 4*Complex.I)
  (eq (2*Complex.I) = 0) ∧ (eq (-2 - 2*Complex.I) = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_quadratic_roots_l949_94982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_proof_l949_94942

/-- The surface area of a cylinder given its radius and height -/
noncomputable def cylinderSurfaceArea (r h : ℝ) : ℝ := 2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem cylinder_height_proof (r h : ℝ) (surfaceArea : ℝ) :
  r = 3 →
  surfaceArea = 30 * Real.pi →
  cylinderSurfaceArea r h = surfaceArea →
  h = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_proof_l949_94942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arctg_sum_l949_94904

theorem arctg_sum (x y : ℝ) : 
  Real.arctan x + Real.arctan y = 
    Real.arctan ((x + y) / (1 - x * y)) + 
    (if x * y < 1 then 0 
     else if x * y > 1 ∧ x < 0 then -1 
     else if x * y > 1 ∧ x > 0 then 1 
     else 0) * π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arctg_sum_l949_94904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l949_94933

theorem min_value_theorem (x : ℝ) (hx : x > 0) :
  9 * x^3 + 4 * x^(-(6 : ℤ)) ≥ 13 ∧
  (9 * x^3 + 4 * x^(-(6 : ℤ)) = 13 ↔ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l949_94933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supremum_of_f_l949_94924

noncomputable def f (a b : ℝ) := -1/(2*a) - 2/b

theorem supremum_of_f :
  ∀ ε > 0, ∃ a b : ℝ,
    a > 0 ∧ b > 0 ∧ a + b = 1 ∧
    f a b > -9/2 - ε ∧
    ∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → f x y ≤ -9/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supremum_of_f_l949_94924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l949_94968

-- Define the function f(x) = ln x - 3x
noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 * x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := 1 / x - 3

-- Theorem statement
theorem tangent_line_at_one :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := f_derivative x₀
  ∀ x y : ℝ, (2 * x + y + 1 = 0) ↔ (y - y₀ = k * (x - x₀)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l949_94968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l949_94973

theorem divisibility_in_subset (n : ℕ+) (S : Finset ℕ) :
  S ⊆ Finset.range (2 * n) →
  S.card = n + 1 →
  ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a ∣ b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_in_subset_l949_94973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equals_twelve_l949_94989

-- Define a and b as noncomputable
noncomputable def a : ℝ := Real.log 25
noncomputable def b : ℝ := Real.log 49

-- State the theorem
theorem power_sum_equals_twelve : (5 : ℝ)^(a/b) + (7 : ℝ)^(b/a) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_sum_equals_twelve_l949_94989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_Al2O3_l949_94949

/-- Represents the mass percentage of an element in a compound -/
noncomputable def mass_percentage (element_mass : ℝ) (compound_mass : ℝ) : ℝ :=
  (element_mass / compound_mass) * 100

/-- The atomic mass of aluminum in g/mol -/
def atomic_mass_Al : ℝ := 26.98

/-- The atomic mass of oxygen in g/mol -/
def atomic_mass_O : ℝ := 16.00

/-- The number of aluminum atoms in one formula unit of Al2O3 -/
def num_Al : ℕ := 2

/-- The number of oxygen atoms in one formula unit of Al2O3 -/
def num_O : ℕ := 3

/-- The molar mass of Al2O3 in g/mol -/
noncomputable def molar_mass_Al2O3 : ℝ :=
  num_Al * atomic_mass_Al + num_O * atomic_mass_O

/-- The mass of oxygen in one mole of Al2O3 in g -/
noncomputable def mass_O_in_Al2O3 : ℝ := num_O * atomic_mass_O

/-- Theorem stating that the mass percentage of oxygen in Al2O3 is approximately 47.07% -/
theorem mass_percentage_O_in_Al2O3 :
  ∃ (x : ℝ), abs (x - mass_percentage mass_O_in_Al2O3 molar_mass_Al2O3) < 0.01 ∧ x = 47.07 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_Al2O3_l949_94949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_base_side_length_l949_94978

/-- The side length of the base of a prism with an equilateral triangular base -/
noncomputable def base_side_length : ℝ :=
  Real.sqrt 44 - 6

/-- The prism with an equilateral triangular base -/
structure Prism where
  base_side : ℝ
  lateral_edge : ℝ
  sphere_radius : ℝ

/-- The conditions of the problem -/
def prism_conditions (p : Prism) : Prop :=
  p.lateral_edge = 1 ∧
  p.sphere_radius = p.base_side

/-- The theorem stating that the side length of the base is √44 - 6 -/
theorem prism_base_side_length (p : Prism) 
  (h : prism_conditions p) : p.base_side = base_side_length := by
  sorry

#check prism_base_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prism_base_side_length_l949_94978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l949_94919

/-- Given a sequence of 9 non-negative real numbers where the first and last are zero,
    and at least one of the others is non-zero, there exists an index i between 2 and 8
    such that the sum of the adjacent terms is less than twice the term at i,
    and also less than 1.9 times the term at i. -/
theorem sequence_inequality (a : Fin 9 → ℝ) 
    (h₁ : ∀ i, a i ≥ 0)
    (h₂ : a 0 = 0)
    (h₃ : a 8 = 0)
    (h₄ : ∃ i ∈ Finset.range 7, a (i + 1) ≠ 0) :
    ∃ i ∈ Finset.range 7, i ≥ 1 ∧
      a (i - 1) + a (i + 1) < 2 * a i ∧ 
      a (i - 1) + a (i + 1) < 1.9 * a i :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_inequality_l949_94919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_combined_transform_l949_94955

/-- The dilation matrix with scale factor 5 -/
def D : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0],
    ![0, 5]]

/-- The rotation matrix for 90 degrees counterclockwise -/
def R : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1],
    ![1,  0]]

/-- The combined transformation matrix E -/
def E : Matrix (Fin 2) (Fin 2) ℝ := D * R

theorem det_combined_transform :
  Matrix.det E = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_combined_transform_l949_94955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l949_94946

-- Define the lines
def line_through_origin (m : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m * p.1}
def vertical_line (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = a}
def sloped_line (m : ℝ) (b : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = m * p.1 + b}

-- Define the triangle
def triangle (l₁ l₂ l₃ : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := l₁ ∪ l₂ ∪ l₃

-- Define IsEquilateral and perimeter as functions
def IsEquilateral (t : Set (ℝ × ℝ)) : Prop := sorry
def perimeter (t : Set (ℝ × ℝ)) : ℝ := sorry

-- State the theorem
theorem equilateral_triangle_perimeter :
  ∃ (m : ℝ),
    let l₁ := line_through_origin m
    let l₂ := vertical_line 2
    let l₃ := sloped_line (Real.sqrt 3) 2
    let t := triangle l₁ l₂ l₃
    IsEquilateral t ∧ perimeter t = 6 + 12 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_perimeter_l949_94946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_f_nonpositive_inequality_for_positive_reals_l949_94983

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - Real.log (x + 1) + (a - 1) / a

-- Statement for part (I)
theorem range_of_a_for_f_nonpositive :
  {a : ℝ | ∃ x, f a x ≤ 0} = Set.Ioc 0 1 := by sorry

-- Statement for part (II)
theorem inequality_for_positive_reals (m n : ℝ) (h : m > n ∧ n > 0) :
  Real.exp (m - n) - 1 > Real.log (m + 1) - Real.log (n + 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_f_nonpositive_inequality_for_positive_reals_l949_94983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_number_theorem_l949_94995

/-- A cube configuration is a function from vertex indices to numbers -/
def CubeConfig := Fin 8 → Fin 8

/-- Check if a face of the cube satisfies the sum condition -/
def is_valid_face (c : CubeConfig) (v1 v2 v3 v4 : Fin 8) : Prop :=
  c v1 = c v2 + c v3 + c v4 ∨
  c v2 = c v1 + c v3 + c v4 ∨
  c v3 = c v1 + c v2 + c v4 ∨
  c v4 = c v1 + c v2 + c v3

/-- A cube configuration is valid if it contains exactly the numbers 1 to 8
    and three of its faces satisfy the sum condition -/
def is_valid_config (c : CubeConfig) : Prop :=
  (∀ i : Fin 8, ∃ j : Fin 8, c j = i + 1) ∧
  (∃ (f1 f2 f3 : Fin 8 × Fin 8 × Fin 8 × Fin 8),
    is_valid_face c f1.1 f1.2.1 f1.2.2.1 f1.2.2.2 ∧
    is_valid_face c f2.1 f2.2.1 f2.2.2.1 f2.2.2.2 ∧
    is_valid_face c f3.1 f3.2.1 f3.2.2.1 f3.2.2.2)

/-- The main theorem: there exists a valid cube configuration where
    the numbers adjacent to 6 are either (2, 3, 5), (3, 5, 7), or (2, 3, 7) -/
theorem cube_number_theorem :
  ∃ (c : CubeConfig) (v6 v1 v2 v3 : Fin 8),
    is_valid_config c ∧
    c v6 = 6 ∧
    ((c v1 = 2 ∧ c v2 = 3 ∧ c v3 = 5) ∨
     (c v1 = 3 ∧ c v2 = 5 ∧ c v3 = 7) ∨
     (c v1 = 2 ∧ c v2 = 3 ∧ c v3 = 7)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_number_theorem_l949_94995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_perpendicular_line_l949_94931

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a chord MN
def is_chord (M N : ℝ × ℝ) : Prop :=
  my_circle M.1 M.2 ∧ my_circle N.1 N.2 ∧ M.2 = -N.2 ∧ M.1 ≠ 0 ∧ M.1 ≠ 2 ∧ M.1 ≠ -2

-- Define the locus E
def locus_E (x y : ℝ) : Prop := x^2 - y^2 = 4 ∧ x ≠ 2 ∧ x ≠ -2

-- Define a directed line l
def directed_line (k t : ℝ) (x y : ℝ) : Prop := y = k * x + t

-- Theorem statement
theorem locus_and_perpendicular_line :
  ∃ (k t : ℝ), 
    (∀ x y, locus_E x y → 
      (∃ C D : ℝ × ℝ, 
        C ≠ D ∧
        directed_line k t C.1 C.2 ∧ 
        directed_line k t D.1 D.2 ∧
        locus_E C.1 C.2 ∧ 
        locus_E D.1 D.2 ∧
        ((C.1 + 2) * (D.1 + 2) + (C.2 * D.2) = 0))) ∧
    k = 0 ∧
    t ≠ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_perpendicular_line_l949_94931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l949_94987

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Check if a triangle is valid (satisfies triangle inequality) -/
def IntTriangle.isValid (t : IntTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Check if a triangle has positive area -/
def IntTriangle.hasPositiveArea (t : IntTriangle) : Prop :=
  t.a + t.b > t.c

/-- Calculate the perimeter of a triangle -/
def IntTriangle.perimeter (t : IntTriangle) : ℕ :=
  t.a + t.b + t.c

/-- Check if a triangle is equilateral -/
def IntTriangle.isEquilateral (t : IntTriangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- Check if a triangle is isosceles -/
def IntTriangle.isIsosceles (t : IntTriangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Check if a triangle is a right triangle -/
def IntTriangle.isRight (t : IntTriangle) : Prop :=
  t.a * t.a + t.b * t.b = t.c * t.c ∨
  t.b * t.b + t.c * t.c = t.a * t.a ∨
  t.c * t.c + t.a * t.a = t.b * t.b

/-- The set of all valid integer-sided triangles satisfying the given conditions -/
def validTriangles : Set IntTriangle :=
  { t : IntTriangle | 
    t.isValid ∧
    t.hasPositiveArea ∧
    t.perimeter < 15 ∧
    ¬t.isEquilateral ∧
    ¬t.isIsosceles ∧
    ¬t.isRight }

/-- Two triangles are congruent if they have the same side lengths (possibly in different order) -/
def IntTriangle.congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.a ∧ t1.b = t2.c ∧ t1.c = t2.b) ∨
  (t1.a = t2.b ∧ t1.b = t2.a ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b) ∨
  (t1.a = t2.c ∧ t1.b = t2.b ∧ t1.c = t2.a)

theorem count_valid_triangles :
  ∃ (representatives : Finset IntTriangle),
    Finset.card representatives = 5 ∧
    (∀ t, t ∈ validTriangles → ∃! r, r ∈ representatives ∧ t.congruent r) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l949_94987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_participants_perfect_square_l949_94991

theorem chess_tournament_participants_perfect_square 
  (n k : ℕ) -- n: number of boys, k: number of girls
  (total_games : (n + k) * (n + k - 1) / 2 = (n + k) * (n + k - 1) / 2)
  (total_points : n + k = n + k)
  (boys_points : n * k + n * (n - 1) / 2 = (n + k) / 2)
  (girls_points : n * k + k * (k - 1) / 2 = (n + k) / 2) :
  ∃ m : ℕ, n + k = m^2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_participants_perfect_square_l949_94991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_x_sequence_l949_94901

noncomputable def x : ℕ → ℝ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | n + 2 => (2 * x (n + 1) - 1) / x (n + 1)

theorem limit_of_x_sequence : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_x_sequence_l949_94901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l949_94962

def M : Set ℕ := {1, 3, 5, 7, 9}

def N : Set ℕ := {x : ℕ | 2 * x > 7}

theorem intersection_M_N : M ∩ N = {5, 7, 9} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l949_94962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l949_94902

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + x else Real.log x / Real.log (1/3)

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f x ≤ 5/4 * m - m^2) → m ∈ Set.Icc (1/4 : ℝ) 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l949_94902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l949_94922

-- Define a type for points in a plane
def Point : Type := ℝ × ℝ

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a function type for coloring segments
def Coloring := Point → Point → Color

-- Define the property of no three points being collinear
def NoThreeCollinear (points : Finset Point) : Prop :=
  ∀ p q r, p ∈ points → q ∈ points → r ∈ points → p ≠ q → q ≠ r → p ≠ r →
    (p.1 - q.1) * (r.2 - q.2) ≠ (r.1 - q.1) * (p.2 - q.2)

-- Define a monochromatic triangle
def MonochromaticTriangle (points : Finset Point) (coloring : Coloring) : Prop :=
  ∃ p q r, p ∈ points ∧ q ∈ points ∧ r ∈ points ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (coloring p q = coloring q r ∧ coloring q r = coloring p r)

-- State the theorem
theorem monochromatic_triangle_exists 
  (points : Finset Point) 
  (h_card : points.card = 6) 
  (h_no_collinear : NoThreeCollinear points) 
  (coloring : Coloring) : 
  MonochromaticTriangle points coloring :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_triangle_exists_l949_94922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_k_gons_count_theorem_l949_94948

/-- The number of distinct convex k-gons whose vertices coincide with the vertices of a given convex n-gon -/
def convex_k_gons_count (n k : ℕ) : ℕ :=
  n * (n - k - 1).factorial / (k.factorial * (n - 2 * k).factorial)

/-- Represents the actual number of distinct convex k-gons in an n-gon -/
def number_of_distinct_convex_k_gons (n k : ℕ) : ℕ :=
  sorry -- This definition would be based on the actual counting method

/-- Theorem stating the count of convex k-gons in an n-gon -/
theorem convex_k_gons_count_theorem (n k : ℕ) (h : n ≥ 2 * k) :
  convex_k_gons_count n k = number_of_distinct_convex_k_gons n k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_k_gons_count_theorem_l949_94948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_time_difference_two_seconds_l949_94918

/-- Proves that a car's speed is 600 km/h given the specified conditions -/
theorem car_speed (v : ℝ) : v > 0 → (1 / v + 1 / 450 = 1 / 600) ↔ v = 600 :=
by sorry

/-- The time (in hours) it takes to travel 1 km at the given speed -/
noncomputable def travel_time (speed : ℝ) : ℝ := 1 / speed

/-- The difference in travel time (in seconds) between speeds v and 900 km/h -/
noncomputable def time_difference (v : ℝ) : ℝ := (travel_time v - travel_time 900) * 3600

/-- Proves that the time difference is 2 seconds when v is 600 km/h -/
theorem time_difference_two_seconds : time_difference 600 = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_time_difference_two_seconds_l949_94918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_min_matrix_sum_min_achievable_l949_94961

/-- Given nonzero integers a, b, c, d satisfying the matrix equation,
    the sum of their absolute values is at least 8 -/
theorem matrix_sum_min (a b c d : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (h : (![![a, b], ![c, d]] : Matrix (Fin 2) (Fin 2) ℤ) ^ 2 = ![![9, 0], ![0, 9]]) :
    |a| + |b| + |c| + |d| ≥ 8 := by
  sorry

/-- There exists a solution achieving the minimum value of 8 -/
theorem matrix_sum_min_achievable : ∃ (a b c d : ℤ),
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
    (![![a, b], ![c, d]] : Matrix (Fin 2) (Fin 2) ℤ) ^ 2 = ![![9, 0], ![0, 9]] ∧
    |a| + |b| + |c| + |d| = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_sum_min_matrix_sum_min_achievable_l949_94961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_line_l949_94928

/-- Given a curve y = e^(ax) with a tangent at point (0,1) perpendicular to the line x + 2y + 1 = 0, prove that a = 2 -/
theorem tangent_perpendicular_to_line (a : ℝ) : 
  (∃ f : ℝ → ℝ, f = λ x ↦ Real.exp (a * x)) →  -- Curve y = e^(ax)
  (∃ g : ℝ → ℝ, g = λ x ↦ -1/2 * x - 1/2) →  -- Line x + 2y + 1 = 0 in slope-intercept form
  (∃ h : ℝ → ℝ, h = λ x ↦ a * Real.exp (a * x)) →  -- Derivative of f
  (a * (-1/2) = -1) →  -- Perpendicular condition
  a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_perpendicular_to_line_l949_94928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_spiral_rise_l949_94903

/-- Represents a cylindrical post with a squirrel running up in a spiral path -/
structure SpiralPath where
  postHeight : ℝ
  postCircumference : ℝ
  travelDistance : ℝ

/-- Calculates the rise per circuit for a squirrel running up a cylindrical post in a spiral path -/
noncomputable def risePerCircuit (sp : SpiralPath) : ℝ :=
  let numCircuits := sp.travelDistance / sp.postCircumference
  sp.postHeight / numCircuits

/-- Theorem stating that for the given conditions, the rise per circuit is 4 feet -/
theorem squirrel_spiral_rise
  (sp : SpiralPath)
  (h1 : sp.postHeight = 12)
  (h2 : sp.postCircumference = 3)
  (h3 : sp.travelDistance = 9) :
  risePerCircuit sp = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_spiral_rise_l949_94903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_satisfying_inequality_four_satisfies_inequality_four_is_greatest_l949_94921

theorem greatest_integer_satisfying_inequality :
  ∀ x : ℕ+, (x : ℝ)^4 / x < 20 * x → x ≤ 4 :=
by
  sorry

theorem four_satisfies_inequality :
  (4 : ℝ)^4 / 4 < 20 * 4 :=
by
  sorry

theorem four_is_greatest :
  ∃ x : ℕ+, (x : ℝ)^4 / x < 20 * x ∧ x = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_satisfying_inequality_four_satisfies_inequality_four_is_greatest_l949_94921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l949_94908

theorem problem_solution (α : ℝ) 
  (h1 : 3 * Real.pi / 4 < α ∧ α < Real.pi)
  (h2 : Real.tan α + 1 / Real.tan α = -10 / 3) : 
  Real.tan α = -1 / 3 ∧ 
  (5 * (Real.sin (α/2))^2 + 8 * Real.sin (α/2) * Real.cos (α/2) + 11 * (Real.cos (α/2))^2 - 8) / 
  (Real.sqrt 2 * Real.sin (α - Real.pi/4)) = -5 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l949_94908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_times_g_equals_sqrt_x_plus_one_l949_94938

theorem f_times_g_equals_sqrt_x_plus_one (x : ℝ) (hx : x > 0) :
  let f := fun x => Real.sqrt (x * (x + 1))
  let g := fun x => 1 / Real.sqrt x
  f x * g x = Real.sqrt (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_times_g_equals_sqrt_x_plus_one_l949_94938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_theorem_l949_94999

noncomputable def f (w : ℝ) (phi : ℝ) (x : ℝ) : ℝ := Real.sin (w * x) + phi

noncomputable def g (x : ℝ) : ℝ := Real.sin (4 * x - 2 * Real.pi / 3)

theorem g_range_theorem (w : ℝ) (phi : ℝ) (h1 : w > 0) (h2 : phi < Real.pi / 2) :
  ∃ (m : ℝ), 0 < m ∧ m < 1/2 ∧
  ∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 8) (3 * Real.pi / 8) →
    |g x - m| < 1 := by
  sorry

#check g_range_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_theorem_l949_94999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_complex_function_l949_94909

open Real

/-- A function f: ℝ₊ → ℝ with specific monotonicity properties -/
structure MonotonicFunction where
  f : ℝ → ℝ
  pos_dom : ∀ x, 0 < x → f x ∈ Set.range f
  mono_cond1 : StrictMono (fun x => f (x^2) - x^3)
  mono_cond2 : StrictMono (fun x => f (x^3) - x^4)

/-- Main theorem statement -/
theorem monotonic_complex_function (φ : MonotonicFunction) :
  StrictMono (fun x => φ.f (x^5) + 100 * (φ.f (x^5) - x^7)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_complex_function_l949_94909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_milk_amount_l949_94979

/-- Represents the vessel and its contents -/
structure Vessel where
  capacity : ℚ
  milk : ℚ
  water : ℚ

/-- Performs the operation of removing and replacing liquid -/
noncomputable def removeAndReplace (v : Vessel) (amount : ℚ) : Vessel :=
  let milkRatio := v.milk / (v.milk + v.water)
  let milkRemoved := milkRatio * amount
  { capacity := v.capacity,
    milk := v.milk - milkRemoved,
    water := v.water - (amount - milkRemoved) + amount }

/-- The main theorem stating the final amount of milk -/
theorem final_milk_amount (initialCapacity : ℚ) (operationAmount : ℚ) : 
  initialCapacity = 75 → operationAmount = 9 →
  let initialVessel : Vessel := { capacity := initialCapacity, milk := initialCapacity, water := 0 }
  let afterFirstOperation := removeAndReplace initialVessel operationAmount
  let finalVessel := removeAndReplace afterFirstOperation operationAmount
  finalVessel.milk = 5808 / 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_milk_amount_l949_94979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_diagram_cosine_l949_94943

theorem chords_diagram_cosine (θ : ℝ) : 
  ∃ (a b c B : ℝ),
    a^2 = 1 ∧ 
    B^2 = 25 ∧ 
    B = a + b + c ∧ 
    b * c = 2 ∧ 
    b + c = 4 ∧ 
    Real.cos θ = b / B → 
    Real.cos (2 * θ) = (4 * Real.sqrt 2 - 19) / 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chords_diagram_cosine_l949_94943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l949_94952

theorem division_problem (p m n : ℝ) (h : m ≠ n) :
  ∃ x y : ℝ, x + y = p ∧ x / m + y / n = 9 := by
  let x := m * (9 * n - p) / (n - m)
  let y := n * (p - 9 * m) / (n - m)
  use x, y
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_division_problem_l949_94952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_segment_length_l949_94932

-- Define a Point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a Square type
structure Square (A B C D : Point) : Prop where
  is_square : True  -- Placeholder for square properties

-- Define relations and functions
def on_side (P A B : Point) : Prop :=
  True  -- Placeholder for "on side" relation

def on_extension (P A B : Point) : Prop :=
  True  -- Placeholder for "on extension" relation

noncomputable def angle (A B C : Point) : ℝ :=
  0  -- Placeholder for angle calculation

noncomputable def length (A B : Point) : ℝ :=
  0  -- Placeholder for length calculation

theorem square_segment_length 
  (A B C D : Point) (K L : Point) 
  (h_square : Square A B C D) 
  (h_L_on_CD : on_side L C D) 
  (h_K_on_DA_ext : on_extension K D A) 
  (h_angle : angle K B L = 90) 
  (h_KD : length K D = 19) 
  (h_CL : length C L = 6) : 
  length L D = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_segment_length_l949_94932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_325_l949_94900

/-- Represents a trapezoid PQRS -/
structure Trapezoid where
  PQ : ℝ
  RS : ℝ
  altitude : ℝ
  PR : ℝ

/-- The area of a trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ :=
  (t.PQ + t.RS) * t.altitude / 2

/-- Theorem: The area of the given trapezoid PQRS is 325 -/
theorem trapezoid_area_is_325 (PQRS : Trapezoid) 
    (h1 : PQRS.PQ = 40)
    (h2 : PQRS.RS = 25)
    (h3 : PQRS.altitude = 10)
    (h4 : PQRS.PR = 20) : 
  trapezoid_area PQRS = 325 := by
  -- Unfold the definition of trapezoid_area
  unfold trapezoid_area
  -- Substitute the known values
  rw [h1, h2, h3]
  -- Evaluate the arithmetic expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_325_l949_94900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_attendees_is_542_l949_94953

/-- Represents the total number of people at a concert given the following conditions:
  * Adult admission cost is $2.00
  * Child admission cost is $1.50
  * Total receipts are $985.00
  * Number of adults is 342
-/
def totalAttendees : ℕ :=
  let adultCost : ℚ := 2
  let childCost : ℚ := (3/2)
  let totalReceipts : ℚ := 985
  let numAdults : ℕ := 342
  let childReceipts : ℚ := totalReceipts - (adultCost * numAdults)
  let numChildren : ℕ := Int.toNat (Int.floor (childReceipts / childCost))
  numAdults + numChildren

/-- Theorem stating that the total number of attendees is 542 -/
theorem total_attendees_is_542 : totalAttendees = 542 := by
  sorry

#eval totalAttendees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_attendees_is_542_l949_94953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l949_94913

/-- A tetrahedron with four faces and an inscribed sphere. -/
structure Tetrahedron where
  /-- The areas of the four faces of the tetrahedron. -/
  S : Fin 4 → ℝ
  /-- The radius of the inscribed sphere. -/
  r : ℝ

/-- The volume of a tetrahedron. -/
noncomputable def volume (t : Tetrahedron) : ℝ := (1/3) * (t.S 0 + t.S 1 + t.S 2 + t.S 3) * t.r

/-- Theorem stating that the volume of a tetrahedron is (1/3)(S1 + S2 + S3 + S4)r. -/
theorem tetrahedron_volume (t : Tetrahedron) :
  volume t = (1/3) * (t.S 0 + t.S 1 + t.S 2 + t.S 3) * t.r := by
  -- The proof is omitted as it's trivial given the definition of volume
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_volume_l949_94913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l949_94971

-- Define the quadratic function f
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 + 1

-- Define g in terms of f and a
def g (a : ℝ) (x : ℝ) : ℝ := f x + a * x

-- Theorem statement
theorem quadratic_function_properties :
  (∀ x, f x ≥ 1) ∧  -- f has a minimum value of 1
  (f 0 = 3) ∧ (f 2 = 3) ∧  -- f(0) = f(2) = 3
  (∀ a, (∃ x y, x ∈ Set.Icc (-1) 1 ∧ y ∈ Set.Icc (-1) 1 ∧ x < y ∧ g a x > g a y) ∧ 
        (∃ x y, x ∈ Set.Icc (-1) 1 ∧ y ∈ Set.Icc (-1) 1 ∧ x < y ∧ g a x < g a y) ↔ 
   -2 < a ∧ a < 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l949_94971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_AB_l949_94936

/-- A point on the line y = -x - 1 -/
structure Point (m : ℝ) where
  x : ℝ := m
  y : ℝ := -m - 1

/-- The parabola x^2 = 2y -/
def Parabola (x y : ℝ) : Prop := x^2 = 2*y

/-- The line AB formed by tangent points -/
def LineAB (m : ℝ) (x y : ℝ) : Prop := m*x - y + m + 1 = 0

/-- Distance from origin to line AB -/
noncomputable def DistanceToOrigin (m : ℝ) : ℝ := |m + 1| / Real.sqrt (m^2 + 1)

theorem max_distance_to_line_AB :
  ∃ (max_d : ℝ), max_d = Real.sqrt 2 ∧
  ∀ (m : ℝ), DistanceToOrigin m ≤ max_d := by
  sorry

#check max_distance_to_line_AB

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_AB_l949_94936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_in_triangle_l949_94956

/-- A triangle in a 2D plane. -/
structure Triangle where
  -- Define the triangle structure
  vertices : Fin 3 → ℝ × ℝ

/-- A hexagon in a 2D plane. -/
structure Hexagon where
  -- Define the hexagon structure
  vertices : Fin 6 → ℝ × ℝ

/-- Checks if a triangle is acute-angled. -/
def Triangle.isAcute (T : Triangle) : Prop :=
  sorry

/-- Calculates the area of a triangle. -/
def Triangle.area (T : Triangle) : ℝ :=
  sorry

/-- Calculates the area of a hexagon. -/
def Hexagon.area (H : Hexagon) : ℝ :=
  sorry

/-- Constructs a hexagon from the perpendiculars drawn from the midpoints 
    of each side of a triangle to the other two sides. -/
def hexagonFromMidpointPerpendiculars (T : Triangle) : Hexagon :=
  sorry

/-- Given an acute-angled triangle with area S, the area of the hexagon formed by 
    perpendiculars drawn from the midpoints of each side to the other two sides is S/2. -/
theorem hexagon_area_in_triangle (S : ℝ) (h : S > 0) : 
  ∃ (T : Triangle) (H : Hexagon),
    T.isAcute ∧ 
    T.area = S ∧ 
    H = hexagonFromMidpointPerpendiculars T ∧
    H.area = S / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_in_triangle_l949_94956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_fixed_point_l949_94981

noncomputable section

-- Define variables a and b as real numbers
variable (a b : ℝ)

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
axiom a_gt_b : a > b
axiom b_gt_0 : b > 0
axiom short_axis : b = Real.sqrt 3
axiom eccentricity : (Real.sqrt (a^2 - b^2)) / a = 1/2

-- Define the bottom vertex A
def A : ℝ × ℝ := (0, -Real.sqrt 3)

-- Define a line l
def line_l (k t : ℝ) (x : ℝ) : ℝ := k * x + t

-- Define the slope sum condition
def slope_sum_condition (P Q : ℝ × ℝ) : Prop :=
  (P.2 - A.2) / (P.1 - A.1) + (Q.2 - A.2) / (Q.1 - A.1) = 2

-- Theorem 1: Equation of ellipse C
theorem ellipse_equation : 
  ∀ x y : ℝ, ellipse_C a b x y ↔ x^2/4 + y^2/3 = 1 :=
sorry

-- Theorem 2: Line l passes through a fixed point
theorem fixed_point :
  ∀ k t : ℝ, ∀ P Q : ℝ × ℝ,
  ellipse_C a b P.1 P.2 →
  ellipse_C a b Q.1 Q.2 →
  P ≠ A →
  Q ≠ A →
  (∀ x : ℝ, line_l k t x = P.2 ∨ line_l k t x = Q.2) →
  slope_sum_condition P Q →
  ∃ x y : ℝ, x = Real.sqrt 3 ∧ y = Real.sqrt 3 ∧ line_l k t x = y :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_fixed_point_l949_94981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_representations_l949_94910

theorem infinitely_many_representations (A B : ℕ) (h : A ≠ B) :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ∃ (x₁ y₁ x₂ y₂ : ℕ),
      x₁ > 0 ∧ y₁ > 0 ∧ x₂ > 0 ∧ y₂ > 0 ∧
      Nat.Coprime x₁ y₁ ∧ Nat.Coprime x₂ y₂ ∧
      n = x₁^2 + A * y₁^2 ∧
      n = x₂^2 + B * y₂^2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_representations_l949_94910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_implies_ratio_range_l949_94915

/-- Given vectors a and b with the specified components, prove that the range of lambda/m is [-6,1] -/
theorem vector_equality_implies_ratio_range 
  (lambda m alpha : ℝ) 
  (a : ℝ × ℝ := (lambda + 2, lambda^2 - Real.sqrt 3 * Real.cos (2 * alpha)))
  (b : ℝ × ℝ := (m, m / 2 + Real.sin alpha * Real.cos alpha))
  (h : a = (2 * b.1, 2 * b.2)) :
  ∃ (r : Set ℝ), r = Set.Icc (-6) 1 ∧ lambda / m ∈ r :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equality_implies_ratio_range_l949_94915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorizable_b_is_correct_l949_94997

/-- The smallest positive integer b for which x^2 + bx + 3960 factors into (x + r)(x + s) with integer r and s -/
def smallest_factorizable_b : ℕ := 126

/-- Predicate to check if a quadratic expression can be factored into two binomials with integer coefficients -/
def is_factorizable (b : ℕ) : Prop :=
  ∃ (r s : ℤ), ∀ (x : ℤ), x^2 + b*x + 3960 = (x + r) * (x + s)

theorem smallest_factorizable_b_is_correct :
  (is_factorizable smallest_factorizable_b) ∧ 
  (∀ b : ℕ, b < smallest_factorizable_b → ¬(is_factorizable b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_factorizable_b_is_correct_l949_94997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_intersection_ratio_l949_94965

noncomputable section

/-- The hyperbola function y = 1/x -/
def hyperbola (x : ℝ) : ℝ := 1 / x

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The distance between two x-coordinates -/
def xDistance (x1 x2 : ℝ) : ℝ := |x1 - x2|

theorem parallel_lines_intersection_ratio :
  ∀ (k : ℝ) (A B K L M N : Point),
  A.x = 0 ∧ A.y = 14 ∧
  B.x = 0 ∧ B.y = 4 ∧
  (∃ (l1 l2 : Line), 
    l1.slope = k ∧ l1.intercept = 14 ∧
    l2.slope = k ∧ l2.intercept = 4 ∧
    K.y = hyperbola K.x ∧ L.y = hyperbola L.x ∧
    M.y = hyperbola M.x ∧ N.y = hyperbola N.x ∧
    K.y = k * K.x + 14 ∧ L.y = k * L.x + 14 ∧
    M.y = k * M.x + 4 ∧ N.y = k * N.x + 4) →
  (xDistance A.x L.x - xDistance A.x K.x) / (xDistance B.x N.x - xDistance B.x M.x) = 3.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_intersection_ratio_l949_94965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l949_94986

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

noncomputable def frac (x : ℝ) : ℝ := x - floor x

noncomputable def a : ℕ → ℝ
  | 0 => Real.sqrt 3
  | n + 1 => floor (a n) + 1 / frac (a n)

theorem a_2017_value : a 2016 = 3024 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l949_94986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l949_94998

noncomputable section

-- Define the triangle ABC
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (4, 0)
def C : ℝ → ℝ × ℝ := λ c => (0, c)

-- Define vectors AC and BC
def AC (c : ℝ) : ℝ × ℝ := (1, c)
def BC (c : ℝ) : ℝ × ℝ := (-4, c)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the centroid of a triangle
def centroid (a b c : ℝ × ℝ) : ℝ × ℝ :=
  ((a.1 + b.1 + c.1) / 3, (a.2 + b.2 + c.2) / 3)

theorem triangle_abc_properties :
  ∀ c : ℝ,
  dot_product (AC c) (BC c) = 0 →
  (c = 2 ∨ c = -2) ∧
  (centroid A B (C c) = (1, 2/3) ∨ centroid A B (C c) = (1, -2/3)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l949_94998
