import Mathlib

namespace NUMINAMATH_CALUDE_perimeter_F₂MN_is_8_l2899_289918

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define the foci F₁ and F₂
variable (F₁ F₂ : ℝ × ℝ)

-- Define points M and N on the ellipse
variable (M N : ℝ × ℝ)

-- Axiom: F₁ and F₂ are foci of the ellipse C
axiom foci_of_C : ∀ (x y : ℝ), C x y → ∃ (a : ℝ), dist (x, y) F₁ + dist (x, y) F₂ = 2 * a

-- Axiom: M and N are on the ellipse C
axiom M_on_C : C M.1 M.2
axiom N_on_C : C N.1 N.2

-- Axiom: M, N, and F₁ are collinear
axiom collinear_MNF₁ : ∃ (t : ℝ), N = F₁ + t • (M - F₁)

-- Theorem: The perimeter of triangle F₂MN is 8
theorem perimeter_F₂MN_is_8 : dist M N + dist M F₂ + dist N F₂ = 8 := by sorry

end NUMINAMATH_CALUDE_perimeter_F₂MN_is_8_l2899_289918


namespace NUMINAMATH_CALUDE_evaluate_expression_l2899_289912

theorem evaluate_expression : (3 : ℝ)^4 - 4 * (3 : ℝ)^2 = 45 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2899_289912


namespace NUMINAMATH_CALUDE_lucy_fish_count_l2899_289950

theorem lucy_fish_count (initial_fish : ℝ) (bought_fish : ℝ) : 
  initial_fish = 212.0 → bought_fish = 280.0 → initial_fish + bought_fish = 492.0 := by
  sorry

end NUMINAMATH_CALUDE_lucy_fish_count_l2899_289950


namespace NUMINAMATH_CALUDE_combined_fuel_efficiency_l2899_289937

/-- Calculates the combined fuel efficiency of two cars -/
theorem combined_fuel_efficiency
  (efficiency1 : ℝ) -- Fuel efficiency of the first car in miles per gallon
  (efficiency2 : ℝ) -- Fuel efficiency of the second car in miles per gallon
  (h1 : efficiency1 = 40) -- Given: Ray's car averages 40 miles per gallon
  (h2 : efficiency2 = 10) -- Given: Tom's car averages 10 miles per gallon
  (distance : ℝ) -- Distance driven by each car
  (h3 : distance > 0) -- Assumption: Distance driven is positive
  : (2 * distance) / ((distance / efficiency1) + (distance / efficiency2)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_combined_fuel_efficiency_l2899_289937


namespace NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2899_289923

theorem fraction_equality_implies_numerator_equality 
  {x y m : ℝ} (h1 : m ≠ 0) (h2 : x / m = y / m) : x = y :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_numerator_equality_l2899_289923


namespace NUMINAMATH_CALUDE_XY₂_atomic_numbers_l2899_289910

/-- Represents an element in the periodic table -/
structure Element where
  atomic_number : ℕ
  charge : ℤ
  group : ℕ

/-- Represents an ionic compound -/
structure IonicCompound where
  metal : Element
  nonmetal : Element
  metal_count : ℕ
  nonmetal_count : ℕ

/-- The XY₂ compound -/
def XY₂ : IonicCompound :=
  { metal := { atomic_number := 12, charge := 2, group := 2 },
    nonmetal := { atomic_number := 9, charge := -1, group := 17 },
    metal_count := 1,
    nonmetal_count := 2 }

theorem XY₂_atomic_numbers :
  XY₂.metal.atomic_number = 12 ∧ XY₂.nonmetal.atomic_number = 9 :=
by sorry

end NUMINAMATH_CALUDE_XY₂_atomic_numbers_l2899_289910


namespace NUMINAMATH_CALUDE_inequality_proof_l2899_289900

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  (x + y + z)^2 / 3 ≥ x * Real.sqrt (y * z) + y * Real.sqrt (z * x) + z * Real.sqrt (x * y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2899_289900


namespace NUMINAMATH_CALUDE_preimage_of_five_one_l2899_289915

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 + p.2, 2 * p.1 - p.2)

-- Theorem statement
theorem preimage_of_five_one :
  ∃ (p : ℝ × ℝ), f p = (5, 1) ∧ p = (2, 3) :=
by
  sorry

end NUMINAMATH_CALUDE_preimage_of_five_one_l2899_289915


namespace NUMINAMATH_CALUDE_seven_swimmer_race_outcomes_l2899_289920

/-- The number of different possible outcomes for 1st-2nd-3rd place in a race with n swimmers and no ties -/
def race_outcomes (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- Theorem: The number of different possible outcomes for 1st-2nd-3rd place in a race with 7 swimmers and no ties is 210 -/
theorem seven_swimmer_race_outcomes : race_outcomes 7 = 210 := by
  sorry

end NUMINAMATH_CALUDE_seven_swimmer_race_outcomes_l2899_289920


namespace NUMINAMATH_CALUDE_atlantic_call_rate_l2899_289952

/-- Proves that the additional charge per minute for Atlantic Call is $0.20 -/
theorem atlantic_call_rate (united_base_rate : ℚ) (united_per_minute : ℚ) 
  (atlantic_base_rate : ℚ) (minutes : ℕ) :
  united_base_rate = 7 →
  united_per_minute = 0.25 →
  atlantic_base_rate = 12 →
  minutes = 100 →
  united_base_rate + united_per_minute * minutes = 
    atlantic_base_rate + (atlantic_base_rate + united_per_minute * minutes - united_base_rate) / minutes →
  (atlantic_base_rate + united_per_minute * minutes - united_base_rate) / minutes = 0.20 := by
  sorry

#check atlantic_call_rate

end NUMINAMATH_CALUDE_atlantic_call_rate_l2899_289952


namespace NUMINAMATH_CALUDE_paige_goldfish_l2899_289902

/-- The number of goldfish Paige initially raised -/
def initial_goldfish : ℕ := sorry

/-- The number of catfish Paige initially raised -/
def initial_catfish : ℕ := 12

/-- The number of fish that disappeared -/
def disappeared_fish : ℕ := 4

/-- The number of fish left -/
def remaining_fish : ℕ := 15

theorem paige_goldfish :
  initial_goldfish = 7 :=
by sorry

end NUMINAMATH_CALUDE_paige_goldfish_l2899_289902


namespace NUMINAMATH_CALUDE_lion_path_theorem_l2899_289980

/-- A broken line path within a circle -/
structure BrokenLinePath where
  points : List (Real × Real)
  inside_circle : ∀ p ∈ points, p.1^2 + p.2^2 ≤ 100

/-- The total length of a broken line path -/
def pathLength (path : BrokenLinePath) : Real :=
  sorry

/-- The sum of turning angles in a broken line path -/
def sumTurningAngles (path : BrokenLinePath) : Real :=
  sorry

/-- Main theorem: If a broken line path within a circle of radius 10 meters
    has a total length of 30,000 meters, then the sum of all turning angles
    along the path is at least 2998 radians -/
theorem lion_path_theorem (path : BrokenLinePath) 
    (h : pathLength path = 30000) :
  sumTurningAngles path ≥ 2998 := by
  sorry

end NUMINAMATH_CALUDE_lion_path_theorem_l2899_289980


namespace NUMINAMATH_CALUDE_fibonacci_square_equality_l2899_289919

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_square_equality :
  ∃! n : ℕ, n > 0 ∧ fib n = n^2 ∧ n = 12 := by sorry

end NUMINAMATH_CALUDE_fibonacci_square_equality_l2899_289919


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2899_289927

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for B ⊆ A
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem for A ∩ B = ∅
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2899_289927


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l2899_289975

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

theorem non_shaded_perimeter (outer : Rectangle) (attached : Rectangle) (shaded : Rectangle) 
  (h_outer_width : outer.width = 12)
  (h_outer_height : outer.height = 10)
  (h_attached_width : attached.width = 3)
  (h_attached_height : attached.height = 4)
  (h_shaded_width : shaded.width = 3)
  (h_shaded_height : shaded.height = 5)
  (h_shaded_area : area shaded = 120)
  (h_shaded_center : shaded.width < outer.width ∧ shaded.height < outer.height) :
  ∃ (non_shaded : Rectangle), perimeter non_shaded = 19 := by
sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l2899_289975


namespace NUMINAMATH_CALUDE_dot_product_range_l2899_289996

theorem dot_product_range (a b : ℝ × ℝ) : 
  let norm_a := Real.sqrt (a.1^2 + a.2^2)
  let angle := Real.arccos ((b.1 * (a.1 - b.1) + b.2 * (a.2 - b.2)) / 
    (Real.sqrt (b.1^2 + b.2^2) * Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)))
  norm_a = 2 ∧ angle = 2 * Real.pi / 3 →
  2 - 4 * Real.sqrt 3 / 3 ≤ a.1 * b.1 + a.2 * b.2 ∧ 
  a.1 * b.1 + a.2 * b.2 ≤ 2 + 4 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l2899_289996


namespace NUMINAMATH_CALUDE_max_perimeter_right_triangle_l2899_289946

theorem max_perimeter_right_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = c^2) (h5 : c = 5) :
  a + b + c ≤ 5 + 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_max_perimeter_right_triangle_l2899_289946


namespace NUMINAMATH_CALUDE_choose_leaders_count_l2899_289991

/-- Represents the number of members in each category -/
structure ClubMembers where
  senior_boys : Nat
  junior_boys : Nat
  senior_girls : Nat
  junior_girls : Nat

/-- Calculates the number of ways to choose a president and vice-president -/
def choose_leaders (members : ClubMembers) : Nat :=
  let boys_combinations := members.senior_boys * members.junior_boys * 2
  let girls_combinations := members.senior_girls * members.junior_girls * 2
  boys_combinations + girls_combinations

/-- Theorem stating the number of ways to choose leaders under given conditions -/
theorem choose_leaders_count (members : ClubMembers) 
  (h1 : members.senior_boys = 6)
  (h2 : members.junior_boys = 6)
  (h3 : members.senior_girls = 6)
  (h4 : members.junior_girls = 6) :
  choose_leaders members = 144 := by
  sorry

#eval choose_leaders ⟨6, 6, 6, 6⟩

end NUMINAMATH_CALUDE_choose_leaders_count_l2899_289991


namespace NUMINAMATH_CALUDE_expected_groups_formula_l2899_289945

/-- A sequence of k zeros and m ones arranged in random order -/
structure BinarySequence where
  k : ℕ
  m : ℕ

/-- The expected number of alternating groups in a BinarySequence -/
noncomputable def expectedGroups (seq : BinarySequence) : ℝ :=
  1 + (2 * seq.k * seq.m : ℝ) / (seq.k + seq.m)

/-- Theorem stating the expected number of alternating groups -/
theorem expected_groups_formula (seq : BinarySequence) :
    expectedGroups seq = 1 + (2 * seq.k * seq.m : ℝ) / (seq.k + seq.m) := by
  sorry

end NUMINAMATH_CALUDE_expected_groups_formula_l2899_289945


namespace NUMINAMATH_CALUDE_triangle_inequality_and_side_length_relations_l2899_289964

/-- Given a triangle with side lengths a, b, c, prove the existence of a triangle
    with side lengths √a, √b, √c and the inequality involving these lengths. -/
theorem triangle_inequality_and_side_length_relations
  (a b c : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a + b > c)
  (hbc : b + c > a)
  (hca : c + a > b) :
  (∃ (x y z : ℝ), x = Real.sqrt a ∧ y = Real.sqrt b ∧ z = Real.sqrt c ∧
    x + y > z ∧ y + z > x ∧ z + x > y) ∧
  Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c ∧
  a + b + c < 2 * (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_and_side_length_relations_l2899_289964


namespace NUMINAMATH_CALUDE_log_monotonic_l2899_289913

-- Define the logarithmic function
noncomputable def log (x : ℝ) : ℝ := Real.log x

-- State the theorem
theorem log_monotonic : 
  ∀ x y : ℝ, x > 0 → y > 0 → x < y → log x < log y :=
by
  sorry

end NUMINAMATH_CALUDE_log_monotonic_l2899_289913


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2899_289934

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define the complement of B
def complement_B : Set ℝ := {x | ¬ (x ∈ B)}

-- State the theorem
theorem intersection_A_complement_B : A ∩ complement_B = {x | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2899_289934


namespace NUMINAMATH_CALUDE_coin_toss_problem_l2899_289960

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem coin_toss_problem (n : ℕ) :
  binomial_probability n 3 0.5 = 0.25 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_problem_l2899_289960


namespace NUMINAMATH_CALUDE_camel_count_theorem_l2899_289967

/-- Represents the number of humps on a camel -/
inductive CamelType
  | dromedary : CamelType  -- one hump
  | bactrian : CamelType   -- two humps

/-- Calculate the number of humps for a given camel type -/
def humps (c : CamelType) : Nat :=
  match c with
  | .dromedary => 1
  | .bactrian => 2

/-- A group of camels -/
structure CamelGroup where
  dromedaryCount : Nat
  bactrianCount : Nat

/-- Calculate the total number of humps in a camel group -/
def totalHumps (g : CamelGroup) : Nat :=
  g.dromedaryCount * humps CamelType.dromedary + g.bactrianCount * humps CamelType.bactrian

/-- Calculate the total number of feet in a camel group -/
def totalFeet (g : CamelGroup) : Nat :=
  (g.dromedaryCount + g.bactrianCount) * 4

/-- Calculate the total number of camels in a group -/
def totalCamels (g : CamelGroup) : Nat :=
  g.dromedaryCount + g.bactrianCount

theorem camel_count_theorem (g : CamelGroup) :
  totalHumps g = 23 → totalFeet g = 60 → totalCamels g = 15 := by
  sorry

end NUMINAMATH_CALUDE_camel_count_theorem_l2899_289967


namespace NUMINAMATH_CALUDE_sampling_methods_correct_l2899_289940

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

-- Define a scenario
structure Scenario where
  total_population : ℕ
  sample_size : ℕ
  has_distinct_groups : Bool
  is_ordered : Bool

-- Define a function to determine the most suitable sampling method
def most_suitable_method (s : Scenario) : SamplingMethod :=
  if s.total_population ≤ 15 then SamplingMethod.SimpleRandom
  else if s.has_distinct_groups then SamplingMethod.Stratified
  else if s.is_ordered then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

-- Theorem to prove
theorem sampling_methods_correct :
  (most_suitable_method ⟨15, 5, false, false⟩ = SamplingMethod.SimpleRandom) ∧
  (most_suitable_method ⟨240, 20, true, false⟩ = SamplingMethod.Stratified) ∧
  (most_suitable_method ⟨950, 25, false, true⟩ = SamplingMethod.Systematic) :=
by sorry

end NUMINAMATH_CALUDE_sampling_methods_correct_l2899_289940


namespace NUMINAMATH_CALUDE_parentheses_removal_l2899_289926

theorem parentheses_removal (a b c : ℝ) : -3*a - (2*b - c) = -3*a - 2*b + c := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l2899_289926


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2899_289914

/-- Represents an ellipse with equation x²/16 + y²/9 = 1 -/
structure StandardEllipse where
  a : ℝ := 4
  b : ℝ := 3

/-- Represents a point on the ellipse -/
structure EllipsePoint where
  x : ℝ
  y : ℝ

/-- Represents a focus of the ellipse -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Theorem: The perimeter of triangle DEF₂ is 16 -/
theorem ellipse_triangle_perimeter
  (e : StandardEllipse)
  (F₁ F F₂ : Focus)
  (D E : EllipsePoint)
  (h1 : F₁.x < F.x) -- F₁ is the left focus
  (h2 : F₂ = F) -- F₂ is the right focus
  (h3 : D.x^2/16 + D.y^2/9 = 1) -- D is on the ellipse
  (h4 : E.x^2/16 + E.y^2/9 = 1) -- E is on the ellipse
  (h5 : ∃ (t : ℝ), D.x = (1-t)*F₁.x + t*E.x ∧ D.y = (1-t)*F₁.y + t*E.y) -- DE passes through F₁
  : (abs (D.x - F₁.x) + abs (D.y - F₁.y)) + 
    (abs (D.x - F₂.x) + abs (D.y - F₂.y)) +
    (abs (E.x - F₁.x) + abs (E.y - F₁.y)) + 
    (abs (E.x - F₂.x) + abs (E.y - F₂.y)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l2899_289914


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_one_neg_two_l2899_289943

/-- Given that the terminal side of angle α passes through point P(1,-2),
    prove that sin α = -2√5/5 -/
theorem sin_alpha_for_point_one_neg_two (α : Real) :
  (∃ (P : ℝ × ℝ), P = (1, -2) ∧ P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  Real.sin α = -2 * Real.sqrt 5 / 5 := by
sorry


end NUMINAMATH_CALUDE_sin_alpha_for_point_one_neg_two_l2899_289943


namespace NUMINAMATH_CALUDE_tangent_line_slope_logarithm_inequality_l2899_289905

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Theorem for the tangent line
theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = k * x₀ ∧ (deriv f) x₀ = k) ↔ k = 1 / Real.exp 1 :=
sorry

-- Theorem for the inequality
theorem logarithm_inequality (a x : ℝ) (ha : a ≥ 1) (hx : x > 0) :
  f x ≤ a * x + (a - 1) / x - 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_logarithm_inequality_l2899_289905


namespace NUMINAMATH_CALUDE_power_sum_equality_l2899_289921

theorem power_sum_equality : (3^2)^3 + (2^3)^2 = 793 := by sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2899_289921


namespace NUMINAMATH_CALUDE_solution_equivalence_l2899_289998

/-- Given prime numbers p and q with p < q, the positive integer solutions (x, y) to 
    1/x + 1/y = 1/p - 1/q are equivalent to the positive integer solutions of 
    ((q - p)x - pq)((q - p)y - pq) = p^2q^2 -/
theorem solution_equivalence (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p < q) :
  ∀ x y : ℕ, x > 0 ∧ y > 0 →
  (1 / x + 1 / y = 1 / p - 1 / q) ↔ 
  ((q - p) * x - p * q) * ((q - p) * y - p * q) = p^2 * q^2 :=
by sorry

end NUMINAMATH_CALUDE_solution_equivalence_l2899_289998


namespace NUMINAMATH_CALUDE_combined_output_fraction_l2899_289982

/-- Represents the production rate of a machine relative to a base rate -/
structure ProductionRate :=
  (rate : ℚ)

/-- Represents a machine with its production rate -/
structure Machine :=
  (name : String)
  (rate : ProductionRate)

/-- The problem setup with four machines and their relative production rates -/
def production_problem (t n o p : Machine) : Prop :=
  t.rate.rate = 4 / 3 * n.rate.rate ∧
  n.rate.rate = 3 / 2 * o.rate.rate ∧
  o.rate = p.rate

/-- The theorem stating that machines N and P produce 6/13 of the total output -/
theorem combined_output_fraction 
  (t n o p : Machine) 
  (h : production_problem t n o p) : 
  (n.rate.rate + p.rate.rate) / (t.rate.rate + n.rate.rate + o.rate.rate + p.rate.rate) = 6 / 13 :=
sorry

end NUMINAMATH_CALUDE_combined_output_fraction_l2899_289982


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2899_289993

theorem smallest_integer_with_remainders (n : ℕ) : 
  n > 1 ∧ 
  n % 5 = 1 ∧ 
  n % 7 = 1 ∧ 
  n % 8 = 1 ∧ 
  (∀ m : ℕ, m > 1 → m % 5 = 1 → m % 7 = 1 → m % 8 = 1 → n ≤ m) →
  n = 281 ∧ 240 < n ∧ n < 359 := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2899_289993


namespace NUMINAMATH_CALUDE_final_worker_count_l2899_289966

/-- Represents the number of bees in a hive -/
structure BeeHive where
  workers : ℕ
  drones : ℕ
  queen : ℕ

def initial_hive : BeeHive := { workers := 400, drones := 75, queen := 1 }

def bees_leave (hive : BeeHive) (workers_leaving : ℕ) (drones_leaving : ℕ) : BeeHive :=
  { workers := hive.workers - workers_leaving,
    drones := hive.drones - drones_leaving,
    queen := hive.queen }

def workers_return (hive : BeeHive) (returning_workers : ℕ) : BeeHive :=
  { workers := hive.workers + returning_workers,
    drones := hive.drones,
    queen := hive.queen }

theorem final_worker_count :
  let hive1 := bees_leave initial_hive 28 12
  let hive2 := workers_return hive1 15
  hive2.workers = 387 := by sorry

end NUMINAMATH_CALUDE_final_worker_count_l2899_289966


namespace NUMINAMATH_CALUDE_smallest_page_number_l2899_289999

theorem smallest_page_number : ∃ n : ℕ, n > 0 ∧ 4 ∣ n ∧ 13 ∣ n ∧ ∀ m : ℕ, (m > 0 ∧ 4 ∣ m ∧ 13 ∣ m) → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_page_number_l2899_289999


namespace NUMINAMATH_CALUDE_least_possible_y_l2899_289939

theorem least_possible_y (x y z : ℤ) 
  (h_x_even : Even x)
  (h_y_odd : Odd y)
  (h_z_odd : Odd z)
  (h_y_minus_x : y - x > 5)
  (h_z_minus_x : ∀ w : ℤ, (Odd w ∧ w - x ≥ 9) → z - x ≤ w - x) : 
  y ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_y_l2899_289939


namespace NUMINAMATH_CALUDE_magic_square_base_is_three_l2899_289941

/-- Represents a 3x3 magic square with elements in base b -/
def MagicSquare (b : ℕ) : Type :=
  Fin 3 → Fin 3 → ℕ

/-- The sum of a row, column, or diagonal in the magic square -/
def MagicSum (b : ℕ) (square : MagicSquare b) : ℕ :=
  square 0 0 + square 0 1 + square 0 2

/-- Predicate to check if a given square is magic -/
def IsMagicSquare (b : ℕ) (square : MagicSquare b) : Prop :=
  (∀ i : Fin 3, square i 0 + square i 1 + square i 2 = MagicSum b square) ∧
  (∀ j : Fin 3, square 0 j + square 1 j + square 2 j = MagicSum b square) ∧
  (square 0 0 + square 1 1 + square 2 2 = MagicSum b square) ∧
  (square 0 2 + square 1 1 + square 2 0 = MagicSum b square)

/-- The specific magic square given in the problem -/
def GivenSquare (b : ℕ) : MagicSquare b :=
  fun i j => match i, j with
  | 0, 0 => 5
  | 0, 1 => 11
  | 0, 2 => 15
  | 1, 0 => 4
  | 1, 1 => 11
  | 1, 2 => 12
  | 2, 0 => 14
  | 2, 1 => 2
  | 2, 2 => 3

theorem magic_square_base_is_three :
  ∃ (b : ℕ), b > 1 ∧ IsMagicSquare b (GivenSquare b) ∧ b = 3 :=
sorry

end NUMINAMATH_CALUDE_magic_square_base_is_three_l2899_289941


namespace NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_7_l2899_289936

theorem largest_integer_less_than_100_remainder_5_mod_7 : 
  ∀ n : ℤ, n < 100 ∧ n % 7 = 5 → n ≤ 96 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_100_remainder_5_mod_7_l2899_289936


namespace NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l2899_289929

theorem greatest_four_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 9996 ∧ 
  n % 17 = 0 ∧ 
  n ≤ 9999 ∧ 
  ∀ m : ℕ, m % 17 = 0 ∧ m ≤ 9999 → m ≤ n := by
sorry

end NUMINAMATH_CALUDE_greatest_four_digit_multiple_of_17_l2899_289929


namespace NUMINAMATH_CALUDE_mario_moving_sidewalk_time_l2899_289948

/-- The time it takes Mario to walk from A to B on a moving sidewalk -/
theorem mario_moving_sidewalk_time (d : ℝ) (w : ℝ) (v : ℝ) : 
  d > 0 ∧ w > 0 ∧ v > 0 →  -- distances and speeds are positive
  d / w = 90 →             -- time to walk when sidewalk is off
  d / v = 45 →             -- time to be carried without walking
  d / (w + v) = 30 :=      -- time to walk on moving sidewalk
by sorry

end NUMINAMATH_CALUDE_mario_moving_sidewalk_time_l2899_289948


namespace NUMINAMATH_CALUDE_difference_of_squares_l2899_289932

theorem difference_of_squares (a b : ℕ) (h1 : a + b = 60) (h2 : a - b = 18) : a^2 - b^2 = 1080 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2899_289932


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2899_289985

theorem solution_set_inequality (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) < 0} = {x : ℝ | a < x ∧ x < 1/a} := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2899_289985


namespace NUMINAMATH_CALUDE_last_digit_of_3_power_10_l2899_289904

theorem last_digit_of_3_power_10 : ∃ n : ℕ, 3^10 ≡ 9 [ZMOD 10] :=
sorry

end NUMINAMATH_CALUDE_last_digit_of_3_power_10_l2899_289904


namespace NUMINAMATH_CALUDE_max_triangle_area_is_85_l2899_289983

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane -/
structure Line where
  slope : ℝ
  point : Point

/-- Triangle formed by three lines -/
structure Triangle where
  l1 : Line
  l2 : Line
  l3 : Line

/-- Rotation of a line around its point -/
def rotate (l : Line) (angle : ℝ) : Line :=
  sorry

/-- Area of a triangle formed by three lines -/
def triangleArea (t : Triangle) : ℝ :=
  sorry

/-- Maximum area of triangle formed by rotating lines -/
def maxTriangleArea (l1 l2 l3 : Line) : ℝ :=
  sorry

theorem max_triangle_area_is_85 :
  let a := Point.mk 0 0
  let b := Point.mk 11 0
  let c := Point.mk 18 0
  let la := Line.mk 1 a
  let lb := Line.mk 0 b  -- Vertical line represented with slope 0
  let lc := Line.mk (-1) c
  maxTriangleArea la lb lc = 85 := by
  sorry

end NUMINAMATH_CALUDE_max_triangle_area_is_85_l2899_289983


namespace NUMINAMATH_CALUDE_lcm_of_three_numbers_specific_lcm_l2899_289956

theorem lcm_of_three_numbers (a b c : ℕ) (hcf : ℕ) (h_hcf : Nat.gcd a (Nat.gcd b c) = hcf) :
  Nat.lcm a (Nat.lcm b c) = a * b * c / hcf :=
by sorry

theorem specific_lcm :
  Nat.lcm 136 (Nat.lcm 144 168) = 411264 :=
by
  have h_hcf : Nat.gcd 136 (Nat.gcd 144 168) = 8 := by sorry
  exact lcm_of_three_numbers 136 144 168 8 h_hcf

end NUMINAMATH_CALUDE_lcm_of_three_numbers_specific_lcm_l2899_289956


namespace NUMINAMATH_CALUDE_largest_four_digit_multiple_of_9_with_digit_sum_27_l2899_289917

/-- Returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Returns true if n is a four-digit number, false otherwise -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem largest_four_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : ℕ, is_four_digit n → n % 9 = 0 → digit_sum n = 27 → n ≤ 9990 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_multiple_of_9_with_digit_sum_27_l2899_289917


namespace NUMINAMATH_CALUDE_train_average_speed_l2899_289909

-- Define the points and distances
def x : ℝ := 0
def y : ℝ := sorry
def z : ℝ := sorry

-- Define the speeds
def speed_xy : ℝ := 300
def speed_yz : ℝ := 100

-- State the theorem
theorem train_average_speed :
  -- Conditions
  (y - x = 2 * (z - y)) →  -- Distance from x to y is twice the distance from y to z
  -- Conclusion
  (z - x) / ((y - x) / speed_xy + (z - y) / speed_yz) = 180 :=
by
  sorry

end NUMINAMATH_CALUDE_train_average_speed_l2899_289909


namespace NUMINAMATH_CALUDE_meaningful_expression_l2899_289924

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 3)) ↔ x > 3 := by
sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2899_289924


namespace NUMINAMATH_CALUDE_square_side_length_l2899_289997

/-- The perimeter of an equilateral triangle with side length s -/
def triangle_perimeter (s : ℝ) : ℝ := 3 * s

/-- The perimeter of a square with side length s -/
def square_perimeter (s : ℝ) : ℝ := 4 * s

/-- The side length of the equilateral triangle -/
def triangle_side : ℝ := 12

theorem square_side_length :
  ∃ (s : ℝ), s = 9 ∧ square_perimeter s = triangle_perimeter triangle_side :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l2899_289997


namespace NUMINAMATH_CALUDE_disjunction_false_implies_negation_true_l2899_289949

variable (p q : Prop)

theorem disjunction_false_implies_negation_true :
  (¬(p ∨ q) → ¬p) ∧ ¬(¬p → ¬(p ∨ q)) := by sorry

end NUMINAMATH_CALUDE_disjunction_false_implies_negation_true_l2899_289949


namespace NUMINAMATH_CALUDE_unique_subset_with_nonempty_intersection_l2899_289953

def A : Set ℕ := {1, 2, 3, 4, 5, 6}
def B : Set ℕ := {4, 5, 6, 7, 8}

theorem unique_subset_with_nonempty_intersection :
  ∃! S : Set ℕ, S ⊆ A ∧ S ∩ B ≠ ∅ ∧ S = {5, 6} := by sorry

end NUMINAMATH_CALUDE_unique_subset_with_nonempty_intersection_l2899_289953


namespace NUMINAMATH_CALUDE_triangle_existence_l2899_289994

/-- Given a square area t, segment length 2s, and angle α, 
    this theorem states the existence condition for a triangle 
    with area t, perimeter 2s, and one angle α. -/
theorem triangle_existence 
  (t s : ℝ) (α : Real) 
  (h_t : t > 0) (h_s : s > 0) (h_α : 0 < α ∧ α < π) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = 2 * s ∧
    1/2 * a * b * Real.sin α = t ∧
    ∃ (β γ : Real), 
      β > 0 ∧ γ > 0 ∧
      α + β + γ = π ∧
      a / Real.sin α = b / Real.sin β ∧
      b / Real.sin β = c / Real.sin γ :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l2899_289994


namespace NUMINAMATH_CALUDE_odot_properties_l2899_289970

/-- The custom operation ⊙ -/
def odot (a : ℝ) (x y : ℝ) : ℝ := 18 + x - a * y

/-- Theorem stating the properties of the ⊙ operation -/
theorem odot_properties :
  ∃ a : ℝ, (odot a 2 3 = 8) ∧ (odot a 3 5 = 1) ∧ (odot a 5 3 = 11) := by
  sorry

end NUMINAMATH_CALUDE_odot_properties_l2899_289970


namespace NUMINAMATH_CALUDE_beam_equation_l2899_289977

/-- The equation for buying beams problem -/
theorem beam_equation (x : ℕ+) (h : x > 1) : 
  (3 : ℚ) * ((x : ℚ) - 1) = 6210 / (x : ℚ) :=
sorry

/-- The total cost of beams in wen -/
def total_cost : ℕ := 6210

/-- The transportation cost per beam in wen -/
def transport_cost : ℕ := 3

/-- The number of beams that can be bought -/
def num_beams : ℕ+ := sorry

end NUMINAMATH_CALUDE_beam_equation_l2899_289977


namespace NUMINAMATH_CALUDE_power_difference_divisibility_l2899_289959

theorem power_difference_divisibility (a b : ℤ) (h : 100 ∣ (a - b)) :
  10000 ∣ (a^100 - b^100) := by
  sorry

end NUMINAMATH_CALUDE_power_difference_divisibility_l2899_289959


namespace NUMINAMATH_CALUDE_unique_triple_l2899_289989

def is_infinite_repeating_decimal (a b c : ℕ+) : Prop :=
  (a + b / 9 : ℚ)^2 = c + 7/9

def fraction_is_integer (c a : ℕ+) : Prop :=
  ∃ k : ℤ, (c + a : ℚ) / (c - a) = k

theorem unique_triple : 
  ∃! (a b c : ℕ+), 
    b < 10 ∧ 
    is_infinite_repeating_decimal a b c ∧ 
    fraction_is_integer c a ∧
    a = 1 ∧ b = 6 ∧ c = 2 := by sorry

end NUMINAMATH_CALUDE_unique_triple_l2899_289989


namespace NUMINAMATH_CALUDE_factorization_of_a_squared_minus_4ab_l2899_289968

theorem factorization_of_a_squared_minus_4ab (a b : ℝ) :
  a^2 - 4*a*b = a*(a - 4*b) := by sorry

end NUMINAMATH_CALUDE_factorization_of_a_squared_minus_4ab_l2899_289968


namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l2899_289990

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < a ∧ 0 < b
  h_a_ge_b : a ≥ b

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The theorem stating the property of the ellipse -/
theorem ellipse_focus_distance (e : Ellipse) (p : PointOnEllipse e) 
  (h_focus1 : ℝ) (h_on_ellipse : e.a = 5 ∧ e.b = 4) :
  h_focus1 = 8 → ∃ h_focus2 : ℝ, h_focus2 = 2 ∧ h_focus1 + h_focus2 = 2 * e.a := by
  sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l2899_289990


namespace NUMINAMATH_CALUDE_stratified_sampling_problem_l2899_289992

/-- Calculates the number of people to be selected from a stratum in stratified sampling -/
def stratified_sample_size (total_population : ℕ) (stratum_size : ℕ) (total_sample_size : ℕ) : ℕ :=
  (total_sample_size * stratum_size) / total_population

/-- The problem statement -/
theorem stratified_sampling_problem (total_population : ℕ) (stratum_size : ℕ) (total_sample_size : ℕ) 
  (h1 : total_population = 360) 
  (h2 : stratum_size = 108) 
  (h3 : total_sample_size = 20) :
  stratified_sample_size total_population stratum_size total_sample_size = 6 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_problem_l2899_289992


namespace NUMINAMATH_CALUDE_slower_bike_speed_l2899_289971

theorem slower_bike_speed 
  (distance : ℝ) 
  (fast_speed : ℝ) 
  (time_difference : ℝ) 
  (h1 : distance = 960) 
  (h2 : fast_speed = 64) 
  (h3 : time_difference = 1) :
  ∃ (slow_speed : ℝ), 
    slow_speed > 0 ∧ 
    distance / slow_speed = distance / fast_speed + time_difference ∧ 
    slow_speed = 60 := by
sorry

end NUMINAMATH_CALUDE_slower_bike_speed_l2899_289971


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2899_289930

theorem perpendicular_lines_a_values (a : ℝ) : 
  (∀ x y : ℝ, a^2 * x + 2 * y + 1 = 0 → x - a * y - 2 = 0 → 
   (a^2 * 1 + 2 * (-a) = 0)) → 
  (a = 2 ∨ a = 0) := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_values_l2899_289930


namespace NUMINAMATH_CALUDE_least_multiple_of_35_greater_than_450_l2899_289906

theorem least_multiple_of_35_greater_than_450 : ∀ n : ℕ, n > 0 ∧ 35 ∣ n ∧ n > 450 → n ≥ 455 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_35_greater_than_450_l2899_289906


namespace NUMINAMATH_CALUDE_average_age_problem_l2899_289955

theorem average_age_problem (age_a age_b age_c : ℝ) :
  age_b = 20 →
  (age_a + age_c) / 2 = 29 →
  (age_a + age_b + age_c) / 3 = 26 := by
sorry

end NUMINAMATH_CALUDE_average_age_problem_l2899_289955


namespace NUMINAMATH_CALUDE_sector_radius_l2899_289976

theorem sector_radius (θ : ℝ) (L : ℝ) (R : ℝ) :
  θ = 60 → L = π → L = (θ * π * R) / 180 → R = 3 :=
by sorry

end NUMINAMATH_CALUDE_sector_radius_l2899_289976


namespace NUMINAMATH_CALUDE_new_shoes_duration_l2899_289933

/-- Given information about shoe costs and durability, prove the duration of new shoes. -/
theorem new_shoes_duration (used_repair_cost : ℝ) (used_duration : ℝ) (new_cost : ℝ) (cost_increase_percentage : ℝ) :
  used_repair_cost = 13.50 →
  used_duration = 1 →
  new_cost = 32.00 →
  cost_increase_percentage = 0.1852 →
  let new_duration := new_cost / (used_repair_cost * (1 + cost_increase_percentage))
  new_duration = 2 := by
  sorry

end NUMINAMATH_CALUDE_new_shoes_duration_l2899_289933


namespace NUMINAMATH_CALUDE_math_majors_consecutive_probability_l2899_289962

/-- The number of people sitting at the round table -/
def total_people : ℕ := 10

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of ways to choose seats for math majors -/
def total_ways : ℕ := Nat.choose total_people math_majors

/-- The number of ways math majors can sit consecutively -/
def consecutive_ways : ℕ := total_people

/-- The probability that all math majors sit in consecutive seats -/
def probability : ℚ := consecutive_ways / total_ways

theorem math_majors_consecutive_probability :
  probability = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_math_majors_consecutive_probability_l2899_289962


namespace NUMINAMATH_CALUDE_retailer_items_sold_l2899_289986

/-- The problem of determining the number of items sold by a retailer -/
theorem retailer_items_sold 
  (profit_per_item : ℝ) 
  (profit_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (min_items_with_discount : ℝ) : 
  profit_per_item = 30 →
  profit_percentage = 0.16 →
  discount_percentage = 0.05 →
  min_items_with_discount = 156.86274509803923 →
  ∃ (items_sold : ℕ), items_sold = 100 := by
  sorry

end NUMINAMATH_CALUDE_retailer_items_sold_l2899_289986


namespace NUMINAMATH_CALUDE_quadratic_root_square_relation_l2899_289995

theorem quadratic_root_square_relation (c : ℝ) : 
  (c > 0) →
  (∃ x₁ x₂ : ℝ, (8 * x₁^2 - 6 * x₁ + 9 * c^2 = 0) ∧ 
                (8 * x₂^2 - 6 * x₂ + 9 * c^2 = 0) ∧ 
                (x₂ = x₁^2)) →
  (c = 1/3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_square_relation_l2899_289995


namespace NUMINAMATH_CALUDE_cookies_per_sitting_l2899_289958

/-- The number of times Theo eats cookies per day -/
def eats_per_day : ℕ := 3

/-- The number of days Theo eats cookies per month -/
def days_per_month : ℕ := 20

/-- The total number of cookies Theo eats in 3 months -/
def total_cookies : ℕ := 2340

/-- The number of months considered -/
def months : ℕ := 3

/-- Theorem stating the number of cookies Theo can eat in one sitting -/
theorem cookies_per_sitting :
  total_cookies / (eats_per_day * days_per_month * months) = 13 := by sorry

end NUMINAMATH_CALUDE_cookies_per_sitting_l2899_289958


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2899_289984

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 12) * (x^2 + 6*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2899_289984


namespace NUMINAMATH_CALUDE_smallest_class_size_l2899_289979

theorem smallest_class_size : ∃ n : ℕ, n > 0 ∧ 
  n % 6 = 3 ∧ 
  n % 8 = 5 ∧ 
  n % 9 = 7 ∧ 
  ∀ m : ℕ, m > 0 → m % 6 = 3 → m % 8 = 5 → m % 9 = 7 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_class_size_l2899_289979


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l2899_289965

def S : Finset ℕ := {3, 5, 9, 10, 12, 15}

theorem largest_lcm_with_15 : 
  Finset.max (S.image (fun x => Nat.lcm 15 x)) = some 60 := by sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l2899_289965


namespace NUMINAMATH_CALUDE_remainder_of_b_86_mod_50_l2899_289972

theorem remainder_of_b_86_mod_50 : (7^86 + 9^86) % 50 = 40 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_b_86_mod_50_l2899_289972


namespace NUMINAMATH_CALUDE_equal_points_per_game_l2899_289951

/-- 
Given a player who scores a total of 36 points in 3 games, 
with points equally distributed among the games,
prove that the player scores 12 points in each game.
-/
theorem equal_points_per_game 
  (total_points : ℕ) 
  (num_games : ℕ) 
  (h1 : total_points = 36) 
  (h2 : num_games = 3) 
  (h3 : total_points % num_games = 0) : 
  total_points / num_games = 12 := by
sorry


end NUMINAMATH_CALUDE_equal_points_per_game_l2899_289951


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2899_289935

theorem inequality_equivalence (x : ℝ) : 
  (1/2: ℝ) ^ (x^2 - 2*x + 3) < (1/2 : ℝ) ^ (2*x^2 + 3*x - 3) ↔ -6 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2899_289935


namespace NUMINAMATH_CALUDE_intersection_range_l2899_289944

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line
def line (k x : ℝ) (y : ℝ) : Prop := y = k*x - 2

-- Define the condition for intersection
def intersection_condition (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line k x y ∧
    ∃ (x' y' : ℝ), circle_C x' y' ∧
      (x - x')^2 + (y - y')^2 ≤ 4

-- Theorem statement
theorem intersection_range :
  ∀ k : ℝ, intersection_condition k ↔ 0 ≤ k ∧ k ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_range_l2899_289944


namespace NUMINAMATH_CALUDE_solve_logarithmic_equation_l2899_289978

-- Define the base-10 logarithm function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem solve_logarithmic_equation :
  ∃ x : ℝ, log10 (3 * x + 4) = 1 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_logarithmic_equation_l2899_289978


namespace NUMINAMATH_CALUDE_center_on_line_common_chord_condition_external_tangent_length_l2899_289911

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C_k (k x y : ℝ) : Prop := (x - k)^2 + (y - Real.sqrt 3 * k)^2 = 4

-- Define the line y = √3x
def line_sqrt3 (x y : ℝ) : Prop := y = Real.sqrt 3 * x

-- Theorem 1: The center of circle C_k always lies on the line y = √3x
theorem center_on_line (k : ℝ) : line_sqrt3 k (Real.sqrt 3 * k) := by sorry

-- Theorem 2: If the common chord length is √15/2, then k = ±1 or k = ±3/4
theorem common_chord_condition (k : ℝ) : 
  (∃ x y : ℝ, circle_O x y ∧ circle_C_k k x y ∧ 
   (x^2 + y^2 = 1 - (15/16))) → 
  (k = 1 ∨ k = -1 ∨ k = 3/4 ∨ k = -3/4) := by sorry

-- Theorem 3: When k = ±3/2, the length of the external common tangent is 2√2
theorem external_tangent_length : 
  ∀ k : ℝ, (k = 3/2 ∨ k = -3/2) → 
  (∃ x1 y1 x2 y2 : ℝ, 
    circle_O x1 y1 ∧ circle_C_k k x2 y2 ∧
    ((x2 - x1)^2 + (y2 - y1)^2 = 8)) := by sorry

end NUMINAMATH_CALUDE_center_on_line_common_chord_condition_external_tangent_length_l2899_289911


namespace NUMINAMATH_CALUDE_product_inequality_l2899_289954

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4*a/(b+c)) * (1 + 4*b/(c+a)) * (1 + 4*c/(a+b)) > 25 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l2899_289954


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2899_289903

/-- A sector that is one-third of a circle --/
structure ThirdCircleSector where
  /-- The radius of the full circle --/
  R : ℝ
  /-- Assumption that R is positive --/
  R_pos : 0 < R

/-- An inscribed circle in the sector --/
structure InscribedCircle (S : ThirdCircleSector) where
  /-- The radius of the inscribed circle --/
  r : ℝ
  /-- Assumption that r is positive --/
  r_pos : 0 < r

/-- The theorem stating the radius of the inscribed circle --/
theorem inscribed_circle_radius (S : ThirdCircleSector) (C : InscribedCircle S) 
    (h : S.R = 6) : C.r = 6 * Real.sqrt 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2899_289903


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l2899_289981

theorem largest_x_sqrt_3x_eq_5x : 
  ∃ (x_max : ℚ), x_max = 3/25 ∧ 
  (∀ x : ℚ, x ≥ 0 → (Real.sqrt (3 * x) = 5 * x) → x ≤ x_max) ∧
  (Real.sqrt (3 * x_max) = 5 * x_max) := by
  sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_l2899_289981


namespace NUMINAMATH_CALUDE_clubsuit_difference_l2899_289942

/-- The clubsuit operation -/
def clubsuit (x y : ℝ) : ℝ := 4*x + 6*y

/-- Theorem stating that (5 ♣ 3) - (1 ♣ 4) = 10 -/
theorem clubsuit_difference : (clubsuit 5 3) - (clubsuit 1 4) = 10 := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_difference_l2899_289942


namespace NUMINAMATH_CALUDE_remainder_three_to_27_mod_13_l2899_289901

theorem remainder_three_to_27_mod_13 : 3^27 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_to_27_mod_13_l2899_289901


namespace NUMINAMATH_CALUDE_distance_to_point_l2899_289922

/-- The distance from the origin to the point (12, 5) on the line y = 5/12 x is 13 -/
theorem distance_to_point : 
  let point : ℝ × ℝ := (12, 5)
  let line (x : ℝ) : ℝ := (5/12) * x
  (point.2 = line point.1) →
  Real.sqrt ((point.1 - 0)^2 + (point.2 - 0)^2) = 13 :=
by sorry

end NUMINAMATH_CALUDE_distance_to_point_l2899_289922


namespace NUMINAMATH_CALUDE_tree_shadow_length_l2899_289987

/-- Given a person and a tree casting shadows, this theorem calculates the length of the tree's shadow. -/
theorem tree_shadow_length 
  (person_height : ℝ) 
  (person_shadow : ℝ) 
  (tree_height : ℝ) 
  (h1 : person_height = 1.5)
  (h2 : person_shadow = 0.5)
  (h3 : tree_height = 30) :
  ∃ (tree_shadow : ℝ), tree_shadow = 10 ∧ 
    person_height / person_shadow = tree_height / tree_shadow :=
by sorry

end NUMINAMATH_CALUDE_tree_shadow_length_l2899_289987


namespace NUMINAMATH_CALUDE_sqrt_product_sqrt_l2899_289938

theorem sqrt_product_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_sqrt_l2899_289938


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2899_289947

theorem complex_fraction_equality : Complex.I * 5 / (1 - Complex.I * 2) = -2 + Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2899_289947


namespace NUMINAMATH_CALUDE_second_diff_constant_correct_y_value_l2899_289908

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the sequence of x values
def x_seq (x₁ d : ℝ) (n : ℕ) : ℝ := x₁ + n * d

-- Define the sequence of y values
def y_seq (a b c x₁ d : ℝ) (n : ℕ) : ℝ := quadratic a b c (x_seq x₁ d n)

-- Define the first difference sequence
def delta_seq (a b c x₁ d : ℝ) (n : ℕ) : ℝ :=
  y_seq a b c x₁ d (n + 1) - y_seq a b c x₁ d n

-- Define the second difference sequence
def delta2_seq (a b c x₁ d : ℝ) (n : ℕ) : ℝ :=
  delta_seq a b c x₁ d (n + 1) - delta_seq a b c x₁ d n

-- Theorem: The second difference is constant
theorem second_diff_constant (a b c x₁ d : ℝ) (h : a ≠ 0) :
  ∃ k, ∀ n, delta2_seq a b c x₁ d n = k :=
sorry

-- Given y values
def given_y_values : List ℝ := [51, 107, 185, 285, 407, 549, 717]

-- Find the incorrect y value and its correct value
def find_incorrect_y (ys : List ℝ) : Option (ℕ × ℝ) :=
sorry

-- Theorem: The identified incorrect y value is 549 and should be 551
theorem correct_y_value :
  find_incorrect_y given_y_values = some (5, 551) :=
sorry

end NUMINAMATH_CALUDE_second_diff_constant_correct_y_value_l2899_289908


namespace NUMINAMATH_CALUDE_ace_ten_king_probability_l2899_289928

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of Tens in a standard deck -/
def num_tens : ℕ := 4

/-- The number of Kings in a standard deck -/
def num_kings : ℕ := 4

/-- The probability of drawing an Ace, then a 10, and then a King from a standard deck -/
def prob_ace_ten_king : ℚ := 8 / 16575

theorem ace_ten_king_probability :
  (num_aces : ℚ) / deck_size *
  num_tens / (deck_size - 1) *
  num_kings / (deck_size - 2) = prob_ace_ten_king := by
  sorry

end NUMINAMATH_CALUDE_ace_ten_king_probability_l2899_289928


namespace NUMINAMATH_CALUDE_tan_double_angle_problem_l2899_289963

open Real

theorem tan_double_angle_problem (θ : ℝ) 
  (h1 : tan (2 * θ) = -2 * sqrt 2) 
  (h2 : π < 2 * θ ∧ 2 * θ < 2 * π) : 
  tan θ = -sqrt 2 / 2 ∧ 
  (2 * (cos (θ / 2))^2 - sin θ - 1) / (sqrt 2 * sin (θ + π / 4)) = 3 + 2 * sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_problem_l2899_289963


namespace NUMINAMATH_CALUDE_inequality_proof_l2899_289925

theorem inequality_proof (n : ℕ) (x : ℝ) (h1 : n > 0) (h2 : x ≥ n^2) :
  n * Real.sqrt (x - n^2) ≤ x / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2899_289925


namespace NUMINAMATH_CALUDE_john_completion_time_l2899_289916

/-- The number of days it takes for Rose to complete the work alone -/
def rose_days : ℝ := 480

/-- The number of days it takes for John and Rose to complete the work together -/
def joint_days : ℝ := 192

/-- The number of days it takes for John to complete the work alone -/
def john_days : ℝ := 320

/-- Theorem stating that given Rose's and joint completion times, John's completion time is 320 days -/
theorem john_completion_time : 
  (1 / john_days + 1 / rose_days = 1 / joint_days) → john_days = 320 :=
by sorry

end NUMINAMATH_CALUDE_john_completion_time_l2899_289916


namespace NUMINAMATH_CALUDE_fifteen_solutions_l2899_289969

/-- The system of equations has exactly 15 distinct real solutions -/
theorem fifteen_solutions :
  ∃! (solutions : Finset (ℝ × ℝ × ℝ × ℝ)),
    (∀ (u v s t : ℝ), (u, v, s, t) ∈ solutions ↔ 
      (u = s + t + s*u*t ∧
       v = t + u + t*u*v ∧
       s = u + v + u*v*s ∧
       t = v + s + v*s*t)) ∧
    solutions.card = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_solutions_l2899_289969


namespace NUMINAMATH_CALUDE_slope_of_line_l2899_289907

/-- The slope of the line 4x + 7y = 28 is -4/7 -/
theorem slope_of_line (x y : ℝ) : 4 * x + 7 * y = 28 → (y - 4) / x = -4 / 7 := by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l2899_289907


namespace NUMINAMATH_CALUDE_frosting_theorem_l2899_289988

/-- Jon's frosting rate in cupcakes per second -/
def jon_rate : ℚ := 1 / 40

/-- Mary's frosting rate in cupcakes per second -/
def mary_rate : ℚ := 1 / 24

/-- Time frame in seconds -/
def time_frame : ℕ := 12 * 60

/-- The number of cupcakes Jon and Mary can frost together in the given time frame -/
def cupcakes_frosted : ℕ := 48

theorem frosting_theorem : 
  ⌊(jon_rate + mary_rate) * time_frame⌋ = cupcakes_frosted := by
  sorry

end NUMINAMATH_CALUDE_frosting_theorem_l2899_289988


namespace NUMINAMATH_CALUDE_amanda_weekly_earnings_l2899_289961

def amanda_hourly_rate : ℝ := 20.00

def monday_appointments : ℕ := 5
def monday_appointment_duration : ℝ := 1.5

def tuesday_appointment_duration : ℝ := 3

def thursday_appointments : ℕ := 2
def thursday_appointment_duration : ℝ := 2

def saturday_appointment_duration : ℝ := 6

def total_hours : ℝ :=
  monday_appointments * monday_appointment_duration +
  tuesday_appointment_duration +
  thursday_appointments * thursday_appointment_duration +
  saturday_appointment_duration

theorem amanda_weekly_earnings :
  amanda_hourly_rate * total_hours = 410.00 := by
  sorry

end NUMINAMATH_CALUDE_amanda_weekly_earnings_l2899_289961


namespace NUMINAMATH_CALUDE_intersection_angle_relation_l2899_289974

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def Circle.intersect (c1 c2 : Circle) : Prop :=
  ∃ p : ℝ × ℝ, (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
                (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2

-- Define the theorem
theorem intersection_angle_relation (c1 c2 : Circle) (α β : ℝ) :
  c1.radius = c2.radius →
  c1.radius > 0 →
  (c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2 > c1.radius^2 →
  Circle.intersect c1 c2 →
  -- Assume α and β are the angles formed at the intersection points
  -- (We don't formally define these angles as it would require more complex geometry)
  β = 3 * α :=
sorry

end NUMINAMATH_CALUDE_intersection_angle_relation_l2899_289974


namespace NUMINAMATH_CALUDE_initial_rabbits_forest_rabbits_l2899_289973

theorem initial_rabbits (initial_weasels : ℕ) (foxes : ℕ) (weeks : ℕ) 
  (weasels_caught_per_fox_per_week : ℕ) (rabbits_caught_per_fox_per_week : ℕ)
  (remaining_animals : ℕ) : ℕ :=
  let total_caught := foxes * weeks * (weasels_caught_per_fox_per_week + rabbits_caught_per_fox_per_week)
  let total_initial := remaining_animals + total_caught
  total_initial - initial_weasels

theorem forest_rabbits : 
  initial_rabbits 100 3 3 4 2 96 = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_rabbits_forest_rabbits_l2899_289973


namespace NUMINAMATH_CALUDE_sector_arc_length_l2899_289957

theorem sector_arc_length (r : ℝ) (A : ℝ) (l : ℝ) : 
  r = 2 → A = π / 3 → A = 1 / 2 * r * l → l = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2899_289957


namespace NUMINAMATH_CALUDE_circle_equation_part1_circle_equation_part2_l2899_289931

-- Part 1
theorem circle_equation_part1 (A B : ℝ × ℝ) (center_line : ℝ → ℝ) :
  A = (5, 2) →
  B = (3, 2) →
  (∀ x y, center_line x = 2*x - y - 3) →
  ∃ h k r, (∀ x y, (x - h)^2 + (y - k)^2 = r^2 ↔ 
    ((x = 5 ∧ y = 2) ∨ (x = 3 ∧ y = 2)) ∧
    center_line h = k) ∧
  h = 4 ∧ k = 5 ∧ r^2 = 10 :=
sorry

-- Part 2
theorem circle_equation_part2 (A : ℝ × ℝ) (sym_line chord_line : ℝ → ℝ) :
  A = (2, 3) →
  (∀ x y, sym_line x = -x - 2*y) →
  (∀ x y, chord_line x = x - y + 1) →
  ∃ h k r, (∀ x y, (x - h)^2 + (y - k)^2 = r^2 ↔ 
    ((x = 2 ∧ y = 3) ∨ 
     (∃ x' y', sym_line x' = y' ∧ (x' - h)^2 + (y' - k)^2 = r^2)) ∧
    (∃ x1 y1 x2 y2, chord_line x1 = y1 ∧ chord_line x2 = y2 ∧
      (x1 - x2)^2 + (y1 - y2)^2 = 8)) ∧
  ((h = 6 ∧ k = -3 ∧ r^2 = 52) ∨ (h = 14 ∧ k = -7 ∧ r^2 = 244)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_part1_circle_equation_part2_l2899_289931
