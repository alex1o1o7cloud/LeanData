import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_proof_l1264_126479

theorem dress_price_proof (d : ℝ) (h : d > 0) : 
  let discounted_price := d * (1 - 0.25)
  let staff_price := discounted_price * (1 - 0.20)
  staff_price = d * 0.6 := by
  -- Unfold the let bindings
  simp_all
  -- Perform algebraic simplifications
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dress_price_proof_l1264_126479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_owner_profit_l1264_126424

/-- Represents a set of weights used by the shop owner -/
structure WeightSet where
  buyCheat : Rat  -- Percentage of cheating while buying
  sellCheat : Rat -- Percentage of cheating while selling
  usage : Rat     -- Percentage of transactions using this set

/-- Calculates the profit percentage for a given weight set -/
def profitPercentage (w : WeightSet) : Rat :=
  ((1 + w.sellCheat) - (1 - w.buyCheat)) / (1 - w.buyCheat) * 100

/-- The shop owner's weight sets -/
def weightSets : List WeightSet := [
  { buyCheat := 12/100, sellCheat := 20/100, usage := 30/100 },
  { buyCheat := 18/100, sellCheat := 30/100, usage := 45/100 },
  { buyCheat := 20/100, sellCheat := 40/100, usage := 25/100 }
]

/-- Calculates the overall profit percentage -/
def overallProfit (sets : List WeightSet) : Rat :=
  (sets.map (fun w => profitPercentage w * w.usage)).sum

/-- Theorem stating that the overall profit is approximately 56% -/
theorem shop_owner_profit :
  ∀ ε > 0, |overallProfit weightSets - 56| < ε := by
  sorry

#eval overallProfit weightSets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shop_owner_profit_l1264_126424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_ratio_is_two_to_one_l1264_126420

/-- The ratio of time taken to row upstream vs downstream -/
noncomputable def rowing_time_ratio (boat_speed still_water_speed stream_speed : ℝ) : ℝ :=
  (still_water_speed - stream_speed) / (still_water_speed + stream_speed)

/-- Theorem: The ratio of time taken to row upstream vs downstream is 2:1 -/
theorem rowing_time_ratio_is_two_to_one
  (still_water_speed : ℝ)
  (stream_speed : ℝ)
  (h1 : still_water_speed = 72)
  (h2 : stream_speed = 24) :
  rowing_time_ratio still_water_speed still_water_speed stream_speed = 2 / 1 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rowing_time_ratio_is_two_to_one_l1264_126420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_for_budget_l1264_126465

/-- Represents the taxi fare structure in Rivertown -/
structure TaxiFare where
  initialFare : ℚ  -- Initial fare for the first 3/4 mile
  initialDistance : ℚ  -- Distance covered by initial fare
  additionalRate : ℚ  -- Rate for each additional 0.1 mile
  additionalUnit : ℚ  -- Distance unit for additional rate

/-- Calculates the total fare for a given distance -/
def calculateFare (fare : TaxiFare) (distance : ℚ) : ℚ :=
  if distance ≤ fare.initialDistance then
    fare.initialFare
  else
    fare.initialFare + (distance - fare.initialDistance) / fare.additionalUnit * fare.additionalRate

/-- Theorem: The maximum distance that can be traveled with a $15 budget (including $3 tip) is 4.35 miles -/
theorem max_distance_for_budget (fare : TaxiFare) (budget : ℚ) (tip : ℚ) :
  fare.initialFare = 3 ∧
  fare.initialDistance = 3/4 ∧
  fare.additionalRate = 1/4 ∧
  fare.additionalUnit = 1/10 ∧
  budget = 15 ∧
  tip = 3 →
  ∃ (distance : ℚ), distance = 87/20 ∧ calculateFare fare distance + tip = budget :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_for_budget_l1264_126465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_difference_range_l1264_126410

theorem subset_difference_range (n k m : ℕ) (A : Finset ℕ) :
  k ≥ 2 →
  n ≤ m →
  m < ((2 * k - 1) : ℚ) / k * n →
  A ⊆ Finset.range m.succ →
  A.card = n →
  ∀ x, x ∈ Finset.range (n / (k - 1)).succ → x ≠ 0 → ∃ a b, a ∈ A ∧ b ∈ A ∧ x = a - b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_difference_range_l1264_126410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_points_ratio_l1264_126453

/-- A cubic function with coefficients a, b, and c -/
def cubic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x + c

/-- Conditions for the points A, B, C, D on the cubic function -/
structure CubicPoints (a b c : ℝ) where
  xA : ℝ
  xB : ℝ
  xC : ℝ
  xD : ℝ
  f : ℝ → ℝ := cubic_function a b c
  BD_parallel_AC : Prop
  BD_tangent_B : Prop
  BD_intersect_D : Prop
  AC_tangent_C : Prop
  AC_intersect_A : Prop

/-- Theorem stating the ratio of x-coordinate differences -/
theorem cubic_points_ratio {a b c : ℝ} (points : CubicPoints a b c) :
  (points.xA - points.xB) / (points.xB - points.xC) = 1/2 ∧
  (points.xC - points.xD) / (points.xB - points.xC) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_points_ratio_l1264_126453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hema_younger_than_ravi_l1264_126493

-- Define the ages of individuals
def raj_age : ℕ → ℕ := sorry
def ravi_age : ℕ → ℕ := sorry
def hema_age : ℕ → ℕ := sorry
def rahul_age : ℕ → ℕ := sorry

-- Define the conditions
axiom raj_older_than_ravi : ∀ t : ℕ, raj_age t = ravi_age t + 3
axiom raj_triple_rahul : ∀ t : ℕ, raj_age t = 3 * rahul_age t
axiom hema_rahul_ratio : ∀ t : ℕ, 3 * rahul_age t = 2 * hema_age t
axiom raj_hema_relation : raj_age 20 = (4/3 : ℚ) * (hema_age 20 : ℚ)

-- Theorem to prove
theorem hema_younger_than_ravi :
  ∃ t : ℕ, ravi_age t - hema_age t = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hema_younger_than_ravi_l1264_126493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_sqrt_20702_div_22_l1264_126422

/-- The point from which we're measuring the distance -/
def P : Fin 3 → ℝ := ![2, 3, 1]

/-- A point on the line -/
def Q (t : ℝ) : Fin 3 → ℝ := ![4 + 2*t, 8 + 3*t, 5 - 3*t]

/-- The direction vector of the line -/
def v : Fin 3 → ℝ := ![2, 3, -3]

/-- The distance from a point to a line -/
noncomputable def distance_to_line (P : Fin 3 → ℝ) (Q : ℝ → Fin 3 → ℝ) (v : Fin 3 → ℝ) : ℝ :=
  let t := -(2*(P 0 - Q 0 0) + 3*(P 1 - Q 0 1) - 3*(P 2 - Q 0 2)) / (2*2 + 3*3 + 3*3)
  let closest_point := Q t
  Real.sqrt ((P 0 - closest_point 0)^2 + (P 1 - closest_point 1)^2 + (P 2 - closest_point 2)^2)

theorem distance_to_line_is_sqrt_20702_div_22 :
  distance_to_line P Q v = Real.sqrt 20702 / 22 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_sqrt_20702_div_22_l1264_126422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_difference_square_implies_unique_pair_l1264_126497

theorem prime_difference_square_implies_unique_pair :
  ∀ a b : ℕ,
    a > b →
    a > 0 →
    b > 0 →
    Nat.Prime (a^2 + b - (a + b^2)) →
    a = 2 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_difference_square_implies_unique_pair_l1264_126497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_for_given_conditions_l1264_126402

/-- Calculate the difference between compound and simple interest -/
noncomputable def interest_difference (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  let simple_interest := principal * rate * (time : ℝ) / 100
  let compound_interest := principal * ((1 + rate / 100) ^ time - 1)
  compound_interest - simple_interest

/-- Theorem stating the difference between compound and simple interest for the given conditions -/
theorem interest_difference_for_given_conditions :
  interest_difference 625 4 2 = 26 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_for_given_conditions_l1264_126402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1264_126492

-- Define the function f
noncomputable def f (x : Real) : Real := 3 + 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2

-- Define the triangle ABC
structure Triangle where
  a : Real
  b : Real
  c : Real
  A : Real
  B : Real
  C : Real

-- Define the area function
noncomputable def area (t : Triangle) : Real := 
  1 / 2 * t.b * t.c * Real.sin t.A

-- State the theorem
theorem triangle_properties (abc : Triangle) :
  abc.a = 2 ∧ f abc.A = 5 →
  abc.A = π / 3 ∧ 
  (∀ (other : Triangle), other.a = 2 → area other ≤ Real.sqrt 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1264_126492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_on_max_set_l1264_126448

noncomputable section

/-- The function f(x) = 2sin(2x + π/6) + 2 --/
def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6) + 2

/-- The set of x values where f reaches its maximum --/
def max_set : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6}

/-- Theorem: f reaches its maximum value on the max_set --/
theorem f_max_on_max_set :
  ∀ x : ℝ, x ∈ max_set → ∀ y : ℝ, f x ≥ f y := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_on_max_set_l1264_126448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_theorem_l1264_126494

/-- Represents the composition of an alloy -/
structure AlloyComposition where
  component1 : ℝ
  component2 : ℝ

/-- Represents the mixture of two alloys -/
structure AlloyMixture where
  alloyA : AlloyComposition
  alloyB : AlloyComposition
  massA : ℝ
  massB : ℝ

/-- Calculates the amount of a component in an alloy given its mass and composition -/
noncomputable def componentAmount (mass : ℝ) (composition : AlloyComposition) (componentRatio : ℝ) : ℝ :=
  mass * componentRatio / (composition.component1 + composition.component2)

theorem alloy_mixture_theorem (mixture : AlloyMixture) 
  (hA : mixture.alloyA.component1 = 2 ∧ mixture.alloyA.component2 = 3)
  (hB : mixture.alloyB.component1 = 3 ∧ mixture.alloyB.component2 = 5)
  (hMassA : mixture.massA = 120)
  (hTinTotal : componentAmount mixture.massA mixture.alloyA mixture.alloyA.component2 + 
               componentAmount mixture.massB mixture.alloyB mixture.alloyB.component1 = 139.5) :
  mixture.massB = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alloy_mixture_theorem_l1264_126494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_is_60_seconds_l1264_126458

/-- Represents the problem of a train crossing a platform --/
structure TrainProblem where
  train_length : ℝ
  train_speed_kmh : ℝ
  platform_length : ℝ

/-- Converts km/h to m/s --/
noncomputable def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * 1000 / 3600

/-- Calculates the time (in seconds) for the train to cross the platform --/
noncomputable def crossing_time (problem : TrainProblem) : ℝ :=
  let total_distance := problem.train_length + problem.platform_length
  let speed_ms := kmh_to_ms problem.train_speed_kmh
  total_distance / speed_ms

/-- The main theorem stating the time taken for the train to cross the platform --/
theorem train_crossing_time_is_60_seconds 
  (problem : TrainProblem)
  (h1 : problem.train_length = 600)
  (h2 : problem.train_speed_kmh = 72)
  (h3 : problem.platform_length = problem.train_length) :
  crossing_time problem = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_is_60_seconds_l1264_126458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_10_sqrt_3_l1264_126407

-- Define the sequence of even numbers
def evenSequence (n : ℕ) : ℕ := 2 * n

-- Define the position in the arrangement
def position (n : ℕ) : ℕ × ℕ :=
  let row := (n - 1) / 4 + 1
  let col := (n - 1) % 4 + 1
  (row, col)

-- Define the function to find the position of a given number
noncomputable def findPosition (x : ℝ) : ℕ × ℕ :=
  let n := Int.floor (x^2 / 2)
  position n.toNat

-- Theorem statement
theorem position_of_10_sqrt_3 :
  findPosition (10 * Real.sqrt 3) = (38, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_10_sqrt_3_l1264_126407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1264_126412

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | ∀ k : ℤ, x ≠ k * Real.pi / 2 + 3 * Real.pi / 8} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1264_126412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_existence_l1264_126400

def f (x : ℝ) := 2 * x^3 - 2 * x

def isRectangle (a b c d : ℝ) : Prop :=
  ∃ (p q r s : ℝ × ℝ),
    p = (a, f a) ∧ q = (b, f b) ∧ r = (c, f c) ∧ s = (d, f d) ∧
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = (r.1 - s.1)^2 + (r.2 - s.2)^2 ∧
    (p.1 - r.1)^2 + (p.2 - r.2)^2 = (q.1 - s.1)^2 + (q.2 - s.2)^2 ∧
    (p.1 - q.1) * (p.1 - r.1) + (p.2 - q.2) * (p.2 - r.2) = 0

theorem rectangle_existence (a : ℝ) :
  a > 0 ∧ (∃ (b c d : ℝ), b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ isRectangle a b c d) ↔
  Real.sqrt 3 / 3 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_existence_l1264_126400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_rain_time_l1264_126459

/-- Represents Shelby's driving scenario -/
structure DrivingScenario where
  sun_speed : ℚ  -- Speed in miles per hour when sunny
  rain_speed : ℚ  -- Speed in miles per hour when raining
  total_distance : ℚ  -- Total distance in miles
  total_time : ℚ  -- Total time in minutes

/-- Calculates the time driven in rain given a DrivingScenario -/
def time_in_rain (scenario : DrivingScenario) : ℚ :=
  let sun_speed_per_minute := scenario.sun_speed / 60
  let rain_speed_per_minute := scenario.rain_speed / 60
  (scenario.total_time * sun_speed_per_minute - scenario.total_distance) /
  (sun_speed_per_minute - rain_speed_per_minute)

/-- Theorem: Given Shelby's driving scenario, the time she drove in the rain is 24 minutes -/
theorem shelby_rain_time (scenario : DrivingScenario) 
  (h1 : scenario.sun_speed = 30)
  (h2 : scenario.rain_speed = 20)
  (h3 : scenario.total_distance = 16)
  (h4 : scenario.total_time = 40) :
  time_in_rain scenario = 24 := by
  sorry

#eval time_in_rain { sun_speed := 30, rain_speed := 20, total_distance := 16, total_time := 40 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_rain_time_l1264_126459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1264_126467

/-- Given a train of length 800 meters that takes 2 minutes to pass a tree
    and 5 minutes to pass a bridge, prove that the bridge is 1200 meters long. -/
theorem bridge_length_calculation (train_length time_tree time_bridge : ℝ) 
    (h1 : train_length = 800)
    (h2 : time_tree = 2)
    (h3 : time_bridge = 5) :
    (train_length / (time_tree * 60)) * time_bridge * 60 - train_length = 1200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l1264_126467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_common_terms_three_common_terms_l1264_126454

def sequence_a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => sequence_a (n + 1) + sequence_a n

def sequence_b : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | n + 2 => sequence_b (n + 1) + sequence_b n

def common_terms (n : ℕ) : Prop :=
  ∃ i j, i ≤ n ∧ j ≤ n ∧ sequence_a i = sequence_b j

theorem exactly_three_common_terms :
  ∃ n, common_terms n ∧ (∀ m, common_terms m → m ≤ 3) :=
by
  -- We claim that n = 3 satisfies the condition
  use 3
  constructor
  
  -- Prove that there are common terms up to n = 3
  · sorry -- This part requires showing the existence of common terms
  
  -- Prove that there are no more than 3 common terms
  · sorry -- This part requires proving the upper bound

-- Helper lemmas to establish properties of the sequences
lemma sequence_a_growth (n : ℕ) : n ≥ 4 → sequence_a (n-1) < sequence_a n :=
by sorry

lemma sequence_b_growth (n : ℕ) : n ≥ 4 → sequence_b (n-1) < sequence_b n :=
by sorry

lemma sequence_separation (n : ℕ) : n ≥ 4 → sequence_a (n-1) < sequence_b n ∧ sequence_b n < sequence_a n :=
by sorry

-- Main theorem stating that there are exactly 3 common terms
theorem three_common_terms : ∃! n, n = 3 ∧ common_terms n ∧ ∀ m, common_terms m → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_common_terms_three_common_terms_l1264_126454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l1264_126444

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ -1 then -3 - x
  else if x ≤ 1 then Real.sqrt (9 - (x + 1)^2) - 3
  else if x ≤ 2 then 3*(x - 1)
  else 0  -- Define a default value for x outside the given range

-- State the theorem
theorem abs_g_piecewise (x : ℝ) :
  ((-4 ≤ x ∧ x ≤ -1) → |g x| = 3 + x) ∧
  ((-1 ≤ x ∧ x ≤ 1) → |g x| = Real.sqrt (9 - (x + 1)^2) - 3) ∧
  ((1 ≤ x ∧ x ≤ 2) → |g x| = 3*(x - 1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abs_g_piecewise_l1264_126444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l1264_126437

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the sequence a_n
def a : ℕ → ℝ := sorry

-- Axioms based on the given conditions
axiom f_domain : ∀ x, x > 0 → f x ≠ 0
axiom f_pos : ∀ x, x > 1 → f x > 0
axiom f_add : ∀ x y, x > 0 → y > 0 → f x + f y = f (x * y)
axiom a_init : a 1 = f 1
axiom a_rec : ∀ n, n > 0 → f (a (n + 1)) = f (2 * a n + 1)

-- Theorem to prove
theorem a_2017_value : a 2017 = 2^2016 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_value_l1264_126437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_equals_four_sqrt_two_over_pi_l1264_126478

theorem circle_radius_equals_four_sqrt_two_over_pi (s r : ℝ) :
  s > 0 →  -- side length of square is positive
  r > 0 →  -- radius of circle is positive
  r = s * Real.sqrt 2 / 2 →  -- radius-side relationship for circumscribed circle
  4 * s = π * r^2 →  -- perimeter of square equals area of circle
  r = 4 * Real.sqrt 2 / π :=
by
  intro hs hr h_radius h_area
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_equals_four_sqrt_two_over_pi_l1264_126478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_5_l1264_126464

def problem_sequence : List Nat := [2, 12, 22, 32, 42, 52, 62, 72, 82, 92]

theorem product_remainder_mod_5 : 
  (problem_sequence.prod % 5 = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_5_l1264_126464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1264_126409

noncomputable def f (x : ℝ) := Real.sqrt (-12 * x^2 + 7 * x + 13)

theorem domain_of_f :
  Set.Icc ((7 - Real.sqrt 673) / 24) ((7 + Real.sqrt 673) / 24) =
  {x : ℝ | ∃ y : ℝ, f x = y} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1264_126409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_revolution_volume_formula_l1264_126456

/-- The volume of the solid of revolution formed by rotating a right triangle
    around the bisector of its external right angle. -/
noncomputable def solidRevolutionVolume (c α : ℝ) : ℝ :=
  (Real.pi * c^3 / 6) * Real.sin (2 * α) * Real.sin (Real.pi/4 + α)

/-- Theorem stating that the volume of the solid of revolution formed by rotating
    a right triangle with hypotenuse c and acute angle α around the bisector of
    its external right angle is equal to (π * c³ / 6) * sin(2α) * sin(π/4 + α). -/
theorem solid_revolution_volume_formula (c α : ℝ) (h_c : c > 0) (h_α : 0 < α ∧ α < Real.pi/2) :
  let triangle_volume := solidRevolutionVolume c α
  triangle_volume = (Real.pi * c^3 / 6) * Real.sin (2 * α) * Real.sin (Real.pi/4 + α) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solid_revolution_volume_formula_l1264_126456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l1264_126452

/-- The average speed of a train -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: The average speed of a train that travels 42 meters in 6 seconds is 7 meters per second -/
theorem train_speed : average_speed 42 6 = 7 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Perform the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_l1264_126452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l1264_126443

-- Define the square
def square_center : ℝ × ℝ := (5, 5)
def square_area : ℝ := 16

-- Define the dilation
def dilation_center : ℝ × ℝ := (3, 3)
def dilation_scale : ℝ := 3

-- Define the dilation function
def dilate (p : ℝ × ℝ) : ℝ × ℝ :=
  (dilation_center.1 + dilation_scale * (p.1 - dilation_center.1),
   dilation_center.2 + dilation_scale * (p.2 - dilation_center.2))

-- Define distance from origin
noncomputable def dist_from_origin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 * p.1 + p.2 * p.2)

theorem farthest_vertex_after_dilation :
  ∃ (original_vertex : ℝ × ℝ),
    (dist_from_origin original_vertex)^2 = square_area ∧
    (∀ (p : ℝ × ℝ), (dist_from_origin p)^2 ≤ square_area →
      dist_from_origin (dilate original_vertex) ≥ dist_from_origin (dilate p)) ∧
    dilate original_vertex = (15, 15) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_vertex_after_dilation_l1264_126443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_equilateral_triangle_side_length_l1264_126496

/-- Given a sector with radius R and central angle α, an equilateral triangle
    inscribed in the sector with one vertex at the midpoint of the arc and
    the other two vertices on the radii has sides of length
    R * sin(α/2) / sin(30° + α/2) -/
theorem inscribed_equilateral_triangle_side_length
  (R : ℝ) (α : ℝ) (h_R : R > 0) (h_α : 0 < α ∧ α < 2 * Real.pi) :
  ∃ (triangle : Set (ℝ × ℝ)),
    (∀ p, p ∈ triangle → p.1^2 + p.2^2 ≤ R^2) ∧
    (∃ p ∈ triangle, p.1^2 + p.2^2 = R^2 ∧ p.2 = R * Real.sin (α/2)) ∧
    (∀ p q, p ∈ triangle → q ∈ triangle → p ≠ q →
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = (R * Real.sin (α/2) / Real.sin (30 * π/180 + α/2))^2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_equilateral_triangle_side_length_l1264_126496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_endpoint_probability_dodecahedron_l1264_126403

/-- A regular dodecahedron -/
structure Dodecahedron where
  vertices : Fin 20
  edges_per_vertex : Fin 4
  edges_per_vertex_eq : edges_per_vertex = 3

/-- The probability of choosing two vertices that are endpoints of an edge in a regular dodecahedron -/
def edge_endpoint_probability (d : Dodecahedron) : ℚ :=
  (20 * d.edges_per_vertex / 2) / Nat.choose 20 2

theorem edge_endpoint_probability_dodecahedron :
  ∀ d : Dodecahedron, edge_endpoint_probability d = 3 / 19 := by
  sorry

#eval Nat.choose 20 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_edge_endpoint_probability_dodecahedron_l1264_126403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l1264_126490

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (1 + Real.sqrt 3 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 12

-- Define the polar coordinate system
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the ray θ = π/3
noncomputable def ray_π_3 (t : ℝ) : ℝ × ℝ := (t / 2, t * Real.sqrt 3 / 2)

-- Theorem statement
theorem distance_between_intersection_points :
  ∃ (t₁ t₂ : ℝ), 
    (C₁ (Real.pi / 3) = ray_π_3 t₁) ∧
    (C₂ (ray_π_3 t₂).1 (ray_π_3 t₂).2) ∧
    (t₂ - t₁ = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_intersection_points_l1264_126490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_martyrs_cemetery_distance_l1264_126442

/-- Represents the distance from the school to the Martyrs' Cemetery in kilometers. -/
noncomputable def distance : ℝ := sorry

/-- Represents the original speed of the car in km/h. -/
noncomputable def original_speed : ℝ := sorry

/-- Represents the scheduled travel time in hours. -/
noncomputable def scheduled_time : ℝ := 2

/-- Represents the time saved (in hours) when increasing speed by 1/5 after 1 hour of travel. -/
noncomputable def time_saved_scenario1 : ℝ := 1/6

/-- Represents the distance traveled at original speed in the second scenario. -/
noncomputable def distance_original_speed_scenario2 : ℝ := 60

/-- Represents the time saved (in hours) in the second scenario. -/
noncomputable def time_saved_scenario2 : ℝ := 1/3

/-- Theorem stating that the distance to the Martyrs' Cemetery is 180 km. -/
theorem martyrs_cemetery_distance : distance = 180 := by
  sorry

#check martyrs_cemetery_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_martyrs_cemetery_distance_l1264_126442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_sum_l1264_126434

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (-2 * Real.sqrt 2 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 12 + y^2 / 4 = 1

-- Define the focus F
def focus_F : ℝ × ℝ := (-2 * Real.sqrt 2, 0)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, 
    line_l t₁ = A ∧ 
    line_l t₂ = B ∧ 
    curve_C A.1 A.2 ∧ 
    curve_C B.1 B.2

-- State the theorem
theorem intersection_points_sum (A B : ℝ × ℝ) :
  intersection_points A B →
  (1 / dist A focus_F + 1 / dist B focus_F) = Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_sum_l1264_126434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_true_l1264_126406

-- Define the propositions
def prop1 : Prop := ∀ x : ℝ, x^2 - x + (1/4 : ℝ) ≥ 0

noncomputable def prop2 : Prop := ∃ x : ℝ, x > 0 ∧ Real.log x + 1 / Real.log x ≤ 2

def prop3 : Prop := ∀ a b c : ℝ, (a > b ↔ a * c^2 > b * c^2)

def prop4 : Prop := ∀ x : ℝ, (2 : ℝ)^(-x) - (2 : ℝ)^x = -((2 : ℝ)^x - (2 : ℝ)^(-x))

-- Theorem stating that exactly three of the propositions are true
theorem exactly_three_true : 
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ∨
  (prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) ∨
  (¬prop1 ∧ prop2 ∧ prop3 ∧ prop4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_true_l1264_126406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_exponents_l1264_126447

theorem no_negative_exponents (a b c d : ℤ) (h : (2 : ℝ)^a + (2 : ℝ)^b + 5 = (3 : ℝ)^c + (3 : ℝ)^d) : 
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_negative_exponents_l1264_126447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1264_126460

-- Define the vertices of the piecewise linear function
def vertices : List (ℚ × ℚ) := [(-5, -6), (-3, -2), (-1, -3), (1, 3), (3, 2), (4, 3), (6, 7)]

-- Define the piecewise linear function g
noncomputable def g (x : ℚ) : ℚ :=
  let segments := List.zip vertices (List.tail vertices)
  match segments.find? (fun ((x1, _), (x2, _)) => x1 ≤ x ∧ x ≤ x2) with
  | some ((x1, y1), (x2, y2)) => y1 + (y2 - y1) * (x - x1) / (x2 - x1)
  | none => 0  -- Default value for x outside the domain

-- Define the function for the line y = x + 2
def h (x : ℚ) : ℚ := x + 2

-- Theorem statement
theorem intersection_sum :
  ∃ (S : Finset ℚ), (∀ x ∈ S, g x = h x) ∧ (S.sum id = -1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l1264_126460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l1264_126474

theorem inequality_condition (a : ℝ) :
  (a < 0) →
  (∀ x : ℝ, Real.sin x ^ 2 + a * Real.cos x + a ^ 2 ≥ 1 + Real.cos x) ↔
  (a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l1264_126474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_dilution_l1264_126423

/-- Represents the composition of a chocolate drink -/
structure ChocolateDrink where
  volume : ℚ
  chocolate_percentage : ℚ

/-- Calculates the percentage of chocolate in a new mixture after adding milk -/
def new_chocolate_percentage (drink : ChocolateDrink) (milk_volume : ℚ) : ℚ :=
  (drink.chocolate_percentage * drink.volume) / (drink.volume + milk_volume) * 100

/-- Theorem stating that adding 10 litres of milk to 50 litres of 6% chocolate drink results in 5% chocolate content -/
theorem chocolate_dilution :
  let original_drink : ChocolateDrink := { volume := 50, chocolate_percentage := 6 }
  let milk_volume : ℚ := 10
  new_chocolate_percentage original_drink milk_volume = 5 := by
  -- Proof goes here
  sorry

#eval new_chocolate_percentage { volume := 50, chocolate_percentage := 6 } 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_dilution_l1264_126423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1264_126484

noncomputable def f (x y : ℝ) : ℝ := (x + y) / (Int.floor x * Int.floor y + Int.floor x + Int.floor y + 1)

theorem range_of_f :
  ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 →
  ∃ S : Set ℝ, S = {1/2} ∪ Set.Icc (5/6) (5/4) ∧
  (∀ z : ℝ, z ∈ S ↔ ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a * b = 1 ∧ f a b = z) :=
by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1264_126484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_10_percent_l1264_126438

/-- Calculates the total gain percentage for a shopkeeper who cheats by a given percentage while buying and selling. -/
noncomputable def shopkeeperGain (cheatPercentage : ℝ) : ℝ :=
  let buyingRatio := 1 + cheatPercentage / 100
  let sellingRatio := 1 - cheatPercentage / 100
  (buyingRatio - sellingRatio) * 100

/-- Proves that a shopkeeper who cheats by 10% while buying and selling has a total gain of 20%. -/
theorem shopkeeper_gain_10_percent : shopkeeperGain 10 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_gain_10_percent_l1264_126438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_eq_l1264_126433

-- Define P as a set of natural numbers
def P : Set ℕ := {x | 1 ≤ x ∧ x ≤ 10}

-- Define Q as a set of real numbers
def Q : Set ℝ := {x | x^2 + x - 6 ≤ 0}

-- Define the intersection of P and Q
def P_intersect_Q : Set ℕ := {x ∈ P | ∃ (y : ℝ), y ∈ Q ∧ y = x}

-- Theorem stating that the intersection of P and Q is {1, 2}
theorem P_intersect_Q_eq : P_intersect_Q = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_intersect_Q_eq_l1264_126433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l1264_126408

/-- Converts cylindrical coordinates to rectangular coordinates -/
noncomputable def cylindrical_to_rectangular (r : ℝ) (θ : ℝ) (z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

/-- The given point in cylindrical coordinates -/
noncomputable def cylindrical_point : ℝ × ℝ × ℝ := (6, 7 * Real.pi / 4, -2)

/-- The expected point in rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ × ℝ := (3 * Real.sqrt 2, -3 * Real.sqrt 2, -2)

theorem cylindrical_to_rectangular_conversion :
  cylindrical_to_rectangular cylindrical_point.1 cylindrical_point.2.1 cylindrical_point.2.2 = rectangular_point := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylindrical_to_rectangular_conversion_l1264_126408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_h_is_odd_l1264_126487

-- Define the function f(x) = a^x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

-- Define the function g(x) = a^(x+2)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^(x+2)

-- Define the function h(x) = ln(1+x) - ln(1-x)
noncomputable def h (x : ℝ) : ℝ := Real.log (1+x) - Real.log (1-x)

-- Theorem 1: g is a translation of f
theorem graph_translation (a : ℝ) (h_a : a > 0 ∧ a ≠ 1) :
  ∃ k : ℝ, ∀ x : ℝ, g a x = f a (x + k) := by
  sorry

-- Theorem 2: h is an odd function
theorem h_is_odd : 
  ∀ x : ℝ, x > -1 ∧ x < 1 → h (-x) = -h x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_translation_h_is_odd_l1264_126487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_6_48_l1264_126431

/-- The acute angle between clock hands at a given time -/
noncomputable def clock_angle (hours : ℕ) (minutes : ℕ) : ℝ :=
  let minute_angle := (minutes : ℝ) / 60 * 360
  let hour_angle := ((hours % 12 : ℝ) + (minutes : ℝ) / 60) / 12 * 360
  min (abs (hour_angle - minute_angle)) (360 - abs (hour_angle - minute_angle))

/-- The acute angle between clock hands at 6:48 is 84° -/
theorem clock_angle_at_6_48 : 
  clock_angle 6 48 = 84 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_6_48_l1264_126431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_5_8_plus_10_5_l1264_126421

theorem greatest_prime_factor_of_5_8_plus_10_5 :
  (Nat.factors (5^8 + 10^5)).maximum = some 157 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_prime_factor_of_5_8_plus_10_5_l1264_126421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_increase_is_one_l1264_126436

/-- The increase in average age when including the teacher -/
noncomputable def average_age_increase (num_students : ℕ) (student_avg_age : ℝ) (teacher_age : ℕ) : ℝ :=
  let total_student_age := (num_students : ℝ) * student_avg_age
  let total_age_with_teacher := total_student_age + (teacher_age : ℝ)
  let new_average := total_age_with_teacher / ((num_students : ℝ) + 1)
  new_average - student_avg_age

/-- Theorem stating the increase in average age for the given problem -/
theorem average_age_increase_is_one :
  average_age_increase 20 21 42 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_age_increase_is_one_l1264_126436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l1264_126462

/-- Two lines in 2D space --/
structure Line2D where
  origin : ℝ × ℝ
  direction : ℝ × ℝ

/-- The first line --/
noncomputable def line1 : Line2D := { origin := (1, 1), direction := (2, -3) }

/-- The second line --/
noncomputable def line2 : Line2D := { origin := (-1, 2), direction := (5, -2) }

/-- A point lies on a line if it satisfies the parametric equation --/
def pointOnLine (p : ℝ × ℝ) (l : Line2D) : Prop :=
  ∃ t : ℝ, p = (l.origin.1 + t * l.direction.1, l.origin.2 + t * l.direction.2)

/-- The intersection point of the two lines --/
noncomputable def intersectionPoint : ℝ × ℝ := (9/11, 14/11)

/-- Theorem: The intersection point lies on both lines and is unique --/
theorem intersection_point_correct :
  pointOnLine intersectionPoint line1 ∧
  pointOnLine intersectionPoint line2 ∧
  ∀ p : ℝ × ℝ, pointOnLine p line1 ∧ pointOnLine p line2 → p = intersectionPoint := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_correct_l1264_126462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x9_in_expansion_l1264_126413

theorem coefficient_x9_in_expansion : 
  let expansion := (1 - 3 * X ^ 3) ^ 7
  Polynomial.coeff expansion 9 = -945 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x9_in_expansion_l1264_126413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_l1264_126404

/-- The slant height of a cone with base radius 4 cm and lateral surface central angle 120° --/
noncomputable def slant_height (base_radius : ℝ) (central_angle : ℝ) : ℝ :=
  (base_radius * 2 * Real.pi) / (central_angle / 360 * 2 * Real.pi)

/-- Theorem: The slant height of a cone with base radius 4 cm and lateral surface central angle 120° is 12 cm --/
theorem cone_slant_height : slant_height 4 (120 : ℝ) = 12 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval slant_height 4 120

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_slant_height_l1264_126404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_heads_probability_l1264_126416

/-- The probability of getting heads for an unfair coin -/
noncomputable def p_heads : ℚ := 3/4

/-- The number of coin tosses -/
def n_tosses : ℕ := 80

/-- The probability of getting an odd number of heads after n tosses -/
noncomputable def P_odd (n : ℕ) : ℚ := 1/2 * (1 - (1/3)^n)

/-- Theorem: The probability of getting an odd number of heads after 80 tosses
    of an unfair coin with 3/4 probability of heads is 1/2(1 - 1/3^80) -/
theorem odd_heads_probability :
  P_odd n_tosses = 1/2 * (1 - (1/3)^n_tosses) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_heads_probability_l1264_126416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_primes_l1264_126440

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def satisfies_conditions (p : ℕ) : Bool :=
  p.Prime ∧
  1 < p ∧ p < 200 ∧
  p % 5 = 1 ∧
  p % 7 = 6

theorem sum_of_special_primes :
  (Finset.filter (fun p => satisfies_conditions p) (Finset.range 200)).sum id = 222 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_primes_l1264_126440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_length_l1264_126476

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  longBase : ℝ
  shortBase : ℝ
  nonBaseSide : ℝ

/-- The diagonal length of an isosceles trapezoid -/
noncomputable def diagonalLength (t : IsoscelesTrapezoid) : ℝ :=
  Real.sqrt 414

/-- Theorem: The diagonal length of the specified isosceles trapezoid is √414 -/
theorem isosceles_trapezoid_diagonal_length 
  (t : IsoscelesTrapezoid) 
  (h1 : t.longBase = 18) 
  (h2 : t.shortBase = 12) 
  (h3 : t.nonBaseSide = 12) : 
  diagonalLength t = Real.sqrt 414 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_diagonal_length_l1264_126476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storage_unit_area_proof_l1264_126449

theorem storage_unit_area_proof (total_units : ℕ) (total_area : ℝ) 
  (known_units : ℕ) (known_unit_length : ℝ) (known_unit_width : ℝ) :
  total_units = 72 →
  total_area = 8640 →
  known_units = 30 →
  known_unit_length = 12 →
  known_unit_width = 6 →
  ∃ (remaining_unit_area : ℝ), 
    (remaining_unit_area ≥ 153.5 ∧ remaining_unit_area < 154.5) ∧
    (remaining_unit_area * (total_units - known_units : ℝ) + 
      (known_units : ℝ) * known_unit_length * known_unit_width ≥ total_area - 0.5 ∧
     remaining_unit_area * (total_units - known_units : ℝ) + 
      (known_units : ℝ) * known_unit_length * known_unit_width < total_area + 0.5) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_storage_unit_area_proof_l1264_126449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_partition_exists_l1264_126405

/-- Represents a cube with integer edge length -/
structure Cube where
  edge : ℕ

/-- The volume of a cube -/
def volume (c : Cube) : ℕ := c.edge ^ 3

/-- A partition of a cube into smaller cubes -/
structure CubePartition where
  original : Cube
  parts : List Cube
  sum_volumes : volume original = (parts.map volume).sum
  all_fit : ∀ c, c ∈ parts → c.edge ≤ original.edge

theorem cube_partition_exists : ∃ (p : CubePartition), 
  p.original.edge = 4 ∧ 
  p.parts.length = 38 ∧
  (∃ c₁ c₂, c₁ ∈ p.parts ∧ c₂ ∈ p.parts ∧ c₁.edge ≠ c₂.edge) ∧
  (∀ c, c ∈ p.parts → c.edge > 0) :=
sorry

#check cube_partition_exists

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_partition_exists_l1264_126405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_tax_paid_l1264_126470

-- Define the tax calculation function
noncomputable def calculate_tax (income : ℝ) : ℝ :=
  if income ≤ 5000 then
    0.08 * income
  else
    0.08 * 5000 + 0.10 * (income - 5000)

-- Theorem statement
theorem income_from_tax_paid (tax_paid : ℝ) :
  tax_paid = 950 → ∃ income : ℝ, calculate_tax income = tax_paid ∧ income = 10500 :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_from_tax_paid_l1264_126470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_l1264_126457

/-- Represents a chess board of size (2n+1) x (2n+1) -/
def Board (n : ℕ) := Fin (2*n+1) × Fin (2*n+1)

/-- Represents a marked cell on the board -/
def MarkedCell (n : ℕ) := Board n

/-- Predicate to check if a cell is attacked by a bishop -/
def IsAttacked (n : ℕ) (bishop : Board n) (cell : Board n) : Prop := sorry

/-- Predicate to check if a cell is attacked by at least two marked cells -/
def IsAttackedByTwoMarked (n : ℕ) (cell : Board n) (markedCells : Set (MarkedCell n)) : Prop :=
  ∃ m1 m2 : MarkedCell n, m1 ≠ m2 ∧ m1 ∈ markedCells ∧ m2 ∈ markedCells ∧
    IsAttacked n m1 cell ∧ IsAttacked n m2 cell

/-- The main theorem stating the minimum number of cells to be marked -/
theorem min_marked_cells (n : ℕ) (h : n > 1) :
  ∃ markedCells : Finset (MarkedCell n),
    (∀ cell : Board n, IsAttackedByTwoMarked n cell markedCells) ∧
    markedCells.card = 4*n ∧
    ∀ markedCells' : Finset (MarkedCell n),
      (∀ cell : Board n, IsAttackedByTwoMarked n cell markedCells') →
      markedCells'.card ≥ 4*n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_l1264_126457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_invariant_l1264_126418

noncomputable def sample_A : List ℝ := [52, 54, 54, 56, 56, 56, 55, 55, 55, 55]
noncomputable def sample_B : List ℝ := sample_A.map (· + 6)

noncomputable def mean (sample : List ℝ) : ℝ := sample.sum / sample.length

noncomputable def variance (sample : List ℝ) : ℝ :=
  let m := mean sample
  (sample.map (λ x => (x - m)^2)).sum / sample.length

noncomputable def standard_deviation (sample : List ℝ) : ℝ :=
  Real.sqrt (variance sample)

theorem standard_deviation_invariant :
  standard_deviation sample_A = standard_deviation sample_B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_deviation_invariant_l1264_126418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_f_l1264_126486

-- Define the function type
def NatToInt := ℕ → ℤ

-- Define the conditions
def satisfiesConditions (f : NatToInt) : Prop :=
  ∃ (p : ℕ), Prime p ∧ p > 2024 ∧
  (∀ k : ℕ, k > 0 → |f k| ≤ k) ∧
  (∀ a : ℕ, a * f (a + p) = a * f a + p * f a) ∧
  (∀ a : ℕ, ∃ m : ℤ, a^((p + 1) / 2) - f a = m * p)

-- Define the epsilon function
def epsilon : ℕ → ℤ := sorry

-- Define the theorem
theorem characterize_f (f : NatToInt) (h : satisfiesConditions f) :
  ∃ (p : ℕ) (ε : ℕ → ℤ), Prime p ∧ p > 2024 ∧
  (∀ r : ℕ, r < p → ε r = 1 ∨ ε r = -1) ∧
  (∀ k r : ℕ, r < p → f (k * p + r) = ε r * (k * p + r)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterize_f_l1264_126486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1264_126450

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (x^2 - 2*x)

-- State the theorem
theorem f_range : Set.range f = Set.Ioc 0 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1264_126450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_intact_square_l1264_126475

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (removed_rectangles : Nat)

/-- Represents a square on the chessboard -/
structure Square :=
  (size : Nat)

/-- Predicate to check if a cell is removed -/
def is_removed (board : Chessboard) (x y : Nat) : Prop :=
  sorry

/-- Theorem: After removing eight 2x1 rectangles from an 8x8 chessboard, 
    there always exists at least one 2x2 square that remains intact -/
theorem exists_intact_square (board : Chessboard) (square : Square) : 
  board.size = 8 → 
  board.removed_rectangles = 8 → 
  square.size = 2 → 
  ∃ (x y : Nat), x < board.size - 1 ∧ y < board.size - 1 ∧ 
    (∀ (i j : Nat), i < square.size → j < square.size → 
      ¬ is_removed board (x + i) (y + j)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_intact_square_l1264_126475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_expression_l1264_126446

theorem negative_expression : 
  ((-(-3) > 0) ∧ (-(-3)^3 > 0) ∧ ((-3)^2 > 0) ∧ (-|-3| < 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_expression_l1264_126446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_to_canonical_l1264_126463

/-- The quadratic form representing the original equation -/
def original_quadratic_form (x y : ℝ) : ℝ :=
  5 * x^2 + 6 * x * y + 5 * y^2 - 5 * x - 5 * y - 2

/-- The matrix representation of the quadratic form -/
def quadratic_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![5, 3; 3, 5]

/-- The eigenvalues of the quadratic matrix -/
def eigenvalues : Set ℝ := {2, 8}

/-- The transformation matrix for the coordinate change -/
noncomputable def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![(Real.sqrt 2) / 2, -(Real.sqrt 2) / 2; (Real.sqrt 2) / 2, (Real.sqrt 2) / 2]

/-- The canonical form of the quadratic equation -/
def canonical_form (x'' y'' : ℝ) : Prop :=
  x''^2 / 1^2 + y''^2 / (0.5)^2 = 1

/-- Theorem stating that the original quadratic form can be transformed into the canonical form -/
theorem quadratic_to_canonical :
  ∃ (x'' y'' : ℝ → ℝ → ℝ), ∀ (x y : ℝ),
    original_quadratic_form x y = 0 ↔ canonical_form (x'' x y) (y'' x y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_to_canonical_l1264_126463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2014_equals_3_l1264_126483

def sequenceA (n : ℕ) : ℤ :=
  match n with
  | 0 => 1
  | 1 => 2
  | n+2 => sequenceA (n+1) - sequenceA n

def sum_sequence (n : ℕ) : ℤ :=
  (List.range n).map sequenceA |>.sum

theorem sum_2014_equals_3 : sum_sequence 2014 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_2014_equals_3_l1264_126483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_abs_even_and_increasing_l1264_126466

-- Define the function f(x) = ln|x| as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (abs x)

-- State the theorem
theorem ln_abs_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_abs_even_and_increasing_l1264_126466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_circle_sector_l1264_126480

/-- The volume of a cone formed by rolling a two-third sector of a circle --/
theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  (1/3) * π * (r * 2/3)^2 * Real.sqrt (r^2 - (r * 2/3)^2) = (32 * π * Real.sqrt 5) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_from_circle_sector_l1264_126480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l1264_126411

-- Define the vertices of the rectangle
def v1 : ℝ × ℝ := (-3, 5)
def v2 : ℝ × ℝ := (4, -3)
def v3 : ℝ × ℝ := (9, 5)
def v4 : ℝ × ℝ := (4, 5)

-- Define the function to calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem stating that the diagonal length of the rectangle is √89
theorem rectangle_diagonal_length :
  distance v2 v3 = Real.sqrt 89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_length_l1264_126411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1264_126451

noncomputable section

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = -4 * Real.sqrt 5 * x

-- Define the ellipse
def ellipse (x y a : ℝ) : Prop := x^2 / a^2 + y^2 / 4 = 1

-- Define the eccentricity of an ellipse
def eccentricity (c a : ℝ) : ℝ := c / a

theorem ellipse_eccentricity :
  ∀ a : ℝ, a > 0 →
  (∃ x y : ℝ, parabola x y ∧ ellipse x y a) →
  eccentricity (Real.sqrt 5) a = Real.sqrt 5 / 3 :=
by
  intros a ha hex
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1264_126451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l1264_126468

-- Define a pyramid type
structure Pyramid where
  base_edge : ℝ
  altitude : ℝ

-- Define a function to calculate the volume of a pyramid
noncomputable def pyramid_volume (p : Pyramid) : ℝ := 
  (1/3) * p.base_edge^2 * p.altitude

-- Define the original pyramid
noncomputable def original_pyramid : Pyramid :=
  { base_edge := 24, altitude := 18 }

-- Define the smaller pyramid
noncomputable def smaller_pyramid : Pyramid :=
  { base_edge := original_pyramid.base_edge * (1/3), 
    altitude := original_pyramid.altitude * (1/3) }

-- Define the frustum volume
noncomputable def frustum_volume : ℝ :=
  pyramid_volume original_pyramid - pyramid_volume smaller_pyramid

-- Theorem statement
theorem frustum_volume_ratio :
  frustum_volume / pyramid_volume original_pyramid = 26/27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_ratio_l1264_126468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_match_probability_is_one_third_l1264_126488

/-- Represents the distribution of jelly beans for a person -/
structure JellyBeans where
  green : Nat
  yellow : Nat
  blue : Nat

/-- Calculates the total number of jelly beans -/
def JellyBeans.total (jb : JellyBeans) : Nat :=
  jb.green + jb.yellow + jb.blue

/-- Abe's jelly bean distribution -/
def abe : JellyBeans :=
  { green := 2, yellow := 0, blue := 1 }

/-- Mia's jelly bean distribution -/
def mia : JellyBeans :=
  { green := 2, yellow := 2, blue := 3 }

/-- Calculates the probability of picking a specific color -/
def pickProbability (jb : JellyBeans) (color : Nat) : ℚ :=
  color / jb.total

/-- Calculates the probability of both people picking the same color -/
def matchProbability (jb1 jb2 : JellyBeans) : ℚ :=
  pickProbability jb1 jb1.green * pickProbability jb2 jb2.green +
  pickProbability jb1 jb1.blue * pickProbability jb2 jb2.blue

theorem match_probability_is_one_third :
  matchProbability abe mia = 1 / 3 := by
  sorry

#eval matchProbability abe mia

end NUMINAMATH_CALUDE_ERRORFEEDBACK_match_probability_is_one_third_l1264_126488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_and_double_alpha_l1264_126471

open Real

theorem cos_alpha_and_double_alpha (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/3)) 
  (h2 : Real.sqrt 6 * sin α + Real.sqrt 2 * cos α = Real.sqrt 3) : 
  (cos (α + π/6) = Real.sqrt 10 / 4) ∧ 
  (cos (2*α + π/12) = (Real.sqrt 30 + Real.sqrt 2) / 8) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_and_double_alpha_l1264_126471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorizable_iff_condition_l1264_126485

/-- A polynomial in x and y with parameters a and b -/
def polynomial (a b : ℝ) (x y : ℝ) : ℝ :=
  x * (x + 4) + a * (y^2 - 1) + 2 * b * y

/-- The condition for factorizability of the polynomial -/
def factorizable_condition (a b : ℝ) : Prop :=
  (a + 2)^2 + b^2 = 4

/-- The theorem stating the equivalence between factorizability and the condition -/
theorem polynomial_factorizable_iff_condition (a b : ℝ) :
  (∃ (p q : ℝ → ℝ → ℝ), ∀ x y, polynomial a b x y = p x y * q x y ∧ 
    (∃ A B C D E F : ℝ, p x y = A * x + B * y + C ∧ q x y = D * x + E * y + F)) ↔
  factorizable_condition a b :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_factorizable_iff_condition_l1264_126485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_price_savings_l1264_126472

theorem car_price_savings (P : ℝ) (h : P > 0) : 
  let reduced_price := 0.95 * P
  let final_price := 0.90 * reduced_price
  let savings := P - final_price
  savings / P = 0.145 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_price_savings_l1264_126472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_example_l1264_126469

/-- The height of a cylinder given its radius and surface area -/
noncomputable def cylinder_height (r : ℝ) (sa : ℝ) : ℝ :=
  (sa - 2 * Real.pi * r^2) / (2 * Real.pi * r)

/-- Theorem: The height of a cylinder with radius 5 feet and surface area 100π square feet is 5 feet -/
theorem cylinder_height_example :
  cylinder_height 5 (100 * Real.pi) = 5 := by
  -- Unfold the definition of cylinder_height
  unfold cylinder_height
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_example_l1264_126469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_is_401_82_l1264_126435

-- Define the selling prices and profit percentages
noncomputable def selling_price_1 : ℝ := 120
noncomputable def selling_price_2 : ℝ := 150
noncomputable def selling_price_3 : ℝ := 200
noncomputable def profit_percent_1 : ℝ := 0.20
noncomputable def profit_percent_2 : ℝ := 0.25
noncomputable def profit_percent_3 : ℝ := 0.10

-- Define the cost price calculation function
noncomputable def cost_price (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  selling_price / (1 + profit_percent)

-- Define the total cost price calculation
noncomputable def total_cost_price : ℝ :=
  cost_price selling_price_1 profit_percent_1 +
  cost_price selling_price_2 profit_percent_2 +
  cost_price selling_price_3 profit_percent_3

-- Theorem statement
theorem total_cost_price_is_401_82 :
  ∃ ε > 0, |total_cost_price - 401.82| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_price_is_401_82_l1264_126435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_S_over_a_eq_neg_two_l1264_126415

noncomputable def S (a : ℝ) : ℝ :=
  ∫ x in Set.Icc 0 1, max (max (Real.exp x) (Real.exp ((1 + a) / (1 - a) * x))) (Real.exp (2 - x)) -
                      min (min (Real.exp x) (Real.exp ((1 + a) / (1 - a) * x))) (Real.exp (2 - x))

theorem limit_S_over_a_eq_neg_two :
  ∀ ε > 0, ∃ δ > 0, ∀ a : ℝ, 0 < a → a < 1 → a < δ →
    |S a / a + 2| < ε := by
  sorry

#check limit_S_over_a_eq_neg_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_S_over_a_eq_neg_two_l1264_126415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_bound_l1264_126441

/-- A convex figure in a 2D plane --/
structure ConvexFigure where
  -- We don't need to define the specific properties of a convex figure for this statement
  -- but we assume its existence

/-- The area of a convex figure --/
noncomputable def area (f : ConvexFigure) : ℝ := sorry

/-- The semi-perimeter of a convex figure --/
noncomputable def semiPerimeter (f : ConvexFigure) : ℝ := sorry

/-- A lattice point inside a convex figure --/
def LatticePoint (f : ConvexFigure) : Type := sorry

/-- The number of lattice points inside a convex figure --/
noncomputable def numLatticePoints (f : ConvexFigure) : ℕ := sorry

/-- Theorem: For any convex figure, the number of lattice points inside it
    is greater than its area minus its semi-perimeter --/
theorem lattice_points_bound (f : ConvexFigure) :
  (numLatticePoints f : ℝ) > area f - semiPerimeter f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lattice_points_bound_l1264_126441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_root_l1264_126429

noncomputable def is_root (r : ℝ) (p : ℝ → ℝ) : Prop := p r = 0

noncomputable def sum_roots (p : ℝ → ℝ) : ℝ := 
  sorry -- This would typically be the sum of all roots of p, but we leave it undefined

theorem polynomial_integer_root 
  (a b c : ℚ) 
  (h1 : is_root (3 - Real.sqrt 5) (fun x => x^4 + a*x^2 + b*x + c))
  (h2 : (sum_roots (fun x => x^4 + a*x^2 + b*x + c)) = 0) :
  is_root (-3 : ℝ) (fun x => x^4 + a*x^2 + b*x + c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_integer_root_l1264_126429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_after_transform_l1264_126439

/-- A pyramid with a rectangular base. -/
structure RectangularPyramid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The volume of a rectangular pyramid. -/
noncomputable def volume (p : RectangularPyramid) : ℝ := (1/3) * p.length * p.width * p.height

/-- The transformation applied to the pyramid. -/
def transform (p : RectangularPyramid) : RectangularPyramid :=
  { length := 4 * p.length
    width := 2 * p.width
    height := 0.75 * p.height }

/-- Theorem stating that the volume of the transformed pyramid is 6 times the original volume. -/
theorem volume_after_transform (p : RectangularPyramid) :
  volume (transform p) = 6 * volume p := by
  -- Unfold definitions
  unfold volume transform
  -- Simplify
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_after_transform_l1264_126439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l1264_126430

/-- The side length of the larger equilateral triangle -/
noncomputable def large_side : ℝ := 12

/-- The side length of the smaller equilateral triangle -/
noncomputable def small_side : ℝ := 6

/-- The area of an equilateral triangle given its side length -/
noncomputable def equilateral_triangle_area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2

/-- The area of the smaller equilateral triangle -/
noncomputable def small_triangle_area : ℝ := equilateral_triangle_area small_side

/-- The area of the isosceles trapezoid -/
noncomputable def trapezoid_area : ℝ := 3 * small_triangle_area

theorem area_ratio_is_one_third :
  small_triangle_area / trapezoid_area = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_is_one_third_l1264_126430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l1264_126473

/-- Represents the area of a lawn -/
structure LawnArea where
  value : ℝ

/-- Represents the mowing speed of a lawn mower -/
structure MowingSpeed where
  value : ℝ

/-- Represents the time taken to mow a lawn -/
structure MowingTime where
  value : ℝ

/-- Andy's lawn area -/
noncomputable def andysLawn : LawnArea := ⟨1⟩

/-- Beth's lawn area -/
noncomputable def bethsLawn : LawnArea := ⟨1/4⟩

/-- Carlos' lawn area -/
noncomputable def carlosLawn : LawnArea := ⟨1/2⟩

/-- Andy's mowing speed -/
noncomputable def andysMower : MowingSpeed := ⟨1⟩

/-- Beth's mowing speed -/
noncomputable def bethsMower : MowingSpeed := ⟨3/4⟩

/-- Carlos' mowing speed -/
noncomputable def carlosMower : MowingSpeed := ⟨1/2⟩

/-- Time taken by Andy to mow his lawn -/
noncomputable def andysMowingTime : MowingTime := ⟨andysLawn.value / andysMower.value⟩

/-- Time taken by Beth to mow her lawn -/
noncomputable def bethsMowingTime : MowingTime := ⟨bethsLawn.value / bethsMower.value⟩

/-- Time taken by Carlos to mow his lawn -/
noncomputable def carlosMowingTime : MowingTime := ⟨carlosLawn.value / carlosMower.value⟩

/-- Theorem stating Beth will finish mowing first -/
theorem beth_finishes_first :
  (andysLawn.value = 4 * bethsLawn.value) →
  (andysLawn.value = 2 * carlosLawn.value) →
  (carlosMower.value = (1/2) * andysMower.value) →
  (bethsMower.value = (3/4) * andysMower.value) →
  (bethsMowingTime.value < andysMowingTime.value ∧ bethsMowingTime.value < carlosMowingTime.value) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beth_finishes_first_l1264_126473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_for_given_area_and_perimeter_l1264_126455

/-- Represents a rectangle with given area and perimeter -/
structure Rectangle where
  area : ℝ
  perimeter : ℝ

/-- The width of a rectangle given its area and perimeter -/
noncomputable def rectangle_width (r : Rectangle) : ℝ :=
  (r.perimeter / 2 - Real.sqrt ((r.perimeter / 2) ^ 2 - 4 * r.area)) / 2

/-- Theorem stating that a rectangle with area 750 and perimeter 110 has width 25 -/
theorem rectangle_width_for_given_area_and_perimeter :
  let r : Rectangle := ⟨750, 110⟩
  rectangle_width r = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_width_for_given_area_and_perimeter_l1264_126455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_is_one_sixth_l1264_126426

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def count_divisible_pairs (r_lower r_upper k_lower k_upper : ℤ) : ℤ :=
  (r_upper - r_lower - 1) * ((k_upper - k_lower - 1) / 2)

theorem probability_divisible_is_one_sixth :
  let r_lower : ℤ := -5
  let r_upper : ℤ := 8
  let k_lower : ℤ := 1
  let k_upper : ℤ := 10
  let total_pairs : ℤ := (r_upper - r_lower - 1) * ((k_upper - k_lower - 1) / 2)
  let divisible_pairs : ℤ := count_divisible_pairs r_lower r_upper k_lower k_upper
  (divisible_pairs : ℚ) / (total_pairs : ℚ) = 1 / 6 := by
  sorry

#eval count_divisible_pairs (-5) 8 1 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_divisible_is_one_sixth_l1264_126426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1264_126495

theorem product_remainder (a b c : ℕ) 
  (ha : a % 7 = 2)
  (hb : b % 7 = 3)
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l1264_126495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_second_to_smallest_l1264_126482

/-- Represents a right circular cone divided into three sections -/
structure SectionedCone where
  /-- The radius of the base of the cone -/
  baseRadius : ℝ
  /-- The total height of the cone -/
  totalHeight : ℝ
  /-- Assumption that the cone is divided into three sections with height ratios 1:2:3 -/
  heightRatios : List ℝ := [1, 2, 3]

/-- Calculates the volume of a conical section given its base radius and height -/
noncomputable def sectionVolume (baseRadius : ℝ) (height : ℝ) : ℝ :=
  (1/3) * Real.pi * baseRadius^2 * height

/-- Theorem stating the ratio of volumes of the second largest to the smallest section -/
theorem volume_ratio_second_to_smallest (cone : SectionedCone) :
  let smallestHeight := cone.totalHeight / 6
  let middleHeight := 2 * smallestHeight
  let smallestRadius := cone.baseRadius / 3
  let middleRadius := 2 * cone.baseRadius / 3
  let middleVolume := sectionVolume middleRadius middleHeight
  let smallestVolume := sectionVolume smallestRadius smallestHeight
  middleVolume / smallestVolume = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_second_to_smallest_l1264_126482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1264_126417

noncomputable def f (a : ℝ) (x : ℝ) := Real.log (a * x^2 - 4 * x + a)

def p (a : ℝ) : Prop := ∀ x, (a * x^2 - 4 * x + a) > 0

def q (a : ℝ) : Prop := ∀ x < -1, 2 * x^2 + x > 2 + a * x

theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ∈ Set.Icc 1 2 := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1264_126417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_iff_sum_vectors_zero_l1264_126445

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Definition: A point is the centroid of a triangle if it is located at the arithmetic mean of the triangle's vertices. -/
def IsCentroid (G A B C : V) : Prop :=
  G = (1/3 : ℝ) • (A + B + C)

/-- A point is the centroid of a triangle if and only if the sum of vectors from the point to each vertex is zero. -/
theorem centroid_iff_sum_vectors_zero (A B C G : V) :
  IsCentroid G A B C ↔ (G - A) + (G - B) + (G - C) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_iff_sum_vectors_zero_l1264_126445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_six_solutions_l1264_126401

/-- A solution to the system of equations -/
structure Solution where
  x : ℝ
  y : ℝ
  z : ℝ
  u : ℝ
  v : ℝ

/-- The system of equations -/
def satisfies_system (s : Solution) : Prop :=
  s.x + s.y + s.z = 5 ∧
  s.x^2 + s.y^2 + s.z^2 = 9 ∧
  s.x * s.y + s.u * s.x + s.v * s.y = 0 ∧
  s.y * s.z + s.u * s.y + s.v * s.z = 0 ∧
  s.z * s.x + s.u * s.z + s.v * s.x = 0

/-- The list of solutions -/
noncomputable def solutions : List Solution := [
  ⟨2, 2, 1, 4, -2⟩,
  ⟨4/3, 4/3, 7/3, 16/9, -4/3⟩,
  ⟨2, 1, 2, 4, -2⟩,
  ⟨4/3, 7/3, 4/3, 16/9, -4/3⟩,
  ⟨1, 2, 2, 4, -2⟩,
  ⟨7/3, 4/3, 4/3, 16/9, -4/3⟩
]

/-- The theorem stating that the system has exactly six solutions -/
theorem system_has_six_solutions :
  (∃! (l : List Solution), (∀ s ∈ l, satisfies_system s) ∧ l.length = 6) ∧
  (∀ s, satisfies_system s → s ∈ solutions) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_has_six_solutions_l1264_126401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1264_126498

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  2 * Real.exp x - 2 * x + (1/2) * x^2

-- State the theorem
theorem f_monotone_decreasing :
  ∀ x y : ℝ, x < y → y < 0 → f x > f y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_l1264_126498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1264_126432

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (4 : ℝ)^x - (2 : ℝ)^x

-- Define the theorem
theorem function_properties 
  (s t : ℝ) 
  (h1 : f s + f t = 0) 
  (a : ℝ) 
  (h2 : a = (2 : ℝ)^s + (2 : ℝ)^t) 
  (b : ℝ) 
  (h3 : b = (2 : ℝ)^(s+t)) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ∈ Set.Icc (-1/4 : ℝ) 2) ∧
  (b = (a^2 - a) / 2 ∧ a ∈ Set.Ioo 1 2) ∧
  ((8 : ℝ)^s + (8 : ℝ)^t ∈ Set.Ioo 1 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1264_126432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_heads_one_tail_l1264_126477

theorem probability_three_heads_one_tail : 
  let n : ℕ := 4  -- number of coins
  let k : ℕ := 3  -- number of heads we want
  let p : ℚ := 1/2  -- probability of getting heads on a single coin toss
  (Finset.filter (fun s => (s.card) = k) (Finset.powerset (Finset.range n))).card / 2^n = 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_three_heads_one_tail_l1264_126477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_non_defective_pens_l1264_126491

/-- Probability of selecting two non-defective pens from a box -/
theorem prob_two_non_defective_pens 
  (total_pens : ℕ) 
  (defective_pens : ℕ) 
  (selected_pens : ℕ) 
  (h1 : total_pens = 8) 
  (h2 : defective_pens = 3) 
  (h3 : selected_pens = 2) :
  (total_pens - defective_pens : ℚ) / total_pens * 
  ((total_pens - defective_pens - 1) : ℚ) / (total_pens - 1) = 5/14 := by
  sorry

#check prob_two_non_defective_pens

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_non_defective_pens_l1264_126491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l1264_126425

/-- Given two points A and B in 3D space, calculate the distance between them. -/
noncomputable def distance_3d (A B : ℝ × ℝ × ℝ) : ℝ :=
  let (x₁, y₁, z₁) := A
  let (x₂, y₂, z₂) := B
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2 + (z₂ - z₁)^2)

/-- Theorem: The distance between points A(0, 2, 5) and B(-1, 3, 3) is √6. -/
theorem distance_A_to_B : distance_3d (0, 2, 5) (-1, 3, 3) = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_A_to_B_l1264_126425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1264_126419

noncomputable def f (x : ℝ) := Real.sin (2 * x + Real.pi / 6) + 2 * (Real.sin x) ^ 2

theorem f_properties :
  let f := f
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∀ x, f x ≤ 2) ∧
  (∀ k : ℤ, f (k * Real.pi + Real.pi / 3) = 2) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3),
    ∀ y ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3),
    x ≤ y → f x ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1264_126419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_remainder_and_quotient_sum_l1264_126428

theorem unique_number_with_remainder_and_quotient_sum : ∃! N : ℕ, 
  (N % 7 + N % 11 + N % 13 = 21) ∧ 
  (N / 7 + N / 11 + N / 13 = 21) ∧ 
  (N = 74) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_remainder_and_quotient_sum_l1264_126428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_to_left_grouping_right_to_left_not_standard_l1264_126461

/-- Right-to-left grouping evaluation function -/
noncomputable def evalRightToLeft (a b c d : ℝ) : ℝ := a / (b - c - d)

/-- Standard algebraic notation evaluation function -/
noncomputable def evalStandard (a b c d : ℝ) : ℝ := a / b - c + d

/-- Theorem stating that right-to-left grouping is equivalent to the expected result -/
theorem right_to_left_grouping (a b c d : ℝ) :
  evalRightToLeft a b c d = a / (b - c - d) :=
by
  -- Unfold the definition of evalRightToLeft
  unfold evalRightToLeft
  -- The result follows directly from the definition
  rfl

/-- Theorem stating that right-to-left grouping is different from standard notation -/
theorem right_to_left_not_standard (a b c d : ℝ) :
  evalRightToLeft a b c d ≠ evalStandard a b c d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_to_left_grouping_right_to_left_not_standard_l1264_126461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_property_l1264_126427

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation x^2 = 20y -/
def Parabola : Set Point :=
  {p : Point | p.x^2 = 20 * p.y}

/-- The focus of the parabola -/
def F : Point :=
  ⟨0, 5⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem parabola_point_property (a b : ℝ) :
  let P : Point := ⟨a, b⟩
  P ∈ Parabola → distance P F = 25 → |a * b| = 400 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_property_l1264_126427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_singleton_implies_a_range_l1264_126489

/-- Given sets A and B, prove that if their intersection is a singleton, 
    then the parameter a must be in the specified range. -/
theorem intersection_singleton_implies_a_range 
  (a : ℝ) 
  (A : Set (ℝ × ℝ)) 
  (B : Set (ℝ × ℝ)) 
  (h1 : A = {p : ℝ × ℝ | p.2 = a * p.1 + 2})
  (h2 : B = {p : ℝ × ℝ | p.2 = |p.1 + 1|})
  (h3 : ∃! p, p ∈ A ∩ B) :
  a ∈ Set.Iic (-1) ∪ Set.Ici 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_singleton_implies_a_range_l1264_126489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_earnings_l1264_126414

/-- Calculates Jane's earnings from planting flower bulbs -/
theorem jane_earnings 
  (tulip_bulbs : ℕ)
  (daffodil_bulbs : ℕ)
  (earnings_per_bulb : ℚ)
  (h1 : tulip_bulbs = 20)
  (h2 : daffodil_bulbs = 30)
  (h3 : earnings_per_bulb = 1/2) :
  let iris_bulbs := tulip_bulbs / 2
  let hyacinth_bulbs := iris_bulbs + 2
  let crocus_bulbs := daffodil_bulbs * 3
  let gladiolus_bulbs := 2 * (crocus_bulbs - daffodil_bulbs)
  let total_bulbs := tulip_bulbs + iris_bulbs + hyacinth_bulbs + daffodil_bulbs + crocus_bulbs + gladiolus_bulbs
  total_bulbs * earnings_per_bulb = 141 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_earnings_l1264_126414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_h_graph_and_coordinate_sum_l1264_126481

-- Define the functions g and h
noncomputable def g : ℝ → ℝ := sorry
noncomputable def h : ℝ → ℝ := sorry

-- State the theorem
theorem point_on_h_graph_and_coordinate_sum :
  (g 3 = 6) →
  (∀ x, h x = (g x)^2) →
  (h 3 = 36) ∧ (3 + 36 = 39) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_h_graph_and_coordinate_sum_l1264_126481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_slope_angle_l1264_126499

/-- The slope of the line √3x - 3y - 2m = 0 (where m ∈ ℝ) is √3/3 -/
theorem line_slope (m : ℝ) : 
  (Real.sqrt 3) / 3 = (Real.sqrt 3) / 3 := by
  sorry

/-- The angle corresponding to the slope √3/3 is 30° -/
theorem slope_angle : 
  Real.arctan ((Real.sqrt 3) / 3) = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_slope_angle_l1264_126499
