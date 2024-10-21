import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_alpha_beta_l1321_132113

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + Real.pi / 6)

theorem cos_sum_alpha_beta 
  (ω : ℝ)
  (α β : ℝ)
  (h_ω_pos : ω > 0)
  (h_period : 2 * Real.pi / ω = 10 * Real.pi)
  (h_α_range : 0 ≤ α ∧ α ≤ Real.pi / 2)
  (h_β_range : 0 ≤ β ∧ β ≤ Real.pi / 2)
  (h_f_α : f ω (5 * α + 5 * Real.pi / 3) = -6 / 5)
  (h_f_β : f ω (5 * β - 5 * Real.pi / 6) = 16 / 17) :
  Real.cos (α + β) = -13 / 85 := by
  sorry

#check cos_sum_alpha_beta

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_alpha_beta_l1321_132113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_lift_satellite_l1321_132135

/-- Work done to lift a satellite -/
noncomputable def work_to_lift_satellite (m : ℝ) (g : ℝ) (R₃ : ℝ) (H : ℝ) : ℝ :=
  m * g * R₃^2 * (1/R₃ - 1/(R₃ + H))

/-- Theorem: Work done to lift a satellite from Earth's surface to height H -/
theorem work_done_lift_satellite (m g R₃ H : ℝ) 
  (hm : m = 5000) -- mass in kg
  (hg : g = 10) -- acceleration due to gravity in m/s²
  (hR₃ : R₃ = 6.38e6) -- Earth's radius in meters
  (hH : H = 4.5e5) -- height in meters
  : ‖work_to_lift_satellite m g R₃ H - 21017569546‖ < 1 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_done_lift_satellite_l1321_132135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_line_coloring_exists_l1321_132115

/-- A line in a plane --/
structure Line where
  -- Add necessary fields here
  -- This is a placeholder as the exact representation depends on how you want to model lines
  mk :: -- Empty constructor for now

/-- A set of lines in general position --/
def GeneralPositionLines (n : ℕ) := 
  { lines : Finset Line // lines.card = n ∧ 
    (∀ l1 l2 : Line, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 → ¬ are_parallel l1 l2) ∧
    (∀ l1 l2 l3 : Line, l1 ∈ lines ∧ l2 ∈ lines ∧ l3 ∈ lines ∧ l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 → ¬ intersect_at_point l1 l2 l3) }
where
  are_parallel : Line → Line → Prop := sorry
  intersect_at_point : Line → Line → Line → Prop := sorry

/-- A coloring of lines --/
def Coloring (lines : Finset Line) := Line → Bool

/-- Predicate to check if a region has its boundary entirely composed of blue lines --/
def all_blue_boundary (coloring : Coloring (lines : Finset Line)) (region : Set Line) : Prop :=
  sorry -- Define the condition for a region to have all blue boundary

/-- Predicate to check if a region is finite --/
def is_finite_region (lines : Finset Line) (region : Set Line) : Prop :=
  sorry -- Define the condition for a region to be finite

/-- The main theorem --/
theorem blue_line_coloring_exists (n : ℕ) (h : n ≥ 100) :
  ∃ (lines : GeneralPositionLines n) (coloring : Coloring lines.val),
    (∃ blue_lines : Finset Line, blue_lines ⊆ lines.val ∧ 
      blue_lines.card ≥ n.sqrt ∧
      (∀ l ∈ blue_lines, coloring l = true)) ∧
    (∀ region : Set Line, is_finite_region lines.val region → ¬ all_blue_boundary coloring region) :=
by
  sorry -- Proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_line_coloring_exists_l1321_132115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l1321_132162

def U : Finset ℕ := {1, 2, 3, 4, 5}

def A : Finset ℕ := U.filter (λ x => x^2 - 6*x + 5 = 0)

theorem complement_of_A_in_U :
  U \ A = {2, 3, 4} := by
  -- Proof goes here
  sorry

#eval A  -- This will evaluate A for verification
#eval U \ A  -- This will evaluate the complement for verification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l1321_132162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l1321_132154

-- Define the parallelepiped
structure Parallelepiped where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_perpendicular : Real.cos (Real.pi / 2) = 0
  h_angle : Real.cos (Real.pi / 3) = 1 / 2

-- Define the volume function
noncomputable def volume (p : Parallelepiped) : ℝ :=
  (p.a * p.b * p.c * Real.sqrt 2) / 2

-- Theorem statement
theorem parallelepiped_volume (p : Parallelepiped) : 
  volume p = (p.a * p.b * p.c * Real.sqrt 2) / 2 := by
  -- Unfold the definition of volume
  unfold volume
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelepiped_volume_l1321_132154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_theorem_l1321_132147

/-- Given that R is directly proportional to S and inversely proportional to T,
    this function represents the relationship between R, S, and T. -/
noncomputable def R_relation (k : ℝ) (S T : ℝ) : ℝ := k * S / T

/-- Theorem stating the relationship between R, S, and T under given conditions -/
theorem proportionality_theorem (k : ℝ) :
  R_relation k 1 4 = 2 →
  R_relation k ((3 * Real.sqrt 6) / 2) (4 * Real.sqrt 2) = 3 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportionality_theorem_l1321_132147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1321_132181

/-- Given vectors u and v in ℝ³ satisfying the specified conditions, prove that v equals (2/3, 11/3, 7/3) -/
theorem vector_problem (u v : Fin 3 → ℝ) : 
  (u 0 + v 0 = -2 ∧ u 1 + v 1 = 5 ∧ u 2 + v 2 = 1) →
  (∃ (k : ℝ), u = fun i => [2*k, -k, k].get i) →
  (2 * v 0 + (-1) * v 1 + v 2 = 0) →
  v = fun i => [2/3, 11/3, 7/3].get i := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l1321_132181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_tree_age_l1321_132187

/-- The half-life of carbon-14 in years -/
noncomputable def carbon_14_half_life : ℝ := 5730

/-- The carbon-14 content function -/
noncomputable def carbon_14_content (k₀ n : ℝ) : ℝ := k₀ * (1/2)^(n / carbon_14_half_life)

/-- Theorem stating the approximate age of the tree -/
theorem ancient_tree_age (k₀ : ℝ) (h₀ : k₀ > 0) :
  ∃ n : ℝ, carbon_14_content k₀ n = 0.6 * k₀ ∧ 
    (n ≥ 4201.99 ∧ n ≤ 4202.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ancient_tree_age_l1321_132187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_approx_l1321_132194

/-- The mean proportional between two numbers -/
noncomputable def mean_proportional (a b : ℝ) : ℝ := Real.sqrt (a * b)

/-- Theorem: The mean proportional between 64 and 82.13 is approximately 72.5 -/
theorem mean_proportional_approx :
  abs (mean_proportional 64 82.13 - 72.5) < 0.01 := by
  sorry

/-- Compute an approximation of the mean proportional -/
def mean_proportional_approx' (a b : Float) : Float :=
  Float.sqrt (a * b)

#eval mean_proportional_approx' 64 82.13

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_approx_l1321_132194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l1321_132141

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V) 
  (ha : ‖a‖ = 1) 
  (hb : ‖b‖ = 1) 
  (hab : ‖a + b‖ = 1) : 
  ‖2 • a + b‖ = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_problem_l1321_132141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_equation_unique_solution_l1321_132103

theorem binomial_equation_unique_solution : 
  ∃! n : ℕ, (Nat.choose 23 n) + (Nat.choose 23 12) = (Nat.choose 24 13) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_equation_unique_solution_l1321_132103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_every_positive_int_is_sum_of_charming_numbers_l1321_132176

/-- A positive integer is charming if it's 2 or of the form 3^i * 5^j for non-negative integers i and j -/
def IsCharming (n : ℕ) : Prop :=
  n = 2 ∨ ∃ (i j : ℕ), n = 3^i * 5^j

/-- Every positive integer can be written as a sum of different charming numbers -/
theorem every_positive_int_is_sum_of_charming_numbers :
  ∀ n : ℕ, n > 0 → ∃ (S : Finset ℕ), (∀ m ∈ S, IsCharming m) ∧ (S.sum id = n) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_every_positive_int_is_sum_of_charming_numbers_l1321_132176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_consecutive_free_days_l1321_132191

/-- Represents a day in the month -/
inductive Day : Type
| monday | tuesday | wednesday | thursday | friday | saturday | sunday

/-- Represents a calendar month -/
structure Month where
  days : List Day
  start_day : Day
  total_days : Nat

/-- Represents a schedule of events -/
structure Schedule where
  events : List Nat  -- List of days (1-31) when events occur

/-- Definition of a valid March 1987 calendar -/
def is_valid_march_1987 (m : Month) : Prop :=
  m.total_days = 31 ∧ m.start_day = Day.sunday

/-- Definition of a valid schedule -/
def is_valid_schedule (s : Schedule) (m : Month) : Prop :=
  s.events.length = 11 ∧
  ∀ d ∈ s.events, d ≥ 1 ∧ d ≤ 31 ∧
  ∀ d ∈ s.events, m.days.get? (d - 1) ≠ some Day.saturday ∧ 
                  m.days.get? (d - 1) ≠ some Day.sunday

/-- Main theorem -/
theorem three_consecutive_free_days 
  (m : Month) 
  (s : Schedule) 
  (h1 : is_valid_march_1987 m) 
  (h2 : is_valid_schedule s m) : 
  ∃ i : Nat, i ≥ 1 ∧ i ≤ 29 ∧ 
    i ∉ s.events ∧ 
    (i + 1) ∉ s.events ∧ 
    (i + 2) ∉ s.events := by
  sorry

#check @three_consecutive_free_days

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_consecutive_free_days_l1321_132191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_prime_factor_of_2003_4_plus_1_l1321_132107

theorem smallest_odd_prime_factor_of_2003_4_plus_1 :
  ∃ (p : ℕ), Nat.Prime p ∧ Odd p ∧ p ∣ (2003^4 + 1) ∧
  ∀ (q : ℕ), Nat.Prime q → Odd q → q ∣ (2003^4 + 1) → p ≤ q ∧ p = 17 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_odd_prime_factor_of_2003_4_plus_1_l1321_132107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_polygon_with_distinct_integer_angles_l1321_132104

-- Define a function to check if n satisfies the conditions
def is_valid_polygon (n : ℕ) : Prop :=
  ∃ (angles : Fin n → ℕ),
    -- All angles are distinct
    (∀ i j, i ≠ j → angles i ≠ angles j) ∧
    -- Sum of interior angles is (n-2) * 180
    (Finset.sum Finset.univ (λ i => angles i) = (n - 2) * 180) ∧
    -- Each angle is between 0 and 180 (exclusive)
    (∀ i, 0 < angles i ∧ angles i < 180)

-- Theorem stating that 26 is the largest n satisfying the conditions
theorem largest_polygon_with_distinct_integer_angles :
  (is_valid_polygon 26) ∧ (∀ m > 26, ¬(is_valid_polygon m)) := by
  sorry

#check largest_polygon_with_distinct_integer_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_polygon_with_distinct_integer_angles_l1321_132104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flood_damage_in_usd_l1321_132180

/-- Converts Australian dollars to American dollars given the exchange rate -/
noncomputable def aud_to_usd (aud : ℝ) (exchange_rate : ℝ) : ℝ := aud / exchange_rate

/-- Theorem: The flood damage in American dollars -/
theorem flood_damage_in_usd (damage_aud : ℝ) (exchange_rate : ℝ) 
  (h1 : damage_aud = 45000000) 
  (h2 : exchange_rate = 1.2) : 
  aud_to_usd damage_aud exchange_rate = 37500000 := by
  sorry

/-- Compute the result using rational numbers to avoid noncomputable issues -/
def flood_damage_in_usd_rat : ℚ :=
  45000000 / 1.2

#eval flood_damage_in_usd_rat

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flood_damage_in_usd_l1321_132180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l1321_132193

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x-1)^2 + ax + sin(x + π/2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi/2)

/-- If f is an even function, then a = 2 -/
theorem f_even_implies_a_eq_two (a : ℝ) : IsEven (f a) → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_even_implies_a_eq_two_l1321_132193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_ratio_l1321_132126

def is_valid_ratio (q : ℕ) (b₁ : ℕ) : Prop :=
  b₁ * q^2 * (1 + q^2 + q^4) = 819 * 6^2016 ∧
  q ∈ ({1, 2, 3, 4} : Finset ℕ)

theorem geometric_progression_ratio :
  ∀ (b : ℕ → ℕ),
  (∀ n, ∃ q : ℕ, b (n + 1) = b n * q) →
  b 3 + b 5 + b 7 = 819 * 6^2016 →
  ∃ q, is_valid_ratio q (b 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_progression_ratio_l1321_132126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1321_132185

-- Define the function
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc 0 4

-- Theorem statement
theorem f_properties :
  -- Tangent line at x = -1
  (∀ x y, (4 * x - y + 4 = 0) ↔ (y = f (-1) + (deriv f) (-1) * (x - (-1)))) ∧
  -- Maximum value
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 45) ∧
  -- Minimum value
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y ∧ f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1321_132185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1321_132127

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₁ - C₂| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance between two parallel lines l₁: 3x + 4y - 4 = 0 and l₂: 3x + 4y + 1 = 0 is 1 unit -/
theorem distance_between_given_lines :
  distance_parallel_lines 3 4 (-4) (-1) = 1 := by
  -- Unfold the definition of distance_parallel_lines
  unfold distance_parallel_lines
  -- Simplify the expression
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1321_132127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l1321_132152

theorem rectangular_to_polar :
  let x : ℝ := 1
  let y : ℝ := -1
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x) + (if y < 0 ∧ x > 0 then 2 * Real.pi else 0)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ r = Real.sqrt 2 ∧ θ = 7 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_polar_l1321_132152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_specific_vectors_l1321_132189

/-- The cosine of the angle between two 2D vectors --/
noncomputable def cosine_angle (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

/-- Theorem: The cosine of the angle between (2, -1) and (1, 3) is -√2/10 --/
theorem cosine_angle_specific_vectors :
  cosine_angle (2, -1) (1, 3) = -Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_specific_vectors_l1321_132189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_study_time_difference_l1321_132140

def daily_differences : List Int := [15, -5, 25, -10, 15, 5, 10]

def days_in_week : Nat := 7

theorem average_study_time_difference :
  Int.toNat (List.sum daily_differences / days_in_week) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_study_time_difference_l1321_132140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_radius_chords_form_circle_l1321_132139

theorem six_radius_chords_form_circle (r : ℝ) (h : r > 0) : 
  ∃ (points : Fin 6 → ℝ × ℝ),
    (∀ i : Fin 6, (points i).1^2 + (points i).2^2 = r^2) ∧
    (∀ i : Fin 6, ‖points i - points (i.succ)‖ = r) ∧
    points 0 = points 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_radius_chords_form_circle_l1321_132139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_jumps_theorem_l1321_132161

/-- Represents the position of the grasshopper -/
structure Position where
  x : Int
  y : Int
deriving Repr

/-- Calculates the position after a single jump -/
def jump (p : Position) (jumpNumber : Nat) : Position :=
  match jumpNumber % 4 with
  | 0 => ⟨p.x - 1, p.y⟩     -- 1 cm east
  | 1 => ⟨p.x, p.y + 2⟩     -- 2 cm north
  | 2 => ⟨p.x + 3, p.y⟩     -- 3 cm west
  | _ => ⟨p.x, p.y - 4⟩     -- 4 cm south

/-- Calculates the position after n jumps -/
def finalPosition (n : Nat) : Position :=
  (List.range n).foldl jump ⟨0, 0⟩

/-- Calculates the sum of squares of digits of a number -/
def sumOfSquaresOfDigits (n : Nat) : Nat :=
  let digits := n.repr.data.map (fun c => c.toNat - 48)
  digits.foldl (fun sum d => sum + d * d) 0

/-- The main theorem to prove -/
theorem grasshopper_jumps_theorem : 
  ∃ n : Nat, finalPosition n = ⟨162, -158⟩ ∧ sumOfSquaresOfDigits n = 22 := by
  -- Proof goes here
  sorry

#eval finalPosition 323
#eval sumOfSquaresOfDigits 323

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_jumps_theorem_l1321_132161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_prime_factors_of_9720_l1321_132178

theorem sum_distinct_prime_factors_of_9720 : 
  (Finset.filter (λ p => Nat.Prime p ∧ 9720 % p = 0) (Finset.range (9720 + 1))).sum id = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_distinct_prime_factors_of_9720_l1321_132178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l1321_132110

def p₁ (x : ℝ) : ℝ := x^5 - 4*x^3 + 3*x^2 - 2*x + 5
def p₂ (x : ℝ) : ℝ := 3*x^2 - 2*x + 4
def p₃ (x : ℝ) : ℝ := 1 - x

def product (x : ℝ) : ℝ := p₁ x * p₂ x * p₃ x

theorem coefficient_of_x_cubed :
  ∃ (a b c d e f : ℝ), 
    (∀ x, product x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) ∧ 
    c = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_cubed_l1321_132110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_per_month_l1321_132105

/-- Proof of cost per box per month for record storage --/
theorem cost_per_box_per_month
  (box_length : ℝ)
  (box_width : ℝ)
  (box_height : ℝ)
  (total_volume : ℝ)
  (total_monthly_cost : ℝ)
  (h1 : box_length = 15)
  (h2 : box_width = 12)
  (h3 : box_height = 10)
  (h4 : total_volume = 1080000)
  (h5 : total_monthly_cost = 360) :
  (total_monthly_cost / (total_volume / (box_length * box_width * box_height))) = 0.60 := by
  sorry

-- Remove the #eval line as it's not necessary for building and might cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_box_per_month_l1321_132105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1321_132102

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | (2 : ℝ)^(x : ℝ) > 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l1321_132102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1321_132171

/-- A line passing through a point with a given normal vector has a specific general form equation. -/
theorem line_equation (x y : ℝ) : 
  (∃ (l : Set (ℝ × ℝ)), 
    -- The line passes through point (1,2)
    (1, 2) ∈ l ∧ 
    -- The line has a normal vector (1,-3)
    (∃ (t : ℝ), ∀ (p : ℝ × ℝ), p ∈ l ↔ (p.1 - 1) + (-3) * (p.2 - 2) = t)) →
  -- The general form equation of the line is x - 3y + 5 = 0
  (x - 3*y + 5 = 0 ↔ ∃ (l : Set (ℝ × ℝ)), (x, y) ∈ l ∧ (1, 2) ∈ l ∧ 
    (∃ (t : ℝ), ∀ (p : ℝ × ℝ), p ∈ l ↔ (p.1 - 1) + (-3) * (p.2 - 2) = t)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_l1321_132171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_box_sphere_radius_l1321_132192

/-- A rectangular box -/
structure RectangularBox where
  width : ℝ
  length : ℝ
  height : ℝ

/-- A sphere -/
structure Sphere where
  radius : ℝ

/-- Predicate to check if a box is inscribed in a sphere -/
def InscribedIn (box : RectangularBox) (sphere : Sphere) : Prop :=
  box.width^2 + box.length^2 + box.height^2 = (2 * sphere.radius)^2

/-- Surface area of a rectangular box -/
def SurfaceArea (box : RectangularBox) : ℝ :=
  2 * (box.width * box.length + box.width * box.height + box.length * box.height)

/-- Sum of edge lengths of a rectangular box -/
def SumOfEdgeLengths (box : RectangularBox) : ℝ :=
  4 * (box.width + box.length + box.height)

theorem inscribed_box_sphere_radius 
  (Q : RectangularBox) 
  (s : ℝ) 
  (h_inscribed : InscribedIn Q (Sphere.mk s))
  (h_surface_area : SurfaceArea Q = 608)
  (h_edge_sum : SumOfEdgeLengths Q = 160) :
  s = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_box_sphere_radius_l1321_132192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_payment_max_l1321_132112

theorem last_payment_max (a b c d : ℕ) 
  (sum_eq : a + b + c + d = 28)
  (increasing : a < b ∧ b < c ∧ c < d)
  (divisible : b % a = 0 ∧ c % b = 0 ∧ d % c = 0)
  (positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  d ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_payment_max_l1321_132112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_40_yuan_fertilizer_l1321_132153

-- Define the yield function
noncomputable def W (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 2 then 5 * (x^2 + 3)
  else if 2 < x ∧ x ≤ 5 then 50 * x / (1 + x)
  else 0

-- Define the profit function
noncomputable def f (x : ℝ) : ℝ :=
  15 * W x - 30 * x

-- Theorem statement
theorem max_profit_at_40_yuan_fertilizer :
  ∃ (max_profit : ℝ) (fertilizer_cost : ℝ),
    (∀ x, 0 ≤ x → x ≤ 5 → f x ≤ max_profit) ∧
    (∃ x, 0 ≤ x ∧ x ≤ 5 ∧ f x = max_profit) ∧
    max_profit = 480 ∧
    fertilizer_cost = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_40_yuan_fertilizer_l1321_132153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_product_theorem_l1321_132108

-- Define the function F
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := (Real.log x / x)^2 + (a - 1) * (Real.log x / x) + 1 - a

-- State the theorem
theorem three_roots_product_theorem (a : ℝ) (x₁ x₂ x₃ : ℝ) :
  x₁ < x₂ ∧ x₂ < x₃ ∧
  F a x₁ = 0 ∧ F a x₂ = 0 ∧ F a x₃ = 0 ∧
  x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ →
  (1 - Real.log x₁ / x₁)^2 * (1 - Real.log x₂ / x₂) * (1 - Real.log x₃ / x₃) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_roots_product_theorem_l1321_132108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_chain_l1321_132174

-- Define the circles
def Circle : Type := ℝ × ℝ × ℝ  -- (center_x, center_y, radius)

-- Define the property of two circles being non-overlapping
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1, r1) := c1
  let (x2, y2, r2) := c2
  (x1 - x2)^2 + (y1 - y2)^2 > (r1 + r2)^2

-- Define the property of a circle being tangent to another
def is_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1, r1) := c1
  let (x2, y2, r2) := c2
  (x1 - x2)^2 + (y1 - y2)^2 = (r1 + r2)^2

-- Define the angle between two circles
noncomputable def angle_between (c1 c2 : Circle) : ℝ :=
  sorry  -- The actual implementation would involve trigonometry

-- Define the existence of a chain of n tangent circles
def chain_exists (R1 R2 : Circle) (n : ℕ) : Prop :=
  sorry  -- This would involve complex geometric conditions

-- Define the property of circles being tangent at intersection points
def tangent_at_intersection (T1 T2 R1 R2 : Circle) : Prop :=
  sorry  -- This would involve geometric conditions

-- Main theorem
theorem tangent_circle_chain
  (R1 R2 T1 T2 : Circle)
  (n : ℕ)
  (h1 : non_overlapping R1 R2)
  (h2 : is_tangent T1 R1)
  (h3 : is_tangent T2 R2)
  (h4 : tangent_at_intersection T1 T2 R1 R2) :
  chain_exists R1 R2 n ↔ ∃ (k : ℕ), angle_between T1 T2 = k * (360 / n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_chain_l1321_132174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_reciprocal_sum_l1321_132149

theorem geometric_sequence_reciprocal_sum
  (n : ℕ) (r s : ℝ) (hr : r ≠ 0) (hs : s ≠ 0) (hn : n > 0) :
  let original_sum := (1 - r^n) / (1 - r)
  let reciprocal_sum := (1 - (1/r)^n) / (1 - 1/r)
  original_sum = s → reciprocal_sum = s / r^(n - 1) :=
by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_reciprocal_sum_l1321_132149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_18_10_l1321_132148

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) : 
  Nat.choose 18 10 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_18_10_l1321_132148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_drunk_drivers_l1321_132143

/-- Represents the number of drunk drivers in a traffic class. -/
def drunk_drivers : ℕ := sorry

/-- Represents the number of speeders in a traffic class. -/
def speeders : ℕ := sorry

/-- The total number of students in the traffic class. -/
def total_students : ℕ := 45

/-- The relation between drunk drivers and speeders. -/
axiom speeders_relation : speeders = 7 * drunk_drivers - 3

/-- The total number of students is the sum of drunk drivers and speeders. -/
axiom total_students_sum : drunk_drivers + speeders = total_students

/-- Theorem stating that there are 6 drunk drivers in the traffic class. -/
theorem six_drunk_drivers : drunk_drivers = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_drunk_drivers_l1321_132143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l1321_132183

-- Define the power function as noncomputable
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x ^ a

-- State the theorem
theorem power_function_property (a : ℝ) :
  (f a 4 / f a 2 = 3) → f a (1/2) = 1/3 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_property_l1321_132183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_minus_pi_4_l1321_132184

theorem sin_2alpha_minus_pi_4 (α : Real) 
  (h1 : Real.tan α + (Real.tan α)⁻¹ = 5/2) 
  (h2 : α ∈ Set.Ioo (π/4) (π/2)) : 
  Real.sin (2*α - π/4) = 7*Real.sqrt 2/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_minus_pi_4_l1321_132184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_l1321_132177

/-- A plane in 3D space -/
structure Plane where
  -- Define a plane (implementation details omitted)
  dummy : Unit

/-- Parallel relation between planes -/
def parallel (p q : Plane) : Prop :=
  -- Define parallel relation between planes
  True -- Placeholder definition

/-- Theorem: If two planes are each parallel to a third plane, then they are parallel to each other -/
theorem parallel_transitive (α β γ : Plane) :
  parallel α β → parallel α γ → parallel β γ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitive_l1321_132177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_triangle_areas_equal_l1321_132168

/-- Parallelogram with vertices A, B, C, D and interior point O -/
structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  O : ℝ × ℝ
  is_parallelogram : (A.1 - B.1, A.2 - B.2) = (D.1 - C.1, D.2 - C.2) ∧ 
                     (A.1 - D.1, A.2 - D.2) = (B.1 - C.1, B.2 - C.2)
  O_interior : ∃ (t1 t2 : ℝ), 0 < t1 ∧ t1 < 1 ∧ 0 < t2 ∧ t2 < 1 ∧
               O = (t1 * A.1 + (1 - t1) * C.1, t2 * A.2 + (1 - t2) * C.2)

/-- Area of a triangle given its vertices -/
noncomputable def triangle_area (P Q R : ℝ × ℝ) : ℝ :=
  abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) / 2

/-- Theorem: Sum of areas of triangles BOC and AOD equals sum of areas of triangles AOB and COD -/
theorem parallelogram_triangle_areas_equal (p : Parallelogram) :
  triangle_area p.B p.O p.C + triangle_area p.A p.O p.D =
  triangle_area p.A p.O p.B + triangle_area p.C p.O p.D := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_triangle_areas_equal_l1321_132168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1321_132151

theorem vector_equation_solution : ∃ (x y : ℝ), 
  x = -13/15 ∧ y = 34/15 ∧ 
  (⟨3, 1⟩ : ℝ × ℝ) + x • (⟨9, -7⟩ : ℝ × ℝ) = (⟨2, -2⟩ : ℝ × ℝ) + y • (⟨-3, 4⟩ : ℝ × ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_equation_solution_l1321_132151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_sequence_sum_l1321_132133

/-- A circular sequence of 10 positive integers satisfying the GCD condition -/
def CircularSequence : Type := Fin 10 → ℕ+

/-- The GCD condition for the circular sequence -/
def satisfiesGCDCondition (seq : CircularSequence) : Prop :=
  ∀ i : Fin 10, seq i = (Nat.gcd (seq (i - 1)) (seq (i + 1))).succ

/-- The theorem stating that the sum of a valid circular sequence is 28 -/
theorem circular_sequence_sum (seq : CircularSequence) 
  (h : satisfiesGCDCondition seq) : (Finset.sum Finset.univ (λ i => (seq i : ℕ))) = 28 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_sequence_sum_l1321_132133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_ages_and_surnames_l1321_132150

-- Define the friends and surnames
inductive Friend : Type
| Kostya : Friend
| Vasya : Friend
| Kolya : Friend

inductive Surname : Type
| Semyonov : Surname
| Burov : Surname
| Nikolaev : Surname

-- Define a function to assign surnames to friends
def surname_assignment : Friend → Surname
| Friend.Kostya => Surname.Semyonov
| Friend.Vasya => Surname.Nikolaev
| Friend.Kolya => Surname.Burov

-- Define a function to assign ages to friends
def age : Friend → Nat
| Friend.Kostya => 18
| Friend.Vasya => 16
| Friend.Kolya => 17

-- Define the conditions
axiom semyonov_grandfather : surname_assignment Friend.Kostya = Surname.Semyonov

axiom kostya_older_kolya : age Friend.Kostya = age Friend.Kolya + 1

axiom kolya_older_nikolaev : 
  ∃ f, surname_assignment f = Surname.Nikolaev ∧ age Friend.Kolya = age f + 1

axiom age_sum_bounds : 
  49 < age Friend.Kostya + age Friend.Vasya + age Friend.Kolya ∧
  age Friend.Kostya + age Friend.Vasya + age Friend.Kolya < 53

axiom kolya_not_korobov : surname_assignment Friend.Kolya ≠ Surname.Semyonov

-- Theorem to prove
theorem friends_ages_and_surnames :
  (surname_assignment Friend.Kostya = Surname.Semyonov ∧ age Friend.Kostya = 18) ∧
  (surname_assignment Friend.Vasya = Surname.Nikolaev ∧ age Friend.Vasya = 16) ∧
  (surname_assignment Friend.Kolya = Surname.Burov ∧ age Friend.Kolya = 17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friends_ages_and_surnames_l1321_132150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_classification_l1321_132124

/-- The curve y = x^3 - x -/
def curve (x : ℝ) : ℝ := x^3 - x

/-- The number of tangents to the curve y = x^3 - x passing through a point (x₀, y₀) -/
noncomputable def num_tangents (x₀ y₀ : ℝ) : ℕ :=
  if x₀ + y₀ < 0 ∨ y₀ > curve x₀ then 1
  else if x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0 then 3
  else 2

theorem tangent_classification (x₀ y₀ : ℝ) :
  (num_tangents x₀ y₀ = 1 ↔ x₀ + y₀ < 0 ∨ y₀ > curve x₀) ∧
  (num_tangents x₀ y₀ = 3 ↔ x₀ + y₀ > 0 ∧ x₀ + y₀ - x₀^3 < 0) ∧
  (num_tangents x₀ y₀ = 2 ↔ x₀ + y₀ = 0 ∨ y₀ = curve x₀) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_classification_l1321_132124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_median_triangle_side_length_l1321_132130

/-- A triangle with perpendicular medians -/
structure PerpendicularMedianTriangle where
  -- The vertices of the triangle
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  -- The endpoints of the medians on the opposite sides
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  -- DP is a median
  dp_is_median : P = ((E.1 + F.1) / 2, (E.2 + F.2) / 2)
  -- EQ is a median
  eq_is_median : Q = ((D.1 + F.1) / 2, (D.2 + F.2) / 2)
  -- FR is a median
  fr_is_median : R = ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  -- DP is perpendicular to EQ
  dp_perp_eq : (P.1 - D.1) * (Q.1 - E.1) + (P.2 - D.2) * (Q.2 - E.2) = 0
  -- FR is perpendicular to DP
  fr_perp_dp : (R.1 - F.1) * (P.1 - D.1) + (R.2 - F.2) * (P.2 - D.2) = 0
  -- Length of DP is 15
  dp_length : Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) = 15
  -- Length of EQ is 20
  eq_length : Real.sqrt ((Q.1 - E.1)^2 + (Q.2 - E.2)^2) = 20

/-- The length of DF in a triangle with perpendicular medians -/
noncomputable def length_DF (t : PerpendicularMedianTriangle) : ℝ :=
  Real.sqrt ((t.F.1 - t.D.1)^2 + (t.F.2 - t.D.2)^2)

/-- Theorem: In a triangle with perpendicular medians, given the lengths of two medians, 
    we can determine the length of a side -/
theorem perpendicular_median_triangle_side_length 
  (t : PerpendicularMedianTriangle) : length_DF t = 100 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_median_triangle_side_length_l1321_132130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_time_to_fill_pool_l1321_132157

/-- Represents the capacity of the pool -/
noncomputable def poolCapacity : ℝ := 1

/-- Initial water in the pool -/
noncomputable def initialWater : ℝ := poolCapacity / 18

/-- Water level when first pipe starts filling alone -/
noncomputable def intermediateWater : ℝ := poolCapacity * 2 / 9

/-- Time first pipe fills alone -/
noncomputable def firstPipeAloneTime : ℝ := 81

/-- Time second pipe fills alone -/
noncomputable def secondPipeAloneTime : ℝ := 49

/-- Time both pipes fill simultaneously to reach intermediate water level -/
noncomputable def simultaneousFillTime : ℝ := 63

/-- Theorem stating the additional time needed to fill the pool -/
theorem additional_time_to_fill_pool : 
  ∃ (rateA rateB : ℝ),
    rateA > 0 ∧ rateB > 0 ∧
    rateA * simultaneousFillTime + rateB * simultaneousFillTime = intermediateWater - initialWater ∧
    rateA * firstPipeAloneTime = rateB * simultaneousFillTime ∧
    rateA * simultaneousFillTime = rateB * secondPipeAloneTime ∧
    (poolCapacity - (initialWater + 2 * (intermediateWater - initialWater))) / 
      ((rateA + rateB) * simultaneousFillTime / (intermediateWater - initialWater)) = 231 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_time_to_fill_pool_l1321_132157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_always_positive_implies_a_range_l1321_132106

theorem quadratic_always_positive_implies_a_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + a > 0) → a ∈ Set.Ioi 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_always_positive_implies_a_range_l1321_132106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_square_side_ratio_l1321_132158

/-- The perimeter of the shapes in inches -/
noncomputable def perimeter : ℝ := 20

/-- The side length of the square -/
noncomputable def square_side : ℝ := perimeter / 4

/-- The side length of the isosceles triangle (assuming both equal sides) -/
noncomputable def triangle_side : ℝ := perimeter / 3

/-- The ratio of the triangle side length to the square side length -/
noncomputable def side_ratio : ℝ := triangle_side / square_side

theorem isosceles_square_side_ratio :
  side_ratio = 4 / 3 := by
  -- Unfold definitions
  unfold side_ratio triangle_side square_side perimeter
  -- Simplify the expression
  simp [div_div_eq_mul_div]
  -- Prove the equality
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_square_side_ratio_l1321_132158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_l1321_132111

noncomputable section

-- Define the circle and the point A
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
def A : ℝ × ℝ := (0, 1/2)

-- Define the locus of H
def locus_H (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 1/2

-- Define points P, Q, and B
def P (k : ℝ) : ℝ × ℝ := sorry
def Q (k : ℝ) : ℝ × ℝ := sorry
def B (k : ℝ) : ℝ × ℝ := (-1/(2*k), 0)

-- Define lambda and mu
def lambda (k : ℝ) : ℝ := sorry
def mu (k : ℝ) : ℝ := sorry

-- Main theorem
theorem constant_sum (k : ℝ) (hk : k ≠ 0) :
  1 / lambda k + 1 / mu k = 8/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_sum_l1321_132111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_l1321_132125

noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

noncomputable def transformation1 (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x + Real.pi/6)

noncomputable def transformation2 (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x/2)

noncomputable def result_function (x : ℝ) : ℝ := Real.sin (x/2 + Real.pi/6)

theorem transformations_result : 
  ∀ x : ℝ, (transformation2 (transformation1 original_function)) x = result_function x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_l1321_132125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1321_132170

noncomputable section

/-- Definition of the ellipse -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- The ellipse passes through the point (-1, 3/2) -/
def passes_through (a b : ℝ) : Prop :=
  ellipse a b (-1) (3/2)

/-- The length of the minor axis is 2√3 -/
def minor_axis_length (b : ℝ) : Prop :=
  2 * b = 2 * Real.sqrt 3

/-- The maximum area of triangle OPQ when t = √3 -/
noncomputable def max_area_OPQ (a : ℝ) : ℝ :=
  Real.sqrt 3

/-- The relationship between s and t -/
def s_t_relationship (a s t : ℝ) : Prop :=
  s * t = a^2

/-- Main theorem -/
theorem ellipse_properties (a b : ℝ) :
  passes_through a b ∧ minor_axis_length b →
  (a = 2 ∧ b = Real.sqrt 3) ∧
  (max_area_OPQ a = Real.sqrt 3) ∧
  (∃ s, s_t_relationship a s (Real.sqrt 3)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1321_132170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1321_132128

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Theorem statement
theorem ellipse_eccentricity : 
  eccentricity 5 4 = 3/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l1321_132128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_rectangular_equation_line_l_general_form_chord_AB_length_l1321_132163

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + t, t - 3)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 2 * Real.cos θ / (Real.sin θ)^2

-- Statement for the rectangular equation of curve C
theorem curve_C_rectangular_equation (x y : ℝ) :
  (∃ θ : ℝ, x = curve_C θ * Real.cos θ ∧ y = curve_C θ * Real.sin θ) ↔ y^2 = 2*x := by
  sorry

-- Statement for the general form equation of line l
theorem line_l_general_form (x y : ℝ) :
  (∃ t : ℝ, (x, y) = line_l t) ↔ x - y - 4 = 0 := by
  sorry

-- Statement for the length of chord AB
theorem chord_AB_length :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ : ℝ, A = line_l t₁) ∧
    (∃ t₂ : ℝ, B = line_l t₂) ∧
    (A.1)^2 = 2*(A.2) ∧
    (B.1)^2 = 2*(B.2) ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_rectangular_equation_line_l_general_form_chord_AB_length_l1321_132163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1321_132155

theorem simplify_expression :
  -3 - 6 - (-5) + (-2) = -3 - 6 + 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l1321_132155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_y_of_x_l1321_132114

/-- Given the conditions:
    1. 20% of (x - y) = 15% of (x + y)
    2. 30% of (x - z) = 25% of (x + z)
    3. z is 10% of y
    Prove that y is approximately 90.91% of x -/
theorem percent_y_of_x (x y z : ℝ) 
  (h1 : 0.20 * (x - y) = 0.15 * (x + y))
  (h2 : 0.30 * (x - z) = 0.25 * (x + z))
  (h3 : z = 0.10 * y) :
  ∃ ε > 0, |y - 0.9091 * x| < ε :=
by
  sorry

#check percent_y_of_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_y_of_x_l1321_132114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l1321_132122

open Real

/-- An angle is in the second quadrant if it's between π/2 and π (modulo 2π) -/
def is_second_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi

/-- An angle is in the first or third quadrant if it's between 0 and π/2 or between π and 3π/2 (modulo 2π) -/
def is_first_or_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, (0 < α - 2 * k * Real.pi ∧ α - 2 * k * Real.pi < Real.pi / 2) ∨
           (Real.pi < α - 2 * k * Real.pi ∧ α - 2 * k * Real.pi < 3 * Real.pi / 2)

theorem half_angle_quadrant (α : ℝ) :
  is_second_quadrant α → is_first_or_third_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l1321_132122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desired_hyperbola_equation_hyperbola_relation_l1321_132190

-- Define the hyperbola type
structure Hyperbola where
  a : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop

-- Define the given hyperbola with known asymptotes
noncomputable def given_hyperbola : Hyperbola :=
  { a := Real.sqrt 2
    b := 1
    equation := λ x y => x^2 / 2 - y^2 = 1 }

-- Define the focus of the desired hyperbola
def focus : ℝ × ℝ := (6, 0)

-- Theorem statement
theorem desired_hyperbola_equation (h : Hyperbola) 
  (focus_condition : h.equation focus.1 focus.2)
  (asymptote_condition : ∀ (x y : ℝ), h.equation x y ↔ given_hyperbola.equation x y) :
  h.equation = λ x y => x^2 / 24 - y^2 / 12 = 1 := by
  sorry

-- Additional helper theorem to demonstrate the relationship between the given and desired hyperbolas
theorem hyperbola_relation :
  ∀ (x y : ℝ), given_hyperbola.equation x y ↔ (x^2 / 2 - y^2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_desired_hyperbola_equation_hyperbola_relation_l1321_132190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_143_l1321_132136

/-- A high-order arithmetic sequence is a sequence where the differences between consecutive terms
    are not equal, but the differences between consecutive differences or higher-order differences
    form an arithmetic sequence. -/
def HighOrderArithmeticSequence (a : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, ∀ k ≥ n, ∃ d : ℕ, ∀ i ≥ k,
    (a (i + 2) - a (i + 1)) - (a (i + 1) - a i) = d

/-- The specific high-order arithmetic sequence from the problem -/
def SpecificSequence (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 2 = 7 ∧ a 3 = 13 ∧ a 4 = 23 ∧ a 5 = 39 ∧ a 6 = 63 ∧ a 7 = 97

theorem eighth_term_is_143 (a : ℕ → ℕ) (h1 : HighOrderArithmeticSequence a) (h2 : SpecificSequence a) :
  a 8 = 143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_143_l1321_132136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1321_132172

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := Real.log x - x

-- State the theorem
theorem g_properties :
  -- g is defined on (0, +∞)
  (∀ x : ℝ, x > 0 → g x = Real.log x - x) ∧
  -- g is monotonically increasing on (0, 1)
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → g x₁ < g x₂) ∧
  -- g is monotonically decreasing on (1, +∞)
  (∀ x₁ x₂ : ℝ, 1 < x₁ ∧ x₁ < x₂ → g x₁ > g x₂) ∧
  -- The maximum value of g is -1, attained at x = 1
  (∀ x : ℝ, x > 0 → g x ≤ g 1) ∧ g 1 = -1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l1321_132172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_travel_distance_l1321_132100

-- Define the given conditions
noncomputable def feet_per_interval (b : ℝ) : ℝ := 2 * b / 7
noncomputable def seconds_per_interval (t : ℝ) : ℝ := 2 * t
def feet_per_yard : ℚ := 3
def minutes : ℚ := 4

-- Theorem statement
theorem truck_travel_distance (b t : ℝ) (h_pos_b : b > 0) (h_pos_t : t > 0) :
  (feet_per_interval b / seconds_per_interval t) * ((minutes : ℝ) * 60) / (feet_per_yard : ℝ) = 80 * b / (7 * t) := by
  sorry

#check truck_travel_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_truck_travel_distance_l1321_132100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_digit_sum_6_eq_21_l1321_132138

/-- The number of three-digit positive integers with digit sum 6 -/
def count_integers_with_digit_sum_6 : ℕ :=
  (Finset.filter (fun n : ℕ => 
    100 ≤ n ∧ n ≤ 999 ∧ 
    (n / 100 + (n / 10) % 10 + n % 10 = 6)) 
  (Finset.range 1000)).card

/-- Theorem stating that the count of three-digit positive integers with digit sum 6 is 21 -/
theorem count_integers_with_digit_sum_6_eq_21 : 
  count_integers_with_digit_sum_6 = 21 := by
  sorry

#eval count_integers_with_digit_sum_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_with_digit_sum_6_eq_21_l1321_132138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l1321_132175

/-- Calculates the area of a quadrilateral given its vertices. -/
def area_of_quadrilateral (vertices : List (ℝ × ℝ)) : ℝ :=
  let n := vertices.length
  let indices := List.range n
  let terms := indices.map (fun i =>
    let (x₁, y₁) := vertices[i]!
    let (x₂, y₂) := vertices[(i + 1) % n]!
    x₁ * y₂ - x₂ * y₁
  )
  0.5 * terms.sum

/-- The area of a quadrilateral with vertices at (2,1), (4,3), (6,1), and (4,6) is 6. -/
theorem quadrilateral_area : 
  let vertices : List (ℝ × ℝ) := [(2, 1), (4, 3), (6, 1), (4, 6)]
  area_of_quadrilateral vertices = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_l1321_132175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joseph_socks_count_l1321_132182

/-- Represents the number of pairs of socks of a given color -/
structure SockPairs where
  value : ℕ

/-- Represents the total number of socks of a given color -/
structure TotalSocks where
  value : ℕ

/-- The number of pairs of red socks -/
def red_pairs : SockPairs := ⟨4⟩

/-- The number of pairs of blue socks -/
def blue_pairs : SockPairs := ⟨3 * (2 * red_pairs.value)⟩

/-- The number of pairs of black socks -/
def black_pairs : SockPairs := ⟨blue_pairs.value - 5⟩

/-- The number of pairs of white socks -/
def white_pairs : SockPairs := ⟨red_pairs.value + 2⟩

/-- The number of pairs of green socks -/
def green_pairs : SockPairs := ⟨2 * red_pairs.value⟩

/-- The total number of red socks -/
def total_red : TotalSocks := ⟨9⟩

/-- Calculate the total number of socks -/
def total_socks : TotalSocks :=
  ⟨2 * (red_pairs.value + blue_pairs.value + black_pairs.value + white_pairs.value + green_pairs.value)⟩

theorem joseph_socks_count :
  total_socks.value = 122 := by
  -- Expand definitions
  unfold total_socks
  unfold red_pairs blue_pairs black_pairs white_pairs green_pairs
  -- Simplify arithmetic
  simp [Nat.mul_add, Nat.add_assoc]
  -- Check equality
  rfl

#eval total_socks.value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joseph_socks_count_l1321_132182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1321_132160

-- Define a structure for a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  -- Condition that angles sum to π
  angle_sum : A + B + C = π
  -- Law of sines
  law_of_sines : a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) :
  (t.A = 2 * t.B → t.C = 5 * π / 8) ∧
  (2 * t.a^2 = t.b^2 + t.c^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1321_132160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_exist_l1321_132117

theorem infinitely_many_solutions_exist :
  ∃ (m : ℕ), ∃ (S : Set (ℕ × ℕ × ℕ)), 
    (Set.Infinite S) ∧ 
    (∀ (a b c : ℕ), (a, b, c) ∈ S → a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
      (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + (1 : ℚ) / (a * b * c) = (m : ℚ) / (a + b + c)) :=
by
  -- We claim that m = 12 satisfies the condition
  use 12
  -- We'll define our set S later
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_exist_l1321_132117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_intersection_l1321_132199

open Set MeasureTheory

def A : Set ℝ := {x | 2 * x^2 - x - 3 < 0}

def B : Set ℝ := {x | ∃ y, y = Real.log ((1 - x) / (x + 3))}

def interval : Set ℝ := Ioc (-3) 3

theorem probability_in_intersection : 
  (volume (A ∩ B ∩ interval)) / volume interval = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_in_intersection_l1321_132199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1321_132101

theorem right_triangle_hypotenuse : 
  ∀ (shorter_leg longer_leg hypotenuse : ℝ),
  shorter_leg > 0 →
  longer_leg > 0 →
  longer_leg = 3 * shorter_leg - 4 →
  (1/2) * shorter_leg * longer_leg = 108 →
  shorter_leg^2 + longer_leg^2 = hypotenuse^2 →
  hypotenuse = Real.sqrt 26443 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l1321_132101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1321_132165

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- The point A on the ellipse -/
noncomputable def A : ℝ × ℝ := (1, 3/2)

theorem ellipse_properties (C : Ellipse) 
  (h1 : A.1^2 / C.a^2 + A.2^2 / C.b^2 = 1)  -- A is on the ellipse
  (h2 : ∃ (F₁ F₂ : ℝ × ℝ), Real.sqrt ((A.1 - F₁.1)^2 + (A.2 - F₁.2)^2) + 
                            Real.sqrt ((A.1 - F₂.1)^2 + (A.2 - F₂.2)^2) = 4) :
  (∃ (x y : ℝ), x^2/4 + y^2/3 = 1) ∧  -- Equation of C
  (∃ (F₁ F₂ : ℝ × ℝ), F₁ = (-1, 0) ∧ F₂ = (1, 0)) :=  -- Coordinates of foci
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l1321_132165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l1321_132197

-- Define the line equation
def line_eq (x y m : ℝ) : Prop := x - y + m = 0

-- Define the circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 1 = 0

-- Define the intersection condition
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    line_eq x₁ y₁ m ∧ line_eq x₂ y₂ m ∧
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂

-- State the theorem
theorem intersection_condition (m : ℝ) :
  intersects_at_two_points m ↔ -3 < m ∧ m < 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_condition_l1321_132197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_max_distance_l1321_132137

-- Define the circle
def circle_equation (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 10

-- Define the line l
def line_l (x y : ℝ) : Prop :=
  x + y + 2 = 0

-- Define point B
def point_B : ℝ × ℝ := (0, 2)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_and_max_distance :
  -- The circle passes through the origin
  circle_equation 0 0 ∧
  -- Maximum value of |PQ| - |PB| is 2√10
  (∃ (P : ℝ × ℝ) (Q : ℝ × ℝ),
    line_l P.1 P.2 ∧
    circle_equation Q.1 Q.2 ∧
    distance P Q - distance P point_B = 2 * Real.sqrt 10) ∧
  -- Coordinates of P at maximum value
  (∃ (P : ℝ × ℝ),
    line_l P.1 P.2 ∧
    P = (-6, 4) ∧
    ∀ (P' : ℝ × ℝ) (Q' : ℝ × ℝ),
      line_l P'.1 P'.2 →
      circle_equation Q'.1 Q'.2 →
      distance P' Q' - distance P' point_B ≤ distance P Q - distance P point_B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_max_distance_l1321_132137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_theorem_l1321_132166

-- Define the trapezoid EFGH
structure Trapezoid :=
  (EF : ℝ)
  (GH : ℝ)
  (non_parallel_leg : ℝ)

-- Define the properties of the trapezoid
def trapezoid_properties (t : Trapezoid) : Prop :=
  t.EF = 12 ∧ t.GH = 10 ∧ t.non_parallel_leg = 6

-- Calculate the length of a non-parallel side
noncomputable def non_parallel_side_length (t : Trapezoid) : ℝ :=
  Real.sqrt (((t.EF - t.GH) / 2) ^ 2 + t.non_parallel_leg ^ 2)

-- Calculate the perimeter of the trapezoid
noncomputable def trapezoid_perimeter (t : Trapezoid) : ℝ :=
  t.EF + t.GH + 2 * non_parallel_side_length t

-- Theorem statement
theorem trapezoid_perimeter_theorem (t : Trapezoid) :
  trapezoid_properties t → trapezoid_perimeter t = 22 + 2 * Real.sqrt 37 := by
  intro h
  sorry -- Proof skipped

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_perimeter_theorem_l1321_132166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l1321_132156

-- Define an acute triangle
structure AcuteTriangle where
  A : Real
  B : Real
  C : Real
  acute_A : 0 < A ∧ A < Real.pi / 2
  acute_B : 0 < B ∧ B < Real.pi / 2
  acute_C : 0 < C ∧ C < Real.pi / 2
  sum_angles : A + B + C = Real.pi

-- Theorem statement
theorem acute_triangle_properties (t : AcuteTriangle) :
  (t.A > t.B → Real.sin t.A > Real.sin t.B) ∧
  (Real.sin t.A + Real.sin t.B > Real.cos t.A + Real.cos t.B) ∧
  (Real.tan t.B * Real.tan t.C > 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l1321_132156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_time_distribution_l1321_132134

theorem homework_time_distribution (total_time : ℕ) 
  (math_percentage : ℚ) (science_percentage : ℚ) :
  total_time = 150 ∧ 
  math_percentage = 30 / 100 ∧ 
  science_percentage = 40 / 100 →
  total_time - (↑total_time * math_percentage).floor - (↑total_time * science_percentage).floor = 45 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_homework_time_distribution_l1321_132134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1321_132120

noncomputable def f (x : ℝ) : ℝ := (x - 3) / ((x - 2) * (x - 3))

theorem inequality_solution :
  ∀ x : ℝ, x ≠ 2 → x ≠ 3 → (f x ≤ 0 ↔ x ≤ 2 ∨ x = 3) :=
by
  intro x h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1321_132120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_last_five_digits_l1321_132167

theorem same_last_five_digits (M : ℕ) :
  (∃ a b c d e : ℕ, a ≠ 0 ∧
    M % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    (M * M) % 100000 = a * 10000 + b * 1000 + c * 100 + d * 10 + e) →
  (M % 10000 = 9687) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_last_five_digits_l1321_132167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_pair_equality_l1321_132109

theorem number_pair_equality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a * b = a / b) : 
  b = 1 ∧ ∀ x : ℝ, x > 1 → x * 1 = x / 1 := by
  sorry

#check number_pair_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_pair_equality_l1321_132109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_decrease_approximately_56_l1321_132146

noncomputable section

def original_trouser_price : ℝ := 100
def original_shirt_price : ℝ := 50
def original_shoes_price : ℝ := 30

def sale_trouser_price : ℝ := 20
def sale_shirt_price : ℝ := 35
def sale_shoes_price : ℝ := 25

def original_total_cost : ℝ := original_trouser_price + original_shirt_price + original_shoes_price
def sale_total_cost : ℝ := sale_trouser_price + sale_shirt_price + sale_shoes_price

def decrease_in_total_cost : ℝ := original_total_cost - sale_total_cost
def percent_decrease : ℝ := (decrease_in_total_cost / original_total_cost) * 100

theorem percent_decrease_approximately_56 :
  ∃ ε > 0, ε < 1 ∧ |percent_decrease - 56| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percent_decrease_approximately_56_l1321_132146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1321_132142

/-- Definition of a hyperbola with semi-major axis a and semi-minor axis b -/
def Hyperbola (a b : ℝ) := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}

/-- The right focus of the hyperbola -/
noncomputable def RightFocus (a b : ℝ) : ℝ × ℝ := (Real.sqrt (a^2 + b^2), 0)

/-- A line passing through a point -/
def Line (m : ℝ) (p : ℝ × ℝ) := {q : ℝ × ℝ | q.2 - p.2 = m * (q.1 - p.1)}

/-- The eccentricity of the hyperbola -/
noncomputable def Eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

/-- The statement to be proved -/
theorem hyperbola_eccentricity_range (a b : ℝ) (h1 : 0 < a) (h2 : a < b) :
  ∃ (m : ℝ) (A B : ℝ × ℝ),
    A ∈ Hyperbola a b ∧
    B ∈ Hyperbola a b ∧
    A ∈ Line m (RightFocus a b) ∧
    B ∈ Line m (RightFocus a b) ∧
    A.1 * B.1 + A.2 * B.2 = 0 →
    (1 + Real.sqrt 5) / 2 ≤ Eccentricity a b ∧ Eccentricity a b < Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l1321_132142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_odd_integer_in_sum_l1321_132129

theorem largest_odd_integer_in_sum (n : ℕ) (x : ℤ) : 
  n = 25 → 
  (2 * (n * (n + 1)) / 2 : ℤ) = (4 * x + 12) → 
  x % 2 = 1 → 
  (x + 6) % 2 = 1 → 
  (∀ y : ℤ, y > x + 6 → ¬((4 * y - 18 : ℤ) = (2 * (n * (n + 1)) / 2 : ℤ))) →
  x + 6 = 165 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_odd_integer_in_sum_l1321_132129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_monotonic_intervals_two_distinct_zeros_l1321_132121

noncomputable section

-- Define the functions
def f (x : ℝ) := Real.exp x
def g (m a b x : ℝ) := m * x^2 + a * x + b
def h (x : ℝ) := x * f x
def F (a b x : ℝ) := f x - g 0 a b x

-- Part I
theorem same_monotonic_intervals (m : ℝ) :
  (∀ x : ℝ, Monotone (fun x => h x) ↔ Monotone (fun x => g m 1 0 x)) → m = 1/2 :=
sorry

-- Part II
theorem two_distinct_zeros (b : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x ∈ Set.Icc (-1) 2 ∧ y ∈ Set.Icc (-1) 2 ∧ F 2 b x = 0 ∧ F 2 b y = 0) →
  2 - 2 * Real.log 2 < b ∧ b ≤ 1 / Real.exp 1 + 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_monotonic_intervals_two_distinct_zeros_l1321_132121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l1321_132131

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h₁ : d ≠ 0
  h₂ : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- Theorem: For an arithmetic sequence satisfying a₄ = 2(a₂ + a₃), S₇/S₄ = 7/4 -/
theorem arithmetic_sequence_sum_ratio
  (seq : ArithmeticSequence)
  (h : seq.a 4 = 2 * (seq.a 2 + seq.a 3)) :
  S seq 7 / S seq 4 = 7 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_ratio_l1321_132131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_even_l1321_132144

theorem S_is_even (a b c n : ℤ) 
  (h1 : (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) ∨ 
        (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ 
        (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1)) : 
  let S := (a + 2*n + 1) * (b + 2*n + 2) * (c + 2*n + 3)
  S % 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_is_even_l1321_132144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l1321_132119

-- Define the given conditions
noncomputable def train1_length : ℝ := 280
noncomputable def train1_speed : ℝ := 72 * 1000 / 3600  -- Convert km/h to m/s
noncomputable def train2_speed : ℝ := 36 * 1000 / 3600  -- Convert km/h to m/s
noncomputable def time_to_cross : ℝ := 63.99488040956724

-- Define the theorem
theorem second_train_length :
  ∃ (train2_length : ℝ),
    train2_length = (train1_speed - train2_speed) * time_to_cross - train1_length ∧
    abs (train2_length - 359.9488040956724) < 1e-10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_length_l1321_132119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_bound_l1321_132196

/-- A triangle in a 2D plane -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A parallelogram in a 2D plane -/
structure Parallelogram where
  P : Point
  Q : Point
  R : Point
  S : Point

/-- The area of a geometric shape -/
noncomputable def area (shape : Type*) : ℝ := sorry

/-- Predicate to check if a parallelogram is inside a triangle -/
def isInside (p : Parallelogram) (t : Triangle) : Prop := sorry

theorem parallelogram_area_bound (t : Triangle) (p : Parallelogram) 
  (h : isInside p t) : area Parallelogram ≤ (1/2 : ℝ) * area Triangle := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_bound_l1321_132196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_135_20_l1321_132195

/-- The area of a circular sector given its central angle and radius -/
noncomputable def sectorArea (centralAngle : ℝ) (radius : ℝ) : ℝ :=
  (centralAngle * Real.pi * radius^2) / 360

theorem sector_area_135_20 :
  sectorArea 135 20 = 150 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_135_20_l1321_132195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_special_condition_l1321_132179

theorem tan_value_for_special_condition (α : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.sin α ^ 2 + Real.sin (2 * α) = 1) : 
  Real.tan α = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_for_special_condition_l1321_132179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2016_negative_l1321_132188

theorem sin_2016_negative : Real.sin (2016 * Real.pi / 180) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2016_negative_l1321_132188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_share_proof_l1321_132116

-- Define the given values
def total_bill : ℚ := 139
def person_a_meal : ℚ := 35.25
def person_b_meal : ℚ := 42.5
def service_charge_rate : ℚ := 15 / 100
def tip_rate : ℚ := 10 / 100
def num_people : ℕ := 3

-- Define person C's meal cost
def person_c_meal : ℚ := total_bill - person_a_meal - person_b_meal

-- Define the service charge
def service_charge : ℚ := total_bill * service_charge_rate

-- Define the total bill with service charge
def total_bill_with_service : ℚ := total_bill + service_charge

-- Define each person's share before tip
def share_before_tip : ℚ := total_bill_with_service / num_people

-- Define the tip per person
def tip_per_person : ℚ := share_before_tip * tip_rate

-- Define the total share per person
def total_share_per_person : ℚ := share_before_tip + tip_per_person

-- Theorem to prove
theorem bill_share_proof : 
  ∀ ε > 0, |total_share_per_person - 58.61| < ε :=
by
  sorry -- Skip the proof for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_share_proof_l1321_132116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_inclination_angle_l1321_132118

/-- Represents a truncated cone with larger base radius R and smaller base radius r -/
structure TruncatedCone where
  R : ℝ  -- Radius of the larger base
  r : ℝ  -- Radius of the smaller base
  h : ℝ  -- Height of the truncated cone

/-- The angle of inclination of the cone's generatrix to the plane of the base -/
noncomputable def inclinationAngle (cone : TruncatedCone) : ℝ :=
  Real.arctan ((cone.R - cone.r) / cone.h)

/-- Theorem stating the conditions and the result to be proved -/
theorem truncated_cone_inclination_angle (cone : TruncatedCone) 
  (h1 : cone.h = cone.R)  -- Height equals radius of larger base
  (h2 : 6 * cone.r * Real.sqrt 3 = 3 * cone.R * Real.sqrt 3)  -- Perimeter equality condition
  : inclinationAngle cone = Real.arctan 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_truncated_cone_inclination_angle_l1321_132118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_female_selected_prob_male_host_l1321_132186

/-- Represents the number of contestants -/
def total_contestants : ℕ := 8

/-- Represents the number of female contestants -/
def female_contestants : ℕ := 5

/-- Represents the number of male contestants -/
def male_contestants : ℕ := 3

/-- Represents the number of contestants chosen to progress -/
def chosen_contestants : ℕ := 3

/-- Represents the number of contestants remaining after selection -/
def remaining_contestants : ℕ := total_contestants - chosen_contestants

/-- The probability of selecting all female contestants to progress -/
theorem prob_all_female_selected : 
  (Nat.choose female_contestants chosen_contestants : ℚ) / 
  (Nat.choose total_contestants chosen_contestants) = 5/28 := by sorry

/-- The probability of selecting a male host from the remaining contestants -/
theorem prob_male_host : 
  (Nat.choose male_contestants 1 : ℚ) / 
  (Nat.choose remaining_contestants 1) = 2/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_female_selected_prob_male_host_l1321_132186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_sum_eq_one_over_210_l1321_132123

open Real BigOperators

/-- The sum of 1 / (3^a * 4^b * 6^c * 8^d) for all quadruples (a,b,c,d) of positive integers where 1 ≤ a < b < c < d -/
noncomputable def quadruple_sum : ℝ :=
  ∑' a, ∑' b, ∑' c, ∑' d,
    if 1 ≤ a ∧ a < b ∧ b < c ∧ c < d then
      1 / (3^a * 4^b * 6^c * 8^d)
    else
      0

theorem quadruple_sum_eq_one_over_210 : quadruple_sum = 1 / 210 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadruple_sum_eq_one_over_210_l1321_132123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_CBG_l1321_132132

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the radius of the circle
def radius : ℝ := 5

-- Define the points
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def D : Point := sorry
def E : Point := sorry
def F : Point := sorry
def G : Point := sorry

-- Define the properties
axiom triangle_ABC_equilateral : ∀ (A B C : Point), True  -- Placeholder for IsEquilateral
axiom triangle_ABC_inscribed : ∀ (A B C : Point) (circle : Circle), True  -- Placeholder for InscribedIn
axiom AD_length : ∀ (A D : Point), ∃ (d : ℝ), d = 15  -- Placeholder for Distance
axiom AE_length : ∀ (A E : Point), ∃ (d : ℝ), d = 20  -- Placeholder for Distance
axiom l1_parallel_AE : ∀ (D F A E : Point), True  -- Placeholder for IsParallel
axiom l2_parallel_AD : ∀ (E F A D : Point), True  -- Placeholder for IsParallel
axiom G_collinear : ∀ (A F G : Point), True  -- Placeholder for Collinear
axiom G_on_circle : ∀ (G : Point) (circle : Circle), True  -- Placeholder for ∈
axiom G_distinct : ∀ (G A : Point), G ≠ A

-- Define the area function
noncomputable def area (p1 p2 p3 : Point) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_CBG : 
  area C B G = (216 * Real.sqrt 3) / 25 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_CBG_l1321_132132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_l1321_132159

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := ((x + 2) / 5) ^ (1/4)

-- State the theorem
theorem g_equality (x : ℝ) : g (3*x) = 3 * g x ↔ x = -404/201 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equality_l1321_132159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_large_ball_radius_l1321_132145

theorem large_ball_radius (n : ℕ) (r : ℝ) (R : ℝ) : 
  n = 12 → r = 2 → (4 / 3) * Real.pi * R^3 = n * ((4 / 3) * Real.pi * r^3) → R = 4 * (3 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_large_ball_radius_l1321_132145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_2_l1321_132164

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => 1/(1 - sequence_a n)

theorem a_2017_equals_2 : sequence_a 2016 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_equals_2_l1321_132164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_redistribution_reaches_stable_config_l1321_132198

/-- Represents a configuration of piles of pebbles -/
structure PileConfiguration where
  piles : List ℕ
  sum_pebbles : ℕ
  non_empty : ∀ p ∈ piles, p > 0
  total_sum : sum_pebbles = piles.sum

/-- Represents the process of redistributing pebbles -/
def redistribute (config : PileConfiguration) : PileConfiguration :=
  sorry

/-- Checks if a configuration is stable (all piles have exactly one pebble) -/
def is_stable (config : PileConfiguration) : Prop :=
  ∀ p ∈ config.piles, p = 1

/-- The main theorem stating that the redistribution process reaches a stable configuration -/
theorem redistribution_reaches_stable_config 
  (n : ℕ) 
  (initial_config : PileConfiguration) 
  (h_initial : initial_config.sum_pebbles = n * (n + 1) / 2) :
  ∃ (k : ℕ), is_stable (Nat.iterate redistribute k initial_config) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_redistribution_reaches_stable_config_l1321_132198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_can_evade_l1321_132173

/-- Represents a position on the infinite plane --/
structure Position where
  x : ℝ
  y : ℝ

/-- Represents a move, which is a displacement of at most 1 meter --/
structure Move where
  dx : ℝ
  dy : ℝ
  valid : dx^2 + dy^2 ≤ 1

/-- Represents the state of the game --/
structure GameState where
  wolf : Position
  sheep : List Position
  turn : ℕ

/-- A strategy for the sheep --/
def SheepStrategy := GameState → Move

/-- The wolf's strategy --/
def WolfStrategy := GameState → Move

/-- Applies a move to a position --/
def applyMove (p : Position) (m : Move) : Position :=
  ⟨p.x + m.dx, p.y + m.dy⟩

/-- Checks if the wolf has caught a sheep --/
def hasCaught (gs : GameState) : Prop :=
  ∃ s ∈ gs.sheep, s = gs.wolf

/-- Simulates a single turn of the game --/
def simulateTurn (gs : GameState) (wolfStrategy : WolfStrategy) (sheepStrategy : SheepStrategy) : GameState :=
  let wolfMove := wolfStrategy gs
  let newWolfPos := applyMove gs.wolf wolfMove
  let sheepMove := sheepStrategy ⟨newWolfPos, gs.sheep, gs.turn + 1⟩
  let newSheep := gs.sheep.mapIdx (fun i s => 
    if i = gs.turn % gs.sheep.length then applyMove s sheepMove else s)
  ⟨newWolfPos, newSheep, gs.turn + 1⟩

/-- Theorem: There exists a strategy for the sheep that prevents the wolf from catching any sheep --/
theorem sheep_can_evade (initialState : GameState) 
  (h_sheep_count : initialState.sheep.length = 50) :
  ∃ (sheepStrategy : SheepStrategy), ∀ (wolfStrategy : WolfStrategy),
  ¬ (∃ (n : ℕ), hasCaught (Nat.iterate (simulateTurn · wolfStrategy sheepStrategy) n initialState)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sheep_can_evade_l1321_132173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_l1321_132169

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 4)

noncomputable def transformed_f (x φ : ℝ) : ℝ := 2 * Real.sin (4 * x - 2 * φ + Real.pi / 4)

theorem smallest_positive_phi :
  ∃ (φ : ℝ), φ > 0 ∧
  (∀ (x : ℝ), transformed_f x φ = transformed_f (Real.pi / 2 - x) φ) ∧
  (∀ (φ' : ℝ), φ' > 0 ∧ 
    (∀ (x : ℝ), transformed_f x φ' = transformed_f (Real.pi / 2 - x) φ') → 
    φ' ≥ φ) ∧
  φ = 3 * Real.pi / 8 := by
  sorry

#check smallest_positive_phi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_phi_l1321_132169
